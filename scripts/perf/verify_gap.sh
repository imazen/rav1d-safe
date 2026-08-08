#!/usr/bin/env bash
# Independent gap-to-dav1d sweep for the verify/compose campaign.
#
# ONE instrument on both sides: wall clock of the whole process, each stream run
# at two frame counts, `total = alpha + beta*frames` fitted so process startup
# (binary load, AVIF/IVF parse, decoder construction) drops out and `beta` is
# ms/frame. That is the same shape as scripts/perf/p2_dav1d_wall.sh; the deltas
# are (1) every thread count is a cell rather than just 1 and 8, (2) dav1d is
# measured BOTH at --framedelay 1 (the tile-threading-only model we implement)
# and at its default (frame threading on, its real shipping configuration), and
# (3) the idle guard is STRICT with no tolerated-load escape hatch, because this
# campaign only runs on an idle box.
#
# NO `nice` ON A TIMED RUN — Darwin maps a niced process onto E-cores and the
# wall clock distorts by ~40x. NO -C target-cpu=native.
#
# Usage: verify_gap.sh <out.tsv> [rounds]
# Env:   BIN (staged arm binaries), AVIF, IVF, ARMS, CELLS, NLO, NHI
#
# Output columns:
#   round  arm  vec  threads  nlo  ms_lo  nhi  ms_hi  foreign_max
set -u
# `.` as the decimal separator, so the EPOCHREALTIME split below is locale-proof.
export LC_ALL=C
OUT=${1:?out.tsv}; ROUNDS=${2:-7}
BIN=${BIN:-$HOME/tmp/rav1d-iv/bin}
AVIF=${AVIF:-$HOME/tmp/rav1d-perf/vec}
IVF=${IVF:-$HOME/tmp/recon-yard/vec}
NLO=${NLO:-2}; NHI=${NHI:-20}
IFS=' ' read -r -a ARMS <<< "${ARMS:-base kern all dav1d_fd1 dav1d_def}"
IFS=' ' read -r -a CELLS <<< "${CELLS:-v4k_8tile:1 v4k_8tile:2 v4k_8tile:4 v4k_8tile:8 v4k_8tile_10b:1 v4k_8tile_10b:2 v4k_8tile_10b:4 v4k_8tile_10b:8}"

# Foreign = anything that is not the agent harness and not an arm under test.
# `ps -o comm` prints the full path on macOS, so the exclusion is built from
# $BIN and follows wherever the arms were staged. dav1d is excluded by name
# because it IS one of the arms. macOS keeps a decaying %cpu for a process that
# has only just exited, which is why the arms must be excluded at all — and for
# the same reason `python3` is excluded: `now_ms`'s fallback forks one twice per
# timed run, and its decaying %cpu made the guard discard the SAME cell four
# times in a row before this exclusion existed (2026-08-08, the st1 campaign).
BIN_RE=$(printf '%s' "$BIN" | sed 's/[][\.*^$/(){}?+|]/\\&/g')
busy_count() {
  ps -A -o %cpu,comm -r | awk -v me="$BIN_RE" \
    'NR>1 && $1>25 && $2 !~ /claude|ClaudeCode|versions\/|dav1d|python3/ && $2 !~ me {c++} END {print c+0}'
}
wait_quiet() {
  local w=0
  while [ "$(busy_count)" -gt 0 ]; do
    sleep 5; w=$((w+5))
    [ $w -ge 1800 ] && { echo "box never went idle" >&2; exit 4; }
  done
}
# Wall clock in ms. `EPOCHREALTIME` (bash 5) is a builtin, so it forks nothing
# and cannot show up in `busy_count`; the python3 fallback is for bash 3/4.
# Either source is fine for the two-point fit: a constant per-read offset (which
# is all a fork costs) cancels out of `beta = (hi - lo) / (NHI - NLO)`.
if [ -n "${EPOCHREALTIME:-}" ]; then
  now_ms() { local t=$EPOCHREALTIME; echo $(( ${t%%.*} * 1000 + 10#${t#*.} / 1000 )); }
else
  now_ms() { python3 -c 'import time;print(int(time.time()*1000))'; }
fi

time_one() {
  local arm=$1 vec=$2 t=$3 n=$4 t0 t1
  t0=$(now_ms)
  case "$arm" in
    dav1d_fd1) dav1d -i "$IVF/$vec.ivf" --muxer null --threads "$t" --framedelay 1 -q --limit "$n" >/dev/null 2>&1 ;;
    dav1d_def) dav1d -i "$IVF/$vec.ivf" --muxer null --threads "$t"                 -q --limit "$n" >/dev/null 2>&1 ;;
    *)         "$BIN/bench_$arm" "$AVIF/$vec.avif" "$t" "$n" 1 w >/dev/null 2>&1 ;;
  esac
  t1=$(now_ms); echo $((t1 - t0))
}

: > "$OUT"
n=${#ARMS[@]}
for round in $(seq 0 $((ROUNDS-1))); do
  for cell in "${CELLS[@]}"; do
    IFS=: read -r vec t <<< "$cell"
    while :; do
      wait_quiet; rows=(); dirty=0; fmax=0
      for k in $(seq 0 $((n-1))); do
        arm=${ARMS[$(( (k + round) % n ))]}
        lo=$(time_one "$arm" "$vec" "$t" "$NLO")
        hi=$(time_one "$arm" "$vec" "$t" "$NHI")
        f=$(busy_count); [ "$f" -gt "$fmax" ] && fmax=$f
        [ "$f" -gt 0 ] && dirty=1
        rows+=("$round	$arm	$vec	$t	$NLO	$lo	$NHI	$hi")
      done
      if [ $dirty -eq 0 ]; then
        for r in "${rows[@]}"; do printf '%s\t%s\n' "$r" "$fmax" >> "$OUT"; done
        echo "[$(date +%H:%M:%S)] r$round $vec t=$t committed (idle)" >&2
        break
      fi
      echo "[$(date +%H:%M:%S)] r$round $vec t=$t DISCARDED (foreign=$fmax)" >&2
    done
  done
done
echo "wrote $OUT" >&2
