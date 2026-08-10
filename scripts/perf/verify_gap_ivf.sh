#!/usr/bin/env bash
# Gap-to-dav1d sweep on IVF streams, so the campaign can see subsystems the 4K
# AVIF grid is structurally blind to.
#
# WHY THIS EXISTS: `v4k_8tile{,_10b}` — the only cells the standing grid holds —
# do ZERO loop restoration. LR is active in 696 of 768 dav1d-test-data vectors
# and can be 40-76% of a vector's decode time, and none of it has ever appeared
# in a gap number. Same for any other tool a single synthetic 4K still happens
# not to switch on.
#
# Same instrument as `verify_gap.sh`, same guards, one input file shared by both
# arms: whole-process wall clock at two frame counts, `total = alpha + beta *
# frames` fitted so binary load / IVF parse / decoder construction drop out.
#   rav1d arms -> $BIN/ivf_<arm> <ivf> <threads> <limit>   (bench_ivf_limit)
#   dav1d arms -> dav1d -i <ivf> --muxer null --threads N [--framedelay 1] --limit
#
# NO `nice` ON A TIMED RUN — Darwin maps a niced process onto E-cores and the
# wall clock distorts by ~40x. NO -C target-cpu=native. Default features.
#
# Usage: verify_gap_ivf.sh <out.tsv> [rounds]
# Env:   BIN, VECDIR, ARMS, CELLS
#        CELLS entries are "<relative ivf path, no .ivf>:<threads>:<nlo>:<nhi>"
#        (per-cell frame counts, because a 3-frame vector and a 140-frame one
#        cannot share a pair).
#
# ARMS entries may carry an in-loop-filter suffix: `<arm>@<inloopfilters>`, e.g.
#   ARMS="base base@norestoration dav1d_fd1 dav1d_fd1@norestoration"
# which maps to `RAV1D_INLOOP=norestoration` on a rav1d arm and
# `--inloopfilters norestoration` on a dav1d one. That makes a filter's COST
# attributable on BOTH decoders through one instrument, instead of profiling
# ours and arguing about theirs. Values are dav1d's spelling —
# all|none|nodeblock|nocdef|norestoration — deliberately, so a cell is the same
# string on both arms. It CHANGES OUTPUT PIXELS: attribution only, never a
# correctness comparison.
#
# Output columns: round arm vec threads nlo ms_lo nhi ms_hi foreign_max
set -u
export LC_ALL=C
OUT=${1:?out.tsv}; ROUNDS=${2:-5}
BIN=${BIN:-$HOME/tmp/vfy3/bin}
VECDIR=${VECDIR:-test-vectors/dav1d-test-data}
IFS=' ' read -r -a ARMS <<< "${ARMS:-base comp dav1d_fd1}"
IFS=' ' read -r -a CELLS <<< "${CELLS:-8-bit/data/00001147:1:1:3 10-bit/issues/318_tx_4x4:1:5:35 8-bit/data/00000645:1:20:140}"

BIN_RE=$(printf '%s' "$BIN" | sed 's/[][\.*^$/(){}?+|]/\\&/g')
busy_count() {
  ps -A -o %cpu,comm -r | awk -v me="$BIN_RE" \
    'NR>1 && $1>25 && $2 !~ /claude|ClaudeCode|versions\/|dav1d|python3/ && $2 !~ me {c++} END {print c+0}'
}
ALLOW_LOAD=${ALLOW_LOAD:-0}
wait_quiet() {
  local w=0
  [ "$ALLOW_LOAD" = 1 ] && return 0
  while [ "$(busy_count)" -gt 0 ]; do
    sleep 5; w=$((w+5))
    [ $w -ge 1800 ] && { echo "box never went idle" >&2; exit 4; }
  done
}
if [ -n "${EPOCHREALTIME:-}" ]; then
  now_ms() { local t=$EPOCHREALTIME; echo $(( ${t%%.*} * 1000 + 10#${t#*.} / 1000 )); }
else
  now_ms() { python3 -c 'import time;print(int(time.time()*1000))'; }
fi

time_one() {
  local spec=$1 vec=$2 t=$3 n=$4 t0 t1 arm il
  arm=${spec%%@*}                      # arm name, `@inloop` stripped
  il=${spec#*@}; [ "$il" = "$spec" ] && il=all
  t0=$(now_ms)
  case "$arm" in
    dav1d_fd1) dav1d -i "$VECDIR/$vec.ivf" --muxer null --threads "$t" --framedelay 1 --inloopfilters "$il" -q --limit "$n" >/dev/null 2>&1 ;;
    dav1d_def) dav1d -i "$VECDIR/$vec.ivf" --muxer null --threads "$t"                --inloopfilters "$il" -q --limit "$n" >/dev/null 2>&1 ;;
    *)         RAV1D_INLOOP="$il" "$BIN/ivf_$arm" "$VECDIR/$vec.ivf" "$t" "$n" "$arm" >/dev/null 2>&1 ;;
  esac
  t1=$(now_ms); echo $((t1 - t0))
}

: > "$OUT"
n=${#ARMS[@]}
for round in $(seq 0 $((ROUNDS-1))); do
  for cell in "${CELLS[@]}"; do
    IFS=: read -r vec t nlo nhi <<< "$cell"
    while :; do
      wait_quiet; rows=(); dirty=0; fmax=0
      for k in $(seq 0 $((n-1))); do
        arm=${ARMS[$(( (k + round) % n ))]}
        lo=$(time_one "$arm" "$vec" "$t" "$nlo")
        hi=$(time_one "$arm" "$vec" "$t" "$nhi")
        f=$(busy_count); [ "$f" -gt "$fmax" ] && fmax=$f
        [ "$f" -gt 0 ] && dirty=1
        rows+=("$round	$arm	$vec	$t	$nlo	$lo	$nhi	$hi")
      done
      if [ $dirty -eq 0 ] || [ "$ALLOW_LOAD" = 1 ]; then
        for r in "${rows[@]}"; do printf '%s\t%s\n' "$r" "$fmax" >> "$OUT"; done
        echo "[$(date +%H:%M:%S)] r$round $vec t=$t committed (foreign=$fmax)" >&2
        break
      fi
      echo "[$(date +%H:%M:%S)] r$round $vec t=$t DISCARDED (foreign=$fmax)" >&2
    done
  done
done
echo "wrote $OUT" >&2
