#!/usr/bin/env bash
# Per-call-site-CLASS tracker attribution sweep.
#
# Same instrument and same shape as scripts/perf/verify_gap.sh — wall clock of
# the whole process, each stream run at two frame counts, `total = alpha +
# beta*frames` fitted so process startup drops out and `beta` is ms/frame —
# with one difference: the `cls_*` arms are ALL THE SAME BINARY, selected by
# `RAV1D_CLS_NULL`. That is deliberate. A per-arm build would let a codegen or
# layout difference masquerade as a class's cost, and the whole point of this
# sweep is that the between-arm delta IS the answer.
#
# Arms:
#   base        default build, real tracker, no instrument
#   cls_none    instrument present, nothing nulled  -> the baseline the deltas
#               are taken against; `cls_none - base` prices the instrument
#   cls_<class> that class's borrows are registered nowhere (UNSOUND; the call
#               on both add and remove survives, only the work goes)
#   cls_all     every class nulled
#   addnop      compile-time global "keep the call, delete the body"
#   untracked   no tracker at all
#   dav1d_fd1   dav1d 1.5.4 at --framedelay 1, the tile-threading-only model
#
# NO `nice` ON A TIMED RUN — Darwin maps a niced process onto E-cores and the
# wall clock distorts by ~40x. NO -C target-cpu=native.
#
# Usage: site_class_sweep.sh <out.tsv> [rounds]
# Output: round  arm  vec  threads  nlo  ms_lo  nhi  ms_hi  foreign_max
set -u
export LC_ALL=C
OUT=${1:?out.tsv}; ROUNDS=${2:-7}
BIN=${BIN:-$HOME/tmp/sitecls/bin}
AVIF=${AVIF:-$HOME/tmp/rav1d-perf/vec}
IVF=${IVF:-$HOME/tmp/recon-yard/vec}
NLO=${NLO:-2}; NHI=${NHI:-20}
IFS=' ' read -r -a ARMS <<< "${ARMS:-base cls_none cls_recon cls_recon+big cls_filter cls_decode cls_other cls_picwb cls_all addnop untracked dav1d_fd1}"
IFS=' ' read -r -a CELLS <<< "${CELLS:-v4k_8tile:1 v4k_8tile:2 v4k_8tile:4 v4k_8tile:8 v4k_8tile_10b:1 v4k_8tile_10b:2 v4k_8tile_10b:4 v4k_8tile_10b:8}"

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
  local arm=$1 vec=$2 t=$3 n=$4 t0 t1
  t0=$(now_ms)
  case "$arm" in
    dav1d_fd1) dav1d -i "$IVF/$vec.ivf" --muxer null --threads "$t" --framedelay 1 -q --limit "$n" >/dev/null 2>&1 ;;
    # `+` in the arm name is the mask's `,` — a comma in an arm name would be a
    # trap for every downstream CSV/awk reader of the TSV.
    cls_*)     spec=${arm#cls_}; RAV1D_CLS_NULL="${spec//+/,}" "$BIN/bench_cls" "$AVIF/$vec.avif" "$t" "$n" 1 w >/dev/null 2>&1 ;;
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
