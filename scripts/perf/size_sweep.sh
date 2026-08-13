#!/usr/bin/env bash
# Decode-cost size sweep: fit `ms_per_frame = alpha + beta*pixels` for BOTH
# decoders, so the per-frame FIXED cost (alpha) is reported separately from the
# per-pixel cost (beta).
#
# Why this exists. Every gap number the rav1d-safe campaign produced is 4K, on
# two vectors. The product case is AVIF STILL IMAGES, which are usually far
# smaller, and a per-frame fixed cost that is invisible at 4K can be most of a
# thumbnail's decode. If our alpha is larger than dav1d's, small AVIFs are worse
# off than the 4K ratios suggest.
#
# Instrument: identical to scripts/perf/verify_gap.sh — wall clock of the whole
# process at TWO frame counts per cell, `total = a + b*frames` fitted so process
# startup (exec, mmap, container parse, decoder construction) drops out and `b`
# is ms/frame. The difference here is that the frame counts are PER CELL: a
# 64x36 frame decodes in ~0.06 ms, so a 2-vs-20 fit would be pure timer noise.
#
# NO `nice` ON A TIMED RUN (Darwin maps niced processes onto E-cores, ~40x wall
# distortion). NO -C target-cpu=native. Run the whole thing under `measlock`.
#
# Usage: size_sweep.sh <out.tsv> [rounds]
# Env:   BIN (dir holding bench_<arm>), AVIF (dir of .avif), IVF (dir of .ivf),
#        ARMS, CELLS
#
# Cell syntax: <vector>:<threads>:<n_lo>:<n_hi>
# Output columns: round arm vec threads nlo ms_lo nhi ms_hi foreign_max
set -u
export LC_ALL=C
OUT=${1:?out.tsv}; ROUNDS=${2:-9}
BIN=${BIN:-$HOME/tmp/szsweep/bin}
AVIF=${AVIF:-$HOME/tmp/szsweep/vec}
IVF=${IVF:-$HOME/tmp/szsweep/ivf}
IFS=' ' read -r -a ARMS <<< "${ARMS:-rs dav1d_fd1}"

# 6 sizes x 2 chroma formats x 2 depths, single tile, one encoder config.
DEFAULT_CELLS=""
for fmt in 420 444; do
  for d in 8b 10b; do
    DEFAULT_CELLS="$DEFAULT_CELLS L64x36_${fmt}_${d}:1:5000:50000"
    DEFAULT_CELLS="$DEFAULT_CELLS L256x144_${fmt}_${d}:1:500:5000"
    DEFAULT_CELLS="$DEFAULT_CELLS L512x288_${fmt}_${d}:1:100:1000"
    DEFAULT_CELLS="$DEFAULT_CELLS L1024x576_${fmt}_${d}:1:20:200"
    DEFAULT_CELLS="$DEFAULT_CELLS L2048x1152_${fmt}_${d}:1:5:50"
    DEFAULT_CELLS="$DEFAULT_CELLS L3840x2160_${fmt}_${d}:1:2:16"
  done
done
IFS=' ' read -r -a CELLS <<< "${CELLS:-$DEFAULT_CELLS}"

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
    dav1d_def) dav1d -i "$IVF/$vec.ivf" --muxer null --threads "$t"                 -q --limit "$n" >/dev/null 2>&1 ;;
    *)         "$BIN/bench_$arm" "$AVIF/$vec.avif" "$t" "$n" 1 w >/dev/null 2>&1 ;;
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
