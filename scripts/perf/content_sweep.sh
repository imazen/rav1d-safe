#!/usr/bin/env bash
# Decode cost vs CONTENT and QUALITY at a FIXED pixel count, plus per-content
# size ladders. The companion to `size_sweep.sh`, which swept size on one
# content class at one quality.
#
# Why this exists. `docs/SIZE_SWEEP.md` found a U-shaped ratio-to-dav1d with a
# hump at 0.6-2.4 MP and named the mechanism "per-block count" as an inference.
# Block count per pixel is not a function of pixel count: it is a function of
# how finely the encoder partitions, which content class and quality move by an
# order of magnitude at CONSTANT size. This harness varies those two axes so the
# inference can be tested instead of assumed.
#
# Instrument: identical to `size_sweep.sh` / `verify_gap.sh` — wall clock of the
# whole process at TWO frame counts per cell, `total = a + b*frames` fitted so
# process startup drops out and `b` is ms/frame. Frame counts are per cell.
#
# NO `nice` on a timed run (Darwin maps niced processes onto E-cores, ~40x wall
# distortion). NO -C target-cpu=native. Run the whole thing under `measlock`.
#
# Difference from `size_sweep.sh`: foreign load is recorded PER ARM (`f_arm`) as
# well as per group (`f_grp`). A per-group maximum cannot tell "one neighbour
# all round" (steady, common-mode, cancels in a paired ratio) from "a neighbour
# during arm B" (bursty, enters the ratio whole; single-run inflation up to +68%
# has been measured on this box).
#
# Usage: content_sweep.sh <out.tsv> [rounds]
# Env:   BIN (dir holding bench_<arm>), AVIF (dir of .avif), IVF (dir of .ivf),
#        ARMS, CELLS (space-separated <vector>:<threads>:<n_lo>:<n_hi>),
#        ALLOW_LOAD=1 to keep going on a busy box (then report ratios only).
# Output columns: round arm vec threads nlo ms_lo nhi ms_hi f_arm f_grp
set -u
export LC_ALL=C
OUT=${1:?out.tsv}; ROUNDS=${2:-9}
BIN=${BIN:-$HOME/tmp/sizehump/bin}
AVIF=${AVIF:-$HOME/tmp/sizehump/vec}
IVF=${IVF:-$HOME/tmp/sizehump/ivf}
IFS=' ' read -r -a ARMS <<< "${ARMS:-rs dav1d_fd1}"
IFS=' ' read -r -a CELLS <<< "${CELLS:?set CELLS}"

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
      wait_quiet; rows=(); dirty=0; fgrp=0
      for k in $(seq 0 $((n-1))); do
        arm=${ARMS[$(( (k + round) % n ))]}
        fa=$(busy_count)
        lo=$(time_one "$arm" "$vec" "$t" "$nlo")
        hi=$(time_one "$arm" "$vec" "$t" "$nhi")
        f=$(busy_count); [ "$f" -gt "$fa" ] && fa=$f
        [ "$fa" -gt "$fgrp" ] && fgrp=$fa
        [ "$fa" -gt 0 ] && dirty=1
        rows+=("$round	$arm	$vec	$t	$nlo	$lo	$nhi	$hi	$fa")
      done
      if [ $dirty -eq 0 ] || [ "$ALLOW_LOAD" = 1 ]; then
        for r in "${rows[@]}"; do printf '%s\t%s\n' "$r" "$fgrp" >> "$OUT"; done
        echo "[$(date +%H:%M:%S)] r$round $vec t=$t committed (f_grp=$fgrp)" >&2
        break
      fi
      echo "[$(date +%H:%M:%S)] r$round $vec t=$t DISCARDED (f_grp=$fgrp)" >&2
    done
  done
done
echo "wrote $OUT" >&2
