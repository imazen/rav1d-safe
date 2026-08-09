#!/usr/bin/env bash
# Per-decode tail sweep. The two-point wall fit in verify_gap.sh gives ONE
# ms/frame number per round, which is the right instrument for a median and the
# wrong one for a tail: 15 rounds is 15 tail samples. This harness instead runs
# `bench_ab_decode <vec> <t> 1 REPS <label>` so every decode is its own timed
# sample, and interleaves the two arms at process granularity with a rotating
# order so thermal/scheduler drift cannot land on one arm.
#
# The distinction matters here because the hypothesis under test is a RARE
# event (contention is ~0.02%): the prediction is little median movement and a
# collapsed tail, so the tail has to be populated.
#
# NO `nice` on a timed run (Darwin maps it to E-cores, ~40x distortion).
#
# Usage: tail_sweep.sh <out.tsv> [rounds] [reps]
# Env:   BIN, AVIF, ARMS, CELLS
# Output: round arm vec threads rep ms foreign
set -u
export LC_ALL=C
OUT=${1:?out.tsv}; ROUNDS=${2:-6}; REPS=${3:-60}
BIN=${BIN:-$HOME/tmp/parklock/bin}
AVIF=${AVIF:-$HOME/tmp/rav1d-perf/vec}
IFS=' ' read -r -a ARMS <<< "${ARMS:-base head}"
IFS=' ' read -r -a CELLS <<< "${CELLS:-v4k_8tile:4 v4k_8tile:8 v4k_8tile:16 v4k_8tile_10b:8 v4k_8tile_10b:16}"

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

: > "$OUT"
n=${#ARMS[@]}
for round in $(seq 0 $((ROUNDS-1))); do
  for cell in "${CELLS[@]}"; do
    IFS=: read -r vec t <<< "$cell"
    while :; do
      wait_quiet; rows=(); dirty=0; fmax=0
      for k in $(seq 0 $((n-1))); do
        arm=${ARMS[$(( (k + round) % n ))]}
        while IFS=$'\t' read -r _tag _lbl _f _th rep _it ms _mspf; do
          rows+=("$round	$arm	$vec	$t	$rep	$ms")
        done < <("$BIN/bench_$arm" "$AVIF/$vec.avif" "$t" 1 "$REPS" "$arm" 2>/dev/null | grep '^RESULT')
        f=$(busy_count); [ "$f" -gt "$fmax" ] && fmax=$f
        [ "$f" -gt 0 ] && dirty=1
      done
      if [ $dirty -eq 0 ]; then
        for r in "${rows[@]}"; do printf '%s\t%s\n' "$r" "$fmax" >> "$OUT"; done
        echo "[$(date +%H:%M:%S)] r$round $vec t=$t committed (${#rows[@]} samples)" >&2
        break
      fi
      echo "[$(date +%H:%M:%S)] r$round $vec t=$t DISCARDED (foreign=$fmax)" >&2
    done
  done
done
echo "wrote $OUT" >&2
