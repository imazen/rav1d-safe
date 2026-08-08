#!/usr/bin/env bash
# Interleaved A/B for the verify/compose-4 campaign.
#
# One cell = one (vector, threads) pair. Inside a round every arm is run
# back-to-back with the ARM ORDER ROTATED by the round index, so a drifting box
# inflates every arm about equally and the paired per-round ratio survives what
# the absolute ms does not.
#
# In-process timer (`bench_ab_decode`'s ms_per_frame) rather than the two-point
# wall fit: process startup is already outside it, and we want the checksum the
# same run emits so every timed row doubles as a bit-identity check.
#
# NO `nice` ON A TIMED RUN — Darwin maps a niced process onto E-cores.
#
# Usage: v4_ab.sh <out.tsv> [rounds]
# Env:   BIN ARMS CELLS ITERS
# Cols:  round arm vec threads ms_per_frame md5 foreign
set -u
export LC_ALL=C
OUT=${1:?out.tsv}; ROUNDS=${2:-9}
BIN=${BIN:-$HOME/tmp/rav1d-v4/bin}
AVIF=${AVIF:-$HOME/tmp/rav1d-perf/vec}
ITERS=${ITERS:-6}
IFS=' ' read -r -a ARMS <<< "${ARMS:-base cast}"
IFS=' ' read -r -a CELLS <<< "${CELLS:-v4k_8tile_10b:1 v4k_8tile:1}"

BIN_RE=$(printf '%s' "$BIN" | sed 's/[][\.*^$/(){}?+|]/\\&/g')
busy_count() {
  ps -A -o %cpu,comm -r | awk -v me="$BIN_RE" \
    'NR>1 && $1>25 && $2 !~ /claude|ClaudeCode|versions\/|python3/ && $2 !~ me {c++} END {print c+0}'
}

: > "$OUT"
n=${#ARMS[@]}
for round in $(seq 0 $((ROUNDS-1))); do
  for cell in "${CELLS[@]}"; do
    IFS=: read -r vec t <<< "$cell"
    fmax=0
    for k in $(seq 0 $((n-1))); do
      arm=${ARMS[$(( (k + round) % n ))]}
      out=$("$BIN/bench_$arm" "$AVIF/$vec.avif" "$t" "$ITERS" 1 "$arm" 2>/dev/null)
      ms=$(printf '%s\n' "$out" | awk '$1=="RESULT"{print $8}' | tail -1)
      md5=$(printf '%s\n' "$out" | awk '$1=="CHECKSUM"{print $5}' | tail -1)
      f=$(busy_count); [ "$f" -gt "$fmax" ] && fmax=$f
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$round" "$arm" "$vec" "$t" "$ms" "$md5" "$f" >> "$OUT"
    done
    echo "[$(date +%H:%M:%S)] r$round $vec t=$t done (foreign=$fmax)" >&2
  done
done
echo "wrote $OUT" >&2
