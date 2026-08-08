#!/usr/bin/env bash
# Interleaved multi-arm A/B at several thread counts, for the t1->t8 scaling
# campaign. Rotating arm order inside every (vector, threads) cell so thermal
# drift and background load hit every arm equally; median + min + max printed
# per cell so a sub-3% claim can be checked against its own noise band.
#
# NO `nice` on a timed run (Darwin maps it to background QoS -> E-cores, ~40x).
# NO -C target-cpu=native.
#
# Usage: scaling_ab.sh <out.tsv> [rounds]
# Env:   BIN (dir of arm binaries), VEC (dir of .avif), ARMS, CELLS, ITERS
#
# Columns: round  arm  vec  threads  iters  ms_per_frame  foreign
set -u
export LC_ALL=C
OUT=${1:?out.tsv}; ROUNDS=${2:-5}
BIN=${BIN:-$HOME/tmp/scal/bin}
VEC=${VEC:-$HOME/tmp/rav1d-perf/vec}
ITERS=${ITERS:-6}
IFS=' ' read -r -a ARMS <<< "${ARMS:-plain bs13 bs10 bs8}"
IFS=' ' read -r -a CELLS <<< "${CELLS:-v4k_8tile:1 v4k_8tile:4 v4k_8tile:8}"

BIN_RE=$(printf '%s' "$BIN" | sed 's/[][\.*^$/(){}?+|]/\\&/g')
busy_count() {
  ps -A -o %cpu,comm -r | awk -v me="$BIN_RE" \
    'NR>1 && $1>25 && $2 !~ /claude|ClaudeCode|versions\/|dav1d|python3/ && $2 !~ me {c++} END {print c+0}'
}

: > "$OUT"
n=${#ARMS[@]}
for round in $(seq 0 $((ROUNDS-1))); do
  for cell in "${CELLS[@]}"; do
    IFS=: read -r vec t <<< "$cell"
    for k in $(seq 0 $((n-1))); do
      arm=${ARMS[$(( (k + round) % n ))]}
      ms=$("$BIN/$arm" "$VEC/$vec.avif" "$t" "$ITERS" 1 s 2>/dev/null \
            | awk '$1=="RESULT"{print $8}' | tail -1)
      f=$(busy_count)
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$round" "$arm" "$vec" "$t" "$ITERS" "$ms" "$f" >> "$OUT"
    done
    echo "[$(date +%H:%M:%S)] r$round $vec t=$t done" >&2
  done
done
echo "wrote $OUT" >&2
