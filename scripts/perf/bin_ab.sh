#!/usr/bin/env bash
# Rotating-order interleaved A/B of two staged bench binaries on one vector.
# Records the foreign>25%CPU count per cell so a contended round is visible.
set -u
OUT=${1:?out}; VEC=${2:?vec}; ITERS=${3:-12}; ROUNDS=${4:-9}; shift 4
ARMS=("$@")
busy() { ps -A -o %cpu,comm -r | awk 'NR>1 && $1>25 && $2 !~ /claude|ClaudeCode|versions\/|lrneon\/bin/ {c++} END {print c+0}'; }
: > "$OUT"
n=${#ARMS[@]}
for r in $(seq 0 $((ROUNDS-1))); do
  for k in $(seq 0 $((n-1))); do
    A=${ARMS[$(( (k + r) % n ))]}
    ms=$("$HOME/tmp/lrneon/bin/bench_$A" "$VEC" 1 "$ITERS" 1 "$A" 2>/dev/null | awk -F'\t' '/^RESULT/{print $8}')
    printf '%s\t%s\t%s\t%s\n' "$r" "$A" "$ms" "$(busy)" >> "$OUT"
  done
  echo "[$(date +%H:%M:%S)] round $r" >&2
done
