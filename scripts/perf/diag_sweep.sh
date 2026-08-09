#!/usr/bin/env bash
# Interleaved, order-rotated arm sweep for the t>=4 scaling diagnosis.
#
# Same discipline as p1_sweep.sh / verify_gap.sh: every arm for a cell runs back
# to back inside a round, the arm order rotates per round so drift cannot land
# on one arm, and a cell whose round saw a foreign process over 25% CPU is
# discarded and re-run rather than committed.
#
# NO `nice` on a timed run (Darwin maps niced onto E-cores, ~40x distortion).
#
# Usage: diag_sweep.sh <out.tsv> [rounds] [iters] [reps]
# Env:   BIN, VEC, ARMS, CELLS
set -u
export LC_ALL=C
OUT=${1:?out.tsv}; ROUNDS=${2:-5}; ITERS=${3:-6}; REPS=${4:-2}
BIN=${BIN:-$HOME/tmp/rav1d-diag/bin}
VEC=${VEC:-$HOME/tmp/rav1d-perf/vec}
IFS=' ' read -r -a ARMS <<< "${ARMS:-base tt ttu}"
IFS=' ' read -r -a CELLS <<< "${CELLS:-v4k_8tile:1 v4k_8tile:2 v4k_8tile:4 v4k_8tile:8}"

BIN_RE=$(printf '%s' "$BIN" | sed 's/[][\.*^$/(){}?+|]/\\&/g')
busy_count() {
  ps -A -o %cpu,comm -r | awk -v me="$BIN_RE" \
    'NR>1 && $1>25 && $2 !~ /claude|ClaudeCode|versions\/|python3/ && $2 !~ me {c++} END {print c+0}'
}
wait_quiet() {
  local w=0
  while [ "$(busy_count)" -gt 0 ]; do
    sleep 5; w=$((w+5)); [ $w -ge 900 ] && { echo "box never went idle" >&2; exit 4; }
  done
}

printf 'round\tvec\tthreads\tarm\tms_per_frame\tmd5\tcpu_ms\tmean_active\n' > "$OUT"
n=${#ARMS[@]}
for round in $(seq 0 $((ROUNDS-1))); do
  for cell in "${CELLS[@]}"; do
    IFS=: read -r vec t <<< "$cell"
    while :; do
      wait_quiet; stage=""; dirty=0
      for k in $(seq 0 $((n-1))); do
        arm=${ARMS[$(( (k + round) % n ))]}
        out=$("$BIN/bench_$arm" "$VEC/$vec.avif" "$t" "$ITERS" "$REPS" "$arm" 2>&1)
        md5=$(echo "$out" | awk -F'\t' '/^CHECKSUM/{print $5}')
        # CPU ms/frame = sum over stage bodies; absent (0) on the no-probe arm.
        cpu=$(echo "$out" | awk '/^PROBE stage_ms_per_frame/{s+=$4} END{printf "%.3f", s+0}')
        ma=$(echo "$out" | awk '/^PROBE mean_active /{print $3}')
        [ -z "$ma" ] && ma=NA
        while IFS= read -r ms; do
          stage="${stage}${round}\t${vec}\t${t}\t${arm}\t${ms}\t${md5}\t${cpu}\t${ma}\n"
        done < <(echo "$out" | awk -F'\t' '/^RESULT/{print $8}')
        [ "$(busy_count)" -gt 0 ] && dirty=1
      done
      if [ $dirty -eq 0 ]; then
        printf "$stage" >> "$OUT"
        echo "[$(date +%H:%M:%S)] round=$round $vec t=$t committed" >&2
        break
      fi
      echo "[$(date +%H:%M:%S)] round=$round $vec t=$t DISCARDED (contended)" >&2
    done
  done
done
echo "wrote $OUT" >&2
