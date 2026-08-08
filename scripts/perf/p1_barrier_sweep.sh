#!/usr/bin/env bash
# Interleaved, order-rotated A/B/C sweep for the P1 barrier work.
#
# All arms for a cell run back to back within a round and the arm order rotates
# every round, so drift cannot land on one arm. The box-idle guard discards and
# re-runs any cell that saw a foreign process above 25% CPU during it.
#
# NO `nice` ON A TIMED RUN. On Darwin a positive nice value maps the process to
# background QoS and distorts wall clock by well over an order of magnitude;
# `nice -n 19` is for BUILDS only.
set -u
OUT=${1:?out.tsv}; ROUNDS=${2:-3}; REPS=${3:-3}; ITERS=${4:-4}
BIN=${BIN:-$HOME/tmp/rav1d-p1fix/bin}
VEC=${VEC:-$HOME/tmp/rav1d-perf/vec}
IFS=' ' read -r -a ARMS <<< "${ARMS:-A B C}"
IFS=' ' read -r -a CELLS <<< "${CELLS:-v4k_8tile:1 v4k_8tile:2 v4k_8tile:4 v4k_8tile:8 v4k_8tile_10b:4 v4k_8tile_10b:8}"

busy_count() { ps -A -o %cpu,comm -r | awk 'NR>1 && $1>25 && $2 !~ /claude|ClaudeCode|versions\// {c++} END {print c+0}'; }
wait_quiet() { local w=0; while [ "$(busy_count)" -gt 0 ]; do sleep 5; w=$((w+5)); [ $w -ge 900 ] && { echo "busy" >&2; exit 4; }; done; }

: > "$OUT"
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
        while IFS= read -r ms; do
          stage="${stage}${round}\t${vec}\t${t}\t${arm}\t${ms}\t${md5}\n"
        done < <(echo "$out" | awk -F'\t' '/^RESULT/{print $8}')
        [ "$(busy_count)" -gt 0 ] && dirty=1
      done
      if [ $dirty -eq 0 ]; then
        printf "$stage" >> "$OUT"
        echo "[$(date +%H:%M:%S)] round=$round $vec t=$t committed  load=$(uptime | sed 's/.*averages*: //')" >&2
        break
      fi
      echo "[$(date +%H:%M:%S)] round=$round $vec t=$t DISCARDED (contended)" >&2
    done
  done
done
echo "wrote $OUT" >&2
