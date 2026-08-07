#!/usr/bin/env bash
# Interleaved, order-rotated sweep for the P2 kernel work.
# Arms run back to back within a round; order rotates each round.
# Box-idle guard discards and re-runs a cell that saw a foreign process >25% CPU.
# NO `nice` ON A TIMED RUN (Darwin background QoS distorts wall clock ~40x).
set -u
OUT=${1:?out.tsv}; ROUNDS=${2:-3}; REPS=${3:-3}; ITERS=${4:-4}
BIN=${BIN:-$HOME/tmp/rav1d-p2k/bin}
VEC=${VEC:-$HOME/tmp/rav1d-perf/vec}
IFS=' ' read -r -a ARMS <<< "${ARMS:-base itx8 cdef lfmask lfbatch}"
IFS=' ' read -r -a CELLS <<< "${CELLS:-v4k_8tile:1 v4k_8tile:2 v4k_8tile:4 v4k_8tile:8 v4k_8tile_10b:1 v4k_8tile_10b:8}"
# Foreign = anything but the agent and OUR OWN arm binaries; the arm under
# test is the measurement, and macOS `ps` keeps a decaying %cpu for a process
# that has only just exited, so counting `bench_*` discards every t>1 cell.
busy_count() { ps -A -o %cpu,comm -r | awk 'NR>1 && $1>25 && $2 !~ /claude|ClaudeCode|versions\/|rav1d-p2k\/bin\/bench_/ {c++} END {print c+0}'; }
wait_quiet() { local w=0; while [ "$(busy_count)" -gt 0 ]; do sleep 5; w=$((w+5)); [ $w -ge 900 ] && { echo busy >&2; exit 4; }; done; }
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
        echo "[$(date +%H:%M:%S)] r$round $vec t=$t committed" >&2
        break
      fi
      echo "[$(date +%H:%M:%S)] r$round $vec t=$t DISCARDED (contended)" >&2
    done
  done
done
echo "wrote $OUT" >&2
