#!/usr/bin/env bash
# dav1d throughput measured by WALL CLOCK with the process-startup intercept
# fitted out, because --frametimes is not a throughput measure once frame
# threading is on: with a full pipeline the CLI drains queued pictures in a
# burst, so the inter-output interval collapses to queue-pop time and reports
# impossible speedups (v1024 t=8 read 0.47 ms/frame against a 15.4 ms t=1).
#
# For each (vector, threads, framedelay) run the same stream at TWO frame
# counts and fit total = alpha + beta*frames; beta is the per-frame decode
# cost, alpha absorbs exec + mmap + sequence-header setup + teardown.
#
# rav1d-safe is included as a wall-clock arm too, over the same two frame
# counts, so both sides are measured by the same instrument.
set -u
OUT=${1:?out.tsv}
ROUNDS=${2:-3}
BENCH=$HOME/tmp/recon-yard/target-main/release/examples/bench_ab_decode
AVIF=$HOME/tmp/rav1d-perf/vec
IVF=$HOME/tmp/recon-yard/vec

busy_count() {
  ps -A -o %cpu,comm -r \
    | awk 'NR>1 && $1>25 && $2 !~ /claude|ClaudeCode|versions\// {c++} END {print c+0}'
}
wait_for_quiet() {
  local w=0
  while [ "$(busy_count)" -gt 0 ]; do
    sleep 10; w=$((w+10)); [ $w -ge 3600 ] && { echo "box busy; refusing" >&2; exit 4; }
  done
}

now_ms() { python3 -c 'import time;print(int(time.time()*1000))'; }

# vector : n_low : n_high  (frame counts for the two-point fit)
CELLS=(
  "v256:50:800"
  "v1024:6:90"
  "v1024_10b:6:90"
  "v4k_1tile:2:20"
  "v4k_1tile_10b:2:20"
  "v4k_8tile:2:20"
  "v4k_8tile_10b:2:20"
)
THREADS=(1 2 4 8)
ARMS=(rs dav1d_fd1 dav1d_fdA)

time_one() {  # arm vec threads nframes -> echoes elapsed ms
  local arm=$1 vec=$2 t=$3 n=$4 t0 t1
  t0=$(now_ms)
  case "$arm" in
    rs)        "$BENCH" "$AVIF/$vec.avif" "$t" "$n" 1 w >/dev/null 2>&1 ;;
    dav1d_fd1) dav1d -i "$IVF/$vec.ivf" --muxer null --threads "$t" --framedelay 1 \
                     -q --limit "$n" >/dev/null 2>&1 ;;
    dav1d_fdA) dav1d -i "$IVF/$vec.ivf" --muxer null --threads "$t" \
                     -q --limit "$n" >/dev/null 2>&1 ;;
  esac
  t1=$(now_ms)
  echo $((t1 - t0))
}

: > "$OUT"
for round in $(seq 0 $((ROUNDS-1))); do
  for cell in "${CELLS[@]}"; do
    IFS=: read -r vec nlo nhi <<< "$cell"
    for t in "${THREADS[@]}"; do
      while : ; do
        wait_for_quiet
        stage=""; dirty=0
        n=${#ARMS[@]}
        for k in $(seq 0 $((n-1))); do
          arm=${ARMS[$(( (k + round) % n ))]}
          lo=$(time_one "$arm" "$vec" "$t" "$nlo")
          hi=$(time_one "$arm" "$vec" "$t" "$nhi")
          stage="${stage}${round}\tWALLFIT\t${arm}\t${vec}\t${t}\t${nlo}\t${lo}\t${nhi}\t${hi}\n"
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
done
echo "wrote $OUT" >&2
