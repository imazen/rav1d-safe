#!/usr/bin/env bash
# dav1d vs our binaries by WALL CLOCK with the process-startup intercept fitted
# out: run each stream at two frame counts, fit total = alpha + beta*frames.
# Same instrument on both sides. NO nice on a timed run.
set -u
OUT=${1:?out.tsv}; ROUNDS=${2:-3}
BIN=${BIN:-$HOME/tmp/rav1d-p2k/bin}
AVIF=$HOME/tmp/rav1d-perf/vec
IVF=$HOME/tmp/recon-yard/vec
CELLS=("v4k_8tile:2:20" "v4k_8tile_10b:2:20")
THREADS=(1 8)
IFS=' ' read -r -a ARMS <<< "${ARMS:-base head dav1d_fd1}"
# Exclude the arms under test by BASENAME (see the note in p2_sweep.sh): macOS
# keeps a decaying %cpu for a just-exited process, so counting our own
# bench_*/dav1d discards every t>1 cell forever.
busy_count() { ps -A -o %cpu,comm -r | awk 'NR>1 && $1>25 && $2 !~ /claude|ClaudeCode|versions\/|bench_|probe_|dav1d/ {c++} END {print c+0}'; }
wait_quiet() { local w=0; while [ "$(busy_count)" -gt 0 ]; do sleep 5; w=$((w+5)); [ $w -ge 900 ] && exit 4; done; }
now_ms() { python3 -c 'import time;print(int(time.time()*1000))'; }
time_one() {
  local arm=$1 vec=$2 t=$3 n=$4 t0 t1
  t0=$(now_ms)
  case "$arm" in
    dav1d_fd1) dav1d -i "$IVF/$vec.ivf" --muxer null --threads "$t" --framedelay 1 -q --limit "$n" >/dev/null 2>&1 ;;
    *)         "$BIN/bench_$arm" "$AVIF/$vec.avif" "$t" "$n" 1 w >/dev/null 2>&1 ;;
  esac
  t1=$(now_ms); echo $((t1 - t0))
}
: > "$OUT"
for round in $(seq 0 $((ROUNDS-1))); do
  for cell in "${CELLS[@]}"; do
    IFS=: read -r vec nlo nhi <<< "$cell"
    for t in "${THREADS[@]}"; do
      while :; do
        wait_quiet; stage=""; dirty=0; n=${#ARMS[@]}
        for k in $(seq 0 $((n-1))); do
          arm=${ARMS[$(( (k + round) % n ))]}
          lo=$(time_one "$arm" "$vec" "$t" "$nlo")
          hi=$(time_one "$arm" "$vec" "$t" "$nhi")
          stage="${stage}${round}\t${arm}\t${vec}\t${t}\t${nlo}\t${lo}\t${nhi}\t${hi}\n"
          [ "$(busy_count)" -gt 0 ] && dirty=1
        done
        if [ $dirty -eq 0 ]; then printf "$stage" >> "$OUT"; echo "[$(date +%H:%M:%S)] $vec t=$t r$round ok" >&2; break; fi
        echo "[$(date +%H:%M:%S)] $vec t=$t r$round DISCARDED" >&2
      done
    done
  done
done
