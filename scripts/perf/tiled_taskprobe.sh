#!/usr/bin/env bash
# Task/stage/occupancy census across the TILED arm at several thread counts.
#
# Every profile in the campaign so far is single-tile or t=1. This one asks the
# tiled question: on a forced-multi-tile vector, what appears at t=8 that is
# absent at t=1? Three quantities per cell, from ONE instrumented binary
# (`--features probe-tasktime`, src/probe_tasktime.rs):
#
#   * per-STAGE busy ms/frame  -- tile_recon vs the five filter stages. The
#     t8/t1 ratio of each stage says WHICH work inflates under threading.
#   * per-WORKER busy ms/frame -- a straggler is one worker far above the rest.
#   * the time-weighted concurrency HISTOGRAM (not just its mean), plus the
#     tail-restricted one: samples where no tile worker is live and at least one
#     filter worker is. That separates "idle pool" from "serial filter tail".
#
# The `ttu` arm is the same binary with the borrow tracker compiled out
# (`probe-tasktime-untracked`), so the tracker's share of any t=8 CPU inflation
# is a subtraction rather than an argument.
#
# NO `nice` (Darwin maps it to background QoS -> E-cores). Run under `measlock`.
# NO -C target-cpu=native.
#
# Usage: tiled_taskprobe.sh <outdir> [rounds]
# Env:   BIN (dir with bench_tt / bench_ttu), VEC (dir of .avif), ARMS, CELLS
set -u
export LC_ALL=C
OUT=${1:?outdir}; ROUNDS=${2:-3}
BIN=${BIN:-$HOME/tmp/tiledprof/bin}
VEC=${VEC:-$HOME/tmp/t8gap/vec}
IFS=' ' read -r -a ARMS <<< "${ARMS:-tt ttu}"
# cell = <vector>:<threads>:<iters>
DEFAULT_CELLS=""
for t in 1 2 4 8; do
  DEFAULT_CELLS="$DEFAULT_CELLS L1024x576_420_8b__t8:$t:40"
  DEFAULT_CELLS="$DEFAULT_CELLS L1024x576_420_8b:$t:40"
  DEFAULT_CELLS="$DEFAULT_CELLS L3840x2160_420_8b__t8:$t:8"
  DEFAULT_CELLS="$DEFAULT_CELLS L3840x2160_420_8b:$t:8"
done
IFS=' ' read -r -a CELLS <<< "${CELLS:-$DEFAULT_CELLS}"

mkdir -p "$OUT"
BIN_RE=$(printf '%s' "$BIN" | sed 's/[][\.*^$/(){}?+|]/\\&/g')
busy_count() {
  ps -A -o %cpu,comm -r | awk -v me="$BIN_RE" \
    'NR>1 && $1>25 && $2 !~ /claude|ClaudeCode|versions\/|dav1d|python3/ && $2 !~ me {c++} END {print c+0}'
}

n=${#ARMS[@]}
for round in $(seq 0 $((ROUNDS-1))); do
  for cell in "${CELLS[@]}"; do
    IFS=: read -r vec t iters <<< "$cell"
    for k in $(seq 0 $((n-1))); do
      arm=${ARMS[$(( (k + round) % n ))]}
      f=$(busy_count)
      log="$OUT/${arm}__${vec}__t${t}__r${round}.txt"
      "$BIN/bench_$arm" "$VEC/$vec.avif" "$t" "$iters" 1 "$arm" > "$log" 2>&1
      printf 'foreign_max\t%s\n' "$f" >> "$log"
      printf 'cell\t%s\t%s\t%s\t%s\t%s\n' "$arm" "$vec" "$t" "$iters" "$round" >> "$log"
    done
    echo "[$(date +%H:%M:%S)] r$round $vec t=$t done" >&2
  done
done
echo "wrote $OUT" >&2
