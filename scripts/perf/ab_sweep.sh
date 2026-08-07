#!/usr/bin/env bash
# Interleaved A/B decode-throughput sweep across two BUILDS of this crate.
#
# Usage:
#   scripts/perf/ab_sweep.sh <vector_dir> <out.tsv> <rounds> <label=bin> [label=bin ...]
#
# Each `label=bin` is a build of `examples/bench_ab_decode.rs` from one of the
# commits under comparison. All arms are run BACK TO BACK for each
# (vector, threads) cell, and their order is ROTATED each round, so no arm
# systematically inherits the others' thermal wake — running arm C last every
# time is enough to invent a several-percent regression on its own.
#
# NO `nice` IS APPLIED. On Darwin a positive nice value maps the process to
# background QoS (efficiency cores), which distorts wall clock by well over an
# order of magnitude. Timed runs must run at default priority.
#
# Emits the harness's raw RESULT/CHECKSUM/GEOM lines, prefixed with the round
# index; a cell whose decode fails is recorded as a FAIL line rather than
# silently dropped (the pre-#445 build genuinely cannot decode multi-tile
# content at threads >= 2).

set -u

# Refuse to run alongside another instance. Two sweeps sharing the box (or,
# worse, the same output file) silently doubles every measurement — an
# orphaned background run cost this harness one entire invalidated dataset.
others=$(pgrep -f '[a]b_sweep.sh' | grep -v "^$$\$" | wc -l | tr -d ' ')
if [ "$others" -gt 1 ]; then
  echo "another ab_sweep.sh is already running ($others procs); refusing" >&2
  exit 3
fi

# Wait for the box to go quiet before timing anything. An editor's background
# `cargo check` can take 800% CPU and will happily invalidate an entire sweep;
# `ps -r` is the reliable instantaneous signal (the load average decays too
# slowly to gate on).
wait_for_quiet() {
  local waited=0
  while [ "$(ps -A -o %cpu,comm -r | awk 'NR>1 && $1>25 {c++} END {print c+0}')" -gt 0 ]; do
    [ $waited -eq 0 ] && echo "waiting for the box to go quiet..." >&2
    sleep 15
    waited=$((waited + 15))
    if [ $waited -ge 900 ]; then
      echo "box still busy after ${waited}s; refusing to produce junk timings" >&2
      exit 4
    fi
  done
}
wait_for_quiet

VEC_DIR=${1:?vector_dir}
OUT=${2:?out.tsv}
ROUNDS=${3:-3}
shift 3
ARMS=("$@")
[ ${#ARMS[@]} -ge 2 ] || { echo "need at least two label=bin arms" >&2; exit 2; }

# vector:iters — iters chosen so one rep is ~0.5-2 s at that size.
CELLS=(
  "v256.avif:300"
  "v1024.avif:10"
  "v1024_10b.avif:10"
  "v4k_1tile.avif:2"
  "v4k_1tile_10b.avif:2"
  "v4k_8tile.avif:2"
  "v4k_8tile_10b.avif:2"
)
THREADS=(1 2 4 8)
REPS=3

: > "$OUT"
for round in $(seq 0 $((ROUNDS - 1))); do
  for cell in "${CELLS[@]}"; do
    vec=${cell%%:*}
    iters=${cell##*:}
    for t in "${THREADS[@]}"; do
      n=${#ARMS[@]}
      for k in $(seq 0 $((n - 1))); do
        arm=${ARMS[$(( (k + round) % n ))]}
        label=${arm%%=*}
        bin=${arm#*=}
        out=$("$bin" "$VEC_DIR/$vec" "$t" "$iters" "$REPS" "$label" 2>&1)
        rc=$?
        if [ $rc -ne 0 ]; then
          printf '%s\tFAIL\t%s\t%s\t%s\trc=%s\n' "$round" "$label" "$vec" "$t" "$rc" >> "$OUT"
        fi
        printf '%s\n' "$out" | grep -E '^(RESULT|CHECKSUM|GEOM)' \
          | sed "s/^/$round\t/" >> "$OUT"
        # Record the 1-minute load average with every cell so a contaminated
        # run is visible in the data instead of being averaged into a verdict.
        load=$(sysctl -n vm.loadavg | awk '{print $2}')
        printf '%s\tLOAD\t%s\t%s\t%s\t%s\n' "$round" "$label" "$vec" "$t" "$load" >> "$OUT"
        printf 'round=%s %-18s t=%s %-6s rc=%s load=%s\n' \
          "$round" "$vec" "$t" "$label" "$rc" "$load" >&2
      done
    done
  done
done
echo "wrote $OUT" >&2
