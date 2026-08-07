#!/usr/bin/env bash
# Interleaved sweep for the sharded borrow tracker.
#
# Thin wrapper over `ab_sweep.sh` that fixes the arm set and the output paths,
# so the sweep behind `benchmarks/shard_tracker_*.meta` is one command.
#
# Usage: scripts/perf/shard_sweep.sh <bindir> <vecdir> <out.tsv> [rounds]
#
# <bindir> holds one build of `examples/bench_ab_decode` per arm:
#   legacy   cargo build --release --example bench_ab_decode --features tracker-legacy
#   shard32  cargo build --release --example bench_ab_decode
#   shard64  cargo build --release --example bench_ab_decode --features shards-64
#
# NO `nice` ON TIMED RUNS. On Darwin a positive nice value maps the process to
# background QoS (efficiency cores) and distorts wall clock by more than an
# order of magnitude. Build under `nice -n 19`; measure at default priority.
set -u
BIN=${1:?bindir}
VEC=${2:?vecdir}
OUT=${3:?out.tsv}
ROUNDS=${4:-3}
here=$(cd "$(dirname "$0")" && pwd)
exec caffeinate -i "$here/ab_sweep.sh" "$VEC" "$OUT" "$ROUNDS" \
  legacy="$BIN/legacy" shard32="$BIN/shard32" shard64="$BIN/shard64"
