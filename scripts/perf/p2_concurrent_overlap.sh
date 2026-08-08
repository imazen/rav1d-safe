#!/usr/bin/env bash
# Run one test binary N-ways concurrently, R rounds, and count how many of the
# N*R processes exited nonzero.
#
# Why: `tile_threading_overlap`'s `multi_threaded_cdef_lpf_race` is
# load-dependent. It passes on an idle box and fails about a third of the time
# under six-way process pressure, because the fault it exposes is a race on the
# process-global `TILE_THREADING` flag between a 1-thread `rav1d_open` and the
# threaded decoders running beside it. A single green run proves nothing; this
# script is the instrument that does.
#
# Usage: p2_concurrent_overlap.sh <test-binary> <tag> [rounds] [parallel]
#   e.g. p2_concurrent_overlap.sh target/release/deps/tile_threading_overlap-* head 4 6
#
# Measured with it (2026-08-07, see benchmarks/p2_kernels_2026-08-07.meta):
#   perf/p1-barrier @ 6686b8f  8/24 and 9/24 FAILED
#   + the tile-threading latch  0/24 FAILED
set -u
BIN=${1:?test binary}
TAG=${2:?tag}
ROUNDS=${3:-4}
PAR=${4:-6}
OUTDIR=${OUTDIR:-${TMPDIR:-/tmp}/p2-conc}
mkdir -p "$OUTDIR"

fails=0
total=0
for r in $(seq 1 "$ROUNDS"); do
  for k in $(seq 1 "$PAR"); do
    (
      "$BIN" --ignored > "$OUTDIR/conc_${TAG}_${r}_$k.log" 2>&1
      echo $? > "$OUTDIR/rc_${TAG}_$k"
    ) &
  done
  wait
  for k in $(seq 1 "$PAR"); do
    total=$((total + 1))
    [ "$(cat "$OUTDIR/rc_${TAG}_$k")" != "0" ] && fails=$((fails + 1))
  done
done
echo "$TAG: $fails/$total FAILED  (logs in $OUTDIR)"
