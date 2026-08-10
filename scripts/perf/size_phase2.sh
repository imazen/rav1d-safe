#!/usr/bin/env bash
# Phase 2 of the size-sweep round, in one measlock hold:
#   (a) continuity anchor  — the campaign's own v4k_8tile / v4k_8tile_10b at
#       t=1, measured by the same instrument on the same day, so the new ladder
#       can be tied to the published 1.29x / 1.43x cells rather than assumed
#       comparable to them.
#   (b) t=8 cells on the ladder — needed to compare ONE 8-thread decode against
#       N single-thread decodes, which is the choice an image server actually
#       makes.
#   (c) concurrent throughput — decodes/second at N = 1..16 single-threaded
#       processes.
#
# NO `nice`. Run the whole thing under `measlock`.
# Usage: size_phase2.sh <outdir> [rounds]
set -u
OUT=${1:?outdir}; R=${2:-7}; mkdir -p "$OUT"
HERE=$(cd "$(dirname "$0")" && pwd)

echo "=== (a) continuity anchor: campaign 4K vectors ===" >&2
AVIF=$HOME/tmp/rav1d-perf/vec IVF=$HOME/tmp/recon-yard/vec \
  CELLS="v4k_8tile:1:2:20 v4k_8tile_10b:1:2:20 v4k_1tile:1:2:20 v4k_1tile_10b:1:2:20" \
  bash "$HERE/size_sweep.sh" "$OUT/anchor_gap.tsv" "$R"

echo "=== (b) ladder at t=8 ===" >&2
CELLS="L1024x576_420_8b:8:50:500 L1024x576_420_10b:8:50:500 L3840x2160_420_8b:8:10:100 L3840x2160_420_10b:8:10:100" \
  bash "$HERE/size_sweep.sh" "$OUT/t8_gap.tsv" "$R"

echo "=== (c) concurrent throughput, 1024x576 4:2:0 8bpc ===" >&2
VEC=L1024x576_420_8b NLO=10 NHI=100 NPROCS="1 2 4 8 12 16" \
  bash "$HERE/concurrent_throughput.sh" "$OUT/conc_1024_8b.tsv" "$R"

echo "=== (c2) concurrent throughput, 1024x576 4:2:0 10bpc ===" >&2
VEC=L1024x576_420_10b NLO=10 NHI=100 NPROCS="1 8 12" \
  bash "$HERE/concurrent_throughput.sh" "$OUT/conc_1024_10b.tsv" "$R"

echo "phase2 done -> $OUT" >&2
