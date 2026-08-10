#!/usr/bin/env bash
# Drive phases 2 and 3 as SEPARATE measlock holds, most-valuable first, so a
# second agent can slot in between them instead of waiting out one long lock.
#   1. depth/size profiles        (Q2 core: where 10bpc loses, and what turns
#                                  on between 512x288 and 1024x576)
#   2. concurrent throughput      (Q1's explicit ask: decodes/second under
#                                  N single-threaded decoders)
#   3. continuity anchor + t=8    (ties the ladder to the campaign's published
#                                  4K cells; prices 1x8-thread vs 8x1-thread)
set -u
OUT=${1:?outdir}; R=${2:-7}; mkdir -p "$OUT"
HERE=$(cd "$(dirname "$0")" && pwd)
ML=$HOME/bin/measlock

CELLS="$(cat "$HOME/tmp/szsweep/prof_cells.txt")" \
  "$ML" szprof -- bash "$HERE/depth_profile.sh" "$OUT/prof" 2>&1 | tail -40

VEC=L1024x576_420_8b NLO=10 NHI=100 NPROCS="1 2 4 8 12 16" \
  "$ML" szconc -- bash "$HERE/concurrent_throughput.sh" "$OUT/conc_1024_8b.tsv" "$R"

VEC=L1024x576_420_10b NLO=10 NHI=100 NPROCS="1 8 12" \
  "$ML" szconc10 -- bash "$HERE/concurrent_throughput.sh" "$OUT/conc_1024_10b.tsv" "$R"

AVIF=$HOME/tmp/rav1d-perf/vec IVF=$HOME/tmp/recon-yard/vec \
  CELLS="v4k_8tile:1:2:20 v4k_8tile_10b:1:2:20" \
  "$ML" szanchor -- bash "$HERE/size_sweep.sh" "$OUT/anchor_gap.tsv" "$R"

CELLS="L1024x576_420_8b:8:50:500 L1024x576_420_10b:8:50:500 L3840x2160_420_8b:8:10:100 L3840x2160_420_10b:8:10:100" \
  "$ML" szt8 -- bash "$HERE/size_sweep.sh" "$OUT/t8_gap.tsv" "$R"

echo "phase23 done -> $OUT" >&2
