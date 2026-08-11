#!/usr/bin/env bash
# The TIMED half of "make the derived rows-per-block rule the default".
#
# Phase W — the SHIPPED default against dav1d on the cells the campaign has been
#   mis-quoting, at t=1 AND t=8, with the pre-2026-08-11 block-count rule
#   (`bps-blocks`) interleaved as the BASE and `bps-half` as the best global
#   constant. The point of the round is that "ours/dav1d" must be the DEFAULT
#   build's number, so plain, base and dav1d are in one interleave.
#
# Phase P — the 3x3 per-plane shift factorial on 512x576, the size sweep's one
#   unexplained cell. `RAV1D_PIN_SHIFT` wrappers (built here) are the only way to
#   move a luma shift without a chroma one; the rows rule and the ladder both
#   move them together, so no combination of arms can separate the planes.
#
# NO `nice` on a timed run (Darwin maps positive nice to background QoS and lands
# the process on E-cores, ~40x). Both phases run under `measlock`.
#
# ROUNDS default to 8 and the REPORT DISCARDS ROUND 0: the first touch of each
# (arm, cell) pair is cold (a smoke run read ms_lo 382 against 326-330 warm on
# the same cell), which is the effect benchmarks/cost_census_2026-08-10.meta
# records. Eight rounds minus the cold one leaves n = 7.
#
# Usage: bpsrows_timed.sh <outdir> [wall_rounds] [pin_rounds]
set -u
OUT=${1:?outdir}; WR=${2:-8}; PR=${3:-8}
BIN=${BIN:-$HOME/tmp/bpsrows/bin}
VEC=${VEC:-$HOME/tmp/bpsrows/vec}
IVF=${IVF:-$HOME/tmp/bpsrows/ivf}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
export PATH="$HOME/bin:$PATH"

# ---- Phase W ---------------------------------------------------------------
WALL_ARMS=${WALL_ARMS:-"plain bpsblocks bpshalf untracked dav1d_fd1"}
CELLS8=""
CELLS1=""
# vector:n_lo:n_hi — n_hi is the IVF's real length, never past it.
for spec in \
  "C1024x192_420_8b__t8:60:600" \
  "C1024x384_420_8b__t8:30:300" \
  "C3840x256_420_8b__t8:12:120" \
  "C1024x576_420_8b__t8:20:200" \
  "C256x2048_420_8b__t8:22:225" \
  "C512x576_420_8b__t8:40:400" \
  "L1024x576_420_10b__t8:20:200" \
  "v4k_8tile:4:40" ; do
  v=${spec%%:*}; n=${spec#*:}
  CELLS8="$CELLS8 $v:8:$n"
  CELLS1="$CELLS1 $v:1:$n"
done

echo "[$(date +%H:%M:%S)] phase W t=8, $WR rounds" >&2
BIN="$BIN" AVIF="$VEC" IVF="$IVF" ARMS="$WALL_ARMS" CELLS="$CELLS8" \
  measlock bpsrows-w8 -- "$here/tiled_wallcpu.sh" "$OUT/wallcpu_t8.tsv" "$WR" \
  2> >(tee "$OUT/w8.log" >&2)

echo "[$(date +%H:%M:%S)] phase W t=1, $WR rounds" >&2
BIN="$BIN" AVIF="$VEC" IVF="$IVF" ARMS="$WALL_ARMS" CELLS="$CELLS1" \
  measlock bpsrows-w1 -- "$here/tiled_wallcpu.sh" "$OUT/wallcpu_t1.tsv" "$WR" \
  2> >(tee "$OUT/w1.log" >&2)

# ---- Phase P ---------------------------------------------------------------
# 512x576 8-bit 4:2:0: luma stride 512, chroma stride 256. The block-count rule
# lands on (10, 8); bps1 = (11, 9), bps-half = (12, 10), the derived rule =
# (11, 10). (12, 9) is the corner NO arm on offer can reach, and the six
# single-plane cells are what make the factorial additive-testable.
mkdir -p "$BIN"
for L in 10 11 12; do
  for C in 8 9 10; do
    w="$BIN/bench_pinL${L}C${C}"
    printf '#!/bin/sh\nRAV1D_PIN_SHIFT="512:%s,256:%s" exec "%s/bench_pin" "$@"\n' \
      "$L" "$C" "$BIN" > "$w"
    chmod +x "$w"
  done
done
PIN_ARMS=${PIN_ARMS:-"plain pinL10C8 pinL10C9 pinL10C10 pinL11C8 pinL11C9 pinL11C10 pinL12C8 pinL12C9 pinL12C10"}
echo "[$(date +%H:%M:%S)] phase P (512x576 per-plane factorial), $PR rounds" >&2
BIN="$BIN" AVIF="$VEC" IVF="$IVF" ARMS="$PIN_ARMS" \
  CELLS="C512x576_420_8b__t8:8:40:400" \
  measlock bpsrows-pin -- "$here/tiled_wallcpu.sh" "$OUT/wallcpu_pin.tsv" "$PR" \
  2> >(tee "$OUT/pin.log" >&2)

echo "[$(date +%H:%M:%S)] done -> $OUT" >&2
