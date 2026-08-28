#!/usr/bin/env bash
# Self-time profiles for the 8bpc-vs-10bpc attribution and for the composition
# of the per-frame FIXED cost (the tiny cell), at t=1.
#
# The tiny cell is profiled because that is where alpha IS the frame: at 64x36
# the pixel work is ~2300 samples and everything else is setup, so the leaf
# table is a direct readout of what a thumbnail decode actually spends on.
#
# NO `nice`. Run under `measlock`.
#
# Usage: depth_profile.sh <outdir>
# Env:   BIN, AVIF, SECS, CELLS ("<vec>:<iters>" ...)
set -u
OUT=${1:?outdir}; mkdir -p "$OUT"
BIN=${BIN:-$HOME/tmp/szsweep/bin/bench_rs}
AVIF=${AVIF:-$HOME/tmp/szsweep/vec}
SECS=${SECS:-60}
HERE=$(cd "$(dirname "$0")" && pwd)
# iters sized so each run stays busy for > SECS + 5 s at the measured ms/frame
IFS=' ' read -r -a CELLS <<< "${CELLS:-\
L3840x2160_420_8b:400 L3840x2160_420_10b:340 \
L1024x576_420_8b:5000 L1024x576_420_10b:4400 \
L64x36_420_8b:1800000 L64x36_420_10b:1300000}"

for cell in "${CELLS[@]}"; do
  IFS=: read -r vec iters <<< "$cell"
  echo "== $vec (iters=$iters, ${SECS}s) ==" >&2
  bash "$HERE/prof_sample.sh" "$BIN" "$AVIF/$vec.avif" 1 "$OUT/$vec.sample.txt" "$iters" "$SECS"
  TOPN=400 python3 "$HERE/sample_selftime.py" "$OUT/$vec.sample.txt" --demangle \
    > "$OUT/$vec.selftime.tsv" 2>/dev/null
  python3 "$HERE/bucket_selftime.py" "$OUT/$vec.selftime.tsv" "$vec" | tee "$OUT/$vec.buckets.txt"
done
echo "wrote $OUT" >&2
