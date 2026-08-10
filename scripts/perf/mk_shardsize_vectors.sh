#!/usr/bin/env bash
# Multi-tile decode vectors for the PICTURE-SIZE sweep of the shard-granularity
# ladder (docs/SHARD_GRANULARITY.md §2 says the error is a function of picture
# HEIGHT; this builds the grid that tests that).
#
# WHY NOT THE EXISTING LADDER
# ---------------------------
# `scripts/perf/mk_size_ladder.sh` forces **one tile** on purpose, and the rung's
# win only exists at t=8 on TILED content (`tile_concurrency() < 2` disarms the
# adaptive shift outright). So every cell here is encoded 4x2 = 8 tiles, the same
# geometry the prior record's headline cell used.
#
# WHY CROPS, NOT DOWNSCALES
# -------------------------
# The existing ladder downscales one photo to each 16:9 size, so detail-per-pixel
# FALLS as the picture grows and content is confounded with geometry. This grid
# CROPS the 4K source (centered, 1:1), so every cell has the same detail per pixel
# and the only thing that changes is the rectangle. That matters here because the
# question is explicitly geometric — rows per tracker block — and because two of
# the three axes deliberately break 16:9 (a fixed-width height ladder and a
# fixed-height width ladder), which no downscale can express.
#
# The consequence, stated so it is not mistaken for a bug: bytes per pixel are NOT
# comparable with `benchmarks/size_sweep_*`, and neither are absolute ms/frame.
# `L1024x576_420_8b__t8` (a downscale) is carried along unchanged as a bridge cell
# so this round can be reconnected to the prior record's headline.
#
# ONE content class (photo) at ONE quality point, like the ladder it extends. Sound
# for geometry questions, not sound for byte-cost questions.
#
# PREREQUISITES: avifenc on PATH, sips (macOS), and
#   cargo build --release --example avif_to_ivf
#
# USAGE: mk_shardsize_vectors.sh [outdir]   (default $HOME/tmp/shardsize)
set -euo pipefail
ROOT="${1:-$HOME/tmp/shardsize}"
SRC4K="${SRC4K:-$HOME/tmp/szsweep/src/src_3840x2160.png}"
SRC="$ROOT/src"; OUT="$ROOT/vec"; IVF="$ROOT/ivf"; LOG="$ROOT/log"
mkdir -p "$SRC" "$OUT" "$IVF" "$LOG"
[ -s "$SRC4K" ] || { echo "missing source $SRC4K" >&2; exit 1; }

# w h  — three axes through one grid:
#   H: width pinned at 1024, height walked          (the axis §2 predicts)
#   W: height pinned at 576, width walked           (the control §2 predicts flat)
#   D: 16:9 diagonal                                (what real content looks like)
CELLS="
1024 192
1024 288
1024 384
1024 576
1024 768
1024 1024
1024 1440
1024 2048
1024 2160
512 576
2048 576
3840 576
512 288
2048 1152
3840 2160
256 2048
3840 256
"

# The last two are DISCRIMINATING cells, added after the first pass and named as
# such so they are not read as part of the ladder. A single global rung is one
# constant for every picture; a rule derived from rows-per-block is not. These
# two are where the two disagree most:
#   256x2048  tall and narrow — stride 256 puts EIGHT rows in a block already at
#             the shipped rule, so a two-shift rung over-coarsens it and a rows
#             rule leaves it alone.
#   3840x256  wide and short — stride 3840 puts HALF a row in a block, further
#             below the target than any ladder cell, so the rung under-coarsens.
# Predictions were written down before either was decoded.

# --- 1. crop ----------------------------------------------------------------
# `sips -c H W` crops CENTERED at 1:1. Every cell is therefore the same photo at
# the same detail level, differing only in the rectangle kept.
while read -r w h; do
  [ -z "${w:-}" ] && continue
  o="$SRC/c_${w}x${h}.png"
  [ -s "$o" ] && continue
  cp "$SRC4K" "$o"
  sips -c "$h" "$w" "$o" >/dev/null
done <<< "$CELLS"

# --- 2. encode, 4x2 = 8 tiles ------------------------------------------------
# Same encoder/speed/quality as mk_size_ladder.sh and ~/tmp/t8gap/enc_tiles.sh;
# the ONLY difference from the single-tile ladder is the two tile flags.
# aom may emit fewer tiles than asked if the frame is too small in superblocks —
# that is why step 3 verifies the tile count rather than trusting the request.
while read -r w h; do
  [ -z "${w:-}" ] && continue
  o="$OUT/C${w}x${h}_420_8b__t8.avif"
  [ -s "$o" ] && continue
  nice -n 19 avifenc -s 6 -q 70 -y 420 -d 8 \
    --tilerowslog2 1 --tilecolslog2 2 -j 8 \
    --ignore-exif --ignore-xmp \
    "$SRC/c_${w}x${h}.png" "$o" > "$LOG/enc_${w}x${h}.log" 2>&1
  printf '%s\t%s\n' "$(basename "$o")" "$(stat -f%z "$o")"
done <<< "$CELLS"

# --- 3. AVIF -> IVF ----------------------------------------------------------
# Frame counts scale with area so every cell's `n_hi` run lands near a second and
# the two-point fit is not measuring timer noise (AGENT_BRIEF §2: a 64x36 frame
# decodes in 45 us).
B="${AVIF_TO_IVF:-target/release/examples/avif_to_ivf}"
[ -x "$B" ] || { echo "build it first: cargo build --release --example avif_to_ivf" >&2; exit 1; }
while read -r w h; do
  [ -z "${w:-}" ] && continue
  v="C${w}x${h}_420_8b__t8"
  [ -s "$IVF/$v.ivf" ] && continue
  n=$(python3 -c "print(max(16, min(600, round(200*589824/($w*$h)))))")
  nice -n 19 "$B" "$OUT/$v.avif" "$n" "$IVF/$v.ivf"
done <<< "$CELLS"

echo
echo "NEXT: scripts/perf/shardsize_tiles.py $OUT   (tile count + dims, per vector)"
echo "THEN: verify every cell against dav1d before timing anything."
