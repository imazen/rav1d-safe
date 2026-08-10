#!/usr/bin/env bash
# Rebuild the decode size-ladder vectors used by docs/SIZE_SWEEP.md and
# docs/SIZE_SWEEP_T8.md.
#
# WHY THIS EXISTS
# ---------------
# The vectors themselves are ~1.2 GB and deliberately NOT in git. They are the
# deterministic output of a pinned `avifenc` invocation, so the recipe is the
# artefact worth keeping — but until this file landed, the recipe lived only in
# `~/tmp/szsweep/`, i.e. in scratch, alongside the vectors it was supposed to
# make recoverable. A wipe would have cost the ability to reproduce two rounds
# of measurement, not just the bytes. There is no /mnt/v mount and no R2
# credential on the macOS box these were built on, so recipe-in-git is the whole
# durability story.
#
# WHAT THE VECTORS ARE, AND THE ONE THING TO KNOW BEFORE REUSING THEM
# -------------------------------------------------------------------
# Six 16:9 sizes x {YUV420, YUV444} x {8, 10} bpc = 24 AVIF stills, all
# downscales of ONE photo (the campaign's `src4k.png`) via `sips`, encoded at
# aom speed 6 / quality 70, no film grain, full range.
#
# **SINGLE TILE IS FORCED** (`--tilerowslog2 0 --tilecolslog2 0`). That is a
# deliberate control, not a property of real AVIFs, and it has already been
# misread once: "all 24 ladder vectors are 1 tile" was reported as a discovery
# about production encodes when it is a constant this script sets. ravif's
# shipped policy is `(px / 1 MP).min(px / min_tile_size^2)`, so a 4K still gets
# ~7 tiles in production and a <=1 MP still gets 1. If you want to measure what
# production emits, drop the two tile flags.
#
# Also: ONE content class at ONE quality point. The project's sweep discipline
# asks for >=3 content classes and low-q density; this ladder has neither, so it
# is sound for size-scaling questions and NOT sound for byte-cost questions.
# A byte-cost claim from this ladder was contradicted by ravif's own broader
# sweep (+0.9%/+2.0%/+7.9% at 2/4/64 tiles, up to +28% at Q30 on smooth
# content) — see the note in ravif's `av1encoder.rs` tile policy.
#
# PREREQUISITES
#   avifenc (libavif CLI)          — encoder, must be on PATH
#   sips                           — macOS built-in, for the downscales
#   cargo build --release --example avif_to_ivf   — for the IVF conversion
#
# USAGE
#   scripts/perf/mk_size_ladder.sh /path/to/src4k.png [outdir]
# Default outdir is $HOME/tmp/szsweep, which is where the committed benchmark
# records expect to find it.
set -euo pipefail

SRC4K="${1:?usage: mk_size_ladder.sh <src4k.png> [outdir]}"
ROOT="${2:-$HOME/tmp/szsweep}"
SRC="$ROOT/src"; OUT="$ROOT/vec"; IVF="$ROOT/ivf"; LOG="$ROOT/log"
mkdir -p "$SRC" "$OUT" "$IVF" "$LOG"

SIZES="64x36 256x144 512x288 1024x576 2048x1152 3840x2160"

# --- 1. downscale -----------------------------------------------------------
# `sips -Z` fits the long edge; every ladder size is 16:9 so height follows.
for wh in $SIZES; do
  w=${wh%x*}
  o="$SRC/src_${wh}.png"
  [ -s "$o" ] && continue
  cp "$SRC4K" "$o"
  sips -Z "$w" "$o" >/dev/null
done

# Expected source md5s from the 2026-08-10 build. A mismatch means a different
# source photo or a different `sips`, and the encoded md5s below will not match
# either — which is a signal, not necessarily an error.
cat <<'EOF' > "$SRC/EXPECTED_SRC_MD5"
33542f00c693ecac96d98c673160904f  src_64x36.png
3fa384f12f7a4cb2f83a4e2f40c8ce49  src_256x144.png
53c517be4f1ec14bfe0e8d91fe602444  src_512x288.png
a2fe3818cd6dbee49f61f01e8a6f3452  src_1024x576.png
7aec246e283e6a84c8c7bf9bb86a4a46  src_2048x1152.png
877d6f2e60f533c780f6d5b3e605fa05  src_3840x2160.png
EOF

# --- 2. encode --------------------------------------------------------------
for wh in $SIZES; do
  for yuv in 420 444; do
    for d in 8 10; do
      o="$OUT/L${wh}_${yuv}_${d}b.avif"
      [ -s "$o" ] && continue
      nice -n 19 avifenc -s 6 -q 70 -y "$yuv" -d "$d" \
        --tilerowslog2 0 --tilecolslog2 0 -j 8 \
        --ignore-exif --ignore-xmp \
        "$SRC/src_${wh}.png" "$o" > "$LOG/enc_${wh}_${yuv}_${d}b.log" 2>&1
      printf '%s\t%s\n' "$(basename "$o")" "$(stat -f%z "$o")"
    done
  done
done

# --- 3. AVIF -> IVF, with per-size frame counts ------------------------------
# Frame counts are chosen so each cell runs long enough to fit `total = a +
# b*frames`: a 45 us frame needs tens of thousands of repeats, a 4K frame does
# not. These are the counts the committed benchmark records were taken with.
B="${AVIF_TO_IVF:-target/release/examples/avif_to_ivf}"
[ -x "$B" ] || { echo "build it first: cargo build --release --example avif_to_ivf" >&2; exit 1; }
frames_for() { case "$1" in
  64x36) echo 50000;; 256x144) echo 5000;; 512x288) echo 1000;;
  1024x576) echo 200;; 2048x1152) echo 50;; 3840x2160) echo 16;; esac; }

for wh in $SIZES; do
  n=$(frames_for "$wh")
  for yuv in 420 444; do
    for d in 8 10; do
      v="L${wh}_${yuv}_${d}b"
      [ -s "$IVF/$v.ivf" ] && continue
      nice -n 19 "$B" "$OUT/$v.avif" "$n" "$IVF/$v.ivf"
    done
  done
done

# --- 4. verify against dav1d BEFORE any timing ------------------------------
# Every perf number on this ladder was taken on a decode already proven
# bit-identical to dav1d 1.5.4. Do not skip this; a fast wrong decode is not a
# result. Expected md5s: benchmarks/size_sweep_vector_md5_2026-08-10.tsv.
echo
echo "NEXT: verify all 24 against dav1d before timing anything —"
echo "  scripts/perf/size_sweep.sh   (and see benchmarks/size_sweep_vector_md5_2026-08-10.tsv)"
echo "Then the t>1 latency/waste sweep: scripts/perf/size_sweep_t8.sh"
