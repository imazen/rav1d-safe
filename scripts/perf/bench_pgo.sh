#!/usr/bin/env bash
# PGO pipeline: instrumented build -> representative profile runs -> merge ->
# profile-use build -> interleaved A/B against the non-PGO binary.
#
# Usage: bench_pgo.sh [--native]
#   --native   add -Ctarget-cpu=native to BOTH instrumented and use builds
#              (flags must match or profile hashes won't resolve)
#
# Outputs land in target/pgo-{instr,use}[-native]/ and target/pgo-data[-native]/.
set -eu

NATIVE_FLAG=""
TAG=""
if [ "${1:-}" = "--native" ]; then
    NATIVE_FLAG="-Ctarget-cpu=native"
    TAG="-native"
fi

# llvm-profdata may not be on PATH; every rustup toolchain ships one.
LLVM_PROFDATA="$(command -v llvm-profdata || true)"
if [ -z "$LLVM_PROFDATA" ]; then
    LLVM_PROFDATA="$(rustc --print sysroot)/lib/rustlib/$(rustc -vV | sed -n 's/^host: //p')/bin/llvm-profdata"
fi
[ -x "$LLVM_PROFDATA" ] || { echo "error: llvm-profdata not found (install llvm or use a rustup toolchain)"; exit 1; }

FEATURES="bitdepth_8,bitdepth_16"
INSTR_DIR="target/pgo-instr$TAG"
USE_DIR="target/pgo-use$TAG"
DATA_DIR="target/pgo-data$TAG"
D="test-vectors/dav1d-test-data"

# Representative profile workload: every bit depth + the intra-heavy 4K AVIF.
# The degenerate 125-byte 00000000.ivf is excluded — it skews the profile
# toward cold-start paths.
IVF_INPUTS=(
    "$D/8-bit/data/00000001.ivf"
    "$D/10-bit/data/00000671.ivf"
    "$D/12-bit/data/00000686.ivf"
)
AVIF_INPUT="test-vectors/bench/photo_4k.avif"

echo "=== 0/4 non-PGO baseline (target/release) ==="
cargo build --release --no-default-features --features "$FEATURES" \
    --example profile_ivf --example profile_avif

echo "=== 1/4 instrumented build ($INSTR_DIR) ==="
CARGO_TARGET_DIR="$INSTR_DIR" \
RUSTFLAGS="-Cprofile-generate=$PWD/$DATA_DIR $NATIVE_FLAG" \
    cargo build --release --no-default-features --features "$FEATURES" \
    --example profile_ivf --example profile_avif

echo "=== 2/4 collecting profiles ==="
rm -rf "$DATA_DIR" && mkdir -p "$DATA_DIR"
for f in "${IVF_INPUTS[@]}"; do
    "$INSTR_DIR/release/examples/profile_ivf" "$f" 30 >/dev/null
done
"$INSTR_DIR/release/examples/profile_avif" "$AVIF_INPUT" 5 >/dev/null
"$LLVM_PROFDATA" merge -o "$DATA_DIR/merged.profdata" "$DATA_DIR"/*.profraw

echo "=== 3/4 profile-use build ($USE_DIR) ==="
CARGO_TARGET_DIR="$USE_DIR" \
RUSTFLAGS="-Cprofile-use=$PWD/$DATA_DIR/merged.profdata $NATIVE_FLAG" \
    cargo build --release --no-default-features --features "$FEATURES" \
    --example profile_ivf --example profile_avif

echo "=== 4/4 A/B: non-PGO vs PGO ==="
echo "--- 10-bit IVF ---"
for r in 1 2 3; do
    target/release/examples/profile_ivf "$D/10-bit/data/00000671.ivf" 50 \
        2>&1 | grep -oE "[0-9]+\.[0-9]+ ms/frame" | tail -1
    "$USE_DIR/release/examples/profile_ivf" "$D/10-bit/data/00000671.ivf" 50 \
        2>&1 | grep -oE "[0-9]+\.[0-9]+ ms/frame" | tail -1
done
echo "--- 4K AVIF ---"
for r in 1 2 3; do
    target/release/examples/profile_avif "$AVIF_INPUT" 15 \
        2>&1 | grep -oE "[0-9]+\.[0-9]+ms" | tail -1
    "$USE_DIR/release/examples/profile_avif" "$AVIF_INPUT" 15 \
        2>&1 | grep -oE "[0-9]+\.[0-9]+ms" | tail -1
done
