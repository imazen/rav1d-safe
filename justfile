# rav1d-safe justfile

# Default recipe - show available commands
default:
    @just --list

# Build without ASM (pure safe Rust + SIMD)
build:
    cargo build --no-default-features --features "bitdepth_8,bitdepth_16" --release

# Build with ASM (original rav1d behavior)
build-asm:
    cargo build --features "asm,bitdepth_8,bitdepth_16" --release

# Build with partial ASM (ASM msac + loopfilter, safe SIMD everything else)
build-partial-asm:
    cargo build --no-default-features --features "bitdepth_8,bitdepth_16,partial_asm" --release

# Run all tests
test:
    cargo test --no-default-features --features "bitdepth_8,bitdepth_16" --release

# Download test vectors
download-vectors:
    bash scripts/download-test-vectors.sh

# Run integration tests (requires test vectors)
test-integration: download-vectors
    cargo test --no-default-features --features "bitdepth_8,bitdepth_16" --test integration_decode -- --ignored

# Run clippy lints
clippy:
    cargo clippy --no-default-features --features "bitdepth_8,bitdepth_16" --all-targets -- -D warnings

# Check code formatting
fmt-check:
    cargo fmt --all -- --check

# Format code
fmt:
    cargo fmt --all

# Run all checks (fmt, clippy, test)
check: fmt-check clippy test

# Cross-compile for aarch64
cross-aarch64:
    cargo check --target aarch64-unknown-linux-gnu --no-default-features --features "bitdepth_8,bitdepth_16"

# Generate documentation
doc:
    cargo doc --no-default-features --features "bitdepth_8,bitdepth_16" --no-deps --open

# Clean build artifacts
clean:
    cargo clean

# Benchmark via zenavif (requires zenavif in ../zenavif)
bench-zenavif:
    #!/usr/bin/env bash
    cd ../zenavif || exit 1
    touch src/lib.rs
    cargo build --release --example decode_avif
    echo "Running 20 decodes..."
    time for i in {1..20}; do \
        ./target/release/examples/decode_avif ../aom-decode/tests/test.avif /dev/null 2>/dev/null; \
    done

# Run managed API example
example-managed:
    cargo run --example managed_decode --no-default-features --features "bitdepth_8,bitdepth_16"

# Coverage report
coverage:
    cargo llvm-cov --no-default-features --features "bitdepth_8,bitdepth_16" --html
    @echo "Open target/llvm-cov/html/index.html"

# Run CI checks locally
ci: fmt-check clippy test test-integration

# Download all test vectors (Argon, dav1d, Fluster)
download-all-vectors:
    bash scripts/download-all-test-vectors.sh

# Run comprehensive test vector validation
test-all-vectors:
    bash scripts/test-all-vectors.sh

# Test against Argon conformance suite
test-argon:
    #!/bin/bash
    echo "Testing against Argon conformance suite..."
    for ivf in $(find test-vectors/argon/argon -name "*.ivf" | head -100); do
        cargo run --release --example managed_decode --no-default-features \
            --features "bitdepth_8,bitdepth_16" -- "$ivf" > /dev/null 2>&1 \
            && echo "✓ $(basename $ivf)" || echo "✗ $(basename $ivf)"
    done

# Run tests with AddressSanitizer (requires nightly)
test-asan:
    RUSTFLAGS="-Z sanitizer=address" cargo +nightly test --no-default-features --features "bitdepth_8,bitdepth_16" --target x86_64-unknown-linux-gnu

# Run tests with LeakSanitizer (requires nightly)
test-lsan:
    RUSTFLAGS="-Z sanitizer=leak" cargo +nightly test --no-default-features --features "bitdepth_8,bitdepth_16" --target x86_64-unknown-linux-gnu

# Benchmark decode (checked, default safety)
bench:
    cargo bench --bench decode --no-default-features --features "bitdepth_8,bitdepth_16"

# Benchmark decode (unchecked indexing)
bench-unchecked:
    cargo bench --bench decode --no-default-features --features "bitdepth_8,bitdepth_16,unchecked"

# Benchmark decode (hand-written asm)
bench-asm:
    cargo bench --bench decode --features "asm,bitdepth_8,bitdepth_16"

# Benchmark decode (partial asm: ASM msac + loopfilter, safe SIMD rest)
bench-partial-asm:
    cargo bench --bench decode --no-default-features --features "bitdepth_8,bitdepth_16,partial_asm"

# Run panic safety tests specifically
test-panic:
    cargo test --no-default-features --features "bitdepth_8,bitdepth_16" --test panic_safety_test --release

# Profile decode: all three modes (asm, safe checked, safe unchecked)
# Uses allintra 8bpc 352x288 (39 frames) and 10-bit film grain (10 frames)
profile iters="500":
    #!/usr/bin/env bash
    set -e
    IVF8="test-vectors/dav1d-test-data/8-bit/intra/av1-1-b8-02-allintra.ivf"
    IVF10="test-vectors/dav1d-test-data/10-bit/film_grain/av1-1-b10-23-film_grain-50.ivf"

    echo "=== ASM (hand-written assembly) ==="
    cargo build --release --features "asm,bitdepth_8,bitdepth_16" --example profile_decode 2>/dev/null
    ./target/release/examples/profile_decode "$IVF8" {{iters}} 2>&1
    ./target/release/examples/profile_decode "$IVF10" {{iters}} 2>&1
    echo ""

    echo "=== Safe-SIMD (checked, forbid(unsafe_code)) ==="
    cargo build --release --no-default-features --features "bitdepth_8,bitdepth_16" --example profile_decode 2>/dev/null
    ./target/release/examples/profile_decode "$IVF8" {{iters}} 2>&1
    ./target/release/examples/profile_decode "$IVF10" {{iters}} 2>&1
    echo ""

    echo "=== Safe-SIMD (unchecked bounds) ==="
    cargo build --release --no-default-features --features "bitdepth_8,bitdepth_16,unchecked" --example profile_decode 2>/dev/null
    ./target/release/examples/profile_decode "$IVF8" {{iters}} 2>&1
    ./target/release/examples/profile_decode "$IVF10" {{iters}} 2>&1
    echo ""

    echo "=== Partial ASM (ASM msac + loopfilter, safe SIMD rest) ==="
    cargo build --release --no-default-features --features "bitdepth_8,bitdepth_16,partial_asm" --example profile_decode 2>/dev/null
    ./target/release/examples/profile_decode "$IVF8" {{iters}} 2>&1
    ./target/release/examples/profile_decode "$IVF10" {{iters}} 2>&1

# Quick profile (100 iterations)
profile-quick:
    just profile 100

# Generate AVIF benchmark images (requires avifdec + avifenc from libavif)
generate-bench-avif avifenc="avifenc" avifdec="avifdec":
    #!/usr/bin/env bash
    set -e
    OUT="test-vectors/bench"
    mkdir -p "$OUT"
    SRC="/mnt/v/datasets/scraping/avif/google-native/8d716f849a1c4448.avif"
    if [ ! -f "$SRC" ]; then
        echo "Source 8K AVIF not found at $SRC"
        echo "Provide any 8K+ AVIF and update the SRC path in the justfile"
        exit 1
    fi
    echo "Decoding 8K source to PNG..."
    {{avifdec}} "$SRC" /tmp/rav1d_bench_8k.png
    echo "Resizing to 4K and 2K..."
    convert /tmp/rav1d_bench_8k.png -resize 3840x /tmp/rav1d_bench_4k.png
    convert /tmp/rav1d_bench_8k.png -resize 1920x /tmp/rav1d_bench_2k.png
    echo "Encoding as AVIF (YUV420, q60)..."
    {{avifenc}} -q 60 -s 6 -y 420 /tmp/rav1d_bench_2k.png "$OUT/photo_2k.avif"
    {{avifenc}} -q 60 -s 6 -y 420 /tmp/rav1d_bench_4k.png "$OUT/photo_4k.avif"
    {{avifenc}} -q 60 -s 6 -y 420 /tmp/rav1d_bench_8k.png "$OUT/photo_8k.avif"
    rm -f /tmp/rav1d_bench_*.png
    echo "Generated:"
    ls -lh "$OUT"/*.avif

# Benchmark AVIF decode (checked, default safety)
bench-avif:
    cargo bench --bench decode_avif --no-default-features --features "bitdepth_8,bitdepth_16"

# Benchmark AVIF decode (unchecked)
bench-avif-unchecked:
    cargo bench --bench decode_avif --no-default-features --features "bitdepth_8,bitdepth_16,unchecked"

# Benchmark AVIF decode (asm)
bench-avif-asm:
    cargo bench --bench decode_avif --features "asm,bitdepth_8,bitdepth_16"

# Benchmark AVIF decode (partial asm)
bench-avif-partial-asm:
    cargo bench --bench decode_avif --no-default-features --features "bitdepth_8,bitdepth_16,partial_asm"

# Run all benchmarks across all four modes for comparison
bench-compare:
    #!/usr/bin/env bash
    set -e
    echo "============================================"
    echo "=== Safe-SIMD (checked, forbid(unsafe))  ==="
    echo "============================================"
    cargo bench --bench decode_avif --no-default-features --features "bitdepth_8,bitdepth_16" 2>&1 | grep -E "photo_|Timer"
    cargo bench --bench decode --no-default-features --features "bitdepth_8,bitdepth_16" 2>&1 | grep -E "bit/|film_grain/|Timer"
    echo ""
    echo "============================================"
    echo "=== Safe-SIMD (unchecked bounds)         ==="
    echo "============================================"
    cargo bench --bench decode_avif --no-default-features --features "bitdepth_8,bitdepth_16,unchecked" 2>&1 | grep -E "photo_|Timer"
    cargo bench --bench decode --no-default-features --features "bitdepth_8,bitdepth_16,unchecked" 2>&1 | grep -E "bit/|film_grain/|Timer"
    echo ""
    echo "============================================"
    echo "=== Partial ASM (ASM msac + loopfilter)  ==="
    echo "============================================"
    cargo bench --bench decode_avif --no-default-features --features "bitdepth_8,bitdepth_16,partial_asm" 2>&1 | grep -E "photo_|Timer"
    cargo bench --bench decode --no-default-features --features "bitdepth_8,bitdepth_16,partial_asm" 2>&1 | grep -E "bit/|film_grain/|Timer"
    echo ""
    echo "============================================"
    echo "=== ASM (hand-written assembly)           ==="
    echo "============================================"
    cargo bench --bench decode_avif --features "asm,bitdepth_8,bitdepth_16" 2>&1 | grep -E "photo_|Timer"
    cargo bench --bench decode --features "asm,bitdepth_8,bitdepth_16" 2>&1 | grep -E "bit/|film_grain/|Timer"
