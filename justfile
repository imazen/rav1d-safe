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

# Quick profile (100 iterations)
profile-quick:
    just profile 100
