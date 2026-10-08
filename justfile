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

# Run all tests (unit + integration via nextest; doctests via cargo test —
# nextest does not run doctests). Process-per-test isolation keeps the
# token-permutation / CPU-mask / thread-count tests from racing.
test:
    cargo nextest run --no-default-features --features "bitdepth_8,bitdepth_16" --release
    cargo test --no-default-features --features "bitdepth_8,bitdepth_16" --release --doc

# Complete tracked/untracked dependency gate; serialize heavy decode tests.
test-all-modes:
    cargo nextest run --release --no-default-features --features "bitdepth_8,bitdepth_16" --test-threads 2
    cargo test --release --no-default-features --features "bitdepth_8,bitdepth_16" --doc
    cargo nextest run --release --no-default-features --features "bitdepth_8,bitdepth_16,untracked" --test-threads 2
    cargo test --release --no-default-features --features "bitdepth_8,bitdepth_16,untracked" --doc

# Reproduce startup observation and the committed 4K decoder stress fixture.
test-thread-start features="bitdepth_8,bitdepth_16" repetitions="100":
    cargo nextest run --release --no-default-features --features "{{features}}" --test thread_cleanup_test --test mt_stress --test-threads 1
    cargo nextest run --release --no-default-features --features "{{features}}" --test thread_cleanup_test -E 'test(=test_multi_threaded_cleanup)' --stress-count {{repetitions}} --test-threads 1

# Cast-range overflow and valid-alignment controls under Stacked Borrows.
test-cast-miri:
    cargo +nightly miri test -p rav1d-disjoint-mut --test cast_range_overflow

# Pass --before/--after, revision labels, and one or more --stream arguments.
bench-ab *args:
    python3 scripts/perf/decoder_bench/ab_bench.py {{args}}

# Require whole-clip dav1d MD5 identity before comparing benchmark binaries.
bench-md5 *args:
    python3 scripts/perf/decoder_bench/compare_md5.py {{args}}

# Sequential A/A and A/B runs for tracked and untracked target directories.
bench-paired-modes *args:
    python3 scripts/perf/decoder_bench/paired_modes.py {{args}}

# Verify warm-up and every timed pass on an explicit benchmark clip.
bench-frame-count binary stream threads="1" repetitions="3":
    RAV1D_THREADS={{threads}} RAV1D_REPS={{repetitions}} RAV1D_FRAME_DELAY=1 RAV1D_LEVEL=native RAV1D_INLOOP=all "{{binary}}" "{{stream}}" 1

# Download test vectors
download-vectors:
    bash scripts/download-test-vectors.sh

# Run integration tests (requires test vectors)
test-integration: download-vectors
    just test-integration-selection test-vectors

# Explicit caller-selected corpus; no fetch or implicit fallback for this path.
test-integration-selection vectors features="bitdepth_8,bitdepth_16":
    RAV1D_TEST_VECTORS="{{vectors}}" cargo nextest run --release --no-default-features --features "{{features}}" --test integration_decode --run-ignored all --test-threads 2

# Threading-race regression gates (zenavif#30 + the original overlap class):
# the ignored tile_threading_overlap tests (incl. multi_threaded_cdef_lpf_race,
# needs in-process parallel decode pressure) + the induced-worker-panic
# error-not-hang tests (private __test_induce_worker_panic feature).
test-threading-races:
    cargo nextest run --release --no-default-features --features "bitdepth_8,bitdepth_16" --test decode_concurrent_md5
    cargo test --release --no-default-features --features "bitdepth_8,bitdepth_16" --test tile_threading_overlap -- --ignored --test-threads 1
    cargo test --release --no-default-features --features "bitdepth_8,bitdepth_16,__test_induce_worker_panic" --test worker_panic_recovery -- --ignored --test-threads 1

# Run clippy lints
clippy:
    cargo clippy --release --no-default-features --features "bitdepth_8,bitdepth_16" --all-targets -- -D warnings

# The untracked mode has a different set of unit tests and helpers.
clippy-untracked:
    cargo clippy --release --no-default-features --features "bitdepth_8,bitdepth_16,untracked" --all-targets -- -D warnings

# Match the library-only CI lint gate; release-only integration tests are separate.
clippy-lib:
    cargo clippy --no-default-features --features "bitdepth_8,bitdepth_16" -- -D warnings

# Native ARM regression for concurrent reference-frame reconstruction.
test-mc-reference:
    cargo nextest run --cargo-profile release-thin --lib -E 'test(interpolation_reads_leave_unrelated_reconstruction_rows_available)'

test-conformance-runner:
    python3 tools/test_conformance_runner.py

# Native example guards check real fallback tier selection in workers.
test-conformance-token-tiers:
    cargo nextest run --release --no-default-features --features "bitdepth_8,bitdepth_16" --example decode_md5 --test-threads 1
    cargo build --release --no-default-features --features "bitdepth_8,bitdepth_16" --example decode_md5
    just test-decode-md5-limit "${CARGO_TARGET_DIR:-target}/release/examples/decode_md5"

# Supplement a previously recorded native-tier matrix with enforced scalar legs.
conformance-scalar-modes tracked untracked:
    #!/usr/bin/env bash
    set -euo pipefail
    for binary in "{{tracked}}" "{{untracked}}"; do
        for threads in 1 2 4 8; do
            bash scripts/conformance_test.sh --binary "$binary" --threads "$threads" --delay 0 --level scalar --expected 803 --stop-on-fail
        done
    done

conformance binary threads="1" delay="0":
    bash scripts/conformance_test.sh --binary "{{binary}}" --threads {{threads}} --delay {{delay}} --expected 803

# Reproduce one sidecar invocation with its original flags and visible output.
decode-vector binary *args:
    "{{binary}}" {{args}}

# A valid committed frame followed by malformed input tests actual stop semantics.
test-decode-md5-limit binary:
    python3 tools/test_decode_md5_limit.py "{{binary}}"

# A valid frame followed by malformed input must never yield benchmark results.
test-benchmark-errors decode_md5 profile_ivf:
    python3 tools/test_decode_md5_limit.py "{{decode_md5}}" --profile-binary "{{profile_ivf}}"

# Matched performance binaries; use fresh target directories to preserve references.
build-bench-modes tracked untracked:
    CARGO_TARGET_DIR="{{tracked}}" cargo build --release --no-default-features --features "bitdepth_8,bitdepth_16" --example profile_ivf --example decode_md5
    CARGO_TARGET_DIR="{{untracked}}" cargo build --release --no-default-features --features "bitdepth_8,bitdepth_16,untracked" --example profile_ivf --example decode_md5

# Complete sidecar oracle at every runtime tier and 1/2/4/8 workers in both modes.
build-conformance-modes tracked untracked:
    CARGO_TARGET_DIR="{{tracked}}" cargo build --release --no-default-features --features "bitdepth_8,bitdepth_16" --example decode_md5
    CARGO_TARGET_DIR="{{untracked}}" cargo build --release --no-default-features --features "bitdepth_8,bitdepth_16,untracked" --example decode_md5

conformance-all-modes tracked untracked:
    #!/usr/bin/env bash
    set -euo pipefail
    for binary in "{{tracked}}" "{{untracked}}"; do
        for threads in 1 2 4 8; do
            bash scripts/conformance_test.sh --binary "$binary" --threads "$threads" --delay 0 --level all --expected 803 --stop-on-fail
        done
    done

# Dependency and CI repair gate, with shared-machine-friendly test parallelism.
check-lead-ci: clippy cross-aarch64
    cargo check --target wasm32-unknown-unknown --no-default-features --features "bitdepth_8,bitdepth_16"
    cargo check --no-default-features --features "bitdepth_8,bitdepth_16,c-ffi"
    cargo nextest run --cargo-profile release-thin --no-default-features --features "bitdepth_8,bitdepth_16" --lib --test-threads 2
    cargo test --profile release-thin --no-default-features --features "bitdepth_8,bitdepth_16" --lib -- --test-threads 8

# Validate workflow expressions and runner/action schemas (requires actionlint).
lint-ci-workflow checker="actionlint":
    "{{checker}}" -shellcheck="" .github/workflows/ci.yml

# Read the intended repository explicitly, including unfinished and failed jobs.
ci-status run:
    gh run view {{run}} --repo imazen/rav1d-safe --json status,conclusion,jobs --jq '{status,conclusion,counts:(.jobs|group_by(.conclusion)|map({conclusion:.[0].conclusion,count:length})),unfinished:[.jobs[]|select(.status!="completed")|{name,status}],failures:[.jobs[]|select(.conclusion=="failure")|{name,databaseId}]}'

# Check code formatting
fmt-check:
    cargo fmt --all -- --check

# Format code + regenerate the public-API surface snapshots (docs/public-api/).
# The snapshot runner lives in the workspace-excluded apidoc/ package, so it
# is never built or run by plain `cargo test` or any CI job.
fmt:
    cargo fmt --all
    cargo test --manifest-path apidoc/Cargo.toml

# Regenerate the public-API surface snapshots only
api-doc:
    cargo test --manifest-path apidoc/Cargo.toml

# Verify the committed snapshots are current
api-doc-check:
    ZEN_API_DOC=check cargo test --manifest-path apidoc/Cargo.toml

# Run all checks (fmt, clippy, test)
check: fmt-check clippy test

# Cross-compile for aarch64
cross-aarch64:
    cargo check --target aarch64-unknown-linux-gnu --no-default-features --features "bitdepth_8,bitdepth_16"

# Cross-compile and test on aarch64 via Docker/QEMU (lib tests only)
test-aarch64:
    cross test --target aarch64-unknown-linux-gnu --no-default-features \
        --features "bitdepth_8,bitdepth_16" --release --lib -- --test-threads=1

# Build and test for WASM with simd128 (lib tests only)
test-wasm:
    cargo test --target wasm32-wasip1 --no-default-features \
        --features "bitdepth_8,bitdepth_16" --lib

# Check WASM compilation only (faster)
check-wasm:
    cargo check --target wasm32-wasip1 --no-default-features \
        --features "bitdepth_8,bitdepth_16"

# Build and test for 32-bit x86 (Linux). i686 binaries run natively on the
# x86_64 host, so nextest hosts them directly.
test-i686:
    cargo nextest run --target i686-unknown-linux-gnu --no-default-features \
        --features "bitdepth_8,bitdepth_16" --release --lib

# The Linux 32-bit container gate complements the native nextest matrix.
test-i686-cross:
    cross test --target i686-unknown-linux-gnu --no-default-features --features "bitdepth_8,bitdepth_16" --release --lib --test decode_md5_committed --test safe_simd_crashes --test fuzz_regression -- --test-threads 1
    cross test --target i686-unknown-linux-gnu --no-default-features --features "bitdepth_8,bitdepth_16" --test decode_md5_committed --test safe_simd_crashes --test fuzz_regression -- --test-threads 1

# Check 32-bit compilation only
check-i686:
    cargo check --target i686-unknown-linux-gnu --no-default-features \
        --features "bitdepth_8,bitdepth_16"

# Run token permutation tests (exercises all CPU tiers; nextest isolates each
# test in its own process, so no --test-threads=1 race-avoidance needed)
test-permutations:
    cargo nextest run --no-default-features --features "bitdepth_8,bitdepth_16" \
        --release -E 'test(token_permutation)'

# E2E decode permutations: smoke test (1 vector x all tiers, ~1s)
test-permutations-smoke:
    cargo nextest run --no-default-features --features "bitdepth_8,bitdepth_16" \
        --release --test decode_permutations -E 'test(test_permutations_smoke)' \
        --no-capture

# E2E decode permutations: full corpus x all tiers (~20 min)
test-permutations-full:
    cargo nextest run --no-default-features --features "bitdepth_8,bitdepth_16" \
        --release --test decode_permutations --no-capture

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

# Coverage report (via nextest)
coverage:
    cargo llvm-cov nextest --no-default-features --features "bitdepth_8,bitdepth_16" --html
    @echo "Open target/llvm-cov/html/index.html"

# Run CI checks locally (incl. cross-arch checks that CI runs — catches
# x86_64-only code that breaks wasm32/aarch64 before it reaches CI)
ci: fmt-check clippy cross-aarch64 check-wasm test test-integration

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

# Benchmark decode (no overlap tracking)
bench-untracked:
    cargo bench --bench decode --no-default-features --features "bitdepth_8,bitdepth_16,untracked"

# Benchmark decode (hand-written asm)
bench-asm:
    cargo bench --bench decode --features "asm,bitdepth_8,bitdepth_16"

# Benchmark decode (partial asm: ASM msac + loopfilter, safe SIMD rest)
bench-partial-asm:
    cargo bench --bench decode --no-default-features --features "bitdepth_8,bitdepth_16,partial_asm"

# PGO pipeline: instrumented build -> train on 8/10/12-bit vectors + 4K AVIF ->
# merged profile -> -Cprofile-use build -> interleaved A/B. Nothing is
# committed: artifacts land in target/pgo-{instr,use,data}/. Needs the dav1d
# test vectors (just download-vectors); llvm-profdata is resolved from PATH
# or the active rustup toolchain.
pgo:
    bash scripts/perf/bench_pgo.sh

# Same, with -Ctarget-cpu=native on both instrumented and use builds
pgo-native:
    bash scripts/perf/bench_pgo.sh --native

# Run panic safety tests specifically
test-panic:
    cargo nextest run --no-default-features --features "bitdepth_8,bitdepth_16" --test panic_safety_test --release

# Profile decode: all four modes (asm, partial asm, safe tracked, safe untracked)
# Uses allintra 8bpc IVF (39 frames) + real photos (4K + 8K AVIF)
profile iters="500" avif_iters="20":
    #!/usr/bin/env bash
    set -e
    IVF8="test-vectors/dav1d-test-data/8-bit/intra/av1-1-b8-02-allintra.ivf"
    AVIF4K="test-vectors/bench/photo_4k.avif"
    AVIF8K="test-vectors/bench/photo_8k.avif"

    echo "=== ASM (hand-written assembly) ==="
    cargo build --release --features "asm,bitdepth_8,bitdepth_16" --example profile_decode --example profile_avif 2>/dev/null
    ./target/release/examples/profile_decode "$IVF8" {{iters}} 2>&1
    ./target/release/examples/profile_avif "$AVIF4K" {{avif_iters}} 2>&1
    ./target/release/examples/profile_avif "$AVIF8K" {{avif_iters}} 2>&1
    echo ""

    echo "=== Safe-SIMD (checked, forbid(unsafe_code)) ==="
    cargo build --release --no-default-features --features "bitdepth_8,bitdepth_16" --example profile_decode --example profile_avif 2>/dev/null
    ./target/release/examples/profile_decode "$IVF8" {{iters}} 2>&1
    ./target/release/examples/profile_avif "$AVIF4K" {{avif_iters}} 2>&1
    ./target/release/examples/profile_avif "$AVIF8K" {{avif_iters}} 2>&1
    echo ""

    echo "=== Safe-SIMD (untracked) ==="
    cargo build --release --no-default-features --features "bitdepth_8,bitdepth_16,untracked" --example profile_decode --example profile_avif 2>/dev/null
    ./target/release/examples/profile_decode "$IVF8" {{iters}} 2>&1
    ./target/release/examples/profile_avif "$AVIF4K" {{avif_iters}} 2>&1
    ./target/release/examples/profile_avif "$AVIF8K" {{avif_iters}} 2>&1
    echo ""

    echo "=== Partial ASM (ASM msac + loopfilter, safe SIMD rest) ==="
    cargo build --release --no-default-features --features "bitdepth_8,bitdepth_16,partial_asm" --example profile_decode --example profile_avif 2>/dev/null
    ./target/release/examples/profile_decode "$IVF8" {{iters}} 2>&1
    ./target/release/examples/profile_avif "$AVIF4K" {{avif_iters}} 2>&1
    ./target/release/examples/profile_avif "$AVIF8K" {{avif_iters}} 2>&1

# Quick profile (100 iterations IVF, 5 iterations AVIF)
profile-quick:
    just profile 100 5

# Generate AVIF benchmark images (requires avifdec + avifenc from libavif)
generate-bench-avif avifenc="avifenc" avifdec="avifdec":
    #!/usr/bin/env bash
    set -e
    OUT="test-vectors/bench"
    mkdir -p "$OUT"
    SRC="${BENCH_AVIF_SOURCE:-/mnt/v/datasets/scraping/avif/google-native/8d716f849a1c4448.avif}"
    if [ ! -f "$SRC" ]; then
        echo "Source 8K AVIF not found at $SRC"
        echo "Set BENCH_AVIF_SOURCE to an 8K+ AVIF file path"
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

# Benchmark AVIF decode (no overlap tracking)
bench-avif-untracked:
    cargo bench --bench decode_avif --no-default-features --features "bitdepth_8,bitdepth_16,untracked"

# Benchmark AVIF decode (asm)
bench-avif-asm:
    cargo bench --bench decode_avif --features "asm,bitdepth_8,bitdepth_16"

# Benchmark AVIF decode (partial asm)
bench-avif-partial-asm:
    cargo bench --bench decode_avif --no-default-features --features "bitdepth_8,bitdepth_16,partial_asm"

# Export tango baseline (run before making changes)
tango-export features="bitdepth_8,bitdepth_16":
    cargo export target/tango -- bench --bench=tango_decode --no-default-features --features "{{features}}"

# Compare against tango baseline (run after making changes)
tango-compare features="bitdepth_8,bitdepth_16":
    cargo bench --bench=tango_decode --no-default-features --features "{{features}}" -- compare target/tango/tango_decode

# Full tango A/B: export baseline from a git ref, then compare HEAD
tango-ab ref="HEAD~1" features="bitdepth_8,bitdepth_16":
    #!/usr/bin/env bash
    set -e
    echo "=== Exporting baseline from {{ref}} ==="
    git stash --include-untracked -q 2>/dev/null || true
    git checkout "{{ref}}" -q
    cargo export target/tango -- bench --bench=tango_decode --no-default-features --features "{{features}}" 2>&1
    git checkout - -q
    git stash pop -q 2>/dev/null || true
    echo ""
    echo "=== Comparing HEAD against {{ref}} ==="
    cargo bench --bench=tango_decode --no-default-features --features "{{features}}" -- compare target/tango/tango_decode 2>&1

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
    echo "=== Safe-SIMD (untracked)         ==="
    echo "============================================"
    cargo bench --bench decode_avif --no-default-features --features "bitdepth_8,bitdepth_16,untracked" 2>&1 | grep -E "photo_|Timer"
    cargo bench --bench decode --no-default-features --features "bitdepth_8,bitdepth_16,untracked" 2>&1 | grep -E "bit/|film_grain/|Timer"
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

# #526: corpus reference MD5s with grain enabled at 1/2/4/8 threads, dev guards on.
test-filmgrain:
    CARGO_BUILD_JOBS=4 nice -n 19 cargo nextest run --test filmgrain_threads --test-threads 1

test-filmgrain-rows:
    CARGO_BUILD_JOBS=4 nice -n 19 cargo nextest run --lib -E 'test(filmgrain_rows)' --test-threads 1

# Native ARM interleaved decoder tiers; requires the explicit IVF fixtures.
arm-tiers-macos:
    mkdir -p "$HOME/tmp"
    CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4 OMP_NUM_THREADS=4 TMPDIR="$HOME/tmp" nice -n 19 cargo bench --locked -p rav1d-safe --bench tier_isolation -- --format=llm > "$HOME/tmp/rav1d-arm-tiers.log" 2>&1


# #526: actual frame contexts plus a 32-tile stream, and concurrent decoders.
test-filmgrain-concurrency:
    CARGO_BUILD_JOBS=2 nice -n 19 cargo nextest run --lib --test filmgrain_threads -E 'binary(filmgrain_threads) | test(parallel_frame_tile_contexts)' --test-threads 1 --success-output immediate
    CARGO_BUILD_JOBS=2 nice -n 19 cargo nextest run --features untracked --lib --test filmgrain_threads -E 'binary(filmgrain_threads) | test(parallel_frame_tile_contexts)' --test-threads 1 --success-output immediate

# Root API and strict/lenient conformance regression checks (issues 525, 522, 523).
test-strictness:
    cargo test --test strictness
    cargo test --doc Settings

# Replay one differential artifact without a sweep; requires system libdav1d.
repro-differential artifact:
    cargo +nightly fuzz run differential_dav1d --features differential {{artifact}} -- -runs=1
