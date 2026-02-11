#!/bin/bash
# Conformance test runner for rav1d-safe
# Runs all dav1d-test-data vectors against decode_md5 and verifies MD5 checksums.
#
# Usage:
#   ./scripts/conformance_test.sh [--level LEVEL] [--category PATTERN] [--stop-on-fail]
#
# Levels:
#   default   - Default build (AVX2 SIMD dispatch)
#   scalar    - force_scalar feature (no SIMD)
#   v2        - target-cpu=x86-64-v2 (SSE4.2, no AVX)
#   v4        - target-cpu=x86-64-v4 (AVX-512)
#   all       - Run all levels sequentially
#
# Examples:
#   ./scripts/conformance_test.sh                          # default level, all tests
#   ./scripts/conformance_test.sh --level scalar           # scalar only
#   ./scripts/conformance_test.sh --level all              # all levels
#   ./scripts/conformance_test.sh --category 8-bit/data    # only 8-bit/data tests
#   ./scripts/conformance_test.sh --stop-on-fail           # stop at first failure

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VECTORS_TSV="$PROJECT_DIR/scripts/extract_test_vectors.py"

LEVEL="default"
CATEGORY=""
STOP_ON_FAIL=false
SKIP_BUILD=false
SKIP_OSSFUZZ=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --level) LEVEL="$2"; shift 2 ;;
        --category) CATEGORY="$2"; shift 2 ;;
        --stop-on-fail) STOP_ON_FAIL=true; shift ;;
        --skip-build) SKIP_BUILD=true; shift ;;
        --include-ossfuzz) SKIP_OSSFUZZ=false; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Build decode_md5 for a given level
build_level() {
    local level="$1"
    local target_dir

    case "$level" in
        default)
            echo "Building decode_md5 (default/AVX2)..."
            cargo build --no-default-features --features "bitdepth_8,bitdepth_16" \
                --example decode_md5 --release 2>&1 | tail -1
            echo "target/release/examples/decode_md5"
            ;;
        scalar)
            echo "Building decode_md5 (force_scalar)..."
            cargo build --no-default-features --features "bitdepth_8,bitdepth_16,force_scalar" \
                --example decode_md5 --release 2>&1 | tail -1
            # force_scalar changes features, so same binary path
            echo "target/release/examples/decode_md5"
            ;;
        v2)
            echo "Building decode_md5 (target-cpu=x86-64-v2)..."
            RUSTFLAGS="-C target-cpu=x86-64-v2" \
                cargo build --no-default-features --features "bitdepth_8,bitdepth_16" \
                --example decode_md5 --release --target-dir target/v2 2>&1 | tail -1
            echo "target/v2/release/examples/decode_md5"
            ;;
        v4)
            echo "Building decode_md5 (target-cpu=x86-64-v4)..."
            RUSTFLAGS="-C target-cpu=x86-64-v4" \
                cargo build --no-default-features --features "bitdepth_8,bitdepth_16" \
                --example decode_md5 --release --target-dir target/v4 2>&1 | tail -1
            echo "target/v4/release/examples/decode_md5"
            ;;
        *)
            echo "Unknown level: $level"
            exit 1
            ;;
    esac
}

# Run conformance tests for a given level
run_tests() {
    local level="$1"
    local binary="$2"
    local pass=0
    local fail=0
    local error=0
    local skip=0
    local total=0
    local failures=""

    echo ""
    echo "=== Conformance tests: $level ==="
    echo "Binary: $binary"
    echo ""

    while IFS=$'\t' read -r bitdepth category test_name file_path expected_md5 filmgrain; do
        # Skip header
        [[ "$bitdepth" == "bitdepth" ]] && continue

        # Skip oss-fuzz if requested (they need a fuzzer harness, not decode_md5)
        if $SKIP_OSSFUZZ && [[ "$bitdepth" == "oss-fuzz" ]]; then
            skip=$((skip + 1))
            continue
        fi

        # Filter by category if specified
        if [[ -n "$CATEGORY" ]] && [[ "$bitdepth/$category" != *"$CATEGORY"* ]]; then
            skip=$((skip + 1))
            continue
        fi

        total=$((total + 1))

        # Check file exists
        if [[ ! -f "$file_path" ]]; then
            echo "MISSING: $file_path"
            error=$((error + 1))
            continue
        fi

        # Build args
        local args=("-q" "$file_path" "$expected_md5")
        if [[ "$filmgrain" == "1" ]]; then
            args=("-q" "--filmgrain" "$file_path" "$expected_md5")
        fi

        # Run decode_md5
        if output=$("$binary" "${args[@]}" 2>&1); then
            pass=$((pass + 1))
        else
            fail=$((fail + 1))
            failures="${failures}  FAIL: ${bitdepth}/${category}/${test_name} (${file_path})\n"
            if [[ -n "$output" ]]; then
                failures="${failures}        ${output}\n"
            fi
            if $STOP_ON_FAIL; then
                echo "FAIL: ${bitdepth}/${category}/${test_name}"
                echo "$output"
                echo ""
                echo "Stopped on first failure."
                echo "Results so far: $pass pass, $fail fail, $error error out of $total tested"
                return 1
            fi
        fi

        # Progress indicator every 50 tests
        if (( total % 50 == 0 )); then
            echo "  ... $total tested ($pass pass, $fail fail)"
        fi
    done < <(python3 "$VECTORS_TSV")

    echo ""
    echo "--- Results for $level ---"
    echo "Total:   $total"
    echo "Passed:  $pass"
    echo "Failed:  $fail"
    echo "Errors:  $error"
    echo "Skipped: $skip"

    if [[ $fail -gt 0 ]]; then
        echo ""
        echo "Failures:"
        echo -e "$failures"
        return 1
    elif [[ $error -gt 0 ]]; then
        echo ""
        echo "Had $error missing files"
        return 1
    else
        echo "ALL PASSED"
        return 0
    fi
}

# Main
cd "$PROJECT_DIR"

if [[ "$LEVEL" == "all" ]]; then
    levels=(default scalar v2 v4)
else
    levels=("$LEVEL")
fi

overall_pass=true

for level in "${levels[@]}"; do
    if ! $SKIP_BUILD; then
        binary=$(build_level "$level" | tail -1)
    else
        case "$level" in
            default) binary="target/release/examples/decode_md5" ;;
            scalar)  binary="target/release/examples/decode_md5" ;;
            v2)      binary="target/v2/release/examples/decode_md5" ;;
            v4)      binary="target/v4/release/examples/decode_md5" ;;
        esac
    fi

    if ! run_tests "$level" "$binary"; then
        overall_pass=false
        if $STOP_ON_FAIL; then
            exit 1
        fi
    fi
done

if $overall_pass; then
    echo ""
    echo "=== ALL CONFORMANCE TESTS PASSED ==="
    exit 0
else
    echo ""
    echo "=== SOME CONFORMANCE TESTS FAILED ==="
    exit 1
fi
