#!/usr/bin/env bash
# Run dav1d sidecar MD5s through decode_md5, preserving all vector flags.
# CPU levels are selected at runtime in one binary. --binary uses a prebuilt
# tracked/untracked/ASM binary; --threads and --delay exercise its threading.
# Example: --binary target/release/examples/decode_md5 --threads 4 --delay 0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LEVEL=native
CATEGORY=""
BINARY=""
MANIFEST=""
THREADS=1
DELAY=0
EXPECTED=""
STOP_ON_FAIL=false
SKIP_BUILD=false
INCLUDE_OSSFUZZ=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --level) LEVEL="$2"; shift 2 ;;
        --category) CATEGORY="$2"; shift 2 ;;
        --binary) BINARY="$2"; SKIP_BUILD=true; shift 2 ;;
        --manifest) MANIFEST="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --delay) DELAY="$2"; shift 2 ;;
        --expected) EXPECTED="$2"; shift 2 ;;
        --stop-on-fail) STOP_ON_FAIL=true; shift ;;
        --skip-build) SKIP_BUILD=true; shift ;;
        --include-ossfuzz) INCLUDE_OSSFUZZ=true; shift ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
[[ "$THREADS" =~ ^[0-9]+$ && "$DELAY" =~ ^[0-9]+$ ]] || exit 2
[[ -z "$EXPECTED" || "$EXPECTED" =~ ^[1-9][0-9]*$ ]] || exit 2

cd "$PROJECT_DIR"
if ! $SKIP_BUILD; then
    cargo build --no-default-features --features bitdepth_8,bitdepth_16 \
        --example decode_md5 --release
fi
BINARY="${BINARY:-target/release/examples/decode_md5}"
[[ -x "$BINARY" ]] || { echo "Missing executable: $BINARY" >&2; exit 2; }
if [[ -z "$MANIFEST" ]]; then
    # Keep the extraction for provenance. The extractor's failure must reach
    # the caller; process substitution previously allowed an empty green run.
    mkdir -p "$HOME/tmp"
    MANIFEST=$(mktemp "$HOME/tmp/rav1d-vectors.XXXXXX.tsv")
    python3 "$SCRIPT_DIR/extract_test_vectors.py" > "$MANIFEST"
fi
[[ -f "$MANIFEST" ]] || { echo "Missing manifest: $MANIFEST" >&2; exit 2; }
if [[ "$LEVEL" == all ]]; then
    case "$(uname -m)" in
        aarch64|arm64) levels=(scalar neon native) ;;
        x86_64|amd64) levels=(scalar v2 v3 v4 native) ;;
        *) levels=(scalar native) ;;
    esac
else
    levels=("${LEVEL/default/native}")
fi

run_tests() {
    local level="$1" pass=0 fail=0 error=0 skip=0 total=0 output
    echo "Conformance: level=$level threads=$THREADS delay=$DELAY binary=$BINARY manifest=$MANIFEST"
    while IFS=$'\t' read -r bitdepth category test_name file_path expected_md5 filmgrain extra_args; do
        [[ "$bitdepth" == bitdepth || -z "$bitdepth" ]] && continue
        if [[ "$bitdepth" == oss-fuzz ]] && ! $INCLUDE_OSSFUZZ; then
            skip=$((skip + 1)); continue
        fi
        if [[ -n "$CATEGORY" && "$bitdepth/$category" != *"$CATEGORY"* ]]; then
            skip=$((skip + 1)); continue
        fi
        total=$((total + 1))
        if [[ ! -f "$file_path" ]]; then
            echo "MISSING: $bitdepth/$category/$test_name $file_path"
            error=$((error + 1)); continue
        fi
        local args=(-q --level "$level" --threads "$THREADS" --delay "$DELAY")
        [[ "$filmgrain" == 1 ]] && args+=(--filmgrain)
        if [[ -n "$extra_args" ]]; then
            local extra=()
            read -r -a extra <<< "$extra_args"
            args+=("${extra[@]}")
        fi
        args+=("$file_path" "$expected_md5")
        if output=$(timeout 120 "$BINARY" "${args[@]}" 2>&1); then
            pass=$((pass + 1))
        else
            fail=$((fail + 1))
            echo "FAIL: $bitdepth/$category/$test_name"
            echo "$output"
            if $STOP_ON_FAIL; then
                echo "Stopped on first failure: pass=$pass fail=$fail errors=$error total=$total"
                return 1
            fi
        fi
        if ((total % 50 == 0)); then
            echo "Progress: total=$total pass=$pass fail=$fail errors=$error"
        fi
    done < "$MANIFEST"
    echo "SUMMARY level=$level threads=$THREADS delay=$DELAY total=$total pass=$pass fail=$fail errors=$error excluded=$skip"
    if ((total == 0)); then
        echo "No vectors selected" >&2; return 1
    fi
    if [[ -n "$EXPECTED" && "$total" != "$EXPECTED" ]]; then
        echo "Expected $EXPECTED vectors, selected $total" >&2; return 1
    fi
    ((fail == 0 && error == 0))
}

overall=0
for level in "${levels[@]}"; do
    if ! run_tests "$level"; then overall=1; fi
    if ((overall != 0)) && $STOP_ON_FAIL; then break; fi
done
exit "$overall"
