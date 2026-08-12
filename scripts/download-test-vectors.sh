#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VECTORS_DIR="${CARGO_TARGET_DIR:-$PROJECT_ROOT/target}/test-vectors"

mkdir -p "$VECTORS_DIR"

echo "Downloading AV1 test vectors to: $VECTORS_DIR"

# AOM test data from Google Cloud Storage
AOM_BASE="https://storage.googleapis.com/aom-test-data"

# Small conformance vectors for basic testing
declare -A VECTORS=(
    # Small test file
    ["test-25fps.ivf"]="$AOM_BASE/test-25fps.ivf"
)

download_file() {
    local name="$1"
    local url="$2"
    local dest="$VECTORS_DIR/$name"

    if [ -f "$dest" ]; then
        echo "  ✓ $name (cached)"
        return 0
    fi

    echo "  ⬇ Downloading $name..."
    if command -v curl &> /dev/null; then
        curl -fsSL "$url" -o "$dest.tmp" && mv "$dest.tmp" "$dest"
    elif command -v wget &> /dev/null; then
        wget -q "$url" -O "$dest.tmp" && mv "$dest.tmp" "$dest"
    else
        echo "  ✗ Error: neither curl nor wget found"
        return 1
    fi

    if [ -f "$dest" ]; then
        local size=$(du -h "$dest" | cut -f1)
        echo "  ✓ $name ($size)"
    else
        echo "  ✗ Failed to download $name"
        return 1
    fi
}

# Download each vector
for name in "${!VECTORS[@]}"; do
    download_file "$name" "${VECTORS[$name]}" || true
done

# Try to download from dav1d test data repo
echo ""
echo "Attempting to clone dav1d test data repository..."
# NOT under $VECTORS_DIR. The conformance harness resolves this corpus as
# `$CARGO_MANIFEST_DIR/test-vectors/dav1d-test-data` (tests/test_vectors.rs),
# while $VECTORS_DIR is `${CARGO_TARGET_DIR:-target}/test-vectors`. Those have
# never been the same directory, so every clone this script made was invisible
# to the tests, which then re-cloned for themselves. Populate the path the
# harness actually reads.
DAV1D_DATA_DIR="$PROJECT_ROOT/test-vectors/dav1d-test-data"
mkdir -p "$PROJECT_ROOT/test-vectors"

# The 766-vector conformance corpus. This is NOT optional: it is the only
# thing that checks decode output against dav1d's reference MD5s.
#
# 2026-08-12: this block used to `2>/dev/null` the clone and, on failure, print
# "not available (optional)" and exit 0. Combined with `continue-on-error` in
# ci.yml, that meant a failed fetch produced a GREEN build with zero
# conformance coverage — and it had been failing on Linux runners, so the
# corpus gate had never actually run in CI. Three layers hid it: this
# swallowed stderr, this exited 0, and the workflow ignored the result anyway.
#
# The sentinel is the same file the test harness checks
# (tests/test_vectors.rs `ensure_dav1d_test_data`), so a partial clone that
# leaves the directory non-empty but incomplete is caught here rather than
# surfacing later as the harness's own `git clone` exiting 128 on a
# non-empty destination.
SENTINEL="$DAV1D_DATA_DIR/8-bit/data/meson.build"

if [ -f "$SENTINEL" ]; then
    echo "  ✓ dav1d test data (cached)"
else
    if [ -d "$DAV1D_DATA_DIR" ]; then
        echo "  ! $DAV1D_DATA_DIR exists but is incomplete (no $SENTINEL) — removing" >&2
        rm -rf "$DAV1D_DATA_DIR"
    fi
    ok=0
    for attempt in 1 2 3; do
        # stderr is NOT discarded: the reason a fetch fails is the whole
        # diagnostic, and discarding it is what made this invisible.
        if git clone --depth 1 https://code.videolan.org/videolan/dav1d-test-data.git "$DAV1D_DATA_DIR"; then
            ok=1
            break
        fi
        echo "  ! clone attempt $attempt/3 failed" >&2
        rm -rf "$DAV1D_DATA_DIR"
        sleep $((attempt * 10))
    done
    if [ "$ok" != "1" ]; then
        echo "  ✗ dav1d test data clone FAILED after 3 attempts" >&2
        exit 1
    fi
    if [ ! -f "$SENTINEL" ]; then
        echo "  ✗ clone reported success but $SENTINEL is missing" >&2
        exit 1
    fi
    echo "  ✓ dav1d test data cloned"
fi

echo ""
echo "Test vectors ready in: $VECTORS_DIR"
ls -lh "$VECTORS_DIR" 2>/dev/null | grep "\.ivf$\|\.obu$" || echo "  (no vectors downloaded yet)"
