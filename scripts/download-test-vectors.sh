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
DAV1D_DATA_DIR="$VECTORS_DIR/dav1d-test-data"

if [ ! -d "$DAV1D_DATA_DIR" ]; then
    if git clone --depth 1 https://code.videolan.org/videolan/dav1d-test-data.git "$DAV1D_DATA_DIR" 2>/dev/null; then
        echo "  ✓ dav1d test data cloned"
    else
        echo "  ⓘ dav1d test data not available (optional)"
    fi
else
    echo "  ✓ dav1d test data (cached)"
fi

echo ""
echo "Test vectors ready in: $VECTORS_DIR"
ls -lh "$VECTORS_DIR" 2>/dev/null | grep "\.ivf$\|\.obu$" || echo "  (no vectors downloaded yet)"
