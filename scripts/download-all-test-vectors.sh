#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TEST_VECTORS_DIR="$PROJECT_ROOT/test-vectors"

echo "=== Downloading AV1 Test Vectors ==="
echo "Target directory: $TEST_VECTORS_DIR"
echo

mkdir -p "$TEST_VECTORS_DIR"/{argon,dav1d-test-data,fluster,libaom}

# 1. dav1d test data (already cloned in previous script)
if [ -d "$TEST_VECTORS_DIR/dav1d-test-data/.git" ]; then
    echo "✓ dav1d-test-data already cloned"
else
    echo "→ Cloning dav1d-test-data..."
    git clone --depth 1 https://code.videolan.org/videolan/dav1d-test-data.git \
        "$TEST_VECTORS_DIR/dav1d-test-data"
    echo "✓ dav1d-test-data cloned"
fi

# 2. Argon conformance suite (delegates format/URL handling to download-argon.sh)
bash "$SCRIPT_DIR/download-argon.sh" "$TEST_VECTORS_DIR/argon"

# 3. Fluster framework and test vectors
if [ -d "$TEST_VECTORS_DIR/fluster/.git" ]; then
    echo "✓ Fluster already cloned"
else
    echo "→ Cloning Fluster framework..."
    git clone --depth 1 https://github.com/fluendo/fluster.git \
        "$TEST_VECTORS_DIR/fluster"
    echo "✓ Fluster cloned"
fi

cd "$TEST_VECTORS_DIR/fluster"

# Download AV1 test suites
echo "→ Downloading AV1-TEST-VECTORS (Google Cloud Storage)..."
./fluster.py download AV1-TEST-VECTORS

echo "→ Downloading CHROMIUM-8bit-AV1-TEST-VECTORS..."
./fluster.py download CHROMIUM-8bit-AV1-TEST-VECTORS

echo "→ Downloading CHROMIUM-10bit-AV1-TEST-VECTORS..."
./fluster.py download CHROMIUM-10bit-AV1-TEST-VECTORS

echo
echo "=== Download Summary ==="
du -sh "$TEST_VECTORS_DIR"/* | sed 's/^/  /'

echo
echo "✓ All test vectors downloaded successfully"
echo
echo "Total size: $(du -sh "$TEST_VECTORS_DIR" | cut -f1)"
echo "Argon:      ~2763 test files (conformance suite)"
echo "dav1d:      ~160,000+ test files"
echo "Fluster:    ~312 IVF files (AV1 + Chromium)"
