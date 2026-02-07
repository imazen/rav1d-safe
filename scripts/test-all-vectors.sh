#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TEST_VECTORS_DIR="$PROJECT_ROOT/test-vectors"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=== rav1d-safe Test Vector Validation ==="
echo

# Check if test vectors exist
if [ ! -d "$TEST_VECTORS_DIR" ]; then
    echo -e "${YELLOW}Warning: test-vectors/ not found${NC}"
    echo "Run: bash scripts/download-all-test-vectors.sh"
    exit 1
fi

# Build decoder
echo "→ Building rav1d-safe decoder..."
cd "$PROJECT_ROOT"
cargo build --release --no-default-features --features "bitdepth_8,bitdepth_16"
echo -e "${GREEN}✓${NC} Build complete"
echo

# Test 1: Integration tests (dav1d-test-data)
echo "=== Test 1: Integration Tests (dav1d-test-data) ==="
if [ -d "$TEST_VECTORS_DIR/dav1d-test-data" ]; then
    cargo test --test integration_decode --no-default-features \
        --features "bitdepth_8,bitdepth_16" --release -- --ignored
    echo -e "${GREEN}✓${NC} Integration tests passed"
else
    echo -e "${YELLOW}⊘${NC} dav1d-test-data not found, skipping"
fi
echo

# Test 2: Managed API tests
echo "=== Test 2: Managed API Unit Tests ==="
cargo test --test managed_api_test --no-default-features \
    --features "bitdepth_8,bitdepth_16" --release
echo -e "${GREEN}✓${NC} Managed API tests passed"
echo

# Test 3: Sample Argon vectors
echo "=== Test 3: Argon Conformance Suite (sample) ==="
if [ -d "$TEST_VECTORS_DIR/argon/argon" ]; then
    ARGON_DIR="$TEST_VECTORS_DIR/argon/argon"
    # Test a few representative files
    SAMPLE_COUNT=0
    PASS_COUNT=0
    FAIL_COUNT=0
    
    for ivf in $(find "$ARGON_DIR" -name "*.ivf" -o -name "*.obu" | head -20); do
        ((SAMPLE_COUNT++))
        if cargo run --release --example managed_decode --no-default-features \
            --features "bitdepth_8,bitdepth_16" -- "$ivf" > /dev/null 2>&1; then
            ((PASS_COUNT++))
            echo -e "  ${GREEN}✓${NC} $(basename $ivf)"
        else
            ((FAIL_COUNT++))
            echo -e "  ${RED}✗${NC} $(basename $ivf)"
        fi
    done
    
    echo
    echo "  Tested: $SAMPLE_COUNT files"
    echo -e "  ${GREEN}Passed: $PASS_COUNT${NC}"
    if [ $FAIL_COUNT -gt 0 ]; then
        echo -e "  ${RED}Failed: $FAIL_COUNT${NC}"
    fi
else
    echo -e "${YELLOW}⊘${NC} Argon suite not found, skipping"
fi
echo

# Test 4: Fluster vectors (sample)
echo "=== Test 4: Fluster Vectors (sample) ==="
if [ -d "$TEST_VECTORS_DIR/fluster/resources" ]; then
    FLUSTER_DIR="$TEST_VECTORS_DIR/fluster/resources/test_vectors/av1"
    SAMPLE_COUNT=0
    PASS_COUNT=0
    FAIL_COUNT=0
    
    for suite in AV1-TEST-VECTORS CHROMIUM-8bit-AV1-TEST-VECTORS CHROMIUM-10bit-AV1-TEST-VECTORS; do
        if [ -d "$FLUSTER_DIR/$suite" ]; then
            echo "  Testing $suite (first 5 files):"
            for ivf in $(find "$FLUSTER_DIR/$suite" -name "*.ivf" | head -5); do
                ((SAMPLE_COUNT++))
                if cargo run --release --example managed_decode --no-default-features \
                    --features "bitdepth_8,bitdepth_16" -- "$ivf" > /dev/null 2>&1; then
                    ((PASS_COUNT++))
                    echo -e "    ${GREEN}✓${NC} $(basename $ivf)"
                else
                    ((FAIL_COUNT++))
                    echo -e "    ${RED}✗${NC} $(basename $ivf)"
                fi
            done
        fi
    done
    
    echo
    echo "  Tested: $SAMPLE_COUNT files"
    echo -e "  ${GREEN}Passed: $PASS_COUNT${NC}"
    if [ $FAIL_COUNT -gt 0 ]; then
        echo -e "  ${RED}Failed: $FAIL_COUNT${NC}"
    fi
else
    echo -e "${YELLOW}⊘${NC} Fluster vectors not found, skipping"
fi
echo

echo "=== Summary ==="
echo -e "${GREEN}✓${NC} All available tests completed"
