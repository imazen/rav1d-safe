# Test Vectors for rav1d-safe

This document describes the test vector infrastructure for validating rav1d-safe against industry-standard AV1 conformance suites.

## Quick Start

```bash
# Download all test vectors (~5.2GB)
just download-all-vectors

# Run comprehensive validation
just test-all-vectors

# Run integration tests
just test-integration
```

## Test Vector Sources

### 1. Argon Conformance Suite (5.1GB, 2,763 files)

**Source:** Alliance for Open Media (AOMedia) official conformance suite  
**URL:** https://streams.videolan.org/argon/argon.tar.zst  
**Description:** Formal verification test suite - exercises every equation in the AV1 specification

**What it tests:**
- Profile 0, 1, 2 (8-bit, 10-bit, 12-bit)
- Core, Stress, and Error test cases
- Annex B and Non-Annex B formats
- Complete spec coverage

**Location:** `test-vectors/argon/argon/`

### 2. dav1d Test Data (109MB, ~160,000 files)

**Source:** VideoLAN dav1d project  
**URL:** https://code.videolan.org/videolan/dav1d-test-data.git  
**Description:** Comprehensive test suite from the dav1d decoder project

**What it tests:**
- 8-bit, 10-bit, 12-bit videos
- Film grain synthesis
- HDR metadata (CLL, mastering display)
- Various resolutions and features
- Fuzzing corpus (oss-fuzz)

**Location:** `test-vectors/dav1d-test-data/`

### 3. AV1-TEST-VECTORS (7.5MB, 240 files)

**Source:** Google Cloud Storage (AOM project)  
**URL:** https://storage.googleapis.com/aom-test-data/  
**Description:** Official AOM test vector catalogue

**What it tests:**
- Quantizer variations
- Different resolutions
- SVC (Scalable Video Coding)
- Intra-only, IntraBC
- Various AV1 features

**Location:** `test-vectors/fluster/resources/test_vectors/av1/AV1-TEST-VECTORS/`

### 4. Chromium Test Vectors (4.4MB, 72 files)

**Source:** Chromium OS project  
**URL:** https://storage.googleapis.com/chromiumos-test-assets-public/  
**Description:** Chromium browser test suite (8-bit and 10-bit)

**What it tests:**
- Real-world browser playback scenarios
- Tiling configurations
- Film grain
- Size changes
- Frame references

**Locations:**
- 8-bit: `test-vectors/fluster/resources/test_vectors/av1/CHROMIUM-8bit-AV1-TEST-VECTORS/`
- 10-bit: `test-vectors/fluster/resources/test_vectors/av1/CHROMIUM-10bit-AV1-TEST-VECTORS/`

## Test Infrastructure

### Scripts

**download-all-test-vectors.sh**
- Downloads and extracts all test vector suites
- Handles Git clones, wget downloads, and tar extraction
- Idempotent (safe to run multiple times)

**test-all-vectors.sh**
- Runs decoder against samples from all suites
- Shows pass/fail status for each file
- Provides summary statistics

### CI Workflow

**test-vectors.yml**
- Runs on push to main/feature branches
- Weekly scheduled run (Sundays)
- Caches test vectors for fast subsequent runs
- Tests samples from each suite (Argon: 50, Fluster: all, dav1d: 30)
- Requires 80% pass rate minimum
- Generates test report artifacts

### Justfile Commands

```bash
just download-all-vectors  # Download all test vectors
just test-all-vectors      # Run comprehensive validation
just test-argon            # Test against Argon suite (100 samples)
just test-integration      # Run integration tests
```

## Test Results

Current status:
- ✅ Integration tests: 2/2 passing (dav1d-test-data)
- ⏳ Argon suite: Testing in progress
- ⏳ Fluster vectors: Testing in progress
- ⏳ Full conformance: CI workflow pending

## Disk Space Requirements

- Argon: 5.1GB (2.5GB compressed)
- dav1d: 109MB
- Fluster: 17MB
- **Total: ~5.2GB**

All test vectors are in `test-vectors/` which is gitignored.

## Adding New Test Suites

To add new test suites via Fluster:

```bash
cd test-vectors/fluster

# List available suites
./fluster.py list | grep AV1

# Download a suite
./fluster.py download SUITE-NAME

# Run decoder against suite
./fluster.py run -d rav1d-safe SUITE-NAME
```

## References

- AV1 Specification: https://aomediacodec.github.io/av1-spec/
- AOMedia: https://aomedia.org/av1-features/get-started/
- Fluster: https://github.com/fluendo/fluster
- dav1d: https://code.videolan.org/videolan/dav1d
- libaom: https://aomedia.googlesource.com/aom/
