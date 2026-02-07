# rav1d-safe

## DO NOT STOP - KEEP PORTING ASM TO SAFE RUST

**DO NOT STOP PORTING ASM. JUST KEEP GOING. DO NOT ASK WHICH MODULE TO PORT NEXT.**
**DO NOT STOP PORTING ASM. JUST KEEP GOING. DO NOT ASK WHICH MODULE TO PORT NEXT.**
**DO NOT STOP PORTING ASM. JUST KEEP GOING. DO NOT ASK WHICH MODULE TO PORT NEXT.**
**DO NOT STOP PORTING ASM. JUST KEEP GOING. DO NOT ASK WHICH MODULE TO PORT NEXT.**
**DO NOT STOP PORTING ASM. JUST KEEP GOING. DO NOT ASK WHICH MODULE TO PORT NEXT.**
**DO NOT STOP PORTING ASM. JUST KEEP GOING. DO NOT ASK WHICH MODULE TO PORT NEXT.**
**DO NOT STOP PORTING ASM. JUST KEEP GOING. DO NOT ASK WHICH MODULE TO PORT NEXT.**
**DO NOT STOP PORTING ASM. JUST KEEP GOING. DO NOT ASK WHICH MODULE TO PORT NEXT.**
**DO NOT STOP PORTING ASM. JUST KEEP GOING. DO NOT ASK WHICH MODULE TO PORT NEXT.**
**DO NOT STOP PORTING ASM. JUST KEEP GOING. DO NOT ASK WHICH MODULE TO PORT NEXT.**

Pick the next unfinished module and port it. Priority order:
1. ~~ipred (~26k lines)~~ **COMPLETE** (all 14 modes, 8bpc + 16bpc)
2. ~~ITX (~11k lines)~~ **COMPLETE** (160 transforms each for 8bpc/16bpc)
3. ~~loopfilter/CDEF~~ **COMPLETE** (8bpc + 16bpc)
4. ~~looprestoration~~ **COMPLETE** (Wiener + SGR 8bpc + 16bpc)
5. ~~ARM NEON mc~~ **COMPLETE** (all 8tap filters, 8bpc + 16bpc)
6. ~~filmgrain~~ **COMPLETE** (x86 + ARM, 8bpc + 16bpc)
7. ~~pal~~ **COMPLETE** (x86 AVX2, ARM uses fallback)
8. ~~refmvs~~ **COMPLETE** (x86 AVX2 + ARM NEON)

**ALL MODULES COMPLETE!** msac symbol_adapt16 now has inline safe_simd (AVX2/NEON).

Safe SIMD fork of rav1d - replacing 160k lines of hand-written assembly with safe Rust intrinsics.

## Quick Commands

```bash
# Build without asm (pure Rust + SIMD intrinsics)
cargo build --no-default-features --features "bitdepth_8,bitdepth_16" --release

# Build with asm (original rav1d behavior)
cargo build --features "asm,bitdepth_8,bitdepth_16" --release

# Run tests
cargo test --no-default-features --features "bitdepth_8,bitdepth_16" --release

# Benchmark via zenavif (20 decodes)
cd /home/lilith/work/zenavif && touch src/lib.rs && cargo build --release --example decode_avif
time for i in {1..20}; do ./target/release/examples/decode_avif /home/lilith/work/aom-decode/tests/test.avif /dev/null 2>/dev/null; done
```

## Feature Flags

- `asm` - Use hand-written assembly (default, original rav1d)
- `bitdepth_8` - 8-bit pixel support
- `bitdepth_16` - 10/12-bit pixel support

## Safe-SIMD Modules

### x86_64 (AVX2)

| Module | Location | Status |
|--------|----------|--------|
| mc | `src/safe_simd/mc.rs` | **Complete** - 8bpc+16bpc |
| itx | `src/safe_simd/itx.rs` | **Complete** - 160 transforms each for 8bpc/16bpc |
| loopfilter | `src/safe_simd/loopfilter.rs` | **Complete** - 8bpc + 16bpc |
| cdef | `src/safe_simd/cdef.rs` | **Complete** - 8bpc + 16bpc |
| looprestoration | `src/safe_simd/looprestoration.rs` | **Complete** - Wiener + SGR 8bpc + 16bpc |
| ipred | `src/safe_simd/ipred.rs` | **Complete** - All 14 modes, 8bpc + 16bpc |
| filmgrain | `src/safe_simd/filmgrain.rs` | **Complete** - 8bpc + 16bpc |
| pal | `src/safe_simd/pal.rs` | **Complete** - pal_idx_finish AVX2 |
| refmvs | `src/safe_simd/refmvs.rs` | **Complete** - splat_mv AVX2 |
| msac | `src/msac.rs` (inline) | **Complete** - symbol_adapt16 AVX2 |

### ARM aarch64 (NEON)

| Module | Location | Status |
|--------|----------|--------|
| mc_arm | `src/safe_simd/mc_arm.rs` | **Complete** - 8bpc+16bpc (all MC functions including 8tap) |
| ipred_arm | `src/safe_simd/ipred_arm.rs` | **Complete** - DC/V/H/paeth/smooth modes (8bpc + 16bpc) |
| cdef_arm | `src/safe_simd/cdef_arm.rs` | **Complete** - All filter sizes (8bpc + 16bpc) |
| loopfilter_arm | `src/safe_simd/loopfilter_arm.rs` | **Complete** - Y/UV H/V filters (8bpc + 16bpc) |
| looprestoration_arm | `src/safe_simd/looprestoration_arm.rs` | **Complete** - Wiener + SGR (5x5, 3x3, mix) 8bpc + 16bpc |
| itx_arm | `src/safe_simd/itx_arm.rs` | **Complete** - 334 FFI functions, 320 dispatch entries |
| filmgrain_arm | `src/safe_simd/filmgrain_arm.rs` | **Complete** - 8bpc + 16bpc |
| refmvs_arm | `src/safe_simd/refmvs_arm.rs` | **Complete** - splat_mv NEON |
| msac | `src/msac.rs` (inline) | **Complete** - symbol_adapt16 NEON |

## Performance Status (2026-02-05)

Full-stack benchmark via zenavif (20 decodes of test.avif):
- ASM: ~1.17s
- Safe-SIMD: ~1.11s
- **Safe-SIMD MATCHES or BEATS ASM performance!**

## Porting Progress (160k lines target)

**SIMD optimized (~32k lines in safe_simd/):**
- MC x86 module (~5k lines): Complete (8bpc + 16bpc)
- MC ARM module (~4k lines): Complete (8bpc + 16bpc all filters including 8tap)
- ITX x86 module (~12k lines): **100% complete** (160 transforms each 8bpc/16bpc)
- ITX ARM module (~6k lines): **100% complete** (334 FFI functions, 320 dispatch entries)
- Loopfilter (~9k lines): Complete (8bpc + 16bpc)
- CDEF (~7k lines): Complete (8bpc + 16bpc)
- Looprestoration (~17k lines): Complete (Wiener + SGR 8bpc + 16bpc)
- ipred (~26k lines): Complete (all 14 modes, 8bpc + 16bpc)
- filmgrain x86 (~1k lines) + ARM (~750 lines): Complete (8bpc + 16bpc)
- pal x86 (~150 lines): Complete (AVX2 pal_idx_finish)
- refmvs x86 (~60 lines) + ARM (~50 lines): Complete (splat_mv)

**msac (inline in src/msac.rs):**
- symbol_adapt16: AVX2 (x86_64) and NEON (aarch64) - parallelized CDF probability calc and comparison
- symbol_adapt4/8: Use scalar Rust fallback (SIMD overhead not worth it for small n_symbols)
- bool functions: Use scalar Rust fallback

**Using Rust fallbacks (SIMD not beneficial):**
- refmvs save_tmvs/load_tmvs: Complex conditional logic, not SIMD-friendly

**Cross-compilation:**
- x86_64: Full support, matches ASM performance
- aarch64: Full support (cargo check --target aarch64-unknown-linux-gnu passes)

## Architecture

### Dispatch Pattern

rav1d uses function pointer dispatch for SIMD:
1. `wrap_fn_ptr!` macro creates type-safe function pointer wrappers
2. For asm: `bd_fn!` macro links to asm symbols, `call` method invokes fn ptr
3. For non-asm: `call` method uses `cfg_if` to call `*_dispatch` directly (no fn ptrs)
4. `*_dispatch` functions do `Desktop64::summon()` or `CpuFlags::AVX2` check, call inner SIMD

### FFI Wrapper Pattern (asm only)

FFI wrappers are gated behind `#[cfg(feature = "asm")]`:
```rust
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn function_8bpc_avx2(
    dst: *const FFISafe<...>,
    // ... other params
) {
    let dst = unsafe { *FFISafe::get(dst) };
    // Call inner implementation
}
```

## Safety Status

**Crate-level deny(unsafe_code) when asm disabled.** `lib.rs` has `#![cfg_attr(not(any(feature = "asm", feature = "c-ffi")), deny(unsafe_code))]` — compiler-enforced safety for the entire non-asm path.

**47/80 modules have explicit `deny(unsafe_code)`:**
- 35 unconditionally safe (decode, recon, lf_mask, lf_apply, ctx, obu, cdf, etc.)
- 12 conditionally safe when asm disabled (cdef, filmgrain, ipred, itx, loopfilter, looprestoration, mc, pal, data, tables, cpu, safe_simd/pal)

**FFI wrappers gated behind `feature = "asm"`** in: cdef, cdef_arm, loopfilter, loopfilter_arm, looprestoration, looprestoration_arm, filmgrain, filmgrain_arm, pal.

**Archmage conversions complete:** cdef constrain_avx2, msac symbol_adapt16 AVX2.

**Feature flags:**
- `unchecked` - Use unchecked slice access in SIMD hot paths (skips bounds checks)
- `src/safe_simd/pixel_access.rs` - Helper module for checked/unchecked slice access

**Unsafe reduction progress (safe_simd/):**
- cdef.rs: Padding functions converted to safe DisjointMut slice access (top/bottom)
- cdef_arm.rs: Same conversion for ARM NEON path
- loopfilter.rs: Dispatch function fully safe (lvl + dst via slice access)
- Remaining: Raw pointer pixel access in inner SIMD functions (mc, itx, ipred, etc.)
  - SIMD intrinsics (`_mm256_storeu_si256` etc.) are inherently unsafe
  - Inner functions take raw pointers; need signature changes to accept slices
  - FFI wrappers in ipred/mc/itx not gated behind `feature = "asm"` (always compiled)

**c-ffi decoupled from fn-ptr dispatch.** The `c-ffi` feature now only controls the 19 `dav1d_*` extern "C" entry points in `src/lib.rs`. Internal DSP dispatch uses direct function calls (no function pointers) when `asm` is disabled.

## Managed Safe API

**Location:** `src/managed.rs` (~970 lines, 100% safe Rust)

A fully safe, zero-copy API for decoding AV1 video. Enforced by `#![deny(unsafe_code)]`.

**Key types:**
- `Decoder` - safe wrapper around `Rav1dContext` (new/with_settings/decode/flush/drop)
- `Settings` - type-safe configuration with `InloopFilters`, `DecodeFrameType` enums
- `Frame` - decoded frame with metadata (width, height, bit depth, color info, HDR)
- `Planes` - enum dispatching to `Planes8`/`Planes16` for type-safe pixel access
- `PlaneView8`/`PlaneView16` - zero-copy 2D strided views holding `DisjointImmutGuard`
- `Error` - simple error enum with `From<Rav1dError>` (no thiserror dependency)

**Color/HDR metadata:**
- `ColorPrimaries`, `TransferCharacteristics`, `MatrixCoefficients` - color space info
- `ColorRange` - Limited vs Full
- `ContentLightLevel` - HDR max/avg nits
- `MasteringDisplay` - SMPTE 2086 with nit conversion helpers

**Input format:**
- Expects raw OBU (Open Bitstream Unit) data, not container formats
- For IVF files, use an IVF parser to extract OBU frames (see `tests/ivf_parser.rs`)
- For Annex B or Section 5 low overhead formats, additional parsing may be needed

**Threading:**
- Default: `threads: 1` (single-threaded, deterministic, synchronous)
- `threads: 0`: Auto-detect cores (frame threading, better performance, asynchronous)
- With frame threading, `decode()` may return `None` for complete frames (call again or `flush()`)

**Usage example:**
```rust
use rav1d_safe::src::managed::Decoder;

let mut decoder = Decoder::new()?;
if let Some(frame) = decoder.decode(obu_data)? {
    match frame.planes() {
        Planes::Depth8(planes) => {
            for row in planes.y().rows() {
                // Process 8-bit row
            }
        }
        Planes::Depth16(planes) => {
            let pixel = planes.y().pixel(0, 0);
        }
    }
}
```

**Tests:**
- `tests/managed_api_test.rs` - unit tests (decoder creation, settings, empty data)
- `tests/integration_decode.rs` - integration tests with real IVF test vectors (2/2 passing)

## CI & Testing Infrastructure

### GitHub Actions Workflows (.github/workflows/ci.yml)

**Build Matrix:**
- OS: ubuntu-latest, windows-latest, macos-latest, ubuntu-24.04-arm
- Features: `bitdepth_8,bitdepth_16` (safe-simd) and `asm,bitdepth_8,bitdepth_16`
- Builds: debug + release
- Tests: unit tests + integration tests (with test vectors)

**Quality Checks:**
- Clippy: `-D warnings` on all targets
- Format: `cargo fmt --check`
- Cross-compile: aarch64-unknown-linux-gnu, x86_64-unknown-linux-musl
- Coverage: `cargo-llvm-cov` → codecov upload

**Test Vectors:**
- Downloads dav1d-test-data repository (~160k+ test files)
- Caches in `target/test-vectors/`
- Organized: 8-bit/, 10-bit/, 12-bit/, oss-fuzz/
- Includes: conformance, film grain, HDR, argon samples

### Test Infrastructure

**Integration Tests (tests/integration_decode.rs):**
- `test_decode_real_bitstream` - decode OBU files via managed API
- `test_decode_hdr_metadata` - extract HDR metadata (CLL, mastering display)
- Uses dav1d-test-data vectors
- Marked `#[ignore]` until OBU format issue resolved

**Test Vector Management (tests/test_vectors.rs):**
- Download/cache infrastructure
- SHA256 verification support
- Extensible for multiple sources (AOM, dav1d, conformance)

**Download Script (scripts/download-test-vectors.sh):**
- Clones dav1d-test-data repository
- Future: AOM test data from Google Cloud Storage
- Cached downloads with size reporting

### Examples

**examples/managed_decode.rs:**
- Full managed API demonstration
- Decodes IVF/OBU files
- Displays frame info, color metadata, HDR data
- Sample pixel access (8-bit and 16-bit)

### Justfile Commands

```bash
just build               # Safe-SIMD build
just build-asm           # ASM build
just test                # Run tests
just download-vectors    # Fetch test vectors
just test-integration    # Integration tests with vectors
just clippy              # Lint checks
just fmt / fmt-check     # Format code / check formatting
just check               # All checks (fmt, clippy, test)
just cross-aarch64       # Cross-compile check
just doc                 # Generate and open docs
just coverage            # HTML coverage report
just ci                  # Run all CI checks locally
```

### Current Status

- ✅ CI workflow configured (not yet pushed to GitHub)
- ✅ Test vectors downloaded (dav1d-test-data cloned)
- ✅ Integration test infrastructure in place
- ✅ Managed API unit tests pass (3/3)
- ✅ **Integration tests PASS (2/2)** - OBU decoding issue RESOLVED
  - Added IVF container parser for test vectors
  - Fixed managed API threading defaults (threads=1 for deterministic behavior)
  - Successfully decodes 64x64 10-bit frames with HDR metadata
- ✅ Justfile for common tasks
- ✅ Example demonstrating managed API

### Integration Test Infrastructure

**IVF Parser (tests/ivf_parser.rs):**
- Parses IVF container format (DKIF signature)
- Extracts raw OBU frames from IVF files
- Used by integration tests to feed proper OBU data to decoder

**Threading Behavior:**
- Managed API defaults to `threads: 1` (single-threaded, deterministic)
- `threads: 0` enables frame threading (better performance, asynchronous behavior)
- With frame threading, `decode()` may return `None` even with complete frames
- Frame threading requires polling `decode()` or `flush()` multiple times


## Test Vectors

All test vectors are located in `test-vectors/` (gitignored, not committed to repo).

### Download All Test Vectors

```bash
bash scripts/download-all-test-vectors.sh
```

This downloads:
- **dav1d-test-data**: ~160,000+ files, 109MB
- **Argon conformance suite**: ~2,763 files, 5.1GB
- **Fluster AV1 vectors**: ~312 IVF files, 17MB
- **Total**: ~5.2GB

### Test Vector Sources

| Source | Location | Files | Size | Description |
|--------|----------|-------|------|-------------|
| **dav1d-test-data** | `test-vectors/dav1d-test-data/` | ~160k | 109MB | VideoLAN test suite (8/10/12-bit, film grain, HDR, argon, oss-fuzz) |
| **Argon Suite** | `test-vectors/argon/argon/` | 2,763 | 5.1GB | Formal verification conformance suite (exercises every AV1 spec equation) |
| **AV1-TEST-VECTORS** | `test-vectors/fluster/resources/test_vectors/av1/AV1-TEST-VECTORS/` | 240 | 7.5MB | Google Cloud Storage test vectors |
| **Chromium 8-bit** | `test-vectors/fluster/resources/test_vectors/av1/CHROMIUM-8bit-AV1-TEST-VECTORS/` | 36 | 2.4MB | Chromium 8-bit test vectors |
| **Chromium 10-bit** | `test-vectors/fluster/resources/test_vectors/av1/CHROMIUM-10bit-AV1-TEST-VECTORS/` | 36 | 2.0MB | Chromium 10-bit test vectors |

### Test Vector URLs

**Primary Sources:**
- dav1d: `https://code.videolan.org/videolan/dav1d-test-data.git`
- Argon: `https://streams.videolan.org/argon/argon.tar.zst`
- AOM: `https://storage.googleapis.com/aom-test-data/`
- Chromium: `https://storage.googleapis.com/chromiumos-test-assets-public/tast/cros/video/test_vectors/av1/`

**Fluster Framework:**
- Repo: `https://github.com/fluendo/fluster`
- Manages downloading and running test suites
- Supports multiple decoders (dav1d, libaom, FFmpeg, GStreamer, etc.)

### Running Tests Against All Vectors

```bash
# Integration tests (uses dav1d-test-data)
just test-integration

# Run against Fluster vectors
cd test-vectors/fluster
./fluster.py run -d rav1d-safe AV1-TEST-VECTORS

# Run against Argon suite
# TODO: Create argon test runner
```

## TODO: CI & Parity Testing

### GitHub Actions Workflows

Build matrix: `{x86_64, aarch64, wasm32-wasi (simd128)} × {linux, macos, windows}`

Workflow must include:
- `cargo build --no-default-features --features "bitdepth_8,bitdepth_16"` (pure safe)
- `cargo build --no-default-features --features "bitdepth_8,bitdepth_16,c-ffi"` (safe + C API)
- `cargo test --release`
- `cargo fmt --check`
- `cargo clippy --all-targets -- -D warnings`
- Code coverage via `cargo-llvm-cov` uploaded to codecov
- aarch64 cross-check via `cargo check --target aarch64-unknown-linux-gnu`
- wasm32 simd128 build check

### Decode Parity Testing

Side-by-side decode with reference decoders to ensure exact pixel-level parity:
1. **dav1d (C)** — primary reference. Decode same bitstream with both dav1d and rav1d-safe, compare output frame-by-frame. Must be bit-exact (no film grain) or statistically identical (with film grain, same seed).
2. **libaom** — secondary reference. The AV1 reference decoder. Useful for catching bugs that both dav1d and rav1d might share.

### Test Video Suite

Need to find and clone:
- **AV1 conformance test vectors** — `aomedia.googlesource.com/aom-testing` or the IETF conformance suite
- **dav1d test suite** — `code.videolan.org/videolan/dav1d-test-data`
- Coverage should include: all profiles (main/high/professional), all bit depths (8/10/12), all chroma subsampling (420/422/444), film grain, screen content coding, intra-only, inter, SVC, error resilience

### Parity Test Harness

Build a test binary or integration test that:
1. Takes an IVF/OBU/webm input
2. Decodes with rav1d-safe (via Rust API)
3. Decodes with dav1d (via C FFI or subprocess)
4. Compares output YUV planes byte-for-byte
5. Reports any mismatches with frame number, plane, coordinates

## Known Issues

(none currently)

## Technical Notes

### Key Constants
- `REST_UNIT_STRIDE = 390` for looprestoration (256 * 3/2 + 3 + 3)
- `intermediate_bits = 4` for 8bpc MC filters
- pmulhrsw rounding: `(a * b + 16384) >> 15`

### SIMD Intrinsics
- Use `#[target_feature(enable = "avx2")]` for FFI wrappers
- Shift intrinsics require const generics: `_mm256_srai_epi32::<11>(sum)`
- Mark inner implementations `unsafe fn` with explicit `unsafe {}` blocks

## Known Issues - Managed API

### Critical: Dav1dDataGuard Missing (Memory Leak on Panic)

**Status:** 🔴 **MUST FIX before production**

**Problem:** `Dav1dData` does not implement `Drop`. If `decode()` panics after `dav1d_data_wrap` but before consuming all data, the internal `RawCArc` reference leaks.

**Location:** `src/managed.rs` - `Decoder::decode()` method

**Solution:** Implement RAII guard:

```rust
struct Dav1dDataGuard(Dav1dData);

impl Dav1dDataGuard {
    fn new(data: &[u8]) -> Result<Self, Error> {
        let mut dav1d_data = Dav1dData::default();
        unsafe {
            let result = dav1d_data_wrap(
                NonNull::new(&mut dav1d_data),
                NonNull::new(data.as_ptr() as *mut u8),
                data.len(),
                Some(null_free),
                None,
            );
            if result.0 < 0 {
                return Err(Error::DecodeFailed(result.0));
            }
        }
        Ok(Self(dav1d_data))
    }
    
    fn as_mut(&mut self) -> &mut Dav1dData {
        &mut self.0
    }
}

impl Drop for Dav1dDataGuard {
    fn drop(&mut self) {
        unsafe {
            dav1d_data_unref(NonNull::new(&mut self.0));
        }
    }
}
```

**Testing:**
- Add panic safety test that panics during decode
- Run with LeakSanitizer (LSAN)
- Verify no leaks with valgrind

### Medium Priority: Memory Leak Tests

**Status:** ⚠️ Recommended

**Need:**
1. Valgrind/ASAN integration in CI
2. Drop tests verifying memory returns to baseline
3. Panic safety tests

**Example test:**
```rust
#[test]
fn test_decoder_drop_frees_memory() {
    let initial = get_memory_usage();
    {
        let mut decoder = Decoder::new().unwrap();
        decoder.decode(test_data).unwrap();
    }
    let final_usage = get_memory_usage();
    assert_eq!(initial, final_usage, "memory leaked");
}
```

### Low Priority: Thread Pool Cleanup Audit

**Status:** ℹ️ Verify

**Need:** Audit `Rav1dContext` thread pool cleanup logic
- Check `Arc<TaskThreadData>` drop implementation  
- Verify worker threads join properly on context drop
- No hanging threads or leaked thread handles

**Location:** `src/internal.rs` - `Rav1dContext` drop logic

