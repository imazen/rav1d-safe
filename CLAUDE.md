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

**Tests:** `tests/managed_api_test.rs` (decoder creation, settings, empty data)

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
