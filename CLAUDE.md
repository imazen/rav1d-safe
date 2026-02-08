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

## MANDATORY: Safe intrinsics strategy

**Rust 1.93+ made value-type SIMD intrinsics safe.** Computation intrinsics (`_mm256_add_epi32`, `_mm256_shuffle_epi8`, etc.) are now safe functions — no `unsafe` needed.

**Two things still require wrappers:**

1. **Pointer intrinsics (load/store)** — `_mm256_loadu_si256` takes `*const __m256i`, which requires `unsafe`. Use `safe_unaligned_simd` crate which wraps these as safe functions taking `&[T; N]` references. Our `loadu_256!`/`storeu_256!` macros dispatch to these.

2. **Target feature dispatch** — intrinsics are only safe when called within a function annotated with `#[target_feature(enable = "avx2")]` (or equivalent). `archmage` handles this via token-based dispatch (`Desktop64::summon()`, `#[arcane]`), so we **never manually write `is_x86_feature_detected!()` checks or `#[target_feature]` annotations on our functions**.

3. **Slice access** — Two APIs in `pixel_access.rs`, both zero-cost (verified identical asm):

   **`Flex` trait** — Use in super hot loops where you'd otherwise reach for pointer arithmetic:
   ```rust
   use crate::src::safe_simd::pixel_access::Flex;
   let c = coeff.flex();      // immutable FlexSlice with [] syntax
   let mut d = dst.flex_mut(); // mutable FlexSliceMut with [] syntax
   d[off] = ((d[off] as i32 + c[idx] as i32).clamp(0, 255)) as u8;
   ```
   - `slice.flex()[i]` / `slice.flex()[start..end]` / `slice.flex()[start..]`
   - `slice.flex_mut()[i] = val` / `slice.flex_mut()[start..end]`
   - Natural `[]` syntax, checked by default, unchecked when `unchecked` feature on

   **`SliceExt` trait** — Simpler single-access API:
   - `slice.at(i)` / `slice.at_mut(i)` — single element
   - `slice.sub(start, len)` / `slice.sub_mut(start, len)` — subslice
   - Import: `use crate::src::safe_simd::pixel_access::SliceExt;`

**Do NOT:**
- Manually add `#[target_feature(enable = "...")]` to new functions — use `#[arcane]` instead
- Manually call `is_x86_feature_detected!()` — use `Desktop64::summon()` / `CpuFlags` instead
- Use raw pointer load/store intrinsics — use `loadu_256!` / `storeu_256!` macros instead
- Block on any nightly-only feature for safety — everything works on stable Rust 1.93+

## HARD RULES — STOP GOING IN CIRCLES

**READ AND OBEY THESE EVERY TIME. DO NOT SKIP.**

1. **`#[arcane]` NEVER needs `#[allow(unsafe_code)]`.** It is safe by design. If you find yourself adding `allow(unsafe_code)` to an `#[arcane]` function, YOU ARE DOING SOMETHING WRONG. The function body itself must be rewritten to not use `unsafe` — use slices, safe macros, and safe intrinsics.

2. **`#[rite]` NEVER needs `#[allow(unsafe_code)]`.** Same as `#[arcane]` — it's a safe inner helper.

3. **Inner SIMD functions (using core::arch intrinsics) are NOT assembly.** `safe_simd/` contains ZERO `asm!` macros. Do NOT gate inner SIMD functions behind `#[cfg(feature = "asm")]`. Only gate `pub unsafe extern "C" fn` FFI wrappers behind asm.

4. **If an `#[arcane]` function won't compile under `forbid(unsafe_code)`, the function body is wrong.** Rewrite the body to use slices + safe macros. Do NOT add `#[allow(unsafe_code)]`. Do NOT gate behind `#[cfg(feature = "asm")]`.

5. **Read the archmage README before touching dispatch.** `Desktop64::summon()` for detection, `#[arcane]` for entry points, `#[rite]` for inner helpers. The prelude re-exports safe intrinsics. `safe_unaligned_simd` provides reference-based load/store.

6. **Conversion pattern for making `#[arcane]` functions safe:**
   - Change `dst: *mut u8` → `dst: &mut [u8]`
   - Change `coeff: *mut i16` → `coeff: &mut [i16]`
   - Replace `unsafe { *ptr.add(n) }` → `slice[n]`
   - Replace `unsafe { _mm256_loadu_si256(ptr) }` → `loadu_256!(&slice[off..off+32], [u8; 32])`
   - Replace `unsafe { _mm256_storeu_si256(ptr, v) }` → `storeu_256!(&mut slice[off..off+32], [u8; 32], v)`
   - Replace `unsafe { _mm_cvtsi32_si128(*(ptr as *const i32)) }` → `loadi32!(&slice[off..off+4])`
   - Remove ALL `unsafe {}` blocks — if intrinsics need unsafe, you're not in a `#[target_feature]` context (use `#[arcane]`/`#[rite]`)

7. **When you don't know how something works, READ THE README/DOCS FIRST.** Do not guess. Do not add workarounds. Especially for archmage, zerocopy, safe_unaligned_simd.

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

**Crate-level deny(unsafe_code) when asm/c-ffi disabled.** `lib.rs` has `#![cfg_attr(not(any(feature = "asm", feature = "c-ffi")), deny(unsafe_code))]` — compiler-enforced safety for the entire non-asm, non-c-ffi path.

**DisjointMut extracted to separate crate** (`crates/disjoint-mut/`):
- Provably safe abstraction (like RefCell for ranges) with always-on bounds checking
- Main crate re-exports + adds AlignedVec-specific impls
- Enables future `forbid(unsafe_code)` at crate level

**C FFI types gated behind `cfg(feature = "c-ffi")`:**
- `DavdPicture`, `DavdData`, `DavdDataProps`, `DavdUserData`, `DavdSettings`, `DavdLogger` — all gated
- `ITUTT35PayloadPtr`, `Dav1dITUTT35` struct (with `Send`/`Sync` impls) — gated; safe type alias when c-ffi off
- `RawArc`, `RawCArc`, `Dav1dContext`, `arc_into_raw` — gated (raw Arc ptr roundtrip)
- `From<Dav1d*>` / `From<Rav1d*> for Dav1d*` conversions (containing `unsafe { CArc::from_raw }`) — all gated
- Safe picture allocator: per-plane `Vec<u8>` from MemPool, no C callbacks needed
- Fallible allocation: `MemPool::pop_init` returns `Result<Vec, TryReserveError>`, propagated as `Rav1dError::ENOMEM`

**Module safety architecture (default build without asm/c-ffi on x86_64):**
- Only **1 module** needs unconditional `#[allow(unsafe_code)]`: **safe_simd**
  - Remaining unsafe: pointer load/store intrinsics (use `safe_unaligned_simd` + macros to eliminate) and FFI wrappers (gate behind `feature = "asm"`)
- 2 modules conditionally allow unsafe by architecture:
  - refmvs: `cfg_attr(feature = "asm", allow(unsafe_code))` — extern C wrappers gated behind asm
  - msac: `cfg_attr(any(asm, aarch64), allow(unsafe_code))` — NEON intrinsics on aarch64 only
- 6 modules use conditional `allow(unsafe_code)` only when c-ffi enabled:
  include/dav1d, c_arc, c_box, log, picture, lib (C API entry point)
- 1 module gated entirely behind asm/c-ffi: send_sync_non_null
- All other modules use **item-level** `#[allow(unsafe_code)]` on specific functions/impls:
  align (4 items), assume (1 fn), ffi_safe (2 fns), disjoint_mut (1 impl),
  c_arc (3 items), c_box (1 fn), internal (4 impls), refmvs (2 fns), msac (4 items)
- 42+ modules with `#![forbid(unsafe_code)]` — permanent, compiler-enforced
- 13 DSP modules with conditional deny when asm disabled

**c-ffi build fully working** (previously blocked by 320 `forge_token_dangerously` errors in safe_simd):
- Fixed: wrapped all `forge_token_dangerously()` calls in `unsafe { }` blocks (Rust 2024 edition compliance)
- Both `cargo check --features c-ffi` and `cargo test --features c-ffi` pass clean

**FFI wrappers gated behind `feature = "asm"`** in: cdef, cdef_arm, loopfilter, loopfilter_arm, looprestoration, looprestoration_arm, filmgrain, filmgrain_arm, pal.

**Archmage conversions complete:** cdef constrain_avx2, msac symbol_adapt16 AVX2.

**Feature flags:**
- `unchecked` - Use unchecked slice access in SIMD hot paths (skips bounds checks)
- `src/safe_simd/pixel_access.rs` - Helper module for checked/unchecked slice access + SIMD macros

**Writing Clean Safe SIMD (the complete pattern):**

Since Rust 1.93, value-type SIMD intrinsics are safe functions. The only remaining sources of `unsafe` in SIMD are:
1. **Pointer load/store** — `_mm256_loadu_si256(*const)` takes raw pointers
2. **Target feature dispatch** — intrinsics are only safe inside `#[target_feature(enable = "...")]` fns

Both are solved without any `unsafe` in user code:

```rust
// 1. Module header — forbid unsafe (load/store macros handle it internally):
#![cfg_attr(not(feature = "unchecked"), forbid(unsafe_code))]

// 2. Import macros from pixel_access:
use super::pixel_access::{loadu_256, storeu_256, load_256, store_256};

// 3. Functions take SLICES, not raw pointers:
// 4. Use #[arcane] for target_feature dispatch (NOT manual #[target_feature]):
#[arcane]
fn process(token: Desktop64, dst: &mut [u8], src: &[u8], w: usize) {
    // Load 32 bytes from slice — safe, bounds-checked:
    let v = load_256!(&src[0..32], [u8; 32]);

    // All computation intrinsics are safe (Rust 1.93+):
    let doubled = _mm256_add_epi8(v, v);
    let shuffled = _mm256_shuffle_epi8(doubled, _mm256_setzero_si256());

    // Store 32 bytes to slice — safe, bounds-checked:
    store_256!(&mut dst[0..32], [u8; 32], shuffled);

    // Or use typed array ref forms (no slice→array conversion):
    let arr: &[u8; 32] = src[0..32].try_into().unwrap();
    let v = loadu_256!(arr);
    storeu_256!(<&mut [u8; 32]>::try_from(&mut dst[0..32]).unwrap(), v);
}
```

**Why this works with `forbid(unsafe_code)`:**
- `#[arcane]` (from archmage crate) handles `#[target_feature]` dispatch via tokens — no manual feature annotations needed
- `load_256!`/`store_256!` expand to `safe_unaligned_simd` calls (safe, bounds-checked) when `unchecked` is off
- Computation intrinsics (`_mm256_add_epi8`, `_mm256_shuffle_epi8`, etc.) are plain safe functions since Rust 1.93
- Result: **zero `unsafe` blocks** in the SIMD function body

**When `unchecked` is ON:** macros expand to `unsafe { _mm256_loadu_si256(ptr) }` with `debug_assert!` only — maximum perf, `deny(unsafe_code)` instead of `forbid`.

**Load/Store macros (in `pixel_access.rs`):**

| Macro | Width | Input | Description |
|-------|-------|-------|-------------|
| `loadu_256!(ref)` | 256 | `&[T; N]` | Load from typed array ref |
| `storeu_256!(ref, v)` | 256 | `&mut [T; N]` | Store to typed array ref |
| `load_256!(slice, T)` | 256 | `&[T]` | Load from slice (auto-converts to `&[T; N]`) |
| `store_256!(slice, T, v)` | 256 | `&mut [T]` | Store to slice (auto-converts to `&mut [T; N]`) |
| `loadu_128!` / `storeu_128!` | 128 | `&[T; N]` | SSE typed-ref variants |
| `load_128!` / `store_128!` | 128 | `&[T]` | SSE from-slice variants |
| `neon_ld1q_u8!` / `neon_st1q_u8!` | 128 | `&[u8; 16]` | aarch64 NEON u8 |
| `neon_ld1q_u16!` / `neon_st1q_u16!` | 128 | `&[u16; 8]` | aarch64 NEON u16 |
| `neon_ld1q_s16!` / `neon_st1q_s16!` | 128 | `&[i16; 8]` | aarch64 NEON i16 |

**Slice access helpers (in `pixel_access.rs`):**

| Helper | Description |
|--------|-------------|
| `row_slice(buf, off, len)` | Immutable `&[u8]` — unchecked when feature enabled |
| `row_slice_mut(buf, off, len)` | Mutable `&mut [u8]` — unchecked when feature enabled |
| `row_slice_u16(buf, off, len)` | Immutable `&[u16]` variant |
| `row_slice_u16_mut(buf, off, len)` | Mutable `&mut [u16]` variant |
| `idx(buf, i)` / `idx_mut(buf, i)` | Single element access |
| `reinterpret_slice(src)` | Safe zerocopy type reinterpretation |

**Migration checklist for converting a SIMD function to safe:**
1. Change fn signature: raw pointers → slices (`*mut u8` → `&mut [u8]`)
2. Add `#[arcane]` attribute, take `Desktop64` token param
3. Replace `unsafe { _mm256_loadu_si256(ptr) }` → `load_256!(&slice[off..off+32], [u8; 32])`
4. Replace `unsafe { _mm256_storeu_si256(ptr, v) }` → `store_256!(&mut slice[off..off+32], [u8; 32], v)`
5. Remove `unsafe {}` blocks around computation intrinsics (they're safe since 1.93)
6. Add `#![cfg_attr(not(feature = "unchecked"), forbid(unsafe_code))]` to module
7. Gate FFI `extern "C"` wrappers behind `#[cfg(feature = "asm")]`

**Unsafe reduction progress (safe_simd/):**
- **Pixels trait gated behind cfg(asm)** — the unsound `Pixels` trait (which returned `*mut u8` from `&self`, bypassing DisjointMut's borrow tracker) is now dead code when asm is disabled
- All safe_simd dispatch functions use tracked DisjointMut guards (full_guard_mut/full_guard) instead of Pixels
- ipred.rs: All 28 inner SIMD fns converted from raw pointers to safe slices
- cdef.rs: Padding functions converted to safe DisjointMut slice access (top/bottom)
- cdef_arm.rs: Same conversion for ARM NEON path
- loopfilter.rs: Dispatch function fully safe (lvl + dst via slice access)
- lf_apply.rs: Bounds checks use assert on pixel_len instead of Pixels
- Remaining: Inner SIMD functions in mc, itx, filmgrain_arm, loopfilter_arm still take raw pointers (derived from tracked guards)

**c-ffi decoupled from fn-ptr dispatch.** The `c-ffi` feature now only controls the 19 `dav1d_*` extern "C" entry points in `src/lib.rs`. Internal DSP dispatch uses direct function calls (no function pointers) when `asm` is disabled.

## Managed Safe API

**Location:** `src/managed.rs` (~970 lines, 100% safe Rust)

A fully safe, zero-copy API for decoding AV1 video. Enforced by `#![forbid(unsafe_code)]`.

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

### ✅ RESOLVED: Thread Cleanup and Joining

**Status:** ✅ **FIXED** (Commit 2e49d9c)

Fixed architecture flaw where worker thread JoinHandles were stored inside Arc<Rav1dContext>, creating circular ownership that prevented proper thread cleanup.

**Solution:** Moved JoinHandles out of Arc and into Decoder struct. Decoder::drop() now signals workers to die and joins them synchronously.

**Verification:**
- All thread cleanup tests pass (run with `--test-threads=1`)
- No deadlocks
- No thread leaks
- Proper panic propagation

See THREAD_FIX_COMPLETE.md for full implementation details.

### ✅ RESOLVED: Panic Safety and Memory Management

**Status:** ✅ **VERIFIED SAFE**

The managed API (`src/managed.rs`) uses the safe `Rav1dData` wrapper with `CArc<[u8]>` (Arc-based smart pointer), not the unsafe `Dav1dData` C FFI struct. The implementation is panic-safe:

1. **Automatic cleanup via RAII**: `Rav1dData` contains `Option<CArc<[u8]>>` which properly implements Drop through Arc's reference counting
2. **Panic safety verified**: Stack unwinding correctly drops `Rav1dData`, cleaning up resources even on panic
3. **No manual memory management**: The managed API never calls `dav1d_data_wrap`/`dav1d_data_unref` directly

**Testing:**
- `tests/panic_safety_test.rs` - 4 tests verifying panic safety and proper Drop behavior
- All tests pass under normal operation and panic conditions
- Memory leak detection via ASAN/LSAN can be added to CI for additional verification

**Note:** The unsafe `Dav1dData` C FFI struct (used when `feature = "c-ffi"` is enabled) does NOT implement Drop and could leak on panic. However, this is not used by the managed API and only affects direct C FFI users who must manage `dav1d_data_unref` manually.

### Recommended: Memory Leak Detection in CI

**Status:** ⚠️ Enhancement

While the managed API is structurally sound, adding ASAN/LSAN to CI would provide additional confidence:

**Justfile additions:**
```bash
# Run tests with AddressSanitizer
test-asan:
    RUSTFLAGS="-Z sanitizer=address" cargo +nightly test --no-default-features --features "bitdepth_8,bitdepth_16" --target x86_64-unknown-linux-gnu

# Run tests with LeakSanitizer
test-lsan:
    RUSTFLAGS="-Z sanitizer=leak" cargo +nightly test --no-default-features --features "bitdepth_8,bitdepth_16" --target x86_64-unknown-linux-gnu
```

**CI workflow addition:**
```yaml
- name: Run tests with ASAN
  run: |
    rustup toolchain install nightly
    RUSTFLAGS="-Z sanitizer=address" cargo +nightly test --no-default-features --features "bitdepth_8,bitdepth_16" --target x86_64-unknown-linux-gnu
```

### Recommended: Thread Pool Cleanup Verification

**Status:** ℹ️ Low Priority

The `Rav1dContext` manages a thread pool for frame threading. While the Drop implementation appears correct, explicit verification would be valuable:

**Areas to verify:**
- `Arc<TaskThreadData>` drop implementation in `src/internal.rs`
- Worker threads join properly on context drop
- No hanging threads or leaked thread handles

**Test approach:**
```rust
#[test]
fn test_decoder_thread_cleanup() {
    let initial_threads = thread_count();
    {
        let mut decoder = Decoder::with_settings(Settings {
            threads: 0, // Auto-detect cores
            ..Default::default()
        }).unwrap();
        decoder.decode(test_data).unwrap();
    }
    // Give OS time to clean up threads
    thread::sleep(Duration::from_millis(100));
    let final_threads = thread_count();
    assert_eq!(initial_threads, final_threads);
}
```



## Safety Levels (Progressive Trust Model)

rav1d-safe uses a progressive safety model where the default is maximum safety, and features opt-in to less safety for more performance.

### Level 0: Default (forbid_unsafe) - Maximum Safety

**Build:** `cargo build --no-default-features --features "bitdepth_8,bitdepth_16"`

**Guarantees:**
- ✅ `#![forbid(unsafe_code)]` - Compiler-enforced, zero unsafe
- ✅ Single-threaded only (no Arc, no threading primitives)
- ✅ Safe-only data structures
- ✅ Bounds-checked slice access
- ✅ Ideal for: audit, fuzzing, correctness verification

**Performance:** Slowest, but provably safe

**Code Structure:**
- Use `cfg(not(feature = "quite-safe"))` for safe-only alternatives
- Example: Use `Rc` instead of `Arc`, `RefCell` instead of `Mutex`

### Level 1: quite-safe - Sound Abstractions

**Build:** Add `--features quite-safe`

**Allows:**
- ✅ Sound abstractions with unsafe internals (`Arc`, `Mutex`, `AtomicU32`)
- ✅ Multi-threading via standard library primitives
- ✅ Still bounds-checked slice access
- ⚠️ Uses `#![deny(unsafe_code)]` where possible
- ⚠️ Allows `#[allow(unsafe_code)]` for sound abstractions only

**Performance:** Good, multi-threaded

**Code Structure:**
- Mark sound abstractions with `#[cfg(feature = "quite-safe")]`
- Document safety invariants for each abstraction
- Keep unsafe code isolated in minimal wrapper modules

### Level 2: unchecked - Performance Mode

**Build:** Add `--features unchecked`

**Allows:**
- ✅ Everything from quite-safe
- ⚠️ Unchecked slice access (`get_unchecked`) in hot paths
- ⚠️ Still maintains safety invariants (debug_assert! checks)

**Performance:** Near-maximum, zero bounds checking overhead

**Code Structure:**
- Use `pixel_access::row_slice()` helpers (conditional on unchecked)
- Always include `debug_assert!` for bounds validation
- Example:
  ```rust
  #[cfg(feature = "unchecked")]
  unsafe { slice.get_unchecked(idx) }
  
  #[cfg(not(feature = "unchecked"))]
  &slice[idx]
  ```

### Level 3: c-ffi - C API Compatibility

**Build:** Add `--features c-ffi`

**Allows:**
- ✅ Everything from unchecked
- ⚠️ `unsafe extern "C"` FFI functions
- ⚠️ Raw pointer conversions at FFI boundary
- ⚠️ C ABI compatibility layer

**Performance:** Maximum (FFI overhead minimal)

**Code Structure:**
- Gate all FFI code with `#[cfg(feature = "c-ffi")]`
- FFI wrappers convert pointers → slices, then call safe inner functions
- Example:
  ```rust
  #[cfg(feature = "c-ffi")]
  pub unsafe extern "C" fn dav1d_filter(
      dst: *mut u8, 
      len: usize
  ) {
      let dst = unsafe { slice::from_raw_parts_mut(dst, len) };
      filter_inner(dst); // Safe function
  }
  ```

### Level 4: asm - Hand-Written Assembly

**Build:** Add `--features asm`

**Allows:**
- ✅ Everything from c-ffi
- ⚠️ Hand-written x86_64/aarch64 assembly
- ⚠️ Function pointer dispatch to asm functions

**Performance:** Maximum (assembly hot paths)

**Code Structure:**
- Gate all asm code with `#[cfg(feature = "asm")]`
- Assembly functions must have safe Rust equivalents for testing
- Document asm safety invariants clearly

### Feature Dependency Chain

```
forbid_unsafe (default)
  └─> quite-safe (enables threading + sound abstractions)
       └─> unchecked (enables unchecked slice access)
            └─> c-ffi (enables C API)
                 └─> asm (enables hand-written assembly)
```

**Cargo.toml:**
```toml
[features]
quite-safe = []
unchecked = ["quite-safe"]
c-ffi = ["unchecked"]
asm = ["c-ffi"]
```

### Audit Guidance

**For auditors reviewing safety:**

1. **Start with forbid_unsafe mode** - Verify it builds and tests pass
2. **Review quite-safe additions** - Check each unsafe abstraction is sound
3. **Review unchecked additions** - Verify bounds are checked in debug
4. **Review c-ffi layer** - Ensure pointer conversions are correct
5. **Review asm (optional)** - Compare against safe Rust equivalent

**Code organization for easy audit:**

```
src/
  managed.rs         - #![forbid(unsafe_code)] always
  lib.rs            - Core decoder logic
  internal.rs       - Data structures
    #[cfg(not(feature = "quite-safe"))]
    - Use Rc, RefCell, no threading
    #[cfg(feature = "quite-safe")]
    - Use Arc, Mutex, threading
  
  safe_simd/        - SIMD implementations
    mc.rs           - Motion compensation
      fn mc_inner(dst: &mut [u8], ...) { ... }  // Safe
      
      #[cfg(feature = "c-ffi")]
      unsafe extern "C" fn mc_ffi(...) { ... }  // FFI wrapper
```

### Migration Checklist

To migrate a module to this architecture:

- [ ] Add `#![cfg_attr(not(feature = "quite-safe"), forbid(unsafe_code))]` at top
- [ ] Move threading primitives behind `#[cfg(feature = "quite-safe")]`
- [ ] Change raw pointer params to slices in inner functions
- [ ] Use `pixel_access` helpers for slice access (respects `unchecked`)
- [ ] Gate FFI wrappers with `#[cfg(feature = "c-ffi")]`
- [ ] Document safety invariants for any unsafe code
- [ ] Test that module builds in all safety levels

### Current Status

**Modules fully migrated:** 
- `src/managed.rs` - already `#![forbid(unsafe_code)]`
- (others TBD)

**Work remaining:** ~20 safe_simd modules need migration


### Implementation Status

**Currently:** The codebase still uses Arc/Mutex/threading unconditionally, so `quite-safe` is effectively required to build. The default forbid_unsafe mode will fail to compile.

**Target:** Conditionally compile single-threaded safe alternatives (Rc/RefCell) for default mode.

**Example migration needed:**
```rust
// In src/internal.rs
#[cfg(feature = "quite-safe")]
use std::sync::{Arc, Mutex};
#[cfg(not(feature = "quite-safe"))]
use std::{rc::Rc as Arc, cell::RefCell as Mutex};

// Decoder context switches between threaded and non-threaded versions
```

This is significant work - likely a week to fully migrate all threading code to be conditional.

For now, use `quite-safe` feature as the practical default until migration is complete.

