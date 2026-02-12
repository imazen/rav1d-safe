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

## NO PERFORMANCE WORK UNTIL 100% DECODE PARITY

**DO NOT do any performance optimization until pixel-perfect decode parity is verified.**
**DO NOT do any performance optimization until pixel-perfect decode parity is verified.**
**DO NOT do any performance optimization until pixel-perfect decode parity is verified.**

This means:
1. All 761 dav1d test vectors (8-bit/10-bit/12-bit) must produce MD5 hashes matching the reference values in `test-vectors/dav1d-test-data/*/data/meson.build`
2. The MD5 verification tests must run automatically via `cargo test` (not `#[ignore]`)
3. Any decode mismatch is a **correctness bug** that MUST be fixed before any perf work

**What counts as "performance work":**
- AVX2/NEON SIMD optimization of existing functions
- Branchless/lookup table optimizations
- Memory layout changes for cache efficiency
- BorrowTracker overhead reduction
- Any change whose primary goal is speed rather than correctness

**What is still allowed:**
- Fixing decode correctness bugs (mismatches vs reference MD5)
- Building/improving test infrastructure
- Code cleanup and refactoring for correctness
- Bug fixes discovered by test vectors

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

## Feature Flag Safety Model

**`forbid(unsafe_code)` is ON by default.** When `asm`, `c-ffi`, or `unchecked` are enabled, it drops to `deny` so modules can use `#[allow(unsafe_code)]` on specific items (FFI wrappers, unchecked slice access, etc).

```
Default (no asm, no c-ffi, no unchecked): #![forbid(unsafe_code)]  — NO exceptions
asm, c-ffi, or unchecked enabled:         #![deny(unsafe_code)]    — modules can #[allow]
```

This means: **every `#[allow(unsafe_code)]` in the codebase MUST be gated behind `cfg(feature = "asm")`, `cfg(feature = "c-ffi")`, `cfg(feature = "unchecked")`, or `cfg(target_arch)` that excludes the default build.** If an `#[allow(unsafe_code)]` item compiles in the default build, `forbid` will reject it.

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
| msac | `src/msac.rs` (inline) | **Complete** - adapt4/adapt8/hi_tok SSE2, adapt16 AVX2 |

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
| msac | `src/msac.rs` (inline) | **Complete** - adapt4/adapt8/hi_tok + adapt16 NEON |

## Performance Status (2026-02-09)

Profiling: 39 frames, 8bpc allintra, release-with-debug, perf stat:
- ASM: 156ms (baseline)
- Safe-SIMD (unchecked): 197ms (1.26x vs asm)
- Safe-SIMD (checked): 293ms (1.88x vs asm)

**Bottleneck #1: DisjointMut BorrowTracker** — 40% of checked decode time.
Fixed: `unchecked` feature now skips tracker entirely (was only unlocking unused constructor).

**Bottleneck #2: msac entropy decoder** — 32% of unchecked decode time.
All symbol_adapt functions now have SIMD: adapt4/adapt8/hi_tok use SSE2 (x86) / NEON (ARM),
adapt16 uses AVX2 (x86) / NEON (ARM). Bool functions remain scalar (optimal as-is).

**SIMD modules are NOT the bottleneck** — looprestoration 6.4%, cdef 5.4%, itx 2.2%, ipred 1.8%.

Previous zenavif benchmark (2026-02-05, full-stack AVIF decode):
- ASM: ~1.17s (20 decodes)
- Safe-SIMD: ~1.11s (20 decodes)
- Note: zenavif currently broken (API mismatch), numbers may be stale

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
- symbol_adapt4/adapt8: SSE2 (x86_64) and NEON (aarch64) - parallel CDF comparison via SIMD
- symbol_adapt16: AVX2 (x86_64) and NEON (aarch64) - parallelized CDF probability calc and comparison
- hi_tok: SSE2 (x86_64) and NEON (aarch64) - monolithic loop with inline adapt4 + renorm + refill
- bool functions: Use scalar Rust fallback (pure GPR, SIMD not beneficial)

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

**MILESTONE: `#![forbid(unsafe_code)]` ACHIEVED for default build** (commit b67f378).

The default build (`cargo build --no-default-features --features "bitdepth_8,bitdepth_16"`) compiles
under `#![forbid(unsafe_code)]` — the compiler guarantees zero unsafe in the entire crate when
neither `asm` nor `c-ffi` features are enabled.

All unsafe is now confined to:
- `rav1d-disjoint-mut` sub-crate (PicBuf, Align types, AlignedVec AsMutPtr impls)
- Code gated behind `#[cfg(feature = "asm")]` or `#[cfg(feature = "c-ffi")]`

**How unsafe was eliminated from the default build:**
- `picture.rs`: `Rav1dPictureDataComponentInner = PicBuf` (type alias to disjoint-mut crate type)
- `msac.rs`: `MsacAsmContextBuf { pos: usize, end: usize }` (indices, not pointers)
- `c_box.rs`: No custom Drop without c-ffi → destructurable, no Pin::new_unchecked
- `c_arc.rs`: `Arc<Box<T>>` with safe view slicing (no StableRef, no raw pointers)
- `assume.rs`: Gated behind c-ffi (only used by picture.rs ExternalAsMutPtr)
- `align.rs`: Align types + ExternalAsMutPtr moved to disjoint-mut crate
- `internal.rs`: Send/Sync auto-derived (no manual unsafe impls needed)
- `partial_simd.rs`: Safe `#[target_feature(enable = "sse2")]` wrappers (Rust 1.93+)

**C FFI types gated behind `cfg(feature = "c-ffi")`:**
- `DavdPicture`, `DavdData`, `DavdDataProps`, `DavdUserData`, `DavdSettings`, `DavdLogger` — all gated
- `ITUTT35PayloadPtr`, `Dav1dITUTT35` struct (with `Send`/`Sync` impls) — gated; safe type alias when c-ffi off
- `RawArc`, `RawCArc`, `Dav1dContext`, `arc_into_raw` — gated (raw Arc ptr roundtrip)
- `From<Dav1d*>` / `From<Rav1d*> for Dav1d*` conversions (containing `unsafe { CArc::from_raw }`) — all gated
- Safe picture allocator: per-plane `Vec<u8>` from MemPool, no C callbacks needed
- Fallible allocation: `MemPool::pop_init` returns `Result<Vec, TryReserveError>`, propagated as `Rav1dError::ENOMEM`

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
- **itx.rs: ✅ FULLY SAFE when asm off** — all 85 #[arcane] fns converted from raw pointers to slices, 0 unsafe outside #[cfg(feature = "asm")] FFI wrappers
- **filmgrain.rs: ✅ 0 allows** — all dispatch safe via zerocopy AsBytes/FromBytes
- **pixel_access.rs: ✅ 0 allows** — SliceExt trait + FlexSlice zero-cost wrapper
- **itx_arm.rs: ✅ 0 allows** — all FFI correctly gated behind asm
- **ipred.rs: ✅ 0 allows** — all 28 inner SIMD fns converted to safe slices
- **mc.rs: ✅ FULLY SAFE when asm off** — 29 rite fns converted from raw pointers to slices, 0 unsafe outside FFI wrappers
- mc_arm.rs: 10 allows (FFI gated, inner fns use NEON intrinsics)
- filmgrain_arm.rs: 8 allows (FFI gated, inner fns use NEON)
- loopfilter_arm.rs: 3 allows
- cdef.rs: 2 allows (test module calling #[target_feature] fns)
- refmvs.rs/refmvs_arm.rs: 1 allow each
- ipred_arm.rs: 1 allow
- All safe_simd dispatch functions use tracked DisjointMut guards
- Pixels trait gated behind cfg(asm) — dead code when asm disabled

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

### Decode Parity Testing (IMPLEMENTED)

**Comparison harness:** `/home/lilith/work/zenavif/examples/compare_libavif.rs`

Compares zenavif (rav1d-safe) vs libavif RGB output at multiple CPU feature levels.

**Reference images:** Pre-generated libavif PNGs at `/mnt/v/output/zenavif/libavif-refs/` (3247 files).
Generated via avifdec at `/home/lilith/work/libavif/build/avifdec`.

**Dataset:** 3261 AVIF files at `/mnt/v/datasets/scraping/avif/` (unsplash, google-native, wikimedia, unsplash-scale).

**CPU Feature Level Override:**
- `rav1d_set_cpu_flags_mask(mask)` — global, applies to all safe_simd dispatch
- `Settings { cpu_flags_mask: mask, .. }` — per-decoder in managed API
- `DecoderConfig::new().cpu_flags_mask(mask)` — per-decoder in zenavif
- All safe_simd dispatch functions check `crate::src::cpu::summon_avx2()` which gates on the mask

| Level | Mask | Description |
|-------|------|-------------|
| v3-avx2 | `0xFFFFFFFF` | AVX2 + FMA (default, full SIMD) |
| v2-sse4 | `0b0111` (7) | SSE4.1 only (no AVX2 dispatch) |
| scalar | `0` | No SIMD (pure Rust scalar) |

**Running comparisons:**
```bash
cd /home/lilith/work/zenavif

# All levels (v3, v2, scalar) on full dataset
./target/release/examples/compare_libavif

# Specific level
./target/release/examples/compare_libavif --level v3
./target/release/examples/compare_libavif --level scalar

# Custom directories
./target/release/examples/compare_libavif /path/to/avif/dir /path/to/refs --level all
```

**Reports:** Written to `/mnt/v/output/zenavif/comparison-{level}.txt`

**Note:** Error categories are vs libavif RGB output (YUV→RGB rounding differences expected):
- Exact: 0 error
- Close: max error ≤ 2 (rounding)
- Minor: max error ≤ 10
- Major: max error > 10 (potential bug)

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



## Feature Dependency Chain

```
default:    #![forbid(unsafe_code)] — compiler-enforced, zero unsafe
  └─> unchecked: get_unchecked in hot paths, debug_assert! bounds checks
       └─> c-ffi: unsafe extern "C" FFI wrappers, raw pointer conversions
            └─> asm: hand-written x86_64/aarch64 assembly via function pointers
```

**Cargo.toml:**
```toml
[features]
default = ["bitdepth_8", "bitdepth_16"]
unchecked = ["rav1d-disjoint-mut/unchecked"]
c-ffi = ["unchecked"]
asm = ["c-ffi"]
```

All unsafe in the default build is confined to the `rav1d-disjoint-mut` sub-crate (PicBuf, Align types, AlignedVec AsMutPtr impls). The main crate is provably safe — auditors only need to review the small sub-crate.

