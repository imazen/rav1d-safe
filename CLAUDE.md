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
6. filmgrain (~13k lines) - scaffolding exists but fallback is faster

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
| filmgrain | `src/safe_simd/filmgrain.rs` | **Scaffolding** - Not wired (slower than fallback) |

### ARM aarch64 (NEON)

| Module | Location | Status |
|--------|----------|--------|
| mc_arm | `src/safe_simd/mc_arm.rs` | **Complete** - 8bpc+16bpc (all MC functions including 8tap) |
| ipred_arm | `src/safe_simd/ipred_arm.rs` | **Complete** - DC/V/H/paeth/smooth modes (8bpc + 16bpc) |
| cdef_arm | `src/safe_simd/cdef_arm.rs` | **Complete** - All filter sizes (8bpc + 16bpc) |
| loopfilter_arm | `src/safe_simd/loopfilter_arm.rs` | **Complete** - Y/UV H/V filters (8bpc + 16bpc) |
| looprestoration_arm | `src/safe_simd/looprestoration_arm.rs` | **Complete** - Wiener + SGR (5x5, 3x3, mix) 8bpc + 16bpc |
| itx_arm | `src/safe_simd/itx_arm.rs` | **Complete** - 334 FFI functions (90 handwritten + 244 macro-generated), 320 dispatch entries matching x86 |

## Performance Status (2026-02-04)

Full-stack benchmark via zenavif (20 decodes of test.avif):
- ASM: ~1.17s
- Safe-SIMD: ~1.18s
- **Safe-SIMD matches ASM performance**

## Porting Progress (160k lines target)

**SIMD optimized (~28k lines in safe_simd/):**
- MC x86 module (~5k lines): Complete (8bpc + 16bpc)
- MC ARM module (~3.9k lines): Complete (8bpc + 16bpc all filters including 8tap)
- ITX x86 module (~12k lines): **100% complete** (160 transforms each 8bpc/16bpc, 320 dispatch entries)
- ITX ARM module (~6k lines): **100% complete** (334 FFI functions, 320 dispatch entries matching x86)
  - Generic transform engine: composes 1D transforms from itx_1d.rs
  - Macro-generated FFI wrappers for all transform type/size combinations
- Loopfilter (~9k lines): Complete (8bpc + 16bpc)
- CDEF (~7k lines): Complete (8bpc + 16bpc)
- Looprestoration (~17k lines): Complete (Wiener + SGR 8bpc + 16bpc)
- ipred (~26k lines): Complete (all 14 modes, 8bpc + 16bpc)

**Using Rust fallbacks (no SIMD needed - performance already matches ASM):**
- filmgrain (~13k lines): Scaffolding exists but fallback is faster
- refmvs (splat_mv, save_tmvs, load_tmvs): Scalar Rust fallback
- pal (pal_idx_finish): Scalar Rust fallback
- msac (symbol_adapt): Scalar Rust fallback

## Architecture

### Dispatch Pattern

rav1d uses function pointer dispatch for SIMD:
1. `wrap_fn_ptr!` macro creates type-safe function pointer wrappers
2. For asm: `bd_fn!` macro links to asm symbols
3. For safe-simd: `decl_fn_safe!` wraps our Rust functions
4. `init_x86_safe_simd` populates dispatch table when asm disabled

### FFI Wrapper Pattern

```rust
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn function_8bpc_avx2(
    // Match exact signature from wrap_fn_ptr! macro
    dst: *const FFISafe<...>,
    // ... other params
) {
    let dst = unsafe { *FFISafe::get(dst) };
    // Call inner implementation
}
```

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
