# Context Handoff - rav1d-safe SIMD Porting

**Date:** 2026-02-04
**Project:** rav1d-safe - Safe SIMD fork of rav1d

## Primary Goal

Port 160k lines of hand-written x86/ARM assembly to safe Rust intrinsics. The user's directive (repeated 10x in MEMORY.md): "CONTINUE PORTING ALL 160K LINES OF RAV1D ASM TO SAFE RUST, DO NOT STOP"

## Current State

### Completed SIMD Implementations (src/safe_simd/mc.rs)

All functions below have true AVX2 SIMD using `#[target_feature(enable = "avx2")]`:

**8-bit pixel depth (8bpc):**
- `avg_8bpc_avx2` - Average two buffers
- `w_avg_8bpc_avx2` - Weighted average
- `mask_8bpc_avx2` - Per-pixel masked blend (32-bit arithmetic to avoid overflow)
- `blend_8bpc_avx2` - Pixel blend
- `blend_v_8bpc_avx2` - Vertical OBMC blend
- `blend_h_8bpc_avx2` - Horizontal OBMC blend

**16-bit pixel depth (16bpc) - all use 32-bit arithmetic:**
- `avg_16bpc_avx2` - rnd=16400, sh=5
- `w_avg_16bpc_avx2` - rnd=131200, sh=8
- `mask_16bpc_avx2` - rnd=524800, sh=10
- `blend_16bpc_avx2`, `blend_v_16bpc_avx2`, `blend_h_16bpc_avx2`

**8-tap filter helpers (not yet in dispatch):**
- `h_filter_8tap_8bpc_avx2()` - Horizontal filter using `_mm256_maddubs_epi16`
- `v_filter_8tap_8bpc_avx2()` - Vertical filter with 32-bit arithmetic
- `get_filter()` - Access coefficients from `dav1d_mc_subpel_filters` table

### Dispatch Pattern

Located in `src/mc.rs`:
```rust
#[cfg(all(feature = "safe-simd", not(feature = "asm"), any(target_arch = "x86", target_arch = "x86_64")))]
fn init_x86_safe_simd<BD: BitDepth>(mut self, flags: CpuFlags) -> Self {
    // Uses decl_fn_safe! macro to wrap safe_simd functions
}
```

### Performance

Benchmark (20 decodes of test.avif):
- asm: ~1.16s
- safe-simd: ~1.31s (~11% slower)

The gap is primarily due to pure Rust fallbacks for 8-tap filters, itx, ipred.

## Next Steps (Priority Order)

### 1. Complete 8-tap Filter Implementation (Highest Impact)

The helpers exist but need public entry points:

```rust
// Need to create these matching asm naming:
pub unsafe extern "C" fn put_8tap_regular_8bpc_avx2(...)
pub unsafe extern "C" fn put_8tap_smooth_8bpc_avx2(...)
// etc. for all 9 filter combinations × put/prep × 2 bitdepths
```

Function signature from `wrap_fn_ptr!(pub unsafe extern "C" fn mc(...)`:
```rust
dst_ptr: *mut DynPixel,
dst_stride: isize,
src_ptr: *const DynPixel,
src_stride: isize,
w: i32,
h: i32,
mx: i32,  // subpixel x position (0-15)
my: i32,  // subpixel y position (0-15)
bitdepth_max: i32,
_dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
_src: *const FFISafe<Rav1dPictureDataComponentOffset>,
```

The 8-tap implementation has 4 cases:
1. Both H and V filters (mx!=0, my!=0): 2-pass through intermediate buffer
2. H only (mx!=0, my==0): Single horizontal pass
3. V only (mx==0, my!=0): Single vertical pass
4. Neither (mx==0, my==0): Simple copy

### 2. Hook into Dispatch Table

Add to `init_x86_safe_simd()` in `src/mc.rs`:
```rust
self.mc[Filter2d::Regular8Tap] = mc::decl_fn_safe!(safe_mc::put_8tap_regular_8bpc_avx2);
// etc.
```

### 3. Other High-Impact Targets

| Module | ASM Lines | Impact |
|--------|-----------|--------|
| itx | ~42k | Inverse transforms - heavily used |
| ipred | ~26k | Intra prediction |
| looprestoration | ~17k | SGR/Wiener filters |
| filmgrain | ~13k | Film grain synthesis |
| loopfilter | ~9k | Deblocking |
| cdef | ~7k | Directional enhancement |

## Key Technical Patterns

### Important Constants
- 8bpc: `intermediate_bits = 4`, PREP_BIAS = 0
- 16bpc: `intermediate_bits = 4`, PREP_BIAS = 8192
- pmulhrsw rounding: `(a * b + 16384) >> 15`
- Access Align16 inner array with `.0` (e.g., `dav1d_obmc_masks.0[w..]`)

### Crate Configuration
- `#![deny(unsafe_op_in_unsafe_fn)]` - need explicit unsafe blocks
- Variable shifts use `_mm256_sra_epi16/32` with `_mm_cvtsi32_si128(sh)` instead of `_mm256_srai_epi16/32`

### Build Commands
```bash
# Build safe-simd
cargo build --no-default-features --features "bitdepth_8,bitdepth_16,safe-simd" --release

# Run tests
cargo test --no-default-features --features "bitdepth_8,bitdepth_16,safe-simd" --release

# Benchmark (in zenavif directory)
/tmp/bench3.sh
```

## Recent Commits

```
92ff6dd docs: update progress and add 8-tap filter plan
674d4b4 wip: add 8-tap filter helper functions for future SIMD implementation
78c9425 feat: implement true AVX2 SIMD for 16bpc functions
7a13904 docs: update blend_v/blend_h status to true SIMD
ed96072 feat: implement true AVX2 SIMD for blend_v and blend_h
```

## Files to Read

1. `/home/lilith/work/rav1d-safe/CLAUDE.md` - Project instructions and progress
2. `/home/lilith/work/rav1d-safe/src/safe_simd/mc.rs` - SIMD implementations
3. `/home/lilith/work/rav1d-safe/src/mc.rs` - Dispatch table and Rust fallbacks
4. `/home/lilith/.claude/projects/-home-lilith/memory/MEMORY.md` - Global memory with directive

## Delete This File

After loading into new session, delete this file:
```bash
rm /home/lilith/work/rav1d-safe/CONTEXT-HANDOFF.md
```
