# rav1d-safe

Safe SIMD fork of rav1d - replacing 160k lines of hand-written assembly with safe Rust intrinsics.

## Quick Commands

```bash
# Build without asm (pure Rust, uses safe-simd implementations)
cargo build --no-default-features --features "bitdepth_8,bitdepth_16,safe-simd" --release

# Build with asm (original rav1d behavior)
cargo build --features "asm,bitdepth_8,bitdepth_16" --release

# Run tests
cargo test --no-default-features --features "bitdepth_8,bitdepth_16,safe-simd" --release

# Check both configurations compile
cargo check --no-default-features --features "bitdepth_8,bitdepth_16,safe-simd"
cargo check --features "asm,bitdepth_8,bitdepth_16"
```

## Feature Flags

- `asm` - Use hand-written assembly (default, original rav1d)
- `safe-simd` - Use safe Rust SIMD implementations via intrinsics
- `bitdepth_8` - 8-bit pixel support
- `bitdepth_16` - 10/12-bit pixel support

## Architecture

### Dispatch Pattern

rav1d uses function pointer dispatch for SIMD:
1. `wrap_fn_ptr!` macro creates type-safe function pointer wrappers
2. `init_x86(CpuFlags)` populates dispatch table based on detected features
3. `bd_fn!` macro links to asm symbols (e.g., `dav1d_avg_8bpc_avx2`)

For safe-simd:
1. `decl_fn_safe!` macro wraps our Rust functions
2. `#[cfg(feature = "safe-simd")]` in `init_x86` uses our implementations
3. `#[target_feature(enable = "avx2")]` enables SIMD codegen

### Files

- `src/safe_simd/mod.rs` - Safe SIMD module root
- `src/safe_simd/mc.rs` - Motion compensation functions
- `src/mc.rs` - Dispatch table (modified to use safe-simd when enabled)
- `src/wrap_fn_ptr.rs` - Added `decl_fn_safe!` macro

## How safe-simd Works

When built with `--features safe-simd` (without `asm`):

1. **SIMD-optimized functions** (in `src/safe_simd/mc.rs`):
   - avg, w_avg, mask, blend, blend_v, blend_h
   - These use AVX2 intrinsics via `#[target_feature(enable = "avx2")]`

2. **Pure Rust fallbacks** (in respective DSP files):
   - mc/mct 8tap filters → `put_8tap_rust`, `prep_8tap_rust`
   - itx → `inv_txfm_add_rust` (uses `itx_1d.rs` 1D transforms)
   - ipred → `ipred_*_rust` functions
   - loopfilter, looprestoration, cdef, filmgrain → all have Rust fallbacks

The dispatch path: `Rav1dMCDSPContext::new()` → `init()` → `init_x86_safe_simd()` (for safe-simd)

## Porting Progress

### Motion Compensation (mc) - src/safe_simd/mc.rs

**SIMD Optimized (using AVX2 intrinsics):**
- [x] `avg_8bpc_avx2` - Average two buffers (true SIMD)
- [x] `avg_16bpc_avx2` - 16-bit average (true SIMD with 32-bit arithmetic)
- [x] `w_avg_8bpc_avx2` - Weighted average (true SIMD)
- [x] `w_avg_16bpc_avx2` - 16-bit weighted average (true SIMD)
- [x] `mask_8bpc_avx2` - Per-pixel masked blend (true SIMD)
- [x] `mask_16bpc_avx2` - 16-bit masked blend (true SIMD)
- [x] `blend_8bpc_avx2` - Pixel blend (true SIMD)
- [x] `blend_16bpc_avx2` - 16-bit blend (true SIMD)
- [x] `blend_v_8bpc` - Vertical OBMC blend (true SIMD)
- [x] `blend_v_16bpc` - Vertical OBMC blend (true SIMD)
- [x] `blend_h_8bpc` - Horizontal OBMC blend (true SIMD)
- [x] `blend_h_16bpc` - Horizontal OBMC blend (true SIMD)

**8-tap Filter (mc/mct):**
- [x] `put_8tap_*_8bpc_avx2` - All 9 filter variants for 8bpc (SIMD)
- [x] `prep_8tap_*_8bpc_avx2` - All 9 filter variants for 8bpc (SIMD)
- [x] `put_8tap_*_16bpc_avx2` - All 9 filter variants for 16bpc (SIMD for H+V, scalar for H/V-only)
- [x] `prep_8tap_*_16bpc_avx2` - All 9 filter variants for 16bpc (SIMD for H+V, scalar for H/V-only)
- [x] `h_filter_8tap_8bpc_avx2` - Horizontal 8-tap filter using maddubs
- [x] `v_filter_8tap_8bpc_avx2` - Vertical 8-tap filter with 32-bit arithmetic
- [x] `h_filter_8tap_16bpc_avx2` - Horizontal 8-tap filter for 16-bit pixels (madd_epi16)
- [x] `v_filter_8tap_16bpc_avx2` - Vertical 8-tap filter for 16bpc put (mullo_epi32)
- [x] `v_filter_8tap_16bpc_prep_avx2` - Vertical 8-tap filter for 16bpc prep (with PREP_BIAS)
- [x] `get_filter_coeff()` - Access filter coefficients from tables

**Bilinear Filter (mc/mct):**
- [x] `put_bilin_8bpc_avx2` - Bilinear put for 8bpc (AVX2 SIMD with maddubs)
- [x] `prep_bilin_8bpc_avx2` - Bilinear prep for 8bpc (AVX2 SIMD)
- [x] `put_bilin_16bpc_avx2` - Bilinear put for 16bpc (SIMD for H+V, scalar for H/V-only)
- [x] `prep_bilin_16bpc_avx2` - Bilinear prep for 16bpc (SIMD for H+V, scalar for H/V-only)
- [x] `h_bilin_16bpc_avx2` - Horizontal bilinear filter for 16bpc (cvtepu16_epi32)
- [x] `v_bilin_16bpc_avx2` - Vertical bilinear filter for 16bpc put
- [x] `v_bilin_16bpc_prep_avx2` - Vertical bilinear filter for 16bpc prep (with PREP_BIAS)

**Weighted Mask (w_mask):**
- [x] `w_mask_444_8bpc_avx2` - 4:4:4 (no subsampling)
- [x] `w_mask_422_8bpc_avx2` - 4:2:2 (horizontal subsampling)
- [x] `w_mask_420_8bpc_avx2` - 4:2:0 (horizontal + vertical subsampling)
- [x] `w_mask_*_16bpc_avx2` - All 3 variants for 16bpc (scalar, not SIMD yet)

**Using Pure Rust Fallbacks:**
- [ ] `mc_scaled` - 10 scaled variants per bitdepth
- [ ] `mct_scaled` - 10 scaled prep variants per bitdepth
- [ ] `warp8x8` / `warp8x8t` - Affine warping
- [ ] `emu_edge` - Edge extension
- [ ] `resize` - Resampling

### Other DSP Categories (all use pure Rust fallbacks)

| Module | ASM Lines | Status |
|--------|-----------|--------|
| itx | ~42k | Pure Rust fallback (`itx_1d.rs` + `itx.rs`) |
| ipred | ~26k | Pure Rust fallback |
| looprestoration | ~17k | Pure Rust fallback |
| filmgrain | ~13k | Pure Rust fallback |
| loopfilter | ~9k | Pure Rust fallback |
| cdef | ~7k | Pure Rust fallback |

**Note:** The pure Rust fallbacks produce correct output (byte-for-byte identical to asm).
They're just slower than SIMD-optimized versions.

## Testing Strategy

1. **Brute-force**: Test edge cases (0, 1, max, min, boundaries)
2. **Differential**: Compare safe-simd vs asm output
3. **Fuzzing**: Random inputs, check for panics

## Known Issues

(none yet)

## Investigation Notes
### 8-tap Filter - COMPLETED

All 8-tap filter variants are now implemented with AVX2 SIMD:
- 8bpc: Full SIMD for H+V, H-only, V-only, and copy cases
- 16bpc: Full SIMD for H+V, H-only, V-only cases (all using AVX2 intrinsics)

### Bilinear Filter - COMPLETED

All bilinear filter variants now use AVX2 SIMD:
- 8bpc: H+V, H-only, V-only, copy
- 16bpc: H+V, H-only, V-only, copy

### Performance Notes (2026-02-04)
Full-stack benchmark via zenavif (20 decodes of test.avif):
- asm: ~1.29s (64.5ms per decode)
- safe-simd: ~1.28s (64ms per decode)
- safe-simd is now ~1% FASTER than asm!

MC is fully optimized with AVX2 SIMD for all 8-tap and bilinear cases.

Remaining DSP modules using Rust fallbacks:
- ITX (inverse transforms) - ~42k asm lines
- Loopfilter - ~9k asm lines
- CDEF - ~7k asm lines
- Looprestoration - ~17k asm lines


### Remaining MC Functions (using Rust fallbacks)

- `mc_scaled`/`mct_scaled` - Variable dx/dy strides per sample
- `warp8x8`/`warp8x8t` - Per-pixel filter selection
- `emu_edge` - Edge extension
- `resize` - 8-tap resampling with variable phase
- `w_mask_16bpc` - Still scalar (8bpc has SIMD)

### Next Targets

Other DSP modules for SIMD optimization:
- CDEF (~7k asm lines) - direction-based filtering, started constrain helper
- Loopfilter (~9k asm lines) - deblocking filter
- ITX (~42k asm lines) - inverse transforms (largest)
