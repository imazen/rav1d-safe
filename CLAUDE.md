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
- [x] `avg_16bpc_avx2` - 16-bit average (scalar, TODO: SIMD)
- [x] `w_avg_8bpc_avx2` - Weighted average (true SIMD)
- [x] `w_avg_16bpc_avx2` - 16-bit weighted average (scalar)
- [x] `mask_8bpc_avx2` - Per-pixel masked blend (true SIMD)
- [x] `mask_16bpc_avx2` - 16-bit masked blend (scalar)
- [x] `blend_8bpc_avx2` - Pixel blend (true SIMD)
- [x] `blend_16bpc_avx2` - 16-bit blend (scalar)
- [x] `blend_v_8bpc/16bpc` - Vertical OBMC blend (scalar)
- [x] `blend_h_8bpc/16bpc` - Horizontal OBMC blend (scalar)

**Using Pure Rust Fallbacks:**
- [ ] `mc` (8tap filters) - 10 filter variants per bitdepth
- [ ] `mct` (prep) - 10 filter variants per bitdepth
- [ ] `mc_scaled` - 10 scaled variants per bitdepth
- [ ] `mct_scaled` - 10 scaled prep variants per bitdepth
- [ ] `w_mask` - 3 variants (420/422/444)
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

(none yet)
