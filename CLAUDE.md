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

## Porting Progress

### Motion Compensation (mc)
- [x] `avg_8bpc_avx2` - Average two buffers (8-bit)
- [x] `avg_16bpc_avx2` - Average two buffers (16-bit, scalar impl)
- [x] `avg_8bpc_sse4` - SSE4 fallback (uses scalar)
- [x] `w_avg_8bpc_avx2` - Weighted average (8-bit)
- [x] `w_avg_16bpc_avx2` - Weighted average (16-bit, scalar impl)
- [x] `mask_8bpc_avx2` - Per-pixel masked blend (8-bit, scalar impl)
- [x] `mask_16bpc_avx2` - Per-pixel masked blend (16-bit, scalar impl)
- [x] `blend_8bpc_avx2` - Pixel blend with per-pixel mask (scalar impl)
- [x] `blend_16bpc_avx2` - Pixel blend (16-bit, scalar impl)
- [x] `blend_v_8bpc_avx2` - Vertical OBMC blend (scalar impl)
- [x] `blend_v_16bpc_avx2` - Vertical OBMC blend (16-bit, scalar impl)
- [x] `blend_h_8bpc_avx2` - Horizontal OBMC blend (scalar impl)
- [x] `blend_h_16bpc_avx2` - Horizontal OBMC blend (16-bit, scalar impl)
- [ ] `mc` (8tap filters) - 10 filter variants
- [ ] `mct` (prep) - 10 filter variants
- ... (45k lines total)

### Other Categories
- [ ] itx - Inverse transforms (42k lines)
- [ ] ipred - Intra prediction (26k lines)
- [ ] looprestoration - SGR/Wiener (17k lines)
- [ ] filmgrain - Film grain synthesis (13k lines)
- [ ] loopfilter - Deblocking (9k lines)
- [ ] cdef - Directional enhancement (7k lines)

## Testing Strategy

1. **Brute-force**: Test edge cases (0, 1, max, min, boundaries)
2. **Differential**: Compare safe-simd vs asm output
3. **Fuzzing**: Random inputs, check for panics

## Known Issues

(none yet)

## Investigation Notes

(none yet)
