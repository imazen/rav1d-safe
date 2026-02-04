# Context Handoff - rav1d-safe SIMD Porting

**Date:** 2026-02-04
**Task:** Continue porting 160k lines of rav1d hand-written assembly to safe Rust SIMD

## Session Summary

This session completed **loopfilter** and **CDEF** safe-simd implementations for 8bpc, and started **looprestoration**.

## Current State

### Completed This Session
1. **Loopfilter (8bpc)** - DONE and WIRED
   - `src/safe_simd/loopfilter.rs` - 574 lines
   - `src/loopfilter.rs` - added `init_x86_safe_simd` dispatch
   - Functions: `lpf_h_sb_y_8bpc_avx2`, `lpf_v_sb_y_8bpc_avx2`, `lpf_h_sb_uv_8bpc_avx2`, `lpf_v_sb_uv_8bpc_avx2`

2. **CDEF (8bpc)** - DONE and WIRED
   - `src/safe_simd/cdef.rs` - expanded with filter functions
   - `src/cdef.rs` - added `init_x86_safe_simd` dispatch
   - Functions: `cdef_filter_8x8_8bpc_avx2`, `cdef_filter_4x8_8bpc_avx2`, `cdef_filter_4x4_8bpc_avx2`, `cdef_find_dir_8bpc_avx2`

3. **Looprestoration** - IN PROGRESS (NOT WIRED)
   - `src/safe_simd/looprestoration.rs` - ~320 lines
   - Implemented: `padding_8bpc`, `wiener_filter7_8bpc_avx2_inner`, `wiener_filter5_8bpc_avx2_inner`
   - FFI wrappers created but NOT connected to dispatch
   - SGR filters not implemented (use Rust fallback)

### Module Status Table
| Module | ASM Lines | Status |
|--------|-----------|--------|
| mc | ~7k | **SIMD complete** (8bpc + 16bpc, x86 + ARM) |
| itx | ~42k | Partial SIMD (DCT 4x4/8x8/16x16, WHT 4x4, IDTX 4x4) |
| loopfilter | ~9k | **SIMD 8bpc** ✓ |
| cdef | ~7k | **SIMD 8bpc** ✓ |
| looprestoration | ~17k | Wiener helpers (not wired), SGR uses fallback |
| ipred | ~26k | Rust fallback |
| filmgrain | ~13k | Rust fallback |

## Key Files

- `src/safe_simd/mod.rs` - Module declarations
- `src/safe_simd/mc.rs` - Motion compensation (complete)
- `src/safe_simd/itx.rs` - Inverse transforms (partial)
- `src/safe_simd/loopfilter.rs` - Loop filter (8bpc complete)
- `src/safe_simd/cdef.rs` - CDEF (8bpc complete)
- `src/safe_simd/looprestoration.rs` - Looprestoration (in progress)

## Build & Test Commands

```bash
# Build safe-simd (no asm)
cargo build --no-default-features --features "bitdepth_8,bitdepth_16" --release

# Build with asm (for comparison)
cargo build --features "asm,bitdepth_8,bitdepth_16" --release

# Test decode (from zenavif directory)
cd /home/lilith/work/zenavif
touch src/lib.rs && cargo build --release --example decode_avif
./target/release/examples/decode_avif /home/lilith/work/aom-decode/tests/test.avif /tmp/out.png

# Compare outputs
compare -metric AE /tmp/asm.png /tmp/safe.png null:

# Performance benchmark (20 decodes)
time for i in {1..20}; do ./target/release/examples/decode_avif /home/lilith/work/aom-decode/tests/test.avif /dev/null 2>/dev/null; done
```

## Next Steps (Priority Order)

1. **Wire looprestoration Wiener filters to dispatch**
   - Add `init_x86_safe_simd` to `src/looprestoration.rs`
   - Test correctness with decode

2. **Implement SGR filters** (looprestoration)
   - `sgr_filter_5x5`, `sgr_filter_3x3`, `sgr_filter_mix`
   - These involve box sums and guided filtering

3. **Add 16bpc variants** for loopfilter, CDEF, looprestoration

4. **Continue to next modules:**
   - ipred (~26k asm lines) - intra prediction
   - filmgrain (~13k asm lines) - film grain synthesis

## Technical Notes

### Dispatch Pattern
```rust
// In each DSP module (e.g., loopfilter.rs, cdef.rs):
#[cfg(all(not(feature = "asm"), target_arch = "x86_64"))]
const fn init_x86_safe_simd<BD: BitDepth>(mut self, flags: CpuFlags) -> Self {
    use crate::src::safe_simd::module_name as safe_mod;
    
    if !flags.contains(CpuFlags::AVX2) { return self; }
    
    match BD::BPC {
        BPC::BPC8 => {
            self.fn_ptr = function_wrapper::decl_fn_safe!(safe_mod::function_8bpc_avx2);
        }
        _ => {}
    }
    self
}
```

### FFI Wrapper Pattern
```rust
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn function_8bpc_avx2(
    // Match exact signature from wrap_fn_ptr! macro in dispatch module
    dst: *const FFISafe<...>,
    // ... other params
) {
    let dst = unsafe { *FFISafe::get(dst) };
    // Call inner implementation
}
```

### Performance Baseline
- ASM: ~1.11s for 20 test.avif decodes
- Safe-SIMD: ~1.11-1.12s (at parity)

## Recent Commits
```
733ba89 docs: update CLAUDE.md with looprestoration progress
ab51d8c wip: implement looprestoration Wiener filter helpers (8bpc)
4bb2634 wip: add looprestoration safe-simd module stub
2e5e1d9 feat: add CDEF safe-simd implementation (8bpc)
27dd602 feat: add loopfilter safe-simd implementation (8bpc)
```

## Important Reminders

1. **MEMORY.md says:** "CONTINUE PORTING ALL 160K LINES OF RAV1D ASM TO SAFE RUST, DO NOT STOP"

2. **Performance must stay at parity** - always benchmark after wiring new functions

3. **Output must be byte-identical** - use `compare -metric AE` to verify

4. **BitDepth8::new() takes `()`** not a value - it's a unit type for 8bpc

5. **Use `#[target_feature(enable = "avx2")]`** for FFI wrappers (not `#[arcane]`)
