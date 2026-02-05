# Handoff: Refactor to Fully Safe Intrinsics

## Progress (2026-02-04)

### Completed
- **pal.rs** (x86_64): Fully refactored with `#[arcane]` + `safe_unaligned_simd` + `partial_simd`
  - Inner function: **ZERO** unsafe blocks
  - FFI wrapper: 3 unavoidable unsafe blocks (pointer-to-slice, token forge)
  - Verified via `cargo asm`: same instructions (vmovdqu, vmovq, vpmaddubsw, etc.)
  - No performance regression

- **mc_arm.rs** (aarch64): Fully refactored with `#[arcane]` + `safe_unaligned_simd`
  - **ZERO** non-FFI unsafe blocks (down from 39)
  - Only 8 unavoidable `forge_token_dangerously()` in FFI wrappers
  - All NEON memory ops use `safe_unaligned_simd::aarch64`
  - ARM cross-compilation verified

- **safe_unaligned_simd 0.2.3+**: Has full aarch64 NEON support!
  - `vld1_u8`, `vld1_u16`, `vld1_s16`, `vld1_s32`, etc.
  - `vld1q_u8`, `vld1q_u16`, `vld1q_s16`, `vld1q_s32`, etc.
  - Multi-register variants: `vld1_u8_x2`, `vld1_u8_x3`, `vld1_u8_x4`, etc.
  - **partial_simd module is now obsolete** - can be removed

- **partial_simd module**: Legacy wrappers (can be deprecated)
  - x86_64: `mm_loadl_epi64`, `mm_storel_epi64` for 64-bit SSE ops
  - aarch64: Was used before discovering safe_unaligned_simd has NEON support

### Findings for archmage

1. **`safe_unaligned_simd` compiles to identical code**:
   `safe_simd::_mm256_loadu_si256(arr)` → `vmovdqu` (same as raw pointer)

2. **Slice-to-array conversion has no overhead**:
   `slice[..32].try_into().unwrap()` optimizes away in release builds

3. **SOLVED: Created `partial_simd` module** for 64-bit operations:
   - `mm_loadl_epi64(&[u8; 8])` → `vmovq` load
   - `mm_storel_epi64(&mut [u8; 8], val)` → `vmovq` store
   - Uses `Is64BitsUnaligned` sealed trait pattern
   - Safe functions with `#[target_feature]` - callable from `#[arcane]` without `unsafe`
   - Zero overhead verified via `cargo asm`
   - Pattern could be upstreamed to safe_unaligned_simd

## Ultimate Goal

**Pure Rust API** - FFI should be feature-gated, not the default. The end state:
- `default` = Pure Rust SIMD with safe intrinsics
- `feature = "ffi"` = extern "C" wrappers for C/rav1d interop

## Phase 1 Goal

Eliminate most `unsafe` blocks in safe_simd modules by using:
1. **`#[arcane]` from archmage** - Makes value-based intrinsics safe (Rust 1.85+)
2. **`safe_unaligned_simd`** - Makes memory operations safe

## Current State

- **Branch:** `feat/fully-safe-intrinsics`
- **Rust:** 1.93 (supports safe value intrinsics)
- **Dependencies:** archmage 0.4.0, safe_unaligned_simd 0.2.4

### Unsafe Breakdown (before refactor)

| Category | Count | Can Eliminate? |
|----------|-------|----------------|
| SIMD value intrinsics | ~1,500 | ✅ Yes - `#[arcane]` |
| SIMD memory ops | ~1,300 | ✅ Yes - `safe_unaligned_simd` |
| FFI pointer ops | ~2,000 | ❌ No - required for `extern "C"` |
| **Total in safe_simd** | ~3,200 | ~50% reducible |

## Pattern: Before → After

### Before (current pattern in most modules)
```rust
#![allow(unsafe_op_in_unsafe_fn)]

#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn my_func(dst: *mut u8, src: *const u8, w: i32) {
    // All intrinsics implicitly unsafe
    let v = _mm256_loadu_si256(src as *const __m256i);
    let result = _mm256_add_epi8(v, v);
    _mm256_storeu_si256(dst as *mut __m256i, result);
}
```

### After (with archmage + safe_unaligned_simd)
```rust
use archmage::{arcane, Desktop64, SimdToken};
use safe_unaligned_simd::x86_64 as safe_simd;

#[arcane]
fn my_func_inner(_token: Desktop64, dst: &mut [u8], src: &[u8]) {
    // Memory ops use safe_unaligned_simd (SAFE!)
    let src_arr: &[u8; 32] = src[..32].try_into().unwrap();
    let v = safe_simd::_mm256_loadu_si256(src_arr);

    // Value intrinsics are SAFE inside #[arcane]!
    let result = _mm256_add_epi8(v, v);

    // Memory ops use safe_unaligned_simd (SAFE!)
    let dst_arr: &mut [u8; 32] = (&mut dst[..32]).try_into().unwrap();
    safe_simd::_mm256_storeu_si256(dst_arr, result);
}

// FFI wrapper (still unsafe - unavoidable)
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn my_func(dst: *mut u8, src: *const u8, w: i32) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    let dst_slice = unsafe { std::slice::from_raw_parts_mut(dst, w as usize) };
    let src_slice = unsafe { std::slice::from_raw_parts(src, w as usize) };
    my_func_inner(token, dst_slice, src_slice);
}
```

## Refactoring Strategy

### Phase 1: x86_64 Modules (Priority Order by value intrinsic count)

1. ~~**pal.rs** (~140 lines)~~ ✅ DONE
2. **refmvs.rs** (~60 lines) - Few value intrinsics, low priority
3. **filmgrain.rs** (~1k lines) - Medium
4. **cdef.rs** (~800 lines) - Medium
5. **loopfilter.rs** (~1.2k lines) - Medium
6. **looprestoration.rs** (~2k lines) - Large
7. **ipred.rs** (~4k lines) - Large (91 value intrinsics)
8. **mc.rs** (~5k lines) - Large (477 value intrinsics)
9. **itx.rs** (~12k lines) - Largest (500 value intrinsics)

### Phase 2: ARM NEON Modules

Same pattern, using `Arm64` token and `safe_unaligned_simd::aarch64` for memory ops.

1. ~~**mc_arm.rs** (~4k lines)~~ ✅ DONE - Zero non-FFI unsafe blocks
2. **ipred_arm.rs** (~2k lines) - Next priority
3. **cdef_arm.rs** (~800 lines)
4. **loopfilter_arm.rs** (~1k lines)
5. **looprestoration_arm.rs** (~2k lines)
6. **itx_arm.rs** (~6k lines)
7. **filmgrain_arm.rs** (~750 lines)
8. **refmvs_arm.rs** (~50 lines)

## Token Mapping

| Architecture | Token | Features |
|--------------|-------|----------|
| x86_64 AVX2 | `Desktop64` / `X64V3Token` | AVX2 + FMA |
| aarch64 NEON | `Arm64` / `NeonToken` | NEON (always available) |

## Testing After Each Module

```bash
# Build and verify
cargo build --no-default-features --features "bitdepth_8,bitdepth_16" --release

# Cross-compile check for ARM
cargo check --target aarch64-unknown-linux-gnu --no-default-features --features "bitdepth_8,bitdepth_16"

# Check asm output
cargo asm --lib --no-default-features --features "bitdepth_8,bitdepth_16" "function_name"

# Benchmark (should stay ~1.11-1.15s)
cd /home/lilith/work/zenavif && touch src/lib.rs && cargo build --release --example decode_avif
time for i in {1..20}; do ./target/release/examples/decode_avif /home/lilith/work/aom-decode/tests/test.avif /dev/null 2>/dev/null; done
```

## FFI Elimination (Future Work)

The current architecture has an artificial FFI boundary:

```
call site → avg::Fn.call() → converts safe types to pointers
         → calls extern "C" fn pointer
         → implementation converts pointers back to slices
```

**Why it exists:** `wrap_fn_ptr!` macro creates `extern "C"` function types to support
linking external assembly. Even when asm is disabled, the same types are used.

**To fully eliminate FFI:**

1. **Change `wrap_fn_ptr!`** to generate Rust `fn` types when `feature = "asm"` is off
2. **Change function signatures** from raw pointers to `Rav1dPictureDataComponentOffset` etc.
3. **Update all `call()` implementations** to not do pointer conversions
4. **This is a large refactor** touching mc.rs, itx.rs, cdef.rs, ipred.rs, etc.

**Alternative: Bypass dispatch entirely**

For pure Rust builds, replace function pointer dispatch with:
```rust
match backend {
    SimdBackend::Avx2 => mc::avg_avx2::<BD>(dst, tmp1, tmp2, w, h, bd),
    SimdBackend::Neon => mc::avg_neon::<BD>(dst, tmp1, tmp2, w, h, bd),
    SimdBackend::Rust => mc::avg_rust::<BD>(dst, tmp1, tmp2, w, h, bd),
}
```

This would require making the `*_rust` fallback functions `pub(crate)` and creating
new SIMD implementations that take safe types directly.

## Notes

1. **Don't change FFI signatures yet** - Requires architectural refactor
2. **Keep `#![allow(unsafe_op_in_unsafe_fn)]`** at module top for FFI wrappers
3. **Use `forge_token_dangerously()`** in FFI wrappers - we know features are available because dispatch already checked
4. **Benchmark after each module** to ensure no performance regression
5. **Commit incrementally** - one module at a time
6. ~~**Partial 64-bit ops still need unsafe**~~ SOLVED: `safe_unaligned_simd` has full NEON support
7. ~~**safe_unaligned_simd lacks aarch64 support**~~ WRONG: 0.2.3+ has full aarch64 NEON support
