# Handoff: Refactor to Fully Safe Intrinsics

## Progress (2026-02-04)

### Completed
- **pal.rs**: Fully refactored with `#[arcane]` and `safe_unaligned_simd`
  - Reduced from implicit unsafe (whole function) to 5 explicit unsafe blocks
  - Only 2 in inner function (for partial 64-bit ops lacking safe wrappers)
  - Verified via `cargo asm`: same instructions (vmovdqu, vpmaddubsw, etc.)
  - No performance regression

### Findings for archmage

1. **`safe_unaligned_simd` compiles to identical code**:
   `safe_simd::_mm256_loadu_si256(arr)` → `vmovdqu` (same as raw pointer)

2. **Slice-to-array conversion has no overhead**:
   `slice[..32].try_into().unwrap()` optimizes away in release builds

3. **Missing safe wrappers** for partial ops:
   - `_mm_loadl_epi64` (64-bit load) - no `Is64BitsUnaligned` trait
   - `_mm_storel_epi64` (64-bit store) - no safe wrapper
   - Consider adding these to safe_unaligned_simd

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

Same pattern, using `Arm64` token and `safe_unaligned_simd::aarch64`.

Note: mc_arm.rs already has `#[arcane]` but still uses `unsafe {}` blocks.
These could be refactored to use safe_unaligned_simd.

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

## Notes

1. **Phase 1: Don't change FFI signatures** - They must match rav1d's dispatch system
2. **Keep `#![allow(unsafe_op_in_unsafe_fn)]`** at module top for FFI wrappers
3. **Use `forge_token_dangerously()`** in FFI wrappers - we know features are available because dispatch already checked
4. **Benchmark after each module** to ensure no performance regression
5. **Commit incrementally** - one module at a time
6. **Partial 64-bit ops still need unsafe** - `_mm_loadl_epi64`, `_mm_storel_epi64` lack safe wrappers
