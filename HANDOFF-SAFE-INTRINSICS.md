# Handoff: Refactor to Fully Safe Intrinsics

## Goal

Eliminate most `unsafe` blocks in safe_simd modules by using:
1. **`#[arcane]` from archmage** - Makes value-based intrinsics safe (Rust 1.85+)
2. **`safe_unaligned_simd`** - Makes memory operations safe

## Current State

- **Branch:** `feat/fully-safe-intrinsics` (created from main at 5148b8b)
- **Rust:** 1.95 nightly (supports safe value intrinsics)
- **Dependencies:** archmage 0.4.0, safe_unaligned_simd 0.2.4 already in Cargo.toml

### Unsafe Breakdown (before refactor)

| Category | Count | Can Eliminate? |
|----------|-------|----------------|
| SIMD value intrinsics | ~1,500 | ✅ Yes - `#[arcane]` |
| SIMD memory ops | ~1,300 | ✅ Yes - `safe_unaligned_simd` |
| FFI pointer ops | ~2,000 | ❌ No - required for `extern "C"` |
| **Total in safe_simd** | ~3,200 | ~50% reducible |

## How It Works

### Before (current pattern)
```rust
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn my_func(dst: *mut u8, src: *const u8, w: i32) {
    // All intrinsics need unsafe blocks
    let v = unsafe { _mm256_loadu_si256(src as *const __m256i) };
    let result = unsafe { _mm256_add_epi8(v, v) };
    unsafe { _mm256_storeu_si256(dst as *mut __m256i, result) };
}
```

### After (with archmage)
```rust
use archmage::{arcane, Desktop64, SimdToken};
use safe_unaligned_simd::x86_64 as safe_simd;

#[arcane]
fn my_func_inner(_token: Desktop64, dst: &mut [u8], src: &[u8]) {
    // Memory ops use safe_unaligned_simd (SAFE!)
    let v = safe_simd::_mm256_loadu_si256(src);

    // Value intrinsics are SAFE inside #[arcane]!
    let result = _mm256_add_epi8(v, v);

    // Memory ops use safe_unaligned_simd (SAFE!)
    safe_simd::_mm256_storeu_si256(dst, result);
}

// FFI wrapper (still unsafe - unavoidable)
pub unsafe extern "C" fn my_func(dst: *mut u8, src: *const u8, w: i32) {
    let token = Desktop64::forge_token_dangerously();
    let dst_slice = std::slice::from_raw_parts_mut(dst, w as usize);
    let src_slice = std::slice::from_raw_parts(src, w as usize);
    my_func_inner(token, dst_slice, src_slice);
}
```

## Refactoring Strategy

### Phase 1: x86_64 Modules (Priority Order)

1. **pal.rs** (~140 lines) - Smallest, good test case
2. **refmvs.rs** (~60 lines) - Small
3. **filmgrain.rs** (~1k lines) - Medium
4. **cdef.rs** (~800 lines) - Medium
5. **loopfilter.rs** (~1.2k lines) - Medium
6. **mc.rs** (~5k lines) - Large, most impact
7. **looprestoration.rs** (~2k lines) - Large
8. **ipred.rs** (~4k lines) - Large
9. **itx.rs** (~12k lines) - Largest, most complex

### Phase 2: ARM NEON Modules

Same order, using `Arm64` token instead of `Desktop64`.

- mc_arm.rs (already partially uses archmage - extend it)
- ipred_arm.rs
- cdef_arm.rs
- loopfilter_arm.rs
- looprestoration_arm.rs
- itx_arm.rs
- filmgrain_arm.rs
- refmvs_arm.rs

### Phase 3: msac (inline in msac.rs)

The msac SIMD is inline in `src/msac.rs`. Same pattern applies.

## Token Mapping

| Architecture | Token | Features |
|--------------|-------|----------|
| x86_64 AVX2 | `Desktop64` / `X64V3Token` | AVX2 + FMA |
| aarch64 NEON | `Arm64` / `NeonToken` | NEON (always available) |

## safe_unaligned_simd Functions

### x86_64 (require feature = "avx512" for some)
```rust
use safe_unaligned_simd::x86_64::*;

// Loads
_mm_loadu_si128(src: &[u8; 16]) -> __m128i
_mm256_loadu_si256(src: &[u8; 32]) -> __m256i

// Stores
_mm_storeu_si128(dst: &mut [u8; 16], v: __m128i)
_mm256_storeu_si256(dst: &mut [u8; 32], v: __m256i)
```

### aarch64
```rust
use safe_unaligned_simd::aarch64::*;

// Reference-based loads/stores
vld1q_u8(src: &[u8; 16]) -> uint8x16_t
vst1q_u8(dst: &mut [u8; 16], v: uint8x16_t)
```

## Testing After Each Module

```bash
# Build and verify
cargo build --no-default-features --features "bitdepth_8,bitdepth_16" --release

# Cross-compile check for ARM
cargo check --target aarch64-unknown-linux-gnu --no-default-features --features "bitdepth_8,bitdepth_16"

# Benchmark (should stay ~1.11-1.15s)
cd /home/lilith/work/zenavif && touch src/lib.rs && cargo build --release --example decode_avif
time for i in {1..20}; do ./target/release/examples/decode_avif /home/lilith/work/aom-decode/tests/test.avif /dev/null 2>/dev/null; done
```

## Expected Outcome

| Metric | Before | After |
|--------|--------|-------|
| unsafe blocks in safe_simd | ~3,200 | ~1,500 |
| Performance | ~1.13s | ~1.13s (unchanged) |
| Safety | Auditable unsafe | Maximally safe |

## mc_arm.rs Reference

`src/safe_simd/mc_arm.rs` already uses archmage partially. Use it as a reference for the pattern:

```rust
use archmage::{arcane, Arm64, SimdToken};

#[arcane]
fn avg_8bpc_inner(_token: Arm64, dst: &mut [u8], ...) {
    // SIMD ops here - value intrinsics are SAFE
}

pub unsafe extern "C" fn avg_8bpc_neon(...) {
    let token = unsafe { Arm64::forge_token_dangerously() };
    // Convert raw pointers to slices, call inner
    avg_8bpc_inner(token, dst_slice, ...);
}
```

## Notes

1. **Don't change FFI signatures** - They must match rav1d's dispatch system
2. **Keep `#![allow(unsafe_op_in_unsafe_fn)]`** at module top for FFI wrappers
3. **Use `forge_token_dangerously()`** in FFI wrappers - we know features are available because dispatch already checked
4. **Benchmark after each module** to ensure no performance regression
5. **Commit incrementally** - one module at a time
