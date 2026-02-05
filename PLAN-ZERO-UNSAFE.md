# Plan: Zero-Unsafe Default Path

## Goal

Make `asm` a 2nd class citizen:
- Default build: Pure Rust with direct function calls, zero `unsafe`, no function pointers
- `feature = "asm"`: Current FFI-based dispatch for linking external assembly

## Current Architecture

```
wrap_fn_ptr!(pub unsafe extern "C" fn avg(...) -> ());

// Creates:
// - avg::FnPtr = unsafe extern "C" fn(...)
// - avg::Fn - wrapper holding function pointer
// - avg::Fn::call() - converts safe types to raw pointers, calls fn ptr

// Dispatch table:
Rav1dMCDSPContext {
    avg: avg::Fn,  // function pointer
    ...
}

// Call site:
f.dsp.mc.avg.call::<BD>(dst, tmp1, tmp2, w, h, bd);
```

## New Architecture

### 1. Safe Inner Functions (already done for most)

The `#[arcane]` inner functions already exist and are safe:
```rust
#[arcane]
fn avg_inner(_token: Desktop64, dst: &mut [u8], tmp1: &[i16], tmp2: &[i16], w: usize, h: usize) {
    // Pure safe SIMD - no unsafe blocks needed
}
```

### 2. Split FFI from Safe Path

```rust
// FFI wrapper - only for feature = "asm"
#[cfg(feature = "asm")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn avg_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    ...
) {
    let token = Desktop64::forge_token_dangerously();
    // convert pointers to slices
    avg_inner(token, dst_slice, tmp1, tmp2, w, h);
}

// Safe dispatch - for default path
#[cfg(not(feature = "asm"))]
pub fn avg_8bpc_avx2(
    dst: &mut impl AsMutSlice<u8>,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
) {
    if let Some(token) = Desktop64::summon() {
        avg_inner(token, dst, tmp1, tmp2, w, h);
    } else {
        avg_rust(dst, tmp1, tmp2, w, h);  // scalar fallback
    }
}
```

### 3. Feature-Gated DSP Context

```rust
#[cfg(feature = "asm")]
pub struct Rav1dMCDSPContext {
    pub avg: avg::Fn,  // function pointer for asm dispatch
    ...
}

#[cfg(not(feature = "asm"))]
pub struct Rav1dMCDSPContext {
    // Empty - no function pointers needed
    // Dispatch happens via static methods
}

impl Rav1dMCDSPContext {
    // Unified API regardless of feature
    pub fn avg<BD: BitDepth>(
        &self,
        dst: Rav1dPictureDataComponentOffset,
        tmp1: &[i16; COMPINTER_LEN],
        tmp2: &[i16; COMPINTER_LEN],
        w: i32,
        h: i32,
        bd: BD,
    ) {
        #[cfg(feature = "asm")]
        {
            self.avg.call::<BD>(dst, tmp1, tmp2, w, h, bd)
        }
        #[cfg(not(feature = "asm"))]
        {
            mc_safe::avg::<BD>(dst, tmp1, tmp2, w, h, bd)
        }
    }
}
```

### 4. Call Site Changes

Before:
```rust
f.dsp.mc.avg.call::<BD>(dst, tmp1, tmp2, w, h, bd);
```

After:
```rust
f.dsp.mc.avg::<BD>(dst, tmp1, tmp2, w, h, bd);
```

## Implementation Order

### Phase 1: Proof of Concept with `avg`
1. Create `mc_safe` module with safe dispatch function
2. Modify `Rav1dMCDSPContext` to be feature-gated
3. Add `avg()` method to context
4. Update call sites (only ~5 places)
5. Verify builds and benchmarks match

### Phase 2: Extend to all MC functions
- w_avg, mask, blend, blend_v, blend_h
- mc (8tap filters - more complex due to filter enum)
- mct, mc_scaled, mct_scaled
- warp8x8, warp8x8t
- emu_edge, resize

### Phase 3: Other DSP modules
- cdef
- loopfilter
- looprestoration
- itx
- ipred
- filmgrain
- pal
- refmvs

## Files to Modify

1. `src/wrap_fn_ptr.rs` - Keep for asm, not used in default path
2. `src/mc.rs` - Feature-gate DSP context, add dispatch methods
3. `src/safe_simd/mc.rs` - Split FFI wrappers from safe core
4. `src/recon.rs` - Update call sites
5. `src/lf_apply.rs` - Update call sites
6. Similar for other DSP modules

## Benefits

1. **Zero unsafe in default path** - All SIMD is safe via archmage
2. **No function pointers** - Direct calls, better inlining
3. **Cleaner API** - `dsp.mc.avg::<BD>(...)` instead of `dsp.mc.avg.call::<BD>(...)`
4. **Smaller binary** - No FFI overhead when asm disabled
5. **Easier auditing** - Safety is compiler-verified

## Risks

1. **Code duplication** - FFI wrappers still needed for asm feature
2. **Large refactor** - Many call sites need updates
3. **Performance regression** - Need to verify dispatch overhead
