# Safety Architecture Summary

## Status: All SIMD Modules Safe ✅

All 20 `safe_simd/` modules enforce `deny(unsafe_code)` when `asm` is disabled.
Zero `unsafe` blocks exist outside `#[cfg(feature = "asm")]` FFI wrappers.

### Feature Dependency Chain

```toml
[features]
default = ["bitdepth_8", "bitdepth_16"]  # Safe-SIMD, no asm, no unsafe

unchecked = []               # Skip bounds checks (debug_assert only)
c-ffi = ["unchecked"]        # C API (dav1d_* entry points)
asm = ["c-ffi"]              # Hand-written assembly
```

### Crate-Level Safety

**src/lib.rs:**
```rust
#![cfg_attr(not(any(feature = "asm", feature = "c-ffi")), deny(unsafe_code))]
```

**Each safe_simd module:**
```rust
#![cfg_attr(not(feature = "asm"), deny(unsafe_code))]
```

## Safe SIMD Architecture

### Archmage Token-Based Dispatch

```rust
use archmage::prelude::*;

// Entry point: #[arcane] adds #[target_feature], makes intrinsics safe
#[arcane]
fn transform(_token: Desktop64, dst: &mut [u8], coeff: &mut [i16]) {
    let v = loadu_128!(&coeff_bytes[0..16]); // safe_unaligned_simd
    let result = _mm_add_epi16(v, v);        // safe (Rust 1.93+ with target_feature)
    storeu_128!(&mut dst_bytes[0..16], result);
}

// Runtime detection: Desktop64::summon() checks CPUID, cached ~1.3ns
pub fn transform_dispatch(dst: &mut [u8], coeff: &mut [i16]) -> bool {
    let Some(token) = Desktop64::summon() else { return false };
    transform(token, dst, coeff);
    true
}
```

### Safe Load/Store Macros (pixel_access.rs)

| Macro | Size | Safe via |
|-------|------|----------|
| `loadu_256!` / `storeu_256!` | 256-bit | `safe_unaligned_simd` |
| `loadu_128!` / `storeu_128!` | 128-bit | `safe_unaligned_simd` |
| `loadi64!` / `storei64!` | 64-bit | value-type intrinsics |
| `loadi32!` / `storei32!` | 32-bit | value-type intrinsics |

All macros: checked by default, unchecked with `unchecked` feature.

### FlexSlice for Hot Loops

```rust
use crate::src::safe_simd::pixel_access::Flex;

let c = coeff.flex();       // FlexSlice — [] syntax, zero overhead
let mut d = dst.flex_mut(); // FlexSliceMut — mutable [] syntax
d[off] = ((d[off] as i32 + c[idx] as i32).clamp(0, 255)) as u8;
```

Verified identical assembly to both raw `[]` and `get_unchecked`.

## Module Safety Status

| Module | unsafe (asm off) | Notes |
|--------|-----------------|-------|
| itx.rs | **0** | 85 arcane fns, all slice-based |
| mc.rs | **0** | 29 rite fns converted to slices |
| ipred.rs | **0** | 28 inner fns converted |
| filmgrain.rs | **0** | Safe via zerocopy AsBytes/FromBytes |
| cdef.rs | **0** | Padding via DisjointMut slice access |
| loopfilter.rs | **0** | Dispatch fully safe |
| looprestoration.rs | **0** | Wiener + SGR safe |
| pal.rs | **0** | AVX2 pal_idx_finish |
| refmvs.rs | **0** | splat_mv |
| pixel_access.rs | **0** | SliceExt + FlexSlice |
| itx_arm.rs | **0** | FFI correctly gated |
| mc_arm.rs | **0** | FFI gated, NEON intrinsics |
| ipred_arm.rs | **0** | FFI gated |
| filmgrain_arm.rs | **0** | FFI gated |
| cdef_arm.rs | **0** | FFI gated |
| loopfilter_arm.rs | **0** | FFI gated |
| looprestoration_arm.rs | **0** | FFI gated |
| refmvs_arm.rs | **0** | FFI gated |
| partial_simd.rs | **0** | 64-bit SIMD helpers |
| mod.rs | **0** | Module declarations |

## Build Verification

```bash
# Default (safe-SIMD, deny(unsafe_code)):
cargo build --no-default-features --features "bitdepth_8,bitdepth_16"

# With unchecked bounds elision:
cargo build --no-default-features --features "unchecked,bitdepth_8,bitdepth_16"

# With C FFI:
cargo build --no-default-features --features "c-ffi,bitdepth_8,bitdepth_16"

# With hand-written assembly:
cargo build --features "asm,bitdepth_8,bitdepth_16"

# Cross-compile check (aarch64):
cargo check --no-default-features --features "bitdepth_8,bitdepth_16" --target aarch64-unknown-linux-gnu
```

All builds pass. 12/12 tests pass in safe-SIMD mode.
