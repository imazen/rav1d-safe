# rav1d-safe: Safe SIMD Fork Strategy

## Goal

Replace rav1d's 160k lines of hand-written x86/ARM assembly with safe Rust SIMD intrinsics using archmage, while maintaining performance parity.

## Status: COMPLETE ✅

All SIMD modules ported. Safe-SIMD matches or beats hand-written assembly performance.

### Ported Modules

| Module | x86 (AVX2) | ARM (NEON) | Lines |
|--------|-----------|------------|-------|
| mc | ✅ | ✅ | ~9k |
| itx | ✅ | ✅ | ~19k |
| ipred | ✅ | ✅ | ~26k |
| looprestoration | ✅ | ✅ | ~17k |
| filmgrain | ✅ | ✅ | ~1.8k |
| loopfilter | ✅ | ✅ | ~9k |
| cdef | ✅ | ✅ | ~7k |
| pal | ✅ | (fallback) | ~150 |
| refmvs | ✅ | ✅ | ~110 |
| msac | ✅ | ✅ | (inline) |

### Safety: All 20 Modules `deny(unsafe_code)` When ASM Off

Zero `unsafe` blocks outside `#[cfg(feature = "asm")]` FFI wrappers.

## Architecture

### Archmage Token-Based Dispatch

```rust
use archmage::prelude::*;

#[arcane]
fn transform(_token: Desktop64, dst: &mut [u8], coeff: &mut [i16]) {
    let v = loadu_128!(&coeff_bytes[0..16]);   // safe load
    let result = _mm_add_epi16(v, v);          // safe (Rust 1.93+)
    storeu_128!(&mut dst_bytes[0..16], result); // safe store
}
```

Key points:
- `#[arcane]` enables target features via proof token — intrinsics become safe
- `safe_unaligned_simd` provides reference-based SIMD load/store (no raw pointers)
- Runtime detection: `Desktop64::summon()` checks CPUID (~1.3ns cached)
- **`#[arcane]` NEVER needs `#[allow(unsafe_code)]`** — rewrite the body instead

### Safe Memory Access (pixel_access.rs)

| Macro | Size | Source |
|-------|------|--------|
| `loadu_256!` / `storeu_256!` | 256-bit | `safe_unaligned_simd` |
| `loadu_128!` / `storeu_128!` | 128-bit | `safe_unaligned_simd` |
| `loadi64!` / `storei64!` | 64-bit | value-type intrinsics |
| `loadi32!` / `storei32!` | 32-bit | value-type intrinsics |

Plus `FlexSlice` — zero-cost `[]` wrapper with optional bounds elision.

## Performance

Full-stack benchmark (20 decodes of test.avif via zenavif):
- **ASM: ~1.17s**
- **Safe-SIMD: ~1.11s**
- ✅ Performance parity achieved

## Success Criteria — All Met

- [x] All rav1d tests pass with safe-SIMD
- [x] Performance within 10% of asm (actually matches/beats)
- [x] Zero unsafe in SIMD code when asm off
- [x] Supports AVX2 (x86_64) and NEON (aarch64)
- [x] Cross-compilation: x86_64, aarch64
