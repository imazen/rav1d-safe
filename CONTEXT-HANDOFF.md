# Context Handoff: MSAC SIMD Porting

## Current State

All major safe_simd modules are complete (39k lines). Only msac remains.

**Git state:** Clean, latest commit f115ab0

## MSAC Overview

msac (Multi-Symbol Arithmetic Coder) is the entropy decoder. It has ~1.8k lines of ASM (671 x86 + 1175 ARM).

### Key Insight: Compile-Time Dispatch

Unlike other modules, msac uses **compile-time cfg_if! dispatch**, not runtime function pointers:

```rust
// From src/msac.rs
cfg_if! {
    if #[cfg(all(feature = "asm", target_feature = "sse2"))] {
        ret = unsafe { dav1d_msac_decode_symbol_adapt4_sse2(...) };
    } else if #[cfg(all(feature = "asm", target_feature = "neon"))] {
        ret = unsafe { dav1d_msac_decode_symbol_adapt4_neon(...) };
    } else {
        ret = rav1d_msac_decode_symbol_adapt_rust(s, cdf, n_symbols);
    }
}
```

**Exception:** `symbol_adapt16` on x86_64 has a runtime function pointer stored in `MsacAsmContext`.

### ASM Functions (7 total per arch)

1. `symbol_adapt4` - 4-symbol CDF decode
2. `symbol_adapt8` - 8-symbol CDF decode  
3. `symbol_adapt16` - 16-symbol CDF decode (only one with fn ptr on x86_64)
4. `bool_adapt` - Boolean with CDF update
5. `bool_equi` - Equiprobable boolean
6. `bool` - Boolean with probability
7. `hi_tok` - High token decode

### SIMD Algorithm (from reading x86/msac.asm)

1. **Parallel CDF probability calculation:** `v[i] = (rng >> 8) * (cdf[i] >> 6) >> 1 + min_prob[n-i-1]`
2. **Vectorized comparison:** Compare all v[i] against `c = dif >> 48`, get bitmask
3. **Symbol lookup:** `tzcnt` on bitmask finds symbol
4. **Vectorized CDF update:** Using pavgw/psubw/psraw for the update formula

### Rust Fallback Location

`src/msac.rs:395` - `rav1d_msac_decode_symbol_adapt_rust()`

Already correct, uses same algorithm but scalar.

### What Needs Porting

Create `src/safe_simd/msac.rs` with:
- AVX2 implementations for x86_64
- NEON implementations for aarch64
- Wire into msac.rs dispatch (tricky because it's cfg_if not fn ptrs)

### Approach Options

**Option A: Add safe_simd cfg branch to cfg_if**
```rust
cfg_if! {
    if #[cfg(all(feature = "asm", target_feature = "sse2"))] { ... }
    else if #[cfg(all(not(feature = "asm"), target_arch = "x86_64"))] {
        // Call safe_simd AVX2 version
    }
    else { ... }
}
```

**Option B: Leave as-is**
The scalar Rust fallback is already used when `feature = "asm"` is disabled. Performance impact may be minimal since CDF search is inherently serial.

### Key Files

- `src/msac.rs` - Main msac implementation with dispatch
- `src/x86/msac.asm` - 671 lines x86 ASM
- `src/arm/64/msac.S` - 587 lines ARM64 ASM

### Build/Test Commands

```bash
cargo build --no-default-features --features "bitdepth_8,bitdepth_16" --release
cargo check --target aarch64-unknown-linux-gnu --no-default-features --features "bitdepth_8,bitdepth_16"
cd /home/lilith/work/zenavif && time for i in {1..20}; do ./target/release/examples/decode_avif test.avif /dev/null; done
```

### Performance Baseline

Current (without msac SIMD): ~1.11s for 20 decodes
ASM version: ~1.17s

Safe-simd already beats ASM, so msac SIMD is optimization, not parity.
