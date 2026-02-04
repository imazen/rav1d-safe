# rav1d-safe: Safe SIMD Fork Strategy

## Goal

Replace rav1d's 160k lines of hand-written x86/ARM assembly with safe Rust intrinsics using archmage, while maintaining performance parity.

## Approach

**Phase 1: Fork & Validate** (Current)
- Fork rav1d 1.1.0
- Keep existing asm behind `asm` feature flag
- Add `safe-simd` feature flag for archmage implementations
- Validate with existing rav1d test suite

**Phase 2: Incremental Porting**
- Port one function at a time
- Brute-force test against scalar AND asm implementations
- Benchmark each ported function
- Start with mc (motion compensation) — largest impact

**Phase 3: Upstream**
- Once performance-validated, propose to memorysafety/rav1d
- Keep `asm` flag for fallback/comparison
- Document any performance gaps

## Assembly Scope

| Category | x86 Lines | Priority | Rationale |
|----------|-----------|----------|-----------|
| mc | 45,252 | 1 | Most called, highest impact |
| itx | 42,372 | 2 | Second most called |
| ipred | 25,656 | 3 | Intra prediction |
| looprestoration | 16,643 | 4 | SGR/Wiener filters |
| filmgrain | 12,748 | 5 | Optional feature |
| loopfilter | 9,312 | 6 | Deblocking |
| cdef | 6,520 | 7 | Smallest |

ARM (72,894 lines) comes after x86 is proven.

## Dispatch Architecture

rav1d uses:
1. `wrap_fn_ptr!` macro for type-safe function pointers
2. `CpuFlags` for runtime detection
3. `init_x86(flags)` populates dispatch table
4. Rust fallbacks exist for all functions

Our approach:
1. Add `#[arcane]` functions with archmage tokens
2. Add parallel `init_x86_safe(flags)` using archmage dispatch
3. Feature flag selects which init to use
4. No changes to calling code

## Archmage Integration

```rust
use archmage::{Desktop64, arcane};

#[arcane]
fn avg_8bpc_avx2(
    _token: Desktop64,  // Proves AVX2+FMA available
    dst: &mut [u8],
    tmp1: &[i16],
    tmp2: &[i16],
    ...
) {
    // Safe intrinsics - archmage enables target features
    let t1 = _mm256_loadu_si256(...);
    let sum = _mm256_add_epi16(t1, t2);
    let avg = _mm256_mulhrs_epi16(sum, round);
    ...
}
```

## Testing Strategy

1. **Brute-force**: Test all edge cases against scalar
2. **Differential**: Test safe-simd vs asm for identical output
3. **Fuzzing**: Random inputs, check for panics/UB
4. **Benchmarks**: Compare against asm, document any gaps

## File Structure

```
rav1d-safe/
├── Cargo.toml          # Add archmage, safe_unaligned_simd deps
├── src/
│   ├── simd/           # New safe SIMD implementations
│   │   ├── mod.rs
│   │   ├── mc.rs       # Motion compensation
│   │   ├── itx.rs      # Inverse transforms
│   │   └── ...
│   └── ... (original rav1d structure)
└── STRATEGY.md
```

## Success Criteria

- [ ] All rav1d tests pass with `safe-simd` feature
- [ ] Performance within 10% of asm (5% target)
- [ ] Zero unsafe in new SIMD code
- [ ] Supports SSE4.1, AVX2, AVX-512

## Open Questions

1. **AVX-512 coverage**: archmage may not have all AVX-512 intrinsics yet
2. **Permute patterns**: Complex lane shuffles may need extension methods
3. **ARM NEON**: archmage supports it, but lower priority than x86
