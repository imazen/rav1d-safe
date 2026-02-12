# Handoff: Port Remaining msac Functions to Safe SIMD

## The Problem

msac (Multi-Symbol Adaptive Coding) is the entropy decoder. It's 32% of decode time. The ASM build gets hand-tuned SSE2/NEON for **all** msac functions. The safe-SIMD build only ported `symbol_adapt16` — the rest fall back to scalar Rust. This is the primary performance gap (1.58x vs ASM on 8bpc).

**What's ported:** `symbol_adapt16` (AVX2 + NEON)
**What's missing:** `symbol_adapt4`, `symbol_adapt8`, `bool`, `bool_equi`, `bool_adapt`, `hi_tok`

## Why This Is Hard

The ASM advantage here isn't SIMD parallelism in the traditional sense. These functions process 1-7 symbols. The ASM wins because:

1. **Zero function call overhead** — all functions share renorm/refill code via `jmp` to common labels. No function prologues, no stack frames.
2. **Registers stay hot** — `dif`, `rng`, `cnt`, `buf` live in GPRs across the entire operation including refill.
3. **Bool functions are pure GPR** — `bool_equi`, `bool`, `bool_adapt` use zero XMM registers. They're just fast scalar with `cmovb` for branchless selection.
4. **hi_tok is a monolithic loop** — it calls adapt4's comparison+CDF logic inline 1-4 times with its own embedded renorm/refill. No function call overhead at all.

The Rust scalar implementations have:
- Function call overhead on every `ctx_norm` → `ctx_refill` path
- Branch prediction tax from the refill check
- Bounds checking overhead (even if minimal per-call, it adds up over 200k+ calls/frame)

## Architecture: The MsacContext

```
struct MsacContext {
    buf: &[u8],  // current position in bitstream
    end: &[u8],  // end of bitstream
    dif: u64,    // difference (the bit window)
    rng: u32,    // range [1, 65535]
    cnt: i32,    // bits remaining in dif before refill needed
    allow_update_cdf: bool,
}
```

Every decode operation follows the same pattern:
1. Compute `v` (threshold) from `rng` and CDF probabilities
2. Compare `dif >> 48` against `v` to find the decoded symbol
3. Update `dif` and `rng` based on the chosen symbol
4. **Renormalize:** `d = clz(rng)`, shift `rng` and `dif` left by `d`
5. **Refill:** if `cnt < d`, read bytes from `buf` into `dif`
6. Optionally update the CDF

Steps 4-5 are shared across ALL functions in the ASM via label jumps.

## File Locations

- **Current Rust:** `src/msac.rs` (1123 lines)
- **x86 SSE2 ASM:** `src/x86/msac.asm` (671 lines) — the reference
- **ARM64 NEON ASM:** `src/arm/64/msac.S` (587 lines) — secondary reference
- **Dispatch:** bottom of `src/msac.rs` (functions `rav1d_msac_decode_*`)

## Function-by-Function Analysis

### 1. `symbol_adapt4` (SSE2 lines 87-219)

**Call frequency:** ~100k/frame. Called by `hi_tok` internally.

**What the ASM does:**
```
1. Load CDF (4 values) into xmm1 as u16x4
2. Broadcast rng into xmm2, mask with 0xff00
3. Compute v[i] = pmulhuw(cdf[i] >> 6 << 7, rng) + min_prob[n-i]
4. Broadcast dif>>48 into xmm3
5. psubusw(v, dif) — saturating subtract: result is 0 where dif >= v
6. pcmpeqw against zero — gives -1 mask where c >= v
7. pmovmskb + tzcnt — extract index of first match = decoded symbol
8. CDF update (if enabled): pavgw/psraw/paddw SIMD trick
9. Jump to shared renorm/refill
```

**Key insight:** The CDF update uses a clever SIMD trick:
```
pavgw(all_ones, cmp_mask)  → gives 32768 where i >= val, -1 where i < val
psubw(result, cdf)          → (32768 - cdf[i]) or (-1 - cdf[i])
psraw(result, rate)          → >> rate
paddw(cdf, result)           → updated CDF
```

This is a single SIMD CDF update instead of the scalar loop. Same pattern works for adapt8.

**Safe Rust approach:**
- Use SSE2 intrinsics (safe since Rust 1.93)
- `_mm_loadu_si64` for the 4-value CDF (or `_mm_loadl_epi64` equivalent)
- Same `pmulhuw`/`psubusw`/`pcmpeqw`/`pmovmskb` chain
- Same `pavgw`/`psraw`/`paddw` CDF update
- Call `ctx_norm` for renorm (can't share labels, but inlining compensates)

**Current Rust:** `rav1d_msac_decode_symbol_adapt4_branchless` — computes v values individually, counts matches with comparisons. Already branchless, but scalar.

### 2. `symbol_adapt8` (SSE2 lines 221-265)

**Call frequency:** ~50k/frame.

**What the ASM does:** Same as adapt4 but uses full 128-bit (8 lanes). The only difference:
- `punpcklqdq m2, m2` to broadcast rng to all 8 lanes
- Full `mova` (128-bit) load of CDF instead of `movq` (64-bit)
- Falls through to adapt4's renorm label

**Safe Rust approach:** Same as adapt4 but 128-bit vectors throughout.

### 3. `bool_equi` (SSE2 lines 406-431)

**Call frequency:** ~80k/frame.

**What the ASM does:** Pure GPR, zero SIMD registers.
```
v = (rng >> 1) | 8     // rng/2 rounded up, minimum 8
vw = v << 48
new_dif = dif - vw
if (new_dif < 0):       // borrow = symbol is 0
    rng = v              // narrow range to [0, v)
    dif unchanged
else:
    rng = rng - v        // narrow range to [v, rng)
    dif = new_dif
// renorm: d = 2 - (v >> 14), special case since 0 <= d <= 2
```

**Key ASM trick:** The renorm computation is simplified — instead of `clz(rng)`, the ASM computes `d = (0xbfff - rng) >> 14` which is valid because `rng` after bool_equi is always in a narrow range. This saves one instruction.

**Safe Rust approach:** The current scalar `rav1d_msac_decode_bool_equi_rust` is already optimal algorithmically. The ASM advantage is purely calling convention — no function call to `ctx_norm`, no bounds check in `ctx_refill`, registers pre-loaded. The only way to close this gap is inlining.

**Recommendation:** Mark `ctx_norm` and `ctx_refill` as `#[inline(always)]` if not already (they are). The compiler should inline these. Profile to verify. If there's still a gap, consider a monolithic version that doesn't call `ctx_norm` at all.

### 4. `bool` (SSE2 lines 433-454)

**Call frequency:** ~60k/frame.

Same as `bool_equi` but `v = (rng >> 8) * (f >> 6) >> 1 + 4` where `f` is the probability parameter. Pure GPR, `cmovb` for branchless. Jumps to shared renorm.

**Safe Rust approach:** Same as bool_equi — the scalar Rust is algorithmically identical. The gap is calling convention.

### 5. `bool_adapt` (SSE2 lines 340-404)

**Call frequency:** ~80k/frame.

Like `bool` but with CDF update. The CDF update is scalar (just 2 values: cdf[0] and count).
```
if (bit):
    cdf[0] -= ((cdf[0] - 32769) >> rate) + 1
else:
    cdf[0] -= cdf[0] >> rate
```

**Safe Rust approach:** Same as bool. The CDF update is trivially scalar.

### 6. `hi_tok` (SSE2 lines 456-614) — THE BIG ONE

**Call frequency:** ~20k/frame, but each call does 1-4 internal iterations.

**What the ASM does:** This is a self-contained loop that avoids ALL function call overhead:

```
tok = -24
loop:
    // Inline adapt4 comparison (XMM)
    compute v[0..3] from CDF
    psubusw + pcmpeqw + pmovmskb → get comparison result

    // Inline CDF update (XMM, if enabled)
    pavgw/psraw/paddw trick

    // Inline renorm (GPR)
    tzcnt → symbol
    sub dif, v[symbol] << 48
    rng = u - v
    d = clz(rng) ^ 15
    shl rng, d; shl dif, d
    cnt -= d

    // Inline refill (GPR, if cnt < 0)
    read from buf, bswap, merge into dif

    tok += 5
    tok_br = symbol
    if tok_br < 3 && tok < 15*5:
        continue loop
    return (tok + 30) / 2
```

The key insight: `hi_tok` embeds the ENTIRE adapt4 + renorm + refill pipeline in a loop body with zero function calls. The ASM uses two copies of this loop body (one with CDF update, one without) selected by the `update_cdf` flag at entry.

**Safe Rust approach:** This is where "collapse into a big function" matters. The current Rust `rav1d_msac_decode_hi_tok_rust` calls `rav1d_msac_decode_symbol_adapt4` up to 4 times — that's 4 function calls, each with its own `ctx_norm` → `ctx_refill` path. The ASM does it all inline.

**Implementation strategy:**
```rust
#[inline(always)]
fn rav1d_msac_decode_hi_tok_simd(s: &mut MsacContext, cdf: &mut [u16; 4]) -> u8 {
    let update = s.allow_update_cdf();
    let mut tok: i32 = -24;

    loop {
        // === INLINE ADAPT4 ===
        // Load CDF into XMM
        // Compute v[0..3] via pmulhuw
        // Compare via psubusw + pcmpeqw
        // tzcnt to get symbol (tok_br)

        // === INLINE CDF UPDATE ===
        if update {
            // pavgw/psraw/paddw trick
            // store updated CDF
        }

        // === INLINE RENORM ===
        // Extract u, v from stored values
        // rng = u - v
        // dif -= v << 48
        // d = clz(rng) ^ 15
        // rng <<= d; dif <<= d
        // cnt -= d

        // === INLINE REFILL ===
        if (s.cnt as u32) < (d as u32) {
            // Read bytes from buf, bswap, merge
        }

        tok += 5;
        let tok_br = symbol;
        if tok_br >= 3 || tok + tok_br as i32 >= 15 * 2 {
            break;
        }
    }

    ((tok + 30) / 2) as u8
}
```

## Implementation Plan

### Phase 1: SSE2 adapt4 + adapt8 (highest impact)

These two have actual SIMD benefit — the vector comparison + CDF update is genuinely faster than scalar.

1. Write `rav1d_msac_decode_symbol_adapt4_sse2_safe` as `#[arcane]` function taking `Desktop64` token
2. Use SSE2 intrinsics: `_mm_loadl_epi64`, `_mm_shuffle_epi32`, `_mm_mulhi_epu16`, `_mm_subs_epu16`, `_mm_cmpeq_epi16`, `_mm_movemask_epi8`
3. CDF update: `_mm_avg_epu16`, `_mm_sra_epi16`, `_mm_add_epi16`
4. Call `ctx_norm` for renorm (let compiler inline it)
5. Similarly for adapt8 with 128-bit full width

**NEON equivalents:**
- `vld1_u16` (4 values) / `vld1q_u16` (8 values)
- `vqdmulhq_s16` (signed doubling multiply high, equivalent to `sqdmulh`)
- `vcgeq_u16` (compare >=)
- NEON doesn't have `movemask` — use `vshrn` + table lookup or horizontal add

### Phase 2: Monolithic hi_tok (second highest impact)

Write a single `#[arcane]` function that embeds the adapt4 SIMD + renorm + refill in a loop. This eliminates 3-12 function calls per hi_tok invocation.

The refill logic must be inlined. Copy the body of `ctx_refill` into the loop, operating on local variables, then write back to `s` only at the end.

### Phase 3: Bool functions (lowest impact, but easy)

These are pure scalar. The only way to speed them up in safe Rust is to ensure `ctx_norm` and `ctx_refill` are fully inlined and the compiler eliminates the redundant loads/stores.

Consider writing monolithic versions that keep `dif`, `rng`, `cnt` in locals and write back once at the end.

### Phase 4 (optional): NEON versions of adapt4/adapt8/hi_tok

Same algorithms as SSE2 but using NEON intrinsics. The ARM ASM uses:
- `sqdmulh` for multiply-high (saturating doubling multiply high)
- `cmhs` for unsigned >=
- `clz` for leading zeros
- Vector shift/extract for finding the symbol

## Critical Details

### Renorm/Refill Inlining

The biggest perf win comes from eliminating the `ctx_norm` → `ctx_refill` function call chain. In the ASM, this is shared code via label jumps. In Rust, the equivalent is `#[inline(always)]` which the compiler already applies.

**But verify this!** Run `cargo asm` on the hot functions to confirm the compiler actually inlines `ctx_norm` and `ctx_refill`. If it doesn't (due to code size heuristics), manually inline them.

### The tzcnt Trick

The ASM uses `tzcnt` (count trailing zeros) on the `pmovmskb` result to find the first matching symbol. In Rust:

```rust
let mask = _mm_movemask_epi8(cmp_result) as u32;
let symbol = mask.trailing_zeros() / 2; // /2 because pmovmskb gives 2 bits per u16 lane
```

### CDF Update SIMD Trick

This is worth understanding because it's the same across adapt4/adapt8/adapt16:

```
// Given: cmp_mask[i] = 0xFFFF if i >= val, else 0x0000
// Want: cdf[i] += (({32768,-1} - cdf[i]) >> rate) + (i >= val ? 1 : 0)

pavgw(all_ones, cmp_mask)   // → 32768 where i >= val, -1 where i < val
psubw(result, old_cdf)      // → delta before shift
psraw(result, rate)          // → delta after shift
paddw(old_cdf - cmp_mask, result)  // → updated CDF
                              // (subtracting cmp_mask adds 1 where i >= val)
```

The `psubw(m0, m1)` in the adapt4 ASM computes `cdf[i] - (i >= val ? -1 : 0)` = `cdf[i] + (i >= val ? 1 : 0)`. Combined with the shifted delta, this gives the complete CDF update in 4 SIMD instructions instead of a scalar loop.

### The `MsacAsmContext` Issue

Currently, `MsacAsmContext` has two modes:
- **ASM mode:** `buf` is a raw pointer pair (`*const u8, *const u8`)
- **Safe mode:** `buf` is `{ pos: usize, end: usize }` (indices into `MsacContext.data`)

The safe SIMD implementations access `s.data` through `MsacContext` (which wraps `MsacAsmContext`). The refill logic in safe mode indexes into `s.data.as_ref()[pos..end]`. If you inline refill into hi_tok, you'll need to manage `pos`, `end`, `dif`, `rng`, `cnt` as locals and write them back at the end.

### Testing

The existing scalar implementations are the ground truth. For every new SIMD function:
1. The existing `cargo test --release` suite exercises all code paths via full decode
2. The MD5 parity tests verify pixel-perfect output
3. Consider adding targeted msac unit tests that compare SIMD vs scalar on random CDF inputs

### Dispatch Pattern

Add new dispatch branches to the `cfg_if!` blocks in the public functions:

```rust
pub fn rav1d_msac_decode_symbol_adapt4(s: &mut MsacContext, cdf: &mut [u16], n_symbols: u8) -> u8 {
    cfg_if! {
        if #[cfg(all(feature = "asm", target_feature = "sse2"))] {
            // existing ASM path
        } else if #[cfg(all(not(feature = "asm"), not(feature = "force_scalar"), target_arch = "x86_64"))] {
            // NEW: SSE2 safe SIMD
            if let Some(token) = crate::src::cpu::summon_sse2_or_whatever() {
                return rav1d_msac_decode_symbol_adapt4_sse2_safe(token, s, cdf, n_symbols);
            }
            // scalar fallback
            rav1d_msac_decode_symbol_adapt4_branchless(s, cdf, n_symbols)
        } else {
            rav1d_msac_decode_symbol_adapt4_branchless(s, cdf, n_symbols)
        }
    }
}
```

**Note on token dispatch:** SSE2 is baseline on x86_64 (always available). You might not need a runtime check at all — just gate on `target_arch = "x86_64"`. Check what archmage provides for SSE2-level tokens. If `Desktop64::summon()` requires AVX2, you may need a lower-tier token or just use `#[target_feature(enable = "sse2")]` directly (which is always true on x86_64 but needed for the compiler to emit SSE2 instructions in `#[arcane]`).

Actually — SSE2 is guaranteed on x86_64, so you can use `#[target_feature(enable = "sse2")]` and it will always be valid. But check if archmage has an SSE2 token type. If not, you can use the function directly without token dispatch since SSE2 is baseline.

## Expected Performance Impact

Based on profiling (39 frames allintra 8bpc):

| Function | Calls/frame | Current | Expected with SIMD |
|----------|------------|---------|-------------------|
| adapt4 | ~100k | scalar branchless | SSE2 vector: ~2x faster |
| adapt8 | ~50k | scalar branchless | SSE2 vector: ~2x faster |
| hi_tok | ~20k (×1-4 iters) | 4× adapt4 calls | monolithic: ~3x faster |
| bool_equi | ~80k | scalar | same (already optimal) |
| bool | ~60k | scalar | same (already optimal) |
| bool_adapt | ~80k | scalar | same (already optimal) |

The adapt4+adapt8+hi_tok optimizations should close most of the remaining 1.58x gap on 8bpc. The bool functions won't benefit from SIMD but might benefit from better inlining.

## Priority

1. **adapt4 SSE2** — most calls, real SIMD benefit, required for hi_tok
2. **hi_tok SSE2** — uses adapt4 internally, monolithic loop is the big win
3. **adapt8 SSE2** — straightforward extension of adapt4
4. **NEON versions** — same algorithms, different intrinsics
5. **Bool inlining audit** — check asm output, verify inlining happens
