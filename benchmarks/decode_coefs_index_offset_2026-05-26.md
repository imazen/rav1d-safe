# decode_coefs index-offset port — A/B (2026-05-26)

Port of dav1d 1.5.0 commit `5ef6b241` ("decode_coefs: Optimize index offset
calculations") into `src/recon.rs` `decode_coefs_class`. Commit `e24b479`.

## Change (bit-exact strength-reduction)
- Cache `slw = min(lw, TX_32X32)`, `slh = min(lh, TX_32X32)`, `tx2dszctx = slw+slh`.
- eob ctx via shifts: `1 + (eob > 2<<tx2dszctx) + (eob > 4<<tx2dszctx)` instead of
  `sw*sh*2` / `sw*sh*4` multiplies (once per block).
- **Hot path:** for `TX_CLASS_2D`, index `levels` by `rc` directly (`levels[rc]`)
  instead of recomputing `x*stride + y`. `rc == x*stride+y` because `stride == 1<<shift`
  and `y < stride`, so the value is already in hand — drops a runtime `imul`+add per
  coefficient in the single hottest loop of the profile (decode_coefs ~45% of 4K AVIF
  checked decode). The 2D path is also bounds-safe: `rc < 1024 < levels.len() (1088)`.

## Validation
- 14/14 `decode_md5_verify` (8/10/12-bit) — bit-exact.

## Benchmark — 4K photo AVIF (3840x2561), checked build, v3/AVX2 path
Interleaved A/B, 60 iters/run, cores 14,15. `before` = parent (2bb8769) recon.rs,
`after` = e24b479. ms/iter:

| round | before | after | delta |
|-------|--------|-------|-------|
| 1 | 210.9 | 204.4 | -3.1% |
| 2 | 204.4 | 199.1 | -2.6% |
| 3 | 205.1 | 200.9 | -2.0% |
| 4 | 207.5 | 201.7 | -2.8% |
| 5 | 253.9 | 227.4 | (system blip — both spiked; ratio still favorable) |

Clean rounds (1-4): before avg 207.0 ms, after avg 201.5 ms → **~2.6% faster**.

Note: upstream measured only +0.46% (Chimera, Sapphire Rapids native build) — but that
number is misleading. Follow-up dav1d commit `63bf075a` (June 2025) revealed that the
hot-loop part of `5ef6b241` was **dead code for ~10 months**: the C used
`if (TX_CLASS_2D)`, which tests the enum constant (= always false), so dav1d always took
the slow `levels + x*stride+y` path. The inner loop also used `rc` instead of `rc_i`.
dav1d's 0.46% therefore came only from the eob-ctx/stride shift changes, not the
per-coefficient index win.

Our Rust port used the **correct** forms from the start — `if tx_class == TxClass::TwoD`
and `rc_i` in the inner loop — so it matches the *fixed* `63bf075a`, not the buggy
original. That is why our measured ~2.6% far exceeds upstream's stated 0.46%: we actually
enable the per-coefficient multiply elimination (and on a bounds-checked safe build,
indexing by an already-validated `rc`/`rc_i` is proportionally more valuable than in
dav1d's raw-pointer native path). Bit-exactness is proven by 14/14 MD5 + the algebraic
identity `rc == x*stride+y` (stride `== 1<<shift`, `y < stride`).

## Other dav1d 1.x algorithmic ports surveyed (not actioned)
- **msac `d22de29c` "minor msac optimizations"** (skip shifting 1s into LSB; invert
  bits once per refill instead of twice per call): **ALREADY PRESENT** in our
  `src/msac.rs` (ctx_refill invert-once `buf ^ 0xff`, ctx_norm `dif << d`, init `dif=0`).
  rav1d ported a newer-than-1.0.0 msac for the Rust path. No work needed.
- **looprestoration SGR/wiener 1.5.1 C rewrites** (`f32b3146`, `9da303e9`, `8291a66e`,
  `a149f5c3`): these rewrite the **scalar reference C** + reduce stack. rav1d-safe runs
  its own AVX2 SIMD for SGR/wiener (the 6.6% in profile is the SIMD path), so the C
  rewrites do not touch our hot path. The 1.5.1 SGR *speed* gain was SSSE3 asm
  (`ef4aff75`), not portable to our AVX2 safe path. Skipped.
