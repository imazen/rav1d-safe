# Where safe / untracked rav1d loses to dav1d on REAL footage (2026-10-06)

The tables further down were measured on synthetic streams. This section repeats the
per-family sampling (`scripts/perf/ptrace_sample.py` + `family_compare.py`, 1 thread, native
dispatch for both, symbolised dav1d 1.5.3 built from source) on real footage: a 4K H.264 clip,
box-downscaled to 1080p, encoded with aomenc and SVT-AV1 (see DECODER_COMPARISON.md,
"Real footage"). **Untracked build, so the tracker is out of the picture.** ms/frame, ours
minus dav1d:

| stream | dav1d | ours | ratio | mc | cdef | decode_ctx | loopfilter | itx | entropy | other families |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| aom 1080p | 11.9 | 24.6 | 2.07x | +5.36 | +1.52 | +1.47 | +1.15 | +0.62 | +0.19 | +2.4 |
| svt 1080p | 8.3 | 14.7 | 1.78x | +2.30 | +1.55 | +0.43 | +0.31 | +0.51 | +0.48 | +0.9 |
| aom 4K | 36.1 | 78.0 | 2.16x | +16.6 | +7.76 | +3.91 | +3.19 | +2.14 | +0.42 | +8.0 |
| svt 4K | 25.2 | 50.3 | 2.00x | +8.09 | +7.43 | +1.45 | +1.10 | +1.77 | +1.79 | +3.8 |

**Reading it.**

- **MC + CDEF are 54-62% of the gap** (MC alone 32-42%, CDEF 12-30%). The SIMD kernels as a
  group (mc, cdef, loopfilter, itx, looprestoration, ipred) are 74-78%; control and glue
  (`decode_b`, `refmvs`, `lf_mask`, `recon`, `block_mut`) are the other ~22-26%.
- **Entropy decoding is at parity** (+0.2..+1.8 ms, +2-15% of dav1d's 4-12 ms). It is the
  biggest family in dav1d's profile and is no longer a gap.
- If MC and CDEF alone matched dav1d the ratios would be about 1.49x (aom 1080p), 1.32x
  (svt 1080p), 1.49x (aom 4K) and 1.38x (svt 4K).

**MC** (aom 4K: ours 24.1 ms vs dav1d 7.5, 3.2x). dav1d spends its MC time in
fused AVX-512 `put/prep_6tap` kernels (2.7 + 1.5 ms) plus 0.9 ms of warp. Ours runs the
8-tap two-pass pipeline (horizontal pass to an i16 mid buffer, then vertical): 
`h_filter_8tap_8bpc_avx2_inner` 6.6 ms and `..._avx512_inner` 2.6 ms, `v_filter_8tap_to_i16`
2.1 + 1.1 ms, `prep_8tap`/`put_8tap` 2.0 + 1.6 ms, warp 2.7 ms (2.9x dav1d's), and 1.75 ms of
Rust glue in `recon::mc`. The biggest single symbol is the **AVX2** horizontal filter, so
much of the work still runs the 256-bit path on an AVX-512 machine. Candidates: dedicated
6-tap kernels (the encoders pick 6-tap filters for narrow blocks; we pay 8 taps and two
extra rows), fusing the h and v passes for small blocks to skip the mid buffer round trip,
AVX-512 h/v filters, a cheaper warp.

**CDEF** (aom 4K: 9.8 ms vs 2.0, 4.9x). Per 8x8 block ours does a padding pass into a u16
temp (`padding_8bpc` 3.4 ms), the filter (`cdef_filter_block_simd_8bpc` 3.7 ms, 3.1x dav1d's
`cdef_filter_8x8`) and the direction search (`cdef_find_dir_simd_8bpc` 1.7 ms vs dav1d's
AVX2 `cdef_dir` 0.3 ms, 5.6x). dav1d's whole padding step is inside `cdef_brow` (0.5 ms), so
**ours is ~7x its padding cost**: the u8->u16 row copies are scalar-looking loops over
bounds-checked indexing, per row, per block. A SIMD `cvtepu8_epi16` copy and a 256-bit
direction finder are the obvious first steps.

**Glue.** `decode_b` is 4.4 ms on aom 4K (the largest single Rust symbol outside the kernels),
`block_mut` 0.5 ms, `get_skip_ctx` 0.3 ms; the earlier note about inlined bounds-check /
`Option` code still applies and is the remaining ~25%.

Caveats: sampling stops the process every tick, so shares are scaled by separately measured
unperturbed ms/frame; families are name-regex heuristics and inlined code is charged to the
enclosing symbol; dav1d runs AVX-512 (icl) kernels on most paths here.

---

# Where safe / untracked rav1d loses to dav1d on synthetic streams (2026-10-01)

Per-family wall time (ms/frame, 1 thread, Zen 4 with AVX-512) of rav1d-safe
(`untracked` and the default tracked build) against a source-built dav1d 1.5.3,
by leaf-PC sampling (`scripts/perf/ptrace_sample.py`, no PMU needed) merged over
many runs and classified with `scripts/perf/family_compare.py`. Totals are from
separate unperturbed runs. Method and pitfalls: `docs/ASM_VS_DAV1D.md`.

## Totals

| stream | dav1d | untracked | default (tracked) |
|---|---:|---:|---:|
| 4K photo AVIF (3840x2561, 40 frames) | 24.3 | 37.2 (+53%) | 42.4 (+74%) |
| 4K intra IVF (16 frames) | 24.8 | 37.7 (+52%) | 42.8 (+72%) |
| 480p inter, sframe (1800 frames) | 0.83 | 1.99 (+141%) | 2.68 |
| 8-bit intra, small frames (39 frames) | 2.05 | 2.82 (+38%) | 3.25 |

## The gap by family, 4K photo AVIF (untracked), ms/frame

| family | dav1d | ours | delta | ours / dav1d |
|---|---:|---:|---:|---:|
| entropy (msac + coefs, merged) | 13.77 | 13.64 | **-0.14** | 0.99x |
| intra prediction (incl. palette idx, edge prep) | 3.29 | 8.14 | **+4.85** | 2.5x |
| loop filter | 1.82 | 4.58 | **+2.76** | 2.5x |
| inverse transforms | 1.23 | 3.37 | **+2.14** | 2.7x |
| decode_b / decode_sb / skip ctx | 0.93 | 2.48 | +1.55 | 2.7x |
| libc memset/memcpy (unsymbolised) | 0.16 | 1.02 | +0.85 | |
| recon glue | 1.83 | 2.45 | +0.63 | 1.3x |
| lf_mask | 1.01 | 1.54 | +0.54 | 1.5x |
| tracked build only: DisjointMut tracker | 0 | 3.90 | **+3.90** | |

**Entropy decoding is at parity or better** (it is also 55% of dav1d's time, so it
looks like the top family in a profile, but it is not a gap). dav1d runs msac as
separate asm functions and rav1d-safe inlines it into `decode_coefs`, so only
the merged family compares fairly.

## Ranked targets (untracked; stills)

1. **Loop filter, the h-direction cores: +2.8 ms (2.5x).** Ours: `apply_h_v4`
   (wd16 h) 0.99, `lpf_h_sb_uv` 0.71, generic `loop_filter_4`/wd cores 0.97,
   `lf_wd8_core` 0.44, `lpf_h_sb_y` 0.43. dav1d: h_sb_y 0.78 + h_sb_uv 0.48 (AVX2!)
   and all v-direction ~0.5. The h-direction (transposing) kernels are the
   gap; the v-direction is close. Bounded kernel work.
2. **Intra prediction: +4.8 ms.** Directional kernels first: `z2` 1.01 vs dav1d
   0.19 (5x), `z3`, plus `intra_pred_dispatch`/`intra_pred_direct` glue ~0.5.
   Then `prepare_intra_edges` 1.47 vs 0.97 (+0.5) and `read_pal_indices` 2.65 vs
   1.45 (+1.2; palette-heavy stream, content dependent).
3. **Inverse transforms: +2.1 ms.** `dct_dct_16x16` ~3.6x dav1d, `dct32` columns,
   `adst8x8`; plus ~0.4 ms of pure dispatch (`itxfm_dispatch_8bpc` 0.19 +
   `Fn::call` 0.20) that dav1d does as a direct function-pointer call.
4. **`decode_b` family: +1.55 ms.** `decode_b` 1.71 vs 0.57 and a standalone
   `get_skip_ctx` costing 0.32 ms (dav1d inlines it to ~0): glue instruction tax
   (see ASM_VS_DAV1D.md: inlined bounds-check/Option/min-max code).
5. **Tracked build: +3.9 ms** (`BorrowTracker::add` ~2.3-2.5, drop glue 0.7) plus
   more in `recon_glue`/`decode_ctx`: this is the cost `untracked` removes
   (docs/UNTRACKED_MODE.md). Reducing the tracker's own cost helps every tracked user.
6. Libc memset/memcpy: ~+0.5-0.9 ms of per-frame clears and copies.

Not worth attacking for stills: entropy decoding.

## Small inter frames (480p, 2.4x slower)

| family | dav1d | ours | delta |
|---|---:|---:|---:|
| motion compensation | 0.167 | 0.845 | **+0.68** |
| decode_b etc. | 0.092 | 0.237 | +0.145 |
| libc + scratch pool (`take/recycle_scratch_component`) | 0.012 | 0.124 | +0.11 |
| loop filter | 0.055 | 0.156 | +0.10 |
| entropy | 0.259 | 0.218 | -0.04 |

MC is 58% of the gap and ~5x dav1d: `warp_h_pass_8bpc` 0.17 + `warp_affine_8x8`
0.08 (dav1d's whole warp 0.056, AVX2), `h_filter_8tap` 0.13, `put_8tap` 0.11,
`v_filter_8tap` 0.06. This stream is warp-heavy (~2,000 warp calls/frame).
If inter content matters, MC (warp first) outranks everything above.

## Caveats
- Sampling stops the process every tick (2 ms), so per-run absolute times are
  perturbed; shares are used, scaled by unperturbed ms/frame.
- Families are name-regex heuristics; inlined code is charged to the enclosing
  symbol. The unsymbolised libc bucket is an inference from offsets.
- dav1d here runs AVX-512 kernels on most paths (it dispatches to avx512icl on
  Zen 4); rav1d-safe's `v4x`/avx512 coverage is partial.
- The photo and intra IVF streams both contain palette blocks, which inflates
  `read_pal_indices`; a palette-free photo would shift ~1 ms away from ipred.
