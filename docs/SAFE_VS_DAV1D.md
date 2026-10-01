# Where safe / untracked rav1d loses to dav1d (2026-10-01)

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
