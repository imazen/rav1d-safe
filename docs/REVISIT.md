# Revisit list

Parked experiments and open questions, with where the work lives. Each experiment is a branch
`revisit/<name>` on `origin`, one commit on top of main at `f3132ee6` (the squashed net change of the
agent branch that produced it). Nothing here is landed or gated beyond what is stated.

Method for every timing below (2026-10-09, r7900x / Zen 4, shared box): `scripts/perf/decoder_bench/paired_modes.py`
against `f3132ee6`, four real-footage clips (aom 1080p, aom 4K, svt 1080p, aom 10-bit 1080p), 1 and 4 workers,
tracked and untracked, 4 rounds x 3 passes, with an A-vs-A control. A result counts only if it beats the A/A spread.
Zen 5 is not measured. Every branch decodes the four clips MD5-identical to dav1d at 1 and 4 workers in both modes.
None has run nextest or the 803-vector corpus.

| Branch | What it tries | Result | Next step |
|--------|---------------|--------|-----------|
| ~~`revisit/mc-v512`~~ | AVX-512 active-row vertical MC + warp pair | **Landed 2026-10-12** on top of the tracker stage, vertical specialization only: const-generic `v_filter_ntap_*_avx512_inner<NT>` dispatch by active tap count (4/6/8). The paired warp kernel was dropped mid-lane (failing negative-destination oracle — see `docs/MC_WARP_PAIR_REVIEW.md`). Clean-cell untracked wins vs the tracker stage: 4K −1.5/−1.6%, aom1080 −1.1%/−0.65%, svt −0.7%/−1.4% (8-round remeasure after the first run's A/A controls dirtied at load 5-8); tracked and 10-bit neutral. Cross-tier md5 40/40 | Done. Zen 5 not measured (double-pumped on Zen 4) |
| ~~`revisit/cdef`~~ | CDEF direction-cost / padding variants | **Landed 2026-10-11.** Kept: contiguous-loop cost helper for `find_dir`, vectorized widening padding copies, scalar-oracle + full-domain parity tests. Dropped after bisect: `index_rect_as` rect borrows and `declare_row_stride` line-buffer hints — declaring a stride re-derives the tracker block shift (`block_shift_rule_rows`), coarsening blocks and costing +2.2-3.3% at 4 workers tracked. Landed build is inside A/A noise there, −0.5-1.4% at 1 worker | Done. `declare_row_stride` on shared buffers needs revisiting before rect borrows can help |
| ~~`revisit/tracker`~~ | Retained-row CDEF borrows in the tracked build | **Landed 2026-10-12** on top of the CDEF stage, minus the shared line-buffer rect/declare parts (dropped in the stage-1 bisect). `cdef_block_mut` reserves the 8x8 destination rect on the already-declared picture buffer across copy-in→filter→copy-out, five call sites swap `with_pixel_guard_mut` → `with_cdef_guard_mut`. Tracked 4-worker wins: 4K −4.0%, aom1080 −1.3%, aom10 t1 −4.1%; all other cells inside A/A noise after an 8-round remeasure of the two borderline cells (svt t4, untracked aom1080 t4) | Done |
| ~~`revisit/mc-fused`~~ | Fused small-block MC (4x4 blocks, four-tap filters only) | **Landed 2026-10-12** on top of mc-v512: `fused_hv_4x4_8bpc<SHIFT>` keeps four horizontal rows in a `__m128i` ring feeding each vertical output — no pooled mid buffer — wired for w=4,h=4,4-tap put/prep at the avx2 and avx512 tiers. Landing fixed a branch defect (put sites wrote `dst[y*stride..]`, dropping `dst_base`; panicked on signed destination rows — caught by the full-gate parity tests). Tracked wins vs s3: aom1080 −1.6%/−3.8% (t1/t4), svt t4 −3.5%; untracked svt t1 −2.0% marginal; all other cells noise after 8-round remeasures. Post-fix spot-retime confirms the wins | Done. Extend per `docs/MC_FUSED_SHAPES.md` census (next task) |
| `revisit/glue` | `decode_b` initialization | Timed 2026-10-10: neutral. 1-worker A/B 0.996-1.011 tracked, 0.997-1.004 untracked; the 4-worker cells that moved 2-5% have off-nominal A/A controls (load reached 10). The change only moves where one `Av1Block::default()` is built (five stores, frame-threaded passes only) | Drop. The ~25% glue share needs profile-driven work (`decode_b`, `refmvs`, `lf_mask`, bounds checks, `block_mut`), not this |

## Which of the five is good groundwork (2026-10-10)

Judged on measured effect, what each leaves behind that is reusable, and how much of the gap to dav1d
it addresses. Timing is Zen 4 only, nextest and the corpus have not been run on any branch, and the
assessment of design is from reading the diffs.

1. **~~`revisit/cdef`: best groundwork.~~ Landed 2026-10-11.** CDEF is 12-30% of the gap (padding ~7x,
   `find_dir` ~5.6x, filter ~3x). The landed change keeps the contiguous-loop `find_dir` cost helper
   (won the A/B over the explicit AVX2 variant 8 clean cells to 1), the vectorized widening padding
   copies, and the full-domain parity + scalar-oracle tests. The exact-window rectangle borrows and
   `declare_row_stride` line-buffer hints were measured to cost +2.2-3.3% at 4 workers tracked — the
   stride declaration coarsens tracker block geometry under tile parallelism — and were dropped
   before landing.
2. **~~`revisit/mc-v512`: closest to landable.~~ Landed 2026-10-12** — the vertical specialization
   (const-generic AVX-512 ntap inner for 4/6/8 taps) landed on top of the tracker stage; the paired
   warp kernel did not (negative-destination oracle failure, `docs/MC_WARP_PAIR_REVIEW.md`). Measured:
   untracked 4K −1.5/−1.6%, aom1080/svt −0.65 to −1.4%; tracked and 10-bit neutral. Zen 5 not measured.
3. **`revisit/mc-fused`: prototype for the biggest item.** MC is 32-42% of the gap and dav1d's advantage is fused
   AVX-512 6-tap kernels. This branch proves the shape (horizontal rows consumed by the vertical pass through a
   four-row ring, no mid buffer) but only for 4x4 four-tap blocks, so the gain is 0-1.6%. Extend it to wider blocks and
   six-tap before judging.
4. **~~`revisit/tracker`: fold into `cdef`.~~ Landed 2026-10-12** — the retained-row
   `cdef_block_mut` machinery (244 lines of `picture.rs` + five call-site swaps) landed on top of the
   CDEF stage. The shared line-buffer parts were already dropped with `cdef`. Measured tracked wins:
   4K 4-worker −4.0%, aom1080 4-worker −1.3%, aom10 1-worker −4.1%; no clean-cell regressions.
5. **`revisit/glue`: drop.** Neutral, three changed lines.

## Other open items

- **Aligned storage.** `AlignedVec64` (about 75 `u8` buffers plus the refmvs arrays) is only needed for
  the `asm` / `c-ffi` builds; the safe SIMD uses unaligned loads. Paired A/B of 64-byte vs 1-byte alignment
  (2026-10-11, r7900x / Zen 4, tip `d34abb60`, four clips, 1 and 4 workers, both modes, with A/A controls):
  median ratios are 0.987-1.012 in 15 of 16 cells; the sixteenth (safe, svt 1080p, 4 workers) is 1.042 against
  an A/A control of 1.059. Alignment costs nothing measurable. Next: plain `Vec<u8>` / `Vec<T>` in the default
  build (the aligned buffer stays for `asm` / `c-ffi`, which need it); the refmvs types need no 64-byte
  alignment. Not done yet. The same session's api-vs-`f3132ee6` timing is usable only for the untracked cells
  (0.995-1.011); the tracked cells ran at load 18-19 and their A/A minimum ratios span 0.794-1.386, so they show
  nothing either way.
- **Smaller public API of rav1d-disjoint-mut.** `dangerously_unchecked` is a documented `unsafe` constructor
  that only a probe example uses. Folding the crate into rav1d-safe as a private module would leave one
  published API, at the cost of its independent Miri / loom / CI setup.
- **Loom and Miri coverage of the audit fixes.** No Loom model covers the wide-id capacity path, and
  `lock_liveness_after_panic` is `cfg(not(miri))`.
- **CI for rav1d-disjoint-mut runs the dev profile only**, so release-only overflow behavior (audit A-01) is
  pinned by a message check, not a release job.
- **`guard_move_release`** has a native contention assertion that fails when the box is busy or the test is
  CPU-throttled; it is a Miri gate.
- **Gap to dav1d** (last clean 2026-10-06, Zen 4): untracked 1.7-2.2x, tracked 2.0-2.8x, libgav1 1.45-1.65x;
  MC + CDEF are 54-62% of the gap. See `docs/SAFE_VS_DAV1D.md`.
- **ARM:** `CpuLevel::Scalar` does not disable baseline NEON dispatchers (applicability review A6).
- **Zen 5:** none of the timing here was measured on it.
