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
| `revisit/mc-v512` | AVX-512 active-row vertical MC + warp pair | Best of the five. Most cells 0.98-0.99 of main, 4K 0.974-0.981; the two 10-bit untracked cells are contaminated (A/A 1.10 / 1.25) | Retime on a quiet box and on Zen 5; run nextest + corpus; land if the 1-2% holds |
| `revisit/cdef` | CDEF direction-cost / padding variants | Tracked 0.973-1.015, a few 1-2% cells, mostly inside noise | Retime quiet; gate before deciding |
| `revisit/tracker` | Retained-row CDEF borrows in the tracked build | Neutral (A/B 0.981-1.025 vs A/A 0.976-1.021) | Drop unless a 4K 4-worker tracked run shows a gain |
| `revisit/mc-fused` | Fused small-block MC (4x4 blocks, four-tap filters only) | Retimed 2026-10-10 (load 2-10): 1-worker cells with a clean A/A are 0.992/0.995/1.003/1.008 tracked and 0.984/0.999 untracked, i.e. 0-1.6%; four outlier untracked cells (0.82, 0.895, 1.135) have a contaminated `before`. The lead's own number was 0-2% | The idea (a ring of four horizontal rows instead of a pooled mid buffer) is the prototype for a general fused H+V kernel; not worth landing alone |
| `revisit/glue` | `decode_b` initialization | Timed 2026-10-10: neutral. 1-worker A/B 0.996-1.011 tracked, 0.997-1.004 untracked; the 4-worker cells that moved 2-5% have off-nominal A/A controls (load reached 10). The change only moves where one `Av1Block::default()` is built (five stores, frame-threaded passes only) | Drop. The ~25% glue share needs profile-driven work (`decode_b`, `refmvs`, `lf_mask`, bounds checks, `block_mut`), not this |

## Which of the five is good groundwork (2026-10-10)

Judged on measured effect, what each leaves behind that is reusable, and how much of the gap to dav1d
it addresses. Timing is Zen 4 only, nextest and the corpus have not been run on any branch, and the
assessment of design is from reading the diffs.

1. **`revisit/cdef`: best groundwork.** CDEF is 12-30% of the gap (padding ~7x, `find_dir` ~5.6x, filter ~3x).
   The branch brings the largest reusable asset: a full-domain parity harness that checks the new CDEF
   padding against the verbatim original (every edge-flag combination, both block shapes, positive and
   negative strides, with and without a declared row stride), plus a cost oracle for the direction finder
   and exact-window rectangle borrows with row-pitch hints. Measured gain is small and mostly inside noise.
   To use it: pick one of its two `find_dir` cost variants (explicit AVX2 vs contiguous loop), run nextest and the
   corpus, retime on a quiet box.
2. **`revisit/mc-v512`: closest to landable.** 129 changed lines of `mc.rs`: AVX-512 vertical MC specialised by active
   tap count (4/6/8) with const generics, plus two parity test files (active rows, paired warp). Bit-exact on the four
   clips; most cells 1-2% faster. It helps the AVX-512 tier only (double-pumped on Zen 4; Zen 5 not measured).
3. **`revisit/mc-fused`: prototype for the biggest item.** MC is 32-42% of the gap and dav1d's advantage is fused
   AVX-512 6-tap kernels. This branch proves the shape (horizontal rows consumed by the vertical pass through a
   four-row ring, no mid buffer) but only for 4x4 four-tap blocks, so the gain is 0-1.6%. Extend it to wider blocks and
   six-tap before judging.
4. **`revisit/tracker`: fold into `cdef`.** It rewrites the same CDEF padding borrows and adds 244 lines to
   `picture.rs`; neutral on its own. Keep only what `cdef` lacks, if anything.
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
