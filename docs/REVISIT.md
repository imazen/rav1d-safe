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
| `revisit/mc-fused` | Fused small-block MC | Inconclusive: the run overlapped other jobs (load 21, A/A 0.79-2.00). The lead's own timing was 0-2% | Retime on a quiet box |
| `revisit/glue` | `decode_b` initialization | Never timed (stopped early) | Time it; it is the only candidate aimed at the ~25% glue share of the gap to dav1d |

## Other open items

- **Aligned storage.** `AlignedVec64` (about 75 `u8` buffers plus the refmvs arrays) is only needed for
  the `asm` / `c-ffi` builds; the safe SIMD uses unaligned loads. A paired A/B of 64-byte vs 1-byte
  alignment is queued (`paired-noalign`). If it is free: plain `Vec<u8>` / `Vec<T>` in the default build, and
  keep the aligned buffer only for `asm`/`c-ffi`. If it helps: a zerocopy `Vec<#[repr(align(64))] Chunk>` for
  `u8` only (no unsafe); the refmvs types need no 64-byte alignment.
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
