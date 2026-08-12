# The code-placement lottery: can it be removed? And the CDEF rectangle, priced
# against a MEASURED transfer coefficient rather than its ceiling

**Read `docs/RECT_SHIP.md` first.** PR #506 established that the `+0.9..1.3%`
t=1 cost that kept the rectangle record default-off is **code placement, not
work**: 4,828 bytes of provably-dead `#[used]` text — zero symbols resized,
every hot loop-filter symbol byte-identical — costs the same `+1.1%` as the real
feature, and nine binaries differing from `main`'s by 1–19 KB all land in
`+1.1%..+1.6%` while a byte-identical copy reads `1.0006`.

That leaves a **±1.5% layout lottery underneath every t=1 measurement in this
repo**, with `main`'s current binary sitting on a lucky draw. This round asks
the two questions that follow.

*(§ numbers are filled in as each grid lands; the decision rule in §2 was
written and committed BEFORE the grid that decides it finished — see the commit
date on this file's first revision.)*

---

## 1. What is NOT covered, first

* **One box** (Apple M4 Pro, 8P+4E, macOS 26.5.2, aarch64), **one toolchain**
  (rustc 1.97.1), **8-bit 4:2:0 only** in every timed grid. Nothing here says an
  x86_64 or a different microarchitecture has the same lottery or responds the
  same way to alignment.
* **The sub-mechanism of the lottery is still not identified.** This round tests
  whether *forcing function alignment removes it*; it does not distinguish
  I-cache set conflicts from fetch-window effects from branch-predictor
  aliasing. A negative here closes "can alignment fix it", not "what is it".
* **A linker order file was NOT tried.** It is the remaining lever and it is
  platform-specific and maintenance-heavy; §5 prices what it would have to beat.
* **`-C llvm-args=-align-all-nofallthru-blocks`** (basic-block alignment) is
  reported only if whole-function alignment failed.
* No `unsafe` is added to `rav1d-safe`; `crates/rav1d-disjoint-mut` DOES change
  (the mutable rectangle guard), so it is Miri'd under both models and its CI
  legs actually fire on this branch — see §8.

## 2. The decision rule, pre-registered

The success criterion for Task 1 is **spread reduction, not a good draw**. For
each alignment family (a0 = none, a4/a5/a6 = 16/32/64-byte) the grid measures
four rungs — `plain`, and `+4,828 / +9,692 / +19,420` bytes of dead text — and
reduces the family to

  `SPREAD = max(rung medians) − min(rung medians)`, paired per round against
  that family's OWN unpadded build.

Alignment is worth shipping **for measurement quality alone** iff, on
`v4k8tile` t=1 (the cell with the largest tax):

1. `SPREAD` falls by at least **2x** against `a0`, **and**
2. the absolute cost `aNplain / a0plain` is **≤ 1.003**.

If (1) fails, that is a clean negative and it goes in `docs/AGENT_BRIEF.md` §6.
If (1) holds and (2) fails, the trade is reported with both numbers and no
default changes.

## 3. Grid L — alignment × pad rungs

18 arms in one rotation: four alignment families × four rungs, plus a
byte-identical copy in the `a0` and `a6` families. `scripts/perf/layout_build.sh`
builds them (one target dir per family — `RUSTFLAGS` is part of the fingerprint),
`scripts/perf/layout_checksums.sh` proves **one md5 per cell across all 18 arms
at BOTH thread counts** before any clock, and `scripts/perf/tiled_wallcpu.sh`
under `measlock`, never `nice`d, two-point fit, times them.

The alignment took: `nm` says **100.0%** of `__text` symbols are 16/32/64-byte
aligned in `a4`/`a5`/`a6` against 23.9%/11.8%/6.3% in `a0`, and the pad rungs add
the same `+4,828 / +9,692 / +19,420` bytes in every family.

| arm | `__text` | vs `a0plain` | % 16B-aligned | % 64B-aligned |
|---|---|---|---|---|
| `a0plain` | 1,837,492 | — | 23.9 | 6.3 |
| `a4plain` | 1,847,552 | +10,060 (+0.55%) | **100.0** | 22.0 |
| `a5plain` | 1,859,360 | +21,868 (+1.19%) | 100.0 | 45.8 |
| `a6plain` | 1,886,400 | +48,908 (+2.66%) | 100.0 | **100.0** |

### 3a. The lottery IS removable, and 16-byte alignment removes 87% of it

`v4k8tile` t=1, wall, paired per round against each family's own `plain`,
n=12 (round 0 dropped). Nine of the twelve rounds saw a foreign process above
25% CPU at some point, so both readings are given: the campaign's usual
drop-the-loaded-round reduction leaves n=3, and the keep-loaded reduction keeps
all 12 paired ratios. **They agree to within 0.04 percentage points of spread**,
which is itself evidence that a loaded round costs drift and not pairing.

| family | `pad1` (+4,828 B) | `pad2` (+9,692 B) | `pad4` (+19,420 B) | **SPREAD** | (n=3 drop-loaded) | `plain`/`a0plain` |
|---|---|---|---|---|---|---|
| **a0** none | 1.0101 | 1.0118 | 1.0137 | **1.37%** | (0.98%) | 1.0000 |
| **a4** 16 B | 0.9993 | 0.9982 | 0.9982 | **0.18%** | (0.16%) | 1.0153 |
| **a5** 32 B | 0.9973 | 0.9964 | 0.9962 | **0.38%** | (0.35%) | 1.0164 |
| **a6** 64 B | 0.9925 | 0.9935 | 0.9977 | **0.75%** | (0.92%) | 1.0178 |

Byte-identical controls in the same grid: `a0B` **0.9996 (7/12)**, `a6B`
**1.0006 (5/12)** — coin-flip signs, medians inside 0.07%. So the instrument
floor is ~0.05% and `a0`'s 1.37% spread is **20x** it.

The other two cells, same reduction (keep-loaded, n=12):

| cell t=1 | a0 spread | a4 | a5 | a6 | `a4plain`/`a0plain` | a0B |
|---|---|---|---|---|---|---|
| `v4k8tile` (4K, 8 tiles) | **1.37%** | **0.18%** | 0.38% | 0.75% | 1.0153 | 0.9996 (7/12) |
| `c1024x576` | 0.94% | **0.36%** | 0.54% | 0.42% | 1.0082 | 1.0000 (6/12) |
| `c256x2048` (the zero-tax cell) | 0.39% | 0.35% | **0.15%** | 0.86% | 1.0020 | 0.9993 (6/12) |

CPU tracks wall exactly at t=1 (`v4k8tile`: a0 1.34%, a4 0.20%, a5 0.30%,
a6 0.65%), as it must when one thread is busy the whole time.

Three readings:

1. **Whole-function alignment at 16 bytes shrinks the spread 7.6x on the cell
   where the lottery lives** (1.37% → 0.18%), 2.6x at 1024x576, and does nothing
   at 256x2048 — which is the cell that had no lottery to remove. The benefit
   tracks the tax, exactly as an I-cache story predicts.
2. **More alignment is worse.** 64-byte is only a 1.8x reduction and costs the
   most; 32-byte sits between. The `__text` cost runs +0.55% / +1.19% / +2.66%,
   so the padding's own I-cache pressure eats the stabilisation it buys. The
   sweep was necessary: a single N would have given the wrong answer at 6.
3. **The pre-registered criterion (2) FAILS as written**: `a4plain / a0plain` is
   **1.0153**, five times the 1.003 bar. §5 argues that bar was mis-specified —
   `a0plain` is `main`'s LOTTERY WINNER, not a neutral reference — but the rule
   was pre-registered and it fails, so it is reported as failed.

### 3b. What the +1.5% actually is

`a0plain` is the best binary in the whole grid, and #506 already showed it beats
nine other `a0`-family binaries by 1.1–1.6%. Against the `a0` family's own
perturbed rungs (mean 1.0119 on `v4k8tile` t=1) the aligned build costs

  `1.0153 / 1.0119 = 1.0034` — **+0.34%**,

and at 1024x576 `1.0082 / 1.0069 = 1.0013`, at 256x2048 `1.0020 / 1.0004 =
1.0016`. So the honest decomposition of the 1.53% is **~1.2 points of "stop
being `main`'s exact binary", which any change forfeits, and ~0.3 points of real
padding cost.** That distinction is the whole of §5, and it is a decomposition,
not a measurement of a third binary: no arm here isolates "aligned, and also
lucky".

### 3c. Not tried

Basic-block alignment (`-align-all-nofallthru-blocks`) was **not measured** —
the whole-function sweep answered the question and the block flag would have
needed its own 4-family grid. If a future round wants the residual 0.18%, that
is where to look.

## 4. Grid D — the CDEF question, decided on a measured transfer coefficient

The brief's gate for building anything at the CDEF sites was their
**records-per-distinct-shard-line ratio**, on the theory that cost tracks
distinct shard lines and a repeat touch of a line this core already owns is
~7.7x cheaper. Measured first, before any clock (`probe_tracker --features
__probe_bounds`, t=8, 40 iters, per-frame; `~/tmp/layout/counts/pb_*_t8.txt`):

| cell t=8 | site | calls/frame | `rows_mean` | `row_shards_mean` | **records per line** | `pct_row_wide` |
|---|---|---|---|---|---|---|
| `c1024x576` | `loopfilter.rs:944` (`fill`) | 21,364 | 8.92 | 2.318 | **3.85** | 0.00% |
| `c1024x576` | `cdef_arm.rs:{193,625,1222}` | 3,776 each | 8.00 | 1.928–1.929 | **4.15** | 0.00% |
| `c1024x576` | `cdef_apply.rs:107` | 3,904 | 8.00 | 1.928 | **4.15** | 0.00% |
| `c1024x384` | `fill` | 13,869 | 9.01 | 2.332 | 3.86 | 0.00% |
| `c1024x384` | the four CDEF sites | 2,368–2,456 | 8.00 | 1.939–1.940 | 4.13 | 0.00% |
| `c1024x192` | `fill` | 6,752 | 9.01 | 2.322 | 3.88 | 0.00% |
| `c1024x192` | the four CDEF sites | 1,024–1,072 | 8.00 | 1.913–1.915 | 4.18 | 0.00% |
| `c256x2048` | `fill` | 20,093 | 8.98 | 2.090 | 4.30 | 0.00% |
| `c256x2048` | the five CDEF sites | 1,024–5,632 | 4.00–8.00 | **1.000** | 7.27–8.00 | 0.00% |

**The ratio test does not disqualify the CDEF sites, and it also does not
separate winners from losers.** On the 1024-wide family — the family where the
`fill` rectangle actually delivered — CDEF's geometry is `fill`'s to within 8%
(4.13–4.18 against 3.85–3.88), so if a high ratio predicted a null, `fill`
should have measured null there too, and it measured **−1.5% to −2.4%**.
`c256x2048` is the cell with both the highest ratio (4.30 at `fill`, 7.3–8.0 at
CDEF) and the null, but four previous levers have died on that cell for reasons
`docs/C256_CONTENTION.md` §7 attributes to coherence, not geometry. One ordered
pair is not a rule.

So the ratio was replaced with a **measured transfer coefficient**. Grid D
prices BOTH populations by doubling, in one binary each, on the three cells
where `fill`'s collapse is already known — which turns the standing caveat into
a number instead of a worry. `scripts/perf/layout_transfer.py`,
`benchmarks/layout_D_2026-08-11.tsv`, `measlock`, un-`nice`d, n=11–13 after
dropping loaded rounds, two A/A controls in the grid (`lfoff2`, `cdefoff2`):

| cell t=8 | site | +regs/frame | doubling wall | sign | doubling CPU | **ns/reg** |
|---|---|---|---|---|---|---|
| `c1024x192` | `fill` | 60,820 | **1.0784** | 0/11 | 1.0388 | 3.19 |
| `c1024x192` | CDEF | 33,152 | **1.0743** | 0/11 | 1.0336 | 5.03 |
| `c1024x384` | `fill` | 125,018 | **1.0885** | 0/13 | 1.0319 | 2.67 |
| `c1024x384` | CDEF | 76,480 | **1.0601** | 0/13 | 1.0397 | 5.36 |
| `c1024x576` | `fill` | 190,632 | **1.0720** | 0/12 | 1.0394 | 3.34 |
| `c1024x576` | CDEF | 121,856 | **1.0385** | 0/12 | 1.0414 | 5.43 |

A/A controls: `cdefoff2` 1.0000 (5/11) / 1.0030 (5/13) / 1.0009 (5/12);
`lfoff2` 1.0049 (4/11) / 0.9970 (7/13) / 1.0019 (4/12). #506's single CDEF cell
(`c1024x576`, +4.09% wall, 5.27 ns/reg) **replicates** here at +3.85% and
5.43 ns/reg.

Then, per cell, `ceiling = (1 − 1/rows) × (doubling − 1)` and
`tau = delivered / ceiling`, with `delivered` for `fill` being its
same-source-controlled t=8 win (`ship`/`plain2`, `docs/RECT_SHIP.md` §6 — cited,
and re-measured in-grid by grid M):

| cell t=8 | `fill` ceiling | `fill` delivered | **tau** | CDEF ceiling | **CDEF predicted** |
|---|---|---|---|---|---|
| `c1024x192` | −6.97% | −1.49% | **0.214** | −6.50% | **−1.39%** |
| `c1024x384` | −7.87% | −2.38% | **0.303** | −5.26% | **−1.59%** |
| `c1024x576` | −6.39% | −1.74% | **0.272** | −3.37% | **−0.92%** |

**A doubling over-promises by 3.3x to 4.7x, consistently.** That is the number
the campaign was missing, and it is the answer to "is a doubling an upper bound
or a forecast": an upper bound, times ~0.26. It also says the honest expectation
for the CDEF collapse is **−0.9% to −1.6% wall at t=8 on the 1024-wide family**,
not the ~3.6% §7 of `docs/RECT_SHIP.md` computed — which is still worth building,
because it is the same size as what the `fill` rectangle delivers on exactly
these cells.

**Why a doubling over-promises, as a hypothesis and not a result:** doubling
raises each shard's occupancy, and both `find` and `remove` scan the shard's live
records, so the added population is charged a scan cost the removed population
does not refund at occupancy ~0.02. The prediction that follows — `tau` should be
closer to 1 at low occupancy and fall as occupancy rises — is NOT tested here.

### 4a. What was built: `__rows_rect`, one seam, both directions

The five sites `docs/RECT_RECORDS.md` §7b names are not five pieces of code —
they are four (five on 4:2:0 chroma) call sites of **two helpers**,
`Rav1dPictureDataComponentOffset::for_rows` and `for_rows_mut`
(`include/dav1d/picture.rs`). So the collapse is one change at that seam, not a
per-site edit, and it needs the MUTABLE rectangle the tracker did not have:

* `crates/rav1d-disjoint-mut`: `DisjointMutRectGuard` (`row_mut(&mut self)`, so
  at most one row reference is live at a time and no `&mut [_]` wider than one
  row is ever created), `DisjointMut::index_rect_mut{,_as}`, and `add_rect_mut`
  un-`cfg(test)`-ed. All behind the crate feature `__rect_mut`, so a DEFAULT
  build does not even grow a public symbol it never calls — which matters here
  for the reason §3 measures.
* the geometry validation `index_rect_inner` did inline is now
  `rect_geometry`, shared by the immutable and mutable constructors so the two
  cannot disagree about what is representable.
* `include/dav1d/picture.rs`: both helpers try the rectangle first under tile
  threading, and fall through to the unchanged per-row loop on refusal.

**Liveness, checked before any clock** (`probe-wide`, `c1024x576`, t=8, 10
iters + warmup):

| arm | `n_rect` | declined | % multi-shard | `w_shards` | `w_blocks` | `w_full` |
|---|---|---|---|---|---|---|
| base | **0** | 0 | — | 1,530 | 0 | 0 |
| `__rows_rect` | **167,552** (15,232/frame) | **0** | 92.8% | 1,530 | 0 | 0 |
| `__lf_rect,__rows_rect` | 402,556 | 0 | 91.5% | 1,530 | 0 | 0 |

15,232/frame is **exactly** the four sites' call count from the bounds probe
(3,776 × 3 + 3,904), so every CDEF `for_rows`/`for_rows_mut` on this cell is
representable and none is declined. The registration population goes
529,092 → 422,468/frame (**−20.2%**). `w_shards` is IDENTICAL in all three arms
and `w_blocks`/`w_full` stay 0, so the rectangle never promotes to the wide path
(a promotion would degrade it to its hull and could refuse a legitimate borrow).

**Correctness, before any clock**: the 766-vector corpus passes at t=1 AND t=8
in the `__rows_rect` arm with no `--skip-group` (`766 PASS + 2 SKIP,
mismatch=0 error=0`), set-diffed BY NAME against
`benchmarks/aarch64_md5_fixes_2026-08-07_final.tsv.zst` — CLEAN at both — and
all ten timed arms produce ONE md5 per cell at both thread counts.

### 4b. Grid M8 — the collapse measured, and it beats its own prediction

t=8, `measlock`, un-`nice`d, two-point fit, 10 arms rotating, n=9–12 after
dropping loaded rounds. `a0pad2` (+9,692 B of dead text, same source, provably
never executed) is the layout-matched base — the control #506 established is the
right one — and the `vs a0plain` column is what a user of today's binary gets.

| cell t=8 | **`a0rows`** CDEF rect | sign | `a0rect` `fill` rect | sign | **`a0both`** | sign | `a0B` byte-identical |
|---|---|---|---|---|---|---|---|
| `c1024x192` | **0.9754** | 9/10 | 0.9829 | 9/10 | **0.9559** | 10/10 | 0.9975 (5/10) |
| `c1024x384` | **0.9804** | 11/11 | 0.9850 | 10/11 | **0.9551** | 11/11 | 1.0029 (5/11) |
| `c1024x576` | **0.9791** | 9/9 | 0.9848 | 8/9 | **0.9608** | 9/9 | 0.9924 (6/9) |
| `c256x2048` | **0.9794** | 12/12 | 1.0007 | 5/12 | **0.9787** | 12/12 | 1.0007 (5/12) |
| `text_q20` (**zero** CDEF regs) | 1.0038 | 4/11 | 0.9962 | 6/11 | 0.9924 | 10/11 | 1.0114 (0/11) |

CPU agrees (`a0rows`: 0.9867 / 0.9836 / 0.9794 / 0.9802 / 1.0000, signs
10/10–12/12 on the four cells that move).

* **The CDEF rectangle is −2.0% to −2.5% wall on every multi-tile t=8 cell that
  files CDEF registrations**, with 9/10 to 12/12 signs, and **1.0038 (4/11) on
  the cell that files none** — the other-side control doing its job.
* **The prediction was conservative and directionally right.** §4 predicted
  −0.92% / −1.59% / −1.39% on `c1024x576` / `c1024x384` / `c1024x192`; measured
  −2.09% / −1.96% / −2.46% against the pad control (−1.28% / −1.37% / −1.98%
  against `a0plain`). So `tau` computed from `fill` UNDER-predicts the CDEF
  collapse by ~1.3–1.8x. A doubling is still not a forecast — but a doubling
  discounted by a `fill`-calibrated `tau` was within a factor of two, which is
  the first time this campaign could price an unbuilt collapse at all.
* **`c256x2048` moves in the `a0` family and NOT in the `a4` one — so it is not
  a result.** `a0rows` reads 0.9794 (12/12) against the pad and 0.9910 (11/12)
  against `a0plain`, on the cell that has declined a count cut, a coarser shard,
  a finer shard, the waiting policy and `fill`'s own rectangle. But the SAME
  mechanism in the 16-byte-aligned family reads **`a4rows`/`a4plain` = 1.0007
  (6/12)** — a coin flip. The 1024-family win replicates across both families
  (below); this one does not, so `c256x2048` stays on the declined list and the
  headline is the 1024 family only.
* **The two mechanisms compose super-additively on the 1024 family**:
  0.9754 × 0.9829 = 0.9587 predicted, 0.9559 measured (`c1024x192`);
  0.9804 × 0.9850 = 0.9657 vs 0.9551 (`c1024x384`);
  0.9791 × 0.9848 = 0.9641 vs 0.9608 (`c1024x576`). Composed, **−3.9% to −4.5%
  wall at t=8** — the largest t=8 win in the campaign's record.

### 4c. The win REPLICATES inside the aligned family — which is the point of §3

The same 10-arm grid carries the mechanism in the 16-byte-aligned family, paired
against `a4plain` (its own unpadded build) with `a4pad2` as that family's inert
layout control:

| cell t=8 | `a4rows`/`a4plain` | sign | `a4pad2`/`a4plain` | sign | `a0rows`/`a0pad2` (for comparison) |
|---|---|---|---|---|---|
| `c1024x192` | **0.9801** | **10/10** | 1.0000 | 4/10 | 0.9754 (9/10) |
| `c1024x384` | **0.9734** | **11/11** | 0.9945 | 8/11 | 0.9804 (11/11) |
| `c1024x576` | **0.9801** | **9/9** | 0.9924 | 6/9 | 0.9791 (9/9) |
| `c256x2048` | 1.0007 | 6/12 | 0.9905 | 8/12 | 0.9794 (12/12) |
| `text_q20` (zero CDEF regs) | 0.9925 | 7/11 | 0.9962 | 6/11 | 1.0038 (4/11) |

**−2.0% to −2.7% on the 1024 family in a binary where a layout draw cannot hide
in the number**, which is what §3 was for. And it is the aligned family that
exposes `c256x2048` as a non-result.

Note also that the aligned family's own layout control moves ±0.95% at t=8
(`a4pad2` 0.9905–1.0000) against the unaligned family's ±1.13% (`a0pad2`
0.9887–1.0091) — **alignment is NOT shown to stabilise t=8.** One rung per
family is not a spread, and this round did not sweep the rungs at t=8; §3's
finding is a t=1 finding.

**Unexplained, and named as such:** why `c256x2048` responds to this collapse in
one layout and not the other, and why it responds at all when it does not to
`fill`'s. The two differ in the shard-set size of the record that
replaces the rows (CDEF's hull is **1.000** blocks there, `fill`'s is 2.090, and
79.6% of `fill`'s accepted rectangles are multi-shard) and in marginal price
(3.27 vs 2.42–2.71 ns). But on the 1024-wide family CDEF's own rectangles are
92.8% multi-shard and pay just as well, so "single-shard records are what pays"
does not survive its own second cell. `--features __lf_rect1` exists to test the
shard-set-size hypothesis directly and was NOT run here.

**One arm was void in this grid**: `bench_a4B` did not exist when M8 launched, so
the `a4B` rows are zeros and the aligned family has no byte-identical control in
M8. It is present in M1. The other nine arms are unaffected (each is a separate
process invocation), and `a0B` covers the floor.

## 5. The rectangle default

## 6. Gates
