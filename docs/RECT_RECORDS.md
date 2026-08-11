# The exact strided-rectangle record: built, sound, live — and at the noise floor
# on the cell it was built for, with the cost model it corrects

**Status: NOT SHIPPED, at the noise floor, and the number that matters is the
one it corrects.** The mechanism this round was opened to test — collapsing
`LfBlock::fill`'s `h` per-row registrations into ONE exact strided-rectangle
record, the third shape after the per-row split and the refuted hull — works, is
exact, is sound at t=8, is live, and measures **−1.2% to −1.4% wall with CPU
flat** on `c256x2048` at t=8, against an identity-control band of ±2-3%.

The reason is a measured price, not a mystery. **A `LfBlock::fill` per-row
registration costs 2.42-2.71 ns at the margin** — measured by ADDING a duplicate
per row in the same binary (`RAV1D_LF_DOUBLE`), which is the only sound way to
price a registration on a contended path — against this cell's **19.71 ns/registration
AVERAGE**. So the largest registration site in the decoder, **31.7% of the
population, is 3.9-4.4% of the tracker's CPU**, and removing 89% of it can only
ever have been worth ~1.5% of wall. It delivered about that.

**Therefore the campaign's cost model needs one word changed, and it is the word
that decides the next lever.** `docs/AGENT_BRIEF.md` §6 records the `c256x2048`
residual as "about one cross-core transfer of the shard's own cache line **per
registration**". Per registration it is not: `fill`'s `h` registrations are 8.98
back-to-back touches of the same **2.09** shard lines from one core, and a repeat
touch of a line this core already owns is ~7.7x cheaper than the average
registration. **The expensive registration is the one whose shard line another
core has taken since — so the cost tracks the number of DISTINCT shard lines a
worker visits, not the number of records it files.** That is the same conclusion
`docs/C256_CONTENTION.md` §4 reached from the granularity side ("the money is the
shard-line footprint"), now measured from the count side, and it means **a count
cut at the DENSEST site is the least valuable count cut available.**

Record: `benchmarks/rect_records_2026-08-11.{meta,*.tsv}`.
Prior art, not re-derived: `docs/C256_CONTENTION.md` (the four refuted levers
and the ns/reg ladder), `docs/BOUNDS_MAP.md` (the rectangle counterfactual and
why the hull is refuted), `docs/BPS_ROWS_DEFAULT.md` (the shipped shift rule).

---

## 1. What is NOT covered, first

* **The cell is not closed.** It sits where `docs/C256_CONTENTION.md` left it,
  at ~2.38x of dav1d at t=8 against a tracker-removed ceiling of ~1.33x, and
  this round moved neither.
* **The rectangle path is NOT the default and is NOT proposed as one.** It sits
  behind `--features __lf_rect`, absent from `default` and from every published
  feature. The brief this round was written to says a winning mechanism should
  ship with its feature deleted; it did not win, so the feature stays as the
  arm it is.
* **Only `LfBlock::fill` was routed through it.** The counterfactual names five
  more sites on this cell whose rectangles are cheaper still on paper
  (`cdef_apply.rs:104:32` and `:121:33`, `safe_simd/cdef_arm.rs:{192,622,1217}`
  — 7.27-8.00 rows on **1.000** shards, so a 7-8x count cut each). They were
  NOT wired, and after `fill`'s result the expected value of doing so is a
  fraction of a null. See §7.
* **No mutable rectangle ships.** `add_rect_mut` is `#[cfg(test)]`; the only
  consumer is the exact rectangle-vs-rectangle detection test, which cannot be
  reached with two immutable records. `compact_write_back`'s per-row MUTABLE
  guards are untouched.
* **One box** (Apple M4 Pro, 8P+4E, macOS 26.5.2, aarch64), **one content
  class** for the primary cell, **8-bit 4:2:0 only** in the timed grid. No
  x86_64, where a locked RMW is a full fence and the multi-shard rectangle's
  extra lock traffic would be dearer, not cheaper — so this round says nothing
  about whether it is *worse* there.
* **`RGB16`/10-bit rectangles are exercised only by the corpus**, not by the
  timed grid.
* **Miri's `shard_liveness` target times out locally** and is reported as a
  timeout, never as green (§6c).
* **The machinery's own cost is measured but small and confounded.** See §5b:
  the same-tree isolation reads `PLACEHOLDER_MACH`, the different-tree build of
  the base commit read −1.9% wall / +0.5% CPU, and a wall-down/CPU-up pair is
  the signature of code layout rather than of instruction count.

## 2. The mechanism, and why it is neither of the two refuted shapes

`LfBlock::fill` copies `h` rows of exactly `W` pixels out of a picture plane
into scratch. It is the largest single registration site in the decoder:
**180,434 of 569,690 registrations per frame (31.7%)** on `c256x2048` at t=8,
one per row.

Two earlier shapes for collapsing that are refuted, in opposite directions:

| shape | what it does | why it fails |
|---|---|---|
| **hull** (`fill_hull`, live only when `!tile_threading_active()`) | reserves `[lo, lo + (h-1)*stride + W)` | reserves the inter-row GAPS, which belong to other columns of the same picture rows. Two tile workers routinely write the same rows at different columns, so the gap reservation converts a genuinely disjoint pair into a **false positive** — measured as 162/358 decode ERRORS under `__probe_rect_hull`, and 2.65x slower where it does pass |
| **March-2026 strided tracker** | exact record, reference over the whole hull | safe code held a `&[T]` covering bytes another thread was mutating: **Miri UB under both memory models**, and a real CI decode failure |

The rectangle record is the third shape and avoids both:

* **The record covers only the segments.** `rect_hit_range` walks rows; it knows
  nothing of the gaps, so a concurrent writer in a gap is not reported. The
  false positive that gates `fill_hull` cannot arise.
* **No reference wider than one row is ever created.** `DisjointImmutRectGuard`
  has no `Deref`; `row(r)` derives that row's slice from the buffer's own
  pointer. The gaps are outside the aliasing model as well as outside the
  tracker.

### 2a. Storage is free, and that is the one genuinely new idea

A record is still the two words a plain interval always was. It stores the
rectangle's **hull**, and `(rows, seg)` is *recovered* from the hull and the
instance's declared row stride:

```text
    span = h1 - h0 = (rows - 1) * s + seg,   1 <= seg <= s
 => span - 1       = (rows - 1) * s + (seg - 1),   0 <= seg - 1 < s
 => rows - 1       = (span - 1) / s          (exact, unique)
 => seg            = span - (rows - 1) * s
```

That is a bijection, not a widening, and it is why `Shard` stays **exactly 128
bytes** with no side table and no second cache line. The per-slot `mutable: u8`
bitmap became a `flags: u16` whose high byte flags rectangle records, so
`alloc`'s empty-shard arm — the measured steady state at occupancy 0.02 — still
publishes both bitmaps with a **single store** and `find` still loads them with
a single load.

`s` comes from `BorrowTracker::row_stride`, set by the existing
`DisjointMut::declare_row_stride` that every picture plane already calls (it
previously fed only the block-shift rule). It moves only under `&mut self`, so
both registrants of a shared byte read the same value, exactly as they do for
`shift` and `mask`.

### 2b. Exactness in both directions, and no approximation anywhere

* **Shard SELECTION uses the hull's blocks.** That is a *superset* of the blocks
  the rows occupy, which is sound: the module header's argument needs only that
  a shared byte's block is in both registrants' shard sets, and a superset
  cannot break that. Under the shipped rows rule a block spans >= 4 picture
  rows, so the two sets are in fact equal (measured: `hull_blocks_mean` ==
  `row_blocks_mean` == 2.090 on this cell).
* **Overlap DETECTION is exact.** Rectangle-vs-interval walks the rows the
  interval can reach. Rectangle-vs-rectangle compares **row by row** rather than
  assuming a common grid — the honest test the brief asks for.
* **Nothing is ever rounded up to make a rectangle fit.** `add_rect` DECLINES,
  and the caller runs its per-row loop unchanged, when: there is no declared
  stride; the caller's stride differs from it; `seg > stride`; `rows >
  MAX_RECT_ROWS` (64); the hull spans more than `MAX_SHARDS_PER_BORROW` blocks;
  a shard is full; or a wide record is live or the tracker is poisoned.
* **Declining rather than promoting is why the wide list needs no rectangle
  support.** A rectangle promoted to the wide list would degrade to its hull and
  could then refuse a legitimate borrow — sound, but a spurious panic, which is
  not acceptable either. Measured: `w_shards = w_blocks = w_full = 0` in the
  rectangle arm, identical to base on every cell (§4).

## 3. The counterfactual, priced BEFORE anything was built

`--features __probe_bounds` on `C256x2048_420_8b__t8` at t=8, at the shipped
shift rule (the `benchmarks/strided_2d_rect_2026-08-10.tsv` row for this site
predates the rows rule and reports `pct_row_wide = 50.82%` under the old
block-count shifts — it does not apply):

| site | n | rows_mean | rows_max | hull_blocks | row_shards_mean | row_shards_max | pct_row_wide |
|---|---|---|---|---|---|---|---|
| `loopfilter.rs:809` (`fill`) | 80,372 | **8.98** | 16 | 2.090 | **2.090** | 4 | **0.00%** |
| `cdef_apply.rs:121:33` | 4,096 | 4.00 | 4 | 1.000 | 1.000 | 1 | 0.00% |
| `cdef_apply.rs:104:32` | 18,272 | 8.00 | 8 | 1.000 | 1.000 | 1 | 0.00% |
| `safe_simd/cdef_arm.rs:622:9` | 22,528 | 7.27 | 8 | 1.000 | 1.000 | 1 | 0.00% |
| `safe_simd/cdef_arm.rs:192:9` | 22,528 | 7.27 | 8 | 1.000 | 1.000 | 1 | 0.00% |
| `safe_simd/cdef_arm.rs:1217:9` | 18,432 | 8.00 | 8 | 1.000 | 1.000 | 1 | 0.00% |

Two things the table said, and both held:

1. **The rectangle is narrow on this cell.** `pct_row_wide = 0.00%` at every
   site, so the quantity that refuted the rectangle on the 2026-08-10 corpus run
   (17-52% wide) is zero here. The mechanism was not pre-refuted.
2. **The saving is `rows / row_shards`, not `rows`.** 8.98 / 2.090 = **4.30x** at
   `fill`, and 7.3-8.0x at the CDEF sites. `docs/BOUNDS_MAP.md` already said
   this ("measured saving 1.4-3.7x, not `h`") and it is why the expectation was
   ~24% of the population rather than ~31%.

## 4. The instrument sees the code — checked before any clock

`--features probe-wide,__lf_rect` and `--features probe-sites,__lf_rect`,
`nice`d (these are count-shaped, not contention-shaped), t=8:

| quantity | base | rect | note |
|---|---|---|---|
| registrations / frame (`probe-sites`, `lost = 0`) | **569,690** | **409,349** | **−28.1%** |
| `loopfilter.rs` `fill` site | 180,434 | 20,093 | one per `fill` call, 8.98x fewer |
| effective shard-touches at `fill` | 180,434 | 41,994 | 20,093 x 2.090 shards, **−24.3% of the total population** |
| `n_rect` accepted / frame | 0 | 22,605 | liveness: a timed arm reading 0 here measured nothing |
| `n_rect_declined` / frame | — | **0** | every `fill` on this cell is representable |
| `n_rect_multi` (accepted, >1 shard) | — | **79.6%** | see §5c: this is what eats the saving |
| `w_shards` / `w_blocks` / `w_full` | 0 / 0 / 0 | **0 / 0 / 0** | the rectangle never promotes to the wide path |

Breadth, same instrument (accepted / declined / % multi, per frame at t=8):

| cell | n_rect | declined | % multi | w_shards base -> rect |
|---|---|---|---|---|
| `c256x2048` | 22,605 | 0 | 79.6% | 0 -> 0 |
| `c1024x576` | 24,035 | 0 | 90.6% | 1224 -> 1224 (pre-existing) |
| `c3840x256` | 30,184 | 9,051 (23.1%) | 96.1% | 1728 -> 1728 (pre-existing) |
| `v4k_8tile` | 333,924 | 84,596 (20.2%) | 96.8% | 0 -> 0 |

`w_shards` is **identical** in both arms on every cell, so the nonzero values on
`c1024x576`/`c3840x256` are the shipped decoder's own and not something the
rectangle introduced.

## 5. Timed

`scripts/perf/tiled_wallcpu.sh` under `measlock`, un-`nice`d, two-point fit
`total = a + b*frames` at 22 and 225 frames so process startup drops out of both
wall and CPU, arms rotating inside every round, ratios PAIRED per round and only
then reduced to a median. **Round 0 discarded** (the first touch of each (arm,
cell) pair is cold) and **any round in which an arm saw a foreign process above
25% CPU discarded whole** (`scripts/perf/rect_report.py`; a loaded round keeps
its paired ratios but not its drift, and one such round moved a median 0.7% in
the first pass).

Two sessions, because the machinery was rewritten between them (§5b). The arms:

| arm | what it is |
|---|---|
| `plain` | this branch's default codegen: the machinery present, the rectangle path OFF |
| **`plainB`** | a **byte-identical copy of `plain`** (same sha256). Its spread against `plain` IS this grid's floor and its sign is a coin flip by construction |
| **`machoff`** | 140f914's tracker source built **in this tree** — the machinery removed, everything else identical. `rect / machoff` is the net-versus-`main` number |
| `rect` | `--features __lf_rect` |
| `rect1` | `--features __lf_rect1`: rectangles accepted only when they land in ONE shard (20.4% of `fill`s here) |
| **`dbloff` / `dblon`** | ONE binary (`--features __probe_lf_hull`), `RAV1D_LF_DOUBLE` off and on: takes each per-row guard TWICE, so **+180,434 registrations/frame and nothing else changes**. Sound because both are immutable over the same bytes |
| `untracked` | the tracker-removed ceiling, bit-identical output |
| `dav1d_fd1` | dav1d 1.5.4 `--framedelay 1` |

Every arm's `CHECKSUM` was verified identical before any timing: **9 arms x 2
thread counts -> ONE md5 per cell**, and identical across all seven measured
cells.

### 5a. The result, both sessions, against BOTH references

`C256x2048_420_8b__t8` at t=8. Session A n=13, session A2 n=12; idle box apart
from the discarded rounds (`foreign_max = 1`).

| arm | wall/`plain` | sign | wall/`machoff` | sign | CPU/`machoff` | sign |
|---|---|---|---|---|---|---|
| **A: `plainB` (identity)** | 1.0026 [0.9692..1.0131] | 4/13 | 1.0119 | 2/13 | 1.0132 | 1/13 |
| A: `machoff` | 0.9845 | 10/13 | 1.0000 | — | 1.0000 | — |
| **A: `rect`** | 0.9704 | 13/13 | **0.9881** [0.9650..1.0013] | **11/13** | **1.0008** | 6/13 |
| A: `rect1` | 1.0077 | 3/13 | 1.0199 | 1/13 | 1.0134 | 1/13 |
| A: `dbloff` | 0.9961 | 8/13 | 1.0092 | 3/13 | 1.0174 | 0/13 |
| **A: `dblon`** | 1.0322 | 0/13 | **1.0398** | 0/13 | **1.0380** | 0/13 |
| **A2: `plainB` (identity)** | 1.0066 [0.9744..1.0225] | 3/12 | 1.0039 | 4/12 | 1.0073 | 0/12 |
| A2: `machoff` | 1.0086 | 3/12 | 1.0000 | — | 1.0000 | — |
| **A2: `rect`** | 0.9973 | 8/12 | **0.9856** [0.9778..1.0291] | **10/12** | **1.0031** | 3/12 |
| A2: `rect1` | 1.0164 | 0/12 | 1.0092 | 3/12 | 1.0106 | 0/12 |
| A2: `dbloff` | 1.0153 | 0/12 | 1.0078 | 2/12 | 1.0148 | 0/12 |
| **A2: `dblon`** | 1.0495 | 0/12 | 1.0466 | 0/12 | 1.0346 | 0/12 |

`untracked` reads 0.5470 / 0.5581 wall and dav1d 0.4187 / 0.4259, i.e. the cell
is unmoved at ~2.35x of dav1d against a ~1.33x ceiling.

**Read it three ways:**

1. **The rectangle is worth −1.2% to −1.4% wall against `main`, replicated
   across two sessions with different binaries (11/13 and 10/12 rounds), and
   0.0% CPU.** A wall gain with no CPU gain at 6.7 busy cores is a
   critical-path effect, not a work reduction, and it is the size of the
   identity control's own band (±2-3%). By the standard this campaign applied to
   `lockrelax` in `docs/C256_CONTENTION.md` §5 — a −1.4% point estimate inside a
   ±1.8% band, declared null — **this does not clear the bar.**
2. **`rect1` is WORSE than `plain` in both sessions** (1.0077, 1.0164, signs 3/13
   and 0/12). One-shard-only rectangles are the strictly-cheaper-per-record
   variant, and they cover only 20.4% of `fill`s; the arm says coverage
   dominates and that the multi-shard lock traffic is NOT what eats the win.
   That was the competing hypothesis and it is refuted.
3. **`dblon` is the measurement that explains everything else.** Adding 180,434
   registrations/frame — the same population the rectangle removes, in the same
   binary, changing nothing else — costs **+3.4% to +4.0% wall and +1.8% to
   +3.7% CPU**, with 0 of 25 rounds on the other side. So a `fill` registration
   is worth **2.42-2.71 ns of CPU** at the margin. The rectangle removes 160,341
   of them net and gains what that predicts. **Nothing is broken; the prize was
   small.**

### 5b. The machinery, priced twice, and made free once

The first implementation put the row stride in `ShardRecs::find`'s parameter list
and the rectangle test inside its loop. Measured cost of that machinery ALONE
(`plain / machoff`, the rectangle path never taken): **+1.6% wall / +1.1% CPU**
(session A), corroborated by a different-tree build of 140f914 at +1.9% wall.
That is more than the rectangle path then recovered.

The fix is in `b5ae4ac`: `find` reverts to exactly its pre-rectangle codegen (no
stride parameter, no rectangle test, `self.mutable` swapped for the low byte of
`flags`), its hit becomes a PREFILTER, and the caller passes it through `refine`
— one load and one branch when the shard holds no rectangle record, inside a
branch that is itself essentially never taken, and otherwise a cold `find_exact`
that redoes the scan exactly.

After the fix (session A2): `plain / machoff` = **1.0086 wall [3/12] / 1.0079 CPU
[1/12]**. The wall cost is gone into the noise; **a ~+0.8% CPU cost remains and
its sign is consistent (1/13 and 1/12 rounds below 1.000 across both sessions).**
That residual is honest and unexplained — the remaining hot-path delta is a `u8`
load becoming a `u16` load and one extra `usize` in `BorrowTracker` — and it is
charged against EVERY cell and every thread count, including t=1 where `fill`
takes the hull path and no rectangle is ever registered. **A default that costs
+0.8% CPU everywhere to buy −1.3% wall on one cell at t=8 is not a trade this
round can justify**, which is the second reason the feature stays a feature.

This is `docs/AGENT_BRIEF.md` §6's last clause — "price the MACHINERY the count
reduction needs" — biting for the second time in this campaign, and the isolation
arm that catches it (`machoff`, the base tracker source built in the same tree
with the same paths) is cheap enough that it should be standard.

## 6. Gates

PLACEHOLDER_GATES

## 7. What this establishes, and what the next lever must be

### 7a. The verdict, as fractions

| claim | status |
|---|---|
| an EXACT strided-rectangle record is representable with no extra storage | **YES** — hull + declared stride is a bijection; `Shard` stays 128 bytes |
| it is sound under tile threading, unlike the hull | **YES** — 766/766 by NAME at t=1 and t=8, and the exact test provably excludes the gaps (12 tests against a brute-force byte-set oracle, 7 of 9 planted mutations caught) |
| it removes the registrations it was aimed at | **YES** — 569,690 -> 409,349 per frame (−28.1%), `fill` 180,434 -> 20,093 |
| it stays off the wide path | **YES** — `w_shards = w_blocks = w_full` unchanged from base on every cell |
| **it makes `c256x2048` t=8 faster** | **NO, at the noise floor**: −1.2 to −1.4% wall vs `main`, CPU 0.000, against a ±2-3% identity band |
| **it should be the default** | **NO** — and the machinery's residual +0.8% CPU is charged to every cell including t=1, where no rectangle is ever registered |
| the five CDEF/`cdef_apply` sites were also routed through it | **NO — not attempted.** See below |

### 7b. Why the CDEF sites were not attempted, stated as an expectation and not as a result

The counterfactual (§3) makes them look better than `fill` on paper: 7.27-8.00
rows on **1.000** shards, so a 7-8x count cut with a strictly-cheaper record
(one shard -> one `try_lock` on add, one lock-free store on release). Their
combined population on this cell is 118,624 registrations/frame, 20.8%.

But `dblon` priced a `fill` registration at 2.42-2.71 ns, and there is no reason
to expect a CDEF-site registration to be dearer — they are the same shape, dense
runs on few shard lines. 118,624 x 2.6 ns = **0.31 CPU ms/frame, 2.7% of the
tracker, ~1.2% of wall at best.** That is inside the band this round could not
resolve a −1.3% in. **Wiring them is a day's work for a number this grid cannot
measure**, and `rect1` — the strictly-cheaper-record variant — came out WORSE
than base twice, which is the closest available evidence about what a
one-shard-rectangle site actually does.

If someone does wire them, the arm to build first is `dblon` at those sites (a
`RAV1D_CDEF_DOUBLE`), because it prices the prize in ten minutes without
implementing anything.

### 7c. The corrected cost model

`docs/AGENT_BRIEF.md` §6 and `docs/C256_CONTENTION.md` §7 both carry
"~19.71 ns per registration, ≥89% of it the uncontended add/remove pair". That
number is a QUOTIENT — tracker CPU divided by registration count — and this round
shows it is not a marginal price anywhere near the hot sites:

| quantity | value | how measured |
|---|---|---|
| cell average | **19.71 ns/registration** | (plain − untracked) CPU / regs, `docs/C256_CONTENTION.md` §7 |
| **marginal, at `LfBlock::fill`** | **2.42-2.71 ns/registration** | `RAV1D_LF_DOUBLE`, +180,434 regs/frame in ONE binary, two sessions |
| ratio | **7.3-8.1x cheaper than average** | |
| `fill`'s share of the population | **31.7%** | `probe-sites` |
| `fill`'s share of the tracker's CPU | **3.9-4.4%** | 180,434 x 2.56 ns / 11.24 CPU ms/frame |

So the tracker's cost is concentrated in registrations that are NOT dense repeats
on a shard the core already owns. `fill` files 8.98 records per 2.09 distinct
shard lines; the 6.89 repeats are nearly free, and they were 24% of the whole
frame's population. **Any future count-cutting lever must be priced by the
DISTINCT SHARD LINES it removes, not by the records** — and `--features
__probe_bounds`' `row_shards_mean` column is exactly that number, per site,
without writing any code.

### 7d. Where this leaves `c256x2048` at t=8

`docs/C256_CONTENTION.md` §9 listed four refused levers. This is the fifth, and
it is the one that was named as the remaining direction in the task brief:

| lever | verdict | number |
|---|---|---|
| registration COUNT cut at other sites (#502) | null | 1.0030 wall for −5.8% of the population |
| COARSER blocks (#501, #503) | null | 0.987 / 0.987 / 0.995 |
| FINER blocks (#504) | adverse, monotone | +2.2% / +11.8% / +21% at −2/−3/−5 shifts |
| waiting policy (#504) | null, bounded at 10.7% if perfect | 0.9857-1.0091, all inside ±1.8% |
| **registration DENSITY at the top site (this round)** | **at the noise floor, and priced** | −1.2 to −1.4% wall / 0.0% CPU for −28.1% of the population; the site is 3.9-4.4% of the tracker |

**The remaining mechanism is unchanged and this round sharpens why**: reduce the
number of DISTINCT shard lines a worker touches, or stop registering. The two
spellings already on the board are `get_mut`-style untracked reads (#492's
21-25%, which removed the record entirely) and worker-keyed records. Neither is
a count cut. And the prize is still bounded by a **1.33x** tracker-removed
ceiling, so this cell asks for the tracker to be nearly free at t=8, not cheaper.

### 7e. What is kept

The rectangle record itself is kept on `main`, behind `__lf_rect`, because it is
the only exact 2-D reservation the tracker has ever had and it cost this round to
establish that it is exact, sound and live. It is now the instrument for pricing
any future 2-D scheme without re-deriving soundness. `MAX_RECT_ROWS`,
`rect_decode`, `rect_hit_range`, `find_from_rect`, `add_rect` and
`DisjointImmutRectGuard` all carry their own doc comments; the tests are in
`crates/rav1d-disjoint-mut/src/tracker_shard.rs`'s `mod tests`.
