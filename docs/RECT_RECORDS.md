# The exact strided-rectangle record: built, sound, live, a small consistent win
# at t=8 — NULL on the cell it was built for, and a code-size REGRESSION at t=1

**Status: NOT the default, and the reasons are measured.** The mechanism this
round was opened to test — collapsing `LfBlock::fill`'s `h` per-row
registrations into ONE exact strided-rectangle record, the third shape after the
per-row split and the refuted hull — works, is exact, is sound under tile
threading, is live, and keeps the corpus at 766/766 by name at t=1 and t=8.

What it measures, on an idle box against TWO controls (a byte-identical copy and
a same-source-different-build layout control), is three things at once:

| where | wall | CPU | sign | replicated? |
|---|---|---|---|---|
| **multi-tile t=8, 5 of 6 cells** | **−1.0% to −1.8%** | −1.3% to **−3.3%** | 6/7 to **11/11** | **yes, two sessions** |
| **`c256x2048` t=8 — the cell it was built for** | −0.2% to −0.5% | −0.6% | **4/7, 5/10, 8/15 — a coin flip** | yes, three grids |
| **`v4k8tile` t=1, where the path never fires** | **+0.9% to +1.3%** | same | **0/11 twice** | **yes, two sessions** |

Both controls sit at 1.000 with coin-flip signs on every one of those cells, so
the t=8 wins and the t=1 regression are both above the floor. **The t=1 cost is
code size, and it is NOT explained** — at t=1 `fill` takes the hull path and no
rectangle is ever registered, so the arm cannot differ in work done. Moving the
attempt out of line (`#[inline(never)]`, the obvious fix for a function that is
`#[inline(always)]` and monomorphised 12 ways) moved it 1.0103 -> 1.0088 and left
the sign at 0/11 (§5e). Paying ~1% at t=1 on 4K content, where this decoder's
largest gap already lives, to buy −1.5% at t=8 is not a trade this round will
make on the strength of an effect it cannot account for.

The prize was small for a reason that is now measured. **A `LfBlock::fill`
per-row registration costs 2.42-2.71 ns at the margin** — measured by ADDING a
duplicate per row in the same binary (`RAV1D_LF_DOUBLE`; +180,434
registrations/frame, +3.4% to +4.0% wall, **0 of 25 rounds on the other side**),
which is the only sound way to price a registration on a contended path. This
cell's AVERAGE is **19.71 ns/registration**. So the largest registration site in
the decoder — **31.7% of the population** — is **3.9-4.4% of the tracker's CPU**,
and removing 89% of it could never have been worth more than ~1.5% of wall.

**That corrects the campaign's cost model in the word that decides the next
lever.** `docs/AGENT_BRIEF.md` §6 records the `c256x2048` residual as "about one
cross-core transfer of the shard's own cache line **per registration**". Per
registration it is not: `fill` files 8.98 records per **2.09** distinct shard
lines, and a repeat touch of a line this core already owns is ~7.7x cheaper than
the average registration. **The expensive registration is the one whose shard
line another core has taken since, so the cost tracks the DISTINCT SHARD LINES a
worker visits, not the records it files** — the same conclusion
`docs/C256_CONTENTION.md` §4 reached from the granularity side, now measured from
the count side. **A count cut at the DENSEST site is the least valuable count cut
available**, which is exactly why the densest site produced the smallest win.

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
  ship with its feature deleted; it wins at t=8 and LOSES at t=1, so the feature
  stays as the arm it is (§5d).
* **Only `LfBlock::fill` was routed through it.** The counterfactual names five
  more sites on this cell whose rectangles are cheaper still on paper
  (`cdef_apply.rs:104:32` and `:121:33`, `safe_simd/cdef_arm.rs:{192,622,1217}`
  — 7.27-8.00 rows on **1.000** shards, so a 7-8x count cut each). They were
  NOT wired. See §7b for the price of doing so, and for the ten-minute arm that
  settles it without implementing anything.
* **The 5-of-6-cells t=8 win is n=6..7 rounds per cell**, not the n=12..15 the
  primary cell got. It is above both controls by sign count on every one of them,
  and it has NOT been replicated in a second session.
* **The t=1 regression is attributed to code size by ELIMINATION, not by
  measurement of code size.** At t=1 the rectangle path is unreachable, so the
  arm cannot differ in work done; what it differs in is `fill`'s size. No
  `cargo asm` or `llvm-lines` was taken and no attempt was made to shrink it
  (moving the rectangle attempt behind `#[inline(never)]`, or out of the
  `W`-monomorphised function, are the obvious things to try and were not tried).
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
* **Miri's `shard_liveness` target times out locally** in all four
  configurations and is reported as a timeout, never as green (§6c). The other 7
  targets are clean under BOTH models.
* **The machinery's own cost is NOT RESOLVABLE on this box.** See §5c: two
  builds of the SAME base source differ by 1.5% wall, so the "+1.6% machinery"
  reading the first implementation produced is inside build-to-build layout
  noise. The machinery was rewritten to be free anyway (`b5ae4ac`), which is
  the right change regardless of whether the measurement supported it.

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

### 5a. Grid C — the decisive one: primary cell, idle box, n=15, TWO controls

`C256x2048_420_8b__t8` at t=8, `foreign_max = 0`, no round discarded for load.
`plainC` is the SAME SOURCE as `plain` built in a second worktree, so it differs
only in code layout; `base` and `machoff` are two builds of 140f914's tracker
source, which makes them a second layout pair.

| arm | wall ms/f | wall/`plain` | [min..max] | sign | CPU/`plain` | sign |
|---|---|---|---|---|---|---|
| `plain` | 3.749 | 1.0000 | — | — | 1.0000 | — |
| **`plainB` byte-identical** | 3.754 | **1.0039** | [0.9727..1.0133] | **4/15** | 1.0031 | 5/15 |
| **`plainC` layout only** | 3.768 | **1.0040** | [0.9752..1.0265] | **5/15** | 0.9953 | 10/15 |
| `machoff` (base src, this tree) | 3.759 | 1.0027 | [0.9702..1.0146] | 6/15 | 0.9902 | 14/15 |
| `base` (base src, other tree) | 3.709 | 0.9881 | [0.9662..1.0133] | 13/15 | 0.9951 | 12/15 |
| **`rect`** | 3.714 | **0.9908** | [0.9690..1.0278] | **8/15** | **1.0002** | 6/15 |

**Two readings, and the second is the one that matters.**

1. **Two builds of the same base source read 1.0027 and 0.9881 — a 1.5% wall
   spread from code layout alone**, and `plainC` (same source as `plain`) reads
   1.0040 against `plainB`'s 1.0039. So on this cell **build-to-build layout
   noise is ~1.5% on wall and ~0.5% on CPU**, and any claim below that from a
   pair of different binaries is unsupportable. That retires the "+1.6%
   machinery" reading of §5c.
2. **On the primary cell `rect` is 0.9908 with an 8/15 SIGN — a coin flip.** The
   cell the round was aimed at is the one where the mechanism does least.

### 5b. Grid B3 — breadth, idle box, and this is where it wins

Ten cells, `plain` / `plainB` / `plainC` / `rect` / dav1d, 9 rounds, loaded
rounds dropped per cell (`foreign_max` 1-5 on a few, none above). Both controls
are in every cell, so each row carries its own floor.

| cell | `rect` wall | sign | `rect` CPU | sign | `plainB` wall (sign) | `plainC` wall (sign) |
|---|---|---|---|---|---|---|
| `c1024x192` t=8 | **0.9900** | **7/7** | 0.9911 | 6/7 | 0.9950 (4/7) | 1.0050 (1/7) |
| `c1024x384` t=8 | **0.9880** | 5/7 | **0.9872** | **7/7** | 0.9970 (4/7) | 1.0030 (2/7) |
| `c1024x576` t=8 | **0.9825** | **7/7** | **0.9849** | **7/7** | 0.9981 (4/7) | 0.9981 (5/7) |
| `c3840x256` t=8 | **0.9867** | 6/7 | 1.0000 | 2/7 | 1.0019 (3/7) | 0.9981 (4/7) |
| **`text_q20` t=8** | **0.9867** | **6/6** | **0.9705** | **6/6** | 1.0000 (3/6) | 1.0000 (3/6) |
| `ui_q20` t=8 | 0.9930 | 5/7 | 0.9948 | 5/7 | 0.9977 (4/7) | 1.0023 (2/7) |
| **`c256x2048` t=8 (primary)** | **0.9955** | **4/7** | 0.9945 | 6/7 | 1.0075 (3/7) | 1.0195 (1/7) |
| `v4k8tile` t=8 | 0.9990 | 3/4 | 0.9976 | 3/4 | 1.0018 (1/4) | 0.9956 (2/4) |

**Five of six multi-tile t=8 cells are −1.0% to −1.8% wall with 5/7-7/7 signs,
while both controls sit within ±0.5% of 1.000 with coin-flip signs.** Screen
text is the best cell in the grid: **−1.3% wall and −3.0% CPU, 6/6 on both.**
`v4k8tile` is flat (n=4 after load-dropping — too few to call), and the primary
cell is the weakest.

**This is the shape `docs/C256_CONTENTION.md` §5c used to REJECT `lockrelax`,
inverted**: that arm was −1.4% on one cell and flat on five others, so it was
called noise. This one is −1.0..−1.8% on five cells and flat on the one it was
built for, which by the same standard is a real effect.

### 5c. Grid A / A2 — the doubling arm, which prices a registration

Sessions A (n=13) and A2 (n=12) on the primary cell also carried
`dbloff`/`dblon`: ONE binary (`--features __probe_lf_hull`), `RAV1D_LF_DOUBLE`
off and on, which takes each per-row guard TWICE. Both guards are immutable over
the same bytes, so it cannot invent an overlap; it changes the count and nothing
else.

| quantity | session A | session A2 |
|---|---|---|
| `dblon` / `dbloff` wall | **1.0337** | **1.0337** |
| `dblon` / `dbloff` CPU | 1.0202 | 1.0182 |
| rounds with `dblon` faster | **0/13** | **0/12** |
| CPU delta for +180,434 regs/frame | +0.488 ms/f | +0.436 ms/f |
| **ns per `fill` registration, marginal** | **2.70** | **2.42** |

`rect1` (rectangles accepted only when they land in ONE shard — 20.4% of `fill`s
here) reads **1.0077** and **1.0164** against `plain`, signs 3/13 and 0/12: WORSE
than base in both sessions despite being the strictly-cheaper record. That
refutes the competing explanation that the multi-shard `add_rect` +
`remove_multi` lock traffic is what eats the win. **Coverage dominates, not the
per-record cost.**

The first implementation's machinery reading (`plain / machoff` = +1.6% wall /
+1.1% CPU) is superseded by §5a: it is inside the 1.5% layout spread. The
machinery was made free anyway in `b5ae4ac` — `find` reverted to exactly its
pre-rectangle codegen, its hit downgraded to a prefilter, the exact test moved
into a cold `refine`/`find_exact` — because that is the right shape whether or
not the measurement could see it.

### 5d. Grids D and E — t=1, where the path never fires, and where it LOSES

At t=1 `tile_threading_active()` is false, so `fill` returns through the hull path
and `add_rect` is never called. The `rect` arm is therefore semantically identical
to `plain` at t=1, and any difference is code size or layout. Idle box.

| cell t=1 | grid | `rect` wall | sign | `plainB` (sign) | `plainC` (sign) |
|---|---|---|---|---|---|
| **`v4k8tile`** | D, n=11 | **1.0126** | **0/11** | 1.0007 (4/11) | 1.0012 (3/11) |
| **`v4k8tile`** | E, n=11 | **1.0103** | **0/11** | 0.9990 (7/11) | 0.9996 (8/11) |
| `c1024x576` | D, n=12 | 1.0032 | 2/12 | 1.0004 (6/12) | 1.0014 (6/12) |
| `c1024x576` | E, n=9 | 1.0044 | 1/9 | 1.0000 (3/9) | 1.0004 (4/9) |
| `c256x2048` | D, n=11 | 1.0027 | 3/11 | 0.9987 (6/11) | 1.0013 (3/11) |
| `text_q20` | D, n=12 | 0.9925 | 9/12 | 0.9989 (6/12) | 1.0032 (5/12) |

**`v4k8tile` at t=1 is +1.0% to +1.3% with 0 of 11 rounds below 1.000 in TWO
independent sessions**, while both controls sit within ±0.1% of 1.000 with
coin-flip signs. That is the clearest signal in the round after `dblon`, and it is
a REGRESSION, on the cell where this decoder's single-thread gap is largest
(`docs/TILED_SCALING.md` §6: the filter chain alone is 4.0x dav1d at 4K t=1).

### 5e. The obvious fix for it was tried and DID NOT WORK

`fill` is `#[inline(always)]` and monomorphised over six `W` values x two bit
depths, so an inlined rectangle attempt is twelve copies of it inside the hottest
function in the filter chain — the natural explanation for a cost paid where the
code cannot run. Moving it into `#[inline(never)] fill_rect` (`c9ee1ec`) is the
whole fix, and it is not enough:

| cell | `rect` (inlined) | `rect2` (out of line) | controls |
|---|---|---|---|
| `v4k8tile` t=1 | 1.0103, **0/11** | **1.0088, 0/11** | 0.9990 / 0.9996 |
| `c1024x576` t=1 | 1.0044, 1/9 | 1.0040, **0/9** | 1.0000 / 1.0004 |
| `c1024x576` t=8 | 0.9922, 10/11 | 0.9922, **11/11** | 0.9981 / 1.0039 |
| `c1024x192` t=8 | 0.9900, 10/10 | **0.9876, 10/10** | 0.9951 / 1.0000 |
| `text_q20` t=8 CPU | 0.9696, 10/10 | **0.9670, 10/10** | 1.0017 / 1.0026 |
| `c256x2048` t=8 | 0.9977, 6/10 | 0.9978, 5/10 | 0.9985 / 1.0232 |

**So the t=1 cost is NOT the inlined code size, and it is not yet explained.** The
out-of-line form is kept because it is equal-or-better everywhere, but the honest
statement is that a ~1% t=1 penalty from code that provably cannot execute is
unresolved, and the next step is `cargo asm` / `cargo llvm-lines` on `fill` — not
more timing.

### 5f. And the t=8 win REPLICATES

Grid E is a second, independent session for four of grid B3's cells and it
reproduces the effect with stronger signs:

| cell t=8 | B3 wall (sign) | E wall (sign) | E CPU (sign) |
|---|---|---|---|
| `c1024x192` | 0.9900 (7/7) | **0.9876 (10/10)** | 0.9944 (9/10) |
| `c1024x576` | 0.9825 (7/7) | **0.9922 (11/11)** | **0.9951 (11/11)** |
| `text_q20` | 0.9867 (6/6) | 0.9885 (8/10) | **0.9670 (10/10)** |
| **`c256x2048` (primary)** | 0.9955 (4/7) | **0.9978 (5/10)** | 0.9938 (6/10) |

Two sessions, two controls per cell, and the same answer in both: **a real
−1% to −1.8% at t=8 everywhere except the cell the round was opened for.**

### 5g. Decision

**The rectangle path stays behind `__lf_rect`.** The t=8 gain is real and
replicated; the t=1 loss is real and replicated; the loss lands on 4K
single-threaded, which is this decoder's worst cell; and the mechanism's own
target cell gains nothing. Making it the default would trade a measured
regression on the worse workload for a measured win on the better one, on the
strength of an unexplained code-size effect. That is not a trade to make in the
same PR that discovered it.

## 6. Gates

Driver `scripts/perf/rect_gates.sh`, logs `~/tmp/rectrec/gates`. Nothing here is
timed, so everything is `nice`d and nothing takes the measurement lock.

### 6a. Correctness

| gate | result |
|---|---|
| **corpus, `__lf_rect` arm, t=1**, no `--skip-group` | **766 PASS + 2 SKIP**, mismatch=0 error=0 |
| **corpus, `__lf_rect` arm, t=8**, no `--skip-group` | **766 PASS + 2 SKIP**, mismatch=0 error=0 |
| **corpus, DEFAULT arm, t=1** | **766 PASS + 2 SKIP** |
| **corpus, DEFAULT arm, t=8** | **766 PASS + 2 SKIP** |
| set-diff BY NAME (key `(group, name)`, value `(status, ACTUAL md5)`) vs `benchmarks/aarch64_md5_fixes_2026-08-07_final.tsv.zst` | **CLEAN on all four**: 0 only-in-baseline, 0 only-in-head, 0 differing |
| set-diff t=1 vs t=8 within each arm | CLEAN, both arms |
| `cargo test --lib`, release AND debug | pass, both |
| tracker unit tests, ONE feature configuration at a time: default, `--no-default-features`, `__rect_1shard`, `__probe_wide`, `__probe_wide,__rect_1shard`, `__probe_sites`, `zerocopy`, `__probe_lock_park`, `__bps_blocks`, `__msb_5` | **42 pass each** (40 under `__rect_1shard`, which `#[cfg]`s out the two tests whose subject it removes) |
| `decode_md5_verify`, `thread_cleanup_test`, `tile_threading_overlap`, `reproduce_overlap`, `mt_stress`, plain AND `-- --ignored` | pass, all |
| every timed arm's `CHECKSUM` before any timing | 9 arms x 2 thread counts -> **ONE md5 per cell**, on all seven cells |
| `cargo fmt --all --check` | rc=0 |
| clippy `-D warnings`: tracker `--all-targets`, `--no-default-features --all-targets`, `--all-targets --features __rect_1shard`, `--all-targets --features __shards_1`, root `--lib`, `--lib --features __lf_rect`, `--lib --features __lf_rect1` | rc=0, all seven |

The DEFAULT arm is not a formality: `ShardRecs::find` — the hottest loop in the
decoder — was edited, so "the rectangle path is off" is a claim about a file that
changed, and 766/766 BY NAME is the evidence rather than the assertion.

### 6b. Test teeth, proven by planting

Every mutation was restored from a `~/tmp` backup COPY, never `git checkout --`,
and verified byte-exact by sha256 AND `git diff --exit-code`.
`tracker_shard.rs` sha256 `7137b697…` before and after the battery.

| planted mutation | result |
|---|---|
| (control, no mutation) | 42 pass |
| `rect_hit_range` always returns `Some` — the record degrades to a HULL | **FAILS (6)** |
| `rect_hit_range` always returns `None` — **detection OFF** | **FAILS (7)** |
| `rect_decode`'s row count off by one | **FAILS (8)** |
| `find` ignores the rect bit (rectangle treated as its hull interval) | **FAILS (2)** |
| rectangle-vs-rectangle compares only row 0 | **FAILS (1)** |
| `add_rect` never scans before registering | **FAILS (3)** |
| `add_rect` drops the O(1) block-span cap (in-loop belt remains) | **passes** — reported as a NON-mutation: the two cap checks are deliberate duplicates of each other |
| `add_rect` drops the in-loop cap belt (O(1) pre-check remains) | **passes** — same reason |
| **both cap checks removed together** | **FAILS (1)** |

The fifth row is why the round's own first test grid was rewritten: with
non-negative offsets only, the registrant's row 0 is always the nearest to the
counterparty, so a mutation that compares only row 0 CANNOT be caught. The grid
now runs signed offsets and catches it. That mutation passed the first time it
was planted, and the test was fixed rather than the finding softened.

**Decoder-level teeth**, because the tracker tests alone do not prove the corpus
would notice a wrong pixel from this path: planting `rect.row(row + 1)` in
`LfBlock::fill`'s rectangle loop gives **277 mismatches of 358** on
`8-bit/data` at t=8; restoring gives **358/358 clean**.
`src/loopfilter.rs` sha256 `4338932e…` before and after.

**`forbid(unsafe_code)` proven ACTIVE, not read**: an
`unsafe { core::mem::transmute(x) }` planted in `src/picture.rs` (no
module-level forbid of its own, compiled in every configuration) fails the build
against **`lib.rs:13:12`** — the campaign brief's anchor. Restored, sha256
`fa02c12b…` before and after, `git diff` clean, lib rebuilt green. No `unsafe`
was added to `rav1d-safe`. The `unsafe` this round adds is all in
`crates/rav1d-disjoint-mut` (which has no `forbid`): one `NonNull::new_unchecked`
+ `add` in `index_rect_inner` and one `slice::from_raw_parts` in
`DisjointImmutRectGuard::row`, each with a safety comment naming the invariant —
the live registration, the alignment check, and the in-bounds proof from the
constructor.

### 6c. Miri, both aliasing models

`cargo +nightly miri test -p rav1d-disjoint-mut --no-fail-fast --test <target>`,
ONE TARGET AT A TIME (Miri aborts the process on first UB and cargo stops at the
first failing TARGET, so a batch run lets later targets never execute and their
silence reads as health). Both Stacked Borrows and Tree Borrows, and both the
default feature set and `__rect_1shard` — the arm that changes WHICH of
`add_rect`'s two registration shapes runs, i.e. a different pointer/reference
sequence. Driver `scripts/perf/rect_miri.sh`, record
`benchmarks/rect_records_miri_2026-08-11.tsv`.

**Miri is the gate this mechanism most needed.** The March-2026 strided tracker
had an exact record and a reference over the whole hull; that combination is UB
under both models, 766 corpus vectors did not see it, and Miri did.
`DisjointImmutRectGuard` has no `Deref` and derives each row from the buffer
pointer specifically so that no reference ever spans a gap — these legs are what
says so instead of the doc asserting it.

**Both models, both feature sets, CLEAN on every non-timeout target:**

| target | SB default | SB `__rect_1shard` | TB default | TB `__rect_1shard` |
|---|---|---|---|---|
| `--lib` | **42 passed** | **40 passed** | **42 passed** | **40 passed** |
| `narrow_release` | 1 | 1 | 1 | 1 |
| `soundness` | 25 | 25 | 25 | 25 |
| `wide_exclusion` | 1 | 1 | 1 | 1 |
| `guard_move_release` | 2 | 2 | 2 | 2 |
| `pic_buf_overflow` | **0 tests ran** | **0 tests ran** | **0 tests ran** | **0 tests ran** |
| `aligned_miri` | **0 tests ran** | **0 tests ran** | **0 tests ran** | **0 tests ran** |
| `shard_liveness` | **TIMEOUT** | **TIMEOUT** | **TIMEOUT** | **TIMEOUT** |

**CLEAN in all four columns on every one of the 7 non-timeout targets**, and
`--lib` includes all 13 new rectangle tests (40 rather than 42 under
`__rect_1shard`, which `#[cfg]`s out the two whose subject that arm removes).
This is BETTER coverage than the previous round achieved: `docs/C256_CONTENTION.md`
§8c had two extra Tree-Borrows timeouts in its `park` corner, and this round has
none.

`shard_liveness` times out (rc=124) exactly as `docs/AGENT_BRIEF.md` warns and as
`docs/C256_CONTENTION.md` §8c recorded; it is reported AS a timeout, never as
green, and CI's Linux Miri legs (whole package, `--all-features`) are what cover
it. `pic_buf_overflow` and `aligned_miri` select **0 tests** under these feature
sets and are reported as 0, never as green.

The exhaustive differential grids are scaled down under `cfg!(miri)` (point count
only — every assertion, oracle and liveness floor is unchanged, and the native
run keeps the full grid). Without that, `--lib` cannot finish inside any sane
timeout, and **a timed-out target reports nothing at all, which is strictly worse
than a smaller grid that still asserts.**

### 6d. Pre-existing failures, verified on the base commit

None of these are this branch's, and each was confirmed by running the same
command in a worktree at 140f914:

| leg | base 140f914 | this branch |
|---|---|---|
| tracker tests `--features __shards_1` | **5 FAIL** (`adaptive_shift_keeps_the_block_count_near_target`, `coarser_blocks_collapse_a_strided_access_onto_fewer_shards`, `declaring_a_stride_installs_the_derived_shift`, `one_tile_does_not_get_the_coarse_shift`, `rows_rule_targets_picture_rows_not_block_count`) | the **same 5 by NAME**, set-diffed |
| tracker tests `--all-features` | does not COMPILE (`probe_shard_of` / `probe_geometry` absent under `__probe_bounds` + `__tracker_legacy`) | same |
| clippy tracker `--all-features --all-targets` | 8 errors | same |
| `cargo clippy --release --all-targets` (root, NOT a CI leg) | fails on `_dev` examples | same |

## 7. What this establishes, and what the next lever must be

### 7a. The verdict, as fractions

| claim | status |
|---|---|
| an EXACT strided-rectangle record is representable with no extra storage | **YES** — hull + declared stride is a bijection; `Shard` stays 128 bytes |
| it is sound under tile threading, unlike the hull | **YES** — 766/766 by NAME at t=1 and t=8, and the exact test provably excludes the gaps (12 tests against a brute-force byte-set oracle, 7 of 9 planted mutations caught) |
| it removes the registrations it was aimed at | **YES** — 569,690 -> 409,349 per frame (−28.1%), `fill` 180,434 -> 20,093 |
| it stays off the wide path | **YES** — `w_shards = w_blocks = w_full` unchanged from base on every cell |
| **it makes `c256x2048` t=8 faster** | **NO** — 0.9955 wall with a **4/7** sign on the breadth grid and 0.9908 with **8/15** on the n=15 grid: a coin flip on the cell the round was aimed at |
| it makes OTHER multi-tile t=8 cells faster | **YES, 5 of 6** — −1.0% to −1.8% wall, −1.3% to −3.0% CPU, signs 5/7 to 7/7, both controls at 1.000 with coin-flip signs |
| **it is free where it does not fire** | **NO** — `v4k8tile` t=1 is **+1.26% with 0/11**, and at t=1 the path is never taken, so that is code size |
| **it should be the default** | **NO.** A +1.26% t=1 regression on 4K, where this decoder's largest gap already lives, is not worth a −1.5% t=8 win — and no attempt was made to shrink `fill` first |
| the five CDEF/`cdef_apply` sites were also routed through it | **NO — not attempted.** See below |

### 7b. Why the CDEF sites were not attempted, as an expectation and not a result

The counterfactual (§3) makes them look better than `fill` on paper: 7.27-8.00
rows on **1.000** shards, so a 7-8x count cut with a strictly-cheaper record (one
shard -> one `try_lock` on add, one lock-free store on release). Their combined
population on the primary cell is 118,624 registrations/frame, 20.8%.

Two measurements argue against expecting much, and one argues for trying:

* **Against:** `dblon` priced a `fill` registration at 2.42-2.71 ns, and a
  CDEF-site registration is the same shape — a dense run on few shard lines.
  118,624 x 2.6 ns = **0.31 CPU ms/frame, 2.7% of the tracker**, and the `fill`
  cut delivered less than its own arithmetic predicted.
* **Against:** `rect1` — the one-shard, strictly-cheaper-record variant — came out
  WORSE than base in both sessions. The CDEF sites are all one-shard, so `rect1`
  is the closest available evidence about what such a site does, and it is
  negative.
* **For:** the t=8 breadth win (§5b) is real on five cells, and the CDEF sites'
  populations are concentrated on exactly the 1024-wide and screen-content cells
  where it is largest.

**The arm to build first is a `RAV1D_CDEF_DOUBLE`, not the rectangle.** It prices
those sites' whole registration population in ten minutes, in one binary, without
implementing anything — and if it reads under ~1% there is nothing to win.

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
| **registration DENSITY at the top site (this round)** | **null ON THIS CELL, and priced** | 0.9955 wall (4/7) / 0.9908 (8/15) for −28.1% of the population; the site is 3.9-4.4% of the tracker. It DOES pay −1.0..−1.8% on five other t=8 cells, and costs +1.26% at t=1 on 4K |

**The remaining mechanism is unchanged and this round sharpens why**: reduce the
number of DISTINCT shard lines a worker touches, or stop registering. The two
spellings already on the board are `get_mut`-style untracked reads (#492's
21-25%, which removed the record entirely) and worker-keyed records. Neither is
a count cut. And the prize is still bounded by a **1.33x** tracker-removed
ceiling, so this cell asks for the tracker to be nearly free at t=8, not cheaper.

### 7e. The one thing here that is worth someone else's time

**The t=8 breadth win is real and it is not this cell's.** −1.0% to −1.8% wall on
`c1024x{192,384,576}`, `c3840x256` and `text_q20`, with `text_q20` at −3.0% CPU
and 6/6 on both, is a bigger and more reproducible effect than anything the four
`c256x2048` levers produced. It is currently unshippable only because of a
+1.26% t=1 code-size cost on `v4k8tile`, and that cost has an obvious untried
fix: `LfBlock::fill` is `#[inline(always)]` and monomorphised over six `W` values
and two bit depths, so the rectangle attempt is duplicated twelve times inside
the hottest function in the filter chain. Moving it behind an `#[inline(never)]`
shim, or out of the `W`-generic body entirely, costs nothing at t=8 (the
rectangle is taken once per `fill`, not once per row) and is the difference
between this arm and a default. **That is the next chunk, and it is small.**

### 7f. What is kept

The rectangle record itself is kept on `main`, behind `__lf_rect`, because it is
the only exact 2-D reservation the tracker has ever had and it cost this round to
establish that it is exact, sound and live. It is now the instrument for pricing
any future 2-D scheme without re-deriving soundness. `MAX_RECT_ROWS`,
`rect_decode`, `rect_hit_range`, `find_from_rect`, `add_rect` and
`DisjointImmutRectGuard` all carry their own doc comments; the tests are in
`crates/rav1d-disjoint-mut/src/tracker_shard.rs`'s `mod tests`.
