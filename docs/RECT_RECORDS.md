# The exact strided-rectangle record: built, sound, live — and NULL on the cell
# it was built for, with the cost model it refutes

**Status: NEGATIVE, and the negative is the deliverable.** The mechanism this
round was opened to test — collapsing `LfBlock::fill`'s `h` per-row
registrations into ONE exact strided-rectangle record, the third shape after
the per-row split and the refuted hull — works, is exact, is sound at t=8, and
is measured **null** on `c256x2048` at t=8.

What ships is the instrument and the number: **removing 28.1% of the
registration population (and 24.3% of the shard-touches) moved wall
`PLACEHOLDER_WALL` and CPU `PLACEHOLDER_CPU` against an identity control band
of `PLACEHOLDER_BAND`.** With `probe-wide` proving the path live
(`n_rect > 0`, `n_rect_declined = 0`, `w_shards = w_blocks = w_full = 0`) and
`probe-sites` proving the population dropped, this is not a null from code that
never ran.

**Therefore the campaign's cost model is wrong in a way that matters.**
`docs/AGENT_BRIEF.md` §6 records the `c256x2048` residual as "about one
cross-core transfer of the shard's own cache line **per registration**". It is
not per registration. It is per shard-line acquisition *that another core has
touched since*, and the `fill` site's `h` registrations are 8.98 back-to-back
acquisitions of the same ~2.09 shard lines from one core — so ~78% of them were
already nearly free and collapsing them to one buys nothing.

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

## 5. Timed: null

PLACEHOLDER_TIMED

## 6. Gates

PLACEHOLDER_GATES

## 7. What this refutes, and what the next lever must be

PLACEHOLDER_VERDICT
