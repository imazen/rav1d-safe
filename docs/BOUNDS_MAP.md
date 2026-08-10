# The bounds map: what a guard reserves, what it touches, and who is next to it

`--features __probe_bounds` — throwaway, `__`-gated, absent from `default` and
from every published feature. Source `crates/rav1d-disjoint-mut/src/bounds_probe.rs`,
drivers `examples/probe_tracker.rs` and `examples/probe_bounds_corpus.rs`,
tables `scripts/perf/bounds_tables.py`, record
`benchmarks/bounds_map_2026-08-10.{meta,tsv.zst}` + `_raw_*.tar.zst`.

## Why it exists

"Widen a guard's reservation to cut the registration count" has been tried
three times and refuted three times, each time only by building it:

* **#469 strided rectangle** — exact record, wide reference. Miri UB under both
  memory models; a real decode failure in CI.
* **#475 hull arm / tile-keyed** — over-reserved the gaps BETWEEN ROWS. Slow
  (2.65x at t=8), not wrong.
* **#485 loop-filter read band** — over-reserved the gaps BETWEEN EDGES, because
  the filter's read set is 2-D sparse. A 128-px row copy-in collided with a
  concurrent 8-px write inside it. Decode failure.

All three are one question — *does the proposed extent intersect anything
another worker is concurrently writing?* — and none of them could be answered
without shipping the change. This is the instrument that answers it first.

## What it records

Per guard acquisition, hooked in `DisjointMut::index`/`index_mut`:

* **site** — `Location::caller()`, the same key `site_probe` uses.
* **reserved extent** — the `[start, end)` the tracker registered, after the
  same clamp.
* **footprint** — what was touched *through that guard*, recorded at the point
  of use, in one of three states which the report labels per site:
  * `rows` — the site declared `(lo, w, rows, stride)`. **Exact.** Declared by
    the strided helpers that already compute that geometry for their own bounds
    arithmetic (`narrow_guard`, `narrow_guard_mut`, `for_rows`, `for_rows_mut`,
    `compact_read_fast`, `compact_write_back_fast`, `LfBlock::fill_hull`), so
    the call costs them nothing.
  * `whole` — `Deref`/`DerefMut` happened and nothing was declared, so the
    footprint is taken as the whole reservation. An **upper bound**: at these
    sites over-reservation reads as zero.
  * `none` — never dereferenced; the guard bought exclusion only.
* **liveness** — a global monotone epoch at acquire and release, plus the worker
  slot.

On each acquire it publishes its record and then scans every other worker's live
set (`SeqCst` fence between the two, so no co-live pair can be missed by both
sides). Every co-live pair is therefore seen exactly once, at the later acquire.

## What it derives

1. **Over-reservation per site** — reserved/footprint, plus the SHAPE of the
   waste split into leading, trailing and inter-row-gap bytes. The row set is
   the tightest extent any coarsening must respect.
2. **The widening budget** — a histogram of the distance from each acquisition
   to the nearest concurrently-live foreign reservation, and separately to the
   nearest foreign **WRITE**. Widening a site by `k` bytes collides in exactly
   the buckets below `k`, so any proposed coarsening is priced by reading a row.
3. **Concurrent-conflict sets** — per ordered site pair: co-live count, whether
   the RESERVATIONS ever intersected, whether the FOOTPRINTS ever did, whether
   the counterparty was a writer, and the closest approach.

## Two self-checks, both with teeth

* **Reconciliation.** `__probe_bounds` implies `probe-sites`, so both run in ONE
  binary. Totals agree to the registration in all eight measured cells —
  11,401,399 at `v4k_8tile` t=8, 936,867,496 on the 358-vector corpus cell,
  `lost=0` on both sides. TEETH: dropping the probe's immutable acquisitions
  moved it to 3,604,424 against an unchanged 11,401,399.
* **`mutable_overlaps` is ground truth.** A concurrent reservation overlap
  involving a mutable record is impossible — `DisjointMut` panics on it — so the
  counter must read 0. It does, in every cell, including over the 407,046
  immutable-vs-immutable overlaps the corpus cell legitimately has. TEETH:
  widening every recorded extent by +-4096 bytes moved it from 0 to 874.
  It also caught a real defect: at `NINST = 1024` the instance table saturated,
  distinct buffers collapsed onto one id, and the scan reported 70 impossible
  overlaps.

## What it says (2026-08-10 run; full record in the `.meta`)

* **At t=8 the shipped decoder has essentially no over-reservation left.** Every
  hot site is `over_ratio = 1.000`, `fp_kind = whole`, reservations 1-16 bytes,
  `never_deref = 0`. The `tile_threading_active()` gate already picks the tight
  path wherever a hull path exists.
* **At t=1 the hull paths are live and the waste is 153x-1680x**, and it is
  **entirely inter-row gaps** (lead and tail waste are 0.00 at every one of
  them). `loopfilter.rs:769:33` reserves 35,756 bytes to touch 90.7.
* **The 4K gap vectors under-report collision risk by ~1000x.** For
  `loopfilter.rs:710:14`, acquisitions with a concurrent WRITE within 4 KiB run
  at 0.09 per million registrations on `v4k_8tile` and 91 per million on the
  corpus. That is why #485's band passed one full 4K-shaped sample and then
  failed on the corpus.
* **#485 is retrodicted.** Its ~124-byte widening lands in the `<=256 B` column:
  **16** predicted collisions across 1406 frames of `8-bit/data`; #485 measured
  1, 2 and 0 errors over three full passes of that group.
* **The counterparty has a name.** `loopfilter.rs:710:14` comes within **232
  bytes** of `cdef_arm.rs:622:9`'s concurrent write, over 2,217,283 co-live
  pairs. Against other readers it has unlimited room.
* **Some sites have zero headroom.** `ctx.rs:99:27` had a concurrent write at
  gap **0** (butt-adjacent) 36 times; widening it by one byte collides.
* **Some sites are free of writers entirely.** `mc.rs:121:61` (181 M
  registrations) and `mc.rs:1342:44` (67 M) never had a concurrent foreign write
  at any distance — they read reference frames, immutable for the whole decode.
  Extent is not their constraint; the tracker's wide path still is.

## What it is NOT

* **An empirical map, not a proof.** A site that never appears is UNKNOWN. A gap
  is the closest approach *observed*.
* **Not a timing instrument.** Every registration publishes, fences and scans;
  `v4k_8tile` t=8 runs ~3.5x slower. No wall-clock number from this build is
  valid, and the perturbation also changes how much real overlap occurs.
* **Concurrency is under-reported** by the seqlock race rate: 4.0% of foreign
  slot reads on `v4k_8tile` t=8, 5.4% on the corpus (`lost_scan`).
* **The `whole` footprint is a real hole at the large-reservation sites.**
  `safe_simd/mc_arm.rs:5971:41` takes 3,536,733 guards of mean **2,466,546
  bytes** on the corpus cell and the report calls its `over_ratio` 1.000 —
  which is the instrument declining to answer, not an answer. Same for
  `mc_arm.rs:6182:41` (541,865 B), `picture.rs:589:26` (4,096 B) and
  `looprestoration.rs:{382,408}` (~2,050 B). Those sites need a
  `probe_declare_rows` call like `narrow_guard` and `fill_hull` have; until
  then their over-reservation column is ABSENT, not zero. (Their concurrency
  column is exact, and says they never meet a concurrent writer.)
* **Sub-`Deref` write sets are not measured.** For the conflict question that is
  the safe direction — a mutable reservation is a superset of the bytes it
  writes, so testing a proposed extent against foreign *reservations* can
  over-predict a conflict but never miss one.
* **Coverage gaps:** x86_64, wasm32, `asm`/`c-ffi`, `unchecked`, t=2/4/16,
  12-bit, film grain (dropped — #479 aborts it above one thread), and the
  `8-bit/{issues,size,intra,mv,mfmv,resize,vq_suite,cdfupdate,quantizer}`
  groups. Loop restoration IS covered: live in 320 of 358 `8-bit/data` vectors
  and 71 of 71 `10-bit/data`, and in **neither** 4K vector.

---

# Part 2 — the verdict table, the standing assertions, and what is left

Derived from the same 2026-08-10 data by `scripts/perf/bounds_verdicts.py`
(deterministic — re-run it against the raw tarball and every table below
regenerates). Record: `benchmarks/bounds_verdicts_2026-08-10.{tsv,meta}`.

## The three verdicts, and the rule that assigns each

Applied mechanically, in this order, per site:

| verdict | rule | what it means |
|---|---|---|
| **UNMEASURED** | `fp_kind = whole` and `res_mean > 64 B` | the instrument declined to answer. **No narrowing verdict may be issued.** Fix: one `probe_declare_rows` call. |
| **BLOCKED-fp** | a concurrent foreign **FOOTPRINT** intersected, counterparty mutable | a genuine conflict. Not observed anywhere at t=8 — see the note below. |
| **BLOCKED-0** | `min_gap_mut == 0` | a concurrent foreign WRITE was **butt-adjacent**. Adding one byte on that side collides. |
| **COARSENABLE-INF** | `n_conc_mut == 0` | the site never met a concurrent foreign write **at any distance**. Extent is not its constraint; the tracker's wide path is. |
| **NARROWABLE-Nx** | `fp_kind = rows` and `over_ratio > 1.02` | the reservation exceeds the declared footprint by Nx. The row set is the tight extent. |
| **COARSEN-N** | otherwise | N = `min_gap_mut`, the closest **observed** approach to a concurrent foreign write. Widening by k collides `coll_k*` times. |

**Why "footprints intersect" never fires with a mutable side, and why that is
not a hole in the instrument.** A concurrent overlap involving a mutable
*reservation* is impossible by construction — `DisjointMut` panics on it — and a
footprint is a subset of its reservation. So at t=8 every surviving pair is in
the third bucket, "neither intersects; free up to `min_gap`", and `min_gap_mut`
is the number the map exists to print. `BLOCKED-fp` is retained because it is
the verdict a *proposed* extent earns once it is fed through the same test, and
because footprint intersection **is** observed 21 times between two immutable
readers (`mc.rs:1342:44` against itself) — a genuinely shared read region, which
is exactly why it cannot be OWNED by one worker.

**Which cell each verdict comes from.** Conflict facts are taken from
`corpus 8-bit/data` t=8 wherever the site runs there, and only from `v4k_8tile`
t=8 otherwise. That is not a preference, it is the finding: the 4K cell
**under-reports the collision rate by three orders of magnitude** (0.09 vs 91
per million registrations at the <= 4 KiB column), and the last column of the
TSV records which cell answered.

## The correction the pair table forces

`benchmarks/bounds_map_2026-08-10.meta` names CDEF (`cdef_arm.rs:622:9`, 232 B)
as the counterparty to the loop filter's read guard. The full pair table says
that is the **third** nearest writer. Sorted by closest approach:

| `loopfilter.rs:710:14` <- | co-live pairs | foreign is a WRITE | closest approach |
|---|---|---|---|
| `loopfilter.rs:887:14` (LF's own write-back) | 1,176,771 | yes, all | **60 B** |
| `include/dav1d/picture.rs:2027:22` | 512,458 | yes, all | **61 B** |
| `safe_simd/cdef_arm.rs:622:9` | 2,217,283 | yes, all | 232 B |
| `loopfilter.rs:710:14` (itself) | 2,907,038 | no | 99 B (immutable, no conflict at any width) |

and the site-level minimum in `BCONC` agrees: `min_gap_mut = 60`. **The loop
filter's nearest concurrent writer is another worker's loop-filter write-back,
60 bytes away.** Everything downstream that quoted 232 B as the budget was
quoting a per-pair figure, not the site's.

## The verdict table

Full machine-readable form: `benchmarks/bounds_verdicts_2026-08-10.tsv` (129 sites).
`res B` and `over` come from the cell named in the TSV's last column; `k=N` is the
cumulative count of acquisitions whose nearest concurrent foreign WRITE was within
N bytes, i.e. how often a widening by N collides.


| site | reg/frame | R/W | res B | over | waste | conc-write encounters | headroom N | 4K-cell N | k=64 | k=256 | k=1K | verdict | nearest concurrent writer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `loopfilter.rs:710:14` | 3,835,042 | R | 10.50 | 1.000 | <= 10 B | 3,952,914 | 60 | 1108 | 2 | 16 | 324 | **COARSEN-60** | loopfilter.rs:887:14@60B |
| `ctx.rs:99:27` | 2,534,988 | W | 2.93 | 1.000 | <= 3 B | 2,568 | 0 | 1 | 2,568 | 2,568 | 2,568 | **BLOCKED-0** | - |
| `safe_simd/cdef_arm.rs:192:9` | 646,912 | R | 7.90 | 1.000 | <= 8 B | 4,146,661 | 118 | 3848 | 0 | 8 | 219 | **COARSEN-118** | safe_simd/cdef_arm.rs:622:9@118B |
| `safe_simd/cdef_arm.rs:622:9` | 646,912 | RW | 5.91 | 1.000 | <= 6 B | 4,355,998 | 80 | 1344 | 0 | 8 | 189 | **COARSEN-80** | safe_simd/cdef_arm.rs:622:9@80B |
| `cdef_apply.rs:121:33` | 414,592 | R | 2.00 | 1.000 | <= 2 B | 1,648,638 | 100 | 3840 | 0 | 8 | 168 | **COARSEN-100** | safe_simd/cdef_arm.rs:622:9@100B |
| `cdef_apply.rs:104:32` | 254,784 | R | 2.00 | 1.000 | <= 2 B | 2,861,832 | 120 | 4032 | 0 | 2 | 30 | **COARSEN-120** | safe_simd/cdef_arm.rs:622:9@398B |
| `safe_simd/cdef_arm.rs:1217:9` | 247,040 | R | 8.00 | 1.000 | <= 8 B | 3,092,619 | 64 | 1320 | 1 | 1 | 26 | **COARSEN-64** | safe_simd/cdef_arm.rs:622:9@64B |
| `recon.rs:2734:46` | 188,130 | R | 1.73 | 1.000 | <= 2 B | 129 | 0 | 4 | 129 | 129 | 129 | **BLOCKED-0** | - |
| `ipred_prepare.rs:34:17` | 188,130 | R | 1.00 | 1.000 | <= 1 B | 214 | 0 | 7 | 214 | 214 | 214 | **BLOCKED-0** | - |
| `ipred_prepare.rs:47:23` | 188,130 | R | 1.00 | 1.000 | <= 1 B | 54 | 0 | 0 | 54 | 54 | 54 | **BLOCKED-0** | - |
| `recon.rs:2735:46` | 188,130 | R | 1.48 | 1.000 | <= 1 B | 0 | INF | INF | 0 | 0 | 0 | **COARSENABLE-INF** | - |
| `ipred_prepare.rs:37:21` | 188,130 | R | 1.00 | 1.000 | <= 1 B | 199 | 0 | 1 | 199 | 199 | 199 | **BLOCKED-0** | - |
| `safe_simd/cdef_arm.rs:223:18` | 161,728 | R | 9.23 | 1.000 | <= 9 B | 41,056 | 0 | 5206 | 52 | 213 | 1,430 | **BLOCKED-0** | - |
| `safe_simd/cdef_arm.rs:247:26` | 141,344 | R | 9.23 | 1.000 | <= 9 B | 1,414,012 | 486 | 24830 | 0 | 0 | 22 | **COARSEN-486** | safe_simd/cdef_arm.rs:622:9@486B |
| `recon.rs:2353:44` | 138,714 | R | 1.75 | 1.000 | <= 2 B | 0 | INF | INF | 0 | 0 | 0 | **COARSENABLE-INF** | - |
| `recon.rs:2352:49` | 138,714 | R | 1.98 | 1.000 | <= 2 B | 336 | 0 | 3 | 336 | 336 | 336 | **BLOCKED-0** | - |
| `env.rs:105:25` | 124,820 | R | 1.00 | 1.000 | <= 1 B | 56 | 0 | 3 | 56 | 56 | 56 | **BLOCKED-0** | - |
| `env.rs:105:72` | 124,820 | R | 1.00 | 1.000 | <= 1 B | 0 | INF | INF | 0 | 0 | 0 | **COARSENABLE-INF** | - |
| `env.rs:89:18` | 94,065 | R | 1.00 | 1.000 | <= 1 B | 0 | INF | INF | 0 | 0 | 0 | **COARSENABLE-INF** | - |
| `decode.rs:1974:35` | 94,065 | W | 2.64 | 1.000 | <= 3 B | 0 | INF | INF | 0 | 0 | 0 | **COARSENABLE-INF** | - |
| `decode.rs:1682:53` | 94,065 | R | 1.00 | 1.000 | <= 1 B | 0 | INF | INF | 0 | 0 | 0 | **COARSENABLE-INF** | - |
| `decode.rs:1973:34` | 94,065 | W | 3.45 | 1.000 | <= 3 B | 1,181 | 0 | 2 | 1,181 | 1,181 | 1,181 | **BLOCKED-0** | - |
| `decode.rs:1446:61` | 94,065 | R | 1.00 | 1.000 | <= 1 B | 0 | INF | INF | 0 | 0 | 0 | **COARSENABLE-INF** | - |
| `decode.rs:1977:48` | 94,065 | W | 1.58 | 1.000 | <= 2 B | 0 | INF | INF | 0 | 0 | 0 | **COARSENABLE-INF** | - |
| `decode.rs:1446:29` | 94,065 | R | 1.00 | 1.000 | <= 1 B | 232 | 1 | 5 | 232 | 232 | 232 | **COARSEN-1** | - |
| `decode.rs:1681:52` | 94,065 | R | 1.00 | 1.000 | <= 1 B | 268 | 0 | 14 | 268 | 268 | 268 | **BLOCKED-0** | - |
| `env.rs:90:24` | 94,065 | R | 1.00 | 1.000 | <= 1 B | 68 | 0 | 1 | 68 | 68 | 68 | **BLOCKED-0** | - |
| `decode.rs:1976:47` | 94,065 | W | 2.01 | 1.000 | <= 2 B | 278 | 0 | 4 | 278 | 278 | 278 | **BLOCKED-0** | - |
| `ipred_prepare.rs:232:24` | 39,505 | R | 19.97 | 1.000 | <= 20 B | 22 | 16 | 632 | 2 | 4 | 5 | **COARSEN-16** | - |
| `owned_recon.rs:937:42` | 25,920 | RW | 335.67 | 1.000 | NOT MEASURED | 24,237 | 0 | 0 | 18 | 45 | 146 | **UNMEASURED** | - |
| `safe_simd/cdef_arm.rs:256:25` | 19,712 | R | 9.22 | 1.000 | <= 9 B | 304 | 338 | 18702 | 0 | 0 | 6 | **COARSEN-338** | - |
| `loopfilter.rs:887:14` | 17,852 | RW | 4.85 | 1.000 | <= 5 B | 1,420,392 | 71 | 63819 | 0 | 4 | 90 | **COARSEN-71** | picture.rs:2027:22@79B |
| `lf_apply.rs:636:60` | 960 | R | 1.00 | 1.000 | <= 1 B | 0 | INF | INF | 0 | 0 | 0 | **COARSENABLE-INF** | - |
| `lf_apply.rs:623:54` | 960 | R | 1.00 | 1.000 | <= 1 B | 0 | INF | INF | 0 | 0 | 0 | **COARSENABLE-INF** | - |
| `cdef_apply.rs:84:26` | 944 | R | 456.23 | 1.000 | NOT MEASURED | 6,644 | 952 | 67056 | 0 | 0 | 1 | **UNMEASURED** | - |
| `cdef_apply.rs:83:30` | 944 | RW | 456.23 | 1.000 | NOT MEASURED | 819 | 0 | INF | 4 | 4 | 61 | **UNMEASURED** | - |
| `cdef_apply.rs:60:22` | 472 | R | 900.38 | 1.000 | NOT MEASURED | 6,581 | 4616 | 29040 | 0 | 0 | 0 | **UNMEASURED** | - |
| `cdef_apply.rs:59:26` | 472 | RW | 900.38 | 1.000 | NOT MEASURED | 339 | 0 | INF | 4 | 7 | 26 | **UNMEASURED** | - |
| `lf_apply.rs:126:30` | 396 | R | 610.25 | 1.000 | NOT MEASURED | 3,012 | 804 | 242136 | 0 | 0 | 2 | **UNMEASURED** | - |
| `lf_apply.rs:125:39` | 396 | RW | 610.25 | 1.000 | NOT MEASURED | 215 | 640 | INF | 0 | 0 | 6 | **UNMEASURED** | - |
| `recon.rs:3904:35` | 272 | RW | 179.64 | 1.000 | NOT MEASURED | 16 | 0 | INF | 3 | 3 | 9 | **UNMEASURED** | - |
| `internal.rs:719:40` | 136 | W | 27.11 | 1.000 | <= 27 B | 3 | 32 | INF | 2 | 3 | 3 | **COARSEN-32** | - |
| `decode.rs:4341:23` | 136 | R | 27.11 | 1.000 | <= 27 B | 0 | INF | INF | 0 | 0 | 0 | **COARSENABLE-INF** | - |
| `recon.rs:3887:27` | 136 | RW | 352.17 | 1.000 | NOT MEASURED | 5 | 0 | INF | 1 | 1 | 2 | **UNMEASURED** | - |
| `internal.rs:726:40` | 136 | W | 14.17 | 1.000 | <= 14 B | 2 | 16 | INF | 2 | 2 | 2 | **COARSEN-16** | - |
| `decode.rs:4350:24` | 136 | R | 14.17 | 1.000 | <= 14 B | 0 | INF | INF | 0 | 0 | 0 | **COARSENABLE-INF** | - |
| `internal.rs:712:24` | 102 | R | 24.36 | 1.000 | <= 24 B | 0 | INF | INF | 0 | 0 | 0 | **COARSENABLE-INF** | - |
| `internal.rs:713:24` | 102 | R | 12.55 | 1.000 | <= 13 B | 0 | INF | 2144 | 0 | 0 | 0 | **COARSENABLE-INF** | - |

## Table 2 — sites the 4K gap vectors NEVER EXECUTE, by corpus registrations/frame

| site | reg/frame | R/W | res B | over | waste | conc-write encounters | headroom N | 4K-cell N | k=64 | k=256 | k=1K | verdict | nearest concurrent writer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `mc.rs:121:61` | 128,999 | R | 1.00 | 1.000 | <= 1 B | 0 | INF | n/a | 0 | 0 | 0 | **COARSENABLE-INF** | - |
| `mc.rs:1342:44` | 47,870 | R | 8.00 | 1.000 | <= 8 B | 0 | INF | n/a | 0 | 0 | 0 | **COARSENABLE-INF** | - |
| `picture.rs:2027:22` | 38,489 | RW | 14.60 | 1.000 | <= 15 B | 1,450,464 | 0 | n/a | 886 | 4,689 | 16,953 | **BLOCKED-0** | picture.rs:2027:22@0B |
| `safe_simd/mc_arm.rs:5655:25` | 20,308 | R | 14.59 | 1.000 | <= 15 B | 707,173 | 0 | n/a | 386 | 2,247 | 8,778 | **BLOCKED-0** | picture.rs:2027:22@0B |
| `owned_recon.rs:570:45` | 7,192 | R | 16.15 | 1.000 | <= 16 B | 369,482 | 0 | n/a | 151 | 1,002 | 3,772 | **BLOCKED-0** | picture.rs:2027:22@0B |
| `owned_recon.rs:505:31` | 6,561 | RW | 11.75 | 1.000 | <= 12 B | 110,559 | 0 | n/a | 89 | 584 | 2,268 | **BLOCKED-0** | - |
| `picture.rs:168:38` | 6,184 | R | 1.00 | 1.000 | <= 1 B | 274,491 | 15 | n/a | 105 | 756 | 2,932 | **COARSEN-15** | - |
| `safe_simd/mc_arm.rs:5118:25` | 4,418 | R | 13.10 | 1.000 | <= 13 B | 116,361 | 16 | n/a | 22 | 184 | 718 | **COARSEN-16** | - |
| `safe_simd/mc_arm.rs:5240:25` | 4,388 | R | 8.37 | 1.000 | <= 8 B | 209,024 | 0 | n/a | 96 | 544 | 2,561 | **BLOCKED-0** | picture.rs:2027:22@0B |
| `owned_recon.rs:433:42` | 3,230 | RW | 8.35 | 1.000 | <= 8 B | 254,334 | 16 | n/a | 70 | 536 | 2,295 | **COARSEN-16** | picture.rs:2027:22@20B |
| `mc.rs:1354:29` | 3,191 | RW | 8.00 | 1.000 | <= 8 B | 233,405 | 0 | n/a | 110 | 696 | 2,911 | **BLOCKED-0** | picture.rs:2027:22@16B |
| `mc.rs:1457:20` | 2,829 | R | 22.96 | 1.000 | <= 23 B | 0 | INF | n/a | 0 | 0 | 0 | **COARSENABLE-INF** | - |
| `refmvs.rs:508:34` | 2,805 | W | 5.55 | 1.000 | <= 6 B | 38,080 | 0 | n/a | 2,227 | 4,169 | 9,075 | **BLOCKED-0** | refmvs.rs:508:34@0B |
| `safe_simd/mc_arm.rs:5971:41` | 2,515 | R | 2466545.57 | 1.000 | NOT MEASURED | 0 | INF | n/a | 0 | 0 | 0 | **UNMEASURED** | - |
| `refmvs.rs:645:25` | 1,606 | R | 1.00 | 1.000 | <= 1 B | 13,701 | 0 | n/a | 701 | 1,551 | 3,382 | **BLOCKED-0** | - |
| `refmvs.rs:1589:33` | 1,589 | R | 1.00 | 1.000 | <= 1 B | 0 | INF | n/a | 0 | 0 | 0 | **COARSENABLE-INF** | - |
| `refmvs.rs:715:25` | 1,567 | R | 1.00 | 1.000 | <= 1 B | 13,373 | 1 | n/a | 673 | 1,427 | 3,084 | **COARSEN-1** | - |
| `env.rs:168:25` | 1,498 | R | 1.00 | 1.000 | <= 1 B | 9 | 4 | n/a | 9 | 9 | 9 | **COARSEN-4** | - |
| `safe_simd/mc_arm.rs:4741:25` | 1,452 | R | 26.89 | 1.000 | <= 27 B | 120,639 | 0 | n/a | 62 | 308 | 1,377 | **BLOCKED-0** | - |
| `looprestoration.rs:463:37` | 1,368 | R | 245.10 | 1.000 | NOT MEASURED | 108,322 | 901 | n/a | 0 | 0 | 1 | **UNMEASURED** | - |
| `refmvs.rs:1674:16` | 1,302 | W | 2.92 | 1.000 | <= 3 B | 2,634 | 6 | n/a | 84 | 263 | 803 | **COARSEN-6** | - |
| `refmvs.rs:1660:29` | 1,302 | R | 1.00 | 1.000 | <= 1 B | 15,085 | 0 | n/a | 615 | 1,440 | 3,343 | **BLOCKED-0** | - |
| `env.rs:169:38` | 1,294 | R | 1.00 | 1.000 | <= 1 B | 9 | 8 | n/a | 9 | 9 | 9 | **COARSEN-8** | - |
| `refmvs.rs:1576:38` | 1,237 | W | 1.00 | 1.000 | <= 1 B | 108,064 | 0 | n/a | 6,684 | 20,866 | 54,794 | **BLOCKED-0** | refmvs.rs:1576:38@0B |
| `picture.rs:589:26` | 1,181 | R | 4096.00 | 1.000 | NOT MEASURED | 0 | INF | n/a | 0 | 0 | 0 | **UNMEASURED** | - |
| `refmvs.rs:1554:33` | 1,106 | R | 1.00 | 1.000 | <= 1 B | 0 | INF | n/a | 0 | 0 | 0 | **COARSENABLE-INF** | - |

The remaining ~100 corpus-only sites are in the TSV.


## The map's own correctness gate: the three refutations

A map that produced a fresh crop of candidates but did not condemn the three
attempts we already know failed would be worthless. This gate is more important
than any candidate below it.

### #469 — the strided rectangle (exact record, hull reference)

The instrument carries the counterfactual directly: for every acquisition it
also tests the **row band** — the whole picture rows the reservation spans,
which is exactly the hull a rectangle hands back.

| cell | row-band hits a foreign reservation | ... a MUTABLE one | ... a foreign FOOTPRINT | ... a MUTABLE footprint |
|---|---|---|---|---|
| corpus `8-bit/data` t=8 (1406 frames) | 590,652 | **18,196** | 384,380 | **7,612** |
| `v4k_8tile` t=8 (3 frames) | 12 | **12** | 6 | **6** |
| `v4k_8tile_10b` t=8 | 0 | 0 | 0 | 0 |

**BLOCKED.** 18,196 collisions per corpus pass. (The corpus row-band column is
softened by `late_stride = 1,915,051` — recycled pool addresses can carry a
stale stride — so treat the magnitude as approximate and the sign as solid; the
4K cells have `late_stride = 3` and are clean, and they are also 1500x quieter.)
The map cannot see #469's *other* defect — the record and the reference being
different objects is an aliasing-model fact that only Miri checks — so this is
one of two independent reasons it fails.

### #475 — the hull arm


| site | n/frame | reserved B | footprint B | over | gap waste | lead | tail | verdict |
|---|---|---|---|---|---|---|---|---|
| `cdef_apply.rs:121:33` | 51,824 | 26,882.0 | 16.00 | **1680.1x** | 26,866.0 | 0.00 | 0.00 | NARROWABLE (row set) |
| `cdef_apply.rs:104:32` | 31,848 | 26,882.0 | 16.00 | **1680.1x** | 26,866.0 | 0.00 | 0.00 | NARROWABLE (row set) |
| `safe_simd/cdef_arm.rs:622:9` | 80,864 | 26,888.0 | 64.00 | **420.1x** | 26,824.0 | 0.00 | 0.00 | NARROWABLE (row set) |
| `safe_simd/cdef_arm.rs:1217:9` | 30,880 | 26,888.0 | 64.00 | **420.1x** | 26,824.0 | 0.00 | 0.00 | NARROWABLE (row set) |
| `loopfilter.rs:769:33` | 372,017 | 35,755.9 | 90.71 | **394.2x** | 35,665.2 | 0.00 | 0.00 | NARROWABLE (row set) |
| `safe_simd/cdef_arm.rs:192:9` | 80,864 | 26,890.0 | 79.94 | **336.4x** | 26,810.1 | 0.00 | 0.00 | NARROWABLE (row set) |


**OVER-RESERVED, 336x to 1680x, and the waste is entirely inter-row gap** —
`lead` and `tail` are 0.00 at every one of the six. `loopfilter.rs:769:33`
reserves 35,756 B to touch 90.7.

Note what the map does **not** say: the LF hull's own row-band counterfactual is
`row_ovl = 0` on the corpus, i.e. it would not actually collide. That is
consistent, not contradictory — #475 was refuted for **cost** (2.65x at t=8,
`docs/MUT_RECON_KERNELS.md` §11c), because a 50-60 KB extent lands on the
tracker's wide path. The map condemns it on over-reservation; §11c condemns it
on wall clock; neither says it is unsound. **Both axes have to be read.**

### #485 — the loop-filter read band


| cell | frames | n | headroom N | k<=64 | k<=256 | k<=1K | k<=4K |
|---|---|---|---|---|---|---|---|
| v4k_8tile t=8 | 1 | 11,505,126 | 1108 | 0 | 0 | 0 | 1 |
| v4k_8tile_10b t=8 | 1 | 11,887,518 | 2136 | 0 | 0 | 0 | 1 |
| corpus 8-bit/data t=8 | 1406 | 64,127,608 | 60 | 2 | 16 | 324 | 5,819 |
| corpus 10-bit/data t=8 | 284 | 873,966 | 194 | 0 | 1 | 18 | 307 |


**BLOCKED.** The band replaced a W in {4..16} px per-edge guard with a 128-px row
copy-in: a widening of up to ~124 B at 8bpc, which lands between the `k<=64`
column (2) and the `k<=256` column (16). **Predicted 2 to 16 decode failures per
1406 frames of `8-bit/data`; #485 measured 1, 2 and 0 on three full passes of
exactly that group.** And **0** predicted on either 4K vector, which is why its
first full 4K-shaped sample came back 753/755 clean.

### Gate verdict

All three are marked. The map passes its own correctness check.

## CONTRADICTED: the V-pass fused run's soundness TEST (PR #488 §19d)

**Status first, so nothing here is read as an objection to a merge that is not
being proposed.** PR #488 built `LF_BATCH_V` 4 -> 32, measured it as **no
wall-clock win** (t=8 ratio 1.0005, 4/8 rounds faster, p=1.000), and **reverted
it** — `src/loopfilter.rs` on that branch is back to `f87b12c`. What it keeps is
the record and an explicit soundness argument, preserved deliberately as "the
useful part". **That argument is what this section contradicts**, because it will
be the basis of the next attempt at this site.

### The test, quoted, and why it is not the collision criterion

> *does the reservation contain a byte no member of the batch reads?*
> Strided hull — yes, the gaps between rows. Read band — yes, the gaps between
> edges. Fused run — **no** (sound).
> — PR #488, `docs/MUT_RECON_KERNELS.md` §19d

The premise is correct: a fused run of adjacent filtering groups is slack-free.
The conclusion does not follow, because **`DisjointMut` does not compare a
reservation against a read set. It compares one live reservation against
another.** Slack-freedom rules out #485's specific failure mode (a foreign write
landing in a gap nothing reads); it does not rule out a foreign write landing in
a byte the run *does* read, and a reservation 8x wider is 8x more of a target.

From the change's own source (`git show 61f88dc:src/loopfilter.rs`, `fill_v`,
`LF_SW_V = 4 * LF_BATCH_V`): under tile threading each live reservation goes from
**16 px to 128 px** — 128 B at 8bpc, 256 B at 10/12bpc.

### The price, from the budget column for `loopfilter.rs:710:14`

| cell | widening | acquisitions with a concurrent WRITE that close | x V-pass share (30.7%) |
|---|---|---|---|
| corpus `8-bit/data` t=8, 1406 frames | +112 B | **2 .. 16** | **0.6 .. 4.9** |
| corpus `10-bit/data` t=8, 284 frames | +224 B | 0 .. 1 | 0.0 .. 0.3 |
| `v4k_8tile` t=8 | +112 B | **0** | 0 |
| `v4k_8tile_10b` t=8 | +224 B | **0** | 0 |

and the site's headroom is **60 B**, to `loopfilter.rs:887:14` — the loop
filter's own write-back in another superblock-row filter task — not the 232 B
the earlier record named.

### The evidence AGAINST this prediction, stated before the argument for it

**PR #488 ran the corpus on the implementation and reports 766/766 at t=1 and
753/753 at t=8.** That is a direct observation and it is not consistent with the
high end of the range above.

It is not consistent with the low end being wrong either, and the reason is in
that PR's own honest gap: it also planted a **genuine** over-reservation (the
un-chunked write-back, above 16 columns) and measured **358/358 pass, 0 errors**
on `8-bit/data --threads 8`. One corpus pass could not see a real widening. #485
is the same story from the other side — 1, 2 and **0** errors on three passes of
the same group. **A single clean pass at a predicted rate of ~1-5 per pass is
weak evidence of absence, which is the whole reason this instrument exists.**

So: prediction not confirmed, not refuted. What would settle it, cheaply:
`--features __probe_bounds` on top of `61f88dc` for one `8-bit/data` t=8 pass —
`mutable_overlaps` must stay 0 and the decode must not panic. That measures the
disputed quantity directly instead of sampling a rare event once.

### What a collision here would MEAN — and this is the part nobody has settled

If a fused reservation overlaps a concurrent foreign write, the byte is one the
run genuinely reads (slack-freedom guarantees that much). Two readings, and they
have opposite consequences:

* **True positive.** The unfused schedule reads group `g`'s tap window at the
  moment it filters `g`; the fused schedule reads all 32 windows **up front**.
  If the decoder's ordering only guarantees "edge `g`'s inputs are final before
  `g` is filtered", reading edge `g+31`'s window early can read pre-write data
  where the sequential schedule read post-write data — **different pixels**, in
  an `unchecked` build with no panic to notice it. Then the fusion is not sound
  and the tracker is right.
* **False positive.** If the ordering guarantee is per-run or coarser, the early
  read is harmless and the panic is spurious.

**Which one holds is a question about the filter task's ordering contract, and
neither PR #488 nor this map answers it.** It is answerable by reading
`src/thread_task.rs`'s sbrow-filter dependency edges, and it should be answered
BEFORE the next attempt, because the two readings call for opposite fixes.
Either way the operational outcome is the same: the tracker panics and the
decode fails.

### What the budget DOES support at this site

At 8bpc a run of 8 groups is 32 px = 32 B, inside the 60 B clearance; 16 groups
is 64 B, outside it. At 16bpc 4 groups already reach 32 B and 8 groups reach
64 B, outside it. So the evidence supports **`LF_BATCH_V = 8`, 8bpc only** — a 2x
count cut on 30.7% of the site, ~589 K registrations/frame at 4K. PR #488's
milliseconds arithmetic then applies to it in full: at the 4.04 ns/registration
marginal rate that is ~2.4 ms/frame of CPU, **<= 0.5% of a t=8 frame**, which
that PR measured as indistinguishable from zero on this box. **The honest joint
conclusion of both rounds is that this site is not worth attacking on extent at
all** — its prize is under the noise floor and its clearance is 60 bytes.

### Caveats on the prediction, all of which I could not remove

* The histogram is **not split H vs V**. The 30.7% scaling assumes the V
  acquisitions carry the H acquisitions' gap distribution.
* It is **direction-blind**: it records distance to the nearest concurrent write
  without recording which side, so a one-sided widening collides with roughly
  half the column.
* `min_gap_mut = 60` is an **observed** minimum over 1406 frames.
* The `rows` figure the standing check computes is `bytes.div_ceil(row_bytes)`,
  which is alignment-blind: a 128-byte reservation on a 128-byte row reads as
  one row even if it straddles two. On the corpus's small frames a 60-byte gap
  can therefore be a *different row* rather than an adjacent column, and the map
  cannot tell those apart from a 1-D byte range.

## The standing assertions

The map's value decays the moment someone widens a guard without re-reading it.
These turn its two load-bearing facts into checks that fail **at the moment of
the widening**, not three weeks later on one CI runner.

The invariant lives at the single funnel every tracked picture-plane reservation
passes through — `Rav1dPictureDataComponent::{slice, slice_mut}` in
`include/dav1d/picture.rs`, via `note_pic_extent`:

1. **A reservation may not span more than ONE picture row.** The inter-row gaps
   belong to other columns of the same rows, and AV1 tiles partition a frame BY
   COLUMN. This is #469's and #475's defect, stated as an invariant.
2. **A reservation may not exceed its file's `PIC_EXTENT_CEILINGS` entry**, or
   one picture row where the file has no entry. This is #485's, and the V-batch's.

`index`/`index_mut` are exempt because one element is the smallest reservation
expressible. Whole-component reservations (`full_guard`, `full_guard_mut`,
`copy_pixels_to`, `copy_from`) are exempt because they are unambiguous,
greppable and deliberate — and because the map measured those sites as having
**no concurrent foreign write at any distance**. Everything at t=1 is exempt by
the `tile_threading_active()` antecedent, which is correct: the hull paths
deliberately over-reserve 153x-1680x there and the count reduction is worth 2.6%
(§11d).

### Why extent and not "reserved <= footprint"

Because at t=8 the ratio is already 1.000 at every hot site — **there is no
over-reservation left to assert against** — and because two of the three refuted
attempts kept `reserved == footprint` while widening. A ratio assertion would
have passed #485, passed the V-batch, and caught only #475.

### The ceilings, and where they come from

Measured by the invariant's own counters over **575,577,925** reservations
across 1,734 frames (committed crash vectors + `dav1d-test-data` `8-bit/data`
and `10-bit/data`, 4 frames each, t=8):

| file | max B under tile threading | registrations | ceiling | why that number |
|---|---|---|---|---|
| `src/loopfilter.rs` | **32** | 90,170,282 | **32** | `LF_BW = 16` px x 2 B. The decoder's largest site and the one with the least headroom. |
| `src/safe_simd/cdef_arm.rs` | 24 | 192,238,382 | 32 | <= 16 px x 2 B |
| `src/lr_apply.rs` | 8 | 1,384,827 | 32 | 4 px x 2 B |
| `src/owned_recon.rs` | 3840 | 26,070,390 | one row | frame-scaling |
| `src/cdef_apply.rs` | 3840 | 72,788,816 | one row | `backup2lines` copies whole rows |
| `src/lf_apply.rs` | 3840 | 95,292 | one row | frame-scaling |
| `src/mc.rs` | 640 | 79,807,872 | one row | not constant-bounded |
| `include/dav1d/picture.rs` | 128 | 63,952,504 | one row | not constant-bounded |
| `src/safe_simd/mc_arm.rs` | 128 | 45,093,084 | one row | not constant-bounded |
| `src/looprestoration.rs` | 352 | 1,992,021 | one row | stripe-width-scaling |
| `src/safe_simd/looprestoration_arm.rs` | 352 | 1,984,455 | one row | stripe-width-scaling |

**`MAX_ROWS_TT = 1` over all 575 M** — under tile threading, nothing in the
shipped decoder reserves across a row boundary. That is the strongest single
fact the standing check pins.

**Honest limitation.** Only the three constant-bounded files get a tight entry.
A widening introduced in a file that legitimately copies whole rows
(`cdef_apply.rs`, `lf_apply.rs`, `owned_recon.rs`) is caught only by the
one-row rule, so a 128-byte in-row widening THERE would pass. #485's band lived
in the loop filter, and so does the V-batch, which is why the loop filter has
the tight entry; a future band written into `lf_apply.rs` would need its own.

### Cost

Compiled under `debug_assertions` **or** `--features probe-sites` only. The
default release build has no counter, no atomic load and no branch at the
funnel; `cargo build --release` is warning-clean and the site is `#[cfg]`-ed out
entirely.

### Coverage

* Plain `cargo test` (debug) carries the check inside the decoder.
* `tests/guard_extent_budget.rs` runs it in **release** with three liveness
  assertions — evaluated at all, evaluated **under tile threading**, and the
  whole-component exemption exercised — so it cannot go quiet.
* CI job **`extent-gate`** runs the corpus leg. The corpus leg is caller-gated
  by `RAV1D_EXTENT_GATE_CORPUS`; the test body never decides to skip.

### Teeth, proved by mutation

| plant | result |
|---|---|
| baseline | ok |
| `--features __probe_lf_hull` + `RAV1D_LF_HULL=1` (#475's shape; **no source edit** — the switch is already in the tree) | **FAILED** — 17 in-decoder panics, `src/loopfilter.rs:769:33 took a 1924 B picture-plane reservation ... spans 16 rows`, and the test's `MAX_ROWS_TT` assertion at `left: 16, right: 1` |
| LF per-row read widened `W` -> `W + 96` (the V-batch's shape, **within one row**) | **FAILED** — `src/loopfilter.rs:710:14 took a 100 B picture-plane reservation while tile threading is active; the measured ceiling for that file is 32 B` |
| an `unsafe` block planted in `tile_threading_active()` | **FAILED to compile** — `error: usage of an unsafe block`, anchored on `lib.rs:13 forbid(unsafe_code)` |
| after restore | ok |

All restores verified byte-exact (`sha256` match on `src/loopfilter.rs` and
`include/dav1d/picture.rs`, `git diff --exit-code` clean) and re-run **after
`touch`**, per the mtime trap in §7.

## The ranked candidate list

**Nothing here is implemented, on purpose.** The value of the map is that the
next round picks from evidence; a half-built candidate would muddy the artefact.
Ranked by registrations/frame x confidence. "Confidence" is how much of the
verdict rests on measurement versus inference, and it is stated per row.

### 1. The 31.4% that can never collide — reference-frame reads (HIGH confidence, needs a design, not an extent)

**293,787,973 of 936,867,496 corpus registrations at t=8 (31.4%) are at sites
whose `no_conc_write` column equals `n` exactly: no concurrent foreign write at
any distance, ever.** The two largest are `mc.rs:121:61` (181,371,944 = 19.4% of
the whole corpus population, 1.00 B each) and `mc.rs:1342:44` (67,305,240,
8.00 B). They read **reference frames**, which are immutable for the entire
decode.

This is the single largest finding in the map and it is not an extent question
at all: those registrations exist to detect a conflict that is structurally
impossible. The lever is to stop tracking that population — a distinct
allocation whose immutability is a decode-lifetime invariant — not to widen or
narrow it.

Two cautions, both measured:

* **Do not reach for a hull.** `mc.rs:121:61`'s eight taps are eight *rows*, so
  fusing them is #475's shape: 27 KB at a 4K stride, straight onto the tracker's
  wide path (2.65x, §11c). Conflict-freedom licenses the *extent*, not the cost.
* **The 4K gap vectors cannot see any of this.** `photo_4k.avif` is an all-intra
  still: `src/mc*` is **0 registrations** there against 309,617,604 (33.0%) on
  the corpus. Any measurement of this population must use the corpus.

### 2. `loopfilter.rs:710:14` — 3,835,042/frame, 33.6% (HIGH confidence, budget = 60 B)

`COARSEN-60`. The decoder's largest site, and the tightest budget in the table.
**Demoted to a non-candidate by the joint reading of this map and PR #488.** The
extent the budget allows is `LF_BATCH_V = 8` at 8bpc only (32 B, inside 60), not
32 (128 B, outside it), and nothing at 16bpc where 4 groups already reach 32 B —
and that 2x cut is ~589 K registrations/frame at 4K, ~2.4 ms/frame of CPU at the
4.04 ns marginal rate, which PR #488 measured as indistinguishable from zero
(t=8 ratio 1.0005, p=1.000) for a change twice as large. **A 60-byte clearance
and a prize under the noise floor: leave it.** The H pass (69.3%) cannot be
helped by batching in any form — its rectangle grows in the ROW direction, so a
fused run takes proportionally more per-row guards and the count is unchanged
(the change's own census: `1.000x`).

### 3. `ctx.rs:99:27` — 2,534,988/frame, 22.2% (HIGH confidence, budget = ZERO)

`BLOCKED-0`. 36 acquisitions had a concurrent foreign WRITE at gap **0** and 263
within 4 bytes, over 60,650,960 corpus registrations. **Widening this site by one
byte collides 36 times per corpus pass.** It is a 32-byte context array, not a
picture buffer, and `CaseSet::set_disjoint`'s mean extent is 2.93 B.

The only lever here is count, and only by restructuring what `set_disjoint`
writes — never by extent. Anyone who reaches for this site should read this row
first; it is the second-largest in the decoder and the map's flattest refusal.

### 4. The CDEF quartet — 1,563,240/frame combined (MEDIUM confidence)

| site | reg/frame (4K t=8) | res B | budget | k<=64 | k<=256 |
|---|---|---|---|---|---|
| `safe_simd/cdef_arm.rs:192:9` | 646,912 | 7.90 | 118 B | 0 | 8 |
| `safe_simd/cdef_arm.rs:622:9` | 646,912 | 5.91 | 80 B | 0 | 8 |
| `cdef_apply.rs:121:33` | 414,592 | 2.00 | 100 B | 0 | 8 |
| `cdef_apply.rs:104:32` | 254,784 | 2.00 | 120 B | 0 | 2 |

All four have **zero** collisions at `k <= 64` on the corpus, which is 8-32x
their current extent. A fusion that keeps each reservation at or under 64 bytes
is inside the measured budget at all four. Medium and not high confidence
because 64 B is close enough to the 80 B minimum at `622:9` that an unmeasured
vector could close it, and because these sites' counts are 4K-cell figures while
their budgets are corpus figures.

### 5. The 3.7 M/frame of `COARSENABLE-INF` context reads (MEDIUM confidence)

`recon.rs:2735:46`, `recon.rs:2353:44`, `env.rs:105:72`, `env.rs:89:18`,
`decode.rs:{1974,1682,1977,1446:61}` and friends — each 94 K-188 K/frame at the
4K cell, each with `n_conc_mut = 0`. Note the **pairing**: at almost every one of
these lines the sibling index on the same source line is `BLOCKED-0`
(`env.rs:105:25` gap 0, `recon.rs:2734:46` gap 0, `decode.rs:1973:34` gap 0).
Two indices, one line, opposite verdicts. Any change here has to treat them
separately, and a fusion that merges the pair inherits the blocked one's zero
budget.

### 6. Close the instrument's own hole — 11,238,576 registrations (1.2%) are `UNMEASURED`

`safe_simd/mc_arm.rs:5971:41` (3,536,733 guards of mean **2,466,546 B**),
`mc_arm.rs:6182:41` (541,865 B), `picture.rs:589:26` (4,096 B),
`looprestoration.rs:{382:29, 408:25}` (~2,050 B), `looprestoration.rs:463:37`
(245 B), `safe_simd/looprestoration_arm.rs:{351:51, 1226:51}` (~250 B),
`owned_recon.rs:937:42` (336 B), `cdef_apply.rs:{59,60,83,84}` (456-900 B),
`lf_apply.rs:{125,126}` (610 B), `recon.rs:{3887,3904}` (180-352 B).

Their `over_ratio = 1.000` is the instrument declining to answer, not an answer.
Fix is one `probe_declare_rows` call each, exactly as `narrow_guard` and
`LfBlock::fill_hull` already have; the concurrency half of their row is exact
and unaffected. Cheapest item on this list and it removes the map's largest
known blind spot.

### 7. Re-run the map on what it has never seen

Not a coarsening candidate — a coverage debt, and the reason it is on the list
is that item 1 was invisible until the corpus ran.

* **205 of 253 sites never execute on either 4K gap vector**, and only 49.5% of
  corpus registrations land at a site the 4K vector runs at all. `src/mc*`
  (33.0%), `looprestoration`/`lr_apply` (0.6%), `refmvs` (2.4%),
  `picture.rs:2027:22` (5.8%) are all **zero** there.
* Never measured at all: x86_64, wasm32, `asm`/`c-ffi`, `unchecked`, t=2/4/16,
  12-bit, film grain, and the `8-bit/{issues,size,intra,mv,mfmv,resize,
  vq_suite,cdfupdate,quantizer}` groups.

