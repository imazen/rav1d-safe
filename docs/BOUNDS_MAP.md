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
  **That site was since cut by 21-25% WITHOUT touching any extent** (PR #492,
  `docs/CTX_TL_SPLIT.md`): half its registrations were the worker-local LEFT
  neighbour context, which is `&mut`-reachable and needed no record at all. The
  map's value here was negative-space — it closed the coarsening family for this
  site early, which is what left "fewer registrations at the same extent" as the
  direction to look in.
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
