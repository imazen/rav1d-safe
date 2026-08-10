# Which ownership model, and when

Decision rules for pixel-buffer ownership in a `forbid(unsafe_code)` decoder, derived from the
2026-08 rav1d-safe campaign. **Every claim here is measured**; the record for each is named. The
point is that the next person picks a model from evidence rather than from taste, and does not
re-derive the four models that lost.

---

## The short answer

| model | verdict | why |
|---|---|---|
| `split_at_mut` the shared plane | **impossible here** | there is no `&mut` to split, at any point |
| Arc-per-tile, split then rejoin | **blocked by the same thing** | the rejoin was never the problem |
| tile-keyed locks over the shared plane | **unsound unless total** | partial keying misses real overlaps |
| exact-record / wide-reference (strided rect) | **UB** | the record and the reference are different objects |
| **owned per-worker, column-compact, one sbrow tall** | **ships** | #482: on the ceiling, +1.6 MB, no `unsafe` |

Copy/reshuffle is **cheap and worth it** — 60:1 in our numbers — but not for the reason people
expect (see §5).

---

## 1. `split_at_mut` on the shared picture — impossible, and the blocker is upstream

The obvious idea: rows are contiguous, so `chunks_mut(stride)` gives provably-disjoint rows, and
`split_at_mut` at tile column boundaries splits each row into disjoint per-tile segments. Safe,
zero-cost, no tracker.

It cannot be done here, for three independent reasons, each verified:

- The plane arrives as a **raw pointer through a C-ABI allocator callback**
  (`include/dav1d/picture.rs:1780`, `unsafe { Rav1dPictureDataComponentInner::new(ptr, len, stride) }`).
  There is no owned slice to split. The pointer *is* the public API.
- It lives behind `Arc<Rav1dPictureData>` (`picture.rs:1287`), shared with `sr_cur`, the ref slots
  and the output queue **during** decode.
- Every tile and filter task opens `fc.data.try_read()` and receives `&Rav1dFrameData`
  (`src/thread_task.rs:1237`, `:1416/:1436/:1451`). **Shared read guards *are* the tile
  parallelism.** And workers are `'static` pool threads (`src/lib.rs:221`), so no scoped-thread
  lifetime can carry a `&'a mut` split into them.

**Generalisable rule:** before designing around a split, check that an exclusive reference exists
*at the point you intend to split*, and that the consumer can accept a non-`'static` lifetime. A
campaign round was lost by verifying the *access pattern* was compatible and never checking whether
the `&mut` was obtainable at all.

**Do not over-generalise this negative.** It rules out splitting *the shared plane*. It says nothing
about owning a *different* buffer — which is what actually worked.

## 2. Arc-per-tile, split at construction, rejoin at the end — right shape, same blocker

Split at allocation while exclusive, hand each tile its own `Arc`, then `Arc::get_mut`/`into_inner`
after the tile tasks join to recover exclusivity for the cross-tile filter pass.

The rejoin half is sound and is the correct way to hand off. `Arc` also solves the `'static`-worker
problem that kills a plain `&'a mut` split. But construction-time splitting still needs an owned
buffer to split, and §1 says there isn't one. Sidestep it by *allocating your own* — §4.

## 3. Keying locks by tile — unsound unless the keying is total

Replace address-hashed shards with one lock per tile index. Tiles are disjoint by AV1's own
guarantee, so tile borrows never need comparing.

Measured (#473) and **rejected as unsound**: a keyed borrow lands in its tile's shard, an unkeyed
one in an address-hashed shard. Those are **different partitions of the same array — they never
meet**, so a genuine recon-vs-filter overlap is silently missed. The filter chain is 6,391,462
registrations/frame at t=8, far too many to route through the wide path (#469's avalanche measured
2.28x slower).

**Rule: a partial keying scheme is worse than none.** Either every population carries the key, or
the key cannot be used to decide exclusion.

## 4. Owned buffers — the model that works, and the four parameters that decide whether it pays

Both shipped variants allocate our own memory and copy out. They differ only in parameters, and the
parameters are worth more than the idea:

| | #474 (first attempt) | **#482 (shipped)** |
|---|---|---|
| granularity | per **tile** | per **worker** |
| extent | full plane | **one superblock row** |
| stride | the **picture's** | **its own** (column-compact) |
| peak RSS | +96.3 / +191.0 MB | **+1.6 / +3.2 MB** |
| tracker on recon | still there (uncontended) | **gone** |

**(a) Own stride, not the picture's.** #474 inherited the picture's stride so frame coordinates
indexed the same pixel — a three-line seam. The cost is structural residency: a tile writes a narrow
column band of *every row it owns*, and a 16 KiB page spans ~4 rows of a 3840-byte stride, so the
tile's **whole row range** goes resident. 60x the memory for a coordinate convenience.

**The two changes are one change.** A kernel signature taking `(&mut [u8], base, stride)` already
carries what a compact buffer needs, so the coordinate translation (#473 priced it at 22 recon sites
+ ~10 `f.cur.stride[..]` reads) is **free once you are changing the signature anyway**. Doing them in
sequence pays it twice; that is why #474 dodged it and #482 got it for nothing.

**(b) Per worker, not per tile.** Fewer buffers (`n_workers`, not `n_tiles`), reused across
superblock rows *and across frames*, so pages fault in once per sequence rather than per frame.

**(c) Bound the extent by what the consumer actually reads, and enforce it.** Reconstruction never
reads a picture row above the current superblock row (`src/recon.rs:2243`, `:3138` gate on the
sb-row boundary and source the top edge from `f.ipred_edge`). So one sbrow is enough — and the bound
is *enforced*, not assumed: `Band::at` translates by subtracting the band origin, and an access above
it fails a `checked_sub`. **The failure mode is a panic, never a silent wrong pixel.**

**(d) The consumer must be a single worker for the region's whole lifetime.** This is the real
precondition. The band is a field of `Rav1dTaskContext` — the worker's own struct, already `&mut` in
every recon signature — so exclusion is a **borrowck fact with no runtime record**. That is why it
needs no tracker, no key, and no `unsafe`.

**Where it does not apply:** the filter chain crosses tile boundaries
(`src/lf_apply.rs:563`/`:608`, CDEF's 2 px overread, LR stripes), so per-tile ownership is settled-
impossible there. Its natural unit is a row band with a **halo**, which is a different (and still
open) problem.

## 5. Copy/reshuffle — yes, and the reason is not the one you expect

**Measured at 4K (`examples/tile_stitch_cost.rs`, real geometry, median of 9):**

- `stitch_sbrow` (the shape a decoder needs): **0.315 ms/frame**
- tracker cost it replaces: **19.7 ms/frame** at t=8

**A 60:1 trade.** Copy whenever `copy_cost << coordination_cost`, and measure both rather than
assuming the copy is the expensive one — the intuition that copying 12 MB must be costly is simply
wrong at these ratios.

**But not for cache reasons.** `write_into_tile` 2.567 ms vs `write_into_pic` 2.553 ms, **bands
overlap — null**. Writing into a small contiguous buffer is *not* faster than strided writes into a
12 MB frame. Budget the copy as pure (cheap) added cost; do not credit it with a locality bonus.

## 6. The trap that produced UB, and it is about neither ownership nor copying

`#469` narrowed the tracker **record** to an exact rectangle and handed back a `&mut` over the
**hull**. The CHANGELOG documented the split as a design choice. It is UB: Miri flags it under both
memory models, and it fired as a real decode failure in CI.

**The tracker record and the reference a guard hands out are DIFFERENT OBJECTS, and both must be
exact.** Rust's aliasing rules bind to the reference; they do not care what your bookkeeping
believes. Every tracker test passed, and all 766 corpus vectors passed — only Miri caught it.

The same asymmetry appears in the guard itself: a guard holding `&'a mut V` is UB when moved
by value, because the move gives the reference a *protector* for the duration of the call while
`Drop` retires the record inside it, authorising another thread to retag those bytes. Fixed in #478
by holding `NonNull<V>` and materialising the reference in `Deref`/`DerefMut` — the same reason
`core::cell::RefMut` does it, and sharper here because another thread genuinely retags.

**If you take one rule from this file:** run Miri on any new guard shape, under both Stacked Borrows
and Tree Borrows, with each test in isolation. It is the only tool here that checks the aliasing
model instead of the bookkeeping, and it has now caught two defects that every other gate missed.

## 7. The four cases, in detail

The models above are not equally applicable. What decides it is **who writes a pixel, and whether
any other worker can touch it while they do.** That question has four different answers here.

### 7a. Intra reconstruction — SOLVED, owned band (#482)

Three properties line up, and all three are needed:

- **Tile independence is a spec guarantee.** Intra prediction, MV prediction and entropy contexts
  all reset at tile boundaries, so a tile worker never reads another tile's pixels while
  reconstructing. This is what makes ownership *legal*.
- **The vertical reach is bounded and already handled.** Reconstruction never reads a picture row
  above the current superblock row: `src/recon.rs:2243`/`:3138` gate on the sb-row boundary and take
  the top edge from `f.ipred_edge` (via `rav1d_prepare_intra_edges`'s `prefilter_toplevel_sb_edge`)
  rather than from the plane. That is what makes a **one-sbrow** band sufficient.
- **The writer is one worker for the task's lifetime.** So `&mut` + borrowck, no record.

Result: on the measured ceiling at all eight cells, +1.6 MB, no `unsafe`. **This is the case
ownership was made for**, and it is worth noticing that it works because the *bitstream format*
guarantees the disjointness — not because we arranged it.

### 7b. Tiling — an enabling axis, not a case of its own

Tiles are *why* 7a is possible. But note the asymmetry that cost a round: tile disjointness is a
sound basis for **ownership** (7a) and an unsound basis for **keyed locking** (§3), because keying
only decides exclusion if *every* population carries the key. Same underlying fact, opposite
conclusions, depending on whether you use it to avoid coordination or to implement it.

Second-order: tile count caps tile parallelism. A small AVIF may be single-tile, in which case
`t>1` cannot help regardless of ownership model — a decode-configuration question, not a memory
one. (Being measured; see the latency/waste round.)

### 7c. Inter prediction — the write side is 7a, the read side is different

Not converted: 228 of 267 `PicOffset` params remain, `src/mc.rs` alone holding 59.

The **writes** are structurally identical to intra — same planes, same block shapes, same single
writer — so the same band should serve them. The **reads** are what differ, and they split three
ways:

- **Reference-frame reads at arbitrary motion vectors.** These are a *different allocation*, shared
  and immutable for the duration of the decode. Shared reads need no exclusion at all. This is the
  bulk of inter and it is the easy part — it simply is not an ownership problem.
- **Reads of the current picture's already-reconstructed neighbours** — OBMC (`src/recon.rs:1904`),
  inter-intra. These are the analogue of intra's top-edge problem, and the same question decides
  them: does the read stay inside the current superblock row? If yes, the band covers it. If it
  reaches above, it needs the `ipred_edge` treatment or the band must grow. **This is the open
  question for inter, and it is answerable by reading the neighbour derivation — not a structural
  blocker.**
- **Intra block copy** (`allow_intrabc`) — src and dst are the *same* picture, which is why
  `mc_put_dispatch` carries a `dst.data.ref_eq(src.data)` test and why #482 declines those frames
  outright. Keep declining until someone works it through deliberately.

So inter is best read as **bounded remaining work, not a new design problem** — with one genuine
unknown (OBMC's vertical reach) that a code read settles.

### 7d. The loop filter — a real write-write overlap, and the only case ownership cannot own

This is qualitatively different and it is worth being precise about why, because "add a halo" is
the obvious answer and it is not sufficient.

- Deblocking **writes on both sides of an edge**. Filtering the horizontal edges at the top of
  superblock row N modifies the bottom rows of row N-1. So worker N writes into what worker N-1 is
  writing — a genuine write-write overlap between workers, not a read-only halo.
- The same holds laterally: `src/lf_apply.rs:563`/`:608` explicitly "fix lpf strength at tile
  col/row boundaries", i.e. the filter runs **across** tile edges rather than stopping at them.
- CDEF reads 2 px beyond its block; LR stripes span tiles.

That is exactly why a frame-global deblock barrier existed in the first place, and why removing it
was only sound after the CDEF padding guards were narrowed to the exact window read. **A halo works
only where one side is read-only. Here it is not.**

**And there is a deeper reason, established by building the band and having it refuted (#485).**
The filter's read set is **2-D SPARSE**: the union of the +-reach tap windows around the edges that
actually filter, with rows where nothing filters not read at all. Every per-row band is
**contiguous**, so it necessarily reserves columns and rows nothing reads — and under concurrent
filtering that over-reservation collides with a legitimate narrow write:

```
current:  &    _[163840..163968]   <- the band's 128-px row copy-in
existing: &mut _[163944..163952]   <- a concurrent 8-px write inside it
```

This is §11c's strided-hull defect **transposed**: the hull reserved the gaps BETWEEN ROWS, the band
reserves the gaps BETWEEN EDGES. The hull version was merely slow (its extent hit the wide path);
the band version is a **false positive, i.e. a decode failure**. Third refutation of "cut the guard
count by widening the reservation", and the first where the widening was contiguous.

**Since 2026-08-10 the widening question is MEASURABLE before it is built.**
`--features __probe_bounds` (`docs/BOUNDS_MAP.md`) records, per guard, the
reserved extent, the footprint actually touched, and the distance to every
concurrently-live foreign reservation — separating "does a foreign RESERVATION
intersect" from "does a foreign FOOTPRINT intersect", which is the whole
decision. For this site it says: `LfBlock::fill`'s per-row read guard comes
within **232 bytes** of `cdef_arm.rs:622:9`'s concurrent write (2,217,283
co-live pairs), and a widening of <=256 bytes collides **16** times across 1406
frames of `8-bit/data`. #485's band widened by ~124 bytes and measured 1, 2 and
0 errors on three passes of that group — retrodicted without writing it. The
same table shows the 4K gap vectors under-report the risk by ~1000x, which is
why the band's first full sample passed.

**Two structural facts that fall out, and both are load-bearing for any future attempt:**

- **`tile_threading_active()` cannot gate a filter-side scheme.** That latch is about concurrent
  TILE workers. The filter's concurrency is between **superblock-row filter tasks**
  (`src/thread_task.rs:1030-1043` — the task for `sby+1` is inserted before the selected one runs),
  which exist whenever `n_tc > 1` **however many tiles the frame has**. Measured: gated that way the
  band provably never armed (census byte-identical to main) and 8-bit/data *still* produced 8 errors
  in one run of two. A correct gate would have to mean "no other thread can be filtering this
  picture", which the decoder does not expose.
- **A sparse read set cannot be owned by a contiguous region.** If you want ownership here, the unit
  has to match the sparsity — per edge, or per tap window — not per row or per band. At which point
  you are back to a fine-grained record, which is what we already have.

The options that remain, in the order the evidence supports:

1. **Cut the guard cost rather than the guard.** `LfBlock::fill` (`src/loopfilter.rs:566`) is
   **3,835,042 registrations/frame at t=8 — 33.6% of the whole decoder's**, measured by a doubling
   arm at **3.61 ms/frame of wall, 4.04 ns each**. One site. Coarsening it (per tap row, or per
   fused group, instead of per tap) is sound, local, and does not need any ownership change.
2. **Partition by edge class, not by region.** Filter the edges wholly interior to a band in
   parallel, and the boundary edges in a separate pass. This is a *scheduling* answer to a
   write-write overlap, and it is the only one that gets the filter off coordination entirely.
   Unbuilt, unpriced.
3. **Ownership with a writable halo + merge.** Each worker filters into its own copy including the
   overlap, then a merge resolves boundary pixels. Correct only if the merge is exactly the
   sequential result — for deblocking, whose output depends on filter *order* across the edge, that
   is a strong claim and would need bit-exactness proof, not argument.

## 7e. The whole-plane-guard audit (#479), and why only one of the three sites was a bug

Three places took a guard over the **entire** picture component. The shape is identical; the verdict
is not, and what separates them is only *whether a concurrent writer exists*.

| site | guards | verdict |
|---|---|---|
| `src/safe_simd/filmgrain_arm.rs:1550,1572,1644,1675` (+ the 3 `full_guard` reads) | `full_guard_mut` / `full_guard` | **REAL BUG, FIXED (#479).** Film grain row bands are handed out to N workers by `fetch_add` on `delayed_fg_progress[0]`, so every band collided with every other. 13 of 768 vectors could not decode above `t=1`; narrowed to the `(bh-1)*stride + pw` band. |
| `src/loopfilter.rs:140,182` and `src/looprestoration.rs:212,258` | `full_guard_mut` (+ `full_guard`) | **UNSOUND UNDER CONCURRENCY, NOT A SHIPPING DEFECT.** All four are inside `#[cfg(feature = "__simd_test")]` — the SIMD-vs-scalar differential harness, a dev feature. |
| `src/safe_simd/mc.rs:12153,12196,12271,12299,12839,12901` and `mc_arm.rs:5689,5971,6098,6182` | `full_guard` (**immutable**) on `src` | **SAFE**, for three independent reasons below. |

**Why the `__simd_test` sites are not a bug to fix.** Measured, not argued: built with
`--features bitdepth_8,bitdepth_16,__simd_test` and run at `--threads 8 --group 8-bit/data`, they
produce **313 errors in 358 vectors** —

```
 current: &mut _[163840..163968]   <- a concurrent 128-px row write
existing:    & _[0..983040]        <- the harness's whole-plane save
```

— the same shape as §7d's refuted band, one axis wider. **Narrowing them cannot fix it**: the
harness *semantically* needs the whole plane (it saves it, restores it, runs the scalar reference
over it, then writes the SIMD output back), so under concurrency it would clobber other workers'
pixels even with tracking off. The correct answer is the one already in the tree — keep it
single-threaded: `tests/decode_md5_committed.rs`'s only threaded test is
`#[cfg(not(feature = "__simd_test"))]`, and CI's `__simd_test` step runs only that file (verified
passing). `examples/md5_inventory` now **fails loud** on `--threads > 1` under `__simd_test` rather
than emitting a TSV of errors that reads as a decoder regression.

**Why the `mc` sites are safe.** They are immutable, so they cannot conflict with each other; the
question is only whether a *mutable* borrow of the same allocation can be live. It cannot:

- **`src` is a reference frame, complete before the current frame starts.** Frame threading is the
  only way a reference could still be under construction, and `n_fc` is hard-pinned to 1 without
  `unchecked` (`src/lib.rs:127`) — while `unchecked` builds create picture buffers with
  `dangerously_unchecked`, i.e. no tracking at all.
- **Intra block copy (`src` == the current picture) is refused before the guard is taken.**
  `mc_put_dispatch` bails on `dst.data.ref_eq(src.data)` (`mc_arm.rs:5648`, `mc.rs:12110`, `:12832`)
  and the scalar `PicOffset` path handles it.
- **`mct_prep_dispatch` has no `ref_eq` bail and does not need one.** It writes a scratch `&mut [i16]`,
  not a picture, and its `src` is only ever `&f.refp[..]`: `MaybeTempPixels::Temp` — the variant that
  reaches `mct` — is used exclusively by compound inter and warp (`src/recon.rs:2927`, `:2941`,
  `:3025`, `:3039`), while intrabc uses `NonTemp` (`:2876`, `:2893`) and OBMC writes into `lap`
  (`:1945`, `:1997`).

**The generalisable rule:** a whole-plane guard is not automatically a defect and not automatically
safe. Ask one question — *can any thread hold a mutable borrow of this allocation while I hold
this?* Film grain: yes, N of them. `__simd_test`: yes, and the harness needs it that way, so it is
quarantined instead of narrowed. `mc`'s reference reads: no, and three separate mechanisms enforce
it.

## 8. Picking a model

0. Before proposing ANY extent change, run the bounds map
   (`--features __probe_bounds`, `docs/BOUNDS_MAP.md`) and read the site's
   widening budget. It costs one build and one decode, and it is the only thing
   in the campaign that has priced a coarsening before it was written. Two
   further facts it has already established: at t=8 the shipped decoder's hot
   sites reserve exactly what they touch (`over_ratio = 1.000`, 1-16 bytes), so
   there is no slack to reclaim there; at t=1 the hull paths over-reserve
   153x-1680x and **all** of it is inter-row gap.
1. Can a single consumer own the region for its whole lifetime? -> **owned buffer** (§4). No
   tracker, no `unsafe`, borrowck proves it. Give it its own stride, size it per worker, bound the
   extent by what is actually read, and enforce that bound with a panic.
2. Do regions genuinely overlap across consumers (halos, cross-tile filters)? -> you need
   coordination. Then the record **and** the reference must both be exact (§6), and partial keying
   is not an option (§3).
3. Is a copy involved? -> price it against the coordination it removes, not against zero (§5).
4. Before designing any split: confirm an exclusive reference exists **at the split point** and the
   consumer accepts a non-`'static` lifetime (§1).
