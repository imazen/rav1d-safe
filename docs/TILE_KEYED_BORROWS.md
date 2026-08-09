# Tile-keyed borrows — feasibility, prices, and what is actually reachable

Round of 2026-08-09 for issue #455. Base `main` @ `ee07b00`, branch
`perf/tile-keyed`. **Nothing here is proposed for merge into `default`.** Every
behavioural change is behind a `probe-*` / `__probe_*` feature.

The task was to prototype and price a **tile-keyed** borrow system in three
variants, and — explicitly — not to inherit PR #468's negative, which applied to
one variant only. This document is the feasibility half; the numbers half is in
`benchmarks/tile_keyed_2026-08-09.meta`.

---

## 0. What is being fixed, and why tile identity is the right key

The tracker shards by ADDRESS: `shard_of(offset >> shift)` through a
multiplicative hash. That single choice has to serve two requirements that pull
in opposite directions, which is what the `BLOCK_SHIFT` ladder in
`tracker_shard.rs` is a record of:

* a **fine** shift separates two tile columns on the same picture rows, but
  makes one strided `w x h` access pay a distinct shard line per ROW;
* a **coarse** shift keeps the strided access on one line, but puts every tile
  column of those rows on that same line.

Both requirements are about TILES. AV1 guarantees tile regions do not overlap
during reconstruction — intra prediction, MV prediction and entropy contexts all
reset at tile boundaries — so if a borrow simply says which tile it belongs to,
both properties hold structurally:

* two tile columns are different keys however close their addresses;
* a strided region is ONE key however tall it is.

Measured consequences of not having the key (from #455's own record): +79%
per-registration cost from t=2 to t=8 (contention), and the 15.6 M -> 22.7 M
registration explosion at t>1 (the per-row split that exists only because the
hull would collide with other tile columns).

---

## 1. Feasibility, with file:line

### 1.1 Where tiling becomes known

`Rav1dFrameHeaderTiling` (`include/dav1d/headers.rs:2558`) carries
`cols`, `rows`, `col_start_sb[]` and `row_start_sb[]` — exact tile boundaries in
superblocks, parsed from the frame header before any reconstruction. Tile
regions are therefore known statically, and because boundaries are superblock
multiples (64 or 128 pixels) they are exact BYTE boundaries at every bit depth.
Two tiles never share a byte during reconstruction.

`rav1d_decode_frame_init` (`src/decode.rs:4355`) already computes
`f.frame_thread.tile_start_off[tile_idx]` per tile at `:4406`.

### 1.2 Where a tile task could receive something

Tasks are per **(tile, superblock row)**, dispatched at
`src/thread_task.rs:1236` (`TaskType::TileEntropy | TaskType::TileReconstruction`),
which sets `tc.ts = tile_idx` and calls into
`rav1d_decode_tile_sbrow` (`src/decode.rs:4118`) with
`t: &mut Rav1dTaskContext` and `f: &Rav1dFrameData`. **`t.ts` is the tile
index and it is in scope for the whole of reconstruction.** That is the key's
source; nothing needs to be plumbed to obtain it.

### 1.3 The picture-plane seam is three lines wide

Every picture reference reconstruction makes is derived from exactly three
acquisitions of `&f.cur.data.as_ref().unwrap().data`:

| site | function |
|---|---|
| `src/recon.rs:1758` | `mc` (motion compensation — takes `cur_data` only for a `ref_eq` identity test at `:1775`; builds no picture reference) |
| `src/recon.rs:2137` | `rav1d_recon_b_intra` |
| `src/recon.rs:2827` | `rav1d_recon_b_inter` |

plus `src/recon.rs:3867` in `rav1d_backup_ipred_edge`. From those, 22 call sites
build a `PicOffset` via `.with_offset::<BD>()`. **That is the entire surface**,
and it is why this round could key reconstruction in one commit.

### 1.4 The premise, verified in source rather than cited

Keying reconstruction is only sound if reconstruction for tile `T` touches no
byte outside `T`'s pixel rectangle. Checked on this tree, not taken from the
earlier round's summary:

* `src/recon.rs:2269-2273` passes `have_left = t.b.x > ts.tiling.col_start`,
  `have_top = t.b.y > ts.tiling.row_start`, and — this is the part worth
  checking — **`ts.tiling.col_end` and `ts.tiling.row_end` as the `w`/`h`
  clamps.** Same at `:3160-3163` and, subsampled, at `:3449-3452`.
* Given those, every edge read in `rav1d_prepare_intra_edges` is inside the
  tile: the top-right at `src/ipred_prepare.rs:229`
  (`px_have = min(8 * tw, 4 * (w - x))`), the left column at `:248`
  (`min(sz, (h - y) << 2)`), and the bottom-left at `:288-293`
  (`have_bottomleft` requires `y + th < h`, then
  `px_have = min(sz, (h - y - th) << 2)`).
* Prediction kernels themselves are pure block writers — `ipred_prepare`
  materialises the neighbour edge into scratch first — so nothing else reaches
  out.
* `rav1d_backup_ipred_edge` (`src/recon.rs:3866`) reads
  `[ts.tiling.col_start, ts.tiling.col_end)` of the sbrow's last row: the
  tile's own columns, inside the tile's own rows.

**A stale doc comment to fix while you are there:** `src/ipred_prepare.rs:127-131`
says "the vertical tile start is assumed to be `0`, and `h` is the vertical
image end". Every caller in `src/recon.rs` passes `ts.tiling.row_end`. The
comment describes a weaker contract than the code has, and this design depends
on the stronger one.

### 1.5 The filter chain genuinely crosses tiles

`src/lf_apply.rs:563` ("fix lpf strength at tile col boundaries") and `:608`
("fix lpf strength at tile row boundaries") adjust filter LEVELS at boundaries —
they are `lflvl` mask writes, not pixel writes. What they establish is that the
deblock filter runs ACROSS the boundary using both sides, which is the
structural reason reconstruction's tile-disjointness does not extend to the
filter chain. The filter chain's own picture accesses are 6,389,542
registrations/frame at t=8 (#455 site-class census), the largest single site
being `src/loopfilter.rs:566` at 3,835,042.

### 1.6 The key must be an ARGUMENT, not a thread-local — measured

The obvious zero-plumbing channel is a `thread_local!` set once per task. It is
disqualified on this platform:

| channel | ns/call, out-of-line callee, best-of-9 interleaved |
|---|---|
| baseline (call + xor) | 0.7526 |
| **argument** | 0.7521 (**+0.000**) |
| global atomic | 0.7535 (+0.001) |
| **thread-local** | 1.2548 (**+0.502**) |

macOS compiles `TLS.with(..)` to an indirect `blr` into `_tlv_get_addr` plus the
stack frame that call forces (disassembly in the record). At 22,700,725
registrations/frame at t=8 that channel alone costs **11.4 ms/frame** against a
whole tracker that costs 19.7 — it would eat the win before the win existed.
A first micro-benchmark that read the key in a loop measured +0.00 ns and was
WRONG: LLVM hoists `_tlv_get_addr` out of a caller's loop, so the read must be
made in an out-of-line callee to be priced at all.

So the key rides on `WithOffset<T>` (`src/with_offset.rs`), which every picture
reference already is, and is preserved by every offset arithmetic operator.

---

## 2. The three variants, priced

### Variant 1 — separate owned buffer per tile

**Not blocked by the C-ABI objection.** PR #468's negative — planes arrive as a
raw pointer through an allocator callback, sit behind `Arc<Rav1dPictureData>`,
and every task holds a `fc.data.try_read()` shared guard — is about the SHARED
picture. It does not apply to buffers we allocate ourselves. Two further facts
narrow it more: in the default (non-`c-ffi`) build the inner type is not a raw
pointer at all but `PicBuf` (`include/dav1d/picture.rs:406`), and
`Rav1dPictureDataComponent::wrap_buf` (`:560`) already constructs a component
from an owned buffer, which is what a tile buffer would be.

**Priced (`examples/tile_stitch_cost.rs`, real `v4k_8tile` geometry — 3840x2160
4:2:0, 4x2 tiles, 34 sbrows, 12.44 MB/frame, median of 9, interleaved):**

| arm | ms/frame | band |
|---|---|---|
| `memcpy_flat` (contiguous 12.4 MB, lower bound) | 0.170 | [0.168..0.629] |
| `stitch_whole` (all tiles, once at end of frame) | 0.315 | [0.296..0.800] |
| **`stitch_sbrow`** (34 sbrow bands x tiles — the shape a decoder needs, because the filter chain consumes sbrow N) | **0.315** | [0.304..0.351] |
| `write_into_pic` (recon writes, shared plane, picture stride) | 2.553 | [2.434..2.612] |
| `write_into_tile` (same writes, private compact tile buffer) | 2.567 | [2.400..2.576] |

Two results:

* **The stitch is affordable: 0.315 ms/frame**, i.e. 1.6% of the 19.7 ms/frame
  the tracker costs at t=8, and it does not get worse when done per-sbrow.
* **The cache-locality bonus does not exist.** The brief's hypothesis that
  "recon may get FASTER from cache locality" measures **null**: 2.567 vs 2.553
  ms with overlapping bands. Writing into a smaller contiguous tile buffer is
  not cheaper than writing into the plane. Variant 1 is therefore
  `write_into_tile + stitch` against `write_into_pic` alone — strictly +0.315
  ms of extra work, bought back only by whatever tracker cost it removes.

**What blocks it is coordinates, not ownership.** Downstream offset arithmetic
is in FRAME coordinates: `4 * (t.b.y * y_dst.pixel_stride::<BD>() + t.b.x)` at
`src/recon.rs:2172` and 21 sibling sites, plus ~10 reads of `f.cur.stride[..]`.
A tile buffer with its own tight stride needs `x - tile_x0`, `y - tile_y0` and a
different stride at every one of them. The translation-free alternative — a
full-width buffer per tile, same stride — costs 4x memory for the tile columns
of a tile row (~50 MB at 4K for 4x2 tiles). Neither is a one-commit change, and
**the honest ceiling for the version that keeps the current API is Variant 2's**:
a per-tile owned buffer wrapped in its own `DisjointMut` is still tracked, just
privately and uncontended.

**Not built. Not measured beyond the stitch.**

### Variant 2 — one lock per tile index — REACHABLE, BUILT, AND A MEASURED NEGATIVE

**Reachable and built** (`probe-tilekey-shard`). A borrow that names a tile goes
to that tile's shard: no shift load, no two divisions, no multiplicative hash,
and no `b0 != b1` escalation however tall the region is (`tracker_shard.rs`,
`add_at_shard`). References are untouched — per-row as today — so this arm has
no aliasing question, only the keyed-vs-unkeyed one in §4.

**And on its own it does not pay.** `tkshard/base`, idle box, `foreign = 0` on
every row, dav1d in the same sweep:

| cell | tkshard/base | bands |
|---|---|---|
| 8bpc t=1 | 1.0285 | disjoint (WORSE) |
| 8bpc t=2 | 1.0338 | disjoint (WORSE) |
| 8bpc t=4 | 0.9964 | OVERLAP |
| 8bpc t=8 | 0.9712 | OVERLAP |
| 10bpc t=1/2/4 | 1.0722 / 1.0687 / 1.0490 | disjoint (WORSE) |
| 10bpc t=8 | 1.0228 | OVERLAP |

**Inter-tile lock contention is not where the money is.** With 128 shards the
multiplicative hash already separates concurrent tile columns well enough that
naming the tile explicitly buys nothing measurable, while the key's own store
and compare cost 2-7% at the cells where there is nothing to win. This is the
`TinyLock`-backoff shape a third time: a plausible contention fix that the
contention does not justify.

The key's value turns out to be as an **enabler**, not as a shard selector — see
Variant 3.

### Variant 3 — precomputed per-tile row refs — SUBSUMED, AND THIS IS THE WIN

The brief's framing is "a per-access borrow is an index into a fixed array
rather than interval arithmetic + hashing + locking". With the key present the
stronger move is available: stop taking `h` per-row borrows at all and take the
strided HULL as ONE registration (`probe-tilekey-hull`) — which also deletes the
per-row compact copy and its write-back, the two memcpys per block that exist
only because per-row guards were the only sound option (#460's unpriced item 3).

`tkboth` (hull + key) against base, same sweep:

| cell | tkboth/base | bands | base/dav1d | tkboth/dav1d |
|---|---|---|---|---|
| 8bpc t=1 | 1.0203 | OVERLAP | 1.303 | 1.329 |
| 8bpc t=2 | **0.8868** | disjoint | 1.580 | **1.401** |
| 8bpc t=4 | **0.8490** | disjoint | 1.631 | **1.385** |
| 8bpc t=8 | **0.8287** | disjoint | 1.860 | **1.541** |
| 10bpc t=1 | 1.0785 | disjoint (worse) | 1.453 | 1.567 |
| 10bpc t=2 | 0.9939 | disjoint | 1.649 | 1.639 |
| 10bpc t=4 | 0.9693 | disjoint | 1.681 | 1.629 |
| 10bpc t=8 | 0.9285 | disjoint | 1.900 | 1.764 |

Since `tkshard` alone is null-to-negative, essentially the whole 8bpc t>1 win is
the hull, and the key is only what makes the hull legal. The count and copy
halves are **not separated** — the hull removes both at once and no arm here
isolates them.

**It does not beat #469.** That branch's own verification round reports
1.309 / 1.332 / 1.323 / 1.477 at 8bpc t=1/2/4/8 against a comparable base;
`tkboth` is 1.329 / 1.401 / 1.385 / 1.541. #469's address-derived column band
wins at every cell. Quoted from its record, not re-measured here. So the honest
summary of this round is that tile identity is a CLEANER key than a column band
derived by division — exact, no Lemire magic, no per-`alloc` column stores — and
on this vector it is also a SLOWER one.

**The hull is not legal without the key, demonstrated rather than argued.** The
hull-only arm (`probe-tilekey-hull` alone, address-hashed shards) **fails to
decode** `v4k_8tile` at t=8:

```
overlapping DisjointMut:
 current: &mut _[960..58576] at include/dav1d/picture.rs:804
existing: &mut _[0..57616]
```

Two 16x16 block hulls at pixel columns 0 and 960 — the tile-column boundary at
3840/4. They share no byte; their hulls overlap because a hull spans the
inter-row gaps, which belong to the next tile column. This is #467's documented
false positive, and adding the key makes the same build decode correctly with
the reference md5.

---

## 3. Liveness and correctness, measured

`probe-tilekey-count` (counters only, never composed with a timed run — #460
recorded what an instrument on this path does to wall clock; this build runs
~20x slow and its ms are meaningless):

| arm | registrations/frame, v4k_8tile 8bpc t=8 | keyed | unkeyed sharded | single-shard |
|---|---|---|---|---|
| base | 26,484,179 | 0 | 20,713,817 | 5,770,361 |
| tile-keyed shard only | 26,484,179 | 13,212,787 | 7,501,030 | 5,770,361 |
| **shard + hull** | **15,540,586 (-41.3%)** | **2,269,194 (-82.8%)** | 7,501,030 | 5,770,361 |

At t=1 `keyed = 0` and the totals are identical to base: `SHARDS_SERIAL == 1`
makes every instance `mask == 0`, the key is never consulted, and the t=1 cell
is untouched **by construction**, not by luck.

Reconstruction is **63.8% of the sharded registrations** at t=8. The remaining
36.2% is the filter chain, and it is unkeyed — see §4.

Output identity: md5 `a00c11f454328023c58af14d55544cff` across
base / shard / hull / shard+hull / untracked x t = 1, 2, 4, 8 — 20 of 20.

### 3.1 The gap sweep, and its n

`benchmarks/tile_keyed_gap_2026-08-09.tsv`, `..._bands_2026-08-09.txt`. Idle
box, strict gate, **`foreign = 0` on every committed row**, dav1d 1.5.4
`--framedelay 1` in the same interleaved sweep with rotating arm order,
two-point wall fit at 2 and 20 frames.

**n = 4 complete rounds at 8bpc and n = 3 at 10bpc, not the 7 this campaign's
bar asks for** — 168 rows, 21 cells discarded, foreign-load column summing to 0.
The reason is coordination, not instrumentation: another agent's test suite
owned the box for most of the window, and when it cleared that agent started its
own TIMED sweep. Both were then polling for an idle box and each was making the
other's cells dirty, so I stopped mine — every discarded attempt is ~90 s of
decoding loaded onto their measurement. The remaining rounds were given up
deliberately. Every cell called "disjoint" below is
disjoint at that n; nothing here should be quoted as a settled three-digit
figure, and the two OVERLAP cells at 8bpc t=1 are exactly the ones a larger n
would be needed to call.

**The carrier is not free at 10bpc.** `base` (this branch, default features:
the wider `WithOffset` plus the recon key stores) against pristine `main`:

| cell | base/main | bands |
|---|---|---|
| 8bpc t=1/2/4/8 | 1.0070 / 0.9992 / 1.0042 / 0.9901 | all OVERLAP |
| 10bpc t=1 | 1.0084 | OVERLAP |
| **10bpc t=2** | **1.0164** | **disjoint** |
| **10bpc t=4** | **1.0207** | **disjoint** |
| 10bpc t=8 | 1.0084 | OVERLAP |

So the inert key costs nothing measurable at 8bpc and **1.6-2.1% at 10bpc t=2
and t=4**, where two bytes per pixel make `WithOffset` 24 bytes instead of 16 in
a `Copy` struct passed by value through every kernel. That is a real cost of the
CHANNEL, before any tracker change, and it is part of why 10bpc barely moves.

---

## 4. Why these arms are UNSOUND, stated precisely

Soundness needs every pair of overlapping borrows to meet in some shard. A keyed
borrow lands in its tile's shard; an unkeyed one lands in an address-hashed
shard. **They are different partitions of the same array and never meet.** A
genuine overlap between reconstruction and the filter chain would be missed.

This is not an oversight in the implementation, it is the design's open edge, and
`crates/rav1d-disjoint-mut/tests/tile_key.rs` asserts it EXECUTABLY
(`keyed_versus_unkeyed_overlap_is_missed_and_that_is_the_open_edge`) so that the
day it starts detecting, the gate fails and says why.

A second-order consequence worth naming because it points the other way: an
unkeyed borrow whose address hash happens to land on shard `k` DOES get compared
against tile `k`'s records. That is a correct comparison, but under
`probe-tilekey-hull` the tile-`k` record may be a HULL, so the pair can produce a
FALSE POSITIVE — a spurious overlap panic — where today it would not. It does not
fire on `v4k_8tile` at either depth (5 arms x 4 thread counts x 2 depths all
decode to the reference md5), but it is a property of the arm, not an absence of
one, and a vector with different geometry could hit it.

Closing it requires **keying the filter chain too**, and the reason that is a
separate change rather than a bigger `sed` is that the filter chain legitimately
straddles: a deblock at a tile-column boundary touches both sides, so those
borrows must stay `TILE_ANY`. Making `TILE_ANY` meet keyed borrows costs either
a scan of every shard or the wide path per unkeyed borrow, and #469 measured
what volume does to the wide path (188,307 wide registrations/frame at t=8 made
the decoder 2.28x SLOWER than base). So the sound design is:

1. key the filter chain from the superblock column it is working on;
2. leave genuinely straddling accesses `TILE_ANY`;
3. route the residual `TILE_ANY` population — which must then be small and
   rarely LIVE — through the existing wide/`state` mechanism.

**Scoped, with file:line, so the next round does not have to find it again.**
The filter chain builds its picture references in per-superblock-column loops
that already have the column in scope, so the key is a lookup and not a
division:

| site | loop | offset built |
|---|---|---|
| `src/lf_apply.rs:656` | `for x in 0..f.sb128w` | `py + x * 128` |
| `src/lf_apply.rs:677` | `for x in 0..f.sb128w` | `pu/pv + x * (128 >> ss_hor)` |
| `src/lf_apply.rs:695` | `rav1d_loopfilter_sbrow_rows`, same shape | |
| `src/cdef_apply.rs:214` | `for sbx in 0..sb64w` | 14 `WithOffset` literals below it |

`frame_hdr.tiling.col_start_sb[]` (`include/dav1d/headers.rs:2568`) turns a
superblock column into a tile column with one array read. Two cautions the next
round must handle rather than discover:

* `filter_plane_cols_*` filters the LEFT edge of column `x`, and the deblock
  taps reach into column `x - 1`. At a tile-column start that is the PREVIOUS
  tile, so those iterations — roughly 3 of 30 at 4K with 4 tile columns — must
  stay `TILE_ANY`. `have_left` is cleared only at `x == 0`, not at tile
  boundaries, which is the source-level confirmation that deblocking crosses
  tile columns here.
* `rav1d_loopfilter_sbrow_rows` filters HORIZONTAL edges using the row above,
  which at a tile-row start belongs to the previous tile ROW — same rule.

Also unresolved, and independent of the tracker: **the hull hands out a `&mut`
over the hull.** Two tile columns' hulls on the same rows are two live
overlapping `&mut`, which is UB by Rust's aliasing model whatever the tracker
records — the same defect PR #470 found in #469, from the other direction (there
the record was narrower than the reference; here the reference is wider than the
bytes touched). The sound form needs the per-row-view guard #470 named. **The
hull arm's wall-clock number is therefore a CEILING from a build with UB in it,
not a candidate.** The shard arm has no such problem: it keeps today's per-row
references and changes only which shard the record goes in.

---

## 5. Gates run

* `tests/tile_key.rs` — 6 mechanism tests, green both with and without the
  feature. Teeth proved by 3 mutations, each caught (3 of 6 tests fail on each),
  each restored by sha256 **and** `git diff --exit-code`:
  key as a global off-switch; keyed twins dropping the key; shard index ignoring
  the key.
* Both mandatory pre-existing hazards re-planted under `--features __probe_wide`:
  `add`'s in-lock `state` re-read deleted -> `wide_exclusion` FAILS; `active()`
  cut to one shard -> FAILS. Restored byte-exact, green again.
* `#![forbid(unsafe_code)]` proved ACTIVELY: a planted `unsafe` block gives
  `error: usage of an unsafe block` against `#![forbid(unsafe_code)]` at
  `src/lib.rs:1`. Restored.
* Corpus and the timed sweep: see `benchmarks/tile_keyed_2026-08-09.meta`.

---

## 6. Not measured — say it before the wins

* **Variant 1 beyond its stitch.** No tile-buffer decoder was built; the only
  Variant 1 numbers here are the copy and the recon-write locality null.
* **x86_64.** Nothing in this round ran there, and the key channel argument is
  platform-specific: the +0.502 ns thread-local penalty is a macOS `_tlv_get_addr`
  property and does NOT transfer to Linux/ELF initial-exec TLS. On x86_64 a
  thread-local key may well be free, which would change the design's shape.
* **Any vector with loop restoration live.** Unchanged structural blindness
  (#455 item 4): both 4K gap vectors are intra-only with LR off, so
  `looprestoration*.rs` and `mc_arm.rs` register zero borrows in this grid while
  LR is active in 696 of 768 corpus vectors. The filter class — the part that is
  NOT keyed — is understated here relative to a corpus vector.
* **Miri.** Not run on the hull arm; §4 says on paper why it would fail.
* **t=16**, every vector but `v4k_8tile{,_10b}`, and `--features unchecked`.
