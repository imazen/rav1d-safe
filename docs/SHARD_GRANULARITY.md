# Shard granularity: the ladder re-fitted, and the wide path that was believed dead

**Status: two measured findings and one arm that is NOT recommended for the
default yet.** Read §1 and §2 before §5 — the counting results are much stronger
than the timing results, and the timing is what decides a default.

Record: `benchmarks/shard_granularity_2026-08-10.{meta,txt}`. Levers named by
`docs/TILED_SCALING.md` §7 item 1 and `benchmarks/strided_2d_2026-08-10.meta` §4.

## What was asked for, and what the ladder turned out to be

Two levers were specified:

1. Re-fit `BLOCKS_PER_SHARD` **scored on tail concurrency**, because it was
   fitted on whole-frame wall and 32-42% of the t=8 gap is idle cores in the
   post-tile filter tail.
2. Collapse the wide-path promotions, so that the exact strided-2D record — whose
   soundness objection was refuted but whose `pct_row_wide` cost was not — becomes
   viable.

**They are one knob, and that is a derivation, not a measurement.** The second
lever's other proposed shape was "a mapping that keeps a short RUN of consecutive
blocks on one shard". A run mapping hashes `block >> k`; but `block = addr >>
shift`, so `block >> k == addr >> (shift + k)` — the run-grouped index *is* the
block index at a shift `k` coarser. Anything that depends only on an access's
SHARD SET (`pct_row_wide`, the shard lines a strided read touches, the
`MAX_SHARDS_PER_BORROW` promotion door) is therefore already covered by the
granularity ladder, and a run mapping cannot reach a point the ladder does not.
Pinned in `tracker_shard.rs::coarser_blocks_collapse_a_strided_access_onto_fewer_shards`,
which also records honestly that it is a derivation-pin and not a guard on the
hash: two planted `shard_of` mutations left its numbers unmoved.

So one ladder answers both, and the cap raise (`MAX_SHARDS_PER_BORROW 4 -> 5`) is
the only genuinely separate shape.

## 1. The shipped decoder DOES take the all-shards wide path — 153 times a frame

`--features probe-wide`, `L1024x576_420_8b__t8` (1024x576, 4x2 = 8 tiles, 9
superblock rows), t=8, counts per 30-frame run. Timer-free, so no measurement
lock and no wall-clock claim.

| rung | shift vs default | slow | multi | **w_shards** | w_blocks | w_full |
|---|---|---|---|---|---|---|
| bpsq (1/4) | +3 | 12,647 | 12,647 | **0** | 0 | 0 |
| bpshalf (1/2) | +2 | 23,910 | 23,910 | **0** | 0 | 0 |
| bps1 (1/1) | +1 | 42,972 | 42,972 | **0** | 0 | 0 |
| **plain (2/1) — ships** | 0 | 81,399 | 72,336 | **4,590** | 0 | 0 |
| bps4 (4/1) | −1 | 188,167 | 157,110 | **14,580** | 0 | 0 |
| bps8 (8/1) | −2 | 327,250 | 302,610 | **14,580** | 0 | 0 |

4,590 per 30 frames is **153 wide promotions per frame**, every one of them
through the `> MAX_SHARDS_PER_BORROW` door, and each one holds **every** active
shard (128). The tracker's own doc comment says "Wide-path promotions are ZERO at
every one of those, on both bit depths" — that measurement was taken on
`v4k_8tile`, and the 4K cell here does read 0 at the default. **The claim is
false on the small tiled vector, which is exactly the cell where the filter tail
is worst** (34.1% of wall, `TILED_SCALING.md` §4).

Going one step COARSER removes them entirely. Going one step FINER triples them
(14,580) and doubles the multi-shard registrations.

`w_full` is **0 at every rung**, which retires the trap flagged for this
direction: a coarser block funnels more simultaneous borrows onto one shard and
could have traded wide-by-shard-count for wide-by-slot-exhaustion. It does not, at
any rung, on either vector. (That trap is real for `SLOTS 7 -> 3`, which is a
different axis and is not touched here.)

Two controls, both inert exactly where they should be:

* `L1024x576_420_8b` (**single tile**), t=8: every rung reads 18,720 / 18,720 / 0.
  The `tile_concurrency() < 2` gate holds, so no rung arms.
* `L1024x576_420_8b__t8` at **t=1**: every rung reads 0 / 0 / 0. `SHARDS_SERIAL = 1`
  gives mask 0, so every block maps to shard 0 and a strided span registers as one
  narrow interval.

4K, same instrument, per 6-frame run: plain 40,271 multi and **0** wide; bps4
2,748 wide; bps8 9,468. So the default sits just inside the cliff at 4K and
already over it at 1024x576.

## 2. Why: the rule's rows-per-block scales with picture HEIGHT

The adaptive rule targets a fixed BLOCK COUNT (`N_SHARDS * BPS` = 256 at the
default). For a plane of `len = h * stride`:

```
    2^shift  ≈  len / 256  =  h * stride / 256
    rows per block  =  2^shift / stride  ≈  h / 256
```

The ratio's documented purpose — the same rows-per-block at every BIT DEPTH — is
real and holds, because a 10-bit plane doubles `len` and `stride` together so they
cancel. **But it says nothing about picture size, and the accesses that pay one
shard line per row are a fixed number of ROWS**: measured `rows_mean` is 7.16-9.02
at every hot strided site, on both vectors, because they are CDEF tap windows and
superblock-row compacts, not fractions of the picture.

So the same 8-row access spreads over `8 / (h/256)` blocks — about 1 at 4K and
about 4 at 1024x576. Measured `row_shards_mean` (`--features __probe_bounds`,
t=8, the tracker's own `shard_of` at the instance's own shift):

| site | 4K, plain | 1024x576, plain | 1024x576, bpshalf |
|---|---|---|---|
| `safe_simd/cdef_arm.rs:622:9` | 2.586 (max 3) | **4.729 (max 5)** | 1.931 (max 2) |
| `safe_simd/cdef_arm.rs:192:9` | 2.587 (max 3) | **4.732 (max 5)** | 1.931 (max 2) |
| `safe_simd/cdef_arm.rs:1217:9` | 2.639 (max 3) | **4.729 (max 5)** | 1.931 (max 2) |
| `cdef_apply.rs:104:32` | 2.639 (max 3) | **4.728 (max 5)** | 1.930 (max 2) |
| `loopfilter.rs:809:17` | 3.675 (max 9) | **6.669 (max 16)** | 2.282 (max 4) |

4.7 distinct shards against a cap of 4 is the whole mechanism. **This is a defect
in the rule's SHAPE, not in the constant** — the constant is only wrong for the
sizes where `h/256` falls below the row count of a tap window. A rung fixes the
sizes it is fitted for; the principled fix is a target keyed on rows-per-block,
i.e. on the STRIDE, which the tracker is not currently told (§6).

## 3. `pct_row_wide` — the strided-2D record's refuting quantity — goes to zero

`benchmarks/strided_2d_2026-08-10.meta` established that an exact 2-D record is
**sound** (`rect_ovl = 0` over 73 M evaluations, permitting all 34,547 hull
collisions) and refuted it on cost, naming `pct_row_wide` as the refuting
quantity: 0.54%-70.59% of would-be 2-D registrations exceed the cap and take all
128 shards.

Measured across the ladder (`pct_row_wide`, t=8):

| site | 4K plain | 4K bpshalf | 1024 plain | 1024 bps1 | 1024 bpshalf | 1024 bpsq |
|---|---|---|---|---|---|---|
| `cdef_arm.rs:622:9` | 0.00% | 0.00% | **72.90%** | 0.00% | **0.00%** | 0.00% |
| `cdef_arm.rs:192:9` | 0.00% | 0.00% | **73.16%** | 0.00% | **0.00%** | 0.00% |
| `cdef_arm.rs:1217:9` | 0.00% | 0.00% | **72.90%** | 0.00% | **0.00%** | 0.00% |
| `cdef_apply.rs:104:32` | 0.00% | 0.00% | **72.81%** | 0.00% | **0.00%** | 0.00% |
| `loopfilter.rs:809:17` | **20.35%** | 0.00% | **62.31%** | 0.00% | **0.00%** | 0.00% |

**One shift coarser takes every site to 0.00%, including the loop filter's, whose
`row_shards_max` drops from 16 to 4.** So the ladder removes the strided record's
refuting quantity outright.

The raised-cap counterfactual is answered from the same run
(`pct_wide_c5/c8/c16`, new columns; the probe's distinct-shard counter was widened
from 8 to 32 so values above 8 are real rather than saturated):

| site | 1024 plain: cap 4 | cap 5 | cap 8 | cap 16 |
|---|---|---|---|---|
| the four CDEF sites | 72.8-73.2% | **0.00%** | 0.00% | 0.00% |
| `loopfilter.rs:809:17` | 62.31% | 44.71% | 26.65% | **0.00%** |
| 4K `loopfilter.rs:809:17` | 20.35% | 9.99% | 0.13% | 0.00% |

`row_shards_max` is exactly **5** at the four CDEF sites, so cap 5 is not a lucky
guess — it is the distribution's maximum, and `BorrowId` can be repacked to hold
exactly five pairs (§4). The loop filter needs 16, which one word cannot hold at
128 shards.

**What this does and does not unblock.** It removes the `pct_row_wide` objection.
It does NOT remove the MACHINERY objection the same record raises: a 2-D record
needs `(stride, w, rows)` beside `(start, end, mut)`, which breaks the
`size_of::<Shard>() == 128` cache-line invariant at `SLOTS = 7`, and turns each
pair comparison into an integer `%` and `/` by a runtime stride. That is the trade
which measured 1.98x and 2.65x slower in two prior arms. So the strided record is
now blocked on ONE priced question instead of two, and it is not built here.

## 4. The cap raise, and what actually bounds it

`MAX_SHARDS_PER_BORROW` could not simply be raised because `BorrowId` must stay
one register-sized word: it is created and destroyed ~50 million times per 4K
frame and travels inside every guard. At the 12-bit `(slot:3, shard:9)` pair it
shipped with, `2 + 2 + 12*4 = 52` bits fit and a fifth pair does not.

`PAIR_BITS` is now derived as `3 + log2(N_SHARDS)` — exactly what the shard index
needs, 10 bits at the default 128 shards — and the bound is a const assert on the
whole word rather than a bare `MAX_SHARDS_PER_BORROW <= 4`. Five pairs then cost
`2 + 3 + 10*5 = 55` bits. `msb-5` is the arm. Six do not fit at any shard count
this build supports, and reaching six by narrowing the slot field would mean
`SLOTS < 8`, which is the shard-full trap.

**This is a live change in the DEFAULT build** (`PAIR_BITS` 12 -> 10), not only
behind the feature, so it carries the full corpus gate in §5.

## 5. Timing

n = 5 rounds, rotating arm order, idle box (`foreign_max = 1`), two-point fit
(20/200 frames at 1024x576, 2/16 at 4K) so process startup cancels, medians with
the min/max band. Full tables: `benchmarks/shard_granularity_2026-08-10.txt`.

### 1024x576, 8 tiles, t=8 — the hard cell (9 superblock rows on 8 workers)

| arm | wall ms/f | band | CPU ms/f | cores | t1->t8 |
|---|---|---|---|---|---|
| **plain — ships** | **3.283** | [3.256..3.322] | 18.322 | 5.58 | 4.416x |
| bps1 (+1 shift) | 2.833 | [2.817..2.856] | 16.706 | 5.90 | 5.086x |
| **bpshalf (+2)** | **2.794** | [2.767..2.811] | 16.556 | 5.92 | 5.175x |
| bpsq (+3) | 2.794 | [2.761..2.806] | 16.528 | 5.91 | 5.175x |
| bps4 (−1) | 4.072 | [4.050..4.139] | 21.828 | 5.36 | 3.565x |
| bps8 (−2) | 4.328 | [4.317..4.356] | 23.411 | 5.41 | 3.359x |
| untracked (ceiling) | 2.344 | [2.328..2.367] | 14.167 | 6.04 | 5.813x |
| dav1d --framedelay 1 | 1.744 | [1.733..1.756] | 11.294 | 6.47 | 6.271x |

`bpshalf` is **0.851x wall and 0.904x CPU**, and the bands do not overlap
(default min 3.256 against head max 2.811). Against dav1d on the same clock:
**wall 1.883x -> 1.602x, CPU 1.622x -> 1.466x.** It closes **52.1%** of the wall
distance to the tracker-free ceiling.

Three-term decomposition recomputed, ms/frame
(`wall(t) = cpu(1)/t + (cpu(t)-cpu(1))/t + (wall(t) - cpu(t)/t)`):

| arm | wall | ideal | +work | +idle | work% | idle% |
|---|---|---|---|---|---|---|
| plain | 3.283 | 1.815 | +0.475 | +0.993 | 32.4 | 67.6 |
| **bpshalf** | 2.794 | 1.807 | **+0.263** | **+0.725** | 26.6 | 73.4 |
| untracked | 2.344 | 1.703 | +0.068 | +0.574 | 10.6 | 89.4 |
| dav1d_fd1 | 1.744 | 1.367 | +0.044 | +0.333 | 11.8 | 88.2 |

Added work **−44.6%** (52% of the way to the tracker-free arm) and idle cores
**−27.0%** (64% of the way). Both halves of the deficit move, which is what
distinguishes this from a lock-cost trade that merely relocates time.

### 4K, 8 tiles, t=8 — no measurable change

`bpshalf` 29.786 [29.500..30.071] against plain 30.286 [29.929..30.429]: 0.983x
with **overlapping bands**, so it is not a claim. CPU 205.571 vs 206.143 = 0.997x,
null. The finer rungs are unambiguously worse: bps4 32.286 (1.066x), bps8 35.857
(1.184x). So at 4K the coarsening is free, not helpful — consistent with §2, since
4K's rows-per-block is already above the tap-window row count.

### Controls

* **t=1, both vectors, every rung: 14.4-14.5 and 185-188 ms/frame** — inside each
  other's bands. `block_shift_rule` returns the constant when `shards <
  SHARDS_CONCURRENT`, so no rung can arm at one thread and none does.
* **Single-tile at t=8** (from the `probe-wide` grid): identical multi/wide counts
  at every rung. The `tiles < 2` half of the gate holds.
* **Registrations per frame are IDENTICAL at every rung** — 682,489 (1024x576) and
  5,615,688 (4K), to the registration, from the independent `probe-sites` census.
  **The knob changes the COST of a registration, not the count.** That is the exact
  converse of #488, which cut the count 1.971x as predicted and got slower; here
  the count is untouched and the wall falls 14.9%.
* At the default only **0.35%** of registrations touch more than one block
  (2,411 of 682,489 per frame) and **153** of those take all 128 shards. Moving
  that 0.35% is the entire effect, which is what "the cost is a core waiting for a
  line" predicts and an instruction-count model does not.
* One band to distrust: `untracked` at 4K/t=1 reads [103.714..176.214]. That is the
  same single-round wall_hi anomaly the prior record documents and did not drop;
  6 of 7 rounds agree and the median (175.214) is unaffected.

### The two objectives disagree, and the disagreement is the finding

Scored on the tail alone (`probe-tasktime`, `TAIL_CONC`), at **4K/t=8** the `bps4`
rung improves BOTH tail scalars — idle fraction 0.878x, idle core-ms 0.933x, tail
mean occupancy 3.004 -> 4.608 workers — while wall is **1.062x, i.e. 6.2%
slower**. It fills the tail by moving cost into it: `deblock_cols` busy goes
14.680 -> 32.268 ms/frame.

**So tail concurrency is a valid objective only at constant work.** The ladder
changes work, and scored on the tail alone it would have selected a rung that is
6.2% slower at 4K and 24% slower at 1024x576. Scored on both, they agree at the
hard cell: `bpshalf` reads wall 0.857x and tail idle core-ms 0.617x there
(tail fraction 33.3% -> 22.4%, against 17.2% for the tracker-free ceiling).

The re-fit was worth doing and the tail instrument was worth building — but the
answer to "score the ladder on the tail" is **"score it on both, and let wall
decide; the tail says why"**, not "replace the objective".

### 5d. The bit-depth and mid-size confirm grid — LOAD-TAGGED, ratios only

Run under `measlock --load-ok` against this branch's own (niced) correctness
gates, so `foreign` reaches 9 on many rows and **no absolute from this grid is
valid**. Per-arm medians of 5, ratios against `plain` in the same interleave:

| cell | bps1 | **bpshalf** | untracked | dav1d_fd1 | fmax |
|---|---|---|---|---|---|
| 1024x576 **10-bit** t=8, wall | 0.854x | **0.861x** | 0.959x | 0.528x | 9 |
| 1024x576 **10-bit** t=8, CPU | 0.914x | **0.899x** | 1.006x | 0.595x | 9 |
| 2048x1152 8-bit t=8, wall | 1.243x | **0.994x** | 0.926x | 0.611x | 9 |
| 3840x2160 **10-bit** t=8, wall | 1.493x | **1.029x** | 0.893x | 0.692x | 9 |
| **2048x1152 8-bit t=1, wall** | **1.032x** | **1.014x** | 0.944x | 0.737x | 9 |

**The t=1 row is this grid's noise calibration and it is not decoration.** At one
thread `block_shift_rule` returns the constant for every rung, so `plain`, `bps1`
and `bpshalf` are the same code path and their true ratio is exactly 1.000. They
read 1.032x and 1.014x. **This grid's noise floor is therefore up to ~3.2%**, and
every ratio inside that band is a null, not a measurement.

Read against that floor:

* **10-bit 1024x576 confirms the 8-bit result** — 0.861x wall / 0.899x CPU,
  far outside the floor and the same sign and magnitude as the idle-box 8-bit cell
  (0.851x / 0.904x). The bit-depth axis holds.
* **2048x1152 (0.994x) and 4K 10-bit (1.029x) are nulls** — inside the floor.
  Consistent with §2: the defect is a function of picture HEIGHT, so it fades as
  the height rises.
* `bps1`'s 1.243x and 1.493x on those two cells have bands spanning 2x
  (`[10.578..30.289]`, `[35.357..65.000]`) and are load artefacts, not results.

## 6. Recommendation, and what it is NOT backed by

**Recommended: `bps-half` as the new default** — `BPS = (1, 2)`, two shifts
coarser. It is 0.851x wall / 0.904x CPU on the tiled 1024x576 cell with disjoint
bands, neutral at 4K, inert at t=1 and on single-tile frames, and identical in
registration count.

**Not flipped in this branch, and these are the reasons, not modesty:**

* **Two picture sizes.** 1024x576 and 4K (plus 2048x1152 and both 10-bit twins in
  the load-tagged confirm grid, §5c). The sweep discipline this repo works to asks
  for tiny / 256 / 1024 / 4K, and §2 says the error is a function of picture HEIGHT,
  so the sizes are the axis that matters most and are the least covered.
* **8-bit 4:2:0 only in the idle-box grid**, all-intra, 8 tiles (4x2), one box
  (M4 Pro, aarch64, macOS). No x86_64 — where LSE-free atomics make every lock
  acquisition dearer, so the sign should hold and the magnitude should not.
* **A rung is a global constant and §7 argues the right fix is not a constant.**
  Shipping `bps-half` would be fitting a second constant to two cells, which is the
  shape of the thing this round found wrong. See §7.

## 7. The derived follow-up, unbuilt

A rung is a global constant and §2 shows the error is size-dependent, so a rung
that fixes 1024x576 over-coarsens 4K (where the fixed ladder already measured
shift 16 at 75.5 ms/frame against 14's 72.8). The shape the mechanism argues for
is a target keyed on **rows per block** rather than on block count:

```
    shift  such that  2^shift / stride  >=  ROWS_PER_BLOCK_MIN
```

with `ROWS_PER_BLOCK_MIN` at least the largest tap-window row count the filters
use (measured `rows_mean` 7.16-9.02, `rows_max` up to 16 at
`loopfilter.rs:809:17`). `BorrowTracker::new` is handed only `len`, so this needs
the stride plumbed to it — `DisjointMut` is generic over buffers that have no
stride at all, so it would be an optional hint set by the picture-plane
constructor, defaulting to the present rule. Not built, not priced.
