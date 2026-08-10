# The shard-granularity rung across picture size, and the rule that replaces it

**Status: the size axis is measured — 17 multi-tile cells, all bit-identical to
dav1d before timing — and it says the shipped rule is the wrong SHAPE, not the
wrong constant. A derived rule keyed on ROWS PER BLOCK is built (`bps-rows`,
default OFF) and measured on the same grid; it is the only arm with no
regression on any cell. Nothing is flipped on.**

Prior art, not re-derived: `docs/SHARD_GRANULARITY.md` +
`benchmarks/shard_granularity_2026-08-10.meta` (PR #500), which measured
`bps-half` at **0.851x wall / 0.904x CPU** on 1024x576 8-tile t=8 and a null at
4K, and refused to flip a default on two sizes. This round is the sizes.

Record: `benchmarks/shard_size_sweep_2026-08-10.{meta,txt}` + the seven TSVs
beside them.

---

## 1. What is NOT covered, first

* **8-bit 4:2:0 only.** No 10-bit twin, no 4:4:4. The prior round's load-tagged
  grid found 10-bit at 1024x576 tracks 8-bit (0.861x vs 0.851x); nothing here
  re-checks it, and the derived rule's bit-depth invariance is argued from the
  allocator's arithmetic (`len` and `stride` both double), **not measured**.
* **One box** — Apple M4 Pro, 8P+4E, aarch64, macOS. No x86_64, where every lock
  acquisition is dearer; the sign should carry and the magnitude should not.
* **One content class** (photo), one quality point, all-intra, one key-frame OBU
  re-decoded. No inter prediction.
* **Loop restoration executes ZERO blocks on all 17 cells** (§6). Nothing here
  measures LR, exactly as with the two 4K gap vectors.
* **8 tiles (4x2) only, t=8 only** apart from the t=1 noise calibration. Tile
  count and thread count are not axes here.
* **One anomaly is unexplained** (§5, the 512x576 row): the derived rule misses a
  7% win that two fixed rungs get there, and the mechanism is not established.
* The derived rule was **fitted on this grid** (`ROWS_PER_BLOCK_MIN = 4` beat 8
  on a replay of the grid's own arms) and has **no held-out size**.

## 2. The grid, and why it is new crops rather than the existing ladder

`scripts/perf/mk_size_ladder.sh` forces **one tile** on purpose, and the rung
cannot arm on a single-tile frame at all (`tile_concurrency() < 2` returns the
constant). So the ladder cannot answer this question. 17 vectors were built with
the same encoder, speed and quality plus `--tilecolslog2 2 --tilerowslog2 1`:
`scripts/perf/mk_shardsize_vectors.sh`.

They are **centred 1:1 crops** of the same 4K photo, not downscales, because a
downscale confounds geometry with detail-per-pixel and two of the three axes
deliberately break 16:9. Consequence, so it is not mistaken for a bug: bytes per
pixel and absolute ms/frame are **not** comparable with `benchmarks/size_sweep_*`.

| axis | cells | what it isolates |
|---|---|---|
| **H** — width pinned at 1024 | h = 192, 288, 384, 576, 768, 1024, 1440, 2048, 2160 | the height dependence `SHARD_GRANULARITY.md` §2 predicts |
| **W** — height pinned at 576 | w = 512, 1024, 2048, 3840 | the control |
| **D** — 16:9 | 512x288, 1024x576, 2048x1152, 3840x2160 | what real content looks like |
| **discriminating** | 256x2048, 3840x256 | where a global rung and a derived rule must disagree |

Tile geometry is **parsed out of each bitstream**
(`scripts/perf/av1_tile_info.py`): 4 x 2 = 8 tiles on all 17. libaom clamps a
tile request against the superblock count and the encoder log records the
*request*, so an unnoticed clamp would be a silently VOID cell. The parser is
validated against the known `L1024x576_420_8b__t8` (4x2, 16x9 sb),
`L1024x576_420_8b` (1x1) and `L3840x2160_420_8b__t8` (4x2, 60x34 sb) before it is
trusted on anything new.

**All 17 are bit-identical to dav1d 1.5.4 at t=1 AND at t=8 before any timing**
(`scripts/perf/shardsize_verify.sh`), and every arm's own checksum matches that
md5 on five spot cells including both discriminating ones.

## 3. The geometry is closed-form, and the tracker confirms it

`Rav1dPicAllocator::alloc_picture_data`, for luma:

```
    stride = (w + 127 & !127) << hbd,   + 64 when that is a multiple of 1024
    len    = stride * (h + 127 & !127)
    shift  = ilog2(len / 256)                  (the shipped rule; TARGET_BLOCKS = 256)
    rows per block = 2^shift / stride
```

Every shift this predicts for all 17 cells matches the shift the tracker reports
through `--features __probe_bounds`. So **rows per block** is computable without
measuring, and it is the axis everything below is plotted against. Two
consequences a block-COUNT rule cannot see:

* **It is a staircase, not a curve.** 1024x288 and 1024x384 have *identical*
  plane geometry (both round to `aligned_h = 384`); h = 2048 and h = 2160 both
  land on shift 13. Height moves rows-per-block in power-of-two steps.
* **When the stride divides the block, a within-row access can never cross a
  block boundary.** w = 512 gives stride 512, which divides every power-of-two
  block from 512 up — and both 512-wide cells register **exactly zero**
  multi-shard borrows at every rung. w = 1024 gives stride 1088, which divides
  nothing, and ~0.3% of all registrations straddle.

**`len` alone cannot express the rule, and that is a proof, not an opinion:** the
1024x1024 plane is 1.11 MB and wants one shift, the 2048x576 plane is 1.35 MB and
wants two, the 1024x2048 plane is 2.23 MB and wants none. The wanted coarsening
is not monotone in `len`, so no function of `len` can produce it.

## 4. Counted first, without a clock

`probe-wide` (shipped counters) and `__probe_bounds` (strided-2D geometry), t=8,
per frame. Full table: `benchmarks/shard_size_sweep_counts_2026-08-10.tsv`.

| cell | rows/blk | s\|b | lf `row_shards_max` | cdef `pct_row_wide` | `multi`/frame | `w_shards`/frame |
|---|---|---|---|---|---|---|
| 3840x256 | 0.53 | no | 20 | 100.00% | 3,634 | **288** |
| 1024x192 | 0.94 | no | 32 | 100.00% | 779 | 0 |
| 1024x288 | 0.94 | no | 32 | 100.00% | 1,165 | 0 |
| 1024x384 | 0.94 | no | 32 | 100.00% | 1,580 | 0 |
| 512x288 | 1.00 | **yes** | 16 | 100.00% | **0** | 0 |
| 1024x576 | 1.88 | no | 16 | 72.40% | 1,747 | **153** |
| 1024x768 | 1.88 | no | 16 | 72.75% | 2,423 | **204** |
| 2048x576 | 1.94 | no | 17 | 61.53% | 2,898 | **158** |
| 512x576 | 2.00 | **yes** | 16 | 0.00% | **0** | 0 |
| 3840x576 | 2.13 | no | 16 | 27.53% | 3,178 | **140** |
| 1024x1024 | 3.76 | no | 8 | 0.00% | 2,145 | 0 |
| 1024x1440 | 3.76 | no | 8 | 0.00% | 3,148 | 0 |
| 2048x1152 | 3.88 | no | 9 | 0.00% | 3,723 | 0 |
| 3840x2160 | 4.27 | no | 9 | 0.00% | 6,726 | 0 |
| 1024x2048 | 7.53 | no | 4 | 0.00% | 2,492 | 0 |
| 1024x2160 | 7.53 | no | 4 | 0.00% | 2,445 | 0 |
| 256x2048 | 8.00 | **yes** | 4 | 0.00% | 1,684 | 0 |

* `pct_row_wide` at the four CDEF sites is **100% below 1 row per block, 61-73%
  between 1.9 and 2.1, and 0.00% from 3.8 up** — the counterfactual cliff the
  prior record found at 1024x576 is a cliff in rows-per-block, and it is crossed
  inside this grid.
* The SHIPPED all-shards promotions are non-zero on **five cells only**, all with
  0.53-2.13 rows per block. Below ~2 the hull is short enough to stay narrow;
  above ~2.2 it is wide enough to fit under the cap. A band, not a threshold —
  and one more reason the wide path is not where the money is (prior doc §5e).
* One shift coarser takes `row_shards_max` at the CDEF sites to 1-2 and
  `pct_row_wide` to 0.00% at every cell, exactly as at 1024x576.
* **The wide path and the wall disagree, again.** At 3840x256 the derived rule
  ends up with *more* all-shards promotions than `bps-half` (216 vs 54) and fewer
  ordinary multi-shard ones (1,066 vs 1,175) — and it is the FASTER of the two
  (0.756 vs 0.775). Independent confirmation of `SHARD_GRANULARITY.md` §5e from a
  new direction: the money is the ordinary multi-shard registrations, not the
  rare catastrophic ones.

## 5. Timed: the crossover, and the derived rule

n = 7 rounds, rotating arm order, two-point fit (`total = a + b*frames`, so
process startup cancels), medians with min/max bands, `measlock`, idle box
(`foreign_max = 1`). Frame counts are per cell, scaled by area, never past the
stream's length.

**Noise floor, measured.** At t=1 `block_shift_rule` provably returns the
constant for every rung, so `plain`/`bps1`/`bpshalf`/`bpsq` are the same code
path and the true ratio is exactly 1.000. Over five cells x 7 rounds they read
**0.999-1.012**, so the floor is **~1.2%** and any ratio inside [0.988, 1.012] is
a null. (A first pass at n=2 read 1.136 on 512x288 and 0.755 on 1024x384; at n=7
those cells read 0.998 and 0.755. The "regression" was noise, and is recorded
because it was nearly reported.)

Wall ratio vs the shipped rule; `*` = the two arms' bands are disjoint.

| cell | rows/blk | wall ms | cores | bps1 | bpshalf | bpsq | **bpsrows** | untracked | dav1d |
|---|---|---|---|---|---|---|---|---|---|
| 3840x256 | 0.53 | 5.815 | 4.99 | 1.013 | 0.775\* | 0.760 | **0.756\*** | 0.629 | 0.462 |
| 1024x192 | 0.94 | 1.448 | 3.92 | 0.820 | 0.785\* | 0.776 | **0.775\*** | 0.646 | 0.481 |
| 1024x288 | 0.94 | 1.786 | 4.70 | 0.902 | 0.893\* | 0.899 | **0.897\*** | 0.726 | 0.569 |
| 1024x384 | 0.94 | 2.448 | 4.98 | 0.890 | 0.761\* | 0.755 | **0.755\*** | 0.617 | 0.445 |
| 512x288 | 1.00 | 0.967 | 4.54 | 0.983 | 0.992 | 1.025 | **0.998** | 0.711 | 0.548 |
| 1024x576 | 1.88 | 3.250 | 5.40 | 0.879 | 0.863\* | 0.863 | **0.875\*** | 0.715 | 0.538 |
| 1024x768 | 1.88 | 3.978 | 6.24 | 0.903 | 0.857\* | 0.866 | **0.870\*** | 0.717 | 0.534 |
| 2048x576 | 1.94 | 6.356 | 5.44 | 0.881 | 0.878\* | 0.886 | **0.876\*** | 0.733 | 0.554 |
| 512x576 | 2.00 | 1.675 | 5.35 | 0.927 | 0.930\* | 0.964 | **0.995** | 0.726 | 0.546 |
| 3840x576 | 2.13 | 11.396 | 5.39 | 0.907 | 0.907\* | 0.910 | **0.894\*** | 0.762 | 0.578 |
| 1024x1024 | 3.76 | 4.584 | 6.49 | 0.978 | 0.968 | 1.006 | **0.963** | 0.836 | 0.635 |
| 1024x1440 | 3.76 | 6.597 | 6.46 | 0.987 | 0.987 | 1.006 | **0.977** | 0.851 | 0.657 |
| 2048x1152 | 3.88 | 9.756 | 6.62 | 0.984 | 0.993 | 0.993 | **0.979** | 0.859 | 0.651 |
| 3840x2160 | 4.27 | 30.143 | 6.82 | 1.007 | 0.995 | 0.995 | **0.998** | 0.886 | 0.711 |
| 1024x2048 | 7.53 | 8.902 | 6.80 | 0.996 | 1.004 | 1.031 | **0.993** | 0.857 | 0.654 |
| 1024x2160 | 7.53 | 8.688 | 6.72 | 0.990 | 0.981 | 1.002 | **0.983** | 0.854 | 0.683 |
| 256x2048 | 8.00 | 3.808 | 6.65 | 0.987 | 0.987 | 0.995 | **0.988** | 0.549 | 0.420 |

### 5a. The crossover

Sorted by rows-per-block the table is close to monotone, and the boundary is
sharp:

* **rows/block ≤ 2.13 → a win with disjoint bands, 7 of 9 such cells**, ranging
  0.756x to 0.907x. (The two exceptions are the two cells where the stride
  divides the block; see below.)
* **rows/block ≥ 3.76 → a null on all 7 such cells.** Point estimates 0.963-1.004,
  bands overlapping, most of them inside the 1.2% floor.

So the crossover sits **between 2.1 and 3.8 rows per block**, which for a
1024-wide 8-bit picture is a height between about 700 and 1000, and for 4K is a
height between about 1100 and 2000. That is why the prior round saw 14.6% at
1024x576 and nothing at 4K: those two cells sit on opposite sides of it.

**The stride-divides-block cells behave differently, and only half-predictably.**
Both 512-wide cells register zero multi-shard borrows at every rung, so the
"fewer shard lines per borrow" mechanism has nothing to remove — and 512x288
(1.00 rows/block) is indeed a null where its 1024-wide neighbours at 0.94 win
21%. But **512x576 wins 7% anyway** (0.927x at `bps1`, bands disjoint) with
`multi = 0`. So a second mechanism is live that the multi-shard counter does not
see — most plausibly the size of each worker's shard-line working SET rather than
the lines per borrow — and this grid does not separate the two.

### 5b. The derived rule, and the two discriminating cells

`bps-rows` picks the shift **per tracker instance** from the declared row stride:
coarsen until a block spans `ROWS_PER_BLOCK_MIN = 4` picture rows, never finer
than the shipped rule, never past a `MIN_BLOCKS = 32` floor. Liveness is read off
the tracker, not predicted — measured luma shifts, plain → rows:

| cell | plain | rows | |
|---|---|---|---|
| 1024x576 | 11 | 13 | +2 |
| 1024x2048 | 13 | 13 | **no change** (already 7.5 rows/block) |
| 256x2048 | 11 | 11 | **no change** (already 8 rows/block) |
| 3840x256 | 11 | 14 | **+3** — further than any rung on offer |
| 512x288 | 9 | 11 | +2 |

**3840x256 is the cell that decides the question.** At 0.53 rows per block the
right step is three, and a global rung has to pick one number for every picture:
`bps1` reads **1.013** (the wrong direction), `bps-half` leaves two points on the
table at 0.775, and the derived rule reads **0.756**. The other discriminating
cell, 256x2048, is a weaker result than predicted: the rule correctly leaves it
alone, but a two-shift rung there is *harmless* (0.987) rather than damaging, so
that cell does not convict the rung.

Over all 17 cells:

| arm | geomean wall | geomean CPU | worst cell | best cell | cells > 1.000 |
|---|---|---|---|---|---|
| bps1 | 0.9414 | 0.9645 | 1.013 | 0.820 | 2 |
| bps-half | **0.9113** | 0.9528 | 1.004 | 0.761 | 1 |
| bps-quarter | 0.9207 | 0.9670 | 1.031 | 0.755 | 5 |
| **bps-rows** | **0.9119** | **0.9490** | **0.998** | 0.755 | **0** |
| untracked (ceiling) | 0.7389 | 0.7975 | 0.886 | 0.549 | 0 |
| dav1d --framedelay 1 | 0.5623 | 0.6610 | 0.711 | 0.420 | 0 |

**Read honestly: on this grid the derived rule and a global `bps-half` are tied**
(0.9119 vs 0.9113 geomean, a difference far inside the floor). The derived rule's
case is not that it is faster on average; it is that

1. it is the only arm that **never regresses** — worst cell 0.998 against 1.004
   for `bps-half`, 1.013 for `bps1` and 1.031 for `bps-quarter`;
2. it reaches a step the ladder does not offer where the geometry calls for it
   (3840x256, 0.756 vs 0.775); and
3. it is bounded by construction rather than fitted to the sizes that happened to
   be measured, which is the specific failure `SHARD_GRANULARITY.md` §6 refused
   to commit to.

Its CPU geomean is also the best of the four (0.9490), which matters because CPU
is the half of the deficit that a coarser block is supposed to attack.

**The unexplained row: 512x576.** `bps1` (chroma +1, luma +1) reads 0.927 with
disjoint bands and `bps-half` (+2, +2) reads 0.930, but the derived rule
(chroma +2, luma +1) reads 0.995 — worse than both arms it sits between, with
bands disjoint from `bps1`'s. That is not monotone in coarsening and I could not
account for it; per-plane shift mixing is the only difference. It is the largest
single miss in the arm and it is not noise.

### 5c. Against dav1d, per size

The rung and the rule move the gap most exactly where the gap is worst:

| cell | plain / dav1d | bps-rows / dav1d |
|---|---|---|
| 1024x384 | 2.25x | **1.70x** |
| 3840x256 | 2.17x | **1.64x** |
| 1024x192 | 2.08x | **1.61x** |
| 1024x768 | 1.87x | 1.63x |
| 1024x576 | 1.86x | 1.63x |
| 2048x576 | 1.80x | 1.58x |
| 3840x576 | 1.73x | 1.55x |
| 1024x1024 | 1.57x | 1.52x |
| 2048x1152 | 1.53x | 1.50x |
| 3840x2160 | 1.41x | 1.40x |
| 256x2048 | 2.38x | 2.35x |

The 1024x576 column (1.86x → 1.63x) reproduces the prior record's independently
measured 1.883x → 1.602x on a different vector of the same size, which is the
cross-check that this harness and that one agree.

**256x2048 is the worst cell in the grid against dav1d (2.38x) and no rung
touches it** — 8 rows per block already, `untracked` at 0.549, so 45% of its wall
is tracker cost that granularity cannot reach. That is a target this round
identified and did not address.

## 6. Two facts the stage census corrected

`--features __ablate` counts execution units per stage per frame
(`examples/probe_tracker` prints them as of this branch). Only itx, cdef and
looprestoration have `ablate::note()` call sites; the other six read 0 whatever
ran.

1. **Loop restoration executes ZERO blocks on all 17 cells.** Same structural
   blindness the 4K gap vectors have.
2. **"CDEF executes zero blocks at 512x288" is a property of the DOWNSCALED
   ladder vector, not of the size.** `L512x288_420_8b` (a 7.5x downscale) reads
   `cdef = 0`; the 1:1 crop at the same size reads **63,488 units/frame**. The
   `AGENT_BRIEF` states it as a size fact under "permission is not execution";
   the sharper version is that **execution is not a function of size either — it
   is a function of content, and a downscale ladder confounds the two.**

## 7. Recommendation

**Do not ship a rung.** A global constant is now measured to be the wrong shape:
it must be one step for every picture, and the right step ranges from 0 to 3
across ordinary sizes — `bps1` regresses at 3840x256, `bps-quarter` regresses on
five cells, and `bps-half`'s worst is a null only because this grid happens not
to contain the size that would punish it.

**The data supports a derived rule and one is built.** `bps-rows` is default-off,
byte-unchanged in the default build, bit-identical on the corpus at t=1 and t=8,
and the only arm with no regression across 17 sizes. What it still needs before
it could be a default:

* the 512x576 anomaly explained (§5b) — a rule with an unexplained 7% miss is not
  ready to be everyone's default;
* 10-bit and 4:4:4 cells, and a held-out size not used to fit
  `ROWS_PER_BLOCK_MIN`;
* an x86_64 leg;
* a decision on the second mechanism (§5a): if the shard-line working SET is what
  matters on stride-divides-block pictures, the rule should be keyed on both, and
  `MIN_BLOCKS` is then doing real work rather than guarding an edge.
