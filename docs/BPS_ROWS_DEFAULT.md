# The derived rows-per-block shard rule, shipped as the default

**Status: FLIPPED. A picture plane's tracker block shift is now derived from its
row stride; the block-COUNT rule it replaced is the `bps-blocks` A/B arm. The
corpus is bit-identical at t=1 and t=8 on the DEFAULT build, and the cells the
campaign has been mis-quoting move 2.09-2.27x -> 1.60-1.71x of dav1d.**

Prior art, not re-derived: `docs/SHARD_GRANULARITY.md` (PR #500, the mechanism)
and `docs/SHARD_SIZE_SWEEP.md` (PR #501, the 17-cell height sweep that built the
rule and measured it default-off). This round is the flip, the numbers for the
SHIPPED build, and the one cell #501 could not explain.

Record: `benchmarks/bps_rows_default_2026-08-11.{meta,txt}` + the TSVs beside it.

---

## 1. Why this had to happen, and what it fixes

#501 left the rule default-off for one stated reason — two picture sizes were
not enough and the error is a function of picture HEIGHT — and then measured the
17-cell height sweep that answers exactly that. The rule matched the best global
constant's geomean (0.9119 vs 0.9113 wall) while being **the only arm that
regressed no cell** (worst 0.998 against 1.004 / 1.013 / 1.031 for the three
rungs), and reached a three-shift step at 3840x256 that no rung on the ladder
offers.

Leaving it off had a cost beyond the lost time. `docs/SHARD_SIZE_SWEEP.md` §5d
tabulates **1.61x / 1.64x / 1.70x** against dav1d for the small-height tiled
cells; those are the GATED ARM's figures. Shipped `main` measured
**2.06x / 2.24x / 2.29x** on the same cells six hours later
(`benchmarks/cost_census_2026-08-10.meta` §1). Both records are internally
honest and the confusion is a foot-gun of the arrangement: the best number in
the file belonged to a build nobody ships. After this change they are the same
build.

## 2. What is NOT covered, first

* **x86_64 is not measured.** Every lock acquisition is dearer there, so the
  sign should carry and the magnitude should not. Reproduced only as a clippy
  target locally.
* **4:4:4, 12-bit, film grain, and inter/video content are not measured.** All
  cells here are 8-bit and 10-bit 4:2:0, all-intra, one key-frame OBU
  re-decoded. 4:4:4 chroma planes are luma-shaped, so the rule treats them like
  a wider picture — argued, not measured.
* **`ROWS_PER_BLOCK_MIN = 4` is still fitted on #501's grid with no held-out
  size.** That weakness is unchanged by shipping it; the ladder is kept
  buildable precisely so the next person can re-fit it.
* **Three of the eight cells do not move** (§5a): `c256x2048`, `v4k8tile` and —
  with a caveat — `c512x576`. Two of those three are *provably* inert (§4a),
  which is what makes them the grid's noise control rather than a
  disappointment.
* **The residual wide path is NOT removed the way a global rung removes it**
  (§6a). At 1024x576 `w_shards` reads 153/frame in BOTH arms, because the derived
  rule only moves buffers that declare a stride and a global rung moves every
  buffer. Priced at about one point of wall by #500 §5e, and not attacked here.
* **`msb-5` is not bundled.** It is ~1% and not separated from zero at n=7.
* **`SLOTS` is not touched**, and the trap that goes with it (a 64-byte shard
  raising the shard-full rate) is re-checked rather than assumed: §6.2.
* **No reservation was widened anywhere**, per the four standing refutations.
* One box (Apple M4 Pro, 8P+4E, macOS, aarch64), one content class (photo), one
  quality point. Loop restoration executes zero blocks on every cell here, the
  same structural blindness the 4K gap vectors have.

## 3. What changed, exactly

```
    stride = (w + 127 & !127) << hbd,   + 64 when that is a multiple of 1024
    len    = stride * (h + 127 & !127)
    base   = ilog2(len / TARGET_BLOCKS)                 # the old rule, TARGET_BLOCKS = 256
    shift  = max(base, min(ceil(log2(ROWS_PER_BLOCK_MIN * stride)),
                           ilog2(len / MIN_BLOCKS)))    # the new one
```

`ROWS_PER_BLOCK_MIN = 4`, `MIN_BLOCKS = 32`, both unchanged from #501. Three
properties are worth restating because they bound the blast radius:

1. **Never finer than the old rule.** The `max` is what makes this a coarsening
   and nothing else, so no buffer gets MORE blocks than it used to.
2. **Only buffers that declare a stride move.** `declare_row_stride` is called
   from `Rav1dPictureDataComponent::from_parts` and nowhere else, so every
   non-picture instance keeps the block-count rule exactly. (That is also why
   the wide path does not go to zero — §6.)
3. **The serial and single-tile gates are the SAME gates**, restated in the rows
   rule rather than inferred from `base`. At `--threads 1`, or on a single-tile
   frame at any thread count, the derived rule provably returns the block-count
   answer, so nothing about a serial decode moves. That is why the whole t=1
   table in §5 is a noise control rather than a claim.

Polarity of the arms is inverted from the two previous rounds:

| build | rule |
|---|---|
| **default** | the derived rows-per-block rule |
| `--features bps-blocks` | the block-count rule that shipped before this commit — **the base arm** |
| `--features bps-1` / `bps-half` / `bps-quarter` / `bps-4` / `bps-8` | the ladder, each pinning its constant and turning the derived rule OFF |
| `--features probe-shiftpin` | `RAV1D_PIN_SHIFT="<stride>:<shift>,…"`, the only instrument that separates a LUMA shift from a CHROMA one |

`bps-rows` is gone. It was the default the moment this landed, and a feature flag
for the default is a flag that can never fail.

## 4. Liveness, read off the tracker

`--features __probe_bounds`, t=8, the `shifts` column is the tracker's own
per-instance shift, not a prediction:

| cell | base luma,chroma | head luma,chroma | `pct_row_wide` (4 CDEF sites) | `row_shards_max` lf |
|---|---|---|---|---|
| c1024x576 | 11, 9 | **13, 11** | 72.0-72.4% -> **0.00%** | 16 -> 4 |
| c512x576 | 10, 8 | **11, 10** | (see §7) | |
| c256x2048 | 11, 9 | 11, 9 | unchanged | unchanged |
| v4k8tile | 14, 13 | 14, 13 | unchanged | unchanged |

The last two rows are the point of §4a.

### 4a. Two cells in this grid are IDENTITY controls, and that is deliberate

At `c256x2048` (8.00 rows per block already) and `v4k8tile` (4.27) the derived
rule computes the SAME shift as the block-count rule, on every plane — verified
off the tracker, not predicted. So on those two cells `plain` and `bps-blocks`
are the same decoder doing the same thing, the true ratio is exactly 1.000, and
whatever they measure is this grid's noise floor.

**Every cell at t=1 is the same kind of control**, for a different reason: the
rule is gated on `shards >= SHARDS_CONCURRENT && tiles >= 2`, so at one thread it
returns the block-count answer for every size. The whole t=1 table is therefore
a floor measurement plus an unchanged gap-to-dav1d, and none of it is a claim
about the change.

A sweep that contains its own controls is the difference between "0.95 is a win"
and "0.95 is a win against a measured floor of X".

## 5. Timed: the SHIPPED default against dav1d

n = 7 rounds (8 run, round 0 discarded — the first touch of each (arm, cell) is
cold), rotating arm order, `measlock`, two-point fit `total = a + b*frames` with
per-cell frame counts never past the stream's length, dav1d 1.5.4
`--framedelay 1` interleaved in the same sweep. `foreign_max = 1`.

**Ratios are PAIRED per round**, because the interleave is what makes that
possible and it matters here: `v4k8tile` reads 47.5 / 50.6 / 48.9 ms/frame
across rounds for ALL FIVE arms including dav1d, and that shared drift cancels
in a paired ratio and does not cancel in a ratio of medians. It is the
difference between a 6% band and a 1% one on that cell.

**Measured twice, on two independently built sets of binaries** (the second
after the last code-touching commit). Both n=7. The tables below are the second;
the first agrees within 1.1% on every cell and is committed beside it.

### 5a. t=8 — the cells the campaign has been mis-quoting

| cell | base ms/f | head ms/f | head/base (paired) | [min..max] | **base/dav1d** | **HEAD/dav1d** |
|---|---|---|---|---|---|---|
| c1024x192 | 1.454 | 1.115 | **0.7656** | [0.7594..0.7698] | 2.093 | **1.601** |
| c1024x384 | 2.470 | 1.844 | **0.7530** | [0.7374..0.7678] | 2.268 | **1.708** |
| c3840x256 | 5.898 | 4.407 | **0.7473** | [0.7287..0.7886] | 2.195 | **1.649** |
| c1024x576 | 3.200 | 2.839 | **0.8889** | [0.8637..0.9022] | 1.825 | **1.616** |
| c1024x576 **10-bit** | 3.444 | 3.100 | **0.9000** | [0.8907..0.9144] | 1.834 | **1.645** |
| c512x576 | 1.658 | 1.608 | 0.9732 | [0.9071..0.9916] | 1.812 | 1.734 |
| c256x2048 | 3.744 | 3.719 | 0.9921 | [0.9552..1.0175] | 2.358 | 2.323 |
| v4k8tile | 49.778 | 49.639 | 1.0000 | [0.9937..1.0093] | 1.389 | 1.395 |

All seven paired rounds are below 1.0 on the first six rows. The last two are
the identity controls of §4a; their spread IS the noise floor, and it is
**±1.0% on v4k8tile and ±3.1% (worst round ±4.5%) on c256x2048**.

CPU per frame, same runs, paired: 0.8748 / 0.8519 / 0.8736 / 0.9261 / 0.9318 on
the five moving cells, all 7/7 rounds below 1.0; `c512x576` 0.9947 (6/7),
`c256x2048` 0.9957 and `v4k8tile` 0.9973 (both controls, spanning 1.0).

**No cell regresses.** The two above 1.000 in point estimate are the provable
identity controls, and they read 0.9921 and 1.0000.

### 5b. t=1 — unchanged by construction, and measured to be

The rule is gated on `shards >= SHARDS_CONCURRENT && tiles >= 2`, so at one
thread every cell keeps the block-count shift. Measured, n=7:

| cell | head/base (paired) | HEAD/dav1d | base/dav1d |
|---|---|---|---|
| c1024x192 | 1.0004 | 1.252 | 1.249 |
| c1024x384 | 1.0004 | 1.258 | 1.260 |
| c3840x256 | 1.0000 | 1.244 | 1.245 |
| c1024x576 | 0.9992 | 1.268 | 1.267 |
| c256x2048 | 0.9996 | 1.321 | 1.321 |
| c512x576 | 0.9972 | 1.285 | 1.285 |
| c1024x576_10b | 0.9989 | 1.337 | 1.339 |
| v4k8tile | 0.9989 | 1.206 | 1.208 |

Every median inside 0.3% of 1.000, every cell spanning it. **This is the whole
t=1 table's job: it is a noise measurement, not a result** — and the
single-thread gap to dav1d is 1.21-1.34x and is not what this change is about.

### 5c. Against the ns-per-registration model

Registrations per frame are **identical between the arms on every cell**
(measured, `--features __probe_bounds`; the knob changes the COST of a
registration, not how many there are). Tracker cost is `arm − untracked` on the
same interleave:

| cell | regs/frame | base ns/reg | head ns/reg | saved |
|---|---|---|---|---|
| c1024x192 | 156,777 | 7.60 | **3.20** | 4.39 |
| c1024x384 | 333,863 | 9.26 | **3.83** | 5.44 |
| c3840x256 | 749,831 | 8.38 | **3.45** | 4.94 |
| c1024x576 | 529,092 | 6.42 | **3.92** | 2.50 |
| c1024x576_10b | 662,694 | 5.80 | **3.83** | 1.97 |
| c512x576 | 283,821 | 6.50 | 6.15 | 0.35 |
| c256x2048 | 569,690 | 19.40 | 19.16 | 0.23 |
| v4k8tile | 8,929,449 | 3.64 | 3.53 | 0.11 |

Two things fall out. First, the campaign's **4.5-6.4 ns band is again refuted as
a cross-cell constant** — 3.64 to 19.40 here — exactly as the cost census found.
Second, and this is the new part: **on every cell that moves, the rule lands the
mean registration at 3.2-3.9 ns, which is the census's UNCONTENDED (t=1) rate of
2.78-3.89 ns.** On those cells the coarser block removes essentially the whole
contention premium rather than a slice of it.

`c256x2048` stays at 19.16 ns, and that is consistent rather than surprising: it
is the cell the census singled out as contention-bound, where #502 measured a
5.8% registration-count cut as a null on both instruments. A granularity cut is
a null there too. Two different levers, same answer — that cell needs a third.

The `c1024x576` count here is 529,092/frame against the cost census's
566,594. That is not a disagreement: the census ran on `main @ 414515c` and
**#502 has landed since**, removing part of the LEFT-context read family. Use
this branch's counts with this branch's timings.

## 6. The counted half — three questions a clock cannot answer

`--features probe-wide` / `__probe_bounds`, t=8 and t=1, both arms, all eight
cells. Full tables: `benchmarks/bps_rows_default_counts_2026-08-11.tsv`.

1. **The registration COUNT is identical between the arms on every cell** —
   156,777 / 333,863 / 749,831 / 529,092 / 569,690 / 283,821 / 662,694 /
   8,929,449 per frame. This knob changes the cost of a registration, never how
   many there are, and that is what makes §5c's ns/registration division legal.
2. **`w_full` is 0 on every row** — 8 cells x 2 arms x 2 thread counts. A coarser
   block funnels more simultaneous borrows onto one shard and could have traded
   wide-by-shard-count for wide-by-SLOT-EXHAUSTION. It does not, anywhere,
   including on `c3840x256` where the rule coarsens by three shifts. (This is
   the trap that makes `SLOTS 7 -> 3` a separate and untouched axis.)
3. **The multi-shard population collapses where the win is**: `multi`/20 frames
   goes 15,580 -> 1,260 at `c1024x192` and 31,600 -> 2,520 at `c1024x384`
   (12.4x and 12.5x), 72,680 -> 21,320 at `c3840x256`, 34,910 -> 23,582 at
   `c1024x576`, 77,156 -> 38,924 at 10-bit — and is **0 in both arms** at
   `c512x576` and unchanged at both identity controls.

### 6a. A correction to the record: the 153 wide promotions are NOT on a picture plane

`docs/SHARD_GRANULARITY.md` §1 found 153 all-shards promotions per frame at
1024x576 on the shipped build and showed one shift coarser removes them. This
branch coarsens BOTH picture planes by two shifts at that cell and `w_shards`
does not move: **3,060 per 20 frames in the base arm and 3,060 in the head arm**,
while `--features bps-1` — one shift coarser on *every* instance, stride or not —
takes it to **0**.

Coarsening is monotone in how many blocks a span covers, so if +1 everywhere
removes them and +2 on the picture planes does not, they are not on a picture
plane. They are on a buffer with no row stride to declare, which the derived
rule cannot reach by construction. #500 §5e priced the whole wide path at about
one point of wall, so this is a named residual rather than a problem — but the
instance is still unidentified, and a global rung is the only arm that currently
removes it.

### 6b. What `pct_row_wide` does and does not say

The `__probe_bounds` RECT columns are the **strided-2-D counterfactual**, not the
shipped path. On the shipped path at `c1024x576` they line up with the wall
(`pct_row_wide` 72.0-72.4% -> 0.00% at the four CDEF sites, `row_shards_max`
5 -> 2, and 16 -> 4 at `loopfilter.rs:809:17`; at `c1024x192` it is 100% -> 0.00%
with the loop filter's max going **32 -> 4**). At `c512x576` they do not:
`multi` and `w_shards` are **0 in every arm** there, so no registration ever
takes the multi-shard or wide path at all, and the 38.67% the counterfactual
reports describes a record shape this decoder does not use. §7 is what that cell
actually measures.

## 7. The 512x576 anomaly — investigated, and it does not reproduce

`docs/SHARD_SIZE_SWEEP.md` §5b left one cell unexplained: the derived rule read
**0.995** there while `bps1` (chroma +1, luma +1) read 0.927 and `bps-half`
(+2, +2) read 0.930, both with disjoint bands. Worse than both arms it sits
between is not monotone in coarsening, and §8 of that doc made it a blocker.

The rule and the ladder both move luma and chroma **together**, so no
combination of arms can say which plane is responsible. `--features
probe-shiftpin` (`RAV1D_PIN_SHIFT="<stride>:<shift>,…"`) pins each plane
independently, which turns the question into a 3x3 factorial: luma shift
{10, 11, 12} x chroma {8, 9, 10}, where (10, 8) is the block-count rule,
(11, 9) is `bps1`, (12, 10) is `bps-half` and (11, 10) is the derived rule.
(12, 9) is a corner no arm on offer can reach.

n = 7, idle box (`foreign_max = 0`), paired per-round ratios against (10, 8):

| luma \ chroma | 8 | 9 | 10 |
|---|---|---|---|
| **10** | 1.0000 (base) | 0.9625 | 0.9918 |
| **11** | 0.9418 | **0.9213** `bps1` | **0.9282** the derived rule |
| **12** | 0.9410 | 0.9184 | **0.9336** `bps-half` |

Every arm in the L11/L12 rows is 0.918-0.942 — one tight cluster — and **the
derived rule sits inside it, between `bps1` and `bps-half`, exactly where
monotonicity says it should.** The unpinned DEFAULT build measured in the same
sweep reads 0.9244, agreeing with its pinned twin (0.9282) to 0.4% and
cross-checking the instrument.

**So the 0.995 does not reproduce.** Two independent n=7 runs here put the
derived rule at 0.9732 and 0.9509 on this cell in the main sweep and 0.9282 in
the idle-box factorial: a small but real win, never a miss.

Three things the factorial does establish:

* **The planes do not interact.** Additivity residuals (measured / `r(L,8) *
  r(10,C)`) are 1.0086, 0.9909, 1.0192, 0.9943 on wall and 0.9979, 0.9920,
  0.9963, 1.0026 on CPU — within ±2% everywhere. Per-plane shift mixing, the
  only mechanism #501 could name, is **not** a mechanism.
* **Luma saturates at +1.** 10->11 is worth 5.8%; 11->12 adds nothing (0.9418 vs
  0.9410).
* **Chroma is non-monotone, and that is the real residual.** 8->9 is worth 3.75%
  and 9->10 gives most of it back (0.9625 vs 0.9918 at fixed luma; the same
  ordering holds at all three luma shifts). The derived rule takes chroma +2
  because `ROWS_PER_BLOCK_MIN` is ONE number for both planes, while a 4:2:0
  chroma plane's tap window is half the luma one. **Measured cost of the single
  shared constant at this cell: ~0.7% against `bps1`.**

That last point is the honest follow-up this round hands on: the rows target
arguably wants to be per-plane (or keyed on the filter's tap window in that
plane), not one constant. It is worth ~0.7% at the one cell where it has been
isolated, and `__rpb_2` / `__rpb_8` / `__rpb_16` plus `probe-shiftpin` are the
instruments to price it properly.

**What this cell is NOT measuring** (§6b): `multi` and `w_shards` are zero in
every arm, because a 512-byte stride divides every power-of-two block from 512
up. Nothing here is a multi-shard or wide-path effect. What is left is which
shard each access maps to — the shard-line working SET that
`SHARD_SIZE_SWEEP.md` §5a guessed at and could not separate. This factorial does
not model it either; it measures it, one plane at a time.

## 8. Gates

Driver `scripts/perf/bpsrows_gates.sh`, logs `~/tmp/bpsrows/gates`. **The corpus
legs are on the DEFAULT arm because the DEFAULT arm is what changed** — for the
two previous rounds they were a formality on a compile-time arm; here every
picture plane's block boundaries move on the shipped path.

| gate | result |
|---|---|
| `cargo test --lib`, release **and** debug | pass, both (80 passed, 8 ignored) |
| tracker crate: default + `__bps_blocks` + all five `__bps_*` rungs + `__rpb_{2,8,16}` + `__msb_5` + `__msb_5,__bps_blocks` + `__probe_shiftpin` | 13 configurations, all pass (29 tests where the rows rule is live, 28 where a rung disables it) |
| **corpus, DEFAULT arm, t=1**, no `--skip-group` | **766 PASS + 2 SKIP** |
| **corpus, DEFAULT arm, t=8**, no `--skip-group` | **766 PASS + 2 SKIP** |
| corpus, `bps-blocks` (the base arm), t=8 | 766 PASS + 2 SKIP |
| set-diff BY NAME (key `(group, name)`, value `(status, ACTUAL md5)`) vs `benchmarks/aarch64_md5_fixes_2026-08-07_final.tsv.zst` | **CLEAN on all three**: 0 only-in-baseline, 0 only-in-head, 0 differing |
| set-diff t=1 vs t=8, DEFAULT arm | CLEAN |
| loop-filter window `debug_assert`, `-C debug-assertions=on`, `8-bit/data` t=8 | **358 vectors, 0 mismatch, 0 error** |
| `mt_stress` threads 1/2/4/8/16 x 5 trials, 4K | pass (re-run with `test-vectors/bench/photo_4k.avif` present — see below) |
| `multi_decoder_pressure` — 12 concurrent decoders x 3 iters over 5 vectors, mixed thread counts | **PASS**, every md5 equals the serial reference |
| `tile_threading_overlap` (3), `reproduce_overlap` (6), `thread_cleanup_test` (2) | 11/11 pass — **but 9 of the 11 need `-- --ignored`; see below** |
| the 10 sweep vectors vs dav1d 1.5.4 at t=1 **and** t=8, before any timing | 10/10 identical, on the FINAL build |
| the EXACT CI clippy/doc/fmt legs | all rc=0 (below) |

**Two gate-hygiene corrections, because both were being reported as green
while running nothing:**

* **`tile_threading_overlap` and `reproduce_overlap` are `#[ignore]`d in the
  repo** (3 and 6 tests). A plain `cargo test --test tile_threading_overlap
  --test reproduce_overlap --test thread_cleanup_test` reports `ok` having run
  **2 of 11** tests. Run with `-- --ignored`: all 9 pass (1.12 s and 0.71 s).
  The `#[ignore]`s predate this branch and are not touched here; what changes is
  that the gate table now says which invocation was used.
* **`test-vectors/dav1d-test-data` and `test-vectors/bench/photo_4k.avif` are
  not in a fresh worktree.** The first corpus pass in this round exited
  `rc=101, lines=0` on all three legs and the set-diff dutifully reported
  "768 only-in-baseline" — loud, correctly. `mt_stress` `.expect()`s the 4K
  vector. Link/copy both before believing any corpus or stress result.

### 8a. Test teeth, proven by planting

Every mutation was restored from a `~/tmp` backup COPY — never `git checkout --`
— and verified byte-exact by sha256 plus `git diff --exit-code`.

| planted mutation | `rows_rule_targets_…` | `declaring_a_stride_installs_…` |
|---|---|---|
| `ROWS_PER_BLOCK_MIN` 4 -> 1 | **FAILS** | **FAILS** |
| `set_row_stride` reverted to a no-op (the seam) | passes | **FAILS** |
| `ROWS_RULE_ACTIVE` forced `false` | **FAILS** | passes |
| none (control) | ok | ok |

Row 2 is why the new test exists: the rule's own test drives
`block_shift_rule_rows` directly and stays green when the seam is re-gated into
a no-op, which is exactly the change this branch reverses. Row 3 is its
complement and is reported as a **gap, not coverage** — the seam test branches
on `ROWS_RULE_ACTIVE` and asserts the inverse when it is off, so it cannot catch
the flag being wrongly cleared; the rule test can and does.

`forbid(unsafe_code)` is proven ACTIVE, not read: an
`unsafe { core::mem::transmute(x) }` planted in `src/picture.rs` (which has no
module-level forbid of its own) fails the build against **`lib.rs:13:12`** —
the campaign brief's anchor, confirmed for the third round running. Restored,
sha256 `fa02c12b7730dbeba3f2304e366d245dc9eb30e35153a5e7ea7fc6856969d5e3`
before and after, `git diff` clean, lib rebuilt green.

**Standing hazards, replanted** under `--features __probe_wide`,
`crates/rav1d-disjoint-mut/tests/wide_exclusion.rs`:

| plant | result |
|---|---|
| baseline | ok (0.07 s) |
| `4af62ae`'s in-lock `state` re-read deleted from `add_contended` | **FAILED** |
| `active()` cut to one shard | **FAILED** |
| after both restores | ok (0.05 s) |

`crates/rav1d-disjoint-mut/src/tracker_shard.rs` restored byte-exact, sha256
`3d7028e01a30466334ea3cacc2d5cff1e0941d89a841bba0046090d7d0b3a166` before and
after. (That digest is this branch's, not #501's — the file changed; re-derive
the baseline rather than comparing to a committed one.)

### 8b. Clippy — the CI legs pass; `--all-targets` fails on BASE too

The four legs CI actually runs, on this branch, all `rc=0`:

```
cargo clippy --no-default-features --features "bitdepth_8,bitdepth_16" -- -D warnings
cargo clippy -p rav1d-disjoint-mut -- -D warnings
cargo clippy -p rav1d-disjoint-mut --no-default-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc -p rav1d-disjoint-mut --no-deps
cargo fmt --all -- --check
```

`cargo clippy --release --all-targets -- -D warnings` (NOT a CI leg) fails on
both the base commit and this branch: aarch64 **81 errors on each**, x86_64 12
on base and 8 on head — the counts differ only because clippy aborts at the
first failing target and the target order differs. The complete set of files
clippy names, over all four runs, is `benches/tier_isolation.rs`,
`examples/{bench_ivf_limit,itx_shape_census,md5_ablate,profile_ivf}.rs`,
`src/safe_simd/{itx_arm,itx_arm_neon_16x16,mod}.rs` and
`tests/thread_cleanup_test.rs`. **Zero findings in `tracker_shard.rs`,
`crates/rav1d-disjoint-mut/src/lib.rs` or `include/dav1d/picture.rs`**, checked
by grepping all four logs for those paths.

## 9. Where this leaves the campaign, and what is next

The four preconditions `SHARD_SIZE_SWEEP.md` §8 set for making this the default:

| #501 asked for | status |
|---|---|
| the 512x576 anomaly explained | **done, and it does not reproduce** (§7). The planes are separable to ±2%; the derived rule sits between `bps1` and `bps-half`, not above both. Residual named: one shared rows target for two planes, ~0.7% at that cell. |
| a 10-bit leg | **done**: 0.9000 wall / 0.9318 CPU, 7/7 rounds, 1.834x -> 1.645x of dav1d. The bit-depth invariance argued from the allocator's arithmetic is now measured on one cell. |
| a held-out size not used to fit `ROWS_PER_BLOCK_MIN` | **NOT done.** `v4k_8tile` is a different vector at a size the grid contains, and it is an identity cell. The constant is still fitted with no held-out size — `__rpb_{2,8,16}` exists so the next person can re-fit it, and was **not swept here**. |
| an x86_64 leg | **NOT done.** Clippy target only. Every lock acquisition is dearer there, so the sign should carry and the magnitude should not. |

Two of four. The two that are missing both argue that the shipped constant could
be better tuned, not that the SHAPE is wrong — and the shape is what this change
is: a rule that adapts per picture, bounded by construction, never finer than
what shipped, with two in-grid identity controls proving the null cells are
genuinely null.

Ranked follow-ups, with what each is worth where it has been measured:

1. **`c256x2048` — 2.32x of dav1d, the worst cell in the grid, and now refused
   by two levers.** #502 measured a 5.8% registration-count cut as null there;
   this measures a granularity cut as null too (0.9921, and it is an identity
   cell so that is not even an attempt). Its tracker cost is 19.16 ns per
   registration against 3.2-3.9 on every cell the rule fixes — pure `TinyLock`
   contention at 8 rows per block with 4 tile columns. It needs a third lever.
2. **A per-plane rows target.** ~0.7% at the one cell where it is isolated;
   `probe-shiftpin` + `__rpb_*` price it.
3. **Name the instance behind the residual 153 wide promotions/frame** (§6a).
   Worth about a point of wall by #500 §5e, and a global rung reaches it while
   the derived rule cannot.
4. **x86_64, 4:4:4, 12-bit, inter content.** Unmeasured axes, in that order.
