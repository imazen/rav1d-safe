# `c256x2048` at t=8: both remaining levers measured, both refuted, and the
# mechanism named with a number

**Status: NEGATIVE, and the negative is the deliverable.** The two levers this
round was opened to test — letting the derived shard rule go FINER than the
block-count answer, and changing the shard lock's waiting policy — are both
measured null-or-adverse on the target cell. Nothing ships. What ships instead
is the instrument that was missing (a contention census), four waiting-policy
arms, and a measured statement of what the residual actually is: **the tracker's
cost per registration DOUBLES with every doubling of worker count while the
registration count is constant, and at most 10.7% of it is waiting.**

Record: `benchmarks/c256_contention_2026-08-11.{meta,*.tsv}`.
Prior art, not re-derived: `docs/BPS_ROWS_DEFAULT.md` (PR #503, the rule),
`docs/SHARD_SIZE_SWEEP.md` (PR #501, the size grid),
`docs/CTX_TL_SPLIT.md` §6c (PR #502, the count cut that measured null here),
`benchmarks/cost_census_2026-08-10.meta` (the cell's ceiling and its 18.78 ns).

---

## 1. What is NOT covered, first

* **The cell is not closed.** It sits at **2.378x** of dav1d at t=8 against a
  tracker-removed ceiling of **1.325x** (n=15; an independent n=7 sweep reads
  2.388x / 1.332x), and this round moved neither.
* **One box** (Apple M4 Pro, 8P+4E, macOS 26.5.2, aarch64), **one vector**
  (`C256x2048_420_8b__t8`, 4x2 tiles, 8-bit 4:2:0, all-intra, one key-frame OBU
  re-decoded), **one content class**. No x86_64 — where every lock acquisition
  is dearer, so a waiting-policy arm could plausibly behave differently and this
  round says nothing about that.
* **No 10-bit, 4:4:4, 12-bit, film grain or inter content.** Loop restoration
  executes zero blocks on this vector, the same structural blindness the 4K grid
  has.
* **The shift ladder was swept on the LUMA/CHROMA pin, not on a shipped rule.**
  A rule that refines would also have to decide what to do on every other
  picture size; this round only establishes that there is nothing to win HERE,
  so no such rule was designed.
* **`SLOTS` was not touched** (the brief's standing trap), no reservation was
  widened anywhere (four standing refutations), and no tile- or worker-keyed
  scheme was built (refuted family).
* **The 627 ns per spin iteration in §6 is a RATIO of two measurements, not a
  direct timing.** It is profile-time divided by counted iterations, both taken
  from the same binary in the same run. What it is made of — the relaxed load of
  a contended line, `isb` under memory pressure, or sampler placement inside a
  32-byte function — is NOT established, and §5 is why that no longer matters.
* **Miri's `shard_liveness` target TIMES OUT locally** in every configuration
  and is reported as a timeout, never as green (§8).
* The cross-cell relax grid (§5c) was **load-tagged** (`foreign_max` 1-2 on 5 of
  6 cells): its ratios are paired per round and hold, its absolute ms/frame are
  inflated (`v4k_8tile` reads 51.8 there against 49.6 on an idle box).

## 2. The cell, and why these were the last two levers

| quantity | value | source |
|---|---|---|
| ours / dav1d, t=8 | **2.378x** (2.388x on the n=7 sweep) | this round, two independent sweeps |
| tracker-removed ceiling | **1.325x** (1.332x, n=7) | `probe-untracked`, bit-identical |
| tracker share of wall | **44.3%** | (3.783 − 2.108) / 3.783 |
| registrations / frame, t=8 | **569,690** | `probe-sites`, `lost = 0` |
| ns per registration, t=8 | **19.71** | (25.177 − 13.951) CPU ms/f ÷ regs |

Two levers had already declined it. **A registration COUNT cut** measured
1.0030 wall / 1.0016 CPU for −5.8% of the population (#502). **Coarsening**
measured 0.987/0.987/0.995 across three rungs (#501) and the derived rule
computes the SAME shift there, so #503 treated it as an identity control rather
than an attempt. That left exactly two directions: go the other way on
granularity, and stop spinning.

## 3. Lever 1 — counted first, and the kill switch is CLEAN

A finer block makes a strided borrow touch more shards, and past
`MAX_SHARDS_PER_BORROW` it promotes to the wide path, which holds every shard.
That is the counter-force, and it is countable without a clock. Per frame,
`--features probe-wide,probe-shiftpin`, 16 rungs:

| pin (luma, chroma) | `multi`/f | `w_shards` | `w_blocks` | `w_full` |
|---|---|---|---|---|
| base = (11, 9) | 1,684 | **0** | **0** | **0** |
| (10, 9) / (10, 8) / (9, 9) / (9, 7) / (8, 9) | 1,684 | **0** | **0** | **0** |
| (11, 8) / (11, 7) / (12, 10) | 1,684 | **0** | **0** | **0** |
| (8, 6) / (11, 6) | 4,884 | **0** | **0** | **0** |
| (7, 9) | 6,263 | **0** | **0** | **0** |
| (7, 6) | 9,463 | **0** | **0** | **0** |
| (6, 9) | 14,877 | **0** | **0** | **0** |
| (6, 6) | 18,077 | **0** | **0** | **0** |

**The wide path is never reached, at any rung, on either plane.** So lever 1 was
NOT killed by the wide path and had to be timed. What the table does show is
where multi-shard registrations appear: luma is flat at 1,684/frame down to
shift 8 and explodes below it, because a 256-byte block is exactly one picture
row on this plane — shift 8 is the last rung at which a ≤16-byte registration
cannot straddle a boundary.

The strided-2-D COUNTERFACTUAL (`__probe_bounds`, what a single 2-D record
would have cost — *not* the shipped path, per `BPS_ROWS_DEFAULT.md` §6b) climbs
in step: `row_shards_max` at the loop filter goes 4 → 8 → 16 → 32 and
`pct_row_wide` 0.00% → 14.9% → 100% → 100% as the luma pin goes 11 → 9 → 8 → 7.

## 4. Lever 1 — timed: refuted, monotone, with a named mechanism

n = 7 rounds (8 run, round 0 discarded — the first touch of each (arm, cell) is
cold), rotating arm order, `measlock`, idle box (`foreign_max = 0`), two-point
fit `total = a + b*frames` at 22 and 225 frames, ratios PAIRED per round.

**`pinL11C9` pins the shifts the shipped rule already computes on this cell, so
it is an identity control** and its spread is this grid's floor.

| arm (luma, chroma) | wall ms/f | [min..max] | vs `plain` | vs the identity control | sign |
|---|---|---|---|---|---|
| `plain` (shipped) | 3.803 | [3.724..3.882] | 1.0000 | — | — |
| **`pinL11C9` identity** | 3.764 | [3.709..3.793] | 0.9845 | **1.0000** | 6/7 |
| (10, 9) | 3.744 | [3.695..3.759] | 0.9858 | **0.9947** [0.9740..1.0093] | 5/7 |
| (10, 8) | 3.803 | [3.759..3.837] | 1.0000 | 1.0117 [1.0026..1.0239] | 7/7 |
| (9, 9) | 3.823 | [3.793..3.887] | 1.0039 | **1.0221** [1.0000..1.0314] | 7/7 |
| (8, 9) | 4.207 | [4.158..4.291] | 1.1149 | **1.1179** [1.1078..1.1445] | 7/7 |
| (7, 9) | 4.502 | [4.473..4.571] | 1.1826 | **1.2026** [1.1857..1.2147] | 7/7 |
| (6, 6) | 4.547 | [4.498..4.591] | 1.1971 | **1.2106** [1.1857..1.2377] | 7/7 |
| (12, 10) COARSER | 3.719 | [3.680..3.749] | 0.9767 | 0.9882 [0.9740..1.0106] | 6/7 |
| `untracked` | 2.094 | [2.064..2.108] | 0.5511 | — | 7/7 |
| dav1d 1.5.4 `--framedelay 1` | 1.586 | [1.581..1.591] | 0.4179 | — | 7/7 |

**One shift finer is a null; two is +2.2%; three is +11.8%; five is +21%.** The
shipped shift sits at the bottom of a flat basin whose wall is on the fine side,
and one shift COARSER is also a null — which is the same answer #501's three
rungs gave from the other direction, now with the identity control that round
did not have.

**The mechanism is the shard-line FOOTPRINT, not the straddles.** At (8, 9) the
multi-shard count is still exactly 1,684/frame and the sharing pattern is
unchanged — a 256-byte block is one whole picture row, so all four tile columns
still land in the same block — and the arm is already **+11.8%**. The only thing
that changed is that each worker now touches 8x as many distinct shard lines per
frame. That is `BPS_ROWS_DEFAULT.md` §5c's conclusion ("the money is the
shard-line footprint") measured from the refining side, and it is why the
coarsening lever and the refining lever cannot both pay on one cell.

**And the separation the lever was reaching for IS achievable — it is just the
worst rung on the ladder.** `scripts/perf/av1_tile_info.py` parses this vector
as **4 tile columns x 2 tile rows, `sb_cols = 4`**, i.e. each tile column is
exactly ONE 64-px superblock wide. The luma row is 256 bytes, so:

| luma shift | block | tile columns per block |
|---|---|---|
| 11 (shipped) | 2048 B | 8 whole rows — all 4 |
| 8 | 256 B | exactly one row — still all 4 |
| 7 | 128 B | 2 |
| **6** | **64 B** | **exactly 1 — fully separated** |

Shift 6 is the rung at which the four tile columns finally stop sharing a block,
and it measures **+21%**. So lever 1 is not dead because the separation is out
of reach; it is dead because the separation is not worth what it costs.

**And that reason generalises**: on a 256-px-wide plane the tile columns share a
block at every shift ≥ 8, and every shift below 8 pays more in shard-line
footprint (and, below 8, in straddles) than the separation is worth.

## 5. Lever 2 — the waiting policy, re-opened here and refuted here

`docs/AGENT_BRIEF.md` §6 records "TinyLock backoff: null, measured twice", and
both were taken where contention is ~0.02% of registrations. This cell is a
different regime and deserved its own row rather than an overwrite of that one.
Four arms, one ladder over what a waiter does:

| arm | policy | `spins`/frame | liveness |
|---|---|---|---|
| `plain` | relaxed-load spin, never yields | 1,007 | — |
| `probe-lock-backoff` | spin 64, then `yield_now` | 756 | live |
| `probe-lock-relax` | exponential pause BETWEEN loads | **216** (4.7x fewer) | live |
| `probe-lock-yield` | `yield_now` every iteration | **108** | live |
| `probe-lock-park` | `parking_lot::RawMutex` — a real park | **0** | live |

### 5a. Timed, n = 15, with an in-grid identity control

`plainB` is a byte-identical COPY of `bench_plain` under another name, so its
spread against `plain` is the floor and its sign is a coin flip by construction.

| arm | wall ms/f | [min..max] | wall ratio | [min..max] | sign | CPU ratio | sign |
|---|---|---|---|---|---|---|---|
| `plain` | 3.783 | [3.719..3.828] | 1.0000 | — | — | 1.0000 | — |
| **`plainB` identity** | 3.788 | [3.734..3.842] | **1.0000** | **[0.9807..1.0172]** | **8/15** | 1.0033 | 9/15 |
| `lockbackoff` | 3.739 | [3.680..3.828] | 0.9896 | [0.9653..1.0065] | 12/15 | 0.9953 | 12/15 |
| `lockrelax` | 3.734 | [3.690..3.778] | 0.9857 | [0.9690..1.0040] | 14/15 | 0.9847 | 14/15 |
| `lockyield` | 3.813 | [3.754..4.089] | 1.0091 | [0.9832..1.0964] | 11/15 | **1.0273** | 15/15 |
| `lockpark` | 3.749 | [3.729..3.833] | 0.9922 | [0.9806..1.0305] | 11/15 | 1.0002 | 8/15 |
| `untracked` | 2.108 | [2.089..2.241] | 0.5608 | [0.5470..0.5932] | 15/15 | 0.5558 | 15/15 |
| dav1d | 1.591 | [1.576..1.670] | 0.4209 | [0.4142..0.4420] | 15/15 | 0.4422 | 15/15 |

**The identity control's band is ±1.8% and every lock arm is inside it.** The
best point estimate is `lockrelax` at −1.4% wall / −1.5% CPU; §5c kills it.

**Recorded because it was nearly reported:** a first pass at n=7 read
`lockbackoff` **0.9739** with 6/7 rounds below 1.000, which is exactly the
sub-3% claim `AGENT_BRIEF` §2 warns about. At n=15 it reads **0.9896**. It did
not reproduce.

### 5b. Why it could never have paid, priced before believing any of it

`sample`, 30 s window, busy-normalised (parked threads bucketed as idle):

| arm | busy | idle% | entropy | kernels | lf | cdef | tracker | **sync** | runtime | other |
|---|---|---|---|---|---|---|---|---|---|---|
| `plain` t=2 | 45,301 | 34.23 | 56.11 | 12.37 | 7.40 | 3.56 | 13.38 | **0.26** | 2.97 | 3.95 |
| `plain` t=8 | 136,724 | 25.33 | 44.92 | 11.11 | 5.14 | 2.22 | 22.91 | **4.75** | 3.39 | 5.57 |
| `lockrelax` t=8 | 137,723 | 24.70 | 44.74 | 11.02 | 5.26 | 2.27 | 23.26 | **4.77** | 3.32 | 5.36 |
| `lockpark` t=8 | 129,474 | **28.97** | 47.41 | 11.67 | 5.23 | 2.32 | 25.16 | **0.69** | 3.28 | 4.24 |

At 4.75% of busy against a measured 25.18 CPU ms/frame, the whole waiting
population is **1.20 CPU ms/frame — 10.7% of the tracker's 11.23**. Recovering
ALL of it at the measured 6.66 busy cores would move the cell 3.783 → 3.603
ms/frame, i.e. **2.378x → 2.265x of dav1d**. The bar is 1.4x. So even a perfect lock was worth
about a ninth of what this cell needs, and the arms measure zero of it.

**The park arm shows exactly where it goes**: `sync` 4.75% → 0.69% and idle
25.33% → 28.97%. The spin time is real and it IS removable as CPU — it just is
not on the critical path, because those cores are already idle a quarter of the
time. That is `AGENT_BRIEF` §2's own warning read in the other direction: a
stage's cost under a spinning lock is not that stage's opportunity, and here the
opportunity is zero even though the cost is real.

`lockrelax` is the complement: it cuts loop ITERATIONS 4.7x (1,007 → 216) and
its `sync` share does not move (4.75% → 4.77%), because each iteration now
contains up to 64 pauses. Live, and null.

### 5c. `lockrelax` across cells — flat everywhere, so its −1.4% is the floor

n = 9, load-tagged (`foreign_max` 1-2), paired ratios:

| cell | `plainB` (identity) | `lockrelax` | `lockbackoff` |
|---|---|---|---|
| c256x2048 t=8 | 0.9922 | 0.9807 (8/9) | 0.9845 (9/9) |
| c1024x576 t=8 | 0.9922 | 1.0000 | 0.9961 |
| c1024x384 t=8 | 1.0000 | 0.9980 | 0.9960 |
| c3840x256 t=8 | 0.9962 | 1.0061 | 0.9961 |
| v4k_8tile t=8 | 1.0054 | 0.9946 | 1.0021 |
| c1024x576 t=1 | 0.9980 | 1.0004 | 0.9968 |

Against the identity control on its own cell the relax arm is 0.9807/0.9922 =
**0.988**, and it is flat on all five other cells. Not shipped.

## 6. Two instrument corrections this round had to make

**A niced counted run measures a different machine.** `c256_counts.sh` is
`nice -n 19` by the campaign's convention, which on Darwin maps to background
QoS and lands all eight workers on the four E-cores. Registration COUNTS are
indifferent to that; CONTENTION counts are not. The first pass read
`lockslow = 1,072/frame`; un-niced under `measlock` the same build reads **96**.
`scripts/perf/c256_contention.sh` exists to keep those two apart and runs
un-niced under the lock even though it only reports counters.

**The counter and the profile disagreed by ~150x, and both are right.** The
contention census says the spin loop runs 1,007-1,373 iterations/frame, and a
`core::hint::spin_loop()` was directly measured at **7.6 ns** per iteration on an
idle core, of which **6.7 ns** is the hint itself (200 M iterations x 3 rounds
under `measlock`, minus an empty-loop control; `examples/spin_cost.rs`
reproduces the hint at 6.77-6.78 ns under load) — which predicts
**0.008-0.010 CPU ms/frame** over the counted population, against a profile that
puts `lock_slow` at 0.86-1.20. Reconciled by running BOTH instruments in ONE binary
(`probe-wide` + `sample`, 9,000 frames): 5,984 leaf samples in `lock_slow`
against that run's own counter of 1,373 spins/frame over 6,949 windowed frames
= **~627 ns per iteration under real contention, 80x the idle-core price.**
The symbol is 32 bytes of text (`nm`: `0x1001b70fc..0x1001b711c`) and
`add_contended` is a separate symbol with 499 samples, so this is not
misattribution to a neighbour. What the 80x is made of is not established —
and §5b is why it stopped mattering.

## 7. What the residual actually is, measured

Thread-count ladder on the target cell, n = 7, idle (`foreign_max = 0`), with
the registration count read off `probe-sites` (`lost = 0`) at each point:

| t | ours ms/f | untracked | dav1d | ours/dav1d | ceiling | tracker CPU ms/f | regs/frame | **ns/reg** | cores |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 12.404 | 11.827 | 9.442 | 1.314 | 1.253 | 0.596 | 272,505 | **2.19** | 1.00 |
| 2 | 7.444 | 6.144 | 4.833 | 1.540 | 1.271 | 2.578 | 569,690 | **4.52** | 1.98 |
| 4 | 4.590 | 3.236 | 2.535 | 1.811 | 1.277 | 5.229 | 569,690 | **9.18** | 3.87 |
| 8 | 3.764 | 2.099 | 1.576 | 2.387 | 1.331 | 11.227 | 569,690 | **19.71** | 6.70 |

Three facts, and together they are the finding:

1. **The registration count is IDENTICAL at t=2, 4 and 8** — 569,690 per frame,
   because the `tile_threading_active()` latch is already thrown at t=2. Nothing
   about the population changes across those three points.
2. **The cost per registration doubles with every doubling of workers** — 4.52
   → 9.18 → 19.71, ratios 2.03 and 2.15. (t=1 is a different population, 272,505.)
3. **At most 10.7% of that is waiting** (§5b), and the measured recovery from
   removing waiting entirely is zero.

So **≥89% of the tracker's t=8 cost is on the UNCONTENDED path**: the ordinary
`add`/`remove` pair getting slower as more cores touch the same shard. One
cross-core transfer of the shard's own cache line per registration, priced at
~17.5 ns of premium over the t=1 rate, is the whole arithmetic.

**Therefore the remaining mechanism must change HOW MANY CORES SHARE A SHARD
LINE.** It cannot be granularity: on a 256-px-wide plane the four tile columns
of a row occupy one block at every shift ≥ 8, and every shift below 8 loses more
than it separates (§4). It cannot be the waiting policy: the sharers still
share (§5). It cannot be the count: #502 removed 5.8% and measured 1.0030. The
only family left is making a registration's record **worker-private rather than
address-shared** — i.e. either not registering at all (the `get_mut` /
untracked-read direction #492 and #502 mined, which removes the sharing by
removing the record) or keying the record on the worker (the tile-keyed family,
refuted on other grounds and deliberately not attempted here).

And the size of the prize is bounded: even at the tracker-removed ceiling this
cell is **1.32x** of dav1d, band [1.299..1.342]. Closing it to 1.4x means
removing essentially ALL tracker cost at t=8 on this geometry, not a slice.

## 8. Gates

Driver `scripts/perf/c256_gates.sh`, logs `~/tmp/c256/gates`. **This branch
adds no shipping behaviour** — every addition is behind a `__`-gated feature
absent from `default` and from every published feature — but the corpus legs run
anyway, because "the default codegen is unchanged" is a claim about a file that
was edited and 766/766 BY NAME is the evidence rather than the assertion.

| gate | result |
|---|---|
| `cargo test --lib`, release **and** debug | pass, both |
| tracker crate: default + `__probe_lock_{backoff,yield,relax,park}` + `__probe_wide` + `__probe_wide,__probe_lock_park` + `__probe_wide,__probe_lock_relax` + `__probe_shiftpin` + `__bps_blocks` + `__rpb_{2,8,16}` + `__msb_5` + `--no-default-features` | 16 configurations, all pass |
| **corpus, DEFAULT arm, t=1**, no `--skip-group` | **766 PASS + 2 SKIP** — re-run at the final HEAD (`85af168`) after the last code-touching commit, still CLEAN |
| **corpus, DEFAULT arm, t=8**, no `--skip-group` | **766 PASS + 2 SKIP** — re-run at the final HEAD, still CLEAN |
| corpus, `probe-lock-park`, t=8 | 766 PASS + 2 SKIP |
| corpus, `probe-lock-relax`, t=8 | 766 PASS + 2 SKIP |
| set-diff BY NAME (key `(group, name)`, value `(status, ACTUAL md5)`) vs `benchmarks/aarch64_md5_fixes_2026-08-07_final.tsv.zst` | **CLEAN on all four**: 0 only-in-baseline, 0 only-in-head, 0 differing |
| set-diff t=1 vs t=8, DEFAULT arm | CLEAN |
| loop-filter window `debug_assert`, `-C debug-assertions=on`, `8-bit/data` t=8 | **358 vectors, mismatch=0, error=0** |
| `mt_stress` threads 1/2/4/8/16 x 5 trials | pass (`test-vectors/bench/photo_4k.avif` copied in — it is gitignored and `mt_stress` `.expect()`s it) |
| `tile_threading_overlap` / `reproduce_overlap` / `thread_cleanup_test` | pass BOTH invocations — plain (runs 2 of 11) and `-- --ignored` (the other 9) |
| `multi_decoder_pressure` — 12 concurrent decoders x 3 iters over 5 vectors | PASS |
| every timed arm's `CHECKSUM` before any timing | 15/15 identical, and identical to **dav1d 1.5.4's md5 at t=1 and t=8** |
| the EXACT CI legs (4 clippy/doc + `cargo fmt --all --check`) | all rc=0 |

### 8a. Test teeth, proven by planting

Every mutation was restored from a `~/tmp` backup COPY — never `git checkout --`
— and verified byte-exact by sha256 **and** `git diff --exit-code`.
`crates/rav1d-disjoint-mut/src/tracker_shard.rs` sha256
`909b570ba3806b5b06f0626a73835242258c4da4e239567338bee029548170e2` before and
after.

| planted mutation | gate | result |
|---|---|---|
| (control) | tracker tests, `__probe_lock_park` | ok |
| park arm's `try_lock()` returns `true` unconditionally | tracker tests, `__probe_lock_park` | **FAILS** |
| default `lock_slow` returns without acquiring | tracker tests, default | **FAILS** |
| relax arm's pause cap 64 → 1 | tracker tests, `__probe_lock_relax` | **passes** — reported as a NON-mutation, not as coverage: the cap is a policy constant, not an invariant, and no test can or should pin it |

**Standing hazards, replanted** under `--features __probe_wide`,
`crates/rav1d-disjoint-mut/tests/wide_exclusion.rs`:

| plant | result |
|---|---|
| baseline | ok |
| `4af62ae`'s in-lock `state` re-read deleted from `add_contended` | **FAILED** |
| `active()` cut to one shard | **FAILED** |
| after both restores | ok |

`forbid(unsafe_code)` is proven ACTIVE, not read: an
`unsafe { core::mem::transmute(x) }` planted in `src/picture.rs` (which has no
module-level forbid of its own, and is compiled in every configuration) fails
the build against **`lib.rs:13:12`** — the campaign brief's anchor, confirmed
for the fourth round running. Restored, sha256
`fa02c12b7730dbeba3f2304e366d245dc9eb30e35153a5e7ea7fc6856969d5e3` before and
after, `git diff` clean, lib rebuilt green.

### 8b. Clippy

The four legs CI actually runs all pass (`rc=0`), as does `cargo fmt --all
--check`. `cargo clippy --release --all-targets -- -D warnings` (NOT a CI leg)
fails on **both** the base commit and this branch — aarch64 79 errors on base
and 91 on head, x86_64 likewise — and the counts differ only because clippy
aborts at the first failing target and the target order differs. The complete
set of files clippy names over all runs is
`src/safe_simd/{itx_arm,itx_arm_neon_16x16,mod}.rs`,
`examples/{bench_ivf_limit,itx_shape_census,md5_ablate,md5_inventory,profile_ivf}.rs`
and `tests/thread_cleanup_test.rs`. **Zero findings in `tracker_shard.rs`,
`crates/rav1d-disjoint-mut/src/lib.rs` or any file this branch touches**,
checked by grepping every log.

### 8c. Miri

`cargo +nightly miri test -p rav1d-disjoint-mut --no-fail-fast --test <target>`,
one target at a time (Miri aborts the process on first UB and cargo stops at the
first failing TARGET, so a batch run lets later targets never execute and their
silence reads as health), Stacked Borrows and Tree Borrows, and **both the
default lock and `__probe_lock_park`** — the arm that replaces the lock's whole
implementation, which is what the tracker's mutual-exclusion argument rests on.
The other three arms only change what a waiter does between attempts and cannot
move that argument; they are covered by the unit-test legs.
`benchmarks/c256_miri_2026-08-11.tsv`.

| target | SB default | SB `park` | TB default | TB `park` |
|---|---|---|---|---|
| `--lib` | 29 passed | 29 passed | 29 passed | **TIMEOUT** |
| `narrow_release` | 1 | 1 | 1 | 1 |
| `soundness` | 25 | 25 | 25 | 25 |
| `wide_exclusion` | 1 | 1 | 1 | **TIMEOUT** |
| `guard_move_release` | 2 | 2 | 2 | 2 |
| `pic_buf_overflow` | **0 tests ran** | 0 | 0 | 0 |
| `aligned_miri` | **0 tests ran** | 0 | 0 | 0 |
| `shard_liveness` | **TIMEOUT** | **TIMEOUT** | **TIMEOUT** | **TIMEOUT** |

**The DEFAULT lock is clean on all 7 non-timeout targets under both models.**

**`shard_liveness` times out (rc=124 at 900 s) in all four configurations** and
is reported AS a timeout, never as green — it is the target
`docs/AGENT_BRIEF.md` warns about on aarch64. CI's Linux Miri legs, which run
the whole package with `--all-features`, are green on this branch (Stacked
Borrows on both workflow runs, Tree Borrows likewise) and are what cover it.

**Two MORE timeouts appear only in the TB x park corner** — `--lib` and
`wide_exclusion` — and they are reported as timeouts, not as findings and not as
green. Both pass under Stacked Borrows with the same feature, and both pass
under Tree Borrows with the default lock, so this is `parking_lot`'s parking
machinery being expensive to interpret under TB rather than anything about the
tracker. **TB coverage of the park arm is therefore 6 of 8 targets, and the park
arm is a measurement arm that will never ship.**

`pic_buf_overflow` and `aligned_miri` select **0 tests** under these feature
sets and are reported as 0, never as green.

## 9. Where this leaves the campaign

`docs/BPS_ROWS_DEFAULT.md` §9 ranked `c256x2048` as follow-up #1, "refused by
two levers, needs a third". It has now been refused by four:

| lever | verdict | number |
|---|---|---|
| registration COUNT cut (#502) | null | 1.0030 wall / 1.0016 CPU for −5.8% of the population |
| COARSER blocks (#501, #503) | null | 0.987 / 0.987 / 0.995; and (12, 10) here reads 0.9882 |
| **FINER blocks (this round)** | **adverse, monotone** | 0.9947 / 1.0221 / 1.1179 / 1.2026 / 1.2106 at −1/−2/−3/−4/−5 shifts |
| **waiting policy (this round)** | **null, and bounded at 10.7% even if perfect** | 0.9857-1.0091 wall, all inside a ±1.8% identity-control band |

The next lever must remove the SHARING, not the count, the granularity or the
wait — and the two spellings of that are already on the board: registering
nothing (the `get_mut` direction, which is where #492's 21-25% and #502's cuts
came from) or keying a record on its worker. Before either is built, note the
size of the prize on this geometry: the ceiling is 1.32x, so this cell is asking
for the tracker to be nearly free at t=8, not cheaper.

`docs/AGENT_BRIEF.md` §6's "TinyLock backoff: null, measured twice" now has a
third row rather than a rewrite: **null a third time, measured where the
mechanism is present, with the mechanism priced at 10.7% and its removal
measured at zero.** That question is closed.
