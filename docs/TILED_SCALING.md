# The tiled t=8 scaling deficit, attributed

**Status: an attribution, not a fix.** Nothing in `src/`, `lib.rs`, `include/`,
`crates/` or `Cargo.toml` changed on this branch — only `scripts/perf/`, `docs/`
and `benchmarks/`. Verified: `git diff b700489..HEAD -- src/ lib.rs include/
crates/ Cargo.toml Cargo.lock build.rs` is empty. (Diff against the recorded
base SHA, never against `main`: all worktrees share one `.git`, so another
agent's merge shows up as reverse-deletions in yours.)

Provenance, arms, hygiene and the raw files: `benchmarks/tiled_scaling_2026-08-10.meta`.

> **Landed 2026-08-12 from PR #499, which was marked "measure-only, do not
> merge".** The label was about the branch, not the record: its nine
> `scripts/perf/tiled_*` tools reached `main` already (via #500, `54e90d8`) and
> `docs/AGENT_BRIEF.md` §6 has cited **this file** by name in **two** rows
> since (*corrected 2026-08-12: the banner first said three, from memory rather
> than a count —* `grep -o 'docs/TILED_SCALING\.md' docs/AGENT_BRIEF.md | wc -l`
> *= 2, the dav1d-tiled-scheduler row and the filter-chain row. A third row, the
> post-tile filter tail, states §4's finding but names only the* `TAIL_CONC`
> *instrument.*) — with the file itself absent from `main`. So the records land; nothing
> that made the branch un-mergeable comes with them (no `src/`, no `crates/`, no
> `Cargo.toml`, and its `.gitignore` reversion of the `test-vectors` symlink line
> is dropped — only the `__pycache__` addition is kept).
>
> **One claim is corrected**, in place and dated: §7's item 1 named a lever that
> has since been BUILT and SHIPPED, and its one-word diagnosis ("contention") was
> subsequently refuted. See §7. §§1-6 are unchanged and are still the current
> position — AGENT_BRIEF §6 quotes §1's +27.3%/+4.4%/+3.2% and §6's 2.8x/4.0x
> filter-chain figure as current.

## Why this document exists

The tiled scaling deficit had been measured three times and profiled zero times.
Every profile in the campaign's ~20 rounds was single-tile or t=1 — the one shape
where **no tile parallelism exists at all** — so nothing that appears only when
several tile workers and several superblock-row filter tasks are live at once
could be seen. This is the first look at the tiled arm.

## The answer, in order of size

Measured at t=8 on an idle box (`foreign_max = 2`, n = 7 rounds, M4 Pro), the
wall gap to `dav1d --framedelay 1` decomposes additively as
`wall(t) = cpu(1)/t + (cpu(t)-cpu(1))/t + (wall(t) - cpu(t)/t)`:

| term | 1024x576, 8 tiles | 4K, 8 tiles | owner |
|---|---|---|---|
| single-thread gap / t | 0.433 ms (28.4%) | 4.062 ms (46.6%) | **not a scaling problem** — the standing t=1 gap, divided by 8 |
| **added work** | 0.447 ms (29.4%) | 1.831 ms (21.0%) | **100% the borrow tracker** |
| **idle cores** | 0.643 ms (42.2%) | 2.821 ms (32.4%) | **~41% the tracker**, the rest a post-tile filter tail |
| total gap | 1.522 ms | 8.714 ms | |

**1. Every bit of the added-CPU half is the borrow tracker, and it is fully
closed by removing it.** CPU growth from t=1 to t=8 on the tiled vectors:

| arm | 1024x576 | 4K |
|---|---|---|
| ours (ships) | **+27.3%** | **+10.5%** |
| ours, tracker compiled out | +4.4% | +3.6% |
| dav1d --framedelay 1 | +3.2% | +3.1% |

The tracker-free arm lands on dav1d's number to within a point. It is
bit-identical output (8 of 8 md5s equal across arms and thread counts).

**2. Inside that, one stage dominates: the post-tile deblocking column pass.**
At 1024x576, of the +2.239 ms/frame that t=4 -> t=8 adds, `deblock_cols` is
+1.349 (60%, a **2.57x** inflation of one stage) and is +0.030 at the ceiling.
At 4K, +4.365 of +7.704 (57%), +0.229 at the ceiling.

**3. The profile names the leaf and finds a symbol that exists only at t=8.**
`TinyLock::lock_slow` — a core spinning for another core's shard — is 0.00% of
busy self time at t=1 **and at t=2** and 1.19% at t=8.

**4. A second deficit survives at the ceiling: the post-tile filter tail**, worth
9-12% of wall, structural, and independent of the tracker.

**5. A separate and larger finding, not about scaling at all: our filter chain
costs 2.8x (1024) / 4.0x (4K) dav1d's *at t=1, with the tracker compiled out* —
the deblock chain alone 5.9x / 5.6x.** That is the compact copy-in/write-back
architecture the safe-guard model requires, and on these vectors it is a bigger
absolute number than everything above.

Work distribution, stragglers and `check_tile` deferral are all measured **not**
to be the problem.

## Instruments

Four, all pre-existing, so the round added measurement and not machinery.

* **`--features probe-tasktime`** (`src/probe_tasktime.rs`) — per-stage and
  per-worker busy ns, a 50 us time-weighted concurrency histogram, the
  tail-restricted histogram (**no** tile worker live and >= 1 filter worker), and
  `check_tile` deferral causes. Two `Instant::now()` per stage execution against
  ~45 stage executions per frame.
* **`--features probe-tasktime-untracked` / `probe-untracked`** — the same source
  with the tracker compiled out (`DisjointMut::new` stores `tracker: None`, so
  `add`/`remove` vanish; **the compact copy-in/write-back stays**, which is what
  makes finding 5 separable from finding 1). This is what turns "the tracker is
  implicated" into a subtraction.
* **`/usr/bin/sample`, self-time leaves**, on the SHIPPING binary, same vector,
  t=1 / t=2 / t=8.
* **`RAV1D_INLOOP` vs `dav1d --inloopfilters`** — the same string on both
  decoders, so each one's own filter cost is measured by one clock. This is the
  cross-decoder check on findings 1, 2 and 5.

Drivers/reducers, all new: `scripts/perf/tiled_taskprobe.sh`,
`tiled_taskprobe_report.py`, `tiled_stage_delta.py`, `tiled_prof.sh`,
`tiled_prof_report.py`, `tiled_wallcpu.sh`, `tiled_wallcpu_report.py`,
`tiled_inloop_ab.sh`, `tiled_inloop_report.py`.

### Three traps, all of which produced a wrong number first

* **At `--threads 1` every stage counter reads 0.000.** `n_tc == 1` creates no
  task worker, so `rav1d_task_run` — where all the stage instrumentation lives —
  is never entered. A per-stage **t8/t1** ratio does not exist; the low arm must
  be t=2 or t=4. That is also the better comparison: t=2/t=4 are on the same code
  path as t=8 (`tile_threading_active()` latched, narrow guards, adaptive block
  shift), so the ratio isolates *adding workers* from *switching code paths*. The
  t=1 wall stays the right denominator for a speedup and the wrong one for a
  per-stage CPU ratio.
* **`sample` samples PARKED threads.** `__psynch_cvwait` is 37.3% of leaves at
  t=8 and 0.0% at t=1, so leaving it in the denominator deflates every busy
  symbol by exactly the amount the pool sleeps — the opposite of the quantity
  being attributed. All percentages here are normalised on busy samples.
* **`n_hi` past the end of the stream silently halves the gap.**
  `bench_*` re-decodes one OBU exactly `n` times whatever you ask;
  `dav1d --limit N` stops at end of stream. The first pass used `n_hi = 24` on a
  16-frame 4K IVF, so the two-point fit divided a short total by a long frame
  delta and dav1d read **94.1 ms/frame instead of 152.1** — which would have put
  the t=1 gap at 1.96x instead of 1.21x and made every 4K conclusion wrong.
  `tiled_wallcpu.sh` now counts each stream and refuses `n_hi > n_frames` with a
  FATAL. Caught by disagreeing with the prior record, not by inspection.

## 1. The CPU inflation is the tracker, and mostly one stage

In-stage busy ms/frame, median of 3 rounds, `foreign_max = 1`, bands in
`benchmarks/tiled_stage_delta_t4_2026-08-10.txt`.

`L1024x576_420_8b__t8` (1024x576, 4x2 = 8 tiles, 9 superblock rows):

| arm | t | busy | tile_recon | deblock_cols | deblock_rows | cdef | wall |
|---|---|---|---|---|---|---|---|
| tracked | 2 | 16.909 | 13.793 | 0.898 | 0.576 | 1.634 | 8.721 |
| tracked | 4 | 16.093 | 13.102 | 0.859 | 0.555 | 1.576 | 4.362 |
| tracked | **8** | **18.332** | 13.514 | **2.208** | 0.684 | 1.972 | **3.302** |
| untracked | 4 | 14.013 | 12.223 | 0.538 | 0.398 | 0.854 | 3.835 |
| untracked | **8** | **14.062** | 12.226 | **0.568** | 0.410 | 0.864 | **2.360** |

t=4 -> t=8 in ms/frame. The bands do not overlap: tracked `deblock_cols` is
[0.856..0.862] at t=4 and [2.175..2.247] at t=8.

| | total | deblock_cols | cdef | deblock_rows | tile_recon |
|---|---|---|---|---|---|
| tracked | **+2.239** | +1.349 (2.570x) | +0.396 | +0.129 | +0.412 |
| untracked | **+0.049** | +0.030 | +0.010 | +0.012 | +0.003 |
| tracker's share | **97.8%** | 97.8% | 97.5% | 90.7% | 99.3% |

`L3840x2160_420_8b__t8` (34 superblock rows):

| | total | deblock_cols | cdef | deblock_rows | tile_recon |
|---|---|---|---|---|---|
| tracked | **+7.704** | +4.365 (1.422x) | +0.683 | +0.515 | +2.452 |
| untracked | **+0.520** | +0.229 | +0.039 | +0.227 | +0.019 |
| tracker's share | **93.3%** | 94.8% | 94.3% | 55.9% | 99.2% |

The tracker's *baseline* tax (tracked/untracked busy at the same thread count)
grows with threads on tiled content and **not** on single-tile content:

| cell | t=2 | t=4 | t=8 |
|---|---|---|---|
| 1024x576 8-tile | 1.143x | 1.148x | **1.304x** |
| 4K 8-tile | 1.087x | 1.092x | **1.132x** |
| 1024x576 1-tile | 1.138x | 1.143x | 1.141x |
| 4K 1-tile | 1.097x | 1.101x | 1.103x |

**The single-tile control is what makes this a finding rather than a
coincidence.** On `L1024x576_420_8b` and `L3840x2160_420_8b` every stage is flat
from t=2 to t=8 in BOTH arms, and every t=4 -> t=8 stage movement is inside its
own min/max band (the report prints `n/a  (|delta| <= widest band ... — NOT a
movement)` rather than a share). Same binary, same thread count, same box: only
the tiling differs.

## 2. The profile names the leaf

`L1024x576_420_8b__t8`, shipping binary, self time normalised on busy samples
(`benchmarks/tiled_selftime_2026-08-10.txt`):

| bucket | t=1 | t=2 | t=8 | t2->t8 pp |
|---|---|---|---|---|
| entropy | 64.05 | 58.48 | 54.15 | -4.33 |
| kernels | 13.01 | 12.35 | 11.62 | -0.73 |
| loop filter | 6.83 | 7.27 | 9.22 | +1.96 |
| cdef | 5.35 | 5.23 | 4.25 | -0.98 |
| **tracker** | **5.98** | **12.06** | **14.58** | **+2.52** |
| **sync (`TinyLock::lock_slow`)** | **0.00** | **0.00** | **1.19** | **+1.19** |
| runtime | 2.51 | 2.39 | 2.55 | +0.16 |

Top risers t=2 -> t=8, percentage points of busy self time:

```
  +1.78   7.33 ->  9.11  BorrowTracker::add::<false>
  +1.19   0.00 ->  1.19  TinyLock::lock_slow
  +0.95   0.85 ->  1.80  LfBlock<BitDepth8>::close
  +0.62   1.83 ->  2.45  safe_simd::loopfilter_arm::lf_compact_run_neon
  +0.38   0.23 ->  0.61  BorrowTracker::add_wide::<true>
  +0.29   1.95 ->  2.24  loopfilter::loopfilter_sb_direct::<BitDepth8>
```

`TinyLock::lock_slow` cannot appear without contention, and it appears only at
t=8. `add_wide::<true>` more than doubles — the wide path holds EVERY active
shard of an instance, so any rate there is disproportionate. `decode_coefs`
falls 3.76 pp purely as a denominator effect. At 4K the same shape, weaker
(tracker 8.11 -> 8.82, `lock_slow` 0.00 -> 0.23): 34 superblock rows against 9
give the filter chain more independent work and less of the frame is spent with
several filter tasks colliding.

### The lock is a bare spin, and that is measurable in the aggregate

`TinyLock::lock_slow` (`crates/rav1d-disjoint-mut/src/tracker_shard.rs:471`)
spins on a relaxed load with `spin_loop()` and **never yields** — the
`yield_now()` is behind `__probe_lock_backoff`, off in shipping builds. So a
worker waiting for a shard burns CPU rather than sleeping, and a stage's measured
cost is partly spin that does not disappear when its work does. That prediction
is testable without a profile, and it holds:

CPU ms/frame *saved* by disabling a filter, `L1024x576_420_8b__t8` at t=8
(`benchmarks/tiled_inloop2_2026-08-10.tsv`):

| arm | deblock off | cdef off | both off | sum of singles vs both |
|---|---|---|---|---|
| ours (ships) | +0.250 | +1.811 | **+4.978** | 2.061 vs 4.978 — **super-additive** |
| untracked | +1.000 | +0.828 | +1.817 | 1.828 vs 1.817 — **additive to 0.6%** |
| dav1d | +0.156 | +0.456 | +0.628 | 0.612 vs 0.628 |

Removing *both* filter stages recovers 4.978 ms, which agrees with the probe's
measured filter-chain total (2.208 + 0.684 + 1.972 = 4.864) to 2.3%. Removing
*either one alone* recovers far less than that stage costs. With the tracker out,
the costs are separable.

**And the spin is visible symbol by symbol.** Profiling the same cell at t=8 with
deblocking on and off (`benchmarks/tiled_spin_selftime_2026-08-10.txt`;
`examples/profile_ivf` under `RAV1D_INLOOP` + `RAV1D_THREADS`, 25 s window, 60
passes), in percentage points of busy self time:

```
RISERS when the deblock work is REMOVED       FALLERS (the ablation is live)
  +5.58   1.45 ->  7.03  TinyLock::lock_slow   -2.45  2.45 -> 0.00  lf_compact_run_neon
  +1.30   2.73 ->  4.03  BorrowTracker::add    -2.15  2.15 -> 0.00  loopfilter_sb_direct
  +1.17   1.55 ->  2.72  cdef_filter_block     -2.10  2.10 -> 0.00  LfBlock::close
  +0.57   0.66 ->  1.23  add_wide::<true>      -1.68  1.68 -> 0.00  LfBlock::open
  +0.37   0.29 ->  0.66  remove_wide           -2.45 48.39 ->45.93  decode_coefs
  +0.31   0.00 ->  0.31  add_contended
```

The loop-filter symbols go to exactly 0.00, so the ablation really fired; the
`sync` bucket **4.85x** (1.45% -> 7.03%) and `add_contended` appears from
nothing. Deleting the work did not delete the contention — it converted it into
spin. **A per-stage cost under a spinning lock is not a per-stage opportunity**,
which is why §7 ranks "make concurrent filter tasks land on different shard
lines" above "make one stage cheaper".

(At 4K the aggregate arms are roughly additive: ours 21.143 + 6.715 = 27.86
against 27.500 for both. The super-additivity is a 1024x576 effect, where the
whole frame is 3.3 ms of wall so the contention window is the entire decode.)

## 3. It is NOT the things the history suggested first

Each measured, not reasoned about:

* **Not stragglers.** Per-worker busy ms/frame at t=8, 1024x576/8-tile:
  `[2.32 2.34 2.27 2.28 2.33 2.37 2.23 2.28]` — spread 0.14 ms on a 2.3 ms mean.
  4K: spread 0.96 on 25.5.
* **Not `check_tile` deferral.** `own_progress` deferrals are 28 per frame
  against 36 admissions at t=2, t=4 **and** t=8 — constant in thread count, 0.78
  rejects per admit. (The frame-global deblock barrier that used to live there
  produced 5.7 rejects per admit and 86% of all deferrals:
  `benchmarks/p1_barrier_2026-08-07.meta`.) `pass2_progress`, `ref_progress` and
  `deblock_barrier` are all 0.
* **Not a uniformly starved pool.** 44.3% of wall at 1024x576/8-tile is spent
  with **exactly 8** workers inside a stage body, 76.4% at 4K. The distribution
  is bimodal, so the mean alone ("5.48 of 8") is the wrong summary:
  `0:1.3 1:5.0 2:14.0 3:6.6 4:14.3 5:5.5 6:6.2 7:2.7 8:44.3`.
* **Not loop restoration or super-resolution.** Both execute zero blocks on
  these vectors, as `docs/BOUNDS_MAP.md` already warned for the 4K pair. Any
  claim about LR at t=8 still needs a vector that runs it.
* **Not dav1d's scheduler.** See §5.

## 4. The second deficit: the post-tile filter tail

The tail-restricted histogram counts samples where **no** worker is in a tile
stage and at least one is in a filter stage — the post-tile chain on its own.

| cell | arm | t | tail % of wall | mean workers | upper-bound recoverable |
|---|---|---|---|---|---|
| 1024x576 8-tile | tracked | 2 | 3.3% | 1.00 | — |
| 1024x576 8-tile | tracked | 4 | 6.7% | 1.00 | — |
| 1024x576 8-tile | tracked | **8** | **34.1%** | **3.22** | 0.673 ms = 20.4% of wall |
| 1024x576 8-tile | untracked | 8 | 17.5% | 2.59 | 0.279 ms = 11.8% of wall |
| 4K 8-tile | tracked | 8 | 19.0% | 3.02 | 3.606 ms = 11.9% of wall |
| 4K 8-tile | untracked | 8 | 13.4% | 2.38 | 2.469 ms = 9.4% of wall |

"recoverable" is the tail's wall minus what it would cost at 8 workers. It is an
**upper bound, not an opportunity**: the chain has real dependencies and 8 is not
reachable.

Two readings, not alternatives:

1. **Most of the tail is downstream of §1.** Removing the tracker halves both the
   tail fraction (34.1% -> 17.5%) and the workers in it. The tail is largely long
   BECAUSE the filter stages are inflated.
2. **A tail survives at the ceiling**, ~9-12% of wall, and it is structural. The
   five filter stages are driven by ONE task per superblock row falling through
   all of them (`src/thread_task.rs:1352-1470`), and consecutive rows are ordered
   by `frame_thread_progress.deblock` and the `copy_lpf` bitmap (`:1391-1420`) —
   CDEF for row N needs `lr_copy_lpf` of row N-1. With 9 rows and 8 workers there
   is not enough independent filter work to fill the pool once the 36 tile tasks
   drain. This is `docs/OWNERSHIP_MODELS.md` §7d option 2 ("partition by edge
   class, not by region... unbuilt, unpriced") arriving from the scheduling side.

## 5. dav1d, and the frame-threading axis we do not have

**Scheduling is not where we differ.** `src/thread_task.rs` is a port of dav1d's
`thread_task.c`: per-(tile, superblock row) tile tasks, one fall-through filter
task per superblock row, the same `copy_lpf` / deblock-progress ordering, and —
since `06160a6` was removed — without the extra frame-global deblock barrier we
once had. What dav1d does not have is a borrow tracker; it indexes the shared
plane through raw pointers. Reading dav1d's scheduler further would be the next
step if the profile pointed at scheduling. It points at a data structure dav1d
does not have.

t=1 -> t=8 speedup, idle box, n=7, two-point fit
(`benchmarks/tiled_wallcpu_2026-08-10.txt`):

| cell | ours | ours, tracker-free | dav1d --framedelay 1 | dav1d default |
|---|---|---|---|---|
| 1024x576, 8 tiles | 4.412x | 5.792x | 6.284x | 7.426x |
| 3840x2160, 8 tiles | 6.152x | 6.731x | 7.144x | 7.658x |

cores busy at t=8: ours 5.62 / 6.80, tracker-free 6.05 / 6.97,
dav1d_fd1 6.49 / 7.37, dav1d_def 8.04 / 8.01.

Removing the tracker recovers **73.7%** of the 1024x576 speedup gap and **58.4%**
of the 4K one. It does not close them: a residual 0.49x / 0.41x remains, which is
§4's tail plus the single-thread gap showing through Amdahl.

**`dav1d_def` is not the comparator, and it is worth saying why.** Its 8.04 cores
come from frame threading stacked on tile threading, worth +18% (1024x576) and
+7% (4K) over `--framedelay 1`. Our checked build hard-pins `n_fc = 1`
(`src/lib.rs:127` — frame threading needs `unchecked`), so that axis is closed to
us by construction and `dav1d_fd1` is the matched-threading-model arm.

### Correcting the framing this round was given

The prior record is load-tagged and its absolutes are inflated; its **ratios**
held, exactly as `AGENT_BRIEF` §2 predicts. Ours/dav1d_fd1 speedup ratio at
1024x576/t=8: 3.893/5.522 = 0.705 there, 4.412/6.284 = 0.702 here.

Also: the "CPU ratio worsens 1.427 -> 2.04" figure conflates two columns. In that
record 2.039 is the **wall** ratio; the CPU ratio goes 1.427 -> 1.763. On this
idle box the same pair reads wall 1.318 -> 1.872 and CPU 1.318 -> 1.623. Both
worsen; wall is the sharper of the two, not CPU.

## 6. The largest single number in the record is not about threading

From the same cross-decoder instrument, filter-chain CPU ms/frame with the
tracker compiled out:

| cell | t | ours (tracker-free) | dav1d | ratio |
|---|---|---|---|---|
| 1024x576 8-tile | 1 | 1.683 | 0.606 | 2.78x |
| 1024x576 8-tile | 8 | 1.817 | 0.628 | 2.89x |
| 4K 8-tile | 1 | 14.572 | 3.642 | 4.00x |
| 4K 8-tile | 8 | 13.500 | 3.500 | 3.86x |

and the deblock chain alone at 4K: **11.5 ms/frame against dav1d's 2.07 at t=1**
(5.6x), 11.2 against 1.93 at t=8 (5.8x).

This is **thread-count-independent** (the ratio is flat across t) and survives
the tracker's removal, so it is neither §1 nor §4. It is the compact
copy-in/write-back the safe-guard model needs — `lf_compact_run_neon`,
`LfBlock::open`, `LfBlock::close` in the profile, and the 35,756-byte hull
reservations `docs/BOUNDS_MAP.md` records at t=1 — against dav1d filtering in
place through pointers. On these vectors it is worth 10.9 ms/frame at 4K, larger
than the entire t=8 scaling gap. **It belongs to the single-thread campaign, not
this one**, and it is flagged here because this round is the first to put a
number on it with dav1d on the same clock.

## 7. What to do next, ranked by what the measurement supports

1. **Make concurrent filter tasks land on DIFFERENT shard lines.** The only lever
   the data points at.

   > **BUILT AND SHIPPED, 2026-08-10/11 — and the diagnosis below is half
   > corrected.** The shift ladder re-run this item asks for was done, scored on
   > the tail, and produced a shipped default: **#500** (`perf/shard-granularity`,
   > the ladder), **#501** (`perf/shard-size`, the same rung swept across picture
   > size — the axis is `rows_per_block = 2^shift / stride`, crossover between
   > 2.1 and 3.8 rows/block, which is why 1024x576 and 4K had disagreed), and
   > **#503** (`perf/bps-rows-default`, the derived rule shipped:
   > `ROWS_PER_BLOCK_MIN = 4`). It is a pure COST change with **zero** count
   > change — registrations identical at every rung, only `pct_row_wide`
   > 72.90% -> 0.00% at five sites — and it moves ours/dav1d **2.09-2.27x ->
   > 1.60-1.71x** on the cells this campaign had been mis-quoting, with no
   > regression on any of 17 cells. `docs/SHARD_SIZE_SWEEP.md`,
   > `docs/BPS_ROWS_DEFAULT.md`.
   >
   > **The word "contention" in the sentence below is REFUTED as the mechanism**
   > (#504, `docs/C256_CONTENTION.md` §7; and this file's own §2 already pointed
   > that way — `lock_slow` is 1.19% of busy self time at t=8, not the bulk).
   > Directly counted on `c256x2048` t=8: contended acquisitions are **0.264%**
   > of registrations, `lock_slow` **10.7%** of the tracker's CPU, and a
   > *perfect* lock would move that cell 2.378x -> 2.265x. Meanwhile the
   > registration count is **identical** at t=2/4/8 while cost per registration
   > **doubles per doubling of workers** (2.19 -> 4.52 -> 9.18 -> 19.71 ns). So
   > >=89% is the **UNCONTENDED** `add`/`remove` pair paying about one cross-core
   > transfer of the shard's own cache line: a **coherence** cost, not a waiting
   > cost. Four levers have now declined that cell — count (#502, 1.0030),
   > coarser (#501, 0.987-0.995), finer (#504, adverse and monotone), waiting
   > policy (#504, null at n=15 against an in-grid identity control).
   >
   > The item's *geometry* insight survives intact and is what shipped: put
   > adjacent superblock rows in different shard blocks. Only the attribution of
   > *why that helps* changes — from "they stop waiting for each other" to "they
   > stop sharing a cache line".
   >
   > One further caution added since: **cost tracks distinct shard LINES visited,
   > not records filed** (#505). `LfBlock::fill` files 8.98 records per 2.09
   > lines, is 31.7% of the population and 3.9-4.4% of tracker CPU, so the
   > 19.71 ns/registration above is an AVERAGE and must not be used as a marginal
   > price for a count cut. `docs/OWNERSHIP_MODELS.md` §7d already ranks "cut the
   guard cost rather than the guard" first; this round adds the missing reason —
   the cost is **contention**, not volume (`deblock_cols` runs 0.0998 ms per
   execution at t=2 and 0.245 ms at t=8 for identical work) — and narrows it
   further: the contention is *between adjacent superblock rows filtering at
   once*, which is precisely the geometry `block_shift_for` puts in the same
   block. `BLOCKS_PER_SHARD = 2` targets ~4.3 picture rows per block and was
   fitted on whole-frame wall on `v4k_8tile`. A shift ladder re-run **scored on
   the tail** (`TAIL_CONC`, not whole-frame wall) is one build and one decode.
   Consequences of §2 for anything else in this family:
   * A count reduction at the same shard footprint may buy little — held row
     guards measured null for that reason, and `remove` as one `fetch_and`
     measured -0.9%.
   * A count reduction bought with a WIDER extent is refuted three times
     (#469 UB, #475 2.65x slower, #485 decode failure); `docs/BOUNDS_MAP.md`
     prices any new proposal before it is written.
   * **Do not price a per-stage saving from a per-stage cost.** §2's
     super-additivity shows a single stage's measured cost is partly spin that
     reappears elsewhere when that stage is removed.
2. **Price the filter tail's structure.** §4 leaves 9-12% of wall on a chain that
   cannot fill 8 workers even at the ceiling. `OWNERSHIP_MODELS.md` §7d option 2
   is the scheduling answer and is unbuilt and unpriced; `TAIL_CONC` is now the
   instrument for it. The axis to sweep is `sbh` against `n_workers` — 9 rows on
   8 workers is the hard case and 34 rows is visibly easier (19.0% vs 34.1%).
3. **Do NOT re-run the shard-COUNT ladder.** 256 shards is already measured worse
   at every thread count on `v4k_8tile`
   (`benchmarks/scaling_shards_2026-08-08.tsv`, 1.0295x at t=8). The lever in (1)
   is the block SHIFT, which is a different knob.
4. **Separately, and bigger: §6.** Not this campaign, but nothing else in the
   record is worth 10.9 ms/frame at 4K.

## Coverage — what this does NOT cover

* **aarch64 only** (M4 Pro, macOS 26.5.2). No x86_64, no wasm32.
* **8-bit YUV420 only**, two vectors, both **all-intra key frames re-decoded**.
  No inter prediction, no reference management. **Loop restoration and
  super-resolution execute zero blocks** on both, so a null from either here is
  not a result.
* **t = 1, 2, 4, 8.** No t=16. The 4-tile cell only from the prior record.
* **8 tiles (4x2) only.** The quantity §4 says matters — `sbh` vs `n_workers` —
  is unswept; the two cells differ in it (9 vs 34) but also in everything else.
* **Default features.** No `asm`, `c-ffi`, `unchecked`, `unsafe-asm`.
* **The occupancy histogram counts a spinning worker as ACTIVE.** A worker inside
  `TinyLock::lock_slow` is inside a stage body, so occupancy over-states useful
  work by whatever `sync` measures (1.19% of busy self time at t=8, 1024x576).
* **The probe adds a 9th thread** (the 50 us sampler) to an 8P+4E box; its effect
  on the occupancy numbers is unquantified. The wall/CPU and inloop grids use
  binaries without it.
* **`--inloopfilters` changes output pixels.** Attribution only — never an md5
  comparison across its values.
* **A profile attempt wasted for a reason the brief already documented.** The
  first symbol-level spin test drove `bench_ivf_limit`, which stops at end of
  stream (200 frames = 0.66 s at t=8) and so cannot outlive a sample window — all
  four cells reported `outlived_window=0` and produced no `.sample`.
  `docs/AGENT_BRIEF.md` §7 already names the fix ("`examples/profile_ivf` takes
  `RAV1D_THREADS` (default 1) — needed because `bench_ivf_limit` exits before
  `sample` can attach"). Re-run with `profile_ivf` and it worked first time. The
  lesson is the brief's own §"DOCS: SEARCH before acting": grep the brief for the
  tool before building around a limitation.
* **dav1d's speedups here are measured** (§5, §6, n=7, idle box). Where a dav1d
  number from `benchmarks/size_sweep_t8_*` is quoted it is labelled as the prior,
  load-tagged record.
