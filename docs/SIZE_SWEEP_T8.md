# Per-image latency vs CPU burned, across the size ladder x thread count

The unfinished half of `docs/SIZE_SWEEP.md` (its own §"Not measured" says
*"t=1 for the whole ladder; only 1024x576 and 3840x2160 were also taken at t=8,
and only at 4:2:0"*). Measure-only round: the diff is measurement scripts, one
example pulled forward, docs and data.
`git diff 0f6bf10..HEAD -- src/ lib.rs include/ crates/ Cargo.toml Cargo.lock build.rs`
is empty.

The question this answers is a product question with two halves, and the second
half had never been measured at all: **per-image latency matters, and so does
the CPU burned to get it.** A latency win bought with cores is a throughput loss
for an image server, so every cell below reports wall AND user+sys.

---

## Read this first: the whole ladder is ONE TILE, and that caps everything

AV1 tile threading cannot exceed the tile count. Before timing anything, the
tile layout was read out of the frame header — not the encoder recipe, the
bitstream — by `scripts/perf/tile_layout.py`
(`uncompressed_header()` -> `tile_info()`, spec 5.9.15, reduced-still form only;
it refuses anything else rather than guessing).

| size | sb cols x rows | tile cols x rows | tiles |
|---|---|---|---|
| 64x36 | 1 x 1 | 1 x 1 | **1** |
| 256x144 | 4 x 3 | 1 x 1 | **1** |
| 512x288 | 8 x 5 | 1 x 1 | **1** |
| 1024x576 | 16 x 9 | 1 x 1 | **1** |
| 2048x1152 | 32 x 18 | 1 x 1 | **1** |
| 3840x2160 | 60 x 34 | 1 x 1 | **1** |

All 24 ladder vectors (6 sizes x {420,444} x {8,10} bpc) read 1 tile.
Data: `benchmarks/size_sweep_tile_layout_2026-08-10.tsv`.

**Teeth on the parser**, because one that could only ever print 1 would prove
nothing: the campaign's own `v4k_8tile.avif` reads **4 x 2 = 8** and
`v4k_1tile_10b.avif` reads 1, from the same code path.

### The campaign's existing t=8 numbers are from a different vector class

Every t=8 figure the rav1d-safe campaign has recorded — the 5.88x t=1->t=8
latency ratio at "4K", the 408.63/424.45/475.39 ms stage-body CPU at t=2/4/8 —
was taken on `v4k_8tile`. Decoded and read out rather than assumed:

| vector | layout | tiles |
|---|---|---|
| `v4k_8tile` (campaign) | **I444** | **8** (4 x 2) |
| `L3840x2160_420_8b` (this ladder) | **I420** | **1** |

Different on BOTH axes. So "4K scales 5.88x to t=8" and "4K scales 1.1x to t=8"
are not in contradiction — they are two different bitstreams, and the ladder's
one is the shape an `avifenc` default still actually has.

### What is left to parallelise when there is one tile

Read from source, not assumed. Per frame pass rav1d-safe enqueues:

* `rav1d_task_create_tile_sbrow` (`src/thread_task.rs:448`) —
  `num_tasks = frame_hdr.tiling.cols * frame_hdr.tiling.rows`
  (`src/thread_task.rs:456`), i.e. **one** tile task at 1 tile;
* `create_filter_sbrow` (`src/thread_task.rs:398`) — **one** task, which walks
  the sb rows running Deblock -> CDEF -> SuperRes -> LR as a single fallthrough
  chain (`src/thread_task.rs:1340-1462`) and re-queues itself for the next row.

So on a single-tile frame there are **two** concurrently-runnable tasks, and the
filter chain lags the tile task by at least one superblock row. Whatever t>1
buys at these sizes is the overlap of those two stages — a two-stage pipeline —
and no thread count can raise that ceiling. The measured `cores busy`
(CPU/frame ÷ wall/frame) is the direct test of that reading.

Also from source: the checked build pins `n_fc = 1` unconditionally
(`src/lib.rs:127`, "Frame threading (n_fc>1) still requires unchecked"), so
rav1d-safe has tile threading only. That is why `dav1d --framedelay 1` is the
like-for-like arm and dav1d's default is a different capability class, not a
different tuning of the same one.

---

## Method

* Host: Apple M4 Pro (`Mac16,11`, 8P+4E, 24 GB), macOS 26.5.2.
  `rav1d-safe` at `main` **0f6bf10** (this round's base; it is 12 commits and
  one merge past `b0a00c3`, the base of `docs/SIZE_SWEEP.md`, so absolutes here
  are NOT comparable to that document's).
* dav1d **1.5.4** (homebrew), in the SAME interleaved sweep, arm order rotating
  per round, in two configurations:
  * `dav1d_fd1` — `--framedelay 1`, tile threading only. **This is the
    like-for-like latency arm**: same threading model our build implements.
  * `dav1d_def` — dav1d's default frame delay. On an IVF of independent still
    frames this lets dav1d pipeline whole FRAMES, which is a throughput
    capability our build does not have. Reported because the task asked for it,
    and labelled as such: **its speedup is not a single-image latency number.**
* One instrument for both quantities: bash's `time` keyword
  (`TIMEFORMAT='%3R %3U %3S'`), i.e. the child's own getrusage as the kernel
  reports it — not a sampler, not `ps`. Every cell is run at TWO frame counts
  and `total = a + b*frames` is fitted for wall and for user+sys separately, so
  process startup (exec, mmap, container parse, decoder construction, thread
  pool spin-up) drops out of both. Frame counts are per cell, from 2,500/25,000
  at 64x36 down to 2/16 at 4K.
* `cores busy` = (CPU ms/frame) / (wall ms/frame). 1.00 means one core; the gap
  between `cores busy` and the speedup is the waste.
* No `nice` on any timed run. No `-C target-cpu=native`. Default features.
  Everything serialised behind `measlock`.
* Ratios and speedups are PAIRED WITHIN A ROUND, then reduced by median with the
  min/max band printed.

### Correctness before timing

All 30 vectors — the 12 ladder cells used here plus the 6 forced-multi-tile ones
built for this round — decode **bit-identically to dav1d 1.5.4** at base
`0f6bf10`: `decode_md5` vs `dav1d --muxer md5`, **30/30 MATCH**,
`benchmarks/size_sweep_t8_vector_md5_2026-08-10.tsv`.

### The forced-multi-tile counterparts

Same source PNG, same encoder, same speed and quality — the only change is
`--tilecolslog2`/`--tilerowslog2`. The byte cost of the tiling is part of the
answer, so it is recorded:

| vector | tiles | bytes | vs single-tile |
|---|---|---|---|
| `L1024x576_420_8b__t4` | 2 x 2 = 4 | 194,770 | **+0.64%** |
| `L1024x576_420_8b__t8` | 4 x 2 = 8 | 195,704 | **+1.13%** |
| `L1024x576_420_10b__t8` | 4 x 2 = 8 | 198,337 | **+1.34%** |
| `L2048x1152_420_8b__t8` | 4 x 2 = 8 | 655,677 | **+0.57%** |
| `L3840x2160_420_8b__t8` | 4 x 2 = 8 | 2,838,071 | **+0.38%** |
| `L3840x2160_420_10b__t8` | 4 x 2 = 8 | 2,878,187 | **+0.39%** |

---

## A `measlock` defect this round tripped over — worth a line in the brief

`measlock`'s `cleanup()` is `rm -rf "$LOCK"` on EXIT, unconditionally. It does
not check that the lock it is deleting is still *its own*. So when holder A's
lock is reclaimed (stale escape, or any other path) and holder B acquires, A's
eventual exit **deletes B's lock**, and the next waiter walks straight in. That
is what happened here: two agents ended up measuring simultaneously at 03:29,
with neither having done anything wrong at its own call site.

Fix shape: have `cleanup()` compare the pid in `$LOCK/owner` with `$$` and only
remove the directory when it matches.

    cleanup() { [ "$(awk '{print $2}' "$LOCK/owner" 2>/dev/null)" = "$$" ] && rm -rf "$LOCK"; }

Consequence for this round, stated plainly: **every row is load-tagged** and the
absolutes are inflated. Paired within-round ratios — speedup vs t=1, CPU
multiplier vs t=1, ours/dav1d at matched t — are the statistics to read.

## Not measured — stated before the results

* **4:4:4.** The thread sweep is 4:2:0 only (the product case for AVIF stills).
  The t=1 4:4:4 ladder is in `docs/SIZE_SWEEP.md`; its thread behaviour is not
  measured here.
* **4096x2304.** The task named it; the ladder's top cell is **3840x2160** and
  no 4096x2304 vector exists. Not extrapolated.
* **12bpc, and 10bpc forced-tile below 1024x576.** Not built.
* **Loop restoration.** `enable_restoration = 0` in every vector on this ladder
  (`docs/SIZE_SWEEP.md` §"The tool set is constant"), so the filter chain that
  t>1 overlaps here is deblock + CDEF only. A stream with LR active has MORE
  filter work to overlap, so the speedups here are a LOWER bound for that case
  and the point is not swept.
* **No inter prediction, one content class, one quality point, aarch64 only** —
  inherited from the ladder, same as `docs/SIZE_SWEEP.md`.
