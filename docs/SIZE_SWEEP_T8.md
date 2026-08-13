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

---

# Results

n=9 complete rounds. **All 1,944 rows are load-tagged** (foreign_max ranged 1-14
across rounds; rounds 4-6 saw the worst). Absolutes are inflated; the paired
within-round ratios below are the statistic to read, and where a band is wide it
is printed wide rather than trimmed.

Full tables: `benchmarks/size_sweep_t8_report_2026-08-10.txt`.
Per-cell TSV: `benchmarks/size_sweep_t8_gap_2026-08-10.tsv`.
Raw rows: `benchmarks/size_sweep_t8_raw_2026-08-10.tsv.zst`.

## 1. On the ladder as encoded, threading is worth 0-16% and never more

rav1d-safe, YUV420, single tile, ms/frame (median of 9) and cores busy:

| size | bpc | t=1 | t=2 | t=4 | t=8 | best speedup | cores @t8 | CPU/decode @t8 |
|---|---|---|---|---|---|---|---|---|
| 64x36 | 8 | **0.047** | 0.064 | 0.065 | 0.068 | **none (t=1 wins)** | 1.15 | **1.75x** |
| 256x144 | 8 | **0.618** | 0.633 | 0.651 | 0.653 | **none (t=1 wins)** | 1.11 | 1.17x |
| 512x288 | 8 | 2.909 | **2.822** | 2.836 | 2.900 | 1.03x @t2 | 1.09 | 1.08x |
| 1024x576 | 8 | 16.267 | 14.800 | **14.656** | 14.733 | **1.16x @t4** | 1.19 | 1.08x |
| 2048x1152 | 8 | 67.778 | 56.481 | **55.778** | 56.333 | 1.11x @t4 | 1.14 | 1.04x |
| 3840x2160 | 8 | 202.643 | 184.929 | **183.857** | 186.857 | 1.11x @t2 | 1.13 | 1.06x |
| 64x36 | 10 | **0.063** | 0.084 | 0.082 | 0.083 | **none** | 1.12 | 1.55x |
| 256x144 | 10 | **0.705** | 0.723 | 0.751 | 0.731 | **none** | 1.09 | 1.13x |
| 512x288 | 10 | 3.202 | **3.107** | 3.189 | 3.136 | 1.03x @t2 | 1.08 | 1.07x |
| 1024x576 | 10 | 17.656 | **16.133** | 16.489 | 16.433 | 1.10x @t2 | 1.18 | 1.10x |
| 2048x1152 | 10 | 69.444 | 64.148 | **63.519** | 63.667 | 1.07x @t2 | 1.14 | 1.04x |
| 3840x2160 | 10 | 223.857 | 207.071 | **206.071** | 206.929 | 1.08x @t2 | 1.14 | 1.04x |

**`cores busy` never exceeds 1.19 at any size or any thread count.** Asking for
8 threads and asking for 2 produce the same wall time and the same CPU, because
the second task is the only extra work there is. The two-task reading of the
source is confirmed by the instrument.

**Below 0.15 MP threading is a straight loss.** At 64x36 t=2 is **1.36x SLOWER**
than t=1 and burns **1.75x the CPU**; at 256x144 it is 2-6% slower for 9-17%
more CPU. That is thread coordination with nothing to coordinate — the frame is
1 to 12 superblocks, so the filter chain never gets far enough behind the tile
task to overlap with it. dav1d pays the same toll (0.65x at 64x36 t=8).

## 2. The mechanism, checked from the other direction

Inverting the measured speedup through the two-stage model
(`B/(A+B) = 1 - 1/S`, `scripts/perf/size_sweep_t8_amdahl.py`) says the
overlappable minority stage is **8.4-13.7% of decode** for rav1d-safe across
512x288..4K, and **3.3-8.6%** for dav1d.

`docs/SIZE_SWEEP.md`'s independent profile puts loopfilter + cdef at
**(1.53+1.09)/27.42 = 9.6%** of ms/MP at 1024x576 and **(1.48+0.37)/24.0 =
7.7%** at 4K. Two instruments, four days and two base commits apart, agree.
The filter chain IS the whole of the available parallelism, and it is small
here because **loop restoration is off in every vector on this ladder** — a
stream with LR active has more to overlap and would do better.

## 3. Tiling the encode is the fix, and it is not decoder-side

Same content, same encoder, same quality; the only change is
`--tilecolslog2`/`--tilerowslog2`. rav1d-safe, YUV420 8bpc:

| cell | tiles | t=1 | t=2 | t=4 | t=8 | speedup @t8 | cores @t8 | CPU/decode @t8 | bytes |
|---|---|---|---|---|---|---|---|---|---|
| 1024x576 | 1 | 16.267 | 14.800 | 14.656 | 14.733 | **1.11x** | 1.19 | 1.08x | — |
| 1024x576 | 4 | 16.044 | 9.222 | 5.922 | 5.422 | **2.91x** | 3.49 | 1.26x | +0.64% |
| 1024x576 | 8 | 16.067 | 9.256 | 5.089 | **4.189** | **3.89x** | 5.48 | 1.46x | +1.13% |
| 2048x1152 | 1 | 67.778 | 56.481 | 55.778 | 56.333 | 1.09x | 1.14 | 1.04x | — |
| 2048x1152 | 8 | 62.519 | 33.963 | 18.481 | **12.667** | **4.96x** | 6.36 | 1.32x | +0.57% |
| 3840x2160 | 1 | 202.643 | 184.929 | 183.857 | 186.857 | 1.07x | 1.13 | 1.06x | — |
| 3840x2160 | 8 | 200.786 | 106.429 | 57.786 | **38.286** | **5.20x** | 6.86 | 1.30x | +0.38% |
| 3840x2160 10b | 8 | 234.071 | 123.429 | 65.714 | **42.929** | **5.41x** | 6.89 | 1.26x | +0.39% |

**A 4K still goes from 187 ms to 38 ms — 4.9x lower latency at the same thread
count — for 0.38% more bytes.** The single-tile t=1 and multi-tile t=1 columns
agree to 1-8%, so the tiling costs essentially nothing when decoded serially:
this is pure upside for a threaded decoder and it is bought in the ENCODER.

**Efficiency of the tiled path is good, not free.** At 4K t=8 we buy 5.20x
latency for 1.30x CPU per decode (S/C = 4.0) — the extra cores are more than
paying for themselves. The 1.30x is the honest overhead of tile boundaries plus
coordination; it is not the 8x that "8 threads" naively suggests, because the
worker threads park rather than spin (`ttd.cond.wait`, `src/thread_task.rs:904`)
so idle capacity costs nothing.

## 4. Where we stand against dav1d on both axes

Paired within round, rav1d-safe / dav1d, YUV420 8bpc:

| cell | tiles | wall t=1 | wall t=8 | CPU t=1 | CPU t=8 |
|---|---|---|---|---|---|
| 512x288 | 1 | 1.122 | 1.129 | 1.122 | 1.206 |
| 1024x576 | 1 | 1.452 | 1.360 | 1.444 | **1.525** |
| 2048x1152 | 1 | 1.523 | 1.420 | 1.520 | **1.559** |
| 3840x2160 | 1 | 1.268 | 1.221 | 1.268 | **1.342** |
| 1024x576 | 8 | 1.427 | **2.04** | 1.427 | — |
| 2048x1152 | 8 | 1.51 | **1.65** | — | — |
| 3840x2160 | 8 | 1.273 | **1.485** | 1.274 | 1.427 |

Two opposite movements, and both matter:

* **On single-tile streams our WALL ratio improves with threads** (1.452 ->
  1.360 at 1024x576) — only because our filter stage is bigger, so we have more
  to hide. **Our CPU ratio gets worse over the same step** (1.444 -> 1.525). We
  are not catching up; we are spending more cores to look closer.
* **On tiled streams our ratio gets clearly worse with threads**: 4K goes
  1.273 at t=1 to **1.485** at t=8 (dav1d 6.23x vs our 5.20x), and 1024x576
  goes 1.43 to 2.04. **Tile-thread scaling is a real and separate deficit from
  the single-thread gap**, and it only becomes visible on a bitstream that has
  tiles — which is why fifteen rounds on `v4k_8tile` saw it and the ladder
  never could.

## 5. dav1d's default mode is a capability we do not have

`dav1d_def` reaches 2.7-3.3x on SINGLE-TILE cells (e.g. 1024x576 8bpc:
11.22 -> 3.50 ms at t=8, cores 3.59, CPU only 1.13x of t=1). That is frame
threading pipelining independent frames out of the IVF, not stage overlap —
which is exactly why the two-stage model returns a nonsense 46-70% for it.

It is not a like-for-like latency number: it decodes several frames at once, so
it cannot reduce the latency of ONE image. For a still-image server that is
irrelevant (you get the same effect from N processes, and
`docs/SIZE_SWEEP.md` already measured that at 8.22x for N=16 against dav1d's
8.84x). For single-stream video it is a genuine gap: the checked build pins
`n_fc = 1` (`src/lib.rs:127`) because frame threading needs `unchecked`.

---

# The decision table

For each cell: the thread count that minimises latency, the point beyond which
threads stop buying anything, and the CPU price. Rules are pre-registered in
`scripts/perf/size_sweep_t8_report.py`'s docstring.

| cell (YUV420) | tiles | best t | latency there | gain vs t=1 | threads stop helping after | CPU @t8 vs t=1 | verdict |
|---|---|---|---|---|---|---|---|
| 64x36 8b | 1 | **1** | 0.047 ms | — | t=1 | 1.75x | **never thread** |
| 64x36 10b | 1 | **1** | 0.063 ms | — | t=1 | 1.55x | **never thread** |
| 256x144 8b | 1 | **1** | 0.618 ms | — | t=1 | 1.17x | **never thread** |
| 256x144 10b | 1 | **1** | 0.705 ms | — | t=1 | 1.13x | **never thread** |
| 512x288 8b | 1 | 2 | 2.822 ms | 1.03x | t=2 | 1.08x | not worth it |
| 512x288 10b | 1 | 2 | 3.107 ms | 1.03x | t=2 | 1.07x | not worth it |
| 1024x576 8b | 1 | **2** | 14.80 ms | 1.11x | **t=2** | 1.08x | t=2 only |
| 1024x576 10b | 1 | **2** | 16.13 ms | 1.10x | **t=2** | 1.10x | t=2 only |
| 2048x1152 8b | 1 | **2** | 56.48 ms | 1.09x | **t=2** | 1.04x | t=2 only |
| 2048x1152 10b | 1 | **2** | 64.15 ms | 1.07x | **t=2** | 1.04x | t=2 only |
| 3840x2160 8b | 1 | **2** | 184.9 ms | 1.11x | **t=2** | 1.06x | t=2 only |
| 3840x2160 10b | 1 | **2** | 207.1 ms | 1.08x | **t=2** | 1.04x | t=2 only |
| 1024x576 8b | 4 | **8** | 5.42 ms | 2.91x | t=4 | 1.26x | thread to 4 |
| 1024x576 8b | 8 | **8** | 4.19 ms | 3.89x | t=4 | 1.46x | thread to 8 |
| 1024x576 10b | 8 | **8** | 4.59 ms | 3.84x | t=4 | 1.47x | thread to 8 |
| 2048x1152 8b | 8 | **8** | 12.67 ms | 4.96x | t=4 | 1.32x | thread to 8 |
| 3840x2160 8b | 8 | **8** | 38.29 ms | 5.20x | **t=8** | 1.30x | thread to 8 |
| 3840x2160 10b | 8 | **8** | 42.93 ms | 5.41x | **t=8** | 1.26x | thread to 8 |

Read plainly:

1. **On the bitstreams `avifenc` produces by default, there is no thread count
   worth using above 2, at any size, at either depth** — and below 0.15 MP
   there is none worth using above 1. The ceiling is 1.16x and it is structural.
2. **The lever is the encoder.** Eight tiles turn a 4K still from 187 ms to
   38 ms for +0.38% bytes, and 1024x576 from 14.7 ms to 4.2 ms for +1.13%.
3. **Threading is cheap in CPU terms wherever it works at all** (1.26-1.46x per
   decode for 2.9-5.4x latency; S/C = 2.3-4.3) and expensive wherever it does
   not (1.75x CPU for a 1.36x SLOWDOWN at 64x36). The bad trades are all at the
   small end and all on single-tile input.
4. **Our tile-thread scaling trails dav1d's** — 5.20x vs 6.23x at 4K/8-tile,
   pushing the ratio from 1.27 to 1.49. That is a separate target from the
   single-thread gap and it is invisible on this ladder's own vectors.

## What this round did NOT settle

* **The box was never idle.** All 1,944 rows carry foreign load (see the
  measlock defect above). A clean-box repeat of the multi-tile t=8 cells is the
  one measurement most likely to move: those are the cells where contention
  bites hardest, so **5.20x at 4K/8-tile is a LOWER bound**.
* **4:4:4 was not swept at t>1**, and no 4096x2304 vector exists.
* **LR is off everywhere here**, so the overlappable stage is deblock+CDEF only.
  Every single-tile speedup above is a lower bound for LR-active content.
* **Why our tiled scaling trails dav1d's was not profiled.** This round measures
  the deficit; it does not attribute it.
