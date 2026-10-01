# Comparing rav1d-safe with dav1d and libgav1 (2026-10-01)

Reproduce with `scripts/perf/decoder_bench/` (harness + driver). dav1d 1.5.3 and
libgav1 `main` (0850323, 2026-09-24) were built from source (gcc, -O3, dav1d with
its asm). All three decoders produce byte-identical output on the streams used
(md5 checked). Machine: Zen 4 (AVX-512), 1-8 pinned cores.

## Protocol (what "fair" means here)

An earlier comparison timed the competitors' CLIs by whole-process wall time
(startup, file read, output handling) against our in-process decode time. That was
unfair, by about 1% on 150-frame streams and more on short ones. The harness now
treats every decoder identically:

- IVF parsed into memory once; a **fresh decoder per pass** (creation and thread
  spawn counted for everyone); every picture drained and released; the clock runs
  **inside the process, around the pass only**;
- one untimed warm-up pass, then 5 timed passes per process;
- contestants **interleaved in rotating order** over 4 rounds, **pinned** to fixed
  cores (not core 2, which hosts other processes here; one foreign process pins a
  core at 100%), frame counts asserted equal, median reported;
- every contestant also run **capped at AVX2** (libgav1 has no AVX-512); this
  changed little (dav1d +1-2%, ours +-1%), so AVX-512 is not what separates them;
- two threading modes, because they measure different things:
  **tile** (frame delay 1: tile/post-filter threading only) and
  **auto** (each decoder's default parallelism: dav1d `max_frame_delay=0`,
  libgav1 `frame_parallel`, rav1d `RAV1D_FRAME_DELAY=0`).
  The first version of this comparison used only `tile` and so understated how
  far every decoder scales on small frames.

Remaining differences not removed: compilers (gcc vs rustc/LLVM; not yet re-run
with clang), our `decode()` copies its input (about 14 us per MB, negligible),
dav1d/libgav1 zero-copy it.

## Results (ms/frame, median)

### 1 thread

| stream | dav1d | libgav1 | rav1d untracked | rav1d default |
|---|---:|---:|---:|---:|
| 4K photo AVIF (40 frames) | 24.5 | 39.9 | 36.9 | 42.2 |
| 4K intra IVF (16) | 24.9 | 40.0 | 37.3 | 42.6 |
| 480p inter (150) | 1.45 | 2.59 | 3.63 | 4.85 |
| small intra (39) | 1.98 | 2.94 | 2.81 | 3.22 |

### 4 threads, tile threading only (frame delay 1)

| stream | dav1d | libgav1 | untracked | default |
|---|---:|---:|---:|---:|
| 4K photo | 7.3 | 11.3 | 10.6 | 17.5 |
| 4K intra | 7.4 | 11.2 | 11.3 | 18.2 |
| 480p inter | 1.46 | 1.68 | 3.57 | **7.72** |
| small intra | 2.07 | 2.54 | 2.94 | 3.36 |

### 4 threads, each decoder's default parallelism (auto)

| stream | dav1d | libgav1 | untracked | default |
|---|---:|---:|---:|---:|
| 4K photo | 7.4 | 11.2 | 11.1 | 16.9 |
| 4K intra | 7.6 | 11.2 | 12.2 | 17.7 |
| 480p inter | **0.96** | 1.71 | **1.61** | 7.92 |
| small intra | 1.64 | 2.14 | 2.10 | 3.69 |

Reading it: libgav1 is 1.5-1.8x slower than dav1d. **rav1d untracked is level
with libgav1** (0.92-1.00x on 4K stills at 1-4 threads; 0.94-1.09x in auto mode)
and 1.3-1.7x behind dav1d. The tracked default is 1.1x libgav1 on stills at 1
thread, then falls behind as threads are added.

## Why the default build gets slower with threads (480p inter, ms/frame wall / CPU)

| arm | t=1 | t=2 | t=4 | t=8 |
|---|---:|---:|---:|---:|
| default (tracker + compact copy) | 4.83 / 4.82 | 7.60 / 8.79 | 7.70 / 8.93 | 7.70 / 8.96 |
| tracker off, compact copy kept (`__probe_untracked`) | 3.65 / 3.64 | 4.30 / 5.10 | 4.34 / 5.15 | 4.53 / 5.36 |
| untracked (no tracker, zero-copy) | 3.65 / 3.65 | 3.53 / 3.75 | 3.58 / 3.83 | 3.74 / 4.02 |

- At 1 thread the tracker costs **1.18 ms (24%)**.
- Going to 2+ threads the default build adds **4.1 ms**, and only **0.8 ms of that is
  the copy path itself**. The other **3.4 ms is tracker cost**, about 2.8x what it
  was at 1 thread: `uses_row_guards()` is true for any multi-threaded decoder, regardless of
  how many tile columns the frame has, so every block is split into per-row guards (read then
  write-back), multiplying registrations. It runs on the serial recon thread, so it
  lands in wall time almost one for one (CPU/wall is only 1.17).
- Untracked is flat from 1 to 8 threads in tile mode: a small frame is essentially serial there
  (dav1d too: 1.45 -> 1.46); the stream's tile count was not checked.
- **Frame threading is the real lever** on small frames: untracked with
  `RAV1D_FRAME_DELAY=4` goes 3.58 -> 1.43 ms at 4 threads (2.5x), and dav1d's auto
  mode 1.46 -> 0.73. The tracked default cannot frame-thread (`n_fc` is clamped
  to 1), which is why it is 4.9x behind untracked on this stream.

## A safe single-thread backend?

At 1 thread no data race is possible (the managed `Decoder` spawns no workers),
so only same-thread aliasing from a decoder bug remains; with `PlainData` elements
and bounds checks that costs wrong pixels, not memory safety. The measured prize is
**12% on 4K stills and 24% on small inter frames** (tracked -> untracked, 1 thread).

It is not free to get safely. A cheap experiment, replacing the tracker's single
lock's atomic swap by a plain load/store (unsound for threads, only an upper bound
for a serial lock), recovered **only 15% of the tracker cost on 480p inter and none
on 4K**. The cost is the bookkeeping (record stores, slot allocation, publish and
retire, shard arithmetic), not the RMW. So a serial backend must be a different
data structure (for example a tiny live-interval list scanned linearly; one thread
has few simultaneously live guards), not the sharded tracker minus its atomics.
Selecting it at runtime also needs care: `DisjointMut` is shared across worker
threads in other modes, and the main crate is `forbid(unsafe_code)`, so the
single-thread invariant has to be enforced inside the crate (or the knob left as
the compile-time `untracked` feature).

## What was fixed (2026-10-01, same day)

Two changes to tracked builds, both validated with the live tracker as oracle (any
real overlap would have panicked):

1. **No per-row guard splitting for single-tile frames, and a single-shard tracker
   layout for small single-tile planes (< 2 MiB).** At 2 threads the 480p stream's
   instructions drop 12.83 G -> 7.87 G (1 thread: 7.30 G); the leftover +8% is the
   multi-thread machinery itself. Multi-tile frames are unchanged. Validation:
   803/803 vectors at 2/4/8 threads and 640 stress runs of the 40 largest vectors,
   zero overlap panics, zero md5 mismatches.
2. **Frame threading in tracked builds, opt-in** (explicit `max_frame_delay > 1`;
   auto stays tile-only so `decode()` does not turn asynchronous unasked). The old
   clamp was removed on the strength of: 803/803 vectors at (2 threads, delay 2),
   (4, 3), (8, 4); 640 stress runs; the tracked film-grain frame-context md5 test.

The suspicion that this was a tracking-code regression or spinlocking was checked:
`v0.6.0` and `main` before this branch show the same 1 -> 2 thread jump, and the
profile has 2 of 6,120 samples in `lock_slow` (0.03%) at 2 threads (2-3% at 8
threads on 4K, before and after, on a loaded box). The cost was policy: per-row
guards multiplying registrations, then the sharded layout turning each hull into
a multi-shard/wide registration.

Tracked default build, ms/frame, interleaved x7, **box loaded (load ~16) so read
only the ratios**:

| stream | before | fixed (tile threads) | fixed + frame threads (d=4) |
|---|---:|---:|---:|
| 480p inter, t=1 | 5.20 | 5.18 | 5.20 |
| 480p inter, t=2 | 10.55 | 6.69 | **4.13** (2.6x) |
| 480p inter, t=4 | 11.92 | 7.90 | **4.29** (2.8x) |
| 480p inter, t=8 | 12.88 | 8.67 | **4.23** (3.0x) |
| small intra, t=4 | 4.64 | 4.86 | **2.02** (2.3x) |
| 4K photo, t=1 | 45.1 | 45.1 | 45.2 |
| 4K photo, t=4 | 28.8 | 28.6 | 37.6 (**0.77x, worse**) |

Frame threading is a win on small frames and a loss on big ones (each frame gets
fewer workers and the working set multiplies). That is what the size-aware automatic
choice below is for.

## Size-aware automatic frame delay (2026-10-01, later)

`max_frame_delay == 0` with `threads > 1` now resolves from the stream's frame size
(managed `Decoder` opens on first data and reads the sequence header; rule in
`size_aware_frame_delay`, `src/lib.rs`). Evidence, rav1d only, `profile_ivf`,
3 interleaved rounds, **quiet box** (load ~1), ms/frame and ratio to tile-only (delay 1):

| stream | build | t | d1 (tile) | d2 | d4 | d8 |
|---|---|--:|---:|---:|---:|---:|
| 480p inter (40 fr) | untracked | 2 | 3.68 | 2.00 (0.54x) | 2.00 | |
| | tracked | 2 | 5.53 | 3.00 (0.54x) | 3.00 | |
| | untracked | 4 | 4.05 | 2.12 (0.52x) | 2.25 (0.56x) | |
| | tracked | 4 | 6.06 | 3.28 (0.54x) | 4.35 (0.72x) | |
| 720p, 1 tile | untracked | 4 | 0.79 | 0.51 (0.64x) | 0.61 (0.77x) | 0.61 |
| | tracked | 4 | 1.28 | 0.87 (0.68x) | 1.05 (0.82x) | 1.05 |
| | untracked | 8 | 0.82 | 0.61 (0.74x) | 0.66 (0.81x) | 0.85 (1.04x) |
| | tracked | 8 | 1.27 | 0.97 (0.76x) | 1.09 (0.86x) | 1.21 (0.95x) |
| 1080p, 1 tile | untracked | 4 | 2.03 | 1.38 (0.68x) | 1.45 (0.72x) | 1.51 |
| | tracked | 4 | 4.02 | 2.73 (0.68x) | 2.69 (0.67x) | 2.68 |
| | untracked | 8 | 2.07 | 1.48 (0.72x) | 1.57 (0.76x) | 2.03 (0.98x) |
| | tracked | 8 | 4.20 | 2.89 (0.69x) | 2.99 (0.71x) | 3.10 (0.74x) |
| 1080p, 4 tile cols | untracked | 4 | 1.25 | 1.18 (0.94x) | 1.43 (1.14x) | 1.20 |
| | tracked | 4 | 2.50 | 2.65 (1.06x) | 2.79 (1.12x) | 2.79 |
| | untracked | 8 | 1.22 | 1.10 (0.90x) | 1.22 (1.00x) | 1.58 (1.29x) |
| | tracked | 8 | 2.49 | 2.28 (0.91x) | 2.24 (0.90x) | 2.65 (1.06x) |
| 4K photo (40 fr) | untracked | 4 | 10.7 | 12.3 (**1.15x**) | 12.3 | |
| | tracked | 4 | 17.3 | 24.5 (**1.42x**) | 24.8 | |
| | untracked | 8 | 11.2 | 7.34 (**0.65x**) | 7.91 (0.70x) | |
| | tracked | 8 | 16.0 | 15.9 (0.99x) | 16.2 | |
| 4K intra IVF (16 fr) | untracked | 4 | 11.3 | 13.3 (1.18x) | 14.3 (1.26x) | |
| | tracked | 4 | 18.0 | 26.1 (1.45x) | 28.1 (1.56x) | |
| | untracked | 8 | 11.6 | 9.06 (0.78x) | 9.75 (0.84x) | |
| | tracked | 8 | 16.7 | 18.0 (1.08x) | 19.5 (1.17x) | |

The 720p/1080p streams are 30-frame panning inter clips cropped from the 4K photo and
encoded with `aomenc` (`--cpu-used=6 --cq-level=32 --lag-in-frames=0`; the 4-tile one
with `--tile-columns=2`). Reading it:

- Tile-mode decoding of a single-tile frame below 4K does not use more than a couple
  of threads (720p/1080p are flat from 4 to 8 threads), so a **second frame in flight
  gives 0.54-0.76x**; deeper pipelines (d4, d8) add working set and are equal or worse
  in nearly every row. The cap is **2**.
- **4K stills** scale in tile mode to 4 threads (37 ms at 1 -> 11 ms at 4 untracked)
  and then stall (11.2 ms at 8); there a second frame loses 1.15-1.45x at 4 threads
  and wins 0.65-0.78x at 8, but only untracked (tracked 0.99-1.17x at 8).
- A 4-tile 1080p frame is the weak case: tile threading already scales, a second frame
  is neutral untracked and 1.06x worse tracked at 4 threads (0.91x better at 8). The
  frame header's tile count is not available when the context is opened, so this is
  accepted rather than special-cased.

The rule: 1 thread -> 1; frames under 6 M luma pixels -> 2; larger -> 1, or 2 for
`untracked` at 8+ threads. Validated with the live tracker as oracle: 803/803 vectors
at 2, 4 and 8 threads with `--delay 0` in both the tracked and `untracked` builds
(small conformance streams now frame-thread at 2 contexts), plus tests
(`src/managed/frame_delay_tests.rs`, mutation-verified). Not verified above 8 threads;
the cap of 2 is deliberately not extrapolated.

### Three decoders, each in its default parallelism (auto), after this change

dav1d 1.5.3 (`max_frame_delay=0`), libgav1 `main` (`frame_parallel`), rav1d with the
size-aware auto delay. Same harness and protocol as above, `FIRST_CPU=0`, 4 rounds x 5
passes. **The box was heavily shared (load ~20, someone else's 12-thread training jobs on
cores 4-23) so absolute times are ~1.5x inflated for every decoder and the multi-thread
rows are noisy: read the ratios.** In particular the tracked build's lock-based tracker
suffers more under oversubscription than the others. A quiet-box rerun is still owed.
Median ms/frame, ratio to dav1d in parentheses:

| stream | t | dav1d | libgav1 | rav1d untracked | rav1d default |
|---|--:|---:|---:|---:|---:|
| 480p inter (150) | 1 | 2.18 | 4.12 (1.89x) | 6.08 (2.79x) | 8.19 (3.76x) |
| | 4 | 1.17 | 2.92 (2.51x) | 2.94 (2.52x) | 4.45 (3.82x) |
| | 8 | 1.25 | 2.59 (2.08x) | 3.28 (2.63x) | 4.78 (3.84x) |
| 720p (30) | 1 | 0.705 | 1.567 (2.22x) | 1.448 (2.05x) | 2.049 (2.91x) |
| | 4 | 0.440 | 0.848 (1.93x) | 0.773 (1.75x) | 1.446 (3.28x) |
| | 8 | 0.611 | 0.863 (1.41x) | 0.944 (1.54x) | 1.764 (2.89x) |
| 1080p (30) | 1 | 2.054 | 3.561 (1.73x) | 3.783 (1.84x) | 5.390 (2.62x) |
| | 4 | 1.149 | 2.213 (1.93x) | 2.070 (1.80x) | 4.169 (3.63x) |
| | 8 | 1.364 | 2.200 (1.61x) | 2.562 (1.88x) | 5.636 (4.13x) |
| 1080p 4-tile (30) | 1 | 2.061 | 3.614 (1.75x) | 3.952 (1.92x) | 5.697 (2.76x) |
| | 4 | 1.058 | 1.738 (1.64x) | 1.916 (1.81x) | 5.620 (5.31x) |
| | 8 | 1.036 | 1.871 (1.81x) | 1.732 (1.67x) | 4.159 (4.02x) |
| 4K photo (40) | 1 | 36.2 | 66.7 (1.84x) | 61.8 (1.70x) | 61.7 (1.70x) |
| | 4 | 11.09 | 18.89 (1.70x) | 17.38 (1.57x) | 30.09 (2.71x) |
| | 8 | 7.18 | 19.05 (2.65x) | 10.97 (1.53x) | 26.98 (3.76x) |

Tile mode only (frame delay 1 for everyone), 4 threads, ratio to dav1d: 720p libgav1
1.04x, untracked 1.70x, default 2.81x; 1080p libgav1 0.96x, untracked 1.76x, default
3.53x; 1080p 4-tile libgav1 1.73x, untracked 2.00x, default 4.50x.

What this adds to the earlier picture:

- **rav1d untracked is level with libgav1 or better in auto mode** on 720p/1080p and
  4K (1.5-1.9x dav1d against libgav1's 1.4-2.7x), and the lead grows with threads on 4K
  (libgav1 stalls at 19 ms from 4 to 8 threads; untracked reaches 11 ms).
- In **tile mode** libgav1 is level with dav1d on single-tile 720p/1080p (it spends its
  threads inside a frame well), and rav1d is 1.7-1.8x behind there: the size-aware
  frame delay recovers that gap on those streams (untracked 720p 1.28 -> 0.77 ms,
  1080p 3.37 -> 2.07 ms at 4 threads), but dav1d and libgav1 also gain from frame
  threading, so the ratio to dav1d stays about 1.7-1.8x in both modes.
- The tracked default build is 2.6-5.3x dav1d on these streams. The multi-tile 1080p row
  (5.31x at 4 threads, noisy: min 4.2 ms) is the one case where the new auto delay is
  slightly worse than tile mode for the tracked build (see the sweep above).

## Where this points

1. ~~Per-row guard splitting for single-tile frames~~ and
2. ~~frame threading for tracked builds~~ and ~~a size-aware automatic frame-delay
   choice~~: done, see "What was fixed" and "Size-aware automatic frame delay".
3. **Tracker at 1 thread:** 12-24%; the lock RMW is only a sixth of it, so a serial
   backend needs a different data structure.
4. Kernel targets for stills unchanged: docs/SAFE_VS_DAV1D.md.
