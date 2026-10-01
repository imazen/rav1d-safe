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
fewer workers and the working set multiplies); that is why it is opt-in. A size-aware
automatic choice would be a sensible follow-up.

## Where this points

1. ~~Per-row guard splitting for single-tile frames~~ and
2. ~~frame threading for tracked builds~~: done, see "What was fixed". Remaining
   there: a size-aware automatic frame-delay choice (big frames get slower with it).
3. **Tracker at 1 thread:** 12-24%; the lock RMW is only a sixth of it, so a serial
   backend needs a different data structure.
4. Kernel targets for stills unchanged: docs/SAFE_VS_DAV1D.md.
