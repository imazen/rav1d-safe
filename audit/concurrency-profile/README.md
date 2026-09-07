# Borrow usage and concurrency profiles, 2026-09-06

The inter-frame slowdown is dominated by wide reference-picture borrows. The
still-frame workloads have a different bottleneck: many small registrations.
Process history also imposes a large, independently reproduced cost on later
single-worker decoders. No production optimization was applied in this study.

User budget: 30 minutes, extended at 03:34 UTC on September 7. This is a bounded
screen on a Ryzen 9 7900X (12 physical cores, 24 logical CPUs), Rust 1.98.1,
LLVM 22.1.8, perf 7.0.14. Repository release settings: fat LTO, one codegen unit,
line tables, default checked decoder, both bit depths. Commands were serialized
under `run-heavy --mem 16G --jobs 8` (nice 19, cgroup cap). CPU affinity was not
pinned. Treat small differences and noisy oversubscribed cells cautiously.

## Workloads and validation

| Label | Input | Output | Actual tiles | Coding |
|---|---|---|---|---|
| multi | dav1d corpus `8-bit/features/non_uniform_tiling.ivf` | 24 × 720×300, 8-bit | 1 column × 4 rows | one key output, 23 inter outputs |
| tiled_first | first packet extracted from that IVF | 1 × 720×300, 8-bit | 1×4 | same key frame, no re-encode |
| tiled_stress | `tile_threading_cdef_lpf_race.obu` | 1 × 1024×1024, 8-bit | 4×8 | key frame |
| single_10b | `lr_sgr_10bpc_noisy_nocdef.obu` | 1 × 256×256, 10-bit | 1×1 | key frame |

Tile geometry comes from decoded frame headers, not filenames. All 27 output
frame hashes match installed upstream `dav1d`, with film grain explicitly on.
The instrumented crate passed 115 native tests and six doctests; two existing
documentation examples remain ignored. The default `no_std` check passed. Three
malformed-input cases were rejected by the new harness.
The native corpus revision and input SHA-256 hashes are recorded alongside the
results. These are four inputs, not a representative compression-quality or
resolution survey; cross-input timing differences do not isolate tile count.

The checked decoder forces `n_fc=1`: multi-frame **input** does not imply frames
concurrently in flight in one decoder. Independent decoder instances provide
the second concurrency dimension here. No `unchecked` feature was enabled.

`examples/profile_concurrency.rs` parses IVF strictly, drains each packet, and
drains/resets at stream end. It rejects decode/flush errors, zero outputs,
truncated packet headers/bodies, and changed output counts. Each persistent
decoder validates the complete output hash sequence before and after timing
against a serial reference. Every timed pass checks the frame count; pixel
hashing is outside timing. Decoder construction, pool startup, and teardown are
also outside timing. This checks repeated use of the same decoder, not only
fresh-instance decoding. Worker panics make the harness fail immediately.

## Baseline matrix

48 cells, five timed repetitions each, plus a calibration pass per cell. One
process per cell isolates the monotone global state. Calibration targets about
300 ms per repetition; actual durations vary with warmup/scheduling. No census
or task instrumentation is enabled in the baseline binary.

Median wall milliseconds per output frame, one decoder:

| Workload | 1 worker | 2 | 4 | 8 | 16 | 24 |
|---|---:|---:|---:|---:|---:|---:|
| multi | 2.048 | 2.966 | 2.765 | 2.857 | 3.396 | 3.496 |
| tiled_first | 4.063 | 2.877 | 1.855 | 1.999 | 2.245 | 2.346 |
| tiled_stress | 12.925 | 9.964 | 5.417 | 3.561 | 3.284 | 3.351 |
| single_10b | 1.753 | 1.597 | 1.608 | 1.598 | 1.685 | 1.680 |

Additional cells: `(workers, instances)` = `(1,2), (1,4), (1,8), (2,4),
`(4,4), (8,4)`. The last oversubscribes the host. Milliseconds/frame across
instances is **aggregate throughput**, not the latency of an individual decode.
For multi, eight single-worker decoders achieve 0.289 ms/output frame, about
7.09× single-decoder throughput. Four eight-worker decoders achieve 0.799
ms/frame. Using more independent single-worker decoders is a strong lead for
this stream when the application has independent streams to schedule.

All replicates, including the noisy single_10b multi-instance cells, are
retained. An initial matrix was superseded after correcting the perf control
acknowledgement parser; its immutable binaries and logs remain in scratch.
Final baseline timing and perf sampling use the same `base` binary SHA-256.

## What the profiles show

12 baseline sample profiles and 12 hardware-counter runs cover 1×1, 8×1, and
8×4 for every workload. Perf control FIFOs enable events after warmup and disable
them before final validation. The tiny control/barrier intervals remain inside
the event window. Recorded commands specify `cycles:u`, 499 Hz, DWARF stacks.
Self percentages are sampled CPU cycles across workers, **not wall time or a
predicted speedup ceiling**. Report filtering omits symbols below 0.3%.

* **multi, eight workers:** wide-read registration 15.81%, wide retirement
  12.43%, spin waiting 13.12%. Visible tracker-related self samples sum to about
  61%. With four decoders, spin waiting rises to 18.16% and tracker self samples
  to about 66%. At one worker, spin waiting has no visible samples and tracker
  self samples total about 14%.
* **tiled_first, eight workers:** ordinary read/write registration totals
  27.09%; spin waiting is below the report threshold. The tracker is expensive
  even when waiting is not.
* **tiled_stress, eight workers:** ordinary read/write registration totals
  36.22%, with spin waiting below the threshold. More workers help until roughly
  16 on this host, but do not remove registration overhead.
* **single_10b:** entropy/coefficient decoding accounts for 48–57% of self
  samples and `memset` 13–15%. Tracker-related self samples are around 1–3%.
  Optimizing the spinlock is unlikely to help this input materially.

**Stack limitation:** worker DWARF unwinding is poor in this environment. A
second capture with 65,528-byte stacks still had zero decoded stack entries in
2,365 of 2,479 worker samples. Self-IP profiles remain available, but the
`calls` reports must not be treated as trustworthy caller attribution. The
call-site census and source inspection provide the attribution below. A future
frame-pointer build should be a separate diagnostic arm, not silently replace
the baseline binary.

## Usage and stage census

The new `probe-usage` / `__probe_usage` feature retains the production sharded
tracker. It aggregates counters in thread-local maps and merges on thread exit;
the driver joins all workers before reporting. It records source sites, read/
write counts, byte and row-size buckets, active shard count/block shift,
construction and reconfiguration events, and occupied slots at allocation.
Construction policy is sampled at constructor entry; if another thread changes
global policy during construction that event can differ from the resulting
fields. Borrow census keys read the actual instance fields. The present cells
use a common worker count, with policy established before those constructors.
It cannot be combined with the legacy tracker selection.

Usage counts cover **the whole process**: serial reference, instance warmups,
timed passes, final validation, and teardown. `lifetime_frames` supplies that
denominator. They are not timed-region-only counts. Range counts include empty
attempts; rectangle counts include accepted rectangles. Occupancy counts
per-shard allocation attempts and can include failed/promoted attempts. Byte
totals describe registered extents, not bytes actually loaded. Construction
bytes are cumulative fixed tracker storage initialized, not peak RSS and not
all decoder allocations. The feature does not measure guard lifetimes or an
address-transition/reuse-distance distribution; those remain useful follow-ups.

Eight-worker, one-instance census, per lifetime output frame:

| Input | Registrations | Mutable share | Extent <16 bytes | Fixed tracker bytes constructed |
|---|---:|---:|---:|---:|
| multi | 104,583 | 41.3% | 79.9% | 10.20 MB |
| tiled_first | 246,306 | 28.4% | 95.2% | 4.85 MB |
| tiled_stress | 1,360,777 | 24.1% | 94.6% | 11.47 MB |
| single_10b | 9,712 | 26.0% | 25.5% | 1.23 MB |

In multi, `src/safe_simd/mc.rs:12847` (`warp8x8_dispatch`) and `:12161`
(`mc_put_dispatch_inner`) both call `full_guard`. Together they register about
46.85 GB of the census's 47.63 GB of extents over 96 output frames: roughly
98.4% of registered bytes in only about 2.5% of registrations. These are
whole-picture reads for motion compensation, not 46 GB of copying.

The separate `tasks` binary enables `probe-tasktime,probe-wide`, without usage
maps. At multi/t8 it records about 996 shard-count promotions, 1,624 block-count
promotions, and 461 full-slot promotions **per timed output frame**. Summed
contended-wait duration is about 1.39 ms/frame (including descheduling and some
probe bookkeeping); waits can overlap, so this is
not 1.39 ms of removable wall time. Mean sampled active task bodies is 2.33 at
eight requested workers and 2.49 at 24. The post-tile filter-only tail occupies
only about 2% of samples there. Full-plane reference traffic is a much stronger
lead than a serial filter tail for this stream.

In contrast, the 32-tile frame reaches mean activity 6.69 at eight workers;
its filter-only tail still has mean activity 5.53. Filter work overlaps across
rows: do not assume all filter stages form one process-wide serial chain.
Single_10b reaches only about 1.27 active task bodies at eight workers.

Task hooks do not cover the serial path: zero task activity at t1 means
**unmeasured**, not idle decoding. Park intervals crossing reset may include
warmup. Task sampling and global cold-path counters can perturb scheduling.
Usage maps extend critical sections and are more intrusive still; their times
and contention counts are not baseline measurements. `probe-count` was avoided
because it selects the legacy tracker. Rectangle counters now reset along with
the other wide counters, and task-probe worker-slot overflow fails loudly.

## Process-history experiment

Three alternating cold/primed pairs per workload at one and four workers.
Priming opens and drops a 24-worker decoder **without decoding input**, before
creating the measured decoder. The measured decoder's pool is persistent.

| One-worker input | Fresh-process median | After priming | Ratio of medians |
|---|---:|---:|---:|
| multi | 2.185 ms | 4.107 ms | 1.88× |
| tiled_first | 4.113 ms | 5.092 ms | 1.24× |
| tiled_stress | 12.933 ms | 18.689 ms | 1.45× |
| single_10b | 1.805 ms | 1.883 ms | 1.04× |

Four-worker arms changed little. The experiment manipulates both monotone
policies together; it does **not** isolate the share due to picture guard
selection versus shard/block configuration. Source inspection identifies both
as candidates. Do not restore the old unsafe practice of turning a global
policy off while another decoder may still be using it.

## Next experiments and proof obligations

1. **Make threading policy local to each decoder/picture.** Preserve the
   monotone fallback until all consumers carry immutable instance policy.
   Test mixed 1/8/24-worker decoders, simultaneous construction/destruction,
   retained pictures/guards, and both orders of creation. Shard mapping must
   remain fixed for every live guard; policy changes require exclusive access.
   The history experiment gives this work a measurable acceptance criterion.
2. **Reduce repeated whole-plane MC reservations.** First derive the actual
   SIMD footprint for 8-tap and warped reads, including negative offsets,
   padding, bit depth, and edge emulation. Compare bounded reservations with a
   shared read lease over a fully completed reference picture. A lease must
   keep the storage alive and exclude every writer for its entire lifetime;
   intra-block-copy/current-picture paths need their existing protection.
   Never narrow the registered record while still constructing a wider Rust
   reference. No new copy band or tile-keyed exclusion scheme is justified by
   these measurements; see `docs/OWNERSHIP_MODELS.md` for prior failed attempts.
3. **Treat registration cost separately from waiting policy.** The still
   profiles justify measuring fewer exact registrations or better metadata
   locality. The inter-frame profile is the appropriate workload for bounded
   backoff/parking experiments, particularly under 32-worker oversubscription.
   Reducing wide-path lock traffic comes first. These counters do not establish
   that any alternative lock is faster.
4. **Investigate repeated construction/clearing and entropy costs separately.**
   The allocation census makes reuse a plausible lead, but is not a lifetime or
   peak-memory profile. The earlier compact-tracker experiment used stills and
   did not establish a decoder win; it needs this inter-frame workload before
   any broader conclusion. `memset` samples in single_10b are not automatically
   tracker initialization; caller attribution is needed.

For any implementation experiment: keep default checked tracking, run the
same baseline binary controls with rotated A/B/A order, retain every sample,
check all output hashes, and repeat the adversarial borrow/loop/protocol tests.
Changes to record lifetime, publication, or reference construction additionally
need the existing Miri and Loom release gates. Profiling is not a soundness
proof and does not advance the 0.3.2 release bookmark or publish a package.

## Reproduction and evidence

From the repository root, choose a scratch output directory. `prepare.py`
expects the checked-out dav1d corpus and tracked crash inputs. Run heavy phases
serially through the workspace wrapper, with `TMPDIR=/home/lilith/tmp`:

```text
python3 audit/concurrency-profile/prepare.py OUT
run-heavy --mem 16G --jobs 8 -- python3 audit/concurrency-profile/build.py OUT
run-heavy --mem 16G --jobs 8 -- python3 audit/concurrency-profile/run.py OUT baseline
```

Repeat `run.py` for `profile`, `stat`, `tasks`, `usage`, and `history`, then run
`validate.py OUT` and `analyze.py OUT` through the same wrapper. Hardware perf
access and installed `dav1d` are prerequisites. The decoder CLI is:
`profile_concurrency INPUT THREADS INSTANCES PASSES [REPS]`.

Committed `results/` contains input/build/source provenance, all final timing
replicates and raw logs in compressed JSONL, census/task summaries, self sample
reports, hardware counters, validation, and native test output. Baseline and
history tables are also plain TSV. Large perf.data files, immutable binaries,
superseded runs, and stack diagnostics remain at
`/home/lilith/tmp/rav1d-concurrency-2026-09-06`.

No default constructor or borrowing API changed. The new methods/modules and
caller instrumentation are behind diagnostic features. The release candidate
source/bookmark remains separate from this profiling work.
