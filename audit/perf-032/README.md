# Compact tracker and x86 layout screening

2026-09-06, Ryzen 9 7900X, x86_64, rustc 1.98.1. Source candidate:
`c5fe631f` (crate source unchanged since `09109d91180ed1eb2ae16cff2793013533679b25`).

**Decision: retain both as experiments; neither earns a decoder change in 0.3.2.**
Compact storage substantially reduces tiny-buffer construction cost but adds
steady-state indirection. Layout improves one concurrent microbenchmark, with
no consistent whole-decoder improvement in this screen. These are preliminary
measurements on two decoder vectors, not general performance estimates.

The performance work ran from 03:08:30 to no later than 03:17:45 UTC on
2026-09-07: **9 minutes 15 seconds**. Focused Miri validation and evidence
recording followed. Production source and the packaged release candidate were
not modified. Executable prototypes remain under
`/home/lilith/tmp/rav1d-perf-experiment-2026-09-06`.

## Changes tried

- `compact.patch`: replace the embedded 128-shard array with a `Vec<Shard>`
  sized to the instance's active mask. Grow the array through `&mut self`
  before installing a larger mask; preserve existing records and poison state.
  Every access still performs the same overlap checks. No new unsafe code.
  Tradeoff: a second allocation and a dynamic array lookup replace the fixed
  array, potentially adding indirection and bounds checks on the hot path.
- `layout.patch`: keep all seven slots and 128-byte shard alignment; explicitly
  use C field layout and paired `(start, end)` records. Lock, liveness, flags,
  allocation state, and slot 0 fit in the first x86 64-byte cache line.
  Capacity, admission, retirement and atomic ordering are unchanged.

Both preserve const construction. Scratch-source historical comments have not
been rewritten into release documentation; the patches are not shipping diffs.

## Synthetic results

In-process `Instant` measurements, rotating arm order, 8 rounds with the first
excluded. Thread creation happens before timing. Every borrow workload writes
random indices in disjoint worker regions, then verifies the final sum.
Construction retains all 4000 buffers until after the timer stops, so it prices
allocation and initialization rather than allocator reuse of one buffer.

| Case | Compact / base | Layout / base | Interpretation |
| --- | ---: | ---: | --- |
| Construct 4000 tiny buffers | 0.0224 | 1.0183 | Compact construction is much cheaper |
| Small buffer, 1 thread | 1.1349 | 1.0575 | Compact slower in all 7 pairs |
| Small buffer, 4 threads | 1.0567 | 0.9887 | Wide timing bands |
| Large buffer, 1 thread | 1.0353 | 0.9758 | Wide timing bands |
| Large buffer, 4 threads | 1.1484 | 0.8532 | Wide timing bands |
| Large buffer, 8 threads | 0.9397 | 0.7778 | Layout faster in all 7 pairs |

Ratios are medians of paired arm/base ratios. Construction medians were
21.10 ms for baseline and 0.462 ms for compact. The eight-thread layout ratio
ranged from 0.621 to 0.855. These are microbenchmark results, not decoder gains;
several other cells have very large variability and cannot support conclusions.

## Current-decoder results

Current rav1d-safe 0.6.0 source, default features, release profile with fat LTO,
no target-cpu=native. Six rotating rounds; round zero excluded. Each process
uses one decoder, warms it, and measures 12 repeated decodes in-process. Parsing,
construction, process startup, and checksum computation are outside the timer.
Every timed call must return a frame. The modified benchmark example is the same
for all arms and adds raw OBU input support and the frame-presence assertion.

| Vector / threads | Compact / base | Layout / base |
| --- | ---: | ---: |
| 256x256 noisy 10-bit LR, 1 | 1.0098 | 1.0243 |
| 256x256 noisy 10-bit LR, 4 | 1.0013 | 1.0083 |
| 1024x1024 tiled 8-bit, 1 | 1.0153 | 0.9921 |
| 1024x1024 tiled 8-bit, 4 | 1.0481 | 0.9943 |

Compact's tiled four-thread ratio was above 1 in all five pairs (1.002–1.074).
That is enough to decline this prototype, not a broad statistical claim about
all workloads. Layout has mixed pair signs in every decoder cell; its synthetic
gain did not establish an end-to-end benefit here.

Warmup-frame MD5 matched across all arms, rounds, and both thread counts:

- `lr_sgr_10bpc_noisy_nocdef.obu`: `bafc939f19adc1451c122baa7a8c828d`
- `tile_threading_cdef_lpf_race.obu`: `51b9c3ab246fda65e2c0a2155588e9a5`

Timed frames were checked for presence, not individually hashed. This screen
compares against the current checked baseline, not against a fresh dav1d run.
It does not cover inter video, film grain, 12-bit, other architectures, or many
simultaneous decoders. There is no A/A layout-control family or controlled CPU
placement, so small differences should not be interpreted as wins.

## Correctness and next direction

Both prototypes passed the unchanged 115-test native suite with aligned,
pic-buf and zerocopy adapters. A new `compact_transition.rs` regression covers
small-to-large growth after a parallelism change, overlap rejection near the
threshold, shrink, regrowth, and preservation of data. Focused Miri status is
recorded in `miri.json`: both prototypes passed all three focused tests under
both Stacked Borrows and Tree Borrows (12 executions total). These checks ran
after the bounded timing pass. Full Miri, Loom and platform CI were not rerun
for these experimental patches; this is not the full release gate.

The compact experiment identifies an allocation opportunity and rejects the
simplest dynamic-array implementation for steady-state decoding. A future
attempt should keep the large-buffer fixed-array path and specialize small or
worker-private storage without charging every borrow for dynamic lookup. The
layout experiment needs a broader native workload and stronger timing controls
before further implementation is justified. Neither requires dropping const
support or making a breaking public API change.

Scripts are evidence of the exact scratch run, not integrated release commands.
Use the recorded scratch directory to rerun them. Raw rows are in `*.jsonl`,
paired summaries in `*-summary.json`, and source/binary hashes and build commands
in the accompanying JSON. All heavy jobs were serialized through run-heavy with
nice/ionice, a 16 GiB memory cap, and eight build jobs.
