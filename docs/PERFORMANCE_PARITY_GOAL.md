# Goal: safe still decoding at upstream rav1d speed

The user activated this goal on 2026-09-07 with a **10% per-workload-group**
and **25% per-cell** slowdown allowance, replacing the proposed 5%/10% limits.
Implementation and evidence are developed in a draft PR for review. This goal
does not authorize publishing a crate.

Current measured progress and outstanding gates are recorded in
[STILL_PARITY_CHECKPOINT.md](STILL_PARITY_CHECKPOINT.md).

## Copyable goal

> Bring the default, checked, safe-Rust build of rav1d-safe to performance
> parity with the current memorysafety/rav1d upstream with assembly enabled,
> prioritizing 2K, 4K, and 8K still-image decoding. Work through reproducible
> baselines, borrowing and scheduling measurements, profiles, implementations,
> adversarial correctness tests, and paired benchmarks. Keep overlap checking
> and the default `forbid(unsafe_code)` guarantee. Do not count unchecked or
> assembly builds of rav1d-safe as satisfying the goal. Follow
> `docs/PERFORMANCE_PARITY_GOAL.md` for the corpus, fairness rules, safety gates,
> and completion criteria. Preserve the history of failed ownership/copying
> approaches and explain what new evidence would justify revisiting one.
> Commit and push validated improvements and reproducible evidence as work
> proceeds. Continue pursuing the largest measured gaps; report partial
> progress honestly and do not declare parity until the holdout gates pass.
> Meet the user-approved 10% group / 25% cell limits below. Investigate
> archmage and magetypes improvements when useful; a separately reviewable
> dependency PR and local Cargo patch are permitted.

## What parity means

The primary target is **elapsed AV1 decode time per visible frame** in an
in-memory consumer, compared with a pinned, current commit of
[memorysafety/rav1d](https://github.com/memorysafety/rav1d). Its assembly and
normal CPU dispatch remain enabled. This is distinct from the unrelated old
`rav1d` package on crates.io and from rav1d-safe's own `asm` configuration.

On the designated release machine, require all of the following, with the
corpus and thresholds fixed before evaluating a candidate:

1. The equally weighted geometric mean of candidate/upstream elapsed-time
   ratios is **at most 1.10** in every resolution × layout × concurrency
   stratum, for both development and untouched holdout sources.
2. No individual mandatory cell is **more than 1.25×** upstream. Use at least
   nine rotated paired rounds, each with at least 500 ms of measured upstream
   decode work. The one-sided 95% bootstrap upper confidence bound on the
   paired median ratio must meet the same limits. A noisy result is unresolved;
   collect more data under controlled load, rather than relaxing the limits.
3. Both persistent-decoder throughput and decoder-create → decode → drop
   latency pass. Report first-use process/pool initialization separately;
   neither excluding it nor amortizing it may hide an application regression.
4. All compared configurations produce exactly the same ordered visible YUV
   frames. No panic, timeout, missing frame, incorrect image, or disabled
   feature can enter the speed aggregate.
5. Peak resident memory and p95 request latency do not regress by more than
   10% against the pinned rav1d-safe baseline under the same workload. Retain
   a representative tiled-video regression set, with a 5% time regression
   budget. Review any deliberate tradeoff with the user before claiming the
   goal complete; the limits do not change automatically.

Initially designate this Ryzen 9 7900X x86-64 machine as the measurement host.
Record the exact CPU, kernel, governor, available ISA, RAM, toolchain, and
compiler flags with each baseline. A claim on this host is not a claim about
all x86 CPUs or ARM. Require native AArch64 correctness and regression checks
before a cross-platform release; set a separately measured ARM parity target
when a designated ARM benchmark host is available.

## Workload contract

For this campaign, the shorthand sizes are UHD classes: **1920×1080,
3840×2160, and 7680×4320**. Also include portrait and odd-dimension cases at
comparable pixel counts. State exact dimensions everywhere. Use native large
sources and documented crops/downsampling; do not invent 8K detail by
upscaling a small source.

Freeze a manifest with at least 12 development and 12 distinct holdout source
images, each set covering photographs, fine natural texture, screen/text,
and maps or documents. Keep related crops and resizes of one source in the
same split. No Kodak or smooth-gradient calibration images. The initial
[two-source investigation](../benchmarks/stills-2026-09-07/README.md) is a
development seed and cannot by itself satisfy the parity goal.

Mandatory coverage:

- Single visible frame; minimum legal tiling and eight tiles. At 8K, a single
  tile is illegal: the decoder enforces maximum tile width 4096 and tile area
  4096×2304. Use and verify 2×2 as the minimum layout at 7680×4320, rather than
  labelling it single-tile. Record actual parsed tile geometry.
- 8-bit and 10-bit, with 4:2:0 and 4:4:4 coverage; 12-bit, monochrome,
  film grain, intrabc, lossless, and AVIF grid items as explicit edge strata.
  Promoting an 8-bit source to 10-bit exercises decoder code but is not HDR
  source coverage. Keep alpha and AVIF container/color conversion timings
  separate from the AV1 core target. Verify actual CDEF and loop-restoration
  activity in designated coverage cells; encoder command-line flags alone
  do not establish that a stage executed.
- At least two independently implemented encoders (libaom and rav1e or
  zenrav1e, named accurately), with two documented quality/effort settings.
  Freeze bitstreams before optimization; never tune encodes to flatter a
  candidate decoder. Equal bitstreams make image-quality metrics unnecessary
  for this decode comparison.
- One decoder at 1, 2, 4, 8, 16, and 24 requested threads. Four independent
  decoders at 1, 4, and 8 threads each, plus a mixed-size request stream.
  Report aggregate frames/s and per-request latency separately. A many-tile
  result cannot substitute for a minimum-tile result.
- Fresh processes, a 24-thread decoder opened before a serial decoder,
  and overlapping independent decoders with different thread counts. Prime
  both comparison arms identically. Include repeated open/decode/flush/drop
  loops to catch process-global policy and lifetime leaks.

Use a balanced, predeclared representative matrix for the expensive edge
strata; the primary content/size/tile/thread matrix remains mandatory. Every
omission is visible in the manifest. Do not average a missing or failed cell
away, or replace difficult sources after seeing results.

## Measurement and implementation protocol

1. Synchronize with Git through the workspace's jj workflow. Preserve the
   release branch, existing work, and ownership marker. Read the current
   [ownership experiment ledger](OWNERSHIP_MODELS.md) and
   [soundness release protocol](RELEASE_SOUNDNESS_PROTOCOL.md).
2. Pin both sources and standalone consumer harnesses, dependency lockfiles,
   compiler, features, and binary SHA-256 values. Avoid root dev-dependency
   feature unification. Use matched release/LTO/target/alignment flags, with
   no `target-cpu=native` advantage for one side. Default checked is primary;
   unchecked and own-assembly builds are diagnostic arms.
3. Match visible output against upstream and independent dav1d before and
   after timing. Count every timed frame. Time direct in-memory calls inside
   each harness; startup, file I/O, hashing, logging, and corpus preparation
   stay outside the core timer. Lifecycle tests have their own explicit timer.
4. Instrument borrowing separately from speed measurement: borrow shapes and
   sizes, active records, shard occupancy, slow-path allocation and spinning,
   retained guards, copies, allocations, and per-worker task/idle/filter time.
   Reset diagnostic counters only when all users are quiescent. Keep probe
   builds out of the primary timing results.
5. Profile only the measured region. Attribute sampled self CPU time and
   blocked/waiting time separately. Compare the same frame across sizes and
   tiles; do not import a video's motion-compensation bottleneck into a still
   diagnosis. Prioritize the largest demonstrated difference from upstream.
6. Make one causal change at a time. Freeze its pre-change binary, run paired
   A/B, and retain only improvements that survive the mandatory regressions.
   Check code-placement effects for small gains. Record failures, rejected
   designs, and their evidence so later runs do not repeat them blindly.
7. Keep heavy builds, encodes, benchmarks, and profilers sequential under
   `scripts/run-heavy`, within the machine's resource limits. Store large
   assets outside Git with paths, checksums, acquisition commands, and backup
   status; commit small manifests, summaries, scripts, and logs/provenance.

Candidate leads should follow the measurements: scalar entropy work and
bounds checks, transform and filter SIMD, guard acquisition frequency and
footprints, private worker storage, allocation reuse, tile scheduling, and
filter tails. A lead is not a promised speedup.

## Safety and release gates

Keep exact reference footprints, overlap exclusion, lifetime/publication
rules, and cross-instance behavior correct in every supported build. Do not
disable the tracker, allow overlapping references, extend a borrow across
unregistered gaps, remove safety assertions, or convert checked accesses to
unchecked ones to meet a speed target. An `unsafe` optimization hidden in a
dependency still needs an abstraction-level proof and adversarial coverage.

Run appropriate nextest and doctest gates, independent decoder conformance,
repeated concurrent lifecycle tests, panic/unwind and adversarial API tests.
Changes to the borrow abstraction or publication protocol require the
applicable Miri (Stacked and Tree Borrows), Loom, no_std, and API/semver checks
from the release protocol. Explain why every resulting reference and every
concurrent operation is valid; test success alone is not a soundness proof.
Preserve `const new()` and the 0.3.x compatibility contract unless the user
separately authorizes a breaking release.

Finish with a reviewable change, raw evidence, reproduction commands, an API
diff, a remaining-risk statement tied to concrete evidence, and the measured
completion table. Performance parity does not authorize publishing a crate.

If parity remains out of reach in a run, checkpoint the best validated code,
remaining per-stratum gaps, rejected experiments, and the next measured lead.
An elapsed session budget, exhausted easy ideas, or a faster unchecked build
is not completion of this goal.
