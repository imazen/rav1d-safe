# First release, latest release, unchecked, and assembly comparison

Measured on 2026-09-07, Ryzen 9 7900X, following the
[upstream comparison](../upstream-2026-09-07/README.md). The first release of
rav1d-safe already has a substantial performance gap on these inputs. Current
assembly closes most of the gap; the supported `unchecked` flag gives a
smaller improvement. Historical multithreaded builds also expose correctness
failures, so speed alone cannot justify returning to an older implementation.

## What was built

- **First published rav1d-safe: 0.1.0**, published 2026-02-12. The downloaded
  `.crate` was SHA-256 verified against crates.io. Checked and `unchecked`
  builds use its original packaged source and lockfile versions.
- **Latest published rav1d-safe: 0.5.7**, published 2026-05-26. Its checked and
  `unchecked` builds likewise use the checksum-verified package.
- **Current rav1d-safe:** source `0834ba8e9c550bb23651fcbf6fb922e0af9ad243`
  (package version 0.6.0; decoder changes from `d1767a1b`). Checked,
  `unchecked`, and `asm` builds use the same consumer harness and lock.
- **Upstream rav1d:** the immutable assembly-enabled binary from
  `d3d1cd67059f47803919be8276650e5870c9fd02`, identified in the previous report.
- **Historical assembly, reconstructed:** 0.1.0 plus only its omitted
  `src/x86` and `src/arm` files, recovered from the publication revision
  `441dad0c1ed21d700efb51ab40392d3c835a1819`. Every file originally in the
  published package remains byte-for-byte unchanged. This is explicitly a
  reconstructed source build, not a working assembly build of the bare package.

Both published packages' advertised `asm` feature **fails to build** because
the required assembly inputs are absent. Current assembly builds successfully
from the repository. The packaging failures and restoration file list are
retained. No assembly source from a newer decoder was substituted into 0.1.0.

The crates.io name `rav1d` also has a 2019 version 0.1.0 belonging to
`rainliu/rav1d`, a different project from `memorysafety/rav1d`. Its metadata is
recorded to prevent treating that earlier name holder as this decoder's release.

## Results

Single worker, matching in-flight frame count:

| Input | 0.1.0 checked | 0.5.7 checked | Current checked | Current unchecked | Current asm | Upstream asm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 24-frame tiled video | 2.200 | 1.882 | 2.076 | 1.786 | 0.518 | 0.360 |
| First tiled frame | 5.206 | 4.493 | 4.109 | 3.689 | 1.787 | 1.396 |
| 32-tile still | 17.080 | 13.783 | 13.066 | 10.438 | 4.095 | 3.050 |
| One-tile 10-bit still | 2.275 | 1.745 | 1.766 | 1.748 | 0.891 | 0.874 |

Eight workers, one frame in flight:

| Input | Current checked | Current unchecked | Current asm | Upstream asm |
| --- | ---: | ---: | ---: | ---: |
| 24-frame tiled video | 2.004 | 1.442 | 0.551 | 0.267 |
| First tiled frame | 2.006 | 1.671 | 0.972 | 0.722 |
| 32-tile still | 3.904 | 2.595 | 0.976 | 0.715 |
| One-tile 10-bit still | 1.631 | 1.624 | 0.955 | 0.919 |

Historical unchecked and reconstructed assembly, one worker:

| Input | 0.1.0 unchecked | 0.1.0 restored asm | Current asm / restored asm |
| --- | ---: | ---: | ---: |
| 24-frame tiled video | 1.783 | 0.441 | 1.17× |
| First tiled frame | 4.628 | 1.617 | 1.11× |
| 32-tile still | 14.662 | 3.790 | 1.08× |
| One-tile 10-bit still | 2.261 | 0.890 | 1.00× |

With frame threading enabled on the video:

| Workers | Current unchecked | Current asm | Upstream asm |
| ---: | ---: | ---: | ---: |
| 8 | 0.884 | 0.337 | 0.164 |
| 24 | 0.862 | 0.346 | 0.190 |

Times are milliseconds per displayed frame, medians of five rotations. The
single-worker table is the clean historical comparison: 0.1.0's public managed
API always selects automatic frame delay and cannot request delay one. At one
worker all versions have exactly one frame in flight. Current, 0.5.7, and
upstream otherwise use `max_frame_delay=1` in the main grid.

The full table includes 1/2/4/8/16/24 workers and four simultaneous decoders at
one/eight workers each for the current checked/unchecked/assembly and upstream
arms. Historical checked/unchecked 0.5.7 and the previous checked example
control are measured at one/eight workers. Historical 0.1.0 and reconstructed
assembly are measured at one worker. Multi-instance milliseconds per frame
describe aggregate throughput, not latency of an individual frame.

The explicitly named `auto-*` arms allow frame threading on the video at
eight/24 workers. The current consumer must handle `NeedMoreData` backpressure:
retrieve old output and retry the **same** packet. This follows the existing
`tests/filmgrain_threads.rs` pump. The initial automatic-delay adapter omitted
this retry and failed preflight; it was corrected before measuring. That
initial failure is a harness error, not a decoder performance or correctness
result. The initial source delta, build hashes, and failed preflight are kept.

## Correctness findings

- **0.1.0 checked, eight workers:** all four smoke tests panic, reporting
  overlapping borrows or the historical maximum of 32 concurrent borrows.
  Its dependency enables the old `single-threaded` tracker feature. These
  configurations receive no performance numbers. The one-worker arm passes.
- **0.5.7 checked, eight workers, 32-tile still:** a longer run panics on
  overlapping mutable borrows, despite a passing short smoke test.
- **0.5.7 unchecked, same case:** the repeated-run hash validation detects
  incorrect pixels: `65a49eb8d39173b7dd9c56cec5352c3a` instead of
  `51b9c3ab246fda65e2c0a2155588e9a5`. Turning checks off hides a problem rather
  than establishing that this concurrent path is correct.

A failed cell is excluded entirely from numeric comparisons, including any
earlier passing samples. Its failed records remain in the raw evidence.
No expected hash, frame count, or correctness assertion was weakened. Passing
these fixtures is an execution check, not a soundness proof for unchecked code.

## Meaning of unchecked

Current `unchecked` enables unchecked SIMD slice access, additional entropy
intrinsics, and unchecked construction for selected hot buffers. Other
`DisjointMut` instances still track borrows. The feature's meaning differs from
0.1.0, where it also enabled the dependency's global `unchecked` feature.
It is not a pure bounds-checking or pure-locking ablation.

Timed-region profiles of the exact current binaries confirm that tracking
survives both features. On the video at eight workers, explicitly named
tracker functions account for **31.19%** of sampled cycles with `unchecked`
(15.90% in `TinyLock::lock_slow`) and **55.62%** with `asm` (29.92% in the lock
slow path). Assembly greatly reduces total decode time, so its larger tracker
percentage does not mean more absolute tracking work. These CPU cycle shares
are not predictions of removable wall time. Remaining scratch/non-picture
tracking and contention are concrete leads for narrowing the assembly gap.

Current checked single-worker video is 5.6% faster than 0.1.0 but 10.3% slower
than 0.5.7. Current checked single-worker stills are about 21–23% faster than
0.1.0. Current assembly is 17.4% slower on this video than the reconstructed
0.1.0 assembly arm. Dependencies and wrapper implementations also differ, so
these release comparisons do not isolate a particular commit as the cause.

The old `probe-untracked` diagnostic was also attempted, alone and with
assembly. Current disjoint-mut intentionally rejects those probes with
`compile_error!("unsound measurement probes are disabled; use a historical benchmark revision")`.
That guard was preserved; those attempts have no timing result. All production
source and default feature choices remain unchanged by this investigation.

## Protocol and provenance

Both projects use Rust 1.98.1 / LLVM 22.1.8, release fat LTO, one codegen unit,
`panic=unwind`, and `RUSTFLAGS='-C llvm-args=-align-all-functions=4'` with the
default compilation target. The common consumer unwind profile differs from
0.1.0 and upstream's own workspace abort profiles. Native CPU dispatch, grain,
strict decoding, all filters/layers, operating point zero, visible output,
and a 120-million-pixel limit are selected. Validation implementations evolve
across releases even when strictness settings match.

The standalone consumer drivers exclude the decoders' development dependencies.
The previous example-based checked binary is a separate control, allowing
build-context/code-generation drift to be observed instead of attributing it
to a decoder change. Package versions and lockfile checksums are preserved;
this is a same-compiler comparison of historical sources, not a recreation of
the historical compiler or hardware.

The controller reads internal timings of persistent, warmed decoders operating
on in-memory packets. Process startup, file reading, IVF parsing, pool creation,
and hashing are outside the timer. Every output is counted during timing;
every visible output plane is hashed before and after it. Hash sequences also
match the independent upstream reference from the previous experiment.
The inputs and output oracle are unchanged from that comparison.

Five rotations vary arm and workload order; all arms use the same repetition
count for a given input. The initial rotation was interrupted by the 0.5.7
checked failure. Valid completed cells were retained, the failed cell remained
failed, and the controller resumed the remaining comparisons. Raw timestamps
preserve this interruption. Five rotations keep the expanded experiment within
the requested time allowance. Failed cells do not acquire medians on resume.
There are **840 successful timed runs**, 208,800 timed output frames, and
3,480 successful before/after/reference hash passes, plus the two retained
timed-attempt correctness failures. The original and resumed controllers ran
for 43 and 219 seconds. The latter reported peak RSS 0.13 GiB and minimum
available memory 27,386 MiB. The two additional profiles ran for nine seconds.
These are wrapper resource observations, not a decoder heap comparison.

All heavy jobs ran sequentially under `run-heavy --mem 16G --jobs 8`, with
nice 19, idle I/O priority, and a hard cgroup memory cap. A shared file lock
serialized the timing controller. No build or profiler overlapped timing.
The wrapper's full resource telemetry is retained in the compressed logs.

Evidence includes checksum-verified registry metadata, exact driver sources and
locks, commands/build outcomes, before/after source delta for the automatic
feed adapter, preflight failures, every timed attempt, and the numerical summary.
Compressed text files are each smaller than 30 KB. Decoder binaries, downloaded
packages, and extracted sources remain in
`/home/lilith/tmp/rav1d-release-perf-2026-09-07`.

The build scripts are archived as executed and use that scratch directory as
their script directory. Reproduction requires preparing the pinned packages
and the driver manifests from `drivers/`, restoring each compressed Cargo.lock,
and adjusting their dependency paths to the chosen scratch/current-source
locations. `restored-files.json` identifies the historical assembly restoration.
Build heavy phases serially through the wrapper, then run:

```sh
env TMPDIR=/home/lilith/tmp /home/lilith/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- \
  python3 benchmarks/release-perf-2026-09-07/compare.py --repo REPO --work-dir SCRATCH --reps 5
```

Use a fresh output directory for a new experiment: the controller deliberately
resumes an existing `matrix.jsonl`. Package/binary/source identities and input
hashes must match before resuming saved measurements.
