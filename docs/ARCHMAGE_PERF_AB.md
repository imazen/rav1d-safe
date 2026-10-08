# Archmage version-only performance comparison

Completed: matched 0.9.29 versus 0.9.30 A/A and A/B timings in both modes on Zen 4. Missing: Zen 5 and consumers without the example builds' development-feature unification.

On 2026-10-08, the live GitHub latest release and main revision and the
package index resolve to archmage 0.9.30 / e2dbab66. The initial build pair used the
previous 0.9.29 pin b8d6c077 and the current pin on identical decoder source
at 8ea8bf18; the completed timing pair below uses f010130a. All 258 tracked Rust files match. Only archmage and
archmage-macros entries differ in the lockfiles; other package versions and
sources match. Both modes use Rust 1.99, release fat LTO, empty RUSTFLAGS
and runtime dispatch. Both tracked binaries have 11,368 bytes of TLS.

[Build logs, binary hashes and source inventory](../benchmarks/archmage_matched_builds_2026-10-08.meta.json)
record successful fresh builds. Each build peaked at 1.29 GiB RSS; the old
scope took 64 seconds and the new scope took 63 seconds. These are build
resource observations, not decoder performance results.

The initial grain-disabled output gate passed all 32 whole-clip comparisons against dav1d 1.5.3: four clips,
two pins, tracked/untracked modes, and one/four workers at frame delay one.
The clips include 8-bit 1080p and 4K footage and 10-bit 1080p footage. The
output check validates nonzero frame counts and rejects decode/flush errors.
It completed in 121 seconds with peak RSS 0.20 GiB, minimum available
25,795 MiB and peak load 1.19.

`just bench-md5` accepts explicit binaries and streams. `just
bench-paired-modes` runs sequential A/A controls and interleaved A/B tests
for both modes, with exclusive result-file creation. Each phase retains
all timing observations and executable/stream hashes. Compare minimums
and medians against the A/A control before attributing a change to archmage.
Results measured here apply to Zen 4; Zen 5 is not measured.

Resolve release state with the GitHub latest-release and main-commit API
endpoints and the sparse package index rather than a cached release page:
[GitHub releases](https://github.com/imazen/archmage/releases),
[main](https://github.com/imazen/archmage/tree/main),
[package index](https://index.crates.io/ar/ch/archmage).

The first timing campaign was interrupted after seven of eight tracked A/A
cases when review found that timing rows used only the warm-up frame count.
No A/B cases ran. [Full interrupted output](../benchmarks/archmage_first_campaign_incomplete_2026-10-08.meta.json)
is preserved, including the failed phase without a completion marker. The
767-second scope peaked at 0.21 GiB RSS, with minimum available 24,831 MiB
and peak load 2.40. Replacement measurements rebuild both pins with the
[same per-pass count check](BENCHMARK_TIMED_FRAMES.md). Earlier output-MD5
checks still establish parity for their recorded executables, independently
of the unfinished timing run.

The replacement pair uses identical Rust source at f010130a, including both
the per-pass frame-count check and decode/drain/flush error propagation.
[Guarded build proof](../benchmarks/archmage_guarded_builds_2026-10-08.meta.json)
records all binary hashes, matching sources, and CLI controls for both pins
in both modes. Each timing binary rejects the malformed-packet control
before any RESULT and decodes the valid control twice. The four MD5 binaries
are byte-identical to those in the 32 successful dav1d comparisons above.
Both example builds enable archmage's testable_dispatch feature; consumers
without that development-feature unification have not been measured.
The two guarded builds took 63 and 68 seconds, each peaking at 1.29 GiB RSS;
the CLI-control scope peaked at 0.02 GiB. All four timing binaries have
11,368 bytes of TLS. The completed guarded timing results are below.

## Completed version comparison

[Full campaign, phase hashes and configuration](../benchmarks/archmage_version_2026-10-08.meta.json)
and [recomputed statistics and process pairs](../benchmarks/archmage_version_analysis_2026-10-08.json)
retain every observation. Each case has four fresh process pairs, with three
timed passes and one untimed warm-up per process. Frame delay is one; both
arms use native dispatch, all in-loop filters and grain enabled. All 32
[matching grain-enabled output checks](../benchmarks/archmage_grain_output_2026-10-08.log)
match dav1d 1.5.3 exactly. The timing scope finished with rc=0 in 3,003
seconds: peak RSS 0.21 GiB, minimum available 24,637 MiB, peak load 5.13.

The table reports percentage changes in milliseconds per frame. Positive
means slower. A/A compares the same old binary to itself; A/B compares
0.9.30 to 0.9.29. The JSON retains absolute times and all four process-pair
ratios, so these medians do not imply twelve independent experiments.

| Mode | Clip | Workers | A/A median % | A/A minimum % | A/B median % | A/B minimum % |
|---|---|---:|---:|---:|---:|---:|
| Tracked | AOM 1080p 8-bit | 1 | +0.098 | -0.018 | +0.534 | +0.654 |
| Tracked | AOM 1080p 8-bit | 4 | +0.312 | -0.801 | +1.450 | -0.423 |
| Tracked | AOM 4K 8-bit | 1 | +0.151 | +0.147 | +0.143 | +0.138 |
| Tracked | AOM 4K 8-bit | 4 | -1.378 | -0.351 | -0.100 | +0.863 |
| Tracked | SVT 1080p 8-bit | 1 | +0.137 | -0.006 | +0.129 | +0.244 |
| Tracked | SVT 1080p 8-bit | 4 | +3.473 | -2.453 | +4.050 | +2.476 |
| Tracked | AOM 1080p 10-bit | 1 | +0.262 | -0.249 | +0.269 | +0.166 |
| Tracked | AOM 1080p 10-bit | 4 | +1.603 | -0.047 | +1.754 | -0.046 |
| Untracked | AOM 1080p 8-bit | 1 | +0.511 | +0.363 | +0.108 | -0.405 |
| Untracked | AOM 1080p 8-bit | 4 | +0.174 | +0.808 | -0.594 | -1.368 |
| Untracked | AOM 4K 8-bit | 1 | -0.579 | -0.076 | -0.070 | +0.094 |
| Untracked | AOM 4K 8-bit | 4 | -0.283 | -1.731 | +0.237 | +0.892 |
| Untracked | SVT 1080p 8-bit | 1 | +0.278 | -0.021 | +0.126 | +0.153 |
| Untracked | SVT 1080p 8-bit | 4 | +3.590 | +2.784 | -3.234 | +2.708 |
| Untracked | AOM 1080p 10-bit | 1 | -0.141 | -0.109 | +0.096 | -0.516 |
| Untracked | AOM 1080p 10-bit | 4 | +0.918 | +2.388 | +0.420 | -0.438 |

Tracked AOM 1080p at one worker shows a small slowdown: median +0.534%,
minimum +0.654%, with all four process-pair median changes positive
(+0.226% to +0.840%). Its A/A median changes +0.098%. This observation
warrants retaining the regression in the record; it does not establish a
universal cost. Other one-worker A/B medians range from -0.070% to +0.269%.
Four-worker observations are more variable: tracked SVT's median is
+4.050% while its A/A control is +3.473%; untracked SVT's median improves
3.235% while its minimum worsens 2.708%. Their process-pair signs are mixed.
This campaign establishes no broad speedup and no consistent four-worker
version effect.

The adopted 0.9.30 pin retains its reviewed cache-publication correction.
These measurements use the standard example builds with testable_dispatch
unified through development dependencies. They do not measure consumers
without that feature, isolate a particular upstream change, or cover Zen 5.
