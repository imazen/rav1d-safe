# Warp destination row addressing

Missing: an optimization resolving the measured untracked 4K one-worker timing cost.
The production repair passes both full decoder suites, three scalar tests
and the runtime-tier sidecar matrix. [Raw before/after and compile logs](../benchmarks/warp_signed_destination_2026-10-08/meta.json)
record executable source fingerprints and wrapped resource lines.

A whole-plane comparison with the original scalar warp reproduces a panic
when the destination stride is negative and the picture policy selects the
direct path. The dispatcher discarded the prefix needed by backward row
walks, and the vertical helper cast a negative offset to usize. The failure
reproduces on the landed MC source, independently of paired horizontal
warp arithmetic. Positive destination/signed source put and prep pass.

The repair passes the full bounded destination slice and a separate byte
base through the private 8-bit and 10/12-bit put helpers. Row addresses add
the signed stride to that base. Interpolation and rounding are unchanged;
prep and public API signatures are unchanged. The existing tight/full
reference fixture supplies zero for its zero-based destination arrays.

The unchanged 8-bit negative-destination assertion now passes under one
and four-worker picture policies. New 10/12-bit cases cover both source
stride signs, four endpoint/random patterns, two affine phase cases and
one/four-worker policies; the entire destination plane and source are
compared, including padding. The broad 8-bit whole-warp put/prep oracle
also passes all 7,768 cases. This focused scope returned rc=0 after fourteen
seconds, peak RSS 1.62 GiB, minimum available 24,181 MiB and peak load 0.82.
These tests do not establish C-FFI negative-stride allocator coverage.

## Feature validation

[Feature scope and exact source hashes](../benchmarks/warp_signed_destination_2026-10-08/features.meta.json)
record passing tracked/untracked all-target lint and ARM/WASM/C-FFI checks.
All three scalar warp tests also pass untracked (0.017 seconds). The
wrapped scope returned rc=0 after 51 seconds, peak RSS 1.57 GiB, minimum
available 24,206 MiB and peak load 2.45. An extra filter used a nonexistent
reference-test name and selected nothing; this is not claimed as reference
fixture coverage. The corrected focused recipe includes the real test,
`warp_reference_windows_cover_both_strides_and_all_depths`, and its completed two-mode
gate and complete decoder suites are recorded below. No assertion changed.

## Complete decoder validation

The [full two-mode gate](../benchmarks/warp_signed_destination_2026-10-08/full-modes.meta.json)
passes all 240 tracked and 223 untracked selected tests, including Argon,
generated threaded vectors, backpressure, MD5 and token permutations.
Both modes pass the four focused warp/reference tests and ten active
doctests. Eighteen existing ignored tests and thirteen ignored doctests
per mode remain outside this scope. Source fingerprints were checked
before and after execution; paired horizontal warp arithmetic is absent.
The wrapped scope returned rc=0 after 1,824 seconds, peak RSS 1.54 GiB,
minimum available 17,927 MiB and peak load 13.50. Full logs are preserved
in ordered bounded parts. No throughput claim follows from these gates.

[Fresh matched binaries](../benchmarks/warp_signed_destination_2026-10-08/builds.meta.json)
use Rust 1.99.0, archmage 0.9.30, the same lockfile and guarded timing
example, generic code generation and release fat LTO in both modes.
The preserved before binaries match published MC production source;
only the signed destination repair changes compiled production code.
The 71-second build scope returned rc=0, peak RSS 1.30 GiB, minimum
available 24,107 MiB and peak load 3.28. Whole-clip MD5 and sidecar results appear below;
The completed timing matrix appears below.

[All 32 whole-clip comparisons](../benchmarks/warp_signed_destination_2026-10-08/clips.meta.json)
match dav1d 1.5.3 with grain enabled, across four 8/10-bit clips, both
modes, before/after and one/four workers at delay one. The 120-second
scope returned rc=0, peak RSS 0.20 GiB, minimum available 25,565 MiB
and peak load 1.15. Both new timing binaries also reject a malformed
second packet without printing timing results; valid input and MD5
frame-limit controls pass. That scope reports peak RSS 0.02 GiB,
minimum available 25,820 MiB, peak load 0.42 and rc=0.
The completed sidecar and A/A plus A/B gates appear below.

## Complete runtime-tier sidecars

The [full forty-configuration gate](../benchmarks/warp_signed_destination_2026-10-08/sidecars.meta.json)
passes all 803 official sidecars in tracked and untracked modes, at
one/two/four/eight workers and scalar/v2/v3/v4/native tiers, with frame
delay zero. Every configuration has zero mismatches or decode errors;
149 manifest rows remain outside the explicit 803-vector selection.
The eight generated manifests are byte-identical. Compared with the
previous MC gate, only their absolute workspace prefix changes; the new
raw manifest is archived separately with its actual hash. Source and
executable fingerprints were verified unchanged after execution.

The wrapped scope returned rc=0 after 4,875 seconds, peak RSS 0.24 GiB,
minimum available 25,480 MiB and peak load 1.90. Ordered bounded parts
reconstruct the complete log and manifest. This is correctness evidence;
the timing-cost investigation remains open.

## Matched timing matrix

The [complete sixteen-case A/A and A/B matrix](../benchmarks/warp_signed_destination_2026-10-08/timing.meta.json)
uses four fresh process pairs per case and three timed passes per invocation,
four 8/10-bit clips, one/four workers, both modes and frame delay one.
Source, stream and executable hashes were verified after execution.
Both arms use the same guarded timing example and Rust 1.99.0 with generic
release fat LTO; their example builds unify `testable_dispatch`. Ordinary
consumer builds without that feature and Zen 5 were not measured.

Untracked AOM 4K at one worker has median A/B +1.2898% and minimum
+0.8796%; all four paired medians increase. Its A/A median is -0.1752%
and minimum -0.0399%. The focused eight-pair repeat below also observes a cost. Tracked 10-bit at one worker also has four
positive paired medians: A/B +0.5375% and minimum +0.3900%, while its
A/A median is +0.6028% and minimum +0.0657%. These results do not
establish performance neutrality or a general speedup. The full analysis
retains every paired ratio, including mixed four-worker results.

The wrapped timing scope returned rc=0 after 2,971 seconds, peak RSS
0.21 GiB, minimum available 25,507 MiB and peak load 1.80. Full raw
phase files, complete command output and analysis are committed.

## Primary landing gate

The [primary checkout gate](../benchmarks/warp_signed_destination_2026-10-08/primary-gate.meta.json)
verifies that all four source fingerprints match the fully tested isolated
repair. Four focused tests pass in each mode, both release all-target lint
recipes pass, and ARM, WASM and C-FFI checks pass. The initial invocation
stopped at a missing ARM target for Rust 1.99.0 after the tracked tests and
both lints had passed; installing that toolchain's ARM/WASM targets and
rerunning the feature recipe resolved it. Both complete logs are retained.
The successful retry returned rc=0 after 35 seconds, peak RSS 1.49 GiB,
minimum available 24,101 MiB and peak load 1.64.

## Focused 4K one-worker repeat

The [eight-pair untracked repeat](../benchmarks/warp_signed_destination_2026-10-08/repeat-4k1.meta.json)
uses the same preserved binaries and clip, with 24 timed observations per
arm. Median time changes from 74.5512185 to 75.218028 milliseconds per
frame (+0.8944%); minimum ratio is +0.2672%. Seven of eight process-pair
medians increase. A/A has median -0.0342%, minimum +0.1558%, and four
positive/four negative paired medians. The original four-pair cost is
observed again; this does not establish a slowdown on every workload.
The negative-stride correctness repair remains required, while its timing
cost remains an optimization target. No pixel expectation was relaxed.

The wrapped scope returned rc=0 after 460 seconds, peak RSS 0.20 GiB,
minimum available 25,534 MiB and peak load 1.24. The completed phase
files, full command output, recomputed analysis and metadata are committed.
Executable hashes remain unchanged.

## Standard put/prep (2026-10-09)

The same oracle, extended to standard (non-warp) put and prep, reproduced the
same wrapped-usize slice start on the unrepaired parent `ee3a594c`: both new
tests fail, at `src/safe_simd/mc.rs` in the 8-bit and 16-bit put helpers
(`mutation-baseline-parent-ee3a594c.log`). `54f78fb2` applies the warp
repair's shape: the helpers take the bounded slice plus a base and add each
signed row offset to it.

Paired timing against the parent (both modes, four real 8/10-bit clips, one
and four workers, delay one, four process pairs of three passes, A/A first):
one-worker A/B medians -0.66% to +0.20% (minimums -0.39% to +0.38%) against
A/A medians -0.28% to +0.55%. Four-worker A/B medians range -5.54% to +0.97%
against A/A -2.30% to +4.76%; the two extreme four-worker cells sit next to
A/A controls that moved as far. Untracked 4K, which showed the warp repair's
cost: one worker -0.04% median / +0.38% minimum, four workers +0.43% / -0.46%.
No cost is distinguishable from noise; Zen 5 is not measured.

Recompute: `just bench-paired-mode-report benchmarks/mc_put_prep_signed_rows_2026-10-09 <safe|untracked>`.
