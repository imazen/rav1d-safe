# Warp destination row addressing

Missing: timing gate and production landing of the repair.
Production remains unchanged while the isolated repair passes both full
decoder suites and three scalar tests. [Raw before/after and compile logs](../benchmarks/warp_signed_destination_2026-10-08/meta.json)
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
throughput validation remains pending.

[All 32 whole-clip comparisons](../benchmarks/warp_signed_destination_2026-10-08/clips.meta.json)
match dav1d 1.5.3 with grain enabled, across four 8/10-bit clips, both
modes, before/after and one/four workers at delay one. The 120-second
scope returned rc=0, peak RSS 0.20 GiB, minimum available 25,565 MiB
and peak load 1.15. Both new timing binaries also reject a malformed
second packet without printing timing results; valid input and MD5
frame-limit controls pass. That scope reports peak RSS 0.02 GiB,
minimum available 25,820 MiB, peak load 0.42 and rc=0.
The completed sidecar gate appears below; A/A plus A/B remains pending.

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
timing and production landing remain pending.
