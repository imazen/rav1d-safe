# Timed decoder frame counts

`profile_ivf` used to report its warm-up frame count on every timing row,
without checking the counts returned by timed decodes. That could hide a
short timed decode behind a plausible per-frame denominator. No such short
decode was observed in the interrupted archmage experiment; the missing
check is a measurement-integrity finding.

Warm-up must now produce a nonzero count. Every timed decode must return
exactly that count before a timing row is emitted. The A/B driver explicitly
selects all in-loop filters, native runtime dispatch and its requested
threads/frame delay; inherited sampling and ablation switches are rejected.
A three-pass 96-frame clip succeeds with the new check. The wrapped build
and CLI check took 31 seconds with peak RSS 1.28 GiB, minimum available
24,413 MiB and peak load 7.40. This is validation, not a speed comparison.

`just bench-frame-count <binary> <stream> <threads> <repetitions>` reproduces
the CLI check. A deliberately shortened timed count must fail before any
RESULT row. The deliberately shortened count (95 versus 96) fails before any RESULT
row. Exact source restoration and a fresh rebuild pass all three timed
decodes and release example clippy with warnings denied. [Full positive,
mutation and restored logs](../benchmarks/timed_frame_count_2026-10-08.meta.json)
record 19/34-second mutation/restored scopes, each peaking at 1.29 GiB RSS. Archmage A/A and A/B measurements will use matched rebuilt
binaries containing this same guard in both arms.

The same review found that frame-drain and flush errors were ignored, and
ordinary decode errors printed diagnostics but still allowed RESULT rows.
The example now fails on all three error returns. The unchanged CLI oracle
uses a committed valid frame followed by a forbidden-bit OBU header:
before repair it exits zero and prints two timing rows; after repair it
exits 101 before any row. A valid one-frame control produces both requested
timing rows. Existing MD5 frame-limit checks and release example Clippy
also pass. [Before/after evidence](../benchmarks/benchmark_errors_2026-10-08.meta.json)
records the 19-second passing scope: peak RSS 1.28 GiB, minimum available
24,977 MiB and peak load 1.63. The malformed control induces decode failure;
drain/flush handling was source-reviewed, not separately induced.
`just test-benchmark-errors <decode_md5> <profile_ivf>` reproduces the gate.

Whole-clip output checks must use the same grain setting as timing. The
profile example uses `Settings::default()` with grain enabled. `just
bench-md5 --filmgrain` now enables grain in both dav1d and each selected
MD5 binary; the default remains explicit grain-disabled comparison.
[Grain-enabled validation](../benchmarks/archmage_grain_output_2026-10-08.log)
passes all 32 comparisons: four clips, two archmage revisions, both modes,
and one/four workers. Their MD5s also match the prior grain-disabled output
for these clips. The 121-second scope peaked at 0.20 GiB RSS, with minimum
available 25,787 MiB and peak load 1.19.

`just bench-paired-report <directory>` requires all four A/A and A/B phases
to have completion markers, matching configurations and streams. It
recomputes medians and minimums from raw samples and retains process-round
ratios: timed passes within one process share scheduling and caches. The
report rejected the live unfinished campaign, then accepted all sixteen
cases after completion. No winner threshold or significance claim is
inferred by this report.
