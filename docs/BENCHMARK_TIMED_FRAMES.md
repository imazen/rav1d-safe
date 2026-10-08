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
