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
RESULT row. That negative control is prepared next; its result is not yet
established. Archmage A/A and A/B measurements will use matched rebuilt
binaries containing this same guard in both arms.
