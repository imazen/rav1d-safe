# Warp destination row addressing

Missing: full decoder, feature, sidecar and timing gates on the repair.
Production remains unchanged while the isolated repaired source passes
three scalar tests. [Raw before/after and compile logs](../benchmarks/warp_signed_destination_2026-10-08/meta.json)
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
