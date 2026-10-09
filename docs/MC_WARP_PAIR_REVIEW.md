# Paired warp-dot experiment

Missing: compilation, executed parity and decoder gates, codegen inspection,
and matched tracked/untracked A/A plus A/B timing. No speedup is claimed.

The baseline is the reviewed MC pair-window candidate with Rust 1.99,
archmage 0.9.30 and MC source SHA
`5eef8c20ef03bb11d1c922cecb47727edf96ca7cf002bc162f33b3dddc8dc8db`.
This separate child changes the 8-bit horizontal warp pass. Two independent
eight-tap dot products occupy separate 128-bit lanes of an AVX2 vector.
Source bytes widen without sign, coefficients widen with sign, and multiply
and reduction use i32. The horizontal rounding and mid-buffer layout remain
as before. The vertical and 10/12-bit arithmetic are unchanged.

For any eight signed-byte coefficients and unsigned-byte pixels, the sum
of absolute products is at most `8 * 255 * 128 = 261120`. Every i32 pair
and partial reduction therefore fits without saturation or wrapping. The
existing horizontal shift by three also fits i16. This arithmetic bound
does not establish correct lane order or whole-warp pixels.

The new dot oracle covers all 193 by 193 coefficient-row pairs at their
linear extrema and deterministic random pixels. A second gate covers every
uniform byte value and every maximum impulse for each filter row. The
whole-warp oracle compares put and prep with the original scalar functions,
including all filter indices, affine phase cases, both source-stride signs,
four pixel patterns, complete output buffers and untouched padding. Prep
uses an odd thirteen-element row stride. An unrelated source row remains
mutably borrowed throughout each decode. Tests hold the token lock.

Run `just test-mc-warp-pairs` before building timing binaries. These tests
are currently prepared only. The broad whole-warp fixture has positive destination stride. A separate
unrun oracle also compares negative destination rows with scalar, under
one/four-worker picture policies, including whole-plane padding. Neither fixture
establishes negative destination coverage until it executes successfully.
The new arithmetic is 8-bit only. Timing must include 10-bit clips as a
control and both modes at one/four workers, with an A/A control, on the same
compiler, dependency pin and guarded profile example.
