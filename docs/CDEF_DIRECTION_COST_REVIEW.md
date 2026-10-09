# CDEF direction-cost experiment

Both cost variants compile and pass the same scalar direction/variance and
padding oracles, and were timed against each other with paired A/A controls.
The contiguous iterator loop won; it is the variant this revision lands.

The preserved Rust 1.98.1 untracked padding candidate has 88 scalar multiply
instructions in its 8bpc direction-finder symbol. That static count does not
measure execution time. The preserved parent experiment widens sixteen signed line sums into two
explicit eight-lane i32 vectors, squares and weights them with AVX2
multiplication, then reduces their exact integer sum. A preserved alternate child
instead squares and weights those sums in one contiguous iterator loop
inside the AVX2 feature context. The landed revision uses the contiguous
loop: on a quiet box the paired 4-round rerun put 15 of 16 A/A cells within
2% (the remaining cell, safe svt-1080p 4-worker at 1.022, re-measured clean
at 8 rounds with noise-level A/B). Of the clean cells the loop was faster in
8 (aom-1080p/4K/aom10 at 1 worker and most 4-worker cells, 0.5-1.2%),
the explicit form in 1 (untracked aom-1080p 4-worker, 1.1%), and the rest
sat inside their own A/A noise. Pixel accumulation and
first-direction tie ordering stay as before. The same helper serves
normalized 8/10/12-bit inputs.

For each direction, lines partition the 64 normalized pixels. For line
length L, Cauchy-Schwarz gives sum(line)^2 / L <= sum(pixel^2).
Each normalized pixel is between -128 and 127, so cost is at most
840 * 64 * 128^2 = 880803840. Every weighted square and nonnegative partial
sum fits i32; rearranging the exact integer sum cannot change it.
Diagonal weights are 840/L for lengths 1..8..1; alternate lengths are
2,4,6,8,8,8,8,8,6,4,2. Padded lanes have zero weight.

The new scalar oracle compares direction and variance at every bit depth,
both stride signs, all 256 flat 8-bit levels scaled to the native maximum,
checkerboards, ramps, one-pixel impulses and inverse impulses, and 4096
deterministic random blocks per sign/depth. It holds the token lock and
an unrelated gap write while both implementations read the block.
Run `just test-cdef-direction-costs` and `just test-cdef-padding` before
matched-binary profiling or full decoder validation.

The current Rust 1.99.0 MC comparison binary has 77 scalar multiply
instructions, four `vpmaddwd`, and five `vpmulld` in its 2,372-byte 8bpc
direction symbol. It uses the reviewed MC snapshot `f910b2f2` and archmage
0.9.30. Keep this static comparison separate from the older Rust 1.98.1
counts and from measured decode throughput.

The negative-stride gap write is now placed in the next lower row, inside
the bounded hull but outside the eight pixels being read. The unchanged
direction/variance comparisons still cover both stride signs. This fixture
strengthening passes for both cost variants.

The contiguous-loop and explicit-AVX2 variants each pass the direction
oracle in 0.015 seconds and all three padding tests. Their 19/14-second
wrapped scopes peak at 1.59/1.72 GiB RSS. [Raw logs and hashes](../benchmarks/cdef_cost_padding_oracles_2026-10-08.meta.json)
identify both source variants; neither is a throughput measurement.

The landed comparison held the signed warp destination repair and line-buffer
row-pitch hints identical in both cost variants. Independent source
comparison confirmed that the explicit and contiguous-loop files differ
only in `cdef_weighted_cost`. The loop's paired-timing win over the explicit
form (8 clean cells to 1) is recorded above; the explicit child was dropped.
