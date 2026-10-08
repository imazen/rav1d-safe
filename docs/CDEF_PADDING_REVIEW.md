# CDEF padding review

Performance and full decoder validation remain pending. The preserved
historical comparison used MC candidate `25ef8af1` and archmage `b8d6c077` on
both sides. The current source includes the reviewed MC pair-window follow-ups and
signed-source repair, with archmage 0.9.30 and Rust 1.99.0. New matched
binaries and timing are required; earlier padding-only binaries are preserved.

The padding copies use equal-length slices: zip widens byte pixels to u16,
and `copy_from_slice` copies u16 pixels. Top and line-buffer bottom reads
reserve two exact row segments through `index_rect_as`. Geometry or stride
hint mismatches retain the existing per-row path. `decode.rs` re-declares
the luma byte pitch after resizing both line buffers, which resets the hint.
With tracking enabled, a chroma pitch different from the declared luma
pitch declines that rectangle path. Untracked builds validate rectangle
geometry without requiring a tracker pitch hint.

The three focused tests pass on native Zen 4: all edge flags, four fixture
block shapes, positive/negative strides, picture/line-buffer bottom rows,
and declared/undeclared hints. Both parity sweeps keep neighbouring gap bytes
mutably borrowed while padding copies run; widening a read into those bytes
must be rejected. A deliberate contiguous top-hull reservation fails the 8-bit gap-write
test with an overlap; exact source restoration passes all three tests.
Complete decoder, sidecar and feature gates remain pending.

Inspection of the fat-LTO untracked binaries built with Rust 1.98.1 shows
SSE2 `punpcklbw` widening in both implementations. The padding_8bpc symbols
contain two such instructions before and six after; their code sizes are
2584 and 3173 bytes. Those static counts do not establish a throughput win.
Real 8/10-bit A/B and controls at one/four threads in both builds are required.

[Mutation and restored-source evidence](../benchmarks/cdef_cost_padding_oracles_2026-10-08.meta.json)
records a 14-second failing scope and 13-second restored scope, with peak
RSS 1.61/1.53 GiB. No tracker assertion or extent limit changed.
