# CDEF padding review

The landed revision keeps the equal-length widening copies but drops the
rectangle borrows after paired timing. An earlier draft reserved two exact
row segments through `index_rect_as` and had `decode.rs` declare the luma
pitch on both line buffers so the tracker could decode rectangle geometry;
on a quiet box the tracked 4-worker cells of the staged build came back
+2.2 to +3.3% over their A/A controls (untracked and 1-worker unchanged).
Bisection showed the records were neutral and the stride declaration was
the cost: `declare_row_stride` re-derives the tracker block shift via
`block_shift_rule_rows`, coarsening blocks toward `ROWS_PER_BLOCK_MIN` rows
whenever the buffer sees `tiles >= 2` at `threads > 1` — the shared line
buffers then collide on fewer shard locks under tile parallelism. With the
declares and rectangle sites both reverted, tracked 4-worker timing is
inside A/A noise and 1-worker cells kept their small gains.

The padding copies use equal-length slices: zip widens byte pixels to u16,
and `copy_from_slice` copies u16 pixels. Top and line-buffer bottom reads
stay on the per-row `slice_as` path. `decode.rs` declares no row pitch on
the line buffers — declaring one is what made the measured regression, and
without it `index_rect_as` declines, so the rectangle path is not used at
all. `index_rect_as` itself remains available where a declared buffer's
geometry benefits from it (e.g. the pre-existing loop-filter caller).

The three focused tests pass on native Zen 4: all edge flags, four fixture
block shapes, positive/negative strides, picture/line-buffer bottom rows,
and declared/undeclared hints. Both parity sweeps keep neighbouring gap bytes
mutably borrowed while padding copies run; widening a read into those bytes
must be rejected. A deliberate contiguous top-hull reservation fails the 8-bit gap-write
test with an overlap; exact source restoration passes all three tests.

Inspection of the fat-LTO untracked binaries built with Rust 1.98.1 shows
SSE2 `punpcklbw` widening in both implementations. The padding_8bpc symbols
contain two such instructions before and six after; their code sizes are
2584 and 3173 bytes. Those static counts do not establish a throughput win.
Real 8/10-bit A/B and controls at one/four threads in both builds are required.

[Mutation and restored-source evidence](../benchmarks/cdef_cost_padding_oracles_2026-10-08.meta.json)
records a 14-second failing scope and 13-second restored scope, with peak
RSS 1.61/1.53 GiB. No tracker assertion or extent limit changed.
