# Native picture-plane admission

`Settings::max_picture_bytes: Option<usize>` admits the sum of requested native
plane allocations before a picture allocates its first plane. The total includes
native row padding and alignment, reference frames, reconstruction pictures,
film-grain copies and pictures retained by callers. `None` retains the existing
pooled allocation path; `Some(0)` refuses every picture.

This is a scoped storage contract. It does not bound decoder workspace, input,
metadata, thread stacks, allocator overhead or excess allocator capacity. It is
not a bound on process RSS. The existing pixel-count limit remains independent.
The managed C-FFI build refuses this new requirement at construction because
its custom allocation callbacks have not been qualified for it.

The reservation belongs to the native picture data, shared by its references
and frame clones. It remains live after reset or decoder destruction while an
owner retains the picture. An allocator copied from a retained picture shares
the same budget. Refusal reports the traced native
`Error::Other("native picture storage limit exceeded")`; the existing exhaustive
public error enum is unchanged.

Bounded mode frees plane vectors instead of returning them to the reusable
pool. Otherwise the charge could disappear while cached storage remained
allocated. This trades cache reuse for a smaller ownership contract. Its cost
must be measured separately; these tests establish no throughput claim. A
future charged cache must retain admission with the cached allocation and
admit any replacement before allocating it.

`just qualify-picture-budget` verifies exact admission boundaries across
native plane geometries, atomic contention and overflow, unwind cleanup,
cloned/retained owners, allocation through copied pictures and real 10-bit
video on one and four workers. Tracked and untracked builds pass; the C-FFI
rejection gate, existing packet/provenance checks, strict Clippy, WASM check,
doctests and API capture also pass. The doctest run retains the pre-existing
ignored low-level examples.

Validation on 2026-10-25: run-heavy rc=0, 81 s, peak RSS 1.15 GiB,
minimum available 14231 MiB, peak load 2.43. This is the test/build process
receipt, not a measurement of the decoder's picture budget.
