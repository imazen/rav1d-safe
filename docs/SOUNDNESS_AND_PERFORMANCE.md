# DisjointMut: a smaller safety argument and less shared work

Analysis dated 2026-09-05, source baseline `26acb2ba`. This combines source
inspection, a new published-crate reproducer, and explicitly historical
performance records. It is not a complete soundness audit or a new performance
measurement.

## Immediate result: published 0.3.1 fails the guard-move test

The independent [`published-disjoint-mut-audit`](../audit/published-disjoint-mut/README.md)
workspace resolves crates.io `rav1d-disjoint-mut =0.3.1`, with the registry
checksum pinned in its lockfile. The existing, entirely safe
`moving_a_mut_guard_into_drop_is_not_ub` test reports UB under both Stacked
Borrows and Tree Borrows. The protected reference is carried by `drop(g)`;
another thread gains access after the destructor retires its registration but
before that call ends. Both processes abort at the first test; they do not
establish a result for the second test.

Current main uses `NonNull` plus lifetime/variance markers and passed the same
regression target under both models during setup. The published 0.3.1 crate
still contains reference fields. Published rav1d-safe 0.5.7 depends on 0.3.1;
this demonstrates a dependency abstraction defect, not AV1-input reachability.
The existing fix should be reviewed for a focused release before treating the
larger performance branch as release-ready. Nothing was published in this review.

## The abstraction's proof obligations

DisjointMut is a runtime borrow checker over regions. Its purpose is reasonable:
accept several readers, or a writer, for each region while unrelated regions
remain usable. The proof has more parts than an overlap predicate:

| Obligation | Required guarantee |
| --- | --- |
| Storage | Initialized, valid, aligned elements in a live allocation; stable address and length while borrowed; no independent owner/tracker authorizing conflicting access |
| Coordinates | Every registration and returned view describes the same elements, with checked arithmetic and one coordinate convention |
| Exclusion | All overlapping acquisitions meet in a common synchronization domain before a reference can be formed |
| Reference footprint | Every returned reference is contained in the registered region, including temporary references used to construct it |
| Lifetime | Registration outlives every derived reference; moves, casts, unwinding, forgetting and destruction preserve this |
| Visibility | A later conflicting borrower observes the earlier borrower's writes through a valid happens-before chain |
| Type boundary | Storage adapters, casts, variance, Send/Sync and feature combinations preserve every preceding obligation |

For example, registering `[0,8) ∪ [16,24)` cannot authorize a `&mut [T]` over
`[0,24)`. The reference itself covers the gap even if the program intends never
to index it. A rectangle guard should return precisely bounded row references;
the current rectangle guards do this. Registering a larger hull is conservative
for memory safety but can reject legitimate concurrent work and break decoding.

The current narrow path publishes under a shard lock, retires with a per-slot
Release store, and scans those flags with Acquire loads. An independent proof
must cover slot reuse, scans that discard dead records, and the lock-mediated
transitive visibility of writes. Proving that two writers never overlap in
wall-clock time is insufficient without this synchronization. Wide records must
interoperate with narrow records; independently correct tile and address-based
trackers are insufficient if the same bytes can be reached through both.

`Copy` does not prevent data races, torn reads, invalid values or aliasing UB.
The contrary statements in the public documentation have been corrected.
Likewise, poisoning is an API policy for failures, not a foundation on which
reference validity may depend. Leaking a guard should retain its reservation
(possibly harming liveness); it must never authorize premature reuse.

Rust's [`UnsafeCell` documentation](https://doc.rust-lang.org/std/cell/struct.UnsafeCell.html)
explicitly preserves exclusive-reference requirements and the prohibition on
data races. `forbid(unsafe_code)` in a caller cannot repair an unsound dependency.

## How to increase confidence without making the tracker more complicated

1. **Keep one small specification independent of the fast tracker.** Define
   conflict using exact sets of element intervals. Differentially test interval,
   rectangle, negative-stride and mixed typed views against that specification.
   Introduce internal element-offset/stride types, or one validated geometry
   object consumed by both registration and pointer construction, so unit
   conversions cannot drift between the two halves.
2. **Model the concurrency protocol.** Add an instrumentable synchronization
   layer and exercise the real slot/publication/release algorithm with
   [Loom](https://github.com/tokio-rs/loom). Small slot/shard configurations can
   expose missed overlaps, reuse and ordering errors; native tests must still
   cover the production geometry and limits. A separately reimplemented toy
   tracker would not validate the production algorithm.
3. **Use Miri for the actual reference API.** Keep both models, multiple seeds,
   supported storage features, guard moves, casts, container moves where legal,
   unwind paths and mixed read/write workloads. Run targets separately when
   budgets matter, record failures/timeouts, and retain anti-vacuity assertions.
   [Miri](https://github.com/rust-lang/miri) complements model checking; passing
   finite executions is not a universal proof.
4. **Constrain the trusted boundary.** Make the raw storage contract explicit
   about stable pointer identity, provenance, initialization, exclusive
   authority and thread safety. Remove the hazardous default pointer-to-slice
   implementation if every supported adapter already overrides it. Keep
   negative compile checks for guard lifetimes, variance and Send/Sync;
   reproducing historical trait bounds alone is not enough.
5. **Separate experiments from the distributable safe abstraction.** Current
   `__probe_untracked`, `__probe_addnop` and related features can deliberately
   remove exclusion from safe constructors/accesses. A double underscore does
   not protect users from Cargo feature unification. Put these experiments in
   an unpublished test variant or patch; no distributable feature should
   silently turn safe calls into unchecked accesses. Verify the packaged
   tarball and all supported feature combinations, not only a workspace build.
6. **Retain proof obligations as executable regression gates.** Deliberately
   perturb a coordinate conversion, a live-slot scan, or a release ordering in
   an isolated experiment and verify the appropriate test fails. Mutation
   checks show whether a test can detect its claimed defect, while the formal
   argument explains why the implementation should satisfy the general rule.

## What makes the decoder slow

The relevant costs are fine-grained registration, repeated writes to shared
tracker cache lines, scheduling restrictions, and the remaining decoder kernels.
They interact: increasing useful concurrency can expose a tracker bottleneck
that a scheduler barrier previously hid. A constant nanoseconds-per-borrow
estimate or an additive sum of isolated speedups is unreliable.

The committed [M4 Pro c256 study](C256_CONTENTION.md) reports 569,690
registrations/frame and 4.52 → 9.18 → 19.71 ns/registration at 2/4/8 workers.
It attributes only 10.7% of the tracker CPU cost to waiting; changing wait
policy did not materially improve that cell. This is consistent with substantial
cost from shared metadata/cache-line movement, but does not directly prove that
every remaining cycle is coherence traffic. Its 2.378× dav1d checked result and
1.325× tracker-removed result are historical, single-cell measurements. The
unchecked experiment estimates opportunity; it is not a sound implementation
proposal. The residual gap also shows the tracker is not the entire problem.

[The corrected x86 records](X64_APPLICABILITY.md) describe much worse historical
thread scaling on an Intel 265K and invalidate several purported comparison
arms. They are evidence that ARM tuning cannot be assumed to transfer, not
current-main measurements on this Ryzen 7900X. Fresh measurements must verify
binary identity, input/frame equality, thread count, feature set and noise.

One current-source complication is avoidable: `ACTIVE_SHARDS` and
`OBSERVED_TILES` are process-global monotone hints. Later buffer construction
inherits the busiest earlier decoder's hints. This does not itself break
exclusion, because a live tracker's mapping is fixed, but makes performance
depend on process history. Per-decoder/per-buffer immutable configuration would
be easier to reason about and benchmark, especially for mixed server workloads.

## A direction that helps both safety and performance

Use ordinary ownership for private work, immutable views for genuinely completed
data, and dynamic regional guards for remaining shared mutation.

- **Worker-private reconstruction and context scratch:** use ordinary mutable
  slices. `src/owned_recon.rs` already does this with a compact one-superblock-row
  buffer per worker. Its copy-out pays for removing many registrations; copying
  must be priced against the coordination it replaces. Keep memory bounded by
  worker count and band size, rather than full-frame copies per tile.
- **Completed reference data:** introduce a typed transition to immutable
  access only after every possible writer has finished. An `Arc` alone does
  not prove this, and partially reconstructed reference frames need more precise
  publication boundaries. Properly frozen data should not register every read.
- **Shared filters:** acquire exact region authority once for a bounded phase
  or kernel, then derive cheap local views. Current row-returning rectangle
  guards are one building block. Never create a hull reference over unowned
  gaps, and do not widen a reservation into another task's legitimate work.
  Deblocking writes on both sides of an edge, so simple read-only halos do not
  solve it; boundary ownership and phase ordering must be explicit.
- **Longer-term partitioning:** default allocation already owns `Vec<u8>`
  planes before putting them behind `Arc`. The previous claim that raw C
  allocator pointers make splitting fundamentally impossible was incorrect
  for this path and has been corrected in [OWNERSHIP_MODELS.md](OWNERSHIP_MODELS.md).
  A zero-copy partition/publication redesign is possible in principle, but
  must account for every filter, reference reader and FFI consumer. Worker-local
  scratch has a much smaller integration scope today.

The next performance experiment should compare a concrete reduction in shared
authority with its copy/working-set cost on current source. Use paired,
in-process decode timings for each fixed configuration, fresh processes for
global-state isolation, verified pixels and frame counts, and separate thread
curves for photographic, screen, narrow/tall, inter, high-bit-depth and
film-grain content. Keep a local benchmark of the tracker for diagnosis; use
whole-decoder results to decide whether a change helps.
