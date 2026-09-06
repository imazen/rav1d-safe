# Release soundness argument and executable protocol

Review: 2026-09-06 UTC (2026-09-05 in Denver), against upstream project
`26acb2ba` plus the local audit changes. This is a conditional argument and
regression record, not a machine-checked proof of all Rust executions or of the
entire AV1 decoder. Release approval must distinguish those claims.

## Scope and immediate release conditions

The claim to establish is: **every safe client of checked `DisjointMut` either
receives a valid view or is refused before an invalid reference is formed**.
Clients need not follow rav1d's schedule, avoid contention, or remember to drop
guards. The trusted assumptions are Rust's memory model, correct storage/cast
dependencies, and contracts at explicit `unsafe` boundaries. Malicious unsafe
storage implementations are outside that claim; malformed safe API arguments
are inside it.

Published 0.3.1 fails the entirely safe guard-move test under both Miri models;
see [the pinned published reproducer](../audit/published-disjoint-mut/README.md).
Current main contains the pointer-guard fix. AV1-input reachability through the
published decoder has not been established by that reproducer.

Five measurement features could also disable enforcement beneath the safe
constructor. They now produce a deliberate compilation error:
`__probe_untracked`, `__probe_noscan`, `__probe_lockonly`, `__probe_tinynop`, and
`__probe_addnop`. Their double underscores never protected against Cargo
feature unification. Historical benchmark revisions retain the original
experiments. The negative build gate verifies the specific safety error, so an
unrelated build failure cannot satisfy it.

The supported safety matrix must be stated explicitly. This review targets the
sharded tracker with `std` both enabled and disabled, and the production
`aligned`, `pic-buf`, and `zerocopy` adapters. Decoder `unchecked`, C callbacks,
assembly, experimental trackers, and every probe configuration do not inherit
a universal proof from those checks. Selecting `dangerously_unchecked` requires
the caller to supply the exclusion proof that the runtime normally provides.

## Const-compatible tracker initialization (0.3.2)

The current release retains `const fn DisjointMut::new`. `TrackerStorage` has
three private variants: unchecked, an eager boxed tracker, or a const-created
`spin::Once<Box<BorrowTracker>>`. Both checked variants use the same production
tracker. There is no fallback to an unchecked state when initialization has
not happened, and `is_checked()` remains const and true in that state.

For lazy construction, every admission calls `get_or_init` before registering
and before creating a payload reference. Once chooses one initializer and
publishes its completed value with Release; other callers obtain that same
value through Acquire. The initializer reads the length using the raw storage
metadata API, not a shared reference covering inline payload elements. The
instance never changes its variant or replaces its tracker under `&self`.
A destructor or panic cleanup uses only the already-initialized tracker and
cannot allocate a fresh conflict domain. An initialization panic grants no
capability; Once poisoning prevents use of incomplete state.

Global hints are sampled at tracker initialization (`new_eager` construction,
or first use of `new`). Their timing can affect placement but cannot split a
live instance's conflict domain. Resize and stride declaration still require
`&mut self`. Resize can leave an unused lazy tracker uninitialized; its later
initializer reads the current length. Stride declaration initializes first so
the hint is installed on the same tracker used by later borrows. Leaked guard
records may remain; no *usable* guard/reference can survive the exclusive
borrow. Reconfiguration does not rely on `Drop` having run.

The new wrapper contains no unsafe code. It adds a trusted synchronization
dependency: `spin` 0.12.3 with only the `once` feature, no std requirement, and
MSRV 1.71 (below this crate's 1.85). Its production initializer publication is
reviewed in source and exercised under native execution and both Miri models.
**The six Loom record models do not instrument spin::Once initialization.**
They continue to check the shared record protocol after initialization. This
boundary is explicit; passing those models is not a proof of the new dependency.
The initializer identity test checks exactly one closure and one returned
tracker address during concurrent first calls. Public API tests additionally
exercise const/statics, both constructors, global-hint changes with a live
lease, resize/stride transitions, moved buffers, leaked guards, and poison.

`new_eager`, `Default`, and allocating slice constructors avoid the Once load
on borrow/drop. The decoder uses `new_eager`. The wrapper remains small and is
covered by the decoder's unchanged 48 KiB task-context size gate. This change
has no new performance measurement; it preserves the eager algorithm and
keeps the earlier performance investigation within the requested time budget.

## Abstract state and obligations

For one storage owner, let `R` be a set of **elements of `T::Target`**, and let a
live capability be `(R, shared)` or `(R, exclusive)`. Two capabilities conflict
iff their sets intersect and at least one is exclusive. A rectangle is the
union of its row intervals; its gaps are not elements of that capability.

| Obligation | Argument in the implementation | Executable challenge |
| --- | --- | --- |
| One live storage authority | Stable initialized allocation; adapters cannot manufacture independent trackers for aliases; external adapter contract states this explicitly | Borrowed storage moves; aligned and PicBuf Miri suites; unsafe adapter review |
| One coordinate system | Normal and typed ranges convert to storage-element units before registration; rectangle rows and registration describe the same set | Independent enumerated element-set oracle, `rect_units`, typed inclusive endpoints, extreme geometry |
| Admission before references | Validate, register, then store a raw pointer in the guard; `Deref` creates a reference only while registration remains live | Miri safe-client tests and overlap controls |
| Shared conflict domain | An overlapping element's block selects a shard held by both registrants; wide acquisitions hold the entire instance's active prefix | Loom narrow/multi/wide models, native wide-path anti-vacuity test |
| Exact reference footprint | Rectangle guards return one row at a time; no temporary slice spans unregistered gaps | Rectangle/gap oracle under both Miri models |
| Capability survives references | Guards contain `NonNull` and lifetime/variance markers; moving a guard does not protect a reference across its destructor | Published/current guard-move comparison; compile-fail lifetime/resize/row-alias examples |
| Retirement publishes writes | A live slot retires with Release; admission observes flags with Acquire; the shard lock carries accumulated visibility to later registrants | Loom payload cells and retirement-order mutation |
| No accidental early retirement | Borrow ID identifies the exact slots; retiring one reader does not retire another; forgetting keeps reservations | Full-slot/wide-reader model, leak/reuse test, native churn and narrow-release suites |
| Global hints cannot change old capabilities | Mapping is stored per tracker, and block storage choice is retained per guard | Global-hint Loom model and same-process live-block transition test |

## Concrete tracker protocol

1. **Validate geometry before touching payload.** Establish allocation bounds,
   alignment, nonoverflowing offsets, and the exact set of elements a view can
   reference. Empty views expose no elements. Refusing an unrepresentable
   rectangle returns `None`; callers must use checked per-row fallback.
2. **Acquire the admission domain.** Narrow admissions acquire their sorted,
   deduplicated shard set. Wide admissions acquire all active shards. All use
   the immutable instance mapping. No acquisition promotes to wide while
   retaining a subset of locks.
3. **Scan a consistent domain.** Inspect live flags and exact records, including
   the wide list. The single-shard fast and contended paths must recheck `state`
   *inside* their lock: a pre-lock check cannot exclude a wide record installed
   in between. Rectangles similarly recheck before committing.
4. **Publish once the admission is complete.** A multi-shard admission reserves
   slots while retaining all its locks and publishes only after all allocations
   succeed. An abandoned partial reservation is never a live capability.
5. **Access only the capability.** Payload references are bounded by borrowing
   their guard. Copies require valid read capabilities over every copied byte.
   `Copy` does not permit races or invalid references.
6. **End references, then retire.** A one-slot retirement stores zero with
   Release. Multi-slot and wide retirements retain the required lock set.
   Poisoning may restrict later operations; memory safety cannot depend on it.
   Forgetting may lose liveness, but must never create an early retirement.

The slot's `allocated` bitmap is a conservative superset, protected by the
shard lock. It may remember a dead slot. It must never omit a live slot. The
allocator's Acquire observation of a retiring zero imports the previous
borrower's payload writes. Subsequent lock release/acquire edges transmit that
visibility even if later scans no longer include that slot. A new allocator
cannot observe the *initial* zero while ignoring an already-published live
record: publication and the lock handoff establish the relevant happens-before
and atomic coherence constraints.

Native `TinyLock` separately requires the usual spin-lock argument: only an
atomic swap reading false grants ownership; an Acquire success pairs with the
previous owner's Release unlock. Failed swaps grant nothing. Spin fairness and
eventual visibility are liveness assumptions, not memory-safety permissions.
Manual multi-lock sections also require review for panic/overflow paths; a
stranded lock is a liveness defect even when it cannot create aliasing UB.

## Process-global threading audit

`TILE_THREADING` is a safety-sensitive policy latch, unlike the two tracker
placement hints. Historically, opening a single-threaded decoder could store
false while another decoder's tile workers were live. Checked hull guards then
rejected legitimate neighbouring work; disabling checks could expose
conflicting references. Current `set_tile_threading(false)` does not store.

The relaxed ordering is supported by a specific publication argument:
`rav1d_open` stores true **before spawning any worker for that multithreaded
decoder**. Worker creation establishes happens-before to its reads; subsequent
jobs use those workers and synchronized context state. Each multithreaded
decoder performs its own store. No later call can reset it. A single-threaded
decoder observing true merely takes the narrower policy. Changing this to an
existing global worker pool would invalidate the startup argument unless the
job handoff supplies the corresponding edge; that change requires a new gate.

A block acquired before that transition remains `BlockMutStorage::Direct`.
Its destructor dispatches on its stored representation, not a fresh global
flag. A compact block similarly retains its own buffer/stride and writes back
under per-row guards. The new transition test holds an actual direct guard,
opens real worker threads, opens single-threaded decoders from another thread,
then checks old-guard retirement and new-block writeback alongside a live gap
borrow. A fresh child process gives a known initial false value, and a marker
proves the child actually executed the assertions.

`ACTIVE_SHARDS` and `OBSERVED_TILES` are sampled at tracker construction or
exclusive reconfiguration. The mask, shift, and row stride used to select,
compare, and retire records live on the tracker. Sampling old or mixed hint
values changes placement cost, not agreement between accesses to that same
instance. Reconfiguration requires `&mut`, excluding usable outstanding safe
references. A forgotten guard may leave conservative records behind; it cannot
later release a record through a recovered safe capability.

The CPU mask is also global, but intersects detected/compile-time CPU features;
it cannot add an unsupported instruction capability. Existing dispatch objects
retain their selected functions. Ablation switches and counters are separate
experimental state, and are not evidence for a threading safety guarantee.
This inspection found no new default-path global reset hole beyond the
historical one; it does not justify making safety depend on process-global
configuration in future APIs. Explicit per-instance policy would make this
argument substantially easier to maintain.

## What Loom does and does not establish

`--cfg disjoint_mut_loom` compiles the production record algorithm with Loom
atomics and tracked metadata cells. The metadata access guard lives as long
as its references and is dropped before explicit unlock. Payload cells make
missing visibility an error even when admissions happen sequentially.

The model uses four shards and the real seven slots, with explicit assertions
that wide and multi-shard paths are reached. It explores schedules up to two
preemptions by default, with no permutation or time cap. Native tests cover
the normal shard configuration. `LOOM_MAX_PREEMPTIONS` extends the model bound.

Only lock waiting is abstracted by a Loom Acquire/Release mutex. Directly
instrumenting the native relaxed-load spin encountered unbounded stale-read
executions and exceeded the branch limit; those runs were failures, not proof.
The mutex adapter introduces no payload access. Its private erased guard
lifetime is confined to a live, immovable tracker and ended by `unlock`.
This explicitly leaves native lock implementation/fairness outside the model.
See [Loom's instrumentation and yielding documentation](https://docs.rs/loom/0.7.2/loom/).

The model checks actual tracker code, but does not model the decoder scheduler,
all buffer lengths, the complete Rust aliasing model, every architecture, or
arbitrary external unsafe implementations. Miri supplies complementary
reference/provenance checks; neither tool converts finite coverage into a
universal theorem.

## Reproducible release gates

Run heavy commands serially through `scripts/review.sh` or the workspace's
`run-heavy` wrapper. Record source revision and diff, rustc/Miri versions,
features, flags, Cargo.lock hash, corpus revision, and full outcome logs.

```sh
scripts/review.sh disjoint
scripts/review.sh no-std
scripts/review.sh loom
scripts/review.sh miri-stacked --test adversarial_api --test rect_units --test soundness
scripts/review.sh miri-tree --test adversarial_api --test rect_units --test soundness
scripts/review.sh threading-protocol
scripts/review.sh decoder-debug
scripts/review.sh decoder-smoke
```

Also gate the published guard-move regression, mutation controls, rejection of
each unsound feature, package contents/build, MSRV/no-std, and the intended
decoder feature/target matrix. API text diff must retain auto traits and `const`
qualifiers; semver checks supplement it, and behavioral changes such as strict
decoding defaults require manual release notes.

For each bug claim, retain a reproducer, a healthy control, and a deliberate
mutation that fails for the claimed reason. An overlap panic must not swallow
a Loom causality failure; a renamed/filtered-away test must not count as a pass.
Publish only from the exact packaged revision after its declared matrix passes.
No crate, advisory, or upstream message is published by this local review.

See the [API review](../audit/api/README.md),
[ownership experiment ledger](OWNERSHIP_MODELS.md#experiment-ledger-reviewed-2026-09-06),
and [initial soundness analysis](SOUNDNESS_AND_PERFORMANCE.md) for linked evidence.
