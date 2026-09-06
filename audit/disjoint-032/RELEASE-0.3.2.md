# 0.3.2 maintenance release protocol

This candidate starts at `dd60e0a61d88121fa764094f71d1d1de537f1f9f`.
Every Rust implementation file in that revision was compared byte-for-byte
with the checksum-verified crates.io 0.3.1 archive before editing. It retains
the published tracker rather than taking the development implementation.

## Why 0.4.0 was proposed, and why 0.3.2 works

The development implementation allocates a sharded tracker during construction
and no longer declares `DisjointMut::new` as `const`. Existing code such as
`static BUFFER: DisjointMut<[u8; 8]> = DisjointMut::new([0; 8]);` stops compiling.
That is a compatibility break; the reported 0.4 requirement was about shipping
that implementation unchanged. See the [Cargo compatibility guidance](https://doc.rust-lang.org/cargo/reference/semver.html)
and [const-function contract](https://doc.rust-lang.org/reference/const_eval.html#const-functions).

The safety correction needs no such change. The published tracker already
initializes its inline records and lock in a const context. Keep that tracker,
replace reference fields in both guard kinds with `NonNull`, and preserve the
reference lifetimes/variance through `PhantomData`. Guard dereferencing derives
a reference whose lifetime is bounded by the borrow of the guard. A guard
passed by value into `drop` no longer protects a reference beyond retirement.
Each of the four zerocopy cast paths transfers the same reservation to the new
guard. A failed cast conservatively leaks its reservation, as in 0.3.1.

| Contract | 0.3.1 → 0.3.2 |
| --- | --- |
| Public functions, methods, traits and bounds | Preserved |
| `new`, `is_checked`, `inner`, `dangerously_unchecked` const use | Preserved |
| Guard lifetime and variance | Preserved using reference-shaped markers |
| Exclusive guard `Send` / `Sync` | Still `T: Sync`, and respectively `V: Send` / `V: Sync` |
| Shared guard `Send` / `Sync` | Still `T: Sync` and `V: Sync` |
| Cargo features and defaults | Same published feature table; no check-disabling probes |
| MSRV and no-std | Rust 1.85 retained; aligned error paths use `alloc` rather than `std` |
| Normal-build dependencies | Unchanged; Loom is confined to a dedicated test cfg |
| Unsafe storage-adapter documentation | Existing memory validity and exclusion obligations made explicit |

The three index/range traits were already sealed in 0.3.1. This patch does not
unseal them or introduce additional sealing. There is no rectangle API or
borrowed mutable-slice adapter in this maintenance surface. The newer
development APIs remain on the development line.

## Conditional soundness argument

For each storage owner, a borrow record is `(start, end, mutable)`, in storage
element units. Two nonempty intervals conflict if they intersect and either is
mutable. One instance lock protects **every** admission and retirement, across
all 64 inline slots and the overflow vector. There is no shard mapping or
process-global threading mode in this tracker.

1. Storage adapters supply a stable, initialized allocation and cannot expose
   aliases through independent trackers. Inline adapters do not create an
   intermediate reference spanning live element borrows.
2. Sealed indices return pointers matching the recorded intervals. Admission
   precedes pointer acquisition; out-of-bounds indexing cannot return a guard.
3. The lock serializes conflict checks and record publication. Overlap
   diagnostics run after the metadata borrow and lock end, so caller panic
   hooks do not run inside the admission lock. An ID denotes
   exactly one live record; overflow rejects before its ID can collide with
   the empty/unchecked sentinels. A forgotten guard retains its record.
4. A guard holds a pointer, not a payload reference. Deref borrows cannot survive
   moving/dropping that guard. Casts retain the record. Only retirement permits
   a conflicting later guard.
5. Payload writes precede the retirement lock's Release unlock. Subsequent
   admission obtains Acquire ownership, publishing the writes to the next
   borrower. An unsuccessful swap never grants ownership. Fairness affects
   progress, never permission to access conflicting data.

Poisoning is conservative refusal. Soundness does not require unwinding,
poisoning, or every guard being dropped. Safe calls on unrelated instances
cannot reset the old instance's lock, records or guard representation.
Unsafe storage implementations and `dangerously_unchecked` retain their
explicit caller obligations. This is a conditional argument and executable
evidence, not a universal theorem about the decoder or all Rust executions.

## Executable gates

`scripts/review-disjoint-032.py` records exact commands, feature sets, model
flags, outcomes, durations, and log hashes. Run its groups serially through
the workspace's `run-heavy --mem 16G --jobs 8 --` wrapper on the review host:

```sh
python3 scripts/review-disjoint-032.py --group native --output "$HOME/tmp/rav1d-review-0.3.2"
python3 scripts/review-disjoint-032.py --group loom --output "$HOME/tmp/rav1d-review-0.3.2"
python3 scripts/review-disjoint-032.py --group miri --output "$HOME/tmp/rav1d-review-0.3.2"
python3 scripts/review-disjoint-032.py --group semver --output "$HOME/tmp/rav1d-review-0.3.2" \
  --baseline "$HOME/tmp/rav1d-review-2026-09-05/releases/rav1d-disjoint-mut-0.3.1"
python3 scripts/review-disjoint-032.py --group package --output "$HOME/tmp/rav1d-review-0.3.2"
```

The native matrix exercises all targets/all features, no-std, no-std with all
storage adapters, doctests, lints and documentation. Miri runs the complete
all-feature suite under Stacked and Tree Borrows, plus no-std API attacks.
The new interval oracle checks independently enumerated element sets; other
tests challenge static construction, cross-thread guard moves, all four casts,
slot exhaustion, overflow reuse, leaked guards, ZSTs, and reversed ranges.

The imported guard-move stress test initially failed its success floor on the
older tracker's retirement lock. It now pauses for one microsecond after an
expected overlap refusal and serializes the two process-global panic hooks.
It keeps the same seven worker threads, attempt counts, success floors, and
positive-refusal assertions. Unexpected panics propagate. These are scheduler
conditions for a useful test, not performance claims.

Three Loom models compile the actual 0.3 record algorithm with instrumented
metadata and payload cells. They cover inline and real-capacity overflow
exclusion/reuse/handoff, and shared-reader retirement versus a writer. Each
actor attempts twice; the default preemption bound is two, with no permutation
or time cap. `LOOM_MAX_PREEMPTIONS` raises the bound. Only native spin waiting
is abstracted by an Acquire/Release mutex; native lock implementation and
fairness remain outside that model. The alternative constructors required by
Loom are behind a non-feature test cfg; normal and Miri builds retain const
construction.

Retain the published-version UB controls and a deliberately broken admission
model alongside the passing candidate. Compare public API snapshots including
auto traits and const qualifiers, and check the feature table separately.

## Shipping this patch

Only publish `rav1d-disjoint-mut` from this maintenance revision. The historical
decoder in the surrounding workspace is not a new decoder release. Before
publication, verify the packaged source/hash, pass the declared platform CI,
review README/release notes, and create the corresponding crate release tag
and GitHub release. This local preparation does not publish or yank anything.

Once 0.3.2 is published, compatible `^0.3` requirements can resolve it. Existing
Cargo.lock files do not update automatically: consumers must update the
dependency. Exact `=0.3.1` pins must change. The dependency patch closes the
guard abstraction defect; it does not backport the decoder's separate global
threading-policy fixes or establish AV1-input reachability of the original UB.
