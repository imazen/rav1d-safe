# DisjointMut Soundness Audit

**Date:** 2026-02-08
**Auditor:** Red-team review with Miri verification (Stacked Borrows + Tree Borrows)
**Verdict:** Three Stacked Borrows UB bugs found and fixed. All other identified issues resolved.

## Bugs Found and Fixed

### CRITICAL: guards held a reference across their own release (FIXED 2026-08-09, #477)

**Found after this audit, by CI, not by it.** `DisjointMutGuard` held
`slice: &'a mut V` and `DisjointImmutGuard` held `slice: &'a V`. Moving a guard
by value into a call — `drop(g)`, `ManuallyDrop::new(g)`, any `f(g)` — protects
that reference for the duration of the call; the guard's `Drop` runs inside that
call and retires the tracker record, which is what lets **another thread** take
the region and retag those bytes. The protected reference is invalidated
mid-call. Reachable from safe code.

**Fix:** both guards hold `*mut V` / `*const V` and materialise the reference in
`Deref`/`DerefMut`. `core::cell::RefMut` holds a `NonNull<T>` for the same
reason. Four `unsafe impl`s restore the auto-traits the reference fields used to
derive (verified against both revisions with identical positive and negative
bound probes).

**Why this audit missed it.** Everything above is about a retag colliding with a
*concurrent* retag; this one needs a retag to collide with a *protector*, which
only exists while a guard is IN a call. Nothing in the suite moved a guard by
value under contention until `tests/narrow_release.rs` (2026-08-08) did — and it
found it on its first Miri run. Reproduces under both Stacked Borrows and Tree
Borrows. Gated by `tests/guard_move_release.rs`, which includes scope-exit
control arms so a future failure that is not about the move is distinguishable.

**Standing lesson:** a Miri arm that only exercises `{ let _g = dm.index_mut(r);
… }` cannot see this class at all. Concurrency tests must move guards.

### CRITICAL: Stacked Borrows UB in `Vec<V>::as_mut_ptr` (FIXED)

`(*ptr).as_mut_ptr()` auto-refs to `&mut Vec<V>`, creating a `Unique` retag on the Vec struct allocation. When two threads call `index_mut` for disjoint ranges concurrently, thread A's `Unique` retag conflicts with thread B's `SharedReadOnly` retag from `(*ptr).len()`.

**Fix:** Changed to `(*ptr).as_ptr().cast_mut()` which only creates `&Vec<V>` (SharedReadOnly). The returned pointer retains write provenance from the original allocator.

### CRITICAL: Stacked Borrows UB in `Box<[V]>` default `as_mut_slice` (FIXED)

The default `as_mut_slice` called `(*ptr).len()` which for `Box<[V]>` traverses `Deref::deref` → `&[V]`. This `SharedReadOnly` retag on the heap conflicts with `&mut [V]` guards on other threads.

**Fix:** Overrode `as_mut_slice` for `Box<[V]>` to use `addr_of_mut!(**ptr)` (raw pointer chain, no intermediate references). Changed `DisjointMut::len()` to use `as_mut_slice().len()` (fat pointer metadata).

### CRITICAL: Stacked Borrows UB in `[V; N]` default `as_mut_slice` (FIXED)

The default `as_mut_slice` called `(*ptr).len()` which auto-refs to `&[V; N]`. For arrays, the data IS the allocation (inline in UnsafeCell), so `&[V; N]` creates a `SharedReadOnly` retag covering the same memory as element guards.

**Fix:** Overrode `as_mut_slice` for `[V; N]` to use `ptr::slice_from_raw_parts_mut(ptr.cast::<V>(), N)` — pure pointer math, no references, compile-time length. Also overrode for `[V]` with identity passthrough.

### HIGH: `unchecked` feature made `new()` unsound (FIXED)

`new()` skipped tracking when the `unchecked` feature was enabled. Since cargo features are additive/unioned across the dependency tree, any crate enabling `unchecked` silently disabled all safety checks everywhere.

**Fix:** `new()` always creates a tracked instance. `dangerously_unchecked()` is gated behind `#[cfg(feature = "unchecked")]`.

### MEDIUM: Empty ranges treated as overlapping (FIXED)

`Bounds::overlaps` reported `50..50` as overlapping with `0..100`. Empty ranges borrow zero bytes and should never conflict.

**Fix:** Added `is_empty()` check — empty ranges never overlap.

### MEDIUM: No panic safety (FIXED — poisoning)

If `get_mut` panicked after borrow registration (e.g. OOB), or if user code panicked while holding a mutable guard, the data structure had no way to signal that it may be in an inconsistent state. Future borrows would succeed, potentially exposing partially-written data.

**Fix:** Added `std::sync::Mutex`-style poisoning. An `AtomicBool` flag on `BorrowTracker` is set when:
1. A panic occurs between borrow registration and reference creation (`BorrowCleanup` scope guard)
2. A mutable guard is dropped during panic unwinding (`DisjointMutGuard::drop` checks `thread::panicking()`)

All future `index()` and `index_mut()` calls check the poison flag and panic with a clear message. Immutable guard panics do NOT poison (read-only access can't corrupt data).

### MEDIUM: Integer overflow in `Bounds` conversions (FIXED)

`From<usize>` computed `index + 1`, `RangeInclusive`/`RangeToInclusive` computed `end + 1`. All overflow at `usize::MAX`.

**Fix:** Changed to `checked_add().expect()` — panics with a clear message instead of silent wraparound.

### LOW: `ExternalAsMutPtr` safety docs incomplete (FIXED)

Docs didn't warn about intermediate `&mut` references causing SB retagging conflicts.

**Fix:** Expanded safety docs with 4 explicit requirements including the `&mut` prohibition.

## 2026-08-28 review — sharded tracker, exact strided-rectangle records, rect guards

**Scope.** Every public item `docs/public-api/rav1d-disjoint-mut.txt` gained
between 6f84a31 (2026-08-07) and 00ad667 (2026-08-27) — `index_rect{,_mut}`,
`index_rect{,_mut}_as`, `DisjointImmutRectGuard`, `DisjointMutRectGuard`,
`declare_row_stride`, `set_parallelism`, `set_tile_concurrency`, the
`probe_*` hooks — and the tracker changes behind them (`tracker_shard.rs`:
rectangle records 3353944 / d4e9222, the one-shard fast path 1a2c165 /
d1d5408, `NonNull` guard fields 825df67 / e0187a3, the lock-free release, the
derived block shift 318a4bc, `SHARDS_SERIAL = 1` #458). Read on `main` @
00ad667. Method: state the invariant, prove each mechanism keeps it, and turn
every gap into a test that fails on the code as found.

**Verdict.** ONE soundness defect, fixed in this revision (Lemma U):
`index_rect{,_mut}` registered the rectangle in BYTES while every other borrow
of a buffer is registered in `T::Target` elements, so on a buffer whose element
is wider than a byte a rectangle and a range over the same elements were
compared in different coordinate systems and a real overlap went unreported —
two live `&mut`, from safe code. Not reachable from rav1d-safe: its four call
sites use `index_rect{,_mut}_as` over `u8` planes, where the two units
coincide. Reachable from the crate's safe API. Everything else below is proved
sound under the stated invariants.

### Definitions

* An instance `D` wraps `L` elements of `T::Target`. **Coordinates** are
  element indices in `[0, L)`; Lemma U is what makes this the unit of every
  registration path.
* A **borrow** is `(F, m)`: a footprint `F ⊆ [0, L)` and a mutability bit.
  Two live borrows **conflict** iff `F₁ ∩ F₂ ≠ ∅` and `m₁ ∨ m₂`.
* **Records.** Interval `[a, b)`. Rectangle `(h0, rows, seg)` on the instance
  stride `s = row_stride`: `F = ⋃_{r<rows} [h0 + r·s, h0 + r·s + seg)` with
  `1 ≤ seg ≤ s`, `rows ≤ MAX_RECT_ROWS`, stored as its hull
  `[h0, h0 + (rows−1)·s + seg)` plus one rect bit. Wide `[a, b)` in the wide
  list.
* **Property S (no missed conflict).** At every instant no two live
  registrations conflict; equivalently, a registration whose footprint
  conflicts with a live one panics before it publishes, and no reference is
  created before registration (`index_mut`: register, then `get_mut`;
  `BorrowCleanup` poisons if `get_mut` panics).
* **Property R (reference ⊆ registration).** Every reference a guard hands out
  lies inside the guard's registered footprint and inside the allocation, and
  the registration is retired only in `Drop`.

### Lemma 1 — block cover

Let a registration have footprint `F` and hull `H = [a, b)`, `F ⊆ H`, and let
`Σ(H) = { shard(blk) : blk ∈ [a >> k, (b−1) >> k] }` for the instance's shift
`k`. If two footprints share an element `x`, then
`a₁ >> k ≤ x >> k ≤ (b₁−1) >> k` and likewise for the second, so
`σ = shard(x >> k) ∈ Σ(H₁) ∩ Σ(H₂)`. `add`, `add_multi` and `add_rect`
register in every shard of `Σ(H)` (a rectangle's hull blocks are a superset of
its row blocks, so this is conservative in the sound direction); `add_wide`
registers in the wide list instead, which Lemma 5 covers. The argument needs
only that `k`, `shard_of` and `mask` are fixed while any record is live: they
move only in `reprovision` and `set_row_stride`, both `&mut self`, and every
guard holds `&'a DisjointMut`, so no guard can be live then. On a `mask == 0`
instance `Σ = {0}` for every borrow, which is why the one-shard fast path may
skip the block arithmetic (#458) and register a multi-block span as one
record. ∎

### Lemma 2 — serialisation inside a shard

In every registration path (`add`, `add_contended`, `add_slow`, `add_multi`,
`add_rect`, `add_wide`) the registrant holds shard `σ`'s lock from before its
scan until after `publish` (`live[slot].store(1, Release)`); the unlock comes
after. So of two registrants both in `σ`, the second acquires `σ` after the
first released it, and its `live_mask` (`load(Acquire)`) observes the first's
slot. `allocated` is a superset of the live bits: a bit is set under the lock
before `publish`, and the word is only ever narrowed under the lock against
the flags. Hence `live_mask(allocated)` never omits a live slot, and the
`allocated ≤ 1` straight-line case is exact because then no slot other than 0
can be live. With Lemma 1, whichever of a conflicting pair registers second
scans a shard containing the first's record. ∎

### Lemma 3 — the overlap tests are exact

(a) For half-open non-empty intervals, `s_i < end ∧ start < e_i` ⟺ they
intersect.

(b) `rect_decode` inverts the hull encoding. `span = (rows−1)·s + seg` with
`1 ≤ seg ≤ s` gives `span − 1 = (rows−1)·s + (seg−1)` with
`0 ≤ seg−1 < s`, so `⌊(span−1)/s⌋ = rows−1` and `seg` follows. Enforced at
registration: `add_rect` declines `seg == 0`, `seg > s`, `rows == 0`,
`rows > MAX_RECT_ROWS`, `s == 0` or `s ≠ stride`.

(c) `rect_hit_range(h0, h1, s, a, b)` returns `Some` iff
`F_rect ∩ [a, b) ≠ ∅`. Clip to `[lo, hi) = [max(a,h0), min(b,h1))`; because
`F_rect ⊆ [h0, h1)` the intersection with `[a, b)` equals the intersection
with `[lo, hi)`, empty if `lo ≥ hi`. Let `r₀ = ⌊(lo−h0)/s⌋`; `lo < h1 ≤ h0 +
rows·s` gives `r₀ ≤ rows−1`. Every row `r < r₀` ends at
`h0 + r·s + seg ≤ h0 + r₀·s ≤ lo` and misses. Row `r₀` starts at `rs ≤ lo`
and meets `[lo, hi)` iff `lo < rs + seg`. Every row `r > r₀` starts at
`rs > lo` and meets iff `rs < hi`, and rows start monotonically, so the first
row with `rs ≥ hi` ends the search. The loop tests exactly these conditions in
order. ∎

(d) `find` is a hull prefilter and `refine` makes it exact: when the shard
holds no rectangle the hull is the footprint; otherwise `find_exact` rescans
every live record, with (a) for intervals and (c) for rectangles — a rescan,
not a continuation, because the first hull hit may be a rectangle that does
not really overlap while a later record does.

(e) `find_from_rect`: against a stored interval it applies (c) with the probe
rectangle and the interval; against a stored rectangle it applies (c) to each
probe row segment `[rs, rs+seg)` and the stored rectangle, both on the one
instance stride. `F_probe ∩ F_stored ≠ ∅` iff some probe row meets
`F_stored`. ∎

(f) Mutability: a mutable registrant scans every live record, an immutable one
only the mutable records — exactly the conflict relation.

### Lemma 4 — publication and release ordering

A record becomes visible only through `publish`, after its fields are written
under the lock; a scanner reads the live flag with `Acquire` under the same
lock and therefore sees complete fields. Release is `live[i].store(0,
Release)` by the guard's owner, after every access through the reference
(`Deref`/`DerefMut` materialise the reference from the `NonNull` field for a
region borrowck ends before `Drop`). The next allocator observes the zero with
`Acquire` under the lock, so the previous owner's accesses happen-before the
next owner's. Only those two parties ever write `live[i]`, never concurrently
(the allocator only after observing zero under the lock). `remove_multi`
retires under all its locks so a registrant holding several of those shards
sees the set retire as one step; a partially retired record could only cause
a spurious panic, never a miss — a stale live record is conservative. ∎

### Lemma 5 — wide records

`add_wide` writes the list and increments `state` while holding EVERY active
shard lock; every narrow registrant of the instance locks inside `active()`
(`Σ ⊆ [0, mask]`, Lemma 1). Each narrow path re-reads `state` under at least
one shard lock — `add` (the in-lock re-read that closed the 4af62ae TOCTOU),
`add_contended`, `add_slow` (which consults the list unconditionally),
`add_multi` and `add_rect` under their locks — so the wide write
happens-before that read. A registrant that sees `state ≠ 0` scans the wide
list with (a), or, for a rectangle, DECLINES and its caller's per-row path
scans it. Conversely `add_wide` scans every active shard with `find` +
`refine` (exact, Lemma 3d) plus the list. ∎

### Lemma 6 — the rect guards (Property R)

`rect_geometry` refuses `seg == 0`, `rows == 0`, `|stride| < seg` (so the
rows are pairwise disjoint), any arithmetic overflow, a base misaligned for
`V`, and `end_asc > len_V` where `len_V = L · size_of::<Target>() /
size_of::<V>()` — the bound is derived from the element count, not the count
divided by `size_of::<V>()` (the second half of Lemma U's defect: harmless,
it only over-refused). Row `r` is `[base + r·stride, +seg)` in `V` units; for
`stride ≥ 0` the rows ascend from `lo_asc = lo`, for `stride < 0` they descend
from row 0 to `lo_asc = lo − (rows−1)·|stride| ≥ 0` (checked_sub), and every
row lies in `[lo_asc, end_asc) ⊆ [0, len_V)`. Each row is exactly one segment
of the registered footprint (Lemma U for the unit). `row_mut` takes `&mut
self`, so at most one `&mut [V]` is live at a time; `row` returns `&[V]` tied
to `&self`; neither guard implements `Deref`, so no reference wider than one
row exists (the March-2026 defect). The base is `NonNull`, so moving the
guard into `drop` carries no protector (#477). `V` validity: `index_rect`
uses `T::Target` (every bit pattern valid), `_as` requires `FromBytes`
(`IntoBytes` too for mutation). ∎

`set_parallelism`, `set_tile_concurrency`, `declare_row_stride` and the block
shift they feed are locality knobs: Lemma 1 needs only per-instance
constancy, which `&mut self` (or read-once-at-`new`) provides. The `probe_*`
hooks are `__probe_bounds`-only and compile to nothing otherwise.

### Lemma U — units, and the defect

Lemmas 1–3 assume every record of an instance is in one coordinate system.
`index`/`index_mut` register a `Bounds` in `T::Target` elements (clamped to
`as_mut_slice().len()`, an element count); `slice_as`/`mut_slice_as` register
`range.mul(size_of::<V>())` on a `u8` buffer, where bytes ARE elements.
`index_rect_inner` registered `lo_asc · size_of::<V>()`, `seg ·
size_of::<V>()`, `stride · size_of::<V>()`: for the `_as` entry points on a
`u8` buffer that is the same unit; for `index_rect{,_mut}` on, say,
`DisjointMut<Vec<u16>>` it is twice the element coordinate.

Counterexample (`tests/rect_units.rs`, fails on 00ad667): `declare_row_stride
(8)` (the tracker compares its stride against the caller's, so the old code
needs `16` to accept — in either form the point stands),
`index_rect_mut(0, seg 2, rows 2, stride 8)` = elements `{0,1} ∪ {8,9}` was
recorded as the byte hull `[0, 20)` with `(rows 2, seg 4, stride 16)` =
footprint `{0..4} ∪ {16..20}`; `index_mut(8..10)` registered `[8, 10)` in
elements; the scan found no overlap; both `&mut` were live over elements
8 and 9. On the same input the inter-row gap `index_mut(2..8)` was a FALSE
positive against the byte-scaled first row `[0, 4)`.

Fix: the rectangle path scales by `size_of::<V>() / size_of::<T::Target>()`
(1 for `index_rect{,_mut}`, `size_of::<V>()` for `_as` over `u8` — exact, and
asserted so no future entry point can break it), and the bound uses the
element length. `declare_row_stride`'s unit is documented as `T::Target`
elements (bytes for the byte planes it exists for). Eight tests: the two miss
cases (either order), the immutable-rectangle/mutable-range mix, the negative
stride, the gap control that writes through both guards, the upper-half
bound, and two byte-buffer controls (`index_rect_mut_as::<u16>` against
`mut_slice_as::<u16>`) that pass on both revisions — the decoder's path,
unchanged by construction. On 00ad667: 6 of 8 fail (2 should-panic tests do
not panic, the gap control panics, the bound refuses, the negative stride
misses); on this revision 8 of 8 pass, under `cargo test` and under Miri with both Stacked Borrows and Tree Borrows (`cargo +nightly miri test --features zerocopy,aligned --test rect_units`, `MIRIFLAGS=-Zmiri-tree-borrows` likewise) — the two byte-buffer controls use `align::AlignedVec64<u8>` because a plain `Vec<u8>` is 1-byte aligned under Miri and the `u16` view is then, correctly, refused.

### What this review does not establish

* `dangerously_unchecked` instances have no tracker; S is the caller's
  obligation under that `unsafe` contract. The rect path returns
  `BorrowId::UNCHECKED` there and creates the same references.
* The `__probe_tinynop` / `__probe_addnop` / `__probe_untracked` arms are
  unsound by design (measurement only), `__`-gated and unpublished; the
  legacy tracker (`__tracker_legacy`) was not re-reviewed.
* Spurious panics (false positives) are a liveness matter, outside S; the
  tracker's own `rect_*` tests cover the routine gap cases.
* A caller that receives `None` from `index_rect*` must take a per-row path.
  In rav1d-safe that is not a discipline but a type-system fact: with
  `#![forbid(unsafe_code)]` the decoder cannot touch a byte without a guard,
  and every guard registers. All four call sites (`picture.rs`
  `for_rows{,_mut}`, `loopfilter.rs` `fill_rect`) do fall back.

## Architecture Assessment

### What's Sound

1. **Core overlap tracking** — `BorrowTracker` uses `parking_lot::Mutex` to serialize registration. Registration before reference creation prevents TOCTOU. Poisoning on panic prevents access to potentially corrupted data.

2. **Guard lifecycle** — RAII-based. Drop deregisters. `ManuallyDrop` in `cast_slice`/`cast` correctly transfers borrow ownership. **Amended 2026-08-09:** the guard must hold the region as a POINTER, never as a reference — see the first entry under "Bugs Found and Fixed". Reverting that is UB, not a style choice.

3. **Sealed trait** — `AsMutPtr` sealed via private supertrait. External types go through `unsafe ExternalAsMutPtr`. `Copy` bound on `Target` prevents torn reads.

4. **Send/Sync bounds** — Correct: `T: Send` for `Send`, `T: Sync` for `Sync`. Tracker uses `Mutex`. **Amended 2026-08-09:** the guards' bounds are now explicit `unsafe impl`s rather than derived, because their region field is a raw pointer. They reproduce the derived bounds exactly; if you touch them, re-run the positive/negative probe pair against the previous revision.

5. **All AsMutPtr impls override `as_mut_slice`** — The default impl (which creates `&T`) is never used. Every concrete type uses reference-free pointer operations.

### What's Subtle But Correct

1. **`Vec::as_ptr().cast_mut()` provenance** — The pointer value stored in Vec retains allocator provenance, not the `&Vec` reference's provenance. Miri confirms under both SB and TB.

2. **`addr_of_mut!(**ptr)` for Box** — Raw pointer chain through Box's compiler-intrinsic deref creates no intermediate references. Miri confirms.

3. **`parking_lot::Mutex` unwind safety** — `parking_lot::Mutex` doesn't poison (unlike `std::sync::Mutex`), but we add our own `AtomicBool` poisoning at the `BorrowTracker` level. Lock always released on unwind. Mutable guard Drop poisons if `thread::panicking()`.

## Remaining Work for crates.io

### Should Do

- [ ] Add README.md with usage examples, safety model, Miri instructions
- [ ] Upgrade to zerocopy 0.8
- [ ] Consider `no_std` support (`parking_lot` → spin lock, or `std`-only tracking)
- [ ] Add `Debug` impl for `DisjointMut`
- [ ] Upgrade to edition 2024
- [ ] Add CI (GitHub Actions with `cargo test`, Miri under SB and TB)

### Nice to Have

- [ ] Property-based tests (proptest) for overlap detection
- [ ] Benchmark tracker overhead
- [ ] `DisjointMut::try_index_mut` returning `Result`
- [ ] Loom tests for concurrent correctness
