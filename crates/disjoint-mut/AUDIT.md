# DisjointMut Soundness Audit

**Date:** 2026-02-08
**Auditor:** Red-team review with Miri verification (Stacked Borrows + Tree Borrows)
**Verdict:** Two real UB bugs found and fixed. Several design issues remain.

## Bugs Found and Fixed

### CRITICAL: Stacked Borrows UB in `Vec<V>::as_mut_ptr` (FIXED)

`(*ptr).as_mut_ptr()` auto-refs to `&mut Vec<V>`, creating a `Unique` retag on the Vec struct allocation. When two threads call `index_mut` for disjoint ranges concurrently, thread A's `Unique` retag on the Vec struct conflicts with thread B's `SharedReadOnly` retag from `(*ptr).len()`.

**Fix:** Changed to `(*ptr).as_ptr().cast_mut()` which only creates `&Vec<V>` (SharedReadOnly). The returned pointer retains write provenance from the original allocator, not from the shared reference.

### CRITICAL: Stacked Borrows UB in `Box<[V]>` default `as_mut_slice` (FIXED)

The default `as_mut_slice` called `(*ptr).len()` which for `Box<[V]>` traverses `&Box<[V]>` → `Deref::deref` → `&[V]`. This creates a `SharedReadOnly` retag on the **heap allocation**, conflicting with `&mut [V]` guards held by other threads.

**Fix:** Overrode `as_mut_slice` for `Box<[V]>` to use `addr_of_mut!(**ptr)` — a raw pointer chain through Box's compiler-intrinsic deref that creates no intermediate references. Also changed `DisjointMut::len()` to read from `as_mut_slice().len()` (fat pointer metadata) instead of `AsMutPtr::len(&self)`.

## Remaining Issues (Not UB, but worth fixing for crates.io)

### HIGH: `unchecked` feature flag is unsound by design

When the `unchecked` cargo feature is enabled, `DisjointMut::new()` — a **safe** function — produces an unchecked instance (tracker=None). This silently removes all runtime overlap checking through safe code, making any overlapping access UB.

Worse: cargo features are **additive and unioned across the dependency tree**. If ANY crate in your dependency tree enables `unchecked`, ALL DisjointMut usage becomes unchecked.

**Recommendation:** Remove the `unchecked` feature entirely. The only way to skip tracking should be the `unsafe fn dangerously_unchecked()` constructor, which properly requires the caller to accept responsibility.

### MEDIUM: Empty ranges treated as overlapping (false positive)

`Bounds::overlaps` reports `50..50` as overlapping with `0..100`. An empty range borrows zero bytes and should never conflict with anything.

```rust
// Current:
fn overlaps(&self, other: &Bounds) -> bool {
    a.start < b.end && b.start < a.end  // 50 < 100 && 0 < 50 → true (wrong)
}

// Fix:
fn overlaps(&self, other: &Bounds) -> bool {
    a.start < a.end && b.start < b.end  // both non-empty
    && a.start < b.end && b.start < a.end  // and actually overlap
}
```

This is a correctness bug (false panic), not a soundness bug.

### MEDIUM: Borrow leak on OOB panic

If `DisjointMutIndex::get_mut` panics (index out of bounds) after the borrow is registered in the tracker but before the guard is constructed, the borrow record is never removed. All future overlapping borrows will panic forever.

This is the same behavior as `RefCell::borrow_mut` panicking — conservative but prevents UB. Document it.

### MEDIUM: Integer overflow in `Bounds` conversions

`From<usize> for Bounds` computes `index..index + 1`. If `index == usize::MAX`, this overflows to `usize::MAX..0` in release mode. Similarly, `RangeInclusive::to_range` and `RangeToInclusive::to_range` overflow when `end == usize::MAX`.

Practically impossible (no real slice has `usize::MAX` elements), but a pedantic reviewer will flag it. Use `checked_add` or `saturating_add`, or document the limitation.

### LOW: `ExternalAsMutPtr` safety contract is under-documented

The `unsafe trait` docs list 3 requirements but miss a critical one: the `as_mut_ptr` implementation must not create `&mut Self` or any intermediate mutable reference that would cause Stacked Borrows retagging conflicts. This is the exact bug we found in the built-in Vec impl.

### LOW: `inner()` returns `*mut T` bypassing tracker

The `inner()` method returns `*mut T` (the container, not the elements). While raw pointers are safe to create, this lets users construct `&mut T` from the pointer, potentially invalidating outstanding guards. Document that dereferencing as `&mut` is UB while guards exist.

## Architecture Assessment

### What's Sound

1. **Core overlap tracking** — The `BorrowTracker` uses a `parking_lot::Mutex` to serialize all borrow registration. Registration happens before reference creation, and the lock ensures atomic check-and-insert. No TOCTOU gap between validation and reference creation.

2. **Guard lifecycle** — Guards are RAII-based. Drop deregisters the borrow. `ManuallyDrop` in `cast_slice`/`cast` correctly transfers borrow ownership without double-deregistration.

3. **Sealed trait** — `AsMutPtr` is sealed via a private `Sealed` supertrait. External types must go through `ExternalAsMutPtr` (which is `unsafe`). The `Copy` bound on `Target` is correct — prevents data races from producing invalid values (torn reads on non-Copy types would be UB).

4. **Send/Sync bounds** — `DisjointMut<T>: Send` when `T: Send`, `DisjointMut<T>: Sync` when `T: Sync`. These are correct. The tracker's `Mutex` handles cross-thread synchronization.

5. **Array AsMutPtr** — `[V; N]::as_mut_ptr` returns `ptr.cast()` which is a pure pointer cast on inline data. No reference creation, no SB issues.

### What's Subtle But Correct

1. **`Vec::as_ptr().cast_mut()` provenance** — The `*const V` returned by `Vec::as_ptr(&self)` has provenance from the original allocator, not from the `&Vec` shared reference. Casting to `*mut V` preserves this provenance. Writing through it is valid because the UnsafeCell provides write permission. Miri confirms this under both SB and TB.

2. **`addr_of_mut!(**ptr)` for Box** — When `ptr: *mut Box<[V]>`, the chain `*ptr` (raw deref) → `*Box` (compiler-intrinsic Box deref) → `addr_of_mut!` produces `*mut [V]` without creating `&[V]` or `&mut [V]`. This relies on Box deref being a compiler built-in, not a `Deref` trait call, when accessed through a raw-pointer-derived place.

3. **`parking_lot::Mutex` panic behavior** — Unlike `std::sync::Mutex`, parking_lot mutexes don't poison on panic. The lock is always released on unwind. Borrow records may leak (see "borrow leak on OOB panic" above) but no deadlock occurs.

## Recommendations for crates.io Publication

### Must Fix

- [ ] Remove `unchecked` cargo feature (replace with `dangerously_unchecked()` constructor only)
- [ ] Fix empty range false positive in `Bounds::overlaps`
- [ ] Add `#![forbid(unsafe_code)]` to tests (currently tests don't use unsafe, enforce it)
- [ ] Add overflow protection to `From<usize> for Bounds` and inclusive range conversions
- [ ] Update `ExternalAsMutPtr` safety docs to warn about intermediate mutable references

### Should Fix

- [ ] Add README.md with usage examples, safety model explanation, Miri instructions
- [ ] Add CHANGELOG.md
- [ ] Upgrade to zerocopy 0.8 (0.7 is EOL)
- [ ] Consider `no_std` support (`parking_lot` → spin lock, or make tracking `std`-only)
- [ ] Add `Debug` impl for `DisjointMut` (currently missing)
- [ ] Add `Clone` impl for `DisjointMut<T>` where `T: Clone`
- [ ] Upgrade to edition 2024
- [ ] Add CI (GitHub Actions with `cargo test`, `cargo miri test` under both SB and TB)
- [ ] Document the borrow-leak-on-panic behavior
- [ ] Consider making `AsMutPtr::len` take `*mut Self` instead of `&self` for consistency

### Nice to Have

- [ ] Property-based tests (proptest/quickcheck) for overlap detection
- [ ] Benchmark tracker overhead vs unchecked
- [ ] `DisjointMut::try_index_mut` that returns `Result` instead of panicking
- [ ] Loom tests for concurrent correctness
