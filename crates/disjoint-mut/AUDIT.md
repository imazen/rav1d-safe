# DisjointMut Soundness Audit

**Date:** 2026-02-08
**Auditor:** Red-team review with Miri verification (Stacked Borrows + Tree Borrows)
**Verdict:** Three Stacked Borrows UB bugs found and fixed. All other identified issues resolved.

## Bugs Found and Fixed

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

## Architecture Assessment

### What's Sound

1. **Core overlap tracking** — `BorrowTracker` uses `parking_lot::Mutex` to serialize registration. Registration before reference creation prevents TOCTOU. Poisoning on panic prevents access to potentially corrupted data.

2. **Guard lifecycle** — RAII-based. Drop deregisters. `ManuallyDrop` in `cast_slice`/`cast` correctly transfers borrow ownership.

3. **Sealed trait** — `AsMutPtr` sealed via private supertrait. External types go through `unsafe ExternalAsMutPtr`. `Copy` bound on `Target` prevents torn reads.

4. **Send/Sync bounds** — Correct: `T: Send` for `Send`, `T: Sync` for `Sync`. Tracker uses `Mutex`.

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
