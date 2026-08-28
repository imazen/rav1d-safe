# Changelog

All notable changes to `rav1d-disjoint-mut` are documented in this file. Format follows [Keep a Changelog](https://keepachangelog.com/). Versions before `0.3.1` were not changelogged; see git history.

## [Unreleased]

### Fixed
- **`index_rect{,_mut}` registered the rectangle in bytes while every other
  borrow is registered in `T::Target` elements** (soundness, safe API). On a
  buffer whose element is wider than a byte — `DisjointMut<Vec<u16>>`, say —
  a rectangle over elements `{0,1,8,9}` was recorded as bytes `{0..4,
  16..20}`, so an `index_mut(8..10)` over the same elements was compared in the
  wrong coordinate system, found no overlap, and a second live `&mut` was
  handed out from safe code; the inter-row gap was a false positive for the
  same reason, and the in-bounds check divided the element count by
  `size_of::<V>()` and refused the upper part of the buffer. Not reachable
  from rav1d-safe (every call site is `index_rect{,_mut}_as` over a `u8`
  plane, where bytes are elements), reachable from the crate's public API. The
  rectangle path now scales by `size_of::<V>() / size_of::<T::Target>()` —
  asserted exact — and bounds against the element length;
  `declare_row_stride`'s unit is documented as `T::Target` elements.
  `tests/rect_units.rs` (8 tests: both miss orders, an immutable/mutable mix,
  a negative stride, the gap control writing through both guards, the bound,
  two byte-buffer controls) fails 6/8 on the previous revision and passes
  under Miri (Stacked and Tree Borrows). Found by the 2026-08-28 deductive
  review in `AUDIT.md`, which also carries the proofs for the sharded tracker,
  the exact rectangle records and the rect guards.

### Added
- **Exact strided-rectangle borrow records.** `BorrowTracker::add_rect_immut`
  registers `rows` segments of `seg` bytes, the instance's declared row stride
  apart, as ONE record whose footprint is exactly those segments — no inter-row
  gap is reserved, so it does not invent the false positive a hull extent does.
  `DisjointMut::index_rect` / `index_rect_as` return a
  `DisjointImmutRectGuard`, which has **no `Deref`**: `row(r)` derives that row's
  slice from the buffer's own pointer, so no reference wider than one row is ever
  materialised (the reverted March-2026 strided tracker had an exact record and a
  hull-wide reference, which is UB under both aliasing models).
  Storage is free — the record stores the hull in the two words a plain interval
  already used, and `(rows, seg)` is recovered from the hull and the stride by an
  exact bijection, so `Shard` remains exactly 128 bytes with no side table. The
  per-slot `mutable: u8` bitmap became `flags: u16` (high byte = rectangle),
  keeping `alloc`'s empty-shard arm at ONE store.
  `add_rect` DECLINES — never approximates — when there is no declared stride, the
  stride does not match, `seg > stride`, `rows > MAX_RECT_ROWS`, the hull spans
  more than `MAX_SHARDS_PER_BORROW` blocks, a shard is full, or a wide record is
  live; the caller then takes its own per-row path. Declining rather than
  promoting is why the wide list needs no rectangle support.
  `find`'s hot loop is unchanged: its hit is a PREFILTER and the caller passes it
  through `refine`, which is one load and one branch when the shard holds no
  rectangle record and otherwise defers to a cold exact rescan.
  12 tests against a brute-force byte-set oracle; Miri clean under Stacked
  Borrows on every non-timeout target. Consumer status and measurements:
  `docs/RECT_RECORDS.md` in the rav1d-safe repo.

### Fixed
- **Soundness: moving a guard by value was UB while another thread held the
  same region.** `DisjointMutGuard`/`DisjointImmutGuard` carried the borrowed
  region as `&'a mut V` / `&'a V`. Passing a guard by value into a call —
  `drop(g)` above all — gives that reference a *protector*, which requires it
  to stay valid for the whole call; but the guard's `Drop` runs inside that
  call and retires the tracker record, which is precisely what lets ANOTHER
  thread take the region and retag those bytes. The protected reference is
  invalidated mid-call. Reachable from safe code, on the release path the
  whole crate is built around. Both guards hold `NonNull<V>` now and
  materialise the reference in `Deref`/`DerefMut`, where borrowck bounds it to
  a region in which the guard cannot be dropped — the same reason
  `core::cell::RefMut` holds a `NonNull<T>`. `NonNull` rather than `*mut V`
  because a reference field carries `nonnull` into every load and a bare raw
  pointer does not: measured +1.1-1.2% at v4k_8tile t=1 in two independent
  interleaved runs, recovered to 1.0005 by `NonNull` plus `#[inline(always)]`
  on the three `Deref`/`DerefMut` bodies. The four `Send`/`Sync` impls
  restore, exactly, what the compiler derived from the reference fields
  (verified by compiling the same positive and negative bound probes against
  both revisions), so there is no auto-trait change. Found by
  `tests/narrow_release.rs` under Miri (CI run 31292996318); reproduces under
  **both** Stacked Borrows and Tree Borrows; gated by the new
  `tests/guard_move_release.rs`, which fails under both models against the old
  guards. No change to the tracker, and no bitstream change: 766/766
  dav1d-test-data vectors byte-identical at `--threads 1`, set-diffed by name
  with the MD5 in the value.

### Added
- **Overlap panics now report the EXISTING borrow's registration site.**
  `BorrowSlots` records `&'static Location` per active borrow, and
  `overlap_panic` prints it: `existing: &mut _[a..b] at src/file.rs:L:C`.
  With `debug_assertions` the callers propagate `#[track_caller]`, so the
  location names the true borrow site; in release it names the `DisjointMut`
  wrapper method. This is how zenavif#30's racing pair (rav1d-safe CDEF
  padding vs the loop filter's compact write-back) was identified — the old
  message named only the panicking side. One pointer store per borrow
  registration.

## [0.3.1] - 2026-05-26

### Fixed
- **Memory safety: `PicBuf::from_vec_aligned` arithmetic overflow** (`68ab197`). `align_offset + usable_len` was an unchecked add; with a non-zero `align_offset` and a `usable_len` near `usize::MAX` it could wrap, letting the bounds `assert!` pass while `usable_len > vec.len()` — exposing an out-of-bounds region (reachable on 32-bit targets with crafted picture dimensions). Now uses `checked_add` and panics on overflow. Regression tests added in `tests/pic_buf_overflow.rs`.

### Changed (technically breaking — see note)
- **Sealed the load-bearing index traits** `DisjointMutIndex`, `SliceBounds`, `TranslateRange` via a private `sealed::IndexLike` supertrait (`6fe6dc8`), closing a soundness hole: these traits are `unsafe`-adjacent (the `DisjointMut` core trusts impls to return in-bounds pointers matching their registered `Bounds`, mirroring `std::slice::SliceIndex`), so an external impl could only be unsound. `cargo-semver-checks` flags trait-sealing as a major change, but the practical break surface is empty — the only code it removes was already unsound. Shipped as a patch deliberately so all `^0.3` dependents receive the soundness + overflow fixes automatically.

### Notes
- `[0.3.0]` (2026-02-14) predates these fixes; `^0.3` users should upgrade to `0.3.1`.
