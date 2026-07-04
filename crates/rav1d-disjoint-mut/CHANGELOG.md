# Changelog

All notable changes to `rav1d-disjoint-mut` are documented in this file. Format follows [Keep a Changelog](https://keepachangelog.com/). Versions before `0.3.1` were not changelogged; see git history.

## [Unreleased]

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
