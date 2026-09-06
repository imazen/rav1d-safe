# Changelog

All notable changes to `rav1d-disjoint-mut` are documented in this file. Format follows [Keep a Changelog](https://keepachangelog.com/). Versions before `0.3.1` were not changelogged; see git history.

## [0.3.2] - Unreleased

### Fixed
- Moving either kind of borrow guard into `drop` or another function could
  protect its stored reference beyond retirement of the borrow record. Another
  thread could then acquire the same region while that reference was still
  protected. Both guards now store `NonNull` and derive references only while
  borrowing the guard. The four zerocopy cast paths transfer the reservation
  without retaining reference fields.
- Fix `aligned` without `std`: the three allocation-error return paths use
  `alloc::collections::TryReserveError`, the same type re-exported by `std`.

### Compatibility
- Retains `pub const fn DisjointMut::new`, all published feature names and
  defaults, Rust 1.85/no-std support, and existing guard `Send`/`Sync` bounds.
- Retains the original per-instance 64-slot tracker with overflow storage.
  The development sharded tracker, rectangle APIs, and experimental features
  are not backported. No normal-build dependency is added.
- Clarifies the existing unsafe storage-adapter obligations: stable valid
  storage, exclusive authority over aliases, and correct thread traits.

### Validation
- Const/static, interval-oracle, slot exhaustion/leak/reuse, cast and guard-move
  regressions; lifetime and thread-bound compile-fail examples; both Miri models.
- Three Loom models of the 0.3 record algorithm with instrumented metadata and
  payload cells. Native spin waiting is abstracted by an Acquire/Release mutex.
- Patch-level API compatibility gates against the actual 0.3.1 crate.

## [0.3.1] - 2026-05-26

### Fixed
- **Memory safety: `PicBuf::from_vec_aligned` arithmetic overflow** (`68ab197`). `align_offset + usable_len` was an unchecked add; with a non-zero `align_offset` and a `usable_len` near `usize::MAX` it could wrap, letting the bounds `assert!` pass while `usable_len > vec.len()` — exposing an out-of-bounds region (reachable on 32-bit targets with crafted picture dimensions). Now uses `checked_add` and panics on overflow. Regression tests added in `tests/pic_buf_overflow.rs`.

### Changed (technically breaking — see note)
- **Sealed the load-bearing index traits** `DisjointMutIndex`, `SliceBounds`, `TranslateRange` via a private `sealed::IndexLike` supertrait (`6fe6dc8`), closing a soundness hole: these traits are `unsafe`-adjacent (the `DisjointMut` core trusts impls to return in-bounds pointers matching their registered `Bounds`, mirroring `std::slice::SliceIndex`), so an incorrect external impl could violate memory safety. `cargo-semver-checks` flags trait-sealing as a major change, and external implementations no longer compile, including implementations that may have been correct. Shipped as a patch deliberately so all `^0.3` dependents receive the soundness + overflow fixes automatically.

### Notes
- `[0.3.0]` (2026-02-14) predates these fixes; `^0.3` users should upgrade to `0.3.1`.
