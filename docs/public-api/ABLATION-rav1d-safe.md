# rav1d-safe Public-API Ablation Report

**Date:** 2026-06-11
**Snapshot commit:** 0346a92a (feat: versioned public-API surface snapshots)
**Snapshot:** `docs/public-api/rav1d-safe.txt`
**Mode:** COMMIT (report only, no source changes)
**Grep template:** `grep -r 'rav1d_safe::<SYMBOL>' /home/lilith/work/ --include='*.rs' --exclude-dir=rav1d-safe --exclude-dir=target --exclude-dir=.jj`
**Scan as of:** 2026-06-11 (checked zenavif, heic, aom-decoder-rs, rav1d-bench, zen-arm-src, pre-filter)

---

## Summary

| Metric | Value |
|--------|-------|
| Default-feature items | 2,666 |
| All-feature items | 2,875 |
| Known external consumer crates | zenavif, heic, rav1d-bench |
| Flagged class A (doc_hidden/deprecated) | 5 items / groups |
| Flagged class B (pub→pub(crate) candidates) | 3 module groups |
| Flagged as percentage of total | ~3–5 % (default surface) |

**Conservative stance:** The `include::dav1d` namespace is 2,018 of 2,666 default items (~76 %). It mirrors the C dav1d header layout and serves as the "C-ABI parity" contract for external integrators who bypass the managed API (the `zenavif/decoder.rs` unsafe-asm path, zen-arm-src, pre-filter). These items are KEPT wholesale per the mission brief.

---

## Module Breakdown (default features, pub lines)

| Module | Pub-line count | External consumers | Verdict |
|--------|---------------|-------------------|---------|
| `include::dav1d::headers` | 1,866 | zenavif (unsafe-asm only), stale forks | KEEP — C-ABI parity |
| `src::managed` | 533 | zenavif, heic, rav1d-bench | KEEP — primary consumer API |
| `include::dav1d::picture` | 95 | zenavif (unsafe-asm only), stale forks | KEEP — C-ABI parity |
| `include::dav1d::dav1d` | 50 | zenavif (unsafe-asm only) | KEEP — C-ABI parity |
| `src::send_sync_non_null` | 15 | zenavif (unsafe-asm only), stale forks | See flag below |
| `include::dav1d::common` | 15 | zenavif (unsafe-asm only) | KEEP — C-ABI parity |
| `include::dav1d::data` | 7 | zenavif (unsafe-asm only) | KEEP — C-ABI parity |
| `src::dav1d_api` (all-feat) | 19 | C-ABI binary consumers | KEEP — deliberate compat contract |
| Root re-exports (pub use) | ~65 | zenavif, heic, rav1d-bench | KEEP |

---

## Findings

### Finding 1 — `include::dav1d::picture::Rav1dPictureDataComponentInner` (class B)

`Rav1dPictureDataComponentInner` is pub-exported from the `picture` module but is never referenced directly by any external consumer found in the scan. It is an internal storage type used only within `Rav1dPictureDataComponent`. Callers interact exclusively with `Rav1dPictureDataComponent` methods (`.index()`, `.slice()`, `.copy_pixels_to()`, etc.) and receive `DisjointImmutGuard`/`DisjointMutGuard` typed on it — they never name it directly.

- **Item:** `pub struct rav1d_safe::include::dav1d::picture::Rav1dPictureDataComponentInner`
- **Hits:** 0 external uses (scan 2026-06-11)
- **Action B:** `pub(crate)` — breaks nobody using the managed API, potentially breaks unsafe-asm integrators who construct `DisjointMut<PicBuf>` directly. Conservative: queue for next 0.x minor with a deprecation cycle.

### Finding 2 — `include::dav1d::picture::with_pixel_guard_immut` / `with_pixel_guard_mut` (class A)

These two free functions are pub-exported but are internal pipeline helpers — they take `WithOffset<&Rav1dPictureDataComponent>` (an internal type alias) as their first argument. No external consumer references them by name. They exist to share the pixel-access pattern across the crate's internal decode pipeline. External callers get pixels via `Planes8`/`Planes16` through the managed API.

- **Items:** `pub fn with_pixel_guard_immut`, `pub fn with_pixel_guard_mut`
- **Hits:** 0 external uses
- **Action A:** `#[doc(hidden)]` immediately, then B: `pub(crate)` in next breaking window.

### Finding 3 — `include::dav1d::picture::PicOffset` duplicate alias (class A)

Two type aliases for the same type exist in the public API:
- `pub type Rav1dPictureDataComponentOffset<'a>` (the primary)
- `pub type PicOffset<'a>` (an undocumented alias, line 1493 in picture.rs)

`PicOffset` has zero external consumers. It appears to be a convenience shorthand retained from internal refactoring.

- **Item:** `pub type PicOffset<'a>`
- **Hits:** 0 external uses
- **Action A:** `#[doc(hidden)]` immediately; B: remove in next breaking window.

### Finding 4 — `include::dav1d::picture::Rav1dPictureDataComponent::wrap_buf` and `Rav1dPictureDataComponentInner::wrap_buf` (class B)

Both `wrap_buf` constructors are public but exist solely to construct picture data buffers inside the decoder pipeline. External consumers never construct `Rav1dPictureDataComponent` directly — they receive it from `Dav1dPicture` fields after a decode call. No external hits found.

- **Items:** `pub fn Rav1dPictureDataComponent::wrap_buf`, `pub fn Rav1dPictureDataComponentInner::wrap_buf`
- **Hits:** 0 external uses
- **Action B:** `pub(crate)` queue, next breaking window.

### Finding 5 — `include::dav1d::headers::Rav1dWarpedMotionParams::abcd` pub field (class A)

The field `abcd: RelaxedAtomic<[i16; 4]>` is pub on `Rav1dWarpedMotionParams` and routes through `src::relaxed_atomic::RelaxedAtomic` — an internal atomic wrapper type. This is the only path through which `rav1d_safe::src::relaxed_atomic` leaks into the public signature. External code that reads this field must name `RelaxedAtomic` without being able to import it (pub(crate) module), making it practically unusable despite appearing in the API.

- **Item:** `pub rav1d_safe::include::dav1d::headers::Rav1dWarpedMotionParams::abcd`
- **Hits:** 0 external uses
- **Design note:** `RelaxedAtomic<[i16; 4]>` here is intentional — the motion params are written by the decoder and read by the film-grain pipeline concurrently. But the pub field exposes an internal type. Consider wrapping with accessor methods returning `[i16; 4]` directly.
- **Action A:** `#[doc(hidden)]` on the field until a clean accessor is added.

---

## What Was NOT Flagged

### `include::dav1d::*` wholesale (2,018 items)

The entire `include::dav1d` namespace is the C-ABI parity surface for integrators using the `unsafe-asm` feature path (which swaps the crate-internal decoder for hand-written assembly via `rav1d`). The `zenavif/decoder.rs` module, `zen-arm-src/zenavif/src/decoder.rs`, and `pre-filter/zenavif/src/decoder.rs` all use this namespace under `#[cfg(not(feature = "unsafe-asm"))]` to match the `rav1d` C FFI layout identically. Flagging individual items here would undermine the compatibility contract. **Not flagged.**

### `src::dav1d_api` (19 all-feature items)

The `dav1d_*` extern-C functions are a deliberate C-ABI compat contract for dynamic-linking consumers. **Not flagged.**

### `src::managed::*` (533 items)

All consumed by verified external code. **Not flagged.**

### `src::send_sync_non_null::SendSyncNonNull`

Used by `zenavif/decoder.rs` (unsafe-asm path) and required in `Dav1dLogger::new`'s signature. This module is currently `pub` only under `cfg(any(feature = "asm", feature = "c-ffi"))`. The all-features snapshot shows it. When neither asm nor c-ffi is active (the common safe path), it is correctly not pub. **Not flagged.**

### `include::common` (1 mod entry, no directly named items)

The `bitdepth::BitDepth` trait appears only as a generic bound in method signatures in `picture`. External callers use it implicitly via turbofish (e.g., `Rav1dPictureDataComponent::copy_pixels_to::<BitDepth8>`) — the trait bound must remain reachable. **Not flagged.**

---

## Top-3 Action Items

1. **`PicOffset` alias** — `#[doc(hidden)]`, 1-line change, zero risk. Eliminates a confusing duplicate in the docs.
2. **`with_pixel_guard_immut` / `with_pixel_guard_mut`** — `#[doc(hidden)]` both free functions. These are plumbing, not API surface; hiding avoids consumer confusion.
3. **`Rav1dWarpedMotionParams::abcd` field** — Add `#[doc(hidden)]` and a public accessor `fn abcd(&self) -> [i16; 4]` that reads through the RelaxedAtomic. Removes the internal-type leak from the documented surface.

---

## Report Paths

- Repo: `/home/lilith/work/zen/rav1d-safe/docs/public-api/ABLATION-rav1d-safe.md`
- Mirror: `/mnt/v/output/api-ablation/rav1d-safe--rav1d-safe.md`
