# rav1d-disjoint-mut Public-API Ablation Report

**Date:** 2026-06-11
**Snapshot commit:** 0346a92a (feat: versioned public-API surface snapshots)
**Snapshot:** `docs/public-api/rav1d-disjoint-mut.txt`
**Crate location:** `crates/rav1d-disjoint-mut/`
**Mode:** COMMIT (report only, no source changes)
**Grep template:** `grep -r 'rav1d_disjoint_mut::<SYMBOL>' /home/lilith/work/ --include='*.rs' --exclude-dir=rav1d-safe --exclude-dir=target --exclude-dir=.jj`
**Scan as of:** 2026-06-11 (checked zenavif, heic, aom-decoder-rs, zenmetrics, rav1d-bench)

---

## Summary

| Metric | Value |
|--------|-------|
| Default-feature items | 223 |
| All-feature items (`aligned` + `pic-buf` + `zerocopy`) | 381 |
| Known external consumers | rav1d-safe (primary), zenavif (indirect via PlaneView8/16) |
| Direct external use of rav1d-disjoint-mut symbols | 0 files found outside rav1d-safe |
| Flagged class A (doc_hidden/deprecated) | 1 group |
| Flagged class B (pub→pub(crate) candidates) | 1 item |
| Flagged as percentage of default total | ~2 % |

**Conservative stance:** `rav1d-disjoint-mut` is explicitly described as a utility crate — its consumers are by definition external (cross-crate is external). The primary documented consumer is `rav1d-safe` itself; however, `PlaneView8`/`PlaneView16` in the managed API hold `DisjointImmutGuard` by value, which means zenavif consumers who call `.rows()` on a `PlaneView8` instantiate the `DisjointImmutGuard` type implicitly. The full API is therefore justified: `DisjointMut`, `DisjointMutGuard`, `DisjointImmutGuard`, `Bounds`, all the index traits, all the resize traits.

The `align` module (`AlignedVec32`, `AlignedVec64`, `ArrayDefault`) is gated behind the `aligned` feature and counts as legitimate SIMD utility surface. The `pic-buf` feature (`PicBuf`, `AsMutPtr` for `PicBuf`) is an implementation detail of rav1d's picture buffer allocation — useful to expose for downstream crates that want to integrate at the same allocation level, but not consumed by anyone in the scan.

---

## Module Breakdown (default features, 223 items)

| Type/Group | Count | External consumers | Verdict |
|------------|-------|--------------------|---------|
| `DisjointMut<T>` struct + methods | ~45 | rav1d-safe, zenavif (indirect) | KEEP |
| `DisjointMutGuard<T,V>` + impls | ~20 | rav1d-safe, zenavif (indirect) | KEEP |
| `DisjointImmutGuard<T,V>` + impls | ~18 | rav1d-safe, zenavif (indirect) | KEEP |
| `DisjointMutArcSlice<T>` + impls | ~30 | rav1d-safe | KEEP |
| `Bounds` + impls | ~15 | rav1d-safe | KEEP |
| `SliceBounds`, `TranslateRange`, `DisjointMutIndex` traits | ~30 | rav1d-safe | KEEP |
| `Resizable`, `TryResizable`, `ResizableWith`, `TryResizableWith`, `Clearable` traits | ~20 | rav1d-safe | KEEP |
| `AsMutPtr` impls for `[V; N]`, `[V]`, `Box<[V]>` | ~35 | rav1d-safe | KEEP |
| `dangerously_unchecked` constructor | 1 | rav1d-safe internals | See flag |
| `align` module (all-features only) | ~90 | rav1d-safe internals | KEEP |
| `pic-buf` / `AsMutPtr` trait (all-features) | ~40 | rav1d-safe internals | See flag |
| `zerocopy` feature additions | ~28 | rav1d-safe SIMD pipeline | KEEP |
| `ExternalAsMutPtr` trait (all-features) | ~12 | rav1d-safe internals | See flag |

---

## Findings

### Finding 1 — `DisjointMut::dangerously_unchecked` constructor (class A)

`pub unsafe const fn dangerously_unchecked(T) -> Self` is a constructor that bypasses all borrow tracking. Its name signals danger, but it is nonetheless a pub item that external crates can call. Within the scan no external crate calls it; it exists for rav1d-safe internals that need to initialize a `DisjointMut` before the borrow tracker is live.

- **Item:** `pub unsafe const fn rav1d_disjoint_mut::DisjointMut<T>::dangerously_unchecked`
- **Hits:** 0 external uses (scan 2026-06-11)
- **Action A:** `#[doc(hidden)]` to keep it off the public docs surface without a semver break. Signal intent to narrow visibility in a future breaking release.

### Finding 2 — `ExternalAsMutPtr` unsafe trait (all-features, class B)

Under all-features, `pub unsafe trait ExternalAsMutPtr` is exported. This trait is the extension point allowing crates to register custom pointer-bearing types with `DisjointMut`. Within the workspace only `rav1d-disjoint-mut/src/lib.rs`'s internal `AlignedVec32`/`AlignedVec64` impls use it. No external crate in the scan implements `ExternalAsMutPtr`.

However: it is the intended extension mechanism — any downstream crate that wants to wrap a custom aligned allocation in `DisjointMut` needs this trait. The `aligned` feature gate makes it all-features only, which is already a signal that it's advanced surface. This is borderline.

- **Item:** `pub unsafe trait rav1d_disjoint_mut::ExternalAsMutPtr` (all-features)
- **Hits:** 0 external implementations found
- **Conservative verdict:** KEEP for now — it's the correct extension point by design; hiding it would silently break any downstream that wants to plug in a custom allocator. Not flagged for action.

### Finding 3 — `pic-buf` feature items (class B, conditional)

The `pic-buf` feature exports `PicBuf`, `AsMutPtr for PicBuf`, and the full `AsMutPtr` unsafe trait (which becomes pub under this feature). `PicBuf` is rav1d's internal picture buffer type — a contiguous `u8` region with an optional `Arc<Box<[u8]>>` owner. No external crate in the scan references `PicBuf` or implements `AsMutPtr for PicBuf`.

Unlike `ExternalAsMutPtr`, `PicBuf` is a concrete rav1d-specific type that doesn't belong in a general-purpose utility crate's stable API. It exists here to move unsafe code into the disjoint-mut crate and keep rav1d-safe's default build under `forbid(unsafe_code)`.

- **Items:** `pub struct rav1d_disjoint_mut::pic_buf::PicBuf` and its `AsMutPtr` impl
- **Hits:** 0 external uses
- **Action B:** Queue for a `pub(crate)` or internal-only feature flag in the next breaking window. Could move the `AsMutPtr` impl for `PicBuf` to rav1d-safe itself (re-sealing the extension point from outside) since no third-party wants to impl AsMutPtr for PicBuf.

---

## What Was NOT Flagged

### `AsMutPtr` base trait impls for `[V; N]`, `[V]`, `Box<[V]>`

These are the general-purpose `AsMutPtr` impls that let `DisjointMut<Box<[u8]>>`, `DisjointMut<[u8; N]>`, etc. work. They're the primary consumer use case for any crate that wants disjoint slice mutation. **Not flagged.**

### All resize traits (`Resizable`, `TryResizable`, `ResizableWith`, `TryResizableWith`, `Clearable`)

Clean, general-purpose extension points matching Rust's standard `Vec::resize`/`try_reserve` semantics. Legit API surface. **Not flagged.**

### `align::AlignedVec32`/`AlignedVec64`, `align::ArrayDefault`

Used internally by rav1d-safe SIMD buffers. Exposing them publicly allows any SIMD-oriented crate to use aligned allocations via the `DisjointMut` borrow-tracking wrapper. This is genuinely useful general API. **Not flagged.**

### `DisjointMutArcSlice<T>`

Used by rav1d-safe for the Arc-shared picture component allocation. Externally, any crate sharing a `DisjointMut` slice across threads via Arc would need this. **Not flagged.**

---

## Top-3 Action Items

1. **`dangerously_unchecked`** — `#[doc(hidden)]`, 1-line change. Not part of the intended consumer API; hides a footgun from docs.
2. **`PicBuf` and `AsMutPtr for PicBuf`** — Queue for next breaking window to either move to rav1d-safe or mark as `#[doc(hidden)]`. The pic-buf feature is rav1d-safe-internal infrastructure.
3. **No third item** — remaining surface is clean, consumed, and justified.

---

## Report Paths

- Repo: `/home/lilith/work/zen/rav1d-safe/docs/public-api/ABLATION-rav1d-disjoint-mut.md`
- Mirror: `/mnt/v/output/api-ablation/rav1d-safe--rav1d-disjoint-mut.md`
