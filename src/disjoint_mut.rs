//! Wrapper that allows concurrent, disjoint mutation of a slice-like owned
//! structure.
//!
//! This module re-exports the core `DisjointMut` type from the `rav1d-disjoint-mut`
//! crate (a provably safe abstraction with always-on bounds checking by default).
//!
//! AlignedVec and Align* `ExternalAsMutPtr` impls live in the `rav1d-align` crate.

// Re-export everything from the rav1d-disjoint-mut crate.
pub use rav1d_disjoint_mut::AsMutPtr;
pub use rav1d_disjoint_mut::DisjointImmutGuard;
pub use rav1d_disjoint_mut::DisjointMut;
pub use rav1d_disjoint_mut::DisjointMutArcSlice;
pub use rav1d_disjoint_mut::DisjointMutGuard;
pub use rav1d_disjoint_mut::DisjointMutSlice;
#[cfg(feature = "c-ffi")]
pub use rav1d_disjoint_mut::ExternalAsMutPtr;
pub use rav1d_disjoint_mut::SliceBounds;
pub use rav1d_disjoint_mut::TryResizable;
pub use rav1d_disjoint_mut::TryResizableWith;

/// Create a [`DisjointMut`] with tracking appropriate for the current build.
///
/// - Default build: always tracked (runtime overlap checking).
/// - `unchecked` feature: untracked (caller-guaranteed disjointness).
///
/// The `unchecked` path is sound because all access patterns in rav1d-safe
/// are verified by running the full conformance suite in checked mode.
#[cfg(not(feature = "unchecked"))]
pub fn dm_new<T: AsMutPtr>(val: T) -> DisjointMut<T> {
    DisjointMut::new(val)
}

/// See checked variant above.
#[cfg(feature = "unchecked")]
#[allow(unsafe_code)]
pub fn dm_new<T: AsMutPtr>(val: T) -> DisjointMut<T> {
    // SAFETY: All borrow patterns in rav1d-safe are verified to be disjoint
    // by running the full 784-vector conformance suite in checked mode.
    unsafe { DisjointMut::dangerously_unchecked(val) }
}

/// Create a [`DisjointMutArcSlice`] with tracking appropriate for the current build.
#[cfg(not(feature = "unchecked"))]
pub fn dm_arc_try_new<T: Copy>(
    n: usize,
    value: T,
) -> Result<DisjointMutArcSlice<T>, alloc::collections::TryReserveError> {
    DisjointMutArcSlice::try_new(n, value)
}

/// See checked variant above.
#[cfg(feature = "unchecked")]
#[allow(unsafe_code)]
pub fn dm_arc_try_new<T: Copy>(
    n: usize,
    value: T,
) -> Result<DisjointMutArcSlice<T>, alloc::collections::TryReserveError> {
    // SAFETY: See dm_new safety comment.
    unsafe { DisjointMutArcSlice::try_new_unchecked(n, value) }
}

extern crate alloc;
