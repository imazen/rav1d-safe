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
pub use rav1d_disjoint_mut::PlainData;
pub use rav1d_disjoint_mut::SliceBounds;
pub use rav1d_disjoint_mut::TryResizable;
pub use rav1d_disjoint_mut::TryResizableWith;

/// Create a [`DisjointMut`] with tracking appropriate for the current build.
///
/// Always tracked, except under the `untracked` feature, where the crate's
/// constructors build tracker-free storage themselves.
pub fn dm_new<T: AsMutPtr>(val: T) -> DisjointMut<T> {
    DisjointMut::new_eager(val)
}

/// Create a [`DisjointMutArcSlice`] with tracking appropriate for the current build.
pub fn dm_arc_try_new<T: PlainData>(
    n: usize,
    value: T,
) -> Result<DisjointMutArcSlice<T>, alloc::collections::TryReserveError> {
    DisjointMutArcSlice::try_new(n, value)
}

extern crate alloc;
