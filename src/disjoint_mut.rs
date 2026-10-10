//! Wrapper that allows concurrent, disjoint mutation of a slice-like owned
//! structure.
//!
//! Crate-internal imports of the `rav1d-disjoint-mut` types, plus the `dm_new` /
//! `dm_arc_try_new` constructors. The types are deliberately not re-exported from
//! here: the managed API does not name them, so their versioning stays the
//! dependency's business.

// Imported for the rest of the crate only.
pub(crate) use rav1d_disjoint_mut::AsMutPtr;
pub(crate) use rav1d_disjoint_mut::DisjointImmutGuard;
pub(crate) use rav1d_disjoint_mut::DisjointMut;
pub(crate) use rav1d_disjoint_mut::DisjointMutArcSlice;
pub(crate) use rav1d_disjoint_mut::DisjointMutGuard;
pub(crate) use rav1d_disjoint_mut::DisjointMutSlice;
#[cfg(feature = "c-ffi")]
pub(crate) use rav1d_disjoint_mut::ExternalAsMutPtr;
pub(crate) use rav1d_disjoint_mut::PlainData;
pub(crate) use rav1d_disjoint_mut::SliceBounds;
pub(crate) use rav1d_disjoint_mut::TryResizable;
pub(crate) use rav1d_disjoint_mut::TryResizableWith;

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
