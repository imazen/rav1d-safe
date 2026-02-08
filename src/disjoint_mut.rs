//! Wrapper that allows concurrent, disjoint mutation of a slice-like owned
//! structure.
//!
//! This module re-exports the core `DisjointMut` type from the `disjoint_mut`
//! crate (a provably safe abstraction with always-on bounds checking by default),
//! and adds `AsMutPtr` implementations for rav1d-specific types (AlignedVec, Align*).

#![deny(unsafe_op_in_unsafe_fn)]

// Re-export everything from the disjoint-mut crate.
pub use disjoint_mut::AsMutPtr;
pub use disjoint_mut::Clearable;
pub use disjoint_mut::DisjointImmutGuard;
pub use disjoint_mut::DisjointMut;
pub use disjoint_mut::DisjointMutArcSlice;
pub use disjoint_mut::DisjointMutGuard;
pub use disjoint_mut::DisjointMutIndex;
pub use disjoint_mut::DisjointMutSlice;
pub use disjoint_mut::ExternalAsMutPtr;
pub use disjoint_mut::Resizable;
pub use disjoint_mut::ResizableWith;
pub use disjoint_mut::SliceBounds;
pub use disjoint_mut::TranslateRange;

// rav1d-specific extensions: AlignedVec AsMutPtr + Resizable impls.

use crate::src::align::AlignedByteChunk;
use crate::src::align::AlignedVec;
use crate::src::assume::assume;

/// Implement Resizable so that `DisjointMut<AlignedVec<V, C>>` gains `.resize()`.
impl<V: Copy, C: AlignedByteChunk> Resizable for AlignedVec<V, C> {
    type Value = V;
    fn resize(&mut self, new_len: usize, value: V) {
        AlignedVec::resize(self, new_len, value)
    }
}

/// SAFETY: We never materialize a `&mut [T]` since we
/// only materialize a `&mut AlignedVec<T, _>` and call [`AlignedVec::as_mut_ptr`] on it,
/// which calls [`Vec::as_mut_ptr`] and never materializes a `&mut [V]`.
#[allow(unsafe_code)]
unsafe impl<T: Copy, C: AlignedByteChunk> ExternalAsMutPtr for AlignedVec<T, C> {
    type Target = T;

    unsafe fn as_mut_ptr(ptr: *mut Self) -> *mut Self::Target {
        // SAFETY: `.as_mut_ptr()` does not materialize a `&mut` to
        // the underlying slice, so we can still allow `&`s into this slice.
        let ptr = unsafe { &mut *ptr }.as_mut_ptr();

        // SAFETY: `AlignedVec` stores `C`s internally,
        // so `*mut T` is really `*mut C`.
        // Since it's stored in a `Vec`, it's aligned.
        unsafe { assume(ptr.cast::<C>().is_aligned()) };

        ptr
    }

    fn len(&self) -> usize {
        self.len()
    }
}
