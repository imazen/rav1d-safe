//! Aligned newtypes and aligned Vec for SIMD-friendly data layout.
//!
//! Thin wrappers around [`struct@aligned::Aligned`] and [`aligned_vec::AVec`] that
//! integrate with [`DisjointMut`](crate::DisjointMut) via [`ExternalAsMutPtr`]
//! and [`Resizable`].
//!
//! The wrapper types preserve the same public API as the previous custom
//! implementations: `Align{4,8,16,32,64}<T>` for stack alignment and
//! `AlignedVec{32,64}<T>` for heap-allocated aligned buffers.

use crate::ExternalAsMutPtr;
use crate::Resizable;
use crate::TryResizable;
use aligned_vec::{AVec, ConstAlign};
use core::ops::{Deref, DerefMut};

/// Create a `TryReserveError` to represent allocation failure from `aligned_vec`.
///
/// `std::collections::TryReserveError` has no public constructor, so we trigger
/// a real one by asking a `Vec` to reserve `usize::MAX`.
fn alloc_err() -> std::collections::TryReserveError {
    alloc::vec::Vec::<u8>::new()
        .try_reserve(usize::MAX)
        .unwrap_err()
}

// =============================================================================
// ArrayDefault — workaround for Default not covering [T; N>32]
// =============================================================================

/// [`Default`] isn't `impl`emented for all arrays `[T; N]`
/// because they were implemented before `const` generics
/// and thus only for low values of `N`.
pub trait ArrayDefault {
    fn default() -> Self;
}

impl<T: ArrayDefault + Copy, const N: usize> ArrayDefault for [T; N] {
    fn default() -> Self {
        [T::default(); N]
    }
}

impl<T> ArrayDefault for Option<T> {
    fn default() -> Self {
        None
    }
}

macro_rules! impl_ArrayDefault {
    ($T:ty) => {
        impl ArrayDefault for $T {
            fn default() -> Self {
                <Self as Default>::default()
            }
        }
    };
}

impl_ArrayDefault!(u8);
impl_ArrayDefault!(i8);
impl_ArrayDefault!(i16);
impl_ArrayDefault!(i32);
impl_ArrayDefault!(u16);

// =============================================================================
// Align wrappers — use aligned::Aligned for the actual alignment guarantee
// =============================================================================

/// Re-export so downstream can refer to `aligned::Aligned` if needed.
pub use aligned::Aligned;

/// Const-compatible accessor for the inner value of an [`struct@Aligned`] wrapper.
///
/// [`struct@Aligned`] is `#[repr(C)]` with layout `{ _alignment: [A; 0], value: T }`.
/// Since `[A; 0]` is zero-sized, the pointer to the struct equals the pointer
/// to `value`. This function uses `transmute` to access the inner `T` in
/// `const` contexts where `Deref` is not available.
///
/// # Safety
///
/// This relies on the documented `#[repr(C)]` layout of [`struct@aligned::Aligned`].
/// The `[A; 0]` field has zero size and the same alignment as `A`, so
/// `&Aligned<A, T>` and `&T` share the same address.
pub const fn aligned_inner<A, T: ?Sized>(aligned: &Aligned<A, T>) -> &T {
    // SAFETY: Aligned<A, T> is repr(C) with [A; 0] (ZST) followed by T.
    // The address of the struct is the address of T.
    unsafe { &*(aligned as *const Aligned<A, T> as *const T) }
}

macro_rules! def_align {
    ($align:literal, $align_ty:ident, $name:ident) => {
        /// Aligned newtype wrapper. Enforces at least
        #[doc = concat!(stringify!($align), "-byte")]
        /// alignment.
        ///
        /// This is a type alias for [`struct@aligned::Aligned`] with the corresponding
        /// alignment marker.
        pub type $name<T> = aligned::Aligned<aligned::$align_ty, T>;

        /// SAFETY: `Aligned<$align_ty, [V; N]>` is `#[repr(C)]` with alignment `$align_ty`.
        /// A pointer to the struct has the same address as the first element of the
        /// inner `[V; N]`, so `ptr.cast::<V>()` is valid.
        /// We never create any references — just a direct pointer cast.
        ///
        /// `as_mut_slice` is overridden to avoid creating `&Aligned<_, [V; N]>`,
        /// which would be a SharedReadOnly tag covering the inline data — UB
        /// under Stacked Borrows when concurrent mutable guards exist.
        unsafe impl<V: Copy, const N: usize> ExternalAsMutPtr
            for aligned::Aligned<aligned::$align_ty, [V; N]>
        {
            type Target = V;

            unsafe fn as_mut_ptr(ptr: *mut Self) -> *mut V {
                ptr.cast()
            }

            unsafe fn as_mut_slice(ptr: *mut Self) -> *mut [V] {
                core::ptr::slice_from_raw_parts_mut(ptr.cast::<V>(), N)
            }

            fn len(&self) -> usize {
                N
            }
        }
    };
}

def_align!(4, A4, Align4);
def_align!(8, A8, Align8);
def_align!(16, A16, Align16);
def_align!(32, A32, Align32);
def_align!(64, A64, Align64);

// ArrayDefault for Aligned — allows Default to work for arrays with N > 32.
impl<A, T: ArrayDefault> ArrayDefault for aligned::Aligned<A, T> {
    fn default() -> Self {
        aligned::Aligned(T::default())
    }
}

// =============================================================================
// AlignedVec wrappers — thin newtypes around aligned_vec::AVec
// =============================================================================

/// A heap-allocated, 64-byte-aligned buffer.
///
/// Wraps [`AVec<T, ConstAlign<64>>`] with a [`Default`] impl and
/// [`ExternalAsMutPtr`] integration for [`DisjointMut`](crate::DisjointMut).
pub struct AlignedVec64<T>(AVec<T, ConstAlign<64>>);

impl<T> AlignedVec64<T> {
    /// Create a new empty aligned vector.
    pub fn new() -> Self {
        Self(AVec::new(64))
    }

    /// Resize the buffer, filling new elements with `value`.
    pub fn resize(&mut self, new_len: usize, value: T)
    where
        T: Clone,
    {
        self.0.resize(new_len, value);
    }
}

impl<T> Default for AlignedVec64<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Deref for AlignedVec64<T> {
    type Target = [T];
    #[inline(always)]
    fn deref(&self) -> &[T] {
        self.0.as_slice()
    }
}

impl<T> DerefMut for AlignedVec64<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut [T] {
        self.0.as_mut_slice()
    }
}

impl<T> AsRef<[T]> for AlignedVec64<T> {
    fn as_ref(&self) -> &[T] {
        self.0.as_slice()
    }
}

impl<T> AsMut<[T]> for AlignedVec64<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut_slice()
    }
}

impl<T: Clone> Resizable for AlignedVec64<T> {
    type Value = T;
    fn resize(&mut self, new_len: usize, value: T) {
        self.0.resize(new_len, value);
    }
}

impl<T: Clone> TryResizable for AlignedVec64<T> {
    type Value = T;
    fn try_resize(
        &mut self,
        new_len: usize,
        value: T,
    ) -> Result<(), std::collections::TryReserveError> {
        if new_len > self.0.len() {
            self.0
                .try_reserve(new_len - self.0.len())
                .map_err(|_| alloc_err())?;
        }
        self.0.resize(new_len, value);
        Ok(())
    }
}

/// SAFETY: We only create `&AVec` (SharedReadOnly), never `&mut AVec`.
/// `as_ptr()` takes `&self` and returns `*const T`, which we cast to `*mut T`.
/// This avoids the Stacked Borrows hazard of creating `&mut AVec` through
/// `as_mut_ptr(&mut self)`, which would issue a Unique retag over the heap data.
unsafe impl<T: Copy> ExternalAsMutPtr for AlignedVec64<T> {
    type Target = T;

    unsafe fn as_mut_ptr(ptr: *mut Self) -> *mut T {
        // SAFETY: Only creates &Self (SharedReadOnly).
        let shared_ref = unsafe { &*ptr };
        shared_ref.0.as_ptr().cast_mut()
    }

    unsafe fn as_mut_slice(ptr: *mut Self) -> *mut [T] {
        // SAFETY: Only creates &Self (SharedReadOnly). Heap data is a
        // separate allocation, so SharedReadOnly doesn't cover it.
        let shared_ref = unsafe { &*ptr };
        core::ptr::slice_from_raw_parts_mut(shared_ref.0.as_ptr().cast_mut(), shared_ref.0.len())
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

/// A heap-allocated, 32-byte-aligned buffer.
///
/// Wraps [`AVec<T, ConstAlign<32>>`] with a [`Default`] impl and
/// [`ExternalAsMutPtr`] integration for [`DisjointMut`](crate::DisjointMut).
pub struct AlignedVec32<T>(AVec<T, ConstAlign<32>>);

impl<T> AlignedVec32<T> {
    /// Create a new empty aligned vector.
    pub fn new() -> Self {
        Self(AVec::new(32))
    }

    /// Resize the buffer, filling new elements with `value`.
    pub fn resize(&mut self, new_len: usize, value: T)
    where
        T: Clone,
    {
        self.0.resize(new_len, value);
    }
}

impl<T> Default for AlignedVec32<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Deref for AlignedVec32<T> {
    type Target = [T];
    #[inline(always)]
    fn deref(&self) -> &[T] {
        self.0.as_slice()
    }
}

impl<T> DerefMut for AlignedVec32<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut [T] {
        self.0.as_mut_slice()
    }
}

impl<T> AsRef<[T]> for AlignedVec32<T> {
    fn as_ref(&self) -> &[T] {
        self.0.as_slice()
    }
}

impl<T> AsMut<[T]> for AlignedVec32<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut_slice()
    }
}

impl<T: Clone> Resizable for AlignedVec32<T> {
    type Value = T;
    fn resize(&mut self, new_len: usize, value: T) {
        self.0.resize(new_len, value);
    }
}

impl<T: Clone> TryResizable for AlignedVec32<T> {
    type Value = T;
    fn try_resize(
        &mut self,
        new_len: usize,
        value: T,
    ) -> Result<(), std::collections::TryReserveError> {
        if new_len > self.0.len() {
            self.0
                .try_reserve(new_len - self.0.len())
                .map_err(|_| alloc_err())?;
        }
        self.0.resize(new_len, value);
        Ok(())
    }
}

/// SAFETY: See [`AlignedVec64`]'s impl for rationale.
unsafe impl<T: Copy> ExternalAsMutPtr for AlignedVec32<T> {
    type Target = T;

    unsafe fn as_mut_ptr(ptr: *mut Self) -> *mut T {
        let shared_ref = unsafe { &*ptr };
        shared_ref.0.as_ptr().cast_mut()
    }

    unsafe fn as_mut_slice(ptr: *mut Self) -> *mut [T] {
        let shared_ref = unsafe { &*ptr };
        core::ptr::slice_from_raw_parts_mut(shared_ref.0.as_ptr().cast_mut(), shared_ref.0.len())
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DisjointMut;

    #[test]
    fn align_type_alias_layout() {
        assert_eq!(core::mem::align_of::<Align4<[u8; 3]>>(), 4);
        assert_eq!(core::mem::align_of::<Align8<[u8; 3]>>(), 8);
        assert_eq!(core::mem::align_of::<Align16<[u8; 3]>>(), 16);
        assert_eq!(core::mem::align_of::<Align32<[u8; 3]>>(), 32);
        assert_eq!(core::mem::align_of::<Align64<[u8; 3]>>(), 64);
    }

    #[test]
    fn align_constructor_and_deref() {
        use aligned::Aligned;

        let x: Align16<[u16; 4]> = Aligned([1u16, 2, 3, 4]);
        assert_eq!(x[2], 3);

        let mut y: Align32<[u8; 8]> = Aligned([0u8; 8]);
        y[5] = 42;
        assert_eq!(y[5], 42);
    }

    #[test]
    fn align_copy() {
        use aligned::Aligned;

        let x: Align16<[u8; 3]> = Aligned([1u8, 2, 3]);
        let y = x;
        let _ = x; // still valid (Copy)
        assert_eq!(y[0], 1);
    }

    #[test]
    fn align_array_default() {
        // Small array (N <= 32) — uses standard Default
        let x: Align16<[u16; 4]> = ArrayDefault::default();
        assert_eq!(x[0], 0);

        // Large array (N > 32) — uses ArrayDefault
        let y: Align8<[[u16; 4]; 41]> = ArrayDefault::default();
        assert_eq!(y[40][3], 0);
    }

    #[test]
    fn align_in_repr_c_struct() {
        use aligned::Aligned;

        #[repr(C)]
        struct Foo {
            a: Align4<[u16; 2]>,
            b: Align16<[u8; 32]>,
        }

        let f = Foo {
            a: Aligned([100u16, 200]),
            b: Aligned([0u8; 32]),
        };
        assert_eq!(f.a[1], 200);
    }

    #[test]
    fn align_disjoint_mut() {
        use aligned::Aligned;

        let dm = DisjointMut::new(Aligned::<aligned::A16, _>([0u8; 64]));
        dm.index_mut(0..32).fill(1);
        dm.index_mut(32..64).fill(2);
        assert!(dm.index(0..32).iter().all(|&x| x == 1));
        assert!(dm.index(32..64).iter().all(|&x| x == 2));
    }

    #[test]
    fn aligned_vec64_basic() {
        let mut v = AlignedVec64::<u8>::new();
        assert_eq!(v.len(), 0);
        v.0.resize(100, 0u8);
        assert_eq!(v.len(), 100);
        v[50] = 42;
        assert_eq!(v[50], 42);
    }

    #[test]
    fn aligned_vec64_default() {
        let v: AlignedVec64<u8> = Default::default();
        assert_eq!(v.len(), 0);
    }

    #[test]
    fn aligned_vec64_disjoint_mut() {
        let mut v = AlignedVec64::<u8>::new();
        v.0.resize(100, 0u8);
        let dm = DisjointMut::new(v);

        dm.index_mut(0..50).fill(1);
        dm.index_mut(50..100).fill(2);

        assert!(dm.index(0..50).iter().all(|&x| x == 1));
        assert!(dm.index(50..100).iter().all(|&x| x == 2));
    }

    #[test]
    fn aligned_vec64_resizable() {
        let v = AlignedVec64::<u8>::new();
        let mut dm = DisjointMut::new(v);
        dm.resize(64, 0xFFu8);
        assert_eq!(dm.index(0..64).len(), 64);
        assert!(dm.index(0..64).iter().all(|&x| x == 0xFF));
    }

    #[test]
    fn deref_to_inner() {
        use aligned::Aligned;

        let x: Align64<[u8; 4]> = Aligned([1, 2, 3, 4]);
        // Access through Deref (no .0 or pattern destructure needed)
        let inner: &[u8; 4] = &*x;
        assert_eq!(*inner, [1, 2, 3, 4]);
    }
}
