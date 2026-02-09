//! Aligned newtypes and aligned Vec for SIMD-friendly data layout.
//!
//! Provides `Align{1,2,4,8,16,32,64}<T>` newtype wrappers that enforce
//! alignment via `#[repr(C, align(N))]`, and `AlignedVec<T, C>` for heap-allocated
//! aligned buffers. All types integrate with `DisjointMut` via `ExternalAsMutPtr`
//! and `Resizable`.

use crate::ExternalAsMutPtr;
use crate::Resizable;
use alloc::vec::Vec;
use core::hint::unreachable_unchecked;
use core::marker::PhantomData;
use core::mem;
use core::mem::MaybeUninit;
use core::ops::Deref;
use core::ops::DerefMut;
use core::ops::Index;
use core::ops::IndexMut;
use core::slice;

/// A stable version of [`core::intrinsics::assume`].
///
/// # Safety
///
/// `condition` must always be `true`.
#[inline(always)]
unsafe fn assume(condition: bool) {
    if !condition {
        // SAFETY: `condition` is `true` by the `# Safety` preconditions.
        unsafe { unreachable_unchecked() };
    }
}

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

// We want this to be implemented for all `T: Default` where `T` is not `[_; _]`,
// but we can't do that, so we can just add individual
// `impl`s here for types we need it for.
impl_ArrayDefault!(u8);
impl_ArrayDefault!(i8);
impl_ArrayDefault!(i16);
impl_ArrayDefault!(i32);
impl_ArrayDefault!(u16);

pub trait AlignedByteChunk
where
    Self: Sized,
{
}

macro_rules! def_align {
    ($align:literal, $name:ident) => {
        #[derive(Clone, Copy)]
        #[repr(C, align($align))]
        pub struct $name<T>(pub T);

        impl<T> From<T> for $name<T> {
            fn from(from: T) -> Self {
                Self(from)
            }
        }

        impl<T: Index<usize>> Index<usize> for $name<T> {
            type Output = T::Output;

            fn index(&self, index: usize) -> &Self::Output {
                &self.0[index]
            }
        }

        impl<T: IndexMut<usize>> IndexMut<usize> for $name<T> {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                &mut self.0[index]
            }
        }

        impl<T: ArrayDefault> ArrayDefault for $name<T> {
            fn default() -> Self {
                Self(T::default())
            }
        }

        impl<T: ArrayDefault> Default for $name<T> {
            fn default() -> Self {
                <Self as ArrayDefault>::default()
            }
        }

        impl AlignedByteChunk for $name<[u8; $align]> {}

        /// SAFETY: We never materialize a `&mut [V]` since we do a direct cast.
        unsafe impl<V: Copy, const N: usize> ExternalAsMutPtr for $name<[V; N]> {
            type Target = V;

            unsafe fn as_mut_ptr(ptr: *mut Self) -> *mut V {
                // SAFETY: `ptr` is safe to deref and thus is aligned.
                unsafe { assume(ptr.is_aligned()) }

                // SAFETY: `$name` (`Align*`) is a `#[repr(C)]` aligned wrapper around `[V; N]`,
                // so a `*mut Self` is the same as a `*mut V` to the first `V`.
                ptr.cast()
            }

            fn len(&self) -> usize {
                N
            }
        }
    };
}

def_align!(1, Align1);
def_align!(2, Align2);
def_align!(4, Align4);
def_align!(8, Align8);
def_align!(16, Align16);
def_align!(32, Align32);
def_align!(64, Align64);

/// A [`Vec`] that uses [`mem::size_of`]`::<C>()` aligned allocations.
///
/// Only works with [`Copy`] types so that we don't have to handle drop logic.
pub struct AlignedVec<T: Copy, C: AlignedByteChunk> {
    inner: Vec<MaybeUninit<C>>,

    /// The number of `T`s in [`Self::inner`] currently initialized.
    len: usize,
    _phantom: PhantomData<T>,
}

impl<T: Copy, C: AlignedByteChunk> AlignedVec<T, C> {
    /// Must check in all constructors.
    const fn check_byte_chunk_type_is_aligned() {
        assert!(mem::size_of::<C>() == mem::align_of::<C>());
    }

    const fn check_inner_type_is_aligned() {
        assert!(mem::align_of::<T>() <= mem::align_of::<C>());
    }

    pub const fn new() -> Self {
        Self::check_byte_chunk_type_is_aligned();
        Self::check_inner_type_is_aligned();

        Self {
            inner: Vec::new(),
            len: 0,
            _phantom: PhantomData,
        }
    }

    /// Return the number of elements in the vector.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn as_ptr(&self) -> *const T {
        self.inner.as_ptr().cast()
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.inner.as_mut_ptr().cast()
    }

    /// Extract a slice containing the entire vector.
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: The first `len` elements have been
        // initialized to `T`s in `Self::resize`.
        // SAFETY: The pointer is sufficiently aligned,
        // as the chunks are always over-aligned with
        // respect to `T`.
        unsafe { slice::from_raw_parts(self.as_ptr(), self.len) }
    }

    /// Extract a mutable slice of the entire vector.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: The first `len` elements have been
        // initialized to `T`s in `Self::resize`.
        // SAFETY: The pointer is sufficiently aligned,
        // as the chunks are always over-aligned with
        // respect to `T`.
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), self.len) }
    }

    pub fn resize(&mut self, new_len: usize, value: T) {
        let old_len = self.len();

        // Resize the underlying vector to have enough chunks for the new length.
        // The `new_bytes` calculation must not overflow,
        // ensuring a mathematical match with the underlying `inner` buffer size.
        // NOTE: one can still pass ludicrous requested buffer lengths, just not unsound ones.
        let new_bytes = mem::size_of::<T>()
            .checked_mul(new_len)
            .expect("Resizing would overflow the underlying aligned buffer");

        let chunk_size = mem::size_of::<C>();
        let new_chunks = if (new_bytes % chunk_size) == 0 {
            new_bytes / chunk_size
        } else {
            // NOTE: can not overflow. This case only occurs on `chunk_size >= 2`.
            (new_bytes / chunk_size) + 1
        };

        // NOTE: We don't need to `drop` any elements if the `Vec` is truncated since `T: Copy`.
        self.inner.resize_with(new_chunks, MaybeUninit::uninit);

        // If we grew the vector, initialize the new elements past `len`.
        for offset in old_len..new_len {
            // SAFETY: We've allocated enough space to write
            // up to `new_len` elements into the buffer.
            // SAFETY: The pointer is sufficiently aligned,
            // as the chunks are always over-aligned with
            // respect to `T`.
            unsafe { self.as_mut_ptr().add(offset).write(value) };
        }

        self.len = new_len;
    }
}

impl<T: Copy, C: AlignedByteChunk> AsRef<[T]> for AlignedVec<T, C> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: Copy, C: AlignedByteChunk> AsMut<[T]> for AlignedVec<T, C> {
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T: Copy, C: AlignedByteChunk> Deref for AlignedVec<T, C> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T: Copy, C: AlignedByteChunk> DerefMut for AlignedVec<T, C> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

// NOTE: Custom impl so that we don't require `T: Default`.
impl<T: Copy, C: AlignedByteChunk> Default for AlignedVec<T, C> {
    fn default() -> Self {
        Self::new()
    }
}

pub type AlignedVec32<T> = AlignedVec<T, Align32<[u8; 32]>>;
pub type AlignedVec64<T> = AlignedVec<T, Align64<[u8; 64]>>;

/// Implement Resizable so that `DisjointMut<AlignedVec<V, C>>` gains `.resize()`.
impl<V: Copy, C: AlignedByteChunk> Resizable for AlignedVec<V, C> {
    type Value = V;
    fn resize(&mut self, new_len: usize, value: V) {
        AlignedVec::resize(self, new_len, value)
    }
}

/// SAFETY: We only create `&AlignedVec` (SharedReadOnly), never `&mut AlignedVec`.
/// Creating `&mut AlignedVec` would produce a Unique retag (Stacked Borrows) covering
/// the inner Vec struct, invalidating concurrent `&AlignedVec` reads from other threads.
/// Instead, we read the data pointer through `as_ptr().cast_mut()` and the length
/// through `self.len()`, both of which only require shared references.
unsafe impl<T: Copy, C: AlignedByteChunk> ExternalAsMutPtr for AlignedVec<T, C> {
    type Target = T;

    unsafe fn as_mut_ptr(ptr: *mut Self) -> *mut Self::Target {
        // SAFETY: Only creates &AlignedVec (SharedReadOnly), not &mut AlignedVec.
        // as_ptr() reads the inner Vec's pointer through a shared reference,
        // which doesn't conflict with concurrent shared borrows from other threads.
        let aligned_ref = unsafe { &*ptr };
        let ptr = aligned_ref.as_ptr().cast_mut();

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

#[test]
#[should_panic]
fn align_vec_fails() {
    let mut v = AlignedVec::<u16, Align8<[u8; 8]>>::new();
    // This resize must fail. Otherwise, the code below creates a very small actual allocation, and
    // consequently a slice reference that points to memory outside the buffer.
    v.resize(isize::MAX as usize + 2, 0u16);
    // Note that in Rust, no single allocation can exceed `isize::MAX` _bytes_. Meaning it is
    // impossible to soundly create a slice of `u16` with `isize::MAX` elements. If we got to this
    // point, everything is broken already. The indexing will the probably also wrap and appear to
    // work.
    assert_eq!(v.as_slice()[isize::MAX as usize], 0);
}

#[test]
#[should_panic]
fn under_aligned_storage_fails() {
    let mut v = AlignedVec::<u128, Align1<[u8; 1]>>::new();

    // Would be UB: unaligned write of type u128 to pointer aligned to 1 (or 8 in most allocators,
    // but statically we only ensure an alignment of 1).
    v.resize(1, 0u128);
}
