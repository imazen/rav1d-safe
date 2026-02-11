//! Provably safe abstraction for concurrent, disjoint mutation of contiguous storage.
//!
//! [`DisjointMut`] wraps a collection and allows non-overlapping mutable borrows
//! through a shared `&` reference. Like [`RefCell`](std::cell::RefCell), it enforces
//! borrowing rules at runtime — but instead of whole-container borrows, it tracks
//! *ranges* and panics only on truly overlapping access.
//!
//! # Safety Model
//!
//! By default, every `.index()` and `.index_mut()` call validates that the requested
//! range doesn't overlap with any outstanding borrow. This makes `DisjointMut` a
//! **sound safe abstraction**: safe code cannot cause undefined behavior.
//!
//! For performance-critical code that has been audited for correctness, the
//! `dangerously_unchecked()` unsafe constructor skips runtime tracking.
//!
//! # Example
//!
//! ```
//! use rav1d_disjoint_mut::DisjointMut;
//!
//! let mut buf = DisjointMut::new(vec![0u8; 100]);
//! // Borrow two non-overlapping regions simultaneously through &buf:
//! let a = buf.index(0..50);
//! let b = buf.index(50..100);
//! assert_eq!(a.len() + b.len(), 100);
//! ```

#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "aligned")]
pub mod align;

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::cell::UnsafeCell;
use core::fmt;
use core::fmt::Debug;
use core::fmt::Display;
use core::fmt::Formatter;
use core::marker::PhantomData;
use core::mem;
#[cfg(feature = "zerocopy")]
use core::mem::ManuallyDrop;
use core::ops::Deref;
use core::ops::DerefMut;
use core::ops::Index;
use core::ops::Range;
use core::ops::RangeFrom;
use core::ops::RangeFull;
use core::ops::RangeInclusive;
use core::ops::RangeTo;
use core::ops::RangeToInclusive;
use core::ptr;
use core::ptr::addr_of_mut;
#[cfg(feature = "zerocopy")]
use zerocopy::AsBytes;
#[cfg(feature = "zerocopy")]
use zerocopy::FromBytes;

// =============================================================================
// Core types
// =============================================================================

/// Wraps an indexable collection to allow unchecked concurrent mutable borrows.
///
/// This wrapper allows users to concurrently mutably borrow disjoint regions or
/// elements from a collection. This is necessary to allow multiple threads to
/// concurrently read and write to disjoint pixel data from the same arrays and
/// vectors.
///
/// Indexing returns a guard which acts as a lock for the borrowed region.
/// By default, borrows are validated at runtime to ensure that mutably borrowed
/// regions are actually disjoint with all other borrows for the lifetime of the
/// returned guard. This makes `DisjointMut` a provably safe abstraction (like `RefCell`).
///
/// For audited hot paths, enable the `unchecked` feature and use
/// `DisjointMut::dangerously_unchecked` to skip tracking.
pub struct DisjointMut<T: ?Sized + AsMutPtr> {
    tracker: Option<checked::BorrowTracker>,

    inner: UnsafeCell<T>,
}

/// SAFETY: If `T: Send`, then sending `DisjointMut<T>` across threads is safe.
/// There is no non-`Sync` state that is left on another thread
/// when `DisjointMut` gets sent to another thread.
unsafe impl<T: ?Sized + AsMutPtr + Send> Send for DisjointMut<T> {}

/// SAFETY: `DisjointMut` only provides disjoint mutable access
/// to `T`'s elements through a shared `&DisjointMut<T>` reference.
/// Thus, sharing/`Send`ing a `&DisjointMut<T>` across threads is safe.
///
/// In checked mode (default), the borrow tracker prevents overlapping borrows,
/// so no data races are possible. In unchecked mode (`dangerously_unchecked`),
/// the caller guarantees disjointness via the `unsafe` constructor contract.
unsafe impl<T: ?Sized + AsMutPtr + Sync> Sync for DisjointMut<T> {}

impl<T: AsMutPtr + Default> Default for DisjointMut<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<T: ?Sized + AsMutPtr> Debug for DisjointMut<T> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_struct("DisjointMut")
            .field("len", &self.len())
            .field("checked", &self.is_checked())
            .finish_non_exhaustive()
    }
}

impl<T: ?Sized + AsMutPtr> DisjointMut<T> {
    /// Returns `true` if this instance performs runtime overlap checking.
    pub const fn is_checked(&self) -> bool {
        self.tracker.is_some()
    }

    /// Returns a raw pointer to the inner container.
    ///
    /// This bypasses the borrow tracker entirely. The pointer is safe to
    /// *read through* for accessing container metadata (e.g. stride, length)
    /// that doesn't alias with element data. Dereferencing as `&mut` or writing
    /// through it requires the caller to uphold disjointness manually.
    ///
    /// # Why this exists
    ///
    /// Some containers store metadata alongside the data pointer (e.g.
    /// `Rav1dPictureDataComponentInner` stores stride). Accessing that metadata
    /// doesn't conflict with element borrows, so requiring a guard would be
    /// unnecessarily restrictive.
    ///
    /// # Safety
    ///
    /// The returned ptr has the safety requirements of [`UnsafeCell::get`].
    /// In particular, the ptr returned by [`AsMutPtr::as_mut_ptr`] may be in use.
    pub const fn inner(&self) -> *mut T {
        self.inner.get()
    }
}

impl<T: AsMutPtr> DisjointMut<T> {
    pub const fn new(value: T) -> Self {
        Self {
            inner: UnsafeCell::new(value),
            #[cfg(not(feature = "unchecked"))]
            tracker: Some(checked::BorrowTracker::new()),
            #[cfg(feature = "unchecked")]
            tracker: None,
        }
    }

    pub fn into_inner(self) -> T {
        self.inner.into_inner()
    }
}

// =============================================================================
// Guard types
// =============================================================================

/// Scope guard that poisons the `DisjointMut` if the indexing operation panics
/// (e.g., out-of-bounds). Disarmed via `mem::forget` on success.
///
/// Rather than cleaning up the leaked borrow record (which would allow the range
/// to be re-borrowed in potentially inconsistent state), we poison the entire
/// data structure. This follows the `std::sync::Mutex` pattern: after a panic,
/// fail loudly on all subsequent access.
struct BorrowCleanup<'a, T: ?Sized + AsMutPtr> {
    parent: Option<&'a DisjointMut<T>>,
}

impl<T: ?Sized + AsMutPtr> Drop for BorrowCleanup<'_, T> {
    fn drop(&mut self) {
        // This only fires on panic (mem::forget on success path).
        // Poison rather than clean up — the data structure is compromised.
        if let Some(parent) = self.parent {
            parent.tracker.as_ref().unwrap().poison();
        }
    }
}

pub struct DisjointMutGuard<'a, T: ?Sized + AsMutPtr, V: ?Sized> {
    slice: &'a mut V,

    phantom: PhantomData<&'a DisjointMut<T>>,

    /// Reference to parent for borrow removal on drop.
    /// `None` when parent was created with `dangerously_unchecked`.
    parent: Option<&'a DisjointMut<T>>,
    /// Unique ID for this borrow registration.
    borrow_id: checked::BorrowId,
}

#[cfg(feature = "zerocopy")]
impl<'a, T: AsMutPtr> DisjointMutGuard<'a, T, [u8]> {
    #[inline] // Inline to see alignment to potentially elide checks.
    fn cast_slice<V: AsBytes + FromBytes>(self) -> DisjointMutGuard<'a, T, [V]> {
        // We don't want to drop the old guard, because we aren't changing or
        // removing the borrow from parent here.
        let mut old_guard = ManuallyDrop::new(self);
        let bytes = mem::take(&mut old_guard.slice);
        DisjointMutGuard {
            slice: V::mut_slice_from(bytes).unwrap(),
            phantom: old_guard.phantom,
            parent: old_guard.parent,
            borrow_id: old_guard.borrow_id,
        }
    }

    #[inline] // Inline to see alignment to potentially elide checks.
    fn cast<V: AsBytes + FromBytes>(self) -> DisjointMutGuard<'a, T, V> {
        let mut old_guard = ManuallyDrop::new(self);
        let bytes = mem::take(&mut old_guard.slice);
        DisjointMutGuard {
            slice: V::mut_from(bytes).unwrap(),
            phantom: old_guard.phantom,
            parent: old_guard.parent,
            borrow_id: old_guard.borrow_id,
        }
    }
}

impl<'a, T: ?Sized + AsMutPtr, V: ?Sized> Deref for DisjointMutGuard<'a, T, V> {
    type Target = V;

    fn deref(&self) -> &Self::Target {
        self.slice
    }
}

impl<'a, T: ?Sized + AsMutPtr, V: ?Sized> DerefMut for DisjointMutGuard<'a, T, V> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.slice
    }
}

pub struct DisjointImmutGuard<'a, T: ?Sized + AsMutPtr, V: ?Sized> {
    slice: &'a V,

    phantom: PhantomData<&'a DisjointMut<T>>,

    parent: Option<&'a DisjointMut<T>>,
    borrow_id: checked::BorrowId,
}

#[cfg(feature = "zerocopy")]
impl<'a, T: AsMutPtr> DisjointImmutGuard<'a, T, [u8]> {
    #[inline]
    fn cast_slice<V: FromBytes>(self) -> DisjointImmutGuard<'a, T, [V]> {
        let mut old_guard = ManuallyDrop::new(self);
        let bytes = mem::take(&mut old_guard.slice);
        DisjointImmutGuard {
            slice: V::slice_from(bytes).unwrap(),
            phantom: old_guard.phantom,
            parent: old_guard.parent,
            borrow_id: old_guard.borrow_id,
        }
    }

    #[inline]
    fn cast<V: FromBytes>(self) -> DisjointImmutGuard<'a, T, V> {
        let mut old_guard = ManuallyDrop::new(self);
        let bytes = mem::take(&mut old_guard.slice);
        DisjointImmutGuard {
            slice: V::ref_from(bytes).unwrap(),
            phantom: old_guard.phantom,
            parent: old_guard.parent,
            borrow_id: old_guard.borrow_id,
        }
    }
}

impl<'a, T: ?Sized + AsMutPtr, V: ?Sized> Deref for DisjointImmutGuard<'a, T, V> {
    type Target = V;

    fn deref(&self) -> &Self::Target {
        self.slice
    }
}

// =============================================================================
// AsMutPtr trait (sealed — only implemented for types in this crate)
// =============================================================================

mod sealed {
    use alloc::boxed::Box;
    use alloc::vec::Vec;

    /// Sealing trait — prevents external implementations of [`AsMutPtr`](super::AsMutPtr).
    ///
    /// This is critical for soundness: an incorrect `AsMutPtr` impl could return
    /// a pointer to invalid memory, causing UB that the runtime checker cannot catch.
    /// By sealing the trait, we ensure only audited impls in this crate exist.
    pub trait Sealed {}

    impl<V: Copy> Sealed for Vec<V> {}
    impl<V: Copy, const N: usize> Sealed for [V; N] {}
    impl<V: Copy> Sealed for [V] {}
    impl<V: Copy> Sealed for Box<[V]> {}
}

/// Convert from a mutable pointer to a collection to a mutable pointer to the
/// underlying slice without ever creating a mutable reference to the slice.
///
/// This trait exists for the same reason as [`Vec::as_mut_ptr`] - we want to
/// create a mutable pointer to the underlying slice without ever creating a
/// mutable reference to the slice.
///
/// # Safety
///
/// This trait must not ever create a mutable reference to the underlying slice,
/// as it may be (partially) immutably borrowed concurrently.
///
/// # Sealed
///
/// This trait is sealed and cannot be implemented outside of this crate.
/// External types can use the [`ExternalAsMutPtr`] unsafe trait to opt in,
/// which requires `Copy` element types for data-race safety.
pub unsafe trait AsMutPtr: sealed::Sealed {
    type Target: Copy;

    /// Convert a mutable pointer to a collection to a mutable pointer to the
    /// underlying slice.
    ///
    /// # Safety
    ///
    /// This method may dereference `ptr` as an immutable reference, so this
    /// pointer must be safely dereferenceable.
    unsafe fn as_mut_slice(ptr: *mut Self) -> *mut [Self::Target] {
        // SAFETY: The safety precondition of this method requires that we can
        // immutably dereference `ptr`.
        let len = unsafe { (*ptr).len() };
        // SAFETY: Mutably dereferencing and calling `.as_mut_ptr()` does not
        // materialize a mutable reference to the underlying slice according to
        // its documentated behavior, so we can still allow concurrent immutable
        // references into that underlying slice.
        let data = unsafe { Self::as_mut_ptr(ptr) };
        ptr::slice_from_raw_parts_mut(data, len)
    }

    /// Convert a mutable pointer to a collection to a mutable pointer to the
    /// first element of the collection.
    ///
    /// # Safety
    ///
    /// This method may dereference `ptr` as an immutable reference, so this
    /// pointer must be safely dereferenceable.
    ///
    /// The returned pointer is only safe to dereference within the bounds of
    /// the underlying collection.
    unsafe fn as_mut_ptr(ptr: *mut Self) -> *mut Self::Target;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Opt-in trait for external types to participate in [`DisjointMut`].
///
/// Implement this trait for your container type so it can be used with
/// `DisjointMut<YourType>`. The `Target` type must be `Copy` to ensure
/// data races cannot cause memory safety issues beyond producing incorrect
/// values (no torn reads on non-`Copy` types).
///
/// # Safety
///
/// Implementors must uphold all of the following:
///
/// 1. **No mutable references to the container or its data.** The
///    `as_mut_ptr` implementation must not create `&mut Self` or
///    `&mut [Self::Target]`. Creating `&mut` causes a Stacked Borrows
///    retag that invalidates concurrent borrows on other threads.
///    Use only shared references (`&Self`) or raw pointer operations.
///
/// 2. **No shared references to element data.** Even `&[Self::Target]`
///    conflicts with `&mut [Self::Target]` guards under Stacked Borrows.
///    If you need the length, read it from container metadata (which lives
///    in a separate allocation from the elements), or override
///    [`AsMutPtr::as_mut_slice`] and use raw pointer metadata.
///
/// 3. **Valid pointer.** The returned `*mut Self::Target` must be valid
///    for reads and writes over `0..self.len()` elements.
///
/// 4. **Stable length.** `len()` must return a consistent value for the
///    lifetime of any outstanding borrow guard.
///
/// See the `Vec<V>` and `Box<[V]>` implementations in this crate for
/// reference patterns.
pub unsafe trait ExternalAsMutPtr {
    type Target: Copy;

    /// Returns a mutable pointer to the first element.
    ///
    /// # Safety
    ///
    /// `ptr` must be safely dereferenceable. The implementation must not
    /// create `&mut Self` or `&mut [Self::Target]` — only shared references
    /// to container metadata or raw pointer operations. See the trait-level
    /// safety docs for full requirements.
    unsafe fn as_mut_ptr(ptr: *mut Self) -> *mut Self::Target;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// Blanket seal for external types
impl<T: ExternalAsMutPtr> sealed::Sealed for T {}

// Blanket AsMutPtr for external types
unsafe impl<T: ExternalAsMutPtr> AsMutPtr for T {
    type Target = T::Target;

    unsafe fn as_mut_ptr(ptr: *mut Self) -> *mut Self::Target {
        unsafe { <T as ExternalAsMutPtr>::as_mut_ptr(ptr) }
    }

    fn len(&self) -> usize {
        <T as ExternalAsMutPtr>::len(self)
    }
}

// =============================================================================
// Core index/index_mut methods
// =============================================================================

impl<T: ?Sized + AsMutPtr> DisjointMut<T> {
    pub fn len(&self) -> usize {
        // Use as_mut_slice to get a fat *mut [T] pointer and read length from
        // the fat pointer metadata. This avoids creating &T which for some
        // container types (e.g. Box<[V]>) would create &[V] to the heap data,
        // conflicting with concurrent &mut [V] guards under Stacked Borrows.
        self.as_mut_slice().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a raw pointer to the underlying element data, bypassing the tracker.
    ///
    /// # Why this exists (instead of using guard `.as_ptr()`)
    ///
    /// FFI boundaries (assembly calls, C interop) need raw pointers. Creating a
    /// tracked guard for the entire buffer would be wrong — assembly code may only
    /// touch a subset, and pointer arithmetic happens on the callee side. This
    /// method provides the base pointer for such offset calculations.
    ///
    /// Similarly, some code needs pointer identity checks (e.g. `ptr == other_ptr`)
    /// without actually borrowing data.
    ///
    /// The pointer requires `unsafe` to dereference, so the caller accepts
    /// responsibility for disjointness — same as any raw pointer in Rust.
    pub fn as_mut_slice(&self) -> *mut [<T as AsMutPtr>::Target] {
        // SAFETY: The inner cell is safe to access immutably. We never create a
        // mutable reference to the inner value.
        unsafe { AsMutPtr::as_mut_slice(self.inner.get()) }
    }

    /// Returns a raw pointer to the first element. See [`Self::as_mut_slice`] for rationale.
    pub fn as_mut_ptr(&self) -> *mut <T as AsMutPtr>::Target {
        // SAFETY: The inner cell is safe to access immutably. We never create a
        // mutable reference to the inner value.
        unsafe { AsMutPtr::as_mut_ptr(self.inner.get()) }
    }

    pub fn get_mut(&mut self) -> &mut T {
        self.inner.get_mut()
    }

    /// Mutably borrow a slice or element.
    ///
    /// Validates that the requested range doesn't overlap with any outstanding
    /// borrow, then creates the `&mut` reference. Panics on overlap, OOB, or
    /// if the data structure has been poisoned by a prior panic.
    #[inline] // Inline to see bounds checks in order to potentially elide them.
    #[track_caller]
    pub fn index_mut<'a, I>(&'a self, index: I) -> DisjointMutGuard<'a, T, I::Output>
    where
        I: Into<Bounds> + Clone,
        I: DisjointMutIndex<[<T as AsMutPtr>::Target]>,
    {
        let bounds = index.clone().into();
        // Register the borrow BEFORE creating the reference.
        // This prevents a TOCTOU gap where two threads could both create
        // references to overlapping ranges before either registers.
        let borrow_id = match &self.tracker {
            Some(tracker) => tracker.add_mut(&bounds),
            None => checked::BorrowId::UNCHECKED,
        };
        let parent = self.tracker.as_ref().map(|_| self);
        // Scope guard: if get_mut panics (OOB), poison the data structure.
        // We don't try to clean up the leaked borrow — poisoning is stricter
        // and prevents all future access, following std::sync::Mutex semantics.
        let cleanup = BorrowCleanup { parent };
        // SAFETY: The borrow has been registered (or we're unchecked).
        // The indexed region is guaranteed disjoint from all other active borrows.
        let slice = unsafe { &mut *index.get_mut(self.as_mut_slice()) };
        // Success — disarm the cleanup guard.
        mem::forget(cleanup);
        DisjointMutGuard {
            slice,
            parent,
            borrow_id,
            phantom: PhantomData,
        }
    }

    /// Immutably borrow a slice or element.
    ///
    /// Validates that the requested range doesn't overlap with any outstanding
    /// mutable borrow, then creates the `&` reference. Panics on overlap, OOB,
    /// or if the data structure has been poisoned by a prior panic.
    #[inline] // Inline to see bounds checks in order to potentially elide them.
    #[track_caller]
    pub fn index<'a, I>(&'a self, index: I) -> DisjointImmutGuard<'a, T, I::Output>
    where
        I: Into<Bounds> + Clone,
        I: DisjointMutIndex<[<T as AsMutPtr>::Target]>,
    {
        let bounds = index.clone().into();
        let borrow_id = match &self.tracker {
            Some(tracker) => tracker.add_immut(&bounds),
            None => checked::BorrowId::UNCHECKED,
        };
        let parent = self.tracker.as_ref().map(|_| self);
        let cleanup = BorrowCleanup { parent };
        // SAFETY: The borrow has been registered (or we're unchecked).
        let slice = unsafe { &*index.get_mut(self.as_mut_slice()).cast_const() };
        mem::forget(cleanup);
        DisjointImmutGuard {
            slice,
            parent,
            borrow_id,
            phantom: PhantomData,
        }
    }
}

// =============================================================================
// Zerocopy cast methods (for u8 buffers → typed access)
// =============================================================================

#[cfg(feature = "zerocopy")]
impl<T: AsMutPtr<Target = u8>> DisjointMut<T> {
    /// Check that a casted slice has the expected length.
    #[inline]
    fn check_cast_slice_len<I, V>(&self, index: I, slice: &[V])
    where
        I: SliceBounds,
    {
        let range = index.to_range(self.len() / mem::size_of::<V>());
        let range_len = range.end - range.start;
        assert!(slice.len() == range_len);
    }

    /// Mutably borrow a slice of a convertible type.
    #[inline]
    #[track_caller]
    pub fn mut_slice_as<'a, I, V>(&'a self, index: I) -> DisjointMutGuard<'a, T, [V]>
    where
        I: SliceBounds,
        V: AsBytes + FromBytes,
    {
        let slice = self.index_mut(index.mul(mem::size_of::<V>())).cast_slice();
        self.check_cast_slice_len(index, &slice);
        slice
    }

    /// Mutably borrow an element of a convertible type.
    #[inline]
    #[track_caller]
    pub fn mut_element_as<'a, V>(&'a self, index: usize) -> DisjointMutGuard<'a, T, V>
    where
        V: AsBytes + FromBytes,
    {
        self.index_mut((index..index + 1).mul(mem::size_of::<V>()))
            .cast()
    }

    /// Immutably borrow a slice of a convertible type.
    #[inline]
    #[track_caller]
    pub fn slice_as<'a, I, V>(&'a self, index: I) -> DisjointImmutGuard<'a, T, [V]>
    where
        I: SliceBounds,
        V: FromBytes,
    {
        let slice = self.index(index.mul(mem::size_of::<V>())).cast_slice();
        self.check_cast_slice_len(index, &slice);
        slice
    }

    /// Immutably borrow an element of a convertible type.
    #[inline]
    #[track_caller]
    pub fn element_as<'a, V>(&'a self, index: usize) -> DisjointImmutGuard<'a, T, V>
    where
        V: FromBytes,
    {
        self.index((index..index + 1).mul(mem::size_of::<V>()))
            .cast()
    }
}

// =============================================================================
// DisjointMutIndex trait (stable SliceIndex equivalent)
// =============================================================================

/// This trait is a stable implementation of [`std::slice::SliceIndex`] to allow
/// for indexing into mutable slice raw pointers.
pub trait DisjointMutIndex<T: ?Sized> {
    type Output: ?Sized;

    /// Returns a mutable pointer to the output at this indexed location.
    ///
    /// # Safety
    ///
    /// `slice` must be a valid, dereferencable pointer.
    unsafe fn get_mut(self, slice: *mut T) -> *mut Self::Output;
}

// =============================================================================
// Range translation traits
// =============================================================================

pub trait TranslateRange {
    fn mul(&self, by: usize) -> Self;
}

impl TranslateRange for usize {
    fn mul(&self, by: usize) -> Self {
        *self * by
    }
}

impl TranslateRange for Range<usize> {
    fn mul(&self, by: usize) -> Self {
        self.start * by..self.end * by
    }
}

impl TranslateRange for RangeFrom<usize> {
    fn mul(&self, by: usize) -> Self {
        self.start * by..
    }
}

impl TranslateRange for RangeInclusive<usize> {
    fn mul(&self, by: usize) -> Self {
        *self.start() * by..=*self.end() * by
    }
}

impl TranslateRange for RangeTo<usize> {
    fn mul(&self, by: usize) -> Self {
        ..self.end * by
    }
}

impl TranslateRange for RangeToInclusive<usize> {
    fn mul(&self, by: usize) -> Self {
        ..=self.end * by
    }
}

impl TranslateRange for RangeFull {
    fn mul(&self, _by: usize) -> Self {
        *self
    }
}

impl TranslateRange for (RangeFrom<usize>, RangeTo<usize>) {
    fn mul(&self, by: usize) -> Self {
        (self.0.start * by.., ..self.1.end * by)
    }
}

// =============================================================================
// Bounds type
// =============================================================================

#[derive(Clone, Default, PartialEq, Eq)]
pub struct Bounds {
    /// A [`Range::end`]` == `[`usize::MAX`] is considered unbounded,
    /// as lengths need to be less than [`isize::MAX`] already.
    pub(crate) range: Range<usize>,
}

impl Display for Bounds {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let Range { start, end } = self.range;
        if start != 0 {
            write!(f, "{start}")?;
        }
        write!(f, "..")?;
        if end != usize::MAX {
            write!(f, "{end}")?;
        }
        Ok(())
    }
}

impl Debug for Bounds {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl Bounds {
    #[cfg(test)]
    fn is_empty(&self) -> bool {
        self.range.start >= self.range.end
    }

    #[cfg(test)]
    fn overlaps(&self, other: &Bounds) -> bool {
        // Empty ranges borrow zero bytes and never conflict.
        if self.is_empty() || other.is_empty() {
            return false;
        }
        let a = &self.range;
        let b = &other.range;
        a.start < b.end && b.start < a.end
    }
}

impl From<usize> for Bounds {
    fn from(index: usize) -> Self {
        Self {
            range: index..index.checked_add(1).expect("index overflow in Bounds"),
        }
    }
}

impl<T: SliceBounds> From<T> for Bounds {
    fn from(range: T) -> Self {
        Self {
            range: range.to_range(usize::MAX),
        }
    }
}

pub trait SliceBounds: TranslateRange + Clone {
    fn to_range(&self, len: usize) -> Range<usize>;
}

impl SliceBounds for Range<usize> {
    fn to_range(&self, _len: usize) -> Range<usize> {
        let Self { start, end } = *self;
        start..end
    }
}

impl SliceBounds for RangeFrom<usize> {
    fn to_range(&self, len: usize) -> Range<usize> {
        let Self { start } = *self;
        start..len
    }
}

impl SliceBounds for RangeInclusive<usize> {
    fn to_range(&self, _len: usize) -> Range<usize> {
        *self.start()..self.end().checked_add(1).expect("range end overflow")
    }
}

impl SliceBounds for RangeTo<usize> {
    fn to_range(&self, _len: usize) -> Range<usize> {
        let Self { end } = *self;
        0..end
    }
}

impl SliceBounds for RangeToInclusive<usize> {
    fn to_range(&self, _len: usize) -> Range<usize> {
        let Self { end } = *self;
        0..end.checked_add(1).expect("range end overflow")
    }
}

impl SliceBounds for RangeFull {
    fn to_range(&self, len: usize) -> Range<usize> {
        0..len
    }
}

/// A majority of slice ranges are of the form `[start..][..len]`.
/// This is easy to express with normal slices where we can do the slicing multiple times,
/// but with [`DisjointMut`], that's harder, so this adds support for
/// `.index((start.., ..len))` to achieve the same.
impl SliceBounds for (RangeFrom<usize>, RangeTo<usize>) {
    fn to_range(&self, _len: usize) -> Range<usize> {
        let (RangeFrom { start }, RangeTo { end: range_len }) = *self;
        start..start + range_len
    }
}

// =============================================================================
// DisjointMutIndex implementations
// =============================================================================

impl<T> DisjointMutIndex<[T]> for usize {
    type Output = <[T] as Index<usize>>::Output;

    #[inline]
    #[track_caller]
    unsafe fn get_mut(self, slice: *mut [T]) -> *mut Self::Output {
        let index = self;
        let len = slice.len();
        if index < len {
            // SAFETY: We have checked that `self` is less than the allocation
            // length therefore cannot overflow.
            unsafe { (slice as *mut T).add(index) }
        } else {
            #[inline(never)]
            #[track_caller]
            fn out_of_bounds(index: usize, len: usize) -> ! {
                panic!("index out of bounds: the len is {len} but the index is {index}")
            }
            out_of_bounds(index, len);
        }
    }
}

impl<T, I> DisjointMutIndex<[T]> for I
where
    I: SliceBounds,
{
    type Output = <[T] as Index<Range<usize>>>::Output;

    #[inline]
    #[track_caller]
    unsafe fn get_mut(self, slice: *mut [T]) -> *mut Self::Output {
        let len = slice.len();
        let Range { start, end } = self.to_range(len);
        if start <= end && end <= len {
            // SAFETY: We have checked bounds.
            let data = unsafe { (slice as *mut T).add(start) };
            ptr::slice_from_raw_parts_mut(data, end - start)
        } else {
            #[inline(never)]
            #[track_caller]
            fn out_of_bounds(start: usize, end: usize, len: usize) -> ! {
                if start > end {
                    panic!("slice index starts at {start} but ends at {end}");
                }
                if end > len {
                    panic!("range end index {end} out of range for slice of length {len}");
                }
                unreachable!();
            }
            out_of_bounds(start, end, len);
        }
    }
}

// =============================================================================
// Bounds tracking (single mutex, holds lock during reference creation)
// =============================================================================

mod checked {
    use super::*;
    use core::panic::Location;
    use core::sync::atomic::{AtomicBool, Ordering};

    /// Lightweight spinlock for borrow tracking.
    ///
    /// Uses a simple `AtomicBool::swap` for the lock — cheaper than
    /// `compare_exchange` on the uncontended fast path because `swap`
    /// is an unconditional store (no branch on old value). For the
    /// single-threaded case (rav1d threads=1), this never spins.
    struct TinyLock(AtomicBool);

    impl TinyLock {
        const fn new() -> Self {
            Self(AtomicBool::new(false))
        }

        #[inline(always)]
        fn lock(&self) -> TinyGuard<'_> {
            // Fast path: swap is cheaper than compare_exchange for uncontended locks.
            // On x86_64, `xchg` is simpler than `lock cmpxchg`.
            if self.0.swap(true, Ordering::Acquire) {
                // Contended — spin. This should essentially never happen in rav1d.
                self.lock_slow();
            }
            TinyGuard(&self.0)
        }

        #[cold]
        #[inline(never)]
        fn lock_slow(&self) {
            loop {
                // Spin-wait: read without writing to avoid cache line bouncing
                while self.0.load(Ordering::Relaxed) {
                    core::hint::spin_loop();
                }
                if !self.0.swap(true, Ordering::Acquire) {
                    return;
                }
            }
        }
    }

    struct TinyGuard<'a>(&'a AtomicBool);

    impl<'a> Drop for TinyGuard<'a> {
        #[inline(always)]
        fn drop(&mut self) {
            self.0.store(false, Ordering::Release);
        }
    }

    /// Maximum concurrent borrows per `DisjointMut` instance.
    /// 32 slots allows use of a u32 bitmask for O(1) free-slot finding.
    const MAX_BORROWS: usize = 32;

    /// A unique identifier for a borrow registration, encoding a slot index.
    #[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
    pub(super) struct BorrowId(u8);

    impl BorrowId {
        /// Sentinel value for unchecked guards (no tracking).
        pub const UNCHECKED: Self = Self(u8::MAX);
    }

    /// Fixed-capacity borrow record storage.
    /// Uses a bitmask for O(1) slot allocation and O(1) deallocation.
    struct BorrowSlots {
        // Parallel arrays — smaller than an array of structs for cache efficiency
        // during the overlap scan (we only need start/end/mutable, not the slot index).
        starts: [usize; MAX_BORROWS],
        ends: [usize; MAX_BORROWS],
        mutable: [bool; MAX_BORROWS],
        /// Bitmask of occupied slots. Bit `i` is set iff slot `i` is active.
        occupied: u32,
    }

    impl BorrowSlots {
        /// Sentinel slot index for empty-range borrows (no real slot allocated).
        const EMPTY_SLOT: u8 = MAX_BORROWS as u8 + 1;

        const fn new() -> Self {
            Self {
                starts: [0; MAX_BORROWS],
                ends: [0; MAX_BORROWS],
                mutable: [false; MAX_BORROWS],
                occupied: 0,
            }
        }

        /// Allocate a slot and return its index.
        /// Empty ranges (start >= end) are stored as slot `EMPTY_SLOT` without
        /// occupying a real slot, saving both alloc and overlap-scan cost.
        #[inline(always)]
        fn alloc(&mut self, start: usize, end: usize, is_mutable: bool) -> u8 {
            if start >= end {
                return Self::EMPTY_SLOT;
            }
            let free = self.occupied.trailing_ones() as usize;
            if free >= MAX_BORROWS {
                panic!("DisjointMut: too many concurrent borrows (max {MAX_BORROWS})");
            }
            self.starts[free] = start;
            self.ends[free] = end;
            self.mutable[free] = is_mutable;
            self.occupied |= 1 << free;
            free as u8
        }

        /// Free a slot by index. Empty-range borrows (EMPTY_SLOT) are no-ops.
        #[inline(always)]
        fn free(&mut self, slot: u8) {
            if slot == Self::EMPTY_SLOT {
                return;
            }
            debug_assert!(
                (self.occupied & (1 << slot)) != 0,
                "BorrowId slot {slot} not occupied"
            );
            self.occupied &= !(1u32 << slot);
        }

        /// Check if the range [start, end) overlaps any active borrow.
        /// Returns the first overlapping slot, or None.
        ///
        /// Empty ranges (start >= end) never overlap and never get allocated,
        /// so all active slots have valid non-empty ranges.
        #[inline(always)]
        fn find_overlap_any(&self, start: usize, end: usize) -> Option<u8> {
            if start >= end {
                return None;
            }
            let mut mask = self.occupied;
            while mask != 0 {
                let i = mask.trailing_zeros() as usize;
                if self.starts[i] < end && start < self.ends[i] {
                    return Some(i as u8);
                }
                mask &= mask - 1; // clear lowest set bit
            }
            None
        }

        /// Check if the range [start, end) overlaps any active MUTABLE borrow.
        /// Returns the first overlapping mutable slot, or None.
        #[inline(always)]
        fn find_overlap_mut(&self, start: usize, end: usize) -> Option<u8> {
            if start >= end {
                return None;
            }
            let mut mask = self.occupied;
            while mask != 0 {
                let i = mask.trailing_zeros() as usize;
                if self.mutable[i] && self.starts[i] < end && start < self.ends[i] {
                    return Some(i as u8);
                }
                mask &= mask - 1;
            }
            None
        }
    }

    /// All active borrows for a single `DisjointMut` instance.
    ///
    /// Uses a fixed-capacity slot array with bitmask for O(1) allocation
    /// and deallocation. Overlap checking scans only occupied slots.
    ///
    /// Like `std::sync::Mutex`, the tracker poisons the data structure when a
    /// thread panics while holding a mutable borrow guard. This prevents
    /// subsequent access to potentially corrupted data.
    pub(super) struct BorrowTracker {
        lock: TinyLock,
        slots: UnsafeCell<BorrowSlots>,
        poisoned: AtomicBool,
    }

    // SAFETY: BorrowSlots is only accessed while TinyLock is held.
    unsafe impl Send for BorrowTracker {}
    unsafe impl Sync for BorrowTracker {}

    impl Default for BorrowTracker {
        fn default() -> Self {
            Self::new()
        }
    }

    impl BorrowTracker {
        pub const fn new() -> Self {
            Self {
                lock: TinyLock::new(),
                slots: UnsafeCell::new(BorrowSlots::new()),
                poisoned: AtomicBool::new(false),
            }
        }

        /// Mark this tracker as poisoned. All future borrow attempts will panic.
        pub fn poison(&self) {
            self.poisoned.store(true, Ordering::Release);
        }

        /// Panic if the tracker has been poisoned.
        #[inline(always)]
        fn check_poisoned(&self) {
            if self.poisoned.load(Ordering::Acquire) {
                Self::poisoned_panic();
            }
        }

        #[cold]
        #[inline(never)]
        fn poisoned_panic() -> ! {
            panic!("DisjointMut poisoned: a thread panicked while holding a mutable borrow");
        }

        /// Report an overlap violation with diagnostic info.
        #[cold]
        #[inline(never)]
        #[track_caller]
        fn overlap_panic(
            new_start: usize,
            new_end: usize,
            new_mutable: bool,
            existing_start: usize,
            existing_end: usize,
            existing_mutable: bool,
        ) -> ! {
            let new_mut_str = if new_mutable { "&mut" } else { "   &" };
            let existing_mut_str = if existing_mutable { "&mut" } else { "   &" };
            let caller = Location::caller();
            #[cfg(feature = "std")]
            {
                let thread = std::thread::current().id();
                panic!(
                    "\toverlapping DisjointMut:\n current: {new_mut_str} _[{new_start}..{new_end}] \
                     on {thread:?} at {caller}\nexisting: {existing_mut_str} _[{existing_start}..{existing_end}]",
                );
            }
            #[cfg(not(feature = "std"))]
            panic!(
                "\toverlapping DisjointMut:\n current: {new_mut_str} _[{new_start}..{new_end}] \
                 at {caller}\nexisting: {existing_mut_str} _[{existing_start}..{existing_end}]",
            );
        }

        /// Register a mutable borrow. Checks against ALL existing borrows.
        #[inline]
        #[track_caller]
        pub fn add_mut(&self, bounds: &Bounds) -> BorrowId {
            let start = bounds.range.start;
            let end = bounds.range.end;
            // Empty ranges can never overlap and don't need a slot.
            if start >= end {
                return BorrowId(BorrowSlots::EMPTY_SLOT);
            }
            self.check_poisoned();
            let _guard = self.lock.lock();
            // SAFETY: TinyLock is held, so we have exclusive access to slots.
            let slots = unsafe { &mut *self.slots.get() };
            if let Some(i) = slots.find_overlap_any(start, end) {
                Self::overlap_panic(
                    start,
                    end,
                    true,
                    slots.starts[i as usize],
                    slots.ends[i as usize],
                    slots.mutable[i as usize],
                );
            }
            BorrowId(slots.alloc(start, end, true))
        }

        /// Register an immutable borrow. Only checks against mutable borrows.
        #[inline]
        #[track_caller]
        pub fn add_immut(&self, bounds: &Bounds) -> BorrowId {
            let start = bounds.range.start;
            let end = bounds.range.end;
            // Empty ranges can never overlap and don't need a slot.
            if start >= end {
                return BorrowId(BorrowSlots::EMPTY_SLOT);
            }
            self.check_poisoned();
            let _guard = self.lock.lock();
            // SAFETY: TinyLock is held, so we have exclusive access to slots.
            let slots = unsafe { &mut *self.slots.get() };
            if let Some(i) = slots.find_overlap_mut(start, end) {
                Self::overlap_panic(
                    start,
                    end,
                    false,
                    slots.starts[i as usize],
                    slots.ends[i as usize],
                    slots.mutable[i as usize],
                );
            }
            BorrowId(slots.alloc(start, end, false))
        }

        /// Remove a borrow by slot index. O(1).
        pub fn remove(&self, id: BorrowId) {
            let _guard = self.lock.lock();
            // SAFETY: TinyLock is held, so we have exclusive access to slots.
            let slots = unsafe { &mut *self.slots.get() };
            slots.free(id.0);
        }
    }
}

// =============================================================================
// Guard Drop impls — deregister borrow on drop
// =============================================================================

impl<'a, T: ?Sized + AsMutPtr, V: ?Sized> Drop for DisjointMutGuard<'a, T, V> {
    fn drop(&mut self) {
        if let Some(parent) = self.parent {
            let tracker = parent.tracker.as_ref().unwrap();
            // If the thread is panicking while we hold a mutable guard,
            // the data may be partially written / inconsistent.
            // Poison the data structure so all future borrows fail.
            #[cfg(feature = "std")]
            if std::thread::panicking() {
                tracker.poison();
            }
            tracker.remove(self.borrow_id);
        }
    }
}

impl<'a, T: ?Sized + AsMutPtr, V: ?Sized> Drop for DisjointImmutGuard<'a, T, V> {
    fn drop(&mut self) {
        if let Some(parent) = self.parent {
            parent.tracker.as_ref().unwrap().remove(self.borrow_id);
        }
    }
}

// =============================================================================
// Generic convenience methods via traits (so external types can opt in)
// =============================================================================

/// Trait for types that support `resize(len, value)`. Implement this for your
/// container type so that `DisjointMut<YourType>` gains a `.resize()` method.
pub trait Resizable {
    type Value;
    fn resize(&mut self, new_len: usize, value: Self::Value);
}

impl<V: Clone> Resizable for Vec<V> {
    type Value = V;
    fn resize(&mut self, new_len: usize, value: V) {
        Vec::resize(self, new_len, value)
    }
}

impl<T: AsMutPtr + Resizable> DisjointMut<T> {
    pub fn resize(&mut self, new_len: usize, value: T::Value) {
        self.inner.get_mut().resize(new_len, value)
    }
}

/// Trait for types that support `clear()`.
pub trait Clearable {
    fn clear(&mut self);
}

impl<V> Clearable for Vec<V> {
    fn clear(&mut self) {
        Vec::clear(self)
    }
}

impl<T: AsMutPtr + Clearable> DisjointMut<T> {
    pub fn clear(&mut self) {
        self.inner.get_mut().clear()
    }
}

/// Trait for types that support `resize_with(len, f)`.
pub trait ResizableWith {
    type Item;
    fn resize_with<F: FnMut() -> Self::Item>(&mut self, new_len: usize, f: F);
}

impl<V> ResizableWith for Vec<V> {
    type Item = V;
    fn resize_with<F: FnMut() -> V>(&mut self, new_len: usize, f: F) {
        Vec::resize_with(self, new_len, f)
    }
}

impl<T: AsMutPtr + ResizableWith> DisjointMut<T> {
    pub fn resize_with<F>(&mut self, new_len: usize, f: F)
    where
        F: FnMut() -> T::Item,
        T: ResizableWith,
    {
        self.inner.get_mut().resize_with(new_len, f)
    }
}

// =============================================================================
// AsMutPtr implementations for standard types
// =============================================================================

/// SAFETY: We only create `&Vec<V>` (shared reference), never `&mut Vec<V>`.
/// This is critical for Stacked Borrows: `&mut Vec` creates a retag-write
/// (Unique) on the Vec struct allocation, which conflicts with concurrent
/// `&Vec` reads of `len`/`as_ptr` from other threads. Using only shared
/// references avoids this data race.
///
/// The returned `*mut V` pointer retains write provenance from the original
/// allocator, not from the reference we read it through. The `UnsafeCell`
/// wrapper in `DisjointMut` provides the permission for concurrent writes
/// to the heap data.
unsafe impl<V: Copy> AsMutPtr for Vec<V> {
    type Target = V;

    unsafe fn as_mut_slice(ptr: *mut Self) -> *mut [Self::Target] {
        // SAFETY: Only creates &Vec (SharedReadOnly). The data pointer value
        // stored inside Vec retains its original allocator provenance.
        let vec_ref = unsafe { &*ptr };
        ptr::slice_from_raw_parts_mut(vec_ref.as_ptr().cast_mut(), vec_ref.len())
    }

    unsafe fn as_mut_ptr(ptr: *mut Self) -> *mut Self::Target {
        // SAFETY: Only creates &Vec (SharedReadOnly), not &mut Vec.
        unsafe { (*ptr).as_ptr().cast_mut() }
    }

    fn len(&self) -> usize {
        self.len()
    }
}

/// SAFETY: Pure pointer operations only — no references created.
/// The array data is inline (same allocation as the UnsafeCell), so we
/// must not create `&[V; N]` or `&[V]` which would conflict with guards.
/// Length is the compile-time constant `N`.
unsafe impl<V: Copy, const N: usize> AsMutPtr for [V; N] {
    type Target = V;

    unsafe fn as_mut_slice(ptr: *mut Self) -> *mut [Self::Target] {
        ptr::slice_from_raw_parts_mut(ptr.cast::<V>(), N)
    }

    unsafe fn as_mut_ptr(ptr: *mut Self) -> *mut V {
        ptr.cast()
    }

    fn len(&self) -> usize {
        N
    }
}

/// SAFETY: Pure pointer operations only — no references created.
/// Like arrays, the slice data IS the allocation, so `&[V]` would conflict.
unsafe impl<V: Copy> AsMutPtr for [V] {
    type Target = V;

    unsafe fn as_mut_slice(ptr: *mut Self) -> *mut [Self::Target] {
        // *mut [V] is already the right type — just pass it through.
        ptr
    }

    unsafe fn as_mut_ptr(ptr: *mut Self) -> *mut Self::Target {
        ptr.cast()
    }

    fn len(&self) -> usize {
        self.len()
    }
}

/// SAFETY: Uses `addr_of_mut!` to obtain `*mut [V]` through the raw pointer
/// chain `*mut Box<[V]>` → `*mut [V]` without creating `&[V]` or `&mut [V]`.
/// Box deref through a raw-pointer-derived place is a compiler built-in
/// operation that does not create intermediate references.
///
/// This is critical for Stacked Borrows: creating `&[V]` to the heap would
/// conflict with concurrent `&mut [V]` guards from other threads.
unsafe impl<V: Copy> AsMutPtr for Box<[V]> {
    type Target = V;

    unsafe fn as_mut_slice(ptr: *mut Self) -> *mut [Self::Target] {
        // SAFETY: addr_of_mut! through raw pointer chain — no &[V] created.
        // Box deref from a raw-pointer place is a compiler intrinsic that
        // follows the Box's internal pointer without creating references.
        unsafe { addr_of_mut!(**ptr) }
    }

    unsafe fn as_mut_ptr(ptr: *mut Self) -> *mut Self::Target {
        // SAFETY: Same raw pointer chain as as_mut_slice, then cast to thin pointer.
        unsafe { addr_of_mut!(**ptr) }.cast()
    }

    fn len(&self) -> usize {
        (**self).len()
    }
}

// =============================================================================
// DisjointMutSlice and DisjointMutArcSlice
// =============================================================================

/// `DisjointMut` always has tracking fields, so we use `Box<[T]>` as the
/// backing store for slice-based DisjointMut instances.
pub type DisjointMutSlice<T> = DisjointMut<Box<[T]>>;

/// A wrapper around an [`Arc`] of a [`DisjointMut`] slice.
/// An `Arc<[_]>` can be created, but adding a [`DisjointMut`] in between
/// requires boxing since `DisjointMut` has tracking fields.
#[derive(Clone)]
pub struct DisjointMutArcSlice<T: Copy> {
    pub inner: Arc<DisjointMutSlice<T>>,
}

impl<T: Copy> FromIterator<T> for DisjointMutArcSlice<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let box_slice = iter.into_iter().collect::<Box<[_]>>();
        Self {
            inner: Arc::new(DisjointMut::new(box_slice)),
        }
    }
}

impl<T: Copy> Default for DisjointMutArcSlice<T> {
    fn default() -> Self {
        [].into_iter().collect()
    }
}

// =============================================================================
// StridedBuf: raw buffer for use in DisjointMut without external unsafe
// =============================================================================

#[cfg(feature = "pic-buf")]
mod pic_buf {
    use super::*;

    /// An owned byte buffer for use in [`DisjointMut`].
    ///
    /// Stores a `Vec<u8>` with an alignment offset and usable length.
    /// The `Vec` owns the heap allocation, so `PicBuf` is automatically
    /// `Send + Sync` (no raw pointers stored, no manual impls needed).
    ///
    /// For picture data: created via [`from_vec_aligned`](Self::from_vec_aligned)
    /// with pool-allocated Vecs.
    ///
    /// For scratch buffers: created via [`from_slice_copy`](Self::from_slice_copy)
    /// which copies the data into an owned Vec.
    #[derive(Default)]
    pub struct PicBuf {
        buf: Vec<u8>,
        /// Byte offset from start of Vec to first usable byte (for alignment).
        align_offset: usize,
        /// Number of usable bytes starting from `align_offset`.
        usable_len: usize,
    }

    // No manual Send/Sync needed — Vec<u8> and usize are both Send + Sync.

    impl PicBuf {
        /// Create an owned buffer from a Vec with alignment offset.
        ///
        /// Takes ownership of the Vec. Computes the alignment offset from the
        /// Vec's data pointer so that the usable region starts at the next
        /// `alignment`-byte boundary.
        ///
        /// # Panics
        ///
        /// Panics if `align_offset + usable_len > vec.len()`.
        pub fn from_vec_aligned(vec: Vec<u8>, alignment: usize, usable_len: usize) -> Self {
            if usable_len == 0 {
                return Self {
                    buf: vec,
                    align_offset: 0,
                    usable_len: 0,
                };
            }
            let align_offset = vec.as_ptr().align_offset(alignment);
            assert!(
                align_offset + usable_len <= vec.len(),
                "PicBuf: aligned region ({} + {}) exceeds Vec length ({})",
                align_offset,
                usable_len,
                vec.len()
            );
            Self {
                buf: vec,
                align_offset,
                usable_len,
            }
        }

        /// Create an owned buffer by copying a byte slice.
        ///
        /// Used for scratch buffers: the data is copied into an owned Vec,
        /// so no raw pointers or lifetime concerns.
        pub fn from_slice_copy(data: &[u8]) -> Self {
            let vec = data.to_vec();
            let len = vec.len();
            Self {
                buf: vec,
                align_offset: 0,
                usable_len: len,
            }
        }

        /// Access the usable byte region as a slice.
        ///
        /// Used to copy data back from a scratch-dst component after writing.
        pub fn as_usable_bytes(&self) -> &[u8] {
            &self.buf[self.align_offset..self.align_offset + self.usable_len]
        }

        /// Take the owned Vec out of this buffer.
        ///
        /// After this call, the buffer is left in a default (empty) state.
        /// Used by picture data to return buffers to a memory pool on drop.
        pub fn take_buf(&mut self) -> Option<Vec<u8>> {
            if self.buf.is_empty() && self.usable_len == 0 {
                None
            } else {
                self.usable_len = 0;
                self.align_offset = 0;
                Some(core::mem::take(&mut self.buf))
            }
        }
    }

    /// SAFETY: Pointer is derived from the Vec's heap allocation through a shared
    /// reference (`&PicBuf` → `&Vec<u8>` → `as_ptr()`). This avoids creating
    /// `&mut Vec` (which would cause a Unique retag under Stacked Borrows).
    /// The heap data is a separate allocation from the Vec header, so the
    /// SharedReadOnly tag on the PicBuf struct does not cover the heap bytes.
    unsafe impl AsMutPtr for PicBuf {
        type Target = u8;

        unsafe fn as_mut_ptr(ptr: *mut Self) -> *mut u8 {
            // SAFETY: SharedReadOnly reference to PicBuf. Reading Vec::as_ptr()
            // only touches the Vec header (ptr, len, cap), not the heap data.
            let this = unsafe { &*ptr };
            this.buf.as_ptr().wrapping_add(this.align_offset).cast_mut()
        }

        unsafe fn as_mut_slice(ptr: *mut Self) -> *mut [u8] {
            let this = unsafe { &*ptr };
            let data_ptr = this.buf.as_ptr().wrapping_add(this.align_offset).cast_mut();
            core::ptr::slice_from_raw_parts_mut(data_ptr, this.usable_len)
        }

        fn len(&self) -> usize {
            self.usable_len
        }
    }

    impl sealed::Sealed for PicBuf {}
}

#[cfg(feature = "pic-buf")]
pub use pic_buf::PicBuf;

// =============================================================================
// Tests
// =============================================================================

#[test]
fn test_overlapping_immut() {
    let mut v: DisjointMut<Vec<u8>> = Default::default();
    v.resize(10, 0u8);

    let guard1 = v.index(0..5);
    let guard2 = v.index(2..);

    assert_eq!(guard1[2], guard2[0]);
}

#[test]
#[should_panic]
#[cfg(not(feature = "unchecked"))]
fn test_overlapping_mut() {
    let mut v: DisjointMut<Vec<u8>> = Default::default();
    v.resize(10, 0u8);

    let guard1 = v.index(0..5);
    let mut guard2 = v.index_mut(2..);

    guard2[0] = 42;
    assert_eq!(guard1[2], 42);
}

#[test]
fn test_range_overlap() {
    fn overlaps(a: impl Into<Bounds>, b: impl Into<Bounds>) -> bool {
        let a = a.into();
        let b = b.into();
        a.overlaps(&b)
    }

    // Range overlap.
    assert!(overlaps(5..7, 4..10));
    assert!(overlaps(4..10, 5..7));

    // RangeFrom overlap.
    assert!(overlaps(5.., 4..10));
    assert!(overlaps(4..10, 5..));

    // RangeTo overlap.
    assert!(overlaps(..7, 4..10));
    assert!(overlaps(4..10, ..7));

    // RangeInclusive overlap.
    assert!(overlaps(5..=7, 7..10));
    assert!(overlaps(7..10, 5..=7));

    // RangeToInclusive overlap.
    assert!(overlaps(..=7, 7..10));
    assert!(overlaps(7..10, ..=7));

    // Range no overlap.
    assert!(!overlaps(5..7, 10..20));
    assert!(!overlaps(10..20, 5..7));

    // RangeFrom no overlap.
    assert!(!overlaps(15.., 4..10));
    assert!(!overlaps(4..10, 15..));

    // RangeTo no overlap.
    assert!(!overlaps(..7, 10..20));
    assert!(!overlaps(10..20, ..7));

    // RangeInclusive no overlap.
    assert!(!overlaps(5..=7, 8..10));
    assert!(!overlaps(8..10, 5..=7));

    // RangeToInclusive no overlap.
    assert!(!overlaps(..=7, 8..10));
    assert!(!overlaps(8..10, ..=7));
}

// NOTE: Tests for aligned/aligned-vec integration are in align.rs
