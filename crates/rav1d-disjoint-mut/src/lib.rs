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
use zerocopy::AsBytes;
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
            tracker: Some(checked::BorrowTracker::new()),
        }
    }

    /// Create a `DisjointMut` without runtime overlap checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that all borrows obtained from this instance
    /// are truly disjoint: no mutable borrow may overlap with any other borrow.
    /// Violating this causes undefined behavior (aliasing `&mut`).
    #[cfg(feature = "unchecked")]
    pub const unsafe fn dangerously_unchecked(value: T) -> Self {
        Self {
            inner: UnsafeCell::new(value),
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
    range: Range<usize>,
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
    fn is_empty(&self) -> bool {
        self.range.start >= self.range.end
    }

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
    use alloc::vec::Vec;
    use core::panic::Location;
    use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    #[cfg(feature = "std")]
    use parking_lot::Mutex;
    #[cfg(not(feature = "std"))]
    use spin::Mutex;

    /// Monotonic ID generator for borrow records.
    static NEXT_ID: AtomicU64 = AtomicU64::new(0);

    /// A unique identifier for a borrow registration.
    #[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
    pub(super) struct BorrowId(u64);

    impl BorrowId {
        fn next() -> Self {
            Self(NEXT_ID.fetch_add(1, Ordering::Relaxed))
        }

        /// Sentinel value for unchecked guards.
        pub const UNCHECKED: Self = Self(u64::MAX);
    }

    struct BorrowRecord {
        id: BorrowId,
        bounds: Bounds,
        mutable: bool,
        location: &'static Location<'static>,
        #[cfg(feature = "std")]
        thread: std::thread::ThreadId,
    }

    impl Debug for BorrowRecord {
        fn fmt(&self, f: &mut Formatter) -> fmt::Result {
            let mut s = f.debug_struct("BorrowRecord");
            s.field("id", &self.id)
                .field("bounds", &self.bounds)
                .field("mutable", &self.mutable)
                .field("location", &self.location);
            #[cfg(feature = "std")]
            s.field("thread", &self.thread);
            s.finish()
        }
    }

    impl Display for BorrowRecord {
        fn fmt(&self, f: &mut Formatter) -> fmt::Result {
            let Self {
                id: _,
                bounds,
                mutable,
                location,
                #[cfg(feature = "std")]
                thread,
            } = self;
            let mutable = if *mutable { "&mut" } else { "   &" };
            #[cfg(feature = "std")]
            return write!(f, "{mutable} _[{bounds}] on {thread:?} at {location}");
            #[cfg(not(feature = "std"))]
            write!(f, "{mutable} _[{bounds}] at {location}")
        }
    }

    /// All active borrows for a single `DisjointMut` instance.
    ///
    /// Like `std::sync::Mutex`, the tracker poisons the data structure when a
    /// thread panics while holding a mutable borrow guard. This prevents
    /// subsequent access to potentially corrupted data.
    pub(super) struct BorrowTracker {
        borrows: Mutex<Vec<BorrowRecord>>,
        poisoned: AtomicBool,
    }

    impl Default for BorrowTracker {
        fn default() -> Self {
            Self::new()
        }
    }

    impl BorrowTracker {
        pub const fn new() -> Self {
            Self {
                borrows: Mutex::new(Vec::new()),
                poisoned: AtomicBool::new(false),
            }
        }

        /// Mark this tracker as poisoned. All future borrow attempts will panic.
        pub fn poison(&self) {
            self.poisoned.store(true, Ordering::Release);
        }

        /// Panic if the tracker has been poisoned.
        fn check_poisoned(&self) {
            if self.poisoned.load(Ordering::Acquire) {
                panic!("DisjointMut poisoned: a thread panicked while holding a mutable borrow");
            }
        }

        /// Register a mutable borrow. Must be called while holding the lock
        /// (i.e., before creating the `&mut` reference).
        ///
        /// Returns the BorrowId for later removal.
        #[track_caller]
        pub fn add_mut(&self, bounds: &Bounds) -> BorrowId {
            self.check_poisoned();
            let id = BorrowId::next();
            let record = BorrowRecord {
                id,
                bounds: bounds.clone(),
                mutable: true,
                location: Location::caller(),
                #[cfg(feature = "std")]
                thread: std::thread::current().id(),
            };
            let mut borrows = self.borrows.lock();
            // Check against ALL existing borrows (both mut and immut)
            for existing in borrows.iter() {
                if bounds.overlaps(&existing.bounds) {
                    panic!("\toverlapping DisjointMut:\n current: {record}\nexisting: {existing}");
                }
            }
            borrows.push(record);
            id
        }

        /// Register an immutable borrow.
        #[track_caller]
        pub fn add_immut(&self, bounds: &Bounds) -> BorrowId {
            self.check_poisoned();
            let id = BorrowId::next();
            let record = BorrowRecord {
                id,
                bounds: bounds.clone(),
                mutable: false,
                location: Location::caller(),
                #[cfg(feature = "std")]
                thread: std::thread::current().id(),
            };
            let mut borrows = self.borrows.lock();
            // Only check against mutable borrows
            for existing in borrows.iter() {
                if existing.mutable && bounds.overlaps(&existing.bounds) {
                    panic!("\toverlapping DisjointMut:\n current: {record}\nexisting: {existing}");
                }
            }
            borrows.push(record);
            id
        }

        /// Remove a borrow by its unique ID.
        pub fn remove(&self, id: BorrowId) {
            let mut borrows = self.borrows.lock();
            let idx = borrows
                .iter()
                .position(|r| r.id == id)
                .expect("BorrowId not found in active borrows");
            borrows.remove(idx);
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

/// A raw mutable byte buffer for use in [`DisjointMut`].
///
/// This type stores a pointer and length, with constructors that are safe
/// to call from external crates. All unsafe is confined to the [`AsMutPtr`]
/// implementation within this crate.
///
/// Primary use case: picture data in rav1d-safe, where the buffer may point
/// into an owned `Vec<u8>` (with alignment offset) or a borrowed `&mut [u8]`.
/// Using `StridedBuf` inside `DisjointMut` avoids needing unsafe
/// `ExternalAsMutPtr` implementations in the main crate.
pub struct StridedBuf {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: StridedBuf is only accessed through DisjointMut's borrow tracking,
// which prevents data races. The contained pointer refers to data that is
// Send+Sync (Vec<u8> or &mut [u8] from a Send+Sync context).
unsafe impl Send for StridedBuf {}
unsafe impl Sync for StridedBuf {}

impl StridedBuf {
    /// Create from a mutable Vec with alignment offset.
    ///
    /// Computes the alignment offset from the Vec's data pointer, then stores
    /// a pointer to the aligned position with the given usable length.
    ///
    /// # Panics
    ///
    /// Panics if `align_offset + usable_len > vec.len()`.
    pub fn from_vec_aligned(vec: &mut Vec<u8>, alignment: usize, usable_len: usize) -> Self {
        let align_offset = vec.as_ptr().align_offset(alignment);
        assert!(
            align_offset + usable_len <= vec.len(),
            "StridedBuf: aligned region ({} + {}) exceeds Vec length ({})",
            align_offset,
            usable_len,
            vec.len()
        );
        Self {
            ptr: vec.as_mut_ptr().wrapping_add(align_offset),
            len: usable_len,
        }
    }

    /// Create from a mutable byte slice.
    pub fn from_byte_slice(slice: &mut [u8]) -> Self {
        Self {
            ptr: slice.as_mut_ptr(),
            len: slice.len(),
        }
    }

    /// Create a dangling buffer with the given alignment and zero length.
    ///
    /// Useful for default/empty states. The pointer is dangling but aligned.
    pub fn dangling_aligned(alignment: usize) -> Self {
        // Use the alignment value itself as the pointer address — it's always
        // a valid aligned non-zero value for a zero-length buffer.
        Self {
            ptr: alignment as *mut u8,
            len: 0,
        }
    }
}

impl Default for StridedBuf {
    fn default() -> Self {
        Self {
            ptr: core::ptr::null_mut(),
            len: 0,
        }
    }
}

/// SAFETY: Pure pointer operations only — the stored pointer is returned directly.
/// `len` returns the stored length. No references are created to the data.
unsafe impl AsMutPtr for StridedBuf {
    type Target = u8;

    unsafe fn as_mut_ptr(ptr: *mut Self) -> *mut u8 {
        unsafe { (*ptr).ptr }
    }

    unsafe fn as_mut_slice(ptr: *mut Self) -> *mut [u8] {
        unsafe {
            let this = &*ptr;
            core::ptr::slice_from_raw_parts_mut(this.ptr, this.len)
        }
    }

    fn len(&self) -> usize {
        self.len
    }
}

impl sealed::Sealed for StridedBuf {}

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
