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
//! [`DisjointMut::dangerously_unchecked()`] `unsafe` constructor skips runtime
//! tracking. The `unsafe` boundary ensures that opting out of tracking is an
//! explicit, auditable decision — not something that can happen via feature
//! unification or accident.
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
// `std` is a FEATURE of the library and a REQUIREMENT of its unit tests: they
// spawn threads and `catch_unwind` to prove the tracker catches overlaps, and
// neither exists in `core`. Without the `cfg(test)` arm,
// `cargo test -p rav1d-disjoint-mut --no-default-features` fails with 28
// `cannot find module or crate std` errors in `tracker_shard`'s test modules —
// which is what the three `Test (…, --no-default-features)` legs of
// `disjoint-mut CI` have been red on (`main` @ ee07b00 included).
#[cfg(any(feature = "std", test))]
extern crate std;

#[cfg(feature = "aligned")]
pub mod align;

/// THROWAWAY contention probe (feature `__probe_count`). Not public API.
#[cfg(feature = "__probe_count")]
pub mod probe;

/// THROWAWAY per-call-site borrow counter (feature `__probe_sites`). Not public API.
#[cfg(feature = "__probe_sites")]
pub mod site_probe;

/// THROWAWAY guard-extent / footprint / concurrency map (feature
/// `__probe_bounds`). Not public API.
#[cfg(feature = "__probe_bounds")]
pub mod bounds_probe;

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
use core::ptr::NonNull;
use core::ptr::addr_of_mut;
#[cfg(feature = "zerocopy")]
use zerocopy::FromBytes;
#[cfg(feature = "zerocopy")]
use zerocopy::Immutable;
#[cfg(feature = "zerocopy")]
use zerocopy::IntoBytes;
#[cfg(feature = "zerocopy")]
use zerocopy::KnownLayout;

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
/// For audited hot paths, use
/// [`DisjointMut::dangerously_unchecked`] to skip tracking.
pub struct DisjointMut<T: ?Sized + AsMutPtr> {
    /// Boxed so that `DisjointMut` stays pointer-sized regardless of how many
    /// shards the tracker carries: `Rav1dTaskContext` embeds ~20 of these and
    /// has a 48 KiB stack-weight gate.
    tracker: Option<Box<checked::BorrowTracker>>,

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

    /// Returns a raw pointer to the inner container, bypassing the borrow tracker.
    ///
    /// **Prefer `as_mut_ptr()` or `as_mut_slice()` for element access.** This
    /// method exists only for reading container metadata (e.g. stride) that
    /// lives outside the element data. Writing through this pointer or creating
    /// `&mut T` from it can violate the tracker's guarantees.
    ///
    /// # Safety
    ///
    /// The returned ptr has the safety requirements of [`UnsafeCell::get`].
    /// In particular, the ptr returned by [`AsMutPtr::as_mut_ptr`] may be in use.
    #[doc(hidden)]
    pub const fn inner(&self) -> *mut T {
        self.inner.get()
    }
}

impl<T: AsMutPtr> DisjointMut<T> {
    /// Creates a new `DisjointMut` with runtime borrow tracking enabled.
    ///
    /// Every `.index()` and `.index_mut()` call will validate that the
    /// requested range doesn't overlap with any outstanding borrow.
    ///
    /// Not `const`: the tracker sizes its shard array from the container's
    /// length, so that a picture plane gets many independently locked shards
    /// while a 32-byte scratch buffer gets one cache line. If the container is
    /// later grown with [`Self::resize`], the shard array is re-sized with it.
    pub fn new(value: T) -> Self {
        let len = AsMutPtr::len(&value);
        Self {
            #[cfg(not(feature = "__probe_untracked"))]
            tracker: Some(Box::new(checked::BorrowTracker::new(len))),
            #[cfg(feature = "__probe_untracked")]
            tracker: {
                let _ = len;
                None
            },
            inner: UnsafeCell::new(value),
        }
    }

    /// Creates a new `DisjointMut` **without** runtime borrow tracking.
    ///
    /// This skips all overlap checking — `.index_mut()` will create `&mut`
    /// references without verifying that they don't alias. This is faster
    /// but the caller must manually ensure that all borrows are disjoint.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that all borrows through this instance
    /// are non-overlapping. Overlapping mutable borrows cause undefined
    /// behavior (aliasing `&mut` references). Verify correctness by
    /// running the full test suite with a tracked (`new()`) instance first.
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

/// A live mutable borrow of one region.
///
/// # Why the region is held as a POINTER and not as `&'a mut V`
///
/// A guard is a value, so safe code can move it: `drop(g)`, `f(g)`,
/// `ManuallyDrop::new(g)`, returning it. Rust's aliasing model gives every
/// reference passed **by value into a call** a *protector* — for the whole of
/// that call the reference must stay valid, and nothing may invalidate it.
///
/// The guard's `Drop` runs at the end of exactly such a call, and it retires
/// the tracker record. Retiring is what authorises **another thread** to take
/// the same region and create its own `&mut` over those bytes — which
/// invalidates ours while the protector is still in force. That is UB, and it
/// is reachable from safe code: `drop(g)` is all it takes.
///
/// A raw pointer has no protector, so moving the guard by value carries no
/// claim over the buffer. The reference is materialised in [`Deref`] /
/// [`DerefMut`], where borrowck bounds it to a region in which the guard cannot
/// be dropped.
///
/// This is precisely why `core::cell::RefMut` holds a `NonNull<T>`; its comment
/// reads *"we use a pointer instead of `&'b mut T` to avoid `noalias`
/// violations, because a `RefMut` argument doesn't hold exclusivity for its
/// whole scope, only until it drops."* `DisjointMut`'s cross-thread release
/// makes the same statement sharper: for us it is not only `noalias`, another
/// thread is genuinely retagging the bytes.
///
/// Found 2026-08-09 by `tests/narrow_release.rs` under Miri; it reproduces
/// under **both** Stacked Borrows and Tree Borrows. Gated by
/// `tests/guard_move_release.rs`.
pub struct DisjointMutGuard<'a, T: ?Sized + AsMutPtr, V: ?Sized> {
    /// Raw by design — see the type-level docs. Always dereferenceable for the
    /// guard's whole life: the registration that made it exclusive is retired
    /// only in `Drop`.
    ///
    /// `NonNull`, not `*mut V`: a reference field carried `nonnull` (and the
    /// niche) into every load, and a bare raw pointer drops both. Measured, so
    /// not a guess — see `benchmarks/miri_guard_move_2026-08-09.meta` §7.
    slice: NonNull<V>,

    /// Carries everything the `&'a mut V` field used to supply except the
    /// reference itself: the lifetime, invariance in `V`, and the auto-trait
    /// bounds (`Send` iff `V: Send`, `Sync` iff `V: Sync`), which a bare
    /// `*mut V` would otherwise drop on the floor.
    phantom: PhantomData<(&'a mut V, &'a DisjointMut<T>)>,

    /// Reference to parent for borrow removal on drop.
    /// `None` when parent was created with `dangerously_unchecked`.
    parent: Option<&'a DisjointMut<T>>,
    /// Unique ID for this borrow registration.
    borrow_id: checked::BorrowId,
    /// THROWAWAY (`__probe_bounds`): handle to this acquisition's live record.
    #[cfg(feature = "__probe_bounds")]
    pub(crate) probe: bounds_probe::Ticket,
}

/// The zerocopy slice cast's failure path, out of line.
///
/// `<[V]>::mut_from_bytes` reports *which* invariant failed in a `CastError`
/// that is several words wide, and `.unwrap()` requires that value to be
/// materialisable in the calling frame. In the release build of
/// `mut_slice_as::<_, u16>` that alone buys the SUCCESS path a 112-byte stack
/// frame plus ten callee-saved spill/reload pairs it never touches — verified
/// in the disassembly, where the only writes into that frame are the stores
/// that build the `CastError` on the cold branch.
///
/// The two things that can actually be wrong here are the byte length and the
/// base alignment, and reporting those needs no wide value. **The predicate is
/// unchanged** — `mut_from_bytes` still decides, and this only moves where the
/// panic is built. A 10-bit-only cost, since at one byte per pixel the whole
/// cast folds away.
#[cfg(feature = "zerocopy")]
#[cold]
#[inline(never)]
#[track_caller]
fn cast_slice_failed<V>(bytes: usize, addr: usize) -> ! {
    panic!(
        "DisjointMut: {} bytes at {:#x} is not a valid [{}] (size {}, align {})",
        bytes,
        addr,
        core::any::type_name::<V>(),
        mem::size_of::<V>(),
        mem::align_of::<V>(),
    );
}

#[cfg(feature = "zerocopy")]
impl<'a, T: AsMutPtr> DisjointMutGuard<'a, T, [u8]> {
    #[inline] // Inline to see alignment to potentially elide checks.
    fn cast_slice<V: IntoBytes + FromBytes + KnownLayout>(self) -> DisjointMutGuard<'a, T, [V]> {
        // We don't want to drop the old guard, because we aren't changing or
        // removing the borrow from parent here.
        let old_guard = ManuallyDrop::new(self);
        // SAFETY: the borrow is still registered (the old guard is in
        // `ManuallyDrop`, so nothing has retired it), which is what makes this
        // region exclusively ours. The reference does not outlive the cast: it
        // is consumed by `mut_from_bytes` and only its address is kept.
        let mut raw = old_guard.slice;
        let bytes: &mut [u8] = unsafe { raw.as_mut() };
        // Both are pure reads of values already live, so LLVM sinks them into
        // the cold arm; they exist only to give `cast_slice_failed` something
        // to report without a `CastError` on the hot path's frame.
        let (n, addr) = (bytes.len(), bytes.as_ptr() as usize);
        DisjointMutGuard {
            slice: match <[V]>::mut_from_bytes(bytes) {
                Ok(v) => NonNull::from(v),
                Err(_) => cast_slice_failed::<V>(n, addr),
            },
            phantom: PhantomData,
            parent: old_guard.parent,
            borrow_id: old_guard.borrow_id,
            #[cfg(feature = "__probe_bounds")]
            probe: old_guard.probe,
        }
    }

    #[inline] // Inline to see alignment to potentially elide checks.
    fn cast<V: IntoBytes + FromBytes + KnownLayout>(self) -> DisjointMutGuard<'a, T, V> {
        let old_guard = ManuallyDrop::new(self);
        // SAFETY: as in `cast_slice`.
        let mut raw = old_guard.slice;
        let bytes: &mut [u8] = unsafe { raw.as_mut() };
        DisjointMutGuard {
            slice: NonNull::from(V::mut_from_bytes(bytes).unwrap()),
            phantom: PhantomData,
            parent: old_guard.parent,
            borrow_id: old_guard.borrow_id,
            #[cfg(feature = "__probe_bounds")]
            probe: old_guard.probe,
        }
    }
}

// A raw pointer field is `!Send` and `!Sync`, and `PhantomData` cannot argue it
// back. These four impls RESTORE, exactly, what the compiler derived while the
// field was `&'a mut V` / `&'a V` — dropping them would be a silent semver break
// for anything that moves a guard between threads.
//
// SAFETY (all four): the guard is a `Copy` pointer plus `Option<&DisjointMut<T>>`
// plus a `u64` id. Sending or sharing it is sending or sharing exactly
// `&'a mut V` (resp. `&'a V`) and `&'a DisjointMut<T>`, whose own auto-trait
// rules are reproduced verbatim in the bounds below:
//
// * `&mut V: Send` iff `V: Send`; `&mut V: Sync` iff `V: Sync`;
// * `&V: Send` iff `V: Sync`;   `&V: Sync` iff `V: Sync`;
// * `&DisjointMut<T>: Send`/`Sync` iff `DisjointMut<T>: Sync`, which the impl
//   above grants for `T: AsMutPtr + Sync`.
//
// The tracker registration the guard names is shared state either way, and it is
// already `Send + Sync` (`BorrowTracker`).
unsafe impl<T: ?Sized + AsMutPtr + Sync, V: ?Sized + Send> Send for DisjointMutGuard<'_, T, V> {}
unsafe impl<T: ?Sized + AsMutPtr + Sync, V: ?Sized + Sync> Sync for DisjointMutGuard<'_, T, V> {}
unsafe impl<T: ?Sized + AsMutPtr + Sync, V: ?Sized + Sync> Send for DisjointImmutGuard<'_, T, V> {}
unsafe impl<T: ?Sized + AsMutPtr + Sync, V: ?Sized + Sync> Sync for DisjointImmutGuard<'_, T, V> {}

/// THROWAWAY (`__probe_bounds`) footprint declaration on the guards.
///
/// A no-op — and byte-for-byte absent — without the feature. The callers are
/// the strided helpers that ALREADY compute `(w, rows, stride)` for their own
/// bounds arithmetic, so declaring costs them nothing but the call.
impl<'a, T: ?Sized + AsMutPtr, V: ?Sized> DisjointMutGuard<'a, T, V> {
    /// The guard's true footprint is `rows` spans of `w` elements, the first at
    /// absolute element offset `lo`, successive ones `stride` elements apart.
    #[inline(always)]
    pub fn probe_declare_rows(&self, lo: usize, w: usize, rows: usize, stride: isize) {
        #[cfg(feature = "__probe_bounds")]
        self.probe.declare_rows(lo, w, rows, stride);
        #[cfg(not(feature = "__probe_bounds"))]
        let _ = (lo, w, rows, stride);
    }
}

impl<'a, T: ?Sized + AsMutPtr, V: ?Sized> DisjointImmutGuard<'a, T, V> {
    /// See [`DisjointMutGuard::probe_declare_rows`].
    #[inline(always)]
    pub fn probe_declare_rows(&self, lo: usize, w: usize, rows: usize, stride: isize) {
        #[cfg(feature = "__probe_bounds")]
        self.probe.declare_rows(lo, w, rows, stride);
        #[cfg(not(feature = "__probe_bounds"))]
        let _ = (lo, w, rows, stride);
    }
}

impl<'a, T: ?Sized + AsMutPtr, V: ?Sized> Deref for DisjointMutGuard<'a, T, V> {
    type Target = V;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        #[cfg(feature = "__probe_bounds")]
        self.probe.mark_read();
        // SAFETY: the registration that makes this region exclusively ours is
        // live for as long as the guard is, and it is retired only in `Drop`.
        // Borrowck ties the result to `&self`, so the guard cannot be dropped
        // or moved while the reference exists.
        unsafe { self.slice.as_ref() }
    }
}

impl<'a, T: ?Sized + AsMutPtr, V: ?Sized> DerefMut for DisjointMutGuard<'a, T, V> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        #[cfg(feature = "__probe_bounds")]
        self.probe.mark_write();
        // SAFETY: as in `deref`, and the borrow is a mutable registration, so
        // no other live borrow overlaps this region.
        unsafe { self.slice.as_mut() }
    }
}

/// A live shared borrow of one region.
///
/// Holds a pointer rather than `&'a V` for the reason spelled out on
/// [`DisjointMutGuard`]: moving the guard by value into a call — `drop(g)`
/// above all — would protect the reference for the duration of that call,
/// while `Drop` inside it retires the record and lets another thread take the
/// region mutably. Same defect, same fix; `core::cell::Ref` holds a
/// `NonNull<T>` for the same reason.
pub struct DisjointImmutGuard<'a, T: ?Sized + AsMutPtr, V: ?Sized> {
    /// `NonNull`, not `*const V` — see [`DisjointMutGuard::slice`].
    slice: NonNull<V>,

    /// See [`DisjointMutGuard::phantom`]. `&'a V` here, so `V: Sync` is what
    /// this guard needs to be `Send` — exactly what the reference field gave.
    phantom: PhantomData<(&'a V, &'a DisjointMut<T>)>,

    parent: Option<&'a DisjointMut<T>>,
    borrow_id: checked::BorrowId,
    /// THROWAWAY (`__probe_bounds`): handle to this acquisition's live record.
    #[cfg(feature = "__probe_bounds")]
    pub(crate) probe: bounds_probe::Ticket,
}

#[cfg(feature = "zerocopy")]
impl<'a, T: AsMutPtr> DisjointImmutGuard<'a, T, [u8]> {
    #[inline]
    fn cast_slice<V: FromBytes + KnownLayout + Immutable>(self) -> DisjointImmutGuard<'a, T, [V]> {
        let old_guard = ManuallyDrop::new(self);
        // SAFETY: the borrow is still registered (the old guard is in
        // `ManuallyDrop`), so no mutable borrow overlaps this region.
        let raw = old_guard.slice;
        let bytes: &[u8] = unsafe { raw.as_ref() };
        // See `cast_slice_failed`: the `CastError` is what puts a 112-byte
        // frame on this function's success path.
        let (n, addr) = (bytes.len(), bytes.as_ptr() as usize);
        DisjointImmutGuard {
            slice: match <[V]>::ref_from_bytes(bytes) {
                Ok(v) => NonNull::from(v),
                Err(_) => cast_slice_failed::<V>(n, addr),
            },
            phantom: PhantomData,
            parent: old_guard.parent,
            borrow_id: old_guard.borrow_id,
            #[cfg(feature = "__probe_bounds")]
            probe: old_guard.probe,
        }
    }

    #[inline]
    fn cast<V: FromBytes + KnownLayout + Immutable>(self) -> DisjointImmutGuard<'a, T, V> {
        let old_guard = ManuallyDrop::new(self);
        // SAFETY: as in `cast_slice`.
        let raw = old_guard.slice;
        let bytes: &[u8] = unsafe { raw.as_ref() };
        DisjointImmutGuard {
            slice: NonNull::from(V::ref_from_bytes(bytes).unwrap()),
            phantom: PhantomData,
            parent: old_guard.parent,
            borrow_id: old_guard.borrow_id,
            #[cfg(feature = "__probe_bounds")]
            probe: old_guard.probe,
        }
    }
}

impl<'a, T: ?Sized + AsMutPtr, V: ?Sized> Deref for DisjointImmutGuard<'a, T, V> {
    type Target = V;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        #[cfg(feature = "__probe_bounds")]
        self.probe.mark_read();
        // SAFETY: the shared registration is live for as long as the guard is
        // and is retired only in `Drop`, so no mutable borrow overlaps this
        // region while the returned reference exists.
        unsafe { self.slice.as_ref() }
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
    impl<V: Copy> Sealed for &mut [V] {}
    impl<V: Copy, const N: usize> Sealed for [V; N] {}
    impl<V: Copy> Sealed for [V] {}
    impl<V: Copy> Sealed for Box<[V]> {}

    /// Sealing trait for index/range traits.
    ///
    /// These implementations are part of the soundness boundary because
    /// `DisjointMut` trusts them to return pointers matching the registered
    /// [`Bounds`](super::Bounds).
    pub trait IndexLike {}

    impl IndexLike for usize {}
    impl IndexLike for core::ops::Range<usize> {}
    impl IndexLike for core::ops::RangeFrom<usize> {}
    impl IndexLike for core::ops::RangeInclusive<usize> {}
    impl IndexLike for core::ops::RangeTo<usize> {}
    impl IndexLike for core::ops::RangeToInclusive<usize> {}
    impl IndexLike for core::ops::RangeFull {}
    impl IndexLike for (core::ops::RangeFrom<usize>, core::ops::RangeTo<usize>) {}
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
///    [`ExternalAsMutPtr::as_mut_slice`] with raw pointer metadata.
///
/// 3. **Valid pointer.** The returned `*mut Self::Target` must be valid
///    for reads and writes over `0..self.len()` elements.
///
/// 4. **Stable length.** `len()` must return a consistent value for the
///    lifetime of any outstanding borrow guard.
///
/// 5. **Inline data requires `as_mut_slice` override.** The default
///    `as_mut_slice` calls `(*ptr).len()` which creates `&Self`. For
///    types where element data is stored inline (e.g. `[V; N]` wrapped
///    in a newtype), this creates a SharedReadOnly tag covering the
///    data, which is UB under Stacked Borrows when concurrent mutable
///    guards exist. **You MUST override `as_mut_slice`** for inline-data
///    types using `ptr::slice_from_raw_parts_mut(ptr.cast(), N)`.
///
/// See the `Vec<V>` and `Aligned<A, [V; N]>` implementations in this
/// crate for reference patterns.
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

    /// Returns a mutable pointer to the underlying slice.
    ///
    /// For types where data lives on the heap (e.g. `Vec`-like), creating
    /// `&Self` to read `len()` is fine — `&Self` only covers the container
    /// metadata, not the heap allocation:
    ///
    /// ```ignore
    /// unsafe fn as_mut_slice(ptr: *mut Self) -> *mut [Self::Target] {
    ///     let this = unsafe { &*ptr };
    ///     ptr::slice_from_raw_parts_mut(this.as_ptr().cast_mut(), this.len())
    /// }
    /// ```
    ///
    /// For types where data is stored **inline** (e.g. `[V; N]` in a newtype),
    /// you must avoid creating `&Self` because it produces a SharedReadOnly
    /// tag covering the element data, invalidating concurrent `&mut` guards
    /// under Stacked Borrows:
    ///
    /// ```ignore
    /// unsafe fn as_mut_slice(ptr: *mut Self) -> *mut [Self::Target] {
    ///     ptr::slice_from_raw_parts_mut(ptr.cast(), N) // no reference created
    /// }
    /// ```
    ///
    /// # Safety
    ///
    /// `ptr` must be safely dereferenceable.
    unsafe fn as_mut_slice(ptr: *mut Self) -> *mut [Self::Target];

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

    unsafe fn as_mut_slice(ptr: *mut Self) -> *mut [Self::Target] {
        unsafe { <T as ExternalAsMutPtr>::as_mut_slice(ptr) }
    }

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

    /// THROWAWAY (`__probe_bounds`): declare this buffer's picture row stride in
    /// BYTES, so the report can evaluate the "widen a guard to the full picture
    /// rows it spans" counterfactual — the shape #485's loop-filter band took.
    ///
    /// A no-op, and absent from codegen, without the feature.
    #[inline(always)]
    pub fn probe_declare_stride(&self, stride_bytes: isize) {
        #[cfg(feature = "__probe_bounds")]
        bounds_probe::declare_stride(
            self.as_mut_ptr() as usize,
            self.as_mut_slice().len(),
            stride_bytes,
        );
        #[cfg(not(feature = "__probe_bounds"))]
        let _ = stride_bytes;
    }

    /// THROWAWAY (`__probe_bounds`): evaluate the STRIDED-RECTANGLE
    /// counterfactual for the `rows` per-row guards the caller is about to
    /// take, WITHOUT registering anything.
    ///
    /// Answers the question `884b4b5`/`424cbbb` settled by argument in March
    /// 2026 and #472 re-opened: would ONE exact `rows x w` record ever reject a
    /// foreign record that `rows` per-row records permit — i.e. is there
    /// concurrent traffic in the inter-row gaps? See
    /// `bounds_probe::eval_rect` (not a doc link: that module is `__probe_bounds`-gated
    /// and absent from a default doc build, which is what broke the Documentation leg).
    ///
    /// A no-op, and absent from codegen, without the feature.
    #[inline(always)]
    pub fn probe_eval_rect(
        &self,
        loc: &'static core::panic::Location<'static>,
        is_mut: bool,
        lo: usize,
        w: usize,
        rows: usize,
        stride: isize,
    ) {
        #[cfg(feature = "__probe_bounds")]
        if let Some(tracker) = self.tracker.as_ref() {
            bounds_probe::eval_rect(
                loc,
                self.as_mut_ptr() as usize,
                is_mut,
                lo,
                w,
                rows,
                stride,
                tracker.probe_geometry(),
            );
        }
        #[cfg(not(feature = "__probe_bounds"))]
        let _ = (loc, is_mut, lo, w, rows, stride);
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
        let mut bounds: Bounds = index.clone().into();
        // Clamp an open-ended range (`index(..)`, `index(n..)` — both encode
        // their end as `usize::MAX`) to the real length. Sound: if the range
        // really did run past the end, `get_mut` below panics and poisons, so
        // no borrow of `len..end` can be missed. Without this the tracker would
        // see an astronomically long span and take its all-shard slow path.
        // `as_mut_slice` is called again below, so this costs nothing.
        clamp_bounds(&mut bounds, self.as_mut_slice().len());
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
        // No reference is created here — see [`DisjointMutGuard`] for why the
        // guard must not carry one.
        // SAFETY (NonNull): `get_mut` derives from `as_mut_slice`, which is a
        // live allocation's pointer, and panics rather than returning null.
        let slice = unsafe { NonNull::new_unchecked(index.get_mut(self.as_mut_slice())) };
        // Success — disarm the cleanup guard.
        mem::forget(cleanup);
        // THROWAWAY (`__probe_bounds`). Registered AFTER the index succeeded so
        // a panicking `get_mut` cannot leave a live record behind, and with the
        // SAME clamped bounds the tracker saw, so the two instruments reconcile
        // registration for registration.
        #[cfg(feature = "__probe_bounds")]
        let probe = match parent {
            Some(_) => bounds_probe::acquire(
                core::panic::Location::caller(),
                self.as_mut_ptr() as usize,
                true,
                bounds.range.start,
                bounds.range.end,
            ),
            None => bounds_probe::Ticket::NONE,
        };
        DisjointMutGuard {
            slice,
            parent,
            borrow_id,
            phantom: PhantomData,
            #[cfg(feature = "__probe_bounds")]
            probe,
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
        let mut bounds: Bounds = index.clone().into();
        // See `index_mut` for why the clamp is here and why it is sound.
        clamp_bounds(&mut bounds, self.as_mut_slice().len());
        let borrow_id = match &self.tracker {
            Some(tracker) => tracker.add_immut(&bounds),
            None => checked::BorrowId::UNCHECKED,
        };
        let parent = self.tracker.as_ref().map(|_| self);
        let cleanup = BorrowCleanup { parent };
        // SAFETY: The borrow has been registered (or we're unchecked). No
        // reference is created here — see [`DisjointImmutGuard`].
        // SAFETY (NonNull): as in `index_mut`.
        let slice = unsafe { NonNull::new_unchecked(index.get_mut(self.as_mut_slice())) };
        mem::forget(cleanup);
        #[cfg(feature = "__probe_bounds")]
        let probe = match parent {
            Some(_) => bounds_probe::acquire(
                core::panic::Location::caller(),
                self.as_mut_ptr() as usize,
                false,
                bounds.range.start,
                bounds.range.end,
            ),
            None => bounds_probe::Ticket::NONE,
        };
        DisjointImmutGuard {
            slice,
            parent,
            borrow_id,
            phantom: PhantomData,
            #[cfg(feature = "__probe_bounds")]
            probe,
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
    ///
    /// # Why `inline(always)`
    ///
    /// At one byte per pixel `V == u8`, every part of this — the `mul(1)`, the
    /// cast, the length check — folds to nothing and the caller is left with a
    /// bare [`Self::index_mut`]. At two bytes per pixel it does not fold, and
    /// under plain `#[inline]` LLVM declines to inline it: the release build
    /// grows an out-of-line `mut_slice_as::<_, u16>` whose HOT path is ~35
    /// aarch64 instructions, of which a 112-byte frame, ten callee-saved
    /// spill/reload pairs and the call/ret are about half. The 112 bytes exist
    /// only to hold the `CastError` the cold `.unwrap()` path would report,
    /// which is why this attribute and [`cast_slice_failed`] are ONE change:
    /// they are strongly super-additive, and either alone is small enough to be
    /// mistaken for noise. Measured on 2aa00c5, v4k_8tile_10b t=1, paired
    /// per-round ratios vs that base, n=9, md5-identical, idle box:
    ///
    /// ```text
    ///   inline(always) alone   0.9862  [0.9713, 0.9971]
    ///   cast_slice_failed alone 0.9820 [0.9684, 0.9962]
    ///   both (shipped)         0.9374  [0.9248, 0.9454]
    /// ```
    ///
    /// Independent halves would predict 0.969; the pair delivers 0.937. A
    /// one-knob A/B would have found ~1.5% with a band nearly touching 1.000
    /// and plausibly stopped there.
    ///
    /// See `benchmarks/verify_compose4_2026-08-08.meta` section 8.
    #[inline(always)]
    #[track_caller]
    pub fn mut_slice_as<'a, I, V>(&'a self, index: I) -> DisjointMutGuard<'a, T, [V]>
    where
        I: SliceBounds,
        V: IntoBytes + FromBytes + KnownLayout,
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
        V: IntoBytes + FromBytes + KnownLayout,
    {
        self.index_mut((index..index + 1).mul(mem::size_of::<V>()))
            .cast()
    }

    /// Immutably borrow a slice of a convertible type.
    ///
    /// `#[inline(always)]` for the reason spelled out on [`Self::mut_slice_as`]:
    /// the wrapper is free at one byte per pixel and an out-of-line call with a
    /// 112-byte frame at two.
    #[inline(always)]
    #[track_caller]
    pub fn slice_as<'a, I, V>(&'a self, index: I) -> DisjointImmutGuard<'a, T, [V]>
    where
        I: SliceBounds,
        V: FromBytes + KnownLayout + Immutable,
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
        V: FromBytes + KnownLayout + Immutable,
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
///
/// # Sealed
///
/// This trait is sealed and cannot be implemented outside of this crate.
/// `DisjointMut` relies on index implementations returning in-bounds pointers
/// matching their registered [`Bounds`].
pub trait DisjointMutIndex<T: ?Sized>: sealed::IndexLike {
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

/// Translate an index from element units to byte units.
///
/// This trait is sealed and cannot be implemented outside of this crate.
pub trait TranslateRange: sealed::IndexLike {
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
        // 3..=5 with by=4 means elements 3,4,5 → bytes 12..=23 (not 12..=20).
        // Each element occupies `by` bytes, so the inclusive end in bytes is
        // one past the last element's start: (end + 1) * by - 1.
        *self.start() * by..=(*self.end() + 1) * by - 1
    }
}

impl TranslateRange for RangeTo<usize> {
    fn mul(&self, by: usize) -> Self {
        ..self.end * by
    }
}

impl TranslateRange for RangeToInclusive<usize> {
    fn mul(&self, by: usize) -> Self {
        // ..=5 with by=4 means elements 0..=5 → bytes 0..=23 (not 0..=20).
        ..=(self.end + 1) * by - 1
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

/// Trim an open-ended [`Bounds`] to the container's real length.
///
/// `RangeFull` / `RangeFrom` encode "to the end" as [`usize::MAX`], which is
/// fine for a scan of plain intervals but makes an address-sharded tracker
/// treat the borrow as spanning the whole 64-bit space.
#[inline(always)]
fn clamp_bounds(bounds: &mut Bounds, len: usize) {
    if bounds.range.end > len {
        bounds.range.end = len;
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

/// Range-like indexes supported by [`DisjointMut`].
///
/// This trait is sealed and cannot be implemented outside of this crate.
pub trait SliceBounds: TranslateRange + Clone + sealed::IndexLike {
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
// Bounds tracking (per-borrow, checked before the reference is created)
// =============================================================================

/// The default tracker: address-block sharded, so concurrent tile workers stop
/// serialising on one lock and one cache line.
#[cfg(not(any(
    feature = "__probe_count",
    feature = "__probe_noscan",
    feature = "__probe_lockonly",
    feature = "__tracker_legacy"
)))]
mod tracker_shard;
#[cfg(not(any(
    feature = "__probe_count",
    feature = "__probe_noscan",
    feature = "__probe_lockonly",
    feature = "__tracker_legacy"
)))]
use tracker_shard as checked;

/// Wide-path reason counters, when `__probe_wide` is on. See the module docs.
#[cfg(all(
    feature = "__probe_wide",
    not(any(
        feature = "__probe_count",
        feature = "__probe_noscan",
        feature = "__probe_lockonly",
        feature = "__tracker_legacy"
    ))
))]
pub use tracker_shard::wide_probe;

/// The single-lock predecessor, kept only so the throwaway `__probe_*`
/// decomposition arms (`benchmarks/tracker_decomp_2026-08-07.meta`) remain
/// reproducible against the tracker they actually measured.
///
/// `__tracker_legacy` selects it with no probe hooks at all, which is the
/// straight A/B baseline arm: same commit, same decoder, only the tracker
/// differs.
#[cfg(any(
    feature = "__probe_count",
    feature = "__probe_noscan",
    feature = "__probe_lockonly",
    feature = "__tracker_legacy"
))]
mod tracker_legacy;
#[cfg(any(
    feature = "__probe_count",
    feature = "__probe_noscan",
    feature = "__probe_lockonly",
    feature = "__tracker_legacy"
))]
use tracker_legacy as checked;

/// Declare the decode parallelism to the borrow tracker.
///
/// The tracker shards each big buffer's records by address so that concurrent
/// tile workers stop serialising on one lock and one cache line, and the number
/// of shards is a real trade: a serial decode is *slower* with more of them
/// (its wide-borrow path holds every active shard), a concurrent one is much
/// faster. Call this before creating the buffers a threaded decode will use.
///
/// Monotone: a later single-threaded declaration never shrinks the shard count
/// out from under a concurrently live multi-threaded decoder.
///
/// A no-op when the tracker is compiled out.
#[inline]
pub fn set_parallelism(n: usize) {
    #[cfg(feature = "__probe_untracked")]
    let _ = n;
    #[cfg(not(feature = "__probe_untracked"))]
    checked::set_parallelism(n);
}

/// Declare how many tiles the frame about to be decoded splits into.
///
/// The companion to [`set_parallelism`], and the other half of the gate on the
/// adaptive block shift: thread count says whether anything is concurrent, the
/// tile split says whether the concurrency is the kind a coarser block helps.
/// Call it once the frame header is parsed and BEFORE that frame's picture is
/// allocated, since the shift is fixed when a buffer's tracker is built.
///
/// Monotone, and a no-op when the tracker is compiled out.
#[inline]
pub fn set_tile_concurrency(n: usize) {
    #[cfg(feature = "__probe_untracked")]
    let _ = n;
    #[cfg(not(feature = "__probe_untracked"))]
    checked::set_tile_concurrency(n);
}

// =============================================================================
// Guard Drop impls — deregister borrow on drop
// =============================================================================

impl<'a, T: ?Sized + AsMutPtr, V: ?Sized> Drop for DisjointMutGuard<'a, T, V> {
    fn drop(&mut self) {
        // Retire the probe record BEFORE the tracker's, so the probe's live
        // interval is a strict subset of the tracker's and a probe-observed
        // concurrency always implies a tracker-observed one.
        #[cfg(feature = "__probe_bounds")]
        bounds_probe::release(self.probe);
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
        #[cfg(feature = "__probe_bounds")]
        bounds_probe::release(self.probe);
        if let Some(parent) = self.parent {
            parent.tracker.as_ref().unwrap().remove(self.borrow_id);
        }
    }
}

// =============================================================================
// Strided-rectangle borrows
// =============================================================================

/// A live shared borrow of `rows` segments of `seg` elements, `stride` elements
/// apart — registered as ONE tracker record whose footprint is EXACTLY those
/// segments.
///
/// # Why this is not the reverted strided tracker, and not a hull
///
/// Two earlier shapes failed, in opposite directions, and this one is neither:
///
/// * A **hull** guard (`[lo, lo + (rows-1)*stride + seg)`) reserves the
///   inter-row GAPS, which belong to other columns of the same picture rows. Two
///   tile workers routinely write the same rows at different columns, so the gap
///   reservation turns a genuinely disjoint pair into a false positive. Measured
///   as decode failures (`__probe_rect_hull`) and, where it does pass, as 2.65x
///   SLOWER. This guard's record covers no gap byte
///   (the tracker's `add_rect_immut`, which stores the rectangle exactly).
/// * The March-2026 strided tracker had an exact record but handed out a
///   reference over the whole hull, so safe code held a `&[T]` covering bytes
///   another thread was mutating: Miri UB under both memory models. **This guard
///   never materialises a reference wider than one row.** There is no `Deref`;
///   [`Self::row`] derives each row's slice from the buffer's own pointer, and
///   the rows are pairwise disjoint.
///
/// The pointer-not-reference field discipline is [`DisjointMutGuard`]'s, for
/// [`DisjointMutGuard`]'s reason: moving the guard by value into `drop` would
/// protect a reference whose bytes another thread is about to retag.
pub struct DisjointImmutRectGuard<'a, T: ?Sized + AsMutPtr, V> {
    /// Row 0's first element. For a NEGATIVE stride this is the HIGHEST-address
    /// row, which is why the tracker is handed an ascending hull computed
    /// separately: the two describe the same byte set.
    base: NonNull<V>,
    /// Elements per row.
    seg: usize,
    rows: usize,
    /// Elements between successive rows; may be negative.
    stride: isize,
    phantom: PhantomData<(&'a V, &'a DisjointMut<T>)>,
    parent: Option<&'a DisjointMut<T>>,
    borrow_id: checked::BorrowId,
}

// SAFETY: the guard is a `Copy` pointer plus `Option<&DisjointMut<T>>`, and the
// data it points at is shared-borrowed for the guard's whole life. Sending it
// therefore needs what sending a `&'a [V]` needs — `V: Sync` — plus `T: Sync`
// for the parent reference, exactly as for `DisjointImmutGuard`.
unsafe impl<T: ?Sized + AsMutPtr + Sync, V: Sync> Send for DisjointImmutRectGuard<'_, T, V> {}
// SAFETY: as above; sharing the guard shares only shared references.
unsafe impl<T: ?Sized + AsMutPtr + Sync, V: Sync> Sync for DisjointImmutRectGuard<'_, T, V> {}

impl<'a, T: ?Sized + AsMutPtr, V> DisjointImmutRectGuard<'a, T, V> {
    /// Elements per row.
    #[inline(always)]
    pub fn seg(&self) -> usize {
        self.seg
    }

    /// Rows the guard covers.
    #[inline(always)]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Row `r` of the rectangle.
    ///
    /// # Panics
    ///
    /// If `r >= self.rows()`.
    #[inline(always)]
    pub fn row(&self, r: usize) -> &[V] {
        assert!(r < self.rows, "row index out of range");
        // SAFETY: the registration covering exactly these `rows` segments is
        // live for as long as the guard is (retired only in `Drop`), so no
        // mutable borrow overlaps any of them. `r < rows`, and the constructor
        // proved that every row of the rectangle lies inside the allocation, so
        // the offset is in bounds and the `seg` elements from it are readable.
        // Borrowck ties the result to `&self`, so the guard cannot be dropped
        // while the reference exists. No reference wider than one row is ever
        // created, which is what keeps the inter-row gaps out of the aliasing
        // model as well as out of the tracker.
        unsafe {
            let p = self.base.as_ptr().offset(r as isize * self.stride);
            core::slice::from_raw_parts(p, self.seg)
        }
    }
}

impl<'a, T: ?Sized + AsMutPtr, V> Drop for DisjointImmutRectGuard<'a, T, V> {
    fn drop(&mut self) {
        if let Some(parent) = self.parent {
            parent.tracker.as_ref().unwrap().remove(self.borrow_id);
        }
    }
}

/// [`DisjointImmutRectGuard`], exclusively — a live EXCLUSIVE borrow of `rows`
/// segments of `seg` elements, `stride` elements apart, registered as ONE
/// tracker record (`add_rect_mut`) whose footprint is exactly those segments.
///
/// Everything the shared guard's type-level docs say about why this is neither
/// a hull nor the reverted March-2026 strided tracker applies unchanged; the
/// two differences are:
///
/// * the record is MUTABLE, so the tracker's `find::<true>` scan rejects a
///   concurrent reader of any of the `rows` segments exactly as the `rows`
///   per-row mutable guards it replaces would have; and
/// * [`Self::row_mut`] takes `&mut self`, so **at most one row reference exists
///   at a time**. That is what keeps this sound without ever materialising a
///   `&mut [V]` wider than one row: the rows are pairwise disjoint (the
///   constructor requires `|stride| >= seg`), and borrowck serialises them
///   anyway.
pub struct DisjointMutRectGuard<'a, T: ?Sized + AsMutPtr, V> {
    /// Row 0's first element; for a NEGATIVE stride, the highest-address row.
    base: NonNull<V>,
    /// Elements per row.
    seg: usize,
    rows: usize,
    /// Elements between successive rows; may be negative.
    stride: isize,
    phantom: PhantomData<(&'a mut V, &'a DisjointMut<T>)>,
    parent: Option<&'a DisjointMut<T>>,
    borrow_id: checked::BorrowId,
}

// SAFETY: the same reasoning as `DisjointMutGuard`'s, whose auto-trait bounds
// these mirror: the guard is a pointer plus `Option<&DisjointMut<T>>`, and the
// bytes it points at are exclusively borrowed for its whole life, so sending it
// needs what sending a `&'a mut [V]` needs.
unsafe impl<T: ?Sized + AsMutPtr + Sync, V: Send> Send for DisjointMutRectGuard<'_, T, V> {}
// SAFETY: as above; sharing the guard hands out no `&mut` (that needs `&mut
// self`), so sharing it shares only shared references.
unsafe impl<T: ?Sized + AsMutPtr + Sync, V: Sync> Sync for DisjointMutRectGuard<'_, T, V> {}

impl<'a, T: ?Sized + AsMutPtr, V> DisjointMutRectGuard<'a, T, V> {
    /// Elements per row.
    #[inline(always)]
    pub fn seg(&self) -> usize {
        self.seg
    }

    /// Rows the guard covers.
    #[inline(always)]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Row `r` of the rectangle, exclusively.
    ///
    /// # Panics
    ///
    /// If `r >= self.rows()`.
    #[inline(always)]
    pub fn row_mut(&mut self, r: usize) -> &mut [V] {
        assert!(r < self.rows, "row index out of range");
        // SAFETY: the MUTABLE registration covering exactly these `rows`
        // segments is live for as long as the guard is (retired only in
        // `Drop`), so no other borrow overlaps any of them. `r < rows`, and the
        // constructor proved every row lies inside the allocation, so the
        // offset is in bounds and the `seg` elements from it are writable.
        // `&mut self` ties the result to the guard and makes at most one row
        // reference live at a time, so no two of these can alias even though
        // the rows are in fact pairwise disjoint (`|stride| >= seg`).
        unsafe {
            let p = self.base.as_ptr().offset(r as isize * self.stride);
            core::slice::from_raw_parts_mut(p, self.seg)
        }
    }
}

impl<'a, T: ?Sized + AsMutPtr, V> Drop for DisjointMutRectGuard<'a, T, V> {
    fn drop(&mut self) {
        if let Some(parent) = self.parent {
            let tracker = parent.tracker.as_ref().unwrap();
            // A panic while an exclusive guard is live may leave the data
            // partially written; poison exactly as `DisjointMutGuard` does.
            #[cfg(feature = "std")]
            if std::thread::panicking() {
                tracker.poison();
            }
            tracker.remove(self.borrow_id);
        }
    }
}

impl<T: ?Sized + AsMutPtr> DisjointMut<T> {
    /// Borrow `rows` rows of `seg` elements, `stride` elements apart, starting
    /// at element `lo`, as ONE exact strided-rectangle registration.
    ///
    /// `None` means the geometry is not representable as a single record here
    /// and **the caller must take its own per-row path** — nothing is
    /// approximated to make a rectangle fit. The tracker's reasons are on
    /// the tracker's `add_rect_immut` (`tracker_shard.rs`); this wrapper adds that the
    /// rectangle must lie inside the buffer and its rows must not overlap each
    /// other.
    ///
    /// An `unchecked` instance (no tracker) always succeeds — there is no record
    /// to represent.
    #[inline]
    #[track_caller]
    pub fn index_rect<'a>(
        &'a self,
        lo: usize,
        seg: usize,
        rows: usize,
        stride: isize,
    ) -> Option<DisjointImmutRectGuard<'a, T, <T as AsMutPtr>::Target>> {
        // SAFETY (of the `V` choice): `V` is the buffer's own element type, so
        // every byte of it is a valid `V` by construction.
        self.index_rect_inner::<<T as AsMutPtr>::Target>(lo, seg, rows, stride)
    }

    /// The shared body of [`Self::index_rect`] and [`Self::index_rect_as`].
    ///
    /// PRIVATE, and it must stay private: it materialises `&[V]` over the
    /// buffer's bytes, which is only sound when every bit pattern is a valid
    /// `V`. The two public entry points establish that — one uses the buffer's
    /// own element type, the other requires `zerocopy::FromBytes`.
    #[inline]
    #[track_caller]
    fn index_rect_inner<'a, V>(
        &'a self,
        lo: usize,
        seg: usize,
        rows: usize,
        stride: isize,
    ) -> Option<DisjointImmutRectGuard<'a, T, V>> {
        let (lo_asc, astride, base) = self.rect_geometry::<V>(lo, seg, rows, stride)?;
        let esz = mem::size_of::<V>();
        let borrow_id = match &self.tracker {
            Some(tracker) => {
                tracker.add_rect_immut(lo_asc * esz, seg * esz, rows, astride.checked_mul(esz)?)?
            }
            None => checked::BorrowId::UNCHECKED,
        };
        let parent = self.tracker.as_ref().map(|_| self);
        Some(DisjointImmutRectGuard {
            base,
            seg,
            rows,
            stride,
            phantom: PhantomData,
            parent,
            borrow_id,
        })
    }

    /// [`Self::index_rect_inner`], exclusively: ONE `add_rect_mut` record over
    /// exactly the `rows` segments, and a guard that hands out one row at a
    /// time.
    ///
    /// PRIVATE for [`Self::index_rect_inner`]'s reason — it materialises
    /// `&mut [V]` over the buffer's bytes, which is only sound when every bit
    /// pattern is a valid `V`; the public entry points establish that.
    #[inline]
    #[track_caller]
    fn index_rect_mut_inner<'a, V>(
        &'a self,
        lo: usize,
        seg: usize,
        rows: usize,
        stride: isize,
    ) -> Option<DisjointMutRectGuard<'a, T, V>> {
        let (lo_asc, astride, base) = self.rect_geometry::<V>(lo, seg, rows, stride)?;
        let esz = mem::size_of::<V>();
        let borrow_id = match &self.tracker {
            Some(tracker) => {
                tracker.add_rect_mut(lo_asc * esz, seg * esz, rows, astride.checked_mul(esz)?)?
            }
            None => checked::BorrowId::UNCHECKED,
        };
        let parent = self.tracker.as_ref().map(|_| self);
        Some(DisjointMutRectGuard {
            base,
            seg,
            rows,
            stride,
            phantom: PhantomData,
            parent,
            borrow_id,
        })
    }

    /// The geometry half of a rectangle borrow, shared by the immutable and
    /// mutable constructors so the two can never disagree about what is
    /// representable.
    ///
    /// Returns `(lo_asc, astride, base)` — the ASCENDING hull's first element,
    /// the absolute stride, and a pointer to ROW 0's first element (which for a
    /// negative stride is the highest-address row). `None` is a REFUSAL: a
    /// zero-sized element, an empty rectangle, rows that would overlap each
    /// other (`|stride| < seg`, which includes `stride == 0`), an arithmetic
    /// overflow, a rectangle running past the end of the buffer, or a base
    /// pointer misaligned for `V`.
    #[inline]
    fn rect_geometry<V>(
        &self,
        lo: usize,
        seg: usize,
        rows: usize,
        stride: isize,
    ) -> Option<(usize, usize, NonNull<V>)> {
        let esz = mem::size_of::<V>();
        if esz == 0 || seg == 0 || rows == 0 {
            return None;
        }
        let astride = stride.unsigned_abs();
        // Rows must not overlap each other, and a zero stride would make every
        // row the same bytes.
        if astride < seg {
            return None;
        }
        // The ASCENDING hull, in elements: for a negative stride, row 0 is the
        // last row in address order.
        let lo_asc = if stride >= 0 {
            lo
        } else {
            lo.checked_sub((rows - 1).checked_mul(astride)?)?
        };
        let span = (rows - 1).checked_mul(astride)?.checked_add(seg)?;
        let end_asc = lo_asc.checked_add(span)?;
        let whole = self.as_mut_slice();
        let len_v = whole.len() / esz;
        if end_asc > len_v {
            return None;
        }
        let base_ptr = whole as *mut <T as AsMutPtr>::Target;
        if (base_ptr as usize) % mem::align_of::<V>() != 0 {
            return None;
        }
        // SAFETY: `base_ptr` is the live allocation's pointer and is aligned for
        // `V` (checked above); `lo` is a row start inside `[lo_asc, end_asc)`
        // and `end_asc <= len_v`, so the offset is in bounds for the
        // allocation. No reference is created here — see the guard types.
        let base = unsafe { NonNull::new_unchecked((base_ptr as *mut V).add(lo)) };
        Some((lo_asc, astride, base))
    }

    /// [`Self::index_rect`], exclusively.
    ///
    /// `None` means the geometry is not representable as one record and the
    /// caller must take its own per-row path; nothing is approximated.
    #[inline]
    #[track_caller]
    pub fn index_rect_mut<'a>(
        &'a self,
        lo: usize,
        seg: usize,
        rows: usize,
        stride: isize,
    ) -> Option<DisjointMutRectGuard<'a, T, <T as AsMutPtr>::Target>> {
        // SAFETY (of the `V` choice): `V` is the buffer's own element type, so
        // every byte of it is a valid `V` by construction.
        self.index_rect_mut_inner::<<T as AsMutPtr>::Target>(lo, seg, rows, stride)
    }
}

#[cfg(feature = "zerocopy")]
impl<T: AsMutPtr<Target = u8>> DisjointMut<T> {
    /// [`Self::index_rect`] with the rows handed back as `&[V]`, for a `V` every
    /// bit pattern of which is valid — the same soundness class as
    /// [`Self::slice_as`]. `lo`, `seg` and `stride` are in `V` elements.
    #[inline]
    #[track_caller]
    pub fn index_rect_as<'a, V>(
        &'a self,
        lo: usize,
        seg: usize,
        rows: usize,
        stride: isize,
    ) -> Option<DisjointImmutRectGuard<'a, T, V>>
    where
        V: FromBytes + KnownLayout + Immutable,
    {
        self.index_rect_inner::<V>(lo, seg, rows, stride)
    }

    /// [`Self::index_rect_as`], exclusively. `V` must additionally be
    /// [`IntoBytes`] — writing through the guard writes `V`'s bytes back into
    /// the `u8` buffer, which is the same soundness class as
    /// [`Self::mut_slice_as`].
    #[inline]
    #[track_caller]
    pub fn index_rect_mut_as<'a, V>(
        &'a self,
        lo: usize,
        seg: usize,
        rows: usize,
        stride: isize,
    ) -> Option<DisjointMutRectGuard<'a, T, V>>
    where
        V: FromBytes + IntoBytes + KnownLayout + Immutable,
    {
        self.index_rect_mut_inner::<V>(lo, seg, rows, stride)
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
        self.inner.get_mut().resize(new_len, value);
        self.retrack();
    }
}

impl<T: ?Sized + AsMutPtr> DisjointMut<T> {
    /// Re-size the tracker for the container's current length.
    ///
    /// `&mut self` means no borrow can be outstanding, which is what makes
    /// re-provisioning the shard array safe. Callers that grow the container
    /// through a `&mut` path must call this, or a buffer that started tiny
    /// keeps a single shard forever.
    #[inline]
    fn retrack(&mut self) {
        let len = self.as_mut_slice().len();
        if let Some(tracker) = self.tracker.as_mut() {
            tracker.reprovision(len);
        }
    }

    /// Declare this buffer's picture row stride in BYTES, so the tracker can
    /// size its blocks in picture ROWS instead of in blocks-per-buffer.
    ///
    /// `&mut self` is the safety argument, exactly as for the resize path: no
    /// borrow can be outstanding while the block boundaries move.
    ///
    /// **This decides the shipped block shift for a picture plane** since
    /// 2026-08-11: the tracker coarsens from the block-count answer until a
    /// block spans `ROWS_PER_BLOCK_MIN` picture rows. Buffers that never call it
    /// (everything that is not a picture plane — there is no stride to declare)
    /// keep the block-count rule under every arm, as does the whole build when a
    /// `__bps_*` ladder rung or `__bps_blocks` is compiled in.
    ///
    /// Not sticky: a later [`Self::resize`] re-derives the shift from `len`
    /// alone and drops the hint. That degrades to the shipped rule, which is a
    /// performance question and never a correctness one — the boundaries only
    /// have to agree between the two registrants of a shared byte, and both read
    /// the same live `shift`. Picture planes are allocated at a fixed size and
    /// never resized, so today no caller hits it.
    #[inline]
    pub fn declare_row_stride(&mut self, stride_bytes: usize) {
        let len = self.as_mut_slice().len();
        if let Some(tracker) = self.tracker.as_mut() {
            tracker.set_row_stride(len, stride_bytes);
        }
    }
}

/// Fallible version of [`Resizable`]. Returns `Err` on allocation failure.
pub trait TryResizable {
    type Value;
    fn try_resize(
        &mut self,
        new_len: usize,
        value: Self::Value,
    ) -> Result<(), alloc::collections::TryReserveError>;
}

impl<V: Clone> TryResizable for Vec<V> {
    type Value = V;
    fn try_resize(
        &mut self,
        new_len: usize,
        value: V,
    ) -> Result<(), alloc::collections::TryReserveError> {
        if new_len > self.len() {
            self.try_reserve(new_len - self.len())?;
        }
        self.resize(new_len, value);
        Ok(())
    }
}

impl<T: AsMutPtr + TryResizable> DisjointMut<T> {
    pub fn try_resize(
        &mut self,
        new_len: usize,
        value: T::Value,
    ) -> Result<(), alloc::collections::TryReserveError> {
        self.inner.get_mut().try_resize(new_len, value)?;
        self.retrack();
        Ok(())
    }
}

/// Fallible version of [`ResizableWith`]. Returns `Err` on allocation failure.
pub trait TryResizableWith {
    type Item;
    fn try_resize_with<F: FnMut() -> Self::Item>(
        &mut self,
        new_len: usize,
        f: F,
    ) -> Result<(), alloc::collections::TryReserveError>;
}

impl<V> TryResizableWith for Vec<V> {
    type Item = V;
    fn try_resize_with<F: FnMut() -> V>(
        &mut self,
        new_len: usize,
        f: F,
    ) -> Result<(), alloc::collections::TryReserveError> {
        if new_len > self.len() {
            self.try_reserve(new_len - self.len())?;
        }
        self.resize_with(new_len, f);
        Ok(())
    }
}

impl<T: AsMutPtr + TryResizableWith> DisjointMut<T> {
    pub fn try_resize_with<F>(
        &mut self,
        new_len: usize,
        f: F,
    ) -> Result<(), alloc::collections::TryReserveError>
    where
        F: FnMut() -> T::Item,
        T: TryResizableWith,
    {
        self.inner.get_mut().try_resize_with(new_len, f)?;
        self.retrack();
        Ok(())
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
        self.inner.get_mut().clear();
        self.retrack();
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
        self.inner.get_mut().resize_with(new_len, f);
        self.retrack();
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

/// SAFETY: Copies the stored mutable slice reference as a raw slice pointer
/// without materializing a reference to the full slice. The data is borrowed
/// from the caller, so creating `&[V]` or `&mut [V]` for metadata lookup would
/// conflict with outstanding guards under Stacked Borrows.
unsafe impl<V: Copy> AsMutPtr for &mut [V] {
    type Target = V;

    unsafe fn as_mut_slice(ptr: *mut Self) -> *mut [Self::Target] {
        // SAFETY: `&mut [V]` and `*mut [V]` have the same pointer layout.
        // Reading the raw pointer value avoids reborrowing the full slice.
        unsafe { ptr.cast::<*mut [V]>().read() }
    }

    unsafe fn as_mut_ptr(ptr: *mut Self) -> *mut V {
        // SAFETY: Same as `as_mut_slice`; casting the raw slice pointer to its
        // data pointer does not create an intermediate reference.
        unsafe { Self::as_mut_slice(ptr).cast() }
    }

    fn len(&self) -> usize {
        // SAFETY: `self` is a reference to the stored slice reference, not to
        // the slice data. Copy the raw slice pointer value and read its metadata.
        unsafe { ptr::from_ref(self).cast::<*const [V]>().read().len() }
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
    /// Use `Deref` instead: `arc_slice.index_mut(0..50)` works directly.
    #[doc(hidden)]
    pub inner: Arc<DisjointMutSlice<T>>,
}

impl<T: Copy> Deref for DisjointMutArcSlice<T> {
    type Target = DisjointMutSlice<T>;

    #[inline]
    fn deref(&self) -> &DisjointMutSlice<T> {
        &self.inner
    }
}

impl<T: Copy> DisjointMutArcSlice<T> {
    /// Create a new `DisjointMutArcSlice` with `n` elements, all set to `value`.
    ///
    /// Returns `Err` on allocation failure instead of panicking.
    pub fn try_new(n: usize, value: T) -> Result<Self, alloc::collections::TryReserveError> {
        let mut v = Vec::new();
        v.try_reserve(n)?;
        v.resize(n, value);
        Ok(Self {
            inner: Arc::new(DisjointMut::new(v.into_boxed_slice())),
        })
    }

    /// Like [`try_new`](Self::try_new) but without borrow tracking.
    ///
    /// # Safety
    ///
    /// See [`DisjointMut::dangerously_unchecked()`].
    pub unsafe fn try_new_unchecked(
        n: usize,
        value: T,
    ) -> Result<Self, alloc::collections::TryReserveError> {
        let mut v = Vec::new();
        v.try_reserve(n)?;
        v.resize(n, value);
        Ok(Self {
            inner: Arc::new(unsafe { DisjointMut::dangerously_unchecked(v.into_boxed_slice()) }),
        })
    }
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
        /// Panics if `align_offset + usable_len` overflows or exceeds `vec.len()`.
        pub fn from_vec_aligned(vec: Vec<u8>, alignment: usize, usable_len: usize) -> Self {
            if usable_len == 0 {
                return Self {
                    buf: vec,
                    align_offset: 0,
                    usable_len: 0,
                };
            }
            let align_offset = vec.as_ptr().align_offset(alignment);
            let region_end = align_offset.checked_add(usable_len).unwrap_or_else(|| {
                panic!("PicBuf: aligned region ({align_offset} + {usable_len}) overflows usize")
            });
            assert!(
                region_end <= vec.len(),
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
#[doc(hidden)]
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

#[test]
fn test_dangerously_unchecked_skips_tracking() {
    use alloc::vec;
    // dangerously_unchecked creates an instance without borrow tracking.
    // Overlapping borrows don't panic (but would be UB in real code).
    let v = unsafe { DisjointMut::dangerously_unchecked(vec![0u8; 100]) };
    assert!(!v.is_checked());

    // This would panic on a tracked instance, but succeeds here:
    let _g1 = v.index_mut(0..50);
    let _g2 = v.index_mut(25..75); // overlaps with g1 — only safe because this is a test
}

#[test]
fn test_new_always_tracked() {
    use alloc::vec;
    let v = DisjointMut::new(vec![0u8; 100]);
    assert!(v.is_checked());
}

#[test]
fn test_range_inclusive_mul() {
    // 3..=5 with by=4: elements 3,4,5 → bytes 12,13,...,23
    let r = (3..=5usize).mul(4);
    assert_eq!(*r.start(), 12);
    assert_eq!(*r.end(), 23);

    // 0..=0 with by=4: element 0 → bytes 0,1,2,3
    let r = (0..=0usize).mul(4);
    assert_eq!(*r.start(), 0);
    assert_eq!(*r.end(), 3);

    // 0..=2 with by=1: elements 0,1,2 → bytes 0,1,2
    let r = (0..=2usize).mul(1);
    assert_eq!(*r.start(), 0);
    assert_eq!(*r.end(), 2);
}

#[test]
fn test_range_to_inclusive_mul() {
    // ..=5 with by=4: elements 0..=5 → bytes 0..=23
    let r = (..=5usize).mul(4);
    assert_eq!(r.end, 23);

    // ..=0 with by=4: element 0 → bytes 0..=3
    let r = (..=0usize).mul(4);
    assert_eq!(r.end, 3);
}

// NOTE: Tests for aligned/aligned-vec integration are in align.rs
