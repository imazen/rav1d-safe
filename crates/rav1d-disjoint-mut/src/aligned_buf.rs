//! A minimal heap buffer whose allocation start is aligned to a const `ALIGN`.
//!
//! This is the only storage `AlignedVec64` needs: create empty, grow with
//! `resize`/`try_reserve`, view as a slice, hand out the base pointer. It replaces the
//! general-purpose `aligned-vec` dependency so that the surface we vouch for is exactly the
//! surface we use (no `extend_from_slice`, `insert`, `drain`, boxed-slice conversions, ...).
//!
//! # Invariants
//!
//! * `ptr` is a valid, `T`-aligned, `ALIGN`-aligned pointer. If `size_of::<T>() == 0` or
//!   `cap == 0` it is dangling (never dereferenced for a read or write, never deallocated);
//!   otherwise it is the start of a live allocation of `cap * size_of::<T>()` bytes made with
//!   [`Self::layout`]`(cap)`.
//! * `len <= cap`, and `ptr[..len]` are initialized values of `T` owned by this buffer.
//! * For zero-sized `T`, `cap` is `usize::MAX` (like `Vec`).
//!
//! The base pointer is stored as a raw pointer, so `as_ptr(&self)` returns a pointer that keeps
//! the allocation's original provenance without creating a `&mut` over the heap data.

use alloc::alloc::{alloc, dealloc, handle_alloc_error, realloc};
use alloc::collections::TryReserveError;
use core::alloc::Layout;
use core::mem::{align_of, size_of};
use core::ptr::{self, NonNull};

pub(crate) struct AlignedBuf<T, const ALIGN: usize> {
    ptr: NonNull<T>,
    len: usize,
    cap: usize,
}

// SAFETY: `AlignedBuf` owns its `T`s exactly like `Vec<T>` does and exposes them only through
// `&self`/`&mut self`.
unsafe impl<T: Send, const ALIGN: usize> Send for AlignedBuf<T, ALIGN> {}
// SAFETY: as above; `&AlignedBuf` only yields `&[T]` and a raw base pointer.
unsafe impl<T: Sync, const ALIGN: usize> Sync for AlignedBuf<T, ALIGN> {}

/// `Vec::<u8>::new().try_reserve(usize::MAX)` is the only way to obtain a real
/// [`TryReserveError`] (it has no public constructor).
fn alloc_err() -> TryReserveError {
    alloc::vec::Vec::<u8>::new()
        .try_reserve(usize::MAX)
        .unwrap_err()
}

impl<T, const ALIGN: usize> AlignedBuf<T, ALIGN> {
    const ZST: bool = size_of::<T>() == 0;
    /// The allocation alignment: at least `ALIGN`, never below `T`'s own.
    const ALLOC_ALIGN: usize = if ALIGN > align_of::<T>() {
        ALIGN
    } else {
        align_of::<T>()
    };
    const CHECK: () = assert!(
        ALIGN.is_power_of_two(),
        "AlignedBuf: ALIGN must be a power of two"
    );

    pub(crate) const fn new() -> Self {
        let () = Self::CHECK;
        Self {
            // A non-null, `ALLOC_ALIGN`-aligned address that is never dereferenced.
            ptr: NonNull::without_provenance(
                core::num::NonZeroUsize::new(Self::ALLOC_ALIGN).unwrap(),
            ),
            len: 0,
            cap: if Self::ZST { usize::MAX } else { 0 },
        }
    }

    #[inline(always)]
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// The base pointer, derived from the stored raw pointer (no reference to the heap data
    /// is created).
    #[inline(always)]
    pub(crate) fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    #[inline(always)]
    pub(crate) fn as_slice(&self) -> &[T] {
        // SAFETY: `ptr[..len]` are initialized values owned by `self` (type invariant) and the
        // pointer is aligned and non-null even when `len == 0`.
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    #[inline(always)]
    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: as in `as_slice`, and `&mut self` makes the access unique.
        unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    fn layout(cap: usize) -> Result<Layout, TryReserveError> {
        let bytes = cap.checked_mul(size_of::<T>()).ok_or_else(alloc_err)?;
        Layout::from_size_align(bytes, Self::ALLOC_ALIGN).map_err(|_| alloc_err())
    }

    /// Ensure room for `additional` more elements, growing geometrically.
    pub(crate) fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        let required = self.len.checked_add(additional).ok_or_else(alloc_err)?;
        if required <= self.cap {
            return Ok(());
        }
        // `Self::ZST` implies `cap == usize::MAX`, so a ZST never reaches this point.
        debug_assert!(!Self::ZST);
        let new_cap = required.max(self.cap.saturating_mul(2)).max(4);
        let new_layout = Self::layout(new_cap)?;
        // SAFETY: `new_layout` has non-zero size (`new_cap >= 4` and `T` is not zero-sized). For
        // `cap > 0`, `ptr` came from `alloc`/`realloc` with `Self::layout(cap)`, and the new
        // size is non-zero and, with the same alignment, valid for `realloc`.
        let raw = unsafe {
            if self.cap == 0 {
                alloc(new_layout)
            } else {
                let old_layout = Self::layout(self.cap)?;
                realloc(
                    self.ptr.as_ptr().cast::<u8>(),
                    old_layout,
                    new_layout.size(),
                )
            }
        };
        let Some(new_ptr) = NonNull::new(raw.cast::<T>()) else {
            return Err(alloc_err());
        };
        self.ptr = new_ptr;
        self.cap = new_cap;
        Ok(())
    }

    fn reserve(&mut self, additional: usize) {
        if self.try_reserve(additional).is_err() {
            let size = self
                .len
                .saturating_add(additional)
                .saturating_mul(size_of::<T>());
            match Layout::from_size_align(size.min(isize::MAX as usize), Self::ALLOC_ALIGN) {
                Ok(layout) => handle_alloc_error(layout),
                Err(_) => panic!("AlignedBuf: capacity overflow"),
            }
        }
    }

    /// Shorten to `new_len`; a no-op if already shorter.
    pub(crate) fn truncate(&mut self, new_len: usize) {
        if new_len >= self.len {
            return;
        }
        let tail_len = self.len - new_len;
        // Shrink `len` first so a panicking `drop` cannot lead to a double drop.
        self.len = new_len;
        // SAFETY: `ptr[new_len..new_len + tail_len]` were initialized and are no longer
        // reachable through `len`.
        unsafe {
            ptr::drop_in_place(ptr::slice_from_raw_parts_mut(
                self.ptr.as_ptr().add(new_len),
                tail_len,
            ));
        }
    }

    /// Resize to `new_len`, filling new elements with clones of `value`.
    pub(crate) fn resize(&mut self, new_len: usize, value: T)
    where
        T: Clone,
    {
        if new_len <= self.len {
            self.truncate(new_len);
            return;
        }
        let extra = new_len - self.len;
        self.reserve(extra);
        // SAFETY: `reserve` guarantees `cap >= len + extra`, so every write below is inside the
        // allocation. `len` is bumped after each write, so a panicking `clone` leaves exactly
        // the initialized prefix to be dropped.
        unsafe {
            let mut p = self.ptr.as_ptr().add(self.len);
            for _ in 1..extra {
                ptr::write(p, value.clone());
                p = p.add(1);
                self.len += 1;
            }
            ptr::write(p, value);
            self.len += 1;
        }
    }
}

impl<T, const ALIGN: usize> Drop for AlignedBuf<T, ALIGN> {
    fn drop(&mut self) {
        // SAFETY: `ptr[..len]` are initialized values owned by `self`; the allocation (if any)
        // was made with `Self::layout(cap)`, which succeeded when it was created.
        unsafe {
            ptr::drop_in_place(ptr::slice_from_raw_parts_mut(self.ptr.as_ptr(), self.len));
            if !Self::ZST && self.cap != 0 {
                let layout =
                    Layout::from_size_align_unchecked(self.cap * size_of::<T>(), Self::ALLOC_ALIGN);
                dealloc(self.ptr.as_ptr().cast::<u8>(), layout);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::String;
    use alloc::vec::Vec;
    use core::sync::atomic::{AtomicUsize, Ordering};

    fn assert_aligned<T, const A: usize>(b: &AlignedBuf<T, A>) {
        assert_eq!(
            b.as_ptr() as usize % A,
            0,
            "base pointer must be {A}-aligned"
        );
    }

    #[test]
    fn empty_is_aligned_and_empty() {
        let b = AlignedBuf::<u8, 64>::new();
        assert_eq!(b.len(), 0);
        assert!(b.as_slice().is_empty());
        assert_aligned(&b);
    }

    #[test]
    fn grows_and_stays_aligned_for_every_step() {
        let mut b = AlignedBuf::<u16, 64>::new();
        for n in [1usize, 3, 4, 5, 31, 32, 33, 100, 1000, 4097] {
            b.resize(n, 7);
            assert_eq!(b.len(), n);
            assert_aligned(&b);
            assert!(b.as_slice().iter().all(|&x| x == 7));
        }
        b.as_mut_slice()[10] = 99;
        b.resize(2000, 7);
        assert_eq!(b.as_slice()[10], 99);
        b.resize(5, 0);
        assert_eq!(b.len(), 5);
    }

    #[test]
    fn alignment_wider_and_narrower_than_the_element() {
        let mut a = AlignedBuf::<u64, 4>::new(); // element alignment wins
        a.resize(9, 1);
        assert_eq!(a.as_ptr() as usize % align_of::<u64>(), 0);
        let mut b = AlignedBuf::<[u8; 3], 32>::new();
        b.resize(11, [1, 2, 3]);
        assert_aligned(&b);
        assert_eq!(b.as_slice()[10], [1, 2, 3]);
    }

    #[test]
    fn zero_sized_elements_never_allocate() {
        let mut b = AlignedBuf::<(), 64>::new();
        b.resize(1_000_000, ());
        assert_eq!(b.len(), 1_000_000);
        b.resize(3, ());
        assert_eq!(b.len(), 3);
        assert_aligned(&b);
    }

    #[test]
    fn try_reserve_reports_overflow_instead_of_panicking() {
        let mut b = AlignedBuf::<u64, 64>::new();
        assert!(b.try_reserve(usize::MAX).is_err());
        assert!(b.try_reserve(usize::MAX / 8 + 1).is_err());
        assert!(b.try_reserve(16).is_ok());
        assert_eq!(b.len(), 0);
    }

    static DROPS: AtomicUsize = AtomicUsize::new(0);

    struct Counted(bool);
    impl Clone for Counted {
        fn clone(&self) -> Self {
            if self.0 {
                panic!("clone failed on purpose");
            }
            Counted(false)
        }
    }
    impl Drop for Counted {
        fn drop(&mut self) {
            DROPS.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn owned_elements_are_dropped_exactly_once() {
        // Plain owning elements: cloned on resize, dropped on truncate and on drop.
        let mut b = AlignedBuf::<String, 32>::new();
        b.resize(5, String::from("abc"));
        assert!(b.as_slice().iter().all(|s| s == "abc"));
        b.resize(2, String::new());
        assert_eq!(b.as_slice().len(), 2);
        drop(b);

        // A panicking `clone` leaves exactly the already-written prefix to drop, once each.
        DROPS.store(0, Ordering::SeqCst);
        let mut c = AlignedBuf::<Counted, 64>::new();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            c.resize(4, Counted(true));
        }));
        assert!(result.is_err());
        assert_eq!(
            c.len(),
            0,
            "clone panicked on the first clone: nothing was written"
        );
        // The by-value `Counted(true)` passed to `resize` was dropped during unwinding.
        assert_eq!(DROPS.load(Ordering::SeqCst), 1);
        drop(c);
        assert_eq!(DROPS.load(Ordering::SeqCst), 1);

        DROPS.store(0, Ordering::SeqCst);
        let mut d = AlignedBuf::<Counted, 64>::new();
        d.resize(3, Counted(false)); // 2 clones + the moved template
        assert_eq!(DROPS.load(Ordering::SeqCst), 0);
        d.resize(1, Counted(false)); // drops 2 stored elements and the unused template
        assert_eq!(DROPS.load(Ordering::SeqCst), 3);
        drop(d); // the one remaining element
        assert_eq!(DROPS.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn slices_match_a_reference_vec() {
        let mut v = Vec::<u8>::new();
        let mut b = AlignedBuf::<u8, 64>::new();
        let mut seed = 12345u32;
        for _ in 0..200 {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let n = (seed >> 8) as usize % 5000;
            let fill = (seed >> 24) as u8;
            v.resize(n, fill);
            b.resize(n, fill);
            assert_eq!(v.as_slice(), b.as_slice());
            assert_aligned(&b);
        }
    }
}
