//! Const construction without embedding the large tracker in every buffer.
//!
//! Each checked instance owns exactly one tracker. A lazy instance publishes it
//! through `spin::Once` before any registration. Its variant and allocation stay
//! fixed while shared references exist. Eager instances avoid the Once load on
//! every borrow; both variants run the same admission and retirement protocol.

use super::checked::BorrowTracker;
use alloc::boxed::Box;

pub(super) enum TrackerStorage {
    Unchecked,
    Eager(Box<BorrowTracker>),
    Lazy(spin::Once<Box<BorrowTracker>>),
}

#[cfg(all(test, not(disjoint_mut_loom)))]
mod tests {
    use super::*;
    use std::sync::Barrier;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn concurrent_first_call_publishes_exactly_one_tracker() {
        let storage = TrackerStorage::new();
        assert!(storage.is_checked());
        assert!(storage.get().is_none());
        let start = Barrier::new(4);
        let initializers = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            let threads: alloc::vec::Vec<_> = (0..4)
                .map(|_| {
                    scope.spawn(|| {
                        start.wait();
                        storage
                            .get_or_init(|| {
                                initializers.fetch_add(1, Ordering::SeqCst);
                                std::thread::yield_now();
                                1024
                            })
                            .unwrap() as *const BorrowTracker as usize
                    })
                })
                .collect();
            let addresses: alloc::vec::Vec<_> =
                threads.into_iter().map(|t| t.join().unwrap()).collect();
            assert!(addresses.iter().all(|p| *p == addresses[0]));
        });
        assert_eq!(initializers.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn eager_storage_never_calls_the_lazy_initializer() {
        let storage = TrackerStorage::eager(1024);
        assert!(core::ptr::eq(
            storage.get().unwrap(),
            storage
                .get_or_init(|| panic!("eager path initialized twice"))
                .unwrap(),
        ));
    }
}

impl TrackerStorage {
    pub(super) const fn new() -> Self {
        Self::Lazy(spin::Once::new())
    }

    pub(super) fn eager(len: usize) -> Self {
        Self::Eager(Box::new(BorrowTracker::new(len)))
    }

    pub(super) const fn is_checked(&self) -> bool {
        !matches!(self, Self::Unchecked)
    }

    #[inline]
    pub(super) fn get_or_init(&self, len: impl FnOnce() -> usize) -> Option<&BorrowTracker> {
        match self {
            Self::Unchecked => None,
            Self::Eager(tracker) => Some(tracker),
            Self::Lazy(cell) => Some(cell.call_once(|| Box::new(BorrowTracker::new(len())))),
        }
    }

    /// Retirement and poisoning must only use an already-published tracker.
    #[inline]
    pub(super) fn get(&self) -> Option<&BorrowTracker> {
        match self {
            Self::Unchecked => None,
            Self::Eager(tracker) => Some(tracker),
            Self::Lazy(cell) => cell.get().map(Box::as_ref),
        }
    }

    /// No usable guard can survive this exclusive borrow. Uninitialized lazy
    /// storage stays lazy; its eventual initializer will read the new length.
    #[inline]
    pub(super) fn get_mut(&mut self) -> Option<&mut BorrowTracker> {
        match self {
            Self::Unchecked => None,
            Self::Eager(tracker) => Some(tracker),
            Self::Lazy(cell) => cell.get_mut().map(Box::as_mut),
        }
    }
}
