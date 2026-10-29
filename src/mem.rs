#![forbid(unsafe_code)]
use crate::src::error::Rav1dError::ENOMEM;
use crate::src::error::Rav1dResult;
use parking_lot::Mutex;
use std::collections::TryReserveError;
use std::sync::Arc;

/// Admission for requested native plane bytes, including codec padding.
/// Bounded pictures do not return their allocations to the uncharged pool.
#[cfg(not(feature = "c-ffi"))]
pub(crate) struct PictureBudget {
    limit: usize,
    used: std::sync::atomic::AtomicUsize,
}

#[cfg(not(feature = "c-ffi"))]
impl PictureBudget {
    pub(crate) fn new(limit: usize) -> Self {
        Self {
            limit,
            used: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    pub(crate) fn reserve(self: &Arc<Self>, bytes: usize) -> Rav1dResult<PictureReservation> {
        use std::sync::atomic::Ordering;
        let mut used = self.used.load(Ordering::Acquire);
        loop {
            let next = used
                .checked_add(bytes)
                .filter(|&next| next <= self.limit)
                .ok_or(crate::src::error::Rav1dError::EPICLIMIT)?;
            match self
                .used
                .compare_exchange_weak(used, next, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => break,
                Err(current) => used = current,
            }
        }
        Ok(PictureReservation {
            budget: Arc::clone(self),
            bytes,
        })
    }

    #[cfg(test)]
    pub(crate) fn used(&self) -> usize {
        self.used.load(std::sync::atomic::Ordering::Acquire)
    }
}

/// One charge follows the entire picture through native references and clones.
#[cfg(not(feature = "c-ffi"))]
pub(crate) struct PictureReservation {
    budget: Arc<PictureBudget>,
    bytes: usize,
}

#[cfg(not(feature = "c-ffi"))]
impl Drop for PictureReservation {
    fn drop(&mut self) {
        self.budget
            .used
            .fetch_sub(self.bytes, std::sync::atomic::Ordering::AcqRel);
    }
}

/// Fallible `Arc<T>` allocation.
///
/// On stable Rust, probes allocation feasibility via `Vec::try_reserve`,
/// then delegates to `Arc::new`. This is best-effort (TOCTOU between the
/// probe and the real allocation), but catches genuine OOM for large `T`.
///
/// TODO: Replace body with `Arc::try_new(value).map_err(|_| ENOMEM)`
/// when `allocator_api` stabilizes (rust-lang/rust#32838).
pub(crate) fn try_arc<T>(value: T) -> Rav1dResult<Arc<T>> {
    // Arc<T> layout: two atomic usizes (strong + weak counts) + T
    let needed = std::mem::size_of::<usize>() * 2 + std::mem::size_of::<T>();
    let mut probe = Vec::<u8>::new();
    probe.try_reserve(needed).map_err(|_| ENOMEM)?;
    drop(probe);
    Ok(Arc::new(value))
}

pub struct MemPool<T> {
    bufs: Mutex<Vec<Vec<T>>>,
}

impl<T> MemPool<T> {
    pub const fn new() -> Self {
        Self {
            bufs: Mutex::new(Vec::new()),
        }
    }

    pub fn _pop(&self, size: usize) -> Result<Vec<T>, TryReserveError> {
        if let Some(mut buf) = self.bufs.lock().pop() {
            if size > buf.capacity() {
                buf.try_reserve(size - buf.len())?;
            }
            return Ok(buf);
        }
        let mut buf = Vec::new();
        buf.try_reserve(size)?;
        Ok(buf)
    }

    /// A version of [`Self::pop`] that initializes the [`Vec`].
    /// When `init_value` is `0`, this uses [`alloc_zeroed`] via [`vec!`]
    /// so the OS can skip zero-initialization for fresh pages.
    ///
    /// [`alloc_zeroed`]: std::alloc::alloc_zeroed
    pub fn pop_init(&self, size: usize, init_value: T) -> Result<Vec<T>, TryReserveError>
    where
        T: Copy,
    {
        if let Some(buf) = self.bufs.lock().pop() {
            if size <= buf.len() {
                return Ok(buf);
            }
        }
        let mut buf = Vec::new();
        buf.try_reserve(size)?;
        buf.resize(size, init_value);
        Ok(buf)
    }

    pub fn push(&self, buf: Vec<T>) {
        self.bufs.lock().push(buf);
    }
}

impl<T> Default for MemPool<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(all(test, not(feature = "c-ffi")))]
mod picture_budget_tests {
    use super::PictureBudget;
    use crate::src::error::Rav1dError;
    use std::sync::{Arc, Barrier};

    #[test]
    fn picture_budget_admission_is_atomic_and_overflow_refuses() {
        let budget = Arc::new(PictureBudget::new(40));
        let admitted = Arc::new(Barrier::new(9));
        let release = Arc::new(Barrier::new(9));
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let budget = budget.clone();
                let admitted = admitted.clone();
                let release = release.clone();
                scope.spawn(move || {
                    let lease = budget.reserve(10);
                    admitted.wait();
                    release.wait();
                    drop(lease);
                });
            }
            admitted.wait();
            assert_eq!(budget.used(), 40);
            assert!(matches!(budget.reserve(1), Err(Rav1dError::EPICLIMIT)));
            release.wait();
        });
        assert_eq!(budget.used(), 0);
        let budget = Arc::new(PictureBudget::new(usize::MAX));
        let lease = budget.reserve(usize::MAX).unwrap();
        assert!(matches!(budget.reserve(1), Err(Rav1dError::EPICLIMIT)));
        drop(lease);
        assert_eq!(budget.used(), 0);
        let budget = Arc::new(PictureBudget::new(0));
        assert!(matches!(budget.reserve(1), Err(Rav1dError::EPICLIMIT)));
    }

    #[test]
    fn picture_budget_unwind_releases_the_uncommitted_reservation() {
        let budget = Arc::new(PictureBudget::new(16));
        let result = std::panic::catch_unwind(|| {
            let _lease = budget.reserve(16).unwrap();
            panic!("controlled failure after admission");
        });
        assert!(result.is_err());
        assert_eq!(budget.used(), 0);
        assert!(budget.reserve(16).is_ok());
    }
}
