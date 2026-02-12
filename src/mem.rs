#![forbid(unsafe_code)]
use crate::src::error::Rav1dError::ENOMEM;
use crate::src::error::Rav1dResult;
use parking_lot::Mutex;
use std::collections::TryReserveError;
use std::sync::Arc;

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
