#![forbid(unsafe_code)]
use parking_lot::Mutex;
use std::collections::TryReserveError;

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
