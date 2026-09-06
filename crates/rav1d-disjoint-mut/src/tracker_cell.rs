//! Lock-protected metadata with access lifetimes visible to Loom.
//!
//! Native builds use ordinary UnsafeCell pointers. The model build keeps a
//! Loom pointer alive for the entire reference lifetime, including until an
//! explicit unlock. Instrumenting just the lock would miss metadata races.

use core::marker::PhantomData;
use core::ops::{Deref, DerefMut};

#[cfg(not(disjoint_mut_loom))]
use core::cell::UnsafeCell;
#[cfg(disjoint_mut_loom)]
use loom::cell::UnsafeCell;

pub(super) struct Cell<T>(UnsafeCell<T>);

impl<T> Cell<T> {
    #[cfg(not(disjoint_mut_loom))]
    pub(super) const fn new(value: T) -> Self {
        Self(UnsafeCell::new(value))
    }

    #[cfg(disjoint_mut_loom)]
    pub(super) fn new(value: T) -> Self {
        Self(UnsafeCell::new(value))
    }

    /// # Safety
    /// Exclude writers for the returned guard's entire lifetime.
    #[inline(always)]
    pub(super) unsafe fn read(&self) -> Read<'_, T> {
        Read {
            ptr: self.0.get(),
            owner: PhantomData,
        }
    }

    /// # Safety
    /// Exclude all other access for the returned guard's entire lifetime.
    #[inline(always)]
    pub(super) unsafe fn write(&self) -> Write<'_, T> {
        Write {
            #[cfg(not(disjoint_mut_loom))]
            ptr: self.0.get(),
            #[cfg(disjoint_mut_loom)]
            ptr: self.0.get_mut(),
            owner: PhantomData,
        }
    }
}

pub(super) struct Read<'a, T> {
    #[cfg(not(disjoint_mut_loom))]
    ptr: *mut T,
    #[cfg(disjoint_mut_loom)]
    ptr: loom::cell::ConstPtr<T>,
    owner: PhantomData<&'a Cell<T>>,
}

impl<T> Deref for Read<'_, T> {
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &T {
        // SAFETY: the constructor requires exclusion of writers and `owner`
        // bounds the pointer by the lifetime of its cell.
        #[cfg(not(disjoint_mut_loom))]
        unsafe {
            &*self.ptr
        }
        #[cfg(disjoint_mut_loom)]
        unsafe {
            self.ptr.deref()
        }
    }
}

pub(super) struct Write<'a, T> {
    #[cfg(not(disjoint_mut_loom))]
    ptr: *mut T,
    #[cfg(disjoint_mut_loom)]
    ptr: loom::cell::MutPtr<T>,
    owner: PhantomData<&'a Cell<T>>,
}

// Explicitly dropping this guard marks the end of metadata access before an
// unlock. Native code has no cleanup; the Loom pointer field ends its tracked
// access when it is dropped immediately after this body.
impl<T> Drop for Write<'_, T> {
    #[inline(always)]
    fn drop(&mut self) {}
}

impl<T> Deref for Write<'_, T> {
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &T {
        // SAFETY: exclusive access was established by the constructor.
        #[cfg(not(disjoint_mut_loom))]
        unsafe {
            &*self.ptr
        }
        #[cfg(disjoint_mut_loom)]
        unsafe {
            self.ptr.deref()
        }
    }
}

impl<T> DerefMut for Write<'_, T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: exclusive access was established by the constructor. This
        // reference cannot outlive the exclusive borrow of the guard.
        #[cfg(not(disjoint_mut_loom))]
        unsafe {
            &mut *self.ptr
        }
        #[cfg(disjoint_mut_loom)]
        unsafe {
            self.ptr.deref()
        }
    }
}
