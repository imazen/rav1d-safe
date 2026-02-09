#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(feature = "c-ffi")]
use crate::src::send_sync_non_null::SendSyncNonNull;
#[cfg(feature = "c-ffi")]
use std::ffi::c_void;
#[cfg(feature = "c-ffi")]
use std::marker::PhantomData;
use std::ops::Deref;
#[cfg(feature = "c-ffi")]
use std::pin::Pin;
#[cfg(feature = "c-ffi")]
use std::ptr::drop_in_place;
#[cfg(feature = "c-ffi")]
use std::ptr::NonNull;

#[cfg(feature = "c-ffi")]
pub type FnFree = unsafe extern "C" fn(ptr: *const u8, cookie: Option<SendSyncNonNull<c_void>>);

/// A `free` "closure", i.e. a [`FnFree`] and an enclosed context [`Self::cookie`].
#[cfg(feature = "c-ffi")]
#[derive(Debug)]
pub struct Free {
    pub free: FnFree,

    /// # Safety
    ///
    /// All accesses to [`Self::cookie`] must be thread-safe
    /// (i.e. [`Self::cookie`] must be [`Send`]` + `[`Sync`]).
    ///
    /// If used from Rust, [`Self::cookie`] is a [`SendSyncNonNull`],
    /// whose constructors ensure this [`Send`]` + `[`Sync`] safety.
    pub cookie: Option<SendSyncNonNull<c_void>>,
}

#[cfg(feature = "c-ffi")]
impl Free {
    /// # Safety
    ///
    /// `ptr` is a [`NonNull`]`<T>` and `free` deallocates it.
    /// It must not be used after this call as it is deallocated.
    pub unsafe fn free(&self, ptr: *mut c_void) {
        // SAFETY: `self` came from `CBox::from_c`,
        // which requires `self.free` to deallocate the `NonNull<T>` passed to it,
        // and `self.cookie` to be passed to it, which it is.
        unsafe { (self.free)(ptr as *const u8, self.cookie) }
    }
}

/// Same as [`core::ptr::Unique`].
#[cfg(feature = "c-ffi")]
#[derive(Debug)]
pub struct Unique<T: ?Sized> {
    pointer: NonNull<T>,
    // NOTE: this marker has no consequences for variance, but is necessary
    // for dropck to understand that we logically own a `T`.
    //
    // For details, see:
    // https://github.com/rust-lang/rfcs/blob/master/text/0769-sound-generic-drop.md#phantom-data
    _marker: PhantomData<T>,
}

#[cfg(feature = "c-ffi")]
/// SAFETY: [`Unique`] is [`Send`] if `T: `[`Send`]
/// because the data it references is unaliased.
unsafe impl<T: Send + ?Sized> Send for Unique<T> {}

#[cfg(feature = "c-ffi")]
/// SAFETY: [`Unique`] is [`Sync`] if `T: `[`Sync`]
unsafe impl<T: Sync + ?Sized> Sync for Unique<T> {}

/// A C/custom [`Box`].
///
/// That is, it is analogous to a [`Box`],
/// but it lets you set a C-style `free` `fn` for deallocation
/// instead of the normal [`Box`] (de)allocator.
/// It can also store a normal [`Box`] as well.
#[derive(Debug)]
pub enum CBox<T: ?Sized> {
    Rust(Box<T>),
    #[cfg(feature = "c-ffi")]
    C {
        /// # SAFETY:
        ///
        /// * Never moved.
        /// * Valid to dereference.
        /// * `free`d by the `free` `fn` ptr below.
        data: Unique<T>,
        free: Free,
    },
}

// Without c-ffi, CBox is always CBox::Rust(Box<T>), so AsRef just delegates to Box.
#[cfg(not(feature = "c-ffi"))]
impl<T: ?Sized> AsRef<T> for CBox<T> {
    fn as_ref(&self) -> &T {
        let Self::Rust(r#box) = self;
        r#box.as_ref()
    }
}

#[cfg(feature = "c-ffi")]
impl<T: ?Sized> AsRef<T> for CBox<T> {
    fn as_ref(&self) -> &T {
        match self {
            Self::Rust(r#box) => r#box.as_ref(),
            // SAFETY: `data` is a `Unique<T>`, which behaves as if it were a `T`,
            // so we can take `&` references of it.
            // Furthermore, `data` is never moved and is valid to dereference,
            // so this reference can live as long as `CBox` and still be valid the whole time.
            Self::C { data, .. } => unsafe { data.pointer.as_ref() },
        }
    }
}

impl<T: ?Sized> Deref for CBox<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

// Without c-ffi, CBox::Rust(Box<T>) doesn't need a custom Drop — Box handles it.
// Only the C variant needs custom Drop to call the C free function.
#[cfg(feature = "c-ffi")]
impl<T: ?Sized> Drop for CBox<T> {
    fn drop(&mut self) {
        match self {
            Self::Rust(_) => {} // Drop normally.
            Self::C { data, free, .. } => {
                let ptr = data.pointer.as_ptr();
                // SAFETY: See below.
                // The [`FnFree`] won't run Rust's `fn drop`,
                // so we have to do this ourselves first.
                unsafe { drop_in_place(ptr) };
                let ptr = ptr.cast();
                // SAFETY: See safety docs on [`Self::data`] and [`Self::from_c`].
                unsafe { free.free(ptr) }
            }
        }
    }
}

impl<T: ?Sized> CBox<T> {
    /// # Safety
    ///
    /// `data` must be valid to dereference
    /// until `free.free` is called on it, which must deallocate it.
    /// `free.free` is always called with `free.cookie`,
    /// which must be accessed thread-safely.
    #[cfg(feature = "c-ffi")]
    pub unsafe fn from_c(data: NonNull<T>, free: Free) -> Self {
        Self::C {
            data: Unique {
                pointer: data,
                _marker: PhantomData,
            },
            free,
        }
    }

    pub fn from_box(data: Box<T>) -> Self {
        Self::Rust(data)
    }

    /// Pin this CBox.
    ///
    /// Only available with c-ffi, where CArc uses `Arc<Pin<CBox<T>>>`.
    /// Without c-ffi, CArc uses `Arc<Box<T>>` directly and doesn't need Pin.
    #[cfg(feature = "c-ffi")]
    pub fn into_pin(self) -> Pin<Self> {
        // With c-ffi, CBox::C variant contains Unique<T> which is !Unpin,
        // but the data is never moved until Drop, making Pin sound.
        // SAFETY:
        // If `self` is `Self::Rust`, `Box` can be pinned.
        // If `self` is `Self::C`, `data` is never moved until [`Self::drop`].
        #[allow(unsafe_code)]
        unsafe {
            Pin::new_unchecked(self)
        }
    }
}

#[cfg(feature = "c-ffi")]
impl<T: ?Sized> From<CBox<T>> for Pin<CBox<T>> {
    fn from(value: CBox<T>) -> Self {
        value.into_pin()
    }
}
