#![deny(unsafe_op_in_unsafe_fn)]

use crate::src::c_box::CBox;
use crate::src::error::Rav1dResult;
#[cfg(feature = "c-ffi")]
use std::marker::PhantomData;
use std::ops::Deref;
#[cfg(feature = "c-ffi")]
use std::pin::Pin;
#[cfg(feature = "c-ffi")]
use std::ptr::NonNull;
use std::sync::Arc;

#[cfg(feature = "c-ffi")]
pub fn arc_into_raw<T: ?Sized>(arc: Arc<T>) -> NonNull<T> {
    let raw = Arc::into_raw(arc).cast_mut();
    // SAFETY: [`Arc::into_raw`] never returns null.
    unsafe { NonNull::new_unchecked(raw) }
}

/// A C/custom [`Arc`].
///
/// That is, it is analogous to an [`Arc`],
/// but it lets you set a C-style `free` `fn` for deallocation
/// instead of the normal [`Box`] (de)allocator.
/// It can also store a normal [`Box`] as well.
///
/// ## c-ffi mode
///
/// With c-ffi, a [`StableRef`] raw pointer is stored inline for performance
/// (avoids double indirection through [`Arc`] and [`CBox`]). The [`CBox`] is
/// [`Pin`]ned to keep this self-referential pointer sound.
///
/// ## Safe mode (default)
///
/// Without c-ffi, [`CBox`] is always `Rust(Box<T>)`, so we unwrap it
/// and store `Arc<Box<T>>` directly — no Pin, no raw pointers.
/// Sub-slice views for `CArc<[T]>` are tracked via `(start, end)` indices.
#[cfg(not(feature = "c-ffi"))]
#[derive(Debug)]
pub struct CArc<T: ?Sized> {
    /// Without c-ffi, CBox is always Rust(Box<T>), so we skip the CBox/Pin
    /// wrapper and just store Arc<Box<T>> for a clean, safe deref chain.
    owner: Arc<Box<T>>,

    /// For `CArc<[T]>`: tracks the current sub-slice view as `(start, end)`.
    /// `None` means "full slice" (the default after construction).
    /// For non-slice types (e.g. `CArc<u8>`), always `None` and unused.
    view: Option<(usize, usize)>,
}

#[cfg(feature = "c-ffi")]
#[derive(Debug)]
pub struct CArc<T: ?Sized> {
    owner: Arc<Pin<CBox<T>>>,

    /// The same as [`Self::stable_ref`] but it never changes.
    #[cfg(debug_assertions)]
    base_stable_ref: StableRef<T>,

    stable_ref: StableRef<T>,
}

/// A stable reference, stored as a raw ptr.
///
/// # Safety
///
/// The raw ptr of a [`StableRef`] must have a stable address.
/// Even if `T`'s owning type, e.x. a [`Box`]`<T>`, is moved,
/// ptrs to `T` must remain valid and thus "stable".
///
/// Thus, it can be stored relative to its owner.
#[cfg(feature = "c-ffi")]
#[derive(Debug)]
struct StableRef<T: ?Sized>(NonNull<T>);

#[cfg(feature = "c-ffi")]
impl<T: ?Sized> Clone for StableRef<T> {
    fn clone(&self) -> Self {
        *self
    }
}

#[cfg(feature = "c-ffi")]
impl<T: ?Sized> Copy for StableRef<T> {}

#[cfg(feature = "c-ffi")]
/// SAFETY: [`StableRef`]`<T>`, if it follows its safety guarantees, is essentially a `&T`/`&mut T`, which is [`Send`] if `T: `[`Send`]`.
#[allow(unsafe_code)]
unsafe impl<T: Send + ?Sized> Send for StableRef<T> {}

#[cfg(feature = "c-ffi")]
/// SAFETY: [`StableRef`]`<T>`, if it follows its safety guarantees, is essentially a `&T`/`&mut T`, which is [`Sync`] if `T: `[`Sync`].
#[allow(unsafe_code)]
unsafe impl<T: Send + ?Sized> Sync for StableRef<T> {}

// ===== AsRef / Deref =====

#[cfg(feature = "c-ffi")]
impl<T: ?Sized> AsRef<T> for CArc<T> {
    #[allow(unsafe_code)]
    fn as_ref(&self) -> &T {
        #[cfg(debug_assertions)]
        {
            use std::mem;
            use std::ptr;
            use to_method::To;

            let real_ref = (*self.owner).as_ref().get_ref();
            assert_eq!(real_ref.to::<NonNull<T>>(), self.base_stable_ref.0);

            let real_ptr = ptr::from_ref(real_ref);
            let stable_ptr = self.stable_ref.0.as_ptr().cast_const();
            let [real_address, stable_address] =
                [real_ptr, stable_ptr].map(|ptr| ptr.cast::<()>() as isize);
            let offset = stable_address - real_address;
            let len = mem::size_of_val(real_ref);
            if offset < 0 || offset > len as isize {
                panic!(
                    "CArc::stable_ref is out of bounds:
    real_ref: {real_ptr:?}
    stable_ref: {stable_ptr:?}
    offset: {offset}
    len: {len}"
                );
            }
        }

        // SAFETY: [`Self::stable_ref`] is a ptr
        // derived from [`Self::owner`]'s through [`CBox::as_ref`]
        // and is thus safe to dereference.
        unsafe { self.stable_ref.0.as_ref() }
    }
}

#[cfg(feature = "c-ffi")]
impl<T: ?Sized> Deref for CArc<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

// Without c-ffi: safe deref through Arc<Box<T>>.
// Separate impls for sized types vs slices because slice views
// need sub-slicing while sized types just deref directly.

#[cfg(not(feature = "c-ffi"))]
impl Deref for CArc<u8> {
    type Target = u8;

    fn deref(&self) -> &u8 {
        &self.owner
    }
}

#[cfg(not(feature = "c-ffi"))]
impl<T> Deref for CArc<[T]> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        let full: &[T] = &self.owner;
        match self.view {
            None => full,
            Some((start, end)) => &full[start..end],
        }
    }
}

// ===== Clone =====

#[cfg(not(feature = "c-ffi"))]
impl<T: ?Sized> Clone for CArc<T> {
    fn clone(&self) -> Self {
        Self {
            owner: self.owner.clone(),
            view: self.view,
        }
    }
}

#[cfg(feature = "c-ffi")]
impl<T: ?Sized> Clone for CArc<T> {
    fn clone(&self) -> Self {
        let Self {
            owner,
            #[cfg(debug_assertions)]
            base_stable_ref,
            stable_ref,
        } = self;
        Self {
            owner: owner.clone(),
            #[cfg(debug_assertions)]
            base_stable_ref: base_stable_ref.clone(),
            stable_ref: stable_ref.clone(),
        }
    }
}

// ===== Construction =====

#[cfg(not(feature = "c-ffi"))]
impl<T: ?Sized> CArc<T> {
    pub fn wrap(owner: CBox<T>) -> Rav1dResult<Self> {
        let CBox::Rust(boxed) = owner;
        Ok(Self {
            owner: Arc::new(boxed), // TODO fallible allocation
            view: None,
        })
    }
}

#[cfg(feature = "c-ffi")]
impl<T: ?Sized> From<Arc<Pin<CBox<T>>>> for CArc<T> {
    fn from(owner: Arc<Pin<CBox<T>>>) -> Self {
        let stable_ref = StableRef((*owner).as_ref().get_ref().into());
        Self {
            owner,
            #[cfg(debug_assertions)]
            base_stable_ref: stable_ref,
            stable_ref,
        }
    }
}

#[cfg(feature = "c-ffi")]
impl<T: ?Sized> CArc<T> {
    pub fn wrap(owner: CBox<T>) -> Rav1dResult<Self> {
        let owner = Arc::new(owner.into_pin()); // TODO fallible allocation
        Ok(owner.into())
    }
}

// ===== RawArc / RawCArc: c-ffi only =====

/// An opaque, raw [`Arc`] ptr.
///
/// See [`Arc::from_raw`], [`Arc::into_raw`], and [`arc_into_raw`].
///
/// The [`PhantomData`] is so it can be FFI-safe
/// without `T` having to be `#[repr(C)]`,
/// which it doesn't since it's opaque,
/// while still keeping `T` in the type.
#[cfg(feature = "c-ffi")]
#[repr(transparent)]
pub struct RawArc<T>(NonNull<PhantomData<T>>);

/// We need a manual `impl` since we don't require `T: Clone`.
///
/// # Safety
///
/// Note that this [`RawArc::clone`] does not call [`Arc::clone`],
/// since implicit clones/copies are expected to be done outside of Rust,
/// for which there is no way to force [`RawArc::clone`] to be called.
/// Instead, [`RawArc::as_ref`] and [`RawArc::into_arc`] are `unsafe`,
/// and require [`RawArc::clone`]s (actual explicit calls
/// or implicit ones outside of Rust) to respect the rules of [`Arc`].
#[cfg(feature = "c-ffi")]
impl<T> Clone for RawArc<T> {
    fn clone(&self) -> Self {
        *self
    }
}

#[cfg(feature = "c-ffi")]
impl<T> Copy for RawArc<T> {}

#[cfg(feature = "c-ffi")]
impl<T> RawArc<T> {
    pub fn from_arc(arc: Arc<T>) -> Self {
        Self(arc_into_raw(arc).cast())
    }

    /// # Safety
    ///
    /// The [`RawArc`] must be originally from [`Self::from_arc`].
    ///
    /// This must not be called after [`Self::into_arc`],
    /// including on [`Clone`]s.
    pub unsafe fn as_ref(&self) -> &T {
        unsafe { self.0.cast().as_ref() }
    }

    /// # Safety
    ///
    /// The [`RawArc`] must be originally from [`Self::from_arc`].
    ///
    /// After calling this, the [`RawArc`] and [`Clone`]s of it may not be used anymore.
    pub unsafe fn into_arc(self) -> Arc<T> {
        let raw = self.0.cast().as_ptr();
        unsafe { Arc::from_raw(raw) }
    }
}

#[cfg(feature = "c-ffi")]
#[repr(transparent)]
pub struct RawCArc<T: ?Sized>(RawArc<Pin<CBox<T>>>);

#[cfg(feature = "c-ffi")]
impl<T: ?Sized> CArc<T> {
    /// Convert into a raw, opaque form suitable for C FFI.
    pub fn into_raw(self) -> RawCArc<T> {
        RawCArc(RawArc::from_arc(self.owner))
    }

    /// # Safety
    ///
    /// The [`RawCArc`] must be originally from [`Self::into_raw`].
    pub unsafe fn from_raw(raw: RawCArc<T>) -> Self {
        let owner = unsafe { raw.0.into_arc() };
        owner.into()
    }
}

// ===== Slice operations =====

#[cfg(not(feature = "c-ffi"))]
impl<T> CArc<[T]> {
    /// Narrow the view of this `CArc<[T]>` to a sub-slice.
    ///
    /// The full slice stays owned by the [`Arc`],
    /// but [`Deref`] will return only the sub-slice.
    pub fn slice_in_place<I>(&mut self, range: I)
    where
        I: std::slice::SliceIndex<[T], Output = [T]>,
    {
        // Get the current view
        let full: &[T] = &self.owner;
        let (cur_start, cur_end) = self.view.unwrap_or((0, full.len()));
        let current = &full[cur_start..cur_end];

        // Apply the new range to the current view
        let sub = &current[range];

        // Compute new absolute indices via pointer arithmetic (safe integer math)
        let byte_offset = sub.as_ptr() as usize - current.as_ptr() as usize;
        let elem_offset = byte_offset / std::mem::size_of::<T>();
        let new_start = cur_start + elem_offset;
        let new_end = new_start + sub.len();

        self.view = Some((new_start, new_end));
    }

    pub fn split_at(this: Self, mid: usize) -> (Self, Self) {
        let mut first = this.clone();
        let mut second = this;
        first.slice_in_place(..mid);
        second.slice_in_place(mid..);
        (first, second)
    }
}

#[cfg(feature = "c-ffi")]
impl<T> CArc<[T]> {
    /// Slice [`Self::stable_ref`] in-place.
    pub fn slice_in_place<I>(&mut self, range: I)
    where
        I: std::slice::SliceIndex<[T], Output = [T]>,
    {
        self.stable_ref = StableRef(self.as_ref()[range].into());
    }

    pub fn split_at(this: Self, mid: usize) -> (Self, Self) {
        let mut first = this.clone();
        let mut second = this;
        first.slice_in_place(..mid);
        second.slice_in_place(mid..);
        (first, second)
    }
}

impl<T> CArc<[T]>
where
    T: Default + 'static,
{
    #[cfg(feature = "c-ffi")]
    pub fn zeroed_slice(size: usize) -> Rav1dResult<Self> {
        let owned_slice = (0..size).map(|_| Default::default()).collect::<Box<[_]>>(); // TODO fallible allocation
        Self::wrap(CBox::from_box(owned_slice))
    }
}
