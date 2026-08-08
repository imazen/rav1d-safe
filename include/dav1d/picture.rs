#![deny(unsafe_op_in_unsafe_fn)]

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, Ordering};

/// Global latch: true once any decoder in this process has used tile threading
/// (n_tc > 1). When true, compact_read/compact_write_back and
/// `with_pixel_guard_*` use per-row guards to avoid stride-padding overlap.
/// When false, they use a single wide guard, which is only sound with no
/// concurrent tile workers.
static TILE_THREADING: AtomicBool = AtomicBool::new(false);

/// Latch tile threading on. Called from decoder initialization.
///
/// MONOTONE ON PURPOSE — never stores `false`. This is process-global state
/// but decoders are per-instance, so a store of `false` from a
/// single-threaded `rav1d_open` used to clobber the flag for every
/// CONCURRENTLY LIVE multi-threaded decoder: their tile workers then took the
/// wide `narrow_guard` path, whose extent is `(h-1) * stride + w` — a 1x16
/// intra left-edge column claims 16,321 pixels to read 16 — and any
/// concurrent row write inside that span is a real conflict. In a checked
/// build that surfaced as a spurious `overlapping DisjointMut` panic in
/// `rav1d_prepare_intra_edges`; in an `unchecked` build it is an undetected
/// data race on picture memory.
///
/// Measured on this branch before the latch: 6 concurrent
/// `tile_threading_overlap --ignored` processes (whose
/// `single_threaded_no_panic` case opens an `n_tc == 1` decoder alongside
/// `multi_threaded_cdef_lpf_race`'s threaded ones) panicked in 8-9 of 24
/// runs. After: 0 of 24. See `benchmarks/p2_kernels_2026-08-07.meta`.
///
/// The cost of latching instead of tracking per decoder is that a process
/// which has ever opened a threaded decoder keeps the per-row path for its
/// single-threaded ones too. That is the correct direction to be wrong in,
/// and a purely single-threaded process never sets the latch at all.
pub fn set_tile_threading(active: bool) {
    if active {
        TILE_THREADING.store(true, Ordering::Relaxed);
    }
}

/// Check if tile threading is active.
pub fn tile_threading_active() -> bool {
    TILE_THREADING.load(Ordering::Relaxed)
}

thread_local! {
    /// Reusable scratch buffers backing [`WithOffset::compact_read_per_row`]
    /// (and the pristine copy kept by the loopfilter's diff write-back).
    /// Pulled on read, returned via [`recycle_compact_scratch`] after the
    /// compact buffer has been written back / consumed.
    ///
    /// Under tile threading the loop filter / ipred / `with_pixel_guard_*`
    /// paths materialize a per-edge / per-block compact buffer; allocating one
    /// `Vec` per call produced ~3M alloc+free pairs decoding a single 8K frame
    /// (issue #17) — cheap on glibc, pathological on the Windows allocator.
    /// Reusing buffers per thread eliminates that churn. It is per-thread,
    /// so each tile thread owns its own buffers with no cross-tile aliasing —
    /// tile threading stays enabled. Two slots because the loopfilter's
    /// tile-threading path holds a working copy and a pristine copy at once.
    static COMPACT_SCRATCH: Cell<[Option<Vec<u8>>; 2]> = const { Cell::new([None, None]) };
}

/// Take a scratch buffer from the thread-local pool (empty `Vec` if the pool
/// is dry). Pair with [`recycle_compact_scratch`].
#[inline]
pub(crate) fn take_compact_scratch() -> Vec<u8> {
    COMPACT_SCRATCH.with(|c| {
        let mut slots = c.take();
        let buf = slots[0].take().or_else(|| slots[1].take());
        c.set(slots);
        buf.unwrap_or_default()
    })
}

/// Return a compact scratch buffer to the thread-local pool so the next
/// [`WithOffset::compact_read_per_row`] on this thread can reuse the allocation.
///
/// Pairs with the `Vec` that `compact_read_per_row` returns. Calling this is an
/// optimization, not a requirement: simply dropping the `Vec` is always correct
/// (the next read just allocates a fresh buffer). If a panic unwinds past a
/// pending recycle the buffer is freed normally — no unsoundness, the pool is
/// merely empty for the next call (same contract as the MC mid-buffer pool).
///
/// Crate-internal: this is decoder plumbing, not part of the public API.
#[inline]
pub(crate) fn recycle_compact_scratch(buf: Vec<u8>) {
    // Keep the larger buffers so re-entrant reads (a read nested inside a
    // `with_pixel_guard_*` closure) don't shrink the pool.
    COMPACT_SCRATCH.with(|c| {
        let mut slots = c.take();
        // Fill an empty slot, else replace the smallest slot if `buf` is larger.
        let smallest = if slots[0].is_none()
            || slots[1].is_some()
                && slots[0].as_ref().map(Vec::capacity) < slots[1].as_ref().map(Vec::capacity)
        {
            0
        } else {
            1
        };
        match &slots[smallest] {
            Some(existing) if existing.capacity() >= buf.capacity() => {}
            _ => slots[smallest] = Some(buf),
        }
        c.set(slots);
    });
}

use crate::include::common::bitdepth::BitDepth;

/// Execute a closure with mutable byte access to a w×h pixel block.
///
/// The closure receives `(bytes, offset, stride)` — a mutable byte slice, the
/// BYTE offset to the first pixel, and the byte stride between rows.
///
/// This is a closure-shaped front end for [`WithOffset::block_mut`], which owns
/// the single-threaded-vs-tile-threaded policy. The two exist because call sites
/// come in two shapes — a closure suits the x86-64 and generic dispatchers, an
/// RAII guard suits the aarch64/wasm ones that need the slice to outlive a
/// borrow — but there must be exactly ONE implementation of the policy behind
/// them. When there were two, the aarch64 half silently kept reserving inter-row
/// gaps long after the x86-64 half stopped, which is the bug this whole change
/// exists to fix.
#[inline]
pub fn with_pixel_guard_mut<BD: BitDepth, R>(
    pic: &crate::src::with_offset::WithOffset<&Rav1dPictureDataComponent>,
    w: usize,
    h: usize,
    f: impl FnOnce(&mut [u8], usize, isize) -> R,
) -> R {
    let mut block = pic.block_mut::<BD>(w, h);
    // `BlockMut::base()` is a PIXEL index — 0 for a compact block, and the
    // last row's index for a direct block on a negative-stride picture. This
    // closure's contract is a BYTE offset, so the scaling happens here rather
    // than inside `BlockMut`, whose own users index the byte slice directly.
    let offset = block.base() * core::mem::size_of::<BD::Pixel>();
    let stride = block.byte_stride();
    // `block` is dropped at the end of this expression, i.e. after `f` has run:
    // a compact block therefore writes back before the caller observes `R`,
    // exactly as the previous open-coded version did.
    f(block.as_mut_bytes(), offset, stride)
}

/// Execute a closure with read-only byte access to a w×h pixel block.
///
/// In single-threaded mode: zero-copy via `narrow_guard`.
/// In multi-threaded mode: copies into a compact buffer with per-row guards
/// (one guard per row, each covering exactly `w` pixels) so the helper is
/// safe to use alongside concurrent tile-thread writes on adjacent blocks.
///
/// The closure receives `(bytes, offset, stride)` — an immutable byte slice,
/// the byte offset to the first pixel, and the byte stride between rows.
///
/// Counterpart to [`with_pixel_guard_mut`]; see [`feedback_mt_safe_batching`]
/// for the rationale on why wide guards (`strided_slice_mut` / `narrow_guard`)
/// are unsafe under tile threading even in the immutable case (they cover
/// `(h-1)*stride + w` bytes and overlap with adjacent tiles' mutable ranges).
#[inline]
pub fn with_pixel_guard_immut<BD: BitDepth, R>(
    pic: &crate::src::with_offset::WithOffset<&Rav1dPictureDataComponent>,
    w: usize,
    h: usize,
    f: impl FnOnce(&[u8], usize, isize) -> R,
) -> R {
    use crate::src::strided::Strided as _;
    let pixel_size = core::mem::size_of::<BD::Pixel>();
    if tile_threading_active() {
        let (buf, byte_stride) = pic.compact_read_per_row::<BD>(w, h);
        let result = f(&buf, 0, byte_stride as isize);
        recycle_compact_scratch(buf);
        result
    } else {
        let (guard, base) = pic.narrow_guard::<BD>(w, h);
        let bytes = guard.as_bytes();
        let offset = base * pixel_size;
        let stride = pic.stride();
        f(bytes, offset, stride)
    }
}
#[cfg(feature = "c-ffi")]
use crate::include::common::validate::validate_input;
#[cfg(feature = "c-ffi")]
use crate::include::dav1d::common::Dav1dDataProps;
use crate::include::dav1d::common::Rav1dDataProps;
use crate::include::dav1d::headers::DRav1d;
use crate::include::dav1d::headers::Dav1dFrameHeader;
use crate::include::dav1d::headers::Dav1dITUTT35;
use crate::include::dav1d::headers::Dav1dPixelLayout;
use crate::include::dav1d::headers::Dav1dSequenceHeader;
use crate::include::dav1d::headers::Rav1dContentLightLevel;
use crate::include::dav1d::headers::Rav1dFrameHeader;
use crate::include::dav1d::headers::Rav1dITUTT35;
use crate::include::dav1d::headers::Rav1dMasteringDisplay;
use crate::include::dav1d::headers::Rav1dPixelLayout;
use crate::include::dav1d::headers::Rav1dSequenceHeader;
#[cfg(feature = "c-ffi")]
use crate::src::assume::assume;
#[cfg(feature = "c-ffi")]
use crate::src::c_arc::RawArc;
use crate::src::disjoint_mut::DisjointImmutGuard;
use crate::src::disjoint_mut::DisjointMut;
use crate::src::disjoint_mut::DisjointMutGuard;
#[cfg(feature = "c-ffi")]
use crate::src::disjoint_mut::ExternalAsMutPtr;
use crate::src::disjoint_mut::SliceBounds;
#[cfg(feature = "c-ffi")]
use crate::src::error::Dav1dResult;
use crate::src::error::Rav1dError;
#[cfg(feature = "c-ffi")]
use crate::src::error::Rav1dError::EINVAL;
use crate::src::error::Rav1dResult;
#[cfg(not(feature = "c-ffi"))]
use crate::src::mem::MemPool;
#[cfg(asm_fn_ptrs)]
use crate::src::pixels::Pixels;
#[cfg(feature = "c-ffi")]
use crate::src::send_sync_non_null::SendSyncNonNull;
use crate::src::strided::Strided;
use crate::src::with_offset::WithOffset;
#[allow(non_camel_case_types)]
type ptrdiff_t = isize;
#[cfg(feature = "c-ffi")]
#[allow(non_camel_case_types)]
type uintptr_t = usize;
#[cfg(not(feature = "c-ffi"))]
use rav1d_disjoint_mut::PicBuf;
use std::array;
use std::ffi::c_int;
#[cfg(feature = "c-ffi")]
use std::ffi::c_void;
use std::iter;
use std::mem;
#[cfg(feature = "c-ffi")]
use std::ptr::NonNull;
use std::sync::Arc;
#[cfg(feature = "c-ffi")]
use to_method::To as _;
use zerocopy::FromBytes;
use zerocopy::Immutable;
use zerocopy::IntoBytes;
use zerocopy::KnownLayout;

pub(crate) const RAV1D_PICTURE_ALIGNMENT: usize = 64;
pub const DAV1D_PICTURE_ALIGNMENT: usize = RAV1D_PICTURE_ALIGNMENT;

/// A raw pointer to picture data that is `Send + Sync`.
///
/// Uses `SendSyncNonNull` from `rav1d-disjoint-mut` so that `Send + Sync` are
/// automatically derived without any `unsafe impl` in this crate.
///
/// Thread safety rationale:
///
/// 1. **Owned buffers**: Points into a `Vec<u8>` stored in
///    the same `Rav1dPictureData` struct, behind `Arc`. The Vec cannot be
///    grown or reallocated through `&Rav1dPictureData`, so the pointer stays
///    valid. Concurrent access is tracked by `DisjointMut`.
///
/// 2. **Borrowed scratch buffers** (`wrap_buf`): Points into a `&mut [BD::Pixel]`
///    that outlives the component. These are single-threaded temporaries in
///    `recon.rs` — they are never shared across threads.
#[cfg(feature = "c-ffi")]
#[derive(Clone, Copy)]
#[repr(transparent)]
struct PicDataPtr(SendSyncNonNull<u8>);

#[cfg(feature = "c-ffi")]
#[allow(unsafe_code)]
impl PicDataPtr {
    /// Create a dangling pointer with [`RAV1D_PICTURE_ALIGNMENT`] alignment.
    fn dangling_aligned() -> Self {
        // SAFETY: NonNull::dangling() is Send+Sync-safe (no real data behind it).
        Self(unsafe {
            SendSyncNonNull::new_unchecked(NonNull::<AlignedPixelChunk>::dangling().cast())
        })
    }

    /// Create from a raw pointer. Returns `None` if null.
    fn new(ptr: *mut u8) -> Option<Self> {
        // SAFETY: The pointer comes from a Vec<u8> or &mut [u8] which are Send+Sync.
        NonNull::new(ptr).map(|nn| Self(unsafe { SendSyncNonNull::new_unchecked(nn) }))
    }

    /// Create from a `NonNull<u8>`.
    fn from_non_null(ptr: NonNull<u8>) -> Self {
        // SAFETY: The pointer comes from a C allocator callback, caller ensures validity.
        Self(unsafe { SendSyncNonNull::new_unchecked(ptr) })
    }

    /// Get the raw pointer.
    fn as_ptr(self) -> *mut u8 {
        self.0.as_ptr().as_ptr()
    }

    /// Check if the pointer is aligned to [`AlignedPixelChunk`].
    fn is_chunk_aligned(self) -> bool {
        self.0.as_ptr().cast::<AlignedPixelChunk>().is_aligned()
    }
}

#[derive(Default)]
#[repr(C)]
pub struct Dav1dPictureParameters {
    pub w: c_int,
    pub h: c_int,
    pub layout: Dav1dPixelLayout,
    pub bpc: c_int,
}

// TODO(kkysen) Eventually the [`impl Default`] might not be needed.
#[derive(Clone, Default)]
#[repr(C)]
pub(crate) struct Rav1dPictureParameters {
    pub w: c_int,
    pub h: c_int,
    pub layout: Rav1dPixelLayout,
    pub bpc: u8,
}

impl From<Dav1dPictureParameters> for Rav1dPictureParameters {
    fn from(value: Dav1dPictureParameters) -> Self {
        let Dav1dPictureParameters { w, h, layout, bpc } = value;
        Self {
            w,
            h,
            layout: layout.try_into().unwrap(),
            bpc: bpc.try_into().unwrap(),
        }
    }
}

impl From<Rav1dPictureParameters> for Dav1dPictureParameters {
    fn from(value: Rav1dPictureParameters) -> Self {
        let Rav1dPictureParameters { w, h, layout, bpc } = value;
        Self {
            w,
            h,
            layout: layout.into(),
            bpc: bpc.into(),
        }
    }
}

#[cfg(feature = "c-ffi")]
#[derive(Default)]
#[repr(C)]
pub struct Dav1dPicture {
    pub seq_hdr: Option<NonNull<Dav1dSequenceHeader>>,
    pub frame_hdr: Option<NonNull<Dav1dFrameHeader>>,
    pub data: [Option<NonNull<c_void>>; 3],
    pub stride: [ptrdiff_t; 2],
    pub p: Dav1dPictureParameters,
    pub m: Dav1dDataProps,
    pub content_light: Option<NonNull<Rav1dContentLightLevel>>,
    pub mastering_display: Option<NonNull<Rav1dMasteringDisplay>>,
    pub itut_t35: Option<NonNull<Dav1dITUTT35>>,
    pub n_itut_t35: usize,
    pub reserved: [uintptr_t; 4],
    pub frame_hdr_ref: Option<RawArc<DRav1d<Rav1dFrameHeader, Dav1dFrameHeader>>>, // opaque, so we can change this
    pub seq_hdr_ref: Option<RawArc<DRav1d<Rav1dSequenceHeader, Dav1dSequenceHeader>>>, // opaque, so we can change this
    pub content_light_ref: Option<RawArc<Rav1dContentLightLevel>>, // opaque, so we can change this
    pub mastering_display_ref: Option<RawArc<Rav1dMasteringDisplay>>, // opaque, so we can change this
    pub itut_t35_ref: Option<RawArc<DRav1d<Box<[Rav1dITUTT35]>, Box<[Dav1dITUTT35]>>>>, // opaque, so we can change this
    pub reserved_ref: [uintptr_t; 4],
    pub r#ref: Option<RawArc<Rav1dPictureData>>, // opaque, so we can change this
    pub allocator_data: Option<SendSyncNonNull<c_void>>,
}

#[derive(Clone, FromBytes, IntoBytes, KnownLayout, Immutable)]
#[repr(C, align(64))]
pub struct AlignedPixelChunk([u8; RAV1D_PICTURE_ALIGNMENT]);

const _: () = assert!(mem::align_of::<AlignedPixelChunk>() == RAV1D_PICTURE_ALIGNMENT);
const _: () = assert!(mem::size_of::<AlignedPixelChunk>() == RAV1D_PICTURE_ALIGNMENT);

/// The guaranteed length multiple of [`Rav1dPictureDataComponentInner`]s.
/// This is checked and [`assume`]d.
const RAV1D_PICTURE_GUARANTEED_MULTIPLE: usize = 64;

/// Actual [`Rav1dPictureData`]'s components should be multiples of this,
/// as this is guaranteed by [`Rav1dPicAllocator::alloc_picture_callback`],
/// though wrapped buffers may only be [`RAV1D_PICTURE_GUARANTEED_MULTIPLE`].
const RAV1D_PICTURE_MULTIPLE: usize = 64 * 64;

/// The inner buffer type for picture data components.
///
/// For c-ffi: a struct with raw pointer, length, and stride (supports C allocator callbacks).
/// Without c-ffi: aliases [`StridedBuf`] from the disjoint-mut crate (all unsafe confined there).
#[cfg(feature = "c-ffi")]
pub struct Rav1dPictureDataComponentInner {
    /// A ptr to the start of this slice of `BitDepth::Pixel`s*,
    /// even if [`Self::stride`] is negative.
    ///
    /// It is aligned to [`RAV1D_PICTURE_ALIGNMENT`].
    ptr: PicDataPtr,

    /// The length of [`Self::ptr`] in [`u8`] bytes.
    ///
    /// It is a multiple of [`RAV1D_PICTURE_GUARANTEED_MULTIPLE`].
    len: usize,

    /// The stride of [`Self::ptr`] in [`u8`] bytes.
    stride: isize,
}

/// Without c-ffi, the inner buffer is a [`PicBuf`] from the disjoint-mut crate.
/// All unsafe for `AsMutPtr` is confined to that crate. Stride is stored separately
/// in [`Rav1dPictureDataComponent`].
#[cfg(not(feature = "c-ffi"))]
pub type Rav1dPictureDataComponentInner = PicBuf;

#[cfg(feature = "c-ffi")]
impl Rav1dPictureDataComponentInner {
    /// `len` and `stride` are in terms of [`u8`] bytes.
    ///
    /// # Safety
    ///
    /// `ptr`, `len`, and `stride` must follow the requirements of [`Dav1dPicAllocator::alloc_picture_callback`].
    unsafe fn new(ptr: Option<NonNull<u8>>, len: usize, stride: isize) -> Self {
        let ptr = match ptr {
            None => {
                return Self {
                    ptr: PicDataPtr::dangling_aligned(),
                    len: 0,
                    stride,
                };
            }
            Some(ptr) => ptr,
        };

        assert!(len != 0); // If `len` was 0, `ptr` should've been `None`.
        assert!(ptr.cast::<AlignedPixelChunk>().is_aligned());

        let ptr = if stride < 0 {
            let ptr = ptr.as_ptr();
            // SAFETY: According to `Dav1dPicAllocator::alloc_picture_callback`,
            // if the `stride` is negative, this is how we get the start of the data.
            // `.offset(-stride)` puts us at one element past the end of the slice,
            // and `.sub(len)` puts us back at the start of the slice.
            let ptr = unsafe { ptr.offset(-stride).sub(len) };
            PicDataPtr::new(ptr).unwrap()
        } else {
            PicDataPtr::from_non_null(ptr)
        };
        // Guaranteed by `Dav1dPicAllocator::alloc_picture_callback`.
        assert!(len % RAV1D_PICTURE_MULTIPLE == 0);
        Self { ptr, len, stride }
    }

    /// # Safety
    ///
    /// As opposed to [`Self::new`], this is safe because `buf` is a `&mut` and thus unique,
    /// so it is sound to further subdivide it into disjoint `&mut`s.
    pub fn wrap_buf<BD: BitDepth>(buf: &mut [BD::Pixel], stride: usize) -> Self {
        let buf = IntoBytes::as_mut_bytes(buf);
        let ptr = PicDataPtr::new(buf.as_mut_ptr()).unwrap();
        assert!(ptr.is_chunk_aligned());
        let len = buf.len();
        assert!(len % RAV1D_PICTURE_GUARANTEED_MULTIPLE == 0);
        let stride = (stride * mem::size_of::<BD::Pixel>()) as isize;
        Self { ptr, len, stride }
    }
}

// SAFETY: We only store the raw pointer (via PicDataPtr), so we never materialize a `&mut`.
#[cfg(feature = "c-ffi")]
#[allow(unsafe_code)]
unsafe impl ExternalAsMutPtr for Rav1dPictureDataComponentInner {
    type Target = u8;

    #[inline] // Inline so callers can see the assume.
    unsafe fn as_mut_ptr(ptr: *mut Self) -> *mut Self::Target {
        // SAFETY: Safe to dereference by unsafe preconditions.
        // Since we don't store any `&mut`s, just a raw ptr, we can have a `&Self`.
        let this = unsafe { &*ptr };

        // Assume this so that the compiler knows `ptr` is aligned.
        // Normally we'd store this as a slice so the compiler would know,
        // but since it's a ptr due to `DisjointMut`, we explicitly assume it here.
        // SAFETY: We already checked this in `Self::new`.
        unsafe { assume(this.ptr.is_chunk_aligned()) };

        this.ptr.as_ptr()
    }

    unsafe fn as_mut_slice(ptr: *mut Self) -> *mut [Self::Target] {
        // SAFETY: Only creates &Self (SharedReadOnly). Data is behind a raw pointer,
        // not inline, so SharedReadOnly doesn't cover element data.
        let this = unsafe { &*ptr };
        // SAFETY: Alignment guaranteed by PicBuf allocation (via AlignedVec).
        unsafe { assume(this.ptr.is_chunk_aligned()) };
        // SAFETY: Length is always a multiple of RAV1D_PICTURE_GUARANTEED_MULTIPLE,
        // enforced by PicBuf::new padding.
        unsafe { assume(this.len % RAV1D_PICTURE_GUARANTEED_MULTIPLE == 0) };
        core::ptr::slice_from_raw_parts_mut(this.ptr.as_ptr(), this.len)
    }

    #[inline] // Inline so callers can see the assume.
    fn len(&self) -> usize {
        // SAFETY: We already checked this in `Self::new`.
        unsafe { assume(self.len % RAV1D_PICTURE_GUARANTEED_MULTIPLE == 0) };
        self.len
    }
}

/// A picture data component: a disjoint-tracked buffer with stride.
///
/// For c-ffi: stride is stored inside the inner type.
/// Without c-ffi: stride is stored alongside the [`DisjointMut<PicBuf>`].
#[cfg(feature = "c-ffi")]
pub struct Rav1dPictureDataComponent {
    data: DisjointMut<Rav1dPictureDataComponentInner>,
}

#[cfg(not(feature = "c-ffi"))]
pub struct Rav1dPictureDataComponent {
    data: DisjointMut<Rav1dPictureDataComponentInner>,
    stride: isize,
}

impl Rav1dPictureDataComponent {
    /// Access the inner [`DisjointMut`].
    #[inline(always)]
    pub(crate) fn dm(&self) -> &DisjointMut<Rav1dPictureDataComponentInner> {
        &self.data
    }

    /// Construct from parts. For c-ffi, stride is inside inner.
    /// For non-c-ffi, stride is stored separately.
    #[cfg(feature = "c-ffi")]
    fn from_parts(inner: Rav1dPictureDataComponentInner, _stride: isize) -> Self {
        Self {
            data: crate::src::disjoint_mut::dm_new(inner),
        }
    }

    #[cfg(not(feature = "c-ffi"))]
    fn from_parts(inner: Rav1dPictureDataComponentInner, stride: isize) -> Self {
        Self {
            data: crate::src::disjoint_mut::dm_new(inner),
            stride,
        }
    }

    /// Extract the owned `Vec<u8>` from this component's inner buffer, if any.
    ///
    /// Returns `Some(vec)` for owned allocations (from `alloc_picture_data`),
    /// `None` for borrowed scratch buffers (from `wrap_buf`).
    /// Used by [`Rav1dPictureData::drop`] to return buffers to the memory pool.
    #[cfg(not(feature = "c-ffi"))]
    fn take_buf(&mut self) -> Option<Vec<u8>> {
        self.data.get_mut().take_buf()
    }

    /// Create from a pixel buffer for use as a scratch source or destination.
    ///
    /// In c-ffi mode: wraps a raw pointer into the caller's buffer (zero-copy).
    /// In safe mode: copies the data into an owned `Vec<u8>` (no raw pointers,
    /// auto `Send + Sync`). For dst-scratch usage, call [`copy_pixels_to`] after
    /// writing to retrieve the results.
    ///
    /// [`copy_pixels_to`]: Self::copy_pixels_to
    pub fn wrap_buf<BD: BitDepth>(buf: &mut [BD::Pixel], stride: usize) -> Self {
        let stride_bytes = (stride * mem::size_of::<BD::Pixel>()) as isize;
        cfg_if::cfg_if! {
            if #[cfg(feature = "c-ffi")] {
                Self::from_parts(Rav1dPictureDataComponentInner::wrap_buf::<BD>(buf, stride), stride_bytes)
            } else {
                let buf_bytes = IntoBytes::as_bytes(buf);
                assert!(buf_bytes.len() % RAV1D_PICTURE_GUARANTEED_MULTIPLE == 0);
                let inner = PicBuf::from_slice_copy(buf_bytes);
                Self::from_parts(inner, stride_bytes)
            }
        }
    }

    /// Copy pixels from this component back into a scratch buffer.
    ///
    /// Used after MC/ipred writes into a copy-backed scratch component
    /// to retrieve the results for subsequent operations (e.g., blend).
    #[cfg(not(feature = "c-ffi"))]
    pub fn copy_pixels_to<BD: BitDepth>(&self, dst: &mut [BD::Pixel]) {
        let n = self.pixel_len::<BD>();
        let guard = self.slice::<BD, _>(..);
        dst[..n].copy_from_slice(&guard[..n]);
    }
}

#[cfg(asm_fn_ptrs)]
impl Pixels for Rav1dPictureDataComponent {
    fn byte_len(&self) -> usize {
        self.dm().len()
    }

    fn as_byte_mut_ptr(&self) -> *mut u8 {
        self.dm().as_mut_ptr()
    }
}

#[cfg(feature = "c-ffi")]
#[allow(unsafe_code)]
impl Strided for Rav1dPictureDataComponent {
    fn stride(&self) -> isize {
        // SAFETY: We're only accessing the `stride` field, not `ptr`.
        unsafe { (*self.dm().inner()).stride }
    }
}

#[cfg(not(feature = "c-ffi"))]
impl Strided for Rav1dPictureDataComponent {
    fn stride(&self) -> isize {
        self.stride
    }
}

impl Rav1dPictureDataComponent {
    /// Length in number of bytes.
    pub fn byte_len(&self) -> usize {
        self.dm().len()
    }

    /// Determine if two components reference the same underlying data.
    pub fn ref_eq(&self, other: &Self) -> bool {
        self.dm().as_mut_ptr() == other.dm().as_mut_ptr()
    }

    /// Length in number of `BitDepth::Pixel`s.
    pub fn pixel_len<BD: BitDepth>(&self) -> usize {
        self.dm().len() / mem::size_of::<BD::Pixel>()
    }

    pub fn pixel_offset<BD: BitDepth>(&self) -> usize {
        let stride = self.stride();
        if stride >= 0 {
            return 0;
        }
        BD::pxstride(self.byte_len() - (-stride) as usize)
    }

    pub fn with_offset<BD: BitDepth>(&self) -> Rav1dPictureDataComponentOffset<'_> {
        Rav1dPictureDataComponentOffset {
            data: self,
            offset: self.pixel_offset::<BD>(),
        }
    }

    /// Strided ptr to [`u8`] bytes.
    ///
    /// Only used by asm (mc emu_edge) and c-ffi (Dav1dPicture conversion).
    #[cfg(any(feature = "asm", feature = "c-ffi"))]
    #[allow(unsafe_code)]
    fn as_strided_byte_mut_ptr(&self) -> *mut u8 {
        let ptr = self.dm().as_mut_ptr();
        let stride = self.stride();
        if stride < 0 {
            // SAFETY: This puts `ptr` one element past the end of the slice of pixels.
            let ptr = unsafe { ptr.add(self.byte_len()) };
            // SAFETY: `stride` is negative and `-stride < len`, so this should stay in bounds.
            let ptr = unsafe { ptr.offset(stride) };
            ptr
        } else {
            ptr
        }
    }

    /// Strided ptr to `BitDepth::Pixel`s.
    #[cfg(any(feature = "asm", feature = "c-ffi"))]
    #[allow(unsafe_code)]
    pub fn as_strided_mut_ptr<BD: BitDepth>(&self) -> *mut BD::Pixel {
        // SAFETY: Transmutation is safe because we verify this with `zerocopy` in `Self::slice`.
        self.as_strided_byte_mut_ptr().cast()
    }

    /// Strided ptr to `BitDepth::Pixel`s.
    #[cfg(feature = "asm")]
    #[allow(unsafe_code)]
    pub fn as_strided_ptr<BD: BitDepth>(&self) -> *const BD::Pixel {
        self.as_strided_mut_ptr::<BD>().cast_const()
    }

    #[cfg(feature = "c-ffi")]
    fn as_dav1d(&self) -> Option<NonNull<c_void>> {
        if self.byte_len() == 0 {
            None
        } else {
            NonNull::new(self.as_strided_byte_mut_ptr().cast())
        }
    }

    pub fn copy_from(&self, src: &Self) {
        let dst = &mut *self.dm().index_mut(..);
        let src = &*src.dm().index(..);
        dst.clone_from_slice(src);
    }

    #[inline] // Inline to see bounds checks in order to potentially elide them.
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn index<'a, BD: BitDepth>(
        &'a self,
        index: usize,
    ) -> DisjointImmutGuard<'a, Rav1dPictureDataComponentInner, BD::Pixel> {
        self.dm().element_as(index)
    }

    #[inline] // Inline to see bounds checks in order to potentially elide them.
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn index_mut<'a, BD: BitDepth>(
        &'a self,
        index: usize,
    ) -> DisjointMutGuard<'a, Rav1dPictureDataComponentInner, BD::Pixel> {
        self.dm().mut_element_as(index)
    }

    #[inline] // Inline to see bounds checks in order to potentially elide them.
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn slice<'a, BD, I>(
        &'a self,
        index: I,
    ) -> DisjointImmutGuard<'a, Rav1dPictureDataComponentInner, [BD::Pixel]>
    where
        BD: BitDepth,
        I: SliceBounds,
    {
        self.dm().slice_as(index)
    }

    #[inline] // Inline to see bounds checks in order to potentially elide them.
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn slice_mut<'a, BD, I>(
        &'a self,
        index: I,
    ) -> DisjointMutGuard<'a, Rav1dPictureDataComponentInner, [BD::Pixel]>
    where
        BD: BitDepth,
        I: SliceBounds,
    {
        self.dm().mut_slice_as(index)
    }
}

pub type Rav1dPictureDataComponentOffset<'a> = WithOffset<&'a Rav1dPictureDataComponent>;

impl<'a> Rav1dPictureDataComponentOffset<'a> {
    #[inline] // Inline to see bounds checks in order to potentially elide them.
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn index<BD: BitDepth>(
        &self,
    ) -> DisjointImmutGuard<'a, Rav1dPictureDataComponentInner, BD::Pixel> {
        self.data.index::<BD>(self.offset)
    }

    #[inline] // Inline to see bounds checks in order to potentially elide them.
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn index_mut<BD: BitDepth>(
        &self,
    ) -> DisjointMutGuard<'a, Rav1dPictureDataComponentInner, BD::Pixel> {
        self.data.index_mut::<BD>(self.offset)
    }

    #[inline] // Inline to see bounds checks in order to potentially elide them.
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn slice<BD: BitDepth>(
        &self,
        len: usize,
    ) -> DisjointImmutGuard<'a, Rav1dPictureDataComponentInner, [BD::Pixel]> {
        self.data.slice::<BD, _>((self.offset.., ..len))
    }

    #[inline] // Inline to see bounds checks in order to potentially elide them.
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn slice_mut<BD: BitDepth>(
        &self,
        len: usize,
    ) -> DisjointMutGuard<'a, Rav1dPictureDataComponentInner, [BD::Pixel]> {
        self.data.slice_mut::<BD, _>((self.offset.., ..len))
    }

    /// Create a tracked mutable guard covering a strided w×h pixel region.
    ///
    /// Handles both positive and negative strides. The returned guard covers
    /// all pixels that would be accessed by iterating h rows with the given
    /// pixel stride, each row being w pixels wide.
    ///
    /// Returns `(guard, base_offset_within_guard)` where `base_offset_within_guard`
    /// is the index within the guard's slice that corresponds to `self.offset`.
    #[inline]
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn strided_slice_mut<BD: BitDepth>(
        &self,
        w: usize,
        h: usize,
    ) -> (
        DisjointMutGuard<'a, Rav1dPictureDataComponentInner, [BD::Pixel]>,
        usize,
    ) {
        self.narrow_guard_mut::<BD>(w, h)
    }

    /// A mutable w×h pixel block that is safe to write while OTHER threads
    /// write other blocks on the same rows.
    ///
    /// # Why this exists
    ///
    /// [`Self::strided_slice_mut`] reserves ONE contiguous range covering the
    /// whole strided span, `(h - 1) * stride + w` pixels — everything from the
    /// block's first pixel to its last, INCLUDING the inter-row gaps, which
    /// belong to other blocks. That is correct single-threaded and wrong under
    /// tile threading: AV1 tiles partition a frame by columns, so two tile
    /// workers routinely write the SAME rows at different columns. The second
    /// one lands in the first one's reserved gap and `DisjointMut` panics —
    /// a false positive, since the two writes are genuinely disjoint.
    ///
    /// Concretely, on a 3840-wide frame a 16×16 block reserves 57,616 bytes in
    /// order to write 256, and a neighbouring 16-byte intra-prediction write
    /// anywhere in those rows trips it. (zenavif#30; the same defect class the
    /// loopfilter and `ipred` compact paths already fixed for themselves.)
    ///
    /// # What it does
    ///
    /// * Tile threading OFF — hands back exactly what `strided_slice_mut`
    ///   does: one contiguous guard, the picture's own stride, no copying.
    ///   Byte-identical behavior and identical cost to before.
    /// * Tile threading ON — copies the block into a compact `w × h` scratch
    ///   buffer through PER-ROW guards (each reserving only the `w` pixels
    ///   actually touched), lets the caller work on that, and writes it back
    ///   per row on drop. No gap is ever reserved, so disjoint neighbours
    ///   cannot collide.
    ///
    /// Callers must take the stride from [`BlockMut::byte_stride`] rather than
    /// from the picture, because the compact buffer has its own stride.
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn block_mut<BD: BitDepth>(&self, w: usize, h: usize) -> BlockMut<'a, BD> {
        if tile_threading_active() {
            #[cfg(feature = "held-row-guards")]
            if w != 0 && h != 0 && h <= MAX_HELD_ROWS {
                return self.block_mut_held::<BD>(w, h);
            }
            let (buf, byte_stride) = self.compact_read_per_row::<BD>(w, h);
            BlockMut {
                storage: BlockMutStorage::Compact { buf },
                dst: *self,
                w,
                h,
                base: 0,
                byte_stride: byte_stride as isize,
            }
        } else {
            use crate::src::strided::Strided as _;
            let byte_stride = self.stride();
            let (guard, base) = self.narrow_guard_mut::<BD>(w, h);
            BlockMut {
                storage: BlockMutStorage::Direct { guard },
                dst: *self,
                w,
                h,
                base,
                byte_stride,
            }
        }
    }

    /// The tile-threading [`Self::block_mut`] path that HOLDS its row guards.
    ///
    /// # Why it is cheaper
    ///
    /// The plain compact path reserves each row twice: `h` IMMUTABLE per-row
    /// guards to copy the block in, and then, on drop, `h` MUTABLE per-row
    /// guards over the very same bytes to copy it back. That is `2h`
    /// registrations for one block. Taking the MUTABLE guards up front and
    /// keeping them alive across the kernel does the same job with `h` — the
    /// read borrows through a guard that already excludes every other borrow,
    /// so the separate read guard was never adding exclusion.
    ///
    /// Measured on a t=8 4K frame before this existed (macOS `sample`, leaf
    /// samples): the read half cost 14,081 samples and the write-back half
    /// 10,105, together 9.3% of the whole decode.
    ///
    /// # Why it is sound
    ///
    /// * **Extents are unchanged.** Each guard covers exactly
    ///   `[row_off, row_off + w)` — byte-for-byte what the read guard and the
    ///   write-back guard each covered. Nothing widens.
    /// * **Exclusion only tightens.** A mutable registration conflicts with
    ///   every live record; the immutable one it replaces conflicted only with
    ///   mutable records. No overlap that the two-pass shape detected can be
    ///   missed here.
    /// * The bytes held are the block this call is about to overwrite in full,
    ///   so a concurrent borrow of them is a real conflict, not an artefact of
    ///   holding longer: dav1d's task schedule never lets another worker read a
    ///   block whose inverse transform has not finished.
    ///
    /// # Why the row cap
    ///
    /// The guards live in an inline array, so `h` is bounded at compile time,
    /// and peak live registrations per thread rise from 1 to `h`. A shard holds
    /// 7 records before it promotes a borrow to the wide list — which locks
    /// every shard of the instance — so the cap is also a guard against turning
    /// a tall block into a wide-path storm. `MAX_HELD_ROWS` covers every
    /// transform height AV1 has; anything taller falls back to the two-pass
    /// path, which is always correct.
    #[cfg(feature = "held-row-guards")]
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    fn block_mut_held<BD: BitDepth>(&self, w: usize, h: usize) -> BlockMut<'a, BD> {
        use crate::src::strided::Strided as _;
        use zerocopy::IntoBytes;
        let pixel_size = core::mem::size_of::<BD::Pixel>();
        let byte_stride = w * pixel_size;
        let needed = h * byte_stride;
        let pxstride = self.data.pixel_stride::<BD>();
        let abs_stride = pxstride.unsigned_abs();
        let mut buf = take_compact_scratch();
        buf.resize(needed, 0);
        let mut rows: RowGuards<'a, BD> = [const { None }; MAX_HELD_ROWS];
        for row in 0..h {
            let row_off = if pxstride >= 0 {
                self.offset + row * abs_stride
            } else {
                self.offset - row * abs_stride
            };
            let guard = self.data.slice_mut::<BD, _>((row_off.., ..w));
            buf[row * byte_stride..][..byte_stride]
                .copy_from_slice(&guard.as_bytes()[..byte_stride]);
            rows[row] = Some(guard);
        }
        BlockMut {
            storage: BlockMutStorage::Held { buf, rows },
            dst: *self,
            w,
            h,
            base: 0,
            byte_stride: byte_stride as isize,
        }
    }

    /// Create a tracked immutable guard covering a strided w×h pixel region.
    #[inline]
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn strided_slice<BD: BitDepth>(
        &self,
        w: usize,
        h: usize,
    ) -> (
        DisjointImmutGuard<'a, Rav1dPictureDataComponentInner, [BD::Pixel]>,
        usize,
    ) {
        self.narrow_guard::<BD>(w, h)
    }

    /// Create a tracked immutable guard covering exactly a w×h pixel block.
    #[inline]
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn narrow_guard<BD: BitDepth>(
        &self,
        w: usize,
        h: usize,
    ) -> (
        DisjointImmutGuard<'a, Rav1dPictureDataComponentInner, [BD::Pixel]>,
        usize,
    ) {
        use crate::src::strided::Strided as _;
        let pxstride = self.data.pixel_stride::<BD>();
        let abs_stride = pxstride.unsigned_abs();
        let total = if h == 0 || w == 0 {
            0
        } else {
            (h - 1) * abs_stride + w
        };
        if pxstride >= 0 {
            let guard = self.data.slice::<BD, _>((self.offset.., ..total));
            (guard, 0)
        } else {
            let start = self.offset + 1 - total;
            let guard = self.data.slice::<BD, _>((start.., ..total));
            (guard, total - 1)
        }
    }

    /// Visit `h` consecutive picture rows of `w` pixels each, IMMUTABLY, taking
    /// ONE borrow registration instead of `h` when no tile worker can be alive.
    ///
    /// # The policy, and why it is the existing one
    ///
    /// A `w×h` strided block is either `h` registrations of exactly `w` pixels,
    /// or one registration of `(h-1)*stride + w` — the hull, which additionally
    /// reserves the inter-row gaps belonging to other columns of the same rows.
    /// [`Rav1dPictureDataComponentOffset::block_mut`] already documents the
    /// trade in full: the hull is "correct single-threaded and wrong under tile
    /// threading", because AV1 tiles partition a frame by COLUMNS, so two tile
    /// workers routinely write the same rows at different columns and the gap
    /// reservation turns a genuinely disjoint pair into a false positive.
    ///
    /// So the choice is made by [`tile_threading_active`] — process-global,
    /// monotone, and never storing `false` — exactly as it is in `block_mut`,
    /// [`with_pixel_guard_immut`] and [`Self::compact_read`]. This helper adds
    /// callers to that policy; it does not introduce one.
    ///
    /// Neither branch can MISS an overlap: the hull is a superset of the `h`
    /// row ranges, and a superset registration conflicts with strictly more.
    /// The only thing at stake is false positives, and the latch is what rules
    /// those out.
    ///
    /// # Why it exists
    ///
    /// Per-row guards over small blocks are the decoder's borrow-count
    /// distribution: measured with `--features probe-sites` on `v4k_8tile` 8bpc
    /// at t=1, the per-row loops in `ipred`, `cdef` and the loopfilter were
    /// 8.4 M of 15.6 M registrations per frame, at a mean extent of ~10 bytes.
    #[inline]
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn for_rows<BD: BitDepth, F: FnMut(usize, &[BD::Pixel])>(
        &self,
        w: usize,
        h: usize,
        mut f: F,
    ) {
        use crate::src::strided::Strided as _;
        if w == 0 || h == 0 {
            return;
        }
        let pxstride = self.data.pixel_stride::<BD>();
        if tile_threading_active() {
            for row in 0..h {
                let off = self.offset.wrapping_add_signed(row as isize * pxstride);
                let guard = self.data.slice::<BD, _>((off.., ..w));
                f(row, &guard);
            }
            return;
        }
        let abs_stride = pxstride.unsigned_abs();
        let total = (h - 1) * abs_stride + w;
        let lo = if pxstride >= 0 {
            self.offset
        } else {
            self.offset - (h - 1) * abs_stride
        };
        let guard = self.data.slice::<BD, _>((lo.., ..total));
        for row in 0..h {
            let idx = if pxstride >= 0 {
                row * abs_stride
            } else {
                (h - 1 - row) * abs_stride
            };
            f(row, &guard[idx..][..w]);
        }
    }

    /// [`Self::for_rows`], mutably. Same policy, same soundness argument.
    #[inline]
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn for_rows_mut<BD: BitDepth, F: FnMut(usize, &mut [BD::Pixel])>(
        &self,
        w: usize,
        h: usize,
        mut f: F,
    ) {
        use crate::src::strided::Strided as _;
        if w == 0 || h == 0 {
            return;
        }
        let pxstride = self.data.pixel_stride::<BD>();
        if tile_threading_active() {
            for row in 0..h {
                let off = self.offset.wrapping_add_signed(row as isize * pxstride);
                let mut guard = self.data.slice_mut::<BD, _>((off.., ..w));
                f(row, &mut guard);
            }
            return;
        }
        let abs_stride = pxstride.unsigned_abs();
        let total = (h - 1) * abs_stride + w;
        let lo = if pxstride >= 0 {
            self.offset
        } else {
            self.offset - (h - 1) * abs_stride
        };
        let mut guard = self.data.slice_mut::<BD, _>((lo.., ..total));
        for row in 0..h {
            let idx = if pxstride >= 0 {
                row * abs_stride
            } else {
                (h - 1 - row) * abs_stride
            };
            f(row, &mut guard[idx..][..w]);
        }
    }

    /// Read a w×h pixel block into a compact Vec using per-row DisjointMut guards.
    ///
    /// When tile threading is active ([`set_tile_threading`]), each row guard covers
    /// exactly `w` pixels, avoiding stride-padding overlap between concurrent tiles.
    /// When single-threaded, uses one guard for the whole block (fast path).
    ///
    /// Returns `(buffer, byte_stride)` where `byte_stride` is `w * pixel_size` when
    /// threading (compact layout) or the original stride when single-threaded.
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn compact_read<BD: BitDepth>(&self, w: usize, h: usize) -> (Vec<u8>, usize) {
        if tile_threading_active() {
            self.compact_read_per_row::<BD>(w, h)
        } else {
            self.compact_read_fast::<BD>(w, h)
        }
    }

    /// Fast path: single guard for the whole block, returns original stride layout.
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    fn compact_read_fast<BD: BitDepth>(&self, w: usize, h: usize) -> (Vec<u8>, usize) {
        use crate::src::strided::Strided as _;
        use zerocopy::IntoBytes;
        let pixel_size = core::mem::size_of::<BD::Pixel>();
        let pxstride = self.data.pixel_stride::<BD>();
        let abs_stride = pxstride.unsigned_abs();
        let total = if h == 0 || w == 0 {
            0
        } else {
            (h - 1) * abs_stride + w
        };
        let start = if pxstride >= 0 {
            self.offset
        } else {
            self.offset + 1 - total
        };
        let guard = self.data.slice::<BD, _>((start.., ..total));
        let byte_stride = abs_stride * pixel_size;
        (guard.as_bytes().to_vec(), byte_stride)
    }

    /// Per-row path: each row guard covers exactly `w` pixels.
    /// Always returns compact stride = w * pixel_size.
    /// Used by the loopfilter (needs compact layout for 2D decomposition)
    /// and by tile threading (needs per-row guards to avoid stride overlap).
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn compact_read_per_row<BD: BitDepth>(&self, w: usize, h: usize) -> (Vec<u8>, usize) {
        use crate::src::strided::Strided as _;
        use zerocopy::IntoBytes;
        let pixel_size = core::mem::size_of::<BD::Pixel>();
        let byte_stride = w * pixel_size;
        let needed = h * byte_stride;
        let pxstride = self.data.pixel_stride::<BD>();
        let abs_stride = pxstride.unsigned_abs();
        // Reuse a thread-local scratch buffer instead of allocating per call
        // (issue #17). `resize` keeps the existing capacity across reuse and
        // only zero-fills when growing past the previous high-water mark; the
        // per-row copy below fully overwrites `buf[..needed]` regardless, so the
        // returned `Vec` is byte-identical to the former `vec![0u8; needed]`.
        let mut buf = take_compact_scratch();
        buf.resize(needed, 0);
        for row in 0..h {
            let row_off = if pxstride >= 0 {
                self.offset + row * abs_stride
            } else {
                self.offset - row * abs_stride
            };
            let guard = self.data.slice::<BD, _>((row_off.., ..w));
            buf[row * byte_stride..][..byte_stride]
                .copy_from_slice(&guard.as_bytes()[..byte_stride]);
        }
        (buf, byte_stride)
    }

    /// Write a compact buffer back to a w×h pixel block.
    ///
    /// Matches the layout produced by [`compact_read`].
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn compact_write_back<BD: BitDepth>(&self, w: usize, h: usize, buf: &[u8]) {
        if tile_threading_active() {
            self.compact_write_back_per_row::<BD>(w, h, buf);
        } else {
            self.compact_write_back_fast::<BD>(w, h, buf);
        }
    }

    /// Fast path write-back: single guard, original stride layout.
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    fn compact_write_back_fast<BD: BitDepth>(&self, w: usize, h: usize, buf: &[u8]) {
        use crate::src::strided::Strided as _;
        use zerocopy::IntoBytes;
        let pxstride = self.data.pixel_stride::<BD>();
        let abs_stride = pxstride.unsigned_abs();
        let total = if h == 0 || w == 0 {
            0
        } else {
            (h - 1) * abs_stride + w
        };
        let start = if pxstride >= 0 {
            self.offset
        } else {
            self.offset + 1 - total
        };
        let mut guard = self.data.slice_mut::<BD, _>((start.., ..total));
        let dst = guard.as_mut_bytes();
        let len = buf.len().min(dst.len());
        dst[..len].copy_from_slice(&buf[..len]);
    }

    /// Per-row write-back: compact stride = w * pixel_size.
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn compact_write_back_per_row<BD: BitDepth>(&self, w: usize, h: usize, buf: &[u8]) {
        use crate::src::strided::Strided as _;
        use zerocopy::IntoBytes;
        let pixel_size = core::mem::size_of::<BD::Pixel>();
        let byte_stride = w * pixel_size;
        let pxstride = self.data.pixel_stride::<BD>();
        let abs_stride = pxstride.unsigned_abs();
        for row in 0..h {
            let row_off = if pxstride >= 0 {
                self.offset + row * abs_stride
            } else {
                self.offset - row * abs_stride
            };
            let mut guard = self.data.slice_mut::<BD, _>((row_off.., ..w));
            guard.as_mut_bytes()[..byte_stride]
                .copy_from_slice(&buf[row * byte_stride..][..byte_stride]);
        }
    }

    /// Per-row write-back that stores ONLY the pixels the caller actually
    /// modified, by diffing `work` against the `pristine` copy taken before
    /// filtering. Rows (and row spans) that are byte-identical are neither
    /// written nor mutably guarded.
    ///
    /// This is the tile-threading-correct write-back for filters whose
    /// *read* region is wider than their *write* region (the loop filter
    /// reads 7 tap rows/cols beyond the ≤6 it can modify): a plain
    /// [`Self::compact_write_back_per_row`] would rewrite — and take `&mut`
    /// guards on — tap-input pixels it never changed, which (a) collides
    /// with concurrent readers that dav1d's task schedule legitimately
    /// allows (CDEF bottom-edge padding reads the 2 rows the deblock task
    /// of the next sbrow only ever *reads*; dav1d's 8-row CDEF lag is
    /// exactly 2 pad rows + 6 modified rows), and (b) risks clobbering a
    /// concurrent writer's output with stale copied bytes. Diffing makes the
    /// write-set equal dav1d's by construction. Found via zenavif#30: the
    /// `overlapping DisjointMut` worker panic (`cdef.rs` padding read vs
    /// this write-back's 7th tap row) wedged the decode wait forever.
    ///
    /// `work` and `pristine` must both use the compact layout produced by
    /// [`Self::compact_read_per_row`] (stride = `w * pixel_size`).
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn compact_write_back_per_row_diff<BD: BitDepth>(
        &self,
        w: usize,
        h: usize,
        work: &[u8],
        pristine: &[u8],
    ) {
        use crate::src::strided::Strided as _;
        use zerocopy::IntoBytes;
        let pixel_size = core::mem::size_of::<BD::Pixel>();
        let byte_stride = w * pixel_size;
        let pxstride = self.data.pixel_stride::<BD>();
        let abs_stride = pxstride.unsigned_abs();
        for row in 0..h {
            let work_row = &work[row * byte_stride..][..byte_stride];
            let pristine_row = &pristine[row * byte_stride..][..byte_stride];
            // Find the modified byte span of this row (usually empty or
            // small — the loop filter touches ≤6 pixels around each edge).
            let Some(first) = iter::zip(work_row, pristine_row).position(|(a, b)| a != b) else {
                continue; // row untouched: no write, no mutable guard
            };
            let last = iter::zip(work_row, pristine_row)
                .rposition(|(a, b)| a != b)
                .expect("a differing byte exists, so rposition finds one");
            // Widen the differing byte span [first..=last] to whole pixels.
            let first_px = first / pixel_size;
            let end_px = last / pixel_size + 1;
            let row_off = if pxstride >= 0 {
                self.offset + row * abs_stride
            } else {
                self.offset - row * abs_stride
            };
            let mut guard = self
                .data
                .slice_mut::<BD, _>((row_off + first_px.., ..end_px - first_px));
            guard.as_mut_bytes()[..(end_px - first_px) * pixel_size]
                .copy_from_slice(&work_row[first_px * pixel_size..end_px * pixel_size]);
        }
    }

    /// Create a tracked mutable guard covering the entire picture component.
    ///
    /// Returns `(guard, offset_within_guard)` where the offset corresponds to
    /// this PicOffset's logical position within the full guard.
    /// Use this when the access pattern is complex (e.g., loopfilter accessing
    /// negative offsets from the base pointer).
    #[inline]
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn full_guard_mut<BD: BitDepth>(
        &self,
    ) -> (
        DisjointMutGuard<'a, Rav1dPictureDataComponentInner, [BD::Pixel]>,
        usize,
    ) {
        let total_pixels = self.data.pixel_len::<BD>();
        let guard = self.data.slice_mut::<BD, _>((0.., ..total_pixels));
        (guard, self.offset)
    }

    /// Create a tracked mutable guard covering exactly a w×h pixel block
    /// starting at this offset. Returns `(guard, 0)` since the guard starts
    /// at self.offset.
    ///
    /// For positive strides, covers `(h-1)*stride + w` pixels from self.offset.
    /// For negative strides, covers the same span but starting `(h-1)*stride`
    /// pixels before self.offset.
    #[inline]
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn narrow_guard_mut<BD: BitDepth>(
        &self,
        w: usize,
        h: usize,
    ) -> (
        DisjointMutGuard<'a, Rav1dPictureDataComponentInner, [BD::Pixel]>,
        usize,
    ) {
        use crate::src::strided::Strided as _;
        let pxstride = self.data.pixel_stride::<BD>();
        let abs_stride = pxstride.unsigned_abs();
        let total = if h == 0 || w == 0 {
            0
        } else {
            (h - 1) * abs_stride + w
        };
        if pxstride >= 0 {
            let guard = self.data.slice_mut::<BD, _>((self.offset.., ..total));
            (guard, 0)
        } else {
            // Negative stride: rows go upward, so the first pixel row
            // is at the highest address and the last row is at the lowest.
            let start = self.offset + 1 - total;
            let guard = self.data.slice_mut::<BD, _>((start.., ..total));
            (guard, total - 1)
        }
    }

    /// Create a tracked immutable guard covering the entire picture component.
    #[inline]
    #[cfg_attr(any(debug_assertions, feature = "probe-sites"), track_caller)]
    pub fn full_guard<BD: BitDepth>(
        &self,
    ) -> (
        DisjointImmutGuard<'a, Rav1dPictureDataComponentInner, [BD::Pixel]>,
        usize,
    ) {
        let total_pixels = self.data.pixel_len::<BD>();
        let guard = self.data.slice::<BD, _>((0.., ..total_pixels));
        (guard, self.offset)
    }
}

#[cfg(feature = "c-ffi")]
pub struct Rav1dPictureData {
    pub data: [Rav1dPictureDataComponent; 3],
    pub(crate) allocator_data: Option<SendSyncNonNull<c_void>>,
    pub(crate) allocator: Rav1dPicAllocator,
}

#[cfg(not(feature = "c-ffi"))]
pub struct Rav1dPictureData {
    pub data: [Rav1dPictureDataComponent; 3],
    pub(crate) allocator: Rav1dPicAllocator,
}

#[cfg(feature = "c-ffi")]
impl Drop for Rav1dPictureData {
    fn drop(&mut self) {
        let Self {
            data,
            allocator_data,
            allocator,
        } = self;
        allocator.dealloc_picture_data(data, *allocator_data);
    }
}

#[cfg(not(feature = "c-ffi"))]
impl Drop for Rav1dPictureData {
    fn drop(&mut self) {
        for component in &mut self.data {
            if let Some(buf) = component.take_buf() {
                if !buf.is_empty() {
                    self.allocator.pool.push(buf);
                }
            }
        }
    }
}

// TODO(kkysen) Eventually the [`impl Default`] might not be needed.
// It's needed currently for a [`mem::take`] that simulates a move,
// but once everything is Rusty, we may not need to clear the `dst` anymore.
// This also applies to the `#[derive(Default)]`
// on [`Rav1dPictureParameters`] and [`Rav1dPixelLayout`].
#[derive(Clone, Default)]
#[repr(C)]
pub(crate) struct Rav1dPicture {
    pub seq_hdr: Option<Arc<DRav1d<Rav1dSequenceHeader, Dav1dSequenceHeader>>>,
    pub frame_hdr: Option<Arc<DRav1d<Rav1dFrameHeader, Dav1dFrameHeader>>>,
    pub data: Option<Arc<Rav1dPictureData>>,
    pub stride: [ptrdiff_t; 2],
    pub p: Rav1dPictureParameters,
    pub m: Rav1dDataProps,
    pub content_light: Option<Arc<Rav1dContentLightLevel>>,
    pub mastering_display: Option<Arc<Rav1dMasteringDisplay>>,
    pub itut_t35: Arc<DRav1d<Box<[Rav1dITUTT35]>, Box<[Dav1dITUTT35]>>>,
}

#[cfg(feature = "c-ffi")]
impl From<Dav1dPicture> for Rav1dPicture {
    fn from(value: Dav1dPicture) -> Self {
        let Dav1dPicture {
            seq_hdr: _,
            frame_hdr: _,
            data: _,
            stride,
            p,
            m,
            content_light: _,
            mastering_display: _,
            itut_t35: _,
            n_itut_t35: _,
            reserved: _,
            frame_hdr_ref,
            seq_hdr_ref,
            content_light_ref,
            mastering_display_ref,
            itut_t35_ref,
            reserved_ref: _,
            r#ref: data_ref,
            allocator_data: _,
        } = value;
        Self {
            // We don't `.update_rav1d()` [`Rav1dSequenceHeader`] because it's meant to be read-only.
            seq_hdr: seq_hdr_ref.map(|raw| {
                // SAFETY: `raw` came from [`RawArc::from_arc`].
                unsafe { raw.into_arc() }
            }),
            // We don't `.update_rav1d()` [`Rav1dFrameHeader`] because it's meant to be read-only.
            frame_hdr: frame_hdr_ref.map(|raw| {
                // SAFETY: `raw` came from [`RawArc::from_arc`].
                unsafe { raw.into_arc() }
            }),
            data: data_ref.map(|raw| {
                // SAFETY: `raw` came from [`RawArc::from_arc`].
                unsafe { raw.into_arc() }
            }),
            stride,
            p: p.into(),
            m: m.into(),
            content_light: content_light_ref.map(|raw| {
                // SAFETY: `raw` came from [`RawArc::from_arc`].
                unsafe { raw.into_arc() }
            }),
            mastering_display: mastering_display_ref.map(|raw| {
                // Safety: `raw` came from [`RawArc::from_arc`].
                unsafe { raw.into_arc() }
            }),
            // We don't `.update_rav1d` [`Rav1dITUTT35`] because never read it.
            itut_t35: itut_t35_ref
                .map(|raw| {
                    // SAFETY: `raw` came from [`RawArc::from_arc`].
                    unsafe { raw.into_arc() }
                })
                .unwrap_or_default(),
        }
    }
}

#[cfg(feature = "c-ffi")]
impl From<Rav1dPicture> for Dav1dPicture {
    fn from(value: Rav1dPicture) -> Self {
        let Rav1dPicture {
            seq_hdr,
            frame_hdr,
            data,
            stride,
            p,
            m,
            content_light,
            mastering_display,
            itut_t35,
        } = value;
        Self {
            // [`DRav1d::from_rav1d`] is called right after [`parse_seq_hdr`].
            seq_hdr: seq_hdr.as_ref().map(|arc| (&arc.as_ref().dav1d).into()),
            // [`DRav1d::from_rav1d`] is called in [`parse_frame_hdr`].
            frame_hdr: frame_hdr.as_ref().map(|arc| (&arc.as_ref().dav1d).into()),
            data: data
                .as_ref()
                .map(|arc| arc.data.each_ref().map(|data| data.as_dav1d()))
                .unwrap_or_default(),
            stride,
            p: p.into(),
            m: m.into(),
            content_light: content_light.as_ref().map(|arc| arc.as_ref().into()),
            mastering_display: mastering_display.as_ref().map(|arc| arc.as_ref().into()),
            // [`DRav1d::from_rav1d`] is called in [`rav1d_parse_obus`].
            itut_t35: Some(NonNull::new(itut_t35.dav1d.as_ptr().cast_mut()).unwrap()),
            n_itut_t35: itut_t35.len(),
            reserved: Default::default(),
            frame_hdr_ref: frame_hdr.map(RawArc::from_arc),
            seq_hdr_ref: seq_hdr.map(RawArc::from_arc),
            content_light_ref: content_light.map(RawArc::from_arc),
            mastering_display_ref: mastering_display.map(RawArc::from_arc),
            itut_t35_ref: Some(itut_t35).map(RawArc::from_arc),
            reserved_ref: Default::default(),
            // Order flipped so that the borrow comes before the move.
            allocator_data: data.as_ref().and_then(|arc| arc.allocator_data),
            r#ref: data.map(RawArc::from_arc),
        }
    }
}

impl Rav1dPicture {
    pub fn lf_offsets<BD: BitDepth>(&self, y: c_int) -> [Rav1dPictureDataComponentOffset<'_>; 3] {
        // Init loopfilter offsets. Point the chroma offsets in 4:0:0 to the luma
        // plane here to avoid having additional in-loop branches in various places.
        // We never use those values, so it doesn't really matter what they point
        // at, as long as the offsets are valid.
        let layout = self.p.layout;
        let has_chroma = layout != Rav1dPixelLayout::I400;
        let data = &self.data.as_ref().unwrap().data;
        array::from_fn(|i| {
            let data = &data[has_chroma as usize * i];
            let ss_ver = layout == Rav1dPixelLayout::I420 && i != 0;
            data.with_offset::<BD>() + (y as isize * data.pixel_stride::<BD>() >> ss_ver as u8)
        })
    }
}

#[cfg(feature = "c-ffi")]
#[derive(Clone)]
#[repr(C)]
pub struct Dav1dPicAllocator {
    /// Custom data to pass to the allocator callbacks.
    ///
    /// # Safety
    ///
    /// All accesses to [`Self::cookie`] must be thread-safe
    /// (i.e. [`Self::cookie`] must be [`Send`]` + `[`Sync`]).
    ///
    /// If used from Rust, [`Self::cookie`] is a [`SendSyncNonNull`],
    /// whose constructors ensure this [`Send`]` + `[`Sync`] safety.
    pub cookie: Option<SendSyncNonNull<c_void>>,

    /// Allocate the picture buffer based on the [`Dav1dPictureParameters`].
    ///
    /// [`data`]`[0]`, [`data`]`[1]` and [`data`]`[2]`
    /// must be [`DAV1D_PICTURE_ALIGNMENT`]-byte aligned
    /// and with a pixel width/height multiple of 128 pixels.
    /// Any allocated memory area should also be padded by [`DAV1D_PICTURE_ALIGNMENT`] bytes.
    /// [`data`]`[1]` and [`data`]`[2]` must share the same [`stride`]`[1]`.
    ///
    /// # Safety
    ///
    /// See [`Self::cookie`]'s safety requirements.
    ///
    /// ### Additional `rav1d` requirement:
    ///
    /// The allocated data must be initialized.
    /// If newly (e.x. not reused) allocated data is zero initialized using OS APIs,
    /// it is possible for this to not be slower than an uninitialized allocation.
    /// For example, see `dav1d_default_picture_alloc` and `MemPool::pop_init`.
    ///
    /// If the allocated data is not initialized,
    /// it is possible there will be reads of uninitialized data.
    /// `rav1d` should not read this data before writing to it first,
    /// but it does not guarantee that it does so.
    /// Instead, initializing the allocated data guarantees all uses of it will be sound.
    ///
    /// # Args
    ///
    /// * `pic`: The picture to allocate the buffer for.
    ///     The callback needs to fill the picture
    ///     [`data`]`[0]`, [`data`]`[1]`, [`data`]`[2]`,
    ///     [`stride`]`[0]`, and [`stride`]`[1]`.
    ///     The allocator can fill the pic [`allocator_data`] pointer
    ///     with a custom pointer that will be passed to
    ///     [`release_picture_callback`].
    ///
    ///     The only fields of `pic` that will be already set are:
    ///     * [`Dav1dPicture::p`]
    ///     * [`Dav1dPicture::seq_hdr`]
    ///     * [`Dav1dPicture::frame_hdr`]
    ///     
    ///     This is not a change from the original `DAV1D_API`,
    ///     just a clarification of it.
    ///
    /// * `cookie`: Custom pointer passed to all calls.
    ///
    /// *Note*: No fields other than [`data`], [`stride`] and [`allocator_data`]
    /// must be filled by this callback.
    ///
    /// # Return
    ///
    /// 0 on success. A negative `DAV1D_ERR` value on error.
    /// <!--- TODO(kkysen) Translate `DAV1D_ERR` -->
    ///
    /// [`data`]: Dav1dPicture::data
    /// [`stride`]: Dav1dPicture::data
    /// [`allocator_data`]: Dav1dPicture::allocator_data
    /// [`release_picture_callback`]: Self::release_picture_callback
    pub alloc_picture_callback: Option<
        unsafe extern "C" fn(
            pic: *mut Dav1dPicture,
            cookie: Option<SendSyncNonNull<c_void>>,
        ) -> Dav1dResult,
    >,

    /// Release the picture buffer.
    ///
    /// # Safety
    ///
    /// If frame threading is used, accesses to `cookie` must be thread-safe.
    ///
    /// If frame threading is used, this function may be called by the main thread
    /// (the thread which calls [`dav1d_get_picture`]),
    /// or any of the frame threads and thus must be thread-safe.
    /// If frame threading is not used, this function will only be called on the main thread.
    ///
    /// # Args
    ///
    /// * `pic`: The picture that was filled by [`alloc_picture_callback`].
    ///     
    ///     The only fields of `pic` that will be set are
    ///     the ones allocated by [`Self::alloc_picture_callback`]:
    ///     * [`Dav1dPicture::data`]
    ///     * [`Dav1dPicture::allocator_data`]
    ///     
    ///     NOTE: This is a slight change from the original `DAV1D_API`, which was underspecified.
    ///     However, all known uses of this API follow this already:
    ///     * `libdav1d`: [`dav1d_default_picture_release`](https://code.videolan.org/videolan/dav1d/-/blob/16ed8e8b99f2fcfffe016e929d3626e15267ad3e/src/picture.c#L85-87)
    ///     * `dav1d`: [`picture_release`](https://code.videolan.org/videolan/dav1d/-/blob/16ed8e8b99f2fcfffe016e929d3626e15267ad3e/tools/dav1d.c#L180-182)
    ///     * `dav1dplay`: [`placebo_release_pic`](https://code.videolan.org/videolan/dav1d/-/blob/16ed8e8b99f2fcfffe016e929d3626e15267ad3e/examples/dp_renderer_placebo.c#L375-383)
    ///     * `libplacebo`: [`pl_release_dav1dpicture`](https://github.com/haasn/libplacebo/blob/34e019bfedaa5a64f268d8f9263db352c0a8f67f/src/include/libplacebo/utils/dav1d_internal.h#L594-L607)
    ///     * `ffmpeg`: [`libdav1d_picture_release`](https://github.com/FFmpeg/FFmpeg/blob/00b288da73f45acb78b74bcc40f73c7ba1fff7cb/libavcodec/libdav1d.c#L124-L129)
    ///
    ///     Making this API safe without this slight tightening of the API
    ///     [is very difficult](https://github.com/memorysafety/rav1d/pull/685#discussion_r1458171639).
    ///
    /// * `cookie`: Custom pointer passed to all calls.
    ///
    /// [`dav1d_get_picture`]: crate::src::lib::dav1d_get_picture
    /// [`alloc_picture_callback`]: Self::alloc_picture_callback
    pub release_picture_callback: Option<
        unsafe extern "C" fn(pic: *mut Dav1dPicture, cookie: Option<SendSyncNonNull<c_void>>) -> (),
    >,
}

#[cfg(feature = "c-ffi")]
#[derive(Clone)]
#[repr(C)]
pub(crate) struct Rav1dPicAllocator {
    /// See [`Dav1dPicAllocator::cookie`].
    ///
    /// # Safety
    ///
    /// If [`Self::is_default`]`()`, then this cookie is a reference to
    /// [`Rav1dContext::picture_pool`], a `&Arc<MemPool<u8>`.
    /// Thus, its lifetime is that of `&c.picture_pool`,
    /// so the lifetime of the `&`[`Rav1dContext`].
    /// This is used from `dav1d_default_picture_alloc`
    /// ([`Self::default`]`().alloc_picture_callback`),
    /// which is called from [`Self::alloc_picture_data`],
    /// which is called further up on the call stack with a `&`[`Rav1dContext`].
    /// Thus, the lifetime will always be valid where used.
    ///
    /// Note that this is an `&Arc<MemPool<u8>` turned into a raw pointer,
    /// not an [`Arc::into_raw`] of that [`Arc`].
    /// This is because storing the [`Arc`] would require C to
    /// free data owned by a [`Dav1dPicAllocator`] potentially,
    /// which it may not do, as there are no current APIs for doing so.
    ///
    /// [`Rav1dContext::picture_pool`]: crate::src::internal::Rav1dContext::picture_pool
    /// [`Rav1dContext`]: crate::src::internal::Rav1dContext
    pub cookie: Option<SendSyncNonNull<c_void>>,

    /// See [`Dav1dPicAllocator::alloc_picture_callback`].
    ///
    /// # Safety
    ///
    /// `pic` is passed as a `&mut`.
    ///
    /// If frame threading is used, accesses to [`Self::cookie`] must be thread-safe,
    /// i.e. [`Self::cookie`] must be [`Send`]` + `[`Sync`].
    pub alloc_picture_callback: unsafe extern "C" fn(
        pic: *mut Dav1dPicture,
        cookie: Option<SendSyncNonNull<c_void>>,
    ) -> Dav1dResult,

    /// See [`Dav1dPicAllocator::release_picture_callback`].
    ///
    /// # Safety
    ///
    /// `pic` is passed as a `&mut`.
    ///
    /// If frame threading is used, accesses to [`Self::cookie`] must be thread-safe,
    /// i.e. [`Self::cookie`] must be [`Send`]` + `[`Sync`].
    pub release_picture_callback:
        unsafe extern "C" fn(pic: *mut Dav1dPicture, cookie: Option<SendSyncNonNull<c_void>>) -> (),
}

/// Safe picture allocator using per-plane `Vec<u8>` buffers from a shared pool.
#[cfg(not(feature = "c-ffi"))]
#[derive(Clone, Default)]
pub(crate) struct Rav1dPicAllocator {
    pub(crate) pool: Arc<MemPool<u8>>,
}

#[cfg(feature = "c-ffi")]
impl TryFrom<Dav1dPicAllocator> for Rav1dPicAllocator {
    type Error = Rav1dError;

    fn try_from(value: Dav1dPicAllocator) -> Result<Self, Self::Error> {
        let Dav1dPicAllocator {
            cookie,
            alloc_picture_callback,
            release_picture_callback,
        } = value;
        Ok(Self {
            cookie,
            alloc_picture_callback: validate_input!(alloc_picture_callback.ok_or(EINVAL))?,
            release_picture_callback: validate_input!(release_picture_callback.ok_or(EINVAL))?,
        })
    }
}

#[cfg(feature = "c-ffi")]
impl From<Rav1dPicAllocator> for Dav1dPicAllocator {
    fn from(value: Rav1dPicAllocator) -> Self {
        let Rav1dPicAllocator {
            cookie,
            alloc_picture_callback,
            release_picture_callback,
        } = value;
        Self {
            cookie,
            alloc_picture_callback: Some(alloc_picture_callback),
            release_picture_callback: Some(release_picture_callback),
        }
    }
}

#[cfg(feature = "c-ffi")]
impl Rav1dPicAllocator {
    pub fn alloc_picture_data(
        &self,
        w: c_int,
        h: c_int,
        seq_hdr: Arc<DRav1d<Rav1dSequenceHeader, Dav1dSequenceHeader>>,
        frame_hdr: Option<Arc<DRav1d<Rav1dFrameHeader, Dav1dFrameHeader>>>,
    ) -> Rav1dResult<Rav1dPicture> {
        let pic = Rav1dPicture {
            p: Rav1dPictureParameters {
                w,
                h,
                layout: seq_hdr.layout,
                bpc: 8 + 2 * seq_hdr.hbd,
            },
            seq_hdr: Some(seq_hdr),
            frame_hdr,
            ..Default::default()
        };
        let mut pic_c = pic.to::<Dav1dPicture>();
        // SAFETY: `pic_c` is a valid `Dav1dPicture` with `data`, `stride`, `allocator_data` unset.
        let result = unsafe { (self.alloc_picture_callback)(&mut pic_c, self.cookie) };
        result.try_to::<Rav1dResult>().unwrap()?;
        // `data`, `stride`, and `allocator_data` are the only fields set by the allocator.
        // Of those, only `data` and `allocator_data` are read through `r#ref`,
        // so we need to read those directly first and allocate the `Arc`.
        let data = pic_c.data;
        let allocator_data = pic_c.allocator_data;
        let mut pic = pic_c.to::<Rav1dPicture>();
        let len = pic.p.pic_len(pic.stride)?;
        // TODO fallible allocation
        pic.data = Some(Arc::new(Rav1dPictureData {
            data: array::from_fn(|i| {
                let ptr = data[i].map(|ptr| ptr.cast::<u8>());
                let len = len[(i != 0) as usize];
                let stride = pic.stride[(i != 0) as usize];
                // SAFETY: These args come from `Self::alloc_picture_callback`.
                let component = unsafe { Rav1dPictureDataComponentInner::new(ptr, len, stride) };
                Rav1dPictureDataComponent::from_parts(component, stride)
            }),
            allocator_data,
            allocator: self.clone(),
        }));
        Ok(pic)
    }

    pub fn dealloc_picture_data(
        &self,
        data: &mut [Rav1dPictureDataComponent; 3],
        allocator_data: Option<SendSyncNonNull<c_void>>,
    ) {
        let data = data.each_mut().map(|data| data.as_dav1d());
        let mut pic_c = Dav1dPicture {
            data,
            allocator_data,
            ..Default::default()
        };
        // SAFETY: `pic_c` contains the same `data` and `allocator_data`
        // that `Self::alloc_picture_data` set, which now get deallocated here.
        unsafe {
            (self.release_picture_callback)(&mut pic_c, self.cookie);
        }
    }
}

#[cfg(not(feature = "c-ffi"))]
impl Rav1dPicAllocator {
    pub fn alloc_picture_data(
        &self,
        w: c_int,
        h: c_int,
        seq_hdr: Arc<DRav1d<Rav1dSequenceHeader, Dav1dSequenceHeader>>,
        frame_hdr: Option<Arc<DRav1d<Rav1dFrameHeader, Dav1dFrameHeader>>>,
    ) -> Rav1dResult<Rav1dPicture> {
        let p = Rav1dPictureParameters {
            w,
            h,
            layout: seq_hdr.layout,
            bpc: 8 + 2 * seq_hdr.hbd,
        };

        let hbd = (p.bpc > 8) as c_int;
        let aligned_w = p.w + 127 & !127;
        let has_chroma = p.layout != Rav1dPixelLayout::I400;
        let ss_hor = (p.layout != Rav1dPixelLayout::I444) as c_int;
        let mut y_stride = (aligned_w << hbd) as isize;
        let mut uv_stride = if has_chroma { y_stride >> ss_hor } else { 0 };
        if y_stride & 1023 == 0 {
            y_stride += RAV1D_PICTURE_ALIGNMENT as isize;
        }
        if uv_stride & 1023 == 0 && has_chroma {
            uv_stride += RAV1D_PICTURE_ALIGNMENT as isize;
        }
        let stride = [y_stride, uv_stride];
        let [y_sz, uv_sz] = p.pic_len(stride)?;

        // Round up to RAV1D_PICTURE_MULTIPLE for allocated data. Use checked
        // arithmetic: an overflow here (only reachable on 32-bit with crafted
        // dimensions whose plane size lands just below usize::MAX) must surface
        // as ENOMEM, not wrap into an under-sized allocation.
        let round_up = |sz: usize| -> Rav1dResult<usize> {
            if sz == 0 {
                Ok(0)
            } else {
                Ok(sz
                    .checked_add(RAV1D_PICTURE_MULTIPLE - 1)
                    .ok_or(Rav1dError::ENOMEM)?
                    & !(RAV1D_PICTURE_MULTIPLE - 1))
            }
        };
        let y_sz = round_up(y_sz)?;
        let uv_sz = round_up(uv_sz)?;

        // Allocate per-plane buffers with alignment padding (checked add — see above).
        let alloc_plane = |sz: usize| -> Result<Vec<u8>, Rav1dError> {
            if sz == 0 {
                return Ok(Vec::new());
            }
            let alloc_len = sz
                .checked_add(RAV1D_PICTURE_ALIGNMENT)
                .ok_or(Rav1dError::ENOMEM)?;
            self.pool
                .pop_init(alloc_len, 0)
                .map_err(|_| Rav1dError::ENOMEM)
        };

        let y_buf = alloc_plane(y_sz)?;
        let u_buf = alloc_plane(uv_sz)?;
        let v_buf = alloc_plane(uv_sz)?;

        let data = [
            Rav1dPictureDataComponent::from_parts(
                PicBuf::from_vec_aligned(y_buf, RAV1D_PICTURE_ALIGNMENT, y_sz),
                y_stride,
            ),
            Rav1dPictureDataComponent::from_parts(
                PicBuf::from_vec_aligned(u_buf, RAV1D_PICTURE_ALIGNMENT, uv_sz),
                uv_stride,
            ),
            Rav1dPictureDataComponent::from_parts(
                PicBuf::from_vec_aligned(v_buf, RAV1D_PICTURE_ALIGNMENT, uv_sz),
                uv_stride,
            ),
        ];

        let pic = Rav1dPicture {
            p,
            seq_hdr: Some(seq_hdr),
            frame_hdr,
            stride,
            data: Some(Arc::new(Rav1dPictureData {
                data,
                allocator: self.clone(),
            })),
            ..Default::default()
        };

        Ok(pic)
    }
}
pub type PicOffset<'a> = Rav1dPictureDataComponentOffset<'a>;

/// Tallest block [`WithOffset::block_mut_held`] will hold row guards for.
///
/// 64 is the tallest AV1 transform (`TX_64X64` / `TX_16X64` / `TX_32X64`), so
/// in practice nothing falls back. The array costs
/// `MAX_HELD_ROWS * size_of::<Option<guard>>()` of stack in the frame that owns
/// the [`BlockMut`]; a guard is a fat slice reference, a parent reference and a
/// `BorrowId`, and `Option` is niche-packed into the slice pointer.
#[cfg(feature = "held-row-guards")]
const MAX_HELD_ROWS: usize = 64;

#[cfg(feature = "held-row-guards")]
type RowGuards<'a, BD> = [Option<
    DisjointMutGuard<'a, Rav1dPictureDataComponentInner, [<BD as BitDepth>::Pixel]>,
>; MAX_HELD_ROWS];

/// Backing storage for [`BlockMut`] — see [`WithOffset::block_mut`].
enum BlockMutStorage<'a, BD: BitDepth> {
    /// Tile threading off: the picture's own memory, one contiguous guard.
    Direct {
        guard: DisjointMutGuard<'a, Rav1dPictureDataComponentInner, [BD::Pixel]>,
    },
    /// Tile threading on: a detached compact copy, written back on drop
    /// through fresh per-row guards.
    Compact { buf: Vec<u8> },
    /// Tile threading on, and the per-row MUTABLE guards are held for the
    /// block's whole life — see [`WithOffset::block_mut_held`]. Half the
    /// registrations of `Compact`, same extents.
    #[cfg(feature = "held-row-guards")]
    Held {
        buf: Vec<u8>,
        rows: RowGuards<'a, BD>,
    },
}

/// A writable w×h pixel block. Obtain one with [`WithOffset::block_mut`], which
/// documents why the compact variant exists.
///
/// On drop, a compact block is written back to the picture through per-row
/// guards. A direct block has nothing to do — the caller wrote the picture in
/// place.
pub struct BlockMut<'a, BD: BitDepth> {
    storage: BlockMutStorage<'a, BD>,
    dst: WithOffset<&'a Rav1dPictureDataComponent>,
    w: usize,
    h: usize,
    base: usize,
    byte_stride: isize,
}

impl<'a, BD: BitDepth> BlockMut<'a, BD> {
    /// Byte stride of the buffer returned by [`Self::as_mut_bytes`].
    ///
    /// NOT necessarily the picture's stride: a compact block is tightly packed
    /// at `w * size_of::<BD::Pixel>()`. Always positive in the compact case;
    /// may be negative in the direct case, exactly as the picture's is.
    #[inline]
    pub fn byte_stride(&self) -> isize {
        self.byte_stride
    }

    /// Index within [`Self::as_mut_bytes`] of the block's first pixel.
    ///
    /// Zero for a compact block; for a direct block with negative stride it is
    /// the last row's offset, matching `strided_slice_mut`'s second return value.
    #[inline]
    pub fn base(&self) -> usize {
        self.base
    }

    /// The writable bytes.
    #[inline]
    pub fn as_mut_bytes(&mut self) -> &mut [u8] {
        match &mut self.storage {
            BlockMutStorage::Direct { guard } => guard.as_mut_bytes(),
            BlockMutStorage::Compact { buf } => buf.as_mut_slice(),
            #[cfg(feature = "held-row-guards")]
            BlockMutStorage::Held { buf, .. } => buf.as_mut_slice(),
        }
    }
}

impl<BD: BitDepth> Drop for BlockMut<'_, BD> {
    fn drop(&mut self) {
        match &mut self.storage {
            BlockMutStorage::Direct { .. } => {}
            BlockMutStorage::Compact { buf } => {
                // Unconditional per-row write-back rather than the loopfilter's
                // diff variant: an inverse transform adds residual across the
                // whole block, so a diff would compare every byte only to
                // discover that every byte changed. Per-row is what keeps the
                // reservations narrow.
                self.dst
                    .compact_write_back_per_row::<BD>(self.w, self.h, buf);
                recycle_compact_scratch(core::mem::take(buf));
            }
            #[cfg(feature = "held-row-guards")]
            BlockMutStorage::Held { buf, rows } => {
                // The rows are already reserved — write straight through the
                // guards taken in `block_mut_held`, then let them drop. Same
                // bytes as `compact_write_back_per_row` would have written,
                // with no second round of registrations.
                let byte_stride = self.byte_stride as usize;
                for (row, slot) in rows.iter_mut().enumerate().take(self.h) {
                    if let Some(guard) = slot {
                        guard.as_mut_bytes()[..byte_stride]
                            .copy_from_slice(&buf[row * byte_stride..][..byte_stride]);
                    }
                }
                // Drop the guards BEFORE recycling, so the scratch is only
                // handed back once the picture is consistent.
                *rows = [const { None }; MAX_HELD_ROWS];
                recycle_compact_scratch(core::mem::take(buf));
            }
        }
    }
}

#[cfg(test)]
mod tile_threading_latch_tests {
    /// `set_tile_threading` must never be able to turn the flag back off.
    ///
    /// It is one process-global bool shared by every decoder, so a
    /// single-threaded `rav1d_open` storing `false` used to push concurrently
    /// live multi-threaded decoders back onto the wide-guard path — whose
    /// borrow extent is `(h-1) * stride + w`, e.g. 16,321 pixels for a 1x16
    /// intra left-edge column — while their tile workers were running. That
    /// showed up as spurious `overlapping DisjointMut` panics under load (8-9
    /// of 24 concurrent runs) and would be an undetected data race in an
    /// `unchecked` build.
    #[test]
    fn set_tile_threading_is_monotone() {
        use super::{set_tile_threading, tile_threading_active};
        // Whatever another test in this binary has already done, `true` sticks.
        set_tile_threading(true);
        assert!(tile_threading_active());
        set_tile_threading(false);
        assert!(
            tile_threading_active(),
            "a single-threaded open must not clear tile threading for \
             concurrently live multi-threaded decoders"
        );
    }
}

/// MECHANISM gate for [`Rav1dPictureDataComponentOffset::for_rows`] and
/// [`Rav1dPictureDataComponentOffset::for_rows_mut`].
///
/// Those two pick between ONE hull registration and `h` per-row ones on
/// [`tile_threading_active`]. Which branch runs is invisible in the decoded
/// pixels, so nothing else in the tree can catch a regression in either
/// direction:
///
/// * per-row taken with threading OFF — the whole point of the helper is lost
///   and 8.4 M registrations per frame come back, with no test going red.
/// * hull taken with threading ON — the hull reserves the inter-row gaps, which
///   belong to other tile COLUMNS. A neighbouring tile's legitimate write then
///   trips a spurious `overlapping DisjointMut` panic (measured 8-9 of 24
///   concurrent runs before [`set_tile_threading`] was made monotone), or, in
///   an `unchecked` build, races undetected.
///
/// So each test asserts which EXTENT got reserved, by holding a byte that only
/// the hull covers and checking whether the tracker rejects the call.
#[cfg(test)]
mod row_guard_policy_tests {
    use super::{Rav1dPictureDataComponent, set_tile_threading, tile_threading_active};
    use crate::include::common::bitdepth::BitDepth8;
    use crate::src::with_offset::WithOffset;
    use std::panic::{self, AssertUnwindSafe};

    const STRIDE: usize = 64;
    const ROWS: usize = 4;
    const W: usize = 8;
    /// First pixel of row 0 to last pixel of row `ROWS-1`, gaps included.
    const HULL: usize = (ROWS - 1) * STRIDE + W;

    fn plane() -> Rav1dPictureDataComponent {
        // `wrap_buf` asserts the byte length is a multiple of 64.
        let mut buf = vec![0u8; STRIDE * ROWS];
        Rav1dPictureDataComponent::wrap_buf::<BitDepth8>(&mut buf, STRIDE)
    }

    /// Does `for_rows_mut` over the `W x ROWS` block at offset 0 conflict with a
    /// live mutable borrow of the single pixel `probe`?
    fn conflicts_with(probe: usize) -> bool {
        let pic = plane();
        let held = pic.slice_mut::<BitDepth8, _>((probe.., ..1));
        let at = WithOffset {
            data: &pic,
            offset: 0,
        };
        let prev = panic::take_hook();
        panic::set_hook(Box::new(|_| {}));
        let r = panic::catch_unwind(AssertUnwindSafe(|| {
            at.for_rows_mut::<BitDepth8, _>(W, ROWS, |_, row| {
                row[0] = 1;
            });
        }));
        panic::set_hook(prev);
        drop(held);
        r.is_err()
    }

    fn assert_hull_branch() {
        assert!(
            !tile_threading_active(),
            "precondition: this must run in a process where no decoder has \
             latched tile threading, or it is testing the other branch"
        );
        // A byte in the INTER-ROW GAP — inside the hull, outside every row's
        // own `[row*STRIDE, row*STRIDE + W)`. Only the hull registration covers
        // it, so a conflict here is proof the hull branch ran.
        assert!(
            conflicts_with(W),
            "with tile threading off, `for_rows_mut` must take ONE hull guard; \
             a gap byte inside [0, {HULL}) did not conflict, so the per-row \
             branch ran instead"
        );
        // Control: past the hull nothing may conflict, or the assertion above
        // would pass for the wrong reason.
        assert!(
            !conflicts_with(HULL),
            "a byte outside the hull must never conflict"
        );
    }

    /// The threading-OFF half, run in a CHILD PROCESS.
    ///
    /// `TILE_THREADING` is a monotone process-global and
    /// `set_tile_threading_is_monotone` (above, same binary) latches it on, so
    /// this branch is simply not observable in the parent — and libtest gives
    /// no ordering guarantee that would fix that. Re-exec'ing ourselves with
    /// `--exact` gives a process running this test and nothing else.
    ///
    /// The child's marker line is checked, not just its exit status: libtest
    /// exits 0 when a filter matches NOTHING, so a renamed test would otherwise
    /// turn this gate green while running no assertions at all.
    #[test]
    fn for_rows_mut_reserves_the_strided_hull_when_tile_threading_is_off() {
        const MARKER: &str = "ROW_GUARD_HULL_CHILD_RAN";
        const NAME: &str = "include::dav1d::picture::row_guard_policy_tests::\
                            for_rows_mut_reserves_the_strided_hull_when_tile_threading_is_off";
        if std::env::var_os("RAV1D_ROW_GUARD_CHILD").is_some() {
            assert_hull_branch();
            println!("{MARKER}");
            return;
        }
        let exe = std::env::current_exe().expect("test binary path");
        let out = std::process::Command::new(exe)
            .args(["--exact", NAME, "--nocapture"])
            .env("RAV1D_ROW_GUARD_CHILD", "1")
            .output()
            .expect("re-exec the test binary");
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            stdout.contains(MARKER),
            "the child process did not reach the assertions (renamed test? \
             libtest exits 0 on an empty filter). status={:?}\nstdout:\n{}\nstderr:\n{}",
            out.status,
            stdout,
            String::from_utf8_lossy(&out.stderr),
        );
        assert!(out.status.success(), "child failed:\n{stdout}");
    }

    /// The threading-ON half. Latches the flag itself, so it is independent of
    /// what else in this binary has run.
    #[test]
    fn for_rows_mut_never_reserves_an_inter_row_gap_when_tile_threading_is_on() {
        set_tile_threading(true);
        assert!(tile_threading_active());
        assert!(
            !conflicts_with(W),
            "with tile threading ON, `for_rows_mut` must take PER-ROW guards; a \
             byte in the inter-row gap conflicted, which is exactly the false \
             positive `block_mut`'s tile-threading branch and the monotone \
             `set_tile_threading` latch exist to prevent"
        );
        // Anti-vacuity: without this, the assertion above would also pass with
        // borrow tracking compiled out entirely (`--features unchecked`).
        assert!(
            conflicts_with(STRIDE),
            "row 1 column 0 IS written by this block, so a live mutable borrow \
             of it must conflict; if it does not, tracking is off in this build \
             and this test gates nothing"
        );
    }
}
