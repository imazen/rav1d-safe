#![allow(dead_code)]
// The load/store macros expand to `unsafe {}` blocks with bounds-checked pointer access.
// These are verified safe by construction (bounds check precedes raw pointer use).
#![allow(clippy::undocumented_unsafe_blocks)]
//! Safe pixel access helpers and SIMD load/store macros for SIMD modules.
//!
//! When the `unchecked` feature is enabled, these use unchecked indexing
//! and raw pointer SIMD access for performance parity. Otherwise, they use
//! bounds-checked indexing and `safe_unaligned_simd` wrappers (no `unsafe`).
//!
//! # SIMD Load/Store Macros
//!
//! These macros provide a clean, unified API for SIMD memory access that
//! switches between safe and unchecked implementations:
//!
//! ```ignore
//! // x86_64 AVX2: load/store 256 bits (32 bytes)
//! let v = loadu_256!(&src[off..off+32], [u8; 32]);
//! storeu_256!(&mut dst[off..off+32], [u8; 32], v);
//!
//! // x86_64 SSE2: load/store 128 bits (16 bytes)
//! let v = loadu_128!(&src[off..off+16], [u8; 16]);
//! storeu_128!(&mut dst[off..off+16], [u8; 16], v);
//!
//! // Direct array reference (no conversion needed):
//! let v = loadu_256!(&arr);  // arr: [u8; 32]
//! storeu_256!(&mut arr, v);  // arr: [u8; 32]
//! ```
//!
//! When `unchecked` is **off** (default):
//! - Uses `safe_unaligned_simd` for memory access (safe, bounds-checked)
//! - Compatible with `#![forbid(unsafe_code)]` in calling modules
//!
//! When `unchecked` is **on**:
//! - Uses raw `core::arch` intrinsics with pointer access (no bounds checks)
//! - `debug_assert!` still validates in debug builds

// When unchecked is off: deny unsafe (bounds-checked path only).
// When unchecked is on: allow unsafe (raw pointer load/store for performance).
#![cfg_attr(not(feature = "unchecked"), deny(unsafe_code))]
#![cfg_attr(feature = "unchecked", allow(unsafe_code))]

use zerocopy::{AsBytes, FromBytes, Ref};

/// Get an immutable slice from a buffer at a given offset.
#[inline(always)]
pub fn row_slice(buf: &[u8], offset: usize, len: usize) -> &[u8] {
    #[cfg(feature = "unchecked")]
    {
        debug_assert!(offset + len <= buf.len());
        // SAFETY: caller guarantees offset + len <= buf.len()
        unsafe { buf.get_unchecked(offset..offset + len) }
    }
    #[cfg(not(feature = "unchecked"))]
    &buf[offset..offset + len]
}

/// Get a mutable slice from a buffer at a given offset.
#[inline(always)]
pub fn row_slice_mut(buf: &mut [u8], offset: usize, len: usize) -> &mut [u8] {
    #[cfg(feature = "unchecked")]
    {
        debug_assert!(offset + len <= buf.len());
        unsafe { buf.get_unchecked_mut(offset..offset + len) }
    }
    #[cfg(not(feature = "unchecked"))]
    &mut buf[offset..offset + len]
}

/// Get an immutable u16 slice from a buffer at a given offset.
#[inline(always)]
pub fn row_slice_u16(buf: &[u16], offset: usize, len: usize) -> &[u16] {
    #[cfg(feature = "unchecked")]
    {
        debug_assert!(offset + len <= buf.len());
        unsafe { buf.get_unchecked(offset..offset + len) }
    }
    #[cfg(not(feature = "unchecked"))]
    &buf[offset..offset + len]
}

/// Get a mutable u16 slice from a buffer at a given offset.
#[inline(always)]
pub fn row_slice_u16_mut(buf: &mut [u16], offset: usize, len: usize) -> &mut [u16] {
    #[cfg(feature = "unchecked")]
    {
        debug_assert!(offset + len <= buf.len());
        unsafe { buf.get_unchecked_mut(offset..offset + len) }
    }
    #[cfg(not(feature = "unchecked"))]
    &mut buf[offset..offset + len]
}

/// Index into a slice, with unchecked access when the feature is enabled.
#[inline(always)]
pub fn idx<T>(buf: &[T], i: usize) -> &T {
    #[cfg(feature = "unchecked")]
    {
        debug_assert!(i < buf.len());
        unsafe { buf.get_unchecked(i) }
    }
    #[cfg(not(feature = "unchecked"))]
    &buf[i]
}

/// Mutably index into a slice, with unchecked access when the feature is enabled.
#[inline(always)]
pub fn idx_mut<T>(buf: &mut [T], i: usize) -> &mut T {
    #[cfg(feature = "unchecked")]
    {
        debug_assert!(i < buf.len());
        unsafe { buf.get_unchecked_mut(i) }
    }
    #[cfg(not(feature = "unchecked"))]
    &mut buf[i]
}

/// Extension trait for slices: checked by default, unchecked with `unchecked` feature.
///
/// Use `slice.at(i)` instead of `slice[i]` in SIMD hot paths. When the
/// `unchecked` feature is enabled, indexing skips bounds checks (with
/// `debug_assert!` in debug builds). Otherwise, normal bounds-checked indexing.
pub trait SliceExt<T> {
    fn at(&self, i: usize) -> &T;
    fn at_mut(&mut self, i: usize) -> &mut T;
    fn sub(&self, start: usize, len: usize) -> &[T];
    fn sub_mut(&mut self, start: usize, len: usize) -> &mut [T];
}

impl<T> SliceExt<T> for [T] {
    #[inline(always)]
    fn at(&self, i: usize) -> &T {
        #[cfg(feature = "unchecked")]
        {
            debug_assert!(i < self.len());
            unsafe { self.get_unchecked(i) }
        }
        #[cfg(not(feature = "unchecked"))]
        &self[i]
    }

    #[inline(always)]
    fn at_mut(&mut self, i: usize) -> &mut T {
        #[cfg(feature = "unchecked")]
        {
            debug_assert!(i < self.len());
            unsafe { self.get_unchecked_mut(i) }
        }
        #[cfg(not(feature = "unchecked"))]
        &mut self[i]
    }

    #[inline(always)]
    fn sub(&self, start: usize, len: usize) -> &[T] {
        #[cfg(feature = "unchecked")]
        {
            debug_assert!(start + len <= self.len());
            unsafe { self.get_unchecked(start..start + len) }
        }
        #[cfg(not(feature = "unchecked"))]
        &self[start..start + len]
    }

    #[inline(always)]
    fn sub_mut(&mut self, start: usize, len: usize) -> &mut [T] {
        #[cfg(feature = "unchecked")]
        {
            debug_assert!(start + len <= self.len());
            unsafe { self.get_unchecked_mut(start..start + len) }
        }
        #[cfg(not(feature = "unchecked"))]
        &mut self[start..start + len]
    }
}

// ---------------------------------------------------------------------------
// FlexSlice — zero-cost `[]` wrapper with configurable bounds checking
// ---------------------------------------------------------------------------

/// Immutable slice wrapper: `slice.flex()[i]` uses normal `[]` syntax but
/// switches between checked (default) and unchecked (`unchecked` feature).
///
/// Verified zero-overhead: generates identical assembly to raw `get_unchecked`
/// in unchecked mode and identical to `[]` in checked mode.
pub struct FlexSlice<'a, T>(pub &'a [T]);

/// Mutable slice wrapper: `slice.flex_mut()[i]` with the same guarantees.
pub struct FlexSliceMut<'a, T>(pub &'a mut [T]);

impl<'a, T> FlexSlice<'a, T> {
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &'a [T] {
        self.0
    }

    #[inline(always)]
    pub fn iter(&self) -> core::slice::Iter<'a, T> {
        self.0.iter()
    }
}

impl<'a, T> FlexSliceMut<'a, T> {
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        self.0
    }

    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.0
    }

    #[inline(always)]
    pub fn iter(&self) -> core::slice::Iter<'_, T> {
        self.0.iter()
    }

    #[inline(always)]
    pub fn iter_mut(&mut self) -> core::slice::IterMut<'_, T> {
        self.0.iter_mut()
    }

    /// Re-borrow as immutable FlexSlice.
    #[inline(always)]
    pub fn flex(&self) -> FlexSlice<'_, T> {
        FlexSlice(self.0)
    }
}

impl<'a, T> core::ops::Deref for FlexSlice<'a, T> {
    type Target = [T];
    #[inline(always)]
    fn deref(&self) -> &[T] {
        self.0
    }
}

impl<'a, T> core::ops::Deref for FlexSliceMut<'a, T> {
    type Target = [T];
    #[inline(always)]
    fn deref(&self) -> &[T] {
        self.0
    }
}

impl<'a, T> core::ops::DerefMut for FlexSliceMut<'a, T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut [T] {
        self.0
    }
}

// --- FlexSlice Index impls ---

impl<T> core::ops::Index<usize> for FlexSlice<'_, T> {
    type Output = T;
    #[inline(always)]
    fn index(&self, i: usize) -> &T {
        #[cfg(feature = "unchecked")]
        {
            debug_assert!(i < self.0.len());
            unsafe { self.0.get_unchecked(i) }
        }
        #[cfg(not(feature = "unchecked"))]
        &self.0[i]
    }
}

impl<T> core::ops::Index<core::ops::Range<usize>> for FlexSlice<'_, T> {
    type Output = [T];
    #[inline(always)]
    fn index(&self, r: core::ops::Range<usize>) -> &[T] {
        #[cfg(feature = "unchecked")]
        {
            debug_assert!(r.end <= self.0.len());
            unsafe { self.0.get_unchecked(r) }
        }
        #[cfg(not(feature = "unchecked"))]
        &self.0[r]
    }
}

impl<T> core::ops::Index<core::ops::RangeFrom<usize>> for FlexSlice<'_, T> {
    type Output = [T];
    #[inline(always)]
    fn index(&self, r: core::ops::RangeFrom<usize>) -> &[T] {
        #[cfg(feature = "unchecked")]
        {
            debug_assert!(r.start <= self.0.len());
            unsafe { self.0.get_unchecked(r) }
        }
        #[cfg(not(feature = "unchecked"))]
        &self.0[r]
    }
}

impl<T> core::ops::Index<core::ops::RangeTo<usize>> for FlexSlice<'_, T> {
    type Output = [T];
    #[inline(always)]
    fn index(&self, r: core::ops::RangeTo<usize>) -> &[T] {
        #[cfg(feature = "unchecked")]
        {
            debug_assert!(r.end <= self.0.len());
            unsafe { self.0.get_unchecked(r) }
        }
        #[cfg(not(feature = "unchecked"))]
        &self.0[r]
    }
}

impl<T> core::ops::Index<core::ops::RangeFull> for FlexSlice<'_, T> {
    type Output = [T];
    #[inline(always)]
    fn index(&self, _r: core::ops::RangeFull) -> &[T] {
        self.0
    }
}

// --- FlexSliceMut Index impls ---

impl<T> core::ops::Index<usize> for FlexSliceMut<'_, T> {
    type Output = T;
    #[inline(always)]
    fn index(&self, i: usize) -> &T {
        #[cfg(feature = "unchecked")]
        {
            debug_assert!(i < self.0.len());
            unsafe { self.0.get_unchecked(i) }
        }
        #[cfg(not(feature = "unchecked"))]
        &self.0[i]
    }
}

impl<T> core::ops::IndexMut<usize> for FlexSliceMut<'_, T> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut T {
        #[cfg(feature = "unchecked")]
        {
            debug_assert!(i < self.0.len());
            unsafe { self.0.get_unchecked_mut(i) }
        }
        #[cfg(not(feature = "unchecked"))]
        &mut self.0[i]
    }
}

impl<T> core::ops::Index<core::ops::Range<usize>> for FlexSliceMut<'_, T> {
    type Output = [T];
    #[inline(always)]
    fn index(&self, r: core::ops::Range<usize>) -> &[T] {
        #[cfg(feature = "unchecked")]
        {
            debug_assert!(r.end <= self.0.len());
            unsafe { self.0.get_unchecked(r) }
        }
        #[cfg(not(feature = "unchecked"))]
        &self.0[r]
    }
}

impl<T> core::ops::IndexMut<core::ops::Range<usize>> for FlexSliceMut<'_, T> {
    #[inline(always)]
    fn index_mut(&mut self, r: core::ops::Range<usize>) -> &mut [T] {
        #[cfg(feature = "unchecked")]
        {
            debug_assert!(r.end <= self.0.len());
            unsafe { self.0.get_unchecked_mut(r) }
        }
        #[cfg(not(feature = "unchecked"))]
        &mut self.0[r]
    }
}

impl<T> core::ops::Index<core::ops::RangeFrom<usize>> for FlexSliceMut<'_, T> {
    type Output = [T];
    #[inline(always)]
    fn index(&self, r: core::ops::RangeFrom<usize>) -> &[T] {
        #[cfg(feature = "unchecked")]
        {
            debug_assert!(r.start <= self.0.len());
            unsafe { self.0.get_unchecked(r) }
        }
        #[cfg(not(feature = "unchecked"))]
        &self.0[r]
    }
}

impl<T> core::ops::IndexMut<core::ops::RangeFrom<usize>> for FlexSliceMut<'_, T> {
    #[inline(always)]
    fn index_mut(&mut self, r: core::ops::RangeFrom<usize>) -> &mut [T] {
        #[cfg(feature = "unchecked")]
        {
            debug_assert!(r.start <= self.0.len());
            unsafe { self.0.get_unchecked_mut(r) }
        }
        #[cfg(not(feature = "unchecked"))]
        &mut self.0[r]
    }
}

impl<T> core::ops::Index<core::ops::RangeTo<usize>> for FlexSliceMut<'_, T> {
    type Output = [T];
    #[inline(always)]
    fn index(&self, r: core::ops::RangeTo<usize>) -> &[T] {
        #[cfg(feature = "unchecked")]
        {
            debug_assert!(r.end <= self.0.len());
            unsafe { self.0.get_unchecked(r) }
        }
        #[cfg(not(feature = "unchecked"))]
        &self.0[r]
    }
}

impl<T> core::ops::IndexMut<core::ops::RangeTo<usize>> for FlexSliceMut<'_, T> {
    #[inline(always)]
    fn index_mut(&mut self, r: core::ops::RangeTo<usize>) -> &mut [T] {
        #[cfg(feature = "unchecked")]
        {
            debug_assert!(r.end <= self.0.len());
            unsafe { self.0.get_unchecked_mut(r) }
        }
        #[cfg(not(feature = "unchecked"))]
        &mut self.0[r]
    }
}

impl<T> core::ops::Index<core::ops::RangeFull> for FlexSliceMut<'_, T> {
    type Output = [T];
    #[inline(always)]
    fn index(&self, _r: core::ops::RangeFull) -> &[T] {
        self.0
    }
}

impl<T> core::ops::IndexMut<core::ops::RangeFull> for FlexSliceMut<'_, T> {
    #[inline(always)]
    fn index_mut(&mut self, _r: core::ops::RangeFull) -> &mut [T] {
        self.0
    }
}

/// Trait to get a `FlexSlice` or `FlexSliceMut` from any slice.
///
/// Use `slice.flex()[i]` in hot loops where you'd otherwise reach for
/// raw pointer arithmetic. Zero-cost: generates identical assembly to
/// both `[]` (checked) and `get_unchecked` (unchecked).
pub trait Flex<T> {
    fn flex(&self) -> FlexSlice<'_, T>;
    fn flex_mut(&mut self) -> FlexSliceMut<'_, T>;
}

impl<T> Flex<T> for [T] {
    #[inline(always)]
    fn flex(&self) -> FlexSlice<'_, T> {
        FlexSlice(self)
    }
    #[inline(always)]
    fn flex_mut(&mut self) -> FlexSliceMut<'_, T> {
        FlexSliceMut(self)
    }
}

/// Safely reinterpret a slice of `[Src; N]` as a slice of `[Dst; N]`
/// when both types have the same size and both implement zerocopy traits.
///
/// This is used to convert `&[LeftPixelRow<BD::Pixel>]` to `&[LeftPixelRow<u8>]`
/// or `&[LeftPixelRow<u16>]` in dispatch functions where the BPC is known at runtime.
///
/// Returns None if the byte layout doesn't match (wrong element size).
#[inline(always)]
pub fn reinterpret_slice<Src: AsBytes, Dst: FromBytes>(src: &[Src]) -> Option<&[Dst]> {
    let bytes = src.as_bytes();
    let r: Ref<&[u8], [Dst]> = Ref::new_slice(bytes)?;
    Some(r.into_slice())
}

/// Safely reinterpret a mutable slice using zerocopy.
#[inline(always)]
pub fn reinterpret_slice_mut<Src: AsBytes + FromBytes, Dst: AsBytes + FromBytes>(
    src: &mut [Src],
) -> Option<&mut [Dst]> {
    let bytes = src.as_bytes_mut();
    let r: Ref<&mut [u8], [Dst]> = Ref::new_slice(bytes)?;
    Some(r.into_mut_slice())
}

/// Safely reinterpret a fixed-size array reference using zerocopy.
/// The source and destination must have the same byte size.
#[inline(always)]
pub fn reinterpret_ref<Src: AsBytes, Dst: FromBytes>(src: &Src) -> Option<&Dst> {
    let bytes = src.as_bytes();
    let r: Ref<&[u8], Dst> = Ref::new(bytes)?;
    Some(r.into_ref())
}

/// Convert a raw pixel pointer + stride into a `(&mut [T], base_offset)` pair.
///
/// The returned slice covers the entire strided w×h region.
/// `base_offset` is the index within the slice corresponding to `ptr` (row 0).
///
/// For positive strides: slice starts at `ptr`, `base_offset = 0`.
/// For negative strides: slice starts at `ptr + (h-1)*stride` (the lowest address),
///   and `base_offset = (h-1) * abs(stride)`.
///
/// # Safety
///
/// - `ptr` must be valid for the strided w×h region
/// - For positive stride: `ptr[0 .. (h-1)*stride + w]` must be valid
/// - For negative stride: `ptr[(h-1)*stride .. w]` must be valid
#[cfg(feature = "asm")]
#[inline(always)]
pub unsafe fn strided_slice_from_ptr<'a, T>(
    ptr: *mut T,
    stride: isize,
    w: usize,
    h: usize,
) -> (&'a mut [T], usize) {
    if h == 0 {
        return (&mut [], 0);
    }
    let abs_stride = stride.unsigned_abs();
    let total = (h - 1) * abs_stride + w;
    if stride >= 0 {
        (std::slice::from_raw_parts_mut(ptr, total), 0)
    } else {
        let base = (h - 1) * abs_stride;
        let start = ptr.offset(-((base) as isize));
        (std::slice::from_raw_parts_mut(start, total), base)
    }
}

// =============================================================================
// SIMD Load/Store Macros
// =============================================================================
//
// These macros abstract over safe_unaligned_simd (bounds-checked, safe) and raw
// core::arch intrinsics (unchecked, pointer-based) depending on the `unchecked`
// feature flag.
//
// Each macro supports two forms:
//   - 1-arg: takes a typed array reference (&[u8; 32], &[u16; 16], etc.)
//   - 2-arg (load) / 3-arg (store): takes a slice + target type, does conversion
//
// All macros expand at the call site, inheriting the caller's #[target_feature].

// --- x86_64 AVX/SSE macros ---

/// Load 256 bits from a typed array reference or a dynamic slice.
///
/// **Array ref form:** `$src` must be a reference to a type implementing
/// `Is256BitsUnaligned` (e.g., `&[u8; 32]`, `&[u16; 16]`, `&[i16; 16]`, `&[i32; 8]`).
///
/// **Slice form:** `$slice` is `&[T]` and `$T` is the target array type.
/// Converts via `try_into().unwrap()` in checked mode; raw pointer in unchecked mode.
///
/// ```ignore
/// let v: __m256i = loadu_256!(&arr);                          // arr: [u8; 32]
/// let v: __m256i = loadu_256!(&src[off..off+32], [u8; 32]);   // from slice
/// ```
#[cfg(target_arch = "x86_64")]
macro_rules! loadu_256 {
    ($src:expr) => {{
        #[cfg(not(feature = "unchecked"))]
        {
            safe_unaligned_simd::x86_64::_mm256_loadu_si256($src)
        }
        #[cfg(feature = "unchecked")]
        {
            #[allow(unsafe_code)]
            unsafe {
                core::arch::x86_64::_mm256_loadu_si256(core::ptr::from_ref($src).cast())
            }
        }
    }};
    ($slice:expr, $T:ty) => {{
        #[cfg(not(feature = "unchecked"))]
        {
            safe_unaligned_simd::x86_64::_mm256_loadu_si256::<$T>(($slice).try_into().unwrap())
        }
        #[cfg(feature = "unchecked")]
        {
            let __s = $slice;
            debug_assert!(core::mem::size_of_val(__s) >= 32);
            #[allow(unsafe_code)]
            unsafe {
                core::arch::x86_64::_mm256_loadu_si256(__s.as_ptr() as *const _)
            }
        }
    }};
}
#[cfg(target_arch = "x86_64")]
pub(crate) use loadu_256;

/// Store 256 bits to a typed array reference or a dynamic slice.
///
/// **Array ref form:** `$dst` must be a mutable reference to a type implementing
/// `Is256BitsUnaligned`.
///
/// **Slice form:** `$slice` is `&mut [T]`, `$T` is the target array type,
/// `$val` is the `__m256i` value.
///
/// ```ignore
/// storeu_256!(&mut arr, v);                              // arr: [u8; 32]
/// storeu_256!(&mut dst[off..off+32], [u8; 32], v);       // from slice
/// ```
#[cfg(target_arch = "x86_64")]
macro_rules! storeu_256 {
    ($dst:expr, $val:expr) => {{
        #[cfg(not(feature = "unchecked"))]
        {
            safe_unaligned_simd::x86_64::_mm256_storeu_si256($dst, $val)
        }
        #[cfg(feature = "unchecked")]
        {
            #[allow(unsafe_code)]
            unsafe {
                core::arch::x86_64::_mm256_storeu_si256(core::ptr::from_mut($dst).cast(), $val)
            }
        }
    }};
    ($slice:expr, $T:ty, $val:expr) => {{
        #[cfg(not(feature = "unchecked"))]
        {
            safe_unaligned_simd::x86_64::_mm256_storeu_si256::<$T>(
                ($slice).try_into().unwrap(),
                $val,
            )
        }
        #[cfg(feature = "unchecked")]
        {
            let __s = $slice;
            debug_assert!(core::mem::size_of_val(__s) >= 32);
            #[allow(unsafe_code)]
            unsafe {
                core::arch::x86_64::_mm256_storeu_si256(__s.as_mut_ptr() as *mut _, $val)
            }
        }
    }};
}
#[cfg(target_arch = "x86_64")]
pub(crate) use storeu_256;

/// Load 512 bits from a typed array reference or a dynamic slice.
///
/// **Array ref form:** `$src` must be a reference to a type implementing
/// `Is512BitsUnaligned` (e.g., `&[u8; 64]`, `&[u16; 32]`, `&[i16; 32]`, `&[i32; 16]`).
///
/// **Slice form:** `$slice` is `&[T]` and `$T` is the target array type.
/// Converts via `try_into().unwrap()` in checked mode; raw pointer in unchecked mode.
///
/// ```ignore
/// let v: __m512i = loadu_512!(&arr);                          // arr: [u8; 64]
/// let v: __m512i = loadu_512!(&src[off..off+64], [u8; 64]);   // from slice
/// ```
#[cfg(target_arch = "x86_64")]
macro_rules! loadu_512 {
    ($src:expr) => {{
        #[cfg(not(feature = "unchecked"))]
        {
            safe_unaligned_simd::x86_64::_mm512_loadu_si512($src)
        }
        #[cfg(feature = "unchecked")]
        {
            #[allow(unsafe_code)]
            unsafe {
                core::arch::x86_64::_mm512_loadu_si512(core::ptr::from_ref($src).cast())
            }
        }
    }};
    ($slice:expr, $T:ty) => {{
        #[cfg(not(feature = "unchecked"))]
        {
            safe_unaligned_simd::x86_64::_mm512_loadu_si512::<$T>(($slice).try_into().unwrap())
        }
        #[cfg(feature = "unchecked")]
        {
            let __s = $slice;
            debug_assert!(core::mem::size_of_val(__s) >= 64);
            #[allow(unsafe_code)]
            unsafe {
                core::arch::x86_64::_mm512_loadu_si512(__s.as_ptr() as *const _)
            }
        }
    }};
}
#[cfg(target_arch = "x86_64")]
pub(crate) use loadu_512;

/// Store 512 bits to a typed array reference or a dynamic slice.
///
/// **Array ref form:** `$dst` must be a mutable reference to a type implementing
/// `Is512BitsUnaligned`.
///
/// **Slice form:** `$slice` is `&mut [T]`, `$T` is the target array type,
/// `$val` is the `__m512i` value.
///
/// ```ignore
/// storeu_512!(&mut arr, v);                              // arr: [u8; 64]
/// storeu_512!(&mut dst[off..off+64], [u8; 64], v);       // from slice
/// ```
#[cfg(target_arch = "x86_64")]
macro_rules! storeu_512 {
    ($dst:expr, $val:expr) => {{
        #[cfg(not(feature = "unchecked"))]
        {
            safe_unaligned_simd::x86_64::_mm512_storeu_si512($dst, $val)
        }
        #[cfg(feature = "unchecked")]
        {
            #[allow(unsafe_code)]
            unsafe {
                core::arch::x86_64::_mm512_storeu_si512(core::ptr::from_mut($dst).cast(), $val)
            }
        }
    }};
    ($slice:expr, $T:ty, $val:expr) => {{
        #[cfg(not(feature = "unchecked"))]
        {
            safe_unaligned_simd::x86_64::_mm512_storeu_si512::<$T>(
                ($slice).try_into().unwrap(),
                $val,
            )
        }
        #[cfg(feature = "unchecked")]
        {
            let __s = $slice;
            debug_assert!(core::mem::size_of_val(__s) >= 64);
            #[allow(unsafe_code)]
            unsafe {
                core::arch::x86_64::_mm512_storeu_si512(__s.as_mut_ptr() as *mut _, $val)
            }
        }
    }};
}
#[cfg(target_arch = "x86_64")]
pub(crate) use storeu_512;

/// Load 128 bits from a typed array reference or a dynamic slice.
///
/// **Array ref form:** `$src` must be a reference to a type implementing
/// `Is128BitsUnaligned` (e.g., `&[u8; 16]`, `&[u16; 8]`, `&[i16; 8]`, `&[i32; 4]`).
///
/// **Slice form:** `$slice` is `&[T]` and `$T` is the target array type.
///
/// ```ignore
/// let v: __m128i = loadu_128!(&arr);                          // arr: [u8; 16]
/// let v: __m128i = loadu_128!(&src[off..off+16], [u8; 16]);   // from slice
/// ```
#[cfg(target_arch = "x86_64")]
macro_rules! loadu_128 {
    ($src:expr) => {{
        #[cfg(not(feature = "unchecked"))]
        {
            safe_unaligned_simd::x86_64::_mm_loadu_si128($src)
        }
        #[cfg(feature = "unchecked")]
        {
            #[allow(unsafe_code)]
            unsafe {
                core::arch::x86_64::_mm_loadu_si128(core::ptr::from_ref($src).cast())
            }
        }
    }};
    ($slice:expr, $T:ty) => {{
        #[cfg(not(feature = "unchecked"))]
        {
            safe_unaligned_simd::x86_64::_mm_loadu_si128::<$T>(($slice).try_into().unwrap())
        }
        #[cfg(feature = "unchecked")]
        {
            let __s = $slice;
            debug_assert!(core::mem::size_of_val(__s) >= 16);
            #[allow(unsafe_code)]
            unsafe {
                core::arch::x86_64::_mm_loadu_si128(__s.as_ptr() as *const _)
            }
        }
    }};
}
#[cfg(target_arch = "x86_64")]
pub(crate) use loadu_128;

/// Store 128 bits to a typed array reference or a dynamic slice.
///
/// **Array ref form:** `$dst` must be a mutable reference to a type implementing
/// `Is128BitsUnaligned`.
///
/// **Slice form:** `$slice` is `&mut [T]`, `$T` is the target array type,
/// `$val` is the `__m128i` value.
///
/// ```ignore
/// storeu_128!(&mut arr, v);                              // arr: [u8; 16]
/// storeu_128!(&mut dst[off..off+16], [u8; 16], v);       // from slice
/// ```
#[cfg(target_arch = "x86_64")]
macro_rules! storeu_128 {
    ($dst:expr, $val:expr) => {{
        #[cfg(not(feature = "unchecked"))]
        {
            safe_unaligned_simd::x86_64::_mm_storeu_si128($dst, $val)
        }
        #[cfg(feature = "unchecked")]
        {
            #[allow(unsafe_code)]
            unsafe {
                core::arch::x86_64::_mm_storeu_si128(core::ptr::from_mut($dst).cast(), $val)
            }
        }
    }};
    ($slice:expr, $T:ty, $val:expr) => {{
        #[cfg(not(feature = "unchecked"))]
        {
            safe_unaligned_simd::x86_64::_mm_storeu_si128::<$T>(($slice).try_into().unwrap(), $val)
        }
        #[cfg(feature = "unchecked")]
        {
            let __s = $slice;
            debug_assert!(core::mem::size_of_val(__s) >= 16);
            #[allow(unsafe_code)]
            unsafe {
                core::arch::x86_64::_mm_storeu_si128(__s.as_mut_ptr() as *mut _, $val)
            }
        }
    }};
}
#[cfg(target_arch = "x86_64")]
pub(crate) use storeu_128;

/// Load 4 bytes from a slice as an `__m128i` (via `_mm_cvtsi32_si128`).
///
/// ```ignore
/// let v = loadi32!(&src[off..off+4]);
/// ```
#[cfg(target_arch = "x86_64")]
macro_rules! loadi32 {
    ($src:expr) => {{
        let bytes: &[u8] = $src;
        let val = i32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        core::arch::x86_64::_mm_cvtsi32_si128(val)
    }};
}
#[cfg(target_arch = "x86_64")]
pub(crate) use loadi32;

/// Store low 4 bytes of an `__m128i` to a slice (via `_mm_cvtsi128_si32`).
///
/// ```ignore
/// storei32!(&mut dst[off..off+4], v);
/// ```
#[cfg(target_arch = "x86_64")]
macro_rules! storei32 {
    ($dst:expr, $val:expr) => {{
        let val = core::arch::x86_64::_mm_cvtsi128_si32($val);
        let bytes = val.to_ne_bytes();
        let dst: &mut [u8] = $dst;
        dst[0] = bytes[0];
        dst[1] = bytes[1];
        dst[2] = bytes[2];
        dst[3] = bytes[3];
    }};
}
#[cfg(target_arch = "x86_64")]
pub(crate) use storei32;

/// Load 8 bytes from a slice as an `__m128i` (via `_mm_set_epi64x`).
///
/// ```ignore
/// let v = loadi64!(&src[off..off+8]);
/// ```
#[cfg(target_arch = "x86_64")]
macro_rules! loadi64 {
    ($src:expr) => {{
        let bytes: &[u8] = $src;
        let lo = i64::from_ne_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]);
        core::arch::x86_64::_mm_set_epi64x(0, lo)
    }};
}
#[cfg(target_arch = "x86_64")]
pub(crate) use loadi64;

/// Store low 8 bytes of an `__m128i` to a slice.
///
/// ```ignore
/// storei64!(&mut dst[off..off+8], v);
/// ```
#[cfg(target_arch = "x86_64")]
macro_rules! storei64 {
    ($dst:expr, $val:expr) => {{
        let val = core::arch::x86_64::_mm_cvtsi128_si64($val);
        let bytes = val.to_ne_bytes();
        let dst: &mut [u8] = $dst;
        dst[..8].copy_from_slice(&bytes);
    }};
}
#[cfg(target_arch = "x86_64")]
pub(crate) use storei64;

// --- aarch64 NEON macros ---

/// Load 128 bits (16 bytes) via NEON vld1q from a typed array reference.
///
/// `$src` must be a reference to a NEON-compatible array type
/// (e.g., `&[u8; 16]`, `&[u16; 8]`, `&[i16; 8]`, `&[u32; 4]`).
///
/// Returns the appropriate NEON vector type (uint8x16_t, uint16x8_t, etc.)
/// based on which variant is used.
///
/// ```ignore
/// let v: uint16x8_t = neon_ld1q_u16!(&arr); // arr: [u16; 8]
/// let v: uint8x16_t = neon_ld1q_u8!(&arr);  // arr: [u8; 16]
/// ```
#[cfg(target_arch = "aarch64")]
macro_rules! neon_ld1q_u8 {
    ($src:expr) => {{
        #[cfg(not(feature = "unchecked"))]
        {
            safe_unaligned_simd::aarch64::vld1q_u8($src)
        }
        #[cfg(feature = "unchecked")]
        {
            #[allow(unsafe_code)]
            unsafe {
                core::arch::aarch64::vld1q_u8(($src).as_ptr())
            }
        }
    }};
}
#[cfg(target_arch = "aarch64")]
pub(crate) use neon_ld1q_u8;

#[cfg(target_arch = "aarch64")]
macro_rules! neon_ld1q_u16 {
    ($src:expr) => {{
        #[cfg(not(feature = "unchecked"))]
        {
            safe_unaligned_simd::aarch64::vld1q_u16($src)
        }
        #[cfg(feature = "unchecked")]
        {
            #[allow(unsafe_code)]
            unsafe {
                core::arch::aarch64::vld1q_u16(($src).as_ptr())
            }
        }
    }};
}
#[cfg(target_arch = "aarch64")]
pub(crate) use neon_ld1q_u16;

#[cfg(target_arch = "aarch64")]
macro_rules! neon_ld1q_s16 {
    ($src:expr) => {{
        #[cfg(not(feature = "unchecked"))]
        {
            safe_unaligned_simd::aarch64::vld1q_s16($src)
        }
        #[cfg(feature = "unchecked")]
        {
            #[allow(unsafe_code)]
            unsafe {
                core::arch::aarch64::vld1q_s16(($src).as_ptr())
            }
        }
    }};
}
#[cfg(target_arch = "aarch64")]
pub(crate) use neon_ld1q_s16;

#[cfg(target_arch = "aarch64")]
macro_rules! neon_st1q_u8 {
    ($dst:expr, $val:expr) => {{
        #[cfg(not(feature = "unchecked"))]
        {
            safe_unaligned_simd::aarch64::vst1q_u8($dst, $val)
        }
        #[cfg(feature = "unchecked")]
        {
            #[allow(unsafe_code)]
            unsafe {
                core::arch::aarch64::vst1q_u8(($dst).as_mut_ptr(), $val)
            }
        }
    }};
}
#[cfg(target_arch = "aarch64")]
pub(crate) use neon_st1q_u8;

#[cfg(target_arch = "aarch64")]
macro_rules! neon_st1q_u16 {
    ($dst:expr, $val:expr) => {{
        #[cfg(not(feature = "unchecked"))]
        {
            safe_unaligned_simd::aarch64::vst1q_u16($dst, $val)
        }
        #[cfg(feature = "unchecked")]
        {
            #[allow(unsafe_code)]
            unsafe {
                core::arch::aarch64::vst1q_u16(($dst).as_mut_ptr(), $val)
            }
        }
    }};
}
#[cfg(target_arch = "aarch64")]
pub(crate) use neon_st1q_u16;

#[cfg(target_arch = "aarch64")]
macro_rules! neon_st1q_s16 {
    ($dst:expr, $val:expr) => {{
        #[cfg(not(feature = "unchecked"))]
        {
            safe_unaligned_simd::aarch64::vst1q_s16($dst, $val)
        }
        #[cfg(feature = "unchecked")]
        {
            #[allow(unsafe_code)]
            unsafe {
                core::arch::aarch64::vst1q_s16(($dst).as_mut_ptr(), $val)
            }
        }
    }};
}
#[cfg(target_arch = "aarch64")]
pub(crate) use neon_st1q_s16;
