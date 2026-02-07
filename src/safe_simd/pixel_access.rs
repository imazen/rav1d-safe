//! Safe pixel access helpers for SIMD modules.
//!
//! When the `unchecked` feature is enabled, these use unchecked indexing
//! for performance parity with raw pointer access. Otherwise, they use
//! normal bounds-checked indexing.

#![cfg_attr(not(feature = "unchecked"), deny(unsafe_code))]

use zerocopy::{AsBytes, FromBytes, Ref};

/// Get an immutable slice from a buffer at a given offset.
#[inline(always)]
pub fn row_slice(buf: &[u8], offset: usize, len: usize) -> &[u8] {
    #[cfg(feature = "unchecked")]
    {
        debug_assert!(offset + len <= buf.len());
        // SAFETY: caller guarantees offset + len <= buf.len()
        #[allow(unsafe_code)]
        unsafe {
            buf.get_unchecked(offset..offset + len)
        }
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
        #[allow(unsafe_code)]
        unsafe {
            buf.get_unchecked_mut(offset..offset + len)
        }
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
        #[allow(unsafe_code)]
        unsafe {
            buf.get_unchecked(offset..offset + len)
        }
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
        #[allow(unsafe_code)]
        unsafe {
            buf.get_unchecked_mut(offset..offset + len)
        }
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
        #[allow(unsafe_code)]
        unsafe {
            buf.get_unchecked(i)
        }
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
        #[allow(unsafe_code)]
        unsafe {
            buf.get_unchecked_mut(i)
        }
    }
    #[cfg(not(feature = "unchecked"))]
    &mut buf[i]
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
