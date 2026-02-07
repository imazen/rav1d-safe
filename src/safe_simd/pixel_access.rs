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

/// Safely reinterpret a fixed-size array reference using zerocopy.
/// The source and destination must have the same byte size.
#[inline(always)]
pub fn reinterpret_ref<Src: AsBytes, Dst: FromBytes>(src: &Src) -> Option<&Dst> {
    let bytes = src.as_bytes();
    let r: Ref<&[u8], Dst> = Ref::new(bytes)?;
    Some(r.into_ref())
}
