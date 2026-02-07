//! Safe pixel access helpers for SIMD modules.
//!
//! When the `unchecked` feature is enabled, these use unchecked indexing
//! for performance parity with raw pointer access. Otherwise, they use
//! normal bounds-checked indexing.

#![cfg_attr(not(feature = "unchecked"), deny(unsafe_code))]

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
