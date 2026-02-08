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

#![cfg_attr(not(feature = "unchecked"), deny(unsafe_code))]

use zerocopy::{AsBytes, FromBytes, Ref};

/// Get an immutable slice from a buffer at a given offset.
#[inline(always)]
pub fn row_slice(buf: &[u8], offset: usize, len: usize) -> &[u8] {
    #[cfg(feature = "unchecked")]
    {
        debug_assert!(offset + len <= buf.len());
        // SAFETY: caller guarantees offset + len <= buf.len()
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

// =============================================================================
// SIMD Load/Store Macros
// =============================================================================
//
// These macros abstract over safe_unaligned_simd (bounds-checked, safe) and raw
// core::arch intrinsics (unchecked, pointer-based) depending on the `unchecked`
// feature flag.
//
// Two forms per operation:
//   - Direct: takes a typed array reference (&[u8; 32], &[u16; 16], etc.)
//   - From-slice: takes a dynamically-sized slice + target type, does conversion
//
// All macros expand at the call site, inheriting the caller's #[target_feature].

// --- x86_64 AVX/SSE macros ---

/// Load 256 bits from a typed array reference.
///
/// `$src` must be a reference to a type implementing `Is256BitsUnaligned`
/// (e.g., `&[u8; 32]`, `&[u16; 16]`, `&[i16; 16]`, `&[i32; 8]`).
///
/// ```ignore
/// let v: __m256i = loadu_256!(&arr); // arr: [u8; 32]
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
            unsafe {
                core::arch::x86_64::_mm256_loadu_si256(core::ptr::from_ref($src).cast())
            }
        }
    }};
}
#[cfg(target_arch = "x86_64")]
pub(crate) use loadu_256;

/// Store 256 bits to a typed array reference.
///
/// `$dst` must be a mutable reference to a type implementing `Is256BitsUnaligned`.
///
/// ```ignore
/// storeu_256!(&mut arr, v); // arr: [u8; 32]
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
            unsafe {
                core::arch::x86_64::_mm256_storeu_si256(core::ptr::from_mut($dst).cast(), $val)
            }
        }
    }};
}
#[cfg(target_arch = "x86_64")]
pub(crate) use storeu_256;

/// Load 128 bits from a typed array reference.
///
/// `$src` must be a reference to a type implementing `Is128BitsUnaligned`
/// (e.g., `&[u8; 16]`, `&[u16; 8]`, `&[i16; 8]`, `&[i32; 4]`).
#[cfg(target_arch = "x86_64")]
macro_rules! loadu_128 {
    ($src:expr) => {{
        #[cfg(not(feature = "unchecked"))]
        {
            safe_unaligned_simd::x86_64::_mm_loadu_si128($src)
        }
        #[cfg(feature = "unchecked")]
        {
            unsafe {
                core::arch::x86_64::_mm_loadu_si128(core::ptr::from_ref($src).cast())
            }
        }
    }};
}
#[cfg(target_arch = "x86_64")]
pub(crate) use loadu_128;

/// Store 128 bits to a typed array reference.
#[cfg(target_arch = "x86_64")]
macro_rules! storeu_128 {
    ($dst:expr, $val:expr) => {{
        #[cfg(not(feature = "unchecked"))]
        {
            safe_unaligned_simd::x86_64::_mm_storeu_si128($dst, $val)
        }
        #[cfg(feature = "unchecked")]
        {
            unsafe {
                core::arch::x86_64::_mm_storeu_si128(core::ptr::from_mut($dst).cast(), $val)
            }
        }
    }};
}
#[cfg(target_arch = "x86_64")]
pub(crate) use storeu_128;

/// Load 256 bits from a dynamic slice, converting to a fixed-size array.
///
/// `$slice` is `&[T]` and `$T` is the target array type (e.g., `[u8; 32]`).
/// When unchecked is off, bounds-checks via `try_into().unwrap()`.
/// When unchecked is on, uses raw pointer access with `debug_assert!`.
///
/// ```ignore
/// let v = load_256!(&src[off..off+32], [u8; 32]);
/// let v = load_256!(&src[off..off+16], [u16; 16]);
/// ```
#[cfg(target_arch = "x86_64")]
macro_rules! load_256 {
    ($slice:expr, $T:ty) => {{
        #[cfg(not(feature = "unchecked"))]
        {
            use safe_unaligned_simd::x86_64 as __sus;
            __sus::_mm256_loadu_si256::<$T>(($slice).try_into().unwrap())
        }
        #[cfg(feature = "unchecked")]
        {
            let __s = $slice;
            debug_assert!(__s.len() * core::mem::size_of_val(&__s[0]) >= 32);
            unsafe {
                core::arch::x86_64::_mm256_loadu_si256(__s.as_ptr() as *const _)
            }
        }
    }};
}
#[cfg(target_arch = "x86_64")]
pub(crate) use load_256;

/// Store 256 bits to a dynamic slice, converting to a fixed-size array.
///
/// ```ignore
/// store_256!(&mut dst[off..off+32], [u8; 32], v);
/// ```
#[cfg(target_arch = "x86_64")]
macro_rules! store_256 {
    ($slice:expr, $T:ty, $val:expr) => {{
        #[cfg(not(feature = "unchecked"))]
        {
            use safe_unaligned_simd::x86_64 as __sus;
            __sus::_mm256_storeu_si256::<$T>(($slice).try_into().unwrap(), $val)
        }
        #[cfg(feature = "unchecked")]
        {
            let __s = $slice;
            debug_assert!(__s.len() * core::mem::size_of_val(&__s[0]) >= 32);
            unsafe {
                core::arch::x86_64::_mm256_storeu_si256(__s.as_mut_ptr() as *mut _, $val)
            }
        }
    }};
}
#[cfg(target_arch = "x86_64")]
pub(crate) use store_256;

/// Load 128 bits from a dynamic slice, converting to a fixed-size array.
///
/// ```ignore
/// let v = load_128!(&src[off..off+16], [u8; 16]);
/// ```
#[cfg(target_arch = "x86_64")]
macro_rules! load_128 {
    ($slice:expr, $T:ty) => {{
        #[cfg(not(feature = "unchecked"))]
        {
            use safe_unaligned_simd::x86_64 as __sus;
            __sus::_mm_loadu_si128::<$T>(($slice).try_into().unwrap())
        }
        #[cfg(feature = "unchecked")]
        {
            let __s = $slice;
            debug_assert!(__s.len() * core::mem::size_of_val(&__s[0]) >= 16);
            unsafe {
                core::arch::x86_64::_mm_loadu_si128(__s.as_ptr() as *const _)
            }
        }
    }};
}
#[cfg(target_arch = "x86_64")]
pub(crate) use load_128;

/// Store 128 bits to a dynamic slice, converting to a fixed-size array.
///
/// ```ignore
/// store_128!(&mut dst[off..off+16], [u8; 16], v);
/// ```
#[cfg(target_arch = "x86_64")]
macro_rules! store_128 {
    ($slice:expr, $T:ty, $val:expr) => {{
        #[cfg(not(feature = "unchecked"))]
        {
            use safe_unaligned_simd::x86_64 as __sus;
            __sus::_mm_storeu_si128::<$T>(($slice).try_into().unwrap(), $val)
        }
        #[cfg(feature = "unchecked")]
        {
            let __s = $slice;
            debug_assert!(__s.len() * core::mem::size_of_val(&__s[0]) >= 16);
            unsafe {
                core::arch::x86_64::_mm_storeu_si128(__s.as_mut_ptr() as *mut _, $val)
            }
        }
    }};
}
#[cfg(target_arch = "x86_64")]
pub(crate) use store_128;

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
            unsafe { core::arch::aarch64::vld1q_u8(($src).as_ptr()) }
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
            unsafe { core::arch::aarch64::vld1q_u16(($src).as_ptr()) }
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
            unsafe { core::arch::aarch64::vld1q_s16(($src).as_ptr()) }
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
            unsafe { core::arch::aarch64::vst1q_u8(($dst).as_mut_ptr(), $val) }
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
            unsafe { core::arch::aarch64::vst1q_u16(($dst).as_mut_ptr(), $val) }
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
            unsafe { core::arch::aarch64::vst1q_s16(($dst).as_mut_ptr(), $val) }
        }
    }};
}
#[cfg(target_arch = "aarch64")]
pub(crate) use neon_st1q_s16;
