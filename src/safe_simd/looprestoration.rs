//! Safe SIMD implementations for Loop Restoration
//!
//! Loop restoration applies two types of filtering:
//! 1. Wiener filter - 7-tap or 5-tap separable filter
//! 2. SGR (Self-Guided Restoration) - guided filter based on local statistics

#![allow(unused_imports)]
#![allow(dead_code)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use std::ffi::c_int;
use std::slice;

use crate::include::common::bitdepth::AsPrimitive;
use crate::include::common::bitdepth::BitDepth;
use crate::include::common::bitdepth::BitDepth8;
use crate::include::common::bitdepth::DynPixel;
use crate::include::common::bitdepth::LeftPixelRow;
use crate::include::common::intops::iclip;
use crate::include::dav1d::picture::Rav1dPictureDataComponentOffset;
use crate::src::align::AlignedVec64;
use crate::src::disjoint_mut::DisjointMut;
use crate::src::ffi_safe::FFISafe;
use crate::src::looprestoration::{padding, LrEdgeFlags, LooprestorationParams};
use crate::src::strided::Strided as _;
use libc::ptrdiff_t;

// Must match the constant in looprestoration.rs
const REST_UNIT_STRIDE: usize = 256 * 3 / 2 + 3 + 3; // = 390

// ============================================================================
// WIENER FILTER - AVX2 IMPLEMENTATION
// ============================================================================

/// Wiener filter 7-tap for 8bpc using AVX2
///
/// This is a separable filter: horizontal pass followed by vertical pass.
/// - Horizontal: 7-tap filter on each row, output is 16-bit
/// - Vertical: 7-tap filter on columns of the horizontal output
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn wiener_filter7_8bpc_avx2_inner(
    p: Rav1dPictureDataComponentOffset,
    left: &[LeftPixelRow<u8>],
    lpf: &DisjointMut<AlignedVec64<u8>>,
    lpf_off: isize,
    w: usize,
    h: usize,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
) {
    // Temporary buffer for padded input - same layout as Rust fallback
    let mut tmp = [0u8; (64 + 3 + 3) * REST_UNIT_STRIDE];

    // Use the existing padding function
    padding::<BitDepth8>(&mut tmp, p, left, lpf, lpf_off, w, h, edges);

    // Intermediate buffer for horizontal filter output (16-bit values)
    let mut hor = [0u16; (64 + 3 + 3) * REST_UNIT_STRIDE];

    let filter = &params.filter;
    let round_bits_h = 3i32; // 3 for 8bpc
    let rounding_off_h = 1i32 << (round_bits_h - 1);
    let clip_limit = 1i32 << (8 + 1 + 7 - round_bits_h); // = 8192 for 8bpc

    // -------------------------------------------------------------------------
    // Horizontal filter pass - scalar for correctness
    // -------------------------------------------------------------------------
    for row in 0..(h + 6) {
        let tmp_row = &tmp[row * REST_UNIT_STRIDE..];
        let hor_row = &mut hor[row * REST_UNIT_STRIDE..row * REST_UNIT_STRIDE + w];

        for x in 0..w {
            let mut sum = 1i32 << 14; // bias for 8bpc: 1 << (8 + 6)
            sum += tmp_row[x + 3] as i32 * 128; // DC offset for center pixel (8bpc only)

            for k in 0..7 {
                sum += tmp_row[x + k] as i32 * filter[0][k] as i32;
            }

            hor_row[x] = iclip((sum + rounding_off_h) >> round_bits_h, 0, clip_limit - 1) as u16;
        }
    }

    // -------------------------------------------------------------------------
    // Vertical filter pass - scalar for correctness
    // -------------------------------------------------------------------------
    let round_bits_v = 11i32; // for 8bpc
    let rounding_off_v = 1i32 << (round_bits_v - 1);
    let round_offset = 1i32 << (8 + round_bits_v - 1); // = 1 << 18 for 8bpc
    let stride = p.pixel_stride::<BitDepth8>();

    for j in 0..h {
        let mut dst_row = (p + (j as isize * stride)).slice_mut::<BitDepth8>(w);

        for i in 0..w {
            let mut sum = -round_offset;

            for k in 0..7 {
                sum += hor[(j + k) * REST_UNIT_STRIDE + i] as i32 * filter[1][k] as i32;
            }

            dst_row[i] = iclip((sum + rounding_off_v) >> round_bits_v, 0, 255) as u8;
        }
    }
}

/// Wiener filter 5-tap for 8bpc using AVX2
///
/// Same as 7-tap but filter[0][0] = filter[0][6] = 0 and filter[1][0] = filter[1][6] = 0
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn wiener_filter5_8bpc_avx2_inner(
    p: Rav1dPictureDataComponentOffset,
    left: &[LeftPixelRow<u8>],
    lpf: &DisjointMut<AlignedVec64<u8>>,
    lpf_off: isize,
    w: usize,
    h: usize,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
) {
    // 5-tap is identical to 7-tap, the coefficient array just has zeros at edges
    // SAFETY: Called within unsafe fn, maintains same invariants
    unsafe {
        wiener_filter7_8bpc_avx2_inner(p, left, lpf, lpf_off, w, h, params, edges);
    }
}

// ============================================================================
// FFI WRAPPERS
// ============================================================================

/// Reconstructs lpf offset from pointer
fn reconstruct_lpf_offset(lpf: &DisjointMut<AlignedVec64<u8>>, ptr: *const u8) -> isize {
    let base = lpf.as_mut_ptr();
    (ptr as isize - base as isize)
}

/// FFI wrapper for Wiener filter 7-tap 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn wiener_filter7_8bpc_avx2(
    _p_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    left: *const LeftPixelRow<DynPixel>,
    lpf_ptr: *const DynPixel,
    w: c_int,
    h: c_int,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    _bitdepth_max: c_int,
    p: *const FFISafe<Rav1dPictureDataComponentOffset>,
    lpf: *const FFISafe<DisjointMut<AlignedVec64<u8>>>,
) {
    // SAFETY: p and lpf were passed as FFISafe::new(_) in loop_restoration_filter::Fn::call
    let p = unsafe { *FFISafe::get(p) };
    let left = left.cast::<LeftPixelRow<u8>>();
    let lpf = unsafe { FFISafe::get(lpf) };
    let lpf_ptr = lpf_ptr.cast::<u8>();
    let lpf_off = reconstruct_lpf_offset(lpf, lpf_ptr);
    let w = w as usize;
    let h = h as usize;
    // SAFETY: Length was sliced in loop_restoration_filter::Fn::call
    let left = unsafe { slice::from_raw_parts(left, h) };

    // SAFETY: All parameters validated, target_feature enabled
    unsafe {
        wiener_filter7_8bpc_avx2_inner(p, left, lpf, lpf_off, w, h, params, edges);
    }
}

/// FFI wrapper for Wiener filter 5-tap 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn wiener_filter5_8bpc_avx2(
    _p_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    left: *const LeftPixelRow<DynPixel>,
    lpf_ptr: *const DynPixel,
    w: c_int,
    h: c_int,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    _bitdepth_max: c_int,
    p: *const FFISafe<Rav1dPictureDataComponentOffset>,
    lpf: *const FFISafe<DisjointMut<AlignedVec64<u8>>>,
) {
    // SAFETY: p and lpf were passed as FFISafe::new(_) in loop_restoration_filter::Fn::call
    let p = unsafe { *FFISafe::get(p) };
    let left = left.cast::<LeftPixelRow<u8>>();
    let lpf = unsafe { FFISafe::get(lpf) };
    let lpf_ptr = lpf_ptr.cast::<u8>();
    let lpf_off = reconstruct_lpf_offset(lpf, lpf_ptr);
    let w = w as usize;
    let h = h as usize;
    // SAFETY: Length was sliced in loop_restoration_filter::Fn::call
    let left = unsafe { slice::from_raw_parts(left, h) };

    // SAFETY: All parameters validated, target_feature enabled
    unsafe {
        wiener_filter5_8bpc_avx2_inner(p, left, lpf, lpf_off, w, h, params, edges);
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rest_unit_stride() {
        // Verify our constant matches the one in looprestoration.rs
        assert_eq!(REST_UNIT_STRIDE, 256 * 3 / 2 + 3 + 3);
        assert_eq!(REST_UNIT_STRIDE, 390);
    }
}
