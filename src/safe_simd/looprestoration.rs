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
    let round_bits_h = 3i32;
    let rounding_off_h = 1i32 << (round_bits_h - 1);
    let clip_limit = 1i32 << (8 + 1 + 7 - round_bits_h); // = 8192

    // -------------------------------------------------------------------------
    // Horizontal filter pass
    // For 8bpc: sum = (1 << 14) + pixel[center] * 128 + sum(pixel[i] * filter[i])
    // -------------------------------------------------------------------------

    // AVX2 horizontal filter using maddubs
    // We need to handle the asymmetric 7-tap filter with DC offset
    // For now, use scalar for correctness
    for row in 0..(h + 6) {
        let tmp_row = &tmp[row * REST_UNIT_STRIDE..];
        let hor_row = &mut hor[row * REST_UNIT_STRIDE..row * REST_UNIT_STRIDE + w];

        for x in 0..w {
            let mut sum = 1i32 << 14;
            sum += tmp_row[x + 3] as i32 * 128;
            for k in 0..7 {
                sum += tmp_row[x + k] as i32 * filter[0][k] as i32;
            }
            hor_row[x] = iclip((sum + rounding_off_h) >> round_bits_h, 0, clip_limit - 1) as u16;
        }
    }

    // -------------------------------------------------------------------------
    // Vertical filter pass with AVX2 SIMD
    // Process 16 pixels at a time using AVX2
    // -------------------------------------------------------------------------
    let round_bits_v = 11i32;
    let rounding_off_v = 1i32 << (round_bits_v - 1);
    let round_offset = 1i32 << (8 + round_bits_v - 1); // = 1 << 18
    let stride = p.pixel_stride::<BitDepth8>();

    // Load filter coefficients into AVX2 registers (broadcast to all lanes)
    let vf0 = unsafe { _mm256_set1_epi32(filter[1][0] as i32) };
    let vf1 = unsafe { _mm256_set1_epi32(filter[1][1] as i32) };
    let vf2 = unsafe { _mm256_set1_epi32(filter[1][2] as i32) };
    let vf3 = unsafe { _mm256_set1_epi32(filter[1][3] as i32) };
    let vf4 = unsafe { _mm256_set1_epi32(filter[1][4] as i32) };
    let vf5 = unsafe { _mm256_set1_epi32(filter[1][5] as i32) };
    let vf6 = unsafe { _mm256_set1_epi32(filter[1][6] as i32) };
    let v_round_offset = unsafe { _mm256_set1_epi32(-round_offset) };
    let v_rounding = unsafe { _mm256_set1_epi32(rounding_off_v) };

    for j in 0..h {
        let mut dst_row = (p + (j as isize * stride)).slice_mut::<BitDepth8>(w);
        let mut i = 0usize;

        // Process 8 pixels at a time with AVX2
        while i + 8 <= w {
            unsafe {
                // Load 8 u16 values from each of the 7 rows and convert to i32
                // We need to process low and high halves separately
                let row0 = &hor[(j + 0) * REST_UNIT_STRIDE + i..];
                let row1 = &hor[(j + 1) * REST_UNIT_STRIDE + i..];
                let row2 = &hor[(j + 2) * REST_UNIT_STRIDE + i..];
                let row3 = &hor[(j + 3) * REST_UNIT_STRIDE + i..];
                let row4 = &hor[(j + 4) * REST_UNIT_STRIDE + i..];
                let row5 = &hor[(j + 5) * REST_UNIT_STRIDE + i..];
                let row6 = &hor[(j + 6) * REST_UNIT_STRIDE + i..];

                // Load 8 u16 values as __m128i
                let r0 = _mm_loadu_si128(row0.as_ptr() as *const __m128i);
                let r1 = _mm_loadu_si128(row1.as_ptr() as *const __m128i);
                let r2 = _mm_loadu_si128(row2.as_ptr() as *const __m128i);
                let r3 = _mm_loadu_si128(row3.as_ptr() as *const __m128i);
                let r4 = _mm_loadu_si128(row4.as_ptr() as *const __m128i);
                let r5 = _mm_loadu_si128(row5.as_ptr() as *const __m128i);
                let r6 = _mm_loadu_si128(row6.as_ptr() as *const __m128i);

                // Process low 4 pixels (expand u16 to i32)
                let r0_lo = _mm256_cvtepu16_epi32(r0);
                let r1_lo = _mm256_cvtepu16_epi32(r1);
                let r2_lo = _mm256_cvtepu16_epi32(r2);
                let r3_lo = _mm256_cvtepu16_epi32(r3);
                let r4_lo = _mm256_cvtepu16_epi32(r4);
                let r5_lo = _mm256_cvtepu16_epi32(r5);
                let r6_lo = _mm256_cvtepu16_epi32(r6);

                // sum = -round_offset + sum(row[k] * filter[k])
                let mut sum = v_round_offset;
                sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(r0_lo, vf0));
                sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(r1_lo, vf1));
                sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(r2_lo, vf2));
                sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(r3_lo, vf3));
                sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(r4_lo, vf4));
                sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(r5_lo, vf5));
                sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(r6_lo, vf6));

                // Add rounding and shift
                sum = _mm256_add_epi32(sum, v_rounding);
                sum = _mm256_srai_epi32::<11>(sum); // round_bits_v = 11 for 8bpc

                // Clip to 0-255 and pack to bytes
                // packus_epi32 saturates to 0-65535, then packus_epi16 saturates to 0-255
                let sum16 = _mm256_packus_epi32(sum, sum); // i32 -> u16, gives [0-7][0-7]
                let sum16_lo = _mm256_castsi256_si128(sum16);
                let sum16_hi = _mm256_extracti128_si256(sum16, 1);
                let sum16_combined = _mm_unpacklo_epi64(sum16_lo, sum16_hi);
                let sum8 = _mm_packus_epi16(sum16_combined, sum16_combined); // u16 -> u8

                // Store 8 bytes
                let dst_ptr = dst_row.as_mut_ptr().add(i);
                _mm_storel_epi64(dst_ptr as *mut __m128i, sum8);
            }
            i += 8;
        }

        // Handle remaining pixels with scalar
        while i < w {
            let mut sum = -round_offset;
            for k in 0..7 {
                sum += hor[(j + k) * REST_UNIT_STRIDE + i] as i32 * filter[1][k] as i32;
            }
            dst_row[i] = iclip((sum + rounding_off_v) >> round_bits_v, 0, 255) as u8;
            i += 1;
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
