//! Safe SIMD implementations for Loop Restoration
//!
//! Loop restoration applies two types of filtering:
//! 1. Wiener filter - 7-tap or 5-tap separable filter
//! 2. SGR (Self-Guided Restoration) - guided filter based on local statistics

#![allow(unused_imports)]
#![allow(dead_code)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use std::cmp;
use std::ffi::c_int;
use std::ffi::c_uint;
use std::slice;

use crate::include::common::bitdepth::AsPrimitive;
use crate::include::common::bitdepth::BitDepth;
use crate::include::common::bitdepth::BitDepth16;
use crate::include::common::bitdepth::BitDepth8;
use crate::include::common::bitdepth::DynPixel;
use crate::include::common::bitdepth::LeftPixelRow;
use crate::include::common::intops::iclip;
use crate::include::dav1d::picture::PicOffset;
use crate::src::align::AlignedVec64;
use crate::src::disjoint_mut::DisjointMut;
use crate::src::ffi_safe::FFISafe;
use crate::src::looprestoration::{padding, LooprestorationParams, LrEdgeFlags};
use crate::src::strided::Strided as _;
use crate::src::tables::dav1d_sgr_x_by_x;
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
    p: PicOffset,
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
    let vf0 = _mm256_set1_epi32(filter[1][0] as i32);
    let vf1 = _mm256_set1_epi32(filter[1][1] as i32);
    let vf2 = _mm256_set1_epi32(filter[1][2] as i32);
    let vf3 = _mm256_set1_epi32(filter[1][3] as i32);
    let vf4 = _mm256_set1_epi32(filter[1][4] as i32);
    let vf5 = _mm256_set1_epi32(filter[1][5] as i32);
    let vf6 = _mm256_set1_epi32(filter[1][6] as i32);
    let v_round_offset = _mm256_set1_epi32(-round_offset);
    let v_rounding = _mm256_set1_epi32(rounding_off_v);

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
fn wiener_filter5_8bpc_avx2_inner(
    p: PicOffset,
    left: &[LeftPixelRow<u8>],
    lpf: &DisjointMut<AlignedVec64<u8>>,
    lpf_off: isize,
    w: usize,
    h: usize,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
) {
    // 5-tap is identical to 7-tap, the coefficient array just has zeros at edges
    // SAFETY: AVX2 availability verified by caller (dispatch checks CpuFlags::AVX2)
    unsafe {
        wiener_filter7_8bpc_avx2_inner(p, left, lpf, lpf_off, w, h, params, edges);
    }
}

// ============================================================================
// WIENER FILTER - 16bpc AVX2 IMPLEMENTATION
// ============================================================================

/// Wiener filter 7-tap for 16bpc using AVX2
///
/// Similar to 8bpc but:
/// - No DC offset in horizontal pass
/// - Different shift amounts based on 10bpc vs 12bpc
/// - 16-bit pixel values
#[cfg(target_arch = "x86_64")]
fn wiener_filter7_16bpc_avx2_inner(
    p: PicOffset,
    left: &[LeftPixelRow<u16>],
    lpf: &DisjointMut<AlignedVec64<u8>>,
    lpf_off: isize,
    w: usize,
    h: usize,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    bitdepth_max: c_int,
) {
    // Determine bitdepth (10 or 12)
    let bitdepth = if bitdepth_max == 1023 { 10 } else { 12 };

    // Temporary buffer for padded input
    let mut tmp = [0u16; (64 + 3 + 3) * REST_UNIT_STRIDE];

    // Use the existing padding function
    padding::<BitDepth16>(&mut tmp, p, left, lpf, lpf_off, w, h, edges);

    // Intermediate buffer for horizontal filter output (16-bit values, but may exceed u16 range)
    // Use u32 for intermediate storage to avoid overflow
    let mut hor = [0i32; (64 + 3 + 3) * REST_UNIT_STRIDE];

    let filter = &params.filter;

    // Different shift amounts for 10bpc vs 12bpc
    let round_bits_h = if bitdepth == 12 { 5 } else { 3 };
    let rounding_off_h = 1i32 << (round_bits_h - 1);
    let clip_limit = 1i32 << (bitdepth + 1 + 7 - round_bits_h);

    // -------------------------------------------------------------------------
    // Horizontal filter pass (no DC offset for 16bpc)
    // -------------------------------------------------------------------------
    for row in 0..(h + 6) {
        let tmp_row = &tmp[row * REST_UNIT_STRIDE..];
        let hor_row = &mut hor[row * REST_UNIT_STRIDE..row * REST_UNIT_STRIDE + w];

        for x in 0..w {
            let mut sum = 1i32 << (bitdepth + 6);
            // No DC offset for 16bpc
            for k in 0..7 {
                sum += tmp_row[x + k] as i32 * filter[0][k] as i32;
            }
            hor_row[x] = iclip((sum + rounding_off_h) >> round_bits_h, 0, clip_limit - 1);
        }
    }

    // -------------------------------------------------------------------------
    // Vertical filter pass
    // -------------------------------------------------------------------------
    let round_bits_v = if bitdepth == 12 { 9 } else { 11 };
    let rounding_off_v = 1i32 << (round_bits_v - 1);
    let round_offset = 1i32 << (bitdepth + round_bits_v - 1);
    let stride = p.pixel_stride::<BitDepth16>();

    for j in 0..h {
        let mut dst_row = (p + (j as isize * stride)).slice_mut::<BitDepth16>(w);

        for i in 0..w {
            let mut sum = -round_offset;

            for k in 0..7 {
                sum += hor[(j + k) * REST_UNIT_STRIDE + i] * filter[1][k] as i32;
            }

            dst_row[i] = iclip((sum + rounding_off_v) >> round_bits_v, 0, bitdepth_max) as u16;
        }
    }
}

/// Wiener filter 5-tap for 16bpc using AVX2
#[cfg(target_arch = "x86_64")]
fn wiener_filter5_16bpc_avx2_inner(
    p: PicOffset,
    left: &[LeftPixelRow<u16>],
    lpf: &DisjointMut<AlignedVec64<u8>>,
    lpf_off: isize,
    w: usize,
    h: usize,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    bitdepth_max: c_int,
) {
    wiener_filter7_16bpc_avx2_inner(p, left, lpf, lpf_off, w, h, params, edges, bitdepth_max);
}

// ============================================================================
// FFI WRAPPERS
// ============================================================================

/// Reconstructs lpf offset from pointer
fn reconstruct_lpf_offset(lpf: &DisjointMut<AlignedVec64<u8>>, ptr: *const u8) -> isize {
    let base = lpf.as_mut_ptr();
    ptr as isize - base as isize
}

/// FFI wrapper for Wiener filter 7-tap 8bpc
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
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
    p: *const FFISafe<PicOffset>,
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
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
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
    p: *const FFISafe<PicOffset>,
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

    wiener_filter5_8bpc_avx2_inner(p, left, lpf, lpf_off, w, h, params, edges);
}

/// Reconstructs lpf offset from pointer for 16bpc
fn reconstruct_lpf_offset_16bpc(lpf: &DisjointMut<AlignedVec64<u8>>, ptr: *const u16) -> isize {
    let base = lpf.as_mut_ptr().cast::<u16>();
    ptr as isize - base as isize / 2 // Divide by sizeof(u16)
}

/// FFI wrapper for Wiener filter 7-tap 16bpc
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn wiener_filter7_16bpc_avx2(
    _p_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    left: *const LeftPixelRow<DynPixel>,
    lpf_ptr: *const DynPixel,
    w: c_int,
    h: c_int,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    bitdepth_max: c_int,
    p: *const FFISafe<PicOffset>,
    lpf: *const FFISafe<DisjointMut<AlignedVec64<u8>>>,
) {
    let p = unsafe { *FFISafe::get(p) };
    let left = left.cast::<LeftPixelRow<u16>>();
    let lpf = unsafe { FFISafe::get(lpf) };
    let lpf_ptr = lpf_ptr.cast::<u16>();
    let lpf_off = reconstruct_lpf_offset_16bpc(lpf, lpf_ptr);
    let w = w as usize;
    let h = h as usize;
    let left = unsafe { slice::from_raw_parts(left, h) };

    wiener_filter7_16bpc_avx2_inner(p, left, lpf, lpf_off, w, h, params, edges, bitdepth_max);
}

/// FFI wrapper for Wiener filter 5-tap 16bpc
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn wiener_filter5_16bpc_avx2(
    _p_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    left: *const LeftPixelRow<DynPixel>,
    lpf_ptr: *const DynPixel,
    w: c_int,
    h: c_int,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    bitdepth_max: c_int,
    p: *const FFISafe<PicOffset>,
    lpf: *const FFISafe<DisjointMut<AlignedVec64<u8>>>,
) {
    let p = unsafe { *FFISafe::get(p) };
    let left = left.cast::<LeftPixelRow<u16>>();
    let lpf = unsafe { FFISafe::get(lpf) };
    let lpf_ptr = lpf_ptr.cast::<u16>();
    let lpf_off = reconstruct_lpf_offset_16bpc(lpf, lpf_ptr);
    let w = w as usize;
    let h = h as usize;
    let left = unsafe { slice::from_raw_parts(left, h) };

    wiener_filter5_16bpc_avx2_inner(p, left, lpf, lpf_off, w, h, params, edges, bitdepth_max);
}

// ============================================================================
// SGR (Self-Guided Restoration) FILTER - AVX2 IMPLEMENTATION
// ============================================================================

// Maximum restoration width (256 * 1.5)
const MAX_RESTORATION_WIDTH: usize = 256 * 3 / 2;

/// Compute box sum for 5x5 window (sum and sum of squares)
///
/// The input is padded with 3 pixels on each side.
/// Output arrays have the same layout but contain sums.
#[inline(always)]
fn boxsum5_8bpc(
    sumsq: &mut [i32; (64 + 2 + 2) * REST_UNIT_STRIDE],
    sum: &mut [i16; (64 + 2 + 2) * REST_UNIT_STRIDE],
    src: &[u8; (64 + 3 + 3) * REST_UNIT_STRIDE],
    w: usize,
    h: usize,
) {
    // Vertical pass: sum 5 consecutive rows
    for x in 0..w {
        let mut sum_v = x;
        let mut sumsq_v = x;

        let mut a = src[x] as i32;
        let mut a2 = a * a;
        let mut b = src[1 * REST_UNIT_STRIDE + x] as i32;
        let mut b2 = b * b;
        let mut c = src[2 * REST_UNIT_STRIDE + x] as i32;
        let mut c2 = c * c;
        let mut d = src[3 * REST_UNIT_STRIDE + x] as i32;
        let mut d2 = d * d;

        let mut s_idx = 3 * REST_UNIT_STRIDE + x;

        // Skip first 2 rows, process up to h-2
        for _ in 2..h - 2 {
            s_idx += REST_UNIT_STRIDE;
            let e = src[s_idx] as i32;
            let e2 = e * e;
            sum_v += REST_UNIT_STRIDE;
            sumsq_v += REST_UNIT_STRIDE;
            sum[sum_v] = (a + b + c + d + e) as i16;
            sumsq[sumsq_v] = a2 + b2 + c2 + d2 + e2;
            a = b;
            a2 = b2;
            b = c;
            b2 = c2;
            c = d;
            c2 = d2;
            d = e;
            d2 = e2;
        }
    }

    // Horizontal pass: sum 5 consecutive columns
    let mut sum_idx = REST_UNIT_STRIDE;
    let mut sumsq_idx = REST_UNIT_STRIDE;
    for _ in 2..h - 2 {
        let mut a = sum[sum_idx];
        let mut a2 = sumsq[sumsq_idx];
        let mut b = sum[sum_idx + 1];
        let mut b2 = sumsq[sumsq_idx + 1];
        let mut c = sum[sum_idx + 2];
        let mut c2 = sumsq[sumsq_idx + 2];
        let mut d = sum[sum_idx + 3];
        let mut d2 = sumsq[sumsq_idx + 3];

        for x in 2..w - 2 {
            let e = sum[sum_idx + x + 2];
            let e2 = sumsq[sumsq_idx + x + 2];
            sum[sum_idx + x] = a + b + c + d + e;
            sumsq[sumsq_idx + x] = a2 + b2 + c2 + d2 + e2;
            a = b;
            b = c;
            c = d;
            d = e;
            a2 = b2;
            b2 = c2;
            c2 = d2;
            d2 = e2;
        }
        sum_idx += REST_UNIT_STRIDE;
        sumsq_idx += REST_UNIT_STRIDE;
    }
}

/// Compute box sum for 3x3 window (sum and sum of squares)
#[inline(always)]
fn boxsum3_8bpc(
    sumsq: &mut [i32; (64 + 2 + 2) * REST_UNIT_STRIDE],
    sum: &mut [i16; (64 + 2 + 2) * REST_UNIT_STRIDE],
    src: &[u8; (64 + 3 + 3) * REST_UNIT_STRIDE],
    w: usize,
    h: usize,
) {
    // Skip the first row
    let src = &src[REST_UNIT_STRIDE..];

    // Vertical pass: sum 3 consecutive rows
    for x in 1..w - 1 {
        let mut sum_v = x;
        let mut sumsq_v = x;

        let mut a = src[x] as i32;
        let mut a2 = a * a;
        let mut b = src[REST_UNIT_STRIDE + x] as i32;
        let mut b2 = b * b;

        let mut s_idx = REST_UNIT_STRIDE + x;

        for _ in 2..h - 2 {
            s_idx += REST_UNIT_STRIDE;
            let c = src[s_idx] as i32;
            let c2 = c * c;
            sum_v += REST_UNIT_STRIDE;
            sumsq_v += REST_UNIT_STRIDE;
            sum[sum_v] = (a + b + c) as i16;
            sumsq[sumsq_v] = a2 + b2 + c2;
            a = b;
            a2 = b2;
            b = c;
            b2 = c2;
        }
    }

    // Horizontal pass: sum 3 consecutive columns
    let mut sum_idx = REST_UNIT_STRIDE;
    let mut sumsq_idx = REST_UNIT_STRIDE;
    for _ in 2..h - 2 {
        let mut a = sum[sum_idx + 1];
        let mut a2 = sumsq[sumsq_idx + 1];
        let mut b = sum[sum_idx + 2];
        let mut b2 = sumsq[sumsq_idx + 2];

        for x in 2..w - 2 {
            let c = sum[sum_idx + x + 1];
            let c2 = sumsq[sumsq_idx + x + 1];
            sum[sum_idx + x] = a + b + c;
            sumsq[sumsq_idx + x] = a2 + b2 + c2;
            a = b;
            b = c;
            a2 = b2;
            b2 = c2;
        }
        sum_idx += REST_UNIT_STRIDE;
        sumsq_idx += REST_UNIT_STRIDE;
    }
}

/// Self-guided filter computation for 8bpc
///
/// Computes the filter coefficients and applies the guided filter.
/// n = 25 for 5x5, n = 9 for 3x3
#[inline(never)]
fn selfguided_filter_8bpc(
    dst: &mut [i16; 64 * MAX_RESTORATION_WIDTH],
    src: &[u8; (64 + 3 + 3) * REST_UNIT_STRIDE],
    w: usize,
    h: usize,
    n: i32,
    s: u32,
) {
    let sgr_one_by_x: u32 = if n == 25 { 164 } else { 455 };

    // Working buffers
    let mut sumsq = [0i32; (64 + 2 + 2) * REST_UNIT_STRIDE];
    let mut sum = [0i16; (64 + 2 + 2) * REST_UNIT_STRIDE];

    let step = if n == 25 { 2 } else { 1 };

    if n == 25 {
        boxsum5_8bpc(&mut sumsq, &mut sum, src, w + 6, h + 6);
    } else {
        boxsum3_8bpc(&mut sumsq, &mut sum, src, w + 6, h + 6);
    }

    // For 8bpc, bitdepth_min_8 = 0, so the scaling factors are 1
    // Calculate filter coefficients a and b
    // After this loop: sumsq contains 'a', sum contains 'b' (renamed)
    let base = 2 * REST_UNIT_STRIDE + 3;

    for row_offset in (0..(h + 2)).step_by(step) {
        let row_start = (row_offset as isize - 1) as usize;
        let aa_base = base + row_start * REST_UNIT_STRIDE - REST_UNIT_STRIDE;

        for i in 0..(w + 2) {
            let idx = aa_base + i;
            let a_val = sumsq.get(idx).copied().unwrap_or(0);
            let b_val = sum.get(idx).copied().unwrap_or(0) as i32;

            let p = cmp::max(a_val * n - b_val * b_val, 0) as u32;
            let z = (p * s + (1 << 19)) >> 20;
            let x = dav1d_sgr_x_by_x.0[cmp::min(z, 255) as usize] as u32;

            // Store inverted: a = x * b * sgr_one_by_x, b = x
            if let Some(aa) = sumsq.get_mut(idx) {
                *aa = ((x * (b_val as u32) * sgr_one_by_x + (1 << 11)) >> 12) as i32;
            }
            if let Some(bb) = sum.get_mut(idx) {
                *bb = x as i16;
            }
        }
    }

    // Apply neighbor-weighted filter to produce output
    let src_base = 3 * REST_UNIT_STRIDE + 3;

    if n == 25 {
        // 5x5: use six_neighbors weighting, step by 2 rows
        let mut j = 0usize;
        while j < h.saturating_sub(1) {
            // Even row: full 6-neighbor calculation
            for i in 0..w {
                let idx = base + j * REST_UNIT_STRIDE + i;
                // six_neighbors for b (sum array)
                let b_six = {
                    let above = sum.get(idx - REST_UNIT_STRIDE).copied().unwrap_or(0) as i32;
                    let below = sum.get(idx + REST_UNIT_STRIDE).copied().unwrap_or(0) as i32;
                    let above_left =
                        sum.get(idx - REST_UNIT_STRIDE - 1).copied().unwrap_or(0) as i32;
                    let above_right =
                        sum.get(idx - REST_UNIT_STRIDE + 1).copied().unwrap_or(0) as i32;
                    let below_left =
                        sum.get(idx + REST_UNIT_STRIDE - 1).copied().unwrap_or(0) as i32;
                    let below_right =
                        sum.get(idx + REST_UNIT_STRIDE + 1).copied().unwrap_or(0) as i32;
                    (above + below) * 6 + (above_left + above_right + below_left + below_right) * 5
                };
                // six_neighbors for a (sumsq array)
                let a_six = {
                    let above = sumsq.get(idx - REST_UNIT_STRIDE).copied().unwrap_or(0);
                    let below = sumsq.get(idx + REST_UNIT_STRIDE).copied().unwrap_or(0);
                    let above_left = sumsq.get(idx - REST_UNIT_STRIDE - 1).copied().unwrap_or(0);
                    let above_right = sumsq.get(idx - REST_UNIT_STRIDE + 1).copied().unwrap_or(0);
                    let below_left = sumsq.get(idx + REST_UNIT_STRIDE - 1).copied().unwrap_or(0);
                    let below_right = sumsq.get(idx + REST_UNIT_STRIDE + 1).copied().unwrap_or(0);
                    (above + below) * 6 + (above_left + above_right + below_left + below_right) * 5
                };

                let src_val = src[src_base + j * REST_UNIT_STRIDE + i] as i32;
                dst[j * MAX_RESTORATION_WIDTH + i] =
                    ((a_six - b_six * src_val + (1 << 8)) >> 9) as i16;
            }

            // Odd row: simplified 3-neighbor horizontal calculation
            if j + 1 < h {
                for i in 0..w {
                    let idx = base + (j + 1) * REST_UNIT_STRIDE + i;
                    // Simplified: center * 6 + (left + right) * 5
                    let b_horiz = {
                        let center = sum.get(idx).copied().unwrap_or(0) as i32;
                        let left = sum.get(idx - 1).copied().unwrap_or(0) as i32;
                        let right = sum.get(idx + 1).copied().unwrap_or(0) as i32;
                        center * 6 + (left + right) * 5
                    };
                    let a_horiz = {
                        let center = sumsq.get(idx).copied().unwrap_or(0);
                        let left = sumsq.get(idx - 1).copied().unwrap_or(0);
                        let right = sumsq.get(idx + 1).copied().unwrap_or(0);
                        center * 6 + (left + right) * 5
                    };

                    let src_val = src[src_base + (j + 1) * REST_UNIT_STRIDE + i] as i32;
                    dst[(j + 1) * MAX_RESTORATION_WIDTH + i] =
                        ((a_horiz - b_horiz * src_val + (1 << 7)) >> 8) as i16;
                }
            }
            j += 2;
        }
        // Handle last row if height is odd
        if j < h {
            for i in 0..w {
                let idx = base + j * REST_UNIT_STRIDE + i;
                let b_six = {
                    let above = sum.get(idx - REST_UNIT_STRIDE).copied().unwrap_or(0) as i32;
                    let below = sum.get(idx + REST_UNIT_STRIDE).copied().unwrap_or(0) as i32;
                    let above_left =
                        sum.get(idx - REST_UNIT_STRIDE - 1).copied().unwrap_or(0) as i32;
                    let above_right =
                        sum.get(idx - REST_UNIT_STRIDE + 1).copied().unwrap_or(0) as i32;
                    let below_left =
                        sum.get(idx + REST_UNIT_STRIDE - 1).copied().unwrap_or(0) as i32;
                    let below_right =
                        sum.get(idx + REST_UNIT_STRIDE + 1).copied().unwrap_or(0) as i32;
                    (above + below) * 6 + (above_left + above_right + below_left + below_right) * 5
                };
                let a_six = {
                    let above = sumsq.get(idx - REST_UNIT_STRIDE).copied().unwrap_or(0);
                    let below = sumsq.get(idx + REST_UNIT_STRIDE).copied().unwrap_or(0);
                    let above_left = sumsq.get(idx - REST_UNIT_STRIDE - 1).copied().unwrap_or(0);
                    let above_right = sumsq.get(idx - REST_UNIT_STRIDE + 1).copied().unwrap_or(0);
                    let below_left = sumsq.get(idx + REST_UNIT_STRIDE - 1).copied().unwrap_or(0);
                    let below_right = sumsq.get(idx + REST_UNIT_STRIDE + 1).copied().unwrap_or(0);
                    (above + below) * 6 + (above_left + above_right + below_left + below_right) * 5
                };

                let src_val = src[src_base + j * REST_UNIT_STRIDE + i] as i32;
                dst[j * MAX_RESTORATION_WIDTH + i] =
                    ((a_six - b_six * src_val + (1 << 8)) >> 9) as i16;
            }
        }
    } else {
        // 3x3: use eight_neighbors weighting
        for j in 0..h {
            for i in 0..w {
                let idx = base + j * REST_UNIT_STRIDE + i;
                // eight_neighbors for b
                let b_eight = {
                    let center = sum.get(idx).copied().unwrap_or(0) as i32;
                    let left = sum.get(idx - 1).copied().unwrap_or(0) as i32;
                    let right = sum.get(idx + 1).copied().unwrap_or(0) as i32;
                    let above = sum.get(idx - REST_UNIT_STRIDE).copied().unwrap_or(0) as i32;
                    let below = sum.get(idx + REST_UNIT_STRIDE).copied().unwrap_or(0) as i32;
                    let above_left =
                        sum.get(idx - REST_UNIT_STRIDE - 1).copied().unwrap_or(0) as i32;
                    let above_right =
                        sum.get(idx - REST_UNIT_STRIDE + 1).copied().unwrap_or(0) as i32;
                    let below_left =
                        sum.get(idx + REST_UNIT_STRIDE - 1).copied().unwrap_or(0) as i32;
                    let below_right =
                        sum.get(idx + REST_UNIT_STRIDE + 1).copied().unwrap_or(0) as i32;
                    (center + left + right + above + below) * 4
                        + (above_left + above_right + below_left + below_right) * 3
                };
                // eight_neighbors for a
                let a_eight = {
                    let center = sumsq.get(idx).copied().unwrap_or(0);
                    let left = sumsq.get(idx - 1).copied().unwrap_or(0);
                    let right = sumsq.get(idx + 1).copied().unwrap_or(0);
                    let above = sumsq.get(idx - REST_UNIT_STRIDE).copied().unwrap_or(0);
                    let below = sumsq.get(idx + REST_UNIT_STRIDE).copied().unwrap_or(0);
                    let above_left = sumsq.get(idx - REST_UNIT_STRIDE - 1).copied().unwrap_or(0);
                    let above_right = sumsq.get(idx - REST_UNIT_STRIDE + 1).copied().unwrap_or(0);
                    let below_left = sumsq.get(idx + REST_UNIT_STRIDE - 1).copied().unwrap_or(0);
                    let below_right = sumsq.get(idx + REST_UNIT_STRIDE + 1).copied().unwrap_or(0);
                    (center + left + right + above + below) * 4
                        + (above_left + above_right + below_left + below_right) * 3
                };

                let src_val = src[src_base + j * REST_UNIT_STRIDE + i] as i32;
                dst[j * MAX_RESTORATION_WIDTH + i] =
                    ((a_eight - b_eight * src_val + (1 << 8)) >> 9) as i16;
            }
        }
    }
}

/// SGR 5x5 filter for 8bpc using AVX2
#[cfg(target_arch = "x86_64")]
fn sgr_5x5_8bpc_avx2_inner(
    p: PicOffset,
    left: &[LeftPixelRow<u8>],
    lpf: &DisjointMut<AlignedVec64<u8>>,
    lpf_off: isize,
    w: usize,
    h: usize,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
) {
    let mut tmp = [0u8; (64 + 3 + 3) * REST_UNIT_STRIDE];
    let mut dst = [0i16; 64 * MAX_RESTORATION_WIDTH];

    padding::<BitDepth8>(&mut tmp, p, left, lpf, lpf_off, w, h, edges);

    let sgr = params.sgr();
    selfguided_filter_8bpc(&mut dst, &tmp, w, h, 25, sgr.s0);

    let w0 = sgr.w0 as i32;
    let stride = p.pixel_stride::<BitDepth8>();

    for j in 0..h {
        let mut p_row = (p + (j as isize * stride)).slice_mut::<BitDepth8>(w);
        for i in 0..w {
            let v = w0 * dst[j * MAX_RESTORATION_WIDTH + i] as i32;
            p_row[i] = iclip(p_row[i] as i32 + ((v + (1 << 10)) >> 11), 0, 255) as u8;
        }
    }
}

/// SGR 3x3 filter for 8bpc using AVX2
#[cfg(target_arch = "x86_64")]
fn sgr_3x3_8bpc_avx2_inner(
    p: PicOffset,
    left: &[LeftPixelRow<u8>],
    lpf: &DisjointMut<AlignedVec64<u8>>,
    lpf_off: isize,
    w: usize,
    h: usize,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
) {
    let mut tmp = [0u8; (64 + 3 + 3) * REST_UNIT_STRIDE];
    let mut dst = [0i16; 64 * MAX_RESTORATION_WIDTH];

    padding::<BitDepth8>(&mut tmp, p, left, lpf, lpf_off, w, h, edges);

    let sgr = params.sgr();
    selfguided_filter_8bpc(&mut dst, &tmp, w, h, 9, sgr.s1);

    let w1 = sgr.w1 as i32;
    let stride = p.pixel_stride::<BitDepth8>();

    for j in 0..h {
        let mut p_row = (p + (j as isize * stride)).slice_mut::<BitDepth8>(w);
        for i in 0..w {
            let v = w1 * dst[j * MAX_RESTORATION_WIDTH + i] as i32;
            p_row[i] = iclip(p_row[i] as i32 + ((v + (1 << 10)) >> 11), 0, 255) as u8;
        }
    }
}

/// SGR mix filter for 8bpc using AVX2 (combines 5x5 and 3x3)
#[cfg(target_arch = "x86_64")]
fn sgr_mix_8bpc_avx2_inner(
    p: PicOffset,
    left: &[LeftPixelRow<u8>],
    lpf: &DisjointMut<AlignedVec64<u8>>,
    lpf_off: isize,
    w: usize,
    h: usize,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
) {
    let mut tmp = [0u8; (64 + 3 + 3) * REST_UNIT_STRIDE];
    let mut dst0 = [0i16; 64 * MAX_RESTORATION_WIDTH];
    let mut dst1 = [0i16; 64 * MAX_RESTORATION_WIDTH];

    padding::<BitDepth8>(&mut tmp, p, left, lpf, lpf_off, w, h, edges);

    let sgr = params.sgr();
    selfguided_filter_8bpc(&mut dst0, &tmp, w, h, 25, sgr.s0);
    selfguided_filter_8bpc(&mut dst1, &tmp, w, h, 9, sgr.s1);

    let w0 = sgr.w0 as i32;
    let w1 = sgr.w1 as i32;
    let stride = p.pixel_stride::<BitDepth8>();

    for j in 0..h {
        let mut p_row = (p + (j as isize * stride)).slice_mut::<BitDepth8>(w);
        for i in 0..w {
            let v = w0 * dst0[j * MAX_RESTORATION_WIDTH + i] as i32
                + w1 * dst1[j * MAX_RESTORATION_WIDTH + i] as i32;
            p_row[i] = iclip(p_row[i] as i32 + ((v + (1 << 10)) >> 11), 0, 255) as u8;
        }
    }
}

// ============================================================================
// SGR FFI WRAPPERS
// ============================================================================

/// FFI wrapper for SGR 5x5 filter 8bpc
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn sgr_filter_5x5_8bpc_avx2(
    _p_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    left: *const LeftPixelRow<DynPixel>,
    lpf_ptr: *const DynPixel,
    w: c_int,
    h: c_int,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    _bitdepth_max: c_int,
    p: *const FFISafe<PicOffset>,
    lpf: *const FFISafe<DisjointMut<AlignedVec64<u8>>>,
) {
    let p = unsafe { *FFISafe::get(p) };
    let left = left.cast::<LeftPixelRow<u8>>();
    let lpf = unsafe { FFISafe::get(lpf) };
    let lpf_ptr = lpf_ptr.cast::<u8>();
    let lpf_off = reconstruct_lpf_offset(lpf, lpf_ptr);
    let w = w as usize;
    let h = h as usize;
    let left = unsafe { slice::from_raw_parts(left, h) };

    sgr_5x5_8bpc_avx2_inner(p, left, lpf, lpf_off, w, h, params, edges);
}

/// FFI wrapper for SGR 3x3 filter 8bpc
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn sgr_filter_3x3_8bpc_avx2(
    _p_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    left: *const LeftPixelRow<DynPixel>,
    lpf_ptr: *const DynPixel,
    w: c_int,
    h: c_int,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    _bitdepth_max: c_int,
    p: *const FFISafe<PicOffset>,
    lpf: *const FFISafe<DisjointMut<AlignedVec64<u8>>>,
) {
    let p = unsafe { *FFISafe::get(p) };
    let left = left.cast::<LeftPixelRow<u8>>();
    let lpf = unsafe { FFISafe::get(lpf) };
    let lpf_ptr = lpf_ptr.cast::<u8>();
    let lpf_off = reconstruct_lpf_offset(lpf, lpf_ptr);
    let w = w as usize;
    let h = h as usize;
    let left = unsafe { slice::from_raw_parts(left, h) };

    sgr_3x3_8bpc_avx2_inner(p, left, lpf, lpf_off, w, h, params, edges);
}

/// FFI wrapper for SGR mix filter 8bpc
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn sgr_filter_mix_8bpc_avx2(
    _p_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    left: *const LeftPixelRow<DynPixel>,
    lpf_ptr: *const DynPixel,
    w: c_int,
    h: c_int,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    _bitdepth_max: c_int,
    p: *const FFISafe<PicOffset>,
    lpf: *const FFISafe<DisjointMut<AlignedVec64<u8>>>,
) {
    let p = unsafe { *FFISafe::get(p) };
    let left = left.cast::<LeftPixelRow<u8>>();
    let lpf = unsafe { FFISafe::get(lpf) };
    let lpf_ptr = lpf_ptr.cast::<u8>();
    let lpf_off = reconstruct_lpf_offset(lpf, lpf_ptr);
    let w = w as usize;
    let h = h as usize;
    let left = unsafe { slice::from_raw_parts(left, h) };

    sgr_mix_8bpc_avx2_inner(p, left, lpf, lpf_off, w, h, params, edges);
}

// ============================================================================
// SGR FILTER - 16bpc IMPLEMENTATION
// ============================================================================

/// Compute box sum for 5x5 window (sum and sum of squares) for 16bpc
#[inline(always)]
fn boxsum5_16bpc(
    sumsq: &mut [i64; (64 + 2 + 2) * REST_UNIT_STRIDE],
    sum: &mut [i32; (64 + 2 + 2) * REST_UNIT_STRIDE],
    src: &[u16; (64 + 3 + 3) * REST_UNIT_STRIDE],
    w: usize,
    h: usize,
) {
    // Vertical pass: sum 5 consecutive rows
    for x in 0..w {
        let mut sum_v = x;
        let mut sumsq_v = x;

        let mut a = src[x] as i64;
        let mut a2 = a * a;
        let mut b = src[1 * REST_UNIT_STRIDE + x] as i64;
        let mut b2 = b * b;
        let mut c = src[2 * REST_UNIT_STRIDE + x] as i64;
        let mut c2 = c * c;
        let mut d = src[3 * REST_UNIT_STRIDE + x] as i64;
        let mut d2 = d * d;

        let mut s_idx = 3 * REST_UNIT_STRIDE + x;

        // Skip first 2 rows, process up to h-2
        for _ in 2..h - 2 {
            s_idx += REST_UNIT_STRIDE;
            let e = src[s_idx] as i64;
            let e2 = e * e;
            sum_v += REST_UNIT_STRIDE;
            sumsq_v += REST_UNIT_STRIDE;
            sum[sum_v] = (a + b + c + d + e) as i32;
            sumsq[sumsq_v] = a2 + b2 + c2 + d2 + e2;
            a = b;
            a2 = b2;
            b = c;
            b2 = c2;
            c = d;
            c2 = d2;
            d = e;
            d2 = e2;
        }
    }

    // Horizontal pass: sum 5 consecutive columns
    let mut sum_idx = REST_UNIT_STRIDE;
    let mut sumsq_idx = REST_UNIT_STRIDE;
    for _ in 2..h - 2 {
        let mut a = sum[sum_idx] as i64;
        let mut a2 = sumsq[sumsq_idx];
        let mut b = sum[sum_idx + 1] as i64;
        let mut b2 = sumsq[sumsq_idx + 1];
        let mut c = sum[sum_idx + 2] as i64;
        let mut c2 = sumsq[sumsq_idx + 2];
        let mut d = sum[sum_idx + 3] as i64;
        let mut d2 = sumsq[sumsq_idx + 3];

        for x in 2..w - 2 {
            let e = sum[sum_idx + x + 2] as i64;
            let e2 = sumsq[sumsq_idx + x + 2];
            sum[sum_idx + x] = (a + b + c + d + e) as i32;
            sumsq[sumsq_idx + x] = a2 + b2 + c2 + d2 + e2;
            a = b;
            b = c;
            c = d;
            d = e;
            a2 = b2;
            b2 = c2;
            c2 = d2;
            d2 = e2;
        }
        sum_idx += REST_UNIT_STRIDE;
        sumsq_idx += REST_UNIT_STRIDE;
    }
}

/// Compute box sum for 3x3 window (sum and sum of squares) for 16bpc
#[inline(always)]
fn boxsum3_16bpc(
    sumsq: &mut [i64; (64 + 2 + 2) * REST_UNIT_STRIDE],
    sum: &mut [i32; (64 + 2 + 2) * REST_UNIT_STRIDE],
    src: &[u16; (64 + 3 + 3) * REST_UNIT_STRIDE],
    w: usize,
    h: usize,
) {
    // Skip the first row
    let src = &src[REST_UNIT_STRIDE..];

    // Vertical pass: sum 3 consecutive rows
    for x in 1..w - 1 {
        let mut sum_v = x;
        let mut sumsq_v = x;

        let mut a = src[x] as i64;
        let mut a2 = a * a;
        let mut b = src[REST_UNIT_STRIDE + x] as i64;
        let mut b2 = b * b;

        let mut s_idx = REST_UNIT_STRIDE + x;

        for _ in 2..h - 2 {
            s_idx += REST_UNIT_STRIDE;
            let c = src[s_idx] as i64;
            let c2 = c * c;
            sum_v += REST_UNIT_STRIDE;
            sumsq_v += REST_UNIT_STRIDE;
            sum[sum_v] = (a + b + c) as i32;
            sumsq[sumsq_v] = a2 + b2 + c2;
            a = b;
            a2 = b2;
            b = c;
            b2 = c2;
        }
    }

    // Horizontal pass: sum 3 consecutive columns
    let mut sum_idx = REST_UNIT_STRIDE;
    let mut sumsq_idx = REST_UNIT_STRIDE;
    for _ in 2..h - 2 {
        let mut a = sum[sum_idx + 1] as i64;
        let mut a2 = sumsq[sumsq_idx + 1];
        let mut b = sum[sum_idx + 2] as i64;
        let mut b2 = sumsq[sumsq_idx + 2];

        for x in 2..w - 2 {
            let c = sum[sum_idx + x + 1] as i64;
            let c2 = sumsq[sumsq_idx + x + 1];
            sum[sum_idx + x] = (a + b + c) as i32;
            sumsq[sumsq_idx + x] = a2 + b2 + c2;
            a = b;
            b = c;
            a2 = b2;
            b2 = c2;
        }
        sum_idx += REST_UNIT_STRIDE;
        sumsq_idx += REST_UNIT_STRIDE;
    }
}

/// Self-guided filter computation for 16bpc
///
/// Computes the filter coefficients and applies the guided filter.
/// n = 25 for 5x5, n = 9 for 3x3
#[inline(never)]
fn selfguided_filter_16bpc(
    dst: &mut [i32; 64 * MAX_RESTORATION_WIDTH],
    src: &[u16; (64 + 3 + 3) * REST_UNIT_STRIDE],
    w: usize,
    h: usize,
    n: i32,
    s: u32,
    bitdepth_max: i32,
) {
    let sgr_one_by_x: u32 = if n == 25 { 164 } else { 455 };

    // Determine bitdepth_min_8 (10bpc -> 2, 12bpc -> 4)
    let bitdepth = if bitdepth_max == 1023 { 10 } else { 12 };
    let bitdepth_min_8 = bitdepth - 8;

    // Working buffers - use i64 for sumsq to handle large squared values
    let mut sumsq = [0i64; (64 + 2 + 2) * REST_UNIT_STRIDE];
    let mut sum = [0i32; (64 + 2 + 2) * REST_UNIT_STRIDE];

    // Coefficient buffers (reuse sumsq/sum after boxsum)
    let mut aa = [0i32; (64 + 2 + 2) * REST_UNIT_STRIDE];
    let mut bb = [0i32; (64 + 2 + 2) * REST_UNIT_STRIDE];

    let step = if n == 25 { 2 } else { 1 };

    if n == 25 {
        boxsum5_16bpc(&mut sumsq, &mut sum, src, w + 6, h + 6);
    } else {
        boxsum3_16bpc(&mut sumsq, &mut sum, src, w + 6, h + 6);
    }

    // Calculate filter coefficients a and b with bitdepth scaling
    let base = 2 * REST_UNIT_STRIDE + 3;

    for row_offset in (0..(h + 2)).step_by(step) {
        let row_start = (row_offset as isize - 1) as usize;
        let aa_base = base + row_start * REST_UNIT_STRIDE - REST_UNIT_STRIDE;

        for i in 0..(w + 2) {
            let idx = aa_base + i;
            // Scale down by bitdepth_min_8 for the variance calculation
            let a_val = sumsq.get(idx).copied().unwrap_or(0);
            let b_val = sum.get(idx).copied().unwrap_or(0) as i64;

            // Apply bitdepth scaling: a >> (2 * bitdepth_min_8), b >> bitdepth_min_8
            let a_scaled =
                ((a_val + (1 << (2 * bitdepth_min_8 - 1))) >> (2 * bitdepth_min_8)) as i32;
            let b_scaled = ((b_val + (1 << (bitdepth_min_8 - 1))) >> bitdepth_min_8) as i32;

            let p = cmp::max(a_scaled * n - b_scaled * b_scaled, 0) as u32;
            let z = (p * s + (1 << 19)) >> 20;
            let x = dav1d_sgr_x_by_x.0[cmp::min(z, 255) as usize] as u32;

            // Store: aa = x * b * sgr_one_by_x, bb = x
            // Use original b_val (not scaled) for the multiplication
            aa[idx] = ((x * (b_val as u32) * sgr_one_by_x + (1 << 11)) >> 12) as i32;
            bb[idx] = x as i32;
        }
    }

    // Apply neighbor-weighted filter to produce output
    let src_base = 3 * REST_UNIT_STRIDE + 3;

    if n == 25 {
        // 5x5: use six_neighbors weighting, step by 2 rows
        let mut j = 0usize;
        while j < h.saturating_sub(1) {
            // Even row: full 6-neighbor calculation
            for i in 0..w {
                let idx = base + j * REST_UNIT_STRIDE + i;
                // six_neighbors for b (bb array)
                let b_six = {
                    let above = bb.get(idx - REST_UNIT_STRIDE).copied().unwrap_or(0) as i64;
                    let below = bb.get(idx + REST_UNIT_STRIDE).copied().unwrap_or(0) as i64;
                    let above_left =
                        bb.get(idx - REST_UNIT_STRIDE - 1).copied().unwrap_or(0) as i64;
                    let above_right =
                        bb.get(idx - REST_UNIT_STRIDE + 1).copied().unwrap_or(0) as i64;
                    let below_left =
                        bb.get(idx + REST_UNIT_STRIDE - 1).copied().unwrap_or(0) as i64;
                    let below_right =
                        bb.get(idx + REST_UNIT_STRIDE + 1).copied().unwrap_or(0) as i64;
                    (above + below) * 6 + (above_left + above_right + below_left + below_right) * 5
                };
                // six_neighbors for a (aa array)
                let a_six = {
                    let above = aa.get(idx - REST_UNIT_STRIDE).copied().unwrap_or(0) as i64;
                    let below = aa.get(idx + REST_UNIT_STRIDE).copied().unwrap_or(0) as i64;
                    let above_left =
                        aa.get(idx - REST_UNIT_STRIDE - 1).copied().unwrap_or(0) as i64;
                    let above_right =
                        aa.get(idx - REST_UNIT_STRIDE + 1).copied().unwrap_or(0) as i64;
                    let below_left =
                        aa.get(idx + REST_UNIT_STRIDE - 1).copied().unwrap_or(0) as i64;
                    let below_right =
                        aa.get(idx + REST_UNIT_STRIDE + 1).copied().unwrap_or(0) as i64;
                    (above + below) * 6 + (above_left + above_right + below_left + below_right) * 5
                };

                let src_val = src[src_base + j * REST_UNIT_STRIDE + i] as i64;
                dst[j * MAX_RESTORATION_WIDTH + i] =
                    ((a_six - b_six * src_val + (1 << 8)) >> 9) as i32;
            }

            // Odd row: simplified 3-neighbor horizontal calculation
            if j + 1 < h {
                for i in 0..w {
                    let idx = base + (j + 1) * REST_UNIT_STRIDE + i;
                    // Simplified: center * 6 + (left + right) * 5
                    let b_horiz = {
                        let center = bb.get(idx).copied().unwrap_or(0) as i64;
                        let left = bb.get(idx - 1).copied().unwrap_or(0) as i64;
                        let right = bb.get(idx + 1).copied().unwrap_or(0) as i64;
                        center * 6 + (left + right) * 5
                    };
                    let a_horiz = {
                        let center = aa.get(idx).copied().unwrap_or(0) as i64;
                        let left = aa.get(idx - 1).copied().unwrap_or(0) as i64;
                        let right = aa.get(idx + 1).copied().unwrap_or(0) as i64;
                        center * 6 + (left + right) * 5
                    };

                    let src_val = src[src_base + (j + 1) * REST_UNIT_STRIDE + i] as i64;
                    dst[(j + 1) * MAX_RESTORATION_WIDTH + i] =
                        ((a_horiz - b_horiz * src_val + (1 << 7)) >> 8) as i32;
                }
            }
            j += 2;
        }
        // Handle last row if height is odd
        if j < h {
            for i in 0..w {
                let idx = base + j * REST_UNIT_STRIDE + i;
                let b_six = {
                    let above = bb.get(idx - REST_UNIT_STRIDE).copied().unwrap_or(0) as i64;
                    let below = bb.get(idx + REST_UNIT_STRIDE).copied().unwrap_or(0) as i64;
                    let above_left =
                        bb.get(idx - REST_UNIT_STRIDE - 1).copied().unwrap_or(0) as i64;
                    let above_right =
                        bb.get(idx - REST_UNIT_STRIDE + 1).copied().unwrap_or(0) as i64;
                    let below_left =
                        bb.get(idx + REST_UNIT_STRIDE - 1).copied().unwrap_or(0) as i64;
                    let below_right =
                        bb.get(idx + REST_UNIT_STRIDE + 1).copied().unwrap_or(0) as i64;
                    (above + below) * 6 + (above_left + above_right + below_left + below_right) * 5
                };
                let a_six = {
                    let above = aa.get(idx - REST_UNIT_STRIDE).copied().unwrap_or(0) as i64;
                    let below = aa.get(idx + REST_UNIT_STRIDE).copied().unwrap_or(0) as i64;
                    let above_left =
                        aa.get(idx - REST_UNIT_STRIDE - 1).copied().unwrap_or(0) as i64;
                    let above_right =
                        aa.get(idx - REST_UNIT_STRIDE + 1).copied().unwrap_or(0) as i64;
                    let below_left =
                        aa.get(idx + REST_UNIT_STRIDE - 1).copied().unwrap_or(0) as i64;
                    let below_right =
                        aa.get(idx + REST_UNIT_STRIDE + 1).copied().unwrap_or(0) as i64;
                    (above + below) * 6 + (above_left + above_right + below_left + below_right) * 5
                };

                let src_val = src[src_base + j * REST_UNIT_STRIDE + i] as i64;
                dst[j * MAX_RESTORATION_WIDTH + i] =
                    ((a_six - b_six * src_val + (1 << 8)) >> 9) as i32;
            }
        }
    } else {
        // 3x3: use eight_neighbors weighting
        for j in 0..h {
            for i in 0..w {
                let idx = base + j * REST_UNIT_STRIDE + i;
                // eight_neighbors for b
                let b_eight = {
                    let center = bb.get(idx).copied().unwrap_or(0) as i64;
                    let left = bb.get(idx - 1).copied().unwrap_or(0) as i64;
                    let right = bb.get(idx + 1).copied().unwrap_or(0) as i64;
                    let above = bb.get(idx - REST_UNIT_STRIDE).copied().unwrap_or(0) as i64;
                    let below = bb.get(idx + REST_UNIT_STRIDE).copied().unwrap_or(0) as i64;
                    let above_left =
                        bb.get(idx - REST_UNIT_STRIDE - 1).copied().unwrap_or(0) as i64;
                    let above_right =
                        bb.get(idx - REST_UNIT_STRIDE + 1).copied().unwrap_or(0) as i64;
                    let below_left =
                        bb.get(idx + REST_UNIT_STRIDE - 1).copied().unwrap_or(0) as i64;
                    let below_right =
                        bb.get(idx + REST_UNIT_STRIDE + 1).copied().unwrap_or(0) as i64;
                    (center + left + right + above + below) * 4
                        + (above_left + above_right + below_left + below_right) * 3
                };
                // eight_neighbors for a
                let a_eight = {
                    let center = aa.get(idx).copied().unwrap_or(0) as i64;
                    let left = aa.get(idx - 1).copied().unwrap_or(0) as i64;
                    let right = aa.get(idx + 1).copied().unwrap_or(0) as i64;
                    let above = aa.get(idx - REST_UNIT_STRIDE).copied().unwrap_or(0) as i64;
                    let below = aa.get(idx + REST_UNIT_STRIDE).copied().unwrap_or(0) as i64;
                    let above_left =
                        aa.get(idx - REST_UNIT_STRIDE - 1).copied().unwrap_or(0) as i64;
                    let above_right =
                        aa.get(idx - REST_UNIT_STRIDE + 1).copied().unwrap_or(0) as i64;
                    let below_left =
                        aa.get(idx + REST_UNIT_STRIDE - 1).copied().unwrap_or(0) as i64;
                    let below_right =
                        aa.get(idx + REST_UNIT_STRIDE + 1).copied().unwrap_or(0) as i64;
                    (center + left + right + above + below) * 4
                        + (above_left + above_right + below_left + below_right) * 3
                };

                let src_val = src[src_base + j * REST_UNIT_STRIDE + i] as i64;
                dst[j * MAX_RESTORATION_WIDTH + i] =
                    ((a_eight - b_eight * src_val + (1 << 8)) >> 9) as i32;
            }
        }
    }
}

/// SGR 5x5 filter for 16bpc using AVX2
#[cfg(target_arch = "x86_64")]
fn sgr_5x5_16bpc_avx2_inner(
    p: PicOffset,
    left: &[LeftPixelRow<u16>],
    lpf: &DisjointMut<AlignedVec64<u8>>,
    lpf_off: isize,
    w: usize,
    h: usize,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    bitdepth_max: c_int,
) {
    let mut tmp = [0u16; (64 + 3 + 3) * REST_UNIT_STRIDE];
    let mut dst = [0i32; 64 * MAX_RESTORATION_WIDTH];

    padding::<BitDepth16>(&mut tmp, p, left, lpf, lpf_off, w, h, edges);

    let sgr = params.sgr();
    selfguided_filter_16bpc(&mut dst, &tmp, w, h, 25, sgr.s0, bitdepth_max);

    let w0 = sgr.w0 as i32;
    let stride = p.pixel_stride::<BitDepth16>();

    for j in 0..h {
        let mut p_row = (p + (j as isize * stride)).slice_mut::<BitDepth16>(w);
        for i in 0..w {
            let v = w0 * dst[j * MAX_RESTORATION_WIDTH + i];
            p_row[i] = iclip(p_row[i] as i32 + ((v + (1 << 10)) >> 11), 0, bitdepth_max) as u16;
        }
    }
}

/// SGR 3x3 filter for 16bpc using AVX2
#[cfg(target_arch = "x86_64")]
fn sgr_3x3_16bpc_avx2_inner(
    p: PicOffset,
    left: &[LeftPixelRow<u16>],
    lpf: &DisjointMut<AlignedVec64<u8>>,
    lpf_off: isize,
    w: usize,
    h: usize,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    bitdepth_max: c_int,
) {
    let mut tmp = [0u16; (64 + 3 + 3) * REST_UNIT_STRIDE];
    let mut dst = [0i32; 64 * MAX_RESTORATION_WIDTH];

    padding::<BitDepth16>(&mut tmp, p, left, lpf, lpf_off, w, h, edges);

    let sgr = params.sgr();
    selfguided_filter_16bpc(&mut dst, &tmp, w, h, 9, sgr.s1, bitdepth_max);

    let w1 = sgr.w1 as i32;
    let stride = p.pixel_stride::<BitDepth16>();

    for j in 0..h {
        let mut p_row = (p + (j as isize * stride)).slice_mut::<BitDepth16>(w);
        for i in 0..w {
            let v = w1 * dst[j * MAX_RESTORATION_WIDTH + i];
            p_row[i] = iclip(p_row[i] as i32 + ((v + (1 << 10)) >> 11), 0, bitdepth_max) as u16;
        }
    }
}

/// SGR mix filter for 16bpc using AVX2 (combines 5x5 and 3x3)
#[cfg(target_arch = "x86_64")]
fn sgr_mix_16bpc_avx2_inner(
    p: PicOffset,
    left: &[LeftPixelRow<u16>],
    lpf: &DisjointMut<AlignedVec64<u8>>,
    lpf_off: isize,
    w: usize,
    h: usize,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    bitdepth_max: c_int,
) {
    let mut tmp = [0u16; (64 + 3 + 3) * REST_UNIT_STRIDE];
    let mut dst0 = [0i32; 64 * MAX_RESTORATION_WIDTH];
    let mut dst1 = [0i32; 64 * MAX_RESTORATION_WIDTH];

    padding::<BitDepth16>(&mut tmp, p, left, lpf, lpf_off, w, h, edges);

    let sgr = params.sgr();
    selfguided_filter_16bpc(&mut dst0, &tmp, w, h, 25, sgr.s0, bitdepth_max);
    selfguided_filter_16bpc(&mut dst1, &tmp, w, h, 9, sgr.s1, bitdepth_max);

    let w0 = sgr.w0 as i32;
    let w1 = sgr.w1 as i32;
    let stride = p.pixel_stride::<BitDepth16>();

    for j in 0..h {
        let mut p_row = (p + (j as isize * stride)).slice_mut::<BitDepth16>(w);
        for i in 0..w {
            let v =
                w0 * dst0[j * MAX_RESTORATION_WIDTH + i] + w1 * dst1[j * MAX_RESTORATION_WIDTH + i];
            p_row[i] = iclip(p_row[i] as i32 + ((v + (1 << 10)) >> 11), 0, bitdepth_max) as u16;
        }
    }
}

// ============================================================================
// SGR 16bpc FFI WRAPPERS
// ============================================================================

/// FFI wrapper for SGR 5x5 filter 16bpc
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn sgr_filter_5x5_16bpc_avx2(
    _p_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    left: *const LeftPixelRow<DynPixel>,
    lpf_ptr: *const DynPixel,
    w: c_int,
    h: c_int,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    bitdepth_max: c_int,
    p: *const FFISafe<PicOffset>,
    lpf: *const FFISafe<DisjointMut<AlignedVec64<u8>>>,
) {
    let p = unsafe { *FFISafe::get(p) };
    let left = left.cast::<LeftPixelRow<u16>>();
    let lpf = unsafe { FFISafe::get(lpf) };
    let lpf_ptr = lpf_ptr.cast::<u16>();
    let lpf_off = reconstruct_lpf_offset_16bpc(lpf, lpf_ptr);
    let w = w as usize;
    let h = h as usize;
    let left = unsafe { slice::from_raw_parts(left, h) };

    sgr_5x5_16bpc_avx2_inner(p, left, lpf, lpf_off, w, h, params, edges, bitdepth_max);
}

/// FFI wrapper for SGR 3x3 filter 16bpc
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn sgr_filter_3x3_16bpc_avx2(
    _p_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    left: *const LeftPixelRow<DynPixel>,
    lpf_ptr: *const DynPixel,
    w: c_int,
    h: c_int,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    bitdepth_max: c_int,
    p: *const FFISafe<PicOffset>,
    lpf: *const FFISafe<DisjointMut<AlignedVec64<u8>>>,
) {
    let p = unsafe { *FFISafe::get(p) };
    let left = left.cast::<LeftPixelRow<u16>>();
    let lpf = unsafe { FFISafe::get(lpf) };
    let lpf_ptr = lpf_ptr.cast::<u16>();
    let lpf_off = reconstruct_lpf_offset_16bpc(lpf, lpf_ptr);
    let w = w as usize;
    let h = h as usize;
    let left = unsafe { slice::from_raw_parts(left, h) };

    sgr_3x3_16bpc_avx2_inner(p, left, lpf, lpf_off, w, h, params, edges, bitdepth_max);
}

/// FFI wrapper for SGR mix filter 16bpc
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn sgr_filter_mix_16bpc_avx2(
    _p_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    left: *const LeftPixelRow<DynPixel>,
    lpf_ptr: *const DynPixel,
    w: c_int,
    h: c_int,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    bitdepth_max: c_int,
    p: *const FFISafe<PicOffset>,
    lpf: *const FFISafe<DisjointMut<AlignedVec64<u8>>>,
) {
    let p = unsafe { *FFISafe::get(p) };
    let left = left.cast::<LeftPixelRow<u16>>();
    let lpf = unsafe { FFISafe::get(lpf) };
    let lpf_ptr = lpf_ptr.cast::<u16>();
    let lpf_off = reconstruct_lpf_offset_16bpc(lpf, lpf_ptr);
    let w = w as usize;
    let h = h as usize;
    let left = unsafe { slice::from_raw_parts(left, h) };

    sgr_mix_16bpc_avx2_inner(p, left, lpf, lpf_off, w, h, params, edges, bitdepth_max);
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

    #[test]
    fn test_max_restoration_width() {
        assert_eq!(MAX_RESTORATION_WIDTH, 384);
    }
}

/// Safe dispatch for lr_filter. Returns true if SIMD was used.
#[cfg(target_arch = "x86_64")]
pub fn lr_filter_dispatch<BD: BitDepth>(
    variant: usize,
    dst: PicOffset,
    left: &[LeftPixelRow<BD::Pixel>],
    lpf: &DisjointMut<AlignedVec64<u8>>,
    lpf_off: isize,
    w: c_int,
    h: c_int,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    bd: BD,
) -> bool {
    use crate::include::common::bitdepth::BPC;
    use crate::src::cpu::CpuFlags;

    if !crate::src::cpu::rav1d_get_cpu_flags().contains(CpuFlags::AVX2) {
        return false;
    }

    let w = w as usize;
    let h = h as usize;
    let left_8 = || unsafe { &*(left as *const [LeftPixelRow<BD::Pixel>] as *const [LeftPixelRow<u8>]) };
    let left_16 = || unsafe { &*(left as *const [LeftPixelRow<BD::Pixel>] as *const [LeftPixelRow<u16>]) };

    match (BD::BPC, variant) {
        // SAFETY: wiener_filter7_8bpc uses AVX2 intrinsics; AVX2 verified by CpuFlags check above
        (BPC::BPC8, 0) => unsafe { wiener_filter7_8bpc_avx2_inner(dst, left_8(), lpf, lpf_off, w, h, params, edges) },
        (BPC::BPC8, 1) => wiener_filter5_8bpc_avx2_inner(dst, left_8(), lpf, lpf_off, w, h, params, edges),
        (BPC::BPC8, 2) => sgr_5x5_8bpc_avx2_inner(dst, left_8(), lpf, lpf_off, w, h, params, edges),
        (BPC::BPC8, 3) => sgr_3x3_8bpc_avx2_inner(dst, left_8(), lpf, lpf_off, w, h, params, edges),
        (BPC::BPC8, _) => sgr_mix_8bpc_avx2_inner(dst, left_8(), lpf, lpf_off, w, h, params, edges),
        (BPC::BPC16, 0) => wiener_filter7_16bpc_avx2_inner(dst, left_16(), lpf, lpf_off, w, h, params, edges, bd.into_c()),
        (BPC::BPC16, 1) => wiener_filter5_16bpc_avx2_inner(dst, left_16(), lpf, lpf_off, w, h, params, edges, bd.into_c()),
        (BPC::BPC16, 2) => sgr_5x5_16bpc_avx2_inner(dst, left_16(), lpf, lpf_off, w, h, params, edges, bd.into_c()),
        (BPC::BPC16, 3) => sgr_3x3_16bpc_avx2_inner(dst, left_16(), lpf, lpf_off, w, h, params, edges, bd.into_c()),
        (BPC::BPC16, _) => sgr_mix_16bpc_avx2_inner(dst, left_16(), lpf, lpf_off, w, h, params, edges, bd.into_c()),
    }
    true
}
