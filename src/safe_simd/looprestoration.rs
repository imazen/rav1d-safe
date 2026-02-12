//! Safe SIMD implementations for Loop Restoration
#![allow(deprecated)] // FFI wrappers need to forge tokens
#![cfg_attr(not(feature = "unchecked"), forbid(unsafe_code))]
#![cfg_attr(feature = "unchecked", deny(unsafe_code))]
//!
//! Loop restoration applies two types of filtering:
//! 1. Wiener filter - 7-tap or 5-tap separable filter
//! 2. SGR (Self-Guided Restoration) - guided filter based on local statistics

#![allow(unused_imports)]
#![allow(dead_code)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
use crate::src::cpu::summon_avx2;
use archmage::{arcane, rite, Desktop64, Server64, SimdToken};
use std::cmp;
use std::ffi::c_int;
use std::ffi::c_uint;
use std::slice;

#[cfg(target_arch = "x86_64")]
use crate::src::safe_simd::partial_simd;
#[cfg(target_arch = "x86_64")]
use crate::src::safe_simd::pixel_access::{
    loadi64, loadu_128, loadu_256, loadu_512, storeu_128, storeu_256, storeu_512,
};

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
#[arcane]
fn wiener_filter7_8bpc_avx2_inner(
    _token: Desktop64,
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

    // Single guard for entire output region
    let (mut p_guard, p_base) = p.strided_slice_mut::<BitDepth8>(w, h);
    for j in 0..h {
        let row_off = p_base.wrapping_add_signed(j as isize * stride);
        let mut i = 0usize;

        // Process 8 pixels at a time with AVX2
        while i + 8 <= w {
            // Load 8 u16 values from each of the 7 rows and convert to i32
            // We need to process low and high halves separately
            let row0 = &hor[(j + 0) * REST_UNIT_STRIDE + i..];
            let row1 = &hor[(j + 1) * REST_UNIT_STRIDE + i..];
            let row2 = &hor[(j + 2) * REST_UNIT_STRIDE + i..];
            let row3 = &hor[(j + 3) * REST_UNIT_STRIDE + i..];
            let row4 = &hor[(j + 4) * REST_UNIT_STRIDE + i..];
            let row5 = &hor[(j + 5) * REST_UNIT_STRIDE + i..];
            let row6 = &hor[(j + 6) * REST_UNIT_STRIDE + i..];

            // Load 8 u16 values as __m128i via safe macro
            let r0 = loadu_128!(&row0[..8], [u16; 8]);
            let r1 = loadu_128!(&row1[..8], [u16; 8]);
            let r2 = loadu_128!(&row2[..8], [u16; 8]);
            let r3 = loadu_128!(&row3[..8], [u16; 8]);
            let r4 = loadu_128!(&row4[..8], [u16; 8]);
            let r5 = loadu_128!(&row5[..8], [u16; 8]);
            let r6 = loadu_128!(&row6[..8], [u16; 8]);

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

            // Store 8 bytes via safe partial_simd
            let dst_arr: &mut [u8; 8] = (&mut p_guard[row_off + i..row_off + i + 8])
                .try_into()
                .unwrap();
            partial_simd::mm_storel_epi64(dst_arr, sum8);
            i += 8;
        }

        // Handle remaining pixels with scalar
        while i < w {
            let mut sum = -round_offset;
            for k in 0..7 {
                sum += hor[(j + k) * REST_UNIT_STRIDE + i] as i32 * filter[1][k] as i32;
            }
            p_guard[row_off + i] = iclip((sum + rounding_off_v) >> round_bits_v, 0, 255) as u8;
            i += 1;
        }
    }
}

/// Wiener filter 5-tap for 8bpc using AVX2
///
/// Same as 7-tap but filter[0][0] = filter[0][6] = 0 and filter[1][0] = filter[1][6] = 0
#[cfg(target_arch = "x86_64")]
#[arcane]
fn wiener_filter5_8bpc_avx2_inner(
    _token: Desktop64,
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
    wiener_filter7_8bpc_avx2_inner(_token, p, left, lpf, lpf_off, w, h, params, edges);
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

    // Single guard for entire output region
    let (mut p_guard, p_base) = p.strided_slice_mut::<BitDepth16>(w, h);
    for j in 0..h {
        let row_off = p_base.wrapping_add_signed(j as isize * stride);
        for i in 0..w {
            let mut sum = -round_offset;

            for k in 0..7 {
                sum += hor[(j + k) * REST_UNIT_STRIDE + i] * filter[1][k] as i32;
            }

            p_guard[row_off + i] =
                iclip((sum + rounding_off_v) >> round_bits_v, 0, bitdepth_max) as u16;
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

    // SAFETY: AVX2 available (checked by dispatch)
    let token = unsafe { Desktop64::forge_token_dangerously() };
    wiener_filter7_8bpc_avx2_inner(token, p, left, lpf, lpf_off, w, h, params, edges);
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

    // SAFETY: AVX2 available (checked by dispatch)
    let token = unsafe { Desktop64::forge_token_dangerously() };
    wiener_filter5_8bpc_avx2_inner(token, p, left, lpf, lpf_off, w, h, params, edges);
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
    //
    // Scalar reference starts cursor aa at STRIDE+3, inner loop i from -1..w+1.
    // So first access is at STRIDE+2. We use aa_base = (row_offset+1)*STRIDE+2
    // with inner loop i from 0..w+2, giving the same index range.

    for row_offset in (0..(h + 2)).step_by(step) {
        let aa_base = (row_offset + 1) * REST_UNIT_STRIDE + 2;

        for i in 0..(w + 2) {
            let idx = aa_base + i;
            let a_val = sumsq[idx];
            let b_val = sum[idx] as i32;

            let p = cmp::max(a_val * n - b_val * b_val, 0) as u32;
            let z = (p * s + (1 << 19)) >> 20;
            let x = dav1d_sgr_x_by_x[cmp::min(z, 255) as usize] as u32;

            // Store inverted: a = x * b * sgr_one_by_x, b = x
            sumsq[idx] = ((x * (b_val as u32) * sgr_one_by_x + (1 << 11)) >> 12) as i32;
            sum[idx] = x as i16;
        }
    }

    // Apply neighbor-weighted filter to produce output
    let base = 2 * REST_UNIT_STRIDE + 3; // matches scalar cursor a/b starting position
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
                    let above = sum[idx - REST_UNIT_STRIDE] as i32;
                    let below = sum[idx + REST_UNIT_STRIDE] as i32;
                    let above_left = sum[idx - REST_UNIT_STRIDE - 1] as i32;
                    let above_right = sum[idx - REST_UNIT_STRIDE + 1] as i32;
                    let below_left = sum[idx + REST_UNIT_STRIDE - 1] as i32;
                    let below_right = sum[idx + REST_UNIT_STRIDE + 1] as i32;
                    (above + below) * 6 + (above_left + above_right + below_left + below_right) * 5
                };
                // six_neighbors for a (sumsq array)
                let a_six = {
                    let above = sumsq[idx - REST_UNIT_STRIDE];
                    let below = sumsq[idx + REST_UNIT_STRIDE];
                    let above_left = sumsq[idx - REST_UNIT_STRIDE - 1];
                    let above_right = sumsq[idx - REST_UNIT_STRIDE + 1];
                    let below_left = sumsq[idx + REST_UNIT_STRIDE - 1];
                    let below_right = sumsq[idx + REST_UNIT_STRIDE + 1];
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
                        let center = sum[idx] as i32;
                        let left = sum[idx - 1] as i32;
                        let right = sum[idx + 1] as i32;
                        center * 6 + (left + right) * 5
                    };
                    let a_horiz = {
                        let center = sumsq[idx];
                        let left = sumsq[idx - 1];
                        let right = sumsq[idx + 1];
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
                    let above = sum[idx - REST_UNIT_STRIDE] as i32;
                    let below = sum[idx + REST_UNIT_STRIDE] as i32;
                    let above_left = sum[idx - REST_UNIT_STRIDE - 1] as i32;
                    let above_right = sum[idx - REST_UNIT_STRIDE + 1] as i32;
                    let below_left = sum[idx + REST_UNIT_STRIDE - 1] as i32;
                    let below_right = sum[idx + REST_UNIT_STRIDE + 1] as i32;
                    (above + below) * 6 + (above_left + above_right + below_left + below_right) * 5
                };
                let a_six = {
                    let above = sumsq[idx - REST_UNIT_STRIDE];
                    let below = sumsq[idx + REST_UNIT_STRIDE];
                    let above_left = sumsq[idx - REST_UNIT_STRIDE - 1];
                    let above_right = sumsq[idx - REST_UNIT_STRIDE + 1];
                    let below_left = sumsq[idx + REST_UNIT_STRIDE - 1];
                    let below_right = sumsq[idx + REST_UNIT_STRIDE + 1];
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
                    let center = sum[idx] as i32;
                    let left = sum[idx - 1] as i32;
                    let right = sum[idx + 1] as i32;
                    let above = sum[idx - REST_UNIT_STRIDE] as i32;
                    let below = sum[idx + REST_UNIT_STRIDE] as i32;
                    let above_left = sum[idx - REST_UNIT_STRIDE - 1] as i32;
                    let above_right = sum[idx - REST_UNIT_STRIDE + 1] as i32;
                    let below_left = sum[idx + REST_UNIT_STRIDE - 1] as i32;
                    let below_right = sum[idx + REST_UNIT_STRIDE + 1] as i32;
                    (center + left + right + above + below) * 4
                        + (above_left + above_right + below_left + below_right) * 3
                };
                // eight_neighbors for a
                let a_eight = {
                    let center = sumsq[idx];
                    let left = sumsq[idx - 1];
                    let right = sumsq[idx + 1];
                    let above = sumsq[idx - REST_UNIT_STRIDE];
                    let below = sumsq[idx + REST_UNIT_STRIDE];
                    let above_left = sumsq[idx - REST_UNIT_STRIDE - 1];
                    let above_right = sumsq[idx - REST_UNIT_STRIDE + 1];
                    let below_left = sumsq[idx + REST_UNIT_STRIDE - 1];
                    let below_right = sumsq[idx + REST_UNIT_STRIDE + 1];
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

// ============================================================================
// AVX2 BOXSUM IMPLEMENTATIONS
// ============================================================================

/// AVX2 vertical boxsum for 5x5 window.
/// Processes 16 columns at a time (u8 src → i16 sum, i32 sumsq).
#[cfg(target_arch = "x86_64")]
#[rite]
fn boxsum5_v_avx2(
    _token: Desktop64,
    sumsq: &mut [i32; (64 + 2 + 2) * REST_UNIT_STRIDE],
    sum: &mut [i16; (64 + 2 + 2) * REST_UNIT_STRIDE],
    src: &[u8; (64 + 3 + 3) * REST_UNIT_STRIDE],
    w: usize,
    h: usize,
) {
    // Process 16 columns at a time
    let mut x = 0usize;
    while x + 16 <= w {
        // Process all output rows for these 16 columns
        for out_row in 2..h - 2 {
            // Load 5 rows of 16 u8 pixels, starting 2 rows before out_row
            let base_row = out_row - 2;
            let r0 = loadu_128!(
                &src[base_row * REST_UNIT_STRIDE + x..base_row * REST_UNIT_STRIDE + x + 16],
                [u8; 16]
            );
            let r1 = loadu_128!(
                &src[(base_row + 1) * REST_UNIT_STRIDE + x
                    ..(base_row + 1) * REST_UNIT_STRIDE + x + 16],
                [u8; 16]
            );
            let r2 = loadu_128!(
                &src[(base_row + 2) * REST_UNIT_STRIDE + x
                    ..(base_row + 2) * REST_UNIT_STRIDE + x + 16],
                [u8; 16]
            );
            let r3 = loadu_128!(
                &src[(base_row + 3) * REST_UNIT_STRIDE + x
                    ..(base_row + 3) * REST_UNIT_STRIDE + x + 16],
                [u8; 16]
            );
            let r4 = loadu_128!(
                &src[(base_row + 4) * REST_UNIT_STRIDE + x
                    ..(base_row + 4) * REST_UNIT_STRIDE + x + 16],
                [u8; 16]
            );

            // Zero-extend u8 to i16 (16 values → 16 i16 values in one ymm)
            let w0 = _mm256_cvtepu8_epi16(r0);
            let w1 = _mm256_cvtepu8_epi16(r1);
            let w2 = _mm256_cvtepu8_epi16(r2);
            let w3 = _mm256_cvtepu8_epi16(r3);
            let w4 = _mm256_cvtepu8_epi16(r4);

            // Sum of 5 rows (i16)
            let sum_v = _mm256_add_epi16(
                _mm256_add_epi16(_mm256_add_epi16(w0, w1), _mm256_add_epi16(w2, w3)),
                w4,
            );

            // Store sum (16 i16 values); out_row-1 matches scalar boxsum5_8bpc
            let sum_offset = (out_row - 1) * REST_UNIT_STRIDE + x;
            storeu_256!(&mut sum[sum_offset..sum_offset + 16], [i16; 16], sum_v);

            // Sum of squares: need i32 precision
            // pmaddwd with ones_16 squares and adds pairs: (a*1 + b*1) isn't right
            // Actually: pmaddwd(a, a) = a[0]*a[0] + a[1]*a[1] per dword
            // That's sum of pairs of squares, not individual squares
            // Instead: use pmulhw/pmullw or unpack to i32 and multiply

            // For sumsq: square each i16 value, accumulate as i32
            // madd(w, ones_16) gives sum of each i16, not square
            // madd(w, w) gives w[0]*w[0] + w[1]*w[1] per dword - pairs
            // We need individual squares. Use unpacklo/hi to i32 then multiply.

            // Actually: _mm256_madd_epi16(w, w) gives sum of adjacent squares
            // For boxsum, we want sum of all 5 squares per column
            // madd gives pairs summed. Not what we want per-column.

            // Better approach: use the raw 128-bit values and extend differently.
            // For sumsq of 5 u8 values, max is 5 * 255^2 = 325125 which fits i32.
            // Process 8 columns at a time for i32.

            // Low 8 values
            let lo_0 = _mm256_cvtepu8_epi32(r0);
            let lo_1 = _mm256_cvtepu8_epi32(r1);
            let lo_2 = _mm256_cvtepu8_epi32(r2);
            let lo_3 = _mm256_cvtepu8_epi32(r3);
            let lo_4 = _mm256_cvtepu8_epi32(r4);

            let sq0 = _mm256_mullo_epi32(lo_0, lo_0);
            let sq1 = _mm256_mullo_epi32(lo_1, lo_1);
            let sq2 = _mm256_mullo_epi32(lo_2, lo_2);
            let sq3 = _mm256_mullo_epi32(lo_3, lo_3);
            let sq4 = _mm256_mullo_epi32(lo_4, lo_4);

            let sumsq_lo = _mm256_add_epi32(
                _mm256_add_epi32(_mm256_add_epi32(sq0, sq1), _mm256_add_epi32(sq2, sq3)),
                sq4,
            );

            storeu_256!(&mut sumsq[sum_offset..sum_offset + 8], [i32; 8], sumsq_lo);

            // High 8 values (shift r0..r4 right by 8 bytes)
            let hi_0 = _mm256_cvtepu8_epi32(_mm_srli_si128::<8>(r0));
            let hi_1 = _mm256_cvtepu8_epi32(_mm_srli_si128::<8>(r1));
            let hi_2 = _mm256_cvtepu8_epi32(_mm_srli_si128::<8>(r2));
            let hi_3 = _mm256_cvtepu8_epi32(_mm_srli_si128::<8>(r3));
            let hi_4 = _mm256_cvtepu8_epi32(_mm_srli_si128::<8>(r4));

            let sq0h = _mm256_mullo_epi32(hi_0, hi_0);
            let sq1h = _mm256_mullo_epi32(hi_1, hi_1);
            let sq2h = _mm256_mullo_epi32(hi_2, hi_2);
            let sq3h = _mm256_mullo_epi32(hi_3, hi_3);
            let sq4h = _mm256_mullo_epi32(hi_4, hi_4);

            let sumsq_hi = _mm256_add_epi32(
                _mm256_add_epi32(_mm256_add_epi32(sq0h, sq1h), _mm256_add_epi32(sq2h, sq3h)),
                sq4h,
            );

            storeu_256!(
                &mut sumsq[sum_offset + 8..sum_offset + 16],
                [i32; 8],
                sumsq_hi
            );
        }
        x += 16;
    }

    // Scalar tail for remaining columns
    for x in x..w {
        let mut a = src[x] as i32;
        let mut a2 = a * a;
        let mut b = src[REST_UNIT_STRIDE + x] as i32;
        let mut b2 = b * b;
        let mut c = src[2 * REST_UNIT_STRIDE + x] as i32;
        let mut c2 = c * c;
        let mut d = src[3 * REST_UNIT_STRIDE + x] as i32;
        let mut d2 = d * d;
        let mut s_idx = 3 * REST_UNIT_STRIDE + x;
        for out_row in 2..h - 2 {
            s_idx += REST_UNIT_STRIDE;
            let e = src[s_idx] as i32;
            let e2 = e * e;
            let sum_v = (out_row - 1) * REST_UNIT_STRIDE + x;
            let sumsq_v = (out_row - 1) * REST_UNIT_STRIDE + x;
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
}

/// AVX2 horizontal boxsum for 5x5 window.
/// Uses shifted loads to avoid sliding-window dependencies.
#[cfg(target_arch = "x86_64")]
#[rite]
fn boxsum5_h_avx2(
    _token: Desktop64,
    sumsq: &mut [i32; (64 + 2 + 2) * REST_UNIT_STRIDE],
    sum: &mut [i16; (64 + 2 + 2) * REST_UNIT_STRIDE],
    w: usize,
    h: usize,
) {
    // Horizontal pass needs a temporary row buffer to avoid WAR hazard
    // (writes to position x overlap with reads from x+2 in next iteration)
    let mut sum_tmp = [0i16; REST_UNIT_STRIDE];
    let mut sumsq_tmp = [0i32; REST_UNIT_STRIDE];

    for row in 1..h - 3 {
        let row_off = row * REST_UNIT_STRIDE;

        // Process 16 i16 sums at a time
        let mut x = 2usize;
        while x + 16 <= w - 2 {
            // 5 shifted loads
            let s0 = loadu_256!(&sum[row_off + x - 2..row_off + x - 2 + 16], [i16; 16]);
            let s1 = loadu_256!(&sum[row_off + x - 1..row_off + x - 1 + 16], [i16; 16]);
            let s2 = loadu_256!(&sum[row_off + x..row_off + x + 16], [i16; 16]);
            let s3 = loadu_256!(&sum[row_off + x + 1..row_off + x + 1 + 16], [i16; 16]);
            let s4 = loadu_256!(&sum[row_off + x + 2..row_off + x + 2 + 16], [i16; 16]);

            let hsum = _mm256_add_epi16(
                _mm256_add_epi16(_mm256_add_epi16(s0, s1), _mm256_add_epi16(s2, s3)),
                s4,
            );
            storeu_256!(&mut sum_tmp[x..x + 16], [i16; 16], hsum);

            // 5 shifted loads for sumsq (8 i32 per ymm, need 2 iterations per 16 cols)
            for off in [0usize, 8] {
                let q0 = loadu_256!(
                    &sumsq[row_off + x + off - 2..row_off + x + off - 2 + 8],
                    [i32; 8]
                );
                let q1 = loadu_256!(
                    &sumsq[row_off + x + off - 1..row_off + x + off - 1 + 8],
                    [i32; 8]
                );
                let q2 = loadu_256!(&sumsq[row_off + x + off..row_off + x + off + 8], [i32; 8]);
                let q3 = loadu_256!(
                    &sumsq[row_off + x + off + 1..row_off + x + off + 1 + 8],
                    [i32; 8]
                );
                let q4 = loadu_256!(
                    &sumsq[row_off + x + off + 2..row_off + x + off + 2 + 8],
                    [i32; 8]
                );

                let hsumsq = _mm256_add_epi32(
                    _mm256_add_epi32(_mm256_add_epi32(q0, q1), _mm256_add_epi32(q2, q3)),
                    q4,
                );
                storeu_256!(&mut sumsq_tmp[x + off..x + off + 8], [i32; 8], hsumsq);
            }
            x += 16;
        }

        // Scalar tail for remaining columns
        while x < w - 2 {
            let a = sum[row_off + x - 2];
            let b = sum[row_off + x - 1];
            let c = sum[row_off + x];
            let d = sum[row_off + x + 1];
            let e = sum[row_off + x + 2];
            sum_tmp[x] = a + b + c + d + e;

            let a2 = sumsq[row_off + x - 2];
            let b2 = sumsq[row_off + x - 1];
            let c2 = sumsq[row_off + x];
            let d2 = sumsq[row_off + x + 1];
            let e2 = sumsq[row_off + x + 2];
            sumsq_tmp[x] = a2 + b2 + c2 + d2 + e2;
            x += 1;
        }

        // Copy temp row back to main arrays
        sum[row_off + 2..row_off + w - 2].copy_from_slice(&sum_tmp[2..w - 2]);
        sumsq[row_off + 2..row_off + w - 2].copy_from_slice(&sumsq_tmp[2..w - 2]);
    }
}

/// AVX2 vertical boxsum for 3x3 window.
#[cfg(target_arch = "x86_64")]
#[rite]
fn boxsum3_v_avx2(
    _token: Desktop64,
    sumsq: &mut [i32; (64 + 2 + 2) * REST_UNIT_STRIDE],
    sum: &mut [i16; (64 + 2 + 2) * REST_UNIT_STRIDE],
    src: &[u8; (64 + 3 + 3) * REST_UNIT_STRIDE],
    w: usize,
    h: usize,
) {
    let src = &src[REST_UNIT_STRIDE..]; // Skip first row (matching scalar)

    let mut x = 1usize;
    while x + 16 <= w - 1 {
        for out_row in 2..h - 2 {
            let base_row = out_row - 2; // src is already offset by 1 row; match scalar window
            let r0 = loadu_128!(
                &src[base_row * REST_UNIT_STRIDE + x..base_row * REST_UNIT_STRIDE + x + 16],
                [u8; 16]
            );
            let r1 = loadu_128!(
                &src[(base_row + 1) * REST_UNIT_STRIDE + x
                    ..(base_row + 1) * REST_UNIT_STRIDE + x + 16],
                [u8; 16]
            );
            let r2 = loadu_128!(
                &src[(base_row + 2) * REST_UNIT_STRIDE + x
                    ..(base_row + 2) * REST_UNIT_STRIDE + x + 16],
                [u8; 16]
            );

            // Sum i16
            let w0 = _mm256_cvtepu8_epi16(r0);
            let w1 = _mm256_cvtepu8_epi16(r1);
            let w2 = _mm256_cvtepu8_epi16(r2);
            let sum_v = _mm256_add_epi16(_mm256_add_epi16(w0, w1), w2);

            let sum_offset = (out_row - 1) * REST_UNIT_STRIDE + x;
            storeu_256!(&mut sum[sum_offset..sum_offset + 16], [i16; 16], sum_v);

            // Sumsq i32 - low 8
            let lo_0 = _mm256_cvtepu8_epi32(r0);
            let lo_1 = _mm256_cvtepu8_epi32(r1);
            let lo_2 = _mm256_cvtepu8_epi32(r2);
            let sq_lo = _mm256_add_epi32(
                _mm256_add_epi32(
                    _mm256_mullo_epi32(lo_0, lo_0),
                    _mm256_mullo_epi32(lo_1, lo_1),
                ),
                _mm256_mullo_epi32(lo_2, lo_2),
            );
            storeu_256!(&mut sumsq[sum_offset..sum_offset + 8], [i32; 8], sq_lo);

            // Sumsq i32 - high 8
            let hi_0 = _mm256_cvtepu8_epi32(_mm_srli_si128::<8>(r0));
            let hi_1 = _mm256_cvtepu8_epi32(_mm_srli_si128::<8>(r1));
            let hi_2 = _mm256_cvtepu8_epi32(_mm_srli_si128::<8>(r2));
            let sq_hi = _mm256_add_epi32(
                _mm256_add_epi32(
                    _mm256_mullo_epi32(hi_0, hi_0),
                    _mm256_mullo_epi32(hi_1, hi_1),
                ),
                _mm256_mullo_epi32(hi_2, hi_2),
            );
            storeu_256!(&mut sumsq[sum_offset + 8..sum_offset + 16], [i32; 8], sq_hi);
        }
        x += 16;
    }

    // Scalar tail
    for x in x..w - 1 {
        let mut a = src[x] as i32;
        let mut a2 = a * a;
        let mut b = src[REST_UNIT_STRIDE + x] as i32;
        let mut b2 = b * b;
        let mut s_idx = REST_UNIT_STRIDE + x;
        for out_row in 2..h - 2 {
            s_idx += REST_UNIT_STRIDE;
            let c = src[s_idx] as i32;
            let c2 = c * c;
            let sum_v = (out_row - 1) * REST_UNIT_STRIDE + x;
            sum[sum_v] = (a + b + c) as i16;
            sumsq[sum_v] = a2 + b2 + c2;
            a = b;
            a2 = b2;
            b = c;
            b2 = c2;
        }
    }
}

/// AVX2 horizontal boxsum for 3x3 window.
#[cfg(target_arch = "x86_64")]
#[rite]
fn boxsum3_h_avx2(
    _token: Desktop64,
    sumsq: &mut [i32; (64 + 2 + 2) * REST_UNIT_STRIDE],
    sum: &mut [i16; (64 + 2 + 2) * REST_UNIT_STRIDE],
    w: usize,
    h: usize,
) {
    let mut sum_tmp = [0i16; REST_UNIT_STRIDE];
    let mut sumsq_tmp = [0i32; REST_UNIT_STRIDE];

    for row in 1..h - 3 {
        let row_off = row * REST_UNIT_STRIDE;
        let mut x = 2usize;

        while x + 16 <= w - 2 {
            let s0 = loadu_256!(&sum[row_off + x - 1..row_off + x - 1 + 16], [i16; 16]);
            let s1 = loadu_256!(&sum[row_off + x..row_off + x + 16], [i16; 16]);
            let s2 = loadu_256!(&sum[row_off + x + 1..row_off + x + 1 + 16], [i16; 16]);
            let hsum = _mm256_add_epi16(_mm256_add_epi16(s0, s1), s2);
            storeu_256!(&mut sum_tmp[x..x + 16], [i16; 16], hsum);

            for off in [0usize, 8] {
                let q0 = loadu_256!(
                    &sumsq[row_off + x + off - 1..row_off + x + off - 1 + 8],
                    [i32; 8]
                );
                let q1 = loadu_256!(&sumsq[row_off + x + off..row_off + x + off + 8], [i32; 8]);
                let q2 = loadu_256!(
                    &sumsq[row_off + x + off + 1..row_off + x + off + 1 + 8],
                    [i32; 8]
                );
                let hsumsq = _mm256_add_epi32(_mm256_add_epi32(q0, q1), q2);
                storeu_256!(&mut sumsq_tmp[x + off..x + off + 8], [i32; 8], hsumsq);
            }
            x += 16;
        }

        while x < w - 2 {
            sum_tmp[x] = sum[row_off + x - 1] + sum[row_off + x] + sum[row_off + x + 1];
            sumsq_tmp[x] = sumsq[row_off + x - 1] + sumsq[row_off + x] + sumsq[row_off + x + 1];
            x += 1;
        }

        sum[row_off + 2..row_off + w - 2].copy_from_slice(&sum_tmp[2..w - 2]);
        sumsq[row_off + 2..row_off + w - 2].copy_from_slice(&sumsq_tmp[2..w - 2]);
    }
}

/// AVX2 selfguided filter for 8bpc.
/// Replaces the scalar version with SIMD boxsum + SIMD neighbor weighting.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn selfguided_filter_8bpc_avx2(
    _token: Desktop64,
    dst: &mut [i16; 64 * MAX_RESTORATION_WIDTH],
    src: &[u8; (64 + 3 + 3) * REST_UNIT_STRIDE],
    w: usize,
    h: usize,
    n: i32,
    s: u32,
) {
    let sgr_one_by_x: u32 = if n == 25 { 164 } else { 455 };

    let mut sumsq = [0i32; (64 + 2 + 2) * REST_UNIT_STRIDE];
    let mut sum = [0i16; (64 + 2 + 2) * REST_UNIT_STRIDE];

    let step = if n == 25 { 2 } else { 1 };

    // AVX2 boxsum
    if n == 25 {
        boxsum5_v_avx2(_token, &mut sumsq, &mut sum, src, w + 6, h + 6);
        boxsum5_h_avx2(_token, &mut sumsq, &mut sum, w + 6, h + 6);
    } else {
        boxsum3_v_avx2(_token, &mut sumsq, &mut sum, src, w + 6, h + 6);
        boxsum3_h_avx2(_token, &mut sumsq, &mut sum, w + 6, h + 6);
    }

    // Coefficient calculation (scalar — table lookup dominates)
    for row_offset in (0..(h + 2)).step_by(step) {
        let aa_base = (row_offset + 1) * REST_UNIT_STRIDE + 2;
        for i in 0..(w + 2) {
            let idx = aa_base + i;
            let a_val = sumsq[idx];
            let b_val = sum[idx] as i32;
            let p = cmp::max(a_val * n - b_val * b_val, 0) as u32;
            let z = (p * s + (1 << 19)) >> 20;
            let x = dav1d_sgr_x_by_x[cmp::min(z, 255) as usize] as u32;
            sumsq[idx] = ((x * (b_val as u32) * sgr_one_by_x + (1 << 11)) >> 12) as i32;
            sum[idx] = x as i16;
        }
    }

    // AVX2 neighbor-weighted filter
    let base = 2 * REST_UNIT_STRIDE + 3;
    let src_base = 3 * REST_UNIT_STRIDE + 3;
    let rounding_9 = _mm256_set1_epi32(1 << 8);
    let rounding_8 = _mm256_set1_epi32(1 << 7);
    let six = _mm256_set1_epi32(6);
    let five = _mm256_set1_epi32(5);
    let four = _mm256_set1_epi32(4);
    let three = _mm256_set1_epi32(3);

    if n == 25 {
        // 5x5: six_neighbors, step by 2 rows
        let mut j = 0usize;
        while j < h.saturating_sub(1) {
            // Even row: full 6-neighbor calculation
            let mut i = 0usize;
            while i + 8 <= w {
                let idx = base + j * REST_UNIT_STRIDE + i;

                // Load 6 neighbors from sum (i16 → i32)
                let sum_above = _mm256_cvtepi16_epi32(loadu_128!(
                    &sum[idx - REST_UNIT_STRIDE..idx - REST_UNIT_STRIDE + 8],
                    [i16; 8]
                ));
                let sum_below = _mm256_cvtepi16_epi32(loadu_128!(
                    &sum[idx + REST_UNIT_STRIDE..idx + REST_UNIT_STRIDE + 8],
                    [i16; 8]
                ));
                let sum_al = _mm256_cvtepi16_epi32(loadu_128!(
                    &sum[idx - REST_UNIT_STRIDE - 1..idx - REST_UNIT_STRIDE - 1 + 8],
                    [i16; 8]
                ));
                let sum_ar = _mm256_cvtepi16_epi32(loadu_128!(
                    &sum[idx - REST_UNIT_STRIDE + 1..idx - REST_UNIT_STRIDE + 1 + 8],
                    [i16; 8]
                ));
                let sum_bl = _mm256_cvtepi16_epi32(loadu_128!(
                    &sum[idx + REST_UNIT_STRIDE - 1..idx + REST_UNIT_STRIDE - 1 + 8],
                    [i16; 8]
                ));
                let sum_br = _mm256_cvtepi16_epi32(loadu_128!(
                    &sum[idx + REST_UNIT_STRIDE + 1..idx + REST_UNIT_STRIDE + 1 + 8],
                    [i16; 8]
                ));

                let b_six = _mm256_add_epi32(
                    _mm256_mullo_epi32(_mm256_add_epi32(sum_above, sum_below), six),
                    _mm256_mullo_epi32(
                        _mm256_add_epi32(
                            _mm256_add_epi32(sum_al, sum_ar),
                            _mm256_add_epi32(sum_bl, sum_br),
                        ),
                        five,
                    ),
                );

                // Load 6 neighbors from sumsq (already i32)
                let sq_above = loadu_256!(
                    &sumsq[idx - REST_UNIT_STRIDE..idx - REST_UNIT_STRIDE + 8],
                    [i32; 8]
                );
                let sq_below = loadu_256!(
                    &sumsq[idx + REST_UNIT_STRIDE..idx + REST_UNIT_STRIDE + 8],
                    [i32; 8]
                );
                let sq_al = loadu_256!(
                    &sumsq[idx - REST_UNIT_STRIDE - 1..idx - REST_UNIT_STRIDE - 1 + 8],
                    [i32; 8]
                );
                let sq_ar = loadu_256!(
                    &sumsq[idx - REST_UNIT_STRIDE + 1..idx - REST_UNIT_STRIDE + 1 + 8],
                    [i32; 8]
                );
                let sq_bl = loadu_256!(
                    &sumsq[idx + REST_UNIT_STRIDE - 1..idx + REST_UNIT_STRIDE - 1 + 8],
                    [i32; 8]
                );
                let sq_br = loadu_256!(
                    &sumsq[idx + REST_UNIT_STRIDE + 1..idx + REST_UNIT_STRIDE + 1 + 8],
                    [i32; 8]
                );

                let a_six = _mm256_add_epi32(
                    _mm256_mullo_epi32(_mm256_add_epi32(sq_above, sq_below), six),
                    _mm256_mullo_epi32(
                        _mm256_add_epi32(
                            _mm256_add_epi32(sq_al, sq_ar),
                            _mm256_add_epi32(sq_bl, sq_br),
                        ),
                        five,
                    ),
                );

                // Load src pixels (u8 → i32)
                let src_val = _mm256_cvtepu8_epi32(loadi64!(
                    &src[src_base + j * REST_UNIT_STRIDE + i
                        ..src_base + j * REST_UNIT_STRIDE + i + 8]
                ));

                // dst = (a_six - b_six * src_val + (1 << 8)) >> 9
                let result = _mm256_srai_epi32::<9>(_mm256_add_epi32(
                    _mm256_sub_epi32(a_six, _mm256_mullo_epi32(b_six, src_val)),
                    rounding_9,
                ));

                // Pack i32 to i16 and store
                let result_16 = _mm256_packs_epi32(result, _mm256_setzero_si256());
                // packs interleaves lanes: need permute to get correct order
                let result_16 = _mm256_permute4x64_epi64::<0xD8>(result_16);
                storeu_128!(
                    &mut dst[j * MAX_RESTORATION_WIDTH + i..j * MAX_RESTORATION_WIDTH + i + 8],
                    [i16; 8],
                    _mm256_castsi256_si128(result_16)
                );

                i += 8;
            }
            // Scalar tail for even row
            while i < w {
                let idx = base + j * REST_UNIT_STRIDE + i;
                let b_six = {
                    let above = sum[idx - REST_UNIT_STRIDE] as i32;
                    let below = sum[idx + REST_UNIT_STRIDE] as i32;
                    let al = sum[idx - REST_UNIT_STRIDE - 1] as i32;
                    let ar = sum[idx - REST_UNIT_STRIDE + 1] as i32;
                    let bl = sum[idx + REST_UNIT_STRIDE - 1] as i32;
                    let br = sum[idx + REST_UNIT_STRIDE + 1] as i32;
                    (above + below) * 6 + (al + ar + bl + br) * 5
                };
                let a_six = {
                    let above = sumsq[idx - REST_UNIT_STRIDE];
                    let below = sumsq[idx + REST_UNIT_STRIDE];
                    let al = sumsq[idx - REST_UNIT_STRIDE - 1];
                    let ar = sumsq[idx - REST_UNIT_STRIDE + 1];
                    let bl = sumsq[idx + REST_UNIT_STRIDE - 1];
                    let br = sumsq[idx + REST_UNIT_STRIDE + 1];
                    (above + below) * 6 + (al + ar + bl + br) * 5
                };
                let src_val = src[src_base + j * REST_UNIT_STRIDE + i] as i32;
                dst[j * MAX_RESTORATION_WIDTH + i] =
                    ((a_six - b_six * src_val + (1 << 8)) >> 9) as i16;
                i += 1;
            }

            // Odd row: 3-neighbor horizontal
            if j + 1 < h {
                let mut i = 0usize;
                while i + 8 <= w {
                    let idx = base + (j + 1) * REST_UNIT_STRIDE + i;

                    let sum_center =
                        _mm256_cvtepi16_epi32(loadu_128!(&sum[idx..idx + 8], [i16; 8]));
                    let sum_left =
                        _mm256_cvtepi16_epi32(loadu_128!(&sum[idx - 1..idx - 1 + 8], [i16; 8]));
                    let sum_right =
                        _mm256_cvtepi16_epi32(loadu_128!(&sum[idx + 1..idx + 1 + 8], [i16; 8]));

                    let b_horiz = _mm256_add_epi32(
                        _mm256_mullo_epi32(sum_center, six),
                        _mm256_mullo_epi32(_mm256_add_epi32(sum_left, sum_right), five),
                    );

                    let sq_center = loadu_256!(&sumsq[idx..idx + 8], [i32; 8]);
                    let sq_left = loadu_256!(&sumsq[idx - 1..idx - 1 + 8], [i32; 8]);
                    let sq_right = loadu_256!(&sumsq[idx + 1..idx + 1 + 8], [i32; 8]);

                    let a_horiz = _mm256_add_epi32(
                        _mm256_mullo_epi32(sq_center, six),
                        _mm256_mullo_epi32(_mm256_add_epi32(sq_left, sq_right), five),
                    );

                    let src_val = _mm256_cvtepu8_epi32(loadi64!(
                        &src[src_base + (j + 1) * REST_UNIT_STRIDE + i
                            ..src_base + (j + 1) * REST_UNIT_STRIDE + i + 8]
                    ));

                    let result = _mm256_srai_epi32::<8>(_mm256_add_epi32(
                        _mm256_sub_epi32(a_horiz, _mm256_mullo_epi32(b_horiz, src_val)),
                        rounding_8,
                    ));

                    let result_16 = _mm256_packs_epi32(result, _mm256_setzero_si256());
                    let result_16 = _mm256_permute4x64_epi64::<0xD8>(result_16);
                    storeu_128!(
                        &mut dst[(j + 1) * MAX_RESTORATION_WIDTH + i
                            ..(j + 1) * MAX_RESTORATION_WIDTH + i + 8],
                        [i16; 8],
                        _mm256_castsi256_si128(result_16)
                    );

                    i += 8;
                }
                // Scalar tail for odd row
                while i < w {
                    let idx = base + (j + 1) * REST_UNIT_STRIDE + i;
                    let b_horiz = {
                        let center = sum[idx] as i32;
                        let left = sum[idx - 1] as i32;
                        let right = sum[idx + 1] as i32;
                        center * 6 + (left + right) * 5
                    };
                    let a_horiz = {
                        let center = sumsq[idx];
                        let left = sumsq[idx - 1];
                        let right = sumsq[idx + 1];
                        center * 6 + (left + right) * 5
                    };
                    let src_val = src[src_base + (j + 1) * REST_UNIT_STRIDE + i] as i32;
                    dst[(j + 1) * MAX_RESTORATION_WIDTH + i] =
                        ((a_horiz - b_horiz * src_val + (1 << 7)) >> 8) as i16;
                    i += 1;
                }
            }
            j += 2;
        }
        // Handle last row if height is odd
        if j < h {
            for i in 0..w {
                let idx = base + j * REST_UNIT_STRIDE + i;
                let b_six = {
                    let above = sum[idx - REST_UNIT_STRIDE] as i32;
                    let below = sum[idx + REST_UNIT_STRIDE] as i32;
                    let al = sum[idx - REST_UNIT_STRIDE - 1] as i32;
                    let ar = sum[idx - REST_UNIT_STRIDE + 1] as i32;
                    let bl = sum[idx + REST_UNIT_STRIDE - 1] as i32;
                    let br = sum[idx + REST_UNIT_STRIDE + 1] as i32;
                    (above + below) * 6 + (al + ar + bl + br) * 5
                };
                let a_six = {
                    let above = sumsq[idx - REST_UNIT_STRIDE];
                    let below = sumsq[idx + REST_UNIT_STRIDE];
                    let al = sumsq[idx - REST_UNIT_STRIDE - 1];
                    let ar = sumsq[idx - REST_UNIT_STRIDE + 1];
                    let bl = sumsq[idx + REST_UNIT_STRIDE - 1];
                    let br = sumsq[idx + REST_UNIT_STRIDE + 1];
                    (above + below) * 6 + (al + ar + bl + br) * 5
                };
                let src_val = src[src_base + j * REST_UNIT_STRIDE + i] as i32;
                dst[j * MAX_RESTORATION_WIDTH + i] =
                    ((a_six - b_six * src_val + (1 << 8)) >> 9) as i16;
            }
        }
    } else {
        // 3x3: eight_neighbors
        for j in 0..h {
            let mut i = 0usize;
            while i + 8 <= w {
                let idx = base + j * REST_UNIT_STRIDE + i;

                // 9 neighbors for sum
                let s_c = _mm256_cvtepi16_epi32(loadu_128!(&sum[idx..idx + 8], [i16; 8]));
                let s_l = _mm256_cvtepi16_epi32(loadu_128!(&sum[idx - 1..idx - 1 + 8], [i16; 8]));
                let s_r = _mm256_cvtepi16_epi32(loadu_128!(&sum[idx + 1..idx + 1 + 8], [i16; 8]));
                let s_a = _mm256_cvtepi16_epi32(loadu_128!(
                    &sum[idx - REST_UNIT_STRIDE..idx - REST_UNIT_STRIDE + 8],
                    [i16; 8]
                ));
                let s_b = _mm256_cvtepi16_epi32(loadu_128!(
                    &sum[idx + REST_UNIT_STRIDE..idx + REST_UNIT_STRIDE + 8],
                    [i16; 8]
                ));
                let s_al = _mm256_cvtepi16_epi32(loadu_128!(
                    &sum[idx - REST_UNIT_STRIDE - 1..idx - REST_UNIT_STRIDE - 1 + 8],
                    [i16; 8]
                ));
                let s_ar = _mm256_cvtepi16_epi32(loadu_128!(
                    &sum[idx - REST_UNIT_STRIDE + 1..idx - REST_UNIT_STRIDE + 1 + 8],
                    [i16; 8]
                ));
                let s_bl = _mm256_cvtepi16_epi32(loadu_128!(
                    &sum[idx + REST_UNIT_STRIDE - 1..idx + REST_UNIT_STRIDE - 1 + 8],
                    [i16; 8]
                ));
                let s_br = _mm256_cvtepi16_epi32(loadu_128!(
                    &sum[idx + REST_UNIT_STRIDE + 1..idx + REST_UNIT_STRIDE + 1 + 8],
                    [i16; 8]
                ));

                let b_eight = _mm256_add_epi32(
                    _mm256_mullo_epi32(
                        _mm256_add_epi32(
                            _mm256_add_epi32(s_c, _mm256_add_epi32(s_l, s_r)),
                            _mm256_add_epi32(s_a, s_b),
                        ),
                        four,
                    ),
                    _mm256_mullo_epi32(
                        _mm256_add_epi32(
                            _mm256_add_epi32(s_al, s_ar),
                            _mm256_add_epi32(s_bl, s_br),
                        ),
                        three,
                    ),
                );

                // 9 neighbors for sumsq
                let q_c = loadu_256!(&sumsq[idx..idx + 8], [i32; 8]);
                let q_l = loadu_256!(&sumsq[idx - 1..idx - 1 + 8], [i32; 8]);
                let q_r = loadu_256!(&sumsq[idx + 1..idx + 1 + 8], [i32; 8]);
                let q_a = loadu_256!(
                    &sumsq[idx - REST_UNIT_STRIDE..idx - REST_UNIT_STRIDE + 8],
                    [i32; 8]
                );
                let q_b = loadu_256!(
                    &sumsq[idx + REST_UNIT_STRIDE..idx + REST_UNIT_STRIDE + 8],
                    [i32; 8]
                );
                let q_al = loadu_256!(
                    &sumsq[idx - REST_UNIT_STRIDE - 1..idx - REST_UNIT_STRIDE - 1 + 8],
                    [i32; 8]
                );
                let q_ar = loadu_256!(
                    &sumsq[idx - REST_UNIT_STRIDE + 1..idx - REST_UNIT_STRIDE + 1 + 8],
                    [i32; 8]
                );
                let q_bl = loadu_256!(
                    &sumsq[idx + REST_UNIT_STRIDE - 1..idx + REST_UNIT_STRIDE - 1 + 8],
                    [i32; 8]
                );
                let q_br = loadu_256!(
                    &sumsq[idx + REST_UNIT_STRIDE + 1..idx + REST_UNIT_STRIDE + 1 + 8],
                    [i32; 8]
                );

                let a_eight = _mm256_add_epi32(
                    _mm256_mullo_epi32(
                        _mm256_add_epi32(
                            _mm256_add_epi32(q_c, _mm256_add_epi32(q_l, q_r)),
                            _mm256_add_epi32(q_a, q_b),
                        ),
                        four,
                    ),
                    _mm256_mullo_epi32(
                        _mm256_add_epi32(
                            _mm256_add_epi32(q_al, q_ar),
                            _mm256_add_epi32(q_bl, q_br),
                        ),
                        three,
                    ),
                );

                let src_val = _mm256_cvtepu8_epi32(loadi64!(
                    &src[src_base + j * REST_UNIT_STRIDE + i
                        ..src_base + j * REST_UNIT_STRIDE + i + 8]
                ));

                let result = _mm256_srai_epi32::<9>(_mm256_add_epi32(
                    _mm256_sub_epi32(a_eight, _mm256_mullo_epi32(b_eight, src_val)),
                    rounding_9,
                ));

                let result_16 = _mm256_packs_epi32(result, _mm256_setzero_si256());
                let result_16 = _mm256_permute4x64_epi64::<0xD8>(result_16);
                storeu_128!(
                    &mut dst[j * MAX_RESTORATION_WIDTH + i..j * MAX_RESTORATION_WIDTH + i + 8],
                    [i16; 8],
                    _mm256_castsi256_si128(result_16)
                );

                i += 8;
            }
            // Scalar tail for 3x3
            while i < w {
                let idx = base + j * REST_UNIT_STRIDE + i;
                let b_eight = {
                    let center = sum[idx] as i32;
                    let left = sum[idx - 1] as i32;
                    let right = sum[idx + 1] as i32;
                    let above = sum[idx - REST_UNIT_STRIDE] as i32;
                    let below = sum[idx + REST_UNIT_STRIDE] as i32;
                    let al = sum[idx - REST_UNIT_STRIDE - 1] as i32;
                    let ar = sum[idx - REST_UNIT_STRIDE + 1] as i32;
                    let bl = sum[idx + REST_UNIT_STRIDE - 1] as i32;
                    let br = sum[idx + REST_UNIT_STRIDE + 1] as i32;
                    (center + left + right + above + below) * 4 + (al + ar + bl + br) * 3
                };
                let a_eight = {
                    let center = sumsq[idx];
                    let left = sumsq[idx - 1];
                    let right = sumsq[idx + 1];
                    let above = sumsq[idx - REST_UNIT_STRIDE];
                    let below = sumsq[idx + REST_UNIT_STRIDE];
                    let al = sumsq[idx - REST_UNIT_STRIDE - 1];
                    let ar = sumsq[idx - REST_UNIT_STRIDE + 1];
                    let bl = sumsq[idx + REST_UNIT_STRIDE - 1];
                    let br = sumsq[idx + REST_UNIT_STRIDE + 1];
                    (center + left + right + above + below) * 4 + (al + ar + bl + br) * 3
                };
                let src_val = src[src_base + j * REST_UNIT_STRIDE + i] as i32;
                dst[j * MAX_RESTORATION_WIDTH + i] =
                    ((a_eight - b_eight * src_val + (1 << 8)) >> 9) as i16;
                i += 1;
            }
        }
    }
}

/// SGR self-guided filter for 8bpc using AVX-512
///
/// Processes 16 pixels per iteration in the neighbor-weighted filter stage
/// (vs 8 for AVX2), using 512-bit registers for i32 accumulation.
/// Boxsum and coefficient calculation remain AVX2/scalar.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn selfguided_filter_8bpc_avx512(
    _token: Server64,
    dst: &mut [i16; 64 * MAX_RESTORATION_WIDTH],
    src: &[u8; (64 + 3 + 3) * REST_UNIT_STRIDE],
    w: usize,
    h: usize,
    n: i32,
    s: u32,
) {
    let sgr_one_by_x: u32 = if n == 25 { 164 } else { 455 };

    let mut sumsq = [0i32; (64 + 2 + 2) * REST_UNIT_STRIDE];
    let mut sum = [0i16; (64 + 2 + 2) * REST_UNIT_STRIDE];

    let step = if n == 25 { 2 } else { 1 };

    // Boxsum using AVX2 (Server64 implies Desktop64 features)
    let avx2_token = Desktop64::summon().unwrap();
    if n == 25 {
        boxsum5_v_avx2(avx2_token, &mut sumsq, &mut sum, src, w + 6, h + 6);
        boxsum5_h_avx2(avx2_token, &mut sumsq, &mut sum, w + 6, h + 6);
    } else {
        boxsum3_v_avx2(avx2_token, &mut sumsq, &mut sum, src, w + 6, h + 6);
        boxsum3_h_avx2(avx2_token, &mut sumsq, &mut sum, w + 6, h + 6);
    }

    // Coefficient calculation (scalar — table lookup dominates)
    for row_offset in (0..(h + 2)).step_by(step) {
        let aa_base = (row_offset + 1) * REST_UNIT_STRIDE + 2;
        for i in 0..(w + 2) {
            let idx = aa_base + i;
            let a_val = sumsq[idx];
            let b_val = sum[idx] as i32;
            let p = cmp::max(a_val * n - b_val * b_val, 0) as u32;
            let z = (p * s + (1 << 19)) >> 20;
            let x = dav1d_sgr_x_by_x[cmp::min(z, 255) as usize] as u32;
            sumsq[idx] = ((x * (b_val as u32) * sgr_one_by_x + (1 << 11)) >> 12) as i32;
            sum[idx] = x as i16;
        }
    }

    // AVX-512 neighbor-weighted filter (16 pixels per iteration)
    let base = 2 * REST_UNIT_STRIDE + 3;
    let src_base = 3 * REST_UNIT_STRIDE + 3;
    let rounding_9 = _mm512_set1_epi32(1 << 8);
    let rounding_8 = _mm512_set1_epi32(1 << 7);
    let six = _mm512_set1_epi32(6);
    let five = _mm512_set1_epi32(5);
    let four = _mm512_set1_epi32(4);
    let three = _mm512_set1_epi32(3);

    if n == 25 {
        // 5x5: six_neighbors, step by 2 rows
        let mut j = 0usize;
        while j < h.saturating_sub(1) {
            // Even row: full 6-neighbor calculation
            let mut i = 0usize;
            while i + 16 <= w {
                let idx = base + j * REST_UNIT_STRIDE + i;

                // Load 16 neighbors from sum (i16 → i32 via 256→512 expand)
                let sum_above = _mm512_cvtepi16_epi32(loadu_256!(
                    &sum[idx - REST_UNIT_STRIDE..idx - REST_UNIT_STRIDE + 16],
                    [i16; 16]
                ));
                let sum_below = _mm512_cvtepi16_epi32(loadu_256!(
                    &sum[idx + REST_UNIT_STRIDE..idx + REST_UNIT_STRIDE + 16],
                    [i16; 16]
                ));
                let sum_al = _mm512_cvtepi16_epi32(loadu_256!(
                    &sum[idx - REST_UNIT_STRIDE - 1..idx - REST_UNIT_STRIDE - 1 + 16],
                    [i16; 16]
                ));
                let sum_ar = _mm512_cvtepi16_epi32(loadu_256!(
                    &sum[idx - REST_UNIT_STRIDE + 1..idx - REST_UNIT_STRIDE + 1 + 16],
                    [i16; 16]
                ));
                let sum_bl = _mm512_cvtepi16_epi32(loadu_256!(
                    &sum[idx + REST_UNIT_STRIDE - 1..idx + REST_UNIT_STRIDE - 1 + 16],
                    [i16; 16]
                ));
                let sum_br = _mm512_cvtepi16_epi32(loadu_256!(
                    &sum[idx + REST_UNIT_STRIDE + 1..idx + REST_UNIT_STRIDE + 1 + 16],
                    [i16; 16]
                ));

                let b_six = _mm512_add_epi32(
                    _mm512_mullo_epi32(_mm512_add_epi32(sum_above, sum_below), six),
                    _mm512_mullo_epi32(
                        _mm512_add_epi32(
                            _mm512_add_epi32(sum_al, sum_ar),
                            _mm512_add_epi32(sum_bl, sum_br),
                        ),
                        five,
                    ),
                );

                // Load 16 neighbors from sumsq (already i32)
                let sq_above = loadu_512!(
                    &sumsq[idx - REST_UNIT_STRIDE..idx - REST_UNIT_STRIDE + 16],
                    [i32; 16]
                );
                let sq_below = loadu_512!(
                    &sumsq[idx + REST_UNIT_STRIDE..idx + REST_UNIT_STRIDE + 16],
                    [i32; 16]
                );
                let sq_al = loadu_512!(
                    &sumsq[idx - REST_UNIT_STRIDE - 1..idx - REST_UNIT_STRIDE - 1 + 16],
                    [i32; 16]
                );
                let sq_ar = loadu_512!(
                    &sumsq[idx - REST_UNIT_STRIDE + 1..idx - REST_UNIT_STRIDE + 1 + 16],
                    [i32; 16]
                );
                let sq_bl = loadu_512!(
                    &sumsq[idx + REST_UNIT_STRIDE - 1..idx + REST_UNIT_STRIDE - 1 + 16],
                    [i32; 16]
                );
                let sq_br = loadu_512!(
                    &sumsq[idx + REST_UNIT_STRIDE + 1..idx + REST_UNIT_STRIDE + 1 + 16],
                    [i32; 16]
                );

                let a_six = _mm512_add_epi32(
                    _mm512_mullo_epi32(_mm512_add_epi32(sq_above, sq_below), six),
                    _mm512_mullo_epi32(
                        _mm512_add_epi32(
                            _mm512_add_epi32(sq_al, sq_ar),
                            _mm512_add_epi32(sq_bl, sq_br),
                        ),
                        five,
                    ),
                );

                // Load src pixels (u8 → i32): load 16 bytes, zero-extend to 16 i32
                let src_bytes = loadu_128!(
                    &src[src_base + j * REST_UNIT_STRIDE + i
                        ..src_base + j * REST_UNIT_STRIDE + i + 16],
                    [u8; 16]
                );
                let src_val = _mm512_cvtepu8_epi32(src_bytes);

                // dst = (a_six - b_six * src_val + (1 << 8)) >> 9
                let result = _mm512_srai_epi32::<9>(_mm512_add_epi32(
                    _mm512_sub_epi32(a_six, _mm512_mullo_epi32(b_six, src_val)),
                    rounding_9,
                ));

                // Pack i32 to i16: 16 i32 → 16 i16 (signed saturation)
                let result_16 = _mm512_cvtsepi32_epi16(result);
                storeu_256!(
                    &mut dst[j * MAX_RESTORATION_WIDTH + i..j * MAX_RESTORATION_WIDTH + i + 16],
                    [i16; 16],
                    result_16
                );

                i += 16;
            }
            // AVX2 tail for remaining 8-pixel chunks
            while i + 8 <= w {
                let idx = base + j * REST_UNIT_STRIDE + i;
                let sum_above = _mm256_cvtepi16_epi32(loadu_128!(
                    &sum[idx - REST_UNIT_STRIDE..idx - REST_UNIT_STRIDE + 8],
                    [i16; 8]
                ));
                let sum_below = _mm256_cvtepi16_epi32(loadu_128!(
                    &sum[idx + REST_UNIT_STRIDE..idx + REST_UNIT_STRIDE + 8],
                    [i16; 8]
                ));
                let sum_al = _mm256_cvtepi16_epi32(loadu_128!(
                    &sum[idx - REST_UNIT_STRIDE - 1..idx - REST_UNIT_STRIDE - 1 + 8],
                    [i16; 8]
                ));
                let sum_ar = _mm256_cvtepi16_epi32(loadu_128!(
                    &sum[idx - REST_UNIT_STRIDE + 1..idx - REST_UNIT_STRIDE + 1 + 8],
                    [i16; 8]
                ));
                let sum_bl = _mm256_cvtepi16_epi32(loadu_128!(
                    &sum[idx + REST_UNIT_STRIDE - 1..idx + REST_UNIT_STRIDE - 1 + 8],
                    [i16; 8]
                ));
                let sum_br = _mm256_cvtepi16_epi32(loadu_128!(
                    &sum[idx + REST_UNIT_STRIDE + 1..idx + REST_UNIT_STRIDE + 1 + 8],
                    [i16; 8]
                ));
                let six_256 = _mm256_set1_epi32(6);
                let five_256 = _mm256_set1_epi32(5);
                let b_six = _mm256_add_epi32(
                    _mm256_mullo_epi32(_mm256_add_epi32(sum_above, sum_below), six_256),
                    _mm256_mullo_epi32(
                        _mm256_add_epi32(
                            _mm256_add_epi32(sum_al, sum_ar),
                            _mm256_add_epi32(sum_bl, sum_br),
                        ),
                        five_256,
                    ),
                );
                let sq_above = loadu_256!(
                    &sumsq[idx - REST_UNIT_STRIDE..idx - REST_UNIT_STRIDE + 8],
                    [i32; 8]
                );
                let sq_below = loadu_256!(
                    &sumsq[idx + REST_UNIT_STRIDE..idx + REST_UNIT_STRIDE + 8],
                    [i32; 8]
                );
                let sq_al = loadu_256!(
                    &sumsq[idx - REST_UNIT_STRIDE - 1..idx - REST_UNIT_STRIDE - 1 + 8],
                    [i32; 8]
                );
                let sq_ar = loadu_256!(
                    &sumsq[idx - REST_UNIT_STRIDE + 1..idx - REST_UNIT_STRIDE + 1 + 8],
                    [i32; 8]
                );
                let sq_bl = loadu_256!(
                    &sumsq[idx + REST_UNIT_STRIDE - 1..idx + REST_UNIT_STRIDE - 1 + 8],
                    [i32; 8]
                );
                let sq_br = loadu_256!(
                    &sumsq[idx + REST_UNIT_STRIDE + 1..idx + REST_UNIT_STRIDE + 1 + 8],
                    [i32; 8]
                );
                let a_six = _mm256_add_epi32(
                    _mm256_mullo_epi32(_mm256_add_epi32(sq_above, sq_below), six_256),
                    _mm256_mullo_epi32(
                        _mm256_add_epi32(
                            _mm256_add_epi32(sq_al, sq_ar),
                            _mm256_add_epi32(sq_bl, sq_br),
                        ),
                        five_256,
                    ),
                );
                let src_val = _mm256_cvtepu8_epi32(loadi64!(
                    &src[src_base + j * REST_UNIT_STRIDE + i
                        ..src_base + j * REST_UNIT_STRIDE + i + 8]
                ));
                let result = _mm256_srai_epi32::<9>(_mm256_add_epi32(
                    _mm256_sub_epi32(a_six, _mm256_mullo_epi32(b_six, src_val)),
                    _mm256_set1_epi32(1 << 8),
                ));
                let result_16 = _mm256_packs_epi32(result, _mm256_setzero_si256());
                let result_16 = _mm256_permute4x64_epi64::<0xD8>(result_16);
                storeu_128!(
                    &mut dst[j * MAX_RESTORATION_WIDTH + i..j * MAX_RESTORATION_WIDTH + i + 8],
                    [i16; 8],
                    _mm256_castsi256_si128(result_16)
                );
                i += 8;
            }
            // Scalar tail for even row
            while i < w {
                let idx = base + j * REST_UNIT_STRIDE + i;
                let b_six = {
                    let above = sum[idx - REST_UNIT_STRIDE] as i32;
                    let below = sum[idx + REST_UNIT_STRIDE] as i32;
                    let al = sum[idx - REST_UNIT_STRIDE - 1] as i32;
                    let ar = sum[idx - REST_UNIT_STRIDE + 1] as i32;
                    let bl = sum[idx + REST_UNIT_STRIDE - 1] as i32;
                    let br = sum[idx + REST_UNIT_STRIDE + 1] as i32;
                    (above + below) * 6 + (al + ar + bl + br) * 5
                };
                let a_six = {
                    let above = sumsq[idx - REST_UNIT_STRIDE];
                    let below = sumsq[idx + REST_UNIT_STRIDE];
                    let al = sumsq[idx - REST_UNIT_STRIDE - 1];
                    let ar = sumsq[idx - REST_UNIT_STRIDE + 1];
                    let bl = sumsq[idx + REST_UNIT_STRIDE - 1];
                    let br = sumsq[idx + REST_UNIT_STRIDE + 1];
                    (above + below) * 6 + (al + ar + bl + br) * 5
                };
                let src_val = src[src_base + j * REST_UNIT_STRIDE + i] as i32;
                dst[j * MAX_RESTORATION_WIDTH + i] =
                    ((a_six - b_six * src_val + (1 << 8)) >> 9) as i16;
                i += 1;
            }

            // Odd row: 3-neighbor horizontal (same as AVX2 version)
            if j + 1 < h {
                let mut i = 0usize;
                while i + 16 <= w {
                    let idx = base + (j + 1) * REST_UNIT_STRIDE + i;

                    let sum_center = _mm512_cvtepi16_epi32(loadu_256!(
                        &sum[idx..idx + 16],
                        [i16; 16]
                    ));
                    let sum_left = _mm512_cvtepi16_epi32(loadu_256!(
                        &sum[idx - 1..idx - 1 + 16],
                        [i16; 16]
                    ));
                    let sum_right = _mm512_cvtepi16_epi32(loadu_256!(
                        &sum[idx + 1..idx + 1 + 16],
                        [i16; 16]
                    ));

                    let b_horiz = _mm512_add_epi32(
                        _mm512_mullo_epi32(sum_center, six),
                        _mm512_mullo_epi32(_mm512_add_epi32(sum_left, sum_right), five),
                    );

                    let sq_center = loadu_512!(&sumsq[idx..idx + 16], [i32; 16]);
                    let sq_left = loadu_512!(&sumsq[idx - 1..idx - 1 + 16], [i32; 16]);
                    let sq_right = loadu_512!(&sumsq[idx + 1..idx + 1 + 16], [i32; 16]);

                    let a_horiz = _mm512_add_epi32(
                        _mm512_mullo_epi32(sq_center, six),
                        _mm512_mullo_epi32(_mm512_add_epi32(sq_left, sq_right), five),
                    );

                    let src_bytes = loadu_128!(
                        &src[src_base + (j + 1) * REST_UNIT_STRIDE + i
                            ..src_base + (j + 1) * REST_UNIT_STRIDE + i + 16],
                        [u8; 16]
                    );
                    let src_val = _mm512_cvtepu8_epi32(src_bytes);

                    let result = _mm512_srai_epi32::<8>(_mm512_add_epi32(
                        _mm512_sub_epi32(a_horiz, _mm512_mullo_epi32(b_horiz, src_val)),
                        rounding_8,
                    ));

                    let result_16 = _mm512_cvtsepi32_epi16(result);
                    storeu_256!(
                        &mut dst[(j + 1) * MAX_RESTORATION_WIDTH + i
                            ..(j + 1) * MAX_RESTORATION_WIDTH + i + 16],
                        [i16; 16],
                        result_16
                    );

                    i += 16;
                }
                // Scalar tail for odd row
                while i < w {
                    let idx = base + (j + 1) * REST_UNIT_STRIDE + i;
                    let b_horiz = {
                        let center = sum[idx] as i32;
                        let left = sum[idx - 1] as i32;
                        let right = sum[idx + 1] as i32;
                        center * 6 + (left + right) * 5
                    };
                    let a_horiz = {
                        let center = sumsq[idx];
                        let left = sumsq[idx - 1];
                        let right = sumsq[idx + 1];
                        center * 6 + (left + right) * 5
                    };
                    let src_val = src[src_base + (j + 1) * REST_UNIT_STRIDE + i] as i32;
                    dst[(j + 1) * MAX_RESTORATION_WIDTH + i] =
                        ((a_horiz - b_horiz * src_val + (1 << 7)) >> 8) as i16;
                    i += 1;
                }
            }
            j += 2;
        }
        // Handle last row if height is odd
        if j < h {
            for i in 0..w {
                let idx = base + j * REST_UNIT_STRIDE + i;
                let b_six = {
                    let above = sum[idx - REST_UNIT_STRIDE] as i32;
                    let below = sum[idx + REST_UNIT_STRIDE] as i32;
                    let al = sum[idx - REST_UNIT_STRIDE - 1] as i32;
                    let ar = sum[idx - REST_UNIT_STRIDE + 1] as i32;
                    let bl = sum[idx + REST_UNIT_STRIDE - 1] as i32;
                    let br = sum[idx + REST_UNIT_STRIDE + 1] as i32;
                    (above + below) * 6 + (al + ar + bl + br) * 5
                };
                let a_six = {
                    let above = sumsq[idx - REST_UNIT_STRIDE];
                    let below = sumsq[idx + REST_UNIT_STRIDE];
                    let al = sumsq[idx - REST_UNIT_STRIDE - 1];
                    let ar = sumsq[idx - REST_UNIT_STRIDE + 1];
                    let bl = sumsq[idx + REST_UNIT_STRIDE - 1];
                    let br = sumsq[idx + REST_UNIT_STRIDE + 1];
                    (above + below) * 6 + (al + ar + bl + br) * 5
                };
                let src_val = src[src_base + j * REST_UNIT_STRIDE + i] as i32;
                dst[j * MAX_RESTORATION_WIDTH + i] =
                    ((a_six - b_six * src_val + (1 << 8)) >> 9) as i16;
            }
        }
    } else {
        // 3x3: eight_neighbors
        for j in 0..h {
            let mut i = 0usize;
            while i + 16 <= w {
                let idx = base + j * REST_UNIT_STRIDE + i;

                // 9 neighbors for sum (i16 → i32)
                let s_c = _mm512_cvtepi16_epi32(loadu_256!(&sum[idx..idx + 16], [i16; 16]));
                let s_l =
                    _mm512_cvtepi16_epi32(loadu_256!(&sum[idx - 1..idx - 1 + 16], [i16; 16]));
                let s_r =
                    _mm512_cvtepi16_epi32(loadu_256!(&sum[idx + 1..idx + 1 + 16], [i16; 16]));
                let s_a = _mm512_cvtepi16_epi32(loadu_256!(
                    &sum[idx - REST_UNIT_STRIDE..idx - REST_UNIT_STRIDE + 16],
                    [i16; 16]
                ));
                let s_b = _mm512_cvtepi16_epi32(loadu_256!(
                    &sum[idx + REST_UNIT_STRIDE..idx + REST_UNIT_STRIDE + 16],
                    [i16; 16]
                ));
                let s_al = _mm512_cvtepi16_epi32(loadu_256!(
                    &sum[idx - REST_UNIT_STRIDE - 1..idx - REST_UNIT_STRIDE - 1 + 16],
                    [i16; 16]
                ));
                let s_ar = _mm512_cvtepi16_epi32(loadu_256!(
                    &sum[idx - REST_UNIT_STRIDE + 1..idx - REST_UNIT_STRIDE + 1 + 16],
                    [i16; 16]
                ));
                let s_bl = _mm512_cvtepi16_epi32(loadu_256!(
                    &sum[idx + REST_UNIT_STRIDE - 1..idx + REST_UNIT_STRIDE - 1 + 16],
                    [i16; 16]
                ));
                let s_br = _mm512_cvtepi16_epi32(loadu_256!(
                    &sum[idx + REST_UNIT_STRIDE + 1..idx + REST_UNIT_STRIDE + 1 + 16],
                    [i16; 16]
                ));

                let b_eight = _mm512_add_epi32(
                    _mm512_mullo_epi32(
                        _mm512_add_epi32(
                            _mm512_add_epi32(s_c, _mm512_add_epi32(s_l, s_r)),
                            _mm512_add_epi32(s_a, s_b),
                        ),
                        four,
                    ),
                    _mm512_mullo_epi32(
                        _mm512_add_epi32(
                            _mm512_add_epi32(s_al, s_ar),
                            _mm512_add_epi32(s_bl, s_br),
                        ),
                        three,
                    ),
                );

                // 9 neighbors for sumsq (already i32)
                let q_c = loadu_512!(&sumsq[idx..idx + 16], [i32; 16]);
                let q_l = loadu_512!(&sumsq[idx - 1..idx - 1 + 16], [i32; 16]);
                let q_r = loadu_512!(&sumsq[idx + 1..idx + 1 + 16], [i32; 16]);
                let q_a = loadu_512!(
                    &sumsq[idx - REST_UNIT_STRIDE..idx - REST_UNIT_STRIDE + 16],
                    [i32; 16]
                );
                let q_b = loadu_512!(
                    &sumsq[idx + REST_UNIT_STRIDE..idx + REST_UNIT_STRIDE + 16],
                    [i32; 16]
                );
                let q_al = loadu_512!(
                    &sumsq[idx - REST_UNIT_STRIDE - 1..idx - REST_UNIT_STRIDE - 1 + 16],
                    [i32; 16]
                );
                let q_ar = loadu_512!(
                    &sumsq[idx - REST_UNIT_STRIDE + 1..idx - REST_UNIT_STRIDE + 1 + 16],
                    [i32; 16]
                );
                let q_bl = loadu_512!(
                    &sumsq[idx + REST_UNIT_STRIDE - 1..idx + REST_UNIT_STRIDE - 1 + 16],
                    [i32; 16]
                );
                let q_br = loadu_512!(
                    &sumsq[idx + REST_UNIT_STRIDE + 1..idx + REST_UNIT_STRIDE + 1 + 16],
                    [i32; 16]
                );

                let a_eight = _mm512_add_epi32(
                    _mm512_mullo_epi32(
                        _mm512_add_epi32(
                            _mm512_add_epi32(q_c, _mm512_add_epi32(q_l, q_r)),
                            _mm512_add_epi32(q_a, q_b),
                        ),
                        four,
                    ),
                    _mm512_mullo_epi32(
                        _mm512_add_epi32(
                            _mm512_add_epi32(q_al, q_ar),
                            _mm512_add_epi32(q_bl, q_br),
                        ),
                        three,
                    ),
                );

                let src_bytes = loadu_128!(
                    &src[src_base + j * REST_UNIT_STRIDE + i
                        ..src_base + j * REST_UNIT_STRIDE + i + 16],
                    [u8; 16]
                );
                let src_val = _mm512_cvtepu8_epi32(src_bytes);

                let result = _mm512_srai_epi32::<9>(_mm512_add_epi32(
                    _mm512_sub_epi32(a_eight, _mm512_mullo_epi32(b_eight, src_val)),
                    rounding_9,
                ));

                let result_16 = _mm512_cvtsepi32_epi16(result);
                storeu_256!(
                    &mut dst[j * MAX_RESTORATION_WIDTH + i..j * MAX_RESTORATION_WIDTH + i + 16],
                    [i16; 16],
                    result_16
                );

                i += 16;
            }
            // Scalar tail for 3x3
            while i < w {
                let idx = base + j * REST_UNIT_STRIDE + i;
                let b_eight = {
                    let center = sum[idx] as i32;
                    let left = sum[idx - 1] as i32;
                    let right = sum[idx + 1] as i32;
                    let above = sum[idx - REST_UNIT_STRIDE] as i32;
                    let below = sum[idx + REST_UNIT_STRIDE] as i32;
                    let al = sum[idx - REST_UNIT_STRIDE - 1] as i32;
                    let ar = sum[idx - REST_UNIT_STRIDE + 1] as i32;
                    let bl = sum[idx + REST_UNIT_STRIDE - 1] as i32;
                    let br = sum[idx + REST_UNIT_STRIDE + 1] as i32;
                    (center + left + right + above + below) * 4 + (al + ar + bl + br) * 3
                };
                let a_eight = {
                    let center = sumsq[idx];
                    let left = sumsq[idx - 1];
                    let right = sumsq[idx + 1];
                    let above = sumsq[idx - REST_UNIT_STRIDE];
                    let below = sumsq[idx + REST_UNIT_STRIDE];
                    let al = sumsq[idx - REST_UNIT_STRIDE - 1];
                    let ar = sumsq[idx - REST_UNIT_STRIDE + 1];
                    let bl = sumsq[idx + REST_UNIT_STRIDE - 1];
                    let br = sumsq[idx + REST_UNIT_STRIDE + 1];
                    (center + left + right + above + below) * 4 + (al + ar + bl + br) * 3
                };
                let src_val = src[src_base + j * REST_UNIT_STRIDE + i] as i32;
                dst[j * MAX_RESTORATION_WIDTH + i] =
                    ((a_eight - b_eight * src_val + (1 << 8)) >> 9) as i16;
                i += 1;
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
    #[cfg(target_arch = "x86_64")]
    if let Some(token) = crate::src::cpu::summon_avx512() {
        selfguided_filter_8bpc_avx512(token, &mut dst, &tmp, w, h, 25, sgr.s0);
    } else if let Some(token) = summon_avx2() {
        selfguided_filter_8bpc_avx2(token, &mut dst, &tmp, w, h, 25, sgr.s0);
    } else {
        selfguided_filter_8bpc(&mut dst, &tmp, w, h, 25, sgr.s0);
    }
    #[cfg(not(target_arch = "x86_64"))]
    selfguided_filter_8bpc(&mut dst, &tmp, w, h, 25, sgr.s0);

    let w0 = sgr.w0 as i32;
    let stride = p.pixel_stride::<BitDepth8>();

    let (mut p_guard, p_base) = p.strided_slice_mut::<BitDepth8>(w, h);
    for j in 0..h {
        let row_off = p_base.wrapping_add_signed(j as isize * stride);
        for i in 0..w {
            let v = w0 * dst[j * MAX_RESTORATION_WIDTH + i] as i32;
            p_guard[row_off + i] = iclip(
                p_guard[row_off + i] as i32 + ((v + (1 << 10)) >> 11),
                0,
                255,
            ) as u8;
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
    #[cfg(target_arch = "x86_64")]
    if let Some(token) = crate::src::cpu::summon_avx512() {
        selfguided_filter_8bpc_avx512(token, &mut dst, &tmp, w, h, 9, sgr.s1);
    } else if let Some(token) = summon_avx2() {
        selfguided_filter_8bpc_avx2(token, &mut dst, &tmp, w, h, 9, sgr.s1);
    } else {
        selfguided_filter_8bpc(&mut dst, &tmp, w, h, 9, sgr.s1);
    }
    #[cfg(not(target_arch = "x86_64"))]
    selfguided_filter_8bpc(&mut dst, &tmp, w, h, 9, sgr.s1);

    let w1 = sgr.w1 as i32;
    let stride = p.pixel_stride::<BitDepth8>();

    let (mut p_guard, p_base) = p.strided_slice_mut::<BitDepth8>(w, h);
    for j in 0..h {
        let row_off = p_base.wrapping_add_signed(j as isize * stride);
        for i in 0..w {
            let v = w1 * dst[j * MAX_RESTORATION_WIDTH + i] as i32;
            p_guard[row_off + i] = iclip(
                p_guard[row_off + i] as i32 + ((v + (1 << 10)) >> 11),
                0,
                255,
            ) as u8;
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
    #[cfg(target_arch = "x86_64")]
    if let Some(token) = crate::src::cpu::summon_avx512() {
        selfguided_filter_8bpc_avx512(token, &mut dst0, &tmp, w, h, 25, sgr.s0);
        selfguided_filter_8bpc_avx512(token, &mut dst1, &tmp, w, h, 9, sgr.s1);
    } else if let Some(token) = summon_avx2() {
        selfguided_filter_8bpc_avx2(token, &mut dst0, &tmp, w, h, 25, sgr.s0);
        selfguided_filter_8bpc_avx2(token, &mut dst1, &tmp, w, h, 9, sgr.s1);
    } else {
        selfguided_filter_8bpc(&mut dst0, &tmp, w, h, 25, sgr.s0);
        selfguided_filter_8bpc(&mut dst1, &tmp, w, h, 9, sgr.s1);
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        selfguided_filter_8bpc(&mut dst0, &tmp, w, h, 25, sgr.s0);
        selfguided_filter_8bpc(&mut dst1, &tmp, w, h, 9, sgr.s1);
    }

    let w0 = sgr.w0 as i32;
    let w1 = sgr.w1 as i32;
    let stride = p.pixel_stride::<BitDepth8>();

    let (mut p_guard, p_base) = p.strided_slice_mut::<BitDepth8>(w, h);
    for j in 0..h {
        let row_off = p_base.wrapping_add_signed(j as isize * stride);
        for i in 0..w {
            let v = w0 * dst0[j * MAX_RESTORATION_WIDTH + i] as i32
                + w1 * dst1[j * MAX_RESTORATION_WIDTH + i] as i32;
            p_guard[row_off + i] = iclip(
                p_guard[row_off + i] as i32 + ((v + (1 << 10)) >> 11),
                0,
                255,
            ) as u8;
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
    // Scalar reference starts cursor aa at STRIDE+3, inner loop i from -1..w+1.
    // So first access is at STRIDE+2. We use aa_base = (row_offset+1)*STRIDE+2
    // with inner loop i from 0..w+2, giving the same index range.

    for row_offset in (0..(h + 2)).step_by(step) {
        let aa_base = (row_offset + 1) * REST_UNIT_STRIDE + 2;

        for i in 0..(w + 2) {
            let idx = aa_base + i;
            // Scale down by bitdepth_min_8 for the variance calculation
            let a_val = sumsq[idx];
            let b_val = sum[idx] as i64;

            // Apply bitdepth scaling: a >> (2 * bitdepth_min_8), b >> bitdepth_min_8
            let a_scaled =
                ((a_val + (1 << (2 * bitdepth_min_8 - 1))) >> (2 * bitdepth_min_8)) as i32;
            let b_scaled = ((b_val + (1 << (bitdepth_min_8 - 1))) >> bitdepth_min_8) as i32;

            let p = cmp::max(a_scaled * n - b_scaled * b_scaled, 0) as u32;
            let z = (p * s + (1 << 19)) >> 20;
            let x = dav1d_sgr_x_by_x[cmp::min(z, 255) as usize] as u32;

            // Store: aa = x * b * sgr_one_by_x, bb = x
            // Use original b_val (not scaled) for the multiplication
            aa[idx] = ((x * (b_val as u32) * sgr_one_by_x + (1 << 11)) >> 12) as i32;
            bb[idx] = x as i32;
        }
    }

    // Apply neighbor-weighted filter to produce output
    let base = 2 * REST_UNIT_STRIDE + 3; // matches scalar cursor a/b starting position
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
                    let above = bb[idx - REST_UNIT_STRIDE] as i64;
                    let below = bb[idx + REST_UNIT_STRIDE] as i64;
                    let above_left = bb[idx - REST_UNIT_STRIDE - 1] as i64;
                    let above_right = bb[idx - REST_UNIT_STRIDE + 1] as i64;
                    let below_left = bb[idx + REST_UNIT_STRIDE - 1] as i64;
                    let below_right = bb[idx + REST_UNIT_STRIDE + 1] as i64;
                    (above + below) * 6 + (above_left + above_right + below_left + below_right) * 5
                };
                // six_neighbors for a (aa array)
                let a_six = {
                    let above = aa[idx - REST_UNIT_STRIDE] as i64;
                    let below = aa[idx + REST_UNIT_STRIDE] as i64;
                    let above_left = aa[idx - REST_UNIT_STRIDE - 1] as i64;
                    let above_right = aa[idx - REST_UNIT_STRIDE + 1] as i64;
                    let below_left = aa[idx + REST_UNIT_STRIDE - 1] as i64;
                    let below_right = aa[idx + REST_UNIT_STRIDE + 1] as i64;
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
                        let center = bb[idx] as i64;
                        let left = bb[idx - 1] as i64;
                        let right = bb[idx + 1] as i64;
                        center * 6 + (left + right) * 5
                    };
                    let a_horiz = {
                        let center = aa[idx] as i64;
                        let left = aa[idx - 1] as i64;
                        let right = aa[idx + 1] as i64;
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
                    let above = bb[idx - REST_UNIT_STRIDE] as i64;
                    let below = bb[idx + REST_UNIT_STRIDE] as i64;
                    let above_left = bb[idx - REST_UNIT_STRIDE - 1] as i64;
                    let above_right = bb[idx - REST_UNIT_STRIDE + 1] as i64;
                    let below_left = bb[idx + REST_UNIT_STRIDE - 1] as i64;
                    let below_right = bb[idx + REST_UNIT_STRIDE + 1] as i64;
                    (above + below) * 6 + (above_left + above_right + below_left + below_right) * 5
                };
                let a_six = {
                    let above = aa[idx - REST_UNIT_STRIDE] as i64;
                    let below = aa[idx + REST_UNIT_STRIDE] as i64;
                    let above_left = aa[idx - REST_UNIT_STRIDE - 1] as i64;
                    let above_right = aa[idx - REST_UNIT_STRIDE + 1] as i64;
                    let below_left = aa[idx + REST_UNIT_STRIDE - 1] as i64;
                    let below_right = aa[idx + REST_UNIT_STRIDE + 1] as i64;
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
                    let center = bb[idx] as i64;
                    let left = bb[idx - 1] as i64;
                    let right = bb[idx + 1] as i64;
                    let above = bb[idx - REST_UNIT_STRIDE] as i64;
                    let below = bb[idx + REST_UNIT_STRIDE] as i64;
                    let above_left = bb[idx - REST_UNIT_STRIDE - 1] as i64;
                    let above_right = bb[idx - REST_UNIT_STRIDE + 1] as i64;
                    let below_left = bb[idx + REST_UNIT_STRIDE - 1] as i64;
                    let below_right = bb[idx + REST_UNIT_STRIDE + 1] as i64;
                    (center + left + right + above + below) * 4
                        + (above_left + above_right + below_left + below_right) * 3
                };
                // eight_neighbors for a
                let a_eight = {
                    let center = aa[idx] as i64;
                    let left = aa[idx - 1] as i64;
                    let right = aa[idx + 1] as i64;
                    let above = aa[idx - REST_UNIT_STRIDE] as i64;
                    let below = aa[idx + REST_UNIT_STRIDE] as i64;
                    let above_left = aa[idx - REST_UNIT_STRIDE - 1] as i64;
                    let above_right = aa[idx - REST_UNIT_STRIDE + 1] as i64;
                    let below_left = aa[idx + REST_UNIT_STRIDE - 1] as i64;
                    let below_right = aa[idx + REST_UNIT_STRIDE + 1] as i64;
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

    let (mut p_guard, p_base) = p.strided_slice_mut::<BitDepth16>(w, h);
    for j in 0..h {
        let row_off = p_base.wrapping_add_signed(j as isize * stride);
        for i in 0..w {
            let v = w0 * dst[j * MAX_RESTORATION_WIDTH + i];
            p_guard[row_off + i] = iclip(
                p_guard[row_off + i] as i32 + ((v + (1 << 10)) >> 11),
                0,
                bitdepth_max,
            ) as u16;
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

    let (mut p_guard, p_base) = p.strided_slice_mut::<BitDepth16>(w, h);
    for j in 0..h {
        let row_off = p_base.wrapping_add_signed(j as isize * stride);
        for i in 0..w {
            let v = w1 * dst[j * MAX_RESTORATION_WIDTH + i];
            p_guard[row_off + i] = iclip(
                p_guard[row_off + i] as i32 + ((v + (1 << 10)) >> 11),
                0,
                bitdepth_max,
            ) as u16;
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

    let (mut p_guard, p_base) = p.strided_slice_mut::<BitDepth16>(w, h);
    for j in 0..h {
        let row_off = p_base.wrapping_add_signed(j as isize * stride);
        for i in 0..w {
            let v =
                w0 * dst0[j * MAX_RESTORATION_WIDTH + i] + w1 * dst1[j * MAX_RESTORATION_WIDTH + i];
            p_guard[row_off + i] = iclip(
                p_guard[row_off + i] as i32 + ((v + (1 << 10)) >> 11),
                0,
                bitdepth_max,
            ) as u16;
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

    let Some(token) = crate::src::cpu::summon_avx2() else {
        return false;
    };

    let w = w as usize;
    let h = h as usize;
    use crate::src::safe_simd::pixel_access::reinterpret_slice;
    let left_8 =
        || -> &[LeftPixelRow<u8>] { reinterpret_slice(left).expect("BD::Pixel layout matches u8") };
    let left_16 = || -> &[LeftPixelRow<u16>] {
        reinterpret_slice(left).expect("BD::Pixel layout matches u16")
    };

    match (BD::BPC, variant) {
        (BPC::BPC8, 0) => {
            wiener_filter7_8bpc_avx2_inner(token, dst, left_8(), lpf, lpf_off, w, h, params, edges)
        }
        (BPC::BPC8, 1) => {
            wiener_filter5_8bpc_avx2_inner(token, dst, left_8(), lpf, lpf_off, w, h, params, edges)
        }
        (BPC::BPC8, 2) => sgr_5x5_8bpc_avx2_inner(dst, left_8(), lpf, lpf_off, w, h, params, edges),
        (BPC::BPC8, 3) => sgr_3x3_8bpc_avx2_inner(dst, left_8(), lpf, lpf_off, w, h, params, edges),
        (BPC::BPC8, _) => sgr_mix_8bpc_avx2_inner(dst, left_8(), lpf, lpf_off, w, h, params, edges),
        (BPC::BPC16, 0) => wiener_filter7_16bpc_avx2_inner(
            dst,
            left_16(),
            lpf,
            lpf_off,
            w,
            h,
            params,
            edges,
            bd.into_c(),
        ),
        (BPC::BPC16, 1) => wiener_filter5_16bpc_avx2_inner(
            dst,
            left_16(),
            lpf,
            lpf_off,
            w,
            h,
            params,
            edges,
            bd.into_c(),
        ),
        (BPC::BPC16, 2) => sgr_5x5_16bpc_avx2_inner(
            dst,
            left_16(),
            lpf,
            lpf_off,
            w,
            h,
            params,
            edges,
            bd.into_c(),
        ),
        (BPC::BPC16, 3) => sgr_3x3_16bpc_avx2_inner(
            dst,
            left_16(),
            lpf,
            lpf_off,
            w,
            h,
            params,
            edges,
            bd.into_c(),
        ),
        (BPC::BPC16, _) => sgr_mix_16bpc_avx2_inner(
            dst,
            left_16(),
            lpf,
            lpf_off,
            w,
            h,
            params,
            edges,
            bd.into_c(),
        ),
    }
    true
}
