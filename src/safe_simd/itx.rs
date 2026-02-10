//! Safe SIMD implementations for ITX (Inverse Transforms)
#![allow(deprecated)] // FFI wrappers need to forge tokens
#![cfg_attr(not(feature = "unchecked"), forbid(unsafe_code))]
#![cfg_attr(feature = "unchecked", deny(unsafe_code))]
//!
//! ITX is the largest DSP module (~42k asm lines). Strategy:
//! 1. Implement full 2D transforms (not just 1D) for common sizes
//! 2. Process multiple rows/columns in parallel
//! 3. Use in-register transposition
//!
//! Most common transforms: DCT_DCT 4x4, 8x8, 16x16

#![allow(unused_imports)]
#![allow(dead_code)]

use archmage::{arcane, Desktop64, SimdToken};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::include::common::bitdepth::AsPrimitive;
use crate::include::common::bitdepth::BitDepth;
use crate::include::common::bitdepth::DynCoef;
use crate::include::common::bitdepth::DynPixel;
use crate::include::common::intops::iclip;
use crate::include::dav1d::picture::PicOffset;
use crate::src::ffi_safe::FFISafe;
use crate::src::safe_simd::pixel_access::Flex;
use crate::src::safe_simd::pixel_access::{
    loadi32, loadi64, loadu_128, loadu_256, storei32, storei64, storeu_128, storeu_256,
};
use std::ffi::c_int;
use std::num::NonZeroUsize;
use std::slice;

// ============================================================================
// CONSTANTS
// ============================================================================

// Trig coefficients for DCT (12-bit fixed point, scaled by 4096)
// cos(π/8) * 4096 ≈ 3784
// sin(π/8) * 4096 ≈ 1567
// cos(π/4) * sqrt(2) = sqrt(2) ≈ 181/128 = 1.414

const SQRT2_BITS: i32 = 8;
const SQRT2_HALF: i32 = 181; // sqrt(2) * 128

// ============================================================================
// 4x4 DCT_DCT - Full 2D SIMD Transform (8bpc)
// ============================================================================

/// Full 2D DCT_DCT 4x4 inverse transform with add-to-destination
/// Uses AVX2 to process all 4 rows simultaneously
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_4x4_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    use crate::src::safe_simd::pixel_access::{loadi32, storei32, storeu_128};
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();

    let row0 = _mm_set_epi32(
        coeff[12] as i32,
        coeff[8] as i32,
        coeff[4] as i32,
        coeff[0] as i32,
    );
    let row1 = _mm_set_epi32(
        coeff[13] as i32,
        coeff[9] as i32,
        coeff[5] as i32,
        coeff[1] as i32,
    );
    let row2 = _mm_set_epi32(
        coeff[14] as i32,
        coeff[10] as i32,
        coeff[6] as i32,
        coeff[2] as i32,
    );
    let row3 = _mm_set_epi32(
        coeff[15] as i32,
        coeff[11] as i32,
        coeff[7] as i32,
        coeff[3] as i32,
    );

    let rows01 = _mm256_set_m128i(row1, row0);
    let rows23 = _mm256_set_m128i(row3, row2);

    // 8bpc clip ranges: i16::MIN..i16::MAX for both row and col
    let row_clip_min = if bitdepth_max == 255 { i16::MIN as i32 } else { (!bitdepth_max) << 7 };
    let row_clip_max = !row_clip_min;
    let col_clip_min = if bitdepth_max == 255 { i16::MIN as i32 } else { (!bitdepth_max) << 5 };
    let col_clip_max = !col_clip_min;

    let (rows01_out, rows23_out) = dct4_2rows_avx2(_token, rows01, rows23, row_clip_min, row_clip_max);

    let r0 = _mm256_castsi256_si128(rows01_out);
    let r1 = _mm256_extracti128_si256(rows01_out, 1);
    let r2 = _mm256_castsi256_si128(rows23_out);
    let r3 = _mm256_extracti128_si256(rows23_out, 1);

    let t01_lo = _mm_unpacklo_epi32(r0, r1);
    let t01_hi = _mm_unpackhi_epi32(r0, r1);
    let t23_lo = _mm_unpacklo_epi32(r2, r3);
    let t23_hi = _mm_unpackhi_epi32(r2, r3);

    let col0 = _mm_unpacklo_epi64(t01_lo, t23_lo);
    let col1 = _mm_unpackhi_epi64(t01_lo, t23_lo);
    let col2 = _mm_unpacklo_epi64(t01_hi, t23_hi);
    let col3 = _mm_unpackhi_epi64(t01_hi, t23_hi);

    // Intermediate clamp (shift=0 for 4x4, so just clamp to col_clip range)
    let cmin = _mm_set1_epi32(col_clip_min);
    let cmax = _mm_set1_epi32(col_clip_max);
    let col0 = _mm_max_epi32(_mm_min_epi32(col0, cmax), cmin);
    let col1 = _mm_max_epi32(_mm_min_epi32(col1, cmax), cmin);
    let col2 = _mm_max_epi32(_mm_min_epi32(col2, cmax), cmin);
    let col3 = _mm_max_epi32(_mm_min_epi32(col3, cmax), cmin);

    let cols01 = _mm256_set_m128i(col1, col0);
    let cols23 = _mm256_set_m128i(col3, col2);

    let (cols01_out, cols23_out) = dct4_2rows_avx2(_token, cols01, cols23, col_clip_min, col_clip_max);

    let rnd = _mm256_set1_epi32(8);
    let cols01_scaled = _mm256_srai_epi32(_mm256_add_epi32(cols01_out, rnd), 4);
    let cols23_scaled = _mm256_srai_epi32(_mm256_add_epi32(cols23_out, rnd), 4);

    let c0 = _mm256_castsi256_si128(cols01_scaled);
    let c1 = _mm256_extracti128_si256(cols01_scaled, 1);
    let c2 = _mm256_castsi256_si128(cols23_scaled);
    let c3 = _mm256_extracti128_si256(cols23_scaled, 1);

    let u01_lo = _mm_unpacklo_epi32(c0, c1);
    let u01_hi = _mm_unpackhi_epi32(c0, c1);
    let u23_lo = _mm_unpacklo_epi32(c2, c3);
    let u23_hi = _mm_unpackhi_epi32(c2, c3);

    let final0 = _mm_unpacklo_epi64(u01_lo, u23_lo);
    let final1 = _mm_unpackhi_epi64(u01_lo, u23_lo);
    let final2 = _mm_unpacklo_epi64(u01_hi, u23_hi);
    let final3 = _mm_unpackhi_epi64(u01_hi, u23_hi);

    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi16(bitdepth_max as i16);

    // Row 0
    let d0 = loadi32!(&dst[..4]);
    let d0_16 = _mm_unpacklo_epi8(d0, zero);
    let d0_32 = _mm_cvtepi16_epi32(d0_16);
    let sum0 = _mm_add_epi32(d0_32, final0);
    let sum0_16 = _mm_packs_epi32(sum0, sum0);
    let sum0_clamped = _mm_max_epi16(_mm_min_epi16(sum0_16, max_val), zero);
    let sum0_8 = _mm_packus_epi16(sum0_clamped, sum0_clamped);
    storei32!(&mut dst[..4], sum0_8);

    // Row 1
    let off1 = dst_stride;
    let d1 = loadi32!(&dst[off1..off1 + 4]);
    let d1_16 = _mm_unpacklo_epi8(d1, zero);
    let d1_32 = _mm_cvtepi16_epi32(d1_16);
    let sum1 = _mm_add_epi32(d1_32, final1);
    let sum1_16 = _mm_packs_epi32(sum1, sum1);
    let sum1_clamped = _mm_max_epi16(_mm_min_epi16(sum1_16, max_val), zero);
    let sum1_8 = _mm_packus_epi16(sum1_clamped, sum1_clamped);
    storei32!(&mut dst[off1..off1 + 4], sum1_8);

    // Row 2
    let off2 = dst_stride * 2;
    let d2 = loadi32!(&dst[off2..off2 + 4]);
    let d2_16 = _mm_unpacklo_epi8(d2, zero);
    let d2_32 = _mm_cvtepi16_epi32(d2_16);
    let sum2 = _mm_add_epi32(d2_32, final2);
    let sum2_16 = _mm_packs_epi32(sum2, sum2);
    let sum2_clamped = _mm_max_epi16(_mm_min_epi16(sum2_16, max_val), zero);
    let sum2_8 = _mm_packus_epi16(sum2_clamped, sum2_clamped);
    storei32!(&mut dst[off2..off2 + 4], sum2_8);

    // Row 3
    let off3 = dst_stride * 3;
    let d3 = loadi32!(&dst[off3..off3 + 4]);
    let d3_16 = _mm_unpacklo_epi8(d3, zero);
    let d3_32 = _mm_cvtepi16_epi32(d3_16);
    let sum3 = _mm_add_epi32(d3_32, final3);
    let sum3_16 = _mm_packs_epi32(sum3, sum3);
    let sum3_clamped = _mm_max_epi16(_mm_min_epi16(sum3_16, max_val), zero);
    let sum3_8 = _mm_packus_epi16(sum3_clamped, sum3_clamped);
    storei32!(&mut dst[off3..off3 + 4], sum3_8);

    // Clear coefficients
    let coeff_bytes = zerocopy::AsBytes::as_bytes_mut(&mut *coeff);
    storeu_128!(
        <&mut [u8; 16]>::try_from(&mut coeff_bytes[..16]).unwrap(),
        _mm_setzero_si128()
    );
    storeu_128!(
        <&mut [u8; 16]>::try_from(&mut coeff_bytes[16..32]).unwrap(),
        _mm_setzero_si128()
    );
}

/// DCT4 butterfly on 2 rows packed in __m256i
/// Each 128-bit lane contains one row: [in0, in1, in2, in3] as i32
/// Outputs are clipped to [clip_min, clip_max] to match scalar behavior.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn dct4_2rows_avx2(
    _token: Desktop64,
    rows01: __m256i,
    rows23: __m256i,
    clip_min: i32,
    clip_max: i32,
) -> (__m256i, __m256i) {
    // DCT4: t0 = (in0 + in2) * 181 + 128 >> 8
    //       t1 = (in0 - in2) * 181 + 128 >> 8
    //       t2 = (in1 * 1567 - in3 * (3784-4096) + 2048 >> 12) - in3
    //       t3 = (in1 * (3784-4096) + in3 * 1567 + 2048 >> 12) + in1

    let sqrt2 = _mm256_set1_epi32(181);
    let rnd8 = _mm256_set1_epi32(128);
    let c1567 = _mm256_set1_epi32(1567);
    let c_312 = _mm256_set1_epi32(3784 - 4096);
    let rnd12 = _mm256_set1_epi32(2048);

    // Process rows01
    let in0_01 = _mm256_shuffle_epi32(rows01, 0b00_00_00_00);
    let in1_01 = _mm256_shuffle_epi32(rows01, 0b01_01_01_01);
    let in2_01 = _mm256_shuffle_epi32(rows01, 0b10_10_10_10);
    let in3_01 = _mm256_shuffle_epi32(rows01, 0b11_11_11_11);

    // t0 = (in0 + in2) * 181 + 128 >> 8
    let sum02_01 = _mm256_add_epi32(in0_01, in2_01);
    let t0_01 = _mm256_srai_epi32(
        _mm256_add_epi32(_mm256_mullo_epi32(sum02_01, sqrt2), rnd8),
        8,
    );

    // t1 = (in0 - in2) * 181 + 128 >> 8
    let diff02_01 = _mm256_sub_epi32(in0_01, in2_01);
    let t1_01 = _mm256_srai_epi32(
        _mm256_add_epi32(_mm256_mullo_epi32(diff02_01, sqrt2), rnd8),
        8,
    );

    // t2 = (in1 * 1567 - in3 * (3784-4096) + 2048 >> 12) - in3
    let mul1_1567_01 = _mm256_mullo_epi32(in1_01, c1567);
    let mul3_312_01 = _mm256_mullo_epi32(in3_01, c_312);
    let t2_inner_01 = _mm256_srai_epi32(
        _mm256_add_epi32(_mm256_sub_epi32(mul1_1567_01, mul3_312_01), rnd12),
        12,
    );
    let t2_01 = _mm256_sub_epi32(t2_inner_01, in3_01);

    // t3 = (in1 * (3784-4096) + in3 * 1567 + 2048 >> 12) + in1
    let mul1_312_01 = _mm256_mullo_epi32(in1_01, c_312);
    let mul3_1567_01 = _mm256_mullo_epi32(in3_01, c1567);
    let t3_inner_01 = _mm256_srai_epi32(
        _mm256_add_epi32(_mm256_add_epi32(mul1_312_01, mul3_1567_01), rnd12),
        12,
    );
    let t3_01 = _mm256_add_epi32(t3_inner_01, in1_01);

    // Output: out0 = clip(t0+t3), out1 = clip(t1+t2), out2 = clip(t1-t2), out3 = clip(t0-t3)
    let vmin = _mm256_set1_epi32(clip_min);
    let vmax = _mm256_set1_epi32(clip_max);
    let out0_01 = _mm256_max_epi32(_mm256_min_epi32(_mm256_add_epi32(t0_01, t3_01), vmax), vmin);
    let out1_01 = _mm256_max_epi32(_mm256_min_epi32(_mm256_add_epi32(t1_01, t2_01), vmax), vmin);
    let out2_01 = _mm256_max_epi32(_mm256_min_epi32(_mm256_sub_epi32(t1_01, t2_01), vmax), vmin);
    let out3_01 = _mm256_max_epi32(_mm256_min_epi32(_mm256_sub_epi32(t0_01, t3_01), vmax), vmin);

    // Interleave outputs back: [out0, out1, out2, out3] per lane
    let mask0 = _mm256_set_epi32(0, 0, 0, -1i32, 0, 0, 0, -1i32);
    let mask1 = _mm256_set_epi32(0, 0, -1i32, 0, 0, 0, -1i32, 0);
    let mask2 = _mm256_set_epi32(0, -1i32, 0, 0, 0, -1i32, 0, 0);
    let mask3 = _mm256_set_epi32(-1i32, 0, 0, 0, -1i32, 0, 0, 0);

    let rows01_out = _mm256_or_si256(
        _mm256_or_si256(
            _mm256_and_si256(out0_01, mask0),
            _mm256_and_si256(_mm256_shuffle_epi32(out1_01, 0b00_00_00_01), mask1),
        ),
        _mm256_or_si256(
            _mm256_and_si256(_mm256_shuffle_epi32(out2_01, 0b00_00_10_00), mask2),
            _mm256_and_si256(_mm256_shuffle_epi32(out3_01, 0b00_11_00_00), mask3),
        ),
    );

    // Same for rows23
    let in0_23 = _mm256_shuffle_epi32(rows23, 0b00_00_00_00);
    let in1_23 = _mm256_shuffle_epi32(rows23, 0b01_01_01_01);
    let in2_23 = _mm256_shuffle_epi32(rows23, 0b10_10_10_10);
    let in3_23 = _mm256_shuffle_epi32(rows23, 0b11_11_11_11);

    let sum02_23 = _mm256_add_epi32(in0_23, in2_23);
    let t0_23 = _mm256_srai_epi32(
        _mm256_add_epi32(_mm256_mullo_epi32(sum02_23, sqrt2), rnd8),
        8,
    );

    let diff02_23 = _mm256_sub_epi32(in0_23, in2_23);
    let t1_23 = _mm256_srai_epi32(
        _mm256_add_epi32(_mm256_mullo_epi32(diff02_23, sqrt2), rnd8),
        8,
    );

    let mul1_1567_23 = _mm256_mullo_epi32(in1_23, c1567);
    let mul3_312_23 = _mm256_mullo_epi32(in3_23, c_312);
    let t2_inner_23 = _mm256_srai_epi32(
        _mm256_add_epi32(_mm256_sub_epi32(mul1_1567_23, mul3_312_23), rnd12),
        12,
    );
    let t2_23 = _mm256_sub_epi32(t2_inner_23, in3_23);

    let mul1_312_23 = _mm256_mullo_epi32(in1_23, c_312);
    let mul3_1567_23 = _mm256_mullo_epi32(in3_23, c1567);
    let t3_inner_23 = _mm256_srai_epi32(
        _mm256_add_epi32(_mm256_add_epi32(mul1_312_23, mul3_1567_23), rnd12),
        12,
    );
    let t3_23 = _mm256_add_epi32(t3_inner_23, in1_23);

    let out0_23 = _mm256_max_epi32(_mm256_min_epi32(_mm256_add_epi32(t0_23, t3_23), vmax), vmin);
    let out1_23 = _mm256_max_epi32(_mm256_min_epi32(_mm256_add_epi32(t1_23, t2_23), vmax), vmin);
    let out2_23 = _mm256_max_epi32(_mm256_min_epi32(_mm256_sub_epi32(t1_23, t2_23), vmax), vmin);
    let out3_23 = _mm256_max_epi32(_mm256_min_epi32(_mm256_sub_epi32(t0_23, t3_23), vmax), vmin);

    let rows23_out = _mm256_or_si256(
        _mm256_or_si256(
            _mm256_and_si256(out0_23, mask0),
            _mm256_and_si256(_mm256_shuffle_epi32(out1_23, 0b00_00_00_01), mask1),
        ),
        _mm256_or_si256(
            _mm256_and_si256(_mm256_shuffle_epi32(out2_23, 0b00_00_10_00), mask2),
            _mm256_and_si256(_mm256_shuffle_epi32(out3_23, 0b00_11_00_00), mask3),
        ),
    );

    (rows01_out, rows23_out)
}

// ============================================================================
// FFI WRAPPERS
// ============================================================================

/// FFI wrapper for 4x4 DCT_DCT 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_4x4_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, _coeff_len as usize * stride + stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_4x4_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// 4x4 DCT_DCT 16bpc
// ============================================================================

/// Full 2D DCT_DCT 4x4 inverse transform with add-to-destination (16bpc)
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_4x4_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize, // stride in bytes
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    // For 16bpc, stride is in bytes but we access u16, so stride_u16 = stride / 2
    let stride_u16 = dst_stride / 2;

    // Load coefficients (column-major storage)
    let row0 = _mm_set_epi32(
        coeff[12] as i32,
        coeff[8] as i32,
        coeff[4] as i32,
        coeff[0] as i32,
    );
    let row1 = _mm_set_epi32(
        coeff[13] as i32,
        coeff[9] as i32,
        coeff[5] as i32,
        coeff[1] as i32,
    );
    let row2 = _mm_set_epi32(
        coeff[14] as i32,
        coeff[10] as i32,
        coeff[6] as i32,
        coeff[2] as i32,
    );
    let row3 = _mm_set_epi32(
        coeff[15] as i32,
        coeff[11] as i32,
        coeff[7] as i32,
        coeff[3] as i32,
    );

    // Pack rows into 256-bit vectors
    let rows01 = _mm256_set_m128i(row1, row0);
    let rows23 = _mm256_set_m128i(row3, row2);

    // 16bpc clip ranges
    let row_clip_min = (!bitdepth_max) << 7;
    let row_clip_max = !row_clip_min;
    let col_clip_min = (!bitdepth_max) << 5;
    let col_clip_max = !col_clip_min;

    // DCT4 butterfly on rows
    let (rows01_out, rows23_out) = dct4_2rows_avx2(_token, rows01, rows23, row_clip_min, row_clip_max);

    // Transpose for column pass
    let r0 = _mm256_castsi256_si128(rows01_out);
    let r1 = _mm256_extracti128_si256(rows01_out, 1);
    let r2 = _mm256_castsi256_si128(rows23_out);
    let r3 = _mm256_extracti128_si256(rows23_out, 1);

    // Transpose 4x4 using unpack
    let t01_lo = _mm_unpacklo_epi32(r0, r1);
    let t01_hi = _mm_unpackhi_epi32(r0, r1);
    let t23_lo = _mm_unpacklo_epi32(r2, r3);
    let t23_hi = _mm_unpackhi_epi32(r2, r3);

    let c0 = _mm_unpacklo_epi64(t01_lo, t23_lo);
    let c1 = _mm_unpackhi_epi64(t01_lo, t23_lo);
    let c2 = _mm_unpacklo_epi64(t01_hi, t23_hi);
    let c3 = _mm_unpackhi_epi64(t01_hi, t23_hi);

    // Intermediate clamp (shift=0 for 4x4, so just clamp to col_clip range)
    let cmin = _mm_set1_epi32(col_clip_min);
    let cmax = _mm_set1_epi32(col_clip_max);
    let c0 = _mm_max_epi32(_mm_min_epi32(c0, cmax), cmin);
    let c1 = _mm_max_epi32(_mm_min_epi32(c1, cmax), cmin);
    let c2 = _mm_max_epi32(_mm_min_epi32(c2, cmax), cmin);
    let c3 = _mm_max_epi32(_mm_min_epi32(c3, cmax), cmin);

    // DCT4 on columns
    let cols01 = _mm256_set_m128i(c1, c0);
    let cols23 = _mm256_set_m128i(c3, c2);
    let (cols01_out, cols23_out) = dct4_2rows_avx2(_token, cols01, cols23, col_clip_min, col_clip_max);

    // Extract final columns
    let col0 = _mm256_castsi256_si128(cols01_out);
    let col1 = _mm256_extracti128_si256(cols01_out, 1);
    let col2 = _mm256_castsi256_si128(cols23_out);
    let col3 = _mm256_extracti128_si256(cols23_out, 1);

    // Transpose back to rows for output
    let t01_lo = _mm_unpacklo_epi32(col0, col1);
    let t01_hi = _mm_unpackhi_epi32(col0, col1);
    let t23_lo = _mm_unpacklo_epi32(col2, col3);
    let t23_hi = _mm_unpackhi_epi32(col2, col3);

    let out0 = _mm_unpacklo_epi64(t01_lo, t23_lo);
    let out1 = _mm_unpackhi_epi64(t01_lo, t23_lo);
    let out2 = _mm_unpacklo_epi64(t01_hi, t23_hi);
    let out3 = _mm_unpackhi_epi64(t01_hi, t23_hi);

    // Add to destination: shift by 4, clamp to [0, bitdepth_max]
    let rnd = _mm_set1_epi32(8);
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi32(bitdepth_max);

    // Row 0
    let dst0 = loadi64!(zerocopy::AsBytes::as_bytes(&dst[..4]));
    let dst0_32 = _mm_unpacklo_epi16(dst0, zero);
    let scaled0 = _mm_srai_epi32(_mm_add_epi32(out0, rnd), 4);
    let sum0 = _mm_add_epi32(dst0_32, scaled0);
    let clamped0 = _mm_max_epi32(_mm_min_epi32(sum0, max_val), zero);
    let packed0 = _mm_packus_epi32(clamped0, clamped0);
    storei64!(zerocopy::AsBytes::as_bytes_mut(&mut dst[..4]), packed0);

    // Row 1
    let dst_off1 = stride_u16;
    let dst1 = loadi64!(zerocopy::AsBytes::as_bytes(&dst[dst_off1..dst_off1 + 4]));
    let dst1_32 = _mm_unpacklo_epi16(dst1, zero);
    let scaled1 = _mm_srai_epi32(_mm_add_epi32(out1, rnd), 4);
    let sum1 = _mm_add_epi32(dst1_32, scaled1);
    let clamped1 = _mm_max_epi32(_mm_min_epi32(sum1, max_val), zero);
    let packed1 = _mm_packus_epi32(clamped1, clamped1);
    storei64!(
        zerocopy::AsBytes::as_bytes_mut(&mut dst[dst_off1..dst_off1 + 4]),
        packed1
    );

    // Row 2
    let dst_off2 = stride_u16 * 2;
    let dst2 = loadi64!(zerocopy::AsBytes::as_bytes(&dst[dst_off2..dst_off2 + 4]));
    let dst2_32 = _mm_unpacklo_epi16(dst2, zero);
    let scaled2 = _mm_srai_epi32(_mm_add_epi32(out2, rnd), 4);
    let sum2 = _mm_add_epi32(dst2_32, scaled2);
    let clamped2 = _mm_max_epi32(_mm_min_epi32(sum2, max_val), zero);
    let packed2 = _mm_packus_epi32(clamped2, clamped2);
    storei64!(
        zerocopy::AsBytes::as_bytes_mut(&mut dst[dst_off2..dst_off2 + 4]),
        packed2
    );

    // Row 3
    let dst_off3 = stride_u16 * 3;
    let dst3 = loadi64!(zerocopy::AsBytes::as_bytes(&dst[dst_off3..dst_off3 + 4]));
    let dst3_32 = _mm_unpacklo_epi16(dst3, zero);
    let scaled3 = _mm_srai_epi32(_mm_add_epi32(out3, rnd), 4);
    let sum3 = _mm_add_epi32(dst3_32, scaled3);
    let clamped3 = _mm_max_epi32(_mm_min_epi32(sum3, max_val), zero);
    let packed3 = _mm_packus_epi32(clamped3, clamped3);
    storei64!(
        zerocopy::AsBytes::as_bytes_mut(&mut dst[dst_off3..dst_off3 + 4]),
        packed3
    );

    // Clear coefficients
    coeff[..16].fill(0);
}

/// FFI wrapper for 4x4 DCT_DCT 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_4x4_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_4x4_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// 4x4 WHT (Walsh-Hadamard Transform)
// ============================================================================

/// WHT4x4 - Walsh-Hadamard Transform
/// Uses the correct formula from itx_1d.rs
#[cfg(target_arch = "x86_64")]
fn inv_txfm_add_wht_wht_4x4_8bpc_avx2_inner(
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    // WHT4 1D from itx_1d.rs:
    // t0 = in0 + in1
    // t2 = in2 - in3
    // t4 = (t0 - t2) >> 1
    // t3 = t4 - in3
    // t1 = t4 - in1
    // out0 = t0 - t3
    // out1 = t3
    // out2 = t1
    // out3 = t2 + t1

    let mut tmp = [0i32; 16];

    // Row transform: load from column-major, store row-major
    // Row y has coeffs at: coeff[y + 0*4], coeff[y + 1*4], coeff[y + 2*4], coeff[y + 3*4]
    for y in 0..4 {
        let in0 = coeff[y] as i32 >> 2;
        let in1 = coeff[y + 4] as i32 >> 2;
        let in2 = coeff[y + 8] as i32 >> 2;
        let in3 = coeff[y + 12] as i32 >> 2;

        let t0 = in0 + in1;
        let t2 = in2 - in3;
        let t4 = (t0 - t2) >> 1;
        let t3 = t4 - in3;
        let t1 = t4 - in1;

        // Store row-major: tmp[y*4 + x]
        tmp[y * 4 + 0] = t0 - t3;
        tmp[y * 4 + 1] = t3;
        tmp[y * 4 + 2] = t1;
        tmp[y * 4 + 3] = t2 + t1;
    }

    // Column transform: in-place on row-major data with stride 4
    for x in 0..4 {
        let in0 = tmp[0 * 4 + x];
        let in1 = tmp[1 * 4 + x];
        let in2 = tmp[2 * 4 + x];
        let in3 = tmp[3 * 4 + x];

        let t0 = in0 + in1;
        let t2 = in2 - in3;
        let t4 = (t0 - t2) >> 1;
        let t3 = t4 - in3;
        let t1 = t4 - in1;

        tmp[0 * 4 + x] = t0 - t3;
        tmp[1 * 4 + x] = t3;
        tmp[2 * 4 + x] = t1;
        tmp[3 * 4 + x] = t2 + t1;
    }

    // Add to destination (row-major in tmp)
    for y in 0..4 {
        let row_off = dst_base.wrapping_add_signed(y as isize * dst_stride);
        for x in 0..4 {
            let d = dst[row_off + x] as i32;
            let c = tmp[y * 4 + x];
            let result = iclip(d + c, 0, bitdepth_max);
            dst[row_off + x] = result as u8;
        }
    }

    // Clear coefficients
    coeff[..16].fill(0);
}

/// FFI wrapper for 4x4 WHT 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_wht_wht_4x4_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let abs_stride = dst_stride.unsigned_abs();
    let buf_size = 3 * abs_stride + 4;
    let (base, dst_slice) = if dst_stride >= 0 {
        (0usize, unsafe {
            std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_size)
        })
    } else {
        let start = unsafe { (dst_ptr as *mut u8).offset(3 * dst_stride) };
        (3 * abs_stride, unsafe {
            std::slice::from_raw_parts_mut(start, buf_size)
        })
    };
    let coeff_slice = unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, 16) };
    inv_txfm_add_wht_wht_4x4_8bpc_avx2_inner(
        dst_slice,
        base,
        dst_stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// WHT4x4 16bpc - Walsh-Hadamard Transform
#[cfg(target_arch = "x86_64")]
fn inv_txfm_add_wht_wht_4x4_16bpc_avx2_inner(
    dst: &mut [u16],
    dst_base: usize,
    dst_stride_u16: isize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut tmp = [0i32; 16];

    // Row transform: load from column-major, store row-major
    for y in 0..4 {
        let in0 = coeff[y] as i32 >> 2;
        let in1 = coeff[y + 4] as i32 >> 2;
        let in2 = coeff[y + 8] as i32 >> 2;
        let in3 = coeff[y + 12] as i32 >> 2;

        let t0 = in0 + in1;
        let t2 = in2 - in3;
        let t4 = (t0 - t2) >> 1;
        let t3 = t4 - in3;
        let t1 = t4 - in1;

        tmp[y * 4 + 0] = t0 - t3;
        tmp[y * 4 + 1] = t3;
        tmp[y * 4 + 2] = t1;
        tmp[y * 4 + 3] = t2 + t1;
    }

    // Column transform
    for x in 0..4 {
        let in0 = tmp[0 * 4 + x];
        let in1 = tmp[1 * 4 + x];
        let in2 = tmp[2 * 4 + x];
        let in3 = tmp[3 * 4 + x];

        let t0 = in0 + in1;
        let t2 = in2 - in3;
        let t4 = (t0 - t2) >> 1;
        let t3 = t4 - in3;
        let t1 = t4 - in1;

        tmp[0 * 4 + x] = t0 - t3;
        tmp[1 * 4 + x] = t3;
        tmp[2 * 4 + x] = t1;
        tmp[3 * 4 + x] = t2 + t1;
    }

    // Add to destination
    for y in 0..4 {
        let row_off = dst_base.wrapping_add_signed(y as isize * dst_stride_u16);
        for x in 0..4 {
            let d = dst[row_off + x] as i32;
            let c = tmp[y * 4 + x];
            let result = iclip(d + c, 0, bitdepth_max);
            dst[row_off + x] = result as u16;
        }
    }

    // Clear coefficients
    coeff[..16].fill(0);
}

/// FFI wrapper for 4x4 WHT 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_wht_wht_4x4_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride_u16 = dst_stride / 2;
    let abs_stride_u16 = stride_u16;
    let buf_size = 3 * abs_stride_u16 + 4;
    let (base, dst_slice) = if stride_u16 >= 0 {
        (0usize, unsafe {
            std::slice::from_raw_parts_mut(dst_ptr as *mut u16, buf_size)
        })
    } else {
        let start = unsafe { (dst_ptr as *mut u16).offset(3 * stride_u16) };
        (3 * abs_stride_u16, unsafe {
            std::slice::from_raw_parts_mut(start, buf_size)
        })
    };
    let coeff_slice = unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, 16) };
    inv_txfm_add_wht_wht_4x4_16bpc_avx2_inner(
        dst_slice,
        base,
        stride_u16,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// IDENTITY TRANSFORM HELPER (shared)
// ============================================================================

/// Identity transform - just scale by sqrt(2) and add to dst
#[cfg(target_arch = "x86_64")]
#[arcane]
pub fn inv_identity_add_4x4_8bpc_avx2(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    // identity4(x) = x + (x * 1697 + 2048) >> 12 ≈ x * sqrt(2)
    // 4x4 IDTX = identity4 on rows, identity4 on cols, shift=0, then final (+ 8) >> 4
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi16(bitdepth_max as i16);
    let identity4 = |v: i32| -> i32 { v + ((v * 1697 + 2048) >> 12) };

    for y in 0..4 {
        let dst_off = y * dst_stride;

        // Load destination
        let d = loadi32!(&dst[dst_off..dst_off + 4]);
        let d16 = _mm_unpacklo_epi8(d, zero);

        // Load coeffs for this row (column-major: y, y+4, y+8, y+12)
        let c0 = coeff[y] as i32;
        let c1 = coeff[y + 4] as i32;
        let c2 = coeff[y + 8] as i32;
        let c3 = coeff[y + 12] as i32;

        // Row: identity4, intermediate clamp to col_clip_range, Col: identity4, final: (+ 8) >> 4
        // For 8bpc: col_clip = i16::MIN..i16::MAX. For 16bpc: col_clip = (!bdmax)<<5..=!((!bdmax)<<5)
        let col_clip_min = if bitdepth_max == 255 { i16::MIN as i32 } else { (!bitdepth_max) << 5 };
        let col_clip_max = !col_clip_min;
        let scale = |v: i32| -> i32 { identity4(identity4(v).clamp(col_clip_min, col_clip_max)) };

        let r0 = (scale(c0) + 8) >> 4;
        let r1 = (scale(c1) + 8) >> 4;
        let r2 = (scale(c2) + 8) >> 4;
        let r3 = (scale(c3) + 8) >> 4;

        // Add to destination
        let result = _mm_set_epi32(r3, r2, r1, r0);
        let d32 = _mm_cvtepi16_epi32(d16);
        let sum = _mm_add_epi32(d32, result);
        let sum16 = _mm_packs_epi32(sum, sum);
        let clamped = _mm_max_epi16(_mm_min_epi16(sum16, max_val), zero);
        let packed = _mm_packus_epi16(clamped, clamped);

        storei32!(&mut dst[dst_off..dst_off + 4], packed);
    }

    // Clear coefficients
    coeff[..16].fill(0);
}

/// FFI wrapper for 4x4 IDTX 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_identity_4x4_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, _coeff_len as usize * stride + stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_identity_add_4x4_8bpc_avx2(_token, dst_slice, stride, coeff_slice, eob, bitdepth_max);
}

// ============================================================================
// 8x8 IDTX (Identity)
// ============================================================================

/// 8x8 IDTX (identity transform)
/// Identity8: out = in * 2
/// For 8x8 IDTX: row pass * 2, col pass * 2 = * 4
/// Plus final shift: (+ 8) >> 4
#[cfg(target_arch = "x86_64")]
#[arcane]
pub fn inv_identity_add_8x8_8bpc_avx2(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi16(bitdepth_max as i16);

    // Identity8: c * 2 per dimension. For 8x8: shift=1, rnd=1.
    // Full pipeline per coeff (must use i32 to avoid i16 overflow):
    //   row = c * 2
    //   inter = (row + 1) >> 1
    //   col = inter * 2
    //   residual = (col + 8) >> 4
    let one = _mm_set1_epi32(1);
    let eight = _mm_set1_epi32(8);

    for y in 0..8 {
        let dst_off = y * dst_stride;

        // Load 8 destination pixels
        let d = loadi64!(&dst[dst_off..dst_off + 8]);
        let d16 = _mm_unpacklo_epi8(d, zero);

        // Load 8 coefficients for this row (column-major: y, y+8, y+16, ...)
        let mut coeffs = [0i16; 8];
        for x in 0..8 {
            coeffs[x] = coeff[y + x * 8];
        }

        // Process in i32 to avoid i16 overflow (identity8 doubles, can exceed i16 range)
        let c_vec = loadu_128!(<&[i16; 8]>::try_from(&coeffs[..]).unwrap());
        let c_lo = _mm_cvtepi16_epi32(c_vec);
        let c_hi = _mm_cvtepi16_epi32(_mm_srli_si128(c_vec, 8));

        // row: c * 2
        let row_lo = _mm_slli_epi32(c_lo, 1);
        let row_hi = _mm_slli_epi32(c_hi, 1);
        // intermediate: (row + 1) >> 1
        let inter_lo = _mm_srai_epi32(_mm_add_epi32(row_lo, one), 1);
        let inter_hi = _mm_srai_epi32(_mm_add_epi32(row_hi, one), 1);
        // col: inter * 2
        let col_lo = _mm_slli_epi32(inter_lo, 1);
        let col_hi = _mm_slli_epi32(inter_hi, 1);
        // final: (col + 8) >> 4
        let res_lo = _mm_srai_epi32(_mm_add_epi32(col_lo, eight), 4);
        let res_hi = _mm_srai_epi32(_mm_add_epi32(col_hi, eight), 4);

        // Pack back to i16 and add to destination
        let res16 = _mm_packs_epi32(res_lo, res_hi);
        let sum = _mm_add_epi16(d16, res16);
        let clamped = _mm_max_epi16(_mm_min_epi16(sum, max_val), zero);
        let packed = _mm_packus_epi16(clamped, clamped);

        storei64!(&mut dst[dst_off..dst_off + 8], packed);
    }

    // Clear coefficients (8x8 = 64 i16 = 128 bytes = 8 x 16-byte stores)
    coeff[..64].fill(0);
}

/// FFI wrapper for 8x8 IDTX 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_identity_8x8_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, _coeff_len as usize * stride + stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_identity_add_8x8_8bpc_avx2(_token, dst_slice, stride, coeff_slice, eob, bitdepth_max);
}

// ============================================================================
// 8x8 DCT_DCT
// ============================================================================

/// DCT4 1D transform (used by DCT8)
#[inline]
fn dct4_1d(c: &mut [i32], stride: usize, min: i32, max: i32) {
    let clip = |v: i32| v.clamp(min, max);

    let in0 = c[0 * stride];
    let in1 = c[1 * stride];
    let in2 = c[2 * stride];
    let in3 = c[3 * stride];

    let t0 = (in0 + in2) * 181 + 128 >> 8;
    let t1 = (in0 - in2) * 181 + 128 >> 8;
    let t2 = (in1 * 1567 - in3 * (3784 - 4096) + 2048 >> 12) - in3;
    let t3 = (in1 * (3784 - 4096) + in3 * 1567 + 2048 >> 12) + in1;

    c[0 * stride] = clip(t0 + t3);
    c[1 * stride] = clip(t1 + t2);
    c[2 * stride] = clip(t1 - t2);
    c[3 * stride] = clip(t0 - t3);
}

/// DCT8 1D transform
#[inline]
fn dct8_1d(c: &mut [i32], stride: usize, min: i32, max: i32) {
    let clip = |v: i32| v.clamp(min, max);

    // First apply DCT4 to even positions
    dct4_1d(c, stride * 2, min, max);

    let in1 = c[1 * stride];
    let in3 = c[3 * stride];
    let in5 = c[5 * stride];
    let in7 = c[7 * stride];

    let t4a = (in1 * 799 - in7 * (4017 - 4096) + 2048 >> 12) - in7;
    let t5a = in5 * 1703 - in3 * 1138 + 1024 >> 11;
    let t6a = in5 * 1138 + in3 * 1703 + 1024 >> 11;
    let t7a = (in1 * (4017 - 4096) + in7 * 799 + 2048 >> 12) + in1;

    let t4 = clip(t4a + t5a);
    let t5a = clip(t4a - t5a);
    let t7 = clip(t7a + t6a);
    let t6a = clip(t7a - t6a);

    let t5 = (t6a - t5a) * 181 + 128 >> 8;
    let t6 = (t6a + t5a) * 181 + 128 >> 8;

    let t0 = c[0 * stride];
    let t1 = c[2 * stride];
    let t2 = c[4 * stride];
    let t3 = c[6 * stride];

    c[0 * stride] = clip(t0 + t7);
    c[1 * stride] = clip(t1 + t6);
    c[2 * stride] = clip(t2 + t5);
    c[3 * stride] = clip(t3 + t4);
    c[4 * stride] = clip(t3 - t4);
    c[5 * stride] = clip(t2 - t5);
    c[6 * stride] = clip(t1 - t6);
    c[7 * stride] = clip(t0 - t7);
}

/// ADST4 1D transform (strided version for rectangular transforms)
#[inline]
fn adst4_1d(c: &mut [i32], stride: usize, min: i32, max: i32) {
    let clip = |v: i32| v.clamp(min, max);

    let in0 = c[0 * stride];
    let in1 = c[1 * stride];
    let in2 = c[2 * stride];
    let in3 = c[3 * stride];

    let out0 =
        ((1321 * in0 + (3803 - 4096) * in2 + (2482 - 4096) * in3 + (3344 - 4096) * in1 + 2048)
            >> 12)
            + in2
            + in3
            + in1;
    let out1 =
        (((2482 - 4096) * in0 - 1321 * in2 - (3803 - 4096) * in3 + (3344 - 4096) * in1 + 2048)
            >> 12)
            + in0
            - in3
            + in1;
    let out2 = (209 * (in0 - in2 + in3) + 128) >> 8;
    let out3 = (((3803 - 4096) * in0 + (2482 - 4096) * in2 - 1321 * in3 - (3344 - 4096) * in1
        + 2048)
        >> 12)
        + in0
        + in2
        - in1;

    c[0 * stride] = clip(out0);
    c[1 * stride] = clip(out1);
    c[2 * stride] = clip(out2);
    c[3 * stride] = clip(out3);
}

/// FlipADST4 1D transform (strided version)
#[inline]
fn flipadst4_1d(c: &mut [i32], stride: usize, min: i32, max: i32) {
    let clip = |v: i32| v.clamp(min, max);

    let in0 = c[0 * stride];
    let in1 = c[1 * stride];
    let in2 = c[2 * stride];
    let in3 = c[3 * stride];

    let out0 =
        ((1321 * in0 + (3803 - 4096) * in2 + (2482 - 4096) * in3 + (3344 - 4096) * in1 + 2048)
            >> 12)
            + in2
            + in3
            + in1;
    let out1 =
        (((2482 - 4096) * in0 - 1321 * in2 - (3803 - 4096) * in3 + (3344 - 4096) * in1 + 2048)
            >> 12)
            + in0
            - in3
            + in1;
    let out2 = (209 * (in0 - in2 + in3) + 128) >> 8;
    let out3 = (((3803 - 4096) * in0 + (2482 - 4096) * in2 - 1321 * in3 - (3344 - 4096) * in1
        + 2048)
        >> 12)
        + in0
        + in2
        - in1;

    // Flip output
    c[0 * stride] = clip(out3);
    c[1 * stride] = clip(out2);
    c[2 * stride] = clip(out1);
    c[3 * stride] = clip(out0);
}

/// ADST8 1D transform (strided version for rectangular transforms)
#[inline]
fn adst8_1d(c: &mut [i32], stride: usize, min: i32, max: i32) {
    let clip = |v: i32| v.clamp(min, max);

    let in0 = c[0 * stride];
    let in1 = c[1 * stride];
    let in2 = c[2 * stride];
    let in3 = c[3 * stride];
    let in4 = c[4 * stride];
    let in5 = c[5 * stride];
    let in6 = c[6 * stride];
    let in7 = c[7 * stride];

    let t0a = (((4076 - 4096) * in7 + 401 * in0 + 2048) >> 12) + in7;
    let t1a = ((401 * in7 - (4076 - 4096) * in0 + 2048) >> 12) - in0;
    let t2a = (((3612 - 4096) * in5 + 1931 * in2 + 2048) >> 12) + in5;
    let t3a = ((1931 * in5 - (3612 - 4096) * in2 + 2048) >> 12) - in2;
    let t4a = (1299 * in3 + 1583 * in4 + 1024) >> 11;
    let t5a = (1583 * in3 - 1299 * in4 + 1024) >> 11;
    let t6a = ((1189 * in1 + (3920 - 4096) * in6 + 2048) >> 12) + in6;
    let t7a = (((3920 - 4096) * in1 - 1189 * in6 + 2048) >> 12) + in1;

    let t0 = clip(t0a + t4a);
    let t1 = clip(t1a + t5a);
    let t2 = clip(t2a + t6a);
    let t3 = clip(t3a + t7a);
    let t4 = clip(t0a - t4a);
    let t5 = clip(t1a - t5a);
    let t6 = clip(t2a - t6a);
    let t7 = clip(t3a - t7a);

    let t4a = (((3784 - 4096) * t4 + 1567 * t5 + 2048) >> 12) + t4;
    let t5a = ((1567 * t4 - (3784 - 4096) * t5 + 2048) >> 12) - t5;
    let t6a = (((3784 - 4096) * t7 - 1567 * t6 + 2048) >> 12) + t7;
    let t7a = ((1567 * t7 + (3784 - 4096) * t6 + 2048) >> 12) + t6;

    let out0 = clip(t0 + t2);
    let out7 = -clip(t1 + t3);
    let t2_final = clip(t0 - t2);
    let t3_final = clip(t1 - t3);
    let out1 = -clip(t4a + t6a);
    let out6 = clip(t5a + t7a);
    let t6_final = clip(t4a - t6a);
    let t7_final = clip(t5a - t7a);

    let out3 = -(((t2_final + t3_final) * 181 + 128) >> 8);
    let out4 = ((t2_final - t3_final) * 181 + 128) >> 8;
    let out2 = ((t6_final + t7_final) * 181 + 128) >> 8;
    let out5 = -(((t6_final - t7_final) * 181 + 128) >> 8);

    c[0 * stride] = out0;
    c[1 * stride] = out1;
    c[2 * stride] = out2;
    c[3 * stride] = out3;
    c[4 * stride] = out4;
    c[5 * stride] = out5;
    c[6 * stride] = out6;
    c[7 * stride] = out7;
}

/// FlipADST8 1D transform (strided version)
#[inline]
fn flipadst8_1d(c: &mut [i32], stride: usize, min: i32, max: i32) {
    let clip = |v: i32| v.clamp(min, max);

    let in0 = c[0 * stride];
    let in1 = c[1 * stride];
    let in2 = c[2 * stride];
    let in3 = c[3 * stride];
    let in4 = c[4 * stride];
    let in5 = c[5 * stride];
    let in6 = c[6 * stride];
    let in7 = c[7 * stride];

    let t0a = (((4076 - 4096) * in7 + 401 * in0 + 2048) >> 12) + in7;
    let t1a = ((401 * in7 - (4076 - 4096) * in0 + 2048) >> 12) - in0;
    let t2a = (((3612 - 4096) * in5 + 1931 * in2 + 2048) >> 12) + in5;
    let t3a = ((1931 * in5 - (3612 - 4096) * in2 + 2048) >> 12) - in2;
    let t4a = (1299 * in3 + 1583 * in4 + 1024) >> 11;
    let t5a = (1583 * in3 - 1299 * in4 + 1024) >> 11;
    let t6a = ((1189 * in1 + (3920 - 4096) * in6 + 2048) >> 12) + in6;
    let t7a = (((3920 - 4096) * in1 - 1189 * in6 + 2048) >> 12) + in1;

    let t0 = clip(t0a + t4a);
    let t1 = clip(t1a + t5a);
    let t2 = clip(t2a + t6a);
    let t3 = clip(t3a + t7a);
    let t4 = clip(t0a - t4a);
    let t5 = clip(t1a - t5a);
    let t6 = clip(t2a - t6a);
    let t7 = clip(t3a - t7a);

    let t4a = (((3784 - 4096) * t4 + 1567 * t5 + 2048) >> 12) + t4;
    let t5a = ((1567 * t4 - (3784 - 4096) * t5 + 2048) >> 12) - t5;
    let t6a = (((3784 - 4096) * t7 - 1567 * t6 + 2048) >> 12) + t7;
    let t7a = ((1567 * t7 + (3784 - 4096) * t6 + 2048) >> 12) + t6;

    let out0 = clip(t0 + t2);
    let out7 = -clip(t1 + t3);
    let t2_final = clip(t0 - t2);
    let t3_final = clip(t1 - t3);
    let out1 = -clip(t4a + t6a);
    let out6 = clip(t5a + t7a);
    let t6_final = clip(t4a - t6a);
    let t7_final = clip(t5a - t7a);

    let out3 = -(((t2_final + t3_final) * 181 + 128) >> 8);
    let out4 = ((t2_final - t3_final) * 181 + 128) >> 8;
    let out2 = ((t6_final + t7_final) * 181 + 128) >> 8;
    let out5 = -(((t6_final - t7_final) * 181 + 128) >> 8);

    // Flip output
    c[0 * stride] = out7;
    c[1 * stride] = out6;
    c[2 * stride] = out5;
    c[3 * stride] = out4;
    c[4 * stride] = out3;
    c[5 * stride] = out2;
    c[6 * stride] = out1;
    c[7 * stride] = out0;
}

/// Full 2D DCT_DCT 8x8 inverse transform with add-to-destination
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_8x8_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    // For 8bpc:
    // row_clip_min/max = i16::MIN/MAX (-32768, 32767)
    // col_clip_min/max = i16::MIN/MAX
    let row_clip_min = i16::MIN as i32;
    let row_clip_max = i16::MAX as i32;
    let col_clip_min = i16::MIN as i32;
    let col_clip_max = i16::MAX as i32;

    // Load coefficients and convert to i32 row-major
    // Input is column-major: coeff[y + x * 8]
    let mut tmp = [0i32; 64];

    // Row transform
    // shift = 1 for 8x8
    let rnd = 1;
    let shift = 1;

    for y in 0..8 {
        // Load row from column-major
        let mut scratch = [0i32; 8];
        for x in 0..8 {
            scratch[x] = coeff[y + x * 8] as i32;
        }
        dct8_1d(&mut scratch[..8], 1, row_clip_min, row_clip_max);
        // Apply intermediate shift and store row-major
        for x in 0..8 {
            tmp[y * 8 + x] = iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
        }
    }

    // Column transform (in-place, row-major with stride 8)
    for x in 0..8 {
        dct8_1d(&mut tmp[x..], 8, col_clip_min, col_clip_max);
    }

    // Add to destination with SIMD
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi16(bitdepth_max as i16);
    let rnd_final = _mm256_set1_epi32(8);

    for y in 0..8 {
        let dst_off = y * dst_stride;

        // Load destination pixels (8 bytes)
        let d = loadi64!(&dst[dst_off..dst_off + 8]);
        let d16 = _mm_unpacklo_epi8(d, zero);

        // Load and scale coefficients
        let c_lo = _mm_set_epi32(
            tmp[y * 8 + 3],
            tmp[y * 8 + 2],
            tmp[y * 8 + 1],
            tmp[y * 8 + 0],
        );
        let c_hi = _mm_set_epi32(
            tmp[y * 8 + 7],
            tmp[y * 8 + 6],
            tmp[y * 8 + 5],
            tmp[y * 8 + 4],
        );

        // Final scaling: (c + 8) >> 4
        let c_lo_256 = _mm256_set_m128i(c_hi, c_lo);
        let c_scaled = _mm256_srai_epi32(_mm256_add_epi32(c_lo_256, rnd_final), 4);

        // Pack to 16-bit
        let c_lo_scaled = _mm256_castsi256_si128(c_scaled);
        let c_hi_scaled = _mm256_extracti128_si256(c_scaled, 1);
        let c16 = _mm_packs_epi32(c_lo_scaled, c_hi_scaled);

        // Add to destination
        let sum = _mm_add_epi16(d16, c16);
        let clamped = _mm_max_epi16(_mm_min_epi16(sum, max_val), zero);
        let packed = _mm_packus_epi16(clamped, clamped);

        // Store 8 pixels
        storei64!(&mut dst[dst_off..dst_off + 8], packed);
    }

    // Clear coefficients
    coeff[..64].fill(0);
}

/// FFI wrapper for 8x8 DCT_DCT 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_8x8_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, _coeff_len as usize * stride + stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_8x8_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// 8x8 DCT_DCT 16bpc
// ============================================================================

/// 8x8 DCT_DCT for 16bpc (10/12-bit pixels)
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_8x8_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize, // stride in bytes
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    // For 16bpc, stride is in bytes but we access u16
    let stride_u16 = dst_stride / 2;

    // For 16bpc: intermediate values have larger range, use i32 throughout
    let row_clip_min = i32::MIN;
    let row_clip_max = i32::MAX;
    let col_clip_min = i32::MIN;
    let col_clip_max = i32::MAX;

    // Load coefficients and convert to i32 row-major
    // Input is column-major: coeff[y + x * 8]
    let mut tmp = [0i32; 64];

    // Row transform
    // shift = 1 for 8x8
    let rnd = 1;
    let shift = 1;

    for y in 0..8 {
        // Load row from column-major
        let mut scratch = [0i32; 8];
        for x in 0..8 {
            scratch[x] = coeff[y + x * 8] as i32;
        }
        dct8_1d(&mut scratch[..8], 1, row_clip_min, row_clip_max);
        // Apply intermediate shift and store row-major
        for x in 0..8 {
            tmp[y * 8 + x] = (scratch[x] + rnd) >> shift;
        }
    }

    // Column transform (in-place, row-major with stride 8)
    for x in 0..8 {
        dct8_1d(&mut tmp[x..], 8, col_clip_min, col_clip_max);
    }

    // Add to destination with SIMD (16bpc = u16 pixels)
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi32(bitdepth_max);
    let rnd_final = _mm_set1_epi32(8);

    for y in 0..8 {
        let dst_off = y * stride_u16;

        // Load destination pixels (8 u16 = 16 bytes)
        let d = loadu_128!(<&[u16; 8]>::try_from(&dst[dst_off..dst_off + 8]).unwrap());
        let d_lo = _mm_unpacklo_epi16(d, zero); // First 4 as i32
        let d_hi = _mm_unpackhi_epi16(d, zero); // Last 4 as i32

        // Load and scale coefficients
        let c_lo = _mm_set_epi32(
            tmp[y * 8 + 3],
            tmp[y * 8 + 2],
            tmp[y * 8 + 1],
            tmp[y * 8 + 0],
        );
        let c_hi = _mm_set_epi32(
            tmp[y * 8 + 7],
            tmp[y * 8 + 6],
            tmp[y * 8 + 5],
            tmp[y * 8 + 4],
        );

        // Final scaling: (c + 8) >> 4
        let c_lo_scaled = _mm_srai_epi32(_mm_add_epi32(c_lo, rnd_final), 4);
        let c_hi_scaled = _mm_srai_epi32(_mm_add_epi32(c_hi, rnd_final), 4);

        // Add to destination
        let sum_lo = _mm_add_epi32(d_lo, c_lo_scaled);
        let sum_hi = _mm_add_epi32(d_hi, c_hi_scaled);

        // Clamp to [0, bitdepth_max]
        let clamped_lo = _mm_max_epi32(_mm_min_epi32(sum_lo, max_val), zero);
        let clamped_hi = _mm_max_epi32(_mm_min_epi32(sum_hi, max_val), zero);

        // Pack to u16 and store
        let packed = _mm_packus_epi32(clamped_lo, clamped_hi);
        storeu_128!(
            <&mut [u16; 8]>::try_from(&mut dst[dst_off..dst_off + 8]).unwrap(),
            packed
        );
    }

    // Clear coefficients
    coeff[..64].fill(0);
}

/// FFI wrapper for 8x8 DCT_DCT 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_8x8_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_8x8_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// 16x16 IDTX (Identity)
// ============================================================================

/// 16x16 IDTX (identity transform)
/// Identity16: out = 2 * in + (in * 1697 + 1024) >> 11
/// For 16x16 IDTX: apply identity16 to rows, then identity16 to cols
/// Plus final shift: (+ 8) >> 4
#[cfg(target_arch = "x86_64")]
#[arcane]
pub fn inv_identity_add_16x16_8bpc_avx2(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi16(bitdepth_max as i16);

    // Identity16 scale factor: f(x) = 2*x + (x*1697 + 1024) >> 11
    // For 16x16: row_pass → intermediate shift (+ 2) >> 2 with clamp → col_pass → (+ 8) >> 4
    // The intermediate shift is shift=2, rnd=2 for 16x16 (from inv_txfm_add_rust)
    // Clamp to col_clip range: i16::MIN..=i16::MAX for 8bpc

    // Row pass: identity16 on each row
    let mut tmp = [[0i32; 16]; 16];
    for y in 0..16 {
        for x in 0..16 {
            let c = coeff[y + x * 16] as i32;
            let r = 2 * c + ((c * 1697 + 1024) >> 11);
            tmp[y][x] = r;
        }
    }

    // Intermediate shift: (val + 2) >> 2, clamped to i16 range
    for y in 0..16 {
        for x in 0..16 {
            tmp[y][x] = ((tmp[y][x] + 2) >> 2).clamp(i16::MIN as i32, i16::MAX as i32);
        }
    }

    // Column pass: identity16, then final shift
    for x in 0..16 {
        for y in 0..16 {
            let c = tmp[y][x];
            let r = 2 * c + ((c * 1697 + 1024) >> 11);
            tmp[y][x] = (r + 8) >> 4;
        }
    }

    // Add to destination with SIMD
    for y in 0..16 {
        let dst_off = y * dst_stride;

        // Load 16 destination pixels
        let d = loadu_128!(<&[u8; 16]>::try_from(&dst[dst_off..dst_off + 16]).unwrap());
        let d_lo = _mm256_cvtepu8_epi16(d);

        // Load 16 transformed coefficients
        let c_vec = _mm256_set_epi16(
            tmp[y][15] as i16,
            tmp[y][14] as i16,
            tmp[y][13] as i16,
            tmp[y][12] as i16,
            tmp[y][11] as i16,
            tmp[y][10] as i16,
            tmp[y][9] as i16,
            tmp[y][8] as i16,
            tmp[y][7] as i16,
            tmp[y][6] as i16,
            tmp[y][5] as i16,
            tmp[y][4] as i16,
            tmp[y][3] as i16,
            tmp[y][2] as i16,
            tmp[y][1] as i16,
            tmp[y][0] as i16,
        );

        // Add and clamp
        let sum = _mm256_add_epi16(d_lo, c_vec);
        let clamped = _mm256_max_epi16(_mm256_min_epi16(sum, max_val), zero);

        // Pack to bytes
        let packed = _mm256_packus_epi16(clamped, clamped);
        let packed_lo = _mm256_castsi256_si128(packed);
        let packed_hi = _mm256_extracti128_si256(packed, 1);
        let result = _mm_unpacklo_epi64(packed_lo, packed_hi);

        storeu_128!(
            <&mut [u8; 16]>::try_from(&mut dst[dst_off..dst_off + 16]).unwrap(),
            result
        );
    }

    // Clear coefficients (16x16 = 256 i16 = 512 bytes = 32 x 16-byte stores)
    coeff[..256].fill(0);
}

/// FFI wrapper for 16x16 IDTX 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_identity_16x16_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, _coeff_len as usize * stride + stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_identity_add_16x16_8bpc_avx2(_token, dst_slice, stride, coeff_slice, eob, bitdepth_max);
}

// ============================================================================
// 16x16 DCT_DCT
// ============================================================================

/// DCT16 1D transform
#[inline]
fn dct16_1d(c: &mut [i32], stride: usize, min: i32, max: i32) {
    let clip = |v: i32| v.clamp(min, max);

    // First apply DCT8 to even positions
    dct8_1d(c, stride * 2, min, max);

    let in1 = c[1 * stride];
    let in3 = c[3 * stride];
    let in5 = c[5 * stride];
    let in7 = c[7 * stride];
    let in9 = c[9 * stride];
    let in11 = c[11 * stride];
    let in13 = c[13 * stride];
    let in15 = c[15 * stride];

    let t8a = (in1 * 401 - in15 * (4076 - 4096) + 2048 >> 12) - in15;
    let t9a = in9 * 1583 - in7 * 1299 + 1024 >> 11;
    let t10a = (in5 * 1931 - in11 * (3612 - 4096) + 2048 >> 12) - in11;
    let t11a = (in13 * (3920 - 4096) - in3 * 1189 + 2048 >> 12) + in13;
    let t12a = (in13 * 1189 + in3 * (3920 - 4096) + 2048 >> 12) + in3;
    let t13a = (in5 * (3612 - 4096) + in11 * 1931 + 2048 >> 12) + in5;
    let t14a = in9 * 1299 + in7 * 1583 + 1024 >> 11;
    let t15a = (in1 * (4076 - 4096) + in15 * 401 + 2048 >> 12) + in1;

    let t8 = clip(t8a + t9a);
    let mut t9 = clip(t8a - t9a);
    let mut t10 = clip(t11a - t10a);
    let mut t11 = clip(t11a + t10a);
    let mut t12 = clip(t12a + t13a);
    let mut t13 = clip(t12a - t13a);
    let mut t14 = clip(t15a - t14a);
    let t15 = clip(t15a + t14a);

    let t9a = (t14 * 1567 - t9 * (3784 - 4096) + 2048 >> 12) - t9;
    let t14a = (t14 * (3784 - 4096) + t9 * 1567 + 2048 >> 12) + t14;
    let t10a = (-(t13 * (3784 - 4096) + t10 * 1567) + 2048 >> 12) - t13;
    let t13a = (t13 * 1567 - t10 * (3784 - 4096) + 2048 >> 12) - t10;

    let t8a = clip(t8 + t11);
    t9 = clip(t9a + t10a);
    t10 = clip(t9a - t10a);
    let t11a = clip(t8 - t11);
    let t12a = clip(t15 - t12);
    t13 = clip(t14a - t13a);
    t14 = clip(t14a + t13a);
    let t15a = clip(t15 + t12);

    let t10a_new = (t13 - t10) * 181 + 128 >> 8;
    let t13a_new = (t13 + t10) * 181 + 128 >> 8;
    t11 = (t12a - t11a) * 181 + 128 >> 8;
    t12 = (t12a + t11a) * 181 + 128 >> 8;

    let t0 = c[0 * stride];
    let t1 = c[2 * stride];
    let t2 = c[4 * stride];
    let t3 = c[6 * stride];
    let t4 = c[8 * stride];
    let t5 = c[10 * stride];
    let t6 = c[12 * stride];
    let t7 = c[14 * stride];

    c[0 * stride] = clip(t0 + t15a);
    c[1 * stride] = clip(t1 + t14);
    c[2 * stride] = clip(t2 + t13a_new);
    c[3 * stride] = clip(t3 + t12);
    c[4 * stride] = clip(t4 + t11);
    c[5 * stride] = clip(t5 + t10a_new);
    c[6 * stride] = clip(t6 + t9);
    c[7 * stride] = clip(t7 + t8a);
    c[8 * stride] = clip(t7 - t8a);
    c[9 * stride] = clip(t6 - t9);
    c[10 * stride] = clip(t5 - t10a_new);
    c[11 * stride] = clip(t4 - t11);
    c[12 * stride] = clip(t3 - t12);
    c[13 * stride] = clip(t2 - t13a_new);
    c[14 * stride] = clip(t1 - t14);
    c[15 * stride] = clip(t0 - t15a);
}

/// ADST16 1D transform (in-place)
#[inline]
fn adst16_1d(c: &mut [i32], stride: usize, min: i32, max: i32) {
    let clip = |v: i32| v.clamp(min, max);

    let in0 = c[0 * stride];
    let in1 = c[1 * stride];
    let in2 = c[2 * stride];
    let in3 = c[3 * stride];
    let in4 = c[4 * stride];
    let in5 = c[5 * stride];
    let in6 = c[6 * stride];
    let in7 = c[7 * stride];
    let in8 = c[8 * stride];
    let in9 = c[9 * stride];
    let in10 = c[10 * stride];
    let in11 = c[11 * stride];
    let in12 = c[12 * stride];
    let in13 = c[13 * stride];
    let in14 = c[14 * stride];
    let in15 = c[15 * stride];

    let mut t0 = ((in15 * (4091 - 4096) + in0 * 201 + 2048) >> 12) + in15;
    let mut t1 = ((in15 * 201 - in0 * (4091 - 4096) + 2048) >> 12) - in0;
    let mut t2 = ((in13 * (3973 - 4096) + in2 * 995 + 2048) >> 12) + in13;
    let mut t3 = ((in13 * 995 - in2 * (3973 - 4096) + 2048) >> 12) - in2;
    let mut t4 = ((in11 * (3703 - 4096) + in4 * 1751 + 2048) >> 12) + in11;
    let mut t5 = ((in11 * 1751 - in4 * (3703 - 4096) + 2048) >> 12) - in4;
    let mut t6 = (in9 * 1645 + in6 * 1220 + 1024) >> 11;
    let mut t7 = (in9 * 1220 - in6 * 1645 + 1024) >> 11;
    let mut t8 = ((in7 * 2751 + in8 * (3035 - 4096) + 2048) >> 12) + in8;
    let mut t9 = ((in7 * (3035 - 4096) - in8 * 2751 + 2048) >> 12) + in7;
    let mut t10 = ((in5 * 2106 + in10 * (3513 - 4096) + 2048) >> 12) + in10;
    let mut t11 = ((in5 * (3513 - 4096) - in10 * 2106 + 2048) >> 12) + in5;
    let mut t12 = ((in3 * 1380 + in12 * (3857 - 4096) + 2048) >> 12) + in12;
    let mut t13 = ((in3 * (3857 - 4096) - in12 * 1380 + 2048) >> 12) + in3;
    let mut t14 = ((in1 * 601 + in14 * (4052 - 4096) + 2048) >> 12) + in14;
    let mut t15 = ((in1 * (4052 - 4096) - in14 * 601 + 2048) >> 12) + in1;

    let t0a = clip(t0 + t8);
    let t1a = clip(t1 + t9);
    let mut t2a = clip(t2 + t10);
    let mut t3a = clip(t3 + t11);
    let mut t4a = clip(t4 + t12);
    let mut t5a = clip(t5 + t13);
    let mut t6a = clip(t6 + t14);
    let mut t7a = clip(t7 + t15);
    let mut t8a = clip(t0 - t8);
    let mut t9a = clip(t1 - t9);
    let mut t10a = clip(t2 - t10);
    let mut t11a = clip(t3 - t11);
    let mut t12a = clip(t4 - t12);
    let mut t13a = clip(t5 - t13);
    let mut t14a = clip(t6 - t14);
    let mut t15a = clip(t7 - t15);

    t8 = ((t8a * (4017 - 4096) + t9a * 799 + 2048) >> 12) + t8a;
    t9 = ((t8a * 799 - t9a * (4017 - 4096) + 2048) >> 12) - t9a;
    t10 = ((t10a * 2276 + t11a * (3406 - 4096) + 2048) >> 12) + t11a;
    t11 = ((t10a * (3406 - 4096) - t11a * 2276 + 2048) >> 12) + t10a;
    t12 = ((t13a * (4017 - 4096) - t12a * 799 + 2048) >> 12) + t13a;
    t13 = ((t13a * 799 + t12a * (4017 - 4096) + 2048) >> 12) + t12a;
    t14 = ((t15a * 2276 - t14a * (3406 - 4096) + 2048) >> 12) - t14a;
    t15 = ((t15a * (3406 - 4096) + t14a * 2276 + 2048) >> 12) + t15a;

    t0 = clip(t0a + t4a);
    t1 = clip(t1a + t5a);
    t2 = clip(t2a + t6a);
    t3 = clip(t3a + t7a);
    t4 = clip(t0a - t4a);
    t5 = clip(t1a - t5a);
    t6 = clip(t2a - t6a);
    t7 = clip(t3a - t7a);
    t8a = clip(t8 + t12);
    t9a = clip(t9 + t13);
    t10a = clip(t10 + t14);
    t11a = clip(t11 + t15);
    t12a = clip(t8 - t12);
    t13a = clip(t9 - t13);
    t14a = clip(t10 - t14);
    t15a = clip(t11 - t15);

    t4a = ((t4 * (3784 - 4096) + t5 * 1567 + 2048) >> 12) + t4;
    t5a = ((t4 * 1567 - t5 * (3784 - 4096) + 2048) >> 12) - t5;
    t6a = ((t7 * (3784 - 4096) - t6 * 1567 + 2048) >> 12) + t7;
    t7a = ((t7 * 1567 + t6 * (3784 - 4096) + 2048) >> 12) + t6;
    t12 = ((t12a * (3784 - 4096) + t13a * 1567 + 2048) >> 12) + t12a;
    t13 = ((t12a * 1567 - t13a * (3784 - 4096) + 2048) >> 12) - t13a;
    t14 = ((t15a * (3784 - 4096) - t14a * 1567 + 2048) >> 12) + t15a;
    t15 = ((t15a * 1567 + t14a * (3784 - 4096) + 2048) >> 12) + t14a;

    c[0 * stride] = clip(t0 + t2);
    c[15 * stride] = -clip(t1 + t3);
    t2a = clip(t0 - t2);
    t3a = clip(t1 - t3);
    c[3 * stride] = -clip(t4a + t6a);
    c[12 * stride] = clip(t5a + t7a);
    t6 = clip(t4a - t6a);
    t7 = clip(t5a - t7a);
    c[1 * stride] = -clip(t8a + t10a);
    c[14 * stride] = clip(t9a + t11a);
    t10 = clip(t8a - t10a);
    t11 = clip(t9a - t11a);
    c[2 * stride] = clip(t12 + t14);
    c[13 * stride] = -clip(t13 + t15);
    t14a = clip(t12 - t14);
    t15a = clip(t13 - t15);

    c[7 * stride] = -(((t2a + t3a) * 181 + 128) >> 8);
    c[8 * stride] = ((t2a - t3a) * 181 + 128) >> 8;
    c[4 * stride] = ((t6 + t7) * 181 + 128) >> 8;
    c[11 * stride] = -(((t6 - t7) * 181 + 128) >> 8);
    c[6 * stride] = ((t10 + t11) * 181 + 128) >> 8;
    c[9 * stride] = -(((t10 - t11) * 181 + 128) >> 8);
    c[5 * stride] = -(((t14a + t15a) * 181 + 128) >> 8);
    c[10 * stride] = ((t14a - t15a) * 181 + 128) >> 8;
}

/// FlipADST16 1D transform (in-place)
#[inline]
fn flipadst16_1d(c: &mut [i32], stride: usize, min: i32, max: i32) {
    // Apply ADST then reverse
    adst16_1d(c, stride, min, max);
    // Swap in place
    for i in 0..8 {
        let tmp = c[i * stride];
        c[i * stride] = c[(15 - i) * stride];
        c[(15 - i) * stride] = tmp;
    }
}

/// Identity4 1D transform (strided, in-place)
#[inline]
fn identity4_1d(c: &mut [i32], stride: usize, _min: i32, _max: i32) {
    // Identity4: out = in + (in * 1697 + 2048) >> 12
    // This is approximately in * sqrt(2) ≈ in * 1.414
    for i in 0..4 {
        let in_0 = c[i * stride];
        c[i * stride] = in_0 + (in_0 * 1697 + 2048 >> 12);
    }
}

/// Identity8 1D transform (strided, in-place)
#[inline]
fn identity8_1d(c: &mut [i32], stride: usize, _min: i32, _max: i32) {
    // For 8pt identity: out = in * 2
    for i in 0..8 {
        c[i * stride] *= 2;
    }
}

/// Identity16 1D transform (in-place)
#[inline]
fn identity16_1d(c: &mut [i32], stride: usize, _min: i32, _max: i32) {
    // Identity16: out = 2 * in + (in * 1697 + 1024) >> 11
    // This is approximately in * 2 * sqrt(2) ≈ in * 2.829
    for i in 0..16 {
        let in_0 = c[i * stride];
        c[i * stride] = 2 * in_0 + (in_0 * 1697 + 1024 >> 11);
    }
}

/// Generic 16x16 transform function
#[inline]
fn inv_txfm_16x16_inner(
    tmp: &mut [i32; 256],
    coeff: &[i16],
    row_transform: fn(&mut [i32], usize, i32, i32),
    col_transform: fn(&mut [i32], usize, i32, i32),
    row_clip_min: i32,
    row_clip_max: i32,
    col_clip_min: i32,
    col_clip_max: i32,
) {
    let rnd = 2;
    let shift = 2;

    // Row transform
    for y in 0..16 {
        // Load row from column-major
        let mut scratch = [0i32; 16];
        for x in 0..16 {
            scratch[x] = coeff[y + x * 16] as i32;
        }
        row_transform(&mut scratch[..16], 1, row_clip_min, row_clip_max);
        // Apply intermediate shift and store row-major
        for x in 0..16 {
            tmp[y * 16 + x] = ((scratch[x] + rnd) >> shift).clamp(col_clip_min, col_clip_max);
        }
    }

    // Column transform (in-place, row-major with stride 16)
    for x in 0..16 {
        col_transform(&mut tmp[x..], 16, col_clip_min, col_clip_max);
    }
}

/// Add transformed coefficients to destination with SIMD
#[cfg(target_arch = "x86_64")]
#[arcane]
fn add_16x16_to_dst(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp: &[i32; 256],
    coeff: &mut [i16],
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();

    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi16(bitdepth_max as i16);
    let rnd_final = _mm256_set1_epi32(8);

    for y in 0..16 {
        let dst_off = y * dst_stride;

        // Load destination pixels (16 bytes)
        let d = loadu_128!(<&[u8; 16]>::try_from(&dst[dst_off..dst_off + 16]).unwrap());
        let d16 = _mm256_cvtepu8_epi16(d);

        // Load and scale coefficients (16 values)
        let c0 = _mm256_set_epi32(
            tmp[y * 16 + 7],
            tmp[y * 16 + 6],
            tmp[y * 16 + 5],
            tmp[y * 16 + 4],
            tmp[y * 16 + 3],
            tmp[y * 16 + 2],
            tmp[y * 16 + 1],
            tmp[y * 16 + 0],
        );
        let c1 = _mm256_set_epi32(
            tmp[y * 16 + 15],
            tmp[y * 16 + 14],
            tmp[y * 16 + 13],
            tmp[y * 16 + 12],
            tmp[y * 16 + 11],
            tmp[y * 16 + 10],
            tmp[y * 16 + 9],
            tmp[y * 16 + 8],
        );

        // Final scaling: (c + 8) >> 4
        let c0_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c0, rnd_final));
        let c1_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c1, rnd_final));

        // Pack to 16-bit
        let c16 = _mm256_packs_epi32(c0_scaled, c1_scaled);
        // Fix lane order after packs
        let c16 = _mm256_permute4x64_epi64::<0b11_01_10_00>(c16);

        // Add to destination
        let sum = _mm256_add_epi16(d16, c16);
        let clamped = _mm256_max_epi16(_mm256_min_epi16(sum, max_val), zero);

        // Pack to 8-bit
        let packed = _mm256_packus_epi16(clamped, clamped);
        let packed = _mm256_permute4x64_epi64::<0b11_01_10_00>(packed);

        // Store 16 pixels
        storeu_128!(
            <&mut [u8; 16]>::try_from(&mut dst[dst_off..dst_off + 16]).unwrap(),
            _mm256_castsi256_si128(packed)
        );
    }

    // Clear coefficients (256 * 2 = 512 bytes = 16 * 32 bytes)
    coeff[..256].fill(0);
}

// ============================================================================
// 16x16 ADST TRANSFORM VARIANTS
// ============================================================================

/// Macro to generate 16x16 transform inner functions
macro_rules! impl_16x16_transform {
    ($name:ident, $row_fn:ident, $col_fn:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[arcane]
        fn $name(
            _token: Desktop64,
            dst: &mut [u8],
            dst_stride: usize,
            coeff: &mut [i16],
            _eob: i32,
            bitdepth_max: i32,
        ) {
            use crate::src::safe_simd::pixel_access::{
                loadi32, loadi64, loadu_128, storei32, storei64, storeu_128,
            };
            let mut dst = dst.flex_mut();
            let mut coeff = coeff.flex_mut();
            let row_clip_min = i16::MIN as i32;
            let row_clip_max = i16::MAX as i32;
            let col_clip_min = i16::MIN as i32;
            let col_clip_max = i16::MAX as i32;

            let mut tmp = [0i32; 256];
            inv_txfm_16x16_inner(
                &mut tmp,
                &*coeff,
                $row_fn,
                $col_fn,
                row_clip_min,
                row_clip_max,
                col_clip_min,
                col_clip_max,
            );
            add_16x16_to_dst(
                _token,
                &mut *dst,
                dst_stride,
                &tmp,
                &mut *coeff,
                bitdepth_max,
            );
        }
    };
}

/// Macro to generate FFI wrappers for 16x16 transforms
macro_rules! impl_16x16_ffi_wrapper {
    ($wrapper:ident, $inner:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2")]
        #[cfg(feature = "asm")]
        pub unsafe extern "C" fn $wrapper(
            dst_ptr: *mut DynPixel,
            dst_stride: isize,
            coeff: *mut DynCoef,
            eob: c_int,
            bitdepth_max: c_int,
            _coeff_len: u16,
            _dst: *const FFISafe<PicOffset>,
        ) {
            let _token = unsafe { Desktop64::forge_token_dangerously() };
            let stride = dst_stride as usize;

            let dst_slice = unsafe {
                std::slice::from_raw_parts_mut(
                    dst_ptr as *mut u8,
                    _coeff_len as usize * stride + stride,
                )
            };

            let coeff_slice =
                unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

            $inner(_token, dst_slice, stride, coeff_slice, eob, bitdepth_max);
        }
    };
}

// Generate inner functions for all 16x16 transform combinations
impl_16x16_transform!(
    inv_txfm_add_adst_dct_16x16_8bpc_avx2_inner,
    adst16_1d,
    dct16_1d
);
impl_16x16_transform!(
    inv_txfm_add_dct_adst_16x16_8bpc_avx2_inner,
    dct16_1d,
    adst16_1d
);
impl_16x16_transform!(
    inv_txfm_add_adst_adst_16x16_8bpc_avx2_inner,
    adst16_1d,
    adst16_1d
);
impl_16x16_transform!(
    inv_txfm_add_flipadst_dct_16x16_8bpc_avx2_inner,
    flipadst16_1d,
    dct16_1d
);
impl_16x16_transform!(
    inv_txfm_add_dct_flipadst_16x16_8bpc_avx2_inner,
    dct16_1d,
    flipadst16_1d
);
impl_16x16_transform!(
    inv_txfm_add_flipadst_flipadst_16x16_8bpc_avx2_inner,
    flipadst16_1d,
    flipadst16_1d
);
impl_16x16_transform!(
    inv_txfm_add_adst_flipadst_16x16_8bpc_avx2_inner,
    adst16_1d,
    flipadst16_1d
);
impl_16x16_transform!(
    inv_txfm_add_flipadst_adst_16x16_8bpc_avx2_inner,
    flipadst16_1d,
    adst16_1d
);
impl_16x16_transform!(
    inv_txfm_add_identity_dct_16x16_8bpc_avx2_inner,
    identity16_1d,
    dct16_1d
);
impl_16x16_transform!(
    inv_txfm_add_dct_identity_16x16_8bpc_avx2_inner,
    dct16_1d,
    identity16_1d
);
impl_16x16_transform!(
    inv_txfm_add_identity_adst_16x16_8bpc_avx2_inner,
    identity16_1d,
    adst16_1d
);
impl_16x16_transform!(
    inv_txfm_add_adst_identity_16x16_8bpc_avx2_inner,
    adst16_1d,
    identity16_1d
);
impl_16x16_transform!(
    inv_txfm_add_identity_flipadst_16x16_8bpc_avx2_inner,
    identity16_1d,
    flipadst16_1d
);
impl_16x16_transform!(
    inv_txfm_add_flipadst_identity_16x16_8bpc_avx2_inner,
    flipadst16_1d,
    identity16_1d
);

// Generate FFI wrappers
impl_16x16_ffi_wrapper!(
    inv_txfm_add_adst_dct_16x16_8bpc_avx2,
    inv_txfm_add_adst_dct_16x16_8bpc_avx2_inner
);
impl_16x16_ffi_wrapper!(
    inv_txfm_add_dct_adst_16x16_8bpc_avx2,
    inv_txfm_add_dct_adst_16x16_8bpc_avx2_inner
);
impl_16x16_ffi_wrapper!(
    inv_txfm_add_adst_adst_16x16_8bpc_avx2,
    inv_txfm_add_adst_adst_16x16_8bpc_avx2_inner
);
impl_16x16_ffi_wrapper!(
    inv_txfm_add_flipadst_dct_16x16_8bpc_avx2,
    inv_txfm_add_flipadst_dct_16x16_8bpc_avx2_inner
);
impl_16x16_ffi_wrapper!(
    inv_txfm_add_dct_flipadst_16x16_8bpc_avx2,
    inv_txfm_add_dct_flipadst_16x16_8bpc_avx2_inner
);
impl_16x16_ffi_wrapper!(
    inv_txfm_add_flipadst_flipadst_16x16_8bpc_avx2,
    inv_txfm_add_flipadst_flipadst_16x16_8bpc_avx2_inner
);
impl_16x16_ffi_wrapper!(
    inv_txfm_add_adst_flipadst_16x16_8bpc_avx2,
    inv_txfm_add_adst_flipadst_16x16_8bpc_avx2_inner
);
impl_16x16_ffi_wrapper!(
    inv_txfm_add_flipadst_adst_16x16_8bpc_avx2,
    inv_txfm_add_flipadst_adst_16x16_8bpc_avx2_inner
);
impl_16x16_ffi_wrapper!(
    inv_txfm_add_identity_dct_16x16_8bpc_avx2,
    inv_txfm_add_identity_dct_16x16_8bpc_avx2_inner
);
impl_16x16_ffi_wrapper!(
    inv_txfm_add_dct_identity_16x16_8bpc_avx2,
    inv_txfm_add_dct_identity_16x16_8bpc_avx2_inner
);
impl_16x16_ffi_wrapper!(
    inv_txfm_add_identity_adst_16x16_8bpc_avx2,
    inv_txfm_add_identity_adst_16x16_8bpc_avx2_inner
);
impl_16x16_ffi_wrapper!(
    inv_txfm_add_adst_identity_16x16_8bpc_avx2,
    inv_txfm_add_adst_identity_16x16_8bpc_avx2_inner
);
impl_16x16_ffi_wrapper!(
    inv_txfm_add_identity_flipadst_16x16_8bpc_avx2,
    inv_txfm_add_identity_flipadst_16x16_8bpc_avx2_inner
);
impl_16x16_ffi_wrapper!(
    inv_txfm_add_flipadst_identity_16x16_8bpc_avx2,
    inv_txfm_add_flipadst_identity_16x16_8bpc_avx2_inner
);

/// Full 2D DCT_DCT 16x16 inverse transform with add-to-destination
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_16x16_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    // For 8bpc: row_clip = col_clip = i16 range
    let row_clip_min = i16::MIN as i32;
    let row_clip_max = i16::MAX as i32;
    let col_clip_min = i16::MIN as i32;
    let col_clip_max = i16::MAX as i32;
    let mut tmp = [0i32; 256];

    // Row transform (shift = 2 for 16x16)
    let rnd = 2;
    let shift = 2;

    for y in 0..16 {
        // Load row from column-major
        let mut scratch = [0i32; 16];
        for x in 0..16 {
            scratch[x] = coeff[y + x * 16] as i32;
        }
        dct16_1d(&mut scratch[..16], 1, row_clip_min, row_clip_max);
        // Apply intermediate shift and store row-major
        for x in 0..16 {
            tmp[y * 16 + x] = iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
        }
    }

    // Column transform (in-place, row-major with stride 16)
    for x in 0..16 {
        dct16_1d(&mut tmp[x..], 16, col_clip_min, col_clip_max);
    }

    // Add to destination with SIMD
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi16(bitdepth_max as i16);
    let rnd_final = _mm256_set1_epi32(8);

    for y in 0..16 {
        let dst_off = y * dst_stride;

        // Load destination pixels (16 bytes)
        let d = loadu_128!(<&[u8; 16]>::try_from(&dst[dst_off..dst_off + 16]).unwrap());
        let d16 = _mm256_cvtepu8_epi16(d);

        // Load and scale coefficients (16 values)
        let c0 = _mm256_set_epi32(
            tmp[y * 16 + 7],
            tmp[y * 16 + 6],
            tmp[y * 16 + 5],
            tmp[y * 16 + 4],
            tmp[y * 16 + 3],
            tmp[y * 16 + 2],
            tmp[y * 16 + 1],
            tmp[y * 16 + 0],
        );
        let c1 = _mm256_set_epi32(
            tmp[y * 16 + 15],
            tmp[y * 16 + 14],
            tmp[y * 16 + 13],
            tmp[y * 16 + 12],
            tmp[y * 16 + 11],
            tmp[y * 16 + 10],
            tmp[y * 16 + 9],
            tmp[y * 16 + 8],
        );

        // Final scaling: (c + 8) >> 4
        let c0_scaled = _mm256_srai_epi32(_mm256_add_epi32(c0, rnd_final), 4);
        let c1_scaled = _mm256_srai_epi32(_mm256_add_epi32(c1, rnd_final), 4);

        // Pack to 16-bit
        let c16 = _mm256_packs_epi32(c0_scaled, c1_scaled);
        // Fix lane order after packs
        let c16 = _mm256_permute4x64_epi64(c16, 0b11_01_10_00);

        // Add to destination
        let sum = _mm256_add_epi16(d16, c16);
        let clamped = _mm256_max_epi16(_mm256_min_epi16(sum, max_val), zero);

        // Pack to 8-bit
        let packed = _mm256_packus_epi16(clamped, clamped);
        let packed = _mm256_permute4x64_epi64(packed, 0b11_01_10_00);

        // Store 16 pixels
        storeu_128!(
            <&mut [u8; 16]>::try_from(&mut dst[dst_off..dst_off + 16]).unwrap(),
            _mm256_castsi256_si128(packed)
        );
    }

    // Clear coefficients (256 * 2 = 512 bytes = 16 * 32 bytes)
    coeff[..256].fill(0);
}

/// FFI wrapper for 16x16 DCT_DCT 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_16x16_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, _coeff_len as usize * stride + stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_16x16_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// 16x16 DCT_DCT 16bpc
// ============================================================================

/// 16x16 DCT_DCT for 16bpc (10/12-bit pixels)
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_16x16_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize, // stride in bytes
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    // For 16bpc, stride is in bytes but we access u16
    let stride_u16 = dst_stride / 2;

    // For 16bpc: use full i32 range for intermediate calculations
    let row_clip_min = i32::MIN;
    let row_clip_max = i32::MAX;
    let col_clip_min = i32::MIN;
    let col_clip_max = i32::MAX;
    let mut tmp = [0i32; 256];

    // Row transform (shift = 2 for 16x16)
    let rnd = 2;
    let shift = 2;

    for y in 0..16 {
        // Load row from column-major
        let mut scratch = [0i32; 16];
        for x in 0..16 {
            scratch[x] = coeff[y + x * 16] as i32;
        }
        dct16_1d(&mut scratch[..16], 1, row_clip_min, row_clip_max);
        // Apply intermediate shift and store row-major
        for x in 0..16 {
            tmp[y * 16 + x] = (scratch[x] + rnd) >> shift;
        }
    }

    // Column transform (in-place, row-major with stride 16)
    for x in 0..16 {
        dct16_1d(&mut tmp[x..], 16, col_clip_min, col_clip_max);
    }

    // Add to destination with SIMD (16bpc = u16 pixels)
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi32(bitdepth_max);
    let rnd_final = _mm256_set1_epi32(8);

    for y in 0..16 {
        let dst_off = y * stride_u16;

        // Load destination pixels (16 u16 = 32 bytes)
        let d = loadu_256!(<&[u16; 16]>::try_from(&dst[dst_off..dst_off + 16]).unwrap());
        // Unpack to i32: low 8 and high 8
        let d_lo = _mm256_unpacklo_epi16(d, _mm256_setzero_si256());
        let d_hi = _mm256_unpackhi_epi16(d, _mm256_setzero_si256());
        // Permute to get correct order after unpack
        let d_0_4 = _mm256_permute2x128_si256(d_lo, d_hi, 0x20); // pixels 0-3, 8-11
        let d_4_8 = _mm256_permute2x128_si256(d_lo, d_hi, 0x31); // pixels 4-7, 12-15

        // Load and scale coefficients (16 values)
        let c0 = _mm256_set_epi32(
            tmp[y * 16 + 3],
            tmp[y * 16 + 2],
            tmp[y * 16 + 1],
            tmp[y * 16 + 0],
            tmp[y * 16 + 11],
            tmp[y * 16 + 10],
            tmp[y * 16 + 9],
            tmp[y * 16 + 8],
        );
        let c1 = _mm256_set_epi32(
            tmp[y * 16 + 7],
            tmp[y * 16 + 6],
            tmp[y * 16 + 5],
            tmp[y * 16 + 4],
            tmp[y * 16 + 15],
            tmp[y * 16 + 14],
            tmp[y * 16 + 13],
            tmp[y * 16 + 12],
        );

        // Final scaling: (c + 8) >> 4
        let c0_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c0, rnd_final));
        let c1_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c1, rnd_final));

        // Add to destination
        let sum0 = _mm256_add_epi32(d_0_4, c0_scaled);
        let sum1 = _mm256_add_epi32(d_4_8, c1_scaled);

        // Clamp to [0, bitdepth_max]
        let clamped0 = _mm256_max_epi32(_mm256_min_epi32(sum0, max_val), zero);
        let clamped1 = _mm256_max_epi32(_mm256_min_epi32(sum1, max_val), zero);

        // Pack to u16 and store
        let packed = _mm256_packus_epi32(clamped0, clamped1);
        // Fix lane order after packus
        let packed = _mm256_permute4x64_epi64(packed, 0b11_01_10_00);
        storeu_256!(
            <&mut [u16; 16]>::try_from(&mut dst[dst_off..dst_off + 16]).unwrap(),
            packed
        );
    }

    // Clear coefficients (256 * 2 = 512 bytes = 16 * 32 bytes)
    coeff[..256].fill(0);
}

/// FFI wrapper for 16x16 DCT_DCT 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_16x16_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_16x16_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// RECTANGULAR TRANSFORMS (4x8, 8x4, etc.)
// ============================================================================

/// Full 2D DCT_DCT 4x8 inverse transform
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_4x8_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    // W=4, H=8, shift=0 for 4x8
    let row_clip_min = i16::MIN as i32;
    let row_clip_max = i16::MAX as i32;
    let col_clip_min = i16::MIN as i32;
    let col_clip_max = i16::MAX as i32;
    let mut tmp = [0i32; 32];

    // is_rect2 = true for 4x8, so apply sqrt(2) scaling
    let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

    // Row transform (4 elements each, 8 rows)
    for y in 0..8 {
        // Load row from column-major with rect2 scaling
        let mut scratch = [0i32; 4];
        for x in 0..4 {
            scratch[x] = rect2_scale(coeff[y + x * 8] as i32);
        }
        dct4_1d(&mut scratch[..4], 1, row_clip_min, row_clip_max);
        // Store row-major (no intermediate shift for 4x8)
        for x in 0..4 {
            tmp[y * 4 + x] = iclip(scratch[x], col_clip_min, col_clip_max);
        }
    }

    // Column transform (8 elements each, 4 columns)
    for x in 0..4 {
        dct8_1d(&mut tmp[x..], 4, col_clip_min, col_clip_max);
    }

    // Add to destination
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi16(bitdepth_max as i16);

    for y in 0..8 {
        let dst_off = y * dst_stride;

        // Load destination (4 bytes)
        let d = loadi32!(&dst[dst_off..dst_off + 4]);
        let d16 = _mm_unpacklo_epi8(d, zero);
        let d32 = _mm_cvtepi16_epi32(d16);

        // Load and scale coefficients
        let c = _mm_set_epi32(
            (tmp[y * 4 + 3] + 8) >> 4,
            (tmp[y * 4 + 2] + 8) >> 4,
            (tmp[y * 4 + 1] + 8) >> 4,
            (tmp[y * 4 + 0] + 8) >> 4,
        );

        let sum = _mm_add_epi32(d32, c);
        let sum16 = _mm_packs_epi32(sum, sum);
        let clamped = _mm_max_epi16(_mm_min_epi16(sum16, max_val), zero);
        let packed = _mm_packus_epi16(clamped, clamped);

        storei32!(&mut dst[dst_off..dst_off + 4], packed);
    }

    // Clear coefficients
    coeff[..32].fill(0);
}

/// FFI wrapper for 4x8 DCT_DCT 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_4x8_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, _coeff_len as usize * stride + stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_4x8_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// Full 2D DCT_DCT 8x4 inverse transform
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_8x4_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    // W=8, H=4, shift=0 for 8x4
    let row_clip_min = i16::MIN as i32;
    let row_clip_max = i16::MAX as i32;
    let col_clip_min = i16::MIN as i32;
    let col_clip_max = i16::MAX as i32;
    let mut tmp = [0i32; 32];

    // is_rect2 = true for 8x4, so apply sqrt(2) scaling
    let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

    // Row transform (8 elements each, 4 rows)
    for y in 0..4 {
        // Load row from column-major with rect2 scaling
        let mut scratch = [0i32; 8];
        for x in 0..8 {
            scratch[x] = rect2_scale(coeff[y + x * 4] as i32);
        }
        dct8_1d(&mut scratch[..8], 1, row_clip_min, row_clip_max);
        // Store row-major (no intermediate shift for 8x4)
        for x in 0..8 {
            tmp[y * 8 + x] = iclip(scratch[x], col_clip_min, col_clip_max);
        }
    }

    // Column transform (4 elements each, 8 columns)
    for x in 0..8 {
        dct4_1d(&mut tmp[x..], 8, col_clip_min, col_clip_max);
    }

    // Add to destination
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi16(bitdepth_max as i16);
    let rnd_final = _mm256_set1_epi32(8);

    for y in 0..4 {
        let dst_off = y * dst_stride;

        // Load destination (8 bytes)
        let d = loadi64!(&dst[dst_off..dst_off + 8]);
        let d16 = _mm_unpacklo_epi8(d, zero);

        // Load and scale coefficients
        let c_lo = _mm_set_epi32(
            tmp[y * 8 + 3],
            tmp[y * 8 + 2],
            tmp[y * 8 + 1],
            tmp[y * 8 + 0],
        );
        let c_hi = _mm_set_epi32(
            tmp[y * 8 + 7],
            tmp[y * 8 + 6],
            tmp[y * 8 + 5],
            tmp[y * 8 + 4],
        );

        let c_lo_256 = _mm256_set_m128i(c_hi, c_lo);
        let c_scaled = _mm256_srai_epi32(_mm256_add_epi32(c_lo_256, rnd_final), 4);

        let c_lo_scaled = _mm256_castsi256_si128(c_scaled);
        let c_hi_scaled = _mm256_extracti128_si256(c_scaled, 1);
        let c16 = _mm_packs_epi32(c_lo_scaled, c_hi_scaled);

        let sum = _mm_add_epi16(d16, c16);
        let clamped = _mm_max_epi16(_mm_min_epi16(sum, max_val), zero);
        let packed = _mm_packus_epi16(clamped, clamped);

        storei64!(&mut dst[dst_off..dst_off + 8], packed);
    }

    // Clear coefficients
    coeff[..32].fill(0);
}

/// FFI wrapper for 8x4 DCT_DCT 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_8x4_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, _coeff_len as usize * stride + stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_8x4_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// ADST VARIANTS FOR 4x8 and 8x4
// ============================================================================

/// Helper macro for 4x8 transforms with configurable row/col transforms
macro_rules! impl_4x8_transform {
    ($name:ident, $row_fn:ident, $col_fn:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[arcane]
        fn $name(
            _token: Desktop64,
            dst: &mut [u8],
            dst_stride: usize,
            coeff: &mut [i16],
            _eob: i32,
            bitdepth_max: i32,
        ) {
            use crate::src::safe_simd::pixel_access::{
                loadi32, loadi64, loadu_128, storei32, storei64, storeu_128,
            };
            let mut dst = dst.flex_mut();
            let mut coeff = coeff.flex_mut();
            let row_clip_min = i16::MIN as i32;
            let row_clip_max = i16::MAX as i32;
            let col_clip_min = i16::MIN as i32;
            let col_clip_max = i16::MAX as i32;
            let mut tmp = [0i32; 32];

            let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

            // Row transform (4 elements each, 8 rows)
            for y in 0..8 {
                let mut scratch = [0i32; 4];
                for x in 0..4 {
                    scratch[x] = rect2_scale(coeff[y + x * 8] as i32);
                }
                $row_fn(&mut scratch[..4], 1, row_clip_min, row_clip_max);
                for x in 0..4 {
                    tmp[y * 4 + x] = iclip(scratch[x], col_clip_min, col_clip_max);
                }
            }

            // Column transform (8 elements each, 4 columns)
            for x in 0..4 {
                $col_fn(&mut tmp[x..], 4, col_clip_min, col_clip_max);
            }

            // Add to destination
            let zero = _mm_setzero_si128();
            let max_val = _mm_set1_epi16(bitdepth_max as i16);

            for y in 0..8 {
                let dst_off = y * dst_stride;
                let d = loadi32!(&dst[dst_off..dst_off + 4]);
                let d16 = _mm_unpacklo_epi8(d, zero);
                let d32 = _mm_cvtepi16_epi32(d16);

                let c = _mm_set_epi32(
                    (tmp[y * 4 + 3] + 8) >> 4,
                    (tmp[y * 4 + 2] + 8) >> 4,
                    (tmp[y * 4 + 1] + 8) >> 4,
                    (tmp[y * 4 + 0] + 8) >> 4,
                );

                let sum = _mm_add_epi32(d32, c);
                let sum16 = _mm_packs_epi32(sum, sum);
                let clamped = _mm_max_epi16(_mm_min_epi16(sum16, max_val), zero);
                let packed = _mm_packus_epi16(clamped, clamped);
                storei32!(&mut dst[dst_off..dst_off + 4], packed);
            }

            coeff[..32].fill(0);
        }
    };
}

/// Helper macro for 8x4 transforms with configurable row/col transforms
macro_rules! impl_8x4_transform {
    ($name:ident, $row_fn:ident, $col_fn:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[arcane]
        fn $name(
            _token: Desktop64,
            dst: &mut [u8],
            dst_stride: usize,
            coeff: &mut [i16],
            _eob: i32,
            bitdepth_max: i32,
        ) {
            use crate::src::safe_simd::pixel_access::{
                loadi32, loadi64, loadu_128, storei32, storei64, storeu_128,
            };
            let mut dst = dst.flex_mut();
            let mut coeff = coeff.flex_mut();
            let row_clip_min = i16::MIN as i32;
            let row_clip_max = i16::MAX as i32;
            let col_clip_min = i16::MIN as i32;
            let col_clip_max = i16::MAX as i32;
            let mut tmp = [0i32; 32];

            let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

            // Row transform (8 elements each, 4 rows)
            for y in 0..4 {
                let mut scratch = [0i32; 8];
                for x in 0..8 {
                    scratch[x] = rect2_scale(coeff[y + x * 4] as i32);
                }
                $row_fn(&mut scratch[..8], 1, row_clip_min, row_clip_max);
                for x in 0..8 {
                    tmp[y * 8 + x] = iclip(scratch[x], col_clip_min, col_clip_max);
                }
            }

            // Column transform (4 elements each, 8 columns)
            for x in 0..8 {
                $col_fn(&mut tmp[x..], 8, col_clip_min, col_clip_max);
            }

            // Add to destination
            for y in 0..4 {
                let dst_off = y * dst_stride;
                for x in 0..8 {
                    let d = dst[dst_off + x] as i32;
                    let c = (tmp[y * 8 + x] + 8) >> 4;
                    let result = iclip(d + c, 0, bitdepth_max);
                    dst[dst_off + x] = result as u8;
                }
            }

            coeff[..32].fill(0);
        }
    };
}

// Generate 4x8 ADST variants
impl_4x8_transform!(inv_txfm_add_adst_dct_4x8_8bpc_avx2_inner, adst4_1d, dct8_1d);
impl_4x8_transform!(inv_txfm_add_dct_adst_4x8_8bpc_avx2_inner, dct4_1d, adst8_1d);
impl_4x8_transform!(
    inv_txfm_add_adst_adst_4x8_8bpc_avx2_inner,
    adst4_1d,
    adst8_1d
);
impl_4x8_transform!(
    inv_txfm_add_flipadst_dct_4x8_8bpc_avx2_inner,
    flipadst4_1d,
    dct8_1d
);
impl_4x8_transform!(
    inv_txfm_add_dct_flipadst_4x8_8bpc_avx2_inner,
    dct4_1d,
    flipadst8_1d
);
impl_4x8_transform!(
    inv_txfm_add_flipadst_flipadst_4x8_8bpc_avx2_inner,
    flipadst4_1d,
    flipadst8_1d
);
impl_4x8_transform!(
    inv_txfm_add_adst_flipadst_4x8_8bpc_avx2_inner,
    adst4_1d,
    flipadst8_1d
);
impl_4x8_transform!(
    inv_txfm_add_flipadst_adst_4x8_8bpc_avx2_inner,
    flipadst4_1d,
    adst8_1d
);

// Generate 8x4 ADST variants
impl_8x4_transform!(inv_txfm_add_adst_dct_8x4_8bpc_avx2_inner, adst8_1d, dct4_1d);
impl_8x4_transform!(inv_txfm_add_dct_adst_8x4_8bpc_avx2_inner, dct8_1d, adst4_1d);
impl_8x4_transform!(
    inv_txfm_add_adst_adst_8x4_8bpc_avx2_inner,
    adst8_1d,
    adst4_1d
);
impl_8x4_transform!(
    inv_txfm_add_flipadst_dct_8x4_8bpc_avx2_inner,
    flipadst8_1d,
    dct4_1d
);
impl_8x4_transform!(
    inv_txfm_add_dct_flipadst_8x4_8bpc_avx2_inner,
    dct8_1d,
    flipadst4_1d
);
impl_8x4_transform!(
    inv_txfm_add_flipadst_flipadst_8x4_8bpc_avx2_inner,
    flipadst8_1d,
    flipadst4_1d
);
impl_8x4_transform!(
    inv_txfm_add_adst_flipadst_8x4_8bpc_avx2_inner,
    adst8_1d,
    flipadst4_1d
);
impl_8x4_transform!(
    inv_txfm_add_flipadst_adst_8x4_8bpc_avx2_inner,
    flipadst8_1d,
    adst4_1d
);

// FFI wrappers for 4x8 ADST variants
macro_rules! impl_4x8_ffi_wrapper {
    ($name:ident, $inner:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2")]
        #[cfg(feature = "asm")]
        pub unsafe extern "C" fn $name(
            dst_ptr: *mut DynPixel,
            dst_stride: isize,
            coeff: *mut DynCoef,
            eob: c_int,
            bitdepth_max: c_int,
            _coeff_len: u16,
            _dst: *const FFISafe<PicOffset>,
        ) {
            let _token = unsafe { Desktop64::forge_token_dangerously() };
            let stride = dst_stride as usize;

            let dst_slice = unsafe {
                std::slice::from_raw_parts_mut(
                    dst_ptr as *mut u8,
                    _coeff_len as usize * stride + stride,
                )
            };

            let coeff_slice =
                unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

            $inner(_token, dst_slice, stride, coeff_slice, eob, bitdepth_max);
        }
    };
}

impl_4x8_ffi_wrapper!(
    inv_txfm_add_adst_dct_4x8_8bpc_avx2,
    inv_txfm_add_adst_dct_4x8_8bpc_avx2_inner
);
impl_4x8_ffi_wrapper!(
    inv_txfm_add_dct_adst_4x8_8bpc_avx2,
    inv_txfm_add_dct_adst_4x8_8bpc_avx2_inner
);
impl_4x8_ffi_wrapper!(
    inv_txfm_add_adst_adst_4x8_8bpc_avx2,
    inv_txfm_add_adst_adst_4x8_8bpc_avx2_inner
);
impl_4x8_ffi_wrapper!(
    inv_txfm_add_flipadst_dct_4x8_8bpc_avx2,
    inv_txfm_add_flipadst_dct_4x8_8bpc_avx2_inner
);
impl_4x8_ffi_wrapper!(
    inv_txfm_add_dct_flipadst_4x8_8bpc_avx2,
    inv_txfm_add_dct_flipadst_4x8_8bpc_avx2_inner
);
impl_4x8_ffi_wrapper!(
    inv_txfm_add_flipadst_flipadst_4x8_8bpc_avx2,
    inv_txfm_add_flipadst_flipadst_4x8_8bpc_avx2_inner
);
impl_4x8_ffi_wrapper!(
    inv_txfm_add_adst_flipadst_4x8_8bpc_avx2,
    inv_txfm_add_adst_flipadst_4x8_8bpc_avx2_inner
);
impl_4x8_ffi_wrapper!(
    inv_txfm_add_flipadst_adst_4x8_8bpc_avx2,
    inv_txfm_add_flipadst_adst_4x8_8bpc_avx2_inner
);

// FFI wrappers for 8x4 ADST variants
macro_rules! impl_8x4_ffi_wrapper {
    ($name:ident, $inner:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2")]
        #[cfg(feature = "asm")]
        pub unsafe extern "C" fn $name(
            dst_ptr: *mut DynPixel,
            dst_stride: isize,
            coeff: *mut DynCoef,
            eob: c_int,
            bitdepth_max: c_int,
            _coeff_len: u16,
            _dst: *const FFISafe<PicOffset>,
        ) {
            let _token = unsafe { Desktop64::forge_token_dangerously() };
            let stride = dst_stride as usize;

            let dst_slice = unsafe {
                std::slice::from_raw_parts_mut(
                    dst_ptr as *mut u8,
                    _coeff_len as usize * stride + stride,
                )
            };

            let coeff_slice =
                unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

            $inner(_token, dst_slice, stride, coeff_slice, eob, bitdepth_max);
        }
    };
}

impl_8x4_ffi_wrapper!(
    inv_txfm_add_adst_dct_8x4_8bpc_avx2,
    inv_txfm_add_adst_dct_8x4_8bpc_avx2_inner
);
impl_8x4_ffi_wrapper!(
    inv_txfm_add_dct_adst_8x4_8bpc_avx2,
    inv_txfm_add_dct_adst_8x4_8bpc_avx2_inner
);
impl_8x4_ffi_wrapper!(
    inv_txfm_add_adst_adst_8x4_8bpc_avx2,
    inv_txfm_add_adst_adst_8x4_8bpc_avx2_inner
);
impl_8x4_ffi_wrapper!(
    inv_txfm_add_flipadst_dct_8x4_8bpc_avx2,
    inv_txfm_add_flipadst_dct_8x4_8bpc_avx2_inner
);
impl_8x4_ffi_wrapper!(
    inv_txfm_add_dct_flipadst_8x4_8bpc_avx2,
    inv_txfm_add_dct_flipadst_8x4_8bpc_avx2_inner
);
impl_8x4_ffi_wrapper!(
    inv_txfm_add_flipadst_flipadst_8x4_8bpc_avx2,
    inv_txfm_add_flipadst_flipadst_8x4_8bpc_avx2_inner
);
impl_8x4_ffi_wrapper!(
    inv_txfm_add_adst_flipadst_8x4_8bpc_avx2,
    inv_txfm_add_adst_flipadst_8x4_8bpc_avx2_inner
);
impl_8x4_ffi_wrapper!(
    inv_txfm_add_flipadst_adst_8x4_8bpc_avx2,
    inv_txfm_add_flipadst_adst_8x4_8bpc_avx2_inner
);

// IDTX for 4x8 and 8x4
impl_4x8_transform!(
    inv_txfm_add_identity_identity_4x8_8bpc_avx2_inner,
    identity4_1d,
    identity8_1d
);
impl_8x4_transform!(
    inv_txfm_add_identity_identity_8x4_8bpc_avx2_inner,
    identity8_1d,
    identity4_1d
);
impl_4x8_ffi_wrapper!(
    inv_txfm_add_identity_identity_4x8_8bpc_avx2,
    inv_txfm_add_identity_identity_4x8_8bpc_avx2_inner
);
impl_8x4_ffi_wrapper!(
    inv_txfm_add_identity_identity_8x4_8bpc_avx2,
    inv_txfm_add_identity_identity_8x4_8bpc_avx2_inner
);

// H_DCT and V_DCT for 4x8 (identity+dct mixes)
impl_4x8_transform!(
    inv_txfm_add_identity_dct_4x8_8bpc_avx2_inner,
    identity4_1d,
    dct8_1d
);
impl_4x8_transform!(
    inv_txfm_add_dct_identity_4x8_8bpc_avx2_inner,
    dct4_1d,
    identity8_1d
);
impl_4x8_ffi_wrapper!(
    inv_txfm_add_identity_dct_4x8_8bpc_avx2,
    inv_txfm_add_identity_dct_4x8_8bpc_avx2_inner
);
impl_4x8_ffi_wrapper!(
    inv_txfm_add_dct_identity_4x8_8bpc_avx2,
    inv_txfm_add_dct_identity_4x8_8bpc_avx2_inner
);

// H_DCT and V_DCT for 8x4
impl_8x4_transform!(
    inv_txfm_add_identity_dct_8x4_8bpc_avx2_inner,
    identity8_1d,
    dct4_1d
);
impl_8x4_transform!(
    inv_txfm_add_dct_identity_8x4_8bpc_avx2_inner,
    dct8_1d,
    identity4_1d
);
impl_8x4_ffi_wrapper!(
    inv_txfm_add_identity_dct_8x4_8bpc_avx2,
    inv_txfm_add_identity_dct_8x4_8bpc_avx2_inner
);
impl_8x4_ffi_wrapper!(
    inv_txfm_add_dct_identity_8x4_8bpc_avx2,
    inv_txfm_add_dct_identity_8x4_8bpc_avx2_inner
);

// H_ADST, V_ADST, H_FLIPADST, V_FLIPADST for 4x8
impl_4x8_transform!(
    inv_txfm_add_identity_adst_4x8_8bpc_avx2_inner,
    identity4_1d,
    adst8_1d
);
impl_4x8_transform!(
    inv_txfm_add_adst_identity_4x8_8bpc_avx2_inner,
    adst4_1d,
    identity8_1d
);
impl_4x8_transform!(
    inv_txfm_add_identity_flipadst_4x8_8bpc_avx2_inner,
    identity4_1d,
    flipadst8_1d
);
impl_4x8_transform!(
    inv_txfm_add_flipadst_identity_4x8_8bpc_avx2_inner,
    flipadst4_1d,
    identity8_1d
);
impl_4x8_ffi_wrapper!(
    inv_txfm_add_identity_adst_4x8_8bpc_avx2,
    inv_txfm_add_identity_adst_4x8_8bpc_avx2_inner
);
impl_4x8_ffi_wrapper!(
    inv_txfm_add_adst_identity_4x8_8bpc_avx2,
    inv_txfm_add_adst_identity_4x8_8bpc_avx2_inner
);
impl_4x8_ffi_wrapper!(
    inv_txfm_add_identity_flipadst_4x8_8bpc_avx2,
    inv_txfm_add_identity_flipadst_4x8_8bpc_avx2_inner
);
impl_4x8_ffi_wrapper!(
    inv_txfm_add_flipadst_identity_4x8_8bpc_avx2,
    inv_txfm_add_flipadst_identity_4x8_8bpc_avx2_inner
);

// H_ADST, V_ADST, H_FLIPADST, V_FLIPADST for 8x4
impl_8x4_transform!(
    inv_txfm_add_identity_adst_8x4_8bpc_avx2_inner,
    identity8_1d,
    adst4_1d
);
impl_8x4_transform!(
    inv_txfm_add_adst_identity_8x4_8bpc_avx2_inner,
    adst8_1d,
    identity4_1d
);
impl_8x4_transform!(
    inv_txfm_add_identity_flipadst_8x4_8bpc_avx2_inner,
    identity8_1d,
    flipadst4_1d
);
impl_8x4_transform!(
    inv_txfm_add_flipadst_identity_8x4_8bpc_avx2_inner,
    flipadst8_1d,
    identity4_1d
);
impl_8x4_ffi_wrapper!(
    inv_txfm_add_identity_adst_8x4_8bpc_avx2,
    inv_txfm_add_identity_adst_8x4_8bpc_avx2_inner
);
impl_8x4_ffi_wrapper!(
    inv_txfm_add_adst_identity_8x4_8bpc_avx2,
    inv_txfm_add_adst_identity_8x4_8bpc_avx2_inner
);
impl_8x4_ffi_wrapper!(
    inv_txfm_add_identity_flipadst_8x4_8bpc_avx2,
    inv_txfm_add_identity_flipadst_8x4_8bpc_avx2_inner
);
impl_8x4_ffi_wrapper!(
    inv_txfm_add_flipadst_identity_8x4_8bpc_avx2,
    inv_txfm_add_flipadst_identity_8x4_8bpc_avx2_inner
);

// ============================================================================
// 8x16 and 16x8 ADST/FLIPADST variants
// ============================================================================

/// Helper macro for 8x16 transforms with configurable row/col transforms
macro_rules! impl_8x16_transform {
    ($name:ident, $row_fn:ident, $col_fn:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[arcane]
        fn $name(
            _token: Desktop64,
            dst: &mut [u8],
            dst_stride: usize,
            coeff: &mut [i16],
            _eob: i32,
            bitdepth_max: i32,
        ) {
            use crate::src::safe_simd::pixel_access::{
                loadi32, loadi64, loadu_128, storei32, storei64, storeu_128,
            };
            let mut dst = dst.flex_mut();
            let mut coeff = coeff.flex_mut();
            let row_clip_min = i16::MIN as i32;
            let row_clip_max = i16::MAX as i32;
            let col_clip_min = i16::MIN as i32;
            let col_clip_max = i16::MAX as i32;
            let mut tmp = [0i32; 128];

            let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

            // Row transform (8 elements each, 16 rows)
            let rnd = 1;
            let shift = 1;
            for y in 0..16 {
                let mut scratch = [0i32; 8];
                for x in 0..8 {
                    scratch[x] = rect2_scale(coeff[y + x * 16] as i32);
                }
                $row_fn(&mut scratch[..8], 1, row_clip_min, row_clip_max);
                for x in 0..8 {
                    tmp[y * 8 + x] = iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
                }
            }

            // Column transform (16 elements each, 8 columns)
            for x in 0..8 {
                $col_fn(&mut tmp[x..], 8, col_clip_min, col_clip_max);
            }

            // Add to destination
            let zero = _mm_setzero_si128();
            let max_val = _mm_set1_epi16(bitdepth_max as i16);
            let rnd_final = _mm256_set1_epi32(8);

            for y in 0..16 {
                let dst_off = y * dst_stride;

                let d = loadi64!(&dst[dst_off..dst_off + 8]);
                let d16 = _mm_unpacklo_epi8(d, zero);

                let c_lo = _mm_set_epi32(
                    tmp[y * 8 + 3],
                    tmp[y * 8 + 2],
                    tmp[y * 8 + 1],
                    tmp[y * 8 + 0],
                );
                let c_hi = _mm_set_epi32(
                    tmp[y * 8 + 7],
                    tmp[y * 8 + 6],
                    tmp[y * 8 + 5],
                    tmp[y * 8 + 4],
                );

                let c_lo_256 = _mm256_set_m128i(c_hi, c_lo);
                let c_scaled = _mm256_srai_epi32(_mm256_add_epi32(c_lo_256, rnd_final), 4);

                let c_lo_scaled = _mm256_castsi256_si128(c_scaled);
                let c_hi_scaled = _mm256_extracti128_si256(c_scaled, 1);
                let c16 = _mm_packs_epi32(c_lo_scaled, c_hi_scaled);

                let sum = _mm_add_epi16(d16, c16);
                let clamped = _mm_max_epi16(_mm_min_epi16(sum, max_val), zero);
                let packed = _mm_packus_epi16(clamped, clamped);

                storei64!(&mut dst[dst_off..dst_off + 8], packed);
            }

            // Clear coefficients
            coeff[..128].fill(0);
        }
    };
}

/// Helper macro for 16x8 transforms with configurable row/col transforms
macro_rules! impl_16x8_transform {
    ($name:ident, $row_fn:ident, $col_fn:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[arcane]
        fn $name(
            _token: Desktop64,
            dst: &mut [u8],
            dst_stride: usize,
            coeff: &mut [i16],
            _eob: i32,
            bitdepth_max: i32,
        ) {
            use crate::src::safe_simd::pixel_access::{
                loadi32, loadi64, loadu_128, storei32, storei64, storeu_128,
            };
            let mut dst = dst.flex_mut();
            let mut coeff = coeff.flex_mut();
            let row_clip_min = i16::MIN as i32;
            let row_clip_max = i16::MAX as i32;
            let col_clip_min = i16::MIN as i32;
            let col_clip_max = i16::MAX as i32;
            let mut tmp = [0i32; 128];

            let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

            // Row transform (16 elements each, 8 rows)
            let rnd = 1;
            let shift = 1;
            for y in 0..8 {
                let mut scratch = [0i32; 16];
                for x in 0..16 {
                    scratch[x] = rect2_scale(coeff[y + x * 8] as i32);
                }
                $row_fn(&mut scratch[..16], 1, row_clip_min, row_clip_max);
                for x in 0..16 {
                    tmp[y * 16 + x] = iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
                }
            }

            // Column transform (8 elements each, 16 columns)
            for x in 0..16 {
                $col_fn(&mut tmp[x..], 16, col_clip_min, col_clip_max);
            }

            // Add to destination
            let zero = _mm256_setzero_si256();
            let max_val = _mm256_set1_epi16(bitdepth_max as i16);
            let rnd_final = _mm256_set1_epi32(8);

            for y in 0..8 {
                let dst_off = y * dst_stride;

                let d = loadu_128!(<&[u8; 16]>::try_from(&dst[dst_off..dst_off + 16]).unwrap());
                let d16 = _mm256_cvtepu8_epi16(d);

                let c0 = _mm256_set_epi32(
                    tmp[y * 16 + 7],
                    tmp[y * 16 + 6],
                    tmp[y * 16 + 5],
                    tmp[y * 16 + 4],
                    tmp[y * 16 + 3],
                    tmp[y * 16 + 2],
                    tmp[y * 16 + 1],
                    tmp[y * 16 + 0],
                );
                let c1 = _mm256_set_epi32(
                    tmp[y * 16 + 15],
                    tmp[y * 16 + 14],
                    tmp[y * 16 + 13],
                    tmp[y * 16 + 12],
                    tmp[y * 16 + 11],
                    tmp[y * 16 + 10],
                    tmp[y * 16 + 9],
                    tmp[y * 16 + 8],
                );

                let c0_scaled = _mm256_srai_epi32(_mm256_add_epi32(c0, rnd_final), 4);
                let c1_scaled = _mm256_srai_epi32(_mm256_add_epi32(c1, rnd_final), 4);

                let c16 = _mm256_packs_epi32(c0_scaled, c1_scaled);
                let c16 = _mm256_permute4x64_epi64(c16, 0b11_01_10_00);

                let sum = _mm256_add_epi16(d16, c16);
                let clamped = _mm256_max_epi16(_mm256_min_epi16(sum, max_val), zero);

                let packed = _mm256_packus_epi16(clamped, clamped);
                let packed = _mm256_permute4x64_epi64(packed, 0b11_01_10_00);

                storeu_128!(
                    <&mut [u8; 16]>::try_from(&mut dst[dst_off..dst_off + 16]).unwrap(),
                    _mm256_castsi256_si128(packed)
                );
            }

            // Clear coefficients
            coeff[..128].fill(0);
        }
    };
}

// Generate 8x16 ADST inner functions
impl_8x16_transform!(
    inv_txfm_add_adst_dct_8x16_8bpc_avx2_inner,
    adst8_1d,
    dct16_1d
);
impl_8x16_transform!(
    inv_txfm_add_dct_adst_8x16_8bpc_avx2_inner,
    dct8_1d,
    adst16_1d
);
impl_8x16_transform!(
    inv_txfm_add_adst_adst_8x16_8bpc_avx2_inner,
    adst8_1d,
    adst16_1d
);
impl_8x16_transform!(
    inv_txfm_add_flipadst_dct_8x16_8bpc_avx2_inner,
    flipadst8_1d,
    dct16_1d
);
impl_8x16_transform!(
    inv_txfm_add_dct_flipadst_8x16_8bpc_avx2_inner,
    dct8_1d,
    flipadst16_1d
);
impl_8x16_transform!(
    inv_txfm_add_flipadst_flipadst_8x16_8bpc_avx2_inner,
    flipadst8_1d,
    flipadst16_1d
);
impl_8x16_transform!(
    inv_txfm_add_adst_flipadst_8x16_8bpc_avx2_inner,
    adst8_1d,
    flipadst16_1d
);
impl_8x16_transform!(
    inv_txfm_add_flipadst_adst_8x16_8bpc_avx2_inner,
    flipadst8_1d,
    adst16_1d
);

// Generate 16x8 ADST inner functions
impl_16x8_transform!(
    inv_txfm_add_adst_dct_16x8_8bpc_avx2_inner,
    adst16_1d,
    dct8_1d
);
impl_16x8_transform!(
    inv_txfm_add_dct_adst_16x8_8bpc_avx2_inner,
    dct16_1d,
    adst8_1d
);
impl_16x8_transform!(
    inv_txfm_add_adst_adst_16x8_8bpc_avx2_inner,
    adst16_1d,
    adst8_1d
);
impl_16x8_transform!(
    inv_txfm_add_flipadst_dct_16x8_8bpc_avx2_inner,
    flipadst16_1d,
    dct8_1d
);
impl_16x8_transform!(
    inv_txfm_add_dct_flipadst_16x8_8bpc_avx2_inner,
    dct16_1d,
    flipadst8_1d
);
impl_16x8_transform!(
    inv_txfm_add_flipadst_flipadst_16x8_8bpc_avx2_inner,
    flipadst16_1d,
    flipadst8_1d
);
impl_16x8_transform!(
    inv_txfm_add_adst_flipadst_16x8_8bpc_avx2_inner,
    adst16_1d,
    flipadst8_1d
);
impl_16x8_transform!(
    inv_txfm_add_flipadst_adst_16x8_8bpc_avx2_inner,
    flipadst16_1d,
    adst8_1d
);

/// FFI wrapper macro for 8x16 transforms
macro_rules! impl_8x16_ffi_wrapper {
    ($name:ident, $inner:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2")]
        #[cfg(feature = "asm")]
        pub unsafe extern "C" fn $name(
            dst_ptr: *mut DynPixel,
            dst_stride: isize,
            coeff: *mut DynCoef,
            eob: c_int,
            bitdepth_max: c_int,
            _coeff_len: u16,
            _dst: *const FFISafe<PicOffset>,
        ) {
            let _token = unsafe { Desktop64::forge_token_dangerously() };
            let stride = dst_stride as usize;

            let dst_slice = unsafe {
                std::slice::from_raw_parts_mut(
                    dst_ptr as *mut u8,
                    _coeff_len as usize * stride + stride,
                )
            };

            let coeff_slice =
                unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

            $inner(_token, dst_slice, stride, coeff_slice, eob, bitdepth_max);
        }
    };
}

/// FFI wrapper macro for 16x8 transforms
macro_rules! impl_16x8_ffi_wrapper {
    ($name:ident, $inner:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2")]
        #[cfg(feature = "asm")]
        pub unsafe extern "C" fn $name(
            dst_ptr: *mut DynPixel,
            dst_stride: isize,
            coeff: *mut DynCoef,
            eob: c_int,
            bitdepth_max: c_int,
            _coeff_len: u16,
            _dst: *const FFISafe<PicOffset>,
        ) {
            let _token = unsafe { Desktop64::forge_token_dangerously() };
            let stride = dst_stride as usize;

            let dst_slice = unsafe {
                std::slice::from_raw_parts_mut(
                    dst_ptr as *mut u8,
                    _coeff_len as usize * stride + stride,
                )
            };

            let coeff_slice =
                unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

            $inner(_token, dst_slice, stride, coeff_slice, eob, bitdepth_max);
        }
    };
}

// Generate 8x16 FFI wrappers
impl_8x16_ffi_wrapper!(
    inv_txfm_add_adst_dct_8x16_8bpc_avx2,
    inv_txfm_add_adst_dct_8x16_8bpc_avx2_inner
);
impl_8x16_ffi_wrapper!(
    inv_txfm_add_dct_adst_8x16_8bpc_avx2,
    inv_txfm_add_dct_adst_8x16_8bpc_avx2_inner
);
impl_8x16_ffi_wrapper!(
    inv_txfm_add_adst_adst_8x16_8bpc_avx2,
    inv_txfm_add_adst_adst_8x16_8bpc_avx2_inner
);
impl_8x16_ffi_wrapper!(
    inv_txfm_add_flipadst_dct_8x16_8bpc_avx2,
    inv_txfm_add_flipadst_dct_8x16_8bpc_avx2_inner
);
impl_8x16_ffi_wrapper!(
    inv_txfm_add_dct_flipadst_8x16_8bpc_avx2,
    inv_txfm_add_dct_flipadst_8x16_8bpc_avx2_inner
);
impl_8x16_ffi_wrapper!(
    inv_txfm_add_flipadst_flipadst_8x16_8bpc_avx2,
    inv_txfm_add_flipadst_flipadst_8x16_8bpc_avx2_inner
);
impl_8x16_ffi_wrapper!(
    inv_txfm_add_adst_flipadst_8x16_8bpc_avx2,
    inv_txfm_add_adst_flipadst_8x16_8bpc_avx2_inner
);
impl_8x16_ffi_wrapper!(
    inv_txfm_add_flipadst_adst_8x16_8bpc_avx2,
    inv_txfm_add_flipadst_adst_8x16_8bpc_avx2_inner
);

// Generate 16x8 FFI wrappers
impl_16x8_ffi_wrapper!(
    inv_txfm_add_adst_dct_16x8_8bpc_avx2,
    inv_txfm_add_adst_dct_16x8_8bpc_avx2_inner
);
impl_16x8_ffi_wrapper!(
    inv_txfm_add_dct_adst_16x8_8bpc_avx2,
    inv_txfm_add_dct_adst_16x8_8bpc_avx2_inner
);
impl_16x8_ffi_wrapper!(
    inv_txfm_add_adst_adst_16x8_8bpc_avx2,
    inv_txfm_add_adst_adst_16x8_8bpc_avx2_inner
);
impl_16x8_ffi_wrapper!(
    inv_txfm_add_flipadst_dct_16x8_8bpc_avx2,
    inv_txfm_add_flipadst_dct_16x8_8bpc_avx2_inner
);
impl_16x8_ffi_wrapper!(
    inv_txfm_add_dct_flipadst_16x8_8bpc_avx2,
    inv_txfm_add_dct_flipadst_16x8_8bpc_avx2_inner
);
impl_16x8_ffi_wrapper!(
    inv_txfm_add_flipadst_flipadst_16x8_8bpc_avx2,
    inv_txfm_add_flipadst_flipadst_16x8_8bpc_avx2_inner
);
impl_16x8_ffi_wrapper!(
    inv_txfm_add_adst_flipadst_16x8_8bpc_avx2,
    inv_txfm_add_adst_flipadst_16x8_8bpc_avx2_inner
);
impl_16x8_ffi_wrapper!(
    inv_txfm_add_flipadst_adst_16x8_8bpc_avx2,
    inv_txfm_add_flipadst_adst_16x8_8bpc_avx2_inner
);

// IDTX for 8x16 and 16x8
impl_8x16_transform!(
    inv_txfm_add_identity_identity_8x16_8bpc_avx2_inner,
    identity8_1d,
    identity16_1d
);
impl_16x8_transform!(
    inv_txfm_add_identity_identity_16x8_8bpc_avx2_inner,
    identity16_1d,
    identity8_1d
);
impl_8x16_ffi_wrapper!(
    inv_txfm_add_identity_identity_8x16_8bpc_avx2,
    inv_txfm_add_identity_identity_8x16_8bpc_avx2_inner
);
impl_16x8_ffi_wrapper!(
    inv_txfm_add_identity_identity_16x8_8bpc_avx2,
    inv_txfm_add_identity_identity_16x8_8bpc_avx2_inner
);

// H_DCT and V_DCT for 8x16
impl_8x16_transform!(
    inv_txfm_add_identity_dct_8x16_8bpc_avx2_inner,
    identity8_1d,
    dct16_1d
);
impl_8x16_transform!(
    inv_txfm_add_dct_identity_8x16_8bpc_avx2_inner,
    dct8_1d,
    identity16_1d
);
impl_8x16_ffi_wrapper!(
    inv_txfm_add_identity_dct_8x16_8bpc_avx2,
    inv_txfm_add_identity_dct_8x16_8bpc_avx2_inner
);
impl_8x16_ffi_wrapper!(
    inv_txfm_add_dct_identity_8x16_8bpc_avx2,
    inv_txfm_add_dct_identity_8x16_8bpc_avx2_inner
);

// H_DCT and V_DCT for 16x8
impl_16x8_transform!(
    inv_txfm_add_identity_dct_16x8_8bpc_avx2_inner,
    identity16_1d,
    dct8_1d
);
impl_16x8_transform!(
    inv_txfm_add_dct_identity_16x8_8bpc_avx2_inner,
    dct16_1d,
    identity8_1d
);
impl_16x8_ffi_wrapper!(
    inv_txfm_add_identity_dct_16x8_8bpc_avx2,
    inv_txfm_add_identity_dct_16x8_8bpc_avx2_inner
);
impl_16x8_ffi_wrapper!(
    inv_txfm_add_dct_identity_16x8_8bpc_avx2,
    inv_txfm_add_dct_identity_16x8_8bpc_avx2_inner
);

// H_ADST, V_ADST, H_FLIPADST, V_FLIPADST for 8x16
impl_8x16_transform!(
    inv_txfm_add_identity_adst_8x16_8bpc_avx2_inner,
    identity8_1d,
    adst16_1d
);
impl_8x16_transform!(
    inv_txfm_add_adst_identity_8x16_8bpc_avx2_inner,
    adst8_1d,
    identity16_1d
);
impl_8x16_transform!(
    inv_txfm_add_identity_flipadst_8x16_8bpc_avx2_inner,
    identity8_1d,
    flipadst16_1d
);
impl_8x16_transform!(
    inv_txfm_add_flipadst_identity_8x16_8bpc_avx2_inner,
    flipadst8_1d,
    identity16_1d
);
impl_8x16_ffi_wrapper!(
    inv_txfm_add_identity_adst_8x16_8bpc_avx2,
    inv_txfm_add_identity_adst_8x16_8bpc_avx2_inner
);
impl_8x16_ffi_wrapper!(
    inv_txfm_add_adst_identity_8x16_8bpc_avx2,
    inv_txfm_add_adst_identity_8x16_8bpc_avx2_inner
);
impl_8x16_ffi_wrapper!(
    inv_txfm_add_identity_flipadst_8x16_8bpc_avx2,
    inv_txfm_add_identity_flipadst_8x16_8bpc_avx2_inner
);
impl_8x16_ffi_wrapper!(
    inv_txfm_add_flipadst_identity_8x16_8bpc_avx2,
    inv_txfm_add_flipadst_identity_8x16_8bpc_avx2_inner
);

// H_ADST, V_ADST, H_FLIPADST, V_FLIPADST for 16x8
impl_16x8_transform!(
    inv_txfm_add_identity_adst_16x8_8bpc_avx2_inner,
    identity16_1d,
    adst8_1d
);
impl_16x8_transform!(
    inv_txfm_add_adst_identity_16x8_8bpc_avx2_inner,
    adst16_1d,
    identity8_1d
);
impl_16x8_transform!(
    inv_txfm_add_identity_flipadst_16x8_8bpc_avx2_inner,
    identity16_1d,
    flipadst8_1d
);
impl_16x8_transform!(
    inv_txfm_add_flipadst_identity_16x8_8bpc_avx2_inner,
    flipadst16_1d,
    identity8_1d
);
impl_16x8_ffi_wrapper!(
    inv_txfm_add_identity_adst_16x8_8bpc_avx2,
    inv_txfm_add_identity_adst_16x8_8bpc_avx2_inner
);
impl_16x8_ffi_wrapper!(
    inv_txfm_add_adst_identity_16x8_8bpc_avx2,
    inv_txfm_add_adst_identity_16x8_8bpc_avx2_inner
);
impl_16x8_ffi_wrapper!(
    inv_txfm_add_identity_flipadst_16x8_8bpc_avx2,
    inv_txfm_add_identity_flipadst_16x8_8bpc_avx2_inner
);
impl_16x8_ffi_wrapper!(
    inv_txfm_add_flipadst_identity_16x8_8bpc_avx2,
    inv_txfm_add_flipadst_identity_16x8_8bpc_avx2_inner
);

/// Full 2D DCT_DCT 8x8 inverse transform
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_8x16_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    // W=8, H=16, shift=1 for 8x16
    let row_clip_min = i16::MIN as i32;
    let row_clip_max = i16::MAX as i32;
    let col_clip_min = i16::MIN as i32;
    let col_clip_max = i16::MAX as i32;
    let mut tmp = [0i32; 128];

    // is_rect2 = true for 8x16
    let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

    // Row transform (8 elements each, 16 rows)
    let rnd = 1;
    let shift = 1;
    for y in 0..16 {
        let mut scratch = [0i32; 8];
        for x in 0..8 {
            scratch[x] = rect2_scale(coeff[y + x * 16] as i32);
        }
        dct8_1d(&mut scratch[..8], 1, row_clip_min, row_clip_max);
        for x in 0..8 {
            tmp[y * 8 + x] = iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
        }
    }

    // Column transform (16 elements each, 8 columns)
    for x in 0..8 {
        dct16_1d(&mut tmp[x..], 8, col_clip_min, col_clip_max);
    }

    // Add to destination
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi16(bitdepth_max as i16);
    let rnd_final = _mm256_set1_epi32(8);

    for y in 0..16 {
        let dst_off = y * dst_stride;

        let d = loadi64!(&dst[dst_off..dst_off + 8]);
        let d16 = _mm_unpacklo_epi8(d, zero);

        let c_lo = _mm_set_epi32(
            tmp[y * 8 + 3],
            tmp[y * 8 + 2],
            tmp[y * 8 + 1],
            tmp[y * 8 + 0],
        );
        let c_hi = _mm_set_epi32(
            tmp[y * 8 + 7],
            tmp[y * 8 + 6],
            tmp[y * 8 + 5],
            tmp[y * 8 + 4],
        );

        let c_lo_256 = _mm256_set_m128i(c_hi, c_lo);
        let c_scaled = _mm256_srai_epi32(_mm256_add_epi32(c_lo_256, rnd_final), 4);

        let c_lo_scaled = _mm256_castsi256_si128(c_scaled);
        let c_hi_scaled = _mm256_extracti128_si256(c_scaled, 1);
        let c16 = _mm_packs_epi32(c_lo_scaled, c_hi_scaled);

        let sum = _mm_add_epi16(d16, c16);
        let clamped = _mm_max_epi16(_mm_min_epi16(sum, max_val), zero);
        let packed = _mm_packus_epi16(clamped, clamped);

        storei64!(&mut dst[dst_off..dst_off + 8], packed);
    }

    // Clear coefficients
    coeff[..128].fill(0);
}

/// FFI wrapper for 8x16 DCT_DCT 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_8x16_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, _coeff_len as usize * stride + stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_8x16_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// Full 2D DCT_DCT 16x8 inverse transform
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_16x8_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    // W=16, H=8, shift=1 for 16x8
    let row_clip_min = i16::MIN as i32;
    let row_clip_max = i16::MAX as i32;
    let col_clip_min = i16::MIN as i32;
    let col_clip_max = i16::MAX as i32;
    let mut tmp = [0i32; 128];

    // is_rect2 = true for 16x8
    let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

    // Row transform (16 elements each, 8 rows)
    let rnd = 1;
    let shift = 1;
    for y in 0..8 {
        let mut scratch = [0i32; 16];
        for x in 0..16 {
            scratch[x] = rect2_scale(coeff[y + x * 8] as i32);
        }
        dct16_1d(&mut scratch[..16], 1, row_clip_min, row_clip_max);
        for x in 0..16 {
            tmp[y * 16 + x] = iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
        }
    }

    // Column transform (8 elements each, 16 columns)
    for x in 0..16 {
        dct8_1d(&mut tmp[x..], 16, col_clip_min, col_clip_max);
    }

    // Add to destination
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi16(bitdepth_max as i16);
    let rnd_final = _mm256_set1_epi32(8);

    for y in 0..8 {
        let dst_off = y * dst_stride;

        let d = loadu_128!(<&[u8; 16]>::try_from(&dst[dst_off..dst_off + 16]).unwrap());
        let d16 = _mm256_cvtepu8_epi16(d);

        let c0 = _mm256_set_epi32(
            tmp[y * 16 + 7],
            tmp[y * 16 + 6],
            tmp[y * 16 + 5],
            tmp[y * 16 + 4],
            tmp[y * 16 + 3],
            tmp[y * 16 + 2],
            tmp[y * 16 + 1],
            tmp[y * 16 + 0],
        );
        let c1 = _mm256_set_epi32(
            tmp[y * 16 + 15],
            tmp[y * 16 + 14],
            tmp[y * 16 + 13],
            tmp[y * 16 + 12],
            tmp[y * 16 + 11],
            tmp[y * 16 + 10],
            tmp[y * 16 + 9],
            tmp[y * 16 + 8],
        );

        let c0_scaled = _mm256_srai_epi32(_mm256_add_epi32(c0, rnd_final), 4);
        let c1_scaled = _mm256_srai_epi32(_mm256_add_epi32(c1, rnd_final), 4);

        let c16 = _mm256_packs_epi32(c0_scaled, c1_scaled);
        let c16 = _mm256_permute4x64_epi64(c16, 0b11_01_10_00);

        let sum = _mm256_add_epi16(d16, c16);
        let clamped = _mm256_max_epi16(_mm256_min_epi16(sum, max_val), zero);

        let packed = _mm256_packus_epi16(clamped, clamped);
        let packed = _mm256_permute4x64_epi64(packed, 0b11_01_10_00);

        storeu_128!(
            <&mut [u8; 16]>::try_from(&mut dst[dst_off..dst_off + 16]).unwrap(),
            _mm256_castsi256_si128(packed)
        );
    }

    // Clear coefficients
    coeff[..128].fill(0);
}

/// FFI wrapper for 16x8 DCT_DCT 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_16x8_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, _coeff_len as usize * stride + stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_16x8_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// RECTANGULAR TRANSFORMS - 16x32, 32x16
// ============================================================================

/// Full 2D DCT_DCT 16x32 inverse transform
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_16x32_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    // W=16, H=32, shift=2 for 16x32
    let row_clip_min = i16::MIN as i32;
    let row_clip_max = i16::MAX as i32;
    let col_clip_min = i16::MIN as i32;
    let col_clip_max = i16::MAX as i32;
    let mut tmp = [0i32; 16 * 32];

    // is_rect2 = true for 16x32
    let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

    // Row transform (16 elements each, 32 rows)
    let rnd = 1;
    let shift = 1;
    for y in 0..32 {
        let mut scratch = [0i32; 16];
        for x in 0..16 {
            scratch[x] = rect2_scale(coeff[y + x * 32] as i32);
        }
        dct16_1d(&mut scratch[..16], 1, row_clip_min, row_clip_max);
        for x in 0..16 {
            tmp[y * 16 + x] = iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
        }
    }

    // Column transform (32 elements each, 16 columns)
    for x in 0..16 {
        dct32_1d(&mut tmp[x..], 16, col_clip_min, col_clip_max);
    }

    // Add to destination
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi16(bitdepth_max as i16);
    let rnd_final = _mm256_set1_epi32(8);

    for y in 0..32 {
        let dst_off = y * dst_stride;

        let d = loadu_128!(<&[u8; 16]>::try_from(&dst[dst_off..dst_off + 16]).unwrap());
        let d16 = _mm256_cvtepu8_epi16(d);

        let c0 = _mm256_set_epi32(
            tmp[y * 16 + 7],
            tmp[y * 16 + 6],
            tmp[y * 16 + 5],
            tmp[y * 16 + 4],
            tmp[y * 16 + 3],
            tmp[y * 16 + 2],
            tmp[y * 16 + 1],
            tmp[y * 16 + 0],
        );
        let c1 = _mm256_set_epi32(
            tmp[y * 16 + 15],
            tmp[y * 16 + 14],
            tmp[y * 16 + 13],
            tmp[y * 16 + 12],
            tmp[y * 16 + 11],
            tmp[y * 16 + 10],
            tmp[y * 16 + 9],
            tmp[y * 16 + 8],
        );

        let c0_scaled = _mm256_srai_epi32(_mm256_add_epi32(c0, rnd_final), 4);
        let c1_scaled = _mm256_srai_epi32(_mm256_add_epi32(c1, rnd_final), 4);

        let c16 = _mm256_packs_epi32(c0_scaled, c1_scaled);
        let c16 = _mm256_permute4x64_epi64(c16, 0b11_01_10_00);

        let sum = _mm256_add_epi16(d16, c16);
        let clamped = _mm256_max_epi16(_mm256_min_epi16(sum, max_val), zero);

        let packed = _mm256_packus_epi16(clamped, clamped);
        let packed = _mm256_permute4x64_epi64(packed, 0b11_01_10_00);

        storeu_128!(
            <&mut [u8; 16]>::try_from(&mut dst[dst_off..dst_off + 16]).unwrap(),
            _mm256_castsi256_si128(packed)
        );
    }

    // Clear coefficients
    coeff[..512].fill(0);
}

/// FFI wrapper for 16x32 DCT_DCT 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_16x32_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, _coeff_len as usize * stride + stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_16x32_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// Full 2D DCT_DCT 32x16 inverse transform
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_32x16_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    // W=32, H=16, shift=2 for 32x16
    let row_clip_min = i16::MIN as i32;
    let row_clip_max = i16::MAX as i32;
    let col_clip_min = i16::MIN as i32;
    let col_clip_max = i16::MAX as i32;
    let mut tmp = [0i32; 32 * 16];

    // is_rect2 = true for 32x16
    let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

    // Row transform (32 elements each, 16 rows)
    let rnd = 1;
    let shift = 1;
    for y in 0..16 {
        let mut scratch = [0i32; 32];
        for x in 0..32 {
            scratch[x] = rect2_scale(coeff[y + x * 16] as i32);
        }
        dct32_1d(&mut scratch[..32], 1, row_clip_min, row_clip_max);
        for x in 0..32 {
            tmp[y * 32 + x] = iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
        }
    }

    // Column transform (16 elements each, 32 columns)
    for x in 0..32 {
        dct16_1d(&mut tmp[x..], 32, col_clip_min, col_clip_max);
    }

    // Add to destination
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi16(bitdepth_max as i16);
    let rnd_final = _mm256_set1_epi32(8);

    for y in 0..16 {
        let dst_off = y * dst_stride;

        // Process 32 pixels in two 16-pixel chunks
        for chunk in 0..2 {
            let chunk_off = chunk * 16;
            let d = loadu_128!(<&[u8; 16]>::try_from(
                &dst[dst_off + chunk_off..dst_off + chunk_off + 16]
            )
            .unwrap());
            let d16 = _mm256_cvtepu8_epi16(d);

            let c0 = _mm256_set_epi32(
                tmp[y * 32 + chunk_off + 7],
                tmp[y * 32 + chunk_off + 6],
                tmp[y * 32 + chunk_off + 5],
                tmp[y * 32 + chunk_off + 4],
                tmp[y * 32 + chunk_off + 3],
                tmp[y * 32 + chunk_off + 2],
                tmp[y * 32 + chunk_off + 1],
                tmp[y * 32 + chunk_off + 0],
            );
            let c1 = _mm256_set_epi32(
                tmp[y * 32 + chunk_off + 15],
                tmp[y * 32 + chunk_off + 14],
                tmp[y * 32 + chunk_off + 13],
                tmp[y * 32 + chunk_off + 12],
                tmp[y * 32 + chunk_off + 11],
                tmp[y * 32 + chunk_off + 10],
                tmp[y * 32 + chunk_off + 9],
                tmp[y * 32 + chunk_off + 8],
            );

            let c0_scaled = _mm256_srai_epi32(_mm256_add_epi32(c0, rnd_final), 4);
            let c1_scaled = _mm256_srai_epi32(_mm256_add_epi32(c1, rnd_final), 4);

            let c16 = _mm256_packs_epi32(c0_scaled, c1_scaled);
            let c16 = _mm256_permute4x64_epi64(c16, 0b11_01_10_00);

            let sum = _mm256_add_epi16(d16, c16);
            let clamped = _mm256_max_epi16(_mm256_min_epi16(sum, max_val), zero);

            let packed = _mm256_packus_epi16(clamped, clamped);
            let packed = _mm256_permute4x64_epi64(packed, 0b11_01_10_00);

            storeu_128!(
                <&mut [u8; 16]>::try_from(&mut dst[dst_off + chunk_off..dst_off + chunk_off + 16])
                    .unwrap(),
                _mm256_castsi256_si128(packed)
            );
        }
    }

    // Clear coefficients
    coeff[..512].fill(0);
}

/// FFI wrapper for 32x16 DCT_DCT 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_32x16_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, _coeff_len as usize * stride + stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_32x16_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// 16x32 and 32x16 IDTX transforms
// ============================================================================

/// 16x32 IDTX inverse transform
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_identity_identity_16x32_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let clip_min = i16::MIN as i32;
    let clip_max = i16::MAX as i32;
    let mut tmp = [0i32; 16 * 32];

    let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

    // Row transform (16 elements each, 32 rows)
    let rnd = 1;
    let shift = 1;
    for y in 0..32 {
        let mut scratch = [0i32; 16];
        for x in 0..16 {
            scratch[x] = rect2_scale(coeff[y + x * 32] as i32);
        }
        identity16_1d(&mut scratch[..16], 1, clip_min, clip_max);
        for x in 0..16 {
            tmp[y * 16 + x] = iclip((scratch[x] + rnd) >> shift, clip_min, clip_max);
        }
    }

    // Column transform (32 elements each, 16 columns)
    for x in 0..16 {
        identity32_1d(&mut tmp[x..], 16, clip_min, clip_max);
    }

    // Add to destination
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi16(bitdepth_max as i16);
    let rnd_final = _mm256_set1_epi32(8);

    for y in 0..32 {
        let dst_off = y * dst_stride;

        let d = loadu_128!(<&[u8; 16]>::try_from(&dst[dst_off..dst_off + 16]).unwrap());
        let d16 = _mm256_cvtepu8_epi16(d);

        let c0 = _mm256_set_epi32(
            tmp[y * 16 + 7],
            tmp[y * 16 + 6],
            tmp[y * 16 + 5],
            tmp[y * 16 + 4],
            tmp[y * 16 + 3],
            tmp[y * 16 + 2],
            tmp[y * 16 + 1],
            tmp[y * 16 + 0],
        );
        let c1 = _mm256_set_epi32(
            tmp[y * 16 + 15],
            tmp[y * 16 + 14],
            tmp[y * 16 + 13],
            tmp[y * 16 + 12],
            tmp[y * 16 + 11],
            tmp[y * 16 + 10],
            tmp[y * 16 + 9],
            tmp[y * 16 + 8],
        );

        let c0_scaled = _mm256_srai_epi32(_mm256_add_epi32(c0, rnd_final), 4);
        let c1_scaled = _mm256_srai_epi32(_mm256_add_epi32(c1, rnd_final), 4);

        let c16 = _mm256_packs_epi32(c0_scaled, c1_scaled);
        let c16 = _mm256_permute4x64_epi64(c16, 0b11_01_10_00);

        let sum = _mm256_add_epi16(d16, c16);
        let clamped = _mm256_max_epi16(_mm256_min_epi16(sum, max_val), zero);

        let packed = _mm256_packus_epi16(clamped, clamped);
        let packed = _mm256_permute4x64_epi64(packed, 0b11_01_10_00);

        storeu_128!(
            <&mut [u8; 16]>::try_from(&mut dst[dst_off..dst_off + 16]).unwrap(),
            _mm256_castsi256_si128(packed)
        );
    }

    // Clear coefficients
    coeff[..512].fill(0);
}

/// FFI wrapper for 16x32 IDTX 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_identity_16x32_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, _coeff_len as usize * stride + stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_identity_identity_16x32_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 32x16 IDTX inverse transform
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_identity_identity_32x16_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let clip_min = i16::MIN as i32;
    let clip_max = i16::MAX as i32;
    let mut tmp = [0i32; 32 * 16];

    let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

    // Row transform (32 elements each, 16 rows)
    let rnd = 1;
    let shift = 1;
    for y in 0..16 {
        let mut scratch = [0i32; 32];
        for x in 0..32 {
            scratch[x] = rect2_scale(coeff[y + x * 16] as i32);
        }
        identity32_1d(&mut scratch[..32], 1, clip_min, clip_max);
        for x in 0..32 {
            tmp[y * 32 + x] = iclip((scratch[x] + rnd) >> shift, clip_min, clip_max);
        }
    }

    // Column transform (16 elements each, 32 columns)
    for x in 0..32 {
        identity16_1d(&mut tmp[x..], 32, clip_min, clip_max);
    }

    // Add to destination
    for y in 0..16 {
        let dst_off = y * dst_stride;
        for x in 0..32 {
            let d = dst[dst_off + x] as i32;
            let c = (tmp[y * 32 + x] + 8) >> 4;
            let result = iclip(d + c, 0, bitdepth_max);
            dst[dst_off + x] = result as u8;
        }
    }

    // Clear coefficients
    coeff[..512].fill(0);
}

/// FFI wrapper for 32x16 IDTX 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_identity_32x16_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, _coeff_len as usize * stride + stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_identity_identity_32x16_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// Full 2D DCT_DCT 32x64 inverse transform
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_32x64_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    // W=32, H=64, shift=2 for 32x64
    let row_clip_min = i16::MIN as i32;
    let row_clip_max = i16::MAX as i32;
    let col_clip_min = i16::MIN as i32;
    let col_clip_max = i16::MAX as i32;
    let mut tmp = [0i32; 32 * 64];

    // is_rect2 = true for 32x64
    let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

    // Row transform (32 elements each, 64 rows)
    // But only 32 rows of coefficients are present for 64-pt transforms
    let rnd = 1;
    let shift = 1;
    for y in 0..32 {
        let mut scratch = [0i32; 32];
        for x in 0..32 {
            scratch[x] = rect2_scale(coeff[y + x * 32] as i32);
        }
        // Use regular dct32 for rows (all 32 inputs are non-zero)
        dct32_1d(&mut scratch[..32], 1, row_clip_min, row_clip_max);
        for x in 0..32 {
            tmp[y * 32 + x] = iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
        }
    }

    // Pad remaining rows with zeros
    for y in 32..64 {
        for x in 0..32 {
            tmp[y * 32 + x] = 0;
        }
    }

    // Column transform (64 elements each, 32 columns)
    for x in 0..32 {
        dct64_1d(&mut tmp[x..], 32, col_clip_min, col_clip_max);
    }

    // Add to destination
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi16(bitdepth_max as i16);
    let rnd_final = _mm256_set1_epi32(8);

    for y in 0..64 {
        let dst_off = y * dst_stride;

        // Process 32 pixels in two 16-pixel chunks
        for chunk in 0..2 {
            let chunk_off = chunk * 16;
            let d = loadu_128!(<&[u8; 16]>::try_from(
                &dst[dst_off + chunk_off..dst_off + chunk_off + 16]
            )
            .unwrap());
            let d16 = _mm256_cvtepu8_epi16(d);

            let c0 = _mm256_set_epi32(
                tmp[y * 32 + chunk_off + 7],
                tmp[y * 32 + chunk_off + 6],
                tmp[y * 32 + chunk_off + 5],
                tmp[y * 32 + chunk_off + 4],
                tmp[y * 32 + chunk_off + 3],
                tmp[y * 32 + chunk_off + 2],
                tmp[y * 32 + chunk_off + 1],
                tmp[y * 32 + chunk_off + 0],
            );
            let c1 = _mm256_set_epi32(
                tmp[y * 32 + chunk_off + 15],
                tmp[y * 32 + chunk_off + 14],
                tmp[y * 32 + chunk_off + 13],
                tmp[y * 32 + chunk_off + 12],
                tmp[y * 32 + chunk_off + 11],
                tmp[y * 32 + chunk_off + 10],
                tmp[y * 32 + chunk_off + 9],
                tmp[y * 32 + chunk_off + 8],
            );

            let c0_scaled = _mm256_srai_epi32(_mm256_add_epi32(c0, rnd_final), 4);
            let c1_scaled = _mm256_srai_epi32(_mm256_add_epi32(c1, rnd_final), 4);

            let c16 = _mm256_packs_epi32(c0_scaled, c1_scaled);
            let c16 = _mm256_permute4x64_epi64(c16, 0b11_01_10_00);

            let sum = _mm256_add_epi16(d16, c16);
            let clamped = _mm256_max_epi16(_mm256_min_epi16(sum, max_val), zero);

            let packed = _mm256_packus_epi16(clamped, clamped);
            let packed = _mm256_permute4x64_epi64(packed, 0b11_01_10_00);

            storeu_128!(
                <&mut [u8; 16]>::try_from(&mut dst[dst_off + chunk_off..dst_off + chunk_off + 16])
                    .unwrap(),
                _mm256_castsi256_si128(packed)
            );
        }
    }

    // Clear coefficients
    coeff[..1024].fill(0);
}

/// FFI wrapper for 32x64 DCT_DCT 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_32x64_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, _coeff_len as usize * stride + stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_32x64_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// Full 2D DCT_DCT 64x32 inverse transform
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_64x32_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    // W=64, H=32, shift=2 for 64x32
    let row_clip_min = i16::MIN as i32;
    let row_clip_max = i16::MAX as i32;
    let col_clip_min = i16::MIN as i32;
    let col_clip_max = i16::MAX as i32;
    let mut tmp = [0i32; 64 * 32];

    // is_rect2 = true for 64x32
    let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

    // Row transform (64 elements each, 32 rows)
    // But only first 32 columns have coefficients for 64-pt transforms
    let rnd = 1;
    let shift = 1;
    for y in 0..32 {
        let mut scratch = [0i32; 64];
        for x in 0..32 {
            scratch[x] = rect2_scale(coeff[y + x * 32] as i32);
        }
        for x in 32..64 {
            scratch[x] = 0;
        }
        dct64_1d(&mut scratch[..64], 1, row_clip_min, row_clip_max);
        for x in 0..64 {
            tmp[y * 64 + x] = iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
        }
    }

    // Column transform (32 elements each, 64 columns)
    // Use regular dct32 - all 32 rows are populated after row transform
    for x in 0..64 {
        dct32_1d(&mut tmp[x..], 64, col_clip_min, col_clip_max);
    }

    // Add to destination
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi16(bitdepth_max as i16);
    let rnd_final = _mm256_set1_epi32(8);

    for y in 0..32 {
        let dst_off = y * dst_stride;

        // Process 64 pixels in four 16-pixel chunks
        for chunk in 0..4 {
            let chunk_off = chunk * 16;
            let d = loadu_128!(<&[u8; 16]>::try_from(
                &dst[dst_off + chunk_off..dst_off + chunk_off + 16]
            )
            .unwrap());
            let d16 = _mm256_cvtepu8_epi16(d);

            let c0 = _mm256_set_epi32(
                tmp[y * 64 + chunk_off + 7],
                tmp[y * 64 + chunk_off + 6],
                tmp[y * 64 + chunk_off + 5],
                tmp[y * 64 + chunk_off + 4],
                tmp[y * 64 + chunk_off + 3],
                tmp[y * 64 + chunk_off + 2],
                tmp[y * 64 + chunk_off + 1],
                tmp[y * 64 + chunk_off + 0],
            );
            let c1 = _mm256_set_epi32(
                tmp[y * 64 + chunk_off + 15],
                tmp[y * 64 + chunk_off + 14],
                tmp[y * 64 + chunk_off + 13],
                tmp[y * 64 + chunk_off + 12],
                tmp[y * 64 + chunk_off + 11],
                tmp[y * 64 + chunk_off + 10],
                tmp[y * 64 + chunk_off + 9],
                tmp[y * 64 + chunk_off + 8],
            );

            let c0_scaled = _mm256_srai_epi32(_mm256_add_epi32(c0, rnd_final), 4);
            let c1_scaled = _mm256_srai_epi32(_mm256_add_epi32(c1, rnd_final), 4);

            let c16 = _mm256_packs_epi32(c0_scaled, c1_scaled);
            let c16 = _mm256_permute4x64_epi64(c16, 0b11_01_10_00);

            let sum = _mm256_add_epi16(d16, c16);
            let clamped = _mm256_max_epi16(_mm256_min_epi16(sum, max_val), zero);

            let packed = _mm256_packus_epi16(clamped, clamped);
            let packed = _mm256_permute4x64_epi64(packed, 0b11_01_10_00);

            storeu_128!(
                <&mut [u8; 16]>::try_from(&mut dst[dst_off + chunk_off..dst_off + chunk_off + 16])
                    .unwrap(),
                _mm256_castsi256_si128(packed)
            );
        }
    }

    // Clear coefficients
    coeff[..1024].fill(0);
}

/// FFI wrapper for 64x32 DCT_DCT 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_64x32_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, _coeff_len as usize * stride + stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_64x32_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// RECTANGULAR TRANSFORMS - 4:1 aspect ratio (4x16, 16x4, 8x32, 32x8)
// ============================================================================

/// Full 2D DCT_DCT 4x16 inverse transform
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_4x16_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    // W=4, H=16, 4:1 ratio
    let row_clip_min = i16::MIN as i32;
    let row_clip_max = i16::MAX as i32;
    let col_clip_min = i16::MIN as i32;
    let col_clip_max = i16::MAX as i32;
    let mut tmp = [0i32; 4 * 16];

    // Row transform (4 elements each, 16 rows), shift=1 for 4x16
    let rnd = 1;
    let shift = 1;
    for y in 0..16 {
        let mut scratch = [0i32; 4];
        for x in 0..4 {
            scratch[x] = coeff[y + x * 16] as i32;
        }
        dct4_1d(&mut scratch[..4], 1, row_clip_min, row_clip_max);
        for x in 0..4 {
            tmp[y * 4 + x] = iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
        }
    }

    // Column transform (16 elements each, 4 columns)
    for x in 0..4 {
        dct16_1d(&mut tmp[x..], 4, col_clip_min, col_clip_max);
    }

    // Add to destination - 4 pixels at a time
    for y in 0..16 {
        let dst_off = y * dst_stride;
        for x in 0..4 {
            let d = dst[dst_off + x] as i32;
            let c = (tmp[y * 4 + x] + 8) >> 4;
            let result = iclip(d + c, 0, bitdepth_max);
            dst[dst_off + x] = result as u8;
        }
    }

    // Clear coefficients
    coeff[..64].fill(0);
}

/// FFI wrapper for 4x16 DCT_DCT 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_4x16_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, _coeff_len as usize * stride + stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_4x16_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// Full 2D DCT_DCT 16x4 inverse transform
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_16x4_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    // W=16, H=4, 4:1 ratio
    let row_clip_min = i16::MIN as i32;
    let row_clip_max = i16::MAX as i32;
    let col_clip_min = i16::MIN as i32;
    let col_clip_max = i16::MAX as i32;
    let mut tmp = [0i32; 16 * 4];

    // rect4 scaling
    // Row transform (16 elements each, 4 rows), shift=1 for 16x4
    let rnd = 1;
    let shift = 1;
    for y in 0..4 {
        let mut scratch = [0i32; 16];
        for x in 0..16 {
            scratch[x] = coeff[y + x * 4] as i32;
        }
        dct16_1d(&mut scratch[..16], 1, row_clip_min, row_clip_max);
        for x in 0..16 {
            tmp[y * 16 + x] = iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
        }
    }

    // Column transform (4 elements each, 16 columns)
    for x in 0..16 {
        dct4_1d(&mut tmp[x..], 16, col_clip_min, col_clip_max);
    }

    // Add to destination
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi16(bitdepth_max as i16);
    let rnd_final = _mm256_set1_epi32(8);

    for y in 0..4 {
        let dst_off = y * dst_stride;

        let d = loadu_128!(<&[u8; 16]>::try_from(&dst[dst_off..dst_off + 16]).unwrap());
        let d16 = _mm256_cvtepu8_epi16(d);

        let c0 = _mm256_set_epi32(
            tmp[y * 16 + 7],
            tmp[y * 16 + 6],
            tmp[y * 16 + 5],
            tmp[y * 16 + 4],
            tmp[y * 16 + 3],
            tmp[y * 16 + 2],
            tmp[y * 16 + 1],
            tmp[y * 16 + 0],
        );
        let c1 = _mm256_set_epi32(
            tmp[y * 16 + 15],
            tmp[y * 16 + 14],
            tmp[y * 16 + 13],
            tmp[y * 16 + 12],
            tmp[y * 16 + 11],
            tmp[y * 16 + 10],
            tmp[y * 16 + 9],
            tmp[y * 16 + 8],
        );

        let c0_scaled = _mm256_srai_epi32(_mm256_add_epi32(c0, rnd_final), 4);
        let c1_scaled = _mm256_srai_epi32(_mm256_add_epi32(c1, rnd_final), 4);

        let c16 = _mm256_packs_epi32(c0_scaled, c1_scaled);
        let c16 = _mm256_permute4x64_epi64(c16, 0b11_01_10_00);

        let sum = _mm256_add_epi16(d16, c16);
        let clamped = _mm256_max_epi16(_mm256_min_epi16(sum, max_val), zero);

        let packed = _mm256_packus_epi16(clamped, clamped);
        let packed = _mm256_permute4x64_epi64(packed, 0b11_01_10_00);

        storeu_128!(
            <&mut [u8; 16]>::try_from(&mut dst[dst_off..dst_off + 16]).unwrap(),
            _mm256_castsi256_si128(packed)
        );
    }

    // Clear coefficients
    coeff[..64].fill(0);
}

/// FFI wrapper for 16x4 DCT_DCT 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_16x4_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, _coeff_len as usize * stride + stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_16x4_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// 4x16 and 16x4 ADST/FLIPADST variants
// ============================================================================

/// Helper macro for 4x16 transforms with configurable row/col transforms
macro_rules! impl_4x16_transform {
    ($name:ident, $row_fn:ident, $col_fn:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[arcane]
        fn $name(
            _token: Desktop64,
            dst: &mut [u8],
            dst_stride: usize,
            coeff: &mut [i16],
            _eob: i32,
            bitdepth_max: i32,
        ) {
            use crate::src::safe_simd::pixel_access::{
                loadi32, loadi64, loadu_128, storei32, storei64, storeu_128,
            };
            let mut dst = dst.flex_mut();
            let mut coeff = coeff.flex_mut();
            let row_clip_min = i16::MIN as i32;
            let row_clip_max = i16::MAX as i32;
            let col_clip_min = i16::MIN as i32;
            let col_clip_max = i16::MAX as i32;
            let mut tmp = [0i32; 4 * 16];

            // Row transform (4 elements each, 16 rows), shift=1 for 4x16
            let rnd = 1;
            let shift = 1;
            for y in 0..16 {
                let mut scratch = [0i32; 4];
                for x in 0..4 {
                    scratch[x] = coeff[y + x * 16] as i32;
                }
                $row_fn(&mut scratch[..4], 1, row_clip_min, row_clip_max);
                for x in 0..4 {
                    tmp[y * 4 + x] = iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
                }
            }

            // Column transform (16 elements each, 4 columns)
            for x in 0..4 {
                $col_fn(&mut tmp[x..], 4, col_clip_min, col_clip_max);
            }

            // Add to destination
            for y in 0..16 {
                let dst_off = y * dst_stride;
                for x in 0..4 {
                    let d = dst[dst_off + x] as i32;
                    let c = (tmp[y * 4 + x] + 8) >> 4;
                    let result = iclip(d + c, 0, bitdepth_max);
                    dst[dst_off + x] = result as u8;
                }
            }

            // Clear coefficients
            coeff[..64].fill(0);
        }
    };
}

/// Helper macro for 16x4 transforms with configurable row/col transforms
macro_rules! impl_16x4_transform {
    ($name:ident, $row_fn:ident, $col_fn:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[arcane]
        fn $name(
            _token: Desktop64,
            dst: &mut [u8],
            dst_stride: usize,
            coeff: &mut [i16],
            _eob: i32,
            bitdepth_max: i32,
        ) {
            use crate::src::safe_simd::pixel_access::{
                loadi32, loadi64, loadu_128, storei32, storei64, storeu_128,
            };
            let mut dst = dst.flex_mut();
            let mut coeff = coeff.flex_mut();
            let row_clip_min = i16::MIN as i32;
            let row_clip_max = i16::MAX as i32;
            let col_clip_min = i16::MIN as i32;
            let col_clip_max = i16::MAX as i32;
            let mut tmp = [0i32; 16 * 4];

            // Row transform (16 elements each, 4 rows), shift=1 for 16x4
            let rnd = 1;
            let shift = 1;
            for y in 0..4 {
                let mut scratch = [0i32; 16];
                for x in 0..16 {
                    scratch[x] = coeff[y + x * 4] as i32;
                }
                $row_fn(&mut scratch[..16], 1, row_clip_min, row_clip_max);
                for x in 0..16 {
                    tmp[y * 16 + x] = iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
                }
            }

            // Column transform (4 elements each, 16 columns)
            for x in 0..16 {
                $col_fn(&mut tmp[x..], 16, col_clip_min, col_clip_max);
            }

            // Add to destination
            let zero = _mm256_setzero_si256();
            let max_val = _mm256_set1_epi16(bitdepth_max as i16);
            let rnd_final = _mm256_set1_epi32(8);

            for y in 0..4 {
                let dst_off = y * dst_stride;

                let d = loadu_128!(<&[u8; 16]>::try_from(&dst[dst_off..dst_off + 16]).unwrap());
                let d16 = _mm256_cvtepu8_epi16(d);

                let c0 = _mm256_set_epi32(
                    tmp[y * 16 + 7],
                    tmp[y * 16 + 6],
                    tmp[y * 16 + 5],
                    tmp[y * 16 + 4],
                    tmp[y * 16 + 3],
                    tmp[y * 16 + 2],
                    tmp[y * 16 + 1],
                    tmp[y * 16 + 0],
                );
                let c1 = _mm256_set_epi32(
                    tmp[y * 16 + 15],
                    tmp[y * 16 + 14],
                    tmp[y * 16 + 13],
                    tmp[y * 16 + 12],
                    tmp[y * 16 + 11],
                    tmp[y * 16 + 10],
                    tmp[y * 16 + 9],
                    tmp[y * 16 + 8],
                );

                let c0_scaled = _mm256_srai_epi32(_mm256_add_epi32(c0, rnd_final), 4);
                let c1_scaled = _mm256_srai_epi32(_mm256_add_epi32(c1, rnd_final), 4);

                let c16 = _mm256_packs_epi32(c0_scaled, c1_scaled);
                let c16 = _mm256_permute4x64_epi64(c16, 0b11_01_10_00);

                let sum = _mm256_add_epi16(d16, c16);
                let clamped = _mm256_max_epi16(_mm256_min_epi16(sum, max_val), zero);

                let packed = _mm256_packus_epi16(clamped, clamped);
                let packed = _mm256_permute4x64_epi64(packed, 0b11_01_10_00);

                storeu_128!(
                    <&mut [u8; 16]>::try_from(&mut dst[dst_off..dst_off + 16]).unwrap(),
                    _mm256_castsi256_si128(packed)
                );
            }

            // Clear coefficients
            coeff[..64].fill(0);
        }
    };
}

// Generate 4x16 ADST inner functions
impl_4x16_transform!(
    inv_txfm_add_adst_dct_4x16_8bpc_avx2_inner,
    adst4_1d,
    dct16_1d
);
impl_4x16_transform!(
    inv_txfm_add_dct_adst_4x16_8bpc_avx2_inner,
    dct4_1d,
    adst16_1d
);
impl_4x16_transform!(
    inv_txfm_add_adst_adst_4x16_8bpc_avx2_inner,
    adst4_1d,
    adst16_1d
);
impl_4x16_transform!(
    inv_txfm_add_flipadst_dct_4x16_8bpc_avx2_inner,
    flipadst4_1d,
    dct16_1d
);
impl_4x16_transform!(
    inv_txfm_add_dct_flipadst_4x16_8bpc_avx2_inner,
    dct4_1d,
    flipadst16_1d
);
impl_4x16_transform!(
    inv_txfm_add_flipadst_flipadst_4x16_8bpc_avx2_inner,
    flipadst4_1d,
    flipadst16_1d
);
impl_4x16_transform!(
    inv_txfm_add_adst_flipadst_4x16_8bpc_avx2_inner,
    adst4_1d,
    flipadst16_1d
);
impl_4x16_transform!(
    inv_txfm_add_flipadst_adst_4x16_8bpc_avx2_inner,
    flipadst4_1d,
    adst16_1d
);

// Generate 16x4 ADST inner functions
impl_16x4_transform!(
    inv_txfm_add_adst_dct_16x4_8bpc_avx2_inner,
    adst16_1d,
    dct4_1d
);
impl_16x4_transform!(
    inv_txfm_add_dct_adst_16x4_8bpc_avx2_inner,
    dct16_1d,
    adst4_1d
);
impl_16x4_transform!(
    inv_txfm_add_adst_adst_16x4_8bpc_avx2_inner,
    adst16_1d,
    adst4_1d
);
impl_16x4_transform!(
    inv_txfm_add_flipadst_dct_16x4_8bpc_avx2_inner,
    flipadst16_1d,
    dct4_1d
);
impl_16x4_transform!(
    inv_txfm_add_dct_flipadst_16x4_8bpc_avx2_inner,
    dct16_1d,
    flipadst4_1d
);
impl_16x4_transform!(
    inv_txfm_add_flipadst_flipadst_16x4_8bpc_avx2_inner,
    flipadst16_1d,
    flipadst4_1d
);
impl_16x4_transform!(
    inv_txfm_add_adst_flipadst_16x4_8bpc_avx2_inner,
    adst16_1d,
    flipadst4_1d
);
impl_16x4_transform!(
    inv_txfm_add_flipadst_adst_16x4_8bpc_avx2_inner,
    flipadst16_1d,
    adst4_1d
);

/// FFI wrapper macro for 4x16 transforms
macro_rules! impl_4x16_ffi_wrapper {
    ($name:ident, $inner:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2")]
        #[cfg(feature = "asm")]
        pub unsafe extern "C" fn $name(
            dst_ptr: *mut DynPixel,
            dst_stride: isize,
            coeff: *mut DynCoef,
            eob: c_int,
            bitdepth_max: c_int,
            _coeff_len: u16,
            _dst: *const FFISafe<PicOffset>,
        ) {
            let _token = unsafe { Desktop64::forge_token_dangerously() };
            let stride = dst_stride as usize;

            let dst_slice = unsafe {
                std::slice::from_raw_parts_mut(
                    dst_ptr as *mut u8,
                    _coeff_len as usize * stride + stride,
                )
            };

            let coeff_slice =
                unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

            $inner(_token, dst_slice, stride, coeff_slice, eob, bitdepth_max);
        }
    };
}

/// FFI wrapper macro for 16x4 transforms
macro_rules! impl_16x4_ffi_wrapper {
    ($name:ident, $inner:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2")]
        #[cfg(feature = "asm")]
        pub unsafe extern "C" fn $name(
            dst_ptr: *mut DynPixel,
            dst_stride: isize,
            coeff: *mut DynCoef,
            eob: c_int,
            bitdepth_max: c_int,
            _coeff_len: u16,
            _dst: *const FFISafe<PicOffset>,
        ) {
            let _token = unsafe { Desktop64::forge_token_dangerously() };
            let stride = dst_stride as usize;

            let dst_slice = unsafe {
                std::slice::from_raw_parts_mut(
                    dst_ptr as *mut u8,
                    _coeff_len as usize * stride + stride,
                )
            };

            let coeff_slice =
                unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

            $inner(_token, dst_slice, stride, coeff_slice, eob, bitdepth_max);
        }
    };
}

// Generate 4x16 FFI wrappers
impl_4x16_ffi_wrapper!(
    inv_txfm_add_adst_dct_4x16_8bpc_avx2,
    inv_txfm_add_adst_dct_4x16_8bpc_avx2_inner
);
impl_4x16_ffi_wrapper!(
    inv_txfm_add_dct_adst_4x16_8bpc_avx2,
    inv_txfm_add_dct_adst_4x16_8bpc_avx2_inner
);
impl_4x16_ffi_wrapper!(
    inv_txfm_add_adst_adst_4x16_8bpc_avx2,
    inv_txfm_add_adst_adst_4x16_8bpc_avx2_inner
);
impl_4x16_ffi_wrapper!(
    inv_txfm_add_flipadst_dct_4x16_8bpc_avx2,
    inv_txfm_add_flipadst_dct_4x16_8bpc_avx2_inner
);
impl_4x16_ffi_wrapper!(
    inv_txfm_add_dct_flipadst_4x16_8bpc_avx2,
    inv_txfm_add_dct_flipadst_4x16_8bpc_avx2_inner
);
impl_4x16_ffi_wrapper!(
    inv_txfm_add_flipadst_flipadst_4x16_8bpc_avx2,
    inv_txfm_add_flipadst_flipadst_4x16_8bpc_avx2_inner
);
impl_4x16_ffi_wrapper!(
    inv_txfm_add_adst_flipadst_4x16_8bpc_avx2,
    inv_txfm_add_adst_flipadst_4x16_8bpc_avx2_inner
);
impl_4x16_ffi_wrapper!(
    inv_txfm_add_flipadst_adst_4x16_8bpc_avx2,
    inv_txfm_add_flipadst_adst_4x16_8bpc_avx2_inner
);

// Generate 16x4 FFI wrappers
impl_16x4_ffi_wrapper!(
    inv_txfm_add_adst_dct_16x4_8bpc_avx2,
    inv_txfm_add_adst_dct_16x4_8bpc_avx2_inner
);
impl_16x4_ffi_wrapper!(
    inv_txfm_add_dct_adst_16x4_8bpc_avx2,
    inv_txfm_add_dct_adst_16x4_8bpc_avx2_inner
);
impl_16x4_ffi_wrapper!(
    inv_txfm_add_adst_adst_16x4_8bpc_avx2,
    inv_txfm_add_adst_adst_16x4_8bpc_avx2_inner
);
impl_16x4_ffi_wrapper!(
    inv_txfm_add_flipadst_dct_16x4_8bpc_avx2,
    inv_txfm_add_flipadst_dct_16x4_8bpc_avx2_inner
);
impl_16x4_ffi_wrapper!(
    inv_txfm_add_dct_flipadst_16x4_8bpc_avx2,
    inv_txfm_add_dct_flipadst_16x4_8bpc_avx2_inner
);
impl_16x4_ffi_wrapper!(
    inv_txfm_add_flipadst_flipadst_16x4_8bpc_avx2,
    inv_txfm_add_flipadst_flipadst_16x4_8bpc_avx2_inner
);
impl_16x4_ffi_wrapper!(
    inv_txfm_add_adst_flipadst_16x4_8bpc_avx2,
    inv_txfm_add_adst_flipadst_16x4_8bpc_avx2_inner
);
impl_16x4_ffi_wrapper!(
    inv_txfm_add_flipadst_adst_16x4_8bpc_avx2,
    inv_txfm_add_flipadst_adst_16x4_8bpc_avx2_inner
);

// IDTX for 4x16 and 16x4
impl_4x16_transform!(
    inv_txfm_add_identity_identity_4x16_8bpc_avx2_inner,
    identity4_1d,
    identity16_1d
);
impl_16x4_transform!(
    inv_txfm_add_identity_identity_16x4_8bpc_avx2_inner,
    identity16_1d,
    identity4_1d
);
impl_4x16_ffi_wrapper!(
    inv_txfm_add_identity_identity_4x16_8bpc_avx2,
    inv_txfm_add_identity_identity_4x16_8bpc_avx2_inner
);
impl_16x4_ffi_wrapper!(
    inv_txfm_add_identity_identity_16x4_8bpc_avx2,
    inv_txfm_add_identity_identity_16x4_8bpc_avx2_inner
);

// H_DCT and V_DCT for 4x16
impl_4x16_transform!(
    inv_txfm_add_identity_dct_4x16_8bpc_avx2_inner,
    identity4_1d,
    dct16_1d
);
impl_4x16_transform!(
    inv_txfm_add_dct_identity_4x16_8bpc_avx2_inner,
    dct4_1d,
    identity16_1d
);
impl_4x16_ffi_wrapper!(
    inv_txfm_add_identity_dct_4x16_8bpc_avx2,
    inv_txfm_add_identity_dct_4x16_8bpc_avx2_inner
);
impl_4x16_ffi_wrapper!(
    inv_txfm_add_dct_identity_4x16_8bpc_avx2,
    inv_txfm_add_dct_identity_4x16_8bpc_avx2_inner
);

// H_DCT and V_DCT for 16x4
impl_16x4_transform!(
    inv_txfm_add_identity_dct_16x4_8bpc_avx2_inner,
    identity16_1d,
    dct4_1d
);
impl_16x4_transform!(
    inv_txfm_add_dct_identity_16x4_8bpc_avx2_inner,
    dct16_1d,
    identity4_1d
);
impl_16x4_ffi_wrapper!(
    inv_txfm_add_identity_dct_16x4_8bpc_avx2,
    inv_txfm_add_identity_dct_16x4_8bpc_avx2_inner
);
impl_16x4_ffi_wrapper!(
    inv_txfm_add_dct_identity_16x4_8bpc_avx2,
    inv_txfm_add_dct_identity_16x4_8bpc_avx2_inner
);

// H_ADST, V_ADST, H_FLIPADST, V_FLIPADST for 4x16
impl_4x16_transform!(
    inv_txfm_add_identity_adst_4x16_8bpc_avx2_inner,
    identity4_1d,
    adst16_1d
);
impl_4x16_transform!(
    inv_txfm_add_adst_identity_4x16_8bpc_avx2_inner,
    adst4_1d,
    identity16_1d
);
impl_4x16_transform!(
    inv_txfm_add_identity_flipadst_4x16_8bpc_avx2_inner,
    identity4_1d,
    flipadst16_1d
);
impl_4x16_transform!(
    inv_txfm_add_flipadst_identity_4x16_8bpc_avx2_inner,
    flipadst4_1d,
    identity16_1d
);
impl_4x16_ffi_wrapper!(
    inv_txfm_add_identity_adst_4x16_8bpc_avx2,
    inv_txfm_add_identity_adst_4x16_8bpc_avx2_inner
);
impl_4x16_ffi_wrapper!(
    inv_txfm_add_adst_identity_4x16_8bpc_avx2,
    inv_txfm_add_adst_identity_4x16_8bpc_avx2_inner
);
impl_4x16_ffi_wrapper!(
    inv_txfm_add_identity_flipadst_4x16_8bpc_avx2,
    inv_txfm_add_identity_flipadst_4x16_8bpc_avx2_inner
);
impl_4x16_ffi_wrapper!(
    inv_txfm_add_flipadst_identity_4x16_8bpc_avx2,
    inv_txfm_add_flipadst_identity_4x16_8bpc_avx2_inner
);

// H_ADST, V_ADST, H_FLIPADST, V_FLIPADST for 16x4
impl_16x4_transform!(
    inv_txfm_add_identity_adst_16x4_8bpc_avx2_inner,
    identity16_1d,
    adst4_1d
);
impl_16x4_transform!(
    inv_txfm_add_adst_identity_16x4_8bpc_avx2_inner,
    adst16_1d,
    identity4_1d
);
impl_16x4_transform!(
    inv_txfm_add_identity_flipadst_16x4_8bpc_avx2_inner,
    identity16_1d,
    flipadst4_1d
);
impl_16x4_transform!(
    inv_txfm_add_flipadst_identity_16x4_8bpc_avx2_inner,
    flipadst16_1d,
    identity4_1d
);
impl_16x4_ffi_wrapper!(
    inv_txfm_add_identity_adst_16x4_8bpc_avx2,
    inv_txfm_add_identity_adst_16x4_8bpc_avx2_inner
);
impl_16x4_ffi_wrapper!(
    inv_txfm_add_adst_identity_16x4_8bpc_avx2,
    inv_txfm_add_adst_identity_16x4_8bpc_avx2_inner
);
impl_16x4_ffi_wrapper!(
    inv_txfm_add_identity_flipadst_16x4_8bpc_avx2,
    inv_txfm_add_identity_flipadst_16x4_8bpc_avx2_inner
);
impl_16x4_ffi_wrapper!(
    inv_txfm_add_flipadst_identity_16x4_8bpc_avx2,
    inv_txfm_add_flipadst_identity_16x4_8bpc_avx2_inner
);

/// Full 2D DCT_DCT 8x32 inverse transform
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_8x32_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    // W=8, H=32, 4:1 ratio
    let row_clip_min = i16::MIN as i32;
    let row_clip_max = i16::MAX as i32;
    let col_clip_min = i16::MIN as i32;
    let col_clip_max = i16::MAX as i32;
    let mut tmp = [0i32; 8 * 32];

    // rect4 scaling
    // Row transform (8 elements each, 32 rows)
    let rnd = 2;
    let shift = 2;
    for y in 0..32 {
        let mut scratch = [0i32; 8];
        for x in 0..8 {
            scratch[x] = coeff[y + x * 32] as i32;
        }
        dct8_1d(&mut scratch[..8], 1, row_clip_min, row_clip_max);
        for x in 0..8 {
            tmp[y * 8 + x] = iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
        }
    }

    // Column transform (32 elements each, 8 columns)
    for x in 0..8 {
        dct32_1d(&mut tmp[x..], 8, col_clip_min, col_clip_max);
    }

    // Add to destination
    for y in 0..32 {
        let dst_off = y * dst_stride;
        for x in 0..8 {
            let d = dst[dst_off + x] as i32;
            let c = (tmp[y * 8 + x] + 8) >> 4;
            let result = iclip(d + c, 0, bitdepth_max);
            dst[dst_off + x] = result as u8;
        }
    }

    // Clear coefficients
    coeff[..256].fill(0);
}

/// FFI wrapper for 8x32 DCT_DCT 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_8x32_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, _coeff_len as usize * stride + stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_8x32_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// Full 2D DCT_DCT 32x8 inverse transform
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_32x8_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    // W=32, H=8, 4:1 ratio
    let row_clip_min = i16::MIN as i32;
    let row_clip_max = i16::MAX as i32;
    let col_clip_min = i16::MIN as i32;
    let col_clip_max = i16::MAX as i32;
    let mut tmp = [0i32; 32 * 8];

    // rect4 scaling
    // Row transform (32 elements each, 8 rows)
    let rnd = 2;
    let shift = 2;
    for y in 0..8 {
        let mut scratch = [0i32; 32];
        for x in 0..32 {
            scratch[x] = coeff[y + x * 8] as i32;
        }
        dct32_1d(&mut scratch[..32], 1, row_clip_min, row_clip_max);
        for x in 0..32 {
            tmp[y * 32 + x] = iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
        }
    }

    // Column transform (8 elements each, 32 columns)
    for x in 0..32 {
        dct8_1d(&mut tmp[x..], 32, col_clip_min, col_clip_max);
    }

    // Add to destination
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi16(bitdepth_max as i16);
    let rnd_final = _mm256_set1_epi32(8);

    for y in 0..8 {
        let dst_off = y * dst_stride;

        // Process 32 pixels in two 16-pixel chunks
        for chunk in 0..2 {
            let chunk_off = chunk * 16;
            let d = loadu_128!(<&[u8; 16]>::try_from(
                &dst[dst_off + chunk_off..dst_off + chunk_off + 16]
            )
            .unwrap());
            let d16 = _mm256_cvtepu8_epi16(d);

            let c0 = _mm256_set_epi32(
                tmp[y * 32 + chunk_off + 7],
                tmp[y * 32 + chunk_off + 6],
                tmp[y * 32 + chunk_off + 5],
                tmp[y * 32 + chunk_off + 4],
                tmp[y * 32 + chunk_off + 3],
                tmp[y * 32 + chunk_off + 2],
                tmp[y * 32 + chunk_off + 1],
                tmp[y * 32 + chunk_off + 0],
            );
            let c1 = _mm256_set_epi32(
                tmp[y * 32 + chunk_off + 15],
                tmp[y * 32 + chunk_off + 14],
                tmp[y * 32 + chunk_off + 13],
                tmp[y * 32 + chunk_off + 12],
                tmp[y * 32 + chunk_off + 11],
                tmp[y * 32 + chunk_off + 10],
                tmp[y * 32 + chunk_off + 9],
                tmp[y * 32 + chunk_off + 8],
            );

            let c0_scaled = _mm256_srai_epi32(_mm256_add_epi32(c0, rnd_final), 4);
            let c1_scaled = _mm256_srai_epi32(_mm256_add_epi32(c1, rnd_final), 4);

            let c16 = _mm256_packs_epi32(c0_scaled, c1_scaled);
            let c16 = _mm256_permute4x64_epi64(c16, 0b11_01_10_00);

            let sum = _mm256_add_epi16(d16, c16);
            let clamped = _mm256_max_epi16(_mm256_min_epi16(sum, max_val), zero);

            let packed = _mm256_packus_epi16(clamped, clamped);
            let packed = _mm256_permute4x64_epi64(packed, 0b11_01_10_00);

            storeu_128!(
                <&mut [u8; 16]>::try_from(&mut dst[dst_off + chunk_off..dst_off + chunk_off + 16])
                    .unwrap(),
                _mm256_castsi256_si128(packed)
            );
        }
    }

    // Clear coefficients
    coeff[..256].fill(0);
}

/// FFI wrapper for 32x8 DCT_DCT 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_32x8_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, _coeff_len as usize * stride + stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_32x8_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// 8x32 and 32x8 IDTX (identity_identity) transforms
// ============================================================================

/// 8x32 IDTX inverse transform
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_identity_identity_8x32_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let clip_min = i16::MIN as i32;
    let clip_max = i16::MAX as i32;
    let mut tmp = [0i32; 8 * 32];

    // Row transform (8 elements each, 32 rows), shift=2 for 8x32
    let rnd = 2;
    let shift = 2;
    for y in 0..32 {
        let mut scratch = [0i32; 8];
        for x in 0..8 {
            scratch[x] = coeff[y + x * 32] as i32;
        }
        identity8_1d(&mut scratch[..8], 1, clip_min, clip_max);
        for x in 0..8 {
            tmp[y * 8 + x] = iclip((scratch[x] + rnd) >> shift, clip_min, clip_max);
        }
    }

    // Column transform (32 elements each, 8 columns)
    for x in 0..8 {
        identity32_1d(&mut tmp[x..], 8, clip_min, clip_max);
    }

    // Add to destination
    for y in 0..32 {
        let dst_off = y * dst_stride;
        for x in 0..8 {
            let d = dst[dst_off + x] as i32;
            let c = (tmp[y * 8 + x] + 8) >> 4;
            let result = iclip(d + c, 0, bitdepth_max);
            dst[dst_off + x] = result as u8;
        }
    }

    // Clear coefficients
    coeff[..256].fill(0);
}

/// FFI wrapper for 8x32 IDTX 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_identity_8x32_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, _coeff_len as usize * stride + stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_identity_identity_8x32_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 32x8 IDTX inverse transform
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_identity_identity_32x8_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let clip_min = i16::MIN as i32;
    let clip_max = i16::MAX as i32;
    let mut tmp = [0i32; 32 * 8];

    // Row transform (32 elements each, 8 rows), shift=2 for 32x8
    let rnd = 2;
    let shift = 2;
    for y in 0..8 {
        let mut scratch = [0i32; 32];
        for x in 0..32 {
            scratch[x] = coeff[y + x * 8] as i32;
        }
        identity32_1d(&mut scratch[..32], 1, clip_min, clip_max);
        for x in 0..32 {
            tmp[y * 32 + x] = iclip((scratch[x] + rnd) >> shift, clip_min, clip_max);
        }
    }

    // Column transform (8 elements each, 32 columns)
    for x in 0..32 {
        identity8_1d(&mut tmp[x..], 32, clip_min, clip_max);
    }

    // Add to destination
    for y in 0..8 {
        let dst_off = y * dst_stride;
        for x in 0..32 {
            let d = dst[dst_off + x] as i32;
            let c = (tmp[y * 32 + x] + 8) >> 4;
            let result = iclip(d + c, 0, bitdepth_max);
            dst[dst_off + x] = result as u8;
        }
    }

    // Clear coefficients
    coeff[..256].fill(0);
}

/// FFI wrapper for 32x8 IDTX 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_identity_32x8_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, _coeff_len as usize * stride + stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_identity_identity_32x8_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// RECTANGULAR TRANSFORMS - 16x64, 64x16
// ============================================================================

/// Full 2D DCT_DCT 16x64 inverse transform
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_16x64_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    // W=16, H=64, 4:1 ratio
    let row_clip_min = i16::MIN as i32;
    let row_clip_max = i16::MAX as i32;
    let col_clip_min = i16::MIN as i32;
    let col_clip_max = i16::MAX as i32;
    let mut tmp = [0i32; 16 * 64];

    // rect4 scaling
    // Row transform (16 elements each, 32 rows - only 32 rows have coefficients)
    let rnd = 2;
    let shift = 2;
    for y in 0..32 {
        let mut scratch = [0i32; 16];
        for x in 0..16 {
            scratch[x] = coeff[y + x * 32] as i32;
        }
        // Use regular dct16 for rows (all 16 inputs are non-zero)
        dct16_1d(&mut scratch[..16], 1, row_clip_min, row_clip_max);
        for x in 0..16 {
            tmp[y * 16 + x] = iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
        }
    }

    // Zero remaining rows
    for y in 32..64 {
        for x in 0..16 {
            tmp[y * 16 + x] = 0;
        }
    }

    // Column transform (64 elements each, 16 columns)
    for x in 0..16 {
        dct64_1d(&mut tmp[x..], 16, col_clip_min, col_clip_max);
    }

    // Add to destination
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi16(bitdepth_max as i16);
    let rnd_final = _mm256_set1_epi32(8);

    for y in 0..64 {
        let dst_off = y * dst_stride;

        let d = loadu_128!(<&[u8; 16]>::try_from(&dst[dst_off..dst_off + 16]).unwrap());
        let d16 = _mm256_cvtepu8_epi16(d);

        let c0 = _mm256_set_epi32(
            tmp[y * 16 + 7],
            tmp[y * 16 + 6],
            tmp[y * 16 + 5],
            tmp[y * 16 + 4],
            tmp[y * 16 + 3],
            tmp[y * 16 + 2],
            tmp[y * 16 + 1],
            tmp[y * 16 + 0],
        );
        let c1 = _mm256_set_epi32(
            tmp[y * 16 + 15],
            tmp[y * 16 + 14],
            tmp[y * 16 + 13],
            tmp[y * 16 + 12],
            tmp[y * 16 + 11],
            tmp[y * 16 + 10],
            tmp[y * 16 + 9],
            tmp[y * 16 + 8],
        );

        let c0_scaled = _mm256_srai_epi32(_mm256_add_epi32(c0, rnd_final), 4);
        let c1_scaled = _mm256_srai_epi32(_mm256_add_epi32(c1, rnd_final), 4);

        let c16 = _mm256_packs_epi32(c0_scaled, c1_scaled);
        let c16 = _mm256_permute4x64_epi64(c16, 0b11_01_10_00);

        let sum = _mm256_add_epi16(d16, c16);
        let clamped = _mm256_max_epi16(_mm256_min_epi16(sum, max_val), zero);

        let packed = _mm256_packus_epi16(clamped, clamped);
        let packed = _mm256_permute4x64_epi64(packed, 0b11_01_10_00);

        storeu_128!(
            <&mut [u8; 16]>::try_from(&mut dst[dst_off..dst_off + 16]).unwrap(),
            _mm256_castsi256_si128(packed)
        );
    }

    // Clear coefficients
    coeff[..512].fill(0);
}

/// FFI wrapper for 16x64 DCT_DCT 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_16x64_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, _coeff_len as usize * stride + stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_16x64_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// Full 2D DCT_DCT 64x16 inverse transform
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_64x16_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    // W=64, H=16, 4:1 ratio
    let row_clip_min = i16::MIN as i32;
    let row_clip_max = i16::MAX as i32;
    let col_clip_min = i16::MIN as i32;
    let col_clip_max = i16::MAX as i32;
    let mut tmp = [0i32; 64 * 16];

    // rect4 scaling
    // Row transform (64 elements each, 16 rows) - only first 32 columns have coefficients
    let rnd = 2;
    let shift = 2;
    for y in 0..16 {
        let mut scratch = [0i32; 64];
        for x in 0..32 {
            scratch[x] = coeff[y + x * 16] as i32;
        }
        for x in 32..64 {
            scratch[x] = 0;
        }
        dct64_1d(&mut scratch[..64], 1, row_clip_min, row_clip_max);
        for x in 0..64 {
            tmp[y * 64 + x] = iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
        }
    }

    // Column transform (16 elements each, 64 columns)
    // Use regular dct16 - all 16 rows are populated after row transform
    for x in 0..64 {
        dct16_1d(&mut tmp[x..], 64, col_clip_min, col_clip_max);
    }

    // Add to destination
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi16(bitdepth_max as i16);
    let rnd_final = _mm256_set1_epi32(8);

    for y in 0..16 {
        let dst_off = y * dst_stride;

        // Process 64 pixels in four 16-pixel chunks
        for chunk in 0..4 {
            let chunk_off = chunk * 16;
            let d = loadu_128!(<&[u8; 16]>::try_from(
                &dst[dst_off + chunk_off..dst_off + chunk_off + 16]
            )
            .unwrap());
            let d16 = _mm256_cvtepu8_epi16(d);

            let c0 = _mm256_set_epi32(
                tmp[y * 64 + chunk_off + 7],
                tmp[y * 64 + chunk_off + 6],
                tmp[y * 64 + chunk_off + 5],
                tmp[y * 64 + chunk_off + 4],
                tmp[y * 64 + chunk_off + 3],
                tmp[y * 64 + chunk_off + 2],
                tmp[y * 64 + chunk_off + 1],
                tmp[y * 64 + chunk_off + 0],
            );
            let c1 = _mm256_set_epi32(
                tmp[y * 64 + chunk_off + 15],
                tmp[y * 64 + chunk_off + 14],
                tmp[y * 64 + chunk_off + 13],
                tmp[y * 64 + chunk_off + 12],
                tmp[y * 64 + chunk_off + 11],
                tmp[y * 64 + chunk_off + 10],
                tmp[y * 64 + chunk_off + 9],
                tmp[y * 64 + chunk_off + 8],
            );

            let c0_scaled = _mm256_srai_epi32(_mm256_add_epi32(c0, rnd_final), 4);
            let c1_scaled = _mm256_srai_epi32(_mm256_add_epi32(c1, rnd_final), 4);

            let c16 = _mm256_packs_epi32(c0_scaled, c1_scaled);
            let c16 = _mm256_permute4x64_epi64(c16, 0b11_01_10_00);

            let sum = _mm256_add_epi16(d16, c16);
            let clamped = _mm256_max_epi16(_mm256_min_epi16(sum, max_val), zero);

            let packed = _mm256_packus_epi16(clamped, clamped);
            let packed = _mm256_permute4x64_epi64(packed, 0b11_01_10_00);

            storeu_128!(
                <&mut [u8; 16]>::try_from(&mut dst[dst_off + chunk_off..dst_off + chunk_off + 16])
                    .unwrap(),
                _mm256_castsi256_si128(packed)
            );
        }
    }

    // Clear coefficients
    coeff[..512].fill(0);
}

/// FFI wrapper for 64x16 DCT_DCT 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_64x16_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, _coeff_len as usize * stride + stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_64x16_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wht4_basic() {
        // WHT is used for lossless mode - test basic functionality
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let mut coeff = [0i16; 16];
        coeff[0] = 64; // DC coefficient

        let mut dst = [128u8; 16];
        let stride = 4isize;

        inv_txfm_add_wht_wht_4x4_8bpc_avx2_inner(&mut dst, 0, stride, &mut coeff, 1, 255);

        // Should have added DC to all pixels
        assert!(dst.iter().all(|&p| p >= 128));
        assert!(coeff.iter().all(|&c| c == 0));
    }
}

// ============================================================================
// ADST 4x4 TRANSFORMS
// ============================================================================

/// ADST4 coefficients (derived from spec)
/// The ADST4 uses these key values:
/// - 1321, 3803, 2482, 3344
/// The code uses (val - 4096) trick to avoid overflow

/// ADST4 1D transform applied to a single 4-element vector (returns 4 outputs)
#[inline(always)]
fn adst4_1d_scalar(in0: i32, in1: i32, in2: i32, in3: i32, min: i32, max: i32) -> (i32, i32, i32, i32) {
    let clip = |v: i32| v.clamp(min, max);

    let out0 =
        ((1321 * in0 + (3803 - 4096) * in2 + (2482 - 4096) * in3 + (3344 - 4096) * in1 + 2048)
            >> 12)
            + in2
            + in3
            + in1;
    let out1 =
        (((2482 - 4096) * in0 - 1321 * in2 - (3803 - 4096) * in3 + (3344 - 4096) * in1 + 2048)
            >> 12)
            + in0
            - in3
            + in1;
    let out2 = (209 * (in0 - in2 + in3) + 128) >> 8;
    let out3 = (((3803 - 4096) * in0 + (2482 - 4096) * in2 - 1321 * in3 - (3344 - 4096) * in1
        + 2048)
        >> 12)
        + in0
        + in2
        - in1;

    (clip(out0), clip(out1), clip(out2), clip(out3))
}

/// DCT4 1D transform (scalar, for combining with ADST)
#[inline(always)]
fn dct4_1d_scalar(in0: i32, in1: i32, in2: i32, in3: i32, min: i32, max: i32) -> (i32, i32, i32, i32) {
    let clip = |v: i32| v.clamp(min, max);
    let t0 = (in0 + in2) * 181 + 128 >> 8;
    let t1 = (in0 - in2) * 181 + 128 >> 8;
    let t2 = ((in1 * 1567 - in3 * (3784 - 4096) + 2048) >> 12) - in3;
    let t3 = ((in1 * (3784 - 4096) + in3 * 1567 + 2048) >> 12) + in1;

    (clip(t0 + t3), clip(t1 + t2), clip(t1 - t2), clip(t0 - t3))
}

/// ADST_DCT 4x4: ADST on rows, DCT on columns
#[cfg(target_arch = "x86_64")]
pub fn inv_txfm_add_adst_dct_4x4_8bpc_avx2_inner(
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    coeff: &mut [i16],
    _eob: i32,
    _bitdepth_max: i32,
) {
    // Load coefficients into a 4x4 matrix
    let mut c = [[0i32; 4]; 4];
    for y in 0..4 {
        for x in 0..4 {
            c[y][x] = coeff[y + x * 4] as i32;
        }
    }

    // First pass: ADST on rows
    let mut tmp = [[0i32; 4]; 4];
    for y in 0..4 {
        let (o0, o1, o2, o3) = adst4_1d_scalar(c[y][0], c[y][1], c[y][2], c[y][3], i16::MIN as i32, i16::MAX as i32);
        tmp[y][0] = o0;
        tmp[y][1] = o1;
        tmp[y][2] = o2;
        tmp[y][3] = o3;
    }

    // Second pass: DCT on columns
    let mut out = [[0i32; 4]; 4];
    for x in 0..4 {
        let (o0, o1, o2, o3) = dct4_1d_scalar(tmp[0][x], tmp[1][x], tmp[2][x], tmp[3][x], i16::MIN as i32, i16::MAX as i32);
        out[0][x] = o0;
        out[1][x] = o1;
        out[2][x] = o2;
        out[3][x] = o3;
    }

    // Add to destination with rounding and clipping
    for y in 0..4 {
        let row_off = dst_base.wrapping_add_signed(y as isize * dst_stride);
        for x in 0..4 {
            let pixel = dst[row_off + x] as i32;
            let val = pixel + ((out[y][x] + 8) >> 4);
            dst[row_off + x] = val.clamp(0, 255) as u8;
        }
    }

    // Clear coefficients
    coeff[..16].fill(0);
}

/// DCT_ADST 4x4: DCT on rows, ADST on columns
#[cfg(target_arch = "x86_64")]
pub fn inv_txfm_add_dct_adst_4x4_8bpc_avx2_inner(
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    coeff: &mut [i16],
    _eob: i32,
    _bitdepth_max: i32,
) {
    // Load coefficients into a 4x4 matrix
    let mut c = [[0i32; 4]; 4];
    for y in 0..4 {
        for x in 0..4 {
            c[y][x] = coeff[y + x * 4] as i32;
        }
    }

    // First pass: DCT on rows
    let mut tmp = [[0i32; 4]; 4];
    for y in 0..4 {
        let (o0, o1, o2, o3) = dct4_1d_scalar(c[y][0], c[y][1], c[y][2], c[y][3], i16::MIN as i32, i16::MAX as i32);
        tmp[y][0] = o0;
        tmp[y][1] = o1;
        tmp[y][2] = o2;
        tmp[y][3] = o3;
    }

    // Second pass: ADST on columns
    let mut out = [[0i32; 4]; 4];
    for x in 0..4 {
        let (o0, o1, o2, o3) = adst4_1d_scalar(tmp[0][x], tmp[1][x], tmp[2][x], tmp[3][x], i16::MIN as i32, i16::MAX as i32);
        out[0][x] = o0;
        out[1][x] = o1;
        out[2][x] = o2;
        out[3][x] = o3;
    }

    // Add to destination with rounding and clipping
    for y in 0..4 {
        let row_off = dst_base.wrapping_add_signed(y as isize * dst_stride);
        for x in 0..4 {
            let pixel = dst[row_off + x] as i32;
            let val = pixel + ((out[y][x] + 8) >> 4);
            dst[row_off + x] = val.clamp(0, 255) as u8;
        }
    }

    // Clear coefficients
    coeff[..16].fill(0);
}

/// ADST_ADST 4x4: ADST on both rows and columns
#[cfg(target_arch = "x86_64")]
pub fn inv_txfm_add_adst_adst_4x4_8bpc_avx2_inner(
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    coeff: &mut [i16],
    _eob: i32,
    _bitdepth_max: i32,
) {
    // Load coefficients
    let mut c = [[0i32; 4]; 4];
    for y in 0..4 {
        for x in 0..4 {
            c[y][x] = coeff[y + x * 4] as i32;
        }
    }

    // First pass: ADST on rows
    let mut tmp = [[0i32; 4]; 4];
    for y in 0..4 {
        let (o0, o1, o2, o3) = adst4_1d_scalar(c[y][0], c[y][1], c[y][2], c[y][3], i16::MIN as i32, i16::MAX as i32);
        tmp[y][0] = o0;
        tmp[y][1] = o1;
        tmp[y][2] = o2;
        tmp[y][3] = o3;
    }

    // Second pass: ADST on columns
    let mut out = [[0i32; 4]; 4];
    for x in 0..4 {
        let (o0, o1, o2, o3) = adst4_1d_scalar(tmp[0][x], tmp[1][x], tmp[2][x], tmp[3][x], i16::MIN as i32, i16::MAX as i32);
        out[0][x] = o0;
        out[1][x] = o1;
        out[2][x] = o2;
        out[3][x] = o3;
    }

    // Add to destination
    for y in 0..4 {
        let row_off = dst_base.wrapping_add_signed(y as isize * dst_stride);
        for x in 0..4 {
            let pixel = dst[row_off + x] as i32;
            let val = pixel + ((out[y][x] + 8) >> 4);
            dst[row_off + x] = val.clamp(0, 255) as u8;
        }
    }

    // Clear coefficients
    coeff[..16].fill(0);
}

// ============================================================================
// ADST FFI WRAPPERS
// ============================================================================

/// FFI wrapper for ADST_DCT 4x4 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_adst_dct_4x4_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let abs_stride = dst_stride.unsigned_abs();
    let buf_size = 3 * abs_stride + 4;
    let (base, dst_slice) = if dst_stride >= 0 {
        (0usize, unsafe {
            std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_size)
        })
    } else {
        let start = unsafe { (dst_ptr as *mut u8).offset(3 * dst_stride) };
        (3 * abs_stride, unsafe {
            std::slice::from_raw_parts_mut(start, buf_size)
        })
    };
    let coeff_slice = unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, 16) };
    inv_txfm_add_adst_dct_4x4_8bpc_avx2_inner(
        dst_slice,
        base,
        dst_stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// FFI wrapper for DCT_ADST 4x4 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_adst_4x4_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let abs_stride = dst_stride.unsigned_abs();
    let buf_size = 3 * abs_stride + 4;
    let (base, dst_slice) = if dst_stride >= 0 {
        (0usize, unsafe {
            std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_size)
        })
    } else {
        let start = unsafe { (dst_ptr as *mut u8).offset(3 * dst_stride) };
        (3 * abs_stride, unsafe {
            std::slice::from_raw_parts_mut(start, buf_size)
        })
    };
    let coeff_slice = unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, 16) };
    inv_txfm_add_dct_adst_4x4_8bpc_avx2_inner(
        dst_slice,
        base,
        dst_stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// FFI wrapper for ADST_ADST 4x4 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_adst_adst_4x4_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let abs_stride = dst_stride.unsigned_abs();
    let buf_size = 3 * abs_stride + 4;
    let (base, dst_slice) = if dst_stride >= 0 {
        (0usize, unsafe {
            std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_size)
        })
    } else {
        let start = unsafe { (dst_ptr as *mut u8).offset(3 * dst_stride) };
        (3 * abs_stride, unsafe {
            std::slice::from_raw_parts_mut(start, buf_size)
        })
    };
    let coeff_slice = unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, 16) };
    inv_txfm_add_adst_adst_4x4_8bpc_avx2_inner(
        dst_slice,
        base,
        dst_stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// FLIPADST TRANSFORMS (reverse order output)
// ============================================================================

/// FlipADST4 1D transform - same as ADST but output in reverse order
#[inline(always)]
fn flipadst4_1d_scalar(in0: i32, in1: i32, in2: i32, in3: i32, min: i32, max: i32) -> (i32, i32, i32, i32) {
    let (o0, o1, o2, o3) = adst4_1d_scalar(in0, in1, in2, in3, min, max);
    (o3, o2, o1, o0) // Flip the output order
}

/// FLIPADST_DCT 4x4: FlipADST on rows, DCT on columns
#[cfg(target_arch = "x86_64")]
pub fn inv_txfm_add_flipadst_dct_4x4_8bpc_avx2_inner(
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    coeff: &mut [i16],
    _eob: i32,
    _bitdepth_max: i32,
) {
    let mut c = [[0i32; 4]; 4];
    for y in 0..4 {
        for x in 0..4 {
            c[y][x] = coeff[y + x * 4] as i32;
        }
    }

    // First pass: FlipADST on rows
    let mut tmp = [[0i32; 4]; 4];
    for y in 0..4 {
        let (o0, o1, o2, o3) = flipadst4_1d_scalar(c[y][0], c[y][1], c[y][2], c[y][3], i16::MIN as i32, i16::MAX as i32);
        tmp[y][0] = o0;
        tmp[y][1] = o1;
        tmp[y][2] = o2;
        tmp[y][3] = o3;
    }

    // Second pass: DCT on columns
    let mut out = [[0i32; 4]; 4];
    for x in 0..4 {
        let (o0, o1, o2, o3) = dct4_1d_scalar(tmp[0][x], tmp[1][x], tmp[2][x], tmp[3][x], i16::MIN as i32, i16::MAX as i32);
        out[0][x] = o0;
        out[1][x] = o1;
        out[2][x] = o2;
        out[3][x] = o3;
    }

    for y in 0..4 {
        let row_off = dst_base.wrapping_add_signed(y as isize * dst_stride);
        for x in 0..4 {
            let pixel = dst[row_off + x] as i32;
            let val = pixel + ((out[y][x] + 8) >> 4);
            dst[row_off + x] = val.clamp(0, 255) as u8;
        }
    }

    coeff[..16].fill(0);
}

/// DCT_FLIPADST 4x4: DCT on rows, FlipADST on columns
#[cfg(target_arch = "x86_64")]
pub fn inv_txfm_add_dct_flipadst_4x4_8bpc_avx2_inner(
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    coeff: &mut [i16],
    _eob: i32,
    _bitdepth_max: i32,
) {
    let mut c = [[0i32; 4]; 4];
    for y in 0..4 {
        for x in 0..4 {
            c[y][x] = coeff[y + x * 4] as i32;
        }
    }

    // First pass: DCT on rows
    let mut tmp = [[0i32; 4]; 4];
    for y in 0..4 {
        let (o0, o1, o2, o3) = dct4_1d_scalar(c[y][0], c[y][1], c[y][2], c[y][3], i16::MIN as i32, i16::MAX as i32);
        tmp[y][0] = o0;
        tmp[y][1] = o1;
        tmp[y][2] = o2;
        tmp[y][3] = o3;
    }

    // Second pass: FlipADST on columns
    let mut out = [[0i32; 4]; 4];
    for x in 0..4 {
        let (o0, o1, o2, o3) = flipadst4_1d_scalar(tmp[0][x], tmp[1][x], tmp[2][x], tmp[3][x], i16::MIN as i32, i16::MAX as i32);
        out[0][x] = o0;
        out[1][x] = o1;
        out[2][x] = o2;
        out[3][x] = o3;
    }

    for y in 0..4 {
        let row_off = dst_base.wrapping_add_signed(y as isize * dst_stride);
        for x in 0..4 {
            let pixel = dst[row_off + x] as i32;
            let val = pixel + ((out[y][x] + 8) >> 4);
            dst[row_off + x] = val.clamp(0, 255) as u8;
        }
    }

    coeff[..16].fill(0);
}

/// ADST_FLIPADST 4x4
#[cfg(target_arch = "x86_64")]
pub fn inv_txfm_add_adst_flipadst_4x4_8bpc_avx2_inner(
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    coeff: &mut [i16],
    _eob: i32,
    _bitdepth_max: i32,
) {
    let mut c = [[0i32; 4]; 4];
    for y in 0..4 {
        for x in 0..4 {
            c[y][x] = coeff[y + x * 4] as i32;
        }
    }

    let mut tmp = [[0i32; 4]; 4];
    for y in 0..4 {
        let (o0, o1, o2, o3) = adst4_1d_scalar(c[y][0], c[y][1], c[y][2], c[y][3], i16::MIN as i32, i16::MAX as i32);
        tmp[y][0] = o0;
        tmp[y][1] = o1;
        tmp[y][2] = o2;
        tmp[y][3] = o3;
    }

    let mut out = [[0i32; 4]; 4];
    for x in 0..4 {
        let (o0, o1, o2, o3) = flipadst4_1d_scalar(tmp[0][x], tmp[1][x], tmp[2][x], tmp[3][x], i16::MIN as i32, i16::MAX as i32);
        out[0][x] = o0;
        out[1][x] = o1;
        out[2][x] = o2;
        out[3][x] = o3;
    }

    for y in 0..4 {
        let row_off = dst_base.wrapping_add_signed(y as isize * dst_stride);
        for x in 0..4 {
            let pixel = dst[row_off + x] as i32;
            let val = pixel + ((out[y][x] + 8) >> 4);
            dst[row_off + x] = val.clamp(0, 255) as u8;
        }
    }
    coeff[..16].fill(0);
}

/// FLIPADST_ADST 4x4
#[cfg(target_arch = "x86_64")]
pub fn inv_txfm_add_flipadst_adst_4x4_8bpc_avx2_inner(
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    coeff: &mut [i16],
    _eob: i32,
    _bitdepth_max: i32,
) {
    let mut c = [[0i32; 4]; 4];
    for y in 0..4 {
        for x in 0..4 {
            c[y][x] = coeff[y + x * 4] as i32;
        }
    }

    let mut tmp = [[0i32; 4]; 4];
    for y in 0..4 {
        let (o0, o1, o2, o3) = flipadst4_1d_scalar(c[y][0], c[y][1], c[y][2], c[y][3], i16::MIN as i32, i16::MAX as i32);
        tmp[y][0] = o0;
        tmp[y][1] = o1;
        tmp[y][2] = o2;
        tmp[y][3] = o3;
    }

    let mut out = [[0i32; 4]; 4];
    for x in 0..4 {
        let (o0, o1, o2, o3) = adst4_1d_scalar(tmp[0][x], tmp[1][x], tmp[2][x], tmp[3][x], i16::MIN as i32, i16::MAX as i32);
        out[0][x] = o0;
        out[1][x] = o1;
        out[2][x] = o2;
        out[3][x] = o3;
    }

    for y in 0..4 {
        let row_off = dst_base.wrapping_add_signed(y as isize * dst_stride);
        for x in 0..4 {
            let pixel = dst[row_off + x] as i32;
            let val = pixel + ((out[y][x] + 8) >> 4);
            dst[row_off + x] = val.clamp(0, 255) as u8;
        }
    }
    coeff[..16].fill(0);
}

/// FLIPADST_FLIPADST 4x4
#[cfg(target_arch = "x86_64")]
pub fn inv_txfm_add_flipadst_flipadst_4x4_8bpc_avx2_inner(
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    coeff: &mut [i16],
    _eob: i32,
    _bitdepth_max: i32,
) {
    let mut c = [[0i32; 4]; 4];
    for y in 0..4 {
        for x in 0..4 {
            c[y][x] = coeff[y + x * 4] as i32;
        }
    }

    let mut tmp = [[0i32; 4]; 4];
    for y in 0..4 {
        let (o0, o1, o2, o3) = flipadst4_1d_scalar(c[y][0], c[y][1], c[y][2], c[y][3], i16::MIN as i32, i16::MAX as i32);
        tmp[y][0] = o0;
        tmp[y][1] = o1;
        tmp[y][2] = o2;
        tmp[y][3] = o3;
    }

    let mut out = [[0i32; 4]; 4];
    for x in 0..4 {
        let (o0, o1, o2, o3) = flipadst4_1d_scalar(tmp[0][x], tmp[1][x], tmp[2][x], tmp[3][x], i16::MIN as i32, i16::MAX as i32);
        out[0][x] = o0;
        out[1][x] = o1;
        out[2][x] = o2;
        out[3][x] = o3;
    }

    for y in 0..4 {
        let row_off = dst_base.wrapping_add_signed(y as isize * dst_stride);
        for x in 0..4 {
            let pixel = dst[row_off + x] as i32;
            let val = pixel + ((out[y][x] + 8) >> 4);
            dst[row_off + x] = val.clamp(0, 255) as u8;
        }
    }
    coeff[..16].fill(0);
}

// FFI wrappers for FlipADST variants
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_flipadst_dct_4x4_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let abs_stride = dst_stride.unsigned_abs();
    let buf_size = 3 * abs_stride + 4;
    let (base, dst_slice) = if dst_stride >= 0 {
        (0usize, unsafe {
            std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_size)
        })
    } else {
        let start = unsafe { (dst_ptr as *mut u8).offset(3 * dst_stride) };
        (3 * abs_stride, unsafe {
            std::slice::from_raw_parts_mut(start, buf_size)
        })
    };
    let coeff_slice = unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, 16) };
    inv_txfm_add_flipadst_dct_4x4_8bpc_avx2_inner(
        dst_slice,
        base,
        dst_stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_flipadst_4x4_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let abs_stride = dst_stride.unsigned_abs();
    let buf_size = 3 * abs_stride + 4;
    let (base, dst_slice) = if dst_stride >= 0 {
        (0usize, unsafe {
            std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_size)
        })
    } else {
        let start = unsafe { (dst_ptr as *mut u8).offset(3 * dst_stride) };
        (3 * abs_stride, unsafe {
            std::slice::from_raw_parts_mut(start, buf_size)
        })
    };
    let coeff_slice = unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, 16) };
    inv_txfm_add_dct_flipadst_4x4_8bpc_avx2_inner(
        dst_slice,
        base,
        dst_stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_adst_flipadst_4x4_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let abs_stride = dst_stride.unsigned_abs();
    let buf_size = 3 * abs_stride + 4;
    let (base, dst_slice) = if dst_stride >= 0 {
        (0usize, unsafe {
            std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_size)
        })
    } else {
        let start = unsafe { (dst_ptr as *mut u8).offset(3 * dst_stride) };
        (3 * abs_stride, unsafe {
            std::slice::from_raw_parts_mut(start, buf_size)
        })
    };
    let coeff_slice = unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, 16) };
    inv_txfm_add_adst_flipadst_4x4_8bpc_avx2_inner(
        dst_slice,
        base,
        dst_stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_flipadst_adst_4x4_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let abs_stride = dst_stride.unsigned_abs();
    let buf_size = 3 * abs_stride + 4;
    let (base, dst_slice) = if dst_stride >= 0 {
        (0usize, unsafe {
            std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_size)
        })
    } else {
        let start = unsafe { (dst_ptr as *mut u8).offset(3 * dst_stride) };
        (3 * abs_stride, unsafe {
            std::slice::from_raw_parts_mut(start, buf_size)
        })
    };
    let coeff_slice = unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, 16) };
    inv_txfm_add_flipadst_adst_4x4_8bpc_avx2_inner(
        dst_slice,
        base,
        dst_stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_flipadst_flipadst_4x4_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let abs_stride = dst_stride.unsigned_abs();
    let buf_size = 3 * abs_stride + 4;
    let (base, dst_slice) = if dst_stride >= 0 {
        (0usize, unsafe {
            std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_size)
        })
    } else {
        let start = unsafe { (dst_ptr as *mut u8).offset(3 * dst_stride) };
        (3 * abs_stride, unsafe {
            std::slice::from_raw_parts_mut(start, buf_size)
        })
    };
    let coeff_slice = unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, 16) };
    inv_txfm_add_flipadst_flipadst_4x4_8bpc_avx2_inner(
        dst_slice,
        base,
        dst_stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// ADST 8x8 TRANSFORMS
// ============================================================================

/// ADST8 1D transform (scalar)
#[inline(always)]
fn adst8_1d_scalar(
    in0: i32,
    in1: i32,
    in2: i32,
    in3: i32,
    in4: i32,
    in5: i32,
    in6: i32,
    in7: i32,
    min: i32,
    max: i32,
) -> (i32, i32, i32, i32, i32, i32, i32, i32) {
    let clip = |v: i32| v.clamp(min, max);

    let t0a = (((4076 - 4096) * in7 + 401 * in0 + 2048) >> 12) + in7;
    let t1a = ((401 * in7 - (4076 - 4096) * in0 + 2048) >> 12) - in0;
    let t2a = (((3612 - 4096) * in5 + 1931 * in2 + 2048) >> 12) + in5;
    let t3a = ((1931 * in5 - (3612 - 4096) * in2 + 2048) >> 12) - in2;
    let t4a = (1299 * in3 + 1583 * in4 + 1024) >> 11;
    let t5a = (1583 * in3 - 1299 * in4 + 1024) >> 11;
    let t6a = ((1189 * in1 + (3920 - 4096) * in6 + 2048) >> 12) + in6;
    let t7a = (((3920 - 4096) * in1 - 1189 * in6 + 2048) >> 12) + in1;

    let t0 = clip(t0a + t4a);
    let t1 = clip(t1a + t5a);
    let t2 = clip(t2a + t6a);
    let t3 = clip(t3a + t7a);
    let t4 = clip(t0a - t4a);
    let t5 = clip(t1a - t5a);
    let t6 = clip(t2a - t6a);
    let t7 = clip(t3a - t7a);

    let t4a = (((3784 - 4096) * t4 + 1567 * t5 + 2048) >> 12) + t4;
    let t5a = ((1567 * t4 - (3784 - 4096) * t5 + 2048) >> 12) - t5;
    let t6a = (((3784 - 4096) * t7 - 1567 * t6 + 2048) >> 12) + t7;
    let t7a = ((1567 * t7 + (3784 - 4096) * t6 + 2048) >> 12) + t6;

    let out0 = clip(t0 + t2);
    let out7 = -clip(t1 + t3);
    let t2_final = clip(t0 - t2);
    let t3_final = clip(t1 - t3);
    let out1 = -clip(t4a + t6a);
    let out6 = clip(t5a + t7a);
    let t6_final = clip(t4a - t6a);
    let t7_final = clip(t5a - t7a);

    let out3 = -(((t2_final + t3_final) * 181 + 128) >> 8);
    let out4 = ((t2_final - t3_final) * 181 + 128) >> 8;
    let out2 = ((t6_final + t7_final) * 181 + 128) >> 8;
    let out5 = -(((t6_final - t7_final) * 181 + 128) >> 8);

    (out0, out1, out2, out3, out4, out5, out6, out7)
}

/// FlipADST8 1D transform - ADST8 with reversed output
#[inline(always)]
fn flipadst8_1d_scalar(
    in0: i32,
    in1: i32,
    in2: i32,
    in3: i32,
    in4: i32,
    in5: i32,
    in6: i32,
    in7: i32,
    min: i32,
    max: i32,
) -> (i32, i32, i32, i32, i32, i32, i32, i32) {
    let (o0, o1, o2, o3, o4, o5, o6, o7) =
        adst8_1d_scalar(in0, in1, in2, in3, in4, in5, in6, in7, min, max);
    (o7, o6, o5, o4, o3, o2, o1, o0)
}

/// DCT8 1D transform (scalar)
#[inline(always)]
fn dct8_1d_scalar(
    in0: i32,
    in1: i32,
    in2: i32,
    in3: i32,
    in4: i32,
    in5: i32,
    in6: i32,
    in7: i32,
    min: i32,
    max: i32,
) -> (i32, i32, i32, i32, i32, i32, i32, i32) {
    let clip = |v: i32| v.clamp(min, max);

    // First do DCT4 on even samples
    let t0 = ((in0 + in4) * 181 + 128) >> 8;
    let t1 = ((in0 - in4) * 181 + 128) >> 8;
    let t2 = ((in2 * 1567 - in6 * (3784 - 4096) + 2048) >> 12) - in6;
    let t3 = ((in2 * (3784 - 4096) + in6 * 1567 + 2048) >> 12) + in2;

    let t0a = clip(t0 + t3);
    let t1a = clip(t1 + t2);
    let t2a = clip(t1 - t2);
    let t3a = clip(t0 - t3);

    // Then do the 8-point specific part
    let t4a = ((in1 * 799 - in7 * (4017 - 4096) + 2048) >> 12) - in7;
    let t5a = (in5 * 1703 - in3 * 1138 + 1024) >> 11;
    let t6a = (in5 * 1138 + in3 * 1703 + 1024) >> 11;
    let t7a = ((in1 * (4017 - 4096) + in7 * 799 + 2048) >> 12) + in1;

    let t4 = clip(t4a + t5a);
    let t5 = clip(t4a - t5a);
    let t7 = clip(t7a + t6a);
    let t6 = clip(t7a - t6a);

    let t5b = ((t6 - t5) * 181 + 128) >> 8;
    let t6b = ((t6 + t5) * 181 + 128) >> 8;

    (
        clip(t0a + t7),
        clip(t1a + t6b),
        clip(t2a + t5b),
        clip(t3a + t4),
        clip(t3a - t4),
        clip(t2a - t5b),
        clip(t1a - t6b),
        clip(t0a - t7),
    )
}

/// Helper macro for 8x8 transform implementations
macro_rules! impl_8x8_transform {
    ($name:ident, $row_fn:ident, $col_fn:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[arcane]
        pub fn $name(
            _token: Desktop64,
            dst: &mut [u8],
            dst_stride: usize,
            coeff: &mut [i16],
            _eob: i32,
            _bitdepth_max: i32,
        ) {
            use crate::src::safe_simd::pixel_access::{
                loadi32, loadi64, loadu_128, storei32, storei64, storeu_128,
            };
            let mut dst = dst.flex_mut();
            let mut coeff = coeff.flex_mut();
            const MIN: i32 = i16::MIN as i32;
            const MAX: i32 = i16::MAX as i32;

            // Load coefficients
            let mut c = [[0i32; 8]; 8];
            for y in 0..8 {
                for x in 0..8 {
                    c[y][x] = coeff[y * 8 + x] as i32;
                }
            }

            // First pass: transform on rows
            let mut tmp = [[0i32; 8]; 8];
            for y in 0..8 {
                let (o0, o1, o2, o3, o4, o5, o6, o7) = $row_fn(
                    c[y][0], c[y][1], c[y][2], c[y][3], c[y][4], c[y][5], c[y][6], c[y][7], MIN,
                    MAX,
                );
                tmp[y][0] = o0;
                tmp[y][1] = o1;
                tmp[y][2] = o2;
                tmp[y][3] = o3;
                tmp[y][4] = o4;
                tmp[y][5] = o5;
                tmp[y][6] = o6;
                tmp[y][7] = o7;
            }

            // Second pass: transform on columns
            let mut out = [[0i32; 8]; 8];
            for x in 0..8 {
                let (o0, o1, o2, o3, o4, o5, o6, o7) = $col_fn(
                    tmp[0][x], tmp[1][x], tmp[2][x], tmp[3][x], tmp[4][x], tmp[5][x], tmp[6][x],
                    tmp[7][x], MIN, MAX,
                );
                out[0][x] = o0;
                out[1][x] = o1;
                out[2][x] = o2;
                out[3][x] = o3;
                out[4][x] = o4;
                out[5][x] = o5;
                out[6][x] = o6;
                out[7][x] = o7;
            }

            // Add to destination with rounding
            for y in 0..8 {
                let dst_off = y * dst_stride;
                for x in 0..8 {
                    let pixel = dst[dst_off + x] as i32;
                    let val = pixel + ((out[y][x] + 8) >> 4);
                    dst[dst_off + x] = val.clamp(0, 255) as u8;
                }
            }

            // Clear coefficients
            coeff[..64].fill(0);
        }
    };
}

// Generate all 8x8 ADST/FlipADST combinations
impl_8x8_transform!(
    inv_txfm_add_adst_dct_8x8_8bpc_avx2_inner,
    adst8_1d_scalar,
    dct8_1d_scalar
);
impl_8x8_transform!(
    inv_txfm_add_dct_adst_8x8_8bpc_avx2_inner,
    dct8_1d_scalar,
    adst8_1d_scalar
);
impl_8x8_transform!(
    inv_txfm_add_adst_adst_8x8_8bpc_avx2_inner,
    adst8_1d_scalar,
    adst8_1d_scalar
);
impl_8x8_transform!(
    inv_txfm_add_flipadst_dct_8x8_8bpc_avx2_inner,
    flipadst8_1d_scalar,
    dct8_1d_scalar
);
impl_8x8_transform!(
    inv_txfm_add_dct_flipadst_8x8_8bpc_avx2_inner,
    dct8_1d_scalar,
    flipadst8_1d_scalar
);
impl_8x8_transform!(
    inv_txfm_add_flipadst_flipadst_8x8_8bpc_avx2_inner,
    flipadst8_1d_scalar,
    flipadst8_1d_scalar
);
impl_8x8_transform!(
    inv_txfm_add_adst_flipadst_8x8_8bpc_avx2_inner,
    adst8_1d_scalar,
    flipadst8_1d_scalar
);
impl_8x8_transform!(
    inv_txfm_add_flipadst_adst_8x8_8bpc_avx2_inner,
    flipadst8_1d_scalar,
    adst8_1d_scalar
);

// FFI wrappers for 8x8 transforms
macro_rules! impl_8x8_ffi_wrapper {
    ($wrapper:ident, $inner:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2")]
        #[cfg(feature = "asm")]
        pub unsafe extern "C" fn $wrapper(
            dst_ptr: *mut DynPixel,
            dst_stride: isize,
            coeff: *mut DynCoef,
            eob: c_int,
            bitdepth_max: c_int,
            _coeff_len: u16,
            _dst: *const FFISafe<PicOffset>,
        ) {
            let _token = unsafe { Desktop64::forge_token_dangerously() };
            let stride = dst_stride as usize;

            let dst_slice = unsafe {
                std::slice::from_raw_parts_mut(
                    dst_ptr as *mut u8,
                    _coeff_len as usize * stride + stride,
                )
            };

            let coeff_slice =
                unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

            $inner(_token, dst_slice, stride, coeff_slice, eob, bitdepth_max);
        }
    };
}

impl_8x8_ffi_wrapper!(
    inv_txfm_add_adst_dct_8x8_8bpc_avx2,
    inv_txfm_add_adst_dct_8x8_8bpc_avx2_inner
);
impl_8x8_ffi_wrapper!(
    inv_txfm_add_dct_adst_8x8_8bpc_avx2,
    inv_txfm_add_dct_adst_8x8_8bpc_avx2_inner
);
impl_8x8_ffi_wrapper!(
    inv_txfm_add_adst_adst_8x8_8bpc_avx2,
    inv_txfm_add_adst_adst_8x8_8bpc_avx2_inner
);
impl_8x8_ffi_wrapper!(
    inv_txfm_add_flipadst_dct_8x8_8bpc_avx2,
    inv_txfm_add_flipadst_dct_8x8_8bpc_avx2_inner
);
impl_8x8_ffi_wrapper!(
    inv_txfm_add_dct_flipadst_8x8_8bpc_avx2,
    inv_txfm_add_dct_flipadst_8x8_8bpc_avx2_inner
);
impl_8x8_ffi_wrapper!(
    inv_txfm_add_flipadst_flipadst_8x8_8bpc_avx2,
    inv_txfm_add_flipadst_flipadst_8x8_8bpc_avx2_inner
);
impl_8x8_ffi_wrapper!(
    inv_txfm_add_adst_flipadst_8x8_8bpc_avx2,
    inv_txfm_add_adst_flipadst_8x8_8bpc_avx2_inner
);
impl_8x8_ffi_wrapper!(
    inv_txfm_add_flipadst_adst_8x8_8bpc_avx2,
    inv_txfm_add_flipadst_adst_8x8_8bpc_avx2_inner
);

// ============================================================================
// V_ADST/H_ADST TRANSFORMS (Identity + ADST combinations)
// ============================================================================

/// Identity transform 4x4 - just pass through values (no transform)
#[inline(always)]
fn identity4_1d_scalar(in0: i32, in1: i32, in2: i32, in3: i32, _min: i32, _max: i32) -> (i32, i32, i32, i32) {
    // identity4(x) = x + (x * 1697 + 2048) >> 12 ≈ x * sqrt(2)
    // Matches rav1d_inv_identity4_1d_c in itx_1d.rs (which also ignores min/max)
    let o0 = in0 + ((in0 * 1697 + 2048) >> 12);
    let o1 = in1 + ((in1 * 1697 + 2048) >> 12);
    let o2 = in2 + ((in2 * 1697 + 2048) >> 12);
    let o3 = in3 + ((in3 * 1697 + 2048) >> 12);
    (o0, o1, o2, o3)
}

/// V_ADST 4x4: Identity on rows, ADST on columns
#[cfg(target_arch = "x86_64")]
pub fn inv_txfm_add_v_adst_4x4_8bpc_avx2_inner(
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    coeff: &mut [i16],
    _eob: i32,
    _bitdepth_max: i32,
) {
    let mut c = [[0i32; 4]; 4];
    for y in 0..4 {
        for x in 0..4 {
            c[y][x] = coeff[y + x * 4] as i32;
        }
    }

    // First pass: Identity on rows
    let mut tmp = [[0i32; 4]; 4];
    for y in 0..4 {
        let (o0, o1, o2, o3) = identity4_1d_scalar(c[y][0], c[y][1], c[y][2], c[y][3], i16::MIN as i32, i16::MAX as i32);
        tmp[y][0] = o0;
        tmp[y][1] = o1;
        tmp[y][2] = o2;
        tmp[y][3] = o3;
    }

    // Intermediate clamp (shift=0 for 4x4, clamp to col_clip range)
    let col_clip_min = i16::MIN as i32;
    let col_clip_max = i16::MAX as i32;
    for y in 0..4 {
        for x in 0..4 {
            tmp[y][x] = tmp[y][x].clamp(col_clip_min, col_clip_max);
        }
    }

    // Second pass: ADST on columns
    let mut out = [[0i32; 4]; 4];
    for x in 0..4 {
        let (o0, o1, o2, o3) = adst4_1d_scalar(tmp[0][x], tmp[1][x], tmp[2][x], tmp[3][x], col_clip_min, col_clip_max);
        out[0][x] = o0;
        out[1][x] = o1;
        out[2][x] = o2;
        out[3][x] = o3;
    }

    for y in 0..4 {
        let row_off = dst_base.wrapping_add_signed(y as isize * dst_stride);
        for x in 0..4 {
            let pixel = dst[row_off + x] as i32;
            let val = pixel + ((out[y][x] + 8) >> 4);
            dst[row_off + x] = val.clamp(0, 255) as u8;
        }
    }
    coeff[..16].fill(0);
}

/// H_ADST 4x4: ADST on rows, Identity on columns
#[cfg(target_arch = "x86_64")]
pub fn inv_txfm_add_h_adst_4x4_8bpc_avx2_inner(
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    coeff: &mut [i16],
    _eob: i32,
    _bitdepth_max: i32,
) {
    let mut c = [[0i32; 4]; 4];
    for y in 0..4 {
        for x in 0..4 {
            c[y][x] = coeff[y + x * 4] as i32;
        }
    }

    // First pass: ADST on rows (H_ADST = first=Adst, second=Identity per reference)
    let mut tmp = [[0i32; 4]; 4];
    for y in 0..4 {
        let (o0, o1, o2, o3) = adst4_1d_scalar(c[y][0], c[y][1], c[y][2], c[y][3], i16::MIN as i32, i16::MAX as i32);
        tmp[y][0] = o0;
        tmp[y][1] = o1;
        tmp[y][2] = o2;
        tmp[y][3] = o3;
    }

    // Second pass: Identity on columns
    let mut out = [[0i32; 4]; 4];
    for x in 0..4 {
        let (o0, o1, o2, o3) = identity4_1d_scalar(tmp[0][x], tmp[1][x], tmp[2][x], tmp[3][x], i16::MIN as i32, i16::MAX as i32);
        out[0][x] = o0;
        out[1][x] = o1;
        out[2][x] = o2;
        out[3][x] = o3;
    }

    for y in 0..4 {
        let row_off = dst_base.wrapping_add_signed(y as isize * dst_stride);
        for x in 0..4 {
            let pixel = dst[row_off + x] as i32;
            let val = pixel + ((out[y][x] + 8) >> 4);
            dst[row_off + x] = val.clamp(0, 255) as u8;
        }
    }
    coeff[..16].fill(0);
}

/// V_FLIPADST 4x4: Identity on rows, FlipADST on columns
#[cfg(target_arch = "x86_64")]
pub fn inv_txfm_add_v_flipadst_4x4_8bpc_avx2_inner(
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    coeff: &mut [i16],
    _eob: i32,
    _bitdepth_max: i32,
) {
    let mut c = [[0i32; 4]; 4];
    for y in 0..4 {
        for x in 0..4 {
            c[y][x] = coeff[y + x * 4] as i32;
        }
    }

    let mut tmp = [[0i32; 4]; 4];
    for y in 0..4 {
        let (o0, o1, o2, o3) = identity4_1d_scalar(c[y][0], c[y][1], c[y][2], c[y][3], i16::MIN as i32, i16::MAX as i32);
        tmp[y][0] = o0;
        tmp[y][1] = o1;
        tmp[y][2] = o2;
        tmp[y][3] = o3;
    }

    // Intermediate clamp (shift=0 for 4x4, clamp to col_clip range)
    let col_clip_min = i16::MIN as i32;
    let col_clip_max = i16::MAX as i32;
    for y in 0..4 {
        for x in 0..4 {
            tmp[y][x] = tmp[y][x].clamp(col_clip_min, col_clip_max);
        }
    }

    let mut out = [[0i32; 4]; 4];
    for x in 0..4 {
        let (o0, o1, o2, o3) = flipadst4_1d_scalar(tmp[0][x], tmp[1][x], tmp[2][x], tmp[3][x], col_clip_min, col_clip_max);
        out[0][x] = o0;
        out[1][x] = o1;
        out[2][x] = o2;
        out[3][x] = o3;
    }

    for y in 0..4 {
        let row_off = dst_base.wrapping_add_signed(y as isize * dst_stride);
        for x in 0..4 {
            let pixel = dst[row_off + x] as i32;
            let val = pixel + ((out[y][x] + 8) >> 4);
            dst[row_off + x] = val.clamp(0, 255) as u8;
        }
    }
    coeff[..16].fill(0);
}

/// H_FLIPADST 4x4: Identity on rows, FlipADST on columns
#[cfg(target_arch = "x86_64")]
pub fn inv_txfm_add_h_flipadst_4x4_8bpc_avx2_inner(
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    coeff: &mut [i16],
    _eob: i32,
    _bitdepth_max: i32,
) {
    let mut c = [[0i32; 4]; 4];
    for y in 0..4 {
        for x in 0..4 {
            c[y][x] = coeff[y + x * 4] as i32;
        }
    }

    // First pass: FlipADST on rows (H_FLIPADST = first=FlipAdst, second=Identity per reference)
    let mut tmp = [[0i32; 4]; 4];
    for y in 0..4 {
        let (o0, o1, o2, o3) = flipadst4_1d_scalar(c[y][0], c[y][1], c[y][2], c[y][3], i16::MIN as i32, i16::MAX as i32);
        tmp[y][0] = o0;
        tmp[y][1] = o1;
        tmp[y][2] = o2;
        tmp[y][3] = o3;
    }

    // Second pass: Identity on columns
    let mut out = [[0i32; 4]; 4];
    for x in 0..4 {
        let (o0, o1, o2, o3) = identity4_1d_scalar(tmp[0][x], tmp[1][x], tmp[2][x], tmp[3][x], i16::MIN as i32, i16::MAX as i32);
        out[0][x] = o0;
        out[1][x] = o1;
        out[2][x] = o2;
        out[3][x] = o3;
    }

    for y in 0..4 {
        let row_off = dst_base.wrapping_add_signed(y as isize * dst_stride);
        for x in 0..4 {
            let pixel = dst[row_off + x] as i32;
            let val = pixel + ((out[y][x] + 8) >> 4);
            dst[row_off + x] = val.clamp(0, 255) as u8;
        }
    }
    coeff[..16].fill(0);
}

// FFI wrappers for V/H ADST
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_adst_4x4_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let abs_stride = dst_stride.unsigned_abs();
    let buf_size = 3 * abs_stride + 4;
    let (base, dst_slice) = if dst_stride >= 0 {
        (0usize, unsafe {
            std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_size)
        })
    } else {
        let start = unsafe { (dst_ptr as *mut u8).offset(3 * dst_stride) };
        (3 * abs_stride, unsafe {
            std::slice::from_raw_parts_mut(start, buf_size)
        })
    };
    let coeff_slice = unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, 16) };
    inv_txfm_add_h_adst_4x4_8bpc_avx2_inner(
        dst_slice,
        base,
        dst_stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_adst_identity_4x4_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let abs_stride = dst_stride.unsigned_abs();
    let buf_size = 3 * abs_stride + 4;
    let (base, dst_slice) = if dst_stride >= 0 {
        (0usize, unsafe {
            std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_size)
        })
    } else {
        let start = unsafe { (dst_ptr as *mut u8).offset(3 * dst_stride) };
        (3 * abs_stride, unsafe {
            std::slice::from_raw_parts_mut(start, buf_size)
        })
    };
    let coeff_slice = unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, 16) };
    inv_txfm_add_v_adst_4x4_8bpc_avx2_inner(
        dst_slice,
        base,
        dst_stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_flipadst_4x4_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let abs_stride = dst_stride.unsigned_abs();
    let buf_size = 3 * abs_stride + 4;
    let (base, dst_slice) = if dst_stride >= 0 {
        (0usize, unsafe {
            std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_size)
        })
    } else {
        let start = unsafe { (dst_ptr as *mut u8).offset(3 * dst_stride) };
        (3 * abs_stride, unsafe {
            std::slice::from_raw_parts_mut(start, buf_size)
        })
    };
    let coeff_slice = unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, 16) };
    inv_txfm_add_h_flipadst_4x4_8bpc_avx2_inner(
        dst_slice,
        base,
        dst_stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_flipadst_identity_4x4_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let abs_stride = dst_stride.unsigned_abs();
    let buf_size = 3 * abs_stride + 4;
    let (base, dst_slice) = if dst_stride >= 0 {
        (0usize, unsafe {
            std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_size)
        })
    } else {
        let start = unsafe { (dst_ptr as *mut u8).offset(3 * dst_stride) };
        (3 * abs_stride, unsafe {
            std::slice::from_raw_parts_mut(start, buf_size)
        })
    };
    let coeff_slice = unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, 16) };
    inv_txfm_add_v_flipadst_4x4_8bpc_avx2_inner(
        dst_slice,
        base,
        dst_stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// V_DCT/H_DCT TRANSFORMS (DCT + Identity combinations)
// ============================================================================

/// H_DCT 4x4: DCT on rows, Identity on columns
#[cfg(target_arch = "x86_64")]
pub fn inv_txfm_add_dct_identity_4x4_8bpc_avx2_inner(
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    coeff: &mut [i16],
    _eob: i32,
    _bitdepth_max: i32,
) {
    let mut c = [[0i32; 4]; 4];
    for y in 0..4 {
        for x in 0..4 {
            c[y][x] = coeff[y + x * 4] as i32;
        }
    }

    // First pass: DCT on rows
    let mut tmp = [[0i32; 4]; 4];
    for y in 0..4 {
        let (o0, o1, o2, o3) = dct4_1d_scalar(c[y][0], c[y][1], c[y][2], c[y][3], i16::MIN as i32, i16::MAX as i32);
        tmp[y][0] = o0;
        tmp[y][1] = o1;
        tmp[y][2] = o2;
        tmp[y][3] = o3;
    }

    // Second pass: Identity on columns
    let mut out = [[0i32; 4]; 4];
    for x in 0..4 {
        let (o0, o1, o2, o3) = identity4_1d_scalar(tmp[0][x], tmp[1][x], tmp[2][x], tmp[3][x], i16::MIN as i32, i16::MAX as i32);
        out[0][x] = o0;
        out[1][x] = o1;
        out[2][x] = o2;
        out[3][x] = o3;
    }

    for y in 0..4 {
        let row_off = dst_base.wrapping_add_signed(y as isize * dst_stride);
        for x in 0..4 {
            let pixel = dst[row_off + x] as i32;
            let val = pixel + ((out[y][x] + 8) >> 4);
            dst[row_off + x] = val.clamp(0, 255) as u8;
        }
    }
    coeff[..16].fill(0);
}

/// V_DCT 4x4: Identity on rows, DCT on columns
#[cfg(target_arch = "x86_64")]
pub fn inv_txfm_add_identity_dct_4x4_8bpc_avx2_inner(
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    coeff: &mut [i16],
    _eob: i32,
    _bitdepth_max: i32,
) {
    let mut c = [[0i32; 4]; 4];
    for y in 0..4 {
        for x in 0..4 {
            c[y][x] = coeff[y + x * 4] as i32;
        }
    }

    // First pass: Identity on rows
    let mut tmp = [[0i32; 4]; 4];
    for y in 0..4 {
        let (o0, o1, o2, o3) = identity4_1d_scalar(c[y][0], c[y][1], c[y][2], c[y][3], i16::MIN as i32, i16::MAX as i32);
        tmp[y][0] = o0;
        tmp[y][1] = o1;
        tmp[y][2] = o2;
        tmp[y][3] = o3;
    }

    // Intermediate clamp (shift=0 for 4x4, just clamp to col_clip range)
    let col_clip_min = i16::MIN as i32;
    let col_clip_max = i16::MAX as i32;
    for y in 0..4 {
        for x in 0..4 {
            tmp[y][x] = tmp[y][x].clamp(col_clip_min, col_clip_max);
        }
    }

    // Second pass: DCT on columns
    let mut out = [[0i32; 4]; 4];
    for x in 0..4 {
        let (o0, o1, o2, o3) = dct4_1d_scalar(tmp[0][x], tmp[1][x], tmp[2][x], tmp[3][x], col_clip_min, col_clip_max);
        out[0][x] = o0;
        out[1][x] = o1;
        out[2][x] = o2;
        out[3][x] = o3;
    }

    for y in 0..4 {
        let row_off = dst_base.wrapping_add_signed(y as isize * dst_stride);
        for x in 0..4 {
            let pixel = dst[row_off + x] as i32;
            let val = pixel + ((out[y][x] + 8) >> 4);
            dst[row_off + x] = val.clamp(0, 255) as u8;
        }
    }
    coeff[..16].fill(0);
}

// FFI wrappers for V/H DCT
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_identity_4x4_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let abs_stride = dst_stride.unsigned_abs();
    let buf_size = 3 * abs_stride + 4;
    let (base, dst_slice) = if dst_stride >= 0 {
        (0usize, unsafe {
            std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_size)
        })
    } else {
        let start = unsafe { (dst_ptr as *mut u8).offset(3 * dst_stride) };
        (3 * abs_stride, unsafe {
            std::slice::from_raw_parts_mut(start, buf_size)
        })
    };
    let coeff_slice = unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, 16) };
    inv_txfm_add_dct_identity_4x4_8bpc_avx2_inner(
        dst_slice,
        base,
        dst_stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_dct_4x4_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let abs_stride = dst_stride.unsigned_abs();
    let buf_size = 3 * abs_stride + 4;
    let (base, dst_slice) = if dst_stride >= 0 {
        (0usize, unsafe {
            std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_size)
        })
    } else {
        let start = unsafe { (dst_ptr as *mut u8).offset(3 * dst_stride) };
        (3 * abs_stride, unsafe {
            std::slice::from_raw_parts_mut(start, buf_size)
        })
    };
    let coeff_slice = unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, 16) };
    inv_txfm_add_identity_dct_4x4_8bpc_avx2_inner(
        dst_slice,
        base,
        dst_stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// V/H ADST/DCT 8x8 TRANSFORMS
// ============================================================================

/// Identity transform 8x8
#[inline(always)]
fn identity8_1d_scalar(
    in0: i32,
    in1: i32,
    in2: i32,
    in3: i32,
    in4: i32,
    in5: i32,
    in6: i32,
    in7: i32,
    _min: i32,
    _max: i32,
) -> (i32, i32, i32, i32, i32, i32, i32, i32) {
    // For 8x8 identity: out = in * 2
    (
        in0 * 2,
        in1 * 2,
        in2 * 2,
        in3 * 2,
        in4 * 2,
        in5 * 2,
        in6 * 2,
        in7 * 2,
    )
}

// Use the macro to generate V/H transforms for 8x8
impl_8x8_transform!(
    inv_txfm_add_identity_adst_8x8_8bpc_avx2_inner,
    identity8_1d_scalar,
    adst8_1d_scalar
);
impl_8x8_transform!(
    inv_txfm_add_adst_identity_8x8_8bpc_avx2_inner,
    adst8_1d_scalar,
    identity8_1d_scalar
);
impl_8x8_transform!(
    inv_txfm_add_identity_flipadst_8x8_8bpc_avx2_inner,
    identity8_1d_scalar,
    flipadst8_1d_scalar
);
impl_8x8_transform!(
    inv_txfm_add_flipadst_identity_8x8_8bpc_avx2_inner,
    flipadst8_1d_scalar,
    identity8_1d_scalar
);
impl_8x8_transform!(
    inv_txfm_add_identity_dct_8x8_8bpc_avx2_inner,
    identity8_1d_scalar,
    dct8_1d_scalar
);
impl_8x8_transform!(
    inv_txfm_add_dct_identity_8x8_8bpc_avx2_inner,
    dct8_1d_scalar,
    identity8_1d_scalar
);

// FFI wrappers
impl_8x8_ffi_wrapper!(
    inv_txfm_add_identity_adst_8x8_8bpc_avx2,
    inv_txfm_add_identity_adst_8x8_8bpc_avx2_inner
);
impl_8x8_ffi_wrapper!(
    inv_txfm_add_adst_identity_8x8_8bpc_avx2,
    inv_txfm_add_adst_identity_8x8_8bpc_avx2_inner
);
impl_8x8_ffi_wrapper!(
    inv_txfm_add_identity_flipadst_8x8_8bpc_avx2,
    inv_txfm_add_identity_flipadst_8x8_8bpc_avx2_inner
);
impl_8x8_ffi_wrapper!(
    inv_txfm_add_flipadst_identity_8x8_8bpc_avx2,
    inv_txfm_add_flipadst_identity_8x8_8bpc_avx2_inner
);
impl_8x8_ffi_wrapper!(
    inv_txfm_add_identity_dct_8x8_8bpc_avx2,
    inv_txfm_add_identity_dct_8x8_8bpc_avx2_inner
);
impl_8x8_ffi_wrapper!(
    inv_txfm_add_dct_identity_8x8_8bpc_avx2,
    inv_txfm_add_dct_identity_8x8_8bpc_avx2_inner
);

// ============================================================================
// 32x32 DCT TRANSFORMS
// ============================================================================

/// DCT32 1D transform (in-place)
#[inline]
fn dct32_1d(c: &mut [i32], stride: usize, min: i32, max: i32) {
    let clip = |v: i32| v.clamp(min, max);

    // First apply DCT16 to even positions
    dct16_1d(c, stride * 2, min, max);

    let in1 = c[1 * stride];
    let in3 = c[3 * stride];
    let in5 = c[5 * stride];
    let in7 = c[7 * stride];
    let in9 = c[9 * stride];
    let in11 = c[11 * stride];
    let in13 = c[13 * stride];
    let in15 = c[15 * stride];
    let in17 = c[17 * stride];
    let in19 = c[19 * stride];
    let in21 = c[21 * stride];
    let in23 = c[23 * stride];
    let in25 = c[25 * stride];
    let in27 = c[27 * stride];
    let in29 = c[29 * stride];
    let in31 = c[31 * stride];

    let t16a = ((in1 * 201 - in31 * (4091 - 4096) + 2048) >> 12) - in31;
    let t17a = ((in17 * (3035 - 4096) - in15 * 2751 + 2048) >> 12) + in17;
    let t18a = ((in9 * 1751 - in23 * (3703 - 4096) + 2048) >> 12) - in23;
    let t19a = ((in25 * (3857 - 4096) - in7 * 1380 + 2048) >> 12) + in25;
    let t20a = ((in5 * 995 - in27 * (3973 - 4096) + 2048) >> 12) - in27;
    let t21a = ((in21 * (3513 - 4096) - in11 * 2106 + 2048) >> 12) + in21;
    let t22a = (in13 * 1220 - in19 * 1645 + 1024) >> 11;
    let t23a = ((in29 * (4052 - 4096) - in3 * 601 + 2048) >> 12) + in29;
    let t24a = ((in29 * 601 + in3 * (4052 - 4096) + 2048) >> 12) + in3;
    let t25a = (in13 * 1645 + in19 * 1220 + 1024) >> 11;
    let t26a = ((in21 * 2106 + in11 * (3513 - 4096) + 2048) >> 12) + in11;
    let t27a = ((in5 * (3973 - 4096) + in27 * 995 + 2048) >> 12) + in5;
    let t28a = ((in25 * 1380 + in7 * (3857 - 4096) + 2048) >> 12) + in7;
    let t29a = ((in9 * (3703 - 4096) + in23 * 1751 + 2048) >> 12) + in9;
    let t30a = ((in17 * 2751 + in15 * (3035 - 4096) + 2048) >> 12) + in15;
    let t31a = ((in1 * (4091 - 4096) + in31 * 201 + 2048) >> 12) + in1;

    let mut t16 = clip(t16a + t17a);
    let mut t17 = clip(t16a - t17a);
    let mut t18 = clip(t19a - t18a);
    let t19 = clip(t19a + t18a);
    let t20 = clip(t20a + t21a);
    let mut t21 = clip(t20a - t21a);
    let mut t22 = clip(t23a - t22a);
    let mut t23 = clip(t23a + t22a);
    let mut t24 = clip(t24a + t25a);
    let mut t25 = clip(t24a - t25a);
    let mut t26 = clip(t27a - t26a);
    let t27 = clip(t27a + t26a);
    let t28 = clip(t28a + t29a);
    let mut t29 = clip(t28a - t29a);
    let mut t30 = clip(t31a - t30a);
    let mut t31 = clip(t31a + t30a);

    let t17a = ((t30 * 799 - t17 * (4017 - 4096) + 2048) >> 12) - t17;
    let t30a = ((t30 * (4017 - 4096) + t17 * 799 + 2048) >> 12) + t30;
    let t18a = ((-(t29 * (4017 - 4096) + t18 * 799) + 2048) >> 12) - t29;
    let t29a = ((t29 * 799 - t18 * (4017 - 4096) + 2048) >> 12) - t18;
    let t21a = (t26 * 1703 - t21 * 1138 + 1024) >> 11;
    let t26a = (t26 * 1138 + t21 * 1703 + 1024) >> 11;
    let t22a = (-(t25 * 1138 + t22 * 1703) + 1024) >> 11;
    let t25a = (t25 * 1703 - t22 * 1138 + 1024) >> 11;

    let t16a = clip(t16 + t19);
    t17 = clip(t17a + t18a);
    t18 = clip(t17a - t18a);
    let t19a = clip(t16 - t19);
    let t20a = clip(t23 - t20);
    t21 = clip(t22a - t21a);
    t22 = clip(t22a + t21a);
    let t23a = clip(t23 + t20);
    let t24a = clip(t24 + t27);
    t25 = clip(t25a + t26a);
    t26 = clip(t25a - t26a);
    let t27a = clip(t24 - t27);
    let t28a = clip(t31 - t28);
    t29 = clip(t30a - t29a);
    t30 = clip(t30a + t29a);
    let t31a = clip(t31 + t28);

    let t18a = ((t29 * 1567 - t18 * (3784 - 4096) + 2048) >> 12) - t18;
    let t29a = ((t29 * (3784 - 4096) + t18 * 1567 + 2048) >> 12) + t29;
    let t19 = ((t28a * 1567 - t19a * (3784 - 4096) + 2048) >> 12) - t19a;
    let t28 = ((t28a * (3784 - 4096) + t19a * 1567 + 2048) >> 12) + t28a;
    let t20 = ((-(t27a * (3784 - 4096) + t20a * 1567) + 2048) >> 12) - t27a;
    let t27 = ((t27a * 1567 - t20a * (3784 - 4096) + 2048) >> 12) - t20a;
    let t21a = ((-(t26 * (3784 - 4096) + t21 * 1567) + 2048) >> 12) - t26;
    let t26a = ((t26 * 1567 - t21 * (3784 - 4096) + 2048) >> 12) - t21;

    t16 = clip(t16a + t23a);
    let t17a = clip(t17 + t22);
    t18 = clip(t18a + t21a);
    let t19a = clip(t19 + t20);
    let t20a = clip(t19 - t20);
    t21 = clip(t18a - t21a);
    let t22a = clip(t17 - t22);
    t23 = clip(t16a - t23a);
    t24 = clip(t31a - t24a);
    let t25a = clip(t30 - t25);
    t26 = clip(t29a - t26a);
    let t27a = clip(t28 - t27);
    let t28a = clip(t28 + t27);
    t29 = clip(t29a + t26a);
    let t30a = clip(t30 + t25);
    t31 = clip(t31a + t24a);

    let t20_final = ((t27a - t20a) * 181 + 128) >> 8;
    let t27_final = ((t27a + t20a) * 181 + 128) >> 8;
    let t21a_final = ((t26 - t21) * 181 + 128) >> 8;
    let t26a_final = ((t26 + t21) * 181 + 128) >> 8;
    let t22_final = ((t25a - t22a) * 181 + 128) >> 8;
    let t25_final = ((t25a + t22a) * 181 + 128) >> 8;
    let t23a = ((t24 - t23) * 181 + 128) >> 8;
    let t24a = ((t24 + t23) * 181 + 128) >> 8;

    let t0 = c[0 * stride];
    let t1 = c[2 * stride];
    let t2 = c[4 * stride];
    let t3 = c[6 * stride];
    let t4 = c[8 * stride];
    let t5 = c[10 * stride];
    let t6 = c[12 * stride];
    let t7 = c[14 * stride];
    let t8 = c[16 * stride];
    let t9 = c[18 * stride];
    let t10 = c[20 * stride];
    let t11 = c[22 * stride];
    let t12 = c[24 * stride];
    let t13 = c[26 * stride];
    let t14 = c[28 * stride];
    let t15 = c[30 * stride];

    c[0 * stride] = clip(t0 + t31);
    c[1 * stride] = clip(t1 + t30a);
    c[2 * stride] = clip(t2 + t29);
    c[3 * stride] = clip(t3 + t28a);
    c[4 * stride] = clip(t4 + t27_final);
    c[5 * stride] = clip(t5 + t26a_final);
    c[6 * stride] = clip(t6 + t25_final);
    c[7 * stride] = clip(t7 + t24a);
    c[8 * stride] = clip(t8 + t23a);
    c[9 * stride] = clip(t9 + t22_final);
    c[10 * stride] = clip(t10 + t21a_final);
    c[11 * stride] = clip(t11 + t20_final);
    c[12 * stride] = clip(t12 + t19a);
    c[13 * stride] = clip(t13 + t18);
    c[14 * stride] = clip(t14 + t17a);
    c[15 * stride] = clip(t15 + t16);
    c[16 * stride] = clip(t15 - t16);
    c[17 * stride] = clip(t14 - t17a);
    c[18 * stride] = clip(t13 - t18);
    c[19 * stride] = clip(t12 - t19a);
    c[20 * stride] = clip(t11 - t20_final);
    c[21 * stride] = clip(t10 - t21a_final);
    c[22 * stride] = clip(t9 - t22_final);
    c[23 * stride] = clip(t8 - t23a);
    c[24 * stride] = clip(t7 - t24a);
    c[25 * stride] = clip(t6 - t25_final);
    c[26 * stride] = clip(t5 - t26a_final);
    c[27 * stride] = clip(t4 - t27_final);
    c[28 * stride] = clip(t3 - t28a);
    c[29 * stride] = clip(t2 - t29);
    c[30 * stride] = clip(t1 - t30a);
    c[31 * stride] = clip(t0 - t31);
}

/// Identity32 1D transform (in-place)
#[inline]
fn identity32_1d(c: &mut [i32], stride: usize, _min: i32, _max: i32) {
    // For 32x32 identity: out = in * 4
    for i in 0..32 {
        c[i * stride] *= 4;
    }
}

/// Generic 32x32 transform function
#[inline]
fn inv_txfm_32x32_inner(
    tmp: &mut [i32; 1024],
    coeff: &[i16],
    row_transform: fn(&mut [i32], usize, i32, i32),
    col_transform: fn(&mut [i32], usize, i32, i32),
    row_clip_min: i32,
    row_clip_max: i32,
    col_clip_min: i32,
    col_clip_max: i32,
) {
    // For 32x32: row_shift = 2, col_shift = 4 (total 6)
    let rnd = 2;
    let shift = 2;

    // Row transform
    for y in 0..32 {
        // Load row from column-major
        let mut scratch = [0i32; 32];
        for x in 0..32 {
            scratch[x] = coeff[y + x * 32] as i32;
        }
        row_transform(&mut scratch[..32], 1, row_clip_min, row_clip_max);
        // Apply intermediate shift and store row-major
        for x in 0..32 {
            tmp[y * 32 + x] = ((scratch[x] + rnd) >> shift).clamp(col_clip_min, col_clip_max);
        }
    }

    // Column transform (in-place, row-major with stride 32)
    for x in 0..32 {
        col_transform(&mut tmp[x..], 32, col_clip_min, col_clip_max);
    }
}

/// Add transformed coefficients to destination with SIMD (32x32)
#[cfg(target_arch = "x86_64")]
#[arcane]
fn add_32x32_to_dst(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp: &[i32; 1024],
    coeff: &mut [i16],
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();

    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi16(bitdepth_max as i16);
    let rnd_final = _mm256_set1_epi32(8); // (+ 8) >> 4

    for y in 0..32 {
        let dst_off = y * dst_stride;

        // Process 32 pixels in two 16-pixel chunks
        for chunk in 0..2 {
            let x_base = chunk * 16;
            let dst_chunk_off = dst_off + x_base;

            // Load destination pixels (16 bytes)
            let d =
                loadu_128!(<&[u8; 16]>::try_from(&dst[dst_chunk_off..dst_chunk_off + 16]).unwrap());
            let d16 = _mm256_cvtepu8_epi16(d);

            // Load coefficients
            let c0 = _mm256_set_epi32(
                tmp[y * 32 + x_base + 7],
                tmp[y * 32 + x_base + 6],
                tmp[y * 32 + x_base + 5],
                tmp[y * 32 + x_base + 4],
                tmp[y * 32 + x_base + 3],
                tmp[y * 32 + x_base + 2],
                tmp[y * 32 + x_base + 1],
                tmp[y * 32 + x_base + 0],
            );
            let c1 = _mm256_set_epi32(
                tmp[y * 32 + x_base + 15],
                tmp[y * 32 + x_base + 14],
                tmp[y * 32 + x_base + 13],
                tmp[y * 32 + x_base + 12],
                tmp[y * 32 + x_base + 11],
                tmp[y * 32 + x_base + 10],
                tmp[y * 32 + x_base + 9],
                tmp[y * 32 + x_base + 8],
            );

            // Final scaling: (c + 8) >> 4
            let c0_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c0, rnd_final));
            let c1_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c1, rnd_final));

            let c16 = _mm256_packs_epi32(c0_scaled, c1_scaled);
            let c16 = _mm256_permute4x64_epi64::<0b11_01_10_00>(c16);

            let sum = _mm256_add_epi16(d16, c16);
            let clamped = _mm256_max_epi16(_mm256_min_epi16(sum, max_val), zero);

            let packed = _mm256_packus_epi16(clamped, clamped);
            let packed = _mm256_permute4x64_epi64::<0b11_01_10_00>(packed);

            storeu_128!(
                <&mut [u8; 16]>::try_from(&mut dst[dst_chunk_off..dst_chunk_off + 16]).unwrap(),
                _mm256_castsi256_si128(packed)
            );
        }
    }

    // Clear coefficients (1024 * 2 = 2048 bytes = 64 * 32 bytes)
    coeff[..1024].fill(0);
}

/// 32x32 DCT_DCT inner function
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_32x32_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let row_clip_min = i16::MIN as i32;
    let row_clip_max = i16::MAX as i32;
    let col_clip_min = i16::MIN as i32;
    let col_clip_max = i16::MAX as i32;

    let mut tmp = [0i32; 1024];
    inv_txfm_32x32_inner(
        &mut tmp,
        &*coeff,
        dct32_1d,
        dct32_1d,
        row_clip_min,
        row_clip_max,
        col_clip_min,
        col_clip_max,
    );
    add_32x32_to_dst(
        _token,
        &mut *dst,
        dst_stride,
        &tmp,
        &mut *coeff,
        bitdepth_max,
    );
}

/// 32x32 IDTX inner function
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_identity_identity_32x32_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let row_clip_min = i16::MIN as i32;
    let row_clip_max = i16::MAX as i32;
    let col_clip_min = i16::MIN as i32;
    let col_clip_max = i16::MAX as i32;

    let mut tmp = [0i32; 1024];
    inv_txfm_32x32_inner(
        &mut tmp,
        &*coeff,
        identity32_1d,
        identity32_1d,
        row_clip_min,
        row_clip_max,
        col_clip_min,
        col_clip_max,
    );
    add_32x32_to_dst(
        _token,
        &mut *dst,
        dst_stride,
        &tmp,
        &mut *coeff,
        bitdepth_max,
    );
}

/// FFI wrapper for 32x32 DCT_DCT 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_32x32_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, _coeff_len as usize * stride + stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_32x32_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// FFI wrapper for 32x32 IDTX 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_identity_32x32_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, _coeff_len as usize * stride + stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_identity_identity_32x32_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// 32x32 DCT TRANSFORMS 16bpc
// ============================================================================

/// Add transformed coefficients to destination with SIMD (32x32 16bpc)
#[cfg(target_arch = "x86_64")]
#[arcane]
fn add_32x32_to_dst_16bpc(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize, // stride in bytes
    tmp: &[i32; 1024],
    coeff: &mut [i16],
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let stride_u16 = dst_stride / 2;

    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi32(bitdepth_max);
    let rnd_final = _mm256_set1_epi32(8); // (+ 8) >> 4

    for y in 0..32 {
        let dst_off = y * stride_u16;

        // Process 32 pixels in four 8-pixel chunks (since we work with i32)
        for chunk in 0..4 {
            let x_base = chunk * 8;
            let dst_chunk_off = dst_off + x_base;

            // Load destination pixels (8 u16 = 16 bytes)
            let d =
                loadu_128!(<&[u16; 8]>::try_from(&dst[dst_chunk_off..dst_chunk_off + 8]).unwrap());
            let d_lo = _mm_unpacklo_epi16(d, _mm_setzero_si128());
            let d_hi = _mm_unpackhi_epi16(d, _mm_setzero_si128());

            // Load coefficients
            let c_lo = _mm_set_epi32(
                tmp[y * 32 + x_base + 3],
                tmp[y * 32 + x_base + 2],
                tmp[y * 32 + x_base + 1],
                tmp[y * 32 + x_base + 0],
            );
            let c_hi = _mm_set_epi32(
                tmp[y * 32 + x_base + 7],
                tmp[y * 32 + x_base + 6],
                tmp[y * 32 + x_base + 5],
                tmp[y * 32 + x_base + 4],
            );

            // Combine to 256-bit for faster processing
            let d32 = _mm256_set_m128i(d_hi, d_lo);
            let c32 = _mm256_set_m128i(c_hi, c_lo);

            // Final scaling: (c + 8) >> 4
            let c_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c32, rnd_final));

            // Add to destination
            let sum = _mm256_add_epi32(d32, c_scaled);

            // Clamp to [0, bitdepth_max]
            let clamped = _mm256_max_epi32(_mm256_min_epi32(sum, max_val), zero);

            // Pack to u16 and store
            let lo = _mm256_castsi256_si128(clamped);
            let hi = _mm256_extracti128_si256(clamped, 1);
            let packed = _mm_packus_epi32(lo, hi);
            storeu_128!(
                <&mut [u16; 8]>::try_from(&mut dst[dst_chunk_off..dst_chunk_off + 8]).unwrap(),
                packed
            );
        }
    }

    // Clear coefficients (1024 * 2 = 2048 bytes = 64 * 32 bytes)
    coeff[..1024].fill(0);
}

/// 32x32 DCT_DCT inner function for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_32x32_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    // For 16bpc: use full i32 range
    let row_clip_min = i32::MIN;
    let row_clip_max = i32::MAX;
    let col_clip_min = i32::MIN;
    let col_clip_max = i32::MAX;

    let mut tmp = [0i32; 1024];
    inv_txfm_32x32_inner(
        &mut tmp,
        &*coeff,
        dct32_1d,
        dct32_1d,
        row_clip_min,
        row_clip_max,
        col_clip_min,
        col_clip_max,
    );
    add_32x32_to_dst_16bpc(
        _token,
        &mut *dst,
        dst_stride,
        &tmp,
        &mut *coeff,
        bitdepth_max,
    );
}

/// FFI wrapper for 32x32 DCT_DCT 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_32x32_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_32x32_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// 64x64 DCT TRANSFORMS
// ============================================================================

/// DCT32 1D transform for tx64 mode (simplified coefficients for in17-in31)
#[inline]
fn dct32_1d_tx64(c: &mut [i32], stride: usize, min: i32, max: i32) {
    let clip = |v: i32| v.clamp(min, max);

    // First apply DCT16 with tx64=1 (simplified)
    dct16_1d_tx64(c, stride * 2, min, max);

    let in1 = c[1 * stride];
    let in3 = c[3 * stride];
    let in5 = c[5 * stride];
    let in7 = c[7 * stride];
    let in9 = c[9 * stride];
    let in11 = c[11 * stride];
    let in13 = c[13 * stride];
    let in15 = c[15 * stride];

    // tx64=1: simplified single-coefficient multiplications
    let t16a = (in1 * 201 + 2048) >> 12;
    let t17a = (in15 * -2751 + 2048) >> 12;
    let t18a = (in9 * 1751 + 2048) >> 12;
    let t19a = (in7 * -1380 + 2048) >> 12;
    let t20a = (in5 * 995 + 2048) >> 12;
    let t21a = (in11 * -2106 + 2048) >> 12;
    let t22a = (in13 * 2440 + 2048) >> 12;
    let t23a = (in3 * -601 + 2048) >> 12;
    let t24a = (in3 * 4052 + 2048) >> 12;
    let t25a = (in13 * 3290 + 2048) >> 12;
    let t26a = (in11 * 3513 + 2048) >> 12;
    let t27a = (in5 * 3973 + 2048) >> 12;
    let t28a = (in7 * 3857 + 2048) >> 12;
    let t29a = (in9 * 3703 + 2048) >> 12;
    let t30a = (in15 * 3035 + 2048) >> 12;
    let t31a = (in1 * 4091 + 2048) >> 12;

    let mut t16 = clip(t16a + t17a);
    let mut t17 = clip(t16a - t17a);
    let mut t18 = clip(t19a - t18a);
    let t19 = clip(t19a + t18a);
    let t20 = clip(t20a + t21a);
    let mut t21 = clip(t20a - t21a);
    let mut t22 = clip(t23a - t22a);
    let mut t23 = clip(t23a + t22a);
    let mut t24 = clip(t24a + t25a);
    let mut t25 = clip(t24a - t25a);
    let mut t26 = clip(t27a - t26a);
    let t27 = clip(t27a + t26a);
    let t28 = clip(t28a + t29a);
    let mut t29 = clip(t28a - t29a);
    let mut t30 = clip(t31a - t30a);
    let mut t31 = clip(t31a + t30a);

    let t17a = ((t30 * 799 - t17 * (4017 - 4096) + 2048) >> 12) - t17;
    let t30a = ((t30 * (4017 - 4096) + t17 * 799 + 2048) >> 12) + t30;
    let t18a = ((-(t29 * (4017 - 4096) + t18 * 799) + 2048) >> 12) - t29;
    let t29a = ((t29 * 799 - t18 * (4017 - 4096) + 2048) >> 12) - t18;
    let t21a = (t26 * 1703 - t21 * 1138 + 1024) >> 11;
    let t26a = (t26 * 1138 + t21 * 1703 + 1024) >> 11;
    let t22a = (-(t25 * 1138 + t22 * 1703) + 1024) >> 11;
    let t25a = (t25 * 1703 - t22 * 1138 + 1024) >> 11;

    let t16a = clip(t16 + t19);
    t17 = clip(t17a + t18a);
    t18 = clip(t17a - t18a);
    let t19a = clip(t16 - t19);
    let t20a = clip(t23 - t20);
    t21 = clip(t22a - t21a);
    t22 = clip(t22a + t21a);
    let t23a = clip(t23 + t20);
    let t24a = clip(t24 + t27);
    t25 = clip(t25a + t26a);
    t26 = clip(t25a - t26a);
    let t27a = clip(t24 - t27);
    let t28a = clip(t31 - t28);
    t29 = clip(t30a - t29a);
    t30 = clip(t30a + t29a);
    let t31a = clip(t31 + t28);

    let t18a = ((t29 * 1567 - t18 * (3784 - 4096) + 2048) >> 12) - t18;
    let t29a = ((t29 * (3784 - 4096) + t18 * 1567 + 2048) >> 12) + t29;
    let t19 = ((t28a * 1567 - t19a * (3784 - 4096) + 2048) >> 12) - t19a;
    let t28 = ((t28a * (3784 - 4096) + t19a * 1567 + 2048) >> 12) + t28a;
    let t20 = ((-(t27a * (3784 - 4096) + t20a * 1567) + 2048) >> 12) - t27a;
    let t27 = ((t27a * 1567 - t20a * (3784 - 4096) + 2048) >> 12) - t20a;
    let t21a = ((-(t26 * (3784 - 4096) + t21 * 1567) + 2048) >> 12) - t26;
    let t26a = ((t26 * 1567 - t21 * (3784 - 4096) + 2048) >> 12) - t21;

    t16 = clip(t16a + t23a);
    let t17a = clip(t17 + t22);
    t18 = clip(t18a + t21a);
    let t19a = clip(t19 + t20);
    let t20a = clip(t19 - t20);
    t21 = clip(t18a - t21a);
    let t22a = clip(t17 - t22);
    t23 = clip(t16a - t23a);
    t24 = clip(t31a - t24a);
    let t25a = clip(t30 - t25);
    t26 = clip(t29a - t26a);
    let t27a = clip(t28 - t27);
    let t28a = clip(t28 + t27);
    t29 = clip(t29a + t26a);
    let t30a = clip(t30 + t25);
    t31 = clip(t31a + t24a);

    let t20_final = ((t27a - t20a) * 181 + 128) >> 8;
    let t27_final = ((t27a + t20a) * 181 + 128) >> 8;
    let t21a_final = ((t26 - t21) * 181 + 128) >> 8;
    let t26a_final = ((t26 + t21) * 181 + 128) >> 8;
    let t22_final = ((t25a - t22a) * 181 + 128) >> 8;
    let t25_final = ((t25a + t22a) * 181 + 128) >> 8;
    let t23a = ((t24 - t23) * 181 + 128) >> 8;
    let t24a = ((t24 + t23) * 181 + 128) >> 8;

    let t0 = c[0 * stride];
    let t1 = c[2 * stride];
    let t2 = c[4 * stride];
    let t3 = c[6 * stride];
    let t4 = c[8 * stride];
    let t5 = c[10 * stride];
    let t6 = c[12 * stride];
    let t7 = c[14 * stride];
    let t8 = c[16 * stride];
    let t9 = c[18 * stride];
    let t10 = c[20 * stride];
    let t11 = c[22 * stride];
    let t12 = c[24 * stride];
    let t13 = c[26 * stride];
    let t14 = c[28 * stride];
    let t15 = c[30 * stride];

    c[0 * stride] = clip(t0 + t31);
    c[1 * stride] = clip(t1 + t30a);
    c[2 * stride] = clip(t2 + t29);
    c[3 * stride] = clip(t3 + t28a);
    c[4 * stride] = clip(t4 + t27_final);
    c[5 * stride] = clip(t5 + t26a_final);
    c[6 * stride] = clip(t6 + t25_final);
    c[7 * stride] = clip(t7 + t24a);
    c[8 * stride] = clip(t8 + t23a);
    c[9 * stride] = clip(t9 + t22_final);
    c[10 * stride] = clip(t10 + t21a_final);
    c[11 * stride] = clip(t11 + t20_final);
    c[12 * stride] = clip(t12 + t19a);
    c[13 * stride] = clip(t13 + t18);
    c[14 * stride] = clip(t14 + t17a);
    c[15 * stride] = clip(t15 + t16);
    c[16 * stride] = clip(t15 - t16);
    c[17 * stride] = clip(t14 - t17a);
    c[18 * stride] = clip(t13 - t18);
    c[19 * stride] = clip(t12 - t19a);
    c[20 * stride] = clip(t11 - t20_final);
    c[21 * stride] = clip(t10 - t21a_final);
    c[22 * stride] = clip(t9 - t22_final);
    c[23 * stride] = clip(t8 - t23a);
    c[24 * stride] = clip(t7 - t24a);
    c[25 * stride] = clip(t6 - t25_final);
    c[26 * stride] = clip(t5 - t26a_final);
    c[27 * stride] = clip(t4 - t27_final);
    c[28 * stride] = clip(t3 - t28a);
    c[29 * stride] = clip(t2 - t29);
    c[30 * stride] = clip(t1 - t30a);
    c[31 * stride] = clip(t0 - t31);
}

/// DCT16 1D transform for tx64 mode (simplified coefficients)
#[inline]
fn dct16_1d_tx64(c: &mut [i32], stride: usize, min: i32, max: i32) {
    let clip = |v: i32| v.clamp(min, max);

    // First apply DCT8 to even positions
    dct8_1d(c, stride * 2, min, max);

    let in1 = c[1 * stride];
    let in3 = c[3 * stride];
    let in5 = c[5 * stride];
    let in7 = c[7 * stride];

    // tx64=1: simplified single-coefficient multiplications
    let t8a = (in1 * 401 + 2048) >> 12;
    let t9a = (in7 * -2598 + 2048) >> 12;
    let t10a = (in5 * 1931 + 2048) >> 12;
    let t11a = (in3 * -1189 + 2048) >> 12;
    let t12a = (in3 * 3920 + 2048) >> 12;
    let t13a = (in5 * 3612 + 2048) >> 12;
    let t14a = (in7 * 3166 + 2048) >> 12;
    let t15a = (in1 * 4076 + 2048) >> 12;

    let t8 = clip(t8a + t9a);
    let mut t9 = clip(t8a - t9a);
    let mut t10 = clip(t11a - t10a);
    let mut t11 = clip(t11a + t10a);
    let mut t12 = clip(t12a + t13a);
    let mut t13 = clip(t12a - t13a);
    let mut t14 = clip(t15a - t14a);
    let t15 = clip(t15a + t14a);

    let t9a = ((t14 * 1567 - t9 * (3784 - 4096) + 2048) >> 12) - t9;
    let t14a = ((t14 * (3784 - 4096) + t9 * 1567 + 2048) >> 12) + t14;
    let t10a = ((-(t13 * (3784 - 4096) + t10 * 1567) + 2048) >> 12) - t13;
    let t13a = ((t13 * 1567 - t10 * (3784 - 4096) + 2048) >> 12) - t10;

    let t8a = clip(t8 + t11);
    t9 = clip(t9a + t10a);
    t10 = clip(t9a - t10a);
    let t11a = clip(t8 - t11);
    let t12a = clip(t15 - t12);
    t13 = clip(t14a - t13a);
    t14 = clip(t14a + t13a);
    let t15a = clip(t15 + t12);

    let t10a = ((t13 - t10) * 181 + 128) >> 8;
    let t13a = ((t13 + t10) * 181 + 128) >> 8;
    t11 = ((t12a - t11a) * 181 + 128) >> 8;
    t12 = ((t12a + t11a) * 181 + 128) >> 8;

    let t0 = c[0 * stride];
    let t1 = c[2 * stride];
    let t2 = c[4 * stride];
    let t3 = c[6 * stride];
    let t4 = c[8 * stride];
    let t5 = c[10 * stride];
    let t6 = c[12 * stride];
    let t7 = c[14 * stride];

    c[0 * stride] = clip(t0 + t15a);
    c[1 * stride] = clip(t1 + t14);
    c[2 * stride] = clip(t2 + t13a);
    c[3 * stride] = clip(t3 + t12);
    c[4 * stride] = clip(t4 + t11);
    c[5 * stride] = clip(t5 + t10a);
    c[6 * stride] = clip(t6 + t9);
    c[7 * stride] = clip(t7 + t8a);
    c[8 * stride] = clip(t7 - t8a);
    c[9 * stride] = clip(t6 - t9);
    c[10 * stride] = clip(t5 - t10a);
    c[11 * stride] = clip(t4 - t11);
    c[12 * stride] = clip(t3 - t12);
    c[13 * stride] = clip(t2 - t13a);
    c[14 * stride] = clip(t1 - t14);
    c[15 * stride] = clip(t0 - t15a);
}

/// DCT64 1D transform (in-place)
#[inline]
fn dct64_1d(c: &mut [i32], stride: usize, min: i32, max: i32) {
    let clip = |v: i32| v.clamp(min, max);

    // First apply DCT32 in tx64 mode to even positions
    dct32_1d_tx64(c, stride * 2, min, max);

    let in1 = c[1 * stride];
    let in3 = c[3 * stride];
    let in5 = c[5 * stride];
    let in7 = c[7 * stride];
    let in9 = c[9 * stride];
    let in11 = c[11 * stride];
    let in13 = c[13 * stride];
    let in15 = c[15 * stride];
    let in17 = c[17 * stride];
    let in19 = c[19 * stride];
    let in21 = c[21 * stride];
    let in23 = c[23 * stride];
    let in25 = c[25 * stride];
    let in27 = c[27 * stride];
    let in29 = c[29 * stride];
    let in31 = c[31 * stride];

    // tx64 simplified coefficients - only use first 32 inputs
    let mut t32a = (in1 * 101 + 2048) >> 12;
    let mut t33a = (in31 * -2824 + 2048) >> 12;
    let mut t34a = (in17 * 1660 + 2048) >> 12;
    let mut t35a = (in15 * -1474 + 2048) >> 12;
    let mut t36a = (in9 * 897 + 2048) >> 12;
    let mut t37a = (in23 * -2191 + 2048) >> 12;
    let mut t38a = (in25 * 2359 + 2048) >> 12;
    let mut t39a = (in7 * -700 + 2048) >> 12;
    let mut t40a = (in5 * 501 + 2048) >> 12;
    let mut t41a = (in27 * -2520 + 2048) >> 12;
    let mut t42a = (in21 * 2019 + 2048) >> 12;
    let mut t43a = (in11 * -1092 + 2048) >> 12;
    let mut t44a = (in13 * 1285 + 2048) >> 12;
    let mut t45a = (in19 * -1842 + 2048) >> 12;
    let mut t46a = (in29 * 2675 + 2048) >> 12;
    let mut t47a = (in3 * -301 + 2048) >> 12;
    let mut t48a = (in3 * 4085 + 2048) >> 12;
    let mut t49a = (in29 * 3102 + 2048) >> 12;
    let mut t50a = (in19 * 3659 + 2048) >> 12;
    let mut t51a = (in13 * 3889 + 2048) >> 12;
    let mut t52a = (in11 * 3948 + 2048) >> 12;
    let mut t53a = (in21 * 3564 + 2048) >> 12;
    let mut t54a = (in27 * 3229 + 2048) >> 12;
    let mut t55a = (in5 * 4065 + 2048) >> 12;
    let mut t56a = (in7 * 4036 + 2048) >> 12;
    let mut t57a = (in25 * 3349 + 2048) >> 12;
    let mut t58a = (in23 * 3461 + 2048) >> 12;
    let mut t59a = (in9 * 3996 + 2048) >> 12;
    let mut t60a = (in15 * 3822 + 2048) >> 12;
    let mut t61a = (in17 * 3745 + 2048) >> 12;
    let mut t62a = (in31 * 2967 + 2048) >> 12;
    let mut t63a = (in1 * 4095 + 2048) >> 12;

    let mut t32 = clip(t32a + t33a);
    let mut t33 = clip(t32a - t33a);
    let mut t34 = clip(t35a - t34a);
    let mut t35 = clip(t35a + t34a);
    let mut t36 = clip(t36a + t37a);
    let mut t37 = clip(t36a - t37a);
    let mut t38 = clip(t39a - t38a);
    let mut t39 = clip(t39a + t38a);
    let mut t40 = clip(t40a + t41a);
    let mut t41 = clip(t40a - t41a);
    let mut t42 = clip(t43a - t42a);
    let mut t43 = clip(t43a + t42a);
    let mut t44 = clip(t44a + t45a);
    let mut t45 = clip(t44a - t45a);
    let mut t46 = clip(t47a - t46a);
    let mut t47 = clip(t47a + t46a);
    let mut t48 = clip(t48a + t49a);
    let mut t49 = clip(t48a - t49a);
    let mut t50 = clip(t51a - t50a);
    let mut t51 = clip(t51a + t50a);
    let mut t52 = clip(t52a + t53a);
    let mut t53 = clip(t52a - t53a);
    let mut t54 = clip(t55a - t54a);
    let mut t55 = clip(t55a + t54a);
    let mut t56 = clip(t56a + t57a);
    let mut t57 = clip(t56a - t57a);
    let mut t58 = clip(t59a - t58a);
    let mut t59 = clip(t59a + t58a);
    let mut t60 = clip(t60a + t61a);
    let mut t61 = clip(t60a - t61a);
    let mut t62 = clip(t63a - t62a);
    let mut t63 = clip(t63a + t62a);

    t33a = ((t33 * (4096 - 4076) + t62 * 401 + 2048) >> 12) - t33;
    t34a = ((t34 * -401 + t61 * (4096 - 4076) + 2048) >> 12) - t61;
    t37a = (t37 * -1299 + t58 * 1583 + 1024) >> 11;
    t38a = (t38 * -1583 + t57 * -1299 + 1024) >> 11;
    t41a = ((t41 * (4096 - 3612) + t54 * 1931 + 2048) >> 12) - t41;
    t42a = ((t42 * -1931 + t53 * (4096 - 3612) + 2048) >> 12) - t53;
    t45a = ((t45 * -1189 + t50 * (3920 - 4096) + 2048) >> 12) + t50;
    t46a = ((t46 * (4096 - 3920) + t49 * -1189 + 2048) >> 12) - t46;
    t49a = ((t46 * -1189 + t49 * (3920 - 4096) + 2048) >> 12) + t49;
    t50a = ((t45 * (3920 - 4096) + t50 * 1189 + 2048) >> 12) + t45;
    t53a = ((t42 * (4096 - 3612) + t53 * 1931 + 2048) >> 12) - t42;
    t54a = ((t41 * 1931 + t54 * (3612 - 4096) + 2048) >> 12) + t54;
    t57a = (t38 * -1299 + t57 * 1583 + 1024) >> 11;
    t58a = (t37 * 1583 + t58 * 1299 + 1024) >> 11;
    t61a = ((t34 * (4096 - 4076) + t61 * 401 + 2048) >> 12) - t34;
    t62a = ((t33 * 401 + t62 * (4076 - 4096) + 2048) >> 12) + t62;

    t32a = clip(t32 + t35);
    t33 = clip(t33a + t34a);
    t34 = clip(t33a - t34a);
    t35a = clip(t32 - t35);
    t36a = clip(t39 - t36);
    t37 = clip(t38a - t37a);
    t38 = clip(t38a + t37a);
    t39a = clip(t39 + t36);
    t40a = clip(t40 + t43);
    t41 = clip(t41a + t42a);
    t42 = clip(t41a - t42a);
    t43a = clip(t40 - t43);
    t44a = clip(t47 - t44);
    t45 = clip(t46a - t45a);
    t46 = clip(t46a + t45a);
    t47a = clip(t47 + t44);
    t48a = clip(t48 + t51);
    t49 = clip(t49a + t50a);
    t50 = clip(t49a - t50a);
    t51a = clip(t48 - t51);
    t52a = clip(t55 - t52);
    t53 = clip(t54a - t53a);
    t54 = clip(t54a + t53a);
    t55a = clip(t55 + t52);
    t56a = clip(t56 + t59);
    t57 = clip(t57a + t58a);
    t58 = clip(t57a - t58a);
    t59a = clip(t56 - t59);
    t60a = clip(t63 - t60);
    t61 = clip(t62a - t61a);
    t62 = clip(t62a + t61a);
    t63a = clip(t63 + t60);

    t34a = ((t34 * (4096 - 4017) + t61 * 799 + 2048) >> 12) - t34;
    t35 = ((t35a * (4096 - 4017) + t60a * 799 + 2048) >> 12) - t35a;
    t36 = ((t36a * -799 + t59a * (4096 - 4017) + 2048) >> 12) - t59a;
    t37a = ((t37 * -799 + t58 * (4096 - 4017) + 2048) >> 12) - t58;
    t42a = (t42 * -1138 + t53 * 1703 + 1024) >> 11;
    t43 = (t43a * -1138 + t52a * 1703 + 1024) >> 11;
    t44 = (t44a * -1703 + t51a * -1138 + 1024) >> 11;
    t45a = (t45 * -1703 + t50 * -1138 + 1024) >> 11;
    t50a = (t45 * -1138 + t50 * 1703 + 1024) >> 11;
    t51 = (t44a * -1138 + t51a * 1703 + 1024) >> 11;
    t52 = (t43a * 1703 + t52a * 1138 + 1024) >> 11;
    t53a = (t42 * 1703 + t53 * 1138 + 1024) >> 11;
    t58a = ((t37 * (4096 - 4017) + t58 * 799 + 2048) >> 12) - t37;
    t59 = ((t36a * (4096 - 4017) + t59a * 799 + 2048) >> 12) - t36a;
    t60 = ((t35a * 799 + t60a * (4017 - 4096) + 2048) >> 12) + t60a;
    t61a = ((t34 * 799 + t61 * (4017 - 4096) + 2048) >> 12) + t61;

    t32 = clip(t32a + t39a);
    t33a = clip(t33 + t38);
    t34 = clip(t34a + t37a);
    t35a = clip(t35 + t36);
    t36a = clip(t35 - t36);
    t37 = clip(t34a - t37a);
    t38a = clip(t33 - t38);
    t39 = clip(t32a - t39a);
    t40 = clip(t47a - t40a);
    t41a = clip(t46 - t41);
    t42 = clip(t45a - t42a);
    t43a = clip(t44 - t43);
    t44a = clip(t44 + t43);
    t45 = clip(t45a + t42a);
    t46a = clip(t46 + t41);
    t47 = clip(t47a + t40a);
    t48 = clip(t48a + t55a);
    t49a = clip(t49 + t54);
    t50 = clip(t50a + t53a);
    t51a = clip(t51 + t52);
    t52a = clip(t51 - t52);
    t53 = clip(t50a - t53a);
    t54a = clip(t49 - t54);
    t55 = clip(t48a - t55a);
    t56 = clip(t63a - t56a);
    t57a = clip(t62 - t57);
    t58 = clip(t61a - t58a);
    t59a = clip(t60 - t59);
    t60a = clip(t60 + t59);
    t61 = clip(t61a + t58a);
    t62a = clip(t62 + t57);
    t63 = clip(t63a + t56a);

    t36 = ((t36a * (4096 - 3784) + t59a * 1567 + 2048) >> 12) - t36a;
    t37a = ((t37 * (4096 - 3784) + t58 * 1567 + 2048) >> 12) - t37;
    t38 = ((t38a * (4096 - 3784) + t57a * 1567 + 2048) >> 12) - t38a;
    t39a = ((t39 * (4096 - 3784) + t56 * 1567 + 2048) >> 12) - t39;
    t40a = ((t40 * -1567 + t55 * (4096 - 3784) + 2048) >> 12) - t55;
    t41 = ((t41a * -1567 + t54a * (4096 - 3784) + 2048) >> 12) - t54a;
    t42a = ((t42 * -1567 + t53 * (4096 - 3784) + 2048) >> 12) - t53;
    t43 = ((t43a * -1567 + t52a * (4096 - 3784) + 2048) >> 12) - t52a;
    t52 = ((t43a * (4096 - 3784) + t52a * 1567 + 2048) >> 12) - t43a;
    t53a = ((t42 * (4096 - 3784) + t53 * 1567 + 2048) >> 12) - t42;
    t54 = ((t41a * (4096 - 3784) + t54a * 1567 + 2048) >> 12) - t41a;
    t55a = ((t40 * (4096 - 3784) + t55 * 1567 + 2048) >> 12) - t40;
    t56a = ((t39 * 1567 + t56 * (3784 - 4096) + 2048) >> 12) + t56;
    t57 = ((t38a * 1567 + t57a * (3784 - 4096) + 2048) >> 12) + t57a;
    t58a = ((t37 * 1567 + t58 * (3784 - 4096) + 2048) >> 12) + t58;
    t59 = ((t36a * 1567 + t59a * (3784 - 4096) + 2048) >> 12) + t59a;

    t32a = clip(t32 + t47);
    t33 = clip(t33a + t46a);
    t34a = clip(t34 + t45);
    t35 = clip(t35a + t44a);
    t36a = clip(t36 + t43);
    t37 = clip(t37a + t42a);
    t38a = clip(t38 + t41);
    t39 = clip(t39a + t40a);
    t40 = clip(t39a - t40a);
    t41a = clip(t38 - t41);
    t42 = clip(t37a - t42a);
    t43a = clip(t36 - t43);
    t44 = clip(t35a - t44a);
    t45a = clip(t34 - t45);
    t46 = clip(t33a - t46a);
    t47a = clip(t32 - t47);
    t48a = clip(t63 - t48);
    t49 = clip(t62a - t49a);
    t50a = clip(t61 - t50);
    t51 = clip(t60a - t51a);
    t52a = clip(t59 - t52);
    t53 = clip(t58a - t53a);
    t54a = clip(t57 - t54);
    t55 = clip(t56a - t55a);
    t56 = clip(t56a + t55a);
    t57a = clip(t57 + t54);
    t58 = clip(t58a + t53a);
    t59a = clip(t59 + t52);
    t60 = clip(t60a + t51a);
    t61a = clip(t61 + t50);
    t62 = clip(t62a + t49a);
    t63a = clip(t63 + t48);

    t40a = ((t55 - t40) * 181 + 128) >> 8;
    t41 = ((t54a - t41a) * 181 + 128) >> 8;
    t42a = ((t53 - t42) * 181 + 128) >> 8;
    t43 = ((t52a - t43a) * 181 + 128) >> 8;
    t44a = ((t51 - t44) * 181 + 128) >> 8;
    t45 = ((t50a - t45a) * 181 + 128) >> 8;
    t46a = ((t49 - t46) * 181 + 128) >> 8;
    t47 = ((t48a - t47a) * 181 + 128) >> 8;
    t48 = ((t47a + t48a) * 181 + 128) >> 8;
    t49a = ((t46 + t49) * 181 + 128) >> 8;
    t50 = ((t45a + t50a) * 181 + 128) >> 8;
    t51a = ((t44 + t51) * 181 + 128) >> 8;
    t52 = ((t43a + t52a) * 181 + 128) >> 8;
    t53a = ((t42 + t53) * 181 + 128) >> 8;
    t54 = ((t41a + t54a) * 181 + 128) >> 8;
    t55a = ((t40 + t55) * 181 + 128) >> 8;

    let t0 = c[0 * stride];
    let t1 = c[2 * stride];
    let t2 = c[4 * stride];
    let t3 = c[6 * stride];
    let t4 = c[8 * stride];
    let t5 = c[10 * stride];
    let t6 = c[12 * stride];
    let t7 = c[14 * stride];
    let t8 = c[16 * stride];
    let t9 = c[18 * stride];
    let t10 = c[20 * stride];
    let t11 = c[22 * stride];
    let t12 = c[24 * stride];
    let t13 = c[26 * stride];
    let t14 = c[28 * stride];
    let t15 = c[30 * stride];
    let t16 = c[32 * stride];
    let t17 = c[34 * stride];
    let t18 = c[36 * stride];
    let t19 = c[38 * stride];
    let t20 = c[40 * stride];
    let t21 = c[42 * stride];
    let t22 = c[44 * stride];
    let t23 = c[46 * stride];
    let t24 = c[48 * stride];
    let t25 = c[50 * stride];
    let t26 = c[52 * stride];
    let t27 = c[54 * stride];
    let t28 = c[56 * stride];
    let t29 = c[58 * stride];
    let t30 = c[60 * stride];
    let t31 = c[62 * stride];

    c[0 * stride] = clip(t0 + t63a);
    c[1 * stride] = clip(t1 + t62);
    c[2 * stride] = clip(t2 + t61a);
    c[3 * stride] = clip(t3 + t60);
    c[4 * stride] = clip(t4 + t59a);
    c[5 * stride] = clip(t5 + t58);
    c[6 * stride] = clip(t6 + t57a);
    c[7 * stride] = clip(t7 + t56);
    c[8 * stride] = clip(t8 + t55a);
    c[9 * stride] = clip(t9 + t54);
    c[10 * stride] = clip(t10 + t53a);
    c[11 * stride] = clip(t11 + t52);
    c[12 * stride] = clip(t12 + t51a);
    c[13 * stride] = clip(t13 + t50);
    c[14 * stride] = clip(t14 + t49a);
    c[15 * stride] = clip(t15 + t48);
    c[16 * stride] = clip(t16 + t47);
    c[17 * stride] = clip(t17 + t46a);
    c[18 * stride] = clip(t18 + t45);
    c[19 * stride] = clip(t19 + t44a);
    c[20 * stride] = clip(t20 + t43);
    c[21 * stride] = clip(t21 + t42a);
    c[22 * stride] = clip(t22 + t41);
    c[23 * stride] = clip(t23 + t40a);
    c[24 * stride] = clip(t24 + t39);
    c[25 * stride] = clip(t25 + t38a);
    c[26 * stride] = clip(t26 + t37);
    c[27 * stride] = clip(t27 + t36a);
    c[28 * stride] = clip(t28 + t35);
    c[29 * stride] = clip(t29 + t34a);
    c[30 * stride] = clip(t30 + t33);
    c[31 * stride] = clip(t31 + t32a);
    c[32 * stride] = clip(t31 - t32a);
    c[33 * stride] = clip(t30 - t33);
    c[34 * stride] = clip(t29 - t34a);
    c[35 * stride] = clip(t28 - t35);
    c[36 * stride] = clip(t27 - t36a);
    c[37 * stride] = clip(t26 - t37);
    c[38 * stride] = clip(t25 - t38a);
    c[39 * stride] = clip(t24 - t39);
    c[40 * stride] = clip(t23 - t40a);
    c[41 * stride] = clip(t22 - t41);
    c[42 * stride] = clip(t21 - t42a);
    c[43 * stride] = clip(t20 - t43);
    c[44 * stride] = clip(t19 - t44a);
    c[45 * stride] = clip(t18 - t45);
    c[46 * stride] = clip(t17 - t46a);
    c[47 * stride] = clip(t16 - t47);
    c[48 * stride] = clip(t15 - t48);
    c[49 * stride] = clip(t14 - t49a);
    c[50 * stride] = clip(t13 - t50);
    c[51 * stride] = clip(t12 - t51a);
    c[52 * stride] = clip(t11 - t52);
    c[53 * stride] = clip(t10 - t53a);
    c[54 * stride] = clip(t9 - t54);
    c[55 * stride] = clip(t8 - t55a);
    c[56 * stride] = clip(t7 - t56);
    c[57 * stride] = clip(t6 - t57a);
    c[58 * stride] = clip(t5 - t58);
    c[59 * stride] = clip(t4 - t59a);
    c[60 * stride] = clip(t3 - t60);
    c[61 * stride] = clip(t2 - t61a);
    c[62 * stride] = clip(t1 - t62);
    c[63 * stride] = clip(t0 - t63a);
}

/// Identity64 1D transform (in-place)
#[inline]
fn identity64_1d(c: &mut [i32], stride: usize, _min: i32, _max: i32) {
    // For 64x64 identity: out = in * 4
    for i in 0..64 {
        c[i * stride] *= 4;
    }
}

/// Generic 64x64 transform function
///
/// AV1 high-frequency zeroing: only 32x32 coefficients are stored for 64x64
/// transforms. Coeff is column-major with stride 32, and has 1024 elements.
#[inline]
fn inv_txfm_64x64_inner(
    tmp: &mut [i32; 4096],
    coeff: &[i16],
    row_transform: fn(&mut [i32], usize, i32, i32),
    col_transform: fn(&mut [i32], usize, i32, i32),
    row_clip_min: i32,
    row_clip_max: i32,
    col_clip_min: i32,
    col_clip_max: i32,
) {
    // For 64x64: shift=2, rnd=2
    let rnd = 2;
    let shift = 2;
    // Row transform - only first 32 rows have stored coefficients
    for y in 0..32 {
        // Load row from column-major (stride=32, only first 32 columns stored)
        let mut scratch = [0i32; 64];
        for x in 0..32 {
            scratch[x] = coeff[y + x * 32] as i32;
        }
        // Zero-extend: columns 32..63 have no stored coefficients
        for x in 32..64 {
            scratch[x] = 0;
        }
        row_transform(&mut scratch[..64], 1, row_clip_min, row_clip_max);
        for x in 0..64 {
            tmp[y * 64 + x] = ((scratch[x] + rnd) >> shift).clamp(col_clip_min, col_clip_max);
        }
    }
    // Rows 32..63 have no stored coefficients - zero them
    for y in 32..64 {
        for x in 0..64 {
            tmp[y * 64 + x] = 0;
        }
    }

    // Column transform (in-place, row-major with stride 64)
    for x in 0..64 {
        col_transform(&mut tmp[x..], 64, col_clip_min, col_clip_max);
    }
}

/// Add transformed coefficients to destination with SIMD (64x64)
#[cfg(target_arch = "x86_64")]
#[arcane]
fn add_64x64_to_dst(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp: &[i32; 4096],
    coeff: &mut [i16],
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();

    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi16(bitdepth_max as i16);
    let rnd_final = _mm256_set1_epi32(8); // (+ 8) >> 4

    for y in 0..64 {
        let dst_off = y * dst_stride;

        // Process 64 pixels in four 16-pixel chunks
        for chunk in 0..4 {
            let x_base = chunk * 16;
            let dst_chunk_off = dst_off + x_base;

            let d =
                loadu_128!(<&[u8; 16]>::try_from(&dst[dst_chunk_off..dst_chunk_off + 16]).unwrap());
            let d16 = _mm256_cvtepu8_epi16(d);

            let c0 = _mm256_set_epi32(
                tmp[y * 64 + x_base + 7],
                tmp[y * 64 + x_base + 6],
                tmp[y * 64 + x_base + 5],
                tmp[y * 64 + x_base + 4],
                tmp[y * 64 + x_base + 3],
                tmp[y * 64 + x_base + 2],
                tmp[y * 64 + x_base + 1],
                tmp[y * 64 + x_base + 0],
            );
            let c1 = _mm256_set_epi32(
                tmp[y * 64 + x_base + 15],
                tmp[y * 64 + x_base + 14],
                tmp[y * 64 + x_base + 13],
                tmp[y * 64 + x_base + 12],
                tmp[y * 64 + x_base + 11],
                tmp[y * 64 + x_base + 10],
                tmp[y * 64 + x_base + 9],
                tmp[y * 64 + x_base + 8],
            );

            // Final scaling: (c + 8) >> 4
            let c0_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c0, rnd_final));
            let c1_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c1, rnd_final));

            let c16 = _mm256_packs_epi32(c0_scaled, c1_scaled);
            let c16 = _mm256_permute4x64_epi64::<0b11_01_10_00>(c16);

            let sum = _mm256_add_epi16(d16, c16);
            let clamped = _mm256_max_epi16(_mm256_min_epi16(sum, max_val), zero);

            let packed = _mm256_packus_epi16(clamped, clamped);
            let packed = _mm256_permute4x64_epi64::<0b11_01_10_00>(packed);

            storeu_128!(
                <&mut [u8; 16]>::try_from(&mut dst[dst_chunk_off..dst_chunk_off + 16]).unwrap(),
                _mm256_castsi256_si128(packed)
            );
        }
    }

    // Clear coefficients (only 1024 stored due to high-frequency zeroing)
    coeff[..1024].fill(0);
}

/// 64x64 DCT_DCT inner function
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_64x64_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let row_clip_min = i16::MIN as i32;
    let row_clip_max = i16::MAX as i32;
    let col_clip_min = i16::MIN as i32;
    let col_clip_max = i16::MAX as i32;

    let mut tmp = [0i32; 4096];
    inv_txfm_64x64_inner(
        &mut tmp,
        &*coeff,
        dct64_1d,
        dct64_1d,
        row_clip_min,
        row_clip_max,
        col_clip_min,
        col_clip_max,
    );
    add_64x64_to_dst(
        _token,
        &mut *dst,
        dst_stride,
        &tmp,
        &mut *coeff,
        bitdepth_max,
    );
}

/// FFI wrapper for 64x64 DCT_DCT 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_64x64_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, _coeff_len as usize * stride + stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_64x64_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// 64x64 DCT TRANSFORMS 16bpc
// ============================================================================

/// Add transformed coefficients to destination with SIMD (64x64 16bpc)
#[cfg(target_arch = "x86_64")]
#[arcane]
fn add_64x64_to_dst_16bpc(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize,
    tmp: &[i32; 4096],
    coeff: &mut [i16],
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let stride_u16 = dst_stride / 2;

    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi32(bitdepth_max);
    let rnd_final = _mm256_set1_epi32(8); // (+ 8) >> 4

    for y in 0..64 {
        let dst_off = y * stride_u16;

        // Process 64 pixels in eight 8-pixel chunks
        for chunk in 0..8 {
            let x_base = chunk * 8;
            let dst_chunk_off = dst_off + x_base;

            // Load destination pixels (8 u16 = 16 bytes)
            let d =
                loadu_128!(<&[u16; 8]>::try_from(&dst[dst_chunk_off..dst_chunk_off + 8]).unwrap());
            let d_lo = _mm_unpacklo_epi16(d, _mm_setzero_si128());
            let d_hi = _mm_unpackhi_epi16(d, _mm_setzero_si128());

            // Load coefficients
            let c_lo = _mm_set_epi32(
                tmp[y * 64 + x_base + 3],
                tmp[y * 64 + x_base + 2],
                tmp[y * 64 + x_base + 1],
                tmp[y * 64 + x_base + 0],
            );
            let c_hi = _mm_set_epi32(
                tmp[y * 64 + x_base + 7],
                tmp[y * 64 + x_base + 6],
                tmp[y * 64 + x_base + 5],
                tmp[y * 64 + x_base + 4],
            );

            // Combine to 256-bit for faster processing
            let d32 = _mm256_set_m128i(d_hi, d_lo);
            let c32 = _mm256_set_m128i(c_hi, c_lo);

            // Final scaling: (c + 8) >> 4
            let c_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c32, rnd_final));

            // Add to destination
            let sum = _mm256_add_epi32(d32, c_scaled);

            // Clamp to [0, bitdepth_max]
            let clamped = _mm256_max_epi32(_mm256_min_epi32(sum, max_val), zero);

            // Pack to u16 and store
            let lo = _mm256_castsi256_si128(clamped);
            let hi = _mm256_extracti128_si256(clamped, 1);
            let packed = _mm_packus_epi32(lo, hi);
            storeu_128!(
                <&mut [u16; 8]>::try_from(&mut dst[dst_chunk_off..dst_chunk_off + 8]).unwrap(),
                packed
            );
        }
    }

    // Clear coefficients (only 1024 stored due to high-frequency zeroing)
    coeff[..1024].fill(0);
}

/// 64x64 DCT_DCT inner function for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_64x64_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    // For 16bpc: use full i32 range
    let row_clip_min = i32::MIN;
    let row_clip_max = i32::MAX;
    let col_clip_min = i32::MIN;
    let col_clip_max = i32::MAX;

    let mut tmp = [0i32; 4096];
    inv_txfm_64x64_inner(
        &mut tmp,
        &*coeff,
        dct64_1d,
        dct64_1d,
        row_clip_min,
        row_clip_max,
        col_clip_min,
        col_clip_max,
    );
    add_64x64_to_dst_16bpc(
        _token,
        &mut *dst,
        dst_stride,
        &tmp,
        &mut *coeff,
        bitdepth_max,
    );
}

/// FFI wrapper for 64x64 DCT_DCT 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_64x64_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_64x64_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// RECTANGULAR DCT TRANSFORMS 16bpc
// ============================================================================

/// 4x8 DCT_DCT for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_4x8_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let stride_u16 = dst_stride / 2;
    let row_clip_min = i32::MIN;
    let row_clip_max = i32::MAX;
    let col_clip_min = i32::MIN;
    let col_clip_max = i32::MAX;
    let mut tmp = [0i32; 32];

    // is_rect2 = true for 4x8, so apply sqrt(2) scaling
    let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

    // Row transform (4 elements each, 8 rows)
    for y in 0..8 {
        let mut scratch = [0i32; 4];
        for x in 0..4 {
            scratch[x] = rect2_scale(coeff[y + x * 8] as i32);
        }
        dct4_1d(&mut scratch[..4], 1, row_clip_min, row_clip_max);
        for x in 0..4 {
            tmp[y * 4 + x] = scratch[x];
        }
    }

    // Column transform
    for x in 0..4 {
        dct8_1d(&mut tmp[x..], 4, col_clip_min, col_clip_max);
    }

    // Add to destination
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi32(bitdepth_max);

    for y in 0..8 {
        let dst_off = y * stride_u16;

        // Load destination (4 u16 = 8 bytes)
        let d = loadi64!(zerocopy::AsBytes::as_bytes(&dst[dst_off..dst_off + 4]));
        let d32 = _mm_unpacklo_epi16(d, zero);

        // Load and scale coefficients
        let c = _mm_set_epi32(
            (tmp[y * 4 + 3] + 8) >> 4,
            (tmp[y * 4 + 2] + 8) >> 4,
            (tmp[y * 4 + 1] + 8) >> 4,
            (tmp[y * 4 + 0] + 8) >> 4,
        );

        let sum = _mm_add_epi32(d32, c);
        let clamped = _mm_max_epi32(_mm_min_epi32(sum, max_val), zero);
        let packed = _mm_packus_epi32(clamped, clamped);

        storei64!(
            zerocopy::AsBytes::as_bytes_mut(&mut dst[dst_off..dst_off + 4]),
            packed
        );
    }

    // Clear coefficients
    coeff[..32].fill(0);
}

/// FFI wrapper for 4x8 DCT_DCT 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_4x8_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_4x8_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 8x4 DCT_DCT for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_8x4_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let stride_u16 = dst_stride / 2;
    let row_clip_min = i32::MIN;
    let row_clip_max = i32::MAX;
    let col_clip_min = i32::MIN;
    let col_clip_max = i32::MAX;
    let mut tmp = [0i32; 32];

    // is_rect2 = true for 8x4
    let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

    // Row transform (8 elements each, 4 rows)
    for y in 0..4 {
        let mut scratch = [0i32; 8];
        for x in 0..8 {
            scratch[x] = rect2_scale(coeff[y + x * 4] as i32);
        }
        dct8_1d(&mut scratch[..8], 1, row_clip_min, row_clip_max);
        for x in 0..8 {
            tmp[y * 8 + x] = scratch[x];
        }
    }

    // Column transform
    for x in 0..8 {
        dct4_1d(&mut tmp[x..], 8, col_clip_min, col_clip_max);
    }

    // Add to destination
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi32(bitdepth_max);

    for y in 0..4 {
        let dst_off = y * stride_u16;

        // Load destination (8 u16 = 16 bytes)
        let d = loadu_128!(<&[u16; 8]>::try_from(&dst[dst_off..dst_off + 8]).unwrap());
        let d_lo = _mm_unpacklo_epi16(d, zero);
        let d_hi = _mm_unpackhi_epi16(d, zero);

        // Load and scale coefficients
        let c_lo = _mm_set_epi32(
            (tmp[y * 8 + 3] + 8) >> 4,
            (tmp[y * 8 + 2] + 8) >> 4,
            (tmp[y * 8 + 1] + 8) >> 4,
            (tmp[y * 8 + 0] + 8) >> 4,
        );
        let c_hi = _mm_set_epi32(
            (tmp[y * 8 + 7] + 8) >> 4,
            (tmp[y * 8 + 6] + 8) >> 4,
            (tmp[y * 8 + 5] + 8) >> 4,
            (tmp[y * 8 + 4] + 8) >> 4,
        );

        let sum_lo = _mm_add_epi32(d_lo, c_lo);
        let sum_hi = _mm_add_epi32(d_hi, c_hi);
        let clamped_lo = _mm_max_epi32(_mm_min_epi32(sum_lo, max_val), zero);
        let clamped_hi = _mm_max_epi32(_mm_min_epi32(sum_hi, max_val), zero);
        let packed = _mm_packus_epi32(clamped_lo, clamped_hi);

        storeu_128!(
            <&mut [u16; 8]>::try_from(&mut dst[dst_off..dst_off + 8]).unwrap(),
            packed
        );
    }

    // Clear coefficients
    coeff[..32].fill(0);
}

/// FFI wrapper for 8x4 DCT_DCT 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_8x4_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_8x4_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 8x16 DCT_DCT for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_8x16_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let stride_u16 = dst_stride / 2;
    let row_clip_min = i32::MIN;
    let row_clip_max = i32::MAX;
    let col_clip_min = i32::MIN;
    let col_clip_max = i32::MAX;
    let mut tmp = [0i32; 128];

    // is_rect2 = true for 8x16
    let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

    // Row transform with shift=1
    let rnd = 1;
    let shift = 1;

    for y in 0..16 {
        let mut scratch = [0i32; 8];
        for x in 0..8 {
            scratch[x] = rect2_scale(coeff[y + x * 16] as i32);
        }
        dct8_1d(&mut scratch[..8], 1, row_clip_min, row_clip_max);
        for x in 0..8 {
            tmp[y * 8 + x] = (scratch[x] + rnd) >> shift;
        }
    }

    // Column transform
    for x in 0..8 {
        dct16_1d(&mut tmp[x..], 8, col_clip_min, col_clip_max);
    }

    // Add to destination
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi32(bitdepth_max);

    for y in 0..16 {
        let dst_off = y * stride_u16;

        // Load destination (8 u16)
        let d = loadu_128!(<&[u16; 8]>::try_from(&dst[dst_off..dst_off + 8]).unwrap());
        let d_lo = _mm_unpacklo_epi16(d, zero);
        let d_hi = _mm_unpackhi_epi16(d, zero);

        let c_lo = _mm_set_epi32(
            (tmp[y * 8 + 3] + 8) >> 4,
            (tmp[y * 8 + 2] + 8) >> 4,
            (tmp[y * 8 + 1] + 8) >> 4,
            (tmp[y * 8 + 0] + 8) >> 4,
        );
        let c_hi = _mm_set_epi32(
            (tmp[y * 8 + 7] + 8) >> 4,
            (tmp[y * 8 + 6] + 8) >> 4,
            (tmp[y * 8 + 5] + 8) >> 4,
            (tmp[y * 8 + 4] + 8) >> 4,
        );

        let sum_lo = _mm_add_epi32(d_lo, c_lo);
        let sum_hi = _mm_add_epi32(d_hi, c_hi);
        let clamped_lo = _mm_max_epi32(_mm_min_epi32(sum_lo, max_val), zero);
        let clamped_hi = _mm_max_epi32(_mm_min_epi32(sum_hi, max_val), zero);
        let packed = _mm_packus_epi32(clamped_lo, clamped_hi);

        storeu_128!(
            <&mut [u16; 8]>::try_from(&mut dst[dst_off..dst_off + 8]).unwrap(),
            packed
        );
    }

    // Clear coefficients
    coeff[..128].fill(0);
}

/// FFI wrapper for 8x16 DCT_DCT 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_8x16_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_8x16_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 16x8 DCT_DCT for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_16x8_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let stride_u16 = dst_stride / 2;
    let row_clip_min = i32::MIN;
    let row_clip_max = i32::MAX;
    let col_clip_min = i32::MIN;
    let col_clip_max = i32::MAX;
    let mut tmp = [0i32; 128];

    // is_rect2 = true for 16x8
    let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

    // Row transform with shift=1
    let rnd = 1;
    let shift = 1;

    for y in 0..8 {
        let mut scratch = [0i32; 16];
        for x in 0..16 {
            scratch[x] = rect2_scale(coeff[y + x * 8] as i32);
        }
        dct16_1d(&mut scratch[..16], 1, row_clip_min, row_clip_max);
        for x in 0..16 {
            tmp[y * 16 + x] = (scratch[x] + rnd) >> shift;
        }
    }

    // Column transform
    for x in 0..16 {
        dct8_1d(&mut tmp[x..], 16, col_clip_min, col_clip_max);
    }

    // Add to destination
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi32(bitdepth_max);
    let rnd_final = _mm256_set1_epi32(8);

    for y in 0..8 {
        let dst_off = y * stride_u16;

        // Load destination (16 u16 = 32 bytes)
        let d = loadu_256!(<&[u16; 16]>::try_from(&dst[dst_off..dst_off + 16]).unwrap());
        let d_lo = _mm256_unpacklo_epi16(d, _mm256_setzero_si256());
        let d_hi = _mm256_unpackhi_epi16(d, _mm256_setzero_si256());
        let d_0_4 = _mm256_permute2x128_si256(d_lo, d_hi, 0x20);
        let d_4_8 = _mm256_permute2x128_si256(d_lo, d_hi, 0x31);

        let c0 = _mm256_set_epi32(
            tmp[y * 16 + 3],
            tmp[y * 16 + 2],
            tmp[y * 16 + 1],
            tmp[y * 16 + 0],
            tmp[y * 16 + 11],
            tmp[y * 16 + 10],
            tmp[y * 16 + 9],
            tmp[y * 16 + 8],
        );
        let c1 = _mm256_set_epi32(
            tmp[y * 16 + 7],
            tmp[y * 16 + 6],
            tmp[y * 16 + 5],
            tmp[y * 16 + 4],
            tmp[y * 16 + 15],
            tmp[y * 16 + 14],
            tmp[y * 16 + 13],
            tmp[y * 16 + 12],
        );

        let c0_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c0, rnd_final));
        let c1_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c1, rnd_final));

        let sum0 = _mm256_add_epi32(d_0_4, c0_scaled);
        let sum1 = _mm256_add_epi32(d_4_8, c1_scaled);

        let clamped0 = _mm256_max_epi32(_mm256_min_epi32(sum0, max_val), zero);
        let clamped1 = _mm256_max_epi32(_mm256_min_epi32(sum1, max_val), zero);

        let packed = _mm256_packus_epi32(clamped0, clamped1);
        let packed = _mm256_permute4x64_epi64(packed, 0b11_01_10_00);
        storeu_256!(
            <&mut [u16; 16]>::try_from(&mut dst[dst_off..dst_off + 16]).unwrap(),
            packed
        );
    }

    // Clear coefficients
    coeff[..128].fill(0);
}

/// FFI wrapper for 16x8 DCT_DCT 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_16x8_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_16x8_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 4x16 DCT_DCT for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_4x16_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let stride_u16 = dst_stride / 2;
    let row_clip_min = i32::MIN;
    let row_clip_max = i32::MAX;
    let col_clip_min = i32::MIN;
    let col_clip_max = i32::MAX;
    let mut tmp = [0i32; 64];

    // is_rect2 = false for 4x16 (4:1 ratio), no rect2_scale
    // Row transform (4 elements each, 16 rows)
    for y in 0..16 {
        let mut scratch = [0i32; 4];
        for x in 0..4 {
            scratch[x] = coeff[y + x * 16] as i32;
        }
        dct4_1d(&mut scratch[..4], 1, row_clip_min, row_clip_max);
        for x in 0..4 {
            tmp[y * 4 + x] = scratch[x];
        }
    }

    // Column transform
    for x in 0..4 {
        dct16_1d(&mut tmp[x..], 4, col_clip_min, col_clip_max);
    }

    // Add to destination
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi32(bitdepth_max);

    for y in 0..16 {
        let dst_off = y * stride_u16;

        let d = loadi64!(zerocopy::AsBytes::as_bytes(&dst[dst_off..dst_off + 4]));
        let d32 = _mm_unpacklo_epi16(d, zero);

        let c = _mm_set_epi32(
            (tmp[y * 4 + 3] + 8) >> 4,
            (tmp[y * 4 + 2] + 8) >> 4,
            (tmp[y * 4 + 1] + 8) >> 4,
            (tmp[y * 4 + 0] + 8) >> 4,
        );

        let sum = _mm_add_epi32(d32, c);
        let clamped = _mm_max_epi32(_mm_min_epi32(sum, max_val), zero);
        let packed = _mm_packus_epi32(clamped, clamped);

        storei64!(
            zerocopy::AsBytes::as_bytes_mut(&mut dst[dst_off..dst_off + 4]),
            packed
        );
    }

    // Clear coefficients
    coeff[..64].fill(0);
}

/// FFI wrapper for 4x16 DCT_DCT 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_4x16_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_4x16_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 16x4 DCT_DCT for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_16x4_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let stride_u16 = dst_stride / 2;
    let row_clip_min = i32::MIN;
    let row_clip_max = i32::MAX;
    let col_clip_min = i32::MIN;
    let col_clip_max = i32::MAX;
    let mut tmp = [0i32; 64];

    // is_rect2 = false for 16x4 (4:1 ratio), no rect2_scale
    // Row transform (16 elements each, 4 rows)
    for y in 0..4 {
        let mut scratch = [0i32; 16];
        for x in 0..16 {
            scratch[x] = coeff[y + x * 4] as i32;
        }
        dct16_1d(&mut scratch[..16], 1, row_clip_min, row_clip_max);
        for x in 0..16 {
            tmp[y * 16 + x] = scratch[x];
        }
    }

    // Column transform
    for x in 0..16 {
        dct4_1d(&mut tmp[x..], 16, col_clip_min, col_clip_max);
    }

    // Add to destination
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi32(bitdepth_max);
    let rnd_final = _mm256_set1_epi32(8);

    for y in 0..4 {
        let dst_off = y * stride_u16;

        let d = loadu_256!(<&[u16; 16]>::try_from(&dst[dst_off..dst_off + 16]).unwrap());
        let d_lo = _mm256_unpacklo_epi16(d, _mm256_setzero_si256());
        let d_hi = _mm256_unpackhi_epi16(d, _mm256_setzero_si256());
        let d_0_4 = _mm256_permute2x128_si256(d_lo, d_hi, 0x20);
        let d_4_8 = _mm256_permute2x128_si256(d_lo, d_hi, 0x31);

        let c0 = _mm256_set_epi32(
            tmp[y * 16 + 3],
            tmp[y * 16 + 2],
            tmp[y * 16 + 1],
            tmp[y * 16 + 0],
            tmp[y * 16 + 11],
            tmp[y * 16 + 10],
            tmp[y * 16 + 9],
            tmp[y * 16 + 8],
        );
        let c1 = _mm256_set_epi32(
            tmp[y * 16 + 7],
            tmp[y * 16 + 6],
            tmp[y * 16 + 5],
            tmp[y * 16 + 4],
            tmp[y * 16 + 15],
            tmp[y * 16 + 14],
            tmp[y * 16 + 13],
            tmp[y * 16 + 12],
        );

        let c0_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c0, rnd_final));
        let c1_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c1, rnd_final));

        let sum0 = _mm256_add_epi32(d_0_4, c0_scaled);
        let sum1 = _mm256_add_epi32(d_4_8, c1_scaled);

        let clamped0 = _mm256_max_epi32(_mm256_min_epi32(sum0, max_val), zero);
        let clamped1 = _mm256_max_epi32(_mm256_min_epi32(sum1, max_val), zero);

        let packed = _mm256_packus_epi32(clamped0, clamped1);
        let packed = _mm256_permute4x64_epi64(packed, 0b11_01_10_00);
        storeu_256!(
            <&mut [u16; 16]>::try_from(&mut dst[dst_off..dst_off + 16]).unwrap(),
            packed
        );
    }

    // Clear coefficients
    coeff[..64].fill(0);
}

/// FFI wrapper for 16x4 DCT_DCT 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_16x4_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_16x4_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 16x32 DCT_DCT for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_16x32_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let stride_u16 = dst_stride / 2;
    let row_clip_min = i32::MIN;
    let row_clip_max = i32::MAX;
    let col_clip_min = i32::MIN;
    let col_clip_max = i32::MAX;
    let mut tmp = [0i32; 512];

    // is_rect2 = true for 16x32
    let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

    // Row transform with shift=1
    let rnd = 1;
    let shift = 1;

    for y in 0..32 {
        let mut scratch = [0i32; 16];
        for x in 0..16 {
            scratch[x] = rect2_scale(coeff[y + x * 32] as i32);
        }
        dct16_1d(&mut scratch[..16], 1, row_clip_min, row_clip_max);
        for x in 0..16 {
            tmp[y * 16 + x] = (scratch[x] + rnd) >> shift;
        }
    }

    // Column transform
    for x in 0..16 {
        dct32_1d(&mut tmp[x..], 16, col_clip_min, col_clip_max);
    }

    // Add to destination
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi32(bitdepth_max);
    let rnd_final = _mm256_set1_epi32(8); // (+ 8) >> 4

    for y in 0..32 {
        let dst_off = y * stride_u16;

        let d = loadu_256!(<&[u16; 16]>::try_from(&dst[dst_off..dst_off + 16]).unwrap());
        let d_lo = _mm256_unpacklo_epi16(d, _mm256_setzero_si256());
        let d_hi = _mm256_unpackhi_epi16(d, _mm256_setzero_si256());
        let d_0_4 = _mm256_permute2x128_si256(d_lo, d_hi, 0x20);
        let d_4_8 = _mm256_permute2x128_si256(d_lo, d_hi, 0x31);

        let c0 = _mm256_set_epi32(
            tmp[y * 16 + 3],
            tmp[y * 16 + 2],
            tmp[y * 16 + 1],
            tmp[y * 16 + 0],
            tmp[y * 16 + 11],
            tmp[y * 16 + 10],
            tmp[y * 16 + 9],
            tmp[y * 16 + 8],
        );
        let c1 = _mm256_set_epi32(
            tmp[y * 16 + 7],
            tmp[y * 16 + 6],
            tmp[y * 16 + 5],
            tmp[y * 16 + 4],
            tmp[y * 16 + 15],
            tmp[y * 16 + 14],
            tmp[y * 16 + 13],
            tmp[y * 16 + 12],
        );

        let c0_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c0, rnd_final));
        let c1_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c1, rnd_final));

        let sum0 = _mm256_add_epi32(d_0_4, c0_scaled);
        let sum1 = _mm256_add_epi32(d_4_8, c1_scaled);

        let clamped0 = _mm256_max_epi32(_mm256_min_epi32(sum0, max_val), zero);
        let clamped1 = _mm256_max_epi32(_mm256_min_epi32(sum1, max_val), zero);

        let packed = _mm256_packus_epi32(clamped0, clamped1);
        let packed = _mm256_permute4x64_epi64(packed, 0b11_01_10_00);
        storeu_256!(
            <&mut [u16; 16]>::try_from(&mut dst[dst_off..dst_off + 16]).unwrap(),
            packed
        );
    }

    // Clear coefficients
    coeff[..512].fill(0);
}

/// FFI wrapper for 16x32 DCT_DCT 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_16x32_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_16x32_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 32x16 DCT_DCT for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_32x16_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let stride_u16 = dst_stride / 2;
    let row_clip_min = i32::MIN;
    let row_clip_max = i32::MAX;
    let col_clip_min = i32::MIN;
    let col_clip_max = i32::MAX;
    let mut tmp = [0i32; 512];

    // is_rect2 = true for 32x16
    let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

    // Row transform with shift=1
    let rnd = 1;
    let shift = 1;

    for y in 0..16 {
        let mut scratch = [0i32; 32];
        for x in 0..32 {
            scratch[x] = rect2_scale(coeff[y + x * 16] as i32);
        }
        dct32_1d(&mut scratch[..32], 1, row_clip_min, row_clip_max);
        for x in 0..32 {
            tmp[y * 32 + x] = (scratch[x] + rnd) >> shift;
        }
    }

    // Column transform
    for x in 0..32 {
        dct16_1d(&mut tmp[x..], 32, col_clip_min, col_clip_max);
    }

    // Add to destination
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi32(bitdepth_max);
    let rnd_final = _mm256_set1_epi32(8); // (+ 8) >> 4

    for y in 0..16 {
        let dst_off = y * stride_u16;

        // Process 32 pixels in four 8-pixel chunks
        for chunk in 0..4 {
            let x_base = chunk * 8;
            let dst_chunk_off = dst_off + x_base;

            let d =
                loadu_128!(<&[u16; 8]>::try_from(&dst[dst_chunk_off..dst_chunk_off + 8]).unwrap());
            let d_lo = _mm_unpacklo_epi16(d, _mm_setzero_si128());
            let d_hi = _mm_unpackhi_epi16(d, _mm_setzero_si128());

            let c_lo = _mm_set_epi32(
                tmp[y * 32 + x_base + 3],
                tmp[y * 32 + x_base + 2],
                tmp[y * 32 + x_base + 1],
                tmp[y * 32 + x_base + 0],
            );
            let c_hi = _mm_set_epi32(
                tmp[y * 32 + x_base + 7],
                tmp[y * 32 + x_base + 6],
                tmp[y * 32 + x_base + 5],
                tmp[y * 32 + x_base + 4],
            );

            let d32 = _mm256_set_m128i(d_hi, d_lo);
            let c32 = _mm256_set_m128i(c_hi, c_lo);

            let c_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c32, rnd_final));
            let sum = _mm256_add_epi32(d32, c_scaled);
            let clamped = _mm256_max_epi32(_mm256_min_epi32(sum, max_val), zero);

            let lo = _mm256_castsi256_si128(clamped);
            let hi = _mm256_extracti128_si256(clamped, 1);
            let packed = _mm_packus_epi32(lo, hi);
            storeu_128!(
                <&mut [u16; 8]>::try_from(&mut dst[dst_chunk_off..dst_chunk_off + 8]).unwrap(),
                packed
            );
        }
    }

    // Clear coefficients
    coeff[..512].fill(0);
}

/// FFI wrapper for 32x16 DCT_DCT 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_32x16_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_32x16_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 8x32 DCT_DCT for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_8x32_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let stride_u16 = dst_stride / 2;
    let row_clip_min = i32::MIN;
    let row_clip_max = i32::MAX;
    let col_clip_min = i32::MIN;
    let col_clip_max = i32::MAX;
    let mut tmp = [0i32; 256];

    // is_rect2 = false for 8x32 (4:1 ratio), no rect2_scale
    // Row transform with shift=1
    let rnd = 1;
    let shift = 1;

    for y in 0..32 {
        let mut scratch = [0i32; 8];
        for x in 0..8 {
            scratch[x] = coeff[y + x * 32] as i32;
        }
        dct8_1d(&mut scratch[..8], 1, row_clip_min, row_clip_max);
        for x in 0..8 {
            tmp[y * 8 + x] = (scratch[x] + rnd) >> shift;
        }
    }

    // Column transform
    for x in 0..8 {
        dct32_1d(&mut tmp[x..], 8, col_clip_min, col_clip_max);
    }

    // Add to destination
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi32(bitdepth_max);

    for y in 0..32 {
        let dst_off = y * stride_u16;

        let d = loadu_128!(<&[u16; 8]>::try_from(&dst[dst_off..dst_off + 8]).unwrap());
        let d_lo = _mm_unpacklo_epi16(d, zero);
        let d_hi = _mm_unpackhi_epi16(d, zero);

        // (+ 4) >> 3 for 8x32
        let c_lo = _mm_set_epi32(
            (tmp[y * 8 + 3] + 4) >> 3,
            (tmp[y * 8 + 2] + 4) >> 3,
            (tmp[y * 8 + 1] + 4) >> 3,
            (tmp[y * 8 + 0] + 4) >> 3,
        );
        let c_hi = _mm_set_epi32(
            (tmp[y * 8 + 7] + 4) >> 3,
            (tmp[y * 8 + 6] + 4) >> 3,
            (tmp[y * 8 + 5] + 4) >> 3,
            (tmp[y * 8 + 4] + 4) >> 3,
        );

        let sum_lo = _mm_add_epi32(d_lo, c_lo);
        let sum_hi = _mm_add_epi32(d_hi, c_hi);
        let clamped_lo = _mm_max_epi32(_mm_min_epi32(sum_lo, max_val), zero);
        let clamped_hi = _mm_max_epi32(_mm_min_epi32(sum_hi, max_val), zero);
        let packed = _mm_packus_epi32(clamped_lo, clamped_hi);

        storeu_128!(
            <&mut [u16; 8]>::try_from(&mut dst[dst_off..dst_off + 8]).unwrap(),
            packed
        );
    }

    // Clear coefficients
    coeff[..256].fill(0);
}

/// FFI wrapper for 8x32 DCT_DCT 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_8x32_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_8x32_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 32x8 DCT_DCT for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_32x8_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let stride_u16 = dst_stride / 2;
    let row_clip_min = i32::MIN;
    let row_clip_max = i32::MAX;
    let col_clip_min = i32::MIN;
    let col_clip_max = i32::MAX;
    let mut tmp = [0i32; 256];

    // is_rect2 = false for 32x8 (4:1 ratio), no rect2_scale
    // Row transform with shift=1
    let rnd = 1;
    let shift = 1;

    for y in 0..8 {
        let mut scratch = [0i32; 32];
        for x in 0..32 {
            scratch[x] = coeff[y + x * 8] as i32;
        }
        dct32_1d(&mut scratch[..32], 1, row_clip_min, row_clip_max);
        for x in 0..32 {
            tmp[y * 32 + x] = (scratch[x] + rnd) >> shift;
        }
    }

    // Column transform
    for x in 0..32 {
        dct8_1d(&mut tmp[x..], 32, col_clip_min, col_clip_max);
    }

    // Add to destination
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi32(bitdepth_max);
    let rnd_final = _mm256_set1_epi32(8); // (+ 8) >> 4

    for y in 0..8 {
        let dst_off = y * stride_u16;

        // Process 32 pixels in four 8-pixel chunks
        for chunk in 0..4 {
            let x_base = chunk * 8;
            let dst_chunk_off = dst_off + x_base;

            let d =
                loadu_128!(<&[u16; 8]>::try_from(&dst[dst_chunk_off..dst_chunk_off + 8]).unwrap());
            let d_lo = _mm_unpacklo_epi16(d, _mm_setzero_si128());
            let d_hi = _mm_unpackhi_epi16(d, _mm_setzero_si128());

            let c_lo = _mm_set_epi32(
                tmp[y * 32 + x_base + 3],
                tmp[y * 32 + x_base + 2],
                tmp[y * 32 + x_base + 1],
                tmp[y * 32 + x_base + 0],
            );
            let c_hi = _mm_set_epi32(
                tmp[y * 32 + x_base + 7],
                tmp[y * 32 + x_base + 6],
                tmp[y * 32 + x_base + 5],
                tmp[y * 32 + x_base + 4],
            );

            let d32 = _mm256_set_m128i(d_hi, d_lo);
            let c32 = _mm256_set_m128i(c_hi, c_lo);

            let c_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c32, rnd_final));
            let sum = _mm256_add_epi32(d32, c_scaled);
            let clamped = _mm256_max_epi32(_mm256_min_epi32(sum, max_val), zero);

            let lo = _mm256_castsi256_si128(clamped);
            let hi = _mm256_extracti128_si256(clamped, 1);
            let packed = _mm_packus_epi32(lo, hi);
            storeu_128!(
                <&mut [u16; 8]>::try_from(&mut dst[dst_chunk_off..dst_chunk_off + 8]).unwrap(),
                packed
            );
        }
    }

    // Clear coefficients
    coeff[..256].fill(0);
}

/// FFI wrapper for 32x8 DCT_DCT 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_32x8_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_32x8_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 32x64 DCT_DCT for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_32x64_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let stride_u16 = dst_stride / 2;
    let row_clip_min = i32::MIN;
    let row_clip_max = i32::MAX;
    let col_clip_min = i32::MIN;
    let col_clip_max = i32::MAX;
    let mut tmp = [0i32; 2048];

    // is_rect2 = true for 32x64
    let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

    // Row transform - only first 32 rows have stored coefficients (high-freq zeroing)
    // Coeff is column-major with stride 32
    for y in 0..32 {
        let mut scratch = [0i32; 32];
        for x in 0..32 {
            scratch[x] = rect2_scale(coeff[y + x * 32] as i32);
        }
        dct32_1d(&mut scratch[..32], 1, row_clip_min, row_clip_max);
        for x in 0..32 {
            tmp[y * 32 + x] = scratch[x];
        }
    }
    // Zero-pad rows 32..63
    for y in 32..64 {
        for x in 0..32 {
            tmp[y * 32 + x] = 0;
        }
    }

    // Column transform
    for x in 0..32 {
        dct64_1d(&mut tmp[x..], 32, col_clip_min, col_clip_max);
    }

    // Add to destination
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi32(bitdepth_max);
    let rnd_final = _mm256_set1_epi32(8); // (+ 8) >> 4

    for y in 0..64 {
        let dst_off = y * stride_u16;

        // Process 32 pixels in four 8-pixel chunks
        for chunk in 0..4 {
            let x_base = chunk * 8;
            let dst_chunk_off = dst_off + x_base;

            let d =
                loadu_128!(<&[u16; 8]>::try_from(&dst[dst_chunk_off..dst_chunk_off + 8]).unwrap());
            let d_lo = _mm_unpacklo_epi16(d, _mm_setzero_si128());
            let d_hi = _mm_unpackhi_epi16(d, _mm_setzero_si128());

            let c_lo = _mm_set_epi32(
                tmp[y * 32 + x_base + 3],
                tmp[y * 32 + x_base + 2],
                tmp[y * 32 + x_base + 1],
                tmp[y * 32 + x_base + 0],
            );
            let c_hi = _mm_set_epi32(
                tmp[y * 32 + x_base + 7],
                tmp[y * 32 + x_base + 6],
                tmp[y * 32 + x_base + 5],
                tmp[y * 32 + x_base + 4],
            );

            let d32 = _mm256_set_m128i(d_hi, d_lo);
            let c32 = _mm256_set_m128i(c_hi, c_lo);

            let c_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c32, rnd_final));
            let sum = _mm256_add_epi32(d32, c_scaled);
            let clamped = _mm256_max_epi32(_mm256_min_epi32(sum, max_val), zero);

            let lo = _mm256_castsi256_si128(clamped);
            let hi = _mm256_extracti128_si256(clamped, 1);
            let packed = _mm_packus_epi32(lo, hi);
            storeu_128!(
                <&mut [u16; 8]>::try_from(&mut dst[dst_chunk_off..dst_chunk_off + 8]).unwrap(),
                packed
            );
        }
    }

    // Clear coefficients (only 1024 stored due to high-frequency zeroing)
    coeff[..1024].fill(0);
}

/// FFI wrapper for 32x64 DCT_DCT 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_32x64_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_32x64_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 64x32 DCT_DCT for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_64x32_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let stride_u16 = dst_stride / 2;
    let row_clip_min = i32::MIN;
    let row_clip_max = i32::MAX;
    let col_clip_min = i32::MIN;
    let col_clip_max = i32::MAX;
    let mut tmp = [0i32; 2048];

    // is_rect2 = true for 64x32
    let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

    // Row transform - only first 32 columns have stored coefficients (high-freq zeroing)
    // Coeff is column-major with stride 32
    for y in 0..32 {
        let mut scratch = [0i32; 64];
        for x in 0..32 {
            scratch[x] = rect2_scale(coeff[y + x * 32] as i32);
        }
        // Zero-extend: columns 32..63 have no stored coefficients
        for x in 32..64 {
            scratch[x] = 0;
        }
        dct64_1d(&mut scratch[..64], 1, row_clip_min, row_clip_max);
        for x in 0..64 {
            tmp[y * 64 + x] = scratch[x];
        }
    }

    // Column transform
    for x in 0..64 {
        dct32_1d(&mut tmp[x..], 64, col_clip_min, col_clip_max);
    }

    // Add to destination
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi32(bitdepth_max);
    let rnd_final = _mm256_set1_epi32(8); // (+ 8) >> 4

    for y in 0..32 {
        let dst_off = y * stride_u16;

        // Process 64 pixels in eight 8-pixel chunks
        for chunk in 0..8 {
            let x_base = chunk * 8;
            let dst_chunk_off = dst_off + x_base;

            let d =
                loadu_128!(<&[u16; 8]>::try_from(&dst[dst_chunk_off..dst_chunk_off + 8]).unwrap());
            let d_lo = _mm_unpacklo_epi16(d, _mm_setzero_si128());
            let d_hi = _mm_unpackhi_epi16(d, _mm_setzero_si128());

            let c_lo = _mm_set_epi32(
                tmp[y * 64 + x_base + 3],
                tmp[y * 64 + x_base + 2],
                tmp[y * 64 + x_base + 1],
                tmp[y * 64 + x_base + 0],
            );
            let c_hi = _mm_set_epi32(
                tmp[y * 64 + x_base + 7],
                tmp[y * 64 + x_base + 6],
                tmp[y * 64 + x_base + 5],
                tmp[y * 64 + x_base + 4],
            );

            let d32 = _mm256_set_m128i(d_hi, d_lo);
            let c32 = _mm256_set_m128i(c_hi, c_lo);

            let c_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c32, rnd_final));
            let sum = _mm256_add_epi32(d32, c_scaled);
            let clamped = _mm256_max_epi32(_mm256_min_epi32(sum, max_val), zero);

            let lo = _mm256_castsi256_si128(clamped);
            let hi = _mm256_extracti128_si256(clamped, 1);
            let packed = _mm_packus_epi32(lo, hi);
            storeu_128!(
                <&mut [u16; 8]>::try_from(&mut dst[dst_chunk_off..dst_chunk_off + 8]).unwrap(),
                packed
            );
        }
    }

    // Clear coefficients (only 1024 stored due to high-frequency zeroing)
    coeff[..1024].fill(0);
}

/// FFI wrapper for 64x32 DCT_DCT 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_64x32_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_64x32_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 16x64 DCT_DCT for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_16x64_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let stride_u16 = dst_stride / 2;
    let row_clip_min = i32::MIN;
    let row_clip_max = i32::MAX;
    let col_clip_min = i32::MIN;
    let col_clip_max = i32::MAX;
    let mut tmp = [0i32; 1024];

    // is_rect2 = false for 16x64 (4:1 ratio), no rect2_scale
    // Row transform - only first 32 rows have stored coefficients (high-freq zeroing)
    // Coeff is column-major with stride 32
    for y in 0..32 {
        let mut scratch = [0i32; 16];
        for x in 0..16 {
            scratch[x] = coeff[y + x * 32] as i32;
        }
        dct16_1d(&mut scratch[..16], 1, row_clip_min, row_clip_max);
        for x in 0..16 {
            tmp[y * 16 + x] = scratch[x];
        }
    }
    // Zero-pad rows 32..63
    for y in 32..64 {
        for x in 0..16 {
            tmp[y * 16 + x] = 0;
        }
    }

    // Column transform
    for x in 0..16 {
        dct64_1d(&mut tmp[x..], 16, col_clip_min, col_clip_max);
    }

    // Add to destination
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi32(bitdepth_max);
    let rnd_final = _mm256_set1_epi32(8); // (+ 8) >> 4

    for y in 0..64 {
        let dst_off = y * stride_u16;

        let d = loadu_256!(<&[u16; 16]>::try_from(&dst[dst_off..dst_off + 16]).unwrap());
        let d_lo = _mm256_unpacklo_epi16(d, _mm256_setzero_si256());
        let d_hi = _mm256_unpackhi_epi16(d, _mm256_setzero_si256());
        let d_0_4 = _mm256_permute2x128_si256(d_lo, d_hi, 0x20);
        let d_4_8 = _mm256_permute2x128_si256(d_lo, d_hi, 0x31);

        let c0 = _mm256_set_epi32(
            tmp[y * 16 + 3],
            tmp[y * 16 + 2],
            tmp[y * 16 + 1],
            tmp[y * 16 + 0],
            tmp[y * 16 + 11],
            tmp[y * 16 + 10],
            tmp[y * 16 + 9],
            tmp[y * 16 + 8],
        );
        let c1 = _mm256_set_epi32(
            tmp[y * 16 + 7],
            tmp[y * 16 + 6],
            tmp[y * 16 + 5],
            tmp[y * 16 + 4],
            tmp[y * 16 + 15],
            tmp[y * 16 + 14],
            tmp[y * 16 + 13],
            tmp[y * 16 + 12],
        );

        let c0_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c0, rnd_final));
        let c1_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c1, rnd_final));

        let sum0 = _mm256_add_epi32(d_0_4, c0_scaled);
        let sum1 = _mm256_add_epi32(d_4_8, c1_scaled);

        let clamped0 = _mm256_max_epi32(_mm256_min_epi32(sum0, max_val), zero);
        let clamped1 = _mm256_max_epi32(_mm256_min_epi32(sum1, max_val), zero);

        let packed = _mm256_packus_epi32(clamped0, clamped1);
        let packed = _mm256_permute4x64_epi64(packed, 0b11_01_10_00);
        storeu_256!(
            <&mut [u16; 16]>::try_from(&mut dst[dst_off..dst_off + 16]).unwrap(),
            packed
        );
    }

    // Clear coefficients (only 512 stored due to high-frequency zeroing)
    coeff[..512].fill(0);
}

/// FFI wrapper for 16x64 DCT_DCT 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_16x64_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_16x64_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 64x16 DCT_DCT for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_64x16_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let stride_u16 = dst_stride / 2;
    let row_clip_min = i32::MIN;
    let row_clip_max = i32::MAX;
    let col_clip_min = i32::MIN;
    let col_clip_max = i32::MAX;
    let mut tmp = [0i32; 1024];

    // is_rect2 = false for 64x16 (4:1 ratio), no rect2_scale
    // Row transform - only first 32 columns have stored coefficients (high-freq zeroing)
    // Coeff is column-major with stride 16
    for y in 0..16 {
        let mut scratch = [0i32; 64];
        for x in 0..32 {
            scratch[x] = coeff[y + x * 16] as i32;
        }
        // Zero-extend: columns 32..63 have no stored coefficients
        for x in 32..64 {
            scratch[x] = 0;
        }
        dct64_1d(&mut scratch[..64], 1, row_clip_min, row_clip_max);
        for x in 0..64 {
            tmp[y * 64 + x] = scratch[x];
        }
    }

    // Column transform
    for x in 0..64 {
        dct16_1d(&mut tmp[x..], 64, col_clip_min, col_clip_max);
    }

    // Add to destination
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi32(bitdepth_max);
    let rnd_final = _mm256_set1_epi32(8); // (+ 8) >> 4

    for y in 0..16 {
        let dst_off = y * stride_u16;

        // Process 64 pixels in eight 8-pixel chunks
        for chunk in 0..8 {
            let x_base = chunk * 8;
            let dst_chunk_off = dst_off + x_base;

            let d =
                loadu_128!(<&[u16; 8]>::try_from(&dst[dst_chunk_off..dst_chunk_off + 8]).unwrap());
            let d_lo = _mm_unpacklo_epi16(d, _mm_setzero_si128());
            let d_hi = _mm_unpackhi_epi16(d, _mm_setzero_si128());

            let c_lo = _mm_set_epi32(
                tmp[y * 64 + x_base + 3],
                tmp[y * 64 + x_base + 2],
                tmp[y * 64 + x_base + 1],
                tmp[y * 64 + x_base + 0],
            );
            let c_hi = _mm_set_epi32(
                tmp[y * 64 + x_base + 7],
                tmp[y * 64 + x_base + 6],
                tmp[y * 64 + x_base + 5],
                tmp[y * 64 + x_base + 4],
            );

            let d32 = _mm256_set_m128i(d_hi, d_lo);
            let c32 = _mm256_set_m128i(c_hi, c_lo);

            let c_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c32, rnd_final));
            let sum = _mm256_add_epi32(d32, c_scaled);
            let clamped = _mm256_max_epi32(_mm256_min_epi32(sum, max_val), zero);

            let lo = _mm256_castsi256_si128(clamped);
            let hi = _mm256_extracti128_si256(clamped, 1);
            let packed = _mm_packus_epi32(lo, hi);
            storeu_128!(
                <&mut [u16; 8]>::try_from(&mut dst[dst_chunk_off..dst_chunk_off + 8]).unwrap(),
                packed
            );
        }
    }

    // Clear coefficients (only 512 stored due to high-frequency zeroing)
    coeff[..512].fill(0);
}

/// FFI wrapper for 64x16 DCT_DCT 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_64x16_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_64x16_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// 8x8 ADST/FLIPADST TRANSFORMS 16bpc
// ============================================================================

/// Helper macro for 8x8 transform implementations (16bpc)
macro_rules! impl_8x8_transform_16bpc {
    ($name:ident, $row_fn:ident, $col_fn:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[arcane]
        pub fn $name(
            _token: Desktop64,
            dst: &mut [u16],
            dst_stride: usize,
            coeff: &mut [i16],
            _eob: i32,
            bitdepth_max: i32,
        ) {
            use crate::src::safe_simd::pixel_access::{
                loadi32, loadi64, loadu_128, storei32, storei64, storeu_128,
            };
            let mut dst = dst.flex_mut();
            let mut coeff = coeff.flex_mut();
            let stride_u16 = dst_stride / 2;
            const MIN: i32 = i32::MIN;
            const MAX: i32 = i32::MAX;

            // Load coefficients (row-major)
            let mut c = [[0i32; 8]; 8];
            for y in 0..8 {
                for x in 0..8 {
                    c[y][x] = coeff[y * 8 + x] as i32;
                }
            }

            // First pass: transform on rows
            let mut tmp = [[0i32; 8]; 8];
            for y in 0..8 {
                let (o0, o1, o2, o3, o4, o5, o6, o7) = $row_fn(
                    c[y][0], c[y][1], c[y][2], c[y][3], c[y][4], c[y][5], c[y][6], c[y][7], MIN,
                    MAX,
                );
                tmp[y][0] = o0;
                tmp[y][1] = o1;
                tmp[y][2] = o2;
                tmp[y][3] = o3;
                tmp[y][4] = o4;
                tmp[y][5] = o5;
                tmp[y][6] = o6;
                tmp[y][7] = o7;
            }

            // Second pass: transform on columns
            let mut out = [[0i32; 8]; 8];
            for x in 0..8 {
                let (o0, o1, o2, o3, o4, o5, o6, o7) = $col_fn(
                    tmp[0][x], tmp[1][x], tmp[2][x], tmp[3][x], tmp[4][x], tmp[5][x], tmp[6][x],
                    tmp[7][x], MIN, MAX,
                );
                out[0][x] = o0;
                out[1][x] = o1;
                out[2][x] = o2;
                out[3][x] = o3;
                out[4][x] = o4;
                out[5][x] = o5;
                out[6][x] = o6;
                out[7][x] = o7;
            }

            // Add to destination with rounding
            for y in 0..8 {
                let dst_off = y * stride_u16;
                for x in 0..8 {
                    let pixel = dst[dst_off + x] as i32;
                    let val = pixel + ((out[y][x] + 8) >> 4);
                    dst[dst_off + x] = val.clamp(0, bitdepth_max) as u16;
                }
            }

            // Clear coefficients
            coeff[..64].fill(0);
        }
    };
}

// Generate all 8x8 ADST/FlipADST combinations for 16bpc
impl_8x8_transform_16bpc!(
    inv_txfm_add_adst_dct_8x8_16bpc_avx2_inner,
    adst8_1d_scalar,
    dct8_1d_scalar
);
impl_8x8_transform_16bpc!(
    inv_txfm_add_dct_adst_8x8_16bpc_avx2_inner,
    dct8_1d_scalar,
    adst8_1d_scalar
);
impl_8x8_transform_16bpc!(
    inv_txfm_add_adst_adst_8x8_16bpc_avx2_inner,
    adst8_1d_scalar,
    adst8_1d_scalar
);
impl_8x8_transform_16bpc!(
    inv_txfm_add_flipadst_dct_8x8_16bpc_avx2_inner,
    flipadst8_1d_scalar,
    dct8_1d_scalar
);
impl_8x8_transform_16bpc!(
    inv_txfm_add_dct_flipadst_8x8_16bpc_avx2_inner,
    dct8_1d_scalar,
    flipadst8_1d_scalar
);
impl_8x8_transform_16bpc!(
    inv_txfm_add_flipadst_flipadst_8x8_16bpc_avx2_inner,
    flipadst8_1d_scalar,
    flipadst8_1d_scalar
);
impl_8x8_transform_16bpc!(
    inv_txfm_add_adst_flipadst_8x8_16bpc_avx2_inner,
    adst8_1d_scalar,
    flipadst8_1d_scalar
);
impl_8x8_transform_16bpc!(
    inv_txfm_add_flipadst_adst_8x8_16bpc_avx2_inner,
    flipadst8_1d_scalar,
    adst8_1d_scalar
);

// FFI wrappers for 8x8 16bpc transforms
macro_rules! impl_8x8_ffi_wrapper_16bpc {
    ($wrapper:ident, $inner:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2")]
        #[cfg(feature = "asm")]
        pub unsafe extern "C" fn $wrapper(
            dst_ptr: *mut DynPixel,
            dst_stride: isize,
            coeff: *mut DynCoef,
            eob: c_int,
            bitdepth_max: c_int,
            _coeff_len: u16,
            _dst: *const FFISafe<PicOffset>,
        ) {
            let _token = unsafe { Desktop64::forge_token_dangerously() };
            let stride = dst_stride as usize;

            let dst_slice = unsafe {
                std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
            };

            let coeff_slice =
                unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

            $inner(_token, dst_slice, stride, coeff_slice, eob, bitdepth_max);
        }
    };
}

impl_8x8_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_dct_8x8_16bpc_avx2,
    inv_txfm_add_adst_dct_8x8_16bpc_avx2_inner
);
impl_8x8_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_adst_8x8_16bpc_avx2,
    inv_txfm_add_dct_adst_8x8_16bpc_avx2_inner
);
impl_8x8_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_adst_8x8_16bpc_avx2,
    inv_txfm_add_adst_adst_8x8_16bpc_avx2_inner
);
impl_8x8_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_dct_8x8_16bpc_avx2,
    inv_txfm_add_flipadst_dct_8x8_16bpc_avx2_inner
);
impl_8x8_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_flipadst_8x8_16bpc_avx2,
    inv_txfm_add_dct_flipadst_8x8_16bpc_avx2_inner
);
impl_8x8_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_flipadst_8x8_16bpc_avx2,
    inv_txfm_add_flipadst_flipadst_8x8_16bpc_avx2_inner
);
impl_8x8_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_flipadst_8x8_16bpc_avx2,
    inv_txfm_add_adst_flipadst_8x8_16bpc_avx2_inner
);
impl_8x8_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_adst_8x8_16bpc_avx2,
    inv_txfm_add_flipadst_adst_8x8_16bpc_avx2_inner
);

// ============================================================================
// 4x4 ADST/FLIPADST TRANSFORMS 16bpc
// ============================================================================

/// Helper macro for 4x4 transform implementations (16bpc)
macro_rules! impl_4x4_transform_16bpc {
    ($name:ident, $row_fn:ident, $col_fn:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[cfg(feature = "asm")]
        pub fn $name(
            dst: &mut [u16],
            dst_base: usize,
            dst_stride_u16: isize,
            coeff: &mut [i16],
            _eob: i32,
            bitdepth_max: i32,
        ) {
            use crate::src::safe_simd::pixel_access::{
                loadi32, loadi64, loadu_128, storei32, storei64, storeu_128,
            };
            // Load coefficients (row-major)
            let mut c = [[0i32; 4]; 4];
            for y in 0..4 {
                for x in 0..4 {
                    c[y][x] = coeff[y + x * 4] as i32;
                }
            }

            // Clip ranges for 16bpc (matching reference inv_txfm_add_rust)
            let row_clip_min = (!bitdepth_max) << 7;
            let row_clip_max = !row_clip_min;
            let col_clip_min = (!bitdepth_max) << 5;
            let col_clip_max = !col_clip_min;

            // First pass: transform on rows
            let mut tmp = [[0i32; 4]; 4];
            for y in 0..4 {
                let (o0, o1, o2, o3) = $row_fn(c[y][0], c[y][1], c[y][2], c[y][3], row_clip_min, row_clip_max);
                tmp[y][0] = o0;
                tmp[y][1] = o1;
                tmp[y][2] = o2;
                tmp[y][3] = o3;
            }

            // Intermediate clip (shift=0 for 4x4)
            for y in 0..4 {
                for x in 0..4 {
                    tmp[y][x] = tmp[y][x].clamp(col_clip_min, col_clip_max);
                }
            }

            // Second pass: transform on columns
            let mut out = [[0i32; 4]; 4];
            for x in 0..4 {
                let (o0, o1, o2, o3) = $col_fn(tmp[0][x], tmp[1][x], tmp[2][x], tmp[3][x], col_clip_min, col_clip_max);
                out[0][x] = o0;
                out[1][x] = o1;
                out[2][x] = o2;
                out[3][x] = o3;
            }

            // Add to destination with rounding
            for y in 0..4 {
                let row_off = dst_base.wrapping_add_signed(y as isize * dst_stride_u16);
                for x in 0..4 {
                    let pixel = dst[row_off + x] as i32;
                    let val = pixel + ((out[y][x] + 8) >> 4);
                    dst[row_off + x] = val.clamp(0, bitdepth_max) as u16;
                }
            }

            // Clear coefficients
            coeff[..16].fill(0);
        }
    };
}

// Generate all 4x4 ADST/FlipADST combinations for 16bpc
impl_4x4_transform_16bpc!(
    inv_txfm_add_adst_dct_4x4_16bpc_avx2_inner,
    adst4_1d_scalar,
    dct4_1d_scalar
);
impl_4x4_transform_16bpc!(
    inv_txfm_add_dct_adst_4x4_16bpc_avx2_inner,
    dct4_1d_scalar,
    adst4_1d_scalar
);
impl_4x4_transform_16bpc!(
    inv_txfm_add_adst_adst_4x4_16bpc_avx2_inner,
    adst4_1d_scalar,
    adst4_1d_scalar
);
impl_4x4_transform_16bpc!(
    inv_txfm_add_flipadst_dct_4x4_16bpc_avx2_inner,
    flipadst4_1d_scalar,
    dct4_1d_scalar
);
impl_4x4_transform_16bpc!(
    inv_txfm_add_dct_flipadst_4x4_16bpc_avx2_inner,
    dct4_1d_scalar,
    flipadst4_1d_scalar
);
impl_4x4_transform_16bpc!(
    inv_txfm_add_flipadst_flipadst_4x4_16bpc_avx2_inner,
    flipadst4_1d_scalar,
    flipadst4_1d_scalar
);
impl_4x4_transform_16bpc!(
    inv_txfm_add_adst_flipadst_4x4_16bpc_avx2_inner,
    adst4_1d_scalar,
    flipadst4_1d_scalar
);
impl_4x4_transform_16bpc!(
    inv_txfm_add_flipadst_adst_4x4_16bpc_avx2_inner,
    flipadst4_1d_scalar,
    adst4_1d_scalar
);

// FFI wrappers for 4x4 16bpc transforms
macro_rules! impl_4x4_ffi_wrapper_16bpc {
    ($wrapper:ident, $inner:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2")]
        #[cfg(feature = "asm")]
        pub unsafe extern "C" fn $wrapper(
            dst_ptr: *mut DynPixel,
            dst_stride: isize,
            coeff: *mut DynCoef,
            eob: c_int,
            bitdepth_max: c_int,
            _coeff_len: u16,
            _dst: *const FFISafe<PicOffset>,
        ) {
            let _token = unsafe { Desktop64::forge_token_dangerously() };
            let stride_u16 = dst_stride / 2;
            let coeff_slice = unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, 16) };
            let abs_stride = stride_u16;
            let (dst_slice, dst_base) = if stride_u16 >= 0 {
                let len = 3 * abs_stride + 4;
                (
                    unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u16, len) },
                    0usize,
                )
            } else {
                let len = 3 * abs_stride + 4;
                let start = unsafe { (dst_ptr as *mut u16).offset(3 * stride_u16) };
                (
                    unsafe { std::slice::from_raw_parts_mut(start, len) },
                    3 * abs_stride,
                )
            };
            $inner(
                dst_slice,
                dst_base,
                stride_u16,
                coeff_slice,
                eob,
                bitdepth_max,
            );
        }
    };
}

impl_4x4_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_dct_4x4_16bpc_avx2,
    inv_txfm_add_adst_dct_4x4_16bpc_avx2_inner
);
impl_4x4_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_adst_4x4_16bpc_avx2,
    inv_txfm_add_dct_adst_4x4_16bpc_avx2_inner
);
impl_4x4_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_adst_4x4_16bpc_avx2,
    inv_txfm_add_adst_adst_4x4_16bpc_avx2_inner
);
impl_4x4_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_dct_4x4_16bpc_avx2,
    inv_txfm_add_flipadst_dct_4x4_16bpc_avx2_inner
);
impl_4x4_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_flipadst_4x4_16bpc_avx2,
    inv_txfm_add_dct_flipadst_4x4_16bpc_avx2_inner
);
impl_4x4_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_flipadst_4x4_16bpc_avx2,
    inv_txfm_add_flipadst_flipadst_4x4_16bpc_avx2_inner
);
impl_4x4_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_flipadst_4x4_16bpc_avx2,
    inv_txfm_add_adst_flipadst_4x4_16bpc_avx2_inner
);
impl_4x4_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_adst_4x4_16bpc_avx2,
    inv_txfm_add_flipadst_adst_4x4_16bpc_avx2_inner
);

// ============================================================================
// 16x16 ADST/FLIPADST TRANSFORMS 16bpc
// ============================================================================

/// Helper macro for 16x16 transform implementations (16bpc)
macro_rules! impl_16x16_transform_16bpc {
    ($name:ident, $row_fn:ident, $col_fn:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[arcane]
        pub fn $name(
            _token: Desktop64,
            dst: &mut [u16],
            dst_stride: usize,
            coeff: &mut [i16],
            _eob: i32,
            bitdepth_max: i32,
        ) {
            use crate::src::safe_simd::pixel_access::{
                loadi32, loadi64, loadu_128, storei32, storei64, storeu_128,
            };
            let mut dst = dst.flex_mut();
            let mut coeff = coeff.flex_mut();
            let stride_u16 = dst_stride / 2;
            const MIN: i32 = i32::MIN;
            const MAX: i32 = i32::MAX;

            // Load coefficients (row-major)
            let mut c = [[0i32; 16]; 16];
            for y in 0..16 {
                for x in 0..16 {
                    c[y][x] = coeff[y * 16 + x] as i32;
                }
            }

            // First pass: transform on rows
            let mut tmp = [[0i32; 16]; 16];
            for y in 0..16 {
                let mut row = [0i32; 16];
                for x in 0..16 {
                    row[x] = c[y][x];
                }
                $row_fn(&mut row, 1, MIN, MAX);
                for x in 0..16 {
                    tmp[y][x] = row[x];
                }
            }

            // Second pass: transform on columns
            let mut out = [[0i32; 16]; 16];
            for x in 0..16 {
                let mut col = [0i32; 16];
                for y in 0..16 {
                    col[y] = tmp[y][x];
                }
                $col_fn(&mut col, 1, MIN, MAX);
                for y in 0..16 {
                    out[y][x] = col[y];
                }
            }

            // Add to destination with rounding
            for y in 0..16 {
                let dst_off = y * stride_u16;
                for x in 0..16 {
                    let pixel = dst[dst_off + x] as i32;
                    let val = pixel + ((out[y][x] + 8) >> 4);
                    dst[dst_off + x] = val.clamp(0, bitdepth_max) as u16;
                }
            }

            // Clear coefficients
            coeff[..256].fill(0);
        }
    };
}

// Generate key 16x16 ADST combinations for 16bpc
impl_16x16_transform_16bpc!(
    inv_txfm_add_adst_dct_16x16_16bpc_avx2_inner,
    adst16_1d,
    dct16_1d
);
impl_16x16_transform_16bpc!(
    inv_txfm_add_dct_adst_16x16_16bpc_avx2_inner,
    dct16_1d,
    adst16_1d
);
impl_16x16_transform_16bpc!(
    inv_txfm_add_adst_adst_16x16_16bpc_avx2_inner,
    adst16_1d,
    adst16_1d
);
impl_16x16_transform_16bpc!(
    inv_txfm_add_flipadst_dct_16x16_16bpc_avx2_inner,
    flipadst16_1d,
    dct16_1d
);
impl_16x16_transform_16bpc!(
    inv_txfm_add_dct_flipadst_16x16_16bpc_avx2_inner,
    dct16_1d,
    flipadst16_1d
);
impl_16x16_transform_16bpc!(
    inv_txfm_add_flipadst_flipadst_16x16_16bpc_avx2_inner,
    flipadst16_1d,
    flipadst16_1d
);
impl_16x16_transform_16bpc!(
    inv_txfm_add_adst_flipadst_16x16_16bpc_avx2_inner,
    adst16_1d,
    flipadst16_1d
);
impl_16x16_transform_16bpc!(
    inv_txfm_add_flipadst_adst_16x16_16bpc_avx2_inner,
    flipadst16_1d,
    adst16_1d
);

// FFI wrappers for 16x16 16bpc transforms
macro_rules! impl_16x16_ffi_wrapper_16bpc {
    ($wrapper:ident, $inner:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2")]
        #[cfg(feature = "asm")]
        pub unsafe extern "C" fn $wrapper(
            dst_ptr: *mut DynPixel,
            dst_stride: isize,
            coeff: *mut DynCoef,
            eob: c_int,
            bitdepth_max: c_int,
            _coeff_len: u16,
            _dst: *const FFISafe<PicOffset>,
        ) {
            let _token = unsafe { Desktop64::forge_token_dangerously() };
            let stride = dst_stride as usize;

            let dst_slice = unsafe {
                std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
            };

            let coeff_slice =
                unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

            $inner(_token, dst_slice, stride, coeff_slice, eob, bitdepth_max);
        }
    };
}

impl_16x16_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_dct_16x16_16bpc_avx2,
    inv_txfm_add_adst_dct_16x16_16bpc_avx2_inner
);
impl_16x16_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_adst_16x16_16bpc_avx2,
    inv_txfm_add_dct_adst_16x16_16bpc_avx2_inner
);
impl_16x16_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_adst_16x16_16bpc_avx2,
    inv_txfm_add_adst_adst_16x16_16bpc_avx2_inner
);
impl_16x16_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_dct_16x16_16bpc_avx2,
    inv_txfm_add_flipadst_dct_16x16_16bpc_avx2_inner
);
impl_16x16_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_flipadst_16x16_16bpc_avx2,
    inv_txfm_add_dct_flipadst_16x16_16bpc_avx2_inner
);
impl_16x16_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_flipadst_16x16_16bpc_avx2,
    inv_txfm_add_flipadst_flipadst_16x16_16bpc_avx2_inner
);
impl_16x16_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_flipadst_16x16_16bpc_avx2,
    inv_txfm_add_adst_flipadst_16x16_16bpc_avx2_inner
);
impl_16x16_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_adst_16x16_16bpc_avx2,
    inv_txfm_add_flipadst_adst_16x16_16bpc_avx2_inner
);

// ============================================================================
// IDENTITY TRANSFORMS 16bpc
// ============================================================================

/// 4x4 IDTX (identity transform) for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
pub fn inv_identity_add_4x4_16bpc_avx2(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let stride_u16 = dst_stride / 2;
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi32(bitdepth_max);

    for y in 0..4 {
        let dst_off = y * stride_u16;

        // Load destination (4 u16)
        let d = loadi64!(zerocopy::AsBytes::as_bytes(&dst[dst_off..dst_off + 4]));
        let d32 = _mm_unpacklo_epi16(d, zero);

        // Load coeffs (column-major: y, y+4, y+8, y+12)
        let c0 = coeff[y] as i32;
        let c1 = coeff[y + 4] as i32;
        let c2 = coeff[y + 8] as i32;
        let c3 = coeff[y + 12] as i32;

        // identity4(x) = x + (x * 1697 + 2048) >> 12 ≈ x * sqrt(2)
        // Row: identity4, shift=0, Col: identity4, final: (+ 8) >> 4
        let identity4 = |v: i32| -> i32 { v + ((v * 1697 + 2048) >> 12) };
        let scale = |v: i32| -> i32 { identity4(identity4(v)) };

        let r0 = (scale(c0) + 8) >> 4;
        let r1 = (scale(c1) + 8) >> 4;
        let r2 = (scale(c2) + 8) >> 4;
        let r3 = (scale(c3) + 8) >> 4;

        // Add to destination
        let result = _mm_set_epi32(r3, r2, r1, r0);
        let sum = _mm_add_epi32(d32, result);
        let clamped = _mm_max_epi32(_mm_min_epi32(sum, max_val), zero);
        let packed = _mm_packus_epi32(clamped, clamped);

        storei64!(
            zerocopy::AsBytes::as_bytes_mut(&mut dst[dst_off..dst_off + 4]),
            packed
        );
    }

    // Clear coefficients
    coeff[..16].fill(0);
}

/// FFI wrapper for 4x4 IDTX 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_identity_4x4_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_identity_add_4x4_16bpc_avx2(_token, dst_slice, stride, coeff_slice, eob, bitdepth_max);
}

/// 8x8 IDTX (identity transform) for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
pub fn inv_identity_add_8x8_16bpc_avx2(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let stride_u16 = dst_stride / 2;
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi32(bitdepth_max);

    for y in 0..8 {
        let dst_off = y * stride_u16;

        // Load destination (8 u16)
        let d = loadu_128!(<&[u16; 8]>::try_from(&dst[dst_off..dst_off + 8]).unwrap());
        let d_lo = _mm_unpacklo_epi16(d, zero);
        let d_hi = _mm_unpackhi_epi16(d, zero);

        // Load coefficients (column-major: y, y+8, ...)
        let mut coeffs = [0i32; 8];
        for x in 0..8 {
            coeffs[x] = coeff[y + x * 8] as i32;
        }

        // Identity8: * 2 per dimension, with intermediate shift between passes
        // For 8x8: shift=1, rnd=1, col_clip = (!bitdepth_max) << 5
        // Row: c * 2. Intermediate: iclip((c*2 + 1) >> 1, ...). Col: * 2. Final: (+ 8) >> 4
        let col_clip_min = (!(bitdepth_max)) << 5;
        let col_clip_max = !col_clip_min;
        let mut results = [0i32; 8];
        for x in 0..8 {
            let row = coeffs[x] * 2; // identity8 row
            let inter = ((row + 1) >> 1).clamp(col_clip_min, col_clip_max); // intermediate shift
            let col = inter * 2; // identity8 col
            results[x] = (col + 8) >> 4; // final shift
        }

        let c_lo = _mm_set_epi32(results[3], results[2], results[1], results[0]);
        let c_hi = _mm_set_epi32(results[7], results[6], results[5], results[4]);

        // Add to destination
        let sum_lo = _mm_add_epi32(d_lo, c_lo);
        let sum_hi = _mm_add_epi32(d_hi, c_hi);
        let clamped_lo = _mm_max_epi32(_mm_min_epi32(sum_lo, max_val), zero);
        let clamped_hi = _mm_max_epi32(_mm_min_epi32(sum_hi, max_val), zero);
        let packed = _mm_packus_epi32(clamped_lo, clamped_hi);

        storeu_128!(
            <&mut [u16; 8]>::try_from(&mut dst[dst_off..dst_off + 8]).unwrap(),
            packed
        );
    }

    // Clear coefficients
    coeff[..64].fill(0);
}

/// FFI wrapper for 8x8 IDTX 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_identity_8x8_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_identity_add_8x8_16bpc_avx2(_token, dst_slice, stride, coeff_slice, eob, bitdepth_max);
}

/// 16x16 IDTX (identity transform) for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
pub fn inv_identity_add_16x16_16bpc_avx2(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let stride_u16 = dst_stride / 2;

    // Identity16 scale factor: f(x) = 2*x + (x*1697 + 1024) >> 11
    // For 16x16: row_pass → intermediate shift (+ 2) >> 2 with clamp → col_pass → (+ 8) >> 4
    // The intermediate shift is shift=2, rnd=2 for 16x16 (from inv_txfm_add_rust)
    // 16bpc clamp: col_clip_min = (!bitdepth_max) << 5, col_clip_max = !col_clip_min
    let identity16_scale = |v: i32| -> i32 { 2 * v + ((v * 1697 + 1024) >> 11) };
    let col_clip_min = (!(bitdepth_max)) << 5;
    let col_clip_max = !col_clip_min;

    // Row pass: identity16 on each row
    let mut tmp = [[0i32; 16]; 16];
    for y in 0..16 {
        for x in 0..16 {
            let c = coeff[y + x * 16] as i32;
            tmp[y][x] = identity16_scale(c);
        }
    }

    // Intermediate shift: (val + 2) >> 2, clamped to col_clip range
    for y in 0..16 {
        for x in 0..16 {
            tmp[y][x] = ((tmp[y][x] + 2) >> 2).clamp(col_clip_min, col_clip_max);
        }
    }

    // Column pass: identity16, then final shift
    for x in 0..16 {
        for y in 0..16 {
            tmp[y][x] = identity16_scale(tmp[y][x]);
        }
    }

    // Add to destination
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi32(bitdepth_max);
    let rnd_final = _mm256_set1_epi32(8);

    for y in 0..16 {
        let dst_off = y * stride_u16;

        let d = loadu_256!(<&[u16; 16]>::try_from(&dst[dst_off..dst_off + 16]).unwrap());
        let d_lo = _mm256_unpacklo_epi16(d, _mm256_setzero_si256());
        let d_hi = _mm256_unpackhi_epi16(d, _mm256_setzero_si256());
        let d_0_4 = _mm256_permute2x128_si256(d_lo, d_hi, 0x20);
        let d_4_8 = _mm256_permute2x128_si256(d_lo, d_hi, 0x31);

        let c0 = _mm256_set_epi32(
            tmp[y][3], tmp[y][2], tmp[y][1], tmp[y][0], tmp[y][11], tmp[y][10], tmp[y][9],
            tmp[y][8],
        );
        let c1 = _mm256_set_epi32(
            tmp[y][7], tmp[y][6], tmp[y][5], tmp[y][4], tmp[y][15], tmp[y][14], tmp[y][13],
            tmp[y][12],
        );

        let c0_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c0, rnd_final));
        let c1_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c1, rnd_final));

        let sum0 = _mm256_add_epi32(d_0_4, c0_scaled);
        let sum1 = _mm256_add_epi32(d_4_8, c1_scaled);

        let clamped0 = _mm256_max_epi32(_mm256_min_epi32(sum0, max_val), zero);
        let clamped1 = _mm256_max_epi32(_mm256_min_epi32(sum1, max_val), zero);

        let packed = _mm256_packus_epi32(clamped0, clamped1);
        let packed = _mm256_permute4x64_epi64(packed, 0b11_01_10_00);
        storeu_256!(
            <&mut [u16; 16]>::try_from(&mut dst[dst_off..dst_off + 16]).unwrap(),
            packed
        );
    }

    // Clear coefficients
    coeff[..256].fill(0);
}

/// FFI wrapper for 16x16 IDTX 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_identity_16x16_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_identity_add_16x16_16bpc_avx2(_token, dst_slice, stride, coeff_slice, eob, bitdepth_max);
}

// ============================================================================
// 32x32 IDTX 16bpc
// ============================================================================

/// 32x32 IDTX inner function for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_identity_identity_32x32_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    // For 16bpc: use full i32 range
    let row_clip_min = i32::MIN;
    let row_clip_max = i32::MAX;
    let col_clip_min = i32::MIN;
    let col_clip_max = i32::MAX;

    let mut tmp = [0i32; 1024];
    inv_txfm_32x32_inner(
        &mut tmp,
        &*coeff,
        identity32_1d,
        identity32_1d,
        row_clip_min,
        row_clip_max,
        col_clip_min,
        col_clip_max,
    );
    add_32x32_to_dst_16bpc(
        _token,
        &mut *dst,
        dst_stride,
        &tmp,
        &mut *coeff,
        bitdepth_max,
    );
}

/// FFI wrapper for 32x32 IDTX 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_identity_32x32_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_identity_identity_32x32_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// Rectangular IDTX 16bpc
// ============================================================================

/// 4x8 IDTX for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_identity_identity_4x8_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let stride_u16 = dst_stride / 2;
    let row_clip_min = i32::MIN;
    let row_clip_max = i32::MAX;
    let col_clip_min = i32::MIN;
    let col_clip_max = i32::MAX;
    let mut tmp = [0i32; 32];

    // is_rect2 = true for 4x8, so apply sqrt(2) scaling
    let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

    // Row transform (4 elements each, 8 rows)
    for y in 0..8 {
        let mut scratch = [0i32; 4];
        for x in 0..4 {
            scratch[x] = rect2_scale(coeff[y + x * 8] as i32);
        }
        identity4_1d(&mut scratch[..4], 1, row_clip_min, row_clip_max);
        for x in 0..4 {
            tmp[y * 4 + x] = scratch[x];
        }
    }

    // Column transform
    for x in 0..4 {
        identity8_1d(&mut tmp[x..], 4, col_clip_min, col_clip_max);
    }

    // Add to destination
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi32(bitdepth_max);

    for y in 0..8 {
        let dst_off = y * stride_u16;

        // Load destination (4 u16 = 8 bytes)
        let d = loadi64!(zerocopy::AsBytes::as_bytes(&dst[dst_off..dst_off + 4]));
        let d32 = _mm_unpacklo_epi16(d, zero);

        // Load and scale coefficients
        let c = _mm_set_epi32(
            (tmp[y * 4 + 3] + 8) >> 4,
            (tmp[y * 4 + 2] + 8) >> 4,
            (tmp[y * 4 + 1] + 8) >> 4,
            (tmp[y * 4 + 0] + 8) >> 4,
        );

        let sum = _mm_add_epi32(d32, c);
        let clamped = _mm_max_epi32(_mm_min_epi32(sum, max_val), zero);
        let packed = _mm_packus_epi32(clamped, clamped);

        storei64!(
            zerocopy::AsBytes::as_bytes_mut(&mut dst[dst_off..dst_off + 4]),
            packed
        );
    }

    // Clear coefficients
    coeff[..32].fill(0);
}

/// FFI wrapper for 4x8 IDTX 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_identity_4x8_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_identity_identity_4x8_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 8x4 IDTX for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_identity_identity_8x4_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let stride_u16 = dst_stride / 2;
    let row_clip_min = i32::MIN;
    let row_clip_max = i32::MAX;
    let col_clip_min = i32::MIN;
    let col_clip_max = i32::MAX;
    let mut tmp = [0i32; 32];

    // is_rect2 = true for 8x4, so apply sqrt(2) scaling
    let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

    // Row transform (8 elements each, 4 rows)
    for y in 0..4 {
        let mut scratch = [0i32; 8];
        for x in 0..8 {
            scratch[x] = rect2_scale(coeff[y + x * 4] as i32);
        }
        identity8_1d(&mut scratch[..8], 1, row_clip_min, row_clip_max);
        for x in 0..8 {
            tmp[y * 8 + x] = scratch[x];
        }
    }

    // Column transform
    for x in 0..8 {
        identity4_1d(&mut tmp[x..], 8, col_clip_min, col_clip_max);
    }

    // Add to destination
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi32(bitdepth_max);

    for y in 0..4 {
        let dst_off = y * stride_u16;

        // Load destination (8 u16 = 16 bytes)
        let d = loadu_128!(<&[u16; 8]>::try_from(&dst[dst_off..dst_off + 8]).unwrap());
        let d_lo = _mm_unpacklo_epi16(d, zero);
        let d_hi = _mm_unpackhi_epi16(d, zero);

        // Load and scale coefficients
        let c_lo = _mm_set_epi32(
            (tmp[y * 8 + 3] + 8) >> 4,
            (tmp[y * 8 + 2] + 8) >> 4,
            (tmp[y * 8 + 1] + 8) >> 4,
            (tmp[y * 8 + 0] + 8) >> 4,
        );
        let c_hi = _mm_set_epi32(
            (tmp[y * 8 + 7] + 8) >> 4,
            (tmp[y * 8 + 6] + 8) >> 4,
            (tmp[y * 8 + 5] + 8) >> 4,
            (tmp[y * 8 + 4] + 8) >> 4,
        );

        let sum_lo = _mm_add_epi32(d_lo, c_lo);
        let sum_hi = _mm_add_epi32(d_hi, c_hi);
        let clamped_lo = _mm_max_epi32(_mm_min_epi32(sum_lo, max_val), zero);
        let clamped_hi = _mm_max_epi32(_mm_min_epi32(sum_hi, max_val), zero);
        let packed = _mm_packus_epi32(clamped_lo, clamped_hi);

        storeu_128!(
            <&mut [u16; 8]>::try_from(&mut dst[dst_off..dst_off + 8]).unwrap(),
            packed
        );
    }

    // Clear coefficients
    coeff[..32].fill(0);
}

/// FFI wrapper for 8x4 IDTX 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_identity_8x4_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_identity_identity_8x4_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 8x16 IDTX for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_identity_identity_8x16_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let stride_u16 = dst_stride / 2;
    let row_clip_min = i32::MIN;
    let row_clip_max = i32::MAX;
    let col_clip_min = i32::MIN;
    let col_clip_max = i32::MAX;
    let mut tmp = [0i32; 128];

    // is_rect2 = true for 8x16, so apply sqrt(2) scaling
    let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

    // Row transform (8 elements each, 16 rows)
    for y in 0..16 {
        let mut scratch = [0i32; 8];
        for x in 0..8 {
            scratch[x] = rect2_scale(coeff[y + x * 16] as i32);
        }
        identity8_1d(&mut scratch[..8], 1, row_clip_min, row_clip_max);
        for x in 0..8 {
            tmp[y * 8 + x] = scratch[x];
        }
    }

    // Column transform
    for x in 0..8 {
        identity16_1d(&mut tmp[x..], 8, col_clip_min, col_clip_max);
    }

    // Add to destination
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi32(bitdepth_max);

    for y in 0..16 {
        let dst_off = y * stride_u16;

        // Load destination (8 u16 = 16 bytes)
        let d = loadu_128!(<&[u16; 8]>::try_from(&dst[dst_off..dst_off + 8]).unwrap());
        let d_lo = _mm_unpacklo_epi16(d, zero);
        let d_hi = _mm_unpackhi_epi16(d, zero);

        // Load and scale coefficients - 8x16 uses >> 4 for final shift
        let c_lo = _mm_set_epi32(
            (tmp[y * 8 + 3] + 8) >> 4,
            (tmp[y * 8 + 2] + 8) >> 4,
            (tmp[y * 8 + 1] + 8) >> 4,
            (tmp[y * 8 + 0] + 8) >> 4,
        );
        let c_hi = _mm_set_epi32(
            (tmp[y * 8 + 7] + 8) >> 4,
            (tmp[y * 8 + 6] + 8) >> 4,
            (tmp[y * 8 + 5] + 8) >> 4,
            (tmp[y * 8 + 4] + 8) >> 4,
        );

        let sum_lo = _mm_add_epi32(d_lo, c_lo);
        let sum_hi = _mm_add_epi32(d_hi, c_hi);
        let clamped_lo = _mm_max_epi32(_mm_min_epi32(sum_lo, max_val), zero);
        let clamped_hi = _mm_max_epi32(_mm_min_epi32(sum_hi, max_val), zero);
        let packed = _mm_packus_epi32(clamped_lo, clamped_hi);

        storeu_128!(
            <&mut [u16; 8]>::try_from(&mut dst[dst_off..dst_off + 8]).unwrap(),
            packed
        );
    }

    // Clear coefficients
    coeff[..128].fill(0);
}

/// FFI wrapper for 8x16 IDTX 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_identity_8x16_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_identity_identity_8x16_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 16x8 IDTX for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_identity_identity_16x8_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let stride_u16 = dst_stride / 2;
    let row_clip_min = i32::MIN;
    let row_clip_max = i32::MAX;
    let col_clip_min = i32::MIN;
    let col_clip_max = i32::MAX;
    let mut tmp = [0i32; 128];

    // is_rect2 = true for 16x8, so apply sqrt(2) scaling
    let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

    // Row transform (16 elements each, 8 rows)
    for y in 0..8 {
        let mut scratch = [0i32; 16];
        for x in 0..16 {
            scratch[x] = rect2_scale(coeff[y + x * 8] as i32);
        }
        identity16_1d(&mut scratch[..16], 1, row_clip_min, row_clip_max);
        for x in 0..16 {
            tmp[y * 16 + x] = scratch[x];
        }
    }

    // Column transform
    for x in 0..16 {
        identity8_1d(&mut tmp[x..], 16, col_clip_min, col_clip_max);
    }

    // Add to destination
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi32(bitdepth_max);

    for y in 0..8 {
        let dst_off = y * stride_u16;

        // Process 16 pixels in two 8-pixel chunks
        for chunk in 0..2 {
            let x_base = chunk * 8;
            let dst_chunk_off = dst_off + x_base;

            let d =
                loadu_128!(<&[u16; 8]>::try_from(&dst[dst_chunk_off..dst_chunk_off + 8]).unwrap());
            let d_lo = _mm_unpacklo_epi16(d, zero);
            let d_hi = _mm_unpackhi_epi16(d, zero);

            let c_lo = _mm_set_epi32(
                (tmp[y * 16 + x_base + 3] + 8) >> 4,
                (tmp[y * 16 + x_base + 2] + 8) >> 4,
                (tmp[y * 16 + x_base + 1] + 8) >> 4,
                (tmp[y * 16 + x_base + 0] + 8) >> 4,
            );
            let c_hi = _mm_set_epi32(
                (tmp[y * 16 + x_base + 7] + 8) >> 4,
                (tmp[y * 16 + x_base + 6] + 8) >> 4,
                (tmp[y * 16 + x_base + 5] + 8) >> 4,
                (tmp[y * 16 + x_base + 4] + 8) >> 4,
            );

            let sum_lo = _mm_add_epi32(d_lo, c_lo);
            let sum_hi = _mm_add_epi32(d_hi, c_hi);
            let clamped_lo = _mm_max_epi32(_mm_min_epi32(sum_lo, max_val), zero);
            let clamped_hi = _mm_max_epi32(_mm_min_epi32(sum_hi, max_val), zero);
            let packed = _mm_packus_epi32(clamped_lo, clamped_hi);

            storeu_128!(
                <&mut [u16; 8]>::try_from(&mut dst[dst_chunk_off..dst_chunk_off + 8]).unwrap(),
                packed
            );
        }
    }

    // Clear coefficients
    coeff[..128].fill(0);
}

/// FFI wrapper for 16x8 IDTX 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_identity_16x8_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_identity_identity_16x8_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 4x16 IDTX for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_identity_identity_4x16_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let stride_u16 = dst_stride / 2;
    let row_clip_min = i32::MIN;
    let row_clip_max = i32::MAX;
    let col_clip_min = i32::MIN;
    let col_clip_max = i32::MAX;
    let mut tmp = [0i32; 64];

    // is_rect2 = false for 4x16 (aspect ratio 4:1), no rect2_scale

    // Row transform (4 elements each, 16 rows)
    for y in 0..16 {
        let mut scratch = [0i32; 4];
        for x in 0..4 {
            scratch[x] = coeff[y + x * 16] as i32;
        }
        identity4_1d(&mut scratch[..4], 1, row_clip_min, row_clip_max);
        for x in 0..4 {
            tmp[y * 4 + x] = scratch[x];
        }
    }

    // Column transform
    for x in 0..4 {
        identity16_1d(&mut tmp[x..], 4, col_clip_min, col_clip_max);
    }

    // Add to destination
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi32(bitdepth_max);

    for y in 0..16 {
        let dst_off = y * stride_u16;

        // Load destination (4 u16 = 8 bytes)
        let d = loadi64!(zerocopy::AsBytes::as_bytes(&dst[dst_off..dst_off + 4]));
        let d32 = _mm_unpacklo_epi16(d, zero);

        // Load and scale coefficients - 4x16 uses >> 4 for final shift
        let c = _mm_set_epi32(
            (tmp[y * 4 + 3] + 8) >> 4,
            (tmp[y * 4 + 2] + 8) >> 4,
            (tmp[y * 4 + 1] + 8) >> 4,
            (tmp[y * 4 + 0] + 8) >> 4,
        );

        let sum = _mm_add_epi32(d32, c);
        let clamped = _mm_max_epi32(_mm_min_epi32(sum, max_val), zero);
        let packed = _mm_packus_epi32(clamped, clamped);

        storei64!(
            zerocopy::AsBytes::as_bytes_mut(&mut dst[dst_off..dst_off + 4]),
            packed
        );
    }

    // Clear coefficients
    coeff[..64].fill(0);
}

/// FFI wrapper for 4x16 IDTX 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_identity_4x16_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_identity_identity_4x16_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 16x4 IDTX for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_identity_identity_16x4_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let stride_u16 = dst_stride / 2;
    let row_clip_min = i32::MIN;
    let row_clip_max = i32::MAX;
    let col_clip_min = i32::MIN;
    let col_clip_max = i32::MAX;
    let mut tmp = [0i32; 64];

    // is_rect2 = false for 16x4 (aspect ratio 4:1), no rect2_scale

    // Row transform (16 elements each, 4 rows)
    for y in 0..4 {
        let mut scratch = [0i32; 16];
        for x in 0..16 {
            scratch[x] = coeff[y + x * 4] as i32;
        }
        identity16_1d(&mut scratch[..16], 1, row_clip_min, row_clip_max);
        for x in 0..16 {
            tmp[y * 16 + x] = scratch[x];
        }
    }

    // Column transform
    for x in 0..16 {
        identity4_1d(&mut tmp[x..], 16, col_clip_min, col_clip_max);
    }

    // Add to destination
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi32(bitdepth_max);

    for y in 0..4 {
        let dst_off = y * stride_u16;

        // Process 16 pixels in two 8-pixel chunks
        for chunk in 0..2 {
            let x_base = chunk * 8;
            let dst_chunk_off = dst_off + x_base;

            let d =
                loadu_128!(<&[u16; 8]>::try_from(&dst[dst_chunk_off..dst_chunk_off + 8]).unwrap());
            let d_lo = _mm_unpacklo_epi16(d, zero);
            let d_hi = _mm_unpackhi_epi16(d, zero);

            let c_lo = _mm_set_epi32(
                (tmp[y * 16 + x_base + 3] + 8) >> 4,
                (tmp[y * 16 + x_base + 2] + 8) >> 4,
                (tmp[y * 16 + x_base + 1] + 8) >> 4,
                (tmp[y * 16 + x_base + 0] + 8) >> 4,
            );
            let c_hi = _mm_set_epi32(
                (tmp[y * 16 + x_base + 7] + 8) >> 4,
                (tmp[y * 16 + x_base + 6] + 8) >> 4,
                (tmp[y * 16 + x_base + 5] + 8) >> 4,
                (tmp[y * 16 + x_base + 4] + 8) >> 4,
            );

            let sum_lo = _mm_add_epi32(d_lo, c_lo);
            let sum_hi = _mm_add_epi32(d_hi, c_hi);
            let clamped_lo = _mm_max_epi32(_mm_min_epi32(sum_lo, max_val), zero);
            let clamped_hi = _mm_max_epi32(_mm_min_epi32(sum_hi, max_val), zero);
            let packed = _mm_packus_epi32(clamped_lo, clamped_hi);

            storeu_128!(
                <&mut [u16; 8]>::try_from(&mut dst[dst_chunk_off..dst_chunk_off + 8]).unwrap(),
                packed
            );
        }
    }

    // Clear coefficients
    coeff[..64].fill(0);
}

/// FFI wrapper for 16x4 IDTX 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_identity_16x4_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_identity_identity_16x4_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 16x32 IDTX for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_identity_identity_16x32_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let stride_u16 = dst_stride / 2;
    let row_clip_min = i32::MIN;
    let row_clip_max = i32::MAX;
    let col_clip_min = i32::MIN;
    let col_clip_max = i32::MAX;
    let mut tmp = [0i32; 512];

    // is_rect2 = true for 16x32, so apply sqrt(2) scaling
    let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

    // Row transform (16 elements each, 32 rows)
    for y in 0..32 {
        let mut scratch = [0i32; 16];
        for x in 0..16 {
            scratch[x] = rect2_scale(coeff[y + x * 32] as i32);
        }
        identity16_1d(&mut scratch[..16], 1, row_clip_min, row_clip_max);
        for x in 0..16 {
            tmp[y * 16 + x] = scratch[x];
        }
    }

    // Column transform
    for x in 0..16 {
        identity32_1d(&mut tmp[x..], 16, col_clip_min, col_clip_max);
    }

    // Add to destination
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi32(bitdepth_max);

    for y in 0..32 {
        let dst_off = y * stride_u16;

        // Process 16 pixels in two 8-pixel chunks
        for chunk in 0..2 {
            let x_base = chunk * 8;
            let dst_chunk_off = dst_off + x_base;

            let d =
                loadu_128!(<&[u16; 8]>::try_from(&dst[dst_chunk_off..dst_chunk_off + 8]).unwrap());
            let d_lo = _mm_unpacklo_epi16(d, zero);
            let d_hi = _mm_unpackhi_epi16(d, zero);

            // 16x32 uses >> 3 for final shift
            let c_lo = _mm_set_epi32(
                (tmp[y * 16 + x_base + 3] + 4) >> 3,
                (tmp[y * 16 + x_base + 2] + 4) >> 3,
                (tmp[y * 16 + x_base + 1] + 4) >> 3,
                (tmp[y * 16 + x_base + 0] + 4) >> 3,
            );
            let c_hi = _mm_set_epi32(
                (tmp[y * 16 + x_base + 7] + 4) >> 3,
                (tmp[y * 16 + x_base + 6] + 4) >> 3,
                (tmp[y * 16 + x_base + 5] + 4) >> 3,
                (tmp[y * 16 + x_base + 4] + 4) >> 3,
            );

            let sum_lo = _mm_add_epi32(d_lo, c_lo);
            let sum_hi = _mm_add_epi32(d_hi, c_hi);
            let clamped_lo = _mm_max_epi32(_mm_min_epi32(sum_lo, max_val), zero);
            let clamped_hi = _mm_max_epi32(_mm_min_epi32(sum_hi, max_val), zero);
            let packed = _mm_packus_epi32(clamped_lo, clamped_hi);

            storeu_128!(
                <&mut [u16; 8]>::try_from(&mut dst[dst_chunk_off..dst_chunk_off + 8]).unwrap(),
                packed
            );
        }
    }

    // Clear coefficients
    coeff[..512].fill(0);
}

/// FFI wrapper for 16x32 IDTX 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_identity_16x32_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_identity_identity_16x32_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 32x16 IDTX for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_identity_identity_32x16_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let stride_u16 = dst_stride / 2;
    let row_clip_min = i32::MIN;
    let row_clip_max = i32::MAX;
    let col_clip_min = i32::MIN;
    let col_clip_max = i32::MAX;
    let mut tmp = [0i32; 512];

    // is_rect2 = true for 32x16, so apply sqrt(2) scaling
    let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

    // Row transform (32 elements each, 16 rows)
    for y in 0..16 {
        let mut scratch = [0i32; 32];
        for x in 0..32 {
            scratch[x] = rect2_scale(coeff[y + x * 16] as i32);
        }
        identity32_1d(&mut scratch[..32], 1, row_clip_min, row_clip_max);
        for x in 0..32 {
            tmp[y * 32 + x] = scratch[x];
        }
    }

    // Column transform
    for x in 0..32 {
        identity16_1d(&mut tmp[x..], 32, col_clip_min, col_clip_max);
    }

    // Add to destination
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi32(bitdepth_max);

    for y in 0..16 {
        let dst_off = y * stride_u16;

        // Process 32 pixels in four 8-pixel chunks
        for chunk in 0..4 {
            let x_base = chunk * 8;
            let dst_chunk_off = dst_off + x_base;

            let d =
                loadu_128!(<&[u16; 8]>::try_from(&dst[dst_chunk_off..dst_chunk_off + 8]).unwrap());
            let d_lo = _mm_unpacklo_epi16(d, zero);
            let d_hi = _mm_unpackhi_epi16(d, zero);

            // 32x16 uses >> 3 for final shift
            let c_lo = _mm_set_epi32(
                (tmp[y * 32 + x_base + 3] + 4) >> 3,
                (tmp[y * 32 + x_base + 2] + 4) >> 3,
                (tmp[y * 32 + x_base + 1] + 4) >> 3,
                (tmp[y * 32 + x_base + 0] + 4) >> 3,
            );
            let c_hi = _mm_set_epi32(
                (tmp[y * 32 + x_base + 7] + 4) >> 3,
                (tmp[y * 32 + x_base + 6] + 4) >> 3,
                (tmp[y * 32 + x_base + 5] + 4) >> 3,
                (tmp[y * 32 + x_base + 4] + 4) >> 3,
            );

            let sum_lo = _mm_add_epi32(d_lo, c_lo);
            let sum_hi = _mm_add_epi32(d_hi, c_hi);
            let clamped_lo = _mm_max_epi32(_mm_min_epi32(sum_lo, max_val), zero);
            let clamped_hi = _mm_max_epi32(_mm_min_epi32(sum_hi, max_val), zero);
            let packed = _mm_packus_epi32(clamped_lo, clamped_hi);

            storeu_128!(
                <&mut [u16; 8]>::try_from(&mut dst[dst_chunk_off..dst_chunk_off + 8]).unwrap(),
                packed
            );
        }
    }

    // Clear coefficients
    coeff[..512].fill(0);
}

/// FFI wrapper for 32x16 IDTX 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_identity_32x16_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_identity_identity_32x16_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 8x32 IDTX for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_identity_identity_8x32_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let stride_u16 = dst_stride / 2;
    let row_clip_min = i32::MIN;
    let row_clip_max = i32::MAX;
    let col_clip_min = i32::MIN;
    let col_clip_max = i32::MAX;
    let mut tmp = [0i32; 256];

    // is_rect2 = false for 8x32 (aspect ratio 4:1), no rect2_scale

    // Row transform (8 elements each, 32 rows)
    for y in 0..32 {
        let mut scratch = [0i32; 8];
        for x in 0..8 {
            scratch[x] = coeff[y + x * 32] as i32;
        }
        identity8_1d(&mut scratch[..8], 1, row_clip_min, row_clip_max);
        for x in 0..8 {
            tmp[y * 8 + x] = scratch[x];
        }
    }

    // Column transform
    for x in 0..8 {
        identity32_1d(&mut tmp[x..], 8, col_clip_min, col_clip_max);
    }

    // Add to destination
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi32(bitdepth_max);

    for y in 0..32 {
        let dst_off = y * stride_u16;

        // Load destination (8 u16 = 16 bytes)
        let d = loadu_128!(<&[u16; 8]>::try_from(&dst[dst_off..dst_off + 8]).unwrap());
        let d_lo = _mm_unpacklo_epi16(d, zero);
        let d_hi = _mm_unpackhi_epi16(d, zero);

        // 8x32 uses >> 3 for final shift
        let c_lo = _mm_set_epi32(
            (tmp[y * 8 + 3] + 4) >> 3,
            (tmp[y * 8 + 2] + 4) >> 3,
            (tmp[y * 8 + 1] + 4) >> 3,
            (tmp[y * 8 + 0] + 4) >> 3,
        );
        let c_hi = _mm_set_epi32(
            (tmp[y * 8 + 7] + 4) >> 3,
            (tmp[y * 8 + 6] + 4) >> 3,
            (tmp[y * 8 + 5] + 4) >> 3,
            (tmp[y * 8 + 4] + 4) >> 3,
        );

        let sum_lo = _mm_add_epi32(d_lo, c_lo);
        let sum_hi = _mm_add_epi32(d_hi, c_hi);
        let clamped_lo = _mm_max_epi32(_mm_min_epi32(sum_lo, max_val), zero);
        let clamped_hi = _mm_max_epi32(_mm_min_epi32(sum_hi, max_val), zero);
        let packed = _mm_packus_epi32(clamped_lo, clamped_hi);

        storeu_128!(
            <&mut [u16; 8]>::try_from(&mut dst[dst_off..dst_off + 8]).unwrap(),
            packed
        );
    }

    // Clear coefficients
    coeff[..256].fill(0);
}

/// FFI wrapper for 8x32 IDTX 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_identity_8x32_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_identity_identity_8x32_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 32x8 IDTX for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_identity_identity_32x8_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let stride_u16 = dst_stride / 2;
    let row_clip_min = i32::MIN;
    let row_clip_max = i32::MAX;
    let col_clip_min = i32::MIN;
    let col_clip_max = i32::MAX;
    let mut tmp = [0i32; 256];

    // is_rect2 = false for 32x8 (aspect ratio 4:1), no rect2_scale

    // Row transform (32 elements each, 8 rows)
    for y in 0..8 {
        let mut scratch = [0i32; 32];
        for x in 0..32 {
            scratch[x] = coeff[y + x * 8] as i32;
        }
        identity32_1d(&mut scratch[..32], 1, row_clip_min, row_clip_max);
        for x in 0..32 {
            tmp[y * 32 + x] = scratch[x];
        }
    }

    // Column transform
    for x in 0..32 {
        identity8_1d(&mut tmp[x..], 32, col_clip_min, col_clip_max);
    }

    // Add to destination
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi32(bitdepth_max);

    for y in 0..8 {
        let dst_off = y * stride_u16;

        // Process 32 pixels in four 8-pixel chunks
        for chunk in 0..4 {
            let x_base = chunk * 8;
            let dst_chunk_off = dst_off + x_base;

            let d =
                loadu_128!(<&[u16; 8]>::try_from(&dst[dst_chunk_off..dst_chunk_off + 8]).unwrap());
            let d_lo = _mm_unpacklo_epi16(d, zero);
            let d_hi = _mm_unpackhi_epi16(d, zero);

            // 32x8 uses >> 3 for final shift
            let c_lo = _mm_set_epi32(
                (tmp[y * 32 + x_base + 3] + 4) >> 3,
                (tmp[y * 32 + x_base + 2] + 4) >> 3,
                (tmp[y * 32 + x_base + 1] + 4) >> 3,
                (tmp[y * 32 + x_base + 0] + 4) >> 3,
            );
            let c_hi = _mm_set_epi32(
                (tmp[y * 32 + x_base + 7] + 4) >> 3,
                (tmp[y * 32 + x_base + 6] + 4) >> 3,
                (tmp[y * 32 + x_base + 5] + 4) >> 3,
                (tmp[y * 32 + x_base + 4] + 4) >> 3,
            );

            let sum_lo = _mm_add_epi32(d_lo, c_lo);
            let sum_hi = _mm_add_epi32(d_hi, c_hi);
            let clamped_lo = _mm_max_epi32(_mm_min_epi32(sum_lo, max_val), zero);
            let clamped_hi = _mm_max_epi32(_mm_min_epi32(sum_hi, max_val), zero);
            let packed = _mm_packus_epi32(clamped_lo, clamped_hi);

            storeu_128!(
                <&mut [u16; 8]>::try_from(&mut dst[dst_chunk_off..dst_chunk_off + 8]).unwrap(),
                packed
            );
        }
    }

    // Clear coefficients
    coeff[..256].fill(0);
}

/// FFI wrapper for 32x8 IDTX 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_identity_32x8_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_identity_identity_32x8_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// Rectangular ADST/FLIPADST 16bpc
// ============================================================================

/// Macro for 4x8 transform variants 16bpc
macro_rules! impl_4x8_transform_16bpc {
    ($name:ident, $row_fn:ident, $col_fn:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[arcane]
        fn $name(
            _token: Desktop64,
            dst: &mut [u16],
            dst_stride: usize,
            coeff: &mut [i16],
            _eob: i32,
            bitdepth_max: i32,
        ) {
            use crate::src::safe_simd::pixel_access::{
                loadi32, loadi64, loadu_128, storei32, storei64, storeu_128,
            };
            let mut dst = dst.flex_mut();
            let mut coeff = coeff.flex_mut();
            let stride_u16 = dst_stride / 2;
            let row_clip_min = i32::MIN;
            let row_clip_max = i32::MAX;
            let col_clip_min = i32::MIN;
            let col_clip_max = i32::MAX;
            let mut tmp = [0i32; 32];

            // is_rect2 = true for 4x8, so apply sqrt(2) scaling
            let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

            // Row transform (4 elements each, 8 rows)
            for y in 0..8 {
                let mut scratch = [0i32; 4];
                for x in 0..4 {
                    scratch[x] = rect2_scale(coeff[y + x * 8] as i32);
                }
                $row_fn(&mut scratch[..4], 1, row_clip_min, row_clip_max);
                for x in 0..4 {
                    tmp[y * 4 + x] = scratch[x];
                }
            }

            // Column transform
            for x in 0..4 {
                $col_fn(&mut tmp[x..], 4, col_clip_min, col_clip_max);
            }

            // Add to destination
            let zero = _mm_setzero_si128();
            let max_val = _mm_set1_epi32(bitdepth_max);

            for y in 0..8 {
                let dst_off = y * stride_u16;

                let d = loadi64!(zerocopy::AsBytes::as_bytes(&dst[dst_off..dst_off + 4]));
                let d32 = _mm_unpacklo_epi16(d, zero);

                let c = _mm_set_epi32(
                    (tmp[y * 4 + 3] + 8) >> 4,
                    (tmp[y * 4 + 2] + 8) >> 4,
                    (tmp[y * 4 + 1] + 8) >> 4,
                    (tmp[y * 4 + 0] + 8) >> 4,
                );

                let sum = _mm_add_epi32(d32, c);
                let clamped = _mm_max_epi32(_mm_min_epi32(sum, max_val), zero);
                let packed = _mm_packus_epi32(clamped, clamped);

                storei64!(
                    zerocopy::AsBytes::as_bytes_mut(&mut dst[dst_off..dst_off + 4]),
                    packed
                );
            }

            // Clear coefficients
            coeff[..32].fill(0);
        }
    };
}

/// Macro for 8x4 transform variants 16bpc
macro_rules! impl_8x4_transform_16bpc {
    ($name:ident, $row_fn:ident, $col_fn:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[arcane]
        fn $name(
            _token: Desktop64,
            dst: &mut [u16],
            dst_stride: usize,
            coeff: &mut [i16],
            _eob: i32,
            bitdepth_max: i32,
        ) {
            use crate::src::safe_simd::pixel_access::{
                loadi32, loadi64, loadu_128, storei32, storei64, storeu_128,
            };
            let mut dst = dst.flex_mut();
            let mut coeff = coeff.flex_mut();
            let stride_u16 = dst_stride / 2;
            let row_clip_min = i32::MIN;
            let row_clip_max = i32::MAX;
            let col_clip_min = i32::MIN;
            let col_clip_max = i32::MAX;
            let mut tmp = [0i32; 32];

            // is_rect2 = true for 8x4, so apply sqrt(2) scaling
            let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

            // Row transform (8 elements each, 4 rows)
            for y in 0..4 {
                let mut scratch = [0i32; 8];
                for x in 0..8 {
                    scratch[x] = rect2_scale(coeff[y + x * 4] as i32);
                }
                $row_fn(&mut scratch[..8], 1, row_clip_min, row_clip_max);
                for x in 0..8 {
                    tmp[y * 8 + x] = scratch[x];
                }
            }

            // Column transform
            for x in 0..8 {
                $col_fn(&mut tmp[x..], 8, col_clip_min, col_clip_max);
            }

            // Add to destination
            let zero = _mm_setzero_si128();
            let max_val = _mm_set1_epi32(bitdepth_max);

            for y in 0..4 {
                let dst_off = y * stride_u16;

                let d = loadu_128!(<&[u16; 8]>::try_from(&dst[dst_off..dst_off + 8]).unwrap());
                let d_lo = _mm_unpacklo_epi16(d, zero);
                let d_hi = _mm_unpackhi_epi16(d, zero);

                let c_lo = _mm_set_epi32(
                    (tmp[y * 8 + 3] + 8) >> 4,
                    (tmp[y * 8 + 2] + 8) >> 4,
                    (tmp[y * 8 + 1] + 8) >> 4,
                    (tmp[y * 8 + 0] + 8) >> 4,
                );
                let c_hi = _mm_set_epi32(
                    (tmp[y * 8 + 7] + 8) >> 4,
                    (tmp[y * 8 + 6] + 8) >> 4,
                    (tmp[y * 8 + 5] + 8) >> 4,
                    (tmp[y * 8 + 4] + 8) >> 4,
                );

                let sum_lo = _mm_add_epi32(d_lo, c_lo);
                let sum_hi = _mm_add_epi32(d_hi, c_hi);
                let clamped_lo = _mm_max_epi32(_mm_min_epi32(sum_lo, max_val), zero);
                let clamped_hi = _mm_max_epi32(_mm_min_epi32(sum_hi, max_val), zero);
                let packed = _mm_packus_epi32(clamped_lo, clamped_hi);

                storeu_128!(
                    <&mut [u16; 8]>::try_from(&mut dst[dst_off..dst_off + 8]).unwrap(),
                    packed
                );
            }

            // Clear coefficients
            coeff[..32].fill(0);
        }
    };
}

/// Macro for FFI wrapper 16bpc
macro_rules! impl_ffi_wrapper_16bpc {
    ($name:ident, $inner:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2")]
        #[cfg(feature = "asm")]
        pub unsafe extern "C" fn $name(
            dst_ptr: *mut DynPixel,
            dst_stride: isize,
            coeff: *mut DynCoef,
            eob: c_int,
            bitdepth_max: c_int,
            _coeff_len: u16,
            _dst: *const FFISafe<PicOffset>,
        ) {
            let _token = unsafe { Desktop64::forge_token_dangerously() };
            let stride = dst_stride as usize;

            let dst_slice = unsafe {
                std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
            };

            let coeff_slice =
                unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

            $inner(_token, dst_slice, stride, coeff_slice, eob, bitdepth_max);
        }
    };
}

// 4x8 ADST/FLIPADST variants 16bpc
impl_4x8_transform_16bpc!(
    inv_txfm_add_adst_dct_4x8_16bpc_avx2_inner,
    adst4_1d,
    dct8_1d
);
impl_4x8_transform_16bpc!(
    inv_txfm_add_dct_adst_4x8_16bpc_avx2_inner,
    dct4_1d,
    adst8_1d
);
impl_4x8_transform_16bpc!(
    inv_txfm_add_adst_adst_4x8_16bpc_avx2_inner,
    adst4_1d,
    adst8_1d
);
impl_4x8_transform_16bpc!(
    inv_txfm_add_flipadst_dct_4x8_16bpc_avx2_inner,
    flipadst4_1d,
    dct8_1d
);
impl_4x8_transform_16bpc!(
    inv_txfm_add_dct_flipadst_4x8_16bpc_avx2_inner,
    dct4_1d,
    flipadst8_1d
);
impl_4x8_transform_16bpc!(
    inv_txfm_add_flipadst_flipadst_4x8_16bpc_avx2_inner,
    flipadst4_1d,
    flipadst8_1d
);
impl_4x8_transform_16bpc!(
    inv_txfm_add_adst_flipadst_4x8_16bpc_avx2_inner,
    adst4_1d,
    flipadst8_1d
);
impl_4x8_transform_16bpc!(
    inv_txfm_add_flipadst_adst_4x8_16bpc_avx2_inner,
    flipadst4_1d,
    adst8_1d
);

impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_dct_4x8_16bpc_avx2,
    inv_txfm_add_adst_dct_4x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_adst_4x8_16bpc_avx2,
    inv_txfm_add_dct_adst_4x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_adst_4x8_16bpc_avx2,
    inv_txfm_add_adst_adst_4x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_dct_4x8_16bpc_avx2,
    inv_txfm_add_flipadst_dct_4x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_flipadst_4x8_16bpc_avx2,
    inv_txfm_add_dct_flipadst_4x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_flipadst_4x8_16bpc_avx2,
    inv_txfm_add_flipadst_flipadst_4x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_flipadst_4x8_16bpc_avx2,
    inv_txfm_add_adst_flipadst_4x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_adst_4x8_16bpc_avx2,
    inv_txfm_add_flipadst_adst_4x8_16bpc_avx2_inner
);

// 8x4 ADST/FLIPADST variants 16bpc
impl_8x4_transform_16bpc!(
    inv_txfm_add_adst_dct_8x4_16bpc_avx2_inner,
    adst8_1d,
    dct4_1d
);
impl_8x4_transform_16bpc!(
    inv_txfm_add_dct_adst_8x4_16bpc_avx2_inner,
    dct8_1d,
    adst4_1d
);
impl_8x4_transform_16bpc!(
    inv_txfm_add_adst_adst_8x4_16bpc_avx2_inner,
    adst8_1d,
    adst4_1d
);
impl_8x4_transform_16bpc!(
    inv_txfm_add_flipadst_dct_8x4_16bpc_avx2_inner,
    flipadst8_1d,
    dct4_1d
);
impl_8x4_transform_16bpc!(
    inv_txfm_add_dct_flipadst_8x4_16bpc_avx2_inner,
    dct8_1d,
    flipadst4_1d
);
impl_8x4_transform_16bpc!(
    inv_txfm_add_flipadst_flipadst_8x4_16bpc_avx2_inner,
    flipadst8_1d,
    flipadst4_1d
);
impl_8x4_transform_16bpc!(
    inv_txfm_add_adst_flipadst_8x4_16bpc_avx2_inner,
    adst8_1d,
    flipadst4_1d
);
impl_8x4_transform_16bpc!(
    inv_txfm_add_flipadst_adst_8x4_16bpc_avx2_inner,
    flipadst8_1d,
    adst4_1d
);

impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_dct_8x4_16bpc_avx2,
    inv_txfm_add_adst_dct_8x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_adst_8x4_16bpc_avx2,
    inv_txfm_add_dct_adst_8x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_adst_8x4_16bpc_avx2,
    inv_txfm_add_adst_adst_8x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_dct_8x4_16bpc_avx2,
    inv_txfm_add_flipadst_dct_8x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_flipadst_8x4_16bpc_avx2,
    inv_txfm_add_dct_flipadst_8x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_flipadst_8x4_16bpc_avx2,
    inv_txfm_add_flipadst_flipadst_8x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_flipadst_8x4_16bpc_avx2,
    inv_txfm_add_adst_flipadst_8x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_adst_8x4_16bpc_avx2,
    inv_txfm_add_flipadst_adst_8x4_16bpc_avx2_inner
);

/// Macro for 8x16 transform variants 16bpc
macro_rules! impl_8x16_transform_16bpc {
    ($name:ident, $row_fn:ident, $col_fn:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[arcane]
        fn $name(
            _token: Desktop64,
            dst: &mut [u16],
            dst_stride: usize,
            coeff: &mut [i16],
            _eob: i32,
            bitdepth_max: i32,
        ) {
            use crate::src::safe_simd::pixel_access::{
                loadi32, loadi64, loadu_128, storei32, storei64, storeu_128,
            };
            let mut dst = dst.flex_mut();
            let mut coeff = coeff.flex_mut();
            let stride_u16 = dst_stride / 2;
            let row_clip_min = i32::MIN;
            let row_clip_max = i32::MAX;
            let col_clip_min = i32::MIN;
            let col_clip_max = i32::MAX;
            let mut tmp = [0i32; 128];

            // is_rect2 = true for 8x16
            let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

            // Row transform (8 elements each, 16 rows)
            for y in 0..16 {
                let mut scratch = [0i32; 8];
                for x in 0..8 {
                    scratch[x] = rect2_scale(coeff[y + x * 16] as i32);
                }
                $row_fn(&mut scratch[..8], 1, row_clip_min, row_clip_max);
                for x in 0..8 {
                    tmp[y * 8 + x] = scratch[x];
                }
            }

            // Column transform
            for x in 0..8 {
                $col_fn(&mut tmp[x..], 8, col_clip_min, col_clip_max);
            }

            // Add to destination
            let zero = _mm_setzero_si128();
            let max_val = _mm_set1_epi32(bitdepth_max);

            for y in 0..16 {
                let dst_off = y * stride_u16;

                let d = loadu_128!(<&[u16; 8]>::try_from(&dst[dst_off..dst_off + 8]).unwrap());
                let d_lo = _mm_unpacklo_epi16(d, zero);
                let d_hi = _mm_unpackhi_epi16(d, zero);

                let c_lo = _mm_set_epi32(
                    (tmp[y * 8 + 3] + 8) >> 4,
                    (tmp[y * 8 + 2] + 8) >> 4,
                    (tmp[y * 8 + 1] + 8) >> 4,
                    (tmp[y * 8 + 0] + 8) >> 4,
                );
                let c_hi = _mm_set_epi32(
                    (tmp[y * 8 + 7] + 8) >> 4,
                    (tmp[y * 8 + 6] + 8) >> 4,
                    (tmp[y * 8 + 5] + 8) >> 4,
                    (tmp[y * 8 + 4] + 8) >> 4,
                );

                let sum_lo = _mm_add_epi32(d_lo, c_lo);
                let sum_hi = _mm_add_epi32(d_hi, c_hi);
                let clamped_lo = _mm_max_epi32(_mm_min_epi32(sum_lo, max_val), zero);
                let clamped_hi = _mm_max_epi32(_mm_min_epi32(sum_hi, max_val), zero);
                let packed = _mm_packus_epi32(clamped_lo, clamped_hi);

                storeu_128!(
                    <&mut [u16; 8]>::try_from(&mut dst[dst_off..dst_off + 8]).unwrap(),
                    packed
                );
            }

            // Clear coefficients
            coeff[..128].fill(0);
        }
    };
}

/// Macro for 16x8 transform variants 16bpc
macro_rules! impl_16x8_transform_16bpc {
    ($name:ident, $row_fn:ident, $col_fn:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[arcane]
        fn $name(
            _token: Desktop64,
            dst: &mut [u16],
            dst_stride: usize,
            coeff: &mut [i16],
            _eob: i32,
            bitdepth_max: i32,
        ) {
            use crate::src::safe_simd::pixel_access::{
                loadi32, loadi64, loadu_128, storei32, storei64, storeu_128,
            };
            let mut dst = dst.flex_mut();
            let mut coeff = coeff.flex_mut();
            let stride_u16 = dst_stride / 2;
            let row_clip_min = i32::MIN;
            let row_clip_max = i32::MAX;
            let col_clip_min = i32::MIN;
            let col_clip_max = i32::MAX;
            let mut tmp = [0i32; 128];

            // is_rect2 = true for 16x8
            let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

            // Row transform (16 elements each, 8 rows)
            for y in 0..8 {
                let mut scratch = [0i32; 16];
                for x in 0..16 {
                    scratch[x] = rect2_scale(coeff[y + x * 8] as i32);
                }
                $row_fn(&mut scratch[..16], 1, row_clip_min, row_clip_max);
                for x in 0..16 {
                    tmp[y * 16 + x] = scratch[x];
                }
            }

            // Column transform
            for x in 0..16 {
                $col_fn(&mut tmp[x..], 16, col_clip_min, col_clip_max);
            }

            // Add to destination
            let zero = _mm_setzero_si128();
            let max_val = _mm_set1_epi32(bitdepth_max);

            for y in 0..8 {
                let dst_off = y * stride_u16;

                for chunk in 0..2 {
                    let x_base = chunk * 8;
                    let dst_chunk_off = dst_off + x_base;

                    let d = loadu_128!(<&[u16; 8]>::try_from(
                        &dst[dst_chunk_off..dst_chunk_off + 8]
                    )
                    .unwrap());
                    let d_lo = _mm_unpacklo_epi16(d, zero);
                    let d_hi = _mm_unpackhi_epi16(d, zero);

                    let c_lo = _mm_set_epi32(
                        (tmp[y * 16 + x_base + 3] + 8) >> 4,
                        (tmp[y * 16 + x_base + 2] + 8) >> 4,
                        (tmp[y * 16 + x_base + 1] + 8) >> 4,
                        (tmp[y * 16 + x_base + 0] + 8) >> 4,
                    );
                    let c_hi = _mm_set_epi32(
                        (tmp[y * 16 + x_base + 7] + 8) >> 4,
                        (tmp[y * 16 + x_base + 6] + 8) >> 4,
                        (tmp[y * 16 + x_base + 5] + 8) >> 4,
                        (tmp[y * 16 + x_base + 4] + 8) >> 4,
                    );

                    let sum_lo = _mm_add_epi32(d_lo, c_lo);
                    let sum_hi = _mm_add_epi32(d_hi, c_hi);
                    let clamped_lo = _mm_max_epi32(_mm_min_epi32(sum_lo, max_val), zero);
                    let clamped_hi = _mm_max_epi32(_mm_min_epi32(sum_hi, max_val), zero);
                    let packed = _mm_packus_epi32(clamped_lo, clamped_hi);

                    storeu_128!(
                        <&mut [u16; 8]>::try_from(&mut dst[dst_chunk_off..dst_chunk_off + 8])
                            .unwrap(),
                        packed
                    );
                }
            }

            // Clear coefficients
            coeff[..128].fill(0);
        }
    };
}

// 8x16 ADST/FLIPADST variants 16bpc
impl_8x16_transform_16bpc!(
    inv_txfm_add_adst_dct_8x16_16bpc_avx2_inner,
    adst8_1d,
    dct16_1d
);
impl_8x16_transform_16bpc!(
    inv_txfm_add_dct_adst_8x16_16bpc_avx2_inner,
    dct8_1d,
    adst16_1d
);
impl_8x16_transform_16bpc!(
    inv_txfm_add_adst_adst_8x16_16bpc_avx2_inner,
    adst8_1d,
    adst16_1d
);
impl_8x16_transform_16bpc!(
    inv_txfm_add_flipadst_dct_8x16_16bpc_avx2_inner,
    flipadst8_1d,
    dct16_1d
);
impl_8x16_transform_16bpc!(
    inv_txfm_add_dct_flipadst_8x16_16bpc_avx2_inner,
    dct8_1d,
    flipadst16_1d
);
impl_8x16_transform_16bpc!(
    inv_txfm_add_flipadst_flipadst_8x16_16bpc_avx2_inner,
    flipadst8_1d,
    flipadst16_1d
);
impl_8x16_transform_16bpc!(
    inv_txfm_add_adst_flipadst_8x16_16bpc_avx2_inner,
    adst8_1d,
    flipadst16_1d
);
impl_8x16_transform_16bpc!(
    inv_txfm_add_flipadst_adst_8x16_16bpc_avx2_inner,
    flipadst8_1d,
    adst16_1d
);

impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_dct_8x16_16bpc_avx2,
    inv_txfm_add_adst_dct_8x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_adst_8x16_16bpc_avx2,
    inv_txfm_add_dct_adst_8x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_adst_8x16_16bpc_avx2,
    inv_txfm_add_adst_adst_8x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_dct_8x16_16bpc_avx2,
    inv_txfm_add_flipadst_dct_8x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_flipadst_8x16_16bpc_avx2,
    inv_txfm_add_dct_flipadst_8x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_flipadst_8x16_16bpc_avx2,
    inv_txfm_add_flipadst_flipadst_8x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_flipadst_8x16_16bpc_avx2,
    inv_txfm_add_adst_flipadst_8x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_adst_8x16_16bpc_avx2,
    inv_txfm_add_flipadst_adst_8x16_16bpc_avx2_inner
);

// 16x8 ADST/FLIPADST variants 16bpc
impl_16x8_transform_16bpc!(
    inv_txfm_add_adst_dct_16x8_16bpc_avx2_inner,
    adst16_1d,
    dct8_1d
);
impl_16x8_transform_16bpc!(
    inv_txfm_add_dct_adst_16x8_16bpc_avx2_inner,
    dct16_1d,
    adst8_1d
);
impl_16x8_transform_16bpc!(
    inv_txfm_add_adst_adst_16x8_16bpc_avx2_inner,
    adst16_1d,
    adst8_1d
);
impl_16x8_transform_16bpc!(
    inv_txfm_add_flipadst_dct_16x8_16bpc_avx2_inner,
    flipadst16_1d,
    dct8_1d
);
impl_16x8_transform_16bpc!(
    inv_txfm_add_dct_flipadst_16x8_16bpc_avx2_inner,
    dct16_1d,
    flipadst8_1d
);
impl_16x8_transform_16bpc!(
    inv_txfm_add_flipadst_flipadst_16x8_16bpc_avx2_inner,
    flipadst16_1d,
    flipadst8_1d
);
impl_16x8_transform_16bpc!(
    inv_txfm_add_adst_flipadst_16x8_16bpc_avx2_inner,
    adst16_1d,
    flipadst8_1d
);
impl_16x8_transform_16bpc!(
    inv_txfm_add_flipadst_adst_16x8_16bpc_avx2_inner,
    flipadst16_1d,
    adst8_1d
);

impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_dct_16x8_16bpc_avx2,
    inv_txfm_add_adst_dct_16x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_adst_16x8_16bpc_avx2,
    inv_txfm_add_dct_adst_16x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_adst_16x8_16bpc_avx2,
    inv_txfm_add_adst_adst_16x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_dct_16x8_16bpc_avx2,
    inv_txfm_add_flipadst_dct_16x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_flipadst_16x8_16bpc_avx2,
    inv_txfm_add_dct_flipadst_16x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_flipadst_16x8_16bpc_avx2,
    inv_txfm_add_flipadst_flipadst_16x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_flipadst_16x8_16bpc_avx2,
    inv_txfm_add_adst_flipadst_16x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_adst_16x8_16bpc_avx2,
    inv_txfm_add_flipadst_adst_16x8_16bpc_avx2_inner
);

/// Macro for 4x16 transform variants 16bpc
macro_rules! impl_4x16_transform_16bpc {
    ($name:ident, $row_fn:ident, $col_fn:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[arcane]
        fn $name(
            _token: Desktop64,
            dst: &mut [u16],
            dst_stride: usize,
            coeff: &mut [i16],
            _eob: i32,
            bitdepth_max: i32,
        ) {
            use crate::src::safe_simd::pixel_access::{
                loadi32, loadi64, loadu_128, storei32, storei64, storeu_128,
            };
            let mut dst = dst.flex_mut();
            let mut coeff = coeff.flex_mut();
            let stride_u16 = dst_stride / 2;
            let row_clip_min = i32::MIN;
            let row_clip_max = i32::MAX;
            let col_clip_min = i32::MIN;
            let col_clip_max = i32::MAX;
            let mut tmp = [0i32; 64];

            // is_rect2 = false for 4x16 (aspect ratio 4:1), no rect2_scale

            // Row transform (4 elements each, 16 rows)
            for y in 0..16 {
                let mut scratch = [0i32; 4];
                for x in 0..4 {
                    scratch[x] = coeff[y + x * 16] as i32;
                }
                $row_fn(&mut scratch[..4], 1, row_clip_min, row_clip_max);
                for x in 0..4 {
                    tmp[y * 4 + x] = scratch[x];
                }
            }

            // Column transform
            for x in 0..4 {
                $col_fn(&mut tmp[x..], 4, col_clip_min, col_clip_max);
            }

            // Add to destination
            let zero = _mm_setzero_si128();
            let max_val = _mm_set1_epi32(bitdepth_max);

            for y in 0..16 {
                let dst_off = y * stride_u16;

                let d = loadi64!(zerocopy::AsBytes::as_bytes(&dst[dst_off..dst_off + 4]));
                let d32 = _mm_unpacklo_epi16(d, zero);

                let c = _mm_set_epi32(
                    (tmp[y * 4 + 3] + 8) >> 4,
                    (tmp[y * 4 + 2] + 8) >> 4,
                    (tmp[y * 4 + 1] + 8) >> 4,
                    (tmp[y * 4 + 0] + 8) >> 4,
                );

                let sum = _mm_add_epi32(d32, c);
                let clamped = _mm_max_epi32(_mm_min_epi32(sum, max_val), zero);
                let packed = _mm_packus_epi32(clamped, clamped);

                storei64!(
                    zerocopy::AsBytes::as_bytes_mut(&mut dst[dst_off..dst_off + 4]),
                    packed
                );
            }

            // Clear coefficients
            coeff[..64].fill(0);
        }
    };
}

/// Macro for 16x4 transform variants 16bpc
macro_rules! impl_16x4_transform_16bpc {
    ($name:ident, $row_fn:ident, $col_fn:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[arcane]
        fn $name(
            _token: Desktop64,
            dst: &mut [u16],
            dst_stride: usize,
            coeff: &mut [i16],
            _eob: i32,
            bitdepth_max: i32,
        ) {
            use crate::src::safe_simd::pixel_access::{
                loadi32, loadi64, loadu_128, storei32, storei64, storeu_128,
            };
            let mut dst = dst.flex_mut();
            let mut coeff = coeff.flex_mut();
            let stride_u16 = dst_stride / 2;
            let row_clip_min = i32::MIN;
            let row_clip_max = i32::MAX;
            let col_clip_min = i32::MIN;
            let col_clip_max = i32::MAX;
            let mut tmp = [0i32; 64];

            // is_rect2 = false for 16x4 (aspect ratio 4:1), no rect2_scale

            // Row transform (16 elements each, 4 rows)
            for y in 0..4 {
                let mut scratch = [0i32; 16];
                for x in 0..16 {
                    scratch[x] = coeff[y + x * 4] as i32;
                }
                $row_fn(&mut scratch[..16], 1, row_clip_min, row_clip_max);
                for x in 0..16 {
                    tmp[y * 16 + x] = scratch[x];
                }
            }

            // Column transform
            for x in 0..16 {
                $col_fn(&mut tmp[x..], 16, col_clip_min, col_clip_max);
            }

            // Add to destination
            let zero = _mm_setzero_si128();
            let max_val = _mm_set1_epi32(bitdepth_max);

            for y in 0..4 {
                let dst_off = y * stride_u16;

                for chunk in 0..2 {
                    let x_base = chunk * 8;
                    let dst_chunk_off = dst_off + x_base;

                    let d = loadu_128!(<&[u16; 8]>::try_from(
                        &dst[dst_chunk_off..dst_chunk_off + 8]
                    )
                    .unwrap());
                    let d_lo = _mm_unpacklo_epi16(d, zero);
                    let d_hi = _mm_unpackhi_epi16(d, zero);

                    let c_lo = _mm_set_epi32(
                        (tmp[y * 16 + x_base + 3] + 8) >> 4,
                        (tmp[y * 16 + x_base + 2] + 8) >> 4,
                        (tmp[y * 16 + x_base + 1] + 8) >> 4,
                        (tmp[y * 16 + x_base + 0] + 8) >> 4,
                    );
                    let c_hi = _mm_set_epi32(
                        (tmp[y * 16 + x_base + 7] + 8) >> 4,
                        (tmp[y * 16 + x_base + 6] + 8) >> 4,
                        (tmp[y * 16 + x_base + 5] + 8) >> 4,
                        (tmp[y * 16 + x_base + 4] + 8) >> 4,
                    );

                    let sum_lo = _mm_add_epi32(d_lo, c_lo);
                    let sum_hi = _mm_add_epi32(d_hi, c_hi);
                    let clamped_lo = _mm_max_epi32(_mm_min_epi32(sum_lo, max_val), zero);
                    let clamped_hi = _mm_max_epi32(_mm_min_epi32(sum_hi, max_val), zero);
                    let packed = _mm_packus_epi32(clamped_lo, clamped_hi);

                    storeu_128!(
                        <&mut [u16; 8]>::try_from(&mut dst[dst_chunk_off..dst_chunk_off + 8])
                            .unwrap(),
                        packed
                    );
                }
            }

            // Clear coefficients
            coeff[..64].fill(0);
        }
    };
}

// 4x16 ADST/FLIPADST variants 16bpc
impl_4x16_transform_16bpc!(
    inv_txfm_add_adst_dct_4x16_16bpc_avx2_inner,
    adst4_1d,
    dct16_1d
);
impl_4x16_transform_16bpc!(
    inv_txfm_add_dct_adst_4x16_16bpc_avx2_inner,
    dct4_1d,
    adst16_1d
);
impl_4x16_transform_16bpc!(
    inv_txfm_add_adst_adst_4x16_16bpc_avx2_inner,
    adst4_1d,
    adst16_1d
);
impl_4x16_transform_16bpc!(
    inv_txfm_add_flipadst_dct_4x16_16bpc_avx2_inner,
    flipadst4_1d,
    dct16_1d
);
impl_4x16_transform_16bpc!(
    inv_txfm_add_dct_flipadst_4x16_16bpc_avx2_inner,
    dct4_1d,
    flipadst16_1d
);
impl_4x16_transform_16bpc!(
    inv_txfm_add_flipadst_flipadst_4x16_16bpc_avx2_inner,
    flipadst4_1d,
    flipadst16_1d
);
impl_4x16_transform_16bpc!(
    inv_txfm_add_adst_flipadst_4x16_16bpc_avx2_inner,
    adst4_1d,
    flipadst16_1d
);
impl_4x16_transform_16bpc!(
    inv_txfm_add_flipadst_adst_4x16_16bpc_avx2_inner,
    flipadst4_1d,
    adst16_1d
);

impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_dct_4x16_16bpc_avx2,
    inv_txfm_add_adst_dct_4x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_adst_4x16_16bpc_avx2,
    inv_txfm_add_dct_adst_4x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_adst_4x16_16bpc_avx2,
    inv_txfm_add_adst_adst_4x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_dct_4x16_16bpc_avx2,
    inv_txfm_add_flipadst_dct_4x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_flipadst_4x16_16bpc_avx2,
    inv_txfm_add_dct_flipadst_4x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_flipadst_4x16_16bpc_avx2,
    inv_txfm_add_flipadst_flipadst_4x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_flipadst_4x16_16bpc_avx2,
    inv_txfm_add_adst_flipadst_4x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_adst_4x16_16bpc_avx2,
    inv_txfm_add_flipadst_adst_4x16_16bpc_avx2_inner
);

// 16x4 ADST/FLIPADST variants 16bpc
impl_16x4_transform_16bpc!(
    inv_txfm_add_adst_dct_16x4_16bpc_avx2_inner,
    adst16_1d,
    dct4_1d
);
impl_16x4_transform_16bpc!(
    inv_txfm_add_dct_adst_16x4_16bpc_avx2_inner,
    dct16_1d,
    adst4_1d
);
impl_16x4_transform_16bpc!(
    inv_txfm_add_adst_adst_16x4_16bpc_avx2_inner,
    adst16_1d,
    adst4_1d
);
impl_16x4_transform_16bpc!(
    inv_txfm_add_flipadst_dct_16x4_16bpc_avx2_inner,
    flipadst16_1d,
    dct4_1d
);
impl_16x4_transform_16bpc!(
    inv_txfm_add_dct_flipadst_16x4_16bpc_avx2_inner,
    dct16_1d,
    flipadst4_1d
);
impl_16x4_transform_16bpc!(
    inv_txfm_add_flipadst_flipadst_16x4_16bpc_avx2_inner,
    flipadst16_1d,
    flipadst4_1d
);
impl_16x4_transform_16bpc!(
    inv_txfm_add_adst_flipadst_16x4_16bpc_avx2_inner,
    adst16_1d,
    flipadst4_1d
);
impl_16x4_transform_16bpc!(
    inv_txfm_add_flipadst_adst_16x4_16bpc_avx2_inner,
    flipadst16_1d,
    adst4_1d
);

impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_dct_16x4_16bpc_avx2,
    inv_txfm_add_adst_dct_16x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_adst_16x4_16bpc_avx2,
    inv_txfm_add_dct_adst_16x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_adst_16x4_16bpc_avx2,
    inv_txfm_add_adst_adst_16x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_dct_16x4_16bpc_avx2,
    inv_txfm_add_flipadst_dct_16x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_flipadst_16x4_16bpc_avx2,
    inv_txfm_add_dct_flipadst_16x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_flipadst_16x4_16bpc_avx2,
    inv_txfm_add_flipadst_flipadst_16x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_flipadst_16x4_16bpc_avx2,
    inv_txfm_add_adst_flipadst_16x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_adst_16x4_16bpc_avx2,
    inv_txfm_add_flipadst_adst_16x4_16bpc_avx2_inner
);

// ============================================================================
// Hybrid identity transforms for 16bpc (H_DCT, V_DCT, H_ADST, V_ADST, etc.)
// ============================================================================

// 4x8 hybrid identity transforms 16bpc
impl_4x8_transform_16bpc!(
    inv_txfm_add_identity_dct_4x8_16bpc_avx2_inner,
    identity4_1d,
    dct8_1d
);
impl_4x8_transform_16bpc!(
    inv_txfm_add_dct_identity_4x8_16bpc_avx2_inner,
    dct4_1d,
    identity8_1d
);
impl_4x8_transform_16bpc!(
    inv_txfm_add_identity_adst_4x8_16bpc_avx2_inner,
    identity4_1d,
    adst8_1d
);
impl_4x8_transform_16bpc!(
    inv_txfm_add_adst_identity_4x8_16bpc_avx2_inner,
    adst4_1d,
    identity8_1d
);
impl_4x8_transform_16bpc!(
    inv_txfm_add_identity_flipadst_4x8_16bpc_avx2_inner,
    identity4_1d,
    flipadst8_1d
);
impl_4x8_transform_16bpc!(
    inv_txfm_add_flipadst_identity_4x8_16bpc_avx2_inner,
    flipadst4_1d,
    identity8_1d
);

impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_dct_4x8_16bpc_avx2,
    inv_txfm_add_identity_dct_4x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_identity_4x8_16bpc_avx2,
    inv_txfm_add_dct_identity_4x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_adst_4x8_16bpc_avx2,
    inv_txfm_add_identity_adst_4x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_identity_4x8_16bpc_avx2,
    inv_txfm_add_adst_identity_4x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_flipadst_4x8_16bpc_avx2,
    inv_txfm_add_identity_flipadst_4x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_identity_4x8_16bpc_avx2,
    inv_txfm_add_flipadst_identity_4x8_16bpc_avx2_inner
);

// 8x4 hybrid identity transforms 16bpc
impl_8x4_transform_16bpc!(
    inv_txfm_add_identity_dct_8x4_16bpc_avx2_inner,
    identity8_1d,
    dct4_1d
);
impl_8x4_transform_16bpc!(
    inv_txfm_add_dct_identity_8x4_16bpc_avx2_inner,
    dct8_1d,
    identity4_1d
);
impl_8x4_transform_16bpc!(
    inv_txfm_add_identity_adst_8x4_16bpc_avx2_inner,
    identity8_1d,
    adst4_1d
);
impl_8x4_transform_16bpc!(
    inv_txfm_add_adst_identity_8x4_16bpc_avx2_inner,
    adst8_1d,
    identity4_1d
);
impl_8x4_transform_16bpc!(
    inv_txfm_add_identity_flipadst_8x4_16bpc_avx2_inner,
    identity8_1d,
    flipadst4_1d
);
impl_8x4_transform_16bpc!(
    inv_txfm_add_flipadst_identity_8x4_16bpc_avx2_inner,
    flipadst8_1d,
    identity4_1d
);

impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_dct_8x4_16bpc_avx2,
    inv_txfm_add_identity_dct_8x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_identity_8x4_16bpc_avx2,
    inv_txfm_add_dct_identity_8x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_adst_8x4_16bpc_avx2,
    inv_txfm_add_identity_adst_8x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_identity_8x4_16bpc_avx2,
    inv_txfm_add_adst_identity_8x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_flipadst_8x4_16bpc_avx2,
    inv_txfm_add_identity_flipadst_8x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_identity_8x4_16bpc_avx2,
    inv_txfm_add_flipadst_identity_8x4_16bpc_avx2_inner
);

// 8x16 hybrid identity transforms 16bpc
impl_8x16_transform_16bpc!(
    inv_txfm_add_identity_dct_8x16_16bpc_avx2_inner,
    identity8_1d,
    dct16_1d
);
impl_8x16_transform_16bpc!(
    inv_txfm_add_dct_identity_8x16_16bpc_avx2_inner,
    dct8_1d,
    identity16_1d
);
impl_8x16_transform_16bpc!(
    inv_txfm_add_identity_adst_8x16_16bpc_avx2_inner,
    identity8_1d,
    adst16_1d
);
impl_8x16_transform_16bpc!(
    inv_txfm_add_adst_identity_8x16_16bpc_avx2_inner,
    adst8_1d,
    identity16_1d
);
impl_8x16_transform_16bpc!(
    inv_txfm_add_identity_flipadst_8x16_16bpc_avx2_inner,
    identity8_1d,
    flipadst16_1d
);
impl_8x16_transform_16bpc!(
    inv_txfm_add_flipadst_identity_8x16_16bpc_avx2_inner,
    flipadst8_1d,
    identity16_1d
);

impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_dct_8x16_16bpc_avx2,
    inv_txfm_add_identity_dct_8x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_identity_8x16_16bpc_avx2,
    inv_txfm_add_dct_identity_8x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_adst_8x16_16bpc_avx2,
    inv_txfm_add_identity_adst_8x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_identity_8x16_16bpc_avx2,
    inv_txfm_add_adst_identity_8x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_flipadst_8x16_16bpc_avx2,
    inv_txfm_add_identity_flipadst_8x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_identity_8x16_16bpc_avx2,
    inv_txfm_add_flipadst_identity_8x16_16bpc_avx2_inner
);

// 16x8 hybrid identity transforms 16bpc
impl_16x8_transform_16bpc!(
    inv_txfm_add_identity_dct_16x8_16bpc_avx2_inner,
    identity16_1d,
    dct8_1d
);
impl_16x8_transform_16bpc!(
    inv_txfm_add_dct_identity_16x8_16bpc_avx2_inner,
    dct16_1d,
    identity8_1d
);
impl_16x8_transform_16bpc!(
    inv_txfm_add_identity_adst_16x8_16bpc_avx2_inner,
    identity16_1d,
    adst8_1d
);
impl_16x8_transform_16bpc!(
    inv_txfm_add_adst_identity_16x8_16bpc_avx2_inner,
    adst16_1d,
    identity8_1d
);
impl_16x8_transform_16bpc!(
    inv_txfm_add_identity_flipadst_16x8_16bpc_avx2_inner,
    identity16_1d,
    flipadst8_1d
);
impl_16x8_transform_16bpc!(
    inv_txfm_add_flipadst_identity_16x8_16bpc_avx2_inner,
    flipadst16_1d,
    identity8_1d
);

impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_dct_16x8_16bpc_avx2,
    inv_txfm_add_identity_dct_16x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_identity_16x8_16bpc_avx2,
    inv_txfm_add_dct_identity_16x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_adst_16x8_16bpc_avx2,
    inv_txfm_add_identity_adst_16x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_identity_16x8_16bpc_avx2,
    inv_txfm_add_adst_identity_16x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_flipadst_16x8_16bpc_avx2,
    inv_txfm_add_identity_flipadst_16x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_identity_16x8_16bpc_avx2,
    inv_txfm_add_flipadst_identity_16x8_16bpc_avx2_inner
);

// 4x16 hybrid identity transforms 16bpc
impl_4x16_transform_16bpc!(
    inv_txfm_add_identity_dct_4x16_16bpc_avx2_inner,
    identity4_1d,
    dct16_1d
);
impl_4x16_transform_16bpc!(
    inv_txfm_add_dct_identity_4x16_16bpc_avx2_inner,
    dct4_1d,
    identity16_1d
);
impl_4x16_transform_16bpc!(
    inv_txfm_add_identity_adst_4x16_16bpc_avx2_inner,
    identity4_1d,
    adst16_1d
);
impl_4x16_transform_16bpc!(
    inv_txfm_add_adst_identity_4x16_16bpc_avx2_inner,
    adst4_1d,
    identity16_1d
);
impl_4x16_transform_16bpc!(
    inv_txfm_add_identity_flipadst_4x16_16bpc_avx2_inner,
    identity4_1d,
    flipadst16_1d
);
impl_4x16_transform_16bpc!(
    inv_txfm_add_flipadst_identity_4x16_16bpc_avx2_inner,
    flipadst4_1d,
    identity16_1d
);

impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_dct_4x16_16bpc_avx2,
    inv_txfm_add_identity_dct_4x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_identity_4x16_16bpc_avx2,
    inv_txfm_add_dct_identity_4x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_adst_4x16_16bpc_avx2,
    inv_txfm_add_identity_adst_4x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_identity_4x16_16bpc_avx2,
    inv_txfm_add_adst_identity_4x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_flipadst_4x16_16bpc_avx2,
    inv_txfm_add_identity_flipadst_4x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_identity_4x16_16bpc_avx2,
    inv_txfm_add_flipadst_identity_4x16_16bpc_avx2_inner
);

// 16x4 hybrid identity transforms 16bpc
impl_16x4_transform_16bpc!(
    inv_txfm_add_identity_dct_16x4_16bpc_avx2_inner,
    identity16_1d,
    dct4_1d
);
impl_16x4_transform_16bpc!(
    inv_txfm_add_dct_identity_16x4_16bpc_avx2_inner,
    dct16_1d,
    identity4_1d
);
impl_16x4_transform_16bpc!(
    inv_txfm_add_identity_adst_16x4_16bpc_avx2_inner,
    identity16_1d,
    adst4_1d
);
impl_16x4_transform_16bpc!(
    inv_txfm_add_adst_identity_16x4_16bpc_avx2_inner,
    adst16_1d,
    identity4_1d
);
impl_16x4_transform_16bpc!(
    inv_txfm_add_identity_flipadst_16x4_16bpc_avx2_inner,
    identity16_1d,
    flipadst4_1d
);
impl_16x4_transform_16bpc!(
    inv_txfm_add_flipadst_identity_16x4_16bpc_avx2_inner,
    flipadst16_1d,
    identity4_1d
);

impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_dct_16x4_16bpc_avx2,
    inv_txfm_add_identity_dct_16x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_identity_16x4_16bpc_avx2,
    inv_txfm_add_dct_identity_16x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_adst_16x4_16bpc_avx2,
    inv_txfm_add_identity_adst_16x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_identity_16x4_16bpc_avx2,
    inv_txfm_add_adst_identity_16x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_flipadst_16x4_16bpc_avx2,
    inv_txfm_add_identity_flipadst_16x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_identity_16x4_16bpc_avx2,
    inv_txfm_add_flipadst_identity_16x4_16bpc_avx2_inner
);

// ============================================================================
// Square hybrid identity transforms for 16bpc
// ============================================================================

/// Macro for 8x8 transform variants 16bpc
macro_rules! impl_8x8_transform_16bpc {
    ($name:ident, $row_fn:ident, $col_fn:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[arcane]
        fn $name(
            _token: Desktop64,
            dst: &mut [u16],
            dst_stride: usize,
            coeff: &mut [i16],
            _eob: i32,
            bitdepth_max: i32,
        ) {
            use crate::src::safe_simd::pixel_access::{
                loadi32, loadi64, loadu_128, storei32, storei64, storeu_128,
            };
            let mut dst = dst.flex_mut();
            let mut coeff = coeff.flex_mut();
            let stride_u16 = dst_stride / 2;
            let row_clip_min = i32::MIN;
            let row_clip_max = i32::MAX;
            let col_clip_min = i32::MIN;
            let col_clip_max = i32::MAX;
            let mut tmp = [0i32; 64];

            // No rect2_scale for square transforms

            // Row transform (8 elements each, 8 rows)
            for y in 0..8 {
                let mut scratch = [0i32; 8];
                for x in 0..8 {
                    scratch[x] = coeff[y + x * 8] as i32;
                }
                $row_fn(&mut scratch[..8], 1, row_clip_min, row_clip_max);
                for x in 0..8 {
                    tmp[y * 8 + x] = scratch[x];
                }
            }

            // Column transform
            for x in 0..8 {
                $col_fn(&mut tmp[x..], 8, col_clip_min, col_clip_max);
            }

            // Add to destination
            let zero = _mm_setzero_si128();
            let max_val = _mm_set1_epi32(bitdepth_max);

            for y in 0..8 {
                let dst_off = y * stride_u16;

                let d = loadu_128!(<&[u16; 8]>::try_from(&dst[dst_off..dst_off + 8]).unwrap());
                let d_lo = _mm_unpacklo_epi16(d, zero);
                let d_hi = _mm_unpackhi_epi16(d, zero);

                let c_lo = _mm_set_epi32(
                    (tmp[y * 8 + 3] + 8) >> 4,
                    (tmp[y * 8 + 2] + 8) >> 4,
                    (tmp[y * 8 + 1] + 8) >> 4,
                    (tmp[y * 8 + 0] + 8) >> 4,
                );
                let c_hi = _mm_set_epi32(
                    (tmp[y * 8 + 7] + 8) >> 4,
                    (tmp[y * 8 + 6] + 8) >> 4,
                    (tmp[y * 8 + 5] + 8) >> 4,
                    (tmp[y * 8 + 4] + 8) >> 4,
                );

                let sum_lo = _mm_add_epi32(d_lo, c_lo);
                let sum_hi = _mm_add_epi32(d_hi, c_hi);
                let clamped_lo = _mm_max_epi32(_mm_min_epi32(sum_lo, max_val), zero);
                let clamped_hi = _mm_max_epi32(_mm_min_epi32(sum_hi, max_val), zero);
                let packed = _mm_packus_epi32(clamped_lo, clamped_hi);

                storeu_128!(
                    <&mut [u16; 8]>::try_from(&mut dst[dst_off..dst_off + 8]).unwrap(),
                    packed
                );
            }

            // Clear coefficients
            coeff[..64].fill(0);
        }
    };
}

// 8x8 hybrid identity transforms 16bpc
impl_8x8_transform_16bpc!(
    inv_txfm_add_identity_dct_8x8_16bpc_avx2_inner,
    identity8_1d,
    dct8_1d
);
impl_8x8_transform_16bpc!(
    inv_txfm_add_dct_identity_8x8_16bpc_avx2_inner,
    dct8_1d,
    identity8_1d
);
impl_8x8_transform_16bpc!(
    inv_txfm_add_identity_adst_8x8_16bpc_avx2_inner,
    identity8_1d,
    adst8_1d
);
impl_8x8_transform_16bpc!(
    inv_txfm_add_adst_identity_8x8_16bpc_avx2_inner,
    adst8_1d,
    identity8_1d
);
impl_8x8_transform_16bpc!(
    inv_txfm_add_identity_flipadst_8x8_16bpc_avx2_inner,
    identity8_1d,
    flipadst8_1d
);
impl_8x8_transform_16bpc!(
    inv_txfm_add_flipadst_identity_8x8_16bpc_avx2_inner,
    flipadst8_1d,
    identity8_1d
);

impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_dct_8x8_16bpc_avx2,
    inv_txfm_add_identity_dct_8x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_identity_8x8_16bpc_avx2,
    inv_txfm_add_dct_identity_8x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_adst_8x8_16bpc_avx2,
    inv_txfm_add_identity_adst_8x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_identity_8x8_16bpc_avx2,
    inv_txfm_add_adst_identity_8x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_flipadst_8x8_16bpc_avx2,
    inv_txfm_add_identity_flipadst_8x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_identity_8x8_16bpc_avx2,
    inv_txfm_add_flipadst_identity_8x8_16bpc_avx2_inner
);

/// Macro for 4x4 transform variants 16bpc
macro_rules! impl_4x4_transform_16bpc {
    ($name:ident, $row_fn:ident, $col_fn:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[arcane]
        fn $name(
            _token: Desktop64,
            dst: &mut [u16],
            dst_stride: usize,
            coeff: &mut [i16],
            _eob: i32,
            bitdepth_max: i32,
        ) {
            use crate::src::safe_simd::pixel_access::{
                loadi32, loadi64, loadu_128, storei32, storei64, storeu_128,
            };
            let mut dst = dst.flex_mut();
            let mut coeff = coeff.flex_mut();
            let stride_u16 = dst_stride / 2;
            let row_clip_min = i32::MIN;
            let row_clip_max = i32::MAX;
            let col_clip_min = i32::MIN;
            let col_clip_max = i32::MAX;
            let mut tmp = [0i32; 16];

            // Row transform (4 elements each, 4 rows)
            for y in 0..4 {
                let mut scratch = [0i32; 4];
                for x in 0..4 {
                    scratch[x] = coeff[y + x * 4] as i32;
                }
                $row_fn(&mut scratch[..4], 1, row_clip_min, row_clip_max);
                for x in 0..4 {
                    tmp[y * 4 + x] = scratch[x];
                }
            }

            // Column transform
            for x in 0..4 {
                $col_fn(&mut tmp[x..], 4, col_clip_min, col_clip_max);
            }

            // Add to destination
            let zero = _mm_setzero_si128();
            let max_val = _mm_set1_epi32(bitdepth_max);

            for y in 0..4 {
                let dst_off = y * stride_u16;

                let d = loadi64!(zerocopy::AsBytes::as_bytes(&dst[dst_off..dst_off + 4]));
                let d32 = _mm_unpacklo_epi16(d, zero);

                let c = _mm_set_epi32(
                    (tmp[y * 4 + 3] + 8) >> 4,
                    (tmp[y * 4 + 2] + 8) >> 4,
                    (tmp[y * 4 + 1] + 8) >> 4,
                    (tmp[y * 4 + 0] + 8) >> 4,
                );

                let sum = _mm_add_epi32(d32, c);
                let clamped = _mm_max_epi32(_mm_min_epi32(sum, max_val), zero);
                let packed = _mm_packus_epi32(clamped, clamped);

                storei64!(
                    zerocopy::AsBytes::as_bytes_mut(&mut dst[dst_off..dst_off + 4]),
                    packed
                );
            }

            // Clear coefficients
            coeff[..16].fill(0);
        }
    };
}

// 4x4 hybrid identity transforms 16bpc
impl_4x4_transform_16bpc!(
    inv_txfm_add_identity_dct_4x4_16bpc_avx2_inner,
    identity4_1d,
    dct4_1d
);
impl_4x4_transform_16bpc!(
    inv_txfm_add_dct_identity_4x4_16bpc_avx2_inner,
    dct4_1d,
    identity4_1d
);
impl_4x4_transform_16bpc!(
    inv_txfm_add_identity_adst_4x4_16bpc_avx2_inner,
    identity4_1d,
    adst4_1d
);
impl_4x4_transform_16bpc!(
    inv_txfm_add_adst_identity_4x4_16bpc_avx2_inner,
    adst4_1d,
    identity4_1d
);
impl_4x4_transform_16bpc!(
    inv_txfm_add_identity_flipadst_4x4_16bpc_avx2_inner,
    identity4_1d,
    flipadst4_1d
);
impl_4x4_transform_16bpc!(
    inv_txfm_add_flipadst_identity_4x4_16bpc_avx2_inner,
    flipadst4_1d,
    identity4_1d
);

impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_dct_4x4_16bpc_avx2,
    inv_txfm_add_identity_dct_4x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_identity_4x4_16bpc_avx2,
    inv_txfm_add_dct_identity_4x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_adst_4x4_16bpc_avx2,
    inv_txfm_add_identity_adst_4x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_identity_4x4_16bpc_avx2,
    inv_txfm_add_adst_identity_4x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_flipadst_4x4_16bpc_avx2,
    inv_txfm_add_identity_flipadst_4x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_identity_4x4_16bpc_avx2,
    inv_txfm_add_flipadst_identity_4x4_16bpc_avx2_inner
);

/// Macro for 16x16 transform variants 16bpc
macro_rules! impl_16x16_transform_16bpc {
    ($name:ident, $row_fn:ident, $col_fn:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[arcane]
        fn $name(
            _token: Desktop64,
            dst: &mut [u16],
            dst_stride: usize,
            coeff: &mut [i16],
            _eob: i32,
            bitdepth_max: i32,
        ) {
            use crate::src::safe_simd::pixel_access::{
                loadi32, loadi64, loadu_128, storei32, storei64, storeu_128,
            };
            let mut dst = dst.flex_mut();
            let mut coeff = coeff.flex_mut();
            let stride_u16 = dst_stride / 2;
            let row_clip_min = i32::MIN;
            let row_clip_max = i32::MAX;
            let col_clip_min = i32::MIN;
            let col_clip_max = i32::MAX;
            let mut tmp = [0i32; 256];

            // Row transform (16 elements each, 16 rows)
            for y in 0..16 {
                let mut scratch = [0i32; 16];
                for x in 0..16 {
                    scratch[x] = coeff[y + x * 16] as i32;
                }
                $row_fn(&mut scratch[..16], 1, row_clip_min, row_clip_max);
                for x in 0..16 {
                    tmp[y * 16 + x] = scratch[x];
                }
            }

            // Column transform
            for x in 0..16 {
                $col_fn(&mut tmp[x..], 16, col_clip_min, col_clip_max);
            }

            // Add to destination
            let zero = _mm_setzero_si128();
            let max_val = _mm_set1_epi32(bitdepth_max);

            for y in 0..16 {
                let dst_off = y * stride_u16;

                for chunk in 0..2 {
                    let x_base = chunk * 8;
                    let dst_chunk_off = dst_off + x_base;

                    let d = loadu_128!(<&[u16; 8]>::try_from(
                        &dst[dst_chunk_off..dst_chunk_off + 8]
                    )
                    .unwrap());
                    let d_lo = _mm_unpacklo_epi16(d, zero);
                    let d_hi = _mm_unpackhi_epi16(d, zero);

                    let c_lo = _mm_set_epi32(
                        (tmp[y * 16 + x_base + 3] + 8) >> 4,
                        (tmp[y * 16 + x_base + 2] + 8) >> 4,
                        (tmp[y * 16 + x_base + 1] + 8) >> 4,
                        (tmp[y * 16 + x_base + 0] + 8) >> 4,
                    );
                    let c_hi = _mm_set_epi32(
                        (tmp[y * 16 + x_base + 7] + 8) >> 4,
                        (tmp[y * 16 + x_base + 6] + 8) >> 4,
                        (tmp[y * 16 + x_base + 5] + 8) >> 4,
                        (tmp[y * 16 + x_base + 4] + 8) >> 4,
                    );

                    let sum_lo = _mm_add_epi32(d_lo, c_lo);
                    let sum_hi = _mm_add_epi32(d_hi, c_hi);
                    let clamped_lo = _mm_max_epi32(_mm_min_epi32(sum_lo, max_val), zero);
                    let clamped_hi = _mm_max_epi32(_mm_min_epi32(sum_hi, max_val), zero);
                    let packed = _mm_packus_epi32(clamped_lo, clamped_hi);

                    storeu_128!(
                        <&mut [u16; 8]>::try_from(&mut dst[dst_chunk_off..dst_chunk_off + 8])
                            .unwrap(),
                        packed
                    );
                }
            }

            // Clear coefficients
            coeff[..256].fill(0);
        }
    };
}

// 16x16 hybrid identity transforms 16bpc
impl_16x16_transform_16bpc!(
    inv_txfm_add_identity_dct_16x16_16bpc_avx2_inner,
    identity16_1d,
    dct16_1d
);
impl_16x16_transform_16bpc!(
    inv_txfm_add_dct_identity_16x16_16bpc_avx2_inner,
    dct16_1d,
    identity16_1d
);
impl_16x16_transform_16bpc!(
    inv_txfm_add_identity_adst_16x16_16bpc_avx2_inner,
    identity16_1d,
    adst16_1d
);
impl_16x16_transform_16bpc!(
    inv_txfm_add_adst_identity_16x16_16bpc_avx2_inner,
    adst16_1d,
    identity16_1d
);
impl_16x16_transform_16bpc!(
    inv_txfm_add_identity_flipadst_16x16_16bpc_avx2_inner,
    identity16_1d,
    flipadst16_1d
);
impl_16x16_transform_16bpc!(
    inv_txfm_add_flipadst_identity_16x16_16bpc_avx2_inner,
    flipadst16_1d,
    identity16_1d
);

impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_dct_16x16_16bpc_avx2,
    inv_txfm_add_identity_dct_16x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_identity_16x16_16bpc_avx2,
    inv_txfm_add_dct_identity_16x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_adst_16x16_16bpc_avx2,
    inv_txfm_add_identity_adst_16x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_identity_16x16_16bpc_avx2,
    inv_txfm_add_adst_identity_16x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_flipadst_16x16_16bpc_avx2,
    inv_txfm_add_identity_flipadst_16x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_identity_16x16_16bpc_avx2,
    inv_txfm_add_flipadst_identity_16x16_16bpc_avx2_inner
);

// ============================================================================
// DISPATCH: Safe entry points for ITX SIMD dispatch
// ============================================================================
//
// These functions encapsulate ALL unsafe (pointer creation + extern "C" calls)
// so that src/itx.rs can remain fully safe in the default build.

use crate::include::common::bitdepth::BPC;
use crate::src::levels::TxfmSize;
use crate::src::levels::TxfmType;
use crate::src::levels::ADST_ADST;
use crate::src::levels::ADST_DCT;
use crate::src::levels::ADST_FLIPADST;
use crate::src::levels::DCT_ADST;
use crate::src::levels::DCT_DCT;
use crate::src::levels::DCT_FLIPADST;
use crate::src::levels::FLIPADST_ADST;
use crate::src::levels::FLIPADST_DCT;
use crate::src::levels::FLIPADST_FLIPADST;
use crate::src::levels::H_ADST;
use crate::src::levels::H_DCT;
use crate::src::levels::H_FLIPADST;
use crate::src::levels::IDTX;
use crate::src::levels::V_ADST;
use crate::src::levels::V_DCT;
use crate::src::levels::V_FLIPADST;
use crate::src::levels::WHT_WHT;
use crate::src::strided::Strided as _;

/// Macro to generate the per-arch/bpc direct dispatch functions for ITX.
/// Each generated function matches on (tx_size, tx_type) and calls the right SIMD function.
macro_rules! impl_itxfm_direct_dispatch {
    (
        fn $fn_name:ident, $mod_path:path,
        itx16: [$(($sz16:expr, $w16:literal, $h16:literal)),* $(,)?],
        itx12: [$(($sz12:expr, $w12:literal, $h12:literal)),* $(,)?],
        itx2: [$(($sz2:expr, $w2:literal, $h2:literal)),* $(,)?],
        itx1: [$(($sz1:expr, $w1:literal, $h1:literal)),* $(,)?],
        wht: ($szw:expr, $ww:literal, $hw:literal),
        $bpc:literal bpc, $ext:ident,
        h_dct_fn: $h_dct_fn:ident, v_dct_fn: $v_dct_fn:ident,
        h_adst_fn: $h_adst_fn:ident, v_adst_fn: $v_adst_fn:ident,
        h_flipadst_fn: $h_flipadst_fn:ident, v_flipadst_fn: $v_flipadst_fn:ident
    ) => {
        paste::paste! {
            #[allow(non_upper_case_globals)]
            #[cfg(feature = "asm")]
            fn $fn_name(
                tx_size: usize,
                tx_type: usize,
                dst_ptr: *mut DynPixel,
                dst_stride: isize,
                coeff: *mut DynCoef,
                eob: i32,
                bitdepth_max: i32,
                coeff_len: u16,
                dst: *const FFISafe<PicOffset>,
            ) -> bool {
                use $mod_path as si;

                macro_rules! c {
                    ($func:expr) => {{
                        // SAFETY: SIMD feature verified by caller. All pointers from valid Rust refs.
                        unsafe { $func(dst_ptr, dst_stride, coeff, eob, bitdepth_max, coeff_len, dst) };
                        return true;
                    }};
                }

                // TxfmSize constants for matching
                const s4x4: usize = TxfmSize::S4x4 as usize;
                const s8x8: usize = TxfmSize::S8x8 as usize;
                const s16x16: usize = TxfmSize::S16x16 as usize;
                const s32x32: usize = TxfmSize::S32x32 as usize;
                const s64x64: usize = TxfmSize::S64x64 as usize;
                const r4x8: usize = TxfmSize::R4x8 as usize;
                const r8x4: usize = TxfmSize::R8x4 as usize;
                const r8x16: usize = TxfmSize::R8x16 as usize;
                const r16x8: usize = TxfmSize::R16x8 as usize;
                const r16x32: usize = TxfmSize::R16x32 as usize;
                const r32x16: usize = TxfmSize::R32x16 as usize;
                const r32x64: usize = TxfmSize::R32x64 as usize;
                const r64x32: usize = TxfmSize::R64x32 as usize;
                const r4x16: usize = TxfmSize::R4x16 as usize;
                const r16x4: usize = TxfmSize::R16x4 as usize;
                const r8x32: usize = TxfmSize::R8x32 as usize;
                const r32x8: usize = TxfmSize::R32x8 as usize;
                const r16x64: usize = TxfmSize::R16x64 as usize;
                const r64x16: usize = TxfmSize::R64x16 as usize;

                match (tx_size, tx_type as TxfmType) {
                    // WHT_WHT (only 4x4)
                    ($szw, WHT_WHT) => c!(si::[<inv_txfm_add_wht_wht_ $ww x $hw _ $bpc bpc_ $ext>]),

                    // itx16 sizes: all 16 non-WHT transform types
                    $(
                        ($sz16, DCT_DCT) => c!(si::[<inv_txfm_add_dct_dct_ $w16 x $h16 _ $bpc bpc_ $ext>]),
                        ($sz16, IDTX) => c!(si::[<inv_txfm_add_identity_identity_ $w16 x $h16 _ $bpc bpc_ $ext>]),
                        ($sz16, ADST_DCT) => c!(si::[<inv_txfm_add_dct_adst_ $w16 x $h16 _ $bpc bpc_ $ext>]),
                        ($sz16, DCT_ADST) => c!(si::[<inv_txfm_add_adst_dct_ $w16 x $h16 _ $bpc bpc_ $ext>]),
                        ($sz16, ADST_ADST) => c!(si::[<inv_txfm_add_adst_adst_ $w16 x $h16 _ $bpc bpc_ $ext>]),
                        ($sz16, FLIPADST_DCT) => c!(si::[<inv_txfm_add_dct_flipadst_ $w16 x $h16 _ $bpc bpc_ $ext>]),
                        ($sz16, DCT_FLIPADST) => c!(si::[<inv_txfm_add_flipadst_dct_ $w16 x $h16 _ $bpc bpc_ $ext>]),
                        ($sz16, FLIPADST_FLIPADST) => c!(si::[<inv_txfm_add_flipadst_flipadst_ $w16 x $h16 _ $bpc bpc_ $ext>]),
                        ($sz16, ADST_FLIPADST) => c!(si::[<inv_txfm_add_flipadst_adst_ $w16 x $h16 _ $bpc bpc_ $ext>]),
                        ($sz16, FLIPADST_ADST) => c!(si::[<inv_txfm_add_adst_flipadst_ $w16 x $h16 _ $bpc bpc_ $ext>]),
                        ($sz16, H_DCT) => c!(si::[<inv_txfm_add_ $h_dct_fn _ $w16 x $h16 _ $bpc bpc_ $ext>]),
                        ($sz16, V_DCT) => c!(si::[<inv_txfm_add_ $v_dct_fn _ $w16 x $h16 _ $bpc bpc_ $ext>]),
                        ($sz16, H_ADST) => c!(si::[<inv_txfm_add_ $h_adst_fn _ $w16 x $h16 _ $bpc bpc_ $ext>]),
                        ($sz16, V_ADST) => c!(si::[<inv_txfm_add_ $v_adst_fn _ $w16 x $h16 _ $bpc bpc_ $ext>]),
                        ($sz16, H_FLIPADST) => c!(si::[<inv_txfm_add_ $h_flipadst_fn _ $w16 x $h16 _ $bpc bpc_ $ext>]),
                        ($sz16, V_FLIPADST) => c!(si::[<inv_txfm_add_ $v_flipadst_fn _ $w16 x $h16 _ $bpc bpc_ $ext>]),
                    )*

                    // itx12 sizes: 12 transform types (no H/V ADST/FLIPADST hybrids)
                    $(
                        ($sz12, DCT_DCT) => c!(si::[<inv_txfm_add_dct_dct_ $w12 x $h12 _ $bpc bpc_ $ext>]),
                        ($sz12, IDTX) => c!(si::[<inv_txfm_add_identity_identity_ $w12 x $h12 _ $bpc bpc_ $ext>]),
                        ($sz12, ADST_DCT) => c!(si::[<inv_txfm_add_dct_adst_ $w12 x $h12 _ $bpc bpc_ $ext>]),
                        ($sz12, DCT_ADST) => c!(si::[<inv_txfm_add_adst_dct_ $w12 x $h12 _ $bpc bpc_ $ext>]),
                        ($sz12, ADST_ADST) => c!(si::[<inv_txfm_add_adst_adst_ $w12 x $h12 _ $bpc bpc_ $ext>]),
                        ($sz12, FLIPADST_DCT) => c!(si::[<inv_txfm_add_dct_flipadst_ $w12 x $h12 _ $bpc bpc_ $ext>]),
                        ($sz12, DCT_FLIPADST) => c!(si::[<inv_txfm_add_flipadst_dct_ $w12 x $h12 _ $bpc bpc_ $ext>]),
                        ($sz12, FLIPADST_FLIPADST) => c!(si::[<inv_txfm_add_flipadst_flipadst_ $w12 x $h12 _ $bpc bpc_ $ext>]),
                        ($sz12, ADST_FLIPADST) => c!(si::[<inv_txfm_add_flipadst_adst_ $w12 x $h12 _ $bpc bpc_ $ext>]),
                        ($sz12, FLIPADST_ADST) => c!(si::[<inv_txfm_add_adst_flipadst_ $w12 x $h12 _ $bpc bpc_ $ext>]),
                        ($sz12, H_DCT) => c!(si::[<inv_txfm_add_ $h_dct_fn _ $w12 x $h12 _ $bpc bpc_ $ext>]),
                        ($sz12, V_DCT) => c!(si::[<inv_txfm_add_ $v_dct_fn _ $w12 x $h12 _ $bpc bpc_ $ext>]),
                    )*

                    // itx2 sizes: DCT_DCT + IDTX only
                    $(
                        ($sz2, DCT_DCT) => c!(si::[<inv_txfm_add_dct_dct_ $w2 x $h2 _ $bpc bpc_ $ext>]),
                        ($sz2, IDTX) => c!(si::[<inv_txfm_add_identity_identity_ $w2 x $h2 _ $bpc bpc_ $ext>]),
                    )*

                    // itx1 sizes: DCT_DCT only
                    $(
                        ($sz1, DCT_DCT) => c!(si::[<inv_txfm_add_dct_dct_ $w1 x $h1 _ $bpc bpc_ $ext>]),
                    )*

                    _ => return false,
                }
            }
        }
    };
}

// x86_64 8bpc direct dispatch
#[cfg(target_arch = "x86_64")]
impl_itxfm_direct_dispatch!(
    fn itxfm_add_direct_x86_8bpc, crate::src::safe_simd::itx,
    itx16: [
        (s4x4, 4, 4),
        (s8x8, 8, 8),
        (r4x8, 4, 8), (r8x4, 8, 4),
        (r4x16, 4, 16), (r16x4, 16, 4),
        (r8x16, 8, 16), (r16x8, 16, 8),
    ],
    itx12: [
        (s16x16, 16, 16),
    ],
    itx2: [
        (r8x32, 8, 32), (r32x8, 32, 8),
        (r16x32, 16, 32), (r32x16, 32, 16),
        (s32x32, 32, 32),
    ],
    itx1: [
        (r16x64, 16, 64), (r32x64, 32, 64),
        (r64x16, 64, 16), (r64x32, 64, 32),
        (s64x64, 64, 64),
    ],
    wht: (s4x4, 4, 4),
    8 bpc, avx2,
    h_dct_fn: dct_identity, v_dct_fn: identity_dct,
    h_adst_fn: adst_identity, v_adst_fn: identity_adst,
    h_flipadst_fn: flipadst_identity, v_flipadst_fn: identity_flipadst
);

// x86_64 16bpc direct dispatch
#[cfg(target_arch = "x86_64")]
impl_itxfm_direct_dispatch!(
    fn itxfm_add_direct_x86_16bpc, crate::src::safe_simd::itx,
    itx16: [
        (s4x4, 4, 4),
        (s8x8, 8, 8),
        (r4x8, 4, 8), (r8x4, 8, 4),
        (r4x16, 4, 16), (r16x4, 16, 4),
        (r8x16, 8, 16), (r16x8, 16, 8),
    ],
    itx12: [
        (s16x16, 16, 16),
    ],
    itx2: [
        (r8x32, 8, 32), (r32x8, 32, 8),
        (r16x32, 16, 32), (r32x16, 32, 16),
        (s32x32, 32, 32),
    ],
    itx1: [
        (r16x64, 16, 64), (r32x64, 32, 64),
        (r64x16, 64, 16), (r64x32, 64, 32),
        (s64x64, 64, 64),
    ],
    wht: (s4x4, 4, 4),
    16 bpc, avx2,
    h_dct_fn: dct_identity, v_dct_fn: identity_dct,
    h_adst_fn: adst_identity, v_adst_fn: identity_adst,
    h_flipadst_fn: flipadst_identity, v_flipadst_fn: identity_flipadst
);

/// 8bpc dispatch: calls inner SIMD functions directly with slices.
/// Arcane functions take (token, dst, stride_usize, coeff, eob, bdmax).
/// Scalar functions take (dst, base, stride_isize, coeff, eob, bdmax).
#[cfg(not(feature = "asm"))]
#[cfg(target_arch = "x86_64")]
#[allow(non_upper_case_globals)]
fn itxfm_dispatch_8bpc(
    token: Desktop64,
    tx_size: usize,
    tx_type: TxfmType,
    dst: &mut [u8],
    base: usize,
    stride_u: usize,
    stride_i: isize,
    coeff: &mut [i16],
    eob: i32,
    bdmax: i32,
) -> bool {
    use crate::src::levels::TxfmSize;

    // Arcane functions: dst starts at pixel (base=0 for positive stride)
    macro_rules! arcane {
        ($func:ident) => {{
            $func(token, &mut dst[base..], stride_u, coeff, eob, bdmax);
            return true;
        }};
    }
    // Scalar functions: pass dst with base offset and signed stride
    macro_rules! scalar {
        ($func:ident) => {{
            $func(dst, base, stride_i, coeff, eob, bdmax);
            return true;
        }};
    }

    const S4x4: usize = TxfmSize::S4x4 as usize;
    const S8x8: usize = TxfmSize::S8x8 as usize;
    const S16x16: usize = TxfmSize::S16x16 as usize;
    const S32x32: usize = TxfmSize::S32x32 as usize;
    const S64x64: usize = TxfmSize::S64x64 as usize;
    const R4x8: usize = TxfmSize::R4x8 as usize;
    const R8x4: usize = TxfmSize::R8x4 as usize;
    const R8x16: usize = TxfmSize::R8x16 as usize;
    const R16x8: usize = TxfmSize::R16x8 as usize;
    const R16x32: usize = TxfmSize::R16x32 as usize;
    const R32x16: usize = TxfmSize::R32x16 as usize;
    const R32x64: usize = TxfmSize::R32x64 as usize;
    const R64x32: usize = TxfmSize::R64x32 as usize;
    const R4x16: usize = TxfmSize::R4x16 as usize;
    const R16x4: usize = TxfmSize::R16x4 as usize;
    const R8x32: usize = TxfmSize::R8x32 as usize;
    const R32x8: usize = TxfmSize::R32x8 as usize;
    const R16x64: usize = TxfmSize::R16x64 as usize;
    const R64x16: usize = TxfmSize::R64x16 as usize;

    match (tx_size, tx_type) {
        // ===== WHT (scalar, 4x4 only) =====
        (S4x4, WHT_WHT) => scalar!(inv_txfm_add_wht_wht_4x4_8bpc_avx2_inner),

        // ===== DCT_DCT (arcane, all 19 sizes) =====
        (S4x4, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_4x4_8bpc_avx2_inner),
        (R4x8, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_4x8_8bpc_avx2_inner),
        (R8x4, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_8x4_8bpc_avx2_inner),
        (R4x16, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_4x16_8bpc_avx2_inner),
        (R16x4, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_16x4_8bpc_avx2_inner),
        (S8x8, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_8x8_8bpc_avx2_inner),
        (R8x16, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_8x16_8bpc_avx2_inner),
        (R16x8, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_16x8_8bpc_avx2_inner),
        (R8x32, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_8x32_8bpc_avx2_inner),
        (R32x8, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_32x8_8bpc_avx2_inner),
        (S16x16, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_16x16_8bpc_avx2_inner),
        (R16x32, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_16x32_8bpc_avx2_inner),
        (R32x16, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_32x16_8bpc_avx2_inner),
        (R16x64, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_16x64_8bpc_avx2_inner),
        (R64x16, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_64x16_8bpc_avx2_inner),
        (S32x32, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_32x32_8bpc_avx2_inner),
        (R32x64, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_32x64_8bpc_avx2_inner),
        (R64x32, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_64x32_8bpc_avx2_inner),
        (S64x64, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_64x64_8bpc_avx2_inner),

        // ===== IDTX (arcane, 8 sizes) =====
        (S4x4, IDTX) => arcane!(inv_identity_add_4x4_8bpc_avx2),
        (S8x8, IDTX) => arcane!(inv_identity_add_8x8_8bpc_avx2),
        (S16x16, IDTX) => arcane!(inv_identity_add_16x16_8bpc_avx2),
        (R8x32, IDTX) => arcane!(inv_txfm_add_identity_identity_8x32_8bpc_avx2_inner),
        (R32x8, IDTX) => arcane!(inv_txfm_add_identity_identity_32x8_8bpc_avx2_inner),
        (R16x32, IDTX) => arcane!(inv_txfm_add_identity_identity_16x32_8bpc_avx2_inner),
        (R32x16, IDTX) => arcane!(inv_txfm_add_identity_identity_32x16_8bpc_avx2_inner),
        (S32x32, IDTX) => arcane!(inv_txfm_add_identity_identity_32x32_8bpc_avx2_inner),

        // ===== 4x4 ADST/FLIPADST/hybrid (scalar, 14 types) =====
        (S4x4, ADST_DCT) => scalar!(inv_txfm_add_dct_adst_4x4_8bpc_avx2_inner),
        (S4x4, DCT_ADST) => scalar!(inv_txfm_add_adst_dct_4x4_8bpc_avx2_inner),
        (S4x4, ADST_ADST) => scalar!(inv_txfm_add_adst_adst_4x4_8bpc_avx2_inner),
        (S4x4, FLIPADST_DCT) => scalar!(inv_txfm_add_dct_flipadst_4x4_8bpc_avx2_inner),
        (S4x4, DCT_FLIPADST) => scalar!(inv_txfm_add_flipadst_dct_4x4_8bpc_avx2_inner),
        (S4x4, FLIPADST_FLIPADST) => scalar!(inv_txfm_add_flipadst_flipadst_4x4_8bpc_avx2_inner),
        (S4x4, ADST_FLIPADST) => scalar!(inv_txfm_add_flipadst_adst_4x4_8bpc_avx2_inner),
        (S4x4, FLIPADST_ADST) => scalar!(inv_txfm_add_adst_flipadst_4x4_8bpc_avx2_inner),
        (S4x4, H_DCT) => scalar!(inv_txfm_add_dct_identity_4x4_8bpc_avx2_inner),
        (S4x4, V_DCT) => scalar!(inv_txfm_add_identity_dct_4x4_8bpc_avx2_inner),
        (S4x4, H_ADST) => scalar!(inv_txfm_add_h_adst_4x4_8bpc_avx2_inner),
        (S4x4, V_ADST) => scalar!(inv_txfm_add_v_adst_4x4_8bpc_avx2_inner),
        (S4x4, H_FLIPADST) => scalar!(inv_txfm_add_h_flipadst_4x4_8bpc_avx2_inner),
        (S4x4, V_FLIPADST) => scalar!(inv_txfm_add_v_flipadst_4x4_8bpc_avx2_inner),

        _ => return false,
    }
}

/// 16bpc dispatch: calls inner SIMD functions directly with slices.
/// All arcane functions take (token, dst: &mut [u16], byte_stride: usize, coeff, eob, bdmax).
/// WHT takes (dst: &mut [u16], base, px_stride: isize, coeff, eob, bdmax).
#[cfg(not(feature = "asm"))]
#[cfg(target_arch = "x86_64")]
#[allow(non_upper_case_globals)]
fn itxfm_dispatch_16bpc(
    token: Desktop64,
    tx_size: usize,
    tx_type: TxfmType,
    dst: &mut [u16],
    base: usize,
    byte_stride: usize,
    px_stride: isize,
    coeff: &mut [i16],
    eob: i32,
    bdmax: i32,
) -> bool {
    use crate::src::levels::TxfmSize;

    // Arcane 16bpc functions take byte_stride as usize
    macro_rules! arcane {
        ($func:ident) => {{
            $func(token, &mut dst[base..], byte_stride, coeff, eob, bdmax);
            return true;
        }};
    }
    // WHT scalar takes pixel stride as isize
    macro_rules! scalar {
        ($func:ident) => {{
            $func(dst, base, px_stride, coeff, eob, bdmax);
            return true;
        }};
    }

    const S4x4: usize = TxfmSize::S4x4 as usize;
    const S8x8: usize = TxfmSize::S8x8 as usize;
    const S16x16: usize = TxfmSize::S16x16 as usize;
    const S32x32: usize = TxfmSize::S32x32 as usize;
    const S64x64: usize = TxfmSize::S64x64 as usize;
    const R4x8: usize = TxfmSize::R4x8 as usize;
    const R8x4: usize = TxfmSize::R8x4 as usize;
    const R8x16: usize = TxfmSize::R8x16 as usize;
    const R16x8: usize = TxfmSize::R16x8 as usize;
    const R16x32: usize = TxfmSize::R16x32 as usize;
    const R32x16: usize = TxfmSize::R32x16 as usize;
    const R32x64: usize = TxfmSize::R32x64 as usize;
    const R64x32: usize = TxfmSize::R64x32 as usize;
    const R4x16: usize = TxfmSize::R4x16 as usize;
    const R16x4: usize = TxfmSize::R16x4 as usize;
    const R8x32: usize = TxfmSize::R8x32 as usize;
    const R32x8: usize = TxfmSize::R32x8 as usize;
    const R16x64: usize = TxfmSize::R16x64 as usize;
    const R64x16: usize = TxfmSize::R64x16 as usize;

    match (tx_size, tx_type) {
        // ===== WHT (scalar, 4x4 only) =====
        (S4x4, WHT_WHT) => scalar!(inv_txfm_add_wht_wht_4x4_16bpc_avx2_inner),

        // ===== DCT_DCT (arcane, all 19 sizes) =====
        (S4x4, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_4x4_16bpc_avx2_inner),
        (R4x8, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_4x8_16bpc_avx2_inner),
        (R8x4, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_8x4_16bpc_avx2_inner),
        (R4x16, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_4x16_16bpc_avx2_inner),
        (R16x4, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_16x4_16bpc_avx2_inner),
        (S8x8, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_8x8_16bpc_avx2_inner),
        (R8x16, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_8x16_16bpc_avx2_inner),
        (R16x8, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_16x8_16bpc_avx2_inner),
        (R8x32, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_8x32_16bpc_avx2_inner),
        (R32x8, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_32x8_16bpc_avx2_inner),
        (S16x16, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_16x16_16bpc_avx2_inner),
        (R16x32, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_16x32_16bpc_avx2_inner),
        (R32x16, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_32x16_16bpc_avx2_inner),
        (R16x64, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_16x64_16bpc_avx2_inner),
        (R64x16, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_64x16_16bpc_avx2_inner),
        (S32x32, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_32x32_16bpc_avx2_inner),
        (R32x64, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_32x64_16bpc_avx2_inner),
        (R64x32, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_64x32_16bpc_avx2_inner),
        (S64x64, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_64x64_16bpc_avx2_inner),

        // ===== IDTX (arcane, 14 sizes) =====
        (S4x4, IDTX) => arcane!(inv_identity_add_4x4_16bpc_avx2),
        (R4x8, IDTX) => arcane!(inv_txfm_add_identity_identity_4x8_16bpc_avx2_inner),
        (R8x4, IDTX) => arcane!(inv_txfm_add_identity_identity_8x4_16bpc_avx2_inner),
        (R4x16, IDTX) => arcane!(inv_txfm_add_identity_identity_4x16_16bpc_avx2_inner),
        (R16x4, IDTX) => arcane!(inv_txfm_add_identity_identity_16x4_16bpc_avx2_inner),
        (S8x8, IDTX) => arcane!(inv_identity_add_8x8_16bpc_avx2),
        (R8x16, IDTX) => arcane!(inv_txfm_add_identity_identity_8x16_16bpc_avx2_inner),
        (R16x8, IDTX) => arcane!(inv_txfm_add_identity_identity_16x8_16bpc_avx2_inner),
        (R8x32, IDTX) => arcane!(inv_txfm_add_identity_identity_8x32_16bpc_avx2_inner),
        (R32x8, IDTX) => arcane!(inv_txfm_add_identity_identity_32x8_16bpc_avx2_inner),
        (S16x16, IDTX) => arcane!(inv_identity_add_16x16_16bpc_avx2),
        (R16x32, IDTX) => arcane!(inv_txfm_add_identity_identity_16x32_16bpc_avx2_inner),
        (R32x16, IDTX) => arcane!(inv_txfm_add_identity_identity_32x16_16bpc_avx2_inner),
        (S32x32, IDTX) => arcane!(inv_txfm_add_identity_identity_32x32_16bpc_avx2_inner),

        _ => return false,
    }
}

/// Safe dispatch entry point for ITX SIMD on x86_64 (non-asm path).
///
/// Calls inner SIMD functions directly with slices — no FFI wrappers, no raw pointers.
/// Returns `true` if a SIMD implementation handled the call.
#[cfg(not(feature = "asm"))]
pub fn itxfm_add_dispatch<BD: BitDepth>(
    tx_size: usize,
    tx_type: usize,
    dst: PicOffset,
    coeff: &mut [BD::Coef],
    eob: i32,
    bd: BD,
) -> bool {
    use crate::src::strided::Strided as _;
    use zerocopy::AsBytes;

    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (tx_size, tx_type, &dst, &coeff, eob, &bd);
        return false;
    }

    #[cfg(target_arch = "x86_64")]
    {
        let Some(token) = Desktop64::summon() else {
            return false;
        };

        let txfm = match crate::src::levels::TxfmSize::from_repr(tx_size) {
            Some(t) => t,
            None => return false,
        };
        let (w, h) = txfm.to_wh();
        let byte_stride_i = dst.stride(); // isize, in bytes
        let byte_stride_u = byte_stride_i.unsigned_abs();
        let bd_c = bd.into_c();

        // Reinterpret coeff as &mut [i16] (safe via zerocopy)
        let coeff_i16: &mut [i16] = zerocopy::FromBytes::mut_slice_from(coeff.as_bytes_mut())
            .expect("coeff alignment/size mismatch for i16 reinterpretation");

        match BD::BPC {
            BPC::BPC8 => {
                let (mut guard, base) = dst.strided_slice_mut::<BD>(w, h);
                let dst_u8: &mut [u8] = guard.as_bytes_mut();
                itxfm_dispatch_8bpc(
                    token,
                    tx_size,
                    tx_type as TxfmType,
                    dst_u8,
                    base,
                    byte_stride_u,
                    byte_stride_i,
                    coeff_i16,
                    eob,
                    bd_c,
                )
            }
            BPC::BPC16 => {
                let (mut guard, base) = dst.strided_slice_mut::<BD>(w, h);
                let dst_bytes: &mut [u8] = guard.as_bytes_mut();
                let dst_u16: &mut [u16] = zerocopy::FromBytes::mut_slice_from(dst_bytes)
                    .expect("dst alignment/size mismatch for u16 reinterpretation");
                let px_stride_i = dst.pixel_stride::<BD>(); // isize, in pixels
                itxfm_dispatch_16bpc(
                    token,
                    tx_size,
                    tx_type as TxfmType,
                    dst_u16,
                    base,
                    byte_stride_u,
                    px_stride_i,
                    coeff_i16,
                    eob,
                    bd_c,
                )
            }
        }
    }
}

/// Safe dispatch entry point for ITX SIMD on x86_64.
///
/// Takes safe Rust types, creates raw pointers internally, dispatches to the
/// right SIMD function. Returns `true` if a SIMD implementation handled the call.
#[cfg(feature = "asm")]
#[allow(unsafe_code)]
pub fn itxfm_add_dispatch<BD: BitDepth>(
    tx_size: usize,
    tx_type: usize,
    dst: PicOffset,
    coeff: &mut [BD::Coef],
    eob: i32,
    bd: BD,
) -> bool {
    use crate::src::levels::TxfmSize;
    use crate::src::safe_simd::pixel_access::Flex;
    use archmage::Desktop64;
    use zerocopy::AsBytes;

    let Some(_token) = Desktop64::summon() else {
        return false;
    };

    // Get transform dimensions for tracked guard
    let txfm = TxfmSize::from_repr(tx_size).unwrap_or_default();
    let (w, h) = txfm.to_wh();

    // Create tracked guard — ensures borrow tracker knows about this access
    let (mut dst_guard, _dst_base) = dst.strided_slice_mut::<BD>(w, h);
    let dst_ptr: *mut DynPixel = dst_guard.as_bytes_mut().as_mut_ptr() as *mut DynPixel;
    let dst_stride = dst.stride();
    let coeff_len = coeff.len() as u16;
    let coeff_ptr = coeff.as_mut_ptr().cast();
    let bd_c = bd.into_c();
    let dst_ffi = FFISafe::new(&dst);

    match BD::BPC {
        BPC::BPC8 => itxfm_add_direct_x86_8bpc(
            tx_size, tx_type, dst_ptr, dst_stride, coeff_ptr, eob, bd_c, coeff_len, dst_ffi,
        ),
        BPC::BPC16 => itxfm_add_direct_x86_16bpc(
            tx_size, tx_type, dst_ptr, dst_stride, coeff_ptr, eob, bd_c, coeff_len, dst_ffi,
        ),
    }
}
