//! Safe SIMD implementations for ITX (Inverse Transforms)
//!
//! ITX is the largest DSP module (~42k asm lines). Strategy:
//! 1. Implement full 2D transforms (not just 1D) for common sizes
//! 2. Process multiple rows/columns in parallel
//! 3. Use in-register transposition
//!
//! Most common transforms: DCT_DCT 4x4, 8x8, 16x16

#![allow(unused_imports)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::include::common::bitdepth::AsPrimitive;
use crate::include::common::bitdepth::BitDepth;
use crate::include::common::bitdepth::DynCoef;
use crate::include::common::bitdepth::DynPixel;
use crate::include::common::intops::iclip;
use crate::include::dav1d::picture::Rav1dPictureDataComponentOffset;
use crate::src::ffi_safe::FFISafe;
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
#[target_feature(enable = "avx2")]
unsafe fn inv_txfm_add_dct_dct_4x4_8bpc_avx2_inner(
    dst: *mut u8,
    dst_stride: isize,
    coeff: *mut i16,
    _eob: i32,
    bitdepth_max: i32,
) {
    // Load coefficients (column-major storage in rav1d)
    // coeff[y + x*4] for position (x,y)
    // row0: coeff[0], coeff[4], coeff[8], coeff[12]
    // row1: coeff[1], coeff[5], coeff[9], coeff[13]
    // etc.

    let c_ptr = coeff;
    let row0 = unsafe {
        _mm_set_epi32(
            *c_ptr.add(12) as i32, *c_ptr.add(8) as i32,
            *c_ptr.add(4) as i32, *c_ptr.add(0) as i32
        )
    };
    let row1 = unsafe {
        _mm_set_epi32(
            *c_ptr.add(13) as i32, *c_ptr.add(9) as i32,
            *c_ptr.add(5) as i32, *c_ptr.add(1) as i32
        )
    };
    let row2 = unsafe {
        _mm_set_epi32(
            *c_ptr.add(14) as i32, *c_ptr.add(10) as i32,
            *c_ptr.add(6) as i32, *c_ptr.add(2) as i32
        )
    };
    let row3 = unsafe {
        _mm_set_epi32(
            *c_ptr.add(15) as i32, *c_ptr.add(11) as i32,
            *c_ptr.add(7) as i32, *c_ptr.add(3) as i32
        )
    };

    // Pack rows into 256-bit vectors for processing
    let rows01 = unsafe { _mm256_set_m128i(row1, row0) };
    let rows23 = unsafe { _mm256_set_m128i(row3, row2) };

    // DCT4 butterfly on rows
    let (rows01_out, rows23_out) = unsafe { dct4_2rows_avx2(rows01, rows23) };

    // Transpose for column pass
    let r0 = unsafe { _mm256_castsi256_si128(rows01_out) };
    let r1 = unsafe { _mm256_extracti128_si256(rows01_out, 1) };
    let r2 = unsafe { _mm256_castsi256_si128(rows23_out) };
    let r3 = unsafe { _mm256_extracti128_si256(rows23_out, 1) };

    // Transpose 4x4 using unpack
    let t01_lo = unsafe { _mm_unpacklo_epi32(r0, r1) };
    let t01_hi = unsafe { _mm_unpackhi_epi32(r0, r1) };
    let t23_lo = unsafe { _mm_unpacklo_epi32(r2, r3) };
    let t23_hi = unsafe { _mm_unpackhi_epi32(r2, r3) };

    let col0 = unsafe { _mm_unpacklo_epi64(t01_lo, t23_lo) };
    let col1 = unsafe { _mm_unpackhi_epi64(t01_lo, t23_lo) };
    let col2 = unsafe { _mm_unpacklo_epi64(t01_hi, t23_hi) };
    let col3 = unsafe { _mm_unpackhi_epi64(t01_hi, t23_hi) };

    let cols01 = unsafe { _mm256_set_m128i(col1, col0) };
    let cols23 = unsafe { _mm256_set_m128i(col3, col2) };

    // DCT4 butterfly on columns
    let (cols01_out, cols23_out) = unsafe { dct4_2rows_avx2(cols01, cols23) };

    // Final scaling: (result + 8) >> 4
    let rnd = unsafe { _mm256_set1_epi32(8) };
    let cols01_scaled = unsafe { _mm256_srai_epi32(_mm256_add_epi32(cols01_out, rnd), 4) };
    let cols23_scaled = unsafe { _mm256_srai_epi32(_mm256_add_epi32(cols23_out, rnd), 4) };

    // Transpose back to row order for storing
    let c0 = unsafe { _mm256_castsi256_si128(cols01_scaled) };
    let c1 = unsafe { _mm256_extracti128_si256(cols01_scaled, 1) };
    let c2 = unsafe { _mm256_castsi256_si128(cols23_scaled) };
    let c3 = unsafe { _mm256_extracti128_si256(cols23_scaled, 1) };

    let u01_lo = unsafe { _mm_unpacklo_epi32(c0, c1) };
    let u01_hi = unsafe { _mm_unpackhi_epi32(c0, c1) };
    let u23_lo = unsafe { _mm_unpacklo_epi32(c2, c3) };
    let u23_hi = unsafe { _mm_unpackhi_epi32(c2, c3) };

    let final0 = unsafe { _mm_unpacklo_epi64(u01_lo, u23_lo) };
    let final1 = unsafe { _mm_unpackhi_epi64(u01_lo, u23_lo) };
    let final2 = unsafe { _mm_unpacklo_epi64(u01_hi, u23_hi) };
    let final3 = unsafe { _mm_unpackhi_epi64(u01_hi, u23_hi) };

    // Add to destination with clamping
    let zero = unsafe { _mm_setzero_si128() };
    let max_val = unsafe { _mm_set1_epi16(bitdepth_max as i16) };

    // Row 0
    unsafe {
        let d0 = _mm_cvtsi32_si128(*(dst as *const i32));
        let d0_16 = _mm_unpacklo_epi8(d0, zero);
        let d0_32 = _mm_cvtepi16_epi32(d0_16);
        let sum0 = _mm_add_epi32(d0_32, final0);
        let sum0_16 = _mm_packs_epi32(sum0, sum0);
        let sum0_clamped = _mm_max_epi16(_mm_min_epi16(sum0_16, max_val), zero);
        let sum0_8 = _mm_packus_epi16(sum0_clamped, sum0_clamped);
        *(dst as *mut i32) = _mm_cvtsi128_si32(sum0_8);
    }

    // Row 1
    unsafe {
        let d1 = _mm_cvtsi32_si128(*(dst.offset(dst_stride) as *const i32));
        let d1_16 = _mm_unpacklo_epi8(d1, zero);
        let d1_32 = _mm_cvtepi16_epi32(d1_16);
        let sum1 = _mm_add_epi32(d1_32, final1);
        let sum1_16 = _mm_packs_epi32(sum1, sum1);
        let sum1_clamped = _mm_max_epi16(_mm_min_epi16(sum1_16, max_val), zero);
        let sum1_8 = _mm_packus_epi16(sum1_clamped, sum1_clamped);
        *(dst.offset(dst_stride) as *mut i32) = _mm_cvtsi128_si32(sum1_8);
    }

    // Row 2
    unsafe {
        let d2 = _mm_cvtsi32_si128(*(dst.offset(dst_stride * 2) as *const i32));
        let d2_16 = _mm_unpacklo_epi8(d2, zero);
        let d2_32 = _mm_cvtepi16_epi32(d2_16);
        let sum2 = _mm_add_epi32(d2_32, final2);
        let sum2_16 = _mm_packs_epi32(sum2, sum2);
        let sum2_clamped = _mm_max_epi16(_mm_min_epi16(sum2_16, max_val), zero);
        let sum2_8 = _mm_packus_epi16(sum2_clamped, sum2_clamped);
        *(dst.offset(dst_stride * 2) as *mut i32) = _mm_cvtsi128_si32(sum2_8);
    }

    // Row 3
    unsafe {
        let d3 = _mm_cvtsi32_si128(*(dst.offset(dst_stride * 3) as *const i32));
        let d3_16 = _mm_unpacklo_epi8(d3, zero);
        let d3_32 = _mm_cvtepi16_epi32(d3_16);
        let sum3 = _mm_add_epi32(d3_32, final3);
        let sum3_16 = _mm_packs_epi32(sum3, sum3);
        let sum3_clamped = _mm_max_epi16(_mm_min_epi16(sum3_16, max_val), zero);
        let sum3_8 = _mm_packus_epi16(sum3_clamped, sum3_clamped);
        *(dst.offset(dst_stride * 3) as *mut i32) = _mm_cvtsi128_si32(sum3_8);
    }

    // Clear coefficients
    unsafe {
        _mm_storeu_si128(coeff as *mut __m128i, _mm_setzero_si128());
        _mm_storeu_si128(coeff.add(8) as *mut __m128i, _mm_setzero_si128());
    }
}

/// DCT4 butterfly on 2 rows packed in __m256i
/// Each 128-bit lane contains one row: [in0, in1, in2, in3] as i32
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dct4_2rows_avx2(
    rows01: __m256i,
    rows23: __m256i,
) -> (__m256i, __m256i) {
    // DCT4: t0 = (in0 + in2) * 181 + 128 >> 8
    //       t1 = (in0 - in2) * 181 + 128 >> 8
    //       t2 = (in1 * 1567 - in3 * (3784-4096) + 2048 >> 12) - in3
    //       t3 = (in1 * (3784-4096) + in3 * 1567 + 2048 >> 12) + in1

    let sqrt2 = unsafe { _mm256_set1_epi32(181) };
    let rnd8 = unsafe { _mm256_set1_epi32(128) };
    let c1567 = unsafe { _mm256_set1_epi32(1567) };
    let c_312 = unsafe { _mm256_set1_epi32(3784 - 4096) };
    let rnd12 = unsafe { _mm256_set1_epi32(2048) };

    // Process rows01
    let in0_01 = unsafe { _mm256_shuffle_epi32(rows01, 0b00_00_00_00) };
    let in1_01 = unsafe { _mm256_shuffle_epi32(rows01, 0b01_01_01_01) };
    let in2_01 = unsafe { _mm256_shuffle_epi32(rows01, 0b10_10_10_10) };
    let in3_01 = unsafe { _mm256_shuffle_epi32(rows01, 0b11_11_11_11) };

    // t0 = (in0 + in2) * 181 + 128 >> 8
    let sum02_01 = unsafe { _mm256_add_epi32(in0_01, in2_01) };
    let t0_01 = unsafe {
        _mm256_srai_epi32(
            _mm256_add_epi32(_mm256_mullo_epi32(sum02_01, sqrt2), rnd8),
            8
        )
    };

    // t1 = (in0 - in2) * 181 + 128 >> 8
    let diff02_01 = unsafe { _mm256_sub_epi32(in0_01, in2_01) };
    let t1_01 = unsafe {
        _mm256_srai_epi32(
            _mm256_add_epi32(_mm256_mullo_epi32(diff02_01, sqrt2), rnd8),
            8
        )
    };

    // t2 = (in1 * 1567 - in3 * (3784-4096) + 2048 >> 12) - in3
    let mul1_1567_01 = unsafe { _mm256_mullo_epi32(in1_01, c1567) };
    let mul3_312_01 = unsafe { _mm256_mullo_epi32(in3_01, c_312) };
    let t2_inner_01 = unsafe {
        _mm256_srai_epi32(
            _mm256_add_epi32(_mm256_sub_epi32(mul1_1567_01, mul3_312_01), rnd12),
            12
        )
    };
    let t2_01 = unsafe { _mm256_sub_epi32(t2_inner_01, in3_01) };

    // t3 = (in1 * (3784-4096) + in3 * 1567 + 2048 >> 12) + in1
    let mul1_312_01 = unsafe { _mm256_mullo_epi32(in1_01, c_312) };
    let mul3_1567_01 = unsafe { _mm256_mullo_epi32(in3_01, c1567) };
    let t3_inner_01 = unsafe {
        _mm256_srai_epi32(
            _mm256_add_epi32(_mm256_add_epi32(mul1_312_01, mul3_1567_01), rnd12),
            12
        )
    };
    let t3_01 = unsafe { _mm256_add_epi32(t3_inner_01, in1_01) };

    // Output: out0 = t0+t3, out1 = t1+t2, out2 = t1-t2, out3 = t0-t3
    let out0_01 = unsafe { _mm256_add_epi32(t0_01, t3_01) };
    let out1_01 = unsafe { _mm256_add_epi32(t1_01, t2_01) };
    let out2_01 = unsafe { _mm256_sub_epi32(t1_01, t2_01) };
    let out3_01 = unsafe { _mm256_sub_epi32(t0_01, t3_01) };

    // Interleave outputs back: [out0, out1, out2, out3] per lane
    let mask0 = unsafe { _mm256_set_epi32(0, 0, 0, -1i32, 0, 0, 0, -1i32) };
    let mask1 = unsafe { _mm256_set_epi32(0, 0, -1i32, 0, 0, 0, -1i32, 0) };
    let mask2 = unsafe { _mm256_set_epi32(0, -1i32, 0, 0, 0, -1i32, 0, 0) };
    let mask3 = unsafe { _mm256_set_epi32(-1i32, 0, 0, 0, -1i32, 0, 0, 0) };

    let rows01_out = unsafe {
        _mm256_or_si256(
            _mm256_or_si256(
                _mm256_and_si256(out0_01, mask0),
                _mm256_and_si256(_mm256_shuffle_epi32(out1_01, 0b00_00_00_01), mask1)
            ),
            _mm256_or_si256(
                _mm256_and_si256(_mm256_shuffle_epi32(out2_01, 0b00_00_10_00), mask2),
                _mm256_and_si256(_mm256_shuffle_epi32(out3_01, 0b00_11_00_00), mask3)
            )
        )
    };

    // Same for rows23
    let in0_23 = unsafe { _mm256_shuffle_epi32(rows23, 0b00_00_00_00) };
    let in1_23 = unsafe { _mm256_shuffle_epi32(rows23, 0b01_01_01_01) };
    let in2_23 = unsafe { _mm256_shuffle_epi32(rows23, 0b10_10_10_10) };
    let in3_23 = unsafe { _mm256_shuffle_epi32(rows23, 0b11_11_11_11) };

    let sum02_23 = unsafe { _mm256_add_epi32(in0_23, in2_23) };
    let t0_23 = unsafe {
        _mm256_srai_epi32(
            _mm256_add_epi32(_mm256_mullo_epi32(sum02_23, sqrt2), rnd8),
            8
        )
    };

    let diff02_23 = unsafe { _mm256_sub_epi32(in0_23, in2_23) };
    let t1_23 = unsafe {
        _mm256_srai_epi32(
            _mm256_add_epi32(_mm256_mullo_epi32(diff02_23, sqrt2), rnd8),
            8
        )
    };

    let mul1_1567_23 = unsafe { _mm256_mullo_epi32(in1_23, c1567) };
    let mul3_312_23 = unsafe { _mm256_mullo_epi32(in3_23, c_312) };
    let t2_inner_23 = unsafe {
        _mm256_srai_epi32(
            _mm256_add_epi32(_mm256_sub_epi32(mul1_1567_23, mul3_312_23), rnd12),
            12
        )
    };
    let t2_23 = unsafe { _mm256_sub_epi32(t2_inner_23, in3_23) };

    let mul1_312_23 = unsafe { _mm256_mullo_epi32(in1_23, c_312) };
    let mul3_1567_23 = unsafe { _mm256_mullo_epi32(in3_23, c1567) };
    let t3_inner_23 = unsafe {
        _mm256_srai_epi32(
            _mm256_add_epi32(_mm256_add_epi32(mul1_312_23, mul3_1567_23), rnd12),
            12
        )
    };
    let t3_23 = unsafe { _mm256_add_epi32(t3_inner_23, in1_23) };

    let out0_23 = unsafe { _mm256_add_epi32(t0_23, t3_23) };
    let out1_23 = unsafe { _mm256_add_epi32(t1_23, t2_23) };
    let out2_23 = unsafe { _mm256_sub_epi32(t1_23, t2_23) };
    let out3_23 = unsafe { _mm256_sub_epi32(t0_23, t3_23) };

    let rows23_out = unsafe {
        _mm256_or_si256(
            _mm256_or_si256(
                _mm256_and_si256(out0_23, mask0),
                _mm256_and_si256(_mm256_shuffle_epi32(out1_23, 0b00_00_00_01), mask1)
            ),
            _mm256_or_si256(
                _mm256_and_si256(_mm256_shuffle_epi32(out2_23, 0b00_00_10_00), mask2),
                _mm256_and_si256(_mm256_shuffle_epi32(out3_23, 0b00_11_00_00), mask3)
            )
        )
    };

    (rows01_out, rows23_out)
}

// ============================================================================
// FFI WRAPPERS
// ============================================================================

/// FFI wrapper for 4x4 DCT_DCT 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_4x4_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    unsafe {
        inv_txfm_add_dct_dct_4x4_8bpc_avx2_inner(
            dst_ptr as *mut u8,
            dst_stride,
            coeff as *mut i16,
            eob,
            bitdepth_max,
        );
    }
}

// ============================================================================
// 4x4 WHT (Walsh-Hadamard Transform)
// ============================================================================

/// WHT4x4 - Walsh-Hadamard Transform
/// Uses the correct formula from itx_1d.rs
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn inv_txfm_add_wht_wht_4x4_8bpc_avx2_inner(
    dst: *mut u8,
    dst_stride: isize,
    coeff: *mut i16,
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

    let c_ptr = coeff;
    let mut tmp = [0i32; 16];

    // Row transform: load from column-major, store row-major
    // Row y has coeffs at: coeff[y + 0*4], coeff[y + 1*4], coeff[y + 2*4], coeff[y + 3*4]
    for y in 0..4 {
        let in0 = unsafe { *c_ptr.add(y) as i32 } >> 2;
        let in1 = unsafe { *c_ptr.add(y + 4) as i32 } >> 2;
        let in2 = unsafe { *c_ptr.add(y + 8) as i32 } >> 2;
        let in3 = unsafe { *c_ptr.add(y + 12) as i32 } >> 2;

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
        let dst_row = unsafe { dst.offset(y as isize * dst_stride) };
        for x in 0..4 {
            let d = unsafe { *dst_row.add(x) as i32 };
            let c = tmp[y * 4 + x];
            let result = iclip(d + c, 0, bitdepth_max);
            unsafe { *dst_row.add(x) = result as u8 };
        }
    }

    // Clear coefficients
    for i in 0..16 {
        unsafe { *c_ptr.add(i) = 0 };
    }
}

/// FFI wrapper for 4x4 WHT 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn inv_txfm_add_wht_wht_4x4_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    unsafe {
        inv_txfm_add_wht_wht_4x4_8bpc_avx2_inner(
            dst_ptr as *mut u8,
            dst_stride,
            coeff as *mut i16,
            eob,
            bitdepth_max,
        );
    }
}

// ============================================================================
// IDENTITY TRANSFORM HELPER (shared)
// ============================================================================

/// Identity transform - just scale by sqrt(2) and add to dst
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn inv_identity_add_4x4_8bpc_avx2(
    dst: *mut u8,
    dst_stride: isize,
    coeff: *mut i16,
    _eob: i32,
    bitdepth_max: i32,
) {
    // Identity: out = (in * 181 + 128) >> 8
    // 4x4 IDTX = identity4 on rows, identity4 on cols
    // Total: * sqrt(2) * sqrt(2) = * 2
    // Plus shift: >> 0 for 4x4, then final (+ 8) >> 4

    let c_ptr = coeff;
    let zero = unsafe { _mm_setzero_si128() };
    let max_val = unsafe { _mm_set1_epi16(bitdepth_max as i16) };

    for y in 0..4 {
        let dst_row = unsafe { dst.offset(y as isize * dst_stride) };

        // Load destination
        let d = unsafe { _mm_cvtsi32_si128(*(dst_row as *const i32)) };
        let d16 = unsafe { _mm_unpacklo_epi8(d, zero) };

        // Load coeffs for this row (column-major: y, y+4, y+8, y+12)
        let c0 = unsafe { *c_ptr.add(y) as i32 };
        let c1 = unsafe { *c_ptr.add(y + 4) as i32 };
        let c2 = unsafe { *c_ptr.add(y + 8) as i32 };
        let c3 = unsafe { *c_ptr.add(y + 12) as i32 };

        // Identity4 scale: (c * 181 + 128) >> 8, twice (row + col)
        let scale = |v: i32| -> i32 {
            let t = (v * 181 + 128) >> 8;
            (t * 181 + 128) >> 8
        };

        // Final shift: (+ 8) >> 4
        let r0 = (scale(c0) + 8) >> 4;
        let r1 = (scale(c1) + 8) >> 4;
        let r2 = (scale(c2) + 8) >> 4;
        let r3 = (scale(c3) + 8) >> 4;

        // Add to destination
        let result = unsafe { _mm_set_epi32(r3, r2, r1, r0) };
        let d32 = unsafe { _mm_cvtepi16_epi32(d16) };
        let sum = unsafe { _mm_add_epi32(d32, result) };
        let sum16 = unsafe { _mm_packs_epi32(sum, sum) };
        let clamped = unsafe { _mm_max_epi16(_mm_min_epi16(sum16, max_val), zero) };
        let packed = unsafe { _mm_packus_epi16(clamped, clamped) };

        unsafe { *(dst_row as *mut i32) = _mm_cvtsi128_si32(packed) };
    }

    // Clear coefficients
    unsafe {
        _mm_storeu_si128(coeff as *mut __m128i, _mm_setzero_si128());
        _mm_storeu_si128(coeff.add(8) as *mut __m128i, _mm_setzero_si128());
    }
}

/// FFI wrapper for 4x4 IDTX 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn inv_txfm_add_identity_identity_4x4_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    unsafe {
        inv_identity_add_4x4_8bpc_avx2(
            dst_ptr as *mut u8,
            dst_stride,
            coeff as *mut i16,
            eob,
            bitdepth_max,
        );
    }
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

/// Full 2D DCT_DCT 8x8 inverse transform with add-to-destination
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn inv_txfm_add_dct_dct_8x8_8bpc_avx2_inner(
    dst: *mut u8,
    dst_stride: isize,
    coeff: *mut i16,
    _eob: i32,
    bitdepth_max: i32,
) {
    // For 8bpc:
    // row_clip_min/max = i16::MIN/MAX (-32768, 32767)
    // col_clip_min/max = i16::MIN/MAX
    let row_clip_min = i16::MIN as i32;
    let row_clip_max = i16::MAX as i32;
    let col_clip_min = i16::MIN as i32;
    let col_clip_max = i16::MAX as i32;

    // Load coefficients and convert to i32 row-major
    // Input is column-major: coeff[y + x * 8]
    let c_ptr = coeff;
    let mut tmp = [0i32; 64];

    // Row transform
    // shift = 1 for 8x8
    let rnd = 1;
    let shift = 1;

    for y in 0..8 {
        // Load row from column-major
        for x in 0..8 {
            tmp[x] = unsafe { *c_ptr.add(y + x * 8) as i32 };
        }
        dct8_1d(&mut tmp[..8], 1, row_clip_min, row_clip_max);
        // Apply intermediate shift and store row-major
        for x in 0..8 {
            tmp[y * 8 + x] = iclip((tmp[x] + rnd) >> shift, col_clip_min, col_clip_max);
        }
    }

    // Column transform (in-place, row-major with stride 8)
    for x in 0..8 {
        dct8_1d(&mut tmp[x..], 8, col_clip_min, col_clip_max);
    }

    // Add to destination with SIMD
    let zero = unsafe { _mm_setzero_si128() };
    let max_val = unsafe { _mm_set1_epi16(bitdepth_max as i16) };
    let rnd_final = unsafe { _mm256_set1_epi32(8) };

    for y in 0..8 {
        let dst_row = unsafe { dst.offset(y as isize * dst_stride) };

        // Load destination pixels (8 bytes)
        let d = unsafe { _mm_loadl_epi64(dst_row as *const __m128i) };
        let d16 = unsafe { _mm_unpacklo_epi8(d, zero) };

        // Load and scale coefficients
        let c_lo = unsafe {
            _mm_set_epi32(
                tmp[y * 8 + 3], tmp[y * 8 + 2],
                tmp[y * 8 + 1], tmp[y * 8 + 0]
            )
        };
        let c_hi = unsafe {
            _mm_set_epi32(
                tmp[y * 8 + 7], tmp[y * 8 + 6],
                tmp[y * 8 + 5], tmp[y * 8 + 4]
            )
        };

        // Final scaling: (c + 8) >> 4
        let c_lo_256 = unsafe { _mm256_set_m128i(c_hi, c_lo) };
        let c_scaled = unsafe { _mm256_srai_epi32(_mm256_add_epi32(c_lo_256, rnd_final), 4) };

        // Pack to 16-bit
        let c_lo_scaled = unsafe { _mm256_castsi256_si128(c_scaled) };
        let c_hi_scaled = unsafe { _mm256_extracti128_si256(c_scaled, 1) };
        let c16 = unsafe { _mm_packs_epi32(c_lo_scaled, c_hi_scaled) };

        // Add to destination
        let sum = unsafe { _mm_add_epi16(d16, c16) };
        let clamped = unsafe { _mm_max_epi16(_mm_min_epi16(sum, max_val), zero) };
        let packed = unsafe { _mm_packus_epi16(clamped, clamped) };

        // Store 8 pixels
        unsafe { _mm_storel_epi64(dst_row as *mut __m128i, packed) };
    }

    // Clear coefficients
    unsafe {
        let zero256 = _mm256_setzero_si256();
        _mm256_storeu_si256(coeff as *mut __m256i, zero256);
        _mm256_storeu_si256((coeff as *mut __m256i).add(1), zero256);
        _mm256_storeu_si256((coeff as *mut __m256i).add(2), zero256);
        _mm256_storeu_si256((coeff as *mut __m256i).add(3), zero256);
    }
}

/// FFI wrapper for 8x8 DCT_DCT 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_8x8_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    unsafe {
        inv_txfm_add_dct_dct_8x8_8bpc_avx2_inner(
            dst_ptr as *mut u8,
            dst_stride,
            coeff as *mut i16,
            eob,
            bitdepth_max,
        );
    }
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

/// Full 2D DCT_DCT 16x16 inverse transform with add-to-destination
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn inv_txfm_add_dct_dct_16x16_8bpc_avx2_inner(
    dst: *mut u8,
    dst_stride: isize,
    coeff: *mut i16,
    _eob: i32,
    bitdepth_max: i32,
) {
    // For 8bpc: row_clip = col_clip = i16 range
    let row_clip_min = i16::MIN as i32;
    let row_clip_max = i16::MAX as i32;
    let col_clip_min = i16::MIN as i32;
    let col_clip_max = i16::MAX as i32;

    let c_ptr = coeff;
    let mut tmp = [0i32; 256];

    // Row transform (shift = 2 for 16x16)
    let rnd = 2;
    let shift = 2;

    for y in 0..16 {
        // Load row from column-major
        for x in 0..16 {
            tmp[x] = unsafe { *c_ptr.add(y + x * 16) as i32 };
        }
        dct16_1d(&mut tmp[..16], 1, row_clip_min, row_clip_max);
        // Apply intermediate shift and store row-major
        for x in 0..16 {
            tmp[y * 16 + x] = iclip((tmp[x] + rnd) >> shift, col_clip_min, col_clip_max);
        }
    }

    // Column transform (in-place, row-major with stride 16)
    for x in 0..16 {
        dct16_1d(&mut tmp[x..], 16, col_clip_min, col_clip_max);
    }

    // Add to destination with SIMD
    let zero = unsafe { _mm256_setzero_si256() };
    let max_val = unsafe { _mm256_set1_epi16(bitdepth_max as i16) };
    let rnd_final = unsafe { _mm256_set1_epi32(8) };

    for y in 0..16 {
        let dst_row = unsafe { dst.offset(y as isize * dst_stride) };

        // Load destination pixels (16 bytes)
        let d = unsafe { _mm_loadu_si128(dst_row as *const __m128i) };
        let d16 = unsafe { _mm256_cvtepu8_epi16(d) };

        // Load and scale coefficients (16 values)
        let c0 = unsafe {
            _mm256_set_epi32(
                tmp[y * 16 + 7], tmp[y * 16 + 6],
                tmp[y * 16 + 5], tmp[y * 16 + 4],
                tmp[y * 16 + 3], tmp[y * 16 + 2],
                tmp[y * 16 + 1], tmp[y * 16 + 0]
            )
        };
        let c1 = unsafe {
            _mm256_set_epi32(
                tmp[y * 16 + 15], tmp[y * 16 + 14],
                tmp[y * 16 + 13], tmp[y * 16 + 12],
                tmp[y * 16 + 11], tmp[y * 16 + 10],
                tmp[y * 16 + 9], tmp[y * 16 + 8]
            )
        };

        // Final scaling: (c + 8) >> 4
        let c0_scaled = unsafe { _mm256_srai_epi32(_mm256_add_epi32(c0, rnd_final), 4) };
        let c1_scaled = unsafe { _mm256_srai_epi32(_mm256_add_epi32(c1, rnd_final), 4) };

        // Pack to 16-bit
        let c16 = unsafe { _mm256_packs_epi32(c0_scaled, c1_scaled) };
        // Fix lane order after packs
        let c16 = unsafe { _mm256_permute4x64_epi64(c16, 0b11_01_10_00) };

        // Add to destination
        let sum = unsafe { _mm256_add_epi16(d16, c16) };
        let clamped = unsafe { _mm256_max_epi16(_mm256_min_epi16(sum, max_val), zero) };

        // Pack to 8-bit
        let packed = unsafe { _mm256_packus_epi16(clamped, clamped) };
        let packed = unsafe { _mm256_permute4x64_epi64(packed, 0b11_01_10_00) };

        // Store 16 pixels
        unsafe { _mm_storeu_si128(dst_row as *mut __m128i, _mm256_castsi256_si128(packed)) };
    }

    // Clear coefficients (256 * 2 = 512 bytes = 16 * 32 bytes)
    unsafe {
        let zero256 = _mm256_setzero_si256();
        for i in 0..16 {
            _mm256_storeu_si256((coeff as *mut __m256i).add(i), zero256);
        }
    }
}

/// FFI wrapper for 16x16 DCT_DCT 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_16x16_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    unsafe {
        inv_txfm_add_dct_dct_16x16_8bpc_avx2_inner(
            dst_ptr as *mut u8,
            dst_stride,
            coeff as *mut i16,
            eob,
            bitdepth_max,
        );
    }
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

        unsafe {
            inv_txfm_add_wht_wht_4x4_8bpc_avx2_inner(
                dst.as_mut_ptr(),
                stride,
                coeff.as_mut_ptr(),
                1,
                255,
            );
        }

        // Should have added DC to all pixels
        assert!(dst.iter().all(|&p| p >= 128));
        assert!(coeff.iter().all(|&c| c == 0));
    }
}
