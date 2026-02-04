//! Safe SIMD implementations of motion compensation functions
//!
//! These replace the hand-written assembly in src/x86/mc_*.asm
//!
//! Note: These functions use `#[target_feature]` instead of archmage tokens
//! because they must match the existing extern "C" signature from rav1d.
//! The runtime CPU check happens at init time in the dispatch table setup.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::include::common::bitdepth::DynPixel;
use crate::include::dav1d::headers::Rav1dFilterMode;
use crate::include::dav1d::picture::Rav1dPictureDataComponentOffset;
use crate::src::ffi_safe::FFISafe;
use crate::src::internal::COMPINTER_LEN;
use crate::src::levels::Filter2d;

/// Rounding constant for pmulhrsw: 1024 = (1 << 10)
/// pmulhrsw computes: (a * b + 16384) >> 15
/// With b=1024: (a * 1024 + 16384) >> 15 ≈ (a + 1) >> 1 (with rounding)
const PW_1024: i16 = 1024;

/// AVG operation for 8-bit pixels using AVX2
///
/// Averages two 16-bit intermediate buffers and packs to 8-bit output.
/// This matches the signature of dav1d_avg_8bpc_avx2.
///
/// # Safety
///
/// - Caller must ensure AVX2 is available (checked at dispatch time)
/// - dst_ptr must be valid for writing w*h bytes
/// - tmp1 and tmp2 must contain at least w*h elements
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn avg_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    _bitdepth_max: i32,
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let w = w as usize;
    let h = h as usize;
    let dst = dst_ptr as *mut u8;

    // Broadcast rounding constant to all lanes
    // SAFETY: AVX2 is available (checked at dispatch time via target_feature)
    let round = _mm256_set1_epi16(PW_1024);

    for row in 0..h {
        let tmp1_row = &tmp1[row * w..][..w];
        let tmp2_row = &tmp2[row * w..][..w];
        // SAFETY: dst_ptr is valid for w*h bytes, dst_stride is correct
        let dst_row =
            unsafe { std::slice::from_raw_parts_mut(dst.offset(row as isize * dst_stride), w) };

        // Process 32 pixels at a time (64 bytes of i16 input → 32 bytes of u8 output)
        let mut col = 0;
        while col + 32 <= w {
            // Load using safe_unaligned_simd
            let t1_lo_arr: &[i16; 16] = tmp1_row[col..col + 16].try_into().unwrap();
            let t1_hi_arr: &[i16; 16] = tmp1_row[col + 16..col + 32].try_into().unwrap();
            let t2_lo_arr: &[i16; 16] = tmp2_row[col..col + 16].try_into().unwrap();
            let t2_hi_arr: &[i16; 16] = tmp2_row[col + 16..col + 32].try_into().unwrap();

            // SAFETY: These loads are safe - safe_unaligned_simd handles alignment
            let t1_lo = safe_unaligned_simd::x86_64::_mm256_loadu_si256(t1_lo_arr);
            let t1_hi = safe_unaligned_simd::x86_64::_mm256_loadu_si256(t1_hi_arr);
            let t2_lo = safe_unaligned_simd::x86_64::_mm256_loadu_si256(t2_lo_arr);
            let t2_hi = safe_unaligned_simd::x86_64::_mm256_loadu_si256(t2_hi_arr);

            // Add: tmp1 + tmp2
            // SAFETY: AVX2 intrinsics are safe when target_feature is enabled
            let sum_lo = _mm256_add_epi16(t1_lo, t2_lo);
            let sum_hi = _mm256_add_epi16(t1_hi, t2_hi);

            // Multiply and round shift: (sum * 1024 + 16384) >> 15
            let avg_lo = _mm256_mulhrs_epi16(sum_lo, round);
            let avg_hi = _mm256_mulhrs_epi16(sum_hi, round);

            // Pack to unsigned bytes with saturation
            let packed = _mm256_packus_epi16(avg_lo, avg_hi);
            // Fix AVX2 lane interleaving
            let result = _mm256_permute4x64_epi64(packed, 0b11_01_10_00);

            // Store 32 bytes
            let dst_arr: &mut [u8; 32] = (&mut dst_row[col..col + 32]).try_into().unwrap();
            safe_unaligned_simd::x86_64::_mm256_storeu_si256(dst_arr, result);

            col += 32;
        }

        // Process 16 pixels at a time for remainder
        while col + 16 <= w {
            let t1_arr: &[i16; 16] = tmp1_row[col..col + 16].try_into().unwrap();
            let t2_arr: &[i16; 16] = tmp2_row[col..col + 16].try_into().unwrap();

            let t1 = safe_unaligned_simd::x86_64::_mm256_loadu_si256(t1_arr);
            let t2 = safe_unaligned_simd::x86_64::_mm256_loadu_si256(t2_arr);

            let sum = _mm256_add_epi16(t1, t2);
            let avg = _mm256_mulhrs_epi16(sum, round);

            // Pack within lanes and extract lower 128 bits
            let packed = _mm256_packus_epi16(avg, avg);
            let lo = _mm256_castsi256_si128(packed);
            let hi = _mm256_extracti128_si256(packed, 1);
            let result = _mm_unpacklo_epi64(lo, hi);

            let dst_arr: &mut [u8; 16] = (&mut dst_row[col..col + 16]).try_into().unwrap();
            safe_unaligned_simd::x86_64::_mm_storeu_si128(dst_arr, result);

            col += 16;
        }

        // Scalar fallback for remaining pixels
        while col < w {
            let sum = tmp1_row[col].wrapping_add(tmp2_row[col]);
            let avg = ((sum as i32 * 1024 + 16384) >> 15).clamp(0, 255) as u8;
            dst_row[col] = avg;
            col += 1;
        }
    }
}

/// AVG operation for 16-bit pixels using AVX2
///
/// For 10/12-bit content. Averages two 16-bit intermediate buffers.
///
/// # Safety
///
/// Same as avg_8bpc_avx2, plus bitdepth_max must be correct for the content.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn avg_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    bitdepth_max: i32,
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let w = w as usize;
    let h = h as usize;
    let dst = dst_ptr as *mut u16;
    // stride is in bytes, convert to u16 elements
    let dst_stride_elems = dst_stride / 2;

    // For 16bpc, intermediate_bits = 4, so shift = 5, round = 16400
    // rnd = (1 << 4) + 8192 * 2 = 16 + 16384 = 16400
    // Result should be clamped to [0, bitdepth_max]
    let rnd = 16400i32;
    let max = bitdepth_max as i32;

    // SAFETY: AVX2 is available (checked at dispatch time via target_feature)
    let rnd_vec = unsafe { _mm256_set1_epi32(rnd) };
    let zero = unsafe { _mm256_setzero_si256() };
    let max_vec = unsafe { _mm256_set1_epi32(max) };

    for row in 0..h {
        let tmp1_ptr = tmp1[row * w..].as_ptr();
        let tmp2_ptr = tmp2[row * w..].as_ptr();
        // SAFETY: dst_ptr is valid, stride is correct
        let dst_row = unsafe { dst.offset(row as isize * dst_stride_elems) };

        let mut col = 0usize;

        // Process 16 pixels at a time (need 32-bit arithmetic)
        while col + 16 <= w {
            // SAFETY: AVX2 intrinsics with valid pointers
            unsafe {
                // Load 16x i16 values
                let t1 = _mm256_loadu_si256(tmp1_ptr.add(col) as *const __m256i);
                let t2 = _mm256_loadu_si256(tmp2_ptr.add(col) as *const __m256i);

                // Sign-extend low 8 values to 32-bit
                let t1_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(t1));
                let t2_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(t2));

                // Sign-extend high 8 values to 32-bit
                let t1_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(t1, 1));
                let t2_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(t2, 1));

                // sum = tmp1 + tmp2 + rnd
                let sum_lo = _mm256_add_epi32(_mm256_add_epi32(t1_lo, t2_lo), rnd_vec);
                let sum_hi = _mm256_add_epi32(_mm256_add_epi32(t1_hi, t2_hi), rnd_vec);

                // >> 5
                let result_lo = _mm256_srai_epi32(sum_lo, 5);
                let result_hi = _mm256_srai_epi32(sum_hi, 5);

                // Clamp to [0, max]
                let clamped_lo = _mm256_min_epi32(_mm256_max_epi32(result_lo, zero), max_vec);
                let clamped_hi = _mm256_min_epi32(_mm256_max_epi32(result_hi, zero), max_vec);

                // Pack 32-bit to 16-bit (unsigned saturation handles the low clamp)
                let packed = _mm256_packus_epi32(clamped_lo, clamped_hi);
                // Fix lane ordering after packus
                let packed = _mm256_permute4x64_epi64(packed, 0b11011000);

                // Store 16x u16
                _mm256_storeu_si256(dst_row.add(col) as *mut __m256i, packed);
            }
            col += 16;
        }

        // Handle remaining pixels with scalar
        while col < w {
            // SAFETY: pointers are valid within bounds
            unsafe {
                let sum = *tmp1_ptr.add(col) as i32 + *tmp2_ptr.add(col) as i32;
                let val = ((sum + rnd) >> 5).clamp(0, max) as u16;
                *dst_row.add(col) = val;
            }
            col += 1;
        }
    }
}

/// SSE4.1 fallback for avg_8bpc
///
/// # Safety
///
/// Same as avg_8bpc_avx2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
pub unsafe extern "C" fn avg_8bpc_sse4(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    bitdepth_max: i32,
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    // For SSE4.1, use scalar fallback for now
    // TODO: Implement proper SSE4.1 version with _mm_* intrinsics
    // SAFETY: avg_scalar is safe to call with valid pointers
    unsafe { avg_scalar(dst_ptr, dst_stride, tmp1, tmp2, w, h, bitdepth_max, _dst) }
}

/// Scalar fallback for avg
///
/// # Safety
///
/// - dst_ptr must be valid for writing w*h bytes
/// - tmp1 and tmp2 must contain at least w*h elements
pub unsafe extern "C" fn avg_scalar(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    _bitdepth_max: i32,
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let w = w as usize;
    let h = h as usize;
    let dst = dst_ptr as *mut u8;

    for row in 0..h {
        let tmp1_row = &tmp1[row * w..][..w];
        let tmp2_row = &tmp2[row * w..][..w];
        // SAFETY: dst_ptr is valid, stride is correct
        let dst_row =
            unsafe { std::slice::from_raw_parts_mut(dst.offset(row as isize * dst_stride), w) };

        for col in 0..w {
            let sum = tmp1_row[col].wrapping_add(tmp2_row[col]);
            let avg = ((sum as i32 * 1024 + 16384) >> 15).clamp(0, 255) as u8;
            dst_row[col] = avg;
        }
    }
}

// =============================================================================
// W_AVG (Weighted Average)
// =============================================================================

/// Weighted average rounding constant for pmulhrsw
const PW_2048: i16 = 2048;

/// Weighted average for 8-bit pixels using AVX2
///
/// Computes: (tmp1 * weight + tmp2 * (16 - weight) + 128) >> 8
/// Using the optimized form from asm:
///   ((((tmp1 - tmp2) * ((weight-16) << 12)) >> 16) + tmp1 + 8) >> 4
///
/// # Safety
///
/// Same requirements as avg_8bpc_avx2, plus weight must be in [0, 16].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn w_avg_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    weight: i32,
    _bitdepth_max: i32,
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let w = w as usize;
    let h = h as usize;
    let dst = dst_ptr as *mut u8;

    // The asm uses: weight_scaled = (weight - 16) << 12 interpreted as signed
    // When weight > 7, use (weight-16) and tmp1 - tmp2
    // When weight <= 7, swap buffers and use -weight
    let (tmp1_ptr, tmp2_ptr, weight_scaled) = if weight > 7 {
        (tmp1, tmp2, ((weight - 16) << 12) as i16)
    } else {
        (tmp2, tmp1, ((-weight) << 12) as i16)
    };

    // SAFETY: AVX2 is available (checked at dispatch time via target_feature)
    let weight_vec = _mm256_set1_epi16(weight_scaled);
    let round = _mm256_set1_epi16(PW_2048);

    for row in 0..h {
        let tmp1_row = &tmp1_ptr[row * w..][..w];
        let tmp2_row = &tmp2_ptr[row * w..][..w];
        // SAFETY: dst_ptr is valid for w*h bytes, dst_stride is correct
        let dst_row =
            unsafe { std::slice::from_raw_parts_mut(dst.offset(row as isize * dst_stride), w) };

        let mut col = 0;
        while col + 32 <= w {
            let t1_lo_arr: &[i16; 16] = tmp1_row[col..col + 16].try_into().unwrap();
            let t1_hi_arr: &[i16; 16] = tmp1_row[col + 16..col + 32].try_into().unwrap();
            let t2_lo_arr: &[i16; 16] = tmp2_row[col..col + 16].try_into().unwrap();
            let t2_hi_arr: &[i16; 16] = tmp2_row[col + 16..col + 32].try_into().unwrap();

            let t1_lo = safe_unaligned_simd::x86_64::_mm256_loadu_si256(t1_lo_arr);
            let t1_hi = safe_unaligned_simd::x86_64::_mm256_loadu_si256(t1_hi_arr);
            let t2_lo = safe_unaligned_simd::x86_64::_mm256_loadu_si256(t2_lo_arr);
            let t2_hi = safe_unaligned_simd::x86_64::_mm256_loadu_si256(t2_hi_arr);

            // diff = tmp1 - tmp2
            let diff_lo = _mm256_sub_epi16(t1_lo, t2_lo);
            let diff_hi = _mm256_sub_epi16(t1_hi, t2_hi);

            // scaled = (diff * weight_scaled) >> 16 (pmulhw gives high 16 bits)
            let scaled_lo = _mm256_mulhi_epi16(diff_lo, weight_vec);
            let scaled_hi = _mm256_mulhi_epi16(diff_hi, weight_vec);

            // result = tmp1 + scaled
            let sum_lo = _mm256_add_epi16(t1_lo, scaled_lo);
            let sum_hi = _mm256_add_epi16(t1_hi, scaled_hi);

            // Final rounding: (sum + 8) >> 4 via pmulhrsw with 2048
            let avg_lo = _mm256_mulhrs_epi16(sum_lo, round);
            let avg_hi = _mm256_mulhrs_epi16(sum_hi, round);

            // Pack to bytes
            let packed = _mm256_packus_epi16(avg_lo, avg_hi);
            let result = _mm256_permute4x64_epi64(packed, 0b11_01_10_00);

            let dst_arr: &mut [u8; 32] = (&mut dst_row[col..col + 32]).try_into().unwrap();
            safe_unaligned_simd::x86_64::_mm256_storeu_si256(dst_arr, result);

            col += 32;
        }

        // Scalar fallback for remaining pixels
        while col < w {
            let a = tmp1_row[col] as i32;
            let b = tmp2_row[col] as i32;
            // Use the optimized formula
            let diff = a - b;
            let scaled = (diff * (weight_scaled as i32)) >> 16;
            let sum = a + scaled;
            let avg = ((sum + 8) >> 4).clamp(0, 255) as u8;
            dst_row[col] = avg;
            col += 1;
        }
    }
}

/// Weighted average for 16-bit pixels using AVX2
///
/// # Safety
///
/// Same as w_avg_8bpc_avx2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn w_avg_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    weight: i32,
    bitdepth_max: i32,
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let w = w as usize;
    let h = h as usize;
    let dst = dst_ptr as *mut u16;
    let dst_stride_elems = dst_stride / 2;

    // For 16bpc: intermediate_bits = 4, sh = 8, rnd = 128 + PREP_BIAS*16 = 131200
    let rnd = 131200i32;
    let max = bitdepth_max as i32;
    let inv_weight = 16 - weight;

    // SAFETY: AVX2 is available (checked at dispatch time via target_feature)
    let rnd_vec = unsafe { _mm256_set1_epi32(rnd) };
    let zero = unsafe { _mm256_setzero_si256() };
    let max_vec = unsafe { _mm256_set1_epi32(max) };
    let weight_vec = unsafe { _mm256_set1_epi32(weight) };
    let inv_weight_vec = unsafe { _mm256_set1_epi32(inv_weight) };

    for row in 0..h {
        let tmp1_ptr = tmp1[row * w..].as_ptr();
        let tmp2_ptr = tmp2[row * w..].as_ptr();
        let dst_row = unsafe { dst.offset(row as isize * dst_stride_elems) };

        let mut col = 0usize;

        // Process 8 pixels at a time (need 32-bit arithmetic)
        while col + 8 <= w {
            // SAFETY: AVX2 intrinsics with valid pointers
            unsafe {
                // Load 8x i16 values and sign-extend to 32-bit
                let t1_16 = _mm_loadu_si128(tmp1_ptr.add(col) as *const __m128i);
                let t2_16 = _mm_loadu_si128(tmp2_ptr.add(col) as *const __m128i);
                let t1 = _mm256_cvtepi16_epi32(t1_16);
                let t2 = _mm256_cvtepi16_epi32(t2_16);

                // val = a * weight + b * inv_weight + rnd
                let term1 = _mm256_mullo_epi32(t1, weight_vec);
                let term2 = _mm256_mullo_epi32(t2, inv_weight_vec);
                let sum = _mm256_add_epi32(_mm256_add_epi32(term1, term2), rnd_vec);

                // >> 8
                let result = _mm256_srai_epi32(sum, 8);

                // Clamp to [0, max]
                let clamped = _mm256_min_epi32(_mm256_max_epi32(result, zero), max_vec);

                // Pack 32-bit to 16-bit
                // packus_epi32 expects two 256-bit inputs, use the same for both halves
                let packed = _mm256_packus_epi32(clamped, clamped);
                // After packus, low 64-bits of each lane have our 8 values
                // Extract as 128-bit
                let lo128 = _mm256_castsi256_si128(packed);
                let hi128 = _mm256_extracti128_si256(packed, 1);
                // Combine: lo128 has [0-3, 0-3], hi128 has [4-7, 4-7]
                let result_128 = _mm_unpacklo_epi64(lo128, hi128);

                // Store 8x u16
                _mm_storeu_si128(dst_row.add(col) as *mut __m128i, result_128);
            }
            col += 8;
        }

        // Handle remaining pixels with scalar
        while col < w {
            // SAFETY: pointers are valid within bounds
            unsafe {
                let a = *tmp1_ptr.add(col) as i32;
                let b = *tmp2_ptr.add(col) as i32;
                let val = (a * weight + b * inv_weight + rnd) >> 8;
                *dst_row.add(col) = val.clamp(0, max) as u16;
            }
            col += 1;
        }
    }
}

/// Scalar fallback for w_avg
///
/// # Safety
///
/// Same as w_avg_8bpc_avx2.
pub unsafe extern "C" fn w_avg_scalar(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    weight: i32,
    _bitdepth_max: i32,
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let w = w as usize;
    let h = h as usize;
    let dst = dst_ptr as *mut u8;

    // For 8bpc: intermediate_bits = 4, sh = 8, rnd = 128 + PREP_BIAS*16 = 128
    let intermediate_bits = 4;
    let sh = intermediate_bits + 4;
    let rnd = (8 << intermediate_bits) + 0 * 16; // PREP_BIAS = 0 for 8bpc

    for row in 0..h {
        let tmp1_row = &tmp1[row * w..][..w];
        let tmp2_row = &tmp2[row * w..][..w];
        let dst_row =
            unsafe { std::slice::from_raw_parts_mut(dst.offset(row as isize * dst_stride), w) };

        for col in 0..w {
            let a = tmp1_row[col] as i32;
            let b = tmp2_row[col] as i32;
            let val = (a * weight + b * (16 - weight) + rnd) >> sh;
            dst_row[col] = val.clamp(0, 255) as u8;
        }
    }
}

// =============================================================================
// MASK (Per-pixel blend with mask)
// =============================================================================

/// Mask blend for 8-bit pixels using AVX2
///
/// Computes: (tmp1 * mask + tmp2 * (64 - mask) + 512) >> 10
/// Using optimized form from asm.
///
/// # Safety
///
/// - Caller must ensure AVX2 is available
/// - dst_ptr, tmp1, tmp2, mask must be valid
/// - mask values must be in [0, 64]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn mask_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask_ptr: *const u8,
    _bitdepth_max: i32,
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let w = w as usize;
    let h = h as usize;
    let dst = dst_ptr as *mut u8;

    // For 8bpc: intermediate_bits = 4, sh = 10, rnd = 512
    // Formula: (tmp1 * m + tmp2 * (64-m) + 512) >> 10
    // Rewrite: ((tmp1 - tmp2) * m + tmp2 * 64 + 512) >> 10
    // SAFETY: AVX2 is available (checked at dispatch time via target_feature)
    let rnd = unsafe { _mm256_set1_epi32(512) };

    for row in 0..h {
        let tmp1_ptr = tmp1[row * w..].as_ptr();
        let tmp2_ptr = tmp2[row * w..].as_ptr();
        // SAFETY: mask_ptr is valid for row * w + w elements
        let mask_row_ptr = unsafe { mask_ptr.add(row * w) };
        // SAFETY: dst is valid for row * stride + w elements
        let dst_row = unsafe { dst.offset(row as isize * dst_stride) };

        let mut col = 0usize;

        // Process 16 pixels at a time with AVX2
        while col + 16 <= w {
            // SAFETY: AVX2 intrinsics are safe with valid pointers, feature is enabled
            unsafe {
                // Load 16x i16 tmp values
                let t1_lo = _mm256_loadu_si256(tmp1_ptr.add(col) as *const __m256i);
                let t2_lo = _mm256_loadu_si256(tmp2_ptr.add(col) as *const __m256i);

                // Load 16 bytes of mask and zero-extend to 16x i16
                let mask_bytes = _mm_loadu_si128(mask_row_ptr.add(col) as *const __m128i);
                let mask_lo = _mm256_cvtepu8_epi16(mask_bytes);

                // Compute (tmp1 - tmp2)
                let diff = _mm256_sub_epi16(t1_lo, t2_lo);

                // Process low 8 elements (need 32-bit for tmp2*64 which can overflow 16-bit)
                let diff_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(diff));
                let mask_lo_32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(mask_lo));
                let t2_lo_32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(t2_lo));

                // Compute (tmp1-tmp2)*mask + tmp2*64 + rnd
                let prod_lo = _mm256_mullo_epi32(diff_lo, mask_lo_32);
                let t2_64_lo = _mm256_slli_epi32(t2_lo_32, 6); // tmp2 * 64 in 32-bit
                let sum_lo = _mm256_add_epi32(_mm256_add_epi32(prod_lo, t2_64_lo), rnd);

                // Process high 8 elements
                let diff_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(diff, 1));
                let mask_hi_32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(mask_lo, 1));
                let t2_hi_32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(t2_lo, 1));

                let prod_hi = _mm256_mullo_epi32(diff_hi, mask_hi_32);
                let t2_64_hi = _mm256_slli_epi32(t2_hi_32, 6);
                let sum_hi = _mm256_add_epi32(_mm256_add_epi32(prod_hi, t2_64_hi), rnd);

                // Shift right by 10
                let result_lo = _mm256_srai_epi32(sum_lo, 10);
                let result_hi = _mm256_srai_epi32(sum_hi, 10);

                // Pack back to 16-bit
                let result_16 = _mm256_packs_epi32(result_lo, result_hi);
                // Fix lane ordering after packs
                let result_16 = _mm256_permute4x64_epi64(result_16, 0b11011000);

                // Pack to 8-bit with unsigned saturation
                let result_8 = _mm256_packus_epi16(result_16, result_16);
                let result_8 = _mm256_permute4x64_epi64(result_8, 0b11011000);

                // Store 16 bytes
                _mm_storeu_si128(
                    dst_row.add(col) as *mut __m128i,
                    _mm256_castsi256_si128(result_8),
                );
            }

            col += 16;
        }

        // Handle remaining pixels with scalar
        while col < w {
            // SAFETY: pointers are valid within bounds
            unsafe {
                let a = *tmp1_ptr.add(col) as i32;
                let b = *tmp2_ptr.add(col) as i32;
                let m = *mask_row_ptr.add(col) as i32;
                let val = (a * m + b * (64 - m) + 512) >> 10;
                *dst_row.add(col) = val.clamp(0, 255) as u8;
            }
            col += 1;
        }
    }
}

/// Mask blend for 16-bit pixels
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn mask_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask_ptr: *const u8,
    bitdepth_max: i32,
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let w = w as usize;
    let h = h as usize;
    let dst = dst_ptr as *mut u16;
    let dst_stride_elems = dst_stride / 2;
    let max = bitdepth_max as i32;

    // For 16bpc: rnd = (32 << 4) + 8192 * 64 = 512 + 524288 = 524800
    let rnd = 524800i32;

    // SAFETY: AVX2 is available (checked at dispatch time via target_feature)
    let rnd_vec = unsafe { _mm256_set1_epi32(rnd) };
    let zero = unsafe { _mm256_setzero_si256() };
    let max_vec = unsafe { _mm256_set1_epi32(max) };
    let sixty_four = unsafe { _mm256_set1_epi32(64) };

    for row in 0..h {
        let tmp1_ptr = tmp1[row * w..].as_ptr();
        let tmp2_ptr = tmp2[row * w..].as_ptr();
        // SAFETY: mask_ptr is valid for row * w + w elements
        let mask_row_ptr = unsafe { mask_ptr.add(row * w) };
        // SAFETY: dst is valid for row * stride + w elements
        let dst_row = unsafe { dst.offset(row as isize * dst_stride_elems) };

        let mut col = 0usize;

        // Process 8 pixels at a time with 32-bit arithmetic
        while col + 8 <= w {
            // SAFETY: AVX2 intrinsics with valid pointers
            unsafe {
                // Load 8x i16 tmp values and sign-extend to 32-bit
                let t1_16 = _mm_loadu_si128(tmp1_ptr.add(col) as *const __m128i);
                let t2_16 = _mm_loadu_si128(tmp2_ptr.add(col) as *const __m128i);
                let t1 = _mm256_cvtepi16_epi32(t1_16);
                let t2 = _mm256_cvtepi16_epi32(t2_16);

                // Load 8 bytes of mask and zero-extend to 32-bit
                let mask_bytes = _mm_loadl_epi64(mask_row_ptr.add(col) as *const __m128i);
                let mask = _mm256_cvtepu8_epi32(mask_bytes);

                // Compute (64 - mask)
                let inv_mask = _mm256_sub_epi32(sixty_four, mask);

                // val = a * m + b * (64 - m) + rnd
                let term1 = _mm256_mullo_epi32(t1, mask);
                let term2 = _mm256_mullo_epi32(t2, inv_mask);
                let sum = _mm256_add_epi32(_mm256_add_epi32(term1, term2), rnd_vec);

                // >> 10
                let result = _mm256_srai_epi32(sum, 10);

                // Clamp to [0, max]
                let clamped = _mm256_min_epi32(_mm256_max_epi32(result, zero), max_vec);

                // Pack 32-bit to 16-bit
                let packed = _mm256_packus_epi32(clamped, clamped);
                let lo128 = _mm256_castsi256_si128(packed);
                let hi128 = _mm256_extracti128_si256(packed, 1);
                let result_128 = _mm_unpacklo_epi64(lo128, hi128);

                // Store 8x u16
                _mm_storeu_si128(dst_row.add(col) as *mut __m128i, result_128);
            }
            col += 8;
        }

        // Handle remaining pixels with scalar
        while col < w {
            // SAFETY: pointers are valid within bounds
            unsafe {
                let a = *tmp1_ptr.add(col) as i32;
                let b = *tmp2_ptr.add(col) as i32;
                let m = *mask_row_ptr.add(col) as i32;
                let val = (a * m + b * (64 - m) + rnd) >> 10;
                *dst_row.add(col) = val.clamp(0, max) as u16;
            }
            col += 1;
        }
    }
}

/// Scalar fallback for mask
pub unsafe extern "C" fn mask_scalar(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask_ptr: *const u8,
    _bitdepth_max: i32,
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let w = w as usize;
    let h = h as usize;
    let dst = dst_ptr as *mut u8;

    // For 8bpc: intermediate_bits = 4, sh = 10, rnd = 512
    let intermediate_bits = 4;
    let sh = intermediate_bits + 6;
    let rnd = (32 << intermediate_bits) + 0 * 64;

    for row in 0..h {
        let tmp1_row = &tmp1[row * w..][..w];
        let tmp2_row = &tmp2[row * w..][..w];
        let mask_row = unsafe { std::slice::from_raw_parts(mask_ptr.add(row * w), w) };
        let dst_row =
            unsafe { std::slice::from_raw_parts_mut(dst.offset(row as isize * dst_stride), w) };

        for col in 0..w {
            let a = tmp1_row[col] as i32;
            let b = tmp2_row[col] as i32;
            let m = mask_row[col] as i32;
            let val = (a * m + b * (64 - m) + rnd) >> sh;
            dst_row[col] = val.clamp(0, 255) as u8;
        }
    }
}

// =============================================================================
// BLEND (Pixel-level blend with mask)
// =============================================================================

use crate::src::internal::{SCRATCH_INTER_INTRA_BUF_LEN, SCRATCH_LAP_LEN};

/// Blend pixels using per-pixel mask
///
/// Computes: dst = (dst * (64 - mask) + tmp * mask + 32) >> 6
///
/// # Safety
///
/// - dst_ptr must be valid for reading and writing w*h pixels
/// - tmp and mask must be valid for reading w*h elements
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn blend_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp: *const [DynPixel; SCRATCH_INTER_INTRA_BUF_LEN],
    w: i32,
    h: i32,
    mask_ptr: *const u8,
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let w = w as usize;
    let h = h as usize;
    let dst = dst_ptr as *mut u8;
    let tmp = tmp as *const u8;

    // Constants for blend formula: (dst * (64-m) + tmp * m + 32) >> 6
    // SAFETY: AVX2 is available
    let sixty_four = unsafe { _mm256_set1_epi16(64) };
    let rnd = unsafe { _mm256_set1_epi16(32) };

    for row in 0..h {
        // SAFETY: pointers valid for row * stride + w
        let dst_row = unsafe { dst.offset(row as isize * dst_stride) };
        let tmp_row = unsafe { tmp.add(row * w) };
        let mask_row = unsafe { mask_ptr.add(row * w) };

        let mut col = 0usize;

        // Process 16 pixels at a time with AVX2
        while col + 16 <= w {
            // SAFETY: AVX2 intrinsics with valid pointers
            unsafe {
                // Load 16 bytes of dst, tmp, mask
                let dst_bytes = _mm_loadu_si128(dst_row.add(col) as *const __m128i);
                let tmp_bytes = _mm_loadu_si128(tmp_row.add(col) as *const __m128i);
                let mask_bytes = _mm_loadu_si128(mask_row.add(col) as *const __m128i);

                // Zero-extend to 16-bit (max value 255 * 64 = 16320 fits in 16-bit)
                let dst_16 = _mm256_cvtepu8_epi16(dst_bytes);
                let tmp_16 = _mm256_cvtepu8_epi16(tmp_bytes);
                let mask_16 = _mm256_cvtepu8_epi16(mask_bytes);

                // Compute (64 - mask)
                let inv_mask = _mm256_sub_epi16(sixty_four, mask_16);

                // dst * (64-m) + tmp * m + 32
                let term1 = _mm256_mullo_epi16(dst_16, inv_mask);
                let term2 = _mm256_mullo_epi16(tmp_16, mask_16);
                let sum = _mm256_add_epi16(_mm256_add_epi16(term1, term2), rnd);

                // >> 6
                let result_16 = _mm256_srli_epi16(sum, 6);

                // Pack to 8-bit with unsigned saturation
                let result_8 = _mm256_packus_epi16(result_16, result_16);
                let result_8 = _mm256_permute4x64_epi64(result_8, 0b11011000);

                // Store 16 bytes
                _mm_storeu_si128(
                    dst_row.add(col) as *mut __m128i,
                    _mm256_castsi256_si128(result_8),
                );
            }
            col += 16;
        }

        // Handle remaining pixels with scalar
        while col < w {
            // SAFETY: pointers valid
            unsafe {
                let a = *dst_row.add(col) as u32;
                let b = *tmp_row.add(col) as u32;
                let m = *mask_row.add(col) as u32;
                let val = (a * (64 - m) + b * m + 32) >> 6;
                *dst_row.add(col) = val as u8;
            }
            col += 1;
        }
    }
}

/// Blend pixels for 16-bit
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn blend_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp: *const [DynPixel; SCRATCH_INTER_INTRA_BUF_LEN],
    w: i32,
    h: i32,
    mask_ptr: *const u8,
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let w = w as usize;
    let h = h as usize;
    let dst = dst_ptr as *mut u16;
    let dst_stride_elems = dst_stride / 2;
    let tmp = tmp as *const u16;

    // SAFETY: AVX2 is available (checked at dispatch time via target_feature)
    let rnd = unsafe { _mm256_set1_epi32(32) };
    let sixty_four = unsafe { _mm256_set1_epi32(64) };

    for row in 0..h {
        // SAFETY: dst is valid for row * stride + w elements
        let dst_row = unsafe { dst.offset(row as isize * dst_stride_elems) };
        let tmp_row = unsafe { tmp.add(row * w) };
        let mask_row = unsafe { mask_ptr.add(row * w) };

        let mut col = 0usize;

        // Process 8 pixels at a time with 32-bit arithmetic
        while col + 8 <= w {
            // SAFETY: AVX2 intrinsics with valid pointers
            unsafe {
                // Load 8x u16 dst and tmp, zero-extend to 32-bit
                let dst_16 = _mm_loadu_si128(dst_row.add(col) as *const __m128i);
                let tmp_16 = _mm_loadu_si128(tmp_row.add(col) as *const __m128i);
                let dst_32 = _mm256_cvtepu16_epi32(dst_16);
                let tmp_32 = _mm256_cvtepu16_epi32(tmp_16);

                // Load 8 bytes of mask and zero-extend to 32-bit
                let mask_bytes = _mm_loadl_epi64(mask_row.add(col) as *const __m128i);
                let mask_32 = _mm256_cvtepu8_epi32(mask_bytes);

                // Compute (64 - mask)
                let inv_mask = _mm256_sub_epi32(sixty_four, mask_32);

                // val = dst * (64-m) + tmp * m + 32
                let term1 = _mm256_mullo_epi32(dst_32, inv_mask);
                let term2 = _mm256_mullo_epi32(tmp_32, mask_32);
                let sum = _mm256_add_epi32(_mm256_add_epi32(term1, term2), rnd);

                // >> 6
                let result = _mm256_srli_epi32(sum, 6);

                // Pack 32-bit to 16-bit (values fit in u16)
                let packed = _mm256_packus_epi32(result, result);
                let lo128 = _mm256_castsi256_si128(packed);
                let hi128 = _mm256_extracti128_si256(packed, 1);
                let result_128 = _mm_unpacklo_epi64(lo128, hi128);

                // Store 8x u16
                _mm_storeu_si128(dst_row.add(col) as *mut __m128i, result_128);
            }
            col += 8;
        }

        // Handle remaining pixels with scalar
        while col < w {
            // SAFETY: pointers valid
            unsafe {
                let a = *dst_row.add(col) as u32;
                let b = *tmp_row.add(col) as u32;
                let m = *mask_row.add(col) as u32;
                let val = (a * (64 - m) + b * m + 32) >> 6;
                *dst_row.add(col) = val as u16;
            }
            col += 1;
        }
    }
}

// =============================================================================
// BLEND_V / BLEND_H (Directional blend for OBMC)
// =============================================================================

use crate::src::tables::dav1d_mc_subpel_filters;
use crate::src::tables::dav1d_obmc_masks;

/// Vertical blend (overlapped block motion compensation)
///
/// Uses predefined obmc_masks table for blend weights.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn blend_v_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp: *const [DynPixel; SCRATCH_LAP_LEN],
    w: i32,
    h: i32,
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let w = w as usize;
    let h = h as usize;
    let dst = dst_ptr as *mut u8;
    let tmp = tmp as *const u8;
    let obmc_mask = &dav1d_obmc_masks.0[h..];

    // SAFETY: AVX2 is available
    let rnd = unsafe { _mm256_set1_epi16(32) };
    let sixty_four = unsafe { _mm256_set1_epi16(64) };

    for row in 0..h {
        // SAFETY: pointers valid for row * stride + w
        let dst_row = unsafe { dst.offset(row as isize * dst_stride) };
        let tmp_row = unsafe { tmp.add(row * w) };
        let m = obmc_mask[row];

        // Broadcast mask value for the whole row
        // SAFETY: AVX2 is available
        let mask_16 = unsafe { _mm256_set1_epi16(m as i16) };
        let inv_mask = unsafe { _mm256_sub_epi16(sixty_four, mask_16) };

        let mut col = 0usize;

        // Process 16 pixels at a time
        while col + 16 <= w {
            // SAFETY: AVX2 intrinsics with valid pointers
            unsafe {
                let dst_bytes = _mm_loadu_si128(dst_row.add(col) as *const __m128i);
                let tmp_bytes = _mm_loadu_si128(tmp_row.add(col) as *const __m128i);

                let dst_16 = _mm256_cvtepu8_epi16(dst_bytes);
                let tmp_16 = _mm256_cvtepu8_epi16(tmp_bytes);

                let term1 = _mm256_mullo_epi16(dst_16, inv_mask);
                let term2 = _mm256_mullo_epi16(tmp_16, mask_16);
                let sum = _mm256_add_epi16(_mm256_add_epi16(term1, term2), rnd);
                let result_16 = _mm256_srli_epi16(sum, 6);

                let result_8 = _mm256_packus_epi16(result_16, result_16);
                let result_8 = _mm256_permute4x64_epi64(result_8, 0b11011000);

                _mm_storeu_si128(
                    dst_row.add(col) as *mut __m128i,
                    _mm256_castsi256_si128(result_8),
                );
            }
            col += 16;
        }

        // Handle remaining pixels
        while col < w {
            // SAFETY: pointers valid
            unsafe {
                let a = *dst_row.add(col) as u32;
                let b = *tmp_row.add(col) as u32;
                let val = (a * (64 - m as u32) + b * m as u32 + 32) >> 6;
                *dst_row.add(col) = val as u8;
            }
            col += 1;
        }
    }
}

/// Horizontal blend (overlapped block motion compensation)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn blend_h_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp: *const [DynPixel; SCRATCH_LAP_LEN],
    w: i32,
    h: i32,
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let w = w as usize;
    let h = h as usize;
    let dst = dst_ptr as *mut u8;
    let tmp = tmp as *const u8;
    let obmc_mask = &dav1d_obmc_masks.0[w..];

    // SAFETY: AVX2 is available
    let rnd = unsafe { _mm256_set1_epi16(32) };
    let sixty_four = unsafe { _mm256_set1_epi16(64) };

    for row in 0..h {
        // SAFETY: pointers valid for row * stride + w
        let dst_row = unsafe { dst.offset(row as isize * dst_stride) };
        let tmp_row = unsafe { tmp.add(row * w) };

        let mut col = 0usize;

        // Process 16 pixels at a time
        while col + 16 <= w {
            // SAFETY: AVX2 intrinsics with valid pointers
            unsafe {
                let dst_bytes = _mm_loadu_si128(dst_row.add(col) as *const __m128i);
                let tmp_bytes = _mm_loadu_si128(tmp_row.add(col) as *const __m128i);

                // Load mask bytes for these columns
                let mask_bytes = _mm_loadu_si128(obmc_mask[col..].as_ptr() as *const __m128i);

                let dst_16 = _mm256_cvtepu8_epi16(dst_bytes);
                let tmp_16 = _mm256_cvtepu8_epi16(tmp_bytes);
                let mask_16 = _mm256_cvtepu8_epi16(mask_bytes);
                let inv_mask = _mm256_sub_epi16(sixty_four, mask_16);

                let term1 = _mm256_mullo_epi16(dst_16, inv_mask);
                let term2 = _mm256_mullo_epi16(tmp_16, mask_16);
                let sum = _mm256_add_epi16(_mm256_add_epi16(term1, term2), rnd);
                let result_16 = _mm256_srli_epi16(sum, 6);

                let result_8 = _mm256_packus_epi16(result_16, result_16);
                let result_8 = _mm256_permute4x64_epi64(result_8, 0b11011000);

                _mm_storeu_si128(
                    dst_row.add(col) as *mut __m128i,
                    _mm256_castsi256_si128(result_8),
                );
            }
            col += 16;
        }

        // Handle remaining pixels
        while col < w {
            // SAFETY: pointers valid
            unsafe {
                let a = *dst_row.add(col) as u32;
                let b = *tmp_row.add(col) as u32;
                let m = obmc_mask[col] as u32;
                let val = (a * (64 - m) + b * m + 32) >> 6;
                *dst_row.add(col) = val as u8;
            }
            col += 1;
        }
    }
}

/// 16-bit blend_v
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn blend_v_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp: *const [DynPixel; SCRATCH_LAP_LEN],
    w: i32,
    h: i32,
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let w = w as usize;
    let h = h as usize;
    let dst = dst_ptr as *mut u16;
    let dst_stride_elems = dst_stride / 2;
    let tmp = tmp as *const u16;
    let obmc_mask = &dav1d_obmc_masks.0[h..];

    // SAFETY: AVX2 is available (checked at dispatch time via target_feature)
    let rnd = unsafe { _mm256_set1_epi32(32) };
    let sixty_four = unsafe { _mm256_set1_epi32(64) };

    for row in 0..h {
        let dst_row = unsafe { dst.offset(row as isize * dst_stride_elems) };
        let tmp_row = unsafe { tmp.add(row * w) };
        let m = obmc_mask[row] as u32;

        // Broadcast mask value for the whole row
        // SAFETY: AVX2 is available
        let mask_32 = unsafe { _mm256_set1_epi32(m as i32) };
        let inv_mask = unsafe { _mm256_sub_epi32(sixty_four, mask_32) };

        let mut col = 0usize;

        // Process 8 pixels at a time with 32-bit arithmetic
        while col + 8 <= w {
            // SAFETY: AVX2 intrinsics with valid pointers
            unsafe {
                // Load 8x u16 dst and tmp, zero-extend to 32-bit
                let dst_16 = _mm_loadu_si128(dst_row.add(col) as *const __m128i);
                let tmp_16 = _mm_loadu_si128(tmp_row.add(col) as *const __m128i);
                let dst_32 = _mm256_cvtepu16_epi32(dst_16);
                let tmp_32 = _mm256_cvtepu16_epi32(tmp_16);

                // val = dst * (64-m) + tmp * m + 32
                let term1 = _mm256_mullo_epi32(dst_32, inv_mask);
                let term2 = _mm256_mullo_epi32(tmp_32, mask_32);
                let sum = _mm256_add_epi32(_mm256_add_epi32(term1, term2), rnd);

                // >> 6
                let result = _mm256_srli_epi32(sum, 6);

                // Pack 32-bit to 16-bit
                let packed = _mm256_packus_epi32(result, result);
                let lo128 = _mm256_castsi256_si128(packed);
                let hi128 = _mm256_extracti128_si256(packed, 1);
                let result_128 = _mm_unpacklo_epi64(lo128, hi128);

                // Store 8x u16
                _mm_storeu_si128(dst_row.add(col) as *mut __m128i, result_128);
            }
            col += 8;
        }

        // Handle remaining pixels with scalar
        while col < w {
            // SAFETY: pointers valid
            unsafe {
                let a = *dst_row.add(col) as u32;
                let b = *tmp_row.add(col) as u32;
                let val = (a * (64 - m) + b * m + 32) >> 6;
                *dst_row.add(col) = val as u16;
            }
            col += 1;
        }
    }
}

/// 16-bit blend_h
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn blend_h_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp: *const [DynPixel; SCRATCH_LAP_LEN],
    w: i32,
    h: i32,
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let w = w as usize;
    let h = h as usize;
    let dst = dst_ptr as *mut u16;
    let dst_stride_elems = dst_stride / 2;
    let tmp = tmp as *const u16;
    let obmc_mask = &dav1d_obmc_masks.0[w..];

    // SAFETY: AVX2 is available (checked at dispatch time via target_feature)
    let rnd = unsafe { _mm256_set1_epi32(32) };
    let sixty_four = unsafe { _mm256_set1_epi32(64) };

    for row in 0..h {
        let dst_row = unsafe { dst.offset(row as isize * dst_stride_elems) };
        let tmp_row = unsafe { tmp.add(row * w) };

        let mut col = 0usize;

        // Process 8 pixels at a time with 32-bit arithmetic
        while col + 8 <= w {
            // SAFETY: AVX2 intrinsics with valid pointers
            unsafe {
                // Load 8x u16 dst and tmp, zero-extend to 32-bit
                let dst_16 = _mm_loadu_si128(dst_row.add(col) as *const __m128i);
                let tmp_16 = _mm_loadu_si128(tmp_row.add(col) as *const __m128i);
                let dst_32 = _mm256_cvtepu16_epi32(dst_16);
                let tmp_32 = _mm256_cvtepu16_epi32(tmp_16);

                // Load 8 bytes of mask and zero-extend to 32-bit
                let mask_bytes = _mm_loadl_epi64(obmc_mask[col..].as_ptr() as *const __m128i);
                let mask_32 = _mm256_cvtepu8_epi32(mask_bytes);
                let inv_mask = _mm256_sub_epi32(sixty_four, mask_32);

                // val = dst * (64-m) + tmp * m + 32
                let term1 = _mm256_mullo_epi32(dst_32, inv_mask);
                let term2 = _mm256_mullo_epi32(tmp_32, mask_32);
                let sum = _mm256_add_epi32(_mm256_add_epi32(term1, term2), rnd);

                // >> 6
                let result = _mm256_srli_epi32(sum, 6);

                // Pack 32-bit to 16-bit
                let packed = _mm256_packus_epi32(result, result);
                let lo128 = _mm256_castsi256_si128(packed);
                let hi128 = _mm256_extracti128_si256(packed, 1);
                let result_128 = _mm_unpacklo_epi64(lo128, hi128);

                // Store 8x u16
                _mm_storeu_si128(dst_row.add(col) as *mut __m128i, result_128);
            }
            col += 8;
        }

        // Handle remaining pixels with scalar
        while col < w {
            // SAFETY: pointers valid
            unsafe {
                let a = *dst_row.add(col) as u32;
                let b = *tmp_row.add(col) as u32;
                let m = obmc_mask[col] as u32;
                let val = (a * (64 - m) + b * m + 32) >> 6;
                *dst_row.add(col) = val as u16;
            }
            col += 1;
        }
    }
}

// =============================================================================
// 8-TAP FILTERS (MC/MCT)
// =============================================================================

/// Stride for intermediate buffer in 8-tap filtering
const MID_STRIDE: usize = 128;

/// Get filter coefficients for a given subpixel position and filter type
#[inline]
fn get_filter(m: usize, d: usize, filter_idx: usize) -> Option<&'static [i8; 8]> {
    if m == 0 {
        return None;
    }
    let m = m - 1;
    let i = if d > 4 {
        filter_idx
    } else {
        3 + (filter_idx & 1)
    };
    Some(&dav1d_mc_subpel_filters.0[i][m])
}

/// Horizontal 8-tap filter for a row of 8bpc pixels
///
/// Processes `w` pixels starting at `src`, writing to `dst` (i16 intermediate)
/// Formula: sum(coeff[i] * src[x + i - 3]) for i in 0..8, then round and shift
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn h_filter_8tap_8bpc_avx2(
    dst: *mut i16,
    src: *const u8,
    w: usize,
    filter: &[i8; 8],
    sh: u8,
) {
    // SAFETY: All operations inside this block require AVX2 which is guaranteed
    // by the target_feature attribute, and pointer operations are valid per caller contract.
    unsafe {
        // For horizontal filtering, we need to load 8 consecutive pixels for each output
        // The source pointer is already offset by -3 (pointing to tap 0)

        // Broadcast filter coefficients
        // We'll use _mm256_maddubs_epi16 which does a[0]*b[0]+a[1]*b[1] for pairs
        // So we need to arrange coefficients for this: [c0,c1,c2,c3,c4,c5,c6,c7] repeated
        let coeff_01 =
            _mm256_set1_epi16(((filter[1] as u8 as i16) << 8) | (filter[0] as u8 as i16));
        let coeff_23 =
            _mm256_set1_epi16(((filter[3] as u8 as i16) << 8) | (filter[2] as u8 as i16));
        let coeff_45 =
            _mm256_set1_epi16(((filter[5] as u8 as i16) << 8) | (filter[4] as u8 as i16));
        let coeff_67 =
            _mm256_set1_epi16(((filter[7] as u8 as i16) << 8) | (filter[6] as u8 as i16));

        let rnd = _mm256_set1_epi16((1i16 << sh) >> 1);

        let mut col = 0usize;

        // Process 16 pixels at a time
        while col + 16 <= w {
            // Load source bytes - we need 8 bytes for each output pixel, offset by tap position
            let s = src.add(col);

            // Load bytes at various offsets for the 8-tap filter
            let src_0_15 = _mm_loadu_si128(s as *const __m128i);
            let src_1_16 = _mm_loadu_si128(s.add(1) as *const __m128i);
            let src_2_17 = _mm_loadu_si128(s.add(2) as *const __m128i);
            let src_3_18 = _mm_loadu_si128(s.add(3) as *const __m128i);
            let src_4_19 = _mm_loadu_si128(s.add(4) as *const __m128i);
            let src_5_20 = _mm_loadu_si128(s.add(5) as *const __m128i);
            let src_6_21 = _mm_loadu_si128(s.add(6) as *const __m128i);
            let src_7_22 = _mm_loadu_si128(s.add(7) as *const __m128i);

            // Interleave bytes for maddubs
            let p01_lo = _mm_unpacklo_epi8(src_0_15, src_1_16);
            let p01_hi = _mm_unpackhi_epi8(src_0_15, src_1_16);
            let p01 = _mm256_set_m128i(p01_hi, p01_lo);

            let p23_lo = _mm_unpacklo_epi8(src_2_17, src_3_18);
            let p23_hi = _mm_unpackhi_epi8(src_2_17, src_3_18);
            let p23 = _mm256_set_m128i(p23_hi, p23_lo);

            let p45_lo = _mm_unpacklo_epi8(src_4_19, src_5_20);
            let p45_hi = _mm_unpackhi_epi8(src_4_19, src_5_20);
            let p45 = _mm256_set_m128i(p45_hi, p45_lo);

            let p67_lo = _mm_unpacklo_epi8(src_6_21, src_7_22);
            let p67_hi = _mm_unpackhi_epi8(src_6_21, src_7_22);
            let p67 = _mm256_set_m128i(p67_hi, p67_lo);

            // Multiply-add pairs
            let ma01 = _mm256_maddubs_epi16(p01, coeff_01);
            let ma23 = _mm256_maddubs_epi16(p23, coeff_23);
            let ma45 = _mm256_maddubs_epi16(p45, coeff_45);
            let ma67 = _mm256_maddubs_epi16(p67, coeff_67);

            // Sum all contributions
            let mut sum = _mm256_add_epi16(ma01, ma23);
            sum = _mm256_add_epi16(sum, ma45);
            sum = _mm256_add_epi16(sum, ma67);

            // Add rounding and shift
            let shift_count = _mm_cvtsi32_si128(sh as i32);
            let result = _mm256_sra_epi16(_mm256_add_epi16(sum, rnd), shift_count);

            // Store 16 i16 values
            _mm256_storeu_si256(dst.add(col) as *mut __m256i, result);

            col += 16;
        }

        // Scalar fallback for remaining pixels
        while col < w {
            let s = src.add(col);
            let mut sum = 0i32;
            for i in 0..8 {
                sum += filter[i] as i32 * (*s.add(i)) as i32;
            }
            *dst.add(col) = ((sum + ((1 << sh) >> 1)) >> sh) as i16;
            col += 1;
        }
    }
}

/// Vertical 8-tap filter from intermediate buffer to output
///
/// Processes `w` pixels for one row, reading from `mid` (8 rows), writing to `dst`
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn v_filter_8tap_8bpc_avx2(
    dst: *mut u8,
    mid: &[[i16; MID_STRIDE]],
    w: usize,
    filter: &[i8; 8],
    sh: u8,
    max: i32,
) {
    // SAFETY: All operations inside this block require AVX2 which is guaranteed
    // by the target_feature attribute, and pointer operations are valid per caller contract.
    unsafe {
        let rnd = _mm256_set1_epi32((1i32 << sh) >> 1);
        let zero = _mm256_setzero_si256();
        let _max_vec = _mm256_set1_epi16(max as i16);

        // Broadcast filter coefficients to 32-bit for multiplication
        let c0 = _mm256_set1_epi32(filter[0] as i32);
        let c1 = _mm256_set1_epi32(filter[1] as i32);
        let c2 = _mm256_set1_epi32(filter[2] as i32);
        let c3 = _mm256_set1_epi32(filter[3] as i32);
        let c4 = _mm256_set1_epi32(filter[4] as i32);
        let c5 = _mm256_set1_epi32(filter[5] as i32);
        let c6 = _mm256_set1_epi32(filter[6] as i32);
        let c7 = _mm256_set1_epi32(filter[7] as i32);

        let mut col = 0usize;

        // Process 8 pixels at a time using 32-bit arithmetic
        while col + 8 <= w {
            // Load 8 i16 values from each of 8 rows and convert to i32
            let m0 =
                _mm256_cvtepi16_epi32(_mm_loadu_si128(mid[0][col..].as_ptr() as *const __m128i));
            let m1 =
                _mm256_cvtepi16_epi32(_mm_loadu_si128(mid[1][col..].as_ptr() as *const __m128i));
            let m2 =
                _mm256_cvtepi16_epi32(_mm_loadu_si128(mid[2][col..].as_ptr() as *const __m128i));
            let m3 =
                _mm256_cvtepi16_epi32(_mm_loadu_si128(mid[3][col..].as_ptr() as *const __m128i));
            let m4 =
                _mm256_cvtepi16_epi32(_mm_loadu_si128(mid[4][col..].as_ptr() as *const __m128i));
            let m5 =
                _mm256_cvtepi16_epi32(_mm_loadu_si128(mid[5][col..].as_ptr() as *const __m128i));
            let m6 =
                _mm256_cvtepi16_epi32(_mm_loadu_si128(mid[6][col..].as_ptr() as *const __m128i));
            let m7 =
                _mm256_cvtepi16_epi32(_mm_loadu_si128(mid[7][col..].as_ptr() as *const __m128i));

            // Multiply each row by its coefficient and accumulate
            let mut sum = _mm256_mullo_epi32(m0, c0);
            sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(m1, c1));
            sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(m2, c2));
            sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(m3, c3));
            sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(m4, c4));
            sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(m5, c5));
            sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(m6, c6));
            sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(m7, c7));

            // Add rounding and shift
            let shift_count = _mm_cvtsi32_si128(sh as i32);
            let shifted = _mm256_sra_epi32(_mm256_add_epi32(sum, rnd), shift_count);

            // Clamp to [0, max] and pack to 16-bit
            let clamped = _mm256_min_epi32(_mm256_max_epi32(shifted, zero), _mm256_set1_epi32(max));

            // Pack 32-bit to 16-bit, then 16-bit to 8-bit
            let packed16 = _mm256_packs_epi32(clamped, clamped);
            let packed16 = _mm256_permute4x64_epi64(packed16, 0b11011000);
            let packed8 = _mm256_packus_epi16(packed16, packed16);

            // Store 8 bytes
            let result_64 = _mm256_extract_epi64(packed8, 0);
            std::ptr::copy_nonoverlapping(&result_64 as *const i64 as *const u8, dst.add(col), 8);

            col += 8;
        }

        // Scalar fallback
        while col < w {
            let mut sum = 0i32;
            for i in 0..8 {
                sum += filter[i] as i32 * mid[i][col] as i32;
            }
            let val = ((sum + ((1 << sh) >> 1)) >> sh).clamp(0, max);
            *dst.add(col) = val as u8;
            col += 1;
        }
    }
}

// =============================================================================
// 8-TAP PUT FUNCTIONS (mc)
// =============================================================================

/// Get filter coefficients for a given subpixel position and filter type
///
/// Returns None if m == 0 (integer position, no filtering needed)
#[inline]
fn get_filter_coeff(m: usize, d: usize, filter_type: Rav1dFilterMode) -> Option<&'static [i8; 8]> {
    let m = m.checked_sub(1)?;
    let i = if d > 4 {
        filter_type as u8
    } else {
        3 + (filter_type as u8 & 1)
    };
    Some(&dav1d_mc_subpel_filters.0[i as usize][m])
}

/// Generic 8-tap put function for 8bpc
///
/// This handles all 4 cases:
/// 1. H+V filtering (mx != 0 && my != 0)
/// 2. H-only filtering (mx != 0 && my == 0)
/// 3. V-only filtering (mx == 0 && my != 0)
/// 4. Simple copy (mx == 0 && my == 0)
///
/// # Safety
///
/// - Caller must ensure AVX2 is available
/// - dst_ptr must be valid for writing w*h bytes
/// - src_ptr must be valid for reading (w+7)*(h+7) bytes (with proper padding)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn put_8tap_8bpc_avx2_impl(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    h_filter_type: Rav1dFilterMode,
    v_filter_type: Rav1dFilterMode,
) {
    let w = w as usize;
    let h = h as usize;
    let mx = mx as usize;
    let my = my as usize;
    let dst = dst_ptr as *mut u8;
    let src = src_ptr as *const u8;

    // For 8bpc: intermediate_bits = 4
    let intermediate_bits = 4u8;

    let fh = get_filter_coeff(mx, w, h_filter_type);
    let fv = get_filter_coeff(my, h, v_filter_type);

    // SAFETY: All operations require AVX2 which is guaranteed by target_feature
    unsafe {
        match (fh, fv) {
            (Some(fh), Some(fv)) => {
                // Case 1: Both H and V filtering
                // First pass: horizontal filter to intermediate buffer
                let tmp_h = h + 7;
                let mut mid = [[0i16; MID_STRIDE]; 135];

                for y in 0..tmp_h {
                    let src_row = src.offset((y as isize - 3) * src_stride);
                    h_filter_8tap_8bpc_avx2(
                        mid[y].as_mut_ptr(),
                        src_row.sub(3), // Offset by -3 for tap 0
                        w,
                        fh,
                        6 - intermediate_bits,
                    );
                }

                // Second pass: vertical filter to output
                for y in 0..h {
                    let dst_row = dst.offset(y as isize * dst_stride);
                    v_filter_8tap_8bpc_avx2(
                        dst_row,
                        &mid[y..],
                        w,
                        fv,
                        6 + intermediate_bits,
                        255,
                    );
                }
            }
            (Some(fh), None) => {
                // Case 2: H-only filtering
                // intermediate_rnd = 32 + (1 << (6 - intermediate_bits)) >> 1 = 32 + 2 = 34
                // But for direct output we need different rounding
                // sh = 6, rnd = 32 for 8bpc
                for y in 0..h {
                    let src_row = src.offset(y as isize * src_stride);
                    let dst_row = dst.offset(y as isize * dst_stride);

                    // Use horizontal filter with sh=6 for direct output
                    // Then clamp to [0, 255]
                    let mut tmp = [0i16; MID_STRIDE];
                    h_filter_8tap_8bpc_avx2(tmp.as_mut_ptr(), src_row.sub(3), w, fh, 6);

                    // Copy and clamp to output
                    for x in 0..w {
                        *dst_row.add(x) = tmp[x].clamp(0, 255) as u8;
                    }
                }
            }
            (None, Some(fv)) => {
                // Case 3: V-only filtering
                // We need to read 8 rows of source for each output row
                for y in 0..h {
                    let dst_row = dst.offset(y as isize * dst_stride);

                    // Build intermediate buffer from 8 source rows
                    let mut mid = [[0i16; MID_STRIDE]; 8];
                    for i in 0..8 {
                        let src_row = src.offset((y as isize + i as isize - 3) * src_stride);
                        for x in 0..w {
                            mid[i][x] = *src_row.add(x) as i16;
                        }
                    }

                    v_filter_8tap_8bpc_avx2(dst_row, &mid, w, fv, 6, 255);
                }
            }
            (None, None) => {
                // Case 4: Simple copy
                for y in 0..h {
                    let src_row = src.offset(y as isize * src_stride);
                    let dst_row = dst.offset(y as isize * dst_stride);
                    std::ptr::copy_nonoverlapping(src_row, dst_row, w);
                }
            }
        }
    }
}

/// Generic put_8tap function wrapper with Filter2d const generic
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn put_8tap_8bpc_avx2<const FILTER: usize>(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    _bitdepth_max: i32,
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
    _src: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let filter = Filter2d::from_repr(FILTER).unwrap();
    let (h_filter, v_filter) = filter.hv();

    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
        put_8tap_8bpc_avx2_impl(
            dst_ptr, dst_stride, src_ptr, src_stride, w, h, mx, my, h_filter, v_filter,
        );
    }
}

// Specific filter type wrappers for dispatch table
// These are needed because decl_fn_safe! requires concrete functions, not generics

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn put_8tap_regular_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
    src: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
    put_8tap_8bpc_avx2::<{ Filter2d::Regular8Tap as usize }>(
        dst_ptr,
        dst_stride,
        src_ptr,
        src_stride,
        w,
        h,
        mx,
        my,
        bitdepth_max,
        dst,
        src,
    )
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn put_8tap_regular_smooth_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
    src: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
    put_8tap_8bpc_avx2::<{ Filter2d::RegularSmooth8Tap as usize }>(
        dst_ptr,
        dst_stride,
        src_ptr,
        src_stride,
        w,
        h,
        mx,
        my,
        bitdepth_max,
        dst,
        src,
    )
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn put_8tap_regular_sharp_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
    src: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
    put_8tap_8bpc_avx2::<{ Filter2d::RegularSharp8Tap as usize }>(
        dst_ptr,
        dst_stride,
        src_ptr,
        src_stride,
        w,
        h,
        mx,
        my,
        bitdepth_max,
        dst,
        src,
    )
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn put_8tap_smooth_regular_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
    src: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
    put_8tap_8bpc_avx2::<{ Filter2d::SmoothRegular8Tap as usize }>(
        dst_ptr,
        dst_stride,
        src_ptr,
        src_stride,
        w,
        h,
        mx,
        my,
        bitdepth_max,
        dst,
        src,
    )
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn put_8tap_smooth_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
    src: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
    put_8tap_8bpc_avx2::<{ Filter2d::Smooth8Tap as usize }>(
        dst_ptr,
        dst_stride,
        src_ptr,
        src_stride,
        w,
        h,
        mx,
        my,
        bitdepth_max,
        dst,
        src,
    )
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn put_8tap_smooth_sharp_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
    src: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
    put_8tap_8bpc_avx2::<{ Filter2d::SmoothSharp8Tap as usize }>(
        dst_ptr,
        dst_stride,
        src_ptr,
        src_stride,
        w,
        h,
        mx,
        my,
        bitdepth_max,
        dst,
        src,
    )
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn put_8tap_sharp_regular_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
    src: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
    put_8tap_8bpc_avx2::<{ Filter2d::SharpRegular8Tap as usize }>(
        dst_ptr,
        dst_stride,
        src_ptr,
        src_stride,
        w,
        h,
        mx,
        my,
        bitdepth_max,
        dst,
        src,
    )
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn put_8tap_sharp_smooth_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
    src: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
    put_8tap_8bpc_avx2::<{ Filter2d::SharpSmooth8Tap as usize }>(
        dst_ptr,
        dst_stride,
        src_ptr,
        src_stride,
        w,
        h,
        mx,
        my,
        bitdepth_max,
        dst,
        src,
    )
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn put_8tap_sharp_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
    src: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
    put_8tap_8bpc_avx2::<{ Filter2d::Sharp8Tap as usize }>(
        dst_ptr,
        dst_stride,
        src_ptr,
        src_stride,
        w,
        h,
        mx,
        my,
        bitdepth_max,
        dst,
        src,
    )
    }
}

// =============================================================================
// 8-TAP PREP FUNCTIONS (mct)
// =============================================================================

/// Generic 8-tap prep function for 8bpc
///
/// Similar to put but writes to i16 intermediate buffer instead of pixel output
///
/// # Safety
///
/// - Caller must ensure AVX2 is available
/// - tmp must be valid for writing w*h i16 values
/// - src_ptr must be valid for reading (w+7)*(h+7) bytes (with proper padding)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn prep_8tap_8bpc_avx2_impl(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    h_filter_type: Rav1dFilterMode,
    v_filter_type: Rav1dFilterMode,
) {
    let w = w as usize;
    let h = h as usize;
    let mx = mx as usize;
    let my = my as usize;
    let src = src_ptr as *const u8;

    // For 8bpc: intermediate_bits = 4
    let intermediate_bits = 4u8;

    let fh = get_filter_coeff(mx, w, h_filter_type);
    let fv = get_filter_coeff(my, h, v_filter_type);

    // SAFETY: All operations require AVX2 which is guaranteed by target_feature
    unsafe {
        match (fh, fv) {
            (Some(fh), Some(fv)) => {
                // Case 1: Both H and V filtering
                let tmp_h = h + 7;
                let mut mid = [[0i16; MID_STRIDE]; 135];

                // Horizontal pass
                for y in 0..tmp_h {
                    let src_row = src.offset((y as isize - 3) * src_stride);
                    h_filter_8tap_8bpc_avx2(
                        mid[y].as_mut_ptr(),
                        src_row.sub(3),
                        w,
                        fh,
                        6 - intermediate_bits,
                    );
                }

                // Vertical pass to intermediate output
                for y in 0..h {
                    let out_row = tmp.add(y * w);
                    v_filter_8tap_to_i16_avx2(&mid[y..], out_row, w, fv, 6 + intermediate_bits);
                }
            }
            (Some(fh), None) => {
                // Case 2: H-only filtering
                for y in 0..h {
                    let src_row = src.offset(y as isize * src_stride);
                    let out_row = tmp.add(y * w);
                    h_filter_8tap_8bpc_avx2(out_row, src_row.sub(3), w, fh, intermediate_bits);
                }
            }
            (None, Some(fv)) => {
                // Case 3: V-only filtering
                for y in 0..h {
                    let out_row = tmp.add(y * w);

                    // Build intermediate buffer from 8 source rows
                    let mut mid = [[0i16; MID_STRIDE]; 8];
                    for i in 0..8 {
                        let src_row = src.offset((y as isize + i as isize - 3) * src_stride);
                        for x in 0..w {
                            mid[i][x] = (*src_row.add(x) as i16) << intermediate_bits;
                        }
                    }

                    v_filter_8tap_to_i16_avx2(&mid, out_row, w, fv, 6);
                }
            }
            (None, None) => {
                // Case 4: Simple copy with intermediate scaling
                for y in 0..h {
                    let src_row = src.offset(y as isize * src_stride);
                    let out_row = tmp.add(y * w);
                    for x in 0..w {
                        *out_row.add(x) = (*src_row.add(x) as i16) << intermediate_bits;
                    }
                }
            }
        }
    }
}

/// Vertical 8-tap filter to i16 output (for prep functions)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn v_filter_8tap_to_i16_avx2(
    mid: &[[i16; MID_STRIDE]],
    dst: *mut i16,
    w: usize,
    filter: &[i8; 8],
    sh: u8,
) {
    // SAFETY: All operations require AVX2 which is guaranteed by target_feature
    unsafe {
        let rnd = _mm256_set1_epi32((1i32 << sh) >> 1);

        let c0 = _mm256_set1_epi32(filter[0] as i32);
        let c1 = _mm256_set1_epi32(filter[1] as i32);
        let c2 = _mm256_set1_epi32(filter[2] as i32);
        let c3 = _mm256_set1_epi32(filter[3] as i32);
        let c4 = _mm256_set1_epi32(filter[4] as i32);
        let c5 = _mm256_set1_epi32(filter[5] as i32);
        let c6 = _mm256_set1_epi32(filter[6] as i32);
        let c7 = _mm256_set1_epi32(filter[7] as i32);

        let mut col = 0usize;

        while col + 8 <= w {
            let m0 =
                _mm256_cvtepi16_epi32(_mm_loadu_si128(mid[0][col..].as_ptr() as *const __m128i));
            let m1 =
                _mm256_cvtepi16_epi32(_mm_loadu_si128(mid[1][col..].as_ptr() as *const __m128i));
            let m2 =
                _mm256_cvtepi16_epi32(_mm_loadu_si128(mid[2][col..].as_ptr() as *const __m128i));
            let m3 =
                _mm256_cvtepi16_epi32(_mm_loadu_si128(mid[3][col..].as_ptr() as *const __m128i));
            let m4 =
                _mm256_cvtepi16_epi32(_mm_loadu_si128(mid[4][col..].as_ptr() as *const __m128i));
            let m5 =
                _mm256_cvtepi16_epi32(_mm_loadu_si128(mid[5][col..].as_ptr() as *const __m128i));
            let m6 =
                _mm256_cvtepi16_epi32(_mm_loadu_si128(mid[6][col..].as_ptr() as *const __m128i));
            let m7 =
                _mm256_cvtepi16_epi32(_mm_loadu_si128(mid[7][col..].as_ptr() as *const __m128i));

            let mut sum = _mm256_mullo_epi32(m0, c0);
            sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(m1, c1));
            sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(m2, c2));
            sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(m3, c3));
            sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(m4, c4));
            sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(m5, c5));
            sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(m6, c6));
            sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(m7, c7));

            let shift_count = _mm_cvtsi32_si128(sh as i32);
            let shifted = _mm256_sra_epi32(_mm256_add_epi32(sum, rnd), shift_count);

            // Pack to i16 (signed saturation is fine for intermediate values)
            let packed = _mm256_packs_epi32(shifted, shifted);
            let packed = _mm256_permute4x64_epi64(packed, 0b11011000);

            // Store 8 i16 values
            _mm_storeu_si128(
                dst.add(col) as *mut __m128i,
                _mm256_castsi256_si128(packed),
            );

            col += 8;
        }

        while col < w {
            let mut sum = 0i32;
            for i in 0..8 {
                sum += filter[i] as i32 * mid[i][col] as i32;
            }
            *dst.add(col) = ((sum + ((1 << sh) >> 1)) >> sh) as i16;
            col += 1;
        }
    }
}

/// Generic prep_8tap function wrapper with Filter2d const generic
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn prep_8tap_8bpc_avx2<const FILTER: usize>(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    _bitdepth_max: i32,
    _src: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let filter = Filter2d::from_repr(FILTER).unwrap();
    let (h_filter, v_filter) = filter.hv();

    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
        prep_8tap_8bpc_avx2_impl(tmp, src_ptr, src_stride, w, h, mx, my, h_filter, v_filter);
    }
}

// Specific filter type wrappers for dispatch table

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn prep_8tap_regular_8bpc_avx2(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
    prep_8tap_8bpc_avx2::<{ Filter2d::Regular8Tap as usize }>(
        tmp, src_ptr, src_stride, w, h, mx, my, bitdepth_max, src,
    )
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn prep_8tap_regular_smooth_8bpc_avx2(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
    prep_8tap_8bpc_avx2::<{ Filter2d::RegularSmooth8Tap as usize }>(
        tmp, src_ptr, src_stride, w, h, mx, my, bitdepth_max, src,
    )
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn prep_8tap_regular_sharp_8bpc_avx2(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
    prep_8tap_8bpc_avx2::<{ Filter2d::RegularSharp8Tap as usize }>(
        tmp, src_ptr, src_stride, w, h, mx, my, bitdepth_max, src,
    )
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn prep_8tap_smooth_regular_8bpc_avx2(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
    prep_8tap_8bpc_avx2::<{ Filter2d::SmoothRegular8Tap as usize }>(
        tmp, src_ptr, src_stride, w, h, mx, my, bitdepth_max, src,
    )
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn prep_8tap_smooth_8bpc_avx2(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
    prep_8tap_8bpc_avx2::<{ Filter2d::Smooth8Tap as usize }>(
        tmp, src_ptr, src_stride, w, h, mx, my, bitdepth_max, src,
    )
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn prep_8tap_smooth_sharp_8bpc_avx2(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
    prep_8tap_8bpc_avx2::<{ Filter2d::SmoothSharp8Tap as usize }>(
        tmp, src_ptr, src_stride, w, h, mx, my, bitdepth_max, src,
    )
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn prep_8tap_sharp_regular_8bpc_avx2(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
    prep_8tap_8bpc_avx2::<{ Filter2d::SharpRegular8Tap as usize }>(
        tmp, src_ptr, src_stride, w, h, mx, my, bitdepth_max, src,
    )
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn prep_8tap_sharp_smooth_8bpc_avx2(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
    prep_8tap_8bpc_avx2::<{ Filter2d::SharpSmooth8Tap as usize }>(
        tmp, src_ptr, src_stride, w, h, mx, my, bitdepth_max, src,
    )
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn prep_8tap_sharp_8bpc_avx2(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
    prep_8tap_8bpc_avx2::<{ Filter2d::Sharp8Tap as usize }>(
        tmp, src_ptr, src_stride, w, h, mx, my, bitdepth_max, src,
    )
    }
}

// =============================================================================
// 16BPC 8-TAP FILTERS
// =============================================================================

/// Generic 8-tap put function for 16bpc
///
/// Similar to 8bpc but handles 16-bit pixels and different intermediate scaling
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn put_8tap_16bpc_avx2_impl(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    h_filter_type: Rav1dFilterMode,
    v_filter_type: Rav1dFilterMode,
) {
    let w = w as usize;
    let h = h as usize;
    let mx = mx as usize;
    let my = my as usize;
    let dst = dst_ptr as *mut u16;
    let src = src_ptr as *const u16;
    let dst_stride_elems = dst_stride / 2;
    let src_stride_elems = src_stride / 2;
    let max = bitdepth_max as i32;

    // For 16bpc: intermediate_bits = 4
    let intermediate_bits = 4i32;

    let fh = get_filter_coeff(mx, w, h_filter_type);
    let fv = get_filter_coeff(my, h, v_filter_type);

    // SAFETY: All operations require AVX2 which is guaranteed by target_feature
    unsafe {
        match (fh, fv) {
            (Some(fh), Some(fv)) => {
                // Case 1: Both H and V filtering - two-pass through intermediate
                let tmp_h = h + 7;
                let mut mid = [[0i32; MID_STRIDE]; 135];

                // Horizontal pass - output is i32 to preserve precision
                for y in 0..tmp_h {
                    let src_row = src.offset((y as isize - 3) * src_stride_elems);
                    let mid_row = &mut mid[y];
                    for x in 0..w {
                        let mut sum = 0i32;
                        for i in 0..8 {
                            let px = *src_row.offset(x as isize + i as isize - 3) as i32;
                            sum += fh[i] as i32 * px;
                        }
                        // Round: (sum + rnd) >> (6 - intermediate_bits)
                        let rnd = (1 << (6 - intermediate_bits)) >> 1;
                        mid_row[x] = (sum + rnd) >> (6 - intermediate_bits);
                    }
                }

                // Vertical pass - i32 -> u16
                for y in 0..h {
                    let dst_row = dst.offset(y as isize * dst_stride_elems);
                    for x in 0..w {
                        let mut sum = 0i32;
                        for i in 0..8 {
                            sum += fv[i] as i32 * mid[y + i][x];
                        }
                        // Round and clamp: (sum + rnd) >> (6 + intermediate_bits)
                        let rnd = (1 << (6 + intermediate_bits)) >> 1;
                        let val = ((sum + rnd) >> (6 + intermediate_bits)).clamp(0, max);
                        *dst_row.add(x) = val as u16;
                    }
                }
            }
            (Some(fh), None) => {
                // Case 2: H-only filtering
                for y in 0..h {
                    let src_row = src.offset(y as isize * src_stride_elems);
                    let dst_row = dst.offset(y as isize * dst_stride_elems);
                    for x in 0..w {
                        let mut sum = 0i32;
                        for i in 0..8 {
                            let px = *src_row.offset(x as isize + i as isize - 3) as i32;
                            sum += fh[i] as i32 * px;
                        }
                        let rnd = 32;
                        let val = ((sum + rnd) >> 6).clamp(0, max);
                        *dst_row.add(x) = val as u16;
                    }
                }
            }
            (None, Some(fv)) => {
                // Case 3: V-only filtering
                for y in 0..h {
                    let dst_row = dst.offset(y as isize * dst_stride_elems);
                    for x in 0..w {
                        let mut sum = 0i32;
                        for i in 0..8 {
                            let px = *src.offset((y as isize + i as isize - 3) * src_stride_elems + x as isize) as i32;
                            sum += fv[i] as i32 * px;
                        }
                        let rnd = 32;
                        let val = ((sum + rnd) >> 6).clamp(0, max);
                        *dst_row.add(x) = val as u16;
                    }
                }
            }
            (None, None) => {
                // Case 4: Simple copy
                for y in 0..h {
                    let src_row = src.offset(y as isize * src_stride_elems);
                    let dst_row = dst.offset(y as isize * dst_stride_elems);
                    std::ptr::copy_nonoverlapping(src_row, dst_row, w);
                }
            }
        }
    }
}

/// Generic 8-tap prep function for 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn prep_8tap_16bpc_avx2_impl(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    h_filter_type: Rav1dFilterMode,
    v_filter_type: Rav1dFilterMode,
) {
    let w = w as usize;
    let h = h as usize;
    let mx = mx as usize;
    let my = my as usize;
    let src = src_ptr as *const u16;
    let src_stride_elems = src_stride / 2;

    // For 16bpc: intermediate_bits = 4, PREP_BIAS = 8192
    let intermediate_bits = 4i32;
    const PREP_BIAS: i32 = 8192;

    let fh = get_filter_coeff(mx, w, h_filter_type);
    let fv = get_filter_coeff(my, h, v_filter_type);

    // SAFETY: All operations require AVX2 which is guaranteed by target_feature
    unsafe {
        match (fh, fv) {
            (Some(fh), Some(fv)) => {
                // Two-pass filtering
                let tmp_h = h + 7;
                let mut mid = [[0i32; MID_STRIDE]; 135];

                // Horizontal pass
                for y in 0..tmp_h {
                    let src_row = src.offset((y as isize - 3) * src_stride_elems);
                    for x in 0..w {
                        let mut sum = 0i32;
                        for i in 0..8 {
                            let px = *src_row.offset(x as isize + i as isize - 3) as i32;
                            sum += fh[i] as i32 * px;
                        }
                        let rnd = (1 << (6 - intermediate_bits)) >> 1;
                        mid[y][x] = (sum + rnd) >> (6 - intermediate_bits);
                    }
                }

                // Vertical pass
                for y in 0..h {
                    let out_row = tmp.add(y * w);
                    for x in 0..w {
                        let mut sum = 0i32;
                        for i in 0..8 {
                            sum += fv[i] as i32 * mid[y + i][x];
                        }
                        let rnd = 32;
                        let val = ((sum + rnd) >> 6) - PREP_BIAS;
                        *out_row.add(x) = val as i16;
                    }
                }
            }
            (Some(fh), None) => {
                // H-only filtering
                for y in 0..h {
                    let src_row = src.offset(y as isize * src_stride_elems);
                    let out_row = tmp.add(y * w);
                    for x in 0..w {
                        let mut sum = 0i32;
                        for i in 0..8 {
                            let px = *src_row.offset(x as isize + i as isize - 3) as i32;
                            sum += fh[i] as i32 * px;
                        }
                        let rnd = (1 << (6 - intermediate_bits)) >> 1;
                        let val = ((sum + rnd) >> (6 - intermediate_bits)) - PREP_BIAS;
                        *out_row.add(x) = val as i16;
                    }
                }
            }
            (None, Some(fv)) => {
                // V-only filtering
                for y in 0..h {
                    let out_row = tmp.add(y * w);
                    for x in 0..w {
                        let mut sum = 0i32;
                        for i in 0..8 {
                            let px = *src.offset((y as isize + i as isize - 3) * src_stride_elems + x as isize) as i32;
                            sum += fv[i] as i32 * px;
                        }
                        let rnd = (1 << (6 - intermediate_bits)) >> 1;
                        let val = ((sum + rnd) >> (6 - intermediate_bits)) - PREP_BIAS;
                        *out_row.add(x) = val as i16;
                    }
                }
            }
            (None, None) => {
                // Simple copy with scaling and bias
                for y in 0..h {
                    let src_row = src.offset(y as isize * src_stride_elems);
                    let out_row = tmp.add(y * w);
                    for x in 0..w {
                        let px = *src_row.add(x) as i32;
                        let val = (px << intermediate_bits) - PREP_BIAS;
                        *out_row.add(x) = val as i16;
                    }
                }
            }
        }
    }
}

/// Generic put_8tap function wrapper for 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn put_8tap_16bpc_avx2<const FILTER: usize>(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
    _src: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let filter = Filter2d::from_repr(FILTER).unwrap();
    let (h_filter, v_filter) = filter.hv();

    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
        put_8tap_16bpc_avx2_impl(
            dst_ptr, dst_stride, src_ptr, src_stride, w, h, mx, my, bitdepth_max, h_filter, v_filter,
        );
    }
}

/// Generic prep_8tap function wrapper for 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn prep_8tap_16bpc_avx2<const FILTER: usize>(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    _bitdepth_max: i32,
    _src: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let filter = Filter2d::from_repr(FILTER).unwrap();
    let (h_filter, v_filter) = filter.hv();

    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
        prep_8tap_16bpc_avx2_impl(tmp, src_ptr, src_stride, w, h, mx, my, h_filter, v_filter);
    }
}

// 16bpc wrapper functions for each filter type

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn put_8tap_regular_16bpc_avx2(
    dst_ptr: *mut DynPixel, dst_stride: isize, src_ptr: *const DynPixel, src_stride: isize,
    w: i32, h: i32, mx: i32, my: i32, bitdepth_max: i32,
    dst: *const FFISafe<Rav1dPictureDataComponentOffset>, src: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    unsafe { put_8tap_16bpc_avx2::<{ Filter2d::Regular8Tap as usize }>(dst_ptr, dst_stride, src_ptr, src_stride, w, h, mx, my, bitdepth_max, dst, src) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn put_8tap_regular_smooth_16bpc_avx2(
    dst_ptr: *mut DynPixel, dst_stride: isize, src_ptr: *const DynPixel, src_stride: isize,
    w: i32, h: i32, mx: i32, my: i32, bitdepth_max: i32,
    dst: *const FFISafe<Rav1dPictureDataComponentOffset>, src: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    unsafe { put_8tap_16bpc_avx2::<{ Filter2d::RegularSmooth8Tap as usize }>(dst_ptr, dst_stride, src_ptr, src_stride, w, h, mx, my, bitdepth_max, dst, src) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn put_8tap_regular_sharp_16bpc_avx2(
    dst_ptr: *mut DynPixel, dst_stride: isize, src_ptr: *const DynPixel, src_stride: isize,
    w: i32, h: i32, mx: i32, my: i32, bitdepth_max: i32,
    dst: *const FFISafe<Rav1dPictureDataComponentOffset>, src: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    unsafe { put_8tap_16bpc_avx2::<{ Filter2d::RegularSharp8Tap as usize }>(dst_ptr, dst_stride, src_ptr, src_stride, w, h, mx, my, bitdepth_max, dst, src) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn put_8tap_smooth_regular_16bpc_avx2(
    dst_ptr: *mut DynPixel, dst_stride: isize, src_ptr: *const DynPixel, src_stride: isize,
    w: i32, h: i32, mx: i32, my: i32, bitdepth_max: i32,
    dst: *const FFISafe<Rav1dPictureDataComponentOffset>, src: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    unsafe { put_8tap_16bpc_avx2::<{ Filter2d::SmoothRegular8Tap as usize }>(dst_ptr, dst_stride, src_ptr, src_stride, w, h, mx, my, bitdepth_max, dst, src) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn put_8tap_smooth_16bpc_avx2(
    dst_ptr: *mut DynPixel, dst_stride: isize, src_ptr: *const DynPixel, src_stride: isize,
    w: i32, h: i32, mx: i32, my: i32, bitdepth_max: i32,
    dst: *const FFISafe<Rav1dPictureDataComponentOffset>, src: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    unsafe { put_8tap_16bpc_avx2::<{ Filter2d::Smooth8Tap as usize }>(dst_ptr, dst_stride, src_ptr, src_stride, w, h, mx, my, bitdepth_max, dst, src) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn put_8tap_smooth_sharp_16bpc_avx2(
    dst_ptr: *mut DynPixel, dst_stride: isize, src_ptr: *const DynPixel, src_stride: isize,
    w: i32, h: i32, mx: i32, my: i32, bitdepth_max: i32,
    dst: *const FFISafe<Rav1dPictureDataComponentOffset>, src: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    unsafe { put_8tap_16bpc_avx2::<{ Filter2d::SmoothSharp8Tap as usize }>(dst_ptr, dst_stride, src_ptr, src_stride, w, h, mx, my, bitdepth_max, dst, src) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn put_8tap_sharp_regular_16bpc_avx2(
    dst_ptr: *mut DynPixel, dst_stride: isize, src_ptr: *const DynPixel, src_stride: isize,
    w: i32, h: i32, mx: i32, my: i32, bitdepth_max: i32,
    dst: *const FFISafe<Rav1dPictureDataComponentOffset>, src: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    unsafe { put_8tap_16bpc_avx2::<{ Filter2d::SharpRegular8Tap as usize }>(dst_ptr, dst_stride, src_ptr, src_stride, w, h, mx, my, bitdepth_max, dst, src) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn put_8tap_sharp_smooth_16bpc_avx2(
    dst_ptr: *mut DynPixel, dst_stride: isize, src_ptr: *const DynPixel, src_stride: isize,
    w: i32, h: i32, mx: i32, my: i32, bitdepth_max: i32,
    dst: *const FFISafe<Rav1dPictureDataComponentOffset>, src: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    unsafe { put_8tap_16bpc_avx2::<{ Filter2d::SharpSmooth8Tap as usize }>(dst_ptr, dst_stride, src_ptr, src_stride, w, h, mx, my, bitdepth_max, dst, src) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn put_8tap_sharp_16bpc_avx2(
    dst_ptr: *mut DynPixel, dst_stride: isize, src_ptr: *const DynPixel, src_stride: isize,
    w: i32, h: i32, mx: i32, my: i32, bitdepth_max: i32,
    dst: *const FFISafe<Rav1dPictureDataComponentOffset>, src: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    unsafe { put_8tap_16bpc_avx2::<{ Filter2d::Sharp8Tap as usize }>(dst_ptr, dst_stride, src_ptr, src_stride, w, h, mx, my, bitdepth_max, dst, src) }
}

// 16bpc prep wrapper functions

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn prep_8tap_regular_16bpc_avx2(
    tmp: *mut i16, src_ptr: *const DynPixel, src_stride: isize, w: i32, h: i32, mx: i32, my: i32,
    bitdepth_max: i32, src: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    unsafe { prep_8tap_16bpc_avx2::<{ Filter2d::Regular8Tap as usize }>(tmp, src_ptr, src_stride, w, h, mx, my, bitdepth_max, src) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn prep_8tap_regular_smooth_16bpc_avx2(
    tmp: *mut i16, src_ptr: *const DynPixel, src_stride: isize, w: i32, h: i32, mx: i32, my: i32,
    bitdepth_max: i32, src: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    unsafe { prep_8tap_16bpc_avx2::<{ Filter2d::RegularSmooth8Tap as usize }>(tmp, src_ptr, src_stride, w, h, mx, my, bitdepth_max, src) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn prep_8tap_regular_sharp_16bpc_avx2(
    tmp: *mut i16, src_ptr: *const DynPixel, src_stride: isize, w: i32, h: i32, mx: i32, my: i32,
    bitdepth_max: i32, src: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    unsafe { prep_8tap_16bpc_avx2::<{ Filter2d::RegularSharp8Tap as usize }>(tmp, src_ptr, src_stride, w, h, mx, my, bitdepth_max, src) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn prep_8tap_smooth_regular_16bpc_avx2(
    tmp: *mut i16, src_ptr: *const DynPixel, src_stride: isize, w: i32, h: i32, mx: i32, my: i32,
    bitdepth_max: i32, src: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    unsafe { prep_8tap_16bpc_avx2::<{ Filter2d::SmoothRegular8Tap as usize }>(tmp, src_ptr, src_stride, w, h, mx, my, bitdepth_max, src) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn prep_8tap_smooth_16bpc_avx2(
    tmp: *mut i16, src_ptr: *const DynPixel, src_stride: isize, w: i32, h: i32, mx: i32, my: i32,
    bitdepth_max: i32, src: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    unsafe { prep_8tap_16bpc_avx2::<{ Filter2d::Smooth8Tap as usize }>(tmp, src_ptr, src_stride, w, h, mx, my, bitdepth_max, src) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn prep_8tap_smooth_sharp_16bpc_avx2(
    tmp: *mut i16, src_ptr: *const DynPixel, src_stride: isize, w: i32, h: i32, mx: i32, my: i32,
    bitdepth_max: i32, src: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    unsafe { prep_8tap_16bpc_avx2::<{ Filter2d::SmoothSharp8Tap as usize }>(tmp, src_ptr, src_stride, w, h, mx, my, bitdepth_max, src) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn prep_8tap_sharp_regular_16bpc_avx2(
    tmp: *mut i16, src_ptr: *const DynPixel, src_stride: isize, w: i32, h: i32, mx: i32, my: i32,
    bitdepth_max: i32, src: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    unsafe { prep_8tap_16bpc_avx2::<{ Filter2d::SharpRegular8Tap as usize }>(tmp, src_ptr, src_stride, w, h, mx, my, bitdepth_max, src) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn prep_8tap_sharp_smooth_16bpc_avx2(
    tmp: *mut i16, src_ptr: *const DynPixel, src_stride: isize, w: i32, h: i32, mx: i32, my: i32,
    bitdepth_max: i32, src: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    unsafe { prep_8tap_16bpc_avx2::<{ Filter2d::SharpSmooth8Tap as usize }>(tmp, src_ptr, src_stride, w, h, mx, my, bitdepth_max, src) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn prep_8tap_sharp_16bpc_avx2(
    tmp: *mut i16, src_ptr: *const DynPixel, src_stride: isize, w: i32, h: i32, mx: i32, my: i32,
    bitdepth_max: i32, src: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    unsafe { prep_8tap_16bpc_avx2::<{ Filter2d::Sharp8Tap as usize }>(tmp, src_ptr, src_stride, w, h, mx, my, bitdepth_max, src) }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_has_avx2() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            is_x86_feature_detected!("avx2")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    #[test]
    fn test_avg_8bpc_avx2_matches_scalar() {
        if !cpu_has_avx2() {
            eprintln!("Skipping AVX2 test - CPU doesn't support it");
            return;
        }

        let test_values: Vec<i16> = vec![
            0,
            1,
            2,
            127,
            128,
            255,
            256,
            511,
            512,
            1023,
            1024,
            -1,
            -128,
            -256,
            -512,
            -1024,
            i16::MIN,
            i16::MAX,
        ];

        let w = 64i32;
        let h = 2i32;

        let mut tmp1 = [0i16; COMPINTER_LEN];
        let mut tmp2 = [0i16; COMPINTER_LEN];
        let mut dst_avx2 = vec![0u8; (w * h) as usize];
        let mut dst_scalar = vec![0u8; (w * h) as usize];

        for &v1 in &test_values {
            for &v2 in &test_values {
                // Fill buffers
                for i in 0..(w * h) as usize {
                    tmp1[i] = v1;
                    tmp2[i] = v2;
                }
                dst_avx2.fill(0);
                dst_scalar.fill(0);

                unsafe {
                    avg_scalar(
                        dst_scalar.as_mut_ptr() as *mut DynPixel,
                        w as isize,
                        &tmp1,
                        &tmp2,
                        w,
                        h,
                        255,
                        std::ptr::null(),
                    );

                    avg_8bpc_avx2(
                        dst_avx2.as_mut_ptr() as *mut DynPixel,
                        w as isize,
                        &tmp1,
                        &tmp2,
                        w,
                        h,
                        255,
                        std::ptr::null(),
                    );
                }

                assert_eq!(
                    dst_avx2,
                    dst_scalar,
                    "Mismatch for v1={}, v2={}: avx2={:?} scalar={:?}",
                    v1,
                    v2,
                    &dst_avx2[..8],
                    &dst_scalar[..8]
                );
            }
        }
    }

    #[test]
    fn test_avg_varying_data() {
        if !cpu_has_avx2() {
            return;
        }

        let w = 128i32;
        let h = 4i32;

        let mut tmp1 = [0i16; COMPINTER_LEN];
        let mut tmp2 = [0i16; COMPINTER_LEN];

        for i in 0..(w * h) as usize {
            tmp1[i] = ((i * 37) % 8192) as i16;
            tmp2[i] = ((i * 73 + 1000) % 8192) as i16;
        }

        let mut dst_avx2 = vec![0u8; (w * h) as usize];
        let mut dst_scalar = vec![0u8; (w * h) as usize];

        unsafe {
            avg_scalar(
                dst_scalar.as_mut_ptr() as *mut DynPixel,
                w as isize,
                &tmp1,
                &tmp2,
                w,
                h,
                255,
                std::ptr::null(),
            );

            avg_8bpc_avx2(
                dst_avx2.as_mut_ptr() as *mut DynPixel,
                w as isize,
                &tmp1,
                &tmp2,
                w,
                h,
                255,
                std::ptr::null(),
            );
        }

        assert_eq!(dst_avx2, dst_scalar, "Results differ for varying data");
    }

    #[test]
    fn test_w_avg_8bpc_avx2_matches_scalar() {
        if !cpu_has_avx2() {
            eprintln!("Skipping AVX2 test - CPU doesn't support it");
            return;
        }

        let test_values: Vec<i16> = vec![
            0, 1, 127, 255, 512, 1023, 2047, 4095, 8191, -1, -128, -512, -1024,
        ];
        let test_weights = [0, 1, 4, 8, 12, 15, 16];

        let w = 64i32;
        let h = 2i32;

        let mut tmp1 = [0i16; COMPINTER_LEN];
        let mut tmp2 = [0i16; COMPINTER_LEN];
        let mut dst_avx2 = vec![0u8; (w * h) as usize];
        let mut dst_scalar = vec![0u8; (w * h) as usize];

        for &weight in &test_weights {
            for &v1 in &test_values {
                for &v2 in &test_values {
                    for i in 0..(w * h) as usize {
                        tmp1[i] = v1;
                        tmp2[i] = v2;
                    }
                    dst_avx2.fill(0);
                    dst_scalar.fill(0);

                    unsafe {
                        w_avg_scalar(
                            dst_scalar.as_mut_ptr() as *mut DynPixel,
                            w as isize,
                            &tmp1,
                            &tmp2,
                            w,
                            h,
                            weight,
                            255,
                            std::ptr::null(),
                        );

                        w_avg_8bpc_avx2(
                            dst_avx2.as_mut_ptr() as *mut DynPixel,
                            w as isize,
                            &tmp1,
                            &tmp2,
                            w,
                            h,
                            weight,
                            255,
                            std::ptr::null(),
                        );
                    }

                    assert_eq!(
                        dst_avx2,
                        dst_scalar,
                        "Mismatch for weight={}, v1={}, v2={}: avx2={:?} scalar={:?}",
                        weight,
                        v1,
                        v2,
                        &dst_avx2[..8],
                        &dst_scalar[..8]
                    );
                }
            }
        }
    }

    #[test]
    fn test_mask_8bpc_matches_scalar() {
        if !cpu_has_avx2() {
            eprintln!("Skipping AVX2 test - CPU doesn't support it");
            return;
        }

        let test_values: Vec<i16> = vec![0, 127, 255, 512, 1023, 4095, -128, -512];
        let test_masks: Vec<u8> = vec![0, 1, 16, 32, 48, 63, 64];

        let w = 64i32;
        let h = 2i32;

        let mut tmp1 = [0i16; COMPINTER_LEN];
        let mut tmp2 = [0i16; COMPINTER_LEN];
        let mut mask = vec![0u8; (w * h) as usize];
        let mut dst_avx2 = vec![0u8; (w * h) as usize];
        let mut dst_scalar = vec![0u8; (w * h) as usize];

        for &m in &test_masks {
            for &v1 in &test_values {
                for &v2 in &test_values {
                    for i in 0..(w * h) as usize {
                        tmp1[i] = v1;
                        tmp2[i] = v2;
                        mask[i] = m;
                    }
                    dst_avx2.fill(0);
                    dst_scalar.fill(0);

                    unsafe {
                        mask_scalar(
                            dst_scalar.as_mut_ptr() as *mut DynPixel,
                            w as isize,
                            &tmp1,
                            &tmp2,
                            w,
                            h,
                            mask.as_ptr(),
                            255,
                            std::ptr::null(),
                        );

                        mask_8bpc_avx2(
                            dst_avx2.as_mut_ptr() as *mut DynPixel,
                            w as isize,
                            &tmp1,
                            &tmp2,
                            w,
                            h,
                            mask.as_ptr(),
                            255,
                            std::ptr::null(),
                        );
                    }

                    assert_eq!(
                        dst_avx2,
                        dst_scalar,
                        "Mismatch for mask={}, v1={}, v2={}: avx2={:?} scalar={:?}",
                        m,
                        v1,
                        v2,
                        &dst_avx2[..8],
                        &dst_scalar[..8]
                    );
                }
            }
        }
    }
}
