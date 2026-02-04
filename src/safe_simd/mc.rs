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
use crate::src::ffi_safe::FFISafe;
use crate::src::internal::COMPINTER_LEN;
use crate::include::dav1d::picture::Rav1dPictureDataComponentOffset;

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
        let dst_row = unsafe {
            std::slice::from_raw_parts_mut(dst.offset(row as isize * dst_stride), w)
        };

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

    // For 16bpc, intermediate_bits = 4, so shift = 5, round = 16
    // Result should be clamped to [0, bitdepth_max]
    let intermediate_bits = 4; // for 10/12 bit
    let sh = intermediate_bits + 1;
    let rnd = (1 << intermediate_bits) + 8192 * 2; // PREP_BIAS = 8192 for 16bpc

    let max = bitdepth_max as i32;

    for row in 0..h {
        let tmp1_row = &tmp1[row * w..][..w];
        let tmp2_row = &tmp2[row * w..][..w];
        // SAFETY: dst_ptr is valid, stride is correct
        let dst_row = unsafe {
            std::slice::from_raw_parts_mut(
                dst.offset(row as isize * dst_stride_elems),
                w
            )
        };

        // Scalar implementation for now - SIMD for 16bpc is more complex
        // TODO: Add AVX2 SIMD path
        for col in 0..w {
            let sum = tmp1_row[col] as i32 + tmp2_row[col] as i32;
            let val = ((sum + rnd) >> sh).clamp(0, max) as u16;
            dst_row[col] = val;
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
        let dst_row = unsafe {
            std::slice::from_raw_parts_mut(dst.offset(row as isize * dst_stride), w)
        };

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
        let dst_row = unsafe {
            std::slice::from_raw_parts_mut(dst.offset(row as isize * dst_stride), w)
        };

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

    // For 16bpc: intermediate_bits = 4, sh = 8, rnd = 128 + PREP_BIAS*16
    let intermediate_bits = 4;
    let sh = intermediate_bits + 4;
    let rnd = (8 << intermediate_bits) + 8192 * 16; // PREP_BIAS = 8192 for 16bpc
    let max = bitdepth_max as i32;

    for row in 0..h {
        let tmp1_row = &tmp1[row * w..][..w];
        let tmp2_row = &tmp2[row * w..][..w];
        let dst_row = unsafe {
            std::slice::from_raw_parts_mut(dst.offset(row as isize * dst_stride_elems), w)
        };

        // Scalar for now
        for col in 0..w {
            let a = tmp1_row[col] as i32;
            let b = tmp2_row[col] as i32;
            let val = (a * weight + b * (16 - weight) + rnd) >> sh;
            dst_row[col] = val.clamp(0, max) as u16;
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
        let dst_row = unsafe {
            std::slice::from_raw_parts_mut(dst.offset(row as isize * dst_stride), w)
        };

        for col in 0..w {
            let a = tmp1_row[col] as i32;
            let b = tmp2_row[col] as i32;
            let val = (a * weight + b * (16 - weight) + rnd) >> sh;
            dst_row[col] = val.clamp(0, 255) as u8;
        }
    }
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
            0, 1, 2, 127, 128, 255, 256, 511, 512, 1023, 1024,
            -1, -128, -256, -512, -1024, i16::MIN, i16::MAX,
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
                    dst_avx2, dst_scalar,
                    "Mismatch for v1={}, v2={}: avx2={:?} scalar={:?}",
                    v1, v2, &dst_avx2[..8], &dst_scalar[..8]
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
            0, 1, 127, 255, 512, 1023, 2047, 4095, 8191,
            -1, -128, -512, -1024,
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
                        dst_avx2, dst_scalar,
                        "Mismatch for weight={}, v1={}, v2={}: avx2={:?} scalar={:?}",
                        weight, v1, v2, &dst_avx2[..8], &dst_scalar[..8]
                    );
                }
            }
        }
    }
}
