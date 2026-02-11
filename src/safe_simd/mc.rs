//! Safe SIMD implementations of motion compensation functions
#![allow(deprecated)] // FFI wrappers need to forge tokens
#![allow(dead_code)]
#![cfg_attr(not(feature = "unchecked"), forbid(unsafe_code))]
#![cfg_attr(feature = "unchecked", deny(unsafe_code))]
//!
//! These replace the hand-written assembly in src/x86/mc_*.asm
//!
//! Uses archmage tokens for safe SIMD invocation:
//! - Desktop64 (X64V3Token) for AVX2+FMA on x86-64
//! - The runtime CPU check happens at init time in the dispatch table setup
//! - extern "C" wrappers use forge_token_dangerously() since features are pre-verified

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::src::safe_simd::pixel_access::Flex;
#[cfg(target_arch = "x86_64")]
use crate::src::safe_simd::pixel_access::{loadi64, loadu_128, loadu_256, storeu_128, storeu_256};
#[cfg(target_arch = "x86_64")]
use archmage::{arcane, rite, Desktop64};

use crate::include::common::bitdepth::BitDepth;
use crate::include::dav1d::headers::Rav1dFilterMode;
#[cfg(target_arch = "x86_64")]
use crate::include::dav1d::headers::Rav1dPixelLayoutSubSampled;
use crate::include::dav1d::picture::PicOffset;
use crate::src::internal::COMPINTER_LEN;
use crate::src::levels::Filter2d;
use crate::src::strided::Strided as _;

use std::cell::Cell;

type Mid16x135 = Box<[[i16; MID_STRIDE]; 135]>;
type Mid32x135 = Box<[[i32; MID_STRIDE]; 135]>;
type Mid16x130 = Box<[[i16; MID_STRIDE]; 130]>;
type Mid32x130 = Box<[[i32; MID_STRIDE]; 130]>;

// Thread-local scratch buffers for MC intermediate data.
// Allocated once on first use (zeroed by Box::new), then reused across all MC
// calls. This eliminates ~520M instructions of memset per decode (25% of total).
// The buffers don't need re-zeroing between calls because every MC function
// fully writes the region it reads (horizontal filter writes rows 0..h+7,
// vertical filter reads only from those written rows).
//
// Pattern: take() before use, set(Some(...)) after use. If a function panics
// between take and put, the next call allocates a fresh buffer (no unsoundness).
thread_local! {
    static MID_I16_135: Cell<Option<Mid16x135>> = const { Cell::new(None) };
    static MID_I32_135: Cell<Option<Mid32x135>> = const { Cell::new(None) };
    static MID_I16_130: Cell<Option<Mid16x130>> = const { Cell::new(None) };
    static MID_I32_130: Cell<Option<Mid32x130>> = const { Cell::new(None) };
}

#[inline(always)]
fn take_mid_i16_135() -> Mid16x135 {
    MID_I16_135
        .with(|c| c.take())
        .unwrap_or_else(|| Box::new([[0; MID_STRIDE]; 135]))
}
#[inline(always)]
fn put_mid_i16_135(mid: Mid16x135) {
    MID_I16_135.with(|c| c.set(Some(mid)));
}
#[inline(always)]
fn take_mid_i32_135() -> Mid32x135 {
    MID_I32_135
        .with(|c| c.take())
        .unwrap_or_else(|| Box::new([[0; MID_STRIDE]; 135]))
}
#[inline(always)]
fn put_mid_i32_135(mid: Mid32x135) {
    MID_I32_135.with(|c| c.set(Some(mid)));
}
#[inline(always)]
fn take_mid_i16_130() -> Mid16x130 {
    MID_I16_130
        .with(|c| c.take())
        .unwrap_or_else(|| Box::new([[0; MID_STRIDE]; 130]))
}
#[inline(always)]
fn put_mid_i16_130(mid: Mid16x130) {
    MID_I16_130.with(|c| c.set(Some(mid)));
}
#[inline(always)]
fn take_mid_i32_130() -> Mid32x130 {
    MID_I32_130
        .with(|c| c.take())
        .unwrap_or_else(|| Box::new([[0; MID_STRIDE]; 130]))
}
#[inline(always)]
fn put_mid_i32_130(mid: Mid32x130) {
    MID_I32_130.with(|c| c.set(Some(mid)));
}

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
#[arcane]
fn avg_8bpc_avx2_safe(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
) {
    let mut dst = dst.flex_mut();
    let w = w as usize;
    let h = h as usize;

    // Broadcast rounding constant to all lanes
    // SAFETY: AVX2 is available (checked at dispatch time via target_feature)
    let round = _mm256_set1_epi16(PW_1024);

    for row in 0..h {
        let tmp1_row = &tmp1[row * w..][..w];
        let tmp2_row = &tmp2[row * w..][..w];
        let dst_row = &mut dst[row * dst_stride..][..w];

        // Process 32 pixels at a time (64 bytes of i16 input → 32 bytes of u8 output)
        let mut col = 0;
        while col + 32 <= w {
            let t1_lo = loadu_256!(&tmp1_row[col..col + 16], [i16; 16]);
            let t1_hi = loadu_256!(&tmp1_row[col + 16..col + 32], [i16; 16]);
            let t2_lo = loadu_256!(&tmp2_row[col..col + 16], [i16; 16]);
            let t2_hi = loadu_256!(&tmp2_row[col + 16..col + 32], [i16; 16]);

            // Add: tmp1 + tmp2
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
            storeu_256!(&mut dst_row[col..col + 32], [u8; 32], result);

            col += 32;
        }

        // Process 16 pixels at a time for remainder
        while col + 16 <= w {
            let t1 = loadu_256!(&tmp1_row[col..col + 16], [i16; 16]);
            let t2 = loadu_256!(&tmp2_row[col..col + 16], [i16; 16]);

            let sum = _mm256_add_epi16(t1, t2);
            let avg = _mm256_mulhrs_epi16(sum, round);

            // Pack within lanes and extract lower 128 bits
            let packed = _mm256_packus_epi16(avg, avg);
            let lo = _mm256_castsi256_si128(packed);
            let hi = _mm256_extracti128_si256(packed, 1);
            let result = _mm_unpacklo_epi64(lo, hi);

            storeu_128!(&mut dst_row[col..col + 16], [u8; 16], result);

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

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn avg_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    _bitdepth_max: i32,
    _dst: *const FFISafe<PicOffset>,
) {
    let dst = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h as usize * dst_stride as usize)
    };
    avg_8bpc_avx2_safe(
        Desktop64::forge_token_dangerously(),
        dst,
        dst_stride as usize,
        tmp1,
        tmp2,
        w,
        h,
    )
}

/// AVG operation for 16-bit pixels using AVX2
///
/// For 10/12-bit content. Averages two 16-bit intermediate buffers.
///
/// # Safety
///
/// Same as avg_8bpc_avx2, plus bitdepth_max must be correct for the content.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn avg_16bpc_avx2_safe(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let w = w as usize;
    let h = h as usize;

    // intermediate_bits = 14 - bitdepth: 4 for 10-bit, 2 for 12-bit
    let intermediate_bits = if (bitdepth_max >> 11) != 0 {
        2i32
    } else {
        4i32
    };
    // sh = intermediate_bits + 1, rnd = (1 << intermediate_bits) + PREP_BIAS * 2
    let sh = intermediate_bits + 1;
    let rnd = (1 << intermediate_bits) + 8192 * 2; // PREP_BIAS = 8192 for 16bpc
    let max = bitdepth_max as i32;

    let rnd_vec = _mm256_set1_epi32(rnd);
    let zero = _mm256_setzero_si256();
    let max_vec = _mm256_set1_epi32(max);

    for row in 0..h {
        let tmp1_row = &tmp1[row * w..][..w];
        let tmp2_row = &tmp2[row * w..][..w];
        // dst is bytes, but we write u16 pixels — each pixel is 2 bytes
        let dst_row_bytes = &mut dst[row * dst_stride..][..w * 2];
        // Reinterpret as u16 slice
        let dst_row: &mut [u16] = zerocopy::FromBytes::mut_slice_from(dst_row_bytes).unwrap();

        let mut col = 0usize;

        while col + 16 <= w {
            let t1 = loadu_256!(&tmp1_row[col..col + 16], [i16; 16]);
            let t2 = loadu_256!(&tmp2_row[col..col + 16], [i16; 16]);

            let t1_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(t1));
            let t2_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(t2));
            let t1_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(t1, 1));
            let t2_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(t2, 1));

            let sum_lo = _mm256_add_epi32(_mm256_add_epi32(t1_lo, t2_lo), rnd_vec);
            let sum_hi = _mm256_add_epi32(_mm256_add_epi32(t1_hi, t2_hi), rnd_vec);

            // srai_epi32 requires const immediate; branch on sh
            let (result_lo, result_hi) = if sh == 3 {
                (_mm256_srai_epi32(sum_lo, 3), _mm256_srai_epi32(sum_hi, 3))
            } else {
                (_mm256_srai_epi32(sum_lo, 5), _mm256_srai_epi32(sum_hi, 5))
            };

            let clamped_lo = _mm256_min_epi32(_mm256_max_epi32(result_lo, zero), max_vec);
            let clamped_hi = _mm256_min_epi32(_mm256_max_epi32(result_hi, zero), max_vec);

            let packed = _mm256_packus_epi32(clamped_lo, clamped_hi);
            let packed = _mm256_permute4x64_epi64(packed, 0b11011000);

            storeu_256!(&mut dst_row[col..col + 16], [u16; 16], packed);

            col += 16;
        }

        while col < w {
            let sum = tmp1_row[col] as i32 + tmp2_row[col] as i32;
            let val = ((sum + rnd) >> sh).clamp(0, max) as u16;
            dst_row[col] = val;
            col += 1;
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn avg_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    bitdepth_max: i32,
    _dst: *const FFISafe<PicOffset>,
) {
    let dst = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h as usize * dst_stride as usize)
    };
    avg_16bpc_avx2_safe(
        Desktop64::forge_token_dangerously(),
        dst,
        dst_stride as usize,
        tmp1,
        tmp2,
        w,
        h,
        bitdepth_max,
    )
}

/// SSE4.1 fallback for avg_8bpc
///
/// # Safety
///
/// Same as avg_8bpc_avx2.
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "sse4.1")]
pub unsafe extern "C" fn avg_8bpc_sse4(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    bitdepth_max: i32,
    _dst: *const FFISafe<PicOffset>,
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
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
pub unsafe extern "C" fn avg_scalar(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    _bitdepth_max: i32,
    _dst: *const FFISafe<PicOffset>,
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
#[arcane]
fn w_avg_8bpc_avx2_safe(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    weight: i32,
) {
    let mut dst = dst.flex_mut();
    let w = w as usize;
    let h = h as usize;

    let (tmp1_ptr, tmp2_ptr, weight_scaled) = if weight > 7 {
        (tmp1, tmp2, ((weight - 16) << 12) as i16)
    } else {
        (tmp2, tmp1, ((-weight) << 12) as i16)
    };

    let weight_vec = _mm256_set1_epi16(weight_scaled);
    let round = _mm256_set1_epi16(PW_2048);

    for row in 0..h {
        let tmp1_row = &tmp1_ptr[row * w..][..w];
        let tmp2_row = &tmp2_ptr[row * w..][..w];
        let dst_row = &mut dst[row * dst_stride..][..w];

        let mut col = 0;
        while col + 32 <= w {
            let t1_lo = loadu_256!(&tmp1_row[col..col + 16], [i16; 16]);
            let t1_hi = loadu_256!(&tmp1_row[col + 16..col + 32], [i16; 16]);
            let t2_lo = loadu_256!(&tmp2_row[col..col + 16], [i16; 16]);
            let t2_hi = loadu_256!(&tmp2_row[col + 16..col + 32], [i16; 16]);

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

            storeu_256!(&mut dst_row[col..col + 32], [u8; 32], result);

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

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
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
    _dst: *const FFISafe<PicOffset>,
) {
    let dst = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h as usize * dst_stride as usize)
    };
    w_avg_8bpc_avx2_safe(
        Desktop64::forge_token_dangerously(),
        dst,
        dst_stride as usize,
        tmp1,
        tmp2,
        w,
        h,
        weight,
    )
}

/// Weighted average for 16-bit pixels using AVX2
///
/// Same as w_avg_8bpc_avx2_safe but for 16-bit content.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn w_avg_16bpc_avx2_safe(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    weight: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let w = w as usize;
    let h = h as usize;

    // intermediate_bits = 14 - bitdepth: 4 for 10-bit, 2 for 12-bit
    let intermediate_bits = if (bitdepth_max >> 11) != 0 {
        2i32
    } else {
        4i32
    };
    // sh = intermediate_bits + 4, rnd = (8 << intermediate_bits) + PREP_BIAS * 16
    let sh = intermediate_bits + 4;
    let rnd = (8 << intermediate_bits) + 8192 * 16; // PREP_BIAS = 8192 for 16bpc
    let max = bitdepth_max as i32;
    let inv_weight = 16 - weight;

    let rnd_vec = _mm256_set1_epi32(rnd);
    let zero = _mm256_setzero_si256();
    let max_vec = _mm256_set1_epi32(max);
    let weight_vec = _mm256_set1_epi32(weight);
    let inv_weight_vec = _mm256_set1_epi32(inv_weight);

    for row in 0..h {
        let tmp1_row = &tmp1[row * w..][..w];
        let tmp2_row = &tmp2[row * w..][..w];
        let dst_row_bytes = &mut dst[row * dst_stride..][..w * 2];
        let dst_row: &mut [u16] = zerocopy::FromBytes::mut_slice_from(dst_row_bytes).unwrap();

        let mut col = 0usize;

        // Process 8 pixels at a time (need 32-bit arithmetic)
        while col + 8 <= w {
            // Load 8x i16 values and sign-extend to 32-bit
            let t1_16 = loadu_128!(&tmp1_row[col..col + 8], [i16; 8]);
            let t2_16 = loadu_128!(&tmp2_row[col..col + 8], [i16; 8]);
            let t1 = _mm256_cvtepi16_epi32(t1_16);
            let t2 = _mm256_cvtepi16_epi32(t2_16);

            // val = a * weight + b * inv_weight + rnd
            let term1 = _mm256_mullo_epi32(t1, weight_vec);
            let term2 = _mm256_mullo_epi32(t2, inv_weight_vec);
            let sum = _mm256_add_epi32(_mm256_add_epi32(term1, term2), rnd_vec);

            // srai_epi32 requires const immediate; branch on sh
            let result = if sh == 6 {
                _mm256_srai_epi32(sum, 6)
            } else {
                _mm256_srai_epi32(sum, 8)
            };

            // Clamp to [0, max]
            let clamped = _mm256_min_epi32(_mm256_max_epi32(result, zero), max_vec);

            // Pack 32-bit to 16-bit
            let packed = _mm256_packus_epi32(clamped, clamped);
            let lo128 = _mm256_castsi256_si128(packed);
            let hi128 = _mm256_extracti128_si256(packed, 1);
            let result_128 = _mm_unpacklo_epi64(lo128, hi128);

            // Store 8x u16
            storeu_128!(&mut dst_row[col..col + 8], [u16; 8], result_128);

            col += 8;
        }

        // Handle remaining pixels with scalar
        while col < w {
            let a = tmp1_row[col] as i32;
            let b = tmp2_row[col] as i32;
            let val = (a * weight + b * inv_weight + rnd) >> sh;
            dst_row[col] = val.clamp(0, max) as u16;
            col += 1;
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
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
    _dst: *const FFISafe<PicOffset>,
) {
    let dst = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h as usize * dst_stride as usize)
    };
    w_avg_16bpc_avx2_safe(
        Desktop64::forge_token_dangerously(),
        dst,
        dst_stride as usize,
        tmp1,
        tmp2,
        w,
        h,
        weight,
        bitdepth_max,
    )
}

/// Scalar fallback for w_avg
///
/// # Safety
///
/// Same as w_avg_8bpc_avx2.
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
pub unsafe extern "C" fn w_avg_scalar(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    weight: i32,
    _bitdepth_max: i32,
    _dst: *const FFISafe<PicOffset>,
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
#[cfg(target_arch = "x86_64")]
#[arcane]
fn mask_8bpc_avx2_safe(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask: &[u8],
) {
    let mut dst = dst.flex_mut();
    let mask = mask.flex();
    let w = w as usize;
    let h = h as usize;

    let rnd = _mm256_set1_epi32(512);

    for row in 0..h {
        let tmp1_row = &tmp1[row * w..][..w];
        let tmp2_row = &tmp2[row * w..][..w];
        let mask_row = &mask[row * w..][..w];
        let dst_row = &mut dst[row * dst_stride..][..w];

        let mut col = 0usize;

        // Process 16 pixels at a time with AVX2
        while col + 16 <= w {
            let t1_lo = loadu_256!(&tmp1_row[col..col + 16], [i16; 16]);
            let t2_lo = loadu_256!(&tmp2_row[col..col + 16], [i16; 16]);
            let mask_bytes = loadu_128!(&mask_row[col..col + 16], [u8; 16]);
            let mask_lo = _mm256_cvtepu8_epi16(mask_bytes);

            // Compute (tmp1 - tmp2)
            let diff = _mm256_sub_epi16(t1_lo, t2_lo);

            // Process low 8 elements (need 32-bit for tmp2*64 which can overflow 16-bit)
            let diff_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(diff));
            let mask_lo_32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(mask_lo));
            let t2_lo_32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(t2_lo));

            let prod_lo = _mm256_mullo_epi32(diff_lo, mask_lo_32);
            let t2_64_lo = _mm256_slli_epi32(t2_lo_32, 6);
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
            let result_16 = _mm256_permute4x64_epi64(result_16, 0b11011000);

            // Pack to 8-bit with unsigned saturation
            let result_8 = _mm256_packus_epi16(result_16, result_16);
            let result_8 = _mm256_permute4x64_epi64(result_8, 0b11011000);

            storeu_128!(
                &mut dst_row[col..col + 16],
                [u8; 16],
                _mm256_castsi256_si128(result_8)
            );

            col += 16;
        }

        // Handle remaining pixels with scalar
        while col < w {
            let a = tmp1_row[col] as i32;
            let b = tmp2_row[col] as i32;
            let m = mask_row[col] as i32;
            let val = (a * m + b * (64 - m) + 512) >> 10;
            dst_row[col] = val.clamp(0, 255) as u8;
            col += 1;
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
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
    _dst: *const FFISafe<PicOffset>,
) {
    let dst = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h as usize * dst_stride as usize)
    };
    let mask = unsafe { std::slice::from_raw_parts(mask_ptr, (w * h) as usize) };
    mask_8bpc_avx2_safe(
        Desktop64::forge_token_dangerously(),
        dst,
        dst_stride as usize,
        tmp1,
        tmp2,
        w,
        h,
        mask,
    )
}

/// Mask blend for 16-bit pixels
#[cfg(target_arch = "x86_64")]
#[arcane]
fn mask_16bpc_avx2_safe(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask: &[u8],
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mask = mask.flex();
    let w = w as usize;
    let h = h as usize;
    let max = bitdepth_max as i32;

    // intermediate_bits = 14 - bitdepth: 4 for 10-bit, 2 for 12-bit
    let intermediate_bits = if (bitdepth_max >> 11) != 0 {
        2i32
    } else {
        4i32
    };
    // sh = intermediate_bits + 6, rnd = (32 << intermediate_bits) + PREP_BIAS * 64
    let sh = intermediate_bits + 6;
    let rnd = (32 << intermediate_bits) + 8192 * 64; // PREP_BIAS = 8192 for 16bpc

    let rnd_vec = _mm256_set1_epi32(rnd);
    let zero = _mm256_setzero_si256();
    let max_vec = _mm256_set1_epi32(max);
    let sixty_four = _mm256_set1_epi32(64);

    for row in 0..h {
        let tmp1_row = &tmp1[row * w..][..w];
        let tmp2_row = &tmp2[row * w..][..w];
        let mask_row = &mask[row * w..][..w];
        let dst_row_bytes = &mut dst[row * dst_stride..][..w * 2];
        let dst_row: &mut [u16] = zerocopy::FromBytes::mut_slice_from(dst_row_bytes).unwrap();

        let mut col = 0usize;

        // Process 8 pixels at a time with 32-bit arithmetic
        while col + 8 <= w {
            let t1_16 = loadu_128!(&tmp1_row[col..col + 8], [i16; 8]);
            let t2_16 = loadu_128!(&tmp2_row[col..col + 8], [i16; 8]);
            let t1 = _mm256_cvtepi16_epi32(t1_16);
            let t2 = _mm256_cvtepi16_epi32(t2_16);

            // Load 8 bytes of mask via zero-padded 16-byte array
            let mut mask_pad = [0u8; 16];
            mask_pad[..8].copy_from_slice(&mask_row[col..col + 8]);
            let mask_bytes = loadu_128!(&mask_pad);
            let mask_32 = _mm256_cvtepu8_epi32(mask_bytes);

            let inv_mask = _mm256_sub_epi32(sixty_four, mask_32);

            let term1 = _mm256_mullo_epi32(t1, mask_32);
            let term2 = _mm256_mullo_epi32(t2, inv_mask);
            let sum = _mm256_add_epi32(_mm256_add_epi32(term1, term2), rnd_vec);

            // srai_epi32 requires const immediate; branch on sh
            let result = if sh == 8 {
                _mm256_srai_epi32(sum, 8)
            } else {
                _mm256_srai_epi32(sum, 10)
            };
            let clamped = _mm256_min_epi32(_mm256_max_epi32(result, zero), max_vec);

            let packed = _mm256_packus_epi32(clamped, clamped);
            let lo128 = _mm256_castsi256_si128(packed);
            let hi128 = _mm256_extracti128_si256(packed, 1);
            let result_128 = _mm_unpacklo_epi64(lo128, hi128);

            storeu_128!(&mut dst_row[col..col + 8], [u16; 8], result_128);

            col += 8;
        }

        // Handle remaining pixels with scalar
        while col < w {
            let a = tmp1_row[col] as i32;
            let b = tmp2_row[col] as i32;
            let m = mask_row[col] as i32;
            let val = (a * m + b * (64 - m) + rnd) >> sh;
            dst_row[col] = val.clamp(0, max) as u16;
            col += 1;
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
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
    _dst: *const FFISafe<PicOffset>,
) {
    let dst = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h as usize * dst_stride as usize)
    };
    let mask = unsafe { std::slice::from_raw_parts(mask_ptr, (w * h) as usize) };
    mask_16bpc_avx2_safe(
        Desktop64::forge_token_dangerously(),
        dst,
        dst_stride as usize,
        tmp1,
        tmp2,
        w,
        h,
        mask,
        bitdepth_max,
    )
}

/// Scalar fallback for mask
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
pub unsafe extern "C" fn mask_scalar(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask_ptr: *const u8,
    _bitdepth_max: i32,
    _dst: *const FFISafe<PicOffset>,
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
#[cfg(target_arch = "x86_64")]
#[arcane]
fn blend_8bpc_avx2_safe(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp: &[u8],
    w: i32,
    h: i32,
    mask: &[u8],
) {
    let mut dst = dst.flex_mut();
    let tmp = tmp.flex();
    let mask = mask.flex();
    let w = w as usize;
    let h = h as usize;

    let sixty_four = _mm256_set1_epi16(64);
    let rnd = _mm256_set1_epi16(32);

    for row in 0..h {
        let dst_row = &mut dst[row * dst_stride..][..w];
        let tmp_row = &tmp[row * w..][..w];
        let mask_row = &mask[row * w..][..w];

        let mut col = 0usize;

        // Process 16 pixels at a time with AVX2
        while col + 16 <= w {
            let dst_bytes = loadu_128!(&dst_row[col..col + 16], [u8; 16]);
            let tmp_bytes = loadu_128!(&tmp_row[col..col + 16], [u8; 16]);
            let mask_bytes = loadu_128!(&mask_row[col..col + 16], [u8; 16]);

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

            storeu_128!(
                &mut dst_row[col..col + 16],
                [u8; 16],
                _mm256_castsi256_si128(result_8)
            );

            col += 16;
        }

        // Handle remaining pixels with scalar
        while col < w {
            let a = dst_row[col] as u32;
            let b = tmp_row[col] as u32;
            let m = mask_row[col] as u32;
            let val = (a * (64 - m) + b * m + 32) >> 6;
            dst_row[col] = val as u8;
            col += 1;
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn blend_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp: *const [DynPixel; SCRATCH_INTER_INTRA_BUF_LEN],
    w: i32,
    h: i32,
    mask_ptr: *const u8,
    _dst: *const FFISafe<PicOffset>,
) {
    let dst = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h as usize * dst_stride as usize)
    };
    let tmp_slice = unsafe { std::slice::from_raw_parts(tmp as *const u8, (w * h) as usize) };
    let mask = unsafe { std::slice::from_raw_parts(mask_ptr, (w * h) as usize) };
    blend_8bpc_avx2_safe(
        Desktop64::forge_token_dangerously(),
        dst,
        dst_stride as usize,
        tmp_slice,
        w,
        h,
        mask,
    )
}

/// Blend pixels for 16-bit
#[cfg(target_arch = "x86_64")]
#[arcane]
fn blend_16bpc_avx2_safe(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp: &[u8],
    w: i32,
    h: i32,
    mask: &[u8],
) {
    let mut dst = dst.flex_mut();
    let tmp = tmp.flex();
    let mask = mask.flex();
    let w = w as usize;
    let h = h as usize;

    let rnd = _mm256_set1_epi32(32);
    let sixty_four = _mm256_set1_epi32(64);

    for row in 0..h {
        let dst_row_bytes = &mut dst[row * dst_stride..][..w * 2];
        let dst_row: &mut [u16] = zerocopy::FromBytes::mut_slice_from(dst_row_bytes).unwrap();
        let tmp_row_bytes = &tmp[row * w * 2..][..w * 2];
        let tmp_row: &[u16] = <u16 as zerocopy::FromBytes>::slice_from(tmp_row_bytes).unwrap();
        let mask_row = &mask[row * w..][..w];

        let mut col = 0usize;

        // Process 8 pixels at a time with 32-bit arithmetic
        while col + 8 <= w {
            let dst_16 = loadu_128!(&dst_row[col..col + 8], [u16; 8]);
            let tmp_16 = loadu_128!(&tmp_row[col..col + 8], [u16; 8]);
            let dst_32 = _mm256_cvtepu16_epi32(dst_16);
            let tmp_32 = _mm256_cvtepu16_epi32(tmp_16);

            // Load 8 bytes of mask via zero-padded 16-byte array
            let mut mask_pad = [0u8; 16];
            mask_pad[..8].copy_from_slice(&mask_row[col..col + 8]);
            let mask_bytes = loadu_128!(&mask_pad);
            let mask_32 = _mm256_cvtepu8_epi32(mask_bytes);

            let inv_mask = _mm256_sub_epi32(sixty_four, mask_32);

            let term1 = _mm256_mullo_epi32(dst_32, inv_mask);
            let term2 = _mm256_mullo_epi32(tmp_32, mask_32);
            let sum = _mm256_add_epi32(_mm256_add_epi32(term1, term2), rnd);

            let result = _mm256_srli_epi32(sum, 6);

            let packed = _mm256_packus_epi32(result, result);
            let lo128 = _mm256_castsi256_si128(packed);
            let hi128 = _mm256_extracti128_si256(packed, 1);
            let result_128 = _mm_unpacklo_epi64(lo128, hi128);

            storeu_128!(&mut dst_row[col..col + 8], [u16; 8], result_128);

            col += 8;
        }

        // Handle remaining pixels with scalar
        while col < w {
            let a = dst_row[col] as u32;
            let b = tmp_row[col] as u32;
            let m = mask_row[col] as u32;
            let val = (a * (64 - m) + b * m + 32) >> 6;
            dst_row[col] = val as u16;
            col += 1;
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn blend_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp: *const [DynPixel; SCRATCH_INTER_INTRA_BUF_LEN],
    w: i32,
    h: i32,
    mask_ptr: *const u8,
    _dst: *const FFISafe<PicOffset>,
) {
    let dst = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h as usize * dst_stride as usize)
    };
    let tmp_slice = unsafe { std::slice::from_raw_parts(tmp as *const u8, (w * h) as usize * 2) };
    let mask = unsafe { std::slice::from_raw_parts(mask_ptr, (w * h) as usize) };
    blend_16bpc_avx2_safe(
        Desktop64::forge_token_dangerously(),
        dst,
        dst_stride as usize,
        tmp_slice,
        w,
        h,
        mask,
    )
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
#[arcane]
fn blend_v_8bpc_avx2_safe(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp: &[u8],
    w: i32,
    h: i32,
) {
    let mut dst = dst.flex_mut();
    let tmp = tmp.flex();
    let w = w as usize;
    let h = h as usize;
    // blend_v: mask indexed by column, offset by w, only process w*3/4 columns
    let obmc_mask = &dav1d_obmc_masks[w..];
    let w_eff = w * 3 >> 2;

    let rnd = _mm256_set1_epi16(32);
    let sixty_four = _mm256_set1_epi16(64);

    for row in 0..h {
        let dst_row = &mut dst[row * dst_stride..][..w_eff];
        // tmp uses full w stride even though we only write w_eff columns
        let tmp_row = &tmp[row * w..][..w_eff];

        let mut col = 0usize;

        // Process 16 pixels at a time
        while col + 16 <= w_eff {
            let dst_bytes = loadu_128!(&dst_row[col..col + 16], [u8; 16]);
            let tmp_bytes = loadu_128!(&tmp_row[col..col + 16], [u8; 16]);
            let mask_bytes = loadu_128!(&obmc_mask[col..col + 16], [u8; 16]);

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

            storeu_128!(
                &mut dst_row[col..col + 16],
                [u8; 16],
                _mm256_castsi256_si128(result_8)
            );

            col += 16;
        }

        // Handle remaining pixels
        while col < w_eff {
            let a = dst_row[col] as u32;
            let b = tmp_row[col] as u32;
            let m = obmc_mask[col] as u32;
            let val = (a * (64 - m) + b * m + 32) >> 6;
            dst_row[col] = val as u8;
            col += 1;
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn blend_v_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp: *const [DynPixel; SCRATCH_LAP_LEN],
    w: i32,
    h: i32,
    _dst: *const FFISafe<PicOffset>,
) {
    let dst = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h as usize * dst_stride as usize)
    };
    let tmp_slice = unsafe { std::slice::from_raw_parts(tmp as *const u8, (w * h) as usize) };
    blend_v_8bpc_avx2_safe(
        Desktop64::forge_token_dangerously(),
        dst,
        dst_stride as usize,
        tmp_slice,
        w,
        h,
    )
}

/// Horizontal blend (overlapped block motion compensation)
#[cfg(target_arch = "x86_64")]
#[arcane]
fn blend_h_8bpc_avx2_safe(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp: &[u8],
    w: i32,
    h: i32,
) {
    let mut dst = dst.flex_mut();
    let tmp = tmp.flex();
    let w = w as usize;
    let h = h as usize;
    // blend_h: mask indexed by row, offset by h, only process h*3/4 rows
    let obmc_mask = &dav1d_obmc_masks[h..];
    let h_eff = h * 3 >> 2;

    let rnd = _mm256_set1_epi16(32);
    let sixty_four = _mm256_set1_epi16(64);

    for row in 0..h_eff {
        let dst_row = &mut dst[row * dst_stride..][..w];
        let tmp_row = &tmp[row * w..][..w];
        let m = obmc_mask[row];

        let mask_16 = _mm256_set1_epi16(m as i16);
        let inv_mask = _mm256_sub_epi16(sixty_four, mask_16);

        let mut col = 0usize;

        // Process 16 pixels at a time
        while col + 16 <= w {
            let dst_bytes = loadu_128!(&dst_row[col..col + 16], [u8; 16]);
            let tmp_bytes = loadu_128!(&tmp_row[col..col + 16], [u8; 16]);

            let dst_16 = _mm256_cvtepu8_epi16(dst_bytes);
            let tmp_16 = _mm256_cvtepu8_epi16(tmp_bytes);

            let term1 = _mm256_mullo_epi16(dst_16, inv_mask);
            let term2 = _mm256_mullo_epi16(tmp_16, mask_16);
            let sum = _mm256_add_epi16(_mm256_add_epi16(term1, term2), rnd);
            let result_16 = _mm256_srli_epi16(sum, 6);

            let result_8 = _mm256_packus_epi16(result_16, result_16);
            let result_8 = _mm256_permute4x64_epi64(result_8, 0b11011000);

            storeu_128!(
                &mut dst_row[col..col + 16],
                [u8; 16],
                _mm256_castsi256_si128(result_8)
            );

            col += 16;
        }

        // Handle remaining pixels
        while col < w {
            let a = dst_row[col] as u32;
            let b = tmp_row[col] as u32;
            let val = (a * (64 - m as u32) + b * m as u32 + 32) >> 6;
            dst_row[col] = val as u8;
            col += 1;
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn blend_h_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp: *const [DynPixel; SCRATCH_LAP_LEN],
    w: i32,
    h: i32,
    _dst: *const FFISafe<PicOffset>,
) {
    let dst = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h as usize * dst_stride as usize)
    };
    let tmp_slice = unsafe { std::slice::from_raw_parts(tmp as *const u8, (w * h) as usize) };
    blend_h_8bpc_avx2_safe(
        Desktop64::forge_token_dangerously(),
        dst,
        dst_stride as usize,
        tmp_slice,
        w,
        h,
    )
}

/// 16-bit blend_v
#[cfg(target_arch = "x86_64")]
#[arcane]
fn blend_v_16bpc_avx2_safe(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp: &[u8],
    w: i32,
    h: i32,
) {
    let mut dst = dst.flex_mut();
    let tmp = tmp.flex();
    let w = w as usize;
    let h = h as usize;
    // blend_v: mask indexed by column, offset by w, only process w*3/4 columns
    let obmc_mask = &dav1d_obmc_masks[w..];
    let w_eff = w * 3 >> 2;

    let rnd = _mm256_set1_epi32(32);
    let sixty_four = _mm256_set1_epi32(64);

    for row in 0..h {
        let dst_row_bytes = &mut dst[row * dst_stride..][..w_eff * 2];
        let dst_row: &mut [u16] = zerocopy::FromBytes::mut_slice_from(dst_row_bytes).unwrap();
        // tmp uses full w stride
        let tmp_row_bytes = &tmp[row * w * 2..][..w_eff * 2];
        let tmp_row: &[u16] = <u16 as zerocopy::FromBytes>::slice_from(tmp_row_bytes).unwrap();

        let mut col = 0usize;

        // Process 8 pixels at a time with 32-bit arithmetic
        while col + 8 <= w_eff {
            let dst_16 = loadu_128!(&dst_row[col..col + 8], [u16; 8]);
            let tmp_16 = loadu_128!(&tmp_row[col..col + 8], [u16; 8]);
            let dst_32 = _mm256_cvtepu16_epi32(dst_16);
            let tmp_32 = _mm256_cvtepu16_epi32(tmp_16);

            // Load 8 bytes of mask via zero-padded 16-byte array
            let mut mask_pad = [0u8; 16];
            mask_pad[..8].copy_from_slice(&obmc_mask[col..col + 8]);
            let mask_bytes = loadu_128!(&mask_pad);
            let mask_32 = _mm256_cvtepu8_epi32(mask_bytes);
            let inv_mask = _mm256_sub_epi32(sixty_four, mask_32);

            let term1 = _mm256_mullo_epi32(dst_32, inv_mask);
            let term2 = _mm256_mullo_epi32(tmp_32, mask_32);
            let sum = _mm256_add_epi32(_mm256_add_epi32(term1, term2), rnd);

            let result = _mm256_srli_epi32(sum, 6);

            let packed = _mm256_packus_epi32(result, result);
            let lo128 = _mm256_castsi256_si128(packed);
            let hi128 = _mm256_extracti128_si256(packed, 1);
            let result_128 = _mm_unpacklo_epi64(lo128, hi128);

            storeu_128!(&mut dst_row[col..col + 8], [u16; 8], result_128);

            col += 8;
        }

        // Handle remaining pixels with scalar
        while col < w_eff {
            let a = dst_row[col] as u32;
            let b = tmp_row[col] as u32;
            let m = obmc_mask[col] as u32;
            let val = (a * (64 - m) + b * m + 32) >> 6;
            dst_row[col] = val as u16;
            col += 1;
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn blend_v_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp: *const [DynPixel; SCRATCH_LAP_LEN],
    w: i32,
    h: i32,
    _dst: *const FFISafe<PicOffset>,
) {
    let dst = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h as usize * dst_stride as usize)
    };
    let tmp_slice = unsafe { std::slice::from_raw_parts(tmp as *const u8, (w * h) as usize * 2) };
    blend_v_16bpc_avx2_safe(
        Desktop64::forge_token_dangerously(),
        dst,
        dst_stride as usize,
        tmp_slice,
        w,
        h,
    )
}

/// 16-bit blend_h
#[cfg(target_arch = "x86_64")]
#[arcane]
fn blend_h_16bpc_avx2_safe(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp: &[u8],
    w: i32,
    h: i32,
) {
    let mut dst = dst.flex_mut();
    let tmp = tmp.flex();
    let w = w as usize;
    let h = h as usize;
    // blend_h: mask indexed by row, offset by h, only process h*3/4 rows
    let obmc_mask = &dav1d_obmc_masks[h..];
    let h_eff = h * 3 >> 2;

    let rnd = _mm256_set1_epi32(32);
    let sixty_four = _mm256_set1_epi32(64);

    for row in 0..h_eff {
        let dst_row_bytes = &mut dst[row * dst_stride..][..w * 2];
        let dst_row: &mut [u16] = zerocopy::FromBytes::mut_slice_from(dst_row_bytes).unwrap();
        let tmp_row_bytes = &tmp[row * w * 2..][..w * 2];
        let tmp_row: &[u16] = <u16 as zerocopy::FromBytes>::slice_from(tmp_row_bytes).unwrap();
        let m = obmc_mask[row] as u32;

        let mask_32 = _mm256_set1_epi32(m as i32);
        let inv_mask = _mm256_sub_epi32(sixty_four, mask_32);

        let mut col = 0usize;

        // Process 8 pixels at a time with 32-bit arithmetic
        while col + 8 <= w {
            let dst_16 = loadu_128!(&dst_row[col..col + 8], [u16; 8]);
            let tmp_16 = loadu_128!(&tmp_row[col..col + 8], [u16; 8]);
            let dst_32 = _mm256_cvtepu16_epi32(dst_16);
            let tmp_32 = _mm256_cvtepu16_epi32(tmp_16);

            let term1 = _mm256_mullo_epi32(dst_32, inv_mask);
            let term2 = _mm256_mullo_epi32(tmp_32, mask_32);
            let sum = _mm256_add_epi32(_mm256_add_epi32(term1, term2), rnd);

            let result = _mm256_srli_epi32(sum, 6);

            let packed = _mm256_packus_epi32(result, result);
            let lo128 = _mm256_castsi256_si128(packed);
            let hi128 = _mm256_extracti128_si256(packed, 1);
            let result_128 = _mm_unpacklo_epi64(lo128, hi128);

            storeu_128!(&mut dst_row[col..col + 8], [u16; 8], result_128);

            col += 8;
        }

        // Handle remaining pixels with scalar
        while col < w {
            let a = dst_row[col] as u32;
            let b = tmp_row[col] as u32;
            let val = (a * (64 - m) + b * m + 32) >> 6;
            dst_row[col] = val as u16;
            col += 1;
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn blend_h_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp: *const [DynPixel; SCRATCH_LAP_LEN],
    w: i32,
    h: i32,
    _dst: *const FFISafe<PicOffset>,
) {
    let dst = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h as usize * dst_stride as usize)
    };
    let tmp_slice = unsafe { std::slice::from_raw_parts(tmp as *const u8, (w * h) as usize * 2) };
    blend_h_16bpc_avx2_safe(
        Desktop64::forge_token_dangerously(),
        dst,
        dst_stride as usize,
        tmp_slice,
        w,
        h,
    )
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
    Some(&dav1d_mc_subpel_filters[i][m])
}

/// Horizontal 8-tap filter for a row of 8bpc pixels
///
/// Processes `w` pixels starting at `src`, writing to `dst` (i16 intermediate)
/// Formula: sum(coeff[i] * src[x + i - 3]) for i in 0..8, then round and shift
#[cfg(target_arch = "x86_64")]
#[rite]
fn h_filter_8tap_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [i16],
    src: &[u8],
    w: usize,
    filter: &[i8; 8],
    sh: u8,
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    // by the target_feature attribute, and pointer operations are valid per caller contract.
    // For horizontal filtering, we need to load 8 consecutive pixels for each output
    // The source pointer is already offset by -3 (pointing to tap 0)

    // Broadcast filter coefficients
    // We'll use _mm256_maddubs_epi16 which does a[0]*b[0]+a[1]*b[1] for pairs
    // So we need to arrange coefficients for this: [c0,c1,c2,c3,c4,c5,c6,c7] repeated
    let coeff_01 = _mm256_set1_epi16(((filter[1] as u8 as i16) << 8) | (filter[0] as u8 as i16));
    let coeff_23 = _mm256_set1_epi16(((filter[3] as u8 as i16) << 8) | (filter[2] as u8 as i16));
    let coeff_45 = _mm256_set1_epi16(((filter[5] as u8 as i16) << 8) | (filter[4] as u8 as i16));
    let coeff_67 = _mm256_set1_epi16(((filter[7] as u8 as i16) << 8) | (filter[6] as u8 as i16));

    let rnd = _mm256_set1_epi16((1i16 << sh) >> 1);

    let mut col = 0usize;

    // Process 16 pixels at a time
    while col + 16 <= w {
        // Load source bytes - we need 8 bytes for each output pixel, offset by tap position
        // s offset = col

        // Load bytes at various offsets for the 8-tap filter
        let src_0_15 = loadu_128!(<&[u8; 16]>::try_from(&src[col..col + 16]).unwrap());
        let src_1_16 = loadu_128!(<&[u8; 16]>::try_from(&src[col + 1..col + 17]).unwrap());
        let src_2_17 = loadu_128!(<&[u8; 16]>::try_from(&src[col + 2..col + 18]).unwrap());
        let src_3_18 = loadu_128!(<&[u8; 16]>::try_from(&src[col + 3..col + 19]).unwrap());
        let src_4_19 = loadu_128!(<&[u8; 16]>::try_from(&src[col + 4..col + 20]).unwrap());
        let src_5_20 = loadu_128!(<&[u8; 16]>::try_from(&src[col + 5..col + 21]).unwrap());
        let src_6_21 = loadu_128!(<&[u8; 16]>::try_from(&src[col + 6..col + 22]).unwrap());
        let src_7_22 = loadu_128!(<&[u8; 16]>::try_from(&src[col + 7..col + 23]).unwrap());

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
        storeu_256!(
            <&mut [i16; 16]>::try_from(&mut dst[col..col + 16]).unwrap(),
            result
        );

        col += 16;
    }

    // Scalar fallback for remaining pixels
    while col < w {
        // s offset = col
        let mut sum = 0i32;
        for i in 0..8 {
            sum += filter[i] as i32 * src[col + i] as i32;
        }
        dst[col] = ((sum + ((1 << sh) >> 1)) >> sh) as i16;
        col += 1;
    }
}
#[cfg(feature = "asm")]
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
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe { h_filter_8tap_8bpc_avx2_inner(token, dst, src, w, filter, sh) }
}

/// Vertical 8-tap filter from intermediate buffer to output
///
/// Processes `w` pixels for one row, reading from `mid` (8 rows), writing to `dst`
#[cfg(target_arch = "x86_64")]
#[rite]
fn v_filter_8tap_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    mid: &[[i16; MID_STRIDE]],
    w: usize,
    filter: &[i8; 8],
    sh: u8,
    max: i32,
) {
    let mut dst = dst.flex_mut();
    // by the target_feature attribute, and pointer operations are valid per caller contract.
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
        let m0 = _mm256_cvtepi16_epi32(loadu_128!(
            <&[i16; 8]>::try_from(&mid[0][col..col + 8]).unwrap()
        ));
        let m1 = _mm256_cvtepi16_epi32(loadu_128!(
            <&[i16; 8]>::try_from(&mid[1][col..col + 8]).unwrap()
        ));
        let m2 = _mm256_cvtepi16_epi32(loadu_128!(
            <&[i16; 8]>::try_from(&mid[2][col..col + 8]).unwrap()
        ));
        let m3 = _mm256_cvtepi16_epi32(loadu_128!(
            <&[i16; 8]>::try_from(&mid[3][col..col + 8]).unwrap()
        ));
        let m4 = _mm256_cvtepi16_epi32(loadu_128!(
            <&[i16; 8]>::try_from(&mid[4][col..col + 8]).unwrap()
        ));
        let m5 = _mm256_cvtepi16_epi32(loadu_128!(
            <&[i16; 8]>::try_from(&mid[5][col..col + 8]).unwrap()
        ));
        let m6 = _mm256_cvtepi16_epi32(loadu_128!(
            <&[i16; 8]>::try_from(&mid[6][col..col + 8]).unwrap()
        ));
        let m7 = _mm256_cvtepi16_epi32(loadu_128!(
            <&[i16; 8]>::try_from(&mid[7][col..col + 8]).unwrap()
        ));

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
        dst[col..col + 8].copy_from_slice(&result_64.to_ne_bytes());

        col += 8;
    }

    // Scalar fallback
    while col < w {
        let mut sum = 0i32;
        for i in 0..8 {
            sum += filter[i] as i32 * mid[i][col] as i32;
        }
        let val = ((sum + ((1 << sh) >> 1)) >> sh).clamp(0, max);
        dst[col] = val as u8;
        col += 1;
    }
}
#[cfg(feature = "asm")]
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
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe { v_filter_8tap_8bpc_avx2_inner(token, dst, mid, w, filter, sh, max) }
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
    Some(&dav1d_mc_subpel_filters[i as usize][m])
}

/// Horizontal 8-tap filter for 8bpc put (H-only case)
/// Outputs directly to u8 with shift and clamp
#[cfg(target_arch = "x86_64")]
#[rite]
fn h_filter_8tap_8bpc_put_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    src: &[u8], // already offset by -3
    w: usize,
    filter: &[i8; 8],
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    // Broadcast filter coefficients for maddubs
    let coeff_01 = _mm256_set1_epi16(((filter[1] as u8 as i16) << 8) | (filter[0] as u8 as i16));
    let coeff_23 = _mm256_set1_epi16(((filter[3] as u8 as i16) << 8) | (filter[2] as u8 as i16));
    let coeff_45 = _mm256_set1_epi16(((filter[5] as u8 as i16) << 8) | (filter[4] as u8 as i16));
    let coeff_67 = _mm256_set1_epi16(((filter[7] as u8 as i16) << 8) | (filter[6] as u8 as i16));

    // For 8bpc H-only put, intermediate_bits=4, rounding = 32 + ((1 << (6-4)) >> 1) = 34
    // This matches dav1d's pw_34 constant and the scalar put_8tap_rust rnd2(6, 34)
    let rnd = _mm256_set1_epi16(34);
    let zero = _mm256_setzero_si256();

    let mut col = 0usize;

    while col + 16 <= w {
        // s offset = col

        let src_0_15 = loadu_128!(<&[u8; 16]>::try_from(&src[col..col + 16]).unwrap());
        let src_1_16 = loadu_128!(<&[u8; 16]>::try_from(&src[col + 1..col + 17]).unwrap());
        let src_2_17 = loadu_128!(<&[u8; 16]>::try_from(&src[col + 2..col + 18]).unwrap());
        let src_3_18 = loadu_128!(<&[u8; 16]>::try_from(&src[col + 3..col + 19]).unwrap());
        let src_4_19 = loadu_128!(<&[u8; 16]>::try_from(&src[col + 4..col + 20]).unwrap());
        let src_5_20 = loadu_128!(<&[u8; 16]>::try_from(&src[col + 5..col + 21]).unwrap());
        let src_6_21 = loadu_128!(<&[u8; 16]>::try_from(&src[col + 6..col + 22]).unwrap());
        let src_7_22 = loadu_128!(<&[u8; 16]>::try_from(&src[col + 7..col + 23]).unwrap());

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

        let ma01 = _mm256_maddubs_epi16(p01, coeff_01);
        let ma23 = _mm256_maddubs_epi16(p23, coeff_23);
        let ma45 = _mm256_maddubs_epi16(p45, coeff_45);
        let ma67 = _mm256_maddubs_epi16(p67, coeff_67);

        let mut sum = _mm256_add_epi16(ma01, ma23);
        sum = _mm256_add_epi16(sum, ma45);
        sum = _mm256_add_epi16(sum, ma67);

        // Add rounding, shift by 6, clamp to [0, 255], pack to u8
        let shift_count = _mm_cvtsi32_si128(6);
        let shifted = _mm256_sra_epi16(_mm256_add_epi16(sum, rnd), shift_count);
        let clamped = _mm256_max_epi16(_mm256_min_epi16(shifted, _mm256_set1_epi16(255)), zero);

        // Pack i16 to u8
        let packed = _mm256_packus_epi16(clamped, clamped);
        let packed = _mm256_permute4x64_epi64(packed, 0b11011000);

        storeu_128!(
            <&mut [u8; 16]>::try_from(&mut dst[col..col + 16]).unwrap(),
            _mm256_castsi256_si128(packed)
        );
        col += 16;
    }

    // Scalar fallback (rnd=34 matches SIMD path above)
    while col < w {
        // s offset = col
        let mut sum = 0i32;
        for i in 0..8 {
            sum += filter[i] as i32 * src[col + i] as i32;
        }
        let val = ((sum + 34) >> 6).clamp(0, 255);
        dst[col] = val as u8;
        col += 1;
    }
}
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn h_filter_8tap_8bpc_put_avx2(
    dst: *mut u8,
    src: *const u8, // already offset by -3
    w: usize,
    filter: &[i8; 8],
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe {
        h_filter_8tap_8bpc_put_avx2_inner(
            token, dst, src, // already offset by -3
            w, filter,
        )
    }
}

/// Vertical 8-tap filter for 8bpc put (V-only case)  
/// Reads directly from u8 source with stride, outputs u8
#[cfg(target_arch = "x86_64")]
#[rite]
fn v_filter_8tap_8bpc_direct_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    src: &[u8], // already positioned at (y-3, 0)
    src_stride: isize,
    w: usize,
    filter: &[i8; 8],
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    let c0 = _mm256_set1_epi32(filter[0] as i32);
    let c1 = _mm256_set1_epi32(filter[1] as i32);
    let c2 = _mm256_set1_epi32(filter[2] as i32);
    let c3 = _mm256_set1_epi32(filter[3] as i32);
    let c4 = _mm256_set1_epi32(filter[4] as i32);
    let c5 = _mm256_set1_epi32(filter[5] as i32);
    let c6 = _mm256_set1_epi32(filter[6] as i32);
    let c7 = _mm256_set1_epi32(filter[7] as i32);

    let rnd = _mm256_set1_epi32(32);
    let zero = _mm256_setzero_si256();
    let max = _mm256_set1_epi32(255);

    let mut col = 0usize;

    while col + 8 <= w {
        // Load 8 u8 from each of 8 rows, zero-extend to i32
        let p0 = _mm256_cvtepu8_epi32(loadi64!(&src[col..col + 8]));
        let p1 = _mm256_cvtepu8_epi32(loadi64!(
            &src[src_stride as usize + col..src_stride as usize + col + 8]
        ));
        let p2 = _mm256_cvtepu8_epi32(loadi64!(
            &src[2 * src_stride as usize + col..2 * src_stride as usize + col + 8]
        ));
        let p3 = _mm256_cvtepu8_epi32(loadi64!(
            &src[3 * src_stride as usize + col..3 * src_stride as usize + col + 8]
        ));
        let p4 = _mm256_cvtepu8_epi32(loadi64!(
            &src[4 * src_stride as usize + col..4 * src_stride as usize + col + 8]
        ));
        let p5 = _mm256_cvtepu8_epi32(loadi64!(
            &src[5 * src_stride as usize + col..5 * src_stride as usize + col + 8]
        ));
        let p6 = _mm256_cvtepu8_epi32(loadi64!(
            &src[6 * src_stride as usize + col..6 * src_stride as usize + col + 8]
        ));
        let p7 = _mm256_cvtepu8_epi32(loadi64!(
            &src[7 * src_stride as usize + col..7 * src_stride as usize + col + 8]
        ));

        // Multiply and accumulate
        let mut sum = _mm256_mullo_epi32(p0, c0);
        sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(p1, c1));
        sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(p2, c2));
        sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(p3, c3));
        sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(p4, c4));
        sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(p5, c5));
        sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(p6, c6));
        sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(p7, c7));

        // Round, shift, clamp
        let shift_count = _mm_cvtsi32_si128(6);
        let shifted = _mm256_sra_epi32(_mm256_add_epi32(sum, rnd), shift_count);
        let clamped = _mm256_min_epi32(_mm256_max_epi32(shifted, zero), max);

        // Pack to u8
        let packed16 = _mm256_packs_epi32(clamped, clamped);
        let packed16 = _mm256_permute4x64_epi64(packed16, 0b11011000);
        let packed8 = _mm256_packus_epi16(packed16, packed16);

        let result_64 = _mm256_extract_epi64(packed8, 0);
        dst[col..col + 8].copy_from_slice(&result_64.to_ne_bytes());

        col += 8;
    }

    // Scalar fallback
    while col < w {
        let mut sum = 0i32;
        for i in 0..8 {
            let px = src[(i as isize * src_stride) as usize + col] as i32;
            sum += filter[i] as i32 * px;
        }
        let val = ((sum + 32) >> 6).clamp(0, 255);
        dst[col] = val as u8;
        col += 1;
    }
}
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn v_filter_8tap_8bpc_direct_avx2(
    dst: *mut u8,
    src: *const u8, // already positioned at (y-3, 0)
    src_stride: isize,
    w: usize,
    filter: &[i8; 8],
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe {
        v_filter_8tap_8bpc_direct_avx2_inner(
            token, dst, src, // already positioned at (y-3, 0)
            src_stride, w, filter,
        )
    }
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
#[rite]
fn put_8tap_8bpc_avx2_impl_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: isize,
    src: &[u8],
    src_base: usize,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    h_filter_type: Rav1dFilterMode,
    v_filter_type: Rav1dFilterMode,
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    let w = w as usize;
    let h = h as usize;
    let mx = mx as usize;
    let my = my as usize;
    let sb = src_base as isize;

    // For 8bpc: intermediate_bits = 4
    let intermediate_bits = 4u8;

    let fh = get_filter_coeff(mx, w, h_filter_type);
    let fv = get_filter_coeff(my, h, v_filter_type);

    match (fh, fv) {
        (Some(fh), Some(fv)) => {
            // Case 1: Both H and V filtering
            // First pass: horizontal filter to intermediate buffer
            let tmp_h = h + 7;
            let mut mid = take_mid_i16_135();

            for y in 0..tmp_h {
                let src_row_base = (sb + (y as isize - 3) * src_stride) as usize;
                h_filter_8tap_8bpc_avx2_inner(
                    _token,
                    &mut mid[y],
                    &src[src_row_base - 3..], // Offset by -3 for tap 0
                    w,
                    fh,
                    6 - intermediate_bits,
                );
            }

            // Second pass: vertical filter to output
            for y in 0..h {
                let dst_row = &mut dst[(y as isize * dst_stride) as usize..];
                v_filter_8tap_8bpc_avx2_inner(
                    _token,
                    dst_row,
                    &mid[y..],
                    w,
                    fv,
                    6 + intermediate_bits,
                    255,
                );
            }
            put_mid_i16_135(mid);
        }
        (Some(fh), None) => {
            // Case 2: H-only filtering (full SIMD)
            for y in 0..h {
                let src_row_base = (sb + y as isize * src_stride) as usize;
                let dst_row = &mut dst[(y as isize * dst_stride) as usize..];
                h_filter_8tap_8bpc_put_avx2_inner(_token, dst_row, &src[src_row_base - 3..], w, fh);
            }
        }
        (None, Some(fv)) => {
            // Case 3: V-only filtering (full SIMD)
            for y in 0..h {
                let src_row_base = (sb + (y as isize - 3) * src_stride) as usize;
                let src_row = &src[src_row_base..];
                let dst_row = &mut dst[(y as isize * dst_stride) as usize..];
                v_filter_8tap_8bpc_direct_avx2_inner(_token, dst_row, src_row, src_stride, w, fv);
            }
        }
        (None, None) => {
            // Case 4: Simple copy
            for y in 0..h {
                let src_row_base = (sb + y as isize * src_stride) as usize;
                let src_row = &src[src_row_base..];
                let dst_row = &mut dst[(y as isize * dst_stride) as usize..];
                dst_row[..w].copy_from_slice(&src_row[..w]);
            }
        }
    }
}
#[cfg(feature = "asm")]
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
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe {
        put_8tap_8bpc_avx2_impl_inner(
            token,
            dst_ptr,
            dst_stride,
            src_ptr,
            0,
            src_stride,
            w,
            h,
            mx,
            my,
            h_filter_type,
            v_filter_type,
        )
    }
}

/// Generic put_8tap function wrapper with Filter2d const generic
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
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
    _dst: *const FFISafe<PicOffset>,
    _src: *const FFISafe<PicOffset>,
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

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn put_8tap_regular_8bpc_avx2_inner(
    _token: Desktop64,
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
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

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
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
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe {
        put_8tap_regular_8bpc_avx2_inner(
            token,
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

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn put_8tap_regular_smooth_8bpc_avx2_inner(
    _token: Desktop64,
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
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

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
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
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe {
        put_8tap_regular_smooth_8bpc_avx2_inner(
            token,
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

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn put_8tap_regular_sharp_8bpc_avx2_inner(
    _token: Desktop64,
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
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

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
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
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe {
        put_8tap_regular_sharp_8bpc_avx2_inner(
            token,
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

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn put_8tap_smooth_regular_8bpc_avx2_inner(
    _token: Desktop64,
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
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

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
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
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe {
        put_8tap_smooth_regular_8bpc_avx2_inner(
            token,
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

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn put_8tap_smooth_8bpc_avx2_inner(
    _token: Desktop64,
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
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

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
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
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe {
        put_8tap_smooth_8bpc_avx2_inner(
            token,
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

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn put_8tap_smooth_sharp_8bpc_avx2_inner(
    _token: Desktop64,
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
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

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
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
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe {
        put_8tap_smooth_sharp_8bpc_avx2_inner(
            token,
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

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn put_8tap_sharp_regular_8bpc_avx2_inner(
    _token: Desktop64,
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
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

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
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
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe {
        put_8tap_sharp_regular_8bpc_avx2_inner(
            token,
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

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn put_8tap_sharp_smooth_8bpc_avx2_inner(
    _token: Desktop64,
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
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

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
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
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe {
        put_8tap_sharp_smooth_8bpc_avx2_inner(
            token,
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

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn put_8tap_sharp_8bpc_avx2_inner(
    _token: Desktop64,
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
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

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
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
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe {
        put_8tap_sharp_8bpc_avx2_inner(
            token,
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
#[rite]
fn prep_8tap_8bpc_avx2_impl_inner(
    _token: Desktop64,
    tmp: &mut [i16],
    src: &[u8],
    src_base: usize,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    h_filter_type: Rav1dFilterMode,
    v_filter_type: Rav1dFilterMode,
) {
    let mut tmp = tmp.flex_mut();
    let src = src.flex();
    let w = w as usize;
    let h = h as usize;
    let mx = mx as usize;
    let my = my as usize;
    let sb = src_base as isize;

    // For 8bpc: intermediate_bits = 4
    let intermediate_bits = 4u8;

    let fh = get_filter_coeff(mx, w, h_filter_type);
    let fv = get_filter_coeff(my, h, v_filter_type);

    match (fh, fv) {
        (Some(fh), Some(fv)) => {
            // Case 1: Both H and V filtering
            let tmp_h = h + 7;
            let mut mid = take_mid_i16_135();

            // Horizontal pass
            for y in 0..tmp_h {
                let src_row_base = (sb + (y as isize - 3) * src_stride) as usize;
                h_filter_8tap_8bpc_avx2_inner(
                    _token,
                    &mut mid[y],
                    &src[src_row_base - 3..],
                    w,
                    fh,
                    6 - intermediate_bits,
                );
            }

            // Vertical pass to intermediate output
            // Scalar uses .rnd(6) for prep V-pass (NOT 6+ib like put)
            // Prep keeps extra precision for compound prediction consumers
            for y in 0..h {
                let out_row = y * w;
                v_filter_8tap_to_i16_avx2_inner(_token, &mid[y..], &mut tmp[out_row..], w, fv, 6);
            }
            put_mid_i16_135(mid);
        }
        (Some(fh), None) => {
            // Case 2: H-only filtering
            // Shift by (6 - intermediate_bits) to match scalar .rnd(6 - intermediate_bits)
            // Same shift as H+V case's H pass
            for y in 0..h {
                let src_row_base = (sb + y as isize * src_stride) as usize;
                let out_row = y * w;
                h_filter_8tap_8bpc_avx2_inner(
                    _token,
                    &mut tmp[out_row..],
                    &src[src_row_base - 3..],
                    w,
                    fh,
                    6 - intermediate_bits,
                );
            }
        }
        (None, Some(fv)) => {
            // Case 3: V-only filtering
            for y in 0..h {
                let out_row = y * w;

                // Build intermediate buffer from 8 source rows
                let mut mid = [[0i16; MID_STRIDE]; 8];
                for i in 0..8 {
                    let src_row =
                        &src[(sb + (y as isize + i as isize - 3) * src_stride) as usize..];
                    for x in 0..w {
                        mid[i][x] = (src_row[x] as i16) << intermediate_bits;
                    }
                }

                v_filter_8tap_to_i16_avx2_inner(_token, &mid, &mut tmp[out_row..], w, fv, 6);
            }
        }
        (None, None) => {
            // Case 4: Simple copy with intermediate scaling
            for y in 0..h {
                let src_row_base = (sb + y as isize * src_stride) as usize;
                let src_row = &src[src_row_base..];
                let out_row = y * w;
                for x in 0..w {
                    tmp[out_row + x] = (src_row[x] as i16) << intermediate_bits;
                }
            }
        }
    }
}
#[cfg(feature = "asm")]
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
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe {
        prep_8tap_8bpc_avx2_impl_inner(
            token,
            tmp,
            src_ptr,
            0,
            src_stride,
            w,
            h,
            mx,
            my,
            h_filter_type,
            v_filter_type,
        )
    }
}

/// Vertical 8-tap filter to i16 output (for prep functions)
#[cfg(target_arch = "x86_64")]
#[rite]
fn v_filter_8tap_to_i16_avx2_inner(
    _token: Desktop64,
    mid: &[[i16; MID_STRIDE]],
    dst: &mut [i16],
    w: usize,
    filter: &[i8; 8],
    sh: u8,
) {
    let mut dst = dst.flex_mut();
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
        let m0 = _mm256_cvtepi16_epi32(loadu_128!(
            <&[i16; 8]>::try_from(&mid[0][col..col + 8]).unwrap()
        ));
        let m1 = _mm256_cvtepi16_epi32(loadu_128!(
            <&[i16; 8]>::try_from(&mid[1][col..col + 8]).unwrap()
        ));
        let m2 = _mm256_cvtepi16_epi32(loadu_128!(
            <&[i16; 8]>::try_from(&mid[2][col..col + 8]).unwrap()
        ));
        let m3 = _mm256_cvtepi16_epi32(loadu_128!(
            <&[i16; 8]>::try_from(&mid[3][col..col + 8]).unwrap()
        ));
        let m4 = _mm256_cvtepi16_epi32(loadu_128!(
            <&[i16; 8]>::try_from(&mid[4][col..col + 8]).unwrap()
        ));
        let m5 = _mm256_cvtepi16_epi32(loadu_128!(
            <&[i16; 8]>::try_from(&mid[5][col..col + 8]).unwrap()
        ));
        let m6 = _mm256_cvtepi16_epi32(loadu_128!(
            <&[i16; 8]>::try_from(&mid[6][col..col + 8]).unwrap()
        ));
        let m7 = _mm256_cvtepi16_epi32(loadu_128!(
            <&[i16; 8]>::try_from(&mid[7][col..col + 8]).unwrap()
        ));

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
        storeu_128!(
            <&mut [i16; 8]>::try_from(&mut dst[col..col + 8]).unwrap(),
            _mm256_castsi256_si128(packed)
        );

        col += 8;
    }

    while col < w {
        let mut sum = 0i32;
        for i in 0..8 {
            sum += filter[i] as i32 * mid[i][col] as i32;
        }
        dst[col] = ((sum + ((1 << sh) >> 1)) >> sh) as i16;
        col += 1;
    }
}
#[cfg(feature = "asm")]
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
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe { v_filter_8tap_to_i16_avx2_inner(token, mid, dst, w, filter, sh) }
}

/// Generic prep_8tap function wrapper with Filter2d const generic
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
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
    _src: *const FFISafe<PicOffset>,
) {
    let filter = Filter2d::from_repr(FILTER).unwrap();
    let (h_filter, v_filter) = filter.hv();

    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
        prep_8tap_8bpc_avx2_impl(tmp, src_ptr, src_stride, w, h, mx, my, h_filter, v_filter);
    }
}

// Specific filter type wrappers for dispatch table

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn prep_8tap_regular_8bpc_avx2_inner(
    _token: Desktop64,
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
        prep_8tap_8bpc_avx2::<{ Filter2d::Regular8Tap as usize }>(
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
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
    src: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe {
        prep_8tap_regular_8bpc_avx2_inner(
            token,
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn prep_8tap_regular_smooth_8bpc_avx2_inner(
    _token: Desktop64,
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
        prep_8tap_8bpc_avx2::<{ Filter2d::RegularSmooth8Tap as usize }>(
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
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
    src: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe {
        prep_8tap_regular_smooth_8bpc_avx2_inner(
            token,
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn prep_8tap_regular_sharp_8bpc_avx2_inner(
    _token: Desktop64,
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
        prep_8tap_8bpc_avx2::<{ Filter2d::RegularSharp8Tap as usize }>(
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
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
    src: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe {
        prep_8tap_regular_sharp_8bpc_avx2_inner(
            token,
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn prep_8tap_smooth_regular_8bpc_avx2_inner(
    _token: Desktop64,
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
        prep_8tap_8bpc_avx2::<{ Filter2d::SmoothRegular8Tap as usize }>(
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
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
    src: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe {
        prep_8tap_smooth_regular_8bpc_avx2_inner(
            token,
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn prep_8tap_smooth_8bpc_avx2_inner(
    _token: Desktop64,
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
        prep_8tap_8bpc_avx2::<{ Filter2d::Smooth8Tap as usize }>(
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
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
    src: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe {
        prep_8tap_smooth_8bpc_avx2_inner(
            token,
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn prep_8tap_smooth_sharp_8bpc_avx2_inner(
    _token: Desktop64,
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
        prep_8tap_8bpc_avx2::<{ Filter2d::SmoothSharp8Tap as usize }>(
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
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
    src: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe {
        prep_8tap_smooth_sharp_8bpc_avx2_inner(
            token,
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn prep_8tap_sharp_regular_8bpc_avx2_inner(
    _token: Desktop64,
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
        prep_8tap_8bpc_avx2::<{ Filter2d::SharpRegular8Tap as usize }>(
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
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
    src: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe {
        prep_8tap_sharp_regular_8bpc_avx2_inner(
            token,
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn prep_8tap_sharp_smooth_8bpc_avx2_inner(
    _token: Desktop64,
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
        prep_8tap_8bpc_avx2::<{ Filter2d::SharpSmooth8Tap as usize }>(
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
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
    src: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe {
        prep_8tap_sharp_smooth_8bpc_avx2_inner(
            token,
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn prep_8tap_sharp_8bpc_avx2_inner(
    _token: Desktop64,
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
        prep_8tap_8bpc_avx2::<{ Filter2d::Sharp8Tap as usize }>(
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
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
    src: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe {
        prep_8tap_sharp_8bpc_avx2_inner(
            token,
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

// =============================================================================
// 16BPC 8-TAP FILTERS
// =============================================================================

/// Horizontal 8-tap filter for 16bpc using AVX2
/// Processes 8 output pixels at a time
/// Output is 32-bit to preserve precision for vertical pass
#[cfg(target_arch = "x86_64")]
#[rite]
fn h_filter_8tap_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [i32],
    src: &[u16],
    w: usize,
    filter: &[i8; 8],
    sh: i32,
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    // Convert filter coefficients from i8 to i16 for pmaddwd
    let coeff0 =
        _mm256_set1_epi32(((filter[1] as i16 as i32) << 16) | (filter[0] as i16 as u16 as i32));
    let coeff2 =
        _mm256_set1_epi32(((filter[3] as i16 as i32) << 16) | (filter[2] as i16 as u16 as i32));
    let coeff4 =
        _mm256_set1_epi32(((filter[5] as i16 as i32) << 16) | (filter[4] as i16 as u16 as i32));
    let coeff6 =
        _mm256_set1_epi32(((filter[7] as i16 as i32) << 16) | (filter[6] as i16 as u16 as i32));

    let rnd = _mm256_set1_epi32((1 << sh) >> 1);
    let shift_count = _mm_cvtsi32_si128(sh);

    let mut col = 0usize;

    // Process 8 output pixels at a time
    // Source pointer is already offset by -3 (pointing to tap 0), matching 8bpc convention
    //
    // LANE FIX: _mm256_unpacklo_epi16 works within 128-bit lanes. Loading 16
    // contiguous u16 into a 256-bit register puts elements 0-7 in lane 0 and
    // 8-15 in lane 1. unpacklo then interleaves elements 0-3 from lane 0 and
    // 8-11 from lane 1, giving outputs {0,1,2,3,8,9,10,11} instead of {0..7}.
    // Fix: construct each source register with lane 0 = [col+X..col+X+8] and
    // lane 1 = [col+X+4..col+X+12], so unpacklo gives correct outputs 0-3
    // (lane 0) and 4-7 (lane 1).
    while col + 8 <= w {
        // Load 128-bit chunks: a[k] = src[col+k..col+k+8]
        let a0 = loadu_128!(<&[u16; 8]>::try_from(&src[col..col + 8]).unwrap());
        let a1 = loadu_128!(<&[u16; 8]>::try_from(&src[col + 1..col + 9]).unwrap());
        let a2 = loadu_128!(<&[u16; 8]>::try_from(&src[col + 2..col + 10]).unwrap());
        let a3 = loadu_128!(<&[u16; 8]>::try_from(&src[col + 3..col + 11]).unwrap());
        let a4 = loadu_128!(<&[u16; 8]>::try_from(&src[col + 4..col + 12]).unwrap());
        let a5 = loadu_128!(<&[u16; 8]>::try_from(&src[col + 5..col + 13]).unwrap());
        let a6 = loadu_128!(<&[u16; 8]>::try_from(&src[col + 6..col + 14]).unwrap());
        let a7 = loadu_128!(<&[u16; 8]>::try_from(&src[col + 7..col + 15]).unwrap());
        let a8 = loadu_128!(<&[u16; 8]>::try_from(&src[col + 8..col + 16]).unwrap());
        let a9 = loadu_128!(<&[u16; 8]>::try_from(&src[col + 9..col + 17]).unwrap());
        let a10 = loadu_128!(<&[u16; 8]>::try_from(&src[col + 10..col + 18]).unwrap());
        let a11 = loadu_128!(<&[u16; 8]>::try_from(&src[col + 11..col + 19]).unwrap());

        // Build 256-bit source regs: lane0 for outputs 0-3, lane1 for outputs 4-7
        let s0 = _mm256_inserti128_si256(_mm256_castsi128_si256(a0), a4, 1);
        let s1 = _mm256_inserti128_si256(_mm256_castsi128_si256(a1), a5, 1);
        let s2 = _mm256_inserti128_si256(_mm256_castsi128_si256(a2), a6, 1);
        let s3 = _mm256_inserti128_si256(_mm256_castsi128_si256(a3), a7, 1);
        let s4 = _mm256_inserti128_si256(_mm256_castsi128_si256(a4), a8, 1);
        let s5 = _mm256_inserti128_si256(_mm256_castsi128_si256(a5), a9, 1);
        let s6 = _mm256_inserti128_si256(_mm256_castsi128_si256(a6), a10, 1);
        let s7 = _mm256_inserti128_si256(_mm256_castsi128_si256(a7), a11, 1);

        // Interleave consecutive pixels for madd_epi16
        // madd_epi16 does: (a[2i] * b[2i]) + (a[2i+1] * b[2i+1]) -> 32-bit
        let p01 = _mm256_unpacklo_epi16(s0, s1);
        let p23 = _mm256_unpacklo_epi16(s2, s3);
        let p45 = _mm256_unpacklo_epi16(s4, s5);
        let p67 = _mm256_unpacklo_epi16(s6, s7);

        // Multiply-add pairs: coeff0=[f0,f1], coeff2=[f2,f3], etc.
        let ma01 = _mm256_madd_epi16(p01, coeff0);
        let ma23 = _mm256_madd_epi16(p23, coeff2);
        let ma45 = _mm256_madd_epi16(p45, coeff4);
        let ma67 = _mm256_madd_epi16(p67, coeff6);

        // Sum all contributions (32-bit)
        let sum = _mm256_add_epi32(_mm256_add_epi32(ma01, ma23), _mm256_add_epi32(ma45, ma67));

        // Add rounding and shift
        let result = _mm256_sra_epi32(_mm256_add_epi32(sum, rnd), shift_count);

        // Store 8 32-bit results
        storeu_256!(
            <&mut [i32; 8]>::try_from(&mut dst[col..col + 8]).unwrap(),
            result
        );

        col += 8;
    }

    // Scalar fallback for remaining pixels
    while col < w {
        let mut sum = 0i32;
        for i in 0..8 {
            sum += filter[i] as i32 * src[col + i] as i32;
        }
        let r = (1 << sh) >> 1;
        dst[col] = (sum + r) >> sh;
        col += 1;
    }
}
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn h_filter_8tap_16bpc_avx2(
    dst: *mut i32,
    src: *const u16,
    w: usize,
    filter: &[i8; 8],
    sh: i32,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe { h_filter_8tap_16bpc_avx2_inner(token, dst, src, w, filter, sh) }
}

/// Vertical 8-tap filter for 16bpc using AVX2
/// Input is 32-bit intermediate, output is 16-bit clamped to [0, max]
#[cfg(target_arch = "x86_64")]
#[rite]
fn v_filter_8tap_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    mid: &[[i32; MID_STRIDE]],
    w: usize,
    y: usize,
    filter: &[i8; 8],
    sh: i32,
    max: i32,
) {
    let mut dst = dst.flex_mut();
    // Convert filter coefficients from i8 to i32 for multiplication
    let coeff: [i32; 8] = [
        filter[0] as i32,
        filter[1] as i32,
        filter[2] as i32,
        filter[3] as i32,
        filter[4] as i32,
        filter[5] as i32,
        filter[6] as i32,
        filter[7] as i32,
    ];

    let rnd = _mm256_set1_epi32((1 << sh) >> 1);
    let shift_count = _mm_cvtsi32_si128(sh);
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi32(max);

    let mut col = 0usize;

    // Process 8 pixels at a time
    while col + 8 <= w {
        // Load 8 intermediate values from each of the 8 tap rows
        let r0 = loadu_256!(<&[i32; 8]>::try_from(&mid[y + 0][col..col + 8]).unwrap());
        let r1 = loadu_256!(<&[i32; 8]>::try_from(&mid[y + 1][col..col + 8]).unwrap());
        let r2 = loadu_256!(<&[i32; 8]>::try_from(&mid[y + 2][col..col + 8]).unwrap());
        let r3 = loadu_256!(<&[i32; 8]>::try_from(&mid[y + 3][col..col + 8]).unwrap());
        let r4 = loadu_256!(<&[i32; 8]>::try_from(&mid[y + 4][col..col + 8]).unwrap());
        let r5 = loadu_256!(<&[i32; 8]>::try_from(&mid[y + 5][col..col + 8]).unwrap());
        let r6 = loadu_256!(<&[i32; 8]>::try_from(&mid[y + 6][col..col + 8]).unwrap());
        let r7 = loadu_256!(<&[i32; 8]>::try_from(&mid[y + 7][col..col + 8]).unwrap());

        // Multiply each row by its coefficient and accumulate
        let c0 = _mm256_set1_epi32(coeff[0]);
        let c1 = _mm256_set1_epi32(coeff[1]);
        let c2 = _mm256_set1_epi32(coeff[2]);
        let c3 = _mm256_set1_epi32(coeff[3]);
        let c4 = _mm256_set1_epi32(coeff[4]);
        let c5 = _mm256_set1_epi32(coeff[5]);
        let c6 = _mm256_set1_epi32(coeff[6]);
        let c7 = _mm256_set1_epi32(coeff[7]);

        let m0 = _mm256_mullo_epi32(r0, c0);
        let m1 = _mm256_mullo_epi32(r1, c1);
        let m2 = _mm256_mullo_epi32(r2, c2);
        let m3 = _mm256_mullo_epi32(r3, c3);
        let m4 = _mm256_mullo_epi32(r4, c4);
        let m5 = _mm256_mullo_epi32(r5, c5);
        let m6 = _mm256_mullo_epi32(r6, c6);
        let m7 = _mm256_mullo_epi32(r7, c7);

        // Sum all
        let sum = _mm256_add_epi32(
            _mm256_add_epi32(_mm256_add_epi32(m0, m1), _mm256_add_epi32(m2, m3)),
            _mm256_add_epi32(_mm256_add_epi32(m4, m5), _mm256_add_epi32(m6, m7)),
        );

        // Add rounding, shift, and clamp
        let shifted = _mm256_sra_epi32(_mm256_add_epi32(sum, rnd), shift_count);
        let clamped_lo = _mm256_max_epi32(shifted, zero);
        let clamped = _mm256_min_epi32(clamped_lo, max_val);

        // Pack from 32-bit to 16-bit
        // _mm256_packus_epi32 packs with unsigned saturation
        // But we need to handle lane crossing...
        // Pack low 4 and high 4 separately, then permute
        let packed = _mm256_packus_epi32(clamped, zero);
        // Permute to fix lane order: [0,1,2,3,x,x,x,x,4,5,6,7,x,x,x,x]
        let result = _mm256_permute4x64_epi64(packed, 0b00_00_10_00); // [0,2,0,0]

        // Store 8 16-bit results
        storeu_128!(
            <&mut [u16; 8]>::try_from(&mut dst[col..col + 8]).unwrap(),
            _mm256_castsi256_si128(result)
        );

        col += 8;
    }

    // Scalar fallback for remaining pixels
    while col < w {
        let mut sum = 0i32;
        for i in 0..8 {
            sum += coeff[i] * mid[y + i][col];
        }
        let r = (1 << sh) >> 1;
        let val = ((sum + r) >> sh).clamp(0, max);
        dst[col] = val as u16;
        col += 1;
    }
}
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn v_filter_8tap_16bpc_avx2(
    dst: *mut u16,
    mid: &[[i32; MID_STRIDE]],
    w: usize,
    y: usize,
    filter: &[i8; 8],
    sh: i32,
    max: i32,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe { v_filter_8tap_16bpc_avx2_inner(token, dst, mid, w, y, filter, sh, max) }
}

/// Vertical 8-tap filter for 16bpc prep (output is i16 with PREP_BIAS subtracted)
#[cfg(target_arch = "x86_64")]
#[rite]
fn v_filter_8tap_16bpc_prep_avx2_inner(
    _token: Desktop64,
    dst: &mut [i16],
    mid: &[[i32; MID_STRIDE]],
    w: usize,
    y: usize,
    filter: &[i8; 8],
    sh: i32,
    prep_bias: i32,
) {
    let mut dst = dst.flex_mut();
    let coeff: [i32; 8] = [
        filter[0] as i32,
        filter[1] as i32,
        filter[2] as i32,
        filter[3] as i32,
        filter[4] as i32,
        filter[5] as i32,
        filter[6] as i32,
        filter[7] as i32,
    ];

    let rnd = _mm256_set1_epi32((1 << sh) >> 1);
    let shift_count = _mm_cvtsi32_si128(sh);
    let bias = _mm256_set1_epi32(prep_bias);

    let mut col = 0usize;

    // Process 8 pixels at a time
    while col + 8 <= w {
        let r0 = loadu_256!(<&[i32; 8]>::try_from(&mid[y + 0][col..col + 8]).unwrap());
        let r1 = loadu_256!(<&[i32; 8]>::try_from(&mid[y + 1][col..col + 8]).unwrap());
        let r2 = loadu_256!(<&[i32; 8]>::try_from(&mid[y + 2][col..col + 8]).unwrap());
        let r3 = loadu_256!(<&[i32; 8]>::try_from(&mid[y + 3][col..col + 8]).unwrap());
        let r4 = loadu_256!(<&[i32; 8]>::try_from(&mid[y + 4][col..col + 8]).unwrap());
        let r5 = loadu_256!(<&[i32; 8]>::try_from(&mid[y + 5][col..col + 8]).unwrap());
        let r6 = loadu_256!(<&[i32; 8]>::try_from(&mid[y + 6][col..col + 8]).unwrap());
        let r7 = loadu_256!(<&[i32; 8]>::try_from(&mid[y + 7][col..col + 8]).unwrap());

        let c0 = _mm256_set1_epi32(coeff[0]);
        let c1 = _mm256_set1_epi32(coeff[1]);
        let c2 = _mm256_set1_epi32(coeff[2]);
        let c3 = _mm256_set1_epi32(coeff[3]);
        let c4 = _mm256_set1_epi32(coeff[4]);
        let c5 = _mm256_set1_epi32(coeff[5]);
        let c6 = _mm256_set1_epi32(coeff[6]);
        let c7 = _mm256_set1_epi32(coeff[7]);

        let m0 = _mm256_mullo_epi32(r0, c0);
        let m1 = _mm256_mullo_epi32(r1, c1);
        let m2 = _mm256_mullo_epi32(r2, c2);
        let m3 = _mm256_mullo_epi32(r3, c3);
        let m4 = _mm256_mullo_epi32(r4, c4);
        let m5 = _mm256_mullo_epi32(r5, c5);
        let m6 = _mm256_mullo_epi32(r6, c6);
        let m7 = _mm256_mullo_epi32(r7, c7);

        let sum = _mm256_add_epi32(
            _mm256_add_epi32(_mm256_add_epi32(m0, m1), _mm256_add_epi32(m2, m3)),
            _mm256_add_epi32(_mm256_add_epi32(m4, m5), _mm256_add_epi32(m6, m7)),
        );

        // Add rounding, shift, subtract bias
        let shifted = _mm256_sra_epi32(_mm256_add_epi32(sum, rnd), shift_count);
        let biased = _mm256_sub_epi32(shifted, bias);

        // Pack from 32-bit to 16-bit (signed)
        let packed = _mm256_packs_epi32(biased, biased);
        // Permute to fix lane order
        let result = _mm256_permute4x64_epi64(packed, 0b00_00_10_00);

        storeu_128!(
            <&mut [i16; 8]>::try_from(&mut dst[col..col + 8]).unwrap(),
            _mm256_castsi256_si128(result)
        );

        col += 8;
    }

    // Scalar fallback
    while col < w {
        let mut sum = 0i32;
        for i in 0..8 {
            sum += coeff[i] * mid[y + i][col];
        }
        let r = (1 << sh) >> 1;
        let val = ((sum + r) >> sh) - prep_bias;
        dst[col] = val as i16;
        col += 1;
    }
}
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn v_filter_8tap_16bpc_prep_avx2(
    dst: *mut i16,
    mid: &[[i32; MID_STRIDE]],
    w: usize,
    y: usize,
    filter: &[i8; 8],
    sh: i32,
    prep_bias: i32,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe { v_filter_8tap_16bpc_prep_avx2_inner(token, dst, mid, w, y, filter, sh, prep_bias) }
}

/// Horizontal 8-tap filter for 16bpc put (H-only case)
/// Input and output are both u16, shift=6, rnd=32, clamp to [0, max]
#[cfg(target_arch = "x86_64")]
#[rite]
fn h_filter_8tap_16bpc_put_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    src: &[u16],
    w: usize,
    filter: &[i8; 8],
    max: i32,
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    // Convert filter coefficients from i8 to i16 for pmaddwd
    let coeff0 =
        _mm256_set1_epi32(((filter[1] as i16 as i32) << 16) | (filter[0] as i16 as u16 as i32));
    let coeff2 =
        _mm256_set1_epi32(((filter[3] as i16 as i32) << 16) | (filter[2] as i16 as u16 as i32));
    let coeff4 =
        _mm256_set1_epi32(((filter[5] as i16 as i32) << 16) | (filter[4] as i16 as u16 as i32));
    let coeff6 =
        _mm256_set1_epi32(((filter[7] as i16 as i32) << 16) | (filter[6] as i16 as u16 as i32));

    // H-only put rounding: 32 + ((1 << (6 - intermediate_bits)) >> 1)
    // 10-bit (max=1023): intermediate_bits=4, rnd=32+2=34
    // 12-bit (max=4095): intermediate_bits=2, rnd=32+8=40
    let intermediate_bits = if (max >> 11) != 0 { 2 } else { 4 };
    let rnd = _mm256_set1_epi32(32 + ((1 << (6 - intermediate_bits)) >> 1));
    let shift_count = _mm_cvtsi32_si128(6);
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi32(max);

    let mut col = 0usize;

    // Source pointer is already offset by -3 (pointing to tap 0), matching 8bpc convention
    // LANE FIX: see h_filter_8tap_16bpc_avx2_inner for explanation
    while col + 8 <= w {
        // Load 128-bit chunks: a[k] = src[col+k..col+k+8]
        let a0 = loadu_128!(<&[u16; 8]>::try_from(&src[col..col + 8]).unwrap());
        let a1 = loadu_128!(<&[u16; 8]>::try_from(&src[col + 1..col + 9]).unwrap());
        let a2 = loadu_128!(<&[u16; 8]>::try_from(&src[col + 2..col + 10]).unwrap());
        let a3 = loadu_128!(<&[u16; 8]>::try_from(&src[col + 3..col + 11]).unwrap());
        let a4 = loadu_128!(<&[u16; 8]>::try_from(&src[col + 4..col + 12]).unwrap());
        let a5 = loadu_128!(<&[u16; 8]>::try_from(&src[col + 5..col + 13]).unwrap());
        let a6 = loadu_128!(<&[u16; 8]>::try_from(&src[col + 6..col + 14]).unwrap());
        let a7 = loadu_128!(<&[u16; 8]>::try_from(&src[col + 7..col + 15]).unwrap());
        let a8 = loadu_128!(<&[u16; 8]>::try_from(&src[col + 8..col + 16]).unwrap());
        let a9 = loadu_128!(<&[u16; 8]>::try_from(&src[col + 9..col + 17]).unwrap());
        let a10 = loadu_128!(<&[u16; 8]>::try_from(&src[col + 10..col + 18]).unwrap());
        let a11 = loadu_128!(<&[u16; 8]>::try_from(&src[col + 11..col + 19]).unwrap());

        // Build 256-bit source regs: lane0 for outputs 0-3, lane1 for outputs 4-7
        let s0 = _mm256_inserti128_si256(_mm256_castsi128_si256(a0), a4, 1);
        let s1 = _mm256_inserti128_si256(_mm256_castsi128_si256(a1), a5, 1);
        let s2 = _mm256_inserti128_si256(_mm256_castsi128_si256(a2), a6, 1);
        let s3 = _mm256_inserti128_si256(_mm256_castsi128_si256(a3), a7, 1);
        let s4 = _mm256_inserti128_si256(_mm256_castsi128_si256(a4), a8, 1);
        let s5 = _mm256_inserti128_si256(_mm256_castsi128_si256(a5), a9, 1);
        let s6 = _mm256_inserti128_si256(_mm256_castsi128_si256(a6), a10, 1);
        let s7 = _mm256_inserti128_si256(_mm256_castsi128_si256(a7), a11, 1);

        // Interleave consecutive pixels for madd_epi16
        let p01 = _mm256_unpacklo_epi16(s0, s1);
        let p23 = _mm256_unpacklo_epi16(s2, s3);
        let p45 = _mm256_unpacklo_epi16(s4, s5);
        let p67 = _mm256_unpacklo_epi16(s6, s7);

        // Multiply-add pairs
        let ma01 = _mm256_madd_epi16(p01, coeff0);
        let ma23 = _mm256_madd_epi16(p23, coeff2);
        let ma45 = _mm256_madd_epi16(p45, coeff4);
        let ma67 = _mm256_madd_epi16(p67, coeff6);

        // Sum all contributions
        let sum = _mm256_add_epi32(_mm256_add_epi32(ma01, ma23), _mm256_add_epi32(ma45, ma67));

        // Add rounding, shift, and clamp
        let shifted = _mm256_sra_epi32(_mm256_add_epi32(sum, rnd), shift_count);
        let clamped_lo = _mm256_max_epi32(shifted, zero);
        let clamped = _mm256_min_epi32(clamped_lo, max_val);

        // Pack from 32-bit to 16-bit
        let packed = _mm256_packus_epi32(clamped, zero);
        let result = _mm256_permute4x64_epi64(packed, 0b00_00_10_00);

        storeu_128!(
            <&mut [u16; 8]>::try_from(&mut dst[col..col + 8]).unwrap(),
            _mm256_castsi256_si128(result)
        );

        col += 8;
    }

    // Scalar fallback for remaining pixels (w < 8 or w not divisible by 8)
    let scalar_rnd = 32 + ((1 << (6 - intermediate_bits)) >> 1);
    while col < w {
        let mut sum = 0i32;
        for i in 0..8 {
            sum += filter[i] as i32 * src[col + i] as i32;
        }
        let val = ((sum + scalar_rnd) >> 6).clamp(0, max);
        dst[col] = val as u16;
        col += 1;
    }
}
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn h_filter_8tap_16bpc_put_avx2(
    dst: *mut u16,
    src: *const u16,
    w: usize,
    filter: &[i8; 8],
    max: i32,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe { h_filter_8tap_16bpc_put_avx2_inner(token, dst, src, w, filter, max) }
}

/// Vertical 8-tap filter for 16bpc put (V-only case)
/// Reads directly from u16 source with stride, outputs u16
#[cfg(target_arch = "x86_64")]
#[rite]
fn v_filter_8tap_16bpc_direct_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    src: &[u16],
    src_stride: isize,
    w: usize,
    filter: &[i8; 8],
    max: i32,
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    let coeff: [i32; 8] = [
        filter[0] as i32,
        filter[1] as i32,
        filter[2] as i32,
        filter[3] as i32,
        filter[4] as i32,
        filter[5] as i32,
        filter[6] as i32,
        filter[7] as i32,
    ];

    let c0 = _mm256_set1_epi32(coeff[0]);
    let c1 = _mm256_set1_epi32(coeff[1]);
    let c2 = _mm256_set1_epi32(coeff[2]);
    let c3 = _mm256_set1_epi32(coeff[3]);
    let c4 = _mm256_set1_epi32(coeff[4]);
    let c5 = _mm256_set1_epi32(coeff[5]);
    let c6 = _mm256_set1_epi32(coeff[6]);
    let c7 = _mm256_set1_epi32(coeff[7]);

    let rnd = _mm256_set1_epi32(32);
    let shift_count = _mm_cvtsi32_si128(6);
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi32(max);

    let mut col = 0usize;

    // Source pointer is already offset by -3 rows (pointing to tap 0), matching 8bpc convention
    let stride_u = src_stride as usize;

    while col + 8 <= w {
        // Load 8 pixels from each of 8 rows (at offsets 0 to 7)
        let p0 = _mm256_cvtepu16_epi32(loadu_128!(
            <&[u16; 8]>::try_from(&src[col..col + 8]).unwrap()
        ));
        let p1 = _mm256_cvtepu16_epi32(loadu_128!(<&[u16; 8]>::try_from(
            &src[stride_u + col..stride_u + col + 8]
        )
        .unwrap()));
        let p2 = _mm256_cvtepu16_epi32(loadu_128!(<&[u16; 8]>::try_from(
            &src[2 * stride_u + col..2 * stride_u + col + 8]
        )
        .unwrap()));
        let p3 = _mm256_cvtepu16_epi32(loadu_128!(<&[u16; 8]>::try_from(
            &src[3 * stride_u + col..3 * stride_u + col + 8]
        )
        .unwrap()));
        let p4 = _mm256_cvtepu16_epi32(loadu_128!(<&[u16; 8]>::try_from(
            &src[4 * stride_u + col..4 * stride_u + col + 8]
        )
        .unwrap()));
        let p5 = _mm256_cvtepu16_epi32(loadu_128!(<&[u16; 8]>::try_from(
            &src[5 * stride_u + col..5 * stride_u + col + 8]
        )
        .unwrap()));
        let p6 = _mm256_cvtepu16_epi32(loadu_128!(<&[u16; 8]>::try_from(
            &src[6 * stride_u + col..6 * stride_u + col + 8]
        )
        .unwrap()));
        let p7 = _mm256_cvtepu16_epi32(loadu_128!(<&[u16; 8]>::try_from(
            &src[7 * stride_u + col..7 * stride_u + col + 8]
        )
        .unwrap()));

        // Multiply and accumulate
        let m0 = _mm256_mullo_epi32(p0, c0);
        let m1 = _mm256_mullo_epi32(p1, c1);
        let m2 = _mm256_mullo_epi32(p2, c2);
        let m3 = _mm256_mullo_epi32(p3, c3);
        let m4 = _mm256_mullo_epi32(p4, c4);
        let m5 = _mm256_mullo_epi32(p5, c5);
        let m6 = _mm256_mullo_epi32(p6, c6);
        let m7 = _mm256_mullo_epi32(p7, c7);

        let sum = _mm256_add_epi32(
            _mm256_add_epi32(_mm256_add_epi32(m0, m1), _mm256_add_epi32(m2, m3)),
            _mm256_add_epi32(_mm256_add_epi32(m4, m5), _mm256_add_epi32(m6, m7)),
        );

        // Add rounding, shift, and clamp
        let shifted = _mm256_sra_epi32(_mm256_add_epi32(sum, rnd), shift_count);
        let clamped_lo = _mm256_max_epi32(shifted, zero);
        let clamped = _mm256_min_epi32(clamped_lo, max_val);

        // Pack from 32-bit to 16-bit
        let packed = _mm256_packus_epi32(clamped, zero);
        let result = _mm256_permute4x64_epi64(packed, 0b00_00_10_00);

        storeu_128!(
            <&mut [u16; 8]>::try_from(&mut dst[col..col + 8]).unwrap(),
            _mm256_castsi256_si128(result)
        );

        col += 8;
    }

    // Scalar fallback
    while col < w {
        let mut sum = 0i32;
        for i in 0..8 {
            let px = src[i * stride_u + col] as i32;
            sum += coeff[i] * px;
        }
        let val = ((sum + 32) >> 6).clamp(0, max);
        dst[col] = val as u16;
        col += 1;
    }
}
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn v_filter_8tap_16bpc_direct_avx2(
    dst: *mut u16,
    src: *const u16,
    src_stride: isize,
    w: usize,
    filter: &[i8; 8],
    max: i32,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe { v_filter_8tap_16bpc_direct_avx2_inner(token, dst, src, src_stride, w, filter, max) }
}

/// Horizontal 8-tap filter for 16bpc prep (H-only case)
/// Input is u16, output is i16 with PREP_BIAS subtracted
#[cfg(target_arch = "x86_64")]
#[rite]
fn h_filter_8tap_16bpc_prep_direct_avx2_inner(
    _token: Desktop64,
    dst: &mut [i16],
    src: &[u16],
    w: usize,
    filter: &[i8; 8],
    sh: i32,
    prep_bias: i32,
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    // Convert filter coefficients from i8 to i16 for pmaddwd
    let coeff0 =
        _mm256_set1_epi32(((filter[1] as i16 as i32) << 16) | (filter[0] as i16 as u16 as i32));
    let coeff2 =
        _mm256_set1_epi32(((filter[3] as i16 as i32) << 16) | (filter[2] as i16 as u16 as i32));
    let coeff4 =
        _mm256_set1_epi32(((filter[5] as i16 as i32) << 16) | (filter[4] as i16 as u16 as i32));
    let coeff6 =
        _mm256_set1_epi32(((filter[7] as i16 as i32) << 16) | (filter[6] as i16 as u16 as i32));

    let rnd = _mm256_set1_epi32((1 << sh) >> 1);
    let shift_count = _mm_cvtsi32_si128(sh);
    let bias = _mm256_set1_epi32(prep_bias);

    let mut col = 0usize;

    // Source pointer is already offset by -3 (pointing to tap 0), matching 8bpc convention
    // LANE FIX: see h_filter_8tap_16bpc_avx2_inner for explanation
    while col + 8 <= w {
        // Load 128-bit chunks: a[k] = src[col+k..col+k+8]
        let a0 = loadu_128!(<&[u16; 8]>::try_from(&src[col..col + 8]).unwrap());
        let a1 = loadu_128!(<&[u16; 8]>::try_from(&src[col + 1..col + 9]).unwrap());
        let a2 = loadu_128!(<&[u16; 8]>::try_from(&src[col + 2..col + 10]).unwrap());
        let a3 = loadu_128!(<&[u16; 8]>::try_from(&src[col + 3..col + 11]).unwrap());
        let a4 = loadu_128!(<&[u16; 8]>::try_from(&src[col + 4..col + 12]).unwrap());
        let a5 = loadu_128!(<&[u16; 8]>::try_from(&src[col + 5..col + 13]).unwrap());
        let a6 = loadu_128!(<&[u16; 8]>::try_from(&src[col + 6..col + 14]).unwrap());
        let a7 = loadu_128!(<&[u16; 8]>::try_from(&src[col + 7..col + 15]).unwrap());
        let a8 = loadu_128!(<&[u16; 8]>::try_from(&src[col + 8..col + 16]).unwrap());
        let a9 = loadu_128!(<&[u16; 8]>::try_from(&src[col + 9..col + 17]).unwrap());
        let a10 = loadu_128!(<&[u16; 8]>::try_from(&src[col + 10..col + 18]).unwrap());
        let a11 = loadu_128!(<&[u16; 8]>::try_from(&src[col + 11..col + 19]).unwrap());

        // Build 256-bit source regs: lane0 for outputs 0-3, lane1 for outputs 4-7
        let s0 = _mm256_inserti128_si256(_mm256_castsi128_si256(a0), a4, 1);
        let s1 = _mm256_inserti128_si256(_mm256_castsi128_si256(a1), a5, 1);
        let s2 = _mm256_inserti128_si256(_mm256_castsi128_si256(a2), a6, 1);
        let s3 = _mm256_inserti128_si256(_mm256_castsi128_si256(a3), a7, 1);
        let s4 = _mm256_inserti128_si256(_mm256_castsi128_si256(a4), a8, 1);
        let s5 = _mm256_inserti128_si256(_mm256_castsi128_si256(a5), a9, 1);
        let s6 = _mm256_inserti128_si256(_mm256_castsi128_si256(a6), a10, 1);
        let s7 = _mm256_inserti128_si256(_mm256_castsi128_si256(a7), a11, 1);

        // Interleave consecutive pixels for madd_epi16
        let p01 = _mm256_unpacklo_epi16(s0, s1);
        let p23 = _mm256_unpacklo_epi16(s2, s3);
        let p45 = _mm256_unpacklo_epi16(s4, s5);
        let p67 = _mm256_unpacklo_epi16(s6, s7);

        // Multiply-add pairs
        let ma01 = _mm256_madd_epi16(p01, coeff0);
        let ma23 = _mm256_madd_epi16(p23, coeff2);
        let ma45 = _mm256_madd_epi16(p45, coeff4);
        let ma67 = _mm256_madd_epi16(p67, coeff6);

        // Sum all contributions
        let sum = _mm256_add_epi32(_mm256_add_epi32(ma01, ma23), _mm256_add_epi32(ma45, ma67));

        // Add rounding, shift, and subtract bias
        let shifted = _mm256_sra_epi32(_mm256_add_epi32(sum, rnd), shift_count);
        let biased = _mm256_sub_epi32(shifted, bias);

        // Pack from 32-bit to 16-bit (signed)
        let packed = _mm256_packs_epi32(biased, biased);
        let result = _mm256_permute4x64_epi64(packed, 0b00_00_10_00);

        storeu_128!(
            <&mut [i16; 8]>::try_from(&mut dst[col..col + 8]).unwrap(),
            _mm256_castsi256_si128(result)
        );

        col += 8;
    }

    // Scalar fallback
    while col < w {
        let mut sum = 0i32;
        for i in 0..8 {
            sum += filter[i] as i32 * src[col + i] as i32;
        }
        let r = (1 << sh) >> 1;
        let val = ((sum + r) >> sh) - prep_bias;
        dst[col] = val as i16;
        col += 1;
    }
}
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn h_filter_8tap_16bpc_prep_direct_avx2(
    dst: *mut i16,
    src: *const u16,
    w: usize,
    filter: &[i8; 8],
    sh: i32,
    prep_bias: i32,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe { h_filter_8tap_16bpc_prep_direct_avx2_inner(token, dst, src, w, filter, sh, prep_bias) }
}

/// Vertical 8-tap filter for 16bpc prep (V-only case)
/// Reads directly from u16 source with stride, outputs i16 with bias
#[cfg(target_arch = "x86_64")]
#[rite]
fn v_filter_8tap_16bpc_prep_direct_avx2_inner(
    _token: Desktop64,
    dst: &mut [i16],
    src: &[u16],
    src_stride: isize,
    w: usize,
    filter: &[i8; 8],
    sh: i32,
    prep_bias: i32,
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    let coeff: [i32; 8] = [
        filter[0] as i32,
        filter[1] as i32,
        filter[2] as i32,
        filter[3] as i32,
        filter[4] as i32,
        filter[5] as i32,
        filter[6] as i32,
        filter[7] as i32,
    ];

    let c0 = _mm256_set1_epi32(coeff[0]);
    let c1 = _mm256_set1_epi32(coeff[1]);
    let c2 = _mm256_set1_epi32(coeff[2]);
    let c3 = _mm256_set1_epi32(coeff[3]);
    let c4 = _mm256_set1_epi32(coeff[4]);
    let c5 = _mm256_set1_epi32(coeff[5]);
    let c6 = _mm256_set1_epi32(coeff[6]);
    let c7 = _mm256_set1_epi32(coeff[7]);

    let rnd = _mm256_set1_epi32((1 << sh) >> 1);
    let shift_count = _mm_cvtsi32_si128(sh);
    let bias = _mm256_set1_epi32(prep_bias);

    let mut col = 0usize;

    // Source pointer is already offset by -3 rows (pointing to tap 0), matching 8bpc convention
    let stride_u = src_stride as usize;

    while col + 8 <= w {
        // Load 8 pixels from each of 8 rows (at offsets 0 to 7)
        let p0 = _mm256_cvtepu16_epi32(loadu_128!(
            <&[u16; 8]>::try_from(&src[col..col + 8]).unwrap()
        ));
        let p1 = _mm256_cvtepu16_epi32(loadu_128!(<&[u16; 8]>::try_from(
            &src[stride_u + col..stride_u + col + 8]
        )
        .unwrap()));
        let p2 = _mm256_cvtepu16_epi32(loadu_128!(<&[u16; 8]>::try_from(
            &src[2 * stride_u + col..2 * stride_u + col + 8]
        )
        .unwrap()));
        let p3 = _mm256_cvtepu16_epi32(loadu_128!(<&[u16; 8]>::try_from(
            &src[3 * stride_u + col..3 * stride_u + col + 8]
        )
        .unwrap()));
        let p4 = _mm256_cvtepu16_epi32(loadu_128!(<&[u16; 8]>::try_from(
            &src[4 * stride_u + col..4 * stride_u + col + 8]
        )
        .unwrap()));
        let p5 = _mm256_cvtepu16_epi32(loadu_128!(<&[u16; 8]>::try_from(
            &src[5 * stride_u + col..5 * stride_u + col + 8]
        )
        .unwrap()));
        let p6 = _mm256_cvtepu16_epi32(loadu_128!(<&[u16; 8]>::try_from(
            &src[6 * stride_u + col..6 * stride_u + col + 8]
        )
        .unwrap()));
        let p7 = _mm256_cvtepu16_epi32(loadu_128!(<&[u16; 8]>::try_from(
            &src[7 * stride_u + col..7 * stride_u + col + 8]
        )
        .unwrap()));

        // Multiply and accumulate
        let m0 = _mm256_mullo_epi32(p0, c0);
        let m1 = _mm256_mullo_epi32(p1, c1);
        let m2 = _mm256_mullo_epi32(p2, c2);
        let m3 = _mm256_mullo_epi32(p3, c3);
        let m4 = _mm256_mullo_epi32(p4, c4);
        let m5 = _mm256_mullo_epi32(p5, c5);
        let m6 = _mm256_mullo_epi32(p6, c6);
        let m7 = _mm256_mullo_epi32(p7, c7);

        let sum = _mm256_add_epi32(
            _mm256_add_epi32(_mm256_add_epi32(m0, m1), _mm256_add_epi32(m2, m3)),
            _mm256_add_epi32(_mm256_add_epi32(m4, m5), _mm256_add_epi32(m6, m7)),
        );

        // Add rounding, shift, and subtract bias
        let shifted = _mm256_sra_epi32(_mm256_add_epi32(sum, rnd), shift_count);
        let biased = _mm256_sub_epi32(shifted, bias);

        // Pack from 32-bit to 16-bit (signed)
        let packed = _mm256_packs_epi32(biased, biased);
        let result = _mm256_permute4x64_epi64(packed, 0b00_00_10_00);

        storeu_128!(
            <&mut [i16; 8]>::try_from(&mut dst[col..col + 8]).unwrap(),
            _mm256_castsi256_si128(result)
        );

        col += 8;
    }

    // Scalar fallback
    while col < w {
        let mut sum = 0i32;
        for i in 0..8 {
            let px = src[i * stride_u + col] as i32;
            sum += coeff[i] * px;
        }
        let r = (1 << sh) >> 1;
        let val = ((sum + r) >> sh) - prep_bias;
        dst[col] = val as i16;
        col += 1;
    }
}
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn v_filter_8tap_16bpc_prep_direct_avx2(
    dst: *mut i16,
    src: *const u16,
    src_stride: isize,
    w: usize,
    filter: &[i8; 8],
    sh: i32,
    prep_bias: i32,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe {
        v_filter_8tap_16bpc_prep_direct_avx2_inner(
            token, dst, src, src_stride, w, filter, sh, prep_bias,
        )
    }
}

/// Generic 8-tap put function for 16bpc
///
/// Similar to 8bpc but handles 16-bit pixels and different intermediate scaling
#[cfg(target_arch = "x86_64")]
#[rite]
fn put_8tap_16bpc_avx2_impl_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: isize,
    src: &[u16],
    src_base: usize,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    h_filter_type: Rav1dFilterMode,
    v_filter_type: Rav1dFilterMode,
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    let w = w as usize;
    let h = h as usize;
    let mx = mx as usize;
    let my = my as usize;
    let dst_stride_elems = dst_stride / 2;
    let src_stride_elems = src_stride / 2;
    let sb = src_base as isize;
    let max = bitdepth_max as i32;

    // intermediate_bits depends on bitdepth: 4 for 10-bit, 2 for 12-bit
    let intermediate_bits = if (bitdepth_max >> 11) != 0 {
        2i32
    } else {
        4i32
    };

    let fh = get_filter_coeff(mx, w, h_filter_type);
    let fv = get_filter_coeff(my, h, v_filter_type);

    // DEBUG: control which paths use SIMD vs scalar
    const USE_SIMD_HV: bool = true;
    const USE_SIMD_H: bool = true;
    const USE_SIMD_V: bool = true;

    match (fh, fv) {
        (Some(fh), Some(fv)) => {
            if USE_SIMD_HV {
                // Case 1: Both H and V filtering - two-pass through intermediate (SIMD)
                let tmp_h = h + 7;
                let mut mid = take_mid_i32_135();
                let h_sh = 6 - intermediate_bits;
                let v_sh = 6 + intermediate_bits;
                for y in 0..tmp_h {
                    let src_off = (sb + (y as isize - 3) * src_stride_elems) as usize;
                    h_filter_8tap_16bpc_avx2_inner(
                        _token,
                        &mut mid[y],
                        &src[src_off - 3..],
                        w,
                        fh,
                        h_sh,
                    );
                }
                for y in 0..h {
                    let dst_row = &mut dst[(y as isize * dst_stride_elems) as usize..];
                    v_filter_8tap_16bpc_avx2_inner(_token, dst_row, &*mid, w, y, fv, v_sh, max);
                }
                put_mid_i32_135(mid);
            } else {
                // Scalar H+V fallback
                let tmp_h = h + 7;
                let mut mid = take_mid_i32_135();
                let h_rnd = (1 << (6 - intermediate_bits)) >> 1;
                let h_sh = 6 - intermediate_bits;
                for y in 0..tmp_h {
                    for x in 0..w {
                        let src_off = (sb + (y as isize - 3) * src_stride_elems) as usize;
                        let mut sum = 0i32;
                        for k in 0..8 {
                            let sx = src_off - 3 + x + k;
                            sum += src[sx] as i32 * fh[k] as i32;
                        }
                        mid[y][x] = (sum + h_rnd) >> h_sh;
                    }
                }
                let v_sh = 6 + intermediate_bits;
                let v_rnd = (1 << v_sh) >> 1;
                for y in 0..h {
                    for x in 0..w {
                        let mut sum = 0i32;
                        for k in 0..8 {
                            sum += mid[y + k][x] * fv[k] as i32;
                        }
                        let val = ((sum + v_rnd) >> v_sh).clamp(0, max);
                        dst[(y as isize * dst_stride_elems) as usize + x] = val as u16;
                    }
                }
                put_mid_i32_135(mid);
            }
        }
        (Some(fh), None) => {
            if USE_SIMD_H {
                // Case 2: H-only filtering (SIMD)
                for y in 0..h {
                    let src_off = (sb + y as isize * src_stride_elems) as usize;
                    let dst_row = &mut dst[(y as isize * dst_stride_elems) as usize..];
                    h_filter_8tap_16bpc_put_avx2_inner(
                        _token,
                        dst_row,
                        &src[src_off - 3..],
                        w,
                        fh,
                        max,
                    );
                }
            } else {
                // Scalar H-only fallback
                let intermediate_rnd = 32 + ((1 << (6 - intermediate_bits)) >> 1);
                for y in 0..h {
                    let src_off = (sb + y as isize * src_stride_elems) as usize;
                    for x in 0..w {
                        let mut sum = 0i32;
                        for k in 0..8 {
                            sum += src[src_off + x + k] as i32 * fh[k] as i32;
                        }
                        let val = ((sum + intermediate_rnd) >> 6).clamp(0, max);
                        dst[(y as isize * dst_stride_elems) as usize + x] = val as u16;
                    }
                }
            }
        }
        (None, Some(fv)) => {
            if USE_SIMD_V {
                // Case 3: V-only filtering (SIMD)
                for y in 0..h {
                    let src_off = (sb + (y as isize - 3) * src_stride_elems) as usize;
                    let dst_row = &mut dst[(y as isize * dst_stride_elems) as usize..];
                    v_filter_8tap_16bpc_direct_avx2_inner(
                        _token,
                        dst_row,
                        &src[src_off..],
                        src_stride_elems,
                        w,
                        fv,
                        max,
                    );
                }
            } else {
                // Scalar V-only fallback
                for y in 0..h {
                    for x in 0..w {
                        let mut sum = 0i32;
                        for k in 0..8 {
                            let src_off =
                                (sb + (y as isize + k as isize - 3) * src_stride_elems) as usize;
                            sum += src[src_off + x] as i32 * fv[k] as i32;
                        }
                        let val = ((sum + 32) >> 6).clamp(0, max);
                        dst[(y as isize * dst_stride_elems) as usize + x] = val as u16;
                    }
                }
            }
        }
        (None, None) => {
            // Case 4: Simple copy
            for y in 0..h {
                let src_row = &src[(sb + y as isize * src_stride_elems) as usize..];
                let dst_row = &mut dst[(y as isize * dst_stride_elems) as usize..];
                dst_row[..w].copy_from_slice(&src_row[..w]);
            }
        }
    }
}
#[cfg(feature = "asm")]
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
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe {
        put_8tap_16bpc_avx2_impl_inner(
            token,
            dst_ptr,
            dst_stride,
            src_ptr,
            0,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            h_filter_type,
            v_filter_type,
        )
    }
}

/// Generic 8-tap prep function for 16bpc
#[cfg(target_arch = "x86_64")]
#[rite]
fn prep_8tap_16bpc_avx2_impl_inner(
    _token: Desktop64,
    tmp: &mut [i16],
    src: &[u16],
    src_base: usize,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    h_filter_type: Rav1dFilterMode,
    v_filter_type: Rav1dFilterMode,
) {
    let mut tmp = tmp.flex_mut();
    let src = src.flex();
    let w = w as usize;
    let h = h as usize;
    let mx = mx as usize;
    let my = my as usize;
    let src_stride_elems = src_stride / 2;
    let sb = src_base as isize;

    // intermediate_bits depends on bitdepth: 4 for 10-bit, 2 for 12-bit
    let intermediate_bits = if (bitdepth_max >> 11) != 0 {
        2i32
    } else {
        4i32
    };
    const PREP_BIAS: i32 = 8192;

    let fh = get_filter_coeff(mx, w, h_filter_type);
    let fv = get_filter_coeff(my, h, v_filter_type);

    match (fh, fv) {
        (Some(fh), Some(fv)) => {
            // Two-pass filtering using SIMD
            let tmp_h = h + 7;
            let mut mid = take_mid_i32_135();
            let h_sh = 6 - intermediate_bits; // = 2 for 16bpc
            let v_sh = 6; // prep uses fixed shift of 6 for vertical pass

            // Horizontal pass using SIMD
            for y in 0..tmp_h {
                let src_off = (sb + (y as isize - 3) * src_stride_elems) as usize;
                h_filter_8tap_16bpc_avx2_inner(
                    _token,
                    &mut mid[y],
                    &src[src_off - 3..],
                    w,
                    fh,
                    h_sh,
                );
            }

            // Vertical pass using SIMD (output to i16 with bias subtraction)
            for y in 0..h {
                let out_row = y * w;
                v_filter_8tap_16bpc_prep_avx2_inner(
                    _token,
                    &mut tmp[out_row..],
                    &*mid,
                    w,
                    y,
                    fv,
                    v_sh,
                    PREP_BIAS,
                );
            }
            put_mid_i32_135(mid);
        }
        (Some(fh), None) => {
            // H-only filtering (SIMD)
            let sh = 6 - intermediate_bits; // = 2 for 16bpc
            for y in 0..h {
                let src_off = (sb + y as isize * src_stride_elems) as usize;
                let out_row = y * w;
                h_filter_8tap_16bpc_prep_direct_avx2_inner(
                    _token,
                    &mut tmp[out_row..],
                    &src[src_off - 3..],
                    w,
                    fh,
                    sh,
                    PREP_BIAS,
                );
            }
        }
        (None, Some(fv)) => {
            // V-only filtering (SIMD)
            let sh = 6 - intermediate_bits; // = 2 for 16bpc
            for y in 0..h {
                let src_off = (sb + (y as isize - 3) * src_stride_elems) as usize;
                let out_row = y * w;
                v_filter_8tap_16bpc_prep_direct_avx2_inner(
                    _token,
                    &mut tmp[out_row..],
                    &src[src_off..],
                    src_stride_elems,
                    w,
                    fv,
                    sh,
                    PREP_BIAS,
                );
            }
        }
        (None, None) => {
            // Simple copy with scaling and bias
            for y in 0..h {
                let src_row = &src[(sb + y as isize * src_stride_elems) as usize..];
                let out_row = y * w;
                for x in 0..w {
                    let px = src_row[x] as i32;
                    let val = (px << intermediate_bits) - PREP_BIAS;
                    tmp[out_row + x] = val as i16;
                }
            }
        }
    }
}
#[cfg(feature = "asm")]
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
    bitdepth_max: i32,
    h_filter_type: Rav1dFilterMode,
    v_filter_type: Rav1dFilterMode,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe {
        prep_8tap_16bpc_avx2_impl_inner(
            token,
            tmp,
            src_ptr,
            0,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            h_filter_type,
            v_filter_type,
        )
    }
}

/// Generic put_8tap function wrapper for 16bpc
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
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
    _dst: *const FFISafe<PicOffset>,
    _src: *const FFISafe<PicOffset>,
) {
    let filter = Filter2d::from_repr(FILTER).unwrap();
    let (h_filter, v_filter) = filter.hv();

    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
        put_8tap_16bpc_avx2_impl(
            dst_ptr,
            dst_stride,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            h_filter,
            v_filter,
        );
    }
}

/// Generic prep_8tap function wrapper for 16bpc
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn prep_8tap_16bpc_avx2<const FILTER: usize>(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    _src: *const FFISafe<PicOffset>,
) {
    let filter = Filter2d::from_repr(FILTER).unwrap();
    let (h_filter, v_filter) = filter.hv();

    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
        prep_8tap_16bpc_avx2_impl(
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            h_filter,
            v_filter,
        );
    }
}

// 16bpc wrapper functions for each filter type

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn put_8tap_regular_16bpc_avx2_inner(
    _token: Desktop64,
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    unsafe {
        put_8tap_16bpc_avx2::<{ Filter2d::Regular8Tap as usize }>(
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

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn put_8tap_regular_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe {
        put_8tap_regular_16bpc_avx2_inner(
            token,
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

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn put_8tap_regular_smooth_16bpc_avx2_inner(
    _token: Desktop64,
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    unsafe {
        put_8tap_16bpc_avx2::<{ Filter2d::RegularSmooth8Tap as usize }>(
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

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn put_8tap_regular_smooth_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe {
        put_8tap_regular_smooth_16bpc_avx2_inner(
            token,
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

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn put_8tap_regular_sharp_16bpc_avx2_inner(
    _token: Desktop64,
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    unsafe {
        put_8tap_16bpc_avx2::<{ Filter2d::RegularSharp8Tap as usize }>(
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

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn put_8tap_regular_sharp_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe {
        put_8tap_regular_sharp_16bpc_avx2_inner(
            token,
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

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn put_8tap_smooth_regular_16bpc_avx2_inner(
    _token: Desktop64,
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    unsafe {
        put_8tap_16bpc_avx2::<{ Filter2d::SmoothRegular8Tap as usize }>(
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

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn put_8tap_smooth_regular_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe {
        put_8tap_smooth_regular_16bpc_avx2_inner(
            token,
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

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn put_8tap_smooth_16bpc_avx2_inner(
    _token: Desktop64,
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    unsafe {
        put_8tap_16bpc_avx2::<{ Filter2d::Smooth8Tap as usize }>(
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

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn put_8tap_smooth_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe {
        put_8tap_smooth_16bpc_avx2_inner(
            token,
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

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn put_8tap_smooth_sharp_16bpc_avx2_inner(
    _token: Desktop64,
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    unsafe {
        put_8tap_16bpc_avx2::<{ Filter2d::SmoothSharp8Tap as usize }>(
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

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn put_8tap_smooth_sharp_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe {
        put_8tap_smooth_sharp_16bpc_avx2_inner(
            token,
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

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn put_8tap_sharp_regular_16bpc_avx2_inner(
    _token: Desktop64,
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    unsafe {
        put_8tap_16bpc_avx2::<{ Filter2d::SharpRegular8Tap as usize }>(
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

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn put_8tap_sharp_regular_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe {
        put_8tap_sharp_regular_16bpc_avx2_inner(
            token,
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

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn put_8tap_sharp_smooth_16bpc_avx2_inner(
    _token: Desktop64,
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    unsafe {
        put_8tap_16bpc_avx2::<{ Filter2d::SharpSmooth8Tap as usize }>(
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

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn put_8tap_sharp_smooth_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe {
        put_8tap_sharp_smooth_16bpc_avx2_inner(
            token,
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

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn put_8tap_sharp_16bpc_avx2_inner(
    _token: Desktop64,
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    unsafe {
        put_8tap_16bpc_avx2::<{ Filter2d::Sharp8Tap as usize }>(
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

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn put_8tap_sharp_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe {
        put_8tap_sharp_16bpc_avx2_inner(
            token,
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

// 16bpc prep wrapper functions

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn prep_8tap_regular_16bpc_avx2_inner(
    _token: Desktop64,
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    unsafe {
        prep_8tap_16bpc_avx2::<{ Filter2d::Regular8Tap as usize }>(
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn prep_8tap_regular_16bpc_avx2(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe {
        prep_8tap_regular_16bpc_avx2_inner(
            token,
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn prep_8tap_regular_smooth_16bpc_avx2_inner(
    _token: Desktop64,
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    unsafe {
        prep_8tap_16bpc_avx2::<{ Filter2d::RegularSmooth8Tap as usize }>(
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn prep_8tap_regular_smooth_16bpc_avx2(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe {
        prep_8tap_regular_smooth_16bpc_avx2_inner(
            token,
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn prep_8tap_regular_sharp_16bpc_avx2_inner(
    _token: Desktop64,
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    unsafe {
        prep_8tap_16bpc_avx2::<{ Filter2d::RegularSharp8Tap as usize }>(
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn prep_8tap_regular_sharp_16bpc_avx2(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe {
        prep_8tap_regular_sharp_16bpc_avx2_inner(
            token,
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn prep_8tap_smooth_regular_16bpc_avx2_inner(
    _token: Desktop64,
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    unsafe {
        prep_8tap_16bpc_avx2::<{ Filter2d::SmoothRegular8Tap as usize }>(
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn prep_8tap_smooth_regular_16bpc_avx2(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe {
        prep_8tap_smooth_regular_16bpc_avx2_inner(
            token,
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn prep_8tap_smooth_16bpc_avx2_inner(
    _token: Desktop64,
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    unsafe {
        prep_8tap_16bpc_avx2::<{ Filter2d::Smooth8Tap as usize }>(
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn prep_8tap_smooth_16bpc_avx2(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe {
        prep_8tap_smooth_16bpc_avx2_inner(
            token,
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn prep_8tap_smooth_sharp_16bpc_avx2_inner(
    _token: Desktop64,
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    unsafe {
        prep_8tap_16bpc_avx2::<{ Filter2d::SmoothSharp8Tap as usize }>(
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn prep_8tap_smooth_sharp_16bpc_avx2(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe {
        prep_8tap_smooth_sharp_16bpc_avx2_inner(
            token,
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn prep_8tap_sharp_regular_16bpc_avx2_inner(
    _token: Desktop64,
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    unsafe {
        prep_8tap_16bpc_avx2::<{ Filter2d::SharpRegular8Tap as usize }>(
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn prep_8tap_sharp_regular_16bpc_avx2(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe {
        prep_8tap_sharp_regular_16bpc_avx2_inner(
            token,
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn prep_8tap_sharp_smooth_16bpc_avx2_inner(
    _token: Desktop64,
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    unsafe {
        prep_8tap_16bpc_avx2::<{ Filter2d::SharpSmooth8Tap as usize }>(
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn prep_8tap_sharp_smooth_16bpc_avx2(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe {
        prep_8tap_sharp_smooth_16bpc_avx2_inner(
            token,
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn prep_8tap_sharp_16bpc_avx2_inner(
    _token: Desktop64,
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    unsafe {
        prep_8tap_16bpc_avx2::<{ Filter2d::Sharp8Tap as usize }>(
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn prep_8tap_sharp_16bpc_avx2(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe {
        prep_8tap_sharp_16bpc_avx2_inner(
            token,
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

// ============================================================================
// BILINEAR FILTER IMPLEMENTATIONS
// ============================================================================
//
// Bilinear filtering uses a simple 2-tap filter:
//   pixel = (16 - mxy) * x0 + mxy * x1
// where mxy is 0-15 (4 bits of fractional precision)
//
// For H+V filtering:
//   1. Horizontal pass: filter to intermediate buffer
//   2. Vertical pass: filter intermediate to output
//
// The intermediate uses extra precision bits to avoid rounding errors.

/// Horizontal bilinear filter for 8bpc using AVX2
/// Processes 32 pixels at a time, outputs to i16 intermediate buffer
#[cfg(target_arch = "x86_64")]
#[rite]
fn h_filter_bilin_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [i16],
    src: &[u8],
    w: usize,
    mx: usize,
    sh: u8,
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    // Bilinear: pixel = (16 - mx) * src[x] + mx * src[x+1]
    // Using maddubs: need pairs of [src[x], src[x+1]] with coeffs [16-mx, mx]
    let mx = mx as i8;
    let coeff0 = (16 - mx) as u8;
    let coeff1 = mx as u8;

    // Create coefficient vector for maddubs: [coeff0, coeff1] repeated
    let coeffs = _mm256_set1_epi16(((coeff1 as i16) << 8) | (coeff0 as i16));

    // Rounding value (handle sh=0 case)
    let rnd = if sh > 0 {
        _mm256_set1_epi16(1 << (sh - 1))
    } else {
        _mm256_setzero_si256()
    };

    // Shift count in register for variable shift
    let sh_reg = _mm_cvtsi32_si128(sh as i32);

    let mut x = 0;
    while x + 32 <= w {
        // Load 33 bytes to get 32 pairs of adjacent pixels
        let src_lo = loadu_256!(<&[u8; 32]>::try_from(&src[x..x + 32]).unwrap());
        let src_hi = loadu_256!(<&[u8; 32]>::try_from(&src[x + 1..x + 33]).unwrap());

        // Interleave for maddubs: need [src[0],src[1]], [src[1],src[2]], ...
        // unpacklo gives us pairs from low halves, unpackhi from high halves
        let pairs_lo = _mm256_unpacklo_epi8(src_lo, src_hi);
        let pairs_hi = _mm256_unpackhi_epi8(src_lo, src_hi);

        // Apply bilinear filter using maddubs
        let result_lo = _mm256_maddubs_epi16(pairs_lo, coeffs);
        let result_hi = _mm256_maddubs_epi16(pairs_hi, coeffs);

        // Add rounding and shift (using variable shift)
        let result_lo = _mm256_sra_epi16(_mm256_add_epi16(result_lo, rnd), sh_reg);
        let result_hi = _mm256_sra_epi16(_mm256_add_epi16(result_hi, rnd), sh_reg);

        // Store results - need to handle the permutation from unpack
        // unpack interleaves within 128-bit lanes, so we need to fix the order
        let lo_128 = _mm256_permute2x128_si256(result_lo, result_hi, 0x20); // lo lanes
        let hi_128 = _mm256_permute2x128_si256(result_lo, result_hi, 0x31); // hi lanes

        storeu_256!(
            <&mut [i16; 16]>::try_from(&mut dst[x..x + 16]).unwrap(),
            lo_128
        );
        storeu_256!(
            <&mut [i16; 16]>::try_from(&mut dst[x + 16..x + 32]).unwrap(),
            hi_128
        );
        x += 32;
    }

    // Scalar fallback for remaining pixels
    while x < w {
        let x0 = src[x] as i32;
        let x1 = src[x + 1] as i32;
        let pixel = (16 - mx as i32) * x0 + mx as i32 * x1;
        let result = if sh > 0 {
            (pixel + (1 << (sh - 1))) >> sh
        } else {
            pixel
        };
        dst[x] = result as i16;
        x += 1;
    }
}
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn h_filter_bilin_8bpc_avx2(dst: *mut i16, src: *const u8, w: usize, mx: usize, sh: u8) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe { h_filter_bilin_8bpc_avx2_inner(token, dst, src, w, mx, sh) }
}

/// Vertical bilinear filter for 8bpc using AVX2
/// Reads from i16 intermediate buffer, outputs to u8
#[cfg(target_arch = "x86_64")]
#[rite]
fn v_filter_bilin_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    mid: &[&[i16]],
    w: usize,
    my: usize,
    sh: u8,
    bd_max: i16,
) {
    let mut dst = dst.flex_mut();
    let my = my as i32;
    let coeff0 = 16 - my;
    let coeff1 = my;

    // Use 32-bit arithmetic to avoid i16 overflow.
    // Intermediate values can be up to 4080 (16*255 for 8bpc), and
    // coeff * mid can reach 16 * 4080 = 65280, exceeding i16 range.
    let c0_32 = _mm256_set1_epi32(coeff0);
    let c1_32 = _mm256_set1_epi32(coeff1);
    let rnd = _mm256_set1_epi32(if sh > 0 { 1 << (sh - 1) } else { 0 });
    let zero = _mm256_setzero_si256();
    let max_32 = _mm256_set1_epi32(bd_max as i32);
    let sh_reg = _mm_cvtsi32_si128(sh as i32);

    let mut x = 0;
    // Process 8 pixels at a time using 32-bit arithmetic
    while x + 8 <= w {
        // Load 8 i16 values from each row and sign-extend to i32
        let row0 =
            _mm256_cvtepi16_epi32(loadu_128!(<&[i16; 8]>::try_from(&mid[0][x..x + 8]).unwrap()));
        let row1 =
            _mm256_cvtepi16_epi32(loadu_128!(<&[i16; 8]>::try_from(&mid[1][x..x + 8]).unwrap()));

        // result = coeff0 * row0 + coeff1 * row1 (32-bit, no overflow)
        let mul0 = _mm256_mullo_epi32(row0, c0_32);
        let mul1 = _mm256_mullo_epi32(row1, c1_32);
        let sum = _mm256_add_epi32(mul0, mul1);

        // Add rounding and shift
        let result = _mm256_sra_epi32(_mm256_add_epi32(sum, rnd), sh_reg);

        // Clamp to [0, bd_max]
        let result = _mm256_max_epi32(result, zero);
        let result = _mm256_min_epi32(result, max_32);

        // Pack 32-bit to 16-bit, then 16-bit to 8-bit
        let packed16 = _mm256_packs_epi32(result, result);
        let packed16 = _mm256_permute4x64_epi64(packed16, 0xD8);
        let packed8 = _mm256_packus_epi16(packed16, packed16);

        // Store 8 bytes
        let result_64 = _mm256_extract_epi64(packed8, 0);
        dst[x..x + 8].copy_from_slice(&result_64.to_ne_bytes());
        x += 8;
    }

    // Scalar fallback
    while x < w {
        let r0 = mid[0][x] as i32;
        let r1 = mid[1][x] as i32;
        let pixel = coeff0 * r0 + coeff1 * r1;
        let result = if sh > 0 {
            ((pixel + (1 << (sh - 1))) >> sh).clamp(0, bd_max as i32)
        } else {
            pixel.clamp(0, bd_max as i32)
        };
        dst[x] = result as u8;
        x += 1;
    }
}
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn v_filter_bilin_8bpc_avx2(
    dst: *mut u8,
    mid: &[&[i16]],
    w: usize,
    my: usize,
    sh: u8,
    bd_max: i16,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe { v_filter_bilin_8bpc_avx2_inner(token, dst, mid, w, my, sh, bd_max) }
}
/// Horizontal bilinear filter for 8bpc put (H-only)
/// Outputs directly to u8 with shift=4 and clamp to [0,255]
#[cfg(target_arch = "x86_64")]
#[rite]
fn h_bilin_8bpc_put_avx2_inner(_token: Desktop64, dst: &mut [u8], src: &[u8], w: usize, mx: usize) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    let mx = mx as i8;
    let coeff0 = (16 - mx) as u8;
    let coeff1 = mx as u8;
    let coeffs = _mm256_set1_epi16(((coeff1 as i16) << 8) | (coeff0 as i16));
    let rnd = _mm256_set1_epi16(8); // (1 << 4) >> 1
    let zero = _mm256_setzero_si256();
    let max = _mm256_set1_epi16(255);

    let mut x = 0;
    while x + 32 <= w {
        let src_lo = loadu_256!(<&[u8; 32]>::try_from(&src[x..x + 32]).unwrap());
        let src_hi = loadu_256!(<&[u8; 32]>::try_from(&src[x + 1..x + 33]).unwrap());

        let pairs_lo = _mm256_unpacklo_epi8(src_lo, src_hi);
        let pairs_hi = _mm256_unpackhi_epi8(src_lo, src_hi);

        let result_lo = _mm256_maddubs_epi16(pairs_lo, coeffs);
        let result_hi = _mm256_maddubs_epi16(pairs_hi, coeffs);

        // Add rounding, shift by 4, clamp
        let sh_reg = _mm_cvtsi32_si128(4);
        let result_lo = _mm256_sra_epi16(_mm256_add_epi16(result_lo, rnd), sh_reg);
        let result_hi = _mm256_sra_epi16(_mm256_add_epi16(result_hi, rnd), sh_reg);

        let result_lo = _mm256_max_epi16(_mm256_min_epi16(result_lo, max), zero);
        let result_hi = _mm256_max_epi16(_mm256_min_epi16(result_hi, max), zero);

        // Pack to u8: packus_epi16 operates within 128-bit lanes, and
        // result_lo/result_hi already have matching lane layout from unpack,
        // so packus(result_lo, result_hi) produces correct sequential order.
        let packed = _mm256_packus_epi16(result_lo, result_hi);

        storeu_256!(
            <&mut [u8; 32]>::try_from(&mut dst[x..x + 32]).unwrap(),
            packed
        );
        x += 32;
    }

    while x < w {
        let x0 = src[x] as i32;
        let x1 = src[x + 1] as i32;
        let pixel = (16 - mx as i32) * x0 + mx as i32 * x1;
        let result = ((pixel + 8) >> 4).clamp(0, 255);
        dst[x] = result as u8;
        x += 1;
    }
}
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn h_bilin_8bpc_put_avx2(dst: *mut u8, src: *const u8, w: usize, mx: usize) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe { h_bilin_8bpc_put_avx2_inner(token, dst, src, w, mx) }
}

/// Vertical bilinear filter for 8bpc put (V-only)
/// Reads directly from u8 source, outputs u8
#[cfg(target_arch = "x86_64")]
#[rite]
fn v_bilin_8bpc_direct_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    src0: &[u8],
    src1: &[u8],
    w: usize,
    my: usize,
) {
    let mut dst = dst.flex_mut();
    let src0 = src0.flex();
    let src1 = src1.flex();
    let my = my as i8;
    let coeff0 = (16 - my) as u8;
    let coeff1 = my as u8;
    let coeffs = _mm256_set1_epi16(((coeff1 as i16) << 8) | (coeff0 as i16));
    let rnd = _mm256_set1_epi16(8);
    let sh_reg = _mm_cvtsi32_si128(4);

    let mut x = 0;
    while x + 32 <= w {
        let s0 = loadu_256!(<&[u8; 32]>::try_from(&src0[x..x + 32]).unwrap());
        let s1 = loadu_256!(<&[u8; 32]>::try_from(&src1[x..x + 32]).unwrap());

        let pairs_lo = _mm256_unpacklo_epi8(s0, s1);
        let pairs_hi = _mm256_unpackhi_epi8(s0, s1);

        let result_lo = _mm256_maddubs_epi16(pairs_lo, coeffs);
        let result_hi = _mm256_maddubs_epi16(pairs_hi, coeffs);

        let result_lo = _mm256_sra_epi16(_mm256_add_epi16(result_lo, rnd), sh_reg);
        let result_hi = _mm256_sra_epi16(_mm256_add_epi16(result_hi, rnd), sh_reg);

        // packus_epi16 operates within 128-bit lanes, so with result_lo
        // holding elements [0..7 | 16..23] and result_hi holding [8..15 | 24..31],
        // packus(result_lo, result_hi) correctly produces [0..31] in order.
        let packed = _mm256_packus_epi16(result_lo, result_hi);

        storeu_256!(
            <&mut [u8; 32]>::try_from(&mut dst[x..x + 32]).unwrap(),
            packed
        );
        x += 32;
    }

    while x < w {
        let x0 = src0[x] as i32;
        let x1 = src1[x] as i32;
        let pixel = (16 - my as i32) * x0 + my as i32 * x1;
        let result = ((pixel + 8) >> 4).clamp(0, 255);
        dst[x] = result as u8;
        x += 1;
    }
}
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn v_bilin_8bpc_direct_avx2(
    dst: *mut u8,
    src0: *const u8,
    src1: *const u8,
    w: usize,
    my: usize,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe { v_bilin_8bpc_direct_avx2_inner(token, dst, src0, src1, w, my) }
}

/// Core bilinear filter implementation for 8bpc
#[cfg(target_arch = "x86_64")]
#[rite]
fn put_bilin_8bpc_avx2_impl_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: isize,
    src: &[u8],
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    let w = w as usize;
    let h = h as usize;
    let mx = mx as usize;
    let my = my as usize;

    // For 8bpc: intermediate_bits = 4
    let intermediate_bits = 4u8;

    match (mx != 0, my != 0) {
        (true, true) => {
            // Case 1: Both H and V filtering
            // First pass: horizontal filter to intermediate buffer
            let tmp_h = h + 1;
            let mut mid = take_mid_i16_130();

            for y in 0..tmp_h {
                let src_row_base = (y as isize * src_stride) as usize;
                let src_row = &src[src_row_base..];
                h_filter_bilin_8bpc_avx2_inner(
                    _token,
                    &mut mid[y],
                    src_row,
                    w,
                    mx,
                    4 - intermediate_bits, // sh = 0 for intermediate
                );
            }

            // Second pass: vertical filter to output
            for y in 0..h {
                let dst_row = &mut dst[(y as isize * dst_stride) as usize..];
                let mid_refs: [&[i16]; 2] = [&mid[y], &mid[y + 1]];
                v_filter_bilin_8bpc_avx2_inner(
                    _token,
                    dst_row,
                    &mid_refs,
                    w,
                    my,
                    4 + intermediate_bits, // sh = 8 for final
                    255,
                );
            }
            put_mid_i16_130(mid);
        }
        (true, false) => {
            // Case 2: H-only filtering (full SIMD)
            for y in 0..h {
                let src_row_base = (y as isize * src_stride) as usize;
                let src_row = &src[src_row_base..];
                let dst_row = &mut dst[(y as isize * dst_stride) as usize..];
                h_bilin_8bpc_put_avx2_inner(_token, dst_row, src_row, w, mx);
            }
        }
        (false, true) => {
            // Case 3: V-only filtering (full SIMD)
            for y in 0..h {
                let src_row0 = &src[(y as isize * src_stride) as usize..];
                let src_row1 = &src[((y + 1) as isize * src_stride) as usize..];
                let dst_row = &mut dst[(y as isize * dst_stride) as usize..];
                v_bilin_8bpc_direct_avx2_inner(_token, dst_row, src_row0, src_row1, w, my);
            }
        }
        (false, false) => {
            // Case 4: Simple copy
            for y in 0..h {
                let src_row_base = (y as isize * src_stride) as usize;
                let src_row = &src[src_row_base..];
                let dst_row = &mut dst[(y as isize * dst_stride) as usize..];
                dst_row[..w].copy_from_slice(&src_row[..w]);
            }
        }
    }
}
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn put_bilin_8bpc_avx2_impl(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe {
        put_bilin_8bpc_avx2_impl_inner(
            token, dst_ptr, dst_stride, src_ptr, src_stride, w, h, mx, my,
        )
    }
}

/// Bilinear put for 8bpc - extern "C" wrapper
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn put_bilin_8bpc_avx2_inner(
    _token: Desktop64,
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
        put_bilin_8bpc_avx2_impl(dst_ptr, dst_stride, src_ptr, src_stride, w, h, mx, my);
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn put_bilin_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    _bitdepth_max: i32,
    _dst: *const FFISafe<PicOffset>,
    _src: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe {
        put_bilin_8bpc_avx2_inner(
            token, dst_ptr, dst_stride, src_ptr, src_stride, w, h, mx, my,
        )
    }
}

/// Core bilinear prep implementation for 8bpc
/// Outputs to i16 intermediate buffer (prep format)
#[cfg(target_arch = "x86_64")]
#[rite]
fn prep_bilin_8bpc_avx2_impl_inner(
    _token: Desktop64,
    tmp: &mut [i16],
    src: &[u8],
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
) {
    let mut tmp = tmp.flex_mut();
    let src = src.flex();
    let w = w as usize;
    let h = h as usize;
    let mx = mx as usize;
    let my = my as usize;

    // For 8bpc: intermediate_bits = 4
    let intermediate_bits = 4u8;

    match (mx != 0, my != 0) {
        (true, true) => {
            // Case 1: Both H and V filtering
            let tmp_h = h + 1;
            let mut mid = take_mid_i16_130();

            for y in 0..tmp_h {
                let src_row_base = (y as isize * src_stride) as usize;
                let src_row = &src[src_row_base..];
                h_filter_bilin_8bpc_avx2_inner(
                    _token,
                    &mut mid[y],
                    src_row,
                    w,
                    mx,
                    4 - intermediate_bits, // sh = 0 for first pass
                );
            }

            // Second pass: vertical filter to output i16 buffer
            // For 8bpc: PREP_BIAS=0, so result = ((pixel + 8) >> 4)
            for y in 0..h {
                let dst_row = y * w;
                for x in 0..w {
                    let r0 = mid[y][x] as i32;
                    let r1 = mid[y + 1][x] as i32;
                    let coeff0 = 16 - my as i32;
                    let coeff1 = my as i32;
                    let pixel = coeff0 * r0 + coeff1 * r1;
                    let result = (pixel + 8) >> 4;
                    tmp[dst_row + x] = result as i16; // PREP_BIAS = 0 for 8bpc
                }
            }
            put_mid_i16_130(mid);
        }
        (true, false) => {
            // Case 2: H-only filtering
            // For 8bpc: intermediate_bits=4, scalar does rnd(4-4)=rnd(0)=no shift
            // PREP_BIAS=0, so output = (16-mx)*src[x] + mx*src[x+1] unshifted
            for y in 0..h {
                let src_row_base = (y as isize * src_stride) as usize;
                let src_row = &src[src_row_base..];
                let dst_row = y * w;

                let mut tmp_buf = [0i16; MID_STRIDE];
                // sh=0: no shift, keep full precision
                h_filter_bilin_8bpc_avx2_inner(
                    _token,
                    &mut tmp_buf,
                    src_row,
                    w,
                    mx,
                    4 - intermediate_bits, // sh = 0 for 8bpc
                );

                // PREP_BIAS = 0 for 8bpc, output directly
                for x in 0..w {
                    tmp[dst_row + x] = tmp_buf[x];
                }
            }
        }
        (false, true) => {
            // Case 3: V-only filtering
            for y in 0..h {
                let dst_row = y * w;

                for x in 0..w {
                    let r0 = src[(y as isize * src_stride + x as isize) as usize] as i32;
                    let r1 = src[((y + 1) as isize * src_stride + x as isize) as usize] as i32;
                    let coeff0 = 16 - my as i32;
                    let coeff1 = my as i32;
                    let pixel = coeff0 * r0 + coeff1 * r1;
                    // V-only prep 8bpc: rnd(4-ib)=rnd(0)=no shift, PREP_BIAS=0
                    // Raw bilinear value [0, 4080] fits in i16
                    tmp[dst_row + x] = pixel as i16;
                }
            }
        }
        (false, false) => {
            // Case 4: Simple copy to prep format
            // For 8bpc: intermediate_bits=4, PREP_BIAS=0
            // Formula: (pixel << intermediate_bits) - PREP_BIAS = pixel << 4
            for y in 0..h {
                let src_row_base = (y as isize * src_stride) as usize;
                let src_row = &src[src_row_base..];
                let dst_row = y * w;
                for x in 0..w {
                    let pixel = src_row[x] as i16;
                    tmp[dst_row + x] = pixel << intermediate_bits;
                }
            }
        }
    }
}
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn prep_bilin_8bpc_avx2_impl(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe { prep_bilin_8bpc_avx2_impl_inner(token, tmp, src_ptr, src_stride, w, h, mx, my) }
}

/// Bilinear prep for 8bpc - extern "C" wrapper
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn prep_bilin_8bpc_avx2_inner(
    _token: Desktop64,
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
        prep_bilin_8bpc_avx2_impl(tmp, src_ptr, src_stride, w, h, mx, my);
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn prep_bilin_8bpc_avx2(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    _bitdepth_max: i32,
    _src: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe { prep_bilin_8bpc_avx2_inner(token, tmp, src_ptr, src_stride, w, h, mx, my) }
}

// ============================================================================
// W_MASK (Weighted Mask) - Compound prediction with per-pixel masking
// ============================================================================
//
// w_mask computes both blended prediction AND outputs the mask for chroma
// The mask is computed from: m = min(38 + (abs_diff + mask_rnd) >> mask_sh, 64)
// Blend: dst = (tmp1 * m + tmp2 * (64 - m) + rnd) >> sh

use crate::src::internal::SEG_MASK_LEN;

/// Core w_mask implementation (fully safe, slice-based)
/// SS_HOR and SS_VER control subsampling: 444=(false,false), 422=(true,false), 420=(true,true)
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn w_mask_8bpc_avx2_safe_impl<const SS_HOR: bool, const SS_VER: bool>(
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask: &mut [u8; SEG_MASK_LEN],
    sign: i32,
) {
    let w = w as usize;
    let h = h as usize;
    let sign = sign as u8;

    // For 8bpc: intermediate_bits = 4, bitdepth = 8, PREP_BIAS = 0
    let intermediate_bits = 4u32;
    let bitdepth = 8u32;
    let sh = intermediate_bits + 6;
    let rnd = (32i32 << intermediate_bits) + 0; // PREP_BIAS = 0 for 8bpc
    let mask_sh = bitdepth + intermediate_bits - 4;
    let mask_rnd = 1u16 << (mask_sh - 5);

    // Mask output dimensions depend on subsampling
    let mask_w = if SS_HOR { w >> 1 } else { w };
    let mut mask_off = 0usize;

    for row_h in 0..h {
        let row_offset = row_h * w;
        let tmp1_row = &tmp1[row_offset..][..w];
        let tmp2_row = &tmp2[row_offset..][..w];
        let dst_row = &mut dst[row_h * dst_stride..][..w];

        let mut x = 0;
        while x < w {
            // Compute mask value m
            let diff = tmp1_row[x].abs_diff(tmp2_row[x]);
            let m = std::cmp::min(38 + ((diff.saturating_add(mask_rnd)) >> mask_sh), 64) as u8;

            // Blend pixels
            let t1 = tmp1_row[x] as i32;
            let t2 = tmp2_row[x] as i32;
            let pixel = (t1 * m as i32 + t2 * (64 - m as i32) + rnd) >> sh;
            dst_row[x] = pixel.clamp(0, 255) as u8;

            if SS_HOR {
                // Process second pixel in pair for horizontal subsampling
                x += 1;
                let diff2 = tmp1_row[x].abs_diff(tmp2_row[x]);
                let n = std::cmp::min(38 + ((diff2.saturating_add(mask_rnd)) >> mask_sh), 64) as u8;

                let t1 = tmp1_row[x] as i32;
                let t2 = tmp2_row[x] as i32;
                let pixel = (t1 * n as i32 + t2 * (64 - n as i32) + rnd) >> sh;
                dst_row[x] = pixel.clamp(0, 255) as u8;

                // Output subsampled mask
                let mask_x = x >> 1;
                if SS_VER && (row_h & 1 != 0) {
                    // Vertical subsampling: average with previous row
                    let prev = mask[mask_off + mask_x];
                    mask[mask_off + mask_x] =
                        (((m as u16 + n as u16 + 2 - sign as u16) + prev as u16) >> 2) as u8;
                } else if SS_VER {
                    // Even row: just store sum (will be averaged on odd row)
                    mask[mask_off + mask_x] = m + n;
                } else {
                    // No vertical subsampling: average of horizontal pair
                    mask[mask_off + mask_x] = ((m as u16 + n as u16 + 1 - sign as u16) >> 1) as u8;
                }
            } else {
                // No horizontal subsampling (444)
                mask[mask_off + x] = m;
            }
            x += 1;
        }

        // Advance mask offset only on appropriate rows
        if !SS_VER || (row_h & 1 != 0) {
            mask_off += mask_w;
        }
    }
}

/// w_mask 444 8bpc (no subsampling)
#[cfg(target_arch = "x86_64")]
#[arcane]
fn w_mask_444_8bpc_avx2_safe(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask: &mut [u8; SEG_MASK_LEN],
    sign: i32,
) {
    let mut dst = dst.flex_mut();
    w_mask_8bpc_avx2_safe_impl::<false, false>(&mut *dst, dst_stride, tmp1, tmp2, w, h, mask, sign);
}

/// w_mask 422 8bpc (horizontal subsampling only)
#[cfg(target_arch = "x86_64")]
#[arcane]
fn w_mask_422_8bpc_avx2_safe(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask: &mut [u8; SEG_MASK_LEN],
    sign: i32,
) {
    let mut dst = dst.flex_mut();
    w_mask_8bpc_avx2_safe_impl::<true, false>(&mut *dst, dst_stride, tmp1, tmp2, w, h, mask, sign);
}

/// w_mask 420 8bpc (horizontal and vertical subsampling)
#[cfg(target_arch = "x86_64")]
#[arcane]
fn w_mask_420_8bpc_avx2_safe(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask: &mut [u8; SEG_MASK_LEN],
    sign: i32,
) {
    let mut dst = dst.flex_mut();
    w_mask_8bpc_avx2_safe_impl::<true, true>(&mut *dst, dst_stride, tmp1, tmp2, w, h, mask, sign);
}

/// w_mask for 4:4:4 (no subsampling)
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn w_mask_444_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask: &mut [u8; SEG_MASK_LEN],
    sign: i32,
    _bitdepth_max: i32,
    _dst: *const FFISafe<PicOffset>,
) {
    let dst = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h as usize * dst_stride as usize)
    };
    w_mask_444_8bpc_avx2_safe(
        Desktop64::forge_token_dangerously(),
        dst,
        dst_stride as usize,
        tmp1,
        tmp2,
        w,
        h,
        mask,
        sign,
    )
}

/// w_mask for 4:2:2 (horizontal subsampling only)
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn w_mask_422_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask: &mut [u8; SEG_MASK_LEN],
    sign: i32,
    _bitdepth_max: i32,
    _dst: *const FFISafe<PicOffset>,
) {
    let dst = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h as usize * dst_stride as usize)
    };
    w_mask_422_8bpc_avx2_safe(
        Desktop64::forge_token_dangerously(),
        dst,
        dst_stride as usize,
        tmp1,
        tmp2,
        w,
        h,
        mask,
        sign,
    )
}

/// w_mask for 4:2:0 (horizontal and vertical subsampling)
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn w_mask_420_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask: &mut [u8; SEG_MASK_LEN],
    sign: i32,
    _bitdepth_max: i32,
    _dst: *const FFISafe<PicOffset>,
) {
    let dst = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h as usize * dst_stride as usize)
    };
    w_mask_420_8bpc_avx2_safe(
        Desktop64::forge_token_dangerously(),
        dst,
        dst_stride as usize,
        tmp1,
        tmp2,
        w,
        h,
        mask,
        sign,
    )
}

// 16bpc w_mask (fully safe, slice-based)
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn w_mask_16bpc_avx2_safe_impl<const SS_HOR: bool, const SS_VER: bool>(
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask: &mut [u8; SEG_MASK_LEN],
    sign: i32,
    bitdepth_max: i32,
) {
    let w = w as usize;
    let h = h as usize;
    let sign = sign as u8;
    let bd_max = bitdepth_max as i32;

    // Determine bitdepth and intermediate_bits from bitdepth_max
    let bitdepth = if bitdepth_max == 1023 { 10u32 } else { 12u32 };
    let intermediate_bits = (14 - bitdepth) as u32; // 4 for 10-bit, 2 for 12-bit
    let sh = intermediate_bits + 6;
    let rnd = (32i32 << intermediate_bits) + 8192 * 64;
    let mask_sh = bitdepth + intermediate_bits - 4;
    let mask_rnd = 1u16 << (mask_sh - 5);

    let mask_w = if SS_HOR { w >> 1 } else { w };
    let mut mask_off = 0usize;

    for row_h in 0..h {
        let row_offset = row_h * w;
        let tmp1_row = &tmp1[row_offset..][..w];
        let tmp2_row = &tmp2[row_offset..][..w];
        // dst is bytes, but we write u16 pixels — each pixel is 2 bytes
        let dst_row_bytes = &mut dst[row_h * dst_stride..][..w * 2];
        let dst_row: &mut [u16] = zerocopy::FromBytes::mut_slice_from(dst_row_bytes).unwrap();

        let mut x = 0;
        while x < w {
            let diff = tmp1_row[x].abs_diff(tmp2_row[x]);
            let m = std::cmp::min(38 + ((diff.saturating_add(mask_rnd)) >> mask_sh), 64) as u8;

            let t1 = tmp1_row[x] as i32;
            let t2 = tmp2_row[x] as i32;
            let pixel = (t1 * m as i32 + t2 * (64 - m as i32) + rnd) >> sh;
            dst_row[x] = pixel.clamp(0, bd_max) as u16;

            if SS_HOR {
                x += 1;
                let diff2 = tmp1_row[x].abs_diff(tmp2_row[x]);
                let n = std::cmp::min(38 + ((diff2.saturating_add(mask_rnd)) >> mask_sh), 64) as u8;

                let t1 = tmp1_row[x] as i32;
                let t2 = tmp2_row[x] as i32;
                let pixel = (t1 * n as i32 + t2 * (64 - n as i32) + rnd) >> sh;
                dst_row[x] = pixel.clamp(0, bd_max) as u16;

                let mask_x = x >> 1;
                if SS_VER && (row_h & 1 != 0) {
                    let prev = mask[mask_off + mask_x];
                    mask[mask_off + mask_x] =
                        (((m as u16 + n as u16 + 2 - sign as u16) + prev as u16) >> 2) as u8;
                } else if SS_VER {
                    mask[mask_off + mask_x] = m + n;
                } else {
                    mask[mask_off + mask_x] = ((m as u16 + n as u16 + 1 - sign as u16) >> 1) as u8;
                }
            } else {
                mask[mask_off + x] = m;
            }
            x += 1;
        }

        if !SS_VER || (row_h & 1 != 0) {
            mask_off += mask_w;
        }
    }
}

/// w_mask 444 16bpc (no subsampling)
#[cfg(target_arch = "x86_64")]
#[arcane]
fn w_mask_444_16bpc_avx2_safe(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask: &mut [u8; SEG_MASK_LEN],
    sign: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    w_mask_16bpc_avx2_safe_impl::<false, false>(
        &mut *dst,
        dst_stride,
        tmp1,
        tmp2,
        w,
        h,
        mask,
        sign,
        bitdepth_max,
    );
}

/// w_mask 422 16bpc (horizontal subsampling only)
#[cfg(target_arch = "x86_64")]
#[arcane]
fn w_mask_422_16bpc_avx2_safe(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask: &mut [u8; SEG_MASK_LEN],
    sign: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    w_mask_16bpc_avx2_safe_impl::<true, false>(
        &mut *dst,
        dst_stride,
        tmp1,
        tmp2,
        w,
        h,
        mask,
        sign,
        bitdepth_max,
    );
}

/// w_mask 420 16bpc (horizontal and vertical subsampling)
#[cfg(target_arch = "x86_64")]
#[arcane]
fn w_mask_420_16bpc_avx2_safe(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask: &mut [u8; SEG_MASK_LEN],
    sign: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    w_mask_16bpc_avx2_safe_impl::<true, true>(
        &mut *dst,
        dst_stride,
        tmp1,
        tmp2,
        w,
        h,
        mask,
        sign,
        bitdepth_max,
    );
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn w_mask_444_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask: &mut [u8; SEG_MASK_LEN],
    sign: i32,
    bitdepth_max: i32,
    _dst: *const FFISafe<PicOffset>,
) {
    let dst = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h as usize * dst_stride as usize)
    };
    w_mask_444_16bpc_avx2_safe(
        Desktop64::forge_token_dangerously(),
        dst,
        dst_stride as usize,
        tmp1,
        tmp2,
        w,
        h,
        mask,
        sign,
        bitdepth_max,
    )
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn w_mask_422_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask: &mut [u8; SEG_MASK_LEN],
    sign: i32,
    bitdepth_max: i32,
    _dst: *const FFISafe<PicOffset>,
) {
    let dst = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h as usize * dst_stride as usize)
    };
    w_mask_422_16bpc_avx2_safe(
        Desktop64::forge_token_dangerously(),
        dst,
        dst_stride as usize,
        tmp1,
        tmp2,
        w,
        h,
        mask,
        sign,
        bitdepth_max,
    )
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn w_mask_420_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask: &mut [u8; SEG_MASK_LEN],
    sign: i32,
    bitdepth_max: i32,
    _dst: *const FFISafe<PicOffset>,
) {
    let dst = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h as usize * dst_stride as usize)
    };
    w_mask_420_16bpc_avx2_safe(
        Desktop64::forge_token_dangerously(),
        dst,
        dst_stride as usize,
        tmp1,
        tmp2,
        w,
        h,
        mask,
        sign,
        bitdepth_max,
    )
}

// ============================================================================
// BILINEAR 16BPC
// ============================================================================

/// Horizontal bilinear filter for 16bpc using AVX2
/// Formula: pixel = 16 * x0 + mx * (x1 - x0) = (16 - mx) * x0 + mx * x1
#[cfg(target_arch = "x86_64")]
#[rite]
fn h_bilin_16bpc_avx2_inner(_token: Desktop64, dst: &mut [i32], src: &[u16], w: usize, mx: i32) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    // Bilinear weights: w0 = (16 - mx), w1 = mx
    let w0 = _mm256_set1_epi32(16 - mx);
    let w1 = _mm256_set1_epi32(mx);

    let mut col = 0usize;

    // Process 8 pixels at a time
    while col + 8 <= w {
        // Load 9 consecutive 16-bit pixels (8 output pixels need pixels 0..8)
        // s offset = col
        let p0_7 = loadu_128!(<&[u16; 8]>::try_from(&src[col..col + 8]).unwrap()); // pixels 0-7
        let p1_8 = loadu_128!(<&[u16; 8]>::try_from(&src[col + 1..col + 9]).unwrap()); // pixels 1-8

        // Zero-extend to 32-bit
        let p0_lo = _mm256_cvtepu16_epi32(p0_7);
        let p1_lo = _mm256_cvtepu16_epi32(p1_8);

        // Compute: w0 * p0 + w1 * p1
        let term0 = _mm256_mullo_epi32(p0_lo, w0);
        let term1 = _mm256_mullo_epi32(p1_lo, w1);
        let result = _mm256_add_epi32(term0, term1);

        storeu_256!(
            <&mut [i32; 8]>::try_from(&mut dst[col..col + 8]).unwrap(),
            result
        );
        col += 8;
    }

    // Scalar fallback
    while col < w {
        let x0 = src[col] as i32;
        let x1 = src[col + 1] as i32;
        dst[col] = 16 * x0 + mx * (x1 - x0);
        col += 1;
    }
}
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn h_bilin_16bpc_avx2(dst: *mut i32, src: *const u16, w: usize, mx: i32) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe { h_bilin_16bpc_avx2_inner(token, dst, src, w, mx) }
}

/// Vertical bilinear filter for 16bpc using AVX2
/// Applies vertical filter to 32-bit intermediate, outputs clamped u16
#[cfg(target_arch = "x86_64")]
#[rite]
fn v_bilin_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    mid: &[[i32; MID_STRIDE]],
    w: usize,
    y: usize,
    my: i32,
    sh: i32,
    max: i32,
) {
    let mut dst = dst.flex_mut();
    let w0 = _mm256_set1_epi32(16 - my);
    let w1 = _mm256_set1_epi32(my);
    let rnd = _mm256_set1_epi32((1 << sh) >> 1);
    let shift_count = _mm_cvtsi32_si128(sh);
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi32(max);

    let mut col = 0usize;

    while col + 8 <= w {
        // Load 8 intermediate values from rows y and y+1
        let r0 = loadu_256!(<&[i32; 8]>::try_from(&mid[y][col..col + 8]).unwrap());
        let r1 = loadu_256!(<&[i32; 8]>::try_from(&mid[y + 1][col..col + 8]).unwrap());

        // Compute: w0 * r0 + w1 * r1
        let term0 = _mm256_mullo_epi32(r0, w0);
        let term1 = _mm256_mullo_epi32(r1, w1);
        let sum = _mm256_add_epi32(term0, term1);

        // Round, shift, clamp
        let shifted = _mm256_sra_epi32(_mm256_add_epi32(sum, rnd), shift_count);
        let clamped = _mm256_min_epi32(_mm256_max_epi32(shifted, zero), max_val);

        // Pack to 16-bit
        let packed = _mm256_packus_epi32(clamped, zero);
        let result = _mm256_permute4x64_epi64(packed, 0b00_00_10_00);

        storeu_128!(
            <&mut [u16; 8]>::try_from(&mut dst[col..col + 8]).unwrap(),
            _mm256_castsi256_si128(result)
        );
        col += 8;
    }

    // Scalar fallback
    while col < w {
        let r0 = mid[y][col];
        let r1 = mid[y + 1][col];
        let pixel = 16 * r0 + my * (r1 - r0);
        let r = (1 << sh) >> 1;
        let val = ((pixel + r) >> sh).clamp(0, max);
        dst[col] = val as u16;
        col += 1;
    }
}
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn v_bilin_16bpc_avx2(
    dst: *mut u16,
    mid: &[[i32; MID_STRIDE]],
    w: usize,
    y: usize,
    my: i32,
    sh: i32,
    max: i32,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe { v_bilin_16bpc_avx2_inner(token, dst, mid, w, y, my, sh, max) }
}

/// Vertical bilinear filter for 16bpc prep - outputs i16 with bias subtraction
#[cfg(target_arch = "x86_64")]
#[rite]
fn v_bilin_16bpc_prep_avx2_inner(
    _token: Desktop64,
    dst: &mut [i16],
    mid: &[[i32; MID_STRIDE]],
    w: usize,
    y: usize,
    my: i32,
    sh: i32,
    prep_bias: i32,
) {
    let mut dst = dst.flex_mut();
    let w0 = _mm256_set1_epi32(16 - my);
    let w1 = _mm256_set1_epi32(my);
    let rnd = _mm256_set1_epi32((1 << sh) >> 1);
    let shift_count = _mm_cvtsi32_si128(sh);
    let bias = _mm256_set1_epi32(prep_bias);

    let mut col = 0usize;

    while col + 8 <= w {
        let r0 = loadu_256!(<&[i32; 8]>::try_from(&mid[y][col..col + 8]).unwrap());
        let r1 = loadu_256!(<&[i32; 8]>::try_from(&mid[y + 1][col..col + 8]).unwrap());

        let term0 = _mm256_mullo_epi32(r0, w0);
        let term1 = _mm256_mullo_epi32(r1, w1);
        let sum = _mm256_add_epi32(term0, term1);

        // Round, shift, subtract bias
        let shifted = _mm256_sra_epi32(_mm256_add_epi32(sum, rnd), shift_count);
        let biased = _mm256_sub_epi32(shifted, bias);

        // Pack to signed 16-bit
        let packed = _mm256_packs_epi32(biased, biased);
        let result = _mm256_permute4x64_epi64(packed, 0b00_00_10_00);

        storeu_128!(
            <&mut [i16; 8]>::try_from(&mut dst[col..col + 8]).unwrap(),
            _mm256_castsi256_si128(result)
        );
        col += 8;
    }

    // Scalar fallback
    while col < w {
        let r0 = mid[y][col];
        let r1 = mid[y + 1][col];
        let pixel = 16 * r0 + my * (r1 - r0);
        let r = (1 << sh) >> 1;
        let val = ((pixel + r) >> sh) - prep_bias;
        dst[col] = val as i16;
        col += 1;
    }
}
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn v_bilin_16bpc_prep_avx2(
    dst: *mut i16,
    mid: &[[i32; MID_STRIDE]],
    w: usize,
    y: usize,
    my: i32,
    sh: i32,
    prep_bias: i32,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe { v_bilin_16bpc_prep_avx2_inner(token, dst, mid, w, y, my, sh, prep_bias) }
}

/// Horizontal bilinear filter for 16bpc put (H-only case)
/// Outputs directly to u16 with shift and clamp
#[cfg(target_arch = "x86_64")]
#[rite]
fn h_bilin_16bpc_put_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    src: &[u16],
    w: usize,
    mx: i32,
    bd_max: i32,
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    let w0 = _mm256_set1_epi32(16 - mx);
    let w1 = _mm256_set1_epi32(mx);
    let rnd = _mm256_set1_epi32(8);
    let shift_count = _mm_cvtsi32_si128(4);
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi32(bd_max);

    let mut col = 0usize;

    while col + 8 <= w {
        // s offset = col
        let p0 = _mm256_cvtepu16_epi32(loadu_128!(
            <&[u16; 8]>::try_from(&src[col..col + 8]).unwrap()
        ));
        let p1 = _mm256_cvtepu16_epi32(loadu_128!(
            <&[u16; 8]>::try_from(&src[col + 1..col + 9]).unwrap()
        ));

        // (16 - mx) * p0 + mx * p1
        let term0 = _mm256_mullo_epi32(p0, w0);
        let term1 = _mm256_mullo_epi32(p1, w1);
        let sum = _mm256_add_epi32(term0, term1);

        // (sum + 8) >> 4, clamp to [0, max]
        let shifted = _mm256_sra_epi32(_mm256_add_epi32(sum, rnd), shift_count);
        let clamped = _mm256_min_epi32(_mm256_max_epi32(shifted, zero), max_val);

        // Pack to u16
        let packed = _mm256_packus_epi32(clamped, zero);
        let result = _mm256_permute4x64_epi64(packed, 0b00_00_10_00);

        storeu_128!(
            <&mut [u16; 8]>::try_from(&mut dst[col..col + 8]).unwrap(),
            _mm256_castsi256_si128(result)
        );
        col += 8;
    }

    // Scalar fallback
    while col < w {
        let x0 = src[col] as i32;
        let x1 = src[col + 1] as i32;
        let pixel = (16 - mx) * x0 + mx * x1;
        let result = ((pixel + 8) >> 4).clamp(0, bd_max);
        dst[col] = result as u16;
        col += 1;
    }
}
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn h_bilin_16bpc_put_avx2(dst: *mut u16, src: *const u16, w: usize, mx: i32, bd_max: i32) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe { h_bilin_16bpc_put_avx2_inner(token, dst, src, w, mx, bd_max) }
}

/// Vertical bilinear filter for 16bpc put (V-only case)
/// Reads directly from u16 source, outputs u16
#[cfg(target_arch = "x86_64")]
#[rite]
fn v_bilin_16bpc_direct_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    src: &[u16],
    src_stride: isize,
    w: usize,
    my: i32,
    bd_max: i32,
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    let w0 = _mm256_set1_epi32(16 - my);
    let w1 = _mm256_set1_epi32(my);
    let rnd = _mm256_set1_epi32(8);
    let shift_count = _mm_cvtsi32_si128(4);
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi32(bd_max);

    let mut col = 0usize;

    while col + 8 <= w {
        // Load from row 0 and row 1
        let p0 = _mm256_cvtepu16_epi32(loadu_128!(
            <&[u16; 8]>::try_from(&src[col..col + 8]).unwrap()
        ));
        let p1 = _mm256_cvtepu16_epi32(loadu_128!(<&[u16; 8]>::try_from(
            &src[src_stride as usize + col..src_stride as usize + col + 8]
        )
        .unwrap()));

        // (16 - my) * p0 + my * p1
        let term0 = _mm256_mullo_epi32(p0, w0);
        let term1 = _mm256_mullo_epi32(p1, w1);
        let sum = _mm256_add_epi32(term0, term1);

        // (sum + 8) >> 4, clamp to [0, max]
        let shifted = _mm256_sra_epi32(_mm256_add_epi32(sum, rnd), shift_count);
        let clamped = _mm256_min_epi32(_mm256_max_epi32(shifted, zero), max_val);

        // Pack to u16
        let packed = _mm256_packus_epi32(clamped, zero);
        let result = _mm256_permute4x64_epi64(packed, 0b00_00_10_00);

        storeu_128!(
            <&mut [u16; 8]>::try_from(&mut dst[col..col + 8]).unwrap(),
            _mm256_castsi256_si128(result)
        );
        col += 8;
    }

    // Scalar fallback
    while col < w {
        let x0 = src[col] as i32;
        let x1 = src[src_stride as usize + col] as i32;
        let pixel = (16 - my) * x0 + my * x1;
        let result = ((pixel + 8) >> 4).clamp(0, bd_max);
        dst[col] = result as u16;
        col += 1;
    }
}
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn v_bilin_16bpc_direct_avx2(
    dst: *mut u16,
    src: *const u16,
    src_stride: isize,
    w: usize,
    my: i32,
    bd_max: i32,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe { v_bilin_16bpc_direct_avx2_inner(token, dst, src, src_stride, w, my, bd_max) }
}

/// Horizontal bilinear filter for 16bpc prep (H-only case)
/// Outputs i16 with PREP_BIAS subtracted
#[cfg(target_arch = "x86_64")]
#[rite]
fn h_bilin_16bpc_prep_direct_avx2_inner(
    _token: Desktop64,
    dst: &mut [i16],
    src: &[u16],
    w: usize,
    mx: i32,
    prep_bias: i32,
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    let w0 = _mm256_set1_epi32(16 - mx);
    let w1 = _mm256_set1_epi32(mx);
    let rnd = _mm256_set1_epi32(8);
    let shift_count = _mm_cvtsi32_si128(4);
    let bias = _mm256_set1_epi32(prep_bias);

    let mut col = 0usize;

    while col + 8 <= w {
        // s offset = col
        let p0 = _mm256_cvtepu16_epi32(loadu_128!(
            <&[u16; 8]>::try_from(&src[col..col + 8]).unwrap()
        ));
        let p1 = _mm256_cvtepu16_epi32(loadu_128!(
            <&[u16; 8]>::try_from(&src[col + 1..col + 9]).unwrap()
        ));

        // (16 - mx) * p0 + mx * p1
        let term0 = _mm256_mullo_epi32(p0, w0);
        let term1 = _mm256_mullo_epi32(p1, w1);
        let sum = _mm256_add_epi32(term0, term1);

        // (sum + 8) >> 4 - bias
        let shifted = _mm256_sra_epi32(_mm256_add_epi32(sum, rnd), shift_count);
        let biased = _mm256_sub_epi32(shifted, bias);

        // Pack to i16
        let packed = _mm256_packs_epi32(biased, biased);
        let result = _mm256_permute4x64_epi64(packed, 0b00_00_10_00);

        storeu_128!(
            <&mut [i16; 8]>::try_from(&mut dst[col..col + 8]).unwrap(),
            _mm256_castsi256_si128(result)
        );
        col += 8;
    }

    // Scalar fallback
    while col < w {
        let x0 = src[col] as i32;
        let x1 = src[col + 1] as i32;
        let pixel = (16 - mx) * x0 + mx * x1;
        let result = ((pixel + 8) >> 4) - prep_bias;
        dst[col] = result as i16;
        col += 1;
    }
}
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn h_bilin_16bpc_prep_direct_avx2(
    dst: *mut i16,
    src: *const u16,
    w: usize,
    mx: i32,
    prep_bias: i32,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe { h_bilin_16bpc_prep_direct_avx2_inner(token, dst, src, w, mx, prep_bias) }
}

/// Vertical bilinear filter for 16bpc prep (V-only case)
/// Reads directly from u16 source, outputs i16 with bias
#[cfg(target_arch = "x86_64")]
#[rite]
fn v_bilin_16bpc_prep_direct_avx2_inner(
    _token: Desktop64,
    dst: &mut [i16],
    src: &[u16],
    src_stride: isize,
    w: usize,
    my: i32,
    prep_bias: i32,
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    let w0 = _mm256_set1_epi32(16 - my);
    let w1 = _mm256_set1_epi32(my);
    let rnd = _mm256_set1_epi32(8);
    let shift_count = _mm_cvtsi32_si128(4);
    let bias = _mm256_set1_epi32(prep_bias);

    let mut col = 0usize;

    while col + 8 <= w {
        // Load from row 0 and row 1
        let p0 = _mm256_cvtepu16_epi32(loadu_128!(
            <&[u16; 8]>::try_from(&src[col..col + 8]).unwrap()
        ));
        let p1 = _mm256_cvtepu16_epi32(loadu_128!(<&[u16; 8]>::try_from(
            &src[src_stride as usize + col..src_stride as usize + col + 8]
        )
        .unwrap()));

        // (16 - my) * p0 + my * p1
        let term0 = _mm256_mullo_epi32(p0, w0);
        let term1 = _mm256_mullo_epi32(p1, w1);
        let sum = _mm256_add_epi32(term0, term1);

        // (sum + 8) >> 4 - bias
        let shifted = _mm256_sra_epi32(_mm256_add_epi32(sum, rnd), shift_count);
        let biased = _mm256_sub_epi32(shifted, bias);

        // Pack to i16
        let packed = _mm256_packs_epi32(biased, biased);
        let result = _mm256_permute4x64_epi64(packed, 0b00_00_10_00);

        storeu_128!(
            <&mut [i16; 8]>::try_from(&mut dst[col..col + 8]).unwrap(),
            _mm256_castsi256_si128(result)
        );
        col += 8;
    }

    // Scalar fallback
    while col < w {
        let x0 = src[col] as i32;
        let x1 = src[src_stride as usize + col] as i32;
        let pixel = (16 - my) * x0 + my * x1;
        let result = ((pixel + 8) >> 4) - prep_bias;
        dst[col] = result as i16;
        col += 1;
    }
}
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn v_bilin_16bpc_prep_direct_avx2(
    dst: *mut i16,
    src: *const u16,
    src_stride: isize,
    w: usize,
    my: i32,
    prep_bias: i32,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe { v_bilin_16bpc_prep_direct_avx2_inner(token, dst, src, src_stride, w, my, prep_bias) }
}

/// Bilinear put for 16bpc
#[cfg(target_arch = "x86_64")]
#[cfg(feature = "asm")]
#[arcane]
unsafe fn put_bilin_16bpc_avx2_inner(
    _token: Desktop64,
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
) {
    let w = w as usize;
    let h = h as usize;
    let mx = mx as usize;
    let my = my as usize;
    let dst = dst_ptr as *mut u16;
    let src = src_ptr as *const u16;
    let dst_stride = dst_stride / 2; // Convert byte stride to u16 stride
    let src_stride = src_stride / 2;
    let bd_max = bitdepth_max as i32;

    // For 16bpc: intermediate_bits = 4
    // H pass shift = 4 - intermediate_bits = 0 (no shift for intermediate)
    // V pass shift = 4 + intermediate_bits = 8
    let intermediate_bits = 4i32;
    let _h_pass_sh = 4 - intermediate_bits; // = 0
    let v_pass_sh = 4 + intermediate_bits; // = 8

    unsafe {
        if mx != 0 {
            if my != 0 {
                // H+V filtering using SIMD
                let tmp_h = h + 1;
                let mut mid = take_mid_i32_130();

                // Horizontal pass using SIMD (output unshifted)
                for y in 0..tmp_h {
                    let src_row = src.offset(y as isize * src_stride);
                    h_bilin_16bpc_avx2_inner(_token, &mut mid[y], src_row, w, mx as i32);
                }

                // Vertical pass using SIMD
                for y in 0..h {
                    let dst_row = dst.offset(y as isize * dst_stride);
                    v_bilin_16bpc_avx2_inner(
                        _token, dst_row, &*mid, w, y, my as i32, v_pass_sh, bd_max,
                    );
                }
                put_mid_i32_130(mid);
            } else {
                // H-only filtering (SIMD)
                for y in 0..h {
                    let src_row = src.offset(y as isize * src_stride);
                    let dst_row = dst.offset(y as isize * dst_stride);
                    h_bilin_16bpc_put_avx2_inner(_token, dst_row, src_row, w, mx as i32, bd_max);
                }
            }
        } else if my != 0 {
            // V-only filtering (SIMD)
            for y in 0..h {
                let src_row = src.offset(y as isize * src_stride);
                let dst_row = dst.offset(y as isize * dst_stride);
                v_bilin_16bpc_direct_avx2_inner(
                    _token, dst_row, src_row, src_stride, w, my as i32, bd_max,
                );
            }
        } else {
            // Simple copy
            for y in 0..h {
                let src_row = src.offset(y as isize * src_stride);
                let dst_row = dst.offset(y as isize * dst_stride);
                std::ptr::copy_nonoverlapping(src_row, dst_row, w);
            }
        }
    }
}

/// Non-FFI wrapper for bilinear put 16bpc (no FFISafe params)
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn put_bilin_16bpc_avx2_impl(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe {
        put_bilin_16bpc_avx2_inner(
            token,
            dst_ptr,
            dst_stride,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn put_bilin_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    _dst: *const FFISafe<PicOffset>,
    _src: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe {
        put_bilin_16bpc_avx2_inner(
            token,
            dst_ptr,
            dst_stride,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
        )
    }
}

/// Bilinear prep for 16bpc (scalar implementation)
#[cfg(target_arch = "x86_64")]
#[cfg(feature = "asm")]
#[arcane]
unsafe fn prep_bilin_16bpc_avx2_inner(
    _token: Desktop64,
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
) {
    let w = w as usize;
    let h = h as usize;
    let mx = mx as usize;
    let my = my as usize;
    let src = src_ptr as *const u16;
    let src_stride = src_stride / 2;

    // For 16bpc prep: PREP_BIAS = 8192
    let prep_bias = 8192i32;

    // For 16bpc: intermediate_bits = 4
    // H pass shift = 4 - intermediate_bits = 0 (no shift for intermediate)
    // V pass shift = 4 + intermediate_bits = 8
    let intermediate_bits = 4i32;
    let _h_pass_sh = 4 - intermediate_bits; // = 0
    let v_pass_sh = 4 + intermediate_bits; // = 8

    unsafe {
        if mx != 0 {
            if my != 0 {
                // H+V filtering using SIMD
                let tmp_h = h + 1;
                let mut mid = take_mid_i32_130();

                // Horizontal pass using SIMD
                for y in 0..tmp_h {
                    let src_row = src.offset(y as isize * src_stride);
                    h_bilin_16bpc_avx2_inner(_token, &mut mid[y], src_row, w, mx as i32);
                }

                // Vertical pass using SIMD (with bias subtraction)
                for y in 0..h {
                    let dst_row = tmp.add(y * w);
                    v_bilin_16bpc_prep_avx2_inner(
                        _token, dst_row, &*mid, w, y, my as i32, v_pass_sh, prep_bias,
                    );
                }
                put_mid_i32_130(mid);
            } else {
                // H-only filtering (SIMD)
                for y in 0..h {
                    let src_row = src.offset(y as isize * src_stride);
                    let dst_row = tmp.add(y * w);
                    h_bilin_16bpc_prep_direct_avx2_inner(
                        _token, dst_row, src_row, w, mx as i32, prep_bias,
                    );
                }
            }
        } else if my != 0 {
            // V-only filtering (SIMD)
            for y in 0..h {
                let src_row = src.offset(y as isize * src_stride);
                let dst_row = tmp.add(y * w);
                v_bilin_16bpc_prep_direct_avx2_inner(
                    _token, dst_row, src_row, src_stride, w, my as i32, prep_bias,
                );
            }
            // Simple copy to prep format
            for y in 0..h {
                let src_row = src.offset(y as isize * src_stride);
                let dst_row = tmp.add(y * w);
                for x in 0..w {
                    let pixel = src_row[x] as i32;
                    *dst_row.add(x) = (pixel - prep_bias) as i16;
                }
            }
        }
    }
}

/// Non-FFI wrapper for bilinear prep 16bpc (no FFISafe params)
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn prep_bilin_16bpc_avx2_impl(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe { prep_bilin_16bpc_avx2_inner(token, tmp, src_ptr, src_stride, w, h, mx, my) }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn prep_bilin_16bpc_avx2(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    _bitdepth_max: i32,
    _src: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    unsafe { prep_bilin_16bpc_avx2_inner(token, tmp, src_ptr, src_stride, w, h, mx, my) }
}

// ============================================================================
// Safe dispatch wrappers for x86_64 AVX2
// Each returns true if SIMD was used (i.e., AVX2 is available).
// ============================================================================

#[cfg(target_arch = "x86_64")]
pub fn avg_dispatch<BD: BitDepth>(
    dst: PicOffset,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    bd: BD,
) -> bool {
    use crate::include::common::bitdepth::BPC;
    let Some(token) = crate::src::cpu::summon_avx2() else {
        return false;
    };
    use zerocopy::AsBytes;
    let (mut dst_guard, dst_base) = dst.full_guard_mut::<BD>();
    let dst_bytes = dst_guard.as_bytes_mut();
    let pixel_size = std::mem::size_of::<BD::Pixel>();
    let dst_offset = dst_base * pixel_size;
    let dst_stride = dst.stride();
    let bd_c = bd.into_c();
    match BD::BPC {
        BPC::BPC8 => avg_8bpc_avx2_safe(
            token,
            &mut dst_bytes[dst_offset..],
            dst_stride as usize,
            tmp1,
            tmp2,
            w,
            h,
        ),
        BPC::BPC16 => avg_16bpc_avx2_safe(
            token,
            &mut dst_bytes[dst_offset..],
            dst_stride as usize,
            tmp1,
            tmp2,
            w,
            h,
            bd_c,
        ),
    }
    true
}

#[cfg(target_arch = "x86_64")]
pub fn w_avg_dispatch<BD: BitDepth>(
    dst: PicOffset,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    weight: i32,
    bd: BD,
) -> bool {
    use crate::include::common::bitdepth::BPC;
    let Some(token) = crate::src::cpu::summon_avx2() else {
        return false;
    };
    use zerocopy::AsBytes;
    let (mut dst_guard, dst_base) = dst.full_guard_mut::<BD>();
    let dst_bytes = dst_guard.as_bytes_mut();
    let pixel_size = std::mem::size_of::<BD::Pixel>();
    let dst_offset = dst_base * pixel_size;
    let dst_stride = dst.stride();
    let bd_c = bd.into_c();
    match BD::BPC {
        BPC::BPC8 => w_avg_8bpc_avx2_safe(
            token,
            &mut dst_bytes[dst_offset..],
            dst_stride as usize,
            tmp1,
            tmp2,
            w,
            h,
            weight,
        ),
        BPC::BPC16 => w_avg_16bpc_avx2_safe(
            token,
            &mut dst_bytes[dst_offset..],
            dst_stride as usize,
            tmp1,
            tmp2,
            w,
            h,
            weight,
            bd_c,
        ),
    }
    true
}

#[cfg(target_arch = "x86_64")]
pub fn mask_dispatch<BD: BitDepth>(
    dst: PicOffset,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask: &[u8],
    bd: BD,
) -> bool {
    use crate::include::common::bitdepth::BPC;
    let Some(token) = crate::src::cpu::summon_avx2() else {
        return false;
    };
    use zerocopy::AsBytes;
    let (mut dst_guard, dst_base) = dst.full_guard_mut::<BD>();
    let dst_bytes = dst_guard.as_bytes_mut();
    let pixel_size = std::mem::size_of::<BD::Pixel>();
    let dst_offset = dst_base * pixel_size;
    let dst_stride = dst.stride() as usize;
    let bd_c = bd.into_c();
    match BD::BPC {
        BPC::BPC8 => mask_8bpc_avx2_safe(
            token,
            &mut dst_bytes[dst_offset..],
            dst_stride,
            tmp1,
            tmp2,
            w,
            h,
            mask,
        ),
        BPC::BPC16 => mask_16bpc_avx2_safe(
            token,
            &mut dst_bytes[dst_offset..],
            dst_stride,
            tmp1,
            tmp2,
            w,
            h,
            mask,
            bd_c,
        ),
    }
    true
}

#[cfg(target_arch = "x86_64")]
pub fn blend_dispatch<BD: BitDepth>(
    dst: PicOffset,
    tmp: &[BD::Pixel; SCRATCH_INTER_INTRA_BUF_LEN],
    w: i32,
    h: i32,
    mask: &[u8],
) -> bool {
    use crate::include::common::bitdepth::BPC;
    let Some(token) = crate::src::cpu::summon_avx2() else {
        return false;
    };
    use zerocopy::AsBytes;
    let (mut dst_guard, dst_base) = dst.full_guard_mut::<BD>();
    let dst_bytes = dst_guard.as_bytes_mut();
    let pixel_size = std::mem::size_of::<BD::Pixel>();
    let dst_offset = dst_base * pixel_size;
    let dst_stride = dst.stride() as usize;
    let tmp_bytes = tmp.as_bytes();
    match BD::BPC {
        BPC::BPC8 => blend_8bpc_avx2_safe(
            token,
            &mut dst_bytes[dst_offset..],
            dst_stride,
            tmp_bytes,
            w,
            h,
            mask,
        ),
        BPC::BPC16 => blend_16bpc_avx2_safe(
            token,
            &mut dst_bytes[dst_offset..],
            dst_stride,
            tmp_bytes,
            w,
            h,
            mask,
        ),
    }
    true
}

#[cfg(target_arch = "x86_64")]
pub fn blend_dir_dispatch<BD: BitDepth>(
    is_h: bool,
    dst: PicOffset,
    tmp: &[BD::Pixel; SCRATCH_LAP_LEN],
    w: i32,
    h: i32,
) -> bool {
    use crate::include::common::bitdepth::BPC;
    let Some(token) = crate::src::cpu::summon_avx2() else {
        return false;
    };
    use zerocopy::AsBytes;
    let (mut dst_guard, dst_base) = dst.full_guard_mut::<BD>();
    let dst_bytes = dst_guard.as_bytes_mut();
    let pixel_size = std::mem::size_of::<BD::Pixel>();
    let dst_offset = dst_base * pixel_size;
    let dst_stride = dst.stride() as usize;
    let tmp_bytes = tmp.as_bytes();
    match (BD::BPC, is_h) {
        (BPC::BPC8, true) => blend_h_8bpc_avx2_safe(
            token,
            &mut dst_bytes[dst_offset..],
            dst_stride,
            tmp_bytes,
            w,
            h,
        ),
        (BPC::BPC8, false) => blend_v_8bpc_avx2_safe(
            token,
            &mut dst_bytes[dst_offset..],
            dst_stride,
            tmp_bytes,
            w,
            h,
        ),
        (BPC::BPC16, true) => blend_h_16bpc_avx2_safe(
            token,
            &mut dst_bytes[dst_offset..],
            dst_stride,
            tmp_bytes,
            w,
            h,
        ),
        (BPC::BPC16, false) => blend_v_16bpc_avx2_safe(
            token,
            &mut dst_bytes[dst_offset..],
            dst_stride,
            tmp_bytes,
            w,
            h,
        ),
    }
    true
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn w_mask_dispatch<BD: BitDepth>(
    layout: Rav1dPixelLayoutSubSampled,
    dst: PicOffset,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask: &mut [u8; SEG_MASK_LEN],
    sign: i32,
    bd: BD,
) -> bool {
    use crate::include::common::bitdepth::BPC;
    let Some(token) = crate::src::cpu::summon_avx2() else {
        return false;
    };
    use zerocopy::AsBytes;
    let (mut dst_guard, dst_base) = dst.full_guard_mut::<BD>();
    let dst_bytes = dst_guard.as_bytes_mut();
    let pixel_size = std::mem::size_of::<BD::Pixel>();
    let dst_offset = dst_base * pixel_size;
    let dst_stride = dst.stride() as usize;
    let bd_c = bd.into_c();
    match (BD::BPC, layout) {
        (BPC::BPC8, Rav1dPixelLayoutSubSampled::I420) => w_mask_420_8bpc_avx2_safe(
            token,
            &mut dst_bytes[dst_offset..],
            dst_stride,
            tmp1,
            tmp2,
            w,
            h,
            mask,
            sign,
        ),
        (BPC::BPC8, Rav1dPixelLayoutSubSampled::I422) => w_mask_422_8bpc_avx2_safe(
            token,
            &mut dst_bytes[dst_offset..],
            dst_stride,
            tmp1,
            tmp2,
            w,
            h,
            mask,
            sign,
        ),
        (BPC::BPC8, Rav1dPixelLayoutSubSampled::I444) => w_mask_444_8bpc_avx2_safe(
            token,
            &mut dst_bytes[dst_offset..],
            dst_stride,
            tmp1,
            tmp2,
            w,
            h,
            mask,
            sign,
        ),
        (BPC::BPC16, Rav1dPixelLayoutSubSampled::I420) => w_mask_420_16bpc_avx2_safe(
            token,
            &mut dst_bytes[dst_offset..],
            dst_stride,
            tmp1,
            tmp2,
            w,
            h,
            mask,
            sign,
            bd_c,
        ),
        (BPC::BPC16, Rav1dPixelLayoutSubSampled::I422) => w_mask_422_16bpc_avx2_safe(
            token,
            &mut dst_bytes[dst_offset..],
            dst_stride,
            tmp1,
            tmp2,
            w,
            h,
            mask,
            sign,
            bd_c,
        ),
        (BPC::BPC16, Rav1dPixelLayoutSubSampled::I444) => w_mask_444_16bpc_avx2_safe(
            token,
            &mut dst_bytes[dst_offset..],
            dst_stride,
            tmp1,
            tmp2,
            w,
            h,
            mask,
            sign,
            bd_c,
        ),
    }
    true
}

/// Safe arcane entry point for put_8tap 8bpc dispatch (calls #[rite] inner fn).
#[cfg(target_arch = "x86_64")]
#[arcane]
fn put_8tap_8bpc_dispatch_inner(
    token: Desktop64,
    dst: &mut [u8],
    dst_stride: isize,
    src: &[u8],
    src_base: usize,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    h_filter: Rav1dFilterMode,
    v_filter: Rav1dFilterMode,
) {
    put_8tap_8bpc_avx2_impl_inner(
        token, dst, dst_stride, src, src_base, src_stride, w, h, mx, my, h_filter, v_filter,
    );
}

/// Safe arcane entry point for put_bilin 8bpc dispatch.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn put_bilin_8bpc_dispatch_inner(
    token: Desktop64,
    dst: &mut [u8],
    dst_stride: isize,
    src: &[u8],
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
) {
    put_bilin_8bpc_avx2_impl_inner(token, dst, dst_stride, src, src_stride, w, h, mx, my);
}

/// Safe arcane entry point for put_8tap 16bpc dispatch.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn put_8tap_16bpc_dispatch_inner(
    token: Desktop64,
    dst: &mut [u16],
    dst_stride: isize,
    src: &[u16],
    src_base: usize,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bd_c: i32,
    h_filter: Rav1dFilterMode,
    v_filter: Rav1dFilterMode,
) {
    put_8tap_16bpc_avx2_impl_inner(
        token, dst, dst_stride, src, src_base, src_stride, w, h, mx, my, bd_c, h_filter, v_filter,
    );
}

// TEMPORARY: debug helper to use scalar prep for 16bpc
fn prep_8tap_16bpc_scalar_fallback(
    _token: Desktop64,
    _tmp: &mut [i16],
    _src: &[u16],
    _src_base: usize,
    _src_stride: isize,
    _w: i32,
    _h: i32,
    _mx: i32,
    _my: i32,
    _h_filter: Rav1dFilterMode,
    _v_filter: Rav1dFilterMode,
) {
    // This function intentionally does nothing - we return false from dispatch to use scalar
}

/// Safe arcane entry point for prep_8tap 8bpc dispatch.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn prep_8tap_8bpc_dispatch_inner(
    token: Desktop64,
    tmp: &mut [i16],
    src: &[u8],
    src_base: usize,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    h_filter: Rav1dFilterMode,
    v_filter: Rav1dFilterMode,
) {
    prep_8tap_8bpc_avx2_impl_inner(
        token, tmp, src, src_base, src_stride, w, h, mx, my, h_filter, v_filter,
    );
}

/// Safe arcane entry point for prep_bilin 8bpc dispatch.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn prep_bilin_8bpc_dispatch_inner(
    token: Desktop64,
    tmp: &mut [i16],
    src: &[u8],
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
) {
    prep_bilin_8bpc_avx2_impl_inner(token, tmp, src, src_stride, w, h, mx, my);
}

/// Safe arcane entry point for prep_8tap 16bpc dispatch.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn prep_8tap_16bpc_dispatch_inner(
    token: Desktop64,
    tmp: &mut [i16],
    src: &[u16],
    src_base: usize,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    h_filter: Rav1dFilterMode,
    v_filter: Rav1dFilterMode,
) {
    prep_8tap_16bpc_avx2_impl_inner(
        token,
        tmp,
        src,
        src_base,
        src_stride,
        w,
        h,
        mx,
        my,
        bitdepth_max,
        h_filter,
        v_filter,
    );
}

#[cfg(target_arch = "x86_64")]
pub fn mc_put_dispatch<BD: BitDepth>(
    filter: Filter2d,
    dst: PicOffset,
    src: PicOffset,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bd: BD,
) -> bool {
    use crate::include::common::bitdepth::BPC;
    use zerocopy::AsBytes;
    let Some(token) = crate::src::cpu::summon_avx2() else {
        return false;
    };

    // When src and dst are from the same picture component (self-referencing frames),
    // we can't hold both a mutable dst guard and immutable src guard simultaneously.
    // Fall through to scalar for this rare case.
    if dst.data.ref_eq(src.data) {
        return false;
    }

    let dst_stride = dst.stride();
    let src_stride = src.stride();
    let pixel_size = std::mem::size_of::<BD::Pixel>();
    match BD::BPC {
        BPC::BPC8 => {
            let (mut dst_guard, dst_base) = dst.full_guard_mut::<BD>();
            let dst_bytes = &mut dst_guard.as_bytes_mut()[dst_base * pixel_size..];
            let (src_guard, src_base) = src.full_guard::<BD>();
            match filter {
                Filter2d::Bilinear => {
                    // Bilinear only accesses current + next row, no negative offsets
                    let src_bytes = &src_guard.as_bytes()[src_base * pixel_size..];
                    put_bilin_8bpc_dispatch_inner(
                        token, dst_bytes, dst_stride, src_bytes, src_stride, w, h, mx, my,
                    );
                }
                _ => {
                    // 8-tap needs rows above the block; pass full buffer + base offset
                    let src_bytes = src_guard.as_bytes();
                    let (h_filter, v_filter) = filter.hv();
                    put_8tap_8bpc_dispatch_inner(
                        token,
                        dst_bytes,
                        dst_stride,
                        src_bytes,
                        src_base * pixel_size,
                        src_stride,
                        w,
                        h,
                        mx,
                        my,
                        h_filter,
                        v_filter,
                    );
                }
            }
        }
        BPC::BPC16 => {
            // TEMPORARY: debug - force scalar for put 16bpc
            // return false;
            let (mut dst_guard, dst_base) = dst.full_guard_mut::<BD>();
            let dst_bytes = &mut dst_guard.as_bytes_mut()[dst_base * pixel_size..];
            let dst_u16: &mut [u16] = zerocopy::Ref::<_, [u16]>::new_slice(dst_bytes)
                .expect("u16 alignment")
                .into_mut_slice();
            let (src_guard, src_base) = src.full_guard::<BD>();
            let bd_c = bd.into_c();
            match filter {
                Filter2d::Bilinear => {
                    // 16bpc bilinear not yet converted to safe slices; fall through to scalar
                    return false;
                }
                _ => {
                    // 8-tap needs rows above the block; pass full buffer + base offset
                    let src_all_bytes = src_guard.as_bytes();
                    let src_u16: &[u16] = zerocopy::Ref::<_, [u16]>::new_slice(src_all_bytes)
                        .expect("u16 alignment")
                        .into_slice();
                    let (h_filter, v_filter) = filter.hv();
                    put_8tap_16bpc_dispatch_inner(
                        token, dst_u16, dst_stride, src_u16, src_base, src_stride, w, h, mx, my,
                        bd_c, h_filter, v_filter,
                    );
                }
            }
        }
    }
    true
}

#[cfg(target_arch = "x86_64")]
pub fn mct_prep_dispatch<BD: BitDepth>(
    filter: Filter2d,
    tmp: &mut [i16],
    src: PicOffset,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bd: BD,
) -> bool {
    use crate::include::common::bitdepth::BPC;
    use zerocopy::AsBytes;
    let Some(token) = crate::src::cpu::summon_avx2() else {
        return false;
    };
    let src_stride = src.stride();
    let pixel_size = std::mem::size_of::<BD::Pixel>();
    match BD::BPC {
        BPC::BPC8 => {
            let (src_guard, src_base) = src.full_guard::<BD>();
            match filter {
                Filter2d::Bilinear => {
                    // Bilinear only accesses current + next row, no negative offsets
                    let src_bytes = &src_guard.as_bytes()[src_base * pixel_size..];
                    prep_bilin_8bpc_dispatch_inner(token, tmp, src_bytes, src_stride, w, h, mx, my);
                }
                _ => {
                    // 8-tap needs rows above the block; pass full buffer + base offset
                    let src_bytes = src_guard.as_bytes();
                    let (h_filter, v_filter) = filter.hv();
                    prep_8tap_8bpc_dispatch_inner(
                        token,
                        tmp,
                        src_bytes,
                        src_base * pixel_size,
                        src_stride,
                        w,
                        h,
                        mx,
                        my,
                        h_filter,
                        v_filter,
                    );
                }
            }
        }
        BPC::BPC16 => {
            let (src_guard, src_base) = src.full_guard::<BD>();
            let bd_c = bd.into_c();
            match filter {
                Filter2d::Bilinear => {
                    // 16bpc bilinear not yet converted to safe slices; fall through to scalar
                    return false;
                }
                _ => {
                    // 8-tap needs rows above the block; pass full buffer + base offset
                    let src_all_bytes = src_guard.as_bytes();
                    let src_u16: &[u16] = zerocopy::Ref::<_, [u16]>::new_slice(src_all_bytes)
                        .expect("u16 alignment")
                        .into_slice();
                    let (h_filter, v_filter) = filter.hv();
                    prep_8tap_16bpc_dispatch_inner(
                        token, tmp, src_u16, src_base, src_stride, w, h, mx, my, bd_c, h_filter,
                        v_filter,
                    );
                }
            }
        }
    }
    let _ = bd;
    true
}

/// No SIMD for scaled variants on x86_64.
#[cfg(target_arch = "x86_64")]
pub fn mc_scaled_dispatch<BD: BitDepth>(
    _filter: Filter2d,
    _dst: PicOffset,
    _src: PicOffset,
    _w: i32,
    _h: i32,
    _mx: i32,
    _my: i32,
    _dx: i32,
    _dy: i32,
    _bd: BD,
) -> bool {
    false
}

/// No SIMD for scaled variants on x86_64.
#[cfg(target_arch = "x86_64")]
pub fn mct_scaled_dispatch<BD: BitDepth>(
    _filter: Filter2d,
    _tmp: &mut [i16],
    _src: PicOffset,
    _w: i32,
    _h: i32,
    _mx: i32,
    _my: i32,
    _dx: i32,
    _dy: i32,
    _bd: BD,
) -> bool {
    false
}

/// No SIMD for warp on x86_64.
#[cfg(target_arch = "x86_64")]
pub fn warp8x8_dispatch<BD: BitDepth>(
    _dst: PicOffset,
    _src: PicOffset,
    _abcd: &[i16; 4],
    _mx: i32,
    _my: i32,
    _bd: BD,
) -> bool {
    false
}

/// No SIMD for warp on x86_64.
#[cfg(target_arch = "x86_64")]
pub fn warp8x8t_dispatch<BD: BitDepth>(
    _tmp: &mut [i16],
    _tmp_stride: usize,
    _src: PicOffset,
    _abcd: &[i16; 4],
    _mx: i32,
    _my: i32,
    _bd: BD,
) -> bool {
    false
}

/// No SIMD for emu_edge on x86_64.
#[cfg(target_arch = "x86_64")]
pub fn emu_edge_dispatch<BD: BitDepth>(
    _bw: isize,
    _bh: isize,
    _iw: isize,
    _ih: isize,
    _x: isize,
    _y: isize,
    _dst: &mut [BD::Pixel; crate::src::internal::EMU_EDGE_LEN],
    _dst_pxstride: usize,
    _src: &crate::include::dav1d::picture::Rav1dPictureDataComponent,
) -> bool {
    false
}

/// No SIMD for resize on x86_64.
#[cfg(target_arch = "x86_64")]
pub fn resize_dispatch<BD: BitDepth>(
    _dst: crate::src::with_offset::WithOffset<
        crate::src::pic_or_buf::PicOrBuf<crate::src::align::AlignedVec64<u8>>,
    >,
    _src: PicOffset,
    _dst_w: usize,
    _h: usize,
    _src_w: usize,
    _dx: i32,
    _mx: i32,
    _bd: BD,
) -> bool {
    false
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
                for i in 0..(w * h) as usize {
                    tmp1[i] = v1;
                    tmp2[i] = v2;
                }
                dst_avx2.fill(0);
                dst_scalar.fill(0);

                let token = crate::src::cpu::summon_avx2().unwrap();
                avg_8bpc_avx2_safe(token, &mut dst_scalar, w as usize, &tmp1, &tmp2, w, h);

                avg_8bpc_avx2_safe(token, &mut dst_avx2, w as usize, &tmp1, &tmp2, w, h);

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

        let mut dst_a = vec![0u8; (w * h) as usize];
        let mut dst_b = vec![0u8; (w * h) as usize];

        let token = crate::src::cpu::summon_avx2().unwrap();
        avg_8bpc_avx2_safe(token, &mut dst_a, w as usize, &tmp1, &tmp2, w, h);

        avg_8bpc_avx2_safe(token, &mut dst_b, w as usize, &tmp1, &tmp2, w, h);

        assert_eq!(dst_a, dst_b, "Results differ for varying data");
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
        let mut dst_a = vec![0u8; (w * h) as usize];
        let mut dst_b = vec![0u8; (w * h) as usize];

        let token = crate::src::cpu::summon_avx2().unwrap();
        for &weight in &test_weights {
            for &v1 in &test_values {
                for &v2 in &test_values {
                    for i in 0..(w * h) as usize {
                        tmp1[i] = v1;
                        tmp2[i] = v2;
                    }
                    dst_a.fill(0);
                    dst_b.fill(0);

                    w_avg_8bpc_avx2_safe(token, &mut dst_a, w as usize, &tmp1, &tmp2, w, h, weight);

                    w_avg_8bpc_avx2_safe(token, &mut dst_b, w as usize, &tmp1, &tmp2, w, h, weight);

                    assert_eq!(
                        dst_a,
                        dst_b,
                        "Mismatch for weight={}, v1={}, v2={}: a={:?} b={:?}",
                        weight,
                        v1,
                        v2,
                        &dst_a[..8],
                        &dst_b[..8]
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
        let mut dst_a = vec![0u8; (w * h) as usize];
        let mut dst_b = vec![0u8; (w * h) as usize];

        let token = crate::src::cpu::summon_avx2().unwrap();
        for &m in &test_masks {
            for &v1 in &test_values {
                for &v2 in &test_values {
                    for i in 0..(w * h) as usize {
                        tmp1[i] = v1;
                        tmp2[i] = v2;
                        mask[i] = m;
                    }
                    dst_a.fill(0);
                    dst_b.fill(0);

                    mask_8bpc_avx2_safe(token, &mut dst_a, w as usize, &tmp1, &tmp2, w, h, &mask);

                    mask_8bpc_avx2_safe(token, &mut dst_b, w as usize, &tmp1, &tmp2, w, h, &mask);

                    assert_eq!(
                        dst_a,
                        dst_b,
                        "Mismatch for mask={}, v1={}, v2={}: a={:?} b={:?}",
                        m,
                        v1,
                        v2,
                        &dst_a[..8],
                        &dst_b[..8]
                    );
                }
            }
        }
    }
}
