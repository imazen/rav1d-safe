//! Safe SIMD implementations of motion compensation functions for ARM NEON
#![allow(deprecated)] // FFI wrappers need to forge tokens
//!
//! These use archmage tokens to safely invoke NEON intrinsics.
//! The extern "C" wrappers are used for FFI compatibility with rav1d's dispatch system.

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
use archmage::{arcane, Arm64, SimdToken};

#[cfg(target_arch = "aarch64")]
use safe_unaligned_simd::aarch64 as safe_simd;

use crate::include::common::bitdepth::BitDepth;
use crate::include::common::bitdepth::DynPixel;
use crate::include::dav1d::headers::Rav1dFilterMode;
#[cfg(target_arch = "aarch64")]
use crate::include::dav1d::headers::Rav1dPixelLayoutSubSampled;
use crate::include::dav1d::picture::PicOffset;
use crate::src::ffi_safe::FFISafe;
use crate::src::internal::COMPINTER_LEN;
use crate::src::internal::SCRATCH_INTER_INTRA_BUF_LEN;
use crate::src::internal::SCRATCH_LAP_LEN;
#[cfg(target_arch = "aarch64")]
use crate::src::internal::SEG_MASK_LEN;
use crate::src::levels::Filter2d;
use crate::src::strided::Strided as _;
use crate::src::tables::dav1d_mc_subpel_filters;

// ============================================================================
// AVG - Average two buffers
// ============================================================================

/// Inner AVG implementation using archmage token
#[cfg(target_arch = "aarch64")]
#[arcane]
fn avg_8bpc_inner(
    _token: Arm64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
) {
    for row in 0..h {
        let tmp1_row = &tmp1[row * w..][..w];
        let tmp2_row = &tmp2[row * w..][..w];
        let dst_row = &mut dst[row * dst_stride..][..w];

        let mut col = 0;

        // Process 16 pixels at a time
        while col + 16 <= w {
            // Load 16 i16 values using safe_simd
            let t1_lo = safe_simd::vld1q_s16(tmp1_row[col..][..8].try_into().unwrap());
            let t1_hi = safe_simd::vld1q_s16(tmp1_row[col + 8..][..8].try_into().unwrap());
            let t2_lo = safe_simd::vld1q_s16(tmp2_row[col..][..8].try_into().unwrap());
            let t2_hi = safe_simd::vld1q_s16(tmp2_row[col + 8..][..8].try_into().unwrap());

            // Add: tmp1 + tmp2 (safe in #[arcane])
            let sum_lo = vaddq_s16(t1_lo, t2_lo);
            let sum_hi = vaddq_s16(t1_hi, t2_hi);

            // vqrdmulhq_n_s16(a, 2048) = (2 * a * 2048 + 0x8000) >> 16
            // = (a * 4096 + 32768) >> 16 = (a * 1024 + 8192) >> 14
            // We want (a * 1024 + 16384) >> 15 = pmulhrsw equivalent
            let avg_lo = vqrdmulhq_n_s16(sum_lo, 2048);
            let avg_hi = vqrdmulhq_n_s16(sum_hi, 2048);

            // Pack to u8 with saturation (safe in #[arcane])
            let packed_lo = vqmovun_s16(avg_lo);
            let packed_hi = vqmovun_s16(avg_hi);
            let result = vcombine_u8(packed_lo, packed_hi);

            // Store using safe_simd
            let dst_arr: &mut [u8; 16] = (&mut dst_row[col..col + 16]).try_into().unwrap();
            safe_simd::vst1q_u8(dst_arr, result);
            col += 16;
        }

        // Process 8 pixels at a time
        while col + 8 <= w {
            let t1 = safe_simd::vld1q_s16(tmp1_row[col..][..8].try_into().unwrap());
            let t2 = safe_simd::vld1q_s16(tmp2_row[col..][..8].try_into().unwrap());
            let sum = vaddq_s16(t1, t2);
            let avg = vqrdmulhq_n_s16(sum, 2048);
            let packed = vqmovun_s16(avg);
            // Store 8 bytes using partial_simd
            let dst_arr: &mut [u8; 8] = (&mut dst_row[col..col + 8]).try_into().unwrap();
            safe_simd::vst1_u8(dst_arr, packed);
            col += 8;
        }

        // Scalar fallback
        while col < w {
            let sum = tmp1_row[col] as i32 + tmp2_row[col] as i32;
            let avg = ((sum * 1024 + 16384) >> 15).clamp(0, 255) as u8;
            dst_row[col] = avg;
            col += 1;
        }
    }
}

/// AVG operation for 8-bit pixels - extern "C" wrapper for dispatch
#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn avg_8bpc_neon(
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

    // SAFETY: This function is only called through dispatch when NEON is available
    // dst_ptr points to valid memory with proper alignment and size
    let (token, dst) = unsafe {
        let token = unsafe { Arm64::forge_token_dangerously() };
        let dst = std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h * dst_stride.unsigned_abs());
        (token, dst)
    };

    avg_8bpc_inner(
        token,
        dst,
        dst_stride as usize,
        tmp1.as_slice(),
        tmp2.as_slice(),
        w,
        h,
    );
}

/// Inner AVG implementation for 16bpc using archmage token
#[cfg(target_arch = "aarch64")]
#[arcane]
fn avg_16bpc_inner(
    _token: Arm64,
    dst: &mut [u16],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    bitdepth_max: i32,
) {
    let intermediate_bits = 4;

    for row in 0..h {
        let tmp1_row = &tmp1[row * w..][..w];
        let tmp2_row = &tmp2[row * w..][..w];
        let dst_row = &mut dst[row * dst_stride..][..w];

        let mut col = 0;

        // Process 8 pixels at a time
        while col + 8 <= w {
            let t1 = safe_simd::vld1q_s16(tmp1_row[col..][..8].try_into().unwrap());
            let t2 = safe_simd::vld1q_s16(tmp2_row[col..][..8].try_into().unwrap());

            // Widen to 32-bit
            let t1_lo = vmovl_s16(vget_low_s16(t1));
            let t1_hi = vmovl_s16(vget_high_s16(t1));
            let t2_lo = vmovl_s16(vget_low_s16(t2));
            let t2_hi = vmovl_s16(vget_high_s16(t2));

            // Add
            let sum_lo = vaddq_s32(t1_lo, t2_lo);
            let sum_hi = vaddq_s32(t1_hi, t2_hi);

            // Round and shift: (sum + (1 << intermediate_bits)) >> (intermediate_bits + 1)
            let rnd = vdupq_n_s32(1 << intermediate_bits);
            let sum_lo_rnd = vaddq_s32(sum_lo, rnd);
            let sum_hi_rnd = vaddq_s32(sum_hi, rnd);

            let avg_lo = vshrq_n_s32::<5>(sum_lo_rnd);
            let avg_hi = vshrq_n_s32::<5>(sum_hi_rnd);

            // Narrow to 16-bit
            let avg_narrow_lo = vqmovn_s32(avg_lo);
            let avg_narrow_hi = vqmovn_s32(avg_hi);
            let avg_16 = vcombine_s16(avg_narrow_lo, avg_narrow_hi);

            // Clamp
            let zero = vdupq_n_s16(0);
            let max = vdupq_n_s16(bitdepth_max as i16);
            let clamped = vmaxq_s16(vminq_s16(avg_16, max), zero);

            let dst_arr: &mut [u16; 8] = (&mut dst_row[col..col + 8]).try_into().unwrap();
            safe_simd::vst1q_u16(dst_arr, vreinterpretq_u16_s16(clamped));
            col += 8;
        }

        // Scalar fallback
        while col < w {
            let sum = tmp1_row[col] as i32 + tmp2_row[col] as i32;
            let rnd = 1 << intermediate_bits;
            let avg = ((sum + rnd) >> (intermediate_bits + 1)).clamp(0, bitdepth_max);
            dst_row[col] = avg as u16;
            col += 1;
        }
    }
}

/// AVG operation for 16-bit pixels - extern "C" wrapper
#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn avg_16bpc_neon(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    bitdepth_max: i32,
    _dst: *const FFISafe<PicOffset>,
) {
    let w = w as usize;
    let h = h as usize;
    let dst_stride_u16 = (dst_stride / 2) as usize;

    // SAFETY: This function is only called through dispatch when NEON is available
    // dst_ptr points to valid memory with proper alignment and size
    let (token, dst) = unsafe {
        let token = unsafe { Arm64::forge_token_dangerously() };
        let dst = std::slice::from_raw_parts_mut(dst_ptr as *mut u16, h * dst_stride_u16);
        (token, dst)
    };

    avg_16bpc_inner(
        token,
        dst,
        dst_stride_u16,
        tmp1.as_slice(),
        tmp2.as_slice(),
        w,
        h,
        bitdepth_max,
    );
}

// ============================================================================
// W_AVG - Weighted average
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[arcane]
fn w_avg_8bpc_inner(
    _token: Arm64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    weight: i32,
) {
    for row in 0..h {
        let tmp1_row = &tmp1[row * w..][..w];
        let tmp2_row = &tmp2[row * w..][..w];
        let dst_row = &mut dst[row * dst_stride..][..w];

        let mut col = 0;

        // Process 8 pixels at a time
        while col + 8 <= w {
            let t1 = safe_simd::vld1q_s16(tmp1_row[col..][..8].try_into().unwrap());
            let t2 = safe_simd::vld1q_s16(tmp2_row[col..][..8].try_into().unwrap());

            // diff = tmp1 - tmp2
            let diff = vsubq_s16(t1, t2);

            // Widen for multiply
            let diff_lo = vmovl_s16(vget_low_s16(diff));
            let diff_hi = vmovl_s16(vget_high_s16(diff));
            let weight_vec = vdupq_n_s32(weight);

            let weighted_lo = vmulq_s32(diff_lo, weight_vec);
            let weighted_hi = vmulq_s32(diff_hi, weight_vec);

            // (weighted + 8) >> 4
            let rnd = vdupq_n_s32(8);
            let shifted_lo = vshrq_n_s32::<4>(vaddq_s32(weighted_lo, rnd));
            let shifted_hi = vshrq_n_s32::<4>(vaddq_s32(weighted_hi, rnd));

            // Narrow back to 16-bit
            let shifted_16 = vcombine_s16(vmovn_s32(shifted_lo), vmovn_s32(shifted_hi));

            // Add tmp2 and apply final scaling
            let sum = vaddq_s16(shifted_16, t2);
            let scaled = vqrdmulhq_n_s16(sum, 2048);
            let packed = vqmovun_s16(scaled);

            let dst_arr: &mut [u8; 8] = (&mut dst_row[col..col + 8]).try_into().unwrap();
            safe_simd::vst1_u8(dst_arr, packed);
            col += 8;
        }

        // Scalar fallback
        while col < w {
            let diff = tmp1_row[col] as i32 - tmp2_row[col] as i32;
            let weighted = ((diff * weight + 8) >> 4) + tmp2_row[col] as i32;
            let scaled = ((weighted * 1024 + 16384) >> 15).clamp(0, 255);
            dst_row[col] = scaled as u8;
            col += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn w_avg_8bpc_neon(
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

    // SAFETY: This function is only called through dispatch when NEON is available
    let (token, dst) = unsafe {
        let token = unsafe { Arm64::forge_token_dangerously() };
        let dst = std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h * dst_stride.unsigned_abs());
        (token, dst)
    };

    w_avg_8bpc_inner(
        token,
        dst,
        dst_stride as usize,
        tmp1.as_slice(),
        tmp2.as_slice(),
        w,
        h,
        weight,
    );
}

#[cfg(target_arch = "aarch64")]
#[arcane]
fn w_avg_16bpc_inner(
    _token: Arm64,
    dst: &mut [u16],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    weight: i32,
    bitdepth_max: i32,
) {
    let intermediate_bits = 4;

    for row in 0..h {
        let tmp1_row = &tmp1[row * w..][..w];
        let tmp2_row = &tmp2[row * w..][..w];
        let dst_row = &mut dst[row * dst_stride..][..w];

        let mut col = 0;

        // Process 4 pixels at a time
        while col + 4 <= w {
            let t1_16 = safe_simd::vld1_s16(tmp1_row[col..][..4].try_into().unwrap());
            let t2_16 = safe_simd::vld1_s16(tmp2_row[col..][..4].try_into().unwrap());
            let t1 = vmovl_s16(t1_16);
            let t2 = vmovl_s16(t2_16);

            let diff = vsubq_s32(t1, t2);
            let weight_vec = vdupq_n_s32(weight);
            let weighted = vmulq_s32(diff, weight_vec);

            let rnd = vdupq_n_s32(8);
            let shifted = vshrq_n_s32::<4>(vaddq_s32(weighted, rnd));
            let sum = vaddq_s32(shifted, t2);

            let rnd2 = vdupq_n_s32(1 << intermediate_bits);
            let result = vshrq_n_s32::<5>(vaddq_s32(sum, rnd2));

            let zero = vdupq_n_s32(0);
            let max = vdupq_n_s32(bitdepth_max);
            let clamped = vmaxq_s32(vminq_s32(result, max), zero);

            let narrow = vmovn_s32(clamped);
            let dst_arr: &mut [u16; 4] = (&mut dst_row[col..col + 4]).try_into().unwrap();
            safe_simd::vst1_u16(dst_arr, vreinterpret_u16_s16(narrow));
            col += 4;
        }

        // Scalar fallback
        while col < w {
            let diff = tmp1_row[col] as i32 - tmp2_row[col] as i32;
            let weighted = ((diff * weight + 8) >> 4) + tmp2_row[col] as i32;
            let rnd = 1 << intermediate_bits;
            let result = ((weighted + rnd) >> (intermediate_bits + 1)).clamp(0, bitdepth_max);
            dst_row[col] = result as u16;
            col += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn w_avg_16bpc_neon(
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
    let w = w as usize;
    let h = h as usize;
    let dst_stride_u16 = (dst_stride / 2) as usize;

    // SAFETY: This function is only called through dispatch when NEON is available
    let (token, dst) = unsafe {
        let token = unsafe { Arm64::forge_token_dangerously() };
        let dst = std::slice::from_raw_parts_mut(dst_ptr as *mut u16, h * dst_stride_u16);
        (token, dst)
    };

    w_avg_16bpc_inner(
        token,
        dst,
        dst_stride_u16,
        tmp1.as_slice(),
        tmp2.as_slice(),
        w,
        h,
        weight,
        bitdepth_max,
    );
}

// ============================================================================
// MASK - Per-pixel masked blend
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[arcane]
fn mask_8bpc_inner(
    _token: Arm64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    mask: &[u8],
) {
    for row in 0..h {
        let tmp1_row = &tmp1[row * w..][..w];
        let tmp2_row = &tmp2[row * w..][..w];
        let mask_row = &mask[row * w..][..w];
        let dst_row = &mut dst[row * dst_stride..][..w];

        let mut col = 0;

        // Process 8 pixels at a time
        while col + 8 <= w {
            let t1 = safe_simd::vld1q_s16(tmp1_row[col..][..8].try_into().unwrap());
            let t2 = safe_simd::vld1q_s16(tmp2_row[col..][..8].try_into().unwrap());
            let m = safe_simd::vld1_u8(mask_row[col..][..8].try_into().unwrap());

            // Widen mask to 16-bit
            let m16 = vreinterpretq_s16_u16(vmovl_u8(m));

            // diff = tmp1 - tmp2
            let diff = vsubq_s16(t1, t2);

            // Widen for multiply
            let diff_lo = vmovl_s16(vget_low_s16(diff));
            let diff_hi = vmovl_s16(vget_high_s16(diff));
            let m_lo = vmovl_s16(vget_low_s16(m16));
            let m_hi = vmovl_s16(vget_high_s16(m16));

            // weighted = diff * mask
            let weighted_lo = vmulq_s32(diff_lo, m_lo);
            let weighted_hi = vmulq_s32(diff_hi, m_hi);

            // (weighted + 32) >> 6
            let rnd = vdupq_n_s32(32);
            let shifted_lo = vshrq_n_s32::<6>(vaddq_s32(weighted_lo, rnd));
            let shifted_hi = vshrq_n_s32::<6>(vaddq_s32(weighted_hi, rnd));

            // Narrow and add tmp2
            let shifted_16 = vcombine_s16(vmovn_s32(shifted_lo), vmovn_s32(shifted_hi));
            let sum = vaddq_s16(shifted_16, t2);
            let scaled = vqrdmulhq_n_s16(sum, 2048);
            let packed = vqmovun_s16(scaled);

            let dst_arr: &mut [u8; 8] = (&mut dst_row[col..col + 8]).try_into().unwrap();
            safe_simd::vst1_u8(dst_arr, packed);
            col += 8;
        }

        // Scalar fallback
        while col < w {
            let diff = tmp1_row[col] as i32 - tmp2_row[col] as i32;
            let m = mask_row[col] as i32;
            let weighted = ((diff * m + 32) >> 6) + tmp2_row[col] as i32;
            let scaled = ((weighted * 1024 + 16384) >> 15).clamp(0, 255);
            dst_row[col] = scaled as u8;
            col += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn mask_8bpc_neon(
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

    // SAFETY: This function is only called through dispatch when NEON is available
    let (token, dst, mask) = unsafe {
        let token = unsafe { Arm64::forge_token_dangerously() };
        let dst = std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h * dst_stride.unsigned_abs());
        let mask = std::slice::from_raw_parts(mask_ptr, w * h);
        (token, dst, mask)
    };

    mask_8bpc_inner(
        token,
        dst,
        dst_stride as usize,
        tmp1.as_slice(),
        tmp2.as_slice(),
        w,
        h,
        mask,
    );
}

#[cfg(target_arch = "aarch64")]
#[arcane]
fn mask_16bpc_inner(
    _token: Arm64,
    dst: &mut [u16],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    mask: &[u8],
    bitdepth_max: i32,
) {
    let intermediate_bits = 4;

    for row in 0..h {
        let tmp1_row = &tmp1[row * w..][..w];
        let tmp2_row = &tmp2[row * w..][..w];
        let mask_row = &mask[row * w..][..w];
        let dst_row = &mut dst[row * dst_stride..][..w];

        let mut col = 0;

        // Process 4 pixels at a time
        while col + 4 <= w {
            let t1_16 = safe_simd::vld1_s16(tmp1_row[col..][..4].try_into().unwrap());
            let t2_16 = safe_simd::vld1_s16(tmp2_row[col..][..4].try_into().unwrap());
            let t1 = vmovl_s16(t1_16);
            let t2 = vmovl_s16(t2_16);

            // Load 4 mask bytes
            let m_bytes: [u8; 8] = [
                mask_row[col],
                mask_row[col + 1],
                mask_row[col + 2],
                mask_row[col + 3],
                0,
                0,
                0,
                0,
            ];
            let m8 = safe_simd::vld1_u8(&m_bytes);
            let m16 = vmovl_u8(m8);
            let m32 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(m16)));

            let diff = vsubq_s32(t1, t2);
            let weighted = vmulq_s32(diff, m32);

            let rnd = vdupq_n_s32(32);
            let shifted = vshrq_n_s32::<6>(vaddq_s32(weighted, rnd));
            let sum = vaddq_s32(shifted, t2);

            let rnd2 = vdupq_n_s32(1 << intermediate_bits);
            let result = vshrq_n_s32::<5>(vaddq_s32(sum, rnd2));

            let zero = vdupq_n_s32(0);
            let max = vdupq_n_s32(bitdepth_max);
            let clamped = vmaxq_s32(vminq_s32(result, max), zero);

            let narrow = vmovn_s32(clamped);
            let dst_arr: &mut [u16; 4] = (&mut dst_row[col..col + 4]).try_into().unwrap();
            safe_simd::vst1_u16(dst_arr, vreinterpret_u16_s16(narrow));
            col += 4;
        }

        // Scalar fallback
        while col < w {
            let diff = tmp1_row[col] as i32 - tmp2_row[col] as i32;
            let m = mask_row[col] as i32;
            let weighted = ((diff * m + 32) >> 6) + tmp2_row[col] as i32;
            let rnd = 1 << intermediate_bits;
            let result = ((weighted + rnd) >> (intermediate_bits + 1)).clamp(0, bitdepth_max);
            dst_row[col] = result as u16;
            col += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn mask_16bpc_neon(
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
    let w = w as usize;
    let h = h as usize;
    let dst_stride_u16 = (dst_stride / 2) as usize;

    // SAFETY: This function is only called through dispatch when NEON is available
    let (token, dst, mask) = unsafe {
        let token = unsafe { Arm64::forge_token_dangerously() };
        let dst = std::slice::from_raw_parts_mut(dst_ptr as *mut u16, h * dst_stride_u16);
        let mask = std::slice::from_raw_parts(mask_ptr, w * h);
        (token, dst, mask)
    };

    mask_16bpc_inner(
        token,
        dst,
        dst_stride_u16,
        tmp1.as_slice(),
        tmp2.as_slice(),
        w,
        h,
        mask,
        bitdepth_max,
    );
}

// ============================================================================
// BLEND - Simple pixel blend
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[arcane]
fn blend_8bpc_inner(
    _token: Arm64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp: &[i16],
    w: usize,
    h: usize,
    mask: &[u8],
) {
    for row in 0..h {
        let tmp_row = &tmp[row * w..][..w];
        let mask_row = &mask[row * w..][..w];
        let dst_row = &mut dst[row * dst_stride..][..w];

        let mut col = 0;

        // Process 8 pixels at a time
        while col + 8 <= w {
            let d = safe_simd::vld1_u8(dst_row[col..][..8].try_into().unwrap());
            let d16 = vreinterpretq_s16_u16(vmovl_u8(d));
            let t = safe_simd::vld1q_s16(tmp_row[col..][..8].try_into().unwrap());
            let m = safe_simd::vld1_u8(mask_row[col..][..8].try_into().unwrap());
            let m16 = vreinterpretq_s16_u16(vmovl_u8(m));

            // diff = tmp - (dst << 4)
            let d_scaled = vshlq_n_s16::<4>(d16);
            let diff = vsubq_s16(t, d_scaled);

            // Widen for multiply
            let diff_lo = vmovl_s16(vget_low_s16(diff));
            let diff_hi = vmovl_s16(vget_high_s16(diff));
            let m_lo = vmovl_s16(vget_low_s16(m16));
            let m_hi = vmovl_s16(vget_high_s16(m16));

            // weighted = diff * mask
            let weighted_lo = vmulq_s32(diff_lo, m_lo);
            let weighted_hi = vmulq_s32(diff_hi, m_hi);

            // (weighted + 32) >> 6
            let rnd = vdupq_n_s32(32);
            let shifted_lo = vshrq_n_s32::<6>(vaddq_s32(weighted_lo, rnd));
            let shifted_hi = vshrq_n_s32::<6>(vaddq_s32(weighted_hi, rnd));

            // Narrow to 16-bit
            let shifted_16 = vcombine_s16(vmovn_s32(shifted_lo), vmovn_s32(shifted_hi));

            // Add d_scaled and shift right by 4
            let sum = vaddq_s16(shifted_16, d_scaled);
            let result = vshrq_n_s16::<4>(vaddq_s16(sum, vdupq_n_s16(8)));

            // Pack to u8
            let packed = vqmovun_s16(result);
            let dst_arr: &mut [u8; 8] = (&mut dst_row[col..col + 8]).try_into().unwrap();
            safe_simd::vst1_u8(dst_arr, packed);
            col += 8;
        }

        // Scalar fallback
        while col < w {
            let d = dst_row[col] as i32;
            let t = tmp_row[col] as i32;
            let m = mask_row[col] as i32;
            let d_scaled = d << 4;
            let diff = t - d_scaled;
            let weighted = (diff * m + 32) >> 6;
            let result = ((d_scaled + weighted + 8) >> 4).clamp(0, 255);
            dst_row[col] = result as u8;
            col += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn blend_8bpc_neon(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp: *const [DynPixel; SCRATCH_INTER_INTRA_BUF_LEN],
    w: i32,
    h: i32,
    mask_ptr: *const u8,
    _dst: *const FFISafe<PicOffset>,
) {
    let w = w as usize;
    let h = h as usize;

    // SAFETY: Pointers are valid and properly aligned
    let (dst, tmp, mask) = unsafe {
        let dst = std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h * dst_stride.unsigned_abs());
        let tmp = std::slice::from_raw_parts(tmp as *const u8, w * h);
        let mask = std::slice::from_raw_parts(mask_ptr, w * h);
        (dst, tmp, mask)
    };

    // blend formula: (dst * (64-m) + tmp * m + 32) >> 6
    for row in 0..h {
        let dst_row = &mut dst[row * dst_stride.unsigned_abs()..][..w];
        let tmp_row = &tmp[row * w..][..w];
        let mask_row = &mask[row * w..][..w];
        for col in 0..w {
            let d = dst_row[col] as u32;
            let t = tmp_row[col] as u32;
            let m = mask_row[col] as u32;
            dst_row[col] = ((d * (64 - m) + t * m + 32) >> 6) as u8;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[arcane]
fn blend_16bpc_inner(
    _token: Arm64,
    dst: &mut [u16],
    dst_stride: usize,
    tmp: &[i16],
    w: usize,
    h: usize,
    mask: &[u8],
) {
    for row in 0..h {
        let tmp_row = &tmp[row * w..][..w];
        let mask_row = &mask[row * w..][..w];
        let dst_row = &mut dst[row * dst_stride..][..w];

        let mut col = 0;

        // Process 4 pixels at a time
        while col + 4 <= w {
            let d_u16 = safe_simd::vld1_u16(dst_row[col..][..4].try_into().unwrap());
            let d = vreinterpretq_s32_u32(vmovl_u16(d_u16));
            let t_16 = safe_simd::vld1_s16(tmp_row[col..][..4].try_into().unwrap());
            let t = vmovl_s16(t_16);

            // Load 4 mask bytes
            let m_bytes: [u8; 8] = [
                mask_row[col],
                mask_row[col + 1],
                mask_row[col + 2],
                mask_row[col + 3],
                0,
                0,
                0,
                0,
            ];
            let m8 = safe_simd::vld1_u8(&m_bytes);
            let m16 = vmovl_u8(m8);
            let m = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(m16)));

            // diff = tmp - dst
            let diff = vsubq_s32(t, d);
            let weighted = vmulq_s32(diff, m);

            // (weighted + 32) >> 6
            let rnd = vdupq_n_s32(32);
            let shifted = vshrq_n_s32::<6>(vaddq_s32(weighted, rnd));
            let result = vaddq_s32(shifted, d);

            // Clamp
            let zero = vdupq_n_s32(0);
            let max = vdupq_n_s32(65535);
            let clamped = vmaxq_s32(vminq_s32(result, max), zero);

            let narrow = vmovn_u32(vreinterpretq_u32_s32(clamped));
            let dst_arr: &mut [u16; 4] = (&mut dst_row[col..col + 4]).try_into().unwrap();
            safe_simd::vst1_u16(dst_arr, narrow);
            col += 4;
        }

        // Scalar fallback
        while col < w {
            let d = dst_row[col] as i32;
            let t = tmp_row[col] as i32;
            let m = mask_row[col] as i32;
            let diff = t - d;
            let weighted = (diff * m + 32) >> 6;
            let result = (d + weighted).clamp(0, 65535);
            dst_row[col] = result as u16;
            col += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn blend_16bpc_neon(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp: *const [DynPixel; SCRATCH_INTER_INTRA_BUF_LEN],
    w: i32,
    h: i32,
    mask_ptr: *const u8,
    _dst: *const FFISafe<PicOffset>,
) {
    let w = w as usize;
    let h = h as usize;
    let dst_stride_u16 = (dst_stride / 2) as usize;

    // SAFETY: Pointers are valid and properly aligned
    let (dst, tmp, mask) = unsafe {
        let dst = std::slice::from_raw_parts_mut(dst_ptr as *mut u16, h * dst_stride_u16);
        let tmp = std::slice::from_raw_parts(tmp as *const u16, w * h);
        let mask = std::slice::from_raw_parts(mask_ptr, w * h);
        (dst, tmp, mask)
    };

    // blend formula: (dst * (64-m) + tmp * m + 32) >> 6
    for row in 0..h {
        let dst_row = &mut dst[row * dst_stride_u16..][..w];
        let tmp_row = &tmp[row * w..][..w];
        let mask_row = &mask[row * w..][..w];
        for col in 0..w {
            let d = dst_row[col] as u32;
            let t = tmp_row[col] as u32;
            let m = mask_row[col] as u32;
            dst_row[col] = ((d * (64 - m) + t * m + 32) >> 6) as u16;
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    #[test]
    fn test_arm_token_available() {
        #[cfg(target_arch = "aarch64")]
        {
            use archmage::{Arm64, SimdToken};
            // NEON is always available on aarch64
            assert!(Arm64::summon().is_some());
        }
    }
}

// ============================================================================
// BLEND_V - Vertical OBMC blend
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[arcane]
fn blend_v_8bpc_inner(
    _token: Arm64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp: &[i16],
    w: usize,
    h: usize,
    obmc_masks: &[u8],
) {
    for row in 0..h {
        let tmp_row = &tmp[row * w..][..w];
        let dst_row = &mut dst[row * dst_stride..][..w];
        let mask = obmc_masks[row];

        let mut col = 0;

        // Process 8 pixels at a time
        while col + 8 <= w {
            let d = safe_simd::vld1_u8(dst_row[col..][..8].try_into().unwrap());
            let d16 = vreinterpretq_s16_u16(vmovl_u8(d));
            let t = safe_simd::vld1q_s16(tmp_row[col..][..8].try_into().unwrap());
            let m16 = vdupq_n_s16(mask as i16);

            // diff = tmp - (dst << 4)
            let d_scaled = vshlq_n_s16::<4>(d16);
            let diff = vsubq_s16(t, d_scaled);

            // Widen for multiply
            let diff_lo = vmovl_s16(vget_low_s16(diff));
            let diff_hi = vmovl_s16(vget_high_s16(diff));
            let m_lo = vmovl_s16(vget_low_s16(m16));
            let m_hi = vmovl_s16(vget_high_s16(m16));

            // weighted = diff * mask
            let weighted_lo = vmulq_s32(diff_lo, m_lo);
            let weighted_hi = vmulq_s32(diff_hi, m_hi);

            // (weighted + 32) >> 6
            let rnd = vdupq_n_s32(32);
            let shifted_lo = vshrq_n_s32::<6>(vaddq_s32(weighted_lo, rnd));
            let shifted_hi = vshrq_n_s32::<6>(vaddq_s32(weighted_hi, rnd));

            // Narrow to 16-bit
            let shifted_16 = vcombine_s16(vmovn_s32(shifted_lo), vmovn_s32(shifted_hi));

            // Add d_scaled and shift right by 4
            let sum = vaddq_s16(shifted_16, d_scaled);
            let result = vshrq_n_s16::<4>(vaddq_s16(sum, vdupq_n_s16(8)));

            // Pack to u8
            let packed = vqmovun_s16(result);
            let dst_arr: &mut [u8; 8] = (&mut dst_row[col..col + 8]).try_into().unwrap();
            safe_simd::vst1_u8(dst_arr, packed);
            col += 8;
        }

        // Scalar fallback
        while col < w {
            let d = dst_row[col] as i32;
            let t = tmp_row[col] as i32;
            let m = mask as i32;
            let d_scaled = d << 4;
            let diff = t - d_scaled;
            let weighted = (diff * m + 32) >> 6;
            let result = ((d_scaled + weighted + 8) >> 4).clamp(0, 255);
            dst_row[col] = result as u8;
            col += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn blend_v_8bpc_neon(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp: *const [DynPixel; SCRATCH_LAP_LEN],
    w: i32,
    h: i32,
    _dst: *const FFISafe<PicOffset>,
) {
    use crate::src::tables::dav1d_obmc_masks;

    let w = w as usize;
    let h = h as usize;

    // SAFETY: Pointers are valid and properly aligned
    let (dst, tmp) = unsafe {
        let dst = std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h * dst_stride.unsigned_abs());
        let tmp = std::slice::from_raw_parts(tmp as *const u8, w * h);
        (dst, tmp)
    };
    let mask = &dav1d_obmc_masks.0[w..];
    let dst_w = w * 3 >> 2;

    // blend_v formula: (dst * (64-m) + tmp * m + 32) >> 6
    for row in 0..h {
        let dst_row = &mut dst[row * dst_stride.unsigned_abs()..][..dst_w];
        let tmp_row = &tmp[row * w..][..dst_w];
        for col in 0..dst_w {
            let d = dst_row[col] as u32;
            let t = tmp_row[col] as u32;
            let m = mask[col] as u32;
            dst_row[col] = ((d * (64 - m) + t * m + 32) >> 6) as u8;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[arcane]
fn blend_v_16bpc_inner(
    _token: Arm64,
    dst: &mut [u16],
    dst_stride: usize,
    tmp: &[i16],
    w: usize,
    h: usize,
    obmc_masks: &[u8],
    bitdepth_max: i32,
) {
    for row in 0..h {
        let tmp_row = &tmp[row * w..][..w];
        let dst_row = &mut dst[row * dst_stride..][..w];
        let mask = obmc_masks[row];

        let mut col = 0;

        // Process 4 pixels at a time
        while col + 4 <= w {
            let d_u16 = safe_simd::vld1_u16(dst_row[col..][..4].try_into().unwrap());
            let d = vreinterpretq_s32_u32(vmovl_u16(d_u16));
            let t_16 = safe_simd::vld1_s16(tmp_row[col..][..4].try_into().unwrap());
            let t = vmovl_s16(t_16);
            let m = vdupq_n_s32(mask as i32);

            // diff = tmp - dst
            let diff = vsubq_s32(t, d);
            let weighted = vmulq_s32(diff, m);

            // (weighted + 32) >> 6
            let rnd = vdupq_n_s32(32);
            let shifted = vshrq_n_s32::<6>(vaddq_s32(weighted, rnd));
            let result = vaddq_s32(shifted, d);

            // Clamp
            let zero = vdupq_n_s32(0);
            let max = vdupq_n_s32(bitdepth_max);
            let clamped = vmaxq_s32(vminq_s32(result, max), zero);

            let narrow = vmovn_u32(vreinterpretq_u32_s32(clamped));
            let dst_arr: &mut [u16; 4] = (&mut dst_row[col..col + 4]).try_into().unwrap();
            safe_simd::vst1_u16(dst_arr, narrow);
            col += 4;
        }

        // Scalar fallback
        while col < w {
            let d = dst_row[col] as i32;
            let t = tmp_row[col] as i32;
            let m = mask as i32;
            let diff = t - d;
            let weighted = (diff * m + 32) >> 6;
            let result = (d + weighted).clamp(0, bitdepth_max);
            dst_row[col] = result as u16;
            col += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn blend_v_16bpc_neon(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp: *const [DynPixel; SCRATCH_LAP_LEN],
    w: i32,
    h: i32,
    _dst: *const FFISafe<PicOffset>,
) {
    use crate::src::tables::dav1d_obmc_masks;

    let w = w as usize;
    let h = h as usize;
    let dst_stride_u16 = (dst_stride / 2) as usize;

    // SAFETY: Pointers are valid and properly aligned
    let (dst, tmp) = unsafe {
        let dst = std::slice::from_raw_parts_mut(dst_ptr as *mut u16, h * dst_stride_u16);
        let tmp = std::slice::from_raw_parts(tmp as *const u16, w * h);
        (dst, tmp)
    };
    let mask = &dav1d_obmc_masks.0[w..];
    let dst_w = w * 3 >> 2;

    // blend_v formula: (dst * (64-m) + tmp * m + 32) >> 6
    for row in 0..h {
        let dst_row = &mut dst[row * dst_stride_u16..][..dst_w];
        let tmp_row = &tmp[row * w..][..dst_w];
        for col in 0..dst_w {
            let d = dst_row[col] as u32;
            let t = tmp_row[col] as u32;
            let m = mask[col] as u32;
            dst_row[col] = ((d * (64 - m) + t * m + 32) >> 6) as u16;
        }
    }
}

// ============================================================================
// BLEND_H - Horizontal OBMC blend
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[arcane]
fn blend_h_8bpc_inner(
    _token: Arm64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp: &[i16],
    w: usize,
    h: usize,
    obmc_masks: &[u8],
) {
    for row in 0..h {
        let tmp_row = &tmp[row * w..][..w];
        let dst_row = &mut dst[row * dst_stride..][..w];

        let mut col = 0;

        // Process pixels, mask varies by column
        while col + 8 <= w {
            let d = safe_simd::vld1_u8(dst_row[col..][..8].try_into().unwrap());
            let d16 = vreinterpretq_s16_u16(vmovl_u8(d));
            let t = safe_simd::vld1q_s16(tmp_row[col..][..8].try_into().unwrap());

            // Load 8 mask values
            let m = safe_simd::vld1_u8(obmc_masks[col..][..8].try_into().unwrap());
            let m16 = vreinterpretq_s16_u16(vmovl_u8(m));

            // diff = tmp - (dst << 4)
            let d_scaled = vshlq_n_s16::<4>(d16);
            let diff = vsubq_s16(t, d_scaled);

            // Widen for multiply
            let diff_lo = vmovl_s16(vget_low_s16(diff));
            let diff_hi = vmovl_s16(vget_high_s16(diff));
            let m_lo = vmovl_s16(vget_low_s16(m16));
            let m_hi = vmovl_s16(vget_high_s16(m16));

            // weighted = diff * mask
            let weighted_lo = vmulq_s32(diff_lo, m_lo);
            let weighted_hi = vmulq_s32(diff_hi, m_hi);

            // (weighted + 32) >> 6
            let rnd = vdupq_n_s32(32);
            let shifted_lo = vshrq_n_s32::<6>(vaddq_s32(weighted_lo, rnd));
            let shifted_hi = vshrq_n_s32::<6>(vaddq_s32(weighted_hi, rnd));

            // Narrow to 16-bit
            let shifted_16 = vcombine_s16(vmovn_s32(shifted_lo), vmovn_s32(shifted_hi));

            // Add d_scaled and shift right by 4
            let sum = vaddq_s16(shifted_16, d_scaled);
            let result = vshrq_n_s16::<4>(vaddq_s16(sum, vdupq_n_s16(8)));

            // Pack to u8
            let packed = vqmovun_s16(result);
            let dst_arr: &mut [u8; 8] = (&mut dst_row[col..col + 8]).try_into().unwrap();
            safe_simd::vst1_u8(dst_arr, packed);
            col += 8;
        }

        // Scalar fallback
        while col < w {
            let d = dst_row[col] as i32;
            let t = tmp_row[col] as i32;
            let m = obmc_masks[col] as i32;
            let d_scaled = d << 4;
            let diff = t - d_scaled;
            let weighted = (diff * m + 32) >> 6;
            let result = ((d_scaled + weighted + 8) >> 4).clamp(0, 255);
            dst_row[col] = result as u8;
            col += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn blend_h_8bpc_neon(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp: *const [DynPixel; SCRATCH_LAP_LEN],
    w: i32,
    h: i32,
    _dst: *const FFISafe<PicOffset>,
) {
    use crate::src::tables::dav1d_obmc_masks;

    let w = w as usize;
    let h = h as usize;
    let mask = &dav1d_obmc_masks.0[h..];
    let h_effective = h * 3 >> 2;

    // SAFETY: Pointers are valid and properly aligned
    let (dst, tmp) = unsafe {
        let dst = std::slice::from_raw_parts_mut(
            dst_ptr as *mut u8,
            h_effective * dst_stride.unsigned_abs(),
        );
        let tmp = std::slice::from_raw_parts(tmp as *const u8, w * h_effective);
        (dst, tmp)
    };

    // blend_h formula: (dst * (64-m) + tmp * m + 32) >> 6
    for row in 0..h_effective {
        let dst_row = &mut dst[row * dst_stride.unsigned_abs()..][..w];
        let tmp_row = &tmp[row * w..][..w];
        let m = mask[row] as u32;
        for col in 0..w {
            let d = dst_row[col] as u32;
            let t = tmp_row[col] as u32;
            dst_row[col] = ((d * (64 - m) + t * m + 32) >> 6) as u8;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[arcane]
fn blend_h_16bpc_inner(
    _token: Arm64,
    dst: &mut [u16],
    dst_stride: usize,
    tmp: &[i16],
    w: usize,
    h: usize,
    obmc_masks: &[u8],
    bitdepth_max: i32,
) {
    for row in 0..h {
        let tmp_row = &tmp[row * w..][..w];
        let dst_row = &mut dst[row * dst_stride..][..w];

        let mut col = 0;

        // Process 4 pixels at a time
        while col + 4 <= w {
            let d_u16 = safe_simd::vld1_u16(dst_row[col..][..4].try_into().unwrap());
            let d = vreinterpretq_s32_u32(vmovl_u16(d_u16));
            let t_16 = safe_simd::vld1_s16(tmp_row[col..][..4].try_into().unwrap());
            let t = vmovl_s16(t_16);

            // Load 4 mask bytes
            let m_bytes: [u8; 8] = [
                obmc_masks[col],
                obmc_masks[col + 1],
                obmc_masks[col + 2],
                obmc_masks[col + 3],
                0,
                0,
                0,
                0,
            ];
            let m8 = safe_simd::vld1_u8(&m_bytes);
            let m16 = vmovl_u8(m8);
            let m = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(m16)));

            // diff = tmp - dst
            let diff = vsubq_s32(t, d);
            let weighted = vmulq_s32(diff, m);

            // (weighted + 32) >> 6
            let rnd = vdupq_n_s32(32);
            let shifted = vshrq_n_s32::<6>(vaddq_s32(weighted, rnd));
            let result = vaddq_s32(shifted, d);

            // Clamp
            let zero = vdupq_n_s32(0);
            let max = vdupq_n_s32(bitdepth_max);
            let clamped = vmaxq_s32(vminq_s32(result, max), zero);

            let narrow = vmovn_u32(vreinterpretq_u32_s32(clamped));
            let dst_arr: &mut [u16; 4] = (&mut dst_row[col..col + 4]).try_into().unwrap();
            safe_simd::vst1_u16(dst_arr, narrow);
            col += 4;
        }

        // Scalar fallback
        while col < w {
            let d = dst_row[col] as i32;
            let t = tmp_row[col] as i32;
            let m = obmc_masks[col] as i32;
            let diff = t - d;
            let weighted = (diff * m + 32) >> 6;
            let result = (d + weighted).clamp(0, bitdepth_max);
            dst_row[col] = result as u16;
            col += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn blend_h_16bpc_neon(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp: *const [DynPixel; SCRATCH_LAP_LEN],
    w: i32,
    h: i32,
    _dst: *const FFISafe<PicOffset>,
) {
    use crate::src::tables::dav1d_obmc_masks;

    let w = w as usize;
    let h = h as usize;
    let mask = &dav1d_obmc_masks.0[h..];
    let h_effective = h * 3 >> 2;
    let dst_stride_u16 = (dst_stride / 2) as usize;

    // SAFETY: Pointers are valid and properly aligned
    let (dst, tmp) = unsafe {
        let dst = std::slice::from_raw_parts_mut(dst_ptr as *mut u16, h_effective * dst_stride_u16);
        let tmp = std::slice::from_raw_parts(tmp as *const u16, w * h_effective);
        (dst, tmp)
    };

    // blend_h formula: (dst * (64-m) + tmp * m + 32) >> 6
    for row in 0..h_effective {
        let dst_row = &mut dst[row * dst_stride_u16..][..w];
        let tmp_row = &tmp[row * w..][..w];
        let m = mask[row] as u32;
        for col in 0..w {
            let d = dst_row[col] as u32;
            let t = tmp_row[col] as u32;
            dst_row[col] = ((d * (64 - m) + t * m + 32) >> 6) as u16;
        }
    }
}

// ============================================================================
// W_MASK - Weighted mask blend (compound prediction with per-pixel masking)
// ============================================================================

/// Core w_mask implementation for 8bpc
/// SS_HOR and SS_VER control subsampling: 444=(false,false), 422=(true,false), 420=(true,true)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn w_mask_8bpc_inner<const SS_HOR: bool, const SS_VER: bool>(
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    mask: &mut [u8],
    sign: u8,
) {
    // For 8bpc: intermediate_bits = 4, bitdepth = 8
    let intermediate_bits = 4i32;
    let sh = intermediate_bits + 6;
    let rnd = (32 << intermediate_bits) + 8192 * 64; // PREP_BIAS = 8192 for 8bpc in compound
    let mask_sh = 8 + intermediate_bits - 4; // bitdepth + intermediate_bits - 4
    let mask_rnd = 1i32 << (mask_sh - 5);

    // Mask output dimensions depend on subsampling
    let mask_w = if SS_HOR { w >> 1 } else { w };

    for y in 0..h {
        let tmp1_row = &tmp1[y * w..][..w];
        let tmp2_row = &tmp2[y * w..][..w];
        let dst_row = &mut dst[y * dst_stride..][..w];
        let mut mask_row = if SS_VER && (y & 1) != 0 {
            None
        } else {
            let mask_y = if SS_VER { y >> 1 } else { y };
            Some(&mut mask[mask_y * mask_w..][..mask_w])
        };

        let mut col = 0;

        // SIMD: process 8 pixels at a time
        while col + 8 <= w {
            let t1 = safe_simd::vld1q_s16(tmp1_row[col..][..8].try_into().unwrap());
            let t2 = safe_simd::vld1q_s16(tmp2_row[col..][..8].try_into().unwrap());

            // Compute diff and mask value
            // abs_diff = |tmp1 - tmp2|
            let diff = vsubq_s16(t1, t2);
            let abs_diff = vabsq_s16(diff);

            // mask = min(38 + (abs_diff + mask_rnd) >> mask_sh, 64)
            let abs_32_lo = vmovl_s16(vget_low_s16(abs_diff));
            let abs_32_hi = vmovl_s16(vget_high_s16(abs_diff));

            let mask_rnd_vec = vdupq_n_s32(mask_rnd);
            let m_lo = vaddq_s32(abs_32_lo, mask_rnd_vec);
            let m_hi = vaddq_s32(abs_32_hi, mask_rnd_vec);

            // Shift by mask_sh (= 8)
            let m_shifted_lo = vshrq_n_s32::<8>(m_lo);
            let m_shifted_hi = vshrq_n_s32::<8>(m_hi);

            // Add 38 and clamp to 64
            let m_lo = vaddq_s32(m_shifted_lo, vdupq_n_s32(38));
            let m_hi = vaddq_s32(m_shifted_hi, vdupq_n_s32(38));
            let m_lo = vminq_s32(m_lo, vdupq_n_s32(64));
            let m_hi = vminq_s32(m_hi, vdupq_n_s32(64));

            // Narrow to 16-bit for blending
            let m_16 = vcombine_s16(vmovn_s32(m_lo), vmovn_s32(m_hi));

            // Apply sign: if sign, swap effective weights
            let m_final = if sign != 0 {
                vsubq_s16(vdupq_n_s16(64), m_16)
            } else {
                m_16
            };
            let inv_m = vsubq_s16(vdupq_n_s16(64), m_final);

            // Widen tmp values to 32-bit for multiply
            let t1_lo = vmovl_s16(vget_low_s16(t1));
            let t1_hi = vmovl_s16(vget_high_s16(t1));
            let t2_lo = vmovl_s16(vget_low_s16(t2));
            let t2_hi = vmovl_s16(vget_high_s16(t2));
            let m_lo_32 = vmovl_s16(vget_low_s16(m_final));
            let m_hi_32 = vmovl_s16(vget_high_s16(m_final));
            let inv_m_lo_32 = vmovl_s16(vget_low_s16(inv_m));
            let inv_m_hi_32 = vmovl_s16(vget_high_s16(inv_m));

            // blend = (tmp1 * m + tmp2 * (64-m) + rnd) >> sh
            let rnd_vec = vdupq_n_s32(rnd);
            let blend_lo = vaddq_s32(
                vaddq_s32(vmulq_s32(t1_lo, m_lo_32), vmulq_s32(t2_lo, inv_m_lo_32)),
                rnd_vec,
            );
            let blend_hi = vaddq_s32(
                vaddq_s32(vmulq_s32(t1_hi, m_hi_32), vmulq_s32(t2_hi, inv_m_hi_32)),
                rnd_vec,
            );

            // Shift by sh (= 10)
            let result_lo = vshrq_n_s32::<10>(blend_lo);
            let result_hi = vshrq_n_s32::<10>(blend_hi);

            // Clamp to [0, 255]
            let zero = vdupq_n_s32(0);
            let max_val = vdupq_n_s32(255);
            let result_lo = vmaxq_s32(vminq_s32(result_lo, max_val), zero);
            let result_hi = vmaxq_s32(vminq_s32(result_hi, max_val), zero);

            // Narrow to u8
            let narrow_lo = vmovn_s32(result_lo);
            let narrow_hi = vmovn_s32(result_hi);
            let narrow_16 = vcombine_s16(narrow_lo, narrow_hi);
            let result_u8 = vqmovun_s16(narrow_16);

            let dst_arr: &mut [u8; 8] = (&mut dst_row[col..col + 8]).try_into().unwrap();
            safe_simd::vst1_u8(dst_arr, result_u8);

            // Store mask if needed
            if let Some(ref mut mask_row) = mask_row {
                // For 444: 1:1 mask storage
                // For 422: horizontal averaging (2 pixels -> 1 mask)
                // For 420: also horizontal averaging
                if !SS_HOR {
                    // 444: store all mask values
                    let m_narrow = vqmovun_s16(m_16);
                    let mask_arr: &mut [u8; 8] = (&mut mask_row[col..col + 8]).try_into().unwrap();
                    safe_simd::vst1_u8(mask_arr, m_narrow);
                } else {
                    // 422/420: average pairs horizontally (unrolled - vgetq_lane requires const)
                    let mask_idx = col >> 1;
                    let m0 = vgetq_lane_s16::<0>(m_16) as i32;
                    let m1 = vgetq_lane_s16::<1>(m_16) as i32;
                    mask_row[mask_idx] = ((m0 + m1 + 1) >> 1) as u8;
                    let m2 = vgetq_lane_s16::<2>(m_16) as i32;
                    let m3 = vgetq_lane_s16::<3>(m_16) as i32;
                    mask_row[mask_idx + 1] = ((m2 + m3 + 1) >> 1) as u8;
                    let m4 = vgetq_lane_s16::<4>(m_16) as i32;
                    let m5 = vgetq_lane_s16::<5>(m_16) as i32;
                    mask_row[mask_idx + 2] = ((m4 + m5 + 1) >> 1) as u8;
                    let m6 = vgetq_lane_s16::<6>(m_16) as i32;
                    let m7 = vgetq_lane_s16::<7>(m_16) as i32;
                    mask_row[mask_idx + 3] = ((m6 + m7 + 1) >> 1) as u8;
                }
            }
            col += 8;
        }

        // Scalar fallback
        while col < w {
            let t1 = tmp1_row[col] as i32;
            let t2 = tmp2_row[col] as i32;
            let diff = t1 - t2;
            let abs_diff = diff.abs();

            // Compute mask
            let mut m = 38 + ((abs_diff + mask_rnd) >> mask_sh);
            m = m.min(64);

            let m_final = if sign != 0 { 64 - m } else { m };
            let inv_m = 64 - m_final;

            // Blend
            let blend = (t1 * m_final + t2 * inv_m + rnd) >> sh;
            dst_row[col] = blend.clamp(0, 255) as u8;

            // Store mask with subsampling
            if let Some(ref mut mask_row) = mask_row {
                if !SS_HOR {
                    mask_row[col] = m as u8;
                } else if (col & 1) == 0 {
                    // For 422/420, store averaged pairs
                    let mask_idx = col >> 1;
                    if col + 1 < w {
                        let t1_next = tmp1_row[col + 1] as i32;
                        let t2_next = tmp2_row[col + 1] as i32;
                        let diff_next = (t1_next - t2_next).abs();
                        let m_next = (38 + ((diff_next + mask_rnd) >> mask_sh)).min(64);
                        mask_row[mask_idx] = ((m + m_next + 1) >> 1) as u8;
                    } else {
                        mask_row[mask_idx] = m as u8;
                    }
                }
            }

            col += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn w_mask_444_8bpc_neon(
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
    let w = w as usize;
    let h = h as usize;

    // SAFETY: Pointers are valid and properly aligned
    let dst = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h * dst_stride.unsigned_abs())
    };

    w_mask_8bpc_inner::<false, false>(
        dst,
        dst_stride as usize,
        tmp1.as_slice(),
        tmp2.as_slice(),
        w,
        h,
        mask.as_mut_slice(),
        sign as u8,
    );
}

#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn w_mask_422_8bpc_neon(
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
    let w = w as usize;
    let h = h as usize;

    // SAFETY: Pointers are valid and properly aligned
    let dst = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h * dst_stride.unsigned_abs())
    };

    w_mask_8bpc_inner::<true, false>(
        dst,
        dst_stride as usize,
        tmp1.as_slice(),
        tmp2.as_slice(),
        w,
        h,
        mask.as_mut_slice(),
        sign as u8,
    );
}

#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn w_mask_420_8bpc_neon(
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
    let w = w as usize;
    let h = h as usize;

    // SAFETY: Pointers are valid and properly aligned
    let dst = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h * dst_stride.unsigned_abs())
    };

    w_mask_8bpc_inner::<true, true>(
        dst,
        dst_stride as usize,
        tmp1.as_slice(),
        tmp2.as_slice(),
        w,
        h,
        mask.as_mut_slice(),
        sign as u8,
    );
}

// ============================================================================
// BILINEAR FILTER - Motion compensation with bilinear interpolation
// ============================================================================

/// Bilinear put for 8bpc - copies or interpolates based on mx/my
#[cfg(target_arch = "aarch64")]
#[arcane]
fn put_bilin_8bpc_inner(
    _token: Arm64,
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
) {
    match (mx, my) {
        (0, 0) => {
            // Simple copy
            for y in 0..h {
                let src_row = &src[y * src_stride..][..w];
                let dst_row = &mut dst[y * dst_stride..][..w];
                dst_row.copy_from_slice(src_row);
            }
        }
        (0, _) => {
            // Vertical-only bilinear
            let my = my as i16;
            let coeff0 = 16 - my;
            let coeff1 = my;

            for y in 0..h {
                let src_row0 = &src[y * src_stride..][..w];
                let src_row1 = &src[(y + 1) * src_stride..][..w];
                let dst_row = &mut dst[y * dst_stride..][..w];

                let mut x = 0;
                while x + 8 <= w {
                    let r0 = safe_simd::vld1_u8(src_row0[x..][..8].try_into().unwrap());
                    let r1 = safe_simd::vld1_u8(src_row1[x..][..8].try_into().unwrap());

                    // Widen to 16-bit
                    let r0_16 = vreinterpretq_s16_u16(vmovl_u8(r0));
                    let r1_16 = vreinterpretq_s16_u16(vmovl_u8(r1));

                    // Multiply by coefficients
                    let c0 = vdupq_n_s16(coeff0);
                    let c1 = vdupq_n_s16(coeff1);
                    let mul0 = vmulq_s16(r0_16, c0);
                    let mul1 = vmulq_s16(r1_16, c1);
                    let sum = vaddq_s16(mul0, mul1);

                    // Round and shift: (sum + 8) >> 4
                    let rnd = vdupq_n_s16(8);
                    let result = vshrq_n_s16::<4>(vaddq_s16(sum, rnd));

                    // Pack to u8 with saturation
                    let packed = vqmovun_s16(result);
                    let dst_arr: &mut [u8; 8] = (&mut dst_row[x..x + 8]).try_into().unwrap();
                    safe_simd::vst1_u8(dst_arr, packed);
                    x += 8;
                }

                // Scalar fallback
                while x < w {
                    let r0 = src_row0[x] as i32;
                    let r1 = src_row1[x] as i32;
                    let pixel = coeff0 as i32 * r0 + coeff1 as i32 * r1;
                    dst_row[x] = ((pixel + 8) >> 4).clamp(0, 255) as u8;
                    x += 1;
                }
            }
        }
        (_, 0) => {
            // Horizontal-only bilinear
            let mx = mx as i16;
            let coeff0 = 16 - mx;
            let coeff1 = mx;

            for y in 0..h {
                let src_row = &src[y * src_stride..][..w + 1];
                let dst_row = &mut dst[y * dst_stride..][..w];

                let mut x = 0;
                while x + 8 <= w {
                    let s0 = safe_simd::vld1_u8(src_row[x..][..8].try_into().unwrap());
                    let s1 = safe_simd::vld1_u8(src_row[x + 1..][..8].try_into().unwrap());

                    // Widen to 16-bit
                    let s0_16 = vreinterpretq_s16_u16(vmovl_u8(s0));
                    let s1_16 = vreinterpretq_s16_u16(vmovl_u8(s1));

                    // Multiply and add
                    let c0 = vdupq_n_s16(coeff0);
                    let c1 = vdupq_n_s16(coeff1);
                    let mul0 = vmulq_s16(s0_16, c0);
                    let mul1 = vmulq_s16(s1_16, c1);
                    let sum = vaddq_s16(mul0, mul1);

                    // Round and shift: (sum + 8) >> 4
                    let rnd = vdupq_n_s16(8);
                    let result = vshrq_n_s16::<4>(vaddq_s16(sum, rnd));

                    // Pack to u8
                    let packed = vqmovun_s16(result);
                    let dst_arr: &mut [u8; 8] = (&mut dst_row[x..x + 8]).try_into().unwrap();
                    safe_simd::vst1_u8(dst_arr, packed);
                    x += 8;
                }

                // Scalar fallback
                while x < w {
                    let s0 = src_row[x] as i32;
                    let s1 = src_row[x + 1] as i32;
                    let pixel = coeff0 as i32 * s0 + coeff1 as i32 * s1;
                    dst_row[x] = ((pixel + 8) >> 4).clamp(0, 255) as u8;
                    x += 1;
                }
            }
        }
        (_, _) => {
            // Both horizontal and vertical bilinear
            // First apply horizontal, then vertical
            let mx = mx as i16;
            let my = my as i16;
            let h_coeff0 = 16 - mx;
            let h_coeff1 = mx;
            let v_coeff0 = 16 - my;
            let v_coeff1 = my;

            // Intermediate buffer for horizontal results
            let mid_stride = w + 16;
            let mut mid = vec![0i16; mid_stride * (h + 1)];

            // Horizontal pass
            for y in 0..h + 1 {
                let src_row = &src[y * src_stride..];
                let mid_row = &mut mid[y * mid_stride..][..w];

                for x in 0..w {
                    let s0 = src_row[x] as i32;
                    let s1 = src_row[x + 1] as i32;
                    let pixel = h_coeff0 as i32 * s0 + h_coeff1 as i32 * s1;
                    mid_row[x] = pixel as i16;
                }
            }

            // Vertical pass
            for y in 0..h {
                let mid_row0 = &mid[y * mid_stride..][..w];
                let mid_row1 = &mid[(y + 1) * mid_stride..][..w];
                let dst_row = &mut dst[y * dst_stride..][..w];

                let mut x = 0;
                while x + 8 <= w {
                    let r0 = safe_simd::vld1q_s16(mid_row0[x..][..8].try_into().unwrap());
                    let r1 = safe_simd::vld1q_s16(mid_row1[x..][..8].try_into().unwrap());

                    // Widen to 32-bit for multiply
                    let r0_lo = vmovl_s16(vget_low_s16(r0));
                    let r0_hi = vmovl_s16(vget_high_s16(r0));
                    let r1_lo = vmovl_s16(vget_low_s16(r1));
                    let r1_hi = vmovl_s16(vget_high_s16(r1));

                    let c0 = vdupq_n_s32(v_coeff0 as i32);
                    let c1 = vdupq_n_s32(v_coeff1 as i32);

                    let sum_lo = vaddq_s32(vmulq_s32(r0_lo, c0), vmulq_s32(r1_lo, c1));
                    let sum_hi = vaddq_s32(vmulq_s32(r0_hi, c0), vmulq_s32(r1_hi, c1));

                    // Round and shift: (sum + 128) >> 8 for combined H+V
                    let rnd = vdupq_n_s32(128);
                    let result_lo = vshrq_n_s32::<8>(vaddq_s32(sum_lo, rnd));
                    let result_hi = vshrq_n_s32::<8>(vaddq_s32(sum_hi, rnd));

                    // Clamp and narrow
                    let zero = vdupq_n_s32(0);
                    let max_val = vdupq_n_s32(255);
                    let result_lo = vmaxq_s32(vminq_s32(result_lo, max_val), zero);
                    let result_hi = vmaxq_s32(vminq_s32(result_hi, max_val), zero);

                    // Narrow to 16-bit then 8-bit
                    let narrow_lo = vmovn_s32(result_lo);
                    let narrow_hi = vmovn_s32(result_hi);
                    let narrow_16 = vcombine_s16(narrow_lo, narrow_hi);
                    let result_u8 = vqmovun_s16(narrow_16);

                    let dst_arr: &mut [u8; 8] = (&mut dst_row[x..x + 8]).try_into().unwrap();
                    safe_simd::vst1_u8(dst_arr, result_u8);
                    x += 8;
                }

                // Scalar fallback
                while x < w {
                    let r0 = mid_row0[x] as i32;
                    let r1 = mid_row1[x] as i32;
                    let pixel = v_coeff0 as i32 * r0 + v_coeff1 as i32 * r1;
                    dst_row[x] = ((pixel + 128) >> 8).clamp(0, 255) as u8;
                    x += 1;
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn put_bilin_8bpc_neon(
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
    let w = w as usize;
    let h = h as usize;

    // SAFETY: This function is only called through dispatch when NEON is available
    // Pointers are valid and properly aligned
    let (token, src, dst) = unsafe {
        let token = unsafe { Arm64::forge_token_dangerously() };
        let src = std::slice::from_raw_parts(
            src_ptr as *const u8,
            (h + 1) * src_stride.unsigned_abs() + w + 1,
        );
        let dst = std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h * dst_stride.unsigned_abs());
        (token, src, dst)
    };

    put_bilin_8bpc_inner(
        token,
        dst,
        dst_stride as usize,
        src,
        src_stride as usize,
        w,
        h,
        mx,
        my,
    );
}

/// Bilinear prep for 8bpc - outputs to intermediate buffer
#[cfg(target_arch = "aarch64")]
#[arcane]
fn prep_bilin_8bpc_inner(
    _token: Arm64,
    tmp: &mut [i16],
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
) {
    // PREP_BIAS for intermediate format
    const PREP_BIAS: i16 = 8192;

    match (mx, my) {
        (0, 0) => {
            // Simple copy to prep format
            for y in 0..h {
                let src_row = &src[y * src_stride..][..w];
                let tmp_row = &mut tmp[y * w..][..w];

                let mut x = 0;
                while x + 8 <= w {
                    let s = safe_simd::vld1_u8(src_row[x..][..8].try_into().unwrap());
                    let s16 = vreinterpretq_s16_u16(vmovl_u8(s));

                    // Scale: (pixel - 512) << 4 for intermediate format
                    // Actually for 8bpc: pixel << 4 with PREP_BIAS offset
                    let scaled = vshlq_n_s16::<4>(s16);
                    let biased = vsubq_s16(scaled, vdupq_n_s16(PREP_BIAS));

                    let tmp_arr: &mut [i16; 8] = (&mut tmp_row[x..x + 8]).try_into().unwrap();
                    safe_simd::vst1q_s16(tmp_arr, biased);
                    x += 8;
                }

                // Scalar fallback
                while x < w {
                    let pixel = src_row[x] as i16;
                    tmp_row[x] = (pixel << 4) - PREP_BIAS;
                    x += 1;
                }
            }
        }
        (0, _) => {
            // Vertical-only bilinear to prep
            let my = my as i16;
            let coeff0 = 16 - my;
            let coeff1 = my;

            for y in 0..h {
                let src_row0 = &src[y * src_stride..][..w];
                let src_row1 = &src[(y + 1) * src_stride..][..w];
                let tmp_row = &mut tmp[y * w..][..w];

                let mut x = 0;
                while x + 8 <= w {
                    let r0 = safe_simd::vld1_u8(src_row0[x..][..8].try_into().unwrap());
                    let r1 = safe_simd::vld1_u8(src_row1[x..][..8].try_into().unwrap());

                    let r0_16 = vreinterpretq_s16_u16(vmovl_u8(r0));
                    let r1_16 = vreinterpretq_s16_u16(vmovl_u8(r1));

                    let c0 = vdupq_n_s16(coeff0);
                    let c1 = vdupq_n_s16(coeff1);
                    let mul0 = vmulq_s16(r0_16, c0);
                    let mul1 = vmulq_s16(r1_16, c1);
                    let sum = vaddq_s16(mul0, mul1);

                    // For prep: no shift, just apply bias
                    let biased = vsubq_s16(sum, vdupq_n_s16(PREP_BIAS));

                    let tmp_arr: &mut [i16; 8] = (&mut tmp_row[x..x + 8]).try_into().unwrap();
                    safe_simd::vst1q_s16(tmp_arr, biased);
                    x += 8;
                }

                // Scalar fallback
                while x < w {
                    let r0 = src_row0[x] as i32;
                    let r1 = src_row1[x] as i32;
                    let pixel = coeff0 as i32 * r0 + coeff1 as i32 * r1;
                    tmp_row[x] = (pixel - PREP_BIAS as i32) as i16;
                    x += 1;
                }
            }
        }
        (_, 0) => {
            // Horizontal-only bilinear to prep
            let mx = mx as i16;
            let coeff0 = 16 - mx;
            let coeff1 = mx;

            for y in 0..h {
                let src_row = &src[y * src_stride..][..w + 1];
                let tmp_row = &mut tmp[y * w..][..w];

                let mut x = 0;
                while x + 8 <= w {
                    let s0 = safe_simd::vld1_u8(src_row[x..][..8].try_into().unwrap());
                    let s1 = safe_simd::vld1_u8(src_row[x + 1..][..8].try_into().unwrap());

                    let s0_16 = vreinterpretq_s16_u16(vmovl_u8(s0));
                    let s1_16 = vreinterpretq_s16_u16(vmovl_u8(s1));

                    let c0 = vdupq_n_s16(coeff0);
                    let c1 = vdupq_n_s16(coeff1);
                    let mul0 = vmulq_s16(s0_16, c0);
                    let mul1 = vmulq_s16(s1_16, c1);
                    let sum = vaddq_s16(mul0, mul1);

                    let biased = vsubq_s16(sum, vdupq_n_s16(PREP_BIAS));

                    let tmp_arr: &mut [i16; 8] = (&mut tmp_row[x..x + 8]).try_into().unwrap();
                    safe_simd::vst1q_s16(tmp_arr, biased);
                    x += 8;
                }

                // Scalar fallback
                while x < w {
                    let s0 = src_row[x] as i32;
                    let s1 = src_row[x + 1] as i32;
                    let pixel = coeff0 as i32 * s0 + coeff1 as i32 * s1;
                    tmp_row[x] = (pixel - PREP_BIAS as i32) as i16;
                    x += 1;
                }
            }
        }
        (_, _) => {
            // Both H+V bilinear to prep
            let mx = mx as i16;
            let my = my as i16;
            let h_coeff0 = 16 - mx;
            let h_coeff1 = mx;
            let v_coeff0 = 16 - my;
            let v_coeff1 = my;

            // Intermediate buffer
            let mid_stride = w + 16;
            let mut mid = vec![0i16; mid_stride * (h + 1)];

            // Horizontal pass
            for y in 0..h + 1 {
                let src_row = &src[y * src_stride..];
                let mid_row = &mut mid[y * mid_stride..][..w];

                for x in 0..w {
                    let s0 = src_row[x] as i32;
                    let s1 = src_row[x + 1] as i32;
                    let pixel = h_coeff0 as i32 * s0 + h_coeff1 as i32 * s1;
                    mid_row[x] = pixel as i16;
                }
            }

            // Vertical pass
            for y in 0..h {
                let mid_row0 = &mid[y * mid_stride..][..w];
                let mid_row1 = &mid[(y + 1) * mid_stride..][..w];
                let tmp_row = &mut tmp[y * w..][..w];

                let mut x = 0;
                while x + 8 <= w {
                    let r0 = safe_simd::vld1q_s16(mid_row0[x..][..8].try_into().unwrap());
                    let r1 = safe_simd::vld1q_s16(mid_row1[x..][..8].try_into().unwrap());

                    // Widen to 32-bit
                    let r0_lo = vmovl_s16(vget_low_s16(r0));
                    let r0_hi = vmovl_s16(vget_high_s16(r0));
                    let r1_lo = vmovl_s16(vget_low_s16(r1));
                    let r1_hi = vmovl_s16(vget_high_s16(r1));

                    let c0 = vdupq_n_s32(v_coeff0 as i32);
                    let c1 = vdupq_n_s32(v_coeff1 as i32);

                    let sum_lo = vaddq_s32(vmulq_s32(r0_lo, c0), vmulq_s32(r1_lo, c1));
                    let sum_hi = vaddq_s32(vmulq_s32(r0_hi, c0), vmulq_s32(r1_hi, c1));

                    // Shift by 4 for prep format
                    let result_lo = vshrq_n_s32::<4>(sum_lo);
                    let result_hi = vshrq_n_s32::<4>(sum_hi);

                    // Narrow to 16-bit
                    let narrow_lo = vmovn_s32(result_lo);
                    let narrow_hi = vmovn_s32(result_hi);
                    let narrow_16 = vcombine_s16(narrow_lo, narrow_hi);

                    // Apply bias
                    let biased = vsubq_s16(narrow_16, vdupq_n_s16(PREP_BIAS));

                    let tmp_arr: &mut [i16; 8] = (&mut tmp_row[x..x + 8]).try_into().unwrap();
                    safe_simd::vst1q_s16(tmp_arr, biased);
                    x += 8;
                }

                // Scalar fallback
                while x < w {
                    let r0 = mid_row0[x] as i32;
                    let r1 = mid_row1[x] as i32;
                    let pixel = v_coeff0 as i32 * r0 + v_coeff1 as i32 * r1;
                    tmp_row[x] = ((pixel >> 4) - PREP_BIAS as i32) as i16;
                    x += 1;
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn prep_bilin_8bpc_neon(
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
    let w = w as usize;
    let h = h as usize;

    // SAFETY: This function is only called through dispatch when NEON is available
    // Pointers are valid and properly aligned
    let (token, src, tmp_slice) = unsafe {
        let token = unsafe { Arm64::forge_token_dangerously() };
        let src = std::slice::from_raw_parts(
            src_ptr as *const u8,
            (h + 1) * src_stride.unsigned_abs() + w + 1,
        );
        let tmp_slice = std::slice::from_raw_parts_mut(tmp, h * w);
        (token, src, tmp_slice)
    };

    prep_bilin_8bpc_inner(token, tmp_slice, src, src_stride as usize, w, h, mx, my);
}

// ============================================================================
// BILINEAR FILTER 16bpc - Motion compensation with bilinear interpolation
// ============================================================================

/// Bilinear put for 16bpc
#[cfg(target_arch = "aarch64")]
#[arcane]
fn put_bilin_16bpc_inner(
    _token: Arm64,
    dst: &mut [u16],
    dst_stride: usize,
    src: &[u16],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
) {
    match (mx, my) {
        (0, 0) => {
            // Simple copy
            for y in 0..h {
                let src_row = &src[y * src_stride..][..w];
                let dst_row = &mut dst[y * dst_stride..][..w];
                dst_row.copy_from_slice(src_row);
            }
        }
        (0, _) => {
            // Vertical-only bilinear
            let coeff0 = 16 - my;
            let coeff1 = my;

            for y in 0..h {
                let src_row0 = &src[y * src_stride..][..w];
                let src_row1 = &src[(y + 1) * src_stride..][..w];
                let dst_row = &mut dst[y * dst_stride..][..w];

                for x in 0..w {
                    let r0 = src_row0[x] as i32;
                    let r1 = src_row1[x] as i32;
                    let pixel = coeff0 * r0 + coeff1 * r1;
                    dst_row[x] = ((pixel + 8) >> 4).clamp(0, bitdepth_max) as u16;
                }
            }
        }
        (_, 0) => {
            // Horizontal-only bilinear
            let coeff0 = 16 - mx;
            let coeff1 = mx;

            for y in 0..h {
                let src_row = &src[y * src_stride..][..w + 1];
                let dst_row = &mut dst[y * dst_stride..][..w];

                for x in 0..w {
                    let s0 = src_row[x] as i32;
                    let s1 = src_row[x + 1] as i32;
                    let pixel = coeff0 * s0 + coeff1 * s1;
                    dst_row[x] = ((pixel + 8) >> 4).clamp(0, bitdepth_max) as u16;
                }
            }
        }
        (_, _) => {
            // Both H+V bilinear
            let h_coeff0 = 16 - mx;
            let h_coeff1 = mx;
            let v_coeff0 = 16 - my;
            let v_coeff1 = my;

            // Intermediate buffer
            let mid_stride = w + 16;
            let mut mid = vec![0i32; mid_stride * (h + 1)];

            // Horizontal pass
            for y in 0..h + 1 {
                let src_row = &src[y * src_stride..];
                let mid_row = &mut mid[y * mid_stride..][..w];

                for x in 0..w {
                    let s0 = src_row[x] as i32;
                    let s1 = src_row[x + 1] as i32;
                    mid_row[x] = h_coeff0 * s0 + h_coeff1 * s1;
                }
            }

            // Vertical pass
            for y in 0..h {
                let mid_row0 = &mid[y * mid_stride..][..w];
                let mid_row1 = &mid[(y + 1) * mid_stride..][..w];
                let dst_row = &mut dst[y * dst_stride..][..w];

                for x in 0..w {
                    let r0 = mid_row0[x];
                    let r1 = mid_row1[x];
                    let pixel = v_coeff0 * r0 + v_coeff1 * r1;
                    // Double shift: (pixel + 128) >> 8
                    dst_row[x] = ((pixel + 128) >> 8).clamp(0, bitdepth_max) as u16;
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn put_bilin_16bpc_neon(
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
    let w = w as usize;
    let h = h as usize;
    let dst_stride_u16 = (dst_stride / 2) as usize;
    let src_stride_u16 = (src_stride / 2) as usize;

    // SAFETY: This function is only called through dispatch when NEON is available
    // Pointers are valid and properly aligned
    let (token, dst, src) = unsafe {
        let token = unsafe { Arm64::forge_token_dangerously() };
        let dst = std::slice::from_raw_parts_mut(dst_ptr as *mut u16, h * dst_stride_u16);
        let src =
            std::slice::from_raw_parts(src_ptr as *const u16, (h + 1) * src_stride_u16 + w + 1);
        (token, dst, src)
    };

    put_bilin_16bpc_inner(
        token,
        dst,
        dst_stride_u16,
        src,
        src_stride_u16,
        w,
        h,
        mx,
        my,
        bitdepth_max,
    );
}

const PREP_BIAS_16BPC: i32 = 8192;

/// Bilinear prep for 16bpc
#[cfg(target_arch = "aarch64")]
#[arcane]
fn prep_bilin_16bpc_inner(
    _token: Arm64,
    tmp: &mut [i16],
    src: &[u16],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
) {
    match (mx, my) {
        (0, 0) => {
            // Simple copy with bias
            for y in 0..h {
                let src_row = &src[y * src_stride..][..w];
                let tmp_row = &mut tmp[y * w..][..w];
                for x in 0..w {
                    tmp_row[x] = (src_row[x] as i32 - PREP_BIAS_16BPC) as i16;
                }
            }
        }
        (0, _) => {
            // Vertical-only bilinear
            let coeff0 = 16 - my;
            let coeff1 = my;

            for y in 0..h {
                let src_row0 = &src[y * src_stride..][..w];
                let src_row1 = &src[(y + 1) * src_stride..][..w];
                let tmp_row = &mut tmp[y * w..][..w];

                for x in 0..w {
                    let r0 = src_row0[x] as i32;
                    let r1 = src_row1[x] as i32;
                    let pixel = coeff0 * r0 + coeff1 * r1;
                    tmp_row[x] = ((pixel >> 4) - PREP_BIAS_16BPC) as i16;
                }
            }
        }
        (_, 0) => {
            // Horizontal-only bilinear
            let coeff0 = 16 - mx;
            let coeff1 = mx;

            for y in 0..h {
                let src_row = &src[y * src_stride..][..w + 1];
                let tmp_row = &mut tmp[y * w..][..w];

                for x in 0..w {
                    let s0 = src_row[x] as i32;
                    let s1 = src_row[x + 1] as i32;
                    let pixel = coeff0 * s0 + coeff1 * s1;
                    tmp_row[x] = ((pixel >> 4) - PREP_BIAS_16BPC) as i16;
                }
            }
        }
        (_, _) => {
            // Both H+V bilinear
            let h_coeff0 = 16 - mx;
            let h_coeff1 = mx;
            let v_coeff0 = 16 - my;
            let v_coeff1 = my;

            // Intermediate buffer
            let mid_stride = w + 16;
            let mut mid = vec![0i32; mid_stride * (h + 1)];

            // Horizontal pass
            for y in 0..h + 1 {
                let src_row = &src[y * src_stride..];
                let mid_row = &mut mid[y * mid_stride..][..w];

                for x in 0..w {
                    let s0 = src_row[x] as i32;
                    let s1 = src_row[x + 1] as i32;
                    mid_row[x] = h_coeff0 * s0 + h_coeff1 * s1;
                }
            }

            // Vertical pass
            for y in 0..h {
                let mid_row0 = &mid[y * mid_stride..][..w];
                let mid_row1 = &mid[(y + 1) * mid_stride..][..w];
                let tmp_row = &mut tmp[y * w..][..w];

                for x in 0..w {
                    let r0 = mid_row0[x];
                    let r1 = mid_row1[x];
                    let pixel = v_coeff0 * r0 + v_coeff1 * r1;
                    tmp_row[x] = ((pixel >> 8) - PREP_BIAS_16BPC) as i16;
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn prep_bilin_16bpc_neon(
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
    let w = w as usize;
    let h = h as usize;
    let src_stride_u16 = (src_stride / 2) as usize;

    // SAFETY: This function is only called through dispatch when NEON is available
    // Pointers are valid and properly aligned
    let (token, src, tmp_slice) = unsafe {
        let token = unsafe { Arm64::forge_token_dangerously() };
        let src =
            std::slice::from_raw_parts(src_ptr as *const u16, (h + 1) * src_stride_u16 + w + 1);
        let tmp_slice = std::slice::from_raw_parts_mut(tmp, h * w);
        (token, src, tmp_slice)
    };

    prep_bilin_16bpc_inner(token, tmp_slice, src, src_stride_u16, w, h, mx, my);
}

// ============================================================================
// W_MASK 16bpc - Weighted mask blend for 16-bit pixels
// ============================================================================

/// Core w_mask implementation for 16bpc
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn w_mask_16bpc_inner<const SS_HOR: bool, const SS_VER: bool>(
    dst: &mut [u16],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    mask: &mut [u8],
    sign: u8,
    bitdepth_max: i32,
) {
    // For 16bpc: intermediate_bits = 4
    let bitdepth = if bitdepth_max == 1023 { 10u32 } else { 12u32 };
    let intermediate_bits = 4i32;
    let sh = intermediate_bits + 6;
    let rnd = (32i32 << intermediate_bits) + 8192 * 64;
    let mask_sh = (bitdepth as i32 + intermediate_bits - 4) as u32;
    let mask_rnd = 1u16 << (mask_sh - 5);

    let mask_w = if SS_HOR { w >> 1 } else { w };

    for y in 0..h {
        let tmp1_row = &tmp1[y * w..][..w];
        let tmp2_row = &tmp2[y * w..][..w];
        let dst_row = &mut dst[y * dst_stride..][..w];
        let mut mask_row = if SS_VER && (y & 1) != 0 {
            None
        } else {
            let mask_y = if SS_VER { y >> 1 } else { y };
            Some(&mut mask[mask_y * mask_w..][..mask_w])
        };

        let mut col = 0;

        // Process pixels (scalar for now - SIMD could be added)
        while col < w {
            let t1 = tmp1_row[col] as i32;
            let t2 = tmp2_row[col] as i32;
            let diff = t1.abs_diff(t2) as u16;

            let m = std::cmp::min(38 + ((diff.saturating_add(mask_rnd)) >> mask_sh), 64) as u8;
            let m_final = if sign != 0 { 64 - m } else { m };
            let inv_m = 64 - m_final;

            let pixel = (t1 * m_final as i32 + t2 * inv_m as i32 + rnd) >> sh;
            dst_row[col] = pixel.clamp(0, bitdepth_max) as u16;

            if let Some(ref mut mask_row) = mask_row {
                if !SS_HOR {
                    mask_row[col] = m;
                } else if (col & 1) == 0 {
                    let mask_idx = col >> 1;
                    if col + 1 < w {
                        let t1_next = tmp1_row[col + 1] as i32;
                        let t2_next = tmp2_row[col + 1] as i32;
                        let diff_next = t1_next.abs_diff(t2_next) as u16;
                        let m_next = std::cmp::min(
                            38 + ((diff_next.saturating_add(mask_rnd)) >> mask_sh),
                            64,
                        ) as u8;
                        mask_row[mask_idx] = ((m as u16 + m_next as u16 + 1) >> 1) as u8;
                    } else {
                        mask_row[mask_idx] = m;
                    }
                }
            }

            col += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn w_mask_444_16bpc_neon(
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
    let w = w as usize;
    let h = h as usize;
    let dst_stride_u16 = (dst_stride / 2) as usize;
    // SAFETY: Pointers are valid and properly aligned
    let dst = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u16, h * dst_stride_u16) };

    w_mask_16bpc_inner::<false, false>(
        dst,
        dst_stride_u16,
        tmp1.as_slice(),
        tmp2.as_slice(),
        w,
        h,
        mask.as_mut_slice(),
        sign as u8,
        bitdepth_max,
    );
}

#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn w_mask_422_16bpc_neon(
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
    let w = w as usize;
    let h = h as usize;
    let dst_stride_u16 = (dst_stride / 2) as usize;
    // SAFETY: Pointers are valid and properly aligned
    let dst = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u16, h * dst_stride_u16) };

    w_mask_16bpc_inner::<true, false>(
        dst,
        dst_stride_u16,
        tmp1.as_slice(),
        tmp2.as_slice(),
        w,
        h,
        mask.as_mut_slice(),
        sign as u8,
        bitdepth_max,
    );
}

#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn w_mask_420_16bpc_neon(
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
    let w = w as usize;
    let h = h as usize;
    let dst_stride_u16 = (dst_stride / 2) as usize;
    // SAFETY: Pointers are valid and properly aligned
    let dst = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u16, h * dst_stride_u16) };

    w_mask_16bpc_inner::<true, true>(
        dst,
        dst_stride_u16,
        tmp1.as_slice(),
        tmp2.as_slice(),
        w,
        h,
        mask.as_mut_slice(),
        sign as u8,
        bitdepth_max,
    );
}

// ============================================================================
// 8-TAP FILTERS - Subpixel interpolation
// ============================================================================

const MID_STRIDE: usize = 128 + 16;

/// Get filter coefficients for subpixel position
/// Returns None if m is 0 (no subpixel offset)
fn get_filter_coeff(m: usize, d: usize, filter_type: Rav1dFilterMode) -> Option<&'static [i8; 8]> {
    let m = m.checked_sub(1)?;
    let i = if d > 4 {
        filter_type as u8
    } else {
        3 + (filter_type as u8 & 1)
    };
    Some(&dav1d_mc_subpel_filters.0[i as usize][m])
}

/// Horizontal 8-tap filter to intermediate buffer (i16)
/// For ARM NEON: uses multiply-accumulate with 8 coefficients
#[cfg(target_arch = "aarch64")]
#[arcane]
#[allow(clippy::too_many_arguments)]
fn h_filter_8tap_8bpc_neon(
    _token: Arm64,
    dst: &mut [i16],
    src: &[u8],
    w: usize,
    filter: &[i8; 8],
    sh: u8,
) {
    let rnd = (1i16 << sh) >> 1;

    let mut col = 0;

    // Process 8 pixels at a time with NEON
    while col + 8 <= w {
        // Load coefficients as i16
        let c0 = filter[0] as i16;
        let c1 = filter[1] as i16;
        let c2 = filter[2] as i16;
        let c3 = filter[3] as i16;
        let c4 = filter[4] as i16;
        let c5 = filter[5] as i16;
        let c6 = filter[6] as i16;
        let c7 = filter[7] as i16;

        // Load 8 source bytes at each tap position
        // src is already offset by -3 for tap 0
        let s0 = safe_simd::vld1_u8(src[col..][..8].try_into().unwrap());
        let s1 = safe_simd::vld1_u8(src[col + 1..][..8].try_into().unwrap());
        let s2 = safe_simd::vld1_u8(src[col + 2..][..8].try_into().unwrap());
        let s3 = safe_simd::vld1_u8(src[col + 3..][..8].try_into().unwrap());
        let s4 = safe_simd::vld1_u8(src[col + 4..][..8].try_into().unwrap());
        let s5 = safe_simd::vld1_u8(src[col + 5..][..8].try_into().unwrap());
        let s6 = safe_simd::vld1_u8(src[col + 6..][..8].try_into().unwrap());
        let s7 = safe_simd::vld1_u8(src[col + 7..][..8].try_into().unwrap());

        // Widen to i16
        let s0_16 = vreinterpretq_s16_u16(vmovl_u8(s0));
        let s1_16 = vreinterpretq_s16_u16(vmovl_u8(s1));
        let s2_16 = vreinterpretq_s16_u16(vmovl_u8(s2));
        let s3_16 = vreinterpretq_s16_u16(vmovl_u8(s3));
        let s4_16 = vreinterpretq_s16_u16(vmovl_u8(s4));
        let s5_16 = vreinterpretq_s16_u16(vmovl_u8(s5));
        let s6_16 = vreinterpretq_s16_u16(vmovl_u8(s6));
        let s7_16 = vreinterpretq_s16_u16(vmovl_u8(s7));

        // Multiply-accumulate each tap
        let mut sum = vmulq_n_s16(s0_16, c0);
        sum = vmlaq_n_s16(sum, s1_16, c1);
        sum = vmlaq_n_s16(sum, s2_16, c2);
        sum = vmlaq_n_s16(sum, s3_16, c3);
        sum = vmlaq_n_s16(sum, s4_16, c4);
        sum = vmlaq_n_s16(sum, s5_16, c5);
        sum = vmlaq_n_s16(sum, s6_16, c6);
        sum = vmlaq_n_s16(sum, s7_16, c7);

        // Add rounding and shift
        let rnd_vec = vdupq_n_s16(rnd);
        let result = vshrq_n_s16::<2>(vaddq_s16(sum, rnd_vec)); // sh = 2 for intermediate_bits = 4

        // Store to intermediate buffer
        let dst_arr: &mut [i16; 8] = (&mut dst[col..col + 8]).try_into().unwrap();
        safe_simd::vst1q_s16(dst_arr, result);
        col += 8;
    }

    // Scalar fallback for remaining pixels
    while col < w {
        let mut sum = 0i32;
        for i in 0..8 {
            sum += filter[i] as i32 * src[col + i] as i32;
        }
        dst[col] = ((sum + (rnd as i32)) >> sh) as i16;
        col += 1;
    }
}

/// Vertical 8-tap filter from intermediate buffer to output (u8)
#[cfg(target_arch = "aarch64")]
#[arcane]
#[allow(clippy::too_many_arguments)]
fn v_filter_8tap_8bpc_neon(
    _token: Arm64,
    dst: &mut [u8],
    mid: &[[i16; MID_STRIDE]],
    w: usize,
    filter: &[i8; 8],
    sh: u8,
    max: u16,
) {
    let rnd = (1i32 << sh) >> 1;
    let _ = max; // Unused for 8bpc, always 255

    let mut col = 0;

    // Process 8 pixels at a time with NEON
    while col + 8 <= w {
        // Load coefficients as i32 for wider accumulation
        let c0 = filter[0] as i32;
        let c1 = filter[1] as i32;
        let c2 = filter[2] as i32;
        let c3 = filter[3] as i32;
        let c4 = filter[4] as i32;
        let c5 = filter[5] as i32;
        let c6 = filter[6] as i32;
        let c7 = filter[7] as i32;

        // Load 8 rows from intermediate buffer
        let r0 = safe_simd::vld1q_s16(mid[0][col..][..8].try_into().unwrap());
        let r1 = safe_simd::vld1q_s16(mid[1][col..][..8].try_into().unwrap());
        let r2 = safe_simd::vld1q_s16(mid[2][col..][..8].try_into().unwrap());
        let r3 = safe_simd::vld1q_s16(mid[3][col..][..8].try_into().unwrap());
        let r4 = safe_simd::vld1q_s16(mid[4][col..][..8].try_into().unwrap());
        let r5 = safe_simd::vld1q_s16(mid[5][col..][..8].try_into().unwrap());
        let r6 = safe_simd::vld1q_s16(mid[6][col..][..8].try_into().unwrap());
        let r7 = safe_simd::vld1q_s16(mid[7][col..][..8].try_into().unwrap());

        // Process low 4 and high 4 separately for i32 accumulation
        let r0_lo = vmovl_s16(vget_low_s16(r0));
        let r0_hi = vmovl_s16(vget_high_s16(r0));
        let r1_lo = vmovl_s16(vget_low_s16(r1));
        let r1_hi = vmovl_s16(vget_high_s16(r1));
        let r2_lo = vmovl_s16(vget_low_s16(r2));
        let r2_hi = vmovl_s16(vget_high_s16(r2));
        let r3_lo = vmovl_s16(vget_low_s16(r3));
        let r3_hi = vmovl_s16(vget_high_s16(r3));
        let r4_lo = vmovl_s16(vget_low_s16(r4));
        let r4_hi = vmovl_s16(vget_high_s16(r4));
        let r5_lo = vmovl_s16(vget_low_s16(r5));
        let r5_hi = vmovl_s16(vget_high_s16(r5));
        let r6_lo = vmovl_s16(vget_low_s16(r6));
        let r6_hi = vmovl_s16(vget_high_s16(r6));
        let r7_lo = vmovl_s16(vget_low_s16(r7));
        let r7_hi = vmovl_s16(vget_high_s16(r7));

        // Multiply-accumulate each tap
        let mut sum_lo = vmulq_n_s32(r0_lo, c0);
        sum_lo = vmlaq_n_s32(sum_lo, r1_lo, c1);
        sum_lo = vmlaq_n_s32(sum_lo, r2_lo, c2);
        sum_lo = vmlaq_n_s32(sum_lo, r3_lo, c3);
        sum_lo = vmlaq_n_s32(sum_lo, r4_lo, c4);
        sum_lo = vmlaq_n_s32(sum_lo, r5_lo, c5);
        sum_lo = vmlaq_n_s32(sum_lo, r6_lo, c6);
        sum_lo = vmlaq_n_s32(sum_lo, r7_lo, c7);

        let mut sum_hi = vmulq_n_s32(r0_hi, c0);
        sum_hi = vmlaq_n_s32(sum_hi, r1_hi, c1);
        sum_hi = vmlaq_n_s32(sum_hi, r2_hi, c2);
        sum_hi = vmlaq_n_s32(sum_hi, r3_hi, c3);
        sum_hi = vmlaq_n_s32(sum_hi, r4_hi, c4);
        sum_hi = vmlaq_n_s32(sum_hi, r5_hi, c5);
        sum_hi = vmlaq_n_s32(sum_hi, r6_hi, c6);
        sum_hi = vmlaq_n_s32(sum_hi, r7_hi, c7);

        // Add rounding
        let rnd_vec = vdupq_n_s32(rnd);
        sum_lo = vaddq_s32(sum_lo, rnd_vec);
        sum_hi = vaddq_s32(sum_hi, rnd_vec);

        // Shift right (sh = 10 = 6 + intermediate_bits)
        let result_lo = vshrq_n_s32::<10>(sum_lo);
        let result_hi = vshrq_n_s32::<10>(sum_hi);

        // Narrow to i16
        let result_16 = vcombine_s16(vqmovn_s32(result_lo), vqmovn_s32(result_hi));

        // Narrow to u8 with saturation
        let result_8 = vqmovun_s16(result_16);

        // Store
        let dst_arr: &mut [u8; 8] = (&mut dst[col..col + 8]).try_into().unwrap();
        safe_simd::vst1_u8(dst_arr, result_8);
        col += 8;
    }

    // Scalar fallback
    while col < w {
        let mut sum = 0i32;
        for i in 0..8 {
            sum += filter[i] as i32 * mid[i][col] as i32;
        }
        dst[col] = ((sum + rnd) >> sh).clamp(0, 255) as u8;
        col += 1;
    }
}

/// Horizontal 8-tap filter directly to output (H-only case)
#[cfg(target_arch = "aarch64")]
#[arcane]
fn h_filter_8tap_8bpc_put_neon(
    _token: Arm64,
    dst: &mut [u8],
    src: &[u8],
    w: usize,
    filter: &[i8; 8],
) {
    let mut col = 0;

    while col + 8 <= w {
        let c0 = filter[0] as i16;
        let c1 = filter[1] as i16;
        let c2 = filter[2] as i16;
        let c3 = filter[3] as i16;
        let c4 = filter[4] as i16;
        let c5 = filter[5] as i16;
        let c6 = filter[6] as i16;
        let c7 = filter[7] as i16;

        let s0 = safe_simd::vld1_u8(src[col..][..8].try_into().unwrap());
        let s1 = safe_simd::vld1_u8(src[col + 1..][..8].try_into().unwrap());
        let s2 = safe_simd::vld1_u8(src[col + 2..][..8].try_into().unwrap());
        let s3 = safe_simd::vld1_u8(src[col + 3..][..8].try_into().unwrap());
        let s4 = safe_simd::vld1_u8(src[col + 4..][..8].try_into().unwrap());
        let s5 = safe_simd::vld1_u8(src[col + 5..][..8].try_into().unwrap());
        let s6 = safe_simd::vld1_u8(src[col + 6..][..8].try_into().unwrap());
        let s7 = safe_simd::vld1_u8(src[col + 7..][..8].try_into().unwrap());

        let s0_16 = vreinterpretq_s16_u16(vmovl_u8(s0));
        let s1_16 = vreinterpretq_s16_u16(vmovl_u8(s1));
        let s2_16 = vreinterpretq_s16_u16(vmovl_u8(s2));
        let s3_16 = vreinterpretq_s16_u16(vmovl_u8(s3));
        let s4_16 = vreinterpretq_s16_u16(vmovl_u8(s4));
        let s5_16 = vreinterpretq_s16_u16(vmovl_u8(s5));
        let s6_16 = vreinterpretq_s16_u16(vmovl_u8(s6));
        let s7_16 = vreinterpretq_s16_u16(vmovl_u8(s7));

        let mut sum = vmulq_n_s16(s0_16, c0);
        sum = vmlaq_n_s16(sum, s1_16, c1);
        sum = vmlaq_n_s16(sum, s2_16, c2);
        sum = vmlaq_n_s16(sum, s3_16, c3);
        sum = vmlaq_n_s16(sum, s4_16, c4);
        sum = vmlaq_n_s16(sum, s5_16, c5);
        sum = vmlaq_n_s16(sum, s6_16, c6);
        sum = vmlaq_n_s16(sum, s7_16, c7);

        // Round and shift for direct output: (sum + 32) >> 6
        let rnd_vec = vdupq_n_s16(32);
        let result = vshrq_n_s16::<6>(vaddq_s16(sum, rnd_vec));

        let result_8 = vqmovun_s16(result);
        let dst_arr: &mut [u8; 8] = (&mut dst[col..col + 8]).try_into().unwrap();
        safe_simd::vst1_u8(dst_arr, result_8);
        col += 8;
    }

    // Scalar fallback
    while col < w {
        let mut sum = 0i32;
        for i in 0..8 {
            sum += filter[i] as i32 * src[col + i] as i32;
        }
        dst[col] = ((sum + 32) >> 6).clamp(0, 255) as u8;
        col += 1;
    }
}

/// Vertical 8-tap filter directly from source (V-only case)
#[cfg(target_arch = "aarch64")]
#[arcane]
#[allow(clippy::too_many_arguments)]
fn v_filter_8tap_8bpc_direct_neon(
    _token: Arm64,
    dst: &mut [u8],
    src: &[u8],
    src_stride: usize,
    w: usize,
    filter: &[i8; 8],
) {
    let mut col = 0;

    while col + 8 <= w {
        let c0 = filter[0] as i32;
        let c1 = filter[1] as i32;
        let c2 = filter[2] as i32;
        let c3 = filter[3] as i32;
        let c4 = filter[4] as i32;
        let c5 = filter[5] as i32;
        let c6 = filter[6] as i32;
        let c7 = filter[7] as i32;

        // Load 8 source rows (src is already offset by -3 rows)
        let r0 = safe_simd::vld1_u8(src[col..][..8].try_into().unwrap());
        let r1 = safe_simd::vld1_u8(src[col + src_stride..][..8].try_into().unwrap());
        let r2 = safe_simd::vld1_u8(src[col + 2 * src_stride..][..8].try_into().unwrap());
        let r3 = safe_simd::vld1_u8(src[col + 3 * src_stride..][..8].try_into().unwrap());
        let r4 = safe_simd::vld1_u8(src[col + 4 * src_stride..][..8].try_into().unwrap());
        let r5 = safe_simd::vld1_u8(src[col + 5 * src_stride..][..8].try_into().unwrap());
        let r6 = safe_simd::vld1_u8(src[col + 6 * src_stride..][..8].try_into().unwrap());
        let r7 = safe_simd::vld1_u8(src[col + 7 * src_stride..][..8].try_into().unwrap());

        // Widen to i32 for accumulation
        let r0_16 = vreinterpretq_s16_u16(vmovl_u8(r0));
        let r1_16 = vreinterpretq_s16_u16(vmovl_u8(r1));
        let r2_16 = vreinterpretq_s16_u16(vmovl_u8(r2));
        let r3_16 = vreinterpretq_s16_u16(vmovl_u8(r3));
        let r4_16 = vreinterpretq_s16_u16(vmovl_u8(r4));
        let r5_16 = vreinterpretq_s16_u16(vmovl_u8(r5));
        let r6_16 = vreinterpretq_s16_u16(vmovl_u8(r6));
        let r7_16 = vreinterpretq_s16_u16(vmovl_u8(r7));

        // Low 4 pixels
        let r0_lo = vmovl_s16(vget_low_s16(r0_16));
        let r1_lo = vmovl_s16(vget_low_s16(r1_16));
        let r2_lo = vmovl_s16(vget_low_s16(r2_16));
        let r3_lo = vmovl_s16(vget_low_s16(r3_16));
        let r4_lo = vmovl_s16(vget_low_s16(r4_16));
        let r5_lo = vmovl_s16(vget_low_s16(r5_16));
        let r6_lo = vmovl_s16(vget_low_s16(r6_16));
        let r7_lo = vmovl_s16(vget_low_s16(r7_16));

        // High 4 pixels
        let r0_hi = vmovl_s16(vget_high_s16(r0_16));
        let r1_hi = vmovl_s16(vget_high_s16(r1_16));
        let r2_hi = vmovl_s16(vget_high_s16(r2_16));
        let r3_hi = vmovl_s16(vget_high_s16(r3_16));
        let r4_hi = vmovl_s16(vget_high_s16(r4_16));
        let r5_hi = vmovl_s16(vget_high_s16(r5_16));
        let r6_hi = vmovl_s16(vget_high_s16(r6_16));
        let r7_hi = vmovl_s16(vget_high_s16(r7_16));

        let mut sum_lo = vmulq_n_s32(r0_lo, c0);
        sum_lo = vmlaq_n_s32(sum_lo, r1_lo, c1);
        sum_lo = vmlaq_n_s32(sum_lo, r2_lo, c2);
        sum_lo = vmlaq_n_s32(sum_lo, r3_lo, c3);
        sum_lo = vmlaq_n_s32(sum_lo, r4_lo, c4);
        sum_lo = vmlaq_n_s32(sum_lo, r5_lo, c5);
        sum_lo = vmlaq_n_s32(sum_lo, r6_lo, c6);
        sum_lo = vmlaq_n_s32(sum_lo, r7_lo, c7);

        let mut sum_hi = vmulq_n_s32(r0_hi, c0);
        sum_hi = vmlaq_n_s32(sum_hi, r1_hi, c1);
        sum_hi = vmlaq_n_s32(sum_hi, r2_hi, c2);
        sum_hi = vmlaq_n_s32(sum_hi, r3_hi, c3);
        sum_hi = vmlaq_n_s32(sum_hi, r4_hi, c4);
        sum_hi = vmlaq_n_s32(sum_hi, r5_hi, c5);
        sum_hi = vmlaq_n_s32(sum_hi, r6_hi, c6);
        sum_hi = vmlaq_n_s32(sum_hi, r7_hi, c7);

        // Round and shift: (sum + 32) >> 6
        let rnd_vec = vdupq_n_s32(32);
        sum_lo = vshrq_n_s32::<6>(vaddq_s32(sum_lo, rnd_vec));
        sum_hi = vshrq_n_s32::<6>(vaddq_s32(sum_hi, rnd_vec));

        let result_16 = vcombine_s16(vqmovn_s32(sum_lo), vqmovn_s32(sum_hi));
        let result_8 = vqmovun_s16(result_16);
        let dst_arr: &mut [u8; 8] = (&mut dst[col..col + 8]).try_into().unwrap();
        safe_simd::vst1_u8(dst_arr, result_8);
        col += 8;
    }

    // Scalar fallback
    while col < w {
        let mut sum = 0i32;
        for i in 0..8 {
            sum += filter[i] as i32 * src[col + i * src_stride] as i32;
        }
        dst[col] = ((sum + 32) >> 6).clamp(0, 255) as u8;
        col += 1;
    }
}

/// Main 8-tap put function for 8bpc
#[cfg(target_arch = "aarch64")]
#[arcane]
#[allow(clippy::too_many_arguments)]
fn put_8tap_8bpc_inner(
    token: Arm64,
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: usize,
    my: usize,
    h_filter_type: Rav1dFilterMode,
    v_filter_type: Rav1dFilterMode,
) {
    let intermediate_bits = 4u8;

    let fh = get_filter_coeff(mx, w, h_filter_type);
    let fv = get_filter_coeff(my, h, v_filter_type);

    match (fh, fv) {
        (Some(fh), Some(fv)) => {
            // Case 1: Both H and V filtering
            let tmp_h = h + 7;
            let mut mid = [[0i16; MID_STRIDE]; 135];

            for y in 0..tmp_h {
                let src_offset = if y >= 3 {
                    (y - 3) * src_stride
                } else {
                    0usize.wrapping_sub((3 - y) * src_stride)
                };
                let src_row = &src[src_offset..];
                h_filter_8tap_8bpc_neon(
                    token,
                    &mut mid[y][..w],
                    src_row,
                    w,
                    fh,
                    6 - intermediate_bits,
                );
            }

            for y in 0..h {
                let dst_row = &mut dst[y * dst_stride..][..w];
                v_filter_8tap_8bpc_neon(
                    token,
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
            for y in 0..h {
                let src_row = &src[y * src_stride..];
                let dst_row = &mut dst[y * dst_stride..][..w];
                h_filter_8tap_8bpc_put_neon(token, dst_row, src_row, w, fh);
            }
        }
        (None, Some(fv)) => {
            // Case 3: V-only filtering
            for y in 0..h {
                let src_offset = if y >= 3 {
                    (y - 3) * src_stride
                } else {
                    0usize.wrapping_sub((3 - y) * src_stride)
                };
                let src_row = &src[src_offset..];
                let dst_row = &mut dst[y * dst_stride..][..w];
                v_filter_8tap_8bpc_direct_neon(token, dst_row, src_row, src_stride, w, fv);
            }
        }
        (None, None) => {
            // Case 4: Simple copy
            for y in 0..h {
                let src_row = &src[y * src_stride..][..w];
                let dst_row = &mut dst[y * dst_stride..][..w];
                dst_row.copy_from_slice(src_row);
            }
        }
    }
}

// ============================================================================
// 8-TAP FFI WRAPPERS
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn get_h_filter_type(filter: Filter2d) -> Rav1dFilterMode {
    match filter {
        Filter2d::Regular8Tap | Filter2d::RegularSmooth8Tap | Filter2d::RegularSharp8Tap => {
            Rav1dFilterMode::Regular8Tap
        }
        Filter2d::Smooth8Tap | Filter2d::SmoothRegular8Tap | Filter2d::SmoothSharp8Tap => {
            Rav1dFilterMode::Smooth8Tap
        }
        Filter2d::Sharp8Tap | Filter2d::SharpRegular8Tap | Filter2d::SharpSmooth8Tap => {
            Rav1dFilterMode::Sharp8Tap
        }
        Filter2d::Bilinear => Rav1dFilterMode::Regular8Tap, // fallback
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn get_v_filter_type(filter: Filter2d) -> Rav1dFilterMode {
    match filter {
        Filter2d::Regular8Tap | Filter2d::SmoothRegular8Tap | Filter2d::SharpRegular8Tap => {
            Rav1dFilterMode::Regular8Tap
        }
        Filter2d::Smooth8Tap | Filter2d::RegularSmooth8Tap | Filter2d::SharpSmooth8Tap => {
            Rav1dFilterMode::Smooth8Tap
        }
        Filter2d::Sharp8Tap | Filter2d::RegularSharp8Tap | Filter2d::SmoothSharp8Tap => {
            Rav1dFilterMode::Sharp8Tap
        }
        Filter2d::Bilinear => Rav1dFilterMode::Regular8Tap, // fallback
    }
}

macro_rules! define_put_8tap_8bpc {
    ($name:ident, $filter:expr) => {
        #[cfg(target_arch = "aarch64")]
        pub unsafe extern "C" fn $name(
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
            let token = unsafe { Arm64::forge_token_dangerously() };
            let w = w as usize;
            let h = h as usize;
            let mx = mx as usize;
            let my = my as usize;
            let dst_stride_u = dst_stride as usize;
            let src_stride_u = src_stride as usize;

            // Offset source pointer back by 3 pixels and 3 rows for filter taps
            let src_base = (src_ptr as *const u8).offset(-3 * src_stride - 3);

            // Create slices
            let src_len = (h + 7) * src_stride_u + w + 7;
            let src = std::slice::from_raw_parts(src_base, src_len);

            let dst_len = h * dst_stride_u + w;
            let dst = std::slice::from_raw_parts_mut(dst_ptr as *mut u8, dst_len);

            // Adjust source slice to account for the -3,-3 offset we added
            let src_adjusted = &src[3 * src_stride_u + 3..];

            put_8tap_8bpc_inner(
                token,
                dst,
                dst_stride_u,
                src_adjusted,
                src_stride_u,
                w,
                h,
                mx,
                my,
                get_h_filter_type($filter),
                get_v_filter_type($filter),
            );
        }
    };
}

define_put_8tap_8bpc!(put_8tap_regular_8bpc_neon, Filter2d::Regular8Tap);
define_put_8tap_8bpc!(
    put_8tap_regular_smooth_8bpc_neon,
    Filter2d::RegularSmooth8Tap
);
define_put_8tap_8bpc!(put_8tap_regular_sharp_8bpc_neon, Filter2d::RegularSharp8Tap);
define_put_8tap_8bpc!(
    put_8tap_smooth_regular_8bpc_neon,
    Filter2d::SmoothRegular8Tap
);
define_put_8tap_8bpc!(put_8tap_smooth_8bpc_neon, Filter2d::Smooth8Tap);
define_put_8tap_8bpc!(put_8tap_smooth_sharp_8bpc_neon, Filter2d::SmoothSharp8Tap);
define_put_8tap_8bpc!(put_8tap_sharp_regular_8bpc_neon, Filter2d::SharpRegular8Tap);
define_put_8tap_8bpc!(put_8tap_sharp_smooth_8bpc_neon, Filter2d::SharpSmooth8Tap);
define_put_8tap_8bpc!(put_8tap_sharp_8bpc_neon, Filter2d::Sharp8Tap);

// ============================================================================
// 8-TAP PREP (to intermediate i16 buffer)
// ============================================================================

/// Vertical 8-tap filter from intermediate buffer to i16 output
#[cfg(target_arch = "aarch64")]
#[arcane]
#[allow(clippy::too_many_arguments)]
fn v_filter_8tap_to_i16_neon(
    _token: Arm64,
    dst: &mut [i16],
    mid: &[[i16; MID_STRIDE]],
    w: usize,
    filter: &[i8; 8],
    sh: u8,
) {
    let rnd = (1i32 << sh) >> 1;

    let mut col = 0;

    while col + 4 <= w {
        let c0 = filter[0] as i32;
        let c1 = filter[1] as i32;
        let c2 = filter[2] as i32;
        let c3 = filter[3] as i32;
        let c4 = filter[4] as i32;
        let c5 = filter[5] as i32;
        let c6 = filter[6] as i32;
        let c7 = filter[7] as i32;

        // Load 4 values from each of 8 rows
        let r0 = safe_simd::vld1_s16(mid[0][col..][..4].try_into().unwrap());
        let r1 = safe_simd::vld1_s16(mid[1][col..][..4].try_into().unwrap());
        let r2 = safe_simd::vld1_s16(mid[2][col..][..4].try_into().unwrap());
        let r3 = safe_simd::vld1_s16(mid[3][col..][..4].try_into().unwrap());
        let r4 = safe_simd::vld1_s16(mid[4][col..][..4].try_into().unwrap());
        let r5 = safe_simd::vld1_s16(mid[5][col..][..4].try_into().unwrap());
        let r6 = safe_simd::vld1_s16(mid[6][col..][..4].try_into().unwrap());
        let r7 = safe_simd::vld1_s16(mid[7][col..][..4].try_into().unwrap());

        // Widen to i32
        let r0_32 = vmovl_s16(r0);
        let r1_32 = vmovl_s16(r1);
        let r2_32 = vmovl_s16(r2);
        let r3_32 = vmovl_s16(r3);
        let r4_32 = vmovl_s16(r4);
        let r5_32 = vmovl_s16(r5);
        let r6_32 = vmovl_s16(r6);
        let r7_32 = vmovl_s16(r7);

        let mut sum = vmulq_n_s32(r0_32, c0);
        sum = vmlaq_n_s32(sum, r1_32, c1);
        sum = vmlaq_n_s32(sum, r2_32, c2);
        sum = vmlaq_n_s32(sum, r3_32, c3);
        sum = vmlaq_n_s32(sum, r4_32, c4);
        sum = vmlaq_n_s32(sum, r5_32, c5);
        sum = vmlaq_n_s32(sum, r6_32, c6);
        sum = vmlaq_n_s32(sum, r7_32, c7);

        let rnd_vec = vdupq_n_s32(rnd);
        sum = vshrq_n_s32::<10>(vaddq_s32(sum, rnd_vec));

        // Narrow to i16
        let result = vqmovn_s32(sum);
        let dst_arr: &mut [i16; 4] = (&mut dst[col..col + 4]).try_into().unwrap();
        safe_simd::vst1_s16(dst_arr, result);
        col += 4;
    }

    // Scalar fallback
    while col < w {
        let mut sum = 0i32;
        for i in 0..8 {
            sum += filter[i] as i32 * mid[i][col] as i32;
        }
        dst[col] = ((sum + rnd) >> sh) as i16;
        col += 1;
    }
}

/// Main 8-tap prep function for 8bpc
#[cfg(target_arch = "aarch64")]
#[arcane]
#[allow(clippy::too_many_arguments)]
fn prep_8tap_8bpc_inner(
    token: Arm64,
    tmp: &mut [i16],
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: usize,
    my: usize,
    h_filter_type: Rav1dFilterMode,
    v_filter_type: Rav1dFilterMode,
) {
    let intermediate_bits = 4u8;

    let fh = get_filter_coeff(mx, w, h_filter_type);
    let fv = get_filter_coeff(my, h, v_filter_type);

    match (fh, fv) {
        (Some(fh), Some(fv)) => {
            // Case 1: Both H and V filtering
            let tmp_h = h + 7;
            let mut mid = [[0i16; MID_STRIDE]; 135];

            for y in 0..tmp_h {
                let src_offset = if y >= 3 {
                    (y - 3) * src_stride
                } else {
                    0usize.wrapping_sub((3 - y) * src_stride)
                };
                let src_row = &src[src_offset..];
                h_filter_8tap_8bpc_neon(
                    token,
                    &mut mid[y][..w],
                    src_row,
                    w,
                    fh,
                    6 - intermediate_bits,
                );
            }

            for y in 0..h {
                let out_row = &mut tmp[y * w..][..w];
                v_filter_8tap_to_i16_neon(token, out_row, &mid[y..], w, fv, 6 + intermediate_bits);
            }
        }
        (Some(fh), None) => {
            // Case 2: H-only filtering
            for y in 0..h {
                let src_row = &src[y * src_stride..];
                let out_row = &mut tmp[y * w..][..w];
                // Output to i16 with intermediate_bits shift
                let mut col = 0;
                while col + 8 <= w {
                    let c0 = fh[0] as i16;
                    let c1 = fh[1] as i16;
                    let c2 = fh[2] as i16;
                    let c3 = fh[3] as i16;
                    let c4 = fh[4] as i16;
                    let c5 = fh[5] as i16;
                    let c6 = fh[6] as i16;
                    let c7 = fh[7] as i16;

                    let s0 = safe_simd::vld1_u8(src_row[col..][..8].try_into().unwrap());
                    let s1 = safe_simd::vld1_u8(src_row[col + 1..][..8].try_into().unwrap());
                    let s2 = safe_simd::vld1_u8(src_row[col + 2..][..8].try_into().unwrap());
                    let s3 = safe_simd::vld1_u8(src_row[col + 3..][..8].try_into().unwrap());
                    let s4 = safe_simd::vld1_u8(src_row[col + 4..][..8].try_into().unwrap());
                    let s5 = safe_simd::vld1_u8(src_row[col + 5..][..8].try_into().unwrap());
                    let s6 = safe_simd::vld1_u8(src_row[col + 6..][..8].try_into().unwrap());
                    let s7 = safe_simd::vld1_u8(src_row[col + 7..][..8].try_into().unwrap());

                    let s0_16 = vreinterpretq_s16_u16(vmovl_u8(s0));
                    let s1_16 = vreinterpretq_s16_u16(vmovl_u8(s1));
                    let s2_16 = vreinterpretq_s16_u16(vmovl_u8(s2));
                    let s3_16 = vreinterpretq_s16_u16(vmovl_u8(s3));
                    let s4_16 = vreinterpretq_s16_u16(vmovl_u8(s4));
                    let s5_16 = vreinterpretq_s16_u16(vmovl_u8(s5));
                    let s6_16 = vreinterpretq_s16_u16(vmovl_u8(s6));
                    let s7_16 = vreinterpretq_s16_u16(vmovl_u8(s7));

                    let mut sum = vmulq_n_s16(s0_16, c0);
                    sum = vmlaq_n_s16(sum, s1_16, c1);
                    sum = vmlaq_n_s16(sum, s2_16, c2);
                    sum = vmlaq_n_s16(sum, s3_16, c3);
                    sum = vmlaq_n_s16(sum, s4_16, c4);
                    sum = vmlaq_n_s16(sum, s5_16, c5);
                    sum = vmlaq_n_s16(sum, s6_16, c6);
                    sum = vmlaq_n_s16(sum, s7_16, c7);

                    // Shift for intermediate: (sum + 8) >> 4
                    let rnd_vec = vdupq_n_s16(8);
                    let result = vshrq_n_s16::<4>(vaddq_s16(sum, rnd_vec));

                    let out_arr: &mut [i16; 8] = (&mut out_row[col..col + 8]).try_into().unwrap();
                    safe_simd::vst1q_s16(out_arr, result);
                    col += 8;
                }
                while col < w {
                    let mut sum = 0i32;
                    for i in 0..8 {
                        sum += fh[i] as i32 * src_row[col + i] as i32;
                    }
                    out_row[col] = ((sum + 8) >> intermediate_bits) as i16;
                    col += 1;
                }
            }
        }
        (None, Some(fv)) => {
            // Case 3: V-only filtering
            for y in 0..h {
                let out_row = &mut tmp[y * w..][..w];

                let mut mid = [[0i16; MID_STRIDE]; 8];
                for i in 0..8 {
                    let src_offset = if y + i >= 3 {
                        (y + i - 3) * src_stride
                    } else {
                        0usize.wrapping_sub((3 - y - i) * src_stride)
                    };
                    for x in 0..w {
                        mid[i][x] = (src[src_offset + x] as i16) << intermediate_bits;
                    }
                }

                v_filter_8tap_to_i16_neon(token, out_row, &mid, w, fv, 6);
            }
        }
        (None, None) => {
            // Case 4: Simple copy with intermediate scaling
            for y in 0..h {
                let src_row = &src[y * src_stride..][..w];
                let out_row = &mut tmp[y * w..][..w];
                for x in 0..w {
                    out_row[x] = (src_row[x] as i16) << intermediate_bits;
                }
            }
        }
    }
}

macro_rules! define_prep_8tap_8bpc {
    ($name:ident, $filter:expr) => {
        #[cfg(target_arch = "aarch64")]
        pub unsafe extern "C" fn $name(
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
            let token = unsafe { Arm64::forge_token_dangerously() };
            let w = w as usize;
            let h = h as usize;
            let mx = mx as usize;
            let my = my as usize;
            let src_stride_u = src_stride as usize;

            // Offset source pointer back by 3 pixels and 3 rows
            let src_base = (src_ptr as *const u8).offset(-3 * src_stride - 3);

            let src_len = (h + 7) * src_stride_u + w + 7;
            let src = std::slice::from_raw_parts(src_base, src_len);

            let tmp_len = h * w;
            let tmp_slice = std::slice::from_raw_parts_mut(tmp, tmp_len);

            let src_adjusted = &src[3 * src_stride_u + 3..];

            prep_8tap_8bpc_inner(
                token,
                tmp_slice,
                src_adjusted,
                src_stride_u,
                w,
                h,
                mx,
                my,
                get_h_filter_type($filter),
                get_v_filter_type($filter),
            );
        }
    };
}

define_prep_8tap_8bpc!(prep_8tap_regular_8bpc_neon, Filter2d::Regular8Tap);
define_prep_8tap_8bpc!(
    prep_8tap_regular_smooth_8bpc_neon,
    Filter2d::RegularSmooth8Tap
);
define_prep_8tap_8bpc!(
    prep_8tap_regular_sharp_8bpc_neon,
    Filter2d::RegularSharp8Tap
);
define_prep_8tap_8bpc!(
    prep_8tap_smooth_regular_8bpc_neon,
    Filter2d::SmoothRegular8Tap
);
define_prep_8tap_8bpc!(prep_8tap_smooth_8bpc_neon, Filter2d::Smooth8Tap);
define_prep_8tap_8bpc!(prep_8tap_smooth_sharp_8bpc_neon, Filter2d::SmoothSharp8Tap);
define_prep_8tap_8bpc!(
    prep_8tap_sharp_regular_8bpc_neon,
    Filter2d::SharpRegular8Tap
);
define_prep_8tap_8bpc!(prep_8tap_sharp_smooth_8bpc_neon, Filter2d::SharpSmooth8Tap);
define_prep_8tap_8bpc!(prep_8tap_sharp_8bpc_neon, Filter2d::Sharp8Tap);

// ============================================================================
// 8-TAP FILTERS 16BPC
// ============================================================================

/// Horizontal 8-tap filter to intermediate buffer (i32) for 16bpc
#[cfg(target_arch = "aarch64")]
#[arcane]
#[allow(clippy::too_many_arguments)]
fn h_filter_8tap_16bpc_neon(
    _token: Arm64,
    dst: &mut [i32],
    src: &[u16],
    w: usize,
    filter: &[i8; 8],
    sh: u8,
) {
    let rnd = (1i32 << sh) >> 1;

    let mut col = 0;

    while col + 4 <= w {
        let c0 = filter[0] as i32;
        let c1 = filter[1] as i32;
        let c2 = filter[2] as i32;
        let c3 = filter[3] as i32;
        let c4 = filter[4] as i32;
        let c5 = filter[5] as i32;
        let c6 = filter[6] as i32;
        let c7 = filter[7] as i32;

        // Load 4 source u16 values at each tap position
        let s0 = safe_simd::vld1_u16(src[col..][..4].try_into().unwrap());
        let s1 = safe_simd::vld1_u16(src[col + 1..][..4].try_into().unwrap());
        let s2 = safe_simd::vld1_u16(src[col + 2..][..4].try_into().unwrap());
        let s3 = safe_simd::vld1_u16(src[col + 3..][..4].try_into().unwrap());
        let s4 = safe_simd::vld1_u16(src[col + 4..][..4].try_into().unwrap());
        let s5 = safe_simd::vld1_u16(src[col + 5..][..4].try_into().unwrap());
        let s6 = safe_simd::vld1_u16(src[col + 6..][..4].try_into().unwrap());
        let s7 = safe_simd::vld1_u16(src[col + 7..][..4].try_into().unwrap());

        // Widen to i32
        let s0_32 = vreinterpretq_s32_u32(vmovl_u16(s0));
        let s1_32 = vreinterpretq_s32_u32(vmovl_u16(s1));
        let s2_32 = vreinterpretq_s32_u32(vmovl_u16(s2));
        let s3_32 = vreinterpretq_s32_u32(vmovl_u16(s3));
        let s4_32 = vreinterpretq_s32_u32(vmovl_u16(s4));
        let s5_32 = vreinterpretq_s32_u32(vmovl_u16(s5));
        let s6_32 = vreinterpretq_s32_u32(vmovl_u16(s6));
        let s7_32 = vreinterpretq_s32_u32(vmovl_u16(s7));

        let mut sum = vmulq_n_s32(s0_32, c0);
        sum = vmlaq_n_s32(sum, s1_32, c1);
        sum = vmlaq_n_s32(sum, s2_32, c2);
        sum = vmlaq_n_s32(sum, s3_32, c3);
        sum = vmlaq_n_s32(sum, s4_32, c4);
        sum = vmlaq_n_s32(sum, s5_32, c5);
        sum = vmlaq_n_s32(sum, s6_32, c6);
        sum = vmlaq_n_s32(sum, s7_32, c7);

        // Add rounding and shift
        let rnd_vec = vdupq_n_s32(rnd);
        let result = vshrq_n_s32::<4>(vaddq_s32(sum, rnd_vec)); // sh typically 4 for 16bpc intermediate

        let dst_arr: &mut [i32; 4] = (&mut dst[col..col + 4]).try_into().unwrap();
        safe_simd::vst1q_s32(dst_arr, result);
        col += 4;
    }

    // Scalar fallback
    while col < w {
        let mut sum = 0i32;
        for i in 0..8 {
            sum += filter[i] as i32 * src[col + i] as i32;
        }
        dst[col] = (sum + rnd) >> sh;
        col += 1;
    }
}

/// Vertical 8-tap filter from i32 intermediate to u16 output for 16bpc
#[cfg(target_arch = "aarch64")]
#[arcane]
#[allow(clippy::too_many_arguments)]
fn v_filter_8tap_16bpc_neon(
    _token: Arm64,
    dst: &mut [u16],
    mid: &[[i32; MID_STRIDE]],
    w: usize,
    filter: &[i8; 8],
    sh: u8,
    max: u16,
) {
    let rnd = (1i32 << sh) >> 1;

    let mut col = 0;

    while col + 4 <= w {
        let c0 = filter[0] as i32;
        let c1 = filter[1] as i32;
        let c2 = filter[2] as i32;
        let c3 = filter[3] as i32;
        let c4 = filter[4] as i32;
        let c5 = filter[5] as i32;
        let c6 = filter[6] as i32;
        let c7 = filter[7] as i32;

        let r0 = safe_simd::vld1q_s32(mid[0][col..][..4].try_into().unwrap());
        let r1 = safe_simd::vld1q_s32(mid[1][col..][..4].try_into().unwrap());
        let r2 = safe_simd::vld1q_s32(mid[2][col..][..4].try_into().unwrap());
        let r3 = safe_simd::vld1q_s32(mid[3][col..][..4].try_into().unwrap());
        let r4 = safe_simd::vld1q_s32(mid[4][col..][..4].try_into().unwrap());
        let r5 = safe_simd::vld1q_s32(mid[5][col..][..4].try_into().unwrap());
        let r6 = safe_simd::vld1q_s32(mid[6][col..][..4].try_into().unwrap());
        let r7 = safe_simd::vld1q_s32(mid[7][col..][..4].try_into().unwrap());

        let mut sum = vmulq_n_s32(r0, c0);
        sum = vmlaq_n_s32(sum, r1, c1);
        sum = vmlaq_n_s32(sum, r2, c2);
        sum = vmlaq_n_s32(sum, r3, c3);
        sum = vmlaq_n_s32(sum, r4, c4);
        sum = vmlaq_n_s32(sum, r5, c5);
        sum = vmlaq_n_s32(sum, r6, c6);
        sum = vmlaq_n_s32(sum, r7, c7);

        // Add rounding
        let rnd_vec = vdupq_n_s32(rnd);
        sum = vaddq_s32(sum, rnd_vec);

        // Shift - typically sh = 8 for 16bpc (4 + 4)
        let result = vshrq_n_s32::<8>(sum);

        // Clamp to [0, max]
        let max_vec = vdupq_n_s32(max as i32);
        let zero = vdupq_n_s32(0);
        let clamped = vminq_s32(vmaxq_s32(result, zero), max_vec);

        // Narrow to u16
        let narrow = vqmovun_s32(clamped);
        let dst_arr: &mut [u16; 4] = (&mut dst[col..col + 4]).try_into().unwrap();
        safe_simd::vst1_u16(dst_arr, narrow);
        col += 4;
    }

    // Scalar fallback
    while col < w {
        let mut sum = 0i64;
        for i in 0..8 {
            sum += filter[i] as i64 * mid[i][col] as i64;
        }
        dst[col] = (((sum + rnd as i64) >> sh) as i32).clamp(0, max as i32) as u16;
        col += 1;
    }
}

/// Horizontal 8-tap filter directly to output for 16bpc (H-only case)
#[cfg(target_arch = "aarch64")]
#[arcane]
#[allow(clippy::too_many_arguments)]
fn h_filter_8tap_16bpc_put_neon(
    _token: Arm64,
    dst: &mut [u16],
    src: &[u16],
    w: usize,
    filter: &[i8; 8],
    max: u16,
) {
    let mut col = 0;

    while col + 4 <= w {
        let c0 = filter[0] as i32;
        let c1 = filter[1] as i32;
        let c2 = filter[2] as i32;
        let c3 = filter[3] as i32;
        let c4 = filter[4] as i32;
        let c5 = filter[5] as i32;
        let c6 = filter[6] as i32;
        let c7 = filter[7] as i32;

        let s0 = safe_simd::vld1_u16(src[col..][..4].try_into().unwrap());
        let s1 = safe_simd::vld1_u16(src[col + 1..][..4].try_into().unwrap());
        let s2 = safe_simd::vld1_u16(src[col + 2..][..4].try_into().unwrap());
        let s3 = safe_simd::vld1_u16(src[col + 3..][..4].try_into().unwrap());
        let s4 = safe_simd::vld1_u16(src[col + 4..][..4].try_into().unwrap());
        let s5 = safe_simd::vld1_u16(src[col + 5..][..4].try_into().unwrap());
        let s6 = safe_simd::vld1_u16(src[col + 6..][..4].try_into().unwrap());
        let s7 = safe_simd::vld1_u16(src[col + 7..][..4].try_into().unwrap());

        let s0_32 = vreinterpretq_s32_u32(vmovl_u16(s0));
        let s1_32 = vreinterpretq_s32_u32(vmovl_u16(s1));
        let s2_32 = vreinterpretq_s32_u32(vmovl_u16(s2));
        let s3_32 = vreinterpretq_s32_u32(vmovl_u16(s3));
        let s4_32 = vreinterpretq_s32_u32(vmovl_u16(s4));
        let s5_32 = vreinterpretq_s32_u32(vmovl_u16(s5));
        let s6_32 = vreinterpretq_s32_u32(vmovl_u16(s6));
        let s7_32 = vreinterpretq_s32_u32(vmovl_u16(s7));

        let mut sum = vmulq_n_s32(s0_32, c0);
        sum = vmlaq_n_s32(sum, s1_32, c1);
        sum = vmlaq_n_s32(sum, s2_32, c2);
        sum = vmlaq_n_s32(sum, s3_32, c3);
        sum = vmlaq_n_s32(sum, s4_32, c4);
        sum = vmlaq_n_s32(sum, s5_32, c5);
        sum = vmlaq_n_s32(sum, s6_32, c6);
        sum = vmlaq_n_s32(sum, s7_32, c7);

        // Round and shift: (sum + 32) >> 6
        let rnd_vec = vdupq_n_s32(32);
        let result = vshrq_n_s32::<6>(vaddq_s32(sum, rnd_vec));

        // Clamp
        let max_vec = vdupq_n_s32(max as i32);
        let zero = vdupq_n_s32(0);
        let clamped = vminq_s32(vmaxq_s32(result, zero), max_vec);

        let narrow = vqmovun_s32(clamped);
        let dst_arr: &mut [u16; 4] = (&mut dst[col..col + 4]).try_into().unwrap();
        safe_simd::vst1_u16(dst_arr, narrow);
        col += 4;
    }

    // Scalar fallback
    while col < w {
        let mut sum = 0i32;
        for i in 0..8 {
            sum += filter[i] as i32 * src[col + i] as i32;
        }
        dst[col] = ((sum + 32) >> 6).clamp(0, max as i32) as u16;
        col += 1;
    }
}

/// Vertical 8-tap filter directly from source for 16bpc (V-only case)
#[cfg(target_arch = "aarch64")]
#[arcane]
#[allow(clippy::too_many_arguments)]
fn v_filter_8tap_16bpc_direct_neon(
    _token: Arm64,
    dst: &mut [u16],
    src: &[u16],
    src_stride: usize,
    w: usize,
    filter: &[i8; 8],
    max: u16,
) {
    let mut col = 0;

    while col + 4 <= w {
        let c0 = filter[0] as i32;
        let c1 = filter[1] as i32;
        let c2 = filter[2] as i32;
        let c3 = filter[3] as i32;
        let c4 = filter[4] as i32;
        let c5 = filter[5] as i32;
        let c6 = filter[6] as i32;
        let c7 = filter[7] as i32;

        let r0 = safe_simd::vld1_u16(src[col..][..4].try_into().unwrap());
        let r1 = safe_simd::vld1_u16(src[col + src_stride..][..4].try_into().unwrap());
        let r2 = safe_simd::vld1_u16(src[col + 2 * src_stride..][..4].try_into().unwrap());
        let r3 = safe_simd::vld1_u16(src[col + 3 * src_stride..][..4].try_into().unwrap());
        let r4 = safe_simd::vld1_u16(src[col + 4 * src_stride..][..4].try_into().unwrap());
        let r5 = safe_simd::vld1_u16(src[col + 5 * src_stride..][..4].try_into().unwrap());
        let r6 = safe_simd::vld1_u16(src[col + 6 * src_stride..][..4].try_into().unwrap());
        let r7 = safe_simd::vld1_u16(src[col + 7 * src_stride..][..4].try_into().unwrap());

        let r0_32 = vreinterpretq_s32_u32(vmovl_u16(r0));
        let r1_32 = vreinterpretq_s32_u32(vmovl_u16(r1));
        let r2_32 = vreinterpretq_s32_u32(vmovl_u16(r2));
        let r3_32 = vreinterpretq_s32_u32(vmovl_u16(r3));
        let r4_32 = vreinterpretq_s32_u32(vmovl_u16(r4));
        let r5_32 = vreinterpretq_s32_u32(vmovl_u16(r5));
        let r6_32 = vreinterpretq_s32_u32(vmovl_u16(r6));
        let r7_32 = vreinterpretq_s32_u32(vmovl_u16(r7));

        let mut sum = vmulq_n_s32(r0_32, c0);
        sum = vmlaq_n_s32(sum, r1_32, c1);
        sum = vmlaq_n_s32(sum, r2_32, c2);
        sum = vmlaq_n_s32(sum, r3_32, c3);
        sum = vmlaq_n_s32(sum, r4_32, c4);
        sum = vmlaq_n_s32(sum, r5_32, c5);
        sum = vmlaq_n_s32(sum, r6_32, c6);
        sum = vmlaq_n_s32(sum, r7_32, c7);

        // Round and shift
        let rnd_vec = vdupq_n_s32(32);
        sum = vshrq_n_s32::<6>(vaddq_s32(sum, rnd_vec));

        // Clamp
        let max_vec = vdupq_n_s32(max as i32);
        let zero = vdupq_n_s32(0);
        let clamped = vminq_s32(vmaxq_s32(sum, zero), max_vec);

        let narrow = vqmovun_s32(clamped);
        let dst_arr: &mut [u16; 4] = (&mut dst[col..col + 4]).try_into().unwrap();
        safe_simd::vst1_u16(dst_arr, narrow);
        col += 4;
    }

    // Scalar fallback
    while col < w {
        let mut sum = 0i32;
        for i in 0..8 {
            sum += filter[i] as i32 * src[col + i * src_stride] as i32;
        }
        dst[col] = ((sum + 32) >> 6).clamp(0, max as i32) as u16;
        col += 1;
    }
}

/// Main 8-tap put function for 16bpc
#[cfg(target_arch = "aarch64")]
#[arcane]
#[allow(clippy::too_many_arguments)]
fn put_8tap_16bpc_inner(
    token: Arm64,
    dst: &mut [u16],
    dst_stride: usize,
    src: &[u16],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: usize,
    my: usize,
    h_filter_type: Rav1dFilterMode,
    v_filter_type: Rav1dFilterMode,
    bitdepth_max: u16,
) {
    let intermediate_bits = 4u8;

    let fh = get_filter_coeff(mx, w, h_filter_type);
    let fv = get_filter_coeff(my, h, v_filter_type);

    match (fh, fv) {
        (Some(fh), Some(fv)) => {
            let tmp_h = h + 7;
            let mut mid = [[0i32; MID_STRIDE]; 135];

            for y in 0..tmp_h {
                let src_offset = if y >= 3 {
                    (y - 3) * src_stride
                } else {
                    0usize.wrapping_sub((3 - y) * src_stride)
                };
                let src_row = &src[src_offset..];
                h_filter_8tap_16bpc_neon(
                    token,
                    &mut mid[y][..w],
                    src_row,
                    w,
                    fh,
                    6 - intermediate_bits,
                );
            }

            for y in 0..h {
                let dst_row = &mut dst[y * dst_stride..][..w];
                v_filter_8tap_16bpc_neon(
                    token,
                    dst_row,
                    &mid[y..],
                    w,
                    fv,
                    6 + intermediate_bits,
                    bitdepth_max,
                );
            }
        }
        (Some(fh), None) => {
            for y in 0..h {
                let src_row = &src[y * src_stride..];
                let dst_row = &mut dst[y * dst_stride..][..w];
                h_filter_8tap_16bpc_put_neon(token, dst_row, src_row, w, fh, bitdepth_max);
            }
        }
        (None, Some(fv)) => {
            for y in 0..h {
                let src_offset = if y >= 3 {
                    (y - 3) * src_stride
                } else {
                    0usize.wrapping_sub((3 - y) * src_stride)
                };
                let src_row = &src[src_offset..];
                let dst_row = &mut dst[y * dst_stride..][..w];
                v_filter_8tap_16bpc_direct_neon(
                    token,
                    dst_row,
                    src_row,
                    src_stride,
                    w,
                    fv,
                    bitdepth_max,
                );
            }
        }
        (None, None) => {
            for y in 0..h {
                let src_row = &src[y * src_stride..][..w];
                let dst_row = &mut dst[y * dst_stride..][..w];
                dst_row.copy_from_slice(src_row);
            }
        }
    }
}

macro_rules! define_put_8tap_16bpc {
    ($name:ident, $filter:expr) => {
        #[cfg(target_arch = "aarch64")]
        pub unsafe extern "C" fn $name(
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
            let token = unsafe { Arm64::forge_token_dangerously() };
            let w = w as usize;
            let h = h as usize;
            let mx = mx as usize;
            let my = my as usize;
            let dst_stride_u16 = (dst_stride / 2) as usize;
            let src_stride_u16 = (src_stride / 2) as usize;

            let src_base = (src_ptr as *const u16).offset(-3 * src_stride_u16 as isize - 3);

            let src_len = (h + 7) * src_stride_u16 + w + 7;
            let src = std::slice::from_raw_parts(src_base, src_len);

            let dst_len = h * dst_stride_u16 + w;
            let dst = std::slice::from_raw_parts_mut(dst_ptr as *mut u16, dst_len);

            let src_adjusted = &src[3 * src_stride_u16 + 3..];

            put_8tap_16bpc_inner(
                token,
                dst,
                dst_stride_u16,
                src_adjusted,
                src_stride_u16,
                w,
                h,
                mx,
                my,
                get_h_filter_type($filter),
                get_v_filter_type($filter),
                bitdepth_max as u16,
            );
        }
    };
}

define_put_8tap_16bpc!(put_8tap_regular_16bpc_neon, Filter2d::Regular8Tap);
define_put_8tap_16bpc!(
    put_8tap_regular_smooth_16bpc_neon,
    Filter2d::RegularSmooth8Tap
);
define_put_8tap_16bpc!(
    put_8tap_regular_sharp_16bpc_neon,
    Filter2d::RegularSharp8Tap
);
define_put_8tap_16bpc!(
    put_8tap_smooth_regular_16bpc_neon,
    Filter2d::SmoothRegular8Tap
);
define_put_8tap_16bpc!(put_8tap_smooth_16bpc_neon, Filter2d::Smooth8Tap);
define_put_8tap_16bpc!(put_8tap_smooth_sharp_16bpc_neon, Filter2d::SmoothSharp8Tap);
define_put_8tap_16bpc!(
    put_8tap_sharp_regular_16bpc_neon,
    Filter2d::SharpRegular8Tap
);
define_put_8tap_16bpc!(put_8tap_sharp_smooth_16bpc_neon, Filter2d::SharpSmooth8Tap);
define_put_8tap_16bpc!(put_8tap_sharp_16bpc_neon, Filter2d::Sharp8Tap);

// ============================================================================
// 8-TAP PREP 16BPC
// ============================================================================

/// Vertical 8-tap filter from i32 intermediate to i16 output for 16bpc prep
#[cfg(target_arch = "aarch64")]
#[arcane]
#[allow(clippy::too_many_arguments)]
fn v_filter_8tap_16bpc_to_i16_neon(
    _token: Arm64,
    dst: &mut [i16],
    mid: &[[i32; MID_STRIDE]],
    w: usize,
    filter: &[i8; 8],
    sh: u8,
) {
    let rnd = (1i32 << sh) >> 1;

    let mut col = 0;

    while col + 4 <= w {
        let c0 = filter[0] as i32;
        let c1 = filter[1] as i32;
        let c2 = filter[2] as i32;
        let c3 = filter[3] as i32;
        let c4 = filter[4] as i32;
        let c5 = filter[5] as i32;
        let c6 = filter[6] as i32;
        let c7 = filter[7] as i32;

        let r0 = safe_simd::vld1q_s32(mid[0][col..][..4].try_into().unwrap());
        let r1 = safe_simd::vld1q_s32(mid[1][col..][..4].try_into().unwrap());
        let r2 = safe_simd::vld1q_s32(mid[2][col..][..4].try_into().unwrap());
        let r3 = safe_simd::vld1q_s32(mid[3][col..][..4].try_into().unwrap());
        let r4 = safe_simd::vld1q_s32(mid[4][col..][..4].try_into().unwrap());
        let r5 = safe_simd::vld1q_s32(mid[5][col..][..4].try_into().unwrap());
        let r6 = safe_simd::vld1q_s32(mid[6][col..][..4].try_into().unwrap());
        let r7 = safe_simd::vld1q_s32(mid[7][col..][..4].try_into().unwrap());

        let mut sum = vmulq_n_s32(r0, c0);
        sum = vmlaq_n_s32(sum, r1, c1);
        sum = vmlaq_n_s32(sum, r2, c2);
        sum = vmlaq_n_s32(sum, r3, c3);
        sum = vmlaq_n_s32(sum, r4, c4);
        sum = vmlaq_n_s32(sum, r5, c5);
        sum = vmlaq_n_s32(sum, r6, c6);
        sum = vmlaq_n_s32(sum, r7, c7);

        let rnd_vec = vdupq_n_s32(rnd);
        sum = vshrq_n_s32::<8>(vaddq_s32(sum, rnd_vec));

        let result = vqmovn_s32(sum);
        let dst_arr: &mut [i16; 4] = (&mut dst[col..col + 4]).try_into().unwrap();
        safe_simd::vst1_s16(dst_arr, result);
        col += 4;
    }

    // Scalar fallback
    while col < w {
        let mut sum = 0i64;
        for i in 0..8 {
            sum += filter[i] as i64 * mid[i][col] as i64;
        }
        dst[col] = ((sum + rnd as i64) >> sh) as i16;
        col += 1;
    }
}

/// Main 8-tap prep function for 16bpc
#[cfg(target_arch = "aarch64")]
#[arcane]
#[allow(clippy::too_many_arguments)]
fn prep_8tap_16bpc_inner(
    token: Arm64,
    tmp: &mut [i16],
    src: &[u16],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: usize,
    my: usize,
    h_filter_type: Rav1dFilterMode,
    v_filter_type: Rav1dFilterMode,
) {
    let intermediate_bits = 4u8;

    let fh = get_filter_coeff(mx, w, h_filter_type);
    let fv = get_filter_coeff(my, h, v_filter_type);

    match (fh, fv) {
        (Some(fh), Some(fv)) => {
            let tmp_h = h + 7;
            let mut mid = [[0i32; MID_STRIDE]; 135];

            for y in 0..tmp_h {
                let src_offset = if y >= 3 {
                    (y - 3) * src_stride
                } else {
                    0usize.wrapping_sub((3 - y) * src_stride)
                };
                let src_row = &src[src_offset..];
                h_filter_8tap_16bpc_neon(
                    token,
                    &mut mid[y][..w],
                    src_row,
                    w,
                    fh,
                    6 - intermediate_bits,
                );
            }

            for y in 0..h {
                let out_row = &mut tmp[y * w..][..w];
                v_filter_8tap_16bpc_to_i16_neon(
                    token,
                    out_row,
                    &mid[y..],
                    w,
                    fv,
                    6 + intermediate_bits,
                );
            }
        }
        (Some(fh), None) => {
            for y in 0..h {
                let src_row = &src[y * src_stride..];
                let out_row = &mut tmp[y * w..][..w];
                let mut col = 0;
                while col + 4 <= w {
                    let c0 = fh[0] as i32;
                    let c1 = fh[1] as i32;
                    let c2 = fh[2] as i32;
                    let c3 = fh[3] as i32;
                    let c4 = fh[4] as i32;
                    let c5 = fh[5] as i32;
                    let c6 = fh[6] as i32;
                    let c7 = fh[7] as i32;

                    let s0 = safe_simd::vld1_u16(src_row[col..][..4].try_into().unwrap());
                    let s1 = safe_simd::vld1_u16(src_row[col + 1..][..4].try_into().unwrap());
                    let s2 = safe_simd::vld1_u16(src_row[col + 2..][..4].try_into().unwrap());
                    let s3 = safe_simd::vld1_u16(src_row[col + 3..][..4].try_into().unwrap());
                    let s4 = safe_simd::vld1_u16(src_row[col + 4..][..4].try_into().unwrap());
                    let s5 = safe_simd::vld1_u16(src_row[col + 5..][..4].try_into().unwrap());
                    let s6 = safe_simd::vld1_u16(src_row[col + 6..][..4].try_into().unwrap());
                    let s7 = safe_simd::vld1_u16(src_row[col + 7..][..4].try_into().unwrap());

                    let s0_32 = vreinterpretq_s32_u32(vmovl_u16(s0));
                    let s1_32 = vreinterpretq_s32_u32(vmovl_u16(s1));
                    let s2_32 = vreinterpretq_s32_u32(vmovl_u16(s2));
                    let s3_32 = vreinterpretq_s32_u32(vmovl_u16(s3));
                    let s4_32 = vreinterpretq_s32_u32(vmovl_u16(s4));
                    let s5_32 = vreinterpretq_s32_u32(vmovl_u16(s5));
                    let s6_32 = vreinterpretq_s32_u32(vmovl_u16(s6));
                    let s7_32 = vreinterpretq_s32_u32(vmovl_u16(s7));

                    let mut sum = vmulq_n_s32(s0_32, c0);
                    sum = vmlaq_n_s32(sum, s1_32, c1);
                    sum = vmlaq_n_s32(sum, s2_32, c2);
                    sum = vmlaq_n_s32(sum, s3_32, c3);
                    sum = vmlaq_n_s32(sum, s4_32, c4);
                    sum = vmlaq_n_s32(sum, s5_32, c5);
                    sum = vmlaq_n_s32(sum, s6_32, c6);
                    sum = vmlaq_n_s32(sum, s7_32, c7);

                    // Shift for intermediate
                    let rnd_vec = vdupq_n_s32(8);
                    let result = vshrq_n_s32::<4>(vaddq_s32(sum, rnd_vec));

                    let narrow = vqmovn_s32(result);
                    let out_arr: &mut [i16; 4] = (&mut out_row[col..col + 4]).try_into().unwrap();
                    safe_simd::vst1_s16(out_arr, narrow);
                    col += 4;
                }
                while col < w {
                    let mut sum = 0i32;
                    for i in 0..8 {
                        sum += fh[i] as i32 * src_row[col + i] as i32;
                    }
                    out_row[col] = ((sum + 8) >> intermediate_bits) as i16;
                    col += 1;
                }
            }
        }
        (None, Some(fv)) => {
            for y in 0..h {
                let out_row = &mut tmp[y * w..][..w];

                let mut mid = [[0i32; MID_STRIDE]; 8];
                for i in 0..8 {
                    let src_offset = if y + i >= 3 {
                        (y + i - 3) * src_stride
                    } else {
                        0usize.wrapping_sub((3 - y - i) * src_stride)
                    };
                    for x in 0..w {
                        mid[i][x] = (src[src_offset + x] as i32) << intermediate_bits;
                    }
                }

                v_filter_8tap_16bpc_to_i16_neon(token, out_row, &mid, w, fv, 6);
            }
        }
        (None, None) => {
            for y in 0..h {
                let src_row = &src[y * src_stride..][..w];
                let out_row = &mut tmp[y * w..][..w];
                for x in 0..w {
                    out_row[x] = (src_row[x] as i32 >> (10 - intermediate_bits)) as i16;
                }
            }
        }
    }
}

macro_rules! define_prep_8tap_16bpc {
    ($name:ident, $filter:expr) => {
        #[cfg(target_arch = "aarch64")]
        pub unsafe extern "C" fn $name(
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
            let token = unsafe { Arm64::forge_token_dangerously() };
            let w = w as usize;
            let h = h as usize;
            let mx = mx as usize;
            let my = my as usize;
            let src_stride_u16 = (src_stride / 2) as usize;

            let src_base = (src_ptr as *const u16).offset(-3 * src_stride_u16 as isize - 3);

            let src_len = (h + 7) * src_stride_u16 + w + 7;
            let src = std::slice::from_raw_parts(src_base, src_len);

            let tmp_len = h * w;
            let tmp_slice = std::slice::from_raw_parts_mut(tmp, tmp_len);

            let src_adjusted = &src[3 * src_stride_u16 + 3..];

            prep_8tap_16bpc_inner(
                token,
                tmp_slice,
                src_adjusted,
                src_stride_u16,
                w,
                h,
                mx,
                my,
                get_h_filter_type($filter),
                get_v_filter_type($filter),
            );
        }
    };
}

define_prep_8tap_16bpc!(prep_8tap_regular_16bpc_neon, Filter2d::Regular8Tap);
define_prep_8tap_16bpc!(
    prep_8tap_regular_smooth_16bpc_neon,
    Filter2d::RegularSmooth8Tap
);
define_prep_8tap_16bpc!(
    prep_8tap_regular_sharp_16bpc_neon,
    Filter2d::RegularSharp8Tap
);
define_prep_8tap_16bpc!(
    prep_8tap_smooth_regular_16bpc_neon,
    Filter2d::SmoothRegular8Tap
);
define_prep_8tap_16bpc!(prep_8tap_smooth_16bpc_neon, Filter2d::Smooth8Tap);
define_prep_8tap_16bpc!(prep_8tap_smooth_sharp_16bpc_neon, Filter2d::SmoothSharp8Tap);
define_prep_8tap_16bpc!(
    prep_8tap_sharp_regular_16bpc_neon,
    Filter2d::SharpRegular8Tap
);
define_prep_8tap_16bpc!(prep_8tap_sharp_smooth_16bpc_neon, Filter2d::SharpSmooth8Tap);
define_prep_8tap_16bpc!(prep_8tap_sharp_16bpc_neon, Filter2d::Sharp8Tap);

// ============================================================================
// Safe dispatch wrappers for aarch64 NEON
// NEON is always available on aarch64, so these always return true.
// ============================================================================

#[cfg(target_arch = "aarch64")]
pub fn avg_dispatch<BD: BitDepth>(
    dst: PicOffset,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    bd: BD,
) -> bool {
    use crate::include::common::bitdepth::BPC;
    use zerocopy::AsBytes;
    let (mut dst_guard, _dst_base) = dst.full_guard_mut::<BD>();
    let dst_ptr = dst_guard.as_bytes_mut().as_mut_ptr() as *mut DynPixel;
    let dst_ptr = unsafe { dst_ptr.add(_dst_base * std::mem::size_of::<BD::Pixel>()) };
    let dst_stride = dst.stride();
    let bd_c = bd.into_c();
    let dst_ffi = FFISafe::new(&dst);
    unsafe {
        match BD::BPC {
            BPC::BPC8 => avg_8bpc_neon(dst_ptr, dst_stride, tmp1, tmp2, w, h, bd_c, dst_ffi),
            BPC::BPC16 => avg_16bpc_neon(dst_ptr, dst_stride, tmp1, tmp2, w, h, bd_c, dst_ffi),
        }
    }
    true
}

#[cfg(target_arch = "aarch64")]
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
    use zerocopy::AsBytes;
    let (mut dst_guard, _dst_base) = dst.full_guard_mut::<BD>();
    let dst_ptr = dst_guard.as_bytes_mut().as_mut_ptr() as *mut DynPixel;
    let dst_ptr = unsafe { dst_ptr.add(_dst_base * std::mem::size_of::<BD::Pixel>()) };
    let dst_stride = dst.stride();
    let bd_c = bd.into_c();
    let dst_ffi = FFISafe::new(&dst);
    unsafe {
        match BD::BPC {
            BPC::BPC8 => {
                w_avg_8bpc_neon(dst_ptr, dst_stride, tmp1, tmp2, w, h, weight, bd_c, dst_ffi)
            }
            BPC::BPC16 => {
                w_avg_16bpc_neon(dst_ptr, dst_stride, tmp1, tmp2, w, h, weight, bd_c, dst_ffi)
            }
        }
    }
    true
}

#[cfg(target_arch = "aarch64")]
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
    use zerocopy::AsBytes;
    let (mut dst_guard, _dst_base) = dst.full_guard_mut::<BD>();
    let dst_ptr = dst_guard.as_bytes_mut().as_mut_ptr() as *mut DynPixel;
    let dst_ptr = unsafe { dst_ptr.add(_dst_base * std::mem::size_of::<BD::Pixel>()) };
    let dst_stride = dst.stride();
    let mask_ptr = mask[..(w * h) as usize].as_ptr();
    let bd_c = bd.into_c();
    let dst_ffi = FFISafe::new(&dst);
    unsafe {
        match BD::BPC {
            BPC::BPC8 => mask_8bpc_neon(
                dst_ptr, dst_stride, tmp1, tmp2, w, h, mask_ptr, bd_c, dst_ffi,
            ),
            BPC::BPC16 => mask_16bpc_neon(
                dst_ptr, dst_stride, tmp1, tmp2, w, h, mask_ptr, bd_c, dst_ffi,
            ),
        }
    }
    true
}

#[cfg(target_arch = "aarch64")]
pub fn blend_dispatch<BD: BitDepth>(
    dst: PicOffset,
    tmp: &[BD::Pixel; SCRATCH_INTER_INTRA_BUF_LEN],
    w: i32,
    h: i32,
    mask: &[u8],
) -> bool {
    use crate::include::common::bitdepth::BPC;
    use zerocopy::AsBytes;
    let (mut dst_guard, _dst_base) = dst.full_guard_mut::<BD>();
    let dst_ptr = dst_guard.as_bytes_mut().as_mut_ptr() as *mut DynPixel;
    let dst_ptr = unsafe { dst_ptr.add(_dst_base * std::mem::size_of::<BD::Pixel>()) };
    let dst_stride = dst.stride();
    let tmp_ptr = std::ptr::from_ref(tmp).cast();
    let mask_ptr = mask[..(w * h) as usize].as_ptr();
    let dst_ffi = FFISafe::new(&dst);
    unsafe {
        match BD::BPC {
            BPC::BPC8 => blend_8bpc_neon(dst_ptr, dst_stride, tmp_ptr, w, h, mask_ptr, dst_ffi),
            BPC::BPC16 => blend_16bpc_neon(dst_ptr, dst_stride, tmp_ptr, w, h, mask_ptr, dst_ffi),
        }
    }
    true
}

#[cfg(target_arch = "aarch64")]
pub fn blend_dir_dispatch<BD: BitDepth>(
    is_h: bool,
    dst: PicOffset,
    tmp: &[BD::Pixel; SCRATCH_LAP_LEN],
    w: i32,
    h: i32,
) -> bool {
    use crate::include::common::bitdepth::BPC;
    use zerocopy::AsBytes;
    let (mut dst_guard, _dst_base) = dst.full_guard_mut::<BD>();
    let dst_ptr = dst_guard.as_bytes_mut().as_mut_ptr() as *mut DynPixel;
    let dst_ptr = unsafe { dst_ptr.add(_dst_base * std::mem::size_of::<BD::Pixel>()) };
    let dst_stride = dst.stride();
    let tmp_ptr = std::ptr::from_ref(tmp).cast();
    let dst_ffi = FFISafe::new(&dst);
    unsafe {
        match (BD::BPC, is_h) {
            (BPC::BPC8, true) => blend_h_8bpc_neon(dst_ptr, dst_stride, tmp_ptr, w, h, dst_ffi),
            (BPC::BPC8, false) => blend_v_8bpc_neon(dst_ptr, dst_stride, tmp_ptr, w, h, dst_ffi),
            (BPC::BPC16, true) => blend_h_16bpc_neon(dst_ptr, dst_stride, tmp_ptr, w, h, dst_ffi),
            (BPC::BPC16, false) => blend_v_16bpc_neon(dst_ptr, dst_stride, tmp_ptr, w, h, dst_ffi),
        }
    }
    true
}

#[cfg(target_arch = "aarch64")]
pub fn w_mask_dispatch<BD: BitDepth>(
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
    use zerocopy::AsBytes;
    let (mut dst_guard, _dst_base) = dst.full_guard_mut::<BD>();
    let dst_ptr = dst_guard.as_bytes_mut().as_mut_ptr() as *mut DynPixel;
    let dst_ptr = unsafe { dst_ptr.add(_dst_base * std::mem::size_of::<BD::Pixel>()) };
    let dst_stride = dst.stride();
    let bd_c = bd.into_c();
    let dst_ffi = FFISafe::new(&dst);
    unsafe {
        match (BD::BPC, layout) {
            (BPC::BPC8, Rav1dPixelLayoutSubSampled::I420) => w_mask_420_8bpc_neon(
                dst_ptr, dst_stride, tmp1, tmp2, w, h, mask, sign, bd_c, dst_ffi,
            ),
            (BPC::BPC8, Rav1dPixelLayoutSubSampled::I422) => w_mask_422_8bpc_neon(
                dst_ptr, dst_stride, tmp1, tmp2, w, h, mask, sign, bd_c, dst_ffi,
            ),
            (BPC::BPC8, Rav1dPixelLayoutSubSampled::I444) => w_mask_444_8bpc_neon(
                dst_ptr, dst_stride, tmp1, tmp2, w, h, mask, sign, bd_c, dst_ffi,
            ),
            (BPC::BPC16, Rav1dPixelLayoutSubSampled::I420) => w_mask_420_16bpc_neon(
                dst_ptr, dst_stride, tmp1, tmp2, w, h, mask, sign, bd_c, dst_ffi,
            ),
            (BPC::BPC16, Rav1dPixelLayoutSubSampled::I422) => w_mask_422_16bpc_neon(
                dst_ptr, dst_stride, tmp1, tmp2, w, h, mask, sign, bd_c, dst_ffi,
            ),
            (BPC::BPC16, Rav1dPixelLayoutSubSampled::I444) => w_mask_444_16bpc_neon(
                dst_ptr, dst_stride, tmp1, tmp2, w, h, mask, sign, bd_c, dst_ffi,
            ),
        }
    }
    true
}

#[cfg(target_arch = "aarch64")]
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
    use Filter2d::*;
    use zerocopy::AsBytes;
    let (mut dst_guard, _dst_base) = dst.full_guard_mut::<BD>();
    let dst_ptr = dst_guard.as_bytes_mut().as_mut_ptr() as *mut DynPixel;
    let dst_ptr = unsafe { dst_ptr.add(_dst_base * std::mem::size_of::<BD::Pixel>()) };
    let dst_stride = dst.stride();
    let (src_guard, _src_base) = src.full_guard::<BD>();
    let src_ptr = src_guard.as_bytes().as_ptr() as *const DynPixel;
    let src_ptr = unsafe { src_ptr.add(_src_base * std::mem::size_of::<BD::Pixel>()) };
    let src_stride = src.stride();
    let bd_c = bd.into_c();
    let dst_ffi = FFISafe::new(&dst);
    let src_ffi = FFISafe::new(&src);
    unsafe {
        match (BD::BPC, filter) {
            (BPC::BPC8, Regular8Tap) => put_8tap_regular_8bpc_neon(
                dst_ptr, dst_stride, src_ptr, src_stride, w, h, mx, my, bd_c, dst_ffi, src_ffi,
            ),
            (BPC::BPC8, RegularSmooth8Tap) => put_8tap_regular_smooth_8bpc_neon(
                dst_ptr, dst_stride, src_ptr, src_stride, w, h, mx, my, bd_c, dst_ffi, src_ffi,
            ),
            (BPC::BPC8, RegularSharp8Tap) => put_8tap_regular_sharp_8bpc_neon(
                dst_ptr, dst_stride, src_ptr, src_stride, w, h, mx, my, bd_c, dst_ffi, src_ffi,
            ),
            (BPC::BPC8, SmoothRegular8Tap) => put_8tap_smooth_regular_8bpc_neon(
                dst_ptr, dst_stride, src_ptr, src_stride, w, h, mx, my, bd_c, dst_ffi, src_ffi,
            ),
            (BPC::BPC8, Smooth8Tap) => put_8tap_smooth_8bpc_neon(
                dst_ptr, dst_stride, src_ptr, src_stride, w, h, mx, my, bd_c, dst_ffi, src_ffi,
            ),
            (BPC::BPC8, SmoothSharp8Tap) => put_8tap_smooth_sharp_8bpc_neon(
                dst_ptr, dst_stride, src_ptr, src_stride, w, h, mx, my, bd_c, dst_ffi, src_ffi,
            ),
            (BPC::BPC8, SharpRegular8Tap) => put_8tap_sharp_regular_8bpc_neon(
                dst_ptr, dst_stride, src_ptr, src_stride, w, h, mx, my, bd_c, dst_ffi, src_ffi,
            ),
            (BPC::BPC8, SharpSmooth8Tap) => put_8tap_sharp_smooth_8bpc_neon(
                dst_ptr, dst_stride, src_ptr, src_stride, w, h, mx, my, bd_c, dst_ffi, src_ffi,
            ),
            (BPC::BPC8, Sharp8Tap) => put_8tap_sharp_8bpc_neon(
                dst_ptr, dst_stride, src_ptr, src_stride, w, h, mx, my, bd_c, dst_ffi, src_ffi,
            ),
            (BPC::BPC8, Bilinear) => put_bilin_8bpc_neon(
                dst_ptr, dst_stride, src_ptr, src_stride, w, h, mx, my, bd_c, dst_ffi, src_ffi,
            ),
            (BPC::BPC16, Regular8Tap) => put_8tap_regular_16bpc_neon(
                dst_ptr, dst_stride, src_ptr, src_stride, w, h, mx, my, bd_c, dst_ffi, src_ffi,
            ),
            (BPC::BPC16, RegularSmooth8Tap) => put_8tap_regular_smooth_16bpc_neon(
                dst_ptr, dst_stride, src_ptr, src_stride, w, h, mx, my, bd_c, dst_ffi, src_ffi,
            ),
            (BPC::BPC16, RegularSharp8Tap) => put_8tap_regular_sharp_16bpc_neon(
                dst_ptr, dst_stride, src_ptr, src_stride, w, h, mx, my, bd_c, dst_ffi, src_ffi,
            ),
            (BPC::BPC16, SmoothRegular8Tap) => put_8tap_smooth_regular_16bpc_neon(
                dst_ptr, dst_stride, src_ptr, src_stride, w, h, mx, my, bd_c, dst_ffi, src_ffi,
            ),
            (BPC::BPC16, Smooth8Tap) => put_8tap_smooth_16bpc_neon(
                dst_ptr, dst_stride, src_ptr, src_stride, w, h, mx, my, bd_c, dst_ffi, src_ffi,
            ),
            (BPC::BPC16, SmoothSharp8Tap) => put_8tap_smooth_sharp_16bpc_neon(
                dst_ptr, dst_stride, src_ptr, src_stride, w, h, mx, my, bd_c, dst_ffi, src_ffi,
            ),
            (BPC::BPC16, SharpRegular8Tap) => put_8tap_sharp_regular_16bpc_neon(
                dst_ptr, dst_stride, src_ptr, src_stride, w, h, mx, my, bd_c, dst_ffi, src_ffi,
            ),
            (BPC::BPC16, SharpSmooth8Tap) => put_8tap_sharp_smooth_16bpc_neon(
                dst_ptr, dst_stride, src_ptr, src_stride, w, h, mx, my, bd_c, dst_ffi, src_ffi,
            ),
            (BPC::BPC16, Sharp8Tap) => put_8tap_sharp_16bpc_neon(
                dst_ptr, dst_stride, src_ptr, src_stride, w, h, mx, my, bd_c, dst_ffi, src_ffi,
            ),
            (BPC::BPC16, Bilinear) => put_bilin_16bpc_neon(
                dst_ptr, dst_stride, src_ptr, src_stride, w, h, mx, my, bd_c, dst_ffi, src_ffi,
            ),
        }
    }
    true
}

#[cfg(target_arch = "aarch64")]
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
    use Filter2d::*;
    let tmp_ptr = tmp[..(w * h) as usize].as_mut_ptr();
    use zerocopy::AsBytes;
    let (src_guard, _src_base) = src.full_guard::<BD>();
    let src_ptr = src_guard.as_bytes().as_ptr() as *const DynPixel;
    let src_ptr = unsafe { src_ptr.add(_src_base * std::mem::size_of::<BD::Pixel>()) };
    let src_stride = src.stride();
    let bd_c = bd.into_c();
    let src_ffi = FFISafe::new(&src);
    unsafe {
        match (BD::BPC, filter) {
            (BPC::BPC8, Regular8Tap) => prep_8tap_regular_8bpc_neon(
                tmp_ptr, src_ptr, src_stride, w, h, mx, my, bd_c, src_ffi,
            ),
            (BPC::BPC8, RegularSmooth8Tap) => prep_8tap_regular_smooth_8bpc_neon(
                tmp_ptr, src_ptr, src_stride, w, h, mx, my, bd_c, src_ffi,
            ),
            (BPC::BPC8, RegularSharp8Tap) => prep_8tap_regular_sharp_8bpc_neon(
                tmp_ptr, src_ptr, src_stride, w, h, mx, my, bd_c, src_ffi,
            ),
            (BPC::BPC8, SmoothRegular8Tap) => prep_8tap_smooth_regular_8bpc_neon(
                tmp_ptr, src_ptr, src_stride, w, h, mx, my, bd_c, src_ffi,
            ),
            (BPC::BPC8, Smooth8Tap) => prep_8tap_smooth_8bpc_neon(
                tmp_ptr, src_ptr, src_stride, w, h, mx, my, bd_c, src_ffi,
            ),
            (BPC::BPC8, SmoothSharp8Tap) => prep_8tap_smooth_sharp_8bpc_neon(
                tmp_ptr, src_ptr, src_stride, w, h, mx, my, bd_c, src_ffi,
            ),
            (BPC::BPC8, SharpRegular8Tap) => prep_8tap_sharp_regular_8bpc_neon(
                tmp_ptr, src_ptr, src_stride, w, h, mx, my, bd_c, src_ffi,
            ),
            (BPC::BPC8, SharpSmooth8Tap) => prep_8tap_sharp_smooth_8bpc_neon(
                tmp_ptr, src_ptr, src_stride, w, h, mx, my, bd_c, src_ffi,
            ),
            (BPC::BPC8, Sharp8Tap) => {
                prep_8tap_sharp_8bpc_neon(tmp_ptr, src_ptr, src_stride, w, h, mx, my, bd_c, src_ffi)
            }
            (BPC::BPC8, Bilinear) => {
                prep_bilin_8bpc_neon(tmp_ptr, src_ptr, src_stride, w, h, mx, my, bd_c, src_ffi)
            }
            (BPC::BPC16, Regular8Tap) => prep_8tap_regular_16bpc_neon(
                tmp_ptr, src_ptr, src_stride, w, h, mx, my, bd_c, src_ffi,
            ),
            (BPC::BPC16, RegularSmooth8Tap) => prep_8tap_regular_smooth_16bpc_neon(
                tmp_ptr, src_ptr, src_stride, w, h, mx, my, bd_c, src_ffi,
            ),
            (BPC::BPC16, RegularSharp8Tap) => prep_8tap_regular_sharp_16bpc_neon(
                tmp_ptr, src_ptr, src_stride, w, h, mx, my, bd_c, src_ffi,
            ),
            (BPC::BPC16, SmoothRegular8Tap) => prep_8tap_smooth_regular_16bpc_neon(
                tmp_ptr, src_ptr, src_stride, w, h, mx, my, bd_c, src_ffi,
            ),
            (BPC::BPC16, Smooth8Tap) => prep_8tap_smooth_16bpc_neon(
                tmp_ptr, src_ptr, src_stride, w, h, mx, my, bd_c, src_ffi,
            ),
            (BPC::BPC16, SmoothSharp8Tap) => prep_8tap_smooth_sharp_16bpc_neon(
                tmp_ptr, src_ptr, src_stride, w, h, mx, my, bd_c, src_ffi,
            ),
            (BPC::BPC16, SharpRegular8Tap) => prep_8tap_sharp_regular_16bpc_neon(
                tmp_ptr, src_ptr, src_stride, w, h, mx, my, bd_c, src_ffi,
            ),
            (BPC::BPC16, SharpSmooth8Tap) => prep_8tap_sharp_smooth_16bpc_neon(
                tmp_ptr, src_ptr, src_stride, w, h, mx, my, bd_c, src_ffi,
            ),
            (BPC::BPC16, Sharp8Tap) => prep_8tap_sharp_16bpc_neon(
                tmp_ptr, src_ptr, src_stride, w, h, mx, my, bd_c, src_ffi,
            ),
            (BPC::BPC16, Bilinear) => {
                prep_bilin_16bpc_neon(tmp_ptr, src_ptr, src_stride, w, h, mx, my, bd_c, src_ffi)
            }
        }
    }
    true
}

/// No SIMD for scaled variants on aarch64.
#[cfg(target_arch = "aarch64")]
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

/// No SIMD for scaled variants on aarch64.
#[cfg(target_arch = "aarch64")]
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

/// No SIMD for warp on aarch64.
#[cfg(target_arch = "aarch64")]
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

/// No SIMD for warp on aarch64.
#[cfg(target_arch = "aarch64")]
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

/// No SIMD for emu_edge on aarch64.
#[cfg(target_arch = "aarch64")]
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

/// No SIMD for resize on aarch64.
#[cfg(target_arch = "aarch64")]
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
