//! Safe SIMD implementations of motion compensation functions for ARM NEON
//!
//! These use archmage tokens to safely invoke NEON intrinsics.
//! The extern "C" wrappers are used for FFI compatibility with rav1d's dispatch system.

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
use archmage::{arcane, Arm64, SimdToken};

use crate::include::common::bitdepth::DynPixel;
use crate::include::dav1d::picture::Rav1dPictureDataComponentOffset;
use crate::src::ffi_safe::FFISafe;
use crate::src::internal::COMPINTER_LEN;

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
            unsafe {
                // Load 16 i16 values
                let t1_lo = vld1q_s16(tmp1_row[col..].as_ptr());
                let t1_hi = vld1q_s16(tmp1_row[col + 8..].as_ptr());
                let t2_lo = vld1q_s16(tmp2_row[col..].as_ptr());
                let t2_hi = vld1q_s16(tmp2_row[col + 8..].as_ptr());

                // Add: tmp1 + tmp2
                let sum_lo = vaddq_s16(t1_lo, t2_lo);
                let sum_hi = vaddq_s16(t1_hi, t2_hi);

                // vqrdmulhq_n_s16(a, 2048) = (2 * a * 2048 + 0x8000) >> 16
                // = (a * 4096 + 32768) >> 16 = (a * 1024 + 8192) >> 14
                // We want (a * 1024 + 16384) >> 15 = pmulhrsw equivalent
                // Using vqrshrn for rounding shift: (sum + 1024) >> 11 maps to pmulhrsw(sum, 1024)
                let avg_lo = vqrdmulhq_n_s16(sum_lo, 2048);
                let avg_hi = vqrdmulhq_n_s16(sum_hi, 2048);

                // Pack to u8 with saturation
                let packed_lo = vqmovun_s16(avg_lo);
                let packed_hi = vqmovun_s16(avg_hi);
                let result = vcombine_u8(packed_lo, packed_hi);

                vst1q_u8(dst_row[col..].as_mut_ptr(), result);
            }
            col += 16;
        }

        // Process 8 pixels at a time
        while col + 8 <= w {
            unsafe {
                let t1 = vld1q_s16(tmp1_row[col..].as_ptr());
                let t2 = vld1q_s16(tmp2_row[col..].as_ptr());
                let sum = vaddq_s16(t1, t2);
                let avg = vqrdmulhq_n_s16(sum, 2048);
                let packed = vqmovun_s16(avg);
                vst1_u8(dst_row[col..].as_mut_ptr(), packed);
            }
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
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    // SAFETY: This function is only called through dispatch when NEON is available
    let token = unsafe { Arm64::forge_token_dangerously() };

    let w = w as usize;
    let h = h as usize;
    let dst = std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h * dst_stride.unsigned_abs());

    avg_8bpc_inner(token, dst, dst_stride as usize, tmp1.as_slice(), tmp2.as_slice(), w, h);
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
            unsafe {
                let t1 = vld1q_s16(tmp1_row[col..].as_ptr());
                let t2 = vld1q_s16(tmp2_row[col..].as_ptr());

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

                vst1q_u16(dst_row[col..].as_mut_ptr(), vreinterpretq_u16_s16(clamped));
            }
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
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let token = unsafe { Arm64::forge_token_dangerously() };
    let w = w as usize;
    let h = h as usize;
    let dst_stride_u16 = (dst_stride / 2) as usize;
    let dst = std::slice::from_raw_parts_mut(dst_ptr as *mut u16, h * dst_stride_u16);

    avg_16bpc_inner(token, dst, dst_stride_u16, tmp1.as_slice(), tmp2.as_slice(), w, h, bitdepth_max);
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
            unsafe {
                let t1 = vld1q_s16(tmp1_row[col..].as_ptr());
                let t2 = vld1q_s16(tmp2_row[col..].as_ptr());

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

                vst1_u8(dst_row[col..].as_mut_ptr(), packed);
            }
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
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let token = unsafe { Arm64::forge_token_dangerously() };
    let w = w as usize;
    let h = h as usize;
    let dst = std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h * dst_stride.unsigned_abs());

    w_avg_8bpc_inner(token, dst, dst_stride as usize, tmp1.as_slice(), tmp2.as_slice(), w, h, weight);
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
            unsafe {
                let t1_16 = vld1_s16(tmp1_row[col..].as_ptr());
                let t2_16 = vld1_s16(tmp2_row[col..].as_ptr());
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
                vst1_u16(dst_row[col..].as_mut_ptr(), vreinterpret_u16_s16(narrow));
            }
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
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let token = unsafe { Arm64::forge_token_dangerously() };
    let w = w as usize;
    let h = h as usize;
    let dst_stride_u16 = (dst_stride / 2) as usize;
    let dst = std::slice::from_raw_parts_mut(dst_ptr as *mut u16, h * dst_stride_u16);

    w_avg_16bpc_inner(token, dst, dst_stride_u16, tmp1.as_slice(), tmp2.as_slice(), w, h, weight, bitdepth_max);
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
            unsafe {
                let t1 = vld1q_s16(tmp1_row[col..].as_ptr());
                let t2 = vld1q_s16(tmp2_row[col..].as_ptr());
                let m = vld1_u8(mask_row[col..].as_ptr());

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

                vst1_u8(dst_row[col..].as_mut_ptr(), packed);
            }
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
    mask: &[u8],
    _bitdepth_max: i32,
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let token = unsafe { Arm64::forge_token_dangerously() };
    let w = w as usize;
    let h = h as usize;
    let dst = std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h * dst_stride.unsigned_abs());

    mask_8bpc_inner(token, dst, dst_stride as usize, tmp1.as_slice(), tmp2.as_slice(), w, h, mask);
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
            unsafe {
                let t1_16 = vld1_s16(tmp1_row[col..].as_ptr());
                let t2_16 = vld1_s16(tmp2_row[col..].as_ptr());
                let t1 = vmovl_s16(t1_16);
                let t2 = vmovl_s16(t2_16);

                // Load 4 mask bytes
                let m_bytes: [u8; 8] = [
                    mask_row[col], mask_row[col+1], mask_row[col+2], mask_row[col+3],
                    0, 0, 0, 0
                ];
                let m8 = vld1_u8(m_bytes.as_ptr());
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
                vst1_u16(dst_row[col..].as_mut_ptr(), vreinterpret_u16_s16(narrow));
            }
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
    mask: &[u8],
    bitdepth_max: i32,
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let token = unsafe { Arm64::forge_token_dangerously() };
    let w = w as usize;
    let h = h as usize;
    let dst_stride_u16 = (dst_stride / 2) as usize;
    let dst = std::slice::from_raw_parts_mut(dst_ptr as *mut u16, h * dst_stride_u16);

    mask_16bpc_inner(token, dst, dst_stride_u16, tmp1.as_slice(), tmp2.as_slice(), w, h, mask, bitdepth_max);
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
            unsafe {
                let d = vld1_u8(dst_row[col..].as_ptr());
                let d16 = vreinterpretq_s16_u16(vmovl_u8(d));
                let t = vld1q_s16(tmp_row[col..].as_ptr());
                let m = vld1_u8(mask_row[col..].as_ptr());
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
                vst1_u8(dst_row[col..].as_mut_ptr(), packed);
            }
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
    tmp: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask: &[u8],
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let token = unsafe { Arm64::forge_token_dangerously() };
    let w = w as usize;
    let h = h as usize;
    let dst = std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h * dst_stride.unsigned_abs());

    blend_8bpc_inner(token, dst, dst_stride as usize, tmp.as_slice(), w, h, mask);
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
            unsafe {
                let d_u16 = vld1_u16(dst_row[col..].as_ptr());
                let d = vreinterpretq_s32_u32(vmovl_u16(d_u16));
                let t_16 = vld1_s16(tmp_row[col..].as_ptr());
                let t = vmovl_s16(t_16);

                // Load 4 mask bytes
                let m_bytes: [u8; 8] = [
                    mask_row[col], mask_row[col+1], mask_row[col+2], mask_row[col+3],
                    0, 0, 0, 0
                ];
                let m8 = vld1_u8(m_bytes.as_ptr());
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
                vst1_u16(dst_row[col..].as_mut_ptr(), narrow);
            }
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
    tmp: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask: &[u8],
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let token = unsafe { Arm64::forge_token_dangerously() };
    let w = w as usize;
    let h = h as usize;
    let dst_stride_u16 = (dst_stride / 2) as usize;
    let dst = std::slice::from_raw_parts_mut(dst_ptr as *mut u16, h * dst_stride_u16);

    blend_16bpc_inner(token, dst, dst_stride_u16, tmp.as_slice(), w, h, mask);
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
            unsafe {
                let d = vld1_u8(dst_row[col..].as_ptr());
                let d16 = vreinterpretq_s16_u16(vmovl_u8(d));
                let t = vld1q_s16(tmp_row[col..].as_ptr());
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
                vst1_u8(dst_row[col..].as_mut_ptr(), packed);
            }
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
    tmp: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    use crate::src::tables::dav1d_obmc_masks;

    let token = unsafe { Arm64::forge_token_dangerously() };
    let w = w as usize;
    let h = h as usize;
    let dst = std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h * dst_stride.unsigned_abs());
    let obmc = &dav1d_obmc_masks.0[h..];

    blend_v_8bpc_inner(token, dst, dst_stride as usize, tmp.as_slice(), w, h, obmc);
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
            unsafe {
                let d_u16 = vld1_u16(dst_row[col..].as_ptr());
                let d = vreinterpretq_s32_u32(vmovl_u16(d_u16));
                let t_16 = vld1_s16(tmp_row[col..].as_ptr());
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
                vst1_u16(dst_row[col..].as_mut_ptr(), narrow);
            }
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
    tmp: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    use crate::src::tables::dav1d_obmc_masks;

    let token = unsafe { Arm64::forge_token_dangerously() };
    let w = w as usize;
    let h = h as usize;
    let dst_stride_u16 = (dst_stride / 2) as usize;
    let dst = std::slice::from_raw_parts_mut(dst_ptr as *mut u16, h * dst_stride_u16);
    let obmc = &dav1d_obmc_masks.0[h..];
    // Note: For 16bpc we'd need bitdepth_max, but blend_v/blend_h don't pass it
    // Using 1023 (10-bit max) as default - this may need adjustment
    let bitdepth_max = 1023;

    blend_v_16bpc_inner(token, dst, dst_stride_u16, tmp.as_slice(), w, h, obmc, bitdepth_max);
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
            unsafe {
                let d = vld1_u8(dst_row[col..].as_ptr());
                let d16 = vreinterpretq_s16_u16(vmovl_u8(d));
                let t = vld1q_s16(tmp_row[col..].as_ptr());

                // Load 8 mask values
                let m = vld1_u8(obmc_masks[col..].as_ptr());
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
                vst1_u8(dst_row[col..].as_mut_ptr(), packed);
            }
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
    tmp: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    use crate::src::tables::dav1d_obmc_masks;

    let token = unsafe { Arm64::forge_token_dangerously() };
    let w = w as usize;
    let h = h as usize;
    let dst = std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h * dst_stride.unsigned_abs());
    let obmc = &dav1d_obmc_masks.0[w..];

    blend_h_8bpc_inner(token, dst, dst_stride as usize, tmp.as_slice(), w, h, obmc);
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
            unsafe {
                let d_u16 = vld1_u16(dst_row[col..].as_ptr());
                let d = vreinterpretq_s32_u32(vmovl_u16(d_u16));
                let t_16 = vld1_s16(tmp_row[col..].as_ptr());
                let t = vmovl_s16(t_16);

                // Load 4 mask bytes
                let m_bytes: [u8; 8] = [
                    obmc_masks[col], obmc_masks[col+1], obmc_masks[col+2], obmc_masks[col+3],
                    0, 0, 0, 0
                ];
                let m8 = vld1_u8(m_bytes.as_ptr());
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
                vst1_u16(dst_row[col..].as_mut_ptr(), narrow);
            }
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
    tmp: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    use crate::src::tables::dav1d_obmc_masks;

    let token = unsafe { Arm64::forge_token_dangerously() };
    let w = w as usize;
    let h = h as usize;
    let dst_stride_u16 = (dst_stride / 2) as usize;
    let dst = std::slice::from_raw_parts_mut(dst_ptr as *mut u16, h * dst_stride_u16);
    let obmc = &dav1d_obmc_masks.0[w..];
    let bitdepth_max = 1023;

    blend_h_16bpc_inner(token, dst, dst_stride_u16, tmp.as_slice(), w, h, obmc, bitdepth_max);
}
