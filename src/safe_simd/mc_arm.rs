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

// ============================================================================
// W_MASK - Weighted mask blend (compound prediction with per-pixel masking)
// ============================================================================

use crate::src::internal::SEG_MASK_LEN;

/// Core w_mask implementation for 8bpc
/// SS_HOR and SS_VER control subsampling: 444=(false,false), 422=(true,false), 420=(true,true)
#[cfg(target_arch = "aarch64")]
#[arcane]
fn w_mask_8bpc_inner<const SS_HOR: bool, const SS_VER: bool>(
    _token: Arm64,
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
        let mask_row = if SS_VER && (y & 1) != 0 {
            None
        } else {
            let mask_y = if SS_VER { y >> 1 } else { y };
            Some(&mut mask[mask_y * mask_w..][..mask_w])
        };

        let mut col = 0;

        // SIMD: process 8 pixels at a time
        while col + 8 <= w {
            unsafe {
                let t1 = vld1q_s16(tmp1_row[col..].as_ptr());
                let t2 = vld1q_s16(tmp2_row[col..].as_ptr());

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
                    rnd_vec
                );
                let blend_hi = vaddq_s32(
                    vaddq_s32(vmulq_s32(t1_hi, m_hi_32), vmulq_s32(t2_hi, inv_m_hi_32)),
                    rnd_vec
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

                vst1_u8(dst_row[col..].as_mut_ptr(), result_u8);

                // Store mask if needed
                if let Some(ref mut mask_row) = mask_row {
                    // For 444: 1:1 mask storage
                    // For 422: horizontal averaging (2 pixels -> 1 mask)
                    // For 420: also horizontal averaging
                    if !SS_HOR {
                        // 444: store all mask values
                        let m_narrow = vqmovun_s16(m_16);
                        vst1_u8(mask_row[col..].as_mut_ptr(), m_narrow);
                    } else {
                        // 422/420: average pairs horizontally
                        let mask_idx = col >> 1;
                        for i in 0..4 {
                            let m0 = vgetq_lane_s16(m_16, (i * 2) as i32) as i32;
                            let m1 = vgetq_lane_s16(m_16, (i * 2 + 1) as i32) as i32;
                            mask_row[mask_idx + i] = ((m0 + m1 + 1) >> 1) as u8;
                        }
                    }
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
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let token = unsafe { Arm64::forge_token_dangerously() };
    let w = w as usize;
    let h = h as usize;
    let dst = std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h * dst_stride.unsigned_abs());

    w_mask_8bpc_inner::<false, false>(
        token, dst, dst_stride as usize,
        tmp1.as_slice(), tmp2.as_slice(),
        w, h, mask.as_mut_slice(), sign as u8
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
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let token = unsafe { Arm64::forge_token_dangerously() };
    let w = w as usize;
    let h = h as usize;
    let dst = std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h * dst_stride.unsigned_abs());

    w_mask_8bpc_inner::<true, false>(
        token, dst, dst_stride as usize,
        tmp1.as_slice(), tmp2.as_slice(),
        w, h, mask.as_mut_slice(), sign as u8
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
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let token = unsafe { Arm64::forge_token_dangerously() };
    let w = w as usize;
    let h = h as usize;
    let dst = std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h * dst_stride.unsigned_abs());

    w_mask_8bpc_inner::<true, true>(
        token, dst, dst_stride as usize,
        tmp1.as_slice(), tmp2.as_slice(),
        w, h, mask.as_mut_slice(), sign as u8
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
                    unsafe {
                        let r0 = vld1_u8(src_row0[x..].as_ptr());
                        let r1 = vld1_u8(src_row1[x..].as_ptr());

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
                        vst1_u8(dst_row[x..].as_mut_ptr(), packed);
                    }
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
                    unsafe {
                        let s0 = vld1_u8(src_row[x..].as_ptr());
                        let s1 = vld1_u8(src_row[x + 1..].as_ptr());

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
                        vst1_u8(dst_row[x..].as_mut_ptr(), packed);
                    }
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
                    unsafe {
                        let r0 = vld1q_s16(mid_row0[x..].as_ptr());
                        let r1 = vld1q_s16(mid_row1[x..].as_ptr());

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

                        vst1_u8(dst_row[x..].as_mut_ptr(), result_u8);
                    }
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
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
    _src: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let token = unsafe { Arm64::forge_token_dangerously() };
    let w = w as usize;
    let h = h as usize;
    let src = std::slice::from_raw_parts(src_ptr as *const u8, (h + 1) * src_stride.unsigned_abs() + w + 1);
    let dst = std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h * dst_stride.unsigned_abs());

    put_bilin_8bpc_inner(token, dst, dst_stride as usize, src, src_stride as usize, w, h, mx, my);
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
                    unsafe {
                        let s = vld1_u8(src_row[x..].as_ptr());
                        let s16 = vreinterpretq_s16_u16(vmovl_u8(s));

                        // Scale: (pixel - 512) << 4 for intermediate format
                        // Actually for 8bpc: pixel << 4 with PREP_BIAS offset
                        let scaled = vshlq_n_s16::<4>(s16);
                        let biased = vsubq_s16(scaled, vdupq_n_s16(PREP_BIAS));

                        vst1q_s16(tmp_row[x..].as_mut_ptr(), biased);
                    }
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
                    unsafe {
                        let r0 = vld1_u8(src_row0[x..].as_ptr());
                        let r1 = vld1_u8(src_row1[x..].as_ptr());

                        let r0_16 = vreinterpretq_s16_u16(vmovl_u8(r0));
                        let r1_16 = vreinterpretq_s16_u16(vmovl_u8(r1));

                        let c0 = vdupq_n_s16(coeff0);
                        let c1 = vdupq_n_s16(coeff1);
                        let mul0 = vmulq_s16(r0_16, c0);
                        let mul1 = vmulq_s16(r1_16, c1);
                        let sum = vaddq_s16(mul0, mul1);

                        // For prep: no shift, just apply bias
                        let biased = vsubq_s16(sum, vdupq_n_s16(PREP_BIAS));

                        vst1q_s16(tmp_row[x..].as_mut_ptr(), biased);
                    }
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
                    unsafe {
                        let s0 = vld1_u8(src_row[x..].as_ptr());
                        let s1 = vld1_u8(src_row[x + 1..].as_ptr());

                        let s0_16 = vreinterpretq_s16_u16(vmovl_u8(s0));
                        let s1_16 = vreinterpretq_s16_u16(vmovl_u8(s1));

                        let c0 = vdupq_n_s16(coeff0);
                        let c1 = vdupq_n_s16(coeff1);
                        let mul0 = vmulq_s16(s0_16, c0);
                        let mul1 = vmulq_s16(s1_16, c1);
                        let sum = vaddq_s16(mul0, mul1);

                        let biased = vsubq_s16(sum, vdupq_n_s16(PREP_BIAS));

                        vst1q_s16(tmp_row[x..].as_mut_ptr(), biased);
                    }
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
                    unsafe {
                        let r0 = vld1q_s16(mid_row0[x..].as_ptr());
                        let r1 = vld1q_s16(mid_row1[x..].as_ptr());

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

                        vst1q_s16(tmp_row[x..].as_mut_ptr(), biased);
                    }
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
    _src: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let token = unsafe { Arm64::forge_token_dangerously() };
    let w = w as usize;
    let h = h as usize;
    let src = std::slice::from_raw_parts(src_ptr as *const u8, (h + 1) * src_stride.unsigned_abs() + w + 1);
    let tmp_slice = std::slice::from_raw_parts_mut(tmp, h * w);

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
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
    _src: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let token = unsafe { Arm64::forge_token_dangerously() };
    let w = w as usize;
    let h = h as usize;
    let dst_stride_u16 = (dst_stride / 2) as usize;
    let src_stride_u16 = (src_stride / 2) as usize;
    let dst = std::slice::from_raw_parts_mut(dst_ptr as *mut u16, h * dst_stride_u16);
    let src = std::slice::from_raw_parts(src_ptr as *const u16, (h + 1) * src_stride_u16 + w + 1);

    put_bilin_16bpc_inner(token, dst, dst_stride_u16, src, src_stride_u16, w, h, mx, my, bitdepth_max);
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
    _src: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let token = unsafe { Arm64::forge_token_dangerously() };
    let w = w as usize;
    let h = h as usize;
    let src_stride_u16 = (src_stride / 2) as usize;
    let src = std::slice::from_raw_parts(src_ptr as *const u16, (h + 1) * src_stride_u16 + w + 1);
    let tmp_slice = std::slice::from_raw_parts_mut(tmp, h * w);

    prep_bilin_16bpc_inner(token, tmp_slice, src, src_stride_u16, w, h, mx, my);
}

// ============================================================================
// W_MASK 16bpc - Weighted mask blend for 16-bit pixels
// ============================================================================

/// Core w_mask implementation for 16bpc
#[cfg(target_arch = "aarch64")]
#[arcane]
fn w_mask_16bpc_inner<const SS_HOR: bool, const SS_VER: bool>(
    _token: Arm64,
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
        let mask_row = if SS_VER && (y & 1) != 0 {
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
                        let m_next = std::cmp::min(38 + ((diff_next.saturating_add(mask_rnd)) >> mask_sh), 64) as u8;
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
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let token = unsafe { Arm64::forge_token_dangerously() };
    let w = w as usize;
    let h = h as usize;
    let dst_stride_u16 = (dst_stride / 2) as usize;
    let dst = std::slice::from_raw_parts_mut(dst_ptr as *mut u16, h * dst_stride_u16);

    w_mask_16bpc_inner::<false, false>(
        token, dst, dst_stride_u16,
        tmp1.as_slice(), tmp2.as_slice(),
        w, h, mask.as_mut_slice(), sign as u8, bitdepth_max
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
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let token = unsafe { Arm64::forge_token_dangerously() };
    let w = w as usize;
    let h = h as usize;
    let dst_stride_u16 = (dst_stride / 2) as usize;
    let dst = std::slice::from_raw_parts_mut(dst_ptr as *mut u16, h * dst_stride_u16);

    w_mask_16bpc_inner::<true, false>(
        token, dst, dst_stride_u16,
        tmp1.as_slice(), tmp2.as_slice(),
        w, h, mask.as_mut_slice(), sign as u8, bitdepth_max
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
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let token = unsafe { Arm64::forge_token_dangerously() };
    let w = w as usize;
    let h = h as usize;
    let dst_stride_u16 = (dst_stride / 2) as usize;
    let dst = std::slice::from_raw_parts_mut(dst_ptr as *mut u16, h * dst_stride_u16);

    w_mask_16bpc_inner::<true, true>(
        token, dst, dst_stride_u16,
        tmp1.as_slice(), tmp2.as_slice(),
        w, h, mask.as_mut_slice(), sign as u8, bitdepth_max
    );
}
