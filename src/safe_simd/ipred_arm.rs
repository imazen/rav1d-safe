//! Safe SIMD implementations of intra prediction functions for ARM NEON
//!
//! Replaces hand-written assembly with safe Rust intrinsics.

#![allow(unused)]

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
use archmage::{arcane, Arm64, SimdToken};

use libc::{c_int, ptrdiff_t};

use crate::include::common::bitdepth::DynPixel;
use crate::include::dav1d::picture::Rav1dPictureDataComponentOffset;
use crate::src::ffi_safe::FFISafe;

// ============================================================================
// DC_128 Prediction (fill with mid-value)
// ============================================================================

/// DC_128 prediction: fill block with 128 (8bpc) or 1 << (bitdepth - 1) (16bpc)
#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn ipred_dc_128_8bpc_neon(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    _topleft: *const DynPixel,
    width: c_int,
    height: c_int,
    _angle: c_int,
    _max_width: c_int,
    _max_height: c_int,
    _bitdepth_max: c_int,
    _topleft_off: usize,
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let width = width as usize;
    let height = height as usize;
    let dst = dst_ptr as *mut u8;

    unsafe {
        let fill_val = vdupq_n_u8(128);

        for y in 0..height {
            let dst_row = dst.offset(y as isize * stride);

            let mut x = 0;
            while x + 16 <= width {
                vst1q_u8(dst_row.add(x), fill_val);
                x += 16;
            }
            while x + 8 <= width {
                vst1_u8(dst_row.add(x), vget_low_u8(fill_val));
                x += 8;
            }
            while x < width {
                *dst_row.add(x) = 128;
                x += 1;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn ipred_dc_128_16bpc_neon(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    _topleft: *const DynPixel,
    width: c_int,
    height: c_int,
    _angle: c_int,
    _max_width: c_int,
    _max_height: c_int,
    bitdepth_max: c_int,
    _topleft_off: usize,
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let width = width as usize;
    let height = height as usize;
    let stride_u16 = (stride / 2) as usize;
    let dst = dst_ptr as *mut u16;
    let fill = ((bitdepth_max + 1) / 2) as u16;

    unsafe {
        let fill_val = vdupq_n_u16(fill);

        for y in 0..height {
            let dst_row = dst.add(y * stride_u16);

            let mut x = 0;
            while x + 8 <= width {
                vst1q_u16(dst_row.add(x), fill_val);
                x += 8;
            }
            while x + 4 <= width {
                vst1_u16(dst_row.add(x), vget_low_u16(fill_val));
                x += 4;
            }
            while x < width {
                *dst_row.add(x) = fill;
                x += 1;
            }
        }
    }
}

// ============================================================================
// Vertical Prediction (copy top row)
// ============================================================================

/// Vertical prediction: copy the top row to all rows in the block
#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn ipred_v_8bpc_neon(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    topleft: *const DynPixel,
    width: c_int,
    height: c_int,
    _angle: c_int,
    _max_width: c_int,
    _max_height: c_int,
    _bitdepth_max: c_int,
    _topleft_off: usize,
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let width = width as usize;
    let height = height as usize;
    let dst = dst_ptr as *mut u8;
    // Top pixels are at topleft + 1
    let top = (topleft as *const u8).add(1);

    unsafe {
        for y in 0..height {
            let dst_row = dst.offset(y as isize * stride);

            let mut x = 0;
            while x + 16 <= width {
                let top_vals = vld1q_u8(top.add(x));
                vst1q_u8(dst_row.add(x), top_vals);
                x += 16;
            }
            while x + 8 <= width {
                let top_vals = vld1_u8(top.add(x));
                vst1_u8(dst_row.add(x), top_vals);
                x += 8;
            }
            while x < width {
                *dst_row.add(x) = *top.add(x);
                x += 1;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn ipred_v_16bpc_neon(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    topleft: *const DynPixel,
    width: c_int,
    height: c_int,
    _angle: c_int,
    _max_width: c_int,
    _max_height: c_int,
    _bitdepth_max: c_int,
    _topleft_off: usize,
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let width = width as usize;
    let height = height as usize;
    let stride_u16 = (stride / 2) as usize;
    let dst = dst_ptr as *mut u16;
    let top = (topleft as *const u16).add(1);

    unsafe {
        for y in 0..height {
            let dst_row = dst.add(y * stride_u16);

            let mut x = 0;
            while x + 8 <= width {
                let top_vals = vld1q_u16(top.add(x));
                vst1q_u16(dst_row.add(x), top_vals);
                x += 8;
            }
            while x + 4 <= width {
                let top_vals = vld1_u16(top.add(x));
                vst1_u16(dst_row.add(x), top_vals);
                x += 4;
            }
            while x < width {
                *dst_row.add(x) = *top.add(x);
                x += 1;
            }
        }
    }
}

// ============================================================================
// Horizontal Prediction (fill from left pixels)
// ============================================================================

/// Horizontal prediction: fill each row with the left pixel
#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn ipred_h_8bpc_neon(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    topleft: *const DynPixel,
    width: c_int,
    height: c_int,
    _angle: c_int,
    _max_width: c_int,
    _max_height: c_int,
    _bitdepth_max: c_int,
    _topleft_off: usize,
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let width = width as usize;
    let height = height as usize;
    let dst = dst_ptr as *mut u8;
    // Left pixels are at topleft - y
    let left = topleft as *const u8;

    unsafe {
        for y in 0..height {
            let dst_row = dst.offset(y as isize * stride);
            let left_val = *left.offset(-(y as isize + 1));
            let fill_val = vdupq_n_u8(left_val);

            let mut x = 0;
            while x + 16 <= width {
                vst1q_u8(dst_row.add(x), fill_val);
                x += 16;
            }
            while x + 8 <= width {
                vst1_u8(dst_row.add(x), vget_low_u8(fill_val));
                x += 8;
            }
            while x < width {
                *dst_row.add(x) = left_val;
                x += 1;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn ipred_h_16bpc_neon(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    topleft: *const DynPixel,
    width: c_int,
    height: c_int,
    _angle: c_int,
    _max_width: c_int,
    _max_height: c_int,
    _bitdepth_max: c_int,
    _topleft_off: usize,
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let width = width as usize;
    let height = height as usize;
    let stride_u16 = (stride / 2) as usize;
    let dst = dst_ptr as *mut u16;
    let left = topleft as *const u16;

    unsafe {
        for y in 0..height {
            let dst_row = dst.add(y * stride_u16);
            let left_val = *left.offset(-(y as isize + 1));
            let fill_val = vdupq_n_u16(left_val);

            let mut x = 0;
            while x + 8 <= width {
                vst1q_u16(dst_row.add(x), fill_val);
                x += 8;
            }
            while x + 4 <= width {
                vst1_u16(dst_row.add(x), vget_low_u16(fill_val));
                x += 4;
            }
            while x < width {
                *dst_row.add(x) = left_val;
                x += 1;
            }
        }
    }
}

// ============================================================================
// DC Prediction (average of top and left)
// ============================================================================

/// DC prediction: average of top and left pixels
#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn ipred_dc_8bpc_neon(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    topleft: *const DynPixel,
    width: c_int,
    height: c_int,
    _angle: c_int,
    _max_width: c_int,
    _max_height: c_int,
    _bitdepth_max: c_int,
    _topleft_off: usize,
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let width = width as usize;
    let height = height as usize;
    let dst = dst_ptr as *mut u8;
    let top = (topleft as *const u8).add(1);
    let left = topleft as *const u8;

    // Calculate average of top and left pixels
    let mut sum = 0u32;
    for i in 0..width {
        sum += unsafe { *top.add(i) } as u32;
    }
    for i in 0..height {
        sum += unsafe { *left.offset(-(i as isize + 1)) } as u32;
    }
    let count = (width + height) as u32;
    let dc = ((sum + (count >> 1)) / count) as u8;

    unsafe {
        let fill_val = vdupq_n_u8(dc);

        for y in 0..height {
            let dst_row = dst.offset(y as isize * stride);

            let mut x = 0;
            while x + 16 <= width {
                vst1q_u8(dst_row.add(x), fill_val);
                x += 16;
            }
            while x + 8 <= width {
                vst1_u8(dst_row.add(x), vget_low_u8(fill_val));
                x += 8;
            }
            while x < width {
                *dst_row.add(x) = dc;
                x += 1;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn ipred_dc_16bpc_neon(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    topleft: *const DynPixel,
    width: c_int,
    height: c_int,
    _angle: c_int,
    _max_width: c_int,
    _max_height: c_int,
    _bitdepth_max: c_int,
    _topleft_off: usize,
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let width = width as usize;
    let height = height as usize;
    let stride_u16 = (stride / 2) as usize;
    let dst = dst_ptr as *mut u16;
    let top = (topleft as *const u16).add(1);
    let left = topleft as *const u16;

    let mut sum = 0u32;
    for i in 0..width {
        sum += unsafe { *top.add(i) } as u32;
    }
    for i in 0..height {
        sum += unsafe { *left.offset(-(i as isize + 1)) } as u32;
    }
    let count = (width + height) as u32;
    let dc = ((sum + (count >> 1)) / count) as u16;

    unsafe {
        let fill_val = vdupq_n_u16(dc);

        for y in 0..height {
            let dst_row = dst.add(y * stride_u16);

            let mut x = 0;
            while x + 8 <= width {
                vst1q_u16(dst_row.add(x), fill_val);
                x += 8;
            }
            while x + 4 <= width {
                vst1_u16(dst_row.add(x), vget_low_u16(fill_val));
                x += 4;
            }
            while x < width {
                *dst_row.add(x) = dc;
                x += 1;
            }
        }
    }
}

// ============================================================================
// DC_TOP Prediction (average of top only)
// ============================================================================

#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn ipred_dc_top_8bpc_neon(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    topleft: *const DynPixel,
    width: c_int,
    height: c_int,
    _angle: c_int,
    _max_width: c_int,
    _max_height: c_int,
    _bitdepth_max: c_int,
    _topleft_off: usize,
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let width = width as usize;
    let height = height as usize;
    let dst = dst_ptr as *mut u8;
    let top = (topleft as *const u8).add(1);

    let mut sum = 0u32;
    for i in 0..width {
        sum += unsafe { *top.add(i) } as u32;
    }
    let dc = ((sum + (width as u32 >> 1)) / width as u32) as u8;

    unsafe {
        let fill_val = vdupq_n_u8(dc);

        for y in 0..height {
            let dst_row = dst.offset(y as isize * stride);

            let mut x = 0;
            while x + 16 <= width {
                vst1q_u8(dst_row.add(x), fill_val);
                x += 16;
            }
            while x + 8 <= width {
                vst1_u8(dst_row.add(x), vget_low_u8(fill_val));
                x += 8;
            }
            while x < width {
                *dst_row.add(x) = dc;
                x += 1;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn ipred_dc_top_16bpc_neon(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    topleft: *const DynPixel,
    width: c_int,
    height: c_int,
    _angle: c_int,
    _max_width: c_int,
    _max_height: c_int,
    _bitdepth_max: c_int,
    _topleft_off: usize,
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let width = width as usize;
    let height = height as usize;
    let stride_u16 = (stride / 2) as usize;
    let dst = dst_ptr as *mut u16;
    let top = (topleft as *const u16).add(1);

    let mut sum = 0u32;
    for i in 0..width {
        sum += unsafe { *top.add(i) } as u32;
    }
    let dc = ((sum + (width as u32 >> 1)) / width as u32) as u16;

    unsafe {
        let fill_val = vdupq_n_u16(dc);

        for y in 0..height {
            let dst_row = dst.add(y * stride_u16);

            let mut x = 0;
            while x + 8 <= width {
                vst1q_u16(dst_row.add(x), fill_val);
                x += 8;
            }
            while x + 4 <= width {
                vst1_u16(dst_row.add(x), vget_low_u16(fill_val));
                x += 4;
            }
            while x < width {
                *dst_row.add(x) = dc;
                x += 1;
            }
        }
    }
}

// ============================================================================
// DC_LEFT Prediction (average of left only)
// ============================================================================

#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn ipred_dc_left_8bpc_neon(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    topleft: *const DynPixel,
    width: c_int,
    height: c_int,
    _angle: c_int,
    _max_width: c_int,
    _max_height: c_int,
    _bitdepth_max: c_int,
    _topleft_off: usize,
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let width = width as usize;
    let height = height as usize;
    let dst = dst_ptr as *mut u8;
    let left = topleft as *const u8;

    let mut sum = 0u32;
    for i in 0..height {
        sum += unsafe { *left.offset(-(i as isize + 1)) } as u32;
    }
    let dc = ((sum + (height as u32 >> 1)) / height as u32) as u8;

    unsafe {
        let fill_val = vdupq_n_u8(dc);

        for y in 0..height {
            let dst_row = dst.offset(y as isize * stride);

            let mut x = 0;
            while x + 16 <= width {
                vst1q_u8(dst_row.add(x), fill_val);
                x += 16;
            }
            while x + 8 <= width {
                vst1_u8(dst_row.add(x), vget_low_u8(fill_val));
                x += 8;
            }
            while x < width {
                *dst_row.add(x) = dc;
                x += 1;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn ipred_dc_left_16bpc_neon(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    topleft: *const DynPixel,
    width: c_int,
    height: c_int,
    _angle: c_int,
    _max_width: c_int,
    _max_height: c_int,
    _bitdepth_max: c_int,
    _topleft_off: usize,
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let width = width as usize;
    let height = height as usize;
    let stride_u16 = (stride / 2) as usize;
    let dst = dst_ptr as *mut u16;
    let left = topleft as *const u16;

    let mut sum = 0u32;
    for i in 0..height {
        sum += unsafe { *left.offset(-(i as isize + 1)) } as u32;
    }
    let dc = ((sum + (height as u32 >> 1)) / height as u32) as u16;

    unsafe {
        let fill_val = vdupq_n_u16(dc);

        for y in 0..height {
            let dst_row = dst.add(y * stride_u16);

            let mut x = 0;
            while x + 8 <= width {
                vst1q_u16(dst_row.add(x), fill_val);
                x += 8;
            }
            while x + 4 <= width {
                vst1_u16(dst_row.add(x), vget_low_u16(fill_val));
                x += 4;
            }
            while x < width {
                *dst_row.add(x) = dc;
                x += 1;
            }
        }
    }
}
