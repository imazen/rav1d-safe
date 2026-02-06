//! Safe SIMD implementations of intra prediction functions for ARM NEON
//!
//! Replaces hand-written assembly with safe Rust intrinsics.

#![allow(unused)]
#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
use archmage::{arcane, Arm64, SimdToken};

use libc::{c_int, ptrdiff_t};

use crate::include::common::bitdepth::DynPixel;
use crate::include::dav1d::picture::PicOffset;
use crate::src::ffi_safe::FFISafe;

// ============================================================================
// DC_128 Prediction (fill with mid-value)
// ============================================================================
#[cfg(any(feature = "asm", feature = "c-ffi"))]

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
    _dst: *const FFISafe<PicOffset>,
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
#[cfg(any(feature = "asm", feature = "c-ffi"))]

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
    _dst: *const FFISafe<PicOffset>,
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
#[cfg(any(feature = "asm", feature = "c-ffi"))]

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
    _dst: *const FFISafe<PicOffset>,
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
#[cfg(any(feature = "asm", feature = "c-ffi"))]

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
    _dst: *const FFISafe<PicOffset>,
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
#[cfg(any(feature = "asm", feature = "c-ffi"))]

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
    _dst: *const FFISafe<PicOffset>,
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
#[cfg(any(feature = "asm", feature = "c-ffi"))]

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
    _dst: *const FFISafe<PicOffset>,
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
#[cfg(any(feature = "asm", feature = "c-ffi"))]

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
    _dst: *const FFISafe<PicOffset>,
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
#[cfg(any(feature = "asm", feature = "c-ffi"))]

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
    _dst: *const FFISafe<PicOffset>,
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
#[cfg(any(feature = "asm", feature = "c-ffi"))]

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
    _dst: *const FFISafe<PicOffset>,
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
#[cfg(any(feature = "asm", feature = "c-ffi"))]

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
    _dst: *const FFISafe<PicOffset>,
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
#[cfg(any(feature = "asm", feature = "c-ffi"))]

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
    _dst: *const FFISafe<PicOffset>,
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
#[cfg(any(feature = "asm", feature = "c-ffi"))]

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
    _dst: *const FFISafe<PicOffset>,
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

// ============================================================================
// Paeth Prediction
// ============================================================================

use crate::src::tables::dav1d_sm_weights;

/// Helper: Paeth predictor
#[inline(always)]
fn paeth(left: i32, top: i32, topleft: i32) -> i32 {
    let base = left + top - topleft;
    let p_left = (base - left).abs();
    let p_top = (base - top).abs();
    let p_tl = (base - topleft).abs();
    
    if p_left <= p_top && p_left <= p_tl {
        left
    } else if p_top <= p_tl {
        top
    } else {
        topleft
    }
}
#[cfg(any(feature = "asm", feature = "c-ffi"))]

#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn ipred_paeth_8bpc_neon(
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
    _dst: *const FFISafe<PicOffset>,
) {
    let width = width as usize;
    let height = height as usize;
    let dst = dst_ptr as *mut u8;
    let tl = topleft as *const u8;
    
    // topleft pixel is at offset 0
    let topleft_val = unsafe { *tl } as i32;
    
    for y in 0..height {
        let dst_row = unsafe { dst.offset(y as isize * stride) };
        let left_val = unsafe { *tl.offset(-(y as isize) - 1) } as i32;
        
        for x in 0..width {
            let top_val = unsafe { *tl.add(x + 1) } as i32;
            let pred = paeth(left_val, top_val, topleft_val);
            unsafe { *dst_row.add(x) = pred as u8; }
        }
    }
}
#[cfg(any(feature = "asm", feature = "c-ffi"))]

#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn ipred_paeth_16bpc_neon(
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
    _dst: *const FFISafe<PicOffset>,
) {
    let width = width as usize;
    let height = height as usize;
    let stride_u16 = (stride / 2) as usize;
    let dst = dst_ptr as *mut u16;
    let tl = topleft as *const u16;
    
    let topleft_val = unsafe { *tl } as i32;
    
    for y in 0..height {
        let dst_row = unsafe { dst.add(y * stride_u16) };
        let left_val = unsafe { *tl.offset(-(y as isize) - 1) } as i32;
        
        for x in 0..width {
            let top_val = unsafe { *tl.add(x + 1) } as i32;
            let pred = paeth(left_val, top_val, topleft_val);
            unsafe { *dst_row.add(x) = pred as u16; }
        }
    }
}

// ============================================================================
// Smooth Prediction
// ============================================================================
#[cfg(any(feature = "asm", feature = "c-ffi"))]

#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn ipred_smooth_8bpc_neon(
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
    _dst: *const FFISafe<PicOffset>,
) {
    let width = width as usize;
    let height = height as usize;
    let dst = dst_ptr as *mut u8;
    let tl = topleft as *const u8;
    
    let weights_hor = &dav1d_sm_weights.0[width..][..width];
    let weights_ver = &dav1d_sm_weights.0[height..][..height];
    let right_val = unsafe { *tl.add(width) } as i32;
    let bottom_val = unsafe { *tl.offset(-(height as isize)) } as i32;
    
    for y in 0..height {
        let dst_row = unsafe { dst.offset(y as isize * stride) };
        let left_val = unsafe { *tl.offset(-(y as isize) - 1) } as i32;
        let w_v = weights_ver[y] as i32;
        
        for x in 0..width {
            let top_val = unsafe { *tl.add(x + 1) } as i32;
            let w_h = weights_hor[x] as i32;
            
            // Vertical: w_v * top + (256 - w_v) * bottom
            let vert = w_v * top_val + (256 - w_v) * bottom_val;
            
            // Horizontal: w_h * left + (256 - w_h) * right
            let hor = w_h * left_val + (256 - w_h) * right_val;
            
            // Result: (vert + hor + 256) >> 9
            let pred = (vert + hor + 256) >> 9;
            unsafe { *dst_row.add(x) = pred as u8; }
        }
    }
}
#[cfg(any(feature = "asm", feature = "c-ffi"))]

#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn ipred_smooth_16bpc_neon(
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
    _dst: *const FFISafe<PicOffset>,
) {
    let width = width as usize;
    let height = height as usize;
    let stride_u16 = (stride / 2) as usize;
    let dst = dst_ptr as *mut u16;
    let tl = topleft as *const u16;
    
    let weights_hor = &dav1d_sm_weights.0[width..][..width];
    let weights_ver = &dav1d_sm_weights.0[height..][..height];
    let right_val = unsafe { *tl.add(width) } as i32;
    let bottom_val = unsafe { *tl.offset(-(height as isize)) } as i32;
    
    for y in 0..height {
        let dst_row = unsafe { dst.add(y * stride_u16) };
        let left_val = unsafe { *tl.offset(-(y as isize) - 1) } as i32;
        let w_v = weights_ver[y] as i32;
        
        for x in 0..width {
            let top_val = unsafe { *tl.add(x + 1) } as i32;
            let w_h = weights_hor[x] as i32;
            
            let vert = w_v * top_val + (256 - w_v) * bottom_val;
            let hor = w_h * left_val + (256 - w_h) * right_val;
            let pred = (vert + hor + 256) >> 9;
            unsafe { *dst_row.add(x) = pred as u16; }
        }
    }
}

// ============================================================================
// Smooth V Prediction (vertical only)
// ============================================================================
#[cfg(any(feature = "asm", feature = "c-ffi"))]

#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn ipred_smooth_v_8bpc_neon(
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
    _dst: *const FFISafe<PicOffset>,
) {
    let width = width as usize;
    let height = height as usize;
    let dst = dst_ptr as *mut u8;
    let tl = topleft as *const u8;
    
    let weights_ver = &dav1d_sm_weights.0[height..][..height];
    let bottom_val = unsafe { *tl.offset(-(height as isize)) } as i32;
    
    for y in 0..height {
        let dst_row = unsafe { dst.offset(y as isize * stride) };
        let w_v = weights_ver[y] as i32;
        
        for x in 0..width {
            let top_val = unsafe { *tl.add(x + 1) } as i32;
            let pred = (w_v * top_val + (256 - w_v) * bottom_val + 128) >> 8;
            unsafe { *dst_row.add(x) = pred as u8; }
        }
    }
}
#[cfg(any(feature = "asm", feature = "c-ffi"))]

#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn ipred_smooth_v_16bpc_neon(
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
    _dst: *const FFISafe<PicOffset>,
) {
    let width = width as usize;
    let height = height as usize;
    let stride_u16 = (stride / 2) as usize;
    let dst = dst_ptr as *mut u16;
    let tl = topleft as *const u16;
    
    let weights_ver = &dav1d_sm_weights.0[height..][..height];
    let bottom_val = unsafe { *tl.offset(-(height as isize)) } as i32;
    
    for y in 0..height {
        let dst_row = unsafe { dst.add(y * stride_u16) };
        let w_v = weights_ver[y] as i32;
        
        for x in 0..width {
            let top_val = unsafe { *tl.add(x + 1) } as i32;
            let pred = (w_v * top_val + (256 - w_v) * bottom_val + 128) >> 8;
            unsafe { *dst_row.add(x) = pred as u16; }
        }
    }
}

// ============================================================================
// Smooth H Prediction (horizontal only)
// ============================================================================
#[cfg(any(feature = "asm", feature = "c-ffi"))]

#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn ipred_smooth_h_8bpc_neon(
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
    _dst: *const FFISafe<PicOffset>,
) {
    let width = width as usize;
    let height = height as usize;
    let dst = dst_ptr as *mut u8;
    let tl = topleft as *const u8;
    
    let weights_hor = &dav1d_sm_weights.0[width..][..width];
    let right_val = unsafe { *tl.add(width) } as i32;
    
    for y in 0..height {
        let dst_row = unsafe { dst.offset(y as isize * stride) };
        let left_val = unsafe { *tl.offset(-(y as isize) - 1) } as i32;
        
        for x in 0..width {
            let w_h = weights_hor[x] as i32;
            let pred = (w_h * left_val + (256 - w_h) * right_val + 128) >> 8;
            unsafe { *dst_row.add(x) = pred as u8; }
        }
    }
}
#[cfg(any(feature = "asm", feature = "c-ffi"))]

#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn ipred_smooth_h_16bpc_neon(
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
    _dst: *const FFISafe<PicOffset>,
) {
    let width = width as usize;
    let height = height as usize;
    let stride_u16 = (stride / 2) as usize;
    let dst = dst_ptr as *mut u16;
    let tl = topleft as *const u16;
    
    let weights_hor = &dav1d_sm_weights.0[width..][..width];
    let right_val = unsafe { *tl.add(width) } as i32;
    
    for y in 0..height {
        let dst_row = unsafe { dst.add(y * stride_u16) };
        let left_val = unsafe { *tl.offset(-(y as isize) - 1) } as i32;
        
        for x in 0..width {
            let w_h = weights_hor[x] as i32;
            let pred = (w_h * left_val + (256 - w_h) * right_val + 128) >> 8;
            unsafe { *dst_row.add(x) = pred as u16; }
        }
    }
}
