//! Safe SIMD implementations of intra prediction functions for ARM NEON
//!
//! Replaces hand-written assembly with safe Rust intrinsics.

#![allow(unused)]
#![cfg_attr(not(feature = "unchecked"), forbid(unsafe_code))]
#![cfg_attr(feature = "unchecked", deny(unsafe_code))]

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
use archmage::{Arm64, SimdToken, arcane};

#[cfg(target_arch = "aarch64")]
use safe_unaligned_simd::aarch64 as safe_simd;

use std::ffi::c_int;
#[allow(non_camel_case_types)]
type ptrdiff_t = isize;

use crate::include::common::bitdepth::DynPixel;
use crate::include::dav1d::picture::PicOffset;
use crate::src::ffi_safe::FFISafe;

#[cfg(feature = "asm")]
mod ffi {
    use super::*;

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
        _dst: *const FFISafe<PicOffset>,
    ) {
        let width = width as usize;
        let height = height as usize;
        let dst = dst_ptr as *mut u8;

        let fill_val = unsafe { vdupq_n_u8(128) };

        for y in 0..height {
            let dst_row = unsafe { dst.offset(y as isize * stride) };

            let mut x = 0;
            while x + 16 <= width {
                unsafe {
                    vst1q_u8(dst_row.add(x), fill_val);
                }
                x += 16;
            }
            while x + 8 <= width {
                unsafe {
                    vst1_u8(dst_row.add(x), vget_low_u8(fill_val));
                }
                x += 8;
            }
            while x < width {
                unsafe {
                    *dst_row.add(x) = 128;
                }
                x += 1;
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
        _dst: *const FFISafe<PicOffset>,
    ) {
        let width = width as usize;
        let height = height as usize;
        let stride_u16 = (stride / 2) as usize;
        let dst = dst_ptr as *mut u16;
        let fill = ((bitdepth_max + 1) / 2) as u16;

        let fill_val = unsafe { vdupq_n_u16(fill) };

        for y in 0..height {
            let dst_row = unsafe { dst.add(y * stride_u16) };

            let mut x = 0;
            while x + 8 <= width {
                unsafe {
                    vst1q_u16(dst_row.add(x), fill_val);
                }
                x += 8;
            }
            while x + 4 <= width {
                unsafe {
                    vst1_u16(dst_row.add(x), vget_low_u16(fill_val));
                }
                x += 4;
            }
            while x < width {
                unsafe {
                    *dst_row.add(x) = fill;
                }
                x += 1;
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
        _dst: *const FFISafe<PicOffset>,
    ) {
        let width = width as usize;
        let height = height as usize;
        let dst = dst_ptr as *mut u8;
        // Top pixels are at topleft + 1
        let top = unsafe { (topleft as *const u8).add(1) };

        for y in 0..height {
            let dst_row = unsafe { dst.offset(y as isize * stride) };

            let mut x = 0;
            while x + 16 <= width {
                let top_vals = unsafe { vld1q_u8(top.add(x)) };
                unsafe {
                    vst1q_u8(dst_row.add(x), top_vals);
                }
                x += 16;
            }
            while x + 8 <= width {
                let top_vals = unsafe { vld1_u8(top.add(x)) };
                unsafe {
                    vst1_u8(dst_row.add(x), top_vals);
                }
                x += 8;
            }
            while x < width {
                unsafe {
                    *dst_row.add(x) = *top.add(x);
                }
                x += 1;
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
        _dst: *const FFISafe<PicOffset>,
    ) {
        let width = width as usize;
        let height = height as usize;
        let stride_u16 = (stride / 2) as usize;
        let dst = dst_ptr as *mut u16;
        let top = unsafe { (topleft as *const u16).add(1) };

        for y in 0..height {
            let dst_row = unsafe { dst.add(y * stride_u16) };

            let mut x = 0;
            while x + 8 <= width {
                let top_vals = unsafe { vld1q_u16(top.add(x)) };
                unsafe {
                    vst1q_u16(dst_row.add(x), top_vals);
                }
                x += 8;
            }
            while x + 4 <= width {
                let top_vals = unsafe { vld1_u16(top.add(x)) };
                unsafe {
                    vst1_u16(dst_row.add(x), top_vals);
                }
                x += 4;
            }
            while x < width {
                unsafe {
                    *dst_row.add(x) = *top.add(x);
                }
                x += 1;
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
        _dst: *const FFISafe<PicOffset>,
    ) {
        let width = width as usize;
        let height = height as usize;
        let dst = dst_ptr as *mut u8;
        // Left pixels are at topleft - y
        let left = topleft as *const u8;

        for y in 0..height {
            let dst_row = unsafe { dst.offset(y as isize * stride) };
            let left_val = unsafe { *left.offset(-(y as isize + 1)) };
            let fill_val = unsafe { vdupq_n_u8(left_val) };

            let mut x = 0;
            while x + 16 <= width {
                unsafe {
                    vst1q_u8(dst_row.add(x), fill_val);
                }
                x += 16;
            }
            while x + 8 <= width {
                unsafe {
                    vst1_u8(dst_row.add(x), vget_low_u8(fill_val));
                }
                x += 8;
            }
            while x < width {
                unsafe {
                    *dst_row.add(x) = left_val;
                }
                x += 1;
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
        _dst: *const FFISafe<PicOffset>,
    ) {
        let width = width as usize;
        let height = height as usize;
        let stride_u16 = (stride / 2) as usize;
        let dst = dst_ptr as *mut u16;
        let left = topleft as *const u16;

        for y in 0..height {
            let dst_row = unsafe { dst.add(y * stride_u16) };
            let left_val = unsafe { *left.offset(-(y as isize + 1)) };
            let fill_val = unsafe { vdupq_n_u16(left_val) };

            let mut x = 0;
            while x + 8 <= width {
                unsafe {
                    vst1q_u16(dst_row.add(x), fill_val);
                }
                x += 8;
            }
            while x + 4 <= width {
                unsafe {
                    vst1_u16(dst_row.add(x), vget_low_u16(fill_val));
                }
                x += 4;
            }
            while x < width {
                unsafe {
                    *dst_row.add(x) = left_val;
                }
                x += 1;
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
        _dst: *const FFISafe<PicOffset>,
    ) {
        let width = width as usize;
        let height = height as usize;
        let dst = dst_ptr as *mut u8;
        let top = unsafe { (topleft as *const u8).add(1) };
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

        let fill_val = unsafe { vdupq_n_u8(dc) };

        for y in 0..height {
            let dst_row = unsafe { dst.offset(y as isize * stride) };

            let mut x = 0;
            while x + 16 <= width {
                unsafe {
                    vst1q_u8(dst_row.add(x), fill_val);
                }
                x += 16;
            }
            while x + 8 <= width {
                unsafe {
                    vst1_u8(dst_row.add(x), vget_low_u8(fill_val));
                }
                x += 8;
            }
            while x < width {
                unsafe {
                    *dst_row.add(x) = dc;
                }
                x += 1;
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
        _dst: *const FFISafe<PicOffset>,
    ) {
        let width = width as usize;
        let height = height as usize;
        let stride_u16 = (stride / 2) as usize;
        let dst = dst_ptr as *mut u16;
        let top = unsafe { (topleft as *const u16).add(1) };
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

        let fill_val = unsafe { vdupq_n_u16(dc) };

        for y in 0..height {
            let dst_row = unsafe { dst.add(y * stride_u16) };

            let mut x = 0;
            while x + 8 <= width {
                unsafe {
                    vst1q_u16(dst_row.add(x), fill_val);
                }
                x += 8;
            }
            while x + 4 <= width {
                unsafe {
                    vst1_u16(dst_row.add(x), vget_low_u16(fill_val));
                }
                x += 4;
            }
            while x < width {
                unsafe {
                    *dst_row.add(x) = dc;
                }
                x += 1;
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
        _dst: *const FFISafe<PicOffset>,
    ) {
        let width = width as usize;
        let height = height as usize;
        let dst = dst_ptr as *mut u8;
        let top = unsafe { (topleft as *const u8).add(1) };

        let mut sum = 0u32;
        for i in 0..width {
            sum += unsafe { *top.add(i) } as u32;
        }
        let dc = ((sum + (width as u32 >> 1)) / width as u32) as u8;

        let fill_val = unsafe { vdupq_n_u8(dc) };

        for y in 0..height {
            let dst_row = unsafe { dst.offset(y as isize * stride) };

            let mut x = 0;
            while x + 16 <= width {
                unsafe {
                    vst1q_u8(dst_row.add(x), fill_val);
                }
                x += 16;
            }
            while x + 8 <= width {
                unsafe {
                    vst1_u8(dst_row.add(x), vget_low_u8(fill_val));
                }
                x += 8;
            }
            while x < width {
                unsafe {
                    *dst_row.add(x) = dc;
                }
                x += 1;
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
        _dst: *const FFISafe<PicOffset>,
    ) {
        let width = width as usize;
        let height = height as usize;
        let stride_u16 = (stride / 2) as usize;
        let dst = dst_ptr as *mut u16;
        let top = unsafe { (topleft as *const u16).add(1) };

        let mut sum = 0u32;
        for i in 0..width {
            sum += unsafe { *top.add(i) } as u32;
        }
        let dc = ((sum + (width as u32 >> 1)) / width as u32) as u16;

        let fill_val = unsafe { vdupq_n_u16(dc) };

        for y in 0..height {
            let dst_row = unsafe { dst.add(y * stride_u16) };

            let mut x = 0;
            while x + 8 <= width {
                unsafe {
                    vst1q_u16(dst_row.add(x), fill_val);
                }
                x += 8;
            }
            while x + 4 <= width {
                unsafe {
                    vst1_u16(dst_row.add(x), vget_low_u16(fill_val));
                }
                x += 4;
            }
            while x < width {
                unsafe {
                    *dst_row.add(x) = dc;
                }
                x += 1;
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

        let fill_val = unsafe { vdupq_n_u8(dc) };

        for y in 0..height {
            let dst_row = unsafe { dst.offset(y as isize * stride) };

            let mut x = 0;
            while x + 16 <= width {
                unsafe {
                    vst1q_u8(dst_row.add(x), fill_val);
                }
                x += 16;
            }
            while x + 8 <= width {
                unsafe {
                    vst1_u8(dst_row.add(x), vget_low_u8(fill_val));
                }
                x += 8;
            }
            while x < width {
                unsafe {
                    *dst_row.add(x) = dc;
                }
                x += 1;
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

        let fill_val = unsafe { vdupq_n_u16(dc) };

        for y in 0..height {
            let dst_row = unsafe { dst.add(y * stride_u16) };

            let mut x = 0;
            while x + 8 <= width {
                unsafe {
                    vst1q_u16(dst_row.add(x), fill_val);
                }
                x += 8;
            }
            while x + 4 <= width {
                unsafe {
                    vst1_u16(dst_row.add(x), vget_low_u16(fill_val));
                }
                x += 4;
            }
            while x < width {
                unsafe {
                    *dst_row.add(x) = dc;
                }
                x += 1;
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
                unsafe {
                    *dst_row.add(x) = pred as u8;
                }
            }
        }
    }

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
                unsafe {
                    *dst_row.add(x) = pred as u16;
                }
            }
        }
    }

    // ============================================================================
    // Smooth Prediction
    // ============================================================================

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

        let weights_hor = &dav1d_sm_weights[width..][..width];
        let weights_ver = &dav1d_sm_weights[height..][..height];
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
                unsafe {
                    *dst_row.add(x) = pred as u8;
                }
            }
        }
    }

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

        let weights_hor = &dav1d_sm_weights[width..][..width];
        let weights_ver = &dav1d_sm_weights[height..][..height];
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
                unsafe {
                    *dst_row.add(x) = pred as u16;
                }
            }
        }
    }

    // ============================================================================
    // Smooth V Prediction (vertical only)
    // ============================================================================

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

        let weights_ver = &dav1d_sm_weights[height..][..height];
        let bottom_val = unsafe { *tl.offset(-(height as isize)) } as i32;

        for y in 0..height {
            let dst_row = unsafe { dst.offset(y as isize * stride) };
            let w_v = weights_ver[y] as i32;

            for x in 0..width {
                let top_val = unsafe { *tl.add(x + 1) } as i32;
                let pred = (w_v * top_val + (256 - w_v) * bottom_val + 128) >> 8;
                unsafe {
                    *dst_row.add(x) = pred as u8;
                }
            }
        }
    }

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

        let weights_ver = &dav1d_sm_weights[height..][..height];
        let bottom_val = unsafe { *tl.offset(-(height as isize)) } as i32;

        for y in 0..height {
            let dst_row = unsafe { dst.add(y * stride_u16) };
            let w_v = weights_ver[y] as i32;

            for x in 0..width {
                let top_val = unsafe { *tl.add(x + 1) } as i32;
                let pred = (w_v * top_val + (256 - w_v) * bottom_val + 128) >> 8;
                unsafe {
                    *dst_row.add(x) = pred as u16;
                }
            }
        }
    }

    // ============================================================================
    // Smooth H Prediction (horizontal only)
    // ============================================================================

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

        let weights_hor = &dav1d_sm_weights[width..][..width];
        let right_val = unsafe { *tl.add(width) } as i32;

        for y in 0..height {
            let dst_row = unsafe { dst.offset(y as isize * stride) };
            let left_val = unsafe { *tl.offset(-(y as isize) - 1) } as i32;

            for x in 0..width {
                let w_h = weights_hor[x] as i32;
                let pred = (w_h * left_val + (256 - w_h) * right_val + 128) >> 8;
                unsafe {
                    *dst_row.add(x) = pred as u8;
                }
            }
        }
    }

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

        let weights_hor = &dav1d_sm_weights[width..][..width];
        let right_val = unsafe { *tl.add(width) } as i32;

        for y in 0..height {
            let dst_row = unsafe { dst.add(y * stride_u16) };
            let left_val = unsafe { *tl.offset(-(y as isize) - 1) } as i32;

            for x in 0..width {
                let w_h = weights_hor[x] as i32;
                let pred = (w_h * left_val + (256 - w_h) * right_val + 128) >> 8;
                unsafe {
                    *dst_row.add(x) = pred as u16;
                }
            }
        }
    }
} // mod ffi

#[cfg(feature = "asm")]
pub use ffi::*;

// ============================================================================
// Safe dispatch wrapper for aarch64 NEON
// ============================================================================

#[cfg(all(feature = "asm", target_arch = "aarch64"))]
use crate::include::common::bitdepth::BitDepth;
#[cfg(all(feature = "asm", target_arch = "aarch64"))]
use crate::src::internal::SCRATCH_EDGE_LEN;
#[cfg(all(feature = "asm", target_arch = "aarch64"))]
use crate::src::strided::Strided as _;

/// Safe dispatch for intra prediction on ARM. Returns true if SIMD was used.
/// NEON is always available on aarch64, so this always returns true for
/// supported modes and false only for unimplemented modes (Z1, Z2, Z3, FILTER).
#[cfg(all(feature = "asm", target_arch = "aarch64"))]
#[allow(unsafe_code)]
pub fn intra_pred_dispatch<BD: BitDepth>(
    mode: usize,
    dst: PicOffset,
    topleft: &[BD::Pixel; SCRATCH_EDGE_LEN],
    topleft_off: usize,
    width: c_int,
    height: c_int,
    angle: c_int,
    max_width: c_int,
    max_height: c_int,
    bd: BD,
) -> bool {
    // Ablation switch (measurement only; const-false without `__ablate`).
    if crate::src::ablate::is_off(crate::src::ablate::Family::IntraPred) {
        return false;
    }
    use crate::include::common::bitdepth::BPC;
    use zerocopy::IntoBytes;

    let w = width as usize;
    let h = height as usize;
    let stride = dst.stride();
    let bd_c = bd.into_c();
    let dst_ffi = FFISafe::new(&dst);

    // Create tracked guard — ensures borrow tracker knows about this access
    let (mut dst_guard, _dst_base) = dst.strided_slice_mut::<BD>(w, h);
    // Get pointer from guard's slice (tracked, not from Pixels)
    let dst_ptr: *mut DynPixel = dst_guard.as_mut_bytes().as_mut_ptr() as *mut DynPixel;
    // topleft is already a safe slice, get pointer for FFI
    let topleft_ptr: *const DynPixel = topleft.as_bytes()
        [topleft_off * std::mem::size_of::<BD::Pixel>()..]
        .as_ptr() as *const DynPixel;

    // SAFETY: NEON always available on aarch64. Pointers derived from tracked guard.
    let handled = unsafe {
        match (BD::BPC, mode) {
            (BPC::BPC8, 0) => {
                ipred_dc_8bpc_neon(
                    dst_ptr,
                    stride,
                    topleft_ptr,
                    width,
                    height,
                    angle,
                    max_width,
                    max_height,
                    bd_c,
                    topleft_off,
                    dst_ffi,
                );
                true
            }
            (BPC::BPC8, 1) => {
                ipred_v_8bpc_neon(
                    dst_ptr,
                    stride,
                    topleft_ptr,
                    width,
                    height,
                    angle,
                    max_width,
                    max_height,
                    bd_c,
                    topleft_off,
                    dst_ffi,
                );
                true
            }
            (BPC::BPC8, 2) => {
                ipred_h_8bpc_neon(
                    dst_ptr,
                    stride,
                    topleft_ptr,
                    width,
                    height,
                    angle,
                    max_width,
                    max_height,
                    bd_c,
                    topleft_off,
                    dst_ffi,
                );
                true
            }
            (BPC::BPC8, 3) => {
                ipred_dc_left_8bpc_neon(
                    dst_ptr,
                    stride,
                    topleft_ptr,
                    width,
                    height,
                    angle,
                    max_width,
                    max_height,
                    bd_c,
                    topleft_off,
                    dst_ffi,
                );
                true
            }
            (BPC::BPC8, 4) => {
                ipred_dc_top_8bpc_neon(
                    dst_ptr,
                    stride,
                    topleft_ptr,
                    width,
                    height,
                    angle,
                    max_width,
                    max_height,
                    bd_c,
                    topleft_off,
                    dst_ffi,
                );
                true
            }
            (BPC::BPC8, 5) => {
                ipred_dc_128_8bpc_neon(
                    dst_ptr,
                    stride,
                    topleft_ptr,
                    width,
                    height,
                    angle,
                    max_width,
                    max_height,
                    bd_c,
                    topleft_off,
                    dst_ffi,
                );
                true
            }
            (BPC::BPC8, 9) => {
                ipred_smooth_8bpc_neon(
                    dst_ptr,
                    stride,
                    topleft_ptr,
                    width,
                    height,
                    angle,
                    max_width,
                    max_height,
                    bd_c,
                    topleft_off,
                    dst_ffi,
                );
                true
            }
            (BPC::BPC8, 10) => {
                ipred_smooth_v_8bpc_neon(
                    dst_ptr,
                    stride,
                    topleft_ptr,
                    width,
                    height,
                    angle,
                    max_width,
                    max_height,
                    bd_c,
                    topleft_off,
                    dst_ffi,
                );
                true
            }
            (BPC::BPC8, 11) => {
                ipred_smooth_h_8bpc_neon(
                    dst_ptr,
                    stride,
                    topleft_ptr,
                    width,
                    height,
                    angle,
                    max_width,
                    max_height,
                    bd_c,
                    topleft_off,
                    dst_ffi,
                );
                true
            }
            (BPC::BPC8, 12) => {
                ipred_paeth_8bpc_neon(
                    dst_ptr,
                    stride,
                    topleft_ptr,
                    width,
                    height,
                    angle,
                    max_width,
                    max_height,
                    bd_c,
                    topleft_off,
                    dst_ffi,
                );
                true
            }
            (BPC::BPC16, 0) => {
                ipred_dc_16bpc_neon(
                    dst_ptr,
                    stride,
                    topleft_ptr,
                    width,
                    height,
                    angle,
                    max_width,
                    max_height,
                    bd_c,
                    topleft_off,
                    dst_ffi,
                );
                true
            }
            (BPC::BPC16, 1) => {
                ipred_v_16bpc_neon(
                    dst_ptr,
                    stride,
                    topleft_ptr,
                    width,
                    height,
                    angle,
                    max_width,
                    max_height,
                    bd_c,
                    topleft_off,
                    dst_ffi,
                );
                true
            }
            (BPC::BPC16, 2) => {
                ipred_h_16bpc_neon(
                    dst_ptr,
                    stride,
                    topleft_ptr,
                    width,
                    height,
                    angle,
                    max_width,
                    max_height,
                    bd_c,
                    topleft_off,
                    dst_ffi,
                );
                true
            }
            (BPC::BPC16, 3) => {
                ipred_dc_left_16bpc_neon(
                    dst_ptr,
                    stride,
                    topleft_ptr,
                    width,
                    height,
                    angle,
                    max_width,
                    max_height,
                    bd_c,
                    topleft_off,
                    dst_ffi,
                );
                true
            }
            (BPC::BPC16, 4) => {
                ipred_dc_top_16bpc_neon(
                    dst_ptr,
                    stride,
                    topleft_ptr,
                    width,
                    height,
                    angle,
                    max_width,
                    max_height,
                    bd_c,
                    topleft_off,
                    dst_ffi,
                );
                true
            }
            (BPC::BPC16, 5) => {
                ipred_dc_128_16bpc_neon(
                    dst_ptr,
                    stride,
                    topleft_ptr,
                    width,
                    height,
                    angle,
                    max_width,
                    max_height,
                    bd_c,
                    topleft_off,
                    dst_ffi,
                );
                true
            }
            (BPC::BPC16, 9) => {
                ipred_smooth_16bpc_neon(
                    dst_ptr,
                    stride,
                    topleft_ptr,
                    width,
                    height,
                    angle,
                    max_width,
                    max_height,
                    bd_c,
                    topleft_off,
                    dst_ffi,
                );
                true
            }
            (BPC::BPC16, 10) => {
                ipred_smooth_v_16bpc_neon(
                    dst_ptr,
                    stride,
                    topleft_ptr,
                    width,
                    height,
                    angle,
                    max_width,
                    max_height,
                    bd_c,
                    topleft_off,
                    dst_ffi,
                );
                true
            }
            (BPC::BPC16, 11) => {
                ipred_smooth_h_16bpc_neon(
                    dst_ptr,
                    stride,
                    topleft_ptr,
                    width,
                    height,
                    angle,
                    max_width,
                    max_height,
                    bd_c,
                    topleft_off,
                    dst_ffi,
                );
                true
            }
            (BPC::BPC16, 12) => {
                ipred_paeth_16bpc_neon(
                    dst_ptr,
                    stride,
                    topleft_ptr,
                    width,
                    height,
                    angle,
                    max_width,
                    max_height,
                    bd_c,
                    topleft_off,
                    dst_ffi,
                );
                true
            }
            _ => false,
        }
    };
    handled
}

// ============================================================================
// Chroma-from-luma prediction (CfL)
// ============================================================================
//
// `src/ipred.rs::cfl_pred` was scalar on aarch64: there is an
// `#[cfg(target_arch = "x86_64")]` dispatch in `cfl_pred_direct` and nothing
// beside it. Measured at t=1 on `v4k_8tile` it is 3.37% of decode inclusive
// (2.46% self) and on `v4k_8tile_10b` 1.79% self — with `cfl_ac_rust`, which
// stays scalar, the pair is 5.87% / 3.55%.
//
// The kernel is four ops per pixel with no cross-lane dependence:
//
//     diff = alpha * ac[x];
//     dst[x] = iclip_pixel(dc + apply_sign(diff.abs() + 32 >> 6, diff));
//
// `alpha * ac` needs 32 bits (`|ac|` reaches ~1<<13 at 8bpc and `|alpha|` is up
// to 16), so the lanes are `int32x4_t` and the i16 input is widened on load.
// `apply_sign(m, t)` for `m >= 0` is `(m ^ (t >> 31)) - (t >> 31)`, which is
// three cheap ops and no select.

/// `iclip_pixel(dc + apply_sign((|alpha*ac| + 32) >> 6, alpha*ac))` for 4 lanes.
#[cfg(target_arch = "aarch64")]
#[archmage::rite(neon)]
fn cfl_lane4(ac: int32x4_t, alpha: int32x4_t, dcv: int32x4_t, pmax: int32x4_t) -> int32x4_t {
    let t = vmulq_s32(ac, alpha);
    let m = vshrq_n_s32::<6>(vaddq_s32(vabsq_s32(t), vdupq_n_s32(32)));
    let s = vshrq_n_s32::<31>(t);
    let signed = vsubq_s32(veorq_s32(m, s), s);
    vminq_s32(vmaxq_s32(vaddq_s32(dcv, signed), vdupq_n_s32(0)), pmax)
}

/// CfL prediction for one row of 8bpc pixels. `w` is a multiple of 4.
#[cfg(target_arch = "aarch64")]
#[archmage::rite(neon)]
fn cfl_row_8bpc(dst: &mut [u8], base: usize, ac: &[i16], w: usize, alpha: i32, dc: i32) {
    let alphav = vdupq_n_s32(alpha);
    let dcv = vdupq_n_s32(dc);
    let pmax = vdupq_n_s32(255);
    for x in (0..w).step_by(4) {
        let a = <&[i16; 4]>::try_from(&ac[x..x + 4]).expect("4 ac");
        let acv = vmovl_s16(safe_unaligned_simd::aarch64::vld1_s16(a));
        let r = cfl_lane4(acv, alphav, dcv, pmax);
        let narrow = vmovn_u32(vreinterpretq_u32_s32(r));
        let bytes = vmovn_u16(vcombine_u16(narrow, narrow));
        let mut tmp = [0u8; 8];
        safe_unaligned_simd::aarch64::vst1_u8(&mut tmp, bytes);
        dst[base + x..base + x + 4].copy_from_slice(&tmp[..4]);
    }
}

/// CfL prediction for one row of 16bpc pixels. `w` is a multiple of 4.
#[cfg(target_arch = "aarch64")]
#[archmage::rite(neon)]
fn cfl_row_16bpc(
    dst: &mut [u16],
    base: usize,
    ac: &[i16],
    w: usize,
    alpha: i32,
    dc: i32,
    bitdepth_max: i32,
) {
    let alphav = vdupq_n_s32(alpha);
    let dcv = vdupq_n_s32(dc);
    let pmax = vdupq_n_s32(bitdepth_max);
    for x in (0..w).step_by(4) {
        let a = <&[i16; 4]>::try_from(&ac[x..x + 4]).expect("4 ac");
        let acv = vmovl_s16(safe_unaligned_simd::aarch64::vld1_s16(a));
        let r = cfl_lane4(acv, alphav, dcv, pmax);
        let out = vmovn_u32(vreinterpretq_u32_s32(r));
        let slot = <&mut [u16; 4]>::try_from(&mut dst[base + x..base + x + 4]).expect("4 px");
        safe_unaligned_simd::aarch64::vst1_u16(slot, out);
    }
}

/// Safe dispatch for `cfl_pred` on aarch64. Returns true if NEON ran it.
///
/// Every AV1 CfL block width is a multiple of 4 (4, 8, 16, 32), so there is no
/// scalar tail; the `w % 4` guard is a belt-and-braces refusal rather than a
/// live path.
#[cfg(target_arch = "aarch64")]
pub fn cfl_pred_dispatch<BD: crate::include::common::bitdepth::BitDepth>(
    dst: &mut crate::src::owned_recon::ReconDst<'_>,
    width: c_int,
    height: c_int,
    dc: c_int,
    ac: &[i16],
    alpha: c_int,
    bd: BD,
) -> bool {
    use crate::include::common::bitdepth::{AsPrimitive, BPC};

    if crate::src::ablate::is_off(crate::src::ablate::Family::IntraPred) {
        return false;
    }
    let Some(_token) = Arm64::summon() else {
        return false;
    };
    let (w, h) = (width as usize, height as usize);
    if w % 4 != 0 || w == 0 || h == 0 {
        return false;
    }
    let ac = &ac[..w * h];
    let bitdepth_max = bd.bitdepth_max().as_::<c_int>();

    // ROW GRANULARITY IS DECIDED BY `for_rows_mut`, NOT HERE.
    //
    // History, because the previous decision here was measured and is now
    // superseded rather than merely re-argued. This site used to hardcode one
    // narrow guard per row, on the strength of an A/B against
    // `with_pixel_guard_mut::<BD>(&dst, w, h)`: the wide form removed
    // `ipred::cfl_pred`'s 2.46% of self time but added +0.79% `add_wide::<true>`
    // and +0.73% `remove_wide`, netting 0.24% of whole-decode wall (396.77 vs
    // 397.71 ms/frame, medians of 9 interleaved rounds, idle box).
    //
    // Both of the terms that made the wide form lose were WIDE-PATH terms, and
    // the wide path is gone at t=1: issue #458's probe measures
    // `wide_total = 0` at every thread count on current `main`. So that A/B was
    // measuring a mechanism that no longer exists, exactly like the two
    // branches #459 retired. `for_rows_mut` takes the hull as ONE ordinary
    // single-shard registration when tile threading is off, and keeps the
    // per-row guards when it is on — where the narrow extent is load-bearing
    // against a neighbouring tile column.
    dst.for_rows_mut::<BD, _>(w, h, |y, px| match BD::BPC {
        BPC::BPC8 => {
            let b: &mut [u8] = crate::src::safe_simd::pixel_access::reinterpret_slice_mut(px)
                .expect("BD::Pixel layout matches u8");
            cfl_row_8bpc_neon(_token, b, &ac[y * w..], w, alpha, dc);
        }
        BPC::BPC16 => {
            let b: &mut [u16] = crate::src::safe_simd::pixel_access::reinterpret_slice_mut(px)
                .expect("BD::Pixel layout matches u16");
            cfl_row_16bpc_neon(_token, b, &ac[y * w..], w, alpha, dc, bitdepth_max);
        }
    });
    true
}

/// `#[arcane]` boundary per row.
///
/// One boundary per block would be cheaper, but the block form needs a wide
/// guard over all `h` rows, and that guard measured more expensive than the
/// kernel saves (see `cfl_pred_dispatch`). The token summon itself is ~1 ns and
/// `h <= 32`, so per-row is the cheaper of the two shapes here.
#[cfg(target_arch = "aarch64")]
#[arcane]
fn cfl_row_8bpc_neon(_token: Arm64, dst: &mut [u8], ac: &[i16], w: usize, alpha: c_int, dc: c_int) {
    cfl_row_8bpc(dst, 0, ac, w, alpha, dc);
}

#[cfg(target_arch = "aarch64")]
#[arcane]
fn cfl_row_16bpc_neon(
    _token: Arm64,
    dst: &mut [u16],
    ac: &[i16],
    w: usize,
    alpha: c_int,
    dc: c_int,
    bitdepth_max: c_int,
) {
    cfl_row_16bpc(dst, 0, ac, w, alpha, dc, bitdepth_max);
}

// Gated on `not(c-ffi)`, not `not(asm)`: the harness round-trips its scratch
// through `Rav1dPictureDataComponent::copy_pixels_to`, which only exists in
// safe (non-`c-ffi`) mode — in `c-ffi` mode `wrap_buf` is zero-copy and there
// is nothing to copy back. `asm` implies `c-ffi`, so this still excludes every
// configuration the old gate did, plus `--features c-ffi` on its own.
#[cfg(all(test, target_arch = "aarch64", not(feature = "c-ffi")))]
mod cfl_parity {
    //! Differential parity for `cfl_pred_dispatch` against the scalar
    //! `src/ipred.rs::cfl_pred` it replaces.
    //!
    //! The scalar path is the conformance oracle, and unlike itx there is no
    //! `__simd_test` dual-compute hook on this call — so this sweep is the
    //! only per-parameter evidence, and the corpus MD5 is the only other one.
    //!
    //! The parameter space is the real one: `alpha` is signalled in
    //! `[-16, 16]`, `ac` holds bitdepth-scaled AC residual, and every AV1 CfL
    //! block is 4..32 on a side. All four are swept, at both bit depths, plus
    //! the saturating extremes that decide whether the clip is right.

    use crate::include::common::bitdepth::{BitDepth, BitDepth8, BitDepth16};
    use crate::include::dav1d::picture::Rav1dPictureDataComponent;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn in_range(&mut self, lo: i32, hi: i32) -> i32 {
            lo + (self.next() % ((hi - lo + 1) as u64)) as i32
        }
    }

    /// Returns (neon, scalar) pixel buffers, and whether NEON claimed the cell.
    fn run<BD: BitDepth>(
        w: usize,
        h: usize,
        dc: i32,
        alpha: i32,
        ac: &[i16],
        pixels: &[BD::Pixel],
        stride: usize,
        bd: BD,
    ) -> (Vec<BD::Pixel>, Vec<BD::Pixel>, bool)
    where
        BD::Pixel: Copy + Default,
    {
        let go = |simd: bool| -> (Vec<BD::Pixel>, bool) {
            let mut px = pixels.to_vec();
            let mut out = vec![BD::Pixel::default(); px.len()];
            let comp = Rav1dPictureDataComponent::wrap_buf::<BD>(&mut px, stride);
            let dst = comp.with_offset::<BD>();
            let handled = if simd {
                super::cfl_pred_dispatch::<BD>(
                    &mut crate::src::owned_recon::ReconDst::Pic(dst),
                    w as i32,
                    h as i32,
                    dc,
                    ac,
                    alpha,
                    bd,
                )
            } else {
                let mut fixed = [0i16; crate::src::internal::SCRATCH_AC_TXTP_LEN];
                fixed[..ac.len()].copy_from_slice(ac);
                crate::src::ipred::cfl_pred_scalar_for_test::<BD>(
                    &mut crate::src::owned_recon::ReconDst::Pic(dst),
                    w as i32,
                    h as i32,
                    dc,
                    &fixed,
                    alpha,
                    bd,
                );
                true
            };
            comp.copy_pixels_to::<BD>(&mut out);
            (out, handled)
        };
        let (neon, handled) = go(true);
        let (scalar, _) = go(false);
        (neon, scalar, handled)
    }

    fn sweep<BD: BitDepth>(bd: BD, seed: u64, what: &str)
    where
        BD::Pixel: Copy + Default + PartialEq + std::fmt::Debug,
    {
        use crate::include::common::bitdepth::AsPrimitive;
        let _lock = crate::src::safe_simd::token_test_lock();
        let pmax = bd.bitdepth_max().as_::<i32>();
        let mut cells = 0usize;
        let mut live = 0usize;
        let mut bad: Vec<String> = Vec::new();

        for &w in &[4usize, 8, 16, 32] {
            for &h in &[4usize, 8, 16, 32] {
                let stride = (w + 16).next_multiple_of(16);
                let mut rng = Rng(seed ^ ((w as u64) << 32) ^ (h as u64));
                for &alpha in &[-16i32, -13, -8, -3, -1, 0, 1, 3, 8, 13, 16] {
                    // `dc` and the AC magnitudes cover the ordinary case and
                    // both saturating ends; a missing or wrong clamp shows up
                    // only at the extremes.
                    for &(dc, scale) in &[
                        (pmax / 2, pmax / 4),
                        (0, pmax),
                        (pmax, pmax),
                        (pmax / 2, 1 << 13),
                        (1, 3),
                    ] {
                        let ac: Vec<i16> = (0..w * h)
                            .map(|_| rng.in_range(-scale, scale).clamp(-32768, 32767) as i16)
                            .collect();
                        let pixels: Vec<BD::Pixel> = (0..stride * h)
                            .map(|_| rng.in_range(0, pmax).as_())
                            .collect();
                        let (neon, scalar, ok) =
                            run::<BD>(w, h, dc, alpha, &ac, &pixels, stride, bd);
                        cells += 1;
                        if ok {
                            live += 1;
                        }
                        if let Some((x, y)) = (0..h)
                            .flat_map(|y| (0..w).map(move |x| (x, y)))
                            .find(|&(x, y)| neon[y * stride + x] != scalar[y * stride + x])
                        {
                            let msg = format!(
                                "{w}x{h} alpha={alpha} dc={dc} scale={scale} at ({x},{y}): \
                                 neon={:?} scalar={:?}",
                                neon[y * stride + x],
                                scalar[y * stride + x]
                            );
                            if bad.len() < 4 {
                                bad.push(msg);
                            }
                        }
                    }
                }
            }
        }
        assert!(cells >= 800, "{what}: only {cells} cells ran");
        assert_eq!(
            live,
            cells,
            "{what}: {} of {cells} cells did NOT take the NEON path — those \
             compared the scalar reference against itself and proved nothing",
            cells - live
        );
        assert!(bad.is_empty(), "{what}: divergence\n  {}", bad.join("\n  "));
    }

    #[test]
    fn cfl_pred_8bpc_matches_scalar() {
        sweep(BitDepth8::new(()), 0x51DE_0001_0000_0001, "cfl 8bpc");
    }

    #[test]
    fn cfl_pred_10bpc_matches_scalar() {
        sweep(BitDepth16::new(1023), 0x51DE_0002_0000_0001, "cfl 10bpc");
    }

    #[test]
    fn cfl_pred_12bpc_matches_scalar() {
        sweep(BitDepth16::new(4095), 0x51DE_0003_0000_0001, "cfl 12bpc");
    }
}

// ============================================================================
// Chroma-from-luma AC (`cfl_ac`) — NEON
// ============================================================================
//
// `src/ipred.rs::cfl_ac_rust` is the conformance oracle for what follows. It
// subsamples the reconstructed luma block into the `ac` scratch, replicates the
// right/bottom padding, then subtracts the block's own DC so the residual is
// zero-mean. Every step is exact integer arithmetic on values that provably fit
// their lane width, which is what makes a vector form bit-identical rather than
// merely close:
//
// * A subsampled sum is at most 4 pixels, and the left shift is `1 + !ss_ver +
//   !ss_hor`, so the product is at most `8 * (2^bd - 1)` = 2,040 / 8,184 /
//   32,760 at 8 / 10 / 12 bpc. Every case fits `i16` (and `u16`) with no
//   truncation, at every bit depth AV1 defines.
// * The DC accumulator is `i32` over at most 32x32 = 1,024 such values, so at
//   most 33.5 M — well inside `i32`. Integer addition is associative, so the
//   pairwise-widening `vpadalq_s16` order gives the identical total as the
//   reference's sequential one.
// * `ac - dc` lands in `[-32760, 32760]`, so `vsubq_s16`'s wrapping subtract
//   and the reference's `-=` agree.
//
// Measured cost of the scalar form before this existed, `v4k_8tile` at t=1
// (macOS `sample`, self time): 2.08% of whole-decode wall at 8bpc and 1.86% at
// 10bpc, of which only 0.26% was borrow-tracker work — i.e. essentially all
// arithmetic, which is why this is a kernel port and not a guard change.
// Record: `benchmarks/st1_kernels_2026-08-08.meta`.

/// Subsample one output row, 4:2:0 (`ss_hor && ss_ver`), 8bpc.
///
/// `out[x] = (top[2x] + top[2x+1] + bot[2x] + bot[2x+1]) << 1`.
#[cfg(target_arch = "aarch64")]
#[arcane]
fn ac_row_420_8bpc(_token: Arm64, out: &mut [i16], top: &[u8], bot: &[u8], n: usize) {
    let mut x = 0;
    while x + 8 <= n {
        let t = safe_simd::vld1q_u8(<&[u8; 16]>::try_from(&top[2 * x..2 * x + 16]).expect("16"));
        let b = safe_simd::vld1q_u8(<&[u8; 16]>::try_from(&bot[2 * x..2 * x + 16]).expect("16"));
        let s = vaddq_u16(vpaddlq_u8(t), vpaddlq_u8(b));
        safe_simd::vst1q_s16(
            <&mut [i16; 8]>::try_from(&mut out[x..x + 8]).expect("8"),
            vreinterpretq_s16_u16(vshlq_n_u16::<1>(s)),
        );
        x += 8;
    }
    while x + 4 <= n {
        let t = safe_simd::vld1_u8(<&[u8; 8]>::try_from(&top[2 * x..2 * x + 8]).expect("8"));
        let b = safe_simd::vld1_u8(<&[u8; 8]>::try_from(&bot[2 * x..2 * x + 8]).expect("8"));
        let s = vadd_u16(vpaddl_u8(t), vpaddl_u8(b));
        safe_simd::vst1_s16(
            <&mut [i16; 4]>::try_from(&mut out[x..x + 4]).expect("4"),
            vreinterpret_s16_u16(vshl_n_u16::<1>(s)),
        );
        x += 4;
    }
    for x in x..n {
        let s =
            top[2 * x] as u16 + top[2 * x + 1] as u16 + bot[2 * x] as u16 + bot[2 * x + 1] as u16;
        out[x] = (s << 1) as i16;
    }
}

/// Subsample one output row, 4:2:2 (`ss_hor && !ss_ver`), 8bpc.
#[cfg(target_arch = "aarch64")]
#[arcane]
fn ac_row_422_8bpc(_token: Arm64, out: &mut [i16], top: &[u8], n: usize) {
    let mut x = 0;
    while x + 8 <= n {
        let t = safe_simd::vld1q_u8(<&[u8; 16]>::try_from(&top[2 * x..2 * x + 16]).expect("16"));
        safe_simd::vst1q_s16(
            <&mut [i16; 8]>::try_from(&mut out[x..x + 8]).expect("8"),
            vreinterpretq_s16_u16(vshlq_n_u16::<2>(vpaddlq_u8(t))),
        );
        x += 8;
    }
    while x + 4 <= n {
        let t = safe_simd::vld1_u8(<&[u8; 8]>::try_from(&top[2 * x..2 * x + 8]).expect("8"));
        safe_simd::vst1_s16(
            <&mut [i16; 4]>::try_from(&mut out[x..x + 4]).expect("4"),
            vreinterpret_s16_u16(vshl_n_u16::<2>(vpaddl_u8(t))),
        );
        x += 4;
    }
    for x in x..n {
        let s = top[2 * x] as u16 + top[2 * x + 1] as u16;
        out[x] = (s << 2) as i16;
    }
}

/// Subsample one output row, 4:4:4 (no subsampling), 8bpc.
#[cfg(target_arch = "aarch64")]
#[arcane]
fn ac_row_444_8bpc(_token: Arm64, out: &mut [i16], top: &[u8], n: usize) {
    let mut x = 0;
    while x + 8 <= n {
        let t = safe_simd::vld1_u8(<&[u8; 8]>::try_from(&top[x..x + 8]).expect("8"));
        safe_simd::vst1q_s16(
            <&mut [i16; 8]>::try_from(&mut out[x..x + 8]).expect("8"),
            vreinterpretq_s16_u16(vshlq_n_u16::<3>(vmovl_u8(t))),
        );
        x += 8;
    }
    for x in x..n {
        out[x] = ((top[x] as u16) << 3) as i16;
    }
}

/// Subsample one output row, 4:2:0, 16bpc (10/12-bit).
#[cfg(target_arch = "aarch64")]
#[arcane]
fn ac_row_420_16bpc(_token: Arm64, out: &mut [i16], top: &[u16], bot: &[u16], n: usize) {
    let mut x = 0;
    while x + 8 <= n {
        let t0 = safe_simd::vld1q_u16(<&[u16; 8]>::try_from(&top[2 * x..2 * x + 8]).expect("8"));
        let t1 =
            safe_simd::vld1q_u16(<&[u16; 8]>::try_from(&top[2 * x + 8..2 * x + 16]).expect("8"));
        let b0 = safe_simd::vld1q_u16(<&[u16; 8]>::try_from(&bot[2 * x..2 * x + 8]).expect("8"));
        let b1 =
            safe_simd::vld1q_u16(<&[u16; 8]>::try_from(&bot[2 * x + 8..2 * x + 16]).expect("8"));
        let s = vaddq_u16(vpaddq_u16(t0, t1), vpaddq_u16(b0, b1));
        safe_simd::vst1q_s16(
            <&mut [i16; 8]>::try_from(&mut out[x..x + 8]).expect("8"),
            vreinterpretq_s16_u16(vshlq_n_u16::<1>(s)),
        );
        x += 8;
    }
    while x + 4 <= n {
        let t = safe_simd::vld1q_u16(<&[u16; 8]>::try_from(&top[2 * x..2 * x + 8]).expect("8"));
        let b = safe_simd::vld1q_u16(<&[u16; 8]>::try_from(&bot[2 * x..2 * x + 8]).expect("8"));
        let s = vadd_u16(
            vpadd_u16(vget_low_u16(t), vget_high_u16(t)),
            vpadd_u16(vget_low_u16(b), vget_high_u16(b)),
        );
        safe_simd::vst1_s16(
            <&mut [i16; 4]>::try_from(&mut out[x..x + 4]).expect("4"),
            vreinterpret_s16_u16(vshl_n_u16::<1>(s)),
        );
        x += 4;
    }
    for x in x..n {
        let s = top[2 * x] + top[2 * x + 1] + bot[2 * x] + bot[2 * x + 1];
        out[x] = (s << 1) as i16;
    }
}

/// Subsample one output row, 4:2:2, 16bpc.
#[cfg(target_arch = "aarch64")]
#[arcane]
fn ac_row_422_16bpc(_token: Arm64, out: &mut [i16], top: &[u16], n: usize) {
    let mut x = 0;
    while x + 8 <= n {
        let t0 = safe_simd::vld1q_u16(<&[u16; 8]>::try_from(&top[2 * x..2 * x + 8]).expect("8"));
        let t1 =
            safe_simd::vld1q_u16(<&[u16; 8]>::try_from(&top[2 * x + 8..2 * x + 16]).expect("8"));
        safe_simd::vst1q_s16(
            <&mut [i16; 8]>::try_from(&mut out[x..x + 8]).expect("8"),
            vreinterpretq_s16_u16(vshlq_n_u16::<2>(vpaddq_u16(t0, t1))),
        );
        x += 8;
    }
    while x + 4 <= n {
        let t = safe_simd::vld1q_u16(<&[u16; 8]>::try_from(&top[2 * x..2 * x + 8]).expect("8"));
        safe_simd::vst1_s16(
            <&mut [i16; 4]>::try_from(&mut out[x..x + 4]).expect("4"),
            vreinterpret_s16_u16(vshl_n_u16::<2>(vpadd_u16(
                vget_low_u16(t),
                vget_high_u16(t),
            ))),
        );
        x += 4;
    }
    for x in x..n {
        let s = top[2 * x] + top[2 * x + 1];
        out[x] = (s << 2) as i16;
    }
}

/// Subsample one output row, 4:4:4, 16bpc.
#[cfg(target_arch = "aarch64")]
#[arcane]
fn ac_row_444_16bpc(_token: Arm64, out: &mut [i16], top: &[u16], n: usize) {
    let mut x = 0;
    while x + 8 <= n {
        let t = safe_simd::vld1q_u16(<&[u16; 8]>::try_from(&top[x..x + 8]).expect("8"));
        safe_simd::vst1q_s16(
            <&mut [i16; 8]>::try_from(&mut out[x..x + 8]).expect("8"),
            vreinterpretq_s16_u16(vshlq_n_u16::<3>(t)),
        );
        x += 8;
    }
    for x in x..n {
        out[x] = (top[x] << 3) as i16;
    }
}

/// Sum `ac` into an `i32`, then subtract `(base + sum) >> log2sz` from every
/// element — the reference's DC-removal tail, vectorised.
///
/// The accumulation order differs from the reference's (pairwise-widening into
/// four lanes rather than left to right) and the result does not, because
/// `i32` addition is associative and the total provably cannot overflow.
#[cfg(target_arch = "aarch64")]
#[arcane]
fn ac_remove_dc(_token: Arm64, ac: &mut [i16], base: i32, log2sz: u32) {
    let n = ac.len();
    let mut acc = vdupq_n_s32(0);
    let mut i = 0;
    while i + 8 <= n {
        let v = safe_simd::vld1q_s16(<&[i16; 8]>::try_from(&ac[i..i + 8]).expect("8"));
        acc = vpadalq_s16(acc, v);
        i += 8;
    }
    let mut sum = base + vaddvq_s32(acc);
    for &v in &ac[i..] {
        sum += v as i32;
    }
    let dc = (sum >> log2sz) as i16;
    let dcv = vdupq_n_s16(dc);
    let mut i = 0;
    while i + 8 <= n {
        let v = safe_simd::vld1q_s16(<&[i16; 8]>::try_from(&ac[i..i + 8]).expect("8"));
        safe_simd::vst1q_s16(
            <&mut [i16; 8]>::try_from(&mut ac[i..i + 8]).expect("8"),
            vsubq_s16(v, dcv),
        );
        i += 8;
    }
    for v in &mut ac[i..] {
        *v -= dc;
    }
}

/// Safe dispatch entry point for `cfl_ac` on aarch64.
///
/// Guard shape is deliberately IDENTICAL to `src/ipred.rs::cfl_ac_rust`'s — one
/// immutable guard per source row — so this change is an arithmetic A/B and
/// nothing else. (Two attempts to move guard shape in this subsystem measured
/// negative in both directions; see the meta.)
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub fn cfl_ac_dispatch<BD: crate::include::common::bitdepth::BitDepth>(
    ac: &mut [i16],
    y_src: &crate::src::owned_recon::ReconSrc<'_>,
    w_pad: c_int,
    h_pad: c_int,
    width: usize,
    height: usize,
    is_ss_hor: bool,
    is_ss_ver: bool,
) -> bool {
    use crate::include::common::bitdepth::BPC;
    use crate::src::safe_simd::pixel_access::reinterpret_slice;
    use crate::src::strided::Strided as _;

    if crate::src::ablate::is_off(crate::src::ablate::Family::IntraPred) {
        return false;
    }
    let Some(token) = Arm64::summon() else {
        return false;
    };
    // AV1 has no 4:4:0, and the reference's shift derivation assumes it.
    if is_ss_ver && !is_ss_hor {
        return false;
    }

    let ac = &mut ac[..width * height];
    let w_pad = w_pad as usize * 4;
    let h_pad = h_pad as usize * 4;
    // The reference `assert!`s these. Decline instead of asserting: a refusal
    // here runs the reference, which asserts anyway, so the invariant is still
    // enforced in exactly one place — and a release build cannot reach the
    // wrapping subtraction below.
    if w_pad >= width || h_pad >= height {
        return false;
    }
    let active_w = width - w_pad;
    let active_h = height - h_pad;
    let ss_hor = is_ss_hor as u8;
    let ss_ver = is_ss_ver as u8;

    let y_pxstride = y_src.pixel_stride::<BD>();
    let src_cols = active_w << ss_hor;
    let row_stride = y_pxstride << ss_ver;

    for y in 0..active_h {
        let aci = y * width;
        let row_pic = y_src.at(y as isize * row_stride);
        let row_guard = row_pic.slice::<BD>(src_cols);
        let row_below_guard;
        let below: Option<&[BD::Pixel]> = if is_ss_ver {
            row_below_guard = row_pic.at(y_pxstride).slice::<BD>(src_cols);
            Some(&*row_below_guard)
        } else {
            None
        };
        let out = &mut ac[aci..aci + active_w];
        match BD::BPC {
            BPC::BPC8 => {
                let top: &[u8] = match reinterpret_slice(&row_guard) {
                    Some(t) => t,
                    None => return false,
                };
                match (is_ss_hor, below) {
                    (true, Some(b)) => {
                        let bot: &[u8] = match reinterpret_slice(b) {
                            Some(t) => t,
                            None => return false,
                        };
                        ac_row_420_8bpc(token, out, top, bot, active_w);
                    }
                    (true, None) => ac_row_422_8bpc(token, out, top, active_w),
                    (false, _) => ac_row_444_8bpc(token, out, top, active_w),
                }
            }
            BPC::BPC16 => {
                let top: &[u16] = match reinterpret_slice(&row_guard) {
                    Some(t) => t,
                    None => return false,
                };
                match (is_ss_hor, below) {
                    (true, Some(b)) => {
                        let bot: &[u16] = match reinterpret_slice(b) {
                            Some(t) => t,
                            None => return false,
                        };
                        ac_row_420_16bpc(token, out, top, bot, active_w);
                    }
                    (true, None) => ac_row_422_16bpc(token, out, top, active_w),
                    (false, _) => ac_row_444_16bpc(token, out, top, active_w),
                }
            }
        }
        drop(row_guard);
        // Right edge: replicate the last real column across the pad.
        for x in active_w..width {
            ac[aci + x] = ac[aci + x - 1];
        }
    }
    // Bottom edge: replicate the last real row across the pad.
    for y in height - h_pad..height {
        let aci = y * width;
        let (src, dst) = ac.split_at_mut(aci);
        dst[..width].copy_from_slice(&src[src.len() - width..]);
    }

    let log2sz = width.trailing_zeros() + height.trailing_zeros();
    ac_remove_dc(token, ac, 1 << log2sz >> 1, log2sz);
    true
}

#[cfg(all(test, target_arch = "aarch64", not(feature = "asm")))]
mod cfl_ac_parity {
    //! Differential parity for `cfl_ac_dispatch` against the scalar
    //! `src/ipred.rs::cfl_ac_rust` it replaces.
    //!
    //! `cfl_ac` has no `__simd_test` dual-compute hook, so this sweep plus the
    //! corpus MD5 are the only per-parameter evidence there is. The parameter
    //! space swept is the real one: all three chroma layouts, every chroma
    //! transform size AV1 defines (4..32 on a side), both padding axes over
    //! their full signalled range, both bit depths, and source pixels driven to
    //! 0 / `bitdepth_max` as well as random, because a shift that overflowed
    //! `i16` would only show at the top of the range.
    //!
    //! The live-cell assertion matters as much as the equality one: a cell the
    //! NEON dispatch declined would compare the reference against itself and
    //! prove nothing.

    use crate::include::common::bitdepth::{BitDepth, BitDepth8, BitDepth16};
    use crate::include::dav1d::picture::Rav1dPictureDataComponent;
    use crate::src::internal::SCRATCH_AC_TXTP_LEN;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn in_range(&mut self, lo: i32, hi: i32) -> i32 {
            lo + (self.next() % ((hi - lo + 1) as u64)) as i32
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run<BD: BitDepth>(
        pixels: &[BD::Pixel],
        stride: usize,
        w_pad: i32,
        h_pad: i32,
        cw: usize,
        ch: usize,
        ss_hor: bool,
        ss_ver: bool,
    ) -> (Vec<i16>, Vec<i16>, bool)
    where
        BD::Pixel: Copy + Default,
    {
        let go = |simd: bool| -> (Vec<i16>, bool) {
            let mut px = pixels.to_vec();
            let comp = Rav1dPictureDataComponent::wrap_buf::<BD>(&mut px, stride);
            let dst = comp.with_offset::<BD>();
            // Prefilled with a recognisable pattern so a kernel that fails to
            // write a lane is caught rather than matching a shared zero.
            let mut ac = [0x5A5Ai16; SCRATCH_AC_TXTP_LEN];
            let handled = if simd {
                super::cfl_ac_dispatch::<BD>(
                    &mut ac,
                    &crate::src::owned_recon::ReconSrc::Pic(dst),
                    w_pad,
                    h_pad,
                    cw,
                    ch,
                    ss_hor,
                    ss_ver,
                )
            } else {
                crate::src::ipred::cfl_ac_scalar_for_test::<BD>(
                    &mut ac,
                    &crate::src::owned_recon::ReconSrc::Pic(dst),
                    w_pad,
                    h_pad,
                    cw,
                    ch,
                    ss_hor,
                    ss_ver,
                );
                true
            };
            (ac[..cw * ch].to_vec(), handled)
        };
        let (neon, handled) = go(true);
        let (scalar, _) = go(false);
        (neon, scalar, handled)
    }

    fn sweep<BD: BitDepth>(bd: BD, seed: u64, what: &str)
    where
        BD::Pixel: Copy + Default + PartialEq + std::fmt::Debug,
    {
        use crate::include::common::bitdepth::AsPrimitive;
        let _lock = crate::src::safe_simd::token_test_lock();
        let pmax = bd.bitdepth_max().as_::<i32>();
        let mut cells = 0usize;
        let mut live = 0usize;
        let mut bad: Vec<String> = Vec::new();

        for &(ss_hor, ss_ver) in &[(true, true), (true, false), (false, false)] {
            for &cw in &[4usize, 8, 16, 32] {
                for &ch in &[4usize, 8, 16, 32] {
                    // Source extent the reference reads: `ch << ss_ver` rows of
                    // `cw << ss_hor` pixels, plus one more row for the 4:2:0
                    // bottom half of the last pair.
                    let src_w = cw << ss_hor as usize;
                    let src_h = ch << ss_ver as usize;
                    // 64 so the byte length is a multiple of
                    // `RAV1D_PICTURE_GUARANTEED_MULTIPLE` at 8bpc too, where a
                    // pixel is one byte and 16 is not enough for `wrap_buf`.
                    let stride = (src_w + 16).next_multiple_of(64);
                    let mut rng =
                        Rng(seed ^ ((cw as u64) << 32) ^ ((ch as u64) << 8) ^ ss_hor as u64);
                    for &fill in &[-1i32, 0, 1] {
                        let pixels: Vec<BD::Pixel> = (0..stride * (src_h + 2))
                            .map(|_| {
                                match fill {
                                    -1 => rng.in_range(0, pmax),
                                    0 => 0,
                                    _ => pmax,
                                }
                                .as_()
                            })
                            .collect();
                        // `w_pad`/`h_pad` are signalled in units of 4 samples
                        // and must leave at least one real column/row.
                        for &wp in &[0i32, 1, 2] {
                            for &hp in &[0i32, 1, 2] {
                                if (wp as usize) * 4 >= cw || (hp as usize) * 4 >= ch {
                                    continue;
                                }
                                let (neon, scalar, ok) =
                                    run::<BD>(&pixels, stride, wp, hp, cw, ch, ss_hor, ss_ver);
                                cells += 1;
                                if ok {
                                    live += 1;
                                }
                                if let Some(i) = (0..cw * ch).find(|&i| neon[i] != scalar[i]) {
                                    let msg = format!(
                                        "{cw}x{ch} ss=({ss_hor},{ss_ver}) pad=({wp},{hp}) \
                                         fill={fill} at {i} ({},{}) : neon={} scalar={}",
                                        i % cw,
                                        i / cw,
                                        neon[i],
                                        scalar[i]
                                    );
                                    if bad.len() < 4 {
                                        bad.push(msg);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(cells >= 300, "{what}: only {cells} cells ran");
        assert_eq!(
            live,
            cells,
            "{what}: {} of {cells} cells did NOT take the NEON path — those \
             compared the scalar reference against itself and proved nothing",
            cells - live
        );
        assert!(bad.is_empty(), "{what}: divergence\n  {}", bad.join("\n  "));
    }

    #[test]
    fn cfl_ac_8bpc_matches_scalar() {
        sweep(BitDepth8::new(()), 0xAC00_0001_0000_0001, "cfl_ac 8bpc");
    }

    #[test]
    fn cfl_ac_10bpc_matches_scalar() {
        sweep(BitDepth16::new(1023), 0xAC00_0002_0000_0001, "cfl_ac 10bpc");
    }

    #[test]
    fn cfl_ac_12bpc_matches_scalar() {
        sweep(BitDepth16::new(4095), 0xAC00_0003_0000_0001, "cfl_ac 12bpc");
    }
}
