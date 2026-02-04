//! Safe SIMD implementations of intra prediction functions
//!
//! Replaces hand-written assembly with safe Rust intrinsics.
//!
//! Implemented so far:
//! - DC_128 prediction (constant fill with mid-value)
//! - Vertical prediction (copy top row)
//! - Horizontal prediction (fill from left pixels)

#![allow(unused)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use libc::{ptrdiff_t, c_int};

use crate::include::common::bitdepth::DynPixel;
use crate::include::dav1d::picture::Rav1dPictureDataComponentOffset;
use crate::src::ffi_safe::FFISafe;

// ============================================================================
// DC_128 Prediction (fill with mid-value)
// ============================================================================

/// DC_128 prediction: fill block with 128 (or 1 << (bitdepth - 1))
///
/// For 8bpc, fills with 128. This is the simplest prediction mode.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn ipred_dc_128_8bpc_avx2(
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
        let fill_val = _mm256_set1_epi8(128u8 as i8);

        for y in 0..height {
            let dst_row = dst.offset(y as isize * stride);

            // Fill row with 128
            let mut x = 0;
            while x + 32 <= width {
                _mm256_storeu_si256(dst_row.add(x) as *mut __m256i, fill_val);
                x += 32;
            }
            while x + 16 <= width {
                _mm_storeu_si128(dst_row.add(x) as *mut __m128i, _mm256_castsi256_si128(fill_val));
                x += 16;
            }
            while x < width {
                *dst_row.add(x) = 128;
                x += 1;
            }
        }
    }
}

// ============================================================================
// Vertical Prediction (copy top row)
// ============================================================================

/// Vertical prediction: copy the top row to all rows in the block
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn ipred_v_8bpc_avx2(
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

    unsafe {
        // Top pixels start at topleft + 1
        let top = (topleft as *const u8).add(1);

        // Load top row into register(s)
        match width {
            4 => {
                let top_val = _mm_cvtsi32_si128(*(top as *const i32));
                for y in 0..height {
                    let dst_row = dst.offset(y as isize * stride);
                    *(dst_row as *mut i32) = _mm_cvtsi128_si32(top_val);
                }
            }
            8 => {
                let top_val = _mm_loadl_epi64(top as *const __m128i);
                for y in 0..height {
                    let dst_row = dst.offset(y as isize * stride);
                    _mm_storel_epi64(dst_row as *mut __m128i, top_val);
                }
            }
            16 => {
                let top_val = _mm_loadu_si128(top as *const __m128i);
                for y in 0..height {
                    let dst_row = dst.offset(y as isize * stride);
                    _mm_storeu_si128(dst_row as *mut __m128i, top_val);
                }
            }
            32 => {
                let top_val = _mm256_loadu_si256(top as *const __m256i);
                for y in 0..height {
                    let dst_row = dst.offset(y as isize * stride);
                    _mm256_storeu_si256(dst_row as *mut __m256i, top_val);
                }
            }
            64 => {
                let top_val0 = _mm256_loadu_si256(top as *const __m256i);
                let top_val1 = _mm256_loadu_si256(top.add(32) as *const __m256i);
                for y in 0..height {
                    let dst_row = dst.offset(y as isize * stride);
                    _mm256_storeu_si256(dst_row as *mut __m256i, top_val0);
                    _mm256_storeu_si256(dst_row.add(32) as *mut __m256i, top_val1);
                }
            }
            _ => {
                // General case
                for y in 0..height {
                    let dst_row = dst.offset(y as isize * stride);
                    std::ptr::copy_nonoverlapping(top, dst_row, width);
                }
            }
        }
    }
}

// ============================================================================
// Horizontal Prediction (fill from left pixels)
// ============================================================================

/// Horizontal prediction: fill each row with the left pixel value
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn ipred_h_8bpc_avx2(
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
    let left_base = topleft as *const u8;

    unsafe {
        for y in 0..height {
            let dst_row = dst.offset(y as isize * stride);
            // Left pixels are at topleft - y - 1
            let left_pixel = *left_base.offset(-(y as isize) - 1);

            // Broadcast pixel value
            let fill_val = _mm256_set1_epi8(left_pixel as i8);

            let mut x = 0;
            while x + 32 <= width {
                _mm256_storeu_si256(dst_row.add(x) as *mut __m256i, fill_val);
                x += 32;
            }
            while x + 16 <= width {
                _mm_storeu_si128(dst_row.add(x) as *mut __m128i, _mm256_castsi256_si128(fill_val));
                x += 16;
            }
            while x < width {
                *dst_row.add(x) = left_pixel;
                x += 1;
            }
        }
    }
}

// ============================================================================
// DC Prediction (average of top and left)
// ============================================================================

/// DC prediction: fill block with average of top and left edge pixels
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn ipred_dc_8bpc_avx2(
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
    let tl = topleft as *const u8;

    unsafe {
        // Sum top pixels
        let mut sum: u32 = 0;
        for x in 0..width {
            sum += *tl.add(1 + x) as u32;
        }
        // Sum left pixels
        for y in 0..height {
            sum += *tl.offset(-(y as isize) - 1) as u32;
        }

        // Calculate average (rounded)
        let total = width + height;
        let dc_val = ((sum + (total as u32 >> 1)) / total as u32) as u8;

        // Fill block
        let fill_val = _mm256_set1_epi8(dc_val as i8);

        for y in 0..height {
            let dst_row = dst.offset(y as isize * stride);

            let mut x = 0;
            while x + 32 <= width {
                _mm256_storeu_si256(dst_row.add(x) as *mut __m256i, fill_val);
                x += 32;
            }
            while x + 16 <= width {
                _mm_storeu_si128(dst_row.add(x) as *mut __m128i, _mm256_castsi256_si128(fill_val));
                x += 16;
            }
            while x < width {
                *dst_row.add(x) = dc_val;
                x += 1;
            }
        }
    }
}

/// DC_TOP prediction: fill block with average of top edge only
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn ipred_dc_top_8bpc_avx2(
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
    let tl = topleft as *const u8;

    unsafe {
        // Sum top pixels
        let mut sum: u32 = 0;
        for x in 0..width {
            sum += *tl.add(1 + x) as u32;
        }

        // Calculate average (rounded)
        let dc_val = ((sum + (width as u32 >> 1)) / width as u32) as u8;

        // Fill block
        let fill_val = _mm256_set1_epi8(dc_val as i8);

        for y in 0..height {
            let dst_row = dst.offset(y as isize * stride);

            let mut x = 0;
            while x + 32 <= width {
                _mm256_storeu_si256(dst_row.add(x) as *mut __m256i, fill_val);
                x += 32;
            }
            while x + 16 <= width {
                _mm_storeu_si128(dst_row.add(x) as *mut __m128i, _mm256_castsi256_si128(fill_val));
                x += 16;
            }
            while x < width {
                *dst_row.add(x) = dc_val;
                x += 1;
            }
        }
    }
}

/// DC_LEFT prediction: fill block with average of left edge only
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn ipred_dc_left_8bpc_avx2(
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
    let tl = topleft as *const u8;

    unsafe {
        // Sum left pixels
        let mut sum: u32 = 0;
        for y in 0..height {
            sum += *tl.offset(-(y as isize) - 1) as u32;
        }

        // Calculate average (rounded)
        let dc_val = ((sum + (height as u32 >> 1)) / height as u32) as u8;

        // Fill block
        let fill_val = _mm256_set1_epi8(dc_val as i8);

        for y in 0..height {
            let dst_row = dst.offset(y as isize * stride);

            let mut x = 0;
            while x + 32 <= width {
                _mm256_storeu_si256(dst_row.add(x) as *mut __m256i, fill_val);
                x += 32;
            }
            while x + 16 <= width {
                _mm_storeu_si128(dst_row.add(x) as *mut __m128i, _mm256_castsi256_si128(fill_val));
                x += 16;
            }
            while x < width {
                *dst_row.add(x) = dc_val;
                x += 1;
            }
        }
    }
}
