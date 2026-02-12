//! Safe SIMD implementations of intra prediction functions
#![allow(deprecated)] // FFI wrappers need to forge tokens
#![cfg_attr(not(feature = "unchecked"), forbid(unsafe_code))]
#![cfg_attr(feature = "unchecked", deny(unsafe_code))]
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

use archmage::{arcane, Desktop64, Server64, SimdToken};
use libc::{c_int, ptrdiff_t};

#[cfg(target_arch = "x86_64")]
use super::partial_simd;
#[cfg(target_arch = "x86_64")]
use crate::src::safe_simd::pixel_access::{
    loadu_128, loadu_256, loadu_512, storeu_128, storeu_256, storeu_512, Flex,
};

use crate::include::common::bitdepth::DynPixel;
use crate::include::dav1d::picture::PicOffset;
use crate::src::ffi_safe::FFISafe;

// ============================================================================
// DC_128 Prediction (fill with mid-value)
// ============================================================================

/// DC_128 prediction: fill block with 128 (or 1 << (bitdepth - 1))
///
/// For 8bpc, fills with 128. This is the simplest prediction mode.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_dc_128_8bpc_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    width: usize,
    height: usize,
) {
    let mut dst = dst.flex_mut();
    let fill_val = _mm256_set1_epi8(128u8 as i8);

    for y in 0..height {
        let row_off = (dst_base as isize + y as isize * stride) as usize;
        let row = &mut dst[row_off..][..width];

        // Fill row with 128
        let mut x = 0;
        while x + 32 <= width {
            storeu_256!((&mut row[x..x + 32]), [u8; 32], fill_val);
            x += 32;
        }
        while x + 16 <= width {
            storeu_128!(
                &mut row[x..x + 16],
                [u8; 16],
                _mm256_castsi256_si128(fill_val)
            );
            x += 16;
        }
        while x < width {
            row[x] = 128;
            x += 1;
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
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
    _dst: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    let buf_len = compute_ipred_buf_len(stride as isize, width as usize, height as usize);
    let dst_sl = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_len) };
    ipred_dc_128_8bpc_inner(
        token,
        dst_sl,
        0,
        stride as isize,
        width as usize,
        height as usize,
    );
}

/// DC_128 prediction using AVX-512 (64-byte stores)
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_dc_128_8bpc_avx512_inner(
    _token: Server64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    width: usize,
    height: usize,
) {
    let mut dst = dst.flex_mut();
    let fill_val = _mm512_set1_epi8(128u8 as i8);
    let fill_256 = _mm256_set1_epi8(128u8 as i8);

    for y in 0..height {
        let row_off = (dst_base as isize + y as isize * stride) as usize;
        let row = &mut dst[row_off..][..width];

        let mut x = 0;
        while x + 64 <= width {
            storeu_512!((&mut row[x..x + 64]), [u8; 64], fill_val);
            x += 64;
        }
        while x + 32 <= width {
            storeu_256!((&mut row[x..x + 32]), [u8; 32], fill_256);
            x += 32;
        }
        while x + 16 <= width {
            storeu_128!(
                &mut row[x..x + 16],
                [u8; 16],
                _mm256_castsi256_si128(fill_256)
            );
            x += 16;
        }
        while x < width {
            row[x] = 128;
            x += 1;
        }
    }
}

/// Vertical prediction using AVX-512 (64-byte loads/stores)
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_v_8bpc_avx512_inner(
    _token: Server64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    topleft: &[u8],
    tl_off: usize,
    width: usize,
    height: usize,
) {
    let mut dst = dst.flex_mut();
    let topleft = topleft.flex();
    let top_off = tl_off + 1;

    match width {
        4 => {
            let top_val = _mm_cvtsi32_si128(i32::from_ne_bytes(
                topleft[top_off..top_off + 4].try_into().unwrap(),
            ));
            for y in 0..height {
                let row_off = (dst_base as isize + y as isize * stride) as usize;
                dst[row_off..row_off + 4]
                    .copy_from_slice(&_mm_cvtsi128_si32(top_val).to_ne_bytes());
            }
        }
        8 => {
            let top_val = partial_simd::mm_loadl_epi64::<[u8; 8]>(
                (&topleft[top_off..top_off + 8]).try_into().unwrap(),
            );
            for y in 0..height {
                let row_off = (dst_base as isize + y as isize * stride) as usize;
                partial_simd::mm_storel_epi64::<[u8; 8]>(
                    (&mut dst[row_off..row_off + 8]).try_into().unwrap(),
                    top_val,
                );
            }
        }
        16 => {
            let top_val = loadu_128!((&topleft[top_off..top_off + 16]), [u8; 16]);
            for y in 0..height {
                let row_off = (dst_base as isize + y as isize * stride) as usize;
                storeu_128!((&mut dst[row_off..row_off + 16]), [u8; 16], top_val);
            }
        }
        32 => {
            let top_val = loadu_256!((&topleft[top_off..top_off + 32]), [u8; 32]);
            for y in 0..height {
                let row_off = (dst_base as isize + y as isize * stride) as usize;
                storeu_256!((&mut dst[row_off..row_off + 32]), [u8; 32], top_val);
            }
        }
        64 => {
            // Single 512-bit load instead of 2x 256-bit
            let top_val = loadu_512!((&topleft[top_off..top_off + 64]), [u8; 64]);
            for y in 0..height {
                let row_off = (dst_base as isize + y as isize * stride) as usize;
                storeu_512!((&mut dst[row_off..row_off + 64]), [u8; 64], top_val);
            }
        }
        _ => {
            for y in 0..height {
                let row_off = (dst_base as isize + y as isize * stride) as usize;
                dst[row_off..row_off + width].copy_from_slice(&topleft[top_off..top_off + width]);
            }
        }
    }
}

// ============================================================================
// Vertical Prediction (copy top row)
// ============================================================================

/// Vertical prediction: copy the top row to all rows in the block
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_v_8bpc_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    topleft: &[u8],
    tl_off: usize,
    width: usize,
    height: usize,
) {
    let mut dst = dst.flex_mut();
    let topleft = topleft.flex();
    // Top pixels start at topleft + 1
    let top_off = tl_off + 1;

    // Load top row into register(s)
    match width {
        4 => {
            let top_val = _mm_cvtsi32_si128(i32::from_ne_bytes(
                topleft[top_off..top_off + 4].try_into().unwrap(),
            ));
            for y in 0..height {
                let row_off = (dst_base as isize + y as isize * stride) as usize;
                dst[row_off..row_off + 4]
                    .copy_from_slice(&_mm_cvtsi128_si32(top_val).to_ne_bytes());
            }
        }
        8 => {
            let top_val = partial_simd::mm_loadl_epi64::<[u8; 8]>(
                (&topleft[top_off..top_off + 8]).try_into().unwrap(),
            );
            for y in 0..height {
                let row_off = (dst_base as isize + y as isize * stride) as usize;
                partial_simd::mm_storel_epi64::<[u8; 8]>(
                    (&mut dst[row_off..row_off + 8]).try_into().unwrap(),
                    top_val,
                );
            }
        }
        16 => {
            let top_val = loadu_128!((&topleft[top_off..top_off + 16]), [u8; 16]);
            for y in 0..height {
                let row_off = (dst_base as isize + y as isize * stride) as usize;
                storeu_128!((&mut dst[row_off..row_off + 16]), [u8; 16], top_val);
            }
        }
        32 => {
            let top_val = loadu_256!((&topleft[top_off..top_off + 32]), [u8; 32]);
            for y in 0..height {
                let row_off = (dst_base as isize + y as isize * stride) as usize;
                storeu_256!((&mut dst[row_off..row_off + 32]), [u8; 32], top_val);
            }
        }
        64 => {
            let top_val0 = loadu_256!((&topleft[top_off..top_off + 32]), [u8; 32]);
            let top_val1 = loadu_256!((&topleft[top_off + 32..top_off + 64]), [u8; 32]);
            for y in 0..height {
                let row_off = (dst_base as isize + y as isize * stride) as usize;
                storeu_256!((&mut dst[row_off..row_off + 32]), [u8; 32], top_val0);
                storeu_256!((&mut dst[row_off + 32..row_off + 64]), [u8; 32], top_val1);
            }
        }
        _ => {
            // General case
            for y in 0..height {
                let row_off = (dst_base as isize + y as isize * stride) as usize;
                dst[row_off..row_off + width].copy_from_slice(&topleft[top_off..top_off + width]);
            }
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
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
    _dst: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    let buf_len = compute_ipred_buf_len(stride as isize, width as usize, height as usize);
    let dst_sl = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_len) };
    let (tl_sl, tl_off) =
        compute_topleft_slice(topleft as *const u8, width as usize, height as usize);
    ipred_v_8bpc_inner(
        token,
        dst_sl,
        0,
        stride as isize,
        tl_sl,
        tl_off,
        width as usize,
        height as usize,
    );
}

// ============================================================================
// Horizontal Prediction (fill from left pixels)
// ============================================================================

/// Horizontal prediction: fill each row with the left pixel value
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_h_8bpc_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    topleft: &[u8],
    tl_off: usize,
    width: usize,
    height: usize,
) {
    let mut dst = dst.flex_mut();
    let topleft = topleft.flex();
    for y in 0..height {
        let row_off = (dst_base as isize + y as isize * stride) as usize;
        let row = &mut dst[row_off..][..width];
        // Left pixels are at topleft - y - 1
        let left_pixel = topleft[tl_off - y - 1];

        // Broadcast pixel value
        let fill_val = _mm256_set1_epi8(left_pixel as i8);

        let mut x = 0;
        while x + 32 <= width {
            storeu_256!((&mut row[x..x + 32]), [u8; 32], fill_val);
            x += 32;
        }
        while x + 16 <= width {
            storeu_128!(
                &mut row[x..x + 16],
                [u8; 16],
                _mm256_castsi256_si128(fill_val)
            );
            x += 16;
        }
        while x < width {
            row[x] = left_pixel;
            x += 1;
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
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
    _dst: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    let buf_len = compute_ipred_buf_len(stride as isize, width as usize, height as usize);
    let dst_sl = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_len) };
    let (tl_sl, tl_off) =
        compute_topleft_slice(topleft as *const u8, width as usize, height as usize);
    ipred_h_8bpc_inner(
        token,
        dst_sl,
        0,
        stride as isize,
        tl_sl,
        tl_off,
        width as usize,
        height as usize,
    );
}

/// Horizontal prediction using AVX-512 (64-byte fills)
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_h_8bpc_avx512_inner(
    _token: Server64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    topleft: &[u8],
    tl_off: usize,
    width: usize,
    height: usize,
) {
    let mut dst = dst.flex_mut();
    let topleft = topleft.flex();
    for y in 0..height {
        let row_off = (dst_base as isize + y as isize * stride) as usize;
        let row = &mut dst[row_off..][..width];
        let left_pixel = topleft[tl_off - y - 1];
        let fill_512 = _mm512_set1_epi8(left_pixel as i8);
        let fill_256 = _mm256_set1_epi8(left_pixel as i8);

        let mut x = 0;
        while x + 64 <= width {
            storeu_512!((&mut row[x..x + 64]), [u8; 64], fill_512);
            x += 64;
        }
        while x + 32 <= width {
            storeu_256!((&mut row[x..x + 32]), [u8; 32], fill_256);
            x += 32;
        }
        while x + 16 <= width {
            storeu_128!(
                &mut row[x..x + 16],
                [u8; 16],
                _mm256_castsi256_si128(fill_256)
            );
            x += 16;
        }
        while x < width {
            row[x] = left_pixel;
            x += 1;
        }
    }
}

// ============================================================================
// DC Prediction AVX-512 variants (8bpc)
// ============================================================================

/// DC prediction using AVX-512 (64-byte stores)
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_dc_8bpc_avx512_inner(
    _token: Server64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    topleft: &[u8],
    tl_off: usize,
    width: usize,
    height: usize,
) {
    let mut dst = dst.flex_mut();
    let topleft = topleft.flex();
    let mut sum: u32 = 0;
    for x in 0..width {
        sum += topleft[tl_off + 1 + x] as u32;
    }
    for y in 0..height {
        sum += topleft[tl_off - y - 1] as u32;
    }
    let total = width + height;
    let dc_val = ((sum + (total as u32 >> 1)) / total as u32) as u8;

    let fill_512 = _mm512_set1_epi8(dc_val as i8);
    let fill_256 = _mm256_set1_epi8(dc_val as i8);

    for y in 0..height {
        let row_off = (dst_base as isize + y as isize * stride) as usize;
        let row = &mut dst[row_off..][..width];
        let mut x = 0;
        while x + 64 <= width {
            storeu_512!((&mut row[x..x + 64]), [u8; 64], fill_512);
            x += 64;
        }
        while x + 32 <= width {
            storeu_256!((&mut row[x..x + 32]), [u8; 32], fill_256);
            x += 32;
        }
        while x + 16 <= width {
            storeu_128!(&mut row[x..x + 16], [u8; 16], _mm256_castsi256_si128(fill_256));
            x += 16;
        }
        while x < width {
            row[x] = dc_val;
            x += 1;
        }
    }
}

/// DC_TOP prediction using AVX-512 (64-byte stores)
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_dc_top_8bpc_avx512_inner(
    _token: Server64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    topleft: &[u8],
    tl_off: usize,
    width: usize,
    height: usize,
) {
    let mut dst = dst.flex_mut();
    let topleft = topleft.flex();
    let mut sum: u32 = 0;
    for x in 0..width {
        sum += topleft[tl_off + 1 + x] as u32;
    }
    let dc_val = ((sum + (width as u32 >> 1)) / width as u32) as u8;

    let fill_512 = _mm512_set1_epi8(dc_val as i8);
    let fill_256 = _mm256_set1_epi8(dc_val as i8);

    for y in 0..height {
        let row_off = (dst_base as isize + y as isize * stride) as usize;
        let row = &mut dst[row_off..][..width];
        let mut x = 0;
        while x + 64 <= width {
            storeu_512!((&mut row[x..x + 64]), [u8; 64], fill_512);
            x += 64;
        }
        while x + 32 <= width {
            storeu_256!((&mut row[x..x + 32]), [u8; 32], fill_256);
            x += 32;
        }
        while x + 16 <= width {
            storeu_128!(&mut row[x..x + 16], [u8; 16], _mm256_castsi256_si128(fill_256));
            x += 16;
        }
        while x < width {
            row[x] = dc_val;
            x += 1;
        }
    }
}

/// DC_LEFT prediction using AVX-512 (64-byte stores)
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_dc_left_8bpc_avx512_inner(
    _token: Server64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    topleft: &[u8],
    tl_off: usize,
    width: usize,
    height: usize,
) {
    let mut dst = dst.flex_mut();
    let topleft = topleft.flex();
    let mut sum: u32 = 0;
    for y in 0..height {
        sum += topleft[tl_off - y - 1] as u32;
    }
    let dc_val = ((sum + (height as u32 >> 1)) / height as u32) as u8;

    let fill_512 = _mm512_set1_epi8(dc_val as i8);
    let fill_256 = _mm256_set1_epi8(dc_val as i8);

    for y in 0..height {
        let row_off = (dst_base as isize + y as isize * stride) as usize;
        let row = &mut dst[row_off..][..width];
        let mut x = 0;
        while x + 64 <= width {
            storeu_512!((&mut row[x..x + 64]), [u8; 64], fill_512);
            x += 64;
        }
        while x + 32 <= width {
            storeu_256!((&mut row[x..x + 32]), [u8; 32], fill_256);
            x += 32;
        }
        while x + 16 <= width {
            storeu_128!(&mut row[x..x + 16], [u8; 16], _mm256_castsi256_si128(fill_256));
            x += 16;
        }
        while x < width {
            row[x] = dc_val;
            x += 1;
        }
    }
}

// ============================================================================
// DC Prediction (average of top and left)
// ============================================================================

/// DC prediction: fill block with average of top and left edge pixels
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_dc_8bpc_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    topleft: &[u8],
    tl_off: usize,
    width: usize,
    height: usize,
) {
    let mut dst = dst.flex_mut();
    let topleft = topleft.flex();
    // Sum top pixels
    let mut sum: u32 = 0;
    for x in 0..width {
        sum += topleft[tl_off + 1 + x] as u32;
    }
    // Sum left pixels
    for y in 0..height {
        sum += topleft[tl_off - y - 1] as u32;
    }

    // Calculate average (rounded)
    let total = width + height;
    let dc_val = ((sum + (total as u32 >> 1)) / total as u32) as u8;

    // Fill block
    let fill_val = _mm256_set1_epi8(dc_val as i8);

    for y in 0..height {
        let row_off = (dst_base as isize + y as isize * stride) as usize;
        let row = &mut dst[row_off..][..width];

        let mut x = 0;
        while x + 32 <= width {
            storeu_256!((&mut row[x..x + 32]), [u8; 32], fill_val);
            x += 32;
        }
        while x + 16 <= width {
            storeu_128!(
                &mut row[x..x + 16],
                [u8; 16],
                _mm256_castsi256_si128(fill_val)
            );
            x += 16;
        }
        while x < width {
            row[x] = dc_val;
            x += 1;
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
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
    _dst: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    let buf_len = compute_ipred_buf_len(stride as isize, width as usize, height as usize);
    let dst_sl = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_len) };
    let (tl_sl, tl_off) =
        compute_topleft_slice(topleft as *const u8, width as usize, height as usize);
    ipred_dc_8bpc_inner(
        token,
        dst_sl,
        0,
        stride as isize,
        tl_sl,
        tl_off,
        width as usize,
        height as usize,
    );
}

/// DC_TOP prediction: fill block with average of top edge only
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_dc_top_8bpc_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    topleft: &[u8],
    tl_off: usize,
    width: usize,
    height: usize,
) {
    let mut dst = dst.flex_mut();
    let topleft = topleft.flex();
    // Sum top pixels
    let mut sum: u32 = 0;
    for x in 0..width {
        sum += topleft[tl_off + 1 + x] as u32;
    }

    // Calculate average (rounded)
    let dc_val = ((sum + (width as u32 >> 1)) / width as u32) as u8;

    // Fill block
    let fill_val = _mm256_set1_epi8(dc_val as i8);

    for y in 0..height {
        let row_off = (dst_base as isize + y as isize * stride) as usize;
        let row = &mut dst[row_off..][..width];

        let mut x = 0;
        while x + 32 <= width {
            storeu_256!((&mut row[x..x + 32]), [u8; 32], fill_val);
            x += 32;
        }
        while x + 16 <= width {
            storeu_128!(
                &mut row[x..x + 16],
                [u8; 16],
                _mm256_castsi256_si128(fill_val)
            );
            x += 16;
        }
        while x < width {
            row[x] = dc_val;
            x += 1;
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
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
    _dst: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    let buf_len = compute_ipred_buf_len(stride as isize, width as usize, height as usize);
    let dst_sl = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_len) };
    let (tl_sl, tl_off) =
        compute_topleft_slice(topleft as *const u8, width as usize, height as usize);
    ipred_dc_top_8bpc_inner(
        token,
        dst_sl,
        0,
        stride as isize,
        tl_sl,
        tl_off,
        width as usize,
        height as usize,
    );
}

/// DC_LEFT prediction: fill block with average of left edge only
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_dc_left_8bpc_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    topleft: &[u8],
    tl_off: usize,
    width: usize,
    height: usize,
) {
    let mut dst = dst.flex_mut();
    let topleft = topleft.flex();
    // Sum left pixels
    let mut sum: u32 = 0;
    for y in 0..height {
        sum += topleft[tl_off - y - 1] as u32;
    }

    // Calculate average (rounded)
    let dc_val = ((sum + (height as u32 >> 1)) / height as u32) as u8;

    // Fill block
    let fill_val = _mm256_set1_epi8(dc_val as i8);

    for y in 0..height {
        let row_off = (dst_base as isize + y as isize * stride) as usize;
        let row = &mut dst[row_off..][..width];

        let mut x = 0;
        while x + 32 <= width {
            storeu_256!((&mut row[x..x + 32]), [u8; 32], fill_val);
            x += 32;
        }
        while x + 16 <= width {
            storeu_128!(
                &mut row[x..x + 16],
                [u8; 16],
                _mm256_castsi256_si128(fill_val)
            );
            x += 16;
        }
        while x < width {
            row[x] = dc_val;
            x += 1;
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
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
    _dst: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    let buf_len = compute_ipred_buf_len(stride as isize, width as usize, height as usize);
    let dst_sl = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_len) };
    let (tl_sl, tl_off) =
        compute_topleft_slice(topleft as *const u8, width as usize, height as usize);
    ipred_dc_left_8bpc_inner(
        token,
        dst_sl,
        0,
        stride as isize,
        tl_sl,
        tl_off,
        width as usize,
        height as usize,
    );
}

// ============================================================================
// PAETH Prediction AVX-512
// ============================================================================

/// PAETH prediction 8bpc using AVX-512 — 16 pixels/iter with mask-based blending.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_paeth_8bpc_avx512_inner(
    _token: Server64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    topleft: &[u8],
    tl_off: usize,
    width: usize,
    height: usize,
) {
    let mut dst = dst.flex_mut();
    let topleft = topleft.flex();
    let topleft_val = topleft[tl_off] as i32;
    let topleft_vec = _mm512_set1_epi32(topleft_val);

    for y in 0..height {
        let row_off = (dst_base as isize + y as isize * stride) as usize;
        let left_val = topleft[tl_off - y - 1] as i32;
        let left_vec = _mm512_set1_epi32(left_val);

        let mut x = 0;
        while x + 16 <= width {
            // Load 16 top pixels → i32
            let top_bytes = loadu_128!(
                &topleft[tl_off + 1 + x..tl_off + 1 + x + 16],
                [u8; 16]
            );
            let top = _mm512_cvtepu8_epi32(top_bytes);

            // base = left + top - topleft
            let base = _mm512_sub_epi32(_mm512_add_epi32(left_vec, top), topleft_vec);

            let ldiff = _mm512_abs_epi32(_mm512_sub_epi32(left_vec, base));
            let tdiff = _mm512_abs_epi32(_mm512_sub_epi32(top, base));
            let tldiff = _mm512_abs_epi32(_mm512_sub_epi32(topleft_vec, base));

            // AVX-512 mask comparisons: cmpgt returns __mmask16
            // ldiff <= tdiff: !(ldiff > tdiff) = ~(cmpgt(ldiff, tdiff))
            let ld_le_td = !_mm512_cmpgt_epi32_mask(ldiff, tdiff);
            let ld_le_tld = !_mm512_cmpgt_epi32_mask(ldiff, tldiff);
            let td_le_tld = !_mm512_cmpgt_epi32_mask(tdiff, tldiff);

            // use_left = ldiff <= tdiff && ldiff <= tldiff
            let use_left = ld_le_td & ld_le_tld;
            // use_top = !use_left && tdiff <= tldiff
            let use_top = !use_left & td_le_tld;

            // Start with topleft, overlay top where use_top, overlay left where use_left
            let result = _mm512_mask_blend_epi32(
                use_left,
                _mm512_mask_blend_epi32(use_top, topleft_vec, top),
                left_vec,
            );

            // Pack i32→u8 directly (values are 0..255, clamping is safe)
            let clamped = _mm512_max_epi32(result, _mm512_setzero_si512());
            let result_u8: __m128i = _mm512_cvtusepi32_epi8(clamped);
            storeu_128!(
                &mut dst[row_off + x..row_off + x + 16],
                [u8; 16],
                result_u8
            );

            x += 16;
        }

        // Scalar fallback
        let row = &mut dst[row_off..][..width];
        while x < width {
            let top_val = topleft[tl_off + 1 + x] as i32;
            let base = left_val + top_val - topleft_val;
            let ldiff = (left_val - base).abs();
            let tdiff = (top_val - base).abs();
            let tldiff = (topleft_val - base).abs();
            let result = if ldiff <= tdiff && ldiff <= tldiff {
                left_val
            } else if tdiff <= tldiff {
                top_val
            } else {
                topleft_val
            };
            row[x] = result as u8;
            x += 1;
        }
    }
}

// ============================================================================
// SMOOTH Prediction AVX-512 (8bpc)
// ============================================================================

/// Smooth prediction 8bpc using AVX-512 — 16 pixels/iter.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_smooth_8bpc_avx512_inner(
    _token: Server64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    topleft: &[u8],
    tl_off: usize,
    width: usize,
    height: usize,
) {
    let mut dst = dst.flex_mut();
    let topleft = topleft.flex();
    let weights_hor = &dav1d_sm_weights[width..][..width];
    let weights_ver = &dav1d_sm_weights[height..][..height];
    let right_val = topleft[tl_off + width] as i32;
    let bottom_val = topleft[tl_off - height] as i32;
    let right_vec = _mm512_set1_epi32(right_val);
    let bottom_vec = _mm512_set1_epi32(bottom_val);
    let rounding = _mm512_set1_epi32(256);
    let c256 = _mm512_set1_epi32(256);
    let zero_512 = _mm512_setzero_si512();

    for y in 0..height {
        let row_off = (dst_base as isize + y as isize * stride) as usize;
        let left_val = topleft[tl_off - y - 1] as i32;
        let left_vec = _mm512_set1_epi32(left_val);
        let w_v = weights_ver[y] as i32;
        let w_v_vec = _mm512_set1_epi32(w_v);
        let w_v_inv = _mm512_sub_epi32(c256, w_v_vec);

        let mut x = 0;
        while x + 16 <= width {
            // Load 16 top pixels → i32
            let top_bytes = loadu_128!(
                &topleft[tl_off + 1 + x..tl_off + 1 + x + 16],
                [u8; 16]
            );
            let top = _mm512_cvtepu8_epi32(top_bytes);

            // Load 16 horizontal weights → i32
            let wh_bytes = loadu_128!(
                &weights_hor[x..x + 16],
                [u8; 16]
            );
            let w_h = _mm512_cvtepu8_epi32(wh_bytes);
            let w_h_inv = _mm512_sub_epi32(c256, w_h);

            let vert = _mm512_add_epi32(
                _mm512_mullo_epi32(w_v_vec, top),
                _mm512_mullo_epi32(w_v_inv, bottom_vec),
            );
            let hor = _mm512_add_epi32(
                _mm512_mullo_epi32(w_h, left_vec),
                _mm512_mullo_epi32(w_h_inv, right_vec),
            );

            let pred = _mm512_add_epi32(vert, hor);
            let result = _mm512_srai_epi32::<9>(_mm512_add_epi32(pred, rounding));

            let clamped = _mm512_max_epi32(result, zero_512);
            let result_u8: __m128i = _mm512_cvtusepi32_epi8(clamped);
            storeu_128!(
                &mut dst[row_off + x..row_off + x + 16],
                [u8; 16],
                result_u8
            );

            x += 16;
        }

        // Scalar fallback
        let row = &mut dst[row_off..][..width];
        while x < width {
            let top_val = topleft[tl_off + 1 + x] as i32;
            let w_h = weights_hor[x] as i32;
            let pred =
                w_v * top_val + (256 - w_v) * bottom_val + w_h * left_val + (256 - w_h) * right_val;
            row[x] = ((pred + 256) >> 9) as u8;
            x += 1;
        }
    }
}

/// Smooth_V prediction 8bpc using AVX-512 — 16 pixels/iter.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_smooth_v_8bpc_avx512_inner(
    _token: Server64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    topleft: &[u8],
    tl_off: usize,
    width: usize,
    height: usize,
) {
    let mut dst = dst.flex_mut();
    let topleft = topleft.flex();
    let weights_ver = &dav1d_sm_weights[height..][..height];
    let bottom_val = topleft[tl_off - height] as i32;
    let bottom_vec = _mm512_set1_epi32(bottom_val);
    let rounding = _mm512_set1_epi32(128);
    let c256 = _mm512_set1_epi32(256);
    let zero_512 = _mm512_setzero_si512();

    for y in 0..height {
        let row_off = (dst_base as isize + y as isize * stride) as usize;
        let w_v = weights_ver[y] as i32;
        let w_v_vec = _mm512_set1_epi32(w_v);
        let w_v_inv = _mm512_sub_epi32(c256, w_v_vec);

        let mut x = 0;
        while x + 16 <= width {
            let top_bytes = loadu_128!(
                &topleft[tl_off + 1 + x..tl_off + 1 + x + 16],
                [u8; 16]
            );
            let top = _mm512_cvtepu8_epi32(top_bytes);

            let pred = _mm512_add_epi32(
                _mm512_mullo_epi32(w_v_vec, top),
                _mm512_mullo_epi32(w_v_inv, bottom_vec),
            );
            let result = _mm512_srai_epi32::<8>(_mm512_add_epi32(pred, rounding));

            let clamped = _mm512_max_epi32(result, zero_512);
            let result_u8: __m128i = _mm512_cvtusepi32_epi8(clamped);
            storeu_128!(
                &mut dst[row_off + x..row_off + x + 16],
                [u8; 16],
                result_u8
            );

            x += 16;
        }

        let row = &mut dst[row_off..][..width];
        while x < width {
            let top_val = topleft[tl_off + 1 + x] as i32;
            let pred = w_v * top_val + (256 - w_v) * bottom_val;
            row[x] = ((pred + 128) >> 8) as u8;
            x += 1;
        }
    }
}

/// Smooth_H prediction 8bpc using AVX-512 — 16 pixels/iter.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_smooth_h_8bpc_avx512_inner(
    _token: Server64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    topleft: &[u8],
    tl_off: usize,
    width: usize,
    height: usize,
) {
    let mut dst = dst.flex_mut();
    let topleft = topleft.flex();
    let weights_hor = &dav1d_sm_weights[width..][..width];
    let right_val = topleft[tl_off + width] as i32;
    let right_vec = _mm512_set1_epi32(right_val);
    let rounding = _mm512_set1_epi32(128);
    let c256 = _mm512_set1_epi32(256);
    let zero_512 = _mm512_setzero_si512();

    for y in 0..height {
        let row_off = (dst_base as isize + y as isize * stride) as usize;
        let left_val = topleft[tl_off - y - 1] as i32;
        let left_vec = _mm512_set1_epi32(left_val);

        let mut x = 0;
        while x + 16 <= width {
            let wh_bytes = loadu_128!(
                &weights_hor[x..x + 16],
                [u8; 16]
            );
            let w_h = _mm512_cvtepu8_epi32(wh_bytes);
            let w_h_inv = _mm512_sub_epi32(c256, w_h);

            let pred = _mm512_add_epi32(
                _mm512_mullo_epi32(w_h, left_vec),
                _mm512_mullo_epi32(w_h_inv, right_vec),
            );
            let result = _mm512_srai_epi32::<8>(_mm512_add_epi32(pred, rounding));

            let clamped = _mm512_max_epi32(result, zero_512);
            let result_u8: __m128i = _mm512_cvtusepi32_epi8(clamped);
            storeu_128!(
                &mut dst[row_off + x..row_off + x + 16],
                [u8; 16],
                result_u8
            );

            x += 16;
        }

        let row = &mut dst[row_off..][..width];
        while x < width {
            let w_h = weights_hor[x] as i32;
            let pred = w_h * left_val + (256 - w_h) * right_val;
            row[x] = ((pred + 128) >> 8) as u8;
            x += 1;
        }
    }
}

// ============================================================================
// PAETH Prediction
// ============================================================================

/// PAETH prediction: each pixel is closest of left, top, or topleft to (left + top - topleft)
///
/// For each pixel at (x, y):
///   base = left + top - topleft
///   ldiff = |left - base|
///   tdiff = |top - base|
///   tldiff = |topleft - base|
///   pick whichever of left/top/topleft has smallest diff
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_paeth_8bpc_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    topleft: &[u8],
    tl_off: usize,
    width: usize,
    height: usize,
) {
    let mut dst = dst.flex_mut();
    let topleft = topleft.flex();
    let topleft_val = topleft[tl_off] as i32;
    let topleft_vec = _mm256_set1_epi32(topleft_val);

    for y in 0..height {
        let row_off = (dst_base as isize + y as isize * stride) as usize;
        let left_val = topleft[tl_off - y - 1] as i32;
        let left_vec = _mm256_set1_epi32(left_val);

        // Process 8 pixels at a time with AVX2
        let mut x = 0;
        while x + 8 <= width {
            // Load 8 top pixels and zero-extend to 32-bit
            let top_bytes = partial_simd::mm_loadl_epi64::<[u8; 8]>(
                (&topleft[tl_off + 1 + x..tl_off + 1 + x + 8])
                    .try_into()
                    .unwrap(),
            );
            let top_lo = _mm256_cvtepu8_epi32(top_bytes);

            // base = left + top - topleft
            let base = _mm256_sub_epi32(_mm256_add_epi32(left_vec, top_lo), topleft_vec);

            // ldiff = |left - base|
            let ldiff = _mm256_abs_epi32(_mm256_sub_epi32(left_vec, base));
            // tdiff = |top - base|
            let tdiff = _mm256_abs_epi32(_mm256_sub_epi32(top_lo, base));
            // tldiff = |topleft - base|
            let tldiff = _mm256_abs_epi32(_mm256_sub_epi32(topleft_vec, base));

            // Comparison: ldiff <= tdiff
            let ld_le_td = _mm256_or_si256(
                _mm256_cmpgt_epi32(tdiff, ldiff),
                _mm256_cmpeq_epi32(ldiff, tdiff),
            );
            // Comparison: ldiff <= tldiff
            let ld_le_tld = _mm256_or_si256(
                _mm256_cmpgt_epi32(tldiff, ldiff),
                _mm256_cmpeq_epi32(ldiff, tldiff),
            );
            // Comparison: tdiff <= tldiff
            let td_le_tld = _mm256_or_si256(
                _mm256_cmpgt_epi32(tldiff, tdiff),
                _mm256_cmpeq_epi32(tdiff, tldiff),
            );

            // if ldiff <= tdiff && ldiff <= tldiff: left
            // else if tdiff <= tldiff: top
            // else: topleft
            let use_left = _mm256_and_si256(ld_le_td, ld_le_tld);
            let use_top = _mm256_andnot_si256(use_left, td_le_tld);

            // Select: start with topleft, blend top if use_top, blend left if use_left
            let result = _mm256_blendv_epi8(
                _mm256_blendv_epi8(topleft_vec, top_lo, use_top),
                left_vec,
                use_left,
            );

            // Pack 32-bit to 8-bit
            let packed = _mm256_shuffle_epi8(
                result,
                _mm256_setr_epi8(
                    0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 4, 8, 12, -1,
                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                ),
            );
            let lo = _mm256_castsi256_si128(packed);
            let hi = _mm256_extracti128_si256::<1>(packed);
            let combined = _mm_unpacklo_epi32(lo, hi);
            partial_simd::mm_storel_epi64::<[u8; 8]>(
                (&mut dst[row_off + x..row_off + x + 8]).try_into().unwrap(),
                combined,
            );

            x += 8;
        }

        // Scalar fallback for remaining pixels
        let row = &mut dst[row_off..][..width];
        while x < width {
            let top_val = topleft[tl_off + 1 + x] as i32;
            let base = left_val + top_val - topleft_val;
            let ldiff = (left_val - base).abs();
            let tdiff = (top_val - base).abs();
            let tldiff = (topleft_val - base).abs();

            let result = if ldiff <= tdiff && ldiff <= tldiff {
                left_val
            } else if tdiff <= tldiff {
                top_val
            } else {
                topleft_val
            };
            row[x] = result as u8;
            x += 1;
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn ipred_paeth_8bpc_avx2(
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
    let token = unsafe { Desktop64::forge_token_dangerously() };
    let buf_len = compute_ipred_buf_len(stride as isize, width as usize, height as usize);
    let dst_sl = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_len) };
    let (tl_sl, tl_off) =
        compute_topleft_slice(topleft as *const u8, width as usize, height as usize);
    ipred_paeth_8bpc_inner(
        token,
        dst_sl,
        0,
        stride as isize,
        tl_sl,
        tl_off,
        width as usize,
        height as usize,
    );
}

// ============================================================================
// SMOOTH Predictions (using weight tables)
// ============================================================================

use crate::src::tables::dav1d_sm_weights;

/// SMOOTH prediction: weighted blend of top/bottom and left/right edges
///
/// pred = w_v[y] * top + (256 - w_v[y]) * bottom + w_h[x] * left + (256 - w_h[x]) * right
/// dst = (pred + 256) >> 9
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_smooth_8bpc_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    topleft: &[u8],
    tl_off: usize,
    width: usize,
    height: usize,
) {
    let mut dst = dst.flex_mut();
    let topleft = topleft.flex();
    let weights_hor = &dav1d_sm_weights[width..][..width];
    let weights_ver = &dav1d_sm_weights[height..][..height];
    let right_val = topleft[tl_off + width] as i32;
    let bottom_val = topleft[tl_off - height] as i32;
    let right_vec = _mm256_set1_epi32(right_val);
    let bottom_vec = _mm256_set1_epi32(bottom_val);
    let rounding = _mm256_set1_epi32(256);
    let c256 = _mm256_set1_epi32(256);

    for y in 0..height {
        let row_off = (dst_base as isize + y as isize * stride) as usize;
        let left_val = topleft[tl_off - y - 1] as i32;
        let left_vec = _mm256_set1_epi32(left_val);
        let w_v = weights_ver[y] as i32;
        let w_v_vec = _mm256_set1_epi32(w_v);
        let w_v_inv = _mm256_sub_epi32(c256, w_v_vec);

        let mut x = 0;
        while x + 8 <= width {
            // Load 8 top pixels
            let top_bytes = partial_simd::mm_loadl_epi64::<[u8; 8]>(
                (&topleft[tl_off + 1 + x..tl_off + 1 + x + 8])
                    .try_into()
                    .unwrap(),
            );
            let top = _mm256_cvtepu8_epi32(top_bytes);

            // Load 8 horizontal weights
            let w_h_bytes = partial_simd::mm_loadl_epi64::<[u8; 8]>(
                (&weights_hor[x..x + 8]).try_into().unwrap(),
            );
            let w_h = _mm256_cvtepu8_epi32(w_h_bytes);
            let w_h_inv = _mm256_sub_epi32(c256, w_h);

            // Vertical component: w_v * top + (256 - w_v) * bottom
            let vert = _mm256_add_epi32(
                _mm256_mullo_epi32(w_v_vec, top),
                _mm256_mullo_epi32(w_v_inv, bottom_vec),
            );

            // Horizontal component: w_h * left + (256 - w_h) * right
            let hor = _mm256_add_epi32(
                _mm256_mullo_epi32(w_h, left_vec),
                _mm256_mullo_epi32(w_h_inv, right_vec),
            );

            // pred = vert + hor, result = (pred + 256) >> 9
            let pred = _mm256_add_epi32(vert, hor);
            let result = _mm256_srai_epi32::<9>(_mm256_add_epi32(pred, rounding));

            // Pack to 8-bit
            let packed = _mm256_shuffle_epi8(
                result,
                _mm256_setr_epi8(
                    0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 4, 8, 12, -1,
                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                ),
            );
            let lo = _mm256_castsi256_si128(packed);
            let hi = _mm256_extracti128_si256::<1>(packed);
            let combined = _mm_unpacklo_epi32(lo, hi);
            partial_simd::mm_storel_epi64::<[u8; 8]>(
                (&mut dst[row_off + x..row_off + x + 8]).try_into().unwrap(),
                combined,
            );

            x += 8;
        }

        // Scalar fallback
        let row = &mut dst[row_off..][..width];
        while x < width {
            let top_val = topleft[tl_off + 1 + x] as i32;
            let w_h = weights_hor[x] as i32;
            let pred =
                w_v * top_val + (256 - w_v) * bottom_val + w_h * left_val + (256 - w_h) * right_val;
            row[x] = ((pred + 256) >> 9) as u8;
            x += 1;
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn ipred_smooth_8bpc_avx2(
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
    let token = unsafe { Desktop64::forge_token_dangerously() };
    let buf_len = compute_ipred_buf_len(stride as isize, width as usize, height as usize);
    let dst_sl = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_len) };
    let (tl_sl, tl_off) =
        compute_topleft_slice(topleft as *const u8, width as usize, height as usize);
    ipred_smooth_8bpc_inner(
        token,
        dst_sl,
        0,
        stride as isize,
        tl_sl,
        tl_off,
        width as usize,
        height as usize,
    );
}

/// SMOOTH_V prediction: vertical-only weighted blend (top/bottom)
///
/// pred = w_v[y] * top + (256 - w_v[y]) * bottom
/// dst = (pred + 128) >> 8
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_smooth_v_8bpc_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    topleft: &[u8],
    tl_off: usize,
    width: usize,
    height: usize,
) {
    let mut dst = dst.flex_mut();
    let topleft = topleft.flex();
    let weights_ver = &dav1d_sm_weights[height..][..height];
    let bottom_val = topleft[tl_off - height] as i32;
    let bottom_vec = _mm256_set1_epi32(bottom_val);
    let rounding = _mm256_set1_epi32(128);
    let c256 = _mm256_set1_epi32(256);

    for y in 0..height {
        let row_off = (dst_base as isize + y as isize * stride) as usize;
        let w_v = weights_ver[y] as i32;
        let w_v_vec = _mm256_set1_epi32(w_v);
        let w_v_inv = _mm256_sub_epi32(c256, w_v_vec);

        let mut x = 0;
        while x + 8 <= width {
            // Load 8 top pixels
            let top_bytes = partial_simd::mm_loadl_epi64::<[u8; 8]>(
                (&topleft[tl_off + 1 + x..tl_off + 1 + x + 8])
                    .try_into()
                    .unwrap(),
            );
            let top = _mm256_cvtepu8_epi32(top_bytes);

            // pred = w_v * top + (256 - w_v) * bottom
            let pred = _mm256_add_epi32(
                _mm256_mullo_epi32(w_v_vec, top),
                _mm256_mullo_epi32(w_v_inv, bottom_vec),
            );

            // result = (pred + 128) >> 8
            let result = _mm256_srai_epi32::<8>(_mm256_add_epi32(pred, rounding));

            // Pack to 8-bit
            let packed = _mm256_shuffle_epi8(
                result,
                _mm256_setr_epi8(
                    0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 4, 8, 12, -1,
                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                ),
            );
            let lo = _mm256_castsi256_si128(packed);
            let hi = _mm256_extracti128_si256::<1>(packed);
            let combined = _mm_unpacklo_epi32(lo, hi);
            partial_simd::mm_storel_epi64::<[u8; 8]>(
                (&mut dst[row_off + x..row_off + x + 8]).try_into().unwrap(),
                combined,
            );

            x += 8;
        }

        // Scalar fallback
        let row = &mut dst[row_off..][..width];
        while x < width {
            let top_val = topleft[tl_off + 1 + x] as i32;
            let pred = w_v * top_val + (256 - w_v) * bottom_val;
            row[x] = ((pred + 128) >> 8) as u8;
            x += 1;
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn ipred_smooth_v_8bpc_avx2(
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
    let token = unsafe { Desktop64::forge_token_dangerously() };
    let buf_len = compute_ipred_buf_len(stride as isize, width as usize, height as usize);
    let dst_sl = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_len) };
    let (tl_sl, tl_off) =
        compute_topleft_slice(topleft as *const u8, width as usize, height as usize);
    ipred_smooth_v_8bpc_inner(
        token,
        dst_sl,
        0,
        stride as isize,
        tl_sl,
        tl_off,
        width as usize,
        height as usize,
    );
}

/// SMOOTH_H prediction: horizontal-only weighted blend (left/right)
///
/// pred = w_h[x] * left + (256 - w_h[x]) * right
/// dst = (pred + 128) >> 8
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_smooth_h_8bpc_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    topleft: &[u8],
    tl_off: usize,
    width: usize,
    height: usize,
) {
    let mut dst = dst.flex_mut();
    let topleft = topleft.flex();
    let weights_hor = &dav1d_sm_weights[width..][..width];
    let right_val = topleft[tl_off + width] as i32;
    let right_vec = _mm256_set1_epi32(right_val);
    let rounding = _mm256_set1_epi32(128);
    let c256 = _mm256_set1_epi32(256);

    for y in 0..height {
        let row_off = (dst_base as isize + y as isize * stride) as usize;
        let left_val = topleft[tl_off - y - 1] as i32;
        let left_vec = _mm256_set1_epi32(left_val);

        let mut x = 0;
        while x + 8 <= width {
            // Load 8 horizontal weights
            let w_h_bytes = partial_simd::mm_loadl_epi64::<[u8; 8]>(
                (&weights_hor[x..x + 8]).try_into().unwrap(),
            );
            let w_h = _mm256_cvtepu8_epi32(w_h_bytes);
            let w_h_inv = _mm256_sub_epi32(c256, w_h);

            // pred = w_h * left + (256 - w_h) * right
            let pred = _mm256_add_epi32(
                _mm256_mullo_epi32(w_h, left_vec),
                _mm256_mullo_epi32(w_h_inv, right_vec),
            );

            // result = (pred + 128) >> 8
            let result = _mm256_srai_epi32::<8>(_mm256_add_epi32(pred, rounding));

            // Pack to 8-bit
            let packed = _mm256_shuffle_epi8(
                result,
                _mm256_setr_epi8(
                    0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 4, 8, 12, -1,
                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                ),
            );
            let lo = _mm256_castsi256_si128(packed);
            let hi = _mm256_extracti128_si256::<1>(packed);
            let combined = _mm_unpacklo_epi32(lo, hi);
            partial_simd::mm_storel_epi64::<[u8; 8]>(
                (&mut dst[row_off + x..row_off + x + 8]).try_into().unwrap(),
                combined,
            );

            x += 8;
        }

        // Scalar fallback
        let row = &mut dst[row_off..][..width];
        while x < width {
            let w_h = weights_hor[x] as i32;
            let pred = w_h * left_val + (256 - w_h) * right_val;
            row[x] = ((pred + 128) >> 8) as u8;
            x += 1;
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn ipred_smooth_h_8bpc_avx2(
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
    let token = unsafe { Desktop64::forge_token_dangerously() };
    let buf_len = compute_ipred_buf_len(stride as isize, width as usize, height as usize);
    let dst_sl = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_len) };
    let (tl_sl, tl_off) =
        compute_topleft_slice(topleft as *const u8, width as usize, height as usize);
    ipred_smooth_h_8bpc_inner(
        token,
        dst_sl,
        0,
        stride as isize,
        tl_sl,
        tl_off,
        width as usize,
        height as usize,
    );
}

// ============================================================================
// FILTER Prediction (filter intra)
// ============================================================================

use crate::src::tables::{dav1d_dr_intra_derivative, dav1d_filter_intra_taps, filter_fn, FLT_INCR};

/// FILTER prediction: uses directional filter taps on 4x2 blocks
///
/// Processes in 4x2 blocks. Each output pixel is:
/// sum = sum(filter[i] * p[i] for i in 0..7)
/// out = (sum + 8) >> 4
///
/// Input pixels:
/// p0 = topleft, p1-p4 = top row (4 pixels), p5-p6 = left column (2 pixels)
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_filter_8bpc_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    topleft: &[u8],
    tl_off: usize,
    width: usize,
    height: usize,
    filt_idx: i32,
    topleft_off: usize,
) {
    let mut dst = dst.flex_mut();
    let topleft = topleft.flex();
    let width = (width / 4) * 4; // Round down to multiple of 4
    let filt_idx = (filt_idx as usize) & 511;

    let filter = &dav1d_filter_intra_taps[filt_idx];

    // Process in 4x2 blocks
    for y in (0..height).step_by(2) {
        let cur_tl_off = topleft_off - y;
        let mut tl_pixel = topleft[tl_off.wrapping_add(cur_tl_off)] as i32;

        let row0_off = (dst_base as isize + y as isize * stride) as usize;
        let row1_off = (dst_base as isize + (y + 1) as isize * stride) as usize;

        for x in (0..width).step_by(4) {
            // Get top 4 pixels (p1-p4)
            // y=0: from topleft buffer; y>=2: from previously-written output row y-1
            let (p1, p2, p3, p4) = if y == 0 {
                let top_base = tl_off.wrapping_add(topleft_off + 1 + x);
                (
                    topleft[top_base] as i32,
                    topleft[top_base + 1] as i32,
                    topleft[top_base + 2] as i32,
                    topleft[top_base + 3] as i32,
                )
            } else {
                let top_row = (dst_base as isize + (y as isize - 1) * stride) as usize;
                (
                    dst[top_row + x] as i32,
                    dst[top_row + x + 1] as i32,
                    dst[top_row + x + 2] as i32,
                    dst[top_row + x + 3] as i32,
                )
            };

            // Get left 2 pixels (p5, p6)
            let (p5, p6) = if x == 0 {
                // From original topleft buffer
                let left_base = tl_off.wrapping_add(cur_tl_off.wrapping_sub(1));
                (
                    topleft[left_base] as i32,
                    topleft[left_base.wrapping_sub(1)] as i32,
                )
            } else {
                // From previously computed output
                (dst[row0_off + x - 1] as i32, dst[row1_off + x - 1] as i32)
            };

            let p0 = tl_pixel;
            let p = [p0, p1, p2, p3, p4, p5, p6];

            // Process 4x2 = 8 output pixels using filter taps
            let flt = filter.as_slice();
            let mut flt_offset = 0;

            // Row 0 (4 pixels)
            for xx in 0..4 {
                let acc = filter_fn(&flt[flt_offset..], p);
                let val = ((acc + 8) >> 4).clamp(0, 255) as u8;
                dst[row0_off + x + xx] = val;
                flt_offset += FLT_INCR;
            }

            // Row 1 (4 pixels)
            for xx in 0..4 {
                let acc = filter_fn(&flt[flt_offset..], p);
                let val = ((acc + 8) >> 4).clamp(0, 255) as u8;
                dst[row1_off + x + xx] = val;
                flt_offset += FLT_INCR;
            }

            // Update topleft for next 4x2 block (8bpc)
            tl_pixel = p4;
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn ipred_filter_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    topleft: *const DynPixel,
    width: c_int,
    height: c_int,
    filt_idx: c_int,
    _max_width: c_int,
    _max_height: c_int,
    _bitdepth_max: c_int,
    topleft_off: usize,
    _dst: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    let buf_len = compute_ipred_buf_len(stride as isize, width as usize, height as usize);
    let dst_sl = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_len) };
    let (tl_sl, tl_off) =
        compute_topleft_slice(topleft as *const u8, width as usize, height as usize);
    ipred_filter_8bpc_inner(
        token,
        dst_sl,
        0,
        stride as isize,
        tl_sl,
        tl_off,
        width as usize,
        height as usize,
        filt_idx as i32,
        topleft_off,
    );
}

// ============================================================================
// Z1 Prediction (angular prediction for angles < 90)
// ============================================================================

/// Z1 prediction: directional prediction using top edge only (angles < 90°)
///
/// For each pixel (x, y):
///   xpos = (y + 1) * dx
///   base = (xpos >> 6) + base_inc * x
///   frac = xpos & 0x3e
///   out = (top[base] * (64 - frac) + top[base+1] * frac + 32) >> 6
/// Z1 intra prediction SIMD inner for 8bpc.
/// Builds preprocessed top edge array internally, handles all cases.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_z1_8bpc_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    topleft: &[u8],
    tl_off: usize,
    width: usize,
    height: usize,
    angle: i32,
) {
    let mut dst = dst.flex_mut();
    let width_i = width as i32;
    let height_i = height as i32;

    // Extract angle flags
    let is_sm = (angle >> 9) & 1 != 0;
    let enable_intra_edge_filter = (angle >> 10) != 0;
    let angle = angle & 511;

    let mut dx = dav1d_dr_intra_derivative[(angle >> 1) as usize] as i32;

    let upsample_above = enable_intra_edge_filter
        && (90 - angle) < 40
        && ((width_i + height_i) as usize) <= (16 >> is_sm as usize);

    // Build preprocessed top edge array
    let mut top_out = [0u8; 64 + 64];
    let (top, max_base_x, base_inc);

    if upsample_above {
        upsample_edge_8bpc(
            &mut top_out,
            width_i + height_i,
            topleft,
            tl_off + 1,
            -1,
            width_i + std::cmp::min(width_i, height_i),
        );
        dx <<= 1;
        top = top_out.as_slice();
        max_base_x = (2 * (width_i + height_i) - 2) as usize;
        base_inc = 2usize;
    } else {
        let filter_strength = if enable_intra_edge_filter {
            get_filter_strength_simple(width_i + height_i, 90 - angle, is_sm)
        } else {
            0
        };
        if filter_strength != 0 {
            filter_edge_8bpc(
                &mut top_out,
                width_i + height_i,
                0,
                width_i + height_i,
                topleft,
                tl_off + 1,
                -1,
                width_i + std::cmp::min(width_i, height_i),
                filter_strength,
            );
            top = top_out.as_slice();
            max_base_x = (width_i + height_i - 1) as usize;
        } else {
            // No preprocessing needed — use topleft directly
            top = &topleft[tl_off + 1..];
            max_base_x = width + std::cmp::min(width, height) - 1;
        }
        base_inc = 1;
    };

    let top = top.flex();

    let rounding = _mm256_set1_epi16(32);

    for y in 0..height_i {
        let xpos = (y + 1) * dx;
        let frac = (xpos & 0x3e) as i16;
        let inv_frac = (64 - frac) as i16;

        let frac_vec = _mm256_set1_epi16(frac);
        let inv_frac_vec = _mm256_set1_epi16(inv_frac);

        let row_off = (dst_base as isize + y as isize * stride) as usize;
        let base0 = (xpos >> 6) as usize;

        let mut x = 0usize;

        // SIMD path - 16 pixels at a time (non-upsampled consecutive access)
        if base_inc == 1 {
            while x + 16 <= width && base0 + x + 16 < max_base_x {
                let base = base0 + x;

                let t0 = loadu_128!((&top[base..base + 16]), [u8; 16]);
                let t1 = loadu_128!((&top[base + 1..base + 17]), [u8; 16]);

                let t0_w = _mm256_cvtepu8_epi16(t0);
                let t1_w = _mm256_cvtepu8_epi16(t1);

                let prod0 = _mm256_mullo_epi16(t0_w, inv_frac_vec);
                let prod1 = _mm256_mullo_epi16(t1_w, frac_vec);
                let sum = _mm256_add_epi16(_mm256_add_epi16(prod0, prod1), rounding);
                let result = _mm256_srai_epi16::<6>(sum);

                let packed = _mm256_packus_epi16(result, result);
                let lo = _mm256_castsi256_si128(packed);
                let hi = _mm256_extracti128_si256::<1>(packed);
                let combined = _mm_unpacklo_epi64(lo, hi);
                storeu_128!(
                    (&mut dst[row_off + x..row_off + x + 16]),
                    [u8; 16],
                    combined
                );

                x += 16;
            }
        }

        // Scalar remainder (also handles upsampled stride-2 access)
        while x < width {
            let base = base0 + base_inc * x;
            if base < max_base_x {
                let t0 = top[base] as i32;
                let t1 = top[base + 1] as i32;
                let v = t0 * inv_frac as i32 + t1 * frac as i32;
                dst[row_off + x] = ((v + 32) >> 6) as u8;
            } else {
                let fill_val = top[max_base_x];
                for xx in x..width {
                    dst[row_off + xx] = fill_val;
                }
                break;
            }
            x += 1;
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn ipred_z1_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    topleft: *const DynPixel,
    width: c_int,
    height: c_int,
    angle: c_int,
    _max_width: c_int,
    _max_height: c_int,
    _bitdepth_max: c_int,
    _topleft_off: usize,
    _dst: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    let buf_len = compute_ipred_buf_len(stride as isize, width as usize, height as usize);
    let dst_sl = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_len) };
    let (tl_sl, tl_off) =
        compute_topleft_slice(topleft as *const u8, width as usize, height as usize);
    ipred_z1_8bpc_inner(
        token,
        dst_sl,
        0,
        stride as isize,
        tl_sl,
        tl_off,
        width as usize,
        height as usize,
        angle as i32,
    );
}

/// Helper: get filter strength (simplified version)
#[inline]
fn get_filter_strength_simple(wh: i32, angle: i32, is_sm: bool) -> i32 {
    if is_sm {
        match (wh, angle) {
            (..=8, 64..) => 2,
            (..=8, 40..) => 1,
            (..=8, ..) => 0,
            (..=16, 48..) => 2,
            (..=16, 20..) => 1,
            (..=16, ..) => 0,
            (..=24, 4..) => 3,
            (..=24, ..) => 0,
            (.., _) => 3,
        }
    } else {
        match (wh, angle) {
            (..=8, 56..) => 1,
            (..=8, ..) => 0,
            (..=16, 40..) => 1,
            (..=16, ..) => 0,
            (..=24, 32..) => 3,
            (..=24, 16..) => 2,
            (..=24, 8..) => 1,
            (..=24, ..) => 0,
            (..=32, 32..) => 3,
            (..=32, 4..) => 2,
            (..=32, ..) => 1,
            (.., _) => 3,
        }
    }
}

// ============================================================================
// Z2 Prediction (angular prediction for angles 90-180)
// ============================================================================

/// Z2 prediction: directional prediction using both top AND left edges (angles 90-180°)
///
/// Unlike Z1 (top only) and Z3 (left only), Z2 blends between edges:
/// - When base_x >= 0: interpolate from top edge
/// - When base_x < 0: interpolate from left edge
/// Filter edge pixels for Z2 prediction (8bpc version of filter_edge from ipred.rs).
fn filter_edge_8bpc(
    out: &mut [u8],
    sz: i32,
    lim_from: i32,
    lim_to: i32,
    inp: &[u8],
    in_off: usize,
    from: i32,
    to: i32,
    strength: i32,
) {
    static KERNEL: [[u8; 5]; 3] = [[0, 4, 8, 4, 0], [0, 5, 6, 5, 0], [2, 4, 4, 4, 2]];
    let mut i = 0;
    while i < std::cmp::min(sz, lim_from) {
        out[i as usize] = inp[in_off.wrapping_add_signed(i.clamp(from, to - 1) as isize)];
        i += 1;
    }
    while i < std::cmp::min(lim_to, sz) {
        let mut s = 0i32;
        for j in 0..5i32 {
            s += inp[in_off.wrapping_add_signed((i - 2 + j).clamp(from, to - 1) as isize)] as i32
                * KERNEL[(strength - 1) as usize][j as usize] as i32;
        }
        out[i as usize] = ((s + 8) >> 4) as u8;
        i += 1;
    }
    while i < sz {
        out[i as usize] = inp[in_off.wrapping_add_signed(i.clamp(from, to - 1) as isize)];
        i += 1;
    }
}

/// Upsample edge pixels for Z2 prediction (8bpc version of upsample_edge from ipred.rs).
fn upsample_edge_8bpc(out: &mut [u8], hsz: i32, inp: &[u8], in_off: usize, from: i32, to: i32) {
    let kernel: [i8; 4] = [-1, 9, 9, -1];
    for i in 0..hsz - 1 {
        out[(i * 2) as usize] = inp[in_off.wrapping_add_signed(i.clamp(from, to - 1) as isize)];
        let mut s = 0i32;
        for j in 0..4i32 {
            s += inp[in_off.wrapping_add_signed((i + j - 1).clamp(from, to - 1) as isize)] as i32
                * kernel[j as usize] as i32;
        }
        out[(i * 2 + 1) as usize] = ((s + 8) >> 4).clamp(0, 255) as u8;
    }
    let i = hsz - 1;
    out[(i * 2) as usize] = inp[in_off.wrapping_add_signed(i.clamp(from, to - 1) as isize)];
}

/// Z2 intra prediction SIMD inner for 8bpc.
/// Builds preprocessed edge array internally, handles all cases (filter/upsample/plain).
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_z2_8bpc_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    topleft: &[u8],
    tl_off: usize,
    width: usize,
    height: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
) {
    let mut dst = dst.flex_mut();
    let width_i = width as i32;
    let height_i = height as i32;

    // Extract angle flags
    let is_sm = (angle >> 9) & 1 != 0;
    let enable_intra_edge_filter = (angle >> 10) != 0;
    let angle = angle & 511;

    let mut dy = dav1d_dr_intra_derivative[((angle - 90) >> 1) as usize] as i32;
    let mut dx = dav1d_dr_intra_derivative[((180 - angle) >> 1) as usize] as i32;

    // Determine upsampling
    let upsample_left = enable_intra_edge_filter
        && (180 - angle) < 40
        && (width_i + height_i) <= (16 >> is_sm as i32);
    let upsample_above = enable_intra_edge_filter
        && (angle - 90) < 40
        && (width_i + height_i) <= (16 >> is_sm as i32);

    // Build preprocessed edge array (same as scalar ipred_z2_rust)
    let mut edge = [0u8; 64 + 64 + 1];
    let edge_tl = 64usize;

    // Top edge preprocessing
    if upsample_above {
        upsample_edge_8bpc(
            &mut edge[edge_tl..],
            width_i + 1,
            topleft,
            tl_off,
            0,
            width_i + 1,
        );
        dx <<= 1;
    } else {
        let filter_strength = if enable_intra_edge_filter {
            get_filter_strength_simple(width_i + height_i, angle - 90, is_sm)
        } else {
            0
        };
        if filter_strength != 0 {
            filter_edge_8bpc(
                &mut edge[edge_tl + 1..],
                width_i,
                0,
                max_width,
                topleft,
                tl_off + 1,
                -1,
                width_i,
                filter_strength,
            );
        } else {
            edge[edge_tl + 1..edge_tl + 1 + width].copy_from_slice(
                &topleft[tl_off + 1..tl_off + 1 + width],
            );
        }
    }

    // Left edge preprocessing
    if upsample_left {
        upsample_edge_8bpc(
            &mut edge[edge_tl - height * 2..],
            height_i + 1,
            topleft,
            tl_off.wrapping_sub(height),
            0,
            height_i + 1,
        );
        dy <<= 1;
    } else {
        let filter_strength = if enable_intra_edge_filter {
            get_filter_strength_simple(width_i + height_i, 180 - angle, is_sm)
        } else {
            0
        };
        if filter_strength != 0 {
            filter_edge_8bpc(
                &mut edge[edge_tl - height..],
                height_i,
                height_i - max_height,
                height_i,
                topleft,
                tl_off.wrapping_sub(height),
                0,
                height_i + 1,
                filter_strength,
            );
        } else {
            edge[edge_tl - height..edge_tl].copy_from_slice(
                &topleft[tl_off - height..tl_off],
            );
        }
    }

    // Corner pixel
    edge[edge_tl] = topleft[tl_off];

    let edge = edge.as_slice().flex();

    let base_inc_x = 1 + upsample_above as usize;
    let left = edge_tl - (1 + upsample_left as usize);

    let rounding = _mm256_set1_epi16(32);

    for y in 0..height_i {
        let xpos = ((1 + upsample_above as i32) << 6) - dx * (y + 1);
        let base_x0 = xpos >> 6;
        let frac_x = (xpos & 0x3e) as i16;
        let inv_frac_x = (64 - frac_x) as i16;

        let row_off = (dst_base as isize + y as isize * stride) as usize;

        // left_count = number of left-edge pixels (where base_x0 + base_inc_x * x < 0)
        let left_count = if base_x0 >= 0 {
            0usize
        } else {
            let needed = (-base_x0) as usize;
            ((needed + base_inc_x - 1) / base_inc_x).min(width)
        };

        // First: process pixels using left edge (x < left_count)
        let mut x = 0usize;
        while x < left_count {
            let ypos = ((y << (6 + upsample_left as i32))) - dy * (x as i32 + 1);
            let base_y = ypos >> 6;
            let frac_y = ypos & 0x3e;
            let inv_frac_y = 64 - frac_y;

            let l0_idx = left.wrapping_add_signed(-base_y as isize);
            let l1_idx = left.wrapping_add_signed(-(base_y + 1) as isize);
            let l0 = edge[l0_idx] as i32;
            let l1 = edge[l1_idx] as i32;
            let v = l0 * inv_frac_y + l1 * frac_y;
            dst[row_off + x] = ((v + 32) >> 6) as u8;
            x += 1;
        }

        // Then: process pixels using top edge (x >= left_count, base_x >= 0)
        if base_inc_x == 1 {
            while x + 16 <= width {
                let base_x = (base_x0 + x as i32) as usize;
                let idx = edge_tl + base_x;
                if idx + 17 > edge.len() {
                    break;
                }

                let t0 = loadu_128!((&edge[idx..idx + 16]), [u8; 16]);
                let t1 = loadu_128!((&edge[idx + 1..idx + 17]), [u8; 16]);

                let t0_w = _mm256_cvtepu8_epi16(t0);
                let t1_w = _mm256_cvtepu8_epi16(t1);

                let frac_vec = _mm256_set1_epi16(frac_x);
                let inv_frac_vec = _mm256_set1_epi16(inv_frac_x);

                let prod0 = _mm256_mullo_epi16(t0_w, inv_frac_vec);
                let prod1 = _mm256_mullo_epi16(t1_w, frac_vec);
                let sum = _mm256_add_epi16(_mm256_add_epi16(prod0, prod1), rounding);
                let result = _mm256_srai_epi16::<6>(sum);

                let packed = _mm256_packus_epi16(result, result);
                let lo = _mm256_castsi256_si128(packed);
                let hi = _mm256_extracti128_si256::<1>(packed);
                let combined = _mm_unpacklo_epi64(lo, hi);
                storeu_128!(
                    (&mut dst[row_off + x..row_off + x + 16]),
                    [u8; 16],
                    combined
                );

                x += 16;
            }
        }

        // Scalar remainder (also handles upsampled case with stride-2 edge access)
        while x < width {
            let base_x = (base_x0 + (base_inc_x * x) as i32) as usize;
            let idx = edge_tl + base_x;
            if idx + 2 > edge.len() {
                break;
            }
            let t0 = edge[idx] as i32;
            let t1 = edge[idx + 1] as i32;
            let v = t0 * inv_frac_x as i32 + t1 * frac_x as i32;
            dst[row_off + x] = ((v + 32) >> 6) as u8;
            x += 1;
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn ipred_z2_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    topleft: *const DynPixel,
    width: c_int,
    height: c_int,
    angle: c_int,
    max_width: c_int,
    max_height: c_int,
    _bitdepth_max: c_int,
    _topleft_off: usize,
    _dst: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    let buf_len = compute_ipred_buf_len(stride as isize, width as usize, height as usize);
    let dst_sl = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_len) };
    let (tl_sl, tl_off) =
        compute_topleft_slice(topleft as *const u8, width as usize, height as usize);
    ipred_z2_8bpc_inner(
        token,
        dst_sl,
        0,
        stride as isize,
        tl_sl,
        tl_off,
        width as usize,
        height as usize,
        angle as i32,
        max_width as i32,
        max_height as i32,
    );
}

// ============================================================================
// Z3 Prediction (angular prediction for angles > 180)
// ============================================================================

/// Z3 prediction: directional prediction using left edge only (angles > 180°)
///
/// Z3 is the mirror of Z1, using the left edge instead of top.
/// Builds preprocessed left edge array internally, handles all cases.
/// Loop order is column-major (outer x, inner y).
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_z3_8bpc_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    topleft: &[u8],
    tl_off: usize,
    width: usize,
    height: usize,
    angle: i32,
) {
    let mut dst = dst.flex_mut();
    let width_i = width as i32;
    let height_i = height as i32;

    // Extract angle flags
    let is_sm = (angle >> 9) & 1 != 0;
    let enable_intra_edge_filter = (angle >> 10) != 0;
    let angle = angle & 511;

    let mut dy = dav1d_dr_intra_derivative[((270 - angle) >> 1) as usize] as usize;

    let upsample_left = enable_intra_edge_filter
        && (angle - 180) < 40
        && (width_i + height_i) <= (16 >> is_sm as i32);

    // Build preprocessed left edge array
    // Scalar uses: left[left_off - base] to index from corner outward
    let mut left_out = [0u8; 64 + 64];
    let (left, left_off, max_base_y, base_inc);

    if upsample_left {
        upsample_edge_8bpc(
            &mut left_out,
            width_i + height_i,
            topleft,
            tl_off - (width + height),
            std::cmp::max(width_i - height_i, 0),
            width_i + height_i + 1,
        );
        left_off = (2 * (width_i + height_i) - 2) as usize;
        max_base_y = left_off;
        dy <<= 1;
        base_inc = 2usize;
        left = left_out.as_slice();
    } else {
        let filter_strength = if enable_intra_edge_filter {
            get_filter_strength_simple(width_i + height_i, angle - 180, is_sm)
        } else {
            0
        };
        if filter_strength != 0 {
            filter_edge_8bpc(
                &mut left_out,
                width_i + height_i,
                0,
                width_i + height_i,
                topleft,
                tl_off - (width + height),
                std::cmp::max(width_i - height_i, 0),
                width_i + height_i + 1,
                filter_strength,
            );
            left_off = (width_i + height_i - 1) as usize;
            max_base_y = left_off;
            left = left_out.as_slice();
        } else {
            // No preprocessing — use topleft directly
            left = topleft;
            left_off = tl_off - 1;
            max_base_y = height + std::cmp::min(width, height) - 1;
        }
        base_inc = 1;
    };

    let left = left.flex();

    // Column-major access pattern
    for x in 0..width {
        let ypos = dy * (x + 1);
        let frac = (ypos & 0x3e) as i32;
        let inv_frac = 64 - frac;

        for y in 0..height_i {
            let base = (ypos >> 6) + base_inc * y as usize;

            if base < max_base_y {
                let l0 = left[left_off - base] as i32;
                let l1 = left[left_off - base - 1] as i32;
                let v = l0 * inv_frac + l1 * frac;
                let pixel_off = (dst_base as isize + y as isize * stride) as usize + x;
                dst[pixel_off] = ((v + 32) >> 6) as u8;
            } else {
                let fill_val = left[left_off - max_base_y];
                for yy in y..height_i {
                    let pixel_off = (dst_base as isize + yy as isize * stride) as usize + x;
                    dst[pixel_off] = fill_val;
                }
                break;
            }
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn ipred_z3_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    topleft: *const DynPixel,
    width: c_int,
    height: c_int,
    angle: c_int,
    _max_width: c_int,
    _max_height: c_int,
    _bitdepth_max: c_int,
    _topleft_off: usize,
    _dst: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    let buf_len = compute_ipred_buf_len(stride as isize, width as usize, height as usize);
    let dst_sl = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_len) };
    let (tl_sl, tl_off) =
        compute_topleft_slice(topleft as *const u8, width as usize, height as usize);
    ipred_z3_8bpc_inner(
        token,
        dst_sl,
        0,
        stride as isize,
        tl_sl,
        tl_off,
        width as usize,
        height as usize,
        angle as i32,
    );
}

/// Compute a conservative buffer length for ipred dst buffers.
#[cfg(target_arch = "x86_64")]
fn compute_ipred_buf_len(stride: isize, width: usize, height: usize) -> usize {
    height.saturating_sub(1) * stride.unsigned_abs() + width
}

/// Construct a topleft slice + offset from a raw pointer.
///
/// The topleft pointer points to the center pixel of a scratch edge buffer.
/// We need both positive offsets (top row) and negative offsets (left column).
/// Returns (slice, offset_of_center_in_slice).
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
unsafe fn compute_topleft_slice<'a>(
    tl_ptr: *const u8,
    width: usize,
    height: usize,
) -> (&'a [u8], usize) {
    // Conservative bounds: need up to height+2 below and width+height+2 above
    let neg_reach = height + 2;
    let pos_reach = width + height + 2;
    let total = neg_reach + pos_reach;
    let base = unsafe { tl_ptr.sub(neg_reach) };
    (
        unsafe { std::slice::from_raw_parts(base, total) },
        neg_reach,
    )
}

// ============================================================================
// 16bpc IMPLEMENTATIONS
// ============================================================================

/// DC_128 prediction for 16bpc: fill block with mid-value
///
/// For 10bpc: fill with 512 (1 << 9)
/// For 12bpc: fill with 2048 (1 << 11)
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_dc_128_16bpc_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    width: usize,
    height: usize,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    // Mid-value is (bitdepth_max + 1) / 2
    let mid_val = ((bitdepth_max + 1) / 2) as u16;
    let fill_val = _mm256_set1_epi16(mid_val as i16);
    let width_bytes = width * 2;

    for y in 0..height {
        let row_off = (dst_base as isize + y as isize * stride) as usize;
        let mut x = 0usize;

        // Process 16 pixels at a time (256-bit / 16-bit = 16 pixels)
        while x + 16 <= width {
            let off = row_off + x * 2;
            storeu_256!((&mut dst[off..off + 32]), [u8; 32], fill_val);
            x += 16;
        }

        // Process 8 pixels at a time
        while x + 8 <= width {
            let off = row_off + x * 2;
            storeu_128!(
                (&mut dst[off..off + 16]),
                [u8; 16],
                _mm256_castsi256_si128(fill_val)
            );
            x += 8;
        }

        // Remaining pixels
        while x < width {
            let off = row_off + x * 2;
            dst[off..off + 2].copy_from_slice(&mid_val.to_ne_bytes());
            x += 1;
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn ipred_dc_128_16bpc_avx2(
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
    let token = unsafe { Desktop64::forge_token_dangerously() };
    let buf_len = compute_ipred_buf_len(stride as isize, width as usize * 2, height as usize);
    let dst_sl = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_len) };
    ipred_dc_128_16bpc_inner(
        token,
        dst_sl,
        0,
        stride as isize,
        width as usize,
        height as usize,
        bitdepth_max as i32,
    );
}

/// Vertical prediction for 16bpc: copy top row to all rows
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_v_16bpc_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    topleft: &[u8],
    tl_off: usize,
    width: usize,
    height: usize,
) {
    let mut dst = dst.flex_mut();
    let topleft = topleft.flex();
    // Top pixels start at topleft + 1 pixel = tl_off + 2 bytes
    let top_off = tl_off + 2;

    for y in 0..height {
        let row_off = (dst_base as isize + y as isize * stride) as usize;
        let mut x = 0usize;

        // Process 16 pixels at a time
        while x + 16 <= width {
            let load_off = top_off + x * 2;
            let top_vals = loadu_256!((&topleft[load_off..load_off + 32]), [u8; 32]);
            let store_off = row_off + x * 2;
            storeu_256!((&mut dst[store_off..store_off + 32]), [u8; 32], top_vals);
            x += 16;
        }

        // Process 8 pixels at a time
        while x + 8 <= width {
            let load_off = top_off + x * 2;
            let top_vals = loadu_128!((&topleft[load_off..load_off + 16]), [u8; 16]);
            let store_off = row_off + x * 2;
            storeu_128!((&mut dst[store_off..store_off + 16]), [u8; 16], top_vals);
            x += 8;
        }

        // Remaining pixels
        while x < width {
            let load_off = top_off + x * 2;
            let store_off = row_off + x * 2;
            dst[store_off..store_off + 2].copy_from_slice(&topleft[load_off..load_off + 2]);
            x += 1;
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn ipred_v_16bpc_avx2(
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
    let token = unsafe { Desktop64::forge_token_dangerously() };
    let buf_len = compute_ipred_buf_len(stride as isize, width as usize * 2, height as usize);
    let dst_sl = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_len) };
    let (tl_sl, tl_off) = compute_topleft_slice(
        topleft as *const u8,
        width as usize * 2,
        height as usize * 2,
    );
    ipred_v_16bpc_inner(
        token,
        dst_sl,
        0,
        stride as isize,
        tl_sl,
        tl_off,
        width as usize,
        height as usize,
    );
}

/// Horizontal prediction for 16bpc: fill each row with its left pixel
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_h_16bpc_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    topleft: &[u8],
    tl_off: usize,
    width: usize,
    height: usize,
) {
    let mut dst = dst.flex_mut();
    let topleft = topleft.flex();
    for y in 0..height {
        let row_off = (dst_base as isize + y as isize * stride) as usize;
        // Left pixel for this row: topleft[-(y+1)] in u16 units = tl_off - (y+1)*2 in bytes
        let left_byte_off = tl_off - (y + 1) * 2;
        let left_val = u16::from_ne_bytes(
            topleft[left_byte_off..left_byte_off + 2]
                .try_into()
                .unwrap(),
        );
        let fill_val = _mm256_set1_epi16(left_val as i16);

        let mut x = 0usize;

        // Process 16 pixels at a time
        while x + 16 <= width {
            let off = row_off + x * 2;
            storeu_256!((&mut dst[off..off + 32]), [u8; 32], fill_val);
            x += 16;
        }

        // Process 8 pixels at a time
        while x + 8 <= width {
            let off = row_off + x * 2;
            storeu_128!(
                (&mut dst[off..off + 16]),
                [u8; 16],
                _mm256_castsi256_si128(fill_val)
            );
            x += 8;
        }

        // Remaining pixels
        while x < width {
            let off = row_off + x * 2;
            dst[off..off + 2].copy_from_slice(&left_val.to_ne_bytes());
            x += 1;
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn ipred_h_16bpc_avx2(
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
    let token = unsafe { Desktop64::forge_token_dangerously() };
    let buf_len = compute_ipred_buf_len(stride as isize, width as usize * 2, height as usize);
    let dst_sl = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_len) };
    let (tl_sl, tl_off) = compute_topleft_slice(
        topleft as *const u8,
        width as usize * 2,
        height as usize * 2,
    );
    ipred_h_16bpc_inner(
        token,
        dst_sl,
        0,
        stride as isize,
        tl_sl,
        tl_off,
        width as usize,
        height as usize,
    );
}

/// DC_128 prediction for 16bpc using AVX-512
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_dc_128_16bpc_avx512_inner(
    _token: Server64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    width: usize,
    height: usize,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mid_val = ((bitdepth_max + 1) / 2) as u16;
    let fill_512 = _mm512_set1_epi16(mid_val as i16);
    let fill_256 = _mm256_set1_epi16(mid_val as i16);

    for y in 0..height {
        let row_off = (dst_base as isize + y as isize * stride) as usize;
        let mut x = 0usize;

        // 32 pixels at a time (512-bit / 16-bit = 32 pixels)
        while x + 32 <= width {
            let off = row_off + x * 2;
            storeu_512!((&mut dst[off..off + 64]), [u8; 64], fill_512);
            x += 32;
        }
        // 16 pixels at a time
        while x + 16 <= width {
            let off = row_off + x * 2;
            storeu_256!((&mut dst[off..off + 32]), [u8; 32], fill_256);
            x += 16;
        }
        // 8 pixels at a time
        while x + 8 <= width {
            let off = row_off + x * 2;
            storeu_128!(
                (&mut dst[off..off + 16]),
                [u8; 16],
                _mm256_castsi256_si128(fill_256)
            );
            x += 8;
        }
        while x < width {
            let off = row_off + x * 2;
            dst[off..off + 2].copy_from_slice(&mid_val.to_ne_bytes());
            x += 1;
        }
    }
}

/// Vertical prediction for 16bpc using AVX-512
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_v_16bpc_avx512_inner(
    _token: Server64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    topleft: &[u8],
    tl_off: usize,
    width: usize,
    height: usize,
) {
    let mut dst = dst.flex_mut();
    let topleft = topleft.flex();
    let top_off = tl_off + 2; // +1 pixel = +2 bytes for u16

    for y in 0..height {
        let row_off = (dst_base as isize + y as isize * stride) as usize;
        let mut x = 0usize;

        // 32 pixels at a time (64 bytes)
        while x + 32 <= width {
            let load_off = top_off + x * 2;
            let top_vals = loadu_512!((&topleft[load_off..load_off + 64]), [u8; 64]);
            let store_off = row_off + x * 2;
            storeu_512!((&mut dst[store_off..store_off + 64]), [u8; 64], top_vals);
            x += 32;
        }
        // 16 pixels at a time (32 bytes)
        while x + 16 <= width {
            let load_off = top_off + x * 2;
            let top_vals = loadu_256!((&topleft[load_off..load_off + 32]), [u8; 32]);
            let store_off = row_off + x * 2;
            storeu_256!((&mut dst[store_off..store_off + 32]), [u8; 32], top_vals);
            x += 16;
        }
        // 8 pixels at a time
        while x + 8 <= width {
            let load_off = top_off + x * 2;
            let top_vals = loadu_128!((&topleft[load_off..load_off + 16]), [u8; 16]);
            let store_off = row_off + x * 2;
            storeu_128!((&mut dst[store_off..store_off + 16]), [u8; 16], top_vals);
            x += 8;
        }
        while x < width {
            let load_off = top_off + x * 2;
            let store_off = row_off + x * 2;
            dst[store_off..store_off + 2].copy_from_slice(&topleft[load_off..load_off + 2]);
            x += 1;
        }
    }
}

/// Horizontal prediction for 16bpc using AVX-512
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_h_16bpc_avx512_inner(
    _token: Server64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    topleft: &[u8],
    tl_off: usize,
    width: usize,
    height: usize,
) {
    let mut dst = dst.flex_mut();
    let topleft = topleft.flex();
    for y in 0..height {
        let row_off = (dst_base as isize + y as isize * stride) as usize;
        let left_byte_off = tl_off - (y + 1) * 2;
        let left_val = u16::from_ne_bytes(
            topleft[left_byte_off..left_byte_off + 2]
                .try_into()
                .unwrap(),
        );
        let fill_512 = _mm512_set1_epi16(left_val as i16);
        let fill_256 = _mm256_set1_epi16(left_val as i16);

        let mut x = 0usize;
        while x + 32 <= width {
            let off = row_off + x * 2;
            storeu_512!((&mut dst[off..off + 64]), [u8; 64], fill_512);
            x += 32;
        }
        while x + 16 <= width {
            let off = row_off + x * 2;
            storeu_256!((&mut dst[off..off + 32]), [u8; 32], fill_256);
            x += 16;
        }
        while x + 8 <= width {
            let off = row_off + x * 2;
            storeu_128!(
                (&mut dst[off..off + 16]),
                [u8; 16],
                _mm256_castsi256_si128(fill_256)
            );
            x += 8;
        }
        while x < width {
            let off = row_off + x * 2;
            dst[off..off + 2].copy_from_slice(&left_val.to_ne_bytes());
            x += 1;
        }
    }
}

// ============================================================================
// DC Prediction AVX-512 variants (16bpc)
// ============================================================================

/// DC prediction for 16bpc using AVX-512
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_dc_16bpc_avx512_inner(
    _token: Server64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    topleft: &[u8],
    tl_off: usize,
    width: usize,
    height: usize,
) {
    let mut dst = dst.flex_mut();
    let topleft = topleft.flex();
    let mut sum = 0u32;
    for i in 1..=width {
        let off = tl_off + i * 2;
        sum += u16::from_ne_bytes(topleft[off..off + 2].try_into().unwrap()) as u32;
    }
    for i in 1..=height {
        let off = tl_off - i * 2;
        sum += u16::from_ne_bytes(topleft[off..off + 2].try_into().unwrap()) as u32;
    }
    let count = (width + height) as u32;
    let avg = ((sum + count / 2) / count) as u16;

    let fill_512 = _mm512_set1_epi16(avg as i16);
    let fill_256 = _mm256_set1_epi16(avg as i16);

    for y in 0..height {
        let row_off = (dst_base as isize + y as isize * stride) as usize;
        let mut x = 0usize;
        while x + 32 <= width {
            let off = row_off + x * 2;
            storeu_512!((&mut dst[off..off + 64]), [u8; 64], fill_512);
            x += 32;
        }
        while x + 16 <= width {
            let off = row_off + x * 2;
            storeu_256!((&mut dst[off..off + 32]), [u8; 32], fill_256);
            x += 16;
        }
        while x + 8 <= width {
            let off = row_off + x * 2;
            storeu_128!((&mut dst[off..off + 16]), [u8; 16], _mm256_castsi256_si128(fill_256));
            x += 8;
        }
        while x < width {
            let off = row_off + x * 2;
            dst[off..off + 2].copy_from_slice(&avg.to_ne_bytes());
            x += 1;
        }
    }
}

/// DC_TOP prediction for 16bpc using AVX-512
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_dc_top_16bpc_avx512_inner(
    _token: Server64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    topleft: &[u8],
    tl_off: usize,
    width: usize,
    height: usize,
) {
    let mut dst = dst.flex_mut();
    let topleft = topleft.flex();
    let mut sum = 0u32;
    for i in 1..=width {
        let off = tl_off + i * 2;
        sum += u16::from_ne_bytes(topleft[off..off + 2].try_into().unwrap()) as u32;
    }
    let avg = ((sum + width as u32 / 2) / width as u32) as u16;

    let fill_512 = _mm512_set1_epi16(avg as i16);
    let fill_256 = _mm256_set1_epi16(avg as i16);

    for y in 0..height {
        let row_off = (dst_base as isize + y as isize * stride) as usize;
        let mut x = 0usize;
        while x + 32 <= width {
            let off = row_off + x * 2;
            storeu_512!((&mut dst[off..off + 64]), [u8; 64], fill_512);
            x += 32;
        }
        while x + 16 <= width {
            let off = row_off + x * 2;
            storeu_256!((&mut dst[off..off + 32]), [u8; 32], fill_256);
            x += 16;
        }
        while x + 8 <= width {
            let off = row_off + x * 2;
            storeu_128!((&mut dst[off..off + 16]), [u8; 16], _mm256_castsi256_si128(fill_256));
            x += 8;
        }
        while x < width {
            let off = row_off + x * 2;
            dst[off..off + 2].copy_from_slice(&avg.to_ne_bytes());
            x += 1;
        }
    }
}

/// DC_LEFT prediction for 16bpc using AVX-512
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_dc_left_16bpc_avx512_inner(
    _token: Server64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    topleft: &[u8],
    tl_off: usize,
    width: usize,
    height: usize,
) {
    let mut dst = dst.flex_mut();
    let topleft = topleft.flex();
    let mut sum = 0u32;
    for i in 1..=height {
        let off = tl_off - i * 2;
        sum += u16::from_ne_bytes(topleft[off..off + 2].try_into().unwrap()) as u32;
    }
    let avg = ((sum + height as u32 / 2) / height as u32) as u16;

    let fill_512 = _mm512_set1_epi16(avg as i16);
    let fill_256 = _mm256_set1_epi16(avg as i16);

    for y in 0..height {
        let row_off = (dst_base as isize + y as isize * stride) as usize;
        let mut x = 0usize;
        while x + 32 <= width {
            let off = row_off + x * 2;
            storeu_512!((&mut dst[off..off + 64]), [u8; 64], fill_512);
            x += 32;
        }
        while x + 16 <= width {
            let off = row_off + x * 2;
            storeu_256!((&mut dst[off..off + 32]), [u8; 32], fill_256);
            x += 16;
        }
        while x + 8 <= width {
            let off = row_off + x * 2;
            storeu_128!((&mut dst[off..off + 16]), [u8; 16], _mm256_castsi256_si128(fill_256));
            x += 8;
        }
        while x < width {
            let off = row_off + x * 2;
            dst[off..off + 2].copy_from_slice(&avg.to_ne_bytes());
            x += 1;
        }
    }
}

/// DC prediction for 16bpc: average of top and left edge pixels
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_dc_16bpc_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    topleft: &[u8],
    tl_off: usize,
    width: usize,
    height: usize,
) {
    let mut dst = dst.flex_mut();
    let topleft = topleft.flex();
    // Calculate average of top row and left column
    let mut sum = 0u32;

    // Sum top row: tl[1..=width] in pixel units = tl_off + 2..tl_off + 2 + width*2 in bytes
    for i in 1..=width {
        let off = tl_off + i * 2;
        sum += u16::from_ne_bytes(topleft[off..off + 2].try_into().unwrap()) as u32;
    }

    // Sum left column: tl[-1..-height] in pixel units
    for i in 1..=height {
        let off = tl_off - i * 2;
        sum += u16::from_ne_bytes(topleft[off..off + 2].try_into().unwrap()) as u32;
    }

    // Average with rounding
    let count = (width + height) as u32;
    let avg = ((sum + count / 2) / count) as u16;

    let fill_val = _mm256_set1_epi16(avg as i16);

    for y in 0..height {
        let row_off = (dst_base as isize + y as isize * stride) as usize;
        let mut x = 0usize;

        while x + 16 <= width {
            let off = row_off + x * 2;
            storeu_256!((&mut dst[off..off + 32]), [u8; 32], fill_val);
            x += 16;
        }

        while x + 8 <= width {
            let off = row_off + x * 2;
            storeu_128!(
                (&mut dst[off..off + 16]),
                [u8; 16],
                _mm256_castsi256_si128(fill_val)
            );
            x += 8;
        }

        while x < width {
            let off = row_off + x * 2;
            dst[off..off + 2].copy_from_slice(&avg.to_ne_bytes());
            x += 1;
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn ipred_dc_16bpc_avx2(
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
    let token = unsafe { Desktop64::forge_token_dangerously() };
    let buf_len = compute_ipred_buf_len(stride as isize, width as usize * 2, height as usize);
    let dst_sl = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_len) };
    let (tl_sl, tl_off) = compute_topleft_slice(
        topleft as *const u8,
        width as usize * 2,
        height as usize * 2,
    );
    ipred_dc_16bpc_inner(
        token,
        dst_sl,
        0,
        stride as isize,
        tl_sl,
        tl_off,
        width as usize,
        height as usize,
    );
}

/// DC_TOP prediction for 16bpc: average of top edge pixels
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_dc_top_16bpc_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    topleft: &[u8],
    tl_off: usize,
    width: usize,
    height: usize,
) {
    let mut dst = dst.flex_mut();
    let topleft = topleft.flex();
    // Calculate average of top row
    let mut sum = 0u32;
    for i in 1..=width {
        let off = tl_off + i * 2;
        sum += u16::from_ne_bytes(topleft[off..off + 2].try_into().unwrap()) as u32;
    }
    let avg = ((sum + width as u32 / 2) / width as u32) as u16;

    let fill_val = _mm256_set1_epi16(avg as i16);

    for y in 0..height {
        let row_off = (dst_base as isize + y as isize * stride) as usize;
        let mut x = 0usize;

        while x + 16 <= width {
            let off = row_off + x * 2;
            storeu_256!((&mut dst[off..off + 32]), [u8; 32], fill_val);
            x += 16;
        }

        while x + 8 <= width {
            let off = row_off + x * 2;
            storeu_128!(
                (&mut dst[off..off + 16]),
                [u8; 16],
                _mm256_castsi256_si128(fill_val)
            );
            x += 8;
        }

        while x < width {
            let off = row_off + x * 2;
            dst[off..off + 2].copy_from_slice(&avg.to_ne_bytes());
            x += 1;
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn ipred_dc_top_16bpc_avx2(
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
    let token = unsafe { Desktop64::forge_token_dangerously() };
    let buf_len = compute_ipred_buf_len(stride as isize, width as usize * 2, height as usize);
    let dst_sl = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_len) };
    let (tl_sl, tl_off) = compute_topleft_slice(
        topleft as *const u8,
        width as usize * 2,
        height as usize * 2,
    );
    ipred_dc_top_16bpc_inner(
        token,
        dst_sl,
        0,
        stride as isize,
        tl_sl,
        tl_off,
        width as usize,
        height as usize,
    );
}

/// DC_LEFT prediction for 16bpc: average of left edge pixels
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_dc_left_16bpc_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    topleft: &[u8],
    tl_off: usize,
    width: usize,
    height: usize,
) {
    let mut dst = dst.flex_mut();
    let topleft = topleft.flex();
    // Calculate average of left column
    let mut sum = 0u32;
    for i in 1..=height {
        let off = tl_off - i * 2;
        sum += u16::from_ne_bytes(topleft[off..off + 2].try_into().unwrap()) as u32;
    }
    let avg = ((sum + height as u32 / 2) / height as u32) as u16;

    let fill_val = _mm256_set1_epi16(avg as i16);

    for y in 0..height {
        let row_off = (dst_base as isize + y as isize * stride) as usize;
        let mut x = 0usize;

        while x + 16 <= width {
            let off = row_off + x * 2;
            storeu_256!((&mut dst[off..off + 32]), [u8; 32], fill_val);
            x += 16;
        }

        while x + 8 <= width {
            let off = row_off + x * 2;
            storeu_128!(
                (&mut dst[off..off + 16]),
                [u8; 16],
                _mm256_castsi256_si128(fill_val)
            );
            x += 8;
        }

        while x < width {
            let off = row_off + x * 2;
            dst[off..off + 2].copy_from_slice(&avg.to_ne_bytes());
            x += 1;
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn ipred_dc_left_16bpc_avx2(
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
    let token = unsafe { Desktop64::forge_token_dangerously() };
    let buf_len = compute_ipred_buf_len(stride as isize, width as usize * 2, height as usize);
    let dst_sl = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_len) };
    let (tl_sl, tl_off) = compute_topleft_slice(
        topleft as *const u8,
        width as usize * 2,
        height as usize * 2,
    );
    ipred_dc_left_16bpc_inner(
        token,
        dst_sl,
        0,
        stride as isize,
        tl_sl,
        tl_off,
        width as usize,
        height as usize,
    );
}

// ============================================================================
// PAETH/SMOOTH 16bpc AVX-512
// ============================================================================

/// PAETH prediction 16bpc using AVX-512 — 16 pixels/iter with mask blending.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_paeth_16bpc_avx512_inner(
    _token: Server64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    topleft: &[u8],
    tl_off: usize,
    width: usize,
    height: usize,
) {
    let mut dst = dst.flex_mut();
    let topleft = topleft.flex();
    let topleft_val = u16::from_ne_bytes(topleft[tl_off..tl_off + 2].try_into().unwrap()) as i32;
    let topleft_vec = _mm512_set1_epi32(topleft_val);

    for y in 0..height {
        let row_off = (dst_base as isize + y as isize * stride) as usize;
        let left_byte_off = tl_off - (y + 1) * 2;
        let left_val = u16::from_ne_bytes(
            topleft[left_byte_off..left_byte_off + 2].try_into().unwrap(),
        ) as i32;
        let left_vec = _mm512_set1_epi32(left_val);

        let mut x = 0;
        while x + 16 <= width {
            let top_byte_off = tl_off + (x + 1) * 2;
            let top_u16 = loadu_256!(
                &topleft[top_byte_off..top_byte_off + 32],
                [u8; 32]
            );
            let top = _mm512_cvtepu16_epi32(top_u16);

            let base = _mm512_sub_epi32(_mm512_add_epi32(left_vec, top), topleft_vec);
            let ldiff = _mm512_abs_epi32(_mm512_sub_epi32(left_vec, base));
            let tdiff = _mm512_abs_epi32(_mm512_sub_epi32(top, base));
            let tldiff = _mm512_abs_epi32(_mm512_sub_epi32(topleft_vec, base));

            let ld_le_td = !_mm512_cmpgt_epi32_mask(ldiff, tdiff);
            let ld_le_tld = !_mm512_cmpgt_epi32_mask(ldiff, tldiff);
            let td_le_tld = !_mm512_cmpgt_epi32_mask(tdiff, tldiff);

            let use_left = ld_le_td & ld_le_tld;
            let use_top = !use_left & td_le_tld;

            let result = _mm512_mask_blend_epi32(
                use_left,
                _mm512_mask_blend_epi32(use_top, topleft_vec, top),
                left_vec,
            );

            // Pack i32→u16 (values are 0..bitdepth_max, unsigned saturation is fine)
            let clamped = _mm512_max_epi32(result, _mm512_setzero_si512());
            let result_u16: __m256i = _mm512_cvtusepi32_epi16(clamped);
            let off = row_off + x * 2;
            storeu_256!(&mut dst[off..off + 32], [u8; 32], result_u16);

            x += 16;
        }

        // Scalar fallback
        while x < width {
            let top_byte_off = tl_off + (x + 1) * 2;
            let top_val =
                u16::from_ne_bytes(topleft[top_byte_off..top_byte_off + 2].try_into().unwrap())
                    as i32;
            let base = left_val + top_val - topleft_val;
            let l_diff = (left_val - base).abs();
            let t_diff = (top_val - base).abs();
            let tl_diff = (topleft_val - base).abs();
            let pred = if l_diff <= t_diff && l_diff <= tl_diff {
                left_val
            } else if t_diff <= tl_diff {
                top_val
            } else {
                topleft_val
            };
            let off = row_off + x * 2;
            dst[off..off + 2].copy_from_slice(&(pred as u16).to_ne_bytes());
            x += 1;
        }
    }
}

/// SMOOTH prediction 16bpc using AVX-512 — 16 pixels/iter.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_smooth_16bpc_avx512_inner(
    _token: Server64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    topleft: &[u8],
    tl_off: usize,
    width: usize,
    height: usize,
) {
    let mut dst = dst.flex_mut();
    let topleft = topleft.flex();
    let weights_hor = &dav1d_sm_weights[width..][..width];
    let weights_ver = &dav1d_sm_weights[height..][..height];
    let right_off = tl_off + width * 2;
    let right_val =
        u16::from_ne_bytes(topleft[right_off..right_off + 2].try_into().unwrap()) as i32;
    let bottom_off = tl_off - height * 2;
    let bottom_val =
        u16::from_ne_bytes(topleft[bottom_off..bottom_off + 2].try_into().unwrap()) as i32;
    let right_vec = _mm512_set1_epi32(right_val);
    let bottom_vec = _mm512_set1_epi32(bottom_val);
    let rounding = _mm512_set1_epi32(256);
    let c256 = _mm512_set1_epi32(256);
    let zero_512 = _mm512_setzero_si512();

    for y in 0..height {
        let row_off = (dst_base as isize + y as isize * stride) as usize;
        let left_byte_off = tl_off - (y + 1) * 2;
        let left_val = u16::from_ne_bytes(
            topleft[left_byte_off..left_byte_off + 2].try_into().unwrap(),
        ) as i32;
        let left_vec = _mm512_set1_epi32(left_val);
        let w_v = weights_ver[y] as i32;
        let w_v_vec = _mm512_set1_epi32(w_v);
        let w_v_inv = _mm512_sub_epi32(c256, w_v_vec);

        let mut x = 0;
        while x + 16 <= width {
            let top_byte_off = tl_off + (x + 1) * 2;
            let top_u16 = loadu_256!(&topleft[top_byte_off..top_byte_off + 32], [u8; 32]);
            let top = _mm512_cvtepu16_epi32(top_u16);

            let wh_bytes = loadu_128!(&weights_hor[x..x + 16], [u8; 16]);
            let w_h = _mm512_cvtepu8_epi32(wh_bytes);
            let w_h_inv = _mm512_sub_epi32(c256, w_h);

            let vert = _mm512_add_epi32(
                _mm512_mullo_epi32(w_v_vec, top),
                _mm512_mullo_epi32(w_v_inv, bottom_vec),
            );
            let hor = _mm512_add_epi32(
                _mm512_mullo_epi32(w_h, left_vec),
                _mm512_mullo_epi32(w_h_inv, right_vec),
            );

            let pred = _mm512_add_epi32(vert, hor);
            let result = _mm512_srai_epi32::<9>(_mm512_add_epi32(pred, rounding));

            let clamped = _mm512_max_epi32(result, zero_512);
            let result_u16: __m256i = _mm512_cvtusepi32_epi16(clamped);
            let off = row_off + x * 2;
            storeu_256!(&mut dst[off..off + 32], [u8; 32], result_u16);

            x += 16;
        }

        while x < width {
            let top_byte_off = tl_off + (1 + x) * 2;
            let top_val =
                u16::from_ne_bytes(topleft[top_byte_off..top_byte_off + 2].try_into().unwrap())
                    as i32;
            let w_h = weights_hor[x] as i32;
            let pred =
                w_v * top_val + (256 - w_v) * bottom_val + w_h * left_val + (256 - w_h) * right_val;
            let off = row_off + x * 2;
            dst[off..off + 2].copy_from_slice(&(((pred + 256) >> 9) as u16).to_ne_bytes());
            x += 1;
        }
    }
}

/// SMOOTH_V prediction 16bpc using AVX-512 — 16 pixels/iter.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_smooth_v_16bpc_avx512_inner(
    _token: Server64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    topleft: &[u8],
    tl_off: usize,
    width: usize,
    height: usize,
) {
    let mut dst = dst.flex_mut();
    let topleft = topleft.flex();
    let weights_ver = &dav1d_sm_weights[height..][..height];
    let bottom_off = tl_off - height * 2;
    let bottom_val =
        u16::from_ne_bytes(topleft[bottom_off..bottom_off + 2].try_into().unwrap()) as i32;
    let bottom_vec = _mm512_set1_epi32(bottom_val);
    let rounding = _mm512_set1_epi32(128);
    let c256 = _mm512_set1_epi32(256);
    let zero_512 = _mm512_setzero_si512();

    for y in 0..height {
        let row_off = (dst_base as isize + y as isize * stride) as usize;
        let w_v = weights_ver[y] as i32;
        let w_v_vec = _mm512_set1_epi32(w_v);
        let w_v_inv = _mm512_sub_epi32(c256, w_v_vec);

        let mut x = 0;
        while x + 16 <= width {
            let top_byte_off = tl_off + (x + 1) * 2;
            let top_u16 = loadu_256!(&topleft[top_byte_off..top_byte_off + 32], [u8; 32]);
            let top = _mm512_cvtepu16_epi32(top_u16);

            let pred = _mm512_add_epi32(
                _mm512_mullo_epi32(w_v_vec, top),
                _mm512_mullo_epi32(w_v_inv, bottom_vec),
            );
            let result = _mm512_srai_epi32::<8>(_mm512_add_epi32(pred, rounding));

            let clamped = _mm512_max_epi32(result, zero_512);
            let result_u16: __m256i = _mm512_cvtusepi32_epi16(clamped);
            let off = row_off + x * 2;
            storeu_256!(&mut dst[off..off + 32], [u8; 32], result_u16);

            x += 16;
        }

        while x < width {
            let top_byte_off = tl_off + (1 + x) * 2;
            let top_val =
                u16::from_ne_bytes(topleft[top_byte_off..top_byte_off + 2].try_into().unwrap())
                    as i32;
            let pred = (w_v * top_val + (256 - w_v) * bottom_val + 128) >> 8;
            let off = row_off + x * 2;
            dst[off..off + 2].copy_from_slice(&(pred as u16).to_ne_bytes());
            x += 1;
        }
    }
}

/// SMOOTH_H prediction 16bpc using AVX-512 — 16 pixels/iter.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_smooth_h_16bpc_avx512_inner(
    _token: Server64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    topleft: &[u8],
    tl_off: usize,
    width: usize,
    height: usize,
) {
    let mut dst = dst.flex_mut();
    let topleft = topleft.flex();
    let weights_hor = &dav1d_sm_weights[width..][..width];
    let right_off = tl_off + width * 2;
    let right_val =
        u16::from_ne_bytes(topleft[right_off..right_off + 2].try_into().unwrap()) as i32;
    let right_vec = _mm512_set1_epi32(right_val);
    let rounding = _mm512_set1_epi32(128);
    let c256 = _mm512_set1_epi32(256);
    let zero_512 = _mm512_setzero_si512();

    for y in 0..height {
        let row_off = (dst_base as isize + y as isize * stride) as usize;
        let left_byte_off = tl_off - (y + 1) * 2;
        let left_val = u16::from_ne_bytes(
            topleft[left_byte_off..left_byte_off + 2].try_into().unwrap(),
        ) as i32;
        let left_vec = _mm512_set1_epi32(left_val);

        let mut x = 0;
        while x + 16 <= width {
            let wh_bytes = loadu_128!(&weights_hor[x..x + 16], [u8; 16]);
            let w_h = _mm512_cvtepu8_epi32(wh_bytes);
            let w_h_inv = _mm512_sub_epi32(c256, w_h);

            let pred = _mm512_add_epi32(
                _mm512_mullo_epi32(w_h, left_vec),
                _mm512_mullo_epi32(w_h_inv, right_vec),
            );
            let result = _mm512_srai_epi32::<8>(_mm512_add_epi32(pred, rounding));

            let clamped = _mm512_max_epi32(result, zero_512);
            let result_u16: __m256i = _mm512_cvtusepi32_epi16(clamped);
            let off = row_off + x * 2;
            storeu_256!(&mut dst[off..off + 32], [u8; 32], result_u16);

            x += 16;
        }

        while x < width {
            let w_h = weights_hor[x] as i32;
            let pred = (w_h * left_val + (256 - w_h) * right_val + 128) >> 8;
            let off = row_off + x * 2;
            dst[off..off + 2].copy_from_slice(&(pred as u16).to_ne_bytes());
            x += 1;
        }
    }
}

/// PAETH prediction for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_paeth_16bpc_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    topleft: &[u8],
    tl_off: usize,
    width: usize,
    height: usize,
) {
    let mut dst = dst.flex_mut();
    let topleft = topleft.flex();
    let topleft_val = u16::from_ne_bytes(topleft[tl_off..tl_off + 2].try_into().unwrap()) as i32;

    for y in 0..height {
        let row_off = (dst_base as isize + y as isize * stride) as usize;
        let left_byte_off = tl_off - (y + 1) * 2;
        let left_val = u16::from_ne_bytes(
            topleft[left_byte_off..left_byte_off + 2]
                .try_into()
                .unwrap(),
        ) as i32;

        // Process each pixel - PAETH is complex so use scalar
        for x in 0..width {
            let top_byte_off = tl_off + (x + 1) * 2;
            let top_val =
                u16::from_ne_bytes(topleft[top_byte_off..top_byte_off + 2].try_into().unwrap())
                    as i32;

            // PAETH: pick closest of left, top, topleft to (left + top - topleft)
            let base = left_val + top_val - topleft_val;
            let l_diff = (left_val - base).abs();
            let t_diff = (top_val - base).abs();
            let tl_diff = (topleft_val - base).abs();

            let pred = if l_diff <= t_diff && l_diff <= tl_diff {
                left_val
            } else if t_diff <= tl_diff {
                top_val
            } else {
                topleft_val
            };

            let off = row_off + x * 2;
            dst[off..off + 2].copy_from_slice(&(pred as u16).to_ne_bytes());
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn ipred_paeth_16bpc_avx2(
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
    let token = unsafe { Desktop64::forge_token_dangerously() };
    let buf_len = compute_ipred_buf_len(stride as isize, width as usize * 2, height as usize);
    let dst_sl = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_len) };
    let (tl_sl, tl_off) = compute_topleft_slice(
        topleft as *const u8,
        width as usize * 2,
        height as usize * 2,
    );
    ipred_paeth_16bpc_inner(
        token,
        dst_sl,
        0,
        stride as isize,
        tl_sl,
        tl_off,
        width as usize,
        height as usize,
    );
}

/// SMOOTH prediction for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_smooth_16bpc_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    topleft: &[u8],
    tl_off: usize,
    width: usize,
    height: usize,
) {
    let mut dst = dst.flex_mut();
    let topleft = topleft.flex();
    let weights_hor = &dav1d_sm_weights[width..][..width];
    let weights_ver = &dav1d_sm_weights[height..][..height];
    let right_off = tl_off + width * 2;
    let right_val =
        u16::from_ne_bytes(topleft[right_off..right_off + 2].try_into().unwrap()) as i32;
    let bottom_off = tl_off - height * 2;
    let bottom_val =
        u16::from_ne_bytes(topleft[bottom_off..bottom_off + 2].try_into().unwrap()) as i32;

    for y in 0..height {
        let row_off = (dst_base as isize + y as isize * stride) as usize;
        let left_byte_off = tl_off - (y + 1) * 2;
        let left_val = u16::from_ne_bytes(
            topleft[left_byte_off..left_byte_off + 2]
                .try_into()
                .unwrap(),
        ) as i32;
        let w_v = weights_ver[y] as i32;

        for x in 0..width {
            let top_byte_off = tl_off + (1 + x) * 2;
            let top_val =
                u16::from_ne_bytes(topleft[top_byte_off..top_byte_off + 2].try_into().unwrap())
                    as i32;
            let w_h = weights_hor[x] as i32;

            // Vertical component: w_v * top + (256 - w_v) * bottom
            let vert = w_v * top_val + (256 - w_v) * bottom_val;
            // Horizontal component: w_h * left + (256 - w_h) * right
            let horz = w_h * left_val + (256 - w_h) * right_val;
            // Combine with rounding
            let pred = (vert + horz + 256) >> 9;
            let off = row_off + x * 2;
            dst[off..off + 2].copy_from_slice(&(pred as u16).to_ne_bytes());
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn ipred_smooth_16bpc_avx2(
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
    let token = unsafe { Desktop64::forge_token_dangerously() };
    let buf_len = compute_ipred_buf_len(stride as isize, width as usize * 2, height as usize);
    let dst_sl = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_len) };
    let (tl_sl, tl_off) = compute_topleft_slice(
        topleft as *const u8,
        width as usize * 2,
        height as usize * 2,
    );
    ipred_smooth_16bpc_inner(
        token,
        dst_sl,
        0,
        stride as isize,
        tl_sl,
        tl_off,
        width as usize,
        height as usize,
    );
}

/// SMOOTH_V prediction for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_smooth_v_16bpc_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    topleft: &[u8],
    tl_off: usize,
    width: usize,
    height: usize,
) {
    let mut dst = dst.flex_mut();
    let topleft = topleft.flex();
    let weights_ver = &dav1d_sm_weights[height..][..height];
    let bottom_off = tl_off - height * 2;
    let bottom_val =
        u16::from_ne_bytes(topleft[bottom_off..bottom_off + 2].try_into().unwrap()) as i32;

    for y in 0..height {
        let row_off = (dst_base as isize + y as isize * stride) as usize;
        let w_v = weights_ver[y] as i32;

        for x in 0..width {
            let top_byte_off = tl_off + (1 + x) * 2;
            let top_val =
                u16::from_ne_bytes(topleft[top_byte_off..top_byte_off + 2].try_into().unwrap())
                    as i32;
            let pred = (w_v * top_val + (256 - w_v) * bottom_val + 128) >> 8;
            let off = row_off + x * 2;
            dst[off..off + 2].copy_from_slice(&(pred as u16).to_ne_bytes());
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn ipred_smooth_v_16bpc_avx2(
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
    let token = unsafe { Desktop64::forge_token_dangerously() };
    let buf_len = compute_ipred_buf_len(stride as isize, width as usize * 2, height as usize);
    let dst_sl = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_len) };
    let (tl_sl, tl_off) = compute_topleft_slice(
        topleft as *const u8,
        width as usize * 2,
        height as usize * 2,
    );
    ipred_smooth_v_16bpc_inner(
        token,
        dst_sl,
        0,
        stride as isize,
        tl_sl,
        tl_off,
        width as usize,
        height as usize,
    );
}

/// SMOOTH_H prediction for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_smooth_h_16bpc_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    topleft: &[u8],
    tl_off: usize,
    width: usize,
    height: usize,
) {
    let mut dst = dst.flex_mut();
    let topleft = topleft.flex();
    let weights_hor = &dav1d_sm_weights[width..][..width];
    let right_off = tl_off + width * 2;
    let right_val =
        u16::from_ne_bytes(topleft[right_off..right_off + 2].try_into().unwrap()) as i32;

    for y in 0..height {
        let row_off = (dst_base as isize + y as isize * stride) as usize;
        let left_byte_off = tl_off - (y + 1) * 2;
        let left_val = u16::from_ne_bytes(
            topleft[left_byte_off..left_byte_off + 2]
                .try_into()
                .unwrap(),
        ) as i32;

        for x in 0..width {
            let w_h = weights_hor[x] as i32;
            let pred = (w_h * left_val + (256 - w_h) * right_val + 128) >> 8;
            let off = row_off + x * 2;
            dst[off..off + 2].copy_from_slice(&(pred as u16).to_ne_bytes());
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn ipred_smooth_h_16bpc_avx2(
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
    let token = unsafe { Desktop64::forge_token_dangerously() };
    let buf_len = compute_ipred_buf_len(stride as isize, width as usize * 2, height as usize);
    let dst_sl = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_len) };
    let (tl_sl, tl_off) = compute_topleft_slice(
        topleft as *const u8,
        width as usize * 2,
        height as usize * 2,
    );
    ipred_smooth_h_16bpc_inner(
        token,
        dst_sl,
        0,
        stride as isize,
        tl_sl,
        tl_off,
        width as usize,
        height as usize,
    );
}

// ============================================================================
// Z1 Prediction 16bpc (angular prediction for angles < 90)
// ============================================================================

/// Z1 prediction for 16bpc: directional prediction using top edge only (angles < 90°)
/// Builds preprocessed top edge array internally, handles all cases.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_z1_16bpc_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    topleft: &[u8],
    tl_off: usize,
    width: usize,
    height: usize,
    angle: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let width_i = width as i32;
    let height_i = height as i32;

    // Extract angle flags
    let is_sm = (angle >> 9) & 1 != 0;
    let enable_intra_edge_filter = (angle >> 10) != 0;
    let angle = angle & 511;

    let mut dx = dav1d_dr_intra_derivative[(angle >> 1) as usize] as i32;

    // tl_off is in bytes, convert to pixel offset
    let tl_pix = tl_off / 2;

    // Helper: read u16 pixel from byte slice at pixel offset
    let rd = |off: usize| -> u16 {
        let b = off * 2;
        u16::from_ne_bytes(topleft[b..b + 2].try_into().unwrap())
    };

    let upsample_above = enable_intra_edge_filter
        && (90 - angle) < 40
        && (width_i + height_i) <= (16 >> is_sm as i32);

    // Build preprocessed top edge array as u16 pixels
    let mut top_px = [0u16; 64 + 64];
    let (max_base_x, base_inc);

    if upsample_above {
        let kernel: [i8; 4] = [-1, 9, 9, -1];
        let hsz = width_i + height_i;
        let in_off = tl_pix + 1;
        let from = -1i32;
        let to = width_i + std::cmp::min(width_i, height_i);
        for i in 0..hsz - 1 {
            top_px[(i * 2) as usize] =
                rd(in_off.wrapping_add_signed(i.clamp(from, to - 1) as isize));
            let mut s = 0i32;
            for j in 0..4i32 {
                s += rd(in_off.wrapping_add_signed((i + j - 1).clamp(from, to - 1) as isize))
                    as i32
                    * kernel[j as usize] as i32;
            }
            top_px[(i * 2 + 1) as usize] = ((s + 8) >> 4).clamp(0, bitdepth_max) as u16;
        }
        let i = hsz - 1;
        top_px[(i * 2) as usize] =
            rd(in_off.wrapping_add_signed(i.clamp(from, to - 1) as isize));
        dx <<= 1;
        max_base_x = (2 * (width_i + height_i) - 2) as usize;
        base_inc = 2usize;
    } else {
        let filter_strength = if enable_intra_edge_filter {
            get_filter_strength_simple(width_i + height_i, 90 - angle, is_sm)
        } else {
            0
        };
        if filter_strength != 0 {
            static KERNEL: [[u8; 5]; 3] = [[0, 4, 8, 4, 0], [0, 5, 6, 5, 0], [2, 4, 4, 4, 2]];
            let in_off = tl_pix + 1;
            let from = -1i32;
            let to = width_i + std::cmp::min(width_i, height_i);
            let lim_from = 0i32;
            let lim_to = width_i + height_i;
            let mut i = 0i32;
            while i < std::cmp::min(width_i + height_i, lim_from) {
                top_px[i as usize] =
                    rd(in_off.wrapping_add_signed(i.clamp(from, to - 1) as isize));
                i += 1;
            }
            while i < std::cmp::min(lim_to, width_i + height_i) {
                let mut s = 0i32;
                for j in 0..5i32 {
                    s += rd(in_off.wrapping_add_signed((i - 2 + j).clamp(from, to - 1) as isize))
                        as i32
                        * KERNEL[(filter_strength - 1) as usize][j as usize] as i32;
                }
                top_px[i as usize] = ((s + 8) >> 4) as u16;
                i += 1;
            }
            while i < width_i + height_i {
                top_px[i as usize] =
                    rd(in_off.wrapping_add_signed(i.clamp(from, to - 1) as isize));
                i += 1;
            }
            max_base_x = (width_i + height_i - 1) as usize;
        } else {
            // No preprocessing — copy top pixels directly
            for i in 0..width + std::cmp::min(width, height) {
                top_px[i] = rd(tl_pix + 1 + i);
            }
            max_base_x = width + std::cmp::min(width, height) - 1;
        }
        base_inc = 1;
    };

    // Convert top_px to bytes for SIMD access
    let top_bytes: &[u8] = zerocopy::AsBytes::as_bytes(&top_px[..]);
    let top_bytes = top_bytes.flex();

    let rounding = _mm256_set1_epi32(32);

    for y in 0..height_i {
        let xpos = (y + 1) * dx;
        let frac = (xpos & 0x3e) as i32;
        let inv_frac = 64 - frac;

        let frac_vec = _mm256_set1_epi32(frac);
        let inv_frac_vec = _mm256_set1_epi32(inv_frac);

        let row_off = (dst_base as isize + y as isize * stride) as usize;
        let base0 = (xpos >> 6) as usize;

        let mut x = 0usize;

        // SIMD: 8 pixels at a time (non-upsampled consecutive access)
        if base_inc == 1 {
            while x + 8 <= width && base0 + x + 8 < max_base_x {
                let base = base0 + x;

                let load0 = base * 2;
                let load1 = (base + 1) * 2;
                let t0 = loadu_128!((&top_bytes[load0..load0 + 16]), [u8; 16]);
                let t1 = loadu_128!((&top_bytes[load1..load1 + 16]), [u8; 16]);

                let t0_w = _mm256_cvtepu16_epi32(t0);
                let t1_w = _mm256_cvtepu16_epi32(t1);

                let prod0 = _mm256_mullo_epi32(t0_w, inv_frac_vec);
                let prod1 = _mm256_mullo_epi32(t1_w, frac_vec);
                let sum = _mm256_add_epi32(_mm256_add_epi32(prod0, prod1), rounding);
                let result = _mm256_srai_epi32::<6>(sum);

                let packed = _mm256_packus_epi32(result, result);
                let lo = _mm256_castsi256_si128(packed);
                let hi = _mm256_extracti128_si256::<1>(packed);
                let combined = _mm_unpacklo_epi64(lo, hi);
                let store_off = row_off + x * 2;
                storeu_128!((&mut dst[store_off..store_off + 16]), [u8; 16], combined);

                x += 8;
            }
        }

        // Scalar remainder (also handles upsampled stride-2 access)
        while x < width {
            let base = base0 + base_inc * x;
            if base < max_base_x {
                let t0 = top_px[base] as i32;
                let t1 = top_px[base + 1] as i32;
                let v = t0 * inv_frac + t1 * frac;
                let off = row_off + x * 2;
                dst[off..off + 2].copy_from_slice(&(((v + 32) >> 6) as u16).to_ne_bytes());
            } else {
                let fill_val = top_px[max_base_x];
                for xx in x..width {
                    let off = row_off + xx * 2;
                    dst[off..off + 2].copy_from_slice(&fill_val.to_ne_bytes());
                }
                break;
            }
            x += 1;
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn ipred_z1_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    topleft: *const DynPixel,
    width: c_int,
    height: c_int,
    angle: c_int,
    _max_width: c_int,
    _max_height: c_int,
    _bitdepth_max: c_int,
    _topleft_off: usize,
    _dst: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    let buf_len = compute_ipred_buf_len(stride as isize, width as usize * 2, height as usize);
    let dst_sl = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_len) };
    let (tl_sl, tl_off) = compute_topleft_slice(
        topleft as *const u8,
        width as usize * 2,
        height as usize * 2,
    );
    ipred_z1_16bpc_inner(
        token,
        dst_sl,
        0,
        stride as isize,
        tl_sl,
        tl_off,
        width as usize,
        height as usize,
        angle as i32,
        _bitdepth_max as i32,
    );
}

// ============================================================================
// Z2 Prediction 16bpc (angular prediction for angles 90-180) { return false; }
// ============================================================================

/// Z2 intra prediction SIMD inner for 16bpc.
/// Builds preprocessed edge array internally, handles all cases.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_z2_16bpc_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    topleft: &[u8],
    tl_off: usize,
    width: usize,
    height: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let width_i = width as i32;
    let height_i = height as i32;

    // Extract angle flags
    let is_sm = (angle >> 9) & 1 != 0;
    let enable_intra_edge_filter = (angle >> 10) != 0;
    let angle = angle & 511;

    let mut dy = dav1d_dr_intra_derivative[((angle - 90) >> 1) as usize] as i32;
    let mut dx = dav1d_dr_intra_derivative[((180 - angle) >> 1) as usize] as i32;

    let upsample_left = enable_intra_edge_filter
        && (180 - angle) < 40
        && (width_i + height_i) <= (16 >> is_sm as i32);
    let upsample_above = enable_intra_edge_filter
        && (angle - 90) < 40
        && (width_i + height_i) <= (16 >> is_sm as i32);

    // Build preprocessed edge array as u16 pixels
    let mut edge_px = [0u16; 64 + 64 + 1];
    let edge_tl = 64usize;

    // Helper: read u16 pixel from byte slice at pixel offset
    let rd = |off: usize| -> u16 {
        let b = off * 2;
        u16::from_ne_bytes(topleft[b..b + 2].try_into().unwrap())
    };
    // tl_off is in bytes, convert to pixel offset
    let tl_pix = tl_off / 2;

    // Top edge preprocessing
    if upsample_above {
        let kernel: [i8; 4] = [-1, 9, 9, -1];
        let hsz = width_i + 1;
        let in_off = tl_pix;
        for i in 0..hsz - 1 {
            edge_px[edge_tl + (i * 2) as usize] = rd(in_off + i.clamp(0, hsz - 1) as usize);
            let mut s = 0i32;
            for j in 0..4i32 {
                s += rd(in_off + (i + j - 1).clamp(0, hsz - 1) as usize) as i32
                    * kernel[j as usize] as i32;
            }
            edge_px[edge_tl + (i * 2 + 1) as usize] = ((s + 8) >> 4).clamp(0, bitdepth_max) as u16;
        }
        let i = hsz - 1;
        edge_px[edge_tl + (i * 2) as usize] = rd(in_off + i.clamp(0, hsz - 1) as usize);
        dx <<= 1;
    } else {
        let filter_strength = if enable_intra_edge_filter {
            get_filter_strength_simple(width_i + height_i, angle - 90, is_sm)
        } else {
            0
        };
        if filter_strength != 0 {
            // Filtered top edge
            static KERNEL: [[u8; 5]; 3] = [[0, 4, 8, 4, 0], [0, 5, 6, 5, 0], [2, 4, 4, 4, 2]];
            let in_off = tl_pix + 1;
            let from = -1i32;
            let to = width_i;
            let lim_from = 0i32;
            let lim_to = max_width;
            let mut i = 0i32;
            while i < std::cmp::min(width_i, lim_from) {
                edge_px[edge_tl + 1 + i as usize] =
                    rd(in_off.wrapping_add_signed(i.clamp(from, to - 1) as isize));
                i += 1;
            }
            while i < std::cmp::min(lim_to, width_i) {
                let mut s = 0i32;
                for j in 0..5i32 {
                    s += rd(in_off.wrapping_add_signed((i - 2 + j).clamp(from, to - 1) as isize))
                        as i32
                        * KERNEL[(filter_strength - 1) as usize][j as usize] as i32;
                }
                edge_px[edge_tl + 1 + i as usize] = ((s + 8) >> 4) as u16;
                i += 1;
            }
            while i < width_i {
                edge_px[edge_tl + 1 + i as usize] =
                    rd(in_off.wrapping_add_signed(i.clamp(from, to - 1) as isize));
                i += 1;
            }
        } else {
            for i in 0..width {
                edge_px[edge_tl + 1 + i] = rd(tl_pix + 1 + i);
            }
        }
    }

    // Left edge preprocessing
    if upsample_left {
        let kernel: [i8; 4] = [-1, 9, 9, -1];
        let hsz = height_i + 1;
        let in_off = tl_pix - height;
        for i in 0..hsz - 1 {
            edge_px[edge_tl - height * 2 + (i * 2) as usize] =
                rd(in_off + i.clamp(0, hsz - 1) as usize);
            let mut s = 0i32;
            for j in 0..4i32 {
                s += rd(in_off + (i + j - 1).clamp(0, hsz - 1) as usize) as i32
                    * kernel[j as usize] as i32;
            }
            edge_px[edge_tl - height * 2 + (i * 2 + 1) as usize] =
                ((s + 8) >> 4).clamp(0, bitdepth_max) as u16;
        }
        let i = hsz - 1;
        edge_px[edge_tl - height * 2 + (i * 2) as usize] =
            rd(in_off + i.clamp(0, hsz - 1) as usize);
        dy <<= 1;
    } else {
        let filter_strength = if enable_intra_edge_filter {
            get_filter_strength_simple(width_i + height_i, 180 - angle, is_sm)
        } else {
            0
        };
        if filter_strength != 0 {
            static KERNEL: [[u8; 5]; 3] = [[0, 4, 8, 4, 0], [0, 5, 6, 5, 0], [2, 4, 4, 4, 2]];
            let in_off = tl_pix - height;
            let from = 0i32;
            let to = height_i + 1;
            let lim_from = height_i - max_height;
            let lim_to = height_i;
            let mut i = 0i32;
            while i < std::cmp::min(height_i, lim_from) {
                edge_px[edge_tl - height + i as usize] =
                    rd(in_off.wrapping_add_signed(i.clamp(from, to - 1) as isize));
                i += 1;
            }
            while i < std::cmp::min(lim_to, height_i) {
                let mut s = 0i32;
                for j in 0..5i32 {
                    s += rd(in_off.wrapping_add_signed((i - 2 + j).clamp(from, to - 1) as isize))
                        as i32
                        * KERNEL[(filter_strength - 1) as usize][j as usize] as i32;
                }
                edge_px[edge_tl - height + i as usize] = ((s + 8) >> 4) as u16;
                i += 1;
            }
            while i < height_i {
                edge_px[edge_tl - height + i as usize] =
                    rd(in_off.wrapping_add_signed(i.clamp(from, to - 1) as isize));
                i += 1;
            }
        } else {
            for i in 0..height {
                edge_px[edge_tl - height + i] = rd(tl_pix - height + i);
            }
        }
    }

    // Corner pixel
    edge_px[edge_tl] = rd(tl_pix);

    // Convert to bytes for SIMD access
    let edge_bytes: &[u8] =
        zerocopy::AsBytes::as_bytes(edge_px.as_slice());
    let edge = edge_bytes.flex();

    let base_inc_x = 1 + upsample_above as usize;
    let left = edge_tl - (1 + upsample_left as usize);

    let rounding = _mm256_set1_epi32(32);

    for y in 0..height_i {
        let xpos = ((1 + upsample_above as i32) << 6) - dx * (y + 1);
        let base_x0 = xpos >> 6;
        let frac_x = (xpos & 0x3e) as i32;
        let inv_frac_x = 64 - frac_x;

        let row_off = (dst_base as isize + y as isize * stride) as usize;

        let left_count = if base_x0 >= 0 {
            0usize
        } else {
            let needed = (-base_x0) as usize;
            ((needed + base_inc_x - 1) / base_inc_x).min(width)
        };

        // First: process pixels using left edge
        let mut x = 0usize;
        while x < left_count {
            let ypos = ((y << (6 + upsample_left as i32))) - dy * (x as i32 + 1);
            let base_y = ypos >> 6;
            let frac_y = ypos & 0x3e;
            let inv_frac_y = 64 - frac_y;

            let l0_pix = left.wrapping_add_signed(-base_y as isize);
            let l1_pix = left.wrapping_add_signed(-(base_y + 1) as isize);
            let l0_off = l0_pix * 2;
            let l1_off = l1_pix * 2;
            let l0 = u16::from_ne_bytes(edge[l0_off..l0_off + 2].try_into().unwrap()) as i32;
            let l1 = u16::from_ne_bytes(edge[l1_off..l1_off + 2].try_into().unwrap()) as i32;
            let v = l0 * inv_frac_y + l1 * frac_y;
            let off = row_off + x * 2;
            dst[off..off + 2].copy_from_slice(&(((v + 32) >> 6) as u16).to_ne_bytes());
            x += 1;
        }

        // Then: process pixels using top edge
        if base_inc_x == 1 {
            while x + 8 <= width {
                let base_x = (base_x0 + x as i32) as usize;
                let load0 = (edge_tl + base_x) * 2;
                let load1 = (edge_tl + base_x + 1) * 2;
                if load1 + 16 > edge.len() {
                    break;
                }
                let t0 = loadu_128!((&edge[load0..load0 + 16]), [u8; 16]);
                let t1 = loadu_128!((&edge[load1..load1 + 16]), [u8; 16]);

                let t0_w = _mm256_cvtepu16_epi32(t0);
                let t1_w = _mm256_cvtepu16_epi32(t1);

                let frac_vec = _mm256_set1_epi32(frac_x);
                let inv_frac_vec = _mm256_set1_epi32(inv_frac_x);

                let prod0 = _mm256_mullo_epi32(t0_w, inv_frac_vec);
                let prod1 = _mm256_mullo_epi32(t1_w, frac_vec);
                let sum = _mm256_add_epi32(_mm256_add_epi32(prod0, prod1), rounding);
                let result = _mm256_srai_epi32::<6>(sum);

                let packed = _mm256_packus_epi32(result, result);
                let lo = _mm256_castsi256_si128(packed);
                let hi = _mm256_extracti128_si256::<1>(packed);
                let combined = _mm_unpacklo_epi64(lo, hi);
                let store_off = row_off + x * 2;
                storeu_128!((&mut dst[store_off..store_off + 16]), [u8; 16], combined);

                x += 8;
            }
        }

        // Scalar remainder
        while x < width {
            let base_x = (base_x0 + (base_inc_x * x) as i32) as usize;
            let t0_off = (edge_tl + base_x) * 2;
            let t1_off = (edge_tl + base_x + 1) * 2;
            if t1_off + 2 > edge.len() {
                break;
            }
            let t0 = u16::from_ne_bytes(edge[t0_off..t0_off + 2].try_into().unwrap()) as i32;
            let t1 = u16::from_ne_bytes(edge[t1_off..t1_off + 2].try_into().unwrap()) as i32;
            let v = t0 * inv_frac_x + t1 * frac_x;
            let off = row_off + x * 2;
            dst[off..off + 2].copy_from_slice(&(((v + 32) >> 6) as u16).to_ne_bytes());
            x += 1;
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn ipred_z2_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    topleft: *const DynPixel,
    width: c_int,
    height: c_int,
    angle: c_int,
    max_width: c_int,
    max_height: c_int,
    _bitdepth_max: c_int,
    _topleft_off: usize,
    _dst: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    let buf_len = compute_ipred_buf_len(stride as isize, width as usize * 2, height as usize);
    let dst_sl = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_len) };
    let (tl_sl, tl_off) = compute_topleft_slice(
        topleft as *const u8,
        width as usize * 2,
        height as usize * 2,
    );
    ipred_z2_16bpc_inner(
        token,
        dst_sl,
        0,
        stride as isize,
        tl_sl,
        tl_off,
        width as usize,
        height as usize,
        angle as i32,
        max_width as i32,
        max_height as i32,
        _bitdepth_max as i32,
    );
}

// ============================================================================
// Z3 Prediction 16bpc (angular prediction for angles > 180) { return false; }
// ============================================================================

/// Z3 prediction for 16bpc: directional prediction using left edge only (angles > 180°)
/// Builds preprocessed left edge array internally, handles all cases.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_z3_16bpc_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    topleft: &[u8],
    tl_off: usize,
    width: usize,
    height: usize,
    angle: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let width_i = width as i32;
    let height_i = height as i32;

    // Extract angle flags
    let is_sm = (angle >> 9) & 1 != 0;
    let enable_intra_edge_filter = (angle >> 10) != 0;
    let angle = angle & 511;

    let mut dy = dav1d_dr_intra_derivative[((270 - angle) >> 1) as usize] as usize;

    // tl_off is in bytes, convert to pixel offset
    let tl_pix = tl_off / 2;

    // Helper: read u16 pixel from byte slice at pixel offset
    let rd = |off: usize| -> u16 {
        let b = off * 2;
        u16::from_ne_bytes(topleft[b..b + 2].try_into().unwrap())
    };

    let upsample_left = enable_intra_edge_filter
        && (angle - 180) < 40
        && (width_i + height_i) <= (16 >> is_sm as i32);

    // Build preprocessed left edge array as u16 pixels
    // left_px[left_off - base] = pixel at distance `base` from corner
    let mut left_px = [0u16; 64 + 64];
    let (left_off, max_base_y, base_inc);
    let use_left_px; // whether to use left_px or topleft directly

    if upsample_left {
        let kernel: [i8; 4] = [-1, 9, 9, -1];
        let hsz = width_i + height_i;
        let in_off = tl_pix - (width + height);
        let from = std::cmp::max(width_i - height_i, 0);
        let to = width_i + height_i + 1;
        for i in 0..hsz - 1 {
            left_px[(i * 2) as usize] =
                rd(in_off.wrapping_add_signed(i.clamp(from, to - 1) as isize));
            let mut s = 0i32;
            for j in 0..4i32 {
                s += rd(in_off.wrapping_add_signed((i + j - 1).clamp(from, to - 1) as isize))
                    as i32
                    * kernel[j as usize] as i32;
            }
            left_px[(i * 2 + 1) as usize] = ((s + 8) >> 4).clamp(0, bitdepth_max) as u16;
        }
        let i = hsz - 1;
        left_px[(i * 2) as usize] =
            rd(in_off.wrapping_add_signed(i.clamp(from, to - 1) as isize));
        left_off = (2 * (width_i + height_i) - 2) as usize;
        max_base_y = left_off;
        dy <<= 1;
        base_inc = 2usize;
        use_left_px = true;
    } else {
        let filter_strength = if enable_intra_edge_filter {
            get_filter_strength_simple(width_i + height_i, angle - 180, is_sm)
        } else {
            0
        };
        if filter_strength != 0 {
            static KERNEL: [[u8; 5]; 3] = [[0, 4, 8, 4, 0], [0, 5, 6, 5, 0], [2, 4, 4, 4, 2]];
            let in_off = tl_pix - (width + height);
            let from = std::cmp::max(width_i - height_i, 0);
            let to = width_i + height_i + 1;
            let lim_from = 0i32;
            let lim_to = width_i + height_i;
            let mut i = 0i32;
            while i < std::cmp::min(width_i + height_i, lim_from) {
                left_px[i as usize] =
                    rd(in_off.wrapping_add_signed(i.clamp(from, to - 1) as isize));
                i += 1;
            }
            while i < std::cmp::min(lim_to, width_i + height_i) {
                let mut s = 0i32;
                for j in 0..5i32 {
                    s += rd(in_off.wrapping_add_signed((i - 2 + j).clamp(from, to - 1) as isize))
                        as i32
                        * KERNEL[(filter_strength - 1) as usize][j as usize] as i32;
                }
                left_px[i as usize] = ((s + 8) >> 4) as u16;
                i += 1;
            }
            while i < width_i + height_i {
                left_px[i as usize] =
                    rd(in_off.wrapping_add_signed(i.clamp(from, to - 1) as isize));
                i += 1;
            }
            left_off = (width_i + height_i - 1) as usize;
            max_base_y = left_off;
            use_left_px = true;
        } else {
            // No preprocessing — access topleft directly
            left_off = 0; // unused for direct topleft access
            max_base_y = height + std::cmp::min(width, height) - 1;
            use_left_px = false;
        }
        base_inc = 1;
    };

    // Column-major access pattern
    for x in 0..width {
        let ypos = dy * (x + 1);
        let frac = (ypos & 0x3e) as i32;
        let inv_frac = 64 - frac;

        for y in 0..height_i {
            let base = (ypos >> 6) + base_inc * y as usize;

            if base < max_base_y {
                let (l0, l1) = if use_left_px {
                    (
                        left_px[left_off - base] as i32,
                        left_px[left_off - base - 1] as i32,
                    )
                } else {
                    // Direct topleft: left[base] = tl[-(base+1)] in pixel units
                    (
                        rd(tl_pix - base - 1) as i32,
                        rd(tl_pix - base - 2) as i32,
                    )
                };
                let v = l0 * inv_frac + l1 * frac;
                let pixel_off = (dst_base as isize + y as isize * stride) as usize + x * 2;
                dst[pixel_off..pixel_off + 2]
                    .copy_from_slice(&(((v + 32) >> 6) as u16).to_ne_bytes());
            } else {
                let fill_val = if use_left_px {
                    left_px[left_off - max_base_y]
                } else {
                    rd(tl_pix - max_base_y - 1)
                };
                for yy in y..height_i {
                    let pixel_off = (dst_base as isize + yy as isize * stride) as usize + x * 2;
                    dst[pixel_off..pixel_off + 2].copy_from_slice(&fill_val.to_ne_bytes());
                }
                break;
            }
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn ipred_z3_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    topleft: *const DynPixel,
    width: c_int,
    height: c_int,
    angle: c_int,
    _max_width: c_int,
    _max_height: c_int,
    _bitdepth_max: c_int,
    _topleft_off: usize,
    _dst: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    let buf_len = compute_ipred_buf_len(stride as isize, width as usize * 2, height as usize);
    let dst_sl = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_len) };
    let (tl_sl, tl_off) = compute_topleft_slice(
        topleft as *const u8,
        width as usize * 2,
        height as usize * 2,
    );
    ipred_z3_16bpc_inner(
        token,
        dst_sl,
        0,
        stride as isize,
        tl_sl,
        tl_off,
        width as usize,
        height as usize,
        angle as i32,
        _bitdepth_max as i32,
    );
}

// ============================================================================
// FILTER Prediction 16bpc
// ============================================================================

/// FILTER prediction for 16bpc: uses 7-tap filter for intra prediction
///
/// Processes in 4x2 blocks. Each output pixel uses 7 input samples.
/// Input pixels: p0 = topleft, p1-p4 = top row (4 pixels), p5-p6 = left column (2 pixels) { return false; }
/// For 16bpc: out = (sum + 8) >> 4, clamped to [0, bitdepth_max]
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_filter_16bpc_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    topleft: &[u8],
    tl_off: usize,
    width: usize,
    height: usize,
    filt_idx: i32,
    bitdepth_max: i32,
    topleft_off: usize,
) {
    let mut dst = dst.flex_mut();
    let topleft = topleft.flex();
    let width = (width as usize / 4) * 4; // Round down to multiple of 4
    let filt_idx = (filt_idx as usize) & 511;

    let filter = &dav1d_filter_intra_taps[filt_idx];

    // Process in 4x2 blocks
    for y in (0..height).step_by(2) {
        let cur_tl_off = topleft_off - y;
        // tl_pixel = topleft at byte offset for pixel position cur_tl_off
        let tl_pixel_off = tl_off.wrapping_add(cur_tl_off * 2);
        let mut tl_pixel =
            u16::from_ne_bytes(topleft[tl_pixel_off..tl_pixel_off + 2].try_into().unwrap()) as i32;

        let row0_off = (dst_base as isize + y as isize * stride) as usize;
        let row1_off = (dst_base as isize + (y + 1) as isize * stride) as usize;

        for x in (0..width).step_by(4) {
            // Get top 4 pixels (p1-p4)
            // y=0: from topleft buffer; y>=2: from previously-written output row y-1
            let (p1, p2, p3, p4) = if y == 0 {
                let top_base = tl_off.wrapping_add((topleft_off + 1 + x) * 2);
                (
                    u16::from_ne_bytes(topleft[top_base..top_base + 2].try_into().unwrap()) as i32,
                    u16::from_ne_bytes(topleft[top_base + 2..top_base + 4].try_into().unwrap())
                        as i32,
                    u16::from_ne_bytes(topleft[top_base + 4..top_base + 6].try_into().unwrap())
                        as i32,
                    u16::from_ne_bytes(topleft[top_base + 6..top_base + 8].try_into().unwrap())
                        as i32,
                )
            } else {
                let top_row = (dst_base as isize + (y as isize - 1) * stride) as usize;
                let tb = top_row + x * 2;
                (
                    u16::from_ne_bytes(dst[tb..tb + 2].try_into().unwrap()) as i32,
                    u16::from_ne_bytes(dst[tb + 2..tb + 4].try_into().unwrap()) as i32,
                    u16::from_ne_bytes(dst[tb + 4..tb + 6].try_into().unwrap()) as i32,
                    u16::from_ne_bytes(dst[tb + 6..tb + 8].try_into().unwrap()) as i32,
                )
            };

            // Get left 2 pixels (p5, p6)
            let (p5, p6) = if x == 0 {
                // From original topleft buffer
                let left_base = tl_off.wrapping_add(cur_tl_off.wrapping_sub(1) * 2);
                let left_base2 = tl_off.wrapping_add(cur_tl_off.wrapping_sub(2) * 2);
                (
                    u16::from_ne_bytes(topleft[left_base..left_base + 2].try_into().unwrap())
                        as i32,
                    u16::from_ne_bytes(topleft[left_base2..left_base2 + 2].try_into().unwrap())
                        as i32,
                )
            } else {
                // From previously computed output
                let p5_off = row0_off + (x - 1) * 2;
                let p6_off = row1_off + (x - 1) * 2;
                (
                    u16::from_ne_bytes(dst[p5_off..p5_off + 2].try_into().unwrap()) as i32,
                    u16::from_ne_bytes(dst[p6_off..p6_off + 2].try_into().unwrap()) as i32,
                )
            };

            let p0 = tl_pixel;
            let p = [p0, p1, p2, p3, p4, p5, p6];

            // Process 4x2 = 8 output pixels using filter taps
            let flt = filter.as_slice();
            let mut flt_offset = 0;

            // Row 0 (4 pixels)
            for xx in 0..4 {
                let acc = filter_fn(&flt[flt_offset..], p);
                let val = ((acc + 8) >> 4).clamp(0, bitdepth_max as i32) as u16;
                let off = row0_off + (x + xx) * 2;
                dst[off..off + 2].copy_from_slice(&val.to_ne_bytes());
                flt_offset += FLT_INCR;
            }

            // Row 1 (4 pixels)
            for xx in 0..4 {
                let acc = filter_fn(&flt[flt_offset..], p);
                let val = ((acc + 8) >> 4).clamp(0, bitdepth_max as i32) as u16;
                let off = row1_off + (x + xx) * 2;
                dst[off..off + 2].copy_from_slice(&val.to_ne_bytes());
                flt_offset += FLT_INCR;
            }

            // Update topleft for next 4x2 block (16bpc)
            tl_pixel = p4;
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn ipred_filter_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    topleft: *const DynPixel,
    width: c_int,
    height: c_int,
    filt_idx: c_int,
    _max_width: c_int,
    _max_height: c_int,
    bitdepth_max: c_int,
    topleft_off: usize,
    _dst: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    let buf_len = compute_ipred_buf_len(stride as isize, width as usize * 2, height as usize);
    let dst_sl = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_len) };
    let (tl_sl, tl_off) = compute_topleft_slice(
        topleft as *const u8,
        width as usize * 2,
        height as usize * 2,
    );
    ipred_filter_16bpc_inner(
        token,
        dst_sl,
        0,
        stride as isize,
        tl_sl,
        tl_off,
        width as usize,
        height as usize,
        filt_idx as i32,
        bitdepth_max as i32,
        topleft_off,
    );
}

// ============================================================================
// Safe dispatch wrapper for x86_64 AVX2
// ============================================================================

use crate::include::common::bitdepth::BitDepth;
use crate::src::internal::SCRATCH_EDGE_LEN;
use crate::src::strided::Strided as _;

/// Safe dispatch for intra prediction. Returns true if SIMD was used.
#[cfg(target_arch = "x86_64")]
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
    use crate::include::common::bitdepth::BPC;
    use zerocopy::AsBytes;

    let Some(token) = crate::src::cpu::summon_avx2() else {
        return false;
    };

    // Try AVX-512 for modes that benefit from wider registers
    #[cfg(target_arch = "x86_64")]
    let avx512_token = crate::src::cpu::summon_avx512();
    #[cfg(not(target_arch = "x86_64"))]
    let avx512_token: Option<Server64> = None;

    let w = width as usize;
    let h = height as usize;
    let byte_stride = dst.stride();
    let bd_c = bd.into_c();

    // Create tracked guard for the dst pixel region
    let (mut dst_guard, dst_base) = dst.strided_slice_mut::<BD>(w, h);
    let dst_base_bytes = dst_base * std::mem::size_of::<BD::Pixel>();

    // Get byte-level views (safe via zerocopy AsBytes)
    let dst_bytes: &mut [u8] = dst_guard.as_bytes_mut();
    let tl_bytes: &[u8] = topleft.as_bytes();

    match (BD::BPC, mode) {
        (BPC::BPC8, 0) => {
            if let Some(t512) = avx512_token {
                ipred_dc_8bpc_avx512_inner(
                    t512,
                    dst_bytes,
                    dst_base_bytes,
                    byte_stride,
                    tl_bytes,
                    topleft_off,
                    w,
                    h,
                )
            } else {
                ipred_dc_8bpc_inner(
                    token,
                    dst_bytes,
                    dst_base_bytes,
                    byte_stride,
                    tl_bytes,
                    topleft_off,
                    w,
                    h,
                )
            }
        }
        (BPC::BPC8, 1) => {
            if let Some(t512) = avx512_token {
                ipred_v_8bpc_avx512_inner(
                    t512,
                    dst_bytes,
                    dst_base_bytes,
                    byte_stride,
                    tl_bytes,
                    topleft_off,
                    w,
                    h,
                )
            } else {
                ipred_v_8bpc_inner(
                    token,
                    dst_bytes,
                    dst_base_bytes,
                    byte_stride,
                    tl_bytes,
                    topleft_off,
                    w,
                    h,
                )
            }
        }
        (BPC::BPC8, 2) => {
            if let Some(t512) = avx512_token {
                ipred_h_8bpc_avx512_inner(
                    t512,
                    dst_bytes,
                    dst_base_bytes,
                    byte_stride,
                    tl_bytes,
                    topleft_off,
                    w,
                    h,
                )
            } else {
                ipred_h_8bpc_inner(
                    token,
                    dst_bytes,
                    dst_base_bytes,
                    byte_stride,
                    tl_bytes,
                    topleft_off,
                    w,
                    h,
                )
            }
        }
        (BPC::BPC8, 3) => {
            if let Some(t512) = avx512_token {
                ipred_dc_left_8bpc_avx512_inner(
                    t512,
                    dst_bytes,
                    dst_base_bytes,
                    byte_stride,
                    tl_bytes,
                    topleft_off,
                    w,
                    h,
                )
            } else {
                ipred_dc_left_8bpc_inner(
                    token,
                    dst_bytes,
                    dst_base_bytes,
                    byte_stride,
                    tl_bytes,
                    topleft_off,
                    w,
                    h,
                )
            }
        }
        (BPC::BPC8, 4) => {
            if let Some(t512) = avx512_token {
                ipred_dc_top_8bpc_avx512_inner(
                    t512,
                    dst_bytes,
                    dst_base_bytes,
                    byte_stride,
                    tl_bytes,
                    topleft_off,
                    w,
                    h,
                )
            } else {
                ipred_dc_top_8bpc_inner(
                    token,
                    dst_bytes,
                    dst_base_bytes,
                    byte_stride,
                    tl_bytes,
                    topleft_off,
                    w,
                    h,
                )
            }
        }
        (BPC::BPC8, 5) => {
            if let Some(t512) = avx512_token {
                ipred_dc_128_8bpc_avx512_inner(
                    t512,
                    dst_bytes,
                    dst_base_bytes,
                    byte_stride,
                    w,
                    h,
                )
            } else {
                ipred_dc_128_8bpc_inner(token, dst_bytes, dst_base_bytes, byte_stride, w, h)
            }
        }
        (BPC::BPC8, 6) => {
            ipred_z1_8bpc_inner(
                token,
                dst_bytes,
                dst_base_bytes,
                byte_stride,
                tl_bytes,
                topleft_off,
                w,
                h,
                angle as i32,
            );
        }
        (BPC::BPC8, 7) => {
            ipred_z2_8bpc_inner(
                token,
                dst_bytes,
                dst_base_bytes,
                byte_stride,
                tl_bytes,
                topleft_off,
                w,
                h,
                angle as i32,
                max_width,
                max_height,
            );
        }
        (BPC::BPC8, 8) => {
            ipred_z3_8bpc_inner(
                token,
                dst_bytes,
                dst_base_bytes,
                byte_stride,
                tl_bytes,
                topleft_off,
                w,
                h,
                angle as i32,
            );
        }
        (BPC::BPC8, 9) => {
            if let Some(t512) = avx512_token {
                ipred_smooth_8bpc_avx512_inner(
                    t512, dst_bytes, dst_base_bytes, byte_stride, tl_bytes, topleft_off, w, h,
                )
            } else {
                ipred_smooth_8bpc_inner(
                    token, dst_bytes, dst_base_bytes, byte_stride, tl_bytes, topleft_off, w, h,
                )
            }
        }
        (BPC::BPC8, 10) => {
            if let Some(t512) = avx512_token {
                ipred_smooth_v_8bpc_avx512_inner(
                    t512, dst_bytes, dst_base_bytes, byte_stride, tl_bytes, topleft_off, w, h,
                )
            } else {
                ipred_smooth_v_8bpc_inner(
                    token, dst_bytes, dst_base_bytes, byte_stride, tl_bytes, topleft_off, w, h,
                )
            }
        }
        (BPC::BPC8, 11) => {
            if let Some(t512) = avx512_token {
                ipred_smooth_h_8bpc_avx512_inner(
                    t512, dst_bytes, dst_base_bytes, byte_stride, tl_bytes, topleft_off, w, h,
                )
            } else {
                ipred_smooth_h_8bpc_inner(
                    token, dst_bytes, dst_base_bytes, byte_stride, tl_bytes, topleft_off, w, h,
                )
            }
        }
        (BPC::BPC8, 12) => {
            if let Some(t512) = avx512_token {
                ipred_paeth_8bpc_avx512_inner(
                    t512, dst_bytes, dst_base_bytes, byte_stride, tl_bytes, topleft_off, w, h,
                )
            } else {
                ipred_paeth_8bpc_inner(
                    token, dst_bytes, dst_base_bytes, byte_stride, tl_bytes, topleft_off, w, h,
                )
            }
        }
        (BPC::BPC8, 13) => {
            ipred_filter_8bpc_inner(
                token,
                dst_bytes,
                dst_base_bytes,
                byte_stride,
                tl_bytes,
                0, // tl_off: full array starts at 0
                w,
                h,
                angle as i32,
                topleft_off,
            )
        }
        (BPC::BPC16, 0) => {
            let tl_off_bytes = topleft_off * 2;
            if let Some(t512) = avx512_token {
                ipred_dc_16bpc_avx512_inner(
                    t512,
                    dst_bytes,
                    dst_base_bytes,
                    byte_stride,
                    tl_bytes,
                    tl_off_bytes,
                    w,
                    h,
                )
            } else {
                ipred_dc_16bpc_inner(
                    token,
                    dst_bytes,
                    dst_base_bytes,
                    byte_stride,
                    tl_bytes,
                    tl_off_bytes,
                    w,
                    h,
                )
            }
        }
        (BPC::BPC16, 1) => {
            let tl_off_bytes = topleft_off * 2;
            if let Some(t512) = avx512_token {
                ipred_v_16bpc_avx512_inner(
                    t512,
                    dst_bytes,
                    dst_base_bytes,
                    byte_stride,
                    tl_bytes,
                    tl_off_bytes,
                    w,
                    h,
                )
            } else {
                ipred_v_16bpc_inner(
                    token,
                    dst_bytes,
                    dst_base_bytes,
                    byte_stride,
                    tl_bytes,
                    tl_off_bytes,
                    w,
                    h,
                )
            }
        }
        (BPC::BPC16, 2) => {
            let tl_off_bytes = topleft_off * 2;
            if let Some(t512) = avx512_token {
                ipred_h_16bpc_avx512_inner(
                    t512,
                    dst_bytes,
                    dst_base_bytes,
                    byte_stride,
                    tl_bytes,
                    tl_off_bytes,
                    w,
                    h,
                )
            } else {
                ipred_h_16bpc_inner(
                    token,
                    dst_bytes,
                    dst_base_bytes,
                    byte_stride,
                    tl_bytes,
                    tl_off_bytes,
                    w,
                    h,
                )
            }
        }
        (BPC::BPC16, 3) => {
            let tl_off_bytes = topleft_off * 2;
            if let Some(t512) = avx512_token {
                ipred_dc_left_16bpc_avx512_inner(
                    t512,
                    dst_bytes,
                    dst_base_bytes,
                    byte_stride,
                    tl_bytes,
                    tl_off_bytes,
                    w,
                    h,
                )
            } else {
                ipred_dc_left_16bpc_inner(
                    token,
                    dst_bytes,
                    dst_base_bytes,
                    byte_stride,
                    tl_bytes,
                    tl_off_bytes,
                    w,
                    h,
                )
            }
        }
        (BPC::BPC16, 4) => {
            let tl_off_bytes = topleft_off * 2;
            if let Some(t512) = avx512_token {
                ipred_dc_top_16bpc_avx512_inner(
                    t512,
                    dst_bytes,
                    dst_base_bytes,
                    byte_stride,
                    tl_bytes,
                    tl_off_bytes,
                    w,
                    h,
                )
            } else {
                ipred_dc_top_16bpc_inner(
                    token,
                    dst_bytes,
                    dst_base_bytes,
                    byte_stride,
                    tl_bytes,
                    tl_off_bytes,
                    w,
                    h,
                )
            }
        }
        (BPC::BPC16, 5) => {
            if let Some(t512) = avx512_token {
                ipred_dc_128_16bpc_avx512_inner(
                    t512,
                    dst_bytes,
                    dst_base_bytes,
                    byte_stride,
                    w,
                    h,
                    bd_c as i32,
                )
            } else {
                ipred_dc_128_16bpc_inner(
                    token,
                    dst_bytes,
                    dst_base_bytes,
                    byte_stride,
                    w,
                    h,
                    bd_c as i32,
                )
            }
        }
        (BPC::BPC16, 6) => {
            let tl_off_bytes = topleft_off * 2;
            ipred_z1_16bpc_inner(
                token,
                dst_bytes,
                dst_base_bytes,
                byte_stride,
                tl_bytes,
                tl_off_bytes,
                w,
                h,
                angle as i32,
                bd_c,
            );
        }
        (BPC::BPC16, 7) => {
            let tl_off_bytes = topleft_off * 2;
            ipred_z2_16bpc_inner(
                token,
                dst_bytes,
                dst_base_bytes,
                byte_stride,
                tl_bytes,
                tl_off_bytes,
                w,
                h,
                angle as i32,
                max_width,
                max_height,
                bd_c,
            );
        }
        (BPC::BPC16, 8) => {
            let tl_off_bytes = topleft_off * 2;
            ipred_z3_16bpc_inner(
                token,
                dst_bytes,
                dst_base_bytes,
                byte_stride,
                tl_bytes,
                tl_off_bytes,
                w,
                h,
                angle as i32,
                bd_c,
            );
        }
        (BPC::BPC16, 9) => {
            let tl_off_bytes = topleft_off * 2;
            if let Some(t512) = avx512_token {
                ipred_smooth_16bpc_avx512_inner(
                    t512, dst_bytes, dst_base_bytes, byte_stride, tl_bytes, tl_off_bytes, w, h,
                )
            } else {
                ipred_smooth_16bpc_inner(
                    token, dst_bytes, dst_base_bytes, byte_stride, tl_bytes, tl_off_bytes, w, h,
                )
            }
        }
        (BPC::BPC16, 10) => {
            let tl_off_bytes = topleft_off * 2;
            if let Some(t512) = avx512_token {
                ipred_smooth_v_16bpc_avx512_inner(
                    t512, dst_bytes, dst_base_bytes, byte_stride, tl_bytes, tl_off_bytes, w, h,
                )
            } else {
                ipred_smooth_v_16bpc_inner(
                    token, dst_bytes, dst_base_bytes, byte_stride, tl_bytes, tl_off_bytes, w, h,
                )
            }
        }
        (BPC::BPC16, 11) => {
            let tl_off_bytes = topleft_off * 2;
            if let Some(t512) = avx512_token {
                ipred_smooth_h_16bpc_avx512_inner(
                    t512, dst_bytes, dst_base_bytes, byte_stride, tl_bytes, tl_off_bytes, w, h,
                )
            } else {
                ipred_smooth_h_16bpc_inner(
                    token, dst_bytes, dst_base_bytes, byte_stride, tl_bytes, tl_off_bytes, w, h,
                )
            }
        }
        (BPC::BPC16, 12) => {
            let tl_off_bytes = topleft_off * 2;
            if let Some(t512) = avx512_token {
                ipred_paeth_16bpc_avx512_inner(
                    t512, dst_bytes, dst_base_bytes, byte_stride, tl_bytes, tl_off_bytes, w, h,
                )
            } else {
                ipred_paeth_16bpc_inner(
                    token, dst_bytes, dst_base_bytes, byte_stride, tl_bytes, tl_off_bytes, w, h,
                )
            }
        }
        (BPC::BPC16, 13) => {
            ipred_filter_16bpc_inner(
                token,
                dst_bytes,
                dst_base_bytes,
                byte_stride,
                tl_bytes,
                0, // tl_off: full array starts at 0
                w,
                h,
                angle as i32,
                bd_c as i32,
                topleft_off,
            )
        }
        _ => return false,
    }
    true
}
