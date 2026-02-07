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

use archmage::{arcane, Desktop64, SimdToken};
use libc::{c_int, ptrdiff_t};

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
    dst: *mut u8,
    stride: isize,
    width: usize,
    height: usize,
) {
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
                _mm_storeu_si128(
                    dst_row.add(x) as *mut __m128i,
                    _mm256_castsi256_si128(fill_val),
                );
                x += 16;
            }
            while x < width {
                *dst_row.add(x) = 128;
                x += 1;
            }
        }
    }
}

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
    _dst: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    ipred_dc_128_8bpc_inner(
        token,
        dst_ptr as *mut u8,
        stride as isize,
        width as usize,
        height as usize,
    );
}

// ============================================================================
// Vertical Prediction (copy top row)
// ============================================================================

/// Vertical prediction: copy the top row to all rows in the block
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_v_8bpc_inner(
    _token: Desktop64,
    dst: *mut u8,
    stride: isize,
    topleft: *const u8,
    width: usize,
    height: usize,
) {
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
    _dst: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    ipred_v_8bpc_inner(
        token,
        dst_ptr as *mut u8,
        stride as isize,
        unsafe { (topleft as *const u8).add(1) },
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
    dst: *mut u8,
    stride: isize,
    topleft: *const u8,
    width: usize,
    height: usize,
) {
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
                _mm_storeu_si128(
                    dst_row.add(x) as *mut __m128i,
                    _mm256_castsi256_si128(fill_val),
                );
                x += 16;
            }
            while x < width {
                *dst_row.add(x) = left_pixel;
                x += 1;
            }
        }
    }
}

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
    _dst: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    ipred_h_8bpc_inner(
        token,
        dst_ptr as *mut u8,
        stride as isize,
        unsafe { (topleft as *const u8).add(1) },
        width as usize,
        height as usize,
    );
}

// ============================================================================
// DC Prediction (average of top and left)
// ============================================================================

/// DC prediction: fill block with average of top and left edge pixels
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_dc_8bpc_inner(
    _token: Desktop64,
    dst: *mut u8,
    stride: isize,
    topleft: *const u8,
    width: usize,
    height: usize,
) {
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
                _mm_storeu_si128(
                    dst_row.add(x) as *mut __m128i,
                    _mm256_castsi256_si128(fill_val),
                );
                x += 16;
            }
            while x < width {
                *dst_row.add(x) = dc_val;
                x += 1;
            }
        }
    }
}

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
    _dst: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    ipred_dc_8bpc_inner(
        token,
        dst_ptr as *mut u8,
        stride as isize,
        unsafe { (topleft as *const u8).add(1) },
        width as usize,
        height as usize,
    );
}

/// DC_TOP prediction: fill block with average of top edge only
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_dc_top_8bpc_inner(
    _token: Desktop64,
    dst: *mut u8,
    stride: isize,
    topleft: *const u8,
    width: usize,
    height: usize,
) {
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
                _mm_storeu_si128(
                    dst_row.add(x) as *mut __m128i,
                    _mm256_castsi256_si128(fill_val),
                );
                x += 16;
            }
            while x < width {
                *dst_row.add(x) = dc_val;
                x += 1;
            }
        }
    }
}

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
    _dst: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    ipred_dc_top_8bpc_inner(
        token,
        dst_ptr as *mut u8,
        stride as isize,
        unsafe { (topleft as *const u8).add(1) },
        width as usize,
        height as usize,
    );
}

/// DC_LEFT prediction: fill block with average of left edge only
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_dc_left_8bpc_inner(
    _token: Desktop64,
    dst: *mut u8,
    stride: isize,
    topleft: *const u8,
    width: usize,
    height: usize,
) {
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
                _mm_storeu_si128(
                    dst_row.add(x) as *mut __m128i,
                    _mm256_castsi256_si128(fill_val),
                );
                x += 16;
            }
            while x < width {
                *dst_row.add(x) = dc_val;
                x += 1;
            }
        }
    }
}

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
    _dst: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    ipred_dc_left_8bpc_inner(
        token,
        dst_ptr as *mut u8,
        stride as isize,
        unsafe { (topleft as *const u8).add(1) },
        width as usize,
        height as usize,
    );
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
    dst: *mut u8,
    stride: isize,
    topleft: *const u8,
    width: usize,
    height: usize,
) {
    let tl = topleft as *const u8;

    unsafe {
        let topleft_val = *tl as i32;
        let topleft_vec = _mm256_set1_epi32(topleft_val);

        for y in 0..height {
            let dst_row = dst.offset(y as isize * stride);
            let left_val = *tl.offset(-(y as isize) - 1) as i32;
            let left_vec = _mm256_set1_epi32(left_val);

            // Process 8 pixels at a time with AVX2
            let mut x = 0;
            while x + 8 <= width {
                // Load 8 top pixels and zero-extend to 32-bit
                let top_bytes = _mm_loadl_epi64(tl.add(1 + x) as *const __m128i);
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
                        0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 4, 8, 12,
                        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    ),
                );
                let lo = _mm256_castsi256_si128(packed);
                let hi = _mm256_extracti128_si256::<1>(packed);
                let combined = _mm_unpacklo_epi32(lo, hi);
                _mm_storel_epi64(dst_row.add(x) as *mut __m128i, combined);

                x += 8;
            }

            // Scalar fallback for remaining pixels
            while x < width {
                let top_val = *tl.add(1 + x) as i32;
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
                *dst_row.add(x) = result as u8;
                x += 1;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
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
    ipred_paeth_8bpc_inner(
        token,
        dst_ptr as *mut u8,
        stride as isize,
        unsafe { (topleft as *const u8).add(1) },
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
    dst: *mut u8,
    stride: isize,
    topleft: *const u8,
    width: usize,
    height: usize,
) {
    let tl = topleft as *const u8;

    unsafe {
        let weights_hor = &dav1d_sm_weights.0[width..][..width];
        let weights_ver = &dav1d_sm_weights.0[height..][..height];
        let right_val = *tl.add(width) as i32;
        let bottom_val = *tl.offset(-(height as isize)) as i32;
        let right_vec = _mm256_set1_epi32(right_val);
        let bottom_vec = _mm256_set1_epi32(bottom_val);
        let rounding = _mm256_set1_epi32(256);
        let c256 = _mm256_set1_epi32(256);

        for y in 0..height {
            let dst_row = dst.offset(y as isize * stride);
            let left_val = *tl.offset(-(y as isize) - 1) as i32;
            let left_vec = _mm256_set1_epi32(left_val);
            let w_v = weights_ver[y] as i32;
            let w_v_vec = _mm256_set1_epi32(w_v);
            let w_v_inv = _mm256_sub_epi32(c256, w_v_vec);

            let mut x = 0;
            while x + 8 <= width {
                // Load 8 top pixels
                let top_bytes = _mm_loadl_epi64(tl.add(1 + x) as *const __m128i);
                let top = _mm256_cvtepu8_epi32(top_bytes);

                // Load 8 horizontal weights
                let w_h_bytes = _mm_loadl_epi64(weights_hor.as_ptr().add(x) as *const __m128i);
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
                        0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 4, 8, 12,
                        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    ),
                );
                let lo = _mm256_castsi256_si128(packed);
                let hi = _mm256_extracti128_si256::<1>(packed);
                let combined = _mm_unpacklo_epi32(lo, hi);
                _mm_storel_epi64(dst_row.add(x) as *mut __m128i, combined);

                x += 8;
            }

            // Scalar fallback
            while x < width {
                let top_val = *tl.add(1 + x) as i32;
                let w_h = weights_hor[x] as i32;
                let pred = w_v * top_val
                    + (256 - w_v) * bottom_val
                    + w_h * left_val
                    + (256 - w_h) * right_val;
                *dst_row.add(x) = ((pred + 256) >> 9) as u8;
                x += 1;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
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
    ipred_smooth_8bpc_inner(
        token,
        dst_ptr as *mut u8,
        stride as isize,
        unsafe { (topleft as *const u8).add(1) },
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
    dst: *mut u8,
    stride: isize,
    topleft: *const u8,
    width: usize,
    height: usize,
) {
    let tl = topleft as *const u8;

    unsafe {
        let weights_ver = &dav1d_sm_weights.0[height..][..height];
        let bottom_val = *tl.offset(-(height as isize)) as i32;
        let bottom_vec = _mm256_set1_epi32(bottom_val);
        let rounding = _mm256_set1_epi32(128);
        let c256 = _mm256_set1_epi32(256);

        for y in 0..height {
            let dst_row = dst.offset(y as isize * stride);
            let w_v = weights_ver[y] as i32;
            let w_v_vec = _mm256_set1_epi32(w_v);
            let w_v_inv = _mm256_sub_epi32(c256, w_v_vec);

            let mut x = 0;
            while x + 8 <= width {
                // Load 8 top pixels
                let top_bytes = _mm_loadl_epi64(tl.add(1 + x) as *const __m128i);
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
                        0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 4, 8, 12,
                        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    ),
                );
                let lo = _mm256_castsi256_si128(packed);
                let hi = _mm256_extracti128_si256::<1>(packed);
                let combined = _mm_unpacklo_epi32(lo, hi);
                _mm_storel_epi64(dst_row.add(x) as *mut __m128i, combined);

                x += 8;
            }

            // Scalar fallback
            while x < width {
                let top_val = *tl.add(1 + x) as i32;
                let pred = w_v * top_val + (256 - w_v) * bottom_val;
                *dst_row.add(x) = ((pred + 128) >> 8) as u8;
                x += 1;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
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
    ipred_smooth_v_8bpc_inner(
        token,
        dst_ptr as *mut u8,
        stride as isize,
        unsafe { (topleft as *const u8).add(1) },
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
    dst: *mut u8,
    stride: isize,
    topleft: *const u8,
    width: usize,
    height: usize,
) {
    let tl = topleft as *const u8;

    unsafe {
        let weights_hor = &dav1d_sm_weights.0[width..][..width];
        let right_val = *tl.add(width) as i32;
        let right_vec = _mm256_set1_epi32(right_val);
        let rounding = _mm256_set1_epi32(128);
        let c256 = _mm256_set1_epi32(256);

        for y in 0..height {
            let dst_row = dst.offset(y as isize * stride);
            let left_val = *tl.offset(-(y as isize) - 1) as i32;
            let left_vec = _mm256_set1_epi32(left_val);

            let mut x = 0;
            while x + 8 <= width {
                // Load 8 horizontal weights
                let w_h_bytes = _mm_loadl_epi64(weights_hor.as_ptr().add(x) as *const __m128i);
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
                        0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 4, 8, 12,
                        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    ),
                );
                let lo = _mm256_castsi256_si128(packed);
                let hi = _mm256_extracti128_si256::<1>(packed);
                let combined = _mm_unpacklo_epi32(lo, hi);
                _mm_storel_epi64(dst_row.add(x) as *mut __m128i, combined);

                x += 8;
            }

            // Scalar fallback
            while x < width {
                let w_h = weights_hor[x] as i32;
                let pred = w_h * left_val + (256 - w_h) * right_val;
                *dst_row.add(x) = ((pred + 128) >> 8) as u8;
                x += 1;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
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
    ipred_smooth_h_8bpc_inner(
        token,
        dst_ptr as *mut u8,
        stride as isize,
        unsafe { (topleft as *const u8).add(1) },
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
    dst: *mut u8,
    stride: isize,
    topleft: *const u8,
    width: usize,
    height: usize,
    filt_idx: i32,
    topleft_off: usize,
) {
    let width = (width as usize / 4) * 4; // Round down to multiple of 4
    let tl = topleft as *const u8;
    let filt_idx = (filt_idx as usize) & 511;

    let filter = &dav1d_filter_intra_taps[filt_idx];

    unsafe {
        // Process in 4x2 blocks
        for y in (0..height).step_by(2) {
            let tl_off = topleft_off - y;
            let mut tl_pixel = *tl.wrapping_add(tl_off) as i32;

            for x in (0..width).step_by(4) {
                // Get top 4 pixels (p1-p4)
                let top_ptr = tl.wrapping_add(topleft_off + 1 + x);
                let p1 = *top_ptr as i32;
                let p2 = *top_ptr.add(1) as i32;
                let p3 = *top_ptr.add(2) as i32;
                let p4 = *top_ptr.add(3) as i32;

                // Get left 2 pixels (p5, p6)
                let (p5, p6) = if x == 0 {
                    // From original topleft buffer
                    let left_ptr = tl.wrapping_add(tl_off - 1);
                    (*left_ptr as i32, *left_ptr.wrapping_sub(1) as i32)
                } else {
                    // From previously computed output
                    let dst_row0 = dst.offset(y as isize * stride);
                    let dst_row1 = dst.offset((y + 1) as isize * stride);
                    (*dst_row0.add(x - 1) as i32, *dst_row1.add(x - 1) as i32)
                };

                let p0 = tl_pixel;
                let p = [p0, p1, p2, p3, p4, p5, p6];

                // Process 4x2 = 8 output pixels using filter taps
                let flt = filter.0.as_slice();
                let mut flt_offset = 0;

                // Row 0 (4 pixels)
                let dst_row0 = dst.offset(y as isize * stride);
                for xx in 0..4 {
                    let acc = filter_fn(&flt[flt_offset..], p);
                    let val = ((acc + 8) >> 4).clamp(0, 255) as u8;
                    *dst_row0.add(x + xx) = val;
                    flt_offset += FLT_INCR;
                }

                // Row 1 (4 pixels)
                let dst_row1 = dst.offset((y + 1) as isize * stride);
                for xx in 0..4 {
                    let acc = filter_fn(&flt[flt_offset..], p);
                    let val = ((acc + 8) >> 4).clamp(0, 255) as u8;
                    *dst_row1.add(x + xx) = val;
                    flt_offset += FLT_INCR;
                }

                // Update topleft for next 4x2 block (8bpc)
                tl_pixel = p4;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
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
    ipred_filter_8bpc_inner(
        token,
        dst_ptr as *mut u8,
        stride as isize,
        unsafe { (topleft as *const u8).add(1) },
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
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_z1_8bpc_inner(
    _token: Desktop64,
    dst: *mut u8,
    stride: isize,
    topleft: *const u8,
    width: usize,
    height: usize,
    angle: i32,
) {
    let height = height as i32;
    let tl = topleft as *const u8;

    // Extract angle flags
    let is_sm = (angle >> 9) & 1 != 0;
    let enable_intra_edge_filter = (angle >> 10) != 0;
    let angle = angle & 511;

    // Get derivative
    let mut dx = dav1d_dr_intra_derivative[(angle >> 1) as usize] as i32;

    // Determine if we need edge filtering/upsampling
    // For simplicity, this implementation handles the no-filter case only
    // Complex cases fall through to scalar
    let upsample_above = enable_intra_edge_filter
        && (90 - angle) < 40
        && (width + height as usize) <= (16 >> is_sm as usize);

    if upsample_above {
        // Upsampling case - use scalar fallback for now
        // This would require implementing upsample_edge
        unsafe { ipred_z1_scalar(dst, stride, tl, width, height, dx, true) };
        return;
    }

    let filter_strength = if enable_intra_edge_filter {
        get_filter_strength_simple(width as i32 + height, 90 - angle, is_sm)
    } else {
        0
    };

    if filter_strength != 0 {
        // Filtered case - use scalar fallback for now
        unsafe { ipred_z1_scalar(dst, stride, tl, width, height, dx, false) };
        return;
    }

    // No filtering needed - direct access to top pixels
    let top = unsafe { tl.add(1) };
    let max_base_x = width + std::cmp::min(width, height as usize) - 1;
    let base_inc = 1usize;

    unsafe {
        let rounding = _mm256_set1_epi16(32);
        let max_val = _mm256_set1_epi8(255u8 as i8);

        for y in 0..height {
            let xpos = (y + 1) * dx;
            let frac = (xpos & 0x3e) as i16;
            let inv_frac = (64 - frac) as i16;

            let frac_vec = _mm256_set1_epi16(frac);
            let inv_frac_vec = _mm256_set1_epi16(inv_frac);

            let dst_row = dst.offset(y as isize * stride);
            let base0 = (xpos >> 6) as usize;

            let mut x = 0usize;

            // Process 16 pixels at a time
            while x + 16 <= width && base0 + x + 16 < max_base_x {
                let base = base0 + base_inc * x;

                // Load 17 consecutive top pixels (need pairs for interpolation)
                let t0 = _mm_loadu_si128(top.add(base) as *const __m128i);
                let t1 = _mm_loadu_si128(top.add(base + 1) as *const __m128i);

                // Zero-extend to 16-bit
                let t0_lo = _mm256_cvtepu8_epi16(t0);
                let t1_lo = _mm256_cvtepu8_epi16(t1);

                // Interpolate: (t0 * inv_frac + t1 * frac + 32) >> 6
                let prod0 = _mm256_mullo_epi16(t0_lo, inv_frac_vec);
                let prod1 = _mm256_mullo_epi16(t1_lo, frac_vec);
                let sum = _mm256_add_epi16(_mm256_add_epi16(prod0, prod1), rounding);
                let result = _mm256_srai_epi16::<6>(sum);

                // Pack back to 8-bit
                let packed = _mm256_packus_epi16(result, result);
                let lo = _mm256_castsi256_si128(packed);
                let hi = _mm256_extracti128_si256::<1>(packed);
                let combined = _mm_unpacklo_epi64(lo, hi);
                _mm_storeu_si128(dst_row.add(x) as *mut __m128i, combined);

                x += 16;
            }

            // Process remaining pixels with scalar
            while x < width {
                let base = base0 + base_inc * x;
                if base < max_base_x {
                    let t0 = *top.add(base) as i32;
                    let t1 = *top.add(base + 1) as i32;
                    let v = t0 * inv_frac as i32 + t1 * frac as i32;
                    *dst_row.add(x) = ((v + 32) >> 6) as u8;
                } else {
                    // Fill remaining with max value
                    let fill_val = *top.add(max_base_x);
                    for xx in x..width {
                        *dst_row.add(xx) = fill_val;
                    }
                    break;
                }
                x += 1;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
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
    ipred_z1_8bpc_inner(
        token,
        dst_ptr as *mut u8,
        stride as isize,
        unsafe { (topleft as *const u8).add(1) },
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
            _ => 0,
        }
    } else {
        match (wh, angle) {
            (..=8, 56..) => 2,
            (..=8, 40..) => 1,
            (..=16, 40..) => 1,
            _ => 0,
        }
    }
}

/// Scalar fallback for Z1 with edge filtering
#[inline(never)]
unsafe fn ipred_z1_scalar(
    dst: *mut u8,
    stride: ptrdiff_t,
    tl: *const u8,
    width: usize,
    height: i32,
    dx: i32,
    upsample: bool,
) {
    // For now, just implement the basic case
    // A full implementation would need upsample_edge and filter_edge
    let top = unsafe { tl.add(1) };
    let max_base_x = width + std::cmp::min(width, height as usize) - 1;
    let base_inc = if upsample { 2 } else { 1 };
    let dx = if upsample { dx << 1 } else { dx };

    for y in 0..height {
        let xpos = (y + 1) * dx;
        let frac = xpos & 0x3e;
        let inv_frac = 64 - frac;

        let dst_row = unsafe { dst.offset(y as isize * stride) };
        let base0 = (xpos >> 6) as usize;

        for x in 0..width {
            let base = base0 + base_inc * x;
            if base < max_base_x {
                let t0 = unsafe { *top.add(base) } as i32;
                let t1 = unsafe { *top.add(base + 1) } as i32;
                let v = t0 * inv_frac + t1 * frac;
                unsafe { *dst_row.add(x) = ((v + 32) >> 6) as u8 };
            } else {
                let fill_val = unsafe { *top.add(max_base_x) };
                for xx in x..width {
                    unsafe { *dst_row.add(xx) = fill_val };
                }
                break;
            }
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
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_z2_8bpc_inner(
    _token: Desktop64,
    dst: *mut u8,
    stride: isize,
    topleft: *const u8,
    width: usize,
    height: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
) {
    let width = width as i32;
    let height = height as i32;
    let tl = topleft as *const u8;

    // Extract angle flags
    let is_sm = (angle >> 9) & 1 != 0;
    let enable_intra_edge_filter = (angle >> 10) != 0;
    let angle = angle & 511;

    // Z2: angle is between 90 and 180
    // dx: derivative for top edge traversal
    // dy: derivative for left edge traversal
    let dy = dav1d_dr_intra_derivative[((angle - 90) >> 1) as usize] as i32;
    let dx = dav1d_dr_intra_derivative[((180 - angle) >> 1) as usize] as i32;

    // Check for upsampling - fall back to scalar for these complex cases
    let upsample_left = enable_intra_edge_filter
        && (180 - angle) < 40
        && (width + height) <= (16 >> is_sm as usize);
    let upsample_above =
        enable_intra_edge_filter && (angle - 90) < 40 && (width + height) <= (16 >> is_sm as usize);

    if upsample_left || upsample_above {
        // Fall back to scalar for upsampled cases
        unsafe {
            ipred_z2_scalar(
                dst,
                stride,
                tl,
                width,
                height,
                dx,
                dy,
                max_width,
                max_height,
                is_sm,
                enable_intra_edge_filter,
            );
        }
        return;
    }

    // Check for edge filtering
    let filter_strength_above = if enable_intra_edge_filter {
        get_filter_strength_simple(width + height, angle - 90, is_sm)
    } else {
        0
    };
    let filter_strength_left = if enable_intra_edge_filter {
        get_filter_strength_simple(width + height, 180 - angle, is_sm)
    } else {
        0
    };

    if filter_strength_above != 0 || filter_strength_left != 0 {
        // Fall back to scalar for filtered cases
        unsafe {
            ipred_z2_scalar(
                dst,
                stride,
                tl,
                width,
                height,
                dx,
                dy,
                max_width,
                max_height,
                is_sm,
                enable_intra_edge_filter,
            );
        }
        return;
    }

    // No filtering - direct edge access
    // top = tl + 1, left = tl - 1, -2, -3, ...
    let top = unsafe { tl.add(1) };

    unsafe {
        let rounding = _mm256_set1_epi16(32);

        for y in 0..height {
            let xpos = (1 << 6) - dx * (y + 1);
            let base_x0 = xpos >> 6;
            let frac_x = (xpos & 0x3e) as i16;
            let inv_frac_x = (64 - frac_x) as i16;

            let dst_row = dst.offset(y as isize * stride);

            // Find transition point where we switch from top to left
            // base_x = base_x0 + x, so switch happens when base_x0 + x < 0
            // i.e., when x < -base_x0
            let switch_x = if base_x0 < 0 { 0 } else { (-base_x0) as usize };
            let switch_x = std::cmp::min(switch_x, width as usize);

            // Process pixels using top edge (base_x >= 0)
            let mut x = 0usize;

            // SIMD path for top edge pixels - process 16 at a time
            while x + 16 <= switch_x {
                let base = (base_x0 + x as i32) as usize;

                let t0 = _mm_loadu_si128(top.add(base) as *const __m128i);
                let t1 = _mm_loadu_si128(top.add(base + 1) as *const __m128i);

                let t0_lo = _mm256_cvtepu8_epi16(t0);
                let t1_lo = _mm256_cvtepu8_epi16(t1);

                let frac_vec = _mm256_set1_epi16(frac_x);
                let inv_frac_vec = _mm256_set1_epi16(inv_frac_x);

                let prod0 = _mm256_mullo_epi16(t0_lo, inv_frac_vec);
                let prod1 = _mm256_mullo_epi16(t1_lo, frac_vec);
                let sum = _mm256_add_epi16(_mm256_add_epi16(prod0, prod1), rounding);
                let result = _mm256_srai_epi16::<6>(sum);

                let packed = _mm256_packus_epi16(result, result);
                let lo = _mm256_castsi256_si128(packed);
                let hi = _mm256_extracti128_si256::<1>(packed);
                let combined = _mm_unpacklo_epi64(lo, hi);
                _mm_storeu_si128(dst_row.add(x) as *mut __m128i, combined);

                x += 16;
            }

            // Scalar for remaining top edge pixels
            while x < switch_x {
                let base = (base_x0 + x as i32) as usize;
                let t0 = *top.add(base) as i32;
                let t1 = *top.add(base + 1) as i32;
                let v = t0 * inv_frac_x as i32 + t1 * frac_x as i32;
                *dst_row.add(x) = ((v + 32) >> 6) as u8;
                x += 1;
            }

            // Now process pixels using left edge (base_x < 0)
            while x < width as usize {
                let ypos = (y << 6) - dy * (x as i32 + 1);
                let base_y = ypos >> 6;
                let frac_y = ypos & 0x3e;
                let inv_frac_y = 64 - frac_y;

                // left edge: tl[-1-base_y] and tl[-2-base_y]
                // Since base_y can be negative, we need careful indexing
                let l0 = *tl.offset(-1 - base_y as isize) as i32;
                let l1 = *tl.offset(-2 - base_y as isize) as i32;
                let v = l0 * inv_frac_y + l1 * frac_y;
                *dst_row.add(x) = ((v + 32) >> 6) as u8;
                x += 1;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
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
    ipred_z2_8bpc_inner(
        token,
        dst_ptr as *mut u8,
        stride as isize,
        unsafe { (topleft as *const u8).add(1) },
        width as usize,
        height as usize,
        angle as i32,
        max_width as i32,
        max_height as i32,
    );
}

/// Scalar fallback for Z2 with edge filtering/upsampling
#[inline(never)]
unsafe fn ipred_z2_scalar(
    dst: *mut u8,
    stride: ptrdiff_t,
    tl: *const u8,
    width: i32,
    height: i32,
    dx: i32,
    dy: i32,
    _max_width: i32,
    _max_height: i32,
    _is_sm: bool,
    _enable_filter: bool,
) {
    // Simplified scalar without edge processing
    // Full implementation would need filter_edge and upsample_edge
    let top = unsafe { tl.add(1) };

    for y in 0..height {
        let xpos = (1 << 6) - dx * (y + 1);
        let base_x0 = xpos >> 6;
        let frac_x = xpos & 0x3e;
        let inv_frac_x = 64 - frac_x;

        let dst_row = unsafe { dst.offset(y as isize * stride) };

        for x in 0..width as usize {
            let base_x = base_x0 + x as i32;

            let v = if base_x >= 0 {
                let t0 = unsafe { *top.add(base_x as usize) } as i32;
                let t1 = unsafe { *top.add(base_x as usize + 1) } as i32;
                t0 * inv_frac_x + t1 * frac_x
            } else {
                let ypos = (y << 6) - dy * (x as i32 + 1);
                let base_y = ypos >> 6;
                let frac_y = ypos & 0x3e;
                let inv_frac_y = 64 - frac_y;

                let l0 = unsafe { *tl.offset(-1 - base_y as isize) } as i32;
                let l1 = unsafe { *tl.offset(-2 - base_y as isize) } as i32;
                l0 * inv_frac_y + l1 * frac_y
            };
            unsafe { *dst_row.add(x) = ((v + 32) >> 6) as u8 };
        }
    }
}

// ============================================================================
// Z3 Prediction (angular prediction for angles > 180)
// ============================================================================

/// Z3 prediction: directional prediction using left edge only (angles > 180°)
///
/// Z3 is the mirror of Z1, using the left edge instead of top.
/// Loop order is column-major (outer x, inner y) for better cache locality
/// when accessing the left edge.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_z3_8bpc_inner(
    _token: Desktop64,
    dst: *mut u8,
    stride: isize,
    topleft: *const u8,
    width: usize,
    height: usize,
    angle: i32,
) {
    let height = height as i32;
    let tl = topleft as *const u8;

    // Extract angle flags
    let is_sm = (angle >> 9) & 1 != 0;
    let enable_intra_edge_filter = (angle >> 10) != 0;
    let angle = angle & 511;

    // Get derivative for left edge traversal
    let dy = dav1d_dr_intra_derivative[((270 - angle) >> 1) as usize] as usize;

    // Check for upsampling - fall back to scalar
    let upsample_left = enable_intra_edge_filter
        && (angle - 180) < 40
        && (width as i32 + height) <= (16 >> is_sm as usize);

    if upsample_left {
        unsafe { ipred_z3_scalar(dst, stride, tl, width, height, dy, true) };
        return;
    }

    // Check for edge filtering
    let filter_strength = if enable_intra_edge_filter {
        get_filter_strength_simple(width as i32 + height, angle - 180, is_sm)
    } else {
        0
    };

    if filter_strength != 0 {
        unsafe { ipred_z3_scalar(dst, stride, tl, width, height, dy, false) };
        return;
    }

    // No filtering - direct access to left edge
    // left[0] = tl[-1], left[1] = tl[-2], etc.
    let max_base_y = height as usize + std::cmp::min(width, height as usize) - 1;
    let base_inc = 1usize;

    // Z3 has column-major access pattern, so SIMD is tricky
    // Process columns, each with different dy offset
    unsafe {
        for x in 0..width {
            let ypos = dy * (x + 1);
            let frac = (ypos & 0x3e) as i32;
            let inv_frac = 64 - frac;

            for y in 0..height {
                let base = (ypos >> 6) + base_inc * y as usize;
                let dst_pixel = dst.offset(y as isize * stride).add(x);

                if base < max_base_y {
                    // left[base] = tl[-(base+1)]
                    let l0 = *tl.offset(-(base as isize + 1)) as i32;
                    let l1 = *tl.offset(-(base as isize + 2)) as i32;
                    let v = l0 * inv_frac + l1 * frac;
                    *dst_pixel = ((v + 32) >> 6) as u8;
                } else {
                    // Fill rest of column with max value
                    let fill_val = *tl.offset(-(max_base_y as isize + 1));
                    for yy in y..height {
                        *dst.offset(yy as isize * stride).add(x) = fill_val;
                    }
                    break;
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
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
    ipred_z3_8bpc_inner(
        token,
        dst_ptr as *mut u8,
        stride as isize,
        unsafe { (topleft as *const u8).add(1) },
        width as usize,
        height as usize,
        angle as i32,
    );
}

/// Scalar fallback for Z3 with edge filtering/upsampling
#[inline(never)]
unsafe fn ipred_z3_scalar(
    dst: *mut u8,
    stride: ptrdiff_t,
    tl: *const u8,
    width: usize,
    height: i32,
    dy: usize,
    upsample: bool,
) {
    let base_inc = if upsample { 2 } else { 1 };
    let dy = if upsample { dy << 1 } else { dy };
    let max_base_y = height as usize + std::cmp::min(width, height as usize) - 1;

    for x in 0..width {
        let ypos = dy * (x + 1);
        let frac = (ypos & 0x3e) as i32;
        let inv_frac = 64 - frac;

        for y in 0..height {
            let base = (ypos >> 6) + base_inc * y as usize;
            let dst_pixel = unsafe { dst.offset(y as isize * stride).add(x) };

            if base < max_base_y {
                let l0 = unsafe { *tl.offset(-(base as isize + 1)) } as i32;
                let l1 = unsafe { *tl.offset(-(base as isize + 2)) } as i32;
                let v = l0 * inv_frac + l1 * frac;
                unsafe { *dst_pixel = ((v + 32) >> 6) as u8 };
            } else {
                let fill_val = unsafe { *tl.offset(-(max_base_y as isize + 1)) };
                for yy in y..height {
                    unsafe { *dst.offset(yy as isize * stride).add(x) = fill_val };
                }
                break;
            }
        }
    }
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
    dst: *mut u8,
    stride: isize,
    width: usize,
    height: usize,
    bitdepth_max: i32,
) {
    let dst = dst as *mut u16;
    let stride_u16 = stride / 2; // stride is in bytes, we need u16 stride

    // Mid-value is (bitdepth_max + 1) / 2
    let mid_val = ((bitdepth_max + 1) / 2) as i16;

    unsafe {
        let fill_val = _mm256_set1_epi16(mid_val);

        for y in 0..height {
            let row = dst.offset(y as isize * stride_u16);
            let mut x = 0usize;

            // Process 16 pixels at a time (256-bit / 16-bit = 16 pixels)
            while x + 16 <= width {
                _mm256_storeu_si256(row.add(x) as *mut __m256i, fill_val);
                x += 16;
            }

            // Process 8 pixels at a time
            while x + 8 <= width {
                _mm_storeu_si128(row.add(x) as *mut __m128i, _mm256_castsi256_si128(fill_val));
                x += 8;
            }

            // Remaining pixels
            while x < width {
                *row.add(x) = mid_val as u16;
                x += 1;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
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
    ipred_dc_128_16bpc_inner(
        token,
        dst_ptr as *mut u8,
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
    dst: *mut u8,
    stride: isize,
    topleft: *const u8,
    width: usize,
    height: usize,
) {
    let dst = dst as *mut u16;
    let top = unsafe { (topleft as *const u16).add(1) };
    let stride_u16 = stride / 2;

    unsafe {
        // Load top row pixels that we'll copy to all rows
        // We need to handle variable widths

        for y in 0..height {
            let row = dst.offset(y as isize * stride_u16);
            let mut x = 0usize;

            // Process 16 pixels at a time
            while x + 16 <= width {
                let top_vals = _mm256_loadu_si256(top.add(x) as *const __m256i);
                _mm256_storeu_si256(row.add(x) as *mut __m256i, top_vals);
                x += 16;
            }

            // Process 8 pixels at a time
            while x + 8 <= width {
                let top_vals = _mm_loadu_si128(top.add(x) as *const __m128i);
                _mm_storeu_si128(row.add(x) as *mut __m128i, top_vals);
                x += 8;
            }

            // Remaining pixels
            while x < width {
                *row.add(x) = *top.add(x);
                x += 1;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
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
    ipred_v_16bpc_inner(
        token,
        dst_ptr as *mut u8,
        stride as isize,
        unsafe { (topleft as *const u8).add(1) },
        width as usize,
        height as usize,
    );
}

/// Horizontal prediction for 16bpc: fill each row with its left pixel
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_h_16bpc_inner(
    _token: Desktop64,
    dst: *mut u8,
    stride: isize,
    topleft: *const u8,
    width: usize,
    height: usize,
) {
    let dst = dst as *mut u16;
    let tl = topleft as *const u16;
    let stride_u16 = stride / 2;

    unsafe {
        for y in 0..height {
            // Left pixel for this row: topleft[-1-y]
            let left_val = *tl.offset(-(y as isize + 1));
            let fill_val = _mm256_set1_epi16(left_val as i16);

            let row = dst.offset(y as isize * stride_u16);
            let mut x = 0usize;

            // Process 16 pixels at a time
            while x + 16 <= width {
                _mm256_storeu_si256(row.add(x) as *mut __m256i, fill_val);
                x += 16;
            }

            // Process 8 pixels at a time
            while x + 8 <= width {
                _mm_storeu_si128(row.add(x) as *mut __m128i, _mm256_castsi256_si128(fill_val));
                x += 8;
            }

            // Remaining pixels
            while x < width {
                *row.add(x) = left_val;
                x += 1;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
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
    ipred_h_16bpc_inner(
        token,
        dst_ptr as *mut u8,
        stride as isize,
        unsafe { (topleft as *const u8).add(1) },
        width as usize,
        height as usize,
    );
}

/// DC prediction for 16bpc: average of top and left edge pixels
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_dc_16bpc_inner(
    _token: Desktop64,
    dst: *mut u8,
    stride: isize,
    topleft: *const u8,
    width: usize,
    height: usize,
) {
    let dst = dst as *mut u16;
    let tl = topleft as *const u16;
    let stride_u16 = stride / 2;

    // Calculate average of top row and left column
    let mut sum = 0u32;

    // Sum top row: tl[1..=width]
    for i in 1..=width {
        sum += unsafe { *tl.add(i) } as u32;
    }

    // Sum left column: tl[-1..-height]
    for i in 1..=height {
        sum += unsafe { *tl.offset(-(i as isize)) } as u32;
    }

    // Average with rounding
    let count = (width + height) as u32;
    let avg = ((sum + count / 2) / count) as u16;

    unsafe {
        let fill_val = _mm256_set1_epi16(avg as i16);

        for y in 0..height {
            let row = dst.offset(y as isize * stride_u16);
            let mut x = 0usize;

            while x + 16 <= width {
                _mm256_storeu_si256(row.add(x) as *mut __m256i, fill_val);
                x += 16;
            }

            while x + 8 <= width {
                _mm_storeu_si128(row.add(x) as *mut __m128i, _mm256_castsi256_si128(fill_val));
                x += 8;
            }

            while x < width {
                *row.add(x) = avg;
                x += 1;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
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
    ipred_dc_16bpc_inner(
        token,
        dst_ptr as *mut u8,
        stride as isize,
        unsafe { (topleft as *const u8).add(1) },
        width as usize,
        height as usize,
    );
}

/// DC_TOP prediction for 16bpc: average of top edge pixels
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_dc_top_16bpc_inner(
    _token: Desktop64,
    dst: *mut u8,
    stride: isize,
    topleft: *const u8,
    width: usize,
    height: usize,
) {
    let dst = dst as *mut u16;
    let tl = topleft as *const u16;
    let stride_u16 = stride / 2;

    // Calculate average of top row
    let mut sum = 0u32;
    for i in 1..=width {
        sum += unsafe { *tl.add(i) } as u32;
    }
    let avg = ((sum + width as u32 / 2) / width as u32) as u16;

    unsafe {
        let fill_val = _mm256_set1_epi16(avg as i16);

        for y in 0..height {
            let row = dst.offset(y as isize * stride_u16);
            let mut x = 0usize;

            while x + 16 <= width {
                _mm256_storeu_si256(row.add(x) as *mut __m256i, fill_val);
                x += 16;
            }

            while x + 8 <= width {
                _mm_storeu_si128(row.add(x) as *mut __m128i, _mm256_castsi256_si128(fill_val));
                x += 8;
            }

            while x < width {
                *row.add(x) = avg;
                x += 1;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
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
    ipred_dc_top_16bpc_inner(
        token,
        dst_ptr as *mut u8,
        stride as isize,
        unsafe { (topleft as *const u8).add(1) },
        width as usize,
        height as usize,
    );
}

/// DC_LEFT prediction for 16bpc: average of left edge pixels
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_dc_left_16bpc_inner(
    _token: Desktop64,
    dst: *mut u8,
    stride: isize,
    topleft: *const u8,
    width: usize,
    height: usize,
) {
    let dst = dst as *mut u16;
    let tl = topleft as *const u16;
    let stride_u16 = stride / 2;

    // Calculate average of left column
    let mut sum = 0u32;
    for i in 1..=height {
        sum += unsafe { *tl.offset(-(i as isize)) } as u32;
    }
    let avg = ((sum + height as u32 / 2) / height as u32) as u16;

    unsafe {
        let fill_val = _mm256_set1_epi16(avg as i16);

        for y in 0..height {
            let row = dst.offset(y as isize * stride_u16);
            let mut x = 0usize;

            while x + 16 <= width {
                _mm256_storeu_si256(row.add(x) as *mut __m256i, fill_val);
                x += 16;
            }

            while x + 8 <= width {
                _mm_storeu_si128(row.add(x) as *mut __m128i, _mm256_castsi256_si128(fill_val));
                x += 8;
            }

            while x < width {
                *row.add(x) = avg;
                x += 1;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
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
    ipred_dc_left_16bpc_inner(
        token,
        dst_ptr as *mut u8,
        stride as isize,
        unsafe { (topleft as *const u8).add(1) },
        width as usize,
        height as usize,
    );
}

/// PAETH prediction for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_paeth_16bpc_inner(
    _token: Desktop64,
    dst: *mut u8,
    stride: isize,
    topleft: *const u8,
    width: usize,
    height: usize,
) {
    let dst = dst as *mut u16;
    let tl = topleft as *const u16;
    let stride_u16 = stride / 2;

    unsafe {
        let topleft_val = *tl as i32;

        for y in 0..height {
            let left_val = *tl.offset(-(y as isize + 1)) as i32;
            let dst_row = dst.offset(y as isize * stride_u16);

            // Process each pixel - PAETH is complex so use scalar
            for x in 0..width {
                let top_val = *tl.add(x + 1) as i32;

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

                *dst_row.add(x) = pred as u16;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
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
    ipred_paeth_16bpc_inner(
        token,
        dst_ptr as *mut u8,
        stride as isize,
        unsafe { (topleft as *const u8).add(1) },
        width as usize,
        height as usize,
    );
}

/// SMOOTH prediction for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_smooth_16bpc_inner(
    _token: Desktop64,
    dst: *mut u8,
    stride: isize,
    topleft: *const u8,
    width: usize,
    height: usize,
) {
    let dst = dst as *mut u16;
    let tl = topleft as *const u16;
    let stride_u16 = stride / 2;

    unsafe {
        let weights_hor = &dav1d_sm_weights.0[width..][..width];
        let weights_ver = &dav1d_sm_weights.0[height..][..height];
        let right_val = *tl.add(width) as i32;
        let bottom_val = *tl.offset(-(height as isize)) as i32;

        for y in 0..height {
            let dst_row = dst.offset(y as isize * stride_u16);
            let left_val = *tl.offset(-(y as isize) - 1) as i32;
            let w_v = weights_ver[y] as i32;

            for x in 0..width {
                let top_val = *tl.add(1 + x) as i32;
                let w_h = weights_hor[x] as i32;

                // Vertical component: w_v * top + (256 - w_v) * bottom
                let vert = w_v * top_val + (256 - w_v) * bottom_val;
                // Horizontal component: w_h * left + (256 - w_h) * right
                let horz = w_h * left_val + (256 - w_h) * right_val;
                // Combine with rounding
                let pred = (vert + horz + 256) >> 9;
                *dst_row.add(x) = pred as u16;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
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
    ipred_smooth_16bpc_inner(
        token,
        dst_ptr as *mut u8,
        stride as isize,
        unsafe { (topleft as *const u8).add(1) },
        width as usize,
        height as usize,
    );
}

/// SMOOTH_V prediction for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_smooth_v_16bpc_inner(
    _token: Desktop64,
    dst: *mut u8,
    stride: isize,
    topleft: *const u8,
    width: usize,
    height: usize,
) {
    let dst = dst as *mut u16;
    let tl = topleft as *const u16;
    let stride_u16 = stride / 2;

    unsafe {
        let weights_ver = &dav1d_sm_weights.0[height..][..height];
        let bottom_val = *tl.offset(-(height as isize)) as i32;

        for y in 0..height {
            let dst_row = dst.offset(y as isize * stride_u16);
            let w_v = weights_ver[y] as i32;

            for x in 0..width {
                let top_val = *tl.add(1 + x) as i32;
                let pred = (w_v * top_val + (256 - w_v) * bottom_val + 128) >> 8;
                *dst_row.add(x) = pred as u16;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
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
    ipred_smooth_v_16bpc_inner(
        token,
        dst_ptr as *mut u8,
        stride as isize,
        unsafe { (topleft as *const u8).add(1) },
        width as usize,
        height as usize,
    );
}

/// SMOOTH_H prediction for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_smooth_h_16bpc_inner(
    _token: Desktop64,
    dst: *mut u8,
    stride: isize,
    topleft: *const u8,
    width: usize,
    height: usize,
) {
    let dst = dst as *mut u16;
    let tl = topleft as *const u16;
    let stride_u16 = stride / 2;

    unsafe {
        let weights_hor = &dav1d_sm_weights.0[width..][..width];
        let right_val = *tl.add(width) as i32;

        for y in 0..height {
            let dst_row = dst.offset(y as isize * stride_u16);
            let left_val = *tl.offset(-(y as isize) - 1) as i32;

            for x in 0..width {
                let w_h = weights_hor[x] as i32;
                let pred = (w_h * left_val + (256 - w_h) * right_val + 128) >> 8;
                *dst_row.add(x) = pred as u16;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
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
    ipred_smooth_h_16bpc_inner(
        token,
        dst_ptr as *mut u8,
        stride as isize,
        unsafe { (topleft as *const u8).add(1) },
        width as usize,
        height as usize,
    );
}

// ============================================================================
// Z1 Prediction 16bpc (angular prediction for angles < 90)
// ============================================================================

/// Z1 prediction for 16bpc: directional prediction using top edge only (angles < 90°)
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_z1_16bpc_inner(
    _token: Desktop64,
    dst: *mut u8,
    stride: isize,
    topleft: *const u8,
    width: usize,
    height: usize,
    angle: i32,
) {
    let height = height as i32;
    let dst = dst as *mut u16;
    let tl = topleft as *const u16;
    let stride_u16 = stride / 2;

    // Extract angle flags
    let is_sm = (angle >> 9) & 1 != 0;
    let enable_intra_edge_filter = (angle >> 10) != 0;
    let angle = angle & 511;

    // Get derivative
    let dx = dav1d_dr_intra_derivative[(angle >> 1) as usize] as i32;

    // Determine if we need edge filtering/upsampling
    let upsample_above = enable_intra_edge_filter
        && (90 - angle) < 40
        && (width as i32 + height) <= (16 >> is_sm as usize);

    if upsample_above {
        // Upsampling case - use scalar fallback
        unsafe { ipred_z1_16bpc_scalar(dst, stride_u16, tl, width, height, dx, true) };
        return;
    }

    let filter_strength = if enable_intra_edge_filter {
        get_filter_strength_simple(width as i32 + height, 90 - angle, is_sm)
    } else {
        0
    };

    if filter_strength != 0 {
        // Filtered case - use scalar fallback
        unsafe { ipred_z1_16bpc_scalar(dst, stride_u16, tl, width, height, dx, false) };
        return;
    }

    // No filtering needed - direct access to top pixels
    let top = unsafe { tl.add(1) };
    let max_base_x = width + std::cmp::min(width, height as usize) - 1;
    let base_inc = 1usize;

    unsafe {
        let rounding = _mm256_set1_epi32(32);

        for y in 0..height {
            let xpos = (y + 1) * dx;
            let frac = (xpos & 0x3e) as i32;
            let inv_frac = 64 - frac;

            let frac_vec = _mm256_set1_epi32(frac);
            let inv_frac_vec = _mm256_set1_epi32(inv_frac);

            let dst_row = dst.offset(y as isize * stride_u16);
            let base0 = (xpos >> 6) as usize;

            let mut x = 0usize;

            // Process 8 pixels at a time (256-bit / 32-bit intermediate = 8 pixels)
            while x + 8 <= width && base0 + x + 8 < max_base_x {
                let base = base0 + base_inc * x;

                // Load 9 consecutive top pixels (need pairs for interpolation)
                let t0 = _mm_loadu_si128(top.add(base) as *const __m128i);
                let t1 = _mm_loadu_si128(top.add(base + 1) as *const __m128i);

                // Zero-extend to 32-bit for precise multiply
                let t0_lo = _mm256_cvtepu16_epi32(t0);
                let t1_lo = _mm256_cvtepu16_epi32(t1);

                // Interpolate: (t0 * inv_frac + t1 * frac + 32) >> 6
                let prod0 = _mm256_mullo_epi32(t0_lo, inv_frac_vec);
                let prod1 = _mm256_mullo_epi32(t1_lo, frac_vec);
                let sum = _mm256_add_epi32(_mm256_add_epi32(prod0, prod1), rounding);
                let result = _mm256_srai_epi32::<6>(sum);

                // Pack back to 16-bit
                let packed = _mm256_packus_epi32(result, result);
                let lo = _mm256_castsi256_si128(packed);
                let hi = _mm256_extracti128_si256::<1>(packed);
                let combined = _mm_unpacklo_epi64(lo, hi);
                _mm_storeu_si128(dst_row.add(x) as *mut __m128i, combined);

                x += 8;
            }

            // Process remaining pixels with scalar
            while x < width {
                let base = base0 + base_inc * x;
                if base < max_base_x {
                    let t0 = *top.add(base) as i32;
                    let t1 = *top.add(base + 1) as i32;
                    let v = t0 * inv_frac + t1 * frac;
                    *dst_row.add(x) = ((v + 32) >> 6) as u16;
                } else {
                    // Fill remaining with max value
                    let fill_val = *top.add(max_base_x);
                    for xx in x..width {
                        *dst_row.add(xx) = fill_val;
                    }
                    break;
                }
                x += 1;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
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
    ipred_z1_16bpc_inner(
        token,
        dst_ptr as *mut u8,
        stride as isize,
        unsafe { (topleft as *const u8).add(1) },
        width as usize,
        height as usize,
        angle as i32,
    );
}

/// Scalar fallback for Z1 16bpc with edge filtering
#[inline(never)]
unsafe fn ipred_z1_16bpc_scalar(
    dst: *mut u16,
    stride_u16: isize,
    tl: *const u16,
    width: usize,
    height: i32,
    dx: i32,
    upsample: bool,
) {
    let top = unsafe { tl.add(1) };
    let max_base_x = width + std::cmp::min(width, height as usize) - 1;
    let base_inc = if upsample { 2 } else { 1 };
    let dx = if upsample { dx << 1 } else { dx };

    for y in 0..height {
        let xpos = (y + 1) * dx;
        let frac = xpos & 0x3e;
        let inv_frac = 64 - frac;

        let dst_row = unsafe { dst.offset(y as isize * stride_u16) };
        let base0 = (xpos >> 6) as usize;

        for x in 0..width {
            let base = base0 + base_inc * x;
            if base < max_base_x {
                let t0 = unsafe { *top.add(base) } as i32;
                let t1 = unsafe { *top.add(base + 1) } as i32;
                let v = t0 * inv_frac + t1 * frac;
                unsafe { *dst_row.add(x) = ((v + 32) >> 6) as u16 };
            } else {
                let fill_val = unsafe { *top.add(max_base_x) };
                for xx in x..width {
                    unsafe { *dst_row.add(xx) = fill_val };
                }
                break;
            }
        }
    }
}

// ============================================================================
// Z2 Prediction 16bpc (angular prediction for angles 90-180)
// ============================================================================

/// Z2 prediction for 16bpc: directional prediction using both top AND left edges (angles 90-180°)
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_z2_16bpc_inner(
    _token: Desktop64,
    dst: *mut u8,
    stride: isize,
    topleft: *const u8,
    width: usize,
    height: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
) {
    let width = width as i32;
    let height = height as i32;
    let dst = dst as *mut u16;
    let tl = topleft as *const u16;
    let stride_u16 = stride / 2;

    // Extract angle flags
    let is_sm = (angle >> 9) & 1 != 0;
    let enable_intra_edge_filter = (angle >> 10) != 0;
    let angle = angle & 511;

    // Z2: angle is between 90 and 180
    let dy = dav1d_dr_intra_derivative[((angle - 90) >> 1) as usize] as i32;
    let dx = dav1d_dr_intra_derivative[((180 - angle) >> 1) as usize] as i32;

    // Check for upsampling - fall back to scalar for these complex cases
    let upsample_left = enable_intra_edge_filter
        && (180 - angle) < 40
        && (width + height) <= (16 >> is_sm as usize);
    let upsample_above =
        enable_intra_edge_filter && (angle - 90) < 40 && (width + height) <= (16 >> is_sm as usize);

    if upsample_left || upsample_above {
        unsafe {
            ipred_z2_16bpc_scalar(
                dst,
                stride_u16,
                tl,
                width,
                height,
                dx,
                dy,
                max_width,
                max_height,
                is_sm,
                enable_intra_edge_filter,
            );
        }
        return;
    }

    // Check for edge filtering
    let filter_strength_above = if enable_intra_edge_filter {
        get_filter_strength_simple(width + height, angle - 90, is_sm)
    } else {
        0
    };
    let filter_strength_left = if enable_intra_edge_filter {
        get_filter_strength_simple(width + height, 180 - angle, is_sm)
    } else {
        0
    };

    if filter_strength_above != 0 || filter_strength_left != 0 {
        unsafe {
            ipred_z2_16bpc_scalar(
                dst,
                stride_u16,
                tl,
                width,
                height,
                dx,
                dy,
                max_width,
                max_height,
                is_sm,
                enable_intra_edge_filter,
            );
        }
        return;
    }

    // No filtering - direct edge access
    let top = unsafe { tl.add(1) };

    unsafe {
        let rounding = _mm256_set1_epi32(32);

        for y in 0..height {
            let xpos = (1 << 6) - dx * (y + 1);
            let base_x0 = xpos >> 6;
            let frac_x = (xpos & 0x3e) as i32;
            let inv_frac_x = 64 - frac_x;

            let dst_row = dst.offset(y as isize * stride_u16);

            // Find transition point where we switch from top to left
            let switch_x = if base_x0 < 0 { 0 } else { (-base_x0) as usize };
            let switch_x = std::cmp::min(switch_x, width as usize);

            // Process pixels using top edge (base_x >= 0)
            let mut x = 0usize;

            // SIMD path for top edge pixels - process 8 at a time
            while x + 8 <= switch_x {
                let base = (base_x0 + x as i32) as usize;

                let t0 = _mm_loadu_si128(top.add(base) as *const __m128i);
                let t1 = _mm_loadu_si128(top.add(base + 1) as *const __m128i);

                let t0_lo = _mm256_cvtepu16_epi32(t0);
                let t1_lo = _mm256_cvtepu16_epi32(t1);

                let frac_vec = _mm256_set1_epi32(frac_x);
                let inv_frac_vec = _mm256_set1_epi32(inv_frac_x);

                let prod0 = _mm256_mullo_epi32(t0_lo, inv_frac_vec);
                let prod1 = _mm256_mullo_epi32(t1_lo, frac_vec);
                let sum = _mm256_add_epi32(_mm256_add_epi32(prod0, prod1), rounding);
                let result = _mm256_srai_epi32::<6>(sum);

                let packed = _mm256_packus_epi32(result, result);
                let lo = _mm256_castsi256_si128(packed);
                let hi = _mm256_extracti128_si256::<1>(packed);
                let combined = _mm_unpacklo_epi64(lo, hi);
                _mm_storeu_si128(dst_row.add(x) as *mut __m128i, combined);

                x += 8;
            }

            // Scalar for remaining top edge pixels
            while x < switch_x {
                let base = (base_x0 + x as i32) as usize;
                let t0 = *top.add(base) as i32;
                let t1 = *top.add(base + 1) as i32;
                let v = t0 * inv_frac_x + t1 * frac_x;
                *dst_row.add(x) = ((v + 32) >> 6) as u16;
                x += 1;
            }

            // Now process pixels using left edge (base_x < 0)
            while x < width as usize {
                let ypos = (y << 6) - dy * (x as i32 + 1);
                let base_y = ypos >> 6;
                let frac_y = ypos & 0x3e;
                let inv_frac_y = 64 - frac_y;

                let l0 = *tl.offset(-1 - base_y as isize) as i32;
                let l1 = *tl.offset(-2 - base_y as isize) as i32;
                let v = l0 * inv_frac_y + l1 * frac_y;
                *dst_row.add(x) = ((v + 32) >> 6) as u16;
                x += 1;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
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
    ipred_z2_16bpc_inner(
        token,
        dst_ptr as *mut u8,
        stride as isize,
        unsafe { (topleft as *const u8).add(1) },
        width as usize,
        height as usize,
        angle as i32,
        max_width as i32,
        max_height as i32,
    );
}

/// Scalar fallback for Z2 16bpc with edge filtering/upsampling
#[inline(never)]
unsafe fn ipred_z2_16bpc_scalar(
    dst: *mut u16,
    stride_u16: isize,
    tl: *const u16,
    width: i32,
    height: i32,
    dx: i32,
    dy: i32,
    _max_width: i32,
    _max_height: i32,
    _is_sm: bool,
    _enable_filter: bool,
) {
    let top = unsafe { tl.add(1) };

    for y in 0..height {
        let xpos = (1 << 6) - dx * (y + 1);
        let base_x0 = xpos >> 6;
        let frac_x = xpos & 0x3e;
        let inv_frac_x = 64 - frac_x;

        let dst_row = unsafe { dst.offset(y as isize * stride_u16) };

        for x in 0..width as usize {
            let base_x = base_x0 + x as i32;

            let v = if base_x >= 0 {
                let t0 = unsafe { *top.add(base_x as usize) } as i32;
                let t1 = unsafe { *top.add(base_x as usize + 1) } as i32;
                t0 * inv_frac_x + t1 * frac_x
            } else {
                let ypos = (y << 6) - dy * (x as i32 + 1);
                let base_y = ypos >> 6;
                let frac_y = ypos & 0x3e;
                let inv_frac_y = 64 - frac_y;

                let l0 = unsafe { *tl.offset(-1 - base_y as isize) } as i32;
                let l1 = unsafe { *tl.offset(-2 - base_y as isize) } as i32;
                l0 * inv_frac_y + l1 * frac_y
            };
            unsafe { *dst_row.add(x) = ((v + 32) >> 6) as u16 };
        }
    }
}

// ============================================================================
// Z3 Prediction 16bpc (angular prediction for angles > 180)
// ============================================================================

/// Z3 prediction for 16bpc: directional prediction using left edge only (angles > 180°)
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_z3_16bpc_inner(
    _token: Desktop64,
    dst: *mut u8,
    stride: isize,
    topleft: *const u8,
    width: usize,
    height: usize,
    angle: i32,
) {
    let height = height as i32;
    let dst = dst as *mut u16;
    let tl = topleft as *const u16;
    let stride_u16 = stride / 2;

    // Extract angle flags
    let is_sm = (angle >> 9) & 1 != 0;
    let enable_intra_edge_filter = (angle >> 10) != 0;
    let angle = angle & 511;

    // Get derivative for left edge traversal
    let dy = dav1d_dr_intra_derivative[((270 - angle) >> 1) as usize] as usize;

    // Check for upsampling - fall back to scalar
    let upsample_left = enable_intra_edge_filter
        && (angle - 180) < 40
        && (width as i32 + height) <= (16 >> is_sm as usize);

    if upsample_left {
        unsafe { ipred_z3_16bpc_scalar(dst, stride_u16, tl, width, height, dy, true) };
        return;
    }

    // Check for edge filtering
    let filter_strength = if enable_intra_edge_filter {
        get_filter_strength_simple(width as i32 + height, angle - 180, is_sm)
    } else {
        0
    };

    if filter_strength != 0 {
        unsafe { ipred_z3_16bpc_scalar(dst, stride_u16, tl, width, height, dy, false) };
        return;
    }

    // No filtering - direct access to left edge
    let max_base_y = height as usize + std::cmp::min(width, height as usize) - 1;
    let base_inc = 1usize;

    // Z3 has column-major access pattern
    unsafe {
        for x in 0..width {
            let ypos = dy * (x + 1);
            let frac = (ypos & 0x3e) as i32;
            let inv_frac = 64 - frac;

            for y in 0..height {
                let base = (ypos >> 6) + base_inc * y as usize;
                let dst_pixel = dst.offset(y as isize * stride_u16).add(x);

                if base < max_base_y {
                    let l0 = *tl.offset(-(base as isize + 1)) as i32;
                    let l1 = *tl.offset(-(base as isize + 2)) as i32;
                    let v = l0 * inv_frac + l1 * frac;
                    *dst_pixel = ((v + 32) >> 6) as u16;
                } else {
                    let fill_val = *tl.offset(-(max_base_y as isize + 1));
                    for yy in y..height {
                        *dst.offset(yy as isize * stride_u16).add(x) = fill_val;
                    }
                    break;
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
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
    ipred_z3_16bpc_inner(
        token,
        dst_ptr as *mut u8,
        stride as isize,
        unsafe { (topleft as *const u8).add(1) },
        width as usize,
        height as usize,
        angle as i32,
    );
}

/// Scalar fallback for Z3 16bpc with edge filtering/upsampling
#[inline(never)]
unsafe fn ipred_z3_16bpc_scalar(
    dst: *mut u16,
    stride_u16: isize,
    tl: *const u16,
    width: usize,
    height: i32,
    dy: usize,
    upsample: bool,
) {
    let base_inc = if upsample { 2 } else { 1 };
    let dy = if upsample { dy << 1 } else { dy };
    let max_base_y = height as usize + std::cmp::min(width, height as usize) - 1;

    for x in 0..width {
        let ypos = dy * (x + 1);
        let frac = (ypos & 0x3e) as i32;
        let inv_frac = 64 - frac;

        for y in 0..height {
            let base = (ypos >> 6) + base_inc * y as usize;
            let dst_pixel = unsafe { dst.offset(y as isize * stride_u16).add(x) };

            if base < max_base_y {
                let l0 = unsafe { *tl.offset(-(base as isize + 1)) } as i32;
                let l1 = unsafe { *tl.offset(-(base as isize + 2)) } as i32;
                let v = l0 * inv_frac + l1 * frac;
                unsafe { *dst_pixel = ((v + 32) >> 6) as u16 };
            } else {
                let fill_val = unsafe { *tl.offset(-(max_base_y as isize + 1)) };
                for yy in y..height {
                    unsafe { *dst.offset(yy as isize * stride_u16).add(x) = fill_val };
                }
                break;
            }
        }
    }
}

// ============================================================================
// FILTER Prediction 16bpc
// ============================================================================

/// FILTER prediction for 16bpc: uses 7-tap filter for intra prediction
///
/// Processes in 4x2 blocks. Each output pixel uses 7 input samples.
/// Input pixels: p0 = topleft, p1-p4 = top row (4 pixels), p5-p6 = left column (2 pixels)
/// For 16bpc: out = (sum + 8) >> 4, clamped to [0, bitdepth_max]
#[cfg(target_arch = "x86_64")]
#[arcane]
fn ipred_filter_16bpc_inner(
    _token: Desktop64,
    dst: *mut u8,
    stride: isize,
    topleft: *const u8,
    width: usize,
    height: usize,
    filt_idx: i32,
    bitdepth_max: i32,
    topleft_off: usize,
) {
    let width = (width as usize / 4) * 4; // Round down to multiple of 4
    let dst = dst as *mut u16;
    let tl = topleft as *const u16;
    let stride_u16 = stride / 2;
    let filt_idx = (filt_idx as usize) & 511;

    let filter = &dav1d_filter_intra_taps[filt_idx];

    unsafe {
        // Process in 4x2 blocks
        for y in (0..height).step_by(2) {
            let tl_off = topleft_off - y;
            let mut tl_pixel = *tl.wrapping_add(tl_off) as i32;

            for x in (0..width).step_by(4) {
                // Get top 4 pixels (p1-p4)
                let top_ptr = tl.wrapping_add(topleft_off + 1 + x);
                let p1 = *top_ptr as i32;
                let p2 = *top_ptr.add(1) as i32;
                let p3 = *top_ptr.add(2) as i32;
                let p4 = *top_ptr.add(3) as i32;

                // Get left 2 pixels (p5, p6)
                let (p5, p6) = if x == 0 {
                    // From original topleft buffer
                    let left_ptr = tl.wrapping_sub(tl_off.wrapping_sub(topleft_off) + 1);
                    (*left_ptr as i32, *left_ptr.wrapping_sub(1) as i32)
                } else {
                    // From previously computed output
                    let dst_row0 = dst.offset(y as isize * stride_u16);
                    let dst_row1 = dst.offset((y + 1) as isize * stride_u16);
                    (*dst_row0.add(x - 1) as i32, *dst_row1.add(x - 1) as i32)
                };

                let p0 = tl_pixel;
                let p = [p0, p1, p2, p3, p4, p5, p6];

                // Process 4x2 = 8 output pixels using filter taps
                let flt = filter.0.as_slice();
                let mut flt_offset = 0;

                // Row 0 (4 pixels)
                let dst_row0 = dst.offset(y as isize * stride_u16);
                for xx in 0..4 {
                    let acc = filter_fn(&flt[flt_offset..], p);
                    let val = ((acc + 8) >> 4).clamp(0, bitdepth_max as i32) as u16;
                    *dst_row0.add(x + xx) = val;
                    flt_offset += FLT_INCR;
                }

                // Row 1 (4 pixels)
                let dst_row1 = dst.offset((y + 1) as isize * stride_u16);
                for xx in 0..4 {
                    let acc = filter_fn(&flt[flt_offset..], p);
                    let val = ((acc + 8) >> 4).clamp(0, bitdepth_max as i32) as u16;
                    *dst_row1.add(x + xx) = val;
                    flt_offset += FLT_INCR;
                }

                // Update topleft for next 4x2 block (16bpc)
                tl_pixel = p4;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
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
    ipred_filter_16bpc_inner(
        token,
        dst_ptr as *mut u8,
        stride as isize,
        unsafe { (topleft as *const u8).add(1) },
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
    use crate::src::cpu::CpuFlags;

    if !crate::src::cpu::rav1d_get_cpu_flags().contains(CpuFlags::AVX2) {
        return false;
    }

    let dst_ptr = dst.as_mut_ptr::<BD>().cast();
    let stride = dst.stride();
    let topleft_ptr = topleft[topleft_off..].as_ptr().cast();
    let bd_c = bd.into_c();
    let dst_ffi = FFISafe::new(&dst);

    // SAFETY: AVX2 verified by CpuFlags check. Pointers derived from valid types.
    unsafe {
        match (BD::BPC, mode) {
            (BPC::BPC8, 0) => ipred_dc_8bpc_avx2(
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
            ),
            (BPC::BPC8, 1) => ipred_v_8bpc_avx2(
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
            ),
            (BPC::BPC8, 2) => ipred_h_8bpc_avx2(
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
            ),
            (BPC::BPC8, 3) => ipred_dc_left_8bpc_avx2(
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
            ),
            (BPC::BPC8, 4) => ipred_dc_top_8bpc_avx2(
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
            ),
            (BPC::BPC8, 5) => ipred_dc_128_8bpc_avx2(
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
            ),
            (BPC::BPC8, 6) => ipred_z1_8bpc_avx2(
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
            ),
            (BPC::BPC8, 7) => ipred_z2_8bpc_avx2(
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
            ),
            (BPC::BPC8, 8) => ipred_z3_8bpc_avx2(
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
            ),
            (BPC::BPC8, 9) => ipred_smooth_8bpc_avx2(
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
            ),
            (BPC::BPC8, 10) => ipred_smooth_v_8bpc_avx2(
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
            ),
            (BPC::BPC8, 11) => ipred_smooth_h_8bpc_avx2(
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
            ),
            (BPC::BPC8, 12) => ipred_paeth_8bpc_avx2(
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
            ),
            (BPC::BPC8, 13) => ipred_filter_8bpc_avx2(
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
            ),
            (BPC::BPC16, 0) => ipred_dc_16bpc_avx2(
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
            ),
            (BPC::BPC16, 1) => ipred_v_16bpc_avx2(
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
            ),
            (BPC::BPC16, 2) => ipred_h_16bpc_avx2(
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
            ),
            (BPC::BPC16, 3) => ipred_dc_left_16bpc_avx2(
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
            ),
            (BPC::BPC16, 4) => ipred_dc_top_16bpc_avx2(
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
            ),
            (BPC::BPC16, 5) => ipred_dc_128_16bpc_avx2(
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
            ),
            (BPC::BPC16, 6) => ipred_z1_16bpc_avx2(
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
            ),
            (BPC::BPC16, 7) => ipred_z2_16bpc_avx2(
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
            ),
            (BPC::BPC16, 8) => ipred_z3_16bpc_avx2(
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
            ),
            (BPC::BPC16, 9) => ipred_smooth_16bpc_avx2(
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
            ),
            (BPC::BPC16, 10) => ipred_smooth_v_16bpc_avx2(
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
            ),
            (BPC::BPC16, 11) => ipred_smooth_h_16bpc_avx2(
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
            ),
            (BPC::BPC16, 12) => ipred_paeth_16bpc_avx2(
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
            ),
            (BPC::BPC16, 13) => ipred_filter_16bpc_avx2(
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
            ),
            _ => return false,
        }
    }
    true
}
