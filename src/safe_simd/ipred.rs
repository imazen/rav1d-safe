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
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let width = width as usize;
    let height = height as usize;
    let dst = dst_ptr as *mut u8;
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
                        0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                        0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
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

// ============================================================================
// SMOOTH Predictions (using weight tables)
// ============================================================================

use crate::src::tables::dav1d_sm_weights;

/// SMOOTH prediction: weighted blend of top/bottom and left/right edges
///
/// pred = w_v[y] * top + (256 - w_v[y]) * bottom + w_h[x] * left + (256 - w_h[x]) * right
/// dst = (pred + 256) >> 9
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
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let width = width as usize;
    let height = height as usize;
    let dst = dst_ptr as *mut u8;
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
                        0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                        0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
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

/// SMOOTH_V prediction: vertical-only weighted blend (top/bottom)
///
/// pred = w_v[y] * top + (256 - w_v[y]) * bottom
/// dst = (pred + 128) >> 8
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
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let width = width as usize;
    let height = height as usize;
    let dst = dst_ptr as *mut u8;
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
                        0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                        0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
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

/// SMOOTH_H prediction: horizontal-only weighted blend (left/right)
///
/// pred = w_h[x] * left + (256 - w_h[x]) * right
/// dst = (pred + 128) >> 8
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
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let width = width as usize;
    let height = height as usize;
    let dst = dst_ptr as *mut u8;
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
                        0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                        0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
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

// ============================================================================
// FILTER Prediction (filter intra)
// ============================================================================

use crate::src::tables::{dav1d_filter_intra_taps, filter_fn, FLT_INCR};

/// FILTER prediction: uses directional filter taps on 4x2 blocks
///
/// Processes in 4x2 blocks. Each output pixel is:
/// sum = sum(filter[i] * p[i] for i in 0..7)
/// out = (sum + 8) >> 4
///
/// Input pixels:
/// p0 = topleft, p1-p4 = top row (4 pixels), p5-p6 = left column (2 pixels)
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
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let width = (width as usize / 4) * 4; // Round down to multiple of 4
    let height = height as usize;
    let dst = dst_ptr as *mut u8;
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

                // Update topleft for next 4x2 block
                tl_pixel = p4;
            }
        }
    }
}
