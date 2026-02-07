//! Safe SIMD implementations for CDEF (Constrained Directional Enhancement Filter)
//!
//! CDEF applies direction-dependent filtering to remove coding artifacts
//! while preserving edges.

#![allow(unused_imports)]

#[cfg(target_arch = "x86_64")]
use archmage::{arcane, rite, Desktop64, SimdToken};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use std::ffi::c_int;
use std::ffi::c_uint;

use crate::include::common::bitdepth::AsPrimitive;
use crate::include::common::bitdepth::BitDepth;
use crate::include::common::bitdepth::DynPixel;
use crate::include::common::bitdepth::LeftPixelRow2px;
use crate::include::common::intops::apply_sign;
use crate::include::dav1d::picture::PicOffset;
use crate::src::align::AlignedVec64;
use crate::src::cdef::CdefBottom;
use crate::src::cdef::CdefEdgeFlags;
use crate::src::cdef::CdefTop;
use crate::src::disjoint_mut::DisjointMut;
use crate::src::ffi_safe::FFISafe;
use crate::src::pic_or_buf::PicOrBuf;
use crate::src::strided::Strided as _;
use crate::src::tables::dav1d_cdef_directions;
use crate::src::with_offset::WithOffset;
use libc::ptrdiff_t;
use std::cmp;

// ============================================================================
// CONSTRAIN FUNCTION (SIMD)
// ============================================================================

/// SIMD version of constrain for AVX2
/// Processes 16 i16 values at once
/// Formula: sign(diff) * min(|diff|, max(0, threshold - (|diff| >> shift)))
#[cfg(target_arch = "x86_64")]
#[rite]
#[inline]
fn constrain_avx2(_t: Desktop64, diff: __m256i, threshold: __m256i, shift: __m128i) -> __m256i {
    let zero = _mm256_setzero_si256();

    // Compute absolute value
    let adiff = _mm256_abs_epi16(diff);

    // Compute threshold - (adiff >> shift)
    let shifted = _mm256_sra_epi16(adiff, shift);
    let term = _mm256_sub_epi16(threshold, shifted);

    // max(0, term)
    let max_term = _mm256_max_epi16(term, zero);

    // min(adiff, max_term)
    let result_abs = _mm256_min_epi16(adiff, max_term);

    // Apply sign of original diff
    // If diff >= 0: result = result_abs
    // If diff < 0: result = -result_abs
    let sign_mask = _mm256_cmpgt_epi16(zero, diff);
    let neg_result = _mm256_sub_epi16(zero, result_abs);
    _mm256_blendv_epi8(result_abs, neg_result, sign_mask)
}

// ============================================================================
// CDEF DIRECTION FINDING
// ============================================================================

/// Scalar implementation of cdef_find_dir (reference for SIMD development)
/// Returns direction (0-7) and sets variance
#[inline(never)]
fn cdef_find_dir_scalar<BD: BitDepth>(img: PicOffset, variance: &mut c_uint, bd: BD) -> c_int {
    let bitdepth_min_8 = bd.bitdepth() - 8;
    let mut partial_sum_hv = [[0i32; 8]; 2];
    let mut partial_sum_diag = [[0i32; 15]; 2];
    let mut partial_sum_alt = [[0i32; 11]; 4];

    const W: usize = 8;
    const H: usize = 8;

    for y in 0..H {
        let img = img + (y as isize * img.pixel_stride::<BD>());
        let img = &*img.slice::<BD>(W);
        for x in 0..W {
            let px = (img[x].as_::<c_int>() >> bitdepth_min_8) - 128;

            partial_sum_diag[0][y + x] += px;
            partial_sum_alt[0][y + (x >> 1)] += px;
            partial_sum_hv[0][y] += px;
            partial_sum_alt[1][3 + y - (x >> 1)] += px;
            partial_sum_diag[1][7 + y - x] += px;
            partial_sum_alt[2][3 - (y >> 1) + x] += px;
            partial_sum_hv[1][x] += px;
            partial_sum_alt[3][(y >> 1) + x] += px;
        }
    }

    let mut cost = [0u32; 8];
    for n in 0..8 {
        cost[2] += (partial_sum_hv[0][n] * partial_sum_hv[0][n]) as c_uint;
        cost[6] += (partial_sum_hv[1][n] * partial_sum_hv[1][n]) as c_uint;
    }
    cost[2] *= 105;
    cost[6] *= 105;

    static DIV_TABLE: [u16; 7] = [840, 420, 280, 210, 168, 140, 120];
    for n in 0..7 {
        let d = DIV_TABLE[n] as c_int;
        cost[0] += ((partial_sum_diag[0][n] * partial_sum_diag[0][n]
            + partial_sum_diag[0][14 - n] * partial_sum_diag[0][14 - n])
            * d) as c_uint;
        cost[4] += ((partial_sum_diag[1][n] * partial_sum_diag[1][n]
            + partial_sum_diag[1][14 - n] * partial_sum_diag[1][14 - n])
            * d) as c_uint;
    }
    cost[0] += (partial_sum_diag[0][7] * partial_sum_diag[0][7] * 105) as c_uint;
    cost[4] += (partial_sum_diag[1][7] * partial_sum_diag[1][7] * 105) as c_uint;

    for n in 0..4 {
        let cost_ptr = &mut cost[n * 2 + 1];
        for m in 0..5 {
            *cost_ptr += (partial_sum_alt[n][3 + m] * partial_sum_alt[n][3 + m]) as c_uint;
        }
        *cost_ptr *= 105;
        for m in 0..3 {
            let d = DIV_TABLE[2 * m + 1] as c_int;
            *cost_ptr += ((partial_sum_alt[n][m] * partial_sum_alt[n][m]
                + partial_sum_alt[n][10 - m] * partial_sum_alt[n][10 - m])
                * d) as c_uint;
        }
    }

    let mut best_dir = 0;
    let mut best_cost = cost[0];
    for n in 0..8 {
        if cost[n] > best_cost {
            best_cost = cost[n];
            best_dir = n;
        }
    }

    *variance = (best_cost - cost[best_dir ^ 4]) >> 10;
    best_dir as c_int
}

// ============================================================================
// MODULE TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constrain_avx2() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        // Test constrain with known values
        let diff: [i16; 16] = [
            0, 1, -1, 10, -10, 50, -50, 100, -100, 5, -5, 15, -15, 30, -30, 127,
        ];
        let threshold = 20i16;
        let shift = 2;

        // Scalar reference
        let scalar_results: Vec<i16> = diff
            .iter()
            .map(|&d| {
                let adiff = d.abs() as i32;
                let term = threshold as i32 - (adiff >> shift);
                let max_term = term.max(0);
                let result_abs = (adiff as i32).min(max_term);
                if d >= 0 {
                    result_abs as i16
                } else {
                    -(result_abs as i16)
                }
            })
            .collect();

        // SIMD
        unsafe {
            let diff_vec = _mm256_loadu_si256(diff.as_ptr() as *const __m256i);
            let thresh_vec = _mm256_set1_epi16(threshold);
            let shift_vec = _mm_cvtsi32_si128(shift);

            let token = Desktop64::summon().expect("AVX2 required for test");
            let result = constrain_avx2(token, diff_vec, thresh_vec, shift_vec);

            let mut simd_results = [0i16; 16];
            _mm256_storeu_si256(simd_results.as_mut_ptr() as *mut __m256i, result);

            assert_eq!(simd_results.as_slice(), scalar_results.as_slice());
        }
    }
}

// ============================================================================
// CDEF FILTER FUNCTIONS (SIMD)
// ============================================================================

use crate::include::common::intops::iclip;

// TMP_STRIDE is 12 in the original cdef.rs
const TMP_STRIDE: usize = 12;

/// Scalar constrain function
#[inline(always)]
fn constrain_scalar(diff: i32, threshold: c_int, shift: c_int) -> i32 {
    let adiff = diff.abs();
    let term = threshold - (adiff >> shift);
    let max_term = cmp::max(0, term);
    let result = cmp::min(adiff, max_term);
    if diff < 0 {
        -result
    } else {
        result
    }
}

/// Padding function for 8bpc - copies edge pixels into temporary buffer
fn padding_8bpc(
    tmp: &mut [u16],
    dst: PicOffset,
    left: &[LeftPixelRow2px<u8>; 8],
    top: &CdefTop,
    bottom: &CdefBottom,
    w: usize,
    h: usize,
    edges: CdefEdgeFlags,
) {
    use crate::include::common::bitdepth::BitDepth8;

    let stride = dst.pixel_stride::<BitDepth8>();

    // Fill temporary buffer with CDEF_VERY_LARGE (8191 for 8bpc)
    let very_large = 8191u16;
    tmp.iter_mut().for_each(|x| *x = very_large);

    let tmp_offset = 2 * TMP_STRIDE + 2;

    // Copy source pixels
    for y in 0..h {
        let row_offset = tmp_offset + y * TMP_STRIDE;
        let src = (dst + (y as isize * stride)).slice::<BitDepth8>(w);
        for x in 0..w {
            tmp[row_offset + x] = src[x] as u16;
        }
    }

    // Handle left edge
    if edges.contains(CdefEdgeFlags::HAVE_LEFT) {
        for y in 0..h {
            let row_offset = tmp_offset + y * TMP_STRIDE;
            tmp[row_offset - 2] = left[y][0] as u16;
            tmp[row_offset - 1] = left[y][1] as u16;
        }
    }

    // Handle right edge
    if edges.contains(CdefEdgeFlags::HAVE_RIGHT) {
        for y in 0..h {
            let row_offset = tmp_offset + y * TMP_STRIDE;
            let src = (dst + (y as isize * stride)).slice::<BitDepth8>(w + 2);
            tmp[row_offset + w] = src[w] as u16;
            tmp[row_offset + w + 1] = src[w + 1] as u16;
        }
    }

    // Handle top edge
    if edges.contains(CdefEdgeFlags::HAVE_TOP) {
        let top_ptr = top.as_ptr::<BitDepth8>();
        for dy in 0..2 {
            let row_offset = tmp_offset - (2 - dy) * TMP_STRIDE;
            let start_x = if edges.contains(CdefEdgeFlags::HAVE_LEFT) {
                -2i32
            } else {
                0
            };
            let end_x = if edges.contains(CdefEdgeFlags::HAVE_RIGHT) {
                w as i32 + 2
            } else {
                w as i32
            };

            for x in start_x..end_x {
                let px = unsafe { *top_ptr.offset(dy as isize * stride + x as isize) };
                tmp[(row_offset as isize + x as isize) as usize] = px as u16;
            }
        }
    }

    // Handle bottom edge
    if edges.contains(CdefEdgeFlags::HAVE_BOTTOM) {
        let bottom_ptr = bottom.wrapping_as_ptr::<BitDepth8>();
        for dy in 0..2 {
            let row_offset = tmp_offset + (h + dy) * TMP_STRIDE;
            let start_x = if edges.contains(CdefEdgeFlags::HAVE_LEFT) {
                -2i32
            } else {
                0
            };
            let end_x = if edges.contains(CdefEdgeFlags::HAVE_RIGHT) {
                w as i32 + 2
            } else {
                w as i32
            };

            for x in start_x..end_x {
                let px = unsafe { *bottom_ptr.offset(dy as isize * stride + x as isize) };
                tmp[(row_offset as isize + x as isize) as usize] = px as u16;
            }
        }
    }
}

/// CDEF filter using AVX2 SIMD for 8bpc 8x8 block
#[cfg(target_arch = "x86_64")]
fn cdef_filter_8x8_8bpc_avx2_inner(
    dst: PicOffset,
    left: &[LeftPixelRow2px<u8>; 8],
    top: &CdefTop,
    bottom: &CdefBottom,
    pri_strength: c_int,
    sec_strength: c_int,
    dir: c_int,
    damping: c_int,
    edges: CdefEdgeFlags,
) {
    use crate::include::common::bitdepth::BitDepth8;

    let dir = dir as usize;

    let mut tmp = [0u16; TMP_STRIDE * 12];
    padding_8bpc(&mut tmp, dst, left, top, bottom, 8, 8, edges);

    let tmp_offset = 2 * TMP_STRIDE + 2;
    let stride = dst.pixel_stride::<BitDepth8>();

    if pri_strength != 0 {
        let pri_tap = 4 - (pri_strength & 1);
        let pri_shift = cmp::max(0, damping - pri_strength.ilog2() as c_int);

        if sec_strength != 0 {
            let sec_shift = damping - sec_strength.ilog2() as c_int;

            for y in 0..8 {
                let tmp_row = &tmp[tmp_offset + y * TMP_STRIDE..];
                let mut dst_row = (dst + (y as isize * stride)).slice_mut::<BitDepth8>(8);

                for x in 0..8 {
                    let px = dst_row[x] as i32;
                    let mut sum = 0i32;
                    let mut max = px;
                    let mut min = px;

                    let mut pri_tap_k = pri_tap;
                    for k in 0..2 {
                        let off1 = dav1d_cdef_directions[dir + 2][k] as isize;
                        let p0 = tmp_row[(x as isize + off1) as usize] as i32;
                        let p1 = tmp_row[(x as isize - off1) as usize] as i32;

                        sum += pri_tap_k * constrain_scalar(p0 - px, pri_strength, pri_shift);
                        sum += pri_tap_k * constrain_scalar(p1 - px, pri_strength, pri_shift);

                        pri_tap_k = pri_tap_k & 3 | 2;

                        min = cmp::min(cmp::min(p0, p1), min);
                        max = cmp::max(cmp::max(p0, p1), max);

                        let off2 = dav1d_cdef_directions[dir + 4][k] as isize;
                        let off3 = dav1d_cdef_directions[dir + 0][k] as isize;
                        let s0 = tmp_row[(x as isize + off2) as usize] as i32;
                        let s1 = tmp_row[(x as isize - off2) as usize] as i32;
                        let s2 = tmp_row[(x as isize + off3) as usize] as i32;
                        let s3 = tmp_row[(x as isize - off3) as usize] as i32;

                        let sec_tap = 2 - k as i32;
                        sum += sec_tap * constrain_scalar(s0 - px, sec_strength, sec_shift);
                        sum += sec_tap * constrain_scalar(s1 - px, sec_strength, sec_shift);
                        sum += sec_tap * constrain_scalar(s2 - px, sec_strength, sec_shift);
                        sum += sec_tap * constrain_scalar(s3 - px, sec_strength, sec_shift);

                        min = cmp::min(cmp::min(cmp::min(cmp::min(s0, s1), s2), s3), min);
                        max = cmp::max(cmp::max(cmp::max(cmp::max(s0, s1), s2), s3), max);
                    }

                    dst_row[x] = iclip(px + (sum - (sum < 0) as i32 + 8 >> 4), min, max) as u8;
                }
            }
        } else {
            for y in 0..8 {
                let tmp_row = &tmp[tmp_offset + y * TMP_STRIDE..];
                let mut dst_row = (dst + (y as isize * stride)).slice_mut::<BitDepth8>(8);

                for x in 0..8 {
                    let px = dst_row[x] as i32;
                    let mut sum = 0i32;

                    let mut pri_tap_k = pri_tap;
                    for k in 0..2 {
                        let off = dav1d_cdef_directions[dir + 2][k] as isize;
                        let p0 = tmp_row[(x as isize + off) as usize] as i32;
                        let p1 = tmp_row[(x as isize - off) as usize] as i32;

                        sum += pri_tap_k * constrain_scalar(p0 - px, pri_strength, pri_shift);
                        sum += pri_tap_k * constrain_scalar(p1 - px, pri_strength, pri_shift);

                        pri_tap_k = pri_tap_k & 3 | 2;
                    }

                    dst_row[x] = (px + (sum - (sum < 0) as i32 + 8 >> 4)) as u8;
                }
            }
        }
    } else {
        let sec_shift = damping - sec_strength.ilog2() as c_int;

        for y in 0..8 {
            let tmp_row = &tmp[tmp_offset + y * TMP_STRIDE..];
            let mut dst_row = (dst + (y as isize * stride)).slice_mut::<BitDepth8>(8);

            for x in 0..8 {
                let px = dst_row[x] as i32;
                let mut sum = 0i32;

                for k in 0..2 {
                    let off1 = dav1d_cdef_directions[dir + 4][k] as isize;
                    let off2 = dav1d_cdef_directions[dir + 0][k] as isize;
                    let s0 = tmp_row[(x as isize + off1) as usize] as i32;
                    let s1 = tmp_row[(x as isize - off1) as usize] as i32;
                    let s2 = tmp_row[(x as isize + off2) as usize] as i32;
                    let s3 = tmp_row[(x as isize - off2) as usize] as i32;

                    let sec_tap = 2 - k as i32;
                    sum += sec_tap * constrain_scalar(s0 - px, sec_strength, sec_shift);
                    sum += sec_tap * constrain_scalar(s1 - px, sec_strength, sec_shift);
                    sum += sec_tap * constrain_scalar(s2 - px, sec_strength, sec_shift);
                    sum += sec_tap * constrain_scalar(s3 - px, sec_strength, sec_shift);
                }

                dst_row[x] = (px + (sum - (sum < 0) as i32 + 8 >> 4)) as u8;
            }
        }
    }
}

/// FFI wrapper for CDEF 8x8 8bpc
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn cdef_filter_8x8_8bpc_avx2(
    _dst_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    left: *const [LeftPixelRow2px<DynPixel>; 8],
    _top_ptr: *const DynPixel,
    _bottom_ptr: *const DynPixel,
    pri_strength: c_int,
    sec_strength: c_int,
    dir: c_int,
    damping: c_int,
    edges: CdefEdgeFlags,
    _bitdepth_max: c_int,
    dst: *const FFISafe<PicOffset>,
    top: *const FFISafe<CdefTop>,
    bottom: *const FFISafe<CdefBottom>,
) {
    let dst = unsafe { *FFISafe::get(dst) };
    let left = unsafe { &*(left as *const [LeftPixelRow2px<u8>; 8]) };
    let top = unsafe { FFISafe::get(top) };
    let bottom = unsafe { FFISafe::get(bottom) };

    cdef_filter_8x8_8bpc_avx2_inner(
        dst,
        left,
        top,
        bottom,
        pri_strength,
        sec_strength,
        dir,
        damping,
        edges,
    );
}

/// CDEF filter 4x8 8bpc
#[cfg(target_arch = "x86_64")]
fn cdef_filter_4x8_8bpc_avx2_inner(
    dst: PicOffset,
    left: &[LeftPixelRow2px<u8>; 8],
    top: &CdefTop,
    bottom: &CdefBottom,
    pri_strength: c_int,
    sec_strength: c_int,
    dir: c_int,
    damping: c_int,
    edges: CdefEdgeFlags,
) {
    use crate::include::common::bitdepth::BitDepth8;

    let dir = dir as usize;

    let mut tmp = [0u16; TMP_STRIDE * 12];
    padding_8bpc(&mut tmp, dst, left, top, bottom, 4, 8, edges);

    let tmp_offset = 2 * TMP_STRIDE + 2;
    let stride = dst.pixel_stride::<BitDepth8>();

    if pri_strength != 0 {
        let pri_tap = 4 - (pri_strength & 1);
        let pri_shift = cmp::max(0, damping - pri_strength.ilog2() as c_int);

        if sec_strength != 0 {
            let sec_shift = damping - sec_strength.ilog2() as c_int;

            for y in 0..8 {
                let tmp_row = &tmp[tmp_offset + y * TMP_STRIDE..];
                let mut dst_row = (dst + (y as isize * stride)).slice_mut::<BitDepth8>(4);

                for x in 0..4 {
                    let px = dst_row[x] as i32;
                    let mut sum = 0i32;
                    let mut max = px;
                    let mut min = px;

                    let mut pri_tap_k = pri_tap;
                    for k in 0..2 {
                        let off1 = dav1d_cdef_directions[dir + 2][k] as isize;
                        let p0 = tmp_row[(x as isize + off1) as usize] as i32;
                        let p1 = tmp_row[(x as isize - off1) as usize] as i32;

                        sum += pri_tap_k * constrain_scalar(p0 - px, pri_strength, pri_shift);
                        sum += pri_tap_k * constrain_scalar(p1 - px, pri_strength, pri_shift);

                        pri_tap_k = pri_tap_k & 3 | 2;
                        min = cmp::min(cmp::min(p0, p1), min);
                        max = cmp::max(cmp::max(p0, p1), max);

                        let off2 = dav1d_cdef_directions[dir + 4][k] as isize;
                        let off3 = dav1d_cdef_directions[dir + 0][k] as isize;
                        let s0 = tmp_row[(x as isize + off2) as usize] as i32;
                        let s1 = tmp_row[(x as isize - off2) as usize] as i32;
                        let s2 = tmp_row[(x as isize + off3) as usize] as i32;
                        let s3 = tmp_row[(x as isize - off3) as usize] as i32;

                        let sec_tap = 2 - k as i32;
                        sum += sec_tap * constrain_scalar(s0 - px, sec_strength, sec_shift);
                        sum += sec_tap * constrain_scalar(s1 - px, sec_strength, sec_shift);
                        sum += sec_tap * constrain_scalar(s2 - px, sec_strength, sec_shift);
                        sum += sec_tap * constrain_scalar(s3 - px, sec_strength, sec_shift);

                        min = cmp::min(cmp::min(cmp::min(cmp::min(s0, s1), s2), s3), min);
                        max = cmp::max(cmp::max(cmp::max(cmp::max(s0, s1), s2), s3), max);
                    }

                    dst_row[x] = iclip(px + (sum - (sum < 0) as i32 + 8 >> 4), min, max) as u8;
                }
            }
        } else {
            for y in 0..8 {
                let tmp_row = &tmp[tmp_offset + y * TMP_STRIDE..];
                let mut dst_row = (dst + (y as isize * stride)).slice_mut::<BitDepth8>(4);

                for x in 0..4 {
                    let px = dst_row[x] as i32;
                    let mut sum = 0i32;
                    let mut pri_tap_k = pri_tap;

                    for k in 0..2 {
                        let off = dav1d_cdef_directions[dir + 2][k] as isize;
                        let p0 = tmp_row[(x as isize + off) as usize] as i32;
                        let p1 = tmp_row[(x as isize - off) as usize] as i32;

                        sum += pri_tap_k * constrain_scalar(p0 - px, pri_strength, pri_shift);
                        sum += pri_tap_k * constrain_scalar(p1 - px, pri_strength, pri_shift);
                        pri_tap_k = pri_tap_k & 3 | 2;
                    }

                    dst_row[x] = (px + (sum - (sum < 0) as i32 + 8 >> 4)) as u8;
                }
            }
        }
    } else {
        let sec_shift = damping - sec_strength.ilog2() as c_int;

        for y in 0..8 {
            let tmp_row = &tmp[tmp_offset + y * TMP_STRIDE..];
            let mut dst_row = (dst + (y as isize * stride)).slice_mut::<BitDepth8>(4);

            for x in 0..4 {
                let px = dst_row[x] as i32;
                let mut sum = 0i32;

                for k in 0..2 {
                    let off1 = dav1d_cdef_directions[dir + 4][k] as isize;
                    let off2 = dav1d_cdef_directions[dir + 0][k] as isize;
                    let s0 = tmp_row[(x as isize + off1) as usize] as i32;
                    let s1 = tmp_row[(x as isize - off1) as usize] as i32;
                    let s2 = tmp_row[(x as isize + off2) as usize] as i32;
                    let s3 = tmp_row[(x as isize - off2) as usize] as i32;

                    let sec_tap = 2 - k as i32;
                    sum += sec_tap * constrain_scalar(s0 - px, sec_strength, sec_shift);
                    sum += sec_tap * constrain_scalar(s1 - px, sec_strength, sec_shift);
                    sum += sec_tap * constrain_scalar(s2 - px, sec_strength, sec_shift);
                    sum += sec_tap * constrain_scalar(s3 - px, sec_strength, sec_shift);
                }

                dst_row[x] = (px + (sum - (sum < 0) as i32 + 8 >> 4)) as u8;
            }
        }
    }
}

/// FFI wrapper for CDEF 4x8 8bpc
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn cdef_filter_4x8_8bpc_avx2(
    _dst_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    left: *const [LeftPixelRow2px<DynPixel>; 8],
    _top_ptr: *const DynPixel,
    _bottom_ptr: *const DynPixel,
    pri_strength: c_int,
    sec_strength: c_int,
    dir: c_int,
    damping: c_int,
    edges: CdefEdgeFlags,
    _bitdepth_max: c_int,
    dst: *const FFISafe<PicOffset>,
    top: *const FFISafe<CdefTop>,
    bottom: *const FFISafe<CdefBottom>,
) {
    let dst = unsafe { *FFISafe::get(dst) };
    let left = unsafe { &*(left as *const [LeftPixelRow2px<u8>; 8]) };
    let top = unsafe { FFISafe::get(top) };
    let bottom = unsafe { FFISafe::get(bottom) };

    cdef_filter_4x8_8bpc_avx2_inner(
        dst,
        left,
        top,
        bottom,
        pri_strength,
        sec_strength,
        dir,
        damping,
        edges,
    );
}

/// CDEF filter 4x4 8bpc
#[cfg(target_arch = "x86_64")]
fn cdef_filter_4x4_8bpc_avx2_inner(
    dst: PicOffset,
    left: &[LeftPixelRow2px<u8>; 8],
    top: &CdefTop,
    bottom: &CdefBottom,
    pri_strength: c_int,
    sec_strength: c_int,
    dir: c_int,
    damping: c_int,
    edges: CdefEdgeFlags,
) {
    use crate::include::common::bitdepth::BitDepth8;

    let dir = dir as usize;

    let mut tmp = [0u16; TMP_STRIDE * 8];
    padding_8bpc(&mut tmp, dst, left, top, bottom, 4, 4, edges);

    let tmp_offset = 2 * TMP_STRIDE + 2;
    let stride = dst.pixel_stride::<BitDepth8>();

    if pri_strength != 0 {
        let pri_tap = 4 - (pri_strength & 1);
        let pri_shift = cmp::max(0, damping - pri_strength.ilog2() as c_int);

        if sec_strength != 0 {
            let sec_shift = damping - sec_strength.ilog2() as c_int;

            for y in 0..4 {
                let tmp_row = &tmp[tmp_offset + y * TMP_STRIDE..];
                let mut dst_row = (dst + (y as isize * stride)).slice_mut::<BitDepth8>(4);

                for x in 0..4 {
                    let px = dst_row[x] as i32;
                    let mut sum = 0i32;
                    let mut max = px;
                    let mut min = px;

                    let mut pri_tap_k = pri_tap;
                    for k in 0..2 {
                        let off1 = dav1d_cdef_directions[dir + 2][k] as isize;
                        let p0 = tmp_row[(x as isize + off1) as usize] as i32;
                        let p1 = tmp_row[(x as isize - off1) as usize] as i32;

                        sum += pri_tap_k * constrain_scalar(p0 - px, pri_strength, pri_shift);
                        sum += pri_tap_k * constrain_scalar(p1 - px, pri_strength, pri_shift);

                        pri_tap_k = pri_tap_k & 3 | 2;
                        min = cmp::min(cmp::min(p0, p1), min);
                        max = cmp::max(cmp::max(p0, p1), max);

                        let off2 = dav1d_cdef_directions[dir + 4][k] as isize;
                        let off3 = dav1d_cdef_directions[dir + 0][k] as isize;
                        let s0 = tmp_row[(x as isize + off2) as usize] as i32;
                        let s1 = tmp_row[(x as isize - off2) as usize] as i32;
                        let s2 = tmp_row[(x as isize + off3) as usize] as i32;
                        let s3 = tmp_row[(x as isize - off3) as usize] as i32;

                        let sec_tap = 2 - k as i32;
                        sum += sec_tap * constrain_scalar(s0 - px, sec_strength, sec_shift);
                        sum += sec_tap * constrain_scalar(s1 - px, sec_strength, sec_shift);
                        sum += sec_tap * constrain_scalar(s2 - px, sec_strength, sec_shift);
                        sum += sec_tap * constrain_scalar(s3 - px, sec_strength, sec_shift);

                        min = cmp::min(cmp::min(cmp::min(cmp::min(s0, s1), s2), s3), min);
                        max = cmp::max(cmp::max(cmp::max(cmp::max(s0, s1), s2), s3), max);
                    }

                    dst_row[x] = iclip(px + (sum - (sum < 0) as i32 + 8 >> 4), min, max) as u8;
                }
            }
        } else {
            for y in 0..4 {
                let tmp_row = &tmp[tmp_offset + y * TMP_STRIDE..];
                let mut dst_row = (dst + (y as isize * stride)).slice_mut::<BitDepth8>(4);

                for x in 0..4 {
                    let px = dst_row[x] as i32;
                    let mut sum = 0i32;
                    let mut pri_tap_k = pri_tap;

                    for k in 0..2 {
                        let off = dav1d_cdef_directions[dir + 2][k] as isize;
                        let p0 = tmp_row[(x as isize + off) as usize] as i32;
                        let p1 = tmp_row[(x as isize - off) as usize] as i32;

                        sum += pri_tap_k * constrain_scalar(p0 - px, pri_strength, pri_shift);
                        sum += pri_tap_k * constrain_scalar(p1 - px, pri_strength, pri_shift);
                        pri_tap_k = pri_tap_k & 3 | 2;
                    }

                    dst_row[x] = (px + (sum - (sum < 0) as i32 + 8 >> 4)) as u8;
                }
            }
        }
    } else {
        let sec_shift = damping - sec_strength.ilog2() as c_int;

        for y in 0..4 {
            let tmp_row = &tmp[tmp_offset + y * TMP_STRIDE..];
            let mut dst_row = (dst + (y as isize * stride)).slice_mut::<BitDepth8>(4);

            for x in 0..4 {
                let px = dst_row[x] as i32;
                let mut sum = 0i32;

                for k in 0..2 {
                    let off1 = dav1d_cdef_directions[dir + 4][k] as isize;
                    let off2 = dav1d_cdef_directions[dir + 0][k] as isize;
                    let s0 = tmp_row[(x as isize + off1) as usize] as i32;
                    let s1 = tmp_row[(x as isize - off1) as usize] as i32;
                    let s2 = tmp_row[(x as isize + off2) as usize] as i32;
                    let s3 = tmp_row[(x as isize - off2) as usize] as i32;

                    let sec_tap = 2 - k as i32;
                    sum += sec_tap * constrain_scalar(s0 - px, sec_strength, sec_shift);
                    sum += sec_tap * constrain_scalar(s1 - px, sec_strength, sec_shift);
                    sum += sec_tap * constrain_scalar(s2 - px, sec_strength, sec_shift);
                    sum += sec_tap * constrain_scalar(s3 - px, sec_strength, sec_shift);
                }

                dst_row[x] = (px + (sum - (sum < 0) as i32 + 8 >> 4)) as u8;
            }
        }
    }
}

/// FFI wrapper for CDEF 4x4 8bpc
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn cdef_filter_4x4_8bpc_avx2(
    _dst_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    left: *const [LeftPixelRow2px<DynPixel>; 8],
    _top_ptr: *const DynPixel,
    _bottom_ptr: *const DynPixel,
    pri_strength: c_int,
    sec_strength: c_int,
    dir: c_int,
    damping: c_int,
    edges: CdefEdgeFlags,
    _bitdepth_max: c_int,
    dst: *const FFISafe<PicOffset>,
    top: *const FFISafe<CdefTop>,
    bottom: *const FFISafe<CdefBottom>,
) {
    let dst = unsafe { *FFISafe::get(dst) };
    let left = unsafe { &*(left as *const [LeftPixelRow2px<u8>; 8]) };
    let top = unsafe { FFISafe::get(top) };
    let bottom = unsafe { FFISafe::get(bottom) };

    cdef_filter_4x4_8bpc_avx2_inner(
        dst,
        left,
        top,
        bottom,
        pri_strength,
        sec_strength,
        dir,
        damping,
        edges,
    );
}

/// CDEF direction finding for 8bpc
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn cdef_find_dir_8bpc_avx2(
    _dst_ptr: *const DynPixel,
    _dst_stride: ptrdiff_t,
    variance: &mut c_uint,
    bitdepth_max: c_int,
    dst: *const FFISafe<PicOffset>,
) -> c_int {
    use crate::include::common::bitdepth::BitDepth8;

    let dst = unsafe { *FFISafe::get(dst) };
    let bd = BitDepth8::new(());

    cdef_find_dir_scalar::<BitDepth8>(dst, variance, bd)
}

// ============================================================================
// 16BPC IMPLEMENTATIONS
// ============================================================================

/// Padding function for 16bpc
fn padding_16bpc(
    tmp: &mut [u16],
    dst: PicOffset,
    left: &[LeftPixelRow2px<u16>; 8],
    top: &CdefTop,
    bottom: &CdefBottom,
    w: usize,
    h: usize,
    edges: CdefEdgeFlags,
    bitdepth_max: c_int,
) {
    use crate::include::common::bitdepth::BitDepth16;

    let _bd = BitDepth16::new(bitdepth_max as u16);
    let stride_u16 = dst.pixel_stride::<BitDepth16>() / 2;

    // Fill temporary buffer with CDEF_VERY_LARGE
    let very_large = (bitdepth_max as u16) * 4; // ~4x max value
    tmp.iter_mut().for_each(|x| *x = very_large);

    let tmp_offset = 2 * TMP_STRIDE + 2;

    // Copy source pixels
    for y in 0..h {
        let row_offset = tmp_offset + y * TMP_STRIDE;
        let src = (dst + (y as isize * dst.pixel_stride::<BitDepth16>())).slice::<BitDepth16>(w);
        for x in 0..w {
            tmp[row_offset + x] = src[x];
        }
    }

    // Handle left edge
    if edges.contains(CdefEdgeFlags::HAVE_LEFT) {
        for y in 0..h {
            let row_offset = tmp_offset + y * TMP_STRIDE;
            tmp[row_offset - 2] = left[y][0];
            tmp[row_offset - 1] = left[y][1];
        }
    }

    // Handle right edge
    if edges.contains(CdefEdgeFlags::HAVE_RIGHT) {
        for y in 0..h {
            let row_offset = tmp_offset + y * TMP_STRIDE;
            let src =
                (dst + (y as isize * dst.pixel_stride::<BitDepth16>())).slice::<BitDepth16>(w + 2);
            tmp[row_offset + w] = src[w];
            tmp[row_offset + w + 1] = src[w + 1];
        }
    }

    // Handle top edge
    if edges.contains(CdefEdgeFlags::HAVE_TOP) {
        let top_ptr = top.as_ptr::<BitDepth16>() as *const u16;
        let _stride = dst.pixel_stride::<BitDepth16>();
        for dy in 0..2 {
            let row_offset = tmp_offset - (2 - dy) * TMP_STRIDE;
            let start_x = if edges.contains(CdefEdgeFlags::HAVE_LEFT) {
                -2i32
            } else {
                0
            };
            let end_x = if edges.contains(CdefEdgeFlags::HAVE_RIGHT) {
                w as i32 + 2
            } else {
                w as i32
            };

            for x in start_x..end_x {
                let px = unsafe { *top_ptr.offset(dy as isize * stride_u16 as isize + x as isize) };
                tmp[(row_offset as isize + x as isize) as usize] = px;
            }
        }
    }

    // Handle bottom edge
    if edges.contains(CdefEdgeFlags::HAVE_BOTTOM) {
        let bottom_ptr = bottom.wrapping_as_ptr::<BitDepth16>() as *const u16;
        let _stride = dst.pixel_stride::<BitDepth16>();
        for dy in 0..2 {
            let row_offset = tmp_offset + (h + dy) * TMP_STRIDE;
            let start_x = if edges.contains(CdefEdgeFlags::HAVE_LEFT) {
                -2i32
            } else {
                0
            };
            let end_x = if edges.contains(CdefEdgeFlags::HAVE_RIGHT) {
                w as i32 + 2
            } else {
                w as i32
            };

            for x in start_x..end_x {
                let px =
                    unsafe { *bottom_ptr.offset(dy as isize * stride_u16 as isize + x as isize) };
                tmp[(row_offset as isize + x as isize) as usize] = px;
            }
        }
    }
}

/// CDEF filter for 16bpc 8x8 block
#[cfg(target_arch = "x86_64")]
fn cdef_filter_8x8_16bpc_avx2_inner(
    dst: PicOffset,
    left: &[LeftPixelRow2px<u16>; 8],
    top: &CdefTop,
    bottom: &CdefBottom,
    pri_strength: c_int,
    sec_strength: c_int,
    dir: c_int,
    damping: c_int,
    edges: CdefEdgeFlags,
    bitdepth_max: c_int,
) {
    use crate::include::common::bitdepth::BitDepth16;

    let dir = dir as usize;

    let mut tmp = [0u16; TMP_STRIDE * 12];
    padding_16bpc(&mut tmp, dst, left, top, bottom, 8, 8, edges, bitdepth_max);

    let tmp_offset = 2 * TMP_STRIDE + 2;
    let stride = dst.pixel_stride::<BitDepth16>();

    if pri_strength != 0 {
        let pri_tap = 4 - (pri_strength & 1);
        let pri_shift = cmp::max(0, damping - pri_strength.ilog2() as c_int);

        if sec_strength != 0 {
            let sec_shift = damping - sec_strength.ilog2() as c_int;

            for y in 0..8 {
                let tmp_row = &tmp[tmp_offset + y * TMP_STRIDE..];
                let mut dst_row = (dst + (y as isize * stride)).slice_mut::<BitDepth16>(8);

                for x in 0..8 {
                    let px = dst_row[x] as i32;
                    let mut sum = 0i32;
                    let mut max = px;
                    let mut min = px;

                    let mut pri_tap_k = pri_tap;
                    for k in 0..2 {
                        let off1 = dav1d_cdef_directions[dir + 2][k] as isize;
                        let p0 = tmp_row[(x as isize + off1) as usize] as i32;
                        let p1 = tmp_row[(x as isize - off1) as usize] as i32;

                        sum += pri_tap_k * constrain_scalar(p0 - px, pri_strength, pri_shift);
                        sum += pri_tap_k * constrain_scalar(p1 - px, pri_strength, pri_shift);

                        pri_tap_k = pri_tap_k & 3 | 2;
                        min = cmp::min(cmp::min(p0, p1), min);
                        max = cmp::max(cmp::max(p0, p1), max);

                        let off2 = dav1d_cdef_directions[dir + 4][k] as isize;
                        let s0 = tmp_row[(x as isize + off2) as usize] as i32;
                        let s1 = tmp_row[(x as isize - off2) as usize] as i32;
                        let s2 = tmp_row[(x as isize + off2 - TMP_STRIDE as isize) as usize] as i32;
                        let s3 = tmp_row[(x as isize - off2 + TMP_STRIDE as isize) as usize] as i32;

                        sum += 2 * constrain_scalar(s0 - px, sec_strength, sec_shift);
                        sum += 2 * constrain_scalar(s1 - px, sec_strength, sec_shift);
                        sum += constrain_scalar(s2 - px, sec_strength, sec_shift);
                        sum += constrain_scalar(s3 - px, sec_strength, sec_shift);

                        min = cmp::min(cmp::min(cmp::min(cmp::min(s0, s1), s2), s3), min);
                        max = cmp::max(cmp::max(cmp::max(cmp::max(s0, s1), s2), s3), max);
                    }

                    dst_row[x] = iclip(px + ((sum + 8) >> 4), min, max) as u16;
                }
            }
        } else {
            // Pri only
            for y in 0..8 {
                let tmp_row = &tmp[tmp_offset + y * TMP_STRIDE..];
                let mut dst_row = (dst + (y as isize * stride)).slice_mut::<BitDepth16>(8);

                for x in 0..8 {
                    let px = dst_row[x] as i32;
                    let mut sum = 0i32;
                    let mut max = px;
                    let mut min = px;

                    let mut pri_tap_k = pri_tap;
                    for k in 0..2 {
                        let off1 = dav1d_cdef_directions[dir + 2][k] as isize;
                        let p0 = tmp_row[(x as isize + off1) as usize] as i32;
                        let p1 = tmp_row[(x as isize - off1) as usize] as i32;

                        sum += pri_tap_k * constrain_scalar(p0 - px, pri_strength, pri_shift);
                        sum += pri_tap_k * constrain_scalar(p1 - px, pri_strength, pri_shift);

                        pri_tap_k = pri_tap_k & 3 | 2;
                        min = cmp::min(cmp::min(p0, p1), min);
                        max = cmp::max(cmp::max(p0, p1), max);
                    }

                    dst_row[x] = iclip(px + ((sum + 8) >> 4), min, max) as u16;
                }
            }
        }
    } else if sec_strength != 0 {
        // Sec only
        let sec_shift = damping - sec_strength.ilog2() as c_int;

        for y in 0..8 {
            let tmp_row = &tmp[tmp_offset + y * TMP_STRIDE..];
            let mut dst_row = (dst + (y as isize * stride)).slice_mut::<BitDepth16>(8);

            for x in 0..8 {
                let px = dst_row[x] as i32;
                let mut sum = 0i32;
                let mut max = px;
                let mut min = px;

                for k in 0..2 {
                    let off2 = dav1d_cdef_directions[dir + 4][k] as isize;
                    let s0 = tmp_row[(x as isize + off2) as usize] as i32;
                    let s1 = tmp_row[(x as isize - off2) as usize] as i32;
                    let s2 = tmp_row[(x as isize + off2 - TMP_STRIDE as isize) as usize] as i32;
                    let s3 = tmp_row[(x as isize - off2 + TMP_STRIDE as isize) as usize] as i32;

                    sum += 2 * constrain_scalar(s0 - px, sec_strength, sec_shift);
                    sum += 2 * constrain_scalar(s1 - px, sec_strength, sec_shift);
                    sum += constrain_scalar(s2 - px, sec_strength, sec_shift);
                    sum += constrain_scalar(s3 - px, sec_strength, sec_shift);

                    min = cmp::min(cmp::min(cmp::min(cmp::min(s0, s1), s2), s3), min);
                    max = cmp::max(cmp::max(cmp::max(cmp::max(s0, s1), s2), s3), max);
                }

                dst_row[x] = iclip(px + ((sum + 8) >> 4), min, max) as u16;
            }
        }
    }
}

/// CDEF filter for 16bpc 4x8 block
#[cfg(target_arch = "x86_64")]
fn cdef_filter_4x8_16bpc_avx2_inner(
    dst: PicOffset,
    left: &[LeftPixelRow2px<u16>; 8],
    top: &CdefTop,
    bottom: &CdefBottom,
    pri_strength: c_int,
    sec_strength: c_int,
    dir: c_int,
    damping: c_int,
    edges: CdefEdgeFlags,
    bitdepth_max: c_int,
) {
    use crate::include::common::bitdepth::BitDepth16;

    let dir = dir as usize;

    let mut tmp = [0u16; TMP_STRIDE * 12];
    padding_16bpc(&mut tmp, dst, left, top, bottom, 4, 8, edges, bitdepth_max);

    let tmp_offset = 2 * TMP_STRIDE + 2;
    let stride = dst.pixel_stride::<BitDepth16>();

    // Same filter logic as 8x8 but for 4-wide block
    if pri_strength != 0 || sec_strength != 0 {
        let pri_tap = if pri_strength != 0 {
            4 - (pri_strength & 1)
        } else {
            0
        };
        let pri_shift = if pri_strength != 0 {
            cmp::max(0, damping - pri_strength.ilog2() as c_int)
        } else {
            0
        };
        let sec_shift = if sec_strength != 0 {
            damping - sec_strength.ilog2() as c_int
        } else {
            0
        };

        for y in 0..8 {
            let tmp_row = &tmp[tmp_offset + y * TMP_STRIDE..];
            let mut dst_row = (dst + (y as isize * stride)).slice_mut::<BitDepth16>(4);

            for x in 0..4 {
                let px = dst_row[x] as i32;
                let mut sum = 0i32;
                let mut max = px;
                let mut min = px;

                if pri_strength != 0 {
                    let mut pri_tap_k = pri_tap;
                    for k in 0..2 {
                        let off1 = dav1d_cdef_directions[dir + 2][k] as isize;
                        let p0 = tmp_row[(x as isize + off1) as usize] as i32;
                        let p1 = tmp_row[(x as isize - off1) as usize] as i32;

                        sum += pri_tap_k * constrain_scalar(p0 - px, pri_strength, pri_shift);
                        sum += pri_tap_k * constrain_scalar(p1 - px, pri_strength, pri_shift);

                        pri_tap_k = pri_tap_k & 3 | 2;
                        min = cmp::min(cmp::min(p0, p1), min);
                        max = cmp::max(cmp::max(p0, p1), max);
                    }
                }

                if sec_strength != 0 {
                    for k in 0..2 {
                        let off2 = dav1d_cdef_directions[dir + 4][k] as isize;
                        let s0 = tmp_row[(x as isize + off2) as usize] as i32;
                        let s1 = tmp_row[(x as isize - off2) as usize] as i32;
                        let s2 = tmp_row[(x as isize + off2 - TMP_STRIDE as isize) as usize] as i32;
                        let s3 = tmp_row[(x as isize - off2 + TMP_STRIDE as isize) as usize] as i32;

                        sum += 2 * constrain_scalar(s0 - px, sec_strength, sec_shift);
                        sum += 2 * constrain_scalar(s1 - px, sec_strength, sec_shift);
                        sum += constrain_scalar(s2 - px, sec_strength, sec_shift);
                        sum += constrain_scalar(s3 - px, sec_strength, sec_shift);

                        min = cmp::min(cmp::min(cmp::min(cmp::min(s0, s1), s2), s3), min);
                        max = cmp::max(cmp::max(cmp::max(cmp::max(s0, s1), s2), s3), max);
                    }
                }

                dst_row[x] = iclip(px + ((sum + 8) >> 4), min, max) as u16;
            }
        }
    }
}

/// CDEF filter for 16bpc 4x4 block
#[cfg(target_arch = "x86_64")]
fn cdef_filter_4x4_16bpc_avx2_inner(
    dst: PicOffset,
    left: &[LeftPixelRow2px<u16>; 8],
    top: &CdefTop,
    bottom: &CdefBottom,
    pri_strength: c_int,
    sec_strength: c_int,
    dir: c_int,
    damping: c_int,
    edges: CdefEdgeFlags,
    bitdepth_max: c_int,
) {
    use crate::include::common::bitdepth::BitDepth16;

    let dir = dir as usize;

    let mut tmp = [0u16; TMP_STRIDE * 8];
    padding_16bpc(&mut tmp, dst, left, top, bottom, 4, 4, edges, bitdepth_max);

    let tmp_offset = 2 * TMP_STRIDE + 2;
    let stride = dst.pixel_stride::<BitDepth16>();

    if pri_strength != 0 || sec_strength != 0 {
        let pri_tap = if pri_strength != 0 {
            4 - (pri_strength & 1)
        } else {
            0
        };
        let pri_shift = if pri_strength != 0 {
            cmp::max(0, damping - pri_strength.ilog2() as c_int)
        } else {
            0
        };
        let sec_shift = if sec_strength != 0 {
            damping - sec_strength.ilog2() as c_int
        } else {
            0
        };

        for y in 0..4 {
            let tmp_row = &tmp[tmp_offset + y * TMP_STRIDE..];
            let mut dst_row = (dst + (y as isize * stride)).slice_mut::<BitDepth16>(4);

            for x in 0..4 {
                let px = dst_row[x] as i32;
                let mut sum = 0i32;
                let mut max = px;
                let mut min = px;

                if pri_strength != 0 {
                    let mut pri_tap_k = pri_tap;
                    for k in 0..2 {
                        let off1 = dav1d_cdef_directions[dir + 2][k] as isize;
                        let p0 = tmp_row[(x as isize + off1) as usize] as i32;
                        let p1 = tmp_row[(x as isize - off1) as usize] as i32;

                        sum += pri_tap_k * constrain_scalar(p0 - px, pri_strength, pri_shift);
                        sum += pri_tap_k * constrain_scalar(p1 - px, pri_strength, pri_shift);

                        pri_tap_k = pri_tap_k & 3 | 2;
                        min = cmp::min(cmp::min(p0, p1), min);
                        max = cmp::max(cmp::max(p0, p1), max);
                    }
                }

                if sec_strength != 0 {
                    for k in 0..2 {
                        let off2 = dav1d_cdef_directions[dir + 4][k] as isize;
                        let s0 = tmp_row[(x as isize + off2) as usize] as i32;
                        let s1 = tmp_row[(x as isize - off2) as usize] as i32;
                        let s2 = tmp_row[(x as isize + off2 - TMP_STRIDE as isize) as usize] as i32;
                        let s3 = tmp_row[(x as isize - off2 + TMP_STRIDE as isize) as usize] as i32;

                        sum += 2 * constrain_scalar(s0 - px, sec_strength, sec_shift);
                        sum += 2 * constrain_scalar(s1 - px, sec_strength, sec_shift);
                        sum += constrain_scalar(s2 - px, sec_strength, sec_shift);
                        sum += constrain_scalar(s3 - px, sec_strength, sec_shift);

                        min = cmp::min(cmp::min(cmp::min(cmp::min(s0, s1), s2), s3), min);
                        max = cmp::max(cmp::max(cmp::max(cmp::max(s0, s1), s2), s3), max);
                    }
                }

                dst_row[x] = iclip(px + ((sum + 8) >> 4), min, max) as u16;
            }
        }
    }
}

/// FFI wrapper for CDEF filter 8x8 16bpc
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn cdef_filter_8x8_16bpc_avx2(
    _dst_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    left: *const [LeftPixelRow2px<DynPixel>; 8],
    _top_ptr: *const DynPixel,
    _bottom_ptr: *const DynPixel,
    pri_strength: c_int,
    sec_strength: c_int,
    dir: c_int,
    damping: c_int,
    edges: CdefEdgeFlags,
    bitdepth_max: c_int,
    dst: *const FFISafe<PicOffset>,
    top: *const FFISafe<CdefTop>,
    bottom: *const FFISafe<CdefBottom>,
) {
    let dst = unsafe { *FFISafe::get(dst) };
    let left = unsafe { &*(left as *const [LeftPixelRow2px<u16>; 8]) };
    let top = unsafe { FFISafe::get(top) };
    let bottom = unsafe { FFISafe::get(bottom) };

    cdef_filter_8x8_16bpc_avx2_inner(
        dst,
        left,
        top,
        bottom,
        pri_strength,
        sec_strength,
        dir,
        damping,
        edges,
        bitdepth_max,
    );
}

/// FFI wrapper for CDEF filter 4x8 16bpc
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn cdef_filter_4x8_16bpc_avx2(
    _dst_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    left: *const [LeftPixelRow2px<DynPixel>; 8],
    _top_ptr: *const DynPixel,
    _bottom_ptr: *const DynPixel,
    pri_strength: c_int,
    sec_strength: c_int,
    dir: c_int,
    damping: c_int,
    edges: CdefEdgeFlags,
    bitdepth_max: c_int,
    dst: *const FFISafe<PicOffset>,
    top: *const FFISafe<CdefTop>,
    bottom: *const FFISafe<CdefBottom>,
) {
    let dst = unsafe { *FFISafe::get(dst) };
    let left = unsafe { &*(left as *const [LeftPixelRow2px<u16>; 8]) };
    let top = unsafe { FFISafe::get(top) };
    let bottom = unsafe { FFISafe::get(bottom) };

    cdef_filter_4x8_16bpc_avx2_inner(
        dst,
        left,
        top,
        bottom,
        pri_strength,
        sec_strength,
        dir,
        damping,
        edges,
        bitdepth_max,
    );
}

/// FFI wrapper for CDEF filter 4x4 16bpc
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn cdef_filter_4x4_16bpc_avx2(
    _dst_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    left: *const [LeftPixelRow2px<DynPixel>; 8],
    _top_ptr: *const DynPixel,
    _bottom_ptr: *const DynPixel,
    pri_strength: c_int,
    sec_strength: c_int,
    dir: c_int,
    damping: c_int,
    edges: CdefEdgeFlags,
    bitdepth_max: c_int,
    dst: *const FFISafe<PicOffset>,
    top: *const FFISafe<CdefTop>,
    bottom: *const FFISafe<CdefBottom>,
) {
    let dst = unsafe { *FFISafe::get(dst) };
    let left = unsafe { &*(left as *const [LeftPixelRow2px<u16>; 8]) };
    let top = unsafe { FFISafe::get(top) };
    let bottom = unsafe { FFISafe::get(bottom) };

    cdef_filter_4x4_16bpc_avx2_inner(
        dst,
        left,
        top,
        bottom,
        pri_strength,
        sec_strength,
        dir,
        damping,
        edges,
        bitdepth_max,
    );
}

/// FFI wrapper for cdef_find_dir 16bpc
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn cdef_find_dir_16bpc_avx2(
    _dst_ptr: *const DynPixel,
    _dst_stride: ptrdiff_t,
    variance: &mut c_uint,
    bitdepth_max: c_int,
    dst: *const FFISafe<PicOffset>,
) -> c_int {
    use crate::include::common::bitdepth::BitDepth16;

    let dst = unsafe { *FFISafe::get(dst) };
    let bd = BitDepth16::new(bitdepth_max as u16);

    cdef_find_dir_scalar::<BitDepth16>(dst, variance, bd)
}

// ============================================================================
// SAFE DISPATCH WRAPPERS
// ============================================================================
// These functions wrap the unsafe extern "C" SIMD functions behind a safe API.
// They take safe Rust types, handle pointer conversion internally, and return
// a bool/Option indicating whether SIMD was used.

/// Safe dispatch for cdef_filter. Returns true if SIMD was used.
#[cfg(target_arch = "x86_64")]
pub fn cdef_filter_dispatch<BD: BitDepth>(
    variant: usize,
    dst: PicOffset,
    left: &[LeftPixelRow2px<BD::Pixel>; 8],
    top: CdefTop,
    bottom: CdefBottom,
    pri_strength: c_int,
    sec_strength: c_int,
    dir: c_int,
    damping: c_int,
    edges: CdefEdgeFlags,
    bd: BD,
) -> bool {
    use crate::include::common::bitdepth::BPC;
    use crate::src::cpu::CpuFlags;

    if !crate::src::cpu::rav1d_get_cpu_flags().contains(CpuFlags::AVX2) {
        return false;
    }

    // Left pointer cast is safe because LeftPixelRow2px<BD::Pixel> has same layout for u8/u16.
    match (BD::BPC, variant) {
        (BPC::BPC8, 0) => {
            let left = unsafe { &*(left as *const _ as *const [LeftPixelRow2px<u8>; 8]) };
            cdef_filter_8x8_8bpc_avx2_inner(
                dst,
                left,
                &top,
                &bottom,
                pri_strength,
                sec_strength,
                dir,
                damping,
                edges,
            );
        }
        (BPC::BPC8, 1) => {
            let left = unsafe { &*(left as *const _ as *const [LeftPixelRow2px<u8>; 8]) };
            cdef_filter_4x8_8bpc_avx2_inner(
                dst,
                left,
                &top,
                &bottom,
                pri_strength,
                sec_strength,
                dir,
                damping,
                edges,
            );
        }
        (BPC::BPC8, _) => {
            let left = unsafe { &*(left as *const _ as *const [LeftPixelRow2px<u8>; 8]) };
            cdef_filter_4x4_8bpc_avx2_inner(
                dst,
                left,
                &top,
                &bottom,
                pri_strength,
                sec_strength,
                dir,
                damping,
                edges,
            );
        }
        (BPC::BPC16, 0) => {
            let left = unsafe { &*(left as *const _ as *const [LeftPixelRow2px<u16>; 8]) };
            cdef_filter_8x8_16bpc_avx2_inner(
                dst,
                left,
                &top,
                &bottom,
                pri_strength,
                sec_strength,
                dir,
                damping,
                edges,
                bd.into_c(),
            );
        }
        (BPC::BPC16, 1) => {
            let left = unsafe { &*(left as *const _ as *const [LeftPixelRow2px<u16>; 8]) };
            cdef_filter_4x8_16bpc_avx2_inner(
                dst,
                left,
                &top,
                &bottom,
                pri_strength,
                sec_strength,
                dir,
                damping,
                edges,
                bd.into_c(),
            );
        }
        (BPC::BPC16, _) => {
            let left = unsafe { &*(left as *const _ as *const [LeftPixelRow2px<u16>; 8]) };
            cdef_filter_4x4_16bpc_avx2_inner(
                dst,
                left,
                &top,
                &bottom,
                pri_strength,
                sec_strength,
                dir,
                damping,
                edges,
                bd.into_c(),
            );
        }
    }
    true
}

/// Safe dispatch for cdef_find_dir. Returns Some(dir) if SIMD was used.
#[cfg(target_arch = "x86_64")]
pub fn cdef_dir_dispatch<BD: BitDepth>(
    dst: PicOffset,
    variance: &mut c_uint,
    bd: BD,
) -> Option<c_int> {
    use crate::src::cpu::CpuFlags;

    if !crate::src::cpu::rav1d_get_cpu_flags().contains(CpuFlags::AVX2) {
        return None;
    }

    Some(cdef_find_dir_scalar(dst, variance, bd))
}
