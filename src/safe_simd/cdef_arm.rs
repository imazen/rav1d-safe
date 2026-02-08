//! Safe ARM NEON implementations for CDEF (Constrained Directional Enhancement Filter)
//!
//! CDEF applies direction-dependent filtering to remove coding artifacts
//! while preserving edges.

#![cfg_attr(not(feature = "asm"), forbid(unsafe_code))]
#![allow(unused_imports)]
#![allow(clippy::too_many_arguments)]

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

use std::cmp;
use std::ffi::c_int;
use std::ffi::c_uint;

use crate::include::common::bitdepth::AsPrimitive;
use crate::include::common::bitdepth::BitDepth;
use crate::include::common::bitdepth::DynPixel;
use crate::include::common::bitdepth::LeftPixelRow2px;
use crate::include::common::intops::iclip;
use crate::include::dav1d::picture::PicOffset;
use crate::src::cdef::CdefBottom;
use crate::src::cdef::CdefEdgeFlags;
use crate::src::cdef::CdefTop;
use crate::src::ffi_safe::FFISafe;
use crate::src::pic_or_buf::PicOrBuf;
use crate::src::strided::Strided as _;
use crate::src::tables::dav1d_cdef_directions;
use crate::src::with_offset::WithOffset;
use libc::ptrdiff_t;

const TMP_STRIDE: usize = 16;

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

// ============================================================================
// 8BPC IMPLEMENTATIONS
// ============================================================================

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

    // Handle top edge (safe slice access via DisjointMut)
    if edges.contains(CdefEdgeFlags::HAVE_TOP) {
        let x_start = if edges.contains(CdefEdgeFlags::HAVE_LEFT) {
            0usize
        } else {
            2
        };
        let x_end = if edges.contains(CdefEdgeFlags::HAVE_RIGHT) {
            w + 4
        } else {
            w + 2
        };
        for dy in 0..2usize {
            let row_offset = tmp_offset - (2 - dy) * TMP_STRIDE;
            let top_row = WithOffset {
                data: top.data,
                offset: top
                    .offset
                    .wrapping_sub(2)
                    .wrapping_add_signed(dy as isize * stride),
            };
            let slice = top_row.data.slice_as::<_, u8>((top_row.offset.., ..x_end));
            for x in x_start..x_end {
                tmp[row_offset + x - 2] = slice[x] as u16;
            }
        }
    }

    // Handle bottom edge (safe slice access via DisjointMut/PicOrBuf)
    if edges.contains(CdefEdgeFlags::HAVE_BOTTOM) {
        let x_start = if edges.contains(CdefEdgeFlags::HAVE_LEFT) {
            0usize
        } else {
            2
        };
        let x_end = if edges.contains(CdefEdgeFlags::HAVE_RIGHT) {
            w + 4
        } else {
            w + 2
        };
        for dy in 0..2usize {
            let row_offset = tmp_offset + (h + dy) * TMP_STRIDE;
            let bottom_row = WithOffset {
                data: bottom.data,
                offset: bottom
                    .offset
                    .wrapping_sub(2)
                    .wrapping_add_signed(dy as isize * stride),
            };
            let slice = match bottom_row.data {
                PicOrBuf::Pic(pic) => {
                    let guard = pic.slice::<BitDepth8, _>((bottom_row.offset.., ..x_end));
                    for x in x_start..x_end {
                        tmp[row_offset + x - 2] = guard[x] as u16;
                    }
                    continue;
                }
                PicOrBuf::Buf(buf) => buf.slice_as::<_, u8>((bottom_row.offset.., ..x_end)),
            };
            for x in x_start..x_end {
                tmp[row_offset + x - 2] = slice[x] as u16;
            }
        }
    }
}

/// CDEF filter inner implementation for 8bpc
fn cdef_filter_block_8bpc_inner(
    dst: PicOffset,
    left: &[LeftPixelRow2px<u8>; 8],
    top: &CdefTop,
    bottom: &CdefBottom,
    pri_strength: c_int,
    sec_strength: c_int,
    dir: c_int,
    damping: c_int,
    edges: CdefEdgeFlags,
    w: usize,
    h: usize,
) {
    use crate::include::common::bitdepth::BitDepth8;

    let dir = dir as usize;

    let mut tmp = [0u16; TMP_STRIDE * 12];
    padding_8bpc(&mut tmp, dst, left, top, bottom, w, h, edges);

    let tmp_offset = 2 * TMP_STRIDE + 2;
    let stride = dst.pixel_stride::<BitDepth8>();

    if pri_strength != 0 {
        let pri_tap = 4 - (pri_strength & 1);
        let pri_shift = cmp::max(0, damping - pri_strength.ilog2() as c_int);

        if sec_strength != 0 {
            let sec_shift = damping - sec_strength.ilog2() as c_int;

            for y in 0..h {
                let tmp_row = &tmp[tmp_offset + y * TMP_STRIDE..];
                let mut dst_row = (dst + (y as isize * stride)).slice_mut::<BitDepth8>(w);

                for x in 0..w {
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
            for y in 0..h {
                let tmp_row = &tmp[tmp_offset + y * TMP_STRIDE..];
                let mut dst_row = (dst + (y as isize * stride)).slice_mut::<BitDepth8>(w);

                for x in 0..w {
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
    } else if sec_strength != 0 {
        let sec_shift = damping - sec_strength.ilog2() as c_int;

        for y in 0..h {
            let tmp_row = &tmp[tmp_offset + y * TMP_STRIDE..];
            let mut dst_row = (dst + (y as isize * stride)).slice_mut::<BitDepth8>(w);

            for x in 0..w {
                let px = dst_row[x] as i32;
                let mut sum = 0i32;
                let mut max = px;
                let mut min = px;

                for k in 0..2 {
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
    }
}

// ============================================================================
// CDEF DIRECTION FINDING
// ============================================================================

/// Scalar implementation of cdef_find_dir for 8bpc
fn cdef_find_dir_8bpc_inner(img: PicOffset, variance: &mut c_uint) -> c_int {
    use crate::include::common::bitdepth::BitDepth8;

    let mut partial_sum_hv = [[0i32; 8]; 2];
    let mut partial_sum_diag = [[0i32; 15]; 2];
    let mut partial_sum_alt = [[0i32; 11]; 4];

    const W: usize = 8;
    const H: usize = 8;

    for y in 0..H {
        let img = img + (y as isize * img.pixel_stride::<BitDepth8>());
        let img = &*img.slice::<BitDepth8>(W);
        for x in 0..W {
            let px = img[x] as i32 - 128;

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
    bitdepth_max: i32,
) {
    use crate::include::common::bitdepth::BitDepth16;

    let stride = dst.pixel_stride::<BitDepth16>();
    // Fill with CDEF_VERY_LARGE (8*2046+1 for 16bpc)
    let very_large = (8 * bitdepth_max + 1) as u16;
    tmp.iter_mut().for_each(|x| *x = very_large);

    let tmp_offset = 2 * TMP_STRIDE + 2;

    // Copy source pixels
    for y in 0..h {
        let row_offset = tmp_offset + y * TMP_STRIDE;
        let src = (dst + (y as isize * stride)).slice::<BitDepth16>(w);
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
            let src = (dst + (y as isize * stride)).slice::<BitDepth16>(w + 2);
            tmp[row_offset + w] = src[w];
            tmp[row_offset + w + 1] = src[w + 1];
        }
    }

    // Handle top edge (safe slice access via DisjointMut)
    if edges.contains(CdefEdgeFlags::HAVE_TOP) {
        let pixel_stride = dst.pixel_stride::<BitDepth16>();
        let x_start = if edges.contains(CdefEdgeFlags::HAVE_LEFT) {
            0usize
        } else {
            2
        };
        let x_end = if edges.contains(CdefEdgeFlags::HAVE_RIGHT) {
            w + 4
        } else {
            w + 2
        };
        for dy in 0..2usize {
            let row_offset = tmp_offset - (2 - dy) * TMP_STRIDE;
            let top_row = WithOffset {
                data: top.data,
                offset: top
                    .offset
                    .wrapping_sub(2)
                    .wrapping_add_signed(dy as isize * pixel_stride),
            };
            let slice = top_row.data.slice_as::<_, u16>((top_row.offset.., ..x_end));
            for x in x_start..x_end {
                tmp[row_offset + x - 2] = slice[x];
            }
        }
    }

    // Handle bottom edge (safe slice access via DisjointMut/PicOrBuf)
    if edges.contains(CdefEdgeFlags::HAVE_BOTTOM) {
        let pixel_stride = dst.pixel_stride::<BitDepth16>();
        let x_start = if edges.contains(CdefEdgeFlags::HAVE_LEFT) {
            0usize
        } else {
            2
        };
        let x_end = if edges.contains(CdefEdgeFlags::HAVE_RIGHT) {
            w + 4
        } else {
            w + 2
        };
        for dy in 0..2usize {
            let row_offset = tmp_offset + (h + dy) * TMP_STRIDE;
            let bottom_row = WithOffset {
                data: bottom.data,
                offset: bottom
                    .offset
                    .wrapping_sub(2)
                    .wrapping_add_signed(dy as isize * pixel_stride),
            };
            let slice = match bottom_row.data {
                PicOrBuf::Pic(pic) => {
                    let guard = pic.slice::<BitDepth16, _>((bottom_row.offset.., ..x_end));
                    for x in x_start..x_end {
                        tmp[row_offset + x - 2] = guard[x];
                    }
                    continue;
                }
                PicOrBuf::Buf(buf) => buf.slice_as::<_, u16>((bottom_row.offset.., ..x_end)),
            };
            for x in x_start..x_end {
                tmp[row_offset + x - 2] = slice[x];
            }
        }
    }
}

/// CDEF filter inner implementation for 16bpc
fn cdef_filter_block_16bpc_inner(
    dst: PicOffset,
    left: &[LeftPixelRow2px<u16>; 8],
    top: &CdefTop,
    bottom: &CdefBottom,
    pri_strength: c_int,
    sec_strength: c_int,
    dir: c_int,
    damping: c_int,
    edges: CdefEdgeFlags,
    w: usize,
    h: usize,
    bitdepth_max: i32,
) {
    use crate::include::common::bitdepth::BitDepth16;

    let dir = dir as usize;

    let mut tmp = [0u16; TMP_STRIDE * 12];
    padding_16bpc(&mut tmp, dst, left, top, bottom, w, h, edges, bitdepth_max);

    let tmp_offset = 2 * TMP_STRIDE + 2;
    let stride = dst.pixel_stride::<BitDepth16>();

    if pri_strength != 0 {
        let pri_tap = 4 - (pri_strength & 1);
        let pri_shift = cmp::max(0, damping - pri_strength.ilog2() as c_int);

        if sec_strength != 0 {
            let sec_shift = damping - sec_strength.ilog2() as c_int;

            for y in 0..h {
                let tmp_row = &tmp[tmp_offset + y * TMP_STRIDE..];
                let mut dst_row = (dst + (y as isize * stride)).slice_mut::<BitDepth16>(w);

                for x in 0..w {
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

                    dst_row[x] = iclip(px + (sum - (sum < 0) as i32 + 8 >> 4), min, max) as u16;
                }
            }
        } else {
            for y in 0..h {
                let tmp_row = &tmp[tmp_offset + y * TMP_STRIDE..];
                let mut dst_row = (dst + (y as isize * stride)).slice_mut::<BitDepth16>(w);

                for x in 0..w {
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

                    dst_row[x] = (px + (sum - (sum < 0) as i32 + 8 >> 4)) as u16;
                }
            }
        }
    } else if sec_strength != 0 {
        let sec_shift = damping - sec_strength.ilog2() as c_int;

        for y in 0..h {
            let tmp_row = &tmp[tmp_offset + y * TMP_STRIDE..];
            let mut dst_row = (dst + (y as isize * stride)).slice_mut::<BitDepth16>(w);

            for x in 0..w {
                let px = dst_row[x] as i32;
                let mut sum = 0i32;
                let mut max = px;
                let mut min = px;

                for k in 0..2 {
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

                dst_row[x] = iclip(px + (sum - (sum < 0) as i32 + 8 >> 4), min, max) as u16;
            }
        }
    }
}

/// Scalar implementation of cdef_find_dir for 16bpc
fn cdef_find_dir_16bpc_inner(img: PicOffset, variance: &mut c_uint, bitdepth_max: i32) -> c_int {
    use crate::include::common::bitdepth::BitDepth16;

    let bitdepth_min_8 = if bitdepth_max == 1023 { 2 } else { 4 }; // 10bpc or 12bpc

    let mut partial_sum_hv = [[0i32; 8]; 2];
    let mut partial_sum_diag = [[0i32; 15]; 2];
    let mut partial_sum_alt = [[0i32; 11]; 4];

    const W: usize = 8;
    const H: usize = 8;

    for y in 0..H {
        let img = img + (y as isize * img.pixel_stride::<BitDepth16>());
        let img = &*img.slice::<BitDepth16>(W);
        for x in 0..W {
            let px = (img[x] as i32 >> bitdepth_min_8) - 128;

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
// FFI WRAPPERS
// ============================================================================

#[cfg(all(feature = "asm", target_arch = "aarch64"))]
pub unsafe extern "C" fn cdef_filter_8x8_8bpc_neon(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    left: *const [LeftPixelRow2px<DynPixel>; 8],
    top: *const DynPixel,
    bottom: *const DynPixel,
    pri_strength: c_int,
    sec_strength: c_int,
    dir: c_int,
    damping: c_int,
    edges: CdefEdgeFlags,
    _bitdepth_max: c_int,
    dst: *const FFISafe<PicOffset>,
    top_ffi: *const FFISafe<CdefTop>,
    bottom_ffi: *const FFISafe<CdefBottom>,
) {
    let dst = unsafe { *FFISafe::get(dst) };
    let left = unsafe { &*(left as *const [LeftPixelRow2px<u8>; 8]) };
    let top = unsafe { &*FFISafe::get(top_ffi) };
    let bottom = unsafe { &*FFISafe::get(bottom_ffi) };

    cdef_filter_block_8bpc_inner(
        dst,
        left,
        top,
        bottom,
        pri_strength,
        sec_strength,
        dir,
        damping,
        edges,
        8,
        8,
    );
}

#[cfg(all(feature = "asm", target_arch = "aarch64"))]
pub unsafe extern "C" fn cdef_filter_4x8_8bpc_neon(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    left: *const [LeftPixelRow2px<DynPixel>; 8],
    top: *const DynPixel,
    bottom: *const DynPixel,
    pri_strength: c_int,
    sec_strength: c_int,
    dir: c_int,
    damping: c_int,
    edges: CdefEdgeFlags,
    _bitdepth_max: c_int,
    dst: *const FFISafe<PicOffset>,
    top_ffi: *const FFISafe<CdefTop>,
    bottom_ffi: *const FFISafe<CdefBottom>,
) {
    let dst = unsafe { *FFISafe::get(dst) };
    let left = unsafe { &*(left as *const [LeftPixelRow2px<u8>; 8]) };
    let top = unsafe { &*FFISafe::get(top_ffi) };
    let bottom = unsafe { &*FFISafe::get(bottom_ffi) };

    cdef_filter_block_8bpc_inner(
        dst,
        left,
        top,
        bottom,
        pri_strength,
        sec_strength,
        dir,
        damping,
        edges,
        4,
        8,
    );
}

#[cfg(all(feature = "asm", target_arch = "aarch64"))]
pub unsafe extern "C" fn cdef_filter_4x4_8bpc_neon(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    left: *const [LeftPixelRow2px<DynPixel>; 8],
    top: *const DynPixel,
    bottom: *const DynPixel,
    pri_strength: c_int,
    sec_strength: c_int,
    dir: c_int,
    damping: c_int,
    edges: CdefEdgeFlags,
    _bitdepth_max: c_int,
    dst: *const FFISafe<PicOffset>,
    top_ffi: *const FFISafe<CdefTop>,
    bottom_ffi: *const FFISafe<CdefBottom>,
) {
    let dst = unsafe { *FFISafe::get(dst) };
    let left = unsafe { &*(left as *const [LeftPixelRow2px<u8>; 8]) };
    let top = unsafe { &*FFISafe::get(top_ffi) };
    let bottom = unsafe { &*FFISafe::get(bottom_ffi) };

    cdef_filter_block_8bpc_inner(
        dst,
        left,
        top,
        bottom,
        pri_strength,
        sec_strength,
        dir,
        damping,
        edges,
        4,
        4,
    );
}

#[cfg(all(feature = "asm", target_arch = "aarch64"))]
pub unsafe extern "C" fn cdef_find_dir_8bpc_neon(
    _dst_ptr: *const DynPixel,
    _dst_stride: ptrdiff_t,
    variance: &mut c_uint,
    _bitdepth_max: c_int,
    dst: *const FFISafe<PicOffset>,
) -> c_int {
    let img = *FFISafe::get(dst);
    cdef_find_dir_8bpc_inner(img, variance)
}

#[cfg(all(feature = "asm", target_arch = "aarch64"))]
pub unsafe extern "C" fn cdef_filter_8x8_16bpc_neon(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    left: *const [LeftPixelRow2px<DynPixel>; 8],
    top: *const DynPixel,
    bottom: *const DynPixel,
    pri_strength: c_int,
    sec_strength: c_int,
    dir: c_int,
    damping: c_int,
    edges: CdefEdgeFlags,
    bitdepth_max: c_int,
    dst: *const FFISafe<PicOffset>,
    top_ffi: *const FFISafe<CdefTop>,
    bottom_ffi: *const FFISafe<CdefBottom>,
) {
    let dst = unsafe { *FFISafe::get(dst) };
    let left = unsafe { &*(left as *const [LeftPixelRow2px<u16>; 8]) };
    let top = unsafe { &*FFISafe::get(top_ffi) };
    let bottom = unsafe { &*FFISafe::get(bottom_ffi) };

    cdef_filter_block_16bpc_inner(
        dst,
        left,
        top,
        bottom,
        pri_strength,
        sec_strength,
        dir,
        damping,
        edges,
        8,
        8,
        bitdepth_max,
    );
}

#[cfg(all(feature = "asm", target_arch = "aarch64"))]
pub unsafe extern "C" fn cdef_filter_4x8_16bpc_neon(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    left: *const [LeftPixelRow2px<DynPixel>; 8],
    top: *const DynPixel,
    bottom: *const DynPixel,
    pri_strength: c_int,
    sec_strength: c_int,
    dir: c_int,
    damping: c_int,
    edges: CdefEdgeFlags,
    bitdepth_max: c_int,
    dst: *const FFISafe<PicOffset>,
    top_ffi: *const FFISafe<CdefTop>,
    bottom_ffi: *const FFISafe<CdefBottom>,
) {
    let dst = unsafe { *FFISafe::get(dst) };
    let left = unsafe { &*(left as *const [LeftPixelRow2px<u16>; 8]) };
    let top = unsafe { &*FFISafe::get(top_ffi) };
    let bottom = unsafe { &*FFISafe::get(bottom_ffi) };

    cdef_filter_block_16bpc_inner(
        dst,
        left,
        top,
        bottom,
        pri_strength,
        sec_strength,
        dir,
        damping,
        edges,
        4,
        8,
        bitdepth_max,
    );
}

#[cfg(all(feature = "asm", target_arch = "aarch64"))]
pub unsafe extern "C" fn cdef_filter_4x4_16bpc_neon(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    left: *const [LeftPixelRow2px<DynPixel>; 8],
    top: *const DynPixel,
    bottom: *const DynPixel,
    pri_strength: c_int,
    sec_strength: c_int,
    dir: c_int,
    damping: c_int,
    edges: CdefEdgeFlags,
    bitdepth_max: c_int,
    dst: *const FFISafe<PicOffset>,
    top_ffi: *const FFISafe<CdefTop>,
    bottom_ffi: *const FFISafe<CdefBottom>,
) {
    let dst = unsafe { *FFISafe::get(dst) };
    let left = unsafe { &*(left as *const [LeftPixelRow2px<u16>; 8]) };
    let top = unsafe { &*FFISafe::get(top_ffi) };
    let bottom = unsafe { &*FFISafe::get(bottom_ffi) };

    cdef_filter_block_16bpc_inner(
        dst,
        left,
        top,
        bottom,
        pri_strength,
        sec_strength,
        dir,
        damping,
        edges,
        4,
        4,
        bitdepth_max,
    );
}

#[cfg(all(feature = "asm", target_arch = "aarch64"))]
pub unsafe extern "C" fn cdef_find_dir_16bpc_neon(
    _dst_ptr: *const DynPixel,
    _dst_stride: ptrdiff_t,
    variance: &mut c_uint,
    bitdepth_max: c_int,
    dst: *const FFISafe<PicOffset>,
) -> c_int {
    let img = *FFISafe::get(dst);
    cdef_find_dir_16bpc_inner(img, variance, bitdepth_max)
}

// ============================================================================
// SAFE DISPATCH WRAPPERS (aarch64)
// ============================================================================

/// Safe dispatch for cdef_filter on aarch64. Returns true if NEON was used.
#[cfg(target_arch = "aarch64")]
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

    let (w, h) = match variant {
        0 => (8, 8),
        1 => (4, 8),
        _ => (4, 4),
    };

    // Call inner functions directly, bypassing FFI wrappers.
    // SAFETY: left pointer cast is safe because LeftPixelRow2px<BD::Pixel> has same layout for u8/u16.
    match BD::BPC {
        BPC::BPC8 => {
            let left: &[LeftPixelRow2px<u8>; 8] =
                crate::src::safe_simd::pixel_access::reinterpret_ref(left)
                    .expect("BD::Pixel layout matches u8");
            cdef_filter_block_8bpc_inner(
                dst,
                left,
                &top,
                &bottom,
                pri_strength,
                sec_strength,
                dir,
                damping,
                edges,
                w,
                h,
            );
        }
        BPC::BPC16 => {
            let left: &[LeftPixelRow2px<u16>; 8] =
                crate::src::safe_simd::pixel_access::reinterpret_ref(left)
                    .expect("BD::Pixel layout matches u16");
            cdef_filter_block_16bpc_inner(
                dst,
                left,
                &top,
                &bottom,
                pri_strength,
                sec_strength,
                dir,
                damping,
                edges,
                w,
                h,
                bd.into_c(),
            );
        }
    }
    true
}

/// Safe dispatch for cdef_find_dir on aarch64. Returns Some(dir).
#[cfg(target_arch = "aarch64")]
pub fn cdef_dir_dispatch<BD: BitDepth>(
    dst: PicOffset,
    variance: &mut c_uint,
    bd: BD,
) -> Option<c_int> {
    use crate::include::common::bitdepth::BPC;

    let dir = match BD::BPC {
        BPC::BPC8 => cdef_find_dir_8bpc_inner(dst, variance),
        BPC::BPC16 => cdef_find_dir_16bpc_inner(dst, variance, bd.into_c()),
    };
    Some(dir)
}
