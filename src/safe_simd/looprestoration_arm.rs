//! Safe ARM NEON implementations for Loop Restoration
//!
//! Wiener filter and SGR (Self-Guided Restoration) implementations for ARM.

#![allow(unused_imports)]
#![allow(clippy::too_many_arguments)]

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

use std::cmp;
use std::ffi::c_int;
use std::slice;

use crate::include::common::bitdepth::AsPrimitive;
use crate::include::common::bitdepth::BitDepth;
use crate::include::common::bitdepth::BitDepth16;
use crate::include::common::bitdepth::BitDepth8;
use crate::include::common::bitdepth::DynPixel;
use crate::include::common::bitdepth::LeftPixelRow;
use crate::include::common::intops::iclip;
use crate::include::dav1d::picture::PicOffset;
use crate::src::align::AlignedVec64;
use crate::src::disjoint_mut::DisjointMut;
use crate::src::ffi_safe::FFISafe;
use crate::src::looprestoration::{padding, LooprestorationParams, LrEdgeFlags};
use crate::src::pixels::Pixels;
use crate::src::strided::Strided as _;
use crate::src::tables::dav1d_sgr_x_by_x;
use libc::ptrdiff_t;

const REST_UNIT_STRIDE: usize = 256 * 3 / 2 + 3 + 3; // = 390
const MAX_RESTORATION_WIDTH: usize = 256 * 3 / 2;

// ============================================================================
// WIENER FILTER IMPLEMENTATIONS - 8BPC
// ============================================================================

fn wiener_filter_8bpc_inner(
    p: PicOffset,
    left: &[LeftPixelRow<u8>],
    lpf: &DisjointMut<AlignedVec64<u8>>,
    lpf_off: isize,
    w: usize,
    h: usize,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    filter_len: usize, // 7 or 5
) {
    let mut tmp = [0u8; (64 + 3 + 3) * REST_UNIT_STRIDE];
    padding::<BitDepth8>(&mut tmp, p, left, lpf, lpf_off, w, h, edges);

    let mut hor = [0u16; (64 + 3 + 3) * REST_UNIT_STRIDE];
    let filter = &params.filter;

    let (center_tap, tap_count, tap_start) = if filter_len == 7 {
        (3, 7, 0)
    } else {
        (2, 5, 1)
    };

    let round_bits_h = 3i32;
    let rounding_off_h = 1i32 << (round_bits_h - 1);
    let clip_limit = 1i32 << (8 + 1 + 7 - round_bits_h);

    // Horizontal filter pass
    let row_count = if filter_len == 7 { h + 6 } else { h + 4 };
    for row in 0..row_count {
        let tmp_row = &tmp[row * REST_UNIT_STRIDE..];
        let hor_row = &mut hor[row * REST_UNIT_STRIDE..row * REST_UNIT_STRIDE + w];

        for x in 0..w {
            let mut sum = 1i32 << 14;
            sum += tmp_row[x + center_tap] as i32 * 128;
            for k in 0..tap_count {
                sum += tmp_row[x + k] as i32 * filter[0][tap_start + k] as i32;
            }
            hor_row[x] = iclip((sum + rounding_off_h) >> round_bits_h, 0, clip_limit - 1) as u16;
        }
    }

    // Vertical filter pass
    let round_bits_v = 11i32;
    let rounding_off_v = 1i32 << (round_bits_v - 1);
    let round_offset = 1i32 << (8 + round_bits_v - 1);
    let stride = p.pixel_stride::<BitDepth8>();

    for j in 0..h {
        let mut dst_row = (p + (j as isize * stride)).slice_mut::<BitDepth8>(w);

        for i in 0..w {
            let mut sum = -round_offset;
            for k in 0..tap_count {
                let row = &hor[(j + k) * REST_UNIT_STRIDE + i..];
                sum += row[0] as i32 * filter[1][tap_start + k] as i32;
            }
            dst_row[i] = iclip((sum + rounding_off_v) >> round_bits_v, 0, 255) as u8;
        }
    }
}

// ============================================================================
// WIENER FILTER IMPLEMENTATIONS - 16BPC
// ============================================================================

fn wiener_filter_16bpc_inner(
    p: PicOffset,
    left: &[LeftPixelRow<u16>],
    lpf: &DisjointMut<AlignedVec64<u8>>,
    lpf_off: isize,
    w: usize,
    h: usize,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    filter_len: usize,
    bitdepth_max: i32,
) {
    let mut tmp = [0u16; (64 + 3 + 3) * REST_UNIT_STRIDE];
    padding::<BitDepth16>(&mut tmp, p, left, lpf, lpf_off, w, h, edges);

    let bitdepth = if bitdepth_max == 1023 { 10 } else { 12 };
    let mut hor = [0i32; (64 + 3 + 3) * REST_UNIT_STRIDE];
    let filter = &params.filter;

    let (center_tap, tap_count, tap_start) = if filter_len == 7 {
        (3, 7, 0)
    } else {
        (2, 5, 1)
    };

    let round_bits_h = (bitdepth + 8 - 11) as i32;
    let rounding_off_h = 1i32 << (round_bits_h - 1).max(0);
    let clip_limit = 1i32 << (bitdepth + 1 + 7 - round_bits_h);

    let row_count = if filter_len == 7 { h + 6 } else { h + 4 };
    for row in 0..row_count {
        let tmp_row = &tmp[row * REST_UNIT_STRIDE..];
        let hor_row = &mut hor[row * REST_UNIT_STRIDE..row * REST_UNIT_STRIDE + w];

        for x in 0..w {
            let mut sum = 1i32 << (bitdepth + 6);
            sum += tmp_row[x + center_tap] as i32 * 128;
            for k in 0..tap_count {
                sum += tmp_row[x + k] as i32 * filter[0][tap_start + k] as i32;
            }
            hor_row[x] = iclip(
                (sum + rounding_off_h) >> round_bits_h.max(0),
                0,
                clip_limit - 1,
            );
        }
    }

    let round_bits_v = 11i32;
    let rounding_off_v = 1i32 << (round_bits_v - 1);
    let round_offset = 1i32 << (bitdepth + round_bits_v - 1);
    let stride = p.pixel_stride::<BitDepth16>();

    for j in 0..h {
        let mut dst_row = (p + (j as isize * stride)).slice_mut::<BitDepth16>(w);

        for i in 0..w {
            let mut sum = -round_offset;
            for k in 0..tap_count {
                let row = &hor[(j + k) * REST_UNIT_STRIDE + i..];
                sum += row[0] * filter[1][tap_start + k] as i32;
            }
            dst_row[i] = iclip((sum + rounding_off_v) >> round_bits_v, 0, bitdepth_max) as u16;
        }
    }
}

// ============================================================================
// FFI WRAPPERS - 8BPC
// ============================================================================

#[cfg(all(feature = "asm", target_arch = "aarch64"))]
pub unsafe extern "C" fn wiener_filter7_8bpc_neon(
    _p_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    left: *const LeftPixelRow<DynPixel>,
    lpf_ptr: *const DynPixel,
    w: c_int,
    h: c_int,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    _bitdepth_max: c_int,
    p: *const FFISafe<PicOffset>,
    lpf: *const FFISafe<DisjointMut<AlignedVec64<u8>>>,
) {
    let p = unsafe { *FFISafe::get(p) };
    let left = unsafe { slice::from_raw_parts(left as *const LeftPixelRow<u8>, h as usize + 3) };
    let lpf = unsafe { FFISafe::get(lpf) };
    let lpf_off = unsafe { lpf_ptr as isize - lpf.as_byte_mut_ptr() as isize };

    wiener_filter_8bpc_inner(
        p, left, lpf, lpf_off, w as usize, h as usize, params, edges, 7,
    );
}

#[cfg(all(feature = "asm", target_arch = "aarch64"))]
pub unsafe extern "C" fn wiener_filter5_8bpc_neon(
    _p_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    left: *const LeftPixelRow<DynPixel>,
    lpf_ptr: *const DynPixel,
    w: c_int,
    h: c_int,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    _bitdepth_max: c_int,
    p: *const FFISafe<PicOffset>,
    lpf: *const FFISafe<DisjointMut<AlignedVec64<u8>>>,
) {
    let p = unsafe { *FFISafe::get(p) };
    let left = unsafe { slice::from_raw_parts(left as *const LeftPixelRow<u8>, h as usize + 2) };
    let lpf = unsafe { FFISafe::get(lpf) };
    let lpf_off = unsafe { lpf_ptr as isize - lpf.as_byte_mut_ptr() as isize };

    wiener_filter_8bpc_inner(
        p, left, lpf, lpf_off, w as usize, h as usize, params, edges, 5,
    );
}

// ============================================================================
// FFI WRAPPERS - 16BPC
// ============================================================================

#[cfg(all(feature = "asm", target_arch = "aarch64"))]
pub unsafe extern "C" fn wiener_filter7_16bpc_neon(
    _p_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    left: *const LeftPixelRow<DynPixel>,
    lpf_ptr: *const DynPixel,
    w: c_int,
    h: c_int,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    bitdepth_max: c_int,
    p: *const FFISafe<PicOffset>,
    lpf: *const FFISafe<DisjointMut<AlignedVec64<u8>>>,
) {
    let p = unsafe { *FFISafe::get(p) };
    let left = unsafe { slice::from_raw_parts(left as *const LeftPixelRow<u16>, h as usize + 3) };
    let lpf = unsafe { FFISafe::get(lpf) };
    let lpf_off = unsafe { (lpf_ptr as isize - lpf.as_byte_mut_ptr() as isize) / 2 };

    wiener_filter_16bpc_inner(
        p,
        left,
        lpf,
        lpf_off,
        w as usize,
        h as usize,
        params,
        edges,
        7,
        bitdepth_max,
    );
}

#[cfg(all(feature = "asm", target_arch = "aarch64"))]
pub unsafe extern "C" fn wiener_filter5_16bpc_neon(
    _p_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    left: *const LeftPixelRow<DynPixel>,
    lpf_ptr: *const DynPixel,
    w: c_int,
    h: c_int,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    bitdepth_max: c_int,
    p: *const FFISafe<PicOffset>,
    lpf: *const FFISafe<DisjointMut<AlignedVec64<u8>>>,
) {
    let p = unsafe { *FFISafe::get(p) };
    let left = unsafe { slice::from_raw_parts(left as *const LeftPixelRow<u16>, h as usize + 2) };
    let lpf = unsafe { FFISafe::get(lpf) };
    let lpf_off = unsafe { (lpf_ptr as isize - lpf.as_byte_mut_ptr() as isize) / 2 };

    wiener_filter_16bpc_inner(
        p,
        left,
        lpf,
        lpf_off,
        w as usize,
        h as usize,
        params,
        edges,
        5,
        bitdepth_max,
    );
}

// ============================================================================
// SGR (Self-Guided Restoration) FILTER IMPLEMENTATIONS
// ============================================================================

/// Compute box sum for 5x5 window (sum and sum of squares) for 8bpc
fn boxsum5_8bpc(
    sumsq: &mut [i32; (64 + 2 + 2) * REST_UNIT_STRIDE],
    sum: &mut [i16; (64 + 2 + 2) * REST_UNIT_STRIDE],
    src: &[u8; (64 + 3 + 3) * REST_UNIT_STRIDE],
    w: usize,
    h: usize,
) {
    // Vertical pass: sum 5 consecutive rows into intermediate buffer
    // Each output row[y] = src[y-2] + src[y-1] + src[y] + src[y+1] + src[y+2]
    for x in 0..w {
        // Initialize sliding window for first 4 rows
        let mut a = src[x] as i32;
        let mut a2 = a * a;
        let mut b = src[1 * REST_UNIT_STRIDE + x] as i32;
        let mut b2 = b * b;
        let mut c = src[2 * REST_UNIT_STRIDE + x] as i32;
        let mut c2 = c * c;
        let mut d = src[3 * REST_UNIT_STRIDE + x] as i32;
        let mut d2 = d * d;

        // Process rows starting from row 2 (first valid output)
        for y in 2..(h - 2) {
            let s_idx = (y + 2) * REST_UNIT_STRIDE + x;
            let e = src[s_idx] as i32;
            let e2 = e * e;

            let out_idx = y * REST_UNIT_STRIDE + x;
            sum[out_idx] = (a + b + c + d + e) as i16;
            sumsq[out_idx] = a2 + b2 + c2 + d2 + e2;

            // Slide window
            a = b;
            a2 = b2;
            b = c;
            b2 = c2;
            c = d;
            c2 = d2;
            d = e;
            d2 = e2;
        }
    }

    // Horizontal pass: sum 5 consecutive columns
    for y in 2..(h - 2) {
        let row_start = y * REST_UNIT_STRIDE;

        let mut a = sum[row_start] as i32;
        let mut a2 = sumsq[row_start];
        let mut b = sum[row_start + 1] as i32;
        let mut b2 = sumsq[row_start + 1];
        let mut c = sum[row_start + 2] as i32;
        let mut c2 = sumsq[row_start + 2];
        let mut d = sum[row_start + 3] as i32;
        let mut d2 = sumsq[row_start + 3];

        for x in 2..(w - 2) {
            let e = sum[row_start + x + 2] as i32;
            let e2 = sumsq[row_start + x + 2];

            sum[row_start + x] = (a + b + c + d + e) as i16;
            sumsq[row_start + x] = a2 + b2 + c2 + d2 + e2;

            a = b;
            a2 = b2;
            b = c;
            b2 = c2;
            c = d;
            c2 = d2;
            d = e;
            d2 = e2;
        }
    }
}

/// Compute box sum for 3x3 window (sum and sum of squares) for 8bpc
fn boxsum3_8bpc(
    sumsq: &mut [i32; (64 + 2 + 2) * REST_UNIT_STRIDE],
    sum: &mut [i16; (64 + 2 + 2) * REST_UNIT_STRIDE],
    src: &[u8; (64 + 3 + 3) * REST_UNIT_STRIDE],
    w: usize,
    h: usize,
) {
    // Vertical pass: sum 3 consecutive rows
    for x in 0..w {
        let mut a = src[x] as i32;
        let mut a2 = a * a;
        let mut b = src[1 * REST_UNIT_STRIDE + x] as i32;
        let mut b2 = b * b;

        for y in 1..(h - 1) {
            let s_idx = (y + 1) * REST_UNIT_STRIDE + x;
            let c = src[s_idx] as i32;
            let c2 = c * c;

            let out_idx = y * REST_UNIT_STRIDE + x;
            sum[out_idx] = (a + b + c) as i16;
            sumsq[out_idx] = a2 + b2 + c2;

            a = b;
            a2 = b2;
            b = c;
            b2 = c2;
        }
    }

    // Horizontal pass: sum 3 consecutive columns
    for y in 1..(h - 1) {
        let row_start = y * REST_UNIT_STRIDE;

        let mut a = sum[row_start] as i32;
        let mut a2 = sumsq[row_start];
        let mut b = sum[row_start + 1] as i32;
        let mut b2 = sumsq[row_start + 1];

        for x in 1..(w - 1) {
            let c = sum[row_start + x + 1] as i32;
            let c2 = sumsq[row_start + x + 1];

            sum[row_start + x] = (a + b + c) as i16;
            sumsq[row_start + x] = a2 + b2 + c2;

            a = b;
            a2 = b2;
            b = c;
            b2 = c2;
        }
    }
}

/// Self-guided filter computation for 8bpc
///
/// Computes the filter coefficients and applies the guided filter.
/// n = 25 for 5x5, n = 9 for 3x3
fn selfguided_filter_8bpc(
    dst: &mut [i16; 64 * MAX_RESTORATION_WIDTH],
    src: &[u8; (64 + 3 + 3) * REST_UNIT_STRIDE],
    w: usize,
    h: usize,
    n: i32,
    s: u32,
) {
    let sgr_one_by_x: u32 = if n == 25 { 164 } else { 455 };

    // Working buffers
    let mut sumsq = [0i32; (64 + 2 + 2) * REST_UNIT_STRIDE];
    let mut sum = [0i16; (64 + 2 + 2) * REST_UNIT_STRIDE];

    let step = if n == 25 { 2 } else { 1 };

    if n == 25 {
        boxsum5_8bpc(&mut sumsq, &mut sum, src, w + 6, h + 6);
    } else {
        boxsum3_8bpc(&mut sumsq, &mut sum, src, w + 6, h + 6);
    }

    // Calculate filter coefficients a and b
    // After this loop: sumsq contains 'a', sum contains 'b' (renamed)
    let base = 2 * REST_UNIT_STRIDE + 3;

    for row_offset in (0..(h + 2)).step_by(step) {
        let row_start = (row_offset as isize - 1) as usize;
        let aa_base = base + row_start * REST_UNIT_STRIDE - REST_UNIT_STRIDE;

        for i in 0..(w + 2) {
            let idx = aa_base + i;
            let a_val = sumsq.get(idx).copied().unwrap_or(0);
            let b_val = sum.get(idx).copied().unwrap_or(0) as i32;

            let p = cmp::max(a_val * n - b_val * b_val, 0) as u32;
            let z = (p * s + (1 << 19)) >> 20;
            let x = dav1d_sgr_x_by_x.0[cmp::min(z, 255) as usize] as u32;

            // Store: a = x * b * sgr_one_by_x, b = x
            if let Some(aa) = sumsq.get_mut(idx) {
                *aa = ((x * (b_val as u32) * sgr_one_by_x + (1 << 11)) >> 12) as i32;
            }
            if let Some(bb) = sum.get_mut(idx) {
                *bb = x as i16;
            }
        }
    }

    // Apply neighbor-weighted filter to produce output
    let src_base = 3 * REST_UNIT_STRIDE + 3;

    if n == 25 {
        // 5x5: use six_neighbors weighting, step by 2 rows
        let mut j = 0usize;
        while j < h.saturating_sub(1) {
            // Even row: full 6-neighbor calculation
            for i in 0..w {
                let idx = base + j * REST_UNIT_STRIDE + i;
                let b_six = {
                    let above = sum
                        .get(idx.wrapping_sub(REST_UNIT_STRIDE))
                        .copied()
                        .unwrap_or(0) as i32;
                    let below = sum.get(idx + REST_UNIT_STRIDE).copied().unwrap_or(0) as i32;
                    let above_left = sum
                        .get(idx.wrapping_sub(REST_UNIT_STRIDE).wrapping_sub(1))
                        .copied()
                        .unwrap_or(0) as i32;
                    let above_right = sum
                        .get(idx.wrapping_sub(REST_UNIT_STRIDE) + 1)
                        .copied()
                        .unwrap_or(0) as i32;
                    let below_left =
                        sum.get(idx + REST_UNIT_STRIDE - 1).copied().unwrap_or(0) as i32;
                    let below_right =
                        sum.get(idx + REST_UNIT_STRIDE + 1).copied().unwrap_or(0) as i32;
                    (above + below) * 6 + (above_left + above_right + below_left + below_right) * 5
                };
                let a_six = {
                    let above = sumsq
                        .get(idx.wrapping_sub(REST_UNIT_STRIDE))
                        .copied()
                        .unwrap_or(0);
                    let below = sumsq.get(idx + REST_UNIT_STRIDE).copied().unwrap_or(0);
                    let above_left = sumsq
                        .get(idx.wrapping_sub(REST_UNIT_STRIDE).wrapping_sub(1))
                        .copied()
                        .unwrap_or(0);
                    let above_right = sumsq
                        .get(idx.wrapping_sub(REST_UNIT_STRIDE) + 1)
                        .copied()
                        .unwrap_or(0);
                    let below_left = sumsq.get(idx + REST_UNIT_STRIDE - 1).copied().unwrap_or(0);
                    let below_right = sumsq.get(idx + REST_UNIT_STRIDE + 1).copied().unwrap_or(0);
                    (above + below) * 6 + (above_left + above_right + below_left + below_right) * 5
                };

                let src_val = src[src_base + j * REST_UNIT_STRIDE + i] as i32;
                dst[j * MAX_RESTORATION_WIDTH + i] =
                    ((a_six - b_six * src_val + (1 << 8)) >> 9) as i16;
            }

            // Odd row: simplified 3-neighbor horizontal calculation
            if j + 1 < h {
                for i in 0..w {
                    let idx = base + (j + 1) * REST_UNIT_STRIDE + i;
                    let b_horiz = {
                        let center = sum.get(idx).copied().unwrap_or(0) as i32;
                        let left = sum.get(idx.wrapping_sub(1)).copied().unwrap_or(0) as i32;
                        let right = sum.get(idx + 1).copied().unwrap_or(0) as i32;
                        center * 6 + (left + right) * 5
                    };
                    let a_horiz = {
                        let center = sumsq.get(idx).copied().unwrap_or(0);
                        let left = sumsq.get(idx.wrapping_sub(1)).copied().unwrap_or(0);
                        let right = sumsq.get(idx + 1).copied().unwrap_or(0);
                        center * 6 + (left + right) * 5
                    };

                    let src_val = src[src_base + (j + 1) * REST_UNIT_STRIDE + i] as i32;
                    dst[(j + 1) * MAX_RESTORATION_WIDTH + i] =
                        ((a_horiz - b_horiz * src_val + (1 << 7)) >> 8) as i16;
                }
            }
            j += 2;
        }
        // Handle last row if height is odd
        if j < h {
            for i in 0..w {
                let idx = base + j * REST_UNIT_STRIDE + i;
                let b_six = {
                    let above = sum
                        .get(idx.wrapping_sub(REST_UNIT_STRIDE))
                        .copied()
                        .unwrap_or(0) as i32;
                    let below = sum.get(idx + REST_UNIT_STRIDE).copied().unwrap_or(0) as i32;
                    let above_left = sum
                        .get(idx.wrapping_sub(REST_UNIT_STRIDE).wrapping_sub(1))
                        .copied()
                        .unwrap_or(0) as i32;
                    let above_right = sum
                        .get(idx.wrapping_sub(REST_UNIT_STRIDE) + 1)
                        .copied()
                        .unwrap_or(0) as i32;
                    let below_left =
                        sum.get(idx + REST_UNIT_STRIDE - 1).copied().unwrap_or(0) as i32;
                    let below_right =
                        sum.get(idx + REST_UNIT_STRIDE + 1).copied().unwrap_or(0) as i32;
                    (above + below) * 6 + (above_left + above_right + below_left + below_right) * 5
                };
                let a_six = {
                    let above = sumsq
                        .get(idx.wrapping_sub(REST_UNIT_STRIDE))
                        .copied()
                        .unwrap_or(0);
                    let below = sumsq.get(idx + REST_UNIT_STRIDE).copied().unwrap_or(0);
                    let above_left = sumsq
                        .get(idx.wrapping_sub(REST_UNIT_STRIDE).wrapping_sub(1))
                        .copied()
                        .unwrap_or(0);
                    let above_right = sumsq
                        .get(idx.wrapping_sub(REST_UNIT_STRIDE) + 1)
                        .copied()
                        .unwrap_or(0);
                    let below_left = sumsq.get(idx + REST_UNIT_STRIDE - 1).copied().unwrap_or(0);
                    let below_right = sumsq.get(idx + REST_UNIT_STRIDE + 1).copied().unwrap_or(0);
                    (above + below) * 6 + (above_left + above_right + below_left + below_right) * 5
                };

                let src_val = src[src_base + j * REST_UNIT_STRIDE + i] as i32;
                dst[j * MAX_RESTORATION_WIDTH + i] =
                    ((a_six - b_six * src_val + (1 << 8)) >> 9) as i16;
            }
        }
    } else {
        // 3x3: use eight_neighbors weighting
        for j in 0..h {
            for i in 0..w {
                let idx = base + j * REST_UNIT_STRIDE + i;
                let b_eight = {
                    let center = sum.get(idx).copied().unwrap_or(0) as i32;
                    let left = sum.get(idx.wrapping_sub(1)).copied().unwrap_or(0) as i32;
                    let right = sum.get(idx + 1).copied().unwrap_or(0) as i32;
                    let above = sum
                        .get(idx.wrapping_sub(REST_UNIT_STRIDE))
                        .copied()
                        .unwrap_or(0) as i32;
                    let below = sum.get(idx + REST_UNIT_STRIDE).copied().unwrap_or(0) as i32;
                    let above_left = sum
                        .get(idx.wrapping_sub(REST_UNIT_STRIDE).wrapping_sub(1))
                        .copied()
                        .unwrap_or(0) as i32;
                    let above_right = sum
                        .get(idx.wrapping_sub(REST_UNIT_STRIDE) + 1)
                        .copied()
                        .unwrap_or(0) as i32;
                    let below_left =
                        sum.get(idx + REST_UNIT_STRIDE - 1).copied().unwrap_or(0) as i32;
                    let below_right =
                        sum.get(idx + REST_UNIT_STRIDE + 1).copied().unwrap_or(0) as i32;
                    (center + left + right + above + below) * 4
                        + (above_left + above_right + below_left + below_right) * 3
                };
                let a_eight = {
                    let center = sumsq.get(idx).copied().unwrap_or(0);
                    let left = sumsq.get(idx.wrapping_sub(1)).copied().unwrap_or(0);
                    let right = sumsq.get(idx + 1).copied().unwrap_or(0);
                    let above = sumsq
                        .get(idx.wrapping_sub(REST_UNIT_STRIDE))
                        .copied()
                        .unwrap_or(0);
                    let below = sumsq.get(idx + REST_UNIT_STRIDE).copied().unwrap_or(0);
                    let above_left = sumsq
                        .get(idx.wrapping_sub(REST_UNIT_STRIDE).wrapping_sub(1))
                        .copied()
                        .unwrap_or(0);
                    let above_right = sumsq
                        .get(idx.wrapping_sub(REST_UNIT_STRIDE) + 1)
                        .copied()
                        .unwrap_or(0);
                    let below_left = sumsq.get(idx + REST_UNIT_STRIDE - 1).copied().unwrap_or(0);
                    let below_right = sumsq.get(idx + REST_UNIT_STRIDE + 1).copied().unwrap_or(0);
                    (center + left + right + above + below) * 4
                        + (above_left + above_right + below_left + below_right) * 3
                };

                let src_val = src[src_base + j * REST_UNIT_STRIDE + i] as i32;
                dst[j * MAX_RESTORATION_WIDTH + i] =
                    ((a_eight - b_eight * src_val + (1 << 8)) >> 9) as i16;
            }
        }
    }
}

/// SGR 5x5 filter for 8bpc
fn sgr_5x5_8bpc_inner(
    p: PicOffset,
    left: &[LeftPixelRow<u8>],
    lpf: &DisjointMut<AlignedVec64<u8>>,
    lpf_off: isize,
    w: usize,
    h: usize,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
) {
    let mut tmp = [0u8; (64 + 3 + 3) * REST_UNIT_STRIDE];
    let mut dst = [0i16; 64 * MAX_RESTORATION_WIDTH];

    padding::<BitDepth8>(&mut tmp, p, left, lpf, lpf_off, w, h, edges);

    let sgr = params.sgr();
    selfguided_filter_8bpc(&mut dst, &tmp, w, h, 25, sgr.s0);

    let w0 = sgr.w0 as i32;
    let stride = p.pixel_stride::<BitDepth8>();

    for j in 0..h {
        let mut p_row = (p + (j as isize * stride)).slice_mut::<BitDepth8>(w);
        for i in 0..w {
            let v = w0 * dst[j * MAX_RESTORATION_WIDTH + i] as i32;
            p_row[i] = iclip(p_row[i] as i32 + ((v + (1 << 10)) >> 11), 0, 255) as u8;
        }
    }
}

/// SGR 3x3 filter for 8bpc
fn sgr_3x3_8bpc_inner(
    p: PicOffset,
    left: &[LeftPixelRow<u8>],
    lpf: &DisjointMut<AlignedVec64<u8>>,
    lpf_off: isize,
    w: usize,
    h: usize,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
) {
    let mut tmp = [0u8; (64 + 3 + 3) * REST_UNIT_STRIDE];
    let mut dst = [0i16; 64 * MAX_RESTORATION_WIDTH];

    padding::<BitDepth8>(&mut tmp, p, left, lpf, lpf_off, w, h, edges);

    let sgr = params.sgr();
    selfguided_filter_8bpc(&mut dst, &tmp, w, h, 9, sgr.s1);

    let w1 = sgr.w1 as i32;
    let stride = p.pixel_stride::<BitDepth8>();

    for j in 0..h {
        let mut p_row = (p + (j as isize * stride)).slice_mut::<BitDepth8>(w);
        for i in 0..w {
            let v = w1 * dst[j * MAX_RESTORATION_WIDTH + i] as i32;
            p_row[i] = iclip(p_row[i] as i32 + ((v + (1 << 10)) >> 11), 0, 255) as u8;
        }
    }
}

/// SGR mix filter for 8bpc (combines 5x5 and 3x3)
fn sgr_mix_8bpc_inner(
    p: PicOffset,
    left: &[LeftPixelRow<u8>],
    lpf: &DisjointMut<AlignedVec64<u8>>,
    lpf_off: isize,
    w: usize,
    h: usize,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
) {
    let mut tmp = [0u8; (64 + 3 + 3) * REST_UNIT_STRIDE];
    let mut dst0 = [0i16; 64 * MAX_RESTORATION_WIDTH];
    let mut dst1 = [0i16; 64 * MAX_RESTORATION_WIDTH];

    padding::<BitDepth8>(&mut tmp, p, left, lpf, lpf_off, w, h, edges);

    let sgr = params.sgr();
    selfguided_filter_8bpc(&mut dst0, &tmp, w, h, 25, sgr.s0);
    selfguided_filter_8bpc(&mut dst1, &tmp, w, h, 9, sgr.s1);

    let w0 = sgr.w0 as i32;
    let w1 = sgr.w1 as i32;
    let stride = p.pixel_stride::<BitDepth8>();

    for j in 0..h {
        let mut p_row = (p + (j as isize * stride)).slice_mut::<BitDepth8>(w);
        for i in 0..w {
            let v = w0 * dst0[j * MAX_RESTORATION_WIDTH + i] as i32
                + w1 * dst1[j * MAX_RESTORATION_WIDTH + i] as i32;
            p_row[i] = iclip(p_row[i] as i32 + ((v + (1 << 10)) >> 11), 0, 255) as u8;
        }
    }
}

// ============================================================================
// SGR FFI WRAPPERS - 8BPC
// ============================================================================

#[cfg(all(feature = "asm", target_arch = "aarch64"))]
pub unsafe extern "C" fn sgr_filter_5x5_8bpc_neon(
    _p_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    left: *const LeftPixelRow<DynPixel>,
    lpf_ptr: *const DynPixel,
    w: c_int,
    h: c_int,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    _bitdepth_max: c_int,
    p: *const FFISafe<PicOffset>,
    lpf: *const FFISafe<DisjointMut<AlignedVec64<u8>>>,
) {
    let p = unsafe { *FFISafe::get(p) };
    let left = unsafe { slice::from_raw_parts(left as *const LeftPixelRow<u8>, h as usize) };
    let lpf = unsafe { FFISafe::get(lpf) };
    let lpf_off = unsafe { lpf_ptr as isize - lpf.as_byte_mut_ptr() as isize };

    sgr_5x5_8bpc_inner(p, left, lpf, lpf_off, w as usize, h as usize, params, edges);
}

#[cfg(all(feature = "asm", target_arch = "aarch64"))]
pub unsafe extern "C" fn sgr_filter_3x3_8bpc_neon(
    _p_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    left: *const LeftPixelRow<DynPixel>,
    lpf_ptr: *const DynPixel,
    w: c_int,
    h: c_int,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    _bitdepth_max: c_int,
    p: *const FFISafe<PicOffset>,
    lpf: *const FFISafe<DisjointMut<AlignedVec64<u8>>>,
) {
    let p = unsafe { *FFISafe::get(p) };
    let left = unsafe { slice::from_raw_parts(left as *const LeftPixelRow<u8>, h as usize) };
    let lpf = unsafe { FFISafe::get(lpf) };
    let lpf_off = unsafe { lpf_ptr as isize - lpf.as_byte_mut_ptr() as isize };

    sgr_3x3_8bpc_inner(p, left, lpf, lpf_off, w as usize, h as usize, params, edges);
}

#[cfg(all(feature = "asm", target_arch = "aarch64"))]
pub unsafe extern "C" fn sgr_filter_mix_8bpc_neon(
    _p_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    left: *const LeftPixelRow<DynPixel>,
    lpf_ptr: *const DynPixel,
    w: c_int,
    h: c_int,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    _bitdepth_max: c_int,
    p: *const FFISafe<PicOffset>,
    lpf: *const FFISafe<DisjointMut<AlignedVec64<u8>>>,
) {
    let p = unsafe { *FFISafe::get(p) };
    let left = unsafe { slice::from_raw_parts(left as *const LeftPixelRow<u8>, h as usize) };
    let lpf = unsafe { FFISafe::get(lpf) };
    let lpf_off = unsafe { lpf_ptr as isize - lpf.as_byte_mut_ptr() as isize };

    sgr_mix_8bpc_inner(p, left, lpf, lpf_off, w as usize, h as usize, params, edges);
}

// ============================================================================
// SGR 16BPC IMPLEMENTATIONS
// ============================================================================

/// Compute box sum for 5x5 window for 16bpc
fn boxsum5_16bpc(
    sumsq: &mut [i64; (64 + 2 + 2) * REST_UNIT_STRIDE],
    sum: &mut [i32; (64 + 2 + 2) * REST_UNIT_STRIDE],
    src: &[u16; (64 + 3 + 3) * REST_UNIT_STRIDE],
    w: usize,
    h: usize,
) {
    // Vertical pass: sum 5 consecutive rows
    for x in 0..w {
        let mut a = src[x] as i64;
        let mut a2 = a * a;
        let mut b = src[1 * REST_UNIT_STRIDE + x] as i64;
        let mut b2 = b * b;
        let mut c = src[2 * REST_UNIT_STRIDE + x] as i64;
        let mut c2 = c * c;
        let mut d = src[3 * REST_UNIT_STRIDE + x] as i64;
        let mut d2 = d * d;

        for y in 2..(h - 2) {
            let s_idx = (y + 2) * REST_UNIT_STRIDE + x;
            let e = src[s_idx] as i64;
            let e2 = e * e;

            let out_idx = y * REST_UNIT_STRIDE + x;
            sum[out_idx] = (a + b + c + d + e) as i32;
            sumsq[out_idx] = a2 + b2 + c2 + d2 + e2;

            a = b;
            a2 = b2;
            b = c;
            b2 = c2;
            c = d;
            c2 = d2;
            d = e;
            d2 = e2;
        }
    }

    // Horizontal pass: sum 5 consecutive columns
    for y in 2..(h - 2) {
        let row_start = y * REST_UNIT_STRIDE;

        let mut a = sum[row_start] as i64;
        let mut a2 = sumsq[row_start];
        let mut b = sum[row_start + 1] as i64;
        let mut b2 = sumsq[row_start + 1];
        let mut c = sum[row_start + 2] as i64;
        let mut c2 = sumsq[row_start + 2];
        let mut d = sum[row_start + 3] as i64;
        let mut d2 = sumsq[row_start + 3];

        for x in 2..(w - 2) {
            let e = sum[row_start + x + 2] as i64;
            let e2 = sumsq[row_start + x + 2];

            sum[row_start + x] = (a + b + c + d + e) as i32;
            sumsq[row_start + x] = a2 + b2 + c2 + d2 + e2;

            a = b;
            a2 = b2;
            b = c;
            b2 = c2;
            c = d;
            c2 = d2;
            d = e;
            d2 = e2;
        }
    }
}

/// Compute box sum for 3x3 window for 16bpc
fn boxsum3_16bpc(
    sumsq: &mut [i64; (64 + 2 + 2) * REST_UNIT_STRIDE],
    sum: &mut [i32; (64 + 2 + 2) * REST_UNIT_STRIDE],
    src: &[u16; (64 + 3 + 3) * REST_UNIT_STRIDE],
    w: usize,
    h: usize,
) {
    // Vertical pass: sum 3 consecutive rows
    for x in 0..w {
        let mut a = src[x] as i64;
        let mut a2 = a * a;
        let mut b = src[1 * REST_UNIT_STRIDE + x] as i64;
        let mut b2 = b * b;

        for y in 1..(h - 1) {
            let s_idx = (y + 1) * REST_UNIT_STRIDE + x;
            let c = src[s_idx] as i64;
            let c2 = c * c;

            let out_idx = y * REST_UNIT_STRIDE + x;
            sum[out_idx] = (a + b + c) as i32;
            sumsq[out_idx] = a2 + b2 + c2;

            a = b;
            a2 = b2;
            b = c;
            b2 = c2;
        }
    }

    // Horizontal pass: sum 3 consecutive columns
    for y in 1..(h - 1) {
        let row_start = y * REST_UNIT_STRIDE;

        let mut a = sum[row_start] as i64;
        let mut a2 = sumsq[row_start];
        let mut b = sum[row_start + 1] as i64;
        let mut b2 = sumsq[row_start + 1];

        for x in 1..(w - 1) {
            let c = sum[row_start + x + 1] as i64;
            let c2 = sumsq[row_start + x + 1];

            sum[row_start + x] = (a + b + c) as i32;
            sumsq[row_start + x] = a2 + b2 + c2;

            a = b;
            a2 = b2;
            b = c;
            b2 = c2;
        }
    }
}

/// Self-guided filter computation for 16bpc
fn selfguided_filter_16bpc(
    dst: &mut [i32; 64 * MAX_RESTORATION_WIDTH],
    src: &[u16; (64 + 3 + 3) * REST_UNIT_STRIDE],
    w: usize,
    h: usize,
    n: i32,
    s: u32,
    bitdepth_max: i32,
) {
    let sgr_one_by_x: u32 = if n == 25 { 164 } else { 455 };
    let bitdepth_min_8 = if bitdepth_max == 1023 { 2 } else { 4 }; // 10bpc - 8 or 12bpc - 8

    // Working buffers - use i64 for sumsq to handle 12bpc without overflow
    let mut sumsq = [0i64; (64 + 2 + 2) * REST_UNIT_STRIDE];
    let mut sum = [0i32; (64 + 2 + 2) * REST_UNIT_STRIDE];

    let step = if n == 25 { 2 } else { 1 };

    if n == 25 {
        boxsum5_16bpc(&mut sumsq, &mut sum, src, w + 6, h + 6);
    } else {
        boxsum3_16bpc(&mut sumsq, &mut sum, src, w + 6, h + 6);
    }

    // Calculate filter coefficients
    let base = 2 * REST_UNIT_STRIDE + 3;

    for row_offset in (0..(h + 2)).step_by(step) {
        let row_start = (row_offset as isize - 1) as usize;
        let aa_base = base + row_start * REST_UNIT_STRIDE - REST_UNIT_STRIDE;

        for i in 0..(w + 2) {
            let idx = aa_base + i;
            let a_val = sumsq.get(idx).copied().unwrap_or(0);
            let b_val = sum.get(idx).copied().unwrap_or(0) as i64;

            // Scale for bitdepth
            let a_scaled = (a_val >> (2 * bitdepth_min_8)) as i64;
            let b_scaled = b_val >> bitdepth_min_8;

            let p = cmp::max(a_scaled * (n as i64) - b_scaled * b_scaled, 0) as u64;
            let z = ((p * (s as u64) + (1 << 19)) >> 20) as u32;
            let x = dav1d_sgr_x_by_x.0[cmp::min(z, 255) as usize] as u32;

            // Store coefficients
            if let Some(aa) = sumsq.get_mut(idx) {
                *aa =
                    ((x as u64 * (b_val as u64) * (sgr_one_by_x as u64) + (1 << 11)) >> 12) as i64;
            }
            if let Some(bb) = sum.get_mut(idx) {
                *bb = x as i32;
            }
        }
    }

    // Apply neighbor-weighted filter to produce output
    let src_base = 3 * REST_UNIT_STRIDE + 3;

    if n == 25 {
        // 5x5: use six_neighbors weighting
        let mut j = 0usize;
        while j < h.saturating_sub(1) {
            for i in 0..w {
                let idx = base + j * REST_UNIT_STRIDE + i;
                let b_six = {
                    let above = sum
                        .get(idx.wrapping_sub(REST_UNIT_STRIDE))
                        .copied()
                        .unwrap_or(0) as i64;
                    let below = sum.get(idx + REST_UNIT_STRIDE).copied().unwrap_or(0) as i64;
                    let above_left = sum
                        .get(idx.wrapping_sub(REST_UNIT_STRIDE).wrapping_sub(1))
                        .copied()
                        .unwrap_or(0) as i64;
                    let above_right = sum
                        .get(idx.wrapping_sub(REST_UNIT_STRIDE) + 1)
                        .copied()
                        .unwrap_or(0) as i64;
                    let below_left =
                        sum.get(idx + REST_UNIT_STRIDE - 1).copied().unwrap_or(0) as i64;
                    let below_right =
                        sum.get(idx + REST_UNIT_STRIDE + 1).copied().unwrap_or(0) as i64;
                    (above + below) * 6 + (above_left + above_right + below_left + below_right) * 5
                };
                let a_six = {
                    let above = sumsq
                        .get(idx.wrapping_sub(REST_UNIT_STRIDE))
                        .copied()
                        .unwrap_or(0);
                    let below = sumsq.get(idx + REST_UNIT_STRIDE).copied().unwrap_or(0);
                    let above_left = sumsq
                        .get(idx.wrapping_sub(REST_UNIT_STRIDE).wrapping_sub(1))
                        .copied()
                        .unwrap_or(0);
                    let above_right = sumsq
                        .get(idx.wrapping_sub(REST_UNIT_STRIDE) + 1)
                        .copied()
                        .unwrap_or(0);
                    let below_left = sumsq.get(idx + REST_UNIT_STRIDE - 1).copied().unwrap_or(0);
                    let below_right = sumsq.get(idx + REST_UNIT_STRIDE + 1).copied().unwrap_or(0);
                    (above + below) * 6 + (above_left + above_right + below_left + below_right) * 5
                };

                let src_val = src[src_base + j * REST_UNIT_STRIDE + i] as i64;
                dst[j * MAX_RESTORATION_WIDTH + i] =
                    ((a_six - b_six * src_val + (1 << 8)) >> 9) as i32;
            }

            if j + 1 < h {
                for i in 0..w {
                    let idx = base + (j + 1) * REST_UNIT_STRIDE + i;
                    let b_horiz = {
                        let center = sum.get(idx).copied().unwrap_or(0) as i64;
                        let left = sum.get(idx.wrapping_sub(1)).copied().unwrap_or(0) as i64;
                        let right = sum.get(idx + 1).copied().unwrap_or(0) as i64;
                        center * 6 + (left + right) * 5
                    };
                    let a_horiz = {
                        let center = sumsq.get(idx).copied().unwrap_or(0);
                        let left = sumsq.get(idx.wrapping_sub(1)).copied().unwrap_or(0);
                        let right = sumsq.get(idx + 1).copied().unwrap_or(0);
                        center * 6 + (left + right) * 5
                    };

                    let src_val = src[src_base + (j + 1) * REST_UNIT_STRIDE + i] as i64;
                    dst[(j + 1) * MAX_RESTORATION_WIDTH + i] =
                        ((a_horiz - b_horiz * src_val + (1 << 7)) >> 8) as i32;
                }
            }
            j += 2;
        }
        if j < h {
            for i in 0..w {
                let idx = base + j * REST_UNIT_STRIDE + i;
                let b_six = {
                    let above = sum
                        .get(idx.wrapping_sub(REST_UNIT_STRIDE))
                        .copied()
                        .unwrap_or(0) as i64;
                    let below = sum.get(idx + REST_UNIT_STRIDE).copied().unwrap_or(0) as i64;
                    let above_left = sum
                        .get(idx.wrapping_sub(REST_UNIT_STRIDE).wrapping_sub(1))
                        .copied()
                        .unwrap_or(0) as i64;
                    let above_right = sum
                        .get(idx.wrapping_sub(REST_UNIT_STRIDE) + 1)
                        .copied()
                        .unwrap_or(0) as i64;
                    let below_left =
                        sum.get(idx + REST_UNIT_STRIDE - 1).copied().unwrap_or(0) as i64;
                    let below_right =
                        sum.get(idx + REST_UNIT_STRIDE + 1).copied().unwrap_or(0) as i64;
                    (above + below) * 6 + (above_left + above_right + below_left + below_right) * 5
                };
                let a_six = {
                    let above = sumsq
                        .get(idx.wrapping_sub(REST_UNIT_STRIDE))
                        .copied()
                        .unwrap_or(0);
                    let below = sumsq.get(idx + REST_UNIT_STRIDE).copied().unwrap_or(0);
                    let above_left = sumsq
                        .get(idx.wrapping_sub(REST_UNIT_STRIDE).wrapping_sub(1))
                        .copied()
                        .unwrap_or(0);
                    let above_right = sumsq
                        .get(idx.wrapping_sub(REST_UNIT_STRIDE) + 1)
                        .copied()
                        .unwrap_or(0);
                    let below_left = sumsq.get(idx + REST_UNIT_STRIDE - 1).copied().unwrap_or(0);
                    let below_right = sumsq.get(idx + REST_UNIT_STRIDE + 1).copied().unwrap_or(0);
                    (above + below) * 6 + (above_left + above_right + below_left + below_right) * 5
                };

                let src_val = src[src_base + j * REST_UNIT_STRIDE + i] as i64;
                dst[j * MAX_RESTORATION_WIDTH + i] =
                    ((a_six - b_six * src_val + (1 << 8)) >> 9) as i32;
            }
        }
    } else {
        // 3x3: use eight_neighbors weighting
        for j in 0..h {
            for i in 0..w {
                let idx = base + j * REST_UNIT_STRIDE + i;
                let b_eight = {
                    let center = sum.get(idx).copied().unwrap_or(0) as i64;
                    let left = sum.get(idx.wrapping_sub(1)).copied().unwrap_or(0) as i64;
                    let right = sum.get(idx + 1).copied().unwrap_or(0) as i64;
                    let above = sum
                        .get(idx.wrapping_sub(REST_UNIT_STRIDE))
                        .copied()
                        .unwrap_or(0) as i64;
                    let below = sum.get(idx + REST_UNIT_STRIDE).copied().unwrap_or(0) as i64;
                    let above_left = sum
                        .get(idx.wrapping_sub(REST_UNIT_STRIDE).wrapping_sub(1))
                        .copied()
                        .unwrap_or(0) as i64;
                    let above_right = sum
                        .get(idx.wrapping_sub(REST_UNIT_STRIDE) + 1)
                        .copied()
                        .unwrap_or(0) as i64;
                    let below_left =
                        sum.get(idx + REST_UNIT_STRIDE - 1).copied().unwrap_or(0) as i64;
                    let below_right =
                        sum.get(idx + REST_UNIT_STRIDE + 1).copied().unwrap_or(0) as i64;
                    (center + left + right + above + below) * 4
                        + (above_left + above_right + below_left + below_right) * 3
                };
                let a_eight = {
                    let center = sumsq.get(idx).copied().unwrap_or(0);
                    let left = sumsq.get(idx.wrapping_sub(1)).copied().unwrap_or(0);
                    let right = sumsq.get(idx + 1).copied().unwrap_or(0);
                    let above = sumsq
                        .get(idx.wrapping_sub(REST_UNIT_STRIDE))
                        .copied()
                        .unwrap_or(0);
                    let below = sumsq.get(idx + REST_UNIT_STRIDE).copied().unwrap_or(0);
                    let above_left = sumsq
                        .get(idx.wrapping_sub(REST_UNIT_STRIDE).wrapping_sub(1))
                        .copied()
                        .unwrap_or(0);
                    let above_right = sumsq
                        .get(idx.wrapping_sub(REST_UNIT_STRIDE) + 1)
                        .copied()
                        .unwrap_or(0);
                    let below_left = sumsq.get(idx + REST_UNIT_STRIDE - 1).copied().unwrap_or(0);
                    let below_right = sumsq.get(idx + REST_UNIT_STRIDE + 1).copied().unwrap_or(0);
                    (center + left + right + above + below) * 4
                        + (above_left + above_right + below_left + below_right) * 3
                };

                let src_val = src[src_base + j * REST_UNIT_STRIDE + i] as i64;
                dst[j * MAX_RESTORATION_WIDTH + i] =
                    ((a_eight - b_eight * src_val + (1 << 8)) >> 9) as i32;
            }
        }
    }
}

/// SGR 5x5 filter for 16bpc
fn sgr_5x5_16bpc_inner(
    p: PicOffset,
    left: &[LeftPixelRow<u16>],
    lpf: &DisjointMut<AlignedVec64<u8>>,
    lpf_off: isize,
    w: usize,
    h: usize,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    bitdepth_max: i32,
) {
    let mut tmp = [0u16; (64 + 3 + 3) * REST_UNIT_STRIDE];
    let mut dst = [0i32; 64 * MAX_RESTORATION_WIDTH];

    padding::<BitDepth16>(&mut tmp, p, left, lpf, lpf_off, w, h, edges);

    let sgr = params.sgr();
    selfguided_filter_16bpc(&mut dst, &tmp, w, h, 25, sgr.s0, bitdepth_max);

    let w0 = sgr.w0 as i32;
    let stride = p.pixel_stride::<BitDepth16>();

    for j in 0..h {
        let mut p_row = (p + (j as isize * stride)).slice_mut::<BitDepth16>(w);
        for i in 0..w {
            let v = w0 * dst[j * MAX_RESTORATION_WIDTH + i];
            p_row[i] = iclip(p_row[i] as i32 + ((v + (1 << 10)) >> 11), 0, bitdepth_max) as u16;
        }
    }
}

/// SGR 3x3 filter for 16bpc
fn sgr_3x3_16bpc_inner(
    p: PicOffset,
    left: &[LeftPixelRow<u16>],
    lpf: &DisjointMut<AlignedVec64<u8>>,
    lpf_off: isize,
    w: usize,
    h: usize,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    bitdepth_max: i32,
) {
    let mut tmp = [0u16; (64 + 3 + 3) * REST_UNIT_STRIDE];
    let mut dst = [0i32; 64 * MAX_RESTORATION_WIDTH];

    padding::<BitDepth16>(&mut tmp, p, left, lpf, lpf_off, w, h, edges);

    let sgr = params.sgr();
    selfguided_filter_16bpc(&mut dst, &tmp, w, h, 9, sgr.s1, bitdepth_max);

    let w1 = sgr.w1 as i32;
    let stride = p.pixel_stride::<BitDepth16>();

    for j in 0..h {
        let mut p_row = (p + (j as isize * stride)).slice_mut::<BitDepth16>(w);
        for i in 0..w {
            let v = w1 * dst[j * MAX_RESTORATION_WIDTH + i];
            p_row[i] = iclip(p_row[i] as i32 + ((v + (1 << 10)) >> 11), 0, bitdepth_max) as u16;
        }
    }
}

/// SGR mix filter for 16bpc
fn sgr_mix_16bpc_inner(
    p: PicOffset,
    left: &[LeftPixelRow<u16>],
    lpf: &DisjointMut<AlignedVec64<u8>>,
    lpf_off: isize,
    w: usize,
    h: usize,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    bitdepth_max: i32,
) {
    let mut tmp = [0u16; (64 + 3 + 3) * REST_UNIT_STRIDE];
    let mut dst0 = [0i32; 64 * MAX_RESTORATION_WIDTH];
    let mut dst1 = [0i32; 64 * MAX_RESTORATION_WIDTH];

    padding::<BitDepth16>(&mut tmp, p, left, lpf, lpf_off, w, h, edges);

    let sgr = params.sgr();
    selfguided_filter_16bpc(&mut dst0, &tmp, w, h, 25, sgr.s0, bitdepth_max);
    selfguided_filter_16bpc(&mut dst1, &tmp, w, h, 9, sgr.s1, bitdepth_max);

    let w0 = sgr.w0 as i32;
    let w1 = sgr.w1 as i32;
    let stride = p.pixel_stride::<BitDepth16>();

    for j in 0..h {
        let mut p_row = (p + (j as isize * stride)).slice_mut::<BitDepth16>(w);
        for i in 0..w {
            let v =
                w0 * dst0[j * MAX_RESTORATION_WIDTH + i] + w1 * dst1[j * MAX_RESTORATION_WIDTH + i];
            p_row[i] = iclip(p_row[i] as i32 + ((v + (1 << 10)) >> 11), 0, bitdepth_max) as u16;
        }
    }
}

// ============================================================================
// SGR FFI WRAPPERS - 16BPC
// ============================================================================

#[cfg(all(feature = "asm", target_arch = "aarch64"))]
pub unsafe extern "C" fn sgr_filter_5x5_16bpc_neon(
    _p_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    left: *const LeftPixelRow<DynPixel>,
    lpf_ptr: *const DynPixel,
    w: c_int,
    h: c_int,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    bitdepth_max: c_int,
    p: *const FFISafe<PicOffset>,
    lpf: *const FFISafe<DisjointMut<AlignedVec64<u8>>>,
) {
    let p = unsafe { *FFISafe::get(p) };
    let left = unsafe { slice::from_raw_parts(left as *const LeftPixelRow<u16>, h as usize) };
    let lpf = unsafe { FFISafe::get(lpf) };
    let lpf_off = unsafe { (lpf_ptr as isize - lpf.as_byte_mut_ptr() as isize) / 2 };

    sgr_5x5_16bpc_inner(
        p,
        left,
        lpf,
        lpf_off,
        w as usize,
        h as usize,
        params,
        edges,
        bitdepth_max,
    );
}

#[cfg(all(feature = "asm", target_arch = "aarch64"))]
pub unsafe extern "C" fn sgr_filter_3x3_16bpc_neon(
    _p_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    left: *const LeftPixelRow<DynPixel>,
    lpf_ptr: *const DynPixel,
    w: c_int,
    h: c_int,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    bitdepth_max: c_int,
    p: *const FFISafe<PicOffset>,
    lpf: *const FFISafe<DisjointMut<AlignedVec64<u8>>>,
) {
    let p = unsafe { *FFISafe::get(p) };
    let left = unsafe { slice::from_raw_parts(left as *const LeftPixelRow<u16>, h as usize) };
    let lpf = unsafe { FFISafe::get(lpf) };
    let lpf_off = unsafe { (lpf_ptr as isize - lpf.as_byte_mut_ptr() as isize) / 2 };

    sgr_3x3_16bpc_inner(
        p,
        left,
        lpf,
        lpf_off,
        w as usize,
        h as usize,
        params,
        edges,
        bitdepth_max,
    );
}

#[cfg(all(feature = "asm", target_arch = "aarch64"))]
pub unsafe extern "C" fn sgr_filter_mix_16bpc_neon(
    _p_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    left: *const LeftPixelRow<DynPixel>,
    lpf_ptr: *const DynPixel,
    w: c_int,
    h: c_int,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    bitdepth_max: c_int,
    p: *const FFISafe<PicOffset>,
    lpf: *const FFISafe<DisjointMut<AlignedVec64<u8>>>,
) {
    let p = unsafe { *FFISafe::get(p) };
    let left = unsafe { slice::from_raw_parts(left as *const LeftPixelRow<u16>, h as usize) };
    let lpf = unsafe { FFISafe::get(lpf) };
    let lpf_off = unsafe { (lpf_ptr as isize - lpf.as_byte_mut_ptr() as isize) / 2 };

    sgr_mix_16bpc_inner(
        p,
        left,
        lpf,
        lpf_off,
        w as usize,
        h as usize,
        params,
        edges,
        bitdepth_max,
    );
}

/// Safe dispatch for lr_filter on aarch64. Returns true if NEON was used.
#[cfg(target_arch = "aarch64")]
pub fn lr_filter_dispatch<BD: BitDepth>(
    variant: usize,
    dst: PicOffset,
    left: &[LeftPixelRow<BD::Pixel>],
    lpf: &DisjointMut<AlignedVec64<u8>>,
    lpf_off: isize,
    w: c_int,
    h: c_int,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    bd: BD,
) -> bool {
    use crate::include::common::bitdepth::BPC;

    let w = w as usize;
    let h = h as usize;
    let left_8 =
        || unsafe { &*(left as *const [LeftPixelRow<BD::Pixel>] as *const [LeftPixelRow<u8>]) };
    let left_16 =
        || unsafe { &*(left as *const [LeftPixelRow<BD::Pixel>] as *const [LeftPixelRow<u16>]) };
    let bd_c = bd.into_c();

    // Call inner functions directly, bypassing FFI wrappers.
    match (BD::BPC, variant) {
        (BPC::BPC8, 0) => {
            wiener_filter_8bpc_inner(dst, left_8(), lpf, lpf_off, w, h, params, edges, 7)
        }
        (BPC::BPC8, 1) => {
            wiener_filter_8bpc_inner(dst, left_8(), lpf, lpf_off, w, h, params, edges, 5)
        }
        (BPC::BPC8, 2) => sgr_5x5_8bpc_inner(dst, left_8(), lpf, lpf_off, w, h, params, edges),
        (BPC::BPC8, 3) => sgr_3x3_8bpc_inner(dst, left_8(), lpf, lpf_off, w, h, params, edges),
        (BPC::BPC8, _) => sgr_mix_8bpc_inner(dst, left_8(), lpf, lpf_off, w, h, params, edges),
        (BPC::BPC16, 0) => {
            wiener_filter_16bpc_inner(dst, left_16(), lpf, lpf_off, w, h, params, edges, 7, bd_c)
        }
        (BPC::BPC16, 1) => {
            wiener_filter_16bpc_inner(dst, left_16(), lpf, lpf_off, w, h, params, edges, 5, bd_c)
        }
        (BPC::BPC16, 2) => {
            sgr_5x5_16bpc_inner(dst, left_16(), lpf, lpf_off, w, h, params, edges, bd_c)
        }
        (BPC::BPC16, 3) => {
            sgr_3x3_16bpc_inner(dst, left_16(), lpf, lpf_off, w, h, params, edges, bd_c)
        }
        (BPC::BPC16, _) => {
            sgr_mix_16bpc_inner(dst, left_16(), lpf, lpf_off, w, h, params, edges, bd_c)
        }
    }
    true
}
