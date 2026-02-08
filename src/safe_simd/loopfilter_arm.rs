//! Safe ARM NEON implementations for Loop Filter (Deblocking Filter)
//!
//! The loop filter removes blocking artifacts at transform block boundaries.

#![cfg_attr(not(feature = "asm"), deny(unsafe_code))]
#![allow(unused_imports)]
#![allow(clippy::too_many_arguments)]
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

use crate::include::common::bitdepth::AsPrimitive;
use crate::include::common::bitdepth::BitDepth;
use crate::include::common::bitdepth::DynPixel;
use crate::include::common::intops::iclip;
use crate::include::dav1d::picture::PicOffset;
use crate::src::align::Align16;
use crate::src::disjoint_mut::DisjointMut;
use crate::src::ffi_safe::FFISafe;
use crate::src::lf_mask::Av1FilterLUT;
use crate::src::with_offset::WithOffset;
use libc::ptrdiff_t;
use std::cmp;
use std::ffi::c_int;

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

#[inline(always)]
fn iclip_diff(v: i32, bitdepth_min_8: u8) -> i32 {
    iclip(
        v,
        -128 * (1 << bitdepth_min_8),
        128 * (1 << bitdepth_min_8) - 1,
    )
}

// ============================================================================
// CORE FILTER IMPLEMENTATIONS
// ============================================================================

/// Compute a buffer index from a base index and signed offset.
#[inline(always)]
fn signed_idx(base: usize, offset: isize) -> usize {
    base.wrapping_add_signed(offset)
}

/// Apply loop filter to an edge (scalar version, safe slice-based)
#[inline]
fn loop_filter_core<BD: BitDepth>(
    buf: &mut [BD::Pixel],
    base_idx: usize,
    e: i32,
    i: i32,
    h: i32,
    stridea: isize,
    strideb: isize,
    wd: i32,
    bitdepth_max: i32,
) {
    let bitdepth_min_8 = (BD::BITDEPTH - 8) as u8;
    let f = 1i32 << bitdepth_min_8;

    for idx in 0..4isize {
        let base = signed_idx(base_idx, idx * stridea);
        let px = |offset: isize| -> usize { signed_idx(base, strideb * offset) };

        let p1 = buf[px(-2)].as_::<i32>();
        let p0 = buf[px(-1)].as_::<i32>();
        let q0 = buf[px(0)].as_::<i32>();
        let q1 = buf[px(1)].as_::<i32>();

        let mut fm = (p1 - p0).abs() <= i
            && (q1 - q0).abs() <= i
            && (p0 - q0).abs() * 2 + ((p1 - q1).abs() >> 1) <= e;

        let (mut p2, mut p3, mut q2, mut q3) = (0, 0, 0, 0);
        let (mut p4, mut p5, mut p6, mut q4, mut q5, mut q6) = (0, 0, 0, 0, 0, 0);

        if wd > 4 {
            p2 = buf[px(-3)].as_::<i32>();
            q2 = buf[px(2)].as_::<i32>();
            fm &= (p2 - p1).abs() <= i && (q2 - q1).abs() <= i;

            if wd > 6 {
                p3 = buf[px(-4)].as_::<i32>();
                q3 = buf[px(3)].as_::<i32>();
                fm &= (p3 - p2).abs() <= i && (q3 - q2).abs() <= i;
            }
        }

        if !fm {
            continue;
        }

        let hm = if h != 0 {
            let hev = (p1 - p0).abs() > h || (q1 - q0).abs() > h;
            !hev
        } else {
            false
        };

        let mut flat8out = false;
        let mut flat8in = false;

        if wd >= 16 {
            p6 = buf[px(-7)].as_::<i32>();
            p5 = buf[px(-6)].as_::<i32>();
            p4 = buf[px(-5)].as_::<i32>();
            q4 = buf[px(4)].as_::<i32>();
            q5 = buf[px(5)].as_::<i32>();
            q6 = buf[px(6)].as_::<i32>();

            flat8out = (p6 - p0).abs() <= f
                && (p5 - p0).abs() <= f
                && (p4 - p0).abs() <= f
                && (q4 - q0).abs() <= f
                && (q5 - q0).abs() <= f
                && (q6 - q0).abs() <= f;
        }

        if wd >= 6 {
            flat8in = (p2 - p0).abs() <= f
                && (p1 - p0).abs() <= f
                && (q1 - q0).abs() <= f
                && (q2 - q0).abs() <= f;
        }

        if wd >= 8 {
            flat8in &= (p3 - p0).abs() <= f && (q3 - q0).abs() <= f;
        }

        let clamp_px = |val: i32| -> BD::Pixel { iclip(val, 0, bitdepth_max).as_::<BD::Pixel>() };

        if wd >= 16 && flat8out && flat8in {
            // Wide 16-tap filter
            buf[px(-6)] = clamp_px(
                (p6 + p6 + p6 + p6 + p6 + p6 * 2 + p5 * 2 + p4 * 2 + p3 + p2 + p1 + p0 + q0 + 8)
                    >> 4,
            );
            buf[px(-5)] = clamp_px(
                (p6 + p6 + p6 + p6 + p6 + p5 * 2 + p4 * 2 + p3 * 2 + p2 + p1 + p0 + q0 + q1 + 8)
                    >> 4,
            );
            buf[px(-4)] = clamp_px(
                (p6 + p6 + p6 + p6 + p5 + p4 * 2 + p3 * 2 + p2 * 2 + p1 + p0 + q0 + q1 + q2 + 8)
                    >> 4,
            );
            buf[px(-3)] = clamp_px(
                (p6 + p6 + p6 + p5 + p4 + p3 * 2 + p2 * 2 + p1 * 2 + p0 + q0 + q1 + q2 + q3 + 8)
                    >> 4,
            );
            buf[px(-2)] = clamp_px(
                (p6 + p6 + p5 + p4 + p3 + p2 * 2 + p1 * 2 + p0 * 2 + q0 + q1 + q2 + q3 + q4 + 8)
                    >> 4,
            );
            buf[px(-1)] = clamp_px(
                (p6 + p5 + p4 + p3 + p2 + p1 * 2 + p0 * 2 + q0 * 2 + q1 + q2 + q3 + q4 + q5 + 8)
                    >> 4,
            );
            buf[px(0)] = clamp_px(
                (p5 + p4 + p3 + p2 + p1 + p0 * 2 + q0 * 2 + q1 * 2 + q2 + q3 + q4 + q5 + q6 + 8)
                    >> 4,
            );
            buf[px(1)] = clamp_px(
                (p4 + p3 + p2 + p1 + p0 + q0 * 2 + q1 * 2 + q2 * 2 + q3 + q4 + q5 + q6 + q6 + 8)
                    >> 4,
            );
            buf[px(2)] = clamp_px(
                (p3 + p2 + p1 + p0 + q0 + q1 * 2 + q2 * 2 + q3 * 2 + q4 + q5 + q6 + q6 + q6 + 8)
                    >> 4,
            );
            buf[px(3)] = clamp_px(
                (p2 + p1 + p0 + q0 + q1 + q2 * 2 + q3 * 2 + q4 * 2 + q5 + q6 + q6 + q6 + q6 + 8)
                    >> 4,
            );
            buf[px(4)] = clamp_px(
                (p1 + p0 + q0 + q1 + q2 + q3 * 2 + q4 * 2 + q5 * 2 + q6 + q6 + q6 + q6 + q6 + 8)
                    >> 4,
            );
            buf[px(5)] = clamp_px(
                (p0 + q0 + q1 + q2 + q3 + q4 * 2 + q5 * 2 + q6 * 2 + q6 + q6 + q6 + q6 + q6 + 8)
                    >> 4,
            );
        } else if wd >= 8 && flat8in {
            // 8-tap filter
            buf[px(-3)] = clamp_px((p3 + p3 + p3 + 2 * p2 + p1 + p0 + q0 + 4) >> 3);
            buf[px(-2)] = clamp_px((p3 + p3 + p2 + 2 * p1 + p0 + q0 + q1 + 4) >> 3);
            buf[px(-1)] = clamp_px((p3 + p2 + p1 + 2 * p0 + q0 + q1 + q2 + 4) >> 3);
            buf[px(0)] = clamp_px((p2 + p1 + p0 + 2 * q0 + q1 + q2 + q3 + 4) >> 3);
            buf[px(1)] = clamp_px((p1 + p0 + q0 + 2 * q1 + q2 + q3 + q3 + 4) >> 3);
            buf[px(2)] = clamp_px((p0 + q0 + q1 + 2 * q2 + q3 + q3 + q3 + 4) >> 3);
        } else if wd >= 6 && flat8in {
            // 6-tap filter
            buf[px(-2)] = clamp_px((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3);
            buf[px(-1)] = clamp_px((p2 + p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3);
            buf[px(0)] = clamp_px((p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3);
            buf[px(1)] = clamp_px((p0 + 2 * q0 + 2 * q1 + 2 * q2 + q2 + 4) >> 3);
        } else if hm {
            // 4-tap filter with hev mask
            let f = iclip_diff((p1 - q1) + 3 * (q0 - p0), bitdepth_min_8);
            let f1 = cmp::min(f + 4, 127 << bitdepth_min_8) >> 3;
            let f2 = cmp::min(f + 3, 127 << bitdepth_min_8) >> 3;
            buf[px(-1)] = clamp_px((p0 + f1).clamp(0, bitdepth_max));
            buf[px(0)] = clamp_px((q0 - f2).clamp(0, bitdepth_max));
        } else {
            // Narrow 4-tap filter
            let f = iclip_diff(3 * (q0 - p0), bitdepth_min_8);
            let f1 = cmp::min(f + 4, 127 << bitdepth_min_8) >> 3;
            let f2 = cmp::min(f + 3, 127 << bitdepth_min_8) >> 3;
            buf[px(-1)] = clamp_px((p0 + f2).clamp(0, bitdepth_max));
            buf[px(0)] = clamp_px((q0 - f1).clamp(0, bitdepth_max));
            let f3 = (f1 + 1) >> 1;
            buf[px(-2)] = clamp_px((p1 + f3).clamp(0, bitdepth_max));
            buf[px(1)] = clamp_px((q1 - f3).clamp(0, bitdepth_max));
        }
    }
}

// ============================================================================
// SUPERBLOCK FILTER IMPLEMENTATIONS
// ============================================================================

#[allow(unsafe_code)]
fn lpf_h_sb_inner<BD: BitDepth, const YUV: usize>(
    dst: *mut BD::Pixel,
    stride: isize,
    mask: &[u32; 3],
    lvl_ptr: *const [u8; 4],
    _b4_stride: isize,
    lut: &Av1FilterLUT,
    w: i32,
    bitdepth_max: i32,
) {
    let vmask = [mask[0], mask[1], mask[2]];

    // Create a slice encompassing the accessible pixel region.
    // For H filter: base = x*4, stridea=1 (across columns), strideb=stride (across rows).
    // Access range: base + idx (0..3) + stride * offset (-7..=6).
    // We need pixels from dst.offset(stride * -7) to dst.offset((w*4-1) + 3 + stride * 6).
    // To avoid computing exact bounds, we use the pointer as a base with a large enough slice.
    // The negative stride offsets mean we need to start 7*|stride| pixels before dst.
    let abs_stride = stride.unsigned_abs();
    // Total rows accessed: 7 before + 1 at center + 6 after = 14 rows
    // Total columns: w*4 + 3 (for the 4 iterations of stridea=1)
    let cols = w as usize * 4 + 3;
    let slice_start_offset = 7 * abs_stride; // how far back from dst the slice starts
    let slice_len = slice_start_offset + cols + 6 * abs_stride;
    // SAFETY: The caller guarantees dst points into a valid pixel buffer with sufficient
    // extent in both directions for the filter's access pattern.
    let buf = unsafe {
        let start = dst.offset(-(7 * stride));
        core::slice::from_raw_parts_mut(start, slice_len)
    };
    // base_idx offset: dst is at index slice_start_offset within buf
    let dst_base = slice_start_offset;

    for x in 0..w as usize {
        // SAFETY: lvl_ptr points to a valid array of [u8; 4] with at least w entries
        let lvl = unsafe { &*lvl_ptr.add(x) };

        if lvl[0] == 0 && lvl[1] == 0 && lvl[2] == 0 && lvl[3] == 0 {
            continue;
        }

        let vm = (vmask[0] >> x) & 1 | ((vmask[1] >> x) & 1) << 1 | ((vmask[2] >> x) & 1) << 2;

        if vm == 0 {
            continue;
        }

        let l = lvl[0] as usize;
        let e = lut.e[l] as i32;
        let i = lut.i[l] as i32;
        let h = (lvl[0] >> 4) as i32;

        let wd = if YUV == 0 {
            match vm {
                1 => 4,
                2 => 6,
                3 | 4 | 5 | 6 | 7 => 8,
                _ => 4,
            }
        } else {
            match vm {
                1 => 4,
                2 | 3 | 4 | 5 | 6 | 7 => 6,
                _ => 4,
            }
        };

        // For horizontal filter, stridea=1 (pixels in a row), strideb=stride (move between rows)
        let base_idx = dst_base + x * 4;
        loop_filter_core::<BD>(buf, base_idx, e, i, h, 1, stride, wd, bitdepth_max);
    }
}

#[allow(unsafe_code)]
fn lpf_v_sb_inner<BD: BitDepth, const YUV: usize>(
    dst: *mut BD::Pixel,
    stride: isize,
    mask: &[u32; 3],
    lvl_ptr: *const [u8; 4],
    b4_stride: isize,
    lut: &Av1FilterLUT,
    w: i32,
    bitdepth_max: i32,
) {
    let vmask = [mask[0], mask[1], mask[2]];
    let b4_stride_u = b4_stride as usize;

    // For V filter: base = y*4*stride, stridea=stride (across rows), strideb=1 (across columns).
    // Access range: base + idx*stride (idx 0..3) + offset (-7..=6).
    // We need pixels from dst.offset(-7) to dst.offset((w-1)*4*stride + 3*stride + 6).
    let abs_stride = stride.unsigned_abs();
    // Total rows: (w-1)*4 + 3 + 1 = w*4 rows accessed via stridea
    let rows = w as usize * 4;
    // Columns: 7 before + 1 center + 6 after = 14
    let slice_start_offset: usize = 7; // pixels before dst
    let slice_len = slice_start_offset + (rows.saturating_sub(1)) * abs_stride + 7;
    // SAFETY: The caller guarantees dst points into a valid pixel buffer with sufficient
    // extent in both directions for the filter's access pattern.
    let buf = unsafe {
        let start = dst.offset(-7);
        core::slice::from_raw_parts_mut(start, slice_len)
    };
    let dst_base = slice_start_offset;

    for y in 0..w as usize {
        // SAFETY: lvl_ptr points to a valid array with sufficient stride-spaced entries
        let lvl = unsafe { &*lvl_ptr.add(y * b4_stride_u) };

        if lvl[0] == 0 && lvl[1] == 0 && lvl[2] == 0 && lvl[3] == 0 {
            continue;
        }

        let vm = (vmask[0] >> y) & 1 | ((vmask[1] >> y) & 1) << 1 | ((vmask[2] >> y) & 1) << 2;

        if vm == 0 {
            continue;
        }

        let l = lvl[0] as usize;
        let e = lut.e[l] as i32;
        let i = lut.i[l] as i32;
        let h = (lvl[0] >> 4) as i32;

        let wd = if YUV == 0 {
            match vm {
                1 => 4,
                2 => 6,
                3 | 4 | 5 | 6 | 7 => 8,
                _ => 4,
            }
        } else {
            match vm {
                1 => 4,
                2 | 3 | 4 | 5 | 6 | 7 => 6,
                _ => 4,
            }
        };

        // For vertical filter, stridea=stride (move between rows), strideb=1 (move in the filter direction)
        let base_idx = signed_idx(dst_base, y as isize * 4 * stride);
        loop_filter_core::<BD>(buf, base_idx, e, i, h, stride, 1, wd, bitdepth_max);
    }
}

// ============================================================================
// FFI WRAPPERS - 8BPC
// ============================================================================

#[cfg(all(feature = "asm", target_arch = "aarch64"))]
pub unsafe extern "C" fn lpf_h_sb_y_8bpc_neon(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    mask: &[u32; 3],
    lvl_ptr: *const [u8; 4],
    b4_stride: ptrdiff_t,
    lut: &Align16<Av1FilterLUT>,
    w: c_int,
    bitdepth_max: c_int,
    _dst: *const FFISafe<PicOffset>,
    _lvl: *const FFISafe<WithOffset<&DisjointMut<Vec<u8>>>>,
) {
    use crate::include::common::bitdepth::BitDepth8;
    lpf_h_sb_inner::<BitDepth8, 0>(
        dst_ptr as *mut u8,
        stride as isize,
        mask,
        lvl_ptr,
        b4_stride as isize,
        &lut.0,
        w,
        bitdepth_max,
    );
}

#[cfg(all(feature = "asm", target_arch = "aarch64"))]
pub unsafe extern "C" fn lpf_v_sb_y_8bpc_neon(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    mask: &[u32; 3],
    lvl_ptr: *const [u8; 4],
    b4_stride: ptrdiff_t,
    lut: &Align16<Av1FilterLUT>,
    w: c_int,
    bitdepth_max: c_int,
    _dst: *const FFISafe<PicOffset>,
    _lvl: *const FFISafe<WithOffset<&DisjointMut<Vec<u8>>>>,
) {
    use crate::include::common::bitdepth::BitDepth8;
    lpf_v_sb_inner::<BitDepth8, 0>(
        dst_ptr as *mut u8,
        stride as isize,
        mask,
        lvl_ptr,
        b4_stride as isize,
        &lut.0,
        w,
        bitdepth_max,
    );
}

#[cfg(all(feature = "asm", target_arch = "aarch64"))]
pub unsafe extern "C" fn lpf_h_sb_uv_8bpc_neon(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    mask: &[u32; 3],
    lvl_ptr: *const [u8; 4],
    b4_stride: ptrdiff_t,
    lut: &Align16<Av1FilterLUT>,
    w: c_int,
    bitdepth_max: c_int,
    _dst: *const FFISafe<PicOffset>,
    _lvl: *const FFISafe<WithOffset<&DisjointMut<Vec<u8>>>>,
) {
    use crate::include::common::bitdepth::BitDepth8;
    lpf_h_sb_inner::<BitDepth8, 1>(
        dst_ptr as *mut u8,
        stride as isize,
        mask,
        lvl_ptr,
        b4_stride as isize,
        &lut.0,
        w,
        bitdepth_max,
    );
}

#[cfg(all(feature = "asm", target_arch = "aarch64"))]
pub unsafe extern "C" fn lpf_v_sb_uv_8bpc_neon(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    mask: &[u32; 3],
    lvl_ptr: *const [u8; 4],
    b4_stride: ptrdiff_t,
    lut: &Align16<Av1FilterLUT>,
    w: c_int,
    bitdepth_max: c_int,
    _dst: *const FFISafe<PicOffset>,
    _lvl: *const FFISafe<WithOffset<&DisjointMut<Vec<u8>>>>,
) {
    use crate::include::common::bitdepth::BitDepth8;
    lpf_v_sb_inner::<BitDepth8, 1>(
        dst_ptr as *mut u8,
        stride as isize,
        mask,
        lvl_ptr,
        b4_stride as isize,
        &lut.0,
        w,
        bitdepth_max,
    );
}

// ============================================================================
// FFI WRAPPERS - 16BPC
// ============================================================================

#[cfg(all(feature = "asm", target_arch = "aarch64"))]
pub unsafe extern "C" fn lpf_h_sb_y_16bpc_neon(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    mask: &[u32; 3],
    lvl_ptr: *const [u8; 4],
    b4_stride: ptrdiff_t,
    lut: &Align16<Av1FilterLUT>,
    w: c_int,
    bitdepth_max: c_int,
    _dst: *const FFISafe<PicOffset>,
    _lvl: *const FFISafe<WithOffset<&DisjointMut<Vec<u8>>>>,
) {
    use crate::include::common::bitdepth::BitDepth16;
    let stride_u16 = (stride / 2) as isize;
    lpf_h_sb_inner::<BitDepth16, 0>(
        dst_ptr as *mut u16,
        stride_u16,
        mask,
        lvl_ptr,
        b4_stride as isize,
        &lut.0,
        w,
        bitdepth_max,
    );
}

#[cfg(all(feature = "asm", target_arch = "aarch64"))]
pub unsafe extern "C" fn lpf_v_sb_y_16bpc_neon(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    mask: &[u32; 3],
    lvl_ptr: *const [u8; 4],
    b4_stride: ptrdiff_t,
    lut: &Align16<Av1FilterLUT>,
    w: c_int,
    bitdepth_max: c_int,
    _dst: *const FFISafe<PicOffset>,
    _lvl: *const FFISafe<WithOffset<&DisjointMut<Vec<u8>>>>,
) {
    use crate::include::common::bitdepth::BitDepth16;
    let stride_u16 = (stride / 2) as isize;
    lpf_v_sb_inner::<BitDepth16, 0>(
        dst_ptr as *mut u16,
        stride_u16,
        mask,
        lvl_ptr,
        b4_stride as isize,
        &lut.0,
        w,
        bitdepth_max,
    );
}

#[cfg(all(feature = "asm", target_arch = "aarch64"))]
pub unsafe extern "C" fn lpf_h_sb_uv_16bpc_neon(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    mask: &[u32; 3],
    lvl_ptr: *const [u8; 4],
    b4_stride: ptrdiff_t,
    lut: &Align16<Av1FilterLUT>,
    w: c_int,
    bitdepth_max: c_int,
    _dst: *const FFISafe<PicOffset>,
    _lvl: *const FFISafe<WithOffset<&DisjointMut<Vec<u8>>>>,
) {
    use crate::include::common::bitdepth::BitDepth16;
    let stride_u16 = (stride / 2) as isize;
    lpf_h_sb_inner::<BitDepth16, 1>(
        dst_ptr as *mut u16,
        stride_u16,
        mask,
        lvl_ptr,
        b4_stride as isize,
        &lut.0,
        w,
        bitdepth_max,
    );
}

#[cfg(all(feature = "asm", target_arch = "aarch64"))]
pub unsafe extern "C" fn lpf_v_sb_uv_16bpc_neon(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    mask: &[u32; 3],
    lvl_ptr: *const [u8; 4],
    b4_stride: ptrdiff_t,
    lut: &Align16<Av1FilterLUT>,
    w: c_int,
    bitdepth_max: c_int,
    _dst: *const FFISafe<PicOffset>,
    _lvl: *const FFISafe<WithOffset<&DisjointMut<Vec<u8>>>>,
) {
    use crate::include::common::bitdepth::BitDepth16;
    let stride_u16 = (stride / 2) as isize;
    lpf_v_sb_inner::<BitDepth16, 1>(
        dst_ptr as *mut u16,
        stride_u16,
        mask,
        lvl_ptr,
        b4_stride as isize,
        &lut.0,
        w,
        bitdepth_max,
    );
}

/// Safe dispatch for loopfilter_sb on aarch64. Returns true if SIMD was used.
#[cfg(target_arch = "aarch64")]
pub fn loopfilter_sb_dispatch<BD: BitDepth>(
    dst: PicOffset,
    stride: ptrdiff_t,
    mask: &[u32; 3],
    lvl: WithOffset<&DisjointMut<Vec<u8>>>,
    b4_stride: isize,
    lut: &Align16<Av1FilterLUT>,
    w: c_int,
    bitdepth_max: c_int,
    is_y: bool,
    is_v: bool,
) -> bool {
    use crate::include::common::bitdepth::BPC;
    use zerocopy::AsBytes;

    // Loopfilter accesses pixels at negative offsets from the base pointer
    // (filtering across block boundaries), so use full_guard_mut.
    let (mut dst_guard, dst_base) = dst.full_guard_mut::<BD>();
    let dst_ptr = {
        let bytes = dst_guard.as_bytes_mut();
        let base_byte = dst_base * std::mem::size_of::<BD::Pixel>();
        &mut bytes[base_byte] as *mut u8 as *mut BD::Pixel
    };
    assert!(lvl.offset <= lvl.data.len());
    // SAFETY: `lvl.offset` is in bounds, checked above.
    let lvl_ptr = unsafe { lvl.data.as_mut_ptr().add(lvl.offset) };
    let lvl_ptr = lvl_ptr.cast::<[u8; 4]>();

    // Call inner functions directly, bypassing FFI wrappers.
    match (is_y, is_v) {
        (true, false) => lpf_h_sb_inner::<BD, 0>(
            dst_ptr,
            stride as isize,
            mask,
            lvl_ptr,
            b4_stride,
            &lut.0,
            w,
            bitdepth_max,
        ),
        (true, true) => lpf_v_sb_inner::<BD, 0>(
            dst_ptr,
            stride as isize,
            mask,
            lvl_ptr,
            b4_stride,
            &lut.0,
            w,
            bitdepth_max,
        ),
        (false, false) => lpf_h_sb_inner::<BD, 1>(
            dst_ptr,
            stride as isize,
            mask,
            lvl_ptr,
            b4_stride,
            &lut.0,
            w,
            bitdepth_max,
        ),
        (false, true) => lpf_v_sb_inner::<BD, 1>(
            dst_ptr,
            stride as isize,
            mask,
            lvl_ptr,
            b4_stride,
            &lut.0,
            w,
            bitdepth_max,
        ),
    }
    true
}
