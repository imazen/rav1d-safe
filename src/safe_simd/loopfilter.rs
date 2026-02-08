//! Safe SIMD implementations for Loop Filter (Deblocking Filter)
//!
//! The loop filter removes blocking artifacts at transform block boundaries.
//! It operates on edges between adjacent blocks, filtering up to 7 pixels
//! on each side of the edge.
//!
//! Key operations:
//! - Filter strength calculation based on quantization
//! - Flatness detection (flat8in, flat8out)
//! - Different filter widths (4, 6, 8, 16 pixels)
//! - Horizontal and vertical edge filtering
//!
//! This module uses safe slice-based pixel access. The dispatch function is fully safe.
//! is in `loopfilter_sb_dispatch` where raw pointers from PicOffset/DisjointMut
//! are converted to slices. All inner functions are fully safe.

#![cfg_attr(not(feature = "asm"), forbid(unsafe_code))]
#![allow(unused_imports)]

#[cfg(target_arch = "x86_64")]
use archmage::{Desktop64, SimdToken};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

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

/// Clamp difference value for bitdepth
#[inline(always)]
fn iclip_diff(v: i32, bitdepth_min_8: u8) -> i32 {
    iclip(
        v,
        -128 * (1 << bitdepth_min_8),
        128 * (1 << bitdepth_min_8) - 1,
    )
}

/// Compute a signed index from a base usize and signed offset.
#[inline(always)]
fn signed_idx(base: usize, offset: isize) -> usize {
    (base as isize + offset) as usize
}

// ============================================================================
// CORE LOOP FILTER (4 pixels at a time)
// ============================================================================

/// Core loop filter for 8bpc - processes 4 pixels
/// `buf` is the pixel buffer, `base` is the offset to the edge point.
/// `stridea` is the stride between the 4 parallel pixels.
/// `strideb` is the stride in the filter direction.
#[cfg(target_arch = "x86_64")]
fn loop_filter_4_8bpc(
    buf: &mut [u8],
    base: usize,
    e: i32,
    i: i32,
    h: i32,
    stridea: isize,
    strideb: isize,
    wd: i32,
    bitdepth_max: i32,
) {
    let f = 1i32;

    for idx in 0..4isize {
        let edge = signed_idx(base, idx * stridea);

        let get_px = |offset: isize| -> i32 { buf[signed_idx(edge, strideb * offset)] as i32 };

        let p1 = get_px(-2);
        let p0 = get_px(-1);
        let q0 = get_px(0);
        let q1 = get_px(1);

        // Filter mask calculation
        let mut fm = (p1 - p0).abs() <= i
            && (q1 - q0).abs() <= i
            && (p0 - q0).abs() * 2 + ((p1 - q1).abs() >> 1) <= e;

        let (mut p2, mut p3, mut q2, mut q3) = (0, 0, 0, 0);
        let (mut p4, mut p5, mut p6, mut q4, mut q5, mut q6) = (0, 0, 0, 0, 0, 0);

        if wd > 4 {
            p2 = get_px(-3);
            q2 = get_px(2);
            fm &= (p2 - p1).abs() <= i && (q2 - q1).abs() <= i;

            if wd > 6 {
                p3 = get_px(-4);
                q3 = get_px(3);
                fm &= (p3 - p2).abs() <= i && (q3 - q2).abs() <= i;
            }
        }

        if !fm {
            continue;
        }

        let mut flat8out = false;
        let mut flat8in = false;

        if wd >= 16 {
            p6 = get_px(-7);
            p5 = get_px(-6);
            p4 = get_px(-5);
            q4 = get_px(4);
            q5 = get_px(5);
            q6 = get_px(6);

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

        // Write helper — sets pixel at offset from edge
        let set_px = |buf: &mut [u8], offset: isize, val: i32| {
            buf[signed_idx(edge, strideb * offset)] = val.clamp(0, bitdepth_max) as u8;
        };

        if wd >= 16 && flat8out && flat8in {
            // Wide filter (16 taps)
            set_px(
                buf,
                -6,
                (p6 + p6 + p6 + p6 + p6 + p6 * 2 + p5 * 2 + p4 * 2 + p3 + p2 + p1 + p0 + q0 + 8)
                    >> 4,
            );
            set_px(
                buf,
                -5,
                (p6 + p6 + p6 + p6 + p6 + p5 * 2 + p4 * 2 + p3 * 2 + p2 + p1 + p0 + q0 + q1 + 8)
                    >> 4,
            );
            set_px(
                buf,
                -4,
                (p6 + p6 + p6 + p6 + p5 + p4 * 2 + p3 * 2 + p2 * 2 + p1 + p0 + q0 + q1 + q2 + 8)
                    >> 4,
            );
            set_px(
                buf,
                -3,
                (p6 + p6 + p6 + p5 + p4 + p3 * 2 + p2 * 2 + p1 * 2 + p0 + q0 + q1 + q2 + q3 + 8)
                    >> 4,
            );
            set_px(
                buf,
                -2,
                (p6 + p6 + p5 + p4 + p3 + p2 * 2 + p1 * 2 + p0 * 2 + q0 + q1 + q2 + q3 + q4 + 8)
                    >> 4,
            );
            set_px(
                buf,
                -1,
                (p6 + p5 + p4 + p3 + p2 + p1 * 2 + p0 * 2 + q0 * 2 + q1 + q2 + q3 + q4 + q5 + 8)
                    >> 4,
            );
            set_px(
                buf,
                0,
                (p5 + p4 + p3 + p2 + p1 + p0 * 2 + q0 * 2 + q1 * 2 + q2 + q3 + q4 + q5 + q6 + 8)
                    >> 4,
            );
            set_px(
                buf,
                1,
                (p4 + p3 + p2 + p1 + p0 + q0 * 2 + q1 * 2 + q2 * 2 + q3 + q4 + q5 + q6 + q6 + 8)
                    >> 4,
            );
            set_px(
                buf,
                2,
                (p3 + p2 + p1 + p0 + q0 + q1 * 2 + q2 * 2 + q3 * 2 + q4 + q5 + q6 + q6 + q6 + 8)
                    >> 4,
            );
            set_px(
                buf,
                3,
                (p2 + p1 + p0 + q0 + q1 + q2 * 2 + q3 * 2 + q4 * 2 + q5 + q6 + q6 + q6 + q6 + 8)
                    >> 4,
            );
            set_px(
                buf,
                4,
                (p1 + p0 + q0 + q1 + q2 + q3 * 2 + q4 * 2 + q5 * 2 + q6 + q6 + q6 + q6 + q6 + 8)
                    >> 4,
            );
            set_px(
                buf,
                5,
                (p0 + q0 + q1 + q2 + q3 + q4 * 2 + q5 * 2 + q6 * 2 + q6 + q6 + q6 + q6 + q6 + 8)
                    >> 4,
            );
        } else if wd >= 8 && flat8in {
            // 8-tap filter
            set_px(buf, -3, (p3 + p3 + p3 + 2 * p2 + p1 + p0 + q0 + 4) >> 3);
            set_px(buf, -2, (p3 + p3 + p2 + 2 * p1 + p0 + q0 + q1 + 4) >> 3);
            set_px(buf, -1, (p3 + p2 + p1 + 2 * p0 + q0 + q1 + q2 + 4) >> 3);
            set_px(buf, 0, (p2 + p1 + p0 + 2 * q0 + q1 + q2 + q3 + 4) >> 3);
            set_px(buf, 1, (p1 + p0 + q0 + 2 * q1 + q2 + q3 + q3 + 4) >> 3);
            set_px(buf, 2, (p0 + q0 + q1 + 2 * q2 + q3 + q3 + q3 + 4) >> 3);
        } else if wd == 6 && flat8in {
            // 6-tap filter
            set_px(buf, -2, (p2 + 2 * p2 + 2 * p1 + 2 * p0 + q0 + 4) >> 3);
            set_px(buf, -1, (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3);
            set_px(buf, 0, (p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3);
            set_px(buf, 1, (p0 + 2 * q0 + 2 * q1 + 2 * q2 + q2 + 4) >> 3);
        } else {
            // Narrow filter (4-tap)
            let hev = (p1 - p0).abs() > h || (q1 - q0).abs() > h;

            if hev {
                let f = iclip_diff(p1 - q1, 0);
                let f = iclip_diff(3 * (q0 - p0) + f, 0);

                let f1 = cmp::min(f + 4, 127) >> 3;
                let f2 = cmp::min(f + 3, 127) >> 3;

                set_px(buf, -1, p0 + f2);
                set_px(buf, 0, q0 - f1);
            } else {
                let f = iclip_diff(3 * (q0 - p0), 0);

                let f1 = cmp::min(f + 4, 127) >> 3;
                let f2 = cmp::min(f + 3, 127) >> 3;

                set_px(buf, -1, p0 + f2);
                set_px(buf, 0, q0 - f1);

                let f = (f1 + 1) >> 1;
                set_px(buf, -2, p1 + f);
                set_px(buf, 1, q1 - f);
            }
        }
    }
}

// ============================================================================
// SUPERBLOCK FILTER FUNCTIONS (8bpc)
// ============================================================================

/// Read level value from lvl slice at the given offset.
/// Each entry is [u8; 4], we read the first byte.
#[inline(always)]
fn read_lvl(lvl: &[[u8; 4]], offset: usize) -> u8 {
    lvl[offset][0]
}

/// Loop filter for Y plane, horizontal edges (8bpc)
#[cfg(target_arch = "x86_64")]
fn lpf_h_sb_y_8bpc_inner(
    buf: &mut [u8],
    mut dst_offset: usize,
    stride: isize,
    vmask: &[u32; 3],
    lvl: &[[u8; 4]],
    b4_stride: isize,
    lut: &Align16<Av1FilterLUT>,
    _w: i32,
    bitdepth_max: i32,
) {
    let stridea = stride;
    let strideb = 1isize;
    let b4_stridea = b4_stride as usize;
    let b4_strideb = 1usize;

    let vm = vmask[0] | vmask[1] | vmask[2];
    let mut lvl_offset = 0usize;

    let mut xy = 1u32;
    while vm & !xy.wrapping_sub(1) != 0 {
        if vm & xy != 0 {
            let lvl_val = read_lvl(lvl, lvl_offset);
            let l = if lvl_val != 0 {
                lvl_val
            } else {
                if lvl_offset >= 4 * b4_strideb {
                    read_lvl(lvl, lvl_offset - 4 * b4_strideb)
                } else {
                    0
                }
            };

            if l != 0 {
                let h = (l >> 4) as i32;
                let e = lut.0.e[l as usize] as i32;
                let i = lut.0.i[l as usize] as i32;

                let idx = if vmask[2] & xy != 0 {
                    16
                } else if vmask[1] & xy != 0 {
                    8
                } else {
                    4
                };

                loop_filter_4_8bpc(
                    buf,
                    dst_offset,
                    e,
                    i,
                    h,
                    stridea,
                    strideb,
                    idx,
                    bitdepth_max,
                );
            }
        }

        xy <<= 1;
        dst_offset = signed_idx(dst_offset, 4 * stridea);
        lvl_offset += 4 * b4_stridea;
    }
}

/// Loop filter for Y plane, vertical edges (8bpc)
#[cfg(target_arch = "x86_64")]
fn lpf_v_sb_y_8bpc_inner(
    buf: &mut [u8],
    mut dst_offset: usize,
    stride: isize,
    vmask: &[u32; 3],
    lvl: &[[u8; 4]],
    b4_stride: isize,
    lut: &Align16<Av1FilterLUT>,
    _w: i32,
    bitdepth_max: i32,
) {
    let stridea = 1isize;
    let strideb = stride;
    let b4_stridea = 1usize;
    let b4_strideb = b4_stride as usize;

    let vm = vmask[0] | vmask[1] | vmask[2];
    let mut lvl_offset = 0usize;

    let mut xy = 1u32;
    while vm & !xy.wrapping_sub(1) != 0 {
        if vm & xy != 0 {
            let lvl_val = read_lvl(lvl, lvl_offset);
            let l = if lvl_val != 0 {
                lvl_val
            } else {
                if lvl_offset >= 4 * b4_strideb {
                    read_lvl(lvl, lvl_offset - 4 * b4_strideb)
                } else {
                    0
                }
            };

            if l != 0 {
                let h = (l >> 4) as i32;
                let e = lut.0.e[l as usize] as i32;
                let i = lut.0.i[l as usize] as i32;

                let idx = if vmask[2] & xy != 0 {
                    16
                } else if vmask[1] & xy != 0 {
                    8
                } else {
                    4
                };

                loop_filter_4_8bpc(
                    buf,
                    dst_offset,
                    e,
                    i,
                    h,
                    stridea,
                    strideb,
                    idx,
                    bitdepth_max,
                );
            }
        }

        xy <<= 1;
        dst_offset = signed_idx(dst_offset, 4 * stridea);
        lvl_offset += 4 * b4_stridea;
    }
}

/// Loop filter for UV planes, horizontal edges (8bpc)
#[cfg(target_arch = "x86_64")]
fn lpf_h_sb_uv_8bpc_inner(
    buf: &mut [u8],
    mut dst_offset: usize,
    stride: isize,
    vmask: &[u32; 3],
    lvl: &[[u8; 4]],
    b4_stride: isize,
    lut: &Align16<Av1FilterLUT>,
    _w: i32,
    bitdepth_max: i32,
) {
    let stridea = stride;
    let strideb = 1isize;
    let b4_stridea = b4_stride as usize;
    let b4_strideb = 1usize;

    let vm = vmask[0] | vmask[1];
    let mut lvl_offset = 0usize;

    let mut xy = 1u32;
    while vm & !xy.wrapping_sub(1) != 0 {
        if vm & xy != 0 {
            let lvl_val = read_lvl(lvl, lvl_offset);
            let l = if lvl_val != 0 {
                lvl_val
            } else {
                if lvl_offset >= 4 * b4_strideb {
                    read_lvl(lvl, lvl_offset - 4 * b4_strideb)
                } else {
                    0
                }
            };

            if l != 0 {
                let h = (l >> 4) as i32;
                let e = lut.0.e[l as usize] as i32;
                let i = lut.0.i[l as usize] as i32;

                let idx = if vmask[1] & xy != 0 { 6 } else { 4 };

                loop_filter_4_8bpc(
                    buf,
                    dst_offset,
                    e,
                    i,
                    h,
                    stridea,
                    strideb,
                    idx,
                    bitdepth_max,
                );
            }
        }

        xy <<= 1;
        dst_offset = signed_idx(dst_offset, 4 * stridea);
        lvl_offset += 4 * b4_stridea;
    }
}

/// Loop filter for UV planes, vertical edges (8bpc)
#[cfg(target_arch = "x86_64")]
fn lpf_v_sb_uv_8bpc_inner(
    buf: &mut [u8],
    mut dst_offset: usize,
    stride: isize,
    vmask: &[u32; 3],
    lvl: &[[u8; 4]],
    b4_stride: isize,
    lut: &Align16<Av1FilterLUT>,
    _w: i32,
    bitdepth_max: i32,
) {
    let stridea = 1isize;
    let strideb = stride;
    let b4_stridea = 1usize;
    let b4_strideb = b4_stride as usize;

    let vm = vmask[0] | vmask[1];
    let mut lvl_offset = 0usize;

    let mut xy = 1u32;
    while vm & !xy.wrapping_sub(1) != 0 {
        if vm & xy != 0 {
            let lvl_val = read_lvl(lvl, lvl_offset);
            let l = if lvl_val != 0 {
                lvl_val
            } else {
                if lvl_offset >= 4 * b4_strideb {
                    read_lvl(lvl, lvl_offset - 4 * b4_strideb)
                } else {
                    0
                }
            };

            if l != 0 {
                let h = (l >> 4) as i32;
                let e = lut.0.e[l as usize] as i32;
                let i = lut.0.i[l as usize] as i32;

                let idx = if vmask[1] & xy != 0 { 6 } else { 4 };

                loop_filter_4_8bpc(
                    buf,
                    dst_offset,
                    e,
                    i,
                    h,
                    stridea,
                    strideb,
                    idx,
                    bitdepth_max,
                );
            }
        }

        xy <<= 1;
        dst_offset = signed_idx(dst_offset, 4 * stridea);
        lvl_offset += 4 * b4_stridea;
    }
}

// ============================================================================
// FFI WRAPPERS (8bpc) — only compiled with asm feature
// ============================================================================

/// FFI wrapper for Y horizontal filter
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn lpf_h_sb_y_8bpc_avx2(
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
    // Determine buffer size needed: conservative upper bound
    let buf_len = compute_buf_len_u8(stride as isize, w);
    let buf = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_len) };
    let lvl =
        unsafe { std::slice::from_raw_parts(lvl_ptr, compute_lvl_len(b4_stride as isize, w)) };
    lpf_h_sb_y_8bpc_inner(
        buf,
        0,
        stride as isize,
        mask,
        lvl,
        b4_stride as isize,
        lut,
        w,
        bitdepth_max,
    );
}

/// FFI wrapper for Y vertical filter
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn lpf_v_sb_y_8bpc_avx2(
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
    let buf_len = compute_buf_len_u8(stride as isize, w);
    let buf = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_len) };
    let lvl =
        unsafe { std::slice::from_raw_parts(lvl_ptr, compute_lvl_len(b4_stride as isize, w)) };
    lpf_v_sb_y_8bpc_inner(
        buf,
        0,
        stride as isize,
        mask,
        lvl,
        b4_stride as isize,
        lut,
        w,
        bitdepth_max,
    );
}

/// FFI wrapper for UV horizontal filter
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn lpf_h_sb_uv_8bpc_avx2(
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
    let buf_len = compute_buf_len_u8(stride as isize, w);
    let buf = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_len) };
    let lvl =
        unsafe { std::slice::from_raw_parts(lvl_ptr, compute_lvl_len(b4_stride as isize, w)) };
    lpf_h_sb_uv_8bpc_inner(
        buf,
        0,
        stride as isize,
        mask,
        lvl,
        b4_stride as isize,
        lut,
        w,
        bitdepth_max,
    );
}

/// FFI wrapper for UV vertical filter
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn lpf_v_sb_uv_8bpc_avx2(
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
    let buf_len = compute_buf_len_u8(stride as isize, w);
    let buf = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_len) };
    let lvl =
        unsafe { std::slice::from_raw_parts(lvl_ptr, compute_lvl_len(b4_stride as isize, w)) };
    lpf_v_sb_uv_8bpc_inner(
        buf,
        0,
        stride as isize,
        mask,
        lvl,
        b4_stride as isize,
        lut,
        w,
        bitdepth_max,
    );
}

// ============================================================================
// 16BPC IMPLEMENTATIONS
// ============================================================================

/// Core loop filter for 16bpc - processes 4 pixels
#[cfg(target_arch = "x86_64")]
fn loop_filter_4_16bpc(
    buf: &mut [u16],
    base: usize,
    e: i32,
    i: i32,
    h: i32,
    stridea: isize,
    strideb: isize,
    wd: i32,
    bitdepth_max: i32,
) {
    let bitdepth_min_8 = if bitdepth_max > 255 {
        if bitdepth_max > 1023 {
            4
        } else {
            2
        }
    } else {
        0
    };
    let f = 1i32 << bitdepth_min_8;

    for idx in 0..4isize {
        let edge = signed_idx(base, idx * stridea);

        let get_px = |offset: isize| -> i32 { buf[signed_idx(edge, strideb * offset)] as i32 };

        let p1 = get_px(-2);
        let p0 = get_px(-1);
        let q0 = get_px(0);
        let q1 = get_px(1);

        let mut fm = (p1 - p0).abs() <= i
            && (q1 - q0).abs() <= i
            && (p0 - q0).abs() * 2 + ((p1 - q1).abs() >> 1) <= e;

        let (mut p2, mut p3, mut q2, mut q3) = (0, 0, 0, 0);
        let (mut p4, mut p5, mut p6, mut q4, mut q5, mut q6) = (0, 0, 0, 0, 0, 0);

        if wd > 4 {
            p2 = get_px(-3);
            q2 = get_px(2);
            fm &= (p2 - p1).abs() <= i && (q2 - q1).abs() <= i;

            if wd > 6 {
                p3 = get_px(-4);
                q3 = get_px(3);
                fm &= (p3 - p2).abs() <= i && (q3 - q2).abs() <= i;
            }
        }

        if !fm {
            continue;
        }

        let mut flat8out = false;
        let mut flat8in = false;

        if wd >= 16 {
            p6 = get_px(-7);
            p5 = get_px(-6);
            p4 = get_px(-5);
            q4 = get_px(4);
            q5 = get_px(5);
            q6 = get_px(6);

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

        let set_px = |buf: &mut [u16], offset: isize, val: i32| {
            buf[signed_idx(edge, strideb * offset)] = val.clamp(0, bitdepth_max) as u16;
        };

        if wd >= 16 && flat8out && flat8in {
            set_px(
                buf,
                -6,
                (p6 + p6 + p6 + p6 + p6 + p6 * 2 + p5 * 2 + p4 * 2 + p3 + p2 + p1 + p0 + q0 + 8)
                    >> 4,
            );
            set_px(
                buf,
                -5,
                (p6 + p6 + p6 + p6 + p6 + p5 * 2 + p4 * 2 + p3 * 2 + p2 + p1 + p0 + q0 + q1 + 8)
                    >> 4,
            );
            set_px(
                buf,
                -4,
                (p6 + p6 + p6 + p6 + p5 + p4 * 2 + p3 * 2 + p2 * 2 + p1 + p0 + q0 + q1 + q2 + 8)
                    >> 4,
            );
            set_px(
                buf,
                -3,
                (p6 + p6 + p6 + p5 + p4 + p3 * 2 + p2 * 2 + p1 * 2 + p0 + q0 + q1 + q2 + q3 + 8)
                    >> 4,
            );
            set_px(
                buf,
                -2,
                (p6 + p6 + p5 + p4 + p3 + p2 * 2 + p1 * 2 + p0 * 2 + q0 + q1 + q2 + q3 + q4 + 8)
                    >> 4,
            );
            set_px(
                buf,
                -1,
                (p6 + p5 + p4 + p3 + p2 + p1 * 2 + p0 * 2 + q0 * 2 + q1 + q2 + q3 + q4 + q5 + 8)
                    >> 4,
            );
            set_px(
                buf,
                0,
                (p5 + p4 + p3 + p2 + p1 + p0 * 2 + q0 * 2 + q1 * 2 + q2 + q3 + q4 + q5 + q6 + 8)
                    >> 4,
            );
            set_px(
                buf,
                1,
                (p4 + p3 + p2 + p1 + p0 + q0 * 2 + q1 * 2 + q2 * 2 + q3 + q4 + q5 + q6 + q6 + 8)
                    >> 4,
            );
            set_px(
                buf,
                2,
                (p3 + p2 + p1 + p0 + q0 + q1 * 2 + q2 * 2 + q3 * 2 + q4 + q5 + q6 + q6 + q6 + 8)
                    >> 4,
            );
            set_px(
                buf,
                3,
                (p2 + p1 + p0 + q0 + q1 + q2 * 2 + q3 * 2 + q4 * 2 + q5 + q6 + q6 + q6 + q6 + 8)
                    >> 4,
            );
            set_px(
                buf,
                4,
                (p1 + p0 + q0 + q1 + q2 + q3 * 2 + q4 * 2 + q5 * 2 + q6 + q6 + q6 + q6 + q6 + 8)
                    >> 4,
            );
            set_px(
                buf,
                5,
                (p0 + q0 + q1 + q2 + q3 + q4 * 2 + q5 * 2 + q6 * 2 + q6 + q6 + q6 + q6 + q6 + 8)
                    >> 4,
            );
        } else if wd >= 8 && flat8in {
            set_px(buf, -3, (p3 + p3 + p3 + 2 * p2 + p1 + p0 + q0 + 4) >> 3);
            set_px(buf, -2, (p3 + p3 + p2 + 2 * p1 + p0 + q0 + q1 + 4) >> 3);
            set_px(buf, -1, (p3 + p2 + p1 + 2 * p0 + q0 + q1 + q2 + 4) >> 3);
            set_px(buf, 0, (p2 + p1 + p0 + 2 * q0 + q1 + q2 + q3 + 4) >> 3);
            set_px(buf, 1, (p1 + p0 + q0 + 2 * q1 + q2 + q3 + q3 + 4) >> 3);
            set_px(buf, 2, (p0 + q0 + q1 + 2 * q2 + q3 + q3 + q3 + 4) >> 3);
        } else if wd >= 6 && flat8in {
            set_px(buf, -2, (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3);
            set_px(buf, -1, (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3);
            set_px(buf, 0, (p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3);
            set_px(buf, 1, (p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3);
        } else {
            let hev = (p1 - p0).abs() > h || (q1 - q0).abs() > h;

            if hev {
                let f1 = iclip_diff((p1 - q1) + 3 * (q0 - p0), bitdepth_min_8 as u8);
                let f2 = (f1 + 4) >> 3;
                let f1 = (f1 + 3) >> 3;

                set_px(buf, -1, iclip(p0 + f2, 0, bitdepth_max));
                set_px(buf, 0, iclip(q0 - f1, 0, bitdepth_max));
            } else {
                let f1 = iclip_diff(3 * (q0 - p0), bitdepth_min_8 as u8);
                let f2 = (f1 + 4) >> 3;
                let f1 = (f1 + 3) >> 3;

                set_px(buf, -1, iclip(p0 + f2, 0, bitdepth_max));
                set_px(buf, 0, iclip(q0 - f1, 0, bitdepth_max));

                let f3 = (f1 + 1) >> 1;
                set_px(buf, -2, iclip(p1 + f3, 0, bitdepth_max));
                set_px(buf, 1, iclip(q1 - f3, 0, bitdepth_max));
            }
        }
    }
}

// ============================================================================
// SUPERBLOCK FILTER FUNCTIONS (16bpc)
// ============================================================================

/// Loop filter Y horizontal 16bpc inner
#[cfg(target_arch = "x86_64")]
fn lpf_h_sb_y_16bpc_inner(
    buf: &mut [u16],
    mut dst_offset: usize,
    stride_u16: isize,
    vmask: &[u32; 3],
    lvl: &[[u8; 4]],
    b4_stride: isize,
    lut: &Align16<Av1FilterLUT>,
    _w: i32,
    bitdepth_max: i32,
) {
    let stridea = stride_u16;
    let strideb = 1isize;
    let b4_stridea = b4_stride as usize;
    let b4_strideb = 1usize;

    let vm = vmask[0] | vmask[1] | vmask[2];
    let mut lvl_offset = 0usize;

    let mut xy = 1u32;
    while vm & !xy.wrapping_sub(1) != 0 {
        if vm & xy != 0 {
            let lvl_val = read_lvl(lvl, lvl_offset);
            let l = if lvl_val != 0 {
                lvl_val
            } else {
                if lvl_offset >= 4 * b4_strideb {
                    read_lvl(lvl, lvl_offset - 4 * b4_strideb)
                } else {
                    0
                }
            };

            if l != 0 {
                let h = (l >> 4) as i32;
                let e = lut.0.e[l as usize] as i32;
                let i = lut.0.i[l as usize] as i32;

                let idx = if vmask[2] & xy != 0 {
                    16
                } else if vmask[1] & xy != 0 {
                    8
                } else {
                    4
                };

                loop_filter_4_16bpc(
                    buf,
                    dst_offset,
                    e,
                    i,
                    h,
                    stridea,
                    strideb,
                    idx,
                    bitdepth_max,
                );
            }
        }

        xy <<= 1;
        dst_offset = signed_idx(dst_offset, 4 * stridea);
        lvl_offset += 4 * b4_stridea;
    }
}

/// Loop filter Y vertical 16bpc inner
#[cfg(target_arch = "x86_64")]
fn lpf_v_sb_y_16bpc_inner(
    buf: &mut [u16],
    mut dst_offset: usize,
    stride_u16: isize,
    vmask: &[u32; 3],
    lvl: &[[u8; 4]],
    b4_stride: isize,
    lut: &Align16<Av1FilterLUT>,
    _w: i32,
    bitdepth_max: i32,
) {
    let stridea = 1isize;
    let strideb = stride_u16;
    let b4_stridea = 1usize;
    let b4_strideb = b4_stride as usize;

    let vm = vmask[0] | vmask[1] | vmask[2];
    let mut lvl_offset = 0usize;

    let mut xy = 1u32;
    while vm & !xy.wrapping_sub(1) != 0 {
        if vm & xy != 0 {
            let lvl_val = read_lvl(lvl, lvl_offset);
            let l = if lvl_val != 0 {
                lvl_val
            } else {
                // Note: original uses b4_strideb (not 4*b4_strideb) for V direction lookback
                if lvl_offset >= b4_strideb {
                    read_lvl(lvl, lvl_offset - b4_strideb)
                } else {
                    0
                }
            };

            if l != 0 {
                let h = (l >> 4) as i32;
                let e = lut.0.e[l as usize] as i32;
                let i = lut.0.i[l as usize] as i32;

                let idx = if vmask[2] & xy != 0 {
                    16
                } else if vmask[1] & xy != 0 {
                    8
                } else {
                    4
                };

                loop_filter_4_16bpc(
                    buf,
                    dst_offset,
                    e,
                    i,
                    h,
                    stridea,
                    strideb,
                    idx,
                    bitdepth_max,
                );
            }
        }

        xy <<= 1;
        dst_offset = signed_idx(dst_offset, 4 * stridea);
        lvl_offset += 4 * b4_stridea;
    }
}

/// Loop filter UV horizontal 16bpc inner
#[cfg(target_arch = "x86_64")]
fn lpf_h_sb_uv_16bpc_inner(
    buf: &mut [u16],
    mut dst_offset: usize,
    stride_u16: isize,
    vmask: &[u32; 3],
    lvl: &[[u8; 4]],
    b4_stride: isize,
    lut: &Align16<Av1FilterLUT>,
    _w: i32,
    bitdepth_max: i32,
) {
    let stridea = stride_u16;
    let strideb = 1isize;
    let b4_stridea = b4_stride as usize;
    let b4_strideb = 1usize;

    let vm = vmask[0] | vmask[1];
    let mut lvl_offset = 0usize;

    let mut xy = 1u32;
    while vm & !xy.wrapping_sub(1) != 0 {
        if vm & xy != 0 {
            let lvl_val = read_lvl(lvl, lvl_offset);
            let l = if lvl_val != 0 {
                lvl_val
            } else {
                if lvl_offset >= 4 * b4_strideb {
                    read_lvl(lvl, lvl_offset - 4 * b4_strideb)
                } else {
                    0
                }
            };

            if l != 0 {
                let h = (l >> 4) as i32;
                let e = lut.0.e[l as usize] as i32;
                let i = lut.0.i[l as usize] as i32;

                let idx = if vmask[1] & xy != 0 { 6 } else { 4 };

                loop_filter_4_16bpc(
                    buf,
                    dst_offset,
                    e,
                    i,
                    h,
                    stridea,
                    strideb,
                    idx,
                    bitdepth_max,
                );
            }
        }

        xy <<= 1;
        dst_offset = signed_idx(dst_offset, 4 * stridea);
        lvl_offset += 4 * b4_stridea;
    }
}

/// Loop filter UV vertical 16bpc inner
#[cfg(target_arch = "x86_64")]
fn lpf_v_sb_uv_16bpc_inner(
    buf: &mut [u16],
    mut dst_offset: usize,
    stride_u16: isize,
    vmask: &[u32; 3],
    lvl: &[[u8; 4]],
    b4_stride: isize,
    lut: &Align16<Av1FilterLUT>,
    _w: i32,
    bitdepth_max: i32,
) {
    let stridea = 1isize;
    let strideb = stride_u16;
    let b4_stridea = 1usize;
    let b4_strideb = b4_stride as usize;

    let vm = vmask[0] | vmask[1];
    let mut lvl_offset = 0usize;

    let mut xy = 1u32;
    while vm & !xy.wrapping_sub(1) != 0 {
        if vm & xy != 0 {
            let lvl_val = read_lvl(lvl, lvl_offset);
            let l = if lvl_val != 0 {
                lvl_val
            } else {
                // Note: original uses b4_strideb (not 4*b4_strideb) for V direction lookback
                if lvl_offset >= b4_strideb {
                    read_lvl(lvl, lvl_offset - b4_strideb)
                } else {
                    0
                }
            };

            if l != 0 {
                let h = (l >> 4) as i32;
                let e = lut.0.e[l as usize] as i32;
                let i = lut.0.i[l as usize] as i32;

                let idx = if vmask[1] & xy != 0 { 6 } else { 4 };

                loop_filter_4_16bpc(
                    buf,
                    dst_offset,
                    e,
                    i,
                    h,
                    stridea,
                    strideb,
                    idx,
                    bitdepth_max,
                );
            }
        }

        xy <<= 1;
        dst_offset = signed_idx(dst_offset, 4 * stridea);
        lvl_offset += 4 * b4_stridea;
    }
}

// ============================================================================
// FFI WRAPPERS (16bpc) — only compiled with asm feature
// ============================================================================

/// FFI wrapper for Y horizontal filter 16bpc
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn lpf_h_sb_y_16bpc_avx2(
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
    let buf_len = compute_buf_len_u16(stride as isize, w);
    let buf = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u16, buf_len) };
    let lvl =
        unsafe { std::slice::from_raw_parts(lvl_ptr, compute_lvl_len(b4_stride as isize, w)) };
    lpf_h_sb_y_16bpc_inner(
        buf,
        0,
        stride as isize / 2,
        mask,
        lvl,
        b4_stride as isize,
        lut,
        w,
        bitdepth_max,
    );
}

/// FFI wrapper for Y vertical filter 16bpc
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn lpf_v_sb_y_16bpc_avx2(
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
    let buf_len = compute_buf_len_u16(stride as isize, w);
    let buf = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u16, buf_len) };
    let lvl =
        unsafe { std::slice::from_raw_parts(lvl_ptr, compute_lvl_len(b4_stride as isize, w)) };
    lpf_v_sb_y_16bpc_inner(
        buf,
        0,
        stride as isize / 2,
        mask,
        lvl,
        b4_stride as isize,
        lut,
        w,
        bitdepth_max,
    );
}

/// FFI wrapper for UV horizontal filter 16bpc
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn lpf_h_sb_uv_16bpc_avx2(
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
    let buf_len = compute_buf_len_u16(stride as isize, w);
    let buf = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u16, buf_len) };
    let lvl =
        unsafe { std::slice::from_raw_parts(lvl_ptr, compute_lvl_len(b4_stride as isize, w)) };
    lpf_h_sb_uv_16bpc_inner(
        buf,
        0,
        stride as isize / 2,
        mask,
        lvl,
        b4_stride as isize,
        lut,
        w,
        bitdepth_max,
    );
}

/// FFI wrapper for UV vertical filter 16bpc
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn lpf_v_sb_uv_16bpc_avx2(
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
    let buf_len = compute_buf_len_u16(stride as isize, w);
    let buf = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u16, buf_len) };
    let lvl =
        unsafe { std::slice::from_raw_parts(lvl_ptr, compute_lvl_len(b4_stride as isize, w)) };
    lpf_v_sb_uv_16bpc_inner(
        buf,
        0,
        stride as isize / 2,
        mask,
        lvl,
        b4_stride as isize,
        lut,
        w,
        bitdepth_max,
    );
}

// ============================================================================
// BUFFER SIZE HELPERS (for FFI wrappers)
// ============================================================================

/// Compute a conservative buffer length for u8 pixel buffers.
/// The filter accesses up to 7 pixels on each side of the edge,
/// and processes up to 32 4-pixel blocks along the stride direction.
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
fn compute_buf_len_u8(stride: isize, _w: i32) -> usize {
    // Up to 32 iterations * 4 * stride + 7 pixels of reach
    (stride.unsigned_abs() * 128 + 8) as usize
}

/// Compute a conservative buffer length for u16 pixel buffers.
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
fn compute_buf_len_u16(stride: isize, _w: i32) -> usize {
    // stride is in bytes for u16, so divide by 2 for element count
    let stride_u16 = stride.unsigned_abs() / 2;
    (stride_u16 * 128 + 8) as usize
}

/// Compute a conservative lvl slice length (in [u8; 4] elements).
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
fn compute_lvl_len(b4_stride: isize, _w: i32) -> usize {
    // Up to 32 iterations * 4 * b4_stride + lookback of 4 * b4_stride
    (b4_stride.unsigned_abs() as usize) * 132 + 4
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iclip_diff() {
        assert_eq!(iclip_diff(100, 0), 100);
        assert_eq!(iclip_diff(-100, 0), -100);
        assert_eq!(iclip_diff(200, 0), 127);
        assert_eq!(iclip_diff(-200, 0), -128);
    }
}

/// Safe dispatch for loopfilter_sb on x86_64. Returns true if SIMD was used.
#[cfg(target_arch = "x86_64")]
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

    let Some(_token) = Desktop64::summon() else { return false };

    assert!(lvl.offset <= lvl.data.len());

    // Safe slice access for lvl data: reinterpret u8 data as &[[u8; 4]] entries
    let lvl_remaining_bytes = lvl.data.len() - lvl.offset;
    let lvl_len = lvl_remaining_bytes / 4;
    let lvl_guard = lvl.data.slice_as::<_, [u8; 4]>((lvl.offset.., ..lvl_len));
    let lvl_slice: &[[u8; 4]] = &lvl_guard;

    match BD::BPC {
        BPC::BPC8 => {
            use crate::include::common::bitdepth::BitDepth8;

            // For 8bpc, the stride is in bytes (= pixels).
            // The filter accesses pixels at offsets from -7*stride to well beyond the base.
            let byte_stride = stride.unsigned_abs() as usize;
            let reach_before = 7 * byte_stride + 7;
            let reach_after = 128 * byte_stride + 7;

            // Safe slice access: get a mutable guard covering the full filter reach
            let start_pixel = dst.offset.wrapping_sub(reach_before);
            let total_pixels = reach_before + reach_after;
            let mut buf_guard = dst
                .data
                .slice_mut::<BitDepth8, _>((start_pixel.., ..total_pixels));
            let buf: &mut [u8] = &mut *buf_guard;
            let base = reach_before;

            match (is_y, is_v) {
                (true, false) => lpf_h_sb_y_8bpc_inner(
                    buf,
                    base,
                    stride as isize,
                    mask,
                    lvl_slice,
                    b4_stride,
                    lut,
                    w,
                    bitdepth_max,
                ),
                (true, true) => lpf_v_sb_y_8bpc_inner(
                    buf,
                    base,
                    stride as isize,
                    mask,
                    lvl_slice,
                    b4_stride,
                    lut,
                    w,
                    bitdepth_max,
                ),
                (false, false) => lpf_h_sb_uv_8bpc_inner(
                    buf,
                    base,
                    stride as isize,
                    mask,
                    lvl_slice,
                    b4_stride,
                    lut,
                    w,
                    bitdepth_max,
                ),
                (false, true) => lpf_v_sb_uv_8bpc_inner(
                    buf,
                    base,
                    stride as isize,
                    mask,
                    lvl_slice,
                    b4_stride,
                    lut,
                    w,
                    bitdepth_max,
                ),
            }
        }
        BPC::BPC16 => {
            use crate::include::common::bitdepth::BitDepth16;

            let u16_stride = (stride / 2).unsigned_abs() as usize;
            let reach_before = 7 * u16_stride + 7;
            let reach_after = 128 * u16_stride + 7;

            // Safe slice access: get a mutable guard covering the full filter reach
            let start_pixel = dst.offset.wrapping_sub(reach_before);
            let total_pixels = reach_before + reach_after;
            let mut buf_guard = dst
                .data
                .slice_mut::<BitDepth16, _>((start_pixel.., ..total_pixels));
            let buf: &mut [u16] = &mut *buf_guard;
            let base = reach_before;

            match (is_y, is_v) {
                (true, false) => lpf_h_sb_y_16bpc_inner(
                    buf,
                    base,
                    stride as isize / 2,
                    mask,
                    lvl_slice,
                    b4_stride,
                    lut,
                    w,
                    bitdepth_max,
                ),
                (true, true) => lpf_v_sb_y_16bpc_inner(
                    buf,
                    base,
                    stride as isize / 2,
                    mask,
                    lvl_slice,
                    b4_stride,
                    lut,
                    w,
                    bitdepth_max,
                ),
                (false, false) => lpf_h_sb_uv_16bpc_inner(
                    buf,
                    base,
                    stride as isize / 2,
                    mask,
                    lvl_slice,
                    b4_stride,
                    lut,
                    w,
                    bitdepth_max,
                ),
                (false, true) => lpf_v_sb_uv_16bpc_inner(
                    buf,
                    base,
                    stride as isize / 2,
                    mask,
                    lvl_slice,
                    b4_stride,
                    lut,
                    w,
                    bitdepth_max,
                ),
            }
        }
    }
    true
}
