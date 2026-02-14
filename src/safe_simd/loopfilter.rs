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

#![cfg_attr(not(feature = "unchecked"), forbid(unsafe_code))]
#![cfg_attr(feature = "unchecked", deny(unsafe_code))]
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
/// Each entry is [u8; 4]; `byte_idx` selects which byte within the entry:
///   0 = H Y, 1 = V Y, 2 = H/V U, 3 = H/V V
/// Returns 0 for out-of-bounds access (= no filtering for that block).
#[inline(always)]
fn read_lvl(lvl: &[[u8; 4]], offset: usize, byte_idx: usize) -> u8 {
    lvl.get(offset).map_or(0, |v| v[byte_idx])
}

/// Loop filter for Y plane, horizontal edges (8bpc)
#[cfg(target_arch = "x86_64")]
fn lpf_h_sb_y_8bpc_inner(
    buf: &mut [u8],
    mut dst_offset: usize,
    stride: isize,
    vmask: &[u32; 3],
    lvl: &[[u8; 4]],
    lvl_base: usize,
    lvl_byte_idx: usize,
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
    let mut lvl_offset = lvl_base;

    let mut xy = 1u32;
    while vm & !xy.wrapping_sub(1) != 0 {
        if vm & xy != 0 {
            let lvl_val = read_lvl(lvl, lvl_offset, lvl_byte_idx);
            let l = if lvl_val != 0 {
                lvl_val
            } else {
                if lvl_offset >= b4_strideb {
                    read_lvl(lvl, lvl_offset - b4_strideb, lvl_byte_idx)
                } else {
                    0
                }
            };

            if l != 0 {
                let h = (l >> 4) as i32;
                let e = lut.e[l as usize] as i32;
                let i = lut.i[l as usize] as i32;

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
        lvl_offset += b4_stridea;
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
    lvl_base: usize,
    lvl_byte_idx: usize,
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
    let mut lvl_offset = lvl_base;

    let mut xy = 1u32;
    while vm & !xy.wrapping_sub(1) != 0 {
        if vm & xy != 0 {
            let lvl_val = read_lvl(lvl, lvl_offset, lvl_byte_idx);
            let l = if lvl_val != 0 {
                lvl_val
            } else {
                if lvl_offset >= b4_strideb {
                    read_lvl(lvl, lvl_offset - b4_strideb, lvl_byte_idx)
                } else {
                    0
                }
            };

            if l != 0 {
                let h = (l >> 4) as i32;
                let e = lut.e[l as usize] as i32;
                let i = lut.i[l as usize] as i32;

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
        lvl_offset += b4_stridea;
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
    lvl_base: usize,
    lvl_byte_idx: usize,
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
    let mut lvl_offset = lvl_base;

    let mut xy = 1u32;
    while vm & !xy.wrapping_sub(1) != 0 {
        if vm & xy != 0 {
            let lvl_val = read_lvl(lvl, lvl_offset, lvl_byte_idx);
            let l = if lvl_val != 0 {
                lvl_val
            } else {
                if lvl_offset >= b4_strideb {
                    read_lvl(lvl, lvl_offset - b4_strideb, lvl_byte_idx)
                } else {
                    0
                }
            };

            if l != 0 {
                let h = (l >> 4) as i32;
                let e = lut.e[l as usize] as i32;
                let i = lut.i[l as usize] as i32;

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
        lvl_offset += b4_stridea;
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
    lvl_base: usize,
    lvl_byte_idx: usize,
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
    let mut lvl_offset = lvl_base;

    let mut xy = 1u32;
    while vm & !xy.wrapping_sub(1) != 0 {
        if vm & xy != 0 {
            let lvl_val = read_lvl(lvl, lvl_offset, lvl_byte_idx);
            let l = if lvl_val != 0 {
                lvl_val
            } else {
                if lvl_offset >= b4_strideb {
                    read_lvl(lvl, lvl_offset - b4_strideb, lvl_byte_idx)
                } else {
                    0
                }
            };

            if l != 0 {
                let h = (l >> 4) as i32;
                let e = lut.e[l as usize] as i32;
                let i = lut.i[l as usize] as i32;

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
        lvl_offset += b4_stridea;
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
        0,
        0,
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
        0,
        0,
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
        0,
        0,
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
        0,
        0,
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
    let e = e << bitdepth_min_8;
    let i = i << bitdepth_min_8;
    let h = h << bitdepth_min_8;

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
            set_px(buf, -2, (p2 + 2 * p2 + 2 * p1 + 2 * p0 + q0 + 4) >> 3);
            set_px(buf, -1, (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3);
            set_px(buf, 0, (p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3);
            set_px(buf, 1, (p0 + 2 * q0 + 2 * q1 + 2 * q2 + q2 + 4) >> 3);
        } else {
            let hev = (p1 - p0).abs() > h || (q1 - q0).abs() > h;

            let bdm8 = bitdepth_min_8 as u8;
            if hev {
                let f = iclip_diff(p1 - q1, bdm8);
                let f = iclip_diff(3 * (q0 - p0) + f, bdm8);

                let f1 = cmp::min(f + 4, (128 << bdm8) - 1) >> 3;
                let f2 = cmp::min(f + 3, (128 << bdm8) - 1) >> 3;

                set_px(buf, -1, iclip(p0 + f2, 0, bitdepth_max));
                set_px(buf, 0, iclip(q0 - f1, 0, bitdepth_max));
            } else {
                let f = iclip_diff(3 * (q0 - p0), bdm8);

                let f1 = cmp::min(f + 4, (128 << bdm8) - 1) >> 3;
                let f2 = cmp::min(f + 3, (128 << bdm8) - 1) >> 3;

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
    lvl_base: usize,
    lvl_byte_idx: usize,
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
    let mut lvl_offset = lvl_base;

    let mut xy = 1u32;
    while vm & !xy.wrapping_sub(1) != 0 {
        if vm & xy != 0 {
            let lvl_val = read_lvl(lvl, lvl_offset, lvl_byte_idx);
            let l = if lvl_val != 0 {
                lvl_val
            } else {
                if lvl_offset >= b4_strideb {
                    read_lvl(lvl, lvl_offset - b4_strideb, lvl_byte_idx)
                } else {
                    0
                }
            };

            if l != 0 {
                let h = (l >> 4) as i32;
                let e = lut.e[l as usize] as i32;
                let i = lut.i[l as usize] as i32;

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
        lvl_offset += b4_stridea;
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
    lvl_base: usize,
    lvl_byte_idx: usize,
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
    let mut lvl_offset = lvl_base;

    let mut xy = 1u32;
    while vm & !xy.wrapping_sub(1) != 0 {
        if vm & xy != 0 {
            let lvl_val = read_lvl(lvl, lvl_offset, lvl_byte_idx);
            let l = if lvl_val != 0 {
                lvl_val
            } else {
                // Note: original uses b4_strideb (not 4*b4_strideb) for V direction lookback
                if lvl_offset >= b4_strideb {
                    read_lvl(lvl, lvl_offset - b4_strideb, lvl_byte_idx)
                } else {
                    0
                }
            };

            if l != 0 {
                let h = (l >> 4) as i32;
                let e = lut.e[l as usize] as i32;
                let i = lut.i[l as usize] as i32;

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
        lvl_offset += b4_stridea;
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
    lvl_base: usize,
    lvl_byte_idx: usize,
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
    let mut lvl_offset = lvl_base;

    let mut xy = 1u32;
    while vm & !xy.wrapping_sub(1) != 0 {
        if vm & xy != 0 {
            let lvl_val = read_lvl(lvl, lvl_offset, lvl_byte_idx);
            let l = if lvl_val != 0 {
                lvl_val
            } else {
                if lvl_offset >= b4_strideb {
                    read_lvl(lvl, lvl_offset - b4_strideb, lvl_byte_idx)
                } else {
                    0
                }
            };

            if l != 0 {
                let h = (l >> 4) as i32;
                let e = lut.e[l as usize] as i32;
                let i = lut.i[l as usize] as i32;

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
        lvl_offset += b4_stridea;
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
    lvl_base: usize,
    lvl_byte_idx: usize,
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
    let mut lvl_offset = lvl_base;

    let mut xy = 1u32;
    while vm & !xy.wrapping_sub(1) != 0 {
        if vm & xy != 0 {
            let lvl_val = read_lvl(lvl, lvl_offset, lvl_byte_idx);
            let l = if lvl_val != 0 {
                lvl_val
            } else {
                // Note: original uses b4_strideb (not 4*b4_strideb) for V direction lookback
                if lvl_offset >= b4_strideb {
                    read_lvl(lvl, lvl_offset - b4_strideb, lvl_byte_idx)
                } else {
                    0
                }
            };

            if l != 0 {
                let h = (l >> 4) as i32;
                let e = lut.e[l as usize] as i32;
                let i = lut.i[l as usize] as i32;

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
        lvl_offset += b4_stridea;
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
        0,
        0,
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
        0,
        0,
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
        0,
        0,
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
        0,
        0,
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
    // Up to 32 iterations * b4_stride + lookback of b4_stride (conservative)
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

    let Some(_token) = crate::src::cpu::summon_avx2() else {
        return false;
    };

    assert!(lvl.offset <= lvl.data.len());

    // Safe slice access for lvl data: reinterpret u8 data as &[[u8; 4]] entries
    // Note: lvl.offset is a BYTE offset (not element offset), so we use .index()
    // with a byte range and then cast via zerocopy, since slice_as() would
    // incorrectly multiply the offset by sizeof([u8; 4]).
    //
    // Include lookback entries: when the current block's level is 0, the scalar
    // fallback reads the PREVIOUS block's level (lvl - 4*b4_strideb bytes).
    // We need to include those entries in the slice so the SIMD path can too.
    let b4_strideb_entries = if !is_v {
        1usize
    } else {
        b4_stride.unsigned_abs() as usize
    };
    let lvl_lookback_bytes = b4_strideb_entries * 4;
    let lvl_start = lvl.offset.saturating_sub(lvl_lookback_bytes);
    // Round down to a 4-byte boundary for zerocopy alignment
    let lvl_start = lvl_start & !3;
    let lvl_remaining_bytes = lvl.data.len() - lvl_start;
    let lvl_len = lvl_remaining_bytes / 4;
    let lvl_byte_end = lvl_start + lvl_len * 4;
    let lvl_byte_guard = lvl.data.index(lvl_start..lvl_byte_end);
    let lvl_slice: &[[u8; 4]] = zerocopy::FromBytes::ref_from_bytes(&*lvl_byte_guard).unwrap();
    // Which byte within each [u8;4] entry to read:
    //   H Y → 0, V Y → 1, H U → 2, H V → 3
    // This is encoded in lvl.offset % 4 by the caller (lf_apply.rs adds +0/+1/+2/+3).
    let lvl_byte_idx = lvl.offset % 4;
    // Base offset: how many [u8;4] entries from the start of lvl_slice to the
    // original lvl.offset's 4-byte-aligned position
    let lvl_base = (lvl.offset - lvl_byte_idx - lvl_start) / 4;

    // Compute actual iterations from vmask to tighten bounds check.
    // Each bit in vmask represents a 4-pixel edge. The highest set bit
    // determines the maximum forward reach, which is much less than 128
    // for SBs near image edges.
    let vm = mask[0] | mask[1] | mask[2];
    if vm == 0 {
        return true; // Nothing to filter
    }
    let max_iter = 32 - vm.leading_zeros() as usize;

    match BD::BPC {
        BPC::BPC8 => {
            use crate::include::common::bitdepth::BitDepth8;

            // For 8bpc, the stride is in bytes (= pixels).
            let byte_stride = stride.unsigned_abs() as usize;

            // Compute reach based on filter direction and actual vmask extent.
            // H filter (is_v=false): iterates rows (stridea=stride), pixel access (strideb=1)
            //   forward: last group at (max_iter-1)*4*stride, +3 lines, +16 pixels
            //   backward: 7 pixels horizontally
            // V filter (is_v=true): iterates columns (stridea=1), row access (strideb=stride)
            //   forward: (max_iter*4-1) columns + 16*stride rows
            //   backward: 7*stride rows
            let (reach_before, reach_after) = if !is_v {
                // H filter: iterates through row groups
                (7, (max_iter * 4 - 1) * byte_stride + 16)
            } else {
                // V filter: iterates through column groups
                (7 * byte_stride, max_iter * 4 - 1 + 16 * byte_stride)
            };

            // Guard: fall back to scalar if buffer bounds are insufficient.
            let buf_pixel_len = dst.data.pixel_len::<BitDepth8>();
            if dst.offset < reach_before || dst.offset.saturating_add(reach_after) > buf_pixel_len {
                return false;
            }

            // Safe slice access: get a mutable guard covering the full filter reach
            let start_pixel = dst.offset - reach_before;
            let total_pixels = (reach_before + reach_after).min(buf_pixel_len - start_pixel);
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
                    lvl_base,
                    lvl_byte_idx,
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
                    lvl_base,
                    lvl_byte_idx,
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
                    lvl_base,
                    lvl_byte_idx,
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
                    lvl_base,
                    lvl_byte_idx,
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

            // Compute reach based on filter direction and actual vmask extent
            let (reach_before, reach_after) = if !is_v {
                // H filter: iterates through row groups
                (7, (max_iter * 4 - 1) * u16_stride + 16)
            } else {
                // V filter: iterates through column groups
                (7 * u16_stride, max_iter * 4 - 1 + 16 * u16_stride)
            };

            // Guard: fall back to scalar if buffer bounds are insufficient.
            let buf_pixel_len = dst.data.pixel_len::<BitDepth16>();
            if dst.offset < reach_before || dst.offset.saturating_add(reach_after) > buf_pixel_len {
                return false;
            }

            // Safe slice access: get a mutable guard covering the full filter reach
            let start_pixel = dst.offset - reach_before;
            let total_pixels = (reach_before + reach_after).min(buf_pixel_len - start_pixel);
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
                    lvl_base,
                    lvl_byte_idx,
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
                    lvl_base,
                    lvl_byte_idx,
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
                    lvl_base,
                    lvl_byte_idx,
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
                    lvl_base,
                    lvl_byte_idx,
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
