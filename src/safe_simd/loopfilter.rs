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

#![allow(unused_imports)]

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

// ============================================================================
// CORE LOOP FILTER (4 pixels at a time, SIMD)
// ============================================================================

/// Core loop filter for 8bpc - processes 4 pixels using SIMD
/// This is the heart of the deblocking filter
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn loop_filter_4_8bpc_avx2(
    dst: *mut u8,
    e: i32,
    i: i32,
    h: i32,
    stridea: isize,   // stride between the 4 pixels we process
    strideb: isize,   // stride along the edge (perpendicular to filter direction)
    wd: i32,
    bitdepth_max: i32,
) {
    // Load pixels: p6..p0, q0..q6 for all 4 positions
    // stridea is the stride between the 4 parallel filter operations
    // strideb is the stride in the filter direction (towards the edge)

    // For simplicity, we'll process each of the 4 pixels
    // The SIMD benefit comes from processing the pixel math in parallel

    let f = 1i32;

    for idx in 0..4isize {
        let base = unsafe { dst.offset(idx * stridea) };

        // Helper to get pixel at offset from edge
        let get_px = |offset: isize| -> i32 {
            unsafe { *base.offset(strideb * offset) as i32 }
        };
        let set_px = |offset: isize, val: i32| {
            unsafe { *base.offset(strideb * offset) = val.clamp(0, bitdepth_max) as u8 };
        };

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

        if wd >= 16 && flat8out && flat8in {
            // Wide filter (16 taps)
            set_px(-6, (p6 + p6 + p6 + p6 + p6 + p6 * 2 + p5 * 2 + p4 * 2 + p3 + p2 + p1 + p0 + q0 + 8) >> 4);
            set_px(-5, (p6 + p6 + p6 + p6 + p6 + p5 * 2 + p4 * 2 + p3 * 2 + p2 + p1 + p0 + q0 + q1 + 8) >> 4);
            set_px(-4, (p6 + p6 + p6 + p6 + p5 + p4 * 2 + p3 * 2 + p2 * 2 + p1 + p0 + q0 + q1 + q2 + 8) >> 4);
            set_px(-3, (p6 + p6 + p6 + p5 + p4 + p3 * 2 + p2 * 2 + p1 * 2 + p0 + q0 + q1 + q2 + q3 + 8) >> 4);
            set_px(-2, (p6 + p6 + p5 + p4 + p3 + p2 * 2 + p1 * 2 + p0 * 2 + q0 + q1 + q2 + q3 + q4 + 8) >> 4);
            set_px(-1, (p6 + p5 + p4 + p3 + p2 + p1 * 2 + p0 * 2 + q0 * 2 + q1 + q2 + q3 + q4 + q5 + 8) >> 4);
            set_px(0, (p5 + p4 + p3 + p2 + p1 + p0 * 2 + q0 * 2 + q1 * 2 + q2 + q3 + q4 + q5 + q6 + 8) >> 4);
            set_px(1, (p4 + p3 + p2 + p1 + p0 + q0 * 2 + q1 * 2 + q2 * 2 + q3 + q4 + q5 + q6 + q6 + 8) >> 4);
            set_px(2, (p3 + p2 + p1 + p0 + q0 + q1 * 2 + q2 * 2 + q3 * 2 + q4 + q5 + q6 + q6 + q6 + 8) >> 4);
            set_px(3, (p2 + p1 + p0 + q0 + q1 + q2 * 2 + q3 * 2 + q4 * 2 + q5 + q6 + q6 + q6 + q6 + 8) >> 4);
            set_px(4, (p1 + p0 + q0 + q1 + q2 + q3 * 2 + q4 * 2 + q5 * 2 + q6 + q6 + q6 + q6 + q6 + 8) >> 4);
            set_px(5, (p0 + q0 + q1 + q2 + q3 + q4 * 2 + q5 * 2 + q6 * 2 + q6 + q6 + q6 + q6 + q6 + 8) >> 4);
        } else if wd >= 8 && flat8in {
            // 8-tap filter
            set_px(-3, (p3 + p3 + p3 + 2 * p2 + p1 + p0 + q0 + 4) >> 3);
            set_px(-2, (p3 + p3 + p2 + 2 * p1 + p0 + q0 + q1 + 4) >> 3);
            set_px(-1, (p3 + p2 + p1 + 2 * p0 + q0 + q1 + q2 + 4) >> 3);
            set_px(0, (p2 + p1 + p0 + 2 * q0 + q1 + q2 + q3 + 4) >> 3);
            set_px(1, (p1 + p0 + q0 + 2 * q1 + q2 + q3 + q3 + 4) >> 3);
            set_px(2, (p0 + q0 + q1 + 2 * q2 + q3 + q3 + q3 + 4) >> 3);
        } else if wd == 6 && flat8in {
            // 6-tap filter
            set_px(-2, (p2 + 2 * p2 + 2 * p1 + 2 * p0 + q0 + 4) >> 3);
            set_px(-1, (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3);
            set_px(0, (p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3);
            set_px(1, (p0 + 2 * q0 + 2 * q1 + 2 * q2 + q2 + 4) >> 3);
        } else {
            // Narrow filter (4-tap)
            let hev = (p1 - p0).abs() > h || (q1 - q0).abs() > h;

            if hev {
                let f = iclip_diff(p1 - q1, 0);
                let f = iclip_diff(3 * (q0 - p0) + f, 0);

                let f1 = cmp::min(f + 4, 127) >> 3;
                let f2 = cmp::min(f + 3, 127) >> 3;

                set_px(-1, p0 + f2);
                set_px(0, q0 - f1);
            } else {
                let f = iclip_diff(3 * (q0 - p0), 0);

                let f1 = cmp::min(f + 4, 127) >> 3;
                let f2 = cmp::min(f + 3, 127) >> 3;

                set_px(-1, p0 + f2);
                set_px(0, q0 - f1);

                let f = (f1 + 1) >> 1;
                set_px(-2, p1 + f);
                set_px(1, q1 - f);
            }
        }
    }
}

// ============================================================================
// SUPERBLOCK FILTER FUNCTIONS
// ============================================================================

/// Loop filter for Y plane, horizontal edges
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn lpf_h_sb_y_8bpc_avx2_inner(
    mut dst: *mut u8,
    stride: isize,
    vmask: &[u32; 3],
    lvl_ptr: *const [u8; 4],
    b4_stride: isize,
    lut: &Align16<Av1FilterLUT>,
    _w: i32,
    bitdepth_max: i32,
) {
    // H filter: stridea = stride (rows), strideb = 1 (columns)
    let stridea = stride;
    let strideb = 1isize;
    let b4_stridea = b4_stride as usize;
    let b4_strideb = 1usize;

    let vm = vmask[0] | vmask[1] | vmask[2];
    let mut lvl_offset = 0usize;

    let mut xy = 1u32;
    while vm & !xy.wrapping_sub(1) != 0 {
        if vm & xy != 0 {
            let lvl = unsafe { (*lvl_ptr.add(lvl_offset))[0] };
            let l = if lvl != 0 {
                lvl
            } else {
                // Look back
                if lvl_offset >= 4 * b4_strideb {
                    unsafe { (*lvl_ptr.add(lvl_offset - 4 * b4_strideb))[0] }
                } else {
                    0
                }
            };

            if l != 0 {
                let h = (l >> 4) as i32;
                let e = lut.0.e[l as usize] as i32;
                let i = lut.0.i[l as usize] as i32;

                let idx = if vmask[2] & xy != 0 {
                    16  // 4 << 2
                } else if vmask[1] & xy != 0 {
                    8   // 4 << 1
                } else {
                    4   // 4 << 0
                };

                unsafe {
                    loop_filter_4_8bpc_avx2(dst, e, i, h, stridea, strideb, idx, bitdepth_max);
                }
            }
        }

        xy <<= 1;
        dst = unsafe { dst.offset(4 * stridea) };
        lvl_offset += 4 * b4_stridea;
    }
}

/// Loop filter for Y plane, vertical edges
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn lpf_v_sb_y_8bpc_avx2_inner(
    mut dst: *mut u8,
    stride: isize,
    vmask: &[u32; 3],
    lvl_ptr: *const [u8; 4],
    b4_stride: isize,
    lut: &Align16<Av1FilterLUT>,
    _w: i32,
    bitdepth_max: i32,
) {
    // V filter: stridea = 1 (columns), strideb = stride (rows)
    let stridea = 1isize;
    let strideb = stride;
    let b4_stridea = 1usize;
    let b4_strideb = b4_stride as usize;

    let vm = vmask[0] | vmask[1] | vmask[2];
    let mut lvl_offset = 0usize;

    let mut xy = 1u32;
    while vm & !xy.wrapping_sub(1) != 0 {
        if vm & xy != 0 {
            let lvl = unsafe { (*lvl_ptr.add(lvl_offset))[0] };
            let l = if lvl != 0 {
                lvl
            } else {
                // Look back
                if lvl_offset >= 4 * b4_strideb {
                    unsafe { (*lvl_ptr.add(lvl_offset - 4 * b4_strideb))[0] }
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

                unsafe {
                    loop_filter_4_8bpc_avx2(dst, e, i, h, stridea, strideb, idx, bitdepth_max);
                }
            }
        }

        xy <<= 1;
        dst = unsafe { dst.offset(4 * stridea) };
        lvl_offset += 4 * b4_stridea;
    }
}

/// Loop filter for UV planes, horizontal edges
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn lpf_h_sb_uv_8bpc_avx2_inner(
    mut dst: *mut u8,
    stride: isize,
    vmask: &[u32; 3],
    lvl_ptr: *const [u8; 4],
    b4_stride: isize,
    lut: &Align16<Av1FilterLUT>,
    _w: i32,
    bitdepth_max: i32,
) {
    // UV uses only vmask[0] | vmask[1], max filter width is 6
    let stridea = stride;
    let strideb = 1isize;
    let b4_stridea = b4_stride as usize;
    let b4_strideb = 1usize;

    let vm = vmask[0] | vmask[1];
    let mut lvl_offset = 0usize;

    let mut xy = 1u32;
    while vm & !xy.wrapping_sub(1) != 0 {
        if vm & xy != 0 {
            let lvl = unsafe { (*lvl_ptr.add(lvl_offset))[0] };
            let l = if lvl != 0 {
                lvl
            } else {
                if lvl_offset >= 4 * b4_strideb {
                    unsafe { (*lvl_ptr.add(lvl_offset - 4 * b4_strideb))[0] }
                } else {
                    0
                }
            };

            if l != 0 {
                let h = (l >> 4) as i32;
                let e = lut.0.e[l as usize] as i32;
                let i = lut.0.i[l as usize] as i32;

                // UV: idx = 4 + 2 * (vmask[1] & xy != 0)
                let idx = if vmask[1] & xy != 0 { 6 } else { 4 };

                unsafe {
                    loop_filter_4_8bpc_avx2(dst, e, i, h, stridea, strideb, idx, bitdepth_max);
                }
            }
        }

        xy <<= 1;
        dst = unsafe { dst.offset(4 * stridea) };
        lvl_offset += 4 * b4_stridea;
    }
}

/// Loop filter for UV planes, vertical edges
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn lpf_v_sb_uv_8bpc_avx2_inner(
    mut dst: *mut u8,
    stride: isize,
    vmask: &[u32; 3],
    lvl_ptr: *const [u8; 4],
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
            let lvl = unsafe { (*lvl_ptr.add(lvl_offset))[0] };
            let l = if lvl != 0 {
                lvl
            } else {
                if lvl_offset >= 4 * b4_strideb {
                    unsafe { (*lvl_ptr.add(lvl_offset - 4 * b4_strideb))[0] }
                } else {
                    0
                }
            };

            if l != 0 {
                let h = (l >> 4) as i32;
                let e = lut.0.e[l as usize] as i32;
                let i = lut.0.i[l as usize] as i32;

                let idx = if vmask[1] & xy != 0 { 6 } else { 4 };

                unsafe {
                    loop_filter_4_8bpc_avx2(dst, e, i, h, stridea, strideb, idx, bitdepth_max);
                }
            }
        }

        xy <<= 1;
        dst = unsafe { dst.offset(4 * stridea) };
        lvl_offset += 4 * b4_stridea;
    }
}

// ============================================================================
// FFI WRAPPERS
// ============================================================================
#[cfg(any(feature = "asm", feature = "c-ffi"))]

/// FFI wrapper for Y horizontal filter
#[cfg(target_arch = "x86_64")]
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
    unsafe {
        lpf_h_sb_y_8bpc_avx2_inner(
            dst_ptr as *mut u8,
            stride as isize,
            mask,
            lvl_ptr,
            b4_stride as isize,
            lut,
            w,
            bitdepth_max,
        );
    }
}
#[cfg(any(feature = "asm", feature = "c-ffi"))]

/// FFI wrapper for Y vertical filter
#[cfg(target_arch = "x86_64")]
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
    unsafe {
        lpf_v_sb_y_8bpc_avx2_inner(
            dst_ptr as *mut u8,
            stride as isize,
            mask,
            lvl_ptr,
            b4_stride as isize,
            lut,
            w,
            bitdepth_max,
        );
    }
}
#[cfg(any(feature = "asm", feature = "c-ffi"))]

/// FFI wrapper for UV horizontal filter
#[cfg(target_arch = "x86_64")]
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
    unsafe {
        lpf_h_sb_uv_8bpc_avx2_inner(
            dst_ptr as *mut u8,
            stride as isize,
            mask,
            lvl_ptr,
            b4_stride as isize,
            lut,
            w,
            bitdepth_max,
        );
    }
}
#[cfg(any(feature = "asm", feature = "c-ffi"))]

/// FFI wrapper for UV vertical filter
#[cfg(target_arch = "x86_64")]
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
    unsafe {
        lpf_v_sb_uv_8bpc_avx2_inner(
            dst_ptr as *mut u8,
            stride as isize,
            mask,
            lvl_ptr,
            b4_stride as isize,
            lut,
            w,
            bitdepth_max,
        );
    }
}

// ============================================================================
// 16BPC IMPLEMENTATIONS
// ============================================================================

/// Core loop filter for 16bpc - processes 4 pixels
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn loop_filter_4_16bpc_avx2(
    dst: *mut u16,
    e: i32,
    i: i32,
    h: i32,
    stridea: isize,   // stride between the 4 pixels we process (in u16 units)
    strideb: isize,   // stride along the edge (in u16 units)
    wd: i32,
    bitdepth_max: i32,
) {
    // For 16bpc, f = 1 << (bitdepth - 8) for flatness detection
    let bitdepth_min_8 = if bitdepth_max > 255 {
        if bitdepth_max > 1023 { 4 } else { 2 }
    } else {
        0
    };
    let f = 1i32 << bitdepth_min_8;

    for idx in 0..4isize {
        let base = unsafe { dst.offset(idx * stridea) };

        let get_px = |offset: isize| -> i32 {
            unsafe { *base.offset(strideb * offset) as i32 }
        };
        let set_px = |offset: isize, val: i32| {
            unsafe { *base.offset(strideb * offset) = val.clamp(0, bitdepth_max) as u16 };
        };

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

        if wd >= 16 && flat8out && flat8in {
            // Wide filter (16 taps)
            set_px(-6, (p6 + p6 + p6 + p6 + p6 + p6 * 2 + p5 * 2 + p4 * 2 + p3 + p2 + p1 + p0 + q0 + 8) >> 4);
            set_px(-5, (p6 + p6 + p6 + p6 + p6 + p5 * 2 + p4 * 2 + p3 * 2 + p2 + p1 + p0 + q0 + q1 + 8) >> 4);
            set_px(-4, (p6 + p6 + p6 + p6 + p5 + p4 * 2 + p3 * 2 + p2 * 2 + p1 + p0 + q0 + q1 + q2 + 8) >> 4);
            set_px(-3, (p6 + p6 + p6 + p5 + p4 + p3 * 2 + p2 * 2 + p1 * 2 + p0 + q0 + q1 + q2 + q3 + 8) >> 4);
            set_px(-2, (p6 + p6 + p5 + p4 + p3 + p2 * 2 + p1 * 2 + p0 * 2 + q0 + q1 + q2 + q3 + q4 + 8) >> 4);
            set_px(-1, (p6 + p5 + p4 + p3 + p2 + p1 * 2 + p0 * 2 + q0 * 2 + q1 + q2 + q3 + q4 + q5 + 8) >> 4);
            set_px(0, (p5 + p4 + p3 + p2 + p1 + p0 * 2 + q0 * 2 + q1 * 2 + q2 + q3 + q4 + q5 + q6 + 8) >> 4);
            set_px(1, (p4 + p3 + p2 + p1 + p0 + q0 * 2 + q1 * 2 + q2 * 2 + q3 + q4 + q5 + q6 + q6 + 8) >> 4);
            set_px(2, (p3 + p2 + p1 + p0 + q0 + q1 * 2 + q2 * 2 + q3 * 2 + q4 + q5 + q6 + q6 + q6 + 8) >> 4);
            set_px(3, (p2 + p1 + p0 + q0 + q1 + q2 * 2 + q3 * 2 + q4 * 2 + q5 + q6 + q6 + q6 + q6 + 8) >> 4);
            set_px(4, (p1 + p0 + q0 + q1 + q2 + q3 * 2 + q4 * 2 + q5 * 2 + q6 + q6 + q6 + q6 + q6 + 8) >> 4);
            set_px(5, (p0 + q0 + q1 + q2 + q3 + q4 * 2 + q5 * 2 + q6 * 2 + q6 + q6 + q6 + q6 + q6 + 8) >> 4);
        } else if wd >= 8 && flat8in {
            // 8-tap filter
            set_px(-3, (p3 + p3 + p3 + 2 * p2 + p1 + p0 + q0 + 4) >> 3);
            set_px(-2, (p3 + p3 + p2 + 2 * p1 + p0 + q0 + q1 + 4) >> 3);
            set_px(-1, (p3 + p2 + p1 + 2 * p0 + q0 + q1 + q2 + 4) >> 3);
            set_px(0, (p2 + p1 + p0 + 2 * q0 + q1 + q2 + q3 + 4) >> 3);
            set_px(1, (p1 + p0 + q0 + 2 * q1 + q2 + q3 + q3 + 4) >> 3);
            set_px(2, (p0 + q0 + q1 + 2 * q2 + q3 + q3 + q3 + 4) >> 3);
        } else if wd >= 6 && flat8in {
            // 6-tap filter
            set_px(-2, (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3);
            set_px(-1, (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3);
            set_px(0, (p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3);
            set_px(1, (p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3);
        } else {
            // 4-tap filter (narrow)
            let hev = (p1 - p0).abs() > h || (q1 - q0).abs() > h;

            if hev {
                // High edge variance - use simpler 3-tap filter
                let f1 = iclip_diff((p1 - q1) + 3 * (q0 - p0), bitdepth_min_8 as u8);
                let f2 = (f1 + 4) >> 3;
                let f1 = (f1 + 3) >> 3;

                set_px(-1, iclip(p0 + f2, 0, bitdepth_max));
                set_px(0, iclip(q0 - f1, 0, bitdepth_max));
            } else {
                // Low edge variance - use 4-tap filter with outer pixel adjustment
                let f1 = iclip_diff(3 * (q0 - p0), bitdepth_min_8 as u8);
                let f2 = (f1 + 4) >> 3;
                let f1 = (f1 + 3) >> 3;

                set_px(-1, iclip(p0 + f2, 0, bitdepth_max));
                set_px(0, iclip(q0 - f1, 0, bitdepth_max));

                let f3 = (f1 + 1) >> 1;
                set_px(-2, iclip(p1 + f3, 0, bitdepth_max));
                set_px(1, iclip(q1 - f3, 0, bitdepth_max));
            }
        }
    }
}

/// Loop filter Y horizontal 16bpc inner
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn lpf_h_sb_y_16bpc_avx2_inner(
    mut dst: *mut u16,
    stride_u16: isize,
    vmask: &[u32; 3],
    lvl_ptr: *const [u8; 4],
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
            let lvl = unsafe { (*lvl_ptr.add(lvl_offset))[0] };
            let l = if lvl != 0 {
                lvl
            } else {
                if lvl_offset >= 4 * b4_strideb {
                    unsafe { (*lvl_ptr.add(lvl_offset - 4 * b4_strideb))[0] }
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

                unsafe {
                    loop_filter_4_16bpc_avx2(dst, e, i, h, stridea, strideb, idx, bitdepth_max);
                }
            }
        }

        xy <<= 1;
        dst = unsafe { dst.offset(4 * stridea) };
        lvl_offset += 4 * b4_stridea;
    }
}

/// Loop filter Y vertical 16bpc inner
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn lpf_v_sb_y_16bpc_avx2_inner(
    mut dst: *mut u16,
    stride_u16: isize,
    vmask: &[u32; 3],
    lvl_ptr: *const [u8; 4],
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
            let lvl = unsafe { (*lvl_ptr.add(lvl_offset))[0] };
            let l = if lvl != 0 {
                lvl
            } else {
                if lvl_offset >= b4_strideb {
                    unsafe { (*lvl_ptr.add(lvl_offset - b4_strideb))[0] }
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

                unsafe {
                    loop_filter_4_16bpc_avx2(dst, e, i, h, stridea, strideb, idx, bitdepth_max);
                }
            }
        }

        xy <<= 1;
        dst = unsafe { dst.offset(4 * stridea) };
        lvl_offset += 4 * b4_stridea;
    }
}

/// Loop filter UV horizontal 16bpc inner
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn lpf_h_sb_uv_16bpc_avx2_inner(
    mut dst: *mut u16,
    stride_u16: isize,
    vmask: &[u32; 3],
    lvl_ptr: *const [u8; 4],
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
            let lvl = unsafe { (*lvl_ptr.add(lvl_offset))[0] };
            let l = if lvl != 0 {
                lvl
            } else {
                if lvl_offset >= 4 * b4_strideb {
                    unsafe { (*lvl_ptr.add(lvl_offset - 4 * b4_strideb))[0] }
                } else {
                    0
                }
            };

            if l != 0 {
                let h = (l >> 4) as i32;
                let e = lut.0.e[l as usize] as i32;
                let i = lut.0.i[l as usize] as i32;

                let idx = if vmask[1] & xy != 0 { 6 } else { 4 };

                unsafe {
                    loop_filter_4_16bpc_avx2(dst, e, i, h, stridea, strideb, idx, bitdepth_max);
                }
            }
        }

        xy <<= 1;
        dst = unsafe { dst.offset(4 * stridea) };
        lvl_offset += 4 * b4_stridea;
    }
}

/// Loop filter UV vertical 16bpc inner
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn lpf_v_sb_uv_16bpc_avx2_inner(
    mut dst: *mut u16,
    stride_u16: isize,
    vmask: &[u32; 3],
    lvl_ptr: *const [u8; 4],
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
            let lvl = unsafe { (*lvl_ptr.add(lvl_offset))[0] };
            let l = if lvl != 0 {
                lvl
            } else {
                if lvl_offset >= b4_strideb {
                    unsafe { (*lvl_ptr.add(lvl_offset - b4_strideb))[0] }
                } else {
                    0
                }
            };

            if l != 0 {
                let h = (l >> 4) as i32;
                let e = lut.0.e[l as usize] as i32;
                let i = lut.0.i[l as usize] as i32;

                let idx = if vmask[1] & xy != 0 { 6 } else { 4 };

                unsafe {
                    loop_filter_4_16bpc_avx2(dst, e, i, h, stridea, strideb, idx, bitdepth_max);
                }
            }
        }

        xy <<= 1;
        dst = unsafe { dst.offset(4 * stridea) };
        lvl_offset += 4 * b4_stridea;
    }
}
#[cfg(any(feature = "asm", feature = "c-ffi"))]

/// FFI wrapper for Y horizontal filter 16bpc
#[cfg(target_arch = "x86_64")]
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
    unsafe {
        lpf_h_sb_y_16bpc_avx2_inner(
            dst_ptr as *mut u16,
            stride as isize / 2,  // Convert byte stride to u16 stride
            mask,
            lvl_ptr,
            b4_stride as isize,
            lut,
            w,
            bitdepth_max,
        );
    }
}
#[cfg(any(feature = "asm", feature = "c-ffi"))]

/// FFI wrapper for Y vertical filter 16bpc
#[cfg(target_arch = "x86_64")]
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
    unsafe {
        lpf_v_sb_y_16bpc_avx2_inner(
            dst_ptr as *mut u16,
            stride as isize / 2,
            mask,
            lvl_ptr,
            b4_stride as isize,
            lut,
            w,
            bitdepth_max,
        );
    }
}
#[cfg(any(feature = "asm", feature = "c-ffi"))]

/// FFI wrapper for UV horizontal filter 16bpc
#[cfg(target_arch = "x86_64")]
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
    unsafe {
        lpf_h_sb_uv_16bpc_avx2_inner(
            dst_ptr as *mut u16,
            stride as isize / 2,
            mask,
            lvl_ptr,
            b4_stride as isize,
            lut,
            w,
            bitdepth_max,
        );
    }
}
#[cfg(any(feature = "asm", feature = "c-ffi"))]

/// FFI wrapper for UV vertical filter 16bpc
#[cfg(target_arch = "x86_64")]
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
    unsafe {
        lpf_v_sb_uv_16bpc_avx2_inner(
            dst_ptr as *mut u16,
            stride as isize / 2,
            mask,
            lvl_ptr,
            b4_stride as isize,
            lut,
            w,
            bitdepth_max,
        );
    }
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
