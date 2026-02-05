//! Safe ARM NEON implementations for Loop Restoration
//!
//! Wiener filter implementation for ARM. SGR filters use fallback.

#![allow(unused_imports)]
#![allow(clippy::too_many_arguments)]

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

use std::cmp;
use std::ffi::c_int;
use std::slice;

use crate::include::common::bitdepth::AsPrimitive;
use crate::include::common::bitdepth::BitDepth8;
use crate::include::common::bitdepth::BitDepth16;
use crate::include::common::bitdepth::DynPixel;
use crate::include::common::bitdepth::LeftPixelRow;
use crate::include::common::intops::iclip;
use crate::include::dav1d::picture::Rav1dPictureDataComponentOffset;
use crate::src::align::AlignedVec64;
use crate::src::disjoint_mut::DisjointMut;
use crate::src::ffi_safe::FFISafe;
use crate::src::looprestoration::{padding, LrEdgeFlags, LooprestorationParams};
use crate::src::strided::Strided as _;
use libc::ptrdiff_t;

const REST_UNIT_STRIDE: usize = 256 * 3 / 2 + 3 + 3; // = 390

// ============================================================================
// WIENER FILTER IMPLEMENTATIONS - 8BPC
// ============================================================================

fn wiener_filter_8bpc_inner(
    p: Rav1dPictureDataComponentOffset,
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
    p: Rav1dPictureDataComponentOffset,
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
            hor_row[x] = iclip((sum + rounding_off_h) >> round_bits_h.max(0), 0, clip_limit - 1);
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

#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn wiener_filter7_8bpc_neon(
    _p_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    left: *const LeftPixelRow<DynPixel>,
    lpf_ptr: *const DynPixel,
    _lpf_stride: ptrdiff_t,
    w: c_int,
    h: c_int,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    _bitdepth_max: c_int,
    p: *const FFISafe<Rav1dPictureDataComponentOffset>,
    lpf: *const FFISafe<&DisjointMut<AlignedVec64<u8>>>,
) {
    let p = unsafe { *FFISafe::get(p) };
    let left = unsafe { slice::from_raw_parts(left as *const LeftPixelRow<u8>, h as usize + 3) };
    let lpf = unsafe { *FFISafe::get(lpf) };
    let lpf_off = lpf_ptr as isize - lpf.as_ptr() as isize;
    
    wiener_filter_8bpc_inner(p, left, lpf, lpf_off, w as usize, h as usize, params, edges, 7);
}

#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn wiener_filter5_8bpc_neon(
    _p_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    left: *const LeftPixelRow<DynPixel>,
    lpf_ptr: *const DynPixel,
    _lpf_stride: ptrdiff_t,
    w: c_int,
    h: c_int,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    _bitdepth_max: c_int,
    p: *const FFISafe<Rav1dPictureDataComponentOffset>,
    lpf: *const FFISafe<&DisjointMut<AlignedVec64<u8>>>,
) {
    let p = unsafe { *FFISafe::get(p) };
    let left = unsafe { slice::from_raw_parts(left as *const LeftPixelRow<u8>, h as usize + 2) };
    let lpf = unsafe { *FFISafe::get(lpf) };
    let lpf_off = lpf_ptr as isize - lpf.as_ptr() as isize;
    
    wiener_filter_8bpc_inner(p, left, lpf, lpf_off, w as usize, h as usize, params, edges, 5);
}

// ============================================================================
// FFI WRAPPERS - 16BPC
// ============================================================================

#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn wiener_filter7_16bpc_neon(
    _p_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    left: *const LeftPixelRow<DynPixel>,
    lpf_ptr: *const DynPixel,
    _lpf_stride: ptrdiff_t,
    w: c_int,
    h: c_int,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    bitdepth_max: c_int,
    p: *const FFISafe<Rav1dPictureDataComponentOffset>,
    lpf: *const FFISafe<&DisjointMut<AlignedVec64<u8>>>,
) {
    let p = unsafe { *FFISafe::get(p) };
    let left = unsafe { slice::from_raw_parts(left as *const LeftPixelRow<u16>, h as usize + 3) };
    let lpf = unsafe { *FFISafe::get(lpf) };
    let lpf_off = (lpf_ptr as isize - lpf.as_ptr() as isize) / 2;
    
    wiener_filter_16bpc_inner(p, left, lpf, lpf_off, w as usize, h as usize, params, edges, 7, bitdepth_max);
}

#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn wiener_filter5_16bpc_neon(
    _p_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    left: *const LeftPixelRow<DynPixel>,
    lpf_ptr: *const DynPixel,
    _lpf_stride: ptrdiff_t,
    w: c_int,
    h: c_int,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    bitdepth_max: c_int,
    p: *const FFISafe<Rav1dPictureDataComponentOffset>,
    lpf: *const FFISafe<&DisjointMut<AlignedVec64<u8>>>,
) {
    let p = unsafe { *FFISafe::get(p) };
    let left = unsafe { slice::from_raw_parts(left as *const LeftPixelRow<u16>, h as usize + 2) };
    let lpf = unsafe { *FFISafe::get(lpf) };
    let lpf_off = (lpf_ptr as isize - lpf.as_ptr() as isize) / 2;
    
    wiener_filter_16bpc_inner(p, left, lpf, lpf_off, w as usize, h as usize, params, edges, 5, bitdepth_max);
}
