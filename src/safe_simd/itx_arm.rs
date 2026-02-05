//! Safe ARM NEON implementations for Inverse Transform (ITX)
//!
//! Implements the inverse transforms for AV1 decoding.
//! Transforms convert frequency-domain coefficients back to spatial-domain pixels.

#![allow(unused_imports)]
#![allow(clippy::too_many_arguments)]

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

use std::ffi::c_int;

use crate::include::common::bitdepth::AsPrimitive;
use crate::include::common::bitdepth::BitDepth8;
use crate::include::common::bitdepth::BitDepth16;
use crate::include::common::bitdepth::DynCoef;
use crate::include::common::bitdepth::DynPixel;
use crate::include::common::intops::iclip;
use crate::include::dav1d::picture::Rav1dPictureDataComponentOffset;
use crate::src::ffi_safe::FFISafe;
use libc::ptrdiff_t;

// ============================================================================
// WHT_WHT 4x4 TRANSFORM
// ============================================================================

/// WHT 4x4 transform for 8bpc
unsafe fn inv_txfm_add_wht_wht_4x4_8bpc_inner(
    dst: *mut u8,
    dst_stride: isize,
    coeff: *mut i16,
    _eob: i32,
    bitdepth_max: i32,
) {
    let c_ptr = coeff;
    let mut tmp = [0i32; 16];

    // Row transform: load from column-major, store row-major
    for y in 0..4 {
        let in0 = unsafe { *c_ptr.add(y) as i32 } >> 2;
        let in1 = unsafe { *c_ptr.add(y + 4) as i32 } >> 2;
        let in2 = unsafe { *c_ptr.add(y + 8) as i32 } >> 2;
        let in3 = unsafe { *c_ptr.add(y + 12) as i32 } >> 2;

        let t0 = in0 + in1;
        let t2 = in2 - in3;
        let t4 = (t0 - t2) >> 1;
        let t3 = t4 - in3;
        let t1 = t4 - in1;

        tmp[y * 4 + 0] = t0 - t3;
        tmp[y * 4 + 1] = t3;
        tmp[y * 4 + 2] = t1;
        tmp[y * 4 + 3] = t2 + t1;
    }

    // Column transform
    for x in 0..4 {
        let in0 = tmp[0 * 4 + x];
        let in1 = tmp[1 * 4 + x];
        let in2 = tmp[2 * 4 + x];
        let in3 = tmp[3 * 4 + x];

        let t0 = in0 + in1;
        let t2 = in2 - in3;
        let t4 = (t0 - t2) >> 1;
        let t3 = t4 - in3;
        let t1 = t4 - in1;

        tmp[0 * 4 + x] = t0 - t3;
        tmp[1 * 4 + x] = t3;
        tmp[2 * 4 + x] = t1;
        tmp[3 * 4 + x] = t2 + t1;
    }

    // Add to destination
    for y in 0..4 {
        let dst_row = unsafe { dst.offset(y as isize * dst_stride) };
        for x in 0..4 {
            let d = unsafe { *dst_row.add(x) as i32 };
            let c = tmp[y * 4 + x];
            let result = iclip(d + c, 0, bitdepth_max);
            unsafe { *dst_row.add(x) = result as u8 };
        }
    }

    // Clear coefficients
    for i in 0..16 {
        unsafe { *c_ptr.add(i) = 0 };
    }
}

/// WHT 4x4 transform for 16bpc
unsafe fn inv_txfm_add_wht_wht_4x4_16bpc_inner(
    dst: *mut u16,
    dst_stride: isize,
    coeff: *mut i32,
    _eob: i32,
    bitdepth_max: i32,
) {
    let c_ptr = coeff;
    let mut tmp = [0i32; 16];

    // Row transform
    for y in 0..4 {
        let in0 = unsafe { *c_ptr.add(y) } >> 2;
        let in1 = unsafe { *c_ptr.add(y + 4) } >> 2;
        let in2 = unsafe { *c_ptr.add(y + 8) } >> 2;
        let in3 = unsafe { *c_ptr.add(y + 12) } >> 2;

        let t0 = in0 + in1;
        let t2 = in2 - in3;
        let t4 = (t0 - t2) >> 1;
        let t3 = t4 - in3;
        let t1 = t4 - in1;

        tmp[y * 4 + 0] = t0 - t3;
        tmp[y * 4 + 1] = t3;
        tmp[y * 4 + 2] = t1;
        tmp[y * 4 + 3] = t2 + t1;
    }

    // Column transform
    for x in 0..4 {
        let in0 = tmp[0 * 4 + x];
        let in1 = tmp[1 * 4 + x];
        let in2 = tmp[2 * 4 + x];
        let in3 = tmp[3 * 4 + x];

        let t0 = in0 + in1;
        let t2 = in2 - in3;
        let t4 = (t0 - t2) >> 1;
        let t3 = t4 - in3;
        let t1 = t4 - in1;

        tmp[0 * 4 + x] = t0 - t3;
        tmp[1 * 4 + x] = t3;
        tmp[2 * 4 + x] = t1;
        tmp[3 * 4 + x] = t2 + t1;
    }

    // Add to destination
    for y in 0..4 {
        let dst_row = unsafe { dst.offset(y as isize * dst_stride) };
        for x in 0..4 {
            let d = unsafe { *dst_row.add(x) as i32 };
            let c = tmp[y * 4 + x];
            let result = iclip(d + c, 0, bitdepth_max);
            unsafe { *dst_row.add(x) = result as u16 };
        }
    }

    // Clear coefficients
    for i in 0..16 {
        unsafe { *c_ptr.add(i) = 0 };
    }
}

// ============================================================================
// DCT_DCT 4x4 TRANSFORM
// ============================================================================

/// DCT4 1D transform constants
const DCT4_C1: i32 = 2896; // cos(pi/8) * 4096
const DCT4_C2: i32 = 2896; // cos(3pi/8) * 4096  
const DCT4_C3: i32 = 1567; // sin(pi/8) * 4096
const DCT4_C4: i32 = 3784; // sin(3pi/8) * 4096

/// DCT4 1D transform
#[inline(always)]
fn dct4_1d(in0: i32, in1: i32, in2: i32, in3: i32) -> [i32; 4] {
    // Stage 1
    let t0 = in0 + in3;
    let t1 = in1 + in2;
    let t2 = in0 - in3;
    let t3 = in1 - in2;

    // Stage 2 (DCT2)
    let s0 = t0 + t1;
    let s1 = t0 - t1;
    
    // Rotation for t2, t3
    let s2 = ((t2 * 1567 + t3 * 3784) + 2048) >> 12;
    let s3 = ((t2 * 3784 - t3 * 1567) + 2048) >> 12;

    [s0, s2, s1, s3]
}

/// DCT 4x4 transform for 8bpc
unsafe fn inv_txfm_add_dct_dct_4x4_8bpc_inner(
    dst: *mut u8,
    dst_stride: isize,
    coeff: *mut i16,
    _eob: i32,
    bitdepth_max: i32,
) {
    let c_ptr = coeff;
    let mut tmp = [0i32; 16];

    // Row transform (coefficients are in column-major order)
    for y in 0..4 {
        let in0 = unsafe { *c_ptr.add(y) as i32 };
        let in1 = unsafe { *c_ptr.add(y + 4) as i32 };
        let in2 = unsafe { *c_ptr.add(y + 8) as i32 };
        let in3 = unsafe { *c_ptr.add(y + 12) as i32 };

        let out = dct4_1d(in0, in1, in2, in3);
        tmp[y * 4 + 0] = out[0];
        tmp[y * 4 + 1] = out[1];
        tmp[y * 4 + 2] = out[2];
        tmp[y * 4 + 3] = out[3];
    }

    // Column transform and add to dst
    for x in 0..4 {
        let in0 = tmp[0 * 4 + x];
        let in1 = tmp[1 * 4 + x];
        let in2 = tmp[2 * 4 + x];
        let in3 = tmp[3 * 4 + x];

        let out = dct4_1d(in0, in1, in2, in3);
        
        // Round and add to destination
        for y in 0..4 {
            let dst_row = unsafe { dst.offset(y as isize * dst_stride) };
            let d = unsafe { *dst_row.add(x) as i32 };
            // Apply rounding: (val + 8) >> 4
            let c = (out[y] + 8) >> 4;
            let result = iclip(d + c, 0, bitdepth_max);
            unsafe { *dst_row.add(x) = result as u8 };
        }
    }

    // Clear coefficients
    for i in 0..16 {
        unsafe { *c_ptr.add(i) = 0 };
    }
}

/// DCT 4x4 transform for 16bpc
unsafe fn inv_txfm_add_dct_dct_4x4_16bpc_inner(
    dst: *mut u16,
    dst_stride: isize,
    coeff: *mut i32,
    _eob: i32,
    bitdepth_max: i32,
) {
    let c_ptr = coeff;
    let mut tmp = [0i32; 16];

    // Row transform
    for y in 0..4 {
        let in0 = unsafe { *c_ptr.add(y) };
        let in1 = unsafe { *c_ptr.add(y + 4) };
        let in2 = unsafe { *c_ptr.add(y + 8) };
        let in3 = unsafe { *c_ptr.add(y + 12) };

        let out = dct4_1d(in0, in1, in2, in3);
        tmp[y * 4 + 0] = out[0];
        tmp[y * 4 + 1] = out[1];
        tmp[y * 4 + 2] = out[2];
        tmp[y * 4 + 3] = out[3];
    }

    // Column transform and add to dst
    for x in 0..4 {
        let in0 = tmp[0 * 4 + x];
        let in1 = tmp[1 * 4 + x];
        let in2 = tmp[2 * 4 + x];
        let in3 = tmp[3 * 4 + x];

        let out = dct4_1d(in0, in1, in2, in3);
        
        for y in 0..4 {
            let dst_row = unsafe { dst.offset(y as isize * dst_stride) };
            let d = unsafe { *dst_row.add(x) as i32 };
            let c = (out[y] + 8) >> 4;
            let result = iclip(d + c, 0, bitdepth_max);
            unsafe { *dst_row.add(x) = result as u16 };
        }
    }

    // Clear coefficients
    for i in 0..16 {
        unsafe { *c_ptr.add(i) = 0 };
    }
}

// ============================================================================
// FFI WRAPPERS
// ============================================================================

#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn inv_txfm_add_wht_wht_4x4_8bpc_neon(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: i32,
    bitdepth_max: i32,
    _coeff_len: u16,
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    inv_txfm_add_wht_wht_4x4_8bpc_inner(
        dst_ptr as *mut u8,
        dst_stride,
        coeff as *mut i16,
        eob,
        bitdepth_max,
    );
}

#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn inv_txfm_add_wht_wht_4x4_16bpc_neon(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: i32,
    bitdepth_max: i32,
    _coeff_len: u16,
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let dst_stride_u16 = dst_stride / 2;
    inv_txfm_add_wht_wht_4x4_16bpc_inner(
        dst_ptr as *mut u16,
        dst_stride_u16,
        coeff as *mut i32,
        eob,
        bitdepth_max,
    );
}

#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_4x4_8bpc_neon(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: i32,
    bitdepth_max: i32,
    _coeff_len: u16,
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    inv_txfm_add_dct_dct_4x4_8bpc_inner(
        dst_ptr as *mut u8,
        dst_stride,
        coeff as *mut i16,
        eob,
        bitdepth_max,
    );
}

#[cfg(target_arch = "aarch64")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_4x4_16bpc_neon(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: i32,
    bitdepth_max: i32,
    _coeff_len: u16,
    _dst: *const FFISafe<Rav1dPictureDataComponentOffset>,
) {
    let dst_stride_u16 = dst_stride / 2;
    inv_txfm_add_dct_dct_4x4_16bpc_inner(
        dst_ptr as *mut u16,
        dst_stride_u16,
        coeff as *mut i32,
        eob,
        bitdepth_max,
    );
}
