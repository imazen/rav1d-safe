// ============================================================================
// ADST 4x4 TRANSFORMS
// ============================================================================

/// ADST4 1D transform applied to a single 4-element vector (returns 4 outputs).
///
/// ADST4 coefficients (derived from spec): 1321, 3803, 2482, 3344.
/// The code uses (val - 4096) trick to avoid overflow.
#[inline(always)]
fn adst4_1d_scalar(
    in0: i32,
    in1: i32,
    in2: i32,
    in3: i32,
    min: i32,
    max: i32,
) -> (i32, i32, i32, i32) {
    let clip = |v: i32| v.clamp(min, max);

    let out0 =
        ((1321 * in0 + (3803 - 4096) * in2 + (2482 - 4096) * in3 + (3344 - 4096) * in1 + 2048)
            >> 12)
            + in2
            + in3
            + in1;
    let out1 =
        (((2482 - 4096) * in0 - 1321 * in2 - (3803 - 4096) * in3 + (3344 - 4096) * in1 + 2048)
            >> 12)
            + in0
            - in3
            + in1;
    let out2 = (209 * (in0 - in2 + in3) + 128) >> 8;
    let out3 = (((3803 - 4096) * in0 + (2482 - 4096) * in2 - 1321 * in3 - (3344 - 4096) * in1
        + 2048)
        >> 12)
        + in0
        + in2
        - in1;

    (clip(out0), clip(out1), clip(out2), clip(out3))
}

/// DCT4 1D transform (scalar, for combining with ADST)
#[inline(always)]
fn dct4_1d_scalar(
    in0: i32,
    in1: i32,
    in2: i32,
    in3: i32,
    min: i32,
    max: i32,
) -> (i32, i32, i32, i32) {
    let clip = |v: i32| v.clamp(min, max);
    let t0 = (in0 + in2) * 181 + 128 >> 8;
    let t1 = (in0 - in2) * 181 + 128 >> 8;
    let t2 = ((in1 * 1567 - in3 * (3784 - 4096) + 2048) >> 12) - in3;
    let t3 = ((in1 * (3784 - 4096) + in3 * 1567 + 2048) >> 12) + in1;

    (clip(t0 + t3), clip(t1 + t2), clip(t1 - t2), clip(t0 - t3))
}

/// ADST_DCT 4x4: ADST on rows, DCT on columns
#[cfg(target_arch = "x86_64")]
pub fn inv_txfm_add_adst_dct_4x4_8bpc_avx2_inner(
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    coeff: &mut [i16],
    _eob: i32,
    _bitdepth_max: i32,
) {
    // Load coefficients into a 4x4 matrix
    let mut c = [[0i32; 4]; 4];
    for y in 0..4 {
        for x in 0..4 {
            c[y][x] = coeff[y + x * 4] as i32;
        }
    }

    // First pass: ADST on rows
    let mut tmp = [[0i32; 4]; 4];
    for y in 0..4 {
        let (o0, o1, o2, o3) = adst4_1d_scalar(
            c[y][0],
            c[y][1],
            c[y][2],
            c[y][3],
            i16::MIN as i32,
            i16::MAX as i32,
        );
        tmp[y][0] = o0;
        tmp[y][1] = o1;
        tmp[y][2] = o2;
        tmp[y][3] = o3;
    }

    // Second pass: DCT on columns
    let mut out = [[0i32; 4]; 4];
    for x in 0..4 {
        let (o0, o1, o2, o3) = dct4_1d_scalar(
            tmp[0][x],
            tmp[1][x],
            tmp[2][x],
            tmp[3][x],
            i16::MIN as i32,
            i16::MAX as i32,
        );
        out[0][x] = o0;
        out[1][x] = o1;
        out[2][x] = o2;
        out[3][x] = o3;
    }

    // Add to destination with rounding and clipping
    for y in 0..4 {
        let row_off = dst_base.wrapping_add_signed(y as isize * dst_stride);
        for x in 0..4 {
            let pixel = dst[row_off + x] as i32;
            let val = pixel + ((out[y][x] + 8) >> 4);
            dst[row_off + x] = val.clamp(0, 255) as u8;
        }
    }

    // Clear coefficients
    coeff[..16].fill(0);
}

/// DCT_ADST 4x4: DCT on rows, ADST on columns
#[cfg(target_arch = "x86_64")]
pub fn inv_txfm_add_dct_adst_4x4_8bpc_avx2_inner(
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    coeff: &mut [i16],
    _eob: i32,
    _bitdepth_max: i32,
) {
    // Load coefficients into a 4x4 matrix
    let mut c = [[0i32; 4]; 4];
    for y in 0..4 {
        for x in 0..4 {
            c[y][x] = coeff[y + x * 4] as i32;
        }
    }

    // First pass: DCT on rows
    let mut tmp = [[0i32; 4]; 4];
    for y in 0..4 {
        let (o0, o1, o2, o3) = dct4_1d_scalar(
            c[y][0],
            c[y][1],
            c[y][2],
            c[y][3],
            i16::MIN as i32,
            i16::MAX as i32,
        );
        tmp[y][0] = o0;
        tmp[y][1] = o1;
        tmp[y][2] = o2;
        tmp[y][3] = o3;
    }

    // Second pass: ADST on columns
    let mut out = [[0i32; 4]; 4];
    for x in 0..4 {
        let (o0, o1, o2, o3) = adst4_1d_scalar(
            tmp[0][x],
            tmp[1][x],
            tmp[2][x],
            tmp[3][x],
            i16::MIN as i32,
            i16::MAX as i32,
        );
        out[0][x] = o0;
        out[1][x] = o1;
        out[2][x] = o2;
        out[3][x] = o3;
    }

    // Add to destination with rounding and clipping
    for y in 0..4 {
        let row_off = dst_base.wrapping_add_signed(y as isize * dst_stride);
        for x in 0..4 {
            let pixel = dst[row_off + x] as i32;
            let val = pixel + ((out[y][x] + 8) >> 4);
            dst[row_off + x] = val.clamp(0, 255) as u8;
        }
    }

    // Clear coefficients
    coeff[..16].fill(0);
}

/// ADST_ADST 4x4: ADST on both rows and columns
#[cfg(target_arch = "x86_64")]
pub fn inv_txfm_add_adst_adst_4x4_8bpc_avx2_inner(
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    coeff: &mut [i16],
    _eob: i32,
    _bitdepth_max: i32,
) {
    // Load coefficients
    let mut c = [[0i32; 4]; 4];
    for y in 0..4 {
        for x in 0..4 {
            c[y][x] = coeff[y + x * 4] as i32;
        }
    }

    // First pass: ADST on rows
    let mut tmp = [[0i32; 4]; 4];
    for y in 0..4 {
        let (o0, o1, o2, o3) = adst4_1d_scalar(
            c[y][0],
            c[y][1],
            c[y][2],
            c[y][3],
            i16::MIN as i32,
            i16::MAX as i32,
        );
        tmp[y][0] = o0;
        tmp[y][1] = o1;
        tmp[y][2] = o2;
        tmp[y][3] = o3;
    }

    // Second pass: ADST on columns
    let mut out = [[0i32; 4]; 4];
    for x in 0..4 {
        let (o0, o1, o2, o3) = adst4_1d_scalar(
            tmp[0][x],
            tmp[1][x],
            tmp[2][x],
            tmp[3][x],
            i16::MIN as i32,
            i16::MAX as i32,
        );
        out[0][x] = o0;
        out[1][x] = o1;
        out[2][x] = o2;
        out[3][x] = o3;
    }

    // Add to destination
    for y in 0..4 {
        let row_off = dst_base.wrapping_add_signed(y as isize * dst_stride);
        for x in 0..4 {
            let pixel = dst[row_off + x] as i32;
            let val = pixel + ((out[y][x] + 8) >> 4);
            dst[row_off + x] = val.clamp(0, 255) as u8;
        }
    }

    // Clear coefficients
    coeff[..16].fill(0);
}

// ============================================================================
// ADST FFI WRAPPERS
// ============================================================================

/// FFI wrapper for ADST_DCT 4x4 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_adst_dct_4x4_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let abs_stride = dst_stride.unsigned_abs();
    let buf_size = 3 * abs_stride + 4;
    let (base, dst_slice) = if dst_stride >= 0 {
        (0usize, unsafe {
            std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_size)
        })
    } else {
        let start = unsafe { (dst_ptr as *mut u8).offset(3 * dst_stride) };
        (3 * abs_stride, unsafe {
            std::slice::from_raw_parts_mut(start, buf_size)
        })
    };
    let coeff_slice = unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, 16) };
    inv_txfm_add_adst_dct_4x4_8bpc_avx2_inner(
        dst_slice,
        base,
        dst_stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// FFI wrapper for DCT_ADST 4x4 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_adst_4x4_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let abs_stride = dst_stride.unsigned_abs();
    let buf_size = 3 * abs_stride + 4;
    let (base, dst_slice) = if dst_stride >= 0 {
        (0usize, unsafe {
            std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_size)
        })
    } else {
        let start = unsafe { (dst_ptr as *mut u8).offset(3 * dst_stride) };
        (3 * abs_stride, unsafe {
            std::slice::from_raw_parts_mut(start, buf_size)
        })
    };
    let coeff_slice = unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, 16) };
    inv_txfm_add_dct_adst_4x4_8bpc_avx2_inner(
        dst_slice,
        base,
        dst_stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// FFI wrapper for ADST_ADST 4x4 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_adst_adst_4x4_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let abs_stride = dst_stride.unsigned_abs();
    let buf_size = 3 * abs_stride + 4;
    let (base, dst_slice) = if dst_stride >= 0 {
        (0usize, unsafe {
            std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_size)
        })
    } else {
        let start = unsafe { (dst_ptr as *mut u8).offset(3 * dst_stride) };
        (3 * abs_stride, unsafe {
            std::slice::from_raw_parts_mut(start, buf_size)
        })
    };
    let coeff_slice = unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, 16) };
    inv_txfm_add_adst_adst_4x4_8bpc_avx2_inner(
        dst_slice,
        base,
        dst_stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// FLIPADST TRANSFORMS (reverse order output)
// ============================================================================

/// FlipADST4 1D transform - same as ADST but output in reverse order
#[inline(always)]
fn flipadst4_1d_scalar(
    in0: i32,
    in1: i32,
    in2: i32,
    in3: i32,
    min: i32,
    max: i32,
) -> (i32, i32, i32, i32) {
    let (o0, o1, o2, o3) = adst4_1d_scalar(in0, in1, in2, in3, min, max);
    (o3, o2, o1, o0) // Flip the output order
}

/// FLIPADST_DCT 4x4: FlipADST on rows, DCT on columns
#[cfg(target_arch = "x86_64")]
pub fn inv_txfm_add_flipadst_dct_4x4_8bpc_avx2_inner(
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    coeff: &mut [i16],
    _eob: i32,
    _bitdepth_max: i32,
) {
    let mut c = [[0i32; 4]; 4];
    for y in 0..4 {
        for x in 0..4 {
            c[y][x] = coeff[y + x * 4] as i32;
        }
    }

    // First pass: FlipADST on rows
    let mut tmp = [[0i32; 4]; 4];
    for y in 0..4 {
        let (o0, o1, o2, o3) = flipadst4_1d_scalar(
            c[y][0],
            c[y][1],
            c[y][2],
            c[y][3],
            i16::MIN as i32,
            i16::MAX as i32,
        );
        tmp[y][0] = o0;
        tmp[y][1] = o1;
        tmp[y][2] = o2;
        tmp[y][3] = o3;
    }

    // Second pass: DCT on columns
    let mut out = [[0i32; 4]; 4];
    for x in 0..4 {
        let (o0, o1, o2, o3) = dct4_1d_scalar(
            tmp[0][x],
            tmp[1][x],
            tmp[2][x],
            tmp[3][x],
            i16::MIN as i32,
            i16::MAX as i32,
        );
        out[0][x] = o0;
        out[1][x] = o1;
        out[2][x] = o2;
        out[3][x] = o3;
    }

    for y in 0..4 {
        let row_off = dst_base.wrapping_add_signed(y as isize * dst_stride);
        for x in 0..4 {
            let pixel = dst[row_off + x] as i32;
            let val = pixel + ((out[y][x] + 8) >> 4);
            dst[row_off + x] = val.clamp(0, 255) as u8;
        }
    }

    coeff[..16].fill(0);
}

/// DCT_FLIPADST 4x4: DCT on rows, FlipADST on columns
#[cfg(target_arch = "x86_64")]
pub fn inv_txfm_add_dct_flipadst_4x4_8bpc_avx2_inner(
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    coeff: &mut [i16],
    _eob: i32,
    _bitdepth_max: i32,
) {
    let mut c = [[0i32; 4]; 4];
    for y in 0..4 {
        for x in 0..4 {
            c[y][x] = coeff[y + x * 4] as i32;
        }
    }

    // First pass: DCT on rows
    let mut tmp = [[0i32; 4]; 4];
    for y in 0..4 {
        let (o0, o1, o2, o3) = dct4_1d_scalar(
            c[y][0],
            c[y][1],
            c[y][2],
            c[y][3],
            i16::MIN as i32,
            i16::MAX as i32,
        );
        tmp[y][0] = o0;
        tmp[y][1] = o1;
        tmp[y][2] = o2;
        tmp[y][3] = o3;
    }

    // Second pass: FlipADST on columns
    let mut out = [[0i32; 4]; 4];
    for x in 0..4 {
        let (o0, o1, o2, o3) = flipadst4_1d_scalar(
            tmp[0][x],
            tmp[1][x],
            tmp[2][x],
            tmp[3][x],
            i16::MIN as i32,
            i16::MAX as i32,
        );
        out[0][x] = o0;
        out[1][x] = o1;
        out[2][x] = o2;
        out[3][x] = o3;
    }

    for y in 0..4 {
        let row_off = dst_base.wrapping_add_signed(y as isize * dst_stride);
        for x in 0..4 {
            let pixel = dst[row_off + x] as i32;
            let val = pixel + ((out[y][x] + 8) >> 4);
            dst[row_off + x] = val.clamp(0, 255) as u8;
        }
    }

    coeff[..16].fill(0);
}

/// ADST_FLIPADST 4x4
#[cfg(target_arch = "x86_64")]
pub fn inv_txfm_add_adst_flipadst_4x4_8bpc_avx2_inner(
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    coeff: &mut [i16],
    _eob: i32,
    _bitdepth_max: i32,
) {
    let mut c = [[0i32; 4]; 4];
    for y in 0..4 {
        for x in 0..4 {
            c[y][x] = coeff[y + x * 4] as i32;
        }
    }

    let mut tmp = [[0i32; 4]; 4];
    for y in 0..4 {
        let (o0, o1, o2, o3) = adst4_1d_scalar(
            c[y][0],
            c[y][1],
            c[y][2],
            c[y][3],
            i16::MIN as i32,
            i16::MAX as i32,
        );
        tmp[y][0] = o0;
        tmp[y][1] = o1;
        tmp[y][2] = o2;
        tmp[y][3] = o3;
    }

    let mut out = [[0i32; 4]; 4];
    for x in 0..4 {
        let (o0, o1, o2, o3) = flipadst4_1d_scalar(
            tmp[0][x],
            tmp[1][x],
            tmp[2][x],
            tmp[3][x],
            i16::MIN as i32,
            i16::MAX as i32,
        );
        out[0][x] = o0;
        out[1][x] = o1;
        out[2][x] = o2;
        out[3][x] = o3;
    }

    for y in 0..4 {
        let row_off = dst_base.wrapping_add_signed(y as isize * dst_stride);
        for x in 0..4 {
            let pixel = dst[row_off + x] as i32;
            let val = pixel + ((out[y][x] + 8) >> 4);
            dst[row_off + x] = val.clamp(0, 255) as u8;
        }
    }
    coeff[..16].fill(0);
}

/// FLIPADST_ADST 4x4
#[cfg(target_arch = "x86_64")]
pub fn inv_txfm_add_flipadst_adst_4x4_8bpc_avx2_inner(
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    coeff: &mut [i16],
    _eob: i32,
    _bitdepth_max: i32,
) {
    let mut c = [[0i32; 4]; 4];
    for y in 0..4 {
        for x in 0..4 {
            c[y][x] = coeff[y + x * 4] as i32;
        }
    }

    let mut tmp = [[0i32; 4]; 4];
    for y in 0..4 {
        let (o0, o1, o2, o3) = flipadst4_1d_scalar(
            c[y][0],
            c[y][1],
            c[y][2],
            c[y][3],
            i16::MIN as i32,
            i16::MAX as i32,
        );
        tmp[y][0] = o0;
        tmp[y][1] = o1;
        tmp[y][2] = o2;
        tmp[y][3] = o3;
    }

    let mut out = [[0i32; 4]; 4];
    for x in 0..4 {
        let (o0, o1, o2, o3) = adst4_1d_scalar(
            tmp[0][x],
            tmp[1][x],
            tmp[2][x],
            tmp[3][x],
            i16::MIN as i32,
            i16::MAX as i32,
        );
        out[0][x] = o0;
        out[1][x] = o1;
        out[2][x] = o2;
        out[3][x] = o3;
    }

    for y in 0..4 {
        let row_off = dst_base.wrapping_add_signed(y as isize * dst_stride);
        for x in 0..4 {
            let pixel = dst[row_off + x] as i32;
            let val = pixel + ((out[y][x] + 8) >> 4);
            dst[row_off + x] = val.clamp(0, 255) as u8;
        }
    }
    coeff[..16].fill(0);
}

/// FLIPADST_FLIPADST 4x4
#[cfg(target_arch = "x86_64")]
pub fn inv_txfm_add_flipadst_flipadst_4x4_8bpc_avx2_inner(
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    coeff: &mut [i16],
    _eob: i32,
    _bitdepth_max: i32,
) {
    let mut c = [[0i32; 4]; 4];
    for y in 0..4 {
        for x in 0..4 {
            c[y][x] = coeff[y + x * 4] as i32;
        }
    }

    let mut tmp = [[0i32; 4]; 4];
    for y in 0..4 {
        let (o0, o1, o2, o3) = flipadst4_1d_scalar(
            c[y][0],
            c[y][1],
            c[y][2],
            c[y][3],
            i16::MIN as i32,
            i16::MAX as i32,
        );
        tmp[y][0] = o0;
        tmp[y][1] = o1;
        tmp[y][2] = o2;
        tmp[y][3] = o3;
    }

    let mut out = [[0i32; 4]; 4];
    for x in 0..4 {
        let (o0, o1, o2, o3) = flipadst4_1d_scalar(
            tmp[0][x],
            tmp[1][x],
            tmp[2][x],
            tmp[3][x],
            i16::MIN as i32,
            i16::MAX as i32,
        );
        out[0][x] = o0;
        out[1][x] = o1;
        out[2][x] = o2;
        out[3][x] = o3;
    }

    for y in 0..4 {
        let row_off = dst_base.wrapping_add_signed(y as isize * dst_stride);
        for x in 0..4 {
            let pixel = dst[row_off + x] as i32;
            let val = pixel + ((out[y][x] + 8) >> 4);
            dst[row_off + x] = val.clamp(0, 255) as u8;
        }
    }
    coeff[..16].fill(0);
}

// FFI wrappers for FlipADST variants
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_flipadst_dct_4x4_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let abs_stride = dst_stride.unsigned_abs();
    let buf_size = 3 * abs_stride + 4;
    let (base, dst_slice) = if dst_stride >= 0 {
        (0usize, unsafe {
            std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_size)
        })
    } else {
        let start = unsafe { (dst_ptr as *mut u8).offset(3 * dst_stride) };
        (3 * abs_stride, unsafe {
            std::slice::from_raw_parts_mut(start, buf_size)
        })
    };
    let coeff_slice = unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, 16) };
    inv_txfm_add_flipadst_dct_4x4_8bpc_avx2_inner(
        dst_slice,
        base,
        dst_stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_flipadst_4x4_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let abs_stride = dst_stride.unsigned_abs();
    let buf_size = 3 * abs_stride + 4;
    let (base, dst_slice) = if dst_stride >= 0 {
        (0usize, unsafe {
            std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_size)
        })
    } else {
        let start = unsafe { (dst_ptr as *mut u8).offset(3 * dst_stride) };
        (3 * abs_stride, unsafe {
            std::slice::from_raw_parts_mut(start, buf_size)
        })
    };
    let coeff_slice = unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, 16) };
    inv_txfm_add_dct_flipadst_4x4_8bpc_avx2_inner(
        dst_slice,
        base,
        dst_stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_adst_flipadst_4x4_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let abs_stride = dst_stride.unsigned_abs();
    let buf_size = 3 * abs_stride + 4;
    let (base, dst_slice) = if dst_stride >= 0 {
        (0usize, unsafe {
            std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_size)
        })
    } else {
        let start = unsafe { (dst_ptr as *mut u8).offset(3 * dst_stride) };
        (3 * abs_stride, unsafe {
            std::slice::from_raw_parts_mut(start, buf_size)
        })
    };
    let coeff_slice = unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, 16) };
    inv_txfm_add_adst_flipadst_4x4_8bpc_avx2_inner(
        dst_slice,
        base,
        dst_stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_flipadst_adst_4x4_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let abs_stride = dst_stride.unsigned_abs();
    let buf_size = 3 * abs_stride + 4;
    let (base, dst_slice) = if dst_stride >= 0 {
        (0usize, unsafe {
            std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_size)
        })
    } else {
        let start = unsafe { (dst_ptr as *mut u8).offset(3 * dst_stride) };
        (3 * abs_stride, unsafe {
            std::slice::from_raw_parts_mut(start, buf_size)
        })
    };
    let coeff_slice = unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, 16) };
    inv_txfm_add_flipadst_adst_4x4_8bpc_avx2_inner(
        dst_slice,
        base,
        dst_stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_flipadst_flipadst_4x4_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let abs_stride = dst_stride.unsigned_abs();
    let buf_size = 3 * abs_stride + 4;
    let (base, dst_slice) = if dst_stride >= 0 {
        (0usize, unsafe {
            std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_size)
        })
    } else {
        let start = unsafe { (dst_ptr as *mut u8).offset(3 * dst_stride) };
        (3 * abs_stride, unsafe {
            std::slice::from_raw_parts_mut(start, buf_size)
        })
    };
    let coeff_slice = unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, 16) };
    inv_txfm_add_flipadst_flipadst_4x4_8bpc_avx2_inner(
        dst_slice,
        base,
        dst_stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// ADST 8x8 TRANSFORMS
// ============================================================================

/// ADST8 1D transform (scalar)
#[inline(always)]
fn adst8_1d_scalar(
    in0: i32,
    in1: i32,
    in2: i32,
    in3: i32,
    in4: i32,
    in5: i32,
    in6: i32,
    in7: i32,
    min: i32,
    max: i32,
) -> (i32, i32, i32, i32, i32, i32, i32, i32) {
    let clip = |v: i32| v.clamp(min, max);

    let t0a = (((4076 - 4096) * in7 + 401 * in0 + 2048) >> 12) + in7;
    let t1a = ((401 * in7 - (4076 - 4096) * in0 + 2048) >> 12) - in0;
    let t2a = (((3612 - 4096) * in5 + 1931 * in2 + 2048) >> 12) + in5;
    let t3a = ((1931 * in5 - (3612 - 4096) * in2 + 2048) >> 12) - in2;
    let t4a = (1299 * in3 + 1583 * in4 + 1024) >> 11;
    let t5a = (1583 * in3 - 1299 * in4 + 1024) >> 11;
    let t6a = ((1189 * in1 + (3920 - 4096) * in6 + 2048) >> 12) + in6;
    let t7a = (((3920 - 4096) * in1 - 1189 * in6 + 2048) >> 12) + in1;

    let t0 = clip(t0a + t4a);
    let t1 = clip(t1a + t5a);
    let t2 = clip(t2a + t6a);
    let t3 = clip(t3a + t7a);
    let t4 = clip(t0a - t4a);
    let t5 = clip(t1a - t5a);
    let t6 = clip(t2a - t6a);
    let t7 = clip(t3a - t7a);

    let t4a = (((3784 - 4096) * t4 + 1567 * t5 + 2048) >> 12) + t4;
    let t5a = ((1567 * t4 - (3784 - 4096) * t5 + 2048) >> 12) - t5;
    let t6a = (((3784 - 4096) * t7 - 1567 * t6 + 2048) >> 12) + t7;
    let t7a = ((1567 * t7 + (3784 - 4096) * t6 + 2048) >> 12) + t6;

    let out0 = clip(t0 + t2);
    let out7 = -clip(t1 + t3);
    let t2_final = clip(t0 - t2);
    let t3_final = clip(t1 - t3);
    let out1 = -clip(t4a + t6a);
    let out6 = clip(t5a + t7a);
    let t6_final = clip(t4a - t6a);
    let t7_final = clip(t5a - t7a);

    let out3 = -(((t2_final + t3_final) * 181 + 128) >> 8);
    let out4 = ((t2_final - t3_final) * 181 + 128) >> 8;
    let out2 = ((t6_final + t7_final) * 181 + 128) >> 8;
    let out5 = -(((t6_final - t7_final) * 181 + 128) >> 8);

    (out0, out1, out2, out3, out4, out5, out6, out7)
}

/// FlipADST8 1D transform - ADST8 with reversed output
#[inline(always)]
fn flipadst8_1d_scalar(
    in0: i32,
    in1: i32,
    in2: i32,
    in3: i32,
    in4: i32,
    in5: i32,
    in6: i32,
    in7: i32,
    min: i32,
    max: i32,
) -> (i32, i32, i32, i32, i32, i32, i32, i32) {
    let (o0, o1, o2, o3, o4, o5, o6, o7) =
        adst8_1d_scalar(in0, in1, in2, in3, in4, in5, in6, in7, min, max);
    (o7, o6, o5, o4, o3, o2, o1, o0)
}

/// DCT8 1D transform (scalar)
#[inline(always)]
fn dct8_1d_scalar(
    in0: i32,
    in1: i32,
    in2: i32,
    in3: i32,
    in4: i32,
    in5: i32,
    in6: i32,
    in7: i32,
    min: i32,
    max: i32,
) -> (i32, i32, i32, i32, i32, i32, i32, i32) {
    let clip = |v: i32| v.clamp(min, max);

    // First do DCT4 on even samples
    let t0 = ((in0 + in4) * 181 + 128) >> 8;
    let t1 = ((in0 - in4) * 181 + 128) >> 8;
    let t2 = ((in2 * 1567 - in6 * (3784 - 4096) + 2048) >> 12) - in6;
    let t3 = ((in2 * (3784 - 4096) + in6 * 1567 + 2048) >> 12) + in2;

    let t0a = clip(t0 + t3);
    let t1a = clip(t1 + t2);
    let t2a = clip(t1 - t2);
    let t3a = clip(t0 - t3);

    // Then do the 8-point specific part
    let t4a = ((in1 * 799 - in7 * (4017 - 4096) + 2048) >> 12) - in7;
    let t5a = (in5 * 1703 - in3 * 1138 + 1024) >> 11;
    let t6a = (in5 * 1138 + in3 * 1703 + 1024) >> 11;
    let t7a = ((in1 * (4017 - 4096) + in7 * 799 + 2048) >> 12) + in1;

    let t4 = clip(t4a + t5a);
    let t5 = clip(t4a - t5a);
    let t7 = clip(t7a + t6a);
    let t6 = clip(t7a - t6a);

    let t5b = ((t6 - t5) * 181 + 128) >> 8;
    let t6b = ((t6 + t5) * 181 + 128) >> 8;

    (
        clip(t0a + t7),
        clip(t1a + t6b),
        clip(t2a + t5b),
        clip(t3a + t4),
        clip(t3a - t4),
        clip(t2a - t5b),
        clip(t1a - t6b),
        clip(t0a - t7),
    )
}

/// Helper macro for 8x8 transform implementations
#[allow(unused_macros)]
macro_rules! impl_8x8_transform {
    ($name:ident, $row_fn:ident, $col_fn:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[arcane]
        pub fn $name(
            _token: Desktop64,
            dst: &mut [u8],
            dst_stride: usize,
            coeff: &mut [i16],
            _eob: i32,
            _bitdepth_max: i32,
        ) {
            use crate::src::safe_simd::pixel_access::{
                loadi32, loadi64, loadu_128, storei32, storei64, storeu_128,
            };
            let mut dst = dst.flex_mut();
            let mut coeff = coeff.flex_mut();
            const MIN: i32 = i16::MIN as i32;
            const MAX: i32 = i16::MAX as i32;

            // Load coefficients
            let mut c = [[0i32; 8]; 8];
            for y in 0..8 {
                for x in 0..8 {
                    c[y][x] = coeff[y * 8 + x] as i32;
                }
            }

            // First pass: transform on rows
            let mut tmp = [[0i32; 8]; 8];
            for y in 0..8 {
                let (o0, o1, o2, o3, o4, o5, o6, o7) = $row_fn(
                    c[y][0], c[y][1], c[y][2], c[y][3], c[y][4], c[y][5], c[y][6], c[y][7], MIN,
                    MAX,
                );
                tmp[y][0] = o0;
                tmp[y][1] = o1;
                tmp[y][2] = o2;
                tmp[y][3] = o3;
                tmp[y][4] = o4;
                tmp[y][5] = o5;
                tmp[y][6] = o6;
                tmp[y][7] = o7;
            }

            // Second pass: transform on columns
            let mut out = [[0i32; 8]; 8];
            for x in 0..8 {
                let (o0, o1, o2, o3, o4, o5, o6, o7) = $col_fn(
                    tmp[0][x], tmp[1][x], tmp[2][x], tmp[3][x], tmp[4][x], tmp[5][x], tmp[6][x],
                    tmp[7][x], MIN, MAX,
                );
                out[0][x] = o0;
                out[1][x] = o1;
                out[2][x] = o2;
                out[3][x] = o3;
                out[4][x] = o4;
                out[5][x] = o5;
                out[6][x] = o6;
                out[7][x] = o7;
            }

            // Add to destination with rounding
            for y in 0..8 {
                let dst_off = y * dst_stride;
                for x in 0..8 {
                    let pixel = dst[dst_off + x] as i32;
                    let val = pixel + ((out[y][x] + 8) >> 4);
                    dst[dst_off + x] = val.clamp(0, 255) as u8;
                }
            }

            // Clear coefficients
            coeff[..64].fill(0);
        }
    };
}

/// 8x8 8bpc variant with SIMD column pass.
/// Row pass uses scalar tuple `row_fn`, column pass uses a SIMD helper that
/// operates on a flat row-major `[i32; 64]` buffer (e.g. `dct8_1d_cols8`).
macro_rules! impl_8x8_transform_simd_col {
    ($name:ident, $row_fn:ident, $simd_col_fn:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[arcane]
        pub fn $name(
            _token: Desktop64,
            dst: &mut [u8],
            dst_stride: usize,
            coeff: &mut [i16],
            _eob: i32,
            _bitdepth_max: i32,
        ) {
            let mut dst = dst.flex_mut();
            let mut coeff = coeff.flex_mut();
            const MIN: i32 = i16::MIN as i32;
            const MAX: i32 = i16::MAX as i32;

            // Row pass: scalar tuple, store row-major to flat tmp.
            let mut tmp = [0i32; 64];
            for y in 0..8 {
                let (o0, o1, o2, o3, o4, o5, o6, o7) = $row_fn(
                    coeff[y * 8] as i32,
                    coeff[y * 8 + 1] as i32,
                    coeff[y * 8 + 2] as i32,
                    coeff[y * 8 + 3] as i32,
                    coeff[y * 8 + 4] as i32,
                    coeff[y * 8 + 5] as i32,
                    coeff[y * 8 + 6] as i32,
                    coeff[y * 8 + 7] as i32,
                    MIN,
                    MAX,
                );
                tmp[y * 8] = o0;
                tmp[y * 8 + 1] = o1;
                tmp[y * 8 + 2] = o2;
                tmp[y * 8 + 3] = o3;
                tmp[y * 8 + 4] = o4;
                tmp[y * 8 + 5] = o5;
                tmp[y * 8 + 6] = o6;
                tmp[y * 8 + 7] = o7;
            }

            // SIMD column pass: 8 cols × 8 rows in one chunk.
            {
                let min_v = _mm256_set1_epi32(MIN);
                let max_v = _mm256_set1_epi32(MAX);
                let mut v = [_mm256_setzero_si256(); 8];
                for i in 0..8 {
                    v[i] = loadu_256!(&tmp[i * 8..i * 8 + 8], [i32; 8]);
                }
                $simd_col_fn(_token, &mut v, min_v, max_v);
                for i in 0..8 {
                    storeu_256!(&mut tmp[i * 8..i * 8 + 8], [i32; 8], v[i]);
                }
            }

            // Add to destination with rounding
            for y in 0..8 {
                let dst_off = y * dst_stride;
                for x in 0..8 {
                    let pixel = dst[dst_off + x] as i32;
                    let val = pixel + ((tmp[y * 8 + x] + 8) >> 4);
                    dst[dst_off + x] = val.clamp(0, 255) as u8;
                }
            }

            coeff[..64].fill(0);
        }
    };
}

// Generate all 8x8 ADST/FlipADST combinations (SIMD col where possible)
impl_8x8_transform_simd_col!(
    inv_txfm_add_adst_dct_8x8_8bpc_avx2_inner,
    adst8_1d_scalar,
    dct8_1d_cols8
);
impl_8x8_transform_simd_col!(
    inv_txfm_add_dct_adst_8x8_8bpc_avx2_inner,
    dct8_1d_scalar,
    adst8_1d_cols8
);
impl_8x8_transform_simd_col!(
    inv_txfm_add_adst_adst_8x8_8bpc_avx2_inner,
    adst8_1d_scalar,
    adst8_1d_cols8
);
impl_8x8_transform_simd_col!(
    inv_txfm_add_flipadst_dct_8x8_8bpc_avx2_inner,
    flipadst8_1d_scalar,
    dct8_1d_cols8
);
impl_8x8_transform_simd_col!(
    inv_txfm_add_dct_flipadst_8x8_8bpc_avx2_inner,
    dct8_1d_scalar,
    flipadst8_1d_cols8
);
impl_8x8_transform_simd_col!(
    inv_txfm_add_flipadst_flipadst_8x8_8bpc_avx2_inner,
    flipadst8_1d_scalar,
    flipadst8_1d_cols8
);
impl_8x8_transform_simd_col!(
    inv_txfm_add_adst_flipadst_8x8_8bpc_avx2_inner,
    adst8_1d_scalar,
    flipadst8_1d_cols8
);
impl_8x8_transform_simd_col!(
    inv_txfm_add_flipadst_adst_8x8_8bpc_avx2_inner,
    flipadst8_1d_scalar,
    adst8_1d_cols8
);

// FFI wrappers for 8x8 transforms
macro_rules! impl_8x8_ffi_wrapper {
    ($wrapper:ident, $inner:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2")]
        #[cfg(feature = "asm")]
        pub unsafe extern "C" fn $wrapper(
            dst_ptr: *mut DynPixel,
            dst_stride: isize,
            coeff: *mut DynCoef,
            eob: c_int,
            bitdepth_max: c_int,
            _coeff_len: u16,
            _dst: *const FFISafe<PicOffset>,
        ) {
            let _token = unsafe { Desktop64::forge_token_dangerously() };
            let stride = dst_stride as usize;

            let dst_slice = unsafe {
                std::slice::from_raw_parts_mut(
                    dst_ptr as *mut u8,
                    _coeff_len as usize * stride + stride,
                )
            };

            let coeff_slice =
                unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

            $inner(_token, dst_slice, stride, coeff_slice, eob, bitdepth_max);
        }
    };
}

impl_8x8_ffi_wrapper!(
    inv_txfm_add_adst_dct_8x8_8bpc_avx2,
    inv_txfm_add_adst_dct_8x8_8bpc_avx2_inner
);
impl_8x8_ffi_wrapper!(
    inv_txfm_add_dct_adst_8x8_8bpc_avx2,
    inv_txfm_add_dct_adst_8x8_8bpc_avx2_inner
);
impl_8x8_ffi_wrapper!(
    inv_txfm_add_adst_adst_8x8_8bpc_avx2,
    inv_txfm_add_adst_adst_8x8_8bpc_avx2_inner
);
impl_8x8_ffi_wrapper!(
    inv_txfm_add_flipadst_dct_8x8_8bpc_avx2,
    inv_txfm_add_flipadst_dct_8x8_8bpc_avx2_inner
);
impl_8x8_ffi_wrapper!(
    inv_txfm_add_dct_flipadst_8x8_8bpc_avx2,
    inv_txfm_add_dct_flipadst_8x8_8bpc_avx2_inner
);
impl_8x8_ffi_wrapper!(
    inv_txfm_add_flipadst_flipadst_8x8_8bpc_avx2,
    inv_txfm_add_flipadst_flipadst_8x8_8bpc_avx2_inner
);
impl_8x8_ffi_wrapper!(
    inv_txfm_add_adst_flipadst_8x8_8bpc_avx2,
    inv_txfm_add_adst_flipadst_8x8_8bpc_avx2_inner
);
impl_8x8_ffi_wrapper!(
    inv_txfm_add_flipadst_adst_8x8_8bpc_avx2,
    inv_txfm_add_flipadst_adst_8x8_8bpc_avx2_inner
);

// ============================================================================
// V_ADST/H_ADST TRANSFORMS (Identity + ADST combinations)
// ============================================================================

/// Identity transform 4x4 - just pass through values (no transform)
#[inline(always)]
fn identity4_1d_scalar(
    in0: i32,
    in1: i32,
    in2: i32,
    in3: i32,
    _min: i32,
    _max: i32,
) -> (i32, i32, i32, i32) {
    // identity4(x) = x + (x * 1697 + 2048) >> 12 ≈ x * sqrt(2)
    // Matches rav1d_inv_identity4_1d_c in itx_1d.rs (which also ignores min/max)
    let o0 = in0 + ((in0 * 1697 + 2048) >> 12);
    let o1 = in1 + ((in1 * 1697 + 2048) >> 12);
    let o2 = in2 + ((in2 * 1697 + 2048) >> 12);
    let o3 = in3 + ((in3 * 1697 + 2048) >> 12);
    (o0, o1, o2, o3)
}

/// V_ADST 4x4: Identity on rows, ADST on columns
#[cfg(target_arch = "x86_64")]
pub fn inv_txfm_add_v_adst_4x4_8bpc_avx2_inner(
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    coeff: &mut [i16],
    _eob: i32,
    _bitdepth_max: i32,
) {
    let mut c = [[0i32; 4]; 4];
    for y in 0..4 {
        for x in 0..4 {
            c[y][x] = coeff[y + x * 4] as i32;
        }
    }

    // First pass: Identity on rows
    let mut tmp = [[0i32; 4]; 4];
    for y in 0..4 {
        let (o0, o1, o2, o3) = identity4_1d_scalar(
            c[y][0],
            c[y][1],
            c[y][2],
            c[y][3],
            i16::MIN as i32,
            i16::MAX as i32,
        );
        tmp[y][0] = o0;
        tmp[y][1] = o1;
        tmp[y][2] = o2;
        tmp[y][3] = o3;
    }

    // Intermediate clamp (shift=0 for 4x4, clamp to col_clip range)
    let col_clip_min = i16::MIN as i32;
    let col_clip_max = i16::MAX as i32;
    for y in 0..4 {
        for x in 0..4 {
            tmp[y][x] = tmp[y][x].clamp(col_clip_min, col_clip_max);
        }
    }

    // Second pass: ADST on columns
    let mut out = [[0i32; 4]; 4];
    for x in 0..4 {
        let (o0, o1, o2, o3) = adst4_1d_scalar(
            tmp[0][x],
            tmp[1][x],
            tmp[2][x],
            tmp[3][x],
            col_clip_min,
            col_clip_max,
        );
        out[0][x] = o0;
        out[1][x] = o1;
        out[2][x] = o2;
        out[3][x] = o3;
    }

    for y in 0..4 {
        let row_off = dst_base.wrapping_add_signed(y as isize * dst_stride);
        for x in 0..4 {
            let pixel = dst[row_off + x] as i32;
            let val = pixel + ((out[y][x] + 8) >> 4);
            dst[row_off + x] = val.clamp(0, 255) as u8;
        }
    }
    coeff[..16].fill(0);
}

/// H_ADST 4x4: ADST on rows, Identity on columns
#[cfg(target_arch = "x86_64")]
pub fn inv_txfm_add_h_adst_4x4_8bpc_avx2_inner(
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    coeff: &mut [i16],
    _eob: i32,
    _bitdepth_max: i32,
) {
    let mut c = [[0i32; 4]; 4];
    for y in 0..4 {
        for x in 0..4 {
            c[y][x] = coeff[y + x * 4] as i32;
        }
    }

    // First pass: ADST on rows (H_ADST = first=Adst, second=Identity per reference)
    let mut tmp = [[0i32; 4]; 4];
    for y in 0..4 {
        let (o0, o1, o2, o3) = adst4_1d_scalar(
            c[y][0],
            c[y][1],
            c[y][2],
            c[y][3],
            i16::MIN as i32,
            i16::MAX as i32,
        );
        tmp[y][0] = o0;
        tmp[y][1] = o1;
        tmp[y][2] = o2;
        tmp[y][3] = o3;
    }

    // Second pass: Identity on columns
    let mut out = [[0i32; 4]; 4];
    for x in 0..4 {
        let (o0, o1, o2, o3) = identity4_1d_scalar(
            tmp[0][x],
            tmp[1][x],
            tmp[2][x],
            tmp[3][x],
            i16::MIN as i32,
            i16::MAX as i32,
        );
        out[0][x] = o0;
        out[1][x] = o1;
        out[2][x] = o2;
        out[3][x] = o3;
    }

    for y in 0..4 {
        let row_off = dst_base.wrapping_add_signed(y as isize * dst_stride);
        for x in 0..4 {
            let pixel = dst[row_off + x] as i32;
            let val = pixel + ((out[y][x] + 8) >> 4);
            dst[row_off + x] = val.clamp(0, 255) as u8;
        }
    }
    coeff[..16].fill(0);
}

/// V_FLIPADST 4x4: Identity on rows, FlipADST on columns
#[cfg(target_arch = "x86_64")]
pub fn inv_txfm_add_v_flipadst_4x4_8bpc_avx2_inner(
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    coeff: &mut [i16],
    _eob: i32,
    _bitdepth_max: i32,
) {
    let mut c = [[0i32; 4]; 4];
    for y in 0..4 {
        for x in 0..4 {
            c[y][x] = coeff[y + x * 4] as i32;
        }
    }

    let mut tmp = [[0i32; 4]; 4];
    for y in 0..4 {
        let (o0, o1, o2, o3) = identity4_1d_scalar(
            c[y][0],
            c[y][1],
            c[y][2],
            c[y][3],
            i16::MIN as i32,
            i16::MAX as i32,
        );
        tmp[y][0] = o0;
        tmp[y][1] = o1;
        tmp[y][2] = o2;
        tmp[y][3] = o3;
    }

    // Intermediate clamp (shift=0 for 4x4, clamp to col_clip range)
    let col_clip_min = i16::MIN as i32;
    let col_clip_max = i16::MAX as i32;
    for y in 0..4 {
        for x in 0..4 {
            tmp[y][x] = tmp[y][x].clamp(col_clip_min, col_clip_max);
        }
    }

    let mut out = [[0i32; 4]; 4];
    for x in 0..4 {
        let (o0, o1, o2, o3) = flipadst4_1d_scalar(
            tmp[0][x],
            tmp[1][x],
            tmp[2][x],
            tmp[3][x],
            col_clip_min,
            col_clip_max,
        );
        out[0][x] = o0;
        out[1][x] = o1;
        out[2][x] = o2;
        out[3][x] = o3;
    }

    for y in 0..4 {
        let row_off = dst_base.wrapping_add_signed(y as isize * dst_stride);
        for x in 0..4 {
            let pixel = dst[row_off + x] as i32;
            let val = pixel + ((out[y][x] + 8) >> 4);
            dst[row_off + x] = val.clamp(0, 255) as u8;
        }
    }
    coeff[..16].fill(0);
}

/// H_FLIPADST 4x4: Identity on rows, FlipADST on columns
#[cfg(target_arch = "x86_64")]
pub fn inv_txfm_add_h_flipadst_4x4_8bpc_avx2_inner(
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    coeff: &mut [i16],
    _eob: i32,
    _bitdepth_max: i32,
) {
    let mut c = [[0i32; 4]; 4];
    for y in 0..4 {
        for x in 0..4 {
            c[y][x] = coeff[y + x * 4] as i32;
        }
    }

    // First pass: FlipADST on rows (H_FLIPADST = first=FlipAdst, second=Identity per reference)
    let mut tmp = [[0i32; 4]; 4];
    for y in 0..4 {
        let (o0, o1, o2, o3) = flipadst4_1d_scalar(
            c[y][0],
            c[y][1],
            c[y][2],
            c[y][3],
            i16::MIN as i32,
            i16::MAX as i32,
        );
        tmp[y][0] = o0;
        tmp[y][1] = o1;
        tmp[y][2] = o2;
        tmp[y][3] = o3;
    }

    // Second pass: Identity on columns
    let mut out = [[0i32; 4]; 4];
    for x in 0..4 {
        let (o0, o1, o2, o3) = identity4_1d_scalar(
            tmp[0][x],
            tmp[1][x],
            tmp[2][x],
            tmp[3][x],
            i16::MIN as i32,
            i16::MAX as i32,
        );
        out[0][x] = o0;
        out[1][x] = o1;
        out[2][x] = o2;
        out[3][x] = o3;
    }

    for y in 0..4 {
        let row_off = dst_base.wrapping_add_signed(y as isize * dst_stride);
        for x in 0..4 {
            let pixel = dst[row_off + x] as i32;
            let val = pixel + ((out[y][x] + 8) >> 4);
            dst[row_off + x] = val.clamp(0, 255) as u8;
        }
    }
    coeff[..16].fill(0);
}

// FFI wrappers for V/H ADST
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_adst_4x4_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let abs_stride = dst_stride.unsigned_abs();
    let buf_size = 3 * abs_stride + 4;
    let (base, dst_slice) = if dst_stride >= 0 {
        (0usize, unsafe {
            std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_size)
        })
    } else {
        let start = unsafe { (dst_ptr as *mut u8).offset(3 * dst_stride) };
        (3 * abs_stride, unsafe {
            std::slice::from_raw_parts_mut(start, buf_size)
        })
    };
    let coeff_slice = unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, 16) };
    inv_txfm_add_h_adst_4x4_8bpc_avx2_inner(
        dst_slice,
        base,
        dst_stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_adst_identity_4x4_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let abs_stride = dst_stride.unsigned_abs();
    let buf_size = 3 * abs_stride + 4;
    let (base, dst_slice) = if dst_stride >= 0 {
        (0usize, unsafe {
            std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_size)
        })
    } else {
        let start = unsafe { (dst_ptr as *mut u8).offset(3 * dst_stride) };
        (3 * abs_stride, unsafe {
            std::slice::from_raw_parts_mut(start, buf_size)
        })
    };
    let coeff_slice = unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, 16) };
    inv_txfm_add_v_adst_4x4_8bpc_avx2_inner(
        dst_slice,
        base,
        dst_stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_flipadst_4x4_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let abs_stride = dst_stride.unsigned_abs();
    let buf_size = 3 * abs_stride + 4;
    let (base, dst_slice) = if dst_stride >= 0 {
        (0usize, unsafe {
            std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_size)
        })
    } else {
        let start = unsafe { (dst_ptr as *mut u8).offset(3 * dst_stride) };
        (3 * abs_stride, unsafe {
            std::slice::from_raw_parts_mut(start, buf_size)
        })
    };
    let coeff_slice = unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, 16) };
    inv_txfm_add_h_flipadst_4x4_8bpc_avx2_inner(
        dst_slice,
        base,
        dst_stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_flipadst_identity_4x4_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let abs_stride = dst_stride.unsigned_abs();
    let buf_size = 3 * abs_stride + 4;
    let (base, dst_slice) = if dst_stride >= 0 {
        (0usize, unsafe {
            std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_size)
        })
    } else {
        let start = unsafe { (dst_ptr as *mut u8).offset(3 * dst_stride) };
        (3 * abs_stride, unsafe {
            std::slice::from_raw_parts_mut(start, buf_size)
        })
    };
    let coeff_slice = unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, 16) };
    inv_txfm_add_v_flipadst_4x4_8bpc_avx2_inner(
        dst_slice,
        base,
        dst_stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// V_DCT/H_DCT TRANSFORMS (DCT + Identity combinations)
// ============================================================================

/// H_DCT 4x4: DCT on rows, Identity on columns
#[cfg(target_arch = "x86_64")]
pub fn inv_txfm_add_dct_identity_4x4_8bpc_avx2_inner(
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    coeff: &mut [i16],
    _eob: i32,
    _bitdepth_max: i32,
) {
    let mut c = [[0i32; 4]; 4];
    for y in 0..4 {
        for x in 0..4 {
            c[y][x] = coeff[y + x * 4] as i32;
        }
    }

    // First pass: DCT on rows
    let mut tmp = [[0i32; 4]; 4];
    for y in 0..4 {
        let (o0, o1, o2, o3) = dct4_1d_scalar(
            c[y][0],
            c[y][1],
            c[y][2],
            c[y][3],
            i16::MIN as i32,
            i16::MAX as i32,
        );
        tmp[y][0] = o0;
        tmp[y][1] = o1;
        tmp[y][2] = o2;
        tmp[y][3] = o3;
    }

    // Second pass: Identity on columns
    let mut out = [[0i32; 4]; 4];
    for x in 0..4 {
        let (o0, o1, o2, o3) = identity4_1d_scalar(
            tmp[0][x],
            tmp[1][x],
            tmp[2][x],
            tmp[3][x],
            i16::MIN as i32,
            i16::MAX as i32,
        );
        out[0][x] = o0;
        out[1][x] = o1;
        out[2][x] = o2;
        out[3][x] = o3;
    }

    for y in 0..4 {
        let row_off = dst_base.wrapping_add_signed(y as isize * dst_stride);
        for x in 0..4 {
            let pixel = dst[row_off + x] as i32;
            let val = pixel + ((out[y][x] + 8) >> 4);
            dst[row_off + x] = val.clamp(0, 255) as u8;
        }
    }
    coeff[..16].fill(0);
}

/// V_DCT 4x4: Identity on rows, DCT on columns
#[cfg(target_arch = "x86_64")]
pub fn inv_txfm_add_identity_dct_4x4_8bpc_avx2_inner(
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    coeff: &mut [i16],
    _eob: i32,
    _bitdepth_max: i32,
) {
    let mut c = [[0i32; 4]; 4];
    for y in 0..4 {
        for x in 0..4 {
            c[y][x] = coeff[y + x * 4] as i32;
        }
    }

    // First pass: Identity on rows
    let mut tmp = [[0i32; 4]; 4];
    for y in 0..4 {
        let (o0, o1, o2, o3) = identity4_1d_scalar(
            c[y][0],
            c[y][1],
            c[y][2],
            c[y][3],
            i16::MIN as i32,
            i16::MAX as i32,
        );
        tmp[y][0] = o0;
        tmp[y][1] = o1;
        tmp[y][2] = o2;
        tmp[y][3] = o3;
    }

    // Intermediate clamp (shift=0 for 4x4, just clamp to col_clip range)
    let col_clip_min = i16::MIN as i32;
    let col_clip_max = i16::MAX as i32;
    for y in 0..4 {
        for x in 0..4 {
            tmp[y][x] = tmp[y][x].clamp(col_clip_min, col_clip_max);
        }
    }

    // Second pass: DCT on columns
    let mut out = [[0i32; 4]; 4];
    for x in 0..4 {
        let (o0, o1, o2, o3) = dct4_1d_scalar(
            tmp[0][x],
            tmp[1][x],
            tmp[2][x],
            tmp[3][x],
            col_clip_min,
            col_clip_max,
        );
        out[0][x] = o0;
        out[1][x] = o1;
        out[2][x] = o2;
        out[3][x] = o3;
    }

    for y in 0..4 {
        let row_off = dst_base.wrapping_add_signed(y as isize * dst_stride);
        for x in 0..4 {
            let pixel = dst[row_off + x] as i32;
            let val = pixel + ((out[y][x] + 8) >> 4);
            dst[row_off + x] = val.clamp(0, 255) as u8;
        }
    }
    coeff[..16].fill(0);
}

// FFI wrappers for V/H DCT
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_identity_4x4_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let abs_stride = dst_stride.unsigned_abs();
    let buf_size = 3 * abs_stride + 4;
    let (base, dst_slice) = if dst_stride >= 0 {
        (0usize, unsafe {
            std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_size)
        })
    } else {
        let start = unsafe { (dst_ptr as *mut u8).offset(3 * dst_stride) };
        (3 * abs_stride, unsafe {
            std::slice::from_raw_parts_mut(start, buf_size)
        })
    };
    let coeff_slice = unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, 16) };
    inv_txfm_add_dct_identity_4x4_8bpc_avx2_inner(
        dst_slice,
        base,
        dst_stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_dct_4x4_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let abs_stride = dst_stride.unsigned_abs();
    let buf_size = 3 * abs_stride + 4;
    let (base, dst_slice) = if dst_stride >= 0 {
        (0usize, unsafe {
            std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_size)
        })
    } else {
        let start = unsafe { (dst_ptr as *mut u8).offset(3 * dst_stride) };
        (3 * abs_stride, unsafe {
            std::slice::from_raw_parts_mut(start, buf_size)
        })
    };
    let coeff_slice = unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, 16) };
    inv_txfm_add_identity_dct_4x4_8bpc_avx2_inner(
        dst_slice,
        base,
        dst_stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// V/H ADST/DCT 8x8 TRANSFORMS
// ============================================================================

/// Identity transform 8x8
#[inline(always)]
fn identity8_1d_scalar(
    in0: i32,
    in1: i32,
    in2: i32,
    in3: i32,
    in4: i32,
    in5: i32,
    in6: i32,
    in7: i32,
    _min: i32,
    _max: i32,
) -> (i32, i32, i32, i32, i32, i32, i32, i32) {
    // For 8x8 identity: out = in * 2
    (
        in0 * 2,
        in1 * 2,
        in2 * 2,
        in3 * 2,
        in4 * 2,
        in5 * 2,
        in6 * 2,
        in7 * 2,
    )
}

// Use the macro to generate V/H transforms for 8x8 (SIMD col)
impl_8x8_transform_simd_col!(
    inv_txfm_add_identity_adst_8x8_8bpc_avx2_inner,
    identity8_1d_scalar,
    adst8_1d_cols8
);
impl_8x8_transform_simd_col!(
    inv_txfm_add_adst_identity_8x8_8bpc_avx2_inner,
    adst8_1d_scalar,
    identity8_1d_cols8
);
impl_8x8_transform_simd_col!(
    inv_txfm_add_identity_flipadst_8x8_8bpc_avx2_inner,
    identity8_1d_scalar,
    flipadst8_1d_cols8
);
impl_8x8_transform_simd_col!(
    inv_txfm_add_flipadst_identity_8x8_8bpc_avx2_inner,
    flipadst8_1d_scalar,
    identity8_1d_cols8
);
impl_8x8_transform_simd_col!(
    inv_txfm_add_identity_dct_8x8_8bpc_avx2_inner,
    identity8_1d_scalar,
    dct8_1d_cols8
);
impl_8x8_transform_simd_col!(
    inv_txfm_add_dct_identity_8x8_8bpc_avx2_inner,
    dct8_1d_scalar,
    identity8_1d_cols8
);

// FFI wrappers
impl_8x8_ffi_wrapper!(
    inv_txfm_add_identity_adst_8x8_8bpc_avx2,
    inv_txfm_add_identity_adst_8x8_8bpc_avx2_inner
);
impl_8x8_ffi_wrapper!(
    inv_txfm_add_adst_identity_8x8_8bpc_avx2,
    inv_txfm_add_adst_identity_8x8_8bpc_avx2_inner
);
impl_8x8_ffi_wrapper!(
    inv_txfm_add_identity_flipadst_8x8_8bpc_avx2,
    inv_txfm_add_identity_flipadst_8x8_8bpc_avx2_inner
);
impl_8x8_ffi_wrapper!(
    inv_txfm_add_flipadst_identity_8x8_8bpc_avx2,
    inv_txfm_add_flipadst_identity_8x8_8bpc_avx2_inner
);
impl_8x8_ffi_wrapper!(
    inv_txfm_add_identity_dct_8x8_8bpc_avx2,
    inv_txfm_add_identity_dct_8x8_8bpc_avx2_inner
);
impl_8x8_ffi_wrapper!(
    inv_txfm_add_dct_identity_8x8_8bpc_avx2,
    inv_txfm_add_dct_identity_8x8_8bpc_avx2_inner
);

