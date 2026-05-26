// ============================================================================
// RECTANGULAR DCT TRANSFORMS 16bpc
// ============================================================================

/// 4x8 DCT_DCT for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_4x8_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize,
    coeff: &mut [i32],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let stride_u16 = dst_stride / 2;
    let row_clip_min = (!bitdepth_max) << 7;
    let row_clip_max = !row_clip_min;
    let col_clip_min = (!bitdepth_max) << 5;
    let col_clip_max = !col_clip_min;
    let mut tmp = [0i32; 32];

    // is_rect2 = true for 4x8, so apply sqrt(2) scaling
    let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

    // Row transform (4 elements each, 8 rows)
    for y in 0..8 {
        let mut scratch = [0i32; 4];
        for x in 0..4 {
            scratch[x] = rect2_scale(coeff[y + x * 8] as i32);
        }
        dct4_1d(&mut scratch[..4], 1, row_clip_min, row_clip_max);
        for x in 0..4 {
            tmp[y * 4 + x] = iclip(scratch[x], col_clip_min, col_clip_max);
        }
    }

    // Column transform
    for x in 0..4 {
        dct8_1d(&mut tmp[x..], 4, col_clip_min, col_clip_max);
    }

    // Add to destination
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi32(bitdepth_max);

    for y in 0..8 {
        let dst_off = y * stride_u16;

        // Load destination (4 u16 = 8 bytes)
        let d = loadi64!(zerocopy::IntoBytes::as_bytes(&dst[dst_off..dst_off + 4]));
        let d32 = _mm_unpacklo_epi16(d, zero);

        // Load and scale coefficients
        let c = _mm_set_epi32(
            (tmp[y * 4 + 3] + 8) >> 4,
            (tmp[y * 4 + 2] + 8) >> 4,
            (tmp[y * 4 + 1] + 8) >> 4,
            (tmp[y * 4 + 0] + 8) >> 4,
        );

        let sum = _mm_add_epi32(d32, c);
        let clamped = _mm_max_epi32(_mm_min_epi32(sum, max_val), zero);
        let packed = _mm_packus_epi32(clamped, clamped);

        storei64!(
            zerocopy::IntoBytes::as_mut_bytes(&mut dst[dst_off..dst_off + 4]),
            packed
        );
    }

    // Clear coefficients
    coeff[..32].fill(0);
}

/// FFI wrapper for 4x8 DCT_DCT 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_4x8_16bpc_avx2(
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
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_4x8_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 8x4 DCT_DCT for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_8x4_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize,
    coeff: &mut [i32],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let stride_u16 = dst_stride / 2;
    let row_clip_min = (!bitdepth_max) << 7;
    let row_clip_max = !row_clip_min;
    let col_clip_min = (!bitdepth_max) << 5;
    let col_clip_max = !col_clip_min;
    let mut tmp = [0i32; 32];

    // is_rect2 = true for 8x4
    let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

    // Row transform (8 elements each, 4 rows)
    for y in 0..4 {
        let mut scratch = [0i32; 8];
        for x in 0..8 {
            scratch[x] = rect2_scale(coeff[y + x * 4] as i32);
        }
        dct8_1d(&mut scratch[..8], 1, row_clip_min, row_clip_max);
        for x in 0..8 {
            tmp[y * 8 + x] = iclip(scratch[x], col_clip_min, col_clip_max);
        }
    }

    // Column transform
    for x in 0..8 {
        dct4_1d(&mut tmp[x..], 8, col_clip_min, col_clip_max);
    }

    // Add to destination
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi32(bitdepth_max);

    for y in 0..4 {
        let dst_off = y * stride_u16;

        // Load destination (8 u16 = 16 bytes)
        let d = loadu_128!(<&[u16; 8]>::try_from(&dst[dst_off..dst_off + 8]).unwrap());
        let d_lo = _mm_unpacklo_epi16(d, zero);
        let d_hi = _mm_unpackhi_epi16(d, zero);

        // Load and scale coefficients
        let c_lo = _mm_set_epi32(
            (tmp[y * 8 + 3] + 8) >> 4,
            (tmp[y * 8 + 2] + 8) >> 4,
            (tmp[y * 8 + 1] + 8) >> 4,
            (tmp[y * 8 + 0] + 8) >> 4,
        );
        let c_hi = _mm_set_epi32(
            (tmp[y * 8 + 7] + 8) >> 4,
            (tmp[y * 8 + 6] + 8) >> 4,
            (tmp[y * 8 + 5] + 8) >> 4,
            (tmp[y * 8 + 4] + 8) >> 4,
        );

        let sum_lo = _mm_add_epi32(d_lo, c_lo);
        let sum_hi = _mm_add_epi32(d_hi, c_hi);
        let clamped_lo = _mm_max_epi32(_mm_min_epi32(sum_lo, max_val), zero);
        let clamped_hi = _mm_max_epi32(_mm_min_epi32(sum_hi, max_val), zero);
        let packed = _mm_packus_epi32(clamped_lo, clamped_hi);

        storeu_128!(
            <&mut [u16; 8]>::try_from(&mut dst[dst_off..dst_off + 8]).unwrap(),
            packed
        );
    }

    // Clear coefficients
    coeff[..32].fill(0);
}

/// FFI wrapper for 8x4 DCT_DCT 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_8x4_16bpc_avx2(
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
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_8x4_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 8x16 DCT_DCT for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_8x16_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize,
    coeff: &mut [i32],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let stride_u16 = dst_stride / 2;
    let row_clip_min = (!bitdepth_max) << 7;
    let row_clip_max = !row_clip_min;
    let col_clip_min = (!bitdepth_max) << 5;
    let col_clip_max = !col_clip_min;
    let mut tmp = [0i32; 128];

    // is_rect2 = true for 8x16
    let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

    // Row transform with shift=1
    let rnd = 1;
    let shift = 1;

    for y in 0..16 {
        let mut scratch = [0i32; 8];
        for x in 0..8 {
            scratch[x] = rect2_scale(coeff[y + x * 16] as i32);
        }
        dct8_1d(&mut scratch[..8], 1, row_clip_min, row_clip_max);
        for x in 0..8 {
            tmp[y * 8 + x] = iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
        }
    }

    // Column transform: SIMD across 8 columns, 16 rows
    {
        let min_v = _mm256_set1_epi32(col_clip_min);
        let max_v = _mm256_set1_epi32(col_clip_max);
        let mut v = [_mm256_setzero_si256(); 16];
        for i in 0..16 {
            v[i] = loadu_256!(&tmp[i * 8..i * 8 + 8], [i32; 8]);
        }
        dct16_1d_cols8(_token, &mut v, min_v, max_v);
        for i in 0..16 {
            storeu_256!(&mut tmp[i * 8..i * 8 + 8], [i32; 8], v[i]);
        }
    }

    // Add to destination
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi32(bitdepth_max);

    for y in 0..16 {
        let dst_off = y * stride_u16;

        // Load destination (8 u16)
        let d = loadu_128!(<&[u16; 8]>::try_from(&dst[dst_off..dst_off + 8]).unwrap());
        let d_lo = _mm_unpacklo_epi16(d, zero);
        let d_hi = _mm_unpackhi_epi16(d, zero);

        let c_lo = _mm_set_epi32(
            (tmp[y * 8 + 3] + 8) >> 4,
            (tmp[y * 8 + 2] + 8) >> 4,
            (tmp[y * 8 + 1] + 8) >> 4,
            (tmp[y * 8 + 0] + 8) >> 4,
        );
        let c_hi = _mm_set_epi32(
            (tmp[y * 8 + 7] + 8) >> 4,
            (tmp[y * 8 + 6] + 8) >> 4,
            (tmp[y * 8 + 5] + 8) >> 4,
            (tmp[y * 8 + 4] + 8) >> 4,
        );

        let sum_lo = _mm_add_epi32(d_lo, c_lo);
        let sum_hi = _mm_add_epi32(d_hi, c_hi);
        let clamped_lo = _mm_max_epi32(_mm_min_epi32(sum_lo, max_val), zero);
        let clamped_hi = _mm_max_epi32(_mm_min_epi32(sum_hi, max_val), zero);
        let packed = _mm_packus_epi32(clamped_lo, clamped_hi);

        storeu_128!(
            <&mut [u16; 8]>::try_from(&mut dst[dst_off..dst_off + 8]).unwrap(),
            packed
        );
    }

    // Clear coefficients
    coeff[..128].fill(0);
}

/// FFI wrapper for 8x16 DCT_DCT 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_8x16_16bpc_avx2(
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
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_8x16_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 16x8 DCT_DCT for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_16x8_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize,
    coeff: &mut [i32],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let stride_u16 = dst_stride / 2;
    let row_clip_min = (!bitdepth_max) << 7;
    let row_clip_max = !row_clip_min;
    let col_clip_min = (!bitdepth_max) << 5;
    let col_clip_max = !col_clip_min;
    let mut tmp = [0i32; 128];

    // is_rect2 = true for 16x8
    let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

    // Row transform with shift=1
    let rnd = 1;
    let shift = 1;

    for y in 0..8 {
        let mut scratch = [0i32; 16];
        for x in 0..16 {
            scratch[x] = rect2_scale(coeff[y + x * 8] as i32);
        }
        dct16_1d(&mut scratch[..16], 1, row_clip_min, row_clip_max);
        for x in 0..16 {
            tmp[y * 16 + x] = iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
        }
    }

    // Column transform: SIMD across 16 columns (8 rows). AVX-512 does both
    // 16-col chunks at once; AVX2 falls back to 2 chunks of 8.
    if let Some(t512) = crate::src::cpu::summon_avx512() {
        dct8_cols_avx512(t512, &mut tmp, 16, 8, col_clip_min, col_clip_max);
    } else {
        let min_v = _mm256_set1_epi32(col_clip_min);
        let max_v = _mm256_set1_epi32(col_clip_max);
        for cx_chunk in 0..2 {
            let cx = cx_chunk * 8;
            let mut v = [_mm256_setzero_si256(); 8];
            for i in 0..8 {
                v[i] = loadu_256!(&tmp[i * 16 + cx..i * 16 + cx + 8], [i32; 8]);
            }
            dct8_1d_cols8(_token, &mut v, min_v, max_v);
            for i in 0..8 {
                storeu_256!(&mut tmp[i * 16 + cx..i * 16 + cx + 8], [i32; 8], v[i]);
            }
        }
    }

    // Add to destination
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi32(bitdepth_max);
    let rnd_final = _mm256_set1_epi32(8);

    for y in 0..8 {
        let dst_off = y * stride_u16;

        // Load destination (16 u16 = 32 bytes)
        let d = loadu_256!(<&[u16; 16]>::try_from(&dst[dst_off..dst_off + 16]).unwrap());
        let d_lo = _mm256_unpacklo_epi16(d, _mm256_setzero_si256());
        let d_hi = _mm256_unpackhi_epi16(d, _mm256_setzero_si256());
        let d_0_4 = _mm256_permute2x128_si256(d_lo, d_hi, 0x20);
        let d_4_8 = _mm256_permute2x128_si256(d_lo, d_hi, 0x31);

        let c0 = _mm256_set_epi32(
            tmp[y * 16 + 7],
            tmp[y * 16 + 6],
            tmp[y * 16 + 5],
            tmp[y * 16 + 4],
            tmp[y * 16 + 3],
            tmp[y * 16 + 2],
            tmp[y * 16 + 1],
            tmp[y * 16 + 0],
        );
        let c1 = _mm256_set_epi32(
            tmp[y * 16 + 15],
            tmp[y * 16 + 14],
            tmp[y * 16 + 13],
            tmp[y * 16 + 12],
            tmp[y * 16 + 11],
            tmp[y * 16 + 10],
            tmp[y * 16 + 9],
            tmp[y * 16 + 8],
        );

        let c0_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c0, rnd_final));
        let c1_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c1, rnd_final));

        let sum0 = _mm256_add_epi32(d_0_4, c0_scaled);
        let sum1 = _mm256_add_epi32(d_4_8, c1_scaled);

        let clamped0 = _mm256_max_epi32(_mm256_min_epi32(sum0, max_val), zero);
        let clamped1 = _mm256_max_epi32(_mm256_min_epi32(sum1, max_val), zero);

        let packed = _mm256_packus_epi32(clamped0, clamped1);
        let packed = _mm256_permute4x64_epi64(packed, 0b11_01_10_00);
        storeu_256!(
            <&mut [u16; 16]>::try_from(&mut dst[dst_off..dst_off + 16]).unwrap(),
            packed
        );
    }

    // Clear coefficients
    coeff[..128].fill(0);
}

/// FFI wrapper for 16x8 DCT_DCT 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_16x8_16bpc_avx2(
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
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_16x8_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 4x16 DCT_DCT for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_4x16_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize,
    coeff: &mut [i32],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let stride_u16 = dst_stride / 2;
    let row_clip_min = (!bitdepth_max) << 7;
    let row_clip_max = !row_clip_min;
    let col_clip_min = (!bitdepth_max) << 5;
    let col_clip_max = !col_clip_min;
    let mut tmp = [0i32; 64];

    // is_rect2 = false for 4x16 (4:1 ratio), no rect2_scale
    // Row transform with shift=1 (4x16 intermediate shift)
    let rnd = 1;
    let shift = 1;

    for y in 0..16 {
        let mut scratch = [0i32; 4];
        for x in 0..4 {
            scratch[x] = coeff[y + x * 16] as i32;
        }
        dct4_1d(&mut scratch[..4], 1, row_clip_min, row_clip_max);
        for x in 0..4 {
            tmp[y * 4 + x] = iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
        }
    }

    // Column transform
    for x in 0..4 {
        dct16_1d(&mut tmp[x..], 4, col_clip_min, col_clip_max);
    }

    // Add to destination
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi32(bitdepth_max);

    for y in 0..16 {
        let dst_off = y * stride_u16;

        let d = loadi64!(zerocopy::IntoBytes::as_bytes(&dst[dst_off..dst_off + 4]));
        let d32 = _mm_unpacklo_epi16(d, zero);

        let c = _mm_set_epi32(
            (tmp[y * 4 + 3] + 8) >> 4,
            (tmp[y * 4 + 2] + 8) >> 4,
            (tmp[y * 4 + 1] + 8) >> 4,
            (tmp[y * 4 + 0] + 8) >> 4,
        );

        let sum = _mm_add_epi32(d32, c);
        let clamped = _mm_max_epi32(_mm_min_epi32(sum, max_val), zero);
        let packed = _mm_packus_epi32(clamped, clamped);

        storei64!(
            zerocopy::IntoBytes::as_mut_bytes(&mut dst[dst_off..dst_off + 4]),
            packed
        );
    }

    // Clear coefficients
    coeff[..64].fill(0);
}

/// FFI wrapper for 4x16 DCT_DCT 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_4x16_16bpc_avx2(
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
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_4x16_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 16x4 DCT_DCT for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_16x4_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize,
    coeff: &mut [i32],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let stride_u16 = dst_stride / 2;
    let row_clip_min = (!bitdepth_max) << 7;
    let row_clip_max = !row_clip_min;
    let col_clip_min = (!bitdepth_max) << 5;
    let col_clip_max = !col_clip_min;
    let mut tmp = [0i32; 64];

    // is_rect2 = false for 16x4 (4:1 ratio), no rect2_scale
    // Row transform with shift=1 (16x4 intermediate shift)
    let rnd = 1;
    let shift = 1;

    for y in 0..4 {
        let mut scratch = [0i32; 16];
        for x in 0..16 {
            scratch[x] = coeff[y + x * 4] as i32;
        }
        dct16_1d(&mut scratch[..16], 1, row_clip_min, row_clip_max);
        for x in 0..16 {
            tmp[y * 16 + x] = iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
        }
    }

    // Column transform
    for x in 0..16 {
        dct4_1d(&mut tmp[x..], 16, col_clip_min, col_clip_max);
    }

    // Add to destination
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi32(bitdepth_max);
    let rnd_final = _mm256_set1_epi32(8);

    for y in 0..4 {
        let dst_off = y * stride_u16;

        let d = loadu_256!(<&[u16; 16]>::try_from(&dst[dst_off..dst_off + 16]).unwrap());
        let d_lo = _mm256_unpacklo_epi16(d, _mm256_setzero_si256());
        let d_hi = _mm256_unpackhi_epi16(d, _mm256_setzero_si256());
        let d_0_4 = _mm256_permute2x128_si256(d_lo, d_hi, 0x20);
        let d_4_8 = _mm256_permute2x128_si256(d_lo, d_hi, 0x31);

        let c0 = _mm256_set_epi32(
            tmp[y * 16 + 7],
            tmp[y * 16 + 6],
            tmp[y * 16 + 5],
            tmp[y * 16 + 4],
            tmp[y * 16 + 3],
            tmp[y * 16 + 2],
            tmp[y * 16 + 1],
            tmp[y * 16 + 0],
        );
        let c1 = _mm256_set_epi32(
            tmp[y * 16 + 15],
            tmp[y * 16 + 14],
            tmp[y * 16 + 13],
            tmp[y * 16 + 12],
            tmp[y * 16 + 11],
            tmp[y * 16 + 10],
            tmp[y * 16 + 9],
            tmp[y * 16 + 8],
        );

        let c0_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c0, rnd_final));
        let c1_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c1, rnd_final));

        let sum0 = _mm256_add_epi32(d_0_4, c0_scaled);
        let sum1 = _mm256_add_epi32(d_4_8, c1_scaled);

        let clamped0 = _mm256_max_epi32(_mm256_min_epi32(sum0, max_val), zero);
        let clamped1 = _mm256_max_epi32(_mm256_min_epi32(sum1, max_val), zero);

        let packed = _mm256_packus_epi32(clamped0, clamped1);
        let packed = _mm256_permute4x64_epi64(packed, 0b11_01_10_00);
        storeu_256!(
            <&mut [u16; 16]>::try_from(&mut dst[dst_off..dst_off + 16]).unwrap(),
            packed
        );
    }

    // Clear coefficients
    coeff[..64].fill(0);
}

/// FFI wrapper for 16x4 DCT_DCT 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_16x4_16bpc_avx2(
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
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_16x4_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 16x32 DCT_DCT for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_16x32_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize,
    coeff: &mut [i32],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let stride_u16 = dst_stride / 2;
    let row_clip_min = (!bitdepth_max) << 7;
    let row_clip_max = !row_clip_min;
    let col_clip_min = (!bitdepth_max) << 5;
    let col_clip_max = !col_clip_min;
    let mut tmp = [0i32; 512];

    // is_rect2 = true for 16x32
    let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

    // Row transform with shift=1
    let rnd = 1;
    let shift = 1;

    for y in 0..32 {
        let mut scratch = [0i32; 16];
        for x in 0..16 {
            scratch[x] = rect2_scale(coeff[y + x * 32] as i32);
        }
        dct16_1d(&mut scratch[..16], 1, row_clip_min, row_clip_max);
        for x in 0..16 {
            tmp[y * 16 + x] = iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
        }
    }

    // Column transform: SIMD across 16 columns, 32 rows
    if let Some(t512) = crate::src::cpu::summon_avx512() {
        dct32_cols_avx512(t512, &mut tmp, 16, 32, col_clip_min, col_clip_max);
    } else {
        let min_v = _mm256_set1_epi32(col_clip_min);
        let max_v = _mm256_set1_epi32(col_clip_max);
        for cx_chunk in 0..2 {
            let cx = cx_chunk * 8;
            let mut v = [_mm256_setzero_si256(); 32];
            for i in 0..32 {
                v[i] = loadu_256!(&tmp[i * 16 + cx..i * 16 + cx + 8], [i32; 8]);
            }
            dct32_1d_cols8(_token, &mut v, min_v, max_v);
            for i in 0..32 {
                storeu_256!(&mut tmp[i * 16 + cx..i * 16 + cx + 8], [i32; 8], v[i]);
            }
        }
    }

    // Add to destination
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi32(bitdepth_max);
    let rnd_final = _mm256_set1_epi32(8); // (+ 8) >> 4

    for y in 0..32 {
        let dst_off = y * stride_u16;

        let d = loadu_256!(<&[u16; 16]>::try_from(&dst[dst_off..dst_off + 16]).unwrap());
        let d_lo = _mm256_unpacklo_epi16(d, _mm256_setzero_si256());
        let d_hi = _mm256_unpackhi_epi16(d, _mm256_setzero_si256());
        let d_0_4 = _mm256_permute2x128_si256(d_lo, d_hi, 0x20);
        let d_4_8 = _mm256_permute2x128_si256(d_lo, d_hi, 0x31);

        let c0 = _mm256_set_epi32(
            tmp[y * 16 + 7],
            tmp[y * 16 + 6],
            tmp[y * 16 + 5],
            tmp[y * 16 + 4],
            tmp[y * 16 + 3],
            tmp[y * 16 + 2],
            tmp[y * 16 + 1],
            tmp[y * 16 + 0],
        );
        let c1 = _mm256_set_epi32(
            tmp[y * 16 + 15],
            tmp[y * 16 + 14],
            tmp[y * 16 + 13],
            tmp[y * 16 + 12],
            tmp[y * 16 + 11],
            tmp[y * 16 + 10],
            tmp[y * 16 + 9],
            tmp[y * 16 + 8],
        );

        let c0_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c0, rnd_final));
        let c1_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c1, rnd_final));

        let sum0 = _mm256_add_epi32(d_0_4, c0_scaled);
        let sum1 = _mm256_add_epi32(d_4_8, c1_scaled);

        let clamped0 = _mm256_max_epi32(_mm256_min_epi32(sum0, max_val), zero);
        let clamped1 = _mm256_max_epi32(_mm256_min_epi32(sum1, max_val), zero);

        let packed = _mm256_packus_epi32(clamped0, clamped1);
        let packed = _mm256_permute4x64_epi64(packed, 0b11_01_10_00);
        storeu_256!(
            <&mut [u16; 16]>::try_from(&mut dst[dst_off..dst_off + 16]).unwrap(),
            packed
        );
    }

    // Clear coefficients
    coeff[..512].fill(0);
}

/// FFI wrapper for 16x32 DCT_DCT 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_16x32_16bpc_avx2(
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
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_16x32_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 32x16 DCT_DCT for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_32x16_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize,
    coeff: &mut [i32],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let stride_u16 = dst_stride / 2;
    let row_clip_min = (!bitdepth_max) << 7;
    let row_clip_max = !row_clip_min;
    let col_clip_min = (!bitdepth_max) << 5;
    let col_clip_max = !col_clip_min;
    let mut tmp = [0i32; 512];

    // is_rect2 = true for 32x16
    let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

    // Row transform with shift=1
    let rnd = 1;
    let shift = 1;

    for y in 0..16 {
        let mut scratch = [0i32; 32];
        for x in 0..32 {
            scratch[x] = rect2_scale(coeff[y + x * 16] as i32);
        }
        dct32_1d(&mut scratch[..32], 1, row_clip_min, row_clip_max);
        for x in 0..32 {
            tmp[y * 32 + x] = iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
        }
    }

    // Column transform: SIMD across 32 columns, 16 rows
    if let Some(t512) = crate::src::cpu::summon_avx512() {
        dct16_cols_avx512(t512, &mut tmp, 32, 16, col_clip_min, col_clip_max);
    } else {
        let min_v = _mm256_set1_epi32(col_clip_min);
        let max_v = _mm256_set1_epi32(col_clip_max);
        for cx_chunk in 0..4 {
            let cx = cx_chunk * 8;
            let mut v = [_mm256_setzero_si256(); 16];
            for i in 0..16 {
                v[i] = loadu_256!(&tmp[i * 32 + cx..i * 32 + cx + 8], [i32; 8]);
            }
            dct16_1d_cols8(_token, &mut v, min_v, max_v);
            for i in 0..16 {
                storeu_256!(&mut tmp[i * 32 + cx..i * 32 + cx + 8], [i32; 8], v[i]);
            }
        }
    }

    // Add to destination
    #[cfg(target_arch = "x86_64")]
    if let Some(t512) = crate::src::cpu::summon_avx512() {
        add_to_dst_16bpc_avx512(t512, &mut *dst, stride_u16, &tmp, 32, 32, 16, bitdepth_max);
        coeff[..512].fill(0);
        return;
    }

    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi32(bitdepth_max);
    let rnd_final = _mm256_set1_epi32(8); // (+ 8) >> 4

    for y in 0..16 {
        let dst_off = y * stride_u16;

        // Process 32 pixels in four 8-pixel chunks
        for chunk in 0..4 {
            let x_base = chunk * 8;
            let dst_chunk_off = dst_off + x_base;

            let d =
                loadu_128!(<&[u16; 8]>::try_from(&dst[dst_chunk_off..dst_chunk_off + 8]).unwrap());
            let d_lo = _mm_unpacklo_epi16(d, _mm_setzero_si128());
            let d_hi = _mm_unpackhi_epi16(d, _mm_setzero_si128());

            let c_lo = _mm_set_epi32(
                tmp[y * 32 + x_base + 3],
                tmp[y * 32 + x_base + 2],
                tmp[y * 32 + x_base + 1],
                tmp[y * 32 + x_base + 0],
            );
            let c_hi = _mm_set_epi32(
                tmp[y * 32 + x_base + 7],
                tmp[y * 32 + x_base + 6],
                tmp[y * 32 + x_base + 5],
                tmp[y * 32 + x_base + 4],
            );

            let d32 = _mm256_set_m128i(d_hi, d_lo);
            let c32 = _mm256_set_m128i(c_hi, c_lo);

            let c_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c32, rnd_final));
            let sum = _mm256_add_epi32(d32, c_scaled);
            let clamped = _mm256_max_epi32(_mm256_min_epi32(sum, max_val), zero);

            let lo = _mm256_castsi256_si128(clamped);
            let hi = _mm256_extracti128_si256(clamped, 1);
            let packed = _mm_packus_epi32(lo, hi);
            storeu_128!(
                <&mut [u16; 8]>::try_from(&mut dst[dst_chunk_off..dst_chunk_off + 8]).unwrap(),
                packed
            );
        }
    }

    // Clear coefficients
    coeff[..512].fill(0);
}

/// FFI wrapper for 32x16 DCT_DCT 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_32x16_16bpc_avx2(
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
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_32x16_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 8x32 DCT_DCT for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_8x32_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize,
    coeff: &mut [i32],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let stride_u16 = dst_stride / 2;
    let row_clip_min = (!bitdepth_max) << 7;
    let row_clip_max = !row_clip_min;
    let col_clip_min = (!bitdepth_max) << 5;
    let col_clip_max = !col_clip_min;
    let mut tmp = [0i32; 256];

    // is_rect2 = false for 8x32 (4:1 ratio), no rect2_scale
    // Row transform with shift=2 (8x32 intermediate shift)
    let rnd = 2;
    let shift = 2;

    for y in 0..32 {
        let mut scratch = [0i32; 8];
        for x in 0..8 {
            scratch[x] = coeff[y + x * 32] as i32;
        }
        dct8_1d(&mut scratch[..8], 1, row_clip_min, row_clip_max);
        for x in 0..8 {
            tmp[y * 8 + x] = iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
        }
    }

    // Column transform: SIMD across 8 columns, 32 rows
    {
        let min_v = _mm256_set1_epi32(col_clip_min);
        let max_v = _mm256_set1_epi32(col_clip_max);
        let mut v = [_mm256_setzero_si256(); 32];
        for i in 0..32 {
            v[i] = loadu_256!(&tmp[i * 8..i * 8 + 8], [i32; 8]);
        }
        dct32_1d_cols8(_token, &mut v, min_v, max_v);
        for i in 0..32 {
            storeu_256!(&mut tmp[i * 8..i * 8 + 8], [i32; 8], v[i]);
        }
    }

    // Add to destination
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi32(bitdepth_max);

    for y in 0..32 {
        let dst_off = y * stride_u16;

        let d = loadu_128!(<&[u16; 8]>::try_from(&dst[dst_off..dst_off + 8]).unwrap());
        let d_lo = _mm_unpacklo_epi16(d, zero);
        let d_hi = _mm_unpackhi_epi16(d, zero);

        // (+ 8) >> 4 for 8x32
        let c_lo = _mm_set_epi32(
            (tmp[y * 8 + 3] + 8) >> 4,
            (tmp[y * 8 + 2] + 8) >> 4,
            (tmp[y * 8 + 1] + 8) >> 4,
            (tmp[y * 8 + 0] + 8) >> 4,
        );
        let c_hi = _mm_set_epi32(
            (tmp[y * 8 + 7] + 8) >> 4,
            (tmp[y * 8 + 6] + 8) >> 4,
            (tmp[y * 8 + 5] + 8) >> 4,
            (tmp[y * 8 + 4] + 8) >> 4,
        );

        let sum_lo = _mm_add_epi32(d_lo, c_lo);
        let sum_hi = _mm_add_epi32(d_hi, c_hi);
        let clamped_lo = _mm_max_epi32(_mm_min_epi32(sum_lo, max_val), zero);
        let clamped_hi = _mm_max_epi32(_mm_min_epi32(sum_hi, max_val), zero);
        let packed = _mm_packus_epi32(clamped_lo, clamped_hi);

        storeu_128!(
            <&mut [u16; 8]>::try_from(&mut dst[dst_off..dst_off + 8]).unwrap(),
            packed
        );
    }

    // Clear coefficients
    coeff[..256].fill(0);
}

/// FFI wrapper for 8x32 DCT_DCT 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_8x32_16bpc_avx2(
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
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_8x32_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 32x8 DCT_DCT for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_32x8_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize,
    coeff: &mut [i32],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let stride_u16 = dst_stride / 2;
    let row_clip_min = (!bitdepth_max) << 7;
    let row_clip_max = !row_clip_min;
    let col_clip_min = (!bitdepth_max) << 5;
    let col_clip_max = !col_clip_min;
    let mut tmp = [0i32; 256];

    // is_rect2 = false for 32x8 (4:1 ratio), no rect2_scale
    // Row transform with shift=2 (32x8 intermediate shift)
    let rnd = 2;
    let shift = 2;

    for y in 0..8 {
        let mut scratch = [0i32; 32];
        for x in 0..32 {
            scratch[x] = coeff[y + x * 8] as i32;
        }
        dct32_1d(&mut scratch[..32], 1, row_clip_min, row_clip_max);
        for x in 0..32 {
            tmp[y * 32 + x] = iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
        }
    }

    // Column transform: SIMD across 32 columns (8 rows). AVX-512 does 16-col
    // chunks (2 chunks); AVX2 falls back to 4 chunks of 8.
    if let Some(t512) = crate::src::cpu::summon_avx512() {
        dct8_cols_avx512(t512, &mut tmp, 32, 8, col_clip_min, col_clip_max);
    } else {
        let min_v = _mm256_set1_epi32(col_clip_min);
        let max_v = _mm256_set1_epi32(col_clip_max);
        for cx_chunk in 0..4 {
            let cx = cx_chunk * 8;
            let mut v = [_mm256_setzero_si256(); 8];
            for i in 0..8 {
                v[i] = loadu_256!(&tmp[i * 32 + cx..i * 32 + cx + 8], [i32; 8]);
            }
            dct8_1d_cols8(_token, &mut v, min_v, max_v);
            for i in 0..8 {
                storeu_256!(&mut tmp[i * 32 + cx..i * 32 + cx + 8], [i32; 8], v[i]);
            }
        }
    }

    // Add to destination
    #[cfg(target_arch = "x86_64")]
    if let Some(t512) = crate::src::cpu::summon_avx512() {
        add_to_dst_16bpc_avx512(t512, &mut *dst, stride_u16, &tmp, 32, 32, 8, bitdepth_max);
        coeff[..256].fill(0);
        return;
    }

    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi32(bitdepth_max);
    let rnd_final = _mm256_set1_epi32(8); // (+ 8) >> 4

    for y in 0..8 {
        let dst_off = y * stride_u16;

        // Process 32 pixels in four 8-pixel chunks
        for chunk in 0..4 {
            let x_base = chunk * 8;
            let dst_chunk_off = dst_off + x_base;

            let d =
                loadu_128!(<&[u16; 8]>::try_from(&dst[dst_chunk_off..dst_chunk_off + 8]).unwrap());
            let d_lo = _mm_unpacklo_epi16(d, _mm_setzero_si128());
            let d_hi = _mm_unpackhi_epi16(d, _mm_setzero_si128());

            let c_lo = _mm_set_epi32(
                tmp[y * 32 + x_base + 3],
                tmp[y * 32 + x_base + 2],
                tmp[y * 32 + x_base + 1],
                tmp[y * 32 + x_base + 0],
            );
            let c_hi = _mm_set_epi32(
                tmp[y * 32 + x_base + 7],
                tmp[y * 32 + x_base + 6],
                tmp[y * 32 + x_base + 5],
                tmp[y * 32 + x_base + 4],
            );

            let d32 = _mm256_set_m128i(d_hi, d_lo);
            let c32 = _mm256_set_m128i(c_hi, c_lo);

            let c_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c32, rnd_final));
            let sum = _mm256_add_epi32(d32, c_scaled);
            let clamped = _mm256_max_epi32(_mm256_min_epi32(sum, max_val), zero);

            let lo = _mm256_castsi256_si128(clamped);
            let hi = _mm256_extracti128_si256(clamped, 1);
            let packed = _mm_packus_epi32(lo, hi);
            storeu_128!(
                <&mut [u16; 8]>::try_from(&mut dst[dst_chunk_off..dst_chunk_off + 8]).unwrap(),
                packed
            );
        }
    }

    // Clear coefficients
    coeff[..256].fill(0);
}

/// FFI wrapper for 32x8 DCT_DCT 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_32x8_16bpc_avx2(
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
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_32x8_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 32x64 DCT_DCT for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_32x64_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize,
    coeff: &mut [i32],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let stride_u16 = dst_stride / 2;
    let row_clip_min = (!bitdepth_max) << 7;
    let row_clip_max = !row_clip_min;
    let col_clip_min = (!bitdepth_max) << 5;
    let col_clip_max = !col_clip_min;
    let mut tmp = [0i32; 2048];

    // is_rect2 = true for 32x64
    let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

    // Row transform with shift=1 (32x64 intermediate shift)
    let rnd = 1;
    let shift = 1;

    // Row transform - only first 32 rows have stored coefficients (high-freq zeroing)
    // Coeff is column-major with stride 32
    for y in 0..32 {
        let mut scratch = [0i32; 32];
        for x in 0..32 {
            scratch[x] = rect2_scale(coeff[y + x * 32] as i32);
        }
        dct32_1d(&mut scratch[..32], 1, row_clip_min, row_clip_max);
        for x in 0..32 {
            tmp[y * 32 + x] = iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
        }
    }
    // Zero-pad rows 32..63
    for y in 32..64 {
        for x in 0..32 {
            tmp[y * 32 + x] = 0;
        }
    }

    // Column transform
    for x in 0..32 {
        dct64_1d(&mut tmp[x..], 32, col_clip_min, col_clip_max);
    }

    // Add to destination
    #[cfg(target_arch = "x86_64")]
    if let Some(t512) = crate::src::cpu::summon_avx512() {
        add_to_dst_16bpc_avx512(t512, &mut *dst, stride_u16, &tmp, 32, 32, 64, bitdepth_max);
        coeff[..1024].fill(0);
        return;
    }

    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi32(bitdepth_max);
    let rnd_final = _mm256_set1_epi32(8); // (+ 8) >> 4

    for y in 0..64 {
        let dst_off = y * stride_u16;

        // Process 32 pixels in four 8-pixel chunks
        for chunk in 0..4 {
            let x_base = chunk * 8;
            let dst_chunk_off = dst_off + x_base;

            let d =
                loadu_128!(<&[u16; 8]>::try_from(&dst[dst_chunk_off..dst_chunk_off + 8]).unwrap());
            let d_lo = _mm_unpacklo_epi16(d, _mm_setzero_si128());
            let d_hi = _mm_unpackhi_epi16(d, _mm_setzero_si128());

            let c_lo = _mm_set_epi32(
                tmp[y * 32 + x_base + 3],
                tmp[y * 32 + x_base + 2],
                tmp[y * 32 + x_base + 1],
                tmp[y * 32 + x_base + 0],
            );
            let c_hi = _mm_set_epi32(
                tmp[y * 32 + x_base + 7],
                tmp[y * 32 + x_base + 6],
                tmp[y * 32 + x_base + 5],
                tmp[y * 32 + x_base + 4],
            );

            let d32 = _mm256_set_m128i(d_hi, d_lo);
            let c32 = _mm256_set_m128i(c_hi, c_lo);

            let c_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c32, rnd_final));
            let sum = _mm256_add_epi32(d32, c_scaled);
            let clamped = _mm256_max_epi32(_mm256_min_epi32(sum, max_val), zero);

            let lo = _mm256_castsi256_si128(clamped);
            let hi = _mm256_extracti128_si256(clamped, 1);
            let packed = _mm_packus_epi32(lo, hi);
            storeu_128!(
                <&mut [u16; 8]>::try_from(&mut dst[dst_chunk_off..dst_chunk_off + 8]).unwrap(),
                packed
            );
        }
    }

    // Clear coefficients (only 1024 stored due to high-frequency zeroing)
    coeff[..1024].fill(0);
}

/// FFI wrapper for 32x64 DCT_DCT 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_32x64_16bpc_avx2(
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
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_32x64_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 64x32 DCT_DCT for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_64x32_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize,
    coeff: &mut [i32],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let stride_u16 = dst_stride / 2;
    let row_clip_min = (!bitdepth_max) << 7;
    let row_clip_max = !row_clip_min;
    let col_clip_min = (!bitdepth_max) << 5;
    let col_clip_max = !col_clip_min;
    let mut tmp = [0i32; 2048];

    // is_rect2 = true for 64x32
    let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

    // Row transform with shift=1 (64x32 intermediate shift)
    let rnd = 1;
    let shift = 1;

    // Row transform - only first 32 columns have stored coefficients (high-freq zeroing)
    // Coeff is column-major with stride 32
    for y in 0..32 {
        let mut scratch = [0i32; 64];
        for x in 0..32 {
            scratch[x] = rect2_scale(coeff[y + x * 32] as i32);
        }
        // Zero-extend: columns 32..63 have no stored coefficients
        for x in 32..64 {
            scratch[x] = 0;
        }
        dct64_1d(&mut scratch[..64], 1, row_clip_min, row_clip_max);
        for x in 0..64 {
            tmp[y * 64 + x] = iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
        }
    }

    // Column transform: SIMD across 64 columns, 32 rows
    if let Some(t512) = crate::src::cpu::summon_avx512() {
        dct32_cols_avx512(t512, &mut tmp, 64, 32, col_clip_min, col_clip_max);
    } else {
        let min_v = _mm256_set1_epi32(col_clip_min);
        let max_v = _mm256_set1_epi32(col_clip_max);
        for cx_chunk in 0..8 {
            let cx = cx_chunk * 8;
            let mut v = [_mm256_setzero_si256(); 32];
            for i in 0..32 {
                v[i] = loadu_256!(&tmp[i * 64 + cx..i * 64 + cx + 8], [i32; 8]);
            }
            dct32_1d_cols8(_token, &mut v, min_v, max_v);
            for i in 0..32 {
                storeu_256!(&mut tmp[i * 64 + cx..i * 64 + cx + 8], [i32; 8], v[i]);
            }
        }
    }

    // Add to destination
    #[cfg(target_arch = "x86_64")]
    if let Some(t512) = crate::src::cpu::summon_avx512() {
        add_to_dst_16bpc_avx512(t512, &mut *dst, stride_u16, &tmp, 64, 64, 32, bitdepth_max);
        coeff[..1024].fill(0);
        return;
    }

    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi32(bitdepth_max);
    let rnd_final = _mm256_set1_epi32(8); // (+ 8) >> 4

    for y in 0..32 {
        let dst_off = y * stride_u16;

        // Process 64 pixels in eight 8-pixel chunks
        for chunk in 0..8 {
            let x_base = chunk * 8;
            let dst_chunk_off = dst_off + x_base;

            let d =
                loadu_128!(<&[u16; 8]>::try_from(&dst[dst_chunk_off..dst_chunk_off + 8]).unwrap());
            let d_lo = _mm_unpacklo_epi16(d, _mm_setzero_si128());
            let d_hi = _mm_unpackhi_epi16(d, _mm_setzero_si128());

            let c_lo = _mm_set_epi32(
                tmp[y * 64 + x_base + 3],
                tmp[y * 64 + x_base + 2],
                tmp[y * 64 + x_base + 1],
                tmp[y * 64 + x_base + 0],
            );
            let c_hi = _mm_set_epi32(
                tmp[y * 64 + x_base + 7],
                tmp[y * 64 + x_base + 6],
                tmp[y * 64 + x_base + 5],
                tmp[y * 64 + x_base + 4],
            );

            let d32 = _mm256_set_m128i(d_hi, d_lo);
            let c32 = _mm256_set_m128i(c_hi, c_lo);

            let c_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c32, rnd_final));
            let sum = _mm256_add_epi32(d32, c_scaled);
            let clamped = _mm256_max_epi32(_mm256_min_epi32(sum, max_val), zero);

            let lo = _mm256_castsi256_si128(clamped);
            let hi = _mm256_extracti128_si256(clamped, 1);
            let packed = _mm_packus_epi32(lo, hi);
            storeu_128!(
                <&mut [u16; 8]>::try_from(&mut dst[dst_chunk_off..dst_chunk_off + 8]).unwrap(),
                packed
            );
        }
    }

    // Clear coefficients (only 1024 stored due to high-frequency zeroing)
    coeff[..1024].fill(0);
}

/// FFI wrapper for 64x32 DCT_DCT 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_64x32_16bpc_avx2(
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
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_64x32_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 16x64 DCT_DCT for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_16x64_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize,
    coeff: &mut [i32],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let stride_u16 = dst_stride / 2;
    let row_clip_min = (!bitdepth_max) << 7;
    let row_clip_max = !row_clip_min;
    let col_clip_min = (!bitdepth_max) << 5;
    let col_clip_max = !col_clip_min;
    let mut tmp = [0i32; 1024];

    // is_rect2 = false for 16x64 (4:1 ratio), no rect2_scale
    // Row transform with shift=2 (16x64 intermediate shift)
    let rnd = 2;
    let shift = 2;

    // Row transform - only first 32 rows have stored coefficients (high-freq zeroing)
    // Coeff is column-major with stride 32
    for y in 0..32 {
        let mut scratch = [0i32; 16];
        for x in 0..16 {
            scratch[x] = coeff[y + x * 32] as i32;
        }
        dct16_1d(&mut scratch[..16], 1, row_clip_min, row_clip_max);
        for x in 0..16 {
            tmp[y * 16 + x] = iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
        }
    }
    // Zero-pad rows 32..63
    for y in 32..64 {
        for x in 0..16 {
            tmp[y * 16 + x] = 0;
        }
    }

    // Column transform
    for x in 0..16 {
        dct64_1d(&mut tmp[x..], 16, col_clip_min, col_clip_max);
    }

    // Add to destination
    #[cfg(target_arch = "x86_64")]
    if let Some(t512) = crate::src::cpu::summon_avx512() {
        add_to_dst_16bpc_avx512(t512, &mut *dst, stride_u16, &tmp, 16, 16, 64, bitdepth_max);
        coeff[..512].fill(0);
        return;
    }

    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi32(bitdepth_max);
    let rnd_final = _mm256_set1_epi32(8); // (+ 8) >> 4

    for y in 0..64 {
        let dst_off = y * stride_u16;

        let d = loadu_256!(<&[u16; 16]>::try_from(&dst[dst_off..dst_off + 16]).unwrap());
        let d_lo = _mm256_unpacklo_epi16(d, _mm256_setzero_si256());
        let d_hi = _mm256_unpackhi_epi16(d, _mm256_setzero_si256());
        let d_0_4 = _mm256_permute2x128_si256(d_lo, d_hi, 0x20);
        let d_4_8 = _mm256_permute2x128_si256(d_lo, d_hi, 0x31);

        let c0 = _mm256_set_epi32(
            tmp[y * 16 + 7],
            tmp[y * 16 + 6],
            tmp[y * 16 + 5],
            tmp[y * 16 + 4],
            tmp[y * 16 + 3],
            tmp[y * 16 + 2],
            tmp[y * 16 + 1],
            tmp[y * 16 + 0],
        );
        let c1 = _mm256_set_epi32(
            tmp[y * 16 + 15],
            tmp[y * 16 + 14],
            tmp[y * 16 + 13],
            tmp[y * 16 + 12],
            tmp[y * 16 + 11],
            tmp[y * 16 + 10],
            tmp[y * 16 + 9],
            tmp[y * 16 + 8],
        );

        let c0_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c0, rnd_final));
        let c1_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c1, rnd_final));

        let sum0 = _mm256_add_epi32(d_0_4, c0_scaled);
        let sum1 = _mm256_add_epi32(d_4_8, c1_scaled);

        let clamped0 = _mm256_max_epi32(_mm256_min_epi32(sum0, max_val), zero);
        let clamped1 = _mm256_max_epi32(_mm256_min_epi32(sum1, max_val), zero);

        let packed = _mm256_packus_epi32(clamped0, clamped1);
        let packed = _mm256_permute4x64_epi64(packed, 0b11_01_10_00);
        storeu_256!(
            <&mut [u16; 16]>::try_from(&mut dst[dst_off..dst_off + 16]).unwrap(),
            packed
        );
    }

    // Clear coefficients (only 512 stored due to high-frequency zeroing)
    coeff[..512].fill(0);
}

/// FFI wrapper for 16x64 DCT_DCT 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_16x64_16bpc_avx2(
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
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_16x64_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 64x16 DCT_DCT for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_64x16_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize,
    coeff: &mut [i32],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let stride_u16 = dst_stride / 2;
    let row_clip_min = (!bitdepth_max) << 7;
    let row_clip_max = !row_clip_min;
    let col_clip_min = (!bitdepth_max) << 5;
    let col_clip_max = !col_clip_min;
    let mut tmp = [0i32; 1024];

    // is_rect2 = false for 64x16 (4:1 ratio), no rect2_scale
    // Row transform with shift=2 (64x16 intermediate shift)
    let rnd = 2;
    let shift = 2;

    // Row transform - only first 32 columns have stored coefficients (high-freq zeroing)
    // Coeff is column-major with stride 16
    for y in 0..16 {
        let mut scratch = [0i32; 64];
        for x in 0..32 {
            scratch[x] = coeff[y + x * 16] as i32;
        }
        // Zero-extend: columns 32..63 have no stored coefficients
        for x in 32..64 {
            scratch[x] = 0;
        }
        dct64_1d(&mut scratch[..64], 1, row_clip_min, row_clip_max);
        for x in 0..64 {
            tmp[y * 64 + x] = iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
        }
    }

    // Column transform: SIMD across 64 columns, 16 rows
    if let Some(t512) = crate::src::cpu::summon_avx512() {
        dct16_cols_avx512(t512, &mut tmp, 64, 16, col_clip_min, col_clip_max);
    } else {
        let min_v = _mm256_set1_epi32(col_clip_min);
        let max_v = _mm256_set1_epi32(col_clip_max);
        for cx_chunk in 0..8 {
            let cx = cx_chunk * 8;
            let mut v = [_mm256_setzero_si256(); 16];
            for i in 0..16 {
                v[i] = loadu_256!(&tmp[i * 64 + cx..i * 64 + cx + 8], [i32; 8]);
            }
            dct16_1d_cols8(_token, &mut v, min_v, max_v);
            for i in 0..16 {
                storeu_256!(&mut tmp[i * 64 + cx..i * 64 + cx + 8], [i32; 8], v[i]);
            }
        }
    }

    // Add to destination
    #[cfg(target_arch = "x86_64")]
    if let Some(t512) = crate::src::cpu::summon_avx512() {
        add_to_dst_16bpc_avx512(t512, &mut *dst, stride_u16, &tmp, 64, 64, 16, bitdepth_max);
        coeff[..512].fill(0);
        return;
    }

    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi32(bitdepth_max);
    let rnd_final = _mm256_set1_epi32(8); // (+ 8) >> 4

    for y in 0..16 {
        let dst_off = y * stride_u16;

        // Process 64 pixels in eight 8-pixel chunks
        for chunk in 0..8 {
            let x_base = chunk * 8;
            let dst_chunk_off = dst_off + x_base;

            let d =
                loadu_128!(<&[u16; 8]>::try_from(&dst[dst_chunk_off..dst_chunk_off + 8]).unwrap());
            let d_lo = _mm_unpacklo_epi16(d, _mm_setzero_si128());
            let d_hi = _mm_unpackhi_epi16(d, _mm_setzero_si128());

            let c_lo = _mm_set_epi32(
                tmp[y * 64 + x_base + 3],
                tmp[y * 64 + x_base + 2],
                tmp[y * 64 + x_base + 1],
                tmp[y * 64 + x_base + 0],
            );
            let c_hi = _mm_set_epi32(
                tmp[y * 64 + x_base + 7],
                tmp[y * 64 + x_base + 6],
                tmp[y * 64 + x_base + 5],
                tmp[y * 64 + x_base + 4],
            );

            let d32 = _mm256_set_m128i(d_hi, d_lo);
            let c32 = _mm256_set_m128i(c_hi, c_lo);

            let c_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c32, rnd_final));
            let sum = _mm256_add_epi32(d32, c_scaled);
            let clamped = _mm256_max_epi32(_mm256_min_epi32(sum, max_val), zero);

            let lo = _mm256_castsi256_si128(clamped);
            let hi = _mm256_extracti128_si256(clamped, 1);
            let packed = _mm_packus_epi32(lo, hi);
            storeu_128!(
                <&mut [u16; 8]>::try_from(&mut dst[dst_chunk_off..dst_chunk_off + 8]).unwrap(),
                packed
            );
        }
    }

    // Clear coefficients (only 512 stored due to high-frequency zeroing)
    coeff[..512].fill(0);
}

/// FFI wrapper for 64x16 DCT_DCT 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_64x16_16bpc_avx2(
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
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_64x16_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// 8x8 ADST/FLIPADST TRANSFORMS 16bpc
// ============================================================================

/// Helper macro for 8x8 transform implementations (16bpc)
#[allow(unused_macros)]
macro_rules! impl_8x8_transform_16bpc {
    ($name:ident, $row_fn:ident, $col_fn:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[arcane]
        pub fn $name(
            _token: Desktop64,
            dst: &mut [u16],
            dst_stride: usize,
            coeff: &mut [i16],
            _eob: i32,
            bitdepth_max: i32,
        ) {
            use crate::src::safe_simd::pixel_access::{
                loadi32, loadi64, loadu_128, storei32, storei64, storeu_128,
            };
            let mut dst = dst.flex_mut();
            let mut coeff = coeff.flex_mut();
            let stride_u16 = dst_stride / 2;
            const MIN: i32 = i32::MIN;
            const MAX: i32 = i32::MAX;

            // Load coefficients (row-major)
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
                let dst_off = y * stride_u16;
                for x in 0..8 {
                    let pixel = dst[dst_off + x] as i32;
                    let val = pixel + ((out[y][x] + 8) >> 4);
                    dst[dst_off + x] = val.clamp(0, bitdepth_max) as u16;
                }
            }

            // Clear coefficients
            coeff[..64].fill(0);
        }
    };
}

/// 8x8 16bpc variant with SIMD column pass.
macro_rules! impl_8x8_transform_16bpc_simd_col {
    ($name:ident, $row_fn:ident, $simd_col_fn:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[arcane]
        pub fn $name(
            _token: Desktop64,
            dst: &mut [u16],
            dst_stride: usize,
            coeff: &mut [i16],
            _eob: i32,
            bitdepth_max: i32,
        ) {
            let mut dst = dst.flex_mut();
            let mut coeff = coeff.flex_mut();
            let stride_u16 = dst_stride / 2;
            const MIN: i32 = i32::MIN;
            const MAX: i32 = i32::MAX;

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

            for y in 0..8 {
                let dst_off = y * stride_u16;
                for x in 0..8 {
                    let pixel = dst[dst_off + x] as i32;
                    let val = pixel + ((tmp[y * 8 + x] + 8) >> 4);
                    dst[dst_off + x] = val.clamp(0, bitdepth_max) as u16;
                }
            }

            coeff[..64].fill(0);
        }
    };
}

// Generate all 8x8 ADST/FlipADST combinations for 16bpc (SIMD col)
impl_8x8_transform_16bpc_simd_col!(
    inv_txfm_add_adst_dct_8x8_16bpc_avx2_inner,
    adst8_1d_scalar,
    dct8_1d_cols8
);
impl_8x8_transform_16bpc_simd_col!(
    inv_txfm_add_dct_adst_8x8_16bpc_avx2_inner,
    dct8_1d_scalar,
    adst8_1d_cols8
);
impl_8x8_transform_16bpc_simd_col!(
    inv_txfm_add_adst_adst_8x8_16bpc_avx2_inner,
    adst8_1d_scalar,
    adst8_1d_cols8
);
impl_8x8_transform_16bpc_simd_col!(
    inv_txfm_add_flipadst_dct_8x8_16bpc_avx2_inner,
    flipadst8_1d_scalar,
    dct8_1d_cols8
);
impl_8x8_transform_16bpc_simd_col!(
    inv_txfm_add_dct_flipadst_8x8_16bpc_avx2_inner,
    dct8_1d_scalar,
    flipadst8_1d_cols8
);
impl_8x8_transform_16bpc_simd_col!(
    inv_txfm_add_flipadst_flipadst_8x8_16bpc_avx2_inner,
    flipadst8_1d_scalar,
    flipadst8_1d_cols8
);
impl_8x8_transform_16bpc_simd_col!(
    inv_txfm_add_adst_flipadst_8x8_16bpc_avx2_inner,
    adst8_1d_scalar,
    flipadst8_1d_cols8
);
impl_8x8_transform_16bpc_simd_col!(
    inv_txfm_add_flipadst_adst_8x8_16bpc_avx2_inner,
    flipadst8_1d_scalar,
    adst8_1d_cols8
);

// FFI wrappers for 8x8 16bpc transforms
macro_rules! impl_8x8_ffi_wrapper_16bpc {
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
                std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
            };

            let coeff_slice =
                unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

            $inner(_token, dst_slice, stride, coeff_slice, eob, bitdepth_max);
        }
    };
}

impl_8x8_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_dct_8x8_16bpc_avx2,
    inv_txfm_add_adst_dct_8x8_16bpc_avx2_inner
);
impl_8x8_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_adst_8x8_16bpc_avx2,
    inv_txfm_add_dct_adst_8x8_16bpc_avx2_inner
);
impl_8x8_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_adst_8x8_16bpc_avx2,
    inv_txfm_add_adst_adst_8x8_16bpc_avx2_inner
);
impl_8x8_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_dct_8x8_16bpc_avx2,
    inv_txfm_add_flipadst_dct_8x8_16bpc_avx2_inner
);
impl_8x8_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_flipadst_8x8_16bpc_avx2,
    inv_txfm_add_dct_flipadst_8x8_16bpc_avx2_inner
);
impl_8x8_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_flipadst_8x8_16bpc_avx2,
    inv_txfm_add_flipadst_flipadst_8x8_16bpc_avx2_inner
);
impl_8x8_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_flipadst_8x8_16bpc_avx2,
    inv_txfm_add_adst_flipadst_8x8_16bpc_avx2_inner
);
impl_8x8_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_adst_8x8_16bpc_avx2,
    inv_txfm_add_flipadst_adst_8x8_16bpc_avx2_inner
);

// ============================================================================
// 4x4 ADST/FLIPADST TRANSFORMS 16bpc
// ============================================================================

/// Helper macro for 4x4 transform implementations (16bpc)
macro_rules! impl_4x4_transform_16bpc {
    ($name:ident, $row_fn:ident, $col_fn:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[cfg(feature = "asm")]
        pub fn $name(
            dst: &mut [u16],
            dst_base: usize,
            dst_stride_u16: isize,
            coeff: &mut [i16],
            _eob: i32,
            bitdepth_max: i32,
        ) {
            use crate::src::safe_simd::pixel_access::{
                loadi32, loadi64, loadu_128, storei32, storei64, storeu_128,
            };
            // Load coefficients (row-major)
            let mut c = [[0i32; 4]; 4];
            for y in 0..4 {
                for x in 0..4 {
                    c[y][x] = coeff[y + x * 4] as i32;
                }
            }

            // Clip ranges for 16bpc (matching reference inv_txfm_add_rust)
            let row_clip_min = (!bitdepth_max) << 7;
            let row_clip_max = !row_clip_min;
            let col_clip_min = (!bitdepth_max) << 5;
            let col_clip_max = !col_clip_min;

            // First pass: transform on rows
            let mut tmp = [[0i32; 4]; 4];
            for y in 0..4 {
                let (o0, o1, o2, o3) = $row_fn(
                    c[y][0],
                    c[y][1],
                    c[y][2],
                    c[y][3],
                    row_clip_min,
                    row_clip_max,
                );
                tmp[y][0] = o0;
                tmp[y][1] = o1;
                tmp[y][2] = o2;
                tmp[y][3] = o3;
            }

            // Intermediate clip (shift=0 for 4x4)
            for y in 0..4 {
                for x in 0..4 {
                    tmp[y][x] = tmp[y][x].clamp(col_clip_min, col_clip_max);
                }
            }

            // Second pass: transform on columns
            let mut out = [[0i32; 4]; 4];
            for x in 0..4 {
                let (o0, o1, o2, o3) = $col_fn(
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

            // Add to destination with rounding
            for y in 0..4 {
                let row_off = dst_base.wrapping_add_signed(y as isize * dst_stride_u16);
                for x in 0..4 {
                    let pixel = dst[row_off + x] as i32;
                    let val = pixel + ((out[y][x] + 8) >> 4);
                    dst[row_off + x] = val.clamp(0, bitdepth_max) as u16;
                }
            }

            // Clear coefficients
            coeff[..16].fill(0);
        }
    };
}

// Generate all 4x4 ADST/FlipADST combinations for 16bpc
impl_4x4_transform_16bpc!(
    inv_txfm_add_adst_dct_4x4_16bpc_avx2_inner,
    adst4_1d_scalar,
    dct4_1d_scalar
);
impl_4x4_transform_16bpc!(
    inv_txfm_add_dct_adst_4x4_16bpc_avx2_inner,
    dct4_1d_scalar,
    adst4_1d_scalar
);
impl_4x4_transform_16bpc!(
    inv_txfm_add_adst_adst_4x4_16bpc_avx2_inner,
    adst4_1d_scalar,
    adst4_1d_scalar
);
impl_4x4_transform_16bpc!(
    inv_txfm_add_flipadst_dct_4x4_16bpc_avx2_inner,
    flipadst4_1d_scalar,
    dct4_1d_scalar
);
impl_4x4_transform_16bpc!(
    inv_txfm_add_dct_flipadst_4x4_16bpc_avx2_inner,
    dct4_1d_scalar,
    flipadst4_1d_scalar
);
impl_4x4_transform_16bpc!(
    inv_txfm_add_flipadst_flipadst_4x4_16bpc_avx2_inner,
    flipadst4_1d_scalar,
    flipadst4_1d_scalar
);
impl_4x4_transform_16bpc!(
    inv_txfm_add_adst_flipadst_4x4_16bpc_avx2_inner,
    adst4_1d_scalar,
    flipadst4_1d_scalar
);
impl_4x4_transform_16bpc!(
    inv_txfm_add_flipadst_adst_4x4_16bpc_avx2_inner,
    flipadst4_1d_scalar,
    adst4_1d_scalar
);

// FFI wrappers for 4x4 16bpc transforms
macro_rules! impl_4x4_ffi_wrapper_16bpc {
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
            let stride_u16 = dst_stride / 2;
            let coeff_slice = unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, 16) };
            let abs_stride = stride_u16;
            let (dst_slice, dst_base) = if stride_u16 >= 0 {
                let len = 3 * abs_stride + 4;
                (
                    unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u16, len) },
                    0usize,
                )
            } else {
                let len = 3 * abs_stride + 4;
                let start = unsafe { (dst_ptr as *mut u16).offset(3 * stride_u16) };
                (
                    unsafe { std::slice::from_raw_parts_mut(start, len) },
                    3 * abs_stride,
                )
            };
            $inner(
                dst_slice,
                dst_base,
                stride_u16,
                coeff_slice,
                eob,
                bitdepth_max,
            );
        }
    };
}

impl_4x4_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_dct_4x4_16bpc_avx2,
    inv_txfm_add_adst_dct_4x4_16bpc_avx2_inner
);
impl_4x4_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_adst_4x4_16bpc_avx2,
    inv_txfm_add_dct_adst_4x4_16bpc_avx2_inner
);
impl_4x4_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_adst_4x4_16bpc_avx2,
    inv_txfm_add_adst_adst_4x4_16bpc_avx2_inner
);
impl_4x4_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_dct_4x4_16bpc_avx2,
    inv_txfm_add_flipadst_dct_4x4_16bpc_avx2_inner
);
impl_4x4_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_flipadst_4x4_16bpc_avx2,
    inv_txfm_add_dct_flipadst_4x4_16bpc_avx2_inner
);
impl_4x4_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_flipadst_4x4_16bpc_avx2,
    inv_txfm_add_flipadst_flipadst_4x4_16bpc_avx2_inner
);
impl_4x4_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_flipadst_4x4_16bpc_avx2,
    inv_txfm_add_adst_flipadst_4x4_16bpc_avx2_inner
);
impl_4x4_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_adst_4x4_16bpc_avx2,
    inv_txfm_add_flipadst_adst_4x4_16bpc_avx2_inner
);

// ============================================================================
// 16x16 ADST/FLIPADST TRANSFORMS 16bpc
// ============================================================================

/// Helper macro for 16x16 transform implementations (16bpc)
#[allow(unused_macros)]
macro_rules! impl_16x16_transform_16bpc {
    ($name:ident, $row_fn:ident, $col_fn:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[arcane]
        pub fn $name(
            _token: Desktop64,
            dst: &mut [u16],
            dst_stride: usize,
            coeff: &mut [i16],
            _eob: i32,
            bitdepth_max: i32,
        ) {
            use crate::src::safe_simd::pixel_access::{
                loadi32, loadi64, loadu_128, storei32, storei64, storeu_128,
            };
            let mut dst = dst.flex_mut();
            let mut coeff = coeff.flex_mut();
            let stride_u16 = dst_stride / 2;
            const MIN: i32 = i32::MIN;
            const MAX: i32 = i32::MAX;

            // Load coefficients (row-major)
            let mut c = [[0i32; 16]; 16];
            for y in 0..16 {
                for x in 0..16 {
                    c[y][x] = coeff[y * 16 + x] as i32;
                }
            }

            // First pass: transform on rows
            let mut tmp = [[0i32; 16]; 16];
            for y in 0..16 {
                let mut row = [0i32; 16];
                for x in 0..16 {
                    row[x] = c[y][x];
                }
                $row_fn(&mut row, 1, MIN, MAX);
                for x in 0..16 {
                    tmp[y][x] = row[x];
                }
            }

            // Second pass: transform on columns
            let mut out = [[0i32; 16]; 16];
            for x in 0..16 {
                let mut col = [0i32; 16];
                for y in 0..16 {
                    col[y] = tmp[y][x];
                }
                $col_fn(&mut col, 1, MIN, MAX);
                for y in 0..16 {
                    out[y][x] = col[y];
                }
            }

            // Add to destination with rounding
            for y in 0..16 {
                let dst_off = y * stride_u16;
                for x in 0..16 {
                    let pixel = dst[dst_off + x] as i32;
                    let val = pixel + ((out[y][x] + 8) >> 4);
                    dst[dst_off + x] = val.clamp(0, bitdepth_max) as u16;
                }
            }

            // Clear coefficients
            coeff[..256].fill(0);
        }
    };
}

/// 16bpc 16x16 transform variant: scalar row + SIMD col pass.
/// Used for transforms where col_fn has a matching `*16x16_cols_simd` helper.
macro_rules! impl_16x16_transform_16bpc_simd_col {
    ($name:ident, $row_fn:ident, $simd_col_fn:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[arcane]
        pub fn $name(
            _token: Desktop64,
            dst: &mut [u16],
            dst_stride: usize,
            coeff: &mut [i16],
            _eob: i32,
            bitdepth_max: i32,
        ) {
            let mut dst = dst.flex_mut();
            let mut coeff = coeff.flex_mut();
            let stride_u16 = dst_stride / 2;
            const MIN: i32 = i32::MIN;
            const MAX: i32 = i32::MAX;

            // Row pass: contiguous stride=1 over 16-element row
            let mut tmp = [0i32; 256];
            for y in 0..16 {
                let mut row = [0i32; 16];
                for x in 0..16 {
                    row[x] = coeff[y * 16 + x] as i32;
                }
                $row_fn(&mut row, 1, MIN, MAX);
                for x in 0..16 {
                    tmp[y * 16 + x] = row[x];
                }
            }

            // SIMD column transform (in-place on row-major tmp)
            $simd_col_fn(_token, &mut tmp, MIN, MAX);

            // Add to destination with (val + 8) >> 4 rounding
            for y in 0..16 {
                let dst_off = y * stride_u16;
                for x in 0..16 {
                    let pixel = dst[dst_off + x] as i32;
                    let val = pixel + ((tmp[y * 16 + x] + 8) >> 4);
                    dst[dst_off + x] = val.clamp(0, bitdepth_max) as u16;
                }
            }

            coeff[..256].fill(0);
        }
    };
}

// Generate key 16x16 ADST combinations for 16bpc
impl_16x16_transform_16bpc_simd_col!(
    inv_txfm_add_adst_dct_16x16_16bpc_avx2_inner,
    adst16_1d,
    dct16x16_cols_simd
);
impl_16x16_transform_16bpc_simd_col!(
    inv_txfm_add_dct_adst_16x16_16bpc_avx2_inner,
    dct16_1d,
    adst16x16_cols_simd
);
impl_16x16_transform_16bpc_simd_col!(
    inv_txfm_add_adst_adst_16x16_16bpc_avx2_inner,
    adst16_1d,
    adst16x16_cols_simd
);
impl_16x16_transform_16bpc_simd_col!(
    inv_txfm_add_flipadst_dct_16x16_16bpc_avx2_inner,
    flipadst16_1d,
    dct16x16_cols_simd
);
impl_16x16_transform_16bpc_simd_col!(
    inv_txfm_add_dct_flipadst_16x16_16bpc_avx2_inner,
    dct16_1d,
    flipadst16x16_cols_simd
);
impl_16x16_transform_16bpc_simd_col!(
    inv_txfm_add_flipadst_flipadst_16x16_16bpc_avx2_inner,
    flipadst16_1d,
    flipadst16x16_cols_simd
);
impl_16x16_transform_16bpc_simd_col!(
    inv_txfm_add_adst_flipadst_16x16_16bpc_avx2_inner,
    adst16_1d,
    flipadst16x16_cols_simd
);
impl_16x16_transform_16bpc_simd_col!(
    inv_txfm_add_flipadst_adst_16x16_16bpc_avx2_inner,
    flipadst16_1d,
    adst16x16_cols_simd
);

// FFI wrappers for 16x16 16bpc transforms
macro_rules! impl_16x16_ffi_wrapper_16bpc {
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
                std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
            };

            let coeff_slice =
                unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

            $inner(_token, dst_slice, stride, coeff_slice, eob, bitdepth_max);
        }
    };
}

impl_16x16_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_dct_16x16_16bpc_avx2,
    inv_txfm_add_adst_dct_16x16_16bpc_avx2_inner
);
impl_16x16_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_adst_16x16_16bpc_avx2,
    inv_txfm_add_dct_adst_16x16_16bpc_avx2_inner
);
impl_16x16_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_adst_16x16_16bpc_avx2,
    inv_txfm_add_adst_adst_16x16_16bpc_avx2_inner
);
impl_16x16_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_dct_16x16_16bpc_avx2,
    inv_txfm_add_flipadst_dct_16x16_16bpc_avx2_inner
);
impl_16x16_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_flipadst_16x16_16bpc_avx2,
    inv_txfm_add_dct_flipadst_16x16_16bpc_avx2_inner
);
impl_16x16_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_flipadst_16x16_16bpc_avx2,
    inv_txfm_add_flipadst_flipadst_16x16_16bpc_avx2_inner
);
impl_16x16_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_flipadst_16x16_16bpc_avx2,
    inv_txfm_add_adst_flipadst_16x16_16bpc_avx2_inner
);
impl_16x16_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_adst_16x16_16bpc_avx2,
    inv_txfm_add_flipadst_adst_16x16_16bpc_avx2_inner
);

