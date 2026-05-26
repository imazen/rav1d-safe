// ============================================================================
// IDENTITY TRANSFORMS 16bpc
// ============================================================================

/// 4x4 IDTX (identity transform) for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
pub fn inv_identity_add_4x4_16bpc_avx2(
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
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi32(bitdepth_max);

    for y in 0..4 {
        let dst_off = y * stride_u16;

        // Load destination (4 u16)
        let d = loadi64!(zerocopy::IntoBytes::as_bytes(&dst[dst_off..dst_off + 4]));
        let d32 = _mm_unpacklo_epi16(d, zero);

        // Load coeffs (column-major: y, y+4, y+8, y+12)
        let c0 = coeff[y] as i32;
        let c1 = coeff[y + 4] as i32;
        let c2 = coeff[y + 8] as i32;
        let c3 = coeff[y + 12] as i32;

        // identity4(x) = x + (x * 1697 + 2048) >> 12 ≈ x * sqrt(2)
        // Row: identity4, shift=0, Col: identity4, final: (+ 8) >> 4
        let identity4 = |v: i32| -> i32 { v + ((v * 1697 + 2048) >> 12) };
        let scale = |v: i32| -> i32 { identity4(identity4(v)) };

        let r0 = (scale(c0) + 8) >> 4;
        let r1 = (scale(c1) + 8) >> 4;
        let r2 = (scale(c2) + 8) >> 4;
        let r3 = (scale(c3) + 8) >> 4;

        // Add to destination
        let result = _mm_set_epi32(r3, r2, r1, r0);
        let sum = _mm_add_epi32(d32, result);
        let clamped = _mm_max_epi32(_mm_min_epi32(sum, max_val), zero);
        let packed = _mm_packus_epi32(clamped, clamped);

        storei64!(
            zerocopy::IntoBytes::as_mut_bytes(&mut dst[dst_off..dst_off + 4]),
            packed
        );
    }

    // Clear coefficients
    coeff[..16].fill(0);
}

/// FFI wrapper for 4x4 IDTX 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_identity_4x4_16bpc_avx2(
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

    inv_identity_add_4x4_16bpc_avx2(_token, dst_slice, stride, coeff_slice, eob, bitdepth_max);
}

/// 8x8 IDTX (identity transform) for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
pub fn inv_identity_add_8x8_16bpc_avx2(
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
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi32(bitdepth_max);

    for y in 0..8 {
        let dst_off = y * stride_u16;

        // Load destination (8 u16)
        let d = loadu_128!(<&[u16; 8]>::try_from(&dst[dst_off..dst_off + 8]).unwrap());
        let d_lo = _mm_unpacklo_epi16(d, zero);
        let d_hi = _mm_unpackhi_epi16(d, zero);

        // Load coefficients (column-major: y, y+8, ...)
        let mut coeffs = [0i32; 8];
        for x in 0..8 {
            coeffs[x] = coeff[y + x * 8] as i32;
        }

        // Identity8: * 2 per dimension, with intermediate shift between passes
        // For 8x8: shift=1, rnd=1, col_clip = (!bitdepth_max) << 5
        // Row: c * 2. Intermediate: iclip((c*2 + 1) >> 1, ...). Col: * 2. Final: (+ 8) >> 4
        let col_clip_min = (!(bitdepth_max)) << 5;
        let col_clip_max = !col_clip_min;
        let mut results = [0i32; 8];
        for x in 0..8 {
            let row = coeffs[x] * 2; // identity8 row
            let inter = ((row + 1) >> 1).clamp(col_clip_min, col_clip_max); // intermediate shift
            let col = inter * 2; // identity8 col
            results[x] = (col + 8) >> 4; // final shift
        }

        let c_lo = _mm_set_epi32(results[3], results[2], results[1], results[0]);
        let c_hi = _mm_set_epi32(results[7], results[6], results[5], results[4]);

        // Add to destination
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
    coeff[..64].fill(0);
}

/// FFI wrapper for 8x8 IDTX 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_identity_8x8_16bpc_avx2(
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

    inv_identity_add_8x8_16bpc_avx2(_token, dst_slice, stride, coeff_slice, eob, bitdepth_max);
}

/// 16x16 IDTX (identity transform) for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
pub fn inv_identity_add_16x16_16bpc_avx2(
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

    // Identity16 scale factor: f(x) = 2*x + (x*1697 + 1024) >> 11
    // For 16x16: row_pass → intermediate shift (+ 2) >> 2 with clamp → col_pass → (+ 8) >> 4
    // The intermediate shift is shift=2, rnd=2 for 16x16 (from inv_txfm_add_rust)
    // 16bpc clamp: col_clip_min = (!bitdepth_max) << 5, col_clip_max = !col_clip_min
    let identity16_scale = |v: i32| -> i32 { 2 * v + ((v * 1697 + 1024) >> 11) };
    let col_clip_min = (!(bitdepth_max)) << 5;
    let col_clip_max = !col_clip_min;

    // Row pass: identity16 on each row
    let mut tmp = [[0i32; 16]; 16];
    for y in 0..16 {
        for x in 0..16 {
            let c = coeff[y + x * 16] as i32;
            tmp[y][x] = identity16_scale(c);
        }
    }

    // Intermediate shift: (val + 2) >> 2, clamped to col_clip range
    for y in 0..16 {
        for x in 0..16 {
            tmp[y][x] = ((tmp[y][x] + 2) >> 2).clamp(col_clip_min, col_clip_max);
        }
    }

    // Column pass: identity16, then final shift
    for x in 0..16 {
        for y in 0..16 {
            tmp[y][x] = identity16_scale(tmp[y][x]);
        }
    }

    // Add to destination
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi32(bitdepth_max);
    let rnd_final = _mm256_set1_epi32(8);

    for y in 0..16 {
        let dst_off = y * stride_u16;

        let d = loadu_256!(<&[u16; 16]>::try_from(&dst[dst_off..dst_off + 16]).unwrap());
        let d_lo = _mm256_unpacklo_epi16(d, _mm256_setzero_si256());
        let d_hi = _mm256_unpackhi_epi16(d, _mm256_setzero_si256());
        let d_0_4 = _mm256_permute2x128_si256(d_lo, d_hi, 0x20);
        let d_4_8 = _mm256_permute2x128_si256(d_lo, d_hi, 0x31);

        let c0 = _mm256_set_epi32(
            tmp[y][7], tmp[y][6], tmp[y][5], tmp[y][4], tmp[y][3], tmp[y][2], tmp[y][1], tmp[y][0],
        );
        let c1 = _mm256_set_epi32(
            tmp[y][15], tmp[y][14], tmp[y][13], tmp[y][12], tmp[y][11], tmp[y][10], tmp[y][9],
            tmp[y][8],
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
    coeff[..256].fill(0);
}

/// FFI wrapper for 16x16 IDTX 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_identity_16x16_16bpc_avx2(
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

    inv_identity_add_16x16_16bpc_avx2(_token, dst_slice, stride, coeff_slice, eob, bitdepth_max);
}

// ============================================================================
// 32x32 IDTX 16bpc
// ============================================================================

/// 32x32 IDTX inner function for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_identity_identity_32x32_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize,
    coeff: &mut [i32],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    // For 16bpc: use full i32 range
    let row_clip_min = (!bitdepth_max) << 7;
    let row_clip_max = !row_clip_min;
    let col_clip_min = (!bitdepth_max) << 5;
    let col_clip_max = !col_clip_min;

    let mut tmp = [0i32; 1024];
    inv_txfm_32x32_inner(
        &mut tmp,
        &*coeff,
        identity32_1d,
        identity32_1d,
        row_clip_min,
        row_clip_max,
        col_clip_min,
        col_clip_max,
    );
    #[cfg(target_arch = "x86_64")]
    if let Some(t512) = crate::src::cpu::summon_avx512() {
        add_to_dst_16bpc_avx512(
            t512,
            &mut *dst,
            dst_stride / 2,
            &tmp,
            32,
            32,
            32,
            bitdepth_max,
        );
    } else {
        add_32x32_to_dst_16bpc(
            _token,
            &mut *dst,
            dst_stride,
            &tmp,
            &mut *coeff,
            bitdepth_max,
        );
        return;
    }
    coeff[..1024].fill(0);
}

/// FFI wrapper for 32x32 IDTX 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_identity_32x32_16bpc_avx2(
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

    inv_txfm_add_identity_identity_32x32_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// Rectangular IDTX 16bpc
// ============================================================================

/// 4x8 IDTX for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_identity_identity_4x8_16bpc_avx2_inner(
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
        identity4_1d(&mut scratch[..4], 1, row_clip_min, row_clip_max);
        for x in 0..4 {
            tmp[y * 4 + x] = iclip(scratch[x], col_clip_min, col_clip_max);
        }
    }

    // Column transform
    for x in 0..4 {
        identity8_1d(&mut tmp[x..], 4, col_clip_min, col_clip_max);
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

/// FFI wrapper for 4x8 IDTX 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_identity_4x8_16bpc_avx2(
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

    inv_txfm_add_identity_identity_4x8_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 8x4 IDTX for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_identity_identity_8x4_16bpc_avx2_inner(
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

    // is_rect2 = true for 8x4, so apply sqrt(2) scaling
    let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

    // Row transform (8 elements each, 4 rows)
    for y in 0..4 {
        let mut scratch = [0i32; 8];
        for x in 0..8 {
            scratch[x] = rect2_scale(coeff[y + x * 4] as i32);
        }
        identity8_1d(&mut scratch[..8], 1, row_clip_min, row_clip_max);
        for x in 0..8 {
            tmp[y * 8 + x] = iclip(scratch[x], col_clip_min, col_clip_max);
        }
    }

    // Column transform
    for x in 0..8 {
        identity4_1d(&mut tmp[x..], 8, col_clip_min, col_clip_max);
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

/// FFI wrapper for 8x4 IDTX 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_identity_8x4_16bpc_avx2(
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

    inv_txfm_add_identity_identity_8x4_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 8x16 IDTX for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_identity_identity_8x16_16bpc_avx2_inner(
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

    // is_rect2 = true for 8x16, so apply sqrt(2) scaling
    let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

    // Row transform with shift=1 (8x16 intermediate shift)
    let rnd = 1;
    let shift = 1;

    for y in 0..16 {
        let mut scratch = [0i32; 8];
        for x in 0..8 {
            scratch[x] = rect2_scale(coeff[y + x * 16] as i32);
        }
        identity8_1d(&mut scratch[..8], 1, row_clip_min, row_clip_max);
        for x in 0..8 {
            tmp[y * 8 + x] = iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
        }
    }

    // Column transform: SIMD identity16 across 8 cols, 16 rows. out = 2*in + ((in*1697+1024)>>11)
    {
        let c1697 = _mm256_set1_epi32(1697);
        let c1024 = _mm256_set1_epi32(1024);
        for i in 0..16 {
            let v = loadu_256!(&tmp[i * 8..i * 8 + 8], [i32; 8]);
            let two_v = _mm256_slli_epi32::<1>(v);
            let mul = _mm256_mullo_epi32(v, c1697);
            let shifted = _mm256_srai_epi32::<11>(_mm256_add_epi32(mul, c1024));
            let result = _mm256_add_epi32(two_v, shifted);
            storeu_256!(&mut tmp[i * 8..i * 8 + 8], [i32; 8], result);
        }
    }

    // Add to destination
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi32(bitdepth_max);

    for y in 0..16 {
        let dst_off = y * stride_u16;

        // Load destination (8 u16 = 16 bytes)
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

/// FFI wrapper for 8x16 IDTX 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_identity_8x16_16bpc_avx2(
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

    inv_txfm_add_identity_identity_8x16_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 16x8 IDTX for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_identity_identity_16x8_16bpc_avx2_inner(
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

    // is_rect2 = true for 16x8, so apply sqrt(2) scaling
    let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

    // Row transform with shift=1 (16x8 intermediate shift)
    let rnd = 1;
    let shift = 1;

    for y in 0..8 {
        let mut scratch = [0i32; 16];
        for x in 0..16 {
            scratch[x] = rect2_scale(coeff[y + x * 8] as i32);
        }
        identity16_1d(&mut scratch[..16], 1, row_clip_min, row_clip_max);
        for x in 0..16 {
            tmp[y * 16 + x] = iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
        }
    }

    // Column transform: SIMD identity8 = *2 across 16 cols (2 chunks of 8), 8 rows
    {
        for cx_chunk in 0..2 {
            let cx = cx_chunk * 8;
            for i in 0..8 {
                let v = loadu_256!(&tmp[i * 16 + cx..i * 16 + cx + 8], [i32; 8]);
                let result = _mm256_slli_epi32::<1>(v);
                storeu_256!(&mut tmp[i * 16 + cx..i * 16 + cx + 8], [i32; 8], result);
            }
        }
    }

    // Add to destination
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi32(bitdepth_max);

    for y in 0..8 {
        let dst_off = y * stride_u16;

        // Process 16 pixels in two 8-pixel chunks
        for chunk in 0..2 {
            let x_base = chunk * 8;
            let dst_chunk_off = dst_off + x_base;

            let d =
                loadu_128!(<&[u16; 8]>::try_from(&dst[dst_chunk_off..dst_chunk_off + 8]).unwrap());
            let d_lo = _mm_unpacklo_epi16(d, zero);
            let d_hi = _mm_unpackhi_epi16(d, zero);

            let c_lo = _mm_set_epi32(
                (tmp[y * 16 + x_base + 3] + 8) >> 4,
                (tmp[y * 16 + x_base + 2] + 8) >> 4,
                (tmp[y * 16 + x_base + 1] + 8) >> 4,
                (tmp[y * 16 + x_base + 0] + 8) >> 4,
            );
            let c_hi = _mm_set_epi32(
                (tmp[y * 16 + x_base + 7] + 8) >> 4,
                (tmp[y * 16 + x_base + 6] + 8) >> 4,
                (tmp[y * 16 + x_base + 5] + 8) >> 4,
                (tmp[y * 16 + x_base + 4] + 8) >> 4,
            );

            let sum_lo = _mm_add_epi32(d_lo, c_lo);
            let sum_hi = _mm_add_epi32(d_hi, c_hi);
            let clamped_lo = _mm_max_epi32(_mm_min_epi32(sum_lo, max_val), zero);
            let clamped_hi = _mm_max_epi32(_mm_min_epi32(sum_hi, max_val), zero);
            let packed = _mm_packus_epi32(clamped_lo, clamped_hi);

            storeu_128!(
                <&mut [u16; 8]>::try_from(&mut dst[dst_chunk_off..dst_chunk_off + 8]).unwrap(),
                packed
            );
        }
    }

    // Clear coefficients
    coeff[..128].fill(0);
}

/// FFI wrapper for 16x8 IDTX 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_identity_16x8_16bpc_avx2(
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

    inv_txfm_add_identity_identity_16x8_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 4x16 IDTX for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_identity_identity_4x16_16bpc_avx2_inner(
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

    // is_rect2 = false for 4x16 (aspect ratio 4:1), no rect2_scale

    // Row transform with shift=1 (4x16 intermediate shift)
    let rnd = 1;
    let shift = 1;

    for y in 0..16 {
        let mut scratch = [0i32; 4];
        for x in 0..4 {
            scratch[x] = coeff[y + x * 16] as i32;
        }
        identity4_1d(&mut scratch[..4], 1, row_clip_min, row_clip_max);
        for x in 0..4 {
            tmp[y * 4 + x] = iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
        }
    }

    // Column transform
    for x in 0..4 {
        identity16_1d(&mut tmp[x..], 4, col_clip_min, col_clip_max);
    }

    // Add to destination
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi32(bitdepth_max);

    for y in 0..16 {
        let dst_off = y * stride_u16;

        // Load destination (4 u16 = 8 bytes)
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

/// FFI wrapper for 4x16 IDTX 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_identity_4x16_16bpc_avx2(
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

    inv_txfm_add_identity_identity_4x16_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 16x4 IDTX for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_identity_identity_16x4_16bpc_avx2_inner(
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

    // is_rect2 = false for 16x4 (aspect ratio 4:1), no rect2_scale

    // Row transform with shift=1 (16x4 intermediate shift)
    let rnd = 1;
    let shift = 1;

    for y in 0..4 {
        let mut scratch = [0i32; 16];
        for x in 0..16 {
            scratch[x] = coeff[y + x * 4] as i32;
        }
        identity16_1d(&mut scratch[..16], 1, row_clip_min, row_clip_max);
        for x in 0..16 {
            tmp[y * 16 + x] = iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
        }
    }

    // Column transform
    for x in 0..16 {
        identity4_1d(&mut tmp[x..], 16, col_clip_min, col_clip_max);
    }

    // Add to destination
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi32(bitdepth_max);

    for y in 0..4 {
        let dst_off = y * stride_u16;

        // Process 16 pixels in two 8-pixel chunks
        for chunk in 0..2 {
            let x_base = chunk * 8;
            let dst_chunk_off = dst_off + x_base;

            let d =
                loadu_128!(<&[u16; 8]>::try_from(&dst[dst_chunk_off..dst_chunk_off + 8]).unwrap());
            let d_lo = _mm_unpacklo_epi16(d, zero);
            let d_hi = _mm_unpackhi_epi16(d, zero);

            let c_lo = _mm_set_epi32(
                (tmp[y * 16 + x_base + 3] + 8) >> 4,
                (tmp[y * 16 + x_base + 2] + 8) >> 4,
                (tmp[y * 16 + x_base + 1] + 8) >> 4,
                (tmp[y * 16 + x_base + 0] + 8) >> 4,
            );
            let c_hi = _mm_set_epi32(
                (tmp[y * 16 + x_base + 7] + 8) >> 4,
                (tmp[y * 16 + x_base + 6] + 8) >> 4,
                (tmp[y * 16 + x_base + 5] + 8) >> 4,
                (tmp[y * 16 + x_base + 4] + 8) >> 4,
            );

            let sum_lo = _mm_add_epi32(d_lo, c_lo);
            let sum_hi = _mm_add_epi32(d_hi, c_hi);
            let clamped_lo = _mm_max_epi32(_mm_min_epi32(sum_lo, max_val), zero);
            let clamped_hi = _mm_max_epi32(_mm_min_epi32(sum_hi, max_val), zero);
            let packed = _mm_packus_epi32(clamped_lo, clamped_hi);

            storeu_128!(
                <&mut [u16; 8]>::try_from(&mut dst[dst_chunk_off..dst_chunk_off + 8]).unwrap(),
                packed
            );
        }
    }

    // Clear coefficients
    coeff[..64].fill(0);
}

/// FFI wrapper for 16x4 IDTX 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_identity_16x4_16bpc_avx2(
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

    inv_txfm_add_identity_identity_16x4_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 16x32 IDTX for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_identity_identity_16x32_16bpc_avx2_inner(
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

    // is_rect2 = true for 16x32, so apply sqrt(2) scaling
    let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

    // Row transform with shift=1 (16x32 intermediate shift)
    let rnd = 1;
    let shift = 1;

    for y in 0..32 {
        let mut scratch = [0i32; 16];
        for x in 0..16 {
            scratch[x] = rect2_scale(coeff[y + x * 32] as i32);
        }
        identity16_1d(&mut scratch[..16], 1, row_clip_min, row_clip_max);
        for x in 0..16 {
            tmp[y * 16 + x] = iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
        }
    }

    // Column transform: SIMD identity32 = *4 across 16 cols (2 chunks of 8), 32 rows
    {
        for cx_chunk in 0..2 {
            let cx = cx_chunk * 8;
            for i in 0..32 {
                let v = loadu_256!(&tmp[i * 16 + cx..i * 16 + cx + 8], [i32; 8]);
                let result = _mm256_slli_epi32::<2>(v);
                storeu_256!(&mut tmp[i * 16 + cx..i * 16 + cx + 8], [i32; 8], result);
            }
        }
    }

    // Add to destination
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi32(bitdepth_max);

    for y in 0..32 {
        let dst_off = y * stride_u16;

        // Process 16 pixels in two 8-pixel chunks
        for chunk in 0..2 {
            let x_base = chunk * 8;
            let dst_chunk_off = dst_off + x_base;

            let d =
                loadu_128!(<&[u16; 8]>::try_from(&dst[dst_chunk_off..dst_chunk_off + 8]).unwrap());
            let d_lo = _mm_unpacklo_epi16(d, zero);
            let d_hi = _mm_unpackhi_epi16(d, zero);

            // 16x32 uses >> 3 for final shift
            let c_lo = _mm_set_epi32(
                (tmp[y * 16 + x_base + 3] + 8) >> 4,
                (tmp[y * 16 + x_base + 2] + 8) >> 4,
                (tmp[y * 16 + x_base + 1] + 8) >> 4,
                (tmp[y * 16 + x_base + 0] + 8) >> 4,
            );
            let c_hi = _mm_set_epi32(
                (tmp[y * 16 + x_base + 7] + 8) >> 4,
                (tmp[y * 16 + x_base + 6] + 8) >> 4,
                (tmp[y * 16 + x_base + 5] + 8) >> 4,
                (tmp[y * 16 + x_base + 4] + 8) >> 4,
            );

            let sum_lo = _mm_add_epi32(d_lo, c_lo);
            let sum_hi = _mm_add_epi32(d_hi, c_hi);
            let clamped_lo = _mm_max_epi32(_mm_min_epi32(sum_lo, max_val), zero);
            let clamped_hi = _mm_max_epi32(_mm_min_epi32(sum_hi, max_val), zero);
            let packed = _mm_packus_epi32(clamped_lo, clamped_hi);

            storeu_128!(
                <&mut [u16; 8]>::try_from(&mut dst[dst_chunk_off..dst_chunk_off + 8]).unwrap(),
                packed
            );
        }
    }

    // Clear coefficients
    coeff[..512].fill(0);
}

/// FFI wrapper for 16x32 IDTX 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_identity_16x32_16bpc_avx2(
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

    inv_txfm_add_identity_identity_16x32_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 32x16 IDTX for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_identity_identity_32x16_16bpc_avx2_inner(
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

    // is_rect2 = true for 32x16, so apply sqrt(2) scaling
    let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

    // Row transform with shift=1 (32x16 intermediate shift)
    let rnd = 1;
    let shift = 1;

    for y in 0..16 {
        let mut scratch = [0i32; 32];
        for x in 0..32 {
            scratch[x] = rect2_scale(coeff[y + x * 16] as i32);
        }
        identity32_1d(&mut scratch[..32], 1, row_clip_min, row_clip_max);
        for x in 0..32 {
            tmp[y * 32 + x] = iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
        }
    }

    // Column transform: SIMD identity16 across 32 cols (4 chunks of 8), 16 rows
    {
        let c1697 = _mm256_set1_epi32(1697);
        let c1024 = _mm256_set1_epi32(1024);
        for cx_chunk in 0..4 {
            let cx = cx_chunk * 8;
            for i in 0..16 {
                let v = loadu_256!(&tmp[i * 32 + cx..i * 32 + cx + 8], [i32; 8]);
                let two_v = _mm256_slli_epi32::<1>(v);
                let mul = _mm256_mullo_epi32(v, c1697);
                let shifted = _mm256_srai_epi32::<11>(_mm256_add_epi32(mul, c1024));
                let result = _mm256_add_epi32(two_v, shifted);
                storeu_256!(&mut tmp[i * 32 + cx..i * 32 + cx + 8], [i32; 8], result);
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

    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi32(bitdepth_max);

    for y in 0..16 {
        let dst_off = y * stride_u16;

        // Process 32 pixels in four 8-pixel chunks
        for chunk in 0..4 {
            let x_base = chunk * 8;
            let dst_chunk_off = dst_off + x_base;

            let d =
                loadu_128!(<&[u16; 8]>::try_from(&dst[dst_chunk_off..dst_chunk_off + 8]).unwrap());
            let d_lo = _mm_unpacklo_epi16(d, zero);
            let d_hi = _mm_unpackhi_epi16(d, zero);

            // 32x16 uses >> 3 for final shift
            let c_lo = _mm_set_epi32(
                (tmp[y * 32 + x_base + 3] + 8) >> 4,
                (tmp[y * 32 + x_base + 2] + 8) >> 4,
                (tmp[y * 32 + x_base + 1] + 8) >> 4,
                (tmp[y * 32 + x_base + 0] + 8) >> 4,
            );
            let c_hi = _mm_set_epi32(
                (tmp[y * 32 + x_base + 7] + 8) >> 4,
                (tmp[y * 32 + x_base + 6] + 8) >> 4,
                (tmp[y * 32 + x_base + 5] + 8) >> 4,
                (tmp[y * 32 + x_base + 4] + 8) >> 4,
            );

            let sum_lo = _mm_add_epi32(d_lo, c_lo);
            let sum_hi = _mm_add_epi32(d_hi, c_hi);
            let clamped_lo = _mm_max_epi32(_mm_min_epi32(sum_lo, max_val), zero);
            let clamped_hi = _mm_max_epi32(_mm_min_epi32(sum_hi, max_val), zero);
            let packed = _mm_packus_epi32(clamped_lo, clamped_hi);

            storeu_128!(
                <&mut [u16; 8]>::try_from(&mut dst[dst_chunk_off..dst_chunk_off + 8]).unwrap(),
                packed
            );
        }
    }

    // Clear coefficients
    coeff[..512].fill(0);
}

/// FFI wrapper for 32x16 IDTX 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_identity_32x16_16bpc_avx2(
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

    inv_txfm_add_identity_identity_32x16_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 8x32 IDTX for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_identity_identity_8x32_16bpc_avx2_inner(
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

    // is_rect2 = false for 8x32 (aspect ratio 4:1), no rect2_scale

    // Row transform with shift=2 (8x32 intermediate shift)
    let rnd = 2;
    let shift = 2;

    for y in 0..32 {
        let mut scratch = [0i32; 8];
        for x in 0..8 {
            scratch[x] = coeff[y + x * 32] as i32;
        }
        identity8_1d(&mut scratch[..8], 1, row_clip_min, row_clip_max);
        for x in 0..8 {
            tmp[y * 8 + x] = iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
        }
    }

    // Column transform: SIMD identity32 = *4 across 8 cols (1 chunk), 32 rows
    {
        for i in 0..32 {
            let v = loadu_256!(&tmp[i * 8..i * 8 + 8], [i32; 8]);
            let result = _mm256_slli_epi32::<2>(v);
            storeu_256!(&mut tmp[i * 8..i * 8 + 8], [i32; 8], result);
        }
    }

    // Add to destination
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi32(bitdepth_max);

    for y in 0..32 {
        let dst_off = y * stride_u16;

        // Load destination (8 u16 = 16 bytes)
        let d = loadu_128!(<&[u16; 8]>::try_from(&dst[dst_off..dst_off + 8]).unwrap());
        let d_lo = _mm_unpacklo_epi16(d, zero);
        let d_hi = _mm_unpackhi_epi16(d, zero);

        // 8x32 uses >> 3 for final shift
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

/// FFI wrapper for 8x32 IDTX 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_identity_8x32_16bpc_avx2(
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

    inv_txfm_add_identity_identity_8x32_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 32x8 IDTX for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_identity_identity_32x8_16bpc_avx2_inner(
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

    // is_rect2 = false for 32x8 (aspect ratio 4:1), no rect2_scale

    // Row transform with shift=2 (32x8 intermediate shift)
    let rnd = 2;
    let shift = 2;

    for y in 0..8 {
        let mut scratch = [0i32; 32];
        for x in 0..32 {
            scratch[x] = coeff[y + x * 8] as i32;
        }
        identity32_1d(&mut scratch[..32], 1, row_clip_min, row_clip_max);
        for x in 0..32 {
            tmp[y * 32 + x] = iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
        }
    }

    // Column transform: SIMD identity8 = *2 across 32 cols (4 chunks of 8), 8 rows
    {
        for cx_chunk in 0..4 {
            let cx = cx_chunk * 8;
            for i in 0..8 {
                let v = loadu_256!(&tmp[i * 32 + cx..i * 32 + cx + 8], [i32; 8]);
                let result = _mm256_slli_epi32::<1>(v);
                storeu_256!(&mut tmp[i * 32 + cx..i * 32 + cx + 8], [i32; 8], result);
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

    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi32(bitdepth_max);

    for y in 0..8 {
        let dst_off = y * stride_u16;

        // Process 32 pixels in four 8-pixel chunks
        for chunk in 0..4 {
            let x_base = chunk * 8;
            let dst_chunk_off = dst_off + x_base;

            let d =
                loadu_128!(<&[u16; 8]>::try_from(&dst[dst_chunk_off..dst_chunk_off + 8]).unwrap());
            let d_lo = _mm_unpacklo_epi16(d, zero);
            let d_hi = _mm_unpackhi_epi16(d, zero);

            // 32x8 uses >> 3 for final shift
            let c_lo = _mm_set_epi32(
                (tmp[y * 32 + x_base + 3] + 8) >> 4,
                (tmp[y * 32 + x_base + 2] + 8) >> 4,
                (tmp[y * 32 + x_base + 1] + 8) >> 4,
                (tmp[y * 32 + x_base + 0] + 8) >> 4,
            );
            let c_hi = _mm_set_epi32(
                (tmp[y * 32 + x_base + 7] + 8) >> 4,
                (tmp[y * 32 + x_base + 6] + 8) >> 4,
                (tmp[y * 32 + x_base + 5] + 8) >> 4,
                (tmp[y * 32 + x_base + 4] + 8) >> 4,
            );

            let sum_lo = _mm_add_epi32(d_lo, c_lo);
            let sum_hi = _mm_add_epi32(d_hi, c_hi);
            let clamped_lo = _mm_max_epi32(_mm_min_epi32(sum_lo, max_val), zero);
            let clamped_hi = _mm_max_epi32(_mm_min_epi32(sum_hi, max_val), zero);
            let packed = _mm_packus_epi32(clamped_lo, clamped_hi);

            storeu_128!(
                <&mut [u16; 8]>::try_from(&mut dst[dst_chunk_off..dst_chunk_off + 8]).unwrap(),
                packed
            );
        }
    }

    // Clear coefficients
    coeff[..256].fill(0);
}

/// FFI wrapper for 32x8 IDTX 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_identity_32x8_16bpc_avx2(
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

    inv_txfm_add_identity_identity_32x8_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// Rectangular ADST/FLIPADST 16bpc
// ============================================================================

/// Macro for 4x8 transform variants 16bpc
macro_rules! impl_4x8_transform_16bpc {
    ($name:ident, $row_fn:ident, $col_fn:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[arcane]
        fn $name(
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
                $row_fn(&mut scratch[..4], 1, row_clip_min, row_clip_max);
                for x in 0..4 {
                    tmp[y * 4 + x] = iclip(scratch[x], col_clip_min, col_clip_max);
                }
            }

            // Column transform
            for x in 0..4 {
                $col_fn(&mut tmp[x..], 4, col_clip_min, col_clip_max);
            }

            // Add to destination
            let zero = _mm_setzero_si128();
            let max_val = _mm_set1_epi32(bitdepth_max);

            for y in 0..8 {
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
            coeff[..32].fill(0);
        }
    };
}

/// Macro for 8x4 transform variants 16bpc
macro_rules! impl_8x4_transform_16bpc {
    ($name:ident, $row_fn:ident, $col_fn:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[arcane]
        fn $name(
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
            let row_clip_min = (!bitdepth_max) << 7;
            let row_clip_max = !row_clip_min;
            let col_clip_min = (!bitdepth_max) << 5;
            let col_clip_max = !col_clip_min;
            let mut tmp = [0i32; 32];

            // is_rect2 = true for 8x4, so apply sqrt(2) scaling
            let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

            // Row transform (8 elements each, 4 rows)
            for y in 0..4 {
                let mut scratch = [0i32; 8];
                for x in 0..8 {
                    scratch[x] = rect2_scale(coeff[y + x * 4] as i32);
                }
                $row_fn(&mut scratch[..8], 1, row_clip_min, row_clip_max);
                for x in 0..8 {
                    tmp[y * 8 + x] = iclip(scratch[x], col_clip_min, col_clip_max);
                }
            }

            // Column transform
            for x in 0..8 {
                $col_fn(&mut tmp[x..], 8, col_clip_min, col_clip_max);
            }

            // Add to destination
            let zero = _mm_setzero_si128();
            let max_val = _mm_set1_epi32(bitdepth_max);

            for y in 0..4 {
                let dst_off = y * stride_u16;

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
            coeff[..32].fill(0);
        }
    };
}

/// Macro for FFI wrapper 16bpc
macro_rules! impl_ffi_wrapper_16bpc {
    ($name:ident, $inner:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2")]
        #[cfg(feature = "asm")]
        pub unsafe extern "C" fn $name(
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

// 4x8 ADST/FLIPADST variants 16bpc
impl_4x8_transform_16bpc!(
    inv_txfm_add_adst_dct_4x8_16bpc_avx2_inner,
    adst4_1d,
    dct8_1d
);
impl_4x8_transform_16bpc!(
    inv_txfm_add_dct_adst_4x8_16bpc_avx2_inner,
    dct4_1d,
    adst8_1d
);
impl_4x8_transform_16bpc!(
    inv_txfm_add_adst_adst_4x8_16bpc_avx2_inner,
    adst4_1d,
    adst8_1d
);
impl_4x8_transform_16bpc!(
    inv_txfm_add_flipadst_dct_4x8_16bpc_avx2_inner,
    flipadst4_1d,
    dct8_1d
);
impl_4x8_transform_16bpc!(
    inv_txfm_add_dct_flipadst_4x8_16bpc_avx2_inner,
    dct4_1d,
    flipadst8_1d
);
impl_4x8_transform_16bpc!(
    inv_txfm_add_flipadst_flipadst_4x8_16bpc_avx2_inner,
    flipadst4_1d,
    flipadst8_1d
);
impl_4x8_transform_16bpc!(
    inv_txfm_add_adst_flipadst_4x8_16bpc_avx2_inner,
    adst4_1d,
    flipadst8_1d
);
impl_4x8_transform_16bpc!(
    inv_txfm_add_flipadst_adst_4x8_16bpc_avx2_inner,
    flipadst4_1d,
    adst8_1d
);

impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_dct_4x8_16bpc_avx2,
    inv_txfm_add_adst_dct_4x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_adst_4x8_16bpc_avx2,
    inv_txfm_add_dct_adst_4x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_adst_4x8_16bpc_avx2,
    inv_txfm_add_adst_adst_4x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_dct_4x8_16bpc_avx2,
    inv_txfm_add_flipadst_dct_4x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_flipadst_4x8_16bpc_avx2,
    inv_txfm_add_dct_flipadst_4x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_flipadst_4x8_16bpc_avx2,
    inv_txfm_add_flipadst_flipadst_4x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_flipadst_4x8_16bpc_avx2,
    inv_txfm_add_adst_flipadst_4x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_adst_4x8_16bpc_avx2,
    inv_txfm_add_flipadst_adst_4x8_16bpc_avx2_inner
);

// 8x4 ADST/FLIPADST variants 16bpc
impl_8x4_transform_16bpc!(
    inv_txfm_add_adst_dct_8x4_16bpc_avx2_inner,
    adst8_1d,
    dct4_1d
);
impl_8x4_transform_16bpc!(
    inv_txfm_add_dct_adst_8x4_16bpc_avx2_inner,
    dct8_1d,
    adst4_1d
);
impl_8x4_transform_16bpc!(
    inv_txfm_add_adst_adst_8x4_16bpc_avx2_inner,
    adst8_1d,
    adst4_1d
);
impl_8x4_transform_16bpc!(
    inv_txfm_add_flipadst_dct_8x4_16bpc_avx2_inner,
    flipadst8_1d,
    dct4_1d
);
impl_8x4_transform_16bpc!(
    inv_txfm_add_dct_flipadst_8x4_16bpc_avx2_inner,
    dct8_1d,
    flipadst4_1d
);
impl_8x4_transform_16bpc!(
    inv_txfm_add_flipadst_flipadst_8x4_16bpc_avx2_inner,
    flipadst8_1d,
    flipadst4_1d
);
impl_8x4_transform_16bpc!(
    inv_txfm_add_adst_flipadst_8x4_16bpc_avx2_inner,
    adst8_1d,
    flipadst4_1d
);
impl_8x4_transform_16bpc!(
    inv_txfm_add_flipadst_adst_8x4_16bpc_avx2_inner,
    flipadst8_1d,
    adst4_1d
);

impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_dct_8x4_16bpc_avx2,
    inv_txfm_add_adst_dct_8x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_adst_8x4_16bpc_avx2,
    inv_txfm_add_dct_adst_8x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_adst_8x4_16bpc_avx2,
    inv_txfm_add_adst_adst_8x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_dct_8x4_16bpc_avx2,
    inv_txfm_add_flipadst_dct_8x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_flipadst_8x4_16bpc_avx2,
    inv_txfm_add_dct_flipadst_8x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_flipadst_8x4_16bpc_avx2,
    inv_txfm_add_flipadst_flipadst_8x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_flipadst_8x4_16bpc_avx2,
    inv_txfm_add_adst_flipadst_8x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_adst_8x4_16bpc_avx2,
    inv_txfm_add_flipadst_adst_8x4_16bpc_avx2_inner
);

/// Macro for 8x16 transform variants 16bpc
macro_rules! impl_8x16_transform_16bpc {
    ($name:ident, $row_fn:ident, $col_fn:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[arcane]
        fn $name(
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
            let row_clip_min = (!bitdepth_max) << 7;
            let row_clip_max = !row_clip_min;
            let col_clip_min = (!bitdepth_max) << 5;
            let col_clip_max = !col_clip_min;
            let mut tmp = [0i32; 128];

            // is_rect2 = true for 8x16
            let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

            // Row transform with shift=1 (8x16 intermediate shift)
            let rnd = 1;
            let shift = 1;

            for y in 0..16 {
                let mut scratch = [0i32; 8];
                for x in 0..8 {
                    scratch[x] = rect2_scale(coeff[y + x * 16] as i32);
                }
                $row_fn(&mut scratch[..8], 1, row_clip_min, row_clip_max);
                for x in 0..8 {
                    tmp[y * 8 + x] = iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
                }
            }

            // Column transform
            for x in 0..8 {
                $col_fn(&mut tmp[x..], 8, col_clip_min, col_clip_max);
            }

            // Add to destination
            let zero = _mm_setzero_si128();
            let max_val = _mm_set1_epi32(bitdepth_max);

            for y in 0..16 {
                let dst_off = y * stride_u16;

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
    };
}

/// Macro for 16x8 transform variants 16bpc
macro_rules! impl_16x8_transform_16bpc {
    ($name:ident, $row_fn:ident, $col_fn:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[arcane]
        fn $name(
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
            let row_clip_min = (!bitdepth_max) << 7;
            let row_clip_max = !row_clip_min;
            let col_clip_min = (!bitdepth_max) << 5;
            let col_clip_max = !col_clip_min;
            let mut tmp = [0i32; 128];

            // is_rect2 = true for 16x8
            let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

            // Row transform with shift=1 (16x8 intermediate shift)
            let rnd = 1;
            let shift = 1;

            for y in 0..8 {
                let mut scratch = [0i32; 16];
                for x in 0..16 {
                    scratch[x] = rect2_scale(coeff[y + x * 8] as i32);
                }
                $row_fn(&mut scratch[..16], 1, row_clip_min, row_clip_max);
                for x in 0..16 {
                    tmp[y * 16 + x] =
                        iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
                }
            }

            // Column transform
            for x in 0..16 {
                $col_fn(&mut tmp[x..], 16, col_clip_min, col_clip_max);
            }

            // Add to destination
            let zero = _mm_setzero_si128();
            let max_val = _mm_set1_epi32(bitdepth_max);

            for y in 0..8 {
                let dst_off = y * stride_u16;

                for chunk in 0..2 {
                    let x_base = chunk * 8;
                    let dst_chunk_off = dst_off + x_base;

                    let d = loadu_128!(
                        <&[u16; 8]>::try_from(&dst[dst_chunk_off..dst_chunk_off + 8]).unwrap()
                    );
                    let d_lo = _mm_unpacklo_epi16(d, zero);
                    let d_hi = _mm_unpackhi_epi16(d, zero);

                    let c_lo = _mm_set_epi32(
                        (tmp[y * 16 + x_base + 3] + 8) >> 4,
                        (tmp[y * 16 + x_base + 2] + 8) >> 4,
                        (tmp[y * 16 + x_base + 1] + 8) >> 4,
                        (tmp[y * 16 + x_base + 0] + 8) >> 4,
                    );
                    let c_hi = _mm_set_epi32(
                        (tmp[y * 16 + x_base + 7] + 8) >> 4,
                        (tmp[y * 16 + x_base + 6] + 8) >> 4,
                        (tmp[y * 16 + x_base + 5] + 8) >> 4,
                        (tmp[y * 16 + x_base + 4] + 8) >> 4,
                    );

                    let sum_lo = _mm_add_epi32(d_lo, c_lo);
                    let sum_hi = _mm_add_epi32(d_hi, c_hi);
                    let clamped_lo = _mm_max_epi32(_mm_min_epi32(sum_lo, max_val), zero);
                    let clamped_hi = _mm_max_epi32(_mm_min_epi32(sum_hi, max_val), zero);
                    let packed = _mm_packus_epi32(clamped_lo, clamped_hi);

                    storeu_128!(
                        <&mut [u16; 8]>::try_from(&mut dst[dst_chunk_off..dst_chunk_off + 8])
                            .unwrap(),
                        packed
                    );
                }
            }

            // Clear coefficients
            coeff[..128].fill(0);
        }
    };
}

// 8x16 ADST/FLIPADST variants 16bpc
impl_8x16_transform_16bpc!(
    inv_txfm_add_adst_dct_8x16_16bpc_avx2_inner,
    adst8_1d,
    dct16_1d
);
impl_8x16_transform_16bpc!(
    inv_txfm_add_dct_adst_8x16_16bpc_avx2_inner,
    dct8_1d,
    adst16_1d
);
impl_8x16_transform_16bpc!(
    inv_txfm_add_adst_adst_8x16_16bpc_avx2_inner,
    adst8_1d,
    adst16_1d
);
impl_8x16_transform_16bpc!(
    inv_txfm_add_flipadst_dct_8x16_16bpc_avx2_inner,
    flipadst8_1d,
    dct16_1d
);
impl_8x16_transform_16bpc!(
    inv_txfm_add_dct_flipadst_8x16_16bpc_avx2_inner,
    dct8_1d,
    flipadst16_1d
);
impl_8x16_transform_16bpc!(
    inv_txfm_add_flipadst_flipadst_8x16_16bpc_avx2_inner,
    flipadst8_1d,
    flipadst16_1d
);
impl_8x16_transform_16bpc!(
    inv_txfm_add_adst_flipadst_8x16_16bpc_avx2_inner,
    adst8_1d,
    flipadst16_1d
);
impl_8x16_transform_16bpc!(
    inv_txfm_add_flipadst_adst_8x16_16bpc_avx2_inner,
    flipadst8_1d,
    adst16_1d
);

impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_dct_8x16_16bpc_avx2,
    inv_txfm_add_adst_dct_8x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_adst_8x16_16bpc_avx2,
    inv_txfm_add_dct_adst_8x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_adst_8x16_16bpc_avx2,
    inv_txfm_add_adst_adst_8x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_dct_8x16_16bpc_avx2,
    inv_txfm_add_flipadst_dct_8x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_flipadst_8x16_16bpc_avx2,
    inv_txfm_add_dct_flipadst_8x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_flipadst_8x16_16bpc_avx2,
    inv_txfm_add_flipadst_flipadst_8x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_flipadst_8x16_16bpc_avx2,
    inv_txfm_add_adst_flipadst_8x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_adst_8x16_16bpc_avx2,
    inv_txfm_add_flipadst_adst_8x16_16bpc_avx2_inner
);

// 16x8 ADST/FLIPADST variants 16bpc
impl_16x8_transform_16bpc!(
    inv_txfm_add_adst_dct_16x8_16bpc_avx2_inner,
    adst16_1d,
    dct8_1d
);
impl_16x8_transform_16bpc!(
    inv_txfm_add_dct_adst_16x8_16bpc_avx2_inner,
    dct16_1d,
    adst8_1d
);
impl_16x8_transform_16bpc!(
    inv_txfm_add_adst_adst_16x8_16bpc_avx2_inner,
    adst16_1d,
    adst8_1d
);
impl_16x8_transform_16bpc!(
    inv_txfm_add_flipadst_dct_16x8_16bpc_avx2_inner,
    flipadst16_1d,
    dct8_1d
);
impl_16x8_transform_16bpc!(
    inv_txfm_add_dct_flipadst_16x8_16bpc_avx2_inner,
    dct16_1d,
    flipadst8_1d
);
impl_16x8_transform_16bpc!(
    inv_txfm_add_flipadst_flipadst_16x8_16bpc_avx2_inner,
    flipadst16_1d,
    flipadst8_1d
);
impl_16x8_transform_16bpc!(
    inv_txfm_add_adst_flipadst_16x8_16bpc_avx2_inner,
    adst16_1d,
    flipadst8_1d
);
impl_16x8_transform_16bpc!(
    inv_txfm_add_flipadst_adst_16x8_16bpc_avx2_inner,
    flipadst16_1d,
    adst8_1d
);

impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_dct_16x8_16bpc_avx2,
    inv_txfm_add_adst_dct_16x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_adst_16x8_16bpc_avx2,
    inv_txfm_add_dct_adst_16x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_adst_16x8_16bpc_avx2,
    inv_txfm_add_adst_adst_16x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_dct_16x8_16bpc_avx2,
    inv_txfm_add_flipadst_dct_16x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_flipadst_16x8_16bpc_avx2,
    inv_txfm_add_dct_flipadst_16x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_flipadst_16x8_16bpc_avx2,
    inv_txfm_add_flipadst_flipadst_16x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_flipadst_16x8_16bpc_avx2,
    inv_txfm_add_adst_flipadst_16x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_adst_16x8_16bpc_avx2,
    inv_txfm_add_flipadst_adst_16x8_16bpc_avx2_inner
);

/// Macro for 4x16 transform variants 16bpc
macro_rules! impl_4x16_transform_16bpc {
    ($name:ident, $row_fn:ident, $col_fn:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[arcane]
        fn $name(
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
            let row_clip_min = (!bitdepth_max) << 7;
            let row_clip_max = !row_clip_min;
            let col_clip_min = (!bitdepth_max) << 5;
            let col_clip_max = !col_clip_min;
            let mut tmp = [0i32; 64];

            // is_rect2 = false for 4x16 (aspect ratio 4:1), no rect2_scale

            // Row transform with shift=1 (4x16 intermediate shift)
            let rnd = 1;
            let shift = 1;

            for y in 0..16 {
                let mut scratch = [0i32; 4];
                for x in 0..4 {
                    scratch[x] = coeff[y + x * 16] as i32;
                }
                $row_fn(&mut scratch[..4], 1, row_clip_min, row_clip_max);
                for x in 0..4 {
                    tmp[y * 4 + x] = iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
                }
            }

            // Column transform
            for x in 0..4 {
                $col_fn(&mut tmp[x..], 4, col_clip_min, col_clip_max);
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
    };
}

/// Macro for 16x4 transform variants 16bpc
macro_rules! impl_16x4_transform_16bpc {
    ($name:ident, $row_fn:ident, $col_fn:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[arcane]
        fn $name(
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
            let row_clip_min = (!bitdepth_max) << 7;
            let row_clip_max = !row_clip_min;
            let col_clip_min = (!bitdepth_max) << 5;
            let col_clip_max = !col_clip_min;
            let mut tmp = [0i32; 64];

            // is_rect2 = false for 16x4 (aspect ratio 4:1), no rect2_scale

            // Row transform with shift=1 (16x4 intermediate shift)
            let rnd = 1;
            let shift = 1;

            for y in 0..4 {
                let mut scratch = [0i32; 16];
                for x in 0..16 {
                    scratch[x] = coeff[y + x * 4] as i32;
                }
                $row_fn(&mut scratch[..16], 1, row_clip_min, row_clip_max);
                for x in 0..16 {
                    tmp[y * 16 + x] =
                        iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
                }
            }

            // Column transform
            for x in 0..16 {
                $col_fn(&mut tmp[x..], 16, col_clip_min, col_clip_max);
            }

            // Add to destination
            let zero = _mm_setzero_si128();
            let max_val = _mm_set1_epi32(bitdepth_max);

            for y in 0..4 {
                let dst_off = y * stride_u16;

                for chunk in 0..2 {
                    let x_base = chunk * 8;
                    let dst_chunk_off = dst_off + x_base;

                    let d = loadu_128!(
                        <&[u16; 8]>::try_from(&dst[dst_chunk_off..dst_chunk_off + 8]).unwrap()
                    );
                    let d_lo = _mm_unpacklo_epi16(d, zero);
                    let d_hi = _mm_unpackhi_epi16(d, zero);

                    let c_lo = _mm_set_epi32(
                        (tmp[y * 16 + x_base + 3] + 8) >> 4,
                        (tmp[y * 16 + x_base + 2] + 8) >> 4,
                        (tmp[y * 16 + x_base + 1] + 8) >> 4,
                        (tmp[y * 16 + x_base + 0] + 8) >> 4,
                    );
                    let c_hi = _mm_set_epi32(
                        (tmp[y * 16 + x_base + 7] + 8) >> 4,
                        (tmp[y * 16 + x_base + 6] + 8) >> 4,
                        (tmp[y * 16 + x_base + 5] + 8) >> 4,
                        (tmp[y * 16 + x_base + 4] + 8) >> 4,
                    );

                    let sum_lo = _mm_add_epi32(d_lo, c_lo);
                    let sum_hi = _mm_add_epi32(d_hi, c_hi);
                    let clamped_lo = _mm_max_epi32(_mm_min_epi32(sum_lo, max_val), zero);
                    let clamped_hi = _mm_max_epi32(_mm_min_epi32(sum_hi, max_val), zero);
                    let packed = _mm_packus_epi32(clamped_lo, clamped_hi);

                    storeu_128!(
                        <&mut [u16; 8]>::try_from(&mut dst[dst_chunk_off..dst_chunk_off + 8])
                            .unwrap(),
                        packed
                    );
                }
            }

            // Clear coefficients
            coeff[..64].fill(0);
        }
    };
}

// 4x16 ADST/FLIPADST variants 16bpc
impl_4x16_transform_16bpc!(
    inv_txfm_add_adst_dct_4x16_16bpc_avx2_inner,
    adst4_1d,
    dct16_1d
);
impl_4x16_transform_16bpc!(
    inv_txfm_add_dct_adst_4x16_16bpc_avx2_inner,
    dct4_1d,
    adst16_1d
);
impl_4x16_transform_16bpc!(
    inv_txfm_add_adst_adst_4x16_16bpc_avx2_inner,
    adst4_1d,
    adst16_1d
);
impl_4x16_transform_16bpc!(
    inv_txfm_add_flipadst_dct_4x16_16bpc_avx2_inner,
    flipadst4_1d,
    dct16_1d
);
impl_4x16_transform_16bpc!(
    inv_txfm_add_dct_flipadst_4x16_16bpc_avx2_inner,
    dct4_1d,
    flipadst16_1d
);
impl_4x16_transform_16bpc!(
    inv_txfm_add_flipadst_flipadst_4x16_16bpc_avx2_inner,
    flipadst4_1d,
    flipadst16_1d
);
impl_4x16_transform_16bpc!(
    inv_txfm_add_adst_flipadst_4x16_16bpc_avx2_inner,
    adst4_1d,
    flipadst16_1d
);
impl_4x16_transform_16bpc!(
    inv_txfm_add_flipadst_adst_4x16_16bpc_avx2_inner,
    flipadst4_1d,
    adst16_1d
);

impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_dct_4x16_16bpc_avx2,
    inv_txfm_add_adst_dct_4x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_adst_4x16_16bpc_avx2,
    inv_txfm_add_dct_adst_4x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_adst_4x16_16bpc_avx2,
    inv_txfm_add_adst_adst_4x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_dct_4x16_16bpc_avx2,
    inv_txfm_add_flipadst_dct_4x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_flipadst_4x16_16bpc_avx2,
    inv_txfm_add_dct_flipadst_4x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_flipadst_4x16_16bpc_avx2,
    inv_txfm_add_flipadst_flipadst_4x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_flipadst_4x16_16bpc_avx2,
    inv_txfm_add_adst_flipadst_4x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_adst_4x16_16bpc_avx2,
    inv_txfm_add_flipadst_adst_4x16_16bpc_avx2_inner
);

// 16x4 ADST/FLIPADST variants 16bpc
impl_16x4_transform_16bpc!(
    inv_txfm_add_adst_dct_16x4_16bpc_avx2_inner,
    adst16_1d,
    dct4_1d
);
impl_16x4_transform_16bpc!(
    inv_txfm_add_dct_adst_16x4_16bpc_avx2_inner,
    dct16_1d,
    adst4_1d
);
impl_16x4_transform_16bpc!(
    inv_txfm_add_adst_adst_16x4_16bpc_avx2_inner,
    adst16_1d,
    adst4_1d
);
impl_16x4_transform_16bpc!(
    inv_txfm_add_flipadst_dct_16x4_16bpc_avx2_inner,
    flipadst16_1d,
    dct4_1d
);
impl_16x4_transform_16bpc!(
    inv_txfm_add_dct_flipadst_16x4_16bpc_avx2_inner,
    dct16_1d,
    flipadst4_1d
);
impl_16x4_transform_16bpc!(
    inv_txfm_add_flipadst_flipadst_16x4_16bpc_avx2_inner,
    flipadst16_1d,
    flipadst4_1d
);
impl_16x4_transform_16bpc!(
    inv_txfm_add_adst_flipadst_16x4_16bpc_avx2_inner,
    adst16_1d,
    flipadst4_1d
);
impl_16x4_transform_16bpc!(
    inv_txfm_add_flipadst_adst_16x4_16bpc_avx2_inner,
    flipadst16_1d,
    adst4_1d
);

impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_dct_16x4_16bpc_avx2,
    inv_txfm_add_adst_dct_16x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_adst_16x4_16bpc_avx2,
    inv_txfm_add_dct_adst_16x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_adst_16x4_16bpc_avx2,
    inv_txfm_add_adst_adst_16x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_dct_16x4_16bpc_avx2,
    inv_txfm_add_flipadst_dct_16x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_flipadst_16x4_16bpc_avx2,
    inv_txfm_add_dct_flipadst_16x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_flipadst_16x4_16bpc_avx2,
    inv_txfm_add_flipadst_flipadst_16x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_flipadst_16x4_16bpc_avx2,
    inv_txfm_add_adst_flipadst_16x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_adst_16x4_16bpc_avx2,
    inv_txfm_add_flipadst_adst_16x4_16bpc_avx2_inner
);

// ============================================================================
// Hybrid identity transforms for 16bpc (H_DCT, V_DCT, H_ADST, V_ADST, etc.)
// ============================================================================

// 4x8 hybrid identity transforms 16bpc
impl_4x8_transform_16bpc!(
    inv_txfm_add_identity_dct_4x8_16bpc_avx2_inner,
    identity4_1d,
    dct8_1d
);
impl_4x8_transform_16bpc!(
    inv_txfm_add_dct_identity_4x8_16bpc_avx2_inner,
    dct4_1d,
    identity8_1d
);
impl_4x8_transform_16bpc!(
    inv_txfm_add_identity_adst_4x8_16bpc_avx2_inner,
    identity4_1d,
    adst8_1d
);
impl_4x8_transform_16bpc!(
    inv_txfm_add_adst_identity_4x8_16bpc_avx2_inner,
    adst4_1d,
    identity8_1d
);
impl_4x8_transform_16bpc!(
    inv_txfm_add_identity_flipadst_4x8_16bpc_avx2_inner,
    identity4_1d,
    flipadst8_1d
);
impl_4x8_transform_16bpc!(
    inv_txfm_add_flipadst_identity_4x8_16bpc_avx2_inner,
    flipadst4_1d,
    identity8_1d
);

impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_dct_4x8_16bpc_avx2,
    inv_txfm_add_identity_dct_4x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_identity_4x8_16bpc_avx2,
    inv_txfm_add_dct_identity_4x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_adst_4x8_16bpc_avx2,
    inv_txfm_add_identity_adst_4x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_identity_4x8_16bpc_avx2,
    inv_txfm_add_adst_identity_4x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_flipadst_4x8_16bpc_avx2,
    inv_txfm_add_identity_flipadst_4x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_identity_4x8_16bpc_avx2,
    inv_txfm_add_flipadst_identity_4x8_16bpc_avx2_inner
);

// 8x4 hybrid identity transforms 16bpc
impl_8x4_transform_16bpc!(
    inv_txfm_add_identity_dct_8x4_16bpc_avx2_inner,
    identity8_1d,
    dct4_1d
);
impl_8x4_transform_16bpc!(
    inv_txfm_add_dct_identity_8x4_16bpc_avx2_inner,
    dct8_1d,
    identity4_1d
);
impl_8x4_transform_16bpc!(
    inv_txfm_add_identity_adst_8x4_16bpc_avx2_inner,
    identity8_1d,
    adst4_1d
);
impl_8x4_transform_16bpc!(
    inv_txfm_add_adst_identity_8x4_16bpc_avx2_inner,
    adst8_1d,
    identity4_1d
);
impl_8x4_transform_16bpc!(
    inv_txfm_add_identity_flipadst_8x4_16bpc_avx2_inner,
    identity8_1d,
    flipadst4_1d
);
impl_8x4_transform_16bpc!(
    inv_txfm_add_flipadst_identity_8x4_16bpc_avx2_inner,
    flipadst8_1d,
    identity4_1d
);

impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_dct_8x4_16bpc_avx2,
    inv_txfm_add_identity_dct_8x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_identity_8x4_16bpc_avx2,
    inv_txfm_add_dct_identity_8x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_adst_8x4_16bpc_avx2,
    inv_txfm_add_identity_adst_8x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_identity_8x4_16bpc_avx2,
    inv_txfm_add_adst_identity_8x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_flipadst_8x4_16bpc_avx2,
    inv_txfm_add_identity_flipadst_8x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_identity_8x4_16bpc_avx2,
    inv_txfm_add_flipadst_identity_8x4_16bpc_avx2_inner
);

// 8x16 hybrid identity transforms 16bpc
impl_8x16_transform_16bpc!(
    inv_txfm_add_identity_dct_8x16_16bpc_avx2_inner,
    identity8_1d,
    dct16_1d
);
impl_8x16_transform_16bpc!(
    inv_txfm_add_dct_identity_8x16_16bpc_avx2_inner,
    dct8_1d,
    identity16_1d
);
impl_8x16_transform_16bpc!(
    inv_txfm_add_identity_adst_8x16_16bpc_avx2_inner,
    identity8_1d,
    adst16_1d
);
impl_8x16_transform_16bpc!(
    inv_txfm_add_adst_identity_8x16_16bpc_avx2_inner,
    adst8_1d,
    identity16_1d
);
impl_8x16_transform_16bpc!(
    inv_txfm_add_identity_flipadst_8x16_16bpc_avx2_inner,
    identity8_1d,
    flipadst16_1d
);
impl_8x16_transform_16bpc!(
    inv_txfm_add_flipadst_identity_8x16_16bpc_avx2_inner,
    flipadst8_1d,
    identity16_1d
);

impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_dct_8x16_16bpc_avx2,
    inv_txfm_add_identity_dct_8x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_identity_8x16_16bpc_avx2,
    inv_txfm_add_dct_identity_8x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_adst_8x16_16bpc_avx2,
    inv_txfm_add_identity_adst_8x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_identity_8x16_16bpc_avx2,
    inv_txfm_add_adst_identity_8x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_flipadst_8x16_16bpc_avx2,
    inv_txfm_add_identity_flipadst_8x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_identity_8x16_16bpc_avx2,
    inv_txfm_add_flipadst_identity_8x16_16bpc_avx2_inner
);

// 16x8 hybrid identity transforms 16bpc
impl_16x8_transform_16bpc!(
    inv_txfm_add_identity_dct_16x8_16bpc_avx2_inner,
    identity16_1d,
    dct8_1d
);
impl_16x8_transform_16bpc!(
    inv_txfm_add_dct_identity_16x8_16bpc_avx2_inner,
    dct16_1d,
    identity8_1d
);
impl_16x8_transform_16bpc!(
    inv_txfm_add_identity_adst_16x8_16bpc_avx2_inner,
    identity16_1d,
    adst8_1d
);
impl_16x8_transform_16bpc!(
    inv_txfm_add_adst_identity_16x8_16bpc_avx2_inner,
    adst16_1d,
    identity8_1d
);
impl_16x8_transform_16bpc!(
    inv_txfm_add_identity_flipadst_16x8_16bpc_avx2_inner,
    identity16_1d,
    flipadst8_1d
);
impl_16x8_transform_16bpc!(
    inv_txfm_add_flipadst_identity_16x8_16bpc_avx2_inner,
    flipadst16_1d,
    identity8_1d
);

impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_dct_16x8_16bpc_avx2,
    inv_txfm_add_identity_dct_16x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_identity_16x8_16bpc_avx2,
    inv_txfm_add_dct_identity_16x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_adst_16x8_16bpc_avx2,
    inv_txfm_add_identity_adst_16x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_identity_16x8_16bpc_avx2,
    inv_txfm_add_adst_identity_16x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_flipadst_16x8_16bpc_avx2,
    inv_txfm_add_identity_flipadst_16x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_identity_16x8_16bpc_avx2,
    inv_txfm_add_flipadst_identity_16x8_16bpc_avx2_inner
);

// 4x16 hybrid identity transforms 16bpc
impl_4x16_transform_16bpc!(
    inv_txfm_add_identity_dct_4x16_16bpc_avx2_inner,
    identity4_1d,
    dct16_1d
);
impl_4x16_transform_16bpc!(
    inv_txfm_add_dct_identity_4x16_16bpc_avx2_inner,
    dct4_1d,
    identity16_1d
);
impl_4x16_transform_16bpc!(
    inv_txfm_add_identity_adst_4x16_16bpc_avx2_inner,
    identity4_1d,
    adst16_1d
);
impl_4x16_transform_16bpc!(
    inv_txfm_add_adst_identity_4x16_16bpc_avx2_inner,
    adst4_1d,
    identity16_1d
);
impl_4x16_transform_16bpc!(
    inv_txfm_add_identity_flipadst_4x16_16bpc_avx2_inner,
    identity4_1d,
    flipadst16_1d
);
impl_4x16_transform_16bpc!(
    inv_txfm_add_flipadst_identity_4x16_16bpc_avx2_inner,
    flipadst4_1d,
    identity16_1d
);

impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_dct_4x16_16bpc_avx2,
    inv_txfm_add_identity_dct_4x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_identity_4x16_16bpc_avx2,
    inv_txfm_add_dct_identity_4x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_adst_4x16_16bpc_avx2,
    inv_txfm_add_identity_adst_4x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_identity_4x16_16bpc_avx2,
    inv_txfm_add_adst_identity_4x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_flipadst_4x16_16bpc_avx2,
    inv_txfm_add_identity_flipadst_4x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_identity_4x16_16bpc_avx2,
    inv_txfm_add_flipadst_identity_4x16_16bpc_avx2_inner
);

// 16x4 hybrid identity transforms 16bpc
impl_16x4_transform_16bpc!(
    inv_txfm_add_identity_dct_16x4_16bpc_avx2_inner,
    identity16_1d,
    dct4_1d
);
impl_16x4_transform_16bpc!(
    inv_txfm_add_dct_identity_16x4_16bpc_avx2_inner,
    dct16_1d,
    identity4_1d
);
impl_16x4_transform_16bpc!(
    inv_txfm_add_identity_adst_16x4_16bpc_avx2_inner,
    identity16_1d,
    adst4_1d
);
impl_16x4_transform_16bpc!(
    inv_txfm_add_adst_identity_16x4_16bpc_avx2_inner,
    adst16_1d,
    identity4_1d
);
impl_16x4_transform_16bpc!(
    inv_txfm_add_identity_flipadst_16x4_16bpc_avx2_inner,
    identity16_1d,
    flipadst4_1d
);
impl_16x4_transform_16bpc!(
    inv_txfm_add_flipadst_identity_16x4_16bpc_avx2_inner,
    flipadst16_1d,
    identity4_1d
);

impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_dct_16x4_16bpc_avx2,
    inv_txfm_add_identity_dct_16x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_identity_16x4_16bpc_avx2,
    inv_txfm_add_dct_identity_16x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_adst_16x4_16bpc_avx2,
    inv_txfm_add_identity_adst_16x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_identity_16x4_16bpc_avx2,
    inv_txfm_add_adst_identity_16x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_flipadst_16x4_16bpc_avx2,
    inv_txfm_add_identity_flipadst_16x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_identity_16x4_16bpc_avx2,
    inv_txfm_add_flipadst_identity_16x4_16bpc_avx2_inner
);

// ============================================================================
// Square hybrid identity transforms for 16bpc
// ============================================================================

/// Macro for 8x8 transform variants 16bpc
#[allow(unused_macros)]
macro_rules! impl_8x8_transform_16bpc {
    ($name:ident, $row_fn:ident, $col_fn:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[arcane]
        fn $name(
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
            let row_clip_min = (!bitdepth_max) << 7;
            let row_clip_max = !row_clip_min;
            let col_clip_min = (!bitdepth_max) << 5;
            let col_clip_max = !col_clip_min;
            let mut tmp = [0i32; 64];

            // No rect2_scale for square transforms

            // Row transform with shift=1 (8x8 intermediate shift)
            let rnd = 1;
            let shift = 1;

            for y in 0..8 {
                let mut scratch = [0i32; 8];
                for x in 0..8 {
                    scratch[x] = coeff[y + x * 8] as i32;
                }
                $row_fn(&mut scratch[..8], 1, row_clip_min, row_clip_max);
                for x in 0..8 {
                    tmp[y * 8 + x] = iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
                }
            }

            // Column transform
            for x in 0..8 {
                $col_fn(&mut tmp[x..], 8, col_clip_min, col_clip_max);
            }

            // Add to destination
            let zero = _mm_setzero_si128();
            let max_val = _mm_set1_epi32(bitdepth_max);

            for y in 0..8 {
                let dst_off = y * stride_u16;

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
            coeff[..64].fill(0);
        }
    };
}

/// 8x8 16bpc strided-style variant with SIMD column pass.
/// Uses scalar strided row 1D + SIMD col pass over flat tmp.
macro_rules! impl_8x8_transform_16bpc_strided_simd_col {
    ($name:ident, $row_fn:ident, $simd_col_fn:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[arcane]
        fn $name(
            _token: Desktop64,
            dst: &mut [u16],
            dst_stride: usize,
            coeff: &mut [i16],
            _eob: i32,
            bitdepth_max: i32,
        ) {
            use crate::src::safe_simd::pixel_access::{loadu_128, storeu_128};
            let mut dst = dst.flex_mut();
            let mut coeff = coeff.flex_mut();
            let stride_u16 = dst_stride / 2;
            let row_clip_min = (!bitdepth_max) << 7;
            let row_clip_max = !row_clip_min;
            let col_clip_min = (!bitdepth_max) << 5;
            let col_clip_max = !col_clip_min;
            let mut tmp = [0i32; 64];

            let rnd = 1;
            let shift = 1;
            for y in 0..8 {
                let mut scratch = [0i32; 8];
                for x in 0..8 {
                    scratch[x] = coeff[y + x * 8] as i32;
                }
                $row_fn(&mut scratch[..8], 1, row_clip_min, row_clip_max);
                for x in 0..8 {
                    tmp[y * 8 + x] = iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
                }
            }

            // SIMD column pass (8 cols x 8 rows, single chunk)
            {
                let min_v = _mm256_set1_epi32(col_clip_min);
                let max_v = _mm256_set1_epi32(col_clip_max);
                let mut v = [_mm256_setzero_si256(); 8];
                for i in 0..8 {
                    v[i] = loadu_256!(&tmp[i * 8..i * 8 + 8], [i32; 8]);
                }
                $simd_col_fn(_token, &mut v, min_v, max_v);
                for i in 0..8 {
                    storeu_256!(&mut tmp[i * 8..i * 8 + 8], [i32; 8], v[i]);
                }
            }

            // Add to destination
            let zero = _mm_setzero_si128();
            let max_val = _mm_set1_epi32(bitdepth_max);
            for y in 0..8 {
                let dst_off = y * stride_u16;
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

            coeff[..64].fill(0);
        }
    };
}

// 8x8 hybrid identity transforms 16bpc (SIMD col)
impl_8x8_transform_16bpc_strided_simd_col!(
    inv_txfm_add_identity_dct_8x8_16bpc_avx2_inner,
    identity8_1d,
    dct8_1d_cols8
);
impl_8x8_transform_16bpc_strided_simd_col!(
    inv_txfm_add_dct_identity_8x8_16bpc_avx2_inner,
    dct8_1d,
    identity8_1d_cols8
);
impl_8x8_transform_16bpc_strided_simd_col!(
    inv_txfm_add_identity_adst_8x8_16bpc_avx2_inner,
    identity8_1d,
    adst8_1d_cols8
);
impl_8x8_transform_16bpc_strided_simd_col!(
    inv_txfm_add_adst_identity_8x8_16bpc_avx2_inner,
    adst8_1d,
    identity8_1d_cols8
);
impl_8x8_transform_16bpc_strided_simd_col!(
    inv_txfm_add_identity_flipadst_8x8_16bpc_avx2_inner,
    identity8_1d,
    flipadst8_1d_cols8
);
impl_8x8_transform_16bpc_strided_simd_col!(
    inv_txfm_add_flipadst_identity_8x8_16bpc_avx2_inner,
    flipadst8_1d,
    identity8_1d_cols8
);

impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_dct_8x8_16bpc_avx2,
    inv_txfm_add_identity_dct_8x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_identity_8x8_16bpc_avx2,
    inv_txfm_add_dct_identity_8x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_adst_8x8_16bpc_avx2,
    inv_txfm_add_identity_adst_8x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_identity_8x8_16bpc_avx2,
    inv_txfm_add_adst_identity_8x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_flipadst_8x8_16bpc_avx2,
    inv_txfm_add_identity_flipadst_8x8_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_identity_8x8_16bpc_avx2,
    inv_txfm_add_flipadst_identity_8x8_16bpc_avx2_inner
);

/// Macro for 4x4 transform variants 16bpc
macro_rules! impl_4x4_transform_16bpc {
    ($name:ident, $row_fn:ident, $col_fn:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[arcane]
        fn $name(
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
            let row_clip_min = (!bitdepth_max) << 7;
            let row_clip_max = !row_clip_min;
            let col_clip_min = (!bitdepth_max) << 5;
            let col_clip_max = !col_clip_min;
            let mut tmp = [0i32; 16];

            // Row transform (4 elements each, 4 rows)
            for y in 0..4 {
                let mut scratch = [0i32; 4];
                for x in 0..4 {
                    scratch[x] = coeff[y + x * 4] as i32;
                }
                $row_fn(&mut scratch[..4], 1, row_clip_min, row_clip_max);
                for x in 0..4 {
                    tmp[y * 4 + x] = iclip(scratch[x], col_clip_min, col_clip_max);
                }
            }

            // Column transform
            for x in 0..4 {
                $col_fn(&mut tmp[x..], 4, col_clip_min, col_clip_max);
            }

            // Add to destination
            let zero = _mm_setzero_si128();
            let max_val = _mm_set1_epi32(bitdepth_max);

            for y in 0..4 {
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
            coeff[..16].fill(0);
        }
    };
}

// 4x4 hybrid identity transforms 16bpc
impl_4x4_transform_16bpc!(
    inv_txfm_add_identity_dct_4x4_16bpc_avx2_inner,
    identity4_1d,
    dct4_1d
);
impl_4x4_transform_16bpc!(
    inv_txfm_add_dct_identity_4x4_16bpc_avx2_inner,
    dct4_1d,
    identity4_1d
);
impl_4x4_transform_16bpc!(
    inv_txfm_add_identity_adst_4x4_16bpc_avx2_inner,
    identity4_1d,
    adst4_1d
);
impl_4x4_transform_16bpc!(
    inv_txfm_add_adst_identity_4x4_16bpc_avx2_inner,
    adst4_1d,
    identity4_1d
);
impl_4x4_transform_16bpc!(
    inv_txfm_add_identity_flipadst_4x4_16bpc_avx2_inner,
    identity4_1d,
    flipadst4_1d
);
impl_4x4_transform_16bpc!(
    inv_txfm_add_flipadst_identity_4x4_16bpc_avx2_inner,
    flipadst4_1d,
    identity4_1d
);

impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_dct_4x4_16bpc_avx2,
    inv_txfm_add_identity_dct_4x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_identity_4x4_16bpc_avx2,
    inv_txfm_add_dct_identity_4x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_adst_4x4_16bpc_avx2,
    inv_txfm_add_identity_adst_4x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_identity_4x4_16bpc_avx2,
    inv_txfm_add_adst_identity_4x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_flipadst_4x4_16bpc_avx2,
    inv_txfm_add_identity_flipadst_4x4_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_identity_4x4_16bpc_avx2,
    inv_txfm_add_flipadst_identity_4x4_16bpc_avx2_inner
);

/// Macro for 16x16 transform variants 16bpc
/// Strided-style 16bpc 16x16 macro with SIMD col pass.
macro_rules! impl_16x16_transform_16bpc_strided_simd_col {
    ($name:ident, $row_fn:ident, $simd_col_fn:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[arcane]
        fn $name(
            _token: Desktop64,
            dst: &mut [u16],
            dst_stride: usize,
            coeff: &mut [i16],
            _eob: i32,
            bitdepth_max: i32,
        ) {
            use crate::src::safe_simd::pixel_access::{loadu_128, storeu_128};
            let mut dst = dst.flex_mut();
            let mut coeff = coeff.flex_mut();
            let stride_u16 = dst_stride / 2;
            let row_clip_min = (!bitdepth_max) << 7;
            let row_clip_max = !row_clip_min;
            let col_clip_min = (!bitdepth_max) << 5;
            let col_clip_max = !col_clip_min;
            let mut tmp = [0i32; 256];

            let rnd = 2;
            let shift = 2;
            for y in 0..16 {
                let mut scratch = [0i32; 16];
                for x in 0..16 {
                    scratch[x] = coeff[y + x * 16] as i32;
                }
                $row_fn(&mut scratch[..16], 1, row_clip_min, row_clip_max);
                for x in 0..16 {
                    tmp[y * 16 + x] =
                        iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
                }
            }

            // SIMD column transform
            $simd_col_fn(_token, &mut tmp, col_clip_min, col_clip_max);

            // Add to destination
            let zero = _mm_setzero_si128();
            let max_val = _mm_set1_epi32(bitdepth_max);
            for y in 0..16 {
                let dst_off = y * stride_u16;
                for chunk in 0..2 {
                    let x_base = chunk * 8;
                    let dst_chunk_off = dst_off + x_base;
                    let d = loadu_128!(
                        <&[u16; 8]>::try_from(&dst[dst_chunk_off..dst_chunk_off + 8]).unwrap()
                    );
                    let d_lo = _mm_unpacklo_epi16(d, zero);
                    let d_hi = _mm_unpackhi_epi16(d, zero);
                    let c_lo = _mm_set_epi32(
                        (tmp[y * 16 + x_base + 3] + 8) >> 4,
                        (tmp[y * 16 + x_base + 2] + 8) >> 4,
                        (tmp[y * 16 + x_base + 1] + 8) >> 4,
                        (tmp[y * 16 + x_base + 0] + 8) >> 4,
                    );
                    let c_hi = _mm_set_epi32(
                        (tmp[y * 16 + x_base + 7] + 8) >> 4,
                        (tmp[y * 16 + x_base + 6] + 8) >> 4,
                        (tmp[y * 16 + x_base + 5] + 8) >> 4,
                        (tmp[y * 16 + x_base + 4] + 8) >> 4,
                    );
                    let sum_lo = _mm_add_epi32(d_lo, c_lo);
                    let sum_hi = _mm_add_epi32(d_hi, c_hi);
                    let clamped_lo = _mm_max_epi32(_mm_min_epi32(sum_lo, max_val), zero);
                    let clamped_hi = _mm_max_epi32(_mm_min_epi32(sum_hi, max_val), zero);
                    let packed = _mm_packus_epi32(clamped_lo, clamped_hi);
                    storeu_128!(
                        <&mut [u16; 8]>::try_from(&mut dst[dst_chunk_off..dst_chunk_off + 8])
                            .unwrap(),
                        packed
                    );
                }
            }

            coeff[..256].fill(0);
        }
    };
}

#[allow(unused_macros)]
macro_rules! impl_16x16_transform_16bpc {
    ($name:ident, $row_fn:ident, $col_fn:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[arcane]
        fn $name(
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
            let row_clip_min = (!bitdepth_max) << 7;
            let row_clip_max = !row_clip_min;
            let col_clip_min = (!bitdepth_max) << 5;
            let col_clip_max = !col_clip_min;
            let mut tmp = [0i32; 256];

            // Row transform with shift=2 (16x16 intermediate shift)
            let rnd = 2;
            let shift = 2;

            for y in 0..16 {
                let mut scratch = [0i32; 16];
                for x in 0..16 {
                    scratch[x] = coeff[y + x * 16] as i32;
                }
                $row_fn(&mut scratch[..16], 1, row_clip_min, row_clip_max);
                for x in 0..16 {
                    tmp[y * 16 + x] =
                        iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
                }
            }

            // Column transform
            for x in 0..16 {
                $col_fn(&mut tmp[x..], 16, col_clip_min, col_clip_max);
            }

            // Add to destination
            let zero = _mm_setzero_si128();
            let max_val = _mm_set1_epi32(bitdepth_max);

            for y in 0..16 {
                let dst_off = y * stride_u16;

                for chunk in 0..2 {
                    let x_base = chunk * 8;
                    let dst_chunk_off = dst_off + x_base;

                    let d = loadu_128!(
                        <&[u16; 8]>::try_from(&dst[dst_chunk_off..dst_chunk_off + 8]).unwrap()
                    );
                    let d_lo = _mm_unpacklo_epi16(d, zero);
                    let d_hi = _mm_unpackhi_epi16(d, zero);

                    let c_lo = _mm_set_epi32(
                        (tmp[y * 16 + x_base + 3] + 8) >> 4,
                        (tmp[y * 16 + x_base + 2] + 8) >> 4,
                        (tmp[y * 16 + x_base + 1] + 8) >> 4,
                        (tmp[y * 16 + x_base + 0] + 8) >> 4,
                    );
                    let c_hi = _mm_set_epi32(
                        (tmp[y * 16 + x_base + 7] + 8) >> 4,
                        (tmp[y * 16 + x_base + 6] + 8) >> 4,
                        (tmp[y * 16 + x_base + 5] + 8) >> 4,
                        (tmp[y * 16 + x_base + 4] + 8) >> 4,
                    );

                    let sum_lo = _mm_add_epi32(d_lo, c_lo);
                    let sum_hi = _mm_add_epi32(d_hi, c_hi);
                    let clamped_lo = _mm_max_epi32(_mm_min_epi32(sum_lo, max_val), zero);
                    let clamped_hi = _mm_max_epi32(_mm_min_epi32(sum_hi, max_val), zero);
                    let packed = _mm_packus_epi32(clamped_lo, clamped_hi);

                    storeu_128!(
                        <&mut [u16; 8]>::try_from(&mut dst[dst_chunk_off..dst_chunk_off + 8])
                            .unwrap(),
                        packed
                    );
                }
            }

            // Clear coefficients
            coeff[..256].fill(0);
        }
    };
}

// 16x16 hybrid identity transforms 16bpc
impl_16x16_transform_16bpc_strided_simd_col!(
    inv_txfm_add_identity_dct_16x16_16bpc_avx2_inner,
    identity16_1d,
    dct16x16_cols_simd
);
impl_16x16_transform_16bpc_strided_simd_col!(
    inv_txfm_add_dct_identity_16x16_16bpc_avx2_inner,
    dct16_1d,
    identity16x16_cols_simd
);
impl_16x16_transform_16bpc_strided_simd_col!(
    inv_txfm_add_identity_adst_16x16_16bpc_avx2_inner,
    identity16_1d,
    adst16x16_cols_simd
);
impl_16x16_transform_16bpc_strided_simd_col!(
    inv_txfm_add_adst_identity_16x16_16bpc_avx2_inner,
    adst16_1d,
    identity16x16_cols_simd
);
impl_16x16_transform_16bpc_strided_simd_col!(
    inv_txfm_add_identity_flipadst_16x16_16bpc_avx2_inner,
    identity16_1d,
    flipadst16x16_cols_simd
);
impl_16x16_transform_16bpc_strided_simd_col!(
    inv_txfm_add_flipadst_identity_16x16_16bpc_avx2_inner,
    flipadst16_1d,
    identity16x16_cols_simd
);

impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_dct_16x16_16bpc_avx2,
    inv_txfm_add_identity_dct_16x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_dct_identity_16x16_16bpc_avx2,
    inv_txfm_add_dct_identity_16x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_adst_16x16_16bpc_avx2,
    inv_txfm_add_identity_adst_16x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_adst_identity_16x16_16bpc_avx2,
    inv_txfm_add_adst_identity_16x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_identity_flipadst_16x16_16bpc_avx2,
    inv_txfm_add_identity_flipadst_16x16_16bpc_avx2_inner
);
impl_ffi_wrapper_16bpc!(
    inv_txfm_add_flipadst_identity_16x16_16bpc_avx2,
    inv_txfm_add_flipadst_identity_16x16_16bpc_avx2_inner
);

