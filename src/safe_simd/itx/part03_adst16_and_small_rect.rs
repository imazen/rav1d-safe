// ============================================================================
// 16x16 ADST TRANSFORM VARIANTS
// ============================================================================

/// Macro to generate 16x16 transform inner functions (legacy, all-scalar).
/// Currently unused — every call site now uses `impl_16x16_transform_simd_col!`.
/// Kept as a reference for the structure and for any future 16bpc or rectangular
/// transforms that need both row+col scalar.
#[allow(unused_macros)]
macro_rules! impl_16x16_transform {
    ($name:ident, $row_fn:ident, $col_fn:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[arcane]
        fn $name(
            _token: Desktop64,
            dst: &mut [u8],
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
            let row_clip_min = i16::MIN as i32;
            let row_clip_max = i16::MAX as i32;
            let col_clip_min = i16::MIN as i32;
            let col_clip_max = i16::MAX as i32;

            let mut tmp = [0i32; 256];
            inv_txfm_16x16_inner(
                &mut tmp,
                &*coeff,
                $row_fn,
                $col_fn,
                row_clip_min,
                row_clip_max,
                col_clip_min,
                col_clip_max,
            );
            add_16x16_to_dst(
                _token,
                &mut *dst,
                dst_stride,
                &tmp,
                &mut *coeff,
                bitdepth_max,
            );
        }
    };
}

/// Macro variant: row pass uses scalar 1D, column pass uses SIMD.
/// Use for transforms where col_fn == dct16_1d (SIMD path exists).
macro_rules! impl_16x16_transform_simd_col {
    ($name:ident, $row_fn:ident, $simd_col_fn:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[arcane]
        fn $name(
            _token: Desktop64,
            dst: &mut [u8],
            dst_stride: usize,
            coeff: &mut [i16],
            _eob: i32,
            bitdepth_max: i32,
        ) {
            let mut dst = dst.flex_mut();
            let mut coeff = coeff.flex_mut();
            let row_clip_min = i16::MIN as i32;
            let row_clip_max = i16::MAX as i32;
            let col_clip_min = i16::MIN as i32;
            let col_clip_max = i16::MAX as i32;

            let mut tmp = [0i32; 256];
            inv_txfm_16x16_row_pass_only(
                &mut tmp,
                &*coeff,
                $row_fn,
                row_clip_min,
                row_clip_max,
                col_clip_min,
                col_clip_max,
            );
            $simd_col_fn(_token, &mut tmp, col_clip_min, col_clip_max);
            add_16x16_to_dst(
                _token,
                &mut *dst,
                dst_stride,
                &tmp,
                &mut *coeff,
                bitdepth_max,
            );
        }
    };
}

/// Macro variant: row pass uses SIMD ADST-16 (8 rows at a time), column pass
/// uses SIMD. `$flipped` is `true` for flipadst row, `false` for plain adst.
macro_rules! impl_16x16_transform_simd_row_adst_col {
    ($name:ident, $flipped:expr, $simd_col_fn:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[arcane]
        fn $name(
            _token: Desktop64,
            dst: &mut [u8],
            dst_stride: usize,
            coeff: &mut [i16],
            _eob: i32,
            bitdepth_max: i32,
        ) {
            let mut dst = dst.flex_mut();
            let mut coeff = coeff.flex_mut();
            let row_clip_min = i16::MIN as i32;
            let row_clip_max = i16::MAX as i32;
            let col_clip_min = i16::MIN as i32;
            let col_clip_max = i16::MAX as i32;
            let mut tmp = [0i32; 256];
            // SIMD row ADST-16: 2 batches of 8 rows, no rect2, shift=2, rnd=2.
            {
                let coeff_slice = coeff.as_slice();
                for y_base in [0usize, 8] {
                    simd_row_adst16_8bpc_8rows(
                        _token,
                        coeff_slice,
                        16,
                        y_base,
                        false,
                        $flipped,
                        2,
                        2,
                        &mut tmp,
                        row_clip_min,
                        row_clip_max,
                        col_clip_min,
                        col_clip_max,
                    );
                }
            }
            $simd_col_fn(_token, &mut tmp, col_clip_min, col_clip_max);
            add_16x16_to_dst(
                _token,
                &mut *dst,
                dst_stride,
                &tmp,
                &mut *coeff,
                bitdepth_max,
            );
        }
    };
}

/// Macro variant: row pass uses SIMD DCT-16 (8 rows at a time), column pass
/// uses SIMD. Use for transforms where row_fn == dct16_1d AND col is SIMD.
/// Same shape as `impl_16x16_transform_simd_col` but skips the scalar
/// dct16_1d for ~1% wall-clock savings.
macro_rules! impl_16x16_transform_simd_row_dct_col {
    ($name:ident, $simd_col_fn:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[arcane]
        fn $name(
            _token: Desktop64,
            dst: &mut [u8],
            dst_stride: usize,
            coeff: &mut [i16],
            _eob: i32,
            bitdepth_max: i32,
        ) {
            let mut dst = dst.flex_mut();
            let mut coeff = coeff.flex_mut();
            let row_clip_min = i16::MIN as i32;
            let row_clip_max = i16::MAX as i32;
            let col_clip_min = i16::MIN as i32;
            let col_clip_max = i16::MAX as i32;
            let mut tmp = [0i32; 256];
            // SIMD row DCT-16: 16 rows (AVX-512 16-row path or AVX2 2x8). no rect2, shift=2, rnd=2.
            {
                let coeff_slice = coeff.as_slice();
                row_dct16_8bpc_block(
                    _token,
                    coeff_slice,
                    16,
                    16,
                    false,
                    2,
                    2,
                    &mut tmp,
                    row_clip_min,
                    row_clip_max,
                    col_clip_min,
                    col_clip_max,
                );
            }
            $simd_col_fn(_token, &mut tmp, col_clip_min, col_clip_max);
            add_16x16_to_dst(
                _token,
                &mut *dst,
                dst_stride,
                &tmp,
                &mut *coeff,
                bitdepth_max,
            );
        }
    };
}

/// Macro to generate FFI wrappers for 16x16 transforms
macro_rules! impl_16x16_ffi_wrapper {
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

// Generate inner functions for all 16x16 transform combinations
impl_16x16_transform_simd_row_adst_col!(
    inv_txfm_add_adst_dct_16x16_8bpc_avx2_inner,
    false,
    dct16x16_cols_simd
);
impl_16x16_transform_simd_row_dct_col!(
    inv_txfm_add_dct_adst_16x16_8bpc_avx2_inner,
    adst16x16_cols_simd
);
impl_16x16_transform_simd_row_adst_col!(
    inv_txfm_add_adst_adst_16x16_8bpc_avx2_inner,
    false,
    adst16x16_cols_simd
);
impl_16x16_transform_simd_row_adst_col!(
    inv_txfm_add_flipadst_dct_16x16_8bpc_avx2_inner,
    true,
    dct16x16_cols_simd
);
impl_16x16_transform_simd_row_dct_col!(
    inv_txfm_add_dct_flipadst_16x16_8bpc_avx2_inner,
    flipadst16x16_cols_simd
);
impl_16x16_transform_simd_row_adst_col!(
    inv_txfm_add_flipadst_flipadst_16x16_8bpc_avx2_inner,
    true,
    flipadst16x16_cols_simd
);
impl_16x16_transform_simd_row_adst_col!(
    inv_txfm_add_adst_flipadst_16x16_8bpc_avx2_inner,
    false,
    flipadst16x16_cols_simd
);
impl_16x16_transform_simd_row_adst_col!(
    inv_txfm_add_flipadst_adst_16x16_8bpc_avx2_inner,
    true,
    adst16x16_cols_simd
);
impl_16x16_transform_simd_col!(
    inv_txfm_add_identity_dct_16x16_8bpc_avx2_inner,
    identity16_1d,
    dct16x16_cols_simd
);
impl_16x16_transform_simd_row_dct_col!(
    inv_txfm_add_dct_identity_16x16_8bpc_avx2_inner,
    identity16x16_cols_simd
);
impl_16x16_transform_simd_col!(
    inv_txfm_add_identity_adst_16x16_8bpc_avx2_inner,
    identity16_1d,
    adst16x16_cols_simd
);
impl_16x16_transform_simd_row_adst_col!(
    inv_txfm_add_adst_identity_16x16_8bpc_avx2_inner,
    false,
    identity16x16_cols_simd
);
impl_16x16_transform_simd_col!(
    inv_txfm_add_identity_flipadst_16x16_8bpc_avx2_inner,
    identity16_1d,
    flipadst16x16_cols_simd
);
impl_16x16_transform_simd_row_adst_col!(
    inv_txfm_add_flipadst_identity_16x16_8bpc_avx2_inner,
    true,
    identity16x16_cols_simd
);

// Generate FFI wrappers
impl_16x16_ffi_wrapper!(
    inv_txfm_add_adst_dct_16x16_8bpc_avx2,
    inv_txfm_add_adst_dct_16x16_8bpc_avx2_inner
);
impl_16x16_ffi_wrapper!(
    inv_txfm_add_dct_adst_16x16_8bpc_avx2,
    inv_txfm_add_dct_adst_16x16_8bpc_avx2_inner
);
impl_16x16_ffi_wrapper!(
    inv_txfm_add_adst_adst_16x16_8bpc_avx2,
    inv_txfm_add_adst_adst_16x16_8bpc_avx2_inner
);
impl_16x16_ffi_wrapper!(
    inv_txfm_add_flipadst_dct_16x16_8bpc_avx2,
    inv_txfm_add_flipadst_dct_16x16_8bpc_avx2_inner
);
impl_16x16_ffi_wrapper!(
    inv_txfm_add_dct_flipadst_16x16_8bpc_avx2,
    inv_txfm_add_dct_flipadst_16x16_8bpc_avx2_inner
);
impl_16x16_ffi_wrapper!(
    inv_txfm_add_flipadst_flipadst_16x16_8bpc_avx2,
    inv_txfm_add_flipadst_flipadst_16x16_8bpc_avx2_inner
);
impl_16x16_ffi_wrapper!(
    inv_txfm_add_adst_flipadst_16x16_8bpc_avx2,
    inv_txfm_add_adst_flipadst_16x16_8bpc_avx2_inner
);
impl_16x16_ffi_wrapper!(
    inv_txfm_add_flipadst_adst_16x16_8bpc_avx2,
    inv_txfm_add_flipadst_adst_16x16_8bpc_avx2_inner
);
impl_16x16_ffi_wrapper!(
    inv_txfm_add_identity_dct_16x16_8bpc_avx2,
    inv_txfm_add_identity_dct_16x16_8bpc_avx2_inner
);
impl_16x16_ffi_wrapper!(
    inv_txfm_add_dct_identity_16x16_8bpc_avx2,
    inv_txfm_add_dct_identity_16x16_8bpc_avx2_inner
);
impl_16x16_ffi_wrapper!(
    inv_txfm_add_identity_adst_16x16_8bpc_avx2,
    inv_txfm_add_identity_adst_16x16_8bpc_avx2_inner
);
impl_16x16_ffi_wrapper!(
    inv_txfm_add_adst_identity_16x16_8bpc_avx2,
    inv_txfm_add_adst_identity_16x16_8bpc_avx2_inner
);
impl_16x16_ffi_wrapper!(
    inv_txfm_add_identity_flipadst_16x16_8bpc_avx2,
    inv_txfm_add_identity_flipadst_16x16_8bpc_avx2_inner
);
impl_16x16_ffi_wrapper!(
    inv_txfm_add_flipadst_identity_16x16_8bpc_avx2,
    inv_txfm_add_flipadst_identity_16x16_8bpc_avx2_inner
);

/// Full 2D DCT_DCT 16x16 inverse transform with add-to-destination
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_16x16_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    // For 8bpc: row_clip = col_clip = i16 range
    let _row_clip_min = i16::MIN as i32;
    let _row_clip_max = i16::MAX as i32;
    let col_clip_min = i16::MIN as i32;
    let col_clip_max = i16::MAX as i32;
    let mut tmp = [0i32; 256];

    // SIMD row transform via i16-packed pmaddwd DCT-16.
    {
        let coeff_arr: &[i16; 256] = coeff.as_slice()[..256].try_into().unwrap();
        let raw = dct16_row_pass_i16_simd(_token, *coeff_arr);

        // Apply rnd=2, shift=2, col_clip (same post-processing as
        // simd_row_dct16_8bpc_8rows with shift=2, rnd=2).
        let rnd_v = _mm256_set1_epi32(2);
        let col_min_v = _mm256_set1_epi32(col_clip_min);
        let col_max_v = _mm256_set1_epi32(col_clip_max);
        for y in 0..16 {
            for chunk in 0..2u32 {
                let b = (chunk * 8) as usize;
                let off = y * 16 + b;
                let v = loadu_256!(&raw[off..off + 8], [i32; 8]);
                let shifted = _mm256_srai_epi32::<2>(_mm256_add_epi32(v, rnd_v));
                let clamped = _mm256_max_epi32(_mm256_min_epi32(shifted, col_max_v), col_min_v);
                storeu_256!(&mut tmp[off..off + 8], [i32; 8], clamped);
            }
        }
    }

    // Column transform: i16-packed pmaddwd (replaces i32 mullo dct16x16_cols_simd)
    let col_out = dct16_col_pass_i16(_token, &tmp);

    // Add to destination with SIMD
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi16(bitdepth_max as i16);
    let rnd_final = _mm256_set1_epi32(8);

    for y in 0..16 {
        let dst_off = y * dst_stride;

        // Load destination pixels (16 bytes)
        let d = loadu_128!(<&[u8; 16]>::try_from(&dst[dst_off..dst_off + 16]).unwrap());
        let d16 = _mm256_cvtepu8_epi16(d);

        // Load column-pass output (16 i32 values as 2 ymm)
        let c0 = loadu_256!(&col_out[y * 16..y * 16 + 8], [i32; 8]);
        let c1 = loadu_256!(&col_out[y * 16 + 8..y * 16 + 16], [i32; 8]);

        // Final scaling: (c + 8) >> 4
        let c0_scaled = _mm256_srai_epi32(_mm256_add_epi32(c0, rnd_final), 4);
        let c1_scaled = _mm256_srai_epi32(_mm256_add_epi32(c1, rnd_final), 4);

        // Pack to 16-bit
        let c16 = _mm256_packs_epi32(c0_scaled, c1_scaled);
        // Fix lane order after packs
        let c16 = _mm256_permute4x64_epi64(c16, 0b11_01_10_00);

        // Add to destination
        let sum = _mm256_add_epi16(d16, c16);
        let clamped = _mm256_max_epi16(_mm256_min_epi16(sum, max_val), zero);

        // Pack to 8-bit
        let packed = _mm256_packus_epi16(clamped, clamped);
        let packed = _mm256_permute4x64_epi64(packed, 0b11_01_10_00);

        // Store 16 pixels
        storeu_128!(
            <&mut [u8; 16]>::try_from(&mut dst[dst_off..dst_off + 16]).unwrap(),
            _mm256_castsi256_si128(packed)
        );
    }

    // Clear coefficients (256 * 2 = 512 bytes = 16 * 32 bytes)
    coeff[..256].fill(0);
}

/// FFI wrapper for 16x16 DCT_DCT 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_16x16_8bpc_avx2(
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
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, _coeff_len as usize * stride + stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_16x16_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// 16x16 DCT_DCT 16bpc
// ============================================================================

/// 16x16 DCT_DCT for 16bpc (10/12-bit pixels)
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_16x16_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize, // stride in bytes
    coeff: &mut [i32],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    // For 16bpc, stride is in bytes but we access u16
    let stride_u16 = dst_stride / 2;

    // For 16bpc: use full i32 range for intermediate calculations
    let row_clip_min = (!bitdepth_max) << 7;
    let row_clip_max = !row_clip_min;
    let col_clip_min = (!bitdepth_max) << 5;
    let col_clip_max = !col_clip_min;
    let mut tmp = [0i32; 256];

    // Row transform (shift = 2 for 16x16)
    let rnd = 2;
    let shift = 2;

    for y in 0..16 {
        // Load row from column-major
        let mut scratch = [0i32; 16];
        for x in 0..16 {
            scratch[x] = coeff[y + x * 16] as i32;
        }
        dct16_1d(&mut scratch[..16], 1, row_clip_min, row_clip_max);
        // Apply intermediate shift and store row-major
        for x in 0..16 {
            tmp[y * 16 + x] = iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
        }
    }

    // Column transform: SIMD across 8 columns at a time
    dct16x16_cols_simd(_token, &mut tmp, col_clip_min, col_clip_max);

    // Add to destination with SIMD (16bpc = u16 pixels)
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi32(bitdepth_max);
    let rnd_final = _mm256_set1_epi32(8);

    for y in 0..16 {
        let dst_off = y * stride_u16;

        // Load destination pixels (16 u16 = 32 bytes)
        let d = loadu_256!(<&[u16; 16]>::try_from(&dst[dst_off..dst_off + 16]).unwrap());
        // Unpack to i32: low 8 and high 8
        let d_lo = _mm256_unpacklo_epi16(d, _mm256_setzero_si256());
        let d_hi = _mm256_unpackhi_epi16(d, _mm256_setzero_si256());
        // Permute to get correct order after unpack
        let d_0_4 = _mm256_permute2x128_si256(d_lo, d_hi, 0x20); // pixels 0-7
        let d_4_8 = _mm256_permute2x128_si256(d_lo, d_hi, 0x31); // pixels 8-15

        // Load and scale coefficients (16 values)
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

        // Final scaling: (c + 8) >> 4
        let c0_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c0, rnd_final));
        let c1_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c1, rnd_final));

        // Add to destination
        let sum0 = _mm256_add_epi32(d_0_4, c0_scaled);
        let sum1 = _mm256_add_epi32(d_4_8, c1_scaled);

        // Clamp to [0, bitdepth_max]
        let clamped0 = _mm256_max_epi32(_mm256_min_epi32(sum0, max_val), zero);
        let clamped1 = _mm256_max_epi32(_mm256_min_epi32(sum1, max_val), zero);

        // Pack to u16 and store
        let packed = _mm256_packus_epi32(clamped0, clamped1);
        // Fix lane order after packus
        let packed = _mm256_permute4x64_epi64(packed, 0b11_01_10_00);
        storeu_256!(
            <&mut [u16; 16]>::try_from(&mut dst[dst_off..dst_off + 16]).unwrap(),
            packed
        );
    }

    // Clear coefficients (256 * 2 = 512 bytes = 16 * 32 bytes)
    coeff[..256].fill(0);
}

/// FFI wrapper for 16x16 DCT_DCT 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_16x16_16bpc_avx2(
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

    inv_txfm_add_dct_dct_16x16_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// RECTANGULAR TRANSFORMS (4x8, 8x4, etc.)
// ============================================================================

/// Full 2D DCT_DCT 4x8 inverse transform
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_4x8_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    // W=4, H=8, shift=0 for 4x8
    let row_clip_min = i16::MIN as i32;
    let row_clip_max = i16::MAX as i32;
    let col_clip_min = i16::MIN as i32;
    let col_clip_max = i16::MAX as i32;
    let mut tmp = [0i32; 32];

    // is_rect2 = true for 4x8, so apply sqrt(2) scaling
    let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

    // Row transform (4 elements each, 8 rows)
    for y in 0..8 {
        // Load row from column-major with rect2 scaling
        let mut scratch = [0i32; 4];
        for x in 0..4 {
            scratch[x] = rect2_scale(coeff[y + x * 8] as i32);
        }
        dct4_1d(&mut scratch[..4], 1, row_clip_min, row_clip_max);
        // Store row-major (no intermediate shift for 4x8)
        for x in 0..4 {
            tmp[y * 4 + x] = iclip(scratch[x], col_clip_min, col_clip_max);
        }
    }

    // Column transform (8 elements each, 4 columns)
    for x in 0..4 {
        dct8_1d(&mut tmp[x..], 4, col_clip_min, col_clip_max);
    }

    // Add to destination
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi16(bitdepth_max as i16);

    for y in 0..8 {
        let dst_off = y * dst_stride;

        // Load destination (4 bytes)
        let d = loadi32!(&dst[dst_off..dst_off + 4]);
        let d16 = _mm_unpacklo_epi8(d, zero);
        let d32 = _mm_cvtepi16_epi32(d16);

        // Load and scale coefficients
        let c = _mm_set_epi32(
            (tmp[y * 4 + 3] + 8) >> 4,
            (tmp[y * 4 + 2] + 8) >> 4,
            (tmp[y * 4 + 1] + 8) >> 4,
            (tmp[y * 4 + 0] + 8) >> 4,
        );

        let sum = _mm_add_epi32(d32, c);
        let sum16 = _mm_packs_epi32(sum, sum);
        let clamped = _mm_max_epi16(_mm_min_epi16(sum16, max_val), zero);
        let packed = _mm_packus_epi16(clamped, clamped);

        storei32!(&mut dst[dst_off..dst_off + 4], packed);
    }

    // Clear coefficients
    coeff[..32].fill(0);
}

/// FFI wrapper for 4x8 DCT_DCT 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_4x8_8bpc_avx2(
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
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, _coeff_len as usize * stride + stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_4x8_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// Full 2D DCT_DCT 8x4 inverse transform
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_8x4_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    // W=8, H=4, shift=0 for 8x4
    let row_clip_min = i16::MIN as i32;
    let row_clip_max = i16::MAX as i32;
    let col_clip_min = i16::MIN as i32;
    let col_clip_max = i16::MAX as i32;
    let mut tmp = [0i32; 32];

    // is_rect2 = true for 8x4, so apply sqrt(2) scaling
    let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

    // Row transform (8 elements each, 4 rows)
    for y in 0..4 {
        // Load row from column-major with rect2 scaling
        let mut scratch = [0i32; 8];
        for x in 0..8 {
            scratch[x] = rect2_scale(coeff[y + x * 4] as i32);
        }
        dct8_1d(&mut scratch[..8], 1, row_clip_min, row_clip_max);
        // Store row-major (no intermediate shift for 8x4)
        for x in 0..8 {
            tmp[y * 8 + x] = iclip(scratch[x], col_clip_min, col_clip_max);
        }
    }

    // Column transform: SIMD across 8 columns, 4 rows (single chunk)
    {
        let min_v = _mm256_set1_epi32(col_clip_min);
        let max_v = _mm256_set1_epi32(col_clip_max);
        let mut v = [_mm256_setzero_si256(); 4];
        for i in 0..4 {
            v[i] = loadu_256!(&tmp[i * 8..i * 8 + 8], [i32; 8]);
        }
        dct4_1d_cols8(_token, &mut v, min_v, max_v);
        for i in 0..4 {
            storeu_256!(&mut tmp[i * 8..i * 8 + 8], [i32; 8], v[i]);
        }
    }

    // Add to destination
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi16(bitdepth_max as i16);
    let rnd_final = _mm256_set1_epi32(8);

    for y in 0..4 {
        let dst_off = y * dst_stride;

        // Load destination (8 bytes)
        let d = loadi64!(&dst[dst_off..dst_off + 8]);
        let d16 = _mm_unpacklo_epi8(d, zero);

        // Load and scale coefficients
        let c_lo = _mm_set_epi32(
            tmp[y * 8 + 3],
            tmp[y * 8 + 2],
            tmp[y * 8 + 1],
            tmp[y * 8 + 0],
        );
        let c_hi = _mm_set_epi32(
            tmp[y * 8 + 7],
            tmp[y * 8 + 6],
            tmp[y * 8 + 5],
            tmp[y * 8 + 4],
        );

        let c_lo_256 = _mm256_set_m128i(c_hi, c_lo);
        let c_scaled = _mm256_srai_epi32(_mm256_add_epi32(c_lo_256, rnd_final), 4);

        let c_lo_scaled = _mm256_castsi256_si128(c_scaled);
        let c_hi_scaled = _mm256_extracti128_si256(c_scaled, 1);
        let c16 = _mm_packs_epi32(c_lo_scaled, c_hi_scaled);

        let sum = _mm_add_epi16(d16, c16);
        let clamped = _mm_max_epi16(_mm_min_epi16(sum, max_val), zero);
        let packed = _mm_packus_epi16(clamped, clamped);

        storei64!(&mut dst[dst_off..dst_off + 8], packed);
    }

    // Clear coefficients
    coeff[..32].fill(0);
}

/// FFI wrapper for 8x4 DCT_DCT 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_8x4_8bpc_avx2(
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
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, _coeff_len as usize * stride + stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_8x4_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// ADST VARIANTS FOR 4x8 and 8x4
// ============================================================================

/// Helper macro for 4x8 transforms with configurable row/col transforms
macro_rules! impl_4x8_transform {
    ($name:ident, $row_fn:ident, $col_fn:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[arcane]
        fn $name(
            _token: Desktop64,
            dst: &mut [u8],
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
            let row_clip_min = i16::MIN as i32;
            let row_clip_max = i16::MAX as i32;
            let col_clip_min = i16::MIN as i32;
            let col_clip_max = i16::MAX as i32;
            let mut tmp = [0i32; 32];

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

            // Column transform (8 elements each, 4 columns)
            for x in 0..4 {
                $col_fn(&mut tmp[x..], 4, col_clip_min, col_clip_max);
            }

            // Add to destination
            let zero = _mm_setzero_si128();
            let max_val = _mm_set1_epi16(bitdepth_max as i16);

            for y in 0..8 {
                let dst_off = y * dst_stride;
                let d = loadi32!(&dst[dst_off..dst_off + 4]);
                let d16 = _mm_unpacklo_epi8(d, zero);
                let d32 = _mm_cvtepi16_epi32(d16);

                let c = _mm_set_epi32(
                    (tmp[y * 4 + 3] + 8) >> 4,
                    (tmp[y * 4 + 2] + 8) >> 4,
                    (tmp[y * 4 + 1] + 8) >> 4,
                    (tmp[y * 4 + 0] + 8) >> 4,
                );

                let sum = _mm_add_epi32(d32, c);
                let sum16 = _mm_packs_epi32(sum, sum);
                let clamped = _mm_max_epi16(_mm_min_epi16(sum16, max_val), zero);
                let packed = _mm_packus_epi16(clamped, clamped);
                storei32!(&mut dst[dst_off..dst_off + 4], packed);
            }

            coeff[..32].fill(0);
        }
    };
}

/// Helper macro for 8x4 transforms with configurable row/col transforms
macro_rules! impl_8x4_transform {
    ($name:ident, $row_fn:ident, $col_fn:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[arcane]
        fn $name(
            _token: Desktop64,
            dst: &mut [u8],
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
            let row_clip_min = i16::MIN as i32;
            let row_clip_max = i16::MAX as i32;
            let col_clip_min = i16::MIN as i32;
            let col_clip_max = i16::MAX as i32;
            let mut tmp = [0i32; 32];

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

            // Column transform (4 elements each, 8 columns)
            for x in 0..8 {
                $col_fn(&mut tmp[x..], 8, col_clip_min, col_clip_max);
            }

            // Add to destination
            for y in 0..4 {
                let dst_off = y * dst_stride;
                for x in 0..8 {
                    let d = dst[dst_off + x] as i32;
                    let c = (tmp[y * 8 + x] + 8) >> 4;
                    let result = iclip(d + c, 0, bitdepth_max);
                    dst[dst_off + x] = result as u8;
                }
            }

            coeff[..32].fill(0);
        }
    };
}

// Generate 4x8 ADST variants
impl_4x8_transform!(inv_txfm_add_adst_dct_4x8_8bpc_avx2_inner, adst4_1d, dct8_1d);
impl_4x8_transform!(inv_txfm_add_dct_adst_4x8_8bpc_avx2_inner, dct4_1d, adst8_1d);
impl_4x8_transform!(
    inv_txfm_add_adst_adst_4x8_8bpc_avx2_inner,
    adst4_1d,
    adst8_1d
);
impl_4x8_transform!(
    inv_txfm_add_flipadst_dct_4x8_8bpc_avx2_inner,
    flipadst4_1d,
    dct8_1d
);
impl_4x8_transform!(
    inv_txfm_add_dct_flipadst_4x8_8bpc_avx2_inner,
    dct4_1d,
    flipadst8_1d
);
impl_4x8_transform!(
    inv_txfm_add_flipadst_flipadst_4x8_8bpc_avx2_inner,
    flipadst4_1d,
    flipadst8_1d
);
impl_4x8_transform!(
    inv_txfm_add_adst_flipadst_4x8_8bpc_avx2_inner,
    adst4_1d,
    flipadst8_1d
);
impl_4x8_transform!(
    inv_txfm_add_flipadst_adst_4x8_8bpc_avx2_inner,
    flipadst4_1d,
    adst8_1d
);

// Generate 8x4 ADST variants
impl_8x4_transform!(inv_txfm_add_adst_dct_8x4_8bpc_avx2_inner, adst8_1d, dct4_1d);
impl_8x4_transform!(inv_txfm_add_dct_adst_8x4_8bpc_avx2_inner, dct8_1d, adst4_1d);
impl_8x4_transform!(
    inv_txfm_add_adst_adst_8x4_8bpc_avx2_inner,
    adst8_1d,
    adst4_1d
);
impl_8x4_transform!(
    inv_txfm_add_flipadst_dct_8x4_8bpc_avx2_inner,
    flipadst8_1d,
    dct4_1d
);
impl_8x4_transform!(
    inv_txfm_add_dct_flipadst_8x4_8bpc_avx2_inner,
    dct8_1d,
    flipadst4_1d
);
impl_8x4_transform!(
    inv_txfm_add_flipadst_flipadst_8x4_8bpc_avx2_inner,
    flipadst8_1d,
    flipadst4_1d
);
impl_8x4_transform!(
    inv_txfm_add_adst_flipadst_8x4_8bpc_avx2_inner,
    adst8_1d,
    flipadst4_1d
);
impl_8x4_transform!(
    inv_txfm_add_flipadst_adst_8x4_8bpc_avx2_inner,
    flipadst8_1d,
    adst4_1d
);

// FFI wrappers for 4x8 ADST variants
macro_rules! impl_4x8_ffi_wrapper {
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

impl_4x8_ffi_wrapper!(
    inv_txfm_add_adst_dct_4x8_8bpc_avx2,
    inv_txfm_add_adst_dct_4x8_8bpc_avx2_inner
);
impl_4x8_ffi_wrapper!(
    inv_txfm_add_dct_adst_4x8_8bpc_avx2,
    inv_txfm_add_dct_adst_4x8_8bpc_avx2_inner
);
impl_4x8_ffi_wrapper!(
    inv_txfm_add_adst_adst_4x8_8bpc_avx2,
    inv_txfm_add_adst_adst_4x8_8bpc_avx2_inner
);
impl_4x8_ffi_wrapper!(
    inv_txfm_add_flipadst_dct_4x8_8bpc_avx2,
    inv_txfm_add_flipadst_dct_4x8_8bpc_avx2_inner
);
impl_4x8_ffi_wrapper!(
    inv_txfm_add_dct_flipadst_4x8_8bpc_avx2,
    inv_txfm_add_dct_flipadst_4x8_8bpc_avx2_inner
);
impl_4x8_ffi_wrapper!(
    inv_txfm_add_flipadst_flipadst_4x8_8bpc_avx2,
    inv_txfm_add_flipadst_flipadst_4x8_8bpc_avx2_inner
);
impl_4x8_ffi_wrapper!(
    inv_txfm_add_adst_flipadst_4x8_8bpc_avx2,
    inv_txfm_add_adst_flipadst_4x8_8bpc_avx2_inner
);
impl_4x8_ffi_wrapper!(
    inv_txfm_add_flipadst_adst_4x8_8bpc_avx2,
    inv_txfm_add_flipadst_adst_4x8_8bpc_avx2_inner
);

// FFI wrappers for 8x4 ADST variants
macro_rules! impl_8x4_ffi_wrapper {
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

impl_8x4_ffi_wrapper!(
    inv_txfm_add_adst_dct_8x4_8bpc_avx2,
    inv_txfm_add_adst_dct_8x4_8bpc_avx2_inner
);
impl_8x4_ffi_wrapper!(
    inv_txfm_add_dct_adst_8x4_8bpc_avx2,
    inv_txfm_add_dct_adst_8x4_8bpc_avx2_inner
);
impl_8x4_ffi_wrapper!(
    inv_txfm_add_adst_adst_8x4_8bpc_avx2,
    inv_txfm_add_adst_adst_8x4_8bpc_avx2_inner
);
impl_8x4_ffi_wrapper!(
    inv_txfm_add_flipadst_dct_8x4_8bpc_avx2,
    inv_txfm_add_flipadst_dct_8x4_8bpc_avx2_inner
);
impl_8x4_ffi_wrapper!(
    inv_txfm_add_dct_flipadst_8x4_8bpc_avx2,
    inv_txfm_add_dct_flipadst_8x4_8bpc_avx2_inner
);
impl_8x4_ffi_wrapper!(
    inv_txfm_add_flipadst_flipadst_8x4_8bpc_avx2,
    inv_txfm_add_flipadst_flipadst_8x4_8bpc_avx2_inner
);
impl_8x4_ffi_wrapper!(
    inv_txfm_add_adst_flipadst_8x4_8bpc_avx2,
    inv_txfm_add_adst_flipadst_8x4_8bpc_avx2_inner
);
impl_8x4_ffi_wrapper!(
    inv_txfm_add_flipadst_adst_8x4_8bpc_avx2,
    inv_txfm_add_flipadst_adst_8x4_8bpc_avx2_inner
);

// IDTX for 4x8 and 8x4
impl_4x8_transform!(
    inv_txfm_add_identity_identity_4x8_8bpc_avx2_inner,
    identity4_1d,
    identity8_1d
);
impl_8x4_transform!(
    inv_txfm_add_identity_identity_8x4_8bpc_avx2_inner,
    identity8_1d,
    identity4_1d
);
impl_4x8_ffi_wrapper!(
    inv_txfm_add_identity_identity_4x8_8bpc_avx2,
    inv_txfm_add_identity_identity_4x8_8bpc_avx2_inner
);
impl_8x4_ffi_wrapper!(
    inv_txfm_add_identity_identity_8x4_8bpc_avx2,
    inv_txfm_add_identity_identity_8x4_8bpc_avx2_inner
);

// H_DCT and V_DCT for 4x8 (identity+dct mixes)
impl_4x8_transform!(
    inv_txfm_add_identity_dct_4x8_8bpc_avx2_inner,
    identity4_1d,
    dct8_1d
);
impl_4x8_transform!(
    inv_txfm_add_dct_identity_4x8_8bpc_avx2_inner,
    dct4_1d,
    identity8_1d
);
impl_4x8_ffi_wrapper!(
    inv_txfm_add_identity_dct_4x8_8bpc_avx2,
    inv_txfm_add_identity_dct_4x8_8bpc_avx2_inner
);
impl_4x8_ffi_wrapper!(
    inv_txfm_add_dct_identity_4x8_8bpc_avx2,
    inv_txfm_add_dct_identity_4x8_8bpc_avx2_inner
);

// H_DCT and V_DCT for 8x4
impl_8x4_transform!(
    inv_txfm_add_identity_dct_8x4_8bpc_avx2_inner,
    identity8_1d,
    dct4_1d
);
impl_8x4_transform!(
    inv_txfm_add_dct_identity_8x4_8bpc_avx2_inner,
    dct8_1d,
    identity4_1d
);
impl_8x4_ffi_wrapper!(
    inv_txfm_add_identity_dct_8x4_8bpc_avx2,
    inv_txfm_add_identity_dct_8x4_8bpc_avx2_inner
);
impl_8x4_ffi_wrapper!(
    inv_txfm_add_dct_identity_8x4_8bpc_avx2,
    inv_txfm_add_dct_identity_8x4_8bpc_avx2_inner
);

// H_ADST, V_ADST, H_FLIPADST, V_FLIPADST for 4x8
impl_4x8_transform!(
    inv_txfm_add_identity_adst_4x8_8bpc_avx2_inner,
    identity4_1d,
    adst8_1d
);
impl_4x8_transform!(
    inv_txfm_add_adst_identity_4x8_8bpc_avx2_inner,
    adst4_1d,
    identity8_1d
);
impl_4x8_transform!(
    inv_txfm_add_identity_flipadst_4x8_8bpc_avx2_inner,
    identity4_1d,
    flipadst8_1d
);
impl_4x8_transform!(
    inv_txfm_add_flipadst_identity_4x8_8bpc_avx2_inner,
    flipadst4_1d,
    identity8_1d
);
impl_4x8_ffi_wrapper!(
    inv_txfm_add_identity_adst_4x8_8bpc_avx2,
    inv_txfm_add_identity_adst_4x8_8bpc_avx2_inner
);
impl_4x8_ffi_wrapper!(
    inv_txfm_add_adst_identity_4x8_8bpc_avx2,
    inv_txfm_add_adst_identity_4x8_8bpc_avx2_inner
);
impl_4x8_ffi_wrapper!(
    inv_txfm_add_identity_flipadst_4x8_8bpc_avx2,
    inv_txfm_add_identity_flipadst_4x8_8bpc_avx2_inner
);
impl_4x8_ffi_wrapper!(
    inv_txfm_add_flipadst_identity_4x8_8bpc_avx2,
    inv_txfm_add_flipadst_identity_4x8_8bpc_avx2_inner
);

// H_ADST, V_ADST, H_FLIPADST, V_FLIPADST for 8x4
impl_8x4_transform!(
    inv_txfm_add_identity_adst_8x4_8bpc_avx2_inner,
    identity8_1d,
    adst4_1d
);
impl_8x4_transform!(
    inv_txfm_add_adst_identity_8x4_8bpc_avx2_inner,
    adst8_1d,
    identity4_1d
);
impl_8x4_transform!(
    inv_txfm_add_identity_flipadst_8x4_8bpc_avx2_inner,
    identity8_1d,
    flipadst4_1d
);
impl_8x4_transform!(
    inv_txfm_add_flipadst_identity_8x4_8bpc_avx2_inner,
    flipadst8_1d,
    identity4_1d
);
impl_8x4_ffi_wrapper!(
    inv_txfm_add_identity_adst_8x4_8bpc_avx2,
    inv_txfm_add_identity_adst_8x4_8bpc_avx2_inner
);
impl_8x4_ffi_wrapper!(
    inv_txfm_add_adst_identity_8x4_8bpc_avx2,
    inv_txfm_add_adst_identity_8x4_8bpc_avx2_inner
);
impl_8x4_ffi_wrapper!(
    inv_txfm_add_identity_flipadst_8x4_8bpc_avx2,
    inv_txfm_add_identity_flipadst_8x4_8bpc_avx2_inner
);
impl_8x4_ffi_wrapper!(
    inv_txfm_add_flipadst_identity_8x4_8bpc_avx2,
    inv_txfm_add_flipadst_identity_8x4_8bpc_avx2_inner
);

// ============================================================================
// 8x16 and 16x8 ADST/FLIPADST variants
// ============================================================================

/// Helper macro for 8x16 transforms with configurable row/col transforms
macro_rules! impl_8x16_transform {
    ($name:ident, $row_fn:ident, $col_fn:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[arcane]
        fn $name(
            _token: Desktop64,
            dst: &mut [u8],
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
            let row_clip_min = i16::MIN as i32;
            let row_clip_max = i16::MAX as i32;
            let col_clip_min = i16::MIN as i32;
            let col_clip_max = i16::MAX as i32;
            let mut tmp = [0i32; 128];

            let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

            // Row transform (8 elements each, 16 rows)
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

            // Column transform (16 elements each, 8 columns)
            for x in 0..8 {
                $col_fn(&mut tmp[x..], 8, col_clip_min, col_clip_max);
            }

            // Add to destination
            let zero = _mm_setzero_si128();
            let max_val = _mm_set1_epi16(bitdepth_max as i16);
            let rnd_final = _mm256_set1_epi32(8);

            for y in 0..16 {
                let dst_off = y * dst_stride;

                let d = loadi64!(&dst[dst_off..dst_off + 8]);
                let d16 = _mm_unpacklo_epi8(d, zero);

                let c_lo = _mm_set_epi32(
                    tmp[y * 8 + 3],
                    tmp[y * 8 + 2],
                    tmp[y * 8 + 1],
                    tmp[y * 8 + 0],
                );
                let c_hi = _mm_set_epi32(
                    tmp[y * 8 + 7],
                    tmp[y * 8 + 6],
                    tmp[y * 8 + 5],
                    tmp[y * 8 + 4],
                );

                let c_lo_256 = _mm256_set_m128i(c_hi, c_lo);
                let c_scaled = _mm256_srai_epi32(_mm256_add_epi32(c_lo_256, rnd_final), 4);

                let c_lo_scaled = _mm256_castsi256_si128(c_scaled);
                let c_hi_scaled = _mm256_extracti128_si256(c_scaled, 1);
                let c16 = _mm_packs_epi32(c_lo_scaled, c_hi_scaled);

                let sum = _mm_add_epi16(d16, c16);
                let clamped = _mm_max_epi16(_mm_min_epi16(sum, max_val), zero);
                let packed = _mm_packus_epi16(clamped, clamped);

                storei64!(&mut dst[dst_off..dst_off + 8], packed);
            }

            // Clear coefficients
            coeff[..128].fill(0);
        }
    };
}

/// Helper macro for 16x8 transforms with configurable row/col transforms
macro_rules! impl_16x8_transform {
    ($name:ident, $row_fn:ident, $col_fn:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[arcane]
        fn $name(
            _token: Desktop64,
            dst: &mut [u8],
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
            let row_clip_min = i16::MIN as i32;
            let row_clip_max = i16::MAX as i32;
            let col_clip_min = i16::MIN as i32;
            let col_clip_max = i16::MAX as i32;
            let mut tmp = [0i32; 128];

            let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

            // Row transform (16 elements each, 8 rows)
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

            // Column transform (8 elements each, 16 columns)
            for x in 0..16 {
                $col_fn(&mut tmp[x..], 16, col_clip_min, col_clip_max);
            }

            // Add to destination
            let zero = _mm256_setzero_si256();
            let max_val = _mm256_set1_epi16(bitdepth_max as i16);
            let rnd_final = _mm256_set1_epi32(8);

            for y in 0..8 {
                let dst_off = y * dst_stride;

                let d = loadu_128!(<&[u8; 16]>::try_from(&dst[dst_off..dst_off + 16]).unwrap());
                let d16 = _mm256_cvtepu8_epi16(d);

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

                let c0_scaled = _mm256_srai_epi32(_mm256_add_epi32(c0, rnd_final), 4);
                let c1_scaled = _mm256_srai_epi32(_mm256_add_epi32(c1, rnd_final), 4);

                let c16 = _mm256_packs_epi32(c0_scaled, c1_scaled);
                let c16 = _mm256_permute4x64_epi64(c16, 0b11_01_10_00);

                let sum = _mm256_add_epi16(d16, c16);
                let clamped = _mm256_max_epi16(_mm256_min_epi16(sum, max_val), zero);

                let packed = _mm256_packus_epi16(clamped, clamped);
                let packed = _mm256_permute4x64_epi64(packed, 0b11_01_10_00);

                storeu_128!(
                    <&mut [u8; 16]>::try_from(&mut dst[dst_off..dst_off + 16]).unwrap(),
                    _mm256_castsi256_si128(packed)
                );
            }

            // Clear coefficients
            coeff[..128].fill(0);
        }
    };
}

// Generate 8x16 ADST inner functions
impl_8x16_transform!(
    inv_txfm_add_adst_dct_8x16_8bpc_avx2_inner,
    adst8_1d,
    dct16_1d
);
impl_8x16_transform!(
    inv_txfm_add_dct_adst_8x16_8bpc_avx2_inner,
    dct8_1d,
    adst16_1d
);
impl_8x16_transform!(
    inv_txfm_add_adst_adst_8x16_8bpc_avx2_inner,
    adst8_1d,
    adst16_1d
);
impl_8x16_transform!(
    inv_txfm_add_flipadst_dct_8x16_8bpc_avx2_inner,
    flipadst8_1d,
    dct16_1d
);
impl_8x16_transform!(
    inv_txfm_add_dct_flipadst_8x16_8bpc_avx2_inner,
    dct8_1d,
    flipadst16_1d
);
impl_8x16_transform!(
    inv_txfm_add_flipadst_flipadst_8x16_8bpc_avx2_inner,
    flipadst8_1d,
    flipadst16_1d
);
impl_8x16_transform!(
    inv_txfm_add_adst_flipadst_8x16_8bpc_avx2_inner,
    adst8_1d,
    flipadst16_1d
);
impl_8x16_transform!(
    inv_txfm_add_flipadst_adst_8x16_8bpc_avx2_inner,
    flipadst8_1d,
    adst16_1d
);

// Generate 16x8 ADST inner functions
impl_16x8_transform!(
    inv_txfm_add_adst_dct_16x8_8bpc_avx2_inner,
    adst16_1d,
    dct8_1d
);
impl_16x8_transform!(
    inv_txfm_add_dct_adst_16x8_8bpc_avx2_inner,
    dct16_1d,
    adst8_1d
);
impl_16x8_transform!(
    inv_txfm_add_adst_adst_16x8_8bpc_avx2_inner,
    adst16_1d,
    adst8_1d
);
impl_16x8_transform!(
    inv_txfm_add_flipadst_dct_16x8_8bpc_avx2_inner,
    flipadst16_1d,
    dct8_1d
);
impl_16x8_transform!(
    inv_txfm_add_dct_flipadst_16x8_8bpc_avx2_inner,
    dct16_1d,
    flipadst8_1d
);
impl_16x8_transform!(
    inv_txfm_add_flipadst_flipadst_16x8_8bpc_avx2_inner,
    flipadst16_1d,
    flipadst8_1d
);
impl_16x8_transform!(
    inv_txfm_add_adst_flipadst_16x8_8bpc_avx2_inner,
    adst16_1d,
    flipadst8_1d
);
impl_16x8_transform!(
    inv_txfm_add_flipadst_adst_16x8_8bpc_avx2_inner,
    flipadst16_1d,
    adst8_1d
);

/// FFI wrapper macro for 8x16 transforms
macro_rules! impl_8x16_ffi_wrapper {
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

/// FFI wrapper macro for 16x8 transforms
macro_rules! impl_16x8_ffi_wrapper {
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

// Generate 8x16 FFI wrappers
impl_8x16_ffi_wrapper!(
    inv_txfm_add_adst_dct_8x16_8bpc_avx2,
    inv_txfm_add_adst_dct_8x16_8bpc_avx2_inner
);
impl_8x16_ffi_wrapper!(
    inv_txfm_add_dct_adst_8x16_8bpc_avx2,
    inv_txfm_add_dct_adst_8x16_8bpc_avx2_inner
);
impl_8x16_ffi_wrapper!(
    inv_txfm_add_adst_adst_8x16_8bpc_avx2,
    inv_txfm_add_adst_adst_8x16_8bpc_avx2_inner
);
impl_8x16_ffi_wrapper!(
    inv_txfm_add_flipadst_dct_8x16_8bpc_avx2,
    inv_txfm_add_flipadst_dct_8x16_8bpc_avx2_inner
);
impl_8x16_ffi_wrapper!(
    inv_txfm_add_dct_flipadst_8x16_8bpc_avx2,
    inv_txfm_add_dct_flipadst_8x16_8bpc_avx2_inner
);
impl_8x16_ffi_wrapper!(
    inv_txfm_add_flipadst_flipadst_8x16_8bpc_avx2,
    inv_txfm_add_flipadst_flipadst_8x16_8bpc_avx2_inner
);
impl_8x16_ffi_wrapper!(
    inv_txfm_add_adst_flipadst_8x16_8bpc_avx2,
    inv_txfm_add_adst_flipadst_8x16_8bpc_avx2_inner
);
impl_8x16_ffi_wrapper!(
    inv_txfm_add_flipadst_adst_8x16_8bpc_avx2,
    inv_txfm_add_flipadst_adst_8x16_8bpc_avx2_inner
);

// Generate 16x8 FFI wrappers
impl_16x8_ffi_wrapper!(
    inv_txfm_add_adst_dct_16x8_8bpc_avx2,
    inv_txfm_add_adst_dct_16x8_8bpc_avx2_inner
);
impl_16x8_ffi_wrapper!(
    inv_txfm_add_dct_adst_16x8_8bpc_avx2,
    inv_txfm_add_dct_adst_16x8_8bpc_avx2_inner
);
impl_16x8_ffi_wrapper!(
    inv_txfm_add_adst_adst_16x8_8bpc_avx2,
    inv_txfm_add_adst_adst_16x8_8bpc_avx2_inner
);
impl_16x8_ffi_wrapper!(
    inv_txfm_add_flipadst_dct_16x8_8bpc_avx2,
    inv_txfm_add_flipadst_dct_16x8_8bpc_avx2_inner
);
impl_16x8_ffi_wrapper!(
    inv_txfm_add_dct_flipadst_16x8_8bpc_avx2,
    inv_txfm_add_dct_flipadst_16x8_8bpc_avx2_inner
);
impl_16x8_ffi_wrapper!(
    inv_txfm_add_flipadst_flipadst_16x8_8bpc_avx2,
    inv_txfm_add_flipadst_flipadst_16x8_8bpc_avx2_inner
);
impl_16x8_ffi_wrapper!(
    inv_txfm_add_adst_flipadst_16x8_8bpc_avx2,
    inv_txfm_add_adst_flipadst_16x8_8bpc_avx2_inner
);
impl_16x8_ffi_wrapper!(
    inv_txfm_add_flipadst_adst_16x8_8bpc_avx2,
    inv_txfm_add_flipadst_adst_16x8_8bpc_avx2_inner
);

// IDTX for 8x16 and 16x8
impl_8x16_transform!(
    inv_txfm_add_identity_identity_8x16_8bpc_avx2_inner,
    identity8_1d,
    identity16_1d
);
impl_16x8_transform!(
    inv_txfm_add_identity_identity_16x8_8bpc_avx2_inner,
    identity16_1d,
    identity8_1d
);
impl_8x16_ffi_wrapper!(
    inv_txfm_add_identity_identity_8x16_8bpc_avx2,
    inv_txfm_add_identity_identity_8x16_8bpc_avx2_inner
);
impl_16x8_ffi_wrapper!(
    inv_txfm_add_identity_identity_16x8_8bpc_avx2,
    inv_txfm_add_identity_identity_16x8_8bpc_avx2_inner
);

// H_DCT and V_DCT for 8x16
impl_8x16_transform!(
    inv_txfm_add_identity_dct_8x16_8bpc_avx2_inner,
    identity8_1d,
    dct16_1d
);
impl_8x16_transform!(
    inv_txfm_add_dct_identity_8x16_8bpc_avx2_inner,
    dct8_1d,
    identity16_1d
);
impl_8x16_ffi_wrapper!(
    inv_txfm_add_identity_dct_8x16_8bpc_avx2,
    inv_txfm_add_identity_dct_8x16_8bpc_avx2_inner
);
impl_8x16_ffi_wrapper!(
    inv_txfm_add_dct_identity_8x16_8bpc_avx2,
    inv_txfm_add_dct_identity_8x16_8bpc_avx2_inner
);

// H_DCT and V_DCT for 16x8
impl_16x8_transform!(
    inv_txfm_add_identity_dct_16x8_8bpc_avx2_inner,
    identity16_1d,
    dct8_1d
);
impl_16x8_transform!(
    inv_txfm_add_dct_identity_16x8_8bpc_avx2_inner,
    dct16_1d,
    identity8_1d
);
impl_16x8_ffi_wrapper!(
    inv_txfm_add_identity_dct_16x8_8bpc_avx2,
    inv_txfm_add_identity_dct_16x8_8bpc_avx2_inner
);
impl_16x8_ffi_wrapper!(
    inv_txfm_add_dct_identity_16x8_8bpc_avx2,
    inv_txfm_add_dct_identity_16x8_8bpc_avx2_inner
);

// H_ADST, V_ADST, H_FLIPADST, V_FLIPADST for 8x16
impl_8x16_transform!(
    inv_txfm_add_identity_adst_8x16_8bpc_avx2_inner,
    identity8_1d,
    adst16_1d
);
impl_8x16_transform!(
    inv_txfm_add_adst_identity_8x16_8bpc_avx2_inner,
    adst8_1d,
    identity16_1d
);
impl_8x16_transform!(
    inv_txfm_add_identity_flipadst_8x16_8bpc_avx2_inner,
    identity8_1d,
    flipadst16_1d
);
impl_8x16_transform!(
    inv_txfm_add_flipadst_identity_8x16_8bpc_avx2_inner,
    flipadst8_1d,
    identity16_1d
);
impl_8x16_ffi_wrapper!(
    inv_txfm_add_identity_adst_8x16_8bpc_avx2,
    inv_txfm_add_identity_adst_8x16_8bpc_avx2_inner
);
impl_8x16_ffi_wrapper!(
    inv_txfm_add_adst_identity_8x16_8bpc_avx2,
    inv_txfm_add_adst_identity_8x16_8bpc_avx2_inner
);
impl_8x16_ffi_wrapper!(
    inv_txfm_add_identity_flipadst_8x16_8bpc_avx2,
    inv_txfm_add_identity_flipadst_8x16_8bpc_avx2_inner
);
impl_8x16_ffi_wrapper!(
    inv_txfm_add_flipadst_identity_8x16_8bpc_avx2,
    inv_txfm_add_flipadst_identity_8x16_8bpc_avx2_inner
);

// H_ADST, V_ADST, H_FLIPADST, V_FLIPADST for 16x8
impl_16x8_transform!(
    inv_txfm_add_identity_adst_16x8_8bpc_avx2_inner,
    identity16_1d,
    adst8_1d
);
impl_16x8_transform!(
    inv_txfm_add_adst_identity_16x8_8bpc_avx2_inner,
    adst16_1d,
    identity8_1d
);
impl_16x8_transform!(
    inv_txfm_add_identity_flipadst_16x8_8bpc_avx2_inner,
    identity16_1d,
    flipadst8_1d
);
impl_16x8_transform!(
    inv_txfm_add_flipadst_identity_16x8_8bpc_avx2_inner,
    flipadst16_1d,
    identity8_1d
);
impl_16x8_ffi_wrapper!(
    inv_txfm_add_identity_adst_16x8_8bpc_avx2,
    inv_txfm_add_identity_adst_16x8_8bpc_avx2_inner
);
impl_16x8_ffi_wrapper!(
    inv_txfm_add_adst_identity_16x8_8bpc_avx2,
    inv_txfm_add_adst_identity_16x8_8bpc_avx2_inner
);
impl_16x8_ffi_wrapper!(
    inv_txfm_add_identity_flipadst_16x8_8bpc_avx2,
    inv_txfm_add_identity_flipadst_16x8_8bpc_avx2_inner
);
impl_16x8_ffi_wrapper!(
    inv_txfm_add_flipadst_identity_16x8_8bpc_avx2,
    inv_txfm_add_flipadst_identity_16x8_8bpc_avx2_inner
);

/// Full 2D DCT_DCT 8x8 inverse transform
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_8x16_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    // W=8, H=16, shift=1 for 8x16
    let row_clip_min = i16::MIN as i32;
    let row_clip_max = i16::MAX as i32;
    let col_clip_min = i16::MIN as i32;
    let col_clip_max = i16::MAX as i32;
    let mut tmp = [0i32; 128];

    // SIMD row transform: 16 rows (AVX-512 16-row path or AVX2 2x8). rect2, shift=1, rnd=1.
    {
        let coeff_slice = coeff.as_slice();
        row_dct8_8bpc_block(
            _token,
            coeff_slice,
            16,
            16,
            true,
            1,
            1,
            &mut tmp,
            row_clip_min,
            row_clip_max,
            col_clip_min,
            col_clip_max,
        );
    }

    // Column transform: SIMD across 8 columns (single chunk, 16 rows)
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
    let max_val = _mm_set1_epi16(bitdepth_max as i16);
    let rnd_final = _mm256_set1_epi32(8);

    for y in 0..16 {
        let dst_off = y * dst_stride;

        let d = loadi64!(&dst[dst_off..dst_off + 8]);
        let d16 = _mm_unpacklo_epi8(d, zero);

        let c_lo = _mm_set_epi32(
            tmp[y * 8 + 3],
            tmp[y * 8 + 2],
            tmp[y * 8 + 1],
            tmp[y * 8 + 0],
        );
        let c_hi = _mm_set_epi32(
            tmp[y * 8 + 7],
            tmp[y * 8 + 6],
            tmp[y * 8 + 5],
            tmp[y * 8 + 4],
        );

        let c_lo_256 = _mm256_set_m128i(c_hi, c_lo);
        let c_scaled = _mm256_srai_epi32(_mm256_add_epi32(c_lo_256, rnd_final), 4);

        let c_lo_scaled = _mm256_castsi256_si128(c_scaled);
        let c_hi_scaled = _mm256_extracti128_si256(c_scaled, 1);
        let c16 = _mm_packs_epi32(c_lo_scaled, c_hi_scaled);

        let sum = _mm_add_epi16(d16, c16);
        let clamped = _mm_max_epi16(_mm_min_epi16(sum, max_val), zero);
        let packed = _mm_packus_epi16(clamped, clamped);

        storei64!(&mut dst[dst_off..dst_off + 8], packed);
    }

    // Clear coefficients
    coeff[..128].fill(0);
}

/// FFI wrapper for 8x16 DCT_DCT 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_8x16_8bpc_avx2(
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
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, _coeff_len as usize * stride + stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_8x16_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// Full 2D DCT_DCT 16x8 inverse transform
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_16x8_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    // W=16, H=8, shift=1 for 16x8
    let row_clip_min = i16::MIN as i32;
    let row_clip_max = i16::MAX as i32;
    let col_clip_min = i16::MIN as i32;
    let col_clip_max = i16::MAX as i32;
    let mut tmp = [0i32; 128];

    // SIMD row transform: 1 batch of 8 rows. is_rect2=true, shift=1, rnd=1.
    {
        let coeff_slice = coeff.as_slice();
        simd_row_dct16_8bpc_8rows(
            _token,
            coeff_slice,
            8,
            0,
            true,
            1,
            1,
            &mut tmp,
            row_clip_min,
            row_clip_max,
            col_clip_min,
            col_clip_max,
        );
    }

    // Column transform: SIMD across 16 columns (2 chunks of 8, 8 rows)
    {
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
    let max_val = _mm256_set1_epi16(bitdepth_max as i16);
    let rnd_final = _mm256_set1_epi32(8);

    for y in 0..8 {
        let dst_off = y * dst_stride;

        let d = loadu_128!(<&[u8; 16]>::try_from(&dst[dst_off..dst_off + 16]).unwrap());
        let d16 = _mm256_cvtepu8_epi16(d);

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

        let c0_scaled = _mm256_srai_epi32(_mm256_add_epi32(c0, rnd_final), 4);
        let c1_scaled = _mm256_srai_epi32(_mm256_add_epi32(c1, rnd_final), 4);

        let c16 = _mm256_packs_epi32(c0_scaled, c1_scaled);
        let c16 = _mm256_permute4x64_epi64(c16, 0b11_01_10_00);

        let sum = _mm256_add_epi16(d16, c16);
        let clamped = _mm256_max_epi16(_mm256_min_epi16(sum, max_val), zero);

        let packed = _mm256_packus_epi16(clamped, clamped);
        let packed = _mm256_permute4x64_epi64(packed, 0b11_01_10_00);

        storeu_128!(
            <&mut [u8; 16]>::try_from(&mut dst[dst_off..dst_off + 16]).unwrap(),
            _mm256_castsi256_si128(packed)
        );
    }

    // Clear coefficients
    coeff[..128].fill(0);
}

/// FFI wrapper for 16x8 DCT_DCT 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_16x8_8bpc_avx2(
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
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, _coeff_len as usize * stride + stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_16x8_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

