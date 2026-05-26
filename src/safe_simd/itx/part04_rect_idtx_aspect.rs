// ============================================================================
// RECTANGULAR TRANSFORMS - 16x32, 32x16
// ============================================================================

/// Full 2D DCT_DCT 16x32 inverse transform
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_16x32_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    // W=16, H=32, shift=2 for 16x32
    let row_clip_min = i16::MIN as i32;
    let row_clip_max = i16::MAX as i32;
    let col_clip_min = i16::MIN as i32;
    let col_clip_max = i16::MAX as i32;
    let mut tmp = [0i32; 16 * 32];

    // SIMD row transform: 4 batches of 8 rows. is_rect2=true, shift=1, rnd=1.
    {
        let coeff_slice = coeff.as_slice();
        for y_base in [0usize, 8, 16, 24] {
            simd_row_dct16_8bpc_8rows(
                _token,
                coeff_slice,
                32,
                y_base,
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
            dct32_1d_cols8_i16(_token, &mut v, min_v, max_v);
            for i in 0..32 {
                storeu_256!(&mut tmp[i * 16 + cx..i * 16 + cx + 8], [i32; 8], v[i]);
            }
        }
    }

    // Add to destination
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi16(bitdepth_max as i16);
    let rnd_final = _mm256_set1_epi32(8);

    for y in 0..32 {
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
    coeff[..512].fill(0);
}

/// FFI wrapper for 16x32 DCT_DCT 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_16x32_8bpc_avx2(
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

    inv_txfm_add_dct_dct_16x32_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// SIMD row ADST-8 for 8xN transforms, 8bpc. Same shape as
/// `simd_row_dct8_8bpc_8rows` but calls `adst8_1d_cols8`. If `flipped`,
/// reverses output order after ADST (flipadst). Currently unwired — kept
/// for the future 8x16/8x32 mixed-row adst transform refactor.
#[cfg(target_arch = "x86_64")]
#[rite]
#[inline(always)]
#[allow(dead_code)]
fn simd_row_adst8_8bpc_8rows(
    token: Desktop64,
    coeff: &[i16],
    coeff_h: usize,
    y_base: usize,
    apply_rect2: bool,
    flipped: bool,
    rnd: i32,
    shift: i32,
    tmp: &mut [i32],
    row_min: i32,
    row_max: i32,
    col_min: i32,
    col_max: i32,
) {
    let row_min_v = _mm256_set1_epi32(row_min);
    let row_max_v = _mm256_set1_epi32(row_max);
    let col_min_v = _mm256_set1_epi32(col_min);
    let col_max_v = _mm256_set1_epi32(col_max);
    let rect2_v = _mm256_set1_epi32(181);
    let bias_v = _mm256_set1_epi32(128);
    let rnd_v = _mm256_set1_epi32(rnd);
    let mut cols = [_mm256_setzero_si256(); 8];
    for x in 0..8 {
        let off = y_base + x * coeff_h;
        let arr: &[i16; 8] = (&coeff[off..off + 8]).try_into().unwrap();
        let v16 = loadu_128!(arr);
        let v32 = _mm256_cvtepi16_epi32(v16);
        cols[x] = if apply_rect2 {
            _mm256_srai_epi32::<8>(_mm256_add_epi32(_mm256_mullo_epi32(v32, rect2_v), bias_v))
        } else {
            v32
        };
    }
    adst8_1d_cols8(token, &mut cols, row_min_v, row_max_v);
    if flipped {
        // ADST output reversed: c[i] <- c[7-i]
        cols.reverse();
    }
    for x in 0..8 {
        let rounded = match shift {
            0 => _mm256_add_epi32(cols[x], rnd_v),
            1 => _mm256_srai_epi32::<1>(_mm256_add_epi32(cols[x], rnd_v)),
            2 => _mm256_srai_epi32::<2>(_mm256_add_epi32(cols[x], rnd_v)),
            _ => _mm256_srai_epi32::<2>(_mm256_add_epi32(cols[x], rnd_v)),
        };
        cols[x] = _mm256_max_epi32(_mm256_min_epi32(rounded, col_max_v), col_min_v);
    }
    let rows = transpose_8x8_i32!(cols);
    let s = 8;
    storeu_256!(
        &mut tmp[(y_base + 0) * s..(y_base + 0) * s + 8],
        [i32; 8],
        rows[0]
    );
    storeu_256!(
        &mut tmp[(y_base + 1) * s..(y_base + 1) * s + 8],
        [i32; 8],
        rows[1]
    );
    storeu_256!(
        &mut tmp[(y_base + 2) * s..(y_base + 2) * s + 8],
        [i32; 8],
        rows[2]
    );
    storeu_256!(
        &mut tmp[(y_base + 3) * s..(y_base + 3) * s + 8],
        [i32; 8],
        rows[3]
    );
    storeu_256!(
        &mut tmp[(y_base + 4) * s..(y_base + 4) * s + 8],
        [i32; 8],
        rows[4]
    );
    storeu_256!(
        &mut tmp[(y_base + 5) * s..(y_base + 5) * s + 8],
        [i32; 8],
        rows[5]
    );
    storeu_256!(
        &mut tmp[(y_base + 6) * s..(y_base + 6) * s + 8],
        [i32; 8],
        rows[6]
    );
    storeu_256!(
        &mut tmp[(y_base + 7) * s..(y_base + 7) * s + 8],
        [i32; 8],
        rows[7]
    );
}

/// SIMD row DCT-8 for 8xN transforms, 8bpc. Processes 8 rows at once via
/// `dct8_1d_cols8` + 8x8 transpose. Coeff is column-major (stride `coeff_h`);
/// writes row-major into `tmp` (stride 8).
#[cfg(target_arch = "x86_64")]
#[rite]
#[inline(always)]
fn simd_row_dct8_8bpc_8rows(
    token: Desktop64,
    coeff: &[i16],
    coeff_h: usize,
    y_base: usize,
    apply_rect2: bool,
    rnd: i32,
    shift: i32,
    tmp: &mut [i32],
    row_min: i32,
    row_max: i32,
    col_min: i32,
    col_max: i32,
) {
    let row_min_v = _mm256_set1_epi32(row_min);
    let row_max_v = _mm256_set1_epi32(row_max);
    let col_min_v = _mm256_set1_epi32(col_min);
    let col_max_v = _mm256_set1_epi32(col_max);
    let rect2_v = _mm256_set1_epi32(181);
    let bias_v = _mm256_set1_epi32(128);
    let rnd_v = _mm256_set1_epi32(rnd);
    let mut cols = [_mm256_setzero_si256(); 8];
    for x in 0..8 {
        let off = y_base + x * coeff_h;
        let arr: &[i16; 8] = (&coeff[off..off + 8]).try_into().unwrap();
        let v16 = loadu_128!(arr);
        let v32 = _mm256_cvtepi16_epi32(v16);
        cols[x] = if apply_rect2 {
            _mm256_srai_epi32::<8>(_mm256_add_epi32(_mm256_mullo_epi32(v32, rect2_v), bias_v))
        } else {
            v32
        };
    }
    dct8_1d_cols8(token, &mut cols, row_min_v, row_max_v);
    for x in 0..8 {
        let rounded = match shift {
            0 => _mm256_add_epi32(cols[x], rnd_v),
            1 => _mm256_srai_epi32::<1>(_mm256_add_epi32(cols[x], rnd_v)),
            2 => _mm256_srai_epi32::<2>(_mm256_add_epi32(cols[x], rnd_v)),
            _ => _mm256_srai_epi32::<2>(_mm256_add_epi32(cols[x], rnd_v)),
        };
        cols[x] = _mm256_max_epi32(_mm256_min_epi32(rounded, col_max_v), col_min_v);
    }
    // 8x8 transpose, store row-major (stride 8).
    let rows = transpose_8x8_i32!(cols);
    let s = 8;
    storeu_256!(
        &mut tmp[(y_base + 0) * s..(y_base + 0) * s + 8],
        [i32; 8],
        rows[0]
    );
    storeu_256!(
        &mut tmp[(y_base + 1) * s..(y_base + 1) * s + 8],
        [i32; 8],
        rows[1]
    );
    storeu_256!(
        &mut tmp[(y_base + 2) * s..(y_base + 2) * s + 8],
        [i32; 8],
        rows[2]
    );
    storeu_256!(
        &mut tmp[(y_base + 3) * s..(y_base + 3) * s + 8],
        [i32; 8],
        rows[3]
    );
    storeu_256!(
        &mut tmp[(y_base + 4) * s..(y_base + 4) * s + 8],
        [i32; 8],
        rows[4]
    );
    storeu_256!(
        &mut tmp[(y_base + 5) * s..(y_base + 5) * s + 8],
        [i32; 8],
        rows[5]
    );
    storeu_256!(
        &mut tmp[(y_base + 6) * s..(y_base + 6) * s + 8],
        [i32; 8],
        rows[6]
    );
    storeu_256!(
        &mut tmp[(y_base + 7) * s..(y_base + 7) * s + 8],
        [i32; 8],
        rows[7]
    );
}

// ============================================================================
// i16-PACKED PMADDWD DCT-8 ROW PASS — attempt #5
// ============================================================================
//
// Goal: replace the existing cross-column i32 row pass (which uses
// `_mm256_mullo_epi32` for the multiplicative stages) with a `pmaddwd`-
// driven i16-packed version mirroring dav1d's `IDCT8_1D_PACKED`
// (`src/x86/itx_avx2.asm:566`). pmaddwd does 2 multiplies + 1 add per
// 32-bit lane in one op (vs `mullo_epi32` + `add` for the i32 path),
// roughly halving the multiplicative-stage instruction count.
//
// Iteration loop: standalone fn + bit-exact unit test vs
// `run_scalar_dct8_per_row`. Wire-up into the live 8x8/8x16/8x32
// transforms is a SEPARATE step after the standalone passes.

/// i16-packed pmaddwd DCT-8 1D row pass. Takes 64 i16 column-major
/// coeffs (input shape: `coeff[y + x*8]` is element `x` of row `y`),
/// runs DCT-8 independently across each of the 8 rows, and returns
/// row-major i32 output (output shape: `out[y*8 + x]` is element `x`
/// of row `y`).
///
/// Bit-exactness target: matches `run_scalar_dct8_per_row` exactly for
/// `row_min = i16::MIN as i32`, `row_max = i16::MAX as i32` (the 8bpc
/// row-pass clip range).
/// Helper: build (col_a, col_b) ymm pair from two i16x8 xmms.
/// Each i32 lane K (K=0..7 → row K) holds (col_a_word, col_b_word) packed as
/// i16 ready for pmaddwd. low half of lane = col_a, high half = col_b.
#[cfg(target_arch = "x86_64")]
#[rite]
#[inline(always)]
fn dct8_row_build_pair(_token: Desktop64, a: __m128i, b: __m128i) -> __m256i {
    let lo = _mm_unpacklo_epi16(a, b);
    let hi = _mm_unpackhi_epi16(a, b);
    _mm256_set_m128i(hi, lo)
}

/// Helper: build a coef-pack ymm broadcast where each i32 lane = (c_lo, c_hi)
/// packed as i16 (low 16 bits = c_lo, high 16 = c_hi). Used as the 2nd arg to
/// `_mm256_madd_epi16` so the result per-lane = `a*c_lo + b*c_hi` for input
/// pair (a, b).
#[cfg(target_arch = "x86_64")]
#[rite]
#[inline(always)]
fn dct8_row_coef_pack(_token: Desktop64, c_lo: i16, c_hi: i16) -> __m256i {
    let packed = ((c_lo as u32) & 0xFFFF) as i32 | (((c_hi as u32) & 0xFFFF) << 16) as i32;
    _mm256_set1_epi32(packed)
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn dct8_row_pass_i16_simd(_token: Desktop64, coeff_col_major: [i16; 64]) -> [i32; 64] {
    // Layout: coeff_col_major[y + x*8] = element x of row y.
    // We process all 8 rows in parallel — ymm lane K corresponds to row K.
    //
    // Pair construction strategy: load each column as 8 i16 into an xmm,
    // then build ymm pairs where each i32 lane K = (col_a's row K word, col_b's row K word).
    //
    //   xmm = _mm_loadu_si128(&coeff[col*8])   // 8 i16 = all 8 rows of one col
    //   ymm_pair_ab = lo_unpack(col_a, col_b) | hi_unpack(col_a, col_b)
    //
    // That ymm now has one i32 lane per row, each holding (col_a_word, col_b_word)
    // packed as i16 ready for pmaddwd.
    //
    // Coefficient pack convention: pmaddwd with packed (p_lo, p_hi) on input
    // (a_lo, a_hi) computes a_lo*p_lo + a_hi*p_hi per i32 lane. So to compute
    // `c1 * col_a + c2 * col_b` we pack (c1, c2) and use input (col_a, col_b).

    // Build column xmms.
    let mut col_xmm = [_mm_setzero_si128(); 8];
    for x in 0..8 {
        let arr: &[i16; 8] = (&coeff_col_major[x * 8..x * 8 + 8]).try_into().unwrap();
        col_xmm[x] = loadu_128!(arr);
    }

    // ----- Build (col_a, col_b) pairs for pmaddwd -----
    // Each ymm pair contains 8 i32 lanes (one per row); each lane holds
    // (col_a_word, col_b_word) packed as i16. pmaddwd with coef_pack(c_a, c_b)
    // then computes (c_a * col_a + c_b * col_b) per i32 lane = per row.
    let pair_17 = dct8_row_build_pair(_token, col_xmm[1], col_xmm[7]);
    let pair_53 = dct8_row_build_pair(_token, col_xmm[5], col_xmm[3]);
    let pair_26 = dct8_row_build_pair(_token, col_xmm[2], col_xmm[6]);
    let pair_04 = dct8_row_build_pair(_token, col_xmm[0], col_xmm[4]);

    let pd_2048 = _mm256_set1_epi32(2048);
    let pd_128 = _mm256_set1_epi32(128);

    // ----- Multiplicative stages — bit-exact to scalar C reference -----
    // All formulas derived in the function-level comment trace above:
    //   t4a = (in1*799 - in7*4017 + 2048) >> 12     [pair_17 = (in1, in7), coefs (799, -4017)]
    //   t7a = (in1*4017 + in7*799  + 2048) >> 12    [coefs (4017, 799)]
    //   t5a = (in5*3406 - in3*2276 + 2048) >> 12    [pair_53 = (in5, in3), coefs (3406, -2276)]
    //         (= (in5*1703 - in3*1138 + 1024) >> 11)
    //   t6a = (in5*2276 + in3*3406 + 2048) >> 12    [coefs (2276, 3406)]
    //         (= (in5*1138 + in3*1703 + 1024) >> 11)
    //   t2  = (in2*1567 - in6*3784 + 2048) >> 12    [pair_26 = (in2, in6), coefs (1567, -3784)]
    //   t3  = (in2*3784 + in6*1567 + 2048) >> 12    [coefs (3784, 1567)]
    //   t0  = (in0*181  + in4*181  + 128 ) >> 8     [pair_04 = (in0, in4), coefs (181, 181)]
    //   t1  = (in0*181  + in4*-181 + 128 ) >> 8     [coefs (181, -181)]
    let t4a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_17, dct8_row_coef_pack(_token, 799, -4017)),
        pd_2048,
    ));
    let t7a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_17, dct8_row_coef_pack(_token, 4017, 799)),
        pd_2048,
    ));
    let t5a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_53, dct8_row_coef_pack(_token, 3406, -2276)),
        pd_2048,
    ));
    let t6a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_53, dct8_row_coef_pack(_token, 2276, 3406)),
        pd_2048,
    ));
    let t2 = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_26, dct8_row_coef_pack(_token, 1567, -3784)),
        pd_2048,
    ));
    let t3 = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_26, dct8_row_coef_pack(_token, 3784, 1567)),
        pd_2048,
    ));
    let t0 = _mm256_srai_epi32::<8>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_04, dct8_row_coef_pack(_token, 181, 181)),
        pd_128,
    ));
    let t1 = _mm256_srai_epi32::<8>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_04, dct8_row_coef_pack(_token, 181, -181)),
        pd_128,
    ));

    // ----- Additive butterfly stages — match scalar exactly (clip vs row range) -----
    // Use i32 add/sub + max/min clamp (= iclip semantics).
    let row_min = i16::MIN as i32;
    let row_max = i16::MAX as i32;
    let row_min_v = _mm256_set1_epi32(row_min);
    let row_max_v = _mm256_set1_epi32(row_max);
    let clip = |v: __m256i| _mm256_max_epi32(_mm256_min_epi32(v, row_max_v), row_min_v);

    // Stage 1 butterfly on (t4a, t5a) and (t6a, t7a):
    //   t4   = clip(t4a + t5a)
    //   t5a' = clip(t4a - t5a)
    //   t7   = clip(t7a + t6a)
    //   t6a' = clip(t7a - t6a)
    let t4 = clip(_mm256_add_epi32(t4a, t5a));
    let t5a_n = clip(_mm256_sub_epi32(t4a, t5a));
    let t7 = clip(_mm256_add_epi32(t7a, t6a));
    let t6a_n = clip(_mm256_sub_epi32(t7a, t6a));

    // Stage 2 — t5/t6 via the 181 sqrt(2) coef:
    //   t5 = ((t6a_n - t5a_n) * 181 + 128) >> 8
    //   t6 = ((t6a_n + t5a_n) * 181 + 128) >> 8
    let c_181 = _mm256_set1_epi32(181);
    let d = _mm256_sub_epi32(t6a_n, t5a_n);
    let t5 = _mm256_srai_epi32::<8>(_mm256_add_epi32(_mm256_mullo_epi32(d, c_181), pd_128));
    let s = _mm256_add_epi32(t6a_n, t5a_n);
    let t6 = _mm256_srai_epi32::<8>(_mm256_add_epi32(_mm256_mullo_epi32(s, c_181), pd_128));

    // DCT-4 even side: t0..t3 already computed above (full DCT-4 closed form).
    // Combine to butterfly outputs (asm tmp0..tmp3 + DCT-8 final butterfly):
    //   tmp0 = clip(t0 + t3); tmp3 = clip(t0 - t3)
    //   tmp1 = clip(t1 + t2); tmp2 = clip(t1 - t2)
    // (These match dct4_1d_internal_c's c[0]..c[3] = clip(t0+t3), clip(t1+t2), clip(t1-t2), clip(t0-t3))
    let tmp0 = clip(_mm256_add_epi32(t0, t3));
    let tmp1 = clip(_mm256_add_epi32(t1, t2));
    let tmp2 = clip(_mm256_sub_epi32(t1, t2));
    let tmp3 = clip(_mm256_sub_epi32(t0, t3));

    // Final DCT-8 butterfly:
    //   out0 = clip(tmp0 + t7)
    //   out1 = clip(tmp1 + t6)
    //   out2 = clip(tmp2 + t5)
    //   out3 = clip(tmp3 + t4)
    //   out4 = clip(tmp3 - t4)
    //   out5 = clip(tmp2 - t5)
    //   out6 = clip(tmp1 - t6)
    //   out7 = clip(tmp0 - t7)
    let mut cols = [_mm256_setzero_si256(); 8];
    cols[0] = clip(_mm256_add_epi32(tmp0, t7));
    cols[1] = clip(_mm256_add_epi32(tmp1, t6));
    cols[2] = clip(_mm256_add_epi32(tmp2, t5));
    cols[3] = clip(_mm256_add_epi32(tmp3, t4));
    cols[4] = clip(_mm256_sub_epi32(tmp3, t4));
    cols[5] = clip(_mm256_sub_epi32(tmp2, t5));
    cols[6] = clip(_mm256_sub_epi32(tmp1, t6));
    cols[7] = clip(_mm256_sub_epi32(tmp0, t7));

    // Transpose 8x8 i32 col-major → row-major and store.
    let rows = transpose_8x8_i32!(cols);
    let mut out = [0i32; 64];
    for y in 0..8 {
        let arr: &mut [i32; 8] = (&mut out[y * 8..y * 8 + 8]).try_into().unwrap();
        storeu_256!(arr, [i32; 8], rows[y]);
    }
    out
}

/// i16-packed pmaddwd DCT-8 column pass.
///
/// Takes row-major i32 `tmp[y*8 + x]` (8 rows × 8 cols, values in i16 range after
/// intermediate shift+clip), runs the DCT-8 column transform using `_mm256_madd_epi16`
/// (pmaddwd) instead of `_mm256_mullo_epi32`, and returns row-major i32 output
/// ready for add-to-dst.
///
/// Each ymm lane (K=0..7) corresponds to column K. For a given "row" index r in the
/// DCT, we have: `xmm[r]` = 8 i16 values (one per column). We pair rows via
/// `dct8_row_build_pair(xmm[row_a], xmm[row_b])` → ymm where each i32 lane K =
/// (row_a_colK, row_b_colK) ready for pmaddwd.
///
/// Benefits vs i32 mullo column pass: replaces 16 `mullo_epi32` (10c Zen3) with
/// 10 `madd_epi16` (5c Zen3) = ~100 cycles saved per 8x8 block.
#[cfg(target_arch = "x86_64")]
#[rite]
fn dct8_col_pass_i16(_token: Desktop64, tmp_row_major: &[i32; 64]) -> [__m256i; 8] {
    // Step 1: Convert each row from 8 × i32 to 8 × i16 in an xmm.
    // Values are guaranteed clipped to i16 range by the intermediate shift+clip.
    let mut row_xmm = [_mm_setzero_si128(); 8];
    for y in 0..8 {
        let v = loadu_256!(&tmp_row_major[y * 8..y * 8 + 8], [i32; 8]);
        let lo128 = _mm256_castsi256_si128(v);
        let hi128 = _mm256_extracti128_si256(v, 1);
        // _mm_packs_epi32: [a0,a1,a2,a3] + [b0,b1,b2,b3] → [a0,a1,a2,a3,b0,b1,b2,b3] i16
        // Since values are in i16 range, saturation is lossless.
        row_xmm[y] = _mm_packs_epi32(lo128, hi128);
    }

    // Step 2: Build (row_a, row_b) pairs for pmaddwd.
    // DCT-8 inputs: even side uses rows 0,2,4,6; odd side uses rows 1,3,5,7.
    // Pair convention matches dct8_row_pass_i16_simd:
    //   pair_17 = (row1, row7) — for t4a, t7a
    //   pair_53 = (row5, row3) — for t5a, t6a
    //   pair_26 = (row2, row6) — for t2, t3  (DCT-4 even)
    //   pair_04 = (row0, row4) — for t0, t1  (DCT-4 even)
    let pair_17 = dct8_row_build_pair(_token, row_xmm[1], row_xmm[7]);
    let pair_53 = dct8_row_build_pair(_token, row_xmm[5], row_xmm[3]);
    let pair_26 = dct8_row_build_pair(_token, row_xmm[2], row_xmm[6]);
    let pair_04 = dct8_row_build_pair(_token, row_xmm[0], row_xmm[4]);

    let pd_2048 = _mm256_set1_epi32(2048);
    let pd_128 = _mm256_set1_epi32(128);

    // Step 3: Multiplicative stages via pmaddwd (same formulas as row pass).
    //   t4a = (in1*799 - in7*4017 + 2048) >> 12
    //   t7a = (in1*4017 + in7*799  + 2048) >> 12
    //   t5a = (in5*3406 - in3*2276 + 2048) >> 12
    //   t6a = (in5*2276 + in3*3406 + 2048) >> 12
    //   t2  = (in2*1567 - in6*3784 + 2048) >> 12
    //   t3  = (in2*3784 + in6*1567 + 2048) >> 12
    //   t0  = (in0*181  + in4*181  + 128 ) >> 8
    //   t1  = (in0*181  - in4*181  + 128 ) >> 8
    let t4a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_17, dct8_row_coef_pack(_token, 799, -4017)),
        pd_2048,
    ));
    let t7a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_17, dct8_row_coef_pack(_token, 4017, 799)),
        pd_2048,
    ));
    let t5a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_53, dct8_row_coef_pack(_token, 3406, -2276)),
        pd_2048,
    ));
    let t6a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_53, dct8_row_coef_pack(_token, 2276, 3406)),
        pd_2048,
    ));
    let t2 = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_26, dct8_row_coef_pack(_token, 1567, -3784)),
        pd_2048,
    ));
    let t3 = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_26, dct8_row_coef_pack(_token, 3784, 1567)),
        pd_2048,
    ));
    let t0 = _mm256_srai_epi32::<8>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_04, dct8_row_coef_pack(_token, 181, 181)),
        pd_128,
    ));
    let t1 = _mm256_srai_epi32::<8>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_04, dct8_row_coef_pack(_token, 181, -181)),
        pd_128,
    ));

    // Step 4: Additive butterfly stages (i32 arithmetic with i16 clip).
    let col_min = i16::MIN as i32;
    let col_max = i16::MAX as i32;
    let col_min_v = _mm256_set1_epi32(col_min);
    let col_max_v = _mm256_set1_epi32(col_max);
    let clip = |v: __m256i| _mm256_max_epi32(_mm256_min_epi32(v, col_max_v), col_min_v);

    // Stage 1 butterfly on odd half:
    let t4 = clip(_mm256_add_epi32(t4a, t5a));
    let t5a_n = clip(_mm256_sub_epi32(t4a, t5a));
    let t7 = clip(_mm256_add_epi32(t7a, t6a));
    let t6a_n = clip(_mm256_sub_epi32(t7a, t6a));

    // Stage 2 — t5/t6 via 181 multiply.
    // t5a_n and t6a_n are i32 in i16 range. Pack them as i16 pair for pmaddwd:
    //   pair = (t6a_n, t5a_n)
    //   t5 = pmaddwd(pair, (181, -181)) + 128 >> 8   [= (t6a_n - t5a_n)*181 + 128 >> 8]
    //   t6 = pmaddwd(pair, (181,  181)) + 128 >> 8   [= (t6a_n + t5a_n)*181 + 128 >> 8]
    let t5a_n_xmm = _mm_packs_epi32(
        _mm256_castsi256_si128(t5a_n),
        _mm256_extracti128_si256(t5a_n, 1),
    );
    let t6a_n_xmm = _mm_packs_epi32(
        _mm256_castsi256_si128(t6a_n),
        _mm256_extracti128_si256(t6a_n, 1),
    );
    let pair_65 = dct8_row_build_pair(_token, t6a_n_xmm, t5a_n_xmm);
    let t5 = _mm256_srai_epi32::<8>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_65, dct8_row_coef_pack(_token, 181, -181)),
        pd_128,
    ));
    let t6 = _mm256_srai_epi32::<8>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_65, dct8_row_coef_pack(_token, 181, 181)),
        pd_128,
    ));

    // DCT-4 even side: t0..t3 already computed above.
    // Combine: tmp0..tmp3 = DCT-4 butterfly outputs.
    let tmp0 = clip(_mm256_add_epi32(t0, t3));
    let tmp1 = clip(_mm256_add_epi32(t1, t2));
    let tmp2 = clip(_mm256_sub_epi32(t1, t2));
    let tmp3 = clip(_mm256_sub_epi32(t0, t3));

    // Final DCT-8 butterfly:
    let mut out = [_mm256_setzero_si256(); 8];
    out[0] = clip(_mm256_add_epi32(tmp0, t7));
    out[1] = clip(_mm256_add_epi32(tmp1, t6));
    out[2] = clip(_mm256_add_epi32(tmp2, t5));
    out[3] = clip(_mm256_add_epi32(tmp3, t4));
    out[4] = clip(_mm256_sub_epi32(tmp3, t4));
    out[5] = clip(_mm256_sub_epi32(tmp2, t5));
    out[6] = clip(_mm256_sub_epi32(tmp1, t6));
    out[7] = clip(_mm256_sub_epi32(tmp0, t7));

    out
}

/// i16-packed pmaddwd DCT-16 column pass.
///
/// Takes row-major i32 `tmp[y*16 + x]` (16 rows x 16 cols, values in i16 range after
/// intermediate shift+clip), runs the DCT-16 column transform using `_mm256_madd_epi16`
/// (pmaddwd) instead of `_mm256_mullo_epi32`, and returns row-major i32 output ready
/// for add-to-dst.
///
/// Processes 8 columns at a time (2 chunks). For each chunk, loads 16 rows of 8 i32,
/// packs to 16 xmm of i16, runs DCT-16 = DCT-8 on even rows + odd-half butterflies,
/// outputs 16 ymm of i32.
///
/// Algorithm: DCT-16 column decomposition:
///   1. DCT-8 on even-indexed rows (0,2,4,6,8,10,12,14) — same pmaddwd as dct8_col_pass_i16
///   2. Odd-half: 4 pmaddwd pairs (stage 1) → butterfly → 2 pmaddwd pairs (stage 2)
///      → butterfly → 2 pmaddwd pairs (stage 3, 181 cross-multiply) → combine
///   3. out[k] = even[k] + odd[k], out[15-k] = even[k] - odd[k]
///
/// Benefits vs i32 mullo column pass: replaces 24 `mullo_epi32` (10c Zen3) with
/// 20 `madd_epi16` (5c Zen3) per 8-col chunk.
#[cfg(target_arch = "x86_64")]
#[rite]
fn dct16_col_pass_i16(_token: Desktop64, tmp_row_major: &[i32; 256]) -> [i32; 256] {
    let mut result = [0i32; 256];
    let col_min = i16::MIN as i32;
    let col_max = i16::MAX as i32;
    let col_min_v = _mm256_set1_epi32(col_min);
    let col_max_v = _mm256_set1_epi32(col_max);
    let clip = |v: __m256i| _mm256_max_epi32(_mm256_min_epi32(v, col_max_v), col_min_v);

    let pd_2048 = _mm256_set1_epi32(2048);
    let pd_128 = _mm256_set1_epi32(128);

    // Process 8 columns at a time (2 chunks for 16 cols total).
    for cx_chunk in 0..2u32 {
        let cx = (cx_chunk * 8) as usize;

        // Step 1: Convert each of the 16 rows from 8 x i32 to 8 x i16 in an xmm.
        // Values are guaranteed clipped to i16 range by the intermediate shift+clip.
        let mut row_xmm = [_mm_setzero_si128(); 16];
        for y in 0..16 {
            let v = loadu_256!(&tmp_row_major[y * 16 + cx..y * 16 + cx + 8], [i32; 8]);
            let lo128 = _mm256_castsi256_si128(v);
            let hi128 = _mm256_extracti128_si256(v, 1);
            row_xmm[y] = _mm_packs_epi32(lo128, hi128);
        }

        // ====== EVEN HALF: DCT-8 on even-indexed rows (0,2,4,6,8,10,12,14) ======
        // This is structurally identical to dct8_col_pass_i16, operating on rows
        // 0,2,4,6,8,10,12,14 instead of 0..7.

        // --- DCT-4 (innermost even) on rows 0,4,8,12 ---
        let pair_0_8 = dct8_row_build_pair(_token, row_xmm[0], row_xmm[8]);
        let pair_4_12 = dct8_row_build_pair(_token, row_xmm[4], row_xmm[12]);

        // t0 = (row0 + row8) * 181 + 128 >> 8
        let e_t0 = _mm256_srai_epi32::<8>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_0_8, dct8_row_coef_pack(_token, 181, 181)),
            pd_128,
        ));
        // t1 = (row0 - row8) * 181 + 128 >> 8
        let e_t1 = _mm256_srai_epi32::<8>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_0_8, dct8_row_coef_pack(_token, 181, -181)),
            pd_128,
        ));
        // t2 = (row4 * 1567 - row12 * 3784 + 2048) >> 12
        let e_t2 = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_4_12, dct8_row_coef_pack(_token, 1567, -3784)),
            pd_2048,
        ));
        // t3 = (row4 * 3784 + row12 * 1567 + 2048) >> 12
        let e_t3 = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_4_12, dct8_row_coef_pack(_token, 3784, 1567)),
            pd_2048,
        ));

        // DCT-4 output
        let dct4_0 = clip(_mm256_add_epi32(e_t0, e_t3));
        let dct4_1 = clip(_mm256_add_epi32(e_t1, e_t2));
        let dct4_2 = clip(_mm256_sub_epi32(e_t1, e_t2));
        let dct4_3 = clip(_mm256_sub_epi32(e_t0, e_t3));

        // --- DCT-8 odd half on rows 2, 6, 10, 14 ---
        let pair_2_14 = dct8_row_build_pair(_token, row_xmm[2], row_xmm[14]);
        let pair_10_6 = dct8_row_build_pair(_token, row_xmm[10], row_xmm[6]);

        // t4a = (row2 * 799 - row14 * 4017 + 2048) >> 12
        let t4a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_2_14, dct8_row_coef_pack(_token, 799, -4017)),
            pd_2048,
        ));
        // t7a = (row2 * 4017 + row14 * 799 + 2048) >> 12
        let t7a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_2_14, dct8_row_coef_pack(_token, 4017, 799)),
            pd_2048,
        ));
        // t5a = (row10 * 3406 - row6 * 2276 + 2048) >> 12
        let t5a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_10_6, dct8_row_coef_pack(_token, 3406, -2276)),
            pd_2048,
        ));
        // t6a = (row10 * 2276 + row6 * 3406 + 2048) >> 12
        let t6a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_10_6, dct8_row_coef_pack(_token, 2276, 3406)),
            pd_2048,
        ));

        // DCT-8 odd stage 1 butterfly
        let e_t4 = clip(_mm256_add_epi32(t4a, t5a));
        let e_t5a_n = clip(_mm256_sub_epi32(t4a, t5a));
        let e_t7 = clip(_mm256_add_epi32(t7a, t6a));
        let e_t6a_n = clip(_mm256_sub_epi32(t7a, t6a));

        // DCT-8 odd stage 2 — sqrt(2) cross-multiply via pmaddwd
        let e_t5a_n_xmm = _mm_packs_epi32(
            _mm256_castsi256_si128(e_t5a_n),
            _mm256_extracti128_si256(e_t5a_n, 1),
        );
        let e_t6a_n_xmm = _mm_packs_epi32(
            _mm256_castsi256_si128(e_t6a_n),
            _mm256_extracti128_si256(e_t6a_n, 1),
        );
        let pair_65 = dct8_row_build_pair(_token, e_t6a_n_xmm, e_t5a_n_xmm);
        // t5 = (t6a_n * 181 - t5a_n * 181 + 128) >> 8 = (t6a_n - t5a_n)*181 + 128 >> 8
        let e_t5 = _mm256_srai_epi32::<8>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_65, dct8_row_coef_pack(_token, 181, -181)),
            pd_128,
        ));
        // t6 = (t6a_n * 181 + t5a_n * 181 + 128) >> 8 = (t6a_n + t5a_n)*181 + 128 >> 8
        let e_t6 = _mm256_srai_epi32::<8>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_65, dct8_row_coef_pack(_token, 181, 181)),
            pd_128,
        ));

        // DCT-8 final butterfly — produces even[0..7]
        let even_0 = clip(_mm256_add_epi32(dct4_0, e_t7));
        let even_1 = clip(_mm256_add_epi32(dct4_1, e_t6));
        let even_2 = clip(_mm256_add_epi32(dct4_2, e_t5));
        let even_3 = clip(_mm256_add_epi32(dct4_3, e_t4));
        let even_4 = clip(_mm256_sub_epi32(dct4_3, e_t4));
        let even_5 = clip(_mm256_sub_epi32(dct4_2, e_t5));
        let even_6 = clip(_mm256_sub_epi32(dct4_1, e_t6));
        let even_7 = clip(_mm256_sub_epi32(dct4_0, e_t7));

        // ====== ODD HALF: butterflies on odd-indexed rows (1,3,5,7,9,11,13,15) ======

        // Stage 1: 4 pmaddwd pairs with trig constants
        let pair_1_15 = dct8_row_build_pair(_token, row_xmm[1], row_xmm[15]);
        let pair_9_7 = dct8_row_build_pair(_token, row_xmm[9], row_xmm[7]);
        let pair_5_11 = dct8_row_build_pair(_token, row_xmm[5], row_xmm[11]);
        let pair_13_3 = dct8_row_build_pair(_token, row_xmm[13], row_xmm[3]);

        // t8a  = (row1 * 401 - row15 * 4076 + 2048) >> 12
        let o_t8a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_1_15, dct8_row_coef_pack(_token, 401, -4076)),
            pd_2048,
        ));
        // t15a = (row1 * 4076 + row15 * 401 + 2048) >> 12
        let o_t15a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_1_15, dct8_row_coef_pack(_token, 4076, 401)),
            pd_2048,
        ));
        // t9a  = (row9 * 3166 - row7 * 2598 + 2048) >> 12
        let o_t9a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_9_7, dct8_row_coef_pack(_token, 3166, -2598)),
            pd_2048,
        ));
        // t14a = (row9 * 2598 + row7 * 3166 + 2048) >> 12
        let o_t14a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_9_7, dct8_row_coef_pack(_token, 2598, 3166)),
            pd_2048,
        ));
        // t10a = (row5 * 1931 - row11 * 3612 + 2048) >> 12
        let o_t10a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_5_11, dct8_row_coef_pack(_token, 1931, -3612)),
            pd_2048,
        ));
        // t13a = (row5 * 3612 + row11 * 1931 + 2048) >> 12
        let o_t13a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_5_11, dct8_row_coef_pack(_token, 3612, 1931)),
            pd_2048,
        ));
        // t11a = (row13 * 3920 - row3 * 1189 + 2048) >> 12
        let o_t11a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_13_3, dct8_row_coef_pack(_token, 3920, -1189)),
            pd_2048,
        ));
        // t12a = (row13 * 1189 + row3 * 3920 + 2048) >> 12
        let o_t12a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_13_3, dct8_row_coef_pack(_token, 1189, 3920)),
            pd_2048,
        ));

        // Additive butterfly 1
        let o_t8 = clip(_mm256_add_epi32(o_t8a, o_t9a));
        let o_t9 = clip(_mm256_sub_epi32(o_t8a, o_t9a));
        let o_t10 = clip(_mm256_sub_epi32(o_t11a, o_t10a));
        let o_t11 = clip(_mm256_add_epi32(o_t11a, o_t10a));
        let o_t12 = clip(_mm256_add_epi32(o_t12a, o_t13a));
        let o_t13 = clip(_mm256_sub_epi32(o_t12a, o_t13a));
        let o_t14 = clip(_mm256_sub_epi32(o_t15a, o_t14a));
        let o_t15 = clip(_mm256_add_epi32(o_t15a, o_t14a));

        // Stage 2: multiplicative butterflies via pmaddwd (values in i16 range from clip)
        let o_t14_xmm = _mm_packs_epi32(
            _mm256_castsi256_si128(o_t14),
            _mm256_extracti128_si256(o_t14, 1),
        );
        let o_t9_xmm = _mm_packs_epi32(
            _mm256_castsi256_si128(o_t9),
            _mm256_extracti128_si256(o_t9, 1),
        );
        let pair_14_9 = dct8_row_build_pair(_token, o_t14_xmm, o_t9_xmm);

        // t9a  = (t14 * 1567 - t9 * 3784 + 2048) >> 12
        let o_t9a_new = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_14_9, dct8_row_coef_pack(_token, 1567, -3784)),
            pd_2048,
        ));
        // t14a = (t14 * 3784 + t9 * 1567 + 2048) >> 12
        let o_t14a_new = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_14_9, dct8_row_coef_pack(_token, 3784, 1567)),
            pd_2048,
        ));

        let o_t13_xmm = _mm_packs_epi32(
            _mm256_castsi256_si128(o_t13),
            _mm256_extracti128_si256(o_t13, 1),
        );
        let o_t10_xmm = _mm_packs_epi32(
            _mm256_castsi256_si128(o_t10),
            _mm256_extracti128_si256(o_t10, 1),
        );
        let pair_13_10 = dct8_row_build_pair(_token, o_t13_xmm, o_t10_xmm);

        // t10a = (-t13 * 3784 - t10 * 1567 + 2048) >> 12
        let o_t10a_new = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_13_10, dct8_row_coef_pack(_token, -3784, -1567)),
            pd_2048,
        ));
        // t13a = (t13 * 1567 - t10 * 3784 + 2048) >> 12
        let o_t13a_new = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_13_10, dct8_row_coef_pack(_token, 1567, -3784)),
            pd_2048,
        ));

        // Additive butterfly 2
        let o_t8a_f = clip(_mm256_add_epi32(o_t8, o_t11));
        let o_t9_f = clip(_mm256_add_epi32(o_t9a_new, o_t10a_new));
        let o_t10_f = clip(_mm256_sub_epi32(o_t9a_new, o_t10a_new));
        let o_t11a_f = clip(_mm256_sub_epi32(o_t8, o_t11));
        let o_t12a_f = clip(_mm256_sub_epi32(o_t15, o_t12));
        let o_t13_f = clip(_mm256_sub_epi32(o_t14a_new, o_t13a_new));
        let o_t14_f = clip(_mm256_add_epi32(o_t14a_new, o_t13a_new));
        let o_t15a_f = clip(_mm256_add_epi32(o_t15, o_t12));

        // Stage 3: sqrt(2) cross-multiply via pmaddwd (values in i16 range from clip)
        let o_t13_f_xmm = _mm_packs_epi32(
            _mm256_castsi256_si128(o_t13_f),
            _mm256_extracti128_si256(o_t13_f, 1),
        );
        let o_t10_f_xmm = _mm_packs_epi32(
            _mm256_castsi256_si128(o_t10_f),
            _mm256_extracti128_si256(o_t10_f, 1),
        );
        let pair_13f_10f = dct8_row_build_pair(_token, o_t13_f_xmm, o_t10_f_xmm);

        // t10a = (t13 * 181 - t10 * 181 + 128) >> 8 = (t13 - t10) * 181 + 128 >> 8
        let o_t10a_f = _mm256_srai_epi32::<8>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_13f_10f, dct8_row_coef_pack(_token, 181, -181)),
            pd_128,
        ));
        // t13a = (t13 * 181 + t10 * 181 + 128) >> 8 = (t13 + t10) * 181 + 128 >> 8
        let o_t13a_f = _mm256_srai_epi32::<8>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_13f_10f, dct8_row_coef_pack(_token, 181, 181)),
            pd_128,
        ));

        let o_t12a_f_xmm = _mm_packs_epi32(
            _mm256_castsi256_si128(o_t12a_f),
            _mm256_extracti128_si256(o_t12a_f, 1),
        );
        let o_t11a_f_xmm = _mm_packs_epi32(
            _mm256_castsi256_si128(o_t11a_f),
            _mm256_extracti128_si256(o_t11a_f, 1),
        );
        let pair_12a_11a = dct8_row_build_pair(_token, o_t12a_f_xmm, o_t11a_f_xmm);

        // t11 = (t12a * 181 - t11a * 181 + 128) >> 8 = (t12a - t11a) * 181 + 128 >> 8
        let o_t11_f = _mm256_srai_epi32::<8>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_12a_11a, dct8_row_coef_pack(_token, 181, -181)),
            pd_128,
        ));
        // t12 = (t12a * 181 + t11a * 181 + 128) >> 8 = (t12a + t11a) * 181 + 128 >> 8
        let o_t12_f = _mm256_srai_epi32::<8>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_12a_11a, dct8_row_coef_pack(_token, 181, 181)),
            pd_128,
        ));

        // ====== FINAL COMBINE: out[k] = clip(even[k] + odd[k]), out[15-k] = clip(even[k] - odd[k]) ======
        // Mapping (from scalar inv_dct16_1d_internal_c):
        //   out[0]  = clip(even[0] + t15a_f)   out[15] = clip(even[0] - t15a_f)
        //   out[1]  = clip(even[1] + t14_f)    out[14] = clip(even[1] - t14_f)
        //   out[2]  = clip(even[2] + t13a_f)   out[13] = clip(even[2] - t13a_f)
        //   out[3]  = clip(even[3] + t12_f)    out[12] = clip(even[3] - t12_f)
        //   out[4]  = clip(even[4] + t11_f)    out[11] = clip(even[4] - t11_f)
        //   out[5]  = clip(even[5] + t10a_f)   out[10] = clip(even[5] - t10a_f)
        //   out[6]  = clip(even[6] + t9_f)     out[9]  = clip(even[6] - t9_f)
        //   out[7]  = clip(even[7] + t8a_f)    out[8]  = clip(even[7] - t8a_f)
        let odd = [
            o_t15a_f, o_t14_f, o_t13a_f, o_t12_f, o_t11_f, o_t10a_f, o_t9_f, o_t8a_f,
        ];
        let even = [
            even_0, even_1, even_2, even_3, even_4, even_5, even_6, even_7,
        ];

        let mut cols = [_mm256_setzero_si256(); 16];
        for k in 0..8 {
            cols[k] = clip(_mm256_add_epi32(even[k], odd[k]));
            cols[15 - k] = clip(_mm256_sub_epi32(even[k], odd[k]));
        }

        // Store results back to row-major output.
        for y in 0..16 {
            storeu_256!(&mut result[y * 16 + cx..y * 16 + cx + 8], [i32; 8], cols[y]);
        }
    }

    result
}

/// i16-packed pmaddwd DCT-16 row pass — operates on 16 rows of 16 i16 coefficients
/// stored in column-major order, producing 256 i32 outputs in row-major order.
///
/// Layout: `coeff_col_major[y + x*16]` = element x of row y (y=0..15, x=0..15).
/// Output: `out[y*16 + x]` = transformed element x of row y.
///
/// Processes two batches of 8 rows (one per ymm lane) using pmaddwd for the
/// initial multiplicative stages and i32 arithmetic for intermediate butterflies.
///
/// Algorithm: DCT-16 = DCT-8 on even-indexed inputs + odd-half butterflies + combine.
/// The DCT-8 even half reuses the pmaddwd pair approach from `dct8_row_pass_i16_simd`.
/// The odd half uses 4 pmaddwd pairs for stage 1, then i32 mullo for stage 2.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn dct16_row_pass_i16_simd(_token: Desktop64, coeff_col_major: [i16; 256]) -> [i32; 256] {
    let mut out = [0i32; 256];

    let row_min = i16::MIN as i32;
    let row_max = i16::MAX as i32;
    let row_min_v = _mm256_set1_epi32(row_min);
    let row_max_v = _mm256_set1_epi32(row_max);
    let clip = |v: __m256i| _mm256_max_epi32(_mm256_min_epi32(v, row_max_v), row_min_v);

    let pd_2048 = _mm256_set1_epi32(2048);
    let pd_128 = _mm256_set1_epi32(128);
    let c_181 = _mm256_set1_epi32(181);

    // Process rows in two batches of 8
    for batch in 0..2u32 {
        let y_base = (batch * 8) as usize;

        // Load 16 column xmms (8 i16 from this batch per column).
        let mut col_xmm = [_mm_setzero_si128(); 16];
        for x in 0..16 {
            let off = y_base + x * 16;
            let arr: &[i16; 8] = (&coeff_col_major[off..off + 8]).try_into().unwrap();
            col_xmm[x] = loadu_128!(arr);
        }

        // ====== EVEN HALF: DCT-8 on even columns (0,2,4,6,8,10,12,14) ======
        // These become the inputs to a standard DCT-8.
        // Rename for clarity: ecol[k] = col_xmm[2*k]
        //   ecol[0]=col0, ecol[1]=col2, ecol[2]=col4, ecol[3]=col6,
        //   ecol[4]=col8, ecol[5]=col10, ecol[6]=col12, ecol[7]=col14
        //
        // DCT-8 structure (non-tx64):
        //   DCT-4 on even of even: ecol[0], ecol[2], ecol[4], ecol[6]
        //     which are original columns 0, 4, 8, 12
        //   DCT-8 odd half on: ecol[1], ecol[3], ecol[5], ecol[7]
        //     which are original columns 2, 6, 10, 14

        // --- DCT-4 (innermost even) on columns 0, 4, 8, 12 ---
        let pair_0_8 = dct8_row_build_pair(_token, col_xmm[0], col_xmm[8]);
        let pair_4_12 = dct8_row_build_pair(_token, col_xmm[4], col_xmm[12]);

        // t0 = (col0 + col8) * 181 + 128 >> 8
        let e_t0 = _mm256_srai_epi32::<8>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_0_8, dct8_row_coef_pack(_token, 181, 181)),
            pd_128,
        ));
        // t1 = (col0 - col8) * 181 + 128 >> 8
        let e_t1 = _mm256_srai_epi32::<8>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_0_8, dct8_row_coef_pack(_token, 181, -181)),
            pd_128,
        ));
        // t2 = (col4 * 1567 - col12 * 3784 + 2048) >> 12
        let e_t2 = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_4_12, dct8_row_coef_pack(_token, 1567, -3784)),
            pd_2048,
        ));
        // t3 = (col4 * 3784 + col12 * 1567 + 2048) >> 12
        let e_t3 = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_4_12, dct8_row_coef_pack(_token, 3784, 1567)),
            pd_2048,
        ));

        // DCT-4 output
        let dct4_0 = clip(_mm256_add_epi32(e_t0, e_t3));
        let dct4_1 = clip(_mm256_add_epi32(e_t1, e_t2));
        let dct4_2 = clip(_mm256_sub_epi32(e_t1, e_t2));
        let dct4_3 = clip(_mm256_sub_epi32(e_t0, e_t3));

        // --- DCT-8 odd half on columns 2, 6, 10, 14 ---
        let pair_2_14 = dct8_row_build_pair(_token, col_xmm[2], col_xmm[14]);
        let pair_10_6 = dct8_row_build_pair(_token, col_xmm[10], col_xmm[6]);

        // t4a = (col2 * 799 - col14 * 4017 + 2048) >> 12
        let t4a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_2_14, dct8_row_coef_pack(_token, 799, -4017)),
            pd_2048,
        ));
        // t7a = (col2 * 4017 + col14 * 799 + 2048) >> 12
        let t7a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_2_14, dct8_row_coef_pack(_token, 4017, 799)),
            pd_2048,
        ));
        // t5a = (col10 * 3406 - col6 * 2276 + 2048) >> 12
        let t5a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_10_6, dct8_row_coef_pack(_token, 3406, -2276)),
            pd_2048,
        ));
        // t6a = (col10 * 2276 + col6 * 3406 + 2048) >> 12
        let t6a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_10_6, dct8_row_coef_pack(_token, 2276, 3406)),
            pd_2048,
        ));

        // DCT-8 odd stage 1 butterfly
        let t4 = clip(_mm256_add_epi32(t4a, t5a));
        let t5a_n = clip(_mm256_sub_epi32(t4a, t5a));
        let t7 = clip(_mm256_add_epi32(t7a, t6a));
        let t6a_n = clip(_mm256_sub_epi32(t7a, t6a));

        // DCT-8 odd stage 2 — sqrt(2) cross-multiply
        let d_65 = _mm256_sub_epi32(t6a_n, t5a_n);
        let t5 = _mm256_srai_epi32::<8>(_mm256_add_epi32(_mm256_mullo_epi32(d_65, c_181), pd_128));
        let s_65 = _mm256_add_epi32(t6a_n, t5a_n);
        let t6 = _mm256_srai_epi32::<8>(_mm256_add_epi32(_mm256_mullo_epi32(s_65, c_181), pd_128));

        // DCT-8 final butterfly
        let even = [
            clip(_mm256_add_epi32(dct4_0, t7)),
            clip(_mm256_add_epi32(dct4_1, t6)),
            clip(_mm256_add_epi32(dct4_2, t5)),
            clip(_mm256_add_epi32(dct4_3, t4)),
            clip(_mm256_sub_epi32(dct4_3, t4)),
            clip(_mm256_sub_epi32(dct4_2, t5)),
            clip(_mm256_sub_epi32(dct4_1, t6)),
            clip(_mm256_sub_epi32(dct4_0, t7)),
        ];

        // ====== ODD HALF: 8 butterflies on odd columns (1,3,5,7,9,11,13,15) ======

        // Stage 1: multiplicative butterflies via pmaddwd (full coefficients)
        let pair_1_15 = dct8_row_build_pair(_token, col_xmm[1], col_xmm[15]);
        let pair_9_7 = dct8_row_build_pair(_token, col_xmm[9], col_xmm[7]);
        let pair_5_11 = dct8_row_build_pair(_token, col_xmm[5], col_xmm[11]);
        let pair_13_3 = dct8_row_build_pair(_token, col_xmm[13], col_xmm[3]);

        // t8a  = (in1 * 401 - in15 * 4076 + 2048) >> 12
        let o_t8a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_1_15, dct8_row_coef_pack(_token, 401, -4076)),
            pd_2048,
        ));
        // t15a = (in1 * 4076 + in15 * 401 + 2048) >> 12
        let o_t15a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_1_15, dct8_row_coef_pack(_token, 4076, 401)),
            pd_2048,
        ));
        // t9a  = (in9 * 3166 - in7 * 2598 + 2048) >> 12  (doubled from >>11 form)
        let o_t9a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_9_7, dct8_row_coef_pack(_token, 3166, -2598)),
            pd_2048,
        ));
        // t14a = (in9 * 2598 + in7 * 3166 + 2048) >> 12  (doubled from >>11 form)
        let o_t14a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_9_7, dct8_row_coef_pack(_token, 2598, 3166)),
            pd_2048,
        ));
        // t10a = (in5 * 1931 - in11 * 3612 + 2048) >> 12
        let o_t10a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_5_11, dct8_row_coef_pack(_token, 1931, -3612)),
            pd_2048,
        ));
        // t13a = (in5 * 3612 + in11 * 1931 + 2048) >> 12
        let o_t13a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_5_11, dct8_row_coef_pack(_token, 3612, 1931)),
            pd_2048,
        ));
        // t11a = (in13 * 3920 - in3 * 1189 + 2048) >> 12
        let o_t11a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_13_3, dct8_row_coef_pack(_token, 3920, -1189)),
            pd_2048,
        ));
        // t12a = (in13 * 1189 + in3 * 3920 + 2048) >> 12
        let o_t12a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_13_3, dct8_row_coef_pack(_token, 1189, 3920)),
            pd_2048,
        ));

        // Additive butterfly 1
        let o_t8 = clip(_mm256_add_epi32(o_t8a, o_t9a));
        let mut o_t9 = clip(_mm256_sub_epi32(o_t8a, o_t9a));
        let mut o_t10 = clip(_mm256_sub_epi32(o_t11a, o_t10a));
        let o_t11 = clip(_mm256_add_epi32(o_t11a, o_t10a));
        let o_t12 = clip(_mm256_add_epi32(o_t12a, o_t13a));
        let mut o_t13 = clip(_mm256_sub_epi32(o_t12a, o_t13a));
        let mut o_t14 = clip(_mm256_sub_epi32(o_t15a, o_t14a));
        let o_t15 = clip(_mm256_add_epi32(o_t15a, o_t14a));

        // Stage 2: multiplicative butterflies (i32 mullo, full-coefficient forms)
        let c_1567 = _mm256_set1_epi32(1567);
        let c_3784 = _mm256_set1_epi32(3784);

        // t9a  = (t14 * 1567 - t9 * 3784 + 2048) >> 12
        let o_t9a_new = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_sub_epi32(
                _mm256_mullo_epi32(o_t14, c_1567),
                _mm256_mullo_epi32(o_t9, c_3784),
            ),
            pd_2048,
        ));
        // t14a = (t14 * 3784 + t9 * 1567 + 2048) >> 12
        let o_t14a_new = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_add_epi32(
                _mm256_mullo_epi32(o_t14, c_3784),
                _mm256_mullo_epi32(o_t9, c_1567),
            ),
            pd_2048,
        ));
        // t10a = (-t13 * 3784 - t10 * 1567 + 2048) >> 12
        let o_t10a_new = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_sub_epi32(
                _mm256_setzero_si256(),
                _mm256_add_epi32(
                    _mm256_mullo_epi32(o_t13, c_3784),
                    _mm256_mullo_epi32(o_t10, c_1567),
                ),
            ),
            pd_2048,
        ));
        // t13a = (t13 * 1567 - t10 * 3784 + 2048) >> 12
        let o_t13a_new = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_sub_epi32(
                _mm256_mullo_epi32(o_t13, c_1567),
                _mm256_mullo_epi32(o_t10, c_3784),
            ),
            pd_2048,
        ));

        // Additive butterfly 2
        let o_t8a_f = clip(_mm256_add_epi32(o_t8, o_t11));
        o_t9 = clip(_mm256_add_epi32(o_t9a_new, o_t10a_new));
        o_t10 = clip(_mm256_sub_epi32(o_t9a_new, o_t10a_new));
        let o_t11a_f = clip(_mm256_sub_epi32(o_t8, o_t11));
        let o_t12a_f = clip(_mm256_sub_epi32(o_t15, o_t12));
        o_t13 = clip(_mm256_sub_epi32(o_t14a_new, o_t13a_new));
        o_t14 = clip(_mm256_add_epi32(o_t14a_new, o_t13a_new));
        let o_t15a_f = clip(_mm256_add_epi32(o_t15, o_t12));

        // Stage 3: sqrt(2) cross-multiply (181/256)
        // t10a = ((t13 - t10) * 181 + 128) >> 8
        let d2 = _mm256_sub_epi32(o_t13, o_t10);
        let o_t10a_f =
            _mm256_srai_epi32::<8>(_mm256_add_epi32(_mm256_mullo_epi32(d2, c_181), pd_128));
        // t13a = ((t13 + t10) * 181 + 128) >> 8
        let s2 = _mm256_add_epi32(o_t13, o_t10);
        let o_t13a_f =
            _mm256_srai_epi32::<8>(_mm256_add_epi32(_mm256_mullo_epi32(s2, c_181), pd_128));
        // t11 = ((t12a - t11a) * 181 + 128) >> 8
        let d3 = _mm256_sub_epi32(o_t12a_f, o_t11a_f);
        let o_t11_f =
            _mm256_srai_epi32::<8>(_mm256_add_epi32(_mm256_mullo_epi32(d3, c_181), pd_128));
        // t12 = ((t12a + t11a) * 181 + 128) >> 8
        let s3 = _mm256_add_epi32(o_t12a_f, o_t11a_f);
        let o_t12_f =
            _mm256_srai_epi32::<8>(_mm256_add_epi32(_mm256_mullo_epi32(s3, c_181), pd_128));

        // ====== FINAL COMBINE: out[k] = clip(even[k] + odd[15-k reversed mapping]) ======
        // Mapping from scalar:
        //   out[0]  = clip(even[0] + t15a_f)   out[15] = clip(even[0] - t15a_f)
        //   out[1]  = clip(even[1] + t14)       out[14] = clip(even[1] - t14)
        //   out[2]  = clip(even[2] + t13a_f)   out[13] = clip(even[2] - t13a_f)
        //   out[3]  = clip(even[3] + t12_f)    out[12] = clip(even[3] - t12_f)
        //   out[4]  = clip(even[4] + t11_f)    out[11] = clip(even[4] - t11_f)
        //   out[5]  = clip(even[5] + t10a_f)   out[10] = clip(even[5] - t10a_f)
        //   out[6]  = clip(even[6] + t9)       out[9]  = clip(even[6] - t9)
        //   out[7]  = clip(even[7] + t8a_f)    out[8]  = clip(even[7] - t8a_f)
        let odd = [
            o_t15a_f, o_t14, o_t13a_f, o_t12_f, o_t11_f, o_t10a_f, o_t9, o_t8a_f,
        ];

        let mut cols = [_mm256_setzero_si256(); 16];
        for k in 0..8 {
            cols[k] = clip(_mm256_add_epi32(even[k], odd[k]));
            cols[15 - k] = clip(_mm256_sub_epi32(even[k], odd[k]));
        }

        // Transpose 16x8 → 8x16 in 2 chunks of 8 columns, store row-major (stride 16).
        for chunk in 0..2u32 {
            let b = (chunk * 8) as usize;
            let chunk_cols: [__m256i; 8] = [
                cols[b],
                cols[b + 1],
                cols[b + 2],
                cols[b + 3],
                cols[b + 4],
                cols[b + 5],
                cols[b + 6],
                cols[b + 7],
            ];
            let rows = transpose_8x8_i32!(chunk_cols);
            for r in 0..8 {
                let dst_off = (y_base + r) * 16 + b;
                let arr: &mut [i32; 8] = (&mut out[dst_off..dst_off + 8]).try_into().unwrap();
                storeu_256!(arr, [i32; 8], rows[r]);
            }
        }
    }
    out
}

/// i16-packed pmaddwd DCT-32 1D row pass. Takes 1024 i16 column-major
/// coeffs (input shape: `coeff[y + x*32]` is element `x` of row `y`),
/// runs DCT-32 independently across each of the 32 rows, and returns
/// row-major i32 output (output shape: `out[y*32 + x]` is element `x`
/// of row `y`).
///
/// Processes 8 rows at a time (4 batches), each batch using ymm registers
/// with one i32 lane per row. The initial multiplicative stage of each
/// sub-DCT (DCT-4, DCT-8 odd, DCT-16 odd, DCT-32 odd) uses pmaddwd on
/// interleaved i16 column pairs — a single pmaddwd replaces a
/// cvtepi16_epi32 + two mullo_epi32 + add_epi32 sequence.
///
/// Bit-exactness target: matches `run_scalar_dct32_per_row` exactly for
/// `row_min = i16::MIN as i32`, `row_max = i16::MAX as i32`.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn dct32_row_pass_i16_simd(_token: Desktop64, coeff_col_major: [i16; 1024]) -> [i32; 1024] {
    let mut out = [0i32; 1024];
    let build_pair = dct8_row_build_pair;
    let coef_pack = dct8_row_coef_pack;

    for batch in 0..4u32 {
        let y_base = (batch * 8) as usize;

        // Load 32 columns x 8 rows as xmm (i16).
        let mut cx = [_mm_setzero_si128(); 32];
        for x in 0..32 {
            let off = y_base + x * 32;
            let arr: &[i16; 8] = (&coeff_col_major[off..off + 8]).try_into().unwrap();
            cx[x] = loadu_128!(arr);
        }

        let pd_2048 = _mm256_set1_epi32(2048);
        let pd_128 = _mm256_set1_epi32(128);
        let row_min_v = _mm256_set1_epi32(i16::MIN as i32);
        let row_max_v = _mm256_set1_epi32(i16::MAX as i32);
        let clip = |v: __m256i| _mm256_max_epi32(_mm256_min_epi32(v, row_max_v), row_min_v);
        let c_181 = _mm256_set1_epi32(181);

        // ===== DCT-4 on columns [0, 8, 16, 24] =====
        let pair_04 = build_pair(_token, cx[0], cx[16]);
        let pair_13 = build_pair(_token, cx[8], cx[24]);

        let dct4_t0 = _mm256_srai_epi32::<8>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_04, coef_pack(_token, 181, 181)),
            pd_128,
        ));
        let dct4_t1 = _mm256_srai_epi32::<8>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_04, coef_pack(_token, 181, -181)),
            pd_128,
        ));
        let dct4_t2 = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_13, coef_pack(_token, 1567, -3784)),
            pd_2048,
        ));
        let dct4_t3 = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_13, coef_pack(_token, 3784, 1567)),
            pd_2048,
        ));

        let dct4_o0 = clip(_mm256_add_epi32(dct4_t0, dct4_t3));
        let dct4_o1 = clip(_mm256_add_epi32(dct4_t1, dct4_t2));
        let dct4_o2 = clip(_mm256_sub_epi32(dct4_t1, dct4_t2));
        let dct4_o3 = clip(_mm256_sub_epi32(dct4_t0, dct4_t3));

        // ===== DCT-8 odd half on columns [4, 12, 20, 28] =====
        let pair_8_17 = build_pair(_token, cx[4], cx[28]);
        let pair_8_53 = build_pair(_token, cx[20], cx[12]);

        let t4a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_8_17, coef_pack(_token, 799, -4017)),
            pd_2048,
        ));
        let t7a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_8_17, coef_pack(_token, 4017, 799)),
            pd_2048,
        ));
        let t5a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_8_53, coef_pack(_token, 3406, -2276)),
            pd_2048,
        ));
        let t6a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_8_53, coef_pack(_token, 2276, 3406)),
            pd_2048,
        ));

        let t4 = clip(_mm256_add_epi32(t4a, t5a));
        let t5a_n = clip(_mm256_sub_epi32(t4a, t5a));
        let t7 = clip(_mm256_add_epi32(t7a, t6a));
        let t6a_n = clip(_mm256_sub_epi32(t7a, t6a));

        let d_56 = _mm256_sub_epi32(t6a_n, t5a_n);
        let t5 = _mm256_srai_epi32::<8>(_mm256_add_epi32(_mm256_mullo_epi32(d_56, c_181), pd_128));
        let s_56 = _mm256_add_epi32(t6a_n, t5a_n);
        let t6 = _mm256_srai_epi32::<8>(_mm256_add_epi32(_mm256_mullo_epi32(s_56, c_181), pd_128));

        let dct8_o0 = clip(_mm256_add_epi32(dct4_o0, t7));
        let dct8_o1 = clip(_mm256_add_epi32(dct4_o1, t6));
        let dct8_o2 = clip(_mm256_add_epi32(dct4_o2, t5));
        let dct8_o3 = clip(_mm256_add_epi32(dct4_o3, t4));
        let dct8_o4 = clip(_mm256_sub_epi32(dct4_o3, t4));
        let dct8_o5 = clip(_mm256_sub_epi32(dct4_o2, t5));
        let dct8_o6 = clip(_mm256_sub_epi32(dct4_o1, t6));
        let dct8_o7 = clip(_mm256_sub_epi32(dct4_o0, t7));

        // ===== DCT-16 odd half on columns [2,6,10,14,18,22,26,30] =====
        let pair_16_1_15 = build_pair(_token, cx[2], cx[30]);
        let pair_16_9_7 = build_pair(_token, cx[18], cx[14]);
        let pair_16_5_11 = build_pair(_token, cx[10], cx[22]);
        let pair_16_13_3 = build_pair(_token, cx[26], cx[6]);

        let t8a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_16_1_15, coef_pack(_token, 401, -4076)),
            pd_2048,
        ));
        let t15a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_16_1_15, coef_pack(_token, 4076, 401)),
            pd_2048,
        ));
        // in9*1583 - in7*1299 doubled: in9*3166 - in7*2598
        let t9a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_16_9_7, coef_pack(_token, 3166, -2598)),
            pd_2048,
        ));
        let t14a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_16_9_7, coef_pack(_token, 2598, 3166)),
            pd_2048,
        ));
        let t10a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_16_5_11, coef_pack(_token, 1931, -3612)),
            pd_2048,
        ));
        let t13a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_16_5_11, coef_pack(_token, 3612, 1931)),
            pd_2048,
        ));
        let t11a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_16_13_3, coef_pack(_token, 3920, -1189)),
            pd_2048,
        ));
        let t12a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_16_13_3, coef_pack(_token, 1189, 3920)),
            pd_2048,
        ));

        // DCT-16 odd stage 1
        let t8 = clip(_mm256_add_epi32(t8a, t9a));
        let mut t9 = clip(_mm256_sub_epi32(t8a, t9a));
        let mut t10 = clip(_mm256_sub_epi32(t11a, t10a));
        let t11 = clip(_mm256_add_epi32(t11a, t10a));
        let t12 = clip(_mm256_add_epi32(t12a, t13a));
        let mut t13 = clip(_mm256_sub_epi32(t12a, t13a));
        let mut t14 = clip(_mm256_sub_epi32(t15a, t14a));
        let t15 = clip(_mm256_add_epi32(t15a, t14a));

        // DCT-16 odd stage 2: trig rotations (1567/3784)
        let c1567 = _mm256_set1_epi32(1567);
        let c3784 = _mm256_set1_epi32(3784);
        let t9a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_sub_epi32(
                _mm256_mullo_epi32(t14, c1567),
                _mm256_mullo_epi32(t9, c3784),
            ),
            pd_2048,
        ));
        let t14a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_add_epi32(
                _mm256_mullo_epi32(t14, c3784),
                _mm256_mullo_epi32(t9, c1567),
            ),
            pd_2048,
        ));
        let t10a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_sub_epi32(
                _mm256_setzero_si256(),
                _mm256_add_epi32(
                    _mm256_mullo_epi32(t13, c3784),
                    _mm256_mullo_epi32(t10, c1567),
                ),
            ),
            pd_2048,
        ));
        let t13a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_sub_epi32(
                _mm256_mullo_epi32(t13, c1567),
                _mm256_mullo_epi32(t10, c3784),
            ),
            pd_2048,
        ));

        // DCT-16 odd stage 3
        let t8a = clip(_mm256_add_epi32(t8, t11));
        t9 = clip(_mm256_add_epi32(t9a, t10a));
        t10 = clip(_mm256_sub_epi32(t9a, t10a));
        let t11a = clip(_mm256_sub_epi32(t8, t11));
        let t12a = clip(_mm256_sub_epi32(t15, t12));
        t13 = clip(_mm256_sub_epi32(t14a, t13a));
        t14 = clip(_mm256_add_epi32(t14a, t13a));
        let t15a = clip(_mm256_add_epi32(t15, t12));

        // DCT-16 odd final: sqrt(2) cross-muls
        let t10a = _mm256_srai_epi32::<8>(_mm256_add_epi32(
            _mm256_mullo_epi32(_mm256_sub_epi32(t13, t10), c_181),
            pd_128,
        ));
        let t13a = _mm256_srai_epi32::<8>(_mm256_add_epi32(
            _mm256_mullo_epi32(_mm256_add_epi32(t13, t10), c_181),
            pd_128,
        ));
        let t11 = _mm256_srai_epi32::<8>(_mm256_add_epi32(
            _mm256_mullo_epi32(_mm256_sub_epi32(t12a, t11a), c_181),
            pd_128,
        ));
        let t12 = _mm256_srai_epi32::<8>(_mm256_add_epi32(
            _mm256_mullo_epi32(_mm256_add_epi32(t12a, t11a), c_181),
            pd_128,
        ));

        // DCT-16 final butterfly
        let dct16_o = [
            clip(_mm256_add_epi32(dct8_o0, t15a)),
            clip(_mm256_add_epi32(dct8_o1, t14)),
            clip(_mm256_add_epi32(dct8_o2, t13a)),
            clip(_mm256_add_epi32(dct8_o3, t12)),
            clip(_mm256_add_epi32(dct8_o4, t11)),
            clip(_mm256_add_epi32(dct8_o5, t10a)),
            clip(_mm256_add_epi32(dct8_o6, t9)),
            clip(_mm256_add_epi32(dct8_o7, t8a)),
            clip(_mm256_sub_epi32(dct8_o7, t8a)),
            clip(_mm256_sub_epi32(dct8_o6, t9)),
            clip(_mm256_sub_epi32(dct8_o5, t10a)),
            clip(_mm256_sub_epi32(dct8_o4, t11)),
            clip(_mm256_sub_epi32(dct8_o3, t12)),
            clip(_mm256_sub_epi32(dct8_o2, t13a)),
            clip(_mm256_sub_epi32(dct8_o1, t14)),
            clip(_mm256_sub_epi32(dct8_o0, t15a)),
        ];

        // ===== DCT-32 odd half on all 16 odd columns =====
        let pair_32_1_31 = build_pair(_token, cx[1], cx[31]);
        let pair_32_17_15 = build_pair(_token, cx[17], cx[15]);
        let pair_32_9_23 = build_pair(_token, cx[9], cx[23]);
        let pair_32_25_7 = build_pair(_token, cx[25], cx[7]);
        let pair_32_5_27 = build_pair(_token, cx[5], cx[27]);
        let pair_32_21_11 = build_pair(_token, cx[21], cx[11]);
        let pair_32_13_19 = build_pair(_token, cx[13], cx[19]);
        let pair_32_29_3 = build_pair(_token, cx[29], cx[3]);

        // Initial 16 trig butterflies (all >>12 +2048)
        let t16a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_32_1_31, coef_pack(_token, 201, -4091)),
            pd_2048,
        ));
        let t31a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_32_1_31, coef_pack(_token, 4091, 201)),
            pd_2048,
        ));
        let t17a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_32_17_15, coef_pack(_token, 3035, -2751)),
            pd_2048,
        ));
        let t30a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_32_17_15, coef_pack(_token, 2751, 3035)),
            pd_2048,
        ));
        let t18a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_32_9_23, coef_pack(_token, 1751, -3703)),
            pd_2048,
        ));
        let t29a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_32_9_23, coef_pack(_token, 3703, 1751)),
            pd_2048,
        ));
        let t19a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_32_25_7, coef_pack(_token, 3857, -1380)),
            pd_2048,
        ));
        let t28a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_32_25_7, coef_pack(_token, 1380, 3857)),
            pd_2048,
        ));
        let t20a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_32_5_27, coef_pack(_token, 995, -3973)),
            pd_2048,
        ));
        let t27a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_32_5_27, coef_pack(_token, 3973, 995)),
            pd_2048,
        ));
        let t21a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_32_21_11, coef_pack(_token, 3513, -2106)),
            pd_2048,
        ));
        let t26a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_32_21_11, coef_pack(_token, 2106, 3513)),
            pd_2048,
        ));
        // in13*1220 - in19*1645 doubled: in13*2440 - in19*3290
        let t22a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_32_13_19, coef_pack(_token, 2440, -3290)),
            pd_2048,
        ));
        let t25a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_32_13_19, coef_pack(_token, 3290, 2440)),
            pd_2048,
        ));
        let t23a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_32_29_3, coef_pack(_token, 4052, -601)),
            pd_2048,
        ));
        let t24a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_madd_epi16(pair_32_29_3, coef_pack(_token, 601, 4052)),
            pd_2048,
        ));

        // DCT-32 odd stage 1
        let mut t16 = clip(_mm256_add_epi32(t16a, t17a));
        let mut t17 = clip(_mm256_sub_epi32(t16a, t17a));
        let mut t18 = clip(_mm256_sub_epi32(t19a, t18a));
        let t19 = clip(_mm256_add_epi32(t19a, t18a));
        let t20 = clip(_mm256_add_epi32(t20a, t21a));
        let mut t21 = clip(_mm256_sub_epi32(t20a, t21a));
        let mut t22 = clip(_mm256_sub_epi32(t23a, t22a));
        let mut t23 = clip(_mm256_add_epi32(t23a, t22a));
        let mut t24 = clip(_mm256_add_epi32(t24a, t25a));
        let mut t25 = clip(_mm256_sub_epi32(t24a, t25a));
        let mut t26 = clip(_mm256_sub_epi32(t27a, t26a));
        let t27 = clip(_mm256_add_epi32(t27a, t26a));
        let t28 = clip(_mm256_add_epi32(t28a, t29a));
        let mut t29 = clip(_mm256_sub_epi32(t28a, t29a));
        let mut t30 = clip(_mm256_sub_epi32(t31a, t30a));
        let mut t31 = clip(_mm256_add_epi32(t31a, t30a));

        // DCT-32 odd stage 2: trig rotations (799/4017 and 1703/1138)
        let c799 = _mm256_set1_epi32(799);
        let c4017 = _mm256_set1_epi32(4017);
        let c1703 = _mm256_set1_epi32(1703);
        let c1138 = _mm256_set1_epi32(1138);
        let pd_1024 = _mm256_set1_epi32(1024);
        let t17a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_sub_epi32(
                _mm256_mullo_epi32(t30, c799),
                _mm256_mullo_epi32(t17, c4017),
            ),
            pd_2048,
        ));
        let t30a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_add_epi32(
                _mm256_mullo_epi32(t30, c4017),
                _mm256_mullo_epi32(t17, c799),
            ),
            pd_2048,
        ));
        let t18a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_sub_epi32(
                _mm256_setzero_si256(),
                _mm256_add_epi32(
                    _mm256_mullo_epi32(t29, c4017),
                    _mm256_mullo_epi32(t18, c799),
                ),
            ),
            pd_2048,
        ));
        let t29a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_sub_epi32(
                _mm256_mullo_epi32(t29, c799),
                _mm256_mullo_epi32(t18, c4017),
            ),
            pd_2048,
        ));
        let t21a = _mm256_srai_epi32::<11>(_mm256_add_epi32(
            _mm256_sub_epi32(
                _mm256_mullo_epi32(t26, c1703),
                _mm256_mullo_epi32(t21, c1138),
            ),
            pd_1024,
        ));
        let t26a = _mm256_srai_epi32::<11>(_mm256_add_epi32(
            _mm256_add_epi32(
                _mm256_mullo_epi32(t26, c1138),
                _mm256_mullo_epi32(t21, c1703),
            ),
            pd_1024,
        ));
        let t22a = _mm256_srai_epi32::<11>(_mm256_add_epi32(
            _mm256_sub_epi32(
                _mm256_setzero_si256(),
                _mm256_add_epi32(
                    _mm256_mullo_epi32(t25, c1138),
                    _mm256_mullo_epi32(t22, c1703),
                ),
            ),
            pd_1024,
        ));
        let t25a = _mm256_srai_epi32::<11>(_mm256_add_epi32(
            _mm256_sub_epi32(
                _mm256_mullo_epi32(t25, c1703),
                _mm256_mullo_epi32(t22, c1138),
            ),
            pd_1024,
        ));

        // DCT-32 odd stage 3
        let t16a = clip(_mm256_add_epi32(t16, t19));
        t17 = clip(_mm256_add_epi32(t17a, t18a));
        t18 = clip(_mm256_sub_epi32(t17a, t18a));
        let t19a = clip(_mm256_sub_epi32(t16, t19));
        let t20a = clip(_mm256_sub_epi32(t23, t20));
        t21 = clip(_mm256_sub_epi32(t22a, t21a));
        t22 = clip(_mm256_add_epi32(t22a, t21a));
        let t23a = clip(_mm256_add_epi32(t23, t20));
        let t24a = clip(_mm256_add_epi32(t24, t27));
        t25 = clip(_mm256_add_epi32(t25a, t26a));
        t26 = clip(_mm256_sub_epi32(t25a, t26a));
        let t27a = clip(_mm256_sub_epi32(t24, t27));
        let t28a = clip(_mm256_sub_epi32(t31, t28));
        t29 = clip(_mm256_sub_epi32(t30a, t29a));
        t30 = clip(_mm256_add_epi32(t30a, t29a));
        let t31a = clip(_mm256_add_epi32(t31, t28));

        // DCT-32 odd stage 4: trig rotations (1567/3784)
        let t18a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_sub_epi32(
                _mm256_mullo_epi32(t29, c1567),
                _mm256_mullo_epi32(t18, c3784),
            ),
            pd_2048,
        ));
        let t29a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_add_epi32(
                _mm256_mullo_epi32(t29, c3784),
                _mm256_mullo_epi32(t18, c1567),
            ),
            pd_2048,
        ));
        let t19 = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_sub_epi32(
                _mm256_mullo_epi32(t28a, c1567),
                _mm256_mullo_epi32(t19a, c3784),
            ),
            pd_2048,
        ));
        let t28 = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_add_epi32(
                _mm256_mullo_epi32(t28a, c3784),
                _mm256_mullo_epi32(t19a, c1567),
            ),
            pd_2048,
        ));
        let t20 = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_sub_epi32(
                _mm256_setzero_si256(),
                _mm256_add_epi32(
                    _mm256_mullo_epi32(t27a, c3784),
                    _mm256_mullo_epi32(t20a, c1567),
                ),
            ),
            pd_2048,
        ));
        let t27 = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_sub_epi32(
                _mm256_mullo_epi32(t27a, c1567),
                _mm256_mullo_epi32(t20a, c3784),
            ),
            pd_2048,
        ));
        let t21a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_sub_epi32(
                _mm256_setzero_si256(),
                _mm256_add_epi32(
                    _mm256_mullo_epi32(t26, c3784),
                    _mm256_mullo_epi32(t21, c1567),
                ),
            ),
            pd_2048,
        ));
        let t26a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_sub_epi32(
                _mm256_mullo_epi32(t26, c1567),
                _mm256_mullo_epi32(t21, c3784),
            ),
            pd_2048,
        ));

        // DCT-32 odd stage 5
        t16 = clip(_mm256_add_epi32(t16a, t23a));
        let t17a = clip(_mm256_add_epi32(t17, t22));
        t18 = clip(_mm256_add_epi32(t18a, t21a));
        let t19a = clip(_mm256_add_epi32(t19, t20));
        let t20a = clip(_mm256_sub_epi32(t19, t20));
        t21 = clip(_mm256_sub_epi32(t18a, t21a));
        let t22a = clip(_mm256_sub_epi32(t17, t22));
        t23 = clip(_mm256_sub_epi32(t16a, t23a));
        t24 = clip(_mm256_sub_epi32(t31a, t24a));
        let t25a = clip(_mm256_sub_epi32(t30, t25));
        t26 = clip(_mm256_sub_epi32(t29a, t26a));
        let t27a = clip(_mm256_sub_epi32(t28, t27));
        let t28a = clip(_mm256_add_epi32(t28, t27));
        t29 = clip(_mm256_add_epi32(t29a, t26a));
        let t30a = clip(_mm256_add_epi32(t30, t25));
        t31 = clip(_mm256_add_epi32(t31a, t24a));

        // DCT-32 odd final: sqrt(2) cross-muls
        let t20_f = _mm256_srai_epi32::<8>(_mm256_add_epi32(
            _mm256_mullo_epi32(_mm256_sub_epi32(t27a, t20a), c_181),
            pd_128,
        ));
        let t27_f = _mm256_srai_epi32::<8>(_mm256_add_epi32(
            _mm256_mullo_epi32(_mm256_add_epi32(t27a, t20a), c_181),
            pd_128,
        ));
        let t21a_f = _mm256_srai_epi32::<8>(_mm256_add_epi32(
            _mm256_mullo_epi32(_mm256_sub_epi32(t26, t21), c_181),
            pd_128,
        ));
        let t26a_f = _mm256_srai_epi32::<8>(_mm256_add_epi32(
            _mm256_mullo_epi32(_mm256_add_epi32(t26, t21), c_181),
            pd_128,
        ));
        let t22_f = _mm256_srai_epi32::<8>(_mm256_add_epi32(
            _mm256_mullo_epi32(_mm256_sub_epi32(t25a, t22a), c_181),
            pd_128,
        ));
        let t25_f = _mm256_srai_epi32::<8>(_mm256_add_epi32(
            _mm256_mullo_epi32(_mm256_add_epi32(t25a, t22a), c_181),
            pd_128,
        ));
        let t23a_f = _mm256_srai_epi32::<8>(_mm256_add_epi32(
            _mm256_mullo_epi32(_mm256_sub_epi32(t24, t23), c_181),
            pd_128,
        ));
        let t24a_f = _mm256_srai_epi32::<8>(_mm256_add_epi32(
            _mm256_mullo_epi32(_mm256_add_epi32(t24, t23), c_181),
            pd_128,
        ));

        // Final DCT-32 butterfly: combine dct16 even outputs with odd half
        let mut cols = [_mm256_setzero_si256(); 32];
        cols[0] = clip(_mm256_add_epi32(dct16_o[0], t31));
        cols[1] = clip(_mm256_add_epi32(dct16_o[1], t30a));
        cols[2] = clip(_mm256_add_epi32(dct16_o[2], t29));
        cols[3] = clip(_mm256_add_epi32(dct16_o[3], t28a));
        cols[4] = clip(_mm256_add_epi32(dct16_o[4], t27_f));
        cols[5] = clip(_mm256_add_epi32(dct16_o[5], t26a_f));
        cols[6] = clip(_mm256_add_epi32(dct16_o[6], t25_f));
        cols[7] = clip(_mm256_add_epi32(dct16_o[7], t24a_f));
        cols[8] = clip(_mm256_add_epi32(dct16_o[8], t23a_f));
        cols[9] = clip(_mm256_add_epi32(dct16_o[9], t22_f));
        cols[10] = clip(_mm256_add_epi32(dct16_o[10], t21a_f));
        cols[11] = clip(_mm256_add_epi32(dct16_o[11], t20_f));
        cols[12] = clip(_mm256_add_epi32(dct16_o[12], t19a));
        cols[13] = clip(_mm256_add_epi32(dct16_o[13], t18));
        cols[14] = clip(_mm256_add_epi32(dct16_o[14], t17a));
        cols[15] = clip(_mm256_add_epi32(dct16_o[15], t16));
        cols[16] = clip(_mm256_sub_epi32(dct16_o[15], t16));
        cols[17] = clip(_mm256_sub_epi32(dct16_o[14], t17a));
        cols[18] = clip(_mm256_sub_epi32(dct16_o[13], t18));
        cols[19] = clip(_mm256_sub_epi32(dct16_o[12], t19a));
        cols[20] = clip(_mm256_sub_epi32(dct16_o[11], t20_f));
        cols[21] = clip(_mm256_sub_epi32(dct16_o[10], t21a_f));
        cols[22] = clip(_mm256_sub_epi32(dct16_o[9], t22_f));
        cols[23] = clip(_mm256_sub_epi32(dct16_o[8], t23a_f));
        cols[24] = clip(_mm256_sub_epi32(dct16_o[7], t24a_f));
        cols[25] = clip(_mm256_sub_epi32(dct16_o[6], t25_f));
        cols[26] = clip(_mm256_sub_epi32(dct16_o[5], t26a_f));
        cols[27] = clip(_mm256_sub_epi32(dct16_o[4], t27_f));
        cols[28] = clip(_mm256_sub_epi32(dct16_o[3], t28a));
        cols[29] = clip(_mm256_sub_epi32(dct16_o[2], t29));
        cols[30] = clip(_mm256_sub_epi32(dct16_o[1], t30a));
        cols[31] = clip(_mm256_sub_epi32(dct16_o[0], t31));

        // Transpose 32x8 -> 8x32 in 4 chunks of 8 columns, store row-major.
        for chunk in 0..4 {
            let b = chunk * 8;
            let chunk_cols: [__m256i; 8] = [
                cols[b],
                cols[b + 1],
                cols[b + 2],
                cols[b + 3],
                cols[b + 4],
                cols[b + 5],
                cols[b + 6],
                cols[b + 7],
            ];
            let rows = transpose_8x8_i32!(chunk_cols);
            for row in 0..8 {
                let y = y_base + row;
                let arr: &mut [i32; 8] = (&mut out[y * 32 + b..y * 32 + b + 8]).try_into().unwrap();
                storeu_256!(arr, [i32; 8], rows[row]);
            }
        }
    }
    out
}

/// SIMD row ADST-16 for 16xN transforms, 8bpc. Same shape as
/// `simd_row_dct16_8bpc_8rows` but calls `adst16_1d_cols8`. If `flipped`,
/// reverses output order after ADST (flipadst).
#[cfg(target_arch = "x86_64")]
#[rite]
#[inline(always)]
fn simd_row_adst16_8bpc_8rows(
    token: Desktop64,
    coeff: &[i16],
    coeff_h: usize,
    y_base: usize,
    apply_rect2: bool,
    flipped: bool,
    rnd: i32,
    shift: i32,
    tmp: &mut [i32],
    row_min: i32,
    row_max: i32,
    col_min: i32,
    col_max: i32,
) {
    let row_min_v = _mm256_set1_epi32(row_min);
    let row_max_v = _mm256_set1_epi32(row_max);
    let col_min_v = _mm256_set1_epi32(col_min);
    let col_max_v = _mm256_set1_epi32(col_max);
    let rect2_v = _mm256_set1_epi32(181);
    let bias_v = _mm256_set1_epi32(128);
    let rnd_v = _mm256_set1_epi32(rnd);
    let mut cols = [_mm256_setzero_si256(); 16];
    for x in 0..16 {
        let off = y_base + x * coeff_h;
        let arr: &[i16; 8] = (&coeff[off..off + 8]).try_into().unwrap();
        let v16 = loadu_128!(arr);
        let v32 = _mm256_cvtepi16_epi32(v16);
        cols[x] = if apply_rect2 {
            _mm256_srai_epi32::<8>(_mm256_add_epi32(_mm256_mullo_epi32(v32, rect2_v), bias_v))
        } else {
            v32
        };
    }
    adst16_1d_cols8(token, &mut cols, row_min_v, row_max_v);
    if flipped {
        cols.reverse();
    }
    for x in 0..16 {
        let rounded = match shift {
            0 => _mm256_add_epi32(cols[x], rnd_v),
            1 => _mm256_srai_epi32::<1>(_mm256_add_epi32(cols[x], rnd_v)),
            2 => _mm256_srai_epi32::<2>(_mm256_add_epi32(cols[x], rnd_v)),
            _ => _mm256_srai_epi32::<2>(_mm256_add_epi32(cols[x], rnd_v)),
        };
        cols[x] = _mm256_max_epi32(_mm256_min_epi32(rounded, col_max_v), col_min_v);
    }
    for chunk in 0..2 {
        let b = chunk * 8;
        let chunk_cols: [__m256i; 8] = [
            cols[b + 0],
            cols[b + 1],
            cols[b + 2],
            cols[b + 3],
            cols[b + 4],
            cols[b + 5],
            cols[b + 6],
            cols[b + 7],
        ];
        let rows = transpose_8x8_i32!(chunk_cols);
        let s = 16;
        storeu_256!(
            &mut tmp[(y_base + 0) * s + b..(y_base + 0) * s + b + 8],
            [i32; 8],
            rows[0]
        );
        storeu_256!(
            &mut tmp[(y_base + 1) * s + b..(y_base + 1) * s + b + 8],
            [i32; 8],
            rows[1]
        );
        storeu_256!(
            &mut tmp[(y_base + 2) * s + b..(y_base + 2) * s + b + 8],
            [i32; 8],
            rows[2]
        );
        storeu_256!(
            &mut tmp[(y_base + 3) * s + b..(y_base + 3) * s + b + 8],
            [i32; 8],
            rows[3]
        );
        storeu_256!(
            &mut tmp[(y_base + 4) * s + b..(y_base + 4) * s + b + 8],
            [i32; 8],
            rows[4]
        );
        storeu_256!(
            &mut tmp[(y_base + 5) * s + b..(y_base + 5) * s + b + 8],
            [i32; 8],
            rows[5]
        );
        storeu_256!(
            &mut tmp[(y_base + 6) * s + b..(y_base + 6) * s + b + 8],
            [i32; 8],
            rows[6]
        );
        storeu_256!(
            &mut tmp[(y_base + 7) * s + b..(y_base + 7) * s + b + 8],
            [i32; 8],
            rows[7]
        );
    }
}

/// SIMD row DCT-16 for 16xN transforms, 8bpc. Processes 8 rows at once via
/// `dct16_1d_cols8` + 16x8 → 8x16 transpose. Coeff is column-major
/// (stride `coeff_h`); writes row-major into `tmp` (stride 16).
#[cfg(target_arch = "x86_64")]
#[rite]
#[inline(always)]
fn simd_row_dct16_8bpc_8rows(
    token: Desktop64,
    coeff: &[i16],
    coeff_h: usize,
    y_base: usize,
    apply_rect2: bool,
    rnd: i32,
    shift: i32,
    tmp: &mut [i32],
    row_min: i32,
    row_max: i32,
    col_min: i32,
    col_max: i32,
) {
    let row_min_v = _mm256_set1_epi32(row_min);
    let row_max_v = _mm256_set1_epi32(row_max);
    let col_min_v = _mm256_set1_epi32(col_min);
    let col_max_v = _mm256_set1_epi32(col_max);
    let rect2_v = _mm256_set1_epi32(181);
    let bias_v = _mm256_set1_epi32(128);
    let rnd_v = _mm256_set1_epi32(rnd);
    let mut cols = [_mm256_setzero_si256(); 16];
    for x in 0..16 {
        let off = y_base + x * coeff_h;
        let arr: &[i16; 8] = (&coeff[off..off + 8]).try_into().unwrap();
        let v16 = loadu_128!(arr);
        let v32 = _mm256_cvtepi16_epi32(v16);
        cols[x] = if apply_rect2 {
            _mm256_srai_epi32::<8>(_mm256_add_epi32(_mm256_mullo_epi32(v32, rect2_v), bias_v))
        } else {
            v32
        };
    }
    dct16_1d_cols8(token, &mut cols, row_min_v, row_max_v);
    for x in 0..16 {
        let rounded = match shift {
            0 => _mm256_add_epi32(cols[x], rnd_v),
            1 => _mm256_srai_epi32::<1>(_mm256_add_epi32(cols[x], rnd_v)),
            2 => _mm256_srai_epi32::<2>(_mm256_add_epi32(cols[x], rnd_v)),
            _ => _mm256_srai_epi32::<2>(_mm256_add_epi32(cols[x], rnd_v)),
        };
        cols[x] = _mm256_max_epi32(_mm256_min_epi32(rounded, col_max_v), col_min_v);
    }
    // Transpose 16x8 → 8x16 in 2 chunks of 8 columns, store row-major (stride 16).
    for chunk in 0..2 {
        let b = chunk * 8;
        let chunk_cols: [__m256i; 8] = [
            cols[b + 0],
            cols[b + 1],
            cols[b + 2],
            cols[b + 3],
            cols[b + 4],
            cols[b + 5],
            cols[b + 6],
            cols[b + 7],
        ];
        let rows = transpose_8x8_i32!(chunk_cols);
        let s = 16;
        storeu_256!(
            &mut tmp[(y_base + 0) * s + b..(y_base + 0) * s + b + 8],
            [i32; 8],
            rows[0]
        );
        storeu_256!(
            &mut tmp[(y_base + 1) * s + b..(y_base + 1) * s + b + 8],
            [i32; 8],
            rows[1]
        );
        storeu_256!(
            &mut tmp[(y_base + 2) * s + b..(y_base + 2) * s + b + 8],
            [i32; 8],
            rows[2]
        );
        storeu_256!(
            &mut tmp[(y_base + 3) * s + b..(y_base + 3) * s + b + 8],
            [i32; 8],
            rows[3]
        );
        storeu_256!(
            &mut tmp[(y_base + 4) * s + b..(y_base + 4) * s + b + 8],
            [i32; 8],
            rows[4]
        );
        storeu_256!(
            &mut tmp[(y_base + 5) * s + b..(y_base + 5) * s + b + 8],
            [i32; 8],
            rows[5]
        );
        storeu_256!(
            &mut tmp[(y_base + 6) * s + b..(y_base + 6) * s + b + 8],
            [i32; 8],
            rows[6]
        );
        storeu_256!(
            &mut tmp[(y_base + 7) * s + b..(y_base + 7) * s + b + 8],
            [i32; 8],
            rows[7]
        );
    }
}

/// SIMD row DCT-32 for 32xN transforms, 8bpc.
/// Loads 8 consecutive rows starting at `y_base` from `coeff` (column-major,
/// stride `coeff_h`). Optionally applies rect2_scale (`* 181 + 128 >> 8`).
/// Runs `dct32_1d_cols8_i16` (8 rows in parallel via SIMD lanes), rounds with
/// `rnd` then arithmetic-right-shifts by `shift`, clips to col range, and
/// transposes 32x8 → 8x32 to store row-major into `tmp` (stride 32).
#[cfg(target_arch = "x86_64")]
#[rite]
#[inline(always)]
fn simd_row_dct32_8bpc_8rows(
    token: Desktop64,
    coeff: &[i16],
    coeff_h: usize,
    y_base: usize,
    apply_rect2: bool,
    rnd: i32,
    shift: i32,
    tmp: &mut [i32],
    row_min: i32,
    row_max: i32,
    col_min: i32,
    col_max: i32,
) {
    let row_min_v = _mm256_set1_epi32(row_min);
    let row_max_v = _mm256_set1_epi32(row_max);
    let col_min_v = _mm256_set1_epi32(col_min);
    let col_max_v = _mm256_set1_epi32(col_max);
    let rect2_v = _mm256_set1_epi32(181);
    let bias_v = _mm256_set1_epi32(128);
    let rnd_v = _mm256_set1_epi32(rnd);
    let mut cols = [_mm256_setzero_si256(); 32];
    for x in 0..32 {
        let off = y_base + x * coeff_h;
        let arr: &[i16; 8] = (&coeff[off..off + 8]).try_into().unwrap();
        let v16 = loadu_128!(arr);
        let v32 = _mm256_cvtepi16_epi32(v16);
        cols[x] = if apply_rect2 {
            _mm256_srai_epi32::<8>(_mm256_add_epi32(_mm256_mullo_epi32(v32, rect2_v), bias_v))
        } else {
            v32
        };
    }
    dct32_1d_cols8_i16(token, &mut cols, row_min_v, row_max_v);
    for x in 0..32 {
        let rounded = match shift {
            1 => _mm256_srai_epi32::<1>(_mm256_add_epi32(cols[x], rnd_v)),
            2 => _mm256_srai_epi32::<2>(_mm256_add_epi32(cols[x], rnd_v)),
            _ => _mm256_add_epi32(cols[x], rnd_v),
        };
        cols[x] = _mm256_max_epi32(_mm256_min_epi32(rounded, col_max_v), col_min_v);
    }
    // Transpose 32x8 → 8x32 (4 chunks of 8 columns), store rows contiguously
    for chunk in 0..4 {
        let b = chunk * 8;
        let chunk_cols: [__m256i; 8] = [
            cols[b + 0],
            cols[b + 1],
            cols[b + 2],
            cols[b + 3],
            cols[b + 4],
            cols[b + 5],
            cols[b + 6],
            cols[b + 7],
        ];
        let rows = transpose_8x8_i32!(chunk_cols);
        let s = 32;
        storeu_256!(
            &mut tmp[(y_base + 0) * s + b..(y_base + 0) * s + b + 8],
            [i32; 8],
            rows[0]
        );
        storeu_256!(
            &mut tmp[(y_base + 1) * s + b..(y_base + 1) * s + b + 8],
            [i32; 8],
            rows[1]
        );
        storeu_256!(
            &mut tmp[(y_base + 2) * s + b..(y_base + 2) * s + b + 8],
            [i32; 8],
            rows[2]
        );
        storeu_256!(
            &mut tmp[(y_base + 3) * s + b..(y_base + 3) * s + b + 8],
            [i32; 8],
            rows[3]
        );
        storeu_256!(
            &mut tmp[(y_base + 4) * s + b..(y_base + 4) * s + b + 8],
            [i32; 8],
            rows[4]
        );
        storeu_256!(
            &mut tmp[(y_base + 5) * s + b..(y_base + 5) * s + b + 8],
            [i32; 8],
            rows[5]
        );
        storeu_256!(
            &mut tmp[(y_base + 6) * s + b..(y_base + 6) * s + b + 8],
            [i32; 8],
            rows[6]
        );
        storeu_256!(
            &mut tmp[(y_base + 7) * s + b..(y_base + 7) * s + b + 8],
            [i32; 8],
            rows[7]
        );
    }
}

/// Full 2D DCT_DCT 32x16 inverse transform
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_32x16_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    // W=32, H=16, shift=2 for 32x16
    let row_clip_min = i16::MIN as i32;
    let row_clip_max = i16::MAX as i32;
    let col_clip_min = i16::MIN as i32;
    let col_clip_max = i16::MAX as i32;
    let mut tmp = [0i32; 32 * 16];

    // SIMD row transform: 2 batches of 8 rows. is_rect2=true, shift=1, rnd=1.
    {
        let coeff_slice = coeff.as_slice();
        for y_base in [0usize, 8] {
            simd_row_dct32_8bpc_8rows(
                _token,
                coeff_slice,
                16,
                y_base,
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
        add_to_dst_8bpc_avx512(t512, &mut *dst, dst_stride, &tmp, 32, 32, 16, bitdepth_max);
        coeff[..512].fill(0);
        return;
    }

    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi16(bitdepth_max as i16);
    let rnd_final = _mm256_set1_epi32(8);

    for y in 0..16 {
        let dst_off = y * dst_stride;

        // Process 32 pixels in two 16-pixel chunks
        for chunk in 0..2 {
            let chunk_off = chunk * 16;
            let d = loadu_128!(
                <&[u8; 16]>::try_from(&dst[dst_off + chunk_off..dst_off + chunk_off + 16]).unwrap()
            );
            let d16 = _mm256_cvtepu8_epi16(d);

            let c0 = _mm256_set_epi32(
                tmp[y * 32 + chunk_off + 7],
                tmp[y * 32 + chunk_off + 6],
                tmp[y * 32 + chunk_off + 5],
                tmp[y * 32 + chunk_off + 4],
                tmp[y * 32 + chunk_off + 3],
                tmp[y * 32 + chunk_off + 2],
                tmp[y * 32 + chunk_off + 1],
                tmp[y * 32 + chunk_off + 0],
            );
            let c1 = _mm256_set_epi32(
                tmp[y * 32 + chunk_off + 15],
                tmp[y * 32 + chunk_off + 14],
                tmp[y * 32 + chunk_off + 13],
                tmp[y * 32 + chunk_off + 12],
                tmp[y * 32 + chunk_off + 11],
                tmp[y * 32 + chunk_off + 10],
                tmp[y * 32 + chunk_off + 9],
                tmp[y * 32 + chunk_off + 8],
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
                <&mut [u8; 16]>::try_from(&mut dst[dst_off + chunk_off..dst_off + chunk_off + 16])
                    .unwrap(),
                _mm256_castsi256_si128(packed)
            );
        }
    }

    // Clear coefficients
    coeff[..512].fill(0);
}

/// FFI wrapper for 32x16 DCT_DCT 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_32x16_8bpc_avx2(
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

    inv_txfm_add_dct_dct_32x16_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// 16x32 and 32x16 IDTX transforms
// ============================================================================

/// 16x32 IDTX inverse transform
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_identity_identity_16x32_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let clip_min = i16::MIN as i32;
    let clip_max = i16::MAX as i32;
    let mut tmp = [0i32; 16 * 32];

    let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

    // Row transform (16 elements each, 32 rows)
    let rnd = 1;
    let shift = 1;
    for y in 0..32 {
        let mut scratch = [0i32; 16];
        for x in 0..16 {
            scratch[x] = rect2_scale(coeff[y + x * 32] as i32);
        }
        identity16_1d(&mut scratch[..16], 1, clip_min, clip_max);
        for x in 0..16 {
            tmp[y * 16 + x] = iclip((scratch[x] + rnd) >> shift, clip_min, clip_max);
        }
    }

    // Column transform: SIMD across 16 columns (2 chunks of 8), identity32 = *4
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
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi16(bitdepth_max as i16);
    let rnd_final = _mm256_set1_epi32(8);

    for y in 0..32 {
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
    coeff[..512].fill(0);
}

/// FFI wrapper for 16x32 IDTX 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_identity_16x32_8bpc_avx2(
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

    inv_txfm_add_identity_identity_16x32_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 32x16 IDTX inverse transform
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_identity_identity_32x16_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let clip_min = i16::MIN as i32;
    let clip_max = i16::MAX as i32;
    let mut tmp = [0i32; 32 * 16];

    let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

    // Row transform (32 elements each, 16 rows)
    let rnd = 1;
    let shift = 1;
    for y in 0..16 {
        let mut scratch = [0i32; 32];
        for x in 0..32 {
            scratch[x] = rect2_scale(coeff[y + x * 16] as i32);
        }
        identity32_1d(&mut scratch[..32], 1, clip_min, clip_max);
        for x in 0..32 {
            tmp[y * 32 + x] = iclip((scratch[x] + rnd) >> shift, clip_min, clip_max);
        }
    }

    // Column transform: SIMD across 32 columns (4 chunks of 8), identity16 = 2*in + (in*1697+1024 >> 11)
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
        add_to_dst_8bpc_avx512(t512, &mut *dst, dst_stride, &tmp, 32, 32, 16, bitdepth_max);
        coeff[..512].fill(0);
        return;
    }

    for y in 0..16 {
        let dst_off = y * dst_stride;
        for x in 0..32 {
            let d = dst[dst_off + x] as i32;
            let c = (tmp[y * 32 + x] + 8) >> 4;
            let result = iclip(d + c, 0, bitdepth_max);
            dst[dst_off + x] = result as u8;
        }
    }

    // Clear coefficients
    coeff[..512].fill(0);
}

/// FFI wrapper for 32x16 IDTX 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_identity_32x16_8bpc_avx2(
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

    inv_txfm_add_identity_identity_32x16_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// Full 2D DCT_DCT 32x64 inverse transform
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_32x64_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    // W=32, H=64, shift=2 for 32x64
    let row_clip_min = i16::MIN as i32;
    let row_clip_max = i16::MAX as i32;
    let col_clip_min = i16::MIN as i32;
    let col_clip_max = i16::MAX as i32;
    let mut tmp = [0i32; 32 * 64];

    // SIMD row transform: 4 batches of 8 rows (only 32 rows of coefficients
    // exist for 64-pt transforms; remaining 32 rows pad below). is_rect2=true,
    // shift=1, rnd=1.
    {
        let coeff_slice = coeff.as_slice();
        for y_base in [0usize, 8, 16, 24] {
            simd_row_dct32_8bpc_8rows(
                _token,
                coeff_slice,
                32,
                y_base,
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
    }

    // Pad remaining rows with zeros
    for y in 32..64 {
        for x in 0..32 {
            tmp[y * 32 + x] = 0;
        }
    }

    // Column transform (64 elements each, 32 columns)
    for x in 0..32 {
        dct64_1d(&mut tmp[x..], 32, col_clip_min, col_clip_max);
    }

    // Add to destination
    #[cfg(target_arch = "x86_64")]
    if let Some(t512) = crate::src::cpu::summon_avx512() {
        add_to_dst_8bpc_avx512(t512, &mut *dst, dst_stride, &tmp, 32, 32, 64, bitdepth_max);
        coeff[..1024].fill(0);
        return;
    }

    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi16(bitdepth_max as i16);
    let rnd_final = _mm256_set1_epi32(8);

    for y in 0..64 {
        let dst_off = y * dst_stride;

        // Process 32 pixels in two 16-pixel chunks
        for chunk in 0..2 {
            let chunk_off = chunk * 16;
            let d = loadu_128!(
                <&[u8; 16]>::try_from(&dst[dst_off + chunk_off..dst_off + chunk_off + 16]).unwrap()
            );
            let d16 = _mm256_cvtepu8_epi16(d);

            let c0 = _mm256_set_epi32(
                tmp[y * 32 + chunk_off + 7],
                tmp[y * 32 + chunk_off + 6],
                tmp[y * 32 + chunk_off + 5],
                tmp[y * 32 + chunk_off + 4],
                tmp[y * 32 + chunk_off + 3],
                tmp[y * 32 + chunk_off + 2],
                tmp[y * 32 + chunk_off + 1],
                tmp[y * 32 + chunk_off + 0],
            );
            let c1 = _mm256_set_epi32(
                tmp[y * 32 + chunk_off + 15],
                tmp[y * 32 + chunk_off + 14],
                tmp[y * 32 + chunk_off + 13],
                tmp[y * 32 + chunk_off + 12],
                tmp[y * 32 + chunk_off + 11],
                tmp[y * 32 + chunk_off + 10],
                tmp[y * 32 + chunk_off + 9],
                tmp[y * 32 + chunk_off + 8],
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
                <&mut [u8; 16]>::try_from(&mut dst[dst_off + chunk_off..dst_off + chunk_off + 16])
                    .unwrap(),
                _mm256_castsi256_si128(packed)
            );
        }
    }

    // Clear coefficients
    coeff[..1024].fill(0);
}

/// FFI wrapper for 32x64 DCT_DCT 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_32x64_8bpc_avx2(
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

    inv_txfm_add_dct_dct_32x64_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// Full 2D DCT_DCT 64x32 inverse transform
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_64x32_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    // W=64, H=32, shift=2 for 64x32
    let row_clip_min = i16::MIN as i32;
    let row_clip_max = i16::MAX as i32;
    let col_clip_min = i16::MIN as i32;
    let col_clip_max = i16::MAX as i32;
    let mut tmp = [0i32; 64 * 32];

    // is_rect2 = true for 64x32
    let rect2_scale = |v: i32| (v * 181 + 128) >> 8;

    // Row transform (64 elements each, 32 rows)
    // But only first 32 columns have coefficients for 64-pt transforms
    let rnd = 1;
    let shift = 1;
    for y in 0..32 {
        let mut scratch = [0i32; 64];
        for x in 0..32 {
            scratch[x] = rect2_scale(coeff[y + x * 32] as i32);
        }
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
            dct32_1d_cols8_i16(_token, &mut v, min_v, max_v);
            for i in 0..32 {
                storeu_256!(&mut tmp[i * 64 + cx..i * 64 + cx + 8], [i32; 8], v[i]);
            }
        }
    }

    // Add to destination
    #[cfg(target_arch = "x86_64")]
    if let Some(t512) = crate::src::cpu::summon_avx512() {
        add_to_dst_8bpc_avx512(t512, &mut *dst, dst_stride, &tmp, 64, 64, 32, bitdepth_max);
        coeff[..1024].fill(0);
        return;
    }

    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi16(bitdepth_max as i16);
    let rnd_final = _mm256_set1_epi32(8);

    for y in 0..32 {
        let dst_off = y * dst_stride;

        // Process 64 pixels in four 16-pixel chunks
        for chunk in 0..4 {
            let chunk_off = chunk * 16;
            let d = loadu_128!(
                <&[u8; 16]>::try_from(&dst[dst_off + chunk_off..dst_off + chunk_off + 16]).unwrap()
            );
            let d16 = _mm256_cvtepu8_epi16(d);

            let c0 = _mm256_set_epi32(
                tmp[y * 64 + chunk_off + 7],
                tmp[y * 64 + chunk_off + 6],
                tmp[y * 64 + chunk_off + 5],
                tmp[y * 64 + chunk_off + 4],
                tmp[y * 64 + chunk_off + 3],
                tmp[y * 64 + chunk_off + 2],
                tmp[y * 64 + chunk_off + 1],
                tmp[y * 64 + chunk_off + 0],
            );
            let c1 = _mm256_set_epi32(
                tmp[y * 64 + chunk_off + 15],
                tmp[y * 64 + chunk_off + 14],
                tmp[y * 64 + chunk_off + 13],
                tmp[y * 64 + chunk_off + 12],
                tmp[y * 64 + chunk_off + 11],
                tmp[y * 64 + chunk_off + 10],
                tmp[y * 64 + chunk_off + 9],
                tmp[y * 64 + chunk_off + 8],
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
                <&mut [u8; 16]>::try_from(&mut dst[dst_off + chunk_off..dst_off + chunk_off + 16])
                    .unwrap(),
                _mm256_castsi256_si128(packed)
            );
        }
    }

    // Clear coefficients
    coeff[..1024].fill(0);
}

/// FFI wrapper for 64x32 DCT_DCT 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_64x32_8bpc_avx2(
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

    inv_txfm_add_dct_dct_64x32_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// RECTANGULAR TRANSFORMS - 4:1 aspect ratio (4x16, 16x4, 8x32, 32x8)
// ============================================================================

/// Full 2D DCT_DCT 4x16 inverse transform
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_4x16_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    // W=4, H=16, 4:1 ratio
    let row_clip_min = i16::MIN as i32;
    let row_clip_max = i16::MAX as i32;
    let col_clip_min = i16::MIN as i32;
    let col_clip_max = i16::MAX as i32;
    let mut tmp = [0i32; 4 * 16];

    // Row transform (4 elements each, 16 rows), shift=1 for 4x16
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

    // Column transform (16 elements each, 4 columns)
    for x in 0..4 {
        dct16_1d(&mut tmp[x..], 4, col_clip_min, col_clip_max);
    }

    // Add to destination - 4 pixels at a time
    for y in 0..16 {
        let dst_off = y * dst_stride;
        for x in 0..4 {
            let d = dst[dst_off + x] as i32;
            let c = (tmp[y * 4 + x] + 8) >> 4;
            let result = iclip(d + c, 0, bitdepth_max);
            dst[dst_off + x] = result as u8;
        }
    }

    // Clear coefficients
    coeff[..64].fill(0);
}

/// FFI wrapper for 4x16 DCT_DCT 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_4x16_8bpc_avx2(
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

    inv_txfm_add_dct_dct_4x16_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// Full 2D DCT_DCT 16x4 inverse transform
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_16x4_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    // W=16, H=4, 4:1 ratio
    let row_clip_min = i16::MIN as i32;
    let row_clip_max = i16::MAX as i32;
    let col_clip_min = i16::MIN as i32;
    let col_clip_max = i16::MAX as i32;
    let mut tmp = [0i32; 16 * 4];

    // rect4 scaling
    // Row transform (16 elements each, 4 rows), shift=1 for 16x4
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

    // Column transform: SIMD across 16 columns (2 chunks of 8), 4 rows
    {
        let min_v = _mm256_set1_epi32(col_clip_min);
        let max_v = _mm256_set1_epi32(col_clip_max);
        for cx_chunk in 0..2 {
            let cx = cx_chunk * 8;
            let mut v = [_mm256_setzero_si256(); 4];
            for i in 0..4 {
                v[i] = loadu_256!(&tmp[i * 16 + cx..i * 16 + cx + 8], [i32; 8]);
            }
            dct4_1d_cols8(_token, &mut v, min_v, max_v);
            for i in 0..4 {
                storeu_256!(&mut tmp[i * 16 + cx..i * 16 + cx + 8], [i32; 8], v[i]);
            }
        }
    }

    // Add to destination
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi16(bitdepth_max as i16);
    let rnd_final = _mm256_set1_epi32(8);

    for y in 0..4 {
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
    coeff[..64].fill(0);
}

/// FFI wrapper for 16x4 DCT_DCT 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_16x4_8bpc_avx2(
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

    inv_txfm_add_dct_dct_16x4_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// 4x16 and 16x4 ADST/FLIPADST variants
// ============================================================================

/// Helper macro for 4x16 transforms with configurable row/col transforms
macro_rules! impl_4x16_transform {
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
            let mut tmp = [0i32; 4 * 16];

            // Row transform (4 elements each, 16 rows), shift=1 for 4x16
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

            // Column transform (16 elements each, 4 columns)
            for x in 0..4 {
                $col_fn(&mut tmp[x..], 4, col_clip_min, col_clip_max);
            }

            // Add to destination
            for y in 0..16 {
                let dst_off = y * dst_stride;
                for x in 0..4 {
                    let d = dst[dst_off + x] as i32;
                    let c = (tmp[y * 4 + x] + 8) >> 4;
                    let result = iclip(d + c, 0, bitdepth_max);
                    dst[dst_off + x] = result as u8;
                }
            }

            // Clear coefficients
            coeff[..64].fill(0);
        }
    };
}

/// Helper macro for 16x4 transforms with configurable row/col transforms
macro_rules! impl_16x4_transform {
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
            let mut tmp = [0i32; 16 * 4];

            // Row transform (16 elements each, 4 rows), shift=1 for 16x4
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

            // Column transform (4 elements each, 16 columns)
            for x in 0..16 {
                $col_fn(&mut tmp[x..], 16, col_clip_min, col_clip_max);
            }

            // Add to destination
            let zero = _mm256_setzero_si256();
            let max_val = _mm256_set1_epi16(bitdepth_max as i16);
            let rnd_final = _mm256_set1_epi32(8);

            for y in 0..4 {
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
            coeff[..64].fill(0);
        }
    };
}

// Generate 4x16 ADST inner functions
impl_4x16_transform!(
    inv_txfm_add_adst_dct_4x16_8bpc_avx2_inner,
    adst4_1d,
    dct16_1d
);
impl_4x16_transform!(
    inv_txfm_add_dct_adst_4x16_8bpc_avx2_inner,
    dct4_1d,
    adst16_1d
);
impl_4x16_transform!(
    inv_txfm_add_adst_adst_4x16_8bpc_avx2_inner,
    adst4_1d,
    adst16_1d
);
impl_4x16_transform!(
    inv_txfm_add_flipadst_dct_4x16_8bpc_avx2_inner,
    flipadst4_1d,
    dct16_1d
);
impl_4x16_transform!(
    inv_txfm_add_dct_flipadst_4x16_8bpc_avx2_inner,
    dct4_1d,
    flipadst16_1d
);
impl_4x16_transform!(
    inv_txfm_add_flipadst_flipadst_4x16_8bpc_avx2_inner,
    flipadst4_1d,
    flipadst16_1d
);
impl_4x16_transform!(
    inv_txfm_add_adst_flipadst_4x16_8bpc_avx2_inner,
    adst4_1d,
    flipadst16_1d
);
impl_4x16_transform!(
    inv_txfm_add_flipadst_adst_4x16_8bpc_avx2_inner,
    flipadst4_1d,
    adst16_1d
);

// Generate 16x4 ADST inner functions
impl_16x4_transform!(
    inv_txfm_add_adst_dct_16x4_8bpc_avx2_inner,
    adst16_1d,
    dct4_1d
);
impl_16x4_transform!(
    inv_txfm_add_dct_adst_16x4_8bpc_avx2_inner,
    dct16_1d,
    adst4_1d
);
impl_16x4_transform!(
    inv_txfm_add_adst_adst_16x4_8bpc_avx2_inner,
    adst16_1d,
    adst4_1d
);
impl_16x4_transform!(
    inv_txfm_add_flipadst_dct_16x4_8bpc_avx2_inner,
    flipadst16_1d,
    dct4_1d
);
impl_16x4_transform!(
    inv_txfm_add_dct_flipadst_16x4_8bpc_avx2_inner,
    dct16_1d,
    flipadst4_1d
);
impl_16x4_transform!(
    inv_txfm_add_flipadst_flipadst_16x4_8bpc_avx2_inner,
    flipadst16_1d,
    flipadst4_1d
);
impl_16x4_transform!(
    inv_txfm_add_adst_flipadst_16x4_8bpc_avx2_inner,
    adst16_1d,
    flipadst4_1d
);
impl_16x4_transform!(
    inv_txfm_add_flipadst_adst_16x4_8bpc_avx2_inner,
    flipadst16_1d,
    adst4_1d
);

/// FFI wrapper macro for 4x16 transforms
macro_rules! impl_4x16_ffi_wrapper {
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

/// FFI wrapper macro for 16x4 transforms
macro_rules! impl_16x4_ffi_wrapper {
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

// Generate 4x16 FFI wrappers
impl_4x16_ffi_wrapper!(
    inv_txfm_add_adst_dct_4x16_8bpc_avx2,
    inv_txfm_add_adst_dct_4x16_8bpc_avx2_inner
);
impl_4x16_ffi_wrapper!(
    inv_txfm_add_dct_adst_4x16_8bpc_avx2,
    inv_txfm_add_dct_adst_4x16_8bpc_avx2_inner
);
impl_4x16_ffi_wrapper!(
    inv_txfm_add_adst_adst_4x16_8bpc_avx2,
    inv_txfm_add_adst_adst_4x16_8bpc_avx2_inner
);
impl_4x16_ffi_wrapper!(
    inv_txfm_add_flipadst_dct_4x16_8bpc_avx2,
    inv_txfm_add_flipadst_dct_4x16_8bpc_avx2_inner
);
impl_4x16_ffi_wrapper!(
    inv_txfm_add_dct_flipadst_4x16_8bpc_avx2,
    inv_txfm_add_dct_flipadst_4x16_8bpc_avx2_inner
);
impl_4x16_ffi_wrapper!(
    inv_txfm_add_flipadst_flipadst_4x16_8bpc_avx2,
    inv_txfm_add_flipadst_flipadst_4x16_8bpc_avx2_inner
);
impl_4x16_ffi_wrapper!(
    inv_txfm_add_adst_flipadst_4x16_8bpc_avx2,
    inv_txfm_add_adst_flipadst_4x16_8bpc_avx2_inner
);
impl_4x16_ffi_wrapper!(
    inv_txfm_add_flipadst_adst_4x16_8bpc_avx2,
    inv_txfm_add_flipadst_adst_4x16_8bpc_avx2_inner
);

// Generate 16x4 FFI wrappers
impl_16x4_ffi_wrapper!(
    inv_txfm_add_adst_dct_16x4_8bpc_avx2,
    inv_txfm_add_adst_dct_16x4_8bpc_avx2_inner
);
impl_16x4_ffi_wrapper!(
    inv_txfm_add_dct_adst_16x4_8bpc_avx2,
    inv_txfm_add_dct_adst_16x4_8bpc_avx2_inner
);
impl_16x4_ffi_wrapper!(
    inv_txfm_add_adst_adst_16x4_8bpc_avx2,
    inv_txfm_add_adst_adst_16x4_8bpc_avx2_inner
);
impl_16x4_ffi_wrapper!(
    inv_txfm_add_flipadst_dct_16x4_8bpc_avx2,
    inv_txfm_add_flipadst_dct_16x4_8bpc_avx2_inner
);
impl_16x4_ffi_wrapper!(
    inv_txfm_add_dct_flipadst_16x4_8bpc_avx2,
    inv_txfm_add_dct_flipadst_16x4_8bpc_avx2_inner
);
impl_16x4_ffi_wrapper!(
    inv_txfm_add_flipadst_flipadst_16x4_8bpc_avx2,
    inv_txfm_add_flipadst_flipadst_16x4_8bpc_avx2_inner
);
impl_16x4_ffi_wrapper!(
    inv_txfm_add_adst_flipadst_16x4_8bpc_avx2,
    inv_txfm_add_adst_flipadst_16x4_8bpc_avx2_inner
);
impl_16x4_ffi_wrapper!(
    inv_txfm_add_flipadst_adst_16x4_8bpc_avx2,
    inv_txfm_add_flipadst_adst_16x4_8bpc_avx2_inner
);

// IDTX for 4x16 and 16x4
impl_4x16_transform!(
    inv_txfm_add_identity_identity_4x16_8bpc_avx2_inner,
    identity4_1d,
    identity16_1d
);
impl_16x4_transform!(
    inv_txfm_add_identity_identity_16x4_8bpc_avx2_inner,
    identity16_1d,
    identity4_1d
);
impl_4x16_ffi_wrapper!(
    inv_txfm_add_identity_identity_4x16_8bpc_avx2,
    inv_txfm_add_identity_identity_4x16_8bpc_avx2_inner
);
impl_16x4_ffi_wrapper!(
    inv_txfm_add_identity_identity_16x4_8bpc_avx2,
    inv_txfm_add_identity_identity_16x4_8bpc_avx2_inner
);

// H_DCT and V_DCT for 4x16
impl_4x16_transform!(
    inv_txfm_add_identity_dct_4x16_8bpc_avx2_inner,
    identity4_1d,
    dct16_1d
);
impl_4x16_transform!(
    inv_txfm_add_dct_identity_4x16_8bpc_avx2_inner,
    dct4_1d,
    identity16_1d
);
impl_4x16_ffi_wrapper!(
    inv_txfm_add_identity_dct_4x16_8bpc_avx2,
    inv_txfm_add_identity_dct_4x16_8bpc_avx2_inner
);
impl_4x16_ffi_wrapper!(
    inv_txfm_add_dct_identity_4x16_8bpc_avx2,
    inv_txfm_add_dct_identity_4x16_8bpc_avx2_inner
);

// H_DCT and V_DCT for 16x4
impl_16x4_transform!(
    inv_txfm_add_identity_dct_16x4_8bpc_avx2_inner,
    identity16_1d,
    dct4_1d
);
impl_16x4_transform!(
    inv_txfm_add_dct_identity_16x4_8bpc_avx2_inner,
    dct16_1d,
    identity4_1d
);
impl_16x4_ffi_wrapper!(
    inv_txfm_add_identity_dct_16x4_8bpc_avx2,
    inv_txfm_add_identity_dct_16x4_8bpc_avx2_inner
);
impl_16x4_ffi_wrapper!(
    inv_txfm_add_dct_identity_16x4_8bpc_avx2,
    inv_txfm_add_dct_identity_16x4_8bpc_avx2_inner
);

// H_ADST, V_ADST, H_FLIPADST, V_FLIPADST for 4x16
impl_4x16_transform!(
    inv_txfm_add_identity_adst_4x16_8bpc_avx2_inner,
    identity4_1d,
    adst16_1d
);
impl_4x16_transform!(
    inv_txfm_add_adst_identity_4x16_8bpc_avx2_inner,
    adst4_1d,
    identity16_1d
);
impl_4x16_transform!(
    inv_txfm_add_identity_flipadst_4x16_8bpc_avx2_inner,
    identity4_1d,
    flipadst16_1d
);
impl_4x16_transform!(
    inv_txfm_add_flipadst_identity_4x16_8bpc_avx2_inner,
    flipadst4_1d,
    identity16_1d
);
impl_4x16_ffi_wrapper!(
    inv_txfm_add_identity_adst_4x16_8bpc_avx2,
    inv_txfm_add_identity_adst_4x16_8bpc_avx2_inner
);
impl_4x16_ffi_wrapper!(
    inv_txfm_add_adst_identity_4x16_8bpc_avx2,
    inv_txfm_add_adst_identity_4x16_8bpc_avx2_inner
);
impl_4x16_ffi_wrapper!(
    inv_txfm_add_identity_flipadst_4x16_8bpc_avx2,
    inv_txfm_add_identity_flipadst_4x16_8bpc_avx2_inner
);
impl_4x16_ffi_wrapper!(
    inv_txfm_add_flipadst_identity_4x16_8bpc_avx2,
    inv_txfm_add_flipadst_identity_4x16_8bpc_avx2_inner
);

// H_ADST, V_ADST, H_FLIPADST, V_FLIPADST for 16x4
impl_16x4_transform!(
    inv_txfm_add_identity_adst_16x4_8bpc_avx2_inner,
    identity16_1d,
    adst4_1d
);
impl_16x4_transform!(
    inv_txfm_add_adst_identity_16x4_8bpc_avx2_inner,
    adst16_1d,
    identity4_1d
);
impl_16x4_transform!(
    inv_txfm_add_identity_flipadst_16x4_8bpc_avx2_inner,
    identity16_1d,
    flipadst4_1d
);
impl_16x4_transform!(
    inv_txfm_add_flipadst_identity_16x4_8bpc_avx2_inner,
    flipadst16_1d,
    identity4_1d
);
impl_16x4_ffi_wrapper!(
    inv_txfm_add_identity_adst_16x4_8bpc_avx2,
    inv_txfm_add_identity_adst_16x4_8bpc_avx2_inner
);
impl_16x4_ffi_wrapper!(
    inv_txfm_add_adst_identity_16x4_8bpc_avx2,
    inv_txfm_add_adst_identity_16x4_8bpc_avx2_inner
);
impl_16x4_ffi_wrapper!(
    inv_txfm_add_identity_flipadst_16x4_8bpc_avx2,
    inv_txfm_add_identity_flipadst_16x4_8bpc_avx2_inner
);
impl_16x4_ffi_wrapper!(
    inv_txfm_add_flipadst_identity_16x4_8bpc_avx2,
    inv_txfm_add_flipadst_identity_16x4_8bpc_avx2_inner
);

/// Full 2D DCT_DCT 8x32 inverse transform
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_8x32_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    // W=8, H=32, 4:1 ratio
    let row_clip_min = i16::MIN as i32;
    let row_clip_max = i16::MAX as i32;
    let col_clip_min = i16::MIN as i32;
    let col_clip_max = i16::MAX as i32;
    let mut tmp = [0i32; 8 * 32];

    // SIMD row transform: 4 batches of 8 rows. No rect2, shift=2, rnd=2.
    {
        let coeff_slice = coeff.as_slice();
        for y_base in [0usize, 8, 16, 24] {
            simd_row_dct8_8bpc_8rows(
                _token,
                coeff_slice,
                32,
                y_base,
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
    }

    // Column transform: SIMD across 8 columns, 32 rows
    {
        let min_v = _mm256_set1_epi32(col_clip_min);
        let max_v = _mm256_set1_epi32(col_clip_max);
        let mut v = [_mm256_setzero_si256(); 32];
        for i in 0..32 {
            v[i] = loadu_256!(&tmp[i * 8..i * 8 + 8], [i32; 8]);
        }
        dct32_1d_cols8_i16(_token, &mut v, min_v, max_v);
        for i in 0..32 {
            storeu_256!(&mut tmp[i * 8..i * 8 + 8], [i32; 8], v[i]);
        }
    }

    // Add to destination
    for y in 0..32 {
        let dst_off = y * dst_stride;
        for x in 0..8 {
            let d = dst[dst_off + x] as i32;
            let c = (tmp[y * 8 + x] + 8) >> 4;
            let result = iclip(d + c, 0, bitdepth_max);
            dst[dst_off + x] = result as u8;
        }
    }

    // Clear coefficients
    coeff[..256].fill(0);
}

/// FFI wrapper for 8x32 DCT_DCT 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_8x32_8bpc_avx2(
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

    inv_txfm_add_dct_dct_8x32_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// Full 2D DCT_DCT 32x8 inverse transform
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_32x8_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    // W=32, H=8, 4:1 ratio
    let row_clip_min = i16::MIN as i32;
    let row_clip_max = i16::MAX as i32;
    let col_clip_min = i16::MIN as i32;
    let col_clip_max = i16::MAX as i32;
    let mut tmp = [0i32; 32 * 8];

    // SIMD row transform: 1 batch of 8 rows (h=8). No rect2, shift=2, rnd=2.
    {
        let coeff_slice = coeff.as_slice();
        simd_row_dct32_8bpc_8rows(
            _token,
            coeff_slice,
            8,
            0,
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

    // Column transform: SIMD across 32 columns (4 chunks of 8), 8 rows
    {
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
        add_to_dst_8bpc_avx512(t512, &mut *dst, dst_stride, &tmp, 32, 32, 8, bitdepth_max);
        coeff[..256].fill(0);
        return;
    }

    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi16(bitdepth_max as i16);
    let rnd_final = _mm256_set1_epi32(8);

    for y in 0..8 {
        let dst_off = y * dst_stride;

        // Process 32 pixels in two 16-pixel chunks
        for chunk in 0..2 {
            let chunk_off = chunk * 16;
            let d = loadu_128!(
                <&[u8; 16]>::try_from(&dst[dst_off + chunk_off..dst_off + chunk_off + 16]).unwrap()
            );
            let d16 = _mm256_cvtepu8_epi16(d);

            let c0 = _mm256_set_epi32(
                tmp[y * 32 + chunk_off + 7],
                tmp[y * 32 + chunk_off + 6],
                tmp[y * 32 + chunk_off + 5],
                tmp[y * 32 + chunk_off + 4],
                tmp[y * 32 + chunk_off + 3],
                tmp[y * 32 + chunk_off + 2],
                tmp[y * 32 + chunk_off + 1],
                tmp[y * 32 + chunk_off + 0],
            );
            let c1 = _mm256_set_epi32(
                tmp[y * 32 + chunk_off + 15],
                tmp[y * 32 + chunk_off + 14],
                tmp[y * 32 + chunk_off + 13],
                tmp[y * 32 + chunk_off + 12],
                tmp[y * 32 + chunk_off + 11],
                tmp[y * 32 + chunk_off + 10],
                tmp[y * 32 + chunk_off + 9],
                tmp[y * 32 + chunk_off + 8],
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
                <&mut [u8; 16]>::try_from(&mut dst[dst_off + chunk_off..dst_off + chunk_off + 16])
                    .unwrap(),
                _mm256_castsi256_si128(packed)
            );
        }
    }

    // Clear coefficients
    coeff[..256].fill(0);
}

/// FFI wrapper for 32x8 DCT_DCT 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_32x8_8bpc_avx2(
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

    inv_txfm_add_dct_dct_32x8_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// 8x32 and 32x8 IDTX (identity_identity) transforms
// ============================================================================

/// 8x32 IDTX inverse transform
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_identity_identity_8x32_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let clip_min = i16::MIN as i32;
    let clip_max = i16::MAX as i32;
    let mut tmp = [0i32; 8 * 32];

    // Row transform (8 elements each, 32 rows), shift=2 for 8x32
    let rnd = 2;
    let shift = 2;
    for y in 0..32 {
        let mut scratch = [0i32; 8];
        for x in 0..8 {
            scratch[x] = coeff[y + x * 32] as i32;
        }
        identity8_1d(&mut scratch[..8], 1, clip_min, clip_max);
        for x in 0..8 {
            tmp[y * 8 + x] = iclip((scratch[x] + rnd) >> shift, clip_min, clip_max);
        }
    }

    // Column transform: SIMD across 8 columns, 32 rows. identity32 = *4
    {
        for i in 0..32 {
            let v = loadu_256!(&tmp[i * 8..i * 8 + 8], [i32; 8]);
            let result = _mm256_slli_epi32::<2>(v);
            storeu_256!(&mut tmp[i * 8..i * 8 + 8], [i32; 8], result);
        }
    }

    // Add to destination
    for y in 0..32 {
        let dst_off = y * dst_stride;
        for x in 0..8 {
            let d = dst[dst_off + x] as i32;
            let c = (tmp[y * 8 + x] + 8) >> 4;
            let result = iclip(d + c, 0, bitdepth_max);
            dst[dst_off + x] = result as u8;
        }
    }

    // Clear coefficients
    coeff[..256].fill(0);
}

/// FFI wrapper for 8x32 IDTX 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_identity_8x32_8bpc_avx2(
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

    inv_txfm_add_identity_identity_8x32_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// 32x8 IDTX inverse transform
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_identity_identity_32x8_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let clip_min = i16::MIN as i32;
    let clip_max = i16::MAX as i32;
    let mut tmp = [0i32; 32 * 8];

    // Row transform (32 elements each, 8 rows), shift=2 for 32x8
    let rnd = 2;
    let shift = 2;
    for y in 0..8 {
        let mut scratch = [0i32; 32];
        for x in 0..32 {
            scratch[x] = coeff[y + x * 8] as i32;
        }
        identity32_1d(&mut scratch[..32], 1, clip_min, clip_max);
        for x in 0..32 {
            tmp[y * 32 + x] = iclip((scratch[x] + rnd) >> shift, clip_min, clip_max);
        }
    }

    // Column transform: SIMD across 32 columns (4 chunks of 8), 8 rows. identity8 = *2
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
        add_to_dst_8bpc_avx512(t512, &mut *dst, dst_stride, &tmp, 32, 32, 8, bitdepth_max);
        coeff[..256].fill(0);
        return;
    }

    for y in 0..8 {
        let dst_off = y * dst_stride;
        for x in 0..32 {
            let d = dst[dst_off + x] as i32;
            let c = (tmp[y * 32 + x] + 8) >> 4;
            let result = iclip(d + c, 0, bitdepth_max);
            dst[dst_off + x] = result as u8;
        }
    }

    // Clear coefficients
    coeff[..256].fill(0);
}

/// FFI wrapper for 32x8 IDTX 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_identity_32x8_8bpc_avx2(
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

    inv_txfm_add_identity_identity_32x8_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// RECTANGULAR TRANSFORMS - 16x64, 64x16
// ============================================================================

/// Full 2D DCT_DCT 16x64 inverse transform
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_16x64_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    // W=16, H=64, 4:1 ratio
    let row_clip_min = i16::MIN as i32;
    let row_clip_max = i16::MAX as i32;
    let col_clip_min = i16::MIN as i32;
    let col_clip_max = i16::MAX as i32;
    let mut tmp = [0i32; 16 * 64];

    // SIMD row transform: 4 batches of 8 rows. No rect2, shift=2, rnd=2.
    {
        let coeff_slice = coeff.as_slice();
        for y_base in [0usize, 8, 16, 24] {
            simd_row_dct16_8bpc_8rows(
                _token,
                coeff_slice,
                32,
                y_base,
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
    }

    // Zero remaining rows
    for y in 32..64 {
        for x in 0..16 {
            tmp[y * 16 + x] = 0;
        }
    }

    // Column transform (64 elements each, 16 columns)
    for x in 0..16 {
        dct64_1d(&mut tmp[x..], 16, col_clip_min, col_clip_max);
    }

    // Add to destination
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi16(bitdepth_max as i16);
    let rnd_final = _mm256_set1_epi32(8);

    for y in 0..64 {
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
    coeff[..512].fill(0);
}

/// FFI wrapper for 16x64 DCT_DCT 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_16x64_8bpc_avx2(
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

    inv_txfm_add_dct_dct_16x64_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// Full 2D DCT_DCT 64x16 inverse transform
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_64x16_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    // W=64, H=16, 4:1 ratio
    let row_clip_min = i16::MIN as i32;
    let row_clip_max = i16::MAX as i32;
    let col_clip_min = i16::MIN as i32;
    let col_clip_max = i16::MAX as i32;
    let mut tmp = [0i32; 64 * 16];

    // rect4 scaling
    // Row transform (64 elements each, 16 rows) - only first 32 columns have coefficients
    let rnd = 2;
    let shift = 2;
    for y in 0..16 {
        let mut scratch = [0i32; 64];
        for x in 0..32 {
            scratch[x] = coeff[y + x * 16] as i32;
        }
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
        add_to_dst_8bpc_avx512(t512, &mut *dst, dst_stride, &tmp, 64, 64, 16, bitdepth_max);
        coeff[..512].fill(0);
        return;
    }

    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi16(bitdepth_max as i16);
    let rnd_final = _mm256_set1_epi32(8);

    for y in 0..16 {
        let dst_off = y * dst_stride;

        // Process 64 pixels in four 16-pixel chunks
        for chunk in 0..4 {
            let chunk_off = chunk * 16;
            let d = loadu_128!(
                <&[u8; 16]>::try_from(&dst[dst_off + chunk_off..dst_off + chunk_off + 16]).unwrap()
            );
            let d16 = _mm256_cvtepu8_epi16(d);

            let c0 = _mm256_set_epi32(
                tmp[y * 64 + chunk_off + 7],
                tmp[y * 64 + chunk_off + 6],
                tmp[y * 64 + chunk_off + 5],
                tmp[y * 64 + chunk_off + 4],
                tmp[y * 64 + chunk_off + 3],
                tmp[y * 64 + chunk_off + 2],
                tmp[y * 64 + chunk_off + 1],
                tmp[y * 64 + chunk_off + 0],
            );
            let c1 = _mm256_set_epi32(
                tmp[y * 64 + chunk_off + 15],
                tmp[y * 64 + chunk_off + 14],
                tmp[y * 64 + chunk_off + 13],
                tmp[y * 64 + chunk_off + 12],
                tmp[y * 64 + chunk_off + 11],
                tmp[y * 64 + chunk_off + 10],
                tmp[y * 64 + chunk_off + 9],
                tmp[y * 64 + chunk_off + 8],
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
                <&mut [u8; 16]>::try_from(&mut dst[dst_off + chunk_off..dst_off + chunk_off + 16])
                    .unwrap(),
                _mm256_castsi256_si128(packed)
            );
        }
    }

    // Clear coefficients
    coeff[..512].fill(0);
}

/// FFI wrapper for 64x16 DCT_DCT 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_64x16_8bpc_avx2(
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

    inv_txfm_add_dct_dct_64x16_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

