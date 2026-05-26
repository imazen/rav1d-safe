// ============================================================================
// 32x32 DCT TRANSFORMS
// ============================================================================

/// DCT32 1D transform (in-place)
#[inline]
fn dct32_1d(c: &mut [i32], stride: usize, min: i32, max: i32) {
    let clip = |v: i32| v.clamp(min, max);

    // First apply DCT16 to even positions
    dct16_1d(c, stride * 2, min, max);

    let in1 = c[1 * stride];
    let in3 = c[3 * stride];
    let in5 = c[5 * stride];
    let in7 = c[7 * stride];
    let in9 = c[9 * stride];
    let in11 = c[11 * stride];
    let in13 = c[13 * stride];
    let in15 = c[15 * stride];
    let in17 = c[17 * stride];
    let in19 = c[19 * stride];
    let in21 = c[21 * stride];
    let in23 = c[23 * stride];
    let in25 = c[25 * stride];
    let in27 = c[27 * stride];
    let in29 = c[29 * stride];
    let in31 = c[31 * stride];

    let t16a = ((in1 * 201 - in31 * (4091 - 4096) + 2048) >> 12) - in31;
    let t17a = ((in17 * (3035 - 4096) - in15 * 2751 + 2048) >> 12) + in17;
    let t18a = ((in9 * 1751 - in23 * (3703 - 4096) + 2048) >> 12) - in23;
    let t19a = ((in25 * (3857 - 4096) - in7 * 1380 + 2048) >> 12) + in25;
    let t20a = ((in5 * 995 - in27 * (3973 - 4096) + 2048) >> 12) - in27;
    let t21a = ((in21 * (3513 - 4096) - in11 * 2106 + 2048) >> 12) + in21;
    let t22a = (in13 * 1220 - in19 * 1645 + 1024) >> 11;
    let t23a = ((in29 * (4052 - 4096) - in3 * 601 + 2048) >> 12) + in29;
    let t24a = ((in29 * 601 + in3 * (4052 - 4096) + 2048) >> 12) + in3;
    let t25a = (in13 * 1645 + in19 * 1220 + 1024) >> 11;
    let t26a = ((in21 * 2106 + in11 * (3513 - 4096) + 2048) >> 12) + in11;
    let t27a = ((in5 * (3973 - 4096) + in27 * 995 + 2048) >> 12) + in5;
    let t28a = ((in25 * 1380 + in7 * (3857 - 4096) + 2048) >> 12) + in7;
    let t29a = ((in9 * (3703 - 4096) + in23 * 1751 + 2048) >> 12) + in9;
    let t30a = ((in17 * 2751 + in15 * (3035 - 4096) + 2048) >> 12) + in15;
    let t31a = ((in1 * (4091 - 4096) + in31 * 201 + 2048) >> 12) + in1;

    let mut t16 = clip(t16a + t17a);
    let mut t17 = clip(t16a - t17a);
    let mut t18 = clip(t19a - t18a);
    let t19 = clip(t19a + t18a);
    let t20 = clip(t20a + t21a);
    let mut t21 = clip(t20a - t21a);
    let mut t22 = clip(t23a - t22a);
    let mut t23 = clip(t23a + t22a);
    let mut t24 = clip(t24a + t25a);
    let mut t25 = clip(t24a - t25a);
    let mut t26 = clip(t27a - t26a);
    let t27 = clip(t27a + t26a);
    let t28 = clip(t28a + t29a);
    let mut t29 = clip(t28a - t29a);
    let mut t30 = clip(t31a - t30a);
    let mut t31 = clip(t31a + t30a);

    let t17a = ((t30 * 799 - t17 * (4017 - 4096) + 2048) >> 12) - t17;
    let t30a = ((t30 * (4017 - 4096) + t17 * 799 + 2048) >> 12) + t30;
    let t18a = ((-(t29 * (4017 - 4096) + t18 * 799) + 2048) >> 12) - t29;
    let t29a = ((t29 * 799 - t18 * (4017 - 4096) + 2048) >> 12) - t18;
    let t21a = (t26 * 1703 - t21 * 1138 + 1024) >> 11;
    let t26a = (t26 * 1138 + t21 * 1703 + 1024) >> 11;
    let t22a = (-(t25 * 1138 + t22 * 1703) + 1024) >> 11;
    let t25a = (t25 * 1703 - t22 * 1138 + 1024) >> 11;

    let t16a = clip(t16 + t19);
    t17 = clip(t17a + t18a);
    t18 = clip(t17a - t18a);
    let t19a = clip(t16 - t19);
    let t20a = clip(t23 - t20);
    t21 = clip(t22a - t21a);
    t22 = clip(t22a + t21a);
    let t23a = clip(t23 + t20);
    let t24a = clip(t24 + t27);
    t25 = clip(t25a + t26a);
    t26 = clip(t25a - t26a);
    let t27a = clip(t24 - t27);
    let t28a = clip(t31 - t28);
    t29 = clip(t30a - t29a);
    t30 = clip(t30a + t29a);
    let t31a = clip(t31 + t28);

    let t18a = ((t29 * 1567 - t18 * (3784 - 4096) + 2048) >> 12) - t18;
    let t29a = ((t29 * (3784 - 4096) + t18 * 1567 + 2048) >> 12) + t29;
    let t19 = ((t28a * 1567 - t19a * (3784 - 4096) + 2048) >> 12) - t19a;
    let t28 = ((t28a * (3784 - 4096) + t19a * 1567 + 2048) >> 12) + t28a;
    let t20 = ((-(t27a * (3784 - 4096) + t20a * 1567) + 2048) >> 12) - t27a;
    let t27 = ((t27a * 1567 - t20a * (3784 - 4096) + 2048) >> 12) - t20a;
    let t21a = ((-(t26 * (3784 - 4096) + t21 * 1567) + 2048) >> 12) - t26;
    let t26a = ((t26 * 1567 - t21 * (3784 - 4096) + 2048) >> 12) - t21;

    t16 = clip(t16a + t23a);
    let t17a = clip(t17 + t22);
    t18 = clip(t18a + t21a);
    let t19a = clip(t19 + t20);
    let t20a = clip(t19 - t20);
    t21 = clip(t18a - t21a);
    let t22a = clip(t17 - t22);
    t23 = clip(t16a - t23a);
    t24 = clip(t31a - t24a);
    let t25a = clip(t30 - t25);
    t26 = clip(t29a - t26a);
    let t27a = clip(t28 - t27);
    let t28a = clip(t28 + t27);
    t29 = clip(t29a + t26a);
    let t30a = clip(t30 + t25);
    t31 = clip(t31a + t24a);

    let t20_final = ((t27a - t20a) * 181 + 128) >> 8;
    let t27_final = ((t27a + t20a) * 181 + 128) >> 8;
    let t21a_final = ((t26 - t21) * 181 + 128) >> 8;
    let t26a_final = ((t26 + t21) * 181 + 128) >> 8;
    let t22_final = ((t25a - t22a) * 181 + 128) >> 8;
    let t25_final = ((t25a + t22a) * 181 + 128) >> 8;
    let t23a = ((t24 - t23) * 181 + 128) >> 8;
    let t24a = ((t24 + t23) * 181 + 128) >> 8;

    let t0 = c[0 * stride];
    let t1 = c[2 * stride];
    let t2 = c[4 * stride];
    let t3 = c[6 * stride];
    let t4 = c[8 * stride];
    let t5 = c[10 * stride];
    let t6 = c[12 * stride];
    let t7 = c[14 * stride];
    let t8 = c[16 * stride];
    let t9 = c[18 * stride];
    let t10 = c[20 * stride];
    let t11 = c[22 * stride];
    let t12 = c[24 * stride];
    let t13 = c[26 * stride];
    let t14 = c[28 * stride];
    let t15 = c[30 * stride];

    c[0 * stride] = clip(t0 + t31);
    c[1 * stride] = clip(t1 + t30a);
    c[2 * stride] = clip(t2 + t29);
    c[3 * stride] = clip(t3 + t28a);
    c[4 * stride] = clip(t4 + t27_final);
    c[5 * stride] = clip(t5 + t26a_final);
    c[6 * stride] = clip(t6 + t25_final);
    c[7 * stride] = clip(t7 + t24a);
    c[8 * stride] = clip(t8 + t23a);
    c[9 * stride] = clip(t9 + t22_final);
    c[10 * stride] = clip(t10 + t21a_final);
    c[11 * stride] = clip(t11 + t20_final);
    c[12 * stride] = clip(t12 + t19a);
    c[13 * stride] = clip(t13 + t18);
    c[14 * stride] = clip(t14 + t17a);
    c[15 * stride] = clip(t15 + t16);
    c[16 * stride] = clip(t15 - t16);
    c[17 * stride] = clip(t14 - t17a);
    c[18 * stride] = clip(t13 - t18);
    c[19 * stride] = clip(t12 - t19a);
    c[20 * stride] = clip(t11 - t20_final);
    c[21 * stride] = clip(t10 - t21a_final);
    c[22 * stride] = clip(t9 - t22_final);
    c[23 * stride] = clip(t8 - t23a);
    c[24 * stride] = clip(t7 - t24a);
    c[25 * stride] = clip(t6 - t25_final);
    c[26 * stride] = clip(t5 - t26a_final);
    c[27 * stride] = clip(t4 - t27_final);
    c[28 * stride] = clip(t3 - t28a);
    c[29 * stride] = clip(t2 - t29);
    c[30 * stride] = clip(t1 - t30a);
    c[31 * stride] = clip(t0 - t31);
}

/// Identity32 1D transform (in-place)
#[inline]
fn identity32_1d(c: &mut [i32], stride: usize, _min: i32, _max: i32) {
    // For 32x32 identity: out = in * 4
    for i in 0..32 {
        c[i * stride] *= 4;
    }
}

/// Generic 32x32 transform function
#[inline]
fn inv_txfm_32x32_inner<C: Copy + Into<i32>>(
    tmp: &mut [i32; 1024],
    coeff: &[C],
    row_transform: fn(&mut [i32], usize, i32, i32),
    col_transform: fn(&mut [i32], usize, i32, i32),
    row_clip_min: i32,
    row_clip_max: i32,
    col_clip_min: i32,
    col_clip_max: i32,
) {
    // For 32x32: row_shift = 2, col_shift = 4 (total 6)
    let rnd = 2;
    let shift = 2;

    // Row transform
    for y in 0..32 {
        // Load row from column-major
        let mut scratch = [0i32; 32];
        for x in 0..32 {
            scratch[x] = coeff[y + x * 32].into();
        }
        row_transform(&mut scratch[..32], 1, row_clip_min, row_clip_max);
        // Apply intermediate shift and store row-major
        for x in 0..32 {
            tmp[y * 32 + x] = ((scratch[x] + rnd) >> shift).clamp(col_clip_min, col_clip_max);
        }
    }

    // Column transform (in-place, row-major with stride 32)
    for x in 0..32 {
        col_transform(&mut tmp[x..], 32, col_clip_min, col_clip_max);
    }
}

/// AVX-512 helper: add transformed i32 coefficients to 8bpc destination.
/// Processes 32 pixels per chunk using 512-bit registers.
/// Used for w>=32 transforms (32x32, 64x64, 32x64, 64x32, 64x16).
#[cfg(target_arch = "x86_64")]
#[arcane]
fn add_to_dst_8bpc_avx512(
    _token: Server64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp: &[i32],
    tmp_stride: usize,
    w: usize,
    h: usize,
    _bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let zero_512 = _mm512_setzero_si512();
    let max_val_512 = _mm512_set1_epi16(255);
    let rnd_final_512 = _mm512_set1_epi32(8);

    for y in 0..h {
        let dst_off = y * dst_stride;
        let mut x = 0usize;

        // Process 32 pixels at a time
        while x + 32 <= w {
            // Load 32 u8 dest pixels → 32 i16
            let d = loadu_256!(&dst[dst_off + x..dst_off + x + 32], [u8; 32]);
            let d16 = _mm512_cvtepu8_epi16(d);

            // Load 32 consecutive i32 coefficients as two __m512i
            let c0 = loadu_512!(&tmp[y * tmp_stride + x..y * tmp_stride + x + 16], [i32; 16]);
            let c1 = loadu_512!(
                &tmp[y * tmp_stride + x + 16..y * tmp_stride + x + 32],
                [i32; 16]
            );

            // Scale: (c + 8) >> 4
            let c0_scaled = _mm512_srai_epi32::<4>(_mm512_add_epi32(c0, rnd_final_512));
            let c1_scaled = _mm512_srai_epi32::<4>(_mm512_add_epi32(c1, rnd_final_512));

            // Pack i32 → i16 using cvtsepi32_epi16 (no lane-crossing issues!)
            let c16_lo = _mm512_cvtsepi32_epi16(c0_scaled); // 16 i16 as __m256i
            let c16_hi = _mm512_cvtsepi32_epi16(c1_scaled); // 16 i16 as __m256i

            // Combine two __m256i → one __m512i of 32 i16
            let c16 = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(c16_lo), c16_hi);

            // Add dest + coeff
            let sum = _mm512_add_epi16(d16, c16);

            // Clamp to [0, 255]
            let clamped = _mm512_max_epi16(_mm512_min_epi16(sum, max_val_512), zero_512);

            // Pack i16 → u8 using unsigned saturation (no lane issues!)
            let packed = _mm512_cvtusepi16_epi8(clamped); // 32 u8 as __m256i

            // Store 32 bytes
            storeu_256!(&mut dst[dst_off + x..dst_off + x + 32], [u8; 32], packed);

            x += 32;
        }

        // AVX2 tail for remaining 16-pixel chunk (when w is not multiple of 32)
        if x + 16 <= w {
            let d = loadu_128!(&dst[dst_off + x..dst_off + x + 16], [u8; 16]);
            let d16 = _mm256_cvtepu8_epi16(d);

            let c0 = loadu_256!(&tmp[y * tmp_stride + x..y * tmp_stride + x + 8], [i32; 8]);
            let c1 = loadu_256!(
                &tmp[y * tmp_stride + x + 8..y * tmp_stride + x + 16],
                [i32; 8]
            );

            let rnd = _mm256_set1_epi32(8);
            let c0_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c0, rnd));
            let c1_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c1, rnd));

            let c16 = _mm256_packs_epi32(c0_scaled, c1_scaled);
            let c16 = _mm256_permute4x64_epi64::<0b11_01_10_00>(c16);

            let sum = _mm256_add_epi16(d16, c16);
            let zero = _mm256_setzero_si256();
            let max_val = _mm256_set1_epi16(255);
            let clamped = _mm256_max_epi16(_mm256_min_epi16(sum, max_val), zero);

            let packed = _mm256_packus_epi16(clamped, clamped);
            let packed = _mm256_permute4x64_epi64::<0b11_01_10_00>(packed);

            storeu_128!(
                &mut dst[dst_off + x..dst_off + x + 16],
                [u8; 16],
                _mm256_castsi256_si128(packed)
            );
        }
    }
}

/// AVX-512 helper: add transformed i32 coefficients to 16bpc destination.
/// Processes 16 pixels per chunk using 512-bit registers.
/// Used for w>=16 transforms (32x32, 64x64, 32x64, 64x32, 16x64, 64x16).
#[cfg(target_arch = "x86_64")]
#[arcane]
fn add_to_dst_16bpc_avx512(
    _token: Server64,
    dst: &mut [u16],
    dst_stride_u16: usize,
    tmp: &[i32],
    tmp_stride: usize,
    w: usize,
    h: usize,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let zero_512 = _mm512_setzero_si512();
    let max_val_512 = _mm512_set1_epi32(bitdepth_max);
    let rnd_final_512 = _mm512_set1_epi32(8);

    for y in 0..h {
        let dst_off = y * dst_stride_u16;
        let mut x = 0usize;

        // Process 16 pixels at a time
        while x + 16 <= w {
            // Load 16 u16 dest pixels → 16 i32
            let d = loadu_256!(&dst[dst_off + x..dst_off + x + 16], [u16; 16]);
            let d32 = _mm512_cvtepu16_epi32(d);

            // Load 16 consecutive i32 coefficients
            let c = loadu_512!(&tmp[y * tmp_stride + x..y * tmp_stride + x + 16], [i32; 16]);

            // Scale: (c + 8) >> 4
            let c_scaled = _mm512_srai_epi32::<4>(_mm512_add_epi32(c, rnd_final_512));

            // Add to destination
            let sum = _mm512_add_epi32(d32, c_scaled);

            // Clamp to [0, bitdepth_max]
            let clamped = _mm512_max_epi32(_mm512_min_epi32(sum, max_val_512), zero_512);

            // Pack i32 → u16 (no lane-crossing issues!)
            let packed = _mm512_cvtusepi32_epi16(clamped); // 16 u16 as __m256i

            // Store 16 u16
            storeu_256!(&mut dst[dst_off + x..dst_off + x + 16], [u16; 16], packed);

            x += 16;
        }

        // AVX2 tail for remaining 8-pixel chunk
        if x + 8 <= w {
            let d = loadu_128!(<&[u16; 8]>::try_from(&dst[dst_off + x..dst_off + x + 8]).unwrap());
            let d_lo = _mm_unpacklo_epi16(d, _mm_setzero_si128());
            let d_hi = _mm_unpackhi_epi16(d, _mm_setzero_si128());
            let d32 = _mm256_set_m128i(d_hi, d_lo);

            let c = loadu_256!(&tmp[y * tmp_stride + x..y * tmp_stride + x + 8], [i32; 8]);

            let rnd = _mm256_set1_epi32(8);
            let c_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c, rnd));
            let sum = _mm256_add_epi32(d32, c_scaled);

            let zero = _mm256_setzero_si256();
            let max_val = _mm256_set1_epi32(bitdepth_max);
            let clamped = _mm256_max_epi32(_mm256_min_epi32(sum, max_val), zero);

            let lo = _mm256_castsi256_si128(clamped);
            let hi = _mm256_extracti128_si256(clamped, 1);
            let packed = _mm_packus_epi32(lo, hi);
            storeu_128!(
                <&mut [u16; 8]>::try_from(&mut dst[dst_off + x..dst_off + x + 8]).unwrap(),
                packed
            );
        }
    }
}

/// Add transformed coefficients to destination with SIMD (32x32)
/// `#[rite]` so it inlines into matching-feature `#[arcane]` callers (zero call cost).
#[cfg(target_arch = "x86_64")]
#[rite]
fn add_32x32_to_dst(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp: &[i32; 1024],
    coeff: &mut [i16],
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();

    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi16(bitdepth_max as i16);
    let rnd_final = _mm256_set1_epi32(8); // (+ 8) >> 4

    for y in 0..32 {
        let dst_off = y * dst_stride;

        // Process 32 pixels in two 16-pixel chunks
        for chunk in 0..2 {
            let x_base = chunk * 16;
            let dst_chunk_off = dst_off + x_base;

            // Load destination pixels (16 bytes)
            let d =
                loadu_128!(<&[u8; 16]>::try_from(&dst[dst_chunk_off..dst_chunk_off + 16]).unwrap());
            let d16 = _mm256_cvtepu8_epi16(d);

            // Load coefficients
            let c0 = _mm256_set_epi32(
                tmp[y * 32 + x_base + 7],
                tmp[y * 32 + x_base + 6],
                tmp[y * 32 + x_base + 5],
                tmp[y * 32 + x_base + 4],
                tmp[y * 32 + x_base + 3],
                tmp[y * 32 + x_base + 2],
                tmp[y * 32 + x_base + 1],
                tmp[y * 32 + x_base + 0],
            );
            let c1 = _mm256_set_epi32(
                tmp[y * 32 + x_base + 15],
                tmp[y * 32 + x_base + 14],
                tmp[y * 32 + x_base + 13],
                tmp[y * 32 + x_base + 12],
                tmp[y * 32 + x_base + 11],
                tmp[y * 32 + x_base + 10],
                tmp[y * 32 + x_base + 9],
                tmp[y * 32 + x_base + 8],
            );

            // Final scaling: (c + 8) >> 4
            let c0_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c0, rnd_final));
            let c1_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c1, rnd_final));

            let c16 = _mm256_packs_epi32(c0_scaled, c1_scaled);
            let c16 = _mm256_permute4x64_epi64::<0b11_01_10_00>(c16);

            let sum = _mm256_add_epi16(d16, c16);
            let clamped = _mm256_max_epi16(_mm256_min_epi16(sum, max_val), zero);

            let packed = _mm256_packus_epi16(clamped, clamped);
            let packed = _mm256_permute4x64_epi64::<0b11_01_10_00>(packed);

            storeu_128!(
                <&mut [u8; 16]>::try_from(&mut dst[dst_chunk_off..dst_chunk_off + 16]).unwrap(),
                _mm256_castsi256_si128(packed)
            );
        }
    }

    // Clear coefficients (1024 * 2 = 2048 bytes = 64 * 32 bytes)
    coeff[..1024].fill(0);
}

/// 32x32 DCT_DCT inner function
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_32x32_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let col_clip_min = i16::MIN as i32;
    let col_clip_max = i16::MAX as i32;

    // SIMD row transform via pmaddwd-based dct32_row_pass_i16_simd.
    // No rect2 for 32x32. Row clips handled internally.
    // Post-process: round+shift+clip to col range (shift=2, rnd=2).
    let raw_coeff: [i16; 1024] = {
        let s = coeff.as_slice();
        let mut arr = [0i16; 1024];
        arr.copy_from_slice(&s[..1024]);
        arr
    };
    let mut tmp = dct32_row_pass_i16_simd(_token, raw_coeff);
    {
        let rnd_v = _mm256_set1_epi32(2);
        let col_min_v = _mm256_set1_epi32(col_clip_min);
        let col_max_v = _mm256_set1_epi32(col_clip_max);
        for i in (0..1024).step_by(8) {
            let v = loadu_256!(&tmp[i..i + 8], [i32; 8]);
            let rounded = _mm256_srai_epi32::<2>(_mm256_add_epi32(v, rnd_v));
            let clamped = _mm256_max_epi32(_mm256_min_epi32(rounded, col_max_v), col_min_v);
            storeu_256!(&mut tmp[i..i + 8], [i32; 8], clamped);
        }
    }
    // SIMD column transform: 8 columns x 4 chunks
    dct32x32_cols_simd(_token, &mut tmp, col_clip_min, col_clip_max);
    if let Some(t512) = crate::src::cpu::summon_avx512() {
        add_to_dst_8bpc_avx512(t512, &mut *dst, dst_stride, &tmp, 32, 32, 32, bitdepth_max);
    } else {
        add_32x32_to_dst(
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

/// 32x32 IDTX inner function
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_identity_identity_32x32_8bpc_avx2_inner(
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
        add_to_dst_8bpc_avx512(t512, &mut *dst, dst_stride, &tmp, 32, 32, 32, bitdepth_max);
    } else {
        add_32x32_to_dst(
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

/// FFI wrapper for 32x32 DCT_DCT 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_32x32_8bpc_avx2(
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

    inv_txfm_add_dct_dct_32x32_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// FFI wrapper for 32x32 IDTX 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_identity_32x32_8bpc_avx2(
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

    inv_txfm_add_identity_identity_32x32_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// 32x32 DCT TRANSFORMS 16bpc
// ============================================================================

/// Add transformed coefficients to destination with SIMD (32x32 16bpc)
/// `#[rite]` so it inlines into matching-feature `#[arcane]` callers (zero call cost).
#[cfg(target_arch = "x86_64")]
#[rite]
fn add_32x32_to_dst_16bpc(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize, // stride in bytes
    tmp: &[i32; 1024],
    coeff: &mut [i32],
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let stride_u16 = dst_stride / 2;

    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi32(bitdepth_max);
    let rnd_final = _mm256_set1_epi32(8); // (+ 8) >> 4

    for y in 0..32 {
        let dst_off = y * stride_u16;

        // Process 32 pixels in four 8-pixel chunks (since we work with i32)
        for chunk in 0..4 {
            let x_base = chunk * 8;
            let dst_chunk_off = dst_off + x_base;

            // Load destination pixels (8 u16 = 16 bytes)
            let d =
                loadu_128!(<&[u16; 8]>::try_from(&dst[dst_chunk_off..dst_chunk_off + 8]).unwrap());
            let d_lo = _mm_unpacklo_epi16(d, _mm_setzero_si128());
            let d_hi = _mm_unpackhi_epi16(d, _mm_setzero_si128());

            // Load coefficients
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

            // Combine to 256-bit for faster processing
            let d32 = _mm256_set_m128i(d_hi, d_lo);
            let c32 = _mm256_set_m128i(c_hi, c_lo);

            // Final scaling: (c + 8) >> 4
            let c_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c32, rnd_final));

            // Add to destination
            let sum = _mm256_add_epi32(d32, c_scaled);

            // Clamp to [0, bitdepth_max]
            let clamped = _mm256_max_epi32(_mm256_min_epi32(sum, max_val), zero);

            // Pack to u16 and store
            let lo = _mm256_castsi256_si128(clamped);
            let hi = _mm256_extracti128_si256(clamped, 1);
            let packed = _mm_packus_epi32(lo, hi);
            storeu_128!(
                <&mut [u16; 8]>::try_from(&mut dst[dst_chunk_off..dst_chunk_off + 8]).unwrap(),
                packed
            );
        }
    }

    // Clear coefficients (1024 * 2 = 2048 bytes = 64 * 32 bytes)
    coeff[..1024].fill(0);
}

/// 32x32 DCT_DCT inner function for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_32x32_16bpc_avx2_inner(
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
        dct32_1d,
        // Column pass: SIMD below
        |_, _, _, _| {},
        row_clip_min,
        row_clip_max,
        col_clip_min,
        col_clip_max,
    );
    // SIMD column transform
    dct32x32_cols_simd(_token, &mut tmp, col_clip_min, col_clip_max);
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

/// FFI wrapper for 32x32 DCT_DCT 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_32x32_16bpc_avx2(
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

    inv_txfm_add_dct_dct_32x32_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// 64x64 DCT TRANSFORMS
// ============================================================================

/// DCT32 1D transform for tx64 mode (simplified coefficients for in17-in31)
#[inline]
fn dct32_1d_tx64(c: &mut [i32], stride: usize, min: i32, max: i32) {
    let clip = |v: i32| v.clamp(min, max);

    // First apply DCT16 with tx64=1 (simplified)
    dct16_1d_tx64(c, stride * 2, min, max);

    let in1 = c[1 * stride];
    let in3 = c[3 * stride];
    let in5 = c[5 * stride];
    let in7 = c[7 * stride];
    let in9 = c[9 * stride];
    let in11 = c[11 * stride];
    let in13 = c[13 * stride];
    let in15 = c[15 * stride];

    // tx64=1: simplified single-coefficient multiplications
    let t16a = (in1 * 201 + 2048) >> 12;
    let t17a = (in15 * -2751 + 2048) >> 12;
    let t18a = (in9 * 1751 + 2048) >> 12;
    let t19a = (in7 * -1380 + 2048) >> 12;
    let t20a = (in5 * 995 + 2048) >> 12;
    let t21a = (in11 * -2106 + 2048) >> 12;
    let t22a = (in13 * 2440 + 2048) >> 12;
    let t23a = (in3 * -601 + 2048) >> 12;
    let t24a = (in3 * 4052 + 2048) >> 12;
    let t25a = (in13 * 3290 + 2048) >> 12;
    let t26a = (in11 * 3513 + 2048) >> 12;
    let t27a = (in5 * 3973 + 2048) >> 12;
    let t28a = (in7 * 3857 + 2048) >> 12;
    let t29a = (in9 * 3703 + 2048) >> 12;
    let t30a = (in15 * 3035 + 2048) >> 12;
    let t31a = (in1 * 4091 + 2048) >> 12;

    let mut t16 = clip(t16a + t17a);
    let mut t17 = clip(t16a - t17a);
    let mut t18 = clip(t19a - t18a);
    let t19 = clip(t19a + t18a);
    let t20 = clip(t20a + t21a);
    let mut t21 = clip(t20a - t21a);
    let mut t22 = clip(t23a - t22a);
    let mut t23 = clip(t23a + t22a);
    let mut t24 = clip(t24a + t25a);
    let mut t25 = clip(t24a - t25a);
    let mut t26 = clip(t27a - t26a);
    let t27 = clip(t27a + t26a);
    let t28 = clip(t28a + t29a);
    let mut t29 = clip(t28a - t29a);
    let mut t30 = clip(t31a - t30a);
    let mut t31 = clip(t31a + t30a);

    let t17a = ((t30 * 799 - t17 * (4017 - 4096) + 2048) >> 12) - t17;
    let t30a = ((t30 * (4017 - 4096) + t17 * 799 + 2048) >> 12) + t30;
    let t18a = ((-(t29 * (4017 - 4096) + t18 * 799) + 2048) >> 12) - t29;
    let t29a = ((t29 * 799 - t18 * (4017 - 4096) + 2048) >> 12) - t18;
    let t21a = (t26 * 1703 - t21 * 1138 + 1024) >> 11;
    let t26a = (t26 * 1138 + t21 * 1703 + 1024) >> 11;
    let t22a = (-(t25 * 1138 + t22 * 1703) + 1024) >> 11;
    let t25a = (t25 * 1703 - t22 * 1138 + 1024) >> 11;

    let t16a = clip(t16 + t19);
    t17 = clip(t17a + t18a);
    t18 = clip(t17a - t18a);
    let t19a = clip(t16 - t19);
    let t20a = clip(t23 - t20);
    t21 = clip(t22a - t21a);
    t22 = clip(t22a + t21a);
    let t23a = clip(t23 + t20);
    let t24a = clip(t24 + t27);
    t25 = clip(t25a + t26a);
    t26 = clip(t25a - t26a);
    let t27a = clip(t24 - t27);
    let t28a = clip(t31 - t28);
    t29 = clip(t30a - t29a);
    t30 = clip(t30a + t29a);
    let t31a = clip(t31 + t28);

    let t18a = ((t29 * 1567 - t18 * (3784 - 4096) + 2048) >> 12) - t18;
    let t29a = ((t29 * (3784 - 4096) + t18 * 1567 + 2048) >> 12) + t29;
    let t19 = ((t28a * 1567 - t19a * (3784 - 4096) + 2048) >> 12) - t19a;
    let t28 = ((t28a * (3784 - 4096) + t19a * 1567 + 2048) >> 12) + t28a;
    let t20 = ((-(t27a * (3784 - 4096) + t20a * 1567) + 2048) >> 12) - t27a;
    let t27 = ((t27a * 1567 - t20a * (3784 - 4096) + 2048) >> 12) - t20a;
    let t21a = ((-(t26 * (3784 - 4096) + t21 * 1567) + 2048) >> 12) - t26;
    let t26a = ((t26 * 1567 - t21 * (3784 - 4096) + 2048) >> 12) - t21;

    t16 = clip(t16a + t23a);
    let t17a = clip(t17 + t22);
    t18 = clip(t18a + t21a);
    let t19a = clip(t19 + t20);
    let t20a = clip(t19 - t20);
    t21 = clip(t18a - t21a);
    let t22a = clip(t17 - t22);
    t23 = clip(t16a - t23a);
    t24 = clip(t31a - t24a);
    let t25a = clip(t30 - t25);
    t26 = clip(t29a - t26a);
    let t27a = clip(t28 - t27);
    let t28a = clip(t28 + t27);
    t29 = clip(t29a + t26a);
    let t30a = clip(t30 + t25);
    t31 = clip(t31a + t24a);

    let t20_final = ((t27a - t20a) * 181 + 128) >> 8;
    let t27_final = ((t27a + t20a) * 181 + 128) >> 8;
    let t21a_final = ((t26 - t21) * 181 + 128) >> 8;
    let t26a_final = ((t26 + t21) * 181 + 128) >> 8;
    let t22_final = ((t25a - t22a) * 181 + 128) >> 8;
    let t25_final = ((t25a + t22a) * 181 + 128) >> 8;
    let t23a = ((t24 - t23) * 181 + 128) >> 8;
    let t24a = ((t24 + t23) * 181 + 128) >> 8;

    let t0 = c[0 * stride];
    let t1 = c[2 * stride];
    let t2 = c[4 * stride];
    let t3 = c[6 * stride];
    let t4 = c[8 * stride];
    let t5 = c[10 * stride];
    let t6 = c[12 * stride];
    let t7 = c[14 * stride];
    let t8 = c[16 * stride];
    let t9 = c[18 * stride];
    let t10 = c[20 * stride];
    let t11 = c[22 * stride];
    let t12 = c[24 * stride];
    let t13 = c[26 * stride];
    let t14 = c[28 * stride];
    let t15 = c[30 * stride];

    c[0 * stride] = clip(t0 + t31);
    c[1 * stride] = clip(t1 + t30a);
    c[2 * stride] = clip(t2 + t29);
    c[3 * stride] = clip(t3 + t28a);
    c[4 * stride] = clip(t4 + t27_final);
    c[5 * stride] = clip(t5 + t26a_final);
    c[6 * stride] = clip(t6 + t25_final);
    c[7 * stride] = clip(t7 + t24a);
    c[8 * stride] = clip(t8 + t23a);
    c[9 * stride] = clip(t9 + t22_final);
    c[10 * stride] = clip(t10 + t21a_final);
    c[11 * stride] = clip(t11 + t20_final);
    c[12 * stride] = clip(t12 + t19a);
    c[13 * stride] = clip(t13 + t18);
    c[14 * stride] = clip(t14 + t17a);
    c[15 * stride] = clip(t15 + t16);
    c[16 * stride] = clip(t15 - t16);
    c[17 * stride] = clip(t14 - t17a);
    c[18 * stride] = clip(t13 - t18);
    c[19 * stride] = clip(t12 - t19a);
    c[20 * stride] = clip(t11 - t20_final);
    c[21 * stride] = clip(t10 - t21a_final);
    c[22 * stride] = clip(t9 - t22_final);
    c[23 * stride] = clip(t8 - t23a);
    c[24 * stride] = clip(t7 - t24a);
    c[25 * stride] = clip(t6 - t25_final);
    c[26 * stride] = clip(t5 - t26a_final);
    c[27 * stride] = clip(t4 - t27_final);
    c[28 * stride] = clip(t3 - t28a);
    c[29 * stride] = clip(t2 - t29);
    c[30 * stride] = clip(t1 - t30a);
    c[31 * stride] = clip(t0 - t31);
}

/// DCT16 1D transform for tx64 mode (simplified coefficients)
#[inline]
fn dct16_1d_tx64(c: &mut [i32], stride: usize, min: i32, max: i32) {
    let clip = |v: i32| v.clamp(min, max);

    // First apply DCT8 to even positions
    dct8_1d(c, stride * 2, min, max);

    let in1 = c[1 * stride];
    let in3 = c[3 * stride];
    let in5 = c[5 * stride];
    let in7 = c[7 * stride];

    // tx64=1: simplified single-coefficient multiplications
    let t8a = (in1 * 401 + 2048) >> 12;
    let t9a = (in7 * -2598 + 2048) >> 12;
    let t10a = (in5 * 1931 + 2048) >> 12;
    let t11a = (in3 * -1189 + 2048) >> 12;
    let t12a = (in3 * 3920 + 2048) >> 12;
    let t13a = (in5 * 3612 + 2048) >> 12;
    let t14a = (in7 * 3166 + 2048) >> 12;
    let t15a = (in1 * 4076 + 2048) >> 12;

    let t8 = clip(t8a + t9a);
    let mut t9 = clip(t8a - t9a);
    let mut t10 = clip(t11a - t10a);
    let mut t11 = clip(t11a + t10a);
    let mut t12 = clip(t12a + t13a);
    let mut t13 = clip(t12a - t13a);
    let mut t14 = clip(t15a - t14a);
    let t15 = clip(t15a + t14a);

    let t9a = ((t14 * 1567 - t9 * (3784 - 4096) + 2048) >> 12) - t9;
    let t14a = ((t14 * (3784 - 4096) + t9 * 1567 + 2048) >> 12) + t14;
    let t10a = ((-(t13 * (3784 - 4096) + t10 * 1567) + 2048) >> 12) - t13;
    let t13a = ((t13 * 1567 - t10 * (3784 - 4096) + 2048) >> 12) - t10;

    let t8a = clip(t8 + t11);
    t9 = clip(t9a + t10a);
    t10 = clip(t9a - t10a);
    let t11a = clip(t8 - t11);
    let t12a = clip(t15 - t12);
    t13 = clip(t14a - t13a);
    t14 = clip(t14a + t13a);
    let t15a = clip(t15 + t12);

    let t10a = ((t13 - t10) * 181 + 128) >> 8;
    let t13a = ((t13 + t10) * 181 + 128) >> 8;
    t11 = ((t12a - t11a) * 181 + 128) >> 8;
    t12 = ((t12a + t11a) * 181 + 128) >> 8;

    let t0 = c[0 * stride];
    let t1 = c[2 * stride];
    let t2 = c[4 * stride];
    let t3 = c[6 * stride];
    let t4 = c[8 * stride];
    let t5 = c[10 * stride];
    let t6 = c[12 * stride];
    let t7 = c[14 * stride];

    c[0 * stride] = clip(t0 + t15a);
    c[1 * stride] = clip(t1 + t14);
    c[2 * stride] = clip(t2 + t13a);
    c[3 * stride] = clip(t3 + t12);
    c[4 * stride] = clip(t4 + t11);
    c[5 * stride] = clip(t5 + t10a);
    c[6 * stride] = clip(t6 + t9);
    c[7 * stride] = clip(t7 + t8a);
    c[8 * stride] = clip(t7 - t8a);
    c[9 * stride] = clip(t6 - t9);
    c[10 * stride] = clip(t5 - t10a);
    c[11 * stride] = clip(t4 - t11);
    c[12 * stride] = clip(t3 - t12);
    c[13 * stride] = clip(t2 - t13a);
    c[14 * stride] = clip(t1 - t14);
    c[15 * stride] = clip(t0 - t15a);
}

/// DCT64 1D transform (in-place)
#[inline]
fn dct64_1d(c: &mut [i32], stride: usize, min: i32, max: i32) {
    let clip = |v: i32| v.clamp(min, max);

    // First apply DCT32 in tx64 mode to even positions
    dct32_1d_tx64(c, stride * 2, min, max);

    let in1 = c[1 * stride];
    let in3 = c[3 * stride];
    let in5 = c[5 * stride];
    let in7 = c[7 * stride];
    let in9 = c[9 * stride];
    let in11 = c[11 * stride];
    let in13 = c[13 * stride];
    let in15 = c[15 * stride];
    let in17 = c[17 * stride];
    let in19 = c[19 * stride];
    let in21 = c[21 * stride];
    let in23 = c[23 * stride];
    let in25 = c[25 * stride];
    let in27 = c[27 * stride];
    let in29 = c[29 * stride];
    let in31 = c[31 * stride];

    // tx64 simplified coefficients - only use first 32 inputs
    let mut t32a = (in1 * 101 + 2048) >> 12;
    let mut t33a = (in31 * -2824 + 2048) >> 12;
    let mut t34a = (in17 * 1660 + 2048) >> 12;
    let mut t35a = (in15 * -1474 + 2048) >> 12;
    let mut t36a = (in9 * 897 + 2048) >> 12;
    let mut t37a = (in23 * -2191 + 2048) >> 12;
    let mut t38a = (in25 * 2359 + 2048) >> 12;
    let mut t39a = (in7 * -700 + 2048) >> 12;
    let mut t40a = (in5 * 501 + 2048) >> 12;
    let mut t41a = (in27 * -2520 + 2048) >> 12;
    let mut t42a = (in21 * 2019 + 2048) >> 12;
    let mut t43a = (in11 * -1092 + 2048) >> 12;
    let mut t44a = (in13 * 1285 + 2048) >> 12;
    let mut t45a = (in19 * -1842 + 2048) >> 12;
    let mut t46a = (in29 * 2675 + 2048) >> 12;
    let mut t47a = (in3 * -301 + 2048) >> 12;
    let mut t48a = (in3 * 4085 + 2048) >> 12;
    let mut t49a = (in29 * 3102 + 2048) >> 12;
    let mut t50a = (in19 * 3659 + 2048) >> 12;
    let mut t51a = (in13 * 3889 + 2048) >> 12;
    let mut t52a = (in11 * 3948 + 2048) >> 12;
    let mut t53a = (in21 * 3564 + 2048) >> 12;
    let mut t54a = (in27 * 3229 + 2048) >> 12;
    let mut t55a = (in5 * 4065 + 2048) >> 12;
    let mut t56a = (in7 * 4036 + 2048) >> 12;
    let mut t57a = (in25 * 3349 + 2048) >> 12;
    let mut t58a = (in23 * 3461 + 2048) >> 12;
    let mut t59a = (in9 * 3996 + 2048) >> 12;
    let mut t60a = (in15 * 3822 + 2048) >> 12;
    let mut t61a = (in17 * 3745 + 2048) >> 12;
    let mut t62a = (in31 * 2967 + 2048) >> 12;
    let mut t63a = (in1 * 4095 + 2048) >> 12;

    let mut t32 = clip(t32a + t33a);
    let mut t33 = clip(t32a - t33a);
    let mut t34 = clip(t35a - t34a);
    let mut t35 = clip(t35a + t34a);
    let mut t36 = clip(t36a + t37a);
    let mut t37 = clip(t36a - t37a);
    let mut t38 = clip(t39a - t38a);
    let mut t39 = clip(t39a + t38a);
    let mut t40 = clip(t40a + t41a);
    let mut t41 = clip(t40a - t41a);
    let mut t42 = clip(t43a - t42a);
    let mut t43 = clip(t43a + t42a);
    let mut t44 = clip(t44a + t45a);
    let mut t45 = clip(t44a - t45a);
    let mut t46 = clip(t47a - t46a);
    let mut t47 = clip(t47a + t46a);
    let mut t48 = clip(t48a + t49a);
    let mut t49 = clip(t48a - t49a);
    let mut t50 = clip(t51a - t50a);
    let mut t51 = clip(t51a + t50a);
    let mut t52 = clip(t52a + t53a);
    let mut t53 = clip(t52a - t53a);
    let mut t54 = clip(t55a - t54a);
    let mut t55 = clip(t55a + t54a);
    let mut t56 = clip(t56a + t57a);
    let mut t57 = clip(t56a - t57a);
    let mut t58 = clip(t59a - t58a);
    let mut t59 = clip(t59a + t58a);
    let mut t60 = clip(t60a + t61a);
    let mut t61 = clip(t60a - t61a);
    let mut t62 = clip(t63a - t62a);
    let mut t63 = clip(t63a + t62a);

    t33a = ((t33 * (4096 - 4076) + t62 * 401 + 2048) >> 12) - t33;
    t34a = ((t34 * -401 + t61 * (4096 - 4076) + 2048) >> 12) - t61;
    t37a = (t37 * -1299 + t58 * 1583 + 1024) >> 11;
    t38a = (t38 * -1583 + t57 * -1299 + 1024) >> 11;
    t41a = ((t41 * (4096 - 3612) + t54 * 1931 + 2048) >> 12) - t41;
    t42a = ((t42 * -1931 + t53 * (4096 - 3612) + 2048) >> 12) - t53;
    t45a = ((t45 * -1189 + t50 * (3920 - 4096) + 2048) >> 12) + t50;
    t46a = ((t46 * (4096 - 3920) + t49 * -1189 + 2048) >> 12) - t46;
    t49a = ((t46 * -1189 + t49 * (3920 - 4096) + 2048) >> 12) + t49;
    t50a = ((t45 * (3920 - 4096) + t50 * 1189 + 2048) >> 12) + t45;
    t53a = ((t42 * (4096 - 3612) + t53 * 1931 + 2048) >> 12) - t42;
    t54a = ((t41 * 1931 + t54 * (3612 - 4096) + 2048) >> 12) + t54;
    t57a = (t38 * -1299 + t57 * 1583 + 1024) >> 11;
    t58a = (t37 * 1583 + t58 * 1299 + 1024) >> 11;
    t61a = ((t34 * (4096 - 4076) + t61 * 401 + 2048) >> 12) - t34;
    t62a = ((t33 * 401 + t62 * (4076 - 4096) + 2048) >> 12) + t62;

    t32a = clip(t32 + t35);
    t33 = clip(t33a + t34a);
    t34 = clip(t33a - t34a);
    t35a = clip(t32 - t35);
    t36a = clip(t39 - t36);
    t37 = clip(t38a - t37a);
    t38 = clip(t38a + t37a);
    t39a = clip(t39 + t36);
    t40a = clip(t40 + t43);
    t41 = clip(t41a + t42a);
    t42 = clip(t41a - t42a);
    t43a = clip(t40 - t43);
    t44a = clip(t47 - t44);
    t45 = clip(t46a - t45a);
    t46 = clip(t46a + t45a);
    t47a = clip(t47 + t44);
    t48a = clip(t48 + t51);
    t49 = clip(t49a + t50a);
    t50 = clip(t49a - t50a);
    t51a = clip(t48 - t51);
    t52a = clip(t55 - t52);
    t53 = clip(t54a - t53a);
    t54 = clip(t54a + t53a);
    t55a = clip(t55 + t52);
    t56a = clip(t56 + t59);
    t57 = clip(t57a + t58a);
    t58 = clip(t57a - t58a);
    t59a = clip(t56 - t59);
    t60a = clip(t63 - t60);
    t61 = clip(t62a - t61a);
    t62 = clip(t62a + t61a);
    t63a = clip(t63 + t60);

    t34a = ((t34 * (4096 - 4017) + t61 * 799 + 2048) >> 12) - t34;
    t35 = ((t35a * (4096 - 4017) + t60a * 799 + 2048) >> 12) - t35a;
    t36 = ((t36a * -799 + t59a * (4096 - 4017) + 2048) >> 12) - t59a;
    t37a = ((t37 * -799 + t58 * (4096 - 4017) + 2048) >> 12) - t58;
    t42a = (t42 * -1138 + t53 * 1703 + 1024) >> 11;
    t43 = (t43a * -1138 + t52a * 1703 + 1024) >> 11;
    t44 = (t44a * -1703 + t51a * -1138 + 1024) >> 11;
    t45a = (t45 * -1703 + t50 * -1138 + 1024) >> 11;
    t50a = (t45 * -1138 + t50 * 1703 + 1024) >> 11;
    t51 = (t44a * -1138 + t51a * 1703 + 1024) >> 11;
    t52 = (t43a * 1703 + t52a * 1138 + 1024) >> 11;
    t53a = (t42 * 1703 + t53 * 1138 + 1024) >> 11;
    t58a = ((t37 * (4096 - 4017) + t58 * 799 + 2048) >> 12) - t37;
    t59 = ((t36a * (4096 - 4017) + t59a * 799 + 2048) >> 12) - t36a;
    t60 = ((t35a * 799 + t60a * (4017 - 4096) + 2048) >> 12) + t60a;
    t61a = ((t34 * 799 + t61 * (4017 - 4096) + 2048) >> 12) + t61;

    t32 = clip(t32a + t39a);
    t33a = clip(t33 + t38);
    t34 = clip(t34a + t37a);
    t35a = clip(t35 + t36);
    t36a = clip(t35 - t36);
    t37 = clip(t34a - t37a);
    t38a = clip(t33 - t38);
    t39 = clip(t32a - t39a);
    t40 = clip(t47a - t40a);
    t41a = clip(t46 - t41);
    t42 = clip(t45a - t42a);
    t43a = clip(t44 - t43);
    t44a = clip(t44 + t43);
    t45 = clip(t45a + t42a);
    t46a = clip(t46 + t41);
    t47 = clip(t47a + t40a);
    t48 = clip(t48a + t55a);
    t49a = clip(t49 + t54);
    t50 = clip(t50a + t53a);
    t51a = clip(t51 + t52);
    t52a = clip(t51 - t52);
    t53 = clip(t50a - t53a);
    t54a = clip(t49 - t54);
    t55 = clip(t48a - t55a);
    t56 = clip(t63a - t56a);
    t57a = clip(t62 - t57);
    t58 = clip(t61a - t58a);
    t59a = clip(t60 - t59);
    t60a = clip(t60 + t59);
    t61 = clip(t61a + t58a);
    t62a = clip(t62 + t57);
    t63 = clip(t63a + t56a);

    t36 = ((t36a * (4096 - 3784) + t59a * 1567 + 2048) >> 12) - t36a;
    t37a = ((t37 * (4096 - 3784) + t58 * 1567 + 2048) >> 12) - t37;
    t38 = ((t38a * (4096 - 3784) + t57a * 1567 + 2048) >> 12) - t38a;
    t39a = ((t39 * (4096 - 3784) + t56 * 1567 + 2048) >> 12) - t39;
    t40a = ((t40 * -1567 + t55 * (4096 - 3784) + 2048) >> 12) - t55;
    t41 = ((t41a * -1567 + t54a * (4096 - 3784) + 2048) >> 12) - t54a;
    t42a = ((t42 * -1567 + t53 * (4096 - 3784) + 2048) >> 12) - t53;
    t43 = ((t43a * -1567 + t52a * (4096 - 3784) + 2048) >> 12) - t52a;
    t52 = ((t43a * (4096 - 3784) + t52a * 1567 + 2048) >> 12) - t43a;
    t53a = ((t42 * (4096 - 3784) + t53 * 1567 + 2048) >> 12) - t42;
    t54 = ((t41a * (4096 - 3784) + t54a * 1567 + 2048) >> 12) - t41a;
    t55a = ((t40 * (4096 - 3784) + t55 * 1567 + 2048) >> 12) - t40;
    t56a = ((t39 * 1567 + t56 * (3784 - 4096) + 2048) >> 12) + t56;
    t57 = ((t38a * 1567 + t57a * (3784 - 4096) + 2048) >> 12) + t57a;
    t58a = ((t37 * 1567 + t58 * (3784 - 4096) + 2048) >> 12) + t58;
    t59 = ((t36a * 1567 + t59a * (3784 - 4096) + 2048) >> 12) + t59a;

    t32a = clip(t32 + t47);
    t33 = clip(t33a + t46a);
    t34a = clip(t34 + t45);
    t35 = clip(t35a + t44a);
    t36a = clip(t36 + t43);
    t37 = clip(t37a + t42a);
    t38a = clip(t38 + t41);
    t39 = clip(t39a + t40a);
    t40 = clip(t39a - t40a);
    t41a = clip(t38 - t41);
    t42 = clip(t37a - t42a);
    t43a = clip(t36 - t43);
    t44 = clip(t35a - t44a);
    t45a = clip(t34 - t45);
    t46 = clip(t33a - t46a);
    t47a = clip(t32 - t47);
    t48a = clip(t63 - t48);
    t49 = clip(t62a - t49a);
    t50a = clip(t61 - t50);
    t51 = clip(t60a - t51a);
    t52a = clip(t59 - t52);
    t53 = clip(t58a - t53a);
    t54a = clip(t57 - t54);
    t55 = clip(t56a - t55a);
    t56 = clip(t56a + t55a);
    t57a = clip(t57 + t54);
    t58 = clip(t58a + t53a);
    t59a = clip(t59 + t52);
    t60 = clip(t60a + t51a);
    t61a = clip(t61 + t50);
    t62 = clip(t62a + t49a);
    t63a = clip(t63 + t48);

    t40a = ((t55 - t40) * 181 + 128) >> 8;
    t41 = ((t54a - t41a) * 181 + 128) >> 8;
    t42a = ((t53 - t42) * 181 + 128) >> 8;
    t43 = ((t52a - t43a) * 181 + 128) >> 8;
    t44a = ((t51 - t44) * 181 + 128) >> 8;
    t45 = ((t50a - t45a) * 181 + 128) >> 8;
    t46a = ((t49 - t46) * 181 + 128) >> 8;
    t47 = ((t48a - t47a) * 181 + 128) >> 8;
    t48 = ((t47a + t48a) * 181 + 128) >> 8;
    t49a = ((t46 + t49) * 181 + 128) >> 8;
    t50 = ((t45a + t50a) * 181 + 128) >> 8;
    t51a = ((t44 + t51) * 181 + 128) >> 8;
    t52 = ((t43a + t52a) * 181 + 128) >> 8;
    t53a = ((t42 + t53) * 181 + 128) >> 8;
    t54 = ((t41a + t54a) * 181 + 128) >> 8;
    t55a = ((t40 + t55) * 181 + 128) >> 8;

    let t0 = c[0 * stride];
    let t1 = c[2 * stride];
    let t2 = c[4 * stride];
    let t3 = c[6 * stride];
    let t4 = c[8 * stride];
    let t5 = c[10 * stride];
    let t6 = c[12 * stride];
    let t7 = c[14 * stride];
    let t8 = c[16 * stride];
    let t9 = c[18 * stride];
    let t10 = c[20 * stride];
    let t11 = c[22 * stride];
    let t12 = c[24 * stride];
    let t13 = c[26 * stride];
    let t14 = c[28 * stride];
    let t15 = c[30 * stride];
    let t16 = c[32 * stride];
    let t17 = c[34 * stride];
    let t18 = c[36 * stride];
    let t19 = c[38 * stride];
    let t20 = c[40 * stride];
    let t21 = c[42 * stride];
    let t22 = c[44 * stride];
    let t23 = c[46 * stride];
    let t24 = c[48 * stride];
    let t25 = c[50 * stride];
    let t26 = c[52 * stride];
    let t27 = c[54 * stride];
    let t28 = c[56 * stride];
    let t29 = c[58 * stride];
    let t30 = c[60 * stride];
    let t31 = c[62 * stride];

    c[0 * stride] = clip(t0 + t63a);
    c[1 * stride] = clip(t1 + t62);
    c[2 * stride] = clip(t2 + t61a);
    c[3 * stride] = clip(t3 + t60);
    c[4 * stride] = clip(t4 + t59a);
    c[5 * stride] = clip(t5 + t58);
    c[6 * stride] = clip(t6 + t57a);
    c[7 * stride] = clip(t7 + t56);
    c[8 * stride] = clip(t8 + t55a);
    c[9 * stride] = clip(t9 + t54);
    c[10 * stride] = clip(t10 + t53a);
    c[11 * stride] = clip(t11 + t52);
    c[12 * stride] = clip(t12 + t51a);
    c[13 * stride] = clip(t13 + t50);
    c[14 * stride] = clip(t14 + t49a);
    c[15 * stride] = clip(t15 + t48);
    c[16 * stride] = clip(t16 + t47);
    c[17 * stride] = clip(t17 + t46a);
    c[18 * stride] = clip(t18 + t45);
    c[19 * stride] = clip(t19 + t44a);
    c[20 * stride] = clip(t20 + t43);
    c[21 * stride] = clip(t21 + t42a);
    c[22 * stride] = clip(t22 + t41);
    c[23 * stride] = clip(t23 + t40a);
    c[24 * stride] = clip(t24 + t39);
    c[25 * stride] = clip(t25 + t38a);
    c[26 * stride] = clip(t26 + t37);
    c[27 * stride] = clip(t27 + t36a);
    c[28 * stride] = clip(t28 + t35);
    c[29 * stride] = clip(t29 + t34a);
    c[30 * stride] = clip(t30 + t33);
    c[31 * stride] = clip(t31 + t32a);
    c[32 * stride] = clip(t31 - t32a);
    c[33 * stride] = clip(t30 - t33);
    c[34 * stride] = clip(t29 - t34a);
    c[35 * stride] = clip(t28 - t35);
    c[36 * stride] = clip(t27 - t36a);
    c[37 * stride] = clip(t26 - t37);
    c[38 * stride] = clip(t25 - t38a);
    c[39 * stride] = clip(t24 - t39);
    c[40 * stride] = clip(t23 - t40a);
    c[41 * stride] = clip(t22 - t41);
    c[42 * stride] = clip(t21 - t42a);
    c[43 * stride] = clip(t20 - t43);
    c[44 * stride] = clip(t19 - t44a);
    c[45 * stride] = clip(t18 - t45);
    c[46 * stride] = clip(t17 - t46a);
    c[47 * stride] = clip(t16 - t47);
    c[48 * stride] = clip(t15 - t48);
    c[49 * stride] = clip(t14 - t49a);
    c[50 * stride] = clip(t13 - t50);
    c[51 * stride] = clip(t12 - t51a);
    c[52 * stride] = clip(t11 - t52);
    c[53 * stride] = clip(t10 - t53a);
    c[54 * stride] = clip(t9 - t54);
    c[55 * stride] = clip(t8 - t55a);
    c[56 * stride] = clip(t7 - t56);
    c[57 * stride] = clip(t6 - t57a);
    c[58 * stride] = clip(t5 - t58);
    c[59 * stride] = clip(t4 - t59a);
    c[60 * stride] = clip(t3 - t60);
    c[61 * stride] = clip(t2 - t61a);
    c[62 * stride] = clip(t1 - t62);
    c[63 * stride] = clip(t0 - t63a);
}

/// Identity64 1D transform (in-place)
#[inline]
fn identity64_1d(c: &mut [i32], stride: usize, _min: i32, _max: i32) {
    // For 64x64 identity: out = in * 4
    for i in 0..64 {
        c[i * stride] *= 4;
    }
}

/// Generic 64x64 transform function
///
/// AV1 high-frequency zeroing: only 32x32 coefficients are stored for 64x64
/// transforms. Coeff is column-major with stride 32, and has 1024 elements.
#[inline]
fn inv_txfm_64x64_inner<C: Copy + Into<i32>>(
    tmp: &mut [i32; 4096],
    coeff: &[C],
    row_transform: fn(&mut [i32], usize, i32, i32),
    col_transform: fn(&mut [i32], usize, i32, i32),
    row_clip_min: i32,
    row_clip_max: i32,
    col_clip_min: i32,
    col_clip_max: i32,
) {
    // For 64x64: shift=2, rnd=2
    let rnd = 2;
    let shift = 2;
    // Row transform - only first 32 rows have stored coefficients
    for y in 0..32 {
        // Load row from column-major (stride=32, only first 32 columns stored)
        let mut scratch = [0i32; 64];
        for x in 0..32 {
            scratch[x] = coeff[y + x * 32].into();
        }
        // Zero-extend: columns 32..63 have no stored coefficients
        for x in 32..64 {
            scratch[x] = 0;
        }
        row_transform(&mut scratch[..64], 1, row_clip_min, row_clip_max);
        for x in 0..64 {
            tmp[y * 64 + x] = ((scratch[x] + rnd) >> shift).clamp(col_clip_min, col_clip_max);
        }
    }
    // Rows 32..63 have no stored coefficients - zero them
    for y in 32..64 {
        for x in 0..64 {
            tmp[y * 64 + x] = 0;
        }
    }

    // Column transform (in-place, row-major with stride 64)
    for x in 0..64 {
        col_transform(&mut tmp[x..], 64, col_clip_min, col_clip_max);
    }
}

/// Add transformed coefficients to destination with SIMD (64x64)
/// `#[rite]` so it inlines into matching-feature `#[arcane]` callers (zero call cost).
#[cfg(target_arch = "x86_64")]
#[rite]
fn add_64x64_to_dst(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp: &[i32; 4096],
    coeff: &mut [i16],
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();

    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi16(bitdepth_max as i16);
    let rnd_final = _mm256_set1_epi32(8); // (+ 8) >> 4

    for y in 0..64 {
        let dst_off = y * dst_stride;

        // Process 64 pixels in four 16-pixel chunks
        for chunk in 0..4 {
            let x_base = chunk * 16;
            let dst_chunk_off = dst_off + x_base;

            let d =
                loadu_128!(<&[u8; 16]>::try_from(&dst[dst_chunk_off..dst_chunk_off + 16]).unwrap());
            let d16 = _mm256_cvtepu8_epi16(d);

            let c0 = _mm256_set_epi32(
                tmp[y * 64 + x_base + 7],
                tmp[y * 64 + x_base + 6],
                tmp[y * 64 + x_base + 5],
                tmp[y * 64 + x_base + 4],
                tmp[y * 64 + x_base + 3],
                tmp[y * 64 + x_base + 2],
                tmp[y * 64 + x_base + 1],
                tmp[y * 64 + x_base + 0],
            );
            let c1 = _mm256_set_epi32(
                tmp[y * 64 + x_base + 15],
                tmp[y * 64 + x_base + 14],
                tmp[y * 64 + x_base + 13],
                tmp[y * 64 + x_base + 12],
                tmp[y * 64 + x_base + 11],
                tmp[y * 64 + x_base + 10],
                tmp[y * 64 + x_base + 9],
                tmp[y * 64 + x_base + 8],
            );

            // Final scaling: (c + 8) >> 4
            let c0_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c0, rnd_final));
            let c1_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c1, rnd_final));

            let c16 = _mm256_packs_epi32(c0_scaled, c1_scaled);
            let c16 = _mm256_permute4x64_epi64::<0b11_01_10_00>(c16);

            let sum = _mm256_add_epi16(d16, c16);
            let clamped = _mm256_max_epi16(_mm256_min_epi16(sum, max_val), zero);

            let packed = _mm256_packus_epi16(clamped, clamped);
            let packed = _mm256_permute4x64_epi64::<0b11_01_10_00>(packed);

            storeu_128!(
                <&mut [u8; 16]>::try_from(&mut dst[dst_chunk_off..dst_chunk_off + 16]).unwrap(),
                _mm256_castsi256_si128(packed)
            );
        }
    }

    // Clear coefficients (only 1024 stored due to high-frequency zeroing)
    coeff[..1024].fill(0);
}

/// 64x64 DCT_DCT inner function
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_64x64_8bpc_avx2_inner(
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

    let mut tmp = [0i32; 4096];
    inv_txfm_64x64_inner(
        &mut tmp,
        &*coeff,
        dct64_1d,
        dct64_1d,
        row_clip_min,
        row_clip_max,
        col_clip_min,
        col_clip_max,
    );
    if let Some(t512) = crate::src::cpu::summon_avx512() {
        add_to_dst_8bpc_avx512(t512, &mut *dst, dst_stride, &tmp, 64, 64, 64, bitdepth_max);
    } else {
        add_64x64_to_dst(
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

/// FFI wrapper for 64x64 DCT_DCT 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_64x64_8bpc_avx2(
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

    inv_txfm_add_dct_dct_64x64_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// 64x64 DCT TRANSFORMS 16bpc
// ============================================================================

/// Add transformed coefficients to destination with SIMD (64x64 16bpc)
/// `#[rite]` so it inlines into matching-feature `#[arcane]` callers (zero call cost).
#[cfg(target_arch = "x86_64")]
#[rite]
fn add_64x64_to_dst_16bpc(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize,
    tmp: &[i32; 4096],
    coeff: &mut [i32],
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let stride_u16 = dst_stride / 2;

    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi32(bitdepth_max);
    let rnd_final = _mm256_set1_epi32(8); // (+ 8) >> 4

    for y in 0..64 {
        let dst_off = y * stride_u16;

        // Process 64 pixels in eight 8-pixel chunks
        for chunk in 0..8 {
            let x_base = chunk * 8;
            let dst_chunk_off = dst_off + x_base;

            // Load destination pixels (8 u16 = 16 bytes)
            let d =
                loadu_128!(<&[u16; 8]>::try_from(&dst[dst_chunk_off..dst_chunk_off + 8]).unwrap());
            let d_lo = _mm_unpacklo_epi16(d, _mm_setzero_si128());
            let d_hi = _mm_unpackhi_epi16(d, _mm_setzero_si128());

            // Load coefficients
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

            // Combine to 256-bit for faster processing
            let d32 = _mm256_set_m128i(d_hi, d_lo);
            let c32 = _mm256_set_m128i(c_hi, c_lo);

            // Final scaling: (c + 8) >> 4
            let c_scaled = _mm256_srai_epi32::<4>(_mm256_add_epi32(c32, rnd_final));

            // Add to destination
            let sum = _mm256_add_epi32(d32, c_scaled);

            // Clamp to [0, bitdepth_max]
            let clamped = _mm256_max_epi32(_mm256_min_epi32(sum, max_val), zero);

            // Pack to u16 and store
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

/// 64x64 DCT_DCT inner function for 16bpc
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_64x64_16bpc_avx2_inner(
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

    let mut tmp = [0i32; 4096];
    inv_txfm_64x64_inner(
        &mut tmp,
        &*coeff,
        dct64_1d,
        dct64_1d,
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
            64,
            64,
            64,
            bitdepth_max,
        );
    } else {
        add_64x64_to_dst_16bpc(
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

/// FFI wrapper for 64x64 DCT_DCT 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_64x64_16bpc_avx2(
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

    inv_txfm_add_dct_dct_64x64_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

