// ============================================================================
// SIMD column-parallel 1D transforms (AVX2)
//
// These functions perform a 1D inverse transform over 8 columns simultaneously,
// using AVX2 32-bit SIMD. They replace the strided scalar 1D column pass in
// the 2D dct_dct transforms, where the column transform reads with stride 16
// (or 32 for 32x32) which prevents auto-vectorization.
//
// Layout: input array is row-major; for an NxN block, row y col x is at index
// `y * stride + x`. The function processes columns [start..start+8] for all N
// rows in parallel.
// ============================================================================

/// AVX2 helper: compute `(a * c1 - b * c2 + rnd) >> SHIFT` over 8 lanes.
#[cfg(target_arch = "x86_64")]
#[rite]
fn mac_msub_shr<const SHIFT: i32>(
    _token: Desktop64,
    a: __m256i,
    c1: i32,
    b: __m256i,
    c2: i32,
    rnd: i32,
) -> __m256i {
    let p1 = _mm256_mullo_epi32(a, _mm256_set1_epi32(c1));
    let p2 = _mm256_mullo_epi32(b, _mm256_set1_epi32(c2));
    let sum = _mm256_add_epi32(_mm256_sub_epi32(p1, p2), _mm256_set1_epi32(rnd));
    _mm256_srai_epi32::<SHIFT>(sum)
}

/// AVX2 helper: compute `(a * c1 + b * c2 + rnd) >> SHIFT` over 8 lanes.
#[cfg(target_arch = "x86_64")]
#[rite]
fn mac_madd_shr<const SHIFT: i32>(
    _token: Desktop64,
    a: __m256i,
    c1: i32,
    b: __m256i,
    c2: i32,
    rnd: i32,
) -> __m256i {
    let p1 = _mm256_mullo_epi32(a, _mm256_set1_epi32(c1));
    let p2 = _mm256_mullo_epi32(b, _mm256_set1_epi32(c2));
    let sum = _mm256_add_epi32(_mm256_add_epi32(p1, p2), _mm256_set1_epi32(rnd));
    _mm256_srai_epi32::<SHIFT>(sum)
}

/// Clamp 8 i32 lanes to `[min, max]`.
#[cfg(target_arch = "x86_64")]
#[rite]
fn clip8(_token: Desktop64, v: __m256i, min_v: __m256i, max_v: __m256i) -> __m256i {
    _mm256_min_epi32(_mm256_max_epi32(v, min_v), max_v)
}

// =============================================================================
// AVX-512 helpers: 16-lane i32 SIMD column transforms (Server64 / X64V4Token)
// =============================================================================

/// AVX-512 helper: compute `(a * c1 - b * c2 + rnd) >> SHIFT` over 16 lanes.
#[cfg(target_arch = "x86_64")]
#[rite]
fn mac_msub_shr_v4<const SHIFT: u32>(
    _token: Server64,
    a: __m512i,
    c1: i32,
    b: __m512i,
    c2: i32,
    rnd: i32,
) -> __m512i {
    let p1 = _mm512_mullo_epi32(a, _mm512_set1_epi32(c1));
    let p2 = _mm512_mullo_epi32(b, _mm512_set1_epi32(c2));
    let sum = _mm512_add_epi32(_mm512_sub_epi32(p1, p2), _mm512_set1_epi32(rnd));
    _mm512_srai_epi32::<SHIFT>(sum)
}

/// AVX-512 helper: compute `(a * c1 + b * c2 + rnd) >> SHIFT` over 16 lanes.
#[cfg(target_arch = "x86_64")]
#[rite]
fn mac_madd_shr_v4<const SHIFT: u32>(
    _token: Server64,
    a: __m512i,
    c1: i32,
    b: __m512i,
    c2: i32,
    rnd: i32,
) -> __m512i {
    let p1 = _mm512_mullo_epi32(a, _mm512_set1_epi32(c1));
    let p2 = _mm512_mullo_epi32(b, _mm512_set1_epi32(c2));
    let sum = _mm512_add_epi32(_mm512_add_epi32(p1, p2), _mm512_set1_epi32(rnd));
    _mm512_srai_epi32::<SHIFT>(sum)
}

/// Clamp 16 i32 lanes to `[min, max]`.
#[cfg(target_arch = "x86_64")]
#[rite]
fn clip16_i32(_token: Server64, v: __m512i, min_v: __m512i, max_v: __m512i) -> __m512i {
    _mm512_min_epi32(_mm512_max_epi32(v, min_v), max_v)
}

/// DCT4 1D transform across 16 columns in parallel.
#[cfg(target_arch = "x86_64")]
#[rite]
fn dct4_1d_cols16(token: Server64, c: &mut [__m512i; 4], min_v: __m512i, max_v: __m512i) {
    let in0 = c[0];
    let in1 = c[1];
    let in2 = c[2];
    let in3 = c[3];
    let sum02 = _mm512_add_epi32(in0, in2);
    let sub02 = _mm512_sub_epi32(in0, in2);
    let t0 = _mm512_srai_epi32::<8>(_mm512_add_epi32(
        _mm512_mullo_epi32(sum02, _mm512_set1_epi32(181)),
        _mm512_set1_epi32(128),
    ));
    let t1 = _mm512_srai_epi32::<8>(_mm512_add_epi32(
        _mm512_mullo_epi32(sub02, _mm512_set1_epi32(181)),
        _mm512_set1_epi32(128),
    ));
    let t2_shifted = mac_msub_shr_v4::<12>(token, in1, 1567, in3, 3784 - 4096, 2048);
    let t2 = _mm512_sub_epi32(t2_shifted, in3);
    let t3_shifted = mac_madd_shr_v4::<12>(token, in1, 3784 - 4096, in3, 1567, 2048);
    let t3 = _mm512_add_epi32(t3_shifted, in1);

    c[0] = clip16_i32(token, _mm512_add_epi32(t0, t3), min_v, max_v);
    c[1] = clip16_i32(token, _mm512_add_epi32(t1, t2), min_v, max_v);
    c[2] = clip16_i32(token, _mm512_sub_epi32(t1, t2), min_v, max_v);
    c[3] = clip16_i32(token, _mm512_sub_epi32(t0, t3), min_v, max_v);
}

/// DCT8 1D transform across 16 columns in parallel.
#[cfg(target_arch = "x86_64")]
#[rite]
fn dct8_1d_cols16(token: Server64, c: &mut [__m512i; 8], min_v: __m512i, max_v: __m512i) {
    let mut even = [c[0], c[2], c[4], c[6]];
    dct4_1d_cols16(token, &mut even, min_v, max_v);
    c[0] = even[0];
    c[2] = even[1];
    c[4] = even[2];
    c[6] = even[3];

    let in1 = c[1];
    let in3 = c[3];
    let in5 = c[5];
    let in7 = c[7];

    let t4a = _mm512_sub_epi32(
        mac_msub_shr_v4::<12>(token, in1, 799, in7, 4017 - 4096, 2048),
        in7,
    );
    let t5a = mac_msub_shr_v4::<11>(token, in5, 1703, in3, 1138, 1024);
    let t6a = mac_madd_shr_v4::<11>(token, in5, 1138, in3, 1703, 1024);
    let t7a = _mm512_add_epi32(
        mac_madd_shr_v4::<12>(token, in1, 4017 - 4096, in7, 799, 2048),
        in1,
    );

    let t4 = clip16_i32(token, _mm512_add_epi32(t4a, t5a), min_v, max_v);
    let t5a_n = clip16_i32(token, _mm512_sub_epi32(t4a, t5a), min_v, max_v);
    let t7 = clip16_i32(token, _mm512_add_epi32(t7a, t6a), min_v, max_v);
    let t6a_n = clip16_i32(token, _mm512_sub_epi32(t7a, t6a), min_v, max_v);

    let d = _mm512_sub_epi32(t6a_n, t5a_n);
    let t5 = _mm512_srai_epi32::<8>(_mm512_add_epi32(
        _mm512_mullo_epi32(d, _mm512_set1_epi32(181)),
        _mm512_set1_epi32(128),
    ));
    let s = _mm512_add_epi32(t6a_n, t5a_n);
    let t6 = _mm512_srai_epi32::<8>(_mm512_add_epi32(
        _mm512_mullo_epi32(s, _mm512_set1_epi32(181)),
        _mm512_set1_epi32(128),
    ));

    let t0 = c[0];
    let t1 = c[2];
    let t2 = c[4];
    let t3 = c[6];

    c[0] = clip16_i32(token, _mm512_add_epi32(t0, t7), min_v, max_v);
    c[1] = clip16_i32(token, _mm512_add_epi32(t1, t6), min_v, max_v);
    c[2] = clip16_i32(token, _mm512_add_epi32(t2, t5), min_v, max_v);
    c[3] = clip16_i32(token, _mm512_add_epi32(t3, t4), min_v, max_v);
    c[4] = clip16_i32(token, _mm512_sub_epi32(t3, t4), min_v, max_v);
    c[5] = clip16_i32(token, _mm512_sub_epi32(t2, t5), min_v, max_v);
    c[6] = clip16_i32(token, _mm512_sub_epi32(t1, t6), min_v, max_v);
    c[7] = clip16_i32(token, _mm512_sub_epi32(t0, t7), min_v, max_v);
}

/// DCT16 1D transform across 16 columns in parallel.
#[cfg(target_arch = "x86_64")]
#[rite]
fn dct16_1d_cols16(token: Server64, c: &mut [__m512i; 16], min_v: __m512i, max_v: __m512i) {
    let mut even = [c[0], c[2], c[4], c[6], c[8], c[10], c[12], c[14]];
    dct8_1d_cols16(token, &mut even, min_v, max_v);
    c[0] = even[0];
    c[2] = even[1];
    c[4] = even[2];
    c[6] = even[3];
    c[8] = even[4];
    c[10] = even[5];
    c[12] = even[6];
    c[14] = even[7];

    let in1 = c[1];
    let in3 = c[3];
    let in5 = c[5];
    let in7 = c[7];
    let in9 = c[9];
    let in11 = c[11];
    let in13 = c[13];
    let in15 = c[15];

    let t8a = _mm512_sub_epi32(
        mac_msub_shr_v4::<12>(token, in1, 401, in15, 4076 - 4096, 2048),
        in15,
    );
    let t9a = mac_msub_shr_v4::<11>(token, in9, 1583, in7, 1299, 1024);
    let t10a = _mm512_sub_epi32(
        mac_msub_shr_v4::<12>(token, in5, 1931, in11, 3612 - 4096, 2048),
        in11,
    );
    let t11a = _mm512_add_epi32(
        mac_msub_shr_v4::<12>(token, in13, 3920 - 4096, in3, 1189, 2048),
        in13,
    );
    let t12a = _mm512_add_epi32(
        mac_madd_shr_v4::<12>(token, in13, 1189, in3, 3920 - 4096, 2048),
        in3,
    );
    let t13a = _mm512_add_epi32(
        mac_madd_shr_v4::<12>(token, in5, 3612 - 4096, in11, 1931, 2048),
        in5,
    );
    let t14a = mac_madd_shr_v4::<11>(token, in9, 1299, in7, 1583, 1024);
    let t15a = _mm512_add_epi32(
        mac_madd_shr_v4::<12>(token, in1, 4076 - 4096, in15, 401, 2048),
        in1,
    );

    let t8 = clip16_i32(token, _mm512_add_epi32(t8a, t9a), min_v, max_v);
    let mut t9 = clip16_i32(token, _mm512_sub_epi32(t8a, t9a), min_v, max_v);
    let mut t10 = clip16_i32(token, _mm512_sub_epi32(t11a, t10a), min_v, max_v);
    let mut t11 = clip16_i32(token, _mm512_add_epi32(t11a, t10a), min_v, max_v);
    let mut t12 = clip16_i32(token, _mm512_add_epi32(t12a, t13a), min_v, max_v);
    let mut t13 = clip16_i32(token, _mm512_sub_epi32(t12a, t13a), min_v, max_v);
    let mut t14 = clip16_i32(token, _mm512_sub_epi32(t15a, t14a), min_v, max_v);
    let t15 = clip16_i32(token, _mm512_add_epi32(t15a, t14a), min_v, max_v);

    let t9a = _mm512_sub_epi32(
        mac_msub_shr_v4::<12>(token, t14, 1567, t9, 3784 - 4096, 2048),
        t9,
    );
    let t14a = _mm512_add_epi32(
        mac_madd_shr_v4::<12>(token, t14, 3784 - 4096, t9, 1567, 2048),
        t14,
    );
    let t10a_inner = _mm512_add_epi32(
        _mm512_mullo_epi32(t13, _mm512_set1_epi32(3784 - 4096)),
        _mm512_mullo_epi32(t10, _mm512_set1_epi32(1567)),
    );
    let t10a = _mm512_sub_epi32(
        _mm512_srai_epi32::<12>(_mm512_add_epi32(
            _mm512_sub_epi32(_mm512_setzero_si512(), t10a_inner),
            _mm512_set1_epi32(2048),
        )),
        t13,
    );
    let t13a = _mm512_sub_epi32(
        mac_msub_shr_v4::<12>(token, t13, 1567, t10, 3784 - 4096, 2048),
        t10,
    );

    let t8a = clip16_i32(token, _mm512_add_epi32(t8, t11), min_v, max_v);
    t9 = clip16_i32(token, _mm512_add_epi32(t9a, t10a), min_v, max_v);
    t10 = clip16_i32(token, _mm512_sub_epi32(t9a, t10a), min_v, max_v);
    let t11a = clip16_i32(token, _mm512_sub_epi32(t8, t11), min_v, max_v);
    let t12a = clip16_i32(token, _mm512_sub_epi32(t15, t12), min_v, max_v);
    t13 = clip16_i32(token, _mm512_sub_epi32(t14a, t13a), min_v, max_v);
    t14 = clip16_i32(token, _mm512_add_epi32(t14a, t13a), min_v, max_v);
    let t15a = clip16_i32(token, _mm512_add_epi32(t15, t12), min_v, max_v);

    let d_13_10 = _mm512_sub_epi32(t13, t10);
    let t10a_new = _mm512_srai_epi32::<8>(_mm512_add_epi32(
        _mm512_mullo_epi32(d_13_10, _mm512_set1_epi32(181)),
        _mm512_set1_epi32(128),
    ));
    let s_13_10 = _mm512_add_epi32(t13, t10);
    let t13a_new = _mm512_srai_epi32::<8>(_mm512_add_epi32(
        _mm512_mullo_epi32(s_13_10, _mm512_set1_epi32(181)),
        _mm512_set1_epi32(128),
    ));
    let d_12a_11a = _mm512_sub_epi32(t12a, t11a);
    t11 = _mm512_srai_epi32::<8>(_mm512_add_epi32(
        _mm512_mullo_epi32(d_12a_11a, _mm512_set1_epi32(181)),
        _mm512_set1_epi32(128),
    ));
    let s_12a_11a = _mm512_add_epi32(t12a, t11a);
    t12 = _mm512_srai_epi32::<8>(_mm512_add_epi32(
        _mm512_mullo_epi32(s_12a_11a, _mm512_set1_epi32(181)),
        _mm512_set1_epi32(128),
    ));

    let t0 = c[0];
    let t1 = c[2];
    let t2 = c[4];
    let t3 = c[6];
    let t4 = c[8];
    let t5 = c[10];
    let t6 = c[12];
    let t7 = c[14];

    c[0] = clip16_i32(token, _mm512_add_epi32(t0, t15a), min_v, max_v);
    c[1] = clip16_i32(token, _mm512_add_epi32(t1, t14), min_v, max_v);
    c[2] = clip16_i32(token, _mm512_add_epi32(t2, t13a_new), min_v, max_v);
    c[3] = clip16_i32(token, _mm512_add_epi32(t3, t12), min_v, max_v);
    c[4] = clip16_i32(token, _mm512_add_epi32(t4, t11), min_v, max_v);
    c[5] = clip16_i32(token, _mm512_add_epi32(t5, t10a_new), min_v, max_v);
    c[6] = clip16_i32(token, _mm512_add_epi32(t6, t9), min_v, max_v);
    c[7] = clip16_i32(token, _mm512_add_epi32(t7, t8a), min_v, max_v);
    c[8] = clip16_i32(token, _mm512_sub_epi32(t7, t8a), min_v, max_v);
    c[9] = clip16_i32(token, _mm512_sub_epi32(t6, t9), min_v, max_v);
    c[10] = clip16_i32(token, _mm512_sub_epi32(t5, t10a_new), min_v, max_v);
    c[11] = clip16_i32(token, _mm512_sub_epi32(t4, t11), min_v, max_v);
    c[12] = clip16_i32(token, _mm512_sub_epi32(t3, t12), min_v, max_v);
    c[13] = clip16_i32(token, _mm512_sub_epi32(t2, t13a_new), min_v, max_v);
    c[14] = clip16_i32(token, _mm512_sub_epi32(t1, t14), min_v, max_v);
    c[15] = clip16_i32(token, _mm512_sub_epi32(t0, t15a), min_v, max_v);
}

/// DCT32 1D transform across 16 columns in parallel (AVX-512).
#[cfg(target_arch = "x86_64")]
#[rite]
fn dct32_1d_cols16(token: Server64, c: &mut [__m512i; 32], min_v: __m512i, max_v: __m512i) {
    // Apply DCT16 to even positions
    let mut even = [
        c[0], c[2], c[4], c[6], c[8], c[10], c[12], c[14], c[16], c[18], c[20], c[22], c[24],
        c[26], c[28], c[30],
    ];
    dct16_1d_cols16(token, &mut even, min_v, max_v);
    c[0] = even[0];
    c[2] = even[1];
    c[4] = even[2];
    c[6] = even[3];
    c[8] = even[4];
    c[10] = even[5];
    c[12] = even[6];
    c[14] = even[7];
    c[16] = even[8];
    c[18] = even[9];
    c[20] = even[10];
    c[22] = even[11];
    c[24] = even[12];
    c[26] = even[13];
    c[28] = even[14];
    c[30] = even[15];

    let in1 = c[1];
    let in3 = c[3];
    let in5 = c[5];
    let in7 = c[7];
    let in9 = c[9];
    let in11 = c[11];
    let in13 = c[13];
    let in15 = c[15];
    let in17 = c[17];
    let in19 = c[19];
    let in21 = c[21];
    let in23 = c[23];
    let in25 = c[25];
    let in27 = c[27];
    let in29 = c[29];
    let in31 = c[31];

    let t16a = _mm512_sub_epi32(
        mac_msub_shr_v4::<12>(token, in1, 201, in31, 4091 - 4096, 2048),
        in31,
    );
    let t17a = _mm512_add_epi32(
        mac_msub_shr_v4::<12>(token, in17, 3035 - 4096, in15, 2751, 2048),
        in17,
    );
    let t18a = _mm512_sub_epi32(
        mac_msub_shr_v4::<12>(token, in9, 1751, in23, 3703 - 4096, 2048),
        in23,
    );
    let t19a = _mm512_add_epi32(
        mac_msub_shr_v4::<12>(token, in25, 3857 - 4096, in7, 1380, 2048),
        in25,
    );
    let t20a = _mm512_sub_epi32(
        mac_msub_shr_v4::<12>(token, in5, 995, in27, 3973 - 4096, 2048),
        in27,
    );
    let t21a = _mm512_add_epi32(
        mac_msub_shr_v4::<12>(token, in21, 3513 - 4096, in11, 2106, 2048),
        in21,
    );
    let t22a = mac_msub_shr_v4::<11>(token, in13, 1220, in19, 1645, 1024);
    let t23a = _mm512_add_epi32(
        mac_msub_shr_v4::<12>(token, in29, 4052 - 4096, in3, 601, 2048),
        in29,
    );
    let t24a = _mm512_add_epi32(
        mac_madd_shr_v4::<12>(token, in29, 601, in3, 4052 - 4096, 2048),
        in3,
    );
    let t25a = mac_madd_shr_v4::<11>(token, in13, 1645, in19, 1220, 1024);
    let t26a = _mm512_add_epi32(
        mac_madd_shr_v4::<12>(token, in21, 2106, in11, 3513 - 4096, 2048),
        in11,
    );
    let t27a = _mm512_add_epi32(
        mac_madd_shr_v4::<12>(token, in5, 3973 - 4096, in27, 995, 2048),
        in5,
    );
    let t28a = _mm512_add_epi32(
        mac_madd_shr_v4::<12>(token, in25, 1380, in7, 3857 - 4096, 2048),
        in7,
    );
    let t29a = _mm512_add_epi32(
        mac_madd_shr_v4::<12>(token, in9, 3703 - 4096, in23, 1751, 2048),
        in9,
    );
    let t30a = _mm512_add_epi32(
        mac_madd_shr_v4::<12>(token, in17, 2751, in15, 3035 - 4096, 2048),
        in15,
    );
    let t31a = _mm512_add_epi32(
        mac_madd_shr_v4::<12>(token, in1, 4091 - 4096, in31, 201, 2048),
        in1,
    );

    let mut t16 = clip16_i32(token, _mm512_add_epi32(t16a, t17a), min_v, max_v);
    let mut t17 = clip16_i32(token, _mm512_sub_epi32(t16a, t17a), min_v, max_v);
    let mut t18 = clip16_i32(token, _mm512_sub_epi32(t19a, t18a), min_v, max_v);
    let t19 = clip16_i32(token, _mm512_add_epi32(t19a, t18a), min_v, max_v);
    let t20 = clip16_i32(token, _mm512_add_epi32(t20a, t21a), min_v, max_v);
    let mut t21 = clip16_i32(token, _mm512_sub_epi32(t20a, t21a), min_v, max_v);
    let mut t22 = clip16_i32(token, _mm512_sub_epi32(t23a, t22a), min_v, max_v);
    let mut t23 = clip16_i32(token, _mm512_add_epi32(t23a, t22a), min_v, max_v);
    let mut t24 = clip16_i32(token, _mm512_add_epi32(t24a, t25a), min_v, max_v);
    let mut t25 = clip16_i32(token, _mm512_sub_epi32(t24a, t25a), min_v, max_v);
    let mut t26 = clip16_i32(token, _mm512_sub_epi32(t27a, t26a), min_v, max_v);
    let t27 = clip16_i32(token, _mm512_add_epi32(t27a, t26a), min_v, max_v);
    let t28 = clip16_i32(token, _mm512_add_epi32(t28a, t29a), min_v, max_v);
    let mut t29 = clip16_i32(token, _mm512_sub_epi32(t28a, t29a), min_v, max_v);
    let mut t30 = clip16_i32(token, _mm512_sub_epi32(t31a, t30a), min_v, max_v);
    let mut t31 = clip16_i32(token, _mm512_add_epi32(t31a, t30a), min_v, max_v);

    let t17a = _mm512_sub_epi32(
        mac_msub_shr_v4::<12>(token, t30, 799, t17, 4017 - 4096, 2048),
        t17,
    );
    let t30a = _mm512_add_epi32(
        mac_madd_shr_v4::<12>(token, t30, 4017 - 4096, t17, 799, 2048),
        t30,
    );
    let t18a_inner = _mm512_add_epi32(
        _mm512_mullo_epi32(t29, _mm512_set1_epi32(4017 - 4096)),
        _mm512_mullo_epi32(t18, _mm512_set1_epi32(799)),
    );
    let t18a = _mm512_sub_epi32(
        _mm512_srai_epi32::<12>(_mm512_add_epi32(
            _mm512_sub_epi32(_mm512_setzero_si512(), t18a_inner),
            _mm512_set1_epi32(2048),
        )),
        t29,
    );
    let t29a = _mm512_sub_epi32(
        mac_msub_shr_v4::<12>(token, t29, 799, t18, 4017 - 4096, 2048),
        t18,
    );
    let t21a = mac_msub_shr_v4::<11>(token, t26, 1703, t21, 1138, 1024);
    let t26a = mac_madd_shr_v4::<11>(token, t26, 1138, t21, 1703, 1024);
    let t22a_inner = _mm512_add_epi32(
        _mm512_mullo_epi32(t25, _mm512_set1_epi32(1138)),
        _mm512_mullo_epi32(t22, _mm512_set1_epi32(1703)),
    );
    let t22a = _mm512_srai_epi32::<11>(_mm512_add_epi32(
        _mm512_sub_epi32(_mm512_setzero_si512(), t22a_inner),
        _mm512_set1_epi32(1024),
    ));
    let t25a = mac_msub_shr_v4::<11>(token, t25, 1703, t22, 1138, 1024);

    let t16a = clip16_i32(token, _mm512_add_epi32(t16, t19), min_v, max_v);
    t17 = clip16_i32(token, _mm512_add_epi32(t17a, t18a), min_v, max_v);
    t18 = clip16_i32(token, _mm512_sub_epi32(t17a, t18a), min_v, max_v);
    let t19a = clip16_i32(token, _mm512_sub_epi32(t16, t19), min_v, max_v);
    let t20a = clip16_i32(token, _mm512_sub_epi32(t23, t20), min_v, max_v);
    t21 = clip16_i32(token, _mm512_sub_epi32(t22a, t21a), min_v, max_v);
    t22 = clip16_i32(token, _mm512_add_epi32(t22a, t21a), min_v, max_v);
    let t23a = clip16_i32(token, _mm512_add_epi32(t23, t20), min_v, max_v);
    let t24a = clip16_i32(token, _mm512_add_epi32(t24, t27), min_v, max_v);
    t25 = clip16_i32(token, _mm512_add_epi32(t25a, t26a), min_v, max_v);
    t26 = clip16_i32(token, _mm512_sub_epi32(t25a, t26a), min_v, max_v);
    let t27a = clip16_i32(token, _mm512_sub_epi32(t24, t27), min_v, max_v);
    let t28a = clip16_i32(token, _mm512_sub_epi32(t31, t28), min_v, max_v);
    t29 = clip16_i32(token, _mm512_sub_epi32(t30a, t29a), min_v, max_v);
    t30 = clip16_i32(token, _mm512_add_epi32(t30a, t29a), min_v, max_v);
    let t31a = clip16_i32(token, _mm512_add_epi32(t31, t28), min_v, max_v);

    let t18a = _mm512_sub_epi32(
        mac_msub_shr_v4::<12>(token, t29, 1567, t18, 3784 - 4096, 2048),
        t18,
    );
    let t29a = _mm512_add_epi32(
        mac_madd_shr_v4::<12>(token, t29, 3784 - 4096, t18, 1567, 2048),
        t29,
    );
    let t19 = _mm512_sub_epi32(
        mac_msub_shr_v4::<12>(token, t28a, 1567, t19a, 3784 - 4096, 2048),
        t19a,
    );
    let t28 = _mm512_add_epi32(
        mac_madd_shr_v4::<12>(token, t28a, 3784 - 4096, t19a, 1567, 2048),
        t28a,
    );
    let t20_inner = _mm512_add_epi32(
        _mm512_mullo_epi32(t27a, _mm512_set1_epi32(3784 - 4096)),
        _mm512_mullo_epi32(t20a, _mm512_set1_epi32(1567)),
    );
    let t20 = _mm512_sub_epi32(
        _mm512_srai_epi32::<12>(_mm512_add_epi32(
            _mm512_sub_epi32(_mm512_setzero_si512(), t20_inner),
            _mm512_set1_epi32(2048),
        )),
        t27a,
    );
    let t27 = _mm512_sub_epi32(
        mac_msub_shr_v4::<12>(token, t27a, 1567, t20a, 3784 - 4096, 2048),
        t20a,
    );
    let t21a_inner = _mm512_add_epi32(
        _mm512_mullo_epi32(t26, _mm512_set1_epi32(3784 - 4096)),
        _mm512_mullo_epi32(t21, _mm512_set1_epi32(1567)),
    );
    let t21a = _mm512_sub_epi32(
        _mm512_srai_epi32::<12>(_mm512_add_epi32(
            _mm512_sub_epi32(_mm512_setzero_si512(), t21a_inner),
            _mm512_set1_epi32(2048),
        )),
        t26,
    );
    let t26a = _mm512_sub_epi32(
        mac_msub_shr_v4::<12>(token, t26, 1567, t21, 3784 - 4096, 2048),
        t21,
    );

    t16 = clip16_i32(token, _mm512_add_epi32(t16a, t23a), min_v, max_v);
    let t17a = clip16_i32(token, _mm512_add_epi32(t17, t22), min_v, max_v);
    t18 = clip16_i32(token, _mm512_add_epi32(t18a, t21a), min_v, max_v);
    let t19a = clip16_i32(token, _mm512_add_epi32(t19, t20), min_v, max_v);
    let t20a = clip16_i32(token, _mm512_sub_epi32(t19, t20), min_v, max_v);
    t21 = clip16_i32(token, _mm512_sub_epi32(t18a, t21a), min_v, max_v);
    let t22a = clip16_i32(token, _mm512_sub_epi32(t17, t22), min_v, max_v);
    t23 = clip16_i32(token, _mm512_sub_epi32(t16a, t23a), min_v, max_v);
    t24 = clip16_i32(token, _mm512_sub_epi32(t31a, t24a), min_v, max_v);
    let t25a = clip16_i32(token, _mm512_sub_epi32(t30, t25), min_v, max_v);
    t26 = clip16_i32(token, _mm512_sub_epi32(t29a, t26a), min_v, max_v);
    let t27a = clip16_i32(token, _mm512_sub_epi32(t28, t27), min_v, max_v);
    let t28a = clip16_i32(token, _mm512_add_epi32(t28, t27), min_v, max_v);
    t29 = clip16_i32(token, _mm512_add_epi32(t29a, t26a), min_v, max_v);
    let t30a = clip16_i32(token, _mm512_add_epi32(t30, t25), min_v, max_v);
    t31 = clip16_i32(token, _mm512_add_epi32(t31a, t24a), min_v, max_v);

    let mul181_sum = |a: __m512i, b: __m512i| {
        let s = _mm512_add_epi32(a, b);
        _mm512_srai_epi32::<8>(_mm512_add_epi32(
            _mm512_mullo_epi32(s, _mm512_set1_epi32(181)),
            _mm512_set1_epi32(128),
        ))
    };
    let mul181_diff = |a: __m512i, b: __m512i| {
        let s = _mm512_sub_epi32(a, b);
        _mm512_srai_epi32::<8>(_mm512_add_epi32(
            _mm512_mullo_epi32(s, _mm512_set1_epi32(181)),
            _mm512_set1_epi32(128),
        ))
    };
    let t20_final = mul181_diff(t27a, t20a);
    let t27_final = mul181_sum(t27a, t20a);
    let t21a_final = mul181_diff(t26, t21);
    let t26a_final = mul181_sum(t26, t21);
    let t22_final = mul181_diff(t25a, t22a);
    let t25_final = mul181_sum(t25a, t22a);
    let t23a = mul181_diff(t24, t23);
    let t24a = mul181_sum(t24, t23);

    let t0 = c[0];
    let t1 = c[2];
    let t2 = c[4];
    let t3 = c[6];
    let t4 = c[8];
    let t5 = c[10];
    let t6 = c[12];
    let t7 = c[14];
    let t8 = c[16];
    let t9 = c[18];
    let t10 = c[20];
    let t11 = c[22];
    let t12 = c[24];
    let t13 = c[26];
    let t14 = c[28];
    let t15 = c[30];

    c[0] = clip16_i32(token, _mm512_add_epi32(t0, t31), min_v, max_v);
    c[1] = clip16_i32(token, _mm512_add_epi32(t1, t30a), min_v, max_v);
    c[2] = clip16_i32(token, _mm512_add_epi32(t2, t29), min_v, max_v);
    c[3] = clip16_i32(token, _mm512_add_epi32(t3, t28a), min_v, max_v);
    c[4] = clip16_i32(token, _mm512_add_epi32(t4, t27_final), min_v, max_v);
    c[5] = clip16_i32(token, _mm512_add_epi32(t5, t26a_final), min_v, max_v);
    c[6] = clip16_i32(token, _mm512_add_epi32(t6, t25_final), min_v, max_v);
    c[7] = clip16_i32(token, _mm512_add_epi32(t7, t24a), min_v, max_v);
    c[8] = clip16_i32(token, _mm512_add_epi32(t8, t23a), min_v, max_v);
    c[9] = clip16_i32(token, _mm512_add_epi32(t9, t22_final), min_v, max_v);
    c[10] = clip16_i32(token, _mm512_add_epi32(t10, t21a_final), min_v, max_v);
    c[11] = clip16_i32(token, _mm512_add_epi32(t11, t20_final), min_v, max_v);
    c[12] = clip16_i32(token, _mm512_add_epi32(t12, t19a), min_v, max_v);
    c[13] = clip16_i32(token, _mm512_add_epi32(t13, t18), min_v, max_v);
    c[14] = clip16_i32(token, _mm512_add_epi32(t14, t17a), min_v, max_v);
    c[15] = clip16_i32(token, _mm512_add_epi32(t15, t16), min_v, max_v);
    c[16] = clip16_i32(token, _mm512_sub_epi32(t15, t16), min_v, max_v);
    c[17] = clip16_i32(token, _mm512_sub_epi32(t14, t17a), min_v, max_v);
    c[18] = clip16_i32(token, _mm512_sub_epi32(t13, t18), min_v, max_v);
    c[19] = clip16_i32(token, _mm512_sub_epi32(t12, t19a), min_v, max_v);
    c[20] = clip16_i32(token, _mm512_sub_epi32(t11, t20_final), min_v, max_v);
    c[21] = clip16_i32(token, _mm512_sub_epi32(t10, t21a_final), min_v, max_v);
    c[22] = clip16_i32(token, _mm512_sub_epi32(t9, t22_final), min_v, max_v);
    c[23] = clip16_i32(token, _mm512_sub_epi32(t8, t23a), min_v, max_v);
    c[24] = clip16_i32(token, _mm512_sub_epi32(t7, t24a), min_v, max_v);
    c[25] = clip16_i32(token, _mm512_sub_epi32(t6, t25_final), min_v, max_v);
    c[26] = clip16_i32(token, _mm512_sub_epi32(t5, t26a_final), min_v, max_v);
    c[27] = clip16_i32(token, _mm512_sub_epi32(t4, t27_final), min_v, max_v);
    c[28] = clip16_i32(token, _mm512_sub_epi32(t3, t28a), min_v, max_v);
    c[29] = clip16_i32(token, _mm512_sub_epi32(t2, t29), min_v, max_v);
    c[30] = clip16_i32(token, _mm512_sub_epi32(t1, t30a), min_v, max_v);
    c[31] = clip16_i32(token, _mm512_sub_epi32(t0, t31), min_v, max_v);
}

/// Run dct32 1D column transform over a row-major buffer using AVX-512.
/// Processes 16 cols at a time. `total_w` must be a multiple of 16.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn dct32_cols_avx512(
    token: Server64,
    tmp: &mut [i32],
    total_w: usize,
    n_rows: usize,
    min: i32,
    max: i32,
) {
    let min_v = _mm512_set1_epi32(min);
    let max_v = _mm512_set1_epi32(max);
    let n_chunks = total_w / 16;
    for cx_chunk in 0..n_chunks {
        let cx = cx_chunk * 16;
        let mut v = [_mm512_setzero_si512(); 32];
        for i in 0..32usize.min(n_rows) {
            let arr_ref: &[i32; 16] = (&tmp[i * total_w + cx..i * total_w + cx + 16])
                .try_into()
                .unwrap();
            v[i] = loadu_512!(arr_ref);
        }
        dct32_1d_cols16(token, &mut v, min_v, max_v);
        for i in 0..32usize.min(n_rows) {
            let arr_ref: &mut [i32; 16] = (&mut tmp[i * total_w + cx..i * total_w + cx + 16])
                .try_into()
                .unwrap();
            storeu_512!(arr_ref, v[i]);
        }
    }
}

/// Run dct16 1D column transform over a row-major buffer using AVX-512.
/// `tmp` has `total_w` cols × `n_rows` rows. Processes 16 cols at a time.
/// `n_chunks` = `total_w / 16`. Caller is responsible for ensuring this divides.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn dct16_cols_avx512(
    token: Server64,
    tmp: &mut [i32],
    total_w: usize,
    n_rows: usize,
    min: i32,
    max: i32,
) {
    let min_v = _mm512_set1_epi32(min);
    let max_v = _mm512_set1_epi32(max);
    let n_chunks = total_w / 16;
    for cx_chunk in 0..n_chunks {
        let cx = cx_chunk * 16;
        let mut v = [_mm512_setzero_si512(); 16];
        for i in 0..16usize.min(n_rows) {
            let arr_ref: &[i32; 16] = (&tmp[i * total_w + cx..i * total_w + cx + 16])
                .try_into()
                .unwrap();
            v[i] = loadu_512!(arr_ref);
        }
        dct16_1d_cols16(token, &mut v, min_v, max_v);
        for i in 0..16usize.min(n_rows) {
            let arr_ref: &mut [i32; 16] = (&mut tmp[i * total_w + cx..i * total_w + cx + 16])
                .try_into()
                .unwrap();
            storeu_512!(arr_ref, v[i]);
        }
    }
}

// =============================================================================
// AVX-512 row pass support: 16-row transpose + 16-row DCT row passes
//
// The AVX2 row pass (`simd_row_dct{8,16,32}_8bpc_8rows`) processes 8 rows per
// call: load 8 i16 per column (one column = 8 rows) → cvtepi16_epi32 → run the
// i32 column transform (each of 8 lanes = one row) → round/shift/clip →
// transpose 8x8 → store row-major.
//
// The AVX-512 row pass processes 16 rows per call: load 16 i16 per column (one
// column = 16 rows) into a `__m256i` → cvtepi16_epi32 to `__m512i` (16 lanes =
// 16 rows) → run the proven `dct{8,16,32}_1d_cols16` (16-lane mullo transform,
// already bit-exact vs scalar) → round/shift/clip → transpose 16xN → store.
//
// Reusing `dct*_1d_cols16` for the math (instead of a hand-written 512-bit
// pmaddwd path) guarantees bit-exactness: it is the same i32 algorithm the
// AVX-512 column pass already uses, which MD5 already exercises on this machine.
// The only new SIMD is the 16x16 i32 transpose below.
// =============================================================================

/// Transpose a 16x16 block of i32 held as `[__m512i; 16]` (16 columns, each a
/// 512-bit vector of 16 i32 = 16 rows) into 16 rows (each a 512-bit vector of
/// 16 i32 = 16 columns). Pure value-intrinsic shuffle; bit-exact, no memory.
///
/// Strategy: 4 stages of unpack/permute, mirroring the classic AVX-512 16x16
/// i32 transpose. Stage 1 interleaves adjacent columns at the 32-bit level,
/// stage 2 at 64-bit, stage 3/4 cross the 128-bit lanes via `shuffle_i32x4`.
#[cfg(target_arch = "x86_64")]
#[allow(unused_macros)]
macro_rules! transpose_16x16_i32 {
    ($a:expr) => {{
        let __c: [__m512i; 16] = $a;
        // Stage 1: unpack 32-bit lanes between column pairs (0,1),(2,3),...
        let __a0 = _mm512_unpacklo_epi32(__c[0], __c[1]);
        let __a1 = _mm512_unpackhi_epi32(__c[0], __c[1]);
        let __a2 = _mm512_unpacklo_epi32(__c[2], __c[3]);
        let __a3 = _mm512_unpackhi_epi32(__c[2], __c[3]);
        let __a4 = _mm512_unpacklo_epi32(__c[4], __c[5]);
        let __a5 = _mm512_unpackhi_epi32(__c[4], __c[5]);
        let __a6 = _mm512_unpacklo_epi32(__c[6], __c[7]);
        let __a7 = _mm512_unpackhi_epi32(__c[6], __c[7]);
        let __a8 = _mm512_unpacklo_epi32(__c[8], __c[9]);
        let __a9 = _mm512_unpackhi_epi32(__c[8], __c[9]);
        let __a10 = _mm512_unpacklo_epi32(__c[10], __c[11]);
        let __a11 = _mm512_unpackhi_epi32(__c[10], __c[11]);
        let __a12 = _mm512_unpacklo_epi32(__c[12], __c[13]);
        let __a13 = _mm512_unpackhi_epi32(__c[12], __c[13]);
        let __a14 = _mm512_unpacklo_epi32(__c[14], __c[15]);
        let __a15 = _mm512_unpackhi_epi32(__c[14], __c[15]);
        // Stage 2: unpack 64-bit lanes between the stage-1 quartets.
        let __b0 = _mm512_unpacklo_epi64(__a0, __a2);
        let __b1 = _mm512_unpackhi_epi64(__a0, __a2);
        let __b2 = _mm512_unpacklo_epi64(__a1, __a3);
        let __b3 = _mm512_unpackhi_epi64(__a1, __a3);
        let __b4 = _mm512_unpacklo_epi64(__a4, __a6);
        let __b5 = _mm512_unpackhi_epi64(__a4, __a6);
        let __b6 = _mm512_unpacklo_epi64(__a5, __a7);
        let __b7 = _mm512_unpackhi_epi64(__a5, __a7);
        let __b8 = _mm512_unpacklo_epi64(__a8, __a10);
        let __b9 = _mm512_unpackhi_epi64(__a8, __a10);
        let __b10 = _mm512_unpacklo_epi64(__a9, __a11);
        let __b11 = _mm512_unpackhi_epi64(__a9, __a11);
        let __b12 = _mm512_unpacklo_epi64(__a12, __a14);
        let __b13 = _mm512_unpackhi_epi64(__a12, __a14);
        let __b14 = _mm512_unpacklo_epi64(__a13, __a15);
        let __b15 = _mm512_unpackhi_epi64(__a13, __a15);
        // Stage 3: shuffle 128-bit lanes to gather quarters (0,1) vs (2,3).
        // shuffle_i32x4 imm 0x88 = lanes (0,2,0,2), 0xDD = lanes (1,3,1,3).
        let __d0 = _mm512_shuffle_i32x4::<0x88>(__b0, __b4);
        let __d1 = _mm512_shuffle_i32x4::<0x88>(__b1, __b5);
        let __d2 = _mm512_shuffle_i32x4::<0x88>(__b2, __b6);
        let __d3 = _mm512_shuffle_i32x4::<0x88>(__b3, __b7);
        let __d4 = _mm512_shuffle_i32x4::<0xDD>(__b0, __b4);
        let __d5 = _mm512_shuffle_i32x4::<0xDD>(__b1, __b5);
        let __d6 = _mm512_shuffle_i32x4::<0xDD>(__b2, __b6);
        let __d7 = _mm512_shuffle_i32x4::<0xDD>(__b3, __b7);
        let __d8 = _mm512_shuffle_i32x4::<0x88>(__b8, __b12);
        let __d9 = _mm512_shuffle_i32x4::<0x88>(__b9, __b13);
        let __d10 = _mm512_shuffle_i32x4::<0x88>(__b10, __b14);
        let __d11 = _mm512_shuffle_i32x4::<0x88>(__b11, __b15);
        let __d12 = _mm512_shuffle_i32x4::<0xDD>(__b8, __b12);
        let __d13 = _mm512_shuffle_i32x4::<0xDD>(__b9, __b13);
        let __d14 = _mm512_shuffle_i32x4::<0xDD>(__b10, __b14);
        let __d15 = _mm512_shuffle_i32x4::<0xDD>(__b11, __b15);
        // Stage 4: final 128-bit lane interleave between the two halves.
        let __r0 = _mm512_shuffle_i32x4::<0x88>(__d0, __d8);
        let __r1 = _mm512_shuffle_i32x4::<0x88>(__d1, __d9);
        let __r2 = _mm512_shuffle_i32x4::<0x88>(__d2, __d10);
        let __r3 = _mm512_shuffle_i32x4::<0x88>(__d3, __d11);
        let __r4 = _mm512_shuffle_i32x4::<0x88>(__d4, __d12);
        let __r5 = _mm512_shuffle_i32x4::<0x88>(__d5, __d13);
        let __r6 = _mm512_shuffle_i32x4::<0x88>(__d6, __d14);
        let __r7 = _mm512_shuffle_i32x4::<0x88>(__d7, __d15);
        let __r8 = _mm512_shuffle_i32x4::<0xDD>(__d0, __d8);
        let __r9 = _mm512_shuffle_i32x4::<0xDD>(__d1, __d9);
        let __r10 = _mm512_shuffle_i32x4::<0xDD>(__d2, __d10);
        let __r11 = _mm512_shuffle_i32x4::<0xDD>(__d3, __d11);
        let __r12 = _mm512_shuffle_i32x4::<0xDD>(__d4, __d12);
        let __r13 = _mm512_shuffle_i32x4::<0xDD>(__d5, __d13);
        let __r14 = _mm512_shuffle_i32x4::<0xDD>(__d6, __d14);
        let __r15 = _mm512_shuffle_i32x4::<0xDD>(__d7, __d15);
        [
            __r0, __r1, __r2, __r3, __r4, __r5, __r6, __r7, __r8, __r9, __r10, __r11, __r12, __r13,
            __r14, __r15,
        ]
    }};
}

/// AVX-512 helper: apply the intermediate shift/round/clip to a 512-bit i32
/// vector (16 rows). `shift` is dynamic so we branch on the common values.
#[cfg(target_arch = "x86_64")]
#[rite]
#[inline(always)]
fn row_shift_clip_v4(
    _token: Server64,
    v: __m512i,
    rnd_v: __m512i,
    shift: i32,
    col_min_v: __m512i,
    col_max_v: __m512i,
) -> __m512i {
    let rounded = match shift {
        0 => _mm512_add_epi32(v, rnd_v),
        1 => _mm512_srai_epi32::<1>(_mm512_add_epi32(v, rnd_v)),
        2 => _mm512_srai_epi32::<2>(_mm512_add_epi32(v, rnd_v)),
        _ => _mm512_srai_epi32::<2>(_mm512_add_epi32(v, rnd_v)),
    };
    _mm512_max_epi32(_mm512_min_epi32(rounded, col_max_v), col_min_v)
}

/// AVX-512 SIMD row DCT-8 for 8xN transforms, 8bpc, processing 16 rows at once.
/// Mirrors `simd_row_dct8_8bpc_8rows` but uses the 16-lane `dct8_1d_cols16`
/// transform and a 16x16 i32 transpose (8 cols of the row block live in the
/// low 8 lanes of each transposed row). Output stride is 8.
#[cfg(target_arch = "x86_64")]
#[rite]
#[inline(always)]
fn simd_row_dct8_8bpc_16rows(
    token: Server64,
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
    let row_min_v = _mm512_set1_epi32(row_min);
    let row_max_v = _mm512_set1_epi32(row_max);
    let col_min_v = _mm512_set1_epi32(col_min);
    let col_max_v = _mm512_set1_epi32(col_max);
    let rect2_v = _mm512_set1_epi32(181);
    let bias_v = _mm512_set1_epi32(128);
    let rnd_v = _mm512_set1_epi32(rnd);
    // Load 16 rows for each of the 8 columns; lane K = row K.
    let mut cols = [_mm512_setzero_si512(); 16];
    for x in 0..8 {
        let off = y_base + x * coeff_h;
        let arr: &[i16; 16] = (&coeff[off..off + 16]).try_into().unwrap();
        let v16 = loadu_256!(arr);
        let v32 = _mm512_cvtepi16_epi32(v16);
        cols[x] = if apply_rect2 {
            _mm512_srai_epi32::<8>(_mm512_add_epi32(_mm512_mullo_epi32(v32, rect2_v), bias_v))
        } else {
            v32
        };
    }
    let mut dct: [__m512i; 8] = [
        cols[0], cols[1], cols[2], cols[3], cols[4], cols[5], cols[6], cols[7],
    ];
    dct8_1d_cols16(token, &mut dct, row_min_v, row_max_v);
    // Cols 8..15 must be zero for the 16x16 transpose to leave junk only in the
    // high 8 i32 lanes of each output row, which we never store.
    for x in 0..8 {
        cols[x] = row_shift_clip_v4(token, dct[x], rnd_v, shift, col_min_v, col_max_v);
    }
    let rows = transpose_16x16_i32!(cols);
    let s = 8;
    for y in 0..16 {
        // Each output row r holds cols[0..8] in its low 8 i32 lanes.
        let lo = _mm512_castsi512_si256(rows[y]);
        storeu_256!(&mut tmp[(y_base + y) * s..(y_base + y) * s + 8], [i32; 8], lo);
    }
}

/// AVX-512 SIMD row DCT-16 for 16xN transforms, 8bpc, processing 16 rows at
/// once. Mirrors `simd_row_dct16_8bpc_8rows`. 16 cols x 16 rows → 16x16
/// transpose → 16 rows x 16 cols stored row-major (stride 16).
#[cfg(target_arch = "x86_64")]
#[rite]
#[inline(always)]
fn simd_row_dct16_8bpc_16rows(
    token: Server64,
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
    let row_min_v = _mm512_set1_epi32(row_min);
    let row_max_v = _mm512_set1_epi32(row_max);
    let col_min_v = _mm512_set1_epi32(col_min);
    let col_max_v = _mm512_set1_epi32(col_max);
    let rect2_v = _mm512_set1_epi32(181);
    let bias_v = _mm512_set1_epi32(128);
    let rnd_v = _mm512_set1_epi32(rnd);
    let mut cols = [_mm512_setzero_si512(); 16];
    for x in 0..16 {
        let off = y_base + x * coeff_h;
        let arr: &[i16; 16] = (&coeff[off..off + 16]).try_into().unwrap();
        let v16 = loadu_256!(arr);
        let v32 = _mm512_cvtepi16_epi32(v16);
        cols[x] = if apply_rect2 {
            _mm512_srai_epi32::<8>(_mm512_add_epi32(_mm512_mullo_epi32(v32, rect2_v), bias_v))
        } else {
            v32
        };
    }
    dct16_1d_cols16(token, &mut cols, row_min_v, row_max_v);
    for x in 0..16 {
        cols[x] = row_shift_clip_v4(token, cols[x], rnd_v, shift, col_min_v, col_max_v);
    }
    let rows = transpose_16x16_i32!(cols);
    let s = 16;
    for y in 0..16 {
        storeu_512!(
            &mut tmp[(y_base + y) * s..(y_base + y) * s + 16],
            [i32; 16],
            rows[y]
        );
    }
}

/// AVX-512 SIMD row DCT-32 for 32xN transforms, 8bpc, processing 16 rows at
/// once. Mirrors `simd_row_dct32_8bpc_8rows`. 32 cols x 16 rows → two 16x16
/// transposes → 16 rows x 32 cols stored row-major (stride 32).
#[cfg(target_arch = "x86_64")]
#[rite]
#[inline(always)]
fn simd_row_dct32_8bpc_16rows(
    token: Server64,
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
    let row_min_v = _mm512_set1_epi32(row_min);
    let row_max_v = _mm512_set1_epi32(row_max);
    let col_min_v = _mm512_set1_epi32(col_min);
    let col_max_v = _mm512_set1_epi32(col_max);
    let rect2_v = _mm512_set1_epi32(181);
    let bias_v = _mm512_set1_epi32(128);
    let rnd_v = _mm512_set1_epi32(rnd);
    let mut cols = [_mm512_setzero_si512(); 32];
    for x in 0..32 {
        let off = y_base + x * coeff_h;
        let arr: &[i16; 16] = (&coeff[off..off + 16]).try_into().unwrap();
        let v16 = loadu_256!(arr);
        let v32 = _mm512_cvtepi16_epi32(v16);
        cols[x] = if apply_rect2 {
            _mm512_srai_epi32::<8>(_mm512_add_epi32(_mm512_mullo_epi32(v32, rect2_v), bias_v))
        } else {
            v32
        };
    }
    dct32_1d_cols16(token, &mut cols, row_min_v, row_max_v);
    for x in 0..32 {
        cols[x] = row_shift_clip_v4(token, cols[x], rnd_v, shift, col_min_v, col_max_v);
    }
    let s = 32;
    for chunk in 0..2 {
        let b = chunk * 16;
        let chunk_cols: [__m512i; 16] = [
            cols[b], cols[b + 1], cols[b + 2], cols[b + 3], cols[b + 4], cols[b + 5], cols[b + 6],
            cols[b + 7], cols[b + 8], cols[b + 9], cols[b + 10], cols[b + 11], cols[b + 12],
            cols[b + 13], cols[b + 14], cols[b + 15],
        ];
        let rows = transpose_16x16_i32!(chunk_cols);
        for y in 0..16 {
            storeu_512!(
                &mut tmp[(y_base + y) * s + b..(y_base + y) * s + b + 16],
                [i32; 16],
                rows[y]
            );
        }
    }
}

/// `#[arcane]` test/entry shims so the `#[rite]` 16-row row passes are callable
/// from a non-target-feature context (tests, and direct dispatch wiring). These
/// just forward to the inner helper, which inlines (zero cost).
#[cfg(target_arch = "x86_64")]
#[arcane]
#[allow(clippy::too_many_arguments)]
fn simd_row_dct8_8bpc_16rows_entry(
    token: Server64,
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
    simd_row_dct8_8bpc_16rows(
        token, coeff, coeff_h, y_base, apply_rect2, rnd, shift, tmp, row_min, row_max, col_min,
        col_max,
    );
}

#[cfg(target_arch = "x86_64")]
#[arcane]
#[allow(clippy::too_many_arguments)]
fn simd_row_dct16_8bpc_16rows_entry(
    token: Server64,
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
    simd_row_dct16_8bpc_16rows(
        token, coeff, coeff_h, y_base, apply_rect2, rnd, shift, tmp, row_min, row_max, col_min,
        col_max,
    );
}

#[cfg(target_arch = "x86_64")]
#[arcane]
#[allow(clippy::too_many_arguments)]
fn simd_row_dct32_8bpc_16rows_entry(
    token: Server64,
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
    simd_row_dct32_8bpc_16rows(
        token, coeff, coeff_h, y_base, apply_rect2, rnd, shift, tmp, row_min, row_max, col_min,
        col_max,
    );
}

/// DCT4 1D transform across 8 columns in parallel.
/// `c[i]` is `__m256i` containing the 8 lanes for row `i`.
#[cfg(target_arch = "x86_64")]
#[rite]
fn dct4_1d_cols8(token: Desktop64, c: &mut [__m256i; 4], min_v: __m256i, max_v: __m256i) {
    let in0 = c[0];
    let in1 = c[1];
    let in2 = c[2];
    let in3 = c[3];
    // t0 = ((in0 + in2) * 181 + 128) >> 8
    let sum02 = _mm256_add_epi32(in0, in2);
    let sub02 = _mm256_sub_epi32(in0, in2);
    let t0 = _mm256_srai_epi32::<8>(_mm256_add_epi32(
        _mm256_mullo_epi32(sum02, _mm256_set1_epi32(181)),
        _mm256_set1_epi32(128),
    ));
    let t1 = _mm256_srai_epi32::<8>(_mm256_add_epi32(
        _mm256_mullo_epi32(sub02, _mm256_set1_epi32(181)),
        _mm256_set1_epi32(128),
    ));
    // t2 = (in1 * 1567 - in3 * (3784 - 4096) + 2048 >> 12) - in3
    let t2_shifted = mac_msub_shr::<12>(token, in1, 1567, in3, 3784 - 4096, 2048);
    let t2 = _mm256_sub_epi32(t2_shifted, in3);
    // t3 = (in1 * (3784 - 4096) + in3 * 1567 + 2048 >> 12) + in1
    let t3_shifted = mac_madd_shr::<12>(token, in1, 3784 - 4096, in3, 1567, 2048);
    let t3 = _mm256_add_epi32(t3_shifted, in1);

    c[0] = clip8(token, _mm256_add_epi32(t0, t3), min_v, max_v);
    c[1] = clip8(token, _mm256_add_epi32(t1, t2), min_v, max_v);
    c[2] = clip8(token, _mm256_sub_epi32(t1, t2), min_v, max_v);
    c[3] = clip8(token, _mm256_sub_epi32(t0, t3), min_v, max_v);
}

/// DCT8 1D transform across 8 columns in parallel.
#[cfg(target_arch = "x86_64")]
#[rite]
fn dct8_1d_cols8(token: Desktop64, c: &mut [__m256i; 8], min_v: __m256i, max_v: __m256i) {
    // Apply DCT4 to even positions
    let mut even = [c[0], c[2], c[4], c[6]];
    dct4_1d_cols8(token, &mut even, min_v, max_v);
    c[0] = even[0];
    c[2] = even[1];
    c[4] = even[2];
    c[6] = even[3];

    let in1 = c[1];
    let in3 = c[3];
    let in5 = c[5];
    let in7 = c[7];

    // t4a = (in1 * 799 - in7 * (4017 - 4096) + 2048 >> 12) - in7
    let t4a = _mm256_sub_epi32(
        mac_msub_shr::<12>(token, in1, 799, in7, 4017 - 4096, 2048),
        in7,
    );
    // t5a = (in5 * 1703 - in3 * 1138 + 1024) >> 11
    let t5a = mac_msub_shr::<11>(token, in5, 1703, in3, 1138, 1024);
    // t6a = (in5 * 1138 + in3 * 1703 + 1024) >> 11
    let t6a = mac_madd_shr::<11>(token, in5, 1138, in3, 1703, 1024);
    // t7a = (in1 * (4017 - 4096) + in7 * 799 + 2048 >> 12) + in1
    let t7a = _mm256_add_epi32(
        mac_madd_shr::<12>(token, in1, 4017 - 4096, in7, 799, 2048),
        in1,
    );

    let t4 = clip8(token, _mm256_add_epi32(t4a, t5a), min_v, max_v);
    let t5a_n = clip8(token, _mm256_sub_epi32(t4a, t5a), min_v, max_v);
    let t7 = clip8(token, _mm256_add_epi32(t7a, t6a), min_v, max_v);
    let t6a_n = clip8(token, _mm256_sub_epi32(t7a, t6a), min_v, max_v);

    // t5 = ((t6a_n - t5a_n) * 181 + 128) >> 8
    let d = _mm256_sub_epi32(t6a_n, t5a_n);
    let t5 = _mm256_srai_epi32::<8>(_mm256_add_epi32(
        _mm256_mullo_epi32(d, _mm256_set1_epi32(181)),
        _mm256_set1_epi32(128),
    ));
    // t6 = ((t6a_n + t5a_n) * 181 + 128) >> 8
    let s = _mm256_add_epi32(t6a_n, t5a_n);
    let t6 = _mm256_srai_epi32::<8>(_mm256_add_epi32(
        _mm256_mullo_epi32(s, _mm256_set1_epi32(181)),
        _mm256_set1_epi32(128),
    ));

    let t0 = c[0];
    let t1 = c[2];
    let t2 = c[4];
    let t3 = c[6];

    c[0] = clip8(token, _mm256_add_epi32(t0, t7), min_v, max_v);
    c[1] = clip8(token, _mm256_add_epi32(t1, t6), min_v, max_v);
    c[2] = clip8(token, _mm256_add_epi32(t2, t5), min_v, max_v);
    c[3] = clip8(token, _mm256_add_epi32(t3, t4), min_v, max_v);
    c[4] = clip8(token, _mm256_sub_epi32(t3, t4), min_v, max_v);
    c[5] = clip8(token, _mm256_sub_epi32(t2, t5), min_v, max_v);
    c[6] = clip8(token, _mm256_sub_epi32(t1, t6), min_v, max_v);
    c[7] = clip8(token, _mm256_sub_epi32(t0, t7), min_v, max_v);
}

/// ADST8 1D transform across 8 columns in parallel.
#[cfg(target_arch = "x86_64")]
#[rite]
fn adst8_1d_cols8(token: Desktop64, c: &mut [__m256i; 8], min_v: __m256i, max_v: __m256i) {
    let in0 = c[0];
    let in1 = c[1];
    let in2 = c[2];
    let in3 = c[3];
    let in4 = c[4];
    let in5 = c[5];
    let in6 = c[6];
    let in7 = c[7];

    let t0a = _mm256_add_epi32(
        mac_madd_shr::<12>(token, in7, 4076 - 4096, in0, 401, 2048),
        in7,
    );
    let t1a = _mm256_sub_epi32(
        mac_msub_shr::<12>(token, in7, 401, in0, 4076 - 4096, 2048),
        in0,
    );
    let t2a = _mm256_add_epi32(
        mac_madd_shr::<12>(token, in5, 3612 - 4096, in2, 1931, 2048),
        in5,
    );
    let t3a = _mm256_sub_epi32(
        mac_msub_shr::<12>(token, in5, 1931, in2, 3612 - 4096, 2048),
        in2,
    );
    let t4a = mac_madd_shr::<11>(token, in3, 1299, in4, 1583, 1024);
    let t5a = mac_msub_shr::<11>(token, in3, 1583, in4, 1299, 1024);
    let t6a = _mm256_add_epi32(
        mac_madd_shr::<12>(token, in1, 1189, in6, 3920 - 4096, 2048),
        in6,
    );
    let t7a = _mm256_add_epi32(
        mac_msub_shr::<12>(token, in1, 3920 - 4096, in6, 1189, 2048),
        in1,
    );

    let t0 = clip8(token, _mm256_add_epi32(t0a, t4a), min_v, max_v);
    let t1 = clip8(token, _mm256_add_epi32(t1a, t5a), min_v, max_v);
    let t2 = clip8(token, _mm256_add_epi32(t2a, t6a), min_v, max_v);
    let t3 = clip8(token, _mm256_add_epi32(t3a, t7a), min_v, max_v);
    let t4 = clip8(token, _mm256_sub_epi32(t0a, t4a), min_v, max_v);
    let t5 = clip8(token, _mm256_sub_epi32(t1a, t5a), min_v, max_v);
    let t6 = clip8(token, _mm256_sub_epi32(t2a, t6a), min_v, max_v);
    let t7 = clip8(token, _mm256_sub_epi32(t3a, t7a), min_v, max_v);

    let t4a = _mm256_add_epi32(
        mac_madd_shr::<12>(token, t4, 3784 - 4096, t5, 1567, 2048),
        t4,
    );
    let t5a = _mm256_sub_epi32(
        mac_msub_shr::<12>(token, t4, 1567, t5, 3784 - 4096, 2048),
        t5,
    );
    let t6a = _mm256_add_epi32(
        mac_msub_shr::<12>(token, t7, 3784 - 4096, t6, 1567, 2048),
        t7,
    );
    let t7a = _mm256_add_epi32(
        mac_madd_shr::<12>(token, t7, 1567, t6, 3784 - 4096, 2048),
        t6,
    );

    let zero = _mm256_setzero_si256();
    let out0 = clip8(token, _mm256_add_epi32(t0, t2), min_v, max_v);
    let out7 = _mm256_sub_epi32(zero, clip8(token, _mm256_add_epi32(t1, t3), min_v, max_v));
    let t2_final = clip8(token, _mm256_sub_epi32(t0, t2), min_v, max_v);
    let t3_final = clip8(token, _mm256_sub_epi32(t1, t3), min_v, max_v);
    let out1 = _mm256_sub_epi32(zero, clip8(token, _mm256_add_epi32(t4a, t6a), min_v, max_v));
    let out6 = clip8(token, _mm256_add_epi32(t5a, t7a), min_v, max_v);
    let t6_final = clip8(token, _mm256_sub_epi32(t4a, t6a), min_v, max_v);
    let t7_final = clip8(token, _mm256_sub_epi32(t5a, t7a), min_v, max_v);

    let mul181_sum = |a: __m256i, b: __m256i| {
        let s = _mm256_add_epi32(a, b);
        _mm256_srai_epi32::<8>(_mm256_add_epi32(
            _mm256_mullo_epi32(s, _mm256_set1_epi32(181)),
            _mm256_set1_epi32(128),
        ))
    };
    let mul181_diff = |a: __m256i, b: __m256i| {
        let s = _mm256_sub_epi32(a, b);
        _mm256_srai_epi32::<8>(_mm256_add_epi32(
            _mm256_mullo_epi32(s, _mm256_set1_epi32(181)),
            _mm256_set1_epi32(128),
        ))
    };
    let out3 = _mm256_sub_epi32(zero, mul181_sum(t2_final, t3_final));
    let out4 = mul181_diff(t2_final, t3_final);
    let out2 = mul181_sum(t6_final, t7_final);
    let out5 = _mm256_sub_epi32(zero, mul181_diff(t6_final, t7_final));

    c[0] = out0;
    c[1] = out1;
    c[2] = out2;
    c[3] = out3;
    c[4] = out4;
    c[5] = out5;
    c[6] = out6;
    c[7] = out7;
}

/// FlipADST8 1D transform across 8 columns in parallel.
/// Same butterflies as ADST8, then reverse outputs.
#[cfg(target_arch = "x86_64")]
#[rite]
fn flipadst8_1d_cols8(token: Desktop64, c: &mut [__m256i; 8], min_v: __m256i, max_v: __m256i) {
    adst8_1d_cols8(token, c, min_v, max_v);
    let t0 = c[0];
    let t1 = c[1];
    let t2 = c[2];
    let t3 = c[3];
    c[0] = c[7];
    c[1] = c[6];
    c[2] = c[5];
    c[3] = c[4];
    c[4] = t3;
    c[5] = t2;
    c[6] = t1;
    c[7] = t0;
}

/// Identity8 1D transform across 8 columns in parallel: out = in * 2.
#[cfg(target_arch = "x86_64")]
#[rite]
fn identity8_1d_cols8(_token: Desktop64, c: &mut [__m256i; 8], _min_v: __m256i, _max_v: __m256i) {
    for i in 0..8 {
        c[i] = _mm256_slli_epi32::<1>(c[i]);
    }
}

/// DCT16 1D transform across 8 columns in parallel.
#[cfg(target_arch = "x86_64")]
#[rite]
fn dct16_1d_cols8(token: Desktop64, c: &mut [__m256i; 16], min_v: __m256i, max_v: __m256i) {
    // Apply DCT8 to even positions
    let mut even = [c[0], c[2], c[4], c[6], c[8], c[10], c[12], c[14]];
    dct8_1d_cols8(token, &mut even, min_v, max_v);
    c[0] = even[0];
    c[2] = even[1];
    c[4] = even[2];
    c[6] = even[3];
    c[8] = even[4];
    c[10] = even[5];
    c[12] = even[6];
    c[14] = even[7];

    let in1 = c[1];
    let in3 = c[3];
    let in5 = c[5];
    let in7 = c[7];
    let in9 = c[9];
    let in11 = c[11];
    let in13 = c[13];
    let in15 = c[15];

    // t8a  = (in1 * 401 - in15 * (4076 - 4096) + 2048 >> 12) - in15
    let t8a = _mm256_sub_epi32(
        mac_msub_shr::<12>(token, in1, 401, in15, 4076 - 4096, 2048),
        in15,
    );
    // t9a  = (in9 * 1583 - in7 * 1299 + 1024) >> 11
    let t9a = mac_msub_shr::<11>(token, in9, 1583, in7, 1299, 1024);
    // t10a = (in5 * 1931 - in11 * (3612 - 4096) + 2048 >> 12) - in11
    let t10a = _mm256_sub_epi32(
        mac_msub_shr::<12>(token, in5, 1931, in11, 3612 - 4096, 2048),
        in11,
    );
    // t11a = (in13 * (3920 - 4096) - in3 * 1189 + 2048 >> 12) + in13
    let t11a = _mm256_add_epi32(
        mac_msub_shr::<12>(token, in13, 3920 - 4096, in3, 1189, 2048),
        in13,
    );
    // t12a = (in13 * 1189 + in3 * (3920 - 4096) + 2048 >> 12) + in3
    let t12a = _mm256_add_epi32(
        mac_madd_shr::<12>(token, in13, 1189, in3, 3920 - 4096, 2048),
        in3,
    );
    // t13a = (in5 * (3612 - 4096) + in11 * 1931 + 2048 >> 12) + in5
    let t13a = _mm256_add_epi32(
        mac_madd_shr::<12>(token, in5, 3612 - 4096, in11, 1931, 2048),
        in5,
    );
    // t14a = (in9 * 1299 + in7 * 1583 + 1024) >> 11
    let t14a = mac_madd_shr::<11>(token, in9, 1299, in7, 1583, 1024);
    // t15a = (in1 * (4076 - 4096) + in15 * 401 + 2048 >> 12) + in1
    let t15a = _mm256_add_epi32(
        mac_madd_shr::<12>(token, in1, 4076 - 4096, in15, 401, 2048),
        in1,
    );

    let t8 = clip8(token, _mm256_add_epi32(t8a, t9a), min_v, max_v);
    let mut t9 = clip8(token, _mm256_sub_epi32(t8a, t9a), min_v, max_v);
    let mut t10 = clip8(token, _mm256_sub_epi32(t11a, t10a), min_v, max_v);
    let mut t11 = clip8(token, _mm256_add_epi32(t11a, t10a), min_v, max_v);
    let mut t12 = clip8(token, _mm256_add_epi32(t12a, t13a), min_v, max_v);
    let mut t13 = clip8(token, _mm256_sub_epi32(t12a, t13a), min_v, max_v);
    let mut t14 = clip8(token, _mm256_sub_epi32(t15a, t14a), min_v, max_v);
    let t15 = clip8(token, _mm256_add_epi32(t15a, t14a), min_v, max_v);

    // t9a  = (t14 * 1567 - t9 * (3784 - 4096) + 2048 >> 12) - t9
    let t9a = _mm256_sub_epi32(
        mac_msub_shr::<12>(token, t14, 1567, t9, 3784 - 4096, 2048),
        t9,
    );
    // t14a = (t14 * (3784 - 4096) + t9 * 1567 + 2048 >> 12) + t14
    let t14a = _mm256_add_epi32(
        mac_madd_shr::<12>(token, t14, 3784 - 4096, t9, 1567, 2048),
        t14,
    );
    // t10a = (-(t13 * (3784 - 4096) + t10 * 1567) + 2048 >> 12) - t13
    let t10a_inner = _mm256_add_epi32(
        _mm256_mullo_epi32(t13, _mm256_set1_epi32(3784 - 4096)),
        _mm256_mullo_epi32(t10, _mm256_set1_epi32(1567)),
    );
    let t10a = _mm256_sub_epi32(
        _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_sub_epi32(_mm256_setzero_si256(), t10a_inner),
            _mm256_set1_epi32(2048),
        )),
        t13,
    );
    // t13a = (t13 * 1567 - t10 * (3784 - 4096) + 2048 >> 12) - t10
    let t13a = _mm256_sub_epi32(
        mac_msub_shr::<12>(token, t13, 1567, t10, 3784 - 4096, 2048),
        t10,
    );

    let t8a = clip8(token, _mm256_add_epi32(t8, t11), min_v, max_v);
    t9 = clip8(token, _mm256_add_epi32(t9a, t10a), min_v, max_v);
    t10 = clip8(token, _mm256_sub_epi32(t9a, t10a), min_v, max_v);
    let t11a = clip8(token, _mm256_sub_epi32(t8, t11), min_v, max_v);
    let t12a = clip8(token, _mm256_sub_epi32(t15, t12), min_v, max_v);
    t13 = clip8(token, _mm256_sub_epi32(t14a, t13a), min_v, max_v);
    t14 = clip8(token, _mm256_add_epi32(t14a, t13a), min_v, max_v);
    let t15a = clip8(token, _mm256_add_epi32(t15, t12), min_v, max_v);

    // t10a_new = ((t13 - t10) * 181 + 128) >> 8
    let d_13_10 = _mm256_sub_epi32(t13, t10);
    let t10a_new = _mm256_srai_epi32::<8>(_mm256_add_epi32(
        _mm256_mullo_epi32(d_13_10, _mm256_set1_epi32(181)),
        _mm256_set1_epi32(128),
    ));
    // t13a_new = ((t13 + t10) * 181 + 128) >> 8
    let s_13_10 = _mm256_add_epi32(t13, t10);
    let t13a_new = _mm256_srai_epi32::<8>(_mm256_add_epi32(
        _mm256_mullo_epi32(s_13_10, _mm256_set1_epi32(181)),
        _mm256_set1_epi32(128),
    ));
    // t11 = ((t12a - t11a) * 181 + 128) >> 8
    let d_12a_11a = _mm256_sub_epi32(t12a, t11a);
    t11 = _mm256_srai_epi32::<8>(_mm256_add_epi32(
        _mm256_mullo_epi32(d_12a_11a, _mm256_set1_epi32(181)),
        _mm256_set1_epi32(128),
    ));
    // t12 = ((t12a + t11a) * 181 + 128) >> 8
    let s_12a_11a = _mm256_add_epi32(t12a, t11a);
    t12 = _mm256_srai_epi32::<8>(_mm256_add_epi32(
        _mm256_mullo_epi32(s_12a_11a, _mm256_set1_epi32(181)),
        _mm256_set1_epi32(128),
    ));

    let t0 = c[0];
    let t1 = c[2];
    let t2 = c[4];
    let t3 = c[6];
    let t4 = c[8];
    let t5 = c[10];
    let t6 = c[12];
    let t7 = c[14];

    c[0] = clip8(token, _mm256_add_epi32(t0, t15a), min_v, max_v);
    c[1] = clip8(token, _mm256_add_epi32(t1, t14), min_v, max_v);
    c[2] = clip8(token, _mm256_add_epi32(t2, t13a_new), min_v, max_v);
    c[3] = clip8(token, _mm256_add_epi32(t3, t12), min_v, max_v);
    c[4] = clip8(token, _mm256_add_epi32(t4, t11), min_v, max_v);
    c[5] = clip8(token, _mm256_add_epi32(t5, t10a_new), min_v, max_v);
    c[6] = clip8(token, _mm256_add_epi32(t6, t9), min_v, max_v);
    c[7] = clip8(token, _mm256_add_epi32(t7, t8a), min_v, max_v);
    c[8] = clip8(token, _mm256_sub_epi32(t7, t8a), min_v, max_v);
    c[9] = clip8(token, _mm256_sub_epi32(t6, t9), min_v, max_v);
    c[10] = clip8(token, _mm256_sub_epi32(t5, t10a_new), min_v, max_v);
    c[11] = clip8(token, _mm256_sub_epi32(t4, t11), min_v, max_v);
    c[12] = clip8(token, _mm256_sub_epi32(t3, t12), min_v, max_v);
    c[13] = clip8(token, _mm256_sub_epi32(t2, t13a_new), min_v, max_v);
    c[14] = clip8(token, _mm256_sub_epi32(t1, t14), min_v, max_v);
    c[15] = clip8(token, _mm256_sub_epi32(t0, t15a), min_v, max_v);
}

/// Run the row-pass only for a 16x16 transform (scalar 1D row × 16 rows).
/// Leaves `tmp` row-major ready for a column-pass.
#[inline]
fn inv_txfm_16x16_row_pass_only(
    tmp: &mut [i32; 256],
    coeff: &[i16],
    row_transform: fn(&mut [i32], usize, i32, i32),
    row_clip_min: i32,
    row_clip_max: i32,
    col_clip_min: i32,
    col_clip_max: i32,
) {
    let rnd = 2;
    let shift = 2;
    for y in 0..16 {
        let mut scratch = [0i32; 16];
        for x in 0..16 {
            scratch[x] = coeff[y + x * 16] as i32;
        }
        row_transform(&mut scratch[..16], 1, row_clip_min, row_clip_max);
        for x in 0..16 {
            tmp[y * 16 + x] = ((scratch[x] + rnd) >> shift).clamp(col_clip_min, col_clip_max);
        }
    }
}

/// ADST16 1D transform across 8 columns in parallel.
#[cfg(target_arch = "x86_64")]
#[rite]
fn adst16_1d_cols8(token: Desktop64, c: &mut [__m256i; 16], min_v: __m256i, max_v: __m256i) {
    let in0 = c[0];
    let in1 = c[1];
    let in2 = c[2];
    let in3 = c[3];
    let in4 = c[4];
    let in5 = c[5];
    let in6 = c[6];
    let in7 = c[7];
    let in8 = c[8];
    let in9 = c[9];
    let in10 = c[10];
    let in11 = c[11];
    let in12 = c[12];
    let in13 = c[13];
    let in14 = c[14];
    let in15 = c[15];

    // First batch of t[0..16]: each is (a*c1 + b*c2 + 2048) >> 12 ± v
    let mut t0 = _mm256_add_epi32(
        mac_madd_shr::<12>(token, in15, 4091 - 4096, in0, 201, 2048),
        in15,
    );
    let mut t1 = _mm256_sub_epi32(
        mac_msub_shr::<12>(token, in15, 201, in0, 4091 - 4096, 2048),
        in0,
    );
    let mut t2 = _mm256_add_epi32(
        mac_madd_shr::<12>(token, in13, 3973 - 4096, in2, 995, 2048),
        in13,
    );
    let mut t3 = _mm256_sub_epi32(
        mac_msub_shr::<12>(token, in13, 995, in2, 3973 - 4096, 2048),
        in2,
    );
    let mut t4 = _mm256_add_epi32(
        mac_madd_shr::<12>(token, in11, 3703 - 4096, in4, 1751, 2048),
        in11,
    );
    let mut t5 = _mm256_sub_epi32(
        mac_msub_shr::<12>(token, in11, 1751, in4, 3703 - 4096, 2048),
        in4,
    );
    let mut t6 = mac_madd_shr::<11>(token, in9, 1645, in6, 1220, 1024);
    let mut t7 = mac_msub_shr::<11>(token, in9, 1220, in6, 1645, 1024);
    let mut t8 = _mm256_add_epi32(
        mac_madd_shr::<12>(token, in7, 2751, in8, 3035 - 4096, 2048),
        in8,
    );
    let mut t9 = _mm256_add_epi32(
        mac_msub_shr::<12>(token, in7, 3035 - 4096, in8, 2751, 2048),
        in7,
    );
    let mut t10 = _mm256_add_epi32(
        mac_madd_shr::<12>(token, in5, 2106, in10, 3513 - 4096, 2048),
        in10,
    );
    let mut t11 = _mm256_add_epi32(
        mac_msub_shr::<12>(token, in5, 3513 - 4096, in10, 2106, 2048),
        in5,
    );
    let mut t12 = _mm256_add_epi32(
        mac_madd_shr::<12>(token, in3, 1380, in12, 3857 - 4096, 2048),
        in12,
    );
    let mut t13 = _mm256_add_epi32(
        mac_msub_shr::<12>(token, in3, 3857 - 4096, in12, 1380, 2048),
        in3,
    );
    let mut t14 = _mm256_add_epi32(
        mac_madd_shr::<12>(token, in1, 601, in14, 4052 - 4096, 2048),
        in14,
    );
    let mut t15 = _mm256_add_epi32(
        mac_msub_shr::<12>(token, in1, 4052 - 4096, in14, 601, 2048),
        in1,
    );

    let t0a = clip8(token, _mm256_add_epi32(t0, t8), min_v, max_v);
    let t1a = clip8(token, _mm256_add_epi32(t1, t9), min_v, max_v);
    let t2a = clip8(token, _mm256_add_epi32(t2, t10), min_v, max_v);
    let t3a = clip8(token, _mm256_add_epi32(t3, t11), min_v, max_v);
    let mut t4a = clip8(token, _mm256_add_epi32(t4, t12), min_v, max_v);
    let mut t5a = clip8(token, _mm256_add_epi32(t5, t13), min_v, max_v);
    let mut t6a = clip8(token, _mm256_add_epi32(t6, t14), min_v, max_v);
    let mut t7a = clip8(token, _mm256_add_epi32(t7, t15), min_v, max_v);
    let mut t8a = clip8(token, _mm256_sub_epi32(t0, t8), min_v, max_v);
    let mut t9a = clip8(token, _mm256_sub_epi32(t1, t9), min_v, max_v);
    let mut t10a = clip8(token, _mm256_sub_epi32(t2, t10), min_v, max_v);
    let mut t11a = clip8(token, _mm256_sub_epi32(t3, t11), min_v, max_v);
    let mut t12a = clip8(token, _mm256_sub_epi32(t4, t12), min_v, max_v);
    let mut t13a = clip8(token, _mm256_sub_epi32(t5, t13), min_v, max_v);
    let mut t14a = clip8(token, _mm256_sub_epi32(t6, t14), min_v, max_v);
    let mut t15a = clip8(token, _mm256_sub_epi32(t7, t15), min_v, max_v);

    t8 = _mm256_add_epi32(
        mac_madd_shr::<12>(token, t8a, 4017 - 4096, t9a, 799, 2048),
        t8a,
    );
    t9 = _mm256_sub_epi32(
        mac_msub_shr::<12>(token, t8a, 799, t9a, 4017 - 4096, 2048),
        t9a,
    );
    t10 = _mm256_add_epi32(
        mac_madd_shr::<12>(token, t10a, 2276, t11a, 3406 - 4096, 2048),
        t11a,
    );
    t11 = _mm256_add_epi32(
        mac_msub_shr::<12>(token, t10a, 3406 - 4096, t11a, 2276, 2048),
        t10a,
    );
    t12 = _mm256_add_epi32(
        mac_msub_shr::<12>(token, t13a, 4017 - 4096, t12a, 799, 2048),
        t13a,
    );
    t13 = _mm256_add_epi32(
        mac_madd_shr::<12>(token, t13a, 799, t12a, 4017 - 4096, 2048),
        t12a,
    );
    t14 = _mm256_sub_epi32(
        mac_msub_shr::<12>(token, t15a, 2276, t14a, 3406 - 4096, 2048),
        t14a,
    );
    t15 = _mm256_add_epi32(
        mac_madd_shr::<12>(token, t15a, 3406 - 4096, t14a, 2276, 2048),
        t15a,
    );

    t0 = clip8(token, _mm256_add_epi32(t0a, t4a), min_v, max_v);
    t1 = clip8(token, _mm256_add_epi32(t1a, t5a), min_v, max_v);
    t2 = clip8(token, _mm256_add_epi32(t2a, t6a), min_v, max_v);
    t3 = clip8(token, _mm256_add_epi32(t3a, t7a), min_v, max_v);
    t4 = clip8(token, _mm256_sub_epi32(t0a, t4a), min_v, max_v);
    t5 = clip8(token, _mm256_sub_epi32(t1a, t5a), min_v, max_v);
    t6 = clip8(token, _mm256_sub_epi32(t2a, t6a), min_v, max_v);
    t7 = clip8(token, _mm256_sub_epi32(t3a, t7a), min_v, max_v);
    t8a = clip8(token, _mm256_add_epi32(t8, t12), min_v, max_v);
    t9a = clip8(token, _mm256_add_epi32(t9, t13), min_v, max_v);
    t10a = clip8(token, _mm256_add_epi32(t10, t14), min_v, max_v);
    t11a = clip8(token, _mm256_add_epi32(t11, t15), min_v, max_v);
    t12a = clip8(token, _mm256_sub_epi32(t8, t12), min_v, max_v);
    t13a = clip8(token, _mm256_sub_epi32(t9, t13), min_v, max_v);
    t14a = clip8(token, _mm256_sub_epi32(t10, t14), min_v, max_v);
    t15a = clip8(token, _mm256_sub_epi32(t11, t15), min_v, max_v);

    t4a = _mm256_add_epi32(
        mac_madd_shr::<12>(token, t4, 3784 - 4096, t5, 1567, 2048),
        t4,
    );
    t5a = _mm256_sub_epi32(
        mac_msub_shr::<12>(token, t4, 1567, t5, 3784 - 4096, 2048),
        t5,
    );
    t6a = _mm256_add_epi32(
        mac_msub_shr::<12>(token, t7, 3784 - 4096, t6, 1567, 2048),
        t7,
    );
    t7a = _mm256_add_epi32(
        mac_madd_shr::<12>(token, t7, 1567, t6, 3784 - 4096, 2048),
        t6,
    );
    t12 = _mm256_add_epi32(
        mac_madd_shr::<12>(token, t12a, 3784 - 4096, t13a, 1567, 2048),
        t12a,
    );
    t13 = _mm256_sub_epi32(
        mac_msub_shr::<12>(token, t12a, 1567, t13a, 3784 - 4096, 2048),
        t13a,
    );
    t14 = _mm256_add_epi32(
        mac_msub_shr::<12>(token, t15a, 3784 - 4096, t14a, 1567, 2048),
        t15a,
    );
    t15 = _mm256_add_epi32(
        mac_madd_shr::<12>(token, t15a, 1567, t14a, 3784 - 4096, 2048),
        t14a,
    );

    let zero = _mm256_setzero_si256();
    // Outputs (some negated)
    let out0 = clip8(token, _mm256_add_epi32(t0, t2), min_v, max_v);
    let out15 = _mm256_sub_epi32(zero, clip8(token, _mm256_add_epi32(t1, t3), min_v, max_v));
    let t2a_new = clip8(token, _mm256_sub_epi32(t0, t2), min_v, max_v);
    let t3a_new = clip8(token, _mm256_sub_epi32(t1, t3), min_v, max_v);
    let out3 = _mm256_sub_epi32(zero, clip8(token, _mm256_add_epi32(t4a, t6a), min_v, max_v));
    let out12 = clip8(token, _mm256_add_epi32(t5a, t7a), min_v, max_v);
    let t6_new = clip8(token, _mm256_sub_epi32(t4a, t6a), min_v, max_v);
    let t7_new = clip8(token, _mm256_sub_epi32(t5a, t7a), min_v, max_v);
    let out1 = _mm256_sub_epi32(
        zero,
        clip8(token, _mm256_add_epi32(t8a, t10a), min_v, max_v),
    );
    let out14 = clip8(token, _mm256_add_epi32(t9a, t11a), min_v, max_v);
    let t10_new = clip8(token, _mm256_sub_epi32(t8a, t10a), min_v, max_v);
    let t11_new = clip8(token, _mm256_sub_epi32(t9a, t11a), min_v, max_v);
    let out2 = clip8(token, _mm256_add_epi32(t12, t14), min_v, max_v);
    let out13 = _mm256_sub_epi32(zero, clip8(token, _mm256_add_epi32(t13, t15), min_v, max_v));
    let t14_new = clip8(token, _mm256_sub_epi32(t12, t14), min_v, max_v);
    let t15_new = clip8(token, _mm256_sub_epi32(t13, t15), min_v, max_v);

    // ((a +/- b) * 181 + 128) >> 8
    let mul181_sum = |a: __m256i, b: __m256i| {
        let s = _mm256_add_epi32(a, b);
        _mm256_srai_epi32::<8>(_mm256_add_epi32(
            _mm256_mullo_epi32(s, _mm256_set1_epi32(181)),
            _mm256_set1_epi32(128),
        ))
    };
    let mul181_diff = |a: __m256i, b: __m256i| {
        let s = _mm256_sub_epi32(a, b);
        _mm256_srai_epi32::<8>(_mm256_add_epi32(
            _mm256_mullo_epi32(s, _mm256_set1_epi32(181)),
            _mm256_set1_epi32(128),
        ))
    };

    let out7 = _mm256_sub_epi32(zero, mul181_sum(t2a_new, t3a_new));
    let out8 = mul181_diff(t2a_new, t3a_new);
    let out4 = mul181_sum(t6_new, t7_new);
    let out11 = _mm256_sub_epi32(zero, mul181_diff(t6_new, t7_new));
    let out6 = mul181_sum(t10_new, t11_new);
    let out9 = _mm256_sub_epi32(zero, mul181_diff(t10_new, t11_new));
    let out5 = _mm256_sub_epi32(zero, mul181_sum(t14_new, t15_new));
    let out10 = mul181_diff(t14_new, t15_new);

    c[0] = out0;
    c[1] = out1;
    c[2] = out2;
    c[3] = out3;
    c[4] = out4;
    c[5] = out5;
    c[6] = out6;
    c[7] = out7;
    c[8] = out8;
    c[9] = out9;
    c[10] = out10;
    c[11] = out11;
    c[12] = out12;
    c[13] = out13;
    c[14] = out14;
    c[15] = out15;
}

/// Run flipadst16 1D column transform: adst16 then swap rows i ↔ 15-i.
/// `#[rite]` so it inlines into matching-feature `#[arcane]` callers (zero call cost).
#[cfg(target_arch = "x86_64")]
#[rite]
fn flipadst16x16_cols_simd(token: Desktop64, tmp: &mut [i32; 256], min: i32, max: i32) {
    let min_v = _mm256_set1_epi32(min);
    let max_v = _mm256_set1_epi32(max);
    for cx_chunk in 0..2 {
        let cx = cx_chunk * 8;
        let mut v = [_mm256_setzero_si256(); 16];
        for i in 0..16 {
            v[i] = loadu_256!(&tmp[i * 16 + cx..i * 16 + cx + 8], [i32; 8]);
        }
        adst16_1d_cols8(token, &mut v, min_v, max_v);
        // Swap rows: i <-> 15-i for i in 0..8
        for i in 0..8 {
            v.swap(i, 15 - i);
        }
        for i in 0..16 {
            storeu_256!(&mut tmp[i * 16 + cx..i * 16 + cx + 8], [i32; 8], v[i]);
        }
    }
}

/// Run identity16 1D column transform over 16x16 row-major buffer (8 cols at a time).
/// identity16: out = 2*in + ((in*1697 + 1024) >> 11). No clipping.
/// `#[rite]` so it inlines into matching-feature `#[arcane]` callers (zero call cost).
#[cfg(target_arch = "x86_64")]
#[rite]
fn identity16x16_cols_simd(_token: Desktop64, tmp: &mut [i32; 256], _min: i32, _max: i32) {
    let c1697 = _mm256_set1_epi32(1697);
    let c1024 = _mm256_set1_epi32(1024);
    for cx_chunk in 0..2 {
        let cx = cx_chunk * 8;
        for i in 0..16 {
            let v = loadu_256!(&tmp[i * 16 + cx..i * 16 + cx + 8], [i32; 8]);
            // 2*v + ((v*1697 + 1024) >> 11)
            let two_v = _mm256_slli_epi32::<1>(v);
            let mul = _mm256_mullo_epi32(v, c1697);
            let shifted = _mm256_srai_epi32::<11>(_mm256_add_epi32(mul, c1024));
            let result = _mm256_add_epi32(two_v, shifted);
            storeu_256!(&mut tmp[i * 16 + cx..i * 16 + cx + 8], [i32; 8], result);
        }
    }
}

/// Run adst16 1D column transform over 16x16 row-major buffer.
/// `#[rite]` so it inlines into matching-feature `#[arcane]` callers (zero call cost).
#[cfg(target_arch = "x86_64")]
#[rite]
fn adst16x16_cols_simd(token: Desktop64, tmp: &mut [i32; 256], min: i32, max: i32) {
    let min_v = _mm256_set1_epi32(min);
    let max_v = _mm256_set1_epi32(max);
    for cx_chunk in 0..2 {
        let cx = cx_chunk * 8;
        let mut v = [_mm256_setzero_si256(); 16];
        for i in 0..16 {
            v[i] = loadu_256!(&tmp[i * 16 + cx..i * 16 + cx + 8], [i32; 8]);
        }
        adst16_1d_cols8(token, &mut v, min_v, max_v);
        for i in 0..16 {
            storeu_256!(&mut tmp[i * 16 + cx..i * 16 + cx + 8], [i32; 8], v[i]);
        }
    }
}

/// Run dct16 1D column transform over 16x16 row-major buffer (16 cols, 16 rows).
/// Processes 8 cols at a time, twice.
/// `#[rite]` so it inlines into matching-feature `#[arcane]` callers (zero call cost).
#[cfg(target_arch = "x86_64")]
#[rite]
fn dct16x16_cols_simd(token: Desktop64, tmp: &mut [i32; 256], min: i32, max: i32) {
    // Try AVX-512 first: processes 16 cols at once instead of 8.
    if let Some(t512) = crate::src::cpu::summon_avx512() {
        dct16_cols_avx512(t512, tmp, 16, 16, min, max);
        return;
    }
    let min_v = _mm256_set1_epi32(min);
    let max_v = _mm256_set1_epi32(max);
    for cx_chunk in 0..2 {
        let cx = cx_chunk * 8;
        let mut v = [_mm256_setzero_si256(); 16];
        for i in 0..16 {
            v[i] = loadu_256!(&tmp[i * 16 + cx..i * 16 + cx + 8], [i32; 8]);
        }
        dct16_1d_cols8(token, &mut v, min_v, max_v);
        for i in 0..16 {
            storeu_256!(&mut tmp[i * 16 + cx..i * 16 + cx + 8], [i32; 8], v[i]);
        }
    }
}

/// DCT32 1D transform across 8 columns in parallel.
#[cfg(target_arch = "x86_64")]
#[rite]
fn dct32_1d_cols8(token: Desktop64, c: &mut [__m256i; 32], min_v: __m256i, max_v: __m256i) {
    // Apply DCT16 to even positions
    let mut even = [
        c[0], c[2], c[4], c[6], c[8], c[10], c[12], c[14], c[16], c[18], c[20], c[22], c[24],
        c[26], c[28], c[30],
    ];
    dct16_1d_cols8(token, &mut even, min_v, max_v);
    c[0] = even[0];
    c[2] = even[1];
    c[4] = even[2];
    c[6] = even[3];
    c[8] = even[4];
    c[10] = even[5];
    c[12] = even[6];
    c[14] = even[7];
    c[16] = even[8];
    c[18] = even[9];
    c[20] = even[10];
    c[22] = even[11];
    c[24] = even[12];
    c[26] = even[13];
    c[28] = even[14];
    c[30] = even[15];

    let in1 = c[1];
    let in3 = c[3];
    let in5 = c[5];
    let in7 = c[7];
    let in9 = c[9];
    let in11 = c[11];
    let in13 = c[13];
    let in15 = c[15];
    let in17 = c[17];
    let in19 = c[19];
    let in21 = c[21];
    let in23 = c[23];
    let in25 = c[25];
    let in27 = c[27];
    let in29 = c[29];
    let in31 = c[31];

    // First batch: 16 t*a values, each (a * c1 ± b * c2 + 2048 >> 12) ± op
    let t16a = _mm256_sub_epi32(
        mac_msub_shr::<12>(token, in1, 201, in31, 4091 - 4096, 2048),
        in31,
    );
    let t17a = _mm256_add_epi32(
        mac_msub_shr::<12>(token, in17, 3035 - 4096, in15, 2751, 2048),
        in17,
    );
    let t18a = _mm256_sub_epi32(
        mac_msub_shr::<12>(token, in9, 1751, in23, 3703 - 4096, 2048),
        in23,
    );
    let t19a = _mm256_add_epi32(
        mac_msub_shr::<12>(token, in25, 3857 - 4096, in7, 1380, 2048),
        in25,
    );
    let t20a = _mm256_sub_epi32(
        mac_msub_shr::<12>(token, in5, 995, in27, 3973 - 4096, 2048),
        in27,
    );
    let t21a = _mm256_add_epi32(
        mac_msub_shr::<12>(token, in21, 3513 - 4096, in11, 2106, 2048),
        in21,
    );
    let t22a = mac_msub_shr::<11>(token, in13, 1220, in19, 1645, 1024);
    let t23a = _mm256_add_epi32(
        mac_msub_shr::<12>(token, in29, 4052 - 4096, in3, 601, 2048),
        in29,
    );
    let t24a = _mm256_add_epi32(
        mac_madd_shr::<12>(token, in29, 601, in3, 4052 - 4096, 2048),
        in3,
    );
    let t25a = mac_madd_shr::<11>(token, in13, 1645, in19, 1220, 1024);
    let t26a = _mm256_add_epi32(
        mac_madd_shr::<12>(token, in21, 2106, in11, 3513 - 4096, 2048),
        in11,
    );
    let t27a = _mm256_add_epi32(
        mac_madd_shr::<12>(token, in5, 3973 - 4096, in27, 995, 2048),
        in5,
    );
    let t28a = _mm256_add_epi32(
        mac_madd_shr::<12>(token, in25, 1380, in7, 3857 - 4096, 2048),
        in7,
    );
    let t29a = _mm256_add_epi32(
        mac_madd_shr::<12>(token, in9, 3703 - 4096, in23, 1751, 2048),
        in9,
    );
    let t30a = _mm256_add_epi32(
        mac_madd_shr::<12>(token, in17, 2751, in15, 3035 - 4096, 2048),
        in15,
    );
    let t31a = _mm256_add_epi32(
        mac_madd_shr::<12>(token, in1, 4091 - 4096, in31, 201, 2048),
        in1,
    );

    let mut t16 = clip8(token, _mm256_add_epi32(t16a, t17a), min_v, max_v);
    let mut t17 = clip8(token, _mm256_sub_epi32(t16a, t17a), min_v, max_v);
    let mut t18 = clip8(token, _mm256_sub_epi32(t19a, t18a), min_v, max_v);
    let t19 = clip8(token, _mm256_add_epi32(t19a, t18a), min_v, max_v);
    let t20 = clip8(token, _mm256_add_epi32(t20a, t21a), min_v, max_v);
    let mut t21 = clip8(token, _mm256_sub_epi32(t20a, t21a), min_v, max_v);
    let mut t22 = clip8(token, _mm256_sub_epi32(t23a, t22a), min_v, max_v);
    let mut t23 = clip8(token, _mm256_add_epi32(t23a, t22a), min_v, max_v);
    let mut t24 = clip8(token, _mm256_add_epi32(t24a, t25a), min_v, max_v);
    let mut t25 = clip8(token, _mm256_sub_epi32(t24a, t25a), min_v, max_v);
    let mut t26 = clip8(token, _mm256_sub_epi32(t27a, t26a), min_v, max_v);
    let t27 = clip8(token, _mm256_add_epi32(t27a, t26a), min_v, max_v);
    let t28 = clip8(token, _mm256_add_epi32(t28a, t29a), min_v, max_v);
    let mut t29 = clip8(token, _mm256_sub_epi32(t28a, t29a), min_v, max_v);
    let mut t30 = clip8(token, _mm256_sub_epi32(t31a, t30a), min_v, max_v);
    let mut t31 = clip8(token, _mm256_add_epi32(t31a, t30a), min_v, max_v);

    // Second batch
    let t17a = _mm256_sub_epi32(
        mac_msub_shr::<12>(token, t30, 799, t17, 4017 - 4096, 2048),
        t17,
    );
    let t30a = _mm256_add_epi32(
        mac_madd_shr::<12>(token, t30, 4017 - 4096, t17, 799, 2048),
        t30,
    );
    // t18a = ((-(t29 * (4017 - 4096) + t18 * 799) + 2048) >> 12) - t29
    let t18a_inner = _mm256_add_epi32(
        _mm256_mullo_epi32(t29, _mm256_set1_epi32(4017 - 4096)),
        _mm256_mullo_epi32(t18, _mm256_set1_epi32(799)),
    );
    let t18a = _mm256_sub_epi32(
        _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_sub_epi32(_mm256_setzero_si256(), t18a_inner),
            _mm256_set1_epi32(2048),
        )),
        t29,
    );
    let t29a = _mm256_sub_epi32(
        mac_msub_shr::<12>(token, t29, 799, t18, 4017 - 4096, 2048),
        t18,
    );
    let t21a = mac_msub_shr::<11>(token, t26, 1703, t21, 1138, 1024);
    let t26a = mac_madd_shr::<11>(token, t26, 1138, t21, 1703, 1024);
    // t22a = (-(t25 * 1138 + t22 * 1703) + 1024) >> 11
    let t22a_inner = _mm256_add_epi32(
        _mm256_mullo_epi32(t25, _mm256_set1_epi32(1138)),
        _mm256_mullo_epi32(t22, _mm256_set1_epi32(1703)),
    );
    let t22a = _mm256_srai_epi32::<11>(_mm256_add_epi32(
        _mm256_sub_epi32(_mm256_setzero_si256(), t22a_inner),
        _mm256_set1_epi32(1024),
    ));
    let t25a = mac_msub_shr::<11>(token, t25, 1703, t22, 1138, 1024);

    let t16a = clip8(token, _mm256_add_epi32(t16, t19), min_v, max_v);
    t17 = clip8(token, _mm256_add_epi32(t17a, t18a), min_v, max_v);
    t18 = clip8(token, _mm256_sub_epi32(t17a, t18a), min_v, max_v);
    let t19a = clip8(token, _mm256_sub_epi32(t16, t19), min_v, max_v);
    let t20a = clip8(token, _mm256_sub_epi32(t23, t20), min_v, max_v);
    t21 = clip8(token, _mm256_sub_epi32(t22a, t21a), min_v, max_v);
    t22 = clip8(token, _mm256_add_epi32(t22a, t21a), min_v, max_v);
    let t23a = clip8(token, _mm256_add_epi32(t23, t20), min_v, max_v);
    let t24a = clip8(token, _mm256_add_epi32(t24, t27), min_v, max_v);
    t25 = clip8(token, _mm256_add_epi32(t25a, t26a), min_v, max_v);
    t26 = clip8(token, _mm256_sub_epi32(t25a, t26a), min_v, max_v);
    let t27a = clip8(token, _mm256_sub_epi32(t24, t27), min_v, max_v);
    let t28a = clip8(token, _mm256_sub_epi32(t31, t28), min_v, max_v);
    t29 = clip8(token, _mm256_sub_epi32(t30a, t29a), min_v, max_v);
    t30 = clip8(token, _mm256_add_epi32(t30a, t29a), min_v, max_v);
    let t31a = clip8(token, _mm256_add_epi32(t31, t28), min_v, max_v);

    // Third batch
    let t18a = _mm256_sub_epi32(
        mac_msub_shr::<12>(token, t29, 1567, t18, 3784 - 4096, 2048),
        t18,
    );
    let t29a = _mm256_add_epi32(
        mac_madd_shr::<12>(token, t29, 3784 - 4096, t18, 1567, 2048),
        t29,
    );
    let t19 = _mm256_sub_epi32(
        mac_msub_shr::<12>(token, t28a, 1567, t19a, 3784 - 4096, 2048),
        t19a,
    );
    let t28 = _mm256_add_epi32(
        mac_madd_shr::<12>(token, t28a, 3784 - 4096, t19a, 1567, 2048),
        t28a,
    );
    // t20 = ((-(t27a * (3784 - 4096) + t20a * 1567) + 2048) >> 12) - t27a
    let t20_inner = _mm256_add_epi32(
        _mm256_mullo_epi32(t27a, _mm256_set1_epi32(3784 - 4096)),
        _mm256_mullo_epi32(t20a, _mm256_set1_epi32(1567)),
    );
    let t20 = _mm256_sub_epi32(
        _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_sub_epi32(_mm256_setzero_si256(), t20_inner),
            _mm256_set1_epi32(2048),
        )),
        t27a,
    );
    let t27 = _mm256_sub_epi32(
        mac_msub_shr::<12>(token, t27a, 1567, t20a, 3784 - 4096, 2048),
        t20a,
    );
    // t21a = ((-(t26 * (3784 - 4096) + t21 * 1567) + 2048) >> 12) - t26
    let t21a_inner = _mm256_add_epi32(
        _mm256_mullo_epi32(t26, _mm256_set1_epi32(3784 - 4096)),
        _mm256_mullo_epi32(t21, _mm256_set1_epi32(1567)),
    );
    let t21a = _mm256_sub_epi32(
        _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_sub_epi32(_mm256_setzero_si256(), t21a_inner),
            _mm256_set1_epi32(2048),
        )),
        t26,
    );
    let t26a = _mm256_sub_epi32(
        mac_msub_shr::<12>(token, t26, 1567, t21, 3784 - 4096, 2048),
        t21,
    );

    t16 = clip8(token, _mm256_add_epi32(t16a, t23a), min_v, max_v);
    let t17a = clip8(token, _mm256_add_epi32(t17, t22), min_v, max_v);
    t18 = clip8(token, _mm256_add_epi32(t18a, t21a), min_v, max_v);
    let t19a = clip8(token, _mm256_add_epi32(t19, t20), min_v, max_v);
    let t20a = clip8(token, _mm256_sub_epi32(t19, t20), min_v, max_v);
    t21 = clip8(token, _mm256_sub_epi32(t18a, t21a), min_v, max_v);
    let t22a = clip8(token, _mm256_sub_epi32(t17, t22), min_v, max_v);
    t23 = clip8(token, _mm256_sub_epi32(t16a, t23a), min_v, max_v);
    t24 = clip8(token, _mm256_sub_epi32(t31a, t24a), min_v, max_v);
    let t25a = clip8(token, _mm256_sub_epi32(t30, t25), min_v, max_v);
    t26 = clip8(token, _mm256_sub_epi32(t29a, t26a), min_v, max_v);
    let t27a = clip8(token, _mm256_sub_epi32(t28, t27), min_v, max_v);
    let t28a = clip8(token, _mm256_add_epi32(t28, t27), min_v, max_v);
    t29 = clip8(token, _mm256_add_epi32(t29a, t26a), min_v, max_v);
    let t30a = clip8(token, _mm256_add_epi32(t30, t25), min_v, max_v);
    t31 = clip8(token, _mm256_add_epi32(t31a, t24a), min_v, max_v);

    // Final mix using 181 constant: ((a ± b) * 181 + 128) >> 8
    let mul181 = |a: __m256i, b: __m256i| -> __m256i {
        let s = _mm256_add_epi32(a, b);
        _mm256_srai_epi32::<8>(_mm256_add_epi32(
            _mm256_mullo_epi32(s, _mm256_set1_epi32(181)),
            _mm256_set1_epi32(128),
        ))
    };
    let mul181_neg = |a: __m256i, b: __m256i| -> __m256i {
        let s = _mm256_sub_epi32(a, b);
        _mm256_srai_epi32::<8>(_mm256_add_epi32(
            _mm256_mullo_epi32(s, _mm256_set1_epi32(181)),
            _mm256_set1_epi32(128),
        ))
    };
    let t20_final = mul181_neg(t27a, t20a);
    let t27_final = mul181(t27a, t20a);
    let t21a_final = mul181_neg(t26, t21);
    let t26a_final = mul181(t26, t21);
    let t22_final = mul181_neg(t25a, t22a);
    let t25_final = mul181(t25a, t22a);
    let t23a = mul181_neg(t24, t23);
    let t24a = mul181(t24, t23);

    // Pull out even-position outputs (already updated by dct16)
    let t0 = c[0];
    let t1 = c[2];
    let t2 = c[4];
    let t3 = c[6];
    let t4 = c[8];
    let t5 = c[10];
    let t6 = c[12];
    let t7 = c[14];
    let t8 = c[16];
    let t9 = c[18];
    let t10 = c[20];
    let t11 = c[22];
    let t12 = c[24];
    let t13 = c[26];
    let t14 = c[28];
    let t15 = c[30];

    c[0] = clip8(token, _mm256_add_epi32(t0, t31), min_v, max_v);
    c[1] = clip8(token, _mm256_add_epi32(t1, t30a), min_v, max_v);
    c[2] = clip8(token, _mm256_add_epi32(t2, t29), min_v, max_v);
    c[3] = clip8(token, _mm256_add_epi32(t3, t28a), min_v, max_v);
    c[4] = clip8(token, _mm256_add_epi32(t4, t27_final), min_v, max_v);
    c[5] = clip8(token, _mm256_add_epi32(t5, t26a_final), min_v, max_v);
    c[6] = clip8(token, _mm256_add_epi32(t6, t25_final), min_v, max_v);
    c[7] = clip8(token, _mm256_add_epi32(t7, t24a), min_v, max_v);
    c[8] = clip8(token, _mm256_add_epi32(t8, t23a), min_v, max_v);
    c[9] = clip8(token, _mm256_add_epi32(t9, t22_final), min_v, max_v);
    c[10] = clip8(token, _mm256_add_epi32(t10, t21a_final), min_v, max_v);
    c[11] = clip8(token, _mm256_add_epi32(t11, t20_final), min_v, max_v);
    c[12] = clip8(token, _mm256_add_epi32(t12, t19a), min_v, max_v);
    c[13] = clip8(token, _mm256_add_epi32(t13, t18), min_v, max_v);
    c[14] = clip8(token, _mm256_add_epi32(t14, t17a), min_v, max_v);
    c[15] = clip8(token, _mm256_add_epi32(t15, t16), min_v, max_v);
    c[16] = clip8(token, _mm256_sub_epi32(t15, t16), min_v, max_v);
    c[17] = clip8(token, _mm256_sub_epi32(t14, t17a), min_v, max_v);
    c[18] = clip8(token, _mm256_sub_epi32(t13, t18), min_v, max_v);
    c[19] = clip8(token, _mm256_sub_epi32(t12, t19a), min_v, max_v);
    c[20] = clip8(token, _mm256_sub_epi32(t11, t20_final), min_v, max_v);
    c[21] = clip8(token, _mm256_sub_epi32(t10, t21a_final), min_v, max_v);
    c[22] = clip8(token, _mm256_sub_epi32(t9, t22_final), min_v, max_v);
    c[23] = clip8(token, _mm256_sub_epi32(t8, t23a), min_v, max_v);
    c[24] = clip8(token, _mm256_sub_epi32(t7, t24a), min_v, max_v);
    c[25] = clip8(token, _mm256_sub_epi32(t6, t25_final), min_v, max_v);
    c[26] = clip8(token, _mm256_sub_epi32(t5, t26a_final), min_v, max_v);
    c[27] = clip8(token, _mm256_sub_epi32(t4, t27_final), min_v, max_v);
    c[28] = clip8(token, _mm256_sub_epi32(t3, t28a), min_v, max_v);
    c[29] = clip8(token, _mm256_sub_epi32(t2, t29), min_v, max_v);
    c[30] = clip8(token, _mm256_sub_epi32(t1, t30a), min_v, max_v);
    c[31] = clip8(token, _mm256_sub_epi32(t0, t31), min_v, max_v);
}

// =============================================================================
// i16-packed pmaddwd column transforms: DCT-4/8/16/32 with pmaddwd butterflies
// =============================================================================
//
// All values entering the column pass are clipped to i16 range after the row
// pass + shift + clip. This means we can pack i32 → i16 and use pmaddwd
// (madd_epi16) for the trig butterflies instead of mullo_epi32. pmaddwd is
// ~2× faster than mullo on Zen3/4 (5c vs 10c latency, 1c vs 2c throughput).
//
// Pattern per pair (a, b):
//   pair = i32_to_i16_pair(a, b)                    // 3 ops: 2 packs + 1 unpack
//   result_add = pmaddwd(pair, coef(c1,  c2))       // 1 op
//   result_sub = pmaddwd(pair, coef(c1, -c2))       // 1 op (reuses pair)
// vs mullo approach:
//   p1 = mullo(a, c1_v); p2 = mullo(b, c2_v); sum/diff = p1 ± p2;  // 3 ops per result
//
// Net: 2 results from 5 ops (3 pack + 2 pmaddwd) vs 6 ops (4 mullo + 2 add/sub).

/// Pack two i32 vectors (assumed in i16 range) into interleaved i16 pairs for pmaddwd.
/// Each 32-bit lane K of the result contains (a[K] as i16, b[K] as i16) packed as
/// two i16 halves, ready for `_mm256_madd_epi16(pair, coef_pack(c_lo, c_hi))`.
#[cfg(target_arch = "x86_64")]
#[rite]
#[inline(always)]
fn i32_to_i16_pair(_token: Desktop64, a: __m256i, b: __m256i) -> __m256i {
    // PRECONDITION: every i32 lane of `a` and `b` is already within i16 range.
    // `_mm_packs_epi32` saturates out-of-range values silently — a violation
    // would produce wrong pixels with no panic. Validate in debug builds so
    // future callers that feed unclipped data fail loudly instead.
    #[cfg(debug_assertions)]
    {
        let mut buf = [0i32; 16];
        let lo: &mut [i32; 8] = (&mut buf[0..8]).try_into().unwrap();
        storeu_256!(lo, [i32; 8], a);
        let hi: &mut [i32; 8] = (&mut buf[8..16]).try_into().unwrap();
        storeu_256!(hi, [i32; 8], b);
        for &v in buf.iter() {
            debug_assert!(
                v >= i16::MIN as i32 && v <= i16::MAX as i32,
                "i32_to_i16_pair: lane {v} outside i16 range — _mm_packs_epi32 would \
                 silently saturate and corrupt output. Caller must clip before packing."
            );
        }
    }
    let a_i16 = _mm_packs_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
    let b_i16 = _mm_packs_epi32(_mm256_castsi256_si128(b), _mm256_extracti128_si256(b, 1));
    _mm256_set_m128i(
        _mm_unpackhi_epi16(a_i16, b_i16),
        _mm_unpacklo_epi16(a_i16, b_i16),
    )
}

/// DCT-4 column transform using pmaddwd for the trig multiply.
/// Input/output: 4 × __m256i (8 i32 lanes each, values in i16 range).
#[cfg(target_arch = "x86_64")]
#[rite]
fn dct4_1d_cols8_i16(token: Desktop64, c: &mut [__m256i; 4], min_v: __m256i, max_v: __m256i) {
    let pd_2048 = _mm256_set1_epi32(2048);
    let pd_128 = _mm256_set1_epi32(128);

    // t0 = ((in0 + in2) * 181 + 128) >> 8
    // t1 = ((in0 - in2) * 181 + 128) >> 8
    // Use pmaddwd: pair_02 = (in0, in2), coef_add = (181, 181), coef_sub = (181, -181)
    let pair_02 = i32_to_i16_pair(token, c[0], c[2]);
    let t0 = _mm256_srai_epi32::<8>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_02, dct8_row_coef_pack(token, 181, 181)),
        pd_128,
    ));
    let t1 = _mm256_srai_epi32::<8>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_02, dct8_row_coef_pack(token, 181, -181)),
        pd_128,
    ));

    // t2 = (in1 * 1567 - in3 * 3784 + 2048) >> 12
    // t3 = (in1 * 3784 + in3 * 1567 + 2048) >> 12
    let pair_13 = i32_to_i16_pair(token, c[1], c[3]);
    let t2 = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_13, dct8_row_coef_pack(token, 1567, -3784)),
        pd_2048,
    ));
    let t3 = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_13, dct8_row_coef_pack(token, 3784, 1567)),
        pd_2048,
    ));

    c[0] = clip8(token, _mm256_add_epi32(t0, t3), min_v, max_v);
    c[1] = clip8(token, _mm256_add_epi32(t1, t2), min_v, max_v);
    c[2] = clip8(token, _mm256_sub_epi32(t1, t2), min_v, max_v);
    c[3] = clip8(token, _mm256_sub_epi32(t0, t3), min_v, max_v);
}

/// DCT-8 column transform using pmaddwd for all trig multiplies.
/// Input/output: 8 × __m256i (8 i32 lanes each, values in i16 range).
#[cfg(target_arch = "x86_64")]
#[rite]
fn dct8_1d_cols8_i16(token: Desktop64, c: &mut [__m256i; 8], min_v: __m256i, max_v: __m256i) {
    // Even half: DCT-4 on positions 0,2,4,6
    let mut even = [c[0], c[2], c[4], c[6]];
    dct4_1d_cols8_i16(token, &mut even, min_v, max_v);
    c[0] = even[0];
    c[2] = even[1];
    c[4] = even[2];
    c[6] = even[3];

    let pd_2048 = _mm256_set1_epi32(2048);
    let pd_128 = _mm256_set1_epi32(128);

    // Odd half: 4 inputs → 4 outputs via 2 pairs
    // t4a = (in1 * 799  - in7 * 4017 + 2048) >> 12
    // t7a = (in1 * 4017 + in7 * 799  + 2048) >> 12
    let pair_17 = i32_to_i16_pair(token, c[1], c[7]);
    let t4a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_17, dct8_row_coef_pack(token, 799, -4017)),
        pd_2048,
    ));
    let t7a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_17, dct8_row_coef_pack(token, 4017, 799)),
        pd_2048,
    ));

    // t5a = (in5 * 3406 - in3 * 2276 + 2048) >> 12
    // t6a = (in5 * 2276 + in3 * 3406 + 2048) >> 12
    let pair_53 = i32_to_i16_pair(token, c[5], c[3]);
    let t5a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_53, dct8_row_coef_pack(token, 3406, -2276)),
        pd_2048,
    ));
    let t6a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_53, dct8_row_coef_pack(token, 2276, 3406)),
        pd_2048,
    ));

    let t4 = clip8(token, _mm256_add_epi32(t4a, t5a), min_v, max_v);
    let t5a_n = clip8(token, _mm256_sub_epi32(t4a, t5a), min_v, max_v);
    let t7 = clip8(token, _mm256_add_epi32(t7a, t6a), min_v, max_v);
    let t6a_n = clip8(token, _mm256_sub_epi32(t7a, t6a), min_v, max_v);

    // t5 = ((t6a_n - t5a_n) * 181 + 128) >> 8
    // t6 = ((t6a_n + t5a_n) * 181 + 128) >> 8
    // Use pmaddwd: pair = (t6a_n, t5a_n)
    let pair_65 = i32_to_i16_pair(token, t6a_n, t5a_n);
    let t5 = _mm256_srai_epi32::<8>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_65, dct8_row_coef_pack(token, 181, -181)),
        pd_128,
    ));
    let t6 = _mm256_srai_epi32::<8>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_65, dct8_row_coef_pack(token, 181, 181)),
        pd_128,
    ));

    let t0 = c[0];
    let t1 = c[2];
    let t2 = c[4];
    let t3 = c[6];

    c[0] = clip8(token, _mm256_add_epi32(t0, t7), min_v, max_v);
    c[1] = clip8(token, _mm256_add_epi32(t1, t6), min_v, max_v);
    c[2] = clip8(token, _mm256_add_epi32(t2, t5), min_v, max_v);
    c[3] = clip8(token, _mm256_add_epi32(t3, t4), min_v, max_v);
    c[4] = clip8(token, _mm256_sub_epi32(t3, t4), min_v, max_v);
    c[5] = clip8(token, _mm256_sub_epi32(t2, t5), min_v, max_v);
    c[6] = clip8(token, _mm256_sub_epi32(t1, t6), min_v, max_v);
    c[7] = clip8(token, _mm256_sub_epi32(t0, t7), min_v, max_v);
}

/// DCT-16 column transform using pmaddwd for all trig multiplies.
/// Input/output: 16 × __m256i (8 i32 lanes each, values in i16 range).
#[cfg(target_arch = "x86_64")]
#[rite]
fn dct16_1d_cols8_i16(token: Desktop64, c: &mut [__m256i; 16], min_v: __m256i, max_v: __m256i) {
    // Even half: DCT-8 on positions 0,2,4,...,14
    let mut even = [c[0], c[2], c[4], c[6], c[8], c[10], c[12], c[14]];
    dct8_1d_cols8_i16(token, &mut even, min_v, max_v);
    c[0] = even[0];
    c[2] = even[1];
    c[4] = even[2];
    c[6] = even[3];
    c[8] = even[4];
    c[10] = even[5];
    c[12] = even[6];
    c[14] = even[7];

    let pd_2048 = _mm256_set1_epi32(2048);
    let pd_128 = _mm256_set1_epi32(128);

    // Odd half stage 1: 8 inputs → 8 outputs via 4 pairs
    // DCT-16 uses: (in1,in15), (in9,in7), (in5,in11), (in13,in3)
    // Coefficients from itx_1d.rs:
    //   t8a  = (in1  * 401  - in15 * 4076 + 2048) >> 12
    //   t15a = (in1  * 4076 + in15 * 401  + 2048) >> 12
    let pair_1_15 = i32_to_i16_pair(token, c[1], c[15]);
    let t8a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_1_15, dct8_row_coef_pack(token, 401, -4076)),
        pd_2048,
    ));
    let t15a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_1_15, dct8_row_coef_pack(token, 4076, 401)),
        pd_2048,
    ));

    //   t9a  = (in9  * 3166 - in7  * 2598 + 2048) >> 12
    //   t14a = (in9  * 2598 + in7  * 3166 + 2048) >> 12
    let pair_9_7 = i32_to_i16_pair(token, c[9], c[7]);
    let t9a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_9_7, dct8_row_coef_pack(token, 3166, -2598)),
        pd_2048,
    ));
    let t14a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_9_7, dct8_row_coef_pack(token, 2598, 3166)),
        pd_2048,
    ));

    //   t10a = (in5  * 1931 - in11 * 3612 + 2048) >> 12
    //   t13a = (in5  * 3612 + in11 * 1931 + 2048) >> 12
    let pair_5_11 = i32_to_i16_pair(token, c[5], c[11]);
    let t10a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_5_11, dct8_row_coef_pack(token, 1931, -3612)),
        pd_2048,
    ));
    let t13a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_5_11, dct8_row_coef_pack(token, 3612, 1931)),
        pd_2048,
    ));

    //   t11a = (in13 * 3920 - in3  * 1189 + 2048) >> 12  (was decomposed as (3920-4096))
    //   t12a = (in13 * 1189 + in3  * 3920 + 2048) >> 12
    let pair_13_3 = i32_to_i16_pair(token, c[13], c[3]);
    let t11a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_13_3, dct8_row_coef_pack(token, 3920, -1189)),
        pd_2048,
    ));
    let t12a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_13_3, dct8_row_coef_pack(token, 1189, 3920)),
        pd_2048,
    ));

    let t8 = clip8(token, _mm256_add_epi32(t8a, t9a), min_v, max_v);
    let mut t9 = clip8(token, _mm256_sub_epi32(t8a, t9a), min_v, max_v);
    let mut t10 = clip8(token, _mm256_sub_epi32(t11a, t10a), min_v, max_v);
    let mut t11 = clip8(token, _mm256_add_epi32(t11a, t10a), min_v, max_v);
    let mut t12 = clip8(token, _mm256_add_epi32(t12a, t13a), min_v, max_v);
    let mut t13 = clip8(token, _mm256_sub_epi32(t12a, t13a), min_v, max_v);
    let mut t14 = clip8(token, _mm256_sub_epi32(t15a, t14a), min_v, max_v);
    let t15 = clip8(token, _mm256_add_epi32(t15a, t14a), min_v, max_v);

    // Odd half stage 2: rotation on (t14,t9) and (t13,t10)
    //   t9a  = (t14 * 1567 - t9  * 3784 + 2048) >> 12
    //   t14a = (t14 * 3784 + t9  * 1567 + 2048) >> 12
    let pair_14_9 = i32_to_i16_pair(token, t14, t9);
    let t9a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_14_9, dct8_row_coef_pack(token, 1567, -3784)),
        pd_2048,
    ));
    let t14a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_14_9, dct8_row_coef_pack(token, 3784, 1567)),
        pd_2048,
    ));

    //   t10a = -(t13 * 3784 + t10 * 1567 + 2048) >> 12  (negated rotation)
    //   t13a = (t13 * 1567 - t10 * 3784 + 2048) >> 12    (wait -- need to check sign)
    //
    // From itx_1d.rs DCT-16 stage 2:
    //   t10a = (-(t13 * (3784-4096) + t10 * 1567) + 2048 >> 12) - t13
    //   Direct: -(t13 * 3784 + t10 * 1567 - 2048) >> 12  ... no, let me re-derive.
    //
    // The decomposed form:
    //   t10a = (-(t13*(3784-4096) + t10*1567) + 2048) >> 12 - t13
    //        = (-t13*(-312) - t10*1567 + 2048) >> 12 - t13
    //        = (t13*312 - t10*1567 + 2048) >> 12 - t13
    //
    // Direct form (non-decomposed):
    //   t10a = (-t13*3784 - t10*1567 + 2048) >> 12
    //        = -(t13*3784 + t10*1567 - 2048) >> 12
    //
    // Are these equal? Let's verify:
    //   (t13*312 - t10*1567 + 2048) >> 12 - t13
    //   = (t13*312 - t10*1567 + 2048 - t13*4096) >> 12
    //   = (t13*(312-4096) - t10*1567 + 2048) >> 12
    //   = (-t13*3784 - t10*1567 + 2048) >> 12  ✓
    //
    // So: t10a = pmaddwd(pair(t13,t10), coef(-3784, -1567)) + 2048 >> 12
    let pair_13_10 = i32_to_i16_pair(token, t13, t10);
    let t10a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_13_10, dct8_row_coef_pack(token, -3784, -1567)),
        pd_2048,
    ));
    //   t13a = (t13 * 1567 - t10 * (3784-4096) + 2048 >> 12) - t10
    //   Direct: (t13 * 1567 - t10 * 3784 + 2048) >> 12
    //
    //   Verify: (t13*1567 - t10*(-312) + 2048) >> 12 - t10
    //         = (t13*1567 + t10*312 + 2048 - t10*4096) >> 12
    //         = (t13*1567 + t10*(312-4096) + 2048) >> 12
    //         = (t13*1567 - t10*3784 + 2048) >> 12  ✓
    let t13a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_13_10, dct8_row_coef_pack(token, 1567, -3784)),
        pd_2048,
    ));

    let t8a = clip8(token, _mm256_add_epi32(t8, t11), min_v, max_v);
    t9 = clip8(token, _mm256_add_epi32(t9a, t10a), min_v, max_v);
    t10 = clip8(token, _mm256_sub_epi32(t9a, t10a), min_v, max_v);
    let t11a = clip8(token, _mm256_sub_epi32(t8, t11), min_v, max_v);
    let t12a = clip8(token, _mm256_sub_epi32(t15, t12), min_v, max_v);
    t13 = clip8(token, _mm256_sub_epi32(t14a, t13a), min_v, max_v);
    t14 = clip8(token, _mm256_add_epi32(t14a, t13a), min_v, max_v);
    let t15a = clip8(token, _mm256_add_epi32(t15, t12), min_v, max_v);

    // Final sqrt(2) stage via 181 multiply
    // t10a = ((t13 - t10) * 181 + 128) >> 8
    // t13a = ((t13 + t10) * 181 + 128) >> 8
    let pair_13_10_f = i32_to_i16_pair(token, t13, t10);
    let t10a_new = _mm256_srai_epi32::<8>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_13_10_f, dct8_row_coef_pack(token, 181, -181)),
        pd_128,
    ));
    let t13a_new = _mm256_srai_epi32::<8>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_13_10_f, dct8_row_coef_pack(token, 181, 181)),
        pd_128,
    ));
    let pair_12a_11a = i32_to_i16_pair(token, t12a, t11a);
    t11 = _mm256_srai_epi32::<8>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_12a_11a, dct8_row_coef_pack(token, 181, -181)),
        pd_128,
    ));
    t12 = _mm256_srai_epi32::<8>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_12a_11a, dct8_row_coef_pack(token, 181, 181)),
        pd_128,
    ));

    let t0 = c[0];
    let t1 = c[2];
    let t2 = c[4];
    let t3 = c[6];
    let t4 = c[8];
    let t5 = c[10];
    let t6 = c[12];
    let t7 = c[14];

    c[0] = clip8(token, _mm256_add_epi32(t0, t15a), min_v, max_v);
    c[1] = clip8(token, _mm256_add_epi32(t1, t14), min_v, max_v);
    c[2] = clip8(token, _mm256_add_epi32(t2, t13a_new), min_v, max_v);
    c[3] = clip8(token, _mm256_add_epi32(t3, t12), min_v, max_v);
    c[4] = clip8(token, _mm256_add_epi32(t4, t11), min_v, max_v);
    c[5] = clip8(token, _mm256_add_epi32(t5, t10a_new), min_v, max_v);
    c[6] = clip8(token, _mm256_add_epi32(t6, t9), min_v, max_v);
    c[7] = clip8(token, _mm256_add_epi32(t7, t8a), min_v, max_v);
    c[8] = clip8(token, _mm256_sub_epi32(t7, t8a), min_v, max_v);
    c[9] = clip8(token, _mm256_sub_epi32(t6, t9), min_v, max_v);
    c[10] = clip8(token, _mm256_sub_epi32(t5, t10a_new), min_v, max_v);
    c[11] = clip8(token, _mm256_sub_epi32(t4, t11), min_v, max_v);
    c[12] = clip8(token, _mm256_sub_epi32(t3, t12), min_v, max_v);
    c[13] = clip8(token, _mm256_sub_epi32(t2, t13a_new), min_v, max_v);
    c[14] = clip8(token, _mm256_sub_epi32(t1, t14), min_v, max_v);
    c[15] = clip8(token, _mm256_sub_epi32(t0, t15a), min_v, max_v);
}

/// DCT-32 column transform using pmaddwd for all trig multiplies.
/// Input/output: 32 × __m256i (8 i32 lanes each, values in i16 range).
///
/// Replaces mullo_epi32 (10c Zen3) with madd_epi16 (5c Zen3) for all
/// multiplicative stages in the DCT-32 odd half and recursively in the
/// DCT-16/8/4 even half.
///
/// Stage 1 alone: 32 mullo_epi32 → 16 madd_epi16 (saves ~160 cycles per 8-col chunk).
/// Total across all stages + recursive even half: ~120 mullo → ~60 madd.
#[cfg(target_arch = "x86_64")]
#[rite]
fn dct32_1d_cols8_i16(token: Desktop64, c: &mut [__m256i; 32], min_v: __m256i, max_v: __m256i) {
    // Even half: DCT-16 on positions 0,2,4,...,30
    let mut even = [
        c[0], c[2], c[4], c[6], c[8], c[10], c[12], c[14], c[16], c[18], c[20], c[22], c[24],
        c[26], c[28], c[30],
    ];
    dct16_1d_cols8_i16(token, &mut even, min_v, max_v);
    c[0] = even[0];
    c[2] = even[1];
    c[4] = even[2];
    c[6] = even[3];
    c[8] = even[4];
    c[10] = even[5];
    c[12] = even[6];
    c[14] = even[7];
    c[16] = even[8];
    c[18] = even[9];
    c[20] = even[10];
    c[22] = even[11];
    c[24] = even[12];
    c[26] = even[13];
    c[28] = even[14];
    c[30] = even[15];

    let pd_2048 = _mm256_set1_epi32(2048);
    let pd_1024 = _mm256_set1_epi32(1024);
    let pd_128 = _mm256_set1_epi32(128);

    // ====== DCT-32 odd half, stage 1: 16 outputs from 8 input pairs ======
    // Each pair (in_a, in_b) produces:
    //   t_lo = (in_a * c1 - in_b * c2 + rnd) >> shift
    //   t_hi = (in_a * c2 + in_b * c1 + rnd) >> shift  (or variants)
    //
    // From itx_1d.rs lines 313-338 (non-tx64 path):
    //   Pair (in1,  in31): c=(201, 4091),  shift=12  → t16a, t31a
    //   Pair (in17, in15): c=(3035, 2751), shift=12  → t17a, t30a
    //   Pair (in9,  in23): c=(1751, 3703), shift=12  → t18a, t29a
    //   Pair (in25, in7):  c=(3857, 1380), shift=12  → t19a, t28a
    //   Pair (in5,  in27): c=(995, 3973),  shift=12  → t20a, t27a
    //   Pair (in21, in11): c=(3513, 2106), shift=12  → t21a, t26a
    //   Pair (in13, in19): c=(2440, 3290), shift=11  → t22a, t25a  (note: >>11 with rnd=1024)
    //   Pair (in29, in3):  c=(4052, 601),  shift=12  → t23a, t24a

    // Wait: the scalar uses 1220,1645 for >>11 (halved coefficients).
    // From itx_1d.rs:329: t22a = in13 * 1220 - in19 * 1645 + 1024 >> 11
    // The >>12 form would use 2440, 3290. The >>11 form uses 1220, 1645.
    // Both are equivalent: (a*2440 - b*3290 + 2048) >> 12 == (a*1220 - b*1645 + 1024) >> 11
    // For pmaddwd we need coefficients in i16 range. 2440 and 3290 fit, but let's match
    // the scalar exactly to avoid rounding differences: use >>11 with 1220, 1645.

    let pair_1_31 = i32_to_i16_pair(token, c[1], c[31]);
    let t16a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_1_31, dct8_row_coef_pack(token, 201, -4091)),
        pd_2048,
    ));
    let t31a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_1_31, dct8_row_coef_pack(token, 4091, 201)),
        pd_2048,
    ));

    let pair_17_15 = i32_to_i16_pair(token, c[17], c[15]);
    let t17a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_17_15, dct8_row_coef_pack(token, 3035, -2751)),
        pd_2048,
    ));
    let t30a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_17_15, dct8_row_coef_pack(token, 2751, 3035)),
        pd_2048,
    ));

    let pair_9_23 = i32_to_i16_pair(token, c[9], c[23]);
    let t18a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_9_23, dct8_row_coef_pack(token, 1751, -3703)),
        pd_2048,
    ));
    let t29a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_9_23, dct8_row_coef_pack(token, 3703, 1751)),
        pd_2048,
    ));

    let pair_25_7 = i32_to_i16_pair(token, c[25], c[7]);
    let t19a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_25_7, dct8_row_coef_pack(token, 3857, -1380)),
        pd_2048,
    ));
    let t28a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_25_7, dct8_row_coef_pack(token, 1380, 3857)),
        pd_2048,
    ));

    let pair_5_27 = i32_to_i16_pair(token, c[5], c[27]);
    let t20a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_5_27, dct8_row_coef_pack(token, 995, -3973)),
        pd_2048,
    ));
    let t27a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_5_27, dct8_row_coef_pack(token, 3973, 995)),
        pd_2048,
    ));

    let pair_21_11 = i32_to_i16_pair(token, c[21], c[11]);
    let t21a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_21_11, dct8_row_coef_pack(token, 3513, -2106)),
        pd_2048,
    ));
    let t26a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_21_11, dct8_row_coef_pack(token, 2106, 3513)),
        pd_2048,
    ));

    // Pair (in13, in19): shift=11, rnd=1024, coefficients 1220/1645
    let pair_13_19 = i32_to_i16_pair(token, c[13], c[19]);
    let t22a = _mm256_srai_epi32::<11>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_13_19, dct8_row_coef_pack(token, 1220, -1645)),
        pd_1024,
    ));
    let t25a = _mm256_srai_epi32::<11>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_13_19, dct8_row_coef_pack(token, 1645, 1220)),
        pd_1024,
    ));

    let pair_29_3 = i32_to_i16_pair(token, c[29], c[3]);
    let t23a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_29_3, dct8_row_coef_pack(token, 4052, -601)),
        pd_2048,
    ));
    let t24a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_29_3, dct8_row_coef_pack(token, 601, 4052)),
        pd_2048,
    ));

    // Additive butterfly after stage 1
    let mut t16 = clip8(token, _mm256_add_epi32(t16a, t17a), min_v, max_v);
    let mut t17 = clip8(token, _mm256_sub_epi32(t16a, t17a), min_v, max_v);
    let mut t18 = clip8(token, _mm256_sub_epi32(t19a, t18a), min_v, max_v);
    let t19 = clip8(token, _mm256_add_epi32(t19a, t18a), min_v, max_v);
    let t20 = clip8(token, _mm256_add_epi32(t20a, t21a), min_v, max_v);
    let mut t21 = clip8(token, _mm256_sub_epi32(t20a, t21a), min_v, max_v);
    let mut t22 = clip8(token, _mm256_sub_epi32(t23a, t22a), min_v, max_v);
    let mut t23 = clip8(token, _mm256_add_epi32(t23a, t22a), min_v, max_v);
    let mut t24 = clip8(token, _mm256_add_epi32(t24a, t25a), min_v, max_v);
    let mut t25 = clip8(token, _mm256_sub_epi32(t24a, t25a), min_v, max_v);
    let mut t26 = clip8(token, _mm256_sub_epi32(t27a, t26a), min_v, max_v);
    let t27 = clip8(token, _mm256_add_epi32(t27a, t26a), min_v, max_v);
    let t28 = clip8(token, _mm256_add_epi32(t28a, t29a), min_v, max_v);
    let mut t29 = clip8(token, _mm256_sub_epi32(t28a, t29a), min_v, max_v);
    let mut t30 = clip8(token, _mm256_sub_epi32(t31a, t30a), min_v, max_v);
    let mut t31 = clip8(token, _mm256_add_epi32(t31a, t30a), min_v, max_v);

    // ====== Stage 2: rotations on (t30,t17), (t29,t18), (t26,t21), (t25,t22) ======
    // (t30, t17): coefficients (4017, 799), shift=12
    //   t17a = (t30 * 799  - t17 * 4017 + 2048) >> 12
    //   t30a = (t30 * 4017 + t17 * 799  + 2048) >> 12
    let pair_30_17 = i32_to_i16_pair(token, t30, t17);
    let t17a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_30_17, dct8_row_coef_pack(token, 799, -4017)),
        pd_2048,
    ));
    let t30a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_30_17, dct8_row_coef_pack(token, 4017, 799)),
        pd_2048,
    ));

    // (t29, t18): negated rotation with (4017, 799), shift=12
    //   From scalar: t18a = (-(t29*(4017-4096) + t18*799) + 2048 >> 12) - t29
    //   Direct:      t18a = (-t29*4017 - t18*799 + 2048) >> 12
    //   And:         t29a = (t29*799 - t18*4017 + 2048) >> 12
    //   Verify t29a: scalar says t29a = (t29*799 - t18*(4017-4096) + 2048 >> 12) - t18
    //                = (t29*799 + t18*79 + 2048) >> 12 - t18
    //                = (t29*799 + t18*79 - t18*4096 + 2048) >> 12
    //                = (t29*799 - t18*4017 + 2048) >> 12  ✓
    let pair_29_18 = i32_to_i16_pair(token, t29, t18);
    let t18a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_29_18, dct8_row_coef_pack(token, -4017, -799)),
        pd_2048,
    ));
    let t29a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_29_18, dct8_row_coef_pack(token, 799, -4017)),
        pd_2048,
    ));

    // (t26, t21): coefficients (2276, 3406) at >>11 with rnd=1024
    //   From scalar: t21a = t26 * 1703 - t21 * 1138 + 1024 >> 11
    //                t26a = t26 * 1138 + t21 * 1703 + 1024 >> 11
    //   Wait, 1703 and 1138 are the >>11 coefficients. The >>12 equivalents
    //   would be 3406 and 2276. Let's use >>11 with 1703/1138 to match scalar.
    let pair_26_21 = i32_to_i16_pair(token, t26, t21);
    let t21a = _mm256_srai_epi32::<11>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_26_21, dct8_row_coef_pack(token, 1703, -1138)),
        pd_1024,
    ));
    let t26a = _mm256_srai_epi32::<11>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_26_21, dct8_row_coef_pack(token, 1138, 1703)),
        pd_1024,
    ));

    // (t25, t22): negated rotation with (1138, 1703) at >>11
    //   From scalar: t22a = -(t25*1138 + t22*1703) + 1024 >> 11
    //                t25a = t25*1703 - t22*1138 + 1024 >> 11
    let pair_25_22 = i32_to_i16_pair(token, t25, t22);
    let t22a = _mm256_srai_epi32::<11>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_25_22, dct8_row_coef_pack(token, -1138, -1703)),
        pd_1024,
    ));
    let t25a = _mm256_srai_epi32::<11>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_25_22, dct8_row_coef_pack(token, 1703, -1138)),
        pd_1024,
    ));

    let t16a = clip8(token, _mm256_add_epi32(t16, t19), min_v, max_v);
    t17 = clip8(token, _mm256_add_epi32(t17a, t18a), min_v, max_v);
    t18 = clip8(token, _mm256_sub_epi32(t17a, t18a), min_v, max_v);
    let t19a = clip8(token, _mm256_sub_epi32(t16, t19), min_v, max_v);
    let t20a = clip8(token, _mm256_sub_epi32(t23, t20), min_v, max_v);
    t21 = clip8(token, _mm256_sub_epi32(t22a, t21a), min_v, max_v);
    t22 = clip8(token, _mm256_add_epi32(t22a, t21a), min_v, max_v);
    let t23a = clip8(token, _mm256_add_epi32(t23, t20), min_v, max_v);
    let t24a = clip8(token, _mm256_add_epi32(t24, t27), min_v, max_v);
    t25 = clip8(token, _mm256_add_epi32(t25a, t26a), min_v, max_v);
    t26 = clip8(token, _mm256_sub_epi32(t25a, t26a), min_v, max_v);
    let t27a = clip8(token, _mm256_sub_epi32(t24, t27), min_v, max_v);
    let t28a = clip8(token, _mm256_sub_epi32(t31, t28), min_v, max_v);
    t29 = clip8(token, _mm256_sub_epi32(t30a, t29a), min_v, max_v);
    t30 = clip8(token, _mm256_add_epi32(t30a, t29a), min_v, max_v);
    let t31a = clip8(token, _mm256_add_epi32(t31, t28), min_v, max_v);

    // ====== Stage 3: rotations on (t29,t18), (t28a,t19a), (t27a,t20a), (t26,t21) ======
    // (t29, t18): coefficients (3784, 1567), shift=12
    //   t18a = (t29 * 1567 - t18 * 3784 + 2048) >> 12
    //   t29a = (t29 * 3784 + t18 * 1567 + 2048) >> 12
    let pair_29_18 = i32_to_i16_pair(token, t29, t18);
    let t18a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_29_18, dct8_row_coef_pack(token, 1567, -3784)),
        pd_2048,
    ));
    let t29a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_29_18, dct8_row_coef_pack(token, 3784, 1567)),
        pd_2048,
    ));

    // (t28a, t19a): same coefficients
    //   t19  = (t28a * 1567 - t19a * 3784 + 2048) >> 12
    //   t28  = (t28a * 3784 + t19a * 1567 + 2048) >> 12
    let pair_28a_19a = i32_to_i16_pair(token, t28a, t19a);
    let t19 = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_28a_19a, dct8_row_coef_pack(token, 1567, -3784)),
        pd_2048,
    ));
    let t28 = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_28a_19a, dct8_row_coef_pack(token, 3784, 1567)),
        pd_2048,
    ));

    // (t27a, t20a): negated rotation with (3784, 1567), shift=12
    //   t20  = (-t27a * 3784 - t20a * 1567 + 2048) >> 12
    //   t27  = (t27a * 1567 - t20a * 3784 + 2048) >> 12
    let pair_27a_20a = i32_to_i16_pair(token, t27a, t20a);
    let t20 = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_27a_20a, dct8_row_coef_pack(token, -3784, -1567)),
        pd_2048,
    ));
    let t27 = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_27a_20a, dct8_row_coef_pack(token, 1567, -3784)),
        pd_2048,
    ));

    // (t26, t21): negated rotation with (3784, 1567), shift=12
    //   t21a = (-t26 * 3784 - t21 * 1567 + 2048) >> 12
    //   t26a = (t26 * 1567 - t21 * 3784 + 2048) >> 12
    let pair_26_21 = i32_to_i16_pair(token, t26, t21);
    let t21a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_26_21, dct8_row_coef_pack(token, -3784, -1567)),
        pd_2048,
    ));
    let t26a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_26_21, dct8_row_coef_pack(token, 1567, -3784)),
        pd_2048,
    ));

    t16 = clip8(token, _mm256_add_epi32(t16a, t23a), min_v, max_v);
    let t17a = clip8(token, _mm256_add_epi32(t17, t22), min_v, max_v);
    t18 = clip8(token, _mm256_add_epi32(t18a, t21a), min_v, max_v);
    let t19a = clip8(token, _mm256_add_epi32(t19, t20), min_v, max_v);
    let t20a = clip8(token, _mm256_sub_epi32(t19, t20), min_v, max_v);
    t21 = clip8(token, _mm256_sub_epi32(t18a, t21a), min_v, max_v);
    let t22a = clip8(token, _mm256_sub_epi32(t17, t22), min_v, max_v);
    t23 = clip8(token, _mm256_sub_epi32(t16a, t23a), min_v, max_v);
    t24 = clip8(token, _mm256_sub_epi32(t31a, t24a), min_v, max_v);
    let t25a = clip8(token, _mm256_sub_epi32(t30, t25), min_v, max_v);
    t26 = clip8(token, _mm256_sub_epi32(t29a, t26a), min_v, max_v);
    let t27a = clip8(token, _mm256_sub_epi32(t28, t27), min_v, max_v);
    let t28a = clip8(token, _mm256_add_epi32(t28, t27), min_v, max_v);
    t29 = clip8(token, _mm256_add_epi32(t29a, t26a), min_v, max_v);
    let t30a = clip8(token, _mm256_add_epi32(t30, t25), min_v, max_v);
    t31 = clip8(token, _mm256_add_epi32(t31a, t24a), min_v, max_v);

    // ====== Final stage: 8 × 181 multiply via pmaddwd pairs ======
    // (t27a, t20a): t20_f = (t27a - t20a)*181+128>>8, t27_f = (t27a + t20a)*181+128>>8
    let pair_27a_20a = i32_to_i16_pair(token, t27a, t20a);
    let t20_final = _mm256_srai_epi32::<8>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_27a_20a, dct8_row_coef_pack(token, 181, -181)),
        pd_128,
    ));
    let t27_final = _mm256_srai_epi32::<8>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_27a_20a, dct8_row_coef_pack(token, 181, 181)),
        pd_128,
    ));

    let pair_26_21 = i32_to_i16_pair(token, t26, t21);
    let t21a_final = _mm256_srai_epi32::<8>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_26_21, dct8_row_coef_pack(token, 181, -181)),
        pd_128,
    ));
    let t26a_final = _mm256_srai_epi32::<8>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_26_21, dct8_row_coef_pack(token, 181, 181)),
        pd_128,
    ));

    let pair_25a_22a = i32_to_i16_pair(token, t25a, t22a);
    let t22_final = _mm256_srai_epi32::<8>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_25a_22a, dct8_row_coef_pack(token, 181, -181)),
        pd_128,
    ));
    let t25_final = _mm256_srai_epi32::<8>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_25a_22a, dct8_row_coef_pack(token, 181, 181)),
        pd_128,
    ));

    let pair_24_23 = i32_to_i16_pair(token, t24, t23);
    let t23a = _mm256_srai_epi32::<8>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_24_23, dct8_row_coef_pack(token, 181, -181)),
        pd_128,
    ));
    let t24a = _mm256_srai_epi32::<8>(_mm256_add_epi32(
        _mm256_madd_epi16(pair_24_23, dct8_row_coef_pack(token, 181, 181)),
        pd_128,
    ));

    // Combine with DCT-16 even-half outputs
    let t0 = c[0];
    let t1 = c[2];
    let t2 = c[4];
    let t3 = c[6];
    let t4 = c[8];
    let t5 = c[10];
    let t6 = c[12];
    let t7 = c[14];
    let t8 = c[16];
    let t9 = c[18];
    let t10 = c[20];
    let t11 = c[22];
    let t12 = c[24];
    let t13 = c[26];
    let t14 = c[28];
    let t15 = c[30];

    c[0] = clip8(token, _mm256_add_epi32(t0, t31), min_v, max_v);
    c[1] = clip8(token, _mm256_add_epi32(t1, t30a), min_v, max_v);
    c[2] = clip8(token, _mm256_add_epi32(t2, t29), min_v, max_v);
    c[3] = clip8(token, _mm256_add_epi32(t3, t28a), min_v, max_v);
    c[4] = clip8(token, _mm256_add_epi32(t4, t27_final), min_v, max_v);
    c[5] = clip8(token, _mm256_add_epi32(t5, t26a_final), min_v, max_v);
    c[6] = clip8(token, _mm256_add_epi32(t6, t25_final), min_v, max_v);
    c[7] = clip8(token, _mm256_add_epi32(t7, t24a), min_v, max_v);
    c[8] = clip8(token, _mm256_add_epi32(t8, t23a), min_v, max_v);
    c[9] = clip8(token, _mm256_add_epi32(t9, t22_final), min_v, max_v);
    c[10] = clip8(token, _mm256_add_epi32(t10, t21a_final), min_v, max_v);
    c[11] = clip8(token, _mm256_add_epi32(t11, t20_final), min_v, max_v);
    c[12] = clip8(token, _mm256_add_epi32(t12, t19a), min_v, max_v);
    c[13] = clip8(token, _mm256_add_epi32(t13, t18), min_v, max_v);
    c[14] = clip8(token, _mm256_add_epi32(t14, t17a), min_v, max_v);
    c[15] = clip8(token, _mm256_add_epi32(t15, t16), min_v, max_v);
    c[16] = clip8(token, _mm256_sub_epi32(t15, t16), min_v, max_v);
    c[17] = clip8(token, _mm256_sub_epi32(t14, t17a), min_v, max_v);
    c[18] = clip8(token, _mm256_sub_epi32(t13, t18), min_v, max_v);
    c[19] = clip8(token, _mm256_sub_epi32(t12, t19a), min_v, max_v);
    c[20] = clip8(token, _mm256_sub_epi32(t11, t20_final), min_v, max_v);
    c[21] = clip8(token, _mm256_sub_epi32(t10, t21a_final), min_v, max_v);
    c[22] = clip8(token, _mm256_sub_epi32(t9, t22_final), min_v, max_v);
    c[23] = clip8(token, _mm256_sub_epi32(t8, t23a), min_v, max_v);
    c[24] = clip8(token, _mm256_sub_epi32(t7, t24a), min_v, max_v);
    c[25] = clip8(token, _mm256_sub_epi32(t6, t25_final), min_v, max_v);
    c[26] = clip8(token, _mm256_sub_epi32(t5, t26a_final), min_v, max_v);
    c[27] = clip8(token, _mm256_sub_epi32(t4, t27_final), min_v, max_v);
    c[28] = clip8(token, _mm256_sub_epi32(t3, t28a), min_v, max_v);
    c[29] = clip8(token, _mm256_sub_epi32(t2, t29), min_v, max_v);
    c[30] = clip8(token, _mm256_sub_epi32(t1, t30a), min_v, max_v);
    c[31] = clip8(token, _mm256_sub_epi32(t0, t31), min_v, max_v);
}

/// Run dct32 1D column transform over 32x32 row-major buffer (32 cols, 32 rows).
/// Processes 8 cols at a time -- 4 chunks total.
/// `#[rite]` so it inlines into matching-feature `#[arcane]` callers (zero call cost).
#[cfg(target_arch = "x86_64")]
#[rite]
fn dct32x32_cols_simd(token: Desktop64, tmp: &mut [i32; 1024], min: i32, max: i32) {
    // Try AVX-512 first: 16 cols per chunk (2 chunks total for 32x32).
    if let Some(t512) = crate::src::cpu::summon_avx512() {
        dct32_cols_avx512(t512, tmp, 32, 32, min, max);
        return;
    }
    // Use i16-packed pmaddwd when values are in i16 range (8bpc column pass).
    // For 16bpc (10/12-bit), values exceed i16 range — use mullo_epi32 fallback.
    let use_i16 = min == i16::MIN as i32 && max == i16::MAX as i32;
    let min_v = _mm256_set1_epi32(min);
    let max_v = _mm256_set1_epi32(max);
    for cx_chunk in 0..4 {
        let cx = cx_chunk * 8;
        let mut v = [_mm256_setzero_si256(); 32];
        for i in 0..32 {
            v[i] = loadu_256!(&tmp[i * 32 + cx..i * 32 + cx + 8], [i32; 8]);
        }
        if use_i16 {
            dct32_1d_cols8_i16(token, &mut v, min_v, max_v);
        } else {
            dct32_1d_cols8(token, &mut v, min_v, max_v);
        }
        for i in 0..32 {
            storeu_256!(&mut tmp[i * 32 + cx..i * 32 + cx + 8], [i32; 8], v[i]);
        }
    }
}

/// Generic 16x16 transform function
#[inline]
fn inv_txfm_16x16_inner(
    tmp: &mut [i32; 256],
    coeff: &[i16],
    row_transform: fn(&mut [i32], usize, i32, i32),
    col_transform: fn(&mut [i32], usize, i32, i32),
    row_clip_min: i32,
    row_clip_max: i32,
    col_clip_min: i32,
    col_clip_max: i32,
) {
    let rnd = 2;
    let shift = 2;

    // Row transform
    for y in 0..16 {
        // Load row from column-major
        let mut scratch = [0i32; 16];
        for x in 0..16 {
            scratch[x] = coeff[y + x * 16] as i32;
        }
        row_transform(&mut scratch[..16], 1, row_clip_min, row_clip_max);
        // Apply intermediate shift and store row-major
        for x in 0..16 {
            tmp[y * 16 + x] = ((scratch[x] + rnd) >> shift).clamp(col_clip_min, col_clip_max);
        }
    }

    // Column transform (in-place, row-major with stride 16)
    for x in 0..16 {
        col_transform(&mut tmp[x..], 16, col_clip_min, col_clip_max);
    }
}

/// Add transformed coefficients to destination with SIMD
/// `#[rite]` so it inlines into matching-feature `#[arcane]` callers (zero call cost).
#[cfg(target_arch = "x86_64")]
#[rite]
fn add_16x16_to_dst(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp: &[i32; 256],
    coeff: &mut [i16],
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();

    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi16(bitdepth_max as i16);
    let rnd_final = _mm256_set1_epi32(8);

    for y in 0..16 {
        let dst_off = y * dst_stride;

        // Load destination pixels (16 bytes)
        let d = loadu_128!(<&[u8; 16]>::try_from(&dst[dst_off..dst_off + 16]).unwrap());
        let d16 = _mm256_cvtepu8_epi16(d);

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

        // Pack to 16-bit
        let c16 = _mm256_packs_epi32(c0_scaled, c1_scaled);
        // Fix lane order after packs
        let c16 = _mm256_permute4x64_epi64::<0b11_01_10_00>(c16);

        // Add to destination
        let sum = _mm256_add_epi16(d16, c16);
        let clamped = _mm256_max_epi16(_mm256_min_epi16(sum, max_val), zero);

        // Pack to 8-bit
        let packed = _mm256_packus_epi16(clamped, clamped);
        let packed = _mm256_permute4x64_epi64::<0b11_01_10_00>(packed);

        // Store 16 pixels
        storeu_128!(
            <&mut [u8; 16]>::try_from(&mut dst[dst_off..dst_off + 16]).unwrap(),
            _mm256_castsi256_si128(packed)
        );
    }

    // Clear coefficients (256 * 2 = 512 bytes = 16 * 32 bytes)
    coeff[..256].fill(0);
}

