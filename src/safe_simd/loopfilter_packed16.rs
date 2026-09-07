// 8-bit wide filtering: eight independent horizontal positions in i16 lanes.
#[cfg(target_arch = "x86_64")]
mod packed16 {
    use super::*;

    #[cfg(test)]
    static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    #[cfg(test)]
    pub(super) fn calls() -> usize {
        CALLS.load(Relaxed)
    }

    #[rite]
    fn transpose8(_token: Desktop64, m: [__m128i; 8]) -> [__m128i; 8] {
        let a0 = _mm_unpacklo_epi16(m[0], m[1]);
        let a1 = _mm_unpackhi_epi16(m[0], m[1]);
        let a2 = _mm_unpacklo_epi16(m[2], m[3]);
        let a3 = _mm_unpackhi_epi16(m[2], m[3]);
        let a4 = _mm_unpacklo_epi16(m[4], m[5]);
        let a5 = _mm_unpackhi_epi16(m[4], m[5]);
        let a6 = _mm_unpacklo_epi16(m[6], m[7]);
        let a7 = _mm_unpackhi_epi16(m[6], m[7]);
        let b0 = _mm_unpacklo_epi32(a0, a2);
        let b1 = _mm_unpackhi_epi32(a0, a2);
        let b2 = _mm_unpacklo_epi32(a1, a3);
        let b3 = _mm_unpackhi_epi32(a1, a3);
        let b4 = _mm_unpacklo_epi32(a4, a6);
        let b5 = _mm_unpackhi_epi32(a4, a6);
        let b6 = _mm_unpacklo_epi32(a5, a7);
        let b7 = _mm_unpackhi_epi32(a5, a7);
        [
            _mm_unpacklo_epi64(b0, b4),
            _mm_unpackhi_epi64(b0, b4),
            _mm_unpacklo_epi64(b1, b5),
            _mm_unpackhi_epi64(b1, b5),
            _mm_unpacklo_epi64(b2, b6),
            _mm_unpackhi_epi64(b2, b6),
            _mm_unpacklo_epi64(b3, b7),
            _mm_unpackhi_epi64(b3, b7),
        ]
    }

    // All input taps and thresholds are zero-extended u8. The wide filter's
    // largest positive sum is 16*255+8=4088. Narrow bounds are those of packed6.
    #[rite]
    fn compute(_token: Desktop64, taps: [__m128i; 14], levels: [u8; 3]) -> [__m128i; 12] {
        let [
            p6_v,
            p5_v,
            p4_v,
            p3_v,
            p2_v,
            p1_v,
            p0_v,
            q0_v,
            q1_v,
            q2_v,
            q3_v,
            q4_v,
            q5_v,
            q6_v,
        ] = taps;
        let [e, i, h] = levels.map(i16::from);
        let i_v = _mm_set1_epi16(i);
        let e_v = _mm_set1_epi16(e);
        let h_v = _mm_set1_epi16(h);
        let f_v = _mm_set1_epi16(1);

        let abs = |a: __m128i, b: __m128i| _mm_abs_epi16(_mm_sub_epi16(a, b));

        let abs_p1p0 = abs(p1_v, p0_v);
        let abs_q1q0 = abs(q1_v, q0_v);
        let abs_p0q0 = abs(p0_v, q0_v);
        let abs_p1q1 = abs(p1_v, q1_v);
        let abs_p2p1 = abs(p2_v, p1_v);
        let abs_q2q1 = abs(q2_v, q1_v);
        let abs_p3p2 = abs(p3_v, p2_v);
        let abs_q3q2 = abs(q3_v, q2_v);

        let not_gt = |a: __m128i, b: __m128i| -> __m128i {
            _mm_andnot_si128(_mm_cmpgt_epi16(a, b), _mm_set1_epi16(-1))
        };

        let m_p1p0 = not_gt(abs_p1p0, i_v);
        let m_q1q0 = not_gt(abs_q1q0, i_v);
        let val_ee = _mm_add_epi16(_mm_slli_epi16::<1>(abs_p0q0), _mm_srli_epi16::<1>(abs_p1q1));
        let m_val = not_gt(val_ee, e_v);
        let m_p2p1 = not_gt(abs_p2p1, i_v);
        let m_q2q1 = not_gt(abs_q2q1, i_v);
        let m_p3p2 = not_gt(abs_p3p2, i_v);
        let m_q3q2 = not_gt(abs_q3q2, i_v);
        let fm_mask = _mm_and_si128(
            _mm_and_si128(_mm_and_si128(m_p1p0, m_q1q0), m_val),
            _mm_and_si128(_mm_and_si128(m_p2p1, m_q2q1), _mm_and_si128(m_p3p2, m_q3q2)),
        );

        let abs_p6p0 = abs(p6_v, p0_v);
        let abs_p5p0 = abs(p5_v, p0_v);
        let abs_p4p0 = abs(p4_v, p0_v);
        let abs_q4q0 = abs(q4_v, q0_v);
        let abs_q5q0 = abs(q5_v, q0_v);
        let abs_q6q0 = abs(q6_v, q0_v);
        let flat8out_mask = _mm_and_si128(
            _mm_and_si128(
                _mm_and_si128(not_gt(abs_p6p0, f_v), not_gt(abs_p5p0, f_v)),
                not_gt(abs_p4p0, f_v),
            ),
            _mm_and_si128(
                _mm_and_si128(not_gt(abs_q4q0, f_v), not_gt(abs_q5q0, f_v)),
                not_gt(abs_q6q0, f_v),
            ),
        );

        let abs_p2p0 = abs(p2_v, p0_v);
        let abs_q2q0 = abs(q2_v, q0_v);
        let abs_p3p0 = abs(p3_v, p0_v);
        let abs_q3q0 = abs(q3_v, q0_v);
        let flat8in_mask = _mm_and_si128(
            _mm_and_si128(not_gt(abs_p2p0, f_v), not_gt(abs_p1p0, f_v)),
            _mm_and_si128(
                _mm_and_si128(not_gt(abs_q1q0, f_v), not_gt(abs_q2q0, f_v)),
                _mm_and_si128(not_gt(abs_p3p0, f_v), not_gt(abs_q3q0, f_v)),
            ),
        );

        let dbl = |v: __m128i| _mm_slli_epi16::<1>(v);
        let add = |a: __m128i, b: __m128i| _mm_add_epi16(a, b);
        let add3 = |a: __m128i, b: __m128i, c: __m128i| add(add(a, b), c);
        let add4 = |a: __m128i, b: __m128i, c: __m128i, d: __m128i| add(add(a, b), add(c, d));
        let c4 = _mm_set1_epi16(4);
        let c8 = _mm_set1_epi16(8);

        // 14-tap outputs (positions -6..5)
        let p6_5 = _mm_add_epi16(
            _mm_add_epi16(_mm_add_epi16(p6_v, p6_v), _mm_add_epi16(p6_v, p6_v)),
            p6_v,
        );
        let q6_5 = _mm_add_epi16(
            _mm_add_epi16(_mm_add_epi16(q6_v, q6_v), _mm_add_epi16(q6_v, q6_v)),
            q6_v,
        );

        let mut s = add(p6_5, _mm_add_epi16(dbl(p6_v), dbl(p5_v)));
        s = add(s, dbl(p4_v));
        s = add(s, add4(p3_v, p2_v, p1_v, p0_v));
        s = add(s, add(q0_v, c8));
        let out_m6 = _mm_srai_epi16::<4>(s);

        let mut s = add(p6_5, _mm_add_epi16(dbl(p5_v), dbl(p4_v)));
        s = add(s, dbl(p3_v));
        s = add(s, add4(p2_v, p1_v, p0_v, q0_v));
        s = add(s, add(q1_v, c8));
        let out_m5 = _mm_srai_epi16::<4>(s);

        let p6_4 = _mm_add_epi16(dbl(p6_v), dbl(p6_v));
        let mut s = add(p6_4, p5_v);
        s = add(s, _mm_add_epi16(dbl(p4_v), dbl(p3_v)));
        s = add(s, dbl(p2_v));
        s = add(s, add4(p1_v, p0_v, q0_v, q1_v));
        s = add(s, add(q2_v, c8));
        let out_m4 = _mm_srai_epi16::<4>(s);

        let p6_3 = add(dbl(p6_v), p6_v);
        let mut s = add(p6_3, _mm_add_epi16(p5_v, p4_v));
        s = add(s, _mm_add_epi16(dbl(p3_v), dbl(p2_v)));
        s = add(s, dbl(p1_v));
        s = add(s, add4(p0_v, q0_v, q1_v, q2_v));
        s = add(s, add(q3_v, c8));
        let out_m3 = _mm_srai_epi16::<4>(s);

        let mut s = add(dbl(p6_v), p5_v);
        s = add(s, _mm_add_epi16(p4_v, p3_v));
        s = add(s, _mm_add_epi16(dbl(p2_v), dbl(p1_v)));
        s = add(s, dbl(p0_v));
        s = add(s, add4(q0_v, q1_v, q2_v, q3_v));
        s = add(s, add(q4_v, c8));
        let out_m2 = _mm_srai_epi16::<4>(s);

        let mut s = add(p6_v, p5_v);
        s = add(s, _mm_add_epi16(p4_v, p3_v));
        s = add(s, p2_v);
        s = add(s, _mm_add_epi16(dbl(p1_v), dbl(p0_v)));
        s = add(s, dbl(q0_v));
        s = add(s, add4(q1_v, q2_v, q3_v, q4_v));
        s = add(s, add(q5_v, c8));
        let out_m1 = _mm_srai_epi16::<4>(s);

        let mut s = add(p5_v, p4_v);
        s = add(s, _mm_add_epi16(p3_v, p2_v));
        s = add(s, p1_v);
        s = add(s, _mm_add_epi16(dbl(p0_v), dbl(q0_v)));
        s = add(s, dbl(q1_v));
        s = add(s, add4(q2_v, q3_v, q4_v, q5_v));
        s = add(s, add(q6_v, c8));
        let out_0 = _mm_srai_epi16::<4>(s);

        let mut s = add(p4_v, p3_v);
        s = add(s, _mm_add_epi16(p2_v, p1_v));
        s = add(s, p0_v);
        s = add(s, _mm_add_epi16(dbl(q0_v), dbl(q1_v)));
        s = add(s, dbl(q2_v));
        s = add(s, add4(q3_v, q4_v, q5_v, q6_v));
        s = add(s, add(q6_v, c8));
        let out_1 = _mm_srai_epi16::<4>(s);

        let mut s = add(p3_v, p2_v);
        s = add(s, _mm_add_epi16(p1_v, p0_v));
        s = add(s, q0_v);
        s = add(s, _mm_add_epi16(dbl(q1_v), dbl(q2_v)));
        s = add(s, dbl(q3_v));
        let q6_3 = add(dbl(q6_v), q6_v);
        s = add(s, add3(q4_v, q5_v, q6_3));
        s = add(s, c8);
        let out_2 = _mm_srai_epi16::<4>(s);

        let q6_4 = _mm_add_epi16(dbl(q6_v), dbl(q6_v));
        let mut s = add(p2_v, p1_v);
        s = add(s, _mm_add_epi16(p0_v, q0_v));
        s = add(s, q1_v);
        s = add(s, _mm_add_epi16(dbl(q2_v), dbl(q3_v)));
        s = add(s, dbl(q4_v));
        s = add(s, add(q5_v, q6_4));
        s = add(s, c8);
        let out_3 = _mm_srai_epi16::<4>(s);

        let mut s = add(p1_v, p0_v);
        s = add(s, _mm_add_epi16(q0_v, q1_v));
        s = add(s, q2_v);
        s = add(s, _mm_add_epi16(dbl(q3_v), dbl(q4_v)));
        s = add(s, dbl(q5_v));
        s = add(s, q6_5);
        s = add(s, c8);
        let out_4 = _mm_srai_epi16::<4>(s);

        let q6_7 = _mm_add_epi16(q6_5, _mm_add_epi16(q6_v, q6_v));
        let mut s = add(p0_v, q0_v);
        s = add(s, _mm_add_epi16(q1_v, q2_v));
        s = add(s, q3_v);
        s = add(s, _mm_add_epi16(dbl(q4_v), dbl(q5_v)));
        s = add(s, q6_7);
        s = add(s, c8);
        let out_5 = _mm_srai_epi16::<4>(s);

        // 8-tap outputs
        let triple = |v: __m128i| _mm_add_epi16(dbl(v), v);
        let out8_m3 = _mm_srai_epi16::<3>(add(
            add4(triple(p3_v), dbl(p2_v), p1_v, p0_v),
            add(q0_v, c4),
        ));
        let out8_m2 = _mm_srai_epi16::<3>(add(
            add4(dbl(p3_v), p2_v, dbl(p1_v), p0_v),
            add3(q0_v, q1_v, c4),
        ));
        let out8_m1 = _mm_srai_epi16::<3>(add(
            add4(p3_v, p2_v, p1_v, dbl(p0_v)),
            add4(q0_v, q1_v, q2_v, c4),
        ));
        let out8_0 = _mm_srai_epi16::<3>(add(
            add4(p2_v, p1_v, p0_v, dbl(q0_v)),
            add4(q1_v, q2_v, q3_v, c4),
        ));
        let out8_1 = _mm_srai_epi16::<3>(add(
            add4(p1_v, p0_v, q0_v, dbl(q1_v)),
            add4(q2_v, q3_v, q3_v, c4),
        ));
        let out8_2 = _mm_srai_epi16::<3>(add(
            add4(p0_v, q0_v, q1_v, dbl(q2_v)),
            add4(q3_v, q3_v, q3_v, c4),
        ));

        // Narrow filter (4-tap)
        let neg128 = _mm_set1_epi16(-128);
        let pos127 = _mm_set1_epi16(127);
        let iclip = |v: __m128i| _mm_min_epi16(_mm_max_epi16(v, neg128), pos127);
        let diff_q0p0 = _mm_sub_epi16(q0_v, p0_v);
        let three_d = _mm_add_epi16(_mm_slli_epi16::<1>(diff_q0p0), diff_q0p0);
        let diff_p1q1 = _mm_sub_epi16(p1_v, q1_v);
        let hev_mask = _mm_or_si128(
            _mm_cmpgt_epi16(abs_p1p0, h_v),
            _mm_cmpgt_epi16(abs_q1q0, h_v),
        );
        let f_hev = iclip(_mm_add_epi16(three_d, iclip(diff_p1q1)));
        let f_no = iclip(three_d);
        let c4i = c4;
        let c3i = _mm_set1_epi16(3);
        let one = _mm_set1_epi16(1);
        let f1_hev = _mm_srai_epi16::<3>(_mm_min_epi16(_mm_add_epi16(f_hev, c4i), pos127));
        let f2_hev = _mm_srai_epi16::<3>(_mm_min_epi16(_mm_add_epi16(f_hev, c3i), pos127));
        let f1_no = _mm_srai_epi16::<3>(_mm_min_epi16(_mm_add_epi16(f_no, c4i), pos127));
        let f2_no = _mm_srai_epi16::<3>(_mm_min_epi16(_mm_add_epi16(f_no, c3i), pos127));
        let f_extra = _mm_srai_epi16::<1>(_mm_add_epi16(f1_no, one));
        let p0_hev = _mm_add_epi16(p0_v, f2_hev);
        let q0_hev = _mm_sub_epi16(q0_v, f1_hev);
        let p0_no = _mm_add_epi16(p0_v, f2_no);
        let q0_no = _mm_sub_epi16(q0_v, f1_no);
        let p1_no = _mm_add_epi16(p1_v, f_extra);
        let q1_no = _mm_sub_epi16(q1_v, f_extra);

        let blendv = |a: __m128i, b: __m128i, mask: __m128i| -> __m128i {
            _mm_or_si128(_mm_andnot_si128(mask, a), _mm_and_si128(mask, b))
        };

        let narrow_p1 = blendv(p1_no, p1_v, hev_mask);
        let narrow_p0 = blendv(p0_no, p0_hev, hev_mask);
        let narrow_q0 = blendv(q0_no, q0_hev, hev_mask);
        let narrow_q1 = blendv(q1_no, q1_v, hev_mask);

        let wide_mask = _mm_and_si128(flat8out_mask, flat8in_mask);

        let mid_m3 = blendv(p2_v, out8_m3, flat8in_mask);
        let mid_m2 = blendv(narrow_p1, out8_m2, flat8in_mask);
        let mid_m1 = blendv(narrow_p0, out8_m1, flat8in_mask);
        let mid_0 = blendv(narrow_q0, out8_0, flat8in_mask);
        let mid_1 = blendv(narrow_q1, out8_1, flat8in_mask);
        let mid_2 = blendv(q2_v, out8_2, flat8in_mask);

        let sel_m6 = blendv(p5_v, out_m6, wide_mask);
        let sel_m5 = blendv(p4_v, out_m5, wide_mask);
        let sel_m4 = blendv(p3_v, out_m4, wide_mask);
        let sel_m3 = blendv(mid_m3, out_m3, wide_mask);
        let sel_m2 = blendv(mid_m2, out_m2, wide_mask);
        let sel_m1 = blendv(mid_m1, out_m1, wide_mask);
        let sel_0 = blendv(mid_0, out_0, wide_mask);
        let sel_1 = blendv(mid_1, out_1, wide_mask);
        let sel_2 = blendv(mid_2, out_2, wide_mask);
        let sel_3 = blendv(q3_v, out_3, wide_mask);
        let sel_4 = blendv(q4_v, out_4, wide_mask);
        let sel_5 = blendv(q5_v, out_5, wide_mask);

        let final_m6 = blendv(p5_v, sel_m6, fm_mask);
        let final_m5 = blendv(p4_v, sel_m5, fm_mask);
        let final_m4 = blendv(p3_v, sel_m4, fm_mask);
        let final_m3 = blendv(p2_v, sel_m3, fm_mask);
        let final_m2 = blendv(p1_v, sel_m2, fm_mask);
        let final_m1 = blendv(p0_v, sel_m1, fm_mask);
        let final_0 = blendv(q0_v, sel_0, fm_mask);
        let final_1 = blendv(q1_v, sel_1, fm_mask);
        let final_2 = blendv(q2_v, sel_2, fm_mask);
        let final_3 = blendv(q3_v, sel_3, fm_mask);
        let final_4 = blendv(q4_v, sel_4, fm_mask);
        let final_5 = blendv(q5_v, sel_5, fm_mask);

        [
            final_m6, final_m5, final_m4, final_m3, final_m2, final_m1, final_0, final_1, final_2,
            final_3, final_4, final_5,
        ]
    }

    // Eight separate rows, exact input taps -7..=6 and output taps -6..=5.
    // The caller must require two adjacent width-16 groups with equal levels.
    #[arcane]
    pub(super) fn apply_h(
        token: Desktop64,
        buf: &mut [u8],
        base: usize,
        stride: isize,
        levels: [u8; 3],
    ) {
        #[cfg(test)]
        CALLS.fetch_add(1, Relaxed);
        debug_assert!(stride.unsigned_abs() >= 14);
        let zero = _mm_setzero_si128();
        let mut lo = [zero; 8];
        let mut hi = [zero; 8];
        for row in 0..8 {
            let start = signed_idx(base, row as isize * stride - 7);
            let bytes: [u8; 8] = buf[start..][..8].try_into().unwrap();
            lo[row] = _mm_cvtepu8_epi16(_mm_cvtsi64_si128(i64::from_le_bytes(bytes)));
            let mut bytes = [0u8; 8];
            bytes[..6].copy_from_slice(&buf[start + 8..][..6]);
            hi[row] = _mm_cvtepu8_epi16(_mm_cvtsi64_si128(i64::from_le_bytes(bytes)));
        }
        let lo = transpose8(token, lo);
        let hi = transpose8(token, hi);
        let mut taps = [zero; 14];
        taps[..8].copy_from_slice(&lo);
        taps[8..].copy_from_slice(&hi[..6]);
        let out = compute(token, taps, levels);
        let lo = transpose8(token, out[..8].try_into().unwrap());
        let hi = transpose8(
            token,
            [out[8], out[9], out[10], out[11], zero, zero, zero, zero],
        );
        for row in 0..8 {
            let start = signed_idx(base, row as isize * stride - 6);
            let bytes = _mm_cvtsi128_si64(_mm_packus_epi16(lo[row], zero)).to_le_bytes();
            buf[start..][..8].copy_from_slice(&bytes);
            let bytes = _mm_cvtsi128_si32(_mm_packus_epi16(hi[row], zero)).to_le_bytes();
            buf[start + 8..][..4].copy_from_slice(&bytes);
        }
    }
}
