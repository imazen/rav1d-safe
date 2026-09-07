// 8-bit six-tap filtering: eight independent positions in i16 lanes.
#[cfg(target_arch = "x86_64")]
mod packed6 {
    use super::*;

    #[cfg(test)]
    static CALLS: [std::sync::atomic::AtomicUsize; 2] =
        [const { std::sync::atomic::AtomicUsize::new(0) }; 2];

    #[cfg(test)]
    pub(super) fn calls() -> [usize; 2] {
        CALLS.each_ref().map(|n| n.load(Relaxed))
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

    // Inputs are zero-extended bytes and thresholds are u8. Six-tap weighted
    // sums plus rounding are at most 8*255+4 = 2044. The narrow pre-clamp
    // expression is in -893..=892, its corrections in -16..=15, and its
    // pre-pixel-clamp result in -16..=271. All arithmetic fits signed i16;
    // packus performs the final scalar pixel clamp.
    #[rite]
    fn compute(_token: Desktop64, taps: [__m128i; 6], levels: [u8; 3]) -> [__m128i; 4] {
        let [p2_v, p1_v, p0_v, q0_v, q1_v, q2_v] = taps;
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

        let not_gt = |a: __m128i, b: __m128i| -> __m128i {
            _mm_andnot_si128(_mm_cmpgt_epi16(a, b), _mm_set1_epi16(-1))
        };

        let m_p1p0 = not_gt(abs_p1p0, i_v);
        let m_q1q0 = not_gt(abs_q1q0, i_v);
        let val_ee = _mm_add_epi16(_mm_slli_epi16::<1>(abs_p0q0), _mm_srli_epi16::<1>(abs_p1q1));
        let m_val = not_gt(val_ee, e_v);
        let m_p2p1 = not_gt(abs_p2p1, i_v);
        let m_q2q1 = not_gt(abs_q2q1, i_v);
        let fm_mask = _mm_and_si128(
            _mm_and_si128(_mm_and_si128(m_p1p0, m_q1q0), m_val),
            _mm_and_si128(m_p2p1, m_q2q1),
        );

        let abs_p2p0 = abs(p2_v, p0_v);
        let abs_q2q0 = abs(q2_v, q0_v);
        let flat_mask = _mm_and_si128(
            _mm_and_si128(not_gt(abs_p2p0, f_v), not_gt(abs_p1p0, f_v)),
            _mm_and_si128(not_gt(abs_q1q0, f_v), not_gt(abs_q2q0, f_v)),
        );

        let p2_3 = _mm_add_epi16(_mm_slli_epi16::<1>(p2_v), p2_v);
        let c4 = _mm_set1_epi16(4);
        let dbl = |v: __m128i| _mm_slli_epi16::<1>(v);

        let out_m2 = _mm_srai_epi16::<3>(_mm_add_epi16(
            _mm_add_epi16(
                _mm_add_epi16(p2_3, dbl(p1_v)),
                _mm_add_epi16(dbl(p0_v), q0_v),
            ),
            c4,
        ));
        let out_m1 = _mm_srai_epi16::<3>(_mm_add_epi16(
            _mm_add_epi16(
                _mm_add_epi16(p2_v, dbl(p1_v)),
                _mm_add_epi16(_mm_add_epi16(dbl(p0_v), dbl(q0_v)), q1_v),
            ),
            c4,
        ));
        let out_0 = _mm_srai_epi16::<3>(_mm_add_epi16(
            _mm_add_epi16(
                _mm_add_epi16(p1_v, dbl(p0_v)),
                _mm_add_epi16(_mm_add_epi16(dbl(q0_v), dbl(q1_v)), q2_v),
            ),
            c4,
        ));
        let q2_3 = _mm_add_epi16(_mm_slli_epi16::<1>(q2_v), q2_v);
        let out_1 = _mm_srai_epi16::<3>(_mm_add_epi16(
            _mm_add_epi16(
                _mm_add_epi16(p0_v, dbl(q0_v)),
                _mm_add_epi16(dbl(q1_v), q2_3),
            ),
            c4,
        ));

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

        let c4i = _mm_set1_epi16(4);
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

        let out_m2_sel = blendv(narrow_p1, out_m2, flat_mask);
        let out_m1_sel = blendv(narrow_p0, out_m1, flat_mask);
        let out_0_sel = blendv(narrow_q0, out_0, flat_mask);
        let out_1_sel = blendv(narrow_q1, out_1, flat_mask);

        let final_p1 = blendv(p1_v, out_m2_sel, fm_mask);
        let final_p0 = blendv(p0_v, out_m1_sel, fm_mask);
        let final_q0 = blendv(q0_v, out_0_sel, fm_mask);
        let final_q1 = blendv(q1_v, out_1_sel, fm_mask);
        [final_p1, final_p0, final_q0, final_q1]
    }

    // H processes eight picture rows, V eight adjacent columns. Input taps
    // are exactly -3..=2, output taps -2..=1. No rounded loads cross that
    // footprint. The caller fuses only two adjacent groups with equal levels.
    #[arcane]
    pub(super) fn apply<const H: bool>(
        token: Desktop64,
        buf: &mut [u8],
        base: usize,
        stride: isize,
        levels: [u8; 3],
    ) {
        #[cfg(test)]
        CALLS[usize::from(H)].fetch_add(1, Relaxed);
        debug_assert!(stride.unsigned_abs() >= if H { 6 } else { 8 });
        let zero = _mm_setzero_si128();
        let mut taps = [zero; 6];
        if H {
            let mut rows = [zero; 8];
            for (row, v) in rows.iter_mut().enumerate() {
                let start = signed_idx(base, row as isize * stride - 3);
                let mut bytes = [0u8; 8];
                bytes[..6].copy_from_slice(&buf[start..][..6]);
                *v = _mm_cvtepu8_epi16(_mm_cvtsi64_si128(i64::from_le_bytes(bytes)));
            }
            let cols = transpose8(token, rows);
            taps.copy_from_slice(&cols[..6]);
        } else {
            for (index, v) in taps.iter_mut().enumerate() {
                let start = signed_idx(base, (index as isize - 3) * stride);
                let bytes: [u8; 8] = buf[start..][..8].try_into().unwrap();
                *v = _mm_cvtepu8_epi16(_mm_cvtsi64_si128(i64::from_le_bytes(bytes)));
            }
        }
        let out = compute(token, taps, levels);
        if H {
            let rows = transpose8(
                token,
                [out[0], out[1], out[2], out[3], zero, zero, zero, zero],
            );
            for (row, v) in rows.into_iter().enumerate() {
                let start = signed_idx(base, row as isize * stride - 2);
                let bytes = _mm_cvtsi128_si32(_mm_packus_epi16(v, zero)).to_le_bytes();
                buf[start..][..4].copy_from_slice(&bytes);
            }
        } else {
            for (index, v) in out.into_iter().enumerate() {
                let start = signed_idx(base, (index as isize - 2) * stride);
                let bytes = _mm_cvtsi128_si64(_mm_packus_epi16(v, zero)).to_le_bytes();
                buf[start..][..8].copy_from_slice(&bytes);
            }
        }
    }
}
