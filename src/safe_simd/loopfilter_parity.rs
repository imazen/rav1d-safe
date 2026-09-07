// Compare leaf SIMD arithmetic with the decoder's actual scalar loop filter.
// Keep this separate from picture ownership and whole-plane comparison tests.
#[cfg(all(test, target_arch = "x86_64", feature = "bitdepth_8"))]
mod mask_parity_tests {
    use super::*;

    // Index order matches the mask census, with each SIMD lane processing one
    // independent edge position. The x16 leaf requires the AVX-512 token.
    const KERNELS: [(usize, bool, usize); 16] = [
        (6, false, 4),
        (8, false, 4),
        (8, false, 8),
        (16, false, 8),
        (16, false, 16),
        (16, false, 4),
        (4, true, 4),
        (6, true, 4),
        (8, true, 4),
        (8, true, 8),
        (16, true, 4),
        (4, false, 4),
        (4, false, 8),
        (6, false, 8),
        (6, true, 8),
        (16, true, 8),
    ];

    fn run_simd(
        kernel: usize,
        buf: &mut [u8],
        base: usize,
        stride: isize,
        levels: [u8; 3],
    ) -> bool {
        let [e, i, h] = levels.map(i32::from);
        if kernel == 4 {
            let Some(token) = crate::src::cpu::summon_avx512() else {
                return false;
            };
            loop_filter_4_8bpc_wd16_simd_v_x16(token, buf, base, e, i, h, stride);
            return true;
        }
        let Some(token) = crate::src::cpu::summon_avx2() else {
            return false;
        };
        match kernel {
            0 => loop_filter_4_8bpc_wd6_simd_v(token, buf, base, e, i, h, stride),
            1 => loop_filter_4_8bpc_wd8_simd_v(token, buf, base, e, i, h, stride),
            2 => loop_filter_4_8bpc_wd8_simd_v_x8(token, buf, base, e, i, h, stride),
            3 => loop_filter_4_8bpc_wd16_simd_v_x8(token, buf, base, e, i, h, stride),
            5 => loop_filter_4_8bpc_wd16_simd_v(token, buf, base, e, i, h, stride),
            6 => loop_filter_4_8bpc_narrow_simd_h(token, buf, base, e, i, h, stride),
            7 => loop_filter_4_8bpc_wd6_simd_h(token, buf, base, e, i, h, stride),
            8 => loop_filter_4_8bpc_wd8_simd_h(token, buf, base, e, i, h, stride),
            9 => loop_filter_4_8bpc_wd8_simd_h_x8(token, buf, base, e, i, h, stride),
            10 => loop_filter_4_8bpc_wd16_simd_h(token, buf, base, e, i, h, stride),
            11 => loop_filter_4_8bpc_narrow_simd_v(token, buf, base, e, i, h, stride),
            12 => loop_filter_4_8bpc_narrow_simd_v_x8(token, buf, base, e, i, h, stride),
            13 => packed6::apply::<false>(token, buf, base, stride, levels),
            14 => packed6::apply::<true>(token, buf, base, stride, levels),
            15 => packed16::apply_h(token, buf, base, stride, levels),
            _ => unreachable!(),
        }
        true
    }

    fn next(state: &mut u64) -> u8 {
        *state ^= *state >> 12;
        *state ^= *state << 25;
        *state ^= *state >> 27;
        state.wrapping_mul(0x2545_f491_4f6c_dd1d) as u8
    }

    fn input(
        lanes: usize,
        horizontal: bool,
        stride: isize,
        offset: usize,
        pattern: usize,
    ) -> (Vec<u8>, usize, isize, isize) {
        let pitch = stride.unsigned_abs();
        let base = pitch
            * if horizontal && stride < 0 {
                lanes + 8
            } else {
                16
            }
            + 8
            + offset;
        let (stridea, strideb) = if horizontal { (stride, 1) } else { (1, stride) };
        let mut state = 0x0dec_afba_df00_d123;
        let mut buf: Vec<u8> = (0..pitch * (lanes + 32) + 64)
            .map(|_| next(&mut state))
            .collect();
        for lane in 0..lanes {
            for k in -7isize..=6 {
                let kind = match pattern {
                    4 => usize::from(lane == 0),
                    5 => lane % 2,
                    _ => pattern,
                };
                let side = if k < 0 { 64u8 } else { 68 };
                let value = match kind {
                    0 => {
                        if k % 2 == 0 {
                            0
                        } else {
                            255
                        }
                    }
                    1 => side,
                    2 => side + if !(-4..=3).contains(&k) { 12 } else { 0 },
                    3 => side + if k == -3 || k == 2 { 4 } else { 0 },
                    6 => ((k * 37 + lane as isize * 53).rem_euclid(256)) as u8,
                    7 => {
                        if k < 0 {
                            0
                        } else {
                            4
                        }
                    }
                    8 => {
                        if k < 0 {
                            251
                        } else {
                            255
                        }
                    }
                    9 => next(&mut state),
                    // Narrow HEV corrections cross zero / 255 before clipping.
                    10 => {
                        if k >= 1 {
                            8
                        } else {
                            0
                        }
                    }
                    11 => {
                        if k <= -2 {
                            247
                        } else {
                            255
                        }
                    }
                    _ => unreachable!(),
                };
                buf[base
                    .checked_add_signed(lane as isize * stridea + k * strideb)
                    .unwrap()] = value;
            }
        }
        (buf, base, stridea, strideb)
    }

    #[test]
    fn test_loopfilter_masks_match_scalar() {
        let _lock = crate::src::safe_simd::token_test_lock();
        let avx2 = crate::src::cpu::summon_avx2().is_some();
        let avx512 = crate::src::cpu::summon_avx512().is_some();
        let mut cells = 0;
        for (kernel, &(width, horizontal, lanes)) in KERNELS.iter().enumerate() {
            if !(if kernel == 4 { avx512 } else { avx2 }) {
                continue;
            }
            for stride in [32, 67, -32, -67] {
                for offset in [0, 3] {
                    for pattern in 0..12 {
                        for levels in [
                            [0, 0, 0],
                            [8, 4, 0],
                            [16, 8, 1],
                            [63, 63, 3],
                            [255, 255, 255],
                        ] {
                            let (pixels, base, stridea, strideb) =
                                input(lanes, horizontal, stride, offset, pattern);
                            let mut actual = pixels.clone();
                            let mut expected = pixels.clone();
                            assert!(run_simd(kernel, &mut actual, base, stride, levels));
                            crate::src::loopfilter::loop_filter_scalar_for_test(
                                &mut expected,
                                base,
                                [stridea, strideb],
                                lanes,
                                levels,
                                width,
                            );
                            assert_eq!(
                                actual.iter().zip(&expected).position(|(a, b)| a != b),
                                None,
                                "kernel={kernel}, width={width}, lanes={lanes}, stride={stride}, offset={offset}, pattern={pattern}, levels={levels:?}"
                            );
                            // A rejected group is unchanged; a partially eligible
                            // group still filters its active lanes. These explicit
                            // cases prevent an all-lanes/any-lane predicate mixup.
                            if pattern == 0 {
                                assert_eq!(actual, pixels);
                            } else if matches!(pattern, 1 | 4 | 5 | 10 | 11) && levels == [16, 8, 1]
                            {
                                assert_ne!(actual, pixels);
                            }
                            cells += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(cells, (usize::from(avx2) * 15 + usize::from(avx512)) * 480);
        eprintln!("loopfilter mask sweep: {cells} live SIMD cells match the scalar decoder");
    }

    #[test]
    fn test_loopfilter_packed6_production_grouping() {
        let _lock = crate::src::safe_simd::token_test_lock();
        let Some(token) = crate::src::cpu::summon_avx2() else {
            return;
        };
        let mut lut: Align16<Av1FilterLUT> = crate::src::align::ArrayDefault::default();
        lut.e.fill(16);
        lut.i.fill(8);
        lut.e[33] = 32;
        let mut cells = 0;
        for horizontal in [false, true] {
            for stride in [160, -160] {
                for byte_idx in [2, 3] {
                    for case in 0..8 {
                        let (mask, expected_fused) = match case {
                            0 | 4 => ([0, 0b11, 0], 1),
                            1 | 5 => ([0, 0b11, 0], 0),
                            2 => ([0b10, 0b01, 0], 0),
                            3 => ([0, 0b101, 0], 0),
                            6 => ([0, 3 << 30, 0], 1),
                            7 => ([0, 1 << 31, 0], 0),
                            _ => unreachable!(),
                        };
                        let b4_stride = 37usize;
                        let lvl_base = 64usize;
                        let (step, lookback) = if horizontal {
                            (b4_stride, 1)
                        } else {
                            (1, b4_stride)
                        };
                        let mut raw_levels = vec![32u8; (lvl_base + 32 * b4_stride + 1) * 4];
                        match case {
                            1 => raw_levels[(lvl_base + step) * 4 + byte_idx] = 33,
                            4 => raw_levels[(lvl_base + step) * 4 + byte_idx] = 0,
                            5 => {
                                raw_levels[lvl_base * 4 + byte_idx] = 0;
                                raw_levels[(lvl_base - lookback) * 4 + byte_idx] = 0;
                            }
                            _ => {}
                        }
                        let levels: Vec<AtomicU8> =
                            raw_levels.iter().copied().map(AtomicU8::new).collect();
                        let (pixels, base, stridea, strideb) = input(128, horizontal, stride, 3, 5);
                        let mut actual = pixels.clone();
                        let mut expected = pixels;
                        let before = packed6::calls();
                        let run = if horizontal {
                            lpf_h_sb_uv_8bpc_inner
                        } else {
                            lpf_v_sb_uv_8bpc_inner
                        };
                        run(
                            token,
                            &mut actual,
                            base,
                            stride,
                            &mask,
                            &levels,
                            lvl_base,
                            byte_idx,
                            b4_stride as isize,
                            &lut,
                            128,
                            255,
                        );
                        let after = packed6::calls();
                        assert_eq!(
                            after[usize::from(horizontal)] - before[usize::from(horizontal)],
                            expected_fused,
                            "fusion liveness: H={horizontal}, case={case}"
                        );
                        for bit in 0..32 {
                            if (mask[0] | mask[1]) & (1 << bit) == 0 {
                                continue;
                            }
                            let at = lvl_base + bit * step;
                            let value = raw_levels[at * 4 + byte_idx];
                            let l = if value != 0 {
                                value
                            } else {
                                at.checked_sub(lookback)
                                    .map_or(0, |n| raw_levels[n * 4 + byte_idx])
                            };
                            if l == 0 {
                                continue;
                            }
                            let width = if mask[1] & (1 << bit) != 0 { 6 } else { 4 };
                            crate::src::loopfilter::loop_filter_scalar_for_test(
                                &mut expected,
                                base.checked_add_signed(bit as isize * 4 * stridea).unwrap(),
                                [stridea, strideb],
                                4,
                                [lut.e[l as usize], lut.i[l as usize], l >> 4],
                                width,
                            );
                        }
                        assert_eq!(
                            actual.iter().zip(&expected).position(|(a, b)| a != b),
                            None,
                            "UV grouping: H={horizontal}, stride={stride}, byte={byte_idx}, case={case}"
                        );
                        cells += 1;
                    }
                }
            }
        }
        assert_eq!(cells, 64);
        eprintln!("packed six-tap grouping: {cells} production-path cases match scalar");
    }

    #[test]
    fn test_loopfilter_packed16_production_grouping() {
        let _lock = crate::src::safe_simd::token_test_lock();
        let Some(token) = crate::src::cpu::summon_avx2() else {
            return;
        };
        let mut lut: Align16<Av1FilterLUT> = crate::src::align::ArrayDefault::default();
        lut.e.fill(16);
        lut.i.fill(8);
        lut.e[33] = 32;
        let mut cells = 0;
        for stride in [160, -160] {
            for byte_idx in [0, 1] {
                for case in 0..8 {
                    let (mask, expected_fused) = match case {
                        0 | 4 => ([0, 0, 0b11], 1),
                        1 | 5 => ([0, 0, 0b11], 0),
                        2 => ([0, 0b10, 0b01], 0),
                        3 => ([0, 0, 0b101], 0),
                        6 => ([0, 0, 3 << 30], 1),
                        7 => ([0, 0, 1 << 31], 0),
                        _ => unreachable!(),
                    };
                    let b4_stride = 37usize;
                    let lvl_base = 64usize;
                    let mut raw_levels = vec![32u8; (lvl_base + 32 * b4_stride + 1) * 4];
                    match case {
                        1 => raw_levels[(lvl_base + b4_stride) * 4 + byte_idx] = 33,
                        4 => raw_levels[(lvl_base + b4_stride) * 4 + byte_idx] = 0,
                        5 => {
                            raw_levels[lvl_base * 4 + byte_idx] = 0;
                            raw_levels[(lvl_base - 1) * 4 + byte_idx] = 0;
                        }
                        _ => {}
                    }
                    let levels: Vec<AtomicU8> =
                        raw_levels.iter().copied().map(AtomicU8::new).collect();
                    let (pixels, base, stridea, strideb) = input(128, true, stride, 3, 5);
                    let mut actual = pixels.clone();
                    let mut expected = pixels;
                    let before = packed16::calls();
                    lpf_h_sb_y_8bpc_inner(
                        token,
                        &mut actual,
                        base,
                        stride,
                        &mask,
                        &levels,
                        lvl_base,
                        byte_idx,
                        b4_stride as isize,
                        &lut,
                        128,
                        255,
                    );
                    assert_eq!(
                        packed16::calls() - before,
                        expected_fused,
                        "wide fusion liveness: stride={stride}, byte={byte_idx}, case={case}"
                    );
                    for bit in 0..32 {
                        if (mask[0] | mask[1] | mask[2]) & (1 << bit) == 0 {
                            continue;
                        }
                        let at = lvl_base + bit * b4_stride;
                        let value = raw_levels[at * 4 + byte_idx];
                        let l = if value != 0 {
                            value
                        } else {
                            raw_levels[(at - 1) * 4 + byte_idx]
                        };
                        if l == 0 {
                            continue;
                        }
                        let width = if mask[2] & (1 << bit) != 0 { 16 } else { 8 };
                        crate::src::loopfilter::loop_filter_scalar_for_test(
                            &mut expected,
                            base.checked_add_signed(bit as isize * 4 * stridea).unwrap(),
                            [stridea, strideb],
                            4,
                            [lut.e[l as usize], lut.i[l as usize], l >> 4],
                            width,
                        );
                    }
                    assert_eq!(
                        actual.iter().zip(&expected).position(|(a, b)| a != b),
                        None,
                        "wide H grouping: stride={stride}, byte={byte_idx}, case={case}"
                    );
                    cells += 1;
                }
            }
        }
        assert_eq!(cells, 32);
        eprintln!("packed wide H grouping: {cells} production-path cases match scalar");
    }

    #[test]
    fn test_loopfilter_packed16_exact_span() {
        let _lock = crate::src::safe_simd::token_test_lock();
        let Some(token) = crate::src::cpu::summon_avx2() else {
            return;
        };
        let mut cells = 0;
        for stride in [14, -14, 19, -19] {
            for offset in [0, 3] {
                for pattern in [1, 3, 10, 11] {
                    let (pixels, base, stridea, strideb) = input(8, true, stride, offset, pattern);
                    let other = base.checked_add_signed(7 * stride).unwrap();
                    let start = base.min(other) - 7;
                    let end = base.max(other) + 7;
                    let levels = [16, 8, 1];
                    let mut actual = pixels.clone();
                    let mut expected = pixels.clone();
                    crate::src::loopfilter::loop_filter_scalar_for_test(
                        &mut expected,
                        base,
                        [stridea, strideb],
                        8,
                        levels,
                        16,
                    );
                    packed16::apply_h(token, &mut actual[start..end], base - start, stride, levels);
                    assert_eq!(
                        actual, expected,
                        "exact H span: stride={stride}, offset={offset}, pattern={pattern}"
                    );
                    assert_ne!(actual, pixels, "exact-span positive control must filter");
                    let mut short = pixels.clone();
                    let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        packed16::apply_h(
                            token,
                            &mut short[start..end - 1],
                            base - start,
                            stride,
                            levels,
                        );
                    }));
                    assert!(rejected.is_err(), "short H span accepted");
                    assert_eq!(
                        short, pixels,
                        "short view must fail while loading, before writes"
                    );
                    cells += 1;
                }
            }
        }
        assert_eq!(cells, 32);
        eprintln!("packed wide H footprint: {cells} exact spans and {cells} short-view rejections");
    }
}
