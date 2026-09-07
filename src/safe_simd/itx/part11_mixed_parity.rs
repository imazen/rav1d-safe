// Exercise production dispatch against the real scalar fallback, including
// scan boundaries, coefficient clearing, and pixels outside the output block.
#[cfg(all(
    test,
    target_arch = "x86_64",
    feature = "bitdepth_8",
    not(feature = "c-ffi")
))]
mod mixed_parity_tests {
    use super::*;
    use crate::include::common::bitdepth::BitDepth8;
    use crate::include::dav1d::picture::Rav1dPictureDataComponent;
    use crate::src::levels::{TxClass, TxfmSize};
    use crate::src::owned_recon::ReconDst;
    use crate::src::scan::dav1d_scans;
    use crate::src::tables::dav1d_tx_type_class;

    const TYPES: [TxfmType; 14] = [
        ADST_DCT,
        DCT_ADST,
        ADST_ADST,
        FLIPADST_DCT,
        DCT_FLIPADST,
        FLIPADST_FLIPADST,
        ADST_FLIPADST,
        FLIPADST_ADST,
        H_DCT,
        V_DCT,
        H_ADST,
        V_ADST,
        H_FLIPADST,
        V_FLIPADST,
    ];

    fn next(state: &mut u64) -> u64 {
        *state ^= *state >> 12;
        *state ^= *state << 25;
        *state ^= *state >> 27;
        state.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn check_cell(
        tx: TxfmSize,
        tx_type: TxfmType,
        eob: usize,
        pattern: usize,
        stride: usize,
        offset: usize,
        live: bool,
    ) {
        let (w, h) = tx.to_wh();
        let count = w * h;
        let bd = BitDepth8::new(());
        let mut state = 0xdeca_fbad_f00d_u64 ^ u64::from(tx_type) ^ (eob as u64) << 8;
        // Extra coefficients and full rows around the destination are sentinels.
        let mut input = vec![0i16; count + 16];
        input[count..].fill(0x1234);
        for i in 0..=eob {
            let pos = match dav1d_tx_type_class[tx_type as usize] {
                TxClass::TwoD => dav1d_scans[tx as usize][i].get() as usize,
                TxClass::H => i,
                TxClass::V => (i & (w - 1)) * h + (i >> w.trailing_zeros()),
            };
            input[pos] = match pattern {
                0 => 1,
                1 => -1,
                2 => i16::MAX,
                3 => i16::MIN,
                4 => {
                    if i % 2 == 0 {
                        i16::MIN
                    } else {
                        i16::MAX
                    }
                }
                5 => (next(&mut state) % 4096) as i16 - 2048,
                6 => {
                    if i == eob {
                        256
                    } else {
                        0
                    }
                }
                7 => next(&mut state) as i16,
                _ => unreachable!(),
            };
            if i == eob && input[pos] == 0 {
                input[pos] = 1;
            }
        }
        let pixels: Vec<u8> = (0..(stride * (h + 2)).next_multiple_of(64))
            .map(|i| match i % 4 {
                0 => 0,
                1 => 128,
                2 => 255,
                _ => next(&mut state) as u8,
            })
            .collect();
        let run = |simd: bool| {
            let mut source = crate::src::safe_simd::aligned_plane(&pixels);
            let mut coeff = input.clone();
            let comp = Rav1dPictureDataComponent::wrap_buf::<BitDepth8>(&mut source, stride);
            let mut dst = ReconDst::Pic(comp.with_offset::<BitDepth8>() + offset as isize);
            if simd {
                assert_eq!(
                    itxfm_add_dispatch::<BitDepth8>(
                        tx as usize,
                        tx_type as usize,
                        &mut dst,
                        &mut coeff,
                        eob as i32,
                        bd,
                    ),
                    live,
                    "dispatch liveness, type={tx_type}"
                );
            } else {
                crate::src::itx::itxfm_add_scalar_fallback::<BitDepth8>(
                    tx as usize,
                    tx_type,
                    &mut dst,
                    &mut coeff,
                    eob as i32,
                    bd,
                );
            }
            let mut out = vec![0; pixels.len()];
            comp.copy_pixels_to::<BitDepth8>(&mut out);
            (out, coeff)
        };
        let actual = run(true);
        if !live {
            assert_eq!(actual, (pixels, input), "declined dispatch wrote data");
            return;
        }
        let expected = run(false);
        assert_eq!(
            actual.0, expected.0,
            "pixels: {w}x{h}, type={tx_type}, eob={eob}, pattern={pattern}, stride={stride}, offset={offset}"
        );
        assert_eq!(actual.1, expected.1, "coefficient clearing: type={tx_type}");
        assert!(actual.1[..count].iter().all(|&c| c == 0));
        assert_eq!(&actual.1[count..], &input[count..]);
        for (i, (&out, &old)) in actual.0.iter().zip(&pixels).enumerate() {
            let in_block = i >= offset && (i - offset) / stride < h && (i - offset) % stride < w;
            if !in_block {
                assert_eq!(out, old, "write outside block at {i}");
            }
        }
    }

    fn sweep(tx: TxfmSize, eobs: &[usize], expected_cells: usize) {
        let _lock = crate::src::safe_simd::token_test_lock();
        if crate::src::cpu::summon_avx2().is_none() {
            eprintln!("Skipping mixed SIMD sweep: AVX2 token unavailable");
            return;
        }
        let (w, h) = tx.to_wh();
        let mut cells = 0;
        for tx_type in TYPES {
            for &eob in eobs {
                for pattern in 0..8 {
                    for stride in [w * 2, w * 4] {
                        for offset in [0, stride + 3] {
                            check_cell(tx, tx_type, eob, pattern, stride, offset, true);
                            cells += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(cells, expected_cells);
        eprintln!("mixed {w}x{h}: {cells} live production dispatch cells match scalar");
    }

    fn token_sweep(tx: TxfmSize) {
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
        let _lock = crate::src::safe_simd::token_test_lock();
        let supported = crate::src::cpu::summon_avx2().is_some();
        let (w, h) = tx.to_wh();
        let mut live_cells = 0;
        let mut declined_cells = 0;
        let report = for_each_token_permutation(CompileTimePolicy::WarnStderr, |_| {
            let live = crate::src::cpu::summon_avx2().is_some();
            for tx_type in TYPES {
                for eob in [0, 31, w * h - 1] {
                    for pattern in [5, 7] {
                        check_cell(tx, tx_type, eob, pattern, 32, 35, live);
                        if live {
                            live_cells += 1;
                        } else {
                            declined_cells += 1;
                        }
                    }
                }
            }
        });
        assert!(report.permutations_run >= 2);
        assert!(
            !supported || live_cells > 0,
            "SIMD dispatch was never exercised"
        );
        assert!(declined_cells > 0, "disabled dispatch was never exercised");
        eprintln!("mixed {w}x{h} token sweep: {live_cells} SIMD, {declined_cells} declined");
    }

    #[test]
    fn test_mixed16_dispatch_matches_scalar() {
        sweep(
            TxfmSize::S16x16,
            &[0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255],
            5824,
        );
    }

    #[test]
    fn test_mixed16_dispatch_cpu_permutations() {
        token_sweep(TxfmSize::S16x16);
    }

    #[test]
    fn test_mixed8_dispatch_matches_scalar() {
        sweep(
            TxfmSize::S8x8,
            &[0, 1, 3, 4, 7, 8, 15, 16, 31, 32, 63],
            4928,
        );
    }

    #[test]
    fn test_mixed8_dispatch_cpu_permutations() {
        token_sweep(TxfmSize::S8x8);
    }

    #[test]
    fn test_mixed8_api_rejects_invalid_views_before_writing() {
        let _lock = crate::src::safe_simd::token_test_lock();
        let Some(token) = crate::src::cpu::summon_avx2() else {
            return;
        };
        // These entry points share the checked-prefix macro. Test refusal
        // before writes, including arithmetic overflow and the final pixel.
        for (dst_len, stride, coeff_len) in [
            (0, 8, 64),
            (63, 8, 64),
            (64, 8, 63),
            (64, 8, 0),
            (126, 17, 64),
            (128, usize::MAX, 64),
            (128, usize::MAX / 7, 64),
        ] {
            let mut pixels = [128; 128];
            let mut coeff = [64; 64];
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                inv_txfm_add_adst_adst_8x8_8bpc_avx2_inner(
                    token,
                    &mut pixels[..dst_len],
                    stride,
                    &mut coeff[..coeff_len],
                    63,
                    255,
                );
            }));
            assert!(
                result.is_err(),
                "invalid view accepted: {dst_len}, {stride}, {coeff_len}"
            );
            assert_eq!(pixels, [128; 128], "rejected call changed pixels");
            assert_eq!(coeff, [64; 64], "rejected call changed coefficients");
        }
        // Positive control: exactly sized views must still be accepted.
        let mut pixels = [128; 64];
        let mut coeff = [0; 64];
        coeff[0] = 1024;
        inv_txfm_add_adst_adst_8x8_8bpc_avx2_inner(token, &mut pixels, 8, &mut coeff, 0, 255);
        assert!(pixels.iter().any(|&p| p != 128));
        assert_eq!(coeff, [0; 64]);
    }
}
