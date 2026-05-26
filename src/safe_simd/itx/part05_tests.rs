// ============================================================================
// TESTS
// ============================================================================

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;

    #[test]
    fn test_wht4_basic() {
        // WHT is used for lossless mode - test basic functionality
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let mut coeff = [0i16; 16];
        coeff[0] = 64; // DC coefficient

        let mut dst = [128u8; 16];
        let stride = 4usize;

        let token = crate::src::cpu::summon_avx2().expect("AVX2 required");
        inv_txfm_add_wht_wht_4x4_8bpc_avx2_inner(token, &mut dst, stride, &mut coeff, 1, 255);

        // Should have added DC to all pixels
        assert!(dst.iter().all(|&p| p >= 128));
        assert!(coeff.iter().all(|&c| c == 0));
    }

    /// Verify WHT 4x4 produces consistent output across all token permutations.
    /// When tokens are disabled, summon_avx2() returns None (dispatch would fall back).
    /// When enabled, SIMD output must match the known-good reference.
    #[test]
    fn test_wht4_token_permutations() {
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

        // Compute reference output once with tokens fully enabled
        let reference = {
            let Some(token) = crate::src::cpu::summon_avx2() else {
                eprintln!("Skipping: AVX2 not available");
                return;
            };
            let mut coeff = [0i16; 16];
            coeff[0] = 64;
            let mut dst = [128u8; 16];
            inv_txfm_add_wht_wht_4x4_8bpc_avx2_inner(token, &mut dst, 4, &mut coeff, 1, 255);
            dst
        };

        let report = for_each_token_permutation(CompileTimePolicy::WarnStderr, |perm| {
            if let Some(token) = crate::src::cpu::summon_avx2() {
                let mut coeff = [0i16; 16];
                coeff[0] = 64;
                let mut dst = [128u8; 16];
                inv_txfm_add_wht_wht_4x4_8bpc_avx2_inner(token, &mut dst, 4, &mut coeff, 1, 255);
                assert_eq!(dst, reference, "WHT output mismatch at: {perm}");
                assert!(
                    coeff.iter().all(|&c| c == 0),
                    "coeffs not zeroed at: {perm}"
                );
            }
            // When token disabled, summon returns None — dispatch would fall back to scalar
        });
        eprintln!("WHT permutations: {}", report.permutations_run);
        assert!(report.permutations_run >= 1);
    }

    // ----------------------------------------------------------------------
    // itx_mul2x_pack! — bit-exact match against scalar reference
    // ----------------------------------------------------------------------

    /// Test helper exercising `itx_mul2x_pack!` from within a target_feature
    /// scope. Computes `madd(paired) >> 12` (dav1d row-pass shift) for each i32
    /// lane, where `paired` is constructed from input i16 arrays `a`, `b` via
    /// `unpacklo_epi16(b, a)` so each lane = `(a_word << 16) | b_word` — same
    /// shape as dav1d's `punpcklwd m_a, m_b` output. Returns the 8 i32 results
    /// post-shift; caller compares against scalar arithmetic.
    ///
    /// `#[arcane]` so the test can call it without an `unsafe` block — the
    /// archmage attribute handles target_feature scoping internally.
    #[arcane]
    fn itx_mul2x_pack_probe(
        _token: Desktop64,
        a: [i16; 16],
        b: [i16; 16],
        coef_a: i16,
        coef_b: i16,
    ) -> [i32; 16] {
        // Load i16 vectors (16 lanes = 256 bit).
        let arr_a: &[i16; 16] = &a;
        let arr_b: &[i16; 16] = &b;
        let va = loadu_256!(arr_a, [i16; 16]);
        let vb = loadu_256!(arr_b, [i16; 16]);
        // Pair: each 32-bit lane = (a_word << 16) | b_word, i.e. low 16-bit
        // half is `a` and high 16-bit half is `b`. This is the dav1d
        // ITX_MUL2X_PACK input shape (the `unpacklo m1, m_a, m_b` form).
        let lo = _mm256_unpacklo_epi16(va, vb);
        let hi = _mm256_unpackhi_epi16(va, vb);
        // Apply macro with rnd = 2048 (== pd_2048), shift = 12.
        let r_lo = itx_mul2x_pack!(lo, coef_a, coef_b, 2048, 12);
        let r_hi = itx_mul2x_pack!(hi, coef_a, coef_b, 2048, 12);
        // Store post-shift i32 lanes.
        let mut out_lo = [0i32; 8];
        let mut out_hi = [0i32; 8];
        let arr_lo: &mut [i32; 8] = &mut out_lo;
        let arr_hi: &mut [i32; 8] = &mut out_hi;
        storeu_256!(arr_lo, [i32; 8], r_lo);
        storeu_256!(arr_hi, [i32; 8], r_hi);
        // Re-interleave so output[i] corresponds to input pair (a[i], b[i]).
        // AVX2 unpacklo/unpackhi work within 128-bit lanes, so:
        //   out_lo[0..4] ← (a[0..4], b[0..4]) pairs
        //   out_hi[0..4] ← (a[4..8], b[4..8]) pairs
        //   out_lo[4..8] ← (a[8..12], b[8..12]) pairs
        //   out_hi[4..8] ← (a[12..16], b[12..16]) pairs
        let mut out = [0i32; 16];
        out[0..4].copy_from_slice(&out_lo[0..4]);
        out[4..8].copy_from_slice(&out_hi[0..4]);
        out[8..12].copy_from_slice(&out_lo[4..8]);
        out[12..16].copy_from_slice(&out_hi[4..8]);
        out
    }

    /// Bit-exact check: itx_mul2x_pack! matches the scalar C reference
    /// `(a*c1 + b*c2 + rnd) >> 12` on a wide range of seeded random inputs.
    ///
    /// This is the regression gate for any future i16-packed row DCT work:
    /// before swapping a scalar 1D path for a `itx_mul2x_pack!`-driven one,
    /// the equivalent test must extend here. The scalar formula is the
    /// arithmetic dav1d's C reference performs — match it bit-exactly.
    #[test]
    fn test_itx_mul2x_pack_matches_scalar() {
        let Some(token) = crate::src::cpu::summon_avx2() else {
            eprintln!("Skipping: AVX2 not available");
            return;
        };
        // Coefficient pairs from the actual AV1 trig tables (dct8/16 row pass).
        // Mix of (small, small), (small, large), (negative, positive),
        // (positive, negative) — the four sign permutations exercised by
        // ITX_MULSUB_2W call sites.
        let coef_pairs: &[(i16, i16)] = &[
            (799, 4017),   // dct8 t4a/t7a
            (3406, 2276),  // dct8 t5a/t6a
            (1567, 3784),  // dct8 t2/t3
            (2896, 2896),  // dct8 t0/t1
            (-2896, 2896), // dct8 t6/t5
            (401, 4076),   // dct16 t8a/t15a
            (3166, 2598),  // dct16 t9a/t14a
            (1931, 3612),  // dct16 t10a/t13a
            (3920, 1189),  // dct16 t11a/t12a
            (-3784, 1567), // dct16 t10a (oddhalf)
            (i16::MIN, 1), // sign extreme
            (1, i16::MAX), // sign extreme
            (i16::MAX, i16::MIN),
        ];

        // Seed-based LCG (no external deps). Lcg64Xsh32-style.
        let mut state: u64 = 0xdead_beef_cafe_babe;
        let mut next_i16 = || -> i16 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 32) as i32 as i16
        };

        const N_TRIALS: usize = 64;
        let mut mismatches = 0u32;
        for trial in 0..N_TRIALS {
            // Random i16 vectors with broad coverage including the high end
            // (where overflow would manifest in a wrong implementation).
            let mut a = [0i16; 16];
            let mut b = [0i16; 16];
            for i in 0..16 {
                a[i] = next_i16();
                b[i] = next_i16();
            }
            // Occasionally inject saturated inputs to exercise overflow corners.
            if trial % 8 == 0 {
                a[trial % 16] = i16::MAX;
                b[(trial + 1) % 16] = i16::MIN;
            }
            for &(c1, c2) in coef_pairs {
                let simd_out = itx_mul2x_pack_probe(token, a, b, c1, c2);
                for i in 0..16 {
                    // Scalar C arithmetic: `(a*c1 + b*c2 + 2048) >> 12`.
                    let scalar =
                        ((a[i] as i32) * (c1 as i32) + (b[i] as i32) * (c2 as i32) + 2048) >> 12;
                    if scalar != simd_out[i] {
                        if mismatches < 8 {
                            eprintln!(
                                "MISMATCH trial={trial} i={i} c1={c1} c2={c2} a={} b={} simd={} scalar={}",
                                a[i], b[i], simd_out[i], scalar
                            );
                        }
                        mismatches += 1;
                    }
                }
            }
        }
        assert_eq!(
            mismatches, 0,
            "itx_mul2x_pack! diverged from scalar C reference (a*c1 + b*c2 + 2048) >> 12"
        );
    }

    // ----------------------------------------------------------------------
    // transpose_8x8_i32! — round-trip identity
    // ----------------------------------------------------------------------

    /// `#[arcane]` helper that builds an 8x8 i32 column-major block, runs the
    /// transpose macro, and stores row-major. Returns the row-major output.
    /// Caller compares against a scalar transpose of the input.
    #[arcane]
    fn transpose_8x8_probe(_token: Desktop64, input_col_major: [i32; 64]) -> [i32; 64] {
        // Load 8 columns into 8 __m256i (each column is 8 i32 lanes).
        let mut cols = [_mm256_setzero_si256(); 8];
        for x in 0..8 {
            let arr: &[i32; 8] = (&input_col_major[x * 8..x * 8 + 8]).try_into().unwrap();
            cols[x] = loadu_256!(arr, [i32; 8]);
        }
        let rows = transpose_8x8_i32!(cols);
        let mut out = [0i32; 64];
        for y in 0..8 {
            let arr: &mut [i32; 8] = (&mut out[y * 8..y * 8 + 8]).try_into().unwrap();
            storeu_256!(arr, [i32; 8], rows[y]);
        }
        out
    }

    /// transpose_8x8_i32! returns the row-major image of the column-major
    /// input — i.e. `out[y*8 + x] == in[x*8 + y]` for all (x, y).
    #[test]
    fn test_transpose_8x8_i32_roundtrip() {
        let Some(token) = crate::src::cpu::summon_avx2() else {
            eprintln!("Skipping: AVX2 not available");
            return;
        };
        // Deterministic checkerboard inputs to make off-by-one shuffles
        // obvious in stderr if the macro regresses.
        let mut col_major = [0i32; 64];
        for x in 0..8 {
            for y in 0..8 {
                // Distinct value per (x, y) — high bits encode column, low
                // bits encode row, so a transpose error shows as a clean
                // bit-pattern mismatch.
                col_major[x * 8 + y] = ((x as i32) << 24) | ((y as i32) << 8) | 0x55;
            }
        }
        let row_major = transpose_8x8_probe(token, col_major);
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(
                    row_major[y * 8 + x],
                    col_major[x * 8 + y],
                    "transpose mismatch at (x={x}, y={y})"
                );
            }
        }
        // Also try a randomized input.
        let mut state: u64 = 0xface_feed_dead_beef;
        let mut rand = || -> i32 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state as i32
        };
        let mut col_major2 = [0i32; 64];
        for v in col_major2.iter_mut() {
            *v = rand();
        }
        let row_major2 = transpose_8x8_probe(token, col_major2);
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(
                    row_major2[y * 8 + x],
                    col_major2[x * 8 + y],
                    "transpose (random) mismatch at (x={x}, y={y})"
                );
            }
        }
    }

    // ----------------------------------------------------------------------
    // Scalar reference for i16-packed DCT row-pass attempts.
    // Future T-5 (i16-packed pmaddwd DCT) work calls this as the
    // bit-exactness oracle; iterates in milliseconds vs 30-second MD5 cycles.
    // ----------------------------------------------------------------------

    /// Run scalar `rav1d_inv_dct8_1d_c` per row on column-major i16 coeff
    /// (the dav1d 8x8 row-pass input shape). Returns row-major i32 output
    /// matching what `simd_row_dct8_8bpc_8rows` writes to `tmp` before the
    /// rounding/shift step.
    ///
    /// Bit-exactness target: ANY new SIMD row-pass implementation must
    /// produce byte-identical i32 output for the same input. Use:
    /// ```ignore
    /// let scalar_out = run_scalar_dct8_per_row(&input);
    /// let simd_out   = call_new_simd_row_pass(&input);
    /// assert_eq!(scalar_out, simd_out);
    /// ```
    #[allow(dead_code)]
    pub(super) fn run_scalar_dct8_per_row(
        coeff_col_major: &[i16; 64],
        row_min: i32,
        row_max: i32,
    ) -> [i32; 64] {
        use std::num::NonZeroUsize;
        let mut tmp_row_major = [0i32; 64];
        for y in 0..8 {
            let mut row = [0i32; 8];
            for x in 0..8 {
                row[x] = coeff_col_major[y + x * 8] as i32;
            }
            crate::src::itx_1d::rav1d_inv_dct8_1d_c(
                &mut row,
                NonZeroUsize::new(1).unwrap(),
                row_min,
                row_max,
            );
            for x in 0..8 {
                tmp_row_major[y * 8 + x] = row[x];
            }
        }
        tmp_row_major
    }

    /// Same shape for DCT-16: 16 rows × 16 cols column-major i16 input
    /// → row-major i32 output (each row independently transformed).
    #[allow(dead_code)]
    pub(super) fn run_scalar_dct16_per_row(
        coeff_col_major: &[i16; 256],
        row_min: i32,
        row_max: i32,
    ) -> [i32; 256] {
        use std::num::NonZeroUsize;
        let mut tmp_row_major = [0i32; 256];
        for y in 0..16 {
            let mut row = [0i32; 16];
            for x in 0..16 {
                row[x] = coeff_col_major[y + x * 16] as i32;
            }
            crate::src::itx_1d::rav1d_inv_dct16_1d_c(
                &mut row,
                NonZeroUsize::new(1).unwrap(),
                row_min,
                row_max,
            );
            for x in 0..16 {
                tmp_row_major[y * 16 + x] = row[x];
            }
        }
        tmp_row_major
    }

    /// Same shape for DCT-32: 32 rows × 32 cols column-major i16 input
    /// → row-major i32 output (each row independently transformed).
    #[allow(dead_code)]
    pub(super) fn run_scalar_dct32_per_row(
        coeff_col_major: &[i16; 1024],
        row_min: i32,
        row_max: i32,
    ) -> [i32; 1024] {
        use std::num::NonZeroUsize;
        let mut tmp_row_major = [0i32; 1024];
        for y in 0..32 {
            let mut row = [0i32; 32];
            for x in 0..32 {
                row[x] = coeff_col_major[y + x * 32] as i32;
            }
            crate::src::itx_1d::rav1d_inv_dct32_1d_c(
                &mut row,
                NonZeroUsize::new(1).unwrap(),
                row_min,
                row_max,
            );
            for x in 0..32 {
                tmp_row_major[y * 32 + x] = row[x];
            }
        }
        tmp_row_major
    }

    /// Seeded deterministic RNG for bit-exact tests — small LCG, no deps.
    #[allow(dead_code)]
    pub(super) fn seeded_i16_block<const N: usize>(seed: u64) -> [i16; N] {
        let mut state = seed.wrapping_mul(6364136223846793005);
        let mut out = [0i16; N];
        for v in out.iter_mut() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Reduce to i16 range; allow full ±32767 to exercise saturation paths.
            *v = (state >> 32) as i16;
        }
        out
    }

    /// Sanity check on the test harness itself: the scalar reference helpers
    /// must accept any seed without panicking and produce deterministic
    /// output. If this fails, no other DCT test can be trusted.
    #[test]
    fn test_dct_scalar_reference_helpers_are_deterministic() {
        let row_min = i16::MIN as i32;
        let row_max = i16::MAX as i32;

        let in8: [i16; 64] = seeded_i16_block(0xdeadbeef);
        let a8 = run_scalar_dct8_per_row(&in8, row_min, row_max);
        let b8 = run_scalar_dct8_per_row(&in8, row_min, row_max);
        assert_eq!(a8, b8, "dct8 scalar reference is non-deterministic");

        let in16: [i16; 256] = seeded_i16_block(0xc0ffee);
        let a16 = run_scalar_dct16_per_row(&in16, row_min, row_max);
        let b16 = run_scalar_dct16_per_row(&in16, row_min, row_max);
        assert_eq!(a16, b16, "dct16 scalar reference is non-deterministic");

        let in32: [i16; 1024] = seeded_i16_block(0xfeedface);
        let a32 = run_scalar_dct32_per_row(&in32, row_min, row_max);
        let b32 = run_scalar_dct32_per_row(&in32, row_min, row_max);
        assert_eq!(a32, b32, "dct32 scalar reference is non-deterministic");
    }

    // ----------------------------------------------------------------------
    // i16-packed pmaddwd DCT-8 row pass — bit-exact vs scalar
    // ----------------------------------------------------------------------

    /// Bit-exact check: `dct8_row_pass_i16_simd` matches
    /// `run_scalar_dct8_per_row` across a range of seeded inputs.
    ///
    /// While `dct8_row_pass_i16_simd` is a stub returning zeros, this test
    /// FAILS — that's the closed iteration loop. Each new SIMD stage added
    /// to the function body must keep this test green.
    #[test]
    fn test_dct8_row_pass_i16_simd_matches_scalar() {
        let Some(token) = crate::src::cpu::summon_avx2() else {
            eprintln!("Skipping: AVX2 not available");
            return;
        };
        let row_min = i16::MIN as i32;
        let row_max = i16::MAX as i32;
        let seeds: &[u64] = &[
            0xdeadbeef,
            0xc0ffee,
            0xfeedface,
            0xbaadf00d,
            0x12345678,
            0xa5a5a5a5,
            0x5a5a5a5a,
            0xffff_ffff_ffff_ffff,
        ];
        let mut total_mismatches = 0u32;
        for &seed in seeds {
            let input: [i16; 64] = seeded_i16_block(seed);
            let scalar_out = run_scalar_dct8_per_row(&input, row_min, row_max);
            let simd_out = dct8_row_pass_i16_simd(token, input);
            if simd_out != scalar_out {
                let mut mism = 0u32;
                for i in 0..64 {
                    if simd_out[i] != scalar_out[i] {
                        if mism < 8 {
                            eprintln!(
                                "seed={:#x} idx={} row={} col={} scalar={} simd={} diff={}",
                                seed,
                                i,
                                i / 8,
                                i % 8,
                                scalar_out[i],
                                simd_out[i],
                                simd_out[i].wrapping_sub(scalar_out[i])
                            );
                        }
                        mism += 1;
                    }
                }
                eprintln!("seed={seed:#x}: {mism} mismatches");
                total_mismatches += mism;
            }
        }
        assert_eq!(
            total_mismatches, 0,
            "dct8_row_pass_i16_simd diverged from scalar reference"
        );
    }

    // ----------------------------------------------------------------------
    // i16-packed pmaddwd DCT-8 COLUMN pass — bit-exact vs i32 mullo col pass
    // ----------------------------------------------------------------------

    /// Bit-exact check: `dct8_col_pass_i16` matches `dct8_1d_cols8` (i32 mullo).
    /// Uses the full pipeline (row pass → intermediate shift → col pass) to
    /// exercise the i16-packed pmaddwd column pass against the i32 reference.
    ///
    /// The test body calls `inv_txfm_add_dct_dct_8x8_8bpc_avx2_inner` (which
    /// now uses dct8_col_pass_i16) and the scalar reference, comparing dst output.
    #[test]
    fn test_dct8_col_pass_i16_matches_i32_col_pass() {
        let Some(token) = crate::src::cpu::summon_avx2() else {
            eprintln!("Skipping: AVX2 not available");
            return;
        };
        // Test via the full 2D transform pipeline: compare safe SIMD output
        // (which now uses dct8_col_pass_i16) against the scalar reference.
        let seeds: &[u64] = &[
            0xdeadbeef,
            0xc0ffee,
            0xfeedface,
            0xbaadf00d,
            0x12345678,
            0xa5a5a5a5,
            0x5a5a5a5a,
            0xffff_ffff_ffff_ffff,
            0x0,
            0x7fff_7fff_7fff_7fff,
            0x8000_8000_8000_8000,
        ];
        let row_min = i16::MIN as i32;
        let row_max = i16::MAX as i32;
        let col_min = i16::MIN as i32;
        let col_max = i16::MAX as i32;
        let mut total_mism = 0u32;
        for &seed in seeds {
            let input: [i16; 64] = seeded_i16_block(seed);

            // --- Scalar reference: full 2D pipeline ---
            // 1. Row pass (scalar)
            let row_out = run_scalar_dct8_per_row(&input, row_min, row_max);
            // 2. Intermediate shift+clip
            let mut scalar_tmp = [0i32; 64];
            for i in 0..64 {
                scalar_tmp[i] = ((row_out[i] + 1) >> 1).clamp(col_min, col_max);
            }
            // 3. Column pass (scalar) — iterate by column
            let mut scalar_col_out = [0i32; 64];
            for x in 0..8 {
                let mut col = [0i32; 8];
                for y in 0..8 {
                    col[y] = scalar_tmp[y * 8 + x];
                }
                crate::src::itx_1d::rav1d_inv_dct8_1d_c(
                    &mut col,
                    NonZeroUsize::new(1).unwrap(),
                    col_min,
                    col_max,
                );
                for y in 0..8 {
                    scalar_col_out[y * 8 + x] = col[y];
                }
            }

            // --- SIMD: full 2D pipeline using the wired function ---
            let mut dst_simd = [128u8; 8 * 8]; // neutral start
            let mut coeff_simd = input;
            inv_txfm_add_dct_dct_8x8_8bpc_avx2_inner(
                token,
                &mut dst_simd,
                8,
                &mut coeff_simd,
                64,
                255,
            );

            // --- Apply the same add-to-dst to the scalar output ---
            let mut dst_scalar = [128u8; 8 * 8];
            for y in 0..8 {
                for x in 0..8 {
                    let c = scalar_col_out[y * 8 + x];
                    let scaled = (c + 8) >> 4;
                    let p = (dst_scalar[y * 8 + x] as i32 + scaled).clamp(0, 255);
                    dst_scalar[y * 8 + x] = p as u8;
                }
            }

            // Compare
            if dst_simd != dst_scalar {
                let mut mism = 0u32;
                for i in 0..64 {
                    if dst_simd[i] != dst_scalar[i] {
                        if mism < 8 {
                            eprintln!(
                                "seed={:#x} idx={} row={} col={} scalar={} simd={}",
                                seed,
                                i,
                                i / 8,
                                i % 8,
                                dst_scalar[i],
                                dst_simd[i],
                            );
                        }
                        mism += 1;
                    }
                }
                eprintln!("seed={seed:#x}: {mism} dst mismatches");
                total_mism += mism;
            }
        }
        assert_eq!(
            total_mism, 0,
            "dct8_col_pass_i16 full pipeline diverged from scalar reference"
        );
    }

    // ----------------------------------------------------------------------
    // i16-packed pmaddwd DCT-16 COLUMN pass — bit-exact vs i32 mullo col pass
    // ----------------------------------------------------------------------

    /// #[arcane] helper: runs SIMD row pass → shift+clip → i16-packed column pass
    /// and returns the 256 i32 column-pass output for comparison with scalar.
    #[cfg(target_arch = "x86_64")]
    #[arcane]
    fn test_dct16_col_i16_pipeline(_token: Desktop64, input: [i16; 256]) -> [i32; 256] {
        let col_min = i16::MIN as i32;
        let col_max = i16::MAX as i32;

        // 1. SIMD row pass
        let simd_row_out = dct16_row_pass_i16_simd(_token, input);

        // 2. Intermediate shift+clip (shift=2, rnd=2 for 16x16)
        let mut simd_tmp = [0i32; 256];
        let rnd_v = _mm256_set1_epi32(2);
        let col_min_v = _mm256_set1_epi32(col_min);
        let col_max_v = _mm256_set1_epi32(col_max);
        for y in 0..16 {
            for chunk in 0..2u32 {
                let b = (chunk * 8) as usize;
                let off = y * 16 + b;
                let v = loadu_256!(&simd_row_out[off..off + 8], [i32; 8]);
                let shifted = _mm256_srai_epi32::<2>(_mm256_add_epi32(v, rnd_v));
                let clamped = _mm256_max_epi32(_mm256_min_epi32(shifted, col_max_v), col_min_v);
                storeu_256!(&mut simd_tmp[off..off + 8], [i32; 8], clamped);
            }
        }

        // 3. i16-packed column pass
        dct16_col_pass_i16(_token, &simd_tmp)
    }

    /// Bit-exact check: `dct16_col_pass_i16` matches the scalar reference
    /// across a range of seeded inputs.
    ///
    /// Full 2D pipeline: row pass → intermediate shift → col pass → add-to-dst.
    /// Compares final pixel output between scalar reference and the i16-packed
    /// pmaddwd column pass.
    #[test]
    fn test_dct16_col_pass_i16_matches_existing() {
        let Some(token) = crate::src::cpu::summon_avx2() else {
            eprintln!("Skipping: AVX2 not available");
            return;
        };
        let seeds: &[u64] = &[
            0xdeadbeef,
            0xc0ffee,
            0xfeedface,
            0xbaadf00d,
            0x12345678,
            0xa5a5a5a5,
            0x5a5a5a5a,
            0xffff_ffff_ffff_ffff,
            0x0,
            0x7fff_7fff_7fff_7fff,
            0x8000_8000_8000_8000,
        ];
        let row_min = i16::MIN as i32;
        let row_max = i16::MAX as i32;
        let col_min = i16::MIN as i32;
        let col_max = i16::MAX as i32;
        let mut total_mism = 0u32;

        for &seed in seeds {
            let input: [i16; 256] = seeded_i16_block(seed);

            // --- Scalar reference: full 2D pipeline ---
            // 1. Row pass (scalar)
            let row_out = run_scalar_dct16_per_row(&input, row_min, row_max);
            // 2. Intermediate shift+clip (shift=2, rnd=2 for 16x16)
            let mut scalar_tmp = [0i32; 256];
            for i in 0..256 {
                scalar_tmp[i] = ((row_out[i] + 2) >> 2).clamp(col_min, col_max);
            }
            // 3. Column pass (scalar) — iterate by column
            let mut scalar_col_out = [0i32; 256];
            for x in 0..16 {
                let mut col = [0i32; 16];
                for y in 0..16 {
                    col[y] = scalar_tmp[y * 16 + x];
                }
                crate::src::itx_1d::rav1d_inv_dct16_1d_c(
                    &mut col,
                    std::num::NonZeroUsize::new(1).unwrap(),
                    col_min,
                    col_max,
                );
                for y in 0..16 {
                    scalar_col_out[y * 16 + x] = col[y];
                }
            }

            // --- SIMD: row pass → shift → i16-packed column pass ---
            let simd_col_out = test_dct16_col_i16_pipeline(token, input);

            // --- Apply add-to-dst to BOTH outputs and compare ---
            let mut dst_scalar = [128u8; 16 * 16];
            let mut dst_simd = [128u8; 16 * 16];
            for y in 0..16 {
                for x in 0..16 {
                    let sc = scalar_col_out[y * 16 + x];
                    let scaled = (sc + 8) >> 4;
                    let p = (128i32 + scaled).clamp(0, 255);
                    dst_scalar[y * 16 + x] = p as u8;

                    let si = simd_col_out[y * 16 + x];
                    let scaled = (si + 8) >> 4;
                    let p = (128i32 + scaled).clamp(0, 255);
                    dst_simd[y * 16 + x] = p as u8;
                }
            }

            if dst_simd != dst_scalar {
                let mut mism = 0u32;
                for i in 0..256 {
                    if dst_simd[i] != dst_scalar[i] {
                        if mism < 8 {
                            eprintln!(
                                "seed={:#x} idx={} row={} col={} scalar={} simd={}",
                                seed,
                                i,
                                i / 16,
                                i % 16,
                                dst_scalar[i],
                                dst_simd[i],
                            );
                        }
                        mism += 1;
                    }
                }
                eprintln!("seed={seed:#x}: {mism} dst mismatches");
                total_mism += mism;
            }
        }
        assert_eq!(
            total_mism, 0,
            "dct16_col_pass_i16 full pipeline diverged from scalar reference"
        );
    }

    /// Direct comparison: `dct16_col_pass_i16` vs `dct16x16_cols_simd` on the
    /// same intermediate `tmp` buffer. This catches rounding differences between
    /// the pmaddwd and mullo column passes.
    #[cfg(target_arch = "x86_64")]
    #[arcane]
    fn test_dct16_col_direct_compare(
        _token: Desktop64,
        tmp: &[i32; 256],
    ) -> (/*pmaddwd*/ [i32; 256], /*mullo*/ [i32; 256]) {
        let col_min = i16::MIN as i32;
        let col_max = i16::MAX as i32;

        // pmaddwd version
        let pmaddwd_out = dct16_col_pass_i16(_token, tmp);

        // mullo version (in-place)
        let mut mullo_tmp = *tmp;
        dct16x16_cols_simd(_token, &mut mullo_tmp, col_min, col_max);

        (pmaddwd_out, mullo_tmp)
    }

    #[test]
    fn test_dct16_col_pass_i16_vs_mullo_direct() {
        let Some(token) = crate::src::cpu::summon_avx2() else {
            eprintln!("Skipping: AVX2 not available");
            return;
        };

        let seeds: &[u64] = &[
            0xdeadbeef,
            0xc0ffee,
            0xfeedface,
            0xbaadf00d,
            0x12345678,
            0xa5a5a5a5,
            0x5a5a5a5a,
            0xffff_ffff_ffff_ffff,
            0x0,
            0x7fff_7fff_7fff_7fff,
            0x8000_8000_8000_8000,
            // Extra random seeds
            0x0102030405060708,
            0xFEDCBA9876543210,
            0x1111111111111111,
            0x9999999999999999,
            0xAAAABBBBCCCCDDDD,
        ];
        let row_min = i16::MIN as i32;
        let row_max = i16::MAX as i32;
        let col_min = i16::MIN as i32;
        let col_max = i16::MAX as i32;
        let mut total_mism = 0u32;

        for &seed in seeds {
            let input: [i16; 256] = seeded_i16_block(seed);

            // Run SIMD row pass + shift to get the intermediate tmp buffer
            let row_out = run_scalar_dct16_per_row(&input, row_min, row_max);
            let mut tmp = [0i32; 256];
            for i in 0..256 {
                tmp[i] = ((row_out[i] + 2) >> 2).clamp(col_min, col_max);
            }

            // Compare the two column pass implementations directly
            let (pmaddwd_out, mullo_out) = test_dct16_col_direct_compare(token, &tmp);

            if pmaddwd_out != mullo_out {
                let mut mism = 0u32;
                for i in 0..256 {
                    if pmaddwd_out[i] != mullo_out[i] {
                        if mism < 8 {
                            eprintln!(
                                "seed={:#x} idx={} row={} col={} mullo={} pmaddwd={}",
                                seed,
                                i,
                                i / 16,
                                i % 16,
                                mullo_out[i],
                                pmaddwd_out[i],
                            );
                        }
                        mism += 1;
                    }
                }
                eprintln!("seed={seed:#x}: {mism} col-pass mismatches");
                total_mism += mism;
            }
        }
        assert_eq!(
            total_mism, 0,
            "dct16_col_pass_i16 diverged from dct16x16_cols_simd"
        );
    }

    // ----------------------------------------------------------------------
    // i16-packed pmaddwd DCT-16 row pass — bit-exact vs scalar
    // ----------------------------------------------------------------------

    /// Bit-exact check: `dct16_row_pass_i16_simd` matches
    /// `run_scalar_dct16_per_row` across a range of seeded inputs.
    #[test]
    fn test_dct16_row_pass_i16_simd_matches_scalar() {
        let Some(token) = crate::src::cpu::summon_avx2() else {
            eprintln!("Skipping: AVX2 not available");
            return;
        };
        let row_min = i16::MIN as i32;
        let row_max = i16::MAX as i32;
        let seeds: &[u64] = &[
            0xdeadbeef, 0xc0ffee, 0xfeedface, 0xbaadf00d, 0x12345678, 0xaabbccdd, 0x11223344,
            0x55667788,
        ];
        let mut total_mismatches = 0u32;
        for &seed in seeds {
            let input: [i16; 256] = seeded_i16_block(seed);
            let scalar_out = run_scalar_dct16_per_row(&input, row_min, row_max);
            let simd_out = dct16_row_pass_i16_simd(token, input);
            if simd_out != scalar_out {
                let mut mism = 0u32;
                for i in 0..256 {
                    if simd_out[i] != scalar_out[i] {
                        if mism < 16 {
                            eprintln!(
                                "seed={:#x} idx={} row={} col={} scalar={} simd={} diff={}",
                                seed,
                                i,
                                i / 16,
                                i % 16,
                                scalar_out[i],
                                simd_out[i],
                                simd_out[i].wrapping_sub(scalar_out[i])
                            );
                        }
                        mism += 1;
                    }
                }
                eprintln!("seed={seed:#x}: {mism} mismatches");
                total_mismatches += mism;
            }
        }
        assert_eq!(
            total_mismatches, 0,
            "dct16_row_pass_i16_simd diverged from scalar reference"
        );
    }

    // ----------------------------------------------------------------------
    // i16-packed pmaddwd DCT-32 row pass -- bit-exact vs scalar
    // ----------------------------------------------------------------------

    /// Bit-exact check: `dct32_row_pass_i16_simd` matches
    /// `run_scalar_dct32_per_row` across a range of seeded inputs.
    #[test]
    fn test_dct32_row_pass_i16_simd_matches_scalar() {
        let Some(token) = crate::src::cpu::summon_avx2() else {
            eprintln!("Skipping: AVX2 not available");
            return;
        };
        let row_min = i16::MIN as i32;
        let row_max = i16::MAX as i32;
        let seeds: &[u64] = &[
            0xdeadbeef,
            0xc0ffee,
            0xfeedface,
            0xbaadf00d,
            0x12345678,
            0xa5a5a5a5,
            0x5a5a5a5a,
            0xffff_ffff_ffff_ffff,
        ];
        let mut total_mismatches = 0u32;
        for &seed in seeds {
            let input: [i16; 1024] = seeded_i16_block(seed);
            let scalar_out = run_scalar_dct32_per_row(&input, row_min, row_max);
            let simd_out = dct32_row_pass_i16_simd(token, input);
            if simd_out != scalar_out {
                let mut mism = 0u32;
                for i in 0..1024 {
                    if simd_out[i] != scalar_out[i] {
                        if mism < 8 {
                            eprintln!(
                                "seed={:#x} idx={} row={} col={} scalar={} simd={} diff={}",
                                seed,
                                i,
                                i / 32,
                                i % 32,
                                scalar_out[i],
                                simd_out[i],
                                simd_out[i].wrapping_sub(scalar_out[i])
                            );
                        }
                        mism += 1;
                    }
                }
                eprintln!("seed={seed:#x}: {mism} mismatches");
                total_mismatches += mism;
            }
        }
        assert_eq!(
            total_mismatches, 0,
            "dct32_row_pass_i16_simd diverged from scalar reference"
        );
    }

    // ----------------------------------------------------------------------
    // AVX-512 16-row DCT row passes — bit-exact vs scalar per-row reference
    // ----------------------------------------------------------------------
    //
    // These exercise `simd_row_dct{8,16,32}_8bpc_16rows` (Server64, 16 lanes)
    // against `run_scalar_dct{8,16,32}_per_row` (full N rows). The 16-row
    // helpers process exactly one batch of 16 rows starting at y_base; for an
    // NxN block where N==16 a single call covers the whole block, and the
    // row-major `tmp` output must match the scalar oracle byte-for-byte with
    // apply_rect2=false, rnd=0, shift=0.

    /// dct8 16-row pass: build a 16-row x 8-col column-major input (coeff_h=16),
    /// run the AVX-512 16-row dct8 pass, compare each row against scalar dct8.
    #[test]
    fn test_simd_row_dct8_16rows_matches_scalar() {
        let Some(token) = crate::src::cpu::summon_avx512() else {
            eprintln!("Skipping: AVX-512 not available");
            return;
        };
        let row_min = i16::MIN as i32;
        let row_max = i16::MAX as i32;
        let seeds: &[u64] = &[
            0xdeadbeef,
            0xc0ffee,
            0xfeedface,
            0xbaadf00d,
            0x12345678,
            0xa5a5a5a5,
            0x5a5a5a5a,
            0xffff_ffff_ffff_ffff,
        ];
        let mut total = 0u32;
        for &seed in seeds {
            // 16 rows x 8 cols, column-major: coeff[y + x*16], 8 cols => 128 elems.
            let input: [i16; 128] = seeded_i16_block(seed);
            // Scalar oracle: per-row dct8 across 16 rows.
            let mut scalar = [0i32; 128]; // row-major 16 rows x 8 cols
            for y in 0..16 {
                let mut row = [0i32; 8];
                for x in 0..8 {
                    row[x] = input[y + x * 16] as i32;
                }
                crate::src::itx_1d::rav1d_inv_dct8_1d_c(
                    &mut row,
                    std::num::NonZeroUsize::new(1).unwrap(),
                    row_min,
                    row_max,
                );
                for x in 0..8 {
                    scalar[y * 8 + x] = row[x];
                }
            }
            let mut simd = [0i32; 128];
            simd_row_dct8_8bpc_16rows_entry(
                token, &input, 16, 0, false, 0, 0, &mut simd, row_min, row_max, row_min, row_max,
            );
            if simd != scalar {
                for i in 0..128 {
                    if simd[i] != scalar[i] && total < 8 {
                        eprintln!(
                            "dct8-16r seed={seed:#x} idx={i} row={} col={} scalar={} simd={}",
                            i / 8,
                            i % 8,
                            scalar[i],
                            simd[i]
                        );
                    }
                    if simd[i] != scalar[i] {
                        total += 1;
                    }
                }
            }
        }
        assert_eq!(total, 0, "simd_row_dct8_8bpc_16rows diverged from scalar");
    }

    /// dct16 16-row pass: 16x16 column-major input, full block in one call.
    #[test]
    fn test_simd_row_dct16_16rows_matches_scalar() {
        let Some(token) = crate::src::cpu::summon_avx512() else {
            eprintln!("Skipping: AVX-512 not available");
            return;
        };
        let row_min = i16::MIN as i32;
        let row_max = i16::MAX as i32;
        let seeds: &[u64] = &[
            0xdeadbeef,
            0xc0ffee,
            0xfeedface,
            0xbaadf00d,
            0x12345678,
            0xa5a5a5a5,
            0x5a5a5a5a,
            0xffff_ffff_ffff_ffff,
        ];
        let mut total = 0u32;
        for &seed in seeds {
            let input: [i16; 256] = seeded_i16_block(seed);
            let scalar = run_scalar_dct16_per_row(&input, row_min, row_max);
            let mut simd = [0i32; 256];
            simd_row_dct16_8bpc_16rows_entry(
                token, &input, 16, 0, false, 0, 0, &mut simd, row_min, row_max, row_min, row_max,
            );
            if simd != scalar {
                for i in 0..256 {
                    if simd[i] != scalar[i] && total < 8 {
                        eprintln!(
                            "dct16-16r seed={seed:#x} idx={i} row={} col={} scalar={} simd={}",
                            i / 16,
                            i % 16,
                            scalar[i],
                            simd[i]
                        );
                    }
                    if simd[i] != scalar[i] {
                        total += 1;
                    }
                }
            }
        }
        assert_eq!(total, 0, "simd_row_dct16_8bpc_16rows diverged from scalar");
    }

    /// dct32 16-row pass: 16 rows x 32 cols column-major input (coeff_h=16),
    /// compared against scalar dct32 per-row over 16 rows.
    #[test]
    fn test_simd_row_dct32_16rows_matches_scalar() {
        let Some(token) = crate::src::cpu::summon_avx512() else {
            eprintln!("Skipping: AVX-512 not available");
            return;
        };
        let row_min = i16::MIN as i32;
        let row_max = i16::MAX as i32;
        let seeds: &[u64] = &[
            0xdeadbeef,
            0xc0ffee,
            0xfeedface,
            0xbaadf00d,
            0x12345678,
            0xa5a5a5a5,
            0x5a5a5a5a,
            0xffff_ffff_ffff_ffff,
        ];
        let mut total = 0u32;
        for &seed in seeds {
            // 16 rows x 32 cols, column-major coeff[y + x*16] => 512 elems.
            let input: [i16; 512] = seeded_i16_block(seed);
            let mut scalar = [0i32; 512]; // row-major 16 rows x 32 cols
            for y in 0..16 {
                let mut row = [0i32; 32];
                for x in 0..32 {
                    row[x] = input[y + x * 16] as i32;
                }
                crate::src::itx_1d::rav1d_inv_dct32_1d_c(
                    &mut row,
                    std::num::NonZeroUsize::new(1).unwrap(),
                    row_min,
                    row_max,
                );
                for x in 0..32 {
                    scalar[y * 32 + x] = row[x];
                }
            }
            let mut simd = [0i32; 512];
            simd_row_dct32_8bpc_16rows_entry(
                token, &input, 16, 0, false, 0, 0, &mut simd, row_min, row_max, row_min, row_max,
            );
            if simd != scalar {
                for i in 0..512 {
                    if simd[i] != scalar[i] && total < 8 {
                        eprintln!(
                            "dct32-16r seed={seed:#x} idx={i} row={} col={} scalar={} simd={}",
                            i / 32,
                            i % 32,
                            scalar[i],
                            simd[i]
                        );
                    }
                    if simd[i] != scalar[i] {
                        total += 1;
                    }
                }
            }
        }
        assert_eq!(total, 0, "simd_row_dct32_8bpc_16rows diverged from scalar");
    }

    /// `dct8_cols_avx512` column pass: process a `total_w` x 8 row-major i32
    /// buffer and compare each transformed column against the scalar
    /// `rav1d_inv_dct8_1d_c` oracle. Exercised on 16- and 32-wide buffers
    /// (the Nx8 transform widths it is wired into) across 8 seeds. Bit-exact.
    #[test]
    fn test_dct8_cols_avx512_matches_scalar() {
        let Some(token) = crate::src::cpu::summon_avx512() else {
            eprintln!("Skipping: AVX-512 not available");
            return;
        };
        let col_min = i16::MIN as i32;
        let col_max = i16::MAX as i32;
        let seeds: &[u64] = &[
            0xdeadbeef,
            0xc0ffee,
            0xfeedface,
            0xbaadf00d,
            0x12345678,
            0xa5a5a5a5,
            0x5a5a5a5a,
            0xffff_ffff_ffff_ffff,
        ];
        let mut total = 0u32;
        for &total_w in &[16usize, 32usize] {
            for &seed in seeds {
                // total_w cols x 8 rows, row-major i32 input, derived from a
                // seeded i16 block clamped to the col-clip range.
                let n = total_w * 8;
                let raw: Vec<i16> = (0..n)
                    .map(|i| {
                        let s = seed
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(i as u64)
                            .wrapping_mul(1442695040888963407);
                        (s >> 33) as i16
                    })
                    .collect();
                let mut input = vec![0i32; n];
                for i in 0..n {
                    input[i] = raw[i] as i32;
                }

                // Scalar oracle: dct8 down each of total_w columns.
                let mut scalar = input.clone();
                for cx in 0..total_w {
                    let mut col = [0i32; 8];
                    for r in 0..8 {
                        col[r] = input[r * total_w + cx];
                    }
                    crate::src::itx_1d::rav1d_inv_dct8_1d_c(
                        &mut col,
                        std::num::NonZeroUsize::new(1).unwrap(),
                        col_min,
                        col_max,
                    );
                    for r in 0..8 {
                        scalar[r * total_w + cx] = col[r];
                    }
                }

                let mut simd = input.clone();
                dct8_cols_avx512(token, &mut simd, total_w, 8, col_min, col_max);

                if simd != scalar {
                    for i in 0..n {
                        if simd[i] != scalar[i] && total < 8 {
                            eprintln!(
                                "dct8-cols512 w={total_w} seed={seed:#x} idx={i} row={} col={} scalar={} simd={}",
                                i / total_w,
                                i % total_w,
                                scalar[i],
                                simd[i]
                            );
                        }
                        if simd[i] != scalar[i] {
                            total += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(total, 0, "dct8_cols_avx512 diverged from scalar");
    }

    /// `dct4_cols_avx512` column pass: process a 16 x 4 row-major i32 buffer
    /// (the 16x4 transform width it is wired into) and compare each transformed
    /// column against the scalar `rav1d_inv_dct4_1d_c` oracle. 8 seeds, bit-exact.
    #[test]
    fn test_dct4_cols_avx512_matches_scalar() {
        let Some(token) = crate::src::cpu::summon_avx512() else {
            eprintln!("Skipping: AVX-512 not available");
            return;
        };
        let col_min = i16::MIN as i32;
        let col_max = i16::MAX as i32;
        let seeds: &[u64] = &[
            0xdeadbeef,
            0xc0ffee,
            0xfeedface,
            0xbaadf00d,
            0x12345678,
            0xa5a5a5a5,
            0x5a5a5a5a,
            0xffff_ffff_ffff_ffff,
        ];
        let total_w = 16usize;
        let n_rows = 4usize;
        let mut total = 0u32;
        for &seed in seeds {
            let n = total_w * n_rows;
            let input: Vec<i32> = (0..n)
                .map(|i| {
                    let s = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(i as u64)
                        .wrapping_mul(1442695040888963407);
                    (s >> 49) as i16 as i32
                })
                .collect();

            let mut scalar = input.clone();
            for cx in 0..total_w {
                let mut col = [0i32; 4];
                for r in 0..n_rows {
                    col[r] = input[r * total_w + cx];
                }
                crate::src::itx_1d::rav1d_inv_dct4_1d_c(
                    &mut col,
                    std::num::NonZeroUsize::new(1).unwrap(),
                    col_min,
                    col_max,
                );
                for r in 0..n_rows {
                    scalar[r * total_w + cx] = col[r];
                }
            }

            let mut simd = input.clone();
            dct4_cols_avx512(token, &mut simd, total_w, n_rows, col_min, col_max);

            if simd != scalar {
                for i in 0..n {
                    if simd[i] != scalar[i] && total < 8 {
                        eprintln!(
                            "dct4-cols512 seed={seed:#x} idx={i} row={} col={} scalar={} simd={}",
                            i / total_w,
                            i % total_w,
                            scalar[i],
                            simd[i]
                        );
                    }
                    if simd[i] != scalar[i] {
                        total += 1;
                    }
                }
            }
        }
        assert_eq!(total, 0, "dct4_cols_avx512 diverged from scalar");
    }

    /// AVX-512 identity column passes vs scalar oracle, over the (total_w,
    /// n_rows) widths they are wired into, 8 seeds each. Bit-exact. Covers
    /// `identity_shift_cols_avx512::<1>` (identity8 = *2),
    /// `identity_shift_cols_avx512::<2>` (identity32 = *4), and
    /// `identity16_cols_avx512` (2*in + ((in*1697+1024)>>11)).
    #[test]
    fn test_identity_cols_avx512_matches_scalar() {
        let Some(token) = crate::src::cpu::summon_avx512() else {
            eprintln!("Skipping: AVX-512 not available");
            return;
        };
        let seeds: &[u64] = &[
            0xdeadbeef,
            0xc0ffee,
            0xfeedface,
            0xbaadf00d,
            0x12345678,
            0xa5a5a5a5,
            0x5a5a5a5a,
            0xffff_ffff_ffff_ffff,
        ];
        // (total_w, n_rows, kind): kind 1=*2, 2=*4, 16=identity16.
        let cases: &[(usize, usize, u8)] = &[
            (16, 8, 1),
            (32, 8, 1),
            (16, 32, 2),
            (32, 16, 16),
            (16, 16, 16),
        ];
        let mut total = 0u32;
        for &(total_w, n_rows, kind) in cases {
            for &seed in seeds {
                let n = total_w * n_rows;
                // Keep inputs small enough that in*1697 stays within i32.
                let input: Vec<i32> = (0..n)
                    .map(|i| {
                        let s = seed
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(i as u64)
                            .wrapping_mul(1442695040888963407);
                        (((s >> 40) as i32) & 0xffff) - 0x8000
                    })
                    .collect();

                // Scalar oracle: apply the matching identity down each column.
                let mut scalar = input.clone();
                for cx in 0..total_w {
                    for r in 0..n_rows {
                        let v = input[r * total_w + cx];
                        scalar[r * total_w + cx] = match kind {
                            1 => v * 2,
                            2 => v * 4,
                            _ => 2 * v + ((v * 1697 + 1024) >> 11),
                        };
                    }
                }

                let mut simd = input.clone();
                match kind {
                    1 => identity_shift_cols_avx512::<1>(token, &mut simd, total_w, n_rows),
                    2 => identity_shift_cols_avx512::<2>(token, &mut simd, total_w, n_rows),
                    _ => identity16_cols_avx512(token, &mut simd, total_w, n_rows),
                }

                if simd != scalar {
                    for i in 0..n {
                        if simd[i] != scalar[i] && total < 8 {
                            eprintln!(
                                "idtx512 kind={kind} w={total_w} rows={n_rows} seed={seed:#x} idx={i} scalar={} simd={}",
                                scalar[i], simd[i]
                            );
                        }
                        if simd[i] != scalar[i] {
                            total += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(total, 0, "identity AVX-512 column pass diverged from scalar");
    }
}

