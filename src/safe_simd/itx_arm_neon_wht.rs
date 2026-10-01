//! Safe ARM NEON WHT 4x4 inverse transform
//!
//! Port of the `inv_txfm_add_wht_wht_4x4_8bpc_neon` assembly function
//! from `src/arm/64/itx.S` to safe Rust NEON intrinsics.
#![allow(clippy::too_many_arguments)]
#![cfg_attr(not(any(feature = "asm", feature = "c-ffi", feature = "partial_asm")), forbid(unsafe_code))]
#![cfg_attr(any(feature = "asm", feature = "c-ffi", feature = "partial_asm"), deny(unsafe_code))]

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
use archmage::{Arm64, arcane, rite};

#[cfg(target_arch = "aarch64")]
use safe_unaligned_simd::aarch64 as safe_simd;

/// NEON implementation of the WHT 4x4 inverse transform for 8bpc.
///
/// This mirrors the assembly in `itx.S` lines 603-629 + the `iwht4` macro
/// and `itx_4x4_end` label.
///
/// Algorithm:
/// 1. Load 4x4 i16 coefficients (column-major) into 4 half-registers
/// 2. Arithmetic shift right by 2
/// 3. iwht4 butterfly (row transform)
/// 4. Transpose 4x4
/// 5. iwht4 butterfly (column transform)
/// 6. Add to destination u8 pixels with unsigned saturation
/// 7. Clear coefficient buffer
#[cfg(target_arch = "aarch64")]
#[arcane]
pub(crate) fn inv_txfm_add_wht_wht_4x4_8bpc_neon_inner(
    _token: Arm64,
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    coeff: &mut [i16],
    _eob: i32,
    _bitdepth_max: i32,
) {
    // Step 1: Load 4x4 i16 coefficients as 4 half-registers (int16x4_t)
    // Column-major: coeff[0..4] is column 0, coeff[4..8] is column 1, etc.
    let v16 = safe_simd::vld1_s16(coeff[0..4].try_into().unwrap());
    let v17 = safe_simd::vld1_s16(coeff[4..8].try_into().unwrap());
    let v18 = safe_simd::vld1_s16(coeff[8..12].try_into().unwrap());
    let v19 = safe_simd::vld1_s16(coeff[12..16].try_into().unwrap());

    // Step 2: Arithmetic shift right by 2
    let v16 = vshr_n_s16::<2>(v16);
    let v17 = vshr_n_s16::<2>(v17);
    let v18 = vshr_n_s16::<2>(v18);
    let v19 = vshr_n_s16::<2>(v19);

    // Step 3: Row transform (iwht4 butterfly)
    let (v16, v17, v18, v19) = iwht4(v16, v17, v18, v19);

    // Step 4: Transpose 4x4 using trn1/trn2
    let (v16, v17, v18, v19) = transpose_4x4h(v16, v17, v18, v19);

    // Step 5: Column transform (iwht4 butterfly again)
    let (v16, v17, v18, v19) = iwht4(v16, v17, v18, v19);

    // Step 6: Combine into int16x8_t pairs and add to destination
    // v16_wide = [row0 | row1], v18_wide = [row2 | row3]
    let v16_wide = vcombine_s16(v16, v17);
    let v18_wide = vcombine_s16(v18, v19);

    // Load 4 rows of 4 destination pixels
    // Each row is 4 bytes, loaded as the low 4 bytes of a uint8x8_t
    let row0_off = dst_base;
    let row1_off = dst_base.wrapping_add_signed(dst_stride);
    let row2_off = dst_base.wrapping_add_signed(dst_stride * 2);
    let row3_off = dst_base.wrapping_add_signed(dst_stride * 3);

    // Load rows 0+1 packed into one uint8x8_t (4 bytes each)
    let mut dst_bytes_01 = [0u8; 8];
    dst_bytes_01[0..4].copy_from_slice(&dst[row0_off..row0_off + 4]);
    dst_bytes_01[4..8].copy_from_slice(&dst[row1_off..row1_off + 4]);
    let v0 = safe_simd::vld1_u8(&dst_bytes_01);

    // Load rows 2+3 packed into one uint8x8_t
    let mut dst_bytes_23 = [0u8; 8];
    dst_bytes_23[0..4].copy_from_slice(&dst[row2_off..row2_off + 4]);
    dst_bytes_23[4..8].copy_from_slice(&dst[row3_off..row3_off + 4]);
    let v1 = safe_simd::vld1_u8(&dst_bytes_23);

    // uaddw: zero-extend u8 to u16 and add to i16 (reinterpreted as u16)
    // This matches the assembly: uaddw v16.8h, v16.8h, v0.8b
    // Two's complement means signed+unsigned add gives correct bits.
    let sum_01 = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(v16_wide), v0));
    let sum_23 = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(v18_wide), v1));

    // sqxtun: signed saturating narrow to unsigned u8
    // Clamps each i16 to [0, 255] and narrows to u8
    let result_01 = vqmovun_s16(sum_01);
    let result_23 = vqmovun_s16(sum_23);

    // Store 4 bytes per row
    let mut out_01 = [0u8; 8];
    safe_simd::vst1_u8(&mut out_01, result_01);
    dst[row0_off..row0_off + 4].copy_from_slice(&out_01[0..4]);
    dst[row1_off..row1_off + 4].copy_from_slice(&out_01[4..8]);

    let mut out_23 = [0u8; 8];
    safe_simd::vst1_u8(&mut out_23, result_23);
    dst[row2_off..row2_off + 4].copy_from_slice(&out_23[0..4]);
    dst[row3_off..row3_off + 4].copy_from_slice(&out_23[4..8]);

    // Step 7: Clear coefficients (matches assembly: st1 {v31.8h}, [x2] with v31=0)
    coeff[0..16].fill(0);
}

/// WHT inverse butterfly: the iwht4 macro from itx.S lines 426-435.
///
/// Input:  (v16=in0, v17=in1, v18=in2, v19=in3)
/// Output: (out0, out1, out2, out3)
///
/// ```text
/// t0 = in0 + in1
/// t2 = in2 - in3
/// t4 = (t0 - t2) >> 1
/// t1 = t4 - in1      (stored in out2 position)
/// t3 = t4 - in3      (stored in out1 position)
/// out0 = t0 - t3
/// out1 = t3
/// out2 = t1
/// out3 = t2 + t1
/// ```
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn iwht4(
    v16: int16x4_t,
    v17: int16x4_t,
    v18: int16x4_t,
    v19: int16x4_t,
) -> (int16x4_t, int16x4_t, int16x4_t, int16x4_t) {
    // Exactly matching assembly register flow:
    // add  v16, v16, v17        => v16 = t0 = in0 + in1
    let t0 = vadd_s16(v16, v17);
    // sub  v21, v18, v19        => v21 = t2 = in2 - in3
    let t2 = vsub_s16(v18, v19);
    // sub  v20, v16, v21        => v20 = t0 - t2
    let diff = vsub_s16(t0, t2);
    // sshr v20, v20, #1         => v20 = t4 = (t0 - t2) >> 1
    let t4 = vshr_n_s16::<1>(diff);
    // sub  v18, v20, v17        => v18 = t1 = t4 - in1
    let t1 = vsub_s16(t4, v17);
    // sub  v17, v20, v19        => v17 = t3 = t4 - in3
    let t3 = vsub_s16(t4, v19);
    // add  v19, v21, v18        => v19 = out3 = t2 + t1
    let out3 = vadd_s16(t2, t1);
    // sub  v16, v16, v17        => v16 = out0 = t0 - t3
    let out0 = vsub_s16(t0, t3);

    (out0, t3, t1, out3)
}

/// 4x4 transpose for int16x4_t vectors.
///
/// Matches the `transpose_4x4h` macro from util.S lines 230-239.
/// Uses trn1/trn2 at 16-bit level, then trn1/trn2 at 32-bit level.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn transpose_4x4h(
    r0: int16x4_t,
    r1: int16x4_t,
    r2: int16x4_t,
    r3: int16x4_t,
) -> (int16x4_t, int16x4_t, int16x4_t, int16x4_t) {
    // First level: interleave at 16-bit granularity
    // trn1 t4.4h, r0.4h, r1.4h
    let t4 = vtrn1_s16(r0, r1);
    // trn2 t5.4h, r0.4h, r1.4h
    let t5 = vtrn2_s16(r0, r1);
    // trn1 t6.4h, r2.4h, r3.4h
    let t6 = vtrn1_s16(r2, r3);
    // trn2 t7.4h, r2.4h, r3.4h
    let t7 = vtrn2_s16(r2, r3);

    // Second level: interleave at 32-bit granularity
    // Reinterpret as int32x2_t for 32-bit transpose
    let t4_32 = vreinterpret_s32_s16(t4);
    let t5_32 = vreinterpret_s32_s16(t5);
    let t6_32 = vreinterpret_s32_s16(t6);
    let t7_32 = vreinterpret_s32_s16(t7);

    // trn1 r0.2s, t4.2s, t6.2s
    let o0_32 = vtrn1_s32(t4_32, t6_32);
    // trn2 r2.2s, t4.2s, t6.2s
    let o2_32 = vtrn2_s32(t4_32, t6_32);
    // trn1 r1.2s, t5.2s, t7.2s
    let o1_32 = vtrn1_s32(t5_32, t7_32);
    // trn2 r3.2s, t5.2s, t7.2s
    let o3_32 = vtrn2_s32(t5_32, t7_32);

    // Reinterpret back to int16x4_t
    let o0 = vreinterpret_s16_s32(o0_32);
    let o1 = vreinterpret_s16_s32(o1_32);
    let o2 = vreinterpret_s16_s32(o2_32);
    let o3 = vreinterpret_s16_s32(o3_32);

    (o0, o1, o2, o3)
}

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;
    use archmage::SimdToken;

    /// Test that the NEON WHT 4x4 produces identical output to the scalar path.
    #[test]
    fn test_wht_4x4_neon_matches_scalar() {
        // Test with a variety of coefficient patterns
        let test_coeffs: &[[i16; 16]] = &[
            // Pattern 1: Mixed positive/negative
            [
                100, -50, 30, -20, 40, -30, 20, -10, 15, -8, 4, -2, 7, -3, 1, 0,
            ],
            // Pattern 2: DC only
            [64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            // Pattern 3: All same
            [
                32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
            ],
            // Pattern 4: Alternating
            [
                100, -100, 100, -100, -100, 100, -100, 100, 100, -100, 100, -100, -100, 100, -100,
                100,
            ],
            // Pattern 5: Large values
            [
                500, -400, 300, -200, 400, -300, 200, -100, 300, -200, 100, 0, 200, -100, 50, -25,
            ],
            // Pattern 6: All zeros (should be identity add)
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ];

        let token = match Arm64::summon() {
            Some(t) => t,
            None => {
                eprintln!("Skipping NEON test: Arm64 not available");
                return;
            }
        };

        let stride: isize = 16;

        for (i, coeffs) in test_coeffs.iter().enumerate() {
            let mut coeff_scalar = *coeffs;
            let mut coeff_neon = *coeffs;

            // Initialize destination with a non-trivial pattern
            let mut dst_scalar = [0u8; 64];
            for (j, byte) in dst_scalar.iter_mut().enumerate() {
                *byte = ((j * 7 + 128) % 256) as u8;
            }
            let mut dst_neon = dst_scalar;

            // Run scalar
            super::super::itx_arm::inv_txfm_add_wht_wht_4x4_8bpc_inner(
                &mut dst_scalar,
                0,
                stride,
                &mut coeff_scalar,
                0,
                255,
            );

            // Run NEON
            inv_txfm_add_wht_wht_4x4_8bpc_neon_inner(
                token,
                &mut dst_neon,
                0,
                stride,
                &mut coeff_neon,
                0,
                255,
            );

            // Compare pixel outputs (only the 4x4 region that was modified)
            for row in 0..4 {
                let off = row * stride as usize;
                assert_eq!(
                    &dst_neon[off..off + 4],
                    &dst_scalar[off..off + 4],
                    "Row {row} mismatch in test pattern {i}: \
                     neon={:?} scalar={:?}",
                    &dst_neon[off..off + 4],
                    &dst_scalar[off..off + 4],
                );
            }

            // Verify coefficients are cleared
            assert!(
                coeff_neon.iter().all(|&c| c == 0),
                "NEON coefficients not zeroed in test pattern {i}"
            );
            assert!(
                coeff_scalar.iter().all(|&c| c == 0),
                "Scalar coefficients not zeroed in test pattern {i}"
            );
        }
    }

    /// Test that the NEON WHT handles saturation correctly at boundaries.
    #[test]
    fn test_wht_4x4_neon_saturation() {
        let token = match Arm64::summon() {
            Some(t) => t,
            None => {
                eprintln!("Skipping NEON test: Arm64 not available");
                return;
            }
        };

        let stride: isize = 4;

        // Test overflow saturation: dst=250, large positive coefficients
        {
            let mut coeff_scalar = [400i16; 16];
            let mut coeff_neon = coeff_scalar;
            let mut dst_scalar = [250u8; 16];
            let mut dst_neon = dst_scalar;

            super::super::itx_arm::inv_txfm_add_wht_wht_4x4_8bpc_inner(
                &mut dst_scalar,
                0,
                stride,
                &mut coeff_scalar,
                0,
                255,
            );
            inv_txfm_add_wht_wht_4x4_8bpc_neon_inner(
                token,
                &mut dst_neon,
                0,
                stride,
                &mut coeff_neon,
                0,
                255,
            );
            assert_eq!(dst_neon, dst_scalar, "Overflow saturation mismatch");
        }

        // Test underflow saturation: dst=5, large negative coefficients
        {
            let mut coeff_scalar = [-400i16; 16];
            let mut coeff_neon = coeff_scalar;
            let mut dst_scalar = [5u8; 16];
            let mut dst_neon = dst_scalar;

            super::super::itx_arm::inv_txfm_add_wht_wht_4x4_8bpc_inner(
                &mut dst_scalar,
                0,
                stride,
                &mut coeff_scalar,
                0,
                255,
            );
            inv_txfm_add_wht_wht_4x4_8bpc_neon_inner(
                token,
                &mut dst_neon,
                0,
                stride,
                &mut coeff_neon,
                0,
                255,
            );
            assert_eq!(dst_neon, dst_scalar, "Underflow saturation mismatch");
        }
    }
}
