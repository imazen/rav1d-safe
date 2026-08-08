//! Safe ARM NEON 16x16 inverse transforms (DCT, ADST, flipADST, identity)
//!
//! Port of the 16x16 inverse transform functions from `src/arm/64/itx.S`
//! (lines 1033-1480) to safe Rust NEON intrinsics. Operates on 16 `int16x8_t`
//! vectors (8 elements each), processing 8 columns in parallel.
//!
//! Each transform half processes 16 `int16x8_t` vectors where:
//! - For the row transform: v[col] holds 8 row values at that column
//!   (8 row transforms in parallel across 16 columns)
//! - For the column transform: v[row] holds 8 column values at that row
//!   (8 column transforms in parallel across 16 rows)
//!
//! The row transform runs twice (rows 0-7, then rows 8-15), storing
//! transposed results to a 512-byte intermediate buffer. The column
//! transform then runs twice (cols 0-7, then cols 8-15), adding results
//! to the destination.

#![allow(clippy::too_many_arguments)]
#![cfg_attr(not(feature = "unchecked"), forbid(unsafe_code))]
#![cfg_attr(feature = "unchecked", deny(unsafe_code))]

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
use archmage::{Arm64, arcane, rite};

#[cfg(target_arch = "aarch64")]
use safe_unaligned_simd::aarch64 as safe_simd;

use super::itx_arm_neon_8x8::{
    idct_8_q, smull_smlal_q, smull_smlsl_q, sqrshrn_pair, transpose_8x8h,
};
use super::itx_arm_neon_common::IDCT_COEFFS;

/// Type alias for 16 NEON vectors (one full 16-point transform state).
#[cfg(target_arch = "aarch64")]
type V16 = [int16x8_t; 16];

// ============================================================================
// IADST16 coefficient table (from itx.S lines 111-116)
// ============================================================================

/// IADST16 coefficients packed as two `int16x8_t` vectors.
///
/// v0: [4091, 201, 3973, 995, 3703, 1751, 3290, 2440]
/// v1: [2751, 3035, 2106, 3513, 1380, 3857, 601, 4052]
#[cfg(target_arch = "aarch64")]
pub(crate) const IADST16_COEFFS_V0: [i16; 8] = [4091, 201, 3973, 995, 3703, 1751, 3290, 2440];
#[cfg(target_arch = "aarch64")]
pub(crate) const IADST16_COEFFS_V1: [i16; 8] = [2751, 3035, 2106, 3513, 1380, 3857, 601, 4052];

// ============================================================================
// 16-point inverse DCT (idct_16 macro from itx.S lines 1033-1114)
// ============================================================================

/// 16-point inverse DCT on 16 int16x8_t vectors.
///
/// Calls `idct_8_q` on even-indexed inputs, then processes odd-indexed inputs
/// through butterfly stages. Uses IDCT coefficients from v0 and v1:
///
/// v0: [2896, 23168, 1567, 3784, 799, 4017, 3406, 2276]  (idct4+idct8)
/// v1: [401, 4076, 3166, 2598, 1931, 3612, 3920, 1189]    (idct16)
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
pub(crate) fn idct_16_q(v: V16) -> V16 {
    // Apply idct_8 to even-indexed inputs: v[0],v[2],v[4],v[6],v[8],v[10],v[12],v[14]
    let (e0, e1, e2, e3, e4, e5, e6, e7) =
        idct_8_q(v[0], v[2], v[4], v[6], v[8], v[10], v[12], v[14]);

    // Load idct16 coefficients (the second 8 shorts)
    // v1: [401, 4076, 3166, 2598, 1931, 3612, 3920, 1189]
    let c1 = safe_simd::vld1q_s16(<&[i16; 8]>::try_from(&IDCT_COEFFS[8..16]).unwrap());

    // Stage 1: Rotation pairs on odd-indexed inputs
    // t8a  = (v[1] * 401 - v[15] * 4076 + 2048) >> 12
    // t15a = (v[1] * 4076 + v[15] * 401 + 2048) >> 12
    let (lo, hi) = smull_smlsl_q(v[1], v[15], c1, 0, 1);
    let t8a = sqrshrn_pair(lo, hi);
    let (lo, hi) = smull_smlal_q(v[1], v[15], c1, 1, 0);
    let t15a = sqrshrn_pair(lo, hi);

    // t9a  = (v[9] * 3166 - v[7] * 2598 + 2048) >> 12
    // t14a = (v[9] * 2598 + v[7] * 3166 + 2048) >> 12
    let (lo, hi) = smull_smlsl_q(v[9], v[7], c1, 2, 3);
    let t9a = sqrshrn_pair(lo, hi);
    let (lo, hi) = smull_smlal_q(v[9], v[7], c1, 3, 2);
    let t14a = sqrshrn_pair(lo, hi);

    // t10a = (v[5] * 1931 - v[11] * 3612 + 2048) >> 12
    // t13a = (v[5] * 3612 + v[11] * 1931 + 2048) >> 12
    let (lo, hi) = smull_smlsl_q(v[5], v[11], c1, 4, 5);
    let t10a = sqrshrn_pair(lo, hi);
    let (lo, hi) = smull_smlal_q(v[5], v[11], c1, 5, 4);
    let t13a = sqrshrn_pair(lo, hi);

    // t11a = (v[13] * 3920 - v[3] * 1189 + 2048) >> 12
    // t12a = (v[13] * 1189 + v[3] * 3920 + 2048) >> 12
    let (lo, hi) = smull_smlsl_q(v[13], v[3], c1, 6, 7);
    let t11a = sqrshrn_pair(lo, hi);
    let (lo, hi) = smull_smlal_q(v[13], v[3], c1, 7, 6);
    let t12a = sqrshrn_pair(lo, hi);

    // Stage 2: Butterfly
    let t9 = vqsubq_s16(t8a, t9a); // t9
    let t8 = vqaddq_s16(t8a, t9a); // t8
    let t14 = vqsubq_s16(t15a, t14a); // t14
    let t15 = vqaddq_s16(t15a, t14a); // t15
    let t10 = vqsubq_s16(t11a, t10a); // t10
    let t11 = vqaddq_s16(t11a, t10a); // t11
    let t12 = vqaddq_s16(t12a, t13a); // t12
    let t13 = vqsubq_s16(t12a, t13a); // t13

    // Stage 3: Rotations using idct4 coefficients: v0.h[2]=1567, v0.h[3]=3784
    let c0 = safe_simd::vld1q_s16(<&[i16; 8]>::try_from(&IDCT_COEFFS[0..8]).unwrap());

    // t9a = (t14 * 1567 - t9 * 3784 + 2048) >> 12
    // t14a = (t14 * 3784 + t9 * 1567 + 2048) >> 12
    let (lo, hi) = smull_smlsl_q(t14, t9, c0, 2, 3);
    let t9a = sqrshrn_pair(lo, hi);
    let (lo, hi) = smull_smlal_q(t14, t9, c0, 3, 2);
    let t14a = sqrshrn_pair(lo, hi);

    // t10a = -(t13 * 3784 + t10 * 1567 + 2048) >> 12  (negated)
    // The assembly does: smull_smlsl(t13, t10, 1567, 3784) → t13a
    //                     smull_smlal(t13, t10, 3784, 1567) → but then negates
    // Actually looking at the assembly more carefully:
    // smull_smlsl v4,v5, v29,v23, v0.h[2],v0.h[3]  -> t13 * 1567 - t10 * 3784 -> t13a
    // smull_smlal v6,v7, v29,v23, v0.h[3],v0.h[2]  -> t13 * 3784 + t10 * 1567 -> t10a
    // then neg v6.4s, v6.4s; neg v7.4s, v7.4s -> negate t10a
    let (lo, hi) = smull_smlsl_q(t13, t10, c0, 2, 3);
    let t13a = sqrshrn_pair(lo, hi);
    let (lo, hi) = smull_smlal_q(t13, t10, c0, 3, 2);
    // Negate the result
    let t10a = sqrshrn_pair(vnegq_s32(lo), vnegq_s32(hi));

    // Stage 4: Butterfly
    let t11a = vqsubq_s16(t8, t11); // t11a
    let t8a = vqaddq_s16(t8, t11); // t8a
    let t12a = vqsubq_s16(t15, t12); // t12a
    let t15a = vqaddq_s16(t15, t12); // t15a
    let t9b = vqaddq_s16(t9a, t10a); // t9
    let t10b = vqsubq_s16(t9a, t10a); // t10
    let t13b = vqsubq_s16(t14a, t13a); // t13
    let t14b = vqaddq_s16(t14a, t13a); // t14

    // Stage 5: Final rotations using v0.h[0]=2896
    // t11 = (t12a * 2896 - t11a * 2896 + 2048) >> 12
    // t12 = (t12a * 2896 + t11a * 2896 + 2048) >> 12
    let (lo, hi) = smull_smlsl_q(t12a, t11a, c0, 0, 0);
    let t11_final = sqrshrn_pair(lo, hi);
    let (lo, hi) = smull_smlal_q(t12a, t11a, c0, 0, 0);
    let t12_final = sqrshrn_pair(lo, hi);

    // t10a = (t13b * 2896 - t10b * 2896 + 2048) >> 12
    // t13a = (t13b * 2896 + t10b * 2896 + 2048) >> 12
    let (lo, hi) = smull_smlsl_q(t13b, t10b, c0, 0, 0);
    let t10a_final = sqrshrn_pair(lo, hi);
    let (lo, hi) = smull_smlal_q(t13b, t10b, c0, 0, 0);
    let t13a_final = sqrshrn_pair(lo, hi);

    // Final butterfly: combine even outputs (e0..e7) with odd outputs
    // out[0]  = e0 + t15a
    // out[15] = e0 - t15a
    // out[1]  = e1 + t14b
    // out[14] = e1 - t14b
    // out[2]  = e2 + t13a_final
    // out[13] = e2 - t13a_final
    // out[3]  = e3 + t12_final
    // out[12] = e3 - t12_final
    // out[4]  = e4 + t11_final
    // out[11] = e4 - t11_final
    // out[5]  = e5 + t10a_final
    // out[10] = e5 - t10a_final
    // out[6]  = e6 + t9b
    // out[9]  = e6 - t9b
    // out[7]  = e7 + t8a
    // out[8]  = e7 - t8a
    [
        vqaddq_s16(e0, t15a),       // out0
        vqaddq_s16(e1, t14b),       // out1
        vqaddq_s16(e2, t13a_final), // out2
        vqaddq_s16(e3, t12_final),  // out3
        vqaddq_s16(e4, t11_final),  // out4
        vqaddq_s16(e5, t10a_final), // out5
        vqaddq_s16(e6, t9b),        // out6
        vqaddq_s16(e7, t8a),        // out7
        vqsubq_s16(e7, t8a),        // out8
        vqsubq_s16(e6, t9b),        // out9
        vqsubq_s16(e5, t10a_final), // out10
        vqsubq_s16(e4, t11_final),  // out11
        vqsubq_s16(e3, t12_final),  // out12
        vqsubq_s16(e2, t13a_final), // out13
        vqsubq_s16(e1, t14b),       // out14
        vqsubq_s16(e0, t15a),       // out15
    ]
}

// ============================================================================
// 16-point inverse ADST (iadst_16 macro from itx.S lines 1130-1298)
// ============================================================================

/// 16-point inverse ADST on 16 int16x8_t vectors.
///
/// Direct port of the `iadst_16` assembly macro. This is the most complex
/// transform, with 4 stages of butterfly operations.
///
/// Input mapping from assembly: v[0]..v[15] map to v16..v31
/// Output mapping: o[0]..o[15] map to the output indices.
///
/// For normal ADST, outputs are in natural order.
/// For flipADST, the caller reverses the output order.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
pub(crate) fn iadst_16_q(v: V16) -> V16 {
    // Load iadst16 coefficients
    let c0 = safe_simd::vld1q_s16(<&[i16; 8]>::try_from(&IADST16_COEFFS_V0[..]).unwrap());
    let c1 = safe_simd::vld1q_s16(<&[i16; 8]>::try_from(&IADST16_COEFFS_V1[..]).unwrap());

    // Stage 1: 8 rotation pairs
    // t0 = (v[15] * 4091 + v[0] * 201) >> 12
    let (lo, hi) = smull_smlal_q(v[15], v[0], c0, 0, 1);
    let t0 = sqrshrn_pair(lo, hi);
    // t1 = (v[15] * 201 - v[0] * 4091) >> 12
    let (lo, hi) = smull_smlsl_q(v[15], v[0], c0, 1, 0);
    let t1 = sqrshrn_pair(lo, hi);

    // t2 = (v[13] * 3973 + v[2] * 995) >> 12
    let (lo, hi) = smull_smlal_q(v[13], v[2], c0, 2, 3);
    let t2 = sqrshrn_pair(lo, hi);
    // t3 = (v[13] * 995 - v[2] * 3973) >> 12
    let (lo, hi) = smull_smlsl_q(v[13], v[2], c0, 3, 2);
    let t3 = sqrshrn_pair(lo, hi);

    // t4 = (v[11] * 3703 + v[4] * 1751) >> 12
    let (lo, hi) = smull_smlal_q(v[11], v[4], c0, 4, 5);
    let t4 = sqrshrn_pair(lo, hi);
    // t5 = (v[11] * 1751 - v[4] * 3703) >> 12
    let (lo, hi) = smull_smlsl_q(v[11], v[4], c0, 5, 4);
    let t5 = sqrshrn_pair(lo, hi);

    // t6 = (v[9] * 3290 + v[6] * 2440) >> 12
    let (lo, hi) = smull_smlal_q(v[9], v[6], c0, 6, 7);
    let t6 = sqrshrn_pair(lo, hi);
    // t7 = (v[9] * 2440 - v[6] * 3290) >> 12
    let (lo, hi) = smull_smlsl_q(v[9], v[6], c0, 7, 6);
    let t7 = sqrshrn_pair(lo, hi);

    // t8 = (v[7] * 2751 + v[8] * 3035) >> 12
    let (lo, hi) = smull_smlal_q(v[7], v[8], c1, 0, 1);
    let t8 = sqrshrn_pair(lo, hi);
    // t9 = (v[7] * 3035 - v[8] * 2751) >> 12
    let (lo, hi) = smull_smlsl_q(v[7], v[8], c1, 1, 0);
    let t9 = sqrshrn_pair(lo, hi);

    // t10 = (v[5] * 2106 + v[10] * 3513) >> 12
    let (lo, hi) = smull_smlal_q(v[5], v[10], c1, 2, 3);
    let t10 = sqrshrn_pair(lo, hi);
    // t11 = (v[5] * 3513 - v[10] * 2106) >> 12
    let (lo, hi) = smull_smlsl_q(v[5], v[10], c1, 3, 2);
    let t11 = sqrshrn_pair(lo, hi);

    // t12 = (v[3] * 1380 + v[12] * 3857) >> 12
    let (lo, hi) = smull_smlal_q(v[3], v[12], c1, 4, 5);
    let t12 = sqrshrn_pair(lo, hi);
    // t13 = (v[3] * 3857 - v[12] * 1380) >> 12
    let (lo, hi) = smull_smlsl_q(v[3], v[12], c1, 5, 4);
    let t13 = sqrshrn_pair(lo, hi);

    // t14 = (v[1] * 601 + v[14] * 4052) >> 12
    let (lo, hi) = smull_smlal_q(v[1], v[14], c1, 6, 7);
    let t14 = sqrshrn_pair(lo, hi);
    // t15 = (v[1] * 4052 - v[14] * 601) >> 12
    let (lo, hi) = smull_smlsl_q(v[1], v[14], c1, 7, 6);
    let t15 = sqrshrn_pair(lo, hi);

    // Load idct coefficients for the remaining stages
    let ci = safe_simd::vld1q_s16(<&[i16; 8]>::try_from(&IDCT_COEFFS[0..8]).unwrap());

    // Stage 2: butterfly pairs
    let s8a = vqsubq_s16(t0, t8); // t8a
    let s0a = vqaddq_s16(t0, t8); // t0a
    let s9a = vqsubq_s16(t1, t9); // t9a
    let s1a = vqaddq_s16(t1, t9); // t1a
    let s2a = vqaddq_s16(t2, t10); // t2a
    let s10a = vqsubq_s16(t2, t10); // t10a
    let s3a = vqaddq_s16(t3, t11); // t3a
    let s11a = vqsubq_s16(t3, t11); // t11a
    let s4a = vqaddq_s16(t4, t12); // t4a
    let s12a = vqsubq_s16(t4, t12); // t12a
    let s5a = vqaddq_s16(t5, t13); // t5a
    let s13a = vqsubq_s16(t5, t13); // t13a
    let s6a = vqaddq_s16(t6, t14); // t6a
    let s14a = vqsubq_s16(t6, t14); // t14a
    let s7a = vqaddq_s16(t7, t15); // t7a
    let s15a = vqsubq_s16(t7, t15); // t15a

    // Stage 3: rotations
    // ci.h[4]=799, ci.h[5]=4017, ci.h[6]=3406, ci.h[7]=2276
    // t8  = (s8a * 4017 + s9a * 799) >> 12
    let (lo, hi) = smull_smlal_q(s8a, s9a, ci, 5, 4);
    let u8_ = sqrshrn_pair(lo, hi);
    // t9  = (s8a * 799 - s9a * 4017) >> 12
    let (lo, hi) = smull_smlsl_q(s8a, s9a, ci, 4, 5);
    let u9 = sqrshrn_pair(lo, hi);

    // t10 = (s10a * 2276 + s11a * 3406) >> 12
    let (lo, hi) = smull_smlal_q(s10a, s11a, ci, 7, 6);
    let u10 = sqrshrn_pair(lo, hi);
    // t11 = (s10a * 3406 - s11a * 2276) >> 12
    let (lo, hi) = smull_smlsl_q(s10a, s11a, ci, 6, 7);
    let u11 = sqrshrn_pair(lo, hi);

    // t12 = -(s13a * 4017 - s12a * 799) >> 12
    let (lo, hi) = smull_smlsl_q(s13a, s12a, ci, 5, 4);
    let u12 = sqrshrn_pair(lo, hi);
    // t13 = (s13a * 799 + s12a * 4017) >> 12
    let (lo, hi) = smull_smlal_q(s13a, s12a, ci, 4, 5);
    let u13 = sqrshrn_pair(lo, hi);

    // t14 = -(s15a * 2276 - s14a * 3406) >> 12
    let (lo, hi) = smull_smlsl_q(s15a, s14a, ci, 7, 6);
    let u14 = sqrshrn_pair(lo, hi);
    // t15 = (s15a * 3406 + s14a * 2276) >> 12
    let (lo, hi) = smull_smlal_q(s15a, s14a, ci, 6, 7);
    let u15 = sqrshrn_pair(lo, hi);

    // Stage 4: butterfly pairs
    let w4 = vqsubq_s16(s0a, s4a); // t4
    let w0 = vqaddq_s16(s0a, s4a); // t0
    let w5 = vqsubq_s16(s1a, s5a); // t5
    let w1 = vqaddq_s16(s1a, s5a); // t1
    let w2 = vqaddq_s16(s2a, s6a); // t2
    let w6 = vqsubq_s16(s2a, s6a); // t6
    let w3 = vqaddq_s16(s3a, s7a); // t3
    let w7 = vqsubq_s16(s3a, s7a); // t7
    let w8a = vqaddq_s16(u8_, u12); // t8a
    let w12a = vqsubq_s16(u8_, u12); // t12a
    let w9a = vqaddq_s16(u9, u13); // t9a
    let w13a = vqsubq_s16(u9, u13); // t13a
    let w10a = vqaddq_s16(u10, u14); // t10a
    let w14a = vqsubq_s16(u10, u14); // t14a
    let w11a = vqaddq_s16(u11, u15); // t11a
    let w15a = vqsubq_s16(u11, u15); // t15a

    // Stage 5: rotations using ci.h[2]=1567, ci.h[3]=3784
    // t4a = (w4 * 3784 + w5 * 1567) >> 12
    let (lo, hi) = smull_smlal_q(w4, w5, ci, 3, 2);
    let x4a = sqrshrn_pair(lo, hi);
    // t5a = (w4 * 1567 - w5 * 3784) >> 12
    let (lo, hi) = smull_smlsl_q(w4, w5, ci, 2, 3);
    let x5a = sqrshrn_pair(lo, hi);

    // t6a = -(w7 * 3784 - w6 * 1567) >> 12 = (w6 * 1567 - w7 * 3784) >> 12
    let (lo, hi) = smull_smlsl_q(w7, w6, ci, 3, 2);
    let x6a = sqrshrn_pair(lo, hi);
    // t7a = (w7 * 1567 + w6 * 3784) >> 12
    let (lo, hi) = smull_smlal_q(w7, w6, ci, 2, 3);
    let x7a = sqrshrn_pair(lo, hi);

    // t12 = (w12a * 3784 + w13a * 1567) >> 12
    let (lo, hi) = smull_smlal_q(w12a, w13a, ci, 3, 2);
    let x12 = sqrshrn_pair(lo, hi);
    // t13 = (w12a * 1567 - w13a * 3784) >> 12
    let (lo, hi) = smull_smlsl_q(w12a, w13a, ci, 2, 3);
    let x13 = sqrshrn_pair(lo, hi);

    // t14 = -(w15a * 3784 - w14a * 1567) >> 12 = (w14a * 1567 - w15a * 3784) >> 12
    let (lo, hi) = smull_smlsl_q(w15a, w14a, ci, 3, 2);
    let x14 = sqrshrn_pair(lo, hi);
    // t15 = (w15a * 1567 + w14a * 3784) >> 12
    let (lo, hi) = smull_smlal_q(w15a, w14a, ci, 2, 3);
    let x15 = sqrshrn_pair(lo, hi);

    // Stage 6: Final butterfly + rotations

    // t2a = w0 - w2
    let t2a = vqsubq_s16(w0, w2);
    // out0 = w0 + w2
    let o0 = vqaddq_s16(w0, w2);
    // t3a = w1 - w3
    let t3a = vqsubq_s16(w1, w3);
    // out15 = -(w1 + w3)
    let o15 = vqnegq_s16(vqaddq_s16(w1, w3));

    // t15a = x13 - x15
    let t15a = vqsubq_s16(x13, x15);
    // out13 = -(x13 + x15)
    let o13 = vqnegq_s16(vqaddq_s16(x13, x15));
    // out2 = x12 + x14
    let o2 = vqaddq_s16(x12, x14);
    // t14a = x12 - x14
    let t14a = vqsubq_s16(x12, x14);

    // out1 = -(w8a + w10a)
    let o1 = vqnegq_s16(vqaddq_s16(w8a, w10a));
    // t10 = w8a - w10a
    let y10 = vqsubq_s16(w8a, w10a);
    // out14 = w9a + w11a
    let o14 = vqaddq_s16(w9a, w11a);
    // t11 = w9a - w11a
    let y11 = vqsubq_s16(w9a, w11a);

    // out3 = -(x4a + x6a)
    let o3 = vqnegq_s16(vqaddq_s16(x4a, x6a));
    // t6 = x4a - x6a
    let y6 = vqsubq_s16(x4a, x6a);
    // out12 = x5a + x7a
    let o12 = vqaddq_s16(x5a, x7a);
    // t7 = x5a - x7a
    let y7 = vqsubq_s16(x5a, x7a);

    // Final rotations using ci.h[0]=2896
    // out8  = (t2a * 2896 - t3a * 2896 + 2048) >> 12
    let (lo, hi) = smull_smlsl_q(t2a, t3a, ci, 0, 0);
    let o8 = sqrshrn_pair(lo, hi);
    // out7  = (t2a * 2896 + t3a * 2896 + 2048) >> 12
    let (lo, hi) = smull_smlal_q(t2a, t3a, ci, 0, 0);
    let o7_pre = sqrshrn_pair(lo, hi);

    // out5  = (t14a * 2896 + t15a * 2896 + 2048) >> 12
    let (lo, hi) = smull_smlal_q(t14a, t15a, ci, 0, 0);
    let o5_pre = sqrshrn_pair(lo, hi);
    // out10 = (t14a * 2896 - t15a * 2896 + 2048) >> 12
    let (lo, hi) = smull_smlsl_q(t14a, t15a, ci, 0, 0);
    let o10 = sqrshrn_pair(lo, hi);

    // out4  = (y6 * 2896 + y7 * 2896 + 2048) >> 12
    let (lo, hi) = smull_smlal_q(y6, y7, ci, 0, 0);
    let o4 = sqrshrn_pair(lo, hi);
    // out11 = (y6 * 2896 - y7 * 2896 + 2048) >> 12
    let (lo, hi) = smull_smlsl_q(y6, y7, ci, 0, 0);
    let o11_pre = sqrshrn_pair(lo, hi);

    // out6  = (y10 * 2896 + y11 * 2896 + 2048) >> 12
    let (lo, hi) = smull_smlal_q(y10, y11, ci, 0, 0);
    let o6 = sqrshrn_pair(lo, hi);
    // out9  = (y10 * 2896 - y11 * 2896 + 2048) >> 12
    let (lo, hi) = smull_smlsl_q(y10, y11, ci, 0, 0);
    let o9_pre = sqrshrn_pair(lo, hi);

    // Apply negations per assembly output mapping
    let o7 = vqnegq_s16(o7_pre);
    let o5 = vqnegq_s16(o5_pre);
    let o11 = vqnegq_s16(o11_pre);
    let o9 = vqnegq_s16(o9_pre);

    [
        o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15,
    ]
}

// ============================================================================
// 16-point identity transform (itx.S lines 1320-1329)
// ============================================================================

/// 16-point identity transform on 16 int16x8_t vectors.
///
/// For 16x16 blocks, the identity transform multiplies by 2*sqrt(2).
/// This is implemented as: x * 2 + (x * ((5793-4096)*2*8) + 0x4000) >> 15
///
/// Assembly:
/// ```text
/// mov w16, #2*(5793-4096)*8   // 27152
/// dup v0.4h, w16
/// sqrdmulh v2.8h, vi.8h, v0.h[0]
/// sqadd    vi.8h, vi.8h, vi.8h    // vi *= 2
/// sqadd    vi.8h, vi.8h, v2.8h    // vi += round(vi * scale)
/// ```
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
pub(crate) fn identity_16_q(mut v: V16) -> V16 {
    // Scale factor: 2*(5793-4096)*8 = 27152
    // This fits in i16 (max 32767)
    let scale = vdupq_n_s16(((5793 - 4096) * 2 * 8) as i16);

    for vi in v.iter_mut() {
        let t = vqrdmulhq_s16(*vi, scale);
        *vi = vqaddq_s16(*vi, *vi); // vi *= 2
        *vi = vqaddq_s16(*vi, t); // vi += round(vi * scale)
    }
    v
}

// ============================================================================
// 16x16 transpose (done as 4 blocks of 8x8)
// ============================================================================

/// Transpose a 16x16 matrix stored as two halves.
///
/// The 16x16 block is split into:
///   top:    rows 0-7    (8 vectors)
///   bottom: rows 8-15   (8 vectors)
///
/// After transpose:
///   left:   columns 0-7  → new rows 0-7
///   right:  columns 8-15 → new rows 8-15
///
/// This uses 4 blocks of 8x8 transpose:
///   TL = transpose(top[0..8].lo)     → rows 0-7, cols 0-7
///   TR = transpose(top[0..8].hi)     → rows 0-7, cols 8-15
///   BL = transpose(bottom[0..8].lo)  → rows 8-15, cols 0-7
///   BR = transpose(bottom[0..8].hi)  → rows 8-15, cols 8-15
///
/// After transpose, swap TR and BL (off-diagonal blocks).
///
/// But actually, in the assembly the 16x16 is stored row-interleaved
/// across the two halves. The horizontal transform processes 16 values
/// per row, stored as two 8x8 groups: rows 0-7 of both halves are
/// processed, then rows 8-15.
///
/// For the 16x16 case specifically, the transpose happens on the
/// intermediate buffer, working with 16-element rows stored as
/// two consecutive int16x8_t vectors.
///
/// This function takes two arrays of 8 vectors each (representing
/// the top and bottom halves) and produces two new arrays.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn transpose_16x16_half(a: [int16x8_t; 8], b: [int16x8_t; 8]) -> ([int16x8_t; 8], [int16x8_t; 8]) {
    let (a0, a1, a2, a3, a4, a5, a6, a7) =
        transpose_8x8h(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
    let (b0, b1, b2, b3, b4, b5, b6, b7) =
        transpose_8x8h(b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]);
    (
        [a0, a1, a2, a3, a4, a5, a6, a7],
        [b0, b1, b2, b3, b4, b5, b6, b7],
    )
}

// ============================================================================
// Add transform output to 8x16 destination block (8bpc)
// ============================================================================

/// Add 16 rows of 8 pixels to destination for 8bpc.
///
/// For each of 16 rows:
///   1. Apply rounding shift right by 4
///   2. Load 8 dst u8 pixels, zero-extend to u16
///   3. Add i16 transform result to u16 pixels
///   4. Saturate i16→u8
///   5. Store 8 bytes
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn add_to_dst_8x16_8bpc(dst: &mut [u8], dst_base: usize, stride: isize, v: V16) {
    for (i, &row) in v.iter().enumerate() {
        let row_off = dst_base.wrapping_add_signed(i as isize * stride);

        // srshr by 4 (rounding shift right)
        let shifted = vrshrq_n_s16::<4>(row);

        // Load 8 u8 destination pixels
        let dst_bytes: [u8; 8] = dst[row_off..row_off + 8].try_into().unwrap();
        let dst_u8 = safe_simd::vld1_u8(&dst_bytes);

        // uaddw: zero-extend u8 to u16, add i16
        let sum = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(shifted), dst_u8));

        // sqxtun: saturating narrow i16→u8
        let result = vqmovun_s16(sum);

        // Store 8 bytes
        let mut out = [0u8; 8];
        safe_simd::vst1_u8(&mut out, result);
        dst[row_off..row_off + 8].copy_from_slice(&out);
    }
}

/// Add 16 rows of 8 pixels to destination for 16bpc.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn add_to_dst_8x16_16bpc(
    dst: &mut [u16],
    dst_base: usize,
    stride: isize,
    v: V16,
    bitdepth_max: i32,
) {
    let bd_max = vdupq_n_s16(bitdepth_max as i16);
    let zero = vdupq_n_s16(0);

    for (i, &row) in v.iter().enumerate() {
        let row_off = dst_base.wrapping_add_signed(i as isize * stride);

        // srshr by 4
        let shifted = vrshrq_n_s16::<4>(row);

        // Load 8 u16 destination pixels
        let dst_arr: [i16; 8] = {
            let mut tmp = [0i16; 8];
            for j in 0..8 {
                tmp[j] = dst[row_off + j] as i16;
            }
            tmp
        };
        let dst_vals = safe_simd::vld1q_s16(&dst_arr);

        // Add
        let sum = vqaddq_s16(dst_vals, shifted);

        // Clamp to [0, bitdepth_max]
        let clamped = vminq_s16(vmaxq_s16(sum, zero), bd_max);

        // Store
        let mut out = [0i16; 8];
        safe_simd::vst1q_s16(&mut out, clamped);
        for j in 0..8 {
            dst[row_off + j] = out[j] as u16;
        }
    }
}

// ============================================================================
// Transform type enum for 16x16 blocks
// ============================================================================

/// Row/column transform type for 16x16 blocks.
#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
pub(crate) enum TxType16 {
    Dct,
    Adst,
    FlipAdst,
    Identity,
}

/// Apply a 1D 16-point transform to 16 int16x8_t vectors.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn apply_tx16(tx: TxType16, v: V16) -> V16 {
    match tx {
        TxType16::Dct => idct_16_q(v),
        TxType16::Adst => iadst_16_q(v),
        TxType16::FlipAdst => {
            let mut out = iadst_16_q(v);
            // Reverse the output order
            out.reverse();
            out
        }
        TxType16::Identity => identity_16_q(v),
    }
}

// ============================================================================
// DC-only fast path (idct_dc 16, 16, 2)
// ============================================================================

/// DC-only fast path for DCT_DCT 16x16 with eob=0.
///
/// Algorithm (from assembly `idct_dc 16, 16, 2`):
/// ```text
/// dc = coeff[0]
/// coeff[0] = 0
/// scale = 2896*8 = 23168
/// dc = sqrdmulh(dc, scale)     // first scaling
/// dc = srshr(dc, 2)            // shift=2 for 16x16
/// dc = sqrdmulh(dc, scale)     // second scaling
/// dc = srshr(dc, 4)            // final shift
/// ```
/// Then broadcast to all 16x16 pixels and add.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn dc_only_16x16_8bpc(dst: &mut [u8], dst_base: usize, dst_stride: isize, coeff: &mut [i16]) {
    let dc = coeff[0];
    coeff[0] = 0;

    let scale = vdupq_n_s16((2896 * 8) as i16); // 23168

    let v = vdupq_n_s16(dc);
    let v = vqrdmulhq_s16(v, scale);
    let v = vrshrq_n_s16::<2>(v); // shift=2 for 16x16
    let v = vqrdmulhq_s16(v, scale);
    let v = vrshrq_n_s16::<4>(v); // final shift

    // Add to 16x16 destination (two 8-wide halves per row)
    for i in 0..16 {
        let row_off = dst_base.wrapping_add_signed(i as isize * dst_stride);

        // Left 8 pixels
        let dst_bytes: [u8; 8] = dst[row_off..row_off + 8].try_into().unwrap();
        let dst_u8 = safe_simd::vld1_u8(&dst_bytes);
        let sum = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(v), dst_u8));
        let result = vqmovun_s16(sum);
        let mut out = [0u8; 8];
        safe_simd::vst1_u8(&mut out, result);
        dst[row_off..row_off + 8].copy_from_slice(&out);

        // Right 8 pixels
        let dst_bytes: [u8; 8] = dst[row_off + 8..row_off + 16].try_into().unwrap();
        let dst_u8 = safe_simd::vld1_u8(&dst_bytes);
        let sum = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(v), dst_u8));
        let result = vqmovun_s16(sum);
        let mut out = [0u8; 8];
        safe_simd::vst1_u8(&mut out, result);
        dst[row_off + 8..row_off + 16].copy_from_slice(&out);
    }
}

/// DC-only fast path for DCT_DCT 16x16 16bpc.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn dc_only_16x16_16bpc(
    dst: &mut [u16],
    dst_base: usize,
    dst_stride: isize,
    coeff: &mut [i32],
    bitdepth_max: i32,
) {
    let dc_val = coeff[0];
    coeff[0] = 0;

    // For 16bpc, scaling is different but same idea
    let scale = 2896i32 * 8;
    let mut dc = ((dc_val as i64 * scale as i64 + 16384) >> 15) as i32;
    dc = (dc + 2) >> 2; // shift=2 for 16x16
    dc = ((dc as i64 * scale as i64 + 16384) >> 15) as i32;
    dc = (dc + 8) >> 4; // final shift

    let dc = dc.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    let dc_vec = vdupq_n_s16(dc);
    let bd_max = vdupq_n_s16(bitdepth_max as i16);
    let zero = vdupq_n_s16(0);

    for i in 0..16 {
        let row_off = dst_base.wrapping_add_signed(i as isize * dst_stride);

        for half in 0..2 {
            let off = row_off + half * 8;
            let mut arr = [0i16; 8];
            for j in 0..8 {
                arr[j] = dst[off + j] as i16;
            }
            let d = safe_simd::vld1q_s16(&arr);
            let sum = vqaddq_s16(d, dc_vec);
            let clamped = vminq_s16(vmaxq_s16(sum, zero), bd_max);
            let mut out = [0i16; 8];
            safe_simd::vst1q_s16(&mut out, clamped);
            for j in 0..8 {
                dst[off + j] = out[j] as u16;
            }
        }
    }
}

// ============================================================================
// Generic 16x16 inverse transform (assembly lines 1374-1464)
// ============================================================================

/// NEON implementation of generic 16x16 inverse transform add for 8bpc.
///
/// Mirrors `inv_txfm_add_16x16_neon` from itx.S lines 1433-1464.
///
/// Algorithm:
/// 1. Process two 8-column halves of the 16x16 coefficient block:
///    a. Load 16 rows x 8 columns into V16
///    b. Clear coefficient buffer
///    c. Apply row transform (or identity)
///    d. For non-identity: shift right by 2
///    e. For identity: apply identity_8x16_shift2 (sqrdmulh + sshr + srhadd)
///    f. Transpose two 8x8 blocks within each half
///    g. Store interleaved to 512-byte temp buffer
///
/// 2. Process two 8-column halves of the transposed result:
///    a. Load 16 rows from temp buffer
///    b. Apply column transform
///    c. Add to destination
/// `eob` below which a 16x16 transform's rows 8..15 are guaranteed zero.
///
/// From `src/arm/64/itx.S`, `def_fn_16x16 <txfm1>, <txfm2>, <eob_half>`:
/// 36 for every pairing of {dct, adst, flipadst} plus identity/identity, and
/// **8** for `dct, identity` (H_DCT) and `identity, dct` (V_DCT).
///
/// Being too LOW is always safe — it just transforms an all-zero row group,
/// which contributes nothing. Being too HIGH silently discards coefficients.
/// So the four pairings AV1 does not permit at 16x16 (H_ADST, V_ADST,
/// H_FLIPADST, V_FLIPADST — dav1d has no 16x16 kernel for them at all) take
/// the conservative 8 rather than an invented value.
#[cfg(target_arch = "aarch64")]
fn eob_half_16x16(row_tx: TxType16, col_tx: TxType16) -> i32 {
    use TxType16::Identity;
    match (row_tx, col_tx) {
        // IDTX is TX_CLASS_2D: it uses the same zigzag as the DTTs.
        (Identity, Identity) => 36,
        // Any other identity pairing is a 1-D scan class.
        (Identity, _) | (_, Identity) => 8,
        _ => 36,
    }
}

#[cfg(target_arch = "aarch64")]
#[arcane]
pub(crate) fn inv_txfm_add_16x16_8bpc_neon(
    _token: Arm64,
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    coeff: &mut [i16],
    eob: i32,
    _bitdepth_max: i32,
    row_tx: TxType16,
    col_tx: TxType16,
) {
    // DC-only fast path for DCT_DCT with eob=0
    if matches!(row_tx, TxType16::Dct) && matches!(col_tx, TxType16::Dct) && eob == 0 {
        dc_only_16x16_8bpc(dst, dst_base, dst_stride, coeff);
        return;
    }

    let is_identity_row = matches!(row_tx, TxType16::Identity);

    // Intermediate buffer: 16 rows x 16 columns = 256 i16 values
    // Layout: tmp[row * 16 + col] in row-major order.
    // After row transform + transpose, this holds the transposed
    // row-transform output ready for the column transform.
    let mut tmp = [0i16; 256];

    // The eob threshold below which the second 8-row group is all-zero and can
    // be skipped. It is PER TRANSFORM TYPE, because the coefficient scan is:
    // `dav1d_tx_type_class` puts H_DCT in TX_CLASS_H, whose `rc` is the scan
    // index itself, so eob = 8 already lands on coefficient row 8 — while the
    // 2-D zigzag needs eob = 36 to get there. Hardcoding 36 for every type
    // dropped real coefficients on H_DCT for every eob in [8, 36).
    //
    // Mirrors `src/arm/64/itx.S`'s `def_fn_16x16 <txfm1>, <txfm2>, <eob_half>`
    // table exactly for the twelve types AV1 permits at 16x16.
    let eob_half = eob_half_16x16(row_tx, col_tx);

    // Row transform: process two groups of 8 rows each
    // Each group loads ALL 16 columns but only 8 rows.
    // half=0: rows 0-7;  half=1: rows 8-15
    for half in 0..2usize {
        // Skip second half if eob is small enough
        if half == 1 && eob < eob_half {
            // tmp is already zero-initialized; leave second half as zeros
            break;
        }

        let row_offset = half * 8;

        // Load: v[col] = rows [row_offset..row_offset+8] of column col
        // coeff is column-major: coeff[row + col*16]
        // So column c, rows r..r+8 = coeff[c*16+r .. c*16+r+8]
        let zero_vec = vdupq_n_s16(0);
        let mut v: V16 = [zero_vec; 16];
        for col in 0..16 {
            let base = col * 16 + row_offset;
            let arr: [i16; 8] = coeff[base..base + 8].try_into().unwrap();
            v[col] = safe_simd::vld1q_s16(&arr);
        }

        // Clear the loaded coefficients
        for col in 0..16 {
            let base = col * 16 + row_offset;
            coeff[base..base + 8].fill(0);
        }

        if is_identity_row {
            // Identity with shift2: sqrdmulh(vi, scale) >> 1, then srhadd
            // Scale factor: (5793-4096)*2*8 = 27152
            let scale = vdupq_n_s16(((5793 - 4096) * 2 * 8) as i16);
            for vi in v.iter_mut() {
                let t = vqrdmulhq_s16(*vi, scale);
                let t = vshrq_n_s16::<1>(t);
                *vi = vrhaddq_s16(*vi, t);
            }
        } else {
            // Apply row transform across 16 columns (8 rows in parallel)
            v = apply_tx16(row_tx, v);

            // Shift right by 2 (16x16 row shift)
            for vi in v.iter_mut() {
                *vi = vrshrq_n_s16::<2>(*vi);
            }
        }

        // Transpose: Before transpose, v[col] holds 8 rows at that column.
        // We split into two 8x8 blocks:
        //   cols 0-7 (v[0..8])  → transpose → now v[row] holds cols 0-7
        //   cols 8-15 (v[8..16]) → transpose → now v[row] holds cols 8-15
        // Then store interleaved to tmp as row-major.
        let top = [v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]];
        let bot = [v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15]];

        let (top_t, bot_t) = transpose_16x16_half(top, bot);

        // After transpose:
        //   top_t[i] = row i, cols 0-7 (for original rows row_offset..row_offset+8)
        //   bot_t[i] = row i, cols 8-15
        // Store to tmp[row * 16 + col]:
        //   - For rows in this half: top_t maps to rows 0-7, bot_t to rows 0-7
        //   - Each row's data: [top_t[i](cols 0-7), bot_t[i](cols 8-15)]
        //
        // Assembly stores interleaved: top_t[0],bot_t[0],top_t[1],bot_t[1],...
        // to contiguous memory at sp + half * 256
        // Which gives: tmp[(half*8+i)*16 + 0..8] = top_t[i]
        //              tmp[(half*8+i)*16 + 8..16] = bot_t[i]
        for i in 0..8 {
            let row = half * 8 + i;
            let mut arr = [0i16; 8];

            safe_simd::vst1q_s16(&mut arr, top_t[i]);
            tmp[row * 16..row * 16 + 8].copy_from_slice(&arr);

            safe_simd::vst1q_s16(&mut arr, bot_t[i]);
            tmp[row * 16 + 8..row * 16 + 16].copy_from_slice(&arr);
        }
    }

    // Column transform: process two groups of 8 columns each
    // half=0: columns 0-7;  half=1: columns 8-15
    // For each group, load ALL 16 rows but only 8 columns.
    // v[row] = 8 column values at that row.
    for half in 0..2usize {
        let col_offset = half * 8;
        let zero_vec = vdupq_n_s16(0);
        let mut v: V16 = [zero_vec; 16];
        for row in 0..16 {
            let off = row * 16 + col_offset;
            let arr: [i16; 8] = tmp[off..off + 8].try_into().unwrap();
            v[row] = safe_simd::vld1q_s16(&arr);
        }

        // Apply column transform across 16 rows (8 columns in parallel)
        v = apply_tx16(col_tx, v);

        // Add to destination (8 pixels per row, offset by col_offset)
        add_to_dst_8x16_8bpc(dst, dst_base + col_offset, dst_stride, v);
    }
}

/// NEON implementation of generic 16x16 inverse transform add for 16bpc.
#[cfg(target_arch = "aarch64")]
#[arcane]
pub(crate) fn inv_txfm_add_16x16_16bpc_neon(
    _token: Arm64,
    dst: &mut [u16],
    dst_base: usize,
    dst_stride: isize,
    coeff: &mut [i32],
    eob: i32,
    bitdepth_max: i32,
    row_tx: TxType16,
    col_tx: TxType16,
) {
    // DC-only fast path
    if matches!(row_tx, TxType16::Dct) && matches!(col_tx, TxType16::Dct) && eob == 0 {
        dc_only_16x16_16bpc(dst, dst_base, dst_stride, coeff, bitdepth_max);
        return;
    }

    let is_identity_row = matches!(row_tx, TxType16::Identity);

    let mut tmp = [0i16; 256];
    // Same per-type table as the 8bpc kernel above. This 16bpc variant is not
    // wired into `itxfm_add_dispatch` (every arm there is BPC8 on purpose —
    // the 16bpc transforms are not yet bit-exact), but the threshold is a
    // property of the coefficient scan, not of the bit depth, so it is fixed
    // here too rather than left as a second copy of the bug.
    let eob_half = eob_half_16x16(row_tx, col_tx);

    // Row transform: two groups of 8 rows
    for half in 0..2usize {
        if half == 1 && eob < eob_half {
            break;
        }

        let row_offset = half * 8;

        // Load: v[col] = rows [row_offset..row_offset+8] of column col
        // coeff is column-major i32: coeff[row + col*16]
        let zero_vec = vdupq_n_s16(0);
        let mut v: V16 = [zero_vec; 16];
        for col in 0..16 {
            let mut arr = [0i16; 8];
            for r in 0..8 {
                let c = coeff[(col * 16) + row_offset + r];
                arr[r] = c.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            }
            v[col] = safe_simd::vld1q_s16(&arr);
        }

        // Clear loaded coefficients
        for col in 0..16 {
            for r in 0..8 {
                coeff[(col * 16) + row_offset + r] = 0;
            }
        }

        if is_identity_row {
            let scale = vdupq_n_s16(((5793 - 4096) * 2 * 8) as i16);
            for vi in v.iter_mut() {
                let t = vqrdmulhq_s16(*vi, scale);
                let t = vshrq_n_s16::<1>(t);
                *vi = vrhaddq_s16(*vi, t);
            }
        } else {
            v = apply_tx16(row_tx, v);

            for vi in v.iter_mut() {
                *vi = vrshrq_n_s16::<2>(*vi);
            }
        }

        // Transpose and store
        let top = [v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]];
        let bot = [v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15]];
        let (top_t, bot_t) = transpose_16x16_half(top, bot);

        for i in 0..8 {
            let row = half * 8 + i;
            let mut arr = [0i16; 8];

            safe_simd::vst1q_s16(&mut arr, top_t[i]);
            tmp[row * 16..row * 16 + 8].copy_from_slice(&arr);

            safe_simd::vst1q_s16(&mut arr, bot_t[i]);
            tmp[row * 16 + 8..row * 16 + 16].copy_from_slice(&arr);
        }
    }

    // Column transform: two groups of 8 columns
    for half in 0..2usize {
        let col_offset = half * 8;
        let zero_vec = vdupq_n_s16(0);
        let mut v: V16 = [zero_vec; 16];
        for row in 0..16 {
            let off = row * 16 + col_offset;
            let arr: [i16; 8] = tmp[off..off + 8].try_into().unwrap();
            v[row] = safe_simd::vld1q_s16(&arr);
        }

        v = apply_tx16(col_tx, v);

        let col_off = half * 8;
        add_to_dst_8x16_16bpc(dst, dst_base + col_off, dst_stride, v, bitdepth_max);
    }
}

// ============================================================================
// Public entry points for each 16x16 transform combination
// ============================================================================

macro_rules! def_16x16_8bpc {
    ($name:ident, $row:expr, $col:expr) => {
        #[cfg(target_arch = "aarch64")]
        #[arcane]
        pub(crate) fn $name(
            token: Arm64,
            dst: &mut [u8],
            dst_base: usize,
            dst_stride: isize,
            coeff: &mut [i16],
            eob: i32,
            bitdepth_max: i32,
        ) {
            inv_txfm_add_16x16_8bpc_neon(
                token,
                dst,
                dst_base,
                dst_stride,
                coeff,
                eob,
                bitdepth_max,
                $row,
                $col,
            );
        }
    };
}

macro_rules! def_16x16_16bpc {
    ($name:ident, $row:expr, $col:expr) => {
        #[cfg(target_arch = "aarch64")]
        #[arcane]
        pub(crate) fn $name(
            token: Arm64,
            dst: &mut [u16],
            dst_base: usize,
            dst_stride: isize,
            coeff: &mut [i32],
            eob: i32,
            bitdepth_max: i32,
        ) {
            inv_txfm_add_16x16_16bpc_neon(
                token,
                dst,
                dst_base,
                dst_stride,
                coeff,
                eob,
                bitdepth_max,
                $row,
                $col,
            );
        }
    };
}

// DCT_DCT
def_16x16_8bpc!(
    inv_txfm_add_dct_dct_16x16_8bpc_neon_inner,
    TxType16::Dct,
    TxType16::Dct
);
def_16x16_16bpc!(
    inv_txfm_add_dct_dct_16x16_16bpc_neon_inner,
    TxType16::Dct,
    TxType16::Dct
);

// IDENTITY_IDENTITY
def_16x16_8bpc!(
    inv_txfm_add_identity_identity_16x16_8bpc_neon_inner,
    TxType16::Identity,
    TxType16::Identity
);
def_16x16_16bpc!(
    inv_txfm_add_identity_identity_16x16_16bpc_neon_inner,
    TxType16::Identity,
    TxType16::Identity
);

// ADST_ADST
def_16x16_8bpc!(
    inv_txfm_add_adst_adst_16x16_8bpc_neon_inner,
    TxType16::Adst,
    TxType16::Adst
);
def_16x16_16bpc!(
    inv_txfm_add_adst_adst_16x16_16bpc_neon_inner,
    TxType16::Adst,
    TxType16::Adst
);

// DCT_ADST: function name = row_col, so row=Dct, col=Adst
def_16x16_8bpc!(
    inv_txfm_add_dct_adst_16x16_8bpc_neon_inner,
    TxType16::Dct,
    TxType16::Adst
);
def_16x16_16bpc!(
    inv_txfm_add_dct_adst_16x16_16bpc_neon_inner,
    TxType16::Dct,
    TxType16::Adst
);

// ADST_DCT: row=Adst, col=Dct
def_16x16_8bpc!(
    inv_txfm_add_adst_dct_16x16_8bpc_neon_inner,
    TxType16::Adst,
    TxType16::Dct
);
def_16x16_16bpc!(
    inv_txfm_add_adst_dct_16x16_16bpc_neon_inner,
    TxType16::Adst,
    TxType16::Dct
);

// DCT_FLIPADST: row=Dct, col=FlipAdst
def_16x16_8bpc!(
    inv_txfm_add_dct_flipadst_16x16_8bpc_neon_inner,
    TxType16::Dct,
    TxType16::FlipAdst
);
def_16x16_16bpc!(
    inv_txfm_add_dct_flipadst_16x16_16bpc_neon_inner,
    TxType16::Dct,
    TxType16::FlipAdst
);

// FLIPADST_DCT: row=FlipAdst, col=Dct
def_16x16_8bpc!(
    inv_txfm_add_flipadst_dct_16x16_8bpc_neon_inner,
    TxType16::FlipAdst,
    TxType16::Dct
);
def_16x16_16bpc!(
    inv_txfm_add_flipadst_dct_16x16_16bpc_neon_inner,
    TxType16::FlipAdst,
    TxType16::Dct
);

// FLIPADST_FLIPADST: row=FlipAdst, col=FlipAdst
def_16x16_8bpc!(
    inv_txfm_add_flipadst_flipadst_16x16_8bpc_neon_inner,
    TxType16::FlipAdst,
    TxType16::FlipAdst
);
def_16x16_16bpc!(
    inv_txfm_add_flipadst_flipadst_16x16_16bpc_neon_inner,
    TxType16::FlipAdst,
    TxType16::FlipAdst
);

// ADST_FLIPADST: row=Adst, col=FlipAdst
def_16x16_8bpc!(
    inv_txfm_add_adst_flipadst_16x16_8bpc_neon_inner,
    TxType16::Adst,
    TxType16::FlipAdst
);
def_16x16_16bpc!(
    inv_txfm_add_adst_flipadst_16x16_16bpc_neon_inner,
    TxType16::Adst,
    TxType16::FlipAdst
);

// FLIPADST_ADST: row=FlipAdst, col=Adst
def_16x16_8bpc!(
    inv_txfm_add_flipadst_adst_16x16_8bpc_neon_inner,
    TxType16::FlipAdst,
    TxType16::Adst
);
def_16x16_16bpc!(
    inv_txfm_add_flipadst_adst_16x16_16bpc_neon_inner,
    TxType16::FlipAdst,
    TxType16::Adst
);

// DCT_IDENTITY (H_DCT = row DCT, col identity)
def_16x16_8bpc!(
    inv_txfm_add_dct_identity_16x16_8bpc_neon_inner,
    TxType16::Dct,
    TxType16::Identity
);
def_16x16_16bpc!(
    inv_txfm_add_dct_identity_16x16_16bpc_neon_inner,
    TxType16::Dct,
    TxType16::Identity
);

// IDENTITY_DCT (V_DCT = row identity, col DCT)
def_16x16_8bpc!(
    inv_txfm_add_identity_dct_16x16_8bpc_neon_inner,
    TxType16::Identity,
    TxType16::Dct
);
def_16x16_16bpc!(
    inv_txfm_add_identity_dct_16x16_16bpc_neon_inner,
    TxType16::Identity,
    TxType16::Dct
);

// ADST_IDENTITY (H_ADST = row ADST, col identity)
def_16x16_8bpc!(
    inv_txfm_add_adst_identity_16x16_8bpc_neon_inner,
    TxType16::Adst,
    TxType16::Identity
);
def_16x16_16bpc!(
    inv_txfm_add_adst_identity_16x16_16bpc_neon_inner,
    TxType16::Adst,
    TxType16::Identity
);

// IDENTITY_ADST (V_ADST = row identity, col ADST)
def_16x16_8bpc!(
    inv_txfm_add_identity_adst_16x16_8bpc_neon_inner,
    TxType16::Identity,
    TxType16::Adst
);
def_16x16_16bpc!(
    inv_txfm_add_identity_adst_16x16_16bpc_neon_inner,
    TxType16::Identity,
    TxType16::Adst
);

// FLIPADST_IDENTITY (H_FLIPADST = row flipadst, col identity)
def_16x16_8bpc!(
    inv_txfm_add_flipadst_identity_16x16_8bpc_neon_inner,
    TxType16::FlipAdst,
    TxType16::Identity
);
def_16x16_16bpc!(
    inv_txfm_add_flipadst_identity_16x16_16bpc_neon_inner,
    TxType16::FlipAdst,
    TxType16::Identity
);

// IDENTITY_FLIPADST (V_FLIPADST = row identity, col flipadst)
def_16x16_8bpc!(
    inv_txfm_add_identity_flipadst_16x16_8bpc_neon_inner,
    TxType16::Identity,
    TxType16::FlipAdst
);
def_16x16_16bpc!(
    inv_txfm_add_identity_flipadst_16x16_16bpc_neon_inner,
    TxType16::Identity,
    TxType16::FlipAdst
);

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;
    use crate::include::common::intops::iclip;
    use archmage::SimdToken;

    /// Maximum per-pixel difference allowed between NEON and scalar.
    /// NEON matches the dav1d assembly (the reference); scalar Rust has known
    /// rounding differences that scale with coefficient magnitude and accumulate
    /// across stages. Larger transforms have more stages.
    const MAX_DIFF: i32 = 40;

    // ---- Scalar reference implementations ----

    fn scalar_dct4_1d(input: &[i32; 4]) -> [i32; 4] {
        let t3a = ((input[1] * 3784 + input[3] * 1567) + 2048) >> 12;
        let t2a = ((input[1] * 1567 - input[3] * 3784) + 2048) >> 12;
        let t0 = ((input[0] * 2896 + input[2] * 2896) + 2048) >> 12;
        let t1 = ((input[0] * 2896 - input[2] * 2896) + 2048) >> 12;
        [t0 + t3a, t1 + t2a, t1 - t2a, t0 - t3a]
    }

    fn scalar_dct8_1d(input: &[i32; 8]) -> [i32; 8] {
        let even = [input[0], input[2], input[4], input[6]];
        let even_out = scalar_dct4_1d(&even);

        let t4a = ((input[1] * 799 - input[7] * 4017) + 2048) >> 12;
        let t7a = ((input[1] * 4017 + input[7] * 799) + 2048) >> 12;
        let t5a = ((input[5] * 3406 - input[3] * 2276) + 2048) >> 12;
        let t6a = ((input[5] * 2276 + input[3] * 3406) + 2048) >> 12;

        let t4 = t4a + t5a;
        let t5_tmp = t4a - t5a;
        let t7 = t7a + t6a;
        let t6_tmp = t7a - t6a;

        let t5 = ((t6_tmp * 2896 - t5_tmp * 2896) + 2048) >> 12;
        let t6 = ((t6_tmp * 2896 + t5_tmp * 2896) + 2048) >> 12;

        [
            even_out[0] + t7,
            even_out[1] + t6,
            even_out[2] + t5,
            even_out[3] + t4,
            even_out[3] - t4,
            even_out[2] - t5,
            even_out[1] - t6,
            even_out[0] - t7,
        ]
    }

    fn scalar_dct16_1d(input: &[i32; 16]) -> [i32; 16] {
        let even: [i32; 8] = [
            input[0], input[2], input[4], input[6], input[8], input[10], input[12], input[14],
        ];
        let even_out = scalar_dct8_1d(&even);

        // Odd-indexed inputs
        let t8a = ((input[1] * 401 - input[15] * 4076) + 2048) >> 12;
        let t15a = ((input[1] * 4076 + input[15] * 401) + 2048) >> 12;
        let t9a = ((input[9] * 3166 - input[7] * 2598) + 2048) >> 12;
        let t14a = ((input[9] * 2598 + input[7] * 3166) + 2048) >> 12;
        let t10a = ((input[5] * 1931 - input[11] * 3612) + 2048) >> 12;
        let t13a = ((input[5] * 3612 + input[11] * 1931) + 2048) >> 12;
        let t11a = ((input[13] * 3920 - input[3] * 1189) + 2048) >> 12;
        let t12a = ((input[13] * 1189 + input[3] * 3920) + 2048) >> 12;

        // Butterfly stage 2
        let t9 = t8a - t9a;
        let t8 = t8a + t9a;
        let t14 = t15a - t14a;
        let t15 = t15a + t14a;
        let t10 = t11a - t10a;
        let t11 = t11a + t10a;
        let t12 = t12a + t13a;
        let t13 = t12a - t13a;

        // Stage 3: rotations
        let t9a = ((t14 * 1567 - t9 * 3784) + 2048) >> 12;
        let t14a = ((t14 * 3784 + t9 * 1567) + 2048) >> 12;
        let t13a = ((t13 * 1567 - t10 * 3784) + 2048) >> 12;
        let t10a = -(((t13 * 3784 + t10 * 1567) + 2048) >> 12);

        // Stage 4
        let t11a = t8 - t11;
        let t8a = t8 + t11;
        let t12a = t15 - t12;
        let t15a = t15 + t12;
        let t9b = t9a + t10a;
        let t10b = t9a - t10a;
        let t13b = t14a - t13a;
        let t14b = t14a + t13a;

        // Stage 5
        let t11f = ((t12a * 2896 - t11a * 2896) + 2048) >> 12;
        let t12f = ((t12a * 2896 + t11a * 2896) + 2048) >> 12;
        let t10af = ((t13b * 2896 - t10b * 2896) + 2048) >> 12;
        let t13af = ((t13b * 2896 + t10b * 2896) + 2048) >> 12;

        [
            even_out[0] + t15a,
            even_out[1] + t14b,
            even_out[2] + t13af,
            even_out[3] + t12f,
            even_out[4] + t11f,
            even_out[5] + t10af,
            even_out[6] + t9b,
            even_out[7] + t8a,
            even_out[7] - t8a,
            even_out[6] - t9b,
            even_out[5] - t10af,
            even_out[4] - t11f,
            even_out[3] - t12f,
            even_out[2] - t13af,
            even_out[1] - t14b,
            even_out[0] - t15a,
        ]
    }

    /// Full scalar reference for DCT_DCT 16x16 8bpc
    fn scalar_dct_dct_16x16(dst: &mut [u8], stride: isize, coeff: &mut [i16]) {
        let mut tmp = [0i32; 256];

        // Row transform
        for y in 0..16 {
            let mut input = [0i32; 16];
            for x in 0..16 {
                input[x] = coeff[y + x * 16] as i32;
            }
            let out = scalar_dct16_1d(&input);
            for x in 0..16 {
                tmp[y * 16 + x] = (out[x] + 2) >> 2; // shift=2
            }
        }

        // Column transform
        for x in 0..16 {
            let mut input = [0i32; 16];
            for y in 0..16 {
                input[y] = tmp[y * 16 + x];
            }
            let out = scalar_dct16_1d(&input);

            for y in 0..16 {
                let row_off = (y as isize * stride) as usize;
                let d = dst[row_off + x] as i32;
                let c = (out[y] + 8) >> 4;
                dst[row_off + x] = iclip(d + c, 0, 255) as u8;
            }
        }

        coeff[0..256].fill(0);
    }

    /// Full scalar reference for IDENTITY_IDENTITY 16x16 8bpc
    fn scalar_identity_identity_16x16(dst: &mut [u8], stride: isize, coeff: &mut [i16]) {
        // For 16x16, the identity transform is: x * 2*sqrt(2) for each dimension
        // Row identity: x * 2*sqrt(2), then >>2, column identity: x * 2*sqrt(2), then >>4
        // The assembly uses: sqrdmulh + sshr>>1 + srhadd for the row (shift2 variant)
        // and the standard identity for the column.
        //
        // For test purposes, use the simple scaling: each coefficient scaled by
        // 2*sqrt(2) * 2*sqrt(2) / (4 * 16) = 8 / 64 = 1/8
        // More precisely: scale = (5793/4096)^2 * 2^(-2-4) = 5793^2/4096^2/64
        //
        // Actually from the assembly:
        // Row: identity_8x16_shift2: sqrdmulh(vi, (5793-4096)*2*8) >> 1, then srhadd with original
        // This approximates vi * 5793/4096 = vi * sqrt(2)*2
        // Then srshr #2 (not applied for identity, the shift2 variant combines them)
        // Column: identity (sqrdmulh + sqadd*2 + sqadd)
        // Then srshr #4

        // Simplest scalar approach matching the generic scalar:
        let sqrt2x2 = 5793i32; // sqrt(2) * 4096
        for y in 0..16 {
            let row_off = (y as isize * stride) as usize;
            for x in 0..16 {
                let c = coeff[y + x * 16] as i32;
                // Row identity with shift2: ((c * 5793 + 2048) >> 12 + 1) >> 1
                // Actually identity_8x16_shift2 computes:
                //   t = sqrdmulh(c, 27152) = (c * 27152 + 16384) >> 15
                //   t = t >> 1  (sshr, not rounding)
                //   result = (c + t + 1) >> 1  (srhadd)
                let t = ((c as i64 * 27152 + 16384) >> 15) as i32;
                let t = t >> 1;
                let row_result = (c + t + 1) >> 1;

                // Column identity (standard):
                //   t2 = sqrdmulh(row_result, 27152) = (row_result * 27152 + 16384) >> 15
                //   result = row_result * 2 + t2
                let t2 = ((row_result as i64 * 27152 + 16384) >> 15) as i32;
                let col_result = row_result.saturating_mul(2).saturating_add(t2);

                // Final shift >>4
                let final_val = (col_result + 8) >> 4;
                let d = dst[row_off + x] as i32;
                dst[row_off + x] = iclip(d + final_val, 0, 255) as u8;
            }
        }

        coeff[0..256].fill(0);
    }

    #[test]
    fn test_dct_dct_16x16_neon_vs_scalar() {
        let token = archmage::Arm64::summon().expect("NEON must be available");

        // Test with specific coefficient patterns
        for pattern in 0..3 {
            let mut coeff_neon = [0i16; 256];
            let mut coeff_scalar = [0i16; 256];

            match pattern {
                0 => {
                    // DC only
                    coeff_neon[0] = 1000;
                    coeff_scalar[0] = 1000;
                }
                1 => {
                    // First few coefficients
                    for i in 0..16 {
                        coeff_neon[i] = ((i as i16 + 1) * 100) % 2000 - 1000;
                        coeff_scalar[i] = coeff_neon[i];
                    }
                }
                2 => {
                    // Random-ish pattern
                    for i in 0..256 {
                        let val = ((i * 37 + 13) % 2001) as i16 - 1000;
                        coeff_neon[i] = val;
                        coeff_scalar[i] = val;
                    }
                }
                _ => unreachable!(),
            }

            let stride = 16isize;
            let mut dst_neon = [128u8; 16 * 16];
            let mut dst_scalar = [128u8; 16 * 16];

            inv_txfm_add_16x16_8bpc_neon(
                token,
                &mut dst_neon,
                0,
                stride,
                &mut coeff_neon,
                255,
                255,
                TxType16::Dct,
                TxType16::Dct,
            );

            scalar_dct_dct_16x16(&mut dst_scalar, stride, &mut coeff_scalar);

            // Allow up to MAX_DIFF per pixel — NEON matches dav1d assembly,
            // scalar Rust has known rounding differences
            let mut max_diff_seen = 0i32;
            for i in 0..256 {
                let diff = (dst_neon[i] as i32 - dst_scalar[i] as i32).abs();
                max_diff_seen = max_diff_seen.max(diff);
            }

            assert!(
                max_diff_seen <= MAX_DIFF,
                "DCT_DCT 16x16 pattern {pattern}: max diff = {max_diff_seen} (expected <= {MAX_DIFF})"
            );
        }
    }

    #[test]
    fn test_identity_identity_16x16_neon_vs_scalar() {
        let token = archmage::Arm64::summon().expect("NEON must be available");

        for pattern in 0..3 {
            let mut coeff_neon = [0i16; 256];
            let mut coeff_scalar = [0i16; 256];

            match pattern {
                0 => {
                    coeff_neon[0] = 500;
                    coeff_scalar[0] = 500;
                }
                1 => {
                    for i in 0..32 {
                        coeff_neon[i] = ((i as i16 + 1) * 50) % 1000 - 500;
                        coeff_scalar[i] = coeff_neon[i];
                    }
                }
                2 => {
                    for i in 0..256 {
                        let val = ((i * 31 + 7) % 1001) as i16 - 500;
                        coeff_neon[i] = val;
                        coeff_scalar[i] = val;
                    }
                }
                _ => unreachable!(),
            }

            let stride = 16isize;
            let mut dst_neon = [128u8; 16 * 16];
            let mut dst_scalar = [128u8; 16 * 16];

            inv_txfm_add_16x16_8bpc_neon(
                token,
                &mut dst_neon,
                0,
                stride,
                &mut coeff_neon,
                255,
                255,
                TxType16::Identity,
                TxType16::Identity,
            );

            scalar_identity_identity_16x16(&mut dst_scalar, stride, &mut coeff_scalar);

            let mut max_diff_seen = 0i32;
            for i in 0..256 {
                let diff = (dst_neon[i] as i32 - dst_scalar[i] as i32).abs();
                max_diff_seen = max_diff_seen.max(diff);
            }

            assert!(
                max_diff_seen <= MAX_DIFF,
                "IDENTITY_IDENTITY 16x16 pattern {pattern}: max diff = {max_diff_seen} (expected <= {MAX_DIFF})"
            );
        }
    }

    #[test]
    fn test_dc_only_16x16() {
        let token = archmage::Arm64::summon().expect("NEON must be available");

        let mut coeff = [0i16; 256];
        coeff[0] = 1000;

        let stride = 16isize;
        let mut dst = [128u8; 16 * 16];

        // DC-only path (eob=0)
        inv_txfm_add_16x16_8bpc_neon(
            token,
            &mut dst,
            0,
            stride,
            &mut coeff,
            0,
            255,
            TxType16::Dct,
            TxType16::Dct,
        );

        // Verify coefficient was cleared
        assert_eq!(coeff[0], 0);

        // All pixels should have the same value (DC uniform)
        let first = dst[0];
        for i in 1..256 {
            assert_eq!(dst[i], first, "DC output not uniform at pixel {i}");
        }
    }
}
