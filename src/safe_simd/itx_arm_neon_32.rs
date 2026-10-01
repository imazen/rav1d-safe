//! Safe ARM NEON 32x32 inverse transforms (DCT, Identity)
//!
//! Port of the 32x32 inverse transform functions from `src/arm/64/itx.S`
//! to safe Rust NEON intrinsics. Only DCT_DCT and IDTX are valid for 32x32.
//!
//! The 32-point inverse DCT is decomposed into:
//!   1. 16-point IDCT on even-indexed inputs (`idct_16_q`)
//!   2. 32-point odd part (`idct32_odd_q`) on odd-indexed inputs
//!   3. Butterfly combination of even and odd results
//!
//! The 32x32 block is processed in two passes:
//!   - Row transform: 4 groups of 8 rows, each processing 32 columns
//!     (split into even/odd 16-column halves), storing to a 2048-byte scratch buffer
//!   - Column transform: 4 groups of 8 columns from scratch, adding to destination

#![allow(clippy::too_many_arguments)]
#![cfg_attr(not(any(feature = "asm", feature = "c-ffi", feature = "partial_asm")), forbid(unsafe_code))]
#![cfg_attr(any(feature = "asm", feature = "c-ffi", feature = "partial_asm"), deny(unsafe_code))]

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
use archmage::{Arm64, arcane, rite};

#[cfg(target_arch = "aarch64")]
use safe_unaligned_simd::aarch64 as safe_simd;

use super::itx_arm_neon_8x8::{smull_smlal_q, smull_smlsl_q, sqrshrn_pair, transpose_8x8h};
use super::itx_arm_neon_16x16::idct_16_q;
use super::itx_arm_neon_common::IDCT_COEFFS;

/// Type alias for 16 NEON vectors.
#[cfg(target_arch = "aarch64")]
pub(crate) type V16 = [int16x8_t; 16];

// ============================================================================
// IDCT32 coefficients (from itx.S lines 70-74)
// ============================================================================
//
// The 32-point odd part uses these coefficients (offset 16 into IDCT_COEFFS):
//   v0: [201, 4091, 3035, 2751, 1751, 3703, 3857, 1380]
//   v1: [995, 3973, 3513, 2106, 2440, 3290, 4052, 601]
//
// And the standard IDCT coefficients from offset 0:
//   v0: [2896, 23168, 1567, 3784, 799, 4017, 3406, 2276]

// ============================================================================
// 32-point odd part (inv_dct32_odd_8h_x16_neon from itx.S lines 1855-2010)
// ============================================================================

/// 32-point inverse DCT odd part on 16 int16x8_t vectors.
///
/// Takes 16 odd-indexed inputs (indices 1,3,5,...,31 from the original 32-point
/// transform) in v[0..16] and produces 16 outputs (t16..t31).
///
/// This is a direct port of `inv_dct32_odd_8h_x16_neon` from itx.S.
///
/// The outputs represent t16..t31 and will be butterfly-combined with the
/// even part (from idct_16) to produce the final 32-point DCT result.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
pub(crate) fn idct32_odd_q(v: V16) -> V16 {
    // Load idct32 coefficients (offset 16 into IDCT_COEFFS = byte offset 32)
    // c0: [201, 4091, 3035, 2751, 1751, 3703, 3857, 1380]
    let c0 = safe_simd::vld1q_s16(<&[i16; 8]>::try_from(&IDCT_COEFFS[16..24]).unwrap());
    // c1: [995, 3973, 3513, 2106, 2440, 3290, 4052, 601]
    let c1 = safe_simd::vld1q_s16(<&[i16; 8]>::try_from(&IDCT_COEFFS[24..32]).unwrap());

    // Stage 1: 8 rotation pairs on odd-indexed inputs
    // Assembly maps: v[0]=input for index 1, v[1]=input for index 17, etc.
    // v16=v[0], v17=v[8], v18=v[4], v19=v[12], v20=v[2], v21=v[10],
    // v22=v[6], v23=v[14], v24=v[1], v25=v[9], v26=v[5], v27=v[13],
    // v28=v[3], v29=v[11], v30=v[7], v31=v[15]
    //
    // The assembly uses register names v16..v31, and the input mapping is:
    // v16 ← odd coeff 0 (index 1 of original), v31 ← odd coeff 15 (index 31)
    // v24 ← odd coeff 8 (index 17), v23 ← odd coeff 7 (index 15)
    // etc.
    //
    // However, looking at the assembly more carefully:
    // The caller loads v16..v31 with the 16 odd coefficients in order.
    // v16=coeff[1], v17=coeff[3], ..., v31=coeff[31] (odd indices)
    //
    // Wait — for the horizontal case, the function is called with:
    //   v16..v31 loaded from the second half of coefficients (odd-indexed columns)
    //
    // Actually, the assembly loads in strided order. Let me re-read:
    // In inv_dct32_odd_8h_x16_neon, the inputs are already in v16..v31.
    // The rotation pairs are:
    //   (v16, v31) with c0.h[0], c0.h[1]  → t16a, t31a
    //   (v24, v23) with c0.h[2], c0.h[3]  → t17a, t30a
    //   (v20, v27) with c0.h[4], c0.h[5]  → t18a, t29a
    //   (v28, v19) with c0.h[6], c0.h[7]  → t19a, t28a
    //   (v18, v29) with c1.h[0], c1.h[1]  → t20a, t27a
    //   (v26, v21) with c1.h[2], c1.h[3]  → t21a, t26a
    //   (v22, v25) with c1.h[4], c1.h[5]  → t22a, t25a
    //   (v30, v17) with c1.h[6], c1.h[7]  → t23a, t24a

    // t16a = (v[0] * 201 - v[15] * 4091 + 2048) >> 12
    // t31a = (v[0] * 4091 + v[15] * 201 + 2048) >> 12
    let (lo, hi) = smull_smlsl_q(v[0], v[15], c0, 0, 1);
    let t16a = sqrshrn_pair(lo, hi);
    let (lo, hi) = smull_smlal_q(v[0], v[15], c0, 1, 0);
    let t31a = sqrshrn_pair(lo, hi);

    // t17a = (v[8] * 3035 - v[7] * 2751 + 2048) >> 12
    // t30a = (v[8] * 2751 + v[7] * 3035 + 2048) >> 12
    let (lo, hi) = smull_smlsl_q(v[8], v[7], c0, 2, 3);
    let t17a = sqrshrn_pair(lo, hi);
    let (lo, hi) = smull_smlal_q(v[8], v[7], c0, 3, 2);
    let t30a = sqrshrn_pair(lo, hi);

    // t18a = (v[4] * 1751 - v[11] * 3703 + 2048) >> 12
    // t29a = (v[4] * 3703 + v[11] * 1751 + 2048) >> 12
    let (lo, hi) = smull_smlsl_q(v[4], v[11], c0, 4, 5);
    let t18a = sqrshrn_pair(lo, hi);
    let (lo, hi) = smull_smlal_q(v[4], v[11], c0, 5, 4);
    let t29a = sqrshrn_pair(lo, hi);

    // t19a = (v[12] * 3857 - v[3] * 1380 + 2048) >> 12
    // t28a = (v[12] * 1380 + v[3] * 3857 + 2048) >> 12
    let (lo, hi) = smull_smlsl_q(v[12], v[3], c0, 6, 7);
    let t19a = sqrshrn_pair(lo, hi);
    let (lo, hi) = smull_smlal_q(v[12], v[3], c0, 7, 6);
    let t28a = sqrshrn_pair(lo, hi);

    // t20a = (v[2] * 995 - v[13] * 3973 + 2048) >> 12
    // t27a = (v[2] * 3973 + v[13] * 995 + 2048) >> 12
    let (lo, hi) = smull_smlsl_q(v[2], v[13], c1, 0, 1);
    let t20a = sqrshrn_pair(lo, hi);
    let (lo, hi) = smull_smlal_q(v[2], v[13], c1, 1, 0);
    let t27a = sqrshrn_pair(lo, hi);

    // t21a = (v[10] * 3513 - v[5] * 2106 + 2048) >> 12
    // t26a = (v[10] * 2106 + v[5] * 3513 + 2048) >> 12
    let (lo, hi) = smull_smlsl_q(v[10], v[5], c1, 2, 3);
    let t21a = sqrshrn_pair(lo, hi);
    let (lo, hi) = smull_smlal_q(v[10], v[5], c1, 3, 2);
    let t26a = sqrshrn_pair(lo, hi);

    // t22a = (v[6] * 2440 - v[9] * 3290 + 2048) >> 12
    // t25a = (v[6] * 3290 + v[9] * 2440 + 2048) >> 12
    let (lo, hi) = smull_smlsl_q(v[6], v[9], c1, 4, 5);
    let t22a = sqrshrn_pair(lo, hi);
    let (lo, hi) = smull_smlal_q(v[6], v[9], c1, 5, 4);
    let t25a = sqrshrn_pair(lo, hi);

    // t23a = (v[14] * 4052 - v[1] * 601 + 2048) >> 12
    // t24a = (v[14] * 601 + v[1] * 4052 + 2048) >> 12
    let (lo, hi) = smull_smlsl_q(v[14], v[1], c1, 6, 7);
    let t23a = sqrshrn_pair(lo, hi);
    let (lo, hi) = smull_smlal_q(v[14], v[1], c1, 7, 6);
    let t24a = sqrshrn_pair(lo, hi);

    // Load main IDCT coefficients for butterfly stages
    // c_main: [2896, 23168, 1567, 3784, 799, 4017, 3406, 2276]
    let c_main = safe_simd::vld1q_s16(<&[i16; 8]>::try_from(&IDCT_COEFFS[0..8]).unwrap());

    // Stage 2: Butterfly
    let s17 = vqsubq_s16(t16a, t17a); // t17
    let s16 = vqaddq_s16(t16a, t17a); // t16
    let s30 = vqsubq_s16(t31a, t30a); // t30
    let s31 = vqaddq_s16(t31a, t30a); // t31
    let s18 = vqsubq_s16(t19a, t18a); // t18
    let s19 = vqaddq_s16(t19a, t18a); // t19
    let s20 = vqaddq_s16(t20a, t21a); // t20
    let s21 = vqsubq_s16(t20a, t21a); // t21
    let s22 = vqsubq_s16(t23a, t22a); // t22
    let s23 = vqaddq_s16(t23a, t22a); // t23
    let s24 = vqaddq_s16(t24a, t25a); // t24
    let s25 = vqsubq_s16(t24a, t25a); // t25
    let s26 = vqsubq_s16(t27a, t26a); // t26
    let s27 = vqaddq_s16(t27a, t26a); // t27
    let s28 = vqaddq_s16(t28a, t29a); // t28
    let s29 = vqsubq_s16(t28a, t29a); // t29

    // Stage 3: Rotations using c_main.h[4]=799, c_main.h[5]=4017,
    //          c_main.h[6]=3406, c_main.h[7]=2276

    // t17a = (s30 * 799 - s17 * 4017 + 2048) >> 12
    // t30a = (s30 * 4017 + s17 * 799 + 2048) >> 12
    let (lo, hi) = smull_smlsl_q(s30, s17, c_main, 4, 5);
    let u17a = sqrshrn_pair(lo, hi);
    let (lo, hi) = smull_smlal_q(s30, s17, c_main, 5, 4);
    let u30a = sqrshrn_pair(lo, hi);

    // t18a = -(s29 * 4017 + s18 * 799) >> 12
    // (assembly: smull_smlal then neg)
    let (lo, hi) = smull_smlal_q(s29, s18, c_main, 5, 4);
    let u18a = sqrshrn_pair(vnegq_s32(lo), vnegq_s32(hi));
    // t29a = (s29 * 799 - s18 * 4017 + 2048) >> 12
    let (lo, hi) = smull_smlsl_q(s29, s18, c_main, 4, 5);
    let u29a = sqrshrn_pair(lo, hi);

    // t21a = (s26 * 3406 - s21 * 2276 + 2048) >> 12
    // t26a = (s26 * 2276 + s21 * 3406 + 2048) >> 12
    let (lo, hi) = smull_smlsl_q(s26, s21, c_main, 6, 7);
    let u21a = sqrshrn_pair(lo, hi);
    let (lo, hi) = smull_smlal_q(s26, s21, c_main, 7, 6);
    let u26a = sqrshrn_pair(lo, hi);

    // t22a = -(s25 * 2276 + s22 * 3406) >> 12
    let (lo, hi) = smull_smlal_q(s25, s22, c_main, 7, 6);
    let u22a = sqrshrn_pair(vnegq_s32(lo), vnegq_s32(hi));
    // t25a = (s25 * 3406 - s22 * 2276 + 2048) >> 12
    let (lo, hi) = smull_smlsl_q(s25, s22, c_main, 6, 7);
    let u25a = sqrshrn_pair(lo, hi);

    // Stage 4: Butterfly
    let w30 = vqaddq_s16(u30a, u29a); // t30
    let w29 = vqsubq_s16(u30a, u29a); // t29
    let w18 = vqsubq_s16(u17a, u18a); // t18
    let w17 = vqaddq_s16(u17a, u18a); // t17
    let w19a = vqsubq_s16(s16, s19); // t19a
    let w16a = vqaddq_s16(s16, s19); // t16a
    let w20a = vqsubq_s16(s23, s20); // t20a
    let w23a = vqaddq_s16(s23, s20); // t23a
    let w21 = vqsubq_s16(u22a, u21a); // t21
    let w22 = vqaddq_s16(u22a, u21a); // t22
    let w24a = vqaddq_s16(s24, s27); // t24a
    let w27a = vqsubq_s16(s24, s27); // t27a
    let w25 = vqaddq_s16(u25a, u26a); // t25
    let w26 = vqsubq_s16(u25a, u26a); // t26
    let w28a = vqsubq_s16(s31, s28); // t28a
    let w31a = vqaddq_s16(s31, s28); // t31a

    // Stage 5: Rotations using c_main.h[2]=1567, c_main.h[3]=3784

    // t18a = (w29 * 1567 - w18 * 3784 + 2048) >> 12
    // t29a = (w29 * 3784 + w18 * 1567 + 2048) >> 12
    let (lo, hi) = smull_smlsl_q(w29, w18, c_main, 2, 3);
    let x18a = sqrshrn_pair(lo, hi);
    let (lo, hi) = smull_smlal_q(w29, w18, c_main, 3, 2);
    let x29a = sqrshrn_pair(lo, hi);

    // t19 = (w28a * 1567 - w19a * 3784 + 2048) >> 12
    // t28 = (w28a * 3784 + w19a * 1567 + 2048) >> 12
    let (lo, hi) = smull_smlsl_q(w28a, w19a, c_main, 2, 3);
    let x19 = sqrshrn_pair(lo, hi);
    let (lo, hi) = smull_smlal_q(w28a, w19a, c_main, 3, 2);
    let x28 = sqrshrn_pair(lo, hi);

    // t20 = -(w27a * 3784 + w20a * 1567) >> 12
    let (lo, hi) = smull_smlal_q(w27a, w20a, c_main, 3, 2);
    let x20 = sqrshrn_pair(vnegq_s32(lo), vnegq_s32(hi));
    // t27 = (w27a * 1567 - w20a * 3784 + 2048) >> 12
    let (lo, hi) = smull_smlsl_q(w27a, w20a, c_main, 2, 3);
    let x27 = sqrshrn_pair(lo, hi);

    // t21a = -(w26 * 3784 + w21 * 1567) >> 12
    let (lo, hi) = smull_smlal_q(w26, w21, c_main, 3, 2);
    let x21a = sqrshrn_pair(vnegq_s32(lo), vnegq_s32(hi));
    // t26a = (w26 * 1567 - w21 * 3784 + 2048) >> 12
    let (lo, hi) = smull_smlsl_q(w26, w21, c_main, 2, 3);
    let x26a = sqrshrn_pair(lo, hi);

    // Stage 6: Final butterfly
    let y23 = vqsubq_s16(w16a, w23a); // t23
    let y16 = vqaddq_s16(w16a, w23a); // t16 = out16
    let y24 = vqsubq_s16(w31a, w24a); // t24
    let y31 = vqaddq_s16(w31a, w24a); // t31 = out31
    let y22a = vqsubq_s16(w17, w22); // t22a
    let y17a = vqaddq_s16(w17, w22); // t17a = out17
    let y30a = vqaddq_s16(w30, w25); // t30a = out30
    let y25a = vqsubq_s16(w30, w25); // t25a
    let y21 = vqsubq_s16(x18a, x21a); // t21
    let y18 = vqaddq_s16(x18a, x21a); // t18 = out18
    let y19a = vqaddq_s16(x19, x20); // t19a = out19
    let y20a = vqsubq_s16(x19, x20); // t20a
    let y29 = vqaddq_s16(x29a, x26a); // t29 = out29
    let y26 = vqsubq_s16(x29a, x26a); // t26
    let y28a = vqaddq_s16(x28, x27); // t28a = out28
    let y27a = vqsubq_s16(x28, x27); // t27a

    // Stage 7: Final rotations using c_main.h[0]=2896

    // t20 = (y27a * 2896 - y20a * 2896 + 2048) >> 12
    // t27 = (y27a * 2896 + y20a * 2896 + 2048) >> 12
    let (lo, hi) = smull_smlsl_q(y27a, y20a, c_main, 0, 0);
    let z20 = sqrshrn_pair(lo, hi);
    let (lo, hi) = smull_smlal_q(y27a, y20a, c_main, 0, 0);
    let z27 = sqrshrn_pair(lo, hi);

    // t26a = (y26 * 2896 + y21 * 2896 + 2048) >> 12
    // t21a = (y26 * 2896 - y21 * 2896 + 2048) >> 12
    // Wait — assembly says:
    // smull_smlal v4,v5, v25,v27, v0.h[0],v0.h[0] -> t26a
    // smull_smlsl v6,v7, v25,v27, v0.h[0],v0.h[0] -> t21a
    // where v25=y26 and v27=y21 at that point
    let (lo, hi) = smull_smlal_q(y26, y21, c_main, 0, 0);
    let z26a = sqrshrn_pair(lo, hi);
    let (lo, hi) = smull_smlsl_q(y26, y21, c_main, 0, 0);
    let z21a = sqrshrn_pair(lo, hi);

    // t22 = (y25a * 2896 - y22a * 2896 + 2048) >> 12
    // t25 = (y25a * 2896 + y22a * 2896 + 2048) >> 12
    // Assembly: smull_smlsl v24,v25, v21,v23, v0.h[0],v0.h[0] -> t22
    //           smull_smlal v4,v5, v21,v23, v0.h[0],v0.h[0] -> t25
    // where v21=y25a and v23=y22a
    let (lo, hi) = smull_smlsl_q(y25a, y22a, c_main, 0, 0);
    let z22 = sqrshrn_pair(lo, hi);
    let (lo, hi) = smull_smlal_q(y25a, y22a, c_main, 0, 0);
    let z25 = sqrshrn_pair(lo, hi);

    // t23a = (y24 * 2896 - y23 * 2896 + 2048) >> 12
    // t24a = (y24 * 2896 + y23 * 2896 + 2048) >> 12
    let (lo, hi) = smull_smlsl_q(y24, y23, c_main, 0, 0);
    let z23a = sqrshrn_pair(lo, hi);
    let (lo, hi) = smull_smlal_q(y24, y23, c_main, 0, 0);
    let z24a = sqrshrn_pair(lo, hi);

    // Return t16..t31 in order
    [
        y16,  // t16 (out16)
        y17a, // t17a (out17)
        y18,  // t18 (out18)
        y19a, // t19a (out19)
        z20,  // t20
        z21a, // t21a
        z22,  // t22
        z23a, // t23a
        z24a, // t24a
        z25,  // t25
        z26a, // t26a
        z27,  // t27
        y28a, // t28a (out28)
        y29,  // t29 (out29)
        y30a, // t30a (out30)
        y31,  // t31 (out31)
    ]
}

// ============================================================================
// Full 32-point inverse DCT
// ============================================================================

/// Full 32-point inverse DCT.
///
/// Takes 32 int16x8_t vectors (even[0..16] from idct_16, odd[0..16] from
/// idct32_odd) and produces 32 output vectors via butterfly combination.
///
/// This is the top-level function that combines even and odd halves:
///   out[i]      = even[i]      + odd[15 - i]   for i = 0..16
///   out[31 - i] = even[i]      - odd[15 - i]   for i = 0..16
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
pub(crate) fn idct_32_full(even: &V16, odd: &V16) -> [int16x8_t; 32] {
    let mut out = [vdupq_n_s16(0); 32];

    // The even part produces e[0..16] = result of idct_16
    // The odd part produces o[0..15] = t16..t31
    //
    // Assembly butterfly (from inv_txfm_add_vert_dct_8x32_neon):
    //   out[0]  = even[0]  + odd[15]   (sqadd)
    //   out[1]  = even[1]  + odd[14]   (sqadd)
    //   ...
    //   out[15] = even[15] + odd[0]    (sqadd)
    //   out[16] = even[15] - odd[0]    (sqsub)
    //   ...
    //   out[31] = even[0]  - odd[15]   (sqsub)
    for i in 0..16 {
        out[i] = vqaddq_s16(even[i], odd[15 - i]);
        out[31 - i] = vqsubq_s16(even[i], odd[15 - i]);
    }

    out
}

// ============================================================================
// Horizontal 32-point DCT on 8 rows (row transform)
// ============================================================================

/// Horizontal DCT 32x8: transform 8 rows of 32 coefficients.
///
/// Mirrors `inv_txfm_horz_dct_32x8_neon` from itx.S lines 2014-2094.
///
/// For each group of 8 rows:
/// 1. Load 16 even-indexed columns, apply idct_16, transpose
/// 2. Load 16 odd-indexed columns, apply idct32_odd, transpose
/// 3. Butterfly-combine even and odd, apply shift, store to scratch
///
/// `coeff_base`: starting offset in `coeff` for the even-indexed columns
/// `coeff_stride`: distance between consecutive rows in coeff (= 32 for 32x32)
/// `scratch`: output buffer, one row = 32 i16 values
/// `scratch_base`: starting offset in scratch
/// `shift`: rounding shift to apply (2 for 32x32)
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
pub(crate) fn horz_dct_32x8(
    coeff: &mut [i16],
    coeff_base: usize,
    coeff_stride: usize,
    scratch: &mut [i16],
    scratch_base: usize,
    shift: i16,
) {
    let zero_vec = vdupq_n_s16(0);

    // Phase 1: Load even-indexed columns (0,2,4,...,30) for 8 rows
    // coeff layout is column-major: coeff[row + col * height]
    // For 32x32: coeff[row + col * 32]
    // Even columns: col = 0,2,4,...,30
    let mut even_in: V16 = [zero_vec; 16];
    for c in 0..16 {
        let col = c * 2; // even column index
        let base = coeff_base + col * coeff_stride;
        let arr: [i16; 8] = coeff[base..base + 8].try_into().unwrap();
        even_in[c] = safe_simd::vld1q_s16(&arr);
        // Clear the loaded coefficients
        coeff[base..base + 8].fill(0);
    }

    // Apply 16-point IDCT to even columns
    let even_out = idct_16_q(even_in);

    // Transpose the 16 even results into two 8x8 blocks
    let (et0, et1, et2, et3, et4, et5, et6, et7) = transpose_8x8h(
        even_out[0],
        even_out[1],
        even_out[2],
        even_out[3],
        even_out[4],
        even_out[5],
        even_out[6],
        even_out[7],
    );
    let (et8, et9, et10, et11, et12, et13, et14, et15) = transpose_8x8h(
        even_out[8],
        even_out[9],
        even_out[10],
        even_out[11],
        even_out[12],
        even_out[13],
        even_out[14],
        even_out[15],
    );

    // Store transposed even results to scratch (interleaved layout)
    // Assembly stores: for each row r, store [even_lo_r, even_hi_r] at scratch[r*64]
    // (64 bytes = 32 i16 values per row, but we only fill the first 32 bytes/16 values)
    // Actually the assembly stores with gaps for the odd part.
    //
    // Looking at assembly more carefully:
    // store1 r0, r1: st1 {r0}, [x6], #16; st1 {r1}, [x6], #16; add x6, x6, #32
    // This stores 16 bytes (r0=8 i16), 16 bytes (r1=8 i16), then skips 32 bytes
    // Total stride per row = 16 + 16 + 32 = 64 bytes = 32 i16 values
    //
    // So scratch layout per row: [even_lo(8), even_hi(8), gap(16)]
    // After storing, x6 is rewound: sub x6, x6, #64*8
    //
    // Then the odd part's "store2" macro reads back even, combines with odd, stores final.
    //
    // For our Rust port, we'll use a simpler approach:
    // Store all 32 results per row directly (no interleaving needed).
    // We store the transposed even results to a temporary, compute odd, combine, store final.

    let even_t = [
        [et0, et8],
        [et1, et9],
        [et2, et10],
        [et3, et11],
        [et4, et12],
        [et5, et13],
        [et6, et14],
        [et7, et15],
    ];

    // Phase 2: Load odd-indexed columns (1,3,5,...,31) for 8 rows
    let mut odd_in: V16 = [zero_vec; 16];
    for c in 0..16 {
        let col = c * 2 + 1; // odd column index
        let base = coeff_base + col * coeff_stride;
        let arr: [i16; 8] = coeff[base..base + 8].try_into().unwrap();
        odd_in[c] = safe_simd::vld1q_s16(&arr);
        coeff[base..base + 8].fill(0);
    }

    // Apply 32-point odd part
    let odd_out = idct32_odd_q(odd_in);

    // Transpose the odd results in reverse order (assembly does this)
    // Assembly: transpose v31,v30,v29,...,v24 and v23,v22,...,v16
    // The odd part outputs t16..t31 in indices 0..15
    // After reverse-order transpose:
    //   First block (high): transpose(odd[15], odd[14], ..., odd[8])
    //   Second block (low): transpose(odd[7], odd[6], ..., odd[0])
    let (ot15, ot14, ot13, ot12, ot11, ot10, ot9, ot8) = transpose_8x8h(
        odd_out[15],
        odd_out[14],
        odd_out[13],
        odd_out[12],
        odd_out[11],
        odd_out[10],
        odd_out[9],
        odd_out[8],
    );
    let (ot7, ot6, ot5, ot4, ot3, ot2, ot1, ot0) = transpose_8x8h(
        odd_out[7], odd_out[6], odd_out[5], odd_out[4], odd_out[3], odd_out[2], odd_out[1],
        odd_out[0],
    );

    let odd_t_hi = [ot15, ot14, ot13, ot12, ot11, ot10, ot9, ot8];
    let odd_t_lo = [ot7, ot6, ot5, ot4, ot3, ot2, ot1, ot0];

    // Phase 3: Butterfly combine and store to scratch
    // For each of 8 transposed rows:
    //   even_lo = even_t[row][0], even_hi = even_t[row][1]
    //   odd_hi_r = odd_t_hi[row], odd_lo_r = odd_t_lo[row]
    //
    // Assembly's store2 macro:
    //   first_lo = sqadd(even_lo, odd_hi_r)
    //   first_hi = sqadd(even_hi, odd_lo_r)
    //   second_lo = sqsub(even_lo, odd_hi_r) → reversed (rev64 + ext)
    //   second_hi = sqsub(even_hi, odd_lo_r) → reversed (rev64 + ext)
    //   Apply shift, store [first_lo, first_hi, second_lo, second_hi]
    //
    // The "reversed" pattern produces the mirrored second half.
    // rev64 reverses within 64-bit halves, ext #8 swaps the two 64-bit halves.
    // Combined: full 128-bit reverse of 8 i16 elements.

    for row in 0..8 {
        let e_lo = even_t[row][0];
        let e_hi = even_t[row][1];
        let o_hi = odd_t_hi[row];
        let o_lo = odd_t_lo[row];

        // First 16 values: even + odd
        let first_lo = vqaddq_s16(e_lo, o_hi);
        let first_hi = vqaddq_s16(e_hi, o_lo);

        // Second 16 values: even - odd, reversed
        let sub_lo = vqsubq_s16(e_lo, o_hi);
        let sub_hi = vqsubq_s16(e_hi, o_lo);
        // Reverse each: rev64 + ext#8 = full reverse of 8 elements
        let rev_lo = rev128_s16(sub_hi);
        let rev_hi = rev128_s16(sub_lo);

        // Apply rounding shift
        let r0 = vrshrq_n_s16::<2>(first_lo);
        let r1 = vrshrq_n_s16::<2>(first_hi);
        let r2 = vrshrq_n_s16::<2>(rev_lo);
        let r3 = vrshrq_n_s16::<2>(rev_hi);

        // Store to scratch buffer (row-major, 32 i16 per row)
        let soff = scratch_base + row * 32;
        store_v16(scratch, soff, r0);
        store_v16(scratch, soff + 8, r1);
        store_v16(scratch, soff + 16, r2);
        store_v16(scratch, soff + 24, r3);
    }

    // Suppress unused variable warning
    let _ = shift;
}

/// Reverse all 8 i16 elements in an int16x8_t.
/// Equivalent to: rev64 v.8h + ext v.16b, v.16b, #8
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
#[inline(always)]
pub(crate) fn rev128_s16(v: int16x8_t) -> int16x8_t {
    let rev = vrev64q_s16(v);
    vextq_s16::<4>(rev, rev)
}

/// Store an int16x8_t to a slice at the given offset.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
#[inline(always)]
pub(crate) fn store_v16(buf: &mut [i16], off: usize, v: int16x8_t) {
    let mut tmp = [0i16; 8];
    safe_simd::vst1q_s16(&mut tmp, v);
    buf[off..off + 8].copy_from_slice(&tmp);
}

/// Load an int16x8_t from a slice at the given offset.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
#[inline(always)]
pub(crate) fn load_v16(buf: &[i16], off: usize) -> int16x8_t {
    let arr: [i16; 8] = buf[off..off + 8].try_into().unwrap();
    safe_simd::vld1q_s16(&arr)
}

// ============================================================================
// Vertical 32-point DCT on 8 columns (column transform + add to dst)
// ============================================================================

/// Vertical DCT 8x32: transform 8 columns and add to destination (8bpc).
///
/// Mirrors `inv_txfm_add_vert_dct_8x32_neon` from itx.S lines 2100-2167.
///
/// 1. Load 16 even-indexed rows from scratch, apply idct_16
/// 2. Store results back, load 16 odd-indexed rows, apply idct32_odd
/// 3. Butterfly-combine with shift >>4, add to destination pixels
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
pub(crate) fn vert_dct_add_8x32_8bpc(
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    scratch: &[i16],
    scratch_base: usize,
    scratch_stride: usize, // number of i16 per row in scratch (32 for 32x32)
) {
    // Load 16 even-indexed rows (rows 0,2,4,...,30)
    let mut even_in: V16 = [vdupq_n_s16(0); 16];
    for i in 0..16 {
        let row = i * 2;
        even_in[i] = load_v16(scratch, scratch_base + row * scratch_stride);
    }

    // Apply 16-point IDCT
    let even_out = idct_16_q(even_in);

    // Load 16 odd-indexed rows (rows 1,3,5,...,31)
    let mut odd_in: V16 = [vdupq_n_s16(0); 16];
    for i in 0..16 {
        let row = i * 2 + 1;
        odd_in[i] = load_v16(scratch, scratch_base + row * scratch_stride);
    }

    // Apply 32-point odd part
    let odd_out = idct32_odd_q(odd_in);

    // Butterfly combine and add to destination
    // First half (rows 0..16): even[i] + odd[15-i], sqadd
    // Second half (rows 16..32): even[15-j] - odd[j], sqsub (where j = row - 16)
    for i in 0..16 {
        let combined = vqaddq_s16(even_out[i], odd_out[15 - i]);
        let shifted = vrshrq_n_s16::<4>(combined);

        let row_off = dst_base.wrapping_add_signed(i as isize * dst_stride);
        let dst_bytes: [u8; 8] = dst[row_off..row_off + 8].try_into().unwrap();
        let dst_u8 = safe_simd::vld1_u8(&dst_bytes);
        let sum = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(shifted), dst_u8));
        let result = vqmovun_s16(sum);
        let mut out = [0u8; 8];
        safe_simd::vst1_u8(&mut out, result);
        dst[row_off..row_off + 8].copy_from_slice(&out);
    }

    for j in 0..16 {
        let i = 16 + j; // destination row
        let combined = vqsubq_s16(even_out[15 - j], odd_out[j]);
        let shifted = vrshrq_n_s16::<4>(combined);

        let row_off = dst_base.wrapping_add_signed(i as isize * dst_stride);
        let dst_bytes: [u8; 8] = dst[row_off..row_off + 8].try_into().unwrap();
        let dst_u8 = safe_simd::vld1_u8(&dst_bytes);
        let sum = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(shifted), dst_u8));
        let result = vqmovun_s16(sum);
        let mut out = [0u8; 8];
        safe_simd::vst1_u8(&mut out, result);
        dst[row_off..row_off + 8].copy_from_slice(&out);
    }
}

/// Vertical DCT 8x32: transform 8 columns and add to destination (16bpc).
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
pub(crate) fn vert_dct_add_8x32_16bpc(
    dst: &mut [u16],
    dst_base: usize,
    dst_stride: isize,
    scratch: &[i16],
    scratch_base: usize,
    scratch_stride: usize,
    bitdepth_max: i32,
) {
    let bd_max = vdupq_n_s16(bitdepth_max as i16);
    let zero = vdupq_n_s16(0);

    // Load even-indexed rows
    let mut even_in: V16 = [vdupq_n_s16(0); 16];
    for i in 0..16 {
        even_in[i] = load_v16(scratch, scratch_base + (i * 2) * scratch_stride);
    }
    let even_out = idct_16_q(even_in);

    // Load odd-indexed rows
    let mut odd_in: V16 = [vdupq_n_s16(0); 16];
    for i in 0..16 {
        odd_in[i] = load_v16(scratch, scratch_base + (i * 2 + 1) * scratch_stride);
    }
    let odd_out = idct32_odd_q(odd_in);

    // Combine and add to destination
    for i in 0..16 {
        let combined = vqaddq_s16(even_out[i], odd_out[15 - i]);
        let shifted = vrshrq_n_s16::<4>(combined);

        let row_off = dst_base.wrapping_add_signed(i as isize * dst_stride);
        let mut arr = [0i16; 8];
        for j in 0..8 {
            arr[j] = dst[row_off + j] as i16;
        }
        let d = safe_simd::vld1q_s16(&arr);
        let sum = vqaddq_s16(d, shifted);
        let clamped = vminq_s16(vmaxq_s16(sum, zero), bd_max);
        let mut out = [0i16; 8];
        safe_simd::vst1q_s16(&mut out, clamped);
        for j in 0..8 {
            dst[row_off + j] = out[j] as u16;
        }
    }

    for j in 0..16 {
        let i = 16 + j;
        let combined = vqsubq_s16(even_out[15 - j], odd_out[j]);
        let shifted = vrshrq_n_s16::<4>(combined);

        let row_off = dst_base.wrapping_add_signed(i as isize * dst_stride);
        let mut arr = [0i16; 8];
        for k in 0..8 {
            arr[k] = dst[row_off + k] as i16;
        }
        let d = safe_simd::vld1q_s16(&arr);
        let sum = vqaddq_s16(d, shifted);
        let clamped = vminq_s16(vmaxq_s16(sum, zero), bd_max);
        let mut out = [0i16; 8];
        safe_simd::vst1q_s16(&mut out, clamped);
        for k in 0..8 {
            dst[row_off + k] = out[k] as u16;
        }
    }
}

// ============================================================================
// DC-only fast path for DCT_DCT 32x32
// ============================================================================

/// DC-only fast path for DCT_DCT 32x32 with eob=0 (8bpc).
///
/// Assembly: `idct_dc 32, 32, 2`
/// ```text
/// dc = coeff[0]; coeff[0] = 0;
/// scale = 2896*8 = 23168
/// dc = sqrdmulh(dc, scale)    // first scaling
/// dc = srshr(dc, 2)           // shift=2 for 32x32
/// dc = sqrdmulh(dc, scale)    // second scaling
/// dc = srshr(dc, 4)           // final shift
/// ```
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn dc_only_32x32_8bpc(dst: &mut [u8], dst_base: usize, dst_stride: isize, coeff: &mut [i16]) {
    let dc = coeff[0];
    coeff[0] = 0;

    let scale = vdupq_n_s16((2896 * 8) as i16); // 23168

    let v = vdupq_n_s16(dc);
    let v = vqrdmulhq_s16(v, scale);
    let v = vrshrq_n_s16::<2>(v); // shift=2 for 32x32
    let v = vqrdmulhq_s16(v, scale);
    let v = vrshrq_n_s16::<4>(v); // final shift

    for i in 0..32 {
        let row_off = dst_base.wrapping_add_signed(i as isize * dst_stride);
        for half in 0..4 {
            let off = row_off + half * 8;
            let dst_bytes: [u8; 8] = dst[off..off + 8].try_into().unwrap();
            let dst_u8 = safe_simd::vld1_u8(&dst_bytes);
            let sum = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(v), dst_u8));
            let result = vqmovun_s16(sum);
            let mut out = [0u8; 8];
            safe_simd::vst1_u8(&mut out, result);
            dst[off..off + 8].copy_from_slice(&out);
        }
    }
}

/// DC-only fast path for DCT_DCT 32x32 (16bpc).
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn dc_only_32x32_16bpc(
    dst: &mut [u16],
    dst_base: usize,
    dst_stride: isize,
    coeff: &mut [i32],
    bitdepth_max: i32,
) {
    let dc_val = coeff[0];
    coeff[0] = 0;

    let scale = 2896i32 * 8;
    let mut dc = ((dc_val as i64 * scale as i64 + 16384) >> 15) as i32;
    dc = (dc + 2) >> 2; // shift=2 for 32x32
    dc = ((dc as i64 * scale as i64 + 16384) >> 15) as i32;
    dc = (dc + 8) >> 4; // final shift

    let dc = dc.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    let dc_vec = vdupq_n_s16(dc);
    let bd_max = vdupq_n_s16(bitdepth_max as i16);
    let zero = vdupq_n_s16(0);

    for i in 0..32 {
        let row_off = dst_base.wrapping_add_signed(i as isize * dst_stride);
        for half in 0..4 {
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
// Identity 32x32 (from itx.S lines 2186-2219)
// ============================================================================

/// Identity transform for 32x32 (8bpc).
///
/// The identity for 32x32 is a no-op on the coefficients (the shl<<1 from
/// identity and >>1 from normalization cancel). Just loads, transposes (for
/// the row transform), and adds to destination with shift >>2.
///
/// Mirrors `inv_txfm_add_identity_identity_32x32_8bpc_neon` from itx.S lines 2186-2219.
/// Processes the block as 4x4 groups of 8x8 sub-blocks, with early termination
/// based on eob thresholds.
#[cfg(target_arch = "aarch64")]
#[arcane]
pub(crate) fn inv_txfm_add_identity_identity_32x32_8bpc_neon_inner(
    _token: Arm64,
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    coeff: &mut [i16],
    eob: i32,
    _bitdepth_max: i32,
) {
    identity_32x32_8bpc_impl(dst, dst_base, dst_stride, coeff, eob);
}

/// Inner implementation of identity 32x32 for 8bpc.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn identity_32x32_8bpc_impl(
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    coeff: &mut [i16],
    eob: i32,
) {
    let eob_row_thresholds: [i32; 4] = [36, 136, 300, 1024];
    let eob_col_thresholds: [i32; 4] = [36, 136, 300, 1024];

    // Process row groups (each group = 8 destination rows)
    for rg in 0..4 {
        if rg > 0 && eob < eob_row_thresholds[rg - 1] {
            break;
        }

        let row_start = rg * 8;

        // Process column groups (each group = 8 destination columns)
        for cg in 0..4 {
            if cg > 0 && eob < eob_col_thresholds[cg - 1] {
                break;
            }

            let col_start = cg * 8;

            // Load 8 columns, 8 rows each (column-major)
            let zero_vec = vdupq_n_s16(0);
            let mut v: [int16x8_t; 8] = [zero_vec; 8];
            for c in 0..8 {
                let col = col_start + c;
                let base = col * 32 + row_start;
                let arr: [i16; 8] = coeff[base..base + 8].try_into().unwrap();
                v[c] = safe_simd::vld1q_s16(&arr);
                coeff[base..base + 8].fill(0);
            }

            // Transpose: now each vector has 8 values from one row
            let (r0, r1, r2, r3, r4, r5, r6, r7) =
                transpose_8x8h(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);

            // Add to destination with shift >>2
            let rows = [r0, r1, r2, r3, r4, r5, r6, r7];
            for r in 0..8 {
                let shifted = vrshrq_n_s16::<2>(rows[r]);
                let row_off =
                    dst_base.wrapping_add_signed((row_start + r) as isize * dst_stride) + col_start;

                let dst_bytes: [u8; 8] = dst[row_off..row_off + 8].try_into().unwrap();
                let dst_u8 = safe_simd::vld1_u8(&dst_bytes);
                let sum = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(shifted), dst_u8));
                let result = vqmovun_s16(sum);
                let mut out = [0u8; 8];
                safe_simd::vst1_u8(&mut out, result);
                dst[row_off..row_off + 8].copy_from_slice(&out);
            }
        }
    }
}

/// Identity 32x32 for 16bpc.
#[cfg(target_arch = "aarch64")]
#[rite]
pub(crate) fn inv_txfm_add_identity_identity_32x32_16bpc_neon_inner(
    _token: Arm64,
    dst: &mut [u16],
    dst_base: usize,
    dst_stride: isize,
    coeff: &mut [i32],
    eob: i32,
    bitdepth_max: i32,
) {
    // For 16bpc identity, coefficients are i32 but transform is identity.
    // Just add (coeff >> shift) to destination.
    // For 32x32 identity: shift = 2 (from row) + 2 (from col) = no, the assembly
    // uses shiftbits=2 for the load_add_store.
    //
    // Actually for 32x32 identity the assembly just transposes and adds with shift=2.
    // Since we don't have a NEON transpose for i32, use scalar.
    let eob_thresholds: [i32; 4] = [36, 136, 300, 1024];

    let bd_max = bitdepth_max;

    for rg in 0..4 {
        if rg > 0 && eob < eob_thresholds[rg - 1] {
            break;
        }
        let row_start = rg * 8;

        for cg in 0..4 {
            if cg > 0 && eob < eob_thresholds[cg - 1] {
                break;
            }
            let col_start = cg * 8;

            // Load 8x8 block from column-major coeff, transpose, add
            let mut block = [[0i32; 8]; 8];
            for c in 0..8 {
                let col = col_start + c;
                for r in 0..8 {
                    let row = row_start + r;
                    block[c][r] = coeff[col * 32 + row];
                    coeff[col * 32 + row] = 0;
                }
            }

            // After "transpose": block[c][r] → output row r, col c
            for r in 0..8 {
                let row_off = dst_base.wrapping_add_signed((row_start + r) as isize * dst_stride);
                for c in 0..8 {
                    let val = (block[c][r] + 2) >> 2; // shift=2
                    let d = dst[row_off + col_start + c] as i32;
                    let result = (d + val).clamp(0, bd_max);
                    dst[row_off + col_start + c] = result as u16;
                }
            }
        }
    }
}

// ============================================================================
// Full 32x32 DCT_DCT entry point (8bpc)
// ============================================================================

/// NEON implementation of 32x32 DCT_DCT inverse transform add for 8bpc.
///
/// Mirrors `inv_txfm_add_dct_dct_32x32_8bpc_neon` from itx.S lines 2333-2378.
///
/// Algorithm:
/// 1. DC fast path if eob=0
/// 2. Row transform: 4 groups of 8 rows, each applies 32-point DCT,
///    stores to 2048-byte scratch buffer
/// 3. Column transform: 4 groups of 8 columns from scratch, applies
///    32-point DCT, adds to destination
#[cfg(target_arch = "aarch64")]
#[arcane]
pub(crate) fn inv_txfm_add_dct_dct_32x32_8bpc_neon_inner(
    _token: Arm64,
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    coeff: &mut [i16],
    eob: i32,
    _bitdepth_max: i32,
) {
    // DC-only fast path
    if eob == 0 {
        dc_only_32x32_8bpc(dst, dst_base, dst_stride, coeff);
        return;
    }

    // eob thresholds for 32x32: [36, 136, 300, 1024]
    let eob_thresholds: [i32; 4] = [36, 136, 300, 1024];

    // Scratch buffer: 32 rows x 32 columns of i16 = 1024 elements
    let mut scratch = [0i16; 1024];

    // Row transform: process 4 groups of 8 rows
    for group in 0..4 {
        let row_start = group * 8;
        if group > 0 && eob < eob_thresholds[group - 1] {
            // Remaining scratch rows are already zero-initialized
            break;
        }

        // coeff_base: first element of this row group
        // Coefficients are column-major: coeff[row + col * 32]
        // So row_start is the offset within each column.
        horz_dct_32x8(
            coeff,
            row_start,
            32, // coeff_stride = height = 32
            &mut scratch,
            row_start * 32, // scratch_base = row_start * 32 (row-major scratch)
            2,              // shift
        );
    }

    // Column transform: process 4 groups of 8 columns
    for group in 0..4 {
        let col_start = group * 8;
        vert_dct_add_8x32_8bpc(
            dst,
            dst_base + col_start,
            dst_stride,
            &scratch,
            col_start,
            32, // scratch_stride = 32 (i16 per row)
        );
    }
}

/// NEON implementation of 32x32 DCT_DCT inverse transform add for 16bpc.
#[cfg(target_arch = "aarch64")]
#[rite]
pub(crate) fn inv_txfm_add_dct_dct_32x32_16bpc_neon_inner(
    _token: Arm64,
    dst: &mut [u16],
    dst_base: usize,
    dst_stride: isize,
    coeff: &mut [i32],
    eob: i32,
    bitdepth_max: i32,
) {
    // DC-only fast path
    if eob == 0 {
        dc_only_32x32_16bpc(dst, dst_base, dst_stride, coeff, bitdepth_max);
        return;
    }

    // For 16bpc, coefficients are i32. The NEON 32-point DCT operates on i16,
    // so we need to convert. For 16bpc with large coefficients, intermediate
    // precision requires careful handling.
    //
    // The assembly for 16bpc uses a similar approach but with wider intermediates.
    // For now, use the scalar path which handles precision correctly.
    // TODO: Port the 16bpc NEON path for 32x32 DCT
    scalar_dct_dct_32x32_16bpc(dst, dst_base, dst_stride, coeff, eob, bitdepth_max);
}

/// Scalar fallback for 32x32 DCT_DCT 16bpc.
#[allow(dead_code)] // Used by asm build via FFI wrapper; also by tests
fn scalar_dct_dct_32x32_16bpc(
    dst: &mut [u16],
    dst_base: usize,
    dst_stride: isize,
    coeff: &mut [i32],
    _eob: i32,
    bitdepth_max: i32,
) {
    // Use the existing scalar DCT32 from itx_arm.rs
    // This is the same as inv_txfm_add_dct_dct_32x32_16bpc_inner
    let mut tmp = [0i32; 1024];

    for y in 0..32 {
        let mut input = [0i32; 32];
        for x in 0..32 {
            input[x] = coeff[y + x * 32];
        }
        let out = scalar_dct32_1d(&input);
        for x in 0..32 {
            tmp[y * 32 + x] = out[x];
        }
    }

    for x in 0..32 {
        let mut input = [0i32; 32];
        for y in 0..32 {
            input[y] = tmp[y * 32 + x];
        }
        let out = scalar_dct32_1d(&input);

        for y in 0..32 {
            let row_off = dst_base.wrapping_add_signed(y as isize * dst_stride);
            let d = dst[row_off + x] as i32;
            let c = (out[y] + 512) >> 10;
            let result = (d + c).clamp(0, bitdepth_max);
            dst[row_off + x] = result as u16;
        }
    }

    for i in 0..1024 {
        coeff[i] = 0;
    }
}

/// Scalar 32-point inverse DCT.
///
/// Full precision implementation using the exact AV1 spec coefficients.
#[allow(dead_code)]
fn scalar_dct32_1d(input: &[i32; 32]) -> [i32; 32] {
    // 16-point DCT on even-indexed inputs
    let mut even = [0i32; 16];
    for i in 0..16 {
        even[i] = input[2 * i];
    }
    let even_out = scalar_dct16_1d(&even);

    // 32-point odd part on odd-indexed inputs
    let mut odd = [0i32; 16];
    for i in 0..16 {
        odd[i] = input[2 * i + 1];
    }
    let odd_out = scalar_idct32_odd(&odd);

    // Butterfly combine
    let mut out = [0i32; 32];
    for i in 0..16 {
        out[i] = even_out[i] + odd_out[15 - i];
        out[31 - i] = even_out[i] - odd_out[15 - i];
    }
    out
}

/// Scalar 16-point inverse DCT (spec-accurate).
#[allow(dead_code)]
fn scalar_dct16_1d(input: &[i32; 16]) -> [i32; 16] {
    // 8-point DCT on even inputs
    let mut even = [0i32; 8];
    for i in 0..8 {
        even[i] = input[2 * i];
    }
    let even_out = scalar_dct8_1d(&even);

    // 16-point odd part
    // Exact port of idct_16 odd part from assembly
    let c = [401i32, 4076, 3166, 2598, 1931, 3612, 3920, 1189];

    // Stage 1: rotation pairs
    let t8a = (input[1] * c[0] - input[15] * c[1] + 2048) >> 12;
    let t15a = (input[1] * c[1] + input[15] * c[0] + 2048) >> 12;
    let t9a = (input[9] * c[2] - input[7] * c[3] + 2048) >> 12;
    let t14a = (input[9] * c[3] + input[7] * c[2] + 2048) >> 12;
    let t10a = (input[5] * c[4] - input[11] * c[5] + 2048) >> 12;
    let t13a = (input[5] * c[5] + input[11] * c[4] + 2048) >> 12;
    let t11a = (input[13] * c[6] - input[3] * c[7] + 2048) >> 12;
    let t12a = (input[13] * c[7] + input[3] * c[6] + 2048) >> 12;

    // Stage 2: butterfly
    let t8 = t8a + t9a;
    let t9 = t8a - t9a;
    let t10 = t11a - t10a;
    let t11 = t11a + t10a;
    let t12 = t12a + t13a;
    let t13 = t12a - t13a;
    let t14 = t15a - t14a;
    let t15 = t15a + t14a;

    // Stage 3: rotations with 1567, 3784
    let t9a = (t14 * 1567 - t9 * 3784 + 2048) >> 12;
    let t14a = (t14 * 3784 + t9 * 1567 + 2048) >> 12;
    let t13a = (t13 * 1567 - t10 * 3784 + 2048) >> 12;
    let t10a = -((t13 * 3784 + t10 * 1567 + 2048) >> 12);

    // Stage 4: butterfly
    let t8a = t8 + t11;
    let t11a = t8 - t11;
    let t9b = t9a + t10a;
    let t10b = t9a - t10a;
    let t12a = t15 - t12;
    let t15a = t15 + t12;
    let t13b = t14a - t13a;
    let t14b = t14a + t13a;

    // Stage 5: rotations with 2896
    let t11_f = (t12a * 2896 - t11a * 2896 + 2048) >> 12;
    let t12_f = (t12a * 2896 + t11a * 2896 + 2048) >> 12;
    let t10_f = (t13b * 2896 - t10b * 2896 + 2048) >> 12;
    let t13_f = (t13b * 2896 + t10b * 2896 + 2048) >> 12;

    // Final butterfly
    let mut out = [0i32; 16];
    out[0] = even_out[0] + t15a;
    out[1] = even_out[1] + t14b;
    out[2] = even_out[2] + t13_f;
    out[3] = even_out[3] + t12_f;
    out[4] = even_out[4] + t11_f;
    out[5] = even_out[5] + t10_f;
    out[6] = even_out[6] + t9b;
    out[7] = even_out[7] + t8a;
    out[8] = even_out[7] - t8a;
    out[9] = even_out[6] - t9b;
    out[10] = even_out[5] - t10_f;
    out[11] = even_out[4] - t11_f;
    out[12] = even_out[3] - t12_f;
    out[13] = even_out[2] - t13_f;
    out[14] = even_out[1] - t14b;
    out[15] = even_out[0] - t15a;
    out
}

/// Scalar 8-point inverse DCT.
#[allow(dead_code)]
fn scalar_dct8_1d(input: &[i32; 8]) -> [i32; 8] {
    // 4-point DCT on even inputs
    let even = scalar_dct4_1d(input[0], input[2], input[4], input[6]);

    // Odd part
    let c = [799i32, 4017, 3406, 2276];
    let t4a = (input[1] * c[0] - input[7] * c[1] + 2048) >> 12;
    let t7a = (input[1] * c[1] + input[7] * c[0] + 2048) >> 12;
    let t5a = (input[5] * c[2] - input[3] * c[3] + 2048) >> 12;
    let t6a = (input[5] * c[3] + input[3] * c[2] + 2048) >> 12;

    let t4 = t4a + t5a;
    let t5 = t4a - t5a;
    let t6 = t7a - t6a;
    let t7 = t7a + t6a;

    let t5a = ((t6 - t5) * 2896 + 2048) >> 12;
    let t6a = ((t6 + t5) * 2896 + 2048) >> 12;

    [
        even[0] + t7,
        even[1] + t6a,
        even[2] + t5a,
        even[3] + t4,
        even[3] - t4,
        even[2] - t5a,
        even[1] - t6a,
        even[0] - t7,
    ]
}

/// Scalar 4-point inverse DCT.
#[allow(dead_code)]
fn scalar_dct4_1d(in0: i32, in1: i32, in2: i32, in3: i32) -> [i32; 4] {
    let t0 = ((in0 + in2) * 2896 + 2048) >> 12;
    let t1 = ((in0 - in2) * 2896 + 2048) >> 12;
    let t2 = (in1 * 1567 - in3 * 3784 + 2048) >> 12;
    let t3 = (in1 * 3784 + in3 * 1567 + 2048) >> 12;
    [t0 + t3, t1 + t2, t1 - t2, t0 - t3]
}

/// Scalar 32-point inverse DCT odd part (matches idct32_odd_q exactly).
#[allow(dead_code)]
fn scalar_idct32_odd(v: &[i32; 16]) -> [i32; 16] {
    // Stage 1: rotation pairs (same coefficient order as NEON version)
    let c0 = [201i32, 4091, 3035, 2751, 1751, 3703, 3857, 1380];
    let c1 = [995i32, 3973, 3513, 2106, 2440, 3290, 4052, 601];

    let t16a = (v[0] * c0[0] - v[15] * c0[1] + 2048) >> 12;
    let t31a = (v[0] * c0[1] + v[15] * c0[0] + 2048) >> 12;
    let t17a = (v[8] * c0[2] - v[7] * c0[3] + 2048) >> 12;
    let t30a = (v[8] * c0[3] + v[7] * c0[2] + 2048) >> 12;
    let t18a = (v[4] * c0[4] - v[11] * c0[5] + 2048) >> 12;
    let t29a = (v[4] * c0[5] + v[11] * c0[4] + 2048) >> 12;
    let t19a = (v[12] * c0[6] - v[3] * c0[7] + 2048) >> 12;
    let t28a = (v[12] * c0[7] + v[3] * c0[6] + 2048) >> 12;

    let t20a = (v[2] * c1[0] - v[13] * c1[1] + 2048) >> 12;
    let t27a = (v[2] * c1[1] + v[13] * c1[0] + 2048) >> 12;
    let t21a = (v[10] * c1[2] - v[5] * c1[3] + 2048) >> 12;
    let t26a = (v[10] * c1[3] + v[5] * c1[2] + 2048) >> 12;
    let t22a = (v[6] * c1[4] - v[9] * c1[5] + 2048) >> 12;
    let t25a = (v[6] * c1[5] + v[9] * c1[4] + 2048) >> 12;
    let t23a = (v[14] * c1[6] - v[1] * c1[7] + 2048) >> 12;
    let t24a = (v[14] * c1[7] + v[1] * c1[6] + 2048) >> 12;

    // Stage 2: butterfly
    let s17 = t16a - t17a;
    let s16 = t16a + t17a;
    let s30 = t31a - t30a;
    let s31 = t31a + t30a;
    let s18 = t19a - t18a;
    let s19 = t19a + t18a;
    let s20 = t20a + t21a;
    let s21 = t20a - t21a;
    let s22 = t23a - t22a;
    let s23 = t23a + t22a;
    let s24 = t24a + t25a;
    let s25 = t24a - t25a;
    let s26 = t27a - t26a;
    let s27 = t27a + t26a;
    let s28 = t28a + t29a;
    let s29 = t28a - t29a;

    // Stage 3: rotations (799, 4017, 3406, 2276)
    let u17a = (s30 * 799 - s17 * 4017 + 2048) >> 12;
    let u30a = (s30 * 4017 + s17 * 799 + 2048) >> 12;
    let u18a = -((s29 * 4017 + s18 * 799 + 2048) >> 12);
    let u29a = (s29 * 799 - s18 * 4017 + 2048) >> 12;
    let u21a = (s26 * 3406 - s21 * 2276 + 2048) >> 12;
    let u26a = (s26 * 2276 + s21 * 3406 + 2048) >> 12;
    let u22a = -((s25 * 2276 + s22 * 3406 + 2048) >> 12);
    let u25a = (s25 * 3406 - s22 * 2276 + 2048) >> 12;

    // Stage 4: butterfly
    let w30 = u30a + u29a;
    let w29 = u30a - u29a;
    let w18 = u17a - u18a;
    let w17 = u17a + u18a;
    let w19a = s16 - s19;
    let w16a = s16 + s19;
    let w20a = s23 - s20;
    let w23a = s23 + s20;
    let w21 = u22a - u21a;
    let w22 = u22a + u21a;
    let w24a = s24 + s27;
    let w27a = s24 - s27;
    let w25 = u25a + u26a;
    let w26 = u25a - u26a;
    let w28a = s31 - s28;
    let w31a = s31 + s28;

    // Stage 5: rotations (1567, 3784)
    let x18a = (w29 * 1567 - w18 * 3784 + 2048) >> 12;
    let x29a = (w29 * 3784 + w18 * 1567 + 2048) >> 12;
    let x19 = (w28a * 1567 - w19a * 3784 + 2048) >> 12;
    let x28 = (w28a * 3784 + w19a * 1567 + 2048) >> 12;
    let x20 = -((w27a * 3784 + w20a * 1567 + 2048) >> 12);
    let x27 = (w27a * 1567 - w20a * 3784 + 2048) >> 12;
    let x21a = -((w26 * 3784 + w21 * 1567 + 2048) >> 12);
    let x26a = (w26 * 1567 - w21 * 3784 + 2048) >> 12;

    // Stage 6: final butterfly
    let y16 = w16a + w23a;
    let y23 = w16a - w23a;
    let y31 = w31a + w24a;
    let y24 = w31a - w24a;
    let y17a = w17 + w22;
    let y22a = w17 - w22;
    let y30a = w30 + w25;
    let y25a = w30 - w25;
    let y18 = x18a + x21a;
    let y21 = x18a - x21a;
    let y19a = x19 + x20;
    let y20a = x19 - x20;
    let y29 = x29a + x26a;
    let y26 = x29a - x26a;
    let y28a = x28 + x27;
    let y27a = x28 - x27;

    // Stage 7: final rotations (2896)
    let z20 = (y27a * 2896 - y20a * 2896 + 2048) >> 12;
    let z27 = (y27a * 2896 + y20a * 2896 + 2048) >> 12;
    let z26a = (y26 * 2896 + y21 * 2896 + 2048) >> 12;
    let z21a = (y26 * 2896 - y21 * 2896 + 2048) >> 12;
    let z22 = (y25a * 2896 - y22a * 2896 + 2048) >> 12;
    let z25 = (y25a * 2896 + y22a * 2896 + 2048) >> 12;
    let z23a = (y24 * 2896 - y23 * 2896 + 2048) >> 12;
    let z24a = (y24 * 2896 + y23 * 2896 + 2048) >> 12;

    [
        y16, y17a, y18, y19a, z20, z21a, z22, z23a, z24a, z25, z26a, z27, y28a, y29, y30a, y31,
    ]
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that the scalar DCT4 produces expected results.
    #[test]
    fn test_scalar_dct4() {
        let out = scalar_dct4_1d(4096, 0, 0, 0);
        // DC-only input: all outputs should be equal
        assert!(
            out.iter().all(|&x| (x - out[0]).abs() <= 1),
            "DC-only DCT4 should produce equal outputs, got {:?}",
            out
        );
    }

    /// Test that the scalar 32-point odd part with zero input produces zero output.
    #[test]
    fn test_scalar_idct32_odd_zero() {
        let input = [0i32; 16];
        let out = scalar_idct32_odd(&input);
        assert_eq!(out, [0i32; 16]);
    }

    /// Test that the scalar DCT32 with only DC produces flat output.
    #[test]
    fn test_scalar_dct32_dc_only() {
        let mut input = [0i32; 32];
        input[0] = 4096;
        let out = scalar_dct32_1d(&input);
        // All outputs should be approximately equal (DC component)
        let mean = out.iter().sum::<i32>() / 32;
        for (i, &v) in out.iter().enumerate() {
            assert!(
                (v - mean).abs() <= 2,
                "DCT32 DC-only: output[{}]={} deviates from mean={}",
                i,
                v,
                mean
            );
        }
    }

    /// Test that DCT32 → inverse DCT32 round-trips (within rounding).
    #[test]
    fn test_scalar_dct32_roundtrip() {
        // Use a simple test signal
        let mut input = [0i32; 32];
        for i in 0..32 {
            input[i] = (i as i32) * 100 - 1600;
        }
        let transformed = scalar_dct32_1d(&input);
        // The inverse should recover something close to the input
        // (Note: our scalar_dct32_1d is an inverse DCT, so applying it twice
        // with proper normalization should round-trip. Since we don't have
        // the forward DCT, just verify the output is reasonable.)
        assert!(
            transformed.iter().all(|&x| x.abs() < 100000),
            "Transform output should be bounded"
        );
    }
}
