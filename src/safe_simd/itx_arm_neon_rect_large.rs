//! Safe ARM NEON rectangular inverse transforms (4x16, 16x4, 8x16, 16x8)
//!
//! Port of the rectangular inverse transform functions from `src/arm/64/itx.S`
//! (lines 1496-1853) to safe Rust NEON intrinsics. These handle blocks where
//! width/height ratio is 4:1 or 1:4 (4x16/16x4) or 2:1/1:2 (8x16/16x8).
//!
//! **16x4** (16 columns, 4 rows):
//!   - Load 16 x int16x4_t, apply 16-point row transform (.4h),
//!     combine pairs to .8h + srshr>>1, transpose_4x8h, apply 4-point
//!     column transform (.8h), add to destination. Done in two 8-col halves.
//!
//! **4x16** (4 columns, 16 rows):
//!   - Load coefficients in two groups of 4 x int16x8_t, apply 4-point row
//!     transform (.8h), srshr>>1, transpose_4x8h, split to 4h, combine
//!     low and high halves, apply 16-point column transform (.4h),
//!     add to 4x16 destination.
//!
//! **16x8** (16 columns, 8 rows):
//!   - Load 16 x int16x8_t, scale_input, apply 16-point row transform (.8h),
//!     srshr>>1, transpose_8x8h, apply 8-point column transform (.8h),
//!     add to destination. Done in two 8-col halves.
//!
//! **8x16** (8 columns, 16 rows):
//!   - Load coefficients in two groups of 8 x int16x8_t, scale_input,
//!     apply 8-point row transform (.8h), srshr>>1, transpose_8x8h,
//!     combine halves, apply 16-point column transform (.8h),
//!     add to 8x16 destination.

#![allow(clippy::too_many_arguments)]
#![cfg_attr(not(any(feature = "asm", feature = "c-ffi", feature = "partial_asm")), forbid(unsafe_code))]
#![cfg_attr(any(feature = "asm", feature = "c-ffi", feature = "partial_asm"), deny(unsafe_code))]

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
use archmage::{Arm64, arcane, rite};

#[cfg(target_arch = "aarch64")]
use safe_unaligned_simd::aarch64 as safe_simd;

use super::itx_arm_neon_8x8::transpose_8x8h;
use super::itx_arm_neon_16x16::{
    IADST16_COEFFS_V0, IADST16_COEFFS_V1, TxType16, iadst_16_q, idct_16_q, identity_16_q,
};
use super::itx_arm_neon_common::IDCT_COEFFS;
use super::itx_arm_neon_rect::{
    RectTxType4, RectTxType8, apply_tx4_q, apply_tx8_q, transpose_4x8h,
};

/// Type alias for 16 NEON 4h vectors (one full 16-point 4h transform state).
#[cfg(target_arch = "aarch64")]
type V16_4h = [int16x4_t; 16];

// ============================================================================
// 16-point transforms on int16x4_t (.4h width)
// ============================================================================

/// Widening multiply-accumulate for int16x4_t: s0*c0_lane + s1*c1_lane.
///
/// Reuses the pattern from itx_arm_neon_rect but inlined here for
/// two-register coefficient vectors (c0 and c1 used by 16-point transforms).
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
#[inline(always)]
fn smull_smlal_4h_q(
    s0: int16x4_t,
    s1: int16x4_t,
    coeffs: int16x8_t,
    c0_lane: usize,
    c1_lane: usize,
) -> int32x4_t {
    use super::itx_arm_neon_rect::smull_smlal_4h;
    smull_smlal_4h(s0, s1, coeffs, c0_lane, c1_lane)
}

/// Widening multiply-subtract for int16x4_t: s0*c0_lane - s1*c1_lane.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
#[inline(always)]
fn smull_smlsl_4h_q(
    s0: int16x4_t,
    s1: int16x4_t,
    coeffs: int16x8_t,
    c0_lane: usize,
    c1_lane: usize,
) -> int32x4_t {
    use super::itx_arm_neon_rect::smull_smlsl_4h;
    smull_smlsl_4h(s0, s1, coeffs, c0_lane, c1_lane)
}

/// 16-point inverse DCT on 16 int16x4_t vectors (4 transforms in parallel).
///
/// Same algorithm as `idct_16_q` but operates on 64-bit (.4h) registers.
/// Matches `inv_dct_4h_x16_neon` from itx.S line 1123.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn idct_16_4h(v: V16_4h) -> V16_4h {
    // Apply idct_8_4h to even-indexed inputs
    use super::itx_arm_neon_rect::idct_8_4h;
    let (e0, e1, e2, e3, e4, e5, e6, e7) =
        idct_8_4h(v[0], v[2], v[4], v[6], v[8], v[10], v[12], v[14]);

    // Load idct16 coefficients (the second 8 shorts)
    let c1 = safe_simd::vld1q_s16(<&[i16; 8]>::try_from(&IDCT_COEFFS[8..16]).unwrap());

    // Stage 1: Rotation pairs on odd-indexed inputs
    let t8a = vqrshrn_n_s32::<12>(smull_smlsl_4h_q(v[1], v[15], c1, 0, 1));
    let t15a = vqrshrn_n_s32::<12>(smull_smlal_4h_q(v[1], v[15], c1, 1, 0));
    let t9a = vqrshrn_n_s32::<12>(smull_smlsl_4h_q(v[9], v[7], c1, 2, 3));
    let t14a = vqrshrn_n_s32::<12>(smull_smlal_4h_q(v[9], v[7], c1, 3, 2));
    let t10a = vqrshrn_n_s32::<12>(smull_smlsl_4h_q(v[5], v[11], c1, 4, 5));
    let t13a = vqrshrn_n_s32::<12>(smull_smlal_4h_q(v[5], v[11], c1, 5, 4));
    let t11a = vqrshrn_n_s32::<12>(smull_smlsl_4h_q(v[13], v[3], c1, 6, 7));
    let t12a = vqrshrn_n_s32::<12>(smull_smlal_4h_q(v[13], v[3], c1, 7, 6));

    // Stage 2: Butterfly
    let t9 = vqsub_s16(t8a, t9a);
    let t8 = vqadd_s16(t8a, t9a);
    let t14 = vqsub_s16(t15a, t14a);
    let t15 = vqadd_s16(t15a, t14a);
    let t10 = vqsub_s16(t11a, t10a);
    let t11 = vqadd_s16(t11a, t10a);
    let t12 = vqadd_s16(t12a, t13a);
    let t13 = vqsub_s16(t12a, t13a);

    // Stage 3: Rotations using idct4 coefficients
    let c0 = safe_simd::vld1q_s16(<&[i16; 8]>::try_from(&IDCT_COEFFS[0..8]).unwrap());

    let t9a = vqrshrn_n_s32::<12>(smull_smlsl_4h_q(t14, t9, c0, 2, 3));
    let t14a = vqrshrn_n_s32::<12>(smull_smlal_4h_q(t14, t9, c0, 3, 2));
    let t13a = vqrshrn_n_s32::<12>(smull_smlsl_4h_q(t13, t10, c0, 2, 3));
    let t10a = vqrshrn_n_s32::<12>(vnegq_s32(smull_smlal_4h_q(t13, t10, c0, 3, 2)));

    // Stage 4: Butterfly
    let t11a = vqsub_s16(t8, t11);
    let t8a = vqadd_s16(t8, t11);
    let t12a = vqsub_s16(t15, t12);
    let t15a = vqadd_s16(t15, t12);
    let t9b = vqadd_s16(t9a, t10a);
    let t10b = vqsub_s16(t9a, t10a);
    let t13b = vqsub_s16(t14a, t13a);
    let t14b = vqadd_s16(t14a, t13a);

    // Stage 5: Final rotations using v0.h[0]=2896
    let t11_final = vqrshrn_n_s32::<12>(smull_smlsl_4h_q(t12a, t11a, c0, 0, 0));
    let t12_final = vqrshrn_n_s32::<12>(smull_smlal_4h_q(t12a, t11a, c0, 0, 0));
    let t10a_final = vqrshrn_n_s32::<12>(smull_smlsl_4h_q(t13b, t10b, c0, 0, 0));
    let t13a_final = vqrshrn_n_s32::<12>(smull_smlal_4h_q(t13b, t10b, c0, 0, 0));

    // Final butterfly
    [
        vqadd_s16(e0, t15a),       // out0
        vqadd_s16(e1, t14b),       // out1
        vqadd_s16(e2, t13a_final), // out2
        vqadd_s16(e3, t12_final),  // out3
        vqadd_s16(e4, t11_final),  // out4
        vqadd_s16(e5, t10a_final), // out5
        vqadd_s16(e6, t9b),        // out6
        vqadd_s16(e7, t8a),        // out7
        vqsub_s16(e7, t8a),        // out8
        vqsub_s16(e6, t9b),        // out9
        vqsub_s16(e5, t10a_final), // out10
        vqsub_s16(e4, t11_final),  // out11
        vqsub_s16(e3, t12_final),  // out12
        vqsub_s16(e2, t13a_final), // out13
        vqsub_s16(e1, t14b),       // out14
        vqsub_s16(e0, t15a),       // out15
    ]
}

/// 16-point inverse ADST on 16 int16x4_t vectors (4 transforms in parallel).
///
/// Same algorithm as `iadst_16_q` but operates on 64-bit (.4h) registers.
/// Matches `inv_adst_4h_x16_neon` from itx.S line 1310.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn iadst_16_4h(v: V16_4h) -> V16_4h {
    let c0 = safe_simd::vld1q_s16(<&[i16; 8]>::try_from(&IADST16_COEFFS_V0[..]).unwrap());
    let c1 = safe_simd::vld1q_s16(<&[i16; 8]>::try_from(&IADST16_COEFFS_V1[..]).unwrap());

    // Stage 1: 8 rotation pairs
    let t0 = vqrshrn_n_s32::<12>(smull_smlal_4h_q(v[15], v[0], c0, 0, 1));
    let t1 = vqrshrn_n_s32::<12>(smull_smlsl_4h_q(v[15], v[0], c0, 1, 0));
    let t2 = vqrshrn_n_s32::<12>(smull_smlal_4h_q(v[13], v[2], c0, 2, 3));
    let t3 = vqrshrn_n_s32::<12>(smull_smlsl_4h_q(v[13], v[2], c0, 3, 2));
    let t4 = vqrshrn_n_s32::<12>(smull_smlal_4h_q(v[11], v[4], c0, 4, 5));
    let t5 = vqrshrn_n_s32::<12>(smull_smlsl_4h_q(v[11], v[4], c0, 5, 4));
    let t6 = vqrshrn_n_s32::<12>(smull_smlal_4h_q(v[9], v[6], c0, 6, 7));
    let t7 = vqrshrn_n_s32::<12>(smull_smlsl_4h_q(v[9], v[6], c0, 7, 6));
    let t8 = vqrshrn_n_s32::<12>(smull_smlal_4h_q(v[7], v[8], c1, 0, 1));
    let t9 = vqrshrn_n_s32::<12>(smull_smlsl_4h_q(v[7], v[8], c1, 1, 0));
    let t10 = vqrshrn_n_s32::<12>(smull_smlal_4h_q(v[5], v[10], c1, 2, 3));
    let t11 = vqrshrn_n_s32::<12>(smull_smlsl_4h_q(v[5], v[10], c1, 3, 2));
    let t12 = vqrshrn_n_s32::<12>(smull_smlal_4h_q(v[3], v[12], c1, 4, 5));
    let t13 = vqrshrn_n_s32::<12>(smull_smlsl_4h_q(v[3], v[12], c1, 5, 4));
    let t14 = vqrshrn_n_s32::<12>(smull_smlal_4h_q(v[1], v[14], c1, 6, 7));
    let t15 = vqrshrn_n_s32::<12>(smull_smlsl_4h_q(v[1], v[14], c1, 7, 6));

    // Load idct coefficients for the remaining stages
    let ci = safe_simd::vld1q_s16(<&[i16; 8]>::try_from(&IDCT_COEFFS[0..8]).unwrap());

    // Stage 2: butterfly pairs
    let s8a = vqsub_s16(t0, t8);
    let s0a = vqadd_s16(t0, t8);
    let s9a = vqsub_s16(t1, t9);
    let s1a = vqadd_s16(t1, t9);
    let s2a = vqadd_s16(t2, t10);
    let s10a = vqsub_s16(t2, t10);
    let s3a = vqadd_s16(t3, t11);
    let s11a = vqsub_s16(t3, t11);
    let s4a = vqadd_s16(t4, t12);
    let s12a = vqsub_s16(t4, t12);
    let s5a = vqadd_s16(t5, t13);
    let s13a = vqsub_s16(t5, t13);
    let s6a = vqadd_s16(t6, t14);
    let s14a = vqsub_s16(t6, t14);
    let s7a = vqadd_s16(t7, t15);
    let s15a = vqsub_s16(t7, t15);

    // Stage 3: rotations
    let u8_ = vqrshrn_n_s32::<12>(smull_smlal_4h_q(s8a, s9a, ci, 5, 4));
    let u9 = vqrshrn_n_s32::<12>(smull_smlsl_4h_q(s8a, s9a, ci, 4, 5));
    let u10 = vqrshrn_n_s32::<12>(smull_smlal_4h_q(s10a, s11a, ci, 7, 6));
    let u11 = vqrshrn_n_s32::<12>(smull_smlsl_4h_q(s10a, s11a, ci, 6, 7));
    let u12 = vqrshrn_n_s32::<12>(smull_smlsl_4h_q(s13a, s12a, ci, 5, 4));
    let u13 = vqrshrn_n_s32::<12>(smull_smlal_4h_q(s13a, s12a, ci, 4, 5));
    let u14 = vqrshrn_n_s32::<12>(smull_smlsl_4h_q(s15a, s14a, ci, 7, 6));
    let u15 = vqrshrn_n_s32::<12>(smull_smlal_4h_q(s15a, s14a, ci, 6, 7));

    // Stage 4: butterfly pairs
    let w4 = vqsub_s16(s0a, s4a);
    let w0 = vqadd_s16(s0a, s4a);
    let w5 = vqsub_s16(s1a, s5a);
    let w1 = vqadd_s16(s1a, s5a);
    let w2 = vqadd_s16(s2a, s6a);
    let w6 = vqsub_s16(s2a, s6a);
    let w3 = vqadd_s16(s3a, s7a);
    let w7 = vqsub_s16(s3a, s7a);
    let w8a = vqadd_s16(u8_, u12);
    let w12a = vqsub_s16(u8_, u12);
    let w9a = vqadd_s16(u9, u13);
    let w13a = vqsub_s16(u9, u13);
    let w10a = vqadd_s16(u10, u14);
    let w14a = vqsub_s16(u10, u14);
    let w11a = vqadd_s16(u11, u15);
    let w15a = vqsub_s16(u11, u15);

    // Stage 5: rotations
    let x4a = vqrshrn_n_s32::<12>(smull_smlal_4h_q(w4, w5, ci, 3, 2));
    let x5a = vqrshrn_n_s32::<12>(smull_smlsl_4h_q(w4, w5, ci, 2, 3));
    let x6a = vqrshrn_n_s32::<12>(smull_smlsl_4h_q(w7, w6, ci, 3, 2));
    let x7a = vqrshrn_n_s32::<12>(smull_smlal_4h_q(w7, w6, ci, 2, 3));
    let x12 = vqrshrn_n_s32::<12>(smull_smlal_4h_q(w12a, w13a, ci, 3, 2));
    let x13 = vqrshrn_n_s32::<12>(smull_smlsl_4h_q(w12a, w13a, ci, 2, 3));
    let x14 = vqrshrn_n_s32::<12>(smull_smlsl_4h_q(w15a, w14a, ci, 3, 2));
    let x15 = vqrshrn_n_s32::<12>(smull_smlal_4h_q(w15a, w14a, ci, 2, 3));

    // Stage 6: Final butterfly + rotations
    let o0 = vqadd_s16(w0, w2);
    let t2a = vqsub_s16(w0, w2);
    let o15 = vqneg_s16(vqadd_s16(w1, w3));
    let t3a = vqsub_s16(w1, w3);
    let o13 = vqneg_s16(vqadd_s16(x13, x15));
    let t15a = vqsub_s16(x13, x15);
    let o2 = vqadd_s16(x12, x14);
    let t14a = vqsub_s16(x12, x14);
    let o1 = vqneg_s16(vqadd_s16(w8a, w10a));
    let y10 = vqsub_s16(w8a, w10a);
    let o14 = vqadd_s16(w9a, w11a);
    let y11 = vqsub_s16(w9a, w11a);
    let o3 = vqneg_s16(vqadd_s16(x4a, x6a));
    let y6 = vqsub_s16(x4a, x6a);
    let o12 = vqadd_s16(x5a, x7a);
    let y7 = vqsub_s16(x5a, x7a);

    // Final rotations using ci.h[0]=2896
    let o8 = vqrshrn_n_s32::<12>(smull_smlsl_4h_q(t2a, t3a, ci, 0, 0));
    let o7_pre = vqrshrn_n_s32::<12>(smull_smlal_4h_q(t2a, t3a, ci, 0, 0));
    let o5_pre = vqrshrn_n_s32::<12>(smull_smlal_4h_q(t14a, t15a, ci, 0, 0));
    let o10 = vqrshrn_n_s32::<12>(smull_smlsl_4h_q(t14a, t15a, ci, 0, 0));
    let o4 = vqrshrn_n_s32::<12>(smull_smlal_4h_q(y6, y7, ci, 0, 0));
    let o11_pre = vqrshrn_n_s32::<12>(smull_smlsl_4h_q(y6, y7, ci, 0, 0));
    let o6 = vqrshrn_n_s32::<12>(smull_smlal_4h_q(y10, y11, ci, 0, 0));
    let o9_pre = vqrshrn_n_s32::<12>(smull_smlsl_4h_q(y10, y11, ci, 0, 0));

    let o7 = vqneg_s16(o7_pre);
    let o5 = vqneg_s16(o5_pre);
    let o11 = vqneg_s16(o11_pre);
    let o9 = vqneg_s16(o9_pre);

    [
        o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15,
    ]
}

/// 16-point identity transform on 16 int16x4_t vectors.
///
/// Matches `inv_identity_4h_x16_neon` from itx.S line 1331.
/// Multiplies by 2*sqrt(2) = 2*(5793/4096) using sqrdmulh + sqadd*2 trick.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn identity_16_4h(mut v: V16_4h) -> V16_4h {
    // Scale factor: 2*(5793-4096)*8 = 27152
    let scale = vdup_n_s16(((5793 - 4096) * 2 * 8) as i16);

    for vi in v.iter_mut() {
        let t = vqrdmulh_s16(*vi, scale);
        *vi = vqadd_s16(*vi, *vi); // vi *= 2
        *vi = vqadd_s16(*vi, t); // vi += round(vi * scale)
    }
    v
}

/// Apply a 1D 16-point transform to 16 int16x4_t vectors.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn apply_tx16_4h(tx: TxType16, v: V16_4h) -> V16_4h {
    match tx {
        TxType16::Dct => idct_16_4h(v),
        TxType16::Adst => iadst_16_4h(v),
        TxType16::FlipAdst => {
            let mut out = iadst_16_4h(v);
            out.reverse();
            out
        }
        TxType16::Identity => identity_16_4h(v),
    }
}

/// Apply a 1D 16-point transform to 16 int16x8_t vectors.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn apply_tx16_q(tx: TxType16, v: [int16x8_t; 16]) -> [int16x8_t; 16] {
    match tx {
        TxType16::Dct => idct_16_q(v),
        TxType16::Adst => iadst_16_q(v),
        TxType16::FlipAdst => {
            let mut out = iadst_16_q(v);
            out.reverse();
            out
        }
        TxType16::Identity => identity_16_q(v),
    }
}

// ============================================================================
// Scale input for 8 int16x8_t vectors
// ============================================================================

/// Scale 8 int16x8_t vectors by 2896*8 using sqrdmulh.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn scale_input_8h_8(v: &mut [int16x8_t; 8]) {
    let scale = vdupq_n_s16((2896 * 8) as i16);
    for vi in v.iter_mut() {
        *vi = vqrdmulhq_s16(*vi, scale);
    }
}

/// Scale 16 int16x8_t vectors by 2896*8 using sqrdmulh.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn scale_input_8h_16(v: &mut [int16x8_t; 16]) {
    let scale = vdupq_n_s16((2896 * 8) as i16);
    for vi in v.iter_mut() {
        *vi = vqrdmulhq_s16(*vi, scale);
    }
}

// ============================================================================
// Add-to-destination helpers
// ============================================================================

/// Add 8 rows of 8 pixels to destination for 8bpc.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn add_to_dst_8x8_8bpc(dst: &mut [u8], dst_base: usize, stride: isize, v: [int16x8_t; 8]) {
    for (i, &row) in v.iter().enumerate() {
        let row_off = dst_base.wrapping_add_signed(i as isize * stride);
        let shifted = vrshrq_n_s16::<4>(row);
        let dst_bytes: [u8; 8] = dst[row_off..row_off + 8].try_into().unwrap();
        let dst_u8 = safe_simd::vld1_u8(&dst_bytes);
        let sum = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(shifted), dst_u8));
        let result = vqmovun_s16(sum);
        let mut out = [0u8; 8];
        safe_simd::vst1_u8(&mut out, result);
        dst[row_off..row_off + 8].copy_from_slice(&out);
    }
}

/// Add 16 rows of 8 pixels to destination for 8bpc.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn add_to_dst_8x16_8bpc(dst: &mut [u8], dst_base: usize, stride: isize, v: [int16x8_t; 16]) {
    for (i, &row) in v.iter().enumerate() {
        let row_off = dst_base.wrapping_add_signed(i as isize * stride);
        let shifted = vrshrq_n_s16::<4>(row);
        let dst_bytes: [u8; 8] = dst[row_off..row_off + 8].try_into().unwrap();
        let dst_u8 = safe_simd::vld1_u8(&dst_bytes);
        let sum = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(shifted), dst_u8));
        let result = vqmovun_s16(sum);
        let mut out = [0u8; 8];
        safe_simd::vst1_u8(&mut out, result);
        dst[row_off..row_off + 8].copy_from_slice(&out);
    }
}

/// Add 16 rows of 4 pixels to destination for 8bpc.
///
/// Takes 16 int16x4_t vectors, processes pairs packed into int16x8_t,
/// adds to 4-pixel-wide destination rows.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn add_to_dst_4x16_8bpc(dst: &mut [u8], dst_base: usize, stride: isize, v: V16_4h) {
    // Process pairs of rows: combine v[2n] and v[2n+1] into one int16x8_t
    for pair_idx in 0..8 {
        let combined = vcombine_s16(v[pair_idx * 2], v[pair_idx * 2 + 1]);
        let shifted = vrshrq_n_s16::<4>(combined);

        let row0_off = dst_base.wrapping_add_signed((pair_idx * 2) as isize * stride);
        let row1_off = dst_base.wrapping_add_signed((pair_idx * 2 + 1) as isize * stride);

        let mut dst_bytes = [0u8; 8];
        dst_bytes[0..4].copy_from_slice(&dst[row0_off..row0_off + 4]);
        dst_bytes[4..8].copy_from_slice(&dst[row1_off..row1_off + 4]);
        let dst_u8 = safe_simd::vld1_u8(&dst_bytes);

        let sum = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(shifted), dst_u8));
        let result = vqmovun_s16(sum);

        let mut out = [0u8; 8];
        safe_simd::vst1_u8(&mut out, result);
        dst[row0_off..row0_off + 4].copy_from_slice(&out[0..4]);
        dst[row1_off..row1_off + 4].copy_from_slice(&out[4..8]);
    }
}

/// Add 4 rows of 8 pixels to destination for 8bpc (reusing the rect helper).
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn add_to_dst_8x4_8bpc(
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    v0: int16x8_t,
    v1: int16x8_t,
    v2: int16x8_t,
    v3: int16x8_t,
) {
    let rows = [v0, v1, v2, v3];
    for (i, &row) in rows.iter().enumerate() {
        let row_off = dst_base.wrapping_add_signed(i as isize * stride);
        let shifted = vrshrq_n_s16::<4>(row);
        let dst_bytes: [u8; 8] = dst[row_off..row_off + 8].try_into().unwrap();
        let dst_u8 = safe_simd::vld1_u8(&dst_bytes);
        let sum = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(shifted), dst_u8));
        let result = vqmovun_s16(sum);
        let mut out = [0u8; 8];
        safe_simd::vst1_u8(&mut out, result);
        dst[row_off..row_off + 8].copy_from_slice(&out);
    }
}

// ============================================================================
// DC-only fast paths
// ============================================================================

/// DC-only fast path for DCT_DCT with the given dimensions.
///
/// For rectangular blocks (w*4==h or h*4==w or w*2==h or h*2==w):
///   - Extra sqrdmulh for rect2 (2:1 ratio)
///   - Extra sqrdmulh for rect4 (4:1 ratio) — actually same scaling
/// shift=1 for 4x16/16x4/8x16/16x8
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn dc_only_rect_8bpc(
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    coeff: &mut [i16],
    w: usize,
    h: usize,
) {
    let dc = coeff[0];
    coeff[0] = 0;

    let scale = vdupq_n_s16((2896 * 8) as i16);
    let v = vdupq_n_s16(dc);

    // First sqrdmulh
    let v = vqrdmulhq_s16(v, scale);
    // Extra sqrdmulh for rectangular (w==2*h or 2*w==h or w==4*h or 4*w==h)
    let v = vqrdmulhq_s16(v, scale);
    // shift=1 for these sizes
    let v = vrshrq_n_s16::<1>(v);
    // Second sqrdmulh
    let v = vqrdmulhq_s16(v, scale);
    // Final srshr >>4
    let v = vrshrq_n_s16::<4>(v);

    if w >= 8 {
        // 8 or 16 wide: process 8 pixels at a time
        for y in 0..h {
            let row_off = dst_base.wrapping_add_signed(y as isize * dst_stride);
            for half in 0..(w / 8) {
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
    } else {
        // 4 wide: process 2 rows at a time (pack 4+4 into uint8x8_t)
        for chunk in 0..(h / 2) {
            let r0_off = dst_base.wrapping_add_signed((chunk * 2) as isize * dst_stride);
            let r1_off = dst_base.wrapping_add_signed((chunk * 2 + 1) as isize * dst_stride);

            let mut bytes = [0u8; 8];
            bytes[0..4].copy_from_slice(&dst[r0_off..r0_off + 4]);
            bytes[4..8].copy_from_slice(&dst[r1_off..r1_off + 4]);
            let dst_u8 = safe_simd::vld1_u8(&bytes);

            let sum = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(v), dst_u8));
            let result = vqmovun_s16(sum);

            let mut out = [0u8; 8];
            safe_simd::vst1_u8(&mut out, result);
            dst[r0_off..r0_off + 4].copy_from_slice(&out[0..4]);
            dst[r1_off..r1_off + 4].copy_from_slice(&out[4..8]);
        }
    }
}

// ============================================================================
// Identity shift helpers for 8x16/16x8 row transforms
// ============================================================================

/// `identity_8x16_shift1`: sqrdmulh + srshr>>1 + sqadd.
///
/// Matches `identity_8x16_shift1` from itx.S lines 1350-1356.
/// Used for 16x4 identity row and 16x8 identity row transforms.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn identity_8x16_shift1_8h(v: &mut [int16x8_t]) {
    let scale = vdupq_n_s16(((5793 - 4096) * 2 * 8) as i16);
    for vi in v.iter_mut() {
        let t = vqrdmulhq_s16(*vi, scale);
        let t = vrshrq_n_s16::<1>(t);
        *vi = vqaddq_s16(*vi, t);
    }
}

// ============================================================================
// Generic 16x4 inverse transform (itx.S lines 1497-1561)
// ============================================================================

/// NEON implementation of 16x4 inverse transform add for 8bpc.
///
/// Mirrors `inv_txfm_add_16x4_neon` from itx.S lines 1497-1561.
///
/// Algorithm:
/// 1. Load 16 x int16x4_t coefficients (64 total)
/// 2. Apply 16-point row transform (.4h width)
/// 3. Combine pairs of 4h into 8h, srshr>>1
/// 4. Transpose 4x8h for each 8-column half
/// 5. Apply 4-point column transform (.8h width)
/// 6. Add to destination (two 8-wide halves)
/// 7. Clear coefficient buffer
#[cfg(target_arch = "aarch64")]
#[arcane]
pub(crate) fn inv_txfm_add_16x4_8bpc_neon(
    _token: Arm64,
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    coeff: &mut [i16],
    eob: i32,
    _bitdepth_max: i32,
    row_tx: TxType16,
    col_tx: RectTxType4,
) {
    // DC-only fast path
    if matches!(row_tx, TxType16::Dct) && matches!(col_tx, RectTxType4::Dct) && eob == 0 {
        dc_only_rect_8bpc(dst, dst_base, dst_stride, coeff, 16, 4);
        coeff[0..64].fill(0);
        return;
    }

    // Handle identity row transform separately (different load pattern)
    if matches!(row_tx, TxType16::Identity) {
        // Load as pairs of 4h into 8h vectors (interleaved layout)
        let mut lo = [vdupq_n_s16(0); 4];
        let mut hi = [vdupq_n_s16(0); 4];

        // Load first 4 columns: v16..v19 as .4h, then v16.d[1]..v19.d[1]
        for i in 0..4 {
            let low = safe_simd::vld1_s16(<&[i16; 4]>::try_from(&coeff[i * 4..i * 4 + 4]).unwrap());
            let high = safe_simd::vld1_s16(
                <&[i16; 4]>::try_from(&coeff[16 + i * 4..16 + i * 4 + 4]).unwrap(),
            );
            lo[i] = vcombine_s16(low, high);
        }
        // Load next 4 columns
        for i in 0..4 {
            let low = safe_simd::vld1_s16(
                <&[i16; 4]>::try_from(&coeff[32 + i * 4..32 + i * 4 + 4]).unwrap(),
            );
            let high = safe_simd::vld1_s16(
                <&[i16; 4]>::try_from(&coeff[48 + i * 4..48 + i * 4 + 4]).unwrap(),
            );
            hi[i] = vcombine_s16(low, high);
        }

        // Apply identity_8x16_shift1 to all 8 vectors
        let mut all = [vdupq_n_s16(0); 8];
        all[..4].copy_from_slice(&lo);
        all[4..].copy_from_slice(&hi);
        identity_8x16_shift1_8h(&mut all);
        lo.copy_from_slice(&all[..4]);
        hi.copy_from_slice(&all[4..]);

        // Transpose and apply column transform for left half
        let (t0, t1, t2, t3) = transpose_4x8h(lo[0], lo[1], lo[2], lo[3]);
        let (c0, c1, c2, c3) = apply_tx4_q(col_tx, t0, t1, t2, t3);
        add_to_dst_8x4_8bpc(dst, dst_base, dst_stride, c0, c1, c2, c3);

        // Transpose and apply column transform for right half
        let (t0, t1, t2, t3) = transpose_4x8h(hi[0], hi[1], hi[2], hi[3]);
        let (c0, c1, c2, c3) = apply_tx4_q(col_tx, t0, t1, t2, t3);
        add_to_dst_8x4_8bpc(dst, dst_base + 8, dst_stride, c0, c1, c2, c3);

        coeff[0..64].fill(0);
        return;
    }

    // Non-identity row transform: load 16 x int16x4_t
    let mut v = [vdup_n_s16(0); 16];
    for i in 0..16 {
        v[i] = safe_simd::vld1_s16(<&[i16; 4]>::try_from(&coeff[i * 4..i * 4 + 4]).unwrap());
    }

    // Apply 16-point row transform (.4h width)
    v = apply_tx16_4h(row_tx, v);

    // Combine pairs into 8h + srshr>>1 for left half (v[0..8] -> v16..v19)
    let mut left = [vdupq_n_s16(0); 4];
    for i in 0..4 {
        let combined = vcombine_s16(v[i], v[i + 4]);
        left[i] = vrshrq_n_s16::<1>(combined);
    }

    // Right half (v[8..16] -> v20..v23)
    let mut right = [vdupq_n_s16(0); 4];
    for i in 0..4 {
        let combined = vcombine_s16(v[8 + i], v[8 + i + 4]);
        right[i] = vrshrq_n_s16::<1>(combined);
    }

    // Transpose and apply column transform for left half
    let (t0, t1, t2, t3) = transpose_4x8h(left[0], left[1], left[2], left[3]);
    let (c0, c1, c2, c3) = apply_tx4_q(col_tx, t0, t1, t2, t3);
    add_to_dst_8x4_8bpc(dst, dst_base, dst_stride, c0, c1, c2, c3);

    // Transpose and apply column transform for right half
    let (t0, t1, t2, t3) = transpose_4x8h(right[0], right[1], right[2], right[3]);
    let (c0, c1, c2, c3) = apply_tx4_q(col_tx, t0, t1, t2, t3);
    add_to_dst_8x4_8bpc(dst, dst_base + 8, dst_stride, c0, c1, c2, c3);

    coeff[0..64].fill(0);
}

// ============================================================================
// Generic 4x16 inverse transform (itx.S lines 1564-1633)
// ============================================================================

/// NEON implementation of 4x16 inverse transform add for 8bpc.
///
/// Mirrors `inv_txfm_add_4x16_neon` from itx.S lines 1564-1633.
///
/// Algorithm:
/// 1. Load coefficients in two groups (low rows 0-7 and high rows 8-15)
/// 2. Apply 4-point row transform (.8h width) on each group
/// 3. srshr>>1
/// 4. Transpose 4x8h, split to 4h halves
/// 5. Apply 16-point column transform (.4h width)
/// 6. Add to 4x16 destination
/// 7. Clear coefficient buffer
#[cfg(target_arch = "aarch64")]
#[arcane]
pub(crate) fn inv_txfm_add_4x16_8bpc_neon(
    _token: Arm64,
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    coeff: &mut [i16],
    eob: i32,
    _bitdepth_max: i32,
    row_tx: RectTxType4,
    col_tx: TxType16,
    eob_half: i32,
) {
    // DC-only fast path
    if matches!(row_tx, RectTxType4::Dct) && matches!(col_tx, TxType16::Dct) && eob == 0 {
        dc_only_rect_8bpc(dst, dst_base, dst_stride, coeff, 4, 16);
        coeff[0..64].fill(0);
        return;
    }

    // Handle identity row transform separately
    let is_identity_row = matches!(row_tx, RectTxType4::Identity);

    // Process high half (rows 8-15) first, if eob is large enough
    let mut high_4h = [vdup_n_s16(0); 8]; // v24..v31 in assembly
    if eob >= eob_half {
        // Load 4 x int16x8_t from the high half
        // Coeff layout: 4 cols x 16 rows, col-major: coeff[row + col*16]
        // High half: rows 8-15 of each col, offset 8 shorts, stride 16 shorts
        let mut high = [vdupq_n_s16(0); 4];
        for i in 0..4 {
            let off = 8 + i * 16;
            high[i] = safe_simd::vld1q_s16(<&[i16; 8]>::try_from(&coeff[off..off + 8]).unwrap());
        }

        if is_identity_row {
            // Identity with shift1 scale: (5793-4096)*8
            let scale = vdupq_n_s16(((5793 - 4096) * 8) as i16);
            for vi in high.iter_mut() {
                let t = vqrdmulhq_s16(*vi, scale);
                let t = vrshrq_n_s16::<1>(t);
                *vi = vqaddq_s16(*vi, t);
            }
        } else {
            // Apply 4-point row transform (.8h width)
            let (h0, h1, h2, h3) = apply_tx4_q(row_tx, high[0], high[1], high[2], high[3]);
            high[0] = vrshrq_n_s16::<1>(h0);
            high[1] = vrshrq_n_s16::<1>(h1);
            high[2] = vrshrq_n_s16::<1>(h2);
            high[3] = vrshrq_n_s16::<1>(h3);
        }

        // Transpose 4x8h
        let (t0, t1, t2, t3) = transpose_4x8h(high[0], high[1], high[2], high[3]);

        // Split into 8 x 4h
        high_4h[0] = vget_low_s16(t0);
        high_4h[4] = vget_high_s16(t0);
        high_4h[1] = vget_low_s16(t1);
        high_4h[5] = vget_high_s16(t1);
        high_4h[2] = vget_low_s16(t2);
        high_4h[6] = vget_high_s16(t2);
        high_4h[3] = vget_low_s16(t3);
        high_4h[7] = vget_high_s16(t3);
    }

    // Process low half (rows 0-7)
    // Load from [x2] with stride 16 shorts: offsets 0, 16, 32, 48
    let mut low = [vdupq_n_s16(0); 4];
    for i in 0..4 {
        let off = i * 16;
        low[i] = safe_simd::vld1q_s16(<&[i16; 8]>::try_from(&coeff[off..off + 8]).unwrap());
    }

    if is_identity_row {
        let scale = vdupq_n_s16(((5793 - 4096) * 8) as i16);
        for vi in low.iter_mut() {
            let t = vqrdmulhq_s16(*vi, scale);
            let t = vrshrq_n_s16::<1>(t);
            *vi = vqaddq_s16(*vi, t);
        }
    } else {
        let (l0, l1, l2, l3) = apply_tx4_q(row_tx, low[0], low[1], low[2], low[3]);
        low[0] = vrshrq_n_s16::<1>(l0);
        low[1] = vrshrq_n_s16::<1>(l1);
        low[2] = vrshrq_n_s16::<1>(l2);
        low[3] = vrshrq_n_s16::<1>(l3);
    }

    // Transpose 4x8h
    let (t0, t1, t2, t3) = transpose_4x8h(low[0], low[1], low[2], low[3]);

    // Split into 8 x 4h
    let mut low_4h = [vdup_n_s16(0); 8];
    low_4h[0] = vget_low_s16(t0);
    low_4h[4] = vget_high_s16(t0);
    low_4h[1] = vget_low_s16(t1);
    low_4h[5] = vget_high_s16(t1);
    low_4h[2] = vget_low_s16(t2);
    low_4h[6] = vget_high_s16(t2);
    low_4h[3] = vget_low_s16(t3);
    low_4h[7] = vget_high_s16(t3);

    // Combine into 16 x 4h: low half = rows 0-7, high half = rows 8-15
    let mut all = [vdup_n_s16(0); 16];
    all[..8].copy_from_slice(&low_4h);
    all[8..].copy_from_slice(&high_4h);

    // Apply 16-point column transform (.4h width)
    all = apply_tx16_4h(col_tx, all);

    // Add to 4x16 destination
    add_to_dst_4x16_8bpc(dst, dst_base, dst_stride, all);

    coeff[0..64].fill(0);
}

// ============================================================================
// Generic 16x8 inverse transform (itx.S lines 1688-1731)
// ============================================================================

/// NEON implementation of 16x8 inverse transform add for 8bpc.
///
/// Mirrors `inv_txfm_add_16x8_neon` from itx.S lines 1688-1731.
///
/// Algorithm:
/// 1. Load 16 x int16x8_t coefficients (128 total)
/// 2. Scale all inputs by 2896*8
/// 3. Apply 16-point row transform (or identity) (.8h width)
/// 4. srshr>>1
/// 5. Transpose two 8x8h blocks
/// 6. Apply 8-point column transform (.8h width) on each half
/// 7. Add to destination
/// 8. Clear coefficient buffer
#[cfg(target_arch = "aarch64")]
#[arcane]
pub(crate) fn inv_txfm_add_16x8_8bpc_neon(
    _token: Arm64,
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    coeff: &mut [i16],
    eob: i32,
    _bitdepth_max: i32,
    row_tx: TxType16,
    col_tx: RectTxType8,
) {
    // DC-only fast path
    if matches!(row_tx, TxType16::Dct) && matches!(col_tx, RectTxType8::Dct) && eob == 0 {
        dc_only_rect_8bpc(dst, dst_base, dst_stride, coeff, 16, 8);
        coeff[0..128].fill(0);
        return;
    }

    // Load 16 x int16x8_t coefficients
    let mut v = [vdupq_n_s16(0); 16];
    for i in 0..16 {
        v[i] = safe_simd::vld1q_s16(<&[i16; 8]>::try_from(&coeff[i * 8..i * 8 + 8]).unwrap());
    }

    // Scale all inputs by 2896*8
    scale_input_8h_16(&mut v);

    if matches!(row_tx, TxType16::Identity) {
        // Identity row: apply identity_8x16_shift1
        identity_8x16_shift1_8h(&mut v);
    } else {
        // Apply 16-point row transform (.8h width)
        v = apply_tx16_q(row_tx, v);

        // srshr>>1
        for vi in v.iter_mut() {
            *vi = vrshrq_n_s16::<1>(*vi);
        }
    }

    // Transpose first 8x8 block (left half)
    let (t0, t1, t2, t3, t4, t5, t6, t7) =
        transpose_8x8h(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);

    // Apply 8-point column transform (.8h) on left half
    let (c0, c1, c2, c3, c4, c5, c6, c7) = apply_tx8_q(col_tx, t0, t1, t2, t3, t4, t5, t6, t7);

    // Add left half to destination
    add_to_dst_8x8_8bpc(dst, dst_base, dst_stride, [c0, c1, c2, c3, c4, c5, c6, c7]);

    // Transpose second 8x8 block (right half)
    let (t0, t1, t2, t3, t4, t5, t6, t7) =
        transpose_8x8h(v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15]);

    // Apply 8-point column transform (.8h) on right half
    let (c0, c1, c2, c3, c4, c5, c6, c7) = apply_tx8_q(col_tx, t0, t1, t2, t3, t4, t5, t6, t7);

    // Add right half to destination (offset by 8 columns)
    add_to_dst_8x8_8bpc(
        dst,
        dst_base + 8,
        dst_stride,
        [c0, c1, c2, c3, c4, c5, c6, c7],
    );

    coeff[0..128].fill(0);
}

// ============================================================================
// Generic 8x16 inverse transform (itx.S lines 1733-1807)
// ============================================================================

/// NEON implementation of 8x16 inverse transform add for 8bpc.
///
/// Mirrors `inv_txfm_add_8x16_neon` from itx.S lines 1733-1807.
///
/// Algorithm:
/// 1. Process coefficients in two groups (rows 0-7 and rows 8-15)
/// 2. Scale each group by 2896*8
/// 3. Apply 8-point row transform (or identity) (.8h width)
/// 4. srshr>>1 (or cancel for identity)
/// 5. Transpose each 8x8 block
/// 6. Combine halves: rows 0-7 from first group, rows 8-15 from second
/// 7. Apply 16-point column transform (.8h width)
/// 8. Add to 8x16 destination
/// 9. Clear coefficient buffer
#[cfg(target_arch = "aarch64")]
#[arcane]
pub(crate) fn inv_txfm_add_8x16_8bpc_neon(
    _token: Arm64,
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    coeff: &mut [i16],
    eob: i32,
    _bitdepth_max: i32,
    row_tx: RectTxType8,
    col_tx: TxType16,
    eob_half: i32,
) {
    // DC-only fast path
    if matches!(row_tx, RectTxType8::Dct) && matches!(col_tx, TxType16::Dct) && eob == 0 {
        dc_only_rect_8bpc(dst, dst_base, dst_stride, coeff, 8, 16);
        coeff[0..128].fill(0);
        return;
    }

    let is_identity_row = matches!(row_tx, RectTxType8::Identity);

    // Process high half (rows 8-15 of each column)
    // Coeff layout: 8 cols x 16 rows, col-major: coeff[row + col*16]
    // High half: offset 8 shorts, stride 16 shorts
    let mut high = [vdupq_n_s16(0); 8];
    if eob >= eob_half {
        for i in 0..8 {
            let off = 8 + i * 16;
            high[i] = safe_simd::vld1q_s16(<&[i16; 8]>::try_from(&coeff[off..off + 8]).unwrap());
        }

        scale_input_8h_8(&mut high);

        if is_identity_row {
            // Identity: scale_input + identity shl #1 cancel with srshr>>1
            // So we just keep the scaled values
        } else {
            let (h0, h1, h2, h3, h4, h5, h6, h7) = apply_tx8_q(
                row_tx, high[0], high[1], high[2], high[3], high[4], high[5], high[6], high[7],
            );
            high[0] = vrshrq_n_s16::<1>(h0);
            high[1] = vrshrq_n_s16::<1>(h1);
            high[2] = vrshrq_n_s16::<1>(h2);
            high[3] = vrshrq_n_s16::<1>(h3);
            high[4] = vrshrq_n_s16::<1>(h4);
            high[5] = vrshrq_n_s16::<1>(h5);
            high[6] = vrshrq_n_s16::<1>(h6);
            high[7] = vrshrq_n_s16::<1>(h7);
        }

        let (h0, h1, h2, h3, h4, h5, h6, h7) = transpose_8x8h(
            high[0], high[1], high[2], high[3], high[4], high[5], high[6], high[7],
        );
        high = [h0, h1, h2, h3, h4, h5, h6, h7];
    }

    // Process low half (rows 0-7 of each column)
    // Coeff layout: offset 0, stride 16 shorts
    let mut low = [vdupq_n_s16(0); 8];
    for i in 0..8 {
        let off = i * 16;
        low[i] = safe_simd::vld1q_s16(<&[i16; 8]>::try_from(&coeff[off..off + 8]).unwrap());
    }

    scale_input_8h_8(&mut low);

    if is_identity_row {
        // Identity: scale_input + identity shl #1 cancel with srshr>>1
    } else {
        let (l0, l1, l2, l3, l4, l5, l6, l7) = apply_tx8_q(
            row_tx, low[0], low[1], low[2], low[3], low[4], low[5], low[6], low[7],
        );
        low[0] = vrshrq_n_s16::<1>(l0);
        low[1] = vrshrq_n_s16::<1>(l1);
        low[2] = vrshrq_n_s16::<1>(l2);
        low[3] = vrshrq_n_s16::<1>(l3);
        low[4] = vrshrq_n_s16::<1>(l4);
        low[5] = vrshrq_n_s16::<1>(l5);
        low[6] = vrshrq_n_s16::<1>(l6);
        low[7] = vrshrq_n_s16::<1>(l7);
    }

    let (l0, l1, l2, l3, l4, l5, l6, l7) = transpose_8x8h(
        low[0], low[1], low[2], low[3], low[4], low[5], low[6], low[7],
    );
    low = [l0, l1, l2, l3, l4, l5, l6, l7];

    // Combine into 16 x int16x8_t: rows 0-7 from low, rows 8-15 from high
    let mut all = [vdupq_n_s16(0); 16];
    all[..8].copy_from_slice(&low);
    all[8..].copy_from_slice(&high);

    // Apply 16-point column transform (.8h width)
    all = apply_tx16_q(col_tx, all);

    // Add to 8x16 destination
    add_to_dst_8x16_8bpc(dst, dst_base, dst_stride, all);

    coeff[0..128].fill(0);
}

// ============================================================================
// Public entry points for 16x4 transforms (16 combinations)
// ============================================================================

macro_rules! def_rect_entry_16x4 {
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
            inv_txfm_add_16x4_8bpc_neon(
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

def_rect_entry_16x4!(
    inv_txfm_add_dct_dct_16x4_8bpc_neon_inner,
    TxType16::Dct,
    RectTxType4::Dct
);
def_rect_entry_16x4!(
    inv_txfm_add_identity_identity_16x4_8bpc_neon_inner,
    TxType16::Identity,
    RectTxType4::Identity
);
def_rect_entry_16x4!(
    inv_txfm_add_dct_adst_16x4_8bpc_neon_inner,
    TxType16::Dct,
    RectTxType4::Adst
);
def_rect_entry_16x4!(
    inv_txfm_add_dct_flipadst_16x4_8bpc_neon_inner,
    TxType16::Dct,
    RectTxType4::FlipAdst
);
def_rect_entry_16x4!(
    inv_txfm_add_dct_identity_16x4_8bpc_neon_inner,
    TxType16::Dct,
    RectTxType4::Identity
);
def_rect_entry_16x4!(
    inv_txfm_add_adst_dct_16x4_8bpc_neon_inner,
    TxType16::Adst,
    RectTxType4::Dct
);
def_rect_entry_16x4!(
    inv_txfm_add_adst_adst_16x4_8bpc_neon_inner,
    TxType16::Adst,
    RectTxType4::Adst
);
def_rect_entry_16x4!(
    inv_txfm_add_adst_flipadst_16x4_8bpc_neon_inner,
    TxType16::Adst,
    RectTxType4::FlipAdst
);
def_rect_entry_16x4!(
    inv_txfm_add_flipadst_dct_16x4_8bpc_neon_inner,
    TxType16::FlipAdst,
    RectTxType4::Dct
);
def_rect_entry_16x4!(
    inv_txfm_add_flipadst_adst_16x4_8bpc_neon_inner,
    TxType16::FlipAdst,
    RectTxType4::Adst
);
def_rect_entry_16x4!(
    inv_txfm_add_flipadst_flipadst_16x4_8bpc_neon_inner,
    TxType16::FlipAdst,
    RectTxType4::FlipAdst
);
def_rect_entry_16x4!(
    inv_txfm_add_identity_dct_16x4_8bpc_neon_inner,
    TxType16::Identity,
    RectTxType4::Dct
);
def_rect_entry_16x4!(
    inv_txfm_add_adst_identity_16x4_8bpc_neon_inner,
    TxType16::Adst,
    RectTxType4::Identity
);
def_rect_entry_16x4!(
    inv_txfm_add_flipadst_identity_16x4_8bpc_neon_inner,
    TxType16::FlipAdst,
    RectTxType4::Identity
);
def_rect_entry_16x4!(
    inv_txfm_add_identity_adst_16x4_8bpc_neon_inner,
    TxType16::Identity,
    RectTxType4::Adst
);
def_rect_entry_16x4!(
    inv_txfm_add_identity_flipadst_16x4_8bpc_neon_inner,
    TxType16::Identity,
    RectTxType4::FlipAdst
);

// ============================================================================
// Public entry points for 4x16 transforms (16 combinations)
// ============================================================================

macro_rules! def_rect_entry_4x16 {
    ($name:ident, $row:expr, $col:expr, $eob_half:literal) => {
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
            inv_txfm_add_4x16_8bpc_neon(
                token,
                dst,
                dst_base,
                dst_stride,
                coeff,
                eob,
                bitdepth_max,
                $row,
                $col,
                $eob_half,
            );
        }
    };
}

def_rect_entry_4x16!(
    inv_txfm_add_dct_dct_4x16_8bpc_neon_inner,
    RectTxType4::Dct,
    TxType16::Dct,
    29
);
def_rect_entry_4x16!(
    inv_txfm_add_identity_identity_4x16_8bpc_neon_inner,
    RectTxType4::Identity,
    TxType16::Identity,
    29
);
def_rect_entry_4x16!(
    inv_txfm_add_dct_adst_4x16_8bpc_neon_inner,
    RectTxType4::Dct,
    TxType16::Adst,
    29
);
def_rect_entry_4x16!(
    inv_txfm_add_dct_flipadst_4x16_8bpc_neon_inner,
    RectTxType4::Dct,
    TxType16::FlipAdst,
    29
);
def_rect_entry_4x16!(
    inv_txfm_add_dct_identity_4x16_8bpc_neon_inner,
    RectTxType4::Dct,
    TxType16::Identity,
    8
);
def_rect_entry_4x16!(
    inv_txfm_add_adst_dct_4x16_8bpc_neon_inner,
    RectTxType4::Adst,
    TxType16::Dct,
    29
);
def_rect_entry_4x16!(
    inv_txfm_add_adst_adst_4x16_8bpc_neon_inner,
    RectTxType4::Adst,
    TxType16::Adst,
    29
);
def_rect_entry_4x16!(
    inv_txfm_add_adst_flipadst_4x16_8bpc_neon_inner,
    RectTxType4::Adst,
    TxType16::FlipAdst,
    29
);
def_rect_entry_4x16!(
    inv_txfm_add_flipadst_dct_4x16_8bpc_neon_inner,
    RectTxType4::FlipAdst,
    TxType16::Dct,
    29
);
def_rect_entry_4x16!(
    inv_txfm_add_flipadst_adst_4x16_8bpc_neon_inner,
    RectTxType4::FlipAdst,
    TxType16::Adst,
    29
);
def_rect_entry_4x16!(
    inv_txfm_add_flipadst_flipadst_4x16_8bpc_neon_inner,
    RectTxType4::FlipAdst,
    TxType16::FlipAdst,
    29
);
def_rect_entry_4x16!(
    inv_txfm_add_identity_dct_4x16_8bpc_neon_inner,
    RectTxType4::Identity,
    TxType16::Dct,
    32
);
def_rect_entry_4x16!(
    inv_txfm_add_adst_identity_4x16_8bpc_neon_inner,
    RectTxType4::Adst,
    TxType16::Identity,
    8
);
def_rect_entry_4x16!(
    inv_txfm_add_flipadst_identity_4x16_8bpc_neon_inner,
    RectTxType4::FlipAdst,
    TxType16::Identity,
    8
);
def_rect_entry_4x16!(
    inv_txfm_add_identity_adst_4x16_8bpc_neon_inner,
    RectTxType4::Identity,
    TxType16::Adst,
    32
);
def_rect_entry_4x16!(
    inv_txfm_add_identity_flipadst_4x16_8bpc_neon_inner,
    RectTxType4::Identity,
    TxType16::FlipAdst,
    32
);

// ============================================================================
// Public entry points for 16x8 transforms (16 combinations)
// ============================================================================

macro_rules! def_rect_entry_16x8 {
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
            inv_txfm_add_16x8_8bpc_neon(
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

def_rect_entry_16x8!(
    inv_txfm_add_dct_dct_16x8_8bpc_neon_inner,
    TxType16::Dct,
    RectTxType8::Dct
);
def_rect_entry_16x8!(
    inv_txfm_add_identity_identity_16x8_8bpc_neon_inner,
    TxType16::Identity,
    RectTxType8::Identity
);
def_rect_entry_16x8!(
    inv_txfm_add_dct_adst_16x8_8bpc_neon_inner,
    TxType16::Dct,
    RectTxType8::Adst
);
def_rect_entry_16x8!(
    inv_txfm_add_dct_flipadst_16x8_8bpc_neon_inner,
    TxType16::Dct,
    RectTxType8::FlipAdst
);
def_rect_entry_16x8!(
    inv_txfm_add_dct_identity_16x8_8bpc_neon_inner,
    TxType16::Dct,
    RectTxType8::Identity
);
def_rect_entry_16x8!(
    inv_txfm_add_adst_dct_16x8_8bpc_neon_inner,
    TxType16::Adst,
    RectTxType8::Dct
);
def_rect_entry_16x8!(
    inv_txfm_add_adst_adst_16x8_8bpc_neon_inner,
    TxType16::Adst,
    RectTxType8::Adst
);
def_rect_entry_16x8!(
    inv_txfm_add_adst_flipadst_16x8_8bpc_neon_inner,
    TxType16::Adst,
    RectTxType8::FlipAdst
);
def_rect_entry_16x8!(
    inv_txfm_add_flipadst_dct_16x8_8bpc_neon_inner,
    TxType16::FlipAdst,
    RectTxType8::Dct
);
def_rect_entry_16x8!(
    inv_txfm_add_flipadst_adst_16x8_8bpc_neon_inner,
    TxType16::FlipAdst,
    RectTxType8::Adst
);
def_rect_entry_16x8!(
    inv_txfm_add_flipadst_flipadst_16x8_8bpc_neon_inner,
    TxType16::FlipAdst,
    RectTxType8::FlipAdst
);
def_rect_entry_16x8!(
    inv_txfm_add_identity_dct_16x8_8bpc_neon_inner,
    TxType16::Identity,
    RectTxType8::Dct
);
def_rect_entry_16x8!(
    inv_txfm_add_adst_identity_16x8_8bpc_neon_inner,
    TxType16::Adst,
    RectTxType8::Identity
);
def_rect_entry_16x8!(
    inv_txfm_add_flipadst_identity_16x8_8bpc_neon_inner,
    TxType16::FlipAdst,
    RectTxType8::Identity
);
def_rect_entry_16x8!(
    inv_txfm_add_identity_adst_16x8_8bpc_neon_inner,
    TxType16::Identity,
    RectTxType8::Adst
);
def_rect_entry_16x8!(
    inv_txfm_add_identity_flipadst_16x8_8bpc_neon_inner,
    TxType16::Identity,
    RectTxType8::FlipAdst
);

// ============================================================================
// Public entry points for 8x16 transforms (16 combinations)
// ============================================================================

macro_rules! def_rect_entry_8x16 {
    ($name:ident, $row:expr, $col:expr, $eob_half:literal) => {
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
            inv_txfm_add_8x16_8bpc_neon(
                token,
                dst,
                dst_base,
                dst_stride,
                coeff,
                eob,
                bitdepth_max,
                $row,
                $col,
                $eob_half,
            );
        }
    };
}

def_rect_entry_8x16!(
    inv_txfm_add_dct_dct_8x16_8bpc_neon_inner,
    RectTxType8::Dct,
    TxType16::Dct,
    43
);
def_rect_entry_8x16!(
    inv_txfm_add_identity_identity_8x16_8bpc_neon_inner,
    RectTxType8::Identity,
    TxType16::Identity,
    43
);
def_rect_entry_8x16!(
    inv_txfm_add_dct_adst_8x16_8bpc_neon_inner,
    RectTxType8::Dct,
    TxType16::Adst,
    43
);
def_rect_entry_8x16!(
    inv_txfm_add_dct_flipadst_8x16_8bpc_neon_inner,
    RectTxType8::Dct,
    TxType16::FlipAdst,
    43
);
def_rect_entry_8x16!(
    inv_txfm_add_dct_identity_8x16_8bpc_neon_inner,
    RectTxType8::Dct,
    TxType16::Identity,
    8
);
def_rect_entry_8x16!(
    inv_txfm_add_adst_dct_8x16_8bpc_neon_inner,
    RectTxType8::Adst,
    TxType16::Dct,
    43
);
def_rect_entry_8x16!(
    inv_txfm_add_adst_adst_8x16_8bpc_neon_inner,
    RectTxType8::Adst,
    TxType16::Adst,
    43
);
def_rect_entry_8x16!(
    inv_txfm_add_adst_flipadst_8x16_8bpc_neon_inner,
    RectTxType8::Adst,
    TxType16::FlipAdst,
    43
);
def_rect_entry_8x16!(
    inv_txfm_add_flipadst_dct_8x16_8bpc_neon_inner,
    RectTxType8::FlipAdst,
    TxType16::Dct,
    43
);
def_rect_entry_8x16!(
    inv_txfm_add_flipadst_adst_8x16_8bpc_neon_inner,
    RectTxType8::FlipAdst,
    TxType16::Adst,
    43
);
def_rect_entry_8x16!(
    inv_txfm_add_flipadst_flipadst_8x16_8bpc_neon_inner,
    RectTxType8::FlipAdst,
    TxType16::FlipAdst,
    43
);
def_rect_entry_8x16!(
    inv_txfm_add_identity_dct_8x16_8bpc_neon_inner,
    RectTxType8::Identity,
    TxType16::Dct,
    64
);
def_rect_entry_8x16!(
    inv_txfm_add_adst_identity_8x16_8bpc_neon_inner,
    RectTxType8::Adst,
    TxType16::Identity,
    8
);
def_rect_entry_8x16!(
    inv_txfm_add_flipadst_identity_8x16_8bpc_neon_inner,
    RectTxType8::FlipAdst,
    TxType16::Identity,
    8
);
def_rect_entry_8x16!(
    inv_txfm_add_identity_adst_8x16_8bpc_neon_inner,
    RectTxType8::Identity,
    TxType16::Adst,
    64
);
def_rect_entry_8x16!(
    inv_txfm_add_identity_flipadst_8x16_8bpc_neon_inner,
    RectTxType8::Identity,
    TxType16::FlipAdst,
    64
);
