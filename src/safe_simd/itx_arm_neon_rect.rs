//! Safe ARM NEON rectangular inverse transforms (4x8, 8x4)
//!
//! Port of the rectangular inverse transform functions from `src/arm/64/itx.S`
//! (lines 944-1030) to safe Rust NEON intrinsics. These handle blocks where
//! width != height, requiring scale_input normalization and mixed-width
//! transform stages.
//!
//! **8x4** (8 columns, 4 rows):
//!   - Load 8 x int16x4_t, scale_input, 8-point row transform (.4h),
//!     transpose, 4-point column transform (.8h), add to 8x4 destination.
//!
//! **4x8** (4 columns, 8 rows):
//!   - Load 4 x int16x8_t, scale_input, 4-point row transform (.8h),
//!     transpose, 8-point column transform (.4h), add to 4x8 destination.

#![allow(clippy::too_many_arguments)]
#![cfg_attr(not(any(feature = "asm", feature = "c-ffi", feature = "partial_asm")), forbid(unsafe_code))]
#![cfg_attr(any(feature = "asm", feature = "c-ffi", feature = "partial_asm"), deny(unsafe_code))]

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
use archmage::{Arm64, arcane, rite};

#[cfg(target_arch = "aarch64")]
use safe_unaligned_simd::aarch64 as safe_simd;

use super::itx_arm_neon_4x4::{iadst_4h, idct_4h, identity_4h};
use super::itx_arm_neon_8x8::{
    IADST8_COEFFS_V0, IADST8_COEFFS_V1, iadst_8_q, idct_4_q, idct_8_q, identity_8_q,
};
use super::itx_arm_neon_common::{IADST4_COEFFS, IDCT_COEFFS, IDENTITY_SCALE, transpose_4x4h};

// ============================================================================
// Scale input (itx.S lines 143-154)
// ============================================================================

/// Scale 4 int16x4_t vectors by 2896*8 using sqrdmulh.
///
/// Matches `scale_input .4h, v0.h[0], v0, v1, v2, v3`.
/// sqrdmulh: round(a * b * 2 / 65536) = round(a * b / 32768)
/// With b = 2896*8 = 23168, this computes round(a * 23168 / 32768).
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn scale_input_4h_4(
    v0: int16x4_t,
    v1: int16x4_t,
    v2: int16x4_t,
    v3: int16x4_t,
) -> (int16x4_t, int16x4_t, int16x4_t, int16x4_t) {
    let scale = vdup_n_s16((2896 * 8) as i16);
    (
        vqrdmulh_s16(v0, scale),
        vqrdmulh_s16(v1, scale),
        vqrdmulh_s16(v2, scale),
        vqrdmulh_s16(v3, scale),
    )
}

/// Scale 8 int16x4_t vectors by 2896*8 using sqrdmulh.
///
/// Matches `scale_input .4h, v0.h[0], v0..v7`.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn scale_input_4h_8(
    v0: int16x4_t,
    v1: int16x4_t,
    v2: int16x4_t,
    v3: int16x4_t,
    v4: int16x4_t,
    v5: int16x4_t,
    v6: int16x4_t,
    v7: int16x4_t,
) -> (
    int16x4_t,
    int16x4_t,
    int16x4_t,
    int16x4_t,
    int16x4_t,
    int16x4_t,
    int16x4_t,
    int16x4_t,
) {
    let scale = vdup_n_s16((2896 * 8) as i16);
    (
        vqrdmulh_s16(v0, scale),
        vqrdmulh_s16(v1, scale),
        vqrdmulh_s16(v2, scale),
        vqrdmulh_s16(v3, scale),
        vqrdmulh_s16(v4, scale),
        vqrdmulh_s16(v5, scale),
        vqrdmulh_s16(v6, scale),
        vqrdmulh_s16(v7, scale),
    )
}

/// Scale 4 int16x8_t vectors by 2896*8 using sqrdmulh.
///
/// Matches `scale_input .8h, v0.h[0], v0..v3`.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
pub(crate) fn scale_input_8h_4(
    v0: int16x8_t,
    v1: int16x8_t,
    v2: int16x8_t,
    v3: int16x8_t,
) -> (int16x8_t, int16x8_t, int16x8_t, int16x8_t) {
    let scale = vdupq_n_s16((2896 * 8) as i16);
    (
        vqrdmulhq_s16(v0, scale),
        vqrdmulhq_s16(v1, scale),
        vqrdmulhq_s16(v2, scale),
        vqrdmulhq_s16(v3, scale),
    )
}

// ============================================================================
// 4x8 transpose (util.S lines 254-264)
// ============================================================================

/// Transpose a 4-column x 8-row block stored in 4 int16x8_t vectors.
///
/// Matches the `transpose_4x8h` macro from util.S:
///   trn1/trn2 at .8h (16-bit pairs)
///   trn1/trn2 at .4s (32-bit groups)
///
/// Input: 4 vectors of 8 lanes each (4 columns, 8 rows).
/// Output: 4 vectors where each has been transposed — the result is
///   an 8-row x 4-col block where each 128-bit register holds two
///   4-element rows (low d = row N, high d = row N+4).
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
pub(crate) fn transpose_4x8h(
    r0: int16x8_t,
    r1: int16x8_t,
    r2: int16x8_t,
    r3: int16x8_t,
) -> (int16x8_t, int16x8_t, int16x8_t, int16x8_t) {
    // Level 1: trn at .8h (16-bit interleave)
    let t4 = vtrn1q_s16(r0, r1);
    let t5 = vtrn2q_s16(r0, r1);
    let t6 = vtrn1q_s16(r2, r3);
    let t7 = vtrn2q_s16(r2, r3);

    // Level 2: trn at .4s (32-bit interleave)
    let t4_32 = vreinterpretq_s32_s16(t4);
    let t5_32 = vreinterpretq_s32_s16(t5);
    let t6_32 = vreinterpretq_s32_s16(t6);
    let t7_32 = vreinterpretq_s32_s16(t7);

    let o0_32 = vtrn1q_s32(t4_32, t6_32);
    let o2_32 = vtrn2q_s32(t4_32, t6_32);
    let o1_32 = vtrn1q_s32(t5_32, t7_32);
    let o3_32 = vtrn2q_s32(t5_32, t7_32);

    (
        vreinterpretq_s16_s32(o0_32),
        vreinterpretq_s16_s32(o1_32),
        vreinterpretq_s16_s32(o2_32),
        vreinterpretq_s16_s32(o3_32),
    )
}

// ============================================================================
// 4-point ADST on int16x8_t (iadst_8x4 macro from itx.S lines 502-556)
// ============================================================================

/// 4-point inverse ADST on four int16x8_t vectors (8 transforms in parallel).
///
/// Uses IADST4 coefficients: [1321, 3803, 2482, 3344]
///
/// This is the `.8h` version of iadst_4h, processing both halves of
/// 128-bit registers using smull/smlal and smull2/smlal2.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn iadst_4_q(
    in0: int16x8_t,
    in1: int16x8_t,
    in2: int16x8_t,
    in3: int16x8_t,
) -> (int16x8_t, int16x8_t, int16x8_t, int16x8_t) {
    // Load IADST4 coefficients
    let coeffs = safe_simd::vld1q_s16(<&[i16; 8]>::try_from(&IADST4_COEFFS[..]).unwrap());

    // v3 = ssubl(in0, in2) — widen to i32
    // Need both lo and hi halves for .8h processing
    let in0_lo = vget_low_s16(in0);
    let in0_hi = vget_high_s16(in0);
    let in2_lo = vget_low_s16(in2);
    let in2_hi = vget_high_s16(in2);
    let in3_lo = vget_low_s16(in3);
    let in3_hi = vget_high_s16(in3);

    let v3_lo = vsubl_s16(in0_lo, in2_lo);
    let v3_hi = vsubl_s16(in0_hi, in2_hi);

    // v4 = in0*1321 + in2*3803 + in3*2482
    let v4_lo = vmull_laneq_s16::<0>(in0_lo, coeffs);
    let v4_lo = vmlal_laneq_s16::<1>(v4_lo, in2_lo, coeffs);
    let v4_lo = vmlal_laneq_s16::<2>(v4_lo, in3_lo, coeffs);
    let v4_hi = vmull_high_laneq_s16::<0>(in0, coeffs);
    let v4_hi = vmlal_high_laneq_s16::<1>(v4_hi, in2, coeffs);
    let v4_hi = vmlal_high_laneq_s16::<2>(v4_hi, in3, coeffs);

    // v7 = in1 * 3344
    let in1_lo = vget_low_s16(in1);
    let v7_lo = vmull_laneq_s16::<3>(in1_lo, coeffs);
    let v7_hi = vmull_high_laneq_s16::<3>(in1, coeffs);

    // v3 = (in0 - in2) + in3 sign-extended
    let v3_lo = vaddw_s16(v3_lo, in3_lo);
    let v3_hi = vaddw_s16(v3_hi, in3_hi);

    // v5 = in0*2482 - in2*1321 - in3*3803
    let v5_lo = vmull_laneq_s16::<2>(in0_lo, coeffs);
    let v5_lo = vmlsl_laneq_s16::<0>(v5_lo, in2_lo, coeffs);
    let v5_lo = vmlsl_laneq_s16::<1>(v5_lo, in3_lo, coeffs);
    let v5_hi = vmull_high_laneq_s16::<2>(in0, coeffs);
    let v5_hi = vmlsl_high_laneq_s16::<0>(v5_hi, in2, coeffs);
    let v5_hi = vmlsl_high_laneq_s16::<1>(v5_hi, in3, coeffs);

    // o2 = v3 * 3344 (32-bit multiply)
    // coeffs.s[2] = 3344 (from h[4]=3344, h[5]=0)
    let sinpi3 = vdupq_n_s32(3344);
    let o2_lo = vmulq_s32(v3_lo, sinpi3);
    let o2_hi = vmulq_s32(v3_hi, sinpi3);

    // o0 = v4 + v7
    let o0_lo = vaddq_s32(v4_lo, v7_lo);
    let o0_hi = vaddq_s32(v4_hi, v7_hi);

    // o1 = v5 + v7
    let o1_lo = vaddq_s32(v5_lo, v7_lo);
    let o1_hi = vaddq_s32(v5_hi, v7_hi);

    // o3 = v4 + v5 - v7
    let o3_lo = vaddq_s32(v4_lo, v5_lo);
    let o3_lo = vsubq_s32(o3_lo, v7_lo);
    let o3_hi = vaddq_s32(v4_hi, v5_hi);
    let o3_hi = vsubq_s32(o3_hi, v7_hi);

    // Narrow all with >>12 rounding
    let o0_narrow = vqrshrn_n_s32::<12>(o0_lo);
    let o0 = vqrshrn_high_n_s32::<12>(o0_narrow, o0_hi);

    let o1_narrow = vqrshrn_n_s32::<12>(o1_lo);
    let o1 = vqrshrn_high_n_s32::<12>(o1_narrow, o1_hi);

    let o2_narrow = vqrshrn_n_s32::<12>(o2_lo);
    let o2 = vqrshrn_high_n_s32::<12>(o2_narrow, o2_hi);

    let o3_narrow = vqrshrn_n_s32::<12>(o3_lo);
    let o3 = vqrshrn_high_n_s32::<12>(o3_narrow, o3_hi);

    (o0, o1, o2, o3)
}

/// 4-point identity transform on four int16x8_t vectors.
///
/// Matches `inv_identity_8h_x4_neon` from itx.S lines 582-593.
/// Multiplies by sqrt(2) using sqrdmulh + sqadd trick.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn identity_4_q(
    v0: int16x8_t,
    v1: int16x8_t,
    v2: int16x8_t,
    v3: int16x8_t,
) -> (int16x8_t, int16x8_t, int16x8_t, int16x8_t) {
    let scale = vdupq_n_s16(IDENTITY_SCALE);

    let h0 = vqrdmulhq_s16(v0, scale);
    let o0 = vqaddq_s16(v0, h0);

    let h1 = vqrdmulhq_s16(v1, scale);
    let o1 = vqaddq_s16(v1, h1);

    let h2 = vqrdmulhq_s16(v2, scale);
    let o2 = vqaddq_s16(v2, h2);

    let h3 = vqrdmulhq_s16(v3, scale);
    let o3 = vqaddq_s16(v3, h3);

    (o0, o1, o2, o3)
}

// ============================================================================
// 8-point transforms on int16x4_t (.4h width)
// ============================================================================

/// Widening multiply-accumulate for int16x4_t: s0*c0_lane + s1*c1_lane.
///
/// Like the .8h version in itx_arm_neon_8x8 but produces only one int32x4_t
/// (4 lanes, not 8).
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
#[inline(always)]
pub(crate) fn smull_smlal_4h(
    s0: int16x4_t,
    s1: int16x4_t,
    coeffs: int16x8_t,
    c0_lane: usize,
    c1_lane: usize,
) -> int32x4_t {
    match (c0_lane, c1_lane) {
        (0, 0) => {
            let r = vmull_laneq_s16::<0>(s0, coeffs);
            vmlal_laneq_s16::<0>(r, s1, coeffs)
        }
        (0, 1) => {
            let r = vmull_laneq_s16::<0>(s0, coeffs);
            vmlal_laneq_s16::<1>(r, s1, coeffs)
        }
        (1, 0) => {
            let r = vmull_laneq_s16::<1>(s0, coeffs);
            vmlal_laneq_s16::<0>(r, s1, coeffs)
        }
        (2, 3) => {
            let r = vmull_laneq_s16::<2>(s0, coeffs);
            vmlal_laneq_s16::<3>(r, s1, coeffs)
        }
        (3, 2) => {
            let r = vmull_laneq_s16::<3>(s0, coeffs);
            vmlal_laneq_s16::<2>(r, s1, coeffs)
        }
        (4, 5) => {
            let r = vmull_laneq_s16::<4>(s0, coeffs);
            vmlal_laneq_s16::<5>(r, s1, coeffs)
        }
        (5, 4) => {
            let r = vmull_laneq_s16::<5>(s0, coeffs);
            vmlal_laneq_s16::<4>(r, s1, coeffs)
        }
        (6, 7) => {
            let r = vmull_laneq_s16::<6>(s0, coeffs);
            vmlal_laneq_s16::<7>(r, s1, coeffs)
        }
        (7, 6) => {
            let r = vmull_laneq_s16::<7>(s0, coeffs);
            vmlal_laneq_s16::<6>(r, s1, coeffs)
        }
        _ => unreachable!(),
    }
}

/// Widening multiply-subtract for int16x4_t: s0*c0_lane - s1*c1_lane.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
#[inline(always)]
pub(crate) fn smull_smlsl_4h(
    s0: int16x4_t,
    s1: int16x4_t,
    coeffs: int16x8_t,
    c0_lane: usize,
    c1_lane: usize,
) -> int32x4_t {
    match (c0_lane, c1_lane) {
        (0, 0) => {
            let r = vmull_laneq_s16::<0>(s0, coeffs);
            vmlsl_laneq_s16::<0>(r, s1, coeffs)
        }
        (1, 0) => {
            let r = vmull_laneq_s16::<1>(s0, coeffs);
            vmlsl_laneq_s16::<0>(r, s1, coeffs)
        }
        (2, 3) => {
            let r = vmull_laneq_s16::<2>(s0, coeffs);
            vmlsl_laneq_s16::<3>(r, s1, coeffs)
        }
        (3, 2) => {
            let r = vmull_laneq_s16::<3>(s0, coeffs);
            vmlsl_laneq_s16::<2>(r, s1, coeffs)
        }
        (4, 5) => {
            let r = vmull_laneq_s16::<4>(s0, coeffs);
            vmlsl_laneq_s16::<5>(r, s1, coeffs)
        }
        (5, 4) => {
            let r = vmull_laneq_s16::<5>(s0, coeffs);
            vmlsl_laneq_s16::<4>(r, s1, coeffs)
        }
        (6, 7) => {
            let r = vmull_laneq_s16::<6>(s0, coeffs);
            vmlsl_laneq_s16::<7>(r, s1, coeffs)
        }
        (7, 6) => {
            let r = vmull_laneq_s16::<7>(s0, coeffs);
            vmlsl_laneq_s16::<6>(r, s1, coeffs)
        }
        _ => unreachable!(),
    }
}

/// 8-point inverse DCT on eight int16x4_t vectors (4 transforms in parallel).
///
/// Same algorithm as `idct_8_q` but operates on 64-bit (.4h) registers.
/// Matches `inv_dct_4h_x8_neon` from itx.S line 751.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
pub(crate) fn idct_8_4h(
    r0: int16x4_t,
    r1: int16x4_t,
    r2: int16x4_t,
    r3: int16x4_t,
    r4: int16x4_t,
    r5: int16x4_t,
    r6: int16x4_t,
    r7: int16x4_t,
) -> (
    int16x4_t,
    int16x4_t,
    int16x4_t,
    int16x4_t,
    int16x4_t,
    int16x4_t,
    int16x4_t,
    int16x4_t,
) {
    // Load IDCT coefficients
    let coeffs = safe_simd::vld1q_s16(<&[i16; 8]>::try_from(&IDCT_COEFFS[0..8]).unwrap());

    // 4-point DCT on even inputs (r0, r2, r4, r6)
    let idct4_coeffs = safe_simd::vld1_s16(<&[i16; 4]>::try_from(&IDCT_COEFFS[0..4]).unwrap());

    // idct_4 on r0, r2, r4, r6 (even-indexed)
    let v6 = vmull_lane_s16::<3>(r2, idct4_coeffs);
    let v6 = vmlal_lane_s16::<2>(v6, r6, idct4_coeffs);
    let v4 = vmull_lane_s16::<2>(r2, idct4_coeffs);
    let v4 = vmlsl_lane_s16::<3>(v4, r6, idct4_coeffs);
    let v2 = vmull_lane_s16::<0>(r0, idct4_coeffs);
    let v2 = vmlal_lane_s16::<0>(v2, r4, idct4_coeffs);
    let t3a = vqrshrn_n_s32::<12>(v6);
    let t2a = vqrshrn_n_s32::<12>(v4);
    let v4 = vmull_lane_s16::<0>(r0, idct4_coeffs);
    let v4 = vmlsl_lane_s16::<0>(v4, r4, idct4_coeffs);
    let t0 = vqrshrn_n_s32::<12>(v2);
    let t1 = vqrshrn_n_s32::<12>(v4);

    let e0 = vqadd_s16(t0, t3a);
    let e3 = vqsub_s16(t0, t3a);
    let e1 = vqadd_s16(t1, t2a);
    let e2 = vqsub_s16(t1, t2a);

    // Process odd inputs: r1, r3, r5, r7
    // t4a = (r1 * 799 - r7 * 4017 + 2048) >> 12
    let v2 = smull_smlsl_4h(r1, r7, coeffs, 4, 5);
    // t7a = (r1 * 4017 + r7 * 799 + 2048) >> 12
    let v4 = smull_smlal_4h(r1, r7, coeffs, 5, 4);
    // t5a = (r5 * 3406 - r3 * 2276 + 2048) >> 12
    let v6 = smull_smlsl_4h(r5, r3, coeffs, 6, 7);

    let t4a = vqrshrn_n_s32::<12>(v2);
    let t7a = vqrshrn_n_s32::<12>(v4);

    // t6a = (r5 * 2276 + r3 * 3406 + 2048) >> 12
    let v2 = smull_smlal_4h(r5, r3, coeffs, 7, 6);

    let t5a = vqrshrn_n_s32::<12>(v6);
    let t6a = vqrshrn_n_s32::<12>(v2);

    // Butterfly
    let t4 = vqadd_s16(t4a, t5a);
    let t5a_new = vqsub_s16(t4a, t5a);
    let t7 = vqadd_s16(t7a, t6a);
    let t6a_new = vqsub_s16(t7a, t6a);

    // Rotation: t5/t6 with coefficient 2896
    let v4 = smull_smlsl_4h(t6a_new, t5a_new, coeffs, 0, 0);
    let v6 = smull_smlal_4h(t6a_new, t5a_new, coeffs, 0, 0);
    let t5 = vqrshrn_n_s32::<12>(v4);
    let t6 = vqrshrn_n_s32::<12>(v6);

    // Final butterfly
    let out0 = vqadd_s16(e0, t7);
    let out7 = vqsub_s16(e0, t7);
    let out1 = vqadd_s16(e1, t6);
    let out6 = vqsub_s16(e1, t6);
    let out2 = vqadd_s16(e2, t5);
    let out5 = vqsub_s16(e2, t5);
    let out3 = vqadd_s16(e3, t4);
    let out4 = vqsub_s16(e3, t4);

    (out0, out1, out2, out3, out4, out5, out6, out7)
}

/// 8-point inverse ADST on eight int16x4_t vectors (4 transforms in parallel).
///
/// Same algorithm as `iadst_8_q` but operates on 64-bit (.4h) registers.
/// Matches `inv_adst_4h_x8_neon` from itx.S line 835.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn iadst_8_4h(
    in0: int16x4_t,
    in1: int16x4_t,
    in2: int16x4_t,
    in3: int16x4_t,
    in4: int16x4_t,
    in5: int16x4_t,
    in6: int16x4_t,
    in7: int16x4_t,
) -> (
    int16x4_t,
    int16x4_t,
    int16x4_t,
    int16x4_t,
    int16x4_t,
    int16x4_t,
    int16x4_t,
    int16x4_t,
) {
    let c0 = safe_simd::vld1q_s16(<&[i16; 8]>::try_from(&IADST8_COEFFS_V0[..]).unwrap());
    let c1 = safe_simd::vld1q_s16(<&[i16; 8]>::try_from(&IADST8_COEFFS_V1[..]).unwrap());

    // Stage 1: 4 rotation pairs
    let t0a = vqrshrn_n_s32::<12>(smull_smlal_4h(in7, in0, c0, 0, 1));
    let t1a = vqrshrn_n_s32::<12>(smull_smlsl_4h(in7, in0, c0, 1, 0));
    let t2a = vqrshrn_n_s32::<12>(smull_smlal_4h(in5, in2, c0, 2, 3));
    let t3a = vqrshrn_n_s32::<12>(smull_smlsl_4h(in5, in2, c0, 3, 2));
    let t4a = vqrshrn_n_s32::<12>(smull_smlal_4h(in3, in4, c0, 4, 5));
    let t5a = vqrshrn_n_s32::<12>(smull_smlsl_4h(in3, in4, c0, 5, 4));
    let t6a = vqrshrn_n_s32::<12>(smull_smlal_4h(in1, in6, c0, 6, 7));
    let t7a = vqrshrn_n_s32::<12>(smull_smlsl_4h(in1, in6, c0, 7, 6));

    // Stage 2: butterfly
    let t0 = vqadd_s16(t0a, t4a);
    let t4 = vqsub_s16(t0a, t4a);
    let t1 = vqadd_s16(t1a, t5a);
    let t5 = vqsub_s16(t1a, t5a);
    let t2 = vqadd_s16(t2a, t6a);
    let t6 = vqsub_s16(t2a, t6a);
    let t3 = vqadd_s16(t3a, t7a);
    let t7 = vqsub_s16(t3a, t7a);

    // Stage 3: rotations on t4/t5 and t7/t6
    let t4a = vqrshrn_n_s32::<12>(smull_smlal_4h(t4, t5, c1, 3, 2));
    let t5a = vqrshrn_n_s32::<12>(smull_smlsl_4h(t4, t5, c1, 2, 3));
    let t6a = vqrshrn_n_s32::<12>(smull_smlsl_4h(t7, t6, c1, 3, 2));
    let t7a = vqrshrn_n_s32::<12>(smull_smlal_4h(t7, t6, c1, 2, 3));

    // Stage 4: final butterflies
    let o0 = vqadd_s16(t0, t2);
    let x2 = vqsub_s16(t0, t2);
    let o7 = vqneg_s16(vqadd_s16(t1, t3));
    let x3 = vqsub_s16(t1, t3);

    let o1 = vqneg_s16(vqadd_s16(t4a, t6a));
    let x6 = vqsub_s16(t4a, t6a);
    let o6 = vqadd_s16(t5a, t7a);
    let x7 = vqsub_s16(t5a, t7a);

    // Stage 5: final rotations with 2896
    let o3_pre = vqrshrn_n_s32::<12>(smull_smlal_4h(x2, x3, c1, 0, 0));
    let o4 = vqrshrn_n_s32::<12>(smull_smlsl_4h(x2, x3, c1, 0, 0));
    let o5_pre = vqrshrn_n_s32::<12>(smull_smlsl_4h(x6, x7, c1, 0, 0));
    let o2 = vqrshrn_n_s32::<12>(smull_smlal_4h(x6, x7, c1, 0, 0));

    let o3 = vqneg_s16(o3_pre);
    let o5 = vqneg_s16(o5_pre);

    (o0, o1, o2, o3, o4, o5, o6, o7)
}

/// 8-point identity transform on eight int16x4_t vectors.
///
/// Matches `inv_identity_4h_x8_neon` from itx.S lines 857-866.
/// Simply saturating left shift by 1 (multiply by 2).
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn identity_8_4h(
    v0: int16x4_t,
    v1: int16x4_t,
    v2: int16x4_t,
    v3: int16x4_t,
    v4: int16x4_t,
    v5: int16x4_t,
    v6: int16x4_t,
    v7: int16x4_t,
) -> (
    int16x4_t,
    int16x4_t,
    int16x4_t,
    int16x4_t,
    int16x4_t,
    int16x4_t,
    int16x4_t,
    int16x4_t,
) {
    (
        vqshl_n_s16::<1>(v0),
        vqshl_n_s16::<1>(v1),
        vqshl_n_s16::<1>(v2),
        vqshl_n_s16::<1>(v3),
        vqshl_n_s16::<1>(v4),
        vqshl_n_s16::<1>(v5),
        vqshl_n_s16::<1>(v6),
        vqshl_n_s16::<1>(v7),
    )
}

// ============================================================================
// Transform type enums and apply helpers
// ============================================================================

/// Transform type for 4-point (used on int16x4_t or int16x8_t).
#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
pub(crate) enum RectTxType4 {
    Dct,
    Adst,
    FlipAdst,
    Identity,
}

/// Transform type for 8-point (used on int16x4_t or int16x8_t).
#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
pub(crate) enum RectTxType8 {
    Dct,
    Adst,
    FlipAdst,
    Identity,
}

/// Apply 4-point transform on 4 int16x8_t vectors (8h width).
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
pub(crate) fn apply_tx4_q(
    tx: RectTxType4,
    v0: int16x8_t,
    v1: int16x8_t,
    v2: int16x8_t,
    v3: int16x8_t,
) -> (int16x8_t, int16x8_t, int16x8_t, int16x8_t) {
    match tx {
        RectTxType4::Dct => idct_4_q(v0, v1, v2, v3),
        RectTxType4::Adst => iadst_4_q(v0, v1, v2, v3),
        RectTxType4::FlipAdst => {
            let (o0, o1, o2, o3) = iadst_4_q(v0, v1, v2, v3);
            (o3, o2, o1, o0)
        }
        RectTxType4::Identity => identity_4_q(v0, v1, v2, v3),
    }
}

/// Apply 4-point transform on 4 int16x4_t vectors (4h width).
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn apply_tx4_4h(
    tx: RectTxType4,
    v0: int16x4_t,
    v1: int16x4_t,
    v2: int16x4_t,
    v3: int16x4_t,
) -> (int16x4_t, int16x4_t, int16x4_t, int16x4_t) {
    match tx {
        RectTxType4::Dct => idct_4h(v0, v1, v2, v3),
        RectTxType4::Adst => iadst_4h(v0, v1, v2, v3),
        RectTxType4::FlipAdst => {
            let (o0, o1, o2, o3) = iadst_4h(v0, v1, v2, v3);
            (o3, o2, o1, o0)
        }
        RectTxType4::Identity => identity_4h(v0, v1, v2, v3),
    }
}

/// Apply 8-point transform on 8 int16x8_t vectors (8h width).
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
pub(crate) fn apply_tx8_q(
    tx: RectTxType8,
    v0: int16x8_t,
    v1: int16x8_t,
    v2: int16x8_t,
    v3: int16x8_t,
    v4: int16x8_t,
    v5: int16x8_t,
    v6: int16x8_t,
    v7: int16x8_t,
) -> (
    int16x8_t,
    int16x8_t,
    int16x8_t,
    int16x8_t,
    int16x8_t,
    int16x8_t,
    int16x8_t,
    int16x8_t,
) {
    match tx {
        RectTxType8::Dct => idct_8_q(v0, v1, v2, v3, v4, v5, v6, v7),
        RectTxType8::Adst => iadst_8_q(v0, v1, v2, v3, v4, v5, v6, v7),
        RectTxType8::FlipAdst => {
            let (o0, o1, o2, o3, o4, o5, o6, o7) = iadst_8_q(v0, v1, v2, v3, v4, v5, v6, v7);
            (o7, o6, o5, o4, o3, o2, o1, o0)
        }
        RectTxType8::Identity => identity_8_q(v0, v1, v2, v3, v4, v5, v6, v7),
    }
}

/// Apply 8-point transform on 8 int16x4_t vectors (4h width).
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
pub(crate) fn apply_tx8_4h(
    tx: RectTxType8,
    v0: int16x4_t,
    v1: int16x4_t,
    v2: int16x4_t,
    v3: int16x4_t,
    v4: int16x4_t,
    v5: int16x4_t,
    v6: int16x4_t,
    v7: int16x4_t,
) -> (
    int16x4_t,
    int16x4_t,
    int16x4_t,
    int16x4_t,
    int16x4_t,
    int16x4_t,
    int16x4_t,
    int16x4_t,
) {
    match tx {
        RectTxType8::Dct => idct_8_4h(v0, v1, v2, v3, v4, v5, v6, v7),
        RectTxType8::Adst => iadst_8_4h(v0, v1, v2, v3, v4, v5, v6, v7),
        RectTxType8::FlipAdst => {
            let (o0, o1, o2, o3, o4, o5, o6, o7) = iadst_8_4h(v0, v1, v2, v3, v4, v5, v6, v7);
            (o7, o6, o5, o4, o3, o2, o1, o0)
        }
        RectTxType8::Identity => identity_8_4h(v0, v1, v2, v3, v4, v5, v6, v7),
    }
}

// ============================================================================
// Add-to-destination for 8x4 (8 pixels wide, 4 rows)
// ============================================================================

/// Add transform output to 8x4 destination pixels for 8bpc.
///
/// Mirrors `load_add_store_8x4` from itx.S lines 211-221.
/// Takes 4 int16x8_t vectors (one per row), applies srshr>>4,
/// adds to u8 destination with unsigned saturation, and stores.
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

        // srshr by 4
        let shifted = vrshrq_n_s16::<4>(row);

        // Load 8 u8 destination pixels
        let dst_bytes: [u8; 8] = dst[row_off..row_off + 8].try_into().unwrap();
        let dst_u8 = safe_simd::vld1_u8(&dst_bytes);

        // uaddw + sqxtun
        let sum = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(shifted), dst_u8));
        let result = vqmovun_s16(sum);

        let mut out = [0u8; 8];
        safe_simd::vst1_u8(&mut out, result);
        dst[row_off..row_off + 8].copy_from_slice(&out);
    }
}

// ============================================================================
// Add-to-destination for 4x8 (4 pixels wide, 8 rows)
// ============================================================================

/// Add transform output to 4x8 destination pixels for 8bpc.
///
/// Mirrors `load_add_store_4x8` from itx.S lines 264-275.
/// Takes 8 int16x4_t vectors (one per row), combines pairs into
/// int16x8_t, applies srshr>>4, adds to u8 destination.
///
/// The assembly combines pairs of 4h vectors into 8h by inserting
/// the upper half: ins v16.d[1], v17.d[0] etc. Then processes
/// two rows at a time from the 8h vector.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn add_to_dst_4x8_8bpc(
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    v0: int16x4_t,
    v1: int16x4_t,
    v2: int16x4_t,
    v3: int16x4_t,
    v4: int16x4_t,
    v5: int16x4_t,
    v6: int16x4_t,
    v7: int16x4_t,
) {
    // Process pairs of rows using combined int16x8_t
    let pairs = [
        vcombine_s16(v0, v1),
        vcombine_s16(v2, v3),
        vcombine_s16(v4, v5),
        vcombine_s16(v6, v7),
    ];

    for (pair_idx, &pair) in pairs.iter().enumerate() {
        // srshr by 4
        let shifted = vrshrq_n_s16::<4>(pair);

        let row0_off = dst_base.wrapping_add_signed((pair_idx * 2) as isize * stride);
        let row1_off = dst_base.wrapping_add_signed((pair_idx * 2 + 1) as isize * stride);

        // Load 4+4 u8 pixels packed into uint8x8_t
        let mut dst_bytes = [0u8; 8];
        dst_bytes[0..4].copy_from_slice(&dst[row0_off..row0_off + 4]);
        dst_bytes[4..8].copy_from_slice(&dst[row1_off..row1_off + 4]);
        let dst_u8 = safe_simd::vld1_u8(&dst_bytes);

        // uaddw + sqxtun
        let sum = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(shifted), dst_u8));
        let result = vqmovun_s16(sum);

        let mut out = [0u8; 8];
        safe_simd::vst1_u8(&mut out, result);
        dst[row0_off..row0_off + 4].copy_from_slice(&out[0..4]);
        dst[row1_off..row1_off + 4].copy_from_slice(&out[4..8]);
    }
}

// ============================================================================
// DC-only fast paths (idct_dc macro, itx.S lines 277-295)
// ============================================================================

/// DC-only fast path for DCT_DCT 8x4 with eob=0.
///
/// For rectangular blocks (w == 2*h or 2*w == h), an extra sqrdmulh
/// is applied. For 8x4: w=8 == 2*h=8, so the extra multiply IS applied.
/// shift=0 for 8x4, so no intermediate srshr.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn dc_only_8x4_8bpc(dst: &mut [u8], dst_base: usize, dst_stride: isize, coeff: &mut [i16]) {
    let dc = coeff[0];
    coeff[0] = 0;

    let scale = vdupq_n_s16((2896 * 8) as i16);
    let v16 = vdupq_n_s16(dc);

    // First sqrdmulh
    let v16 = vqrdmulhq_s16(v16, scale);
    // Extra sqrdmulh for rectangular (w == 2*h)
    let v16 = vqrdmulhq_s16(v16, scale);
    // shift=0, skip srshr
    // Second sqrdmulh
    let v16 = vqrdmulhq_s16(v16, scale);
    // Final srshr >>4
    let v16 = vrshrq_n_s16::<4>(v16);

    // Add to 4 rows of 8 pixels each
    for i in 0..4 {
        let row_off = dst_base.wrapping_add_signed(i as isize * dst_stride);
        let dst_bytes: [u8; 8] = dst[row_off..row_off + 8].try_into().unwrap();
        let dst_u8 = safe_simd::vld1_u8(&dst_bytes);
        let sum = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(v16), dst_u8));
        let result = vqmovun_s16(sum);
        let mut out = [0u8; 8];
        safe_simd::vst1_u8(&mut out, result);
        dst[row_off..row_off + 8].copy_from_slice(&out);
    }
}

/// DC-only fast path for DCT_DCT 4x8 with eob=0.
///
/// For 4x8: 2*w=8 == h=8, so the extra multiply IS applied.
/// shift=0 for 4x8, so no intermediate srshr.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn dc_only_4x8_8bpc(dst: &mut [u8], dst_base: usize, dst_stride: isize, coeff: &mut [i16]) {
    let dc = coeff[0];
    coeff[0] = 0;

    let scale = vdupq_n_s16((2896 * 8) as i16);
    let v16 = vdupq_n_s16(dc);

    // First sqrdmulh
    let v16 = vqrdmulhq_s16(v16, scale);
    // Extra sqrdmulh for rectangular (2*w == h)
    let v16 = vqrdmulhq_s16(v16, scale);
    // shift=0, skip srshr
    // Second sqrdmulh
    let v16 = vqrdmulhq_s16(v16, scale);
    // Final srshr >>4
    let v16 = vrshrq_n_s16::<4>(v16);

    // Add to 8 rows of 4 pixels each (using the DC w4 pattern)
    // Process 4 rows at a time
    for chunk in 0..2 {
        let base = chunk * 4;
        let r0_off = dst_base.wrapping_add_signed((base) as isize * dst_stride);
        let r1_off = dst_base.wrapping_add_signed((base + 1) as isize * dst_stride);
        let r2_off = dst_base.wrapping_add_signed((base + 2) as isize * dst_stride);
        let r3_off = dst_base.wrapping_add_signed((base + 3) as isize * dst_stride);

        // Load 4 bytes per row, pack into uint8x8_t
        let mut bytes01 = [0u8; 8];
        bytes01[0..4].copy_from_slice(&dst[r0_off..r0_off + 4]);
        bytes01[4..8].copy_from_slice(&dst[r1_off..r1_off + 4]);
        let d01 = safe_simd::vld1_u8(&bytes01);

        let mut bytes23 = [0u8; 8];
        bytes23[0..4].copy_from_slice(&dst[r2_off..r2_off + 4]);
        bytes23[4..8].copy_from_slice(&dst[r3_off..r3_off + 4]);
        let d23 = safe_simd::vld1_u8(&bytes23);

        let sum01 = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(v16), d01));
        let sum23 = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(v16), d23));

        let r01 = vqmovun_s16(sum01);
        let r23 = vqmovun_s16(sum23);

        let mut out01 = [0u8; 8];
        safe_simd::vst1_u8(&mut out01, r01);
        dst[r0_off..r0_off + 4].copy_from_slice(&out01[0..4]);
        dst[r1_off..r1_off + 4].copy_from_slice(&out01[4..8]);

        let mut out23 = [0u8; 8];
        safe_simd::vst1_u8(&mut out23, r23);
        dst[r2_off..r2_off + 4].copy_from_slice(&out23[0..4]);
        dst[r3_off..r3_off + 4].copy_from_slice(&out23[4..8]);
    }
}

// ============================================================================
// Generic 8x4 inverse transform (itx.S lines 944-969)
// ============================================================================

/// NEON implementation of 8x4 inverse transform add for 8bpc.
///
/// Mirrors `inv_txfm_add_8x4_neon` from itx.S lines 944-969.
///
/// Algorithm:
/// 1. Load 8 x int16x4_t coefficients (32 total)
/// 2. Scale all inputs by 2896*8 via sqrdmulh
/// 3. Apply row transform (8-point on .4h width)
/// 4. Transpose: two 4x4h transposes, combine halves into .8h
/// 5. Apply column transform (4-point on .8h width)
/// 6. Add to 8x4 destination with srshr>>4
/// 7. Clear coefficient buffer
#[cfg(target_arch = "aarch64")]
#[arcane]
pub(crate) fn inv_txfm_add_8x4_8bpc_neon(
    _token: Arm64,
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    coeff: &mut [i16],
    eob: i32,
    _bitdepth_max: i32,
    row_tx: RectTxType8,
    col_tx: RectTxType4,
) {
    // DC-only fast path for DCT_DCT with eob=0
    if matches!(row_tx, RectTxType8::Dct) && matches!(col_tx, RectTxType4::Dct) && eob == 0 {
        dc_only_8x4_8bpc(dst, dst_base, dst_stride, coeff);
        coeff[0..32].fill(0);
        return;
    }

    // Step 1: Load 8 x 4h coefficients
    // Layout: 8 columns of 4 rows each, stored sequentially
    let v16 = safe_simd::vld1_s16(<&[i16; 4]>::try_from(&coeff[0..4]).unwrap());
    let v17 = safe_simd::vld1_s16(<&[i16; 4]>::try_from(&coeff[4..8]).unwrap());
    let v18 = safe_simd::vld1_s16(<&[i16; 4]>::try_from(&coeff[8..12]).unwrap());
    let v19 = safe_simd::vld1_s16(<&[i16; 4]>::try_from(&coeff[12..16]).unwrap());
    let v20 = safe_simd::vld1_s16(<&[i16; 4]>::try_from(&coeff[16..20]).unwrap());
    let v21 = safe_simd::vld1_s16(<&[i16; 4]>::try_from(&coeff[20..24]).unwrap());
    let v22 = safe_simd::vld1_s16(<&[i16; 4]>::try_from(&coeff[24..28]).unwrap());
    let v23 = safe_simd::vld1_s16(<&[i16; 4]>::try_from(&coeff[28..32]).unwrap());

    // Step 2: Scale all inputs by 2896*8
    let (v16, v17, v18, v19, v20, v21, v22, v23) =
        scale_input_4h_8(v16, v17, v18, v19, v20, v21, v22, v23);

    // Step 3: Row transform (8-point on .4h width)
    let (v16, v17, v18, v19, v20, v21, v22, v23) =
        apply_tx8_4h(row_tx, v16, v17, v18, v19, v20, v21, v22, v23);

    // Step 4: Transpose
    // Two 4x4h transposes (top 4 vectors and bottom 4 vectors)
    let (v16, v17, v18, v19) = transpose_4x4h(v16, v17, v18, v19);
    let (v20, v21, v22, v23) = transpose_4x4h(v20, v21, v22, v23);

    // Combine halves: ins v16.d[1], v20.d[0] etc.
    let c0 = vcombine_s16(v16, v20);
    let c1 = vcombine_s16(v17, v21);
    let c2 = vcombine_s16(v18, v22);
    let c3 = vcombine_s16(v19, v23);

    // Step 5: Column transform (4-point on .8h width)
    let (c0, c1, c2, c3) = apply_tx4_q(col_tx, c0, c1, c2, c3);

    // Step 6: Add to 8x4 destination with >>4 shift
    add_to_dst_8x4_8bpc(dst, dst_base, dst_stride, c0, c1, c2, c3);

    // Step 7: Clear coefficients
    coeff[0..32].fill(0);
}

// ============================================================================
// Generic 4x8 inverse transform (itx.S lines 971-995)
// ============================================================================

/// NEON implementation of 4x8 inverse transform add for 8bpc.
///
/// Mirrors `inv_txfm_add_4x8_neon` from itx.S lines 971-995.
///
/// Algorithm:
/// 1. Load 4 x int16x8_t coefficients (32 total)
/// 2. Scale all inputs by 2896*8 via sqrdmulh
/// 3. Apply row transform (4-point on .8h width)
/// 4. Transpose 4x8: produces 4 x int16x8_t where low d = row N, high d = row N+4
/// 5. Split into 8 x int16x4_t
/// 6. Apply column transform (8-point on .4h width)
/// 7. Add to 4x8 destination with srshr>>4
/// 8. Clear coefficient buffer
#[cfg(target_arch = "aarch64")]
#[arcane]
pub(crate) fn inv_txfm_add_4x8_8bpc_neon(
    _token: Arm64,
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    coeff: &mut [i16],
    eob: i32,
    _bitdepth_max: i32,
    row_tx: RectTxType4,
    col_tx: RectTxType8,
) {
    // DC-only fast path for DCT_DCT with eob=0
    if matches!(row_tx, RectTxType4::Dct) && matches!(col_tx, RectTxType8::Dct) && eob == 0 {
        dc_only_4x8_8bpc(dst, dst_base, dst_stride, coeff);
        coeff[0..32].fill(0);
        return;
    }

    // Step 1: Load 4 x 8h coefficients
    let v16 = safe_simd::vld1q_s16(<&[i16; 8]>::try_from(&coeff[0..8]).unwrap());
    let v17 = safe_simd::vld1q_s16(<&[i16; 8]>::try_from(&coeff[8..16]).unwrap());
    let v18 = safe_simd::vld1q_s16(<&[i16; 8]>::try_from(&coeff[16..24]).unwrap());
    let v19 = safe_simd::vld1q_s16(<&[i16; 8]>::try_from(&coeff[24..32]).unwrap());

    // Step 2: Scale all inputs by 2896*8
    let (v16, v17, v18, v19) = scale_input_8h_4(v16, v17, v18, v19);

    // Step 3: Row transform (4-point on .8h width)
    let (v16, v17, v18, v19) = apply_tx4_q(row_tx, v16, v17, v18, v19);

    // Step 4: Transpose 4x8
    let (v16, v17, v18, v19) = transpose_4x8h(v16, v17, v18, v19);

    // Step 5: Split into 8 x int16x4_t
    // After transpose_4x8h, each 128-bit register contains two 4-element rows:
    //   v16.d[0] = row 0, v16.d[1] = row 4
    //   v17.d[0] = row 1, v17.d[1] = row 5
    //   v18.d[0] = row 2, v18.d[1] = row 6
    //   v19.d[0] = row 3, v19.d[1] = row 7
    // Assembly: ins v20.d[0], v16.d[1] etc.
    let r0 = vget_low_s16(v16);
    let r4 = vget_high_s16(v16);
    let r1 = vget_low_s16(v17);
    let r5 = vget_high_s16(v17);
    let r2 = vget_low_s16(v18);
    let r6 = vget_high_s16(v18);
    let r3 = vget_low_s16(v19);
    let r7 = vget_high_s16(v19);

    // Step 6: Column transform (8-point on .4h width)
    let (r0, r1, r2, r3, r4, r5, r6, r7) = apply_tx8_4h(col_tx, r0, r1, r2, r3, r4, r5, r6, r7);

    // Step 7: Add to 4x8 destination with >>4 shift
    add_to_dst_4x8_8bpc(dst, dst_base, dst_stride, r0, r1, r2, r3, r4, r5, r6, r7);

    // Step 8: Clear coefficients
    coeff[0..32].fill(0);
}

// ============================================================================
// Public entry points for 8x4 transforms (16 combinations)
// ============================================================================

macro_rules! def_rect_entry_8x4 {
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
            inv_txfm_add_8x4_8bpc_neon(
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

def_rect_entry_8x4!(
    inv_txfm_add_dct_dct_8x4_8bpc_neon_inner,
    RectTxType8::Dct,
    RectTxType4::Dct
);
def_rect_entry_8x4!(
    inv_txfm_add_identity_identity_8x4_8bpc_neon_inner,
    RectTxType8::Identity,
    RectTxType4::Identity
);
def_rect_entry_8x4!(
    inv_txfm_add_dct_adst_8x4_8bpc_neon_inner,
    RectTxType8::Dct,
    RectTxType4::Adst
);
def_rect_entry_8x4!(
    inv_txfm_add_dct_flipadst_8x4_8bpc_neon_inner,
    RectTxType8::Dct,
    RectTxType4::FlipAdst
);
def_rect_entry_8x4!(
    inv_txfm_add_dct_identity_8x4_8bpc_neon_inner,
    RectTxType8::Dct,
    RectTxType4::Identity
);
def_rect_entry_8x4!(
    inv_txfm_add_adst_dct_8x4_8bpc_neon_inner,
    RectTxType8::Adst,
    RectTxType4::Dct
);
def_rect_entry_8x4!(
    inv_txfm_add_adst_adst_8x4_8bpc_neon_inner,
    RectTxType8::Adst,
    RectTxType4::Adst
);
def_rect_entry_8x4!(
    inv_txfm_add_adst_flipadst_8x4_8bpc_neon_inner,
    RectTxType8::Adst,
    RectTxType4::FlipAdst
);
def_rect_entry_8x4!(
    inv_txfm_add_flipadst_dct_8x4_8bpc_neon_inner,
    RectTxType8::FlipAdst,
    RectTxType4::Dct
);
def_rect_entry_8x4!(
    inv_txfm_add_flipadst_adst_8x4_8bpc_neon_inner,
    RectTxType8::FlipAdst,
    RectTxType4::Adst
);
def_rect_entry_8x4!(
    inv_txfm_add_flipadst_flipadst_8x4_8bpc_neon_inner,
    RectTxType8::FlipAdst,
    RectTxType4::FlipAdst
);
def_rect_entry_8x4!(
    inv_txfm_add_identity_dct_8x4_8bpc_neon_inner,
    RectTxType8::Identity,
    RectTxType4::Dct
);
def_rect_entry_8x4!(
    inv_txfm_add_adst_identity_8x4_8bpc_neon_inner,
    RectTxType8::Adst,
    RectTxType4::Identity
);
def_rect_entry_8x4!(
    inv_txfm_add_flipadst_identity_8x4_8bpc_neon_inner,
    RectTxType8::FlipAdst,
    RectTxType4::Identity
);
def_rect_entry_8x4!(
    inv_txfm_add_identity_adst_8x4_8bpc_neon_inner,
    RectTxType8::Identity,
    RectTxType4::Adst
);
def_rect_entry_8x4!(
    inv_txfm_add_identity_flipadst_8x4_8bpc_neon_inner,
    RectTxType8::Identity,
    RectTxType4::FlipAdst
);

// ============================================================================
// Public entry points for 4x8 transforms (16 combinations)
// ============================================================================

macro_rules! def_rect_entry_4x8 {
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
            inv_txfm_add_4x8_8bpc_neon(
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

def_rect_entry_4x8!(
    inv_txfm_add_dct_dct_4x8_8bpc_neon_inner,
    RectTxType4::Dct,
    RectTxType8::Dct
);
def_rect_entry_4x8!(
    inv_txfm_add_identity_identity_4x8_8bpc_neon_inner,
    RectTxType4::Identity,
    RectTxType8::Identity
);
def_rect_entry_4x8!(
    inv_txfm_add_dct_adst_4x8_8bpc_neon_inner,
    RectTxType4::Dct,
    RectTxType8::Adst
);
def_rect_entry_4x8!(
    inv_txfm_add_dct_flipadst_4x8_8bpc_neon_inner,
    RectTxType4::Dct,
    RectTxType8::FlipAdst
);
def_rect_entry_4x8!(
    inv_txfm_add_dct_identity_4x8_8bpc_neon_inner,
    RectTxType4::Dct,
    RectTxType8::Identity
);
def_rect_entry_4x8!(
    inv_txfm_add_adst_dct_4x8_8bpc_neon_inner,
    RectTxType4::Adst,
    RectTxType8::Dct
);
def_rect_entry_4x8!(
    inv_txfm_add_adst_adst_4x8_8bpc_neon_inner,
    RectTxType4::Adst,
    RectTxType8::Adst
);
def_rect_entry_4x8!(
    inv_txfm_add_adst_flipadst_4x8_8bpc_neon_inner,
    RectTxType4::Adst,
    RectTxType8::FlipAdst
);
def_rect_entry_4x8!(
    inv_txfm_add_flipadst_dct_4x8_8bpc_neon_inner,
    RectTxType4::FlipAdst,
    RectTxType8::Dct
);
def_rect_entry_4x8!(
    inv_txfm_add_flipadst_adst_4x8_8bpc_neon_inner,
    RectTxType4::FlipAdst,
    RectTxType8::Adst
);
def_rect_entry_4x8!(
    inv_txfm_add_flipadst_flipadst_4x8_8bpc_neon_inner,
    RectTxType4::FlipAdst,
    RectTxType8::FlipAdst
);
def_rect_entry_4x8!(
    inv_txfm_add_identity_dct_4x8_8bpc_neon_inner,
    RectTxType4::Identity,
    RectTxType8::Dct
);
def_rect_entry_4x8!(
    inv_txfm_add_adst_identity_4x8_8bpc_neon_inner,
    RectTxType4::Adst,
    RectTxType8::Identity
);
def_rect_entry_4x8!(
    inv_txfm_add_flipadst_identity_4x8_8bpc_neon_inner,
    RectTxType4::FlipAdst,
    RectTxType8::Identity
);
def_rect_entry_4x8!(
    inv_txfm_add_identity_adst_4x8_8bpc_neon_inner,
    RectTxType4::Identity,
    RectTxType8::Adst
);
def_rect_entry_4x8!(
    inv_txfm_add_identity_flipadst_4x8_8bpc_neon_inner,
    RectTxType4::Identity,
    RectTxType8::FlipAdst
);
