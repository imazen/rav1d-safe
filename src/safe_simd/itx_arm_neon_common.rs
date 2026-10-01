//! Common NEON utilities for inverse transforms
//!
//! Shared constants, transpose, and add-to-destination helpers used by all
//! transform sizes. Ported from assembly macros in `src/arm/64/itx.S`.

#![allow(clippy::too_many_arguments)]
#![cfg_attr(not(any(feature = "asm", feature = "c-ffi", feature = "partial_asm")), forbid(unsafe_code))]
#![cfg_attr(any(feature = "asm", feature = "c-ffi", feature = "partial_asm"), deny(unsafe_code))]

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
use archmage::rite;

#[cfg(target_arch = "aarch64")]
use safe_unaligned_simd::aarch64 as safe_simd;

// ============================================================================
// Coefficient tables from assembly (itx.S lines 62-116)
// ============================================================================

/// IDCT coefficients: [2896, 2896*8, 1567, 3784, 799, 4017, 3406, 2276, ...]
pub(crate) const IDCT_COEFFS: [i16; 32] = [
    2896,
    (2896 * 8) as i16, // 23168 fits in i16 (-32768..32767)
    1567,
    3784,
    // idct8
    799,
    4017,
    3406,
    2276,
    // idct16 (first 4)
    401,
    4076,
    3166,
    2598,
    // idct16 (second 4)
    1931,
    3612,
    3920,
    1189,
    // idct32 (first 4)
    201,
    4091,
    3035,
    2751,
    // idct32 (second 4)
    1751,
    3703,
    3857,
    1380,
    // idct32 (third 4)
    995,
    3973,
    3513,
    2106,
    // idct32 (fourth 4)
    2440,
    3290,
    4052,
    601,
];

/// IADST4 coefficients: [1321, 3803, 2482, 3344, 3344, 0]
/// Note: h[4..5] can be interpreted as s[2] = 3344 (32-bit)
pub(crate) const IADST4_COEFFS: [i16; 8] = [1321, 3803, 2482, 3344, 3344, 0, 0, 0];

/// The identity scale factor: (5793 - 4096) * 8 = 13576
/// Used by the identity transform: x * 5793/4096 via sqrdmulh + sqadd trick.
pub(crate) const IDENTITY_SCALE: i16 = ((5793 - 4096) * 8) as i16;

// ============================================================================
// Transpose
// ============================================================================

/// 4x4 transpose for int16x4_t vectors.
///
/// Matches the `transpose_4x4h` macro from util.S lines 230-239.
/// Uses trn1/trn2 at 16-bit level, then trn1/trn2 at 32-bit level.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
pub(crate) fn transpose_4x4h(
    r0: int16x4_t,
    r1: int16x4_t,
    r2: int16x4_t,
    r3: int16x4_t,
) -> (int16x4_t, int16x4_t, int16x4_t, int16x4_t) {
    // First level: interleave at 16-bit granularity
    let t4 = vtrn1_s16(r0, r1);
    let t5 = vtrn2_s16(r0, r1);
    let t6 = vtrn1_s16(r2, r3);
    let t7 = vtrn2_s16(r2, r3);

    // Second level: interleave at 32-bit granularity
    let t4_32 = vreinterpret_s32_s16(t4);
    let t5_32 = vreinterpret_s32_s16(t5);
    let t6_32 = vreinterpret_s32_s16(t6);
    let t7_32 = vreinterpret_s32_s16(t7);

    let o0_32 = vtrn1_s32(t4_32, t6_32);
    let o2_32 = vtrn2_s32(t4_32, t6_32);
    let o1_32 = vtrn1_s32(t5_32, t7_32);
    let o3_32 = vtrn2_s32(t5_32, t7_32);

    let o0 = vreinterpret_s16_s32(o0_32);
    let o1 = vreinterpret_s16_s32(o1_32);
    let o2 = vreinterpret_s16_s32(o2_32);
    let o3 = vreinterpret_s16_s32(o3_32);

    (o0, o1, o2, o3)
}

// ============================================================================
// Add-to-destination helper (itx_4x4_end pattern)
// ============================================================================

/// Add transform output to destination pixels for 4x4 block (8bpc).
///
/// Mirrors the `L(itx_4x4_end)` label in itx.S lines 652-663.
/// The transform output is in 4 int16x4_t vectors (one per row).
/// We combine them into pairs of int16x8_t, apply rounding shift >>4,
/// add to u8 destination pixels with unsigned saturation, and store.
///
/// When `shift` is true, applies srshr>>4 (for non-WHT transforms).
/// WHT transforms skip the shift (it's already absorbed).
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
pub(crate) fn add_to_dst_4x4_8bpc(
    dst: &mut [u8],
    dst_base: usize,
    stride: isize,
    v16: int16x4_t,
    v17: int16x4_t,
    v18: int16x4_t,
    v19: int16x4_t,
    apply_shift: bool,
) {
    // Combine rows into wide pairs: v16_wide = [row0 | row1], v18_wide = [row2 | row3]
    let v16_wide = vcombine_s16(v16, v17);
    let v18_wide = vcombine_s16(v18, v19);

    // Apply rounding shift >>4 if needed (non-WHT transforms)
    let v16_wide = if apply_shift {
        vrshrq_n_s16::<4>(v16_wide)
    } else {
        v16_wide
    };
    let v18_wide = if apply_shift {
        vrshrq_n_s16::<4>(v18_wide)
    } else {
        v18_wide
    };

    // Load 4 rows of 4 destination pixels
    let row0_off = dst_base;
    let row1_off = dst_base.wrapping_add_signed(stride);
    let row2_off = dst_base.wrapping_add_signed(stride * 2);
    let row3_off = dst_base.wrapping_add_signed(stride * 3);

    // Load rows 0+1 packed into one uint8x8_t
    let mut dst_bytes_01 = [0u8; 8];
    dst_bytes_01[0..4].copy_from_slice(&dst[row0_off..row0_off + 4]);
    dst_bytes_01[4..8].copy_from_slice(&dst[row1_off..row1_off + 4]);
    let v0 = safe_simd::vld1_u8(&dst_bytes_01);

    // Load rows 2+3 packed into one uint8x8_t
    let mut dst_bytes_23 = [0u8; 8];
    dst_bytes_23[0..4].copy_from_slice(&dst[row2_off..row2_off + 4]);
    dst_bytes_23[4..8].copy_from_slice(&dst[row3_off..row3_off + 4]);
    let v1 = safe_simd::vld1_u8(&dst_bytes_23);

    // uaddw: zero-extend u8 to u16, add to i16 (reinterpreted as u16)
    let sum_01 = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(v16_wide), v0));
    let sum_23 = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(v18_wide), v1));

    // sqxtun: signed saturating narrow to unsigned u8
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
}
