
use archmage::{Desktop64, Server64, SimdToken, arcane, rite};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::include::common::bitdepth::AsPrimitive;
use crate::include::common::bitdepth::BitDepth;
use crate::include::common::bitdepth::DynCoef;
use crate::include::common::bitdepth::DynPixel;
use crate::include::common::intops::iclip;
use crate::include::dav1d::picture::PicOffset;
use crate::src::ffi_safe::FFISafe;
use crate::src::safe_simd::pixel_access::Flex;
use crate::src::safe_simd::pixel_access::{
    loadi32, loadi64, loadu_128, loadu_256, loadu_512, storei32, storei64, storeu_128, storeu_256,
    storeu_512,
};
use std::ffi::c_int;
use std::num::NonZeroUsize;
use std::slice;

// ============================================================================
// CONSTANTS
// ============================================================================

// Trig coefficients for DCT (12-bit fixed point, scaled by 4096)
// cos(π/8) * 4096 ≈ 3784
// sin(π/8) * 4096 ≈ 1567
// cos(π/4) * sqrt(2) = sqrt(2) ≈ 181/128 = 1.414

const SQRT2_BITS: i32 = 8;
const SQRT2_HALF: i32 = 181; // sqrt(2) * 128

// ============================================================================
// SHARED SIMD MACROS
// ============================================================================

/// 8x8 i32 in-register transpose, AVX2.
///
/// Input: `$a` is an `[__m256i; 8]` expression where each lane holds an
/// 8-element i32 column.
/// Output: an `[__m256i; 8]` array where each lane holds an 8-element i32 row.
///
/// Cost: 24 safe intrinsic calls (8 `unpacklo_epi32` + 8 `unpackhi_epi32`
/// folded with `unpacklo_epi64`/`unpackhi_epi64` and 8 `permute2x128_si256`).
/// All computation intrinsics are safe (Rust 1.93+); produces byte-identical
/// asm to the hand-inlined sequence — verified via cargo asm.
///
/// Used by the `simd_row_*_8bpc_8rows` helpers and any other site that needs
/// to flip an 8-wide column-major block to row-major before scalar-store.
#[cfg(target_arch = "x86_64")]
macro_rules! transpose_8x8_i32 {
    ($a:expr) => {{
        let __cols: [__m256i; 8] = $a;
        let __t0 = _mm256_unpacklo_epi32(__cols[0], __cols[1]);
        let __t1 = _mm256_unpackhi_epi32(__cols[0], __cols[1]);
        let __t2 = _mm256_unpacklo_epi32(__cols[2], __cols[3]);
        let __t3 = _mm256_unpackhi_epi32(__cols[2], __cols[3]);
        let __t4 = _mm256_unpacklo_epi32(__cols[4], __cols[5]);
        let __t5 = _mm256_unpackhi_epi32(__cols[4], __cols[5]);
        let __t6 = _mm256_unpacklo_epi32(__cols[6], __cols[7]);
        let __t7 = _mm256_unpackhi_epi32(__cols[6], __cols[7]);
        let __u0 = _mm256_unpacklo_epi64(__t0, __t2);
        let __u1 = _mm256_unpackhi_epi64(__t0, __t2);
        let __u2 = _mm256_unpacklo_epi64(__t1, __t3);
        let __u3 = _mm256_unpackhi_epi64(__t1, __t3);
        let __u4 = _mm256_unpacklo_epi64(__t4, __t6);
        let __u5 = _mm256_unpackhi_epi64(__t4, __t6);
        let __u6 = _mm256_unpacklo_epi64(__t5, __t7);
        let __u7 = _mm256_unpackhi_epi64(__t5, __t7);
        let __r0 = _mm256_permute2x128_si256::<0x20>(__u0, __u4);
        let __r1 = _mm256_permute2x128_si256::<0x20>(__u1, __u5);
        let __r2 = _mm256_permute2x128_si256::<0x20>(__u2, __u6);
        let __r3 = _mm256_permute2x128_si256::<0x20>(__u3, __u7);
        let __r4 = _mm256_permute2x128_si256::<0x31>(__u0, __u4);
        let __r5 = _mm256_permute2x128_si256::<0x31>(__u1, __u5);
        let __r6 = _mm256_permute2x128_si256::<0x31>(__u2, __u6);
        let __r7 = _mm256_permute2x128_si256::<0x31>(__u3, __u7);
        [__r0, __r1, __r2, __r3, __r4, __r5, __r6, __r7]
    }};
}

/// dav1d ITX_MUL2X_PACK equivalent — bit-exact to the C reference path.
///
/// Computes `dst_lo = lo * coef_a + hi * coef_b` for each i32 lane, where
/// `(lo, hi)` are the two i16 halves of every i32 lane of `$paired`. This is
/// exactly what `pmaddwd` does, so the macro packs `coef_a` (low 16) and
/// `coef_b` (high 16) into a 32-bit broadcast and emits a single `pmaddwd` +
/// `paddd` + `psrad`. The rounded value matches dav1d's row pass arithmetic
/// `(p1 + p2 + 2048) >> 12`.
///
/// Input shape: `$paired` is `_mm256_unpacklo_epi16(a, b)` (or `_unpackhi`),
/// i.e. each 32-bit lane is `(a_word << 16) | b_word` — call sites construct
/// this directly. `$coef_a`/`$coef_b` are signed 16-bit values written as
/// `i32` literals (we mask to u16 when packing so negatives round-trip
/// correctly).
///
/// Output: i32 lanes post-shift. Pack to i16 via `_mm256_packs_epi32` to
/// resume the i16 row pass.
///
/// All operands are computation intrinsics — safe since Rust 1.93+.
/// `$shift` is a literal because `_mm256_srai_epi32` is const-generic.
#[cfg(target_arch = "x86_64")]
#[allow(unused_macros)]
macro_rules! itx_mul2x_pack {
    ($paired:expr, $coef_a:expr, $coef_b:expr, $rnd:expr, $shift:literal) => {{
        let __coef_a: i32 = $coef_a as i32;
        let __coef_b: i32 = $coef_b as i32;
        let __packed_coef = _mm256_set1_epi32(
            ((__coef_a as u32) & 0xFFFF) as i32 | (((__coef_b as u32) & 0xFFFF) << 16) as i32,
        );
        let __prod = _mm256_madd_epi16($paired, __packed_coef);
        let __rounded = _mm256_add_epi32(__prod, _mm256_set1_epi32($rnd));
        _mm256_srai_epi32::<$shift>(__rounded)
    }};
}

// ============================================================================
// 4x4 DCT_DCT - Full 2D SIMD Transform (8bpc)
// ============================================================================

/// Full 2D DCT_DCT 4x4 inverse transform with add-to-destination
/// Uses AVX2 to process all 4 rows simultaneously
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_4x4_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    use crate::src::safe_simd::pixel_access::{loadi32, storei32, storeu_128};
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();

    let row0 = _mm_set_epi32(
        coeff[12] as i32,
        coeff[8] as i32,
        coeff[4] as i32,
        coeff[0] as i32,
    );
    let row1 = _mm_set_epi32(
        coeff[13] as i32,
        coeff[9] as i32,
        coeff[5] as i32,
        coeff[1] as i32,
    );
    let row2 = _mm_set_epi32(
        coeff[14] as i32,
        coeff[10] as i32,
        coeff[6] as i32,
        coeff[2] as i32,
    );
    let row3 = _mm_set_epi32(
        coeff[15] as i32,
        coeff[11] as i32,
        coeff[7] as i32,
        coeff[3] as i32,
    );

    let rows01 = _mm256_set_m128i(row1, row0);
    let rows23 = _mm256_set_m128i(row3, row2);

    // 8bpc clip ranges: i16::MIN..i16::MAX for both row and col
    let row_clip_min = if bitdepth_max == 255 {
        i16::MIN as i32
    } else {
        (!bitdepth_max) << 7
    };
    let row_clip_max = !row_clip_min;
    let col_clip_min = if bitdepth_max == 255 {
        i16::MIN as i32
    } else {
        (!bitdepth_max) << 5
    };
    let col_clip_max = !col_clip_min;

    let (rows01_out, rows23_out) =
        dct4_2rows_avx2(_token, rows01, rows23, row_clip_min, row_clip_max);

    let r0 = _mm256_castsi256_si128(rows01_out);
    let r1 = _mm256_extracti128_si256(rows01_out, 1);
    let r2 = _mm256_castsi256_si128(rows23_out);
    let r3 = _mm256_extracti128_si256(rows23_out, 1);

    let t01_lo = _mm_unpacklo_epi32(r0, r1);
    let t01_hi = _mm_unpackhi_epi32(r0, r1);
    let t23_lo = _mm_unpacklo_epi32(r2, r3);
    let t23_hi = _mm_unpackhi_epi32(r2, r3);

    let col0 = _mm_unpacklo_epi64(t01_lo, t23_lo);
    let col1 = _mm_unpackhi_epi64(t01_lo, t23_lo);
    let col2 = _mm_unpacklo_epi64(t01_hi, t23_hi);
    let col3 = _mm_unpackhi_epi64(t01_hi, t23_hi);

    // Intermediate clamp (shift=0 for 4x4, so just clamp to col_clip range)
    let cmin = _mm_set1_epi32(col_clip_min);
    let cmax = _mm_set1_epi32(col_clip_max);
    let col0 = _mm_max_epi32(_mm_min_epi32(col0, cmax), cmin);
    let col1 = _mm_max_epi32(_mm_min_epi32(col1, cmax), cmin);
    let col2 = _mm_max_epi32(_mm_min_epi32(col2, cmax), cmin);
    let col3 = _mm_max_epi32(_mm_min_epi32(col3, cmax), cmin);

    let cols01 = _mm256_set_m128i(col1, col0);
    let cols23 = _mm256_set_m128i(col3, col2);

    let (cols01_out, cols23_out) =
        dct4_2rows_avx2(_token, cols01, cols23, col_clip_min, col_clip_max);

    let rnd = _mm256_set1_epi32(8);
    let cols01_scaled = _mm256_srai_epi32(_mm256_add_epi32(cols01_out, rnd), 4);
    let cols23_scaled = _mm256_srai_epi32(_mm256_add_epi32(cols23_out, rnd), 4);

    let c0 = _mm256_castsi256_si128(cols01_scaled);
    let c1 = _mm256_extracti128_si256(cols01_scaled, 1);
    let c2 = _mm256_castsi256_si128(cols23_scaled);
    let c3 = _mm256_extracti128_si256(cols23_scaled, 1);

    let u01_lo = _mm_unpacklo_epi32(c0, c1);
    let u01_hi = _mm_unpackhi_epi32(c0, c1);
    let u23_lo = _mm_unpacklo_epi32(c2, c3);
    let u23_hi = _mm_unpackhi_epi32(c2, c3);

    let final0 = _mm_unpacklo_epi64(u01_lo, u23_lo);
    let final1 = _mm_unpackhi_epi64(u01_lo, u23_lo);
    let final2 = _mm_unpacklo_epi64(u01_hi, u23_hi);
    let final3 = _mm_unpackhi_epi64(u01_hi, u23_hi);

    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi16(bitdepth_max as i16);

    // Row 0
    let d0 = loadi32!(&dst[..4]);
    let d0_16 = _mm_unpacklo_epi8(d0, zero);
    let d0_32 = _mm_cvtepi16_epi32(d0_16);
    let sum0 = _mm_add_epi32(d0_32, final0);
    let sum0_16 = _mm_packs_epi32(sum0, sum0);
    let sum0_clamped = _mm_max_epi16(_mm_min_epi16(sum0_16, max_val), zero);
    let sum0_8 = _mm_packus_epi16(sum0_clamped, sum0_clamped);
    storei32!(&mut dst[..4], sum0_8);

    // Row 1
    let off1 = dst_stride;
    let d1 = loadi32!(&dst[off1..off1 + 4]);
    let d1_16 = _mm_unpacklo_epi8(d1, zero);
    let d1_32 = _mm_cvtepi16_epi32(d1_16);
    let sum1 = _mm_add_epi32(d1_32, final1);
    let sum1_16 = _mm_packs_epi32(sum1, sum1);
    let sum1_clamped = _mm_max_epi16(_mm_min_epi16(sum1_16, max_val), zero);
    let sum1_8 = _mm_packus_epi16(sum1_clamped, sum1_clamped);
    storei32!(&mut dst[off1..off1 + 4], sum1_8);

    // Row 2
    let off2 = dst_stride * 2;
    let d2 = loadi32!(&dst[off2..off2 + 4]);
    let d2_16 = _mm_unpacklo_epi8(d2, zero);
    let d2_32 = _mm_cvtepi16_epi32(d2_16);
    let sum2 = _mm_add_epi32(d2_32, final2);
    let sum2_16 = _mm_packs_epi32(sum2, sum2);
    let sum2_clamped = _mm_max_epi16(_mm_min_epi16(sum2_16, max_val), zero);
    let sum2_8 = _mm_packus_epi16(sum2_clamped, sum2_clamped);
    storei32!(&mut dst[off2..off2 + 4], sum2_8);

    // Row 3
    let off3 = dst_stride * 3;
    let d3 = loadi32!(&dst[off3..off3 + 4]);
    let d3_16 = _mm_unpacklo_epi8(d3, zero);
    let d3_32 = _mm_cvtepi16_epi32(d3_16);
    let sum3 = _mm_add_epi32(d3_32, final3);
    let sum3_16 = _mm_packs_epi32(sum3, sum3);
    let sum3_clamped = _mm_max_epi16(_mm_min_epi16(sum3_16, max_val), zero);
    let sum3_8 = _mm_packus_epi16(sum3_clamped, sum3_clamped);
    storei32!(&mut dst[off3..off3 + 4], sum3_8);

    // Clear coefficients
    let coeff_bytes = zerocopy::IntoBytes::as_mut_bytes(&mut *coeff);
    storeu_128!(
        <&mut [u8; 16]>::try_from(&mut coeff_bytes[..16]).unwrap(),
        _mm_setzero_si128()
    );
    storeu_128!(
        <&mut [u8; 16]>::try_from(&mut coeff_bytes[16..32]).unwrap(),
        _mm_setzero_si128()
    );
}

/// DCT4 butterfly on 2 rows packed in __m256i
/// Each 128-bit lane contains one row: [in0, in1, in2, in3] as i32
/// Outputs are clipped to [clip_min, clip_max] to match scalar behavior.
/// `#[rite]` so it inlines into matching-feature `#[arcane]` callers (zero call cost).
#[cfg(target_arch = "x86_64")]
#[rite]
fn dct4_2rows_avx2(
    _token: Desktop64,
    rows01: __m256i,
    rows23: __m256i,
    clip_min: i32,
    clip_max: i32,
) -> (__m256i, __m256i) {
    // DCT4: t0 = (in0 + in2) * 181 + 128 >> 8
    //       t1 = (in0 - in2) * 181 + 128 >> 8
    //       t2 = (in1 * 1567 - in3 * (3784-4096) + 2048 >> 12) - in3
    //       t3 = (in1 * (3784-4096) + in3 * 1567 + 2048 >> 12) + in1

    let sqrt2 = _mm256_set1_epi32(181);
    let rnd8 = _mm256_set1_epi32(128);
    let c1567 = _mm256_set1_epi32(1567);
    let c_312 = _mm256_set1_epi32(3784 - 4096);
    let rnd12 = _mm256_set1_epi32(2048);

    // Process rows01
    let in0_01 = _mm256_shuffle_epi32(rows01, 0b00_00_00_00);
    let in1_01 = _mm256_shuffle_epi32(rows01, 0b01_01_01_01);
    let in2_01 = _mm256_shuffle_epi32(rows01, 0b10_10_10_10);
    let in3_01 = _mm256_shuffle_epi32(rows01, 0b11_11_11_11);

    // t0 = (in0 + in2) * 181 + 128 >> 8
    let sum02_01 = _mm256_add_epi32(in0_01, in2_01);
    let t0_01 = _mm256_srai_epi32(
        _mm256_add_epi32(_mm256_mullo_epi32(sum02_01, sqrt2), rnd8),
        8,
    );

    // t1 = (in0 - in2) * 181 + 128 >> 8
    let diff02_01 = _mm256_sub_epi32(in0_01, in2_01);
    let t1_01 = _mm256_srai_epi32(
        _mm256_add_epi32(_mm256_mullo_epi32(diff02_01, sqrt2), rnd8),
        8,
    );

    // t2 = (in1 * 1567 - in3 * (3784-4096) + 2048 >> 12) - in3
    let mul1_1567_01 = _mm256_mullo_epi32(in1_01, c1567);
    let mul3_312_01 = _mm256_mullo_epi32(in3_01, c_312);
    let t2_inner_01 = _mm256_srai_epi32(
        _mm256_add_epi32(_mm256_sub_epi32(mul1_1567_01, mul3_312_01), rnd12),
        12,
    );
    let t2_01 = _mm256_sub_epi32(t2_inner_01, in3_01);

    // t3 = (in1 * (3784-4096) + in3 * 1567 + 2048 >> 12) + in1
    let mul1_312_01 = _mm256_mullo_epi32(in1_01, c_312);
    let mul3_1567_01 = _mm256_mullo_epi32(in3_01, c1567);
    let t3_inner_01 = _mm256_srai_epi32(
        _mm256_add_epi32(_mm256_add_epi32(mul1_312_01, mul3_1567_01), rnd12),
        12,
    );
    let t3_01 = _mm256_add_epi32(t3_inner_01, in1_01);

    // Output: out0 = clip(t0+t3), out1 = clip(t1+t2), out2 = clip(t1-t2), out3 = clip(t0-t3)
    let vmin = _mm256_set1_epi32(clip_min);
    let vmax = _mm256_set1_epi32(clip_max);
    let out0_01 = _mm256_max_epi32(_mm256_min_epi32(_mm256_add_epi32(t0_01, t3_01), vmax), vmin);
    let out1_01 = _mm256_max_epi32(_mm256_min_epi32(_mm256_add_epi32(t1_01, t2_01), vmax), vmin);
    let out2_01 = _mm256_max_epi32(_mm256_min_epi32(_mm256_sub_epi32(t1_01, t2_01), vmax), vmin);
    let out3_01 = _mm256_max_epi32(_mm256_min_epi32(_mm256_sub_epi32(t0_01, t3_01), vmax), vmin);

    // Interleave outputs back: [out0, out1, out2, out3] per lane
    let mask0 = _mm256_set_epi32(0, 0, 0, -1i32, 0, 0, 0, -1i32);
    let mask1 = _mm256_set_epi32(0, 0, -1i32, 0, 0, 0, -1i32, 0);
    let mask2 = _mm256_set_epi32(0, -1i32, 0, 0, 0, -1i32, 0, 0);
    let mask3 = _mm256_set_epi32(-1i32, 0, 0, 0, -1i32, 0, 0, 0);

    let rows01_out = _mm256_or_si256(
        _mm256_or_si256(
            _mm256_and_si256(out0_01, mask0),
            _mm256_and_si256(_mm256_shuffle_epi32(out1_01, 0b00_00_00_01), mask1),
        ),
        _mm256_or_si256(
            _mm256_and_si256(_mm256_shuffle_epi32(out2_01, 0b00_00_10_00), mask2),
            _mm256_and_si256(_mm256_shuffle_epi32(out3_01, 0b00_11_00_00), mask3),
        ),
    );

    // Same for rows23
    let in0_23 = _mm256_shuffle_epi32(rows23, 0b00_00_00_00);
    let in1_23 = _mm256_shuffle_epi32(rows23, 0b01_01_01_01);
    let in2_23 = _mm256_shuffle_epi32(rows23, 0b10_10_10_10);
    let in3_23 = _mm256_shuffle_epi32(rows23, 0b11_11_11_11);

    let sum02_23 = _mm256_add_epi32(in0_23, in2_23);
    let t0_23 = _mm256_srai_epi32(
        _mm256_add_epi32(_mm256_mullo_epi32(sum02_23, sqrt2), rnd8),
        8,
    );

    let diff02_23 = _mm256_sub_epi32(in0_23, in2_23);
    let t1_23 = _mm256_srai_epi32(
        _mm256_add_epi32(_mm256_mullo_epi32(diff02_23, sqrt2), rnd8),
        8,
    );

    let mul1_1567_23 = _mm256_mullo_epi32(in1_23, c1567);
    let mul3_312_23 = _mm256_mullo_epi32(in3_23, c_312);
    let t2_inner_23 = _mm256_srai_epi32(
        _mm256_add_epi32(_mm256_sub_epi32(mul1_1567_23, mul3_312_23), rnd12),
        12,
    );
    let t2_23 = _mm256_sub_epi32(t2_inner_23, in3_23);

    let mul1_312_23 = _mm256_mullo_epi32(in1_23, c_312);
    let mul3_1567_23 = _mm256_mullo_epi32(in3_23, c1567);
    let t3_inner_23 = _mm256_srai_epi32(
        _mm256_add_epi32(_mm256_add_epi32(mul1_312_23, mul3_1567_23), rnd12),
        12,
    );
    let t3_23 = _mm256_add_epi32(t3_inner_23, in1_23);

    let out0_23 = _mm256_max_epi32(_mm256_min_epi32(_mm256_add_epi32(t0_23, t3_23), vmax), vmin);
    let out1_23 = _mm256_max_epi32(_mm256_min_epi32(_mm256_add_epi32(t1_23, t2_23), vmax), vmin);
    let out2_23 = _mm256_max_epi32(_mm256_min_epi32(_mm256_sub_epi32(t1_23, t2_23), vmax), vmin);
    let out3_23 = _mm256_max_epi32(_mm256_min_epi32(_mm256_sub_epi32(t0_23, t3_23), vmax), vmin);

    let rows23_out = _mm256_or_si256(
        _mm256_or_si256(
            _mm256_and_si256(out0_23, mask0),
            _mm256_and_si256(_mm256_shuffle_epi32(out1_23, 0b00_00_00_01), mask1),
        ),
        _mm256_or_si256(
            _mm256_and_si256(_mm256_shuffle_epi32(out2_23, 0b00_00_10_00), mask2),
            _mm256_and_si256(_mm256_shuffle_epi32(out3_23, 0b00_11_00_00), mask3),
        ),
    );

    (rows01_out, rows23_out)
}

// ============================================================================
// FFI WRAPPERS
// ============================================================================

/// FFI wrapper for 4x4 DCT_DCT 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_4x4_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, _coeff_len as usize * stride + stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_4x4_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// 4x4 DCT_DCT 16bpc
// ============================================================================

/// Full 2D DCT_DCT 4x4 inverse transform with add-to-destination (16bpc)
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_4x4_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize, // stride in bytes
    coeff: &mut [i32],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    // For 16bpc, stride is in bytes but we access u16, so stride_u16 = stride / 2
    let stride_u16 = dst_stride / 2;

    // Load coefficients (column-major storage)
    let row0 = _mm_set_epi32(
        coeff[12] as i32,
        coeff[8] as i32,
        coeff[4] as i32,
        coeff[0] as i32,
    );
    let row1 = _mm_set_epi32(
        coeff[13] as i32,
        coeff[9] as i32,
        coeff[5] as i32,
        coeff[1] as i32,
    );
    let row2 = _mm_set_epi32(
        coeff[14] as i32,
        coeff[10] as i32,
        coeff[6] as i32,
        coeff[2] as i32,
    );
    let row3 = _mm_set_epi32(
        coeff[15] as i32,
        coeff[11] as i32,
        coeff[7] as i32,
        coeff[3] as i32,
    );

    // Pack rows into 256-bit vectors
    let rows01 = _mm256_set_m128i(row1, row0);
    let rows23 = _mm256_set_m128i(row3, row2);

    // 16bpc clip ranges
    let row_clip_min = (!bitdepth_max) << 7;
    let row_clip_max = !row_clip_min;
    let col_clip_min = (!bitdepth_max) << 5;
    let col_clip_max = !col_clip_min;

    // DCT4 butterfly on rows
    let (rows01_out, rows23_out) =
        dct4_2rows_avx2(_token, rows01, rows23, row_clip_min, row_clip_max);

    // Transpose for column pass
    let r0 = _mm256_castsi256_si128(rows01_out);
    let r1 = _mm256_extracti128_si256(rows01_out, 1);
    let r2 = _mm256_castsi256_si128(rows23_out);
    let r3 = _mm256_extracti128_si256(rows23_out, 1);

    // Transpose 4x4 using unpack
    let t01_lo = _mm_unpacklo_epi32(r0, r1);
    let t01_hi = _mm_unpackhi_epi32(r0, r1);
    let t23_lo = _mm_unpacklo_epi32(r2, r3);
    let t23_hi = _mm_unpackhi_epi32(r2, r3);

    let c0 = _mm_unpacklo_epi64(t01_lo, t23_lo);
    let c1 = _mm_unpackhi_epi64(t01_lo, t23_lo);
    let c2 = _mm_unpacklo_epi64(t01_hi, t23_hi);
    let c3 = _mm_unpackhi_epi64(t01_hi, t23_hi);

    // Intermediate clamp (shift=0 for 4x4, so just clamp to col_clip range)
    let cmin = _mm_set1_epi32(col_clip_min);
    let cmax = _mm_set1_epi32(col_clip_max);
    let c0 = _mm_max_epi32(_mm_min_epi32(c0, cmax), cmin);
    let c1 = _mm_max_epi32(_mm_min_epi32(c1, cmax), cmin);
    let c2 = _mm_max_epi32(_mm_min_epi32(c2, cmax), cmin);
    let c3 = _mm_max_epi32(_mm_min_epi32(c3, cmax), cmin);

    // DCT4 on columns
    let cols01 = _mm256_set_m128i(c1, c0);
    let cols23 = _mm256_set_m128i(c3, c2);
    let (cols01_out, cols23_out) =
        dct4_2rows_avx2(_token, cols01, cols23, col_clip_min, col_clip_max);

    // Extract final columns
    let col0 = _mm256_castsi256_si128(cols01_out);
    let col1 = _mm256_extracti128_si256(cols01_out, 1);
    let col2 = _mm256_castsi256_si128(cols23_out);
    let col3 = _mm256_extracti128_si256(cols23_out, 1);

    // Transpose back to rows for output
    let t01_lo = _mm_unpacklo_epi32(col0, col1);
    let t01_hi = _mm_unpackhi_epi32(col0, col1);
    let t23_lo = _mm_unpacklo_epi32(col2, col3);
    let t23_hi = _mm_unpackhi_epi32(col2, col3);

    let out0 = _mm_unpacklo_epi64(t01_lo, t23_lo);
    let out1 = _mm_unpackhi_epi64(t01_lo, t23_lo);
    let out2 = _mm_unpacklo_epi64(t01_hi, t23_hi);
    let out3 = _mm_unpackhi_epi64(t01_hi, t23_hi);

    // Add to destination: shift by 4, clamp to [0, bitdepth_max]
    let rnd = _mm_set1_epi32(8);
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi32(bitdepth_max);

    // Row 0
    let dst0 = loadi64!(zerocopy::IntoBytes::as_bytes(&dst[..4]));
    let dst0_32 = _mm_unpacklo_epi16(dst0, zero);
    let scaled0 = _mm_srai_epi32(_mm_add_epi32(out0, rnd), 4);
    let sum0 = _mm_add_epi32(dst0_32, scaled0);
    let clamped0 = _mm_max_epi32(_mm_min_epi32(sum0, max_val), zero);
    let packed0 = _mm_packus_epi32(clamped0, clamped0);
    storei64!(zerocopy::IntoBytes::as_mut_bytes(&mut dst[..4]), packed0);

    // Row 1
    let dst_off1 = stride_u16;
    let dst1 = loadi64!(zerocopy::IntoBytes::as_bytes(&dst[dst_off1..dst_off1 + 4]));
    let dst1_32 = _mm_unpacklo_epi16(dst1, zero);
    let scaled1 = _mm_srai_epi32(_mm_add_epi32(out1, rnd), 4);
    let sum1 = _mm_add_epi32(dst1_32, scaled1);
    let clamped1 = _mm_max_epi32(_mm_min_epi32(sum1, max_val), zero);
    let packed1 = _mm_packus_epi32(clamped1, clamped1);
    storei64!(
        zerocopy::IntoBytes::as_mut_bytes(&mut dst[dst_off1..dst_off1 + 4]),
        packed1
    );

    // Row 2
    let dst_off2 = stride_u16 * 2;
    let dst2 = loadi64!(zerocopy::IntoBytes::as_bytes(&dst[dst_off2..dst_off2 + 4]));
    let dst2_32 = _mm_unpacklo_epi16(dst2, zero);
    let scaled2 = _mm_srai_epi32(_mm_add_epi32(out2, rnd), 4);
    let sum2 = _mm_add_epi32(dst2_32, scaled2);
    let clamped2 = _mm_max_epi32(_mm_min_epi32(sum2, max_val), zero);
    let packed2 = _mm_packus_epi32(clamped2, clamped2);
    storei64!(
        zerocopy::IntoBytes::as_mut_bytes(&mut dst[dst_off2..dst_off2 + 4]),
        packed2
    );

    // Row 3
    let dst_off3 = stride_u16 * 3;
    let dst3 = loadi64!(zerocopy::IntoBytes::as_bytes(&dst[dst_off3..dst_off3 + 4]));
    let dst3_32 = _mm_unpacklo_epi16(dst3, zero);
    let scaled3 = _mm_srai_epi32(_mm_add_epi32(out3, rnd), 4);
    let sum3 = _mm_add_epi32(dst3_32, scaled3);
    let clamped3 = _mm_max_epi32(_mm_min_epi32(sum3, max_val), zero);
    let packed3 = _mm_packus_epi32(clamped3, clamped3);
    storei64!(
        zerocopy::IntoBytes::as_mut_bytes(&mut dst[dst_off3..dst_off3 + 4]),
        packed3
    );

    // Clear coefficients
    coeff[..16].fill(0);
}

/// FFI wrapper for 4x4 DCT_DCT 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_4x4_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_4x4_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// 4x4 WHT (Walsh-Hadamard Transform)
// ============================================================================

/// WHT4x4 - Walsh-Hadamard Transform (SSE2 SIMD)
///
/// Processes all 4 rows/columns in parallel using SSE2 registers.
/// WHT butterfly: t0=a+b, t2=c-d, t4=(t0-t2)>>1, t3=t4-d, t1=t4-b,
///   out0=t0-t3, out1=t3, out2=t1, out3=t2+t1
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_wht_wht_4x4_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();

    // Load 4 columns from column-major i16 coefficients, sign-extend to i32, shift >>2
    let col0 = _mm_srai_epi32(
        _mm_set_epi32(
            coeff[3] as i32,
            coeff[2] as i32,
            coeff[1] as i32,
            coeff[0] as i32,
        ),
        2,
    );
    let col1 = _mm_srai_epi32(
        _mm_set_epi32(
            coeff[7] as i32,
            coeff[6] as i32,
            coeff[5] as i32,
            coeff[4] as i32,
        ),
        2,
    );
    let col2 = _mm_srai_epi32(
        _mm_set_epi32(
            coeff[11] as i32,
            coeff[10] as i32,
            coeff[9] as i32,
            coeff[8] as i32,
        ),
        2,
    );
    let col3 = _mm_srai_epi32(
        _mm_set_epi32(
            coeff[15] as i32,
            coeff[14] as i32,
            coeff[13] as i32,
            coeff[12] as i32,
        ),
        2,
    );

    // Row pass: WHT butterfly on all 4 rows in parallel
    let t0 = _mm_add_epi32(col0, col1);
    let t2 = _mm_sub_epi32(col2, col3);
    let t4 = _mm_srai_epi32(_mm_sub_epi32(t0, t2), 1);
    let t3 = _mm_sub_epi32(t4, col3);
    let t1 = _mm_sub_epi32(t4, col1);
    let r0 = _mm_sub_epi32(t0, t3);
    let r1 = t3;
    let r2 = t1;
    let r3 = _mm_add_epi32(t2, t1);

    // 4x4 transpose
    let t01_lo = _mm_unpacklo_epi32(r0, r1);
    let t01_hi = _mm_unpackhi_epi32(r0, r1);
    let t23_lo = _mm_unpacklo_epi32(r2, r3);
    let t23_hi = _mm_unpackhi_epi32(r2, r3);
    let row0 = _mm_unpacklo_epi64(t01_lo, t23_lo);
    let row1 = _mm_unpackhi_epi64(t01_lo, t23_lo);
    let row2 = _mm_unpacklo_epi64(t01_hi, t23_hi);
    let row3 = _mm_unpackhi_epi64(t01_hi, t23_hi);

    // Column pass: WHT butterfly on all 4 columns in parallel
    let t0 = _mm_add_epi32(row0, row1);
    let t2 = _mm_sub_epi32(row2, row3);
    let t4 = _mm_srai_epi32(_mm_sub_epi32(t0, t2), 1);
    let t3 = _mm_sub_epi32(t4, row3);
    let t1 = _mm_sub_epi32(t4, row1);
    let final0 = _mm_sub_epi32(t0, t3);
    let final1 = t3;
    let final2 = t1;
    let final3 = _mm_add_epi32(t2, t1);

    // Add to destination pixels and clamp to [0, bitdepth_max]
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi16(bitdepth_max as i16);

    let d0 = loadi32!(&dst[..4]);
    let d0_32 = _mm_cvtepi16_epi32(_mm_unpacklo_epi8(d0, zero));
    let sum0 = _mm_add_epi32(d0_32, final0);
    let sum0_8 = _mm_packus_epi16(
        _mm_max_epi16(_mm_min_epi16(_mm_packs_epi32(sum0, sum0), max_val), zero),
        zero,
    );
    storei32!(&mut dst[..4], sum0_8);

    let off1 = dst_stride;
    let d1 = loadi32!(&dst[off1..off1 + 4]);
    let d1_32 = _mm_cvtepi16_epi32(_mm_unpacklo_epi8(d1, zero));
    let sum1 = _mm_add_epi32(d1_32, final1);
    let sum1_8 = _mm_packus_epi16(
        _mm_max_epi16(_mm_min_epi16(_mm_packs_epi32(sum1, sum1), max_val), zero),
        zero,
    );
    storei32!(&mut dst[off1..off1 + 4], sum1_8);

    let off2 = dst_stride * 2;
    let d2 = loadi32!(&dst[off2..off2 + 4]);
    let d2_32 = _mm_cvtepi16_epi32(_mm_unpacklo_epi8(d2, zero));
    let sum2 = _mm_add_epi32(d2_32, final2);
    let sum2_8 = _mm_packus_epi16(
        _mm_max_epi16(_mm_min_epi16(_mm_packs_epi32(sum2, sum2), max_val), zero),
        zero,
    );
    storei32!(&mut dst[off2..off2 + 4], sum2_8);

    let off3 = dst_stride * 3;
    let d3 = loadi32!(&dst[off3..off3 + 4]);
    let d3_32 = _mm_cvtepi16_epi32(_mm_unpacklo_epi8(d3, zero));
    let sum3 = _mm_add_epi32(d3_32, final3);
    let sum3_8 = _mm_packus_epi16(
        _mm_max_epi16(_mm_min_epi16(_mm_packs_epi32(sum3, sum3), max_val), zero),
        zero,
    );
    storei32!(&mut dst[off3..off3 + 4], sum3_8);

    // Clear coefficients
    let coeff_bytes = zerocopy::IntoBytes::as_mut_bytes(&mut *coeff);
    storeu_128!(
        <&mut [u8; 16]>::try_from(&mut coeff_bytes[..16]).unwrap(),
        _mm_setzero_si128()
    );
    storeu_128!(
        <&mut [u8; 16]>::try_from(&mut coeff_bytes[16..32]).unwrap(),
        _mm_setzero_si128()
    );
}

/// FFI wrapper for 4x4 WHT 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_wht_wht_4x4_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let abs_stride = dst_stride.unsigned_abs();
    let buf_size = 3 * abs_stride + 4;
    let dst_slice = if dst_stride >= 0 {
        unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_size) }
    } else {
        let start = unsafe { (dst_ptr as *mut u8).offset(3 * dst_stride) };
        let base = 3 * abs_stride;
        unsafe { std::slice::from_raw_parts_mut(start.add(base), buf_size - base) }
    };
    let coeff_slice = unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, 16) };
    inv_txfm_add_wht_wht_4x4_8bpc_avx2_inner(
        _token,
        dst_slice,
        abs_stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

/// WHT4x4 16bpc - Walsh-Hadamard Transform (SSE2 SIMD)
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_wht_wht_4x4_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    byte_stride: usize,
    coeff: &mut [i32],
    _eob: i32,
    bitdepth_max: i32,
) {
    let dst_stride_u16 = byte_stride / 2;
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();

    // Load 4 columns from column-major i32 coefficients, shift >>2
    let col0 = _mm_srai_epi32(loadu_128!(&coeff[0..4], [i32; 4]), 2);
    let col1 = _mm_srai_epi32(loadu_128!(&coeff[4..8], [i32; 4]), 2);
    let col2 = _mm_srai_epi32(loadu_128!(&coeff[8..12], [i32; 4]), 2);
    let col3 = _mm_srai_epi32(loadu_128!(&coeff[12..16], [i32; 4]), 2);

    // Row pass: WHT butterfly on all 4 rows in parallel
    let t0 = _mm_add_epi32(col0, col1);
    let t2 = _mm_sub_epi32(col2, col3);
    let t4 = _mm_srai_epi32(_mm_sub_epi32(t0, t2), 1);
    let t3 = _mm_sub_epi32(t4, col3);
    let t1 = _mm_sub_epi32(t4, col1);
    let r0 = _mm_sub_epi32(t0, t3);
    let r1 = t3;
    let r2 = t1;
    let r3 = _mm_add_epi32(t2, t1);

    // 4x4 transpose
    let t01_lo = _mm_unpacklo_epi32(r0, r1);
    let t01_hi = _mm_unpackhi_epi32(r0, r1);
    let t23_lo = _mm_unpacklo_epi32(r2, r3);
    let t23_hi = _mm_unpackhi_epi32(r2, r3);
    let row0 = _mm_unpacklo_epi64(t01_lo, t23_lo);
    let row1 = _mm_unpackhi_epi64(t01_lo, t23_lo);
    let row2 = _mm_unpacklo_epi64(t01_hi, t23_hi);
    let row3 = _mm_unpackhi_epi64(t01_hi, t23_hi);

    // Column pass: WHT butterfly on all 4 columns in parallel
    let t0 = _mm_add_epi32(row0, row1);
    let t2 = _mm_sub_epi32(row2, row3);
    let t4 = _mm_srai_epi32(_mm_sub_epi32(t0, t2), 1);
    let t3 = _mm_sub_epi32(t4, row3);
    let t1 = _mm_sub_epi32(t4, row1);
    let final0 = _mm_sub_epi32(t0, t3);
    let final1 = t3;
    let final2 = t1;
    let final3 = _mm_add_epi32(t2, t1);

    // Add to destination pixels and clamp to [0, bitdepth_max]
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi32(bitdepth_max);

    // Row 0: load 4 u16, widen to i32, add, clamp, pack back
    let d0 = loadi64!(zerocopy::IntoBytes::as_bytes(&dst[..4]));
    let d0_32 = _mm_cvtepu16_epi32(d0);
    let sum0 = _mm_max_epi32(_mm_min_epi32(_mm_add_epi32(d0_32, final0), max_val), zero);
    let sum0_16 = _mm_packus_epi32(sum0, sum0);
    storei64!(zerocopy::IntoBytes::as_mut_bytes(&mut dst[..4]), sum0_16);

    let off1 = dst_stride_u16;
    let d1 = loadi64!(zerocopy::IntoBytes::as_bytes(&dst[off1..off1 + 4]));
    let d1_32 = _mm_cvtepu16_epi32(d1);
    let sum1 = _mm_max_epi32(_mm_min_epi32(_mm_add_epi32(d1_32, final1), max_val), zero);
    let sum1_16 = _mm_packus_epi32(sum1, sum1);
    storei64!(
        zerocopy::IntoBytes::as_mut_bytes(&mut dst[off1..off1 + 4]),
        sum1_16
    );

    let off2 = dst_stride_u16 * 2;
    let d2 = loadi64!(zerocopy::IntoBytes::as_bytes(&dst[off2..off2 + 4]));
    let d2_32 = _mm_cvtepu16_epi32(d2);
    let sum2 = _mm_max_epi32(_mm_min_epi32(_mm_add_epi32(d2_32, final2), max_val), zero);
    let sum2_16 = _mm_packus_epi32(sum2, sum2);
    storei64!(
        zerocopy::IntoBytes::as_mut_bytes(&mut dst[off2..off2 + 4]),
        sum2_16
    );

    let off3 = dst_stride_u16 * 3;
    let d3 = loadi64!(zerocopy::IntoBytes::as_bytes(&dst[off3..off3 + 4]));
    let d3_32 = _mm_cvtepu16_epi32(d3);
    let sum3 = _mm_max_epi32(_mm_min_epi32(_mm_add_epi32(d3_32, final3), max_val), zero);
    let sum3_16 = _mm_packus_epi32(sum3, sum3);
    storei64!(
        zerocopy::IntoBytes::as_mut_bytes(&mut dst[off3..off3 + 4]),
        sum3_16
    );

    // Clear coefficients (16 i32 = 64 bytes)
    let coeff_bytes = zerocopy::IntoBytes::as_mut_bytes(&mut *coeff);
    storeu_128!(
        <&mut [u8; 16]>::try_from(&mut coeff_bytes[..16]).unwrap(),
        _mm_setzero_si128()
    );
    storeu_128!(
        <&mut [u8; 16]>::try_from(&mut coeff_bytes[16..32]).unwrap(),
        _mm_setzero_si128()
    );
    storeu_128!(
        <&mut [u8; 16]>::try_from(&mut coeff_bytes[32..48]).unwrap(),
        _mm_setzero_si128()
    );
    storeu_128!(
        <&mut [u8; 16]>::try_from(&mut coeff_bytes[48..64]).unwrap(),
        _mm_setzero_si128()
    );
}

/// FFI wrapper for 4x4 WHT 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_wht_wht_4x4_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let abs_stride = dst_stride.unsigned_abs();
    let buf_size_u16 = (3 * abs_stride + 8) / 2; // 3 rows + 4 pixels
    let dst_slice = if dst_stride >= 0 {
        unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u16, buf_size_u16) }
    } else {
        let start = unsafe { (dst_ptr as *mut u16).offset(3 * (dst_stride / 2)) };
        let base_u16 = 3 * (abs_stride / 2);
        unsafe { std::slice::from_raw_parts_mut(start.add(base_u16), buf_size_u16 - base_u16) }
    };
    let coeff_slice = unsafe { std::slice::from_raw_parts_mut(coeff as *mut i32, 16) };
    inv_txfm_add_wht_wht_4x4_16bpc_avx2_inner(
        _token,
        dst_slice,
        abs_stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// IDENTITY TRANSFORM HELPER (shared)
// ============================================================================

/// Identity transform - just scale by sqrt(2) and add to dst
#[cfg(target_arch = "x86_64")]
#[arcane]
pub fn inv_identity_add_4x4_8bpc_avx2(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    // identity4(x) = x + (x * 1697 + 2048) >> 12 ≈ x * sqrt(2)
    // 4x4 IDTX = identity4 on rows, identity4 on cols, shift=0, then final (+ 8) >> 4
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi16(bitdepth_max as i16);
    let identity4 = |v: i32| -> i32 { v + ((v * 1697 + 2048) >> 12) };

    for y in 0..4 {
        let dst_off = y * dst_stride;

        // Load destination
        let d = loadi32!(&dst[dst_off..dst_off + 4]);
        let d16 = _mm_unpacklo_epi8(d, zero);

        // Load coeffs for this row (column-major: y, y+4, y+8, y+12)
        let c0 = coeff[y] as i32;
        let c1 = coeff[y + 4] as i32;
        let c2 = coeff[y + 8] as i32;
        let c3 = coeff[y + 12] as i32;

        // Row: identity4, intermediate clamp to col_clip_range, Col: identity4, final: (+ 8) >> 4
        // For 8bpc: col_clip = i16::MIN..i16::MAX. For 16bpc: col_clip = (!bdmax)<<5..=!((!bdmax)<<5)
        let col_clip_min = if bitdepth_max == 255 {
            i16::MIN as i32
        } else {
            (!bitdepth_max) << 5
        };
        let col_clip_max = !col_clip_min;
        let scale = |v: i32| -> i32 { identity4(identity4(v).clamp(col_clip_min, col_clip_max)) };

        let r0 = (scale(c0) + 8) >> 4;
        let r1 = (scale(c1) + 8) >> 4;
        let r2 = (scale(c2) + 8) >> 4;
        let r3 = (scale(c3) + 8) >> 4;

        // Add to destination
        let result = _mm_set_epi32(r3, r2, r1, r0);
        let d32 = _mm_cvtepi16_epi32(d16);
        let sum = _mm_add_epi32(d32, result);
        let sum16 = _mm_packs_epi32(sum, sum);
        let clamped = _mm_max_epi16(_mm_min_epi16(sum16, max_val), zero);
        let packed = _mm_packus_epi16(clamped, clamped);

        storei32!(&mut dst[dst_off..dst_off + 4], packed);
    }

    // Clear coefficients
    coeff[..16].fill(0);
}

/// FFI wrapper for 4x4 IDTX 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_identity_4x4_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, _coeff_len as usize * stride + stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_identity_add_4x4_8bpc_avx2(_token, dst_slice, stride, coeff_slice, eob, bitdepth_max);
}

// ============================================================================
// 8x8 IDTX (Identity)
// ============================================================================

/// 8x8 IDTX (identity transform)
/// Identity8: out = in * 2
/// For 8x8 IDTX: row pass * 2, col pass * 2 = * 4
/// Plus final shift: (+ 8) >> 4
#[cfg(target_arch = "x86_64")]
#[arcane]
pub fn inv_identity_add_8x8_8bpc_avx2(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi16(bitdepth_max as i16);

    // Identity8: c * 2 per dimension. For 8x8: shift=1, rnd=1.
    // Full pipeline per coeff (must use i32 to avoid i16 overflow):
    //   row = c * 2
    //   inter = (row + 1) >> 1
    //   col = inter * 2
    //   residual = (col + 8) >> 4
    let one = _mm_set1_epi32(1);
    let eight = _mm_set1_epi32(8);

    for y in 0..8 {
        let dst_off = y * dst_stride;

        // Load 8 destination pixels
        let d = loadi64!(&dst[dst_off..dst_off + 8]);
        let d16 = _mm_unpacklo_epi8(d, zero);

        // Load 8 coefficients for this row (column-major: y, y+8, y+16, ...)
        let mut coeffs = [0i16; 8];
        for x in 0..8 {
            coeffs[x] = coeff[y + x * 8];
        }

        // Process in i32 to avoid i16 overflow (identity8 doubles, can exceed i16 range)
        let c_vec = loadu_128!(<&[i16; 8]>::try_from(&coeffs[..]).unwrap());
        let c_lo = _mm_cvtepi16_epi32(c_vec);
        let c_hi = _mm_cvtepi16_epi32(_mm_srli_si128(c_vec, 8));

        // row: c * 2
        let row_lo = _mm_slli_epi32(c_lo, 1);
        let row_hi = _mm_slli_epi32(c_hi, 1);
        // intermediate: (row + 1) >> 1
        let inter_lo = _mm_srai_epi32(_mm_add_epi32(row_lo, one), 1);
        let inter_hi = _mm_srai_epi32(_mm_add_epi32(row_hi, one), 1);
        // col: inter * 2
        let col_lo = _mm_slli_epi32(inter_lo, 1);
        let col_hi = _mm_slli_epi32(inter_hi, 1);
        // final: (col + 8) >> 4
        let res_lo = _mm_srai_epi32(_mm_add_epi32(col_lo, eight), 4);
        let res_hi = _mm_srai_epi32(_mm_add_epi32(col_hi, eight), 4);

        // Pack back to i16 and add to destination
        let res16 = _mm_packs_epi32(res_lo, res_hi);
        let sum = _mm_add_epi16(d16, res16);
        let clamped = _mm_max_epi16(_mm_min_epi16(sum, max_val), zero);
        let packed = _mm_packus_epi16(clamped, clamped);

        storei64!(&mut dst[dst_off..dst_off + 8], packed);
    }

    // Clear coefficients (8x8 = 64 i16 = 128 bytes = 8 x 16-byte stores)
    coeff[..64].fill(0);
}

/// FFI wrapper for 8x8 IDTX 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_identity_8x8_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, _coeff_len as usize * stride + stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_identity_add_8x8_8bpc_avx2(_token, dst_slice, stride, coeff_slice, eob, bitdepth_max);
}

// ============================================================================
// 8x8 DCT_DCT
// ============================================================================

/// DCT4 1D transform (used by DCT8)
#[inline]
fn dct4_1d(c: &mut [i32], stride: usize, min: i32, max: i32) {
    let clip = |v: i32| v.clamp(min, max);

    let in0 = c[0 * stride];
    let in1 = c[1 * stride];
    let in2 = c[2 * stride];
    let in3 = c[3 * stride];

    let t0 = (in0 + in2) * 181 + 128 >> 8;
    let t1 = (in0 - in2) * 181 + 128 >> 8;
    let t2 = (in1 * 1567 - in3 * (3784 - 4096) + 2048 >> 12) - in3;
    let t3 = (in1 * (3784 - 4096) + in3 * 1567 + 2048 >> 12) + in1;

    c[0 * stride] = clip(t0 + t3);
    c[1 * stride] = clip(t1 + t2);
    c[2 * stride] = clip(t1 - t2);
    c[3 * stride] = clip(t0 - t3);
}

/// DCT8 1D transform
#[inline]
fn dct8_1d(c: &mut [i32], stride: usize, min: i32, max: i32) {
    let clip = |v: i32| v.clamp(min, max);

    // First apply DCT4 to even positions
    dct4_1d(c, stride * 2, min, max);

    let in1 = c[1 * stride];
    let in3 = c[3 * stride];
    let in5 = c[5 * stride];
    let in7 = c[7 * stride];

    let t4a = (in1 * 799 - in7 * (4017 - 4096) + 2048 >> 12) - in7;
    let t5a = in5 * 1703 - in3 * 1138 + 1024 >> 11;
    let t6a = in5 * 1138 + in3 * 1703 + 1024 >> 11;
    let t7a = (in1 * (4017 - 4096) + in7 * 799 + 2048 >> 12) + in1;

    let t4 = clip(t4a + t5a);
    let t5a = clip(t4a - t5a);
    let t7 = clip(t7a + t6a);
    let t6a = clip(t7a - t6a);

    let t5 = (t6a - t5a) * 181 + 128 >> 8;
    let t6 = (t6a + t5a) * 181 + 128 >> 8;

    let t0 = c[0 * stride];
    let t1 = c[2 * stride];
    let t2 = c[4 * stride];
    let t3 = c[6 * stride];

    c[0 * stride] = clip(t0 + t7);
    c[1 * stride] = clip(t1 + t6);
    c[2 * stride] = clip(t2 + t5);
    c[3 * stride] = clip(t3 + t4);
    c[4 * stride] = clip(t3 - t4);
    c[5 * stride] = clip(t2 - t5);
    c[6 * stride] = clip(t1 - t6);
    c[7 * stride] = clip(t0 - t7);
}

/// ADST4 1D transform (strided version for rectangular transforms)
#[inline]
fn adst4_1d(c: &mut [i32], stride: usize, min: i32, max: i32) {
    let clip = |v: i32| v.clamp(min, max);

    let in0 = c[0 * stride];
    let in1 = c[1 * stride];
    let in2 = c[2 * stride];
    let in3 = c[3 * stride];

    let out0 =
        ((1321 * in0 + (3803 - 4096) * in2 + (2482 - 4096) * in3 + (3344 - 4096) * in1 + 2048)
            >> 12)
            + in2
            + in3
            + in1;
    let out1 =
        (((2482 - 4096) * in0 - 1321 * in2 - (3803 - 4096) * in3 + (3344 - 4096) * in1 + 2048)
            >> 12)
            + in0
            - in3
            + in1;
    let out2 = (209 * (in0 - in2 + in3) + 128) >> 8;
    let out3 = (((3803 - 4096) * in0 + (2482 - 4096) * in2 - 1321 * in3 - (3344 - 4096) * in1
        + 2048)
        >> 12)
        + in0
        + in2
        - in1;

    c[0 * stride] = clip(out0);
    c[1 * stride] = clip(out1);
    c[2 * stride] = clip(out2);
    c[3 * stride] = clip(out3);
}

/// FlipADST4 1D transform (strided version)
#[inline]
fn flipadst4_1d(c: &mut [i32], stride: usize, min: i32, max: i32) {
    let clip = |v: i32| v.clamp(min, max);

    let in0 = c[0 * stride];
    let in1 = c[1 * stride];
    let in2 = c[2 * stride];
    let in3 = c[3 * stride];

    let out0 =
        ((1321 * in0 + (3803 - 4096) * in2 + (2482 - 4096) * in3 + (3344 - 4096) * in1 + 2048)
            >> 12)
            + in2
            + in3
            + in1;
    let out1 =
        (((2482 - 4096) * in0 - 1321 * in2 - (3803 - 4096) * in3 + (3344 - 4096) * in1 + 2048)
            >> 12)
            + in0
            - in3
            + in1;
    let out2 = (209 * (in0 - in2 + in3) + 128) >> 8;
    let out3 = (((3803 - 4096) * in0 + (2482 - 4096) * in2 - 1321 * in3 - (3344 - 4096) * in1
        + 2048)
        >> 12)
        + in0
        + in2
        - in1;

    // Flip output
    c[0 * stride] = clip(out3);
    c[1 * stride] = clip(out2);
    c[2 * stride] = clip(out1);
    c[3 * stride] = clip(out0);
}

/// ADST8 1D transform (strided version for rectangular transforms)
#[inline]
fn adst8_1d(c: &mut [i32], stride: usize, min: i32, max: i32) {
    let clip = |v: i32| v.clamp(min, max);

    let in0 = c[0 * stride];
    let in1 = c[1 * stride];
    let in2 = c[2 * stride];
    let in3 = c[3 * stride];
    let in4 = c[4 * stride];
    let in5 = c[5 * stride];
    let in6 = c[6 * stride];
    let in7 = c[7 * stride];

    let t0a = (((4076 - 4096) * in7 + 401 * in0 + 2048) >> 12) + in7;
    let t1a = ((401 * in7 - (4076 - 4096) * in0 + 2048) >> 12) - in0;
    let t2a = (((3612 - 4096) * in5 + 1931 * in2 + 2048) >> 12) + in5;
    let t3a = ((1931 * in5 - (3612 - 4096) * in2 + 2048) >> 12) - in2;
    let t4a = (1299 * in3 + 1583 * in4 + 1024) >> 11;
    let t5a = (1583 * in3 - 1299 * in4 + 1024) >> 11;
    let t6a = ((1189 * in1 + (3920 - 4096) * in6 + 2048) >> 12) + in6;
    let t7a = (((3920 - 4096) * in1 - 1189 * in6 + 2048) >> 12) + in1;

    let t0 = clip(t0a + t4a);
    let t1 = clip(t1a + t5a);
    let t2 = clip(t2a + t6a);
    let t3 = clip(t3a + t7a);
    let t4 = clip(t0a - t4a);
    let t5 = clip(t1a - t5a);
    let t6 = clip(t2a - t6a);
    let t7 = clip(t3a - t7a);

    let t4a = (((3784 - 4096) * t4 + 1567 * t5 + 2048) >> 12) + t4;
    let t5a = ((1567 * t4 - (3784 - 4096) * t5 + 2048) >> 12) - t5;
    let t6a = (((3784 - 4096) * t7 - 1567 * t6 + 2048) >> 12) + t7;
    let t7a = ((1567 * t7 + (3784 - 4096) * t6 + 2048) >> 12) + t6;

    let out0 = clip(t0 + t2);
    let out7 = -clip(t1 + t3);
    let t2_final = clip(t0 - t2);
    let t3_final = clip(t1 - t3);
    let out1 = -clip(t4a + t6a);
    let out6 = clip(t5a + t7a);
    let t6_final = clip(t4a - t6a);
    let t7_final = clip(t5a - t7a);

    let out3 = -(((t2_final + t3_final) * 181 + 128) >> 8);
    let out4 = ((t2_final - t3_final) * 181 + 128) >> 8;
    let out2 = ((t6_final + t7_final) * 181 + 128) >> 8;
    let out5 = -(((t6_final - t7_final) * 181 + 128) >> 8);

    c[0 * stride] = out0;
    c[1 * stride] = out1;
    c[2 * stride] = out2;
    c[3 * stride] = out3;
    c[4 * stride] = out4;
    c[5 * stride] = out5;
    c[6 * stride] = out6;
    c[7 * stride] = out7;
}

/// FlipADST8 1D transform (strided version)
#[inline]
fn flipadst8_1d(c: &mut [i32], stride: usize, min: i32, max: i32) {
    let clip = |v: i32| v.clamp(min, max);

    let in0 = c[0 * stride];
    let in1 = c[1 * stride];
    let in2 = c[2 * stride];
    let in3 = c[3 * stride];
    let in4 = c[4 * stride];
    let in5 = c[5 * stride];
    let in6 = c[6 * stride];
    let in7 = c[7 * stride];

    let t0a = (((4076 - 4096) * in7 + 401 * in0 + 2048) >> 12) + in7;
    let t1a = ((401 * in7 - (4076 - 4096) * in0 + 2048) >> 12) - in0;
    let t2a = (((3612 - 4096) * in5 + 1931 * in2 + 2048) >> 12) + in5;
    let t3a = ((1931 * in5 - (3612 - 4096) * in2 + 2048) >> 12) - in2;
    let t4a = (1299 * in3 + 1583 * in4 + 1024) >> 11;
    let t5a = (1583 * in3 - 1299 * in4 + 1024) >> 11;
    let t6a = ((1189 * in1 + (3920 - 4096) * in6 + 2048) >> 12) + in6;
    let t7a = (((3920 - 4096) * in1 - 1189 * in6 + 2048) >> 12) + in1;

    let t0 = clip(t0a + t4a);
    let t1 = clip(t1a + t5a);
    let t2 = clip(t2a + t6a);
    let t3 = clip(t3a + t7a);
    let t4 = clip(t0a - t4a);
    let t5 = clip(t1a - t5a);
    let t6 = clip(t2a - t6a);
    let t7 = clip(t3a - t7a);

    let t4a = (((3784 - 4096) * t4 + 1567 * t5 + 2048) >> 12) + t4;
    let t5a = ((1567 * t4 - (3784 - 4096) * t5 + 2048) >> 12) - t5;
    let t6a = (((3784 - 4096) * t7 - 1567 * t6 + 2048) >> 12) + t7;
    let t7a = ((1567 * t7 + (3784 - 4096) * t6 + 2048) >> 12) + t6;

    let out0 = clip(t0 + t2);
    let out7 = -clip(t1 + t3);
    let t2_final = clip(t0 - t2);
    let t3_final = clip(t1 - t3);
    let out1 = -clip(t4a + t6a);
    let out6 = clip(t5a + t7a);
    let t6_final = clip(t4a - t6a);
    let t7_final = clip(t5a - t7a);

    let out3 = -(((t2_final + t3_final) * 181 + 128) >> 8);
    let out4 = ((t2_final - t3_final) * 181 + 128) >> 8;
    let out2 = ((t6_final + t7_final) * 181 + 128) >> 8;
    let out5 = -(((t6_final - t7_final) * 181 + 128) >> 8);

    // Flip output
    c[0 * stride] = out7;
    c[1 * stride] = out6;
    c[2 * stride] = out5;
    c[3 * stride] = out4;
    c[4 * stride] = out3;
    c[5 * stride] = out2;
    c[6 * stride] = out1;
    c[7 * stride] = out0;
}

/// Full 2D DCT_DCT 8x8 inverse transform with add-to-destination
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_8x8_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    // For 8bpc:
    // row_clip_min/max = i16::MIN/MAX (-32768, 32767)
    // col_clip_min/max = i16::MIN/MAX
    let _row_clip_min = i16::MIN as i32;
    let _row_clip_max = i16::MAX as i32;
    let col_clip_min = i16::MIN as i32;
    let col_clip_max = i16::MAX as i32;

    // Load coefficients and convert to i32 row-major
    // Input is column-major: coeff[y + x * 8]
    let mut tmp = [0i32; 64];

    // i16-packed pmaddwd row pass (T-5 #5: bit-exact proven against scalar reference).
    {
        let coeff_arr: &[i16; 64] = coeff.as_slice()[..64].try_into().unwrap();
        let raw = dct8_row_pass_i16_simd(_token, *coeff_arr);
        // Apply intermediate shift (rnd=1, shift=1 for 8x8) + clip to col range.
        let rnd_v = _mm256_set1_epi32(1);
        let col_min_v = _mm256_set1_epi32(col_clip_min);
        let col_max_v = _mm256_set1_epi32(col_clip_max);
        for y in 0..8 {
            let v = loadu_256!(&raw[y * 8..y * 8 + 8], [i32; 8]);
            let shifted = _mm256_srai_epi32::<1>(_mm256_add_epi32(v, rnd_v));
            let clipped = _mm256_max_epi32(_mm256_min_epi32(shifted, col_max_v), col_min_v);
            storeu_256!(&mut tmp[y * 8..y * 8 + 8], [i32; 8], clipped);
        }
    }

    // Column transform: i16-packed pmaddwd (replaces i32 mullo column pass)
    let col_out = dct8_col_pass_i16(_token, &tmp);

    // Add to destination with SIMD — col_out[y] is ymm of 8 i32 (one per column)
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi16(bitdepth_max as i16);
    let rnd_final = _mm256_set1_epi32(8);

    for y in 0..8 {
        let dst_off = y * dst_stride;

        // Load destination pixels (8 bytes)
        let d = loadi64!(&dst[dst_off..dst_off + 8]);
        let d16 = _mm_unpacklo_epi8(d, zero);

        // Final scaling: (c + 8) >> 4
        let c_scaled = _mm256_srai_epi32(_mm256_add_epi32(col_out[y], rnd_final), 4);

        // Pack to 16-bit
        let c_lo_scaled = _mm256_castsi256_si128(c_scaled);
        let c_hi_scaled = _mm256_extracti128_si256(c_scaled, 1);
        let c16 = _mm_packs_epi32(c_lo_scaled, c_hi_scaled);

        // Add to destination
        let sum = _mm_add_epi16(d16, c16);
        let clamped = _mm_max_epi16(_mm_min_epi16(sum, max_val), zero);
        let packed = _mm_packus_epi16(clamped, clamped);

        // Store 8 pixels
        storei64!(&mut dst[dst_off..dst_off + 8], packed);
    }

    // Clear coefficients
    coeff[..64].fill(0);
}

/// FFI wrapper for 8x8 DCT_DCT 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_8x8_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, _coeff_len as usize * stride + stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_8x8_8bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// 8x8 DCT_DCT 16bpc
// ============================================================================

/// 8x8 DCT_DCT for 16bpc (10/12-bit pixels)
#[cfg(target_arch = "x86_64")]
#[arcane]
fn inv_txfm_add_dct_dct_8x8_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_stride: usize, // stride in bytes
    coeff: &mut [i32],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    // For 16bpc, stride is in bytes but we access u16
    let stride_u16 = dst_stride / 2;

    // For 16bpc: intermediate values have larger range, use i32 throughout
    let row_clip_min = (!bitdepth_max) << 7;
    let row_clip_max = !row_clip_min;
    let col_clip_min = (!bitdepth_max) << 5;
    let col_clip_max = !col_clip_min;

    // Load coefficients and convert to i32 row-major
    // Input is column-major: coeff[y + x * 8]
    let mut tmp = [0i32; 64];

    // Row transform
    // shift = 1 for 8x8
    let rnd = 1;
    let shift = 1;

    for y in 0..8 {
        // Load row from column-major
        let mut scratch = [0i32; 8];
        for x in 0..8 {
            scratch[x] = coeff[y + x * 8] as i32;
        }
        dct8_1d(&mut scratch[..8], 1, row_clip_min, row_clip_max);
        // Apply intermediate shift and store row-major
        for x in 0..8 {
            tmp[y * 8 + x] = iclip((scratch[x] + rnd) >> shift, col_clip_min, col_clip_max);
        }
    }

    // Column transform: SIMD across 8 columns (single chunk)
    {
        let min_v = _mm256_set1_epi32(col_clip_min);
        let max_v = _mm256_set1_epi32(col_clip_max);
        let mut v = [_mm256_setzero_si256(); 8];
        for i in 0..8 {
            v[i] = loadu_256!(&tmp[i * 8..i * 8 + 8], [i32; 8]);
        }
        dct8_1d_cols8(_token, &mut v, min_v, max_v);
        for i in 0..8 {
            storeu_256!(&mut tmp[i * 8..i * 8 + 8], [i32; 8], v[i]);
        }
    }

    // Add to destination with SIMD (16bpc = u16 pixels)
    let zero = _mm_setzero_si128();
    let max_val = _mm_set1_epi32(bitdepth_max);
    let rnd_final = _mm_set1_epi32(8);

    for y in 0..8 {
        let dst_off = y * stride_u16;

        // Load destination pixels (8 u16 = 16 bytes)
        let d = loadu_128!(<&[u16; 8]>::try_from(&dst[dst_off..dst_off + 8]).unwrap());
        let d_lo = _mm_unpacklo_epi16(d, zero); // First 4 as i32
        let d_hi = _mm_unpackhi_epi16(d, zero); // Last 4 as i32

        // Load 8 contiguous i32 coefficients in one AVX2 load, split into two SSE halves
        let c_256 = loadu_256!(&tmp[y * 8..y * 8 + 8], [i32; 8]);
        let c_lo = _mm256_castsi256_si128(c_256);
        let c_hi = _mm256_extracti128_si256(c_256, 1);

        // Final scaling: (c + 8) >> 4
        let c_lo_scaled = _mm_srai_epi32(_mm_add_epi32(c_lo, rnd_final), 4);
        let c_hi_scaled = _mm_srai_epi32(_mm_add_epi32(c_hi, rnd_final), 4);

        // Add to destination
        let sum_lo = _mm_add_epi32(d_lo, c_lo_scaled);
        let sum_hi = _mm_add_epi32(d_hi, c_hi_scaled);

        // Clamp to [0, bitdepth_max]
        let clamped_lo = _mm_max_epi32(_mm_min_epi32(sum_lo, max_val), zero);
        let clamped_hi = _mm_max_epi32(_mm_min_epi32(sum_hi, max_val), zero);

        // Pack to u16 and store
        let packed = _mm_packus_epi32(clamped_lo, clamped_hi);
        storeu_128!(
            <&mut [u16; 8]>::try_from(&mut dst[dst_off..dst_off + 8]).unwrap(),
            packed
        );
    }

    // Clear coefficients
    coeff[..64].fill(0);
}

/// FFI wrapper for 8x8 DCT_DCT 16bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_dct_dct_8x8_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u16, _coeff_len as usize * stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_txfm_add_dct_dct_8x8_16bpc_avx2_inner(
        _token,
        dst_slice,
        stride,
        coeff_slice,
        eob,
        bitdepth_max,
    );
}

// ============================================================================
// 16x16 IDTX (Identity)
// ============================================================================

/// 16x16 IDTX (identity transform)
/// Identity16: out = 2 * in + (in * 1697 + 1024) >> 11
/// For 16x16 IDTX: apply identity16 to rows, then identity16 to cols
/// Plus final shift: (+ 8) >> 4
#[cfg(target_arch = "x86_64")]
#[arcane]
pub fn inv_identity_add_16x16_8bpc_avx2(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mut coeff = coeff.flex_mut();
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi16(bitdepth_max as i16);

    // Identity16 scale factor: f(x) = 2*x + (x*1697 + 1024) >> 11
    // For 16x16: row_pass → intermediate shift (+ 2) >> 2 with clamp → col_pass → (+ 8) >> 4
    // The intermediate shift is shift=2, rnd=2 for 16x16 (from inv_txfm_add_rust)
    // Clamp to col_clip range: i16::MIN..=i16::MAX for 8bpc

    // Row pass: identity16 on each row
    let mut tmp = [[0i32; 16]; 16];
    for y in 0..16 {
        for x in 0..16 {
            let c = coeff[y + x * 16] as i32;
            let r = 2 * c + ((c * 1697 + 1024) >> 11);
            tmp[y][x] = r;
        }
    }

    // Intermediate shift: (val + 2) >> 2, clamped to i16 range
    for y in 0..16 {
        for x in 0..16 {
            tmp[y][x] = ((tmp[y][x] + 2) >> 2).clamp(i16::MIN as i32, i16::MAX as i32);
        }
    }

    // Column pass: identity16, then final shift
    for x in 0..16 {
        for y in 0..16 {
            let c = tmp[y][x];
            let r = 2 * c + ((c * 1697 + 1024) >> 11);
            tmp[y][x] = (r + 8) >> 4;
        }
    }

    // Add to destination with SIMD
    for y in 0..16 {
        let dst_off = y * dst_stride;

        // Load 16 destination pixels
        let d = loadu_128!(<&[u8; 16]>::try_from(&dst[dst_off..dst_off + 16]).unwrap());
        let d_lo = _mm256_cvtepu8_epi16(d);

        // Load 16 transformed coefficients
        let c_vec = _mm256_set_epi16(
            tmp[y][15] as i16,
            tmp[y][14] as i16,
            tmp[y][13] as i16,
            tmp[y][12] as i16,
            tmp[y][11] as i16,
            tmp[y][10] as i16,
            tmp[y][9] as i16,
            tmp[y][8] as i16,
            tmp[y][7] as i16,
            tmp[y][6] as i16,
            tmp[y][5] as i16,
            tmp[y][4] as i16,
            tmp[y][3] as i16,
            tmp[y][2] as i16,
            tmp[y][1] as i16,
            tmp[y][0] as i16,
        );

        // Add and clamp
        let sum = _mm256_add_epi16(d_lo, c_vec);
        let clamped = _mm256_max_epi16(_mm256_min_epi16(sum, max_val), zero);

        // Pack to bytes
        let packed = _mm256_packus_epi16(clamped, clamped);
        let packed_lo = _mm256_castsi256_si128(packed);
        let packed_hi = _mm256_extracti128_si256(packed, 1);
        let result = _mm_unpacklo_epi64(packed_lo, packed_hi);

        storeu_128!(
            <&mut [u8; 16]>::try_from(&mut dst[dst_off..dst_off + 16]).unwrap(),
            result
        );
    }

    // Clear coefficients (16x16 = 256 i16 = 512 bytes = 32 x 16-byte stores)
    coeff[..256].fill(0);
}

/// FFI wrapper for 16x16 IDTX 8bpc
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(feature = "asm")]
pub unsafe extern "C" fn inv_txfm_add_identity_identity_16x16_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: c_int,
    bitdepth_max: c_int,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) {
    let _token = unsafe { Desktop64::forge_token_dangerously() };
    let stride = dst_stride as usize;

    let dst_slice = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, _coeff_len as usize * stride + stride)
    };

    let coeff_slice =
        unsafe { std::slice::from_raw_parts_mut(coeff as *mut i16, _coeff_len as usize) };

    inv_identity_add_16x16_8bpc_avx2(_token, dst_slice, stride, coeff_slice, eob, bitdepth_max);
}

// ============================================================================
// 16x16 DCT_DCT
// ============================================================================

/// DCT16 1D transform
#[inline]
fn dct16_1d(c: &mut [i32], stride: usize, min: i32, max: i32) {
    let clip = |v: i32| v.clamp(min, max);

    // First apply DCT8 to even positions
    dct8_1d(c, stride * 2, min, max);

    let in1 = c[1 * stride];
    let in3 = c[3 * stride];
    let in5 = c[5 * stride];
    let in7 = c[7 * stride];
    let in9 = c[9 * stride];
    let in11 = c[11 * stride];
    let in13 = c[13 * stride];
    let in15 = c[15 * stride];

    let t8a = (in1 * 401 - in15 * (4076 - 4096) + 2048 >> 12) - in15;
    let t9a = in9 * 1583 - in7 * 1299 + 1024 >> 11;
    let t10a = (in5 * 1931 - in11 * (3612 - 4096) + 2048 >> 12) - in11;
    let t11a = (in13 * (3920 - 4096) - in3 * 1189 + 2048 >> 12) + in13;
    let t12a = (in13 * 1189 + in3 * (3920 - 4096) + 2048 >> 12) + in3;
    let t13a = (in5 * (3612 - 4096) + in11 * 1931 + 2048 >> 12) + in5;
    let t14a = in9 * 1299 + in7 * 1583 + 1024 >> 11;
    let t15a = (in1 * (4076 - 4096) + in15 * 401 + 2048 >> 12) + in1;

    let t8 = clip(t8a + t9a);
    let mut t9 = clip(t8a - t9a);
    let mut t10 = clip(t11a - t10a);
    let mut t11 = clip(t11a + t10a);
    let mut t12 = clip(t12a + t13a);
    let mut t13 = clip(t12a - t13a);
    let mut t14 = clip(t15a - t14a);
    let t15 = clip(t15a + t14a);

    let t9a = (t14 * 1567 - t9 * (3784 - 4096) + 2048 >> 12) - t9;
    let t14a = (t14 * (3784 - 4096) + t9 * 1567 + 2048 >> 12) + t14;
    let t10a = (-(t13 * (3784 - 4096) + t10 * 1567) + 2048 >> 12) - t13;
    let t13a = (t13 * 1567 - t10 * (3784 - 4096) + 2048 >> 12) - t10;

    let t8a = clip(t8 + t11);
    t9 = clip(t9a + t10a);
    t10 = clip(t9a - t10a);
    let t11a = clip(t8 - t11);
    let t12a = clip(t15 - t12);
    t13 = clip(t14a - t13a);
    t14 = clip(t14a + t13a);
    let t15a = clip(t15 + t12);

    let t10a_new = (t13 - t10) * 181 + 128 >> 8;
    let t13a_new = (t13 + t10) * 181 + 128 >> 8;
    t11 = (t12a - t11a) * 181 + 128 >> 8;
    t12 = (t12a + t11a) * 181 + 128 >> 8;

    let t0 = c[0 * stride];
    let t1 = c[2 * stride];
    let t2 = c[4 * stride];
    let t3 = c[6 * stride];
    let t4 = c[8 * stride];
    let t5 = c[10 * stride];
    let t6 = c[12 * stride];
    let t7 = c[14 * stride];

    c[0 * stride] = clip(t0 + t15a);
    c[1 * stride] = clip(t1 + t14);
    c[2 * stride] = clip(t2 + t13a_new);
    c[3 * stride] = clip(t3 + t12);
    c[4 * stride] = clip(t4 + t11);
    c[5 * stride] = clip(t5 + t10a_new);
    c[6 * stride] = clip(t6 + t9);
    c[7 * stride] = clip(t7 + t8a);
    c[8 * stride] = clip(t7 - t8a);
    c[9 * stride] = clip(t6 - t9);
    c[10 * stride] = clip(t5 - t10a_new);
    c[11 * stride] = clip(t4 - t11);
    c[12 * stride] = clip(t3 - t12);
    c[13 * stride] = clip(t2 - t13a_new);
    c[14 * stride] = clip(t1 - t14);
    c[15 * stride] = clip(t0 - t15a);
}

/// ADST16 1D transform (in-place)
#[inline]
fn adst16_1d(c: &mut [i32], stride: usize, min: i32, max: i32) {
    let clip = |v: i32| v.clamp(min, max);

    let in0 = c[0 * stride];
    let in1 = c[1 * stride];
    let in2 = c[2 * stride];
    let in3 = c[3 * stride];
    let in4 = c[4 * stride];
    let in5 = c[5 * stride];
    let in6 = c[6 * stride];
    let in7 = c[7 * stride];
    let in8 = c[8 * stride];
    let in9 = c[9 * stride];
    let in10 = c[10 * stride];
    let in11 = c[11 * stride];
    let in12 = c[12 * stride];
    let in13 = c[13 * stride];
    let in14 = c[14 * stride];
    let in15 = c[15 * stride];

    let mut t0 = ((in15 * (4091 - 4096) + in0 * 201 + 2048) >> 12) + in15;
    let mut t1 = ((in15 * 201 - in0 * (4091 - 4096) + 2048) >> 12) - in0;
    let mut t2 = ((in13 * (3973 - 4096) + in2 * 995 + 2048) >> 12) + in13;
    let mut t3 = ((in13 * 995 - in2 * (3973 - 4096) + 2048) >> 12) - in2;
    let mut t4 = ((in11 * (3703 - 4096) + in4 * 1751 + 2048) >> 12) + in11;
    let mut t5 = ((in11 * 1751 - in4 * (3703 - 4096) + 2048) >> 12) - in4;
    let mut t6 = (in9 * 1645 + in6 * 1220 + 1024) >> 11;
    let mut t7 = (in9 * 1220 - in6 * 1645 + 1024) >> 11;
    let mut t8 = ((in7 * 2751 + in8 * (3035 - 4096) + 2048) >> 12) + in8;
    let mut t9 = ((in7 * (3035 - 4096) - in8 * 2751 + 2048) >> 12) + in7;
    let mut t10 = ((in5 * 2106 + in10 * (3513 - 4096) + 2048) >> 12) + in10;
    let mut t11 = ((in5 * (3513 - 4096) - in10 * 2106 + 2048) >> 12) + in5;
    let mut t12 = ((in3 * 1380 + in12 * (3857 - 4096) + 2048) >> 12) + in12;
    let mut t13 = ((in3 * (3857 - 4096) - in12 * 1380 + 2048) >> 12) + in3;
    let mut t14 = ((in1 * 601 + in14 * (4052 - 4096) + 2048) >> 12) + in14;
    let mut t15 = ((in1 * (4052 - 4096) - in14 * 601 + 2048) >> 12) + in1;

    let t0a = clip(t0 + t8);
    let t1a = clip(t1 + t9);
    let mut t2a = clip(t2 + t10);
    let mut t3a = clip(t3 + t11);
    let mut t4a = clip(t4 + t12);
    let mut t5a = clip(t5 + t13);
    let mut t6a = clip(t6 + t14);
    let mut t7a = clip(t7 + t15);
    let mut t8a = clip(t0 - t8);
    let mut t9a = clip(t1 - t9);
    let mut t10a = clip(t2 - t10);
    let mut t11a = clip(t3 - t11);
    let mut t12a = clip(t4 - t12);
    let mut t13a = clip(t5 - t13);
    let mut t14a = clip(t6 - t14);
    let mut t15a = clip(t7 - t15);

    t8 = ((t8a * (4017 - 4096) + t9a * 799 + 2048) >> 12) + t8a;
    t9 = ((t8a * 799 - t9a * (4017 - 4096) + 2048) >> 12) - t9a;
    t10 = ((t10a * 2276 + t11a * (3406 - 4096) + 2048) >> 12) + t11a;
    t11 = ((t10a * (3406 - 4096) - t11a * 2276 + 2048) >> 12) + t10a;
    t12 = ((t13a * (4017 - 4096) - t12a * 799 + 2048) >> 12) + t13a;
    t13 = ((t13a * 799 + t12a * (4017 - 4096) + 2048) >> 12) + t12a;
    t14 = ((t15a * 2276 - t14a * (3406 - 4096) + 2048) >> 12) - t14a;
    t15 = ((t15a * (3406 - 4096) + t14a * 2276 + 2048) >> 12) + t15a;

    t0 = clip(t0a + t4a);
    t1 = clip(t1a + t5a);
    t2 = clip(t2a + t6a);
    t3 = clip(t3a + t7a);
    t4 = clip(t0a - t4a);
    t5 = clip(t1a - t5a);
    t6 = clip(t2a - t6a);
    t7 = clip(t3a - t7a);
    t8a = clip(t8 + t12);
    t9a = clip(t9 + t13);
    t10a = clip(t10 + t14);
    t11a = clip(t11 + t15);
    t12a = clip(t8 - t12);
    t13a = clip(t9 - t13);
    t14a = clip(t10 - t14);
    t15a = clip(t11 - t15);

    t4a = ((t4 * (3784 - 4096) + t5 * 1567 + 2048) >> 12) + t4;
    t5a = ((t4 * 1567 - t5 * (3784 - 4096) + 2048) >> 12) - t5;
    t6a = ((t7 * (3784 - 4096) - t6 * 1567 + 2048) >> 12) + t7;
    t7a = ((t7 * 1567 + t6 * (3784 - 4096) + 2048) >> 12) + t6;
    t12 = ((t12a * (3784 - 4096) + t13a * 1567 + 2048) >> 12) + t12a;
    t13 = ((t12a * 1567 - t13a * (3784 - 4096) + 2048) >> 12) - t13a;
    t14 = ((t15a * (3784 - 4096) - t14a * 1567 + 2048) >> 12) + t15a;
    t15 = ((t15a * 1567 + t14a * (3784 - 4096) + 2048) >> 12) + t14a;

    c[0 * stride] = clip(t0 + t2);
    c[15 * stride] = -clip(t1 + t3);
    t2a = clip(t0 - t2);
    t3a = clip(t1 - t3);
    c[3 * stride] = -clip(t4a + t6a);
    c[12 * stride] = clip(t5a + t7a);
    t6 = clip(t4a - t6a);
    t7 = clip(t5a - t7a);
    c[1 * stride] = -clip(t8a + t10a);
    c[14 * stride] = clip(t9a + t11a);
    t10 = clip(t8a - t10a);
    t11 = clip(t9a - t11a);
    c[2 * stride] = clip(t12 + t14);
    c[13 * stride] = -clip(t13 + t15);
    t14a = clip(t12 - t14);
    t15a = clip(t13 - t15);

    c[7 * stride] = -(((t2a + t3a) * 181 + 128) >> 8);
    c[8 * stride] = ((t2a - t3a) * 181 + 128) >> 8;
    c[4 * stride] = ((t6 + t7) * 181 + 128) >> 8;
    c[11 * stride] = -(((t6 - t7) * 181 + 128) >> 8);
    c[6 * stride] = ((t10 + t11) * 181 + 128) >> 8;
    c[9 * stride] = -(((t10 - t11) * 181 + 128) >> 8);
    c[5 * stride] = -(((t14a + t15a) * 181 + 128) >> 8);
    c[10 * stride] = ((t14a - t15a) * 181 + 128) >> 8;
}

/// FlipADST16 1D transform (in-place)
#[inline]
fn flipadst16_1d(c: &mut [i32], stride: usize, min: i32, max: i32) {
    // Apply ADST then reverse
    adst16_1d(c, stride, min, max);
    // Swap in place
    for i in 0..8 {
        c.swap(i * stride, (15 - i) * stride);
    }
}

/// Identity4 1D transform (strided, in-place)
#[inline]
fn identity4_1d(c: &mut [i32], stride: usize, _min: i32, _max: i32) {
    // Identity4: out = in + (in * 1697 + 2048) >> 12
    // This is approximately in * sqrt(2) ≈ in * 1.414
    for i in 0..4 {
        let in_0 = c[i * stride];
        c[i * stride] = in_0 + (in_0 * 1697 + 2048 >> 12);
    }
}

/// Identity8 1D transform (strided, in-place)
#[inline]
fn identity8_1d(c: &mut [i32], stride: usize, _min: i32, _max: i32) {
    // For 8pt identity: out = in * 2
    for i in 0..8 {
        c[i * stride] *= 2;
    }
}

/// Identity16 1D transform (in-place)
#[inline]
fn identity16_1d(c: &mut [i32], stride: usize, _min: i32, _max: i32) {
    // Identity16: out = 2 * in + (in * 1697 + 1024) >> 11
    // This is approximately in * 2 * sqrt(2) ≈ in * 2.829
    for i in 0..16 {
        let in_0 = c[i * stride];
        c[i * stride] = 2 * in_0 + (in_0 * 1697 + 1024 >> 11);
    }
}

