//! Safe SIMD implementations for CDEF (Constrained Directional Enhancement Filter)
//!
//! CDEF applies direction-dependent filtering to remove coding artifacts
//! while preserving edges.

#![deny(unsafe_code)]
#![allow(unused_imports)]
#![allow(dead_code)]

#[cfg(target_arch = "x86_64")]
use archmage::{arcane, rite, Desktop64, SimdToken};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use std::ffi::c_int;
use std::ffi::c_uint;

use crate::include::common::bitdepth::AsPrimitive;
use crate::include::common::bitdepth::BitDepth;
use crate::include::common::bitdepth::DynPixel;
use crate::include::common::bitdepth::LeftPixelRow2px;
use crate::include::common::intops::apply_sign;
use crate::include::dav1d::picture::PicOffset;
use crate::src::align::AlignedVec64;
use crate::src::cdef::CdefBottom;
use crate::src::cdef::CdefEdgeFlags;
use crate::src::cdef::CdefTop;
use crate::src::disjoint_mut::DisjointMut;
use crate::src::ffi_safe::FFISafe;
use crate::src::pic_or_buf::PicOrBuf;
use crate::src::strided::Strided as _;
use crate::src::tables::dav1d_cdef_directions;
use crate::src::with_offset::WithOffset;
use crate::src::safe_simd::pixel_access::Flex;
use libc::ptrdiff_t;
use std::cmp;

// ============================================================================
// CONSTRAIN FUNCTION (SIMD)
// ============================================================================

/// SIMD version of constrain for AVX2
/// Processes 16 i16 values at once
/// Formula: sign(diff) * min(|diff|, max(0, threshold - (|diff| >> shift)))
#[cfg(target_arch = "x86_64")]
#[rite]
fn constrain_avx2(_t: Desktop64, diff: __m256i, threshold: __m256i, shift: __m128i) -> __m256i {
    let zero = _mm256_setzero_si256();

    // Compute absolute value
    let adiff = _mm256_abs_epi16(diff);

    // Compute threshold - (adiff >> shift)
    let shifted = _mm256_sra_epi16(adiff, shift);
    let term = _mm256_sub_epi16(threshold, shifted);

    // max(0, term)
    let max_term = _mm256_max_epi16(term, zero);

    // min(adiff, max_term)
    let result_abs = _mm256_min_epi16(adiff, max_term);

    // Apply sign of original diff
    // If diff >= 0: result = result_abs
    // If diff < 0: result = -result_abs
    let sign_mask = _mm256_cmpgt_epi16(zero, diff);
    let neg_result = _mm256_sub_epi16(zero, result_abs);
    _mm256_blendv_epi8(result_abs, neg_result, sign_mask)
}

/// 128-bit constrain — processes 8 i16 values at once (one row of 8 pixels).
#[cfg(target_arch = "x86_64")]
#[rite]
fn constrain_128(_t: Desktop64, diff: __m128i, threshold: __m128i, shift: __m128i) -> __m128i {
    let zero = _mm_setzero_si128();
    let adiff = _mm_abs_epi16(diff);
    let shifted = _mm_sra_epi16(adiff, shift);
    let term = _mm_sub_epi16(threshold, shifted);
    let max_term = _mm_max_epi16(term, zero);
    let result_abs = _mm_min_epi16(adiff, max_term);
    let sign_mask = _mm_cmpgt_epi16(zero, diff);
    let neg_result = _mm_sub_epi16(zero, result_abs);
    _mm_blendv_epi8(result_abs, neg_result, sign_mask)
}

/// Vectorized CDEF filter for 8bpc — processes 8 pixels per row using SSE.
/// Handles all block sizes (8x8, 4x8, 4x4) via w/h parameters.
/// For w=4, processes 8 lanes but only stores the low 4 bytes.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn cdef_filter_block_simd_8bpc(
    t: Desktop64,
    tmp: &[u16],
    tmp_offset: usize,
    dst: PicOffset,
    stride: isize,
    w: usize,
    h: usize,
    dir: usize,
    pri_strength: c_int,
    sec_strength: c_int,
    damping: c_int,
) {
    use super::pixel_access::{loadu_128, storei32, storeu_128};
    use crate::include::common::bitdepth::BitDepth8;

    let zero = _mm_setzero_si128();

    // Single guard for entire output region
    let (mut p_guard, p_base) = dst.strided_slice_mut::<BitDepth8>(w, h);

    if pri_strength != 0 {
        let pri_tap = 4 - (pri_strength & 1);
        let pri_shift = cmp::max(0, damping - pri_strength.ilog2() as c_int);
        let pri_thresh = _mm_set1_epi16(pri_strength as i16);
        let pri_shift_v = _mm_cvtsi32_si128(pri_shift);

        if sec_strength != 0 {
            // Both primary and secondary — full filter with min/max clamping
            let sec_shift = damping - sec_strength.ilog2() as c_int;
            let sec_thresh = _mm_set1_epi16(sec_strength as i16);
            let sec_shift_v = _mm_cvtsi32_si128(sec_shift);

            for y in 0..h {
                let base = tmp_offset + y * TMP_STRIDE;
                let px = loadu_128!(&tmp[base..base + 8], [u16; 8]);
                let mut sum = zero;
                let mut min_v = px;
                let mut max_v = px;

                let mut pri_tap_k = pri_tap;
                for k in 0..2 {
                    let off = dav1d_cdef_directions[dir + 2][k] as isize;
                    let p0_i = (base as isize + off) as usize;
                    let p1_i = (base as isize - off) as usize;
                    let p0 = loadu_128!(&tmp[p0_i..p0_i + 8], [u16; 8]);
                    let p1 = loadu_128!(&tmp[p1_i..p1_i + 8], [u16; 8]);

                    let c0 = constrain_128(t, _mm_sub_epi16(p0, px), pri_thresh, pri_shift_v);
                    let c1 = constrain_128(t, _mm_sub_epi16(p1, px), pri_thresh, pri_shift_v);

                    let tap_v = _mm_set1_epi16(pri_tap_k as i16);
                    sum = _mm_add_epi16(sum, _mm_mullo_epi16(tap_v, _mm_add_epi16(c0, c1)));
                    pri_tap_k = pri_tap_k & 3 | 2;

                    // Use unsigned min to ignore boundary fill (0x8001 = large unsigned)
                    min_v = _mm_min_epu16(min_v, _mm_min_epu16(p0, p1));
                    // Use signed max to ignore boundary fill (0x8001 = -32767 signed)
                    max_v = _mm_max_epi16(max_v, _mm_max_epi16(p0, p1));

                    let off2 = dav1d_cdef_directions[dir + 4][k] as isize;
                    let off3 = dav1d_cdef_directions[dir + 0][k] as isize;
                    let s0_i = (base as isize + off2) as usize;
                    let s1_i = (base as isize - off2) as usize;
                    let s2_i = (base as isize + off3) as usize;
                    let s3_i = (base as isize - off3) as usize;
                    let s0 = loadu_128!(&tmp[s0_i..s0_i + 8], [u16; 8]);
                    let s1 = loadu_128!(&tmp[s1_i..s1_i + 8], [u16; 8]);
                    let s2 = loadu_128!(&tmp[s2_i..s2_i + 8], [u16; 8]);
                    let s3 = loadu_128!(&tmp[s3_i..s3_i + 8], [u16; 8]);

                    let sec_tap_k = (2 - k as i32) as i16;
                    let sec_tap_v = _mm_set1_epi16(sec_tap_k);
                    let ds0 = constrain_128(t, _mm_sub_epi16(s0, px), sec_thresh, sec_shift_v);
                    let ds1 = constrain_128(t, _mm_sub_epi16(s1, px), sec_thresh, sec_shift_v);
                    let ds2 = constrain_128(t, _mm_sub_epi16(s2, px), sec_thresh, sec_shift_v);
                    let ds3 = constrain_128(t, _mm_sub_epi16(s3, px), sec_thresh, sec_shift_v);

                    let sec_sum = _mm_add_epi16(_mm_add_epi16(ds0, ds1), _mm_add_epi16(ds2, ds3));
                    sum = _mm_add_epi16(sum, _mm_mullo_epi16(sec_tap_v, sec_sum));

                    min_v = _mm_min_epu16(
                        min_v,
                        _mm_min_epu16(_mm_min_epu16(s0, s1), _mm_min_epu16(s2, s3)),
                    );
                    max_v = _mm_max_epi16(
                        max_v,
                        _mm_max_epi16(_mm_max_epi16(s0, s1), _mm_max_epi16(s2, s3)),
                    );
                }

                // Rounding: (sum - (sum < 0) + 8) >> 4
                let neg_mask = _mm_cmpgt_epi16(zero, sum);
                let adjusted = _mm_add_epi16(sum, neg_mask);
                let adjusted = _mm_add_epi16(adjusted, _mm_set1_epi16(8));
                let adjusted = _mm_srai_epi16::<4>(adjusted);
                let result = _mm_add_epi16(px, adjusted);
                let result = _mm_max_epi16(result, min_v);
                let result = _mm_min_epi16(result, max_v);
                let result_u8 = _mm_packus_epi16(result, zero);

                let row_off = p_base.wrapping_add_signed(y as isize * stride);
                if w == 8 {
                    let mut out = [0u8; 16];
                    storeu_128!(&mut out, result_u8);
                    p_guard[row_off..row_off + 8].copy_from_slice(&out[0..8]);
                } else {
                    storei32!(&mut p_guard[row_off..row_off + 4], result_u8);
                }
            }
        } else {
            // Primary only — no min/max clamping
            for y in 0..h {
                let base = tmp_offset + y * TMP_STRIDE;
                let px = loadu_128!(&tmp[base..base + 8], [u16; 8]);
                let mut sum = zero;

                let mut pri_tap_k = pri_tap;
                for k in 0..2 {
                    let off = dav1d_cdef_directions[dir + 2][k] as isize;
                    let p0_i = (base as isize + off) as usize;
                    let p1_i = (base as isize - off) as usize;
                    let p0 = loadu_128!(&tmp[p0_i..p0_i + 8], [u16; 8]);
                    let p1 = loadu_128!(&tmp[p1_i..p1_i + 8], [u16; 8]);

                    let c0 = constrain_128(t, _mm_sub_epi16(p0, px), pri_thresh, pri_shift_v);
                    let c1 = constrain_128(t, _mm_sub_epi16(p1, px), pri_thresh, pri_shift_v);

                    let tap_v = _mm_set1_epi16(pri_tap_k as i16);
                    sum = _mm_add_epi16(sum, _mm_mullo_epi16(tap_v, _mm_add_epi16(c0, c1)));
                    pri_tap_k = pri_tap_k & 3 | 2;
                }

                let neg_mask = _mm_cmpgt_epi16(zero, sum);
                let adjusted = _mm_add_epi16(sum, neg_mask);
                let adjusted = _mm_add_epi16(adjusted, _mm_set1_epi16(8));
                let adjusted = _mm_srai_epi16::<4>(adjusted);
                let result = _mm_add_epi16(px, adjusted);
                let result_u8 = _mm_packus_epi16(result, zero);

                let row_off = p_base.wrapping_add_signed(y as isize * stride);
                if w == 8 {
                    let mut out = [0u8; 16];
                    storeu_128!(&mut out, result_u8);
                    p_guard[row_off..row_off + 8].copy_from_slice(&out[0..8]);
                } else {
                    storei32!(&mut p_guard[row_off..row_off + 4], result_u8);
                }
            }
        }
    } else {
        // Secondary only — no min/max clamping
        let sec_shift = damping - sec_strength.ilog2() as c_int;
        let sec_thresh = _mm_set1_epi16(sec_strength as i16);
        let sec_shift_v = _mm_cvtsi32_si128(sec_shift);

        for y in 0..h {
            let base = tmp_offset + y * TMP_STRIDE;
            let px = loadu_128!(&tmp[base..base + 8], [u16; 8]);
            let mut sum = zero;

            for k in 0..2 {
                let off2 = dav1d_cdef_directions[dir + 4][k] as isize;
                let off3 = dav1d_cdef_directions[dir + 0][k] as isize;
                let s0_i = (base as isize + off2) as usize;
                let s1_i = (base as isize - off2) as usize;
                let s2_i = (base as isize + off3) as usize;
                let s3_i = (base as isize - off3) as usize;
                let s0 = loadu_128!(&tmp[s0_i..s0_i + 8], [u16; 8]);
                let s1 = loadu_128!(&tmp[s1_i..s1_i + 8], [u16; 8]);
                let s2 = loadu_128!(&tmp[s2_i..s2_i + 8], [u16; 8]);
                let s3 = loadu_128!(&tmp[s3_i..s3_i + 8], [u16; 8]);

                let sec_tap_k = (2 - k as i32) as i16;
                let sec_tap_v = _mm_set1_epi16(sec_tap_k);
                let ds0 = constrain_128(t, _mm_sub_epi16(s0, px), sec_thresh, sec_shift_v);
                let ds1 = constrain_128(t, _mm_sub_epi16(s1, px), sec_thresh, sec_shift_v);
                let ds2 = constrain_128(t, _mm_sub_epi16(s2, px), sec_thresh, sec_shift_v);
                let ds3 = constrain_128(t, _mm_sub_epi16(s3, px), sec_thresh, sec_shift_v);

                let sec_sum = _mm_add_epi16(_mm_add_epi16(ds0, ds1), _mm_add_epi16(ds2, ds3));
                sum = _mm_add_epi16(sum, _mm_mullo_epi16(sec_tap_v, sec_sum));
            }

            let neg_mask = _mm_cmpgt_epi16(zero, sum);
            let adjusted = _mm_add_epi16(sum, neg_mask);
            let adjusted = _mm_add_epi16(adjusted, _mm_set1_epi16(8));
            let adjusted = _mm_srai_epi16::<4>(adjusted);
            let result = _mm_add_epi16(px, adjusted);
            let result_u8 = _mm_packus_epi16(result, zero);

            let row_off = p_base.wrapping_add_signed(y as isize * stride);
            if w == 8 {
                let mut out = [0u8; 16];
                storeu_128!(&mut out, result_u8);
                p_guard[row_off..row_off + 8].copy_from_slice(&out[0..8]);
            } else {
                storei32!(&mut p_guard[row_off..row_off + 4], result_u8);
            }
        }
    }
}

// ============================================================================
// CDEF DIRECTION FINDING
// ============================================================================

/// SSE SIMD implementation of cdef_find_dir for 8bpc.
/// Vectorizes diagonal/alt partial sum accumulation: SIMD load/add/store of 8 i16
/// values at variable offsets, reducing 8 scalar adds to 1 SIMD add per direction.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn cdef_find_dir_simd_8bpc(_t: Desktop64, img: PicOffset, variance: &mut c_uint) -> c_int {
    use super::pixel_access::{loadu_128, storeu_128};
    use crate::include::common::bitdepth::BitDepth8;

    let sub128 = _mm_set1_epi16(128);

    // Padded to 16 i16 for unaligned SIMD access at variable offsets.
    let mut partial_sum_diag0 = [0i16; 16]; // indices 0..14
    let mut partial_sum_diag1 = [0i16; 16]; // indices 0..14
    let mut partial_sum_alt0 = [0i16; 16]; // indices 0..10
    let mut partial_sum_alt1 = [0i16; 16]; // indices 0..10
    let mut partial_sum_alt2 = [0i16; 16]; // indices 0..10
    let mut partial_sum_alt3 = [0i16; 16]; // indices 0..10
    let mut hv0 = [0i16; 8];
    let mut hv1_vec = _mm_setzero_si128();

    let stride = img.pixel_stride::<BitDepth8>();

    for y in 0..8usize {
        let row_img = img + (y as isize * stride);
        let row_slice = row_img.slice::<BitDepth8>(8);

        // Load 8 u8 pixels → i16 → subtract 128
        let mut row_bytes = [0u8; 16];
        row_bytes[0..8].copy_from_slice(&row_slice[..8]);
        let row_u8 = loadu_128!(&row_bytes);
        let row = _mm_sub_epi16(_mm_cvtepu8_epi16(row_u8), sub128);

        // Extract to array for scatter operations (alt[0], alt[1], hv[0])
        let mut px = [0i16; 8];
        storeu_128!(&mut px, row);

        // hv[0][y] = sum of row
        hv0[y] = px[0] + px[1] + px[2] + px[3] + px[4] + px[5] + px[6] + px[7];

        // hv[1]: vertical accumulation (each lane = one column)
        hv1_vec = _mm_add_epi16(hv1_vec, row);

        // diag[0][y+x]: 8 consecutive values at offset y (SIMD add)
        let d0 = loadu_128!(&partial_sum_diag0[y..y + 8], [i16; 8]);
        storeu_128!(
            <&mut [i16; 8]>::try_from(&mut partial_sum_diag0[y..y + 8]).unwrap(),
            _mm_add_epi16(d0, row)
        );

        // diag[1][7+y-x]: reversed row at offset y (SIMD byte-reverse + add)
        let rev = _mm_shuffle_epi8(
            row,
            _mm_set_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14),
        );
        let d1 = loadu_128!(&partial_sum_diag1[y..y + 8], [i16; 8]);
        storeu_128!(
            <&mut [i16; 8]>::try_from(&mut partial_sum_diag1[y..y + 8]).unwrap(),
            _mm_add_epi16(d1, rev)
        );

        // alt[0][y + (x >> 1)]: pairwise sums → 4 values at offset y
        let pair0 = px[0] + px[1];
        let pair1 = px[2] + px[3];
        let pair2 = px[4] + px[5];
        let pair3 = px[6] + px[7];
        partial_sum_alt0[y] += pair0;
        partial_sum_alt0[y + 1] += pair1;
        partial_sum_alt0[y + 2] += pair2;
        partial_sum_alt0[y + 3] += pair3;

        // alt[1][3 + y - (x >> 1)]: reversed pairwise sums at offset y
        partial_sum_alt1[y] += pair3;
        partial_sum_alt1[y + 1] += pair2;
        partial_sum_alt1[y + 2] += pair1;
        partial_sum_alt1[y + 3] += pair0;

        // alt[2][3 - (y >> 1) + x]: 8 consecutive at offset (3 - y/2) (SIMD add)
        let base2 = 3 - (y >> 1);
        let a2 = loadu_128!(&partial_sum_alt2[base2..base2 + 8], [i16; 8]);
        storeu_128!(
            <&mut [i16; 8]>::try_from(&mut partial_sum_alt2[base2..base2 + 8]).unwrap(),
            _mm_add_epi16(a2, row)
        );

        // alt[3][(y >> 1) + x]: 8 consecutive at offset (y/2) (SIMD add)
        let base3 = y >> 1;
        let a3 = loadu_128!(&partial_sum_alt3[base3..base3 + 8], [i16; 8]);
        storeu_128!(
            <&mut [i16; 8]>::try_from(&mut partial_sum_alt3[base3..base3 + 8]).unwrap(),
            _mm_add_epi16(a3, row)
        );
    }

    let mut hv1 = [0i16; 8];
    storeu_128!(&mut hv1, hv1_vec);

    // === Cost computation (scalar — small fixed-size loops) ===
    let mut cost = [0u32; 8];
    for n in 0..8 {
        cost[2] += (hv0[n] as i32 * hv0[n] as i32) as u32;
        cost[6] += (hv1[n] as i32 * hv1[n] as i32) as u32;
    }
    cost[2] *= 105;
    cost[6] *= 105;

    static DIV_TABLE: [u16; 7] = [840, 420, 280, 210, 168, 140, 120];
    for n in 0..7 {
        let d = DIV_TABLE[n] as i32;
        cost[0] += ((partial_sum_diag0[n] as i32 * partial_sum_diag0[n] as i32
            + partial_sum_diag0[14 - n] as i32 * partial_sum_diag0[14 - n] as i32)
            * d) as u32;
        cost[4] += ((partial_sum_diag1[n] as i32 * partial_sum_diag1[n] as i32
            + partial_sum_diag1[14 - n] as i32 * partial_sum_diag1[14 - n] as i32)
            * d) as u32;
    }
    cost[0] += (partial_sum_diag0[7] as i32 * partial_sum_diag0[7] as i32 * 105) as u32;
    cost[4] += (partial_sum_diag1[7] as i32 * partial_sum_diag1[7] as i32 * 105) as u32;

    let alt_arrays: [&[i16; 16]; 4] = [
        &partial_sum_alt0,
        &partial_sum_alt1,
        &partial_sum_alt2,
        &partial_sum_alt3,
    ];
    for n in 0..4 {
        let cost_ptr = &mut cost[n * 2 + 1];
        for m in 0..5 {
            let v = alt_arrays[n][3 + m] as i32;
            *cost_ptr += (v * v) as u32;
        }
        *cost_ptr *= 105;
        for m in 0..3 {
            let d = DIV_TABLE[2 * m + 1] as i32;
            let a = alt_arrays[n][m] as i32;
            let b = alt_arrays[n][10 - m] as i32;
            *cost_ptr += ((a * a + b * b) * d) as u32;
        }
    }

    let mut best_dir = 0;
    let mut best_cost = cost[0];
    for n in 0..8 {
        if cost[n] > best_cost {
            best_cost = cost[n];
            best_dir = n;
        }
    }

    *variance = (best_cost - cost[best_dir ^ 4]) >> 10;
    best_dir as c_int
}

/// Scalar implementation of cdef_find_dir (reference for SIMD development)
/// Returns direction (0-7) and sets variance
#[inline(never)]
fn cdef_find_dir_scalar<BD: BitDepth>(img: PicOffset, variance: &mut c_uint, bd: BD) -> c_int {
    let bitdepth_min_8 = bd.bitdepth() - 8;
    let mut partial_sum_hv = [[0i32; 8]; 2];
    let mut partial_sum_diag = [[0i32; 15]; 2];
    let mut partial_sum_alt = [[0i32; 11]; 4];

    const W: usize = 8;
    const H: usize = 8;

    for y in 0..H {
        let img = img + (y as isize * img.pixel_stride::<BD>());
        let img = &*img.slice::<BD>(W);
        for x in 0..W {
            let px = (img[x].as_::<c_int>() >> bitdepth_min_8) - 128;

            partial_sum_diag[0][y + x] += px;
            partial_sum_alt[0][y + (x >> 1)] += px;
            partial_sum_hv[0][y] += px;
            partial_sum_alt[1][3 + y - (x >> 1)] += px;
            partial_sum_diag[1][7 + y - x] += px;
            partial_sum_alt[2][3 - (y >> 1) + x] += px;
            partial_sum_hv[1][x] += px;
            partial_sum_alt[3][(y >> 1) + x] += px;
        }
    }

    let mut cost = [0u32; 8];
    for n in 0..8 {
        cost[2] += (partial_sum_hv[0][n] * partial_sum_hv[0][n]) as c_uint;
        cost[6] += (partial_sum_hv[1][n] * partial_sum_hv[1][n]) as c_uint;
    }
    cost[2] *= 105;
    cost[6] *= 105;

    static DIV_TABLE: [u16; 7] = [840, 420, 280, 210, 168, 140, 120];
    for n in 0..7 {
        let d = DIV_TABLE[n] as c_int;
        cost[0] += ((partial_sum_diag[0][n] * partial_sum_diag[0][n]
            + partial_sum_diag[0][14 - n] * partial_sum_diag[0][14 - n])
            * d) as c_uint;
        cost[4] += ((partial_sum_diag[1][n] * partial_sum_diag[1][n]
            + partial_sum_diag[1][14 - n] * partial_sum_diag[1][14 - n])
            * d) as c_uint;
    }
    cost[0] += (partial_sum_diag[0][7] * partial_sum_diag[0][7] * 105) as c_uint;
    cost[4] += (partial_sum_diag[1][7] * partial_sum_diag[1][7] * 105) as c_uint;

    for n in 0..4 {
        let cost_ptr = &mut cost[n * 2 + 1];
        for m in 0..5 {
            *cost_ptr += (partial_sum_alt[n][3 + m] * partial_sum_alt[n][3 + m]) as c_uint;
        }
        *cost_ptr *= 105;
        for m in 0..3 {
            let d = DIV_TABLE[2 * m + 1] as c_int;
            *cost_ptr += ((partial_sum_alt[n][m] * partial_sum_alt[n][m]
                + partial_sum_alt[n][10 - m] * partial_sum_alt[n][10 - m])
                * d) as c_uint;
        }
    }

    let mut best_dir = 0;
    let mut best_cost = cost[0];
    for n in 0..8 {
        if cost[n] > best_cost {
            best_cost = cost[n];
            best_dir = n;
        }
    }

    *variance = (best_cost - cost[best_dir ^ 4]) >> 10;
    best_dir as c_int
}

/// SSE SIMD implementation of cdef_find_dir for 16bpc.
/// Same structure as 8bpc: pixels are shifted right by bitdepth_min_8 and
/// centered around 0, giving the same partial sum ranges as 8bpc.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn cdef_find_dir_simd_16bpc(
    _t: Desktop64,
    img: PicOffset,
    variance: &mut c_uint,
    bitdepth: u8,
) -> c_int {
    use super::pixel_access::{loadu_128, storeu_128};
    use crate::include::common::bitdepth::BitDepth16;

    let shift = (bitdepth - 8) as i32;
    let shift_v = _mm_cvtsi32_si128(shift);
    let sub128 = _mm_set1_epi16(128);

    // Padded to 16 i16 for unaligned SIMD access at variable offsets.
    let mut partial_sum_diag0 = [0i16; 16]; // indices 0..14
    let mut partial_sum_diag1 = [0i16; 16]; // indices 0..14
    let mut partial_sum_alt0 = [0i16; 16]; // indices 0..10
    let mut partial_sum_alt1 = [0i16; 16]; // indices 0..10
    let mut partial_sum_alt2 = [0i16; 16]; // indices 0..10
    let mut partial_sum_alt3 = [0i16; 16]; // indices 0..10
    let mut hv0 = [0i16; 8];
    let mut hv1_vec = _mm_setzero_si128();

    let stride = img.pixel_stride::<BitDepth16>();

    for y in 0..8usize {
        let row_img = img + (y as isize * stride);
        let row_slice = row_img.slice::<BitDepth16>(8);

        // Load 8 u16 pixels, shift right by bitdepth_min_8, truncate to i16, subtract 128
        let mut row_u16 = [0u16; 8];
        row_u16.copy_from_slice(&row_slice[..8]);
        let raw = loadu_128!(&row_u16);
        let shifted = _mm_srl_epi16(raw, shift_v);
        let row = _mm_sub_epi16(shifted, sub128);

        // Extract to array for scatter operations
        let mut px = [0i16; 8];
        storeu_128!(&mut px, row);

        // hv[0][y] = sum of row
        hv0[y] = px[0] + px[1] + px[2] + px[3] + px[4] + px[5] + px[6] + px[7];

        // hv[1]: vertical accumulation (each lane = one column)
        hv1_vec = _mm_add_epi16(hv1_vec, row);

        // diag[0][y+x]: 8 consecutive values at offset y (SIMD add)
        let d0 = loadu_128!(&partial_sum_diag0[y..y + 8], [i16; 8]);
        storeu_128!(
            <&mut [i16; 8]>::try_from(&mut partial_sum_diag0[y..y + 8]).unwrap(),
            _mm_add_epi16(d0, row)
        );

        // diag[1][7+y-x]: reversed row at offset y (SIMD byte-reverse + add)
        let rev = _mm_shuffle_epi8(
            row,
            _mm_set_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14),
        );
        let d1 = loadu_128!(&partial_sum_diag1[y..y + 8], [i16; 8]);
        storeu_128!(
            <&mut [i16; 8]>::try_from(&mut partial_sum_diag1[y..y + 8]).unwrap(),
            _mm_add_epi16(d1, rev)
        );

        // alt[0][y + (x >> 1)]: pairwise sums → 4 values at offset y
        let pair0 = px[0] + px[1];
        let pair1 = px[2] + px[3];
        let pair2 = px[4] + px[5];
        let pair3 = px[6] + px[7];
        partial_sum_alt0[y] += pair0;
        partial_sum_alt0[y + 1] += pair1;
        partial_sum_alt0[y + 2] += pair2;
        partial_sum_alt0[y + 3] += pair3;

        // alt[1][3 + y - (x >> 1)]: reversed pairwise sums at offset y
        partial_sum_alt1[y] += pair3;
        partial_sum_alt1[y + 1] += pair2;
        partial_sum_alt1[y + 2] += pair1;
        partial_sum_alt1[y + 3] += pair0;

        // alt[2][3 - (y >> 1) + x]: 8 consecutive at offset (3 - y/2) (SIMD add)
        let base2 = 3 - (y >> 1);
        let a2 = loadu_128!(&partial_sum_alt2[base2..base2 + 8], [i16; 8]);
        storeu_128!(
            <&mut [i16; 8]>::try_from(&mut partial_sum_alt2[base2..base2 + 8]).unwrap(),
            _mm_add_epi16(a2, row)
        );

        // alt[3][(y >> 1) + x]: 8 consecutive at offset (y/2) (SIMD add)
        let base3 = y >> 1;
        let a3 = loadu_128!(&partial_sum_alt3[base3..base3 + 8], [i16; 8]);
        storeu_128!(
            <&mut [i16; 8]>::try_from(&mut partial_sum_alt3[base3..base3 + 8]).unwrap(),
            _mm_add_epi16(a3, row)
        );
    }

    let mut hv1 = [0i16; 8];
    storeu_128!(&mut hv1, hv1_vec);

    // === Cost computation (scalar — small fixed-size loops) ===
    let mut cost = [0u32; 8];
    for n in 0..8 {
        cost[2] += (hv0[n] as i32 * hv0[n] as i32) as u32;
        cost[6] += (hv1[n] as i32 * hv1[n] as i32) as u32;
    }
    cost[2] *= 105;
    cost[6] *= 105;

    static DIV_TABLE: [u16; 7] = [840, 420, 280, 210, 168, 140, 120];
    for n in 0..7 {
        let d = DIV_TABLE[n] as i32;
        cost[0] += ((partial_sum_diag0[n] as i32 * partial_sum_diag0[n] as i32
            + partial_sum_diag0[14 - n] as i32 * partial_sum_diag0[14 - n] as i32)
            * d) as u32;
        cost[4] += ((partial_sum_diag1[n] as i32 * partial_sum_diag1[n] as i32
            + partial_sum_diag1[14 - n] as i32 * partial_sum_diag1[14 - n] as i32)
            * d) as u32;
    }
    cost[0] += (partial_sum_diag0[7] as i32 * partial_sum_diag0[7] as i32 * 105) as u32;
    cost[4] += (partial_sum_diag1[7] as i32 * partial_sum_diag1[7] as i32 * 105) as u32;

    let alt_arrays: [&[i16; 16]; 4] = [
        &partial_sum_alt0,
        &partial_sum_alt1,
        &partial_sum_alt2,
        &partial_sum_alt3,
    ];
    for n in 0..4 {
        let cost_ptr = &mut cost[n * 2 + 1];
        for m in 0..5 {
            let v = alt_arrays[n][3 + m] as i32;
            *cost_ptr += (v * v) as u32;
        }
        *cost_ptr *= 105;
        for m in 0..3 {
            let d = DIV_TABLE[2 * m + 1] as i32;
            let a = alt_arrays[n][m] as i32;
            let b = alt_arrays[n][10 - m] as i32;
            *cost_ptr += ((a * a + b * b) * d) as u32;
        }
    }

    let mut best_dir = 0;
    let mut best_cost = cost[0];
    for n in 0..8 {
        if cost[n] > best_cost {
            best_cost = cost[n];
            best_dir = n;
        }
    }

    *variance = (best_cost - cost[best_dir ^ 4]) >> 10;
    best_dir as c_int
}

// ============================================================================
// MODULE TESTS
// ============================================================================

// Test module gated behind `asm` because calling #[target_feature] functions
// from a non-target_feature context requires unsafe, which is forbidden in default build.
#[cfg(all(test, feature = "asm"))]
#[allow(unsafe_code)]
mod tests {
    use super::*;
    #[cfg(target_arch = "x86_64")]
    use crate::src::safe_simd::pixel_access::{loadu_256, storeu_256};

    #[test]
    fn test_constrain_avx2() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        // Test constrain with known values
        let diff: [i16; 16] = [
            0, 1, -1, 10, -10, 50, -50, 100, -100, 5, -5, 15, -15, 30, -30, 127,
        ];
        let threshold = 20i16;
        let shift = 2;

        // Scalar reference
        let scalar_results: Vec<i16> = diff
            .iter()
            .map(|&d| {
                let adiff = d.abs() as i32;
                let term = threshold as i32 - (adiff >> shift);
                let max_term = term.max(0);
                let result_abs = (adiff as i32).min(max_term);
                if d >= 0 {
                    result_abs as i16
                } else {
                    -(result_abs as i16)
                }
            })
            .collect();

        // SAFETY: We checked is_x86_feature_detected!("avx2") above.
        unsafe {
            let token = crate::src::cpu::summon_avx2().expect("AVX2 required for test");
            let diff_vec = loadu_256!(&diff);
            let thresh_vec = _mm256_set1_epi16(threshold);
            let shift_vec = _mm_cvtsi32_si128(shift);

            let result = constrain_avx2(token, diff_vec, thresh_vec, shift_vec);

            let mut simd_results = [0i16; 16];
            storeu_256!(&mut simd_results, result);

            assert_eq!(simd_results.as_slice(), scalar_results.as_slice());
        }
    }
}

// ============================================================================
// CDEF FILTER FUNCTIONS (SIMD)
// ============================================================================

use crate::include::common::intops::iclip;

// TMP_STRIDE is 12 in the original cdef.rs
const TMP_STRIDE: usize = 12;

/// Scalar constrain function
#[inline(always)]
fn constrain_scalar(diff: i32, threshold: c_int, shift: c_int) -> i32 {
    let adiff = diff.abs();
    let term = threshold - (adiff >> shift);
    let max_term = cmp::max(0, term);
    let result = cmp::min(adiff, max_term);
    if diff < 0 {
        -result
    } else {
        result
    }
}

/// Scalar CDEF filter fallback for 8bpc, used on non-x86_64 or when AVX2 unavailable.
fn cdef_filter_block_scalar_8bpc(
    tmp: &[u16],
    tmp_offset: usize,
    dst: PicOffset,
    stride: isize,
    w: usize,
    h: usize,
    dir: usize,
    pri_strength: c_int,
    sec_strength: c_int,
    damping: c_int,
) {
    use crate::include::common::bitdepth::BitDepth8;

    // Single guard for entire output region
    let (mut p_guard, p_base) = dst.strided_slice_mut::<BitDepth8>(w, h);
    let tmp = tmp.flex();
    let mut p_guard = p_guard.flex_mut();

    if pri_strength != 0 {
        let pri_tap = 4 - (pri_strength & 1);
        let pri_shift = cmp::max(0, damping - pri_strength.ilog2() as c_int);

        if sec_strength != 0 {
            let sec_shift = damping - sec_strength.ilog2() as c_int;

            for y in 0..h {
                let row_base = (tmp_offset + y * TMP_STRIDE) as isize;
                let row_off = p_base.wrapping_add_signed(y as isize * stride);

                for x in 0..w {
                    let px = p_guard[row_off + x] as i32;
                    let mut sum = 0i32;
                    let mut max = px;
                    let mut min = px;
                    let base = row_base + x as isize;

                    let mut pri_tap_k = pri_tap;
                    for k in 0..2 {
                        let off1 = dav1d_cdef_directions[dir + 2][k] as isize;
                        // Sign-extend: fill 0xC000u16 → -16384i16 → -16384i32
                        let p0 = tmp[(base + off1) as usize] as i16 as i32;
                        let p1 = tmp[(base - off1) as usize] as i16 as i32;

                        sum += pri_tap_k * constrain_scalar(p0 - px, pri_strength, pri_shift);
                        sum += pri_tap_k * constrain_scalar(p1 - px, pri_strength, pri_shift);

                        pri_tap_k = pri_tap_k & 3 | 2;

                        // Unsigned min: fill (-16384 as u32 = very large) not selected
                        min = cmp::min(cmp::min(p0 as u32, p1 as u32), min as u32) as i32;
                        // Signed max: fill (-16384) not selected
                        max = cmp::max(cmp::max(p0, p1), max);

                        let off2 = dav1d_cdef_directions[dir + 4][k] as isize;
                        let off3 = dav1d_cdef_directions[dir + 0][k] as isize;
                        let s0 = tmp[(base + off2) as usize] as i16 as i32;
                        let s1 = tmp[(base - off2) as usize] as i16 as i32;
                        let s2 = tmp[(base + off3) as usize] as i16 as i32;
                        let s3 = tmp[(base - off3) as usize] as i16 as i32;

                        let sec_tap = 2 - k as i32;
                        sum += sec_tap * constrain_scalar(s0 - px, sec_strength, sec_shift);
                        sum += sec_tap * constrain_scalar(s1 - px, sec_strength, sec_shift);
                        sum += sec_tap * constrain_scalar(s2 - px, sec_strength, sec_shift);
                        sum += sec_tap * constrain_scalar(s3 - px, sec_strength, sec_shift);

                        min = cmp::min(
                            cmp::min(
                                cmp::min(s0 as u32, s1 as u32),
                                cmp::min(s2 as u32, s3 as u32),
                            ),
                            min as u32,
                        ) as i32;
                        max = cmp::max(cmp::max(cmp::max(cmp::max(s0, s1), s2), s3), max);
                    }

                    p_guard[row_off + x] =
                        iclip(px + (sum - (sum < 0) as i32 + 8 >> 4), min, max) as u8;
                }
            }
        } else {
            for y in 0..h {
                let row_base = (tmp_offset + y * TMP_STRIDE) as isize;
                let row_off = p_base.wrapping_add_signed(y as isize * stride);

                for x in 0..w {
                    let px = p_guard[row_off + x] as i32;
                    let mut sum = 0i32;
                    let base = row_base + x as isize;

                    let mut pri_tap_k = pri_tap;
                    for k in 0..2 {
                        let off = dav1d_cdef_directions[dir + 2][k] as isize;
                        let p0 = tmp[(base + off) as usize] as i32;
                        let p1 = tmp[(base - off) as usize] as i32;

                        sum += pri_tap_k * constrain_scalar(p0 - px, pri_strength, pri_shift);
                        sum += pri_tap_k * constrain_scalar(p1 - px, pri_strength, pri_shift);

                        pri_tap_k = pri_tap_k & 3 | 2;
                    }

                    p_guard[row_off + x] = (px + (sum - (sum < 0) as i32 + 8 >> 4)) as u8;
                }
            }
        }
    } else {
        let sec_shift = damping - sec_strength.ilog2() as c_int;

        for y in 0..h {
            let row_base = (tmp_offset + y * TMP_STRIDE) as isize;
            let row_off = p_base.wrapping_add_signed(y as isize * stride);

            for x in 0..w {
                let px = p_guard[row_off + x] as i32;
                let mut sum = 0i32;
                let base = row_base + x as isize;

                for k in 0..2 {
                    let off1 = dav1d_cdef_directions[dir + 4][k] as isize;
                    let off2 = dav1d_cdef_directions[dir + 0][k] as isize;
                    let s0 = tmp[(base + off1) as usize] as i32;
                    let s1 = tmp[(base - off1) as usize] as i32;
                    let s2 = tmp[(base + off2) as usize] as i32;
                    let s3 = tmp[(base - off2) as usize] as i32;

                    let sec_tap = 2 - k as i32;
                    sum += sec_tap * constrain_scalar(s0 - px, sec_strength, sec_shift);
                    sum += sec_tap * constrain_scalar(s1 - px, sec_strength, sec_shift);
                    sum += sec_tap * constrain_scalar(s2 - px, sec_strength, sec_shift);
                    sum += sec_tap * constrain_scalar(s3 - px, sec_strength, sec_shift);
                }

                p_guard[row_off + x] = (px + (sum - (sum < 0) as i32 + 8 >> 4)) as u8;
            }
        }
    }
}

/// Padding function for 8bpc - copies edge pixels into temporary buffer.
/// Merged loops: source + left + right in one pass to halve DisjointMut calls.
fn padding_8bpc(
    tmp: &mut [u16],
    dst: PicOffset,
    left: &[LeftPixelRow2px<u8>; 8],
    top: &CdefTop,
    bottom: &CdefBottom,
    w: usize,
    h: usize,
    edges: CdefEdgeFlags,
) {
    use crate::include::common::bitdepth::BitDepth8;

    let stride = dst.pixel_stride::<BitDepth8>();

    // Fill with CDEF_VERY_LARGE: 0xC000 (49152 unsigned, -16384 signed).
    // This value is chosen so that:
    //   - _mm_min_epu16 (unsigned) won't select it as min (49152 > any pixel)
    //   - _mm_max_epi16 (signed) won't select it as max (-16384 < any pixel)
    //   - constrain() returns 0 for it (|diff| > threshold for all pixels)
    //   - |fill - pixel| never equals i16::MIN for any valid pixel (avoids abs overflow)
    //     Worst case: |0xC000 - 0| = 16384, which fits in i16.
    let very_large = 0xC000u16;
    tmp.fill(very_large);
    let mut tmp = tmp.flex_mut();

    let tmp_offset = 2 * TMP_STRIDE + 2;
    let need_left = edges.contains(CdefEdgeFlags::HAVE_LEFT);
    let need_right = edges.contains(CdefEdgeFlags::HAVE_RIGHT);

    // Single pass: copy source pixels + left/right edges per row.
    // This uses one DisjointMut slice per row instead of two (source + right).
    let slice_w = w + if need_right { 2 } else { 0 };
    for y in 0..h {
        let row_offset = tmp_offset + y * TMP_STRIDE;

        // Left edge (from separate left[] array, not PicOffset)
        if need_left {
            tmp[row_offset - 2] = left[y][0] as u16;
            tmp[row_offset - 1] = left[y][1] as u16;
        }

        // Source pixels + right edge in one DisjointMut access
        let src = (dst + (y as isize * stride)).slice::<BitDepth8>(slice_w);
        for x in 0..slice_w {
            tmp[row_offset + x] = src[x] as u16;
        }
    }

    // Handle top edge (safe slice access via DisjointMut)
    if edges.contains(CdefEdgeFlags::HAVE_TOP) {
        // Adjust offset by -2 so index 0 corresponds to 2 pixels left of block
        let x_start = if edges.contains(CdefEdgeFlags::HAVE_LEFT) {
            0usize
        } else {
            2
        };
        let x_end = if edges.contains(CdefEdgeFlags::HAVE_RIGHT) {
            w + 4
        } else {
            w + 2
        };
        for dy in 0..2usize {
            let row_offset = tmp_offset - (2 - dy) * TMP_STRIDE;
            let top_row = WithOffset {
                data: top.data,
                offset: top
                    .offset
                    .wrapping_sub(2)
                    .wrapping_add_signed(dy as isize * stride),
            };
            let slice = top_row.data.slice_as::<_, u8>((top_row.offset.., ..x_end));
            for x in x_start..x_end {
                tmp[row_offset + x - 2] = slice[x] as u16;
            }
        }
    }

    // Handle bottom edge (safe slice access via DisjointMut/PicOrBuf)
    if edges.contains(CdefEdgeFlags::HAVE_BOTTOM) {
        let x_start = if edges.contains(CdefEdgeFlags::HAVE_LEFT) {
            0usize
        } else {
            2
        };
        let x_end = if edges.contains(CdefEdgeFlags::HAVE_RIGHT) {
            w + 4
        } else {
            w + 2
        };
        for dy in 0..2usize {
            let row_offset = tmp_offset + (h + dy) * TMP_STRIDE;
            let bottom_row = WithOffset {
                data: bottom.data,
                offset: bottom
                    .offset
                    .wrapping_sub(2)
                    .wrapping_add_signed(dy as isize * stride),
            };
            let slice = match bottom_row.data {
                PicOrBuf::Pic(pic) => {
                    let guard = pic.slice::<BitDepth8, _>((bottom_row.offset.., ..x_end));
                    // Copy into tmp inline since guard lifetime is limited
                    for x in x_start..x_end {
                        tmp[row_offset + x - 2] = guard[x] as u16;
                    }
                    continue;
                }
                PicOrBuf::Buf(buf) => buf.slice_as::<_, u8>((bottom_row.offset.., ..x_end)),
            };
            for x in x_start..x_end {
                tmp[row_offset + x - 2] = slice[x] as u16;
            }
        }
    }
}

/// CDEF filter using AVX2 SIMD for 8bpc 8x8 block
#[cfg(target_arch = "x86_64")]
fn cdef_filter_8x8_8bpc_avx2_inner(
    dst: PicOffset,
    left: &[LeftPixelRow2px<u8>; 8],
    top: &CdefTop,
    bottom: &CdefBottom,
    pri_strength: c_int,
    sec_strength: c_int,
    dir: c_int,
    damping: c_int,
    edges: CdefEdgeFlags,
) {
    use crate::include::common::bitdepth::BitDepth8;

    let mut tmp = [0u16; TMP_STRIDE * 12];
    padding_8bpc(&mut tmp, dst, left, top, bottom, 8, 8, edges);

    let tmp_offset = 2 * TMP_STRIDE + 2;
    let stride = dst.pixel_stride::<BitDepth8>();

    #[cfg(target_arch = "x86_64")]
    if let Some(token) = crate::src::cpu::summon_avx2() {
        cdef_filter_block_simd_8bpc(
            token,
            &tmp,
            tmp_offset,
            dst,
            stride,
            8,
            8,
            dir as usize,
            pri_strength,
            sec_strength,
            damping,
        );
        return;
    }

    cdef_filter_block_scalar_8bpc(
        &tmp,
        tmp_offset,
        dst,
        stride,
        8,
        8,
        dir as usize,
        pri_strength,
        sec_strength,
        damping,
    );
}

/// FFI wrapper for CDEF 8x8 8bpc
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn cdef_filter_8x8_8bpc_avx2(
    _dst_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    left: *const [LeftPixelRow2px<DynPixel>; 8],
    _top_ptr: *const DynPixel,
    _bottom_ptr: *const DynPixel,
    pri_strength: c_int,
    sec_strength: c_int,
    dir: c_int,
    damping: c_int,
    edges: CdefEdgeFlags,
    _bitdepth_max: c_int,
    dst: *const FFISafe<PicOffset>,
    top: *const FFISafe<CdefTop>,
    bottom: *const FFISafe<CdefBottom>,
) {
    let dst = unsafe { *FFISafe::get(dst) };
    let left = unsafe { &*(left as *const [LeftPixelRow2px<u8>; 8]) };
    let top = unsafe { FFISafe::get(top) };
    let bottom = unsafe { FFISafe::get(bottom) };

    cdef_filter_8x8_8bpc_avx2_inner(
        dst,
        left,
        top,
        bottom,
        pri_strength,
        sec_strength,
        dir,
        damping,
        edges,
    );
}

/// CDEF filter 4x8 8bpc
#[cfg(target_arch = "x86_64")]
fn cdef_filter_4x8_8bpc_avx2_inner(
    dst: PicOffset,
    left: &[LeftPixelRow2px<u8>; 8],
    top: &CdefTop,
    bottom: &CdefBottom,
    pri_strength: c_int,
    sec_strength: c_int,
    dir: c_int,
    damping: c_int,
    edges: CdefEdgeFlags,
) {
    use crate::include::common::bitdepth::BitDepth8;

    let mut tmp = [0u16; TMP_STRIDE * 12];
    padding_8bpc(&mut tmp, dst, left, top, bottom, 4, 8, edges);

    let tmp_offset = 2 * TMP_STRIDE + 2;
    let stride = dst.pixel_stride::<BitDepth8>();

    #[cfg(target_arch = "x86_64")]
    if let Some(token) = crate::src::cpu::summon_avx2() {
        cdef_filter_block_simd_8bpc(
            token,
            &tmp,
            tmp_offset,
            dst,
            stride,
            4,
            8,
            dir as usize,
            pri_strength,
            sec_strength,
            damping,
        );
        return;
    }

    cdef_filter_block_scalar_8bpc(
        &tmp,
        tmp_offset,
        dst,
        stride,
        4,
        8,
        dir as usize,
        pri_strength,
        sec_strength,
        damping,
    );
}

/// FFI wrapper for CDEF 4x8 8bpc
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn cdef_filter_4x8_8bpc_avx2(
    _dst_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    left: *const [LeftPixelRow2px<DynPixel>; 8],
    _top_ptr: *const DynPixel,
    _bottom_ptr: *const DynPixel,
    pri_strength: c_int,
    sec_strength: c_int,
    dir: c_int,
    damping: c_int,
    edges: CdefEdgeFlags,
    _bitdepth_max: c_int,
    dst: *const FFISafe<PicOffset>,
    top: *const FFISafe<CdefTop>,
    bottom: *const FFISafe<CdefBottom>,
) {
    let dst = unsafe { *FFISafe::get(dst) };
    let left = unsafe { &*(left as *const [LeftPixelRow2px<u8>; 8]) };
    let top = unsafe { FFISafe::get(top) };
    let bottom = unsafe { FFISafe::get(bottom) };

    cdef_filter_4x8_8bpc_avx2_inner(
        dst,
        left,
        top,
        bottom,
        pri_strength,
        sec_strength,
        dir,
        damping,
        edges,
    );
}

/// CDEF filter 4x4 8bpc
#[cfg(target_arch = "x86_64")]
fn cdef_filter_4x4_8bpc_avx2_inner(
    dst: PicOffset,
    left: &[LeftPixelRow2px<u8>; 8],
    top: &CdefTop,
    bottom: &CdefBottom,
    pri_strength: c_int,
    sec_strength: c_int,
    dir: c_int,
    damping: c_int,
    edges: CdefEdgeFlags,
) {
    use crate::include::common::bitdepth::BitDepth8;

    let mut tmp = [0u16; TMP_STRIDE * 8];
    padding_8bpc(&mut tmp, dst, left, top, bottom, 4, 4, edges);

    let tmp_offset = 2 * TMP_STRIDE + 2;
    let stride = dst.pixel_stride::<BitDepth8>();

    #[cfg(target_arch = "x86_64")]
    if let Some(token) = crate::src::cpu::summon_avx2() {
        cdef_filter_block_simd_8bpc(
            token,
            &tmp,
            tmp_offset,
            dst,
            stride,
            4,
            4,
            dir as usize,
            pri_strength,
            sec_strength,
            damping,
        );
        return;
    }

    cdef_filter_block_scalar_8bpc(
        &tmp,
        tmp_offset,
        dst,
        stride,
        4,
        4,
        dir as usize,
        pri_strength,
        sec_strength,
        damping,
    );
}

/// FFI wrapper for CDEF 4x4 8bpc
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn cdef_filter_4x4_8bpc_avx2(
    _dst_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    left: *const [LeftPixelRow2px<DynPixel>; 8],
    _top_ptr: *const DynPixel,
    _bottom_ptr: *const DynPixel,
    pri_strength: c_int,
    sec_strength: c_int,
    dir: c_int,
    damping: c_int,
    edges: CdefEdgeFlags,
    _bitdepth_max: c_int,
    dst: *const FFISafe<PicOffset>,
    top: *const FFISafe<CdefTop>,
    bottom: *const FFISafe<CdefBottom>,
) {
    let dst = unsafe { *FFISafe::get(dst) };
    let left = unsafe { &*(left as *const [LeftPixelRow2px<u8>; 8]) };
    let top = unsafe { FFISafe::get(top) };
    let bottom = unsafe { FFISafe::get(bottom) };

    cdef_filter_4x4_8bpc_avx2_inner(
        dst,
        left,
        top,
        bottom,
        pri_strength,
        sec_strength,
        dir,
        damping,
        edges,
    );
}

/// CDEF direction finding for 8bpc
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn cdef_find_dir_8bpc_avx2(
    _dst_ptr: *const DynPixel,
    _dst_stride: ptrdiff_t,
    variance: &mut c_uint,
    bitdepth_max: c_int,
    dst: *const FFISafe<PicOffset>,
) -> c_int {
    use crate::include::common::bitdepth::BitDepth8;

    let dst = unsafe { *FFISafe::get(dst) };
    let token = crate::src::cpu::summon_avx2().expect("AVX2 required");

    cdef_find_dir_simd_8bpc(token, dst, variance)
}

// ============================================================================
// 16BPC IMPLEMENTATIONS
// ============================================================================

/// Padding function for 16bpc
fn padding_16bpc(
    tmp: &mut [u16],
    dst: PicOffset,
    left: &[LeftPixelRow2px<u16>; 8],
    top: &CdefTop,
    bottom: &CdefBottom,
    w: usize,
    h: usize,
    edges: CdefEdgeFlags,
    bitdepth_max: c_int,
) {
    use crate::include::common::bitdepth::BitDepth16;

    let _bd = BitDepth16::new(bitdepth_max as u16);
    // Fill with CDEF_VERY_LARGE: 0xC000 (49152 unsigned, -16384 signed).
    // Same value as 8bpc — works for all bit depths up to 14-bit.
    let very_large = 0xC000u16;
    tmp.fill(very_large);
    let mut tmp = tmp.flex_mut();

    let tmp_offset = 2 * TMP_STRIDE + 2;
    let pixel_stride = dst.pixel_stride::<BitDepth16>();
    let need_left = edges.contains(CdefEdgeFlags::HAVE_LEFT);
    let need_right = edges.contains(CdefEdgeFlags::HAVE_RIGHT);

    // Single pass: copy source pixels + left/right edges per row.
    let slice_w = w + if need_right { 2 } else { 0 };
    for y in 0..h {
        let row_offset = tmp_offset + y * TMP_STRIDE;

        // Left edge (from separate left[] array)
        if need_left {
            tmp[row_offset - 2] = left[y][0];
            tmp[row_offset - 1] = left[y][1];
        }

        // Source pixels + right edge in one DisjointMut access
        let src = (dst + (y as isize * pixel_stride)).slice::<BitDepth16>(slice_w);
        for x in 0..slice_w {
            tmp[row_offset + x] = src[x];
        }
    }

    // Handle top edge (safe slice access via DisjointMut)
    if edges.contains(CdefEdgeFlags::HAVE_TOP) {
        let pixel_stride = dst.pixel_stride::<BitDepth16>();
        let x_start = if edges.contains(CdefEdgeFlags::HAVE_LEFT) {
            0usize
        } else {
            2
        };
        let x_end = if edges.contains(CdefEdgeFlags::HAVE_RIGHT) {
            w + 4
        } else {
            w + 2
        };
        for dy in 0..2usize {
            let row_offset = tmp_offset - (2 - dy) * TMP_STRIDE;
            let top_row = WithOffset {
                data: top.data,
                offset: top
                    .offset
                    .wrapping_sub(2)
                    .wrapping_add_signed(dy as isize * pixel_stride),
            };
            let slice = top_row.data.slice_as::<_, u16>((top_row.offset.., ..x_end));
            for x in x_start..x_end {
                tmp[row_offset + x - 2] = slice[x];
            }
        }
    }

    // Handle bottom edge (safe slice access via DisjointMut/PicOrBuf)
    if edges.contains(CdefEdgeFlags::HAVE_BOTTOM) {
        let pixel_stride = dst.pixel_stride::<BitDepth16>();
        let x_start = if edges.contains(CdefEdgeFlags::HAVE_LEFT) {
            0usize
        } else {
            2
        };
        let x_end = if edges.contains(CdefEdgeFlags::HAVE_RIGHT) {
            w + 4
        } else {
            w + 2
        };
        for dy in 0..2usize {
            let row_offset = tmp_offset + (h + dy) * TMP_STRIDE;
            let bottom_row = WithOffset {
                data: bottom.data,
                offset: bottom
                    .offset
                    .wrapping_sub(2)
                    .wrapping_add_signed(dy as isize * pixel_stride),
            };
            let slice = match bottom_row.data {
                PicOrBuf::Pic(pic) => {
                    let guard = pic.slice::<BitDepth16, _>((bottom_row.offset.., ..x_end));
                    for x in x_start..x_end {
                        tmp[row_offset + x - 2] = guard[x];
                    }
                    continue;
                }
                PicOrBuf::Buf(buf) => buf.slice_as::<_, u16>((bottom_row.offset.., ..x_end)),
            };
            for x in x_start..x_end {
                tmp[row_offset + x - 2] = slice[x];
            }
        }
    }
}

/// Vectorized CDEF filter for 16bpc — processes 8 pixels per row using SSE.
/// For w=4, processes 8 lanes but only stores the low 4 u16 values.
/// 16bpc pixels (0..4095) fit in i16; constrain results also fit in i16.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn cdef_filter_block_simd_16bpc(
    t: Desktop64,
    tmp: &[u16],
    tmp_offset: usize,
    dst: PicOffset,
    stride: isize,
    w: usize,
    h: usize,
    dir: usize,
    pri_strength: c_int,
    sec_strength: c_int,
    damping: c_int,
    bitdepth_max: c_int,
) {
    use super::pixel_access::{loadu_128, storeu_128};
    use crate::include::common::bitdepth::BitDepth16;

    let zero = _mm_setzero_si128();
    let bd_max = _mm_set1_epi16(bitdepth_max as i16);
    let bitdepth_min_8 = ((bitdepth_max + 1) as u32).ilog2() as c_int - 8;

    // Single guard for entire output region
    let (mut p_guard, p_base) = dst.strided_slice_mut::<BitDepth16>(w, h);

    if pri_strength != 0 {
        let pri_tap = 4 - (pri_strength >> bitdepth_min_8 & 1);
        let pri_shift = cmp::max(0, damping - pri_strength.ilog2() as c_int);
        let pri_thresh = _mm_set1_epi16(pri_strength as i16);
        let pri_shift_v = _mm_cvtsi32_si128(pri_shift);

        if sec_strength != 0 {
            let sec_shift = damping - sec_strength.ilog2() as c_int;
            let sec_thresh = _mm_set1_epi16(sec_strength as i16);
            let sec_shift_v = _mm_cvtsi32_si128(sec_shift);

            for y in 0..h {
                let base = tmp_offset + y * TMP_STRIDE;
                let px = loadu_128!(&tmp[base..base + 8], [u16; 8]);
                let mut sum = zero;
                let mut min_v = px;
                let mut max_v = px;

                let mut pri_tap_k = pri_tap;
                for k in 0..2 {
                    let off = dav1d_cdef_directions[dir + 2][k] as isize;
                    let p0_i = (base as isize + off) as usize;
                    let p1_i = (base as isize - off) as usize;
                    let p0 = loadu_128!(&tmp[p0_i..p0_i + 8], [u16; 8]);
                    let p1 = loadu_128!(&tmp[p1_i..p1_i + 8], [u16; 8]);

                    let c0 = constrain_128(t, _mm_sub_epi16(p0, px), pri_thresh, pri_shift_v);
                    let c1 = constrain_128(t, _mm_sub_epi16(p1, px), pri_thresh, pri_shift_v);

                    let tap_v = _mm_set1_epi16(pri_tap_k as i16);
                    sum = _mm_add_epi16(sum, _mm_mullo_epi16(tap_v, _mm_add_epi16(c0, c1)));
                    pri_tap_k = pri_tap_k & 3 | 2;

                    // Use unsigned min to ignore boundary fill (0x8001 = large unsigned)
                    min_v = _mm_min_epu16(min_v, _mm_min_epu16(p0, p1));
                    // Use signed max to ignore boundary fill (0x8001 = -32767 signed)
                    max_v = _mm_max_epi16(max_v, _mm_max_epi16(p0, p1));

                    let off2 = dav1d_cdef_directions[dir + 4][k] as isize;
                    let off3 = dav1d_cdef_directions[dir + 0][k] as isize;
                    let s0_i = (base as isize + off2) as usize;
                    let s1_i = (base as isize - off2) as usize;
                    let s2_i = (base as isize + off3) as usize;
                    let s3_i = (base as isize - off3) as usize;
                    let s0 = loadu_128!(&tmp[s0_i..s0_i + 8], [u16; 8]);
                    let s1 = loadu_128!(&tmp[s1_i..s1_i + 8], [u16; 8]);
                    let s2 = loadu_128!(&tmp[s2_i..s2_i + 8], [u16; 8]);
                    let s3 = loadu_128!(&tmp[s3_i..s3_i + 8], [u16; 8]);

                    let sec_tap_k = (2 - k as i32) as i16;
                    let sec_tap_v = _mm_set1_epi16(sec_tap_k);
                    let ds0 = constrain_128(t, _mm_sub_epi16(s0, px), sec_thresh, sec_shift_v);
                    let ds1 = constrain_128(t, _mm_sub_epi16(s1, px), sec_thresh, sec_shift_v);
                    let ds2 = constrain_128(t, _mm_sub_epi16(s2, px), sec_thresh, sec_shift_v);
                    let ds3 = constrain_128(t, _mm_sub_epi16(s3, px), sec_thresh, sec_shift_v);

                    let sec_sum = _mm_add_epi16(_mm_add_epi16(ds0, ds1), _mm_add_epi16(ds2, ds3));
                    sum = _mm_add_epi16(sum, _mm_mullo_epi16(sec_tap_v, sec_sum));

                    min_v = _mm_min_epu16(
                        min_v,
                        _mm_min_epu16(_mm_min_epu16(s0, s1), _mm_min_epu16(s2, s3)),
                    );
                    max_v = _mm_max_epi16(
                        max_v,
                        _mm_max_epi16(_mm_max_epi16(s0, s1), _mm_max_epi16(s2, s3)),
                    );
                }

                // Rounding: (sum - (sum < 0) + 8) >> 4
                let neg_mask = _mm_cmpgt_epi16(zero, sum);
                let adjusted = _mm_add_epi16(sum, neg_mask);
                let adjusted = _mm_add_epi16(adjusted, _mm_set1_epi16(8));
                let adjusted = _mm_srai_epi16::<4>(adjusted);
                let result = _mm_add_epi16(px, adjusted);
                let result = _mm_max_epi16(result, min_v);
                let result = _mm_min_epi16(result, max_v);

                let mut out = [0u16; 8];
                storeu_128!(&mut out, result);
                let row_off = p_base.wrapping_add_signed(y as isize * stride);
                p_guard[row_off..row_off + w].copy_from_slice(&out[..w]);
            }
        } else {
            // Primary only
            for y in 0..h {
                let base = tmp_offset + y * TMP_STRIDE;
                let px = loadu_128!(&tmp[base..base + 8], [u16; 8]);
                let mut sum = zero;

                let mut pri_tap_k = pri_tap;
                for k in 0..2 {
                    let off = dav1d_cdef_directions[dir + 2][k] as isize;
                    let p0_i = (base as isize + off) as usize;
                    let p1_i = (base as isize - off) as usize;
                    let p0 = loadu_128!(&tmp[p0_i..p0_i + 8], [u16; 8]);
                    let p1 = loadu_128!(&tmp[p1_i..p1_i + 8], [u16; 8]);

                    let c0 = constrain_128(t, _mm_sub_epi16(p0, px), pri_thresh, pri_shift_v);
                    let c1 = constrain_128(t, _mm_sub_epi16(p1, px), pri_thresh, pri_shift_v);

                    let tap_v = _mm_set1_epi16(pri_tap_k as i16);
                    sum = _mm_add_epi16(sum, _mm_mullo_epi16(tap_v, _mm_add_epi16(c0, c1)));
                    pri_tap_k = pri_tap_k & 3 | 2;
                }

                let neg_mask = _mm_cmpgt_epi16(zero, sum);
                let adjusted = _mm_add_epi16(sum, neg_mask);
                let adjusted = _mm_add_epi16(adjusted, _mm_set1_epi16(8));
                let adjusted = _mm_srai_epi16::<4>(adjusted);
                let result = _mm_add_epi16(px, adjusted);
                // Clamp to [0, bitdepth_max]
                let result = _mm_max_epi16(result, zero);
                let result = _mm_min_epi16(result, bd_max);

                let mut out = [0u16; 8];
                storeu_128!(&mut out, result);
                let row_off = p_base.wrapping_add_signed(y as isize * stride);
                p_guard[row_off..row_off + w].copy_from_slice(&out[..w]);
            }
        }
    } else {
        // Secondary only
        let sec_shift = damping - sec_strength.ilog2() as c_int;
        let sec_thresh = _mm_set1_epi16(sec_strength as i16);
        let sec_shift_v = _mm_cvtsi32_si128(sec_shift);

        for y in 0..h {
            let base = tmp_offset + y * TMP_STRIDE;
            let px = loadu_128!(&tmp[base..base + 8], [u16; 8]);
            let mut sum = zero;

            for k in 0..2 {
                let off2 = dav1d_cdef_directions[dir + 4][k] as isize;
                let off3 = dav1d_cdef_directions[dir + 0][k] as isize;
                let s0_i = (base as isize + off2) as usize;
                let s1_i = (base as isize - off2) as usize;
                let s2_i = (base as isize + off3) as usize;
                let s3_i = (base as isize - off3) as usize;
                let s0 = loadu_128!(&tmp[s0_i..s0_i + 8], [u16; 8]);
                let s1 = loadu_128!(&tmp[s1_i..s1_i + 8], [u16; 8]);
                let s2 = loadu_128!(&tmp[s2_i..s2_i + 8], [u16; 8]);
                let s3 = loadu_128!(&tmp[s3_i..s3_i + 8], [u16; 8]);

                let sec_tap_k = (2 - k as i32) as i16;
                let sec_tap_v = _mm_set1_epi16(sec_tap_k);
                let ds0 = constrain_128(t, _mm_sub_epi16(s0, px), sec_thresh, sec_shift_v);
                let ds1 = constrain_128(t, _mm_sub_epi16(s1, px), sec_thresh, sec_shift_v);
                let ds2 = constrain_128(t, _mm_sub_epi16(s2, px), sec_thresh, sec_shift_v);
                let ds3 = constrain_128(t, _mm_sub_epi16(s3, px), sec_thresh, sec_shift_v);

                let sec_sum = _mm_add_epi16(_mm_add_epi16(ds0, ds1), _mm_add_epi16(ds2, ds3));
                sum = _mm_add_epi16(sum, _mm_mullo_epi16(sec_tap_v, sec_sum));
            }

            let neg_mask = _mm_cmpgt_epi16(zero, sum);
            let adjusted = _mm_add_epi16(sum, neg_mask);
            let adjusted = _mm_add_epi16(adjusted, _mm_set1_epi16(8));
            let adjusted = _mm_srai_epi16::<4>(adjusted);
            let result = _mm_add_epi16(px, adjusted);
            // Clamp to [0, bitdepth_max]
            let result = _mm_max_epi16(result, zero);
            let result = _mm_min_epi16(result, bd_max);

            let mut out = [0u16; 8];
            storeu_128!(&mut out, result);
            let row_off = p_base.wrapping_add_signed(y as isize * stride);
            p_guard[row_off..row_off + w].copy_from_slice(&out[..w]);
        }
    }
}

/// Scalar CDEF filter fallback for 16bpc.
fn cdef_filter_block_scalar_16bpc(
    tmp: &[u16],
    tmp_offset: usize,
    dst: PicOffset,
    stride: isize,
    w: usize,
    h: usize,
    dir: usize,
    pri_strength: c_int,
    sec_strength: c_int,
    damping: c_int,
    bitdepth_max: c_int,
) {
    use crate::include::common::bitdepth::BitDepth16;

    let bitdepth_min_8 = ((bitdepth_max + 1) as u32).ilog2() as c_int - 8;

    // Single guard for entire output region
    let (mut p_guard, p_base) = dst.strided_slice_mut::<BitDepth16>(w, h);
    let tmp = tmp.flex();
    let mut p_guard = p_guard.flex_mut();

    if pri_strength != 0 {
        let pri_tap = 4 - (pri_strength >> bitdepth_min_8 & 1);
        let pri_shift = cmp::max(0, damping - pri_strength.ilog2() as c_int);

        if sec_strength != 0 {
            let sec_shift = damping - sec_strength.ilog2() as c_int;

            for y in 0..h {
                let row_base = (tmp_offset + y * TMP_STRIDE) as isize;
                let row_off = p_base.wrapping_add_signed(y as isize * stride);

                for x in 0..w {
                    let px = p_guard[row_off + x] as i32;
                    let mut sum = 0i32;
                    let mut max = px;
                    let mut min = px;
                    let base = row_base + x as isize;

                    let mut pri_tap_k = pri_tap;
                    for k in 0..2 {
                        let off1 = dav1d_cdef_directions[dir + 2][k] as isize;
                        // Sign-extend: fill 0xC000u16 → -16384i16 → -16384i32
                        let p0 = tmp[(base + off1) as usize] as i16 as i32;
                        let p1 = tmp[(base - off1) as usize] as i16 as i32;

                        sum += pri_tap_k * constrain_scalar(p0 - px, pri_strength, pri_shift);
                        sum += pri_tap_k * constrain_scalar(p1 - px, pri_strength, pri_shift);

                        pri_tap_k = pri_tap_k & 3 | 2;

                        // Unsigned min: fill (-16384 as u32 = very large) not selected
                        min = cmp::min(cmp::min(p0 as u32, p1 as u32), min as u32) as i32;
                        // Signed max: fill (-16384) not selected
                        max = cmp::max(cmp::max(p0, p1), max);

                        let off2 = dav1d_cdef_directions[dir + 4][k] as isize;
                        let off3 = dav1d_cdef_directions[dir + 0][k] as isize;
                        let s0 = tmp[(base + off2) as usize] as i16 as i32;
                        let s1 = tmp[(base - off2) as usize] as i16 as i32;
                        let s2 = tmp[(base + off3) as usize] as i16 as i32;
                        let s3 = tmp[(base - off3) as usize] as i16 as i32;

                        let sec_tap = 2 - k as i32;
                        sum += sec_tap * constrain_scalar(s0 - px, sec_strength, sec_shift);
                        sum += sec_tap * constrain_scalar(s1 - px, sec_strength, sec_shift);
                        sum += sec_tap * constrain_scalar(s2 - px, sec_strength, sec_shift);
                        sum += sec_tap * constrain_scalar(s3 - px, sec_strength, sec_shift);

                        min = cmp::min(
                            cmp::min(
                                cmp::min(s0 as u32, s1 as u32),
                                cmp::min(s2 as u32, s3 as u32),
                            ),
                            min as u32,
                        ) as i32;
                        max = cmp::max(cmp::max(cmp::max(cmp::max(s0, s1), s2), s3), max);
                    }

                    p_guard[row_off + x] =
                        iclip(px + (sum - (sum < 0) as i32 + 8 >> 4), min, max) as u16;
                }
            }
        } else {
            for y in 0..h {
                let row_base = (tmp_offset + y * TMP_STRIDE) as isize;
                let row_off = p_base.wrapping_add_signed(y as isize * stride);

                for x in 0..w {
                    let px = p_guard[row_off + x] as i32;
                    let mut sum = 0i32;
                    let base = row_base + x as isize;

                    let mut pri_tap_k = pri_tap;
                    for k in 0..2 {
                        let off = dav1d_cdef_directions[dir + 2][k] as isize;
                        let p0 = tmp[(base + off) as usize] as i32;
                        let p1 = tmp[(base - off) as usize] as i32;

                        sum += pri_tap_k * constrain_scalar(p0 - px, pri_strength, pri_shift);
                        sum += pri_tap_k * constrain_scalar(p1 - px, pri_strength, pri_shift);

                        pri_tap_k = pri_tap_k & 3 | 2;
                    }

                    let result = px + (sum - (sum < 0) as i32 + 8 >> 4);
                    p_guard[row_off + x] = iclip(result, 0, bitdepth_max) as u16;
                }
            }
        }
    } else {
        let sec_shift = damping - sec_strength.ilog2() as c_int;

        for y in 0..h {
            let row_base = (tmp_offset + y * TMP_STRIDE) as isize;
            let row_off = p_base.wrapping_add_signed(y as isize * stride);

            for x in 0..w {
                let px = p_guard[row_off + x] as i32;
                let mut sum = 0i32;
                let base = row_base + x as isize;

                for k in 0..2 {
                    let off1 = dav1d_cdef_directions[dir + 4][k] as isize;
                    let off2 = dav1d_cdef_directions[dir + 0][k] as isize;
                    let s0 = tmp[(base + off1) as usize] as i32;
                    let s1 = tmp[(base - off1) as usize] as i32;
                    let s2 = tmp[(base + off2) as usize] as i32;
                    let s3 = tmp[(base - off2) as usize] as i32;

                    let sec_tap = 2 - k as i32;
                    sum += sec_tap * constrain_scalar(s0 - px, sec_strength, sec_shift);
                    sum += sec_tap * constrain_scalar(s1 - px, sec_strength, sec_shift);
                    sum += sec_tap * constrain_scalar(s2 - px, sec_strength, sec_shift);
                    sum += sec_tap * constrain_scalar(s3 - px, sec_strength, sec_shift);
                }

                let result = px + (sum - (sum < 0) as i32 + 8 >> 4);
                p_guard[row_off + x] = iclip(result, 0, bitdepth_max) as u16;
            }
        }
    }
}

/// CDEF filter for 16bpc 8x8 block
#[cfg(target_arch = "x86_64")]
fn cdef_filter_8x8_16bpc_avx2_inner(
    dst: PicOffset,
    left: &[LeftPixelRow2px<u16>; 8],
    top: &CdefTop,
    bottom: &CdefBottom,
    pri_strength: c_int,
    sec_strength: c_int,
    dir: c_int,
    damping: c_int,
    edges: CdefEdgeFlags,
    bitdepth_max: c_int,
) {
    use crate::include::common::bitdepth::BitDepth16;

    let mut tmp = [0u16; TMP_STRIDE * 12];
    padding_16bpc(&mut tmp, dst, left, top, bottom, 8, 8, edges, bitdepth_max);

    let tmp_offset = 2 * TMP_STRIDE + 2;
    let stride = dst.pixel_stride::<BitDepth16>();

    #[cfg(target_arch = "x86_64")]
    if let Some(token) = crate::src::cpu::summon_avx2() {
        cdef_filter_block_simd_16bpc(
            token,
            &tmp,
            tmp_offset,
            dst,
            stride,
            8,
            8,
            dir as usize,
            pri_strength,
            sec_strength,
            damping,
            bitdepth_max,
        );
        return;
    }

    cdef_filter_block_scalar_16bpc(
        &tmp,
        tmp_offset,
        dst,
        stride,
        8,
        8,
        dir as usize,
        pri_strength,
        sec_strength,
        damping,
        bitdepth_max,
    );
}

/// CDEF filter for 16bpc 4x8 block
#[cfg(target_arch = "x86_64")]
fn cdef_filter_4x8_16bpc_avx2_inner(
    dst: PicOffset,
    left: &[LeftPixelRow2px<u16>; 8],
    top: &CdefTop,
    bottom: &CdefBottom,
    pri_strength: c_int,
    sec_strength: c_int,
    dir: c_int,
    damping: c_int,
    edges: CdefEdgeFlags,
    bitdepth_max: c_int,
) {
    use crate::include::common::bitdepth::BitDepth16;

    let mut tmp = [0u16; TMP_STRIDE * 12];
    padding_16bpc(&mut tmp, dst, left, top, bottom, 4, 8, edges, bitdepth_max);

    let tmp_offset = 2 * TMP_STRIDE + 2;
    let stride = dst.pixel_stride::<BitDepth16>();

    #[cfg(target_arch = "x86_64")]
    if let Some(token) = crate::src::cpu::summon_avx2() {
        cdef_filter_block_simd_16bpc(
            token,
            &tmp,
            tmp_offset,
            dst,
            stride,
            4,
            8,
            dir as usize,
            pri_strength,
            sec_strength,
            damping,
            bitdepth_max,
        );
        return;
    }

    cdef_filter_block_scalar_16bpc(
        &tmp,
        tmp_offset,
        dst,
        stride,
        4,
        8,
        dir as usize,
        pri_strength,
        sec_strength,
        damping,
        bitdepth_max,
    );
}

/// CDEF filter for 16bpc 4x4 block
#[cfg(target_arch = "x86_64")]
fn cdef_filter_4x4_16bpc_avx2_inner(
    dst: PicOffset,
    left: &[LeftPixelRow2px<u16>; 8],
    top: &CdefTop,
    bottom: &CdefBottom,
    pri_strength: c_int,
    sec_strength: c_int,
    dir: c_int,
    damping: c_int,
    edges: CdefEdgeFlags,
    bitdepth_max: c_int,
) {
    use crate::include::common::bitdepth::BitDepth16;

    let mut tmp = [0u16; TMP_STRIDE * 8];
    padding_16bpc(&mut tmp, dst, left, top, bottom, 4, 4, edges, bitdepth_max);

    let tmp_offset = 2 * TMP_STRIDE + 2;
    let stride = dst.pixel_stride::<BitDepth16>();

    #[cfg(target_arch = "x86_64")]
    if let Some(token) = crate::src::cpu::summon_avx2() {
        cdef_filter_block_simd_16bpc(
            token,
            &tmp,
            tmp_offset,
            dst,
            stride,
            4,
            4,
            dir as usize,
            pri_strength,
            sec_strength,
            damping,
            bitdepth_max,
        );
        return;
    }

    cdef_filter_block_scalar_16bpc(
        &tmp,
        tmp_offset,
        dst,
        stride,
        4,
        4,
        dir as usize,
        pri_strength,
        sec_strength,
        damping,
        bitdepth_max,
    );
}

/// FFI wrapper for CDEF filter 8x8 16bpc
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn cdef_filter_8x8_16bpc_avx2(
    _dst_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    left: *const [LeftPixelRow2px<DynPixel>; 8],
    _top_ptr: *const DynPixel,
    _bottom_ptr: *const DynPixel,
    pri_strength: c_int,
    sec_strength: c_int,
    dir: c_int,
    damping: c_int,
    edges: CdefEdgeFlags,
    bitdepth_max: c_int,
    dst: *const FFISafe<PicOffset>,
    top: *const FFISafe<CdefTop>,
    bottom: *const FFISafe<CdefBottom>,
) {
    let dst = unsafe { *FFISafe::get(dst) };
    let left = unsafe { &*(left as *const [LeftPixelRow2px<u16>; 8]) };
    let top = unsafe { FFISafe::get(top) };
    let bottom = unsafe { FFISafe::get(bottom) };

    cdef_filter_8x8_16bpc_avx2_inner(
        dst,
        left,
        top,
        bottom,
        pri_strength,
        sec_strength,
        dir,
        damping,
        edges,
        bitdepth_max,
    );
}

/// FFI wrapper for CDEF filter 4x8 16bpc
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn cdef_filter_4x8_16bpc_avx2(
    _dst_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    left: *const [LeftPixelRow2px<DynPixel>; 8],
    _top_ptr: *const DynPixel,
    _bottom_ptr: *const DynPixel,
    pri_strength: c_int,
    sec_strength: c_int,
    dir: c_int,
    damping: c_int,
    edges: CdefEdgeFlags,
    bitdepth_max: c_int,
    dst: *const FFISafe<PicOffset>,
    top: *const FFISafe<CdefTop>,
    bottom: *const FFISafe<CdefBottom>,
) {
    let dst = unsafe { *FFISafe::get(dst) };
    let left = unsafe { &*(left as *const [LeftPixelRow2px<u16>; 8]) };
    let top = unsafe { FFISafe::get(top) };
    let bottom = unsafe { FFISafe::get(bottom) };

    cdef_filter_4x8_16bpc_avx2_inner(
        dst,
        left,
        top,
        bottom,
        pri_strength,
        sec_strength,
        dir,
        damping,
        edges,
        bitdepth_max,
    );
}

/// FFI wrapper for CDEF filter 4x4 16bpc
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn cdef_filter_4x4_16bpc_avx2(
    _dst_ptr: *mut DynPixel,
    _stride: ptrdiff_t,
    left: *const [LeftPixelRow2px<DynPixel>; 8],
    _top_ptr: *const DynPixel,
    _bottom_ptr: *const DynPixel,
    pri_strength: c_int,
    sec_strength: c_int,
    dir: c_int,
    damping: c_int,
    edges: CdefEdgeFlags,
    bitdepth_max: c_int,
    dst: *const FFISafe<PicOffset>,
    top: *const FFISafe<CdefTop>,
    bottom: *const FFISafe<CdefBottom>,
) {
    let dst = unsafe { *FFISafe::get(dst) };
    let left = unsafe { &*(left as *const [LeftPixelRow2px<u16>; 8]) };
    let top = unsafe { FFISafe::get(top) };
    let bottom = unsafe { FFISafe::get(bottom) };

    cdef_filter_4x4_16bpc_avx2_inner(
        dst,
        left,
        top,
        bottom,
        pri_strength,
        sec_strength,
        dir,
        damping,
        edges,
        bitdepth_max,
    );
}

/// FFI wrapper for cdef_find_dir 16bpc
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn cdef_find_dir_16bpc_avx2(
    _dst_ptr: *const DynPixel,
    _dst_stride: ptrdiff_t,
    variance: &mut c_uint,
    bitdepth_max: c_int,
    dst: *const FFISafe<PicOffset>,
) -> c_int {
    use crate::include::common::bitdepth::BitDepth16;

    let dst = unsafe { *FFISafe::get(dst) };
    let bd = BitDepth16::new(bitdepth_max as u16);

    let token = unsafe { Desktop64::forge_token_dangerously() };
    cdef_find_dir_simd_16bpc(token, dst, variance, bd.bitdepth() as u8)
}

// ============================================================================
// SAFE DISPATCH WRAPPERS
// ============================================================================
// These functions wrap the unsafe extern "C" SIMD functions behind a safe API.
// They take safe Rust types, handle pointer conversion internally, and return
// a bool/Option indicating whether SIMD was used.

/// Safe dispatch for cdef_filter. Returns true if SIMD was used.
#[cfg(target_arch = "x86_64")]
pub fn cdef_filter_dispatch<BD: BitDepth>(
    variant: usize,
    dst: PicOffset,
    left: &[LeftPixelRow2px<BD::Pixel>; 8],
    top: CdefTop,
    bottom: CdefBottom,
    pri_strength: c_int,
    sec_strength: c_int,
    dir: c_int,
    damping: c_int,
    edges: CdefEdgeFlags,
    bd: BD,
) -> bool {
    use crate::include::common::bitdepth::BPC;

    let Some(_token) = crate::src::cpu::summon_avx2() else {
        return false;
    };

    // Left pointer cast is safe because LeftPixelRow2px<BD::Pixel> has same layout for u8/u16.
    match (BD::BPC, variant) {
        (BPC::BPC8, 0) => {
            let left: &[LeftPixelRow2px<u8>; 8] =
                crate::src::safe_simd::pixel_access::reinterpret_ref(left)
                    .expect("BD::Pixel layout matches u8");
            cdef_filter_8x8_8bpc_avx2_inner(
                dst,
                left,
                &top,
                &bottom,
                pri_strength,
                sec_strength,
                dir,
                damping,
                edges,
            );
        }
        (BPC::BPC8, 1) => {
            let left: &[LeftPixelRow2px<u8>; 8] =
                crate::src::safe_simd::pixel_access::reinterpret_ref(left)
                    .expect("BD::Pixel layout matches u8");
            cdef_filter_4x8_8bpc_avx2_inner(
                dst,
                left,
                &top,
                &bottom,
                pri_strength,
                sec_strength,
                dir,
                damping,
                edges,
            );
        }
        (BPC::BPC8, _) => {
            let left: &[LeftPixelRow2px<u8>; 8] =
                crate::src::safe_simd::pixel_access::reinterpret_ref(left)
                    .expect("BD::Pixel layout matches u8");
            cdef_filter_4x4_8bpc_avx2_inner(
                dst,
                left,
                &top,
                &bottom,
                pri_strength,
                sec_strength,
                dir,
                damping,
                edges,
            );
        }
        (BPC::BPC16, 0) => {
            let left: &[LeftPixelRow2px<u16>; 8] =
                crate::src::safe_simd::pixel_access::reinterpret_ref(left)
                    .expect("BD::Pixel layout matches u16");
            cdef_filter_8x8_16bpc_avx2_inner(
                dst,
                left,
                &top,
                &bottom,
                pri_strength,
                sec_strength,
                dir,
                damping,
                edges,
                bd.into_c(),
            );
        }
        (BPC::BPC16, 1) => {
            let left: &[LeftPixelRow2px<u16>; 8] =
                crate::src::safe_simd::pixel_access::reinterpret_ref(left)
                    .expect("BD::Pixel layout matches u16");
            cdef_filter_4x8_16bpc_avx2_inner(
                dst,
                left,
                &top,
                &bottom,
                pri_strength,
                sec_strength,
                dir,
                damping,
                edges,
                bd.into_c(),
            );
        }
        (BPC::BPC16, _) => {
            let left: &[LeftPixelRow2px<u16>; 8] =
                crate::src::safe_simd::pixel_access::reinterpret_ref(left)
                    .expect("BD::Pixel layout matches u16");
            cdef_filter_4x4_16bpc_avx2_inner(
                dst,
                left,
                &top,
                &bottom,
                pri_strength,
                sec_strength,
                dir,
                damping,
                edges,
                bd.into_c(),
            );
        }
    }
    true
}

/// Safe dispatch for cdef_find_dir. Returns Some(dir) if SIMD was used.
#[cfg(target_arch = "x86_64")]
pub fn cdef_dir_dispatch<BD: BitDepth>(
    dst: PicOffset,
    variance: &mut c_uint,
    bd: BD,
) -> Option<c_int> {
    use crate::include::common::bitdepth::BPC;

    let Some(token) = crate::src::cpu::summon_avx2() else {
        return None;
    };

    match BD::BPC {
        BPC::BPC8 => Some(cdef_find_dir_simd_8bpc(token, dst, variance)),
        BPC::BPC16 => Some(cdef_find_dir_simd_16bpc(token, dst, variance, bd.bitdepth() as u8)),
    }
}
