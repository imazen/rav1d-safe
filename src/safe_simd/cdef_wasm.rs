//! Safe wasm128 SIMD implementations for CDEF (Constrained Directional Enhancement Filter)
//!
//! Port of the SSE 128-bit CDEF filter from cdef.rs to WebAssembly SIMD128.
//! Key mapping: SSE __m128i → wasm32 v128, all 128-bit operations.

#![deny(unsafe_code)]
#![allow(dead_code)]

use archmage::{Wasm128Token, arcane, rite};
use core::arch::wasm32::*;

use std::cmp;
use std::ffi::c_int;

use crate::include::common::bitdepth::BitDepth;
use crate::include::common::bitdepth::LeftPixelRow2px;
use crate::include::dav1d::picture::PicOffset;
use crate::src::cdef::CdefBottom;
use crate::src::cdef::CdefEdgeFlags;
use crate::src::cdef::CdefTop;
use crate::src::strided::Strided as _;
use crate::src::tables::dav1d_cdef_directions;

// Re-use constants and shared functions from the main cdef module
use super::cdef::{TMP_STRIDE, padding_8bpc, padding_16bpc};

// ============================================================================
// CONSTRAIN FUNCTION (wasm128 SIMD)
// ============================================================================

/// wasm128 version of constrain — processes 8 i16 values at once.
/// Formula: sign(diff) * min(|diff|, max(0, threshold - (|diff| >> shift)))
///
/// Key mapping from SSE:
///   _mm_blendv_epi8(a, b, mask) → v128_bitselect(b, a, mask) [args flipped!]
///   _mm_sra_epi16(a, count) → i16x8_shr(a, extract_lane(count))
#[rite]
fn constrain_wasm128(_t: Wasm128Token, diff: v128, threshold: v128, shift: u32) -> v128 {
    let zero = i16x8_splat(0);

    // Compute absolute value
    let adiff = i16x8_abs(diff);

    // Compute threshold - (adiff >> shift)
    let shifted = i16x8_shr(adiff, shift);
    let term = i16x8_sub(threshold, shifted);

    // max(0, term)
    let max_term = i16x8_max(term, zero);

    // min(adiff, max_term)
    let result_abs = i16x8_min(adiff, max_term);

    // Apply sign of original diff
    // If diff >= 0: result = result_abs
    // If diff < 0: result = -result_abs
    let sign_mask = i16x8_gt(zero, diff);
    let neg_result = i16x8_sub(zero, result_abs);
    // v128_bitselect(a, b, mask): selects a where mask=1, b where mask=0
    // SSE _mm_blendv_epi8(a, b, mask) selects b where mask=1, a where mask=0
    // So: _mm_blendv_epi8(result_abs, neg_result, sign_mask)
    //   → v128_bitselect(neg_result, result_abs, sign_mask)
    v128_bitselect(neg_result, result_abs, sign_mask)
}

// ============================================================================
// CDEF FILTER 8BPC (wasm128 SIMD)
// ============================================================================

/// Vectorized CDEF filter for 8bpc using wasm128 — processes 8 pixels per row.
/// Handles all block sizes (8x8, 4x8, 4x4) via w/h parameters.
#[arcane]
fn cdef_filter_block_simd_8bpc(
    t: Wasm128Token,
    tmp: &[u16],
    tmp_offset: usize,
    dst: PicOffset,
    w: usize,
    h: usize,
    dir: usize,
    pri_strength: c_int,
    sec_strength: c_int,
    damping: c_int,
) {
    use super::pixel_access::{wasm_load_128, wasm_store_128, wasm_storei32};
    use crate::include::common::bitdepth::BitDepth8;

    let zero = i16x8_splat(0);

    // Tile-threading-safe block view: contiguous when threading is off, a
    // compact per-row copy when it is on. See `WithOffset::block_mut`.
    // Taps are read from the padded `tmp` copy, so only the w×h output region
    // is touched here.
    let mut block = dst.block_mut::<BitDepth8>(w, h);
    // The block has its own stride (compact when tile threading is on), so it
    // supersedes the caller-supplied picture stride. 8bpc: 1 byte per pixel,
    // so the byte stride is the pixel stride.
    let stride = block.byte_stride();
    let p_base = block.base();
    let p_guard = block.as_mut_bytes();

    if pri_strength != 0 {
        let pri_tap = 4 - (pri_strength & 1);
        let pri_shift = cmp::max(0, damping - pri_strength.ilog2() as c_int) as u32;
        let pri_thresh = i16x8_splat(pri_strength as i16);

        if sec_strength != 0 {
            // Both primary and secondary — full filter with min/max clamping
            let sec_shift = (damping - sec_strength.ilog2() as c_int) as u32;
            let sec_thresh = i16x8_splat(sec_strength as i16);

            for y in 0..h {
                let base = tmp_offset + y * TMP_STRIDE;
                let px = wasm_load_128!(&tmp[base..base + 8], [u16; 8]);
                let mut sum = zero;
                let mut min_v = px;
                let mut max_v = px;

                let mut pri_tap_k = pri_tap;
                for k in 0..2 {
                    let off = dav1d_cdef_directions[dir + 2][k] as isize;
                    let p0_i = (base as isize + off) as usize;
                    let p1_i = (base as isize - off) as usize;
                    let p0 = wasm_load_128!(&tmp[p0_i..p0_i + 8], [u16; 8]);
                    let p1 = wasm_load_128!(&tmp[p1_i..p1_i + 8], [u16; 8]);

                    let c0 = constrain_wasm128(t, i16x8_sub(p0, px), pri_thresh, pri_shift);
                    let c1 = constrain_wasm128(t, i16x8_sub(p1, px), pri_thresh, pri_shift);

                    let tap_v = i16x8_splat(pri_tap_k as i16);
                    sum = i16x8_add(sum, i16x8_mul(tap_v, i16x8_add(c0, c1)));
                    pri_tap_k = pri_tap_k & 3 | 2;

                    // Use unsigned min to ignore boundary fill (0x8001 = large unsigned)
                    min_v = u16x8_min(min_v, u16x8_min(p0, p1));
                    // Use signed max to ignore boundary fill (0x8001 = -32767 signed)
                    max_v = i16x8_max(max_v, i16x8_max(p0, p1));

                    let off2 = dav1d_cdef_directions[dir + 4][k] as isize;
                    let off3 = dav1d_cdef_directions[dir + 0][k] as isize;
                    let s0_i = (base as isize + off2) as usize;
                    let s1_i = (base as isize - off2) as usize;
                    let s2_i = (base as isize + off3) as usize;
                    let s3_i = (base as isize - off3) as usize;
                    let s0 = wasm_load_128!(&tmp[s0_i..s0_i + 8], [u16; 8]);
                    let s1 = wasm_load_128!(&tmp[s1_i..s1_i + 8], [u16; 8]);
                    let s2 = wasm_load_128!(&tmp[s2_i..s2_i + 8], [u16; 8]);
                    let s3 = wasm_load_128!(&tmp[s3_i..s3_i + 8], [u16; 8]);

                    let sec_tap_k = (2 - k as i32) as i16;
                    let sec_tap_v = i16x8_splat(sec_tap_k);
                    let ds0 = constrain_wasm128(t, i16x8_sub(s0, px), sec_thresh, sec_shift);
                    let ds1 = constrain_wasm128(t, i16x8_sub(s1, px), sec_thresh, sec_shift);
                    let ds2 = constrain_wasm128(t, i16x8_sub(s2, px), sec_thresh, sec_shift);
                    let ds3 = constrain_wasm128(t, i16x8_sub(s3, px), sec_thresh, sec_shift);

                    let sec_sum = i16x8_add(i16x8_add(ds0, ds1), i16x8_add(ds2, ds3));
                    sum = i16x8_add(sum, i16x8_mul(sec_tap_v, sec_sum));

                    min_v = u16x8_min(min_v, u16x8_min(u16x8_min(s0, s1), u16x8_min(s2, s3)));
                    max_v = i16x8_max(max_v, i16x8_max(i16x8_max(s0, s1), i16x8_max(s2, s3)));
                }

                // Rounding: (sum - (sum < 0) + 8) >> 4
                let neg_mask = i16x8_gt(zero, sum);
                let adjusted = i16x8_add(sum, neg_mask);
                let adjusted = i16x8_add(adjusted, i16x8_splat(8));
                let adjusted = i16x8_shr(adjusted, 4);
                let result = i16x8_add(px, adjusted);
                let result = i16x8_max(result, min_v);
                let result = i16x8_min(result, max_v);
                let result_u8 = u8x16_narrow_i16x8(result, zero);

                let row_off = p_base.wrapping_add_signed(y as isize * stride);
                if w == 8 {
                    let mut out = [0u8; 16];
                    wasm_store_128!(&mut out, result_u8);
                    p_guard[row_off..row_off + 8].copy_from_slice(&out[0..8]);
                } else {
                    wasm_storei32!(&mut p_guard[row_off..row_off + 4], result_u8);
                }
            }
        } else {
            // Primary only — no min/max clamping
            for y in 0..h {
                let base = tmp_offset + y * TMP_STRIDE;
                let px = wasm_load_128!(&tmp[base..base + 8], [u16; 8]);
                let mut sum = zero;

                let mut pri_tap_k = pri_tap;
                for k in 0..2 {
                    let off = dav1d_cdef_directions[dir + 2][k] as isize;
                    let p0_i = (base as isize + off) as usize;
                    let p1_i = (base as isize - off) as usize;
                    let p0 = wasm_load_128!(&tmp[p0_i..p0_i + 8], [u16; 8]);
                    let p1 = wasm_load_128!(&tmp[p1_i..p1_i + 8], [u16; 8]);

                    let c0 = constrain_wasm128(t, i16x8_sub(p0, px), pri_thresh, pri_shift);
                    let c1 = constrain_wasm128(t, i16x8_sub(p1, px), pri_thresh, pri_shift);

                    let tap_v = i16x8_splat(pri_tap_k as i16);
                    sum = i16x8_add(sum, i16x8_mul(tap_v, i16x8_add(c0, c1)));
                    pri_tap_k = pri_tap_k & 3 | 2;
                }

                let neg_mask = i16x8_gt(zero, sum);
                let adjusted = i16x8_add(sum, neg_mask);
                let adjusted = i16x8_add(adjusted, i16x8_splat(8));
                let adjusted = i16x8_shr(adjusted, 4);
                let result = i16x8_add(px, adjusted);
                let result_u8 = u8x16_narrow_i16x8(result, zero);

                let row_off = p_base.wrapping_add_signed(y as isize * stride);
                if w == 8 {
                    let mut out = [0u8; 16];
                    wasm_store_128!(&mut out, result_u8);
                    p_guard[row_off..row_off + 8].copy_from_slice(&out[0..8]);
                } else {
                    wasm_storei32!(&mut p_guard[row_off..row_off + 4], result_u8);
                }
            }
        }
    } else {
        // Secondary only — no min/max clamping
        let sec_shift = (damping - sec_strength.ilog2() as c_int) as u32;
        let sec_thresh = i16x8_splat(sec_strength as i16);

        for y in 0..h {
            let base = tmp_offset + y * TMP_STRIDE;
            let px = wasm_load_128!(&tmp[base..base + 8], [u16; 8]);
            let mut sum = zero;

            for k in 0..2 {
                let off2 = dav1d_cdef_directions[dir + 4][k] as isize;
                let off3 = dav1d_cdef_directions[dir + 0][k] as isize;
                let s0_i = (base as isize + off2) as usize;
                let s1_i = (base as isize - off2) as usize;
                let s2_i = (base as isize + off3) as usize;
                let s3_i = (base as isize - off3) as usize;
                let s0 = wasm_load_128!(&tmp[s0_i..s0_i + 8], [u16; 8]);
                let s1 = wasm_load_128!(&tmp[s1_i..s1_i + 8], [u16; 8]);
                let s2 = wasm_load_128!(&tmp[s2_i..s2_i + 8], [u16; 8]);
                let s3 = wasm_load_128!(&tmp[s3_i..s3_i + 8], [u16; 8]);

                let sec_tap_k = (2 - k as i32) as i16;
                let sec_tap_v = i16x8_splat(sec_tap_k);
                let ds0 = constrain_wasm128(t, i16x8_sub(s0, px), sec_thresh, sec_shift);
                let ds1 = constrain_wasm128(t, i16x8_sub(s1, px), sec_thresh, sec_shift);
                let ds2 = constrain_wasm128(t, i16x8_sub(s2, px), sec_thresh, sec_shift);
                let ds3 = constrain_wasm128(t, i16x8_sub(s3, px), sec_thresh, sec_shift);

                let sec_sum = i16x8_add(i16x8_add(ds0, ds1), i16x8_add(ds2, ds3));
                sum = i16x8_add(sum, i16x8_mul(sec_tap_v, sec_sum));
            }

            let neg_mask = i16x8_gt(zero, sum);
            let adjusted = i16x8_add(sum, neg_mask);
            let adjusted = i16x8_add(adjusted, i16x8_splat(8));
            let adjusted = i16x8_shr(adjusted, 4);
            let result = i16x8_add(px, adjusted);
            let result_u8 = u8x16_narrow_i16x8(result, zero);

            let row_off = p_base.wrapping_add_signed(y as isize * stride);
            if w == 8 {
                let mut out = [0u8; 16];
                wasm_store_128!(&mut out, result_u8);
                p_guard[row_off..row_off + 8].copy_from_slice(&out[0..8]);
            } else {
                wasm_storei32!(&mut p_guard[row_off..row_off + 4], result_u8);
            }
        }
    }
}

// ============================================================================
// CDEF FILTER 16BPC (wasm128 SIMD)
// ============================================================================

/// Vectorized CDEF filter for 16bpc using wasm128 — processes 8 pixels per row.
#[arcane]
fn cdef_filter_block_simd_16bpc(
    t: Wasm128Token,
    tmp: &[u16],
    tmp_offset: usize,
    dst: PicOffset,
    w: usize,
    h: usize,
    dir: usize,
    pri_strength: c_int,
    sec_strength: c_int,
    damping: c_int,
    bitdepth_max: c_int,
) {
    use super::pixel_access::{wasm_load_128, wasm_store_128};
    use crate::include::common::bitdepth::BitDepth16;

    let zero = i16x8_splat(0);
    let bd_max = i16x8_splat(bitdepth_max as i16);
    let bitdepth_min_8 = ((bitdepth_max + 1) as u32).ilog2() as c_int - 8;

    // Tile-threading-safe block view: contiguous when threading is off, a
    // compact per-row copy when it is on. See `WithOffset::block_mut`.
    // Taps are read from the padded `tmp` copy, so only the w×h output region
    // is touched here.
    let mut block = dst.block_mut::<BitDepth16>(w, h);
    // The block has its own stride (compact when tile threading is on), so it
    // supersedes the caller-supplied picture stride. `p_guard` is indexed in
    // pixels, so halve the byte stride.
    let stride = block.byte_stride() / 2;
    let p_base = block.base();
    let p_guard: &mut [u16] = zerocopy::FromBytes::mut_from_bytes(block.as_mut_bytes())
        .expect("dst alignment/size mismatch for u16 reinterpretation");

    if pri_strength != 0 {
        let pri_tap = 4 - (pri_strength >> bitdepth_min_8 & 1);
        let pri_shift = cmp::max(0, damping - pri_strength.ilog2() as c_int) as u32;
        let pri_thresh = i16x8_splat(pri_strength as i16);

        if sec_strength != 0 {
            let sec_shift = (damping - sec_strength.ilog2() as c_int) as u32;
            let sec_thresh = i16x8_splat(sec_strength as i16);

            for y in 0..h {
                let base = tmp_offset + y * TMP_STRIDE;
                let px = wasm_load_128!(&tmp[base..base + 8], [u16; 8]);
                let mut sum = zero;
                let mut min_v = px;
                let mut max_v = px;

                let mut pri_tap_k = pri_tap;
                for k in 0..2 {
                    let off = dav1d_cdef_directions[dir + 2][k] as isize;
                    let p0_i = (base as isize + off) as usize;
                    let p1_i = (base as isize - off) as usize;
                    let p0 = wasm_load_128!(&tmp[p0_i..p0_i + 8], [u16; 8]);
                    let p1 = wasm_load_128!(&tmp[p1_i..p1_i + 8], [u16; 8]);

                    let c0 = constrain_wasm128(t, i16x8_sub(p0, px), pri_thresh, pri_shift);
                    let c1 = constrain_wasm128(t, i16x8_sub(p1, px), pri_thresh, pri_shift);

                    let tap_v = i16x8_splat(pri_tap_k as i16);
                    sum = i16x8_add(sum, i16x8_mul(tap_v, i16x8_add(c0, c1)));
                    pri_tap_k = pri_tap_k & 3 | 2;

                    min_v = u16x8_min(min_v, u16x8_min(p0, p1));
                    max_v = i16x8_max(max_v, i16x8_max(p0, p1));

                    let off2 = dav1d_cdef_directions[dir + 4][k] as isize;
                    let off3 = dav1d_cdef_directions[dir + 0][k] as isize;
                    let s0_i = (base as isize + off2) as usize;
                    let s1_i = (base as isize - off2) as usize;
                    let s2_i = (base as isize + off3) as usize;
                    let s3_i = (base as isize - off3) as usize;
                    let s0 = wasm_load_128!(&tmp[s0_i..s0_i + 8], [u16; 8]);
                    let s1 = wasm_load_128!(&tmp[s1_i..s1_i + 8], [u16; 8]);
                    let s2 = wasm_load_128!(&tmp[s2_i..s2_i + 8], [u16; 8]);
                    let s3 = wasm_load_128!(&tmp[s3_i..s3_i + 8], [u16; 8]);

                    let sec_tap_k = (2 - k as i32) as i16;
                    let sec_tap_v = i16x8_splat(sec_tap_k);
                    let ds0 = constrain_wasm128(t, i16x8_sub(s0, px), sec_thresh, sec_shift);
                    let ds1 = constrain_wasm128(t, i16x8_sub(s1, px), sec_thresh, sec_shift);
                    let ds2 = constrain_wasm128(t, i16x8_sub(s2, px), sec_thresh, sec_shift);
                    let ds3 = constrain_wasm128(t, i16x8_sub(s3, px), sec_thresh, sec_shift);

                    let sec_sum = i16x8_add(i16x8_add(ds0, ds1), i16x8_add(ds2, ds3));
                    sum = i16x8_add(sum, i16x8_mul(sec_tap_v, sec_sum));

                    min_v = u16x8_min(min_v, u16x8_min(u16x8_min(s0, s1), u16x8_min(s2, s3)));
                    max_v = i16x8_max(max_v, i16x8_max(i16x8_max(s0, s1), i16x8_max(s2, s3)));
                }

                let neg_mask = i16x8_gt(zero, sum);
                let adjusted = i16x8_add(sum, neg_mask);
                let adjusted = i16x8_add(adjusted, i16x8_splat(8));
                let adjusted = i16x8_shr(adjusted, 4);
                let result = i16x8_add(px, adjusted);
                let result = i16x8_max(result, min_v);
                let result = i16x8_min(result, max_v);

                let mut out = [0u16; 8];
                wasm_store_128!(&mut out, result);
                let row_off = p_base.wrapping_add_signed(y as isize * stride);
                p_guard[row_off..row_off + w].copy_from_slice(&out[..w]);
            }
        } else {
            // Primary only
            for y in 0..h {
                let base = tmp_offset + y * TMP_STRIDE;
                let px = wasm_load_128!(&tmp[base..base + 8], [u16; 8]);
                let mut sum = zero;

                let mut pri_tap_k = pri_tap;
                for k in 0..2 {
                    let off = dav1d_cdef_directions[dir + 2][k] as isize;
                    let p0_i = (base as isize + off) as usize;
                    let p1_i = (base as isize - off) as usize;
                    let p0 = wasm_load_128!(&tmp[p0_i..p0_i + 8], [u16; 8]);
                    let p1 = wasm_load_128!(&tmp[p1_i..p1_i + 8], [u16; 8]);

                    let c0 = constrain_wasm128(t, i16x8_sub(p0, px), pri_thresh, pri_shift);
                    let c1 = constrain_wasm128(t, i16x8_sub(p1, px), pri_thresh, pri_shift);

                    let tap_v = i16x8_splat(pri_tap_k as i16);
                    sum = i16x8_add(sum, i16x8_mul(tap_v, i16x8_add(c0, c1)));
                    pri_tap_k = pri_tap_k & 3 | 2;
                }

                let neg_mask = i16x8_gt(zero, sum);
                let adjusted = i16x8_add(sum, neg_mask);
                let adjusted = i16x8_add(adjusted, i16x8_splat(8));
                let adjusted = i16x8_shr(adjusted, 4);
                let result = i16x8_add(px, adjusted);
                // Clamp to [0, bitdepth_max]
                let result = i16x8_max(result, zero);
                let result = i16x8_min(result, bd_max);

                let mut out = [0u16; 8];
                wasm_store_128!(&mut out, result);
                let row_off = p_base.wrapping_add_signed(y as isize * stride);
                p_guard[row_off..row_off + w].copy_from_slice(&out[..w]);
            }
        }
    } else {
        // Secondary only
        let sec_shift = (damping - sec_strength.ilog2() as c_int) as u32;
        let sec_thresh = i16x8_splat(sec_strength as i16);

        for y in 0..h {
            let base = tmp_offset + y * TMP_STRIDE;
            let px = wasm_load_128!(&tmp[base..base + 8], [u16; 8]);
            let mut sum = zero;

            for k in 0..2 {
                let off2 = dav1d_cdef_directions[dir + 4][k] as isize;
                let off3 = dav1d_cdef_directions[dir + 0][k] as isize;
                let s0_i = (base as isize + off2) as usize;
                let s1_i = (base as isize - off2) as usize;
                let s2_i = (base as isize + off3) as usize;
                let s3_i = (base as isize - off3) as usize;
                let s0 = wasm_load_128!(&tmp[s0_i..s0_i + 8], [u16; 8]);
                let s1 = wasm_load_128!(&tmp[s1_i..s1_i + 8], [u16; 8]);
                let s2 = wasm_load_128!(&tmp[s2_i..s2_i + 8], [u16; 8]);
                let s3 = wasm_load_128!(&tmp[s3_i..s3_i + 8], [u16; 8]);

                let sec_tap_k = (2 - k as i32) as i16;
                let sec_tap_v = i16x8_splat(sec_tap_k);
                let ds0 = constrain_wasm128(t, i16x8_sub(s0, px), sec_thresh, sec_shift);
                let ds1 = constrain_wasm128(t, i16x8_sub(s1, px), sec_thresh, sec_shift);
                let ds2 = constrain_wasm128(t, i16x8_sub(s2, px), sec_thresh, sec_shift);
                let ds3 = constrain_wasm128(t, i16x8_sub(s3, px), sec_thresh, sec_shift);

                let sec_sum = i16x8_add(i16x8_add(ds0, ds1), i16x8_add(ds2, ds3));
                sum = i16x8_add(sum, i16x8_mul(sec_tap_v, sec_sum));
            }

            let neg_mask = i16x8_gt(zero, sum);
            let adjusted = i16x8_add(sum, neg_mask);
            let adjusted = i16x8_add(adjusted, i16x8_splat(8));
            let adjusted = i16x8_shr(adjusted, 4);
            let result = i16x8_add(px, adjusted);
            // Clamp to [0, bitdepth_max]
            let result = i16x8_max(result, zero);
            let result = i16x8_min(result, bd_max);

            let mut out = [0u16; 8];
            wasm_store_128!(&mut out, result);
            let row_off = p_base.wrapping_add_signed(y as isize * stride);
            p_guard[row_off..row_off + w].copy_from_slice(&out[..w]);
        }
    }
}

// ============================================================================
// INNER DISPATCH FUNCTIONS
// ============================================================================

fn cdef_filter_8x8_8bpc_wasm_inner(
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
    // Still needed by the scalar fallback below, which writes through the
    // picture's own stride.
    let stride = dst.pixel_stride::<BitDepth8>();

    if let Some(token) = crate::src::cpu::summon_wasm128() {
        cdef_filter_block_simd_8bpc(
            token,
            &tmp,
            tmp_offset,
            dst,
            8,
            8,
            dir as usize,
            pri_strength,
            sec_strength,
            damping,
        );
        return;
    }

    super::cdef::cdef_filter_block_scalar_8bpc(
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

fn cdef_filter_4x8_8bpc_wasm_inner(
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
    // Still needed by the scalar fallback below, which writes through the
    // picture's own stride.
    let stride = dst.pixel_stride::<BitDepth8>();

    if let Some(token) = crate::src::cpu::summon_wasm128() {
        cdef_filter_block_simd_8bpc(
            token,
            &tmp,
            tmp_offset,
            dst,
            4,
            8,
            dir as usize,
            pri_strength,
            sec_strength,
            damping,
        );
        return;
    }

    super::cdef::cdef_filter_block_scalar_8bpc(
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

fn cdef_filter_4x4_8bpc_wasm_inner(
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
    padding_8bpc(&mut tmp, dst, left, top, bottom, 4, 4, edges);

    let tmp_offset = 2 * TMP_STRIDE + 2;
    // Still needed by the scalar fallback below, which writes through the
    // picture's own stride.
    let stride = dst.pixel_stride::<BitDepth8>();

    if let Some(token) = crate::src::cpu::summon_wasm128() {
        cdef_filter_block_simd_8bpc(
            token,
            &tmp,
            tmp_offset,
            dst,
            4,
            4,
            dir as usize,
            pri_strength,
            sec_strength,
            damping,
        );
        return;
    }

    super::cdef::cdef_filter_block_scalar_8bpc(
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

fn cdef_filter_8x8_16bpc_wasm_inner(
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
    // Still needed by the scalar fallback below, which writes through the
    // picture's own stride.
    let stride = dst.pixel_stride::<BitDepth16>();

    if let Some(token) = crate::src::cpu::summon_wasm128() {
        cdef_filter_block_simd_16bpc(
            token,
            &tmp,
            tmp_offset,
            dst,
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

    super::cdef::cdef_filter_block_scalar_16bpc(
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

fn cdef_filter_4x8_16bpc_wasm_inner(
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
    // Still needed by the scalar fallback below, which writes through the
    // picture's own stride.
    let stride = dst.pixel_stride::<BitDepth16>();

    if let Some(token) = crate::src::cpu::summon_wasm128() {
        cdef_filter_block_simd_16bpc(
            token,
            &tmp,
            tmp_offset,
            dst,
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

    super::cdef::cdef_filter_block_scalar_16bpc(
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

fn cdef_filter_4x4_16bpc_wasm_inner(
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
    padding_16bpc(&mut tmp, dst, left, top, bottom, 4, 4, edges, bitdepth_max);

    let tmp_offset = 2 * TMP_STRIDE + 2;
    // Still needed by the scalar fallback below, which writes through the
    // picture's own stride.
    let stride = dst.pixel_stride::<BitDepth16>();

    if let Some(token) = crate::src::cpu::summon_wasm128() {
        cdef_filter_block_simd_16bpc(
            token,
            &tmp,
            tmp_offset,
            dst,
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

    super::cdef::cdef_filter_block_scalar_16bpc(
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

// ============================================================================
// PUBLIC DISPATCH
// ============================================================================

/// Dispatch CDEF filter to wasm128 SIMD implementation.
/// Returns `true` if handled, `false` to fall through to scalar.
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

    // Check if wasm128 is available at compile time
    if crate::src::cpu::summon_wasm128().is_none() {
        return false;
    }

    match (BD::BPC, variant) {
        (BPC::BPC8, 0) => {
            let left: &[LeftPixelRow2px<u8>; 8] =
                crate::src::safe_simd::pixel_access::reinterpret_ref(left)
                    .expect("BD::Pixel layout matches u8");
            cdef_filter_8x8_8bpc_wasm_inner(
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
            cdef_filter_4x8_8bpc_wasm_inner(
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
            cdef_filter_4x4_8bpc_wasm_inner(
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
            cdef_filter_8x8_16bpc_wasm_inner(
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
            cdef_filter_4x8_16bpc_wasm_inner(
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
            cdef_filter_4x4_16bpc_wasm_inner(
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
