//! Safe wasm128 SIMD implementations for ITX (Inverse Transforms)
//!
//! Implements the most common transforms for AVIF still images using
//! column-parallel processing: load 4 columns as i32x4 vectors,
//! apply butterfly across all 4 rows simultaneously.
//!
//! Currently implemented: DCT_DCT 4x4, 8x8 (8bpc and 16bpc)

#![deny(unsafe_code)]
#![allow(dead_code)]

use core::arch::wasm32::*;

use crate::include::common::bitdepth::BitDepth;
use crate::include::dav1d::picture::PicOffset;
use crate::src::levels::TxfmSize;
use crate::src::safe_simd::pixel_access::{wasm_loadi32, wasm_storei32};
use zerocopy::IntoBytes;

use crate::src::levels::DCT_DCT;

// ============================================================================
// DCT4 BUTTERFLY (processes 4 rows simultaneously)
// ============================================================================

/// 4-point DCT butterfly applied to 4 rows simultaneously.
///
/// Input: c0..c3 are columns of the 4×4 matrix.
///   c0[i] = row_i, element 0
///   c1[i] = row_i, element 1
///   ...
///
/// DCT4 formula per row:
///   t0 = (in0 + in2) * 181 + 128 >> 8
///   t1 = (in0 - in2) * 181 + 128 >> 8
///   t2 = (in1 * 1567 - in3 * (3784-4096) + 2048 >> 12) - in3
///   t3 = (in1 * (3784-4096) + in3 * 1567 + 2048 >> 12) + in1
///   out0 = clip(t0 + t3), out1 = clip(t1 + t2),
///   out2 = clip(t1 - t2), out3 = clip(t0 - t3)
#[inline(always)]
fn dct4_4rows(
    c0: v128,
    c1: v128,
    c2: v128,
    c3: v128,
    clip_min: i32,
    clip_max: i32,
) -> (v128, v128, v128, v128) {
    let sqrt2 = i32x4_splat(181);
    let rnd8 = i32x4_splat(128);
    let c1567_v = i32x4_splat(1567);
    let c_312_v = i32x4_splat(3784 - 4096); // -312
    let rnd12 = i32x4_splat(2048);

    // t0 = (c0 + c2) * 181 + 128 >> 8
    let t0 = i32x4_shr(i32x4_add(i32x4_mul(i32x4_add(c0, c2), sqrt2), rnd8), 8);
    // t1 = (c0 - c2) * 181 + 128 >> 8
    let t1 = i32x4_shr(i32x4_add(i32x4_mul(i32x4_sub(c0, c2), sqrt2), rnd8), 8);
    // t2 = (c1 * 1567 - c3 * (3784-4096) + 2048 >> 12) - c3
    let t2 = i32x4_sub(
        i32x4_shr(
            i32x4_add(
                i32x4_sub(i32x4_mul(c1, c1567_v), i32x4_mul(c3, c_312_v)),
                rnd12,
            ),
            12,
        ),
        c3,
    );
    // t3 = (c1 * (3784-4096) + c3 * 1567 + 2048 >> 12) + c1
    let t3 = i32x4_add(
        i32x4_shr(
            i32x4_add(
                i32x4_add(i32x4_mul(c1, c_312_v), i32x4_mul(c3, c1567_v)),
                rnd12,
            ),
            12,
        ),
        c1,
    );

    let vmin = i32x4_splat(clip_min);
    let vmax = i32x4_splat(clip_max);
    let out0 = i32x4_max(i32x4_min(i32x4_add(t0, t3), vmax), vmin);
    let out1 = i32x4_max(i32x4_min(i32x4_add(t1, t2), vmax), vmin);
    let out2 = i32x4_max(i32x4_min(i32x4_sub(t1, t2), vmax), vmin);
    let out3 = i32x4_max(i32x4_min(i32x4_sub(t0, t3), vmax), vmin);

    (out0, out1, out2, out3)
}

// ============================================================================
// 4x4 TRANSPOSE
// ============================================================================

/// Transpose a 4x4 matrix stored in 4 i32x4 registers.
/// Input: r0=[a0,a1,a2,a3], r1=[b0,b1,b2,b3], r2=[c0,c1,c2,c3], r3=[d0,d1,d2,d3]
/// Output: [a0,b0,c0,d0], [a1,b1,c1,d1], [a2,b2,c2,d2], [a3,b3,c3,d3]
#[inline(always)]
fn transpose_4x4(r0: v128, r1: v128, r2: v128, r3: v128) -> (v128, v128, v128, v128) {
    // Interleave low/high pairs
    let t01_lo = i32x4_shuffle::<0, 4, 1, 5>(r0, r1); // [a0, b0, a1, b1]
    let t01_hi = i32x4_shuffle::<2, 6, 3, 7>(r0, r1); // [a2, b2, a3, b3]
    let t23_lo = i32x4_shuffle::<0, 4, 1, 5>(r2, r3); // [c0, d0, c1, d1]
    let t23_hi = i32x4_shuffle::<2, 6, 3, 7>(r2, r3); // [c2, d2, c3, d3]

    // Combine to final columns
    let c0 = i64x2_shuffle::<0, 2>(t01_lo, t23_lo); // [a0, b0, c0, d0]
    let c1 = i64x2_shuffle::<1, 3>(t01_lo, t23_lo); // [a1, b1, c1, d1]
    let c2 = i64x2_shuffle::<0, 2>(t01_hi, t23_hi); // [a2, b2, c2, d2]
    let c3 = i64x2_shuffle::<1, 3>(t01_hi, t23_hi); // [a3, b3, c3, d3]

    (c0, c1, c2, c3)
}

// ============================================================================
// 4x4 DCT_DCT 8bpc
// ============================================================================

/// Full 2D DCT_DCT 4x4 inverse transform with add-to-destination (8bpc, wasm128)
fn inv_txfm_add_dct_dct_4x4_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
    // Load 4 columns from column-major coefficients
    // coeff layout (column-major): col0=[0,1,2,3], col1=[4,5,6,7], ...
    let c0 = i32x4(
        coeff[0] as i32,
        coeff[1] as i32,
        coeff[2] as i32,
        coeff[3] as i32,
    );
    let c1 = i32x4(
        coeff[4] as i32,
        coeff[5] as i32,
        coeff[6] as i32,
        coeff[7] as i32,
    );
    let c2 = i32x4(
        coeff[8] as i32,
        coeff[9] as i32,
        coeff[10] as i32,
        coeff[11] as i32,
    );
    let c3 = i32x4(
        coeff[12] as i32,
        coeff[13] as i32,
        coeff[14] as i32,
        coeff[15] as i32,
    );

    // Clip ranges
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

    // Row transform: DCT4 across all 4 rows simultaneously
    let (r0, r1, r2, r3) = dct4_4rows(c0, c1, c2, c3, row_clip_min, row_clip_max);

    // Transpose: rows → columns for column transform
    let (tc0, tc1, tc2, tc3) = transpose_4x4(r0, r1, r2, r3);

    // Intermediate clamp (shift=0 for 4x4)
    let cmin = i32x4_splat(col_clip_min);
    let cmax = i32x4_splat(col_clip_max);
    let tc0 = i32x4_max(i32x4_min(tc0, cmax), cmin);
    let tc1 = i32x4_max(i32x4_min(tc1, cmax), cmin);
    let tc2 = i32x4_max(i32x4_min(tc2, cmax), cmin);
    let tc3 = i32x4_max(i32x4_min(tc3, cmax), cmin);

    // Column transform: DCT4 across all 4 columns
    let (f0, f1, f2, f3) = dct4_4rows(tc0, tc1, tc2, tc3, col_clip_min, col_clip_max);

    // Transpose back to row-major for output
    let (out0, out1, out2, out3) = transpose_4x4(f0, f1, f2, f3);

    // Scale: (val + 8) >> 4
    let rnd = i32x4_splat(8);
    let out0 = i32x4_shr(i32x4_add(out0, rnd), 4);
    let out1 = i32x4_shr(i32x4_add(out1, rnd), 4);
    let out2 = i32x4_shr(i32x4_add(out2, rnd), 4);
    let out3 = i32x4_shr(i32x4_add(out3, rnd), 4);

    // Add to destination pixels and clamp
    let zero = i32x4_splat(0);
    let max_val = i32x4_splat(bitdepth_max);

    // Row 0
    let d0 = wasm_loadi32!(&dst[..4]);
    let d0_wide = i32x4(
        u8x16_extract_lane::<0>(d0) as i32,
        u8x16_extract_lane::<1>(d0) as i32,
        u8x16_extract_lane::<2>(d0) as i32,
        u8x16_extract_lane::<3>(d0) as i32,
    );
    let sum0 = i32x4_max(i32x4_min(i32x4_add(d0_wide, out0), max_val), zero);
    let packed0 = u8x16(
        i32x4_extract_lane::<0>(sum0) as u8,
        i32x4_extract_lane::<1>(sum0) as u8,
        i32x4_extract_lane::<2>(sum0) as u8,
        i32x4_extract_lane::<3>(sum0) as u8,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    );
    wasm_storei32!(&mut dst[..4], packed0);

    // Row 1
    let off1 = dst_stride;
    let d1 = wasm_loadi32!(&dst[off1..off1 + 4]);
    let d1_wide = i32x4(
        u8x16_extract_lane::<0>(d1) as i32,
        u8x16_extract_lane::<1>(d1) as i32,
        u8x16_extract_lane::<2>(d1) as i32,
        u8x16_extract_lane::<3>(d1) as i32,
    );
    let sum1 = i32x4_max(i32x4_min(i32x4_add(d1_wide, out1), max_val), zero);
    let packed1 = u8x16(
        i32x4_extract_lane::<0>(sum1) as u8,
        i32x4_extract_lane::<1>(sum1) as u8,
        i32x4_extract_lane::<2>(sum1) as u8,
        i32x4_extract_lane::<3>(sum1) as u8,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    );
    wasm_storei32!(&mut dst[off1..off1 + 4], packed1);

    // Row 2
    let off2 = dst_stride * 2;
    let d2 = wasm_loadi32!(&dst[off2..off2 + 4]);
    let d2_wide = i32x4(
        u8x16_extract_lane::<0>(d2) as i32,
        u8x16_extract_lane::<1>(d2) as i32,
        u8x16_extract_lane::<2>(d2) as i32,
        u8x16_extract_lane::<3>(d2) as i32,
    );
    let sum2 = i32x4_max(i32x4_min(i32x4_add(d2_wide, out2), max_val), zero);
    let packed2 = u8x16(
        i32x4_extract_lane::<0>(sum2) as u8,
        i32x4_extract_lane::<1>(sum2) as u8,
        i32x4_extract_lane::<2>(sum2) as u8,
        i32x4_extract_lane::<3>(sum2) as u8,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    );
    wasm_storei32!(&mut dst[off2..off2 + 4], packed2);

    // Row 3
    let off3 = dst_stride * 3;
    let d3 = wasm_loadi32!(&dst[off3..off3 + 4]);
    let d3_wide = i32x4(
        u8x16_extract_lane::<0>(d3) as i32,
        u8x16_extract_lane::<1>(d3) as i32,
        u8x16_extract_lane::<2>(d3) as i32,
        u8x16_extract_lane::<3>(d3) as i32,
    );
    let sum3 = i32x4_max(i32x4_min(i32x4_add(d3_wide, out3), max_val), zero);
    let packed3 = u8x16(
        i32x4_extract_lane::<0>(sum3) as u8,
        i32x4_extract_lane::<1>(sum3) as u8,
        i32x4_extract_lane::<2>(sum3) as u8,
        i32x4_extract_lane::<3>(sum3) as u8,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    );
    wasm_storei32!(&mut dst[off3..off3 + 4], packed3);

    // Clear coefficients
    coeff[..16].fill(0);
}

// ============================================================================
// 4x4 DCT_DCT 16bpc
// ============================================================================

/// Full 2D DCT_DCT 4x4 inverse transform with add-to-destination (16bpc, wasm128)
fn inv_txfm_add_dct_dct_4x4_16bpc(
    dst: &mut [u16],
    dst_stride_u16: usize,
    coeff: &mut [i32],
    _eob: i32,
    bitdepth_max: i32,
) {
    // Load 4 columns from column-major coefficients
    let c0 = i32x4(coeff[0], coeff[1], coeff[2], coeff[3]);
    let c1 = i32x4(coeff[4], coeff[5], coeff[6], coeff[7]);
    let c2 = i32x4(coeff[8], coeff[9], coeff[10], coeff[11]);
    let c3 = i32x4(coeff[12], coeff[13], coeff[14], coeff[15]);

    let row_clip_min = (!bitdepth_max) << 7;
    let row_clip_max = !row_clip_min;
    let col_clip_min = (!bitdepth_max) << 5;
    let col_clip_max = !col_clip_min;

    // Row transform
    let (r0, r1, r2, r3) = dct4_4rows(c0, c1, c2, c3, row_clip_min, row_clip_max);

    // Transpose
    let (tc0, tc1, tc2, tc3) = transpose_4x4(r0, r1, r2, r3);

    // Intermediate clamp (shift=0 for 4x4)
    let cmin = i32x4_splat(col_clip_min);
    let cmax = i32x4_splat(col_clip_max);
    let tc0 = i32x4_max(i32x4_min(tc0, cmax), cmin);
    let tc1 = i32x4_max(i32x4_min(tc1, cmax), cmin);
    let tc2 = i32x4_max(i32x4_min(tc2, cmax), cmin);
    let tc3 = i32x4_max(i32x4_min(tc3, cmax), cmin);

    // Column transform
    let (f0, f1, f2, f3) = dct4_4rows(tc0, tc1, tc2, tc3, col_clip_min, col_clip_max);

    // Transpose back
    let (out0, out1, out2, out3) = transpose_4x4(f0, f1, f2, f3);

    // Scale: (val + 8) >> 4
    let rnd = i32x4_splat(8);
    let out0 = i32x4_shr(i32x4_add(out0, rnd), 4);
    let out1 = i32x4_shr(i32x4_add(out1, rnd), 4);
    let out2 = i32x4_shr(i32x4_add(out2, rnd), 4);
    let out3 = i32x4_shr(i32x4_add(out3, rnd), 4);

    let zero = i32x4_splat(0);
    let max_val = i32x4_splat(bitdepth_max);

    // Add to destination and clamp (16bpc: u16 pixels)
    for (row_idx, out_row) in [(0, out0), (1, out1), (2, out2), (3, out3)] {
        let off = row_idx * dst_stride_u16;
        let d = i32x4(
            dst[off] as i32,
            dst[off + 1] as i32,
            dst[off + 2] as i32,
            dst[off + 3] as i32,
        );
        let sum = i32x4_max(i32x4_min(i32x4_add(d, out_row), max_val), zero);
        dst[off] = i32x4_extract_lane::<0>(sum) as u16;
        dst[off + 1] = i32x4_extract_lane::<1>(sum) as u16;
        dst[off + 2] = i32x4_extract_lane::<2>(sum) as u16;
        dst[off + 3] = i32x4_extract_lane::<3>(sum) as u16;
    }

    // Clear coefficients
    coeff[..16].fill(0);
}

// ============================================================================
// DCT8 BUTTERFLY (processes 4 rows simultaneously via column vectors)
// ============================================================================

/// 8-point DCT butterfly applied to 4 rows simultaneously.
///
/// Takes 8 column vectors (c0..c7), each i32x4 holding the same column index
/// across 4 different rows. Returns 8 output column vectors.
///
/// DCT8 = DCT4 on even-indexed elements + butterfly on odd-indexed elements.
/// Follows rav1d scalar inv_dct8_1d_internal_c exactly.
#[inline(always)]
fn dct8_4rows(
    c0: v128,
    c1: v128,
    c2: v128,
    c3: v128,
    c4: v128,
    c5: v128,
    c6: v128,
    c7: v128,
    clip_min: i32,
    clip_max: i32,
) -> (v128, v128, v128, v128, v128, v128, v128, v128) {
    let vmin = i32x4_splat(clip_min);
    let vmax = i32x4_splat(clip_max);
    let clip = |v: v128| i32x4_max(i32x4_min(v, vmax), vmin);

    // Even: DCT4 on c0, c2, c4, c6
    let (e0, e1, e2, e3) = dct4_4rows(c0, c2, c4, c6, clip_min, clip_max);

    // Odd butterfly
    let c799_v = i32x4_splat(799);
    let c4017_off = i32x4_splat(4017 - 4096); // -79 offset for overflow-safe form
    let c1703_v = i32x4_splat(1703);
    let c1138_v = i32x4_splat(1138);
    let sqrt2 = i32x4_splat(181);
    let rnd12 = i32x4_splat(2048);
    let rnd11 = i32x4_splat(1024);
    let rnd8 = i32x4_splat(128);

    // t4a = (c1 * 799 - c7 * (4017-4096) + 2048 >> 12) - c7
    let t4a = i32x4_sub(
        i32x4_shr(
            i32x4_add(
                i32x4_sub(i32x4_mul(c1, c799_v), i32x4_mul(c7, c4017_off)),
                rnd12,
            ),
            12,
        ),
        c7,
    );
    // t7a = (c1 * (4017-4096) + c7 * 799 + 2048 >> 12) + c1
    let t7a = i32x4_add(
        i32x4_shr(
            i32x4_add(
                i32x4_add(i32x4_mul(c1, c4017_off), i32x4_mul(c7, c799_v)),
                rnd12,
            ),
            12,
        ),
        c1,
    );
    // t5a = c5 * 1703 - c3 * 1138 + 1024 >> 11
    let t5a = i32x4_shr(
        i32x4_add(
            i32x4_sub(i32x4_mul(c5, c1703_v), i32x4_mul(c3, c1138_v)),
            rnd11,
        ),
        11,
    );
    // t6a = c5 * 1138 + c3 * 1703 + 1024 >> 11
    let t6a = i32x4_shr(
        i32x4_add(
            i32x4_add(i32x4_mul(c5, c1138_v), i32x4_mul(c3, c1703_v)),
            rnd11,
        ),
        11,
    );

    // Butterfly
    let t4 = clip(i32x4_add(t4a, t5a));
    let t5a_diff = clip(i32x4_sub(t4a, t5a));
    let t7 = clip(i32x4_add(t7a, t6a));
    let t6a_diff = clip(i32x4_sub(t7a, t6a));

    // Rotation
    let t5 = i32x4_shr(
        i32x4_add(i32x4_mul(i32x4_sub(t6a_diff, t5a_diff), sqrt2), rnd8),
        8,
    );
    let t6 = i32x4_shr(
        i32x4_add(i32x4_mul(i32x4_add(t6a_diff, t5a_diff), sqrt2), rnd8),
        8,
    );

    // Final combine: even ± odd
    let out0 = clip(i32x4_add(e0, t7));
    let out1 = clip(i32x4_add(e1, t6));
    let out2 = clip(i32x4_add(e2, t5));
    let out3 = clip(i32x4_add(e3, t4));
    let out4 = clip(i32x4_sub(e3, t4));
    let out5 = clip(i32x4_sub(e2, t5));
    let out6 = clip(i32x4_sub(e1, t6));
    let out7 = clip(i32x4_sub(e0, t7));

    (out0, out1, out2, out3, out4, out5, out6, out7)
}

// ============================================================================
// 8x4 TRANSPOSE (transpose an 8×4 matrix to 4×8)
// ============================================================================

/// Transpose 8 column vectors of 4 elements each into 4 column vectors of 8 elements.
/// Input: c0..c7 are columns [row0, row1, row2, row3].
/// Output: r0..r3 where each r is two i32x4 registers (lo=elements 0-3, hi=elements 4-7).
///
/// This is used for the 8x8 transform: after the row DCT8 produces 8 output columns
/// of 4 rows, we transpose to get 4 rows of 8 elements for the column DCT8.
#[inline(always)]
fn transpose_8x4_to_4x8(
    c0: v128,
    c1: v128,
    c2: v128,
    c3: v128,
    c4: v128,
    c5: v128,
    c6: v128,
    c7: v128,
) -> ((v128, v128), (v128, v128), (v128, v128), (v128, v128)) {
    // First, transpose the 4x4 blocks
    let (t0_lo, t1_lo, t2_lo, t3_lo) = transpose_4x4(c0, c1, c2, c3);
    let (t0_hi, t1_hi, t2_hi, t3_hi) = transpose_4x4(c4, c5, c6, c7);
    // Now t0_lo = [c0[0], c1[0], c2[0], c3[0]] — first 4 elements of row 0
    // and t0_hi = [c4[0], c5[0], c6[0], c7[0]] — last 4 elements of row 0
    (
        (t0_lo, t0_hi),
        (t1_lo, t1_hi),
        (t2_lo, t2_hi),
        (t3_lo, t3_hi),
    )
}

// ============================================================================
// 8x8 DCT_DCT 8bpc
// ============================================================================

/// Full 2D DCT_DCT 8x8 inverse transform with add-to-destination (8bpc, wasm128)
///
/// Processes the 8x8 matrix in two halves (rows 0-3 and rows 4-7), each using
/// 8 i32x4 column vectors for the 8-point DCT butterfly.
fn inv_txfm_add_dct_dct_8x8_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i16],
    _eob: i32,
    bitdepth_max: i32,
) {
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

    // Load coefficients: column-major, 8 columns × 8 rows
    // Process rows 0-3 first, then rows 4-7
    let load_col_lo = |col: usize| -> v128 {
        let base = col * 8;
        i32x4(
            coeff[base] as i32,
            coeff[base + 1] as i32,
            coeff[base + 2] as i32,
            coeff[base + 3] as i32,
        )
    };
    let load_col_hi = |col: usize| -> v128 {
        let base = col * 8 + 4;
        i32x4(
            coeff[base] as i32,
            coeff[base + 1] as i32,
            coeff[base + 2] as i32,
            coeff[base + 3] as i32,
        )
    };

    // Row DCT8 on rows 0-3 (low halves of all 8 columns)
    let (r0_lo, r1_lo, r2_lo, r3_lo, r4_lo, r5_lo, r6_lo, r7_lo) = dct8_4rows(
        load_col_lo(0),
        load_col_lo(1),
        load_col_lo(2),
        load_col_lo(3),
        load_col_lo(4),
        load_col_lo(5),
        load_col_lo(6),
        load_col_lo(7),
        row_clip_min,
        row_clip_max,
    );

    // Row DCT8 on rows 4-7 (high halves of all 8 columns)
    let (r0_hi, r1_hi, r2_hi, r3_hi, r4_hi, r5_hi, r6_hi, r7_hi) = dct8_4rows(
        load_col_hi(0),
        load_col_hi(1),
        load_col_hi(2),
        load_col_hi(3),
        load_col_hi(4),
        load_col_hi(5),
        load_col_hi(6),
        load_col_hi(7),
        row_clip_min,
        row_clip_max,
    );

    // Transpose: 8 columns × 8 rows → 8 columns × 8 rows
    // Each output column needs values from all 8 input rows.
    // We have the row results split into lo (rows 0-3) and hi (rows 4-7).
    //
    // After row DCT, r0_lo = [row0_out0, row1_out0, row2_out0, row3_out0]
    //                r0_hi = [row4_out0, row5_out0, row6_out0, row7_out0]
    // These are already "column 0" of the row-transformed matrix!
    // So the transpose from row-major → column-major is already done by the
    // column-parallel structure. We just need to apply intermediate clamp and shift.

    // Intermediate shift and clamp (shift=1 for 8x8)
    let rnd_shift = i32x4_splat(1); // 1 << 1 >> 1 = 1
    let cmin = i32x4_splat(col_clip_min);
    let cmax = i32x4_splat(col_clip_max);
    let clamp_shift = |v: v128| -> v128 {
        i32x4_max(i32x4_min(i32x4_shr(i32x4_add(v, rnd_shift), 1), cmax), cmin)
    };

    let c0_lo = clamp_shift(r0_lo);
    let c0_hi = clamp_shift(r0_hi);
    let c1_lo = clamp_shift(r1_lo);
    let c1_hi = clamp_shift(r1_hi);
    let c2_lo = clamp_shift(r2_lo);
    let c2_hi = clamp_shift(r2_hi);
    let c3_lo = clamp_shift(r3_lo);
    let c3_hi = clamp_shift(r3_hi);
    let c4_lo = clamp_shift(r4_lo);
    let c4_hi = clamp_shift(r4_hi);
    let c5_lo = clamp_shift(r5_lo);
    let c5_hi = clamp_shift(r5_hi);
    let c6_lo = clamp_shift(r6_lo);
    let c6_hi = clamp_shift(r6_hi);
    let c7_lo = clamp_shift(r7_lo);
    let c7_hi = clamp_shift(r7_hi);

    // Column DCT8: we now need to apply DCT8 along the column direction.
    // Each "column" has 8 values split across _lo (rows 0-3) and _hi (rows 4-7).
    // But dct8_4rows processes 4 independent rows. We need to repack.
    //
    // For the column transform, we need:
    //   col_j elements = [c_j_lo[0], c_j_lo[1], c_j_lo[2], c_j_lo[3],
    //                     c_j_hi[0], c_j_hi[1], c_j_hi[2], c_j_hi[3]]
    // That's an 8-element column. To apply DCT8, we need the 8 elements of
    // each column accessible as the "8 inputs" to dct8.
    //
    // With i32x4 (4 elements), we process 4 columns simultaneously.
    // Transpose the 8×4 blocks to get column data in the right layout.

    // Transpose lo block: 8 columns of 4 rows → 4 rows of 8 columns
    let ((row0_lo, row0_hi), (row1_lo, row1_hi), (row2_lo, row2_hi), (row3_lo, row3_hi)) =
        transpose_8x4_to_4x8(c0_lo, c1_lo, c2_lo, c3_lo, c4_lo, c5_lo, c6_lo, c7_lo);

    // Transpose hi block
    let ((row4_lo, row4_hi), (row5_lo, row5_hi), (row6_lo, row6_hi), (row7_lo, row7_hi)) =
        transpose_8x4_to_4x8(c0_hi, c1_hi, c2_hi, c3_hi, c4_hi, c5_hi, c6_hi, c7_hi);

    // Column DCT8 on columns 0-3 (using _lo parts of each row)
    let (f0_lo, f1_lo, f2_lo, f3_lo, f4_lo, f5_lo, f6_lo, f7_lo) = dct8_4rows(
        row0_lo,
        row1_lo,
        row2_lo,
        row3_lo,
        row4_lo,
        row5_lo,
        row6_lo,
        row7_lo,
        col_clip_min,
        col_clip_max,
    );

    // Column DCT8 on columns 4-7 (using _hi parts of each row)
    let (f0_hi, f1_hi, f2_hi, f3_hi, f4_hi, f5_hi, f6_hi, f7_hi) = dct8_4rows(
        row0_hi,
        row1_hi,
        row2_hi,
        row3_hi,
        row4_hi,
        row5_hi,
        row6_hi,
        row7_hi,
        col_clip_min,
        col_clip_max,
    );

    // Scale: (val + 8) >> 4, then add to destination and clamp
    let rnd = i32x4_splat(8);

    // Helper: scale, add to dst pixels, clamp, write back one row of 8 pixels
    #[inline(always)]
    fn write_row_8bpc(dst: &mut [u8], off: usize, out_lo: v128, out_hi: v128, bdmax: i32) {
        let rnd = i32x4_splat(8);
        let out_lo = i32x4_shr(i32x4_add(out_lo, rnd), 4);
        let out_hi = i32x4_shr(i32x4_add(out_hi, rnd), 4);
        // Extract and write pixels 0-3
        dst[off + 0] =
            (dst[off + 0] as i32 + i32x4_extract_lane::<0>(out_lo)).clamp(0, bdmax) as u8;
        dst[off + 1] =
            (dst[off + 1] as i32 + i32x4_extract_lane::<1>(out_lo)).clamp(0, bdmax) as u8;
        dst[off + 2] =
            (dst[off + 2] as i32 + i32x4_extract_lane::<2>(out_lo)).clamp(0, bdmax) as u8;
        dst[off + 3] =
            (dst[off + 3] as i32 + i32x4_extract_lane::<3>(out_lo)).clamp(0, bdmax) as u8;
        // Extract and write pixels 4-7
        dst[off + 4] =
            (dst[off + 4] as i32 + i32x4_extract_lane::<0>(out_hi)).clamp(0, bdmax) as u8;
        dst[off + 5] =
            (dst[off + 5] as i32 + i32x4_extract_lane::<1>(out_hi)).clamp(0, bdmax) as u8;
        dst[off + 6] =
            (dst[off + 6] as i32 + i32x4_extract_lane::<2>(out_hi)).clamp(0, bdmax) as u8;
        dst[off + 7] =
            (dst[off + 7] as i32 + i32x4_extract_lane::<3>(out_hi)).clamp(0, bdmax) as u8;
    }
    let _ = rnd; // used inside write_row_8bpc

    write_row_8bpc(dst, 0, f0_lo, f0_hi, bitdepth_max);
    write_row_8bpc(dst, dst_stride, f1_lo, f1_hi, bitdepth_max);
    write_row_8bpc(dst, dst_stride * 2, f2_lo, f2_hi, bitdepth_max);
    write_row_8bpc(dst, dst_stride * 3, f3_lo, f3_hi, bitdepth_max);
    write_row_8bpc(dst, dst_stride * 4, f4_lo, f4_hi, bitdepth_max);
    write_row_8bpc(dst, dst_stride * 5, f5_lo, f5_hi, bitdepth_max);
    write_row_8bpc(dst, dst_stride * 6, f6_lo, f6_hi, bitdepth_max);
    write_row_8bpc(dst, dst_stride * 7, f7_lo, f7_hi, bitdepth_max);

    // Clear coefficients
    coeff[..64].fill(0);
}

// ============================================================================
// DISPATCH
// ============================================================================

/// Safe dispatch for ITX on wasm32. Returns true if a SIMD implementation handled the call.
pub fn itxfm_add_dispatch<BD: BitDepth>(
    tx_size: usize,
    tx_type: usize,
    dst: &mut crate::src::owned_recon::ReconDst<'_>,
    coeff: &mut [BD::Coef],
    eob: i32,
    bd: BD,
) -> bool {
    use crate::include::common::bitdepth::BPC;

    let txfm = match TxfmSize::from_repr(tx_size) {
        Some(t) => t,
        None => return false,
    };
    let (w, h) = txfm.to_wh();
    let bd_c = bd.into_c();

    // Only handle DCT_DCT for now
    if tx_type as u8 != DCT_DCT {
        return false;
    }

    match BD::BPC {
        BPC::BPC8 => {
            // Reinterpret coeff as &mut [i16] via zerocopy
            let coeff_i16: &mut [i16] = zerocopy::FromBytes::mut_from_bytes(coeff.as_mut_bytes())
                .expect("coeff alignment/size mismatch for i16 reinterpretation");

            // Tile-threading-safe block view: contiguous when threading is
            // off, a compact per-row copy when it is on. See
            // `WithOffset::block_mut`.
            let mut block = dst.block_mut::<BD>(w, h);
            let byte_stride_u = block.byte_stride().unsigned_abs();
            let dst_u8: &mut [u8] = block.as_mut_bytes();

            match txfm {
                TxfmSize::S4x4 => {
                    inv_txfm_add_dct_dct_4x4_8bpc(dst_u8, byte_stride_u, coeff_i16, eob, bd_c);
                    true
                }
                TxfmSize::S8x8 => {
                    inv_txfm_add_dct_dct_8x8_8bpc(dst_u8, byte_stride_u, coeff_i16, eob, bd_c);
                    true
                }
                _ => false,
            }
        }
        BPC::BPC16 => {
            let coeff_i32: &mut [i32] = zerocopy::FromBytes::mut_from_bytes(coeff.as_mut_bytes())
                .expect("coeff alignment/size mismatch for i32 reinterpretation");

            // Tile-threading-safe block view: contiguous when threading is
            // off, a compact per-row copy when it is on. See
            // `WithOffset::block_mut`.
            let mut block = dst.block_mut::<BD>(w, h);
            let byte_stride_u = block.byte_stride().unsigned_abs();
            let dst_bytes: &mut [u8] = block.as_mut_bytes();
            let dst_u16: &mut [u16] = zerocopy::FromBytes::mut_from_bytes(dst_bytes)
                .expect("dst alignment/size mismatch for u16 reinterpretation");
            let stride_u16 = byte_stride_u / 2;

            match txfm {
                TxfmSize::S4x4 => {
                    inv_txfm_add_dct_dct_4x4_16bpc(dst_u16, stride_u16, coeff_i32, eob, bd_c);
                    true
                }
                _ => false,
            }
        }
    }
}
