//! Loop restoration on aarch64: a safe NEON tier over `src/looprestoration.rs`.
//!
//! # History — what used to be here, and why measuring mattered
//!
//! Until 2026-08-07 this file held 1,531 lines that opened `//! Safe ARM NEON
//! implementations for Loop Restoration`, imported `core::arch::aarch64::*`,
//! and contained **zero aarch64 intrinsic calls**: a hand-written scalar
//! re-implementation of the reference, dispatched unconditionally. It was
//! *slower* than the code it shadowed (8bpc `00001147`: 204.42 vs 192.83
//! ms/frame whole-decode, ratio 1.060) and was deleted rather than ported on
//! faith. That measurement, and the corpus-wide activity scan that made it
//! possible, are recorded in
//! `benchmarks/lr_arm_vs_reference_2026-08-07.meta`.
//!
//! Two facts from that record shape this file:
//!
//! * **The 4K gap vectors do no loop restoration at all.** `v4k_8tile{,_10b}`
//!   measure LR at 0.0 ms because LR is switched off in those bitstreams, so
//!   nothing in the gap-to-dav1d table can see this code. LR is active in
//!   **696 of 768** dav1d-test-data vectors; `md5_inventory --activity`
//!   (needs `--features __ablate`) is the instrument that says which.
//! * The profile blamed `selfguided_filter` (9.5% of decode self time on
//!   `10-bit/issues/318_tx_4x4`) and the Wiener filter (2.0%), plus a large
//!   `_platform_memset` share from the per-unit scratch.
//!
//! # What this tier does
//!
//! Every variant the dispatcher can be handed is covered at both bit depths:
//! Wiener (the 7-tap window, which also expresses the 5-tap case with zero
//! outer coefficients) and self-guided 5x5 / 3x3 / mix.
//!
//! The three kernels, in decreasing order of measured cost:
//!
//! 1. **`boxsum{3,5}`, fused.** The reference walks each column top-to-bottom
//!    for the vertical sum (a 390-element stride per step) and then slides
//!    horizontally in place. This tier computes ONE row of vertical sums into
//!    a small row buffer and immediately slides it into the destination row.
//!    That turns a column-major pass over a 106 KB array into a row-major
//!    stream over three (or five) source rows, and — because the horizontal
//!    input is a separate buffer — removes the read-after-overwrite aliasing
//!    that forces the reference's scalar carry chain.
//! 2. **The `a`/`b` (`sgr_x_by_x`) loop.** Vectorised 16 columns at a time.
//!    The 256-entry `dav1d_sgr_x_by_x` lookup is done with four `vqtbl4q_u8`
//!    over the table's four 64-byte quarters: `vqtbl4q_u8` returns 0 for any
//!    index above 63, and the index is offset by a wrapping `vsubq_u8` per
//!    quarter, so exactly one quarter contributes per lane and the four
//!    results OR together. `z` is clamped to 255 before the narrow, so no
//!    index can alias into another quarter.
//! 3. **The six/eight-neighbour output pass**, and the final `w0/w1` blend
//!    into the picture.
//!
//! # Exactness
//!
//! The reference is the oracle, and the arithmetic here is written to match it
//! bit for bit, including where it relies on wrapping:
//!
//! * `x * b * sgr_one_by_x` is `c_uint` (u32) arithmetic in the reference and
//!   genuinely overflows at 12bpc on flat content (255 * 102375 * 455 is
//!   1.19e10). `vmulq_u32` wraps identically. Do not "fix" this to u64.
//! * `dst[i] = (...).as_()` is a truncating cast, so the narrow is
//!   `vmovn_s32`, never a saturating `vqmovn_s32`.
//! * The 8bpc horizontal Wiener pass folds the separate `tmp[i+3] * 128` term
//!   into tap 3. `lr_apply.rs` builds `filter[0][3] = -2*(f0+f1+f2)` and only
//!   adds 128 itself for non-8bpc; the AV1 tap ranges bound the folded value
//!   to [0, 218], and this pass accumulates in i32, so the fold cannot
//!   overflow. (The comment in `lr_apply.rs` about handling +128 separately is
//!   about i16 accumulators in hand-written asm, which this is not.)
//!
//! Two gates enforce that: `--features __simd_test` re-runs the scalar
//! reference after every call and compares the `w`x`h` output region, and
//! `examples/md5_inventory` set-diffs all 768 corpus vectors BY NAME with the
//! actual MD5 in the key.
//!
//! # Guard discipline
//!
//! Reads of the picture happen only inside `looprestoration::padding`, which
//! already takes one guard per source row. The write side here is likewise one
//! guard per destination row, exactly `w` pixels wide — never a single wide
//! guard spanning `(h - 1) * stride + w`, which would overlap rows a
//! concurrent tile worker owns. Everything else (`tmp`, `hor`, `sumsq`, `sum`,
//! `dst`) is a private stack buffer.

#![cfg_attr(not(feature = "unchecked"), forbid(unsafe_code))]
#![cfg_attr(feature = "unchecked", deny(unsafe_code))]
#![allow(unused_imports)]
#![allow(clippy::too_many_arguments)]

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
use archmage::{Arm64, arcane, rite};
#[cfg(target_arch = "aarch64")]
use safe_unaligned_simd::aarch64 as safe_simd;

use std::cmp;
use std::ffi::c_int;

use crate::include::common::bitdepth::BitDepth;
use crate::include::common::bitdepth::LeftPixelRow;
use crate::include::dav1d::picture::PicOffset;
use crate::src::align::AlignedVec64;
use crate::src::disjoint_mut::DisjointMut;
use crate::src::looprestoration::{LooprestorationParams, LrEdgeFlags};

#[cfg(target_arch = "aarch64")]
use crate::include::common::bitdepth::BitDepth16;
#[cfg(target_arch = "aarch64")]
use crate::include::common::bitdepth::BitDepth8;
#[cfg(target_arch = "aarch64")]
use crate::include::common::intops::iclip;
#[cfg(target_arch = "aarch64")]
use crate::src::looprestoration::padding;
#[cfg(target_arch = "aarch64")]
use crate::src::strided::Strided as _;
#[cfg(target_arch = "aarch64")]
use crate::src::tables::dav1d_sgr_x_by_x;

/// Restoration-unit scratch stride, matching `looprestoration::REST_UNIT_STRIDE`.
#[cfg(target_arch = "aarch64")]
const S: usize = 256 * 3 / 2 + 3 + 3; // 390
/// Widest restoration unit (`256 * 1.5`), the `dst` row stride.
#[cfg(target_arch = "aarch64")]
const MAXW: usize = 256 * 3 / 2; // 384
#[cfg(target_arch = "aarch64")]
const TMP_LEN: usize = (64 + 3 + 3) * S;
#[cfg(target_arch = "aarch64")]
const BOX_LEN: usize = (64 + 2 + 2) * S;
#[cfg(target_arch = "aarch64")]
const DST_LEN: usize = 64 * MAXW;

// ============================================================================
// WIENER
// ============================================================================

/// Horizontal 7-tap pass, 8bpc: `tmp` (u8) -> `hor` (u16), `h + 6` rows.
///
/// `taps` already has the 8bpc `* 128` centre term folded into index 3.
#[cfg(target_arch = "aarch64")]
#[arcane]
fn wiener_hor_8bpc(_token: Arm64, tmp: &[u8; TMP_LEN], hor: &mut [u16; TMP_LEN], w: usize, h: usize, taps: &[i16; 8]) {
    // (sum + rounding_off_h) >> 3, clipped to [0, (1 << 13) - 1].
    const BIAS: i32 = (1 << 14) + (1 << 2);
    let vbias = vdupq_n_s32(BIAS);
    let vmax = vdupq_n_s32((1 << 13) - 1);
    let vzero = vdupq_n_s32(0);
    let t: [int16x8_t; 7] = core::array::from_fn(|k| vdupq_n_s16(taps[k]));

    for row in 0..h + 6 {
        let base = row * S;
        let src = &tmp[base..base + w + 6];
        let dst = &mut hor[base..base + w];
        let mut x = 0;
        while x + 8 <= w {
            let mut lo = vbias;
            let mut hi = vbias;
            for k in 0..7 {
                let v = safe_simd::vld1_u8(src[x + k..][..8].try_into().unwrap());
                let v16 = vreinterpretq_s16_u16(vmovl_u8(v));
                lo = vmlal_s16(lo, vget_low_s16(v16), vget_low_s16(t[k]));
                hi = vmlal_high_s16(hi, v16, t[k]);
            }
            let lo = vminq_s32(vmaxq_s32(vshrq_n_s32::<3>(lo), vzero), vmax);
            let hi = vminq_s32(vmaxq_s32(vshrq_n_s32::<3>(hi), vzero), vmax);
            let packed = vcombine_u16(vmovn_u32(vreinterpretq_u32_s32(lo)), vmovn_u32(vreinterpretq_u32_s32(hi)));
            safe_simd::vst1q_u16((&mut dst[x..x + 8]).try_into().unwrap(), packed);
            x += 8;
        }
        while x < w {
            let mut sum = BIAS;
            for k in 0..7 {
                sum += src[x + k] as i32 * taps[k] as i32;
            }
            dst[x] = iclip(sum >> 3, 0, (1 << 13) - 1) as u16;
            x += 1;
        }
    }
}

/// Vertical 7-tap pass, 8bpc: `hor` (u16) -> picture (u8), one row guard per row.
#[cfg(target_arch = "aarch64")]
#[arcane]
fn wiener_ver_8bpc(_token: Arm64, hor: &[u16; TMP_LEN], p: PicOffset, w: usize, h: usize, taps: &[i16; 8]) {
    // -round_offset + rounding_off_v, then >> 11, clipped to [0, 255].
    const BIAS: i32 = -(1 << 18) + (1 << 10);
    let vbias = vdupq_n_s32(BIAS);
    let t: [int16x8_t; 7] = core::array::from_fn(|k| vdupq_n_s16(taps[k]));
    let stride = p.pixel_stride::<BitDepth8>();

    for j in 0..h {
        let mut dst = (p + (j as isize * stride)).slice_mut::<BitDepth8>(w);
        let mut x = 0;
        while x + 8 <= w {
            let mut lo = vbias;
            let mut hi = vbias;
            for k in 0..7 {
                let v = safe_simd::vld1q_u16(hor[(j + k) * S + x..][..8].try_into().unwrap());
                let v16 = vreinterpretq_s16_u16(v);
                lo = vmlal_s16(lo, vget_low_s16(v16), vget_low_s16(t[k]));
                hi = vmlal_high_s16(hi, v16, t[k]);
            }
            // `vqmovn` saturates i32 -> i16 and `vqmovun` i16 -> u8; the value
            // is far inside i16 before the second step, so the pair is exactly
            // `iclip(_, 0, 255)`.
            let packed = vqmovun_s16(vcombine_s16(
                vqmovn_s32(vshrq_n_s32::<11>(lo)),
                vqmovn_s32(vshrq_n_s32::<11>(hi)),
            ));
            safe_simd::vst1_u8((&mut dst[x..x + 8]).try_into().unwrap(), packed);
            x += 8;
        }
        while x < w {
            let mut sum = BIAS;
            for k in 0..7 {
                sum += hor[(j + k) * S + x] as i32 * taps[k] as i32;
            }
            dst[x] = iclip(sum >> 11, 0, 255) as u8;
            x += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn wiener_8bpc(
    token: Arm64,
    p: PicOffset,
    left: &[LeftPixelRow<u8>],
    lpf: &DisjointMut<AlignedVec64<u8>>,
    lpf_off: isize,
    w: usize,
    h: usize,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
) {
    let mut tmp = [0u8; TMP_LEN];
    padding::<BitDepth8>(&mut tmp, p, left, lpf, lpf_off, w, h, edges);
    let mut hor = [0u16; TMP_LEN];

    // Fold the 8bpc `tmp[i + 3] * 128` term into tap 3; see the module header.
    let mut hf = params.filter[0];
    hf[3] += 128;
    wiener_hor_8bpc(token, &tmp, &mut hor, w, h, &hf);
    wiener_ver_8bpc(token, &hor, p, w, h, &params.filter[1]);
}

/// Horizontal 7-tap pass, 16bpc. `round_bits_h` is 3 (10bpc) or 5 (12bpc), so
/// the shift is a runtime `vshlq_s32` by a negative count rather than a const.
#[cfg(target_arch = "aarch64")]
#[arcane]
fn wiener_hor_16bpc(
    _token: Arm64,
    tmp: &[u16; TMP_LEN],
    hor: &mut [u16; TMP_LEN],
    w: usize,
    h: usize,
    taps: &[i16; 8],
    bitdepth: i32,
) {
    let round_bits_h = 3 + (bitdepth == 12) as i32 * 2;
    let bias = (1 << (bitdepth + 6)) + (1 << (round_bits_h - 1));
    let clip_max = (1 << (bitdepth + 1 + 7 - round_bits_h)) - 1;
    let vbias = vdupq_n_s32(bias);
    let vsh = vdupq_n_s32(-round_bits_h);
    let vmax = vdupq_n_s32(clip_max);
    let vzero = vdupq_n_s32(0);
    let t: [int16x8_t; 7] = core::array::from_fn(|k| vdupq_n_s16(taps[k]));

    for row in 0..h + 6 {
        let base = row * S;
        let src = &tmp[base..base + w + 6];
        let dst = &mut hor[base..base + w];
        let mut x = 0;
        while x + 8 <= w {
            let mut lo = vbias;
            let mut hi = vbias;
            for k in 0..7 {
                // Pixels are <= 4095, so the u16 -> i16 reinterpret is exact.
                let v = safe_simd::vld1q_u16(src[x + k..][..8].try_into().unwrap());
                let v16 = vreinterpretq_s16_u16(v);
                lo = vmlal_s16(lo, vget_low_s16(v16), vget_low_s16(t[k]));
                hi = vmlal_high_s16(hi, v16, t[k]);
            }
            let lo = vminq_s32(vmaxq_s32(vshlq_s32(lo, vsh), vzero), vmax);
            let hi = vminq_s32(vmaxq_s32(vshlq_s32(hi, vsh), vzero), vmax);
            let packed = vcombine_u16(vmovn_u32(vreinterpretq_u32_s32(lo)), vmovn_u32(vreinterpretq_u32_s32(hi)));
            safe_simd::vst1q_u16((&mut dst[x..x + 8]).try_into().unwrap(), packed);
            x += 8;
        }
        while x < w {
            let mut sum = bias;
            for k in 0..7 {
                sum += src[x + k] as i32 * taps[k] as i32;
            }
            dst[x] = iclip(sum >> round_bits_h, 0, clip_max) as u16;
            x += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[arcane]
fn wiener_ver_16bpc(
    _token: Arm64,
    hor: &[u16; TMP_LEN],
    p: PicOffset,
    w: usize,
    h: usize,
    taps: &[i16; 8],
    bitdepth: i32,
    bitdepth_max: i32,
) {
    let round_bits_v = 11 - (bitdepth == 12) as i32 * 2;
    let bias = -(1 << (bitdepth + round_bits_v - 1)) + (1 << (round_bits_v - 1));
    let vbias = vdupq_n_s32(bias);
    let vsh = vdupq_n_s32(-round_bits_v);
    let vmax = vdupq_n_s32(bitdepth_max);
    let vzero = vdupq_n_s32(0);
    let t: [int16x8_t; 7] = core::array::from_fn(|k| vdupq_n_s16(taps[k]));
    let stride = p.pixel_stride::<BitDepth16>();

    for j in 0..h {
        let mut dst = (p + (j as isize * stride)).slice_mut::<BitDepth16>(w);
        let mut x = 0;
        while x + 8 <= w {
            let mut lo = vbias;
            let mut hi = vbias;
            for k in 0..7 {
                // `hor` is clipped to <= 32767, so the reinterpret stays positive.
                let v = safe_simd::vld1q_u16(hor[(j + k) * S + x..][..8].try_into().unwrap());
                let v16 = vreinterpretq_s16_u16(v);
                lo = vmlal_s16(lo, vget_low_s16(v16), vget_low_s16(t[k]));
                hi = vmlal_high_s16(hi, v16, t[k]);
            }
            let lo = vminq_s32(vmaxq_s32(vshlq_s32(lo, vsh), vzero), vmax);
            let hi = vminq_s32(vmaxq_s32(vshlq_s32(hi, vsh), vzero), vmax);
            let packed = vcombine_u16(vmovn_u32(vreinterpretq_u32_s32(lo)), vmovn_u32(vreinterpretq_u32_s32(hi)));
            safe_simd::vst1q_u16((&mut dst[x..x + 8]).try_into().unwrap(), packed);
            x += 8;
        }
        while x < w {
            let mut sum = bias;
            for k in 0..7 {
                sum += hor[(j + k) * S + x] as i32 * taps[k] as i32;
            }
            dst[x] = iclip(sum >> round_bits_v, 0, bitdepth_max) as u16;
            x += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn wiener_16bpc(
    token: Arm64,
    p: PicOffset,
    left: &[LeftPixelRow<u16>],
    lpf: &DisjointMut<AlignedVec64<u8>>,
    lpf_off: isize,
    w: usize,
    h: usize,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    bitdepth_max: i32,
) {
    let mut tmp = [0u16; TMP_LEN];
    padding::<BitDepth16>(&mut tmp, p, left, lpf, lpf_off, w, h, edges);
    let mut hor = [0u16; TMP_LEN];
    let bitdepth = if bitdepth_max == 1023 { 10 } else { 12 };
    wiener_hor_16bpc(token, &tmp, &mut hor, w, h, &params.filter[0], bitdepth);
    wiener_ver_16bpc(token, &hor, p, w, h, &params.filter[1], bitdepth, bitdepth_max);
}

// ============================================================================
// SELF-GUIDED: BOX SUMS
// ============================================================================
//
// Index contract, derived from `boxsum{3,5}` in `src/looprestoration.rs` with
// `bw = w + 6`, `bh = h + 6`, for every output row `r` in `1..=bh - 4`:
//
//   boxsum3: out[r][x] = sum over dy in 0..3, dx in -1..=1 of src[r + dy][x + dx]
//   boxsum5: out[r][x] = sum over dy in -1..4, dx in -2..=2 of src[r + dy][x + dx]
//
// for `x` in `2..bw - 2`. Positions outside that rectangle are never read by
// the `a`/`b` loop or the neighbour pass, which is why nothing here has to
// reproduce the reference's zero fill outside it.
//
// The reference computes the vertical sums into the full 106 KB array
// column-major and then slides horizontally in place. Here one row of vertical
// sums goes into a small row buffer and is slid immediately, which is
// row-major on the source and leaves the horizontal input un-aliased.

/// One fused box-sum row for 8bpc. `vs`/`vq` are scratch, at least `bw` long.
#[cfg(target_arch = "aarch64")]
#[rite]
fn box_row_8bpc<const N: usize>(
    _token: Arm64,
    src: &[u8; TMP_LEN],
    r: usize,
    bw: usize,
    vs: &mut [u16; S],
    vq: &mut [u32; S],
    out_sum: &mut [i16],
    out_sq: &mut [i32],
) {
    // Vertical: N rows starting at `r - (N == 5) as usize`.
    let top = r - (N == 5) as usize;
    let mut x = 0;
    while x + 8 <= bw {
        let mut s = vdupq_n_u16(0);
        let mut q0 = vdupq_n_u32(0);
        let mut q1 = vdupq_n_u32(0);
        for dy in 0..N {
            let v = safe_simd::vld1_u8(src[(top + dy) * S + x..][..8].try_into().unwrap());
            let v16 = vmovl_u8(v);
            s = vaddq_u16(s, v16);
            q0 = vmlal_u16(q0, vget_low_u16(v16), vget_low_u16(v16));
            q1 = vmlal_high_u16(q1, v16, v16);
        }
        safe_simd::vst1q_u16((&mut vs[x..x + 8]).try_into().unwrap(), s);
        safe_simd::vst1q_u32((&mut vq[x..x + 4]).try_into().unwrap(), q0);
        safe_simd::vst1q_u32((&mut vq[x + 4..x + 8]).try_into().unwrap(), q1);
        x += 8;
    }
    while x < bw {
        let mut s = 0u16;
        let mut q = 0u32;
        for dy in 0..N {
            let v = src[(top + dy) * S + x] as u16;
            s += v;
            q += v as u32 * v as u32;
        }
        vs[x] = s;
        vq[x] = q;
        x += 1;
    }

    // Horizontal: N-wide centred slide over the row buffer.
    let half = N / 2;
    let mut x = 2;
    while x + 8 <= bw - 2 {
        let mut s = vdupq_n_u16(0);
        let mut q0 = vdupq_n_u32(0);
        let mut q1 = vdupq_n_u32(0);
        for dx in 0..N {
            s = vaddq_u16(s, safe_simd::vld1q_u16(vs[x + dx - half..][..8].try_into().unwrap()));
            q0 = vaddq_u32(q0, safe_simd::vld1q_u32(vq[x + dx - half..][..4].try_into().unwrap()));
            q1 = vaddq_u32(q1, safe_simd::vld1q_u32(vq[x + dx - half + 4..][..4].try_into().unwrap()));
        }
        safe_simd::vst1q_s16((&mut out_sum[x..x + 8]).try_into().unwrap(), vreinterpretq_s16_u16(s));
        safe_simd::vst1q_s32((&mut out_sq[x..x + 4]).try_into().unwrap(), vreinterpretq_s32_u32(q0));
        safe_simd::vst1q_s32((&mut out_sq[x + 4..x + 8]).try_into().unwrap(), vreinterpretq_s32_u32(q1));
        x += 8;
    }
    while x < bw - 2 {
        let mut s = 0u16;
        let mut q = 0u32;
        for dx in 0..N {
            s = s.wrapping_add(vs[x + dx - half]);
            q = q.wrapping_add(vq[x + dx - half]);
        }
        out_sum[x] = s as i16;
        out_sq[x] = q as i32;
        x += 1;
    }
}

/// One fused box-sum row for 16bpc. Row sums exceed `u16` after the horizontal
/// slide (5 * 5 * 4095 = 102,375), so the row buffer is `u32`.
#[cfg(target_arch = "aarch64")]
#[rite]
fn box_row_16bpc<const N: usize>(
    _token: Arm64,
    src: &[u16; TMP_LEN],
    r: usize,
    bw: usize,
    vs: &mut [u32; S],
    vq: &mut [u32; S],
    out_sum: &mut [i32],
    out_sq: &mut [i32],
) {
    let top = r - (N == 5) as usize;
    let mut x = 0;
    while x + 8 <= bw {
        let mut s0 = vdupq_n_u32(0);
        let mut s1 = vdupq_n_u32(0);
        let mut q0 = vdupq_n_u32(0);
        let mut q1 = vdupq_n_u32(0);
        for dy in 0..N {
            let v = safe_simd::vld1q_u16(src[(top + dy) * S + x..][..8].try_into().unwrap());
            s0 = vaddw_u16(s0, vget_low_u16(v));
            s1 = vaddw_high_u16(s1, v);
            q0 = vmlal_u16(q0, vget_low_u16(v), vget_low_u16(v));
            q1 = vmlal_high_u16(q1, v, v);
        }
        safe_simd::vst1q_u32((&mut vs[x..x + 4]).try_into().unwrap(), s0);
        safe_simd::vst1q_u32((&mut vs[x + 4..x + 8]).try_into().unwrap(), s1);
        safe_simd::vst1q_u32((&mut vq[x..x + 4]).try_into().unwrap(), q0);
        safe_simd::vst1q_u32((&mut vq[x + 4..x + 8]).try_into().unwrap(), q1);
        x += 8;
    }
    while x < bw {
        let mut s = 0u32;
        let mut q = 0u32;
        for dy in 0..N {
            let v = src[(top + dy) * S + x] as u32;
            s += v;
            q += v * v;
        }
        vs[x] = s;
        vq[x] = q;
        x += 1;
    }

    let half = N / 2;
    let mut x = 2;
    while x + 4 <= bw - 2 {
        let mut s = vdupq_n_u32(0);
        let mut q = vdupq_n_u32(0);
        for dx in 0..N {
            s = vaddq_u32(s, safe_simd::vld1q_u32(vs[x + dx - half..][..4].try_into().unwrap()));
            q = vaddq_u32(q, safe_simd::vld1q_u32(vq[x + dx - half..][..4].try_into().unwrap()));
        }
        safe_simd::vst1q_s32((&mut out_sum[x..x + 4]).try_into().unwrap(), vreinterpretq_s32_u32(s));
        safe_simd::vst1q_s32((&mut out_sq[x..x + 4]).try_into().unwrap(), vreinterpretq_s32_u32(q));
        x += 4;
    }
    while x < bw - 2 {
        let mut s = 0u32;
        let mut q = 0u32;
        for dx in 0..N {
            s = s.wrapping_add(vs[x + dx - half]);
            q = q.wrapping_add(vq[x + dx - half]);
        }
        out_sum[x] = s as i32;
        out_sq[x] = q as i32;
        x += 1;
    }
}

// ============================================================================
// SELF-GUIDED: THE sgr_x_by_x LOOP
// ============================================================================

/// 256-entry `dav1d_sgr_x_by_x` gather for 16 lanes.
///
/// `vqtbl4q_u8` addresses 64 bytes and returns 0 outside that range; the
/// wrapping `vsubq_u8` moves each quarter's window over the index, so exactly
/// one of the four contributes per lane. Callers must have clamped the index
/// to 255 already (`vminq_u32` before the narrow), or a lane above 255 would
/// silently pick 0 instead of the reference's `min(z, 255)`.
#[cfg(target_arch = "aarch64")]
#[rite]
fn sgr_lut16(_token: Arm64, idx: uint8x16_t) -> uint8x16_t {
    let t: &[u8; 256] = &dav1d_sgr_x_by_x;
    let q0 = uint8x16x4_t(
        safe_simd::vld1q_u8(t[0..16].try_into().unwrap()),
        safe_simd::vld1q_u8(t[16..32].try_into().unwrap()),
        safe_simd::vld1q_u8(t[32..48].try_into().unwrap()),
        safe_simd::vld1q_u8(t[48..64].try_into().unwrap()),
    );
    let q1 = uint8x16x4_t(
        safe_simd::vld1q_u8(t[64..80].try_into().unwrap()),
        safe_simd::vld1q_u8(t[80..96].try_into().unwrap()),
        safe_simd::vld1q_u8(t[96..112].try_into().unwrap()),
        safe_simd::vld1q_u8(t[112..128].try_into().unwrap()),
    );
    let q2 = uint8x16x4_t(
        safe_simd::vld1q_u8(t[128..144].try_into().unwrap()),
        safe_simd::vld1q_u8(t[144..160].try_into().unwrap()),
        safe_simd::vld1q_u8(t[160..176].try_into().unwrap()),
        safe_simd::vld1q_u8(t[176..192].try_into().unwrap()),
    );
    let q3 = uint8x16x4_t(
        safe_simd::vld1q_u8(t[192..208].try_into().unwrap()),
        safe_simd::vld1q_u8(t[208..224].try_into().unwrap()),
        safe_simd::vld1q_u8(t[224..240].try_into().unwrap()),
        safe_simd::vld1q_u8(t[240..256].try_into().unwrap()),
    );
    let r0 = vqtbl4q_u8(q0, idx);
    let r1 = vqtbl4q_u8(q1, vsubq_u8(idx, vdupq_n_u8(64)));
    let r2 = vqtbl4q_u8(q2, vsubq_u8(idx, vdupq_n_u8(128)));
    let r3 = vqtbl4q_u8(q3, vsubq_u8(idx, vdupq_n_u8(192)));
    vorrq_u8(vorrq_u8(r0, r1), vorrq_u8(r2, r3))
}

/// `z = min((p * s + (1 << 19)) >> 20, 255)` for one 4-lane group.
#[cfg(target_arch = "aarch64")]
#[rite]
fn sgr_z(_token: Arm64, a: int32x4_t, b: int32x4_t, n: i32, s: u32) -> uint32x4_t {
    let p = vmaxq_s32(vsubq_s32(vmulq_n_s32(a, n), vmulq_s32(b, b)), vdupq_n_s32(0));
    let p = vreinterpretq_u32_s32(p);
    let z = vshrq_n_u32::<20>(vaddq_u32(vmulq_u32(p, vdupq_n_u32(s)), vdupq_n_u32(1 << 19)));
    vminq_u32(z, vdupq_n_u32(255))
}

/// `aa = (x * b * one_by_x + (1 << 11)) >> 12`, in wrapping u32 exactly like
/// the reference's `c_uint` arithmetic (it genuinely overflows at 12bpc).
#[cfg(target_arch = "aarch64")]
#[rite]
fn sgr_aa(_token: Arm64, x: uint32x4_t, b: int32x4_t, one_by_x: u32) -> int32x4_t {
    let prod = vmulq_u32(vmulq_u32(x, vreinterpretq_u32_s32(b)), vdupq_n_u32(one_by_x));
    vreinterpretq_s32_u32(vshrq_n_u32::<12>(vaddq_u32(prod, vdupq_n_u32(1 << 11))))
}

/// Pack four `z` groups into the 16 byte indices `sgr_lut16` wants.
#[cfg(target_arch = "aarch64")]
#[rite]
fn sgr_pack_idx(_token: Arm64, z: [uint32x4_t; 4]) -> uint8x16_t {
    let a = vcombine_u16(vmovn_u32(z[0]), vmovn_u32(z[1]));
    let b = vcombine_u16(vmovn_u32(z[2]), vmovn_u32(z[3]));
    vcombine_u8(vmovn_u16(a), vmovn_u16(b))
}

/// Unpack the 16 looked-up `x` bytes back into four u32 groups.
#[cfg(target_arch = "aarch64")]
#[rite]
fn sgr_unpack_x(_token: Arm64, x: uint8x16_t) -> [uint32x4_t; 4] {
    let lo = vmovl_u8(vget_low_u8(x));
    let hi = vmovl_high_u8(x);
    [
        vmovl_u16(vget_low_u16(lo)),
        vmovl_high_u16(lo),
        vmovl_u16(vget_low_u16(hi)),
        vmovl_high_u16(hi),
    ]
}

// ============================================================================
// SELF-GUIDED: 8BPC
// ============================================================================

/// The `a`/`b` loop for 8bpc (`bitdepth_min_8 == 0`, so no pre-shift).
#[cfg(target_arch = "aarch64")]
#[arcane]
fn sgr_ab_8bpc(
    token: Arm64,
    sumsq: &mut [i32; BOX_LEN],
    sum: &mut [i16; BOX_LEN],
    w: usize,
    h: usize,
    n: i32,
    s: u32,
    one_by_x: u32,
    step: usize,
) {
    let cols = w + 2;
    let mut row = 0;
    while row < h + 2 {
        let base = (row + 1) * S + 2;
        let mut i = 0;
        while i + 16 <= cols {
            let mut z = [vdupq_n_u32(0); 4];
            let mut bs = [vdupq_n_s32(0); 4];
            for g in 0..4 {
                let a = safe_simd::vld1q_s32(sumsq[base + i + g * 4..][..4].try_into().unwrap());
                let b = vmovl_s16(safe_simd::vld1_s16(sum[base + i + g * 4..][..4].try_into().unwrap()));
                bs[g] = b;
                z[g] = sgr_z(token, a, b, n, s);
            }
            let xs = sgr_unpack_x(token, sgr_lut16(token, sgr_pack_idx(token, z)));
            for g in 0..4 {
                let aa = sgr_aa(token, xs[g], bs[g], one_by_x);
                safe_simd::vst1q_s32((&mut sumsq[base + i + g * 4..][..4]).try_into().unwrap(), aa);
                safe_simd::vst1_s16(
                    (&mut sum[base + i + g * 4..][..4]).try_into().unwrap(),
                    vmovn_s32(vreinterpretq_s32_u32(xs[g])),
                );
            }
            i += 16;
        }
        while i < cols {
            let idx = base + i;
            let a_val = sumsq[idx];
            let b_val = sum[idx] as i32;
            let p = cmp::max(a_val * n - b_val * b_val, 0) as u32;
            let z = (p.wrapping_mul(s).wrapping_add(1 << 19)) >> 20;
            let x = dav1d_sgr_x_by_x[cmp::min(z, 255) as usize] as u32;
            sumsq[idx] = ((x.wrapping_mul(b_val as u32).wrapping_mul(one_by_x)).wrapping_add(1 << 11) >> 12) as i32;
            sum[idx] = x as i16;
            i += 1;
        }
        row += step;
    }
}

/// `6 * (up + dn) + 5 * (upl + upr + dnl + dnr)` over an i32 plane, 4 lanes.
#[cfg(target_arch = "aarch64")]
#[rite]
fn six_i32(_token: Arm64, p: &[i32], i: usize) -> int32x4_t {
    let up = safe_simd::vld1q_s32(p[i - S..][..4].try_into().unwrap());
    let dn = safe_simd::vld1q_s32(p[i + S..][..4].try_into().unwrap());
    let upl = safe_simd::vld1q_s32(p[i - S - 1..][..4].try_into().unwrap());
    let upr = safe_simd::vld1q_s32(p[i - S + 1..][..4].try_into().unwrap());
    let dnl = safe_simd::vld1q_s32(p[i + S - 1..][..4].try_into().unwrap());
    let dnr = safe_simd::vld1q_s32(p[i + S + 1..][..4].try_into().unwrap());
    vmlaq_n_s32(
        vmulq_n_s32(vaddq_s32(up, dn), 6),
        vaddq_s32(vaddq_s32(upl, upr), vaddq_s32(dnl, dnr)),
        5,
    )
}

/// `6 * c + 5 * (l + r)` over an i32 plane, 4 lanes.
#[cfg(target_arch = "aarch64")]
#[rite]
fn mid_i32(_token: Arm64, p: &[i32], i: usize) -> int32x4_t {
    let c = safe_simd::vld1q_s32(p[i..][..4].try_into().unwrap());
    let l = safe_simd::vld1q_s32(p[i - 1..][..4].try_into().unwrap());
    let r = safe_simd::vld1q_s32(p[i + 1..][..4].try_into().unwrap());
    vmlaq_n_s32(vmulq_n_s32(c, 6), vaddq_s32(l, r), 5)
}

/// `4 * (c + l + r + up + dn) + 3 * corners` over an i32 plane, 4 lanes.
#[cfg(target_arch = "aarch64")]
#[rite]
fn eight_i32(_token: Arm64, p: &[i32], i: usize) -> int32x4_t {
    let c = safe_simd::vld1q_s32(p[i..][..4].try_into().unwrap());
    let l = safe_simd::vld1q_s32(p[i - 1..][..4].try_into().unwrap());
    let r = safe_simd::vld1q_s32(p[i + 1..][..4].try_into().unwrap());
    let up = safe_simd::vld1q_s32(p[i - S..][..4].try_into().unwrap());
    let dn = safe_simd::vld1q_s32(p[i + S..][..4].try_into().unwrap());
    let upl = safe_simd::vld1q_s32(p[i - S - 1..][..4].try_into().unwrap());
    let upr = safe_simd::vld1q_s32(p[i - S + 1..][..4].try_into().unwrap());
    let dnl = safe_simd::vld1q_s32(p[i + S - 1..][..4].try_into().unwrap());
    let dnr = safe_simd::vld1q_s32(p[i + S + 1..][..4].try_into().unwrap());
    vmlaq_n_s32(
        vmulq_n_s32(vaddq_s32(vaddq_s32(vaddq_s32(c, l), vaddq_s32(r, up)), dn), 4),
        vaddq_s32(vaddq_s32(upl, upr), vaddq_s32(dnl, dnr)),
        3,
    )
}

/// Same shapes over the i16 `sum` plane, widened to i32 on load.
#[cfg(target_arch = "aarch64")]
#[rite]
fn ld4_i16(_token: Arm64, p: &[i16], i: usize) -> int32x4_t {
    vmovl_s16(safe_simd::vld1_s16(p[i..][..4].try_into().unwrap()))
}

#[cfg(target_arch = "aarch64")]
#[rite]
fn six_i16(token: Arm64, p: &[i16], i: usize) -> int32x4_t {
    let up = ld4_i16(token, p, i - S);
    let dn = ld4_i16(token, p, i + S);
    let upl = ld4_i16(token, p, i - S - 1);
    let upr = ld4_i16(token, p, i - S + 1);
    let dnl = ld4_i16(token, p, i + S - 1);
    let dnr = ld4_i16(token, p, i + S + 1);
    vmlaq_n_s32(
        vmulq_n_s32(vaddq_s32(up, dn), 6),
        vaddq_s32(vaddq_s32(upl, upr), vaddq_s32(dnl, dnr)),
        5,
    )
}

#[cfg(target_arch = "aarch64")]
#[rite]
fn mid_i16(token: Arm64, p: &[i16], i: usize) -> int32x4_t {
    let c = ld4_i16(token, p, i);
    let l = ld4_i16(token, p, i - 1);
    let r = ld4_i16(token, p, i + 1);
    vmlaq_n_s32(vmulq_n_s32(c, 6), vaddq_s32(l, r), 5)
}

#[cfg(target_arch = "aarch64")]
#[rite]
fn eight_i16(token: Arm64, p: &[i16], i: usize) -> int32x4_t {
    let c = ld4_i16(token, p, i);
    let l = ld4_i16(token, p, i - 1);
    let r = ld4_i16(token, p, i + 1);
    let up = ld4_i16(token, p, i - S);
    let dn = ld4_i16(token, p, i + S);
    let upl = ld4_i16(token, p, i - S - 1);
    let upr = ld4_i16(token, p, i - S + 1);
    let dnl = ld4_i16(token, p, i + S - 1);
    let dnr = ld4_i16(token, p, i + S + 1);
    vmlaq_n_s32(
        vmulq_n_s32(vaddq_s32(vaddq_s32(vaddq_s32(c, l), vaddq_s32(r, up)), dn), 4),
        vaddq_s32(vaddq_s32(upl, upr), vaddq_s32(dnl, dnr)),
        3,
    )
}

#[cfg(target_arch = "aarch64")]
fn selfguided_8bpc(
    token: Arm64,
    dst: &mut [i16; DST_LEN],
    src: &[u8; TMP_LEN],
    w: usize,
    h: usize,
    n: i32,
    s: u32,
    sumsq: &mut [i32; BOX_LEN],
    sum: &mut [i16; BOX_LEN],
) {
    let one_by_x: u32 = if n == 25 { 164 } else { 455 };
    let step = if n == 25 { 2 } else { 1 };
    let (bw, bh) = (w + 6, h + 6);

    boxsum_8bpc(token, sumsq, sum, src, bw, bh, n);
    sgr_ab_8bpc(token, sumsq, sum, w, h, n, s, one_by_x, step);
    sgr_out_8bpc(token, dst, src, sumsq, sum, w, h, n);
}

#[cfg(target_arch = "aarch64")]
#[arcane]
fn boxsum_8bpc(
    token: Arm64,
    sumsq: &mut [i32; BOX_LEN],
    sum: &mut [i16; BOX_LEN],
    src: &[u8; TMP_LEN],
    bw: usize,
    bh: usize,
    n: i32,
) {
    let mut vs = [0u16; S];
    let mut vq = [0u32; S];
    for r in 1..=bh - 4 {
        let (os, oq) = (&mut sum[r * S..r * S + bw], &mut sumsq[r * S..r * S + bw]);
        if n == 25 {
            box_row_8bpc::<5>(token, src, r, bw, &mut vs, &mut vq, os, oq);
        } else {
            box_row_8bpc::<3>(token, src, r, bw, &mut vs, &mut vq, os, oq);
        }
    }
}

/// Neighbour-weighted output pass, 8bpc.
#[cfg(target_arch = "aarch64")]
#[arcane]
fn sgr_out_8bpc(
    token: Arm64,
    dst: &mut [i16; DST_LEN],
    src: &[u8; TMP_LEN],
    sumsq: &[i32; BOX_LEN],
    sum: &[i16; BOX_LEN],
    w: usize,
    h: usize,
    n: i32,
) {
    let base = 2 * S + 3;
    let src_base = 3 * S + 3;

    // `(b - a * src + rnd) >> sh`, narrowed with a TRUNCATING `vmovn_s32`
    // because the reference's `.as_()` into `BD::Coef` truncates.
    macro_rules! emit {
        ($bv:expr, $av:expr, $sidx:expr, $didx:expr, $rnd:expr, $sh:expr) => {{
            let px = vmovl_u16(vget_low_u16(vmovl_u8(safe_simd::vld1_u8(
                src[$sidx..][..8].try_into().unwrap(),
            ))));
            let v = vaddq_s32(
                vsubq_s32($bv, vmulq_s32($av, vreinterpretq_s32_u32(px))),
                vdupq_n_s32($rnd),
            );
            safe_simd::vst1_s16(
                (&mut dst[$didx..][..4]).try_into().unwrap(),
                vmovn_s32(vshrq_n_s32::<$sh>(v)),
            );
        }};
    }

    if n == 25 {
        let mut j = 0;
        while j + 1 < h {
            for phase in 0..2 {
                let rowa = base + (j + phase) * S;
                let sidx0 = src_base + (j + phase) * S;
                let didx0 = (j + phase) * MAXW;
                let mut i = 0;
                while i + 4 <= w {
                    let (bv, av) = if phase == 0 {
                        (six_i32(token, sumsq, rowa + i), six_i16(token, sum, rowa + i))
                    } else {
                        (mid_i32(token, sumsq, rowa + i), mid_i16(token, sum, rowa + i))
                    };
                    if phase == 0 {
                        emit!(bv, av, sidx0 + i, didx0 + i, 1 << 8, 9);
                    } else {
                        emit!(bv, av, sidx0 + i, didx0 + i, 1 << 7, 8);
                    }
                    i += 4;
                }
                while i < w {
                    let (b, a) = if phase == 0 {
                        (six_s(sumsq, rowa + i), six_s16(sum, rowa + i))
                    } else {
                        (mid_s(sumsq, rowa + i), mid_s16(sum, rowa + i))
                    };
                    let px = src[sidx0 + i] as i32;
                    dst[didx0 + i] = if phase == 0 {
                        ((b - a * px + (1 << 8)) >> 9) as i16
                    } else {
                        ((b - a * px + (1 << 7)) >> 8) as i16
                    };
                    i += 1;
                }
            }
            j += 2;
        }
        if j + 1 == h {
            let rowa = base + j * S;
            let sidx0 = src_base + j * S;
            let didx0 = j * MAXW;
            let mut i = 0;
            while i + 4 <= w {
                let bv = six_i32(token, sumsq, rowa + i);
                let av = six_i16(token, sum, rowa + i);
                emit!(bv, av, sidx0 + i, didx0 + i, 1 << 8, 9);
                i += 4;
            }
            while i < w {
                let b = six_s(sumsq, rowa + i);
                let a = six_s16(sum, rowa + i);
                dst[didx0 + i] = ((b - a * src[sidx0 + i] as i32 + (1 << 8)) >> 9) as i16;
                i += 1;
            }
        }
    } else {
        for j in 0..h {
            let rowa = base + j * S;
            let sidx0 = src_base + j * S;
            let didx0 = j * MAXW;
            let mut i = 0;
            while i + 4 <= w {
                let bv = eight_i32(token, sumsq, rowa + i);
                let av = eight_i16(token, sum, rowa + i);
                emit!(bv, av, sidx0 + i, didx0 + i, 1 << 8, 9);
                i += 4;
            }
            while i < w {
                let b = eight_s(sumsq, rowa + i);
                let a = eight_s16(sum, rowa + i);
                dst[didx0 + i] = ((b - a * src[sidx0 + i] as i32 + (1 << 8)) >> 9) as i16;
                i += 1;
            }
        }
    }
}

// Scalar tail helpers, kept next to their vector twins so the two forms can be
// read against each other.
#[cfg(target_arch = "aarch64")]
fn six_s(p: &[i32], i: usize) -> i32 {
    (p[i - S] + p[i + S]) * 6 + (p[i - S - 1] + p[i - S + 1] + p[i + S - 1] + p[i + S + 1]) * 5
}
#[cfg(target_arch = "aarch64")]
fn mid_s(p: &[i32], i: usize) -> i32 {
    p[i] * 6 + (p[i - 1] + p[i + 1]) * 5
}
#[cfg(target_arch = "aarch64")]
fn eight_s(p: &[i32], i: usize) -> i32 {
    (p[i] + p[i - 1] + p[i + 1] + p[i - S] + p[i + S]) * 4
        + (p[i - S - 1] + p[i - S + 1] + p[i + S - 1] + p[i + S + 1]) * 3
}
#[cfg(target_arch = "aarch64")]
fn six_s16(p: &[i16], i: usize) -> i32 {
    (p[i - S] as i32 + p[i + S] as i32) * 6
        + (p[i - S - 1] as i32 + p[i - S + 1] as i32 + p[i + S - 1] as i32 + p[i + S + 1] as i32) * 5
}
#[cfg(target_arch = "aarch64")]
fn mid_s16(p: &[i16], i: usize) -> i32 {
    p[i] as i32 * 6 + (p[i - 1] as i32 + p[i + 1] as i32) * 5
}
#[cfg(target_arch = "aarch64")]
fn eight_s16(p: &[i16], i: usize) -> i32 {
    (p[i] as i32 + p[i - 1] as i32 + p[i + 1] as i32 + p[i - S] as i32 + p[i + S] as i32) * 4
        + (p[i - S - 1] as i32 + p[i - S + 1] as i32 + p[i + S - 1] as i32 + p[i + S + 1] as i32) * 3
}

/// Blend one or two self-guided outputs into the picture, 8bpc.
#[cfg(target_arch = "aarch64")]
#[arcane]
fn sgr_apply_8bpc(
    _token: Arm64,
    p: PicOffset,
    w: usize,
    h: usize,
    d0: &[i16; DST_LEN],
    w0: i32,
    d1: Option<&[i16; DST_LEN]>,
    w1: i32,
) {
    let stride = p.pixel_stride::<BitDepth8>();
    for j in 0..h {
        let mut row = (p + (j as isize * stride)).slice_mut::<BitDepth8>(w);
        let mut i = 0;
        while i + 8 <= w {
            let a = safe_simd::vld1q_s16(d0[j * MAXW + i..][..8].try_into().unwrap());
            let mut lo = vmulq_n_s32(vmovl_s16(vget_low_s16(a)), w0);
            let mut hi = vmulq_n_s32(vmovl_high_s16(a), w0);
            if let Some(d1) = d1 {
                let b = safe_simd::vld1q_s16(d1[j * MAXW + i..][..8].try_into().unwrap());
                lo = vmlaq_n_s32(lo, vmovl_s16(vget_low_s16(b)), w1);
                hi = vmlaq_n_s32(hi, vmovl_high_s16(b), w1);
            }
            let lo = vshrq_n_s32::<11>(vaddq_s32(lo, vdupq_n_s32(1 << 10)));
            let hi = vshrq_n_s32::<11>(vaddq_s32(hi, vdupq_n_s32(1 << 10)));
            let add = vcombine_s16(vmovn_s32(lo), vmovn_s32(hi));
            let px = vreinterpretq_s16_u16(vmovl_u8(safe_simd::vld1_u8(row[i..][..8].try_into().unwrap())));
            safe_simd::vst1_u8(
                (&mut row[i..i + 8]).try_into().unwrap(),
                vqmovun_s16(vaddq_s16(px, add)),
            );
            i += 8;
        }
        while i < w {
            let mut v = w0 * d0[j * MAXW + i] as i32;
            if let Some(d1) = d1 {
                v += w1 * d1[j * MAXW + i] as i32;
            }
            row[i] = iclip(row[i] as i32 + ((v + (1 << 10)) >> 11), 0, 255) as u8;
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn sgr_8bpc(
    token: Arm64,
    p: PicOffset,
    left: &[LeftPixelRow<u8>],
    lpf: &DisjointMut<AlignedVec64<u8>>,
    lpf_off: isize,
    w: usize,
    h: usize,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    variant: usize,
) {
    let mut tmp = [0u8; TMP_LEN];
    padding::<BitDepth8>(&mut tmp, p, left, lpf, lpf_off, w, h, edges);
    let sgr = params.sgr();
    let mut sumsq = [0i32; BOX_LEN];
    let mut sum = [0i16; BOX_LEN];
    let mut d0 = [0i16; DST_LEN];

    match variant {
        2 => {
            selfguided_8bpc(token, &mut d0, &tmp, w, h, 25, sgr.s0, &mut sumsq, &mut sum);
            sgr_apply_8bpc(token, p, w, h, &d0, sgr.w0 as i32, None, 0);
        }
        3 => {
            selfguided_8bpc(token, &mut d0, &tmp, w, h, 9, sgr.s1, &mut sumsq, &mut sum);
            sgr_apply_8bpc(token, p, w, h, &d0, sgr.w1 as i32, None, 0);
        }
        _ => {
            let mut d1 = [0i16; DST_LEN];
            selfguided_8bpc(token, &mut d0, &tmp, w, h, 25, sgr.s0, &mut sumsq, &mut sum);
            selfguided_8bpc(token, &mut d1, &tmp, w, h, 9, sgr.s1, &mut sumsq, &mut sum);
            sgr_apply_8bpc(token, p, w, h, &d0, sgr.w0 as i32, Some(&d1), sgr.w1 as i32);
        }
    }
}

// ============================================================================
// SELF-GUIDED: 16BPC
// ============================================================================

/// The `a`/`b` loop for 16bpc. `bitdepth_min_8` is 2 or 4, so `a` and `b` are
/// pre-shifted (with rounding) before `p` — but the UNSHIFTED `b` is what goes
/// into the `aa` update, exactly as the reference does.
#[cfg(target_arch = "aarch64")]
#[arcane]
fn sgr_ab_16bpc(
    token: Arm64,
    sumsq: &mut [i32; BOX_LEN],
    sum: &mut [i32; BOX_LEN],
    w: usize,
    h: usize,
    n: i32,
    s: u32,
    one_by_x: u32,
    step: usize,
    bdm8: i32,
) {
    let cols = w + 2;
    let va_rnd = vdupq_n_s32((1 << (2 * bdm8)) >> 1);
    let vb_rnd = vdupq_n_s32((1 << bdm8) >> 1);
    let va_sh = vdupq_n_s32(-2 * bdm8);
    let vb_sh = vdupq_n_s32(-bdm8);
    let mut row = 0;
    while row < h + 2 {
        let base = (row + 1) * S + 2;
        let mut i = 0;
        while i + 16 <= cols {
            let mut z = [vdupq_n_u32(0); 4];
            let mut braw = [vdupq_n_s32(0); 4];
            for g in 0..4 {
                let a_raw = safe_simd::vld1q_s32(sumsq[base + i + g * 4..][..4].try_into().unwrap());
                let b_raw = safe_simd::vld1q_s32(sum[base + i + g * 4..][..4].try_into().unwrap());
                braw[g] = b_raw;
                let a = vshlq_s32(vaddq_s32(a_raw, va_rnd), va_sh);
                let b = vshlq_s32(vaddq_s32(b_raw, vb_rnd), vb_sh);
                z[g] = sgr_z(token, a, b, n, s);
            }
            let xs = sgr_unpack_x(token, sgr_lut16(token, sgr_pack_idx(token, z)));
            for g in 0..4 {
                let aa = sgr_aa(token, xs[g], braw[g], one_by_x);
                safe_simd::vst1q_s32((&mut sumsq[base + i + g * 4..][..4]).try_into().unwrap(), aa);
                safe_simd::vst1q_s32(
                    (&mut sum[base + i + g * 4..][..4]).try_into().unwrap(),
                    vreinterpretq_s32_u32(xs[g]),
                );
            }
            i += 16;
        }
        while i < cols {
            let idx = base + i;
            let a_raw = sumsq[idx];
            let b_raw = sum[idx];
            let a = (a_raw + ((1 << (2 * bdm8)) >> 1)) >> (2 * bdm8);
            let b = (b_raw + ((1 << bdm8) >> 1)) >> bdm8;
            let p = cmp::max(a * n - b * b, 0) as u32;
            let z = (p.wrapping_mul(s).wrapping_add(1 << 19)) >> 20;
            let x = dav1d_sgr_x_by_x[cmp::min(z, 255) as usize] as u32;
            sumsq[idx] = ((x.wrapping_mul(b_raw as u32).wrapping_mul(one_by_x)).wrapping_add(1 << 11) >> 12) as i32;
            sum[idx] = x as i32;
            i += 1;
        }
        row += step;
    }
}

/// Neighbour-weighted output pass, 16bpc (both planes are i32).
#[cfg(target_arch = "aarch64")]
#[arcane]
fn sgr_out_16bpc(
    token: Arm64,
    dst: &mut [i32; DST_LEN],
    src: &[u16; TMP_LEN],
    sumsq: &[i32; BOX_LEN],
    sum: &[i32; BOX_LEN],
    w: usize,
    h: usize,
    n: i32,
) {
    let base = 2 * S + 3;
    let src_base = 3 * S + 3;

    macro_rules! emit {
        ($bv:expr, $av:expr, $sidx:expr, $didx:expr, $rnd:expr, $sh:expr) => {{
            let px = vreinterpretq_s32_u32(vmovl_u16(safe_simd::vld1_u16(
                src[$sidx..][..4].try_into().unwrap(),
            )));
            let v = vaddq_s32(vsubq_s32($bv, vmulq_s32($av, px)), vdupq_n_s32($rnd));
            safe_simd::vst1q_s32((&mut dst[$didx..][..4]).try_into().unwrap(), vshrq_n_s32::<$sh>(v));
        }};
    }

    if n == 25 {
        let mut j = 0;
        while j + 1 < h {
            for phase in 0..2 {
                let rowa = base + (j + phase) * S;
                let sidx0 = src_base + (j + phase) * S;
                let didx0 = (j + phase) * MAXW;
                let mut i = 0;
                while i + 4 <= w {
                    let (bv, av) = if phase == 0 {
                        (six_i32(token, sumsq, rowa + i), six_i32(token, sum, rowa + i))
                    } else {
                        (mid_i32(token, sumsq, rowa + i), mid_i32(token, sum, rowa + i))
                    };
                    if phase == 0 {
                        emit!(bv, av, sidx0 + i, didx0 + i, 1 << 8, 9);
                    } else {
                        emit!(bv, av, sidx0 + i, didx0 + i, 1 << 7, 8);
                    }
                    i += 4;
                }
                while i < w {
                    let (b, a) = if phase == 0 {
                        (six_s(sumsq, rowa + i), six_s(sum, rowa + i))
                    } else {
                        (mid_s(sumsq, rowa + i), mid_s(sum, rowa + i))
                    };
                    let px = src[sidx0 + i] as i32;
                    dst[didx0 + i] = if phase == 0 {
                        (b - a * px + (1 << 8)) >> 9
                    } else {
                        (b - a * px + (1 << 7)) >> 8
                    };
                    i += 1;
                }
            }
            j += 2;
        }
        if j + 1 == h {
            let rowa = base + j * S;
            let sidx0 = src_base + j * S;
            let didx0 = j * MAXW;
            let mut i = 0;
            while i + 4 <= w {
                let bv = six_i32(token, sumsq, rowa + i);
                let av = six_i32(token, sum, rowa + i);
                emit!(bv, av, sidx0 + i, didx0 + i, 1 << 8, 9);
                i += 4;
            }
            while i < w {
                let b = six_s(sumsq, rowa + i);
                let a = six_s(sum, rowa + i);
                dst[didx0 + i] = (b - a * src[sidx0 + i] as i32 + (1 << 8)) >> 9;
                i += 1;
            }
        }
    } else {
        for j in 0..h {
            let rowa = base + j * S;
            let sidx0 = src_base + j * S;
            let didx0 = j * MAXW;
            let mut i = 0;
            while i + 4 <= w {
                let bv = eight_i32(token, sumsq, rowa + i);
                let av = eight_i32(token, sum, rowa + i);
                emit!(bv, av, sidx0 + i, didx0 + i, 1 << 8, 9);
                i += 4;
            }
            while i < w {
                let b = eight_s(sumsq, rowa + i);
                let a = eight_s(sum, rowa + i);
                dst[didx0 + i] = (b - a * src[sidx0 + i] as i32 + (1 << 8)) >> 9;
                i += 1;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[arcane]
fn sgr_apply_16bpc(
    _token: Arm64,
    p: PicOffset,
    w: usize,
    h: usize,
    d0: &[i32; DST_LEN],
    w0: i32,
    d1: Option<&[i32; DST_LEN]>,
    w1: i32,
    bitdepth_max: i32,
) {
    let stride = p.pixel_stride::<BitDepth16>();
    let vmax = vdupq_n_s32(bitdepth_max);
    let vzero = vdupq_n_s32(0);
    for j in 0..h {
        let mut row = (p + (j as isize * stride)).slice_mut::<BitDepth16>(w);
        let mut i = 0;
        while i + 4 <= w {
            let a = safe_simd::vld1q_s32(d0[j * MAXW + i..][..4].try_into().unwrap());
            let mut v = vmulq_n_s32(a, w0);
            if let Some(d1) = d1 {
                let b = safe_simd::vld1q_s32(d1[j * MAXW + i..][..4].try_into().unwrap());
                v = vmlaq_n_s32(v, b, w1);
            }
            let add = vshrq_n_s32::<11>(vaddq_s32(v, vdupq_n_s32(1 << 10)));
            let px = vreinterpretq_s32_u32(vmovl_u16(safe_simd::vld1_u16(row[i..][..4].try_into().unwrap())));
            let out = vminq_s32(vmaxq_s32(vaddq_s32(px, add), vzero), vmax);
            safe_simd::vst1_u16(
                (&mut row[i..i + 4]).try_into().unwrap(),
                vmovn_u32(vreinterpretq_u32_s32(out)),
            );
            i += 4;
        }
        while i < w {
            let mut v = w0 * d0[j * MAXW + i];
            if let Some(d1) = d1 {
                v += w1 * d1[j * MAXW + i];
            }
            row[i] = iclip(row[i] as i32 + ((v + (1 << 10)) >> 11), 0, bitdepth_max) as u16;
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn selfguided_16bpc(
    token: Arm64,
    dst: &mut [i32; DST_LEN],
    src: &[u16; TMP_LEN],
    w: usize,
    h: usize,
    n: i32,
    s: u32,
    bdm8: i32,
    sumsq: &mut [i32; BOX_LEN],
    sum: &mut [i32; BOX_LEN],
) {
    let one_by_x: u32 = if n == 25 { 164 } else { 455 };
    let step = if n == 25 { 2 } else { 1 };
    let (bw, bh) = (w + 6, h + 6);

    boxsum_16bpc(token, sumsq, sum, src, bw, bh, n);
    sgr_ab_16bpc(token, sumsq, sum, w, h, n, s, one_by_x, step, bdm8);
    sgr_out_16bpc(token, dst, src, sumsq, sum, w, h, n);
}

#[cfg(target_arch = "aarch64")]
#[arcane]
fn boxsum_16bpc(
    token: Arm64,
    sumsq: &mut [i32; BOX_LEN],
    sum: &mut [i32; BOX_LEN],
    src: &[u16; TMP_LEN],
    bw: usize,
    bh: usize,
    n: i32,
) {
    let mut vs = [0u32; S];
    let mut vq = [0u32; S];
    for r in 1..=bh - 4 {
        let (os, oq) = (&mut sum[r * S..r * S + bw], &mut sumsq[r * S..r * S + bw]);
        if n == 25 {
            box_row_16bpc::<5>(token, src, r, bw, &mut vs, &mut vq, os, oq);
        } else {
            box_row_16bpc::<3>(token, src, r, bw, &mut vs, &mut vq, os, oq);
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn sgr_16bpc(
    token: Arm64,
    p: PicOffset,
    left: &[LeftPixelRow<u16>],
    lpf: &DisjointMut<AlignedVec64<u8>>,
    lpf_off: isize,
    w: usize,
    h: usize,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    variant: usize,
    bitdepth_max: i32,
) {
    let mut tmp = [0u16; TMP_LEN];
    padding::<BitDepth16>(&mut tmp, p, left, lpf, lpf_off, w, h, edges);
    let sgr = params.sgr();
    let bdm8 = if bitdepth_max == 1023 { 2 } else { 4 };
    let mut sumsq = [0i32; BOX_LEN];
    let mut sum = [0i32; BOX_LEN];
    let mut d0 = [0i32; DST_LEN];

    match variant {
        2 => {
            selfguided_16bpc(token, &mut d0, &tmp, w, h, 25, sgr.s0, bdm8, &mut sumsq, &mut sum);
            sgr_apply_16bpc(token, p, w, h, &d0, sgr.w0 as i32, None, 0, bitdepth_max);
        }
        3 => {
            selfguided_16bpc(token, &mut d0, &tmp, w, h, 9, sgr.s1, bdm8, &mut sumsq, &mut sum);
            sgr_apply_16bpc(token, p, w, h, &d0, sgr.w1 as i32, None, 0, bitdepth_max);
        }
        _ => {
            let mut d1 = [0i32; DST_LEN];
            selfguided_16bpc(token, &mut d0, &tmp, w, h, 25, sgr.s0, bdm8, &mut sumsq, &mut sum);
            selfguided_16bpc(token, &mut d1, &tmp, w, h, 9, sgr.s1, bdm8, &mut sumsq, &mut sum);
            sgr_apply_16bpc(
                token,
                p,
                w,
                h,
                &d0,
                sgr.w0 as i32,
                Some(&d1),
                sgr.w1 as i32,
                bitdepth_max,
            );
        }
    }
}

// ============================================================================
// DISPATCH
// ============================================================================

/// Route a loop-restoration call to the aarch64 NEON tier.
///
/// Returns `false` — leaving the caller to run `src/looprestoration.rs` — when
/// the `__ablate` measurement switch is set, or on the (unreachable on
/// aarch64) path where `Arm64::summon()` fails.
///
/// This is also where the `__ablate` activity counter records how much loop
/// restoration a bitstream actually asks for; the count is taken BEFORE the
/// ablation early-return, so it reflects what the bitstream wants rather than
/// what SIMD handled.
#[cfg(target_arch = "aarch64")]
pub fn lr_filter_dispatch<BD: BitDepth>(
    variant: usize,
    dst: PicOffset,
    left: &[LeftPixelRow<BD::Pixel>],
    lpf: &DisjointMut<AlignedVec64<u8>>,
    lpf_off: isize,
    w: c_int,
    h: c_int,
    params: &LooprestorationParams,
    edges: LrEdgeFlags,
    bd: BD,
) -> bool {
    use crate::include::common::bitdepth::BPC;
    use crate::src::safe_simd::pixel_access::reinterpret_slice;
    use archmage::SimdToken as _;

    crate::src::ablate::note(
        crate::src::ablate::Family::LoopRestoration,
        (w as i64 * h as i64).unsigned_abs(),
    );
    if crate::src::ablate::is_off(crate::src::ablate::Family::LoopRestoration) {
        return false;
    }

    let Some(token) = Arm64::summon() else {
        return false;
    };
    #[cfg(feature = "__lrvarcov")]
    {
        use std::sync::atomic::{AtomicU8, Ordering};
        static SEEN: [AtomicU8; 10] = [const { AtomicU8::new(0) }; 10];
        let cell = (BD::BPC == crate::include::common::bitdepth::BPC::BPC16) as usize * 5
            + variant.min(4);
        if SEEN[cell].swap(1, Ordering::Relaxed) == 0 {
            let name = ["wiener7", "wiener5", "sgr_5x5", "sgr_3x3", "sgr_mix"][variant.min(4)];
            let bpc = if cell >= 5 { "16bpc" } else { "8bpc" };
            eprintln!("LRVAR\t{bpc}\t{name}");
        }
    }
    let w = w as usize;
    let h = h as usize;
    let bd_c = bd.into_c();

    match BD::BPC {
        BPC::BPC8 => {
            let left: &[LeftPixelRow<u8>] =
                reinterpret_slice(left).expect("BD::Pixel layout matches u8");
            match variant {
                0 | 1 => wiener_8bpc(token, dst, left, lpf, lpf_off, w, h, params, edges),
                v => sgr_8bpc(token, dst, left, lpf, lpf_off, w, h, params, edges, v),
            }
        }
        BPC::BPC16 => {
            let left: &[LeftPixelRow<u16>] =
                reinterpret_slice(left).expect("BD::Pixel layout matches u16");
            match variant {
                0 | 1 => wiener_16bpc(token, dst, left, lpf, lpf_off, w, h, params, edges, bd_c),
                v => sgr_16bpc(token, dst, left, lpf, lpf_off, w, h, params, edges, v, bd_c),
            }
        }
    }
    true
}

/// Non-aarch64 builds have no tier here; the caller runs the reference.
#[cfg(not(target_arch = "aarch64"))]
pub fn lr_filter_dispatch<BD: BitDepth>(
    _variant: usize,
    _dst: PicOffset,
    _left: &[LeftPixelRow<BD::Pixel>],
    _lpf: &DisjointMut<AlignedVec64<u8>>,
    _lpf_off: isize,
    w: c_int,
    h: c_int,
    _params: &LooprestorationParams,
    _edges: LrEdgeFlags,
    _bd: BD,
) -> bool {
    crate::src::ablate::note(
        crate::src::ablate::Family::LoopRestoration,
        (w as i64 * h as i64).unsigned_abs(),
    );
    false
}
