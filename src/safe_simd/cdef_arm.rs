//! Safe ARM NEON implementations for CDEF (Constrained Directional Enhancement Filter)
//!
//! CDEF applies direction-dependent filtering to remove coding artifacts
//! while preserving edges.
//!
//! # Scratch-buffer representation
//!
//! `tmp` is a `[u16; 12 * 12]` window holding the block plus two pixels of
//! context on every side, with unavailable positions set to [`CDEF_VERY_LARGE`]
//! `== 0x8000`. That constant is `i16::MIN` reinterpreted, and it is chosen —
//! as it is in dav1d's own `src/arm/64/cdef_tmpl.S` and in `src/cdef.rs`'s
//! scalar reference, which fills with `i16::MIN` — so that a sentinel is
//! *self-neutralising* in all three places it is read:
//!
//! * `min` is taken with an UNSIGNED compare (`umin` / [`vminq_u16`]), where
//!   `0x8000` is larger than any pixel, so it never lowers the minimum;
//! * `max` is taken with a SIGNED compare (`smax` / [`vmaxq_s16`]), where
//!   `0x8000` is `-32768`, so it never raises the maximum;
//! * `constrain` computes `clip = max(0, threshold - (|diff| >> shift))` with an
//!   unsigned saturating subtract, and a sentinel's `|diff|` is ~32768, which
//!   drives `clip` to 0 for every strength/damping pair the spec can signal
//!   (proved by `constrain_neutralises_sentinel_over_full_param_space`), so the
//!   tap contributes exactly 0 to `sum`.
//!
//! Using a smaller sentinel (e.g. 8191) satisfies the first and third bullets
//! but NOT the second: an edge block would take `max = 8191`, disabling the
//! upper half of the `iclip` that the scalar reference performs. That is a real
//! divergence, and `sentinel_must_not_raise_max` is the regression test for it.
//!
//! # Exact-window padding guards
//!
//! The top/bottom padding loops read `x_start..x_end` of a row that sits two
//! rows outside the block. `x_start` is 2 when `HAVE_LEFT` is absent, i.e. the
//! block is at the left edge of the frame, so the two left-padding columns are
//! *skipped* — but they live at `offset - 2`, which is the **tail of the
//! previous row**. Guarding from `offset` instead of `offset + x_start` would
//! therefore lock 2 pixels this code never reads, in a row a concurrent tile
//! worker may legitimately be writing (`backup2lines` saves whole rows of
//! `cdef_line_buf`), producing a false `DisjointMut` overlap panic. So every
//! guard here starts at `offset + x_start` and is `x_end - x_start` long.
//! `src/cdef.rs`'s scalar reference has carried this discipline since the
//! i686 report; these kernels are held to the same rule.
//!
//! The WRITE side is one guard per destination row, exactly `w` pixels wide —
//! never a single wide guard spanning `(h - 1) * stride + w`. The 12-pixel tap
//! window is a READ of `tmp`, a private stack buffer, so it never straddles a
//! picture row at all: the only picture reads are the padding copies above, and
//! each of those is bounded to the columns it actually touches.

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
use std::ffi::c_uint;

use crate::include::common::bitdepth::AsPrimitive;
use crate::include::common::bitdepth::BitDepth;
use crate::include::common::bitdepth::DynPixel;
use crate::include::common::bitdepth::LeftPixelRow2px;
use crate::include::common::intops::iclip;
use crate::include::dav1d::picture::PicOffset;
use crate::src::cdef::CdefBottom;
use crate::src::cdef::CdefEdgeFlags;
use crate::src::cdef::CdefTop;
use crate::src::ffi_safe::FFISafe;
use crate::src::pic_or_buf::PicOrBuf;
use crate::src::strided::Strided as _;
use crate::src::tables::dav1d_cdef_directions;
use crate::src::with_offset::WithOffset;
// Used only by the `extern "C"` dispatch wrappers, which are asm-gated.
#[cfg_attr(not(all(feature = "asm", target_arch = "aarch64")), allow(dead_code))]
#[allow(non_camel_case_types)]
type ptrdiff_t = isize;

// Must match the row stride used in dav1d_cdef_directions (12).
const TMP_STRIDE: usize = 12;

/// Number of `u16` entries in the padded scratch window: 12 columns x
/// (8 block rows + 4 context rows).
const TMP_LEN: usize = TMP_STRIDE * 12;

/// Sentinel for "no pixel here" — `i16::MIN` reinterpreted as `u16`.
/// See the module docs for why this exact value and not a smaller one.
const CDEF_VERY_LARGE: u16 = 0x8000;

/// `tmp` offset of the block's top-left pixel (2 context rows, 2 context cols).
const TMP_OFFSET: usize = 2 * TMP_STRIDE + 2;

/// Derive `bitdepth_min_8` from the signalled `bitdepth_max`.
///
/// `bitdepth_max` is `(1 << bitdepth) - 1`, so `ilog2(bitdepth_max + 1)` is the
/// bit depth. Doing it this way (rather than `if bitdepth_max == 1023 {2} else {4}`)
/// also gives the right answer for 8-bit content decoded through the 16bpc path,
/// which happens in a `--no-default-features --features bitdepth_16` build.
#[inline(always)]
fn bitdepth_min_8_of(bitdepth_max: c_int) -> c_int {
    ((bitdepth_max as u32 + 1).ilog2() as c_int - 8).max(0)
}

/// Scalar `constrain`, matching `src/cdef.rs`'s `constrain` exactly.
#[inline(always)]
fn constrain_scalar(diff: i32, threshold: c_int, shift: c_int) -> i32 {
    let adiff = diff.abs();
    let term = threshold - (adiff >> shift);
    let max_term = cmp::max(0, term);
    let result = cmp::min(adiff, max_term);
    if diff < 0 { -result } else { result }
}

/// Read a `tmp` entry with the scalar reference's sign convention: the sentinel
/// reads back as `i16::MIN`, exactly as `src/cdef.rs`'s `[i16]` scratch does.
#[inline(always)]
fn tmp_at(tmp: &[u16; TMP_LEN], idx: usize) -> i32 {
    tmp[idx] as i16 as i32
}

// ============================================================================
// PADDING (shared shape, per-bit-depth pixel loads)
// ============================================================================

/// `dst[..N] = src[..N] as u16`, with a compile-time trip count.
///
/// The fixed-size array is the point: with a runtime length LLVM emits a
/// byte-at-a-time loop, and with `[u8; N]` it recognises the structured widen
/// and issues `ushll` (see the fixed-size-array note in the project's
/// performance guidance).
#[inline(always)]
fn widen_n<const N: usize>(dst: &mut [u16], src: &[u8]) {
    let a = <&[u8; N]>::try_from(&src[..N]).unwrap();
    for i in 0..N {
        dst[i] = a[i] as u16;
    }
}

/// Dispatch [`widen_n`] over the handful of lengths CDEF padding can ask for:
/// a block row is `w` or `w + 2` wide and a context row is `x_end - x_start`,
/// with `w` in {4, 8}, so `n` is one of 4, 6, 8, 10, 12.
#[inline(always)]
fn widen_row(dst: &mut [u16], src: &[u8], n: usize) {
    match n {
        4 => widen_n::<4>(dst, src),
        6 => widen_n::<6>(dst, src),
        8 => widen_n::<8>(dst, src),
        10 => widen_n::<10>(dst, src),
        12 => widen_n::<12>(dst, src),
        _ => {
            for i in 0..n {
                dst[i] = src[i] as u16;
            }
        }
    }
}

/// Padding function for 8bpc — copies the block and its available context into
/// the scratch window; everything else keeps the [`CDEF_VERY_LARGE`] sentinel.
fn padding_8bpc(
    tmp: &mut [u16; TMP_LEN],
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

    // Copy source pixels, and the two right-context pixels in the SAME borrow.
    // Two guards over `[0, w)` and `[0, w + 2)` of one row cover exactly the
    // same bytes one guard over `[0, w + 2)` does, so folding them is a strict
    // reduction in borrow COUNT with no widening — and CDEF's remaining cost is
    // borrow count (profiled 2026-08-07: tracker add + guard drop was 1.60% of
    // an 8bpc t=1 decode against 0.22% for the filter arithmetic itself).
    let read_w = if edges.contains(CdefEdgeFlags::HAVE_RIGHT) {
        w + 2
    } else {
        w
    };
    for y in 0..h {
        let row_offset = TMP_OFFSET + y * TMP_STRIDE;
        let src = (dst + (y as isize * stride)).slice::<BitDepth8>(read_w);
        widen_row(&mut tmp[row_offset..], &src, read_w);
    }

    // Handle left edge
    if edges.contains(CdefEdgeFlags::HAVE_LEFT) {
        for y in 0..h {
            let row_offset = TMP_OFFSET + y * TMP_STRIDE;
            tmp[row_offset - 2] = left[y][0] as u16;
            tmp[row_offset - 1] = left[y][1] as u16;
        }
    }

    let (x_start, x_end) = pad_x_window(edges, w);

    // Handle top edge (safe slice access via DisjointMut)
    if edges.contains(CdefEdgeFlags::HAVE_TOP) {
        for dy in 0..2usize {
            let row_offset = TMP_OFFSET - (2 - dy) * TMP_STRIDE;
            let top_row = WithOffset {
                data: top.data,
                offset: top
                    .offset
                    .wrapping_sub(2)
                    .wrapping_add_signed(dy as isize * stride),
            };
            // Guard exactly the columns read (`x_start..x_end`) — see the
            // module note on exact-window padding guards.
            let slice = top_row
                .data
                .slice_as::<_, u8>((top_row.offset + x_start.., ..x_end - x_start));
            widen_row(
                &mut tmp[row_offset + x_start - 2..],
                &slice,
                x_end - x_start,
            );
        }
    }

    // Handle bottom edge (safe slice access via DisjointMut/PicOrBuf)
    if edges.contains(CdefEdgeFlags::HAVE_BOTTOM) {
        for dy in 0..2usize {
            let row_offset = TMP_OFFSET + (h + dy) * TMP_STRIDE;
            let bottom_row = WithOffset {
                data: bottom.data,
                offset: bottom
                    .offset
                    .wrapping_sub(2)
                    .wrapping_add_signed(dy as isize * stride),
            };
            // Same exact-window discipline as the top loop above.
            let slice = match bottom_row.data {
                PicOrBuf::Pic(pic) => {
                    let guard = pic
                        .slice::<BitDepth8, _>((bottom_row.offset + x_start.., ..x_end - x_start));
                    widen_row(
                        &mut tmp[row_offset + x_start - 2..],
                        &guard,
                        x_end - x_start,
                    );
                    continue;
                }
                PicOrBuf::Buf(buf) => {
                    buf.slice_as::<_, u8>((bottom_row.offset + x_start.., ..x_end - x_start))
                }
            };
            widen_row(
                &mut tmp[row_offset + x_start - 2..],
                &slice,
                x_end - x_start,
            );
        }
    }
}

/// Column window the top/bottom padding rows actually read. Split out so the
/// two bit depths cannot drift apart on the guard-extent rule.
#[inline(always)]
fn pad_x_window(edges: CdefEdgeFlags, w: usize) -> (usize, usize) {
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
    (x_start, x_end)
}

/// Padding function for 16bpc.
fn padding_16bpc(
    tmp: &mut [u16; TMP_LEN],
    dst: PicOffset,
    left: &[LeftPixelRow2px<u16>; 8],
    top: &CdefTop,
    bottom: &CdefBottom,
    w: usize,
    h: usize,
    edges: CdefEdgeFlags,
) {
    use crate::include::common::bitdepth::BitDepth16;

    let stride = dst.pixel_stride::<BitDepth16>();

    // Copy source pixels + the two right-context pixels in ONE borrow per row;
    // see the note in `padding_8bpc`.
    let read_w = if edges.contains(CdefEdgeFlags::HAVE_RIGHT) {
        w + 2
    } else {
        w
    };
    for y in 0..h {
        let row_offset = TMP_OFFSET + y * TMP_STRIDE;
        let src = (dst + (y as isize * stride)).slice::<BitDepth16>(read_w);
        tmp[row_offset..row_offset + read_w].copy_from_slice(&src[..read_w]);
    }

    // Handle left edge
    if edges.contains(CdefEdgeFlags::HAVE_LEFT) {
        for y in 0..h {
            let row_offset = TMP_OFFSET + y * TMP_STRIDE;
            tmp[row_offset - 2] = left[y][0];
            tmp[row_offset - 1] = left[y][1];
        }
    }

    let (x_start, x_end) = pad_x_window(edges, w);

    // Handle top edge (safe slice access via DisjointMut)
    if edges.contains(CdefEdgeFlags::HAVE_TOP) {
        for dy in 0..2usize {
            let row_offset = TMP_OFFSET - (2 - dy) * TMP_STRIDE;
            let top_row = WithOffset {
                data: top.data,
                offset: top
                    .offset
                    .wrapping_sub(2)
                    .wrapping_add_signed(dy as isize * stride),
            };
            // Guard exactly the columns read (`x_start..x_end`) — see the
            // module note on exact-window padding guards.
            let slice = top_row
                .data
                .slice_as::<_, u16>((top_row.offset + x_start.., ..x_end - x_start));
            tmp[row_offset + x_start - 2..row_offset + x_end - 2]
                .copy_from_slice(&slice[..x_end - x_start]);
        }
    }

    // Handle bottom edge (safe slice access via DisjointMut/PicOrBuf)
    if edges.contains(CdefEdgeFlags::HAVE_BOTTOM) {
        for dy in 0..2usize {
            let row_offset = TMP_OFFSET + (h + dy) * TMP_STRIDE;
            let bottom_row = WithOffset {
                data: bottom.data,
                offset: bottom
                    .offset
                    .wrapping_sub(2)
                    .wrapping_add_signed(dy as isize * stride),
            };
            // Same exact-window discipline as the top loop above.
            let dst_range = row_offset + x_start - 2..row_offset + x_end - 2;
            match bottom_row.data {
                PicOrBuf::Pic(pic) => {
                    let guard = pic
                        .slice::<BitDepth16, _>((bottom_row.offset + x_start.., ..x_end - x_start));
                    tmp[dst_range].copy_from_slice(&guard[..x_end - x_start]);
                }
                PicOrBuf::Buf(buf) => {
                    let slice =
                        buf.slice_as::<_, u16>((bottom_row.offset + x_start.., ..x_end - x_start));
                    tmp[dst_range].copy_from_slice(&slice[..x_end - x_start]);
                }
            }
        }
    }
}

// ============================================================================
// NEON FILTER CORE (bit-depth independent)
// ============================================================================
//
// One vector per destination ROW. Every tap the scalar loop reads at
// `tmp[base + x + off]` for x in 0..w is a contiguous run, so a whole row's
// worth of one tap is a single 8-lane load; the twelve taps (2 primary
// offsets and 4 secondary, each used with + and -) become twelve loads
// instead of twelve loads PER PIXEL.
//
// Everything is computed in i16/u16 and is exactly what `constrain_scalar`
// and the scalar loop do — no reassociation, no rounding-shift substitution:
//   * A real tap and a real pixel are both in `0..=bitdepth_max` (<= 4095), so
//     `p - px` and `|p - px|` fit i16 with room to spare. A sentinel tap makes
//     the signed difference wrap, but it also forces `clip == 0`, and
//     `clamp(anything, -0, 0) == 0`, so the wrapped value never escapes.
//   * `sum` is bounded by 4 primary taps x tap-weight 4 x threshold 240 plus
//     8 secondary taps x weight 2 x threshold 64, i.e. |sum| < 4096 at 12bpc
//     (and < 512 at 8bpc).
//   * the final `px + (sum - (sum < 0) + 8 >> 4)` can leave `0..=bitdepth_max`
//     on the primary-only path (which the scalar does not clip), and the scalar
//     stores it with a truncating `as`, so the 8bpc narrow here is `vmovn_u16`
//     (truncating), NOT `vqmovun_s16` (saturating), and the 16bpc store is the
//     raw i16 lane reinterpreted.
//
// `w` is 4 or 8; a 4-wide block still computes 8 lanes and stores 4. Both the
// taps AND `px` of the spare lanes come from `tmp`, so a spare `px` lane can
// itself be the `0x8000` sentinel. That is harmless: every NEON op here is
// lane-independent, a sentinel `px` drives `clip` to 0 against any tap (so
// `sum` stays 0 in that lane), and `min`/`max` keep it out of the clip bounds
// for the same reason they keep out a sentinel tap. Those lanes are discarded
// by the `w`-wide store.
//
// Index bounds, so the fixed-size-array loads below can never fail: `base`
// is `26 + y * 12` with y <= 7 (h <= 8), so base <= 110; the direction table's
// offsets run -22..=26; and a load reads 8 lanes, so the extremes are
// 26 - 22 = 4 and 110 + 26 + 7 = 143, exactly inside `tmp`'s 144 entries.

/// One 8-lane tap load from the padded scratch buffer.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn cdef_tap(tmp: &[u16; TMP_LEN], idx: usize) -> uint16x8_t {
    safe_simd::vld1q_u16(<&[u16; 8]>::try_from(&tmp[idx..idx + 8]).unwrap())
}

/// Lane-wise `constrain(p - px, threshold, shift)`.
///
/// This is dav1d's `handle_pixel` formulation:
/// `clip = uqsub(threshold, |p - px| >> shift)` then `clamp(p - px, -clip, clip)`.
/// It equals `src/cdef.rs`'s `sign(diff) * min(|diff|, max(0, threshold - (|diff| >> shift)))`
/// because `clip >= 0` makes the clamp `sign(diff) * min(|diff|, clip)`.
///
/// `neg_shift` is `-shift`: NEON has no variable right shift, so a negative
/// `vshlq_u16` count is the right shift.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn cdef_constrain(
    p: uint16x8_t,
    px: uint16x8_t,
    threshold: uint16x8_t,
    neg_shift: int16x8_t,
) -> int16x8_t {
    let adiff = vabdq_u16(p, px);
    let clip = vreinterpretq_s16_u16(vqsubq_u16(threshold, vshlq_u16(adiff, neg_shift)));
    let diff = vsubq_s16(vreinterpretq_s16_u16(p), vreinterpretq_s16_u16(px));
    vmaxq_s16(vminq_s16(diff, clip), vnegq_s16(clip))
}

/// `px + (sum - (sum < 0) + 8 >> 4)`, exactly as the scalar writes it.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn cdef_apply(px: int16x8_t, sum: int16x8_t) -> int16x8_t {
    let neg = vreinterpretq_s16_u16(vcltq_s16(sum, vdupq_n_s16(0)));
    let biased = vaddq_s16(vaddq_s16(sum, neg), vdupq_n_s16(8));
    vaddq_s16(px, vshrq_n_s16::<4>(biased))
}

/// Filter one destination row: eight lanes in, eight filtered lanes out.
///
/// `PRI` / `SEC` select which of the scalar reference's three branches this is;
/// they are const so each branch monomorphises without a runtime test, and so
/// the primary-only branch provably skips the min/max clip the scalar also
/// skips.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn cdef_filter_row<const PRI: bool, const SEC: bool>(
    tmp: &[u16; TMP_LEN],
    base: usize,
    px: uint16x8_t,
    dir: usize,
    pri_threshold: uint16x8_t,
    pri_neg_shift: int16x8_t,
    sec_threshold: uint16x8_t,
    sec_neg_shift: int16x8_t,
    pri_tap: i16,
) -> int16x8_t {
    let px_s = vreinterpretq_s16_u16(px);
    let mut sum = vdupq_n_s16(0);
    // `min` accumulates with an UNSIGNED compare and `max` with a SIGNED one,
    // so the 0x8000 sentinel is inert in both — see the module docs.
    let mut lo = px;
    let mut hi = px_s;
    let mut pri_tap_k = pri_tap;

    for k in 0..2 {
        if PRI {
            let off = dav1d_cdef_directions[dir + 2][k] as isize;
            let p0 = cdef_tap(tmp, (base as isize + off) as usize);
            let p1 = cdef_tap(tmp, (base as isize - off) as usize);
            sum = vmlaq_n_s16(
                sum,
                cdef_constrain(p0, px, pri_threshold, pri_neg_shift),
                pri_tap_k,
            );
            sum = vmlaq_n_s16(
                sum,
                cdef_constrain(p1, px, pri_threshold, pri_neg_shift),
                pri_tap_k,
            );
            // If `pri_tap_k == 4`, then it becomes 2, else it remains 3.
            pri_tap_k = pri_tap_k & 3 | 2;
            if SEC {
                lo = vminq_u16(lo, vminq_u16(p0, p1));
                hi = vmaxq_s16(
                    hi,
                    vmaxq_s16(vreinterpretq_s16_u16(p0), vreinterpretq_s16_u16(p1)),
                );
            }
        }

        if SEC {
            let off2 = dav1d_cdef_directions[dir + 4][k] as isize;
            let off3 = dav1d_cdef_directions[dir][k] as isize;
            let s0 = cdef_tap(tmp, (base as isize + off2) as usize);
            let s1 = cdef_tap(tmp, (base as isize - off2) as usize);
            let s2 = cdef_tap(tmp, (base as isize + off3) as usize);
            let s3 = cdef_tap(tmp, (base as isize - off3) as usize);

            // `sec_tap` starts at 2 and becomes 1.
            let sec_tap = 2 - k as i16;
            sum = vmlaq_n_s16(
                sum,
                cdef_constrain(s0, px, sec_threshold, sec_neg_shift),
                sec_tap,
            );
            sum = vmlaq_n_s16(
                sum,
                cdef_constrain(s1, px, sec_threshold, sec_neg_shift),
                sec_tap,
            );
            sum = vmlaq_n_s16(
                sum,
                cdef_constrain(s2, px, sec_threshold, sec_neg_shift),
                sec_tap,
            );
            sum = vmlaq_n_s16(
                sum,
                cdef_constrain(s3, px, sec_threshold, sec_neg_shift),
                sec_tap,
            );

            lo = vminq_u16(lo, vminq_u16(vminq_u16(s0, s1), vminq_u16(s2, s3)));
            hi = vmaxq_s16(
                hi,
                vmaxq_s16(
                    vmaxq_s16(vreinterpretq_s16_u16(s0), vreinterpretq_s16_u16(s1)),
                    vmaxq_s16(vreinterpretq_s16_u16(s2), vreinterpretq_s16_u16(s3)),
                ),
            );
        }
    }

    let out = cdef_apply(px_s, sum);
    // The scalar clips only when a secondary strength contributed a min/max;
    // the primary-only branch stores the raw value.
    if SEC {
        vminq_s16(vmaxq_s16(out, vreinterpretq_s16_u16(lo)), hi)
    } else {
        out
    }
}

/// Shift/threshold vectors shared by every row of a block.
#[cfg(target_arch = "aarch64")]
struct FilterParams {
    pri_threshold: u16,
    pri_neg_shift: i16,
    sec_threshold: u16,
    sec_neg_shift: i16,
    pri_tap: i16,
}

#[cfg(target_arch = "aarch64")]
fn filter_params(
    pri_strength: c_int,
    sec_strength: c_int,
    damping: c_int,
    bitdepth_min_8: c_int,
) -> FilterParams {
    // `ilog2` panics on 0, and the scalar only reaches these two lines inside
    // its `!= 0` branches — a secondary-only block really does arrive here
    // with `pri_strength == 0` (caught by `tile_threading_parity`).
    let pri_shift = if pri_strength != 0 {
        cmp::max(0, damping - pri_strength.ilog2() as c_int)
    } else {
        0
    };
    // `damping - ilog2(sec_strength)` is non-negative for every strength the
    // spec can signal; the scalar `>>` would panic in debug if it were not.
    let sec_shift = if sec_strength != 0 {
        damping - sec_strength.ilog2() as c_int
    } else {
        0
    };
    debug_assert!(sec_shift >= 0);
    FilterParams {
        pri_threshold: pri_strength as u16,
        pri_neg_shift: -(pri_shift as i16),
        sec_threshold: sec_strength as u16,
        sec_neg_shift: -(sec_shift.max(0) as i16),
        // The caller scaled the CDEF level by `bitdepth_min_8` before it got
        // here (`y_pri_lvl = (y_lvl >> 2) << bitdepth_min_8` in cdef_apply),
        // so the tap-selecting parity bit is bit `bitdepth_min_8`, not bit 0.
        // dav1d does the same shift in its NEON path (`src/arm/64/cdef_tmpl.S`
        // `lsr w9, w3, w9` / `and w9, w9, #1` under `.if \bpc == 16`).
        pri_tap: (4 - (pri_strength >> bitdepth_min_8 & 1)) as i16,
    }
}

// ============================================================================
// 8BPC FILTER
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[arcane]
fn cdef_filter_block_8bpc_neon<const W: usize, const H: usize, const PRI: bool, const SEC: bool>(
    _token: Arm64,
    dst: PicOffset,
    tmp: &[u16; TMP_LEN],
    dir: usize,
    p: &FilterParams,
) {
    use crate::include::common::bitdepth::BitDepth8;

    let stride = dst.pixel_stride::<BitDepth8>();
    let pri_threshold = vdupq_n_u16(p.pri_threshold);
    let pri_neg_shift = vdupq_n_s16(p.pri_neg_shift);
    let sec_threshold = vdupq_n_u16(p.sec_threshold);
    let sec_neg_shift = vdupq_n_s16(p.sec_neg_shift);

    for y in 0..H {
        let base = TMP_OFFSET + y * TMP_STRIDE;
        // `px` comes from `tmp`, not from a second read of `dst`: `padding_*`
        // already copied this row's pixels there and nothing has written the
        // row since, so the values are identical and this saves a read through
        // the guard. For W == 4 the upper four lanes pick up whatever `tmp`
        // holds beyond the block (a right-context pixel or the sentinel); the
        // kernel is lane-independent and bounded for both, and those lanes are
        // discarded by the W-wide store.
        let px = cdef_tap(tmp, base);

        let out = cdef_filter_row::<PRI, SEC>(
            tmp,
            base,
            px,
            dir,
            pri_threshold,
            pri_neg_shift,
            sec_threshold,
            sec_neg_shift,
            p.pri_tap,
        );
        let packed = vmovn_u16(vreinterpretq_u16_s16(out));

        // One guard per destination row, exactly W pixels wide.
        let mut dst_row = (dst + (y as isize * stride)).slice_mut::<BitDepth8>(W);
        if W == 8 {
            safe_simd::vst1_u8(<&mut [u8; 8]>::try_from(&mut dst_row[..8]).unwrap(), packed);
        } else {
            let mut buf = [0u8; 8];
            safe_simd::vst1_u8(&mut buf, packed);
            dst_row[..W].copy_from_slice(&buf[..W]);
        }
    }
}

/// Route to the right `(W, H, PRI, SEC)` monomorphisation.
#[cfg(target_arch = "aarch64")]
fn cdef_filter_block_8bpc_dispatch<const W: usize, const H: usize>(
    token: Arm64,
    dst: PicOffset,
    tmp: &[u16; TMP_LEN],
    dir: usize,
    pri_strength: c_int,
    sec_strength: c_int,
    p: &FilterParams,
) {
    match (pri_strength != 0, sec_strength != 0) {
        (true, true) => cdef_filter_block_8bpc_neon::<W, H, true, true>(token, dst, tmp, dir, p),
        (true, false) => cdef_filter_block_8bpc_neon::<W, H, true, false>(token, dst, tmp, dir, p),
        (false, true) => cdef_filter_block_8bpc_neon::<W, H, false, true>(token, dst, tmp, dir, p),
        (false, false) => {}
    }
}

/// CDEF filter inner implementation for 8bpc.
fn cdef_filter_block_8bpc_inner(
    dst: PicOffset,
    left: &[LeftPixelRow2px<u8>; 8],
    top: &CdefTop,
    bottom: &CdefBottom,
    pri_strength: c_int,
    sec_strength: c_int,
    dir: c_int,
    damping: c_int,
    edges: CdefEdgeFlags,
    w: usize,
    h: usize,
) {
    // With both strengths zero the scalar takes no row guard and writes
    // nothing; neither path here may either (it would be a no-op write plus
    // `h` extra borrows).
    if pri_strength == 0 && sec_strength == 0 {
        return;
    }

    let dir = dir as usize;
    let mut tmp = [CDEF_VERY_LARGE; TMP_LEN];
    padding_8bpc(&mut tmp, dst, left, top, bottom, w, h, edges);
    let params = filter_params(pri_strength, sec_strength, damping, 0);

    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken as _;
        if let Some(token) = archmage::Arm64::summon() {
            match (w, h) {
                (8, 8) => cdef_filter_block_8bpc_dispatch::<8, 8>(
                    token,
                    dst,
                    &tmp,
                    dir,
                    pri_strength,
                    sec_strength,
                    &params,
                ),
                (4, 8) => cdef_filter_block_8bpc_dispatch::<4, 8>(
                    token,
                    dst,
                    &tmp,
                    dir,
                    pri_strength,
                    sec_strength,
                    &params,
                ),
                _ => cdef_filter_block_8bpc_dispatch::<4, 4>(
                    token,
                    dst,
                    &tmp,
                    dir,
                    pri_strength,
                    sec_strength,
                    &params,
                ),
            }
            return;
        }
    }

    cdef_filter_block_scalar::<crate::include::common::bitdepth::BitDepth8>(
        dst,
        &tmp,
        pri_strength,
        sec_strength,
        dir,
        damping,
        w,
        h,
        0,
    );
}

// ============================================================================
// 16BPC FILTER
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[arcane]
fn cdef_filter_block_16bpc_neon<
    const W: usize,
    const H: usize,
    const PRI: bool,
    const SEC: bool,
>(
    _token: Arm64,
    dst: PicOffset,
    tmp: &[u16; TMP_LEN],
    dir: usize,
    p: &FilterParams,
) {
    use crate::include::common::bitdepth::BitDepth16;

    let stride = dst.pixel_stride::<BitDepth16>();
    let pri_threshold = vdupq_n_u16(p.pri_threshold);
    let pri_neg_shift = vdupq_n_s16(p.pri_neg_shift);
    let sec_threshold = vdupq_n_u16(p.sec_threshold);
    let sec_neg_shift = vdupq_n_s16(p.sec_neg_shift);

    for y in 0..H {
        let base = TMP_OFFSET + y * TMP_STRIDE;
        // `px` from `tmp` — see the note in the 8bpc kernel.
        let px = cdef_tap(tmp, base);

        let out = cdef_filter_row::<PRI, SEC>(
            tmp,
            base,
            px,
            dir,
            pri_threshold,
            pri_neg_shift,
            sec_threshold,
            sec_neg_shift,
            p.pri_tap,
        );
        let packed = vreinterpretq_u16_s16(out);

        // One guard per destination row, exactly W pixels wide.
        let mut dst_row = (dst + (y as isize * stride)).slice_mut::<BitDepth16>(W);
        if W == 8 {
            safe_simd::vst1q_u16(
                <&mut [u16; 8]>::try_from(&mut dst_row[..8]).unwrap(),
                packed,
            );
        } else {
            let mut buf = [0u16; 8];
            safe_simd::vst1q_u16(&mut buf, packed);
            dst_row[..W].copy_from_slice(&buf[..W]);
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn cdef_filter_block_16bpc_dispatch<const W: usize, const H: usize>(
    token: Arm64,
    dst: PicOffset,
    tmp: &[u16; TMP_LEN],
    dir: usize,
    pri_strength: c_int,
    sec_strength: c_int,
    p: &FilterParams,
) {
    match (pri_strength != 0, sec_strength != 0) {
        (true, true) => cdef_filter_block_16bpc_neon::<W, H, true, true>(token, dst, tmp, dir, p),
        (true, false) => cdef_filter_block_16bpc_neon::<W, H, true, false>(token, dst, tmp, dir, p),
        (false, true) => cdef_filter_block_16bpc_neon::<W, H, false, true>(token, dst, tmp, dir, p),
        (false, false) => {}
    }
}

/// CDEF filter inner implementation for 16bpc.
fn cdef_filter_block_16bpc_inner(
    dst: PicOffset,
    left: &[LeftPixelRow2px<u16>; 8],
    top: &CdefTop,
    bottom: &CdefBottom,
    pri_strength: c_int,
    sec_strength: c_int,
    dir: c_int,
    damping: c_int,
    edges: CdefEdgeFlags,
    w: usize,
    h: usize,
    bitdepth_max: i32,
) {
    if pri_strength == 0 && sec_strength == 0 {
        return;
    }

    let dir = dir as usize;
    let bitdepth_min_8 = bitdepth_min_8_of(bitdepth_max);
    let mut tmp = [CDEF_VERY_LARGE; TMP_LEN];
    padding_16bpc(&mut tmp, dst, left, top, bottom, w, h, edges);
    let params = filter_params(pri_strength, sec_strength, damping, bitdepth_min_8);

    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken as _;
        if let Some(token) = archmage::Arm64::summon() {
            match (w, h) {
                (8, 8) => cdef_filter_block_16bpc_dispatch::<8, 8>(
                    token,
                    dst,
                    &tmp,
                    dir,
                    pri_strength,
                    sec_strength,
                    &params,
                ),
                (4, 8) => cdef_filter_block_16bpc_dispatch::<4, 8>(
                    token,
                    dst,
                    &tmp,
                    dir,
                    pri_strength,
                    sec_strength,
                    &params,
                ),
                _ => cdef_filter_block_16bpc_dispatch::<4, 4>(
                    token,
                    dst,
                    &tmp,
                    dir,
                    pri_strength,
                    sec_strength,
                    &params,
                ),
            }
            return;
        }
    }

    cdef_filter_block_scalar::<crate::include::common::bitdepth::BitDepth16>(
        dst,
        &tmp,
        pri_strength,
        sec_strength,
        dir,
        damping,
        w,
        h,
        bitdepth_min_8,
    );
}

// ============================================================================
// SCALAR FALLBACK (no NEON token)
// ============================================================================

/// Scalar filter over the same scratch representation, transcribed from
/// `src/cdef.rs`'s `cdef_filter_block_rust` — including the unsigned `min` /
/// signed `max` that make the sentinel inert on both sides.
///
/// Reachable only if `Arm64::summon()` fails, which it cannot on a NEON-mandatory
/// aarch64 target; it exists so the module has no unreachable-panic path and so
/// the tests can drive the reference arithmetic over this buffer layout.
fn cdef_filter_block_scalar<BD: BitDepth>(
    dst: PicOffset,
    tmp: &[u16; TMP_LEN],
    pri_strength: c_int,
    sec_strength: c_int,
    dir: usize,
    damping: c_int,
    w: usize,
    h: usize,
    bitdepth_min_8: c_int,
) {
    let stride = dst.pixel_stride::<BD>();

    if pri_strength != 0 {
        let pri_tap = 4 - (pri_strength >> bitdepth_min_8 & 1);
        let pri_shift = cmp::max(0, damping - pri_strength.ilog2() as c_int);

        if sec_strength != 0 {
            let sec_shift = damping - sec_strength.ilog2() as c_int;
            for y in 0..h {
                let base = TMP_OFFSET + y * TMP_STRIDE;
                let mut dst_row = (dst + (y as isize * stride)).slice_mut::<BD>(w);
                for x in 0..w {
                    let px = dst_row[x].as_::<c_int>();
                    let mut sum = 0;
                    let mut max = px;
                    let mut min = px;
                    let bx = (base + x) as isize;
                    let mut pri_tap_k = pri_tap;
                    for k in 0..2 {
                        let off1 = dav1d_cdef_directions[dir + 2][k] as isize;
                        let p0 = tmp_at(tmp, (bx + off1) as usize);
                        let p1 = tmp_at(tmp, (bx - off1) as usize);
                        sum += pri_tap_k * constrain_scalar(p0 - px, pri_strength, pri_shift);
                        sum += pri_tap_k * constrain_scalar(p1 - px, pri_strength, pri_shift);
                        pri_tap_k = pri_tap_k & 3 | 2;
                        min = cmp::min(p0 as c_uint, min as c_uint) as c_int;
                        max = cmp::max(p0, max);
                        min = cmp::min(p1 as c_uint, min as c_uint) as c_int;
                        max = cmp::max(p1, max);

                        let off2 = dav1d_cdef_directions[dir + 4][k] as isize;
                        let off3 = dav1d_cdef_directions[dir][k] as isize;
                        let s0 = tmp_at(tmp, (bx + off2) as usize);
                        let s1 = tmp_at(tmp, (bx - off2) as usize);
                        let s2 = tmp_at(tmp, (bx + off3) as usize);
                        let s3 = tmp_at(tmp, (bx - off3) as usize);
                        let sec_tap = 2 - k as c_int;
                        sum += sec_tap * constrain_scalar(s0 - px, sec_strength, sec_shift);
                        sum += sec_tap * constrain_scalar(s1 - px, sec_strength, sec_shift);
                        sum += sec_tap * constrain_scalar(s2 - px, sec_strength, sec_shift);
                        sum += sec_tap * constrain_scalar(s3 - px, sec_strength, sec_shift);
                        min = cmp::min(s0 as c_uint, min as c_uint) as c_int;
                        max = cmp::max(s0, max);
                        min = cmp::min(s1 as c_uint, min as c_uint) as c_int;
                        max = cmp::max(s1, max);
                        min = cmp::min(s2 as c_uint, min as c_uint) as c_int;
                        max = cmp::max(s2, max);
                        min = cmp::min(s3 as c_uint, min as c_uint) as c_int;
                        max = cmp::max(s3, max);
                    }
                    dst_row[x] = iclip(px + (sum - (sum < 0) as c_int + 8 >> 4), min, max)
                        .as_::<BD::Pixel>();
                }
            }
        } else {
            for y in 0..h {
                let base = TMP_OFFSET + y * TMP_STRIDE;
                let mut dst_row = (dst + (y as isize * stride)).slice_mut::<BD>(w);
                for x in 0..w {
                    let px = dst_row[x].as_::<c_int>();
                    let mut sum = 0;
                    let bx = (base + x) as isize;
                    let mut pri_tap_k = pri_tap;
                    for k in 0..2 {
                        let off = dav1d_cdef_directions[dir + 2][k] as isize;
                        let p0 = tmp_at(tmp, (bx + off) as usize);
                        let p1 = tmp_at(tmp, (bx - off) as usize);
                        sum += pri_tap_k * constrain_scalar(p0 - px, pri_strength, pri_shift);
                        sum += pri_tap_k * constrain_scalar(p1 - px, pri_strength, pri_shift);
                        pri_tap_k = pri_tap_k & 3 | 2;
                    }
                    dst_row[x] = (px + (sum - (sum < 0) as c_int + 8 >> 4)).as_::<BD::Pixel>();
                }
            }
        }
    } else {
        let sec_shift = damping - sec_strength.ilog2() as c_int;
        for y in 0..h {
            let base = TMP_OFFSET + y * TMP_STRIDE;
            let mut dst_row = (dst + (y as isize * stride)).slice_mut::<BD>(w);
            for x in 0..w {
                let px = dst_row[x].as_::<c_int>();
                let mut sum = 0;
                let mut max = px;
                let mut min = px;
                let bx = (base + x) as isize;
                for k in 0..2 {
                    let off2 = dav1d_cdef_directions[dir + 4][k] as isize;
                    let off3 = dav1d_cdef_directions[dir][k] as isize;
                    let s0 = tmp_at(tmp, (bx + off2) as usize);
                    let s1 = tmp_at(tmp, (bx - off2) as usize);
                    let s2 = tmp_at(tmp, (bx + off3) as usize);
                    let s3 = tmp_at(tmp, (bx - off3) as usize);
                    let sec_tap = 2 - k as c_int;
                    sum += sec_tap * constrain_scalar(s0 - px, sec_strength, sec_shift);
                    sum += sec_tap * constrain_scalar(s1 - px, sec_strength, sec_shift);
                    sum += sec_tap * constrain_scalar(s2 - px, sec_strength, sec_shift);
                    sum += sec_tap * constrain_scalar(s3 - px, sec_strength, sec_shift);
                    min = cmp::min(s0 as c_uint, min as c_uint) as c_int;
                    max = cmp::max(s0, max);
                    min = cmp::min(s1 as c_uint, min as c_uint) as c_int;
                    max = cmp::max(s1, max);
                    min = cmp::min(s2 as c_uint, min as c_uint) as c_int;
                    max = cmp::max(s2, max);
                    min = cmp::min(s3 as c_uint, min as c_uint) as c_int;
                    max = cmp::max(s3, max);
                }
                dst_row[x] =
                    iclip(px + (sum - (sum < 0) as c_int + 8 >> 4), min, max).as_::<BD::Pixel>();
            }
        }
    }
}

// ============================================================================
// CDEF DIRECTION FINDING
// ============================================================================
//
// The 8x8 direction search accumulates eight sets of partial sums (2 hv, 2
// diagonal, 4 "alt" 2:1 diagonals) over 64 pixels, then scores 8 directions
// and takes an argmax.
//
// Only the ACCUMULATION is vectorised. Each of the six scatter targets is a run
// of consecutive entries at a row-dependent offset, so a whole row lands with
// one unaligned load / add / store:
//   diag0[y      ..+8] += row          diag1[y      ..+8] += reverse(row)
//   alt2 [3-(y>>1)..+8] += row         alt3 [(y>>1)  ..+8] += row
//   alt0 [y      ..+4] += pairs(row)   alt1 [y      ..+4] += reverse(pairs(row))
//   hv0[y] = horizontal sum(row)       hv1 += row
//
// Partial sums are accumulated in i16, not the reference's i32. That is exact,
// not an approximation: `px` is in -128..=127 and no partial sum takes more
// than 8 terms, so |partial| <= 1024 — an order of magnitude inside i16. They
// are widened back to i32 before any squaring, so the cost arithmetic is
// bit-for-bit the reference's.
//
// The COST computation and the ARGMAX stay scalar and byte-identical to
// `cdef_find_dir_rust`. That is deliberate: the argmax is `if cost[n] >
// best_cost`, i.e. STRICT, so the LOWEST index wins a tie, and flat/synthetic
// content produces ties constantly. A vectorised max-reduce would have to
// reproduce that index-priority exactly, and getting it wrong yields a
// different-but-plausible direction rather than a visible failure. Eight u32
// comparisons are not the bottleneck; the 512 scalar accumulator updates were.

/// `arr[base..base + 8] += v`.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn acc8(arr: &mut [i16; 16], base: usize, v: int16x8_t) {
    let cur = safe_simd::vld1q_s16(<&[i16; 8]>::try_from(&arr[base..base + 8]).unwrap());
    safe_simd::vst1q_s16(
        <&mut [i16; 8]>::try_from(&mut arr[base..base + 8]).unwrap(),
        vaddq_s16(cur, v),
    );
}

/// `arr[base..base + 4] += v`.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn acc4(arr: &mut [i16; 16], base: usize, v: int16x4_t) {
    let cur = safe_simd::vld1_s16(<&[i16; 4]>::try_from(&arr[base..base + 4]).unwrap());
    safe_simd::vst1_s16(
        <&mut [i16; 4]>::try_from(&mut arr[base..base + 4]).unwrap(),
        vadd_s16(cur, v),
    );
}

/// Reverse all eight lanes of a vector.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn rev8(v: int16x8_t) -> int16x8_t {
    // `vrev64q` reverses within each 64-bit half; `vext<4>` then swaps halves.
    let halves = vrev64q_s16(v);
    vextq_s16::<4>(halves, halves)
}

/// Direction search over eight already-normalised rows (`px - 128` per lane).
///
/// Returns the direction and writes the variance, exactly as
/// `cdef_find_dir_rust` does.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn cdef_dir_core(rows: &[int16x8_t; 8], variance: &mut c_uint) -> c_int {
    let mut diag0 = [0i16; 16];
    let mut diag1 = [0i16; 16];
    let mut alt0 = [0i16; 16];
    let mut alt1 = [0i16; 16];
    let mut alt2 = [0i16; 16];
    let mut alt3 = [0i16; 16];
    let mut hv0 = [0i16; 8];
    let mut hv1v = vdupq_n_s16(0);

    for y in 0..8usize {
        let row = rows[y];

        // partial_sum_hv[0][y] += px   (whole row collapses to one entry)
        hv0[y] = vaddvq_s16(row);
        // partial_sum_hv[1][x] += px   (one lane per column)
        hv1v = vaddq_s16(hv1v, row);

        // partial_sum_diag[0][y + x] += px
        acc8(&mut diag0, y, row);
        // partial_sum_diag[1][7 + y - x] += px
        acc8(&mut diag1, y, rev8(row));
        // partial_sum_alt[2][3 - (y >> 1) + x] += px
        acc8(&mut alt2, 3 - (y >> 1), row);
        // partial_sum_alt[3][(y >> 1) + x] += px
        acc8(&mut alt3, y >> 1, row);

        // partial_sum_alt[0][y + (x >> 1)] += px   (adjacent columns pair up)
        let pairs = vget_low_s16(vpaddq_s16(row, row));
        acc4(&mut alt0, y, pairs);
        // partial_sum_alt[1][3 + y - (x >> 1)] += px
        acc4(&mut alt1, y, vrev64_s16(pairs));
    }

    let mut hv1 = [0i16; 8];
    safe_simd::vst1q_s16(&mut hv1, hv1v);

    cdef_dir_cost(
        &diag0,
        &diag1,
        [&alt0, &alt1, &alt2, &alt3],
        &hv0,
        &hv1,
        variance,
    )
}

/// Cost + argmax, byte-identical to `cdef_find_dir_rust`'s tail.
///
/// Kept scalar on purpose — see the note above on the strict-`>` tie-break.
fn cdef_dir_cost(
    diag0: &[i16; 16],
    diag1: &[i16; 16],
    alt: [&[i16; 16]; 4],
    hv0: &[i16; 8],
    hv1: &[i16; 8],
    variance: &mut c_uint,
) -> c_int {
    let mut cost = [0u32; 8];
    for n in 0..8 {
        cost[2] += (hv0[n] as c_int * hv0[n] as c_int) as c_uint;
        cost[6] += (hv1[n] as c_int * hv1[n] as c_int) as c_uint;
    }
    cost[2] *= 105;
    cost[6] *= 105;

    static DIV_TABLE: [u16; 7] = [840, 420, 280, 210, 168, 140, 120];
    for n in 0..7 {
        let d = DIV_TABLE[n] as c_int;
        cost[0] += ((diag0[n] as c_int * diag0[n] as c_int
            + diag0[14 - n] as c_int * diag0[14 - n] as c_int)
            * d) as c_uint;
        cost[4] += ((diag1[n] as c_int * diag1[n] as c_int
            + diag1[14 - n] as c_int * diag1[14 - n] as c_int)
            * d) as c_uint;
    }
    cost[0] += (diag0[7] as c_int * diag0[7] as c_int * 105) as c_uint;
    cost[4] += (diag1[7] as c_int * diag1[7] as c_int * 105) as c_uint;

    for n in 0..4 {
        let cost_ptr = &mut cost[n * 2 + 1];
        for m in 0..5 {
            let v = alt[n][3 + m] as c_int;
            *cost_ptr += (v * v) as c_uint;
        }
        *cost_ptr *= 105;
        for m in 0..3 {
            let d = DIV_TABLE[2 * m + 1] as c_int;
            let a = alt[n][m] as c_int;
            let b = alt[n][10 - m] as c_int;
            *cost_ptr += ((a * a + b * b) * d) as c_uint;
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

#[cfg(target_arch = "aarch64")]
#[arcane]
fn cdef_find_dir_8bpc_neon(_token: Arm64, img: PicOffset, variance: &mut c_uint) -> c_int {
    use crate::include::common::bitdepth::BitDepth8;

    let stride = img.pixel_stride::<BitDepth8>();
    let c128 = vdupq_n_s16(128);
    let mut rows = [vdupq_n_s16(0); 8];
    for y in 0..8usize {
        let row = img + (y as isize * stride);
        // One guard per row, exactly the 8 pixels the search reads.
        let guard = row.slice::<BitDepth8>(8);
        let v = safe_simd::vld1_u8(<&[u8; 8]>::try_from(&guard[..8]).unwrap());
        drop(guard);
        rows[y] = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(v)), c128);
    }
    cdef_dir_core(&rows, variance)
}

#[cfg(target_arch = "aarch64")]
#[arcane]
fn cdef_find_dir_16bpc_neon(
    _token: Arm64,
    img: PicOffset,
    variance: &mut c_uint,
    bitdepth_min_8: c_int,
) -> c_int {
    use crate::include::common::bitdepth::BitDepth16;

    let stride = img.pixel_stride::<BitDepth16>();
    let c128 = vdupq_n_s16(128);
    let neg_shift = vdupq_n_s16(-(bitdepth_min_8 as i16));
    let mut rows = [vdupq_n_s16(0); 8];
    for y in 0..8usize {
        let row = img + (y as isize * stride);
        // One guard per row, exactly the 8 pixels the search reads.
        let guard = row.slice::<BitDepth16>(8);
        let px = safe_simd::vld1q_u16(<&[u16; 8]>::try_from(&guard[..8]).unwrap());
        drop(guard);
        rows[y] = vsubq_s16(vreinterpretq_s16_u16(vshlq_u16(px, neg_shift)), c128);
    }
    cdef_dir_core(&rows, variance)
}

/// Scalar implementation of cdef_find_dir, transcribed from `cdef_find_dir_rust`.
///
/// Reachable only if `Arm64::summon()` fails.
fn cdef_find_dir_scalar<BD: BitDepth>(
    img: PicOffset,
    variance: &mut c_uint,
    bitdepth_min_8: c_int,
) -> c_int {
    let mut partial_sum_hv = [[0i16; 8]; 2];
    let mut diag0 = [0i16; 16];
    let mut diag1 = [0i16; 16];
    let mut alt = [[0i16; 16]; 4];

    for y in 0..8usize {
        let img = img + (y as isize * img.pixel_stride::<BD>());
        let img = &*img.slice::<BD>(8);
        for x in 0..8usize {
            let px = ((img[x].as_::<c_int>() >> bitdepth_min_8) - 128) as i16;

            diag0[y + x] += px;
            alt[0][y + (x >> 1)] += px;
            partial_sum_hv[0][y] += px;
            alt[1][3 + y - (x >> 1)] += px;
            diag1[7 + y - x] += px;
            alt[2][3 - (y >> 1) + x] += px;
            partial_sum_hv[1][x] += px;
            alt[3][(y >> 1) + x] += px;
        }
    }

    cdef_dir_cost(
        &diag0,
        &diag1,
        [&alt[0], &alt[1], &alt[2], &alt[3]],
        &partial_sum_hv[0],
        &partial_sum_hv[1],
        variance,
    )
}

fn cdef_find_dir_8bpc_inner(img: PicOffset, variance: &mut c_uint) -> c_int {
    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken as _;
        if let Some(token) = archmage::Arm64::summon() {
            return cdef_find_dir_8bpc_neon(token, img, variance);
        }
    }
    cdef_find_dir_scalar::<crate::include::common::bitdepth::BitDepth8>(img, variance, 0)
}

fn cdef_find_dir_16bpc_inner(img: PicOffset, variance: &mut c_uint, bitdepth_max: i32) -> c_int {
    let bitdepth_min_8 = bitdepth_min_8_of(bitdepth_max);
    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken as _;
        if let Some(token) = archmage::Arm64::summon() {
            return cdef_find_dir_16bpc_neon(token, img, variance, bitdepth_min_8);
        }
    }
    cdef_find_dir_scalar::<crate::include::common::bitdepth::BitDepth16>(
        img,
        variance,
        bitdepth_min_8,
    )
}

// ============================================================================
// FFI WRAPPERS
// ============================================================================

#[cfg(all(feature = "asm", target_arch = "aarch64"))]
pub unsafe extern "C" fn cdef_filter_8x8_8bpc_neon(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    left: *const [LeftPixelRow2px<DynPixel>; 8],
    top: *const DynPixel,
    bottom: *const DynPixel,
    pri_strength: c_int,
    sec_strength: c_int,
    dir: c_int,
    damping: c_int,
    edges: CdefEdgeFlags,
    _bitdepth_max: c_int,
    dst: *const FFISafe<PicOffset>,
    top_ffi: *const FFISafe<CdefTop>,
    bottom_ffi: *const FFISafe<CdefBottom>,
) {
    let dst = unsafe { *FFISafe::get(dst) };
    let left = unsafe { &*(left as *const [LeftPixelRow2px<u8>; 8]) };
    let top = unsafe { &*FFISafe::get(top_ffi) };
    let bottom = unsafe { &*FFISafe::get(bottom_ffi) };

    cdef_filter_block_8bpc_inner(
        dst,
        left,
        top,
        bottom,
        pri_strength,
        sec_strength,
        dir,
        damping,
        edges,
        8,
        8,
    );
}

#[cfg(all(feature = "asm", target_arch = "aarch64"))]
pub unsafe extern "C" fn cdef_filter_4x8_8bpc_neon(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    left: *const [LeftPixelRow2px<DynPixel>; 8],
    top: *const DynPixel,
    bottom: *const DynPixel,
    pri_strength: c_int,
    sec_strength: c_int,
    dir: c_int,
    damping: c_int,
    edges: CdefEdgeFlags,
    _bitdepth_max: c_int,
    dst: *const FFISafe<PicOffset>,
    top_ffi: *const FFISafe<CdefTop>,
    bottom_ffi: *const FFISafe<CdefBottom>,
) {
    let dst = unsafe { *FFISafe::get(dst) };
    let left = unsafe { &*(left as *const [LeftPixelRow2px<u8>; 8]) };
    let top = unsafe { &*FFISafe::get(top_ffi) };
    let bottom = unsafe { &*FFISafe::get(bottom_ffi) };

    cdef_filter_block_8bpc_inner(
        dst,
        left,
        top,
        bottom,
        pri_strength,
        sec_strength,
        dir,
        damping,
        edges,
        4,
        8,
    );
}

#[cfg(all(feature = "asm", target_arch = "aarch64"))]
pub unsafe extern "C" fn cdef_filter_4x4_8bpc_neon(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    left: *const [LeftPixelRow2px<DynPixel>; 8],
    top: *const DynPixel,
    bottom: *const DynPixel,
    pri_strength: c_int,
    sec_strength: c_int,
    dir: c_int,
    damping: c_int,
    edges: CdefEdgeFlags,
    _bitdepth_max: c_int,
    dst: *const FFISafe<PicOffset>,
    top_ffi: *const FFISafe<CdefTop>,
    bottom_ffi: *const FFISafe<CdefBottom>,
) {
    let dst = unsafe { *FFISafe::get(dst) };
    let left = unsafe { &*(left as *const [LeftPixelRow2px<u8>; 8]) };
    let top = unsafe { &*FFISafe::get(top_ffi) };
    let bottom = unsafe { &*FFISafe::get(bottom_ffi) };

    cdef_filter_block_8bpc_inner(
        dst,
        left,
        top,
        bottom,
        pri_strength,
        sec_strength,
        dir,
        damping,
        edges,
        4,
        4,
    );
}

#[cfg(all(feature = "asm", target_arch = "aarch64"))]
pub unsafe extern "C" fn cdef_find_dir_8bpc_neon_ffi(
    _dst_ptr: *const DynPixel,
    _dst_stride: ptrdiff_t,
    variance: &mut c_uint,
    _bitdepth_max: c_int,
    dst: *const FFISafe<PicOffset>,
) -> c_int {
    let img = *FFISafe::get(dst);
    cdef_find_dir_8bpc_inner(img, variance)
}

#[cfg(all(feature = "asm", target_arch = "aarch64"))]
pub unsafe extern "C" fn cdef_filter_8x8_16bpc_neon(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    left: *const [LeftPixelRow2px<DynPixel>; 8],
    top: *const DynPixel,
    bottom: *const DynPixel,
    pri_strength: c_int,
    sec_strength: c_int,
    dir: c_int,
    damping: c_int,
    edges: CdefEdgeFlags,
    bitdepth_max: c_int,
    dst: *const FFISafe<PicOffset>,
    top_ffi: *const FFISafe<CdefTop>,
    bottom_ffi: *const FFISafe<CdefBottom>,
) {
    let dst = unsafe { *FFISafe::get(dst) };
    let left = unsafe { &*(left as *const [LeftPixelRow2px<u16>; 8]) };
    let top = unsafe { &*FFISafe::get(top_ffi) };
    let bottom = unsafe { &*FFISafe::get(bottom_ffi) };

    cdef_filter_block_16bpc_inner(
        dst,
        left,
        top,
        bottom,
        pri_strength,
        sec_strength,
        dir,
        damping,
        edges,
        8,
        8,
        bitdepth_max,
    );
}

#[cfg(all(feature = "asm", target_arch = "aarch64"))]
pub unsafe extern "C" fn cdef_filter_4x8_16bpc_neon(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    left: *const [LeftPixelRow2px<DynPixel>; 8],
    top: *const DynPixel,
    bottom: *const DynPixel,
    pri_strength: c_int,
    sec_strength: c_int,
    dir: c_int,
    damping: c_int,
    edges: CdefEdgeFlags,
    bitdepth_max: c_int,
    dst: *const FFISafe<PicOffset>,
    top_ffi: *const FFISafe<CdefTop>,
    bottom_ffi: *const FFISafe<CdefBottom>,
) {
    let dst = unsafe { *FFISafe::get(dst) };
    let left = unsafe { &*(left as *const [LeftPixelRow2px<u16>; 8]) };
    let top = unsafe { &*FFISafe::get(top_ffi) };
    let bottom = unsafe { &*FFISafe::get(bottom_ffi) };

    cdef_filter_block_16bpc_inner(
        dst,
        left,
        top,
        bottom,
        pri_strength,
        sec_strength,
        dir,
        damping,
        edges,
        4,
        8,
        bitdepth_max,
    );
}

#[cfg(all(feature = "asm", target_arch = "aarch64"))]
pub unsafe extern "C" fn cdef_filter_4x4_16bpc_neon(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    left: *const [LeftPixelRow2px<DynPixel>; 8],
    top: *const DynPixel,
    bottom: *const DynPixel,
    pri_strength: c_int,
    sec_strength: c_int,
    dir: c_int,
    damping: c_int,
    edges: CdefEdgeFlags,
    bitdepth_max: c_int,
    dst: *const FFISafe<PicOffset>,
    top_ffi: *const FFISafe<CdefTop>,
    bottom_ffi: *const FFISafe<CdefBottom>,
) {
    let dst = unsafe { *FFISafe::get(dst) };
    let left = unsafe { &*(left as *const [LeftPixelRow2px<u16>; 8]) };
    let top = unsafe { &*FFISafe::get(top_ffi) };
    let bottom = unsafe { &*FFISafe::get(bottom_ffi) };

    cdef_filter_block_16bpc_inner(
        dst,
        left,
        top,
        bottom,
        pri_strength,
        sec_strength,
        dir,
        damping,
        edges,
        4,
        4,
        bitdepth_max,
    );
}

#[cfg(all(feature = "asm", target_arch = "aarch64"))]
pub unsafe extern "C" fn cdef_find_dir_16bpc_neon_ffi(
    _dst_ptr: *const DynPixel,
    _dst_stride: ptrdiff_t,
    variance: &mut c_uint,
    bitdepth_max: c_int,
    dst: *const FFISafe<PicOffset>,
) -> c_int {
    let img = *FFISafe::get(dst);
    cdef_find_dir_16bpc_inner(img, variance, bitdepth_max)
}

// ============================================================================
// SAFE DISPATCH WRAPPERS (aarch64)
// ============================================================================

/// Safe dispatch for cdef_filter on aarch64. Returns true if NEON was used.
#[cfg(target_arch = "aarch64")]
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
    // Ablation switch (measurement only; const-false without `__ablate`).
    if crate::src::ablate::is_off(crate::src::ablate::Family::Cdef) {
        return false;
    }
    use crate::include::common::bitdepth::BPC;

    let (w, h) = match variant {
        0 => (8, 8),
        1 => (4, 8),
        _ => (4, 4),
    };

    // Call inner functions directly, bypassing FFI wrappers.
    match BD::BPC {
        BPC::BPC8 => {
            let left: &[LeftPixelRow2px<u8>; 8] =
                crate::src::safe_simd::pixel_access::reinterpret_ref(left)
                    .expect("BD::Pixel layout matches u8");
            cdef_filter_block_8bpc_inner(
                dst,
                left,
                &top,
                &bottom,
                pri_strength,
                sec_strength,
                dir,
                damping,
                edges,
                w,
                h,
            );
        }
        BPC::BPC16 => {
            let left: &[LeftPixelRow2px<u16>; 8] =
                crate::src::safe_simd::pixel_access::reinterpret_ref(left)
                    .expect("BD::Pixel layout matches u16");
            cdef_filter_block_16bpc_inner(
                dst,
                left,
                &top,
                &bottom,
                pri_strength,
                sec_strength,
                dir,
                damping,
                edges,
                w,
                h,
                bd.into_c(),
            );
        }
    }
    true
}

/// Safe dispatch for cdef_find_dir on aarch64. Returns Some(dir).
#[cfg(target_arch = "aarch64")]
pub fn cdef_dir_dispatch<BD: BitDepth>(
    dst: PicOffset,
    variance: &mut c_uint,
    bd: BD,
) -> Option<c_int> {
    // Ablation switch (measurement only; const-false without `__ablate`).
    if crate::src::ablate::is_off(crate::src::ablate::Family::Cdef) {
        return None;
    }
    use crate::include::common::bitdepth::BPC;

    let dir = match BD::BPC {
        BPC::BPC8 => cdef_find_dir_8bpc_inner(dst, variance),
        BPC::BPC16 => cdef_find_dir_16bpc_inner(dst, variance, bd.into_c()),
    };
    Some(dir)
}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;
    use archmage::SimdToken as _;

    /// Faithful transcription of `src/cdef.rs`'s scalar filter, operating on a
    /// plain `[i16]` scratch buffer and a plain destination row. This is the
    /// ORACLE: it is written from the reference, not from the NEON kernel, and
    /// it uses the reference's `i16::MIN` sentinel with unsigned-min /
    /// signed-max, so any semantic drift in the vector path shows up here.
    fn oracle_filter(
        tmp: &[i16; TMP_LEN],
        dst: &mut [i32; 8],
        w: usize,
        y: usize,
        pri_strength: c_int,
        sec_strength: c_int,
        dir: usize,
        damping: c_int,
        bitdepth_min_8: c_int,
    ) {
        let base = TMP_OFFSET + y * TMP_STRIDE;
        let pri_tap0 = 4 - (pri_strength >> bitdepth_min_8 & 1);
        let pri_shift = if pri_strength != 0 {
            cmp::max(0, damping - pri_strength.ilog2() as c_int)
        } else {
            0
        };
        let sec_shift = if sec_strength != 0 {
            damping - sec_strength.ilog2() as c_int
        } else {
            0
        };

        for x in 0..w {
            let px = dst[x];
            let mut sum = 0;
            let mut max = px;
            let mut min = px;
            let bx = (base + x) as isize;
            let mut pri_tap_k = pri_tap0;
            for k in 0..2 {
                if pri_strength != 0 {
                    let off1 = dav1d_cdef_directions[dir + 2][k] as isize;
                    let p0 = tmp[(bx + off1) as usize] as c_int;
                    let p1 = tmp[(bx - off1) as usize] as c_int;
                    sum += pri_tap_k * constrain_scalar(p0 - px, pri_strength, pri_shift);
                    sum += pri_tap_k * constrain_scalar(p1 - px, pri_strength, pri_shift);
                    pri_tap_k = pri_tap_k & 3 | 2;
                    if sec_strength != 0 {
                        min = cmp::min(p0 as c_uint, min as c_uint) as c_int;
                        max = cmp::max(p0, max);
                        min = cmp::min(p1 as c_uint, min as c_uint) as c_int;
                        max = cmp::max(p1, max);
                    }
                }
                if sec_strength != 0 {
                    let off2 = dav1d_cdef_directions[dir + 4][k] as isize;
                    let off3 = dav1d_cdef_directions[dir][k] as isize;
                    let s = [
                        tmp[(bx + off2) as usize] as c_int,
                        tmp[(bx - off2) as usize] as c_int,
                        tmp[(bx + off3) as usize] as c_int,
                        tmp[(bx - off3) as usize] as c_int,
                    ];
                    let sec_tap = 2 - k as c_int;
                    for &v in &s {
                        sum += sec_tap * constrain_scalar(v - px, sec_strength, sec_shift);
                        min = cmp::min(v as c_uint, min as c_uint) as c_int;
                        max = cmp::max(v, max);
                    }
                }
            }
            let v = px + (sum - (sum < 0) as c_int + 8 >> 4);
            dst[x] = if sec_strength != 0 {
                iclip(v, min, max)
            } else {
                v
            };
        }
    }

    /// Run the NEON row kernel over an `i16` scratch buffer.
    fn neon_filter(
        tmp: &[i16; TMP_LEN],
        dst: &[i32; 8],
        y: usize,
        pri_strength: c_int,
        sec_strength: c_int,
        dir: usize,
        damping: c_int,
        bitdepth_min_8: c_int,
    ) -> [i16; 8] {
        let token = Arm64::summon().expect("aarch64 always has NEON");
        let mut utmp = [0u16; TMP_LEN];
        for i in 0..TMP_LEN {
            utmp[i] = tmp[i] as u16;
        }
        let mut px = [0u16; 8];
        for x in 0..8 {
            px[x] = dst[x] as u16;
        }
        let p = filter_params(pri_strength, sec_strength, damping, bitdepth_min_8);
        run_row(
            token,
            &utmp,
            &px,
            y,
            dir,
            &p,
            pri_strength != 0,
            sec_strength != 0,
        )
    }

    #[arcane]
    fn run_row(
        _token: Arm64,
        tmp: &[u16; TMP_LEN],
        px: &[u16; 8],
        y: usize,
        dir: usize,
        p: &FilterParams,
        pri: bool,
        sec: bool,
    ) -> [i16; 8] {
        let base = TMP_OFFSET + y * TMP_STRIDE;
        let pxv = safe_simd::vld1q_u16(px);
        let pt = vdupq_n_u16(p.pri_threshold);
        let pn = vdupq_n_s16(p.pri_neg_shift);
        let st = vdupq_n_u16(p.sec_threshold);
        let sn = vdupq_n_s16(p.sec_neg_shift);
        let out = match (pri, sec) {
            (true, true) => {
                cdef_filter_row::<true, true>(tmp, base, pxv, dir, pt, pn, st, sn, p.pri_tap)
            }
            (true, false) => {
                cdef_filter_row::<true, false>(tmp, base, pxv, dir, pt, pn, st, sn, p.pri_tap)
            }
            (false, true) => {
                cdef_filter_row::<false, true>(tmp, base, pxv, dir, pt, pn, st, sn, p.pri_tap)
            }
            (false, false) => vreinterpretq_s16_u16(pxv),
        };
        let mut o = [0i16; 8];
        safe_simd::vst1q_s16(&mut o, out);
        o
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            (self.0 >> 33) as u32
        }
        fn below(&mut self, n: u32) -> u32 {
            self.next() % n
        }
    }

    /// The strengths and dampings `src/cdef_apply.rs` can actually produce.
    fn param_space(bitdepth_min_8: c_int) -> (Vec<c_int>, Vec<c_int>, Vec<c_int>) {
        // `adj_y_pri_lvl = adjust_strength((y_lvl >> 2) << bd_min_8, var)`, whose
        // range is 0..=15 << bd_min_8 (the `* (4 + i) + 8 >> 4` factor is <= 1).
        let pri: Vec<c_int> = (0..=(15 << bitdepth_min_8)).collect();
        // `y_sec_lvl` is one of {0, 1, 2, 4} << bd_min_8.
        let sec: Vec<c_int> = [0, 1, 2, 4].iter().map(|v| v << bitdepth_min_8).collect();
        // `damping` is `frame_hdr.cdef.damping + bd_min_8`, and chroma uses -1.
        let damping: Vec<c_int> = (2 + bitdepth_min_8..=6 + bitdepth_min_8).collect();
        (pri, sec, damping)
    }

    /// The sentinel must neutralise itself in `constrain` for every parameter
    /// the spec can signal — this is what lets the wrapped signed difference of
    /// a `0x8000` tap never escape into `sum`.
    #[test]
    fn constrain_neutralises_sentinel_over_full_param_space() {
        for bd_min_8 in [0, 2, 4] {
            let bitdepth_max = (1 << (8 + bd_min_8)) - 1;
            let (pri, sec, dampings) = param_space(bd_min_8);
            for &damping in &dampings {
                for strengths in [&pri, &sec] {
                    for &strength in strengths.iter() {
                        if strength == 0 {
                            continue;
                        }
                        let shift = cmp::max(0, damping - strength.ilog2() as c_int);
                        for px in [0, 1, 128, bitdepth_max / 2, bitdepth_max] {
                            // Reference: `tmp` holds i16::MIN, diff = -32768 - px.
                            let scalar = constrain_scalar(-32768 - px, strength, shift);
                            // NEON: `vabdq_u16(0x8000, px)` = 32768 - px, then
                            // an unsigned saturating subtract.
                            let adiff_neon = (32768u32 - px as u32) as u16;
                            let clip_neon =
                                (strength as u16).saturating_sub(adiff_neon >> shift.min(15));
                            assert_eq!(
                                scalar, 0,
                                "scalar constrain of a sentinel tap must be 0 \
                                 (strength={strength} damping={damping} shift={shift} px={px})"
                            );
                            assert_eq!(
                                clip_neon, 0,
                                "NEON clip of a sentinel tap must be 0 \
                                 (strength={strength} damping={damping} shift={shift} px={px})"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Randomised differential test of the row kernel against the oracle, with
    /// sentinels placed in every padding shape CDEF can see.
    fn differential_rows(bd_min_8: c_int, seed: u64, iters: usize) {
        let bitdepth_max = (1i32 << (8 + bd_min_8)) - 1;
        let (pri_space, sec_space, damp_space) = param_space(bd_min_8);
        let mut rng = Rng(seed);
        let mut checked = 0usize;

        for _ in 0..iters {
            let w = if rng.next() & 1 == 0 { 4 } else { 8 };
            let h = if w == 4 && rng.next() & 1 == 0 { 4 } else { 8 };
            // Random edge availability drives where the sentinels land.
            let have_top = rng.next() & 1 == 0;
            let have_bottom = rng.next() & 1 == 0;
            let have_left = rng.next() & 1 == 0;
            let have_right = rng.next() & 1 == 0;

            let mut tmp = [i16::MIN; TMP_LEN];
            // Real pixels: the block, plus whichever context is available.
            let y_lo = if have_top { 0 } else { 2 };
            let y_hi = if have_bottom { h + 4 } else { h + 2 };
            let x_lo = if have_left { 0 } else { 2 };
            let x_hi = if have_right { w + 4 } else { w + 2 };
            for yy in y_lo..y_hi {
                for xx in x_lo..x_hi {
                    // Bias hard toward flat/extreme content: a binding upper
                    // clip needs neighbours close together and a big delta.
                    let v = match rng.below(4) {
                        0 => 0,
                        1 => bitdepth_max,
                        2 => (bitdepth_max / 2) + rng.below(3) as i32 - 1,
                        _ => rng.below(bitdepth_max as u32 + 1) as i32,
                    };
                    tmp[yy * TMP_STRIDE + xx] = v as i16;
                }
            }

            let pri = pri_space[rng.below(pri_space.len() as u32) as usize];
            let sec = sec_space[rng.below(sec_space.len() as u32) as usize];
            if pri == 0 && sec == 0 {
                continue;
            }
            let damping = damp_space[rng.below(damp_space.len() as u32) as usize];
            let dir = rng.below(8) as usize;

            for y in 0..h {
                let mut dst = [0i32; 8];
                for x in 0..8 {
                    dst[x] = tmp[(y + 2) * TMP_STRIDE + 2 + x] as i32;
                }
                // Lanes at and beyond `w` are left exactly as `tmp` holds them,
                // sentinel included — that is what the kernel really sees,
                // because it now takes `px` from `tmp` rather than re-reading
                // `dst`. They are discarded by the `w`-wide store; the point is
                // that a `0x8000` pixel lane must not perturb the lanes that
                // are kept.
                let neon = neon_filter(&tmp, &dst, y, pri, sec, dir, damping, bd_min_8);
                let mut expect = dst;
                oracle_filter(&tmp, &mut expect, w, y, pri, sec, dir, damping, bd_min_8);
                for x in 0..w {
                    let got = if bd_min_8 == 0 {
                        neon[x] as u16 as u8 as i32
                    } else {
                        neon[x] as u16 as i32
                    };
                    let want = if bd_min_8 == 0 {
                        expect[x] as u8 as i32
                    } else {
                        expect[x] as u16 as i32
                    };
                    assert_eq!(
                        got, want,
                        "row mismatch bd_min_8={bd_min_8} w={w} h={h} y={y} x={x} \
                         pri={pri} sec={sec} dir={dir} damping={damping} \
                         edges=T{have_top}B{have_bottom}L{have_left}R{have_right}"
                    );
                    checked += 1;
                }
            }
        }
        assert!(checked > 1000, "test did not exercise anything: {checked}");
    }

    #[test]
    fn neon_row_matches_oracle_8bpc() {
        differential_rows(0, 0x1234_5678_9abc_def0, 4000);
    }

    #[test]
    fn neon_row_matches_oracle_10bpc() {
        differential_rows(2, 0x0fed_cba9_8765_4321, 4000);
    }

    #[test]
    fn neon_row_matches_oracle_12bpc() {
        differential_rows(4, 0xdead_beef_cafe_f00d, 4000);
    }

    /// The `max` accumulated over the taps with the OLD sentinel convention
    /// (`8191`, folded into `max` with a signed compare). Used only to prove
    /// that the old convention was a REAL divergence, not a theoretical one.
    fn oracle_filter_old_sentinel(
        tmp: &[i16; TMP_LEN],
        dst: &mut [i32; 8],
        w: usize,
        y: usize,
        pri_strength: c_int,
        sec_strength: c_int,
        dir: usize,
        damping: c_int,
    ) {
        let base = TMP_OFFSET + y * TMP_STRIDE;
        // The old kernel stored the sentinel as 8191 in a `u16` scratch and
        // read every tap as a non-negative `i32`, so `max` saw 8191.
        let tap = |i: usize| -> c_int {
            let v = tmp[i];
            if v == i16::MIN { 8191 } else { v as c_int }
        };
        let pri_tap0 = 4 - (pri_strength & 1);
        let pri_shift = if pri_strength != 0 {
            cmp::max(0, damping - pri_strength.ilog2() as c_int)
        } else {
            0
        };
        let sec_shift = if sec_strength != 0 {
            damping - sec_strength.ilog2() as c_int
        } else {
            0
        };

        for x in 0..w {
            let px = dst[x];
            let mut sum = 0;
            let mut max = px;
            let mut min = px;
            let bx = (base + x) as isize;
            let mut pri_tap_k = pri_tap0;
            for k in 0..2 {
                if pri_strength != 0 {
                    let off1 = dav1d_cdef_directions[dir + 2][k] as isize;
                    let p0 = tap((bx + off1) as usize);
                    let p1 = tap((bx - off1) as usize);
                    sum += pri_tap_k * constrain_scalar(p0 - px, pri_strength, pri_shift);
                    sum += pri_tap_k * constrain_scalar(p1 - px, pri_strength, pri_shift);
                    pri_tap_k = pri_tap_k & 3 | 2;
                    if sec_strength != 0 {
                        min = cmp::min(cmp::min(p0, p1), min);
                        max = cmp::max(cmp::max(p0, p1), max);
                    }
                }
                if sec_strength != 0 {
                    let off2 = dav1d_cdef_directions[dir + 4][k] as isize;
                    let off3 = dav1d_cdef_directions[dir][k] as isize;
                    let s = [
                        tap((bx + off2) as usize),
                        tap((bx - off2) as usize),
                        tap((bx + off3) as usize),
                        tap((bx - off3) as usize),
                    ];
                    let sec_tap = 2 - k as c_int;
                    for &v in &s {
                        sum += sec_tap * constrain_scalar(v - px, sec_strength, sec_shift);
                        min = cmp::min(v, min);
                        max = cmp::max(v, max);
                    }
                }
            }
            let v = px + (sum - (sum < 0) as c_int + 8 >> 4);
            dst[x] = if sec_strength != 0 {
                iclip(v, min, max)
            } else {
                v
            };
        }
    }

    /// A sentinel tap must not raise `max`.
    ///
    /// This searches near-flat 8bpc content — the only shape where the upper
    /// half of the `iclip` can bind — for a case the pre-2026-08-07 kernel got
    /// wrong. It asserts BOTH that such a case exists (so the fix is not
    /// theoretical) and that the current kernel matches the reference on it.
    #[test]
    fn sentinel_must_not_raise_max() {
        let mut rng = Rng(0xa5a5_0f0f_1234_9999);
        let mut divergences = 0usize;
        let mut first: Option<String> = None;

        for _ in 0..200_000 {
            // Near-flat content: taps within a couple of steps of the pixel is
            // what lets `px + delta` overshoot the true max.
            let dc = 8 + rng.below(240) as i32;
            let spread = 1 + rng.below(4) as i32;
            // Some context missing, so sentinels are in the tap set.
            let have_top = rng.next() & 1 == 0;
            let have_bottom = rng.next() & 1 == 0;
            let have_left = rng.next() & 1 == 0;
            let have_right = rng.next() & 1 == 0;
            if have_top && have_bottom && have_left && have_right {
                continue;
            }
            let (w, h) = (8usize, 8usize);
            let mut tmp = [i16::MIN; TMP_LEN];
            let y_lo = if have_top { 0 } else { 2 };
            let y_hi = if have_bottom { h + 4 } else { h + 2 };
            let x_lo = if have_left { 0 } else { 2 };
            let x_hi = if have_right { w + 4 } else { w + 2 };
            for yy in y_lo..y_hi {
                for xx in x_lo..x_hi {
                    tmp[yy * TMP_STRIDE + xx] = (dc + rng.below(spread as u32 + 1) as i32) as i16;
                }
            }

            let pri = rng.below(16) as c_int;
            let sec = [0, 1, 2, 4][rng.below(4) as usize];
            if pri == 0 && sec == 0 {
                continue;
            }
            let damping = 2 + rng.below(5) as c_int;
            let dir = rng.below(8) as usize;

            for y in 0..h {
                let mut dst = [0i32; 8];
                for x in 0..8 {
                    dst[x] = tmp[(y + 2) * TMP_STRIDE + 2 + x] as i32;
                }
                let neon = neon_filter(&tmp, &dst, y, pri, sec, dir, damping, 0);
                let mut expect = dst;
                oracle_filter(&tmp, &mut expect, w, y, pri, sec, dir, damping, 0);
                let mut old = dst;
                oracle_filter_old_sentinel(&tmp, &mut old, w, y, pri, sec, dir, damping);
                for x in 0..w {
                    assert_eq!(
                        neon[x] as u16 as u8 as i32, expect[x] as u8 as i32,
                        "sentinel raised max at y={y} x={x} pri={pri} sec={sec} \
                         dir={dir} damping={damping}"
                    );
                    if old[x] != expect[x] {
                        divergences += 1;
                        if first.is_none() {
                            first = Some(format!(
                                "y={y} x={x} pri={pri} sec={sec} dir={dir} damping={damping} \
                                 edges=T{have_top}B{have_bottom}L{have_left}R{have_right} \
                                 old={} ref={}",
                                old[x], expect[x]
                            ));
                        }
                    }
                }
            }
            if divergences > 0 && first.is_some() {
                // One confirmed divergence is enough to make the point; keep
                // going a bit so the count is meaningful, then stop.
                if divergences > 32 {
                    break;
                }
            }
        }

        assert!(
            divergences > 0,
            "the 8191-sentinel convention never diverged from the reference in \
             this search — if that holds up, the max fix is a simplification, \
             not a bug fix, and this test should say so"
        );
        eprintln!(
            "8191-sentinel convention diverges from the reference; first case: {}",
            first.unwrap()
        );
    }

    /// Direction search: NEON accumulation vs the scalar reference, including
    /// the flat/tied content that stresses the strict-`>` tie-break.
    #[test]
    fn neon_dir_matches_scalar() {
        let token = Arm64::summon().expect("aarch64 always has NEON");
        let mut rng = Rng(0xfeed_face_0000_0001);

        let mut cases: Vec<[[i32; 8]; 8]> = Vec::new();
        // All-flat at several levels — every cost is equal, so the tie-break
        // decides, and it must decide 0.
        for level in [0, 1, 128, 254, 255] {
            cases.push([[level; 8]; 8]);
        }
        // Pure horizontal / vertical / diagonal edges: exact ties between
        // symmetric direction pairs.
        let mut horiz = [[0i32; 8]; 8];
        let mut vert = [[0i32; 8]; 8];
        let mut diag = [[0i32; 8]; 8];
        let mut anti = [[0i32; 8]; 8];
        for y in 0..8 {
            for x in 0..8 {
                horiz[y][x] = if y < 4 { 0 } else { 255 };
                vert[y][x] = if x < 4 { 0 } else { 255 };
                diag[y][x] = if x >= y { 255 } else { 0 };
                anti[y][x] = if x + y >= 7 { 255 } else { 0 };
            }
        }
        cases.extend([horiz, vert, diag, anti]);
        // Two-level checkerboards and near-flat blocks: dense ties.
        for step in [1u32, 2, 4] {
            let mut c = [[0i32; 8]; 8];
            for y in 0..8 {
                for x in 0..8 {
                    c[y][x] = 128 + (((x as u32 / step + y as u32 / step) & 1) as i32);
                }
            }
            cases.push(c);
        }
        // Random content, including deliberately low-amplitude.
        for _ in 0..3000 {
            let mut c = [[0i32; 8]; 8];
            let amp = 1 + rng.below(255);
            let dc = rng.below(256 - amp);
            for y in 0..8 {
                for x in 0..8 {
                    c[y][x] = (dc + rng.below(amp + 1)) as i32;
                }
            }
            cases.push(c);
        }

        let mut ties = 0usize;
        for (i, case) in cases.iter().enumerate() {
            for bd_min_8 in [0, 2, 4] {
                let shift = bd_min_8;
                let mut rows = [[0u16; 8]; 8];
                for y in 0..8 {
                    for x in 0..8 {
                        rows[y][x] = (case[y][x] << shift) as u16;
                    }
                }
                let (dir_scalar, var_scalar) = dir_oracle(&rows, bd_min_8);
                let mut var_neon = 0u32;
                let dir_neon = run_dir(token, &rows, bd_min_8, &mut var_neon);
                assert_eq!(
                    (dir_neon, var_neon),
                    (dir_scalar, var_scalar),
                    "dir mismatch case {i} bd_min_8={bd_min_8}"
                );
                if dir_scalar == 0 {
                    ties += 1;
                }
            }
        }
        assert!(ties > 0, "no tie-break case exercised");
    }

    #[arcane]
    fn run_dir(
        _token: Arm64,
        rows: &[[u16; 8]; 8],
        bitdepth_min_8: c_int,
        variance: &mut c_uint,
    ) -> c_int {
        let c128 = vdupq_n_s16(128);
        let neg_shift = vdupq_n_s16(-(bitdepth_min_8 as i16));
        let mut v = [vdupq_n_s16(0); 8];
        for y in 0..8 {
            let px = safe_simd::vld1q_u16(&rows[y]);
            v[y] = vsubq_s16(vreinterpretq_s16_u16(vshlq_u16(px, neg_shift)), c128);
        }
        cdef_dir_core(&v, variance)
    }

    /// Direct transcription of `cdef_find_dir_rust`, in i32 as it is written.
    fn dir_oracle(rows: &[[u16; 8]; 8], bitdepth_min_8: c_int) -> (c_int, c_uint) {
        let mut partial_sum_hv = [[0i32; 8]; 2];
        let mut partial_sum_diag = [[0i32; 15]; 2];
        let mut partial_sum_alt = [[0i32; 11]; 4];

        for y in 0..8 {
            for x in 0..8 {
                let px = ((rows[y][x] as c_int) >> bitdepth_min_8) - 128;
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
        (best_dir as c_int, (best_cost - cost[best_dir ^ 4]) >> 10)
    }
}
