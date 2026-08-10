//! aarch64 NEON kernels for the deblocking loop filter.
//!
//! The loop filter removes blocking artifacts at transform-block boundaries:
//! for each 4-pixel group along an edge it decides, per pixel, whether to
//! filter at all and with which of four tap widths, then applies that filter
//! across the edge.
//!
//! Everything here operates on the COMPACT SCRATCH rectangle that
//! `src/loopfilter.rs`'s `LfBlock` opens — never on picture memory. That is
//! the point of the seam: `LfBlock::open` and `close` keep sole ownership of
//! which picture pixels are guarded and written, so a kernel change can never
//! be a `DisjointMut` change. See [`lf_compact_run_neon`].
//!
//! History worth knowing: until 2026-08-07 this file said "Safe ARM NEON
//! implementations" and imported `core::arch::aarch64::*` but contained no
//! intrinsic CALL, and its `loopfilter_sb_dispatch` returned false — so the
//! aarch64 "NEON" loop filter was the scalar reference, and 600 lines of
//! never-called scalar `loop_filter_core` / `lpf_*_sb_inner` / `extern "C"`
//! wrappers sat here looking like the reference implementation. They were not
//! reachable (the `asm` build links `src/arm/64/loopfilter.S`, whose symbols
//! are `dav1d_`-prefixed) and their 6-tap formula did not even match
//! `src/loopfilter.rs`'s. Deleted. **The reference for this filter is
//! `src/loopfilter.rs::loop_filter`** — nothing else.

#![cfg_attr(not(feature = "unchecked"), forbid(unsafe_code))]
#![cfg_attr(feature = "unchecked", deny(unsafe_code))]
#![allow(clippy::too_many_arguments)]
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
use archmage::{Arm64, SimdToken as _, arcane, rite};
#[cfg(target_arch = "aarch64")]
use safe_unaligned_simd::aarch64 as safe_simd;

#[cfg(target_arch = "aarch64")]
use crate::src::loopfilter::{LF_BATCH_H, LF_BATCH_V, LF_BLOCK_LEN, LF_SW_H, LF_SW_V};

use crate::include::common::bitdepth::BitDepth;
use crate::include::dav1d::picture::PicOffset;
use crate::src::align::Align16;
use crate::src::lf_mask::Av1FilterLUT;
use crate::src::with_offset::WithOffset;
use std::sync::atomic::AtomicU8;
// Used only by `loopfilter_sb_dispatch`, which is aarch64-only.
#[cfg(target_arch = "aarch64")]
#[allow(non_camel_case_types)]
type ptrdiff_t = isize;
use std::ffi::c_int;

/// Whole-superblock-edge dispatch for `loopfilter_sb` on aarch64.
///
/// Always returns false, and that is not a TODO. aarch64's NEON tier is wired
/// one level DOWN, at [`lf_compact_run_neon`]: `loop_filter_sb128_rust` already
/// resolves every 4-pixel group's parameters, fuses consecutive same-`wd`
/// groups, and reads the fused rectangle into a compact scratch under one guard
/// per picture row. Re-deriving all of that here — which is what the x86
/// dispatch does, because on x86 that machinery is the SIMD path's own — would
/// duplicate the guard policy in two places for no gain.
///
/// Returning false hands control to `loop_filter_sb128_rust`, whose
/// `LfBlock::filter_run` calls the kernels. The scalar `loop_filter` still runs
/// where the kernels decline: the picture-edge case where even a single group's
/// rectangle leaves the plane and the filter falls back to `DirectTaps`.
///
/// Keep the signature: `src/loopfilter.rs` calls it under
/// `#[cfg(target_arch = "aarch64")]` alongside the x86 and wasm dispatches.
#[cfg(target_arch = "aarch64")]
pub fn loopfilter_sb_dispatch<BD: BitDepth>(
    _dst: PicOffset,
    _stride: ptrdiff_t,
    _mask: &[u32; 3],
    _lvl: WithOffset<&[AtomicU8]>,
    _b4_stride: isize,
    _lut: &Align16<Av1FilterLUT>,
    _w: c_int,
    _bitdepth_max: c_int,
    _is_y: bool,
    _is_v: bool,
) -> bool {
    // Ablation switch (measurement only; const-false without `__ablate`).
    if crate::src::ablate::is_off(crate::src::ablate::Family::LoopFilter) {
        return false;
    }
    false
}

// ============================================================================
// NEON DEBLOCKING LOOP FILTER
// ============================================================================
//
// These kernels replace the per-column scalar `loop_filter` inside the compact
// scratch rectangle that `src/loopfilter.rs` already opens under one guard per
// picture row. NOTHING about which picture pixels are guarded or written back
// changes here: `LfBlock::open` and `LfBlock::close` are untouched and the
// kernel only ever touches `LfScratch::buf`, a plain `[u16; 256]` array with no
// borrow tracking on it at all.
//
// ## Layout
//
// The scratch is a padded rectangle whose row stride is per DIRECTION:
// `LF_SW_V` = 128 for V (a fused run is `4 * groups` wide, up to
// `4 * LF_BATCH_V`), `LF_SW_H` = 16 for H (whose runs grow DOWN the rectangle,
// so its rows never need to be wider than `2 * reach`).
//
//   V (`HV::V`, horizontal edges, `deblock_rows`): tap-major. Tap `k` of lane
//   `j` is `buf[base + k * LF_SW_V + j]`, so ALL lanes of one tap are contiguous
//   — a plain `vld1q_u16`, no transpose.
//
//   H (`HV::H`, vertical edges, `deblock_cols`): lane-major. Tap `k` of lane
//   `j` is `buf[base + j * LF_SW_H + k]`, so the kernel transposes 8x8 `u16`
//   tiles in registers (the structure dav1d's `lpf_h_sb_*` asm uses), filters,
//   and transposes back. Padding the scratch row stride to a fixed 16 is what
//   makes each row exactly two 8-lane loads at every filter width.
//
// ## Why `u16` at every bit depth
//
// One kernel serves 8, 10 and 12 bits. Every quantity the filter forms fits
// `u16` unsigned: pixels (<= 4095), the `e` comparison `2*|p0-q0| + |p1-q1|/2`
// (<= 10237), and the flat filters' weighted sums, whose weights total 16 and
// whose worst case is `16 * 4095 + 8 = 65528`. The narrow filter needs signs,
// so it runs in `i16` on the same registers (`3*(q0-p0) + f` peaks at 14333).
//
// The wide filter accumulates by RECURRENCE (`s += a + b - c - d`), which can
// transiently exceed `u16` — that is fine and deliberate: `u16` add/sub are
// modular, every partial sum is congruent to the true value mod 2^16, and the
// true value is back under 65520 at each point the accumulator is read.

/// Thresholds for one fused run.
///
/// `e`/`i`/`h` come from the filter LUT and CHANGE PER GROUP inside a fused
/// run — the run only fuses on matching `wd` — so they are PER GROUP, not
/// splats. Held as `NG`-entry arrays (one per fused group) rather than
/// sixteen-entry per-lane ones: a group is four adjacent lanes, so [`lane_thr`]
/// expands two groups into an 8-lane vector in two instructions and the run
/// never touches memory for them.
///
/// **`NG` is per DIRECTION, and that is a measured requirement, not tidiness.**
/// A first cut of the V batch lift dropped these arrays and had `lane_thr` read
/// `params.get(..)` directly, to avoid materialising 32 entries per run. That
/// put a slice bounds check and an `Option` branch inside the 8-lane chunk loop
/// of BOTH kernels, and the H pass — 69.3% of the work, which gains nothing
/// from a wider batch — paid it for nothing. Measured at
/// `benchmarks/lf_vbatch_2026-08-10_v1.tsv`: 8bpc t=1 +3.0%, t=8 +7.9% against
/// base. `LF_GROUPS_H` = 4 keeps H's codegen identical to before the batch
/// change.
#[cfg(target_arch = "aarch64")]
pub(crate) struct LfLaneThresholds<const NG: usize> {
    e: [u16; NG],
    i: [u16; NG],
    h: [u16; NG],
    /// `1 << bitdepth_min_8`, the flatness threshold.
    f: u16,
    bd_max: u16,
    /// `(128 << bitdepth_min_8) - 1`; the low clip is `-(hi + 1)`.
    clip_hi: i16,
}

/// Fused groups the H kernel can see — [`LF_BATCH_H`], because H's rectangle
/// grows in the ROW direction and fusing further cannot reduce its guard count.
#[cfg(target_arch = "aarch64")]
const LF_GROUPS_H: usize = LF_BATCH_H;

/// Fused groups the V kernel can see — [`LF_BATCH_V`].
///
/// This used to be asserted equal to the batch constant, which is what blocked
/// the V pass's batch from rising at all. The GUARD batch and the KERNEL batch
/// are different concerns: this bounds the threshold table the 8-lane chunk
/// loop indexes, the batch bounds how many groups one borrow registration
/// covers. They are equal here only because a V run's lanes and its groups are
/// the same population.
#[cfg(target_arch = "aarch64")]
const LF_GROUPS_V: usize = LF_BATCH_V;
#[cfg(target_arch = "aarch64")]
const _: () = assert!(4 * LF_GROUPS_V <= LF_SW_V);
#[cfg(target_arch = "aarch64")]
const _: () = assert!(4 * LF_GROUPS_H <= LF_SW_H);

/// Tap reach of a filter width. MUST mirror `src::loopfilter::lf_reach`.
#[cfg(target_arch = "aarch64")]
const fn reach_of(wd: c_int) -> usize {
    if wd >= 16 {
        7
    } else if wd > 6 {
        4
    } else if wd > 4 {
        3
    } else {
        2
    }
}

/// Load 8 lanes at `idx`, widening from the scratch's pixel width.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn ld8_u8(buf: &[u8; LF_BLOCK_LEN], idx: usize) -> uint16x8_t {
    vmovl_u8(safe_simd::vld1_u8(
        <&[u8; 8]>::try_from(&buf[idx..idx + 8]).unwrap(),
    ))
}

/// Store 8 lanes at `idx`. The narrow is exact: every value the filter writes
/// has already been clipped to `[0, bitdepth_max]`, which is the same
/// truncation the scalar's `as_::<BD::Pixel>()` performs.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn st8_u8(buf: &mut [u8; LF_BLOCK_LEN], idx: usize, v: uint16x8_t) {
    safe_simd::vst1_u8(
        <&mut [u8; 8]>::try_from(&mut buf[idx..idx + 8]).unwrap(),
        vmovn_u16(v),
    );
}

/// Load 8 lanes at `idx`.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn ld8_u16(buf: &[u16; LF_BLOCK_LEN], idx: usize) -> uint16x8_t {
    safe_simd::vld1q_u16(<&[u16; 8]>::try_from(&buf[idx..idx + 8]).unwrap())
}

/// Store 8 lanes at `idx`.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn st8_u16(buf: &mut [u16; LF_BLOCK_LEN], idx: usize, v: uint16x8_t) {
    safe_simd::vst1q_u16(
        <&mut [u16; 8]>::try_from(&mut buf[idx..idx + 8]).unwrap(),
        v,
    );
}

/// Standard 8x8 `u16` transpose: `out[r]` lane `c` = `inp[c]` lane `r`.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn transpose8(m: [uint16x8_t; 8]) -> [uint16x8_t; 8] {
    // Level 1: 16-bit granularity.
    let a0 = vtrn1q_u16(m[0], m[1]);
    let a1 = vtrn2q_u16(m[0], m[1]);
    let a2 = vtrn1q_u16(m[2], m[3]);
    let a3 = vtrn2q_u16(m[2], m[3]);
    let a4 = vtrn1q_u16(m[4], m[5]);
    let a5 = vtrn2q_u16(m[4], m[5]);
    let a6 = vtrn1q_u16(m[6], m[7]);
    let a7 = vtrn2q_u16(m[6], m[7]);

    // Level 2: 32-bit granularity.
    let r = |x| vreinterpretq_u32_u16(x);
    let b0 = vtrn1q_u32(r(a0), r(a2));
    let b2 = vtrn2q_u32(r(a0), r(a2));
    let b1 = vtrn1q_u32(r(a1), r(a3));
    let b3 = vtrn2q_u32(r(a1), r(a3));
    let b4 = vtrn1q_u32(r(a4), r(a6));
    let b6 = vtrn2q_u32(r(a4), r(a6));
    let b5 = vtrn1q_u32(r(a5), r(a7));
    let b7 = vtrn2q_u32(r(a5), r(a7));

    // Level 3: 64-bit granularity.
    let s = |x| vreinterpretq_u64_u32(x);
    let t = |x| vreinterpretq_u16_u64(x);
    [
        t(vtrn1q_u64(s(b0), s(b4))),
        t(vtrn1q_u64(s(b1), s(b5))),
        t(vtrn1q_u64(s(b2), s(b6))),
        t(vtrn1q_u64(s(b3), s(b7))),
        t(vtrn2q_u64(s(b0), s(b4))),
        t(vtrn2q_u64(s(b1), s(b5))),
        t(vtrn2q_u64(s(b2), s(b6))),
        t(vtrn2q_u64(s(b3), s(b7))),
    ]
}

/// `min(max(v, 0), bd_max)` in the signed domain, i.e. `BitDepth::iclip_pixel`.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn clip_px(v: int16x8_t, bd_max: int16x8_t) -> uint16x8_t {
    vreinterpretq_u16_s16(vminq_s16(vmaxq_s16(v, vdupq_n_s16(0)), bd_max))
}

/// The whole filter for one 8-lane chunk, in place on `t`.
///
/// `t[n]` is tap `n - 7`, so `t[6]` is `p0` and `t[7]` is `q0`. Taps outside
/// `[-reach, reach)` are NOT read — the caller leaves them zeroed, because for
/// the V direction their scratch index would be negative.
///
/// Every branch of the scalar `loop_filter` ladder is evaluated and blended;
/// the ladder's `else if` chain becomes disjoint lane masks. A lane whose `fm`
/// is clear keeps its input taps, which is the vector form of the scalar's
/// `continue`.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn lf_core<const WD: c_int>(
    t: &mut [uint16x8_t; 14],
    e: uint16x8_t,
    i: uint16x8_t,
    hthr: uint16x8_t,
    f: uint16x8_t,
    bd_max: int16x8_t,
    clip_hi: int16x8_t,
) {
    let p6 = t[0];
    let p5 = t[1];
    let p4 = t[2];
    let p3 = t[3];
    let p2 = t[4];
    let p1 = t[5];
    let p0 = t[6];
    let q0 = t[7];
    let q1 = t[8];
    let q2 = t[9];
    let q3 = t[10];
    let q4 = t[11];
    let q5 = t[12];
    let q6 = t[13];

    // ---- filter mask -------------------------------------------------
    let ad_p1p0 = vabdq_u16(p1, p0);
    let ad_q1q0 = vabdq_u16(q1, q0);
    let ad_p0q0 = vabdq_u16(p0, q0);
    let ad_p1q1 = vabdq_u16(p1, q1);

    let mut fm = vandq_u16(vcleq_u16(ad_p1p0, i), vcleq_u16(ad_q1q0, i));
    let ecmp = vaddq_u16(vaddq_u16(ad_p0q0, ad_p0q0), vshrq_n_u16::<1>(ad_p1q1));
    fm = vandq_u16(fm, vcleq_u16(ecmp, e));

    if WD > 4 {
        fm = vandq_u16(fm, vcleq_u16(vabdq_u16(p2, p1), i));
        fm = vandq_u16(fm, vcleq_u16(vabdq_u16(q2, q1), i));
        if WD > 6 {
            fm = vandq_u16(fm, vcleq_u16(vabdq_u16(p3, p2), i));
            fm = vandq_u16(fm, vcleq_u16(vabdq_u16(q3, q2), i));
        }
    }

    // ---- flatness ----------------------------------------------------
    let mut flat8in = vdupq_n_u16(0);
    if WD >= 6 {
        flat8in = vandq_u16(vcleq_u16(vabdq_u16(p2, p0), f), vcleq_u16(ad_p1p0, f));
        flat8in = vandq_u16(flat8in, vcleq_u16(ad_q1q0, f));
        flat8in = vandq_u16(flat8in, vcleq_u16(vabdq_u16(q2, q0), f));
        if WD >= 8 {
            flat8in = vandq_u16(flat8in, vcleq_u16(vabdq_u16(p3, p0), f));
            flat8in = vandq_u16(flat8in, vcleq_u16(vabdq_u16(q3, q0), f));
        }
    }

    // ---- branch masks ------------------------------------------------
    // Mirrors the scalar ladder exactly:
    //   wd >= 16 && flat8out && flat8in   -> wide
    //   wd >= 8  && flat8in               -> 8-tap
    //   wd == 6  && flat8in               -> 6-tap
    //   otherwise                         -> narrow
    let mut m_wide = vdupq_n_u16(0);
    if WD >= 16 {
        let mut flat8out = vandq_u16(
            vcleq_u16(vabdq_u16(p6, p0), f),
            vcleq_u16(vabdq_u16(p5, p0), f),
        );
        flat8out = vandq_u16(flat8out, vcleq_u16(vabdq_u16(p4, p0), f));
        flat8out = vandq_u16(flat8out, vcleq_u16(vabdq_u16(q4, q0), f));
        flat8out = vandq_u16(flat8out, vcleq_u16(vabdq_u16(q5, q0), f));
        flat8out = vandq_u16(flat8out, vcleq_u16(vabdq_u16(q6, q0), f));
        m_wide = vandq_u16(vandq_u16(fm, flat8out), flat8in);
    }
    // `flat8in` is all-zero for WD == 4, so `m_flat` is empty there.
    let m_flat = vbicq_u16(vandq_u16(fm, flat8in), m_wide);
    let m_narrow = vbicq_u16(fm, vorrq_u16(m_wide, m_flat));

    // ---- narrow (4-tap) ----------------------------------------------
    // Always computed: it is the fallback of every width.
    let clip_lo = vsubq_s16(vdupq_n_s16(0), vaddq_s16(clip_hi, vdupq_n_s16(1)));
    let sclip = |v: int16x8_t| vminq_s16(vmaxq_s16(v, clip_lo), clip_hi);
    let s16 = |v: uint16x8_t| vreinterpretq_s16_u16(v);

    let hev = vorrq_u16(vcgtq_u16(ad_p1p0, hthr), vcgtq_u16(ad_q1q0, hthr));
    let base3 = vmulq_n_s16(vsubq_s16(s16(q0), s16(p0)), 3);
    let f_hev = sclip(vaddq_s16(base3, sclip(vsubq_s16(s16(p1), s16(q1)))));
    let f_nohev = sclip(base3);
    let ff = vbslq_s16(hev, f_hev, f_nohev);
    let f1 = vshrq_n_s16::<3>(vminq_s16(vaddq_s16(ff, vdupq_n_s16(4)), clip_hi));
    let f2 = vshrq_n_s16::<3>(vminq_s16(vaddq_s16(ff, vdupq_n_s16(3)), clip_hi));
    let f3 = vshrq_n_s16::<1>(vaddq_s16(f1, vdupq_n_s16(1)));

    let n_m1 = clip_px(vaddq_s16(s16(p0), f2), bd_max);
    let n_p0 = clip_px(vsubq_s16(s16(q0), f1), bd_max);
    // The hev branch writes only -1 and 0; -2 and +1 keep their input there.
    let n_m2 = vbslq_u16(hev, p1, clip_px(vaddq_s16(s16(p1), f3), bd_max));
    let n_p1 = vbslq_u16(hev, q1, clip_px(vsubq_s16(s16(q1), f3), bd_max));

    let mut o = [
        t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[9], t[10], t[11], t[12],
    ];
    // `o[n]` is tap `n - 6`, i.e. the 12 taps the wide filter can write.
    o[4] = vbslq_u16(m_narrow, n_m2, o[4]);
    o[5] = vbslq_u16(m_narrow, n_m1, o[5]);
    o[6] = vbslq_u16(m_narrow, n_p0, o[6]);
    o[7] = vbslq_u16(m_narrow, n_p1, o[7]);

    // ---- flat filters ------------------------------------------------
    if WD >= 8 {
        // 8-tap, weights summing to 8.
        let mut s = vaddq_u16(
            vaddq_u16(vmulq_n_u16(p3, 3), vaddq_u16(p2, p2)),
            vaddq_u16(vaddq_u16(p1, p0), q0),
        );
        let four = vdupq_n_u16(4);
        let rnd = |s: uint16x8_t| vshrq_n_u16::<3>(vaddq_u16(s, four));
        let step = |s: uint16x8_t, a: uint16x8_t, b: uint16x8_t, c: uint16x8_t, d: uint16x8_t| {
            vsubq_u16(vaddq_u16(s, vaddq_u16(a, b)), vaddq_u16(c, d))
        };
        o[3] = vbslq_u16(m_flat, rnd(s), o[3]);
        s = step(s, p1, q1, p3, p2);
        o[4] = vbslq_u16(m_flat, rnd(s), o[4]);
        s = step(s, p0, q2, p3, p1);
        o[5] = vbslq_u16(m_flat, rnd(s), o[5]);
        s = step(s, q0, q3, p3, p0);
        o[6] = vbslq_u16(m_flat, rnd(s), o[6]);
        s = step(s, q1, q3, p2, q0);
        o[7] = vbslq_u16(m_flat, rnd(s), o[7]);
        s = step(s, q2, q3, p1, q1);
        o[8] = vbslq_u16(m_flat, rnd(s), o[8]);
    } else if WD == 6 {
        // 6-tap, weights summing to 8.
        let mut s = vaddq_u16(
            vaddq_u16(vmulq_n_u16(p2, 3), vaddq_u16(p1, p1)),
            vaddq_u16(vaddq_u16(p0, p0), q0),
        );
        let four = vdupq_n_u16(4);
        let rnd = |s: uint16x8_t| vshrq_n_u16::<3>(vaddq_u16(s, four));
        let step = |s: uint16x8_t, a: uint16x8_t, b: uint16x8_t, c: uint16x8_t, d: uint16x8_t| {
            vsubq_u16(vaddq_u16(s, vaddq_u16(a, b)), vaddq_u16(c, d))
        };
        o[4] = vbslq_u16(m_flat, rnd(s), o[4]);
        s = step(s, q0, q1, p2, p2);
        o[5] = vbslq_u16(m_flat, rnd(s), o[5]);
        s = step(s, q1, q2, p2, p1);
        o[6] = vbslq_u16(m_flat, rnd(s), o[6]);
        s = step(s, q2, q2, p1, p0);
        o[7] = vbslq_u16(m_flat, rnd(s), o[7]);
    }

    if WD >= 16 {
        // 13-term wide filter, weights summing to 16.
        let mut s = vaddq_u16(
            vaddq_u16(
                vmulq_n_u16(p6, 7),
                vaddq_u16(vaddq_u16(p5, p5), vaddq_u16(p4, p4)),
            ),
            vaddq_u16(vaddq_u16(vaddq_u16(p3, p2), vaddq_u16(p1, p0)), q0),
        );
        let eight = vdupq_n_u16(8);
        let rnd = |s: uint16x8_t| vshrq_n_u16::<4>(vaddq_u16(s, eight));
        let step = |s: uint16x8_t, a: uint16x8_t, b: uint16x8_t, c: uint16x8_t, d: uint16x8_t| {
            vsubq_u16(vaddq_u16(s, vaddq_u16(a, b)), vaddq_u16(c, d))
        };
        o[0] = vbslq_u16(m_wide, rnd(s), o[0]);
        s = step(s, p3, q1, p6, p6);
        o[1] = vbslq_u16(m_wide, rnd(s), o[1]);
        s = step(s, p2, q2, p6, p5);
        o[2] = vbslq_u16(m_wide, rnd(s), o[2]);
        s = step(s, p1, q3, p6, p4);
        o[3] = vbslq_u16(m_wide, rnd(s), o[3]);
        s = step(s, p0, q4, p6, p3);
        o[4] = vbslq_u16(m_wide, rnd(s), o[4]);
        s = step(s, q0, q5, p6, p2);
        o[5] = vbslq_u16(m_wide, rnd(s), o[5]);
        s = step(s, q1, q6, p6, p1);
        o[6] = vbslq_u16(m_wide, rnd(s), o[6]);
        s = step(s, q2, q6, p5, p0);
        o[7] = vbslq_u16(m_wide, rnd(s), o[7]);
        s = step(s, q3, q6, p4, q0);
        o[8] = vbslq_u16(m_wide, rnd(s), o[8]);
        s = step(s, q4, q6, p3, q1);
        o[9] = vbslq_u16(m_wide, rnd(s), o[9]);
        s = step(s, q5, q6, p2, q2);
        o[10] = vbslq_u16(m_wide, rnd(s), o[10]);
        s = step(s, q6, q6, p1, q3);
        o[11] = vbslq_u16(m_wide, rnd(s), o[11]);
    }

    // Only the taps this width can write are stored back — the SAME range the
    // run kernels store to the scratch, from the same `written_taps`, so the
    // two can never drift apart. Writing wider would be harmless in the
    // scratch but would break the contract that a `wd` never touches outside
    // `+-reach`, which is what bounds the run kernels' indexing.
    let (lo, hi) = written_taps(WD);
    t[lo..hi].copy_from_slice(&o[lo - 1..hi - 1]);
}

/// The two run kernels, one pair per scratch pixel width.
///
/// A macro rather than a generic because the leaf load/store must know the
/// concrete element type and `#[rite]`'s `#[target_feature]` cannot ride on a
/// trait method. The INDEX MATH — which is the part that can silently read the
/// wrong tap — is written once, here.
#[cfg(target_arch = "aarch64")]
macro_rules! lf_runs {
    ($px:ty, $vname:ident, $hname:ident, $ld:ident, $st:ident) => {
        /// V direction (`HV::V`): tap-major scratch, no transpose.
        ///
        /// Tap `k` of lane `j` is `buf[base + k * LF_SW_V + j]`, so all lanes of
        /// one tap are contiguous.
        #[rite(neon)]
        fn $vname<const WD: c_int, const SW: usize>(
            buf: &mut [$px; LF_BLOCK_LEN],
            base: usize,
            n_lanes: usize,
            thr: &LfLaneThresholds<LF_GROUPS_V>,
        ) {
            // `SW` is a CONST generic, not a parameter: a runtime stride turns
            // every tap address in this loop into a multiply and showed up as a
            // new `lf_dispatch` self-time leaf (`~/tmp` profile, 2026-08-10).
            // Only four values exist, and `SW == LF_SW_H` reproduces base's
            // addressing exactly.
            const { assert!(SW.is_power_of_two() && SW >= 16) };
            let sw = SW;
            let reach = reach_of(WD);
            let bd_max = vdupq_n_s16(thr.bd_max as i16);
            let clip_hi = vdupq_n_s16(thr.clip_hi);
            let f = vdupq_n_u16(thr.f);
            let (wlo, whi) = written_taps(WD);

            // `sw` is the run's OWN row stride, not a constant: a short V run
            // keeps `LF_SW_H`'s 16 and therefore base's exact scratch
            // footprint, and only a long one spreads out. A fixed 128 made a
            // 1-group rectangle span 1,792 bytes instead of 224.
            debug_assert!(base >= reach * sw && 2 * reach * sw <= LF_BLOCK_LEN);
            debug_assert!(n_lanes <= sw);
            let mut lane = 0usize;
            while lane < n_lanes {
                let c = lane / 8;
                let e = lane_thr(&thr.e, c);
                let i = lane_thr(&thr.i, c);
                let h = lane_thr(&thr.h, c);

                // Slot `n` (0..2*reach) is tap `n - reach`.
                let slot0 = base - reach * sw + lane;
                let mut t = [vdupq_n_u16(0); 14];
                for n in 0..2 * reach {
                    t[n + 7 - reach] = $ld(buf, slot0 + n * sw);
                }

                lf_core::<WD>(&mut t, e, i, h, f, bd_max, clip_hi);

                for n in wlo..whi {
                    $st(buf, slot0 + (n + reach - 7) * sw, t[n]);
                }
                lane += 8;
            }
        }

        /// H direction (`HV::H`): lane-major scratch, transposed in registers.
        ///
        /// Tap `k` of lane `j` is `buf[base + j * LF_SW_H + k]`, so each lane is
        /// one scratch row and the kernel must transpose 8x8 tiles — the same
        /// structure dav1d's `lpf_h_sb_*` asm uses.
        #[rite(neon)]
        fn $hname<const WD: c_int>(
            buf: &mut [$px; LF_BLOCK_LEN],
            base: usize,
            n_lanes: usize,
            thr: &LfLaneThresholds<LF_GROUPS_H>,
        ) {
            let reach = reach_of(WD);
            let bd_max = vdupq_n_s16(thr.bd_max as i16);
            let clip_hi = vdupq_n_s16(thr.clip_hi);
            let f = vdupq_n_u16(thr.f);
            let (wlo, whi) = written_taps(WD);

            debug_assert!(base >= reach);
            // Slot 0 (tap `-reach`) of lane `j` is `buf[j * LF_SW_H + row_off]`.
            let row_off = base - reach;
            let wide = 2 * reach > 8;

            let mut lane = 0usize;
            while lane < n_lanes {
                let c = lane / 8;
                let e = lane_thr(&thr.e, c);
                let i = lane_thr(&thr.i, c);
                let h = lane_thr(&thr.h, c);

                let mut rows = [vdupq_n_u16(0); 8];
                for (j, r) in rows.iter_mut().enumerate() {
                    *r = $ld(buf, (lane + j) * LF_SW_H + row_off);
                }
                let cols_lo = transpose8(rows);
                let mut t = [vdupq_n_u16(0); 14];
                for n in 0..core::cmp::min(2 * reach, 8) {
                    t[n + 7 - reach] = cols_lo[n];
                }

                let mut cols_hi = [vdupq_n_u16(0); 8];
                if wide {
                    for (j, r) in rows.iter_mut().enumerate() {
                        *r = $ld(buf, (lane + j) * LF_SW_H + row_off + 8);
                    }
                    cols_hi = transpose8(rows);
                    for n in 8..2 * reach {
                        t[n + 7 - reach] = cols_hi[n - 8];
                    }
                }

                lf_core::<WD>(&mut t, e, i, h, f, bd_max, clip_hi);

                // Refresh only the written slots; the rest still hold what was
                // loaded, so the round trip is a no-op for them.
                let mut out_lo = cols_lo;
                let mut out_hi = cols_hi;
                for n in wlo..whi {
                    let slot = n + reach - 7;
                    if slot < 8 {
                        out_lo[slot] = t[n];
                    } else {
                        out_hi[slot - 8] = t[n];
                    }
                }
                let back = transpose8(out_lo);
                for (j, r) in back.iter().enumerate() {
                    $st(buf, (lane + j) * LF_SW_H + row_off, *r);
                }
                if wide {
                    let back = transpose8(out_hi);
                    for (j, r) in back.iter().enumerate() {
                        $st(buf, (lane + j) * LF_SW_H + row_off + 8, *r);
                    }
                }
                lane += 8;
            }
        }
    };
}

/// `t` indices this width writes: taps -6..5 at 16, -3..2 at 8, -2..1 at 6
/// and 4. Every one is inside `+-reach`, which is what lets the run kernels
/// store through the same slot range they loaded.
#[cfg(target_arch = "aarch64")]
const fn written_taps(wd: c_int) -> (usize, usize) {
    if wd >= 16 {
        (1, 13)
    } else if wd >= 8 {
        (4, 10)
    } else {
        (5, 9)
    }
}

/// The 8-lane threshold vector for chunk `c`: groups `2c` and `2c + 1`, four
/// lanes each.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn lane_thr<const NG: usize>(v: &[u16; NG], c: usize) -> uint16x8_t {
    vcombine_u16(vdup_n_u16(v[2 * c]), vdup_n_u16(v[2 * c + 1]))
}

#[cfg(target_arch = "aarch64")]
lf_runs!(u8, lf_run_v_u8, lf_run_h_u8, ld8_u8, st8_u8);
#[cfg(target_arch = "aarch64")]
lf_runs!(u16, lf_run_v_u16, lf_run_h_u16, ld8_u16, st8_u16);

/// `wd` dispatch for one V stride. Split out so the `sw` match above stays four
/// arms rather than sixteen.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn lf_v_u8<const SW: usize>(
    buf: &mut [u8; LF_BLOCK_LEN],
    base: usize,
    n_lanes: usize,
    thr: &LfLaneThresholds<LF_GROUPS_V>,
    wd: c_int,
) {
    match wd {
        4 => lf_run_v_u8::<4, SW>(buf, base, n_lanes, thr),
        6 => lf_run_v_u8::<6, SW>(buf, base, n_lanes, thr),
        8 => lf_run_v_u8::<8, SW>(buf, base, n_lanes, thr),
        _ => lf_run_v_u8::<16, SW>(buf, base, n_lanes, thr),
    }
}

#[cfg(target_arch = "aarch64")]
#[arcane]
fn lf_dispatch_u8(
    _token: Arm64,
    buf: &mut [u8; LF_BLOCK_LEN],
    base: usize,
    is_v: bool,
    n_lanes: usize,
    sw: usize,
    thr_v: &LfLaneThresholds<LF_GROUPS_V>,
    thr_h: &LfLaneThresholds<LF_GROUPS_H>,
    wd: c_int,
) {
    match (is_v, wd) {
        (true, wd) => match sw {
            LF_SW_H => lf_v_u8::<LF_SW_H>(buf, base, n_lanes, thr_v, wd),
            32 => lf_v_u8::<32>(buf, base, n_lanes, thr_v, wd),
            64 => lf_v_u8::<64>(buf, base, n_lanes, thr_v, wd),
            _ => lf_v_u8::<LF_SW_V>(buf, base, n_lanes, thr_v, wd),
        },
        (false, 4) => lf_run_h_u8::<4>(buf, base, n_lanes, thr_h),
        (false, 6) => lf_run_h_u8::<6>(buf, base, n_lanes, thr_h),
        (false, 8) => lf_run_h_u8::<8>(buf, base, n_lanes, thr_h),
        (false, _) => lf_run_h_u8::<16>(buf, base, n_lanes, thr_h),
    }
}

/// `wd` dispatch for one V stride. Split out so the `sw` match above stays four
/// arms rather than sixteen.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn lf_v_u16<const SW: usize>(
    buf: &mut [u16; LF_BLOCK_LEN],
    base: usize,
    n_lanes: usize,
    thr: &LfLaneThresholds<LF_GROUPS_V>,
    wd: c_int,
) {
    match wd {
        4 => lf_run_v_u16::<4, SW>(buf, base, n_lanes, thr),
        6 => lf_run_v_u16::<6, SW>(buf, base, n_lanes, thr),
        8 => lf_run_v_u16::<8, SW>(buf, base, n_lanes, thr),
        _ => lf_run_v_u16::<16, SW>(buf, base, n_lanes, thr),
    }
}

#[cfg(target_arch = "aarch64")]
#[arcane]
fn lf_dispatch_u16(
    _token: Arm64,
    buf: &mut [u16; LF_BLOCK_LEN],
    base: usize,
    is_v: bool,
    n_lanes: usize,
    sw: usize,
    thr_v: &LfLaneThresholds<LF_GROUPS_V>,
    thr_h: &LfLaneThresholds<LF_GROUPS_H>,
    wd: c_int,
) {
    match (is_v, wd) {
        (true, wd) => match sw {
            LF_SW_H => lf_v_u16::<LF_SW_H>(buf, base, n_lanes, thr_v, wd),
            32 => lf_v_u16::<32>(buf, base, n_lanes, thr_v, wd),
            64 => lf_v_u16::<64>(buf, base, n_lanes, thr_v, wd),
            _ => lf_v_u16::<LF_SW_V>(buf, base, n_lanes, thr_v, wd),
        },
        (false, 4) => lf_run_h_u16::<4>(buf, base, n_lanes, thr_h),
        (false, 6) => lf_run_h_u16::<6>(buf, base, n_lanes, thr_h),
        (false, 8) => lf_run_h_u16::<8>(buf, base, n_lanes, thr_h),
        (false, _) => lf_run_h_u16::<16>(buf, base, n_lanes, thr_h),
    }
}

/// Filter one fused run of 4-pixel groups inside the compact scratch.
///
/// `scratch` is the raw bytes of `LfScratch::buf`, split back into a typed
/// array here so the caller can stay bit-depth-generic. Returns `false`
/// without touching it when the run is not one the kernels cover, so the
/// caller runs its scalar reference instead.
#[cfg(target_arch = "aarch64")]
pub(crate) fn lf_compact_run_neon(
    bpc: crate::include::common::bitdepth::BPC,
    scratch: &mut [u8],
    base: usize,
    is_v: bool,
    n_lanes: usize,
    sw: usize,
    params: &[(u8, u8, u8, c_int)],
    wd: c_int,
    bitdepth_min_8: u8,
    bd_max: u16,
) -> bool {
    // Ablation switch (measurement only; const-false without `__ablate`).
    if crate::src::ablate::is_off(crate::src::ablate::Family::LoopFilter) {
        return false;
    }
    use crate::include::common::bitdepth::BPC;
    use zerocopy::FromBytes as _;

    // The AV1 widths: 4/8/16 on luma, 4/6 on chroma. Anything else would be a
    // caller bug, but fall back rather than mis-filter.
    let groups_cap = if is_v { LF_GROUPS_V } else { LF_GROUPS_H };
    if !matches!(wd, 4 | 6 | 8 | 16) || n_lanes == 0 || n_lanes > sw {
        return false;
    }
    if params.len() > groups_cap {
        return false;
    }
    if is_v && !(sw.is_power_of_two() && (LF_SW_H..=LF_SW_V).contains(&sw)) {
        return false;
    }
    if !is_v && sw != LF_SW_H {
        return false;
    }
    let Some(token) = Arm64::summon() else {
        return false;
    };

    // Unfilled groups keep zero thresholds. Those lanes are pad columns (V) or
    // pad rows (H) of the scratch that `close` never compares, so whatever the
    // kernel does with them is inert.
    let mut thr_v = LfLaneThresholds::<LF_GROUPS_V> {
        e: [0; LF_GROUPS_V],
        i: [0; LF_GROUPS_V],
        h: [0; LF_GROUPS_V],
        f: 1 << bitdepth_min_8,
        bd_max,
        clip_hi: (128i16 << bitdepth_min_8) - 1,
    };
    let mut thr_h = LfLaneThresholds::<LF_GROUPS_H> {
        e: [0; LF_GROUPS_H],
        i: [0; LF_GROUPS_H],
        h: [0; LF_GROUPS_H],
        f: thr_v.f,
        bd_max,
        clip_hi: thr_v.clip_hi,
    };
    for (g, &(e, i, h, _)) in params.iter().enumerate() {
        let (e, i, h) = (
            (e as u16) << bitdepth_min_8,
            (i as u16) << bitdepth_min_8,
            (h as u16) << bitdepth_min_8,
        );
        if is_v {
            thr_v.e[g] = e;
            thr_v.i[g] = i;
            thr_v.h[g] = h;
        } else {
            thr_h.e[g] = e;
            thr_h.i[g] = i;
            thr_h.h[g] = h;
        }
    }

    match bpc {
        BPC::BPC8 => {
            let Ok(buf) = <&mut [u8; LF_BLOCK_LEN]>::try_from(scratch) else {
                return false;
            };
            lf_dispatch_u8(token, buf, base, is_v, n_lanes, sw, &thr_v, &thr_h, wd);
        }
        BPC::BPC16 => {
            let Ok(buf) = <[u16; LF_BLOCK_LEN]>::mut_from_bytes(scratch) else {
                return false;
            };
            lf_dispatch_u16(token, buf, base, is_v, n_lanes, sw, &thr_v, &thr_h, wd);
        }
    }
    true
}

// ============================================================================
// WRITE-BACK DIFF SCAN
// ============================================================================
//
// `LfBlock::close` may only write — and may only take a MUTABLE guard on —
// pixels the filter actually changed, because the tap window it read is wider
// than the span it can write and a guard over a merely-read tap is a real
// conflict with a concurrent tile worker (zenavif#30). So the span has to be
// found by diffing against the pristine copy, per row, every time.
//
// Scalar, that is a `position` plus an `rposition` over up to 16 elements for
// each of up to 16 rows, and it measured as 285 `sample` leaves (3.11% of a
// t=1 8bpc frame) — MORE than the filter arithmetic itself after this port.
// One `vceqq` plus a nibble movemask answers both ends at once.

/// Lane ordinals, for masking off the pad columns beyond `w`.
#[cfg(target_arch = "aarch64")]
const LANE_IDX: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

/// Per-lane "differs" bytes -> first and last set lane.
///
/// `vshrn` by 4 packs each 0x00/0xFF byte into one nibble of a `u64`, in lane
/// order, which is aarch64's stand-in for a movemask.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn first_last_set(diff: uint8x16_t) -> Option<(usize, usize)> {
    let packed = vshrn_n_u16::<4>(vreinterpretq_u16_u8(diff));
    let m = vget_lane_u64::<0>(vreinterpret_u64_u8(packed));
    if m == 0 {
        None
    } else {
        Some((
            (m.trailing_zeros() / 4) as usize,
            ((63 - m.leading_zeros()) / 4) as usize,
        ))
    }
}

#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn valid_mask(w: usize) -> uint8x16_t {
    vcltq_u8(safe_simd::vld1q_u8(&LANE_IDX), vdupq_n_u8(w as u8))
}

#[cfg(target_arch = "aarch64")]
#[arcane]
fn diff_span_u8(
    _token: Arm64,
    work: &[u8; LF_BLOCK_LEN],
    pristine: &[u8; LF_BLOCK_LEN],
    off: usize,
    w: usize,
) -> Option<(usize, usize)> {
    let a = safe_simd::vld1q_u8(<&[u8; 16]>::try_from(&work[off..off + 16]).unwrap());
    let b = safe_simd::vld1q_u8(<&[u8; 16]>::try_from(&pristine[off..off + 16]).unwrap());
    first_last_set(vandq_u8(vmvnq_u8(vceqq_u8(a, b)), valid_mask(w)))
}

#[cfg(target_arch = "aarch64")]
#[arcane]
fn diff_span_u16(
    _token: Arm64,
    work: &[u16; LF_BLOCK_LEN],
    pristine: &[u16; LF_BLOCK_LEN],
    off: usize,
    w: usize,
) -> Option<(usize, usize)> {
    let ne = |k: usize| -> uint8x8_t {
        let a = ld8_u16(work, off + k);
        let b = ld8_u16(pristine, off + k);
        vmvn_u8(vmovn_u16(vceqq_u16(a, b)))
    };
    first_last_set(vandq_u8(vcombine_u8(ne(0), ne(8)), valid_mask(w)))
}

/// Span of the `w` scratch pixels at flat index `off` that the filter changed.
///
/// `off` is `row * sw + col`, so the caller can scan a wide V rectangle in
/// 16-column chunks and take the same mutable guards it would have taken as
/// several narrower runs. Returns `None` for an untouched chunk, exactly like
/// the scalar `position`/`rposition` pair it replaces — including for a chunk
/// whose changes all sit in the pad columns beyond `w`, which the caller must
/// not write back.
#[cfg(target_arch = "aarch64")]
pub(crate) fn lf_diff_span(
    bpc: crate::include::common::bitdepth::BPC,
    work: &[u8],
    pristine: &[u8],
    off: usize,
    w: usize,
) -> Option<Option<(usize, usize)>> {
    // Ablation switch (measurement only; const-false without `__ablate`).
    if crate::src::ablate::is_off(crate::src::ablate::Family::LoopFilter) {
        return None;
    }
    use crate::include::common::bitdepth::BPC;
    use zerocopy::FromBytes as _;

    // One `vld1q` window, and it must stay inside the scratch: the caller
    // chunks at 16 and every scratch row stride is a multiple of 16, so a chunk
    // never straddles a row and `off + 16` never leaves the array.
    debug_assert!(w <= 16 && off + 16 <= LF_BLOCK_LEN);
    let token = Arm64::summon()?;
    match bpc {
        BPC::BPC8 => {
            let work = <&[u8; LF_BLOCK_LEN]>::try_from(work).ok()?;
            let pristine = <&[u8; LF_BLOCK_LEN]>::try_from(pristine).ok()?;
            Some(diff_span_u8(token, work, pristine, off, w))
        }
        BPC::BPC16 => {
            let work = <[u16; LF_BLOCK_LEN]>::ref_from_bytes(work).ok()?;
            let pristine = <[u16; LF_BLOCK_LEN]>::ref_from_bytes(pristine).ok()?;
            Some(diff_span_u16(token, work, pristine, off, w))
        }
    }
}
