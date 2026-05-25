//! Safe SIMD implementations for Loop Filter (Deblocking Filter)
//!
//! The loop filter removes blocking artifacts at transform block boundaries.
//! It operates on edges between adjacent blocks, filtering up to 7 pixels
//! on each side of the edge.
//!
//! Key operations:
//! - Filter strength calculation based on quantization
//! - Flatness detection (flat8in, flat8out)
//! - Different filter widths (4, 6, 8, 16 pixels)
//! - Horizontal and vertical edge filtering
//!
//! This module uses safe slice-based pixel access. The dispatch function is fully safe.
//! The level cache `&[AtomicU8]` is passed directly to inner functions which read
//! entries on demand via `Relaxed` atomic loads. No intermediate gather buffer is needed.
//! PicOffset pixel data is converted to slices. All inner functions are fully safe.

#![cfg_attr(not(feature = "unchecked"), forbid(unsafe_code))]
#![cfg_attr(feature = "unchecked", deny(unsafe_code))]
#![allow(unused_imports)]

#[cfg(target_arch = "x86_64")]
use archmage::{Desktop64, SimdToken, arcane, rite};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::include::common::bitdepth::AsPrimitive;
use crate::include::common::bitdepth::BitDepth;
use crate::include::common::bitdepth::DynPixel;
use crate::include::common::intops::iclip;
use crate::include::dav1d::picture::PicOffset;
use crate::src::align::Align16;
use crate::src::ffi_safe::FFISafe;
use crate::src::lf_mask::Av1FilterLUT;
use crate::src::with_offset::WithOffset;
use std::sync::atomic::AtomicU8;
use std::sync::atomic::Ordering::Relaxed;
#[allow(non_camel_case_types)]
type ptrdiff_t = isize;
use std::cmp;
use std::ffi::c_int;

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Clamp difference value for bitdepth
#[inline(always)]
fn iclip_diff(v: i32, bitdepth_min_8: u8) -> i32 {
    iclip(
        v,
        -128 * (1 << bitdepth_min_8),
        128 * (1 << bitdepth_min_8) - 1,
    )
}

/// Compute a signed index from a base usize and signed offset.
#[inline(always)]
fn signed_idx(base: usize, offset: isize) -> usize {
    (base as isize + offset) as usize
}

// ============================================================================
// CORE LOOP FILTER (4 pixels at a time)
// ============================================================================

/// Core loop filter for 8bpc - processes 4 pixels
/// `buf` is the pixel buffer, `base` is the offset to the edge point.
/// `stridea` is the stride between the 4 parallel pixels.
/// `strideb` is the stride in the filter direction.
///
/// `#[rite]` inlines this directly into matching-feature callers (e.g.
/// `lpf_h_sb_y_8bpc_inner` is `#[arcane]` V2 — `loop_filter_4_8bpc` inlines
/// into its body so the per-edge SIMD dispatch is a direct call chain
/// with no target_feature trampoline anywhere.
#[cfg(any(target_arch = "x86_64", target_arch = "wasm32"))]
#[cfg_attr(target_arch = "x86_64", rite)]
fn loop_filter_4_8bpc(
    #[cfg(target_arch = "x86_64")] _token: Desktop64,
    buf: &mut [u8],
    base: usize,
    e: i32,
    i: i32,
    h: i32,
    stridea: isize,
    strideb: isize,
    wd: i32,
    bitdepth_max: i32,
) {
    // Fast paths: SIMD v-filter (stridea==1, contiguous 4-byte column loads).
    // Token already provided by caller — direct dispatch, no per-edge summon.
    #[cfg(target_arch = "x86_64")]
    if stridea == 1 && bitdepth_max == 255 {
        match wd {
            4 => {
                loop_filter_4_8bpc_narrow_simd_v(_token, buf, base, e, i, h, strideb);
                return;
            }
            6 => {
                loop_filter_4_8bpc_wd6_simd_v(_token, buf, base, e, i, h, strideb);
                return;
            }
            8 => {
                loop_filter_4_8bpc_wd8_simd_v(_token, buf, base, e, i, h, strideb);
                return;
            }
            16 => {
                loop_filter_4_8bpc_wd16_simd_v(_token, buf, base, e, i, h, strideb);
                return;
            }
            _ => {}
        }
    }
    // SIMD h-filter: stridea==stride, strideb==1. 4 lanes are 4 different rows;
    // we load contiguous N-byte chunks per row and transpose 4xN i32 into
    // pixel-position vectors.
    #[cfg(target_arch = "x86_64")]
    if strideb == 1 && stridea != 1 && bitdepth_max == 255 {
        match wd {
            4 => {
                loop_filter_4_8bpc_narrow_simd_h(_token, buf, base, e, i, h, stridea);
                return;
            }
            8 => {
                loop_filter_4_8bpc_wd8_simd_h(_token, buf, base, e, i, h, stridea);
                return;
            }
            16 => {
                loop_filter_4_8bpc_wd16_simd_h(_token, buf, base, e, i, h, stridea);
                return;
            }
            _ => {}
        }
    }
    let f = 1i32;

    for idx in 0..4isize {
        let edge = signed_idx(base, idx * stridea);

        let get_px = |offset: isize| -> i32 { buf[signed_idx(edge, strideb * offset)] as i32 };

        let p1 = get_px(-2);
        let p0 = get_px(-1);
        let q0 = get_px(0);
        let q1 = get_px(1);

        // Filter mask calculation
        let mut fm = (p1 - p0).abs() <= i
            && (q1 - q0).abs() <= i
            && (p0 - q0).abs() * 2 + ((p1 - q1).abs() >> 1) <= e;

        let (mut p2, mut p3, mut q2, mut q3) = (0, 0, 0, 0);
        let (mut p4, mut p5, mut p6, mut q4, mut q5, mut q6) = (0, 0, 0, 0, 0, 0);

        if wd > 4 {
            p2 = get_px(-3);
            q2 = get_px(2);
            fm &= (p2 - p1).abs() <= i && (q2 - q1).abs() <= i;

            if wd > 6 {
                p3 = get_px(-4);
                q3 = get_px(3);
                fm &= (p3 - p2).abs() <= i && (q3 - q2).abs() <= i;
            }
        }

        if !fm {
            continue;
        }

        let mut flat8out = false;
        let mut flat8in = false;

        if wd >= 16 {
            p6 = get_px(-7);
            p5 = get_px(-6);
            p4 = get_px(-5);
            q4 = get_px(4);
            q5 = get_px(5);
            q6 = get_px(6);

            flat8out = (p6 - p0).abs() <= f
                && (p5 - p0).abs() <= f
                && (p4 - p0).abs() <= f
                && (q4 - q0).abs() <= f
                && (q5 - q0).abs() <= f
                && (q6 - q0).abs() <= f;
        }

        if wd >= 6 {
            flat8in = (p2 - p0).abs() <= f
                && (p1 - p0).abs() <= f
                && (q1 - q0).abs() <= f
                && (q2 - q0).abs() <= f;
        }

        if wd >= 8 {
            flat8in &= (p3 - p0).abs() <= f && (q3 - q0).abs() <= f;
        }

        // Write helper — sets pixel at offset from edge
        let set_px = |buf: &mut [u8], offset: isize, val: i32| {
            buf[signed_idx(edge, strideb * offset)] = val.clamp(0, bitdepth_max) as u8;
        };

        if wd >= 16 && flat8out && flat8in {
            // Wide filter (16 taps)
            set_px(
                buf,
                -6,
                (p6 + p6 + p6 + p6 + p6 + p6 * 2 + p5 * 2 + p4 * 2 + p3 + p2 + p1 + p0 + q0 + 8)
                    >> 4,
            );
            set_px(
                buf,
                -5,
                (p6 + p6 + p6 + p6 + p6 + p5 * 2 + p4 * 2 + p3 * 2 + p2 + p1 + p0 + q0 + q1 + 8)
                    >> 4,
            );
            set_px(
                buf,
                -4,
                (p6 + p6 + p6 + p6 + p5 + p4 * 2 + p3 * 2 + p2 * 2 + p1 + p0 + q0 + q1 + q2 + 8)
                    >> 4,
            );
            set_px(
                buf,
                -3,
                (p6 + p6 + p6 + p5 + p4 + p3 * 2 + p2 * 2 + p1 * 2 + p0 + q0 + q1 + q2 + q3 + 8)
                    >> 4,
            );
            set_px(
                buf,
                -2,
                (p6 + p6 + p5 + p4 + p3 + p2 * 2 + p1 * 2 + p0 * 2 + q0 + q1 + q2 + q3 + q4 + 8)
                    >> 4,
            );
            set_px(
                buf,
                -1,
                (p6 + p5 + p4 + p3 + p2 + p1 * 2 + p0 * 2 + q0 * 2 + q1 + q2 + q3 + q4 + q5 + 8)
                    >> 4,
            );
            set_px(
                buf,
                0,
                (p5 + p4 + p3 + p2 + p1 + p0 * 2 + q0 * 2 + q1 * 2 + q2 + q3 + q4 + q5 + q6 + 8)
                    >> 4,
            );
            set_px(
                buf,
                1,
                (p4 + p3 + p2 + p1 + p0 + q0 * 2 + q1 * 2 + q2 * 2 + q3 + q4 + q5 + q6 + q6 + 8)
                    >> 4,
            );
            set_px(
                buf,
                2,
                (p3 + p2 + p1 + p0 + q0 + q1 * 2 + q2 * 2 + q3 * 2 + q4 + q5 + q6 + q6 + q6 + 8)
                    >> 4,
            );
            set_px(
                buf,
                3,
                (p2 + p1 + p0 + q0 + q1 + q2 * 2 + q3 * 2 + q4 * 2 + q5 + q6 + q6 + q6 + q6 + 8)
                    >> 4,
            );
            set_px(
                buf,
                4,
                (p1 + p0 + q0 + q1 + q2 + q3 * 2 + q4 * 2 + q5 * 2 + q6 + q6 + q6 + q6 + q6 + 8)
                    >> 4,
            );
            set_px(
                buf,
                5,
                (p0 + q0 + q1 + q2 + q3 + q4 * 2 + q5 * 2 + q6 * 2 + q6 + q6 + q6 + q6 + q6 + 8)
                    >> 4,
            );
        } else if wd >= 8 && flat8in {
            // 8-tap filter
            set_px(buf, -3, (p3 + p3 + p3 + 2 * p2 + p1 + p0 + q0 + 4) >> 3);
            set_px(buf, -2, (p3 + p3 + p2 + 2 * p1 + p0 + q0 + q1 + 4) >> 3);
            set_px(buf, -1, (p3 + p2 + p1 + 2 * p0 + q0 + q1 + q2 + 4) >> 3);
            set_px(buf, 0, (p2 + p1 + p0 + 2 * q0 + q1 + q2 + q3 + 4) >> 3);
            set_px(buf, 1, (p1 + p0 + q0 + 2 * q1 + q2 + q3 + q3 + 4) >> 3);
            set_px(buf, 2, (p0 + q0 + q1 + 2 * q2 + q3 + q3 + q3 + 4) >> 3);
        } else if wd == 6 && flat8in {
            // 6-tap filter
            set_px(buf, -2, (p2 + 2 * p2 + 2 * p1 + 2 * p0 + q0 + 4) >> 3);
            set_px(buf, -1, (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3);
            set_px(buf, 0, (p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3);
            set_px(buf, 1, (p0 + 2 * q0 + 2 * q1 + 2 * q2 + q2 + 4) >> 3);
        } else {
            // Narrow filter (4-tap)
            let hev = (p1 - p0).abs() > h || (q1 - q0).abs() > h;

            if hev {
                let f = iclip_diff(p1 - q1, 0);
                let f = iclip_diff(3 * (q0 - p0) + f, 0);

                let f1 = cmp::min(f + 4, 127) >> 3;
                let f2 = cmp::min(f + 3, 127) >> 3;

                set_px(buf, -1, p0 + f2);
                set_px(buf, 0, q0 - f1);
            } else {
                let f = iclip_diff(3 * (q0 - p0), 0);

                let f1 = cmp::min(f + 4, 127) >> 3;
                let f2 = cmp::min(f + 3, 127) >> 3;

                set_px(buf, -1, p0 + f2);
                set_px(buf, 0, q0 - f1);

                let f = (f1 + 1) >> 1;
                set_px(buf, -2, p1 + f);
                set_px(buf, 1, q1 - f);
            }
        }
    }
}

// ============================================================================
// SIMD inner loop filter for the wd=6 V-FILTER case (wd=6, strideb>1)
// ============================================================================

/// SIMD wd=6 loop filter for 8bpc V-FILTER direction.
/// Processes 4 filter positions (4 adjacent cols) in parallel. Loads p2..q2
/// contiguously (4-byte each), computes fm + flat8in, computes 6-tap filter
/// outputs, computes narrow filter fallback, mask-selects per lane.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn loop_filter_4_8bpc_wd6_simd_v(
    _token: Desktop64,
    buf: &mut [u8],
    base: usize,
    e: i32,
    i: i32,
    h: i32,
    strideb: isize,
) {
    let load4 = |off: isize| -> __m128i {
        let start = signed_idx(base, strideb * off);
        let bytes = [buf[start], buf[start + 1], buf[start + 2], buf[start + 3]];
        let as_i32 = i32::from_le_bytes(bytes);
        _mm_cvtepu8_epi32(_mm_cvtsi32_si128(as_i32))
    };

    let p2_v = load4(-3);
    let p1_v = load4(-2);
    let p0_v = load4(-1);
    let q0_v = load4(0);
    let q1_v = load4(1);
    let q2_v = load4(2);

    let i_v = _mm_set1_epi32(i);
    let e_v = _mm_set1_epi32(e);
    let h_v = _mm_set1_epi32(h);
    let f_v = _mm_set1_epi32(1); // flat threshold for 8bpc

    let abs = |a: __m128i, b: __m128i| _mm_abs_epi32(_mm_sub_epi32(a, b));

    let abs_p1p0 = abs(p1_v, p0_v);
    let abs_q1q0 = abs(q1_v, q0_v);
    let abs_p0q0 = abs(p0_v, q0_v);
    let abs_p1q1 = abs(p1_v, q1_v);
    let abs_p2p1 = abs(p2_v, p1_v);
    let abs_q2q1 = abs(q2_v, q1_v);

    let not_gt = |a: __m128i, b: __m128i| -> __m128i {
        _mm_andnot_si128(_mm_cmpgt_epi32(a, b), _mm_set1_epi32(-1))
    };

    // fm = (abs_p1p0<=i) && (abs_q1q0<=i) && (2*abs_p0q0 + abs_p1q1>>1 <= e)
    //      && (abs_p2p1<=i) && (abs_q2q1<=i)
    let m_p1p0 = not_gt(abs_p1p0, i_v);
    let m_q1q0 = not_gt(abs_q1q0, i_v);
    let val_ee = _mm_add_epi32(_mm_slli_epi32::<1>(abs_p0q0), _mm_srli_epi32::<1>(abs_p1q1));
    let m_val = not_gt(val_ee, e_v);
    let m_p2p1 = not_gt(abs_p2p1, i_v);
    let m_q2q1 = not_gt(abs_q2q1, i_v);
    let fm_mask = _mm_and_si128(
        _mm_and_si128(_mm_and_si128(m_p1p0, m_q1q0), m_val),
        _mm_and_si128(m_p2p1, m_q2q1),
    );

    // flat8in = abs(p2-p0)<=f && abs(p1-p0)<=f && abs(q1-q0)<=f && abs(q2-q0)<=f
    let abs_p2p0 = abs(p2_v, p0_v);
    let abs_q2q0 = abs(q2_v, q0_v);
    let flat_mask = _mm_and_si128(
        _mm_and_si128(not_gt(abs_p2p0, f_v), not_gt(abs_p1p0, f_v)),
        _mm_and_si128(not_gt(abs_q1q0, f_v), not_gt(abs_q2q0, f_v)),
    );

    // 6-tap filter outputs (used when fm && flat8in):
    //   out[-2] = (p2 + 2*p2 + 2*p1 + 2*p0 + q0 + 4) >> 3
    //   out[-1] = (p2 + 2*p1 + 2*p0 + 2*q0 + q1 + 4) >> 3
    //   out[ 0] = (p1 + 2*p0 + 2*q0 + 2*q1 + q2 + 4) >> 3
    //   out[ 1] = (p0 + 2*q0 + 2*q1 + 2*q2 + q2 + 4) >> 3
    let p2_3 = _mm_add_epi32(_mm_slli_epi32::<1>(p2_v), p2_v); // 3 * p2
    let c4 = _mm_set1_epi32(4);
    let dbl = |v: __m128i| _mm_slli_epi32::<1>(v); // 2*v

    let out_m2 = _mm_srai_epi32::<3>(_mm_add_epi32(
        _mm_add_epi32(
            _mm_add_epi32(p2_3, dbl(p1_v)),
            _mm_add_epi32(dbl(p0_v), q0_v),
        ),
        c4,
    ));
    let out_m1 = _mm_srai_epi32::<3>(_mm_add_epi32(
        _mm_add_epi32(
            _mm_add_epi32(p2_v, dbl(p1_v)),
            _mm_add_epi32(_mm_add_epi32(dbl(p0_v), dbl(q0_v)), q1_v),
        ),
        c4,
    ));
    let out_0 = _mm_srai_epi32::<3>(_mm_add_epi32(
        _mm_add_epi32(
            _mm_add_epi32(p1_v, dbl(p0_v)),
            _mm_add_epi32(_mm_add_epi32(dbl(q0_v), dbl(q1_v)), q2_v),
        ),
        c4,
    ));
    let q2_3 = _mm_add_epi32(_mm_slli_epi32::<1>(q2_v), q2_v); // 3 * q2 (substitutes for q2+q2 at the end)
    let out_1 = _mm_srai_epi32::<3>(_mm_add_epi32(
        _mm_add_epi32(
            _mm_add_epi32(p0_v, dbl(q0_v)),
            _mm_add_epi32(dbl(q1_v), q2_3),
        ),
        c4,
    ));

    // Narrow filter outputs (used when fm && !flat8in):
    let neg128 = _mm_set1_epi32(-128);
    let pos127 = _mm_set1_epi32(127);
    let iclip = |v: __m128i| _mm_min_epi32(_mm_max_epi32(v, neg128), pos127);

    let diff_q0p0 = _mm_sub_epi32(q0_v, p0_v);
    let three_d = _mm_add_epi32(_mm_slli_epi32::<1>(diff_q0p0), diff_q0p0);
    let diff_p1q1 = _mm_sub_epi32(p1_v, q1_v);

    let hev_mask = _mm_or_si128(
        _mm_cmpgt_epi32(abs_p1p0, h_v),
        _mm_cmpgt_epi32(abs_q1q0, h_v),
    );

    let f_hev = iclip(_mm_add_epi32(three_d, iclip(diff_p1q1)));
    let f_no = iclip(three_d);

    let c4i = _mm_set1_epi32(4);
    let c3i = _mm_set1_epi32(3);
    let one = _mm_set1_epi32(1);

    let f1_hev = _mm_srai_epi32::<3>(_mm_min_epi32(_mm_add_epi32(f_hev, c4i), pos127));
    let f2_hev = _mm_srai_epi32::<3>(_mm_min_epi32(_mm_add_epi32(f_hev, c3i), pos127));
    let f1_no = _mm_srai_epi32::<3>(_mm_min_epi32(_mm_add_epi32(f_no, c4i), pos127));
    let f2_no = _mm_srai_epi32::<3>(_mm_min_epi32(_mm_add_epi32(f_no, c3i), pos127));
    let f_extra = _mm_srai_epi32::<1>(_mm_add_epi32(f1_no, one));

    let p0_hev = _mm_add_epi32(p0_v, f2_hev);
    let q0_hev = _mm_sub_epi32(q0_v, f1_hev);
    let p0_no = _mm_add_epi32(p0_v, f2_no);
    let q0_no = _mm_sub_epi32(q0_v, f1_no);
    let p1_no = _mm_add_epi32(p1_v, f_extra);
    let q1_no = _mm_sub_epi32(q1_v, f_extra);

    let blendv = |a: __m128i, b: __m128i, mask: __m128i| -> __m128i {
        _mm_or_si128(_mm_andnot_si128(mask, a), _mm_and_si128(mask, b))
    };

    // Narrow outputs (selected by hev mask within !flat8in branch):
    let narrow_p1 = blendv(p1_no, p1_v, hev_mask);
    let narrow_p0 = blendv(p0_no, p0_hev, hev_mask);
    let narrow_q0 = blendv(q0_no, q0_hev, hev_mask);
    let narrow_q1 = blendv(q1_no, q1_v, hev_mask);

    // Select between 6-tap (flat8in) and narrow (!flat8in):
    //   For wd=6, only positions -2, -1, 0, 1 are written.
    //   At -2, -1: keep narrow's p1, p0 (narrow doesn't write -2 if hev)
    //   Wait — narrow only writes -1, 0 if hev; -2, -1, 0, 1 if !hev.
    //   6-tap writes -2, -1, 0, 1.
    //   So for "narrow path" outputs at -2 and 1: use p1_v / q1_v (unchanged) if hev,
    //     or p1_no / q1_no (updated) if !hev. blendv already does that.
    //   At -1 and 0: narrow always updates them. narrow_p0 / narrow_q0 are correct.
    let out_m2_sel = blendv(narrow_p1, out_m2, flat_mask);
    let out_m1_sel = blendv(narrow_p0, out_m1, flat_mask);
    let out_0_sel = blendv(narrow_q0, out_0, flat_mask);
    let out_1_sel = blendv(narrow_q1, out_1, flat_mask);

    // Apply fm mask: if !fm, keep original
    let final_p1 = blendv(p1_v, out_m2_sel, fm_mask);
    let final_p0 = blendv(p0_v, out_m1_sel, fm_mask);
    let final_q0 = blendv(q0_v, out_0_sel, fm_mask);
    let final_q1 = blendv(q1_v, out_1_sel, fm_mask);

    // Pack 4 i32 (clipped to [0,255]) -> 4 u8.
    let pack4 = |v: __m128i| -> i32 {
        let u16x4 = _mm_packus_epi32(v, v);
        let u8x4 = _mm_packus_epi16(u16x4, u16x4);
        _mm_cvtsi128_si32(u8x4)
    };
    let store4 = |buf: &mut [u8], packed: i32, off: isize| {
        let start = signed_idx(base, strideb * off);
        let bytes = packed.to_le_bytes();
        buf[start] = bytes[0];
        buf[start + 1] = bytes[1];
        buf[start + 2] = bytes[2];
        buf[start + 3] = bytes[3];
    };
    store4(buf, pack4(final_p1), -2);
    store4(buf, pack4(final_p0), -1);
    store4(buf, pack4(final_q0), 0);
    store4(buf, pack4(final_q1), 1);
}

// ============================================================================
// SIMD inner loop filter for the wd=8 V-FILTER case (wd=8, strideb>1)
// ============================================================================

/// SIMD wd=8 loop filter for 8bpc V-FILTER direction.
/// Processes 4 filter positions (4 adjacent cols) in parallel. Loads p3..q3
/// contiguously, computes fm + flat8in, computes 8-tap filter outputs (6 positions),
/// computes narrow filter fallback, mask-selects per lane.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn loop_filter_4_8bpc_wd8_simd_v(
    _token: Desktop64,
    buf: &mut [u8],
    base: usize,
    e: i32,
    i: i32,
    h: i32,
    strideb: isize,
) {
    let load4 = |off: isize| -> __m128i {
        let start = signed_idx(base, strideb * off);
        let bytes = [buf[start], buf[start + 1], buf[start + 2], buf[start + 3]];
        let as_i32 = i32::from_le_bytes(bytes);
        _mm_cvtepu8_epi32(_mm_cvtsi32_si128(as_i32))
    };

    let p3_v = load4(-4);
    let p2_v = load4(-3);
    let p1_v = load4(-2);
    let p0_v = load4(-1);
    let q0_v = load4(0);
    let q1_v = load4(1);
    let q2_v = load4(2);
    let q3_v = load4(3);

    let i_v = _mm_set1_epi32(i);
    let e_v = _mm_set1_epi32(e);
    let h_v = _mm_set1_epi32(h);
    let f_v = _mm_set1_epi32(1);

    let abs = |a: __m128i, b: __m128i| _mm_abs_epi32(_mm_sub_epi32(a, b));

    let abs_p1p0 = abs(p1_v, p0_v);
    let abs_q1q0 = abs(q1_v, q0_v);
    let abs_p0q0 = abs(p0_v, q0_v);
    let abs_p1q1 = abs(p1_v, q1_v);
    let abs_p2p1 = abs(p2_v, p1_v);
    let abs_q2q1 = abs(q2_v, q1_v);
    let abs_p3p2 = abs(p3_v, p2_v);
    let abs_q3q2 = abs(q3_v, q2_v);

    let not_gt = |a: __m128i, b: __m128i| -> __m128i {
        _mm_andnot_si128(_mm_cmpgt_epi32(a, b), _mm_set1_epi32(-1))
    };

    let m_p1p0 = not_gt(abs_p1p0, i_v);
    let m_q1q0 = not_gt(abs_q1q0, i_v);
    let val_ee = _mm_add_epi32(_mm_slli_epi32::<1>(abs_p0q0), _mm_srli_epi32::<1>(abs_p1q1));
    let m_val = not_gt(val_ee, e_v);
    let m_p2p1 = not_gt(abs_p2p1, i_v);
    let m_q2q1 = not_gt(abs_q2q1, i_v);
    let m_p3p2 = not_gt(abs_p3p2, i_v);
    let m_q3q2 = not_gt(abs_q3q2, i_v);
    let fm_mask = _mm_and_si128(
        _mm_and_si128(_mm_and_si128(m_p1p0, m_q1q0), m_val),
        _mm_and_si128(_mm_and_si128(m_p2p1, m_q2q1), _mm_and_si128(m_p3p2, m_q3q2)),
    );

    // flat8in = abs(p2-p0)<=f && abs(p1-p0)<=f && abs(q1-q0)<=f && abs(q2-q0)<=f
    //          && abs(p3-p0)<=f && abs(q3-q0)<=f
    let abs_p2p0 = abs(p2_v, p0_v);
    let abs_q2q0 = abs(q2_v, q0_v);
    let abs_p3p0 = abs(p3_v, p0_v);
    let abs_q3q0 = abs(q3_v, q0_v);
    let flat_mask = _mm_and_si128(
        _mm_and_si128(not_gt(abs_p2p0, f_v), not_gt(abs_p1p0, f_v)),
        _mm_and_si128(
            _mm_and_si128(not_gt(abs_q1q0, f_v), not_gt(abs_q2q0, f_v)),
            _mm_and_si128(not_gt(abs_p3p0, f_v), not_gt(abs_q3q0, f_v)),
        ),
    );

    // 8-tap filter outputs (positions -3..2):
    //   out[-3] = (p3 + p3 + p3 + 2*p2 + p1 + p0 + q0 + 4) >> 3
    //   out[-2] = (p3 + p3 + p2 + 2*p1 + p0 + q0 + q1 + 4) >> 3
    //   out[-1] = (p3 + p2 + p1 + 2*p0 + q0 + q1 + q2 + 4) >> 3
    //   out[ 0] = (p2 + p1 + p0 + 2*q0 + q1 + q2 + q3 + 4) >> 3
    //   out[ 1] = (p1 + p0 + q0 + 2*q1 + q2 + q3 + q3 + 4) >> 3
    //   out[ 2] = (p0 + q0 + q1 + 2*q2 + q3 + q3 + q3 + 4) >> 3
    let dbl = |v: __m128i| _mm_slli_epi32::<1>(v);
    let triple = |v: __m128i| _mm_add_epi32(dbl(v), v);
    let c4 = _mm_set1_epi32(4);
    let add = |a: __m128i, b: __m128i| _mm_add_epi32(a, b);
    let add3 = |a: __m128i, b: __m128i, c: __m128i| add(add(a, b), c);
    let add4 = |a: __m128i, b: __m128i, c: __m128i, d: __m128i| add(add(a, b), add(c, d));

    let out_m3 = _mm_srai_epi32::<3>(add(
        add4(triple(p3_v), dbl(p2_v), p1_v, p0_v),
        add(q0_v, c4),
    ));
    let out_m2 = _mm_srai_epi32::<3>(add(
        add4(dbl(p3_v), p2_v, dbl(p1_v), p0_v),
        add3(q0_v, q1_v, c4),
    ));
    let out_m1 = _mm_srai_epi32::<3>(add(
        add4(p3_v, p2_v, p1_v, dbl(p0_v)),
        add4(q0_v, q1_v, q2_v, c4),
    ));
    let out_0 = _mm_srai_epi32::<3>(add(
        add4(p2_v, p1_v, p0_v, dbl(q0_v)),
        add4(q1_v, q2_v, q3_v, c4),
    ));
    let out_1 = _mm_srai_epi32::<3>(add(
        add4(p1_v, p0_v, q0_v, dbl(q1_v)),
        add4(q2_v, q3_v, q3_v, c4),
    ));
    let out_2 = _mm_srai_epi32::<3>(add(
        add4(p0_v, q0_v, q1_v, dbl(q2_v)),
        add4(q3_v, q3_v, q3_v, c4),
    ));

    // Narrow filter (4-tap)
    let neg128 = _mm_set1_epi32(-128);
    let pos127 = _mm_set1_epi32(127);
    let iclip = |v: __m128i| _mm_min_epi32(_mm_max_epi32(v, neg128), pos127);

    let diff_q0p0 = _mm_sub_epi32(q0_v, p0_v);
    let three_d = _mm_add_epi32(_mm_slli_epi32::<1>(diff_q0p0), diff_q0p0);
    let diff_p1q1 = _mm_sub_epi32(p1_v, q1_v);

    let hev_mask = _mm_or_si128(
        _mm_cmpgt_epi32(abs_p1p0, h_v),
        _mm_cmpgt_epi32(abs_q1q0, h_v),
    );

    let f_hev = iclip(_mm_add_epi32(three_d, iclip(diff_p1q1)));
    let f_no = iclip(three_d);

    let c4i = _mm_set1_epi32(4);
    let c3i = _mm_set1_epi32(3);
    let one = _mm_set1_epi32(1);

    let f1_hev = _mm_srai_epi32::<3>(_mm_min_epi32(_mm_add_epi32(f_hev, c4i), pos127));
    let f2_hev = _mm_srai_epi32::<3>(_mm_min_epi32(_mm_add_epi32(f_hev, c3i), pos127));
    let f1_no = _mm_srai_epi32::<3>(_mm_min_epi32(_mm_add_epi32(f_no, c4i), pos127));
    let f2_no = _mm_srai_epi32::<3>(_mm_min_epi32(_mm_add_epi32(f_no, c3i), pos127));
    let f_extra = _mm_srai_epi32::<1>(_mm_add_epi32(f1_no, one));

    let p0_hev = _mm_add_epi32(p0_v, f2_hev);
    let q0_hev = _mm_sub_epi32(q0_v, f1_hev);
    let p0_no = _mm_add_epi32(p0_v, f2_no);
    let q0_no = _mm_sub_epi32(q0_v, f1_no);
    let p1_no = _mm_add_epi32(p1_v, f_extra);
    let q1_no = _mm_sub_epi32(q1_v, f_extra);

    let blendv = |a: __m128i, b: __m128i, mask: __m128i| -> __m128i {
        _mm_or_si128(_mm_andnot_si128(mask, a), _mm_and_si128(mask, b))
    };

    let narrow_p1 = blendv(p1_no, p1_v, hev_mask);
    let narrow_p0 = blendv(p0_no, p0_hev, hev_mask);
    let narrow_q0 = blendv(q0_no, q0_hev, hev_mask);
    let narrow_q1 = blendv(q1_no, q1_v, hev_mask);

    // Select between 8-tap (flat_mask) and narrow (otherwise):
    //   At -3, 2: narrow doesn't touch, keep p3/q2 from original
    //   At -2, 1: narrow output (depending on hev)
    //   At -1, 0: narrow output (always written)
    let out_m3_sel = blendv(p2_v, out_m3, flat_mask); // narrow keeps original p2 (= position -3 in 8-tap)
    let out_m2_sel = blendv(narrow_p1, out_m2, flat_mask);
    let out_m1_sel = blendv(narrow_p0, out_m1, flat_mask);
    let out_0_sel = blendv(narrow_q0, out_0, flat_mask);
    let out_1_sel = blendv(narrow_q1, out_1, flat_mask);
    let out_2_sel = blendv(q2_v, out_2, flat_mask);

    // Apply fm mask: if !fm, keep original
    let final_p2 = blendv(p2_v, out_m3_sel, fm_mask);
    let final_p1 = blendv(p1_v, out_m2_sel, fm_mask);
    let final_p0 = blendv(p0_v, out_m1_sel, fm_mask);
    let final_q0 = blendv(q0_v, out_0_sel, fm_mask);
    let final_q1 = blendv(q1_v, out_1_sel, fm_mask);
    let final_q2 = blendv(q2_v, out_2_sel, fm_mask);

    let pack4 = |v: __m128i| -> i32 {
        let u16x4 = _mm_packus_epi32(v, v);
        let u8x4 = _mm_packus_epi16(u16x4, u16x4);
        _mm_cvtsi128_si32(u8x4)
    };
    let store4 = |buf: &mut [u8], packed: i32, off: isize| {
        let start = signed_idx(base, strideb * off);
        let bytes = packed.to_le_bytes();
        buf[start] = bytes[0];
        buf[start + 1] = bytes[1];
        buf[start + 2] = bytes[2];
        buf[start + 3] = bytes[3];
    };
    // Wait — at -3 we should only store if flat (8-tap writes -3). Otherwise keep original.
    // The final_p2 already encodes this via blendv. But narrow doesn't touch -3 at all,
    // and fm_mask=0 keeps original. So always storing final_p2 is correct.
    store4(buf, pack4(final_p2), -3);
    store4(buf, pack4(final_p1), -2);
    store4(buf, pack4(final_p0), -1);
    store4(buf, pack4(final_q0), 0);
    store4(buf, pack4(final_q1), 1);
    store4(buf, pack4(final_q2), 2);
}

// ============================================================================
// SIMD inner loop filter for the wd=16 V-FILTER case (wd=16, strideb>1)
// ============================================================================

/// SIMD wd=16 loop filter for 8bpc V-FILTER direction.
/// The widest case: handles 14-tap filter (writes 12 outputs at positions -6..5),
/// 8-tap fallback (when flat8out=false but flat8in=true), narrow fallback
/// (when !flat8in), or original (when !fm).
#[cfg(target_arch = "x86_64")]
#[arcane]
fn loop_filter_4_8bpc_wd16_simd_v(
    _token: Desktop64,
    buf: &mut [u8],
    base: usize,
    e: i32,
    i: i32,
    h: i32,
    strideb: isize,
) {
    let load4 = |off: isize| -> __m128i {
        let start = signed_idx(base, strideb * off);
        let bytes = [buf[start], buf[start + 1], buf[start + 2], buf[start + 3]];
        let as_i32 = i32::from_le_bytes(bytes);
        _mm_cvtepu8_epi32(_mm_cvtsi32_si128(as_i32))
    };

    // Load all 14 pixels: p6..p0, q0..q6
    let p6_v = load4(-7);
    let p5_v = load4(-6);
    let p4_v = load4(-5);
    let p3_v = load4(-4);
    let p2_v = load4(-3);
    let p1_v = load4(-2);
    let p0_v = load4(-1);
    let q0_v = load4(0);
    let q1_v = load4(1);
    let q2_v = load4(2);
    let q3_v = load4(3);
    let q4_v = load4(4);
    let q5_v = load4(5);
    let q6_v = load4(6);

    let i_v = _mm_set1_epi32(i);
    let e_v = _mm_set1_epi32(e);
    let h_v = _mm_set1_epi32(h);
    let f_v = _mm_set1_epi32(1);

    let abs = |a: __m128i, b: __m128i| _mm_abs_epi32(_mm_sub_epi32(a, b));

    let abs_p1p0 = abs(p1_v, p0_v);
    let abs_q1q0 = abs(q1_v, q0_v);
    let abs_p0q0 = abs(p0_v, q0_v);
    let abs_p1q1 = abs(p1_v, q1_v);
    let abs_p2p1 = abs(p2_v, p1_v);
    let abs_q2q1 = abs(q2_v, q1_v);
    let abs_p3p2 = abs(p3_v, p2_v);
    let abs_q3q2 = abs(q3_v, q2_v);

    let not_gt = |a: __m128i, b: __m128i| -> __m128i {
        _mm_andnot_si128(_mm_cmpgt_epi32(a, b), _mm_set1_epi32(-1))
    };

    let m_p1p0 = not_gt(abs_p1p0, i_v);
    let m_q1q0 = not_gt(abs_q1q0, i_v);
    let val_ee = _mm_add_epi32(_mm_slli_epi32::<1>(abs_p0q0), _mm_srli_epi32::<1>(abs_p1q1));
    let m_val = not_gt(val_ee, e_v);
    let m_p2p1 = not_gt(abs_p2p1, i_v);
    let m_q2q1 = not_gt(abs_q2q1, i_v);
    let m_p3p2 = not_gt(abs_p3p2, i_v);
    let m_q3q2 = not_gt(abs_q3q2, i_v);
    let fm_mask = _mm_and_si128(
        _mm_and_si128(_mm_and_si128(m_p1p0, m_q1q0), m_val),
        _mm_and_si128(_mm_and_si128(m_p2p1, m_q2q1), _mm_and_si128(m_p3p2, m_q3q2)),
    );

    // flat8out = abs(p6/p5/p4-p0)<=f && abs(q4/q5/q6-q0)<=f
    let abs_p6p0 = abs(p6_v, p0_v);
    let abs_p5p0 = abs(p5_v, p0_v);
    let abs_p4p0 = abs(p4_v, p0_v);
    let abs_q4q0 = abs(q4_v, q0_v);
    let abs_q5q0 = abs(q5_v, q0_v);
    let abs_q6q0 = abs(q6_v, q0_v);
    let flat8out_mask = _mm_and_si128(
        _mm_and_si128(
            _mm_and_si128(not_gt(abs_p6p0, f_v), not_gt(abs_p5p0, f_v)),
            not_gt(abs_p4p0, f_v),
        ),
        _mm_and_si128(
            _mm_and_si128(not_gt(abs_q4q0, f_v), not_gt(abs_q5q0, f_v)),
            not_gt(abs_q6q0, f_v),
        ),
    );

    // flat8in = abs(p2/p1/q1/q2-p0/q0)<=f && abs(p3/q3-p0/q0)<=f
    let abs_p2p0 = abs(p2_v, p0_v);
    let abs_q2q0 = abs(q2_v, q0_v);
    let abs_p3p0 = abs(p3_v, p0_v);
    let abs_q3q0 = abs(q3_v, q0_v);
    let flat8in_mask = _mm_and_si128(
        _mm_and_si128(not_gt(abs_p2p0, f_v), not_gt(abs_p1p0, f_v)),
        _mm_and_si128(
            _mm_and_si128(not_gt(abs_q1q0, f_v), not_gt(abs_q2q0, f_v)),
            _mm_and_si128(not_gt(abs_p3p0, f_v), not_gt(abs_q3q0, f_v)),
        ),
    );

    // Helpers
    let dbl = |v: __m128i| _mm_slli_epi32::<1>(v);
    let add = |a: __m128i, b: __m128i| _mm_add_epi32(a, b);
    let add3 = |a: __m128i, b: __m128i, c: __m128i| add(add(a, b), c);
    let add4 = |a: __m128i, b: __m128i, c: __m128i, d: __m128i| add(add(a, b), add(c, d));
    let c4 = _mm_set1_epi32(4);
    let c8 = _mm_set1_epi32(8);

    // Wide 14-tap filter outputs (positions -6..5). Each is sum of 13 weighted
    // values + 8, shifted right by 4.
    //   out[-6] = (p6*6 + p6*2 + p5*2 + p4*2 + p3 + p2 + p1 + p0 + q0 + 8) >> 4
    // Note: in the scalar, "p6 + p6 + p6 + p6 + p6 + p6 * 2" simplifies to p6 * 7.
    //       But adding scalars and the "*2" terms separately, we get sum = p6*5 + p6*2 = p6*7.
    let p6_5 = _mm_add_epi32(
        _mm_add_epi32(_mm_add_epi32(p6_v, p6_v), _mm_add_epi32(p6_v, p6_v)),
        p6_v,
    ); // p6*5
    let q6_5 = _mm_add_epi32(
        _mm_add_epi32(_mm_add_epi32(q6_v, q6_v), _mm_add_epi32(q6_v, q6_v)),
        q6_v,
    ); // q6*5

    // out[-6] = p6*7 + p5*2 + p4*2 + p3 + p2 + p1 + p0 + q0 + 8 >> 4
    let mut s = add(p6_5, _mm_add_epi32(dbl(p6_v), dbl(p5_v)));
    s = add(s, dbl(p4_v));
    s = add(s, add4(p3_v, p2_v, p1_v, p0_v));
    s = add(s, add(q0_v, c8));
    let out_m6 = _mm_srai_epi32::<4>(s);

    // out[-5] = (p6*5 + p5*2 + p4*2 + p3*2 + p2 + p1 + p0 + q0 + q1 + 8) >> 4
    let mut s = add(p6_5, _mm_add_epi32(dbl(p5_v), dbl(p4_v)));
    s = add(s, dbl(p3_v));
    s = add(s, add4(p2_v, p1_v, p0_v, q0_v));
    s = add(s, add(q1_v, c8));
    let out_m5 = _mm_srai_epi32::<4>(s);

    // out[-4] = (p6*4 + p5 + p4*2 + p3*2 + p2*2 + p1 + p0 + q0 + q1 + q2 + 8) >> 4
    let p6_4 = _mm_add_epi32(dbl(p6_v), dbl(p6_v));
    let mut s = add(p6_4, p5_v);
    s = add(s, _mm_add_epi32(dbl(p4_v), dbl(p3_v)));
    s = add(s, dbl(p2_v));
    s = add(s, add4(p1_v, p0_v, q0_v, q1_v));
    s = add(s, add(q2_v, c8));
    let out_m4 = _mm_srai_epi32::<4>(s);

    // out[-3] = (p6*3 + p5 + p4 + p3*2 + p2*2 + p1*2 + p0 + q0 + q1 + q2 + q3 + 8) >> 4
    let p6_3 = add(dbl(p6_v), p6_v);
    let mut s = add(p6_3, _mm_add_epi32(p5_v, p4_v));
    s = add(s, _mm_add_epi32(dbl(p3_v), dbl(p2_v)));
    s = add(s, dbl(p1_v));
    s = add(s, add4(p0_v, q0_v, q1_v, q2_v));
    s = add(s, add(q3_v, c8));
    let out_m3 = _mm_srai_epi32::<4>(s);

    // out[-2] = (p6*2 + p5 + p4 + p3 + p2*2 + p1*2 + p0*2 + q0 + q1 + q2 + q3 + q4 + 8) >> 4
    let mut s = add(dbl(p6_v), p5_v);
    s = add(s, _mm_add_epi32(p4_v, p3_v));
    s = add(s, _mm_add_epi32(dbl(p2_v), dbl(p1_v)));
    s = add(s, dbl(p0_v));
    s = add(s, add4(q0_v, q1_v, q2_v, q3_v));
    s = add(s, add(q4_v, c8));
    let out_m2 = _mm_srai_epi32::<4>(s);

    // out[-1] = (p6 + p5 + p4 + p3 + p2 + p1*2 + p0*2 + q0*2 + q1 + q2 + q3 + q4 + q5 + 8) >> 4
    let mut s = add(p6_v, p5_v);
    s = add(s, _mm_add_epi32(p4_v, p3_v));
    s = add(s, p2_v);
    s = add(s, _mm_add_epi32(dbl(p1_v), dbl(p0_v)));
    s = add(s, dbl(q0_v));
    s = add(s, add4(q1_v, q2_v, q3_v, q4_v));
    s = add(s, add(q5_v, c8));
    let out_m1 = _mm_srai_epi32::<4>(s);

    // out[ 0] = (p5 + p4 + p3 + p2 + p1 + p0*2 + q0*2 + q1*2 + q2 + q3 + q4 + q5 + q6 + 8) >> 4
    let mut s = add(p5_v, p4_v);
    s = add(s, _mm_add_epi32(p3_v, p2_v));
    s = add(s, p1_v);
    s = add(s, _mm_add_epi32(dbl(p0_v), dbl(q0_v)));
    s = add(s, dbl(q1_v));
    s = add(s, add4(q2_v, q3_v, q4_v, q5_v));
    s = add(s, add(q6_v, c8));
    let out_0 = _mm_srai_epi32::<4>(s);

    // out[ 1] = (p4 + p3 + p2 + p1 + p0 + q0*2 + q1*2 + q2*2 + q3 + q4 + q5 + q6 + q6 + 8) >> 4
    let mut s = add(p4_v, p3_v);
    s = add(s, _mm_add_epi32(p2_v, p1_v));
    s = add(s, p0_v);
    s = add(s, _mm_add_epi32(dbl(q0_v), dbl(q1_v)));
    s = add(s, dbl(q2_v));
    s = add(s, add4(q3_v, q4_v, q5_v, q6_v));
    s = add(s, add(q6_v, c8));
    let out_1 = _mm_srai_epi32::<4>(s);

    // out[ 2] = (p3 + p2 + p1 + p0 + q0 + q1*2 + q2*2 + q3*2 + q4 + q5 + q6*3 + 8) >> 4
    let mut s = add(p3_v, p2_v);
    s = add(s, _mm_add_epi32(p1_v, p0_v));
    s = add(s, q0_v);
    s = add(s, _mm_add_epi32(dbl(q1_v), dbl(q2_v)));
    s = add(s, dbl(q3_v));
    let q6_3 = add(dbl(q6_v), q6_v);
    s = add(s, add3(q4_v, q5_v, q6_3));
    s = add(s, c8);
    let out_2 = _mm_srai_epi32::<4>(s);

    // out[ 3] = (p2 + p1 + p0 + q0 + q1 + q2*2 + q3*2 + q4*2 + q5 + q6*4 + 8) >> 4
    let q6_4 = _mm_add_epi32(dbl(q6_v), dbl(q6_v));
    let mut s = add(p2_v, p1_v);
    s = add(s, _mm_add_epi32(p0_v, q0_v));
    s = add(s, q1_v);
    s = add(s, _mm_add_epi32(dbl(q2_v), dbl(q3_v)));
    s = add(s, dbl(q4_v));
    s = add(s, add(q5_v, q6_4));
    s = add(s, c8);
    let out_3 = _mm_srai_epi32::<4>(s);

    // out[ 4] = (p1 + p0 + q0 + q1 + q2 + q3*2 + q4*2 + q5*2 + q6*5 + 8) >> 4
    let mut s = add(p1_v, p0_v);
    s = add(s, _mm_add_epi32(q0_v, q1_v));
    s = add(s, q2_v);
    s = add(s, _mm_add_epi32(dbl(q3_v), dbl(q4_v)));
    s = add(s, dbl(q5_v));
    s = add(s, q6_5);
    s = add(s, c8);
    let out_4 = _mm_srai_epi32::<4>(s);

    // out[ 5] = (p0 + q0 + q1 + q2 + q3 + q4*2 + q5*2 + q6*2 + q6*5 + 8) >> 4
    //        = (p0 + q0 + q1 + q2 + q3 + q4*2 + q5*2 + q6*7 + 8) >> 4
    let q6_7 = _mm_add_epi32(q6_5, _mm_add_epi32(q6_v, q6_v));
    let mut s = add(p0_v, q0_v);
    s = add(s, _mm_add_epi32(q1_v, q2_v));
    s = add(s, q3_v);
    s = add(s, _mm_add_epi32(dbl(q4_v), dbl(q5_v)));
    s = add(s, q6_7);
    s = add(s, c8);
    let out_5 = _mm_srai_epi32::<4>(s);

    // 8-tap filter outputs (positions -3..2) for !flat8out && flat8in case.
    // (same as wd=8 path)
    let triple = |v: __m128i| _mm_add_epi32(dbl(v), v);
    let out8_m3 = _mm_srai_epi32::<3>(add(
        add4(triple(p3_v), dbl(p2_v), p1_v, p0_v),
        add(q0_v, c4),
    ));
    let out8_m2 = _mm_srai_epi32::<3>(add(
        add4(dbl(p3_v), p2_v, dbl(p1_v), p0_v),
        add3(q0_v, q1_v, c4),
    ));
    let out8_m1 = _mm_srai_epi32::<3>(add(
        add4(p3_v, p2_v, p1_v, dbl(p0_v)),
        add4(q0_v, q1_v, q2_v, c4),
    ));
    let out8_0 = _mm_srai_epi32::<3>(add(
        add4(p2_v, p1_v, p0_v, dbl(q0_v)),
        add4(q1_v, q2_v, q3_v, c4),
    ));
    let out8_1 = _mm_srai_epi32::<3>(add(
        add4(p1_v, p0_v, q0_v, dbl(q1_v)),
        add4(q2_v, q3_v, q3_v, c4),
    ));
    let out8_2 = _mm_srai_epi32::<3>(add(
        add4(p0_v, q0_v, q1_v, dbl(q2_v)),
        add4(q3_v, q3_v, q3_v, c4),
    ));

    // Narrow filter (4-tap)
    let neg128 = _mm_set1_epi32(-128);
    let pos127 = _mm_set1_epi32(127);
    let iclip = |v: __m128i| _mm_min_epi32(_mm_max_epi32(v, neg128), pos127);
    let diff_q0p0 = _mm_sub_epi32(q0_v, p0_v);
    let three_d = _mm_add_epi32(_mm_slli_epi32::<1>(diff_q0p0), diff_q0p0);
    let diff_p1q1 = _mm_sub_epi32(p1_v, q1_v);
    let hev_mask = _mm_or_si128(
        _mm_cmpgt_epi32(abs_p1p0, h_v),
        _mm_cmpgt_epi32(abs_q1q0, h_v),
    );
    let f_hev = iclip(_mm_add_epi32(three_d, iclip(diff_p1q1)));
    let f_no = iclip(three_d);
    let c4i = c4;
    let c3i = _mm_set1_epi32(3);
    let one = _mm_set1_epi32(1);
    let f1_hev = _mm_srai_epi32::<3>(_mm_min_epi32(_mm_add_epi32(f_hev, c4i), pos127));
    let f2_hev = _mm_srai_epi32::<3>(_mm_min_epi32(_mm_add_epi32(f_hev, c3i), pos127));
    let f1_no = _mm_srai_epi32::<3>(_mm_min_epi32(_mm_add_epi32(f_no, c4i), pos127));
    let f2_no = _mm_srai_epi32::<3>(_mm_min_epi32(_mm_add_epi32(f_no, c3i), pos127));
    let f_extra = _mm_srai_epi32::<1>(_mm_add_epi32(f1_no, one));
    let p0_hev = _mm_add_epi32(p0_v, f2_hev);
    let q0_hev = _mm_sub_epi32(q0_v, f1_hev);
    let p0_no = _mm_add_epi32(p0_v, f2_no);
    let q0_no = _mm_sub_epi32(q0_v, f1_no);
    let p1_no = _mm_add_epi32(p1_v, f_extra);
    let q1_no = _mm_sub_epi32(q1_v, f_extra);

    let blendv = |a: __m128i, b: __m128i, mask: __m128i| -> __m128i {
        _mm_or_si128(_mm_andnot_si128(mask, a), _mm_and_si128(mask, b))
    };

    let narrow_p1 = blendv(p1_no, p1_v, hev_mask);
    let narrow_p0 = blendv(p0_no, p0_hev, hev_mask);
    let narrow_q0 = blendv(q0_no, q0_hev, hev_mask);
    let narrow_q1 = blendv(q1_no, q1_v, hev_mask);

    // Three-way select:
    //   flat8out && flat8in: 14-tap wide filter (writes positions -6..5)
    //   !flat8out && flat8in: 8-tap filter (writes -3..2)
    //   !flat8in: narrow filter (writes -1, 0 always; -2, 1 if !hev)
    //   !fm: keep original
    let wide_mask = _mm_and_si128(flat8out_mask, flat8in_mask);

    // First: pick between 8-tap and narrow based on flat8in (when not wide):
    let mid_m3 = blendv(p2_v, out8_m3, flat8in_mask); // narrow doesn't touch -3
    let mid_m2 = blendv(narrow_p1, out8_m2, flat8in_mask);
    let mid_m1 = blendv(narrow_p0, out8_m1, flat8in_mask);
    let mid_0 = blendv(narrow_q0, out8_0, flat8in_mask);
    let mid_1 = blendv(narrow_q1, out8_1, flat8in_mask);
    let mid_2 = blendv(q2_v, out8_2, flat8in_mask);

    // Then: pick between wide (14-tap) and mid based on wide_mask:
    let sel_m6 = blendv(p5_v, out_m6, wide_mask); // outside 8-tap range: keep original
    let sel_m5 = blendv(p4_v, out_m5, wide_mask);
    let sel_m4 = blendv(p3_v, out_m4, wide_mask);
    let sel_m3 = blendv(mid_m3, out_m3, wide_mask);
    let sel_m2 = blendv(mid_m2, out_m2, wide_mask);
    let sel_m1 = blendv(mid_m1, out_m1, wide_mask);
    let sel_0 = blendv(mid_0, out_0, wide_mask);
    let sel_1 = blendv(mid_1, out_1, wide_mask);
    let sel_2 = blendv(mid_2, out_2, wide_mask);
    let sel_3 = blendv(q3_v, out_3, wide_mask);
    let sel_4 = blendv(q4_v, out_4, wide_mask);
    let sel_5 = blendv(q5_v, out_5, wide_mask);

    // Apply fm mask: if !fm, keep original
    let final_m6 = blendv(p5_v, sel_m6, fm_mask);
    let final_m5 = blendv(p4_v, sel_m5, fm_mask);
    let final_m4 = blendv(p3_v, sel_m4, fm_mask);
    let final_m3 = blendv(p2_v, sel_m3, fm_mask);
    let final_m2 = blendv(p1_v, sel_m2, fm_mask);
    let final_m1 = blendv(p0_v, sel_m1, fm_mask);
    let final_0 = blendv(q0_v, sel_0, fm_mask);
    let final_1 = blendv(q1_v, sel_1, fm_mask);
    let final_2 = blendv(q2_v, sel_2, fm_mask);
    let final_3 = blendv(q3_v, sel_3, fm_mask);
    let final_4 = blendv(q4_v, sel_4, fm_mask);
    let final_5 = blendv(q5_v, sel_5, fm_mask);

    let pack4 = |v: __m128i| -> i32 {
        let u16x4 = _mm_packus_epi32(v, v);
        let u8x4 = _mm_packus_epi16(u16x4, u16x4);
        _mm_cvtsi128_si32(u8x4)
    };
    let store4 = |buf: &mut [u8], packed: i32, off: isize| {
        let start = signed_idx(base, strideb * off);
        let bytes = packed.to_le_bytes();
        buf[start] = bytes[0];
        buf[start + 1] = bytes[1];
        buf[start + 2] = bytes[2];
        buf[start + 3] = bytes[3];
    };
    store4(buf, pack4(final_m6), -6);
    store4(buf, pack4(final_m5), -5);
    store4(buf, pack4(final_m4), -4);
    store4(buf, pack4(final_m3), -3);
    store4(buf, pack4(final_m2), -2);
    store4(buf, pack4(final_m1), -1);
    store4(buf, pack4(final_0), 0);
    store4(buf, pack4(final_1), 1);
    store4(buf, pack4(final_2), 2);
    store4(buf, pack4(final_3), 3);
    store4(buf, pack4(final_4), 4);
    store4(buf, pack4(final_5), 5);
}

// ============================================================================
// SIMD inner loop filter for the narrow 4-tap H-FILTER case (wd=4, stridea==stride)
// ============================================================================

/// SIMD narrow 4-tap loop filter for 8bpc H-FILTER direction.
/// In h-filter, 4 filter positions are 4 different ROWS (stridea=stride) and
/// the filter pixels are at column offsets -2/-1/0/1 (strideb=1, contiguous).
/// We load 4 contiguous bytes per row (each row holds one lane's p1/p0/q0/q1),
/// transpose 4x4 to get pixel-position vectors, do the same SIMD compute as
/// the v-filter narrow path, then transpose back to row layout and store.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn loop_filter_4_8bpc_narrow_simd_h(
    _token: Desktop64,
    buf: &mut [u8],
    base: usize,
    e: i32,
    i: i32,
    h: i32,
    stridea: isize,
) {
    // Load one row of 4 bytes as 4 i32 lanes per row.
    let load_row = |row: isize| -> __m128i {
        let start = signed_idx(base, row * stridea - 2);
        let bytes = [buf[start], buf[start + 1], buf[start + 2], buf[start + 3]];
        let as_i32 = i32::from_le_bytes(bytes);
        _mm_cvtepu8_epi32(_mm_cvtsi32_si128(as_i32))
    };

    // row_v[k] = [p1_k, p0_k, q0_k, q1_k] (4 i32 lanes from row k)
    let r0 = load_row(0);
    let r1 = load_row(1);
    let r2 = load_row(2);
    let r3 = load_row(3);

    // Transpose 4x4 i32: rows -> columns
    // r0=[a0,a1,a2,a3], r1=[b0,b1,b2,b3], r2=[c0,c1,c2,c3], r3=[d0,d1,d2,d3]
    // Want: out0=[a0,b0,c0,d0], out1=[a1,b1,c1,d1], etc.
    let t0 = _mm_unpacklo_epi32(r0, r1); // [a0,b0,a1,b1]
    let t1 = _mm_unpackhi_epi32(r0, r1); // [a2,b2,a3,b3]
    let t2 = _mm_unpacklo_epi32(r2, r3); // [c0,d0,c1,d1]
    let t3 = _mm_unpackhi_epi32(r2, r3); // [c2,d2,c3,d3]
    let p1_v = _mm_unpacklo_epi64(t0, t2); // [a0,b0,c0,d0]
    let p0_v = _mm_unpackhi_epi64(t0, t2); // [a1,b1,c1,d1]
    let q0_v = _mm_unpacklo_epi64(t1, t3); // [a2,b2,c2,d2]
    let q1_v = _mm_unpackhi_epi64(t1, t3); // [a3,b3,c3,d3]

    // Same SIMD compute as v-filter narrow path
    let i_v = _mm_set1_epi32(i);
    let e_v = _mm_set1_epi32(e);
    let h_v = _mm_set1_epi32(h);

    let abs_p1p0 = _mm_abs_epi32(_mm_sub_epi32(p1_v, p0_v));
    let abs_q1q0 = _mm_abs_epi32(_mm_sub_epi32(q1_v, q0_v));
    let abs_p0q0 = _mm_abs_epi32(_mm_sub_epi32(p0_v, q0_v));
    let abs_p1q1 = _mm_abs_epi32(_mm_sub_epi32(p1_v, q1_v));

    let not_gt = |a: __m128i, b: __m128i| -> __m128i {
        _mm_andnot_si128(_mm_cmpgt_epi32(a, b), _mm_set1_epi32(-1))
    };
    let m_p1p0 = not_gt(abs_p1p0, i_v);
    let m_q1q0 = not_gt(abs_q1q0, i_v);
    let val = _mm_add_epi32(_mm_slli_epi32::<1>(abs_p0q0), _mm_srli_epi32::<1>(abs_p1q1));
    let m_val = not_gt(val, e_v);
    let fm_mask = _mm_and_si128(_mm_and_si128(m_p1p0, m_q1q0), m_val);

    let hev_mask = _mm_or_si128(
        _mm_cmpgt_epi32(abs_p1p0, h_v),
        _mm_cmpgt_epi32(abs_q1q0, h_v),
    );

    let neg128 = _mm_set1_epi32(-128);
    let pos127 = _mm_set1_epi32(127);
    let iclip = |v: __m128i| _mm_min_epi32(_mm_max_epi32(v, neg128), pos127);

    let diff_q0p0 = _mm_sub_epi32(q0_v, p0_v);
    let three_d = _mm_add_epi32(_mm_slli_epi32::<1>(diff_q0p0), diff_q0p0);
    let diff_p1q1 = _mm_sub_epi32(p1_v, q1_v);

    let f_hev = iclip(_mm_add_epi32(three_d, iclip(diff_p1q1)));
    let f_no = iclip(three_d);

    let c4 = _mm_set1_epi32(4);
    let c3 = _mm_set1_epi32(3);
    let one = _mm_set1_epi32(1);

    let f1_hev = _mm_srai_epi32::<3>(_mm_min_epi32(_mm_add_epi32(f_hev, c4), pos127));
    let f2_hev = _mm_srai_epi32::<3>(_mm_min_epi32(_mm_add_epi32(f_hev, c3), pos127));
    let f1_no = _mm_srai_epi32::<3>(_mm_min_epi32(_mm_add_epi32(f_no, c4), pos127));
    let f2_no = _mm_srai_epi32::<3>(_mm_min_epi32(_mm_add_epi32(f_no, c3), pos127));
    let f_extra = _mm_srai_epi32::<1>(_mm_add_epi32(f1_no, one));

    let p0_hev = _mm_add_epi32(p0_v, f2_hev);
    let q0_hev = _mm_sub_epi32(q0_v, f1_hev);
    let p0_no = _mm_add_epi32(p0_v, f2_no);
    let q0_no = _mm_sub_epi32(q0_v, f1_no);
    let p1_no = _mm_add_epi32(p1_v, f_extra);
    let q1_no = _mm_sub_epi32(q1_v, f_extra);

    let blendv = |a: __m128i, b: __m128i, mask: __m128i| -> __m128i {
        _mm_or_si128(_mm_andnot_si128(mask, a), _mm_and_si128(mask, b))
    };
    let p1_filt = blendv(p1_no, p1_v, hev_mask);
    let p0_filt = blendv(p0_no, p0_hev, hev_mask);
    let q0_filt = blendv(q0_no, q0_hev, hev_mask);
    let q1_filt = blendv(q1_no, q1_v, hev_mask);

    let p1_final = blendv(p1_v, p1_filt, fm_mask);
    let p0_final = blendv(p0_v, p0_filt, fm_mask);
    let q0_final = blendv(q0_v, q0_filt, fm_mask);
    let q1_final = blendv(q1_v, q1_filt, fm_mask);

    // Clip to [0, 255]
    let zero = _mm_setzero_si128();
    let max_u8 = _mm_set1_epi32(255);
    let clip_u8 = |v: __m128i| _mm_min_epi32(_mm_max_epi32(v, zero), max_u8);
    let p1_final = clip_u8(p1_final);
    let p0_final = clip_u8(p0_final);
    let q0_final = clip_u8(q0_final);
    let q1_final = clip_u8(q1_final);

    // Transpose back: pixel-position vectors -> row vectors
    let t0 = _mm_unpacklo_epi32(p1_final, p0_final); // [a0,a1,b0,b1]
    let t1 = _mm_unpackhi_epi32(p1_final, p0_final); // [c0,c1,d0,d1]
    let t2 = _mm_unpacklo_epi32(q0_final, q1_final); // [a2,a3,b2,b3]
    let t3 = _mm_unpackhi_epi32(q0_final, q1_final); // [c2,c3,d2,d3]
    let row0 = _mm_unpacklo_epi64(t0, t2); // [a0,a1,a2,a3]
    let row1 = _mm_unpackhi_epi64(t0, t2); // [b0,b1,b2,b3]
    let row2 = _mm_unpacklo_epi64(t1, t3); // [c0,c1,c2,c3]
    let row3 = _mm_unpackhi_epi64(t1, t3); // [d0,d1,d2,d3]

    // Each row vector has 4 i32 lanes (already clipped to [0,255]). Pack to 4 u8.
    let pack_row = |v: __m128i| -> i32 {
        let u16x4 = _mm_packus_epi32(v, v);
        let u8x4 = _mm_packus_epi16(u16x4, u16x4);
        _mm_cvtsi128_si32(u8x4)
    };
    let store_row = |buf: &mut [u8], packed: i32, row: isize| {
        let start = signed_idx(base, row * stridea - 2);
        let bytes = packed.to_le_bytes();
        buf[start] = bytes[0];
        buf[start + 1] = bytes[1];
        buf[start + 2] = bytes[2];
        buf[start + 3] = bytes[3];
    };
    store_row(buf, pack_row(row0), 0);
    store_row(buf, pack_row(row1), 1);
    store_row(buf, pack_row(row2), 2);
    store_row(buf, pack_row(row3), 3);
}

// ============================================================================
// SIMD inner loop filter for the wd=8 H-FILTER case (wd=8, stridea==stride)
// ============================================================================

/// SIMD wd=8 loop filter for 8bpc H-FILTER direction.
/// Per row k, load 8 contiguous bytes (p3..q3) as two __m128i (4 i32 lanes each).
/// Do two 4x4 transposes to get 8 pixel-position vectors, run the same compute
/// as the v-filter wd=8 path, then transpose back to row layout for store.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn loop_filter_4_8bpc_wd8_simd_h(
    _token: Desktop64,
    buf: &mut [u8],
    base: usize,
    e: i32,
    i: i32,
    h: i32,
    stridea: isize,
) {
    // Per row k, load p3..q3 = 8 contiguous bytes at offset base + k*stridea - 4 .. +4
    let load_row_lo = |row: isize| -> __m128i {
        let start = signed_idx(base, row * stridea - 4);
        let bytes = [buf[start], buf[start + 1], buf[start + 2], buf[start + 3]];
        let as_i32 = i32::from_le_bytes(bytes);
        _mm_cvtepu8_epi32(_mm_cvtsi32_si128(as_i32))
    };
    let load_row_hi = |row: isize| -> __m128i {
        let start = signed_idx(base, row * stridea);
        let bytes = [buf[start], buf[start + 1], buf[start + 2], buf[start + 3]];
        let as_i32 = i32::from_le_bytes(bytes);
        _mm_cvtepu8_epi32(_mm_cvtsi32_si128(as_i32))
    };

    let r0_lo = load_row_lo(0); // row 0: [p3, p2, p1, p0]
    let r1_lo = load_row_lo(1);
    let r2_lo = load_row_lo(2);
    let r3_lo = load_row_lo(3);
    let r0_hi = load_row_hi(0); // row 0: [q0, q1, q2, q3]
    let r1_hi = load_row_hi(1);
    let r2_hi = load_row_hi(2);
    let r3_hi = load_row_hi(3);

    // Transpose 4x4 i32: rows -> columns
    let transpose4 = |r0: __m128i, r1: __m128i, r2: __m128i, r3: __m128i| -> [__m128i; 4] {
        let t0 = _mm_unpacklo_epi32(r0, r1);
        let t1 = _mm_unpackhi_epi32(r0, r1);
        let t2 = _mm_unpacklo_epi32(r2, r3);
        let t3 = _mm_unpackhi_epi32(r2, r3);
        [
            _mm_unpacklo_epi64(t0, t2),
            _mm_unpackhi_epi64(t0, t2),
            _mm_unpacklo_epi64(t1, t3),
            _mm_unpackhi_epi64(t1, t3),
        ]
    };

    let lo = transpose4(r0_lo, r1_lo, r2_lo, r3_lo);
    let hi = transpose4(r0_hi, r1_hi, r2_hi, r3_hi);
    let p3_v = lo[0];
    let p2_v = lo[1];
    let p1_v = lo[2];
    let p0_v = lo[3];
    let q0_v = hi[0];
    let q1_v = hi[1];
    let q2_v = hi[2];
    let q3_v = hi[3];

    // Same compute as v-filter wd=8
    let i_v = _mm_set1_epi32(i);
    let e_v = _mm_set1_epi32(e);
    let h_v = _mm_set1_epi32(h);
    let f_v = _mm_set1_epi32(1);

    let abs = |a: __m128i, b: __m128i| _mm_abs_epi32(_mm_sub_epi32(a, b));

    let abs_p1p0 = abs(p1_v, p0_v);
    let abs_q1q0 = abs(q1_v, q0_v);
    let abs_p0q0 = abs(p0_v, q0_v);
    let abs_p1q1 = abs(p1_v, q1_v);
    let abs_p2p1 = abs(p2_v, p1_v);
    let abs_q2q1 = abs(q2_v, q1_v);
    let abs_p3p2 = abs(p3_v, p2_v);
    let abs_q3q2 = abs(q3_v, q2_v);

    let not_gt = |a: __m128i, b: __m128i| -> __m128i {
        _mm_andnot_si128(_mm_cmpgt_epi32(a, b), _mm_set1_epi32(-1))
    };

    let m_p1p0 = not_gt(abs_p1p0, i_v);
    let m_q1q0 = not_gt(abs_q1q0, i_v);
    let val_ee = _mm_add_epi32(_mm_slli_epi32::<1>(abs_p0q0), _mm_srli_epi32::<1>(abs_p1q1));
    let m_val = not_gt(val_ee, e_v);
    let m_p2p1 = not_gt(abs_p2p1, i_v);
    let m_q2q1 = not_gt(abs_q2q1, i_v);
    let m_p3p2 = not_gt(abs_p3p2, i_v);
    let m_q3q2 = not_gt(abs_q3q2, i_v);
    let fm_mask = _mm_and_si128(
        _mm_and_si128(_mm_and_si128(m_p1p0, m_q1q0), m_val),
        _mm_and_si128(_mm_and_si128(m_p2p1, m_q2q1), _mm_and_si128(m_p3p2, m_q3q2)),
    );

    let abs_p2p0 = abs(p2_v, p0_v);
    let abs_q2q0 = abs(q2_v, q0_v);
    let abs_p3p0 = abs(p3_v, p0_v);
    let abs_q3q0 = abs(q3_v, q0_v);
    let flat_mask = _mm_and_si128(
        _mm_and_si128(not_gt(abs_p2p0, f_v), not_gt(abs_p1p0, f_v)),
        _mm_and_si128(
            _mm_and_si128(not_gt(abs_q1q0, f_v), not_gt(abs_q2q0, f_v)),
            _mm_and_si128(not_gt(abs_p3p0, f_v), not_gt(abs_q3q0, f_v)),
        ),
    );

    let dbl = |v: __m128i| _mm_slli_epi32::<1>(v);
    let triple = |v: __m128i| _mm_add_epi32(dbl(v), v);
    let c4 = _mm_set1_epi32(4);
    let add = |a: __m128i, b: __m128i| _mm_add_epi32(a, b);
    let add3 = |a: __m128i, b: __m128i, c: __m128i| add(add(a, b), c);
    let add4 = |a: __m128i, b: __m128i, c: __m128i, d: __m128i| add(add(a, b), add(c, d));

    let out_m3 = _mm_srai_epi32::<3>(add(
        add4(triple(p3_v), dbl(p2_v), p1_v, p0_v),
        add(q0_v, c4),
    ));
    let out_m2 = _mm_srai_epi32::<3>(add(
        add4(dbl(p3_v), p2_v, dbl(p1_v), p0_v),
        add3(q0_v, q1_v, c4),
    ));
    let out_m1 = _mm_srai_epi32::<3>(add(
        add4(p3_v, p2_v, p1_v, dbl(p0_v)),
        add4(q0_v, q1_v, q2_v, c4),
    ));
    let out_0 = _mm_srai_epi32::<3>(add(
        add4(p2_v, p1_v, p0_v, dbl(q0_v)),
        add4(q1_v, q2_v, q3_v, c4),
    ));
    let out_1 = _mm_srai_epi32::<3>(add(
        add4(p1_v, p0_v, q0_v, dbl(q1_v)),
        add4(q2_v, q3_v, q3_v, c4),
    ));
    let out_2 = _mm_srai_epi32::<3>(add(
        add4(p0_v, q0_v, q1_v, dbl(q2_v)),
        add4(q3_v, q3_v, q3_v, c4),
    ));

    let neg128 = _mm_set1_epi32(-128);
    let pos127 = _mm_set1_epi32(127);
    let iclip = |v: __m128i| _mm_min_epi32(_mm_max_epi32(v, neg128), pos127);
    let diff_q0p0 = _mm_sub_epi32(q0_v, p0_v);
    let three_d = _mm_add_epi32(_mm_slli_epi32::<1>(diff_q0p0), diff_q0p0);
    let diff_p1q1 = _mm_sub_epi32(p1_v, q1_v);
    let hev_mask = _mm_or_si128(
        _mm_cmpgt_epi32(abs_p1p0, h_v),
        _mm_cmpgt_epi32(abs_q1q0, h_v),
    );
    let f_hev = iclip(_mm_add_epi32(three_d, iclip(diff_p1q1)));
    let f_no = iclip(three_d);
    let c4i = _mm_set1_epi32(4);
    let c3i = _mm_set1_epi32(3);
    let one = _mm_set1_epi32(1);
    let f1_hev = _mm_srai_epi32::<3>(_mm_min_epi32(_mm_add_epi32(f_hev, c4i), pos127));
    let f2_hev = _mm_srai_epi32::<3>(_mm_min_epi32(_mm_add_epi32(f_hev, c3i), pos127));
    let f1_no = _mm_srai_epi32::<3>(_mm_min_epi32(_mm_add_epi32(f_no, c4i), pos127));
    let f2_no = _mm_srai_epi32::<3>(_mm_min_epi32(_mm_add_epi32(f_no, c3i), pos127));
    let f_extra = _mm_srai_epi32::<1>(_mm_add_epi32(f1_no, one));
    let p0_hev = _mm_add_epi32(p0_v, f2_hev);
    let q0_hev = _mm_sub_epi32(q0_v, f1_hev);
    let p0_no = _mm_add_epi32(p0_v, f2_no);
    let q0_no = _mm_sub_epi32(q0_v, f1_no);
    let p1_no = _mm_add_epi32(p1_v, f_extra);
    let q1_no = _mm_sub_epi32(q1_v, f_extra);

    let blendv = |a: __m128i, b: __m128i, mask: __m128i| -> __m128i {
        _mm_or_si128(_mm_andnot_si128(mask, a), _mm_and_si128(mask, b))
    };

    let narrow_p1 = blendv(p1_no, p1_v, hev_mask);
    let narrow_p0 = blendv(p0_no, p0_hev, hev_mask);
    let narrow_q0 = blendv(q0_no, q0_hev, hev_mask);
    let narrow_q1 = blendv(q1_no, q1_v, hev_mask);

    let out_m3_sel = blendv(p2_v, out_m3, flat_mask);
    let out_m2_sel = blendv(narrow_p1, out_m2, flat_mask);
    let out_m1_sel = blendv(narrow_p0, out_m1, flat_mask);
    let out_0_sel = blendv(narrow_q0, out_0, flat_mask);
    let out_1_sel = blendv(narrow_q1, out_1, flat_mask);
    let out_2_sel = blendv(q2_v, out_2, flat_mask);

    let final_p2 = blendv(p2_v, out_m3_sel, fm_mask);
    let final_p1 = blendv(p1_v, out_m2_sel, fm_mask);
    let final_p0 = blendv(p0_v, out_m1_sel, fm_mask);
    let final_q0 = blendv(q0_v, out_0_sel, fm_mask);
    let final_q1 = blendv(q1_v, out_1_sel, fm_mask);
    let final_q2 = blendv(q2_v, out_2_sel, fm_mask);

    // p3 and q3 are unchanged by 8-tap filter; keep originals.
    let final_p3 = p3_v;
    let final_q3 = q3_v;

    // Clip to [0, 255] (8-tap outputs may exceed range)
    let zero = _mm_setzero_si128();
    let max_u8 = _mm_set1_epi32(255);
    let clip_u8 = |v: __m128i| _mm_min_epi32(_mm_max_epi32(v, zero), max_u8);
    let final_p2 = clip_u8(final_p2);
    let final_p1 = clip_u8(final_p1);
    let final_p0 = clip_u8(final_p0);
    let final_q0 = clip_u8(final_q0);
    let final_q1 = clip_u8(final_q1);
    let final_q2 = clip_u8(final_q2);

    // Transpose back: pixel-position vectors -> row vectors
    let row_back_lo = {
        let t0 = _mm_unpacklo_epi32(final_p3, final_p2); // [a_p3, a_p2, b_p3, b_p2]
        let t1 = _mm_unpackhi_epi32(final_p3, final_p2); // [c_p3, c_p2, d_p3, d_p2]
        let t2 = _mm_unpacklo_epi32(final_p1, final_p0); // [a_p1, a_p0, b_p1, b_p0]
        let t3 = _mm_unpackhi_epi32(final_p1, final_p0); // [c_p1, c_p0, d_p1, d_p0]
        [
            _mm_unpacklo_epi64(t0, t2), // row 0: [p3, p2, p1, p0]
            _mm_unpackhi_epi64(t0, t2), // row 1
            _mm_unpacklo_epi64(t1, t3), // row 2
            _mm_unpackhi_epi64(t1, t3), // row 3
        ]
    };
    let row_back_hi = {
        let t0 = _mm_unpacklo_epi32(final_q0, final_q1);
        let t1 = _mm_unpackhi_epi32(final_q0, final_q1);
        let t2 = _mm_unpacklo_epi32(final_q2, final_q3);
        let t3 = _mm_unpackhi_epi32(final_q2, final_q3);
        [
            _mm_unpacklo_epi64(t0, t2),
            _mm_unpackhi_epi64(t0, t2),
            _mm_unpacklo_epi64(t1, t3),
            _mm_unpackhi_epi64(t1, t3),
        ]
    };

    let pack_row = |v: __m128i| -> i32 {
        let u16x4 = _mm_packus_epi32(v, v);
        let u8x4 = _mm_packus_epi16(u16x4, u16x4);
        _mm_cvtsi128_si32(u8x4)
    };
    let store_row = |buf: &mut [u8], packed_lo: i32, packed_hi: i32, row: isize| {
        let start_lo = signed_idx(base, row * stridea - 4);
        let bytes_lo = packed_lo.to_le_bytes();
        buf[start_lo] = bytes_lo[0];
        buf[start_lo + 1] = bytes_lo[1];
        buf[start_lo + 2] = bytes_lo[2];
        buf[start_lo + 3] = bytes_lo[3];
        let start_hi = signed_idx(base, row * stridea);
        let bytes_hi = packed_hi.to_le_bytes();
        buf[start_hi] = bytes_hi[0];
        buf[start_hi + 1] = bytes_hi[1];
        buf[start_hi + 2] = bytes_hi[2];
        buf[start_hi + 3] = bytes_hi[3];
    };
    store_row(buf, pack_row(row_back_lo[0]), pack_row(row_back_hi[0]), 0);
    store_row(buf, pack_row(row_back_lo[1]), pack_row(row_back_hi[1]), 1);
    store_row(buf, pack_row(row_back_lo[2]), pack_row(row_back_hi[2]), 2);
    store_row(buf, pack_row(row_back_lo[3]), pack_row(row_back_hi[3]), 3);
}

// ============================================================================
// SIMD inner loop filter for the wd=16 H-FILTER case (wd=16, stridea==stride)
// ============================================================================

/// SIMD wd=16 loop filter for 8bpc H-FILTER direction.
/// Per row, load 16 contiguous bytes (p6..q6 plus 2 unused) as 4 __m128i,
/// transpose 4 chunks × 4 rows × 4 cols into pixel-position vectors,
/// compute (same as v-filter wd=16), transpose back, store.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn loop_filter_4_8bpc_wd16_simd_h(
    _token: Desktop64,
    buf: &mut [u8],
    base: usize,
    e: i32,
    i: i32,
    h: i32,
    stridea: isize,
) {
    // Load 4 i32 lanes per chunk per row. Each row has 16 bytes covering -7..8.
    let load_chunk = |row: isize, chunk_off: isize| -> __m128i {
        let start = signed_idx(base, row * stridea + chunk_off);
        let bytes = [buf[start], buf[start + 1], buf[start + 2], buf[start + 3]];
        let as_i32 = i32::from_le_bytes(bytes);
        _mm_cvtepu8_epi32(_mm_cvtsi32_si128(as_i32))
    };

    // chunk_off values: -7 (p6..p3), -3 (p2..q0), 1 (q1..q4), 5 (q5..q6+pad)
    let r0_c0 = load_chunk(0, -7);
    let r1_c0 = load_chunk(1, -7);
    let r2_c0 = load_chunk(2, -7);
    let r3_c0 = load_chunk(3, -7);
    let r0_c1 = load_chunk(0, -3);
    let r1_c1 = load_chunk(1, -3);
    let r2_c1 = load_chunk(2, -3);
    let r3_c1 = load_chunk(3, -3);
    let r0_c2 = load_chunk(0, 1);
    let r1_c2 = load_chunk(1, 1);
    let r2_c2 = load_chunk(2, 1);
    let r3_c2 = load_chunk(3, 1);
    let r0_c3 = load_chunk(0, 5);
    let r1_c3 = load_chunk(1, 5);
    let r2_c3 = load_chunk(2, 5);
    let r3_c3 = load_chunk(3, 5);

    let transpose4 = |r0: __m128i, r1: __m128i, r2: __m128i, r3: __m128i| -> [__m128i; 4] {
        let t0 = _mm_unpacklo_epi32(r0, r1);
        let t1 = _mm_unpackhi_epi32(r0, r1);
        let t2 = _mm_unpacklo_epi32(r2, r3);
        let t3 = _mm_unpackhi_epi32(r2, r3);
        [
            _mm_unpacklo_epi64(t0, t2),
            _mm_unpackhi_epi64(t0, t2),
            _mm_unpacklo_epi64(t1, t3),
            _mm_unpackhi_epi64(t1, t3),
        ]
    };
    let c0 = transpose4(r0_c0, r1_c0, r2_c0, r3_c0);
    let c1 = transpose4(r0_c1, r1_c1, r2_c1, r3_c1);
    let c2 = transpose4(r0_c2, r1_c2, r2_c2, r3_c2);
    let c3 = transpose4(r0_c3, r1_c3, r2_c3, r3_c3);

    let p6_v = c0[0];
    let p5_v = c0[1];
    let p4_v = c0[2];
    let p3_v = c0[3];
    let p2_v = c1[0];
    let p1_v = c1[1];
    let p0_v = c1[2];
    let q0_v = c1[3];
    let q1_v = c2[0];
    let q2_v = c2[1];
    let q3_v = c2[2];
    let q4_v = c2[3];
    let q5_v = c3[0];
    let q6_v = c3[1];
    // c3[2], c3[3] are padding (unused)

    // Same compute body as v-filter wd=16
    let i_v = _mm_set1_epi32(i);
    let e_v = _mm_set1_epi32(e);
    let h_v = _mm_set1_epi32(h);
    let f_v = _mm_set1_epi32(1);

    let abs = |a: __m128i, b: __m128i| _mm_abs_epi32(_mm_sub_epi32(a, b));

    let abs_p1p0 = abs(p1_v, p0_v);
    let abs_q1q0 = abs(q1_v, q0_v);
    let abs_p0q0 = abs(p0_v, q0_v);
    let abs_p1q1 = abs(p1_v, q1_v);
    let abs_p2p1 = abs(p2_v, p1_v);
    let abs_q2q1 = abs(q2_v, q1_v);
    let abs_p3p2 = abs(p3_v, p2_v);
    let abs_q3q2 = abs(q3_v, q2_v);

    let not_gt = |a: __m128i, b: __m128i| -> __m128i {
        _mm_andnot_si128(_mm_cmpgt_epi32(a, b), _mm_set1_epi32(-1))
    };

    let m_p1p0 = not_gt(abs_p1p0, i_v);
    let m_q1q0 = not_gt(abs_q1q0, i_v);
    let val_ee = _mm_add_epi32(_mm_slli_epi32::<1>(abs_p0q0), _mm_srli_epi32::<1>(abs_p1q1));
    let m_val = not_gt(val_ee, e_v);
    let m_p2p1 = not_gt(abs_p2p1, i_v);
    let m_q2q1 = not_gt(abs_q2q1, i_v);
    let m_p3p2 = not_gt(abs_p3p2, i_v);
    let m_q3q2 = not_gt(abs_q3q2, i_v);
    let fm_mask = _mm_and_si128(
        _mm_and_si128(_mm_and_si128(m_p1p0, m_q1q0), m_val),
        _mm_and_si128(_mm_and_si128(m_p2p1, m_q2q1), _mm_and_si128(m_p3p2, m_q3q2)),
    );

    let abs_p6p0 = abs(p6_v, p0_v);
    let abs_p5p0 = abs(p5_v, p0_v);
    let abs_p4p0 = abs(p4_v, p0_v);
    let abs_q4q0 = abs(q4_v, q0_v);
    let abs_q5q0 = abs(q5_v, q0_v);
    let abs_q6q0 = abs(q6_v, q0_v);
    let flat8out_mask = _mm_and_si128(
        _mm_and_si128(
            _mm_and_si128(not_gt(abs_p6p0, f_v), not_gt(abs_p5p0, f_v)),
            not_gt(abs_p4p0, f_v),
        ),
        _mm_and_si128(
            _mm_and_si128(not_gt(abs_q4q0, f_v), not_gt(abs_q5q0, f_v)),
            not_gt(abs_q6q0, f_v),
        ),
    );

    let abs_p2p0 = abs(p2_v, p0_v);
    let abs_q2q0 = abs(q2_v, q0_v);
    let abs_p3p0 = abs(p3_v, p0_v);
    let abs_q3q0 = abs(q3_v, q0_v);
    let flat8in_mask = _mm_and_si128(
        _mm_and_si128(not_gt(abs_p2p0, f_v), not_gt(abs_p1p0, f_v)),
        _mm_and_si128(
            _mm_and_si128(not_gt(abs_q1q0, f_v), not_gt(abs_q2q0, f_v)),
            _mm_and_si128(not_gt(abs_p3p0, f_v), not_gt(abs_q3q0, f_v)),
        ),
    );

    let dbl = |v: __m128i| _mm_slli_epi32::<1>(v);
    let add = |a: __m128i, b: __m128i| _mm_add_epi32(a, b);
    let add3 = |a: __m128i, b: __m128i, c: __m128i| add(add(a, b), c);
    let add4 = |a: __m128i, b: __m128i, c: __m128i, d: __m128i| add(add(a, b), add(c, d));
    let c4 = _mm_set1_epi32(4);
    let c8 = _mm_set1_epi32(8);

    // 14-tap wide filter outputs (positions -6..5)
    let p6_5 = _mm_add_epi32(
        _mm_add_epi32(_mm_add_epi32(p6_v, p6_v), _mm_add_epi32(p6_v, p6_v)),
        p6_v,
    );
    let q6_5 = _mm_add_epi32(
        _mm_add_epi32(_mm_add_epi32(q6_v, q6_v), _mm_add_epi32(q6_v, q6_v)),
        q6_v,
    );

    let mut s = add(p6_5, _mm_add_epi32(dbl(p6_v), dbl(p5_v)));
    s = add(s, dbl(p4_v));
    s = add(s, add4(p3_v, p2_v, p1_v, p0_v));
    s = add(s, add(q0_v, c8));
    let out_m6 = _mm_srai_epi32::<4>(s);

    let mut s = add(p6_5, _mm_add_epi32(dbl(p5_v), dbl(p4_v)));
    s = add(s, dbl(p3_v));
    s = add(s, add4(p2_v, p1_v, p0_v, q0_v));
    s = add(s, add(q1_v, c8));
    let out_m5 = _mm_srai_epi32::<4>(s);

    let p6_4 = _mm_add_epi32(dbl(p6_v), dbl(p6_v));
    let mut s = add(p6_4, p5_v);
    s = add(s, _mm_add_epi32(dbl(p4_v), dbl(p3_v)));
    s = add(s, dbl(p2_v));
    s = add(s, add4(p1_v, p0_v, q0_v, q1_v));
    s = add(s, add(q2_v, c8));
    let out_m4 = _mm_srai_epi32::<4>(s);

    let p6_3 = add(dbl(p6_v), p6_v);
    let mut s = add(p6_3, _mm_add_epi32(p5_v, p4_v));
    s = add(s, _mm_add_epi32(dbl(p3_v), dbl(p2_v)));
    s = add(s, dbl(p1_v));
    s = add(s, add4(p0_v, q0_v, q1_v, q2_v));
    s = add(s, add(q3_v, c8));
    let out_m3 = _mm_srai_epi32::<4>(s);

    let mut s = add(dbl(p6_v), p5_v);
    s = add(s, _mm_add_epi32(p4_v, p3_v));
    s = add(s, _mm_add_epi32(dbl(p2_v), dbl(p1_v)));
    s = add(s, dbl(p0_v));
    s = add(s, add4(q0_v, q1_v, q2_v, q3_v));
    s = add(s, add(q4_v, c8));
    let out_m2 = _mm_srai_epi32::<4>(s);

    let mut s = add(p6_v, p5_v);
    s = add(s, _mm_add_epi32(p4_v, p3_v));
    s = add(s, p2_v);
    s = add(s, _mm_add_epi32(dbl(p1_v), dbl(p0_v)));
    s = add(s, dbl(q0_v));
    s = add(s, add4(q1_v, q2_v, q3_v, q4_v));
    s = add(s, add(q5_v, c8));
    let out_m1 = _mm_srai_epi32::<4>(s);

    let mut s = add(p5_v, p4_v);
    s = add(s, _mm_add_epi32(p3_v, p2_v));
    s = add(s, p1_v);
    s = add(s, _mm_add_epi32(dbl(p0_v), dbl(q0_v)));
    s = add(s, dbl(q1_v));
    s = add(s, add4(q2_v, q3_v, q4_v, q5_v));
    s = add(s, add(q6_v, c8));
    let out_0 = _mm_srai_epi32::<4>(s);

    let mut s = add(p4_v, p3_v);
    s = add(s, _mm_add_epi32(p2_v, p1_v));
    s = add(s, p0_v);
    s = add(s, _mm_add_epi32(dbl(q0_v), dbl(q1_v)));
    s = add(s, dbl(q2_v));
    s = add(s, add4(q3_v, q4_v, q5_v, q6_v));
    s = add(s, add(q6_v, c8));
    let out_1 = _mm_srai_epi32::<4>(s);

    let mut s = add(p3_v, p2_v);
    s = add(s, _mm_add_epi32(p1_v, p0_v));
    s = add(s, q0_v);
    s = add(s, _mm_add_epi32(dbl(q1_v), dbl(q2_v)));
    s = add(s, dbl(q3_v));
    let q6_3 = add(dbl(q6_v), q6_v);
    s = add(s, add3(q4_v, q5_v, q6_3));
    s = add(s, c8);
    let out_2 = _mm_srai_epi32::<4>(s);

    let q6_4 = _mm_add_epi32(dbl(q6_v), dbl(q6_v));
    let mut s = add(p2_v, p1_v);
    s = add(s, _mm_add_epi32(p0_v, q0_v));
    s = add(s, q1_v);
    s = add(s, _mm_add_epi32(dbl(q2_v), dbl(q3_v)));
    s = add(s, dbl(q4_v));
    s = add(s, add(q5_v, q6_4));
    s = add(s, c8);
    let out_3 = _mm_srai_epi32::<4>(s);

    let mut s = add(p1_v, p0_v);
    s = add(s, _mm_add_epi32(q0_v, q1_v));
    s = add(s, q2_v);
    s = add(s, _mm_add_epi32(dbl(q3_v), dbl(q4_v)));
    s = add(s, dbl(q5_v));
    s = add(s, q6_5);
    s = add(s, c8);
    let out_4 = _mm_srai_epi32::<4>(s);

    let q6_7 = _mm_add_epi32(q6_5, _mm_add_epi32(q6_v, q6_v));
    let mut s = add(p0_v, q0_v);
    s = add(s, _mm_add_epi32(q1_v, q2_v));
    s = add(s, q3_v);
    s = add(s, _mm_add_epi32(dbl(q4_v), dbl(q5_v)));
    s = add(s, q6_7);
    s = add(s, c8);
    let out_5 = _mm_srai_epi32::<4>(s);

    // 8-tap outputs (when !flat8out && flat8in)
    let triple = |v: __m128i| _mm_add_epi32(dbl(v), v);
    let out8_m3 = _mm_srai_epi32::<3>(add(
        add4(triple(p3_v), dbl(p2_v), p1_v, p0_v),
        add(q0_v, c4),
    ));
    let out8_m2 = _mm_srai_epi32::<3>(add(
        add4(dbl(p3_v), p2_v, dbl(p1_v), p0_v),
        add3(q0_v, q1_v, c4),
    ));
    let out8_m1 = _mm_srai_epi32::<3>(add(
        add4(p3_v, p2_v, p1_v, dbl(p0_v)),
        add4(q0_v, q1_v, q2_v, c4),
    ));
    let out8_0 = _mm_srai_epi32::<3>(add(
        add4(p2_v, p1_v, p0_v, dbl(q0_v)),
        add4(q1_v, q2_v, q3_v, c4),
    ));
    let out8_1 = _mm_srai_epi32::<3>(add(
        add4(p1_v, p0_v, q0_v, dbl(q1_v)),
        add4(q2_v, q3_v, q3_v, c4),
    ));
    let out8_2 = _mm_srai_epi32::<3>(add(
        add4(p0_v, q0_v, q1_v, dbl(q2_v)),
        add4(q3_v, q3_v, q3_v, c4),
    ));

    // Narrow filter
    let neg128 = _mm_set1_epi32(-128);
    let pos127 = _mm_set1_epi32(127);
    let iclip = |v: __m128i| _mm_min_epi32(_mm_max_epi32(v, neg128), pos127);
    let diff_q0p0 = _mm_sub_epi32(q0_v, p0_v);
    let three_d = _mm_add_epi32(_mm_slli_epi32::<1>(diff_q0p0), diff_q0p0);
    let diff_p1q1 = _mm_sub_epi32(p1_v, q1_v);
    let hev_mask = _mm_or_si128(
        _mm_cmpgt_epi32(abs_p1p0, h_v),
        _mm_cmpgt_epi32(abs_q1q0, h_v),
    );
    let f_hev = iclip(_mm_add_epi32(three_d, iclip(diff_p1q1)));
    let f_no = iclip(three_d);
    let c4i = c4;
    let c3i = _mm_set1_epi32(3);
    let one = _mm_set1_epi32(1);
    let f1_hev = _mm_srai_epi32::<3>(_mm_min_epi32(_mm_add_epi32(f_hev, c4i), pos127));
    let f2_hev = _mm_srai_epi32::<3>(_mm_min_epi32(_mm_add_epi32(f_hev, c3i), pos127));
    let f1_no = _mm_srai_epi32::<3>(_mm_min_epi32(_mm_add_epi32(f_no, c4i), pos127));
    let f2_no = _mm_srai_epi32::<3>(_mm_min_epi32(_mm_add_epi32(f_no, c3i), pos127));
    let f_extra = _mm_srai_epi32::<1>(_mm_add_epi32(f1_no, one));
    let p0_hev = _mm_add_epi32(p0_v, f2_hev);
    let q0_hev = _mm_sub_epi32(q0_v, f1_hev);
    let p0_no = _mm_add_epi32(p0_v, f2_no);
    let q0_no = _mm_sub_epi32(q0_v, f1_no);
    let p1_no = _mm_add_epi32(p1_v, f_extra);
    let q1_no = _mm_sub_epi32(q1_v, f_extra);

    let blendv = |a: __m128i, b: __m128i, mask: __m128i| -> __m128i {
        _mm_or_si128(_mm_andnot_si128(mask, a), _mm_and_si128(mask, b))
    };

    let narrow_p1 = blendv(p1_no, p1_v, hev_mask);
    let narrow_p0 = blendv(p0_no, p0_hev, hev_mask);
    let narrow_q0 = blendv(q0_no, q0_hev, hev_mask);
    let narrow_q1 = blendv(q1_no, q1_v, hev_mask);

    let wide_mask = _mm_and_si128(flat8out_mask, flat8in_mask);

    let mid_m3 = blendv(p2_v, out8_m3, flat8in_mask);
    let mid_m2 = blendv(narrow_p1, out8_m2, flat8in_mask);
    let mid_m1 = blendv(narrow_p0, out8_m1, flat8in_mask);
    let mid_0 = blendv(narrow_q0, out8_0, flat8in_mask);
    let mid_1 = blendv(narrow_q1, out8_1, flat8in_mask);
    let mid_2 = blendv(q2_v, out8_2, flat8in_mask);

    let sel_m6 = blendv(p5_v, out_m6, wide_mask);
    let sel_m5 = blendv(p4_v, out_m5, wide_mask);
    let sel_m4 = blendv(p3_v, out_m4, wide_mask);
    let sel_m3 = blendv(mid_m3, out_m3, wide_mask);
    let sel_m2 = blendv(mid_m2, out_m2, wide_mask);
    let sel_m1 = blendv(mid_m1, out_m1, wide_mask);
    let sel_0 = blendv(mid_0, out_0, wide_mask);
    let sel_1 = blendv(mid_1, out_1, wide_mask);
    let sel_2 = blendv(mid_2, out_2, wide_mask);
    let sel_3 = blendv(q3_v, out_3, wide_mask);
    let sel_4 = blendv(q4_v, out_4, wide_mask);
    let sel_5 = blendv(q5_v, out_5, wide_mask);

    let final_m6 = blendv(p5_v, sel_m6, fm_mask);
    let final_m5 = blendv(p4_v, sel_m5, fm_mask);
    let final_m4 = blendv(p3_v, sel_m4, fm_mask);
    let final_m3 = blendv(p2_v, sel_m3, fm_mask);
    let final_m2 = blendv(p1_v, sel_m2, fm_mask);
    let final_m1 = blendv(p0_v, sel_m1, fm_mask);
    let final_0 = blendv(q0_v, sel_0, fm_mask);
    let final_1 = blendv(q1_v, sel_1, fm_mask);
    let final_2 = blendv(q2_v, sel_2, fm_mask);
    let final_3 = blendv(q3_v, sel_3, fm_mask);
    let final_4 = blendv(q4_v, sel_4, fm_mask);
    let final_5 = blendv(q5_v, sel_5, fm_mask);

    // Clip to [0, 255]
    let zero = _mm_setzero_si128();
    let max_u8 = _mm_set1_epi32(255);
    let clip_u8 = |v: __m128i| _mm_min_epi32(_mm_max_epi32(v, zero), max_u8);
    let final_m6 = clip_u8(final_m6);
    let final_m5 = clip_u8(final_m5);
    let final_m4 = clip_u8(final_m4);
    let final_m3 = clip_u8(final_m3);
    let final_m2 = clip_u8(final_m2);
    let final_m1 = clip_u8(final_m1);
    let final_0 = clip_u8(final_0);
    let final_1 = clip_u8(final_1);
    let final_2 = clip_u8(final_2);
    let final_3 = clip_u8(final_3);
    let final_4 = clip_u8(final_4);
    let final_5 = clip_u8(final_5);

    // Transpose back: chunk 0 = [p6 (orig), p5, p4, p3 -> positions -7, -6, -5, -4]
    // We didn't update p6, so put back originals at lane 0 of chunk 0.
    let back_c0 = {
        let t0 = _mm_unpacklo_epi32(p6_v, final_m6);
        let t1 = _mm_unpackhi_epi32(p6_v, final_m6);
        let t2 = _mm_unpacklo_epi32(final_m5, final_m4);
        let t3 = _mm_unpackhi_epi32(final_m5, final_m4);
        [
            _mm_unpacklo_epi64(t0, t2),
            _mm_unpackhi_epi64(t0, t2),
            _mm_unpacklo_epi64(t1, t3),
            _mm_unpackhi_epi64(t1, t3),
        ]
    };
    let back_c1 = {
        let t0 = _mm_unpacklo_epi32(final_m3, final_m2);
        let t1 = _mm_unpackhi_epi32(final_m3, final_m2);
        let t2 = _mm_unpacklo_epi32(final_m1, final_0);
        let t3 = _mm_unpackhi_epi32(final_m1, final_0);
        [
            _mm_unpacklo_epi64(t0, t2),
            _mm_unpackhi_epi64(t0, t2),
            _mm_unpacklo_epi64(t1, t3),
            _mm_unpackhi_epi64(t1, t3),
        ]
    };
    let back_c2 = {
        let t0 = _mm_unpacklo_epi32(final_1, final_2);
        let t1 = _mm_unpackhi_epi32(final_1, final_2);
        let t2 = _mm_unpacklo_epi32(final_3, final_4);
        let t3 = _mm_unpackhi_epi32(final_3, final_4);
        [
            _mm_unpacklo_epi64(t0, t2),
            _mm_unpackhi_epi64(t0, t2),
            _mm_unpacklo_epi64(t1, t3),
            _mm_unpackhi_epi64(t1, t3),
        ]
    };

    let pack_row = |v: __m128i| -> i32 {
        let u16x4 = _mm_packus_epi32(v, v);
        let u8x4 = _mm_packus_epi16(u16x4, u16x4);
        _mm_cvtsi128_si32(u8x4)
    };
    let store_4bytes = |buf: &mut [u8], packed: i32, row: isize, chunk_off: isize| {
        let start = signed_idx(base, row * stridea + chunk_off);
        let bytes = packed.to_le_bytes();
        buf[start] = bytes[0];
        buf[start + 1] = bytes[1];
        buf[start + 2] = bytes[2];
        buf[start + 3] = bytes[3];
    };
    // Store chunks 0, 1, 2 (we keep chunk 3 = q5/q6 - need to update q5 only)
    // chunk 0: -7..-4 (positions -7..-4: p6, p5, p4, p3 → final_m6 wrong: p6 untouched! Use originals at offset -7)
    // Actually our back_c0 above puts p6_v at lane 0 (= position -7). For row k, the byte at offset -7 should remain p6 unchanged.
    // Hmm wait, the wd=16 filter doesn't write position -7 (p6) at all. So we should put p6_v (original) there.
    // Let me re-check.
    //
    // 14-tap filter writes positions -6..5 (12 outputs). Positions -7 (p6) and 6 (q6) are unchanged.
    // For correctness: at row layout, we need to store all 16 bytes but p6 and q6 stay as original.
    //   chunk 0 covers offsets -7..-4 → [p6, p5, p4, p3]. p6 must be unchanged.
    //   chunk 1 covers offsets -3..0  → [p2, p1, p0, q0]. All updated.
    //   chunk 2 covers offsets 1..4  → [q1, q2, q3, q4]. All updated.
    //   chunk 3 covers offsets 5..8  → [q5, q6, ?, ?]. q5 updated, q6 unchanged, ? unused.
    //
    // For chunk 3, we'd write back q5 (updated) and q6 (original). Plus 2 unused bytes that aren't part of the loopfilter pixels but ARE part of the buffer. Reading + writing them must preserve their values.
    //
    // To avoid corrupting the 2 unused bytes, we have to load them and write them back. Let me just use scalar stores for q5 and skip q6+unused.

    // Store chunk 0 (offsets -7..-4): each row gets [p6 (unchanged), final_m6, final_m5, final_m4]
    store_4bytes(buf, pack_row(back_c0[0]), 0, -7);
    store_4bytes(buf, pack_row(back_c0[1]), 1, -7);
    store_4bytes(buf, pack_row(back_c0[2]), 2, -7);
    store_4bytes(buf, pack_row(back_c0[3]), 3, -7);
    // Store chunk 1 (offsets -3..0): full update
    store_4bytes(buf, pack_row(back_c1[0]), 0, -3);
    store_4bytes(buf, pack_row(back_c1[1]), 1, -3);
    store_4bytes(buf, pack_row(back_c1[2]), 2, -3);
    store_4bytes(buf, pack_row(back_c1[3]), 3, -3);
    // Store chunk 2 (offsets 1..4): full update
    store_4bytes(buf, pack_row(back_c2[0]), 0, 1);
    store_4bytes(buf, pack_row(back_c2[1]), 1, 1);
    store_4bytes(buf, pack_row(back_c2[2]), 2, 1);
    store_4bytes(buf, pack_row(back_c2[3]), 3, 1);
    // Store q5 only at offset 5 per row (scalar — extract lane k from final_5)
    let mut q5_arr = [0i32; 4];
    safe_unaligned_simd::x86_64::_mm_storeu_si128(&mut q5_arr, final_5);
    for k in 0..4 {
        let start = signed_idx(base, k as isize * stridea + 5);
        buf[start] = q5_arr[k] as u8;
    }
}

// ============================================================================
// SIMD inner loop filter for the narrow 4-tap V-FILTER case (wd=4, strideb>1)
// ============================================================================

/// SIMD narrow 4-tap loop filter for 8bpc V-FILTER direction.
/// In v-filter, 4 filter positions are 4 ADJACENT columns (stridea=1) and
/// the filter pixels are at row offsets (strideb=stride). This means each
/// of p1/p0/q0/q1 is a contiguous 4-byte slice that we can load with a
/// single i32 load + widen — much faster than the h-filter gather pattern.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn loop_filter_4_8bpc_narrow_simd_v(
    _token: Desktop64,
    buf: &mut [u8],
    base: usize,
    e: i32,
    i: i32,
    h: i32,
    strideb: isize,
) {
    // base + k for k in 0..4 are the 4 filter positions
    // Pixels at row offsets -2, -1, 0, 1 from each filter position
    let load4 = |off: isize| -> __m128i {
        let start = signed_idx(base, strideb * off);
        let bytes = [buf[start], buf[start + 1], buf[start + 2], buf[start + 3]];
        let as_i32 = i32::from_le_bytes(bytes);
        let v4u8 = _mm_cvtsi32_si128(as_i32);
        _mm_cvtepu8_epi32(v4u8)
    };

    let p1_v = load4(-2);
    let p0_v = load4(-1);
    let q0_v = load4(0);
    let q1_v = load4(1);

    let i_v = _mm_set1_epi32(i);
    let e_v = _mm_set1_epi32(e);
    let h_v = _mm_set1_epi32(h);

    let abs_p1p0 = _mm_abs_epi32(_mm_sub_epi32(p1_v, p0_v));
    let abs_q1q0 = _mm_abs_epi32(_mm_sub_epi32(q1_v, q0_v));
    let abs_p0q0 = _mm_abs_epi32(_mm_sub_epi32(p0_v, q0_v));
    let abs_p1q1 = _mm_abs_epi32(_mm_sub_epi32(p1_v, q1_v));

    let not_gt = |a: __m128i, b: __m128i| -> __m128i {
        _mm_andnot_si128(_mm_cmpgt_epi32(a, b), _mm_set1_epi32(-1))
    };
    let m_p1p0 = not_gt(abs_p1p0, i_v);
    let m_q1q0 = not_gt(abs_q1q0, i_v);
    let val = _mm_add_epi32(_mm_slli_epi32::<1>(abs_p0q0), _mm_srli_epi32::<1>(abs_p1q1));
    let m_val = not_gt(val, e_v);
    let fm_mask = _mm_and_si128(_mm_and_si128(m_p1p0, m_q1q0), m_val);

    let hev_mask = _mm_or_si128(
        _mm_cmpgt_epi32(abs_p1p0, h_v),
        _mm_cmpgt_epi32(abs_q1q0, h_v),
    );

    let neg128 = _mm_set1_epi32(-128);
    let pos127 = _mm_set1_epi32(127);
    let iclip = |v: __m128i| _mm_min_epi32(_mm_max_epi32(v, neg128), pos127);

    let diff_q0p0 = _mm_sub_epi32(q0_v, p0_v);
    let three_d = _mm_add_epi32(_mm_slli_epi32::<1>(diff_q0p0), diff_q0p0);
    let diff_p1q1 = _mm_sub_epi32(p1_v, q1_v);

    let f_hev = iclip(_mm_add_epi32(three_d, iclip(diff_p1q1)));
    let f_nohev = iclip(three_d);

    let c4 = _mm_set1_epi32(4);
    let c3 = _mm_set1_epi32(3);
    let one = _mm_set1_epi32(1);

    let f1_hev = _mm_srai_epi32::<3>(_mm_min_epi32(_mm_add_epi32(f_hev, c4), pos127));
    let f2_hev = _mm_srai_epi32::<3>(_mm_min_epi32(_mm_add_epi32(f_hev, c3), pos127));
    let f1_no = _mm_srai_epi32::<3>(_mm_min_epi32(_mm_add_epi32(f_nohev, c4), pos127));
    let f2_no = _mm_srai_epi32::<3>(_mm_min_epi32(_mm_add_epi32(f_nohev, c3), pos127));
    let f_extra = _mm_srai_epi32::<1>(_mm_add_epi32(f1_no, one));

    let p0_hev = _mm_add_epi32(p0_v, f2_hev);
    let q0_hev = _mm_sub_epi32(q0_v, f1_hev);
    let p0_no = _mm_add_epi32(p0_v, f2_no);
    let q0_no = _mm_sub_epi32(q0_v, f1_no);
    let p1_no = _mm_add_epi32(p1_v, f_extra);
    let q1_no = _mm_sub_epi32(q1_v, f_extra);

    let blendv = |a: __m128i, b: __m128i, mask: __m128i| -> __m128i {
        _mm_or_si128(_mm_andnot_si128(mask, a), _mm_and_si128(mask, b))
    };
    let p1_filt = blendv(p1_no, p1_v, hev_mask);
    let p0_filt = blendv(p0_no, p0_hev, hev_mask);
    let q0_filt = blendv(q0_no, q0_hev, hev_mask);
    let q1_filt = blendv(q1_no, q1_v, hev_mask);

    let p1_final = blendv(p1_v, p1_filt, fm_mask);
    let p0_final = blendv(p0_v, p0_filt, fm_mask);
    let q0_final = blendv(q0_v, q0_filt, fm_mask);
    let q1_final = blendv(q1_v, q1_filt, fm_mask);

    // Pack 4 i32 lanes back to 4 u8 (each clipped to [0,255]).
    // Use _mm_packus_epi32 then _mm_packus_epi16.
    let pack4 = |v: __m128i| -> i32 {
        let u16x4 = _mm_packus_epi32(v, v); // low 4 u16
        let u8x4 = _mm_packus_epi16(u16x4, u16x4); // low 4 u8
        _mm_cvtsi128_si32(u8x4)
    };
    let store4 = |buf: &mut [u8], packed: i32, off: isize| {
        let start = signed_idx(base, strideb * off);
        let bytes = packed.to_le_bytes();
        buf[start] = bytes[0];
        buf[start + 1] = bytes[1];
        buf[start + 2] = bytes[2];
        buf[start + 3] = bytes[3];
    };
    store4(buf, pack4(p1_final), -2);
    store4(buf, pack4(p0_final), -1);
    store4(buf, pack4(q0_final), 0);
    store4(buf, pack4(q1_final), 1);
}

// ============================================================================
// SIMD inner narrow loop filter — V-FILTER, 8 columns at a time (YMM)
// ============================================================================

/// SIMD narrow 4-tap loop filter for 8bpc V-FILTER direction.
/// Wider variant: processes **8** adjacent filter positions (8 columns) per
/// call by widening from `__m128i` (4 i32 lanes) to `__m256i` (8 i32 lanes).
/// Caller must pre-verify the next 4 columns also use narrow (wd=4) and the
/// same `l`/`e`/`i`/`h` parameters as the current group; the outer dispatcher
/// merges two `vmask[0]`-only adjacent edges into one of these calls.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn loop_filter_4_8bpc_narrow_simd_v_x8(
    _token: Desktop64,
    buf: &mut [u8],
    base: usize,
    e: i32,
    i: i32,
    h: i32,
    strideb: isize,
) {
    // Load 8 contiguous u8 columns at a given row offset, widen to 8 i32.
    let load8 = |off: isize| -> __m256i {
        let start = signed_idx(base, strideb * off);
        let lo = i64::from_ne_bytes([
            buf[start],
            buf[start + 1],
            buf[start + 2],
            buf[start + 3],
            buf[start + 4],
            buf[start + 5],
            buf[start + 6],
            buf[start + 7],
        ]);
        let v8u8 = _mm_set_epi64x(0, lo);
        _mm256_cvtepu8_epi32(v8u8)
    };

    let p1_v = load8(-2);
    let p0_v = load8(-1);
    let q0_v = load8(0);
    let q1_v = load8(1);

    let i_v = _mm256_set1_epi32(i);
    let e_v = _mm256_set1_epi32(e);
    let h_v = _mm256_set1_epi32(h);

    let abs_p1p0 = _mm256_abs_epi32(_mm256_sub_epi32(p1_v, p0_v));
    let abs_q1q0 = _mm256_abs_epi32(_mm256_sub_epi32(q1_v, q0_v));
    let abs_p0q0 = _mm256_abs_epi32(_mm256_sub_epi32(p0_v, q0_v));
    let abs_p1q1 = _mm256_abs_epi32(_mm256_sub_epi32(p1_v, q1_v));

    let not_gt = |a: __m256i, b: __m256i| -> __m256i {
        _mm256_andnot_si256(_mm256_cmpgt_epi32(a, b), _mm256_set1_epi32(-1))
    };
    let m_p1p0 = not_gt(abs_p1p0, i_v);
    let m_q1q0 = not_gt(abs_q1q0, i_v);
    let val = _mm256_add_epi32(
        _mm256_slli_epi32::<1>(abs_p0q0),
        _mm256_srli_epi32::<1>(abs_p1q1),
    );
    let m_val = not_gt(val, e_v);
    let fm_mask = _mm256_and_si256(_mm256_and_si256(m_p1p0, m_q1q0), m_val);

    let hev_mask = _mm256_or_si256(
        _mm256_cmpgt_epi32(abs_p1p0, h_v),
        _mm256_cmpgt_epi32(abs_q1q0, h_v),
    );

    let neg128 = _mm256_set1_epi32(-128);
    let pos127 = _mm256_set1_epi32(127);
    let iclip = |v: __m256i| _mm256_min_epi32(_mm256_max_epi32(v, neg128), pos127);

    let diff_q0p0 = _mm256_sub_epi32(q0_v, p0_v);
    let three_d = _mm256_add_epi32(_mm256_slli_epi32::<1>(diff_q0p0), diff_q0p0);
    let diff_p1q1 = _mm256_sub_epi32(p1_v, q1_v);

    let f_hev = iclip(_mm256_add_epi32(three_d, iclip(diff_p1q1)));
    let f_nohev = iclip(three_d);

    let c4 = _mm256_set1_epi32(4);
    let c3 = _mm256_set1_epi32(3);
    let one = _mm256_set1_epi32(1);

    let f1_hev = _mm256_srai_epi32::<3>(_mm256_min_epi32(_mm256_add_epi32(f_hev, c4), pos127));
    let f2_hev = _mm256_srai_epi32::<3>(_mm256_min_epi32(_mm256_add_epi32(f_hev, c3), pos127));
    let f1_no = _mm256_srai_epi32::<3>(_mm256_min_epi32(_mm256_add_epi32(f_nohev, c4), pos127));
    let f2_no = _mm256_srai_epi32::<3>(_mm256_min_epi32(_mm256_add_epi32(f_nohev, c3), pos127));
    let f_extra = _mm256_srai_epi32::<1>(_mm256_add_epi32(f1_no, one));

    let p0_hev = _mm256_add_epi32(p0_v, f2_hev);
    let q0_hev = _mm256_sub_epi32(q0_v, f1_hev);
    let p0_no = _mm256_add_epi32(p0_v, f2_no);
    let q0_no = _mm256_sub_epi32(q0_v, f1_no);
    let p1_no = _mm256_add_epi32(p1_v, f_extra);
    let q1_no = _mm256_sub_epi32(q1_v, f_extra);

    let blendv = |a: __m256i, b: __m256i, mask: __m256i| -> __m256i {
        _mm256_or_si256(_mm256_andnot_si256(mask, a), _mm256_and_si256(mask, b))
    };
    let p1_filt = blendv(p1_no, p1_v, hev_mask);
    let p0_filt = blendv(p0_no, p0_hev, hev_mask);
    let q0_filt = blendv(q0_no, q0_hev, hev_mask);
    let q1_filt = blendv(q1_no, q1_v, hev_mask);

    let p1_final = blendv(p1_v, p1_filt, fm_mask);
    let p0_final = blendv(p0_v, p0_filt, fm_mask);
    let q0_final = blendv(q0_v, q0_filt, fm_mask);
    let q1_final = blendv(q1_v, q1_filt, fm_mask);

    // Pack 8 i32 lanes (clamped to [0,255]) back to 8 u8.
    //
    // After _mm256_packus_epi32(v,v): [u16x8 in low lane | u16x8 in high lane],
    //   where each lane's u16x8 = (lo_i32x4_packed, lo_i32x4_packed). I.e. the
    //   low qword (8 bytes -> 4 u16) of each lane has the 4 lane-local i32s
    //   packed to u16.
    // After _mm256_packus_epi16(u16x, u16x): each 128-bit lane has 16 u8 where
    //   the low 4 bytes are the lane-local i32x4 packed to u8 (the rest is a
    //   duplicate we ignore).
    // permute4x64::<0b1000> rearranges qwords: result[qword 0] = src[qword 0]
    //   (lane0 low 8 bytes), result[qword 1] = src[qword 2] (lane1 low 8 bytes).
    //   We only care about bytes [0..4] of qword 0 (a0..a3) and bytes [0..4]
    //   of qword 1 (a4..a7); together they form 8 contiguous bytes in [0..8].
    // But qword 0's bytes [4..8] are a duplicate of a0..a3 — they get
    //   overwritten by qword 1's a4..a7 only if we shift. Easier: use a
    //   shuffle_epi8 to gather the right bytes.
    //
    // Use permutevar8x32_epi32 on the packus_epi16 result. We need lanes
    //   [0, 4, 0, 4, ?, ?, ?, ?] — actually no: we operate on the u8 result
    //   where lane0 dwords are [u32 dup of a0..a3]x4 and lane1 dwords are
    //   [u32 dup of a4..a7]x4. We want low 64 bits = a0..a3,a4..a7. That's
    //   dword indices [0, 4] picked into [0, 1].
    let pack8 = |v: __m256i| -> i64 {
        let u16x = _mm256_packus_epi32(v, v);
        let u8x = _mm256_packus_epi16(u16x, u16x);
        // Permute control: gather i32 lane 0 of low half and i32 lane 0 of
        // high half. _mm256_permutevar8x32_epi32 indexes 8 i32 lanes globally
        // (lanes 0..3 = low half, lanes 4..7 = high half).
        let idx = _mm256_setr_epi32(0, 4, 0, 0, 0, 0, 0, 0);
        let p = _mm256_permutevar8x32_epi32(u8x, idx);
        let lo128 = _mm256_castsi256_si128(p);
        _mm_cvtsi128_si64(lo128)
    };
    let store8 = |buf: &mut [u8], packed: i64, off: isize| {
        let start = signed_idx(base, strideb * off);
        let bytes = packed.to_ne_bytes();
        buf[start..start + 8].copy_from_slice(&bytes);
    };
    store8(buf, pack8(p1_final), -2);
    store8(buf, pack8(p0_final), -1);
    store8(buf, pack8(q0_final), 0);
    store8(buf, pack8(q1_final), 1);
}

// ============================================================================
// SUPERBLOCK FILTER FUNCTIONS (8bpc)
// ============================================================================

/// Read level value from lvl slice at the given offset.
/// Each logical entry is 4 consecutive `AtomicU8`; `byte_idx` selects which byte:
///   0 = H Y, 1 = V Y, 2 = H/V U, 3 = H/V V
/// Returns 0 for out-of-bounds access (= no filtering for that block).
#[inline(always)]
fn read_lvl(lvl: &[AtomicU8], offset: usize, byte_idx: usize) -> u8 {
    let idx = offset * 4 + byte_idx;
    lvl.get(idx).map_or(0, |v| v.load(Relaxed))
}

/// Loop filter for Y plane, horizontal edges (8bpc)
#[cfg(any(target_arch = "x86_64", target_arch = "wasm32"))]
#[cfg_attr(target_arch = "x86_64", arcane)]
#[allow(unused_mut)]
fn lpf_h_sb_y_8bpc_inner(
    #[cfg(target_arch = "x86_64")] _token: Desktop64,
    buf: &mut [u8],
    mut dst_offset: usize,
    stride: isize,
    vmask: &[u32; 3],
    lvl: &[AtomicU8],
    lvl_base: usize,
    lvl_byte_idx: usize,
    b4_stride: isize,
    lut: &Align16<Av1FilterLUT>,
    _w: i32,
    bitdepth_max: i32,
) {
    let stridea = stride;
    let strideb = 1isize;
    let b4_stridea = b4_stride as usize;
    let b4_strideb = 1usize;

    let vm = vmask[0] | vmask[1] | vmask[2];
    let mut lvl_offset = lvl_base;

    let mut xy = 1u32;
    while vm & !xy.wrapping_sub(1) != 0 {
        if vm & xy != 0 {
            let lvl_val = read_lvl(lvl, lvl_offset, lvl_byte_idx);
            let l = if lvl_val != 0 {
                lvl_val
            } else {
                if lvl_offset >= b4_strideb {
                    read_lvl(lvl, lvl_offset - b4_strideb, lvl_byte_idx)
                } else {
                    0
                }
            };

            if l != 0 {
                let h = (l >> 4) as i32;
                let e = lut.e[l as usize] as i32;
                let i = lut.i[l as usize] as i32;

                let idx = if vmask[2] & xy != 0 {
                    16
                } else if vmask[1] & xy != 0 {
                    8
                } else {
                    4
                };

                loop_filter_4_8bpc(
                    #[cfg(target_arch = "x86_64")]
                    _token,
                    buf,
                    dst_offset,
                    e,
                    i,
                    h,
                    stridea,
                    strideb,
                    idx,
                    bitdepth_max,
                );
            }
        }

        xy <<= 1;
        dst_offset = signed_idx(dst_offset, 4 * stridea);
        lvl_offset += b4_stridea;
    }
}

/// Loop filter for Y plane, vertical edges (8bpc)
#[cfg(any(target_arch = "x86_64", target_arch = "wasm32"))]
#[cfg_attr(target_arch = "x86_64", arcane)]
#[allow(unused_mut)]
fn lpf_v_sb_y_8bpc_inner(
    #[cfg(target_arch = "x86_64")] _token: Desktop64,
    buf: &mut [u8],
    mut dst_offset: usize,
    stride: isize,
    vmask: &[u32; 3],
    lvl: &[AtomicU8],
    lvl_base: usize,
    lvl_byte_idx: usize,
    b4_stride: isize,
    lut: &Align16<Av1FilterLUT>,
    _w: i32,
    bitdepth_max: i32,
) {
    let stridea = 1isize;
    let strideb = stride;
    let b4_stridea = 1usize;
    let b4_strideb = b4_stride as usize;

    let vm = vmask[0] | vmask[1] | vmask[2];
    let mut lvl_offset = lvl_base;

    // Helper: compute (l, h, e, i) for the edge at `lvl_offset` (or 0 if no
    // filter). Mirrors the original per-iteration logic.
    let derive_levels = |lvl_offset: usize| -> Option<(u8, i32, i32, i32)> {
        let lvl_val = read_lvl(lvl, lvl_offset, lvl_byte_idx);
        let l = if lvl_val != 0 {
            lvl_val
        } else if lvl_offset >= b4_strideb {
            read_lvl(lvl, lvl_offset - b4_strideb, lvl_byte_idx)
        } else {
            0
        };
        if l == 0 {
            None
        } else {
            let h = (l >> 4) as i32;
            let e = lut.e[l as usize] as i32;
            let i = lut.i[l as usize] as i32;
            Some((l, h, e, i))
        }
    };

    let mut xy = 1u32;
    while vm & !xy.wrapping_sub(1) != 0 {
        if vm & xy != 0 {
            if let Some((l, h, e, i)) = derive_levels(lvl_offset) {
                // Determine current edge's wd-tier.
                let idx = if vmask[2] & xy != 0 {
                    16
                } else if vmask[1] & xy != 0 {
                    8
                } else {
                    4
                };

                // Fast path: x8 YMM kernel for narrow (wd=4) v-filter when
                // the next adjacent edge is also wd=4 with identical filter
                // parameters. Doubles throughput per call (8 cols vs 4).
                #[cfg(target_arch = "x86_64")]
                {
                    let next_xy = xy.wrapping_shl(1);
                    let next_is_narrow = next_xy != 0
                        && (vmask[0] & next_xy) != 0
                        && (vmask[1] & next_xy) == 0
                        && (vmask[2] & next_xy) == 0;
                    if idx == 4 && next_is_narrow && bitdepth_max == 255 {
                        if let Some((l2, _, _, _)) = derive_levels(lvl_offset + b4_stridea) {
                            if l2 == l {
                                loop_filter_4_8bpc_narrow_simd_v_x8(
                                    _token, buf, dst_offset, e, i, h, strideb,
                                );
                                // Advance both edges in one step.
                                xy = next_xy << 1;
                                dst_offset = signed_idx(dst_offset, 8 * stridea);
                                lvl_offset += 2 * b4_stridea;
                                continue;
                            }
                        }
                    }
                }

                loop_filter_4_8bpc(
                    #[cfg(target_arch = "x86_64")]
                    _token,
                    buf,
                    dst_offset,
                    e,
                    i,
                    h,
                    stridea,
                    strideb,
                    idx,
                    bitdepth_max,
                );
            }
        }

        xy <<= 1;
        dst_offset = signed_idx(dst_offset, 4 * stridea);
        lvl_offset += b4_stridea;
    }
}

/// Loop filter for UV planes, horizontal edges (8bpc)
#[cfg(any(target_arch = "x86_64", target_arch = "wasm32"))]
#[cfg_attr(target_arch = "x86_64", arcane)]
#[allow(unused_mut)]
fn lpf_h_sb_uv_8bpc_inner(
    #[cfg(target_arch = "x86_64")] _token: Desktop64,
    buf: &mut [u8],
    mut dst_offset: usize,
    stride: isize,
    vmask: &[u32; 3],
    lvl: &[AtomicU8],
    lvl_base: usize,
    lvl_byte_idx: usize,
    b4_stride: isize,
    lut: &Align16<Av1FilterLUT>,
    _w: i32,
    bitdepth_max: i32,
) {
    let stridea = stride;
    let strideb = 1isize;
    let b4_stridea = b4_stride as usize;
    let b4_strideb = 1usize;

    let vm = vmask[0] | vmask[1];
    let mut lvl_offset = lvl_base;

    let mut xy = 1u32;
    while vm & !xy.wrapping_sub(1) != 0 {
        if vm & xy != 0 {
            let lvl_val = read_lvl(lvl, lvl_offset, lvl_byte_idx);
            let l = if lvl_val != 0 {
                lvl_val
            } else {
                if lvl_offset >= b4_strideb {
                    read_lvl(lvl, lvl_offset - b4_strideb, lvl_byte_idx)
                } else {
                    0
                }
            };

            if l != 0 {
                let h = (l >> 4) as i32;
                let e = lut.e[l as usize] as i32;
                let i = lut.i[l as usize] as i32;

                let idx = if vmask[1] & xy != 0 { 6 } else { 4 };

                loop_filter_4_8bpc(
                    #[cfg(target_arch = "x86_64")]
                    _token,
                    buf,
                    dst_offset,
                    e,
                    i,
                    h,
                    stridea,
                    strideb,
                    idx,
                    bitdepth_max,
                );
            }
        }

        xy <<= 1;
        dst_offset = signed_idx(dst_offset, 4 * stridea);
        lvl_offset += b4_stridea;
    }
}

/// Loop filter for UV planes, vertical edges (8bpc)
#[cfg(any(target_arch = "x86_64", target_arch = "wasm32"))]
#[cfg_attr(target_arch = "x86_64", arcane)]
#[allow(unused_mut)]
fn lpf_v_sb_uv_8bpc_inner(
    #[cfg(target_arch = "x86_64")] _token: Desktop64,
    buf: &mut [u8],
    mut dst_offset: usize,
    stride: isize,
    vmask: &[u32; 3],
    lvl: &[AtomicU8],
    lvl_base: usize,
    lvl_byte_idx: usize,
    b4_stride: isize,
    lut: &Align16<Av1FilterLUT>,
    _w: i32,
    bitdepth_max: i32,
) {
    let stridea = 1isize;
    let strideb = stride;
    let b4_stridea = 1usize;
    let b4_strideb = b4_stride as usize;

    let vm = vmask[0] | vmask[1];
    let mut lvl_offset = lvl_base;

    let mut xy = 1u32;
    while vm & !xy.wrapping_sub(1) != 0 {
        if vm & xy != 0 {
            let lvl_val = read_lvl(lvl, lvl_offset, lvl_byte_idx);
            let l = if lvl_val != 0 {
                lvl_val
            } else {
                if lvl_offset >= b4_strideb {
                    read_lvl(lvl, lvl_offset - b4_strideb, lvl_byte_idx)
                } else {
                    0
                }
            };

            if l != 0 {
                let h = (l >> 4) as i32;
                let e = lut.e[l as usize] as i32;
                let i = lut.i[l as usize] as i32;

                let idx = if vmask[1] & xy != 0 { 6 } else { 4 };

                loop_filter_4_8bpc(
                    #[cfg(target_arch = "x86_64")]
                    _token,
                    buf,
                    dst_offset,
                    e,
                    i,
                    h,
                    stridea,
                    strideb,
                    idx,
                    bitdepth_max,
                );
            }
        }

        xy <<= 1;
        dst_offset = signed_idx(dst_offset, 4 * stridea);
        lvl_offset += b4_stridea;
    }
}

// ============================================================================
// FFI WRAPPERS (8bpc) — only compiled with asm feature
// ============================================================================

/// FFI wrapper for Y horizontal filter
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn lpf_h_sb_y_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    mask: &[u32; 3],
    lvl_ptr: *const [u8; 4],
    b4_stride: ptrdiff_t,
    lut: &Align16<Av1FilterLUT>,
    w: c_int,
    bitdepth_max: c_int,
    _dst: *const FFISafe<PicOffset>,
    _lvl: *const FFISafe<WithOffset<&[AtomicU8]>>,
) {
    // Determine buffer size needed: conservative upper bound
    let buf_len = compute_buf_len_u8(stride as isize, w);
    let buf = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_len) };
    let lvl_byte_len = compute_lvl_len(b4_stride as isize, w) * 4;
    let lvl = unsafe { std::slice::from_raw_parts(lvl_ptr as *const AtomicU8, lvl_byte_len) };
    let token = Desktop64::summon().expect("AVX2 implies Desktop64");
    lpf_h_sb_y_8bpc_inner(
        token,
        buf,
        0,
        stride as isize,
        mask,
        lvl,
        0,
        0,
        b4_stride as isize,
        lut,
        w,
        bitdepth_max,
    );
}

/// FFI wrapper for Y vertical filter
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn lpf_v_sb_y_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    mask: &[u32; 3],
    lvl_ptr: *const [u8; 4],
    b4_stride: ptrdiff_t,
    lut: &Align16<Av1FilterLUT>,
    w: c_int,
    bitdepth_max: c_int,
    _dst: *const FFISafe<PicOffset>,
    _lvl: *const FFISafe<WithOffset<&[AtomicU8]>>,
) {
    let buf_len = compute_buf_len_u8(stride as isize, w);
    let buf = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_len) };
    let lvl_byte_len = compute_lvl_len(b4_stride as isize, w) * 4;
    let lvl = unsafe { std::slice::from_raw_parts(lvl_ptr as *const AtomicU8, lvl_byte_len) };
    let token = Desktop64::summon().expect("AVX2 implies Desktop64");
    lpf_v_sb_y_8bpc_inner(
        token,
        buf,
        0,
        stride as isize,
        mask,
        lvl,
        0,
        0,
        b4_stride as isize,
        lut,
        w,
        bitdepth_max,
    );
}

/// FFI wrapper for UV horizontal filter
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn lpf_h_sb_uv_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    mask: &[u32; 3],
    lvl_ptr: *const [u8; 4],
    b4_stride: ptrdiff_t,
    lut: &Align16<Av1FilterLUT>,
    w: c_int,
    bitdepth_max: c_int,
    _dst: *const FFISafe<PicOffset>,
    _lvl: *const FFISafe<WithOffset<&[AtomicU8]>>,
) {
    let buf_len = compute_buf_len_u8(stride as isize, w);
    let buf = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_len) };
    let lvl_byte_len = compute_lvl_len(b4_stride as isize, w) * 4;
    let lvl = unsafe { std::slice::from_raw_parts(lvl_ptr as *const AtomicU8, lvl_byte_len) };
    let token = Desktop64::summon().expect("AVX2 implies Desktop64");
    lpf_h_sb_uv_8bpc_inner(
        token,
        buf,
        0,
        stride as isize,
        mask,
        lvl,
        0,
        0,
        b4_stride as isize,
        lut,
        w,
        bitdepth_max,
    );
}

/// FFI wrapper for UV vertical filter
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn lpf_v_sb_uv_8bpc_avx2(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    mask: &[u32; 3],
    lvl_ptr: *const [u8; 4],
    b4_stride: ptrdiff_t,
    lut: &Align16<Av1FilterLUT>,
    w: c_int,
    bitdepth_max: c_int,
    _dst: *const FFISafe<PicOffset>,
    _lvl: *const FFISafe<WithOffset<&[AtomicU8]>>,
) {
    let buf_len = compute_buf_len_u8(stride as isize, w);
    let buf = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u8, buf_len) };
    let lvl_byte_len = compute_lvl_len(b4_stride as isize, w) * 4;
    let lvl = unsafe { std::slice::from_raw_parts(lvl_ptr as *const AtomicU8, lvl_byte_len) };
    let token = Desktop64::summon().expect("AVX2 implies Desktop64");
    lpf_v_sb_uv_8bpc_inner(
        token,
        buf,
        0,
        stride as isize,
        mask,
        lvl,
        0,
        0,
        b4_stride as isize,
        lut,
        w,
        bitdepth_max,
    );
}

// ============================================================================
// 16BPC IMPLEMENTATIONS
// ============================================================================

/// Core loop filter for 16bpc - processes 4 pixels
#[cfg(any(target_arch = "x86_64", target_arch = "wasm32"))]
fn loop_filter_4_16bpc(
    buf: &mut [u16],
    base: usize,
    e: i32,
    i: i32,
    h: i32,
    stridea: isize,
    strideb: isize,
    wd: i32,
    bitdepth_max: i32,
) {
    let bitdepth_min_8 = if bitdepth_max > 255 {
        if bitdepth_max > 1023 { 4 } else { 2 }
    } else {
        0
    };
    let f = 1i32 << bitdepth_min_8;
    let e = e << bitdepth_min_8;
    let i = i << bitdepth_min_8;
    let h = h << bitdepth_min_8;

    for idx in 0..4isize {
        let edge = signed_idx(base, idx * stridea);

        let get_px = |offset: isize| -> i32 { buf[signed_idx(edge, strideb * offset)] as i32 };

        let p1 = get_px(-2);
        let p0 = get_px(-1);
        let q0 = get_px(0);
        let q1 = get_px(1);

        let mut fm = (p1 - p0).abs() <= i
            && (q1 - q0).abs() <= i
            && (p0 - q0).abs() * 2 + ((p1 - q1).abs() >> 1) <= e;

        let (mut p2, mut p3, mut q2, mut q3) = (0, 0, 0, 0);
        let (mut p4, mut p5, mut p6, mut q4, mut q5, mut q6) = (0, 0, 0, 0, 0, 0);

        if wd > 4 {
            p2 = get_px(-3);
            q2 = get_px(2);
            fm &= (p2 - p1).abs() <= i && (q2 - q1).abs() <= i;

            if wd > 6 {
                p3 = get_px(-4);
                q3 = get_px(3);
                fm &= (p3 - p2).abs() <= i && (q3 - q2).abs() <= i;
            }
        }

        if !fm {
            continue;
        }

        let mut flat8out = false;
        let mut flat8in = false;

        if wd >= 16 {
            p6 = get_px(-7);
            p5 = get_px(-6);
            p4 = get_px(-5);
            q4 = get_px(4);
            q5 = get_px(5);
            q6 = get_px(6);

            flat8out = (p6 - p0).abs() <= f
                && (p5 - p0).abs() <= f
                && (p4 - p0).abs() <= f
                && (q4 - q0).abs() <= f
                && (q5 - q0).abs() <= f
                && (q6 - q0).abs() <= f;
        }

        if wd >= 6 {
            flat8in = (p2 - p0).abs() <= f
                && (p1 - p0).abs() <= f
                && (q1 - q0).abs() <= f
                && (q2 - q0).abs() <= f;
        }

        if wd >= 8 {
            flat8in &= (p3 - p0).abs() <= f && (q3 - q0).abs() <= f;
        }

        let set_px = |buf: &mut [u16], offset: isize, val: i32| {
            buf[signed_idx(edge, strideb * offset)] = val.clamp(0, bitdepth_max) as u16;
        };

        if wd >= 16 && flat8out && flat8in {
            set_px(
                buf,
                -6,
                (p6 + p6 + p6 + p6 + p6 + p6 * 2 + p5 * 2 + p4 * 2 + p3 + p2 + p1 + p0 + q0 + 8)
                    >> 4,
            );
            set_px(
                buf,
                -5,
                (p6 + p6 + p6 + p6 + p6 + p5 * 2 + p4 * 2 + p3 * 2 + p2 + p1 + p0 + q0 + q1 + 8)
                    >> 4,
            );
            set_px(
                buf,
                -4,
                (p6 + p6 + p6 + p6 + p5 + p4 * 2 + p3 * 2 + p2 * 2 + p1 + p0 + q0 + q1 + q2 + 8)
                    >> 4,
            );
            set_px(
                buf,
                -3,
                (p6 + p6 + p6 + p5 + p4 + p3 * 2 + p2 * 2 + p1 * 2 + p0 + q0 + q1 + q2 + q3 + 8)
                    >> 4,
            );
            set_px(
                buf,
                -2,
                (p6 + p6 + p5 + p4 + p3 + p2 * 2 + p1 * 2 + p0 * 2 + q0 + q1 + q2 + q3 + q4 + 8)
                    >> 4,
            );
            set_px(
                buf,
                -1,
                (p6 + p5 + p4 + p3 + p2 + p1 * 2 + p0 * 2 + q0 * 2 + q1 + q2 + q3 + q4 + q5 + 8)
                    >> 4,
            );
            set_px(
                buf,
                0,
                (p5 + p4 + p3 + p2 + p1 + p0 * 2 + q0 * 2 + q1 * 2 + q2 + q3 + q4 + q5 + q6 + 8)
                    >> 4,
            );
            set_px(
                buf,
                1,
                (p4 + p3 + p2 + p1 + p0 + q0 * 2 + q1 * 2 + q2 * 2 + q3 + q4 + q5 + q6 + q6 + 8)
                    >> 4,
            );
            set_px(
                buf,
                2,
                (p3 + p2 + p1 + p0 + q0 + q1 * 2 + q2 * 2 + q3 * 2 + q4 + q5 + q6 + q6 + q6 + 8)
                    >> 4,
            );
            set_px(
                buf,
                3,
                (p2 + p1 + p0 + q0 + q1 + q2 * 2 + q3 * 2 + q4 * 2 + q5 + q6 + q6 + q6 + q6 + 8)
                    >> 4,
            );
            set_px(
                buf,
                4,
                (p1 + p0 + q0 + q1 + q2 + q3 * 2 + q4 * 2 + q5 * 2 + q6 + q6 + q6 + q6 + q6 + 8)
                    >> 4,
            );
            set_px(
                buf,
                5,
                (p0 + q0 + q1 + q2 + q3 + q4 * 2 + q5 * 2 + q6 * 2 + q6 + q6 + q6 + q6 + q6 + 8)
                    >> 4,
            );
        } else if wd >= 8 && flat8in {
            set_px(buf, -3, (p3 + p3 + p3 + 2 * p2 + p1 + p0 + q0 + 4) >> 3);
            set_px(buf, -2, (p3 + p3 + p2 + 2 * p1 + p0 + q0 + q1 + 4) >> 3);
            set_px(buf, -1, (p3 + p2 + p1 + 2 * p0 + q0 + q1 + q2 + 4) >> 3);
            set_px(buf, 0, (p2 + p1 + p0 + 2 * q0 + q1 + q2 + q3 + 4) >> 3);
            set_px(buf, 1, (p1 + p0 + q0 + 2 * q1 + q2 + q3 + q3 + 4) >> 3);
            set_px(buf, 2, (p0 + q0 + q1 + 2 * q2 + q3 + q3 + q3 + 4) >> 3);
        } else if wd >= 6 && flat8in {
            set_px(buf, -2, (p2 + 2 * p2 + 2 * p1 + 2 * p0 + q0 + 4) >> 3);
            set_px(buf, -1, (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3);
            set_px(buf, 0, (p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3);
            set_px(buf, 1, (p0 + 2 * q0 + 2 * q1 + 2 * q2 + q2 + 4) >> 3);
        } else {
            let hev = (p1 - p0).abs() > h || (q1 - q0).abs() > h;

            let bdm8 = bitdepth_min_8 as u8;
            if hev {
                let f = iclip_diff(p1 - q1, bdm8);
                let f = iclip_diff(3 * (q0 - p0) + f, bdm8);

                let f1 = cmp::min(f + 4, (128 << bdm8) - 1) >> 3;
                let f2 = cmp::min(f + 3, (128 << bdm8) - 1) >> 3;

                set_px(buf, -1, iclip(p0 + f2, 0, bitdepth_max));
                set_px(buf, 0, iclip(q0 - f1, 0, bitdepth_max));
            } else {
                let f = iclip_diff(3 * (q0 - p0), bdm8);

                let f1 = cmp::min(f + 4, (128 << bdm8) - 1) >> 3;
                let f2 = cmp::min(f + 3, (128 << bdm8) - 1) >> 3;

                set_px(buf, -1, iclip(p0 + f2, 0, bitdepth_max));
                set_px(buf, 0, iclip(q0 - f1, 0, bitdepth_max));

                let f3 = (f1 + 1) >> 1;
                set_px(buf, -2, iclip(p1 + f3, 0, bitdepth_max));
                set_px(buf, 1, iclip(q1 - f3, 0, bitdepth_max));
            }
        }
    }
}

// ============================================================================
// SUPERBLOCK FILTER FUNCTIONS (16bpc)
// ============================================================================

/// Loop filter Y horizontal 16bpc inner
#[cfg(any(target_arch = "x86_64", target_arch = "wasm32"))]
fn lpf_h_sb_y_16bpc_inner(
    buf: &mut [u16],
    mut dst_offset: usize,
    stride_u16: isize,
    vmask: &[u32; 3],
    lvl: &[AtomicU8],
    lvl_base: usize,
    lvl_byte_idx: usize,
    b4_stride: isize,
    lut: &Align16<Av1FilterLUT>,
    _w: i32,
    bitdepth_max: i32,
) {
    let stridea = stride_u16;
    let strideb = 1isize;
    let b4_stridea = b4_stride as usize;
    let b4_strideb = 1usize;

    let vm = vmask[0] | vmask[1] | vmask[2];
    let mut lvl_offset = lvl_base;

    let mut xy = 1u32;
    while vm & !xy.wrapping_sub(1) != 0 {
        if vm & xy != 0 {
            let lvl_val = read_lvl(lvl, lvl_offset, lvl_byte_idx);
            let l = if lvl_val != 0 {
                lvl_val
            } else {
                if lvl_offset >= b4_strideb {
                    read_lvl(lvl, lvl_offset - b4_strideb, lvl_byte_idx)
                } else {
                    0
                }
            };

            if l != 0 {
                let h = (l >> 4) as i32;
                let e = lut.e[l as usize] as i32;
                let i = lut.i[l as usize] as i32;

                let idx = if vmask[2] & xy != 0 {
                    16
                } else if vmask[1] & xy != 0 {
                    8
                } else {
                    4
                };

                loop_filter_4_16bpc(
                    buf,
                    dst_offset,
                    e,
                    i,
                    h,
                    stridea,
                    strideb,
                    idx,
                    bitdepth_max,
                );
            }
        }

        xy <<= 1;
        dst_offset = signed_idx(dst_offset, 4 * stridea);
        lvl_offset += b4_stridea;
    }
}

/// Loop filter Y vertical 16bpc inner
#[cfg(any(target_arch = "x86_64", target_arch = "wasm32"))]
fn lpf_v_sb_y_16bpc_inner(
    buf: &mut [u16],
    mut dst_offset: usize,
    stride_u16: isize,
    vmask: &[u32; 3],
    lvl: &[AtomicU8],
    lvl_base: usize,
    lvl_byte_idx: usize,
    b4_stride: isize,
    lut: &Align16<Av1FilterLUT>,
    _w: i32,
    bitdepth_max: i32,
) {
    let stridea = 1isize;
    let strideb = stride_u16;
    let b4_stridea = 1usize;
    let b4_strideb = b4_stride as usize;

    let vm = vmask[0] | vmask[1] | vmask[2];
    let mut lvl_offset = lvl_base;

    let mut xy = 1u32;
    while vm & !xy.wrapping_sub(1) != 0 {
        if vm & xy != 0 {
            let lvl_val = read_lvl(lvl, lvl_offset, lvl_byte_idx);
            let l = if lvl_val != 0 {
                lvl_val
            } else {
                // Note: original uses b4_strideb (not 4*b4_strideb) for V direction lookback
                if lvl_offset >= b4_strideb {
                    read_lvl(lvl, lvl_offset - b4_strideb, lvl_byte_idx)
                } else {
                    0
                }
            };

            if l != 0 {
                let h = (l >> 4) as i32;
                let e = lut.e[l as usize] as i32;
                let i = lut.i[l as usize] as i32;

                let idx = if vmask[2] & xy != 0 {
                    16
                } else if vmask[1] & xy != 0 {
                    8
                } else {
                    4
                };

                loop_filter_4_16bpc(
                    buf,
                    dst_offset,
                    e,
                    i,
                    h,
                    stridea,
                    strideb,
                    idx,
                    bitdepth_max,
                );
            }
        }

        xy <<= 1;
        dst_offset = signed_idx(dst_offset, 4 * stridea);
        lvl_offset += b4_stridea;
    }
}

/// Loop filter UV horizontal 16bpc inner
#[cfg(any(target_arch = "x86_64", target_arch = "wasm32"))]
fn lpf_h_sb_uv_16bpc_inner(
    buf: &mut [u16],
    mut dst_offset: usize,
    stride_u16: isize,
    vmask: &[u32; 3],
    lvl: &[AtomicU8],
    lvl_base: usize,
    lvl_byte_idx: usize,
    b4_stride: isize,
    lut: &Align16<Av1FilterLUT>,
    _w: i32,
    bitdepth_max: i32,
) {
    let stridea = stride_u16;
    let strideb = 1isize;
    let b4_stridea = b4_stride as usize;
    let b4_strideb = 1usize;

    let vm = vmask[0] | vmask[1];
    let mut lvl_offset = lvl_base;

    let mut xy = 1u32;
    while vm & !xy.wrapping_sub(1) != 0 {
        if vm & xy != 0 {
            let lvl_val = read_lvl(lvl, lvl_offset, lvl_byte_idx);
            let l = if lvl_val != 0 {
                lvl_val
            } else {
                if lvl_offset >= b4_strideb {
                    read_lvl(lvl, lvl_offset - b4_strideb, lvl_byte_idx)
                } else {
                    0
                }
            };

            if l != 0 {
                let h = (l >> 4) as i32;
                let e = lut.e[l as usize] as i32;
                let i = lut.i[l as usize] as i32;

                let idx = if vmask[1] & xy != 0 { 6 } else { 4 };

                loop_filter_4_16bpc(
                    buf,
                    dst_offset,
                    e,
                    i,
                    h,
                    stridea,
                    strideb,
                    idx,
                    bitdepth_max,
                );
            }
        }

        xy <<= 1;
        dst_offset = signed_idx(dst_offset, 4 * stridea);
        lvl_offset += b4_stridea;
    }
}

/// Loop filter UV vertical 16bpc inner
#[cfg(any(target_arch = "x86_64", target_arch = "wasm32"))]
fn lpf_v_sb_uv_16bpc_inner(
    buf: &mut [u16],
    mut dst_offset: usize,
    stride_u16: isize,
    vmask: &[u32; 3],
    lvl: &[AtomicU8],
    lvl_base: usize,
    lvl_byte_idx: usize,
    b4_stride: isize,
    lut: &Align16<Av1FilterLUT>,
    _w: i32,
    bitdepth_max: i32,
) {
    let stridea = 1isize;
    let strideb = stride_u16;
    let b4_stridea = 1usize;
    let b4_strideb = b4_stride as usize;

    let vm = vmask[0] | vmask[1];
    let mut lvl_offset = lvl_base;

    let mut xy = 1u32;
    while vm & !xy.wrapping_sub(1) != 0 {
        if vm & xy != 0 {
            let lvl_val = read_lvl(lvl, lvl_offset, lvl_byte_idx);
            let l = if lvl_val != 0 {
                lvl_val
            } else {
                // Note: original uses b4_strideb (not 4*b4_strideb) for V direction lookback
                if lvl_offset >= b4_strideb {
                    read_lvl(lvl, lvl_offset - b4_strideb, lvl_byte_idx)
                } else {
                    0
                }
            };

            if l != 0 {
                let h = (l >> 4) as i32;
                let e = lut.e[l as usize] as i32;
                let i = lut.i[l as usize] as i32;

                let idx = if vmask[1] & xy != 0 { 6 } else { 4 };

                loop_filter_4_16bpc(
                    buf,
                    dst_offset,
                    e,
                    i,
                    h,
                    stridea,
                    strideb,
                    idx,
                    bitdepth_max,
                );
            }
        }

        xy <<= 1;
        dst_offset = signed_idx(dst_offset, 4 * stridea);
        lvl_offset += b4_stridea;
    }
}

// ============================================================================
// FFI WRAPPERS (16bpc) — only compiled with asm feature
// ============================================================================

/// FFI wrapper for Y horizontal filter 16bpc
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn lpf_h_sb_y_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    mask: &[u32; 3],
    lvl_ptr: *const [u8; 4],
    b4_stride: ptrdiff_t,
    lut: &Align16<Av1FilterLUT>,
    w: c_int,
    bitdepth_max: c_int,
    _dst: *const FFISafe<PicOffset>,
    _lvl: *const FFISafe<WithOffset<&[AtomicU8]>>,
) {
    let buf_len = compute_buf_len_u16(stride as isize, w);
    let buf = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u16, buf_len) };
    let lvl_byte_len = compute_lvl_len(b4_stride as isize, w) * 4;
    let lvl = unsafe { std::slice::from_raw_parts(lvl_ptr as *const AtomicU8, lvl_byte_len) };
    lpf_h_sb_y_16bpc_inner(
        buf,
        0,
        stride as isize / 2,
        mask,
        lvl,
        0,
        0,
        b4_stride as isize,
        lut,
        w,
        bitdepth_max,
    );
}

/// FFI wrapper for Y vertical filter 16bpc
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn lpf_v_sb_y_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    mask: &[u32; 3],
    lvl_ptr: *const [u8; 4],
    b4_stride: ptrdiff_t,
    lut: &Align16<Av1FilterLUT>,
    w: c_int,
    bitdepth_max: c_int,
    _dst: *const FFISafe<PicOffset>,
    _lvl: *const FFISafe<WithOffset<&[AtomicU8]>>,
) {
    let buf_len = compute_buf_len_u16(stride as isize, w);
    let buf = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u16, buf_len) };
    let lvl_byte_len = compute_lvl_len(b4_stride as isize, w) * 4;
    let lvl = unsafe { std::slice::from_raw_parts(lvl_ptr as *const AtomicU8, lvl_byte_len) };
    lpf_v_sb_y_16bpc_inner(
        buf,
        0,
        stride as isize / 2,
        mask,
        lvl,
        0,
        0,
        b4_stride as isize,
        lut,
        w,
        bitdepth_max,
    );
}

/// FFI wrapper for UV horizontal filter 16bpc
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn lpf_h_sb_uv_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    mask: &[u32; 3],
    lvl_ptr: *const [u8; 4],
    b4_stride: ptrdiff_t,
    lut: &Align16<Av1FilterLUT>,
    w: c_int,
    bitdepth_max: c_int,
    _dst: *const FFISafe<PicOffset>,
    _lvl: *const FFISafe<WithOffset<&[AtomicU8]>>,
) {
    let buf_len = compute_buf_len_u16(stride as isize, w);
    let buf = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u16, buf_len) };
    let lvl_byte_len = compute_lvl_len(b4_stride as isize, w) * 4;
    let lvl = unsafe { std::slice::from_raw_parts(lvl_ptr as *const AtomicU8, lvl_byte_len) };
    lpf_h_sb_uv_16bpc_inner(
        buf,
        0,
        stride as isize / 2,
        mask,
        lvl,
        0,
        0,
        b4_stride as isize,
        lut,
        w,
        bitdepth_max,
    );
}

/// FFI wrapper for UV vertical filter 16bpc
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn lpf_v_sb_uv_16bpc_avx2(
    dst_ptr: *mut DynPixel,
    stride: ptrdiff_t,
    mask: &[u32; 3],
    lvl_ptr: *const [u8; 4],
    b4_stride: ptrdiff_t,
    lut: &Align16<Av1FilterLUT>,
    w: c_int,
    bitdepth_max: c_int,
    _dst: *const FFISafe<PicOffset>,
    _lvl: *const FFISafe<WithOffset<&[AtomicU8]>>,
) {
    let buf_len = compute_buf_len_u16(stride as isize, w);
    let buf = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u16, buf_len) };
    let lvl_byte_len = compute_lvl_len(b4_stride as isize, w) * 4;
    let lvl = unsafe { std::slice::from_raw_parts(lvl_ptr as *const AtomicU8, lvl_byte_len) };
    lpf_v_sb_uv_16bpc_inner(
        buf,
        0,
        stride as isize / 2,
        mask,
        lvl,
        0,
        0,
        b4_stride as isize,
        lut,
        w,
        bitdepth_max,
    );
}

// ============================================================================
// BUFFER SIZE HELPERS (for FFI wrappers)
// ============================================================================

/// Compute a conservative buffer length for u8 pixel buffers.
/// The filter accesses up to 7 pixels on each side of the edge,
/// and processes up to 32 4-pixel blocks along the stride direction.
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
fn compute_buf_len_u8(stride: isize, _w: i32) -> usize {
    // Up to 32 iterations * 4 * stride + 7 pixels of reach
    (stride.unsigned_abs() * 128 + 8) as usize
}

/// Compute a conservative buffer length for u16 pixel buffers.
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
fn compute_buf_len_u16(stride: isize, _w: i32) -> usize {
    // stride is in bytes for u16, so divide by 2 for element count
    let stride_u16 = stride.unsigned_abs() / 2;
    (stride_u16 * 128 + 8) as usize
}

/// Compute a conservative lvl slice length (in [u8; 4] elements).
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
fn compute_lvl_len(b4_stride: isize, _w: i32) -> usize {
    // Up to 32 iterations * b4_stride + lookback of b4_stride (conservative)
    (b4_stride.unsigned_abs() as usize) * 132 + 4
}

/// Safe dispatch for loopfilter_sb on x86_64. Returns true if SIMD was used.
#[cfg(target_arch = "x86_64")]
pub fn loopfilter_sb_dispatch<BD: BitDepth>(
    dst: PicOffset,
    stride: ptrdiff_t,
    mask: &[u32; 3],
    lvl: WithOffset<&[AtomicU8]>,
    b4_stride: isize,
    lut: &Align16<Av1FilterLUT>,
    w: c_int,
    bitdepth_max: c_int,
    is_y: bool,
    is_v: bool,
) -> bool {
    use crate::include::common::bitdepth::BPC;

    // Summon Desktop64 (AVX2+FMA+BMI2) token once at the outer dispatch —
    // passed through to the `#[arcane]` outer inners so per-edge SIMD calls
    // inline (no trampoline). AVX2 unlocks YMM-wide intrinsics if the
    // inner kernels widen.
    let Some(token) = crate::src::cpu::summon_avx2() else {
        return false;
    };

    assert!(lvl.offset <= lvl.data.len());

    // Direct slice access for lvl data: read AtomicU8 values on demand.
    //
    // Include lookback entries: when the current block's level is 0, the inner
    // functions read the PREVIOUS block's level (lvl_offset - b4_strideb).
    // We include those entries by starting the slice early.
    let b4_strideb_entries = if !is_v {
        1usize
    } else {
        b4_stride.unsigned_abs() as usize
    };
    let lvl_lookback_bytes = b4_strideb_entries * 4;
    let lvl_start = lvl.offset.saturating_sub(lvl_lookback_bytes) & !3;
    let lvl_slice = &lvl.data[lvl_start..];
    // Which byte within each 4-byte entry to read:
    //   H Y → 0, V Y → 1, H U → 2, H V → 3
    // This is encoded in lvl.offset % 4 by the caller (lf_apply.rs adds +0/+1/+2/+3).
    let lvl_byte_idx = lvl.offset % 4;
    // Base offset: how many 4-byte entries from the start of lvl_slice to the
    // original lvl.offset's 4-byte-aligned position
    let lvl_base = (lvl.offset - lvl_byte_idx - lvl_start) / 4;

    // Compute actual iterations from vmask to tighten bounds check.
    let vm = mask[0] | mask[1] | mask[2];
    if vm == 0 {
        return true; // Nothing to filter
    }
    let max_iter = 32 - vm.leading_zeros() as usize;

    match BD::BPC {
        BPC::BPC8 => {
            use crate::include::common::bitdepth::BitDepth8;

            // For 8bpc, the stride is in bytes (= pixels).
            let byte_stride = stride.unsigned_abs() as usize;

            // Compute reach based on filter direction and actual vmask extent.
            // H filter (is_v=false): iterates rows (stridea=stride), pixel access (strideb=1)
            //   forward: last group at (max_iter-1)*4*stride, +3 lines, +16 pixels
            //   backward: 7 pixels horizontally
            // V filter (is_v=true): iterates columns (stridea=1), row access (strideb=stride)
            //   forward: (max_iter*4-1) columns + 16*stride rows
            //   backward: 7*stride rows
            let (reach_before, reach_after) = if !is_v {
                // H filter: iterates through row groups
                (7, (max_iter * 4 - 1) * byte_stride + 16)
            } else {
                // V filter: iterates through column groups
                (7 * byte_stride, max_iter * 4 - 1 + 16 * byte_stride)
            };

            // Guard: fall back to scalar if buffer bounds are insufficient.
            let buf_pixel_len = dst.data.pixel_len::<BitDepth8>();
            if dst.offset < reach_before || dst.offset.saturating_add(reach_after) > buf_pixel_len {
                return false;
            }

            // COW: single-threaded uses the original wide guard (zero-copy),
            // multi-threaded decomposes into a 2D compact buffer with per-row guards.
            let use_compact = crate::include::dav1d::picture::tile_threading_active();

            let start_pixel = dst.offset - reach_before;
            let total_pixels = (reach_before + reach_after).min(buf_pixel_len - start_pixel);

            if use_compact {
                let (cw, ch, cstart, cbase) = if !is_v {
                    (7 + 16, max_iter * 4, dst.offset - 7, 7usize)
                } else {
                    let cw = max_iter * 4;
                    (
                        cw,
                        7 + 16, // 23 rows: 7 above + 16 below
                        dst.offset.saturating_sub(7 * byte_stride),
                        7 * cw,
                    )
                };
                let lpf_pic = crate::src::with_offset::WithOffset {
                    data: dst.data,
                    offset: cstart,
                };
                let (mut cb, cs) = lpf_pic.compact_read_per_row::<BitDepth8>(cw, ch);
                let buf: &mut [u8] = &mut cb;
                let base = cbase;
                let stride_i = cs as isize;
                match (is_y, is_v) {
                    (true, false) => lpf_h_sb_y_8bpc_inner(
                        token,
                        buf,
                        base,
                        stride_i,
                        mask,
                        lvl_slice,
                        lvl_base,
                        lvl_byte_idx,
                        b4_stride,
                        lut,
                        w,
                        bitdepth_max,
                    ),
                    (true, true) => lpf_v_sb_y_8bpc_inner(
                        token,
                        buf,
                        base,
                        stride_i,
                        mask,
                        lvl_slice,
                        lvl_base,
                        lvl_byte_idx,
                        b4_stride,
                        lut,
                        w,
                        bitdepth_max,
                    ),
                    (false, false) => lpf_h_sb_uv_8bpc_inner(
                        token,
                        buf,
                        base,
                        stride_i,
                        mask,
                        lvl_slice,
                        lvl_base,
                        lvl_byte_idx,
                        b4_stride,
                        lut,
                        w,
                        bitdepth_max,
                    ),
                    (false, true) => lpf_v_sb_uv_8bpc_inner(
                        token,
                        buf,
                        base,
                        stride_i,
                        mask,
                        lvl_slice,
                        lvl_base,
                        lvl_byte_idx,
                        b4_stride,
                        lut,
                        w,
                        bitdepth_max,
                    ),
                }
                lpf_pic.compact_write_back_per_row::<BitDepth8>(cw, ch, &cb);
            } else {
                let mut guard = dst
                    .data
                    .slice_mut::<BitDepth8, _>((start_pixel.., ..total_pixels));
                let buf: &mut [u8] = &mut *guard;
                let base = reach_before;
                let stride_i = stride as isize;
                match (is_y, is_v) {
                    (true, false) => lpf_h_sb_y_8bpc_inner(
                        token,
                        buf,
                        base,
                        stride_i,
                        mask,
                        lvl_slice,
                        lvl_base,
                        lvl_byte_idx,
                        b4_stride,
                        lut,
                        w,
                        bitdepth_max,
                    ),
                    (true, true) => lpf_v_sb_y_8bpc_inner(
                        token,
                        buf,
                        base,
                        stride_i,
                        mask,
                        lvl_slice,
                        lvl_base,
                        lvl_byte_idx,
                        b4_stride,
                        lut,
                        w,
                        bitdepth_max,
                    ),
                    (false, false) => lpf_h_sb_uv_8bpc_inner(
                        token,
                        buf,
                        base,
                        stride_i,
                        mask,
                        lvl_slice,
                        lvl_base,
                        lvl_byte_idx,
                        b4_stride,
                        lut,
                        w,
                        bitdepth_max,
                    ),
                    (false, true) => lpf_v_sb_uv_8bpc_inner(
                        token,
                        buf,
                        base,
                        stride_i,
                        mask,
                        lvl_slice,
                        lvl_base,
                        lvl_byte_idx,
                        b4_stride,
                        lut,
                        w,
                        bitdepth_max,
                    ),
                }
            }
        }
        BPC::BPC16 => {
            use crate::include::common::bitdepth::BitDepth16;

            let u16_stride = (stride / 2).unsigned_abs() as usize;

            // Compute reach based on filter direction and actual vmask extent
            let (reach_before, reach_after) = if !is_v {
                // H filter: iterates through row groups
                (7, (max_iter * 4 - 1) * u16_stride + 16)
            } else {
                // V filter: iterates through column groups
                (7 * u16_stride, max_iter * 4 - 1 + 16 * u16_stride)
            };

            // Guard: fall back to scalar if buffer bounds are insufficient.
            let buf_pixel_len = dst.data.pixel_len::<BitDepth16>();
            if dst.offset < reach_before || dst.offset.saturating_add(reach_after) > buf_pixel_len {
                return false;
            }

            // COW: single-threaded uses the original wide guard (zero-copy),
            // multi-threaded decomposes into a 2D compact buffer with per-row guards.
            let use_compact = crate::include::dav1d::picture::tile_threading_active();

            if use_compact {
                let (compact_w, compact_h, start_pixel, base) = if !is_v {
                    // H filter: 23 pixels wide (-7 to +15), max_iter*4 rows tall
                    let w = 7 + 16; // 23
                    let h = max_iter * 4;
                    let start = dst.offset - 7;
                    (w, h, start, 7usize)
                } else {
                    // V filter: max_iter*4 + 16 pixels wide, 23 rows tall
                    let w = max_iter * 4;
                    let h = 7 + 16; // 23
                    let start = dst.offset.saturating_sub(7 * u16_stride);
                    (w, h, start, 7 * w)
                };
                let lpf_pic = crate::src::with_offset::WithOffset {
                    data: dst.data,
                    offset: start_pixel,
                };
                let (mut compact, compact_stride) =
                    lpf_pic.compact_read_per_row::<BitDepth16>(compact_w, compact_h);
                let buf: &mut [u16] =
                    zerocopy::FromBytes::mut_from_bytes(&mut compact[..]).unwrap();
                let stride_i = (compact_stride / 2) as isize;

                match (is_y, is_v) {
                    (true, false) => lpf_h_sb_y_16bpc_inner(
                        buf,
                        base,
                        stride_i,
                        mask,
                        lvl_slice,
                        lvl_base,
                        lvl_byte_idx,
                        b4_stride,
                        lut,
                        w,
                        bitdepth_max,
                    ),
                    (true, true) => lpf_v_sb_y_16bpc_inner(
                        buf,
                        base,
                        stride_i,
                        mask,
                        lvl_slice,
                        lvl_base,
                        lvl_byte_idx,
                        b4_stride,
                        lut,
                        w,
                        bitdepth_max,
                    ),
                    (false, false) => lpf_h_sb_uv_16bpc_inner(
                        buf,
                        base,
                        stride_i,
                        mask,
                        lvl_slice,
                        lvl_base,
                        lvl_byte_idx,
                        b4_stride,
                        lut,
                        w,
                        bitdepth_max,
                    ),
                    (false, true) => lpf_v_sb_uv_16bpc_inner(
                        buf,
                        base,
                        stride_i,
                        mask,
                        lvl_slice,
                        lvl_base,
                        lvl_byte_idx,
                        b4_stride,
                        lut,
                        w,
                        bitdepth_max,
                    ),
                }
                lpf_pic.compact_write_back_per_row::<BitDepth16>(compact_w, compact_h, &compact);
            } else {
                let start_pixel = dst.offset - reach_before;
                let total_pixels = (reach_before + reach_after).min(buf_pixel_len - start_pixel);
                let mut guard = dst
                    .data
                    .slice_mut::<BitDepth16, _>((start_pixel.., ..total_pixels));
                let buf: &mut [u16] = &mut *guard;
                let base = reach_before;
                let stride_i = stride as isize / 2;

                match (is_y, is_v) {
                    (true, false) => lpf_h_sb_y_16bpc_inner(
                        buf,
                        base,
                        stride_i,
                        mask,
                        lvl_slice,
                        lvl_base,
                        lvl_byte_idx,
                        b4_stride,
                        lut,
                        w,
                        bitdepth_max,
                    ),
                    (true, true) => lpf_v_sb_y_16bpc_inner(
                        buf,
                        base,
                        stride_i,
                        mask,
                        lvl_slice,
                        lvl_base,
                        lvl_byte_idx,
                        b4_stride,
                        lut,
                        w,
                        bitdepth_max,
                    ),
                    (false, false) => lpf_h_sb_uv_16bpc_inner(
                        buf,
                        base,
                        stride_i,
                        mask,
                        lvl_slice,
                        lvl_base,
                        lvl_byte_idx,
                        b4_stride,
                        lut,
                        w,
                        bitdepth_max,
                    ),
                    (false, true) => lpf_v_sb_uv_16bpc_inner(
                        buf,
                        base,
                        stride_i,
                        mask,
                        lvl_slice,
                        lvl_base,
                        lvl_byte_idx,
                        b4_stride,
                        lut,
                        w,
                        bitdepth_max,
                    ),
                }
            }
        }
    }
    true
}

/// Safe dispatch for loopfilter_sb on wasm32. Returns true if handled.
///
/// The inner filter functions are scalar (no SIMD intrinsics). The `&[AtomicU8]`
/// level cache is passed directly to inner functions which load entries on demand.
#[cfg(target_arch = "wasm32")]
pub fn loopfilter_sb_dispatch<BD: BitDepth>(
    dst: PicOffset,
    stride: ptrdiff_t,
    mask: &[u32; 3],
    lvl: WithOffset<&[AtomicU8]>,
    b4_stride: isize,
    lut: &Align16<Av1FilterLUT>,
    w: c_int,
    bitdepth_max: c_int,
    is_y: bool,
    is_v: bool,
) -> bool {
    use crate::include::common::bitdepth::BPC;

    assert!(lvl.offset <= lvl.data.len());

    // Direct slice access for lvl data: read AtomicU8 values on demand.
    let b4_strideb_entries = if !is_v {
        1usize
    } else {
        b4_stride.unsigned_abs() as usize
    };
    let lvl_lookback_bytes = b4_strideb_entries * 4;
    let lvl_start = lvl.offset.saturating_sub(lvl_lookback_bytes) & !3;
    let lvl_slice = &lvl.data[lvl_start..];
    let lvl_byte_idx = lvl.offset % 4;
    let lvl_base = (lvl.offset - lvl_byte_idx - lvl_start) / 4;

    let vm = mask[0] | mask[1] | mask[2];
    if vm == 0 {
        return true;
    }
    let max_iter = 32 - vm.leading_zeros() as usize;

    match BD::BPC {
        BPC::BPC8 => {
            use crate::include::common::bitdepth::BitDepth8;

            let byte_stride = stride.unsigned_abs() as usize;

            let (reach_before, reach_after) = if !is_v {
                (7, (max_iter * 4 - 1) * byte_stride + 16)
            } else {
                (7 * byte_stride, max_iter * 4 - 1 + 16 * byte_stride)
            };

            let buf_pixel_len = dst.data.pixel_len::<BitDepth8>();
            if dst.offset < reach_before || dst.offset.saturating_add(reach_after) > buf_pixel_len {
                return false;
            }

            let start_pixel = dst.offset - reach_before;
            let total_pixels = (reach_before + reach_after).min(buf_pixel_len - start_pixel);
            let mut buf_guard = dst
                .data
                .slice_mut::<BitDepth8, _>((start_pixel.., ..total_pixels));
            let buf: &mut [u8] = &mut *buf_guard;
            let base = reach_before;

            match (is_y, is_v) {
                (true, false) => lpf_h_sb_y_8bpc_inner(
                    token,
                    buf,
                    base,
                    stride as isize,
                    mask,
                    lvl_slice,
                    lvl_base,
                    lvl_byte_idx,
                    b4_stride,
                    lut,
                    w,
                    bitdepth_max,
                ),
                (true, true) => lpf_v_sb_y_8bpc_inner(
                    token,
                    buf,
                    base,
                    stride as isize,
                    mask,
                    lvl_slice,
                    lvl_base,
                    lvl_byte_idx,
                    b4_stride,
                    lut,
                    w,
                    bitdepth_max,
                ),
                (false, false) => lpf_h_sb_uv_8bpc_inner(
                    token,
                    buf,
                    base,
                    stride as isize,
                    mask,
                    lvl_slice,
                    lvl_base,
                    lvl_byte_idx,
                    b4_stride,
                    lut,
                    w,
                    bitdepth_max,
                ),
                (false, true) => lpf_v_sb_uv_8bpc_inner(
                    token,
                    buf,
                    base,
                    stride as isize,
                    mask,
                    lvl_slice,
                    lvl_base,
                    lvl_byte_idx,
                    b4_stride,
                    lut,
                    w,
                    bitdepth_max,
                ),
            }
        }
        BPC::BPC16 => {
            use crate::include::common::bitdepth::BitDepth16;

            let u16_stride = (stride / 2).unsigned_abs() as usize;

            let (reach_before, reach_after) = if !is_v {
                (7, (max_iter * 4 - 1) * u16_stride + 16)
            } else {
                (7 * u16_stride, max_iter * 4 - 1 + 16 * u16_stride)
            };

            let buf_pixel_len = dst.data.pixel_len::<BitDepth16>();
            if dst.offset < reach_before || dst.offset.saturating_add(reach_after) > buf_pixel_len {
                return false;
            }

            let start_pixel = dst.offset - reach_before;
            let total_pixels = (reach_before + reach_after).min(buf_pixel_len - start_pixel);
            let mut buf_guard = dst
                .data
                .slice_mut::<BitDepth16, _>((start_pixel.., ..total_pixels));
            let buf: &mut [u16] = &mut *buf_guard;
            let base = reach_before;

            match (is_y, is_v) {
                (true, false) => lpf_h_sb_y_16bpc_inner(
                    buf,
                    base,
                    stride as isize / 2,
                    mask,
                    lvl_slice,
                    lvl_base,
                    lvl_byte_idx,
                    b4_stride,
                    lut,
                    w,
                    bitdepth_max,
                ),
                (true, true) => lpf_v_sb_y_16bpc_inner(
                    buf,
                    base,
                    stride as isize / 2,
                    mask,
                    lvl_slice,
                    lvl_base,
                    lvl_byte_idx,
                    b4_stride,
                    lut,
                    w,
                    bitdepth_max,
                ),
                (false, false) => lpf_h_sb_uv_16bpc_inner(
                    buf,
                    base,
                    stride as isize / 2,
                    mask,
                    lvl_slice,
                    lvl_base,
                    lvl_byte_idx,
                    b4_stride,
                    lut,
                    w,
                    bitdepth_max,
                ),
                (false, true) => lpf_v_sb_uv_16bpc_inner(
                    buf,
                    base,
                    stride as isize / 2,
                    mask,
                    lvl_slice,
                    lvl_base,
                    lvl_byte_idx,
                    b4_stride,
                    lut,
                    w,
                    bitdepth_max,
                ),
            }
        }
    }
    true
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iclip_diff() {
        assert_eq!(iclip_diff(100, 0), 100);
        assert_eq!(iclip_diff(-100, 0), -100);
        assert_eq!(iclip_diff(200, 0), 127);
        assert_eq!(iclip_diff(-200, 0), -128);
    }
}
