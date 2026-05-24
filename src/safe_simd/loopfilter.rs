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
use archmage::{Desktop64, SimdToken, X64V2Token, arcane};
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
#[cfg(any(target_arch = "x86_64", target_arch = "wasm32"))]
fn loop_filter_4_8bpc(
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
    #[cfg(target_arch = "x86_64")]
    if stridea == 1 && bitdepth_max == 255 {
        if let Some(token) = X64V2Token::summon() {
            match wd {
                4 => {
                    loop_filter_4_8bpc_narrow_simd_v(token, buf, base, e, i, h, strideb);
                    return;
                }
                6 => {
                    loop_filter_4_8bpc_wd6_simd_v(token, buf, base, e, i, h, strideb);
                    return;
                }
                _ => {}
            }
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
    _token: X64V2Token,
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
    let val_ee = _mm_add_epi32(
        _mm_slli_epi32::<1>(abs_p0q0),
        _mm_srli_epi32::<1>(abs_p1q1),
    );
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
    _token: X64V2Token,
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
        let bytes = [
            buf[start],
            buf[start + 1],
            buf[start + 2],
            buf[start + 3],
        ];
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
    let val = _mm_add_epi32(
        _mm_slli_epi32::<1>(abs_p0q0),
        _mm_srli_epi32::<1>(abs_p1q1),
    );
    let m_val = not_gt(val, e_v);
    let fm_mask = _mm_and_si128(_mm_and_si128(m_p1p0, m_q1q0), m_val);

    let hev_mask = _mm_or_si128(_mm_cmpgt_epi32(abs_p1p0, h_v), _mm_cmpgt_epi32(abs_q1q0, h_v));

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
        let u16x4 = _mm_packus_epi32(v, v);    // low 4 u16
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
fn lpf_h_sb_y_8bpc_inner(
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
fn lpf_v_sb_y_8bpc_inner(
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
fn lpf_h_sb_uv_8bpc_inner(
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
fn lpf_v_sb_uv_8bpc_inner(
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
    lpf_h_sb_y_8bpc_inner(
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
    lpf_v_sb_y_8bpc_inner(
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
    lpf_h_sb_uv_8bpc_inner(
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
    lpf_v_sb_uv_8bpc_inner(
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

    let Some(_token) = crate::src::cpu::summon_avx2() else {
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
