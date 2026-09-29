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
use crate::src::safe_simd::partial_simd::{mm_loadl_epi64, mm_storel_epi64};
#[cfg(target_arch = "x86_64")]
use crate::src::safe_simd::pixel_access::{loadi32, loadi64, loadu_128, storei64, storeu_128};
#[cfg(target_arch = "x86_64")]
use archmage::{Desktop64, Server64, SimdToken, arcane, rite};
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
            6 => {
                loop_filter_4_8bpc_wd6_simd_h(_token, buf, base, e, i, h, stridea);
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
        loadi32!(&buf[start..start + 4])
    };

    let Some(out) = lf_wd6_8bpc_core_u8(
        [
            load4(-3),
            load4(-2),
            load4(-1),
            load4(0),
            load4(1),
            load4(2),
        ],
        e,
        i,
        h,
        _mm_cvtsi32_si128(-1), // 4 live lanes
    ) else {
        return;
    };

    let pack4 = |v: __m128i| -> i32 { _mm_cvtsi128_si32(_mm_packus_epi16(v, v)) };
    let store4 = |buf: &mut [u8], packed: i32, off: isize| {
        let start = signed_idx(base, strideb * off);
        let bytes = packed.to_le_bytes();
        buf[start..start + 4].copy_from_slice(&bytes);
    };
    store4(buf, pack4(out[0]), -2);
    store4(buf, pack4(out[1]), -1);
    store4(buf, pack4(out[2]), 0);
    store4(buf, pack4(out[3]), 1);
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
    // 4-byte row loads keep the pixels packed as u8 lanes 0-3;
    // lanes 4-7 stay zero and are masked out of the early-out check.
    let load4 = |off: isize| -> __m128i {
        let start = signed_idx(base, strideb * off);
        loadi32!(&buf[start..start + 4])
    };

    let taps = [
        load4(-4),
        load4(-3),
        load4(-2),
        load4(-1),
        load4(0),
        load4(1),
        load4(2),
        load4(3),
    ];
    let Some([final_p2, final_p1, final_p0, final_q0, final_q1, final_q2]) =
        lf_wd8_8bpc_core_u8(taps, e, i, h, 0x0F)
    else {
        return;
    };

    // fm_mask=0 keeps original via the blend; always storing final_* is
    // correct since they select originals at unfiltered lanes.
    let store4 = |buf: &mut [u8], v: __m128i, off: isize| {
        let start = signed_idx(base, strideb * off);
        buf[start..start + 4].copy_from_slice(&_mm_cvtsi128_si32(v).to_le_bytes());
    };
    store4(buf, final_p2, -3);
    store4(buf, final_p1, -2);
    store4(buf, final_p0, -1);
    store4(buf, final_q0, 0);
    store4(buf, final_q1, 1);
    store4(buf, final_q2, 2);
}

// ============================================================================
// SIMD wd=8 V-FILTER x8 widen — processes 8 adjacent edges with same level
// ============================================================================

/// u8-lane wd=8 filter core shared by the V and H 8-column kernels.
/// `taps` = [p3,p2,p1,p0,q0,q1,q2,q3], each holding 8 pixel lanes in the
/// low 8 bytes (upper lanes are don't-care). Returns `None` when the
/// filter mask is all-false, in which case no output position changes —
/// identical to storing the originals, so callers may skip stores.
///
/// Mirrors dav1d's FILTER macro: pixels stay packed as u8 through the
/// mask computation (`subs_epu8`+`cmpeq` for unsigned compares), move to
/// the i8 domain (^0x80) for the saturating narrow filter, and the 8-tap
/// outputs are `maddubs` pairs accumulated in i16 — never widened to i32.
#[cfg(target_arch = "x86_64")]
#[rite(v3)]
fn lf_wd8_8bpc_core_u8(
    taps: [__m128i; 8],
    e: i32,
    i: i32,
    h: i32,
    live_lanes: i32,
) -> Option<[__m128i; 6]> {
    let [p3_v, p2_v, p1_v, p0_v, q0_v, q1_v, q2_v, q3_v] = taps;

    let zero = _mm_setzero_si128();
    let all_ones = _mm_set1_epi8(-1);
    let x80 = _mm_set1_epi8(-128);

    let i_v = _mm_set1_epi8(i as i8);
    let h_v = _mm_set1_epi8(h as i8);
    let f_v = _mm_set1_epi8(1);

    let absu = |a: __m128i, b: __m128i| _mm_or_si128(_mm_subs_epu8(a, b), _mm_subs_epu8(b, a));
    // mask 0xFF where a <= t (unsigned): saturating sub hits 0 exactly then
    let le_u8 = |a: __m128i, t: __m128i| _mm_cmpeq_epi8(_mm_subs_epu8(a, t), zero);
    let gt_u8 = |a: __m128i, t: __m128i| _mm_andnot_si128(le_u8(a, t), all_ones);

    let abs_p1p0 = absu(p1_v, p0_v);
    let abs_q1q0 = absu(q1_v, q0_v);
    let abs_p0q0 = absu(p0_v, q0_v);
    let abs_p1q1 = absu(p1_v, q1_v);
    let abs_p2p1 = absu(p2_v, p1_v);
    let abs_q2q1 = absu(q2_v, q1_v);
    let abs_p3p2 = absu(p3_v, p2_v);
    let abs_q3q2 = absu(q3_v, q2_v);

    // E term: 2*|p0-q0| + (|p1-q1|>>1) <= e — must be EXACT: at e == 255
    // a true sum >= 256 still rejects, so the saturating-u8 form dav1d
    // uses (paddusb) would diverge from the scalar reference. Widen just
    // this one term to u16 lanes.
    let a16 = _mm_unpacklo_epi8(abs_p0q0, zero);
    let b16 = _mm_unpacklo_epi8(abs_p1q1, zero);
    let sum16 = _mm_add_epi16(_mm_add_epi16(a16, a16), _mm_srli_epi16::<1>(b16));
    let le_e16 = _mm_andnot_si128(
        _mm_cmpgt_epi16(sum16, _mm_set1_epi16(e as i16)),
        _mm_set1_epi16(-1),
    );
    // Narrow the u16 lane mask to u8: 0xFFFF>>8 = 0xFF, 0 → 0
    let le_e8 = _mm_packus_epi16(_mm_srli_epi16::<8>(le_e16), _mm_srli_epi16::<8>(le_e16));

    let fm_mask = _mm_and_si128(
        _mm_and_si128(le_u8(abs_p1p0, i_v), le_u8(abs_q1q0, i_v)),
        _mm_and_si128(
            _mm_and_si128(le_e8, le_u8(abs_p2p1, i_v)),
            _mm_and_si128(
                _mm_and_si128(le_u8(abs_q2q1, i_v), le_u8(abs_p3p2, i_v)),
                le_u8(abs_q3q2, i_v),
            ),
        ),
    );

    // Early-out: an all-false filter mask selects the original
    // pixels at every lane — the flat-mask, filter arithmetic,
    // blends and stores below are all dead work (~85%/64% of
    // calls on the 4K-t8 census). Skipping them is bit-identical.
    // `live_lanes` masks off dead upper lanes (always-masked zeros).
    if _mm_movemask_epi8(fm_mask) & live_lanes == 0 {
        return None;
    }

    let abs_p2p0 = absu(p2_v, p0_v);
    let abs_q2q0 = absu(q2_v, q0_v);
    let abs_p3p0 = absu(p3_v, p0_v);
    let abs_q3q0 = absu(q3_v, q0_v);
    let flat_mask = _mm_and_si128(
        _mm_and_si128(le_u8(abs_p2p0, f_v), le_u8(abs_p1p0, f_v)),
        _mm_and_si128(
            _mm_and_si128(le_u8(abs_q1q0, f_v), le_u8(abs_q2q0, f_v)),
            _mm_and_si128(le_u8(abs_p3p0, f_v), le_u8(abs_q3q0, f_v)),
        ),
    );

    // 8-tap outputs: interleave tap pairs once, then one maddubs per
    // coefficient pair (u8 pixels × i8 coeffs → i16), i16 accumulate,
    // mulhrs by 4096 == (x+4)>>3. Max sum is 8*255 = 2040 < i16 max.
    let pr_p3p2 = _mm_unpacklo_epi8(p3_v, p2_v);
    let pr_p1p0 = _mm_unpacklo_epi8(p1_v, p0_v);
    let pr_q0q1 = _mm_unpacklo_epi8(q0_v, q1_v);
    let pr_q2q3 = _mm_unpacklo_epi8(q2_v, q3_v);
    let mad = |pair: __m128i, ca: u16, cb: u16| {
        _mm_maddubs_epi16(pair, _mm_set1_epi16(((cb << 8) | ca) as i16))
    };
    let pw_4096 = _mm_set1_epi16(4096);
    let shr3 = |v: __m128i| _mm_mulhrs_epi16(v, pw_4096);
    let pack = |v: __m128i| _mm_packus_epi16(v, v);

    let s_m3 = _mm_add_epi16(
        _mm_add_epi16(mad(pr_p3p2, 3, 2), mad(pr_p1p0, 1, 1)),
        mad(pr_q0q1, 1, 0),
    );
    let s_m2 = _mm_add_epi16(
        _mm_add_epi16(mad(pr_p3p2, 2, 1), mad(pr_p1p0, 2, 1)),
        mad(pr_q0q1, 1, 1),
    );
    let s_m1 = _mm_add_epi16(
        _mm_add_epi16(mad(pr_p3p2, 1, 1), mad(pr_p1p0, 1, 2)),
        _mm_add_epi16(mad(pr_q0q1, 1, 1), mad(pr_q2q3, 1, 0)),
    );
    let s_0 = _mm_add_epi16(
        _mm_add_epi16(mad(pr_p3p2, 0, 1), mad(pr_p1p0, 1, 1)),
        _mm_add_epi16(mad(pr_q0q1, 2, 1), mad(pr_q2q3, 1, 1)),
    );
    let s_1 = _mm_add_epi16(
        mad(pr_p1p0, 1, 1),
        _mm_add_epi16(mad(pr_q0q1, 1, 2), mad(pr_q2q3, 1, 2)),
    );
    let s_2 = _mm_add_epi16(
        mad(pr_p1p0, 0, 1),
        _mm_add_epi16(mad(pr_q0q1, 1, 1), mad(pr_q2q3, 2, 3)),
    );

    let out_m3 = pack(shr3(s_m3));
    let out_m2 = pack(shr3(s_m2));
    let out_m1 = pack(shr3(s_m1));
    let out_0 = pack(shr3(s_0));
    let out_1 = pack(shr3(s_1));
    let out_2 = pack(shr3(s_2));

    // Narrow filter in the signed (xor 0x80) domain — dav1d's exact
    // sequence: saturating i8 ops are iclip_diff by construction.
    let p1s = _mm_xor_si128(p1_v, x80);
    let q1s = _mm_xor_si128(q1_v, x80);
    let p0s = _mm_xor_si128(p0_v, x80);
    let q0s = _mm_xor_si128(q0_v, x80);

    let hev_mask = _mm_or_si128(gt_u8(abs_p1p0, h_v), gt_u8(abs_q1q0, h_v));

    let d_p1q1 = _mm_subs_epi8(p1s, q1s);
    let d_q0p0 = _mm_subs_epi8(q0s, p0s);
    let f = _mm_and_si128(d_p1q1, hev_mask);
    let f = _mm_adds_epi8(f, d_q0p0);
    let f = _mm_adds_epi8(f, d_q0p0);
    let f = _mm_adds_epi8(f, d_q0p0);
    let f = _mm_and_si128(f, fm_mask);

    // signed >>3 on i8: &0xf8 clears the low bits so a 16-bit logical
    // shift stays byte-local, then xor/sub 0x10 restores the sign.
    let shr3_i8 = |v: __m128i| {
        let s = _mm_srli_epi16::<3>(_mm_and_si128(v, _mm_set1_epi8(-8)));
        _mm_sub_epi8(_mm_xor_si128(s, _mm_set1_epi8(0x10)), _mm_set1_epi8(0x10))
    };
    let f2 = shr3_i8(_mm_adds_epi8(f, _mm_set1_epi8(3)));
    let f1 = shr3_i8(_mm_adds_epi8(f, _mm_set1_epi8(4)));

    let p0n = _mm_xor_si128(_mm_adds_epi8(p0s, f2), x80);
    let q0n = _mm_xor_si128(_mm_subs_epi8(q0s, f1), x80);
    // (f1+1)>>1 in the u8 domain via avg, restricted to !hev lanes.
    let f3 = _mm_sub_epi8(
        _mm_avg_epu8(_mm_xor_si128(f1, x80), zero),
        _mm_set1_epi8(0x40),
    );
    let f3 = _mm_andnot_si128(hev_mask, f3);
    let p1n = _mm_xor_si128(_mm_adds_epi8(p1s, f3), x80);
    let q1n = _mm_xor_si128(_mm_subs_epi8(q1s, f3), x80);

    // flat ? flat8 : narrow, then fm ? filtered : original
    let sel_p2 = _mm_blendv_epi8(p2_v, out_m3, flat_mask);
    let sel_p1 = _mm_blendv_epi8(p1n, out_m2, flat_mask);
    let sel_p0 = _mm_blendv_epi8(p0n, out_m1, flat_mask);
    let sel_q0 = _mm_blendv_epi8(q0n, out_0, flat_mask);
    let sel_q1 = _mm_blendv_epi8(q1n, out_1, flat_mask);
    let sel_q2 = _mm_blendv_epi8(q2_v, out_2, flat_mask);

    let final_p2 = _mm_blendv_epi8(p2_v, sel_p2, fm_mask);
    let final_p1 = _mm_blendv_epi8(p1_v, sel_p1, fm_mask);
    let final_p0 = _mm_blendv_epi8(p0_v, sel_p0, fm_mask);
    let final_q0 = _mm_blendv_epi8(q0_v, sel_q0, fm_mask);
    let final_q1 = _mm_blendv_epi8(q1_v, sel_q1, fm_mask);
    let final_q2 = _mm_blendv_epi8(q2_v, sel_q2, fm_mask);

    Some([final_p2, final_p1, final_p0, final_q0, final_q1, final_q2])
}

/// SIMD wd=8 loop filter for 8bpc V-FILTER direction, **8-column variant**.
/// Used by the outer dispatcher when two adjacent edges with vmask[1] set
/// share the same `l` (and therefore the same e/i/h derived parameters).
#[cfg(target_arch = "x86_64")]
#[arcane]
fn loop_filter_4_8bpc_wd8_simd_v_x8(
    _token: Desktop64,
    buf: &mut [u8],
    base: usize,
    e: i32,
    i: i32,
    h: i32,
    strideb: isize,
) {
    let load8 = |off: isize| -> __m128i {
        let start = signed_idx(base, strideb * off);
        loadi64!(&buf[start..start + 8])
    };

    let taps = [
        load8(-4),
        load8(-3),
        load8(-2),
        load8(-1),
        load8(0),
        load8(1),
        load8(2),
        load8(3),
    ];
    let Some([final_p2, final_p1, final_p0, final_q0, final_q1, final_q2]) =
        lf_wd8_8bpc_core_u8(taps, e, i, h, 0xFF)
    else {
        return;
    };

    let store8 = |buf: &mut [u8], v: __m128i, off: isize| {
        let start = signed_idx(base, strideb * off);
        storei64!(&mut buf[start..start + 8], v);
    };
    store8(buf, final_p2, -3);
    store8(buf, final_p1, -2);
    store8(buf, final_p0, -1);
    store8(buf, final_q0, 0);
    store8(buf, final_q1, 1);
    store8(buf, final_q2, 2);
}

// ============================================================================
// SIMD wd=16 V-FILTER x8 widen — processes 8 adjacent edges with same level
// ============================================================================

/// SIMD wd=16 loop filter for 8bpc V-FILTER direction, **8-column variant**.
/// Processes two adjacent wd=16 edges with the same level in the 8 live u8
/// lanes of a single XMM register — same `lf_wd16_8bpc_core_u8` as the
/// 4-lane kernel, just a wider live mask.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn loop_filter_4_8bpc_wd16_simd_v_x8(
    _token: Desktop64,
    buf: &mut [u8],
    base: usize,
    e: i32,
    i: i32,
    h: i32,
    strideb: isize,
) {
    let load8 = |off: isize| -> __m128i {
        let start = signed_idx(base, strideb * off);
        loadi64!(&buf[start..start + 8])
    };

    let Some(out) = lf_wd16_8bpc_core_u8(
        [
            load8(-7),
            load8(-6),
            load8(-5),
            load8(-4),
            load8(-3),
            load8(-2),
            load8(-1),
            load8(0),
            load8(1),
            load8(2),
            load8(3),
            load8(4),
            load8(5),
            load8(6),
        ],
        e,
        i,
        h,
        _mm_cvtsi64_si128(-1), // 8 live lanes
    ) else {
        return;
    };

    let store8 = |buf: &mut [u8], v: __m128i, off: isize| {
        let start = signed_idx(base, strideb * off);
        let packed = _mm_cvtsi128_si64(_mm_packus_epi16(v, v)).to_le_bytes();
        buf[start..start + 8].copy_from_slice(&packed);
    };
    for (k, v) in out.iter().enumerate() {
        store8(buf, *v, k as isize - 6);
    }
}

// ============================================================================
// SIMD wd=16 V-FILTER x16 widen (AVX-512) — processes 16 adjacent edges
// ============================================================================

/// SIMD wd=16 loop filter for 8bpc V-FILTER direction, **16-column variant**.
/// Used by the outer dispatcher when four adjacent wd=16 edges (16 columns)
/// share the same filter level. Bit-exact with the x8 / x4 / scalar paths.
///
/// u8-lane form: the 14 taps load as `__m128i` (16 pixel positions per lane
/// group) and the predicate masks stay in the u8 domain via AVX-512VL
/// `epu8` compares producing `__mmask16` directly. The E term is computed in
/// u16 lanes because `2*|p0-q0| + |p1-q1|>>1 <= e` must not saturate at
/// e == 255. Filter accumulators are `__m256i` i16 — every output weight
/// sums to <= 16 so the numerator stays under 255*16+8 < 32767 — and all
/// blends use `_mm256_mask_blend_epi16`. Outputs pack once at the end with
/// `packus_epi16` (signed saturate), which is exactly the scalar clip.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn loop_filter_4_8bpc_wd16_simd_v_x16(
    _token: Server64,
    buf: &mut [u8],
    base: usize,
    e: i32,
    i: i32,
    h: i32,
    strideb: isize,
) {
    let load16 = |off: isize| -> __m128i {
        let start = signed_idx(base, strideb * off);
        loadu_128!(&buf[start..start + 16], [u8; 16])
    };

    // u8 taps: predicate masks only.
    let p6_u8 = load16(-7);
    let p5_u8 = load16(-6);
    let p4_u8 = load16(-5);
    let p3_u8 = load16(-4);
    let p2_u8 = load16(-3);
    let p1_u8 = load16(-2);
    let p0_u8 = load16(-1);
    let q0_u8 = load16(0);
    let q1_u8 = load16(1);
    let q2_u8 = load16(2);
    let q3_u8 = load16(3);
    let q4_u8 = load16(4);
    let q5_u8 = load16(5);
    let q6_u8 = load16(6);

    let i_v = _mm_set1_epi8(i as i8);
    let h_v = _mm_set1_epi8(h as i8);
    let f_v = _mm_set1_epi8(1);

    let absu = |a: __m128i, b: __m128i| _mm_or_si128(_mm_subs_epu8(a, b), _mm_subs_epu8(b, a));

    let abs_p1p0 = absu(p1_u8, p0_u8);
    let abs_q1q0 = absu(q1_u8, q0_u8);
    let abs_p0q0 = absu(p0_u8, q0_u8);
    let abs_p1q1 = absu(p1_u8, q1_u8);
    let abs_p2p1 = absu(p2_u8, p1_u8);
    let abs_q2q1 = absu(q2_u8, q1_u8);
    let abs_p3p2 = absu(p3_u8, p2_u8);
    let abs_q3q2 = absu(q3_u8, q2_u8);

    // fm: all the <=i terms stay in the u8 domain; the E term widens to u16
    // because it must reject true sums >= 256 (saturating u8 would pass them
    // at e == 255 where the scalar reference does not).
    let a16 = _mm256_cvtepu8_epi16(abs_p0q0);
    let b16 = _mm256_cvtepu8_epi16(abs_p1q1);
    let sum16 = _mm256_add_epi16(_mm256_add_epi16(a16, a16), _mm256_srli_epi16::<1>(b16));
    let m_val = _mm256_cmple_epi16_mask(sum16, _mm256_set1_epi16(e as i16));

    let fm_mask = _mm_cmple_epu8_mask(abs_p1p0, i_v)
        & _mm_cmple_epu8_mask(abs_q1q0, i_v)
        & m_val
        & _mm_cmple_epu8_mask(abs_p2p1, i_v)
        & _mm_cmple_epu8_mask(abs_q2q1, i_v)
        & _mm_cmple_epu8_mask(abs_p3p2, i_v)
        & _mm_cmple_epu8_mask(abs_q3q2, i_v);

    let flat8out_mask = _mm_cmple_epu8_mask(absu(p6_u8, p0_u8), f_v)
        & _mm_cmple_epu8_mask(absu(p5_u8, p0_u8), f_v)
        & _mm_cmple_epu8_mask(absu(p4_u8, p0_u8), f_v)
        & _mm_cmple_epu8_mask(absu(q4_u8, q0_u8), f_v)
        & _mm_cmple_epu8_mask(absu(q5_u8, q0_u8), f_v)
        & _mm_cmple_epu8_mask(absu(q6_u8, q0_u8), f_v);

    let flat8in_mask = _mm_cmple_epu8_mask(absu(p2_u8, p0_u8), f_v)
        & _mm_cmple_epu8_mask(abs_p1p0, f_v)
        & _mm_cmple_epu8_mask(abs_q1q0, f_v)
        & _mm_cmple_epu8_mask(absu(q2_u8, q0_u8), f_v)
        & _mm_cmple_epu8_mask(absu(p3_u8, p0_u8), f_v)
        & _mm_cmple_epu8_mask(absu(q3_u8, q0_u8), f_v);

    let hev_mask = _mm_cmpgt_epu8_mask(abs_p1p0, h_v) | _mm_cmpgt_epu8_mask(abs_q1q0, h_v);

    // i16 taps: filter accumulators and original-pixel blends.
    let w16 = |v: __m128i| _mm256_cvtepu8_epi16(v);
    let p6_v = w16(p6_u8);
    let p5_v = w16(p5_u8);
    let p4_v = w16(p4_u8);
    let p3_v = w16(p3_u8);
    let p2_v = w16(p2_u8);
    let p1_v = w16(p1_u8);
    let p0_v = w16(p0_u8);
    let q0_v = w16(q0_u8);
    let q1_v = w16(q1_u8);
    let q2_v = w16(q2_u8);
    let q3_v = w16(q3_u8);
    let q4_v = w16(q4_u8);
    let q5_v = w16(q5_u8);
    let q6_v = w16(q6_u8);

    let dbl = |v: __m256i| _mm256_slli_epi16::<1>(v);
    let add = |a: __m256i, b: __m256i| _mm256_add_epi16(a, b);
    let add3 = |a: __m256i, b: __m256i, c: __m256i| add(add(a, b), c);
    let add4 = |a: __m256i, b: __m256i, c: __m256i, d: __m256i| add(add(a, b), add(c, d));
    let c4 = _mm256_set1_epi16(4);
    let c8 = _mm256_set1_epi16(8);

    // 14-tap outputs (positions -6..5)
    let p6_5 = add(add(dbl(p6_v), dbl(p6_v)), p6_v);
    let q6_5 = add(add(dbl(q6_v), dbl(q6_v)), q6_v);

    let mut s = add(p6_5, add(dbl(p6_v), dbl(p5_v)));
    s = add(s, dbl(p4_v));
    s = add(s, add4(p3_v, p2_v, p1_v, p0_v));
    s = add(s, add(q0_v, c8));
    let out_m6 = _mm256_srai_epi16::<4>(s);

    let mut s = add(p6_5, add(dbl(p5_v), dbl(p4_v)));
    s = add(s, dbl(p3_v));
    s = add(s, add4(p2_v, p1_v, p0_v, q0_v));
    s = add(s, add(q1_v, c8));
    let out_m5 = _mm256_srai_epi16::<4>(s);

    let p6_4 = add(dbl(p6_v), dbl(p6_v));
    let mut s = add(p6_4, p5_v);
    s = add(s, add(dbl(p4_v), dbl(p3_v)));
    s = add(s, dbl(p2_v));
    s = add(s, add4(p1_v, p0_v, q0_v, q1_v));
    s = add(s, add(q2_v, c8));
    let out_m4 = _mm256_srai_epi16::<4>(s);

    let p6_3 = add(dbl(p6_v), p6_v);
    let mut s = add(p6_3, add(p5_v, p4_v));
    s = add(s, add(dbl(p3_v), dbl(p2_v)));
    s = add(s, dbl(p1_v));
    s = add(s, add4(p0_v, q0_v, q1_v, q2_v));
    s = add(s, add(q3_v, c8));
    let out_m3 = _mm256_srai_epi16::<4>(s);

    let mut s = add(dbl(p6_v), p5_v);
    s = add(s, add(p4_v, p3_v));
    s = add(s, add(dbl(p2_v), dbl(p1_v)));
    s = add(s, dbl(p0_v));
    s = add(s, add4(q0_v, q1_v, q2_v, q3_v));
    s = add(s, add(q4_v, c8));
    let out_m2 = _mm256_srai_epi16::<4>(s);

    let mut s = add(p6_v, p5_v);
    s = add(s, add(p4_v, p3_v));
    s = add(s, p2_v);
    s = add(s, add(dbl(p1_v), dbl(p0_v)));
    s = add(s, dbl(q0_v));
    s = add(s, add4(q1_v, q2_v, q3_v, q4_v));
    s = add(s, add(q5_v, c8));
    let out_m1 = _mm256_srai_epi16::<4>(s);

    let mut s = add(p5_v, p4_v);
    s = add(s, add(p3_v, p2_v));
    s = add(s, p1_v);
    s = add(s, add(dbl(p0_v), dbl(q0_v)));
    s = add(s, dbl(q1_v));
    s = add(s, add4(q2_v, q3_v, q4_v, q5_v));
    s = add(s, add(q6_v, c8));
    let out_0 = _mm256_srai_epi16::<4>(s);

    let mut s = add(p4_v, p3_v);
    s = add(s, add(p2_v, p1_v));
    s = add(s, p0_v);
    s = add(s, add(dbl(q0_v), dbl(q1_v)));
    s = add(s, dbl(q2_v));
    s = add(s, add4(q3_v, q4_v, q5_v, q6_v));
    s = add(s, add(q6_v, c8));
    let out_1 = _mm256_srai_epi16::<4>(s);

    let mut s = add(p3_v, p2_v);
    s = add(s, add(p1_v, p0_v));
    s = add(s, q0_v);
    s = add(s, add(dbl(q1_v), dbl(q2_v)));
    s = add(s, dbl(q3_v));
    let q6_3 = add(dbl(q6_v), q6_v);
    s = add(s, add3(q4_v, q5_v, q6_3));
    s = add(s, c8);
    let out_2 = _mm256_srai_epi16::<4>(s);

    let q6_4 = add(dbl(q6_v), dbl(q6_v));
    let mut s = add(p2_v, p1_v);
    s = add(s, add(p0_v, q0_v));
    s = add(s, q1_v);
    s = add(s, add(dbl(q2_v), dbl(q3_v)));
    s = add(s, dbl(q4_v));
    s = add(s, add(q5_v, q6_4));
    s = add(s, c8);
    let out_3 = _mm256_srai_epi16::<4>(s);

    let mut s = add(p1_v, p0_v);
    s = add(s, add(q0_v, q1_v));
    s = add(s, q2_v);
    s = add(s, add(dbl(q3_v), dbl(q4_v)));
    s = add(s, dbl(q5_v));
    s = add(s, q6_5);
    s = add(s, c8);
    let out_4 = _mm256_srai_epi16::<4>(s);

    let q6_7 = add(q6_5, dbl(q6_v));
    let mut s = add(p0_v, q0_v);
    s = add(s, add(q1_v, q2_v));
    s = add(s, q3_v);
    s = add(s, add(dbl(q4_v), dbl(q5_v)));
    s = add(s, q6_7);
    s = add(s, c8);
    let out_5 = _mm256_srai_epi16::<4>(s);

    // 8-tap outputs
    let triple = |v: __m256i| add(dbl(v), v);
    let out8_m3 = _mm256_srai_epi16::<3>(add(
        add4(triple(p3_v), dbl(p2_v), p1_v, p0_v),
        add(q0_v, c4),
    ));
    let out8_m2 = _mm256_srai_epi16::<3>(add(
        add4(dbl(p3_v), p2_v, dbl(p1_v), p0_v),
        add3(q0_v, q1_v, c4),
    ));
    let out8_m1 = _mm256_srai_epi16::<3>(add(
        add4(p3_v, p2_v, p1_v, dbl(p0_v)),
        add4(q0_v, q1_v, q2_v, c4),
    ));
    let out8_0 = _mm256_srai_epi16::<3>(add(
        add4(p2_v, p1_v, p0_v, dbl(q0_v)),
        add4(q1_v, q2_v, q3_v, c4),
    ));
    let out8_1 = _mm256_srai_epi16::<3>(add(
        add4(p1_v, p0_v, q0_v, dbl(q1_v)),
        add4(q2_v, q3_v, q3_v, c4),
    ));
    let out8_2 = _mm256_srai_epi16::<3>(add(
        add4(p0_v, q0_v, q1_v, dbl(q2_v)),
        add4(q3_v, q3_v, q3_v, c4),
    ));

    // Narrow filter (4-tap) — i16 domain, filter clipped to i8 range.
    let neg128 = _mm256_set1_epi16(-128);
    let pos127 = _mm256_set1_epi16(127);
    let iclip = |v: __m256i| _mm256_min_epi16(_mm256_max_epi16(v, neg128), pos127);
    let diff_q0p0 = _mm256_sub_epi16(q0_v, p0_v);
    let three_d = _mm256_add_epi16(_mm256_slli_epi16::<1>(diff_q0p0), diff_q0p0);
    let diff_p1q1 = _mm256_sub_epi16(p1_v, q1_v);
    let f_hev = iclip(_mm256_add_epi16(three_d, iclip(diff_p1q1)));
    let f_no = iclip(three_d);
    let c3i = _mm256_set1_epi16(3);
    let one = _mm256_set1_epi16(1);
    let f1_hev = _mm256_srai_epi16::<3>(_mm256_min_epi16(_mm256_add_epi16(f_hev, c4), pos127));
    let f2_hev = _mm256_srai_epi16::<3>(_mm256_min_epi16(_mm256_add_epi16(f_hev, c3i), pos127));
    let f1_no = _mm256_srai_epi16::<3>(_mm256_min_epi16(_mm256_add_epi16(f_no, c4), pos127));
    let f2_no = _mm256_srai_epi16::<3>(_mm256_min_epi16(_mm256_add_epi16(f_no, c3i), pos127));
    let f_extra = _mm256_srai_epi16::<1>(_mm256_add_epi16(f1_no, one));
    let p0_hev = _mm256_add_epi16(p0_v, f2_hev);
    let q0_hev = _mm256_sub_epi16(q0_v, f1_hev);
    let p0_no = _mm256_add_epi16(p0_v, f2_no);
    let q0_no = _mm256_sub_epi16(q0_v, f1_no);
    let p1_no = _mm256_add_epi16(p1_v, f_extra);
    let q1_no = _mm256_sub_epi16(q1_v, f_extra);

    let blendv =
        |a: __m256i, b: __m256i, k: __mmask16| -> __m256i { _mm256_mask_blend_epi16(k, a, b) };

    let narrow_p1 = blendv(p1_no, p1_v, hev_mask);
    let narrow_p0 = blendv(p0_no, p0_hev, hev_mask);
    let narrow_q0 = blendv(q0_no, q0_hev, hev_mask);
    let narrow_q1 = blendv(q1_no, q1_v, hev_mask);

    let wide_mask = flat8out_mask & flat8in_mask;

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

    // Pack 16 i16 -> 16 u8 per row: packus_epi16 saturates to [0,255], which
    // is exactly the scalar output clip. Per-lane interleave is undone by the
    // qword gather before extracting the low xmm.
    let store16 = |buf: &mut [u8], v: __m256i, off: isize| {
        let packed = _mm256_permute4x64_epi64::<0b11011000>(_mm256_packus_epi16(v, v));
        let start = signed_idx(base, strideb * off);
        storeu_128!(
            &mut buf[start..start + 16],
            [u8; 16],
            _mm256_castsi256_si128(packed)
        );
    };
    store16(buf, final_m6, -6);
    store16(buf, final_m5, -5);
    store16(buf, final_m4, -4);
    store16(buf, final_m3, -3);
    store16(buf, final_m2, -2);
    store16(buf, final_m1, -1);
    store16(buf, final_0, 0);
    store16(buf, final_1, 1);
    store16(buf, final_2, 2);
    store16(buf, final_3, 3);
    store16(buf, final_4, 4);
    store16(buf, final_5, 5);
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
        loadi32!(&buf[start..start + 4])
    };

    let Some(out) = lf_wd16_8bpc_core_u8(
        [
            load4(-7),
            load4(-6),
            load4(-5),
            load4(-4),
            load4(-3),
            load4(-2),
            load4(-1),
            load4(0),
            load4(1),
            load4(2),
            load4(3),
            load4(4),
            load4(5),
            load4(6),
        ],
        e,
        i,
        h,
        _mm_cvtsi32_si128(-1), // 4 live lanes
    ) else {
        return;
    };

    let pack4 = |v: __m128i| -> i32 { _mm_cvtsi128_si32(_mm_packus_epi16(v, v)) };
    let store4 = |buf: &mut [u8], packed: i32, off: isize| {
        let start = signed_idx(base, strideb * off);
        let bytes = packed.to_le_bytes();
        buf[start..start + 4].copy_from_slice(&bytes);
    };
    for (k, v) in out.iter().enumerate() {
        store4(buf, pack4(*v), k as isize - 6);
    }
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
    // Load 4 rows of 4 bytes ([p1,p0,q0,q1] per row) into ONE xmm as dwords,
    // then column-extract the taps with pshufb: out[j] gathers byte j of each
    // row's dword. 4 live u8 lanes per tap vector.
    let load_row = |row: isize| -> __m128i {
        let start = signed_idx(base, row * stridea - 2);
        loadi32!(&buf[start..start + 4])
    };
    let rows = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(load_row(0), load_row(1)),
        _mm_unpacklo_epi32(load_row(2), load_row(3)),
    );
    let tap = |j: i8| -> __m128i {
        _mm_shuffle_epi8(
            rows,
            _mm_setr_epi8(
                j,
                j + 4,
                j + 8,
                j + 12,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
            ),
        )
    };

    let Some(out) = lf_narrow_8bpc_core_u8(
        [tap(0), tap(1), tap(2), tap(3)],
        e,
        i,
        h,
        _mm_cvtsi32_si128(-1), // 4 live lanes
    ) else {
        return;
    };

    // Pack each i16 output to 4 u8 (the [0,255] clip), reassemble the
    // position-major vector [p1|p0|q0|q1] dwords, then pshufb-extract each
    // row back out and store.
    let pack = |v: __m128i| -> i32 { _mm_cvtsi128_si32(_mm_packus_epi16(v, v)) };
    let packed = _mm_setr_epi32(pack(out[0]), pack(out[1]), pack(out[2]), pack(out[3]));
    let row = |k: i8| -> __m128i {
        _mm_shuffle_epi8(
            packed,
            _mm_setr_epi8(
                k,
                k + 4,
                k + 8,
                k + 12,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
            ),
        )
    };
    let store_row = |buf: &mut [u8], v: __m128i, r: isize| {
        let start = signed_idx(base, r * stridea - 2);
        buf[start..start + 4].copy_from_slice(&_mm_cvtsi128_si32(v).to_le_bytes());
    };
    store_row(buf, row(0), 0);
    store_row(buf, row(1), 1);
    store_row(buf, row(2), 2);
    store_row(buf, row(3), 3);
}

/// SIMD narrow 4-tap loop filter for 8bpc H-FILTER direction, **8-row
/// variant**. Two adjacent 4-pixel groups along the edge share the same
/// level, so rows 0..7 load in two xmm packs, column-extract into 8-live-lane
/// tap vectors, run the shared narrow core once, then transpose back and
/// store 4 bytes at -2 per row. Reads and writes the same footprint as two
/// `narrow_simd_h` calls.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn loop_filter_4_8bpc_narrow_simd_h_x8(
    _token: Desktop64,
    buf: &mut [u8],
    base: usize,
    e: i32,
    i: i32,
    h: i32,
    stridea: isize,
) {
    let load_row = |row: isize| -> __m128i {
        let start = signed_idx(base, row * stridea - 2);
        loadi32!(&buf[start..start + 4])
    };
    let lo = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(load_row(0), load_row(1)),
        _mm_unpacklo_epi32(load_row(2), load_row(3)),
    );
    let hi = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(load_row(4), load_row(5)),
        _mm_unpacklo_epi32(load_row(6), load_row(7)),
    );
    // tap j = byte j of each row-dword, lo rows then hi rows → 8 live lanes.
    let tap = |j: i8| -> __m128i {
        let pat = _mm_setr_epi8(
            j,
            j + 4,
            j + 8,
            j + 12,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
        );
        _mm_unpacklo_epi32(_mm_shuffle_epi8(lo, pat), _mm_shuffle_epi8(hi, pat))
    };

    let Some(out) = lf_narrow_8bpc_core_u8(
        [tap(0), tap(1), tap(2), tap(3)],
        e,
        i,
        h,
        _mm_cvtsi64_si128(-1), // 8 live lanes
    ) else {
        return;
    };

    // Pack each i16 output to u8 then byte-transpose 4 positions × 8 rows:
    // unpacklo_epi16(pairs) yields rows 0-3 as dwords, unpackhi rows 4-7.
    let pack = |v: __m128i| -> __m128i { _mm_packus_epi16(v, v) };
    let ab = _mm_unpacklo_epi8(pack(out[0]), pack(out[1]));
    let cd = _mm_unpacklo_epi8(pack(out[2]), pack(out[3]));
    let rows03 = _mm_unpacklo_epi16(ab, cd);
    let rows47 = _mm_unpackhi_epi16(ab, cd);
    let store_row = |buf: &mut [u8], v: i32, r: isize| {
        let start = signed_idx(base, r * stridea - 2);
        buf[start..start + 4].copy_from_slice(&v.to_le_bytes());
    };
    store_row(buf, _mm_extract_epi32::<0>(rows03), 0);
    store_row(buf, _mm_extract_epi32::<1>(rows03), 1);
    store_row(buf, _mm_extract_epi32::<2>(rows03), 2);
    store_row(buf, _mm_extract_epi32::<3>(rows03), 3);
    store_row(buf, _mm_extract_epi32::<0>(rows47), 4);
    store_row(buf, _mm_extract_epi32::<1>(rows47), 5);
    store_row(buf, _mm_extract_epi32::<2>(rows47), 6);
    store_row(buf, _mm_extract_epi32::<3>(rows47), 7);
}

// ============================================================================
// SIMD inner loop filter for the wd=6 H-FILTER case (wd=6, stridea==stride)
// ============================================================================

/// SIMD wd=6 loop filter for 8bpc H-FILTER direction.
/// Per row k, load 4 bytes at offset -3 = [p2, p1, p0, q0] and 4 bytes at
/// offset +1 = [q1, q2, ?, ?]. Transpose each 4-byte half independently to
/// get pixel-position vectors, run the same compute as the v-filter wd=6
/// path, then transpose back to row layout for store. The wd=6 filter only
/// writes positions -2, -1, 0, 1 (4 positions per row).
#[cfg(target_arch = "x86_64")]
#[arcane]
fn loop_filter_4_8bpc_wd6_simd_h(
    _token: Desktop64,
    buf: &mut [u8],
    base: usize,
    e: i32,
    i: i32,
    h: i32,
    stridea: isize,
) {
    // Per row: 4 bytes at -3 = [p2,p1,p0,q0], 2 bytes at +1 = [q1,q2].
    // Build each half into one xmm (one dword per row) and column-extract
    // taps with pshufb. 4 live u8 lanes per tap vector.
    let load_lo = |row: isize| -> __m128i {
        let start = signed_idx(base, row * stridea - 3);
        loadi32!(&buf[start..start + 4])
    };
    let load_hi = |row: isize| -> __m128i {
        // Only 2 bytes survive ([q1,q2]); lanes 2-3 of the row dword are
        // dead. Never touch +3/+4 — at a plane edge those belong to the next
        // row a concurrent tile worker may be writing (#524).
        let start = signed_idx(base, row * stridea + 1);
        _mm_cvtsi32_si128(u16::from_le_bytes(buf[start..start + 2].try_into().unwrap()) as i32)
    };
    let lo = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(load_lo(0), load_lo(1)),
        _mm_unpacklo_epi32(load_lo(2), load_lo(3)),
    );
    let hi = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(load_hi(0), load_hi(1)),
        _mm_unpacklo_epi32(load_hi(2), load_hi(3)),
    );
    let col = |v: __m128i, j: i8| -> __m128i {
        _mm_shuffle_epi8(
            v,
            _mm_setr_epi8(
                j,
                j + 4,
                j + 8,
                j + 12,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
            ),
        )
    };

    let Some(out) = lf_wd6_8bpc_core_u8(
        [
            col(lo, 0), // p2
            col(lo, 1), // p1
            col(lo, 2), // p0
            col(lo, 3), // q0
            col(hi, 0), // q1
            col(hi, 1), // q2
        ],
        e,
        i,
        h,
        _mm_cvtsi32_si128(-1), // 4 live lanes
    ) else {
        return;
    };

    // Writes are positions -2..=1 = 4 bytes at row*stridea - 2.
    let pack = |v: __m128i| -> i32 { _mm_cvtsi128_si32(_mm_packus_epi16(v, v)) };
    let packed = _mm_setr_epi32(pack(out[0]), pack(out[1]), pack(out[2]), pack(out[3]));
    let row = |k: i8| -> __m128i {
        _mm_shuffle_epi8(
            packed,
            _mm_setr_epi8(
                k,
                k + 4,
                k + 8,
                k + 12,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
            ),
        )
    };
    let store_row = |buf: &mut [u8], v: __m128i, r: isize| {
        let start = signed_idx(base, r * stridea - 2);
        buf[start..start + 4].copy_from_slice(&_mm_cvtsi128_si32(v).to_le_bytes());
    };
    store_row(buf, row(0), 0);
    store_row(buf, row(1), 1);
    store_row(buf, row(2), 2);
    store_row(buf, row(3), 3);
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
    // Load each row's 8-byte filter window [p3..q3] = offsets -4..+3,
    // then a 4x8 u8 transpose yields the 8 pixel-position vectors the
    // shared u8 core expects (4 row lanes in each vector's low bytes).
    let load_row = |row: isize| -> __m128i {
        let start = signed_idx(base, row * stridea - 4);
        loadi64!(&buf[start..start + 8])
    };
    let r0 = load_row(0);
    let r1 = load_row(1);
    let r2 = load_row(2);
    let r3 = load_row(3);

    // cols j: byte-interleave row pairs, then dword-interleave to get
    // [col_j rows0-3] u32s.
    let a01 = _mm_unpacklo_epi8(r0, r1);
    let a23 = _mm_unpacklo_epi8(r2, r3);
    let b_lo = _mm_unpacklo_epi16(a01, a23); // [c0|c1|c2|c3] rows 0-3
    let b_hi = _mm_unpackhi_epi16(a01, a23); // [c4|c5|c6|c7] rows 0-3

    let taps = [
        _mm_shuffle_epi32::<0x00>(b_lo),
        _mm_shuffle_epi32::<0x55>(b_lo),
        _mm_shuffle_epi32::<0xAA>(b_lo),
        _mm_shuffle_epi32::<0xFF>(b_lo),
        _mm_shuffle_epi32::<0x00>(b_hi),
        _mm_shuffle_epi32::<0x55>(b_hi),
        _mm_shuffle_epi32::<0xAA>(b_hi),
        _mm_shuffle_epi32::<0xFF>(b_hi),
    ];

    let Some([final_p2, final_p1, final_p0, final_q0, final_q1, final_q2]) =
        lf_wd8_8bpc_core_u8(taps, e, i, h, 0x0F)
    else {
        return;
    };

    // Back-transpose the 8 position vecs (4B rows each) into 4 row vecs
    // of 8B: byte-interleave col pairs, dword-interleave to row dwords,
    // then pair each row's lo/hi half. Untouched positions (p3, q3) are
    // zero-padded — only the middle 6 bytes of each row are stored.
    let z = _mm_setzero_si128();
    let v = [
        z, final_p2, final_p1, final_p0, final_q0, final_q1, final_q2, z,
    ];
    let t01 = _mm_unpacklo_epi8(v[0], v[1]);
    let t23 = _mm_unpacklo_epi8(v[2], v[3]);
    let t45 = _mm_unpacklo_epi8(v[4], v[5]);
    let t67 = _mm_unpacklo_epi8(v[6], v[7]);
    let c_lo = _mm_unpacklo_epi16(t01, t23); // [row0 c0-3|row1|row2|row3]
    let c_hi = _mm_unpacklo_epi16(t45, t67); // [row0 c4-7|row1|row2|row3]
    let rows01 = _mm_unpacklo_epi32(c_lo, c_hi); // row0 lo64 | row1 hi64
    let rows23 = _mm_unpackhi_epi32(c_lo, c_hi); // row2 lo64 | row3 hi64
    let rows = [
        rows01,
        _mm_unpackhi_epi64(rows01, rows01),
        rows23,
        _mm_unpackhi_epi64(rows23, rows23),
    ];

    // Window byte j sits at offset j-4; writing bytes 1..6 stores
    // offsets -3..+2 = [p2,p1,p0,q0,q1,q2] — the positions 8-tap updates.
    for k in 0..4 {
        let start = signed_idx(base, k as isize * stridea - 3);
        let b = _mm_cvtsi128_si64(rows[k]).to_ne_bytes();
        buf[start..start + 6].copy_from_slice(&b[1..7]);
    }
}

// ============================================================================
// SIMD wd=8 H-FILTER x8 widen — processes 8 rows in parallel (YMM)
// ============================================================================

/// SIMD wd=8 loop filter for 8bpc H-FILTER direction, **8-row variant**.
/// Doubles wd8_simd_h's lane count from XMM (4 i32 = 4 rows) to YMM (8 i32 =
/// 8 rows). Used by the outer dispatcher when two adjacent 4-row edges share
/// the same level.
///
/// Approach: do two 4×4 i32 transposes (one per 4-row group), then combine
/// the halves via inserti128 to form YMM pixel-position vectors with rows
/// 0-3 in the low lane and rows 4-7 in the high lane. Run the same compute
/// as wd8_simd_v_x8. To store, split YMM back into XMM halves and reverse-
/// transpose each 4-row group separately. Positions -4 (p3) and +3 (q3) are
/// unchanged by 8-tap and preserved by writing only offsets -3..+2 (6 bytes
/// per row).
#[cfg(target_arch = "x86_64")]
#[arcane]
fn loop_filter_4_8bpc_wd8_simd_h_x8(
    _token: Desktop64,
    buf: &mut [u8],
    base: usize,
    e: i32,
    i: i32,
    h: i32,
    stridea: isize,
) {
    // Load each row's 8-byte filter window [p3..q3] = offsets -4..+3,
    // then an 8x8 u8 transpose yields the 8 pixel-position vectors the
    // shared u8 core expects (8 row lanes in each vector's low bytes).
    let load_row = |row: isize| -> __m128i {
        let start = signed_idx(base, row * stridea - 4);
        loadi64!(&buf[start..start + 8])
    };

    // 8x8 byte transpose: r[k] = row k's window → out[j] = position j's
    // 8 row bytes (low half). Involution — reuse it for the store side.
    let transpose8x8 = |r: [__m128i; 8]| -> [__m128i; 8] {
        let a01 = _mm_unpacklo_epi8(r[0], r[1]);
        let a23 = _mm_unpacklo_epi8(r[2], r[3]);
        let a45 = _mm_unpacklo_epi8(r[4], r[5]);
        let a67 = _mm_unpacklo_epi8(r[6], r[7]);
        let b_lo_a = _mm_unpacklo_epi16(a01, a23); // cols 0-3, rows 0-3
        let b_hi_a = _mm_unpackhi_epi16(a01, a23); // cols 4-7, rows 0-3
        let b_lo_b = _mm_unpacklo_epi16(a45, a67); // cols 0-3, rows 4-7
        let b_hi_b = _mm_unpackhi_epi16(a45, a67); // cols 4-7, rows 4-7
        let c01 = _mm_unpacklo_epi32(b_lo_a, b_lo_b); // col0 | col1
        let c23 = _mm_unpackhi_epi32(b_lo_a, b_lo_b); // col2 | col3
        let c45 = _mm_unpacklo_epi32(b_hi_a, b_hi_b); // col4 | col5
        let c67 = _mm_unpackhi_epi32(b_hi_a, b_hi_b); // col6 | col7
        let z = _mm_setzero_si128();
        [
            _mm_unpacklo_epi64(c01, z),
            _mm_unpackhi_epi64(c01, z),
            _mm_unpacklo_epi64(c23, z),
            _mm_unpackhi_epi64(c23, z),
            _mm_unpacklo_epi64(c45, z),
            _mm_unpackhi_epi64(c45, z),
            _mm_unpacklo_epi64(c67, z),
            _mm_unpackhi_epi64(c67, z),
        ]
    };

    let rows = [
        load_row(0),
        load_row(1),
        load_row(2),
        load_row(3),
        load_row(4),
        load_row(5),
        load_row(6),
        load_row(7),
    ];
    let taps = transpose8x8(rows);

    let Some([final_p2, final_p1, final_p0, final_q0, final_q1, final_q2]) =
        lf_wd8_8bpc_core_u8(taps, e, i, h, 0xFF)
    else {
        return;
    };

    // Transpose the filtered positions back to rows. Pad the unwritten
    // slots (window bytes 0 and 7 = p3/q3) with zeros — only the middle
    // 6 bytes of each row are stored, so they never reach memory.
    let z = _mm_setzero_si128();
    let out_rows = transpose8x8([
        z, final_p2, final_p1, final_p0, final_q0, final_q1, final_q2, z,
    ]);

    // Window byte j sits at offset j-4; writing bytes 1..6 stores
    // offsets -3..+2 = [p2,p1,p0,q0,q1,q2] — the positions 8-tap updates.
    for k in 0..8 {
        let start = signed_idx(base, k as isize * stridea - 3);
        let b = _mm_cvtsi128_si64(out_rows[k]).to_ne_bytes();
        buf[start..start + 6].copy_from_slice(&b[1..7]);
    }
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
    // Per row: 4 dwords at -7 (p6..p3), -3 (p2..q0), +1 (q1..q4), +5 (q5,q6).
    // Pack each 4-byte column group across the 4 rows into one xmm
    // (one dword per row), then pshufb column-extract the tap vectors.
    let load_chunk = |row: isize, chunk_off: isize| -> __m128i {
        let start = signed_idx(base, row * stridea + chunk_off);
        loadi32!(&buf[start..start + 4])
    };
    // The last chunk needs only q5/q6: dead lanes are zero-filled rather
    // than reading the +7/+8 tail, which at a plane edge is the next row a
    // concurrent tile worker may be writing (#524).
    let load_chunk2 = |row: isize, chunk_off: isize| -> __m128i {
        let start = signed_idx(base, row * stridea + chunk_off);
        _mm_cvtsi32_si128(u16::from_le_bytes(buf[start..start + 2].try_into().unwrap()) as i32)
    };
    let pack_rows = |l: &dyn Fn(isize) -> __m128i| -> __m128i {
        _mm_unpacklo_epi64(
            _mm_unpacklo_epi32(l(0), l(1)),
            _mm_unpacklo_epi32(l(2), l(3)),
        )
    };
    let c0 = pack_rows(&|r| load_chunk(r, -7));
    let c1 = pack_rows(&|r| load_chunk(r, -3));
    let c2 = pack_rows(&|r| load_chunk(r, 1));
    let c3 = pack_rows(&|r| load_chunk2(r, 5));
    let col = |v: __m128i, j: i8| -> __m128i {
        _mm_shuffle_epi8(
            v,
            _mm_setr_epi8(
                j,
                j + 4,
                j + 8,
                j + 12,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
            ),
        )
    };

    let Some(out) = lf_wd16_8bpc_core_u8(
        [
            col(c0, 0), // p6
            col(c0, 1), // p5
            col(c0, 2), // p4
            col(c0, 3), // p3
            col(c1, 0), // p2
            col(c1, 1), // p1
            col(c1, 2), // p0
            col(c1, 3), // q0
            col(c2, 0), // q1
            col(c2, 1), // q2
            col(c2, 2), // q3
            col(c2, 3), // q4
            col(c3, 0), // q5
            col(c3, 1), // q6
        ],
        e,
        i,
        h,
        _mm_cvtsi32_si128(-1), // 4 live lanes
    ) else {
        return;
    };

    // Writes are positions -6..=5 = 12 contiguous bytes at row*stridea - 6.
    // Pack each output position to a dword, then transpose 12 positions × 4
    // rows back into per-row byte strings: [A|B|C] dwords give 12 bytes.
    let pack = |v: __m128i| -> i32 { _mm_cvtsi128_si32(_mm_packus_epi16(v, v)) };
    let a = _mm_setr_epi32(pack(out[0]), pack(out[1]), pack(out[2]), pack(out[3]));
    let b = _mm_setr_epi32(pack(out[4]), pack(out[5]), pack(out[6]), pack(out[7]));
    let c = _mm_setr_epi32(pack(out[8]), pack(out[9]), pack(out[10]), pack(out[11]));
    let store_row = |buf: &mut [u8], r: isize| {
        let sel = |v: __m128i, k: i8| -> __m128i {
            _mm_shuffle_epi8(
                v,
                _mm_setr_epi8(
                    k,
                    k + 4,
                    k + 8,
                    k + 12,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ),
            )
        };
        let k = r as i8;
        let row = _mm_unpacklo_epi64(
            _mm_unpacklo_epi32(sel(a, k), sel(b, k)),
            _mm_unpacklo_epi32(sel(c, k), _mm_setzero_si128()),
        );
        let start = signed_idx(base, r * stridea - 6);
        let b8 = _mm_cvtsi128_si64(row).to_le_bytes();
        buf[start..start + 8].copy_from_slice(&b8);
        let b4 = _mm_extract_epi32::<2>(row).to_le_bytes();
        buf[start + 8..start + 12].copy_from_slice(&b4);
    };
    store_row(buf, 0);
    store_row(buf, 1);
    store_row(buf, 2);
    store_row(buf, 3);
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
    let load4 = |off: isize| -> __m128i {
        let start = signed_idx(base, strideb * off);
        loadi32!(&buf[start..start + 4])
    };

    let Some(out) = lf_narrow_8bpc_core_u8(
        [load4(-2), load4(-1), load4(0), load4(1)],
        e,
        i,
        h,
        _mm_cvtsi32_si128(-1), // 4 live lanes
    ) else {
        return;
    };

    // packus_epi16 = the scalar [0,255] clip; low 4 u8 lanes are live.
    let pack4 = |v: __m128i| -> i32 { _mm_cvtsi128_si32(_mm_packus_epi16(v, v)) };
    let store4 = |buf: &mut [u8], packed: i32, off: isize| {
        let start = signed_idx(base, strideb * off);
        let bytes = packed.to_le_bytes();
        buf[start..start + 4].copy_from_slice(&bytes);
    };
    store4(buf, pack4(out[0]), -2);
    store4(buf, pack4(out[1]), -1);
    store4(buf, pack4(out[2]), 0);
    store4(buf, pack4(out[3]), 1);
}

/// wd=6 (6-tap) loop filter core shared by the v/h kernels. `taps` =
/// [p2, p1, p0, q0, q1, q2] pixel-position vectors with live u8 lanes in the
/// low bytes. Returns the four filtered outputs (positions -2,-1,0,1) as i16
/// vectors, or `None` when every live lane fails `fm`.
///
/// Same u8-mask/i16-math structure as [`lf_narrow_8bpc_core_u8`]; the 6-tap
/// accumulators max out at 255*8+4 < 32767.
#[cfg(target_arch = "x86_64")]
#[rite(v3)]
fn lf_wd6_8bpc_core_u8(
    [p2_u8, p1_u8, p0_u8, q0_u8, q1_u8, q2_u8]: [__m128i; 6],
    e: i32,
    i: i32,
    h: i32,
    live_mask: __m128i,
) -> Option<[__m128i; 4]> {
    let zero = _mm_setzero_si128();
    let i_v8 = _mm_set1_epi8(i as i8);
    let h_v8 = _mm_set1_epi8(h as i8);
    let f_v8 = _mm_set1_epi8(1);

    let absu = |a: __m128i, b: __m128i| _mm_or_si128(_mm_subs_epu8(a, b), _mm_subs_epu8(b, a));
    let le_u8 = |a: __m128i, t: __m128i| _mm_cmpeq_epi8(_mm_subs_epu8(a, t), zero);

    let abs_p1p0 = absu(p1_u8, p0_u8);
    let abs_q1q0 = absu(q1_u8, q0_u8);
    let abs_p0q0 = absu(p0_u8, q0_u8);
    let abs_p1q1 = absu(p1_u8, q1_u8);
    let abs_p2p1 = absu(p2_u8, p1_u8);
    let abs_q2q1 = absu(q2_u8, q1_u8);

    // E term in u16 (exact above 255): mask narrowed back to u8.
    let a16 = _mm_unpacklo_epi8(abs_p0q0, zero);
    let b16 = _mm_unpacklo_epi8(abs_p1q1, zero);
    let sum16 = _mm_add_epi16(_mm_add_epi16(a16, a16), _mm_srli_epi16::<1>(b16));
    let le_e16 = _mm_andnot_si128(
        _mm_cmpgt_epi16(sum16, _mm_set1_epi16(e as i16)),
        _mm_set1_epi16(-1),
    );
    let le_e8 = _mm_packus_epi16(_mm_srli_epi16::<8>(le_e16), _mm_srli_epi16::<8>(le_e16));

    let fm_mask = _mm_and_si128(
        _mm_and_si128(
            _mm_and_si128(le_u8(abs_p1p0, i_v8), le_u8(abs_q1q0, i_v8)),
            le_e8,
        ),
        _mm_and_si128(le_u8(abs_p2p1, i_v8), le_u8(abs_q2q1, i_v8)),
    );
    if _mm_test_all_zeros(fm_mask, live_mask) != 0 {
        return None;
    }

    // flat8in = |p2-p0|<=1 && |p1-p0|<=1 && |q1-q0|<=1 && |q2-q0|<=1
    let flat_mask = _mm_and_si128(
        _mm_and_si128(le_u8(absu(p2_u8, p0_u8), f_v8), le_u8(abs_p1p0, f_v8)),
        _mm_and_si128(le_u8(abs_q1q0, f_v8), le_u8(absu(q2_u8, q0_u8), f_v8)),
    );

    let hev_mask = _mm_or_si128(
        _mm_andnot_si128(le_u8(abs_p1p0, h_v8), _mm_set1_epi8(-1)),
        _mm_andnot_si128(le_u8(abs_q1q0, h_v8), _mm_set1_epi8(-1)),
    );

    // i16 domain for the filters.
    let w16 = |v: __m128i| _mm_unpacklo_epi8(v, zero);
    let p2_v = w16(p2_u8);
    let p1_v = w16(p1_u8);
    let p0_v = w16(p0_u8);
    let q0_v = w16(q0_u8);
    let q1_v = w16(q1_u8);
    let q2_v = w16(q2_u8);

    let dbl = |v: __m128i| _mm_slli_epi16::<1>(v);
    let c4 = _mm_set1_epi16(4);

    // 6-tap outputs (used when fm && flat8in):
    let out_m2 = _mm_srai_epi16::<3>(_mm_add_epi16(
        _mm_add_epi16(
            _mm_add_epi16(_mm_add_epi16(dbl(p2_v), p2_v), dbl(p1_v)),
            _mm_add_epi16(dbl(p0_v), q0_v),
        ),
        c4,
    ));
    let out_m1 = _mm_srai_epi16::<3>(_mm_add_epi16(
        _mm_add_epi16(
            _mm_add_epi16(p2_v, dbl(p1_v)),
            _mm_add_epi16(_mm_add_epi16(dbl(p0_v), dbl(q0_v)), q1_v),
        ),
        c4,
    ));
    let out_0 = _mm_srai_epi16::<3>(_mm_add_epi16(
        _mm_add_epi16(
            _mm_add_epi16(p1_v, dbl(p0_v)),
            _mm_add_epi16(_mm_add_epi16(dbl(q0_v), dbl(q1_v)), q2_v),
        ),
        c4,
    ));
    let q2_3 = _mm_add_epi16(dbl(q2_v), q2_v);
    let out_1 = _mm_srai_epi16::<3>(_mm_add_epi16(
        _mm_add_epi16(
            _mm_add_epi16(p0_v, dbl(q0_v)),
            _mm_add_epi16(dbl(q1_v), q2_3),
        ),
        c4,
    ));

    // Narrow filter (used when fm && !flat8in).
    let neg128 = _mm_set1_epi16(-128);
    let pos127 = _mm_set1_epi16(127);
    let iclip = |v: __m128i| _mm_min_epi16(_mm_max_epi16(v, neg128), pos127);

    let diff_q0p0 = _mm_sub_epi16(q0_v, p0_v);
    let three_d = _mm_add_epi16(_mm_slli_epi16::<1>(diff_q0p0), diff_q0p0);
    let diff_p1q1 = _mm_sub_epi16(p1_v, q1_v);

    let f_hev = iclip(_mm_add_epi16(three_d, iclip(diff_p1q1)));
    let f_no = iclip(three_d);

    let c3 = _mm_set1_epi16(3);
    let one = _mm_set1_epi16(1);

    let f1_hev = _mm_srai_epi16::<3>(_mm_min_epi16(_mm_add_epi16(f_hev, c4), pos127));
    let f2_hev = _mm_srai_epi16::<3>(_mm_min_epi16(_mm_add_epi16(f_hev, c3), pos127));
    let f1_no = _mm_srai_epi16::<3>(_mm_min_epi16(_mm_add_epi16(f_no, c4), pos127));
    let f2_no = _mm_srai_epi16::<3>(_mm_min_epi16(_mm_add_epi16(f_no, c3), pos127));
    let f_extra = _mm_srai_epi16::<1>(_mm_add_epi16(f1_no, one));

    let p0_hev = _mm_add_epi16(p0_v, f2_hev);
    let q0_hev = _mm_sub_epi16(q0_v, f1_hev);
    let p0_no = _mm_add_epi16(p0_v, f2_no);
    let q0_no = _mm_sub_epi16(q0_v, f1_no);
    let p1_no = _mm_add_epi16(p1_v, f_extra);
    let q1_no = _mm_sub_epi16(q1_v, f_extra);

    let expand = |m: __m128i| _mm_unpacklo_epi8(m, m);
    let hev16 = expand(hev_mask);
    let flat16 = expand(flat_mask);
    let fm16 = expand(fm_mask);
    let blendv = |a: __m128i, b: __m128i, mask: __m128i| -> __m128i {
        _mm_or_si128(_mm_andnot_si128(mask, a), _mm_and_si128(mask, b))
    };

    let narrow_p1 = blendv(p1_no, p1_v, hev16);
    let narrow_p0 = blendv(p0_no, p0_hev, hev16);
    let narrow_q0 = blendv(q0_no, q0_hev, hev16);
    let narrow_q1 = blendv(q1_no, q1_v, hev16);

    let out_m2_sel = blendv(narrow_p1, out_m2, flat16);
    let out_m1_sel = blendv(narrow_p0, out_m1, flat16);
    let out_0_sel = blendv(narrow_q0, out_0, flat16);
    let out_1_sel = blendv(narrow_q1, out_1, flat16);

    Some([
        blendv(p1_v, out_m2_sel, fm16),
        blendv(p0_v, out_m1_sel, fm16),
        blendv(q0_v, out_0_sel, fm16),
        blendv(q1_v, out_1_sel, fm16),
    ])
}

/// wd=16 (14-tap) loop filter core shared by the v/h 4-lane kernels.
/// `taps` = [p6..q6] pixel-position vectors with live u8 lanes in the low
/// bytes. Returns the twelve filtered outputs (positions -6..=5) as i16
/// vectors, or `None` when every live lane fails `fm`.
///
/// Same u8-mask/i16-math structure as the narrow/wd6 cores: masks in the
/// u8 domain, E term in u16 for exactness above 255, 14-tap accumulators
/// max at 255*16+8 < 32767 so i16 holds them without saturation.
#[cfg(target_arch = "x86_64")]
#[rite(v3)]
fn lf_wd16_8bpc_core_u8(
    taps: [__m128i; 14],
    e: i32,
    i: i32,
    h: i32,
    live_mask: __m128i,
) -> Option<[__m128i; 12]> {
    let [
        p6_u8,
        p5_u8,
        p4_u8,
        p3_u8,
        p2_u8,
        p1_u8,
        p0_u8,
        q0_u8,
        q1_u8,
        q2_u8,
        q3_u8,
        q4_u8,
        q5_u8,
        q6_u8,
    ] = taps;

    let zero = _mm_setzero_si128();
    let i_v8 = _mm_set1_epi8(i as i8);
    let h_v8 = _mm_set1_epi8(h as i8);
    let f_v8 = _mm_set1_epi8(1);

    let absu = |a: __m128i, b: __m128i| _mm_or_si128(_mm_subs_epu8(a, b), _mm_subs_epu8(b, a));
    let le_u8 = |a: __m128i, t: __m128i| _mm_cmpeq_epi8(_mm_subs_epu8(a, t), zero);

    let abs_p1p0 = absu(p1_u8, p0_u8);
    let abs_q1q0 = absu(q1_u8, q0_u8);
    let abs_p0q0 = absu(p0_u8, q0_u8);
    let abs_p1q1 = absu(p1_u8, q1_u8);
    let abs_p2p1 = absu(p2_u8, p1_u8);
    let abs_q2q1 = absu(q2_u8, q1_u8);
    let abs_p3p2 = absu(p3_u8, p2_u8);
    let abs_q3q2 = absu(q3_u8, q2_u8);

    // E term in u16 (exact above 255).
    let a16 = _mm_unpacklo_epi8(abs_p0q0, zero);
    let b16 = _mm_unpacklo_epi8(abs_p1q1, zero);
    let sum16 = _mm_add_epi16(_mm_add_epi16(a16, a16), _mm_srli_epi16::<1>(b16));
    let le_e16 = _mm_andnot_si128(
        _mm_cmpgt_epi16(sum16, _mm_set1_epi16(e as i16)),
        _mm_set1_epi16(-1),
    );
    let le_e8 = _mm_packus_epi16(_mm_srli_epi16::<8>(le_e16), _mm_srli_epi16::<8>(le_e16));

    let fm_mask = _mm_and_si128(
        _mm_and_si128(le_u8(abs_p1p0, i_v8), le_u8(abs_q1q0, i_v8)),
        _mm_and_si128(
            _mm_and_si128(le_e8, le_u8(abs_p2p1, i_v8)),
            _mm_and_si128(
                _mm_and_si128(le_u8(abs_q2q1, i_v8), le_u8(abs_p3p2, i_v8)),
                le_u8(abs_q3q2, i_v8),
            ),
        ),
    );
    if _mm_test_all_zeros(fm_mask, live_mask) != 0 {
        return None;
    }

    let flat8out_mask = _mm_and_si128(
        _mm_and_si128(
            le_u8(absu(p6_u8, p0_u8), f_v8),
            _mm_and_si128(
                le_u8(absu(p5_u8, p0_u8), f_v8),
                le_u8(absu(p4_u8, p0_u8), f_v8),
            ),
        ),
        _mm_and_si128(
            le_u8(absu(q4_u8, q0_u8), f_v8),
            _mm_and_si128(
                le_u8(absu(q5_u8, q0_u8), f_v8),
                le_u8(absu(q6_u8, q0_u8), f_v8),
            ),
        ),
    );

    let flat8in_mask = _mm_and_si128(
        _mm_and_si128(
            le_u8(absu(p2_u8, p0_u8), f_v8),
            _mm_and_si128(le_u8(abs_p1p0, f_v8), le_u8(abs_q1q0, f_v8)),
        ),
        _mm_and_si128(
            le_u8(absu(q2_u8, q0_u8), f_v8),
            _mm_and_si128(
                le_u8(absu(p3_u8, p0_u8), f_v8),
                le_u8(absu(q3_u8, q0_u8), f_v8),
            ),
        ),
    );

    let hev_mask = _mm_or_si128(
        _mm_andnot_si128(le_u8(abs_p1p0, h_v8), _mm_set1_epi8(-1)),
        _mm_andnot_si128(le_u8(abs_q1q0, h_v8), _mm_set1_epi8(-1)),
    );

    // i16 domain for the filters and the original-pixel blends.
    let w16 = |v: __m128i| _mm_unpacklo_epi8(v, zero);
    let p6_v = w16(p6_u8);
    let p5_v = w16(p5_u8);
    let p4_v = w16(p4_u8);
    let p3_v = w16(p3_u8);
    let p2_v = w16(p2_u8);
    let p1_v = w16(p1_u8);
    let p0_v = w16(p0_u8);
    let q0_v = w16(q0_u8);
    let q1_v = w16(q1_u8);
    let q2_v = w16(q2_u8);
    let q3_v = w16(q3_u8);
    let q4_v = w16(q4_u8);
    let q5_v = w16(q5_u8);
    let q6_v = w16(q6_u8);

    let dbl = |v: __m128i| _mm_slli_epi16::<1>(v);
    let add = |a: __m128i, b: __m128i| _mm_add_epi16(a, b);
    let add3 = |a: __m128i, b: __m128i, c: __m128i| add(add(a, b), c);
    let add4 = |a: __m128i, b: __m128i, c: __m128i, d: __m128i| add(add(a, b), add(c, d));
    let c4 = _mm_set1_epi16(4);
    let c8 = _mm_set1_epi16(8);

    // 14-tap outputs (positions -6..5)
    let p6_5 = add(add(dbl(p6_v), dbl(p6_v)), p6_v);
    let q6_5 = add(add(dbl(q6_v), dbl(q6_v)), q6_v);

    let mut s = add(p6_5, add(dbl(p6_v), dbl(p5_v)));
    s = add(s, dbl(p4_v));
    s = add(s, add4(p3_v, p2_v, p1_v, p0_v));
    s = add(s, add(q0_v, c8));
    let out_m6 = _mm_srai_epi16::<4>(s);

    let mut s = add(p6_5, add(dbl(p5_v), dbl(p4_v)));
    s = add(s, dbl(p3_v));
    s = add(s, add4(p2_v, p1_v, p0_v, q0_v));
    s = add(s, add(q1_v, c8));
    let out_m5 = _mm_srai_epi16::<4>(s);

    let p6_4 = add(dbl(p6_v), dbl(p6_v));
    let mut s = add(p6_4, p5_v);
    s = add(s, add(dbl(p4_v), dbl(p3_v)));
    s = add(s, dbl(p2_v));
    s = add(s, add4(p1_v, p0_v, q0_v, q1_v));
    s = add(s, add(q2_v, c8));
    let out_m4 = _mm_srai_epi16::<4>(s);

    let p6_3 = add(dbl(p6_v), p6_v);
    let mut s = add(p6_3, add(p5_v, p4_v));
    s = add(s, add(dbl(p3_v), dbl(p2_v)));
    s = add(s, dbl(p1_v));
    s = add(s, add4(p0_v, q0_v, q1_v, q2_v));
    s = add(s, add(q3_v, c8));
    let out_m3 = _mm_srai_epi16::<4>(s);

    let mut s = add(dbl(p6_v), p5_v);
    s = add(s, add(p4_v, p3_v));
    s = add(s, add(dbl(p2_v), dbl(p1_v)));
    s = add(s, dbl(p0_v));
    s = add(s, add4(q0_v, q1_v, q2_v, q3_v));
    s = add(s, add(q4_v, c8));
    let out_m2 = _mm_srai_epi16::<4>(s);

    let mut s = add(p6_v, p5_v);
    s = add(s, add(p4_v, p3_v));
    s = add(s, p2_v);
    s = add(s, add(dbl(p1_v), dbl(p0_v)));
    s = add(s, dbl(q0_v));
    s = add(s, add4(q1_v, q2_v, q3_v, q4_v));
    s = add(s, add(q5_v, c8));
    let out_m1 = _mm_srai_epi16::<4>(s);

    let mut s = add(p5_v, p4_v);
    s = add(s, add(p3_v, p2_v));
    s = add(s, p1_v);
    s = add(s, add(dbl(p0_v), dbl(q0_v)));
    s = add(s, dbl(q1_v));
    s = add(s, add4(q2_v, q3_v, q4_v, q5_v));
    s = add(s, add(q6_v, c8));
    let out_0 = _mm_srai_epi16::<4>(s);

    let mut s = add(p4_v, p3_v);
    s = add(s, add(p2_v, p1_v));
    s = add(s, p0_v);
    s = add(s, add(dbl(q0_v), dbl(q1_v)));
    s = add(s, dbl(q2_v));
    s = add(s, add4(q3_v, q4_v, q5_v, q6_v));
    s = add(s, add(q6_v, c8));
    let out_1 = _mm_srai_epi16::<4>(s);

    let mut s = add(p3_v, p2_v);
    s = add(s, add(p1_v, p0_v));
    s = add(s, q0_v);
    s = add(s, add(dbl(q1_v), dbl(q2_v)));
    s = add(s, dbl(q3_v));
    let q6_3 = add(dbl(q6_v), q6_v);
    s = add(s, add3(q4_v, q5_v, q6_3));
    s = add(s, c8);
    let out_2 = _mm_srai_epi16::<4>(s);

    let q6_4 = add(dbl(q6_v), dbl(q6_v));
    let mut s = add(p2_v, p1_v);
    s = add(s, add(p0_v, q0_v));
    s = add(s, q1_v);
    s = add(s, add(dbl(q2_v), dbl(q3_v)));
    s = add(s, dbl(q4_v));
    s = add(s, add(q5_v, q6_4));
    s = add(s, c8);
    let out_3 = _mm_srai_epi16::<4>(s);

    let mut s = add(p1_v, p0_v);
    s = add(s, add(q0_v, q1_v));
    s = add(s, q2_v);
    s = add(s, add(dbl(q3_v), dbl(q4_v)));
    s = add(s, dbl(q5_v));
    s = add(s, q6_5);
    s = add(s, c8);
    let out_4 = _mm_srai_epi16::<4>(s);

    let q6_7 = add(q6_5, dbl(q6_v));
    let mut s = add(p0_v, q0_v);
    s = add(s, add(q1_v, q2_v));
    s = add(s, q3_v);
    s = add(s, add(dbl(q4_v), dbl(q5_v)));
    s = add(s, q6_7);
    s = add(s, c8);
    let out_5 = _mm_srai_epi16::<4>(s);

    // 8-tap outputs
    let triple = |v: __m128i| add(dbl(v), v);
    let out8_m3 = _mm_srai_epi16::<3>(add(
        add4(triple(p3_v), dbl(p2_v), p1_v, p0_v),
        add(q0_v, c4),
    ));
    let out8_m2 = _mm_srai_epi16::<3>(add(
        add4(dbl(p3_v), p2_v, dbl(p1_v), p0_v),
        add3(q0_v, q1_v, c4),
    ));
    let out8_m1 = _mm_srai_epi16::<3>(add(
        add4(p3_v, p2_v, p1_v, dbl(p0_v)),
        add4(q0_v, q1_v, q2_v, c4),
    ));
    let out8_0 = _mm_srai_epi16::<3>(add(
        add4(p2_v, p1_v, p0_v, dbl(q0_v)),
        add4(q1_v, q2_v, q3_v, c4),
    ));
    let out8_1 = _mm_srai_epi16::<3>(add(
        add4(p1_v, p0_v, q0_v, dbl(q1_v)),
        add4(q2_v, q3_v, q3_v, c4),
    ));
    let out8_2 = _mm_srai_epi16::<3>(add(
        add4(p0_v, q0_v, q1_v, dbl(q2_v)),
        add4(q3_v, q3_v, q3_v, c4),
    ));

    // Narrow filter (4-tap) — i16 domain.
    let neg128 = _mm_set1_epi16(-128);
    let pos127 = _mm_set1_epi16(127);
    let iclip = |v: __m128i| _mm_min_epi16(_mm_max_epi16(v, neg128), pos127);
    let diff_q0p0 = _mm_sub_epi16(q0_v, p0_v);
    let three_d = _mm_add_epi16(_mm_slli_epi16::<1>(diff_q0p0), diff_q0p0);
    let diff_p1q1 = _mm_sub_epi16(p1_v, q1_v);
    let f_hev = iclip(_mm_add_epi16(three_d, iclip(diff_p1q1)));
    let f_no = iclip(three_d);
    let c3i = _mm_set1_epi16(3);
    let one = _mm_set1_epi16(1);
    let f1_hev = _mm_srai_epi16::<3>(_mm_min_epi16(_mm_add_epi16(f_hev, c4), pos127));
    let f2_hev = _mm_srai_epi16::<3>(_mm_min_epi16(_mm_add_epi16(f_hev, c3i), pos127));
    let f1_no = _mm_srai_epi16::<3>(_mm_min_epi16(_mm_add_epi16(f_no, c4), pos127));
    let f2_no = _mm_srai_epi16::<3>(_mm_min_epi16(_mm_add_epi16(f_no, c3i), pos127));
    let f_extra = _mm_srai_epi16::<1>(_mm_add_epi16(f1_no, one));
    let p0_hev = _mm_add_epi16(p0_v, f2_hev);
    let q0_hev = _mm_sub_epi16(q0_v, f1_hev);
    let p0_no = _mm_add_epi16(p0_v, f2_no);
    let q0_no = _mm_sub_epi16(q0_v, f1_no);
    let p1_no = _mm_add_epi16(p1_v, f_extra);
    let q1_no = _mm_sub_epi16(q1_v, f_extra);

    let expand = |m: __m128i| _mm_unpacklo_epi8(m, m);
    let hev16 = expand(hev_mask);
    let flat8in16 = expand(flat8in_mask);
    let wide16 = _mm_and_si128(flat8in16, expand(flat8out_mask));
    let fm16 = expand(fm_mask);
    let blendv = |a: __m128i, b: __m128i, mask: __m128i| -> __m128i {
        _mm_or_si128(_mm_andnot_si128(mask, a), _mm_and_si128(mask, b))
    };

    let narrow_p1 = blendv(p1_no, p1_v, hev16);
    let narrow_p0 = blendv(p0_no, p0_hev, hev16);
    let narrow_q0 = blendv(q0_no, q0_hev, hev16);
    let narrow_q1 = blendv(q1_no, q1_v, hev16);

    let mid_m3 = blendv(p2_v, out8_m3, flat8in16);
    let mid_m2 = blendv(narrow_p1, out8_m2, flat8in16);
    let mid_m1 = blendv(narrow_p0, out8_m1, flat8in16);
    let mid_0 = blendv(narrow_q0, out8_0, flat8in16);
    let mid_1 = blendv(narrow_q1, out8_1, flat8in16);
    let mid_2 = blendv(q2_v, out8_2, flat8in16);

    let sel_m6 = blendv(p5_v, out_m6, wide16);
    let sel_m5 = blendv(p4_v, out_m5, wide16);
    let sel_m4 = blendv(p3_v, out_m4, wide16);
    let sel_m3 = blendv(mid_m3, out_m3, wide16);
    let sel_m2 = blendv(mid_m2, out_m2, wide16);
    let sel_m1 = blendv(mid_m1, out_m1, wide16);
    let sel_0 = blendv(mid_0, out_0, wide16);
    let sel_1 = blendv(mid_1, out_1, wide16);
    let sel_2 = blendv(mid_2, out_2, wide16);
    let sel_3 = blendv(q3_v, out_3, wide16);
    let sel_4 = blendv(q4_v, out_4, wide16);
    let sel_5 = blendv(q5_v, out_5, wide16);

    Some([
        blendv(p5_v, sel_m6, fm16),
        blendv(p4_v, sel_m5, fm16),
        blendv(p3_v, sel_m4, fm16),
        blendv(p2_v, sel_m3, fm16),
        blendv(p1_v, sel_m2, fm16),
        blendv(p0_v, sel_m1, fm16),
        blendv(q0_v, sel_0, fm16),
        blendv(q1_v, sel_1, fm16),
        blendv(q2_v, sel_2, fm16),
        blendv(q3_v, sel_3, fm16),
        blendv(q4_v, sel_4, fm16),
        blendv(q5_v, sel_5, fm16),
    ])
}

/// Narrow (4-tap) loop filter core shared by the v/h and x4/x8 kernels.
/// `taps` = [p1, p0, q0, q1] pixel-position vectors with live u8 lanes in the
/// low bytes; `live_mask` covers exactly the live mask bytes for the ptest
/// early-out. Returns the four filtered outputs as i16 vectors (i16 lanes
/// match the live u8 lanes 1:1 via `unpacklo_epi8`), or `None` when every
/// live lane fails `fm` — the caller then leaves the pixels untouched.
///
/// Masks stay in the u8 domain (`subs_epu8` + `cmpeq_epi8`); the E term
/// widens to u16 because `2*|p0-q0| + |p1-q1|>>1 <= e` must reject true sums
/// >= 256 like the wd8 core. Filter arithmetic runs in i16 lanes — the
/// narrow filter's intermediates reach +-382, outside u8.
#[cfg(target_arch = "x86_64")]
#[rite(v3)]
fn lf_narrow_8bpc_core_u8(
    [p1_u8, p0_u8, q0_u8, q1_u8]: [__m128i; 4],
    e: i32,
    i: i32,
    h: i32,
    live_mask: __m128i,
) -> Option<[__m128i; 4]> {
    let zero = _mm_setzero_si128();
    let i_v8 = _mm_set1_epi8(i as i8);
    let h_v8 = _mm_set1_epi8(h as i8);

    let absu = |a: __m128i, b: __m128i| _mm_or_si128(_mm_subs_epu8(a, b), _mm_subs_epu8(b, a));
    let le_u8 = |a: __m128i, t: __m128i| _mm_cmpeq_epi8(_mm_subs_epu8(a, t), zero);

    let abs_p1p0 = absu(p1_u8, p0_u8);
    let abs_q1q0 = absu(q1_u8, q0_u8);
    let abs_p0q0 = absu(p0_u8, q0_u8);
    let abs_p1q1 = absu(p1_u8, q1_u8);

    // E term in u16 (exact above 255): mask narrowed back to u8.
    let a16 = _mm_unpacklo_epi8(abs_p0q0, zero);
    let b16 = _mm_unpacklo_epi8(abs_p1q1, zero);
    let sum16 = _mm_add_epi16(_mm_add_epi16(a16, a16), _mm_srli_epi16::<1>(b16));
    let le_e16 = _mm_andnot_si128(
        _mm_cmpgt_epi16(sum16, _mm_set1_epi16(e as i16)),
        _mm_set1_epi16(-1),
    );
    let le_e8 = _mm_packus_epi16(_mm_srli_epi16::<8>(le_e16), _mm_srli_epi16::<8>(le_e16));

    let fm_mask = _mm_and_si128(
        _mm_and_si128(le_u8(abs_p1p0, i_v8), le_u8(abs_q1q0, i_v8)),
        le_e8,
    );
    // Dead lanes compare equal on zero inputs, so probe only live bytes.
    if _mm_test_all_zeros(fm_mask, live_mask) != 0 {
        return None;
    }

    let hev_mask = _mm_or_si128(
        _mm_andnot_si128(le_u8(abs_p1p0, h_v8), _mm_set1_epi8(-1)),
        _mm_andnot_si128(le_u8(abs_q1q0, h_v8), _mm_set1_epi8(-1)),
    );

    // i16 domain for the filter itself.
    let w16 = |v: __m128i| _mm_unpacklo_epi8(v, zero);
    let p1_v = w16(p1_u8);
    let p0_v = w16(p0_u8);
    let q0_v = w16(q0_u8);
    let q1_v = w16(q1_u8);

    let neg128 = _mm_set1_epi16(-128);
    let pos127 = _mm_set1_epi16(127);
    let iclip = |v: __m128i| _mm_min_epi16(_mm_max_epi16(v, neg128), pos127);

    let diff_q0p0 = _mm_sub_epi16(q0_v, p0_v);
    let three_d = _mm_add_epi16(_mm_slli_epi16::<1>(diff_q0p0), diff_q0p0);
    let diff_p1q1 = _mm_sub_epi16(p1_v, q1_v);

    let f_hev = iclip(_mm_add_epi16(three_d, iclip(diff_p1q1)));
    let f_no = iclip(three_d);

    let c4 = _mm_set1_epi16(4);
    let c3 = _mm_set1_epi16(3);
    let one = _mm_set1_epi16(1);

    let f1_hev = _mm_srai_epi16::<3>(_mm_min_epi16(_mm_add_epi16(f_hev, c4), pos127));
    let f2_hev = _mm_srai_epi16::<3>(_mm_min_epi16(_mm_add_epi16(f_hev, c3), pos127));
    let f1_no = _mm_srai_epi16::<3>(_mm_min_epi16(_mm_add_epi16(f_no, c4), pos127));
    let f2_no = _mm_srai_epi16::<3>(_mm_min_epi16(_mm_add_epi16(f_no, c3), pos127));
    let f_extra = _mm_srai_epi16::<1>(_mm_add_epi16(f1_no, one));

    let p0_hev = _mm_add_epi16(p0_v, f2_hev);
    let q0_hev = _mm_sub_epi16(q0_v, f1_hev);
    let p0_no = _mm_add_epi16(p0_v, f2_no);
    let q0_no = _mm_sub_epi16(q0_v, f1_no);
    let p1_no = _mm_add_epi16(p1_v, f_extra);
    let q1_no = _mm_sub_epi16(q1_v, f_extra);

    // u8 lane masks expand to whole i16 lanes for blendv_epi8.
    let expand = |m: __m128i| _mm_unpacklo_epi8(m, m);
    let hev16 = expand(hev_mask);
    let fm16 = expand(fm_mask);
    let blendv = |a: __m128i, b: __m128i, mask: __m128i| -> __m128i {
        _mm_or_si128(_mm_andnot_si128(mask, a), _mm_and_si128(mask, b))
    };
    let p1_filt = blendv(p1_no, p1_v, hev16);
    let p0_filt = blendv(p0_no, p0_hev, hev16);
    let q0_filt = blendv(q0_no, q0_hev, hev16);
    let q1_filt = blendv(q1_no, q1_v, hev16);

    Some([
        blendv(p1_v, p1_filt, fm16),
        blendv(p0_v, p0_filt, fm16),
        blendv(q0_v, q0_filt, fm16),
        blendv(q1_v, q1_filt, fm16),
    ])
}

// ============================================================================
// SIMD inner narrow loop filter — V-FILTER, 8 columns at a time (YMM)
// ============================================================================

/// SIMD narrow 4-tap loop filter for 8bpc V-FILTER direction.
/// Wider variant: processes **8** adjacent filter positions (8 columns) per
/// call — 8 live u8 lanes per tap, all in `__m128i`. Caller must pre-verify
/// the next 4 columns also use narrow (wd=4) and the same `l`/`e`/`i`/`h`
/// parameters as the current group; the outer dispatcher merges two
/// `vmask[0]`-only adjacent edges into one of these calls.
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
    let load8 = |off: isize| -> __m128i {
        let start = signed_idx(base, strideb * off);
        loadi64!(&buf[start..start + 8])
    };

    let Some(out) = lf_narrow_8bpc_core_u8(
        [load8(-2), load8(-1), load8(0), load8(1)],
        e,
        i,
        h,
        _mm_cvtsi64_si128(-1), // 8 live lanes
    ) else {
        return;
    };

    // packus_epi16 = the scalar [0,255] clip; bytes 0-7 hold the results.
    let pack8 = |v: __m128i| -> __m128i { _mm_packus_epi16(v, v) };
    let store8 = |buf: &mut [u8], v: __m128i, off: isize| {
        let start = signed_idx(base, strideb * off);
        storei64!(&mut buf[start..start + 8], v);
    };
    store8(buf, pack8(out[0]), -2);
    store8(buf, pack8(out[1]), -1);
    store8(buf, pack8(out[2]), 0);
    store8(buf, pack8(out[3]), 1);
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
#[cfg_attr(not(target_arch = "x86_64"), allow(unused_variables))]
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

    // Helper: same as v-filter dispatcher
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
                let idx = if vmask[2] & xy != 0 {
                    16
                } else if vmask[1] & xy != 0 {
                    8
                } else {
                    4
                };

                // Eight-position kernels require an adjacent group with the
                // same width and effective level. The wide H footprint also
                // requires independent rows, each containing all 14 taps.
                #[cfg(target_arch = "x86_64")]
                {
                    let next_xy = xy.wrapping_shl(1);
                    if next_xy != 0 && bitdepth_max == 255 {
                        let next_idx = if vmask[2] & next_xy != 0 {
                            16
                        } else if vmask[1] & next_xy != 0 {
                            8
                        } else if vmask[0] & next_xy != 0 {
                            4
                        } else {
                            0
                        };
                        if next_idx == idx
                            && (idx == 4 || idx == 8 || idx == 16 && stride.unsigned_abs() >= 14)
                            && let Some((l2, _, _, _)) = derive_levels(lvl_offset + b4_stridea)
                            && l2 == l
                        {
                            if idx == 16 {
                                packed16::apply_h(
                                    _token,
                                    buf,
                                    dst_offset,
                                    stride,
                                    [e as u8, i as u8, h as u8],
                                );
                            } else if idx == 8 {
                                loop_filter_4_8bpc_wd8_simd_h_x8(
                                    _token, buf, dst_offset, e, i, h, stridea,
                                );
                            } else {
                                loop_filter_4_8bpc_narrow_simd_h_x8(
                                    _token, buf, dst_offset, e, i, h, stridea,
                                );
                            }
                            xy = next_xy << 1;
                            dst_offset = signed_idx(dst_offset, 8 * stridea);
                            lvl_offset += 2 * b4_stridea;
                            continue;
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

/// Loop filter for Y plane, vertical edges (8bpc)
#[cfg(any(target_arch = "x86_64", target_arch = "wasm32"))]
#[cfg_attr(target_arch = "x86_64", arcane)]
#[allow(unused_mut)]
#[cfg_attr(not(target_arch = "x86_64"), allow(unused_variables))]
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

                // Tier of the edge at bitmask position `bit` (0 = no edge).
                let tier_at = |bit: u32| -> i32 {
                    if vmask[2] & bit != 0 {
                        16
                    } else if vmask[1] & bit != 0 {
                        8
                    } else if vmask[0] & bit != 0 {
                        4
                    } else {
                        0
                    }
                };

                // Fast path: x16 ZMM (AVX-512) kernel when the next THREE
                // adjacent edges (16 columns total) all share the same wd=16
                // tier AND the same `l`. Quadruples throughput per call (16
                // cols vs 4) on Zen 4 / x86-64-v4. Bit-exact with x8/x4/scalar.
                #[cfg(target_arch = "x86_64")]
                if idx == 16 && bitdepth_max == 255 {
                    let xy1 = xy.wrapping_shl(1);
                    let xy2 = xy.wrapping_shl(2);
                    let xy3 = xy.wrapping_shl(3);
                    if xy3 != 0
                        && tier_at(xy1) == 16
                        && tier_at(xy2) == 16
                        && tier_at(xy3) == 16
                        && let Some(token) = crate::src::cpu::summon_avx512()
                        && let Some((l1, _, _, _)) = derive_levels(lvl_offset + b4_stridea)
                        && l1 == l
                        && let Some((l2, _, _, _)) = derive_levels(lvl_offset + 2 * b4_stridea)
                        && l2 == l
                        && let Some((l3, _, _, _)) = derive_levels(lvl_offset + 3 * b4_stridea)
                        && l3 == l
                    {
                        loop_filter_4_8bpc_wd16_simd_v_x16(
                            token, buf, dst_offset, e, i, h, strideb,
                        );
                        xy = xy3 << 1;
                        dst_offset = signed_idx(dst_offset, 16 * stridea);
                        lvl_offset += 4 * b4_stridea;
                        continue;
                    }
                }

                // Fast path: x8 YMM kernel when the next adjacent edge has
                // the same wd-tier AND the same `l` (and therefore same e/i/h).
                // Doubles throughput per call (8 cols vs 4).
                #[cfg(target_arch = "x86_64")]
                {
                    let next_xy = xy.wrapping_shl(1);
                    if next_xy != 0 && bitdepth_max == 255 {
                        // Compute next edge's wd-tier (must match current `idx`).
                        let next_idx = if vmask[2] & next_xy != 0 {
                            16
                        } else if vmask[1] & next_xy != 0 {
                            8
                        } else if vmask[0] & next_xy != 0 {
                            4
                        } else {
                            0 // No edge at next position
                        };
                        if next_idx == idx
                            && let Some((l2, _, _, _)) = derive_levels(lvl_offset + b4_stridea)
                            && l2 == l
                        {
                            match idx {
                                4 => {
                                    loop_filter_4_8bpc_narrow_simd_v_x8(
                                        _token, buf, dst_offset, e, i, h, strideb,
                                    );
                                    xy = next_xy << 1;
                                    dst_offset = signed_idx(dst_offset, 8 * stridea);
                                    lvl_offset += 2 * b4_stridea;
                                    continue;
                                }
                                8 => {
                                    loop_filter_4_8bpc_wd8_simd_v_x8(
                                        _token, buf, dst_offset, e, i, h, strideb,
                                    );
                                    xy = next_xy << 1;
                                    dst_offset = signed_idx(dst_offset, 8 * stridea);
                                    lvl_offset += 2 * b4_stridea;
                                    continue;
                                }
                                16 => {
                                    loop_filter_4_8bpc_wd16_simd_v_x8(
                                        _token, buf, dst_offset, e, i, h, strideb,
                                    );
                                    xy = next_xy << 1;
                                    dst_offset = signed_idx(dst_offset, 8 * stridea);
                                    lvl_offset += 2 * b4_stridea;
                                    continue;
                                }
                                _ => {}
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

                // Two adjacent six-tap groups with equal levels touch
                // independent positions in the existing checked window.
                #[cfg(target_arch = "x86_64")]
                {
                    let next_xy = xy.wrapping_shl(1);
                    if idx == 6
                        && bitdepth_max == 255
                        && next_xy != 0
                        && vmask[1] & next_xy != 0
                        && stride.unsigned_abs() >= 6
                    {
                        let next_offset = lvl_offset + b4_stridea;
                        let next_value = read_lvl(lvl, next_offset, lvl_byte_idx);
                        let next_level = if next_value != 0 {
                            next_value
                        } else if next_offset >= b4_strideb {
                            read_lvl(lvl, next_offset - b4_strideb, lvl_byte_idx)
                        } else {
                            0
                        };
                        if next_level == l {
                            packed6::apply::<true>(
                                _token,
                                buf,
                                dst_offset,
                                stride,
                                [e as u8, i as u8, h as u8],
                            );
                            xy = next_xy << 1;
                            dst_offset = signed_idx(dst_offset, 8 * stridea);
                            lvl_offset += 2 * b4_stridea;
                            continue;
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

                // Two adjacent six-tap groups with equal levels touch
                // independent positions in the existing checked window.
                #[cfg(target_arch = "x86_64")]
                {
                    let next_xy = xy.wrapping_shl(1);
                    if idx == 6
                        && bitdepth_max == 255
                        && next_xy != 0
                        && vmask[1] & next_xy != 0
                        && stride.unsigned_abs() >= 8
                    {
                        let next_offset = lvl_offset + b4_stridea;
                        let next_value = read_lvl(lvl, next_offset, lvl_byte_idx);
                        let next_level = if next_value != 0 {
                            next_value
                        } else if next_offset >= b4_strideb {
                            read_lvl(lvl, next_offset - b4_strideb, lvl_byte_idx)
                        } else {
                            0
                        };
                        if next_level == l {
                            packed6::apply::<false>(
                                _token,
                                buf,
                                dst_offset,
                                stride,
                                [e as u8, i as u8, h as u8],
                            );
                            xy = next_xy << 1;
                            dst_offset = signed_idx(dst_offset, 8 * stridea);
                            lvl_offset += 2 * b4_stridea;
                            continue;
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

// ============================================================================
// FFI WRAPPERS (8bpc) — only compiled with asm feature
//
// AUDITED 2026-08-08, during the `summon().unwrap()` sweep behind
// `tests/decode_permutations.rs`. The four 8bpc wrappers below keep an
// `expect` on the token — the only ones left in `src/safe_simd/` — and that is
// deliberate, because they are NOT the mc_arm/filmgrain_arm defect class:
//
//   * They have NO callers. Under `asm` the loop-filter table is built by
//     `bd_fn!` (include/common/bitdepth.rs), which resolves to the NASM symbol
//     `dav1d_lpf_h_sb_y_8bpc_avx2`, not to this mangled Rust one. Without
//     `asm` they are not compiled at all. The permutation gate cannot reach
//     them either way.
//   * There is no fallback to gate TO: an `extern "C"` shim must fill its
//     destination or corrupt it, and silently doing nothing would be worse
//     than the panic.
//
// The message they used to carry, "AVX2 implies Desktop64", was FALSE:
// `Desktop64` is x86-64-v3 (AVX2 + FMA + BMI1/2 + LZCNT + MOVBE + F16C), a
// strict superset of AVX2, so an AVX2-only CPU would have tripped it. It now
// states the real precondition. If these ever gain a caller, gate them the way
// the live x86 dispatch in this same file does — `crate::src::cpu::summon_avx2()`
// plus a real fallback (see `lpf_dispatch`) — instead of asserting.
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
    // See the AUDITED note on the FFI-wrapper banner above.
    let token = Desktop64::summon().expect(
        "x86-64-v3 (Desktop64) token required; #[target_feature(avx2)] alone does not imply it",
    );
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
    // See the AUDITED note on the FFI-wrapper banner above.
    let token = Desktop64::summon().expect(
        "x86-64-v3 (Desktop64) token required; #[target_feature(avx2)] alone does not imply it",
    );
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
    // See the AUDITED note on the FFI-wrapper banner above.
    let token = Desktop64::summon().expect(
        "x86-64-v3 (Desktop64) token required; #[target_feature(avx2)] alone does not imply it",
    );
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
    // See the AUDITED note on the FFI-wrapper banner above.
    let token = Desktop64::summon().expect(
        "x86-64-v3 (Desktop64) token required; #[target_feature(avx2)] alone does not imply it",
    );
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

// ----------------------------------------------------------------------------
// SIMD helpers shared by all 16bpc kernels: 4 i32 lanes = 4 pixel positions.
// Loads widen u16 -> i32 via _mm_cvtepu16_epi32; stores clamp to
// [0, bitdepth_max] then pack i32 -> u16 via _mm_packus_epi32 (lossless once
// clamped) and write 8 bytes with mm_storel_epi64.
// ----------------------------------------------------------------------------

/// Narrow (4-tap) filter math shared by every 16bpc SIMD kernel.
/// `neg`/`pos` are the iclip bounds `-(128<<bdm8)` / `(128<<bdm8)-1`.
/// Returns (hev_mask, narrow_p1, narrow_p0, narrow_q0, narrow_q1) — narrow
/// outputs already carry the hev blend; the caller blends under `fm_mask`.
#[cfg(target_arch = "x86_64")]
#[rite(v3)]
#[allow(clippy::too_many_arguments)]
fn lf16_narrow_core(
    p1_v: __m128i,
    p0_v: __m128i,
    q0_v: __m128i,
    q1_v: __m128i,
    abs_p1p0: __m128i,
    abs_q1q0: __m128i,
    h_v: __m128i,
    neg: __m128i,
    pos: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i, __m128i) {
    let iclip = |v: __m128i| _mm_min_epi32(_mm_max_epi32(v, neg), pos);
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

    let f1_hev = _mm_srai_epi32::<3>(_mm_min_epi32(_mm_add_epi32(f_hev, c4i), pos));
    let f2_hev = _mm_srai_epi32::<3>(_mm_min_epi32(_mm_add_epi32(f_hev, c3i), pos));
    let f1_no = _mm_srai_epi32::<3>(_mm_min_epi32(_mm_add_epi32(f_no, c4i), pos));
    let f2_no = _mm_srai_epi32::<3>(_mm_min_epi32(_mm_add_epi32(f_no, c3i), pos));
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
    (
        hev_mask,
        blendv(p1_no, p1_v, hev_mask),
        blendv(p0_no, p0_hev, hev_mask),
        blendv(q0_no, q0_hev, hev_mask),
        blendv(q1_no, q1_v, hev_mask),
    )
}

/// 6-tap outputs for positions -2,-1,0,1 (used by the wd=6 path).
#[cfg(target_arch = "x86_64")]
#[rite(v3)]
fn lf16_tap6(
    p2_v: __m128i,
    p1_v: __m128i,
    p0_v: __m128i,
    q0_v: __m128i,
    q1_v: __m128i,
    q2_v: __m128i,
) -> [__m128i; 4] {
    let c4 = _mm_set1_epi32(4);
    let dbl = |v: __m128i| _mm_slli_epi32::<1>(v);
    let add = |a: __m128i, b: __m128i| _mm_add_epi32(a, b);
    let add4 = |a: __m128i, b: __m128i, c: __m128i, d: __m128i| add(add(a, b), add(c, d));
    [
        // out[-2] = (3*p2 + 2*p1 + 2*p0 + q0 + 4) >> 3
        _mm_srai_epi32::<3>(add(
            add4(add(dbl(p2_v), p2_v), dbl(p1_v), dbl(p0_v), q0_v),
            c4,
        )),
        // out[-1] = (p2 + 2*p1 + 2*p0 + 2*q0 + q1 + 4) >> 3
        _mm_srai_epi32::<3>(add(
            add4(p2_v, dbl(p1_v), dbl(p0_v), dbl(q0_v)),
            add(q1_v, c4),
        )),
        // out[ 0] = (p1 + 2*p0 + 2*q0 + 2*q1 + q2 + 4) >> 3
        _mm_srai_epi32::<3>(add(
            add4(p1_v, dbl(p0_v), dbl(q0_v), dbl(q1_v)),
            add(q2_v, c4),
        )),
        // out[ 1] = (p0 + 2*q0 + 2*q1 + 3*q2 + 4) >> 3
        _mm_srai_epi32::<3>(add(
            add4(p0_v, dbl(q0_v), dbl(q1_v), add(dbl(q2_v), q2_v)),
            c4,
        )),
    ]
}

/// 8-tap outputs for positions -3..=2 (used by the wd=8 path and the
/// !flat8out arm of wd=16).
#[cfg(target_arch = "x86_64")]
#[rite(v3)]
#[allow(clippy::too_many_arguments)]
fn lf16_tap8(
    p3_v: __m128i,
    p2_v: __m128i,
    p1_v: __m128i,
    p0_v: __m128i,
    q0_v: __m128i,
    q1_v: __m128i,
    q2_v: __m128i,
    q3_v: __m128i,
) -> [__m128i; 6] {
    let c4 = _mm_set1_epi32(4);
    let dbl = |v: __m128i| _mm_slli_epi32::<1>(v);
    let triple = |v: __m128i| _mm_add_epi32(dbl(v), v);
    let add = |a: __m128i, b: __m128i| _mm_add_epi32(a, b);
    let add3 = |a: __m128i, b: __m128i, c: __m128i| add(add(a, b), c);
    let add4 = |a: __m128i, b: __m128i, c: __m128i, d: __m128i| add(add(a, b), add(c, d));
    [
        // out[-3] = (p3*3 + p2*2 + p1 + p0 + q0 + 4) >> 3
        _mm_srai_epi32::<3>(add(
            add4(triple(p3_v), dbl(p2_v), p1_v, p0_v),
            add(q0_v, c4),
        )),
        // out[-2] = (p3*2 + p2 + p1*2 + p0 + q0 + q1 + 4) >> 3
        _mm_srai_epi32::<3>(add(
            add4(dbl(p3_v), p2_v, dbl(p1_v), p0_v),
            add3(q0_v, q1_v, c4),
        )),
        // out[-1] = (p3 + p2 + p1 + p0*2 + q0 + q1 + q2 + 4) >> 3
        _mm_srai_epi32::<3>(add(
            add4(p3_v, p2_v, p1_v, dbl(p0_v)),
            add4(q0_v, q1_v, q2_v, c4),
        )),
        // out[ 0] = (p2 + p1 + p0 + q0*2 + q1 + q2 + q3 + 4) >> 3
        _mm_srai_epi32::<3>(add(
            add4(p2_v, p1_v, p0_v, dbl(q0_v)),
            add4(q1_v, q2_v, q3_v, c4),
        )),
        // out[ 1] = (p1 + p0 + q0 + q1*2 + q2 + q3*2 + 4) >> 3
        _mm_srai_epi32::<3>(add(
            add4(p1_v, p0_v, q0_v, dbl(q1_v)),
            add4(q2_v, q3_v, q3_v, c4),
        )),
        // out[ 2] = (p0 + q0 + q1 + q2*2 + q3*3 + 4) >> 3
        _mm_srai_epi32::<3>(add(
            add4(p0_v, q0_v, q1_v, dbl(q2_v)),
            add4(q3_v, q3_v, q3_v, c4),
        )),
    ]
}

/// 14-tap outputs for positions -6..=5 (used by the flat8out && flat8in arm of
/// wd=16). `px` = p6..p0, q0..q6 (14 lanes-groups).
#[cfg(target_arch = "x86_64")]
#[rite(v3)]
fn lf16_tap14(px: &[__m128i; 14]) -> [__m128i; 12] {
    let [
        p6_v,
        p5_v,
        p4_v,
        p3_v,
        p2_v,
        p1_v,
        p0_v,
        q0_v,
        q1_v,
        q2_v,
        q3_v,
        q4_v,
        q5_v,
        q6_v,
    ] = *px;
    let c8 = _mm_set1_epi32(8);
    let dbl = |v: __m128i| _mm_slli_epi32::<1>(v);
    let add = |a: __m128i, b: __m128i| _mm_add_epi32(a, b);
    let add3 = |a: __m128i, b: __m128i, c: __m128i| add(add(a, b), c);
    let add4 = |a: __m128i, b: __m128i, c: __m128i, d: __m128i| add(add(a, b), add(c, d));
    let x5 = |v: __m128i| add(add4(v, v, v, v), v);

    let p6_5 = x5(p6_v);
    let q6_5 = x5(q6_v);
    let mut out = [_mm_setzero_si128(); 12];

    // out[-6] = (p6*7 + p5*2 + p4*2 + p3 + p2 + p1 + p0 + q0 + 8) >> 4
    let mut s = add(p6_5, add(dbl(p6_v), dbl(p5_v)));
    s = add(s, dbl(p4_v));
    s = add(s, add4(p3_v, p2_v, p1_v, p0_v));
    s = add(s, add(q0_v, c8));
    out[0] = _mm_srai_epi32::<4>(s);

    // out[-5] = (p6*5 + p5*2 + p4*2 + p3*2 + p2 + p1 + p0 + q0 + q1 + 8) >> 4
    let mut s = add(p6_5, add(dbl(p5_v), dbl(p4_v)));
    s = add(s, dbl(p3_v));
    s = add(s, add4(p2_v, p1_v, p0_v, q0_v));
    s = add(s, add(q1_v, c8));
    out[1] = _mm_srai_epi32::<4>(s);

    // out[-4] = (p6*4 + p5 + p4*2 + p3*2 + p2*2 + p1 + p0 + q0 + q1 + q2 + 8) >> 4
    let mut s = add(add(dbl(p6_v), dbl(p6_v)), p5_v);
    s = add(s, add(dbl(p4_v), dbl(p3_v)));
    s = add(s, dbl(p2_v));
    s = add(s, add4(p1_v, p0_v, q0_v, q1_v));
    s = add(s, add(q2_v, c8));
    out[2] = _mm_srai_epi32::<4>(s);

    // out[-3] = (p6*3 + p5 + p4 + p3*2 + p2*2 + p1*2 + p0 + q0 + q1 + q2 + q3 + 8) >> 4
    let mut s = add(add(dbl(p6_v), p6_v), add(p5_v, p4_v));
    s = add(s, add(dbl(p3_v), dbl(p2_v)));
    s = add(s, dbl(p1_v));
    s = add(s, add4(p0_v, q0_v, q1_v, q2_v));
    s = add(s, add(q3_v, c8));
    out[3] = _mm_srai_epi32::<4>(s);

    // out[-2] = (p6*2 + p5 + p4 + p3 + p2*2 + p1*2 + p0*2 + q0 + q1 + q2 + q3 + q4 + 8) >> 4
    let mut s = add(dbl(p6_v), p5_v);
    s = add(s, add(p4_v, p3_v));
    s = add(s, add(dbl(p2_v), dbl(p1_v)));
    s = add(s, dbl(p0_v));
    s = add(s, add4(q0_v, q1_v, q2_v, q3_v));
    s = add(s, add(q4_v, c8));
    out[4] = _mm_srai_epi32::<4>(s);

    // out[-1] = (p6 + p5 + p4 + p3 + p2 + p1*2 + p0*2 + q0*2 + q1 + q2 + q3 + q4 + q5 + 8) >> 4
    let mut s = add(p6_v, p5_v);
    s = add(s, add(p4_v, p3_v));
    s = add(s, p2_v);
    s = add(s, add(dbl(p1_v), dbl(p0_v)));
    s = add(s, dbl(q0_v));
    s = add(s, add4(q1_v, q2_v, q3_v, q4_v));
    s = add(s, add(q5_v, c8));
    out[5] = _mm_srai_epi32::<4>(s);

    // out[ 0] = (p5 + p4 + p3 + p2 + p1 + p0*2 + q0*2 + q1*2 + q2 + q3 + q4 + q5 + q6 + 8) >> 4
    let mut s = add(p5_v, p4_v);
    s = add(s, add(p3_v, p2_v));
    s = add(s, p1_v);
    s = add(s, add(dbl(p0_v), dbl(q0_v)));
    s = add(s, dbl(q1_v));
    s = add(s, add4(q2_v, q3_v, q4_v, q5_v));
    s = add(s, add(q6_v, c8));
    out[6] = _mm_srai_epi32::<4>(s);

    // out[ 1] = (p4 + p3 + p2 + p1 + p0 + q0*2 + q1*2 + q2*2 + q3 + q4 + q5 + q6*2 + 8) >> 4
    let mut s = add(p4_v, p3_v);
    s = add(s, add(p2_v, p1_v));
    s = add(s, p0_v);
    s = add(s, add(dbl(q0_v), dbl(q1_v)));
    s = add(s, dbl(q2_v));
    s = add(s, add4(q3_v, q4_v, q5_v, q6_v));
    s = add(s, add(q6_v, c8));
    out[7] = _mm_srai_epi32::<4>(s);

    // out[ 2] = (p3 + p2 + p1 + p0 + q0 + q1*2 + q2*2 + q3*2 + q4 + q5 + q6*3 + 8) >> 4
    let mut s = add(p3_v, p2_v);
    s = add(s, add(p1_v, p0_v));
    s = add(s, q0_v);
    s = add(s, add(dbl(q1_v), dbl(q2_v)));
    s = add(s, dbl(q3_v));
    s = add(s, add3(q4_v, q5_v, add(dbl(q6_v), q6_v)));
    s = add(s, c8);
    out[8] = _mm_srai_epi32::<4>(s);

    // out[ 3] = (p2 + p1 + p0 + q0 + q1 + q2*2 + q3*2 + q4*2 + q5 + q6*4 + 8) >> 4
    let mut s = add(p2_v, p1_v);
    s = add(s, add(p0_v, q0_v));
    s = add(s, q1_v);
    s = add(s, add(dbl(q2_v), dbl(q3_v)));
    s = add(s, dbl(q4_v));
    s = add(s, add(q5_v, add(dbl(q6_v), dbl(q6_v))));
    s = add(s, c8);
    out[9] = _mm_srai_epi32::<4>(s);

    // out[ 4] = (p1 + p0 + q0 + q1 + q2 + q3*2 + q4*2 + q5*2 + q6*5 + 8) >> 4
    let mut s = add(p1_v, p0_v);
    s = add(s, add(q0_v, q1_v));
    s = add(s, q2_v);
    s = add(s, add(dbl(q3_v), dbl(q4_v)));
    s = add(s, dbl(q5_v));
    s = add(s, q6_5);
    s = add(s, c8);
    out[10] = _mm_srai_epi32::<4>(s);

    // out[ 5] = (p0 + q0 + q1 + q2 + q3 + q4*2 + q5*2 + q6*7 + 8) >> 4
    let mut s = add(p0_v, q0_v);
    s = add(s, add(q1_v, q2_v));
    s = add(s, q3_v);
    s = add(s, add(dbl(q4_v), dbl(q5_v)));
    s = add(s, add(q6_5, dbl(q6_v)));
    s = add(s, c8);
    out[11] = _mm_srai_epi32::<4>(s);

    out
}

/// Load 4 u16 pixels as 4 i32 lanes (8-byte load, exactly the tapped pixels —
/// no over-read past the mask-derived window).
#[cfg(target_arch = "x86_64")]
#[rite(v3)]
fn lf16_load4(buf: &[u16], start: usize) -> __m128i {
    let px: &[u16; 4] = buf[start..start + 4].try_into().unwrap();
    _mm_cvtepu16_epi32(mm_loadl_epi64(px))
}

/// Load 2 u16 pixels as 2 i32 lanes (zero-filled dead lanes 2..3) — the h-filter
/// tail-chunk guard for #524: lanes past the mask-derived window must never be
/// read because a concurrent tile worker may own them.
#[cfg(target_arch = "x86_64")]
#[rite(v3)]
fn lf16_load2(buf: &[u16], start: usize) -> __m128i {
    // One `try_into` range check → fixed [u16; 2] → single 4-byte load
    // (per-element indexing on the dynamic slice stays two trappable loads
    // that LLVM cannot merge).
    let px: &[u16; 2] = buf[start..start + 2].try_into().unwrap();
    let as_i64 = ((px[0] as u32) | ((px[1] as u32) << 16)) as i32 as i64;
    _mm_cvtepu16_epi32(_mm_cvtsi64_si128(as_i64))
}

/// Clamp 4 i32 lanes to [0, bd_max] and store as 4 u16 (8 bytes).
#[cfg(target_arch = "x86_64")]
#[rite(v3)]
fn lf16_store4(buf: &mut [u16], start: usize, v: __m128i, bd_max_v: __m128i) {
    let clipped = _mm_min_epi32(_mm_max_epi32(v, _mm_setzero_si128()), bd_max_v);
    let packed = _mm_packus_epi32(clipped, clipped);
    let dst: &mut [u16; 4] = (&mut buf[start..start + 4]).try_into().unwrap();
    mm_storel_epi64(dst, packed);
}

/// 4x4 i32 transpose shared by all 16bpc H-direction kernels.
#[cfg(target_arch = "x86_64")]
#[rite(v3)]
fn lf16_transpose4(r0: __m128i, r1: __m128i, r2: __m128i, r3: __m128i) -> [__m128i; 4] {
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
}

// ----------------------------------------------------------------------------
// SIMD kernels, 16bpc V-FILTER (stridea == 1, contiguous column loads)
// ----------------------------------------------------------------------------

/// SIMD narrow 4-tap loop filter for 16bpc V direction.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn loop_filter_4_16bpc_narrow_simd_v(
    _token: Desktop64,
    buf: &mut [u16],
    base: usize,
    e: i32,
    i: i32,
    h: i32,
    strideb: isize,
    bdm8: i32,
    bd_max: i32,
) {
    let p1_v = lf16_load4(buf, signed_idx(base, strideb * -2));
    let p0_v = lf16_load4(buf, signed_idx(base, strideb * -1));
    let q0_v = lf16_load4(buf, base);
    let q1_v = lf16_load4(buf, signed_idx(base, strideb));

    let i_v = _mm_set1_epi32(i);
    let e_v = _mm_set1_epi32(e);
    let h_v = _mm_set1_epi32(h);
    let neg = _mm_set1_epi32(-(128 << bdm8));
    let pos = _mm_set1_epi32((128 << bdm8) - 1);
    let bdv = _mm_set1_epi32(bd_max);

    let abs = |a: __m128i, b: __m128i| _mm_abs_epi32(_mm_sub_epi32(a, b));
    let abs_p1p0 = abs(p1_v, p0_v);
    let abs_q1q0 = abs(q1_v, q0_v);
    let abs_p0q0 = abs(p0_v, q0_v);
    let abs_p1q1 = abs(p1_v, q1_v);

    let not_gt = |a: __m128i, b: __m128i| -> __m128i {
        _mm_andnot_si128(_mm_cmpgt_epi32(a, b), _mm_set1_epi32(-1))
    };
    let val = _mm_add_epi32(_mm_slli_epi32::<1>(abs_p0q0), _mm_srli_epi32::<1>(abs_p1q1));
    let fm_mask = _mm_and_si128(
        _mm_and_si128(not_gt(abs_p1p0, i_v), not_gt(abs_q1q0, i_v)),
        not_gt(val, e_v),
    );

    let (_hev, np1, np0, nq0, nq1) =
        lf16_narrow_core(p1_v, p0_v, q0_v, q1_v, abs_p1p0, abs_q1q0, h_v, neg, pos);

    let blendv = |a: __m128i, b: __m128i, mask: __m128i| -> __m128i {
        _mm_or_si128(_mm_andnot_si128(mask, a), _mm_and_si128(mask, b))
    };
    lf16_store4(
        buf,
        signed_idx(base, strideb * -2),
        blendv(p1_v, np1, fm_mask),
        bdv,
    );
    lf16_store4(
        buf,
        signed_idx(base, strideb * -1),
        blendv(p0_v, np0, fm_mask),
        bdv,
    );
    lf16_store4(buf, base, blendv(q0_v, nq0, fm_mask), bdv);
    lf16_store4(
        buf,
        signed_idx(base, strideb),
        blendv(q1_v, nq1, fm_mask),
        bdv,
    );
}

/// SIMD wd=6 loop filter for 16bpc V direction (taps -3..=2, writes -2..=1).
#[cfg(target_arch = "x86_64")]
#[arcane]
fn loop_filter_4_16bpc_wd6_simd_v(
    _token: Desktop64,
    buf: &mut [u16],
    base: usize,
    e: i32,
    i: i32,
    h: i32,
    strideb: isize,
    bdm8: i32,
    bd_max: i32,
) {
    let p2_v = lf16_load4(buf, signed_idx(base, strideb * -3));
    let p1_v = lf16_load4(buf, signed_idx(base, strideb * -2));
    let p0_v = lf16_load4(buf, signed_idx(base, strideb * -1));
    let q0_v = lf16_load4(buf, base);
    let q1_v = lf16_load4(buf, signed_idx(base, strideb));
    let q2_v = lf16_load4(buf, signed_idx(base, strideb * 2));

    let i_v = _mm_set1_epi32(i);
    let e_v = _mm_set1_epi32(e);
    let h_v = _mm_set1_epi32(h);
    let f_v = _mm_set1_epi32(1 << bdm8);
    let neg = _mm_set1_epi32(-(128 << bdm8));
    let pos = _mm_set1_epi32((128 << bdm8) - 1);
    let bdv = _mm_set1_epi32(bd_max);

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
    let val_ee = _mm_add_epi32(_mm_slli_epi32::<1>(abs_p0q0), _mm_srli_epi32::<1>(abs_p1q1));
    let fm_mask = _mm_and_si128(
        _mm_and_si128(
            _mm_and_si128(not_gt(abs_p1p0, i_v), not_gt(abs_q1q0, i_v)),
            not_gt(val_ee, e_v),
        ),
        _mm_and_si128(not_gt(abs_p2p1, i_v), not_gt(abs_q2q1, i_v)),
    );

    let abs_p2p0 = abs(p2_v, p0_v);
    let abs_q2q0 = abs(q2_v, q0_v);
    let flat_mask = _mm_and_si128(
        _mm_and_si128(not_gt(abs_p2p0, f_v), not_gt(abs_p1p0, f_v)),
        _mm_and_si128(not_gt(abs_q1q0, f_v), not_gt(abs_q2q0, f_v)),
    );

    let [o_m2, o_m1, o_0, o_1] = lf16_tap6(p2_v, p1_v, p0_v, q0_v, q1_v, q2_v);
    let (_hev, np1, np0, nq0, nq1) =
        lf16_narrow_core(p1_v, p0_v, q0_v, q1_v, abs_p1p0, abs_q1q0, h_v, neg, pos);

    let blendv = |a: __m128i, b: __m128i, mask: __m128i| -> __m128i {
        _mm_or_si128(_mm_andnot_si128(mask, a), _mm_and_si128(mask, b))
    };
    let s_m2 = blendv(np1, o_m2, flat_mask);
    let s_m1 = blendv(np0, o_m1, flat_mask);
    let s_0 = blendv(nq0, o_0, flat_mask);
    let s_1 = blendv(nq1, o_1, flat_mask);

    lf16_store4(
        buf,
        signed_idx(base, strideb * -2),
        blendv(p1_v, s_m2, fm_mask),
        bdv,
    );
    lf16_store4(
        buf,
        signed_idx(base, strideb * -1),
        blendv(p0_v, s_m1, fm_mask),
        bdv,
    );
    lf16_store4(buf, base, blendv(q0_v, s_0, fm_mask), bdv);
    lf16_store4(
        buf,
        signed_idx(base, strideb),
        blendv(q1_v, s_1, fm_mask),
        bdv,
    );
}

/// SIMD wd=8 loop filter for 16bpc V direction (taps -4..=3, writes -3..=2).
#[cfg(target_arch = "x86_64")]
#[arcane]
fn loop_filter_4_16bpc_wd8_simd_v(
    _token: Desktop64,
    buf: &mut [u16],
    base: usize,
    e: i32,
    i: i32,
    h: i32,
    strideb: isize,
    bdm8: i32,
    bd_max: i32,
) {
    let p3_v = lf16_load4(buf, signed_idx(base, strideb * -4));
    let p2_v = lf16_load4(buf, signed_idx(base, strideb * -3));
    let p1_v = lf16_load4(buf, signed_idx(base, strideb * -2));
    let p0_v = lf16_load4(buf, signed_idx(base, strideb * -1));
    let q0_v = lf16_load4(buf, base);
    let q1_v = lf16_load4(buf, signed_idx(base, strideb));
    let q2_v = lf16_load4(buf, signed_idx(base, strideb * 2));
    let q3_v = lf16_load4(buf, signed_idx(base, strideb * 3));

    let i_v = _mm_set1_epi32(i);
    let e_v = _mm_set1_epi32(e);
    let h_v = _mm_set1_epi32(h);
    let f_v = _mm_set1_epi32(1 << bdm8);
    let neg = _mm_set1_epi32(-(128 << bdm8));
    let pos = _mm_set1_epi32((128 << bdm8) - 1);
    let bdv = _mm_set1_epi32(bd_max);

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
    let val_ee = _mm_add_epi32(_mm_slli_epi32::<1>(abs_p0q0), _mm_srli_epi32::<1>(abs_p1q1));
    let fm_mask = _mm_and_si128(
        _mm_and_si128(
            _mm_and_si128(not_gt(abs_p1p0, i_v), not_gt(abs_q1q0, i_v)),
            not_gt(val_ee, e_v),
        ),
        _mm_and_si128(
            _mm_and_si128(not_gt(abs_p2p1, i_v), not_gt(abs_q2q1, i_v)),
            _mm_and_si128(not_gt(abs_p3p2, i_v), not_gt(abs_q3q2, i_v)),
        ),
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

    let [o_m3, o_m2, o_m1, o_0, o_1, o_2] =
        lf16_tap8(p3_v, p2_v, p1_v, p0_v, q0_v, q1_v, q2_v, q3_v);
    let (_hev, np1, np0, nq0, nq1) =
        lf16_narrow_core(p1_v, p0_v, q0_v, q1_v, abs_p1p0, abs_q1q0, h_v, neg, pos);

    let blendv = |a: __m128i, b: __m128i, mask: __m128i| -> __m128i {
        _mm_or_si128(_mm_andnot_si128(mask, a), _mm_and_si128(mask, b))
    };
    // narrow touches only -2..=1; positions -3 and +2 keep originals there.
    let s_m3 = blendv(p2_v, o_m3, flat_mask);
    let s_m2 = blendv(np1, o_m2, flat_mask);
    let s_m1 = blendv(np0, o_m1, flat_mask);
    let s_0 = blendv(nq0, o_0, flat_mask);
    let s_1 = blendv(nq1, o_1, flat_mask);
    let s_2 = blendv(q2_v, o_2, flat_mask);

    lf16_store4(
        buf,
        signed_idx(base, strideb * -3),
        blendv(p2_v, s_m3, fm_mask),
        bdv,
    );
    lf16_store4(
        buf,
        signed_idx(base, strideb * -2),
        blendv(p1_v, s_m2, fm_mask),
        bdv,
    );
    lf16_store4(
        buf,
        signed_idx(base, strideb * -1),
        blendv(p0_v, s_m1, fm_mask),
        bdv,
    );
    lf16_store4(buf, base, blendv(q0_v, s_0, fm_mask), bdv);
    lf16_store4(
        buf,
        signed_idx(base, strideb),
        blendv(q1_v, s_1, fm_mask),
        bdv,
    );
    lf16_store4(
        buf,
        signed_idx(base, strideb * 2),
        blendv(q2_v, s_2, fm_mask),
        bdv,
    );
}

/// SIMD wd=16 loop filter for 16bpc V direction (taps -7..=6, writes -6..=5).
#[cfg(target_arch = "x86_64")]
#[arcane]
fn loop_filter_4_16bpc_wd16_simd_v(
    _token: Desktop64,
    buf: &mut [u16],
    base: usize,
    e: i32,
    i: i32,
    h: i32,
    strideb: isize,
    bdm8: i32,
    bd_max: i32,
) {
    let p6_v = lf16_load4(buf, signed_idx(base, strideb * -7));
    let p5_v = lf16_load4(buf, signed_idx(base, strideb * -6));
    let p4_v = lf16_load4(buf, signed_idx(base, strideb * -5));
    let p3_v = lf16_load4(buf, signed_idx(base, strideb * -4));
    let p2_v = lf16_load4(buf, signed_idx(base, strideb * -3));
    let p1_v = lf16_load4(buf, signed_idx(base, strideb * -2));
    let p0_v = lf16_load4(buf, signed_idx(base, strideb * -1));
    let q0_v = lf16_load4(buf, base);
    let q1_v = lf16_load4(buf, signed_idx(base, strideb));
    let q2_v = lf16_load4(buf, signed_idx(base, strideb * 2));
    let q3_v = lf16_load4(buf, signed_idx(base, strideb * 3));
    let q4_v = lf16_load4(buf, signed_idx(base, strideb * 4));
    let q5_v = lf16_load4(buf, signed_idx(base, strideb * 5));
    let q6_v = lf16_load4(buf, signed_idx(base, strideb * 6));

    let i_v = _mm_set1_epi32(i);
    let e_v = _mm_set1_epi32(e);
    let h_v = _mm_set1_epi32(h);
    let f_v = _mm_set1_epi32(1 << bdm8);
    let neg = _mm_set1_epi32(-(128 << bdm8));
    let pos = _mm_set1_epi32((128 << bdm8) - 1);
    let bdv = _mm_set1_epi32(bd_max);

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
    let val_ee = _mm_add_epi32(_mm_slli_epi32::<1>(abs_p0q0), _mm_srli_epi32::<1>(abs_p1q1));
    let fm_mask = _mm_and_si128(
        _mm_and_si128(
            _mm_and_si128(not_gt(abs_p1p0, i_v), not_gt(abs_q1q0, i_v)),
            not_gt(val_ee, e_v),
        ),
        _mm_and_si128(
            _mm_and_si128(not_gt(abs_p2p1, i_v), not_gt(abs_q2q1, i_v)),
            _mm_and_si128(not_gt(abs_p3p2, i_v), not_gt(abs_q3q2, i_v)),
        ),
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

    let wide = lf16_tap14(&[
        p6_v, p5_v, p4_v, p3_v, p2_v, p1_v, p0_v, q0_v, q1_v, q2_v, q3_v, q4_v, q5_v, q6_v,
    ]);
    let [
        o_m6,
        o_m5,
        o_m4,
        o_m3,
        o_m2,
        o_m1,
        o_0,
        o_1,
        o_2,
        o_3,
        o_4,
        o_5,
    ] = wide;

    let [o8_m3, o8_m2, o8_m1, o8_0, o8_1, o8_2] =
        lf16_tap8(p3_v, p2_v, p1_v, p0_v, q0_v, q1_v, q2_v, q3_v);
    let (_hev, np1, np0, nq0, nq1) =
        lf16_narrow_core(p1_v, p0_v, q0_v, q1_v, abs_p1p0, abs_q1q0, h_v, neg, pos);

    let blendv = |a: __m128i, b: __m128i, mask: __m128i| -> __m128i {
        _mm_or_si128(_mm_andnot_si128(mask, a), _mm_and_si128(mask, b))
    };
    let wide_mask = _mm_and_si128(flat8out_mask, flat8in_mask);

    let mid_m3 = blendv(p2_v, o8_m3, flat8in_mask);
    let mid_m2 = blendv(np1, o8_m2, flat8in_mask);
    let mid_m1 = blendv(np0, o8_m1, flat8in_mask);
    let mid_0 = blendv(nq0, o8_0, flat8in_mask);
    let mid_1 = blendv(nq1, o8_1, flat8in_mask);
    let mid_2 = blendv(q2_v, o8_2, flat8in_mask);

    let sel_m6 = blendv(p5_v, o_m6, wide_mask);
    let sel_m5 = blendv(p4_v, o_m5, wide_mask);
    let sel_m4 = blendv(p3_v, o_m4, wide_mask);
    let sel_m3 = blendv(mid_m3, o_m3, wide_mask);
    let sel_m2 = blendv(mid_m2, o_m2, wide_mask);
    let sel_m1 = blendv(mid_m1, o_m1, wide_mask);
    let sel_0 = blendv(mid_0, o_0, wide_mask);
    let sel_1 = blendv(mid_1, o_1, wide_mask);
    let sel_2 = blendv(mid_2, o_2, wide_mask);
    let sel_3 = blendv(q3_v, o_3, wide_mask);
    let sel_4 = blendv(q4_v, o_4, wide_mask);
    let sel_5 = blendv(q5_v, o_5, wide_mask);

    let finals = [
        blendv(p5_v, sel_m6, fm_mask),
        blendv(p4_v, sel_m5, fm_mask),
        blendv(p3_v, sel_m4, fm_mask),
        blendv(p2_v, sel_m3, fm_mask),
        blendv(p1_v, sel_m2, fm_mask),
        blendv(p0_v, sel_m1, fm_mask),
        blendv(q0_v, sel_0, fm_mask),
        blendv(q1_v, sel_1, fm_mask),
        blendv(q2_v, sel_2, fm_mask),
        blendv(q3_v, sel_3, fm_mask),
        blendv(q4_v, sel_4, fm_mask),
        blendv(q5_v, sel_5, fm_mask),
    ];
    for (k, v) in finals.iter().enumerate() {
        let off = k as isize - 6;
        lf16_store4(buf, signed_idx(base, strideb * off), *v, bdv);
    }
}

// ----------------------------------------------------------------------------
// SIMD kernels, 16bpc H-FILTER (stridea == stride, per-row chunk loads +
// 4x4 i32 transpose into pixel-position vectors)
// ----------------------------------------------------------------------------

/// SIMD narrow 4-tap loop filter for 16bpc H direction.
/// Each row contributes one 8-byte load at offset -2 = [p1, p0, q0, q1].
#[cfg(target_arch = "x86_64")]
#[arcane]
fn loop_filter_4_16bpc_narrow_simd_h(
    _token: Desktop64,
    buf: &mut [u16],
    base: usize,
    e: i32,
    i: i32,
    h: i32,
    stridea: isize,
    bdm8: i32,
    bd_max: i32,
) {
    let rows = [
        lf16_load4(buf, signed_idx(base, -2)),
        lf16_load4(buf, signed_idx(base, stridea - 2)),
        lf16_load4(buf, signed_idx(base, 2 * stridea - 2)),
        lf16_load4(buf, signed_idx(base, 3 * stridea - 2)),
    ];
    let [p1_v, p0_v, q0_v, q1_v] = lf16_transpose4(rows[0], rows[1], rows[2], rows[3]);

    let i_v = _mm_set1_epi32(i);
    let e_v = _mm_set1_epi32(e);
    let h_v = _mm_set1_epi32(h);
    let neg = _mm_set1_epi32(-(128 << bdm8));
    let pos = _mm_set1_epi32((128 << bdm8) - 1);
    let bdv = _mm_set1_epi32(bd_max);

    let abs = |a: __m128i, b: __m128i| _mm_abs_epi32(_mm_sub_epi32(a, b));
    let abs_p1p0 = abs(p1_v, p0_v);
    let abs_q1q0 = abs(q1_v, q0_v);
    let abs_p0q0 = abs(p0_v, q0_v);
    let abs_p1q1 = abs(p1_v, q1_v);

    let not_gt = |a: __m128i, b: __m128i| -> __m128i {
        _mm_andnot_si128(_mm_cmpgt_epi32(a, b), _mm_set1_epi32(-1))
    };
    let val = _mm_add_epi32(_mm_slli_epi32::<1>(abs_p0q0), _mm_srli_epi32::<1>(abs_p1q1));
    let fm_mask = _mm_and_si128(
        _mm_and_si128(not_gt(abs_p1p0, i_v), not_gt(abs_q1q0, i_v)),
        not_gt(val, e_v),
    );

    let (_hev, np1, np0, nq0, nq1) =
        lf16_narrow_core(p1_v, p0_v, q0_v, q1_v, abs_p1p0, abs_q1q0, h_v, neg, pos);

    let blendv = |a: __m128i, b: __m128i, mask: __m128i| -> __m128i {
        _mm_or_si128(_mm_andnot_si128(mask, a), _mm_and_si128(mask, b))
    };
    let f_p1 = blendv(p1_v, np1, fm_mask);
    let f_p0 = blendv(p0_v, np0, fm_mask);
    let f_q0 = blendv(q0_v, nq0, fm_mask);
    let f_q1 = blendv(q1_v, nq1, fm_mask);

    // Transpose back to row layout: [p1,p0,q0,q1] per row, store at offset -2.
    let back = lf16_transpose4(f_p1, f_p0, f_q0, f_q1);
    for (k, row) in back.iter().enumerate() {
        let start = signed_idx(base, k as isize * stridea - 2);
        lf16_store4(buf, start, *row, bdv);
    }
}

/// SIMD wd=6 loop filter for 16bpc H direction.
/// Per row: 8-byte load at -3 = [p2,p1,p0,q0] + 4-byte load at +1 = [q1,q2].
/// The +1 tail chunk pulls only 2 pixels — its lanes 2..3 would read +3/+4,
/// past the mask-derived window a concurrent tile worker may own (#524).
#[cfg(target_arch = "x86_64")]
#[arcane]
fn loop_filter_4_16bpc_wd6_simd_h(
    _token: Desktop64,
    buf: &mut [u16],
    base: usize,
    e: i32,
    i: i32,
    h: i32,
    stridea: isize,
    bdm8: i32,
    bd_max: i32,
) {
    let lo = lf16_transpose4(
        lf16_load4(buf, signed_idx(base, -3)),
        lf16_load4(buf, signed_idx(base, stridea - 3)),
        lf16_load4(buf, signed_idx(base, 2 * stridea - 3)),
        lf16_load4(buf, signed_idx(base, 3 * stridea - 3)),
    );
    let hi = lf16_transpose4(
        lf16_load2(buf, signed_idx(base, 1)),
        lf16_load2(buf, signed_idx(base, stridea + 1)),
        lf16_load2(buf, signed_idx(base, 2 * stridea + 1)),
        lf16_load2(buf, signed_idx(base, 3 * stridea + 1)),
    );
    let p2_v = lo[0];
    let p1_v = lo[1];
    let p0_v = lo[2];
    let q0_v = lo[3];
    let q1_v = hi[0];
    let q2_v = hi[1];

    let i_v = _mm_set1_epi32(i);
    let e_v = _mm_set1_epi32(e);
    let h_v = _mm_set1_epi32(h);
    let f_v = _mm_set1_epi32(1 << bdm8);
    let neg = _mm_set1_epi32(-(128 << bdm8));
    let pos = _mm_set1_epi32((128 << bdm8) - 1);
    let bdv = _mm_set1_epi32(bd_max);

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
    let val_ee = _mm_add_epi32(_mm_slli_epi32::<1>(abs_p0q0), _mm_srli_epi32::<1>(abs_p1q1));
    let fm_mask = _mm_and_si128(
        _mm_and_si128(
            _mm_and_si128(not_gt(abs_p1p0, i_v), not_gt(abs_q1q0, i_v)),
            not_gt(val_ee, e_v),
        ),
        _mm_and_si128(not_gt(abs_p2p1, i_v), not_gt(abs_q2q1, i_v)),
    );

    let abs_p2p0 = abs(p2_v, p0_v);
    let abs_q2q0 = abs(q2_v, q0_v);
    let flat_mask = _mm_and_si128(
        _mm_and_si128(not_gt(abs_p2p0, f_v), not_gt(abs_p1p0, f_v)),
        _mm_and_si128(not_gt(abs_q1q0, f_v), not_gt(abs_q2q0, f_v)),
    );

    let [o_m2, o_m1, o_0, o_1] = lf16_tap6(p2_v, p1_v, p0_v, q0_v, q1_v, q2_v);
    let (_hev, np1, np0, nq0, nq1) =
        lf16_narrow_core(p1_v, p0_v, q0_v, q1_v, abs_p1p0, abs_q1q0, h_v, neg, pos);

    let blendv = |a: __m128i, b: __m128i, mask: __m128i| -> __m128i {
        _mm_or_si128(_mm_andnot_si128(mask, a), _mm_and_si128(mask, b))
    };
    let f_p1 = blendv(p1_v, blendv(np1, o_m2, flat_mask), fm_mask);
    let f_p0 = blendv(p0_v, blendv(np0, o_m1, flat_mask), fm_mask);
    let f_q0 = blendv(q0_v, blendv(nq0, o_0, flat_mask), fm_mask);
    let f_q1 = blendv(q1_v, blendv(nq1, o_1, flat_mask), fm_mask);

    let back = lf16_transpose4(f_p1, f_p0, f_q0, f_q1);
    for (k, row) in back.iter().enumerate() {
        let start = signed_idx(base, k as isize * stridea - 2);
        lf16_store4(buf, start, *row, bdv);
    }
}

/// SIMD wd=8 loop filter for 16bpc H direction.
/// Per row: two 8-byte loads at -4 and 0 covering p3..q3 exactly.
/// Writes -3..=2 (6 px); positions -4/+3 are re-stored with their original
/// values so each row's write window stays 4-aligned.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn loop_filter_4_16bpc_wd8_simd_h(
    _token: Desktop64,
    buf: &mut [u16],
    base: usize,
    e: i32,
    i: i32,
    h: i32,
    stridea: isize,
    bdm8: i32,
    bd_max: i32,
) {
    let lo = lf16_transpose4(
        lf16_load4(buf, signed_idx(base, -4)),
        lf16_load4(buf, signed_idx(base, stridea - 4)),
        lf16_load4(buf, signed_idx(base, 2 * stridea - 4)),
        lf16_load4(buf, signed_idx(base, 3 * stridea - 4)),
    );
    let hi = lf16_transpose4(
        lf16_load4(buf, base),
        lf16_load4(buf, signed_idx(base, stridea)),
        lf16_load4(buf, signed_idx(base, 2 * stridea)),
        lf16_load4(buf, signed_idx(base, 3 * stridea)),
    );
    let p3_v = lo[0];
    let p2_v = lo[1];
    let p1_v = lo[2];
    let p0_v = lo[3];
    let q0_v = hi[0];
    let q1_v = hi[1];
    let q2_v = hi[2];
    let q3_v = hi[3];

    let i_v = _mm_set1_epi32(i);
    let e_v = _mm_set1_epi32(e);
    let h_v = _mm_set1_epi32(h);
    let f_v = _mm_set1_epi32(1 << bdm8);
    let neg = _mm_set1_epi32(-(128 << bdm8));
    let pos = _mm_set1_epi32((128 << bdm8) - 1);
    let bdv = _mm_set1_epi32(bd_max);

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
    let val_ee = _mm_add_epi32(_mm_slli_epi32::<1>(abs_p0q0), _mm_srli_epi32::<1>(abs_p1q1));
    let fm_mask = _mm_and_si128(
        _mm_and_si128(
            _mm_and_si128(not_gt(abs_p1p0, i_v), not_gt(abs_q1q0, i_v)),
            not_gt(val_ee, e_v),
        ),
        _mm_and_si128(
            _mm_and_si128(not_gt(abs_p2p1, i_v), not_gt(abs_q2q1, i_v)),
            _mm_and_si128(not_gt(abs_p3p2, i_v), not_gt(abs_q3q2, i_v)),
        ),
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

    let [o_m3, o_m2, o_m1, o_0, o_1, o_2] =
        lf16_tap8(p3_v, p2_v, p1_v, p0_v, q0_v, q1_v, q2_v, q3_v);
    let (_hev, np1, np0, nq0, nq1) =
        lf16_narrow_core(p1_v, p0_v, q0_v, q1_v, abs_p1p0, abs_q1q0, h_v, neg, pos);

    let blendv = |a: __m128i, b: __m128i, mask: __m128i| -> __m128i {
        _mm_or_si128(_mm_andnot_si128(mask, a), _mm_and_si128(mask, b))
    };
    let f_p2 = blendv(p2_v, blendv(p2_v, o_m3, flat_mask), fm_mask);
    let f_p1 = blendv(p1_v, blendv(np1, o_m2, flat_mask), fm_mask);
    let f_p0 = blendv(p0_v, blendv(np0, o_m1, flat_mask), fm_mask);
    let f_q0 = blendv(q0_v, blendv(nq0, o_0, flat_mask), fm_mask);
    let f_q1 = blendv(q1_v, blendv(nq1, o_1, flat_mask), fm_mask);
    let f_q2 = blendv(q2_v, blendv(q2_v, o_2, flat_mask), fm_mask);

    // Write back the full -4..=3 span per row (p3/q3 keep original pixels —
    // same behavior as the 8bpc wd8 h-kernel, keeps stores 4-aligned).
    let back_lo = lf16_transpose4(p3_v, f_p2, f_p1, f_p0);
    let back_hi = lf16_transpose4(f_q0, f_q1, f_q2, q3_v);
    for k in 0..4isize {
        lf16_store4(
            buf,
            signed_idx(base, k * stridea - 4),
            back_lo[k as usize],
            bdv,
        );
        lf16_store4(buf, signed_idx(base, k * stridea), back_hi[k as usize], bdv);
    }
}

/// SIMD wd=16 loop filter for 16bpc H direction.
/// Per row: 8-byte loads at -7, -3, +1 and a guarded 4-byte load at +5
/// (only q5/q6 are real; lanes 2..3 would read +7/+8 — past the window, #524).
/// Writes -6..=5; endpoints -7/+6 keep original pixels for aligned stores.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn loop_filter_4_16bpc_wd16_simd_h(
    _token: Desktop64,
    buf: &mut [u16],
    base: usize,
    e: i32,
    i: i32,
    h: i32,
    stridea: isize,
    bdm8: i32,
    bd_max: i32,
) {
    let c0 = lf16_transpose4(
        lf16_load4(buf, signed_idx(base, -7)),
        lf16_load4(buf, signed_idx(base, stridea - 7)),
        lf16_load4(buf, signed_idx(base, 2 * stridea - 7)),
        lf16_load4(buf, signed_idx(base, 3 * stridea - 7)),
    );
    let c1 = lf16_transpose4(
        lf16_load4(buf, signed_idx(base, -3)),
        lf16_load4(buf, signed_idx(base, stridea - 3)),
        lf16_load4(buf, signed_idx(base, 2 * stridea - 3)),
        lf16_load4(buf, signed_idx(base, 3 * stridea - 3)),
    );
    let c2 = lf16_transpose4(
        lf16_load4(buf, signed_idx(base, 1)),
        lf16_load4(buf, signed_idx(base, stridea + 1)),
        lf16_load4(buf, signed_idx(base, 2 * stridea + 1)),
        lf16_load4(buf, signed_idx(base, 3 * stridea + 1)),
    );
    let c3 = lf16_transpose4(
        lf16_load2(buf, signed_idx(base, 5)),
        lf16_load2(buf, signed_idx(base, stridea + 5)),
        lf16_load2(buf, signed_idx(base, 2 * stridea + 5)),
        lf16_load2(buf, signed_idx(base, 3 * stridea + 5)),
    );
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

    let i_v = _mm_set1_epi32(i);
    let e_v = _mm_set1_epi32(e);
    let h_v = _mm_set1_epi32(h);
    let f_v = _mm_set1_epi32(1 << bdm8);
    let neg = _mm_set1_epi32(-(128 << bdm8));
    let pos = _mm_set1_epi32((128 << bdm8) - 1);
    let bdv = _mm_set1_epi32(bd_max);

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
    let val_ee = _mm_add_epi32(_mm_slli_epi32::<1>(abs_p0q0), _mm_srli_epi32::<1>(abs_p1q1));
    let fm_mask = _mm_and_si128(
        _mm_and_si128(
            _mm_and_si128(not_gt(abs_p1p0, i_v), not_gt(abs_q1q0, i_v)),
            not_gt(val_ee, e_v),
        ),
        _mm_and_si128(
            _mm_and_si128(not_gt(abs_p2p1, i_v), not_gt(abs_q2q1, i_v)),
            _mm_and_si128(not_gt(abs_p3p2, i_v), not_gt(abs_q3q2, i_v)),
        ),
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

    let [
        o_m6,
        o_m5,
        o_m4,
        o_m3,
        o_m2,
        o_m1,
        o_0,
        o_1,
        o_2,
        o_3,
        o_4,
        o_5,
    ] = lf16_tap14(&[
        p6_v, p5_v, p4_v, p3_v, p2_v, p1_v, p0_v, q0_v, q1_v, q2_v, q3_v, q4_v, q5_v, q6_v,
    ]);
    let [o8_m3, o8_m2, o8_m1, o8_0, o8_1, o8_2] =
        lf16_tap8(p3_v, p2_v, p1_v, p0_v, q0_v, q1_v, q2_v, q3_v);
    let (_hev, np1, np0, nq0, nq1) =
        lf16_narrow_core(p1_v, p0_v, q0_v, q1_v, abs_p1p0, abs_q1q0, h_v, neg, pos);

    let blendv = |a: __m128i, b: __m128i, mask: __m128i| -> __m128i {
        _mm_or_si128(_mm_andnot_si128(mask, a), _mm_and_si128(mask, b))
    };
    let wide_mask = _mm_and_si128(flat8out_mask, flat8in_mask);

    let mid_m3 = blendv(p2_v, o8_m3, flat8in_mask);
    let mid_m2 = blendv(np1, o8_m2, flat8in_mask);
    let mid_m1 = blendv(np0, o8_m1, flat8in_mask);
    let mid_0 = blendv(nq0, o8_0, flat8in_mask);
    let mid_1 = blendv(nq1, o8_1, flat8in_mask);
    let mid_2 = blendv(q2_v, o8_2, flat8in_mask);

    let sel_m6 = blendv(p5_v, o_m6, wide_mask);
    let sel_m5 = blendv(p4_v, o_m5, wide_mask);
    let sel_m4 = blendv(p3_v, o_m4, wide_mask);
    let sel_m3 = blendv(mid_m3, o_m3, wide_mask);
    let sel_m2 = blendv(mid_m2, o_m2, wide_mask);
    let sel_m1 = blendv(mid_m1, o_m1, wide_mask);
    let sel_0 = blendv(mid_0, o_0, wide_mask);
    let sel_1 = blendv(mid_1, o_1, wide_mask);
    let sel_2 = blendv(mid_2, o_2, wide_mask);
    let sel_3 = blendv(q3_v, o_3, wide_mask);
    let sel_4 = blendv(q4_v, o_4, wide_mask);
    let sel_5 = blendv(q5_v, o_5, wide_mask);

    // Position -> final vector, indices -6..=5.
    let finals = [
        blendv(p5_v, sel_m6, fm_mask),
        blendv(p4_v, sel_m5, fm_mask),
        blendv(p3_v, sel_m4, fm_mask),
        blendv(p2_v, sel_m3, fm_mask),
        blendv(p1_v, sel_m2, fm_mask),
        blendv(p0_v, sel_m1, fm_mask),
        blendv(q0_v, sel_0, fm_mask),
        blendv(q1_v, sel_1, fm_mask),
        blendv(q2_v, sel_2, fm_mask),
        blendv(q3_v, sel_3, fm_mask),
        blendv(q4_v, sel_4, fm_mask),
        blendv(q5_v, sel_5, fm_mask),
    ];
    // Transpose back in three groups of 4 positions: each group's rows hold
    // [pos, pos+1, pos+2, pos+3] for one filter row. Stores cover exactly the
    // -6..=+5 written span (12 px, three 8-byte chunks).
    let back_a = lf16_transpose4(finals[0], finals[1], finals[2], finals[3]);
    let back_b = lf16_transpose4(finals[4], finals[5], finals[6], finals[7]);
    let back_c = lf16_transpose4(finals[8], finals[9], finals[10], finals[11]);
    for k in 0..4isize {
        let start = signed_idx(base, k * stridea - 6);
        lf16_store4(buf, start, back_a[k as usize], bdv);
        lf16_store4(buf, start + 4, back_b[k as usize], bdv);
        lf16_store4(buf, start + 8, back_c[k as usize], bdv);
    }
}

/// Core loop filter for 16bpc - processes 4 pixels
#[cfg(any(target_arch = "x86_64", target_arch = "wasm32"))]
#[cfg_attr(target_arch = "x86_64", rite)]
fn loop_filter_4_16bpc(
    #[cfg(target_arch = "x86_64")] _token: Desktop64,
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

    // SIMD fast paths — same mask math as scalar, 4 i32 lanes on widened u16.
    // Loads touch exactly the mask-derived window (h kernels chunk + guard
    // tails like the 8bpc #524 fix), stores clamp to [0, bitdepth_max].
    #[cfg(target_arch = "x86_64")]
    if stridea == 1 {
        match wd {
            4 => {
                loop_filter_4_16bpc_narrow_simd_v(
                    _token,
                    buf,
                    base,
                    e,
                    i,
                    h,
                    strideb,
                    bitdepth_min_8,
                    bitdepth_max,
                );
                return;
            }
            6 => {
                loop_filter_4_16bpc_wd6_simd_v(
                    _token,
                    buf,
                    base,
                    e,
                    i,
                    h,
                    strideb,
                    bitdepth_min_8,
                    bitdepth_max,
                );
                return;
            }
            8 => {
                loop_filter_4_16bpc_wd8_simd_v(
                    _token,
                    buf,
                    base,
                    e,
                    i,
                    h,
                    strideb,
                    bitdepth_min_8,
                    bitdepth_max,
                );
                return;
            }
            16 => {
                loop_filter_4_16bpc_wd16_simd_v(
                    _token,
                    buf,
                    base,
                    e,
                    i,
                    h,
                    strideb,
                    bitdepth_min_8,
                    bitdepth_max,
                );
                return;
            }
            _ => {}
        }
    }
    #[cfg(target_arch = "x86_64")]
    if strideb == 1 && stridea != 1 {
        match wd {
            4 => {
                loop_filter_4_16bpc_narrow_simd_h(
                    _token,
                    buf,
                    base,
                    e,
                    i,
                    h,
                    stridea,
                    bitdepth_min_8,
                    bitdepth_max,
                );
                return;
            }
            6 => {
                loop_filter_4_16bpc_wd6_simd_h(
                    _token,
                    buf,
                    base,
                    e,
                    i,
                    h,
                    stridea,
                    bitdepth_min_8,
                    bitdepth_max,
                );
                return;
            }
            8 => {
                loop_filter_4_16bpc_wd8_simd_h(
                    _token,
                    buf,
                    base,
                    e,
                    i,
                    h,
                    stridea,
                    bitdepth_min_8,
                    bitdepth_max,
                );
                return;
            }
            16 => {
                loop_filter_4_16bpc_wd16_simd_h(
                    _token,
                    buf,
                    base,
                    e,
                    i,
                    h,
                    stridea,
                    bitdepth_min_8,
                    bitdepth_max,
                );
                return;
            }
            _ => {}
        }
    }

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
#[cfg_attr(target_arch = "x86_64", arcane)]
#[allow(unused_mut)]
#[cfg_attr(not(target_arch = "x86_64"), allow(unused_variables))]
fn lpf_h_sb_y_16bpc_inner(
    #[cfg(target_arch = "x86_64")] _token: Desktop64,
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

/// Loop filter Y vertical 16bpc inner
#[cfg(any(target_arch = "x86_64", target_arch = "wasm32"))]
#[cfg_attr(target_arch = "x86_64", arcane)]
#[allow(unused_mut)]
#[cfg_attr(not(target_arch = "x86_64"), allow(unused_variables))]
fn lpf_v_sb_y_16bpc_inner(
    #[cfg(target_arch = "x86_64")] _token: Desktop64,
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

/// Loop filter UV horizontal 16bpc inner
#[cfg(any(target_arch = "x86_64", target_arch = "wasm32"))]
#[cfg_attr(target_arch = "x86_64", arcane)]
#[allow(unused_mut)]
#[cfg_attr(not(target_arch = "x86_64"), allow(unused_variables))]
fn lpf_h_sb_uv_16bpc_inner(
    #[cfg(target_arch = "x86_64")] _token: Desktop64,
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

/// Loop filter UV vertical 16bpc inner
#[cfg(any(target_arch = "x86_64", target_arch = "wasm32"))]
#[cfg_attr(target_arch = "x86_64", arcane)]
#[allow(unused_mut)]
#[cfg_attr(not(target_arch = "x86_64"), allow(unused_variables))]
fn lpf_v_sb_uv_16bpc_inner(
    #[cfg(target_arch = "x86_64")] _token: Desktop64,
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
    // See the AUDITED note on the 8bpc FFI-wrapper banner above.
    let token = Desktop64::summon().expect(
        "x86-64-v3 (Desktop64) token required; #[target_feature(avx2)] alone does not imply it",
    );
    lpf_h_sb_y_16bpc_inner(
        token,
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
    // See the AUDITED note on the 8bpc FFI-wrapper banner above.
    let token = Desktop64::summon().expect(
        "x86-64-v3 (Desktop64) token required; #[target_feature(avx2)] alone does not imply it",
    );
    lpf_v_sb_y_16bpc_inner(
        token,
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
    // See the AUDITED note on the 8bpc FFI-wrapper banner above.
    let token = Desktop64::summon().expect(
        "x86-64-v3 (Desktop64) token required; #[target_feature(avx2)] alone does not imply it",
    );
    lpf_h_sb_uv_16bpc_inner(
        token,
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
    // See the AUDITED note on the 8bpc FFI-wrapper banner above.
    let token = Desktop64::summon().expect(
        "x86-64-v3 (Desktop64) token required; #[target_feature(avx2)] alone does not imply it",
    );
    lpf_v_sb_uv_16bpc_inner(
        token,
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
            //   backward: 7 (luma) / 3 (chroma) pixels horizontally
            // V filter (is_v=true): iterates columns (stridea=1), row access (strideb=stride)
            //   forward: (max_iter*4-1) columns + tap_after*stride rows
            //   backward: 7 (luma) / 3 (chroma) rows
            //
            // The backward reach must match the plane's true tap span (luma
            // wd16 reads p6 at -7; chroma wd6 reads p2 at -3): dav1d's CDEF
            // lag ahead of the deblock task is 8 luma / 4 chroma rows, so a
            // uniform 7-row window over-reads rows 4..=7 above a chroma edge
            // — rows the previous sbrow's CDEF task is still legitimately
            // writing. Guarding them races that task (zenavif#30).
            let tap_before = if is_y { 7 } else { 3 };
            // Reach PAST the edge (#457). The tap ladder is symmetric — reads
            // span p6..q6 for luma wd16 (7 after, indices 0..=+6) and p2..q2
            // for chroma wd6 (3 after) — and since #524 every 8bpc kernel's
            // LOADS stop there too, in both directions. The H values below stay
            // at the old 9/5 anyway: they used to be the 4-byte chunked
            // transpose loads' rounding (wd16 covered -7..8, wd6 +1..+4), and
            // keeping them means this fallback predicate accepts exactly the
            // edges it accepted before, so the SIMD-vs-scalar decision — and
            // therefore every output byte — is unchanged by that fix. They are
            // a deliberate superset of the reach, not a claim about it; the
            // reach itself is `lf_run_reach`, which is what sizes the window.
            //
            // The previous "+16" reached a full 16 rows below a V edge — into
            // the NEXT superblock row, whose reconstruction runs concurrently
            // now that the deblock barrier is gone; compact_read_per_row
            // genuinely memcpys the window, so that read RACED the neighbour's
            // BlockMut write-back (checked builds: intermittent overlap panics
            // at t>=2). Exact windows stay inside the rows/cols the task graph
            // already orders.
            let tap_after = if is_y {
                if is_v { 7 } else { 9 }
            } else if is_v {
                3
            } else {
                5
            };
            let (reach_before, reach_after) = if !is_v {
                // H filter: iterates through row groups
                (tap_before, (max_iter * 4 - 1) * byte_stride + tap_after)
            } else {
                // V filter: iterates through column groups
                (
                    tap_before * byte_stride,
                    max_iter * 4 - 1 + tap_after * byte_stride,
                )
            };

            // Guard: fall back to scalar if buffer bounds are insufficient.
            // Deliberately the PLANE worst case above, not the per-run window
            // below: keeping this predicate on `tap_*` keeps the SIMD-vs-scalar
            // decision — and therefore every output byte — exactly what it was
            // before the window narrowed. Narrowing the window can only make
            // the guard a subset of what this test already proved in bounds.
            let buf_pixel_len = dst.data.pixel_len::<BitDepth8>();
            if dst.offset < reach_before || dst.offset.saturating_add(reach_after) > buf_pixel_len {
                return false;
            }

            // The window the guard and the copy actually cover (#494, #524).
            //
            // Both directions size from the tap reach of the widest width THIS
            // RUN's mask can select, not the widest the plane allows;
            // `lf_compact_window` and `lf_run_reach` carry the proof for each.
            //
            // For V (horizontal edges) a mask-derived window cannot read past
            // the superblock row, where `tap_before/tap_after` of 7 reads 3
            // rows past it at every level-0 edge in the last 4-row band. Those
            // rows belong to the tile worker reconstructing the next superblock
            // row (`owned_recon.rs::stitch_sbrow`) and to that row's own
            // DeblockCols task, both of which run concurrently at `t > 1` with
            // no deblock barrier — an OBSERVED read/write race on x86_64,
            // where this dispatcher owns the guard policy that
            // `LfBlock::open` owns on aarch64 (which is why aarch64 is clean:
            // it sizes from the group's own `wd`).
            //
            // For H (vertical edges) a mask-derived window cannot read past the
            // end of its own picture row, where `tap_after` of 5 reads one
            // column past it at a chroma edge in the last 4-column group of a
            // 384-stride plane — and that column is the next row's first pixel,
            // which the next superblock row's stitch is writing (#524).
            let (win_before, win_after) = {
                let r = crate::src::loopfilter::lf_run_reach(is_y, mask);
                (r, r)
            };
            let (win_reach_before, win_reach_after) = if !is_v {
                (win_before, (max_iter * 4 - 1) * byte_stride + win_after)
            } else {
                (
                    win_before * byte_stride,
                    max_iter * 4 - 1 + win_after * byte_stride,
                )
            };

            // COW: single-threaded uses the original wide guard (zero-copy),
            // multi-threaded decomposes into a 2D compact buffer with per-row guards.
            let use_compact = dst.data.uses_row_guards();

            let start_pixel = dst.offset - win_reach_before;
            let total_pixels =
                (win_reach_before + win_reach_after).min(buf_pixel_len - start_pixel);

            if use_compact {
                // The single source of truth for this geometry, so the window
                // the guards reserve is the one `src/loopfilter.rs`'s unit
                // tests check for row/superblock-row containment (#524).
                let (cw, ch, cstart, cbase) = crate::src::loopfilter::lf_compact_window(
                    is_v,
                    is_y,
                    mask,
                    max_iter,
                    dst.offset,
                    byte_stride,
                );
                let lpf_pic = crate::src::with_offset::WithOffset {
                    data: dst.data,
                    offset: cstart,
                };
                let (mut cb, cs) = lpf_pic.compact_read_per_row::<BitDepth8>(cw, ch);
                // Pristine copy for the diff write-back: the filter READS 7
                // tap rows/cols beyond the ≤6 it can modify, and only the
                // modified pixels may be written (or mutably guarded) back —
                // see `compact_write_back_per_row_diff` (zenavif#30).
                let mut pristine = crate::include::dav1d::picture::take_compact_scratch();
                pristine.clear();
                pristine.extend_from_slice(&cb);
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
                lpf_pic.compact_write_back_per_row_diff::<BitDepth8>(cw, ch, &cb, &pristine);
                crate::include::dav1d::picture::recycle_compact_scratch(cb);
                crate::include::dav1d::picture::recycle_compact_scratch(pristine);
            } else {
                let mut guard = dst
                    .data
                    .slice_mut::<BitDepth8, _>((start_pixel.., ..total_pixels));
                let buf: &mut [u8] = &mut *guard;
                let base = win_reach_before;
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

            // Compute reach based on filter direction and actual vmask extent.
            // Backward reach matches the plane's true tap span — see the
            // 8bpc path (zenavif#30): a uniform 7 over-reads chroma rows the
            // previous sbrow's CDEF task is still writing.
            let tap_before = if is_y { 7 } else { 3 };
            // Exact after-the-edge reach (#457): the symmetric tap ladder
            // reads p6..q6 luma / p2..q2 chroma. The 16bpc kernels are pure
            // scalar — no chunked loads — so there is no rounding past it in
            // either direction.
            let tap_after = if is_y { 7 } else { 3 };
            let (reach_before, reach_after) = if !is_v {
                // H filter: iterates through row groups
                (tap_before, (max_iter * 4 - 1) * u16_stride + tap_after)
            } else {
                // V filter: iterates through column groups
                (
                    tap_before * u16_stride,
                    max_iter * 4 - 1 + tap_after * u16_stride,
                )
            };

            // Guard: fall back to scalar if buffer bounds are insufficient.
            // On the PLANE worst case, not the per-run window below, so the
            // SIMD-vs-scalar decision is bit-for-bit what it was — see the
            // 8bpc arm.
            let buf_pixel_len = dst.data.pixel_len::<BitDepth16>();
            if dst.offset < reach_before || dst.offset.saturating_add(reach_after) > buf_pixel_len {
                return false;
            }

            // Per-run window (#494, #524). Same rule and same reason as the
            // 8bpc arm: the extent is the reach of the widest width this run's
            // mask can select, so a V window cannot read past the superblock
            // row and an H window cannot read past the end of its picture row,
            // both of which are concurrently reconstructed at `t > 1`.
            let (win_before, win_after) = {
                let r = crate::src::loopfilter::lf_run_reach(is_y, mask);
                (r, r)
            };
            let (win_reach_before, win_reach_after) = if !is_v {
                (win_before, (max_iter * 4 - 1) * u16_stride + win_after)
            } else {
                (
                    win_before * u16_stride,
                    max_iter * 4 - 1 + win_after * u16_stride,
                )
            };

            // COW: single-threaded uses the original wide guard (zero-copy),
            // multi-threaded decomposes into a 2D compact buffer with per-row guards.
            let use_compact = dst.data.uses_row_guards();

            if use_compact {
                // Same single source of truth as the 8bpc arm (#524).
                let (compact_w, compact_h, start_pixel, base) =
                    crate::src::loopfilter::lf_compact_window(
                        is_v, is_y, mask, max_iter, dst.offset, u16_stride,
                    );
                let lpf_pic = crate::src::with_offset::WithOffset {
                    data: dst.data,
                    offset: start_pixel,
                };
                let (mut compact, compact_stride) =
                    lpf_pic.compact_read_per_row::<BitDepth16>(compact_w, compact_h);
                // Pristine copy for the diff write-back — see the 8bpc path
                // and `compact_write_back_per_row_diff` (zenavif#30).
                let mut pristine = crate::include::dav1d::picture::take_compact_scratch();
                pristine.clear();
                pristine.extend_from_slice(&compact);
                let buf: &mut [u16] =
                    zerocopy::FromBytes::mut_from_bytes(&mut compact[..]).unwrap();
                let stride_i = (compact_stride / 2) as isize;

                match (is_y, is_v) {
                    (true, false) => lpf_h_sb_y_16bpc_inner(
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
                    (true, true) => lpf_v_sb_y_16bpc_inner(
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
                    (false, false) => lpf_h_sb_uv_16bpc_inner(
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
                    (false, true) => lpf_v_sb_uv_16bpc_inner(
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
                lpf_pic.compact_write_back_per_row_diff::<BitDepth16>(
                    compact_w, compact_h, &compact, &pristine,
                );
                crate::include::dav1d::picture::recycle_compact_scratch(compact);
                crate::include::dav1d::picture::recycle_compact_scratch(pristine);
            } else {
                let start_pixel = dst.offset - win_reach_before;
                let total_pixels =
                    (win_reach_before + win_reach_after).min(buf_pixel_len - start_pixel);
                let mut guard = dst
                    .data
                    .slice_mut::<BitDepth16, _>((start_pixel.., ..total_pixels));
                let buf: &mut [u16] = &mut *guard;
                let base = win_reach_before;
                let stride_i = stride as isize / 2;

                match (is_y, is_v) {
                    (true, false) => lpf_h_sb_y_16bpc_inner(
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
                    (true, true) => lpf_v_sb_y_16bpc_inner(
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
                    (false, false) => lpf_h_sb_uv_16bpc_inner(
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
                    (false, true) => lpf_v_sb_uv_16bpc_inner(
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
    }
    true
}

/// Safe dispatch for loopfilter_sb on wasm32. Returns true if handled.
///
/// The inner filter functions are scalar (no SIMD intrinsics). The `&[AtomicU8]`
/// level cache is passed directly to inner functions which load entries on demand.
///
/// # KNOWN HAZARD, NOT FIXED: this window is wider than #494's was
///
/// It takes ONE **mutable** guard over `reach_before = 7 * stride` ..
/// `reach_after = ... + 16 * stride`, i.e. up to **16 rows below** a V edge,
/// where the filter reads at most `lf_reach(wd)` = 7 and usually 2. That is
/// wider than the constant-7 x86 window that raced concurrent reconstruction at
/// t=8 (#494), and mutable rather than immutable, so under wasm threads it
/// would conflict rather than merely over-read. It is left alone deliberately:
/// nothing in this campaign can execute wasm32 threads, and here `reach_after`
/// is BOTH the window and the scalar-fallback predicate, so narrowing it
/// without the `win_*` / `reach_*` split the x86 arms now have would change
/// which runs fall back to scalar — i.e. possibly change wasm output bytes,
/// unverifiably.
///
/// To fix it, copy the x86 arms: keep this predicate on the plane worst case,
/// add `win_before`/`win_after` from `crate::src::loopfilter::lf_run_reach` for
/// the V window, and re-run the corpus on a wasm host. The `debug_assert!` in
/// `loopfilter_sb_direct` already covers the invariant on every architecture,
/// so a wasm debug build would report a violation.
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

include!("loopfilter_parity.rs");
include!("loopfilter_packed6.rs");

include!("loopfilter_packed16.rs");
