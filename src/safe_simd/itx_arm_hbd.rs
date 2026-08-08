//! High-bit-depth (10/12-bit) inverse transforms on aarch64, in 32-bit lanes.
//!
//! # Why this module exists separately from `itx_arm_neon_*`
//!
//! The `itx_arm_neon_{4x4,8x8,16x16,32,64,rect,rect_large,large_rect}` kernels
//! are ports of dav1d's `itx.S`, which keeps the transform state in `int16x8_t`
//! — legal at 8bpc, where the spec's row/column clips are exactly
//! `i16::MIN..=i16::MAX`. At 10/12bpc the clips widen to `(!bitdepth_max) << 7`
//! / `<< 5`, so the state no longer fits in 16 bits. Those files nevertheless
//! define `*_16bpc_*` entry points that clamp to `i16` and run the same 16-bit
//! arithmetic; a `__simd_test_log` decode of the 4K 10-bit vector logged 5,038
//! `ITX_MISMATCH` for 16x16 alone, `nbad = 256` (every pixel of the block) on
//! 3,814 of them. They are not wired into `itxfm_add_dispatch` for that reason.
//!
//! This module is the other half of that fix: instead of widening the 16-bit
//! ports, it vectorises the *generic* reference in `src/itx.rs` +
//! `src/itx_1d.rs` directly, in `int32x4_t` lanes. Four lanes = four
//! independent 1-D transforms, and every lane executes the identical i32
//! operation sequence the scalar reference executes — so bit-exactness is
//! structural, not empirical. The 2-D driver below is a line-by-line
//! transliteration of `src/itx.rs::inv_txfm_add`, and each 1-D kernel of the
//! matching `inv_*_1d_internal_c`.
//!
//! # Lane arrangement
//!
//! * Row pass: lanes are four **rows**. `coeff` is column-major
//!   (`coeff[y + x * sh]`), so four consecutive rows at one column index are
//!   *contiguous* — one `vld1q_s32` per transform input, no gather.
//! * The result is transposed 4x4 on the way into `tmp` (row-major), which is
//!   also where the reference's `iclip(tmp[i] + rnd >> shift)` step is folded.
//! * Column pass: lanes are four **columns**. `tmp[y * w + x]` makes four
//!   adjacent columns contiguous, so again one `vld1q_s32` per input.
//!
//! # Rounding-shift note
//!
//! `(a * k + 2048) >> 12` maps to `vrshrq_n_s32::<12>(vmulq_n_s32(a, k))`.
//! SRSHR computes the rounding addend without an intermediate overflow, while
//! the scalar `a * k + 2048` wraps. They differ only for a product within 2048
//! of `i32::MAX` — which the scalar reference cannot produce for a conformant
//! stream (`src/itx_1d.rs` is the conformance oracle and a debug build of it
//! panics on overflow). The parity sweep in `itx_arm_parity.rs` and the
//! 766-vector corpus are what hold that claim.

#![allow(clippy::too_many_arguments)]
#![cfg_attr(not(feature = "unchecked"), forbid(unsafe_code))]
#![cfg_attr(feature = "unchecked", deny(unsafe_code))]

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
use archmage::{Arm64, arcane, rite};

#[cfg(target_arch = "aarch64")]
use safe_unaligned_simd::aarch64 as safe_simd;

/// One 1-D transform family. Mirrors `inv_txfm_add_rust`'s local `Type`.
#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Kind {
    Dct,
    Adst,
    FlipAdst,
    Identity,
}

#[cfg(target_arch = "aarch64")]
type V = int32x4_t;

/// The largest (w, h) this module handles. 32- and 64-point transforms keep
/// running on the scalar reference; see `hbd_supported`.
#[cfg(target_arch = "aarch64")]
const MAXDIM: usize = 16;

/// `iclip(v, min, max)`.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn clip4(v: V, min: V, max: V) -> V {
    vminq_s32(vmaxq_s32(v, min), max)
}

/// `(a * ka + 2048) >> 12`
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn m12(a: V, ka: i32) -> V {
    vrshrq_n_s32::<12>(vmulq_n_s32(a, ka))
}

/// `(a * ka + b * kb + 2048) >> 12`
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn mm12(a: V, ka: i32, b: V, kb: i32) -> V {
    vrshrq_n_s32::<12>(vmlaq_n_s32(vmulq_n_s32(a, ka), b, kb))
}

/// `(a * ka + b * kb + 1024) >> 11`
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn mm11(a: V, ka: i32, b: V, kb: i32) -> V {
    vrshrq_n_s32::<11>(vmlaq_n_s32(vmulq_n_s32(a, ka), b, kb))
}

/// `(a * 181 + 128) >> 8`
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn m181(a: V) -> V {
    vrshrq_n_s32::<8>(vmulq_n_s32(a, 181))
}

// ============================================================================
// 1-D kernels — transliterated from src/itx_1d.rs (tx64 = 0 throughout)
// ============================================================================

/// `inv_dct4_1d_internal_c`, `tx64 == 0`.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn dct4(c: &mut [V; 4], min: V, max: V) {
    let (in0, in1, in2, in3) = (c[0], c[1], c[2], c[3]);

    let t0 = m181(vaddq_s32(in0, in2));
    let t1 = m181(vsubq_s32(in0, in2));
    let t2 = vsubq_s32(mm12(in1, 1567, in3, -(3784 - 4096)), in3);
    let t3 = vaddq_s32(mm12(in1, 3784 - 4096, in3, 1567), in1);

    c[0] = clip4(vaddq_s32(t0, t3), min, max);
    c[1] = clip4(vaddq_s32(t1, t2), min, max);
    c[2] = clip4(vsubq_s32(t1, t2), min, max);
    c[3] = clip4(vsubq_s32(t0, t3), min, max);
}

/// `inv_dct8_1d_internal_c`, `tx64 == 0`.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn dct8(c: &mut [V; 8], min: V, max: V) {
    // The reference recurses with `stride << 1`, i.e. over the even indices.
    let mut e = [c[0], c[2], c[4], c[6]];
    dct4(&mut e, min, max);

    let (in1, in3, in5, in7) = (c[1], c[3], c[5], c[7]);

    let t4a = vsubq_s32(mm12(in1, 799, in7, -(4017 - 4096)), in7);
    let mut t5a = mm11(in5, 1703, in3, -1138);
    let mut t6a = mm11(in5, 1138, in3, 1703);
    let t7a = vaddq_s32(mm12(in1, 4017 - 4096, in7, 799), in1);

    let t4 = clip4(vaddq_s32(t4a, t5a), min, max);
    t5a = clip4(vsubq_s32(t4a, t5a), min, max);
    let t7 = clip4(vaddq_s32(t7a, t6a), min, max);
    t6a = clip4(vsubq_s32(t7a, t6a), min, max);

    let t5 = m181(vsubq_s32(t6a, t5a));
    let t6 = m181(vaddq_s32(t6a, t5a));

    let (t0, t1, t2, t3) = (e[0], e[1], e[2], e[3]);

    c[0] = clip4(vaddq_s32(t0, t7), min, max);
    c[1] = clip4(vaddq_s32(t1, t6), min, max);
    c[2] = clip4(vaddq_s32(t2, t5), min, max);
    c[3] = clip4(vaddq_s32(t3, t4), min, max);
    c[4] = clip4(vsubq_s32(t3, t4), min, max);
    c[5] = clip4(vsubq_s32(t2, t5), min, max);
    c[6] = clip4(vsubq_s32(t1, t6), min, max);
    c[7] = clip4(vsubq_s32(t0, t7), min, max);
}

/// `inv_dct16_1d_internal_c`, `tx64 == 0`.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn dct16(c: &mut [V; 16], min: V, max: V) {
    let mut e = [c[0], c[2], c[4], c[6], c[8], c[10], c[12], c[14]];
    dct8(&mut e, min, max);

    let (in1, in3, in5, in7) = (c[1], c[3], c[5], c[7]);
    let (in9, in11, in13, in15) = (c[9], c[11], c[13], c[15]);

    let mut t8a = vsubq_s32(mm12(in1, 401, in15, -(4076 - 4096)), in15);
    let mut t9a = mm11(in9, 1583, in7, -1299);
    let mut t10a = vsubq_s32(mm12(in5, 1931, in11, -(3612 - 4096)), in11);
    let mut t11a = vaddq_s32(mm12(in13, 3920 - 4096, in3, -1189), in13);
    let mut t12a = vaddq_s32(mm12(in13, 1189, in3, 3920 - 4096), in3);
    let mut t13a = vaddq_s32(mm12(in5, 3612 - 4096, in11, 1931), in5);
    let mut t14a = mm11(in9, 1299, in7, 1583);
    let mut t15a = vaddq_s32(mm12(in1, 4076 - 4096, in15, 401), in1);

    let t8 = clip4(vaddq_s32(t8a, t9a), min, max);
    let mut t9 = clip4(vsubq_s32(t8a, t9a), min, max);
    let mut t10 = clip4(vsubq_s32(t11a, t10a), min, max);
    let mut t11 = clip4(vaddq_s32(t11a, t10a), min, max);
    let mut t12 = clip4(vaddq_s32(t12a, t13a), min, max);
    let mut t13 = clip4(vsubq_s32(t12a, t13a), min, max);
    let mut t14 = clip4(vsubq_s32(t15a, t14a), min, max);
    let t15 = clip4(vaddq_s32(t15a, t14a), min, max);

    t9a = vsubq_s32(mm12(t14, 1567, t9, -(3784 - 4096)), t9);
    t14a = vaddq_s32(mm12(t14, 3784 - 4096, t9, 1567), t14);
    // `(-(t13 * (3784 - 4096) + t10 * 1567) + 2048 >> 12) - t13`
    t10a = vsubq_s32(
        vrshrq_n_s32::<12>(vnegq_s32(vmlaq_n_s32(
            vmulq_n_s32(t13, 3784 - 4096),
            t10,
            1567,
        ))),
        t13,
    );
    t13a = vsubq_s32(mm12(t13, 1567, t10, -(3784 - 4096)), t10);
    t8a = clip4(vaddq_s32(t8, t11), min, max);
    t9 = clip4(vaddq_s32(t9a, t10a), min, max);
    t10 = clip4(vsubq_s32(t9a, t10a), min, max);
    t11a = clip4(vsubq_s32(t8, t11), min, max);
    t12a = clip4(vsubq_s32(t15, t12), min, max);
    t13 = clip4(vsubq_s32(t14a, t13a), min, max);
    t14 = clip4(vaddq_s32(t14a, t13a), min, max);
    t15a = clip4(vaddq_s32(t15, t12), min, max);

    t10a = m181(vsubq_s32(t13, t10));
    t13a = m181(vaddq_s32(t13, t10));
    t11 = m181(vsubq_s32(t12a, t11a));
    t12 = m181(vaddq_s32(t12a, t11a));

    let (t0, t1, t2, t3) = (e[0], e[1], e[2], e[3]);
    let (t4, t5, t6, t7) = (e[4], e[5], e[6], e[7]);

    c[0] = clip4(vaddq_s32(t0, t15a), min, max);
    c[1] = clip4(vaddq_s32(t1, t14), min, max);
    c[2] = clip4(vaddq_s32(t2, t13a), min, max);
    c[3] = clip4(vaddq_s32(t3, t12), min, max);
    c[4] = clip4(vaddq_s32(t4, t11), min, max);
    c[5] = clip4(vaddq_s32(t5, t10a), min, max);
    c[6] = clip4(vaddq_s32(t6, t9), min, max);
    c[7] = clip4(vaddq_s32(t7, t8a), min, max);
    c[8] = clip4(vsubq_s32(t7, t8a), min, max);
    c[9] = clip4(vsubq_s32(t6, t9), min, max);
    c[10] = clip4(vsubq_s32(t5, t10a), min, max);
    c[11] = clip4(vsubq_s32(t4, t11), min, max);
    c[12] = clip4(vsubq_s32(t3, t12), min, max);
    c[13] = clip4(vsubq_s32(t2, t13a), min, max);
    c[14] = clip4(vsubq_s32(t1, t14), min, max);
    c[15] = clip4(vsubq_s32(t0, t15a), min, max);
}

/// `inv_adst4_1d_internal_c`, results in forward output order.
///
/// The reference takes no clip here (`_min` / `_max` are unused).
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn adst4_core(c: &[V; 4]) -> [V; 4] {
    let (in0, in1, in2, in3) = (c[0], c[1], c[2], c[3]);

    // 1321*in0 + (3803-4096)*in2 + (2482-4096)*in3 + (3344-4096)*in1 + 2048 >> 12
    let mut a = vmulq_n_s32(in0, 1321);
    a = vmlaq_n_s32(a, in2, 3803 - 4096);
    a = vmlaq_n_s32(a, in3, 2482 - 4096);
    a = vmlaq_n_s32(a, in1, 3344 - 4096);
    let o0 = vaddq_s32(vaddq_s32(vaddq_s32(vrshrq_n_s32::<12>(a), in2), in3), in1);

    // (2482-4096)*in0 - 1321*in2 - (3803-4096)*in3 + (3344-4096)*in1 + 2048 >> 12
    let mut b = vmulq_n_s32(in0, 2482 - 4096);
    b = vmlsq_n_s32(b, in2, 1321);
    b = vmlsq_n_s32(b, in3, 3803 - 4096);
    b = vmlaq_n_s32(b, in1, 3344 - 4096);
    let o1 = vaddq_s32(vsubq_s32(vaddq_s32(vrshrq_n_s32::<12>(b), in0), in3), in1);

    // 209 * (in0 - in2 + in3) + 128 >> 8
    let o2 = vrshrq_n_s32::<8>(vmulq_n_s32(vaddq_s32(vsubq_s32(in0, in2), in3), 209));

    // (3803-4096)*in0 + (2482-4096)*in2 - 1321*in3 - (3344-4096)*in1 + 2048 >> 12
    let mut d = vmulq_n_s32(in0, 3803 - 4096);
    d = vmlaq_n_s32(d, in2, 2482 - 4096);
    d = vmlsq_n_s32(d, in3, 1321);
    d = vmlsq_n_s32(d, in1, 3344 - 4096);
    let o3 = vsubq_s32(vaddq_s32(vaddq_s32(vrshrq_n_s32::<12>(d), in0), in2), in1);

    [o0, o1, o2, o3]
}

/// `inv_adst8_1d_internal_c`, results in forward output order.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn adst8_core(c: &[V; 8], min: V, max: V) -> [V; 8] {
    let (in0, in1, in2, in3) = (c[0], c[1], c[2], c[3]);
    let (in4, in5, in6, in7) = (c[4], c[5], c[6], c[7]);

    let t0a = vaddq_s32(mm12(in7, 4076 - 4096, in0, 401), in7);
    let t1a = vsubq_s32(mm12(in7, 401, in0, -(4076 - 4096)), in0);
    let t2a = vaddq_s32(mm12(in5, 3612 - 4096, in2, 1931), in5);
    let t3a = vsubq_s32(mm12(in5, 1931, in2, -(3612 - 4096)), in2);
    let mut t4a = mm11(in3, 1299, in4, 1583);
    let mut t5a = mm11(in3, 1583, in4, -1299);
    let mut t6a = vaddq_s32(mm12(in1, 1189, in6, 3920 - 4096), in6);
    let mut t7a = vaddq_s32(mm12(in1, 3920 - 4096, in6, -1189), in1);

    let t0 = clip4(vaddq_s32(t0a, t4a), min, max);
    let t1 = clip4(vaddq_s32(t1a, t5a), min, max);
    let mut t2 = clip4(vaddq_s32(t2a, t6a), min, max);
    let mut t3 = clip4(vaddq_s32(t3a, t7a), min, max);
    let t4 = clip4(vsubq_s32(t0a, t4a), min, max);
    let t5 = clip4(vsubq_s32(t1a, t5a), min, max);
    let mut t6 = clip4(vsubq_s32(t2a, t6a), min, max);
    let mut t7 = clip4(vsubq_s32(t3a, t7a), min, max);

    t4a = vaddq_s32(mm12(t4, 3784 - 4096, t5, 1567), t4);
    t5a = vsubq_s32(mm12(t4, 1567, t5, -(3784 - 4096)), t5);
    t6a = vaddq_s32(mm12(t7, 3784 - 4096, t6, -1567), t7);
    t7a = vaddq_s32(mm12(t7, 1567, t6, 3784 - 4096), t6);

    let mut o = [vdupq_n_s32(0); 8];
    o[0] = clip4(vaddq_s32(t0, t2), min, max);
    o[7] = vnegq_s32(clip4(vaddq_s32(t1, t3), min, max));
    t2 = clip4(vsubq_s32(t0, t2), min, max);
    t3 = clip4(vsubq_s32(t1, t3), min, max);
    o[1] = vnegq_s32(clip4(vaddq_s32(t4a, t6a), min, max));
    o[6] = clip4(vaddq_s32(t5a, t7a), min, max);
    t6 = clip4(vsubq_s32(t4a, t6a), min, max);
    t7 = clip4(vsubq_s32(t5a, t7a), min, max);

    o[3] = vnegq_s32(m181(vaddq_s32(t2, t3)));
    o[4] = m181(vsubq_s32(t2, t3));
    o[2] = m181(vaddq_s32(t6, t7));
    o[5] = vnegq_s32(m181(vsubq_s32(t6, t7)));
    o
}

/// `inv_adst16_1d_internal_c`, results in forward output order.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn adst16_core(c: &[V; 16], min: V, max: V) -> [V; 16] {
    let (in0, in1, in2, in3) = (c[0], c[1], c[2], c[3]);
    let (in4, in5, in6, in7) = (c[4], c[5], c[6], c[7]);
    let (in8, in9, in10, in11) = (c[8], c[9], c[10], c[11]);
    let (in12, in13, in14, in15) = (c[12], c[13], c[14], c[15]);

    let mut t0 = vaddq_s32(mm12(in15, 4091 - 4096, in0, 201), in15);
    let mut t1 = vsubq_s32(mm12(in15, 201, in0, -(4091 - 4096)), in0);
    let mut t2 = vaddq_s32(mm12(in13, 3973 - 4096, in2, 995), in13);
    let mut t3 = vsubq_s32(mm12(in13, 995, in2, -(3973 - 4096)), in2);
    let mut t4 = vaddq_s32(mm12(in11, 3703 - 4096, in4, 1751), in11);
    let mut t5 = vsubq_s32(mm12(in11, 1751, in4, -(3703 - 4096)), in4);
    let mut t6 = mm11(in9, 1645, in6, 1220);
    let mut t7 = mm11(in9, 1220, in6, -1645);
    let mut t8 = vaddq_s32(mm12(in7, 2751, in8, 3035 - 4096), in8);
    let mut t9 = vaddq_s32(mm12(in7, 3035 - 4096, in8, -2751), in7);
    let mut t10 = vaddq_s32(mm12(in5, 2106, in10, 3513 - 4096), in10);
    let mut t11 = vaddq_s32(mm12(in5, 3513 - 4096, in10, -2106), in5);
    let mut t12 = vaddq_s32(mm12(in3, 1380, in12, 3857 - 4096), in12);
    let mut t13 = vaddq_s32(mm12(in3, 3857 - 4096, in12, -1380), in3);
    let mut t14 = vaddq_s32(mm12(in1, 601, in14, 4052 - 4096), in14);
    let mut t15 = vaddq_s32(mm12(in1, 4052 - 4096, in14, -601), in1);

    let t0a = clip4(vaddq_s32(t0, t8), min, max);
    let t1a = clip4(vaddq_s32(t1, t9), min, max);
    let mut t2a = clip4(vaddq_s32(t2, t10), min, max);
    let mut t3a = clip4(vaddq_s32(t3, t11), min, max);
    let mut t4a = clip4(vaddq_s32(t4, t12), min, max);
    let mut t5a = clip4(vaddq_s32(t5, t13), min, max);
    let mut t6a = clip4(vaddq_s32(t6, t14), min, max);
    let mut t7a = clip4(vaddq_s32(t7, t15), min, max);
    let mut t8a = clip4(vsubq_s32(t0, t8), min, max);
    let mut t9a = clip4(vsubq_s32(t1, t9), min, max);
    let mut t10a = clip4(vsubq_s32(t2, t10), min, max);
    let mut t11a = clip4(vsubq_s32(t3, t11), min, max);
    let mut t12a = clip4(vsubq_s32(t4, t12), min, max);
    let mut t13a = clip4(vsubq_s32(t5, t13), min, max);
    let mut t14a = clip4(vsubq_s32(t6, t14), min, max);
    let mut t15a = clip4(vsubq_s32(t7, t15), min, max);

    t8 = vaddq_s32(mm12(t8a, 4017 - 4096, t9a, 799), t8a);
    t9 = vsubq_s32(mm12(t8a, 799, t9a, -(4017 - 4096)), t9a);
    t10 = vaddq_s32(mm12(t10a, 2276, t11a, 3406 - 4096), t11a);
    t11 = vaddq_s32(mm12(t10a, 3406 - 4096, t11a, -2276), t10a);
    t12 = vaddq_s32(mm12(t13a, 4017 - 4096, t12a, -799), t13a);
    t13 = vaddq_s32(mm12(t13a, 799, t12a, 4017 - 4096), t12a);
    t14 = vsubq_s32(mm12(t15a, 2276, t14a, -(3406 - 4096)), t14a);
    t15 = vaddq_s32(mm12(t15a, 3406 - 4096, t14a, 2276), t15a);

    t0 = clip4(vaddq_s32(t0a, t4a), min, max);
    t1 = clip4(vaddq_s32(t1a, t5a), min, max);
    t2 = clip4(vaddq_s32(t2a, t6a), min, max);
    t3 = clip4(vaddq_s32(t3a, t7a), min, max);
    t4 = clip4(vsubq_s32(t0a, t4a), min, max);
    t5 = clip4(vsubq_s32(t1a, t5a), min, max);
    t6 = clip4(vsubq_s32(t2a, t6a), min, max);
    t7 = clip4(vsubq_s32(t3a, t7a), min, max);
    t8a = clip4(vaddq_s32(t8, t12), min, max);
    t9a = clip4(vaddq_s32(t9, t13), min, max);
    t10a = clip4(vaddq_s32(t10, t14), min, max);
    t11a = clip4(vaddq_s32(t11, t15), min, max);
    t12a = clip4(vsubq_s32(t8, t12), min, max);
    t13a = clip4(vsubq_s32(t9, t13), min, max);
    t14a = clip4(vsubq_s32(t10, t14), min, max);
    t15a = clip4(vsubq_s32(t11, t15), min, max);

    t4a = vaddq_s32(mm12(t4, 3784 - 4096, t5, 1567), t4);
    t5a = vsubq_s32(mm12(t4, 1567, t5, -(3784 - 4096)), t5);
    t6a = vaddq_s32(mm12(t7, 3784 - 4096, t6, -1567), t7);
    t7a = vaddq_s32(mm12(t7, 1567, t6, 3784 - 4096), t6);
    t12 = vaddq_s32(mm12(t12a, 3784 - 4096, t13a, 1567), t12a);
    t13 = vsubq_s32(mm12(t12a, 1567, t13a, -(3784 - 4096)), t13a);
    t14 = vaddq_s32(mm12(t15a, 3784 - 4096, t14a, -1567), t15a);
    t15 = vaddq_s32(mm12(t15a, 1567, t14a, 3784 - 4096), t14a);

    let mut o = [vdupq_n_s32(0); 16];
    o[0] = clip4(vaddq_s32(t0, t2), min, max);
    o[15] = vnegq_s32(clip4(vaddq_s32(t1, t3), min, max));
    t2a = clip4(vsubq_s32(t0, t2), min, max);
    t3a = clip4(vsubq_s32(t1, t3), min, max);
    o[3] = vnegq_s32(clip4(vaddq_s32(t4a, t6a), min, max));
    o[12] = clip4(vaddq_s32(t5a, t7a), min, max);
    t6 = clip4(vsubq_s32(t4a, t6a), min, max);
    t7 = clip4(vsubq_s32(t5a, t7a), min, max);
    o[1] = vnegq_s32(clip4(vaddq_s32(t8a, t10a), min, max));
    o[14] = clip4(vaddq_s32(t9a, t11a), min, max);
    t10 = clip4(vsubq_s32(t8a, t10a), min, max);
    t11 = clip4(vsubq_s32(t9a, t11a), min, max);
    o[2] = clip4(vaddq_s32(t12, t14), min, max);
    o[13] = vnegq_s32(clip4(vaddq_s32(t13, t15), min, max));
    t14a = clip4(vsubq_s32(t12, t14), min, max);
    t15a = clip4(vsubq_s32(t13, t15), min, max);

    o[7] = vnegq_s32(m181(vaddq_s32(t2a, t3a)));
    o[8] = m181(vsubq_s32(t2a, t3a));
    o[4] = m181(vaddq_s32(t6, t7));
    o[11] = vnegq_s32(m181(vsubq_s32(t6, t7)));
    o[6] = m181(vaddq_s32(t10, t11));
    o[9] = vnegq_s32(m181(vsubq_s32(t10, t11)));
    o[5] = vnegq_s32(m181(vaddq_s32(t14a, t15a)));
    o[10] = m181(vsubq_s32(t14a, t15a));
    o
}

/// Apply one 1-D transform of length `n` to `v[..n]`.
///
/// `kind` and `n` are runtime values, but the branch is taken once per group of
/// four transforms, not per element, so it is far below the cost of the
/// transform itself. Keeping them runtime is what holds the monomorphisation
/// count at one for the whole 16bpc itx path.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn apply1d(kind: Kind, n: usize, v: &mut [V], min: V, max: V) {
    match (kind, n) {
        (Kind::Dct, 4) => dct4((&mut v[..4]).try_into().unwrap(), min, max),
        (Kind::Dct, 8) => dct8((&mut v[..8]).try_into().unwrap(), min, max),
        (Kind::Dct, 16) => dct16((&mut v[..16]).try_into().unwrap(), min, max),
        (Kind::Adst, 4) => {
            let o = adst4_core((&v[..4]).try_into().unwrap());
            v[..4].copy_from_slice(&o);
        }
        (Kind::FlipAdst, 4) => {
            let o = adst4_core((&v[..4]).try_into().unwrap());
            for i in 0..4 {
                v[i] = o[3 - i];
            }
        }
        (Kind::Adst, 8) => {
            let o = adst8_core((&v[..8]).try_into().unwrap(), min, max);
            v[..8].copy_from_slice(&o);
        }
        (Kind::FlipAdst, 8) => {
            let o = adst8_core((&v[..8]).try_into().unwrap(), min, max);
            for i in 0..8 {
                v[i] = o[7 - i];
            }
        }
        (Kind::Adst, 16) => {
            let o = adst16_core((&v[..16]).try_into().unwrap(), min, max);
            v[..16].copy_from_slice(&o);
        }
        (Kind::FlipAdst, 16) => {
            let o = adst16_core((&v[..16]).try_into().unwrap(), min, max);
            for i in 0..16 {
                v[i] = o[15 - i];
            }
        }
        (Kind::Identity, 4) => {
            for e in v[..4].iter_mut() {
                *e = vaddq_s32(*e, m12(*e, 1697));
            }
        }
        (Kind::Identity, 8) => {
            for e in v[..8].iter_mut() {
                *e = vaddq_s32(*e, *e);
            }
        }
        (Kind::Identity, 16) => {
            for e in v[..16].iter_mut() {
                *e = vaddq_s32(vaddq_s32(*e, *e), vrshrq_n_s32::<11>(vmulq_n_s32(*e, 1697)));
            }
        }
        // `hbd_supported` is what keeps this unreachable; a shape it lets
        // through with no kernel here would silently produce zeros, so it
        // panics instead of writing wrong pixels.
        _ => unreachable!("itx_arm_hbd: no 1-D kernel for {kind:?} n={n}"),
    }
}

/// Transpose four `int32x4_t` (rows) into four `int32x4_t` (columns).
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn transpose4x4(v: [V; 4]) -> [V; 4] {
    let a = vtrn1q_s32(v[0], v[1]);
    let b = vtrn2q_s32(v[0], v[1]);
    let c = vtrn1q_s32(v[2], v[3]);
    let d = vtrn2q_s32(v[2], v[3]);
    let (a64, b64) = (vreinterpretq_s64_s32(a), vreinterpretq_s64_s32(b));
    let (c64, d64) = (vreinterpretq_s64_s32(c), vreinterpretq_s64_s32(d));
    [
        vreinterpretq_s32_s64(vtrn1q_s64(a64, c64)),
        vreinterpretq_s32_s64(vtrn1q_s64(b64, d64)),
        vreinterpretq_s32_s64(vtrn2q_s64(a64, c64)),
        vreinterpretq_s32_s64(vtrn2q_s64(b64, d64)),
    ]
}

/// Whether this module has kernels for `(w, h)`.
///
/// 32- and 64-point transforms are not ported; those sizes keep running on
/// `src/itx.rs`'s scalar reference.
#[cfg(target_arch = "aarch64")]
pub(crate) fn hbd_supported(w: usize, h: usize) -> bool {
    w <= MAXDIM && h <= MAXDIM
}

/// `iclip_pixel(dst + delta)` for a 4-pixel group. `delta` is already final.
#[cfg(target_arch = "aarch64")]
#[rite(neon)]
fn add_clip4(dst: &mut [u16], off: usize, delta: V, bd_max: V) {
    let arr = <&[u16; 4]>::try_from(&dst[off..off + 4]).expect("4 pixels");
    let cur = vreinterpretq_s32_u32(vmovl_u16(safe_simd::vld1_u16(arr)));
    let sum = vminq_s32(vmaxq_s32(vaddq_s32(cur, delta), vdupq_n_s32(0)), bd_max);
    let out = vmovn_u32(vreinterpretq_u32_s32(sum));
    let slot = <&mut [u16; 4]>::try_from(&mut dst[off..off + 4]).expect("4 pixels");
    safe_simd::vst1_u16(slot, out);
}

/// The DC-only shortcut's scalar half: `src/itx.rs::inv_txfm_add`'s `eob <
/// has_dc_only` branch, up to but not including the pixel add.
#[cfg(target_arch = "aarch64")]
pub(crate) fn hbd_dc_value(w: usize, h: usize, shift: u32, coeff: &mut [i32]) -> i32 {
    let is_rect2 = w * 2 == h || h * 2 == w;
    let rnd: i32 = (1 << shift) >> 1;
    let mut dc = coeff[0];
    coeff[0] = 0;
    if is_rect2 {
        dc = (dc * 181 + 128) >> 8;
    }
    dc = (dc * 181 + 128) >> 8;
    dc = (dc + rnd) >> shift;
    (dc * 181 + 128 + 2048) >> 12
}

/// Add one already-computed row of residual to `w` destination pixels.
///
/// Taking `dst` as exactly one row is what lets the caller hold ONE NARROW
/// GUARD PER ROW instead of a wide guard over the whole block. See
/// `itxfm_add_dispatch`'s 16bpc arm for the measurement that decided it.
#[cfg(target_arch = "aarch64")]
#[arcane]
pub(crate) fn add_row_hbd_neon(
    _token: Arm64,
    dst: &mut [u16],
    tmp_row: &[i32],
    w: usize,
    bitdepth_max: i32,
) {
    let bd_max = vdupq_n_s32(bitdepth_max);
    for x in (0..w).step_by(4) {
        let a = <&[i32; 4]>::try_from(&tmp_row[x..x + 4]).expect("4 lanes");
        let delta = vshrq_n_s32::<4>(vaddq_s32(safe_simd::vld1q_s32(a), vdupq_n_s32(8)));
        add_clip4(dst, x, delta, bd_max);
    }
}

/// Add a constant DC to `w` destination pixels.
#[cfg(target_arch = "aarch64")]
#[arcane]
pub(crate) fn add_row_dc_hbd_neon(
    _token: Arm64,
    dst: &mut [u16],
    w: usize,
    dc: i32,
    bitdepth_max: i32,
) {
    let bd_max = vdupq_n_s32(bitdepth_max);
    let dcv = vdupq_n_s32(dc);
    for x in (0..w).step_by(4) {
        add_clip4(dst, x, dcv, bd_max);
    }
}

/// The 16bpc 2-D inverse transform, vectorised, WITHOUT the pixel add.
///
/// Transliteration of `src/itx.rs::inv_txfm_add` for `BD::BITDEPTH != 8`, with
/// `w, h <= 16` (so `sw == w` and `sh == h`, and the reference's
/// zero-padded-tail cases cannot arise). The residual lands in `tmp` row-major
/// and the caller adds it a row at a time.
#[cfg(target_arch = "aarch64")]
#[arcane]
pub(crate) fn inv_txfm_hbd_neon(
    _token: Arm64,
    w: usize,
    h: usize,
    first: Kind,
    second: Kind,
    shift: u32,
    coeff: &mut [i32],
    bitdepth_max: i32,
    tmp: &mut [i32; MAXDIM * MAXDIM],
) {
    debug_assert!(w <= MAXDIM && h <= MAXDIM);
    debug_assert!(w % 4 == 0 && h % 4 == 0);

    let is_rect2 = w * 2 == h || h * 2 == w;
    let rnd: i32 = (1 << shift) >> 1;

    let row_clip_min = vdupq_n_s32((!bitdepth_max) << 7);
    let row_clip_max = vdupq_n_s32(!((!bitdepth_max) << 7));
    let col_clip_min = vdupq_n_s32((!bitdepth_max) << 5);
    let col_clip_max = vdupq_n_s32(!((!bitdepth_max) << 5));
    let rnd_v = vdupq_n_s32(rnd);
    let shr = vdupq_n_s32(-(shift as i32));

    // ---- row pass: four rows per iteration, lanes = rows ----
    let mut y0 = 0usize;
    while y0 < h {
        let mut v = [vdupq_n_s32(0); MAXDIM];
        for x in 0..w {
            let base = y0 + x * h;
            let a = <&[i32; 4]>::try_from(&coeff[base..base + 4]).expect("4 coeffs");
            let t = safe_simd::vld1q_s32(a);
            v[x] = if is_rect2 { m181(t) } else { t };
        }

        apply1d(first, w, &mut v, row_clip_min, row_clip_max);

        // Reference: `tmp[i] = iclip(tmp[i] + rnd >> shift, col_clip_*)`.
        for e in v[..w].iter_mut() {
            *e = clip4(
                vshlq_s32(vaddq_s32(*e, rnd_v), shr),
                col_clip_min,
                col_clip_max,
            );
        }

        for x0 in (0..w).step_by(4) {
            let t = transpose4x4([v[x0], v[x0 + 1], v[x0 + 2], v[x0 + 3]]);
            for (j, tj) in t.iter().enumerate() {
                let off = (y0 + j) * w + x0;
                let slot = <&mut [i32; 4]>::try_from(&mut tmp[off..off + 4]).expect("4 lanes");
                safe_simd::vst1q_s32(slot, *tj);
            }
        }
        y0 += 4;
    }

    coeff[..w * h].fill(0);

    // ---- column pass: four columns per iteration, lanes = columns ----
    for x0 in (0..w).step_by(4) {
        let mut u = [vdupq_n_s32(0); MAXDIM];
        for y in 0..h {
            let off = y * w + x0;
            let a = <&[i32; 4]>::try_from(&tmp[off..off + 4]).expect("4 lanes");
            u[y] = safe_simd::vld1q_s32(a);
        }
        apply1d(second, h, &mut u, col_clip_min, col_clip_max);
        for y in 0..h {
            let off = y * w + x0;
            let slot = <&mut [i32; 4]>::try_from(&mut tmp[off..off + 4]).expect("4 lanes");
            safe_simd::vst1q_s32(slot, u[y]);
        }
    }
    let _ = bitdepth_max;
}
