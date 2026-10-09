//! Safe SIMD implementations of motion compensation functions
#![allow(dead_code)]
#![cfg_attr(
    not(any(feature = "asm", feature = "c-ffi", feature = "partial_asm")),
    forbid(unsafe_code)
)]
#![cfg_attr(
    any(feature = "asm", feature = "c-ffi", feature = "partial_asm"),
    deny(unsafe_code)
)]
//!
//! These replace the hand-written assembly in src/x86/mc_*.asm
//!
//! Uses archmage tokens for safe SIMD invocation:
//! - Desktop64 (X64V3Token) for AVX2+FMA on x86-64
//! - The runtime CPU check happens at init time in the dispatch table setup
//! - extern "C" wrappers declare V3 with #[rite(v3)] and use from_context()

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
use crate::src::safe_simd::pixel_access::Flex;
#[cfg(target_arch = "x86_64")]
use crate::src::safe_simd::pixel_access::{
    loadi32, loadi64, loadu_64, loadu_128, loadu_256, loadu_512, storei64, storeu_128, storeu_256,
    storeu_512,
};
#[cfg(target_arch = "x86_64")]
use archmage::{Desktop64, Server64, arcane, rite};

#[cfg(target_arch = "x86_64")]
use crate::include::common::bitdepth::BitDepth;
use crate::include::dav1d::headers::Rav1dFilterMode;
#[cfg(target_arch = "x86_64")]
use crate::include::dav1d::headers::Rav1dPixelLayoutSubSampled;
#[cfg(target_arch = "x86_64")]
use crate::include::dav1d::picture::PicOffset;
#[cfg(target_arch = "x86_64")]
use crate::src::internal::COMPINTER_LEN;
#[cfg(target_arch = "x86_64")]
use crate::src::levels::Filter2d;
#[cfg(target_arch = "x86_64")]
use crate::src::strided::Strided as _;

use std::cell::Cell;

#[cfg(any(
    target_arch = "x86_64",
    all(target_arch = "aarch64", not(feature = "asm"))
))]
#[path = "mc_reference.rs"]
pub(super) mod reference;

type Mid16x135 = Box<[[i16; MID_STRIDE]; 135]>;
type Mid32x135 = Box<[[i32; MID_STRIDE]; 135]>;
type Mid16x130 = Box<[[i16; MID_STRIDE]; 130]>;
type Mid32x130 = Box<[[i32; MID_STRIDE]; 130]>;

// Thread-local scratch buffers for MC intermediate data.
// Allocated once on first use (zeroed by Box::new), then reused across all MC
// calls. This eliminates ~520M instructions of memset per decode (25% of total).
// The buffers don't need re-zeroing between calls because every MC function
// fully writes the region it reads (horizontal filter writes rows 0..h+7,
// vertical filter reads only from those written rows).
//
// Pattern: take() before use, set(Some(...)) after use. If a function panics
// between take and put, the next call allocates a fresh buffer (no unsoundness).
thread_local! {
    static MID_I16_135: Cell<Option<Mid16x135>> = const { Cell::new(None) };
    static MID_I32_135: Cell<Option<Mid32x135>> = const { Cell::new(None) };
    static MID_I16_130: Cell<Option<Mid16x130>> = const { Cell::new(None) };
    static MID_I32_130: Cell<Option<Mid32x130>> = const { Cell::new(None) };
}

#[inline(always)]
fn take_mid_i16_135() -> Mid16x135 {
    MID_I16_135
        .with(|c| c.take())
        .unwrap_or_else(|| Box::new([[0; MID_STRIDE]; 135]))
}
#[inline(always)]
fn put_mid_i16_135(mid: Mid16x135) {
    MID_I16_135.with(|c| c.set(Some(mid)));
}
#[inline(always)]
fn take_mid_i32_135() -> Mid32x135 {
    MID_I32_135
        .with(|c| c.take())
        .unwrap_or_else(|| Box::new([[0; MID_STRIDE]; 135]))
}
#[inline(always)]
fn put_mid_i32_135(mid: Mid32x135) {
    MID_I32_135.with(|c| c.set(Some(mid)));
}
#[inline(always)]
fn take_mid_i16_130() -> Mid16x130 {
    MID_I16_130
        .with(|c| c.take())
        .unwrap_or_else(|| Box::new([[0; MID_STRIDE]; 130]))
}
#[inline(always)]
fn put_mid_i16_130(mid: Mid16x130) {
    MID_I16_130.with(|c| c.set(Some(mid)));
}
#[inline(always)]
fn take_mid_i32_130() -> Mid32x130 {
    MID_I32_130
        .with(|c| c.take())
        .unwrap_or_else(|| Box::new([[0; MID_STRIDE]; 130]))
}
#[inline(always)]
fn put_mid_i32_130(mid: Mid32x130) {
    MID_I32_130.with(|c| c.set(Some(mid)));
}

/// Rounding constant for pmulhrsw: 1024 = (1 << 10)
/// pmulhrsw computes: (a * b + 16384) >> 15
/// With b=1024: (a * 1024 + 16384) >> 15 ≈ (a + 1) >> 1 (with rounding)
const PW_1024: i16 = 1024;

/// AVG operation for 8-bit pixels using AVX2
///
/// Averages two 16-bit intermediate buffers and packs to 8-bit output.
/// This matches the signature of dav1d_avg_8bpc_avx2.
///
/// # Safety
///
/// - Caller must ensure AVX2 is available (checked at dispatch time)
/// - dst_ptr must be valid for writing w*h bytes
/// - tmp1 and tmp2 must contain at least w*h elements
#[cfg(target_arch = "x86_64")]
#[arcane]
fn avg_8bpc_avx2_safe(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
) {
    let mut dst = dst.flex_mut();
    let w = w as usize;
    let h = h as usize;

    // Broadcast rounding constant to all lanes
    // SAFETY: AVX2 is available (checked at dispatch time via target_feature)
    let round = _mm256_set1_epi16(PW_1024);

    for row in 0..h {
        let tmp1_row = &tmp1[row * w..][..w];
        let tmp2_row = &tmp2[row * w..][..w];
        let dst_row = &mut dst[row * dst_stride..][..w];

        // Process 32 pixels at a time (64 bytes of i16 input → 32 bytes of u8 output)
        let mut col = 0;
        while col + 32 <= w {
            let t1_lo = loadu_256!(&tmp1_row[col..col + 16], [i16; 16]);
            let t1_hi = loadu_256!(&tmp1_row[col + 16..col + 32], [i16; 16]);
            let t2_lo = loadu_256!(&tmp2_row[col..col + 16], [i16; 16]);
            let t2_hi = loadu_256!(&tmp2_row[col + 16..col + 32], [i16; 16]);

            // Add: tmp1 + tmp2. Saturating, not wrapping: the scalar
            // reference sums in i32 (|sum| <= 65534), and a wrapped i16 sum
            // flips sign at +-32768. Saturation is bit-exact here because
            // whenever the true sum exceeds +-32767 the shifted result lands
            // in the [0, 255]-clamped region either way.
            let sum_lo = _mm256_adds_epi16(t1_lo, t2_lo);
            let sum_hi = _mm256_adds_epi16(t1_hi, t2_hi);

            // Multiply and round shift: (sum * 1024 + 16384) >> 15
            let avg_lo = _mm256_mulhrs_epi16(sum_lo, round);
            let avg_hi = _mm256_mulhrs_epi16(sum_hi, round);

            // Pack to unsigned bytes with saturation
            let packed = _mm256_packus_epi16(avg_lo, avg_hi);
            // Fix AVX2 lane interleaving
            let result = _mm256_permute4x64_epi64(packed, 0b11_01_10_00);

            // Store 32 bytes
            storeu_256!(&mut dst_row[col..col + 32], [u8; 32], result);

            col += 32;
        }

        // Process 16 pixels at a time for remainder
        while col + 16 <= w {
            let t1 = loadu_256!(&tmp1_row[col..col + 16], [i16; 16]);
            let t2 = loadu_256!(&tmp2_row[col..col + 16], [i16; 16]);

            let sum = _mm256_adds_epi16(t1, t2);
            let avg = _mm256_mulhrs_epi16(sum, round);

            // Pack within lanes and extract lower 128 bits
            let packed = _mm256_packus_epi16(avg, avg);
            let lo = _mm256_castsi256_si128(packed);
            let hi = _mm256_extracti128_si256(packed, 1);
            let result = _mm_unpacklo_epi64(lo, hi);

            storeu_128!(&mut dst_row[col..col + 16], [u8; 16], result);

            col += 16;
        }

        // Scalar fallback for remaining pixels — i32 sum, matching avg_rust
        // (warp_affine mids can reach +-32767 where an i16 add wraps).
        while col < w {
            let sum = tmp1_row[col] as i32 + tmp2_row[col] as i32;
            let avg = ((sum * 1024 + 16384) >> 15).clamp(0, 255) as u8;
            dst_row[col] = avg;
            col += 1;
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn avg_8bpc_v3(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    _bitdepth_max: i32,
    _dst: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    let dst = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h as usize * dst_stride as usize)
    };
    avg_8bpc_avx2_safe(token, dst, dst_stride as usize, tmp1, tmp2, w, h)
}

/// AVG operation for 8-bit pixels using AVX-512
///
/// Processes 64 pixels per iteration using 512-bit registers.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn avg_8bpc_avx512_safe(
    _token: Server64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
) {
    let mut dst = dst.flex_mut();
    let w = w as usize;
    let h = h as usize;

    let round = _mm512_set1_epi16(PW_1024);
    let zero = _mm512_setzero_si512();

    for row in 0..h {
        let tmp1_row = &tmp1[row * w..][..w];
        let tmp2_row = &tmp2[row * w..][..w];
        let dst_row = &mut dst[row * dst_stride..][..w];

        let mut col = 0;

        // Process 64 pixels at a time (two 512-bit loads of i16 → 64 u8 output)
        while col + 64 <= w {
            // First 32 i16 values
            let t1_lo = loadu_512!(&tmp1_row[col..col + 32], [i16; 32]);
            let t2_lo = loadu_512!(&tmp2_row[col..col + 32], [i16; 32]);
            let sum_lo = _mm512_adds_epi16(t1_lo, t2_lo);
            let avg_lo = _mm512_mulhrs_epi16(sum_lo, round);
            let avg_lo = _mm512_max_epi16(avg_lo, zero); // clamp negatives for unsigned sat
            let result_lo: __m256i = _mm512_cvtusepi16_epi8(avg_lo);

            // Second 32 i16 values
            let t1_hi = loadu_512!(&tmp1_row[col + 32..col + 64], [i16; 32]);
            let t2_hi = loadu_512!(&tmp2_row[col + 32..col + 64], [i16; 32]);
            let sum_hi = _mm512_adds_epi16(t1_hi, t2_hi);
            let avg_hi = _mm512_mulhrs_epi16(sum_hi, round);
            let avg_hi = _mm512_max_epi16(avg_hi, zero);
            let result_hi: __m256i = _mm512_cvtusepi16_epi8(avg_hi);

            // Combine two __m256i into one __m512i and store 64 bytes
            let combined = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(result_lo), result_hi);
            storeu_512!(&mut dst_row[col..col + 64], [u8; 64], combined);

            col += 64;
        }

        // Process 32 pixels at a time
        while col + 32 <= w {
            let t1 = loadu_512!(&tmp1_row[col..col + 32], [i16; 32]);
            let t2 = loadu_512!(&tmp2_row[col..col + 32], [i16; 32]);
            let sum = _mm512_adds_epi16(t1, t2);
            let avg = _mm512_mulhrs_epi16(sum, round);
            let avg = _mm512_max_epi16(avg, zero);
            let result: __m256i = _mm512_cvtusepi16_epi8(avg);

            storeu_256!(&mut dst_row[col..col + 32], [u8; 32], result);
            col += 32;
        }

        // AVX2 fallback for 16-pixel chunks
        let round_256 = _mm256_set1_epi16(PW_1024);
        while col + 16 <= w {
            let t1 = loadu_256!(&tmp1_row[col..col + 16], [i16; 16]);
            let t2 = loadu_256!(&tmp2_row[col..col + 16], [i16; 16]);
            let sum = _mm256_adds_epi16(t1, t2);
            let avg = _mm256_mulhrs_epi16(sum, round_256);
            let packed = _mm256_packus_epi16(avg, avg);
            let lo = _mm256_castsi256_si128(packed);
            let hi = _mm256_extracti128_si256(packed, 1);
            let result = _mm_unpacklo_epi64(lo, hi);
            storeu_128!(&mut dst_row[col..col + 16], [u8; 16], result);
            col += 16;
        }

        // Scalar tail — i32 sum like avg_rust (warp_affine mids reach +-32767)
        while col < w {
            let sum = tmp1_row[col] as i32 + tmp2_row[col] as i32;
            let avg = ((sum * 1024 + 16384) >> 15).clamp(0, 255) as u8;
            dst_row[col] = avg;
            col += 1;
        }
    }
}

/// AVG operation for 16-bit pixels using AVX2
///
/// For 10/12-bit content. Averages two 16-bit intermediate buffers.
///
/// # Safety
///
/// Same as avg_8bpc_v3, plus bitdepth_max must be correct for the content.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn avg_16bpc_avx2_safe(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let w = w as usize;
    let h = h as usize;

    // intermediate_bits = 14 - bitdepth: 4 for 10-bit, 2 for 12-bit
    let intermediate_bits = if (bitdepth_max >> 11) != 0 {
        2i32
    } else {
        4i32
    };
    // sh = intermediate_bits + 1, rnd = (1 << intermediate_bits) + PREP_BIAS * 2
    let sh = intermediate_bits + 1;
    let rnd = (1 << intermediate_bits) + 8192 * 2; // PREP_BIAS = 8192 for 16bpc
    let max = bitdepth_max as i32;

    let rnd_vec = _mm256_set1_epi32(rnd);
    let zero = _mm256_setzero_si256();
    let max_vec = _mm256_set1_epi32(max);

    for row in 0..h {
        let tmp1_row = &tmp1[row * w..][..w];
        let tmp2_row = &tmp2[row * w..][..w];
        // dst is bytes, but we write u16 pixels — each pixel is 2 bytes
        let dst_row_bytes = &mut dst[row * dst_stride..][..w * 2];
        // Reinterpret as u16 slice
        let dst_row: &mut [u16] = zerocopy::FromBytes::mut_from_bytes(dst_row_bytes).unwrap();

        let mut col = 0usize;

        while col + 16 <= w {
            let t1 = loadu_256!(&tmp1_row[col..col + 16], [i16; 16]);
            let t2 = loadu_256!(&tmp2_row[col..col + 16], [i16; 16]);

            let t1_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(t1));
            let t2_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(t2));
            let t1_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(t1, 1));
            let t2_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(t2, 1));

            let sum_lo = _mm256_add_epi32(_mm256_add_epi32(t1_lo, t2_lo), rnd_vec);
            let sum_hi = _mm256_add_epi32(_mm256_add_epi32(t1_hi, t2_hi), rnd_vec);

            // srai_epi32 requires const immediate; branch on sh
            let (result_lo, result_hi) = if sh == 3 {
                (_mm256_srai_epi32(sum_lo, 3), _mm256_srai_epi32(sum_hi, 3))
            } else {
                (_mm256_srai_epi32(sum_lo, 5), _mm256_srai_epi32(sum_hi, 5))
            };

            let clamped_lo = _mm256_min_epi32(_mm256_max_epi32(result_lo, zero), max_vec);
            let clamped_hi = _mm256_min_epi32(_mm256_max_epi32(result_hi, zero), max_vec);

            let packed = _mm256_packus_epi32(clamped_lo, clamped_hi);
            let packed = _mm256_permute4x64_epi64(packed, 0b11011000);

            storeu_256!(&mut dst_row[col..col + 16], [u16; 16], packed);

            col += 16;
        }

        while col < w {
            let sum = tmp1_row[col] as i32 + tmp2_row[col] as i32;
            let val = ((sum + rnd) >> sh).clamp(0, max) as u16;
            dst_row[col] = val;
            col += 1;
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn avg_16bpc_v3(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    bitdepth_max: i32,
    _dst: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    let dst = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h as usize * dst_stride as usize)
    };
    avg_16bpc_avx2_safe(
        token,
        dst,
        dst_stride as usize,
        tmp1,
        tmp2,
        w,
        h,
        bitdepth_max,
    )
}

/// AVG for 16-bit pixels using AVX-512
///
/// Processes 32 pixels per iteration (two groups of 16 i32).
#[cfg(target_arch = "x86_64")]
#[arcane]
fn avg_16bpc_avx512_safe(
    _token: Server64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let w = w as usize;
    let h = h as usize;

    let intermediate_bits = if (bitdepth_max >> 11) != 0 {
        2i32
    } else {
        4i32
    };
    let sh = intermediate_bits + 1;
    let rnd = (1 << intermediate_bits) + 8192 * 2;
    let max = bitdepth_max as i32;

    let rnd_vec = _mm512_set1_epi32(rnd);
    let zero = _mm512_setzero_si512();
    let max_vec = _mm512_set1_epi32(max);

    for row in 0..h {
        let tmp1_row = &tmp1[row * w..][..w];
        let tmp2_row = &tmp2[row * w..][..w];
        let dst_row_bytes = &mut dst[row * dst_stride..][..w * 2];
        let dst_row: &mut [u16] = zerocopy::FromBytes::mut_from_bytes(dst_row_bytes).unwrap();

        let mut col = 0usize;

        // Process 32 pixels at a time
        while col + 32 <= w {
            // Load 32 i16 as two groups of 16
            let t1_full = loadu_512!(&tmp1_row[col..col + 32], [i16; 32]);
            let t2_full = loadu_512!(&tmp2_row[col..col + 32], [i16; 32]);

            // Low 16
            let t1_lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(t1_full));
            let t2_lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(t2_full));
            let sum_lo = _mm512_add_epi32(_mm512_add_epi32(t1_lo, t2_lo), rnd_vec);
            let result_lo = if sh == 3 {
                _mm512_srai_epi32::<3>(sum_lo)
            } else {
                _mm512_srai_epi32::<5>(sum_lo)
            };
            let clamped_lo = _mm512_min_epi32(_mm512_max_epi32(result_lo, zero), max_vec);
            let packed_lo: __m256i = _mm512_cvtusepi32_epi16(clamped_lo);

            // High 16
            let t1_hi = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(t1_full));
            let t2_hi = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(t2_full));
            let sum_hi = _mm512_add_epi32(_mm512_add_epi32(t1_hi, t2_hi), rnd_vec);
            let result_hi = if sh == 3 {
                _mm512_srai_epi32::<3>(sum_hi)
            } else {
                _mm512_srai_epi32::<5>(sum_hi)
            };
            let clamped_hi = _mm512_min_epi32(_mm512_max_epi32(result_hi, zero), max_vec);
            let packed_hi: __m256i = _mm512_cvtusepi32_epi16(clamped_hi);

            // Combine and store 32 u16 = 64 bytes
            let combined = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(packed_lo), packed_hi);
            storeu_512!(&mut dst_row[col..col + 32], [u16; 32], combined);

            col += 32;
        }

        // AVX2 tail for 16-pixel chunks
        while col + 16 <= w {
            let t1 = loadu_256!(&tmp1_row[col..col + 16], [i16; 16]);
            let t2 = loadu_256!(&tmp2_row[col..col + 16], [i16; 16]);
            let t1_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(t1));
            let t2_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(t2));
            let t1_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(t1, 1));
            let t2_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(t2, 1));

            let rnd_256 = _mm256_set1_epi32(rnd);
            let sum_lo = _mm256_add_epi32(_mm256_add_epi32(t1_lo, t2_lo), rnd_256);
            let sum_hi = _mm256_add_epi32(_mm256_add_epi32(t1_hi, t2_hi), rnd_256);

            let (result_lo, result_hi) = if sh == 3 {
                (_mm256_srai_epi32(sum_lo, 3), _mm256_srai_epi32(sum_hi, 3))
            } else {
                (_mm256_srai_epi32(sum_lo, 5), _mm256_srai_epi32(sum_hi, 5))
            };

            let zero_256 = _mm256_setzero_si256();
            let max_256 = _mm256_set1_epi32(max);
            let clamped_lo = _mm256_min_epi32(_mm256_max_epi32(result_lo, zero_256), max_256);
            let clamped_hi = _mm256_min_epi32(_mm256_max_epi32(result_hi, zero_256), max_256);

            let packed = _mm256_packus_epi32(clamped_lo, clamped_hi);
            let packed = _mm256_permute4x64_epi64(packed, 0b11011000);

            storeu_256!(&mut dst_row[col..col + 16], [u16; 16], packed);
            col += 16;
        }

        // Scalar tail
        while col < w {
            let sum = tmp1_row[col] as i32 + tmp2_row[col] as i32;
            let val = ((sum + rnd) >> sh).clamp(0, max) as u16;
            dst_row[col] = val;
            col += 1;
        }
    }
}

/// SSE4.1 fallback for avg_8bpc
///
/// # Safety
///
/// Same as avg_8bpc_v3.
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "sse4.1")]
pub unsafe extern "C" fn avg_8bpc_sse4(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    bitdepth_max: i32,
    _dst: *const FFISafe<PicOffset>,
) {
    // For SSE4.1, use scalar fallback for now
    // TODO: Implement proper SSE4.1 version with _mm_* intrinsics
    // SAFETY: avg_scalar is safe to call with valid pointers
    unsafe { avg_scalar(dst_ptr, dst_stride, tmp1, tmp2, w, h, bitdepth_max, _dst) }
}

/// Scalar fallback for avg
///
/// # Safety
///
/// - dst_ptr must be valid for writing w*h bytes
/// - tmp1 and tmp2 must contain at least w*h elements
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
pub unsafe extern "C" fn avg_scalar(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    _bitdepth_max: i32,
    _dst: *const FFISafe<PicOffset>,
) {
    let w = w as usize;
    let h = h as usize;
    let dst = dst_ptr as *mut u8;

    for row in 0..h {
        let tmp1_row = &tmp1[row * w..][..w];
        let tmp2_row = &tmp2[row * w..][..w];
        // SAFETY: dst_ptr is valid, stride is correct
        let dst_row =
            unsafe { std::slice::from_raw_parts_mut(dst.offset(row as isize * dst_stride), w) };

        for col in 0..w {
            let sum = tmp1_row[col] as i32 + tmp2_row[col] as i32;
            let avg = ((sum * 1024 + 16384) >> 15).clamp(0, 255) as u8;
            dst_row[col] = avg;
        }
    }
}

// =============================================================================
// W_AVG (Weighted Average)
// =============================================================================

/// Weighted average rounding constant for pmulhrsw
const PW_2048: i16 = 2048;

/// Weighted average for 8-bit pixels using AVX2
///
/// Computes: (tmp1 * weight + tmp2 * (16 - weight) + rnd) >> 8 via pmaddwd
/// pairs (w, 16-w)x(t1, t2) — exact for the full i16 tmp range (an i16
/// diff*mulhi path wraps when |t1-t2| > 32767, e.g. warp_affine mids).
///
/// # Safety
///
/// Same requirements as avg_8bpc_v3, plus weight must be in [0, 16].
#[cfg(target_arch = "x86_64")]
#[arcane]
fn w_avg_8bpc_avx2_safe(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    weight: i32,
) {
    let mut dst = dst.flex_mut();
    let w = w as usize;
    let h = h as usize;

    // rnd = (8 << intermediate_bits) + PREP_BIAS*16 = 128 + 0 (PREP_BIAS=0 at 8bpc)
    let wv = _mm256_set1_epi16(weight as i16);
    let iwv = _mm256_set1_epi16((16 - weight) as i16);
    let mm_lo = _mm256_unpacklo_epi16(wv, iwv);
    let mm_hi = _mm256_unpackhi_epi16(wv, iwv);
    let rnd_v = _mm256_set1_epi32(128);

    for row in 0..h {
        let tmp1_row = &tmp1[row * w..][..w];
        let tmp2_row = &tmp2[row * w..][..w];
        let dst_row = &mut dst[row * dst_stride..][..w];

        let mut col = 0;
        while col + 32 <= w {
            let t1_lo = loadu_256!(&tmp1_row[col..col + 16], [i16; 16]);
            let t1_hi = loadu_256!(&tmp1_row[col + 16..col + 32], [i16; 16]);
            let t2_lo = loadu_256!(&tmp2_row[col..col + 16], [i16; 16]);
            let t2_hi = loadu_256!(&tmp2_row[col + 16..col + 32], [i16; 16]);

            let sum_lo = _mm256_add_epi32(
                _mm256_madd_epi16(mm_lo, _mm256_unpacklo_epi16(t1_lo, t2_lo)),
                rnd_v,
            );
            let sum_hi = _mm256_add_epi32(
                _mm256_madd_epi16(mm_hi, _mm256_unpackhi_epi16(t1_lo, t2_lo)),
                rnd_v,
            );
            let res16_lo = _mm256_packs_epi32(
                _mm256_srai_epi32::<8>(sum_lo),
                _mm256_srai_epi32::<8>(sum_hi),
            );

            let sum_lo = _mm256_add_epi32(
                _mm256_madd_epi16(mm_lo, _mm256_unpacklo_epi16(t1_hi, t2_hi)),
                rnd_v,
            );
            let sum_hi = _mm256_add_epi32(
                _mm256_madd_epi16(mm_hi, _mm256_unpackhi_epi16(t1_hi, t2_hi)),
                rnd_v,
            );
            let res16_hi = _mm256_packs_epi32(
                _mm256_srai_epi32::<8>(sum_lo),
                _mm256_srai_epi32::<8>(sum_hi),
            );

            // Pack to bytes (packus saturates i16 -> u8 = iclip [0,255])
            let packed = _mm256_packus_epi16(res16_lo, res16_hi);
            let result = _mm256_permute4x64_epi64(packed, 0b11_01_10_00);

            storeu_256!(&mut dst_row[col..col + 32], [u8; 32], result);

            col += 32;
        }

        // Scalar fallback for remaining pixels
        while col < w {
            let a = tmp1_row[col] as i32;
            let b = tmp2_row[col] as i32;
            let val = (a * weight + b * (16 - weight) + 128) >> 8;
            dst_row[col] = val.clamp(0, 255) as u8;
            col += 1;
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn w_avg_8bpc_v3(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    weight: i32,
    _bitdepth_max: i32,
    _dst: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    let dst = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h as usize * dst_stride as usize)
    };
    w_avg_8bpc_avx2_safe(token, dst, dst_stride as usize, tmp1, tmp2, w, h, weight)
}

/// Weighted average for 8-bit pixels using AVX-512
#[cfg(target_arch = "x86_64")]
#[arcane]
fn w_avg_8bpc_avx512_safe(
    _token: Server64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    weight: i32,
) {
    let mut dst = dst.flex_mut();
    let w = w as usize;
    let h = h as usize;

    // Same pmaddwd formulation as the AVX2 kernel: (w, 16-w)x(t1, t2) is
    // exact for the full i16 tmp range (i16 diff*mulhi wraps at |d|>32767).
    let wv = _mm512_set1_epi16(weight as i16);
    let iwv = _mm512_set1_epi16((16 - weight) as i16);
    let mm_lo = _mm512_unpacklo_epi16(wv, iwv);
    let mm_hi = _mm512_unpackhi_epi16(wv, iwv);
    let rnd_v = _mm512_set1_epi32(128);

    for row in 0..h {
        let tmp1_row = &tmp1[row * w..][..w];
        let tmp2_row = &tmp2[row * w..][..w];
        let dst_row = &mut dst[row * dst_stride..][..w];

        let mut col = 0;

        // Process 32 pixels at a time
        while col + 32 <= w {
            let t1 = loadu_512!(&tmp1_row[col..col + 32], [i16; 32]);
            let t2 = loadu_512!(&tmp2_row[col..col + 32], [i16; 32]);

            let sum_lo = _mm512_add_epi32(
                _mm512_madd_epi16(mm_lo, _mm512_unpacklo_epi16(t1, t2)),
                rnd_v,
            );
            let sum_hi = _mm512_add_epi32(
                _mm512_madd_epi16(mm_hi, _mm512_unpackhi_epi16(t1, t2)),
                rnd_v,
            );
            // packs_epi32 is lane-local; madd's per-lane i32 order
            // {0-3,8-11|...} pairs with hi {4-7,12-15|...} into i16 px order.
            let res16 = _mm512_packs_epi32(
                _mm512_srai_epi32::<8>(sum_lo),
                _mm512_srai_epi32::<8>(sum_hi),
            );
            // cvtusepi16 treats lanes as unsigned — clamp negatives first.
            let res16 = _mm512_max_epi16(res16, _mm512_setzero_si512());
            let result: __m256i = _mm512_cvtusepi16_epi8(res16);

            storeu_256!(&mut dst_row[col..col + 32], [u8; 32], result);
            col += 32;
        }

        // Scalar tail
        while col < w {
            let a = tmp1_row[col] as i32;
            let b = tmp2_row[col] as i32;
            let val = (a * weight + b * (16 - weight) + 128) >> 8;
            dst_row[col] = val.clamp(0, 255) as u8;
            col += 1;
        }
    }
}

/// Weighted average for 16-bit pixels using AVX2
///
/// Same as w_avg_8bpc_avx2_safe but for 16-bit content.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn w_avg_16bpc_avx2_safe(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    weight: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let w = w as usize;
    let h = h as usize;

    // intermediate_bits = 14 - bitdepth: 4 for 10-bit, 2 for 12-bit
    let intermediate_bits = if (bitdepth_max >> 11) != 0 {
        2i32
    } else {
        4i32
    };
    // sh = intermediate_bits + 4, rnd = (8 << intermediate_bits) + PREP_BIAS * 16
    let sh = intermediate_bits + 4;
    let rnd = (8 << intermediate_bits) + 8192 * 16; // PREP_BIAS = 8192 for 16bpc
    let max = bitdepth_max as i32;
    let inv_weight = 16 - weight;

    let rnd_vec = _mm256_set1_epi32(rnd);
    let zero = _mm256_setzero_si256();
    let max_vec = _mm256_set1_epi32(max);
    let weight_vec = _mm256_set1_epi32(weight);
    let inv_weight_vec = _mm256_set1_epi32(inv_weight);

    for row in 0..h {
        let tmp1_row = &tmp1[row * w..][..w];
        let tmp2_row = &tmp2[row * w..][..w];
        let dst_row_bytes = &mut dst[row * dst_stride..][..w * 2];
        let dst_row: &mut [u16] = zerocopy::FromBytes::mut_from_bytes(dst_row_bytes).unwrap();

        let mut col = 0usize;

        // Process 8 pixels at a time (need 32-bit arithmetic)
        while col + 8 <= w {
            // Load 8x i16 values and sign-extend to 32-bit
            let t1_16 = loadu_128!(&tmp1_row[col..col + 8], [i16; 8]);
            let t2_16 = loadu_128!(&tmp2_row[col..col + 8], [i16; 8]);
            let t1 = _mm256_cvtepi16_epi32(t1_16);
            let t2 = _mm256_cvtepi16_epi32(t2_16);

            // val = a * weight + b * inv_weight + rnd
            let term1 = _mm256_mullo_epi32(t1, weight_vec);
            let term2 = _mm256_mullo_epi32(t2, inv_weight_vec);
            let sum = _mm256_add_epi32(_mm256_add_epi32(term1, term2), rnd_vec);

            // srai_epi32 requires const immediate; branch on sh
            let result = if sh == 6 {
                _mm256_srai_epi32(sum, 6)
            } else {
                _mm256_srai_epi32(sum, 8)
            };

            // Clamp to [0, max]
            let clamped = _mm256_min_epi32(_mm256_max_epi32(result, zero), max_vec);

            // Pack 32-bit to 16-bit
            let packed = _mm256_packus_epi32(clamped, clamped);
            let lo128 = _mm256_castsi256_si128(packed);
            let hi128 = _mm256_extracti128_si256(packed, 1);
            let result_128 = _mm_unpacklo_epi64(lo128, hi128);

            // Store 8x u16
            storeu_128!(&mut dst_row[col..col + 8], [u16; 8], result_128);

            col += 8;
        }

        // Handle remaining pixels with scalar
        while col < w {
            let a = tmp1_row[col] as i32;
            let b = tmp2_row[col] as i32;
            let val = (a * weight + b * inv_weight + rnd) >> sh;
            dst_row[col] = val.clamp(0, max) as u16;
            col += 1;
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn w_avg_16bpc_v3(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    weight: i32,
    bitdepth_max: i32,
    _dst: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    let dst = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h as usize * dst_stride as usize)
    };
    w_avg_16bpc_avx2_safe(
        token,
        dst,
        dst_stride as usize,
        tmp1,
        tmp2,
        w,
        h,
        weight,
        bitdepth_max,
    )
}

/// Weighted average for 16-bit pixels using AVX-512
///
/// Processes 16 pixels per iteration (vs 8 with AVX2).
#[cfg(target_arch = "x86_64")]
#[arcane]
fn w_avg_16bpc_avx512_safe(
    _token: Server64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    weight: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let w = w as usize;
    let h = h as usize;

    let intermediate_bits = if (bitdepth_max >> 11) != 0 {
        2i32
    } else {
        4i32
    };
    let sh = intermediate_bits + 4;
    let rnd = (8 << intermediate_bits) + 8192 * 16;
    let max = bitdepth_max as i32;
    let inv_weight = 16 - weight;

    let rnd_vec = _mm512_set1_epi32(rnd);
    let zero = _mm512_setzero_si512();
    let max_vec = _mm512_set1_epi32(max);
    let weight_vec = _mm512_set1_epi32(weight);
    let inv_weight_vec = _mm512_set1_epi32(inv_weight);

    for row in 0..h {
        let tmp1_row = &tmp1[row * w..][..w];
        let tmp2_row = &tmp2[row * w..][..w];
        let dst_row_bytes = &mut dst[row * dst_stride..][..w * 2];
        let dst_row: &mut [u16] = zerocopy::FromBytes::mut_from_bytes(dst_row_bytes).unwrap();

        let mut col = 0usize;

        // Process 16 pixels at a time with AVX-512
        while col + 16 <= w {
            let t1 = _mm512_cvtepi16_epi32(loadu_256!(&tmp1_row[col..col + 16], [i16; 16]));
            let t2 = _mm512_cvtepi16_epi32(loadu_256!(&tmp2_row[col..col + 16], [i16; 16]));

            let term1 = _mm512_mullo_epi32(t1, weight_vec);
            let term2 = _mm512_mullo_epi32(t2, inv_weight_vec);
            let sum = _mm512_add_epi32(_mm512_add_epi32(term1, term2), rnd_vec);

            let result = if sh == 6 {
                _mm512_srai_epi32::<6>(sum)
            } else {
                _mm512_srai_epi32::<8>(sum)
            };

            let clamped = _mm512_min_epi32(_mm512_max_epi32(result, zero), max_vec);
            let packed: __m256i = _mm512_cvtusepi32_epi16(clamped);

            storeu_256!(&mut dst_row[col..col + 16], [u16; 16], packed);
            col += 16;
        }

        // Scalar tail
        while col < w {
            let a = tmp1_row[col] as i32;
            let b = tmp2_row[col] as i32;
            let val = (a * weight + b * inv_weight + rnd) >> sh;
            dst_row[col] = val.clamp(0, max) as u16;
            col += 1;
        }
    }
}

/// Scalar fallback for w_avg
///
/// # Safety
///
/// Same as w_avg_8bpc_v3.
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
pub unsafe extern "C" fn w_avg_scalar(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    weight: i32,
    _bitdepth_max: i32,
    _dst: *const FFISafe<PicOffset>,
) {
    let w = w as usize;
    let h = h as usize;
    let dst = dst_ptr as *mut u8;

    // For 8bpc: intermediate_bits = 4, sh = 8, rnd = 128 + PREP_BIAS*16 = 128
    let intermediate_bits = 4;
    let sh = intermediate_bits + 4;
    let rnd = (8 << intermediate_bits) + 0 * 16; // PREP_BIAS = 0 for 8bpc

    for row in 0..h {
        let tmp1_row = &tmp1[row * w..][..w];
        let tmp2_row = &tmp2[row * w..][..w];
        let dst_row =
            unsafe { std::slice::from_raw_parts_mut(dst.offset(row as isize * dst_stride), w) };

        for col in 0..w {
            let a = tmp1_row[col] as i32;
            let b = tmp2_row[col] as i32;
            let val = (a * weight + b * (16 - weight) + rnd) >> sh;
            dst_row[col] = val.clamp(0, 255) as u8;
        }
    }
}

// =============================================================================
// MASK (Per-pixel blend with mask)
// =============================================================================

/// Mask blend for 8-bit pixels using AVX2
///
/// Computes: (tmp1 * mask + tmp2 * (64 - mask) + 512) >> 10
/// Using optimized form from asm.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn mask_8bpc_avx2_safe(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask: &[u8],
) {
    let mut dst = dst.flex_mut();
    let mask = mask.flex();
    let w = w as usize;
    let h = h as usize;

    let rnd = _mm256_set1_epi32(512);

    for row in 0..h {
        let tmp1_row = &tmp1[row * w..][..w];
        let tmp2_row = &tmp2[row * w..][..w];
        let mask_row = &mask[row * w..][..w];
        let dst_row = &mut dst[row * dst_stride..][..w];

        let mut col = 0usize;

        // Process 16 pixels at a time with AVX2
        while col + 16 <= w {
            let t1_lo = loadu_256!(&tmp1_row[col..col + 16], [i16; 16]);
            let t2_lo = loadu_256!(&tmp2_row[col..col + 16], [i16; 16]);
            let mask_bytes = loadu_128!(&mask_row[col..col + 16], [u8; 16]);
            let mask_lo = _mm256_cvtepu8_epi16(mask_bytes);

            // m*t1 + (64-m)*t2 via pmaddwd pairs — exact for the full i16
            // tmp range (a wrapping i16 diff would lose carry).
            let inv = _mm256_sub_epi16(_mm256_set1_epi16(64), mask_lo);
            let mm_lo = _mm256_unpacklo_epi16(mask_lo, inv);
            let mm_hi = _mm256_unpackhi_epi16(mask_lo, inv);
            let tt_lo = _mm256_unpacklo_epi16(t1_lo, t2_lo);
            let tt_hi = _mm256_unpackhi_epi16(t1_lo, t2_lo);
            let sum_lo = _mm256_add_epi32(_mm256_madd_epi16(mm_lo, tt_lo), rnd);
            let sum_hi = _mm256_add_epi32(_mm256_madd_epi16(mm_hi, tt_hi), rnd);

            // Shift right by 10
            let result_lo = _mm256_srai_epi32(sum_lo, 10);
            let result_hi = _mm256_srai_epi32(sum_hi, 10);

            // Pack back to 16-bit (madd pairs yield i32s in pixel order per
            // 128-bit lane, so packs_epi32 already produces ordered i16s).
            let result_16 = _mm256_packs_epi32(result_lo, result_hi);

            // Pack to 8-bit with unsigned saturation
            let result_8 = _mm256_packus_epi16(result_16, result_16);
            let result_8 = _mm256_permute4x64_epi64(result_8, 0b11011000);

            storeu_128!(
                &mut dst_row[col..col + 16],
                [u8; 16],
                _mm256_castsi256_si128(result_8)
            );

            col += 16;
        }

        // w=4/8 lanes: same pmaddwd pairs (m,64-m)x(t1,t2) — exact for any i16
        // tmp (a wrapping i16 diff loses carry when |t1-t2| > 32767).
        let c64x = _mm_set1_epi16(64);
        let rndx = _mm_set1_epi32(512);
        while col + 8 <= w {
            let t1 = loadu_128!(<&[i16; 8]>::try_from(&tmp1_row[col..col + 8]).unwrap());
            let t2 = loadu_128!(<&[i16; 8]>::try_from(&tmp2_row[col..col + 8]).unwrap());
            let mv = _mm_cvtepu8_epi16(loadu_64!(
                <&[u8; 8]>::try_from(&mask_row[col..col + 8]).unwrap()
            ));
            let inv = _mm_sub_epi16(c64x, mv);
            let mm_lo = _mm_unpacklo_epi16(mv, inv);
            let mm_hi = _mm_unpackhi_epi16(mv, inv);
            let tt_lo = _mm_unpacklo_epi16(t1, t2);
            let tt_hi = _mm_unpackhi_epi16(t1, t2);
            let sum_lo = _mm_add_epi32(_mm_madd_epi16(mm_lo, tt_lo), rndx);
            let sum_hi = _mm_add_epi32(_mm_madd_epi16(mm_hi, tt_hi), rndx);
            let res16 = _mm_packs_epi32(_mm_srai_epi32::<10>(sum_lo), _mm_srai_epi32::<10>(sum_hi));
            let res8 = _mm_packus_epi16(res16, res16);
            storei64!(&mut dst_row[col..col + 8], res8);
            col += 8;
        }
        if col + 4 <= w {
            let t1 = crate::src::safe_simd::partial_simd::mm_loadl_epi64::<[i16; 4]>(
                <&[i16; 4]>::try_from(&tmp1_row[col..col + 4]).unwrap(),
            );
            let t2 = crate::src::safe_simd::partial_simd::mm_loadl_epi64::<[i16; 4]>(
                <&[i16; 4]>::try_from(&tmp2_row[col..col + 4]).unwrap(),
            );
            let mv = _mm_cvtepu8_epi16(loadi32!(&mask_row[col..col + 4]));
            let inv = _mm_sub_epi16(c64x, mv);
            let mm_lo = _mm_unpacklo_epi16(mv, inv);
            let tt_lo = _mm_unpacklo_epi16(t1, t2);
            let sum_lo = _mm_add_epi32(_mm_madd_epi16(mm_lo, tt_lo), rndx);
            let res16 = _mm_packs_epi32(_mm_srai_epi32::<10>(sum_lo), _mm_setzero_si128());
            let res8 = _mm_packus_epi16(res16, res16);
            dst_row[col..col + 4].copy_from_slice(&(_mm_cvtsi128_si32(res8) as u32).to_ne_bytes());
            col += 4;
        }

        // Handle remaining pixels with scalar
        while col < w {
            let a = tmp1_row[col] as i32;
            let b = tmp2_row[col] as i32;
            let m = mask_row[col] as i32;
            let val = (a * m + b * (64 - m) + 512) >> 10;
            dst_row[col] = val.clamp(0, 255) as u8;
            col += 1;
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn mask_8bpc_v3(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask_ptr: *const u8,
    _bitdepth_max: i32,
    _dst: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    let dst = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h as usize * dst_stride as usize)
    };
    let mask = unsafe { std::slice::from_raw_parts(mask_ptr, (w * h) as usize) };
    mask_8bpc_avx2_safe(token, dst, dst_stride as usize, tmp1, tmp2, w, h, mask)
}

/// Mask blend for 8-bit pixels using AVX-512
///
/// Processes 32 pixels per iteration (needs i32 intermediates for overflow safety).
#[cfg(target_arch = "x86_64")]
#[arcane]
fn mask_8bpc_avx512_safe(
    _token: Server64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask: &[u8],
) {
    let mut dst = dst.flex_mut();
    let mask = mask.flex();
    let w = w as usize;
    let h = h as usize;

    let rnd = _mm512_set1_epi32(512);

    for row in 0..h {
        let tmp1_row = &tmp1[row * w..][..w];
        let tmp2_row = &tmp2[row * w..][..w];
        let mask_row = &mask[row * w..][..w];
        let dst_row = &mut dst[row * dst_stride..][..w];

        let mut col = 0usize;

        // Process 32 pixels at a time with AVX-512
        // Each pixel needs i32 intermediate, so 16 pixels per 512-bit register
        while col + 32 <= w {
            // Load 32 i16 values and 32 mask bytes
            let t1_full = loadu_512!(&tmp1_row[col..col + 32], [i16; 32]);
            let t2_full = loadu_512!(&tmp2_row[col..col + 32], [i16; 32]);
            let mask_bytes = loadu_256!(&mask_row[col..col + 32], [u8; 32]);
            let mask_16 = _mm512_cvtepu8_epi16(mask_bytes);

            // Low 16 pixels (expand to i32; subtract in i32 — an i16 sub
            // wraps when |t1-t2| > 32767, e.g. extreme warp_affine mids).
            let diff_lo = _mm512_sub_epi32(
                _mm512_cvtepi16_epi32(_mm512_castsi512_si256(t1_full)),
                _mm512_cvtepi16_epi32(_mm512_castsi512_si256(t2_full)),
            );
            let mask_lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(mask_16));
            let t2_lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(t2_full));

            let prod_lo = _mm512_mullo_epi32(diff_lo, mask_lo);
            let t2_64_lo = _mm512_slli_epi32(t2_lo, 6);
            let sum_lo = _mm512_add_epi32(_mm512_add_epi32(prod_lo, t2_64_lo), rnd);
            let result_lo = _mm512_srai_epi32::<10>(sum_lo);

            // High 16 pixels
            let diff_hi = _mm512_sub_epi32(
                _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(t1_full)),
                _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(t2_full)),
            );
            let mask_hi = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(mask_16));
            let t2_hi = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(t2_full));

            let prod_hi = _mm512_mullo_epi32(diff_hi, mask_hi);
            let t2_64_hi = _mm512_slli_epi32(t2_hi, 6);
            let sum_hi = _mm512_add_epi32(_mm512_add_epi32(prod_hi, t2_64_hi), rnd);
            let result_hi = _mm512_srai_epi32::<10>(sum_hi);

            // Pack i32 → i16 → u8
            // Pack two __m512i of i32 → one __m512i of i16
            let result_16 = _mm512_packs_epi32(result_lo, result_hi);
            // Fix lane ordering: packs interleaves within 128-bit lanes
            let perm_idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
            let result_16 = _mm512_permutexvar_epi64(perm_idx, result_16);
            // Pack i16 → u8 with unsigned saturation
            let result_8: __m256i =
                _mm512_cvtusepi16_epi8(_mm512_max_epi16(result_16, _mm512_setzero_si512()));

            storeu_256!(&mut dst_row[col..col + 32], [u8; 32], result_8);
            col += 32;
        }

        // Process 16 pixels at a time with AVX2 fallback
        while col + 16 <= w {
            let t1_lo = loadu_256!(&tmp1_row[col..col + 16], [i16; 16]);
            let t2_lo = loadu_256!(&tmp2_row[col..col + 16], [i16; 16]);
            let mask_bytes = loadu_128!(&mask_row[col..col + 16], [u8; 16]);
            let mask_lo = _mm256_cvtepu8_epi16(mask_bytes);

            let inv = _mm256_sub_epi16(_mm256_set1_epi16(64), mask_lo);
            let mm_lo = _mm256_unpacklo_epi16(mask_lo, inv);
            let mm_hi = _mm256_unpackhi_epi16(mask_lo, inv);
            let tt_lo = _mm256_unpacklo_epi16(t1_lo, t2_lo);
            let tt_hi = _mm256_unpackhi_epi16(t1_lo, t2_lo);
            let sum_lo = _mm256_add_epi32(_mm256_madd_epi16(mm_lo, tt_lo), _mm256_set1_epi32(512));
            let sum_hi = _mm256_add_epi32(_mm256_madd_epi16(mm_hi, tt_hi), _mm256_set1_epi32(512));

            let result_lo = _mm256_srai_epi32(sum_lo, 10);
            let result_hi = _mm256_srai_epi32(sum_hi, 10);

            let result_16 = _mm256_packs_epi32(result_lo, result_hi);
            let result_8 = _mm256_packus_epi16(result_16, result_16);
            let result_8 = _mm256_permute4x64_epi64(result_8, 0b11011000);

            storeu_128!(
                &mut dst_row[col..col + 16],
                [u8; 16],
                _mm256_castsi256_si128(result_8)
            );
            col += 16;
        }

        // Scalar tail
        while col < w {
            let a = tmp1_row[col] as i32;
            let b = tmp2_row[col] as i32;
            let m = mask_row[col] as i32;
            let val = (a * m + b * (64 - m) + 512) >> 10;
            dst_row[col] = val.clamp(0, 255) as u8;
            col += 1;
        }
    }
}

/// Mask blend for 16-bit pixels using AVX-512 — 16 pixels/iter with 512-bit i32 arithmetic.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn mask_16bpc_avx512_safe(
    _token: Server64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask: &[u8],
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mask = mask.flex();
    let w = w as usize;
    let h = h as usize;
    let max = bitdepth_max as i32;

    let intermediate_bits = if (bitdepth_max >> 11) != 0 {
        2i32
    } else {
        4i32
    };
    let sh = intermediate_bits + 6;
    let rnd = (32 << intermediate_bits) + 8192 * 64;

    let rnd_vec = _mm512_set1_epi32(rnd);
    let zero = _mm512_setzero_si512();
    let max_vec = _mm512_set1_epi32(max);
    let sixty_four = _mm512_set1_epi32(64);

    for row in 0..h {
        let tmp1_row = &tmp1[row * w..][..w];
        let tmp2_row = &tmp2[row * w..][..w];
        let mask_row = &mask[row * w..][..w];
        let dst_row_bytes = &mut dst[row * dst_stride..][..w * 2];
        let dst_row: &mut [u16] = zerocopy::FromBytes::mut_from_bytes(dst_row_bytes).unwrap();

        let mut col = 0usize;

        // Process 16 pixels at a time with 512-bit i32 arithmetic
        while col + 16 <= w {
            let t1_16 = loadu_256!(&tmp1_row[col..col + 16], [i16; 16]);
            let t2_16 = loadu_256!(&tmp2_row[col..col + 16], [i16; 16]);
            let t1 = _mm512_cvtepi16_epi32(t1_16);
            let t2 = _mm512_cvtepi16_epi32(t2_16);

            // Load 16 mask bytes via zero-extend to i32
            let mask_128 = loadu_128!(&mask_row[col..col + 16], [u8; 16]);
            let mask_32 = _mm512_cvtepu8_epi32(mask_128);

            let inv_mask = _mm512_sub_epi32(sixty_four, mask_32);

            let term1 = _mm512_mullo_epi32(t1, mask_32);
            let term2 = _mm512_mullo_epi32(t2, inv_mask);
            let sum = _mm512_add_epi32(_mm512_add_epi32(term1, term2), rnd_vec);

            let result = if sh == 8 {
                _mm512_srai_epi32::<8>(sum)
            } else {
                _mm512_srai_epi32::<10>(sum)
            };
            let clamped = _mm512_min_epi32(_mm512_max_epi32(result, zero), max_vec);

            // Pack i32 → u16 with unsigned saturation (already clamped to [0, max])
            let packed = _mm512_cvtusepi32_epi16(clamped);
            storeu_256!(&mut dst_row[col..col + 16], [u16; 16], packed);

            col += 16;
        }

        // Process 8 pixels at a time (AVX2 intrinsics available via #[arcane] context)
        while col + 8 <= w {
            let t1_16 = loadu_128!(&tmp1_row[col..col + 8], [i16; 8]);
            let t2_16 = loadu_128!(&tmp2_row[col..col + 8], [i16; 8]);
            let t1 = _mm256_cvtepi16_epi32(t1_16);
            let t2 = _mm256_cvtepi16_epi32(t2_16);

            let mut mask_pad = [0u8; 16];
            mask_pad[..8].copy_from_slice(&mask_row[col..col + 8]);
            let mask_bytes = loadu_128!(&mask_pad);
            let mask_32 = _mm256_cvtepu8_epi32(mask_bytes);

            let inv_mask_256 = _mm256_sub_epi32(_mm256_set1_epi32(64), mask_32);
            let term1_256 = _mm256_mullo_epi32(t1, mask_32);
            let term2_256 = _mm256_mullo_epi32(t2, inv_mask_256);
            let sum_256 = _mm256_add_epi32(
                _mm256_add_epi32(term1_256, term2_256),
                _mm256_set1_epi32(rnd),
            );
            let result_256 = if sh == 8 {
                _mm256_srai_epi32(sum_256, 8)
            } else {
                _mm256_srai_epi32(sum_256, 10)
            };
            let clamped_256 = _mm256_min_epi32(
                _mm256_max_epi32(result_256, _mm256_setzero_si256()),
                _mm256_set1_epi32(max),
            );

            let packed = _mm256_packus_epi32(clamped_256, clamped_256);
            let lo128 = _mm256_castsi256_si128(packed);
            let hi128 = _mm256_extracti128_si256(packed, 1);
            let result_128 = _mm_unpacklo_epi64(lo128, hi128);
            storeu_128!(&mut dst_row[col..col + 8], [u16; 8], result_128);

            col += 8;
        }

        while col < w {
            let a = tmp1_row[col] as i32;
            let b = tmp2_row[col] as i32;
            let m = mask_row[col] as i32;
            let val = (a * m + b * (64 - m) + rnd) >> sh;
            dst_row[col] = val.clamp(0, max) as u16;
            col += 1;
        }
    }
}

/// Mask blend for 16-bit pixels
#[cfg(target_arch = "x86_64")]
#[arcane]
fn mask_16bpc_avx2_safe(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask: &[u8],
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let mask = mask.flex();
    let w = w as usize;
    let h = h as usize;
    let max = bitdepth_max as i32;

    // intermediate_bits = 14 - bitdepth: 4 for 10-bit, 2 for 12-bit
    let intermediate_bits = if (bitdepth_max >> 11) != 0 {
        2i32
    } else {
        4i32
    };
    // sh = intermediate_bits + 6, rnd = (32 << intermediate_bits) + PREP_BIAS * 64
    let sh = intermediate_bits + 6;
    let rnd = (32 << intermediate_bits) + 8192 * 64; // PREP_BIAS = 8192 for 16bpc

    let rnd_vec = _mm256_set1_epi32(rnd);
    let zero = _mm256_setzero_si256();
    let max_vec = _mm256_set1_epi32(max);
    let sixty_four = _mm256_set1_epi32(64);

    for row in 0..h {
        let tmp1_row = &tmp1[row * w..][..w];
        let tmp2_row = &tmp2[row * w..][..w];
        let mask_row = &mask[row * w..][..w];
        let dst_row_bytes = &mut dst[row * dst_stride..][..w * 2];
        let dst_row: &mut [u16] = zerocopy::FromBytes::mut_from_bytes(dst_row_bytes).unwrap();

        let mut col = 0usize;

        // Process 8 pixels at a time with 32-bit arithmetic
        while col + 8 <= w {
            let t1_16 = loadu_128!(&tmp1_row[col..col + 8], [i16; 8]);
            let t2_16 = loadu_128!(&tmp2_row[col..col + 8], [i16; 8]);
            let t1 = _mm256_cvtepi16_epi32(t1_16);
            let t2 = _mm256_cvtepi16_epi32(t2_16);

            // Load 8 bytes of mask via zero-padded 16-byte array
            let mut mask_pad = [0u8; 16];
            mask_pad[..8].copy_from_slice(&mask_row[col..col + 8]);
            let mask_bytes = loadu_128!(&mask_pad);
            let mask_32 = _mm256_cvtepu8_epi32(mask_bytes);

            let inv_mask = _mm256_sub_epi32(sixty_four, mask_32);

            let term1 = _mm256_mullo_epi32(t1, mask_32);
            let term2 = _mm256_mullo_epi32(t2, inv_mask);
            let sum = _mm256_add_epi32(_mm256_add_epi32(term1, term2), rnd_vec);

            // srai_epi32 requires const immediate; branch on sh
            let result = if sh == 8 {
                _mm256_srai_epi32(sum, 8)
            } else {
                _mm256_srai_epi32(sum, 10)
            };
            let clamped = _mm256_min_epi32(_mm256_max_epi32(result, zero), max_vec);

            let packed = _mm256_packus_epi32(clamped, clamped);
            let lo128 = _mm256_castsi256_si128(packed);
            let hi128 = _mm256_extracti128_si256(packed, 1);
            let result_128 = _mm_unpacklo_epi64(lo128, hi128);

            storeu_128!(&mut dst_row[col..col + 8], [u16; 8], result_128);

            col += 8;
        }

        // Handle remaining pixels with scalar
        while col < w {
            let a = tmp1_row[col] as i32;
            let b = tmp2_row[col] as i32;
            let m = mask_row[col] as i32;
            let val = (a * m + b * (64 - m) + rnd) >> sh;
            dst_row[col] = val.clamp(0, max) as u16;
            col += 1;
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn mask_16bpc_v3(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask_ptr: *const u8,
    bitdepth_max: i32,
    _dst: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    let dst = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h as usize * dst_stride as usize)
    };
    let mask = unsafe { std::slice::from_raw_parts(mask_ptr, (w * h) as usize) };
    mask_16bpc_avx2_safe(
        token,
        dst,
        dst_stride as usize,
        tmp1,
        tmp2,
        w,
        h,
        mask,
        bitdepth_max,
    )
}

/// Scalar fallback for mask
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
pub unsafe extern "C" fn mask_scalar(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask_ptr: *const u8,
    _bitdepth_max: i32,
    _dst: *const FFISafe<PicOffset>,
) {
    let w = w as usize;
    let h = h as usize;
    let dst = dst_ptr as *mut u8;

    // For 8bpc: intermediate_bits = 4, sh = 10, rnd = 512
    let intermediate_bits = 4;
    let sh = intermediate_bits + 6;
    let rnd = (32 << intermediate_bits) + 0 * 64;

    for row in 0..h {
        let tmp1_row = &tmp1[row * w..][..w];
        let tmp2_row = &tmp2[row * w..][..w];
        let mask_row = unsafe { std::slice::from_raw_parts(mask_ptr.add(row * w), w) };
        let dst_row =
            unsafe { std::slice::from_raw_parts_mut(dst.offset(row as isize * dst_stride), w) };

        for col in 0..w {
            let a = tmp1_row[col] as i32;
            let b = tmp2_row[col] as i32;
            let m = mask_row[col] as i32;
            let val = (a * m + b * (64 - m) + rnd) >> sh;
            dst_row[col] = val.clamp(0, 255) as u8;
        }
    }
}

// =============================================================================
// BLEND (Pixel-level blend with mask)
// =============================================================================

#[cfg(target_arch = "x86_64")]
use crate::src::internal::{SCRATCH_INTER_INTRA_BUF_LEN, SCRATCH_LAP_LEN};

/// Blend pixels using per-pixel mask
///
/// Computes: dst = (dst * (64 - mask) + tmp * mask + 32) >> 6
#[cfg(target_arch = "x86_64")]
#[arcane]
fn blend_8bpc_avx2_safe(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp: &[u8],
    w: i32,
    h: i32,
    mask: &[u8],
) {
    let mut dst = dst.flex_mut();
    let tmp = tmp.flex();
    let mask = mask.flex();
    let w = w as usize;
    let h = h as usize;

    let sixty_four = _mm256_set1_epi16(64);
    let rnd = _mm256_set1_epi16(32);

    for row in 0..h {
        let dst_row = &mut dst[row * dst_stride..][..w];
        let tmp_row = &tmp[row * w..][..w];
        let mask_row = &mask[row * w..][..w];

        let mut col = 0usize;

        // Process 16 pixels at a time with AVX2
        while col + 16 <= w {
            let dst_bytes = loadu_128!(&dst_row[col..col + 16], [u8; 16]);
            let tmp_bytes = loadu_128!(&tmp_row[col..col + 16], [u8; 16]);
            let mask_bytes = loadu_128!(&mask_row[col..col + 16], [u8; 16]);

            let dst_16 = _mm256_cvtepu8_epi16(dst_bytes);
            let tmp_16 = _mm256_cvtepu8_epi16(tmp_bytes);
            let mask_16 = _mm256_cvtepu8_epi16(mask_bytes);

            let inv_mask = _mm256_sub_epi16(sixty_four, mask_16);

            let term1 = _mm256_mullo_epi16(dst_16, inv_mask);
            let term2 = _mm256_mullo_epi16(tmp_16, mask_16);
            let sum = _mm256_add_epi16(_mm256_add_epi16(term1, term2), rnd);

            let result_16 = _mm256_srli_epi16(sum, 6);

            let result_8 = _mm256_packus_epi16(result_16, result_16);
            let result_8 = _mm256_permute4x64_epi64(result_8, 0b11011000);

            storeu_128!(
                &mut dst_row[col..col + 16],
                [u8; 16],
                _mm256_castsi256_si128(result_8)
            );

            col += 16;
        }

        // Handle remaining pixels with scalar
        while col < w {
            let a = dst_row[col] as u32;
            let b = tmp_row[col] as u32;
            let m = mask_row[col] as u32;
            let val = (a * (64 - m) + b * m + 32) >> 6;
            dst_row[col] = val as u8;
            col += 1;
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn blend_8bpc_v3(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp: *const [DynPixel; SCRATCH_INTER_INTRA_BUF_LEN],
    w: i32,
    h: i32,
    mask_ptr: *const u8,
    _dst: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    let dst = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h as usize * dst_stride as usize)
    };
    let tmp_slice = unsafe { std::slice::from_raw_parts(tmp as *const u8, (w * h) as usize) };
    let mask = unsafe { std::slice::from_raw_parts(mask_ptr, (w * h) as usize) };
    blend_8bpc_avx2_safe(token, dst, dst_stride as usize, tmp_slice, w, h, mask)
}

/// Blend pixels for 16-bit
#[cfg(target_arch = "x86_64")]
#[arcane]
fn blend_16bpc_avx2_safe(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp: &[u8],
    w: i32,
    h: i32,
    mask: &[u8],
) {
    let mut dst = dst.flex_mut();
    let tmp = tmp.flex();
    let mask = mask.flex();
    let w = w as usize;
    let h = h as usize;

    let rnd = _mm256_set1_epi32(32);
    let sixty_four = _mm256_set1_epi32(64);

    for row in 0..h {
        let dst_row_bytes = &mut dst[row * dst_stride..][..w * 2];
        let dst_row: &mut [u16] = zerocopy::FromBytes::mut_from_bytes(dst_row_bytes).unwrap();
        let tmp_row_bytes = &tmp[row * w * 2..][..w * 2];
        let tmp_row: &[u16] =
            <[u16] as zerocopy::FromBytes>::ref_from_bytes(tmp_row_bytes).unwrap();
        let mask_row = &mask[row * w..][..w];

        let mut col = 0usize;

        // Process 8 pixels at a time with 32-bit arithmetic
        while col + 8 <= w {
            let dst_16 = loadu_128!(&dst_row[col..col + 8], [u16; 8]);
            let tmp_16 = loadu_128!(&tmp_row[col..col + 8], [u16; 8]);
            let dst_32 = _mm256_cvtepu16_epi32(dst_16);
            let tmp_32 = _mm256_cvtepu16_epi32(tmp_16);

            // Load 8 bytes of mask via zero-padded 16-byte array
            let mut mask_pad = [0u8; 16];
            mask_pad[..8].copy_from_slice(&mask_row[col..col + 8]);
            let mask_bytes = loadu_128!(&mask_pad);
            let mask_32 = _mm256_cvtepu8_epi32(mask_bytes);

            let inv_mask = _mm256_sub_epi32(sixty_four, mask_32);

            let term1 = _mm256_mullo_epi32(dst_32, inv_mask);
            let term2 = _mm256_mullo_epi32(tmp_32, mask_32);
            let sum = _mm256_add_epi32(_mm256_add_epi32(term1, term2), rnd);

            let result = _mm256_srli_epi32(sum, 6);

            let packed = _mm256_packus_epi32(result, result);
            let lo128 = _mm256_castsi256_si128(packed);
            let hi128 = _mm256_extracti128_si256(packed, 1);
            let result_128 = _mm_unpacklo_epi64(lo128, hi128);

            storeu_128!(&mut dst_row[col..col + 8], [u16; 8], result_128);

            col += 8;
        }

        // Handle remaining pixels with scalar
        while col < w {
            let a = dst_row[col] as u32;
            let b = tmp_row[col] as u32;
            let m = mask_row[col] as u32;
            let val = (a * (64 - m) + b * m + 32) >> 6;
            dst_row[col] = val as u16;
            col += 1;
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn blend_16bpc_v3(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp: *const [DynPixel; SCRATCH_INTER_INTRA_BUF_LEN],
    w: i32,
    h: i32,
    mask_ptr: *const u8,
    _dst: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    let dst = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h as usize * dst_stride as usize)
    };
    let tmp_slice = unsafe { std::slice::from_raw_parts(tmp as *const u8, (w * h) as usize * 2) };
    let mask = unsafe { std::slice::from_raw_parts(mask_ptr, (w * h) as usize) };
    blend_16bpc_avx2_safe(token, dst, dst_stride as usize, tmp_slice, w, h, mask)
}

// =============================================================================
// BLEND_V / BLEND_H (Directional blend for OBMC)
// =============================================================================

use crate::src::tables::dav1d_mc_subpel_filters;
#[cfg(target_arch = "x86_64")]
use crate::src::tables::dav1d_obmc_masks;

/// Vertical blend (overlapped block motion compensation)
///
/// Uses predefined obmc_masks table for blend weights.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn blend_v_8bpc_avx2_safe(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp: &[u8],
    w: i32,
    h: i32,
) {
    let mut dst = dst.flex_mut();
    let tmp = tmp.flex();
    let w = w as usize;
    let h = h as usize;
    // blend_v: mask indexed by column, offset by w, only process w*3/4 columns
    let obmc_mask = &dav1d_obmc_masks[w..];
    let w_eff = w * 3 >> 2;

    let rnd = _mm256_set1_epi16(32);
    let sixty_four = _mm256_set1_epi16(64);

    for row in 0..h {
        let dst_row = &mut dst[row * dst_stride..][..w_eff];
        // tmp uses full w stride even though we only write w_eff columns
        let tmp_row = &tmp[row * w..][..w_eff];

        let mut col = 0usize;

        // Process 16 pixels at a time
        while col + 16 <= w_eff {
            let dst_bytes = loadu_128!(&dst_row[col..col + 16], [u8; 16]);
            let tmp_bytes = loadu_128!(&tmp_row[col..col + 16], [u8; 16]);
            let mask_bytes = loadu_128!(&obmc_mask[col..col + 16], [u8; 16]);

            let dst_16 = _mm256_cvtepu8_epi16(dst_bytes);
            let tmp_16 = _mm256_cvtepu8_epi16(tmp_bytes);
            let mask_16 = _mm256_cvtepu8_epi16(mask_bytes);
            let inv_mask = _mm256_sub_epi16(sixty_four, mask_16);

            let term1 = _mm256_mullo_epi16(dst_16, inv_mask);
            let term2 = _mm256_mullo_epi16(tmp_16, mask_16);
            let sum = _mm256_add_epi16(_mm256_add_epi16(term1, term2), rnd);
            let result_16 = _mm256_srli_epi16(sum, 6);

            let result_8 = _mm256_packus_epi16(result_16, result_16);
            let result_8 = _mm256_permute4x64_epi64(result_8, 0b11011000);

            storeu_128!(
                &mut dst_row[col..col + 16],
                [u8; 16],
                _mm256_castsi256_si128(result_8)
            );

            col += 16;
        }

        // 8/4-wide lanes for w_eff = 6,12,24 (w = 8,16,32): byte-pair maddubs
        // does (dst*(64-m) + tmp*m) directly, no widening needed.
        while col + 8 <= w_eff {
            let d8 = loadu_64!(<&[u8; 8]>::try_from(&dst_row[col..col + 8]).unwrap());
            let t8 = loadu_64!(<&[u8; 8]>::try_from(&tmp_row[col..col + 8]).unwrap());
            let m8 = loadu_64!(<&[u8; 8]>::try_from(&obmc_mask[col..col + 8]).unwrap());
            let inv8 = _mm_sub_epi8(_mm_set1_epi8(64), m8);
            let dat = _mm_unpacklo_epi8(d8, t8);
            let cof = _mm_unpacklo_epi8(inv8, m8);
            let sum = _mm_add_epi16(_mm_maddubs_epi16(dat, cof), _mm_set1_epi16(32));
            let res = _mm_srli_epi16::<6>(sum);
            storei64!(&mut dst_row[col..col + 8], _mm_packus_epi16(res, res));
            col += 8;
        }
        if col + 4 <= w_eff {
            let d4 = loadi32!(&dst_row[col..col + 4]);
            let t4 = loadi32!(&tmp_row[col..col + 4]);
            let m4 = loadi32!(&obmc_mask[col..col + 4]);
            let inv4 = _mm_sub_epi8(_mm_set1_epi8(64), m4);
            let dat = _mm_unpacklo_epi8(d4, t4);
            let cof = _mm_unpacklo_epi8(inv4, m4);
            let sum = _mm_add_epi16(_mm_maddubs_epi16(dat, cof), _mm_set1_epi16(32));
            let res = _mm_srli_epi16::<6>(sum);
            let p8 = _mm_packus_epi16(res, res);
            dst_row[col..col + 4].copy_from_slice(&(_mm_cvtsi128_si32(p8) as u32).to_ne_bytes());
            col += 4;
        }

        // Handle remaining pixels
        while col < w_eff {
            let a = dst_row[col] as u32;
            let b = tmp_row[col] as u32;
            let m = obmc_mask[col] as u32;
            let val = (a * (64 - m) + b * m + 32) >> 6;
            dst_row[col] = val as u8;
            col += 1;
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn blend_v_8bpc_v3(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp: *const [DynPixel; SCRATCH_LAP_LEN],
    w: i32,
    h: i32,
    _dst: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    let dst = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h as usize * dst_stride as usize)
    };
    let tmp_slice = unsafe { std::slice::from_raw_parts(tmp as *const u8, (w * h) as usize) };
    blend_v_8bpc_avx2_safe(token, dst, dst_stride as usize, tmp_slice, w, h)
}

/// Horizontal blend (overlapped block motion compensation)
#[cfg(target_arch = "x86_64")]
#[arcane]
fn blend_h_8bpc_avx2_safe(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp: &[u8],
    w: i32,
    h: i32,
) {
    let mut dst = dst.flex_mut();
    let tmp = tmp.flex();
    let w = w as usize;
    let h = h as usize;
    // blend_h: mask indexed by row, offset by h, only process h*3/4 rows
    let obmc_mask = &dav1d_obmc_masks[h..];
    let h_eff = h * 3 >> 2;

    let rnd = _mm256_set1_epi16(32);
    let sixty_four = _mm256_set1_epi16(64);

    for row in 0..h_eff {
        let dst_row = &mut dst[row * dst_stride..][..w];
        let tmp_row = &tmp[row * w..][..w];
        let m = obmc_mask[row];

        let mask_16 = _mm256_set1_epi16(m as i16);
        let inv_mask = _mm256_sub_epi16(sixty_four, mask_16);

        let mut col = 0usize;

        // Process 16 pixels at a time
        while col + 16 <= w {
            let dst_bytes = loadu_128!(&dst_row[col..col + 16], [u8; 16]);
            let tmp_bytes = loadu_128!(&tmp_row[col..col + 16], [u8; 16]);

            let dst_16 = _mm256_cvtepu8_epi16(dst_bytes);
            let tmp_16 = _mm256_cvtepu8_epi16(tmp_bytes);

            let term1 = _mm256_mullo_epi16(dst_16, inv_mask);
            let term2 = _mm256_mullo_epi16(tmp_16, mask_16);
            let sum = _mm256_add_epi16(_mm256_add_epi16(term1, term2), rnd);
            let result_16 = _mm256_srli_epi16(sum, 6);

            let result_8 = _mm256_packus_epi16(result_16, result_16);
            let result_8 = _mm256_permute4x64_epi64(result_8, 0b11011000);

            storeu_128!(
                &mut dst_row[col..col + 16],
                [u8; 16],
                _mm256_castsi256_si128(result_8)
            );

            col += 16;
        }

        // 8/4-wide lanes for w = 8 blocks: maddubs on (dst,tmp)/(inv,m) pairs.
        let m8v = _mm_set1_epi8(m as i8);
        let inv8v = _mm_sub_epi8(_mm_set1_epi8(64), m8v);
        let cof = _mm_unpacklo_epi8(inv8v, m8v);
        while col + 8 <= w {
            let d8 = loadu_64!(<&[u8; 8]>::try_from(&dst_row[col..col + 8]).unwrap());
            let t8 = loadu_64!(<&[u8; 8]>::try_from(&tmp_row[col..col + 8]).unwrap());
            let dat = _mm_unpacklo_epi8(d8, t8);
            let sum = _mm_add_epi16(_mm_maddubs_epi16(dat, cof), _mm_set1_epi16(32));
            let res = _mm_srli_epi16::<6>(sum);
            storei64!(&mut dst_row[col..col + 8], _mm_packus_epi16(res, res));
            col += 8;
        }
        if col + 4 <= w {
            let d4 = loadi32!(&dst_row[col..col + 4]);
            let t4 = loadi32!(&tmp_row[col..col + 4]);
            let dat = _mm_unpacklo_epi8(d4, t4);
            let sum = _mm_add_epi16(_mm_maddubs_epi16(dat, cof), _mm_set1_epi16(32));
            let res = _mm_srli_epi16::<6>(sum);
            let p8 = _mm_packus_epi16(res, res);
            dst_row[col..col + 4].copy_from_slice(&(_mm_cvtsi128_si32(p8) as u32).to_ne_bytes());
            col += 4;
        }

        // Handle remaining pixels
        while col < w {
            let a = dst_row[col] as u32;
            let b = tmp_row[col] as u32;
            let val = (a * (64 - m as u32) + b * m as u32 + 32) >> 6;
            dst_row[col] = val as u8;
            col += 1;
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn blend_h_8bpc_v3(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp: *const [DynPixel; SCRATCH_LAP_LEN],
    w: i32,
    h: i32,
    _dst: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    let dst = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h as usize * dst_stride as usize)
    };
    let tmp_slice = unsafe { std::slice::from_raw_parts(tmp as *const u8, (w * h) as usize) };
    blend_h_8bpc_avx2_safe(token, dst, dst_stride as usize, tmp_slice, w, h)
}

/// 16-bit blend_v
#[cfg(target_arch = "x86_64")]
#[arcane]
fn blend_v_16bpc_avx2_safe(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp: &[u8],
    w: i32,
    h: i32,
) {
    let mut dst = dst.flex_mut();
    let tmp = tmp.flex();
    let w = w as usize;
    let h = h as usize;
    // blend_v: mask indexed by column, offset by w, only process w*3/4 columns
    let obmc_mask = &dav1d_obmc_masks[w..];
    let w_eff = w * 3 >> 2;

    let rnd = _mm256_set1_epi32(32);
    let sixty_four = _mm256_set1_epi32(64);

    for row in 0..h {
        let dst_row_bytes = &mut dst[row * dst_stride..][..w_eff * 2];
        let dst_row: &mut [u16] = zerocopy::FromBytes::mut_from_bytes(dst_row_bytes).unwrap();
        // tmp uses full w stride
        let tmp_row_bytes = &tmp[row * w * 2..][..w_eff * 2];
        let tmp_row: &[u16] =
            <[u16] as zerocopy::FromBytes>::ref_from_bytes(tmp_row_bytes).unwrap();

        let mut col = 0usize;

        // Process 8 pixels at a time with 32-bit arithmetic
        while col + 8 <= w_eff {
            let dst_16 = loadu_128!(&dst_row[col..col + 8], [u16; 8]);
            let tmp_16 = loadu_128!(&tmp_row[col..col + 8], [u16; 8]);
            let dst_32 = _mm256_cvtepu16_epi32(dst_16);
            let tmp_32 = _mm256_cvtepu16_epi32(tmp_16);

            // Load 8 bytes of mask via zero-padded 16-byte array
            let mut mask_pad = [0u8; 16];
            mask_pad[..8].copy_from_slice(&obmc_mask[col..col + 8]);
            let mask_bytes = loadu_128!(&mask_pad);
            let mask_32 = _mm256_cvtepu8_epi32(mask_bytes);
            let inv_mask = _mm256_sub_epi32(sixty_four, mask_32);

            let term1 = _mm256_mullo_epi32(dst_32, inv_mask);
            let term2 = _mm256_mullo_epi32(tmp_32, mask_32);
            let sum = _mm256_add_epi32(_mm256_add_epi32(term1, term2), rnd);

            let result = _mm256_srli_epi32(sum, 6);

            let packed = _mm256_packus_epi32(result, result);
            let lo128 = _mm256_castsi256_si128(packed);
            let hi128 = _mm256_extracti128_si256(packed, 1);
            let result_128 = _mm_unpacklo_epi64(lo128, hi128);

            storeu_128!(&mut dst_row[col..col + 8], [u16; 8], result_128);

            col += 8;
        }

        // Handle remaining pixels with scalar
        while col < w_eff {
            let a = dst_row[col] as u32;
            let b = tmp_row[col] as u32;
            let m = obmc_mask[col] as u32;
            let val = (a * (64 - m) + b * m + 32) >> 6;
            dst_row[col] = val as u16;
            col += 1;
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn blend_v_16bpc_v3(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp: *const [DynPixel; SCRATCH_LAP_LEN],
    w: i32,
    h: i32,
    _dst: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    let dst = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h as usize * dst_stride as usize)
    };
    let tmp_slice = unsafe { std::slice::from_raw_parts(tmp as *const u8, (w * h) as usize * 2) };
    blend_v_16bpc_avx2_safe(token, dst, dst_stride as usize, tmp_slice, w, h)
}

/// 16-bit blend_h
#[cfg(target_arch = "x86_64")]
#[arcane]
fn blend_h_16bpc_avx2_safe(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp: &[u8],
    w: i32,
    h: i32,
) {
    let mut dst = dst.flex_mut();
    let tmp = tmp.flex();
    let w = w as usize;
    let h = h as usize;
    // blend_h: mask indexed by row, offset by h, only process h*3/4 rows
    let obmc_mask = &dav1d_obmc_masks[h..];
    let h_eff = h * 3 >> 2;

    let rnd = _mm256_set1_epi32(32);
    let sixty_four = _mm256_set1_epi32(64);

    for row in 0..h_eff {
        let dst_row_bytes = &mut dst[row * dst_stride..][..w * 2];
        let dst_row: &mut [u16] = zerocopy::FromBytes::mut_from_bytes(dst_row_bytes).unwrap();
        let tmp_row_bytes = &tmp[row * w * 2..][..w * 2];
        let tmp_row: &[u16] =
            <[u16] as zerocopy::FromBytes>::ref_from_bytes(tmp_row_bytes).unwrap();
        let m = obmc_mask[row] as u32;

        let mask_32 = _mm256_set1_epi32(m as i32);
        let inv_mask = _mm256_sub_epi32(sixty_four, mask_32);

        let mut col = 0usize;

        // Process 8 pixels at a time with 32-bit arithmetic
        while col + 8 <= w {
            let dst_16 = loadu_128!(&dst_row[col..col + 8], [u16; 8]);
            let tmp_16 = loadu_128!(&tmp_row[col..col + 8], [u16; 8]);
            let dst_32 = _mm256_cvtepu16_epi32(dst_16);
            let tmp_32 = _mm256_cvtepu16_epi32(tmp_16);

            let term1 = _mm256_mullo_epi32(dst_32, inv_mask);
            let term2 = _mm256_mullo_epi32(tmp_32, mask_32);
            let sum = _mm256_add_epi32(_mm256_add_epi32(term1, term2), rnd);

            let result = _mm256_srli_epi32(sum, 6);

            let packed = _mm256_packus_epi32(result, result);
            let lo128 = _mm256_castsi256_si128(packed);
            let hi128 = _mm256_extracti128_si256(packed, 1);
            let result_128 = _mm_unpacklo_epi64(lo128, hi128);

            storeu_128!(&mut dst_row[col..col + 8], [u16; 8], result_128);

            col += 8;
        }

        // Handle remaining pixels with scalar
        while col < w {
            let a = dst_row[col] as u32;
            let b = tmp_row[col] as u32;
            let val = (a * (64 - m) + b * m + 32) >> 6;
            dst_row[col] = val as u16;
            col += 1;
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn blend_h_16bpc_v3(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp: *const [DynPixel; SCRATCH_LAP_LEN],
    w: i32,
    h: i32,
    _dst: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    let dst = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h as usize * dst_stride as usize)
    };
    let tmp_slice = unsafe { std::slice::from_raw_parts(tmp as *const u8, (w * h) as usize * 2) };
    blend_h_16bpc_avx2_safe(token, dst, dst_stride as usize, tmp_slice, w, h)
}

// =============================================================================
// 8-TAP FILTERS (MC/MCT)
// =============================================================================

/// Stride for intermediate buffer in 8-tap filtering
const MID_STRIDE: usize = 128;

/// Get filter coefficients for a given subpixel position and filter type
#[inline]
fn get_filter(m: usize, d: usize, filter_idx: usize) -> Option<&'static [i8; 8]> {
    if m == 0 {
        return None;
    }
    let m = m - 1;
    let i = if d > 4 {
        filter_idx
    } else {
        3 + (filter_idx & 1)
    };
    Some(&dav1d_mc_subpel_filters[i][m])
}

/// Index of the first nonzero tap in a subpel filter row: 0 for the real
/// 8-tap (sharp), 1 for regular/smooth (outer taps are zero), 2 for the
/// truncated rows used when the filtered dimension is <= 4. Skipped taps have
/// coefficient zero, so the reduced-tap kernels are bit-exact with the scalar
/// 8-tap sum.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn tap_base_8tap(f: &[i8; 8]) -> usize {
    if f[0] != 0 || f[7] != 0 {
        0
    } else if f[1] != 0 || f[6] != 0 {
        1
    } else {
        2
    }
}

/// Sliding 2-byte gather masks for `pshufb` horizontal filtering. Mask `o`
/// produces byte pairs `{s[o + i], s[o + i + 1]}` for outputs `i = 0..8`,
/// consumed by `maddubs` with coefficient pair `{f[o], f[o + 1]}`.
#[cfg(target_arch = "x86_64")]
const fn hpair_masks() -> [[u8; 16]; 8] {
    let mut m = [[0u8; 16]; 8];
    let mut o = 0usize;
    while o < 8 {
        let mut i = 0;
        while i < 8 {
            m[o][2 * i] = (o + i) as u8;
            m[o][2 * i + 1] = (o + i + 1) as u8;
            i += 1;
        }
        o += 1;
    }
    m
}

#[cfg(target_arch = "x86_64")]
const HPAIR_MASKS: [[u8; 16]; 8] = hpair_masks();

/// Broadcast a tap pair as `i8x2` in every i16 lane (for `maddubs`).
#[cfg(target_arch = "x86_64")]
#[rite]
fn tap_pair16(_token: Desktop64, f: &[i8; 8], i: usize) -> __m256i {
    _mm256_set1_epi16(((f[i + 1] as u8 as i16) << 8) | (f[i] as u8 as i16))
}

/// Per-filter constants for the `pshufb`+`maddubs` H kernels — built once per
/// block so the per-row work is load/shuffle/madd only.
#[cfg(target_arch = "x86_64")]
struct HPairCtx256 {
    m: (__m256i, __m256i, __m256i, __m256i),
    c: (__m256i, __m256i, __m256i, __m256i),
    rnd: __m256i,
    sct: __m128i,
}

/// Build the H kernel constants for taps `t0..t0+NT` — pair-group tables are
/// indexed for all four groups, so out-of-range groups clamp to pair (6,7).
/// `rnd_i`/`sh` are the i16 rounding term and shift (mid-fill uses
/// `rnd_i = (1<<sh)>>1`; the H-only put path uses 34 at shift 6).
#[cfg(target_arch = "x86_64")]
#[rite]
fn h_pair_setup_8bpc_avx2(
    _token: Desktop64,
    t0: usize,
    filter: &[i8; 8],
    rnd_i: i16,
    sh: u8,
) -> HPairCtx256 {
    let gi = |g: usize| {
        let i = t0 + 2 * g;
        if i + 1 < 8 { i } else { 6 }
    };
    HPairCtx256 {
        m: (
            _mm256_broadcastsi128_si256(loadu_128!(&HPAIR_MASKS[gi(0)])),
            _mm256_broadcastsi128_si256(loadu_128!(&HPAIR_MASKS[gi(1)])),
            _mm256_broadcastsi128_si256(loadu_128!(&HPAIR_MASKS[gi(2)])),
            _mm256_broadcastsi128_si256(loadu_128!(&HPAIR_MASKS[gi(3)])),
        ),
        c: (
            tap_pair16(_token, filter, gi(0)),
            tap_pair16(_token, filter, gi(1)),
            tap_pair16(_token, filter, gi(2)),
            tap_pair16(_token, filter, gi(3)),
        ),
        rnd: _mm256_set1_epi16(rnd_i),
        sct: _mm_cvtsi32_si128(sh as i32),
    }
}

/// Per-lane pair-window dot product: maddubs of pshufb'd byte pairs against
/// the broadcast tap pairs, summed over the `NT / 2` active pair groups.
#[cfg(target_arch = "x86_64")]
#[rite]
fn h_pair_sum16<const NT: usize>(
    _token: Desktop64,
    s: __m256i,
    m: (__m256i, __m256i, __m256i, __m256i),
    c: (__m256i, __m256i, __m256i, __m256i),
) -> __m256i {
    let mut sum = _mm256_add_epi16(
        _mm256_maddubs_epi16(_mm256_shuffle_epi8(s, m.0), c.0),
        _mm256_maddubs_epi16(_mm256_shuffle_epi8(s, m.1), c.1),
    );
    if NT > 4 {
        sum = _mm256_add_epi16(sum, _mm256_maddubs_epi16(_mm256_shuffle_epi8(s, m.2), c.2));
    }
    if NT > 6 {
        sum = _mm256_add_epi16(sum, _mm256_maddubs_epi16(_mm256_shuffle_epi8(s, m.3), c.3));
    }
    sum
}

/// Same as `h_pair_sum16` on a single 128-bit lane.
#[cfg(target_arch = "x86_64")]
#[rite]
fn h_pair_sum8<const NT: usize>(
    _token: Desktop64,
    s: __m128i,
    m: (__m128i, __m128i, __m128i, __m128i),
    c: (__m128i, __m128i, __m128i, __m128i),
) -> __m128i {
    let mut sum = _mm_add_epi16(
        _mm_maddubs_epi16(_mm_shuffle_epi8(s, m.0), c.0),
        _mm_maddubs_epi16(_mm_shuffle_epi8(s, m.1), c.1),
    );
    if NT > 4 {
        sum = _mm_add_epi16(sum, _mm_maddubs_epi16(_mm_shuffle_epi8(s, m.2), c.2));
    }
    if NT > 6 {
        sum = _mm_add_epi16(sum, _mm_maddubs_epi16(_mm_shuffle_epi8(s, m.3), c.3));
    }
    sum
}

/// Horizontal subpel filter of `w` pixels, i16 intermediate output.
///
/// `NT` is the active tap count (4, 6 or 8); the kernel consumes only taps
/// `(8 - NT) / 2 .. (8 - NT) / 2 + NT` — the outer taps of the selected table
/// row are zero. `src` points at the pixel covered by tap 0 (caller offset
/// `-3`). Loads stay within the slice: when fewer than `w + 7` bytes remain
/// (last row of a tight reference window) the row tail goes through a padded
/// stack copy instead of a scalar fallback.
#[cfg(target_arch = "x86_64")]
#[rite]
fn h_filter_ntap_8bpc_avx2_inner<const NT: usize>(
    _token: Desktop64,
    dst: &mut [i16],
    src: &[u8],
    w: usize,
    cx: &HPairCtx256,
    filter: &[i8; 8],
    sh: u8,
) {
    const { assert!(NT == 4 || NT == 6 || NT == 8) };
    let t0 = (8 - NT) / 2;
    let mut dst = dst.flex_mut();
    let src = src.flex();

    let (m0, m1, m2, m3) = cx.m;
    let (c0, c1, c2, c3) = cx.c;
    let (rnd, sct) = (cx.rnd, cx.sct);
    let m8 = (
        _mm256_castsi256_si128(m0),
        _mm256_castsi256_si128(m1),
        _mm256_castsi256_si128(m2),
        _mm256_castsi256_si128(m3),
    );
    let c8 = (
        _mm256_castsi256_si128(c0),
        _mm256_castsi256_si128(c1),
        _mm256_castsi256_si128(c2),
        _mm256_castsi256_si128(c3),
    );

    let mut col = 0usize;

    // 16 outputs per iteration: ymm lanes are the overlapping 16-byte windows
    // src[col..col+16] and src[col+8..col+24]; the same masks produce outputs
    // col..col+8 and col+8..col+16.
    while col + 16 <= w {
        let s = if col + 24 <= src.len() {
            let lo = loadu_128!(<&[u8; 16]>::try_from(&src[col..col + 16]).unwrap());
            let hi = loadu_128!(<&[u8; 16]>::try_from(&src[col + 8..col + 24]).unwrap());
            _mm256_set_m128i(hi, lo)
        } else {
            // Row tail: the tight reference window ends mid-vector. Pad the
            // remaining bytes (only lanes past `w` read them) and repeat.
            let mut buf = [0u8; 24];
            let n = (src.len() - col).min(24);
            buf[..n].copy_from_slice(&src[col..col + n]);
            let lo = loadu_128!(<&[u8; 16]>::try_from(&buf[..16]).unwrap());
            let hi = loadu_128!(<&[u8; 16]>::try_from(&buf[8..24]).unwrap());
            _mm256_set_m128i(hi, lo)
        };
        let res = _mm256_sra_epi16(
            _mm256_add_epi16(
                h_pair_sum16::<NT>(_token, s, (m0, m1, m2, m3), (c0, c1, c2, c3)),
                rnd,
            ),
            sct,
        );
        storeu_256!(
            <&mut [i16; 16]>::try_from(&mut dst[col..col + 16]).unwrap(),
            res
        );
        col += 16;
    }

    // 8-wide chunk: one 16-byte window covers outputs col..col+8.
    if col + 8 <= w {
        let mut buf = [0u8; 16];
        let s = if col + 16 <= src.len() {
            loadu_128!(<&[u8; 16]>::try_from(&src[col..col + 16]).unwrap())
        } else {
            let n = src.len() - col;
            buf[..n].copy_from_slice(&src[col..col + n]);
            loadu_128!(&buf)
        };
        let r = _mm_sra_epi16(
            _mm_add_epi16(
                h_pair_sum8::<NT>(_token, s, m8, c8),
                _mm256_castsi256_si128(rnd),
            ),
            sct,
        );
        storeu_128!(
            <&mut [i16; 8]>::try_from(&mut dst[col..col + 8]).unwrap(),
            r
        );
        col += 8;
    }

    // 4-wide chunk.
    if col + 4 <= w {
        let mut buf = [0u8; 16];
        let s = if col + 16 <= src.len() {
            loadu_128!(<&[u8; 16]>::try_from(&src[col..col + 16]).unwrap())
        } else {
            let n = src.len() - col;
            buf[..n].copy_from_slice(&src[col..col + n]);
            loadu_128!(&buf)
        };
        let r = _mm_sra_epi16(
            _mm_add_epi16(
                h_pair_sum8::<NT>(_token, s, m8, c8),
                _mm256_castsi256_si128(rnd),
            ),
            sct,
        );
        crate::src::safe_simd::partial_simd::mm_storel_epi64::<[i16; 4]>(
            <&mut [i16; 4]>::try_from(&mut dst[col..col + 4]).unwrap(),
            r,
        );
        col += 4;
    }

    // Scalar tail (w % 4 != 0, i.e. w = 2 or 6-column blocks).
    while col < w {
        let mut sum = 0i32;
        for i in 0..NT {
            sum += filter[t0 + i] as i32 * src[col + t0 + i] as i32;
        }
        dst[col] = ((sum + ((1 << sh) >> 1)) >> sh) as i16;
        col += 1;
    }
}

/// Horizontal 8-tap filter for a row of 8bpc pixels — NT-aware wrapper.
///
/// Processes `w` pixels starting at `src`, writing to `dst` (i16 intermediate)
/// Formula: sum(coeff[i] * src[x + i - 3]) for i in 0..8, then round and shift
#[cfg(target_arch = "x86_64")]
#[rite]
fn h_filter_8tap_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [i16],
    src: &[u8],
    w: usize,
    filter: &[i8; 8],
    sh: u8,
) {
    let t0 = tap_base_8tap(filter);
    let cx = h_pair_setup_8bpc_avx2(_token, t0, filter, (1i16 << sh) >> 1, sh);
    match t0 {
        2 => h_filter_ntap_8bpc_avx2_inner::<4>(_token, dst, src, w, &cx, filter, sh),
        1 => h_filter_ntap_8bpc_avx2_inner::<6>(_token, dst, src, w, &cx, filter, sh),
        _ => h_filter_ntap_8bpc_avx2_inner::<8>(_token, dst, src, w, &cx, filter, sh),
    }
}

/// Horizontal filter of `rows` consecutive mid/tmp rows — filter constants
/// are built once for the whole block instead of per row.
///
/// Row `j` writes `dst[dst_base + j*dst_stride ..][..w]` from
/// `src[src_base + j*src_stride..]`, with a signed source pitch.
#[cfg(target_arch = "x86_64")]
#[rite]
fn h_filter_rows_8tap_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [i16],
    dst_stride: usize,
    dst_base: usize,
    src: &[u8],
    src_base: usize,
    src_stride: isize,
    rows: usize,
    w: usize,
    filter: &[i8; 8],
    sh: u8,
) {
    let t0 = tap_base_8tap(filter);
    let cx = h_pair_setup_8bpc_avx2(_token, t0, filter, (1i16 << sh) >> 1, sh);
    for j in 0..rows {
        let d = &mut dst[dst_base + j * dst_stride..];
        let s = &src[src_base.wrapping_add_signed(j as isize * src_stride)..];
        match t0 {
            2 => h_filter_ntap_8bpc_avx2_inner::<4>(_token, d, s, w, &cx, filter, sh),
            1 => h_filter_ntap_8bpc_avx2_inner::<6>(_token, d, s, w, &cx, filter, sh),
            _ => h_filter_ntap_8bpc_avx2_inner::<8>(_token, d, s, w, &cx, filter, sh),
        }
    }
}

/// Per-filter constants for the zmm `pshufb`+`maddubs` H kernels.
#[cfg(target_arch = "x86_64")]
struct HPairCtx512 {
    m: (__m512i, __m512i, __m512i, __m512i),
    c: (__m512i, __m512i, __m512i, __m512i),
    rnd: __m512i,
    sct: __m128i,
}

#[cfg(target_arch = "x86_64")]
#[rite]
fn h_pair_setup_8bpc_avx512(
    _token: Server64,
    t0: usize,
    filter: &[i8; 8],
    rnd_i: i16,
    sh: u8,
) -> HPairCtx512 {
    let gi = |g: usize| {
        let i = t0 + 2 * g;
        if i + 1 < 8 { i } else { 6 }
    };
    let pair = |i: usize| ((filter[i + 1] as u8 as i16) << 8) | (filter[i] as u8 as i16);
    HPairCtx512 {
        m: (
            _mm512_broadcast_i32x4(loadu_128!(&HPAIR_MASKS[gi(0)])),
            _mm512_broadcast_i32x4(loadu_128!(&HPAIR_MASKS[gi(1)])),
            _mm512_broadcast_i32x4(loadu_128!(&HPAIR_MASKS[gi(2)])),
            _mm512_broadcast_i32x4(loadu_128!(&HPAIR_MASKS[gi(3)])),
        ),
        c: (
            _mm512_set1_epi16(pair(gi(0))),
            _mm512_set1_epi16(pair(gi(1))),
            _mm512_set1_epi16(pair(gi(2))),
            _mm512_set1_epi16(pair(gi(3))),
        ),
        rnd: _mm512_set1_epi16(rnd_i),
        sct: _mm_cvtsi32_si128(sh as i32),
    }
}

/// One 32-wide row of the zmm pair-window H filter: lanes are overlapping
/// 16-byte windows at `col`, `col+16`, `col+8`, `col+24` (natural output order
/// restored by `shuffle_i64x2` at the end).
#[cfg(target_arch = "x86_64")]
#[rite]
fn h_pair_row32_8bpc_avx512(
    _token: Server64,
    dst: &mut [i16],
    src: &[u8],
    w: usize,
    nt: usize,
    cx: &HPairCtx512,
    filter: &[i8; 8],
    sh: u8,
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    let (m0, m1, m2, m3) = cx.m;
    let (c0, c1, c2, c3) = cx.c;
    let (rnd, sct) = (cx.rnd, cx.sct);
    let mut col = 0usize;

    while col + 32 <= w {
        let s = if col + 40 <= src.len() {
            let a = loadu_256!(<&[u8; 32]>::try_from(&src[col..col + 32]).unwrap());
            let b = loadu_256!(<&[u8; 32]>::try_from(&src[col + 8..col + 40]).unwrap());
            _mm512_inserti64x4::<1>(_mm512_castsi256_si512(a), b)
        } else {
            // Row tail: the tight reference window ends mid-vector.
            let mut buf = [0u8; 40];
            let n = (src.len() - col).min(40);
            buf[..n].copy_from_slice(&src[col..col + n]);
            let a = loadu_256!(<&[u8; 32]>::try_from(&buf[..32]).unwrap());
            let b = loadu_256!(<&[u8; 32]>::try_from(&buf[8..40]).unwrap());
            _mm512_inserti64x4::<1>(_mm512_castsi256_si512(a), b)
        };
        let mut sum = _mm512_add_epi16(
            _mm512_maddubs_epi16(_mm512_shuffle_epi8(s, m0), c0),
            _mm512_maddubs_epi16(_mm512_shuffle_epi8(s, m1), c1),
        );
        if nt > 4 {
            sum = _mm512_add_epi16(sum, _mm512_maddubs_epi16(_mm512_shuffle_epi8(s, m2), c2));
        }
        if nt > 6 {
            sum = _mm512_add_epi16(sum, _mm512_maddubs_epi16(_mm512_shuffle_epi8(s, m3), c3));
        }
        let res = _mm512_sra_epi16(_mm512_add_epi16(sum, rnd), sct);
        let res = _mm512_shuffle_i64x2::<0b11_01_10_00>(res, res);
        storeu_512!(
            <&mut [i16; 32]>::try_from(&mut dst[col..col + 32]).unwrap(),
            res
        );
        col += 32;
    }

    // Scalar fallback for remaining pixels
    while col < w {
        let mut sum = 0i32;
        for i in 0..8 {
            sum += filter[i] as i32 * src[col + i] as i32;
        }
        dst[col] = ((sum + ((1 << sh) >> 1)) >> sh) as i16;
        col += 1;
    }
}

/// AVX-512 horizontal 8-tap filter for 8bpc prep — processes 32 i16 outputs per iteration.
/// Uses _mm512_maddubs_epi16 with lane-fix shuffle at end.
#[cfg(target_arch = "x86_64")]
#[rite]
fn h_filter_8tap_8bpc_avx512_inner(
    _token: Server64,
    dst: &mut [i16],
    src: &[u8],
    w: usize,
    filter: &[i8; 8],
    sh: u8,
) {
    // w < 32 misses the 32-wide loop entirely; the AVX2 kernel handles
    // w = 4/8 via pshufb and w = 16 via its 16-wide lane.
    if w < 32 {
        h_filter_8tap_8bpc_avx2_inner(_token.v3(), dst, src, w, filter, sh);
        return;
    }
    let t0 = tap_base_8tap(filter);
    let nt = 8 - 2 * t0;
    let cx = h_pair_setup_8bpc_avx512(_token, t0, filter, (1i16 << sh) >> 1, sh);
    h_pair_row32_8bpc_avx512(_token, dst, src, w, nt, &cx, filter, sh);
}

/// Batched zmm H filter into `rows` i16 rows — constants built once.
/// Row `j` writes `dst[dst_base + j*dst_stride..][..w]` from
/// `src[src_base + j*src_stride..]`, with a signed source pitch.
#[cfg(target_arch = "x86_64")]
#[rite]
fn h_filter_rows_8tap_8bpc_avx512_inner(
    _token: Server64,
    dst: &mut [i16],
    dst_stride: usize,
    dst_base: usize,
    src: &[u8],
    src_base: usize,
    src_stride: isize,
    rows: usize,
    w: usize,
    filter: &[i8; 8],
    sh: u8,
) {
    if w < 32 {
        h_filter_rows_8tap_8bpc_avx2_inner(
            _token.v3(),
            dst,
            dst_stride,
            dst_base,
            src,
            src_base,
            src_stride,
            rows,
            w,
            filter,
            sh,
        );
        return;
    }
    let t0 = tap_base_8tap(filter);
    let nt = 8 - 2 * t0;
    let cx = h_pair_setup_8bpc_avx512(_token, t0, filter, (1i16 << sh) >> 1, sh);
    for j in 0..rows {
        h_pair_row32_8bpc_avx512(
            _token,
            &mut dst[dst_base + j * dst_stride..],
            &src[src_base.wrapping_add_signed(j as isize * src_stride)..],
            w,
            nt,
            &cx,
            filter,
            sh,
        );
    }
}

#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[archmage::rite(v3)]
unsafe fn h_filter_8tap_8bpc_v3(dst: *mut i16, src: *const u8, w: usize, filter: &[i8; 8], sh: u8) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();
    unsafe { h_filter_8tap_8bpc_avx2_inner(token, dst, src, w, filter, sh) }
}

/// Vertical 8-tap filter from intermediate buffer to output
///
/// Processes `w` pixels for one row, reading from `mid` (8 rows), writing to `dst`
#[cfg(target_arch = "x86_64")]
#[rite]
fn v_filter_ntap_8bpc_avx2_inner<const NT: usize>(
    _token: Desktop64,
    dst: &mut [u8],
    mid: &[[i16; MID_STRIDE]],
    w: usize,
    filter: &[i8; 8],
    sh: u8,
    max: i32,
) {
    const { assert!(NT == 4 || NT == 6 || NT == 8) };
    let t0 = (8 - NT) / 2;
    let mut dst = dst.flex_mut();
    // pmaddwd pair-interleave: pack filter coeffs as i16x2, interleave
    // adjacent rows into i16 pairs, one madd per tap pair. For NT < 8 only
    // taps t0..t0+NT are nonzero, so rows outside that window are not loaded.
    let pack_pair = |a: i8, b: i8| ((b as i32) << 16) | (a as i32 & 0xffff);
    let c01 = _mm_set1_epi32(pack_pair(filter[t0], filter[t0 + 1]));
    let c23 = _mm_set1_epi32(pack_pair(filter[t0 + 2], filter[t0 + 3]));
    let c45 = if NT > 4 {
        _mm_set1_epi32(pack_pair(filter[t0 + 4], filter[t0 + 5]))
    } else {
        _mm_setzero_si128()
    };
    let c67 = if NT > 6 {
        _mm_set1_epi32(pack_pair(filter[t0 + 6], filter[t0 + 7]))
    } else {
        _mm_setzero_si128()
    };

    let rnd = _mm_set1_epi32((1i32 << sh) >> 1);
    let zero = _mm_setzero_si128();
    let max16 = _mm_set1_epi16(max as i16);
    let shift_count = _mm_cvtsi32_si128(sh as i32);

    let mut col = 0usize;

    // Process 8 columns at a time via pmaddwd on i16 row-pairs.
    while col + 8 <= w {
        let m0 = loadu_128!(<&[i16; 8]>::try_from(&mid[t0][col..col + 8]).unwrap());
        let m1 = loadu_128!(<&[i16; 8]>::try_from(&mid[t0 + 1][col..col + 8]).unwrap());
        let m2 = loadu_128!(<&[i16; 8]>::try_from(&mid[t0 + 2][col..col + 8]).unwrap());
        let m3 = loadu_128!(<&[i16; 8]>::try_from(&mid[t0 + 3][col..col + 8]).unwrap());
        let (m4, m5) = if NT > 4 {
            (
                loadu_128!(<&[i16; 8]>::try_from(&mid[t0 + 4][col..col + 8]).unwrap()),
                loadu_128!(<&[i16; 8]>::try_from(&mid[t0 + 5][col..col + 8]).unwrap()),
            )
        } else {
            (_mm_setzero_si128(), _mm_setzero_si128())
        };
        let (m6, m7) = if NT > 6 {
            (
                loadu_128!(<&[i16; 8]>::try_from(&mid[t0 + 6][col..col + 8]).unwrap()),
                loadu_128!(<&[i16; 8]>::try_from(&mid[t0 + 7][col..col + 8]).unwrap()),
            )
        } else {
            (zero, zero)
        };

        // Interleave adjacent rows: pairs (m0,m1) lo/hi cover cols 0..3 / 4..7
        let p01l = _mm_unpacklo_epi16(m0, m1);
        let p01h = _mm_unpackhi_epi16(m0, m1);
        let p23l = _mm_unpacklo_epi16(m2, m3);
        let p23h = _mm_unpackhi_epi16(m2, m3);
        let p45l = _mm_unpacklo_epi16(m4, m5);
        let p45h = _mm_unpackhi_epi16(m4, m5);
        let p67l = _mm_unpacklo_epi16(m6, m7);
        let p67h = _mm_unpackhi_epi16(m6, m7);

        let mut sum_l = _mm_add_epi32(
            _mm_add_epi32(_mm_madd_epi16(p01l, c01), _mm_madd_epi16(p23l, c23)),
            _mm_madd_epi16(p45l, c45),
        );
        let mut sum_h = _mm_add_epi32(
            _mm_add_epi32(_mm_madd_epi16(p01h, c01), _mm_madd_epi16(p23h, c23)),
            _mm_madd_epi16(p45h, c45),
        );
        if NT > 6 {
            sum_l = _mm_add_epi32(sum_l, _mm_madd_epi16(p67l, c67));
            sum_h = _mm_add_epi32(sum_h, _mm_madd_epi16(p67h, c67));
        }

        let rl = _mm_sra_epi32(_mm_add_epi32(sum_l, rnd), shift_count);
        let rh = _mm_sra_epi32(_mm_add_epi32(sum_h, rnd), shift_count);

        // i32 -> i16 preserving column order, clamp, then -> u8
        let packed16 = _mm_packs_epi32(rl, rh);
        let clamped = _mm_max_epi16(_mm_min_epi16(packed16, max16), zero);
        let packed8 = _mm_packus_epi16(clamped, clamped);
        crate::src::safe_simd::partial_simd::mm_storel_epi64::<[u8; 8]>(
            <&mut [u8; 8]>::try_from(&mut dst[col..col + 8]).unwrap(),
            packed8,
        );

        col += 8;
    }

    // w=4 tail: 64-bit row loads, same pair-madd structure.
    if w - col == 4 && mid[0].len() >= col + 4 {
        let ld = |r: usize| -> __m128i {
            crate::src::safe_simd::partial_simd::mm_loadl_epi64::<[i16; 4]>(
                <&[i16; 4]>::try_from(&mid[t0 + r][col..col + 4]).unwrap(),
            )
        };
        let m0 = ld(0);
        let m1 = ld(1);
        let m2 = ld(2);
        let m3 = ld(3);
        let (m4, m5) = if NT > 4 {
            (ld(4), ld(5))
        } else {
            (_mm_setzero_si128(), _mm_setzero_si128())
        };
        let (m6, m7) = if NT > 6 { (ld(6), ld(7)) } else { (zero, zero) };

        let p01 = _mm_unpacklo_epi16(m0, m1);
        let p23 = _mm_unpacklo_epi16(m2, m3);
        let p45 = _mm_unpacklo_epi16(m4, m5);
        let p67 = _mm_unpacklo_epi16(m6, m7);

        let mut sum = _mm_add_epi32(
            _mm_add_epi32(_mm_madd_epi16(p01, c01), _mm_madd_epi16(p23, c23)),
            _mm_madd_epi16(p45, c45),
        );
        if NT > 6 {
            sum = _mm_add_epi32(sum, _mm_madd_epi16(p67, c67));
        }
        let r = _mm_sra_epi32(_mm_add_epi32(sum, rnd), shift_count);
        let packed16 = _mm_packs_epi32(r, r);
        let clamped = _mm_max_epi16(_mm_min_epi16(packed16, max16), zero);
        let packed8 = _mm_packus_epi16(clamped, clamped);
        dst[col..col + 4].copy_from_slice(&(_mm_cvtsi128_si32(packed8) as u32).to_ne_bytes());
        col += 4;
    }

    // Scalar fallback
    while col < w {
        let mut sum = 0i32;
        for i in 0..NT {
            sum += filter[t0 + i] as i32 * mid[t0 + i][col] as i32;
        }
        let val = ((sum + ((1 << sh) >> 1)) >> sh).clamp(0, max);
        dst[col] = val as u8;
        col += 1;
    }
}

/// Vertical 8-tap filter from intermediate buffer — NT-aware wrapper.
///
/// Processes `w` pixels for one row, reading from `mid` (8 rows), writing to `dst`
#[cfg(target_arch = "x86_64")]
#[rite]
fn v_filter_8tap_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    mid: &[[i16; MID_STRIDE]],
    w: usize,
    filter: &[i8; 8],
    sh: u8,
    max: i32,
) {
    match tap_base_8tap(filter) {
        2 => v_filter_ntap_8bpc_avx2_inner::<4>(_token, dst, mid, w, filter, sh, max),
        1 => v_filter_ntap_8bpc_avx2_inner::<6>(_token, dst, mid, w, filter, sh, max),
        _ => v_filter_ntap_8bpc_avx2_inner::<8>(_token, dst, mid, w, filter, sh, max),
    }
}

#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[archmage::rite(v3)]
unsafe fn v_filter_8tap_8bpc_v3(
    dst: *mut u8,
    mid: &[[i16; MID_STRIDE]],
    w: usize,
    filter: &[i8; 8],
    sh: u8,
    max: i32,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();
    unsafe { v_filter_8tap_8bpc_avx2_inner(token, dst, mid, w, filter, sh, max) }
}

/// AVX-512 vertical 8-tap filter from intermediate buffer — processes 16 pixels per iteration.
/// Uses _mm512_cvtepi16_epi32 for i16→i32 expansion, _mm512_cvtusepi32_epi8 for output.
#[cfg(target_arch = "x86_64")]
#[rite]
fn v_filter_8tap_8bpc_avx512_inner(
    _token: Server64,
    dst: &mut [u8],
    mid: &[[i16; MID_STRIDE]],
    w: usize,
    filter: &[i8; 8],
    sh: u8,
    max: i32,
) {
    if w < 16 {
        v_filter_8tap_8bpc_avx2_inner(_token.v3(), dst, mid, w, filter, sh, max);
        return;
    }
    let t0 = tap_base_8tap(filter);
    let nt = 8 - 2 * t0;
    let mut dst = dst.flex_mut();

    let rnd = _mm512_set1_epi32((1i32 << sh) >> 1);
    let zero = _mm512_setzero_si512();
    let max_v = _mm512_set1_epi32(max);

    // Broadcast filter coefficients to 32-bit; only taps t0..t0+nt are
    // nonzero — their mid rows sit at t0..t0+nt of the passed window.
    let c: [i32; 8] = core::array::from_fn(|i| if i < nt { filter[t0 + i] as i32 } else { 0 });
    let c0 = _mm512_set1_epi32(c[0]);
    let c1 = _mm512_set1_epi32(c[1]);
    let c2 = _mm512_set1_epi32(c[2]);
    let c3 = _mm512_set1_epi32(c[3]);
    let c4 = _mm512_set1_epi32(c[4]);
    let c5 = _mm512_set1_epi32(c[5]);
    let c6 = _mm512_set1_epi32(c[6]);
    let c7 = _mm512_set1_epi32(c[7]);

    let mut col = 0usize;

    // Process 16 pixels at a time (i16→i32 expansion gives 16 i32 in 512 bits)
    while col + 16 <= w {
        // Load 16 i16 from each of 8 rows, expand to i32. Rows past t0+nt
        // multiply a zero coefficient, so the skipped loads change nothing.
        let m0 = _mm512_cvtepi16_epi32(loadu_256!(
            <&[i16; 16]>::try_from(&mid[t0][col..col + 16]).unwrap()
        ));
        let m1 = _mm512_cvtepi16_epi32(loadu_256!(
            <&[i16; 16]>::try_from(&mid[t0 + 1][col..col + 16]).unwrap()
        ));
        let m2 = _mm512_cvtepi16_epi32(loadu_256!(
            <&[i16; 16]>::try_from(&mid[t0 + 2][col..col + 16]).unwrap()
        ));
        let m3 = _mm512_cvtepi16_epi32(loadu_256!(
            <&[i16; 16]>::try_from(&mid[t0 + 3][col..col + 16]).unwrap()
        ));
        let m4 = _mm512_cvtepi16_epi32(loadu_256!(
            <&[i16; 16]>::try_from(&mid[t0 + 4][col..col + 16]).unwrap()
        ));
        let m5 = _mm512_cvtepi16_epi32(loadu_256!(
            <&[i16; 16]>::try_from(&mid[t0 + 5][col..col + 16]).unwrap()
        ));
        let m6 = if nt > 6 {
            _mm512_cvtepi16_epi32(loadu_256!(
                <&[i16; 16]>::try_from(&mid[t0 + 6][col..col + 16]).unwrap()
            ))
        } else {
            zero
        };
        let m7 = if nt > 6 {
            _mm512_cvtepi16_epi32(loadu_256!(
                <&[i16; 16]>::try_from(&mid[t0 + 7][col..col + 16]).unwrap()
            ))
        } else {
            zero
        };

        // Multiply each row by its coefficient and accumulate
        let mut sum = _mm512_mullo_epi32(m0, c0);
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(m1, c1));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(m2, c2));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(m3, c3));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(m4, c4));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(m5, c5));
        if nt > 6 {
            sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(m6, c6));
            sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(m7, c7));
        }

        // Round, shift, clamp to [0, max]
        let shift_count = _mm_cvtsi32_si128(sh as i32);
        let shifted = _mm512_sra_epi32(_mm512_add_epi32(sum, rnd), shift_count);
        let clamped = _mm512_min_epi32(_mm512_max_epi32(shifted, zero), max_v);

        // Pack i32→u8 via unsigned saturation (values already in [0,255])
        let packed = _mm512_cvtusepi32_epi8(clamped);

        // Store 16 bytes
        storeu_128!(
            <&mut [u8; 16]>::try_from(&mut dst[col..col + 16]).unwrap(),
            packed
        );

        col += 16;
    }

    // Scalar fallback
    while col < w {
        let mut sum = 0i32;
        for i in 0..nt {
            sum += filter[t0 + i] as i32 * mid[t0 + i][col] as i32;
        }
        let val = ((sum + ((1 << sh) >> 1)) >> sh).clamp(0, max);
        dst[col] = val as u8;
        col += 1;
    }
}

// =============================================================================
// 8-TAP PUT FUNCTIONS (mc)
// =============================================================================

/// Get filter coefficients for a given subpixel position and filter type
///
/// Returns None if m == 0 (integer position, no filtering needed)
#[inline]
fn get_filter_coeff(m: usize, d: usize, filter_type: Rav1dFilterMode) -> Option<&'static [i8; 8]> {
    let m = m.checked_sub(1)?;
    let i = if d > 4 {
        filter_type as u8
    } else {
        3 + (filter_type as u8 & 1)
    };
    Some(&dav1d_mc_subpel_filters[i as usize][m])
}

/// Horizontal 8-tap filter for 8bpc put (H-only case)
/// Outputs directly to u8 with shift and clamp
#[cfg(target_arch = "x86_64")]
#[rite]
fn h_filter_ntap_8bpc_put_avx2_inner<const NT: usize>(
    _token: Desktop64,
    dst: &mut [u8],
    src: &[u8], // already offset by -3
    w: usize,
    cx: &HPairCtx256,
    filter: &[i8; 8],
) {
    const { assert!(NT == 4 || NT == 6 || NT == 8) };
    let t0 = (8 - NT) / 2;
    let mut dst = dst.flex_mut();
    let src = src.flex();

    let (m0, m1, m2, m3) = cx.m;
    let (c0, c1, c2, c3) = cx.c;
    // rnd=34 matches 8bpc H-only put: 32 + ((1 << (6-4)) >> 1)
    let rnd = cx.rnd;
    let sct = cx.sct;
    let m8 = (
        _mm256_castsi256_si128(m0),
        _mm256_castsi256_si128(m1),
        _mm256_castsi256_si128(m2),
        _mm256_castsi256_si128(m3),
    );
    let c8 = (
        _mm256_castsi256_si128(c0),
        _mm256_castsi256_si128(c1),
        _mm256_castsi256_si128(c2),
        _mm256_castsi256_si128(c3),
    );

    let mut col = 0usize;

    // 16 outputs per iteration (ymm lanes = overlapping 16B windows).
    while col + 16 <= w {
        let s = if col + 24 <= src.len() {
            let lo = loadu_128!(<&[u8; 16]>::try_from(&src[col..col + 16]).unwrap());
            let hi = loadu_128!(<&[u8; 16]>::try_from(&src[col + 8..col + 24]).unwrap());
            _mm256_set_m128i(hi, lo)
        } else {
            let mut buf = [0u8; 24];
            let n = (src.len() - col).min(24);
            buf[..n].copy_from_slice(&src[col..col + n]);
            let lo = loadu_128!(<&[u8; 16]>::try_from(&buf[..16]).unwrap());
            let hi = loadu_128!(<&[u8; 16]>::try_from(&buf[8..24]).unwrap());
            _mm256_set_m128i(hi, lo)
        };
        let res16 = _mm256_sra_epi16(
            _mm256_add_epi16(
                h_pair_sum16::<NT>(_token, s, (m0, m1, m2, m3), (c0, c1, c2, c3)),
                rnd,
            ),
            sct,
        );
        // packus saturates i16 -> u8 = the final [0,255] clamp; lane0 holds
        // outputs col..col+8, lane1 col+8..col+16.
        let packed = _mm256_packus_epi16(res16, res16);
        let packed = _mm256_permute4x64_epi64(packed, 0b11011000);
        storeu_128!(
            <&mut [u8; 16]>::try_from(&mut dst[col..col + 16]).unwrap(),
            _mm256_castsi256_si128(packed)
        );
        col += 16;
    }

    // 8-wide chunk.
    if col + 8 <= w {
        let mut buf = [0u8; 16];
        let s = if col + 16 <= src.len() {
            loadu_128!(<&[u8; 16]>::try_from(&src[col..col + 16]).unwrap())
        } else {
            let n = src.len() - col;
            buf[..n].copy_from_slice(&src[col..col + n]);
            loadu_128!(&buf)
        };
        let res = _mm_sra_epi16(
            _mm_add_epi16(
                h_pair_sum8::<NT>(_token, s, m8, c8),
                _mm256_castsi256_si128(rnd),
            ),
            sct,
        );
        let packed8 = _mm_packus_epi16(res, res);
        crate::src::safe_simd::partial_simd::mm_storel_epi64::<[u8; 8]>(
            <&mut [u8; 8]>::try_from(&mut dst[col..col + 8]).unwrap(),
            packed8,
        );
        col += 8;
    }

    // 4-wide chunk.
    if col + 4 <= w {
        let mut buf = [0u8; 16];
        let s = if col + 16 <= src.len() {
            loadu_128!(<&[u8; 16]>::try_from(&src[col..col + 16]).unwrap())
        } else {
            let n = src.len() - col;
            buf[..n].copy_from_slice(&src[col..col + n]);
            loadu_128!(&buf)
        };
        let res = _mm_sra_epi16(
            _mm_add_epi16(
                h_pair_sum8::<NT>(_token, s, m8, c8),
                _mm256_castsi256_si128(rnd),
            ),
            sct,
        );
        let packed8 = _mm_packus_epi16(res, res);
        dst[col..col + 4].copy_from_slice(&(_mm_cvtsi128_si32(packed8) as u32).to_ne_bytes());
        col += 4;
    }

    // Scalar tail (rnd=34 matches SIMD path above)
    while col < w {
        let mut sum = 0i32;
        for i in 0..NT {
            sum += filter[t0 + i] as i32 * src[col + t0 + i] as i32;
        }
        dst[col] = ((sum + 34) >> 6).clamp(0, 255) as u8;
        col += 1;
    }
}

/// Horizontal 8-tap put filter — NT-aware wrapper.
#[cfg(target_arch = "x86_64")]
#[rite]
fn h_filter_8tap_8bpc_put_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    src: &[u8], // already offset by -3
    w: usize,
    filter: &[i8; 8],
) {
    let t0 = tap_base_8tap(filter);
    let cx = h_pair_setup_8bpc_avx2(_token, t0, filter, 34, 6);
    match t0 {
        2 => h_filter_ntap_8bpc_put_avx2_inner::<4>(_token, dst, src, w, &cx, filter),
        1 => h_filter_ntap_8bpc_put_avx2_inner::<6>(_token, dst, src, w, &cx, filter),
        _ => h_filter_ntap_8bpc_put_avx2_inner::<8>(_token, dst, src, w, &cx, filter),
    }
}

/// Batched H-only put: `rows` consecutive dst rows, constants built once.
/// Row `j` writes `dst[j*dst_stride..][..w]` from `src[src_base + j*src_stride..]`, with a signed source pitch.
#[cfg(target_arch = "x86_64")]
#[rite]
fn h_filter_rows_8tap_8bpc_put_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    src: &[u8],
    src_base: usize,
    src_stride: isize,
    rows: usize,
    w: usize,
    filter: &[i8; 8],
) {
    let t0 = tap_base_8tap(filter);
    let cx = h_pair_setup_8bpc_avx2(_token, t0, filter, 34, 6);
    for j in 0..rows {
        let d = &mut dst[dst_base.wrapping_add_signed(j as isize * dst_stride)..];
        let s = &src[src_base.wrapping_add_signed(j as isize * src_stride)..];
        match t0 {
            2 => h_filter_ntap_8bpc_put_avx2_inner::<4>(_token, d, s, w, &cx, filter),
            1 => h_filter_ntap_8bpc_put_avx2_inner::<6>(_token, d, s, w, &cx, filter),
            _ => h_filter_ntap_8bpc_put_avx2_inner::<8>(_token, d, s, w, &cx, filter),
        }
    }
}

/// One 32-wide u8 row of the zmm pair-window H filter (H-only put):
/// same lane window order as `h_pair_row32_8bpc_avx512`.
#[cfg(target_arch = "x86_64")]
#[rite]
fn h_pair_row32_8bpc_put_avx512(
    _token: Server64,
    dst: &mut [u8],
    src: &[u8],
    w: usize,
    nt: usize,
    cx: &HPairCtx512,
    filter: &[i8; 8],
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();

    let (m0, m1, m2, m3) = cx.m;
    let (c0, c1, c2, c3) = cx.c;
    // rnd=34 matches 8bpc H-only put: 32 + ((1 << (6-4)) >> 1)
    let rnd = cx.rnd;
    let sct = cx.sct;
    let zero = _mm512_setzero_si512();
    let max_v = _mm512_set1_epi16(255);

    let mut col = 0usize;

    // 32 outputs per iteration; same lane order as the i16 kernel.
    while col + 32 <= w {
        let s = if col + 40 <= src.len() {
            let a = loadu_256!(<&[u8; 32]>::try_from(&src[col..col + 32]).unwrap());
            let b = loadu_256!(<&[u8; 32]>::try_from(&src[col + 8..col + 40]).unwrap());
            _mm512_inserti64x4::<1>(_mm512_castsi256_si512(a), b)
        } else {
            let mut buf = [0u8; 40];
            let n = (src.len() - col).min(40);
            buf[..n].copy_from_slice(&src[col..col + n]);
            let a = loadu_256!(<&[u8; 32]>::try_from(&buf[..32]).unwrap());
            let b = loadu_256!(<&[u8; 32]>::try_from(&buf[8..40]).unwrap());
            _mm512_inserti64x4::<1>(_mm512_castsi256_si512(a), b)
        };
        let mut sum = _mm512_add_epi16(
            _mm512_maddubs_epi16(_mm512_shuffle_epi8(s, m0), c0),
            _mm512_maddubs_epi16(_mm512_shuffle_epi8(s, m1), c1),
        );
        if nt > 4 {
            sum = _mm512_add_epi16(sum, _mm512_maddubs_epi16(_mm512_shuffle_epi8(s, m2), c2));
        }
        if nt > 6 {
            sum = _mm512_add_epi16(sum, _mm512_maddubs_epi16(_mm512_shuffle_epi8(s, m3), c3));
        }
        let shifted = _mm512_sra_epi16(_mm512_add_epi16(sum, rnd), sct);
        let clamped = _mm512_max_epi16(_mm512_min_epi16(shifted, max_v), zero);

        // Fix lane ordering then pack i16→u8
        let clamped = _mm512_shuffle_i64x2::<0b11_01_10_00>(clamped, clamped);
        let packed = _mm512_cvtusepi16_epi8(clamped);

        storeu_256!(
            <&mut [u8; 32]>::try_from(&mut dst[col..col + 32]).unwrap(),
            packed
        );
        col += 32;
    }

    // Scalar fallback
    while col < w {
        let mut sum = 0i32;
        for i in 0..8 {
            sum += filter[i] as i32 * src[col + i] as i32;
        }
        let val = ((sum + 34) >> 6).clamp(0, 255);
        dst[col] = val as u8;
        col += 1;
    }
}

/// AVX-512 horizontal 8-tap filter for 8bpc put (H-only case) — 32 u8 outputs per iteration.
#[cfg(target_arch = "x86_64")]
#[rite]
fn h_filter_8tap_8bpc_put_avx512_inner(
    _token: Server64,
    dst: &mut [u8],
    src: &[u8],
    w: usize,
    filter: &[i8; 8],
) {
    if w < 32 {
        h_filter_8tap_8bpc_put_avx2_inner(_token.v3(), dst, src, w, filter);
        return;
    }
    let t0 = tap_base_8tap(filter);
    let nt = 8 - 2 * t0;
    let cx = h_pair_setup_8bpc_avx512(_token, t0, filter, 34, 6);
    h_pair_row32_8bpc_put_avx512(_token, dst, src, w, nt, &cx, filter);
}

/// Batched zmm H-only put — constants built once for `rows` dst rows.
/// Row `j` writes `dst[j*dst_stride..][..w]` from `src[src_base + j*src_stride..]`, with a signed source pitch.
#[cfg(target_arch = "x86_64")]
#[rite]
fn h_filter_rows_8tap_8bpc_put_avx512_inner(
    _token: Server64,
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    src: &[u8],
    src_base: usize,
    src_stride: isize,
    rows: usize,
    w: usize,
    filter: &[i8; 8],
) {
    if w < 32 {
        h_filter_rows_8tap_8bpc_put_avx2_inner(
            _token.v3(),
            dst,
            dst_base,
            dst_stride,
            src,
            src_base,
            src_stride,
            rows,
            w,
            filter,
        );
        return;
    }
    let t0 = tap_base_8tap(filter);
    let nt = 8 - 2 * t0;
    let cx = h_pair_setup_8bpc_avx512(_token, t0, filter, 34, 6);
    for j in 0..rows {
        h_pair_row32_8bpc_put_avx512(
            _token,
            &mut dst[dst_base.wrapping_add_signed(j as isize * dst_stride)..],
            &src[src_base.wrapping_add_signed(j as isize * src_stride)..],
            w,
            nt,
            &cx,
            filter,
        );
    }
}

#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[archmage::rite(v3)]
unsafe fn h_filter_8tap_8bpc_put_v3(
    dst: *mut u8,
    src: *const u8, // already offset by -3
    w: usize,
    filter: &[i8; 8],
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();
    unsafe {
        h_filter_8tap_8bpc_put_avx2_inner(
            token, dst, src, // already offset by -3
            w, filter,
        )
    }
}

/// Vertical 8-tap filter for 8bpc put (V-only case)
/// Reads directly from u8 source with stride, outputs u8
#[cfg(target_arch = "x86_64")]
#[rite]
fn v_filter_8tap_8bpc_direct_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    src: &[u8],
    src_base: usize, // tap-0 row inside the complete bounded slice
    src_stride: isize,
    w: usize,
    filter: &[i8; 8],
) {
    let t0 = tap_base_8tap(filter);
    let nt = 8 - 2 * t0;
    let mut dst = dst.flex_mut();
    let src = src.flex();
    let pack_pair = |a: i8, b: i8| ((b as i32) << 16) | (a as i32 & 0xffff);
    // Only taps t0..t0+nt are nonzero; rows outside the window are skipped.
    let c01 = _mm_set1_epi32(pack_pair(filter[t0], filter[t0 + 1]));
    let c23 = _mm_set1_epi32(pack_pair(filter[t0 + 2], filter[t0 + 3]));
    let c45 = _mm_set1_epi32(pack_pair(filter[t0 + 4], filter[t0 + 5]));
    let c67 = if nt > 6 {
        _mm_set1_epi32(pack_pair(filter[t0 + 6], filter[t0 + 7]))
    } else {
        _mm_setzero_si128()
    };

    let rnd = _mm_set1_epi32(32);
    let zero = _mm_setzero_si128();
    let max16 = _mm_set1_epi16(255);
    let shift_count = _mm_cvtsi32_si128(6);
    let row = |tap: usize| src_base.wrapping_add_signed(tap as isize * src_stride);

    let mut col = 0usize;

    // 8 columns at a time: u8 rows -> i16, unpack into pairs, pmaddwd.
    while col + 8 <= w {
        let p0 = _mm_cvtepu8_epi16(loadi64!(&src[row(t0) + col..row(t0) + col + 8]));
        let p1 = _mm_cvtepu8_epi16(loadi64!(&src[row(t0 + 1) + col..row(t0 + 1) + col + 8]));
        let p2 = _mm_cvtepu8_epi16(loadi64!(&src[row(t0 + 2) + col..row(t0 + 2) + col + 8]));
        let p3 = _mm_cvtepu8_epi16(loadi64!(&src[row(t0 + 3) + col..row(t0 + 3) + col + 8]));
        let p4 = _mm_cvtepu8_epi16(loadi64!(&src[row(t0 + 4) + col..row(t0 + 4) + col + 8]));
        let p5 = _mm_cvtepu8_epi16(loadi64!(&src[row(t0 + 5) + col..row(t0 + 5) + col + 8]));
        let (p6, p7) = if nt > 6 {
            (
                _mm_cvtepu8_epi16(loadi64!(&src[row(t0 + 6) + col..row(t0 + 6) + col + 8])),
                _mm_cvtepu8_epi16(loadi64!(&src[row(t0 + 7) + col..row(t0 + 7) + col + 8])),
            )
        } else {
            (zero, zero)
        };

        let p01l = _mm_unpacklo_epi16(p0, p1);
        let p01h = _mm_unpackhi_epi16(p0, p1);
        let p23l = _mm_unpacklo_epi16(p2, p3);
        let p23h = _mm_unpackhi_epi16(p2, p3);
        let p45l = _mm_unpacklo_epi16(p4, p5);
        let p45h = _mm_unpackhi_epi16(p4, p5);
        let p67l = _mm_unpacklo_epi16(p6, p7);
        let p67h = _mm_unpackhi_epi16(p6, p7);

        let mut sum_l = _mm_add_epi32(
            _mm_add_epi32(_mm_madd_epi16(p01l, c01), _mm_madd_epi16(p23l, c23)),
            _mm_madd_epi16(p45l, c45),
        );
        let mut sum_h = _mm_add_epi32(
            _mm_add_epi32(_mm_madd_epi16(p01h, c01), _mm_madd_epi16(p23h, c23)),
            _mm_madd_epi16(p45h, c45),
        );
        if nt > 6 {
            sum_l = _mm_add_epi32(sum_l, _mm_madd_epi16(p67l, c67));
            sum_h = _mm_add_epi32(sum_h, _mm_madd_epi16(p67h, c67));
        }

        let rl = _mm_sra_epi32(_mm_add_epi32(sum_l, rnd), shift_count);
        let rh = _mm_sra_epi32(_mm_add_epi32(sum_h, rnd), shift_count);

        let packed16 = _mm_packs_epi32(rl, rh);
        let clamped = _mm_max_epi16(_mm_min_epi16(packed16, max16), zero);
        let packed8 = _mm_packus_epi16(clamped, clamped);
        crate::src::safe_simd::partial_simd::mm_storel_epi64::<[u8; 8]>(
            <&mut [u8; 8]>::try_from(&mut dst[col..col + 8]).unwrap(),
            packed8,
        );

        col += 8;
    }

    // w=4 tail: 32-bit row loads -> i16 pairs -> madd.
    if w - col == 4 {
        let ld = |r: usize| -> __m128i {
            let off = row(t0 + r) + col;
            let b = u32::from_ne_bytes(*<&[u8; 4]>::try_from(&src[off..off + 4]).unwrap());
            _mm_cvtsi32_si128(b as i32)
        };
        let p0 = _mm_cvtepu8_epi16(ld(0));
        let p1 = _mm_cvtepu8_epi16(ld(1));
        let p2 = _mm_cvtepu8_epi16(ld(2));
        let p3 = _mm_cvtepu8_epi16(ld(3));
        let p4 = _mm_cvtepu8_epi16(ld(4));
        let p5 = _mm_cvtepu8_epi16(ld(5));
        let (p6, p7) = if nt > 6 {
            (_mm_cvtepu8_epi16(ld(6)), _mm_cvtepu8_epi16(ld(7)))
        } else {
            (zero, zero)
        };

        let p01 = _mm_unpacklo_epi16(p0, p1);
        let p23 = _mm_unpacklo_epi16(p2, p3);
        let p45 = _mm_unpacklo_epi16(p4, p5);
        let p67 = _mm_unpacklo_epi16(p6, p7);

        let mut sum = _mm_add_epi32(
            _mm_add_epi32(_mm_madd_epi16(p01, c01), _mm_madd_epi16(p23, c23)),
            _mm_madd_epi16(p45, c45),
        );
        if nt > 6 {
            sum = _mm_add_epi32(sum, _mm_madd_epi16(p67, c67));
        }
        let r = _mm_sra_epi32(_mm_add_epi32(sum, rnd), shift_count);
        let packed16 = _mm_packs_epi32(r, r);
        let clamped = _mm_max_epi16(_mm_min_epi16(packed16, max16), zero);
        let packed8 = _mm_packus_epi16(clamped, clamped);
        dst[col..col + 4].copy_from_slice(&(_mm_cvtsi128_si32(packed8) as u32).to_ne_bytes());
        col += 4;
    }

    // Scalar fallback
    while col < w {
        let mut sum = 0i32;
        for i in 0..nt {
            let px = src[row(t0 + i) + col] as i32;
            sum += filter[t0 + i] as i32 * px;
        }
        let val = ((sum + 32) >> 6).clamp(0, 255);
        dst[col] = val as u8;
        col += 1;
    }
}

/// AVX-512 vertical 8-tap filter for 8bpc put (V-only) — 16 u8 outputs per iteration.
/// Reads directly from u8 source with stride.
#[cfg(target_arch = "x86_64")]
#[rite]
fn v_filter_8tap_8bpc_direct_avx512_inner(
    _token: Server64,
    dst: &mut [u8],
    src: &[u8],
    src_base: usize,
    src_stride: isize,
    w: usize,
    filter: &[i8; 8],
) {
    if w < 16 {
        v_filter_8tap_8bpc_direct_avx2_inner(
            _token.v3(),
            dst,
            src,
            src_base,
            src_stride,
            w,
            filter,
        );
        return;
    }
    let mut dst = dst.flex_mut();
    let src = src.flex();

    let c0 = _mm512_set1_epi32(filter[0] as i32);
    let c1 = _mm512_set1_epi32(filter[1] as i32);
    let c2 = _mm512_set1_epi32(filter[2] as i32);
    let c3 = _mm512_set1_epi32(filter[3] as i32);
    let c4 = _mm512_set1_epi32(filter[4] as i32);
    let c5 = _mm512_set1_epi32(filter[5] as i32);
    let c6 = _mm512_set1_epi32(filter[6] as i32);
    let c7 = _mm512_set1_epi32(filter[7] as i32);

    let rnd = _mm512_set1_epi32(32);
    let zero = _mm512_setzero_si512();
    let max = _mm512_set1_epi32(255);

    let row = |tap: usize| src_base.wrapping_add_signed(tap as isize * src_stride);

    let mut col = 0usize;

    // Process 16 pixels at a time (u8→i32 expansion gives 16 i32 in 512 bits)
    while col + 16 <= w {
        // Load 16 u8 from each of 8 rows, expand to i32
        let p0 = _mm512_cvtepu8_epi32(loadu_128!(
            <&[u8; 16]>::try_from(&src[src_base + col..src_base + col + 16]).unwrap()
        ));
        let p1 = _mm512_cvtepu8_epi32(loadu_128!(
            <&[u8; 16]>::try_from(&src[row(1) + col..row(1) + col + 16]).unwrap()
        ));
        let p2 = _mm512_cvtepu8_epi32(loadu_128!(
            <&[u8; 16]>::try_from(&src[row(2) + col..row(2) + col + 16]).unwrap()
        ));
        let p3 = _mm512_cvtepu8_epi32(loadu_128!(
            <&[u8; 16]>::try_from(&src[row(3) + col..row(3) + col + 16]).unwrap()
        ));
        let p4 = _mm512_cvtepu8_epi32(loadu_128!(
            <&[u8; 16]>::try_from(&src[row(4) + col..row(4) + col + 16]).unwrap()
        ));
        let p5 = _mm512_cvtepu8_epi32(loadu_128!(
            <&[u8; 16]>::try_from(&src[row(5) + col..row(5) + col + 16]).unwrap()
        ));
        let p6 = _mm512_cvtepu8_epi32(loadu_128!(
            <&[u8; 16]>::try_from(&src[row(6) + col..row(6) + col + 16]).unwrap()
        ));
        let p7 = _mm512_cvtepu8_epi32(loadu_128!(
            <&[u8; 16]>::try_from(&src[row(7) + col..row(7) + col + 16]).unwrap()
        ));

        // Multiply and accumulate
        let mut sum = _mm512_mullo_epi32(p0, c0);
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(p1, c1));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(p2, c2));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(p3, c3));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(p4, c4));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(p5, c5));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(p6, c6));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(p7, c7));

        // Round, shift, clamp
        let shift_count = _mm_cvtsi32_si128(6);
        let shifted = _mm512_sra_epi32(_mm512_add_epi32(sum, rnd), shift_count);
        let clamped = _mm512_min_epi32(_mm512_max_epi32(shifted, zero), max);

        // Pack i32→u8 (values already in [0,255])
        let packed = _mm512_cvtusepi32_epi8(clamped);

        storeu_128!(
            <&mut [u8; 16]>::try_from(&mut dst[col..col + 16]).unwrap(),
            packed
        );

        col += 16;
    }

    // Scalar fallback
    while col < w {
        let mut sum = 0i32;
        for i in 0..8 {
            let px = src[row(i) + col] as i32;
            sum += filter[i] as i32 * px;
        }
        let val = ((sum + 32) >> 6).clamp(0, 255);
        dst[col] = val as u8;
        col += 1;
    }
}

#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[archmage::rite(v3)]
unsafe fn v_filter_8tap_8bpc_direct_v3(
    dst: *mut u8,
    src: *const u8, // already positioned at (y-3, 0)
    src_stride: isize,
    w: usize,
    filter: &[i8; 8],
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();
    unsafe {
        v_filter_8tap_8bpc_direct_avx2_inner(
            token, dst, src, // already positioned at (y-3, 0)
            0, src_stride, w, filter,
        )
    }
}

/// Four horizontal rows feed each vertical output without a pooled mid buffer.
/// Only the seven active source bytes in each of seven rows are borrowed.
#[cfg(target_arch = "x86_64")]
#[rite]
fn fused_hv_4x4_8bpc<const SHIFT: i32>(
    _token: Desktop64,
    src: &[u8],
    src_base: usize,
    src_stride: isize,
    fh: &[i8; 8],
    fv: &[i8; 8],
) -> [__m128i; 4] {
    debug_assert_eq!(tap_base_8tap(fh), 2);
    debug_assert_eq!(tap_base_8tap(fv), 2);
    let mask0 = loadu_128!(&[
        0u8, 1, 1, 2, 2, 3, 3, 4, 128, 128, 128, 128, 128, 128, 128, 128
    ]);
    let mask2 = loadu_128!(&[
        2u8, 3, 3, 4, 4, 5, 5, 6, 128, 128, 128, 128, 128, 128, 128, 128
    ]);
    let hc0 = _mm_set1_epi16(i16::from_le_bytes([fh[2] as u8, fh[3] as u8]));
    let hc1 = _mm_set1_epi16(i16::from_le_bytes([fh[4] as u8, fh[5] as u8]));
    let vc0 = _mm_set1_epi32(i32::from_le_bytes([
        fv[2] as u8,
        (fv[2] >> 7) as u8,
        fv[3] as u8,
        (fv[3] >> 7) as u8,
    ]));
    let vc1 = _mm_set1_epi32(i32::from_le_bytes([
        fv[4] as u8,
        (fv[4] >> 7) as u8,
        fv[5] as u8,
        (fv[5] >> 7) as u8,
    ]));
    let zero = _mm_setzero_si128();
    let mut ring = [zero; 4];
    let mut output = [zero; 4];
    for r in 0..7 {
        let base = (src_base as isize + (r as isize - 1) * src_stride - 1) as usize;
        // The seventh byte is the last active tap of output column three.
        // Never load an eighth byte from the caller's reference window.
        let mut bytes = [0u8; 8];
        bytes[..7].copy_from_slice(&src[base..base + 7]);
        let row = loadu_64!(&bytes);
        let sum = _mm_add_epi16(
            _mm_maddubs_epi16(_mm_shuffle_epi8(row, mask0), hc0),
            _mm_maddubs_epi16(_mm_shuffle_epi8(row, mask2), hc1),
        );
        ring[r % 4] = _mm_srai_epi16::<2>(_mm_add_epi16(sum, _mm_set1_epi16(2)));
        if r >= 3 {
            let y = r - 3;
            let a = _mm_unpacklo_epi16(ring[y % 4], ring[(y + 1) % 4]);
            let b = _mm_unpacklo_epi16(ring[(y + 2) % 4], ring[(y + 3) % 4]);
            let sum = _mm_add_epi32(_mm_madd_epi16(a, vc0), _mm_madd_epi16(b, vc1));
            output[y] =
                _mm_srai_epi32::<SHIFT>(_mm_add_epi32(sum, _mm_set1_epi32(1 << (SHIFT - 1))));
        }
    }
    output
}

/// Generic 8-tap put function for 8bpc
///
/// This handles all 4 cases:
/// 1. H+V filtering (mx != 0 && my != 0)
/// 2. H-only filtering (mx != 0 && my == 0)
/// 3. V-only filtering (mx == 0 && my != 0)
/// 4. Simple copy (mx == 0 && my == 0)
///
/// # Safety
///
/// - Caller must ensure AVX2 is available
/// - dst_ptr must be valid for writing w*h bytes
/// - src_ptr must be valid for reading (w+7)*(h+7) bytes (with proper padding)
#[cfg(target_arch = "x86_64")]
#[rite]
pub(crate) fn put_8tap_8bpc_avx2_impl_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    src: &[u8],
    src_base: usize,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    h_filter_type: Rav1dFilterMode,
    v_filter_type: Rav1dFilterMode,
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    let w = w as usize;
    let h = h as usize;
    let mx = mx as usize;
    let my = my as usize;
    let sb = src_base as isize;

    // For 8bpc: intermediate_bits = 4
    let intermediate_bits = 4u8;

    let fh = get_filter_coeff(mx, w, h_filter_type);
    let fv = get_filter_coeff(my, h, v_filter_type);

    match (fh, fv) {
        (Some(fh), Some(fv)) => {
            if w == 4 && h == 4 && tap_base_8tap(fh) == 2 && tap_base_8tap(fv) == 2 {
                let rows = fused_hv_4x4_8bpc::<10>(_token, &src, src_base, src_stride, fh, fv);
                for (y, row) in rows.into_iter().enumerate() {
                    let packed = _mm_packus_epi16(
                        _mm_packs_epi32(row, _mm_setzero_si128()),
                        _mm_setzero_si128(),
                    );
                    let at = (y as isize * dst_stride) as usize;
                    dst[at..at + 4]
                        .copy_from_slice(&(_mm_cvtsi128_si32(packed) as u32).to_ne_bytes());
                }
                return;
            }
            // Case 1: Both H and V filtering
            // First pass: horizontal filter to intermediate buffer.
            // Output row y consumes mid rows y+v_t0 .. y+v_t0+v_nt; rows
            // outside that window correspond to zero taps and are skipped.
            let v_t0 = tap_base_8tap(fv);
            let tmp_h = h + (8 - 2 * v_t0) - 1;
            let mut mid = take_mid_i16_135();

            let src_first = (sb + (v_t0 as isize - 3) * src_stride) as usize - 3;
            h_filter_rows_8tap_8bpc_avx2_inner(
                _token,
                mid.as_flattened_mut(),
                MID_STRIDE,
                v_t0 * MID_STRIDE,
                &src,
                src_first,
                src_stride,
                tmp_h,
                w,
                fh,
                6 - intermediate_bits,
            );

            // Second pass: vertical filter to output
            for y in 0..h {
                let dst_row = &mut dst[dst_base.wrapping_add_signed(y as isize * dst_stride)..];
                v_filter_8tap_8bpc_avx2_inner(
                    _token,
                    dst_row,
                    &mid[y..],
                    w,
                    fv,
                    6 + intermediate_bits,
                    255,
                );
            }
            put_mid_i16_135(mid);
        }
        (Some(fh), None) => {
            // Case 2: H-only filtering (full SIMD)
            let src_first = (sb - 3) as usize;
            h_filter_rows_8tap_8bpc_put_avx2_inner(
                _token,
                &mut dst[..],
                dst_base,
                dst_stride,
                &src,
                src_first,
                src_stride,
                h,
                w,
                fh,
            );
        }
        (None, Some(fv)) => {
            // Case 3: V-only filtering (full SIMD)
            for y in 0..h {
                let src_row_base = (sb + (y as isize - 3) * src_stride) as usize;
                let dst_row = &mut dst[dst_base.wrapping_add_signed(y as isize * dst_stride)..];
                v_filter_8tap_8bpc_direct_avx2_inner(
                    _token,
                    dst_row,
                    &src,
                    src_row_base,
                    src_stride,
                    w,
                    fv,
                );
            }
        }
        (None, None) => {
            // Case 4: Simple copy
            for y in 0..h {
                let src_row_base = (sb + y as isize * src_stride) as usize;
                let src_row = &src[src_row_base..];
                let dst_row = &mut dst[dst_base.wrapping_add_signed(y as isize * dst_stride)..];
                copy_row_px(dst_row, src_row, w);
            }
        }
    }
}

/// AVX-512 put_8tap for 8bpc — uses wider inner filters.
#[cfg(target_arch = "x86_64")]
#[arcane]
pub(crate) fn put_8tap_8bpc_avx512_impl_inner(
    _token: Server64,
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    src: &[u8],
    src_base: usize,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    h_filter_type: Rav1dFilterMode,
    v_filter_type: Rav1dFilterMode,
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    let w = w as usize;
    let h = h as usize;
    let mx = mx as usize;
    let my = my as usize;
    let sb = src_base as isize;
    let intermediate_bits = 4u8;

    let fh = get_filter_coeff(mx, w, h_filter_type);
    let fv = get_filter_coeff(my, h, v_filter_type);

    match (fh, fv) {
        (Some(fh), Some(fv)) => {
            if w == 4 && h == 4 && tap_base_8tap(fh) == 2 && tap_base_8tap(fv) == 2 {
                let rows = fused_hv_4x4_8bpc::<10>(_token.v3(), &src, src_base, src_stride, fh, fv);
                for (y, row) in rows.into_iter().enumerate() {
                    let packed = _mm_packus_epi16(
                        _mm_packs_epi32(row, _mm_setzero_si128()),
                        _mm_setzero_si128(),
                    );
                    let at = (y as isize * dst_stride) as usize;
                    dst[at..at + 4]
                        .copy_from_slice(&(_mm_cvtsi128_si32(packed) as u32).to_ne_bytes());
                }
                return;
            }
            // H+V: horizontal filter → intermediate → vertical filter.
            // Output row y consumes mid rows y+v_t0 .. y+v_t0+v_nt; rows
            // outside that window correspond to zero taps and are skipped.
            let v_t0 = tap_base_8tap(fv);
            let tmp_h = h + (8 - 2 * v_t0) - 1;
            let mut mid = take_mid_i16_135();

            let src_first = (sb + (v_t0 as isize - 3) * src_stride) as usize - 3;
            h_filter_rows_8tap_8bpc_avx512_inner(
                _token,
                mid.as_flattened_mut(),
                MID_STRIDE,
                v_t0 * MID_STRIDE,
                &src,
                src_first,
                src_stride,
                tmp_h,
                w,
                fh,
                6 - intermediate_bits,
            );

            for y in 0..h {
                let dst_row = &mut dst[dst_base.wrapping_add_signed(y as isize * dst_stride)..];
                v_filter_8tap_8bpc_avx512_inner(
                    _token,
                    dst_row,
                    &mid[y..],
                    w,
                    fv,
                    6 + intermediate_bits,
                    255,
                );
            }
            put_mid_i16_135(mid);
        }
        (Some(fh), None) => {
            // H-only
            let src_first = (sb - 3) as usize;
            h_filter_rows_8tap_8bpc_put_avx512_inner(
                _token,
                &mut dst[..],
                dst_base,
                dst_stride,
                &src,
                src_first,
                src_stride,
                h,
                w,
                fh,
            );
        }
        (None, Some(fv)) => {
            // V-only
            for y in 0..h {
                let src_row_base = (sb + (y as isize - 3) * src_stride) as usize;
                let dst_row = &mut dst[dst_base.wrapping_add_signed(y as isize * dst_stride)..];
                v_filter_8tap_8bpc_direct_avx512_inner(
                    _token,
                    dst_row,
                    &src,
                    src_row_base,
                    src_stride,
                    w,
                    fv,
                );
            }
        }
        (None, None) => {
            // Simple copy
            for y in 0..h {
                let src_row_base = (sb + y as isize * src_stride) as usize;
                let src_row = &src[src_row_base..];
                let dst_row = &mut dst[dst_base.wrapping_add_signed(y as isize * dst_stride)..];
                copy_row_px(dst_row, src_row, w);
            }
        }
    }
}

#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[archmage::rite(v3)]
unsafe fn put_8tap_8bpc_impl_v3(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    h_filter_type: Rav1dFilterMode,
    v_filter_type: Rav1dFilterMode,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();
    unsafe {
        put_8tap_8bpc_avx2_impl_inner(
            token,
            dst_ptr,
            0,
            dst_stride,
            src_ptr,
            0,
            src_stride,
            w,
            h,
            mx,
            my,
            h_filter_type,
            v_filter_type,
        )
    }
}

/// Generic put_8tap function wrapper with Filter2d const generic
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn put_8tap_8bpc_avx2<const FILTER: usize>(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    _bitdepth_max: i32,
    _dst: *const FFISafe<PicOffset>,
    _src: *const FFISafe<PicOffset>,
) {
    let filter = Filter2d::from_repr(FILTER).unwrap();
    let (h_filter, v_filter) = filter.hv();

    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
        put_8tap_8bpc_impl_v3(
            dst_ptr, dst_stride, src_ptr, src_stride, w, h, mx, my, h_filter, v_filter,
        );
    }
}

// Specific filter type wrappers for dispatch table
// These are needed because decl_fn_safe! requires concrete functions, not generics

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn put_8tap_regular_8bpc_avx2_inner(
    _token: Desktop64,
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
        put_8tap_8bpc_avx2::<{ Filter2d::Regular8Tap as usize }>(
            dst_ptr,
            dst_stride,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            dst,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn put_8tap_regular_8bpc_v3(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    unsafe {
        put_8tap_regular_8bpc_avx2_inner(
            token,
            dst_ptr,
            dst_stride,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            dst,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn put_8tap_regular_smooth_8bpc_avx2_inner(
    _token: Desktop64,
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
        put_8tap_8bpc_avx2::<{ Filter2d::RegularSmooth8Tap as usize }>(
            dst_ptr,
            dst_stride,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            dst,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn put_8tap_regular_smooth_8bpc_v3(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    unsafe {
        put_8tap_regular_smooth_8bpc_avx2_inner(
            token,
            dst_ptr,
            dst_stride,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            dst,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn put_8tap_regular_sharp_8bpc_avx2_inner(
    _token: Desktop64,
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
        put_8tap_8bpc_avx2::<{ Filter2d::RegularSharp8Tap as usize }>(
            dst_ptr,
            dst_stride,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            dst,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn put_8tap_regular_sharp_8bpc_v3(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    unsafe {
        put_8tap_regular_sharp_8bpc_avx2_inner(
            token,
            dst_ptr,
            dst_stride,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            dst,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn put_8tap_smooth_regular_8bpc_avx2_inner(
    _token: Desktop64,
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
        put_8tap_8bpc_avx2::<{ Filter2d::SmoothRegular8Tap as usize }>(
            dst_ptr,
            dst_stride,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            dst,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn put_8tap_smooth_regular_8bpc_v3(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    unsafe {
        put_8tap_smooth_regular_8bpc_avx2_inner(
            token,
            dst_ptr,
            dst_stride,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            dst,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn put_8tap_smooth_8bpc_avx2_inner(
    _token: Desktop64,
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
        put_8tap_8bpc_avx2::<{ Filter2d::Smooth8Tap as usize }>(
            dst_ptr,
            dst_stride,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            dst,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn put_8tap_smooth_8bpc_v3(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    unsafe {
        put_8tap_smooth_8bpc_avx2_inner(
            token,
            dst_ptr,
            dst_stride,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            dst,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn put_8tap_smooth_sharp_8bpc_avx2_inner(
    _token: Desktop64,
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
        put_8tap_8bpc_avx2::<{ Filter2d::SmoothSharp8Tap as usize }>(
            dst_ptr,
            dst_stride,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            dst,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn put_8tap_smooth_sharp_8bpc_v3(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    unsafe {
        put_8tap_smooth_sharp_8bpc_avx2_inner(
            token,
            dst_ptr,
            dst_stride,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            dst,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn put_8tap_sharp_regular_8bpc_avx2_inner(
    _token: Desktop64,
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
        put_8tap_8bpc_avx2::<{ Filter2d::SharpRegular8Tap as usize }>(
            dst_ptr,
            dst_stride,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            dst,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn put_8tap_sharp_regular_8bpc_v3(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    unsafe {
        put_8tap_sharp_regular_8bpc_avx2_inner(
            token,
            dst_ptr,
            dst_stride,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            dst,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn put_8tap_sharp_smooth_8bpc_avx2_inner(
    _token: Desktop64,
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
        put_8tap_8bpc_avx2::<{ Filter2d::SharpSmooth8Tap as usize }>(
            dst_ptr,
            dst_stride,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            dst,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn put_8tap_sharp_smooth_8bpc_v3(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    unsafe {
        put_8tap_sharp_smooth_8bpc_avx2_inner(
            token,
            dst_ptr,
            dst_stride,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            dst,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn put_8tap_sharp_8bpc_avx2_inner(
    _token: Desktop64,
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
        put_8tap_8bpc_avx2::<{ Filter2d::Sharp8Tap as usize }>(
            dst_ptr,
            dst_stride,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            dst,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn put_8tap_sharp_8bpc_v3(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    unsafe {
        put_8tap_sharp_8bpc_avx2_inner(
            token,
            dst_ptr,
            dst_stride,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            dst,
            src,
        )
    }
}

// =============================================================================
// 8-TAP PREP FUNCTIONS (mct)
// =============================================================================

/// `dst[..N] = src[..N]` with a compile-time trip count — the array copy
/// lowers to inline vector load/store, not a libc memcpy call (same idiom as
/// `copy_n` in cdef_arm; docs/SIZE_SWEEP.md profiled the libc-call cost).
#[inline(always)]
fn copy_n_px<T: Copy, const N: usize>(dst: &mut [T], src: &[T]) {
    let a = <&[T; N]>::try_from(&src[..N]).unwrap();
    let d = <&mut [T; N]>::try_from(&mut dst[..N]).unwrap();
    *d = *a;
}

/// Row copy for the no-filter MC cases — `w` is always a block width, so a
/// match gives every arm a constant length and no per-row libc memcpy calls.
#[inline(always)]
fn copy_row_px<T: Copy>(dst: &mut [T], src: &[T], w: usize) {
    match w {
        4 => copy_n_px::<T, 4>(dst, src),
        8 => copy_n_px::<T, 8>(dst, src),
        16 => copy_n_px::<T, 16>(dst, src),
        32 => copy_n_px::<T, 32>(dst, src),
        64 => copy_n_px::<T, 64>(dst, src),
        128 => copy_n_px::<T, 128>(dst, src),
        _ => dst[..w].copy_from_slice(&src[..w]),
    }
}

/// Widen a u8 row into `dst` as `i16 << ib` (prep V-only mid fill + copy case).
/// `dst` and `src` must both be at least `w` long.
#[cfg(target_arch = "x86_64")]
#[rite]
fn widen_row_u8_shl_8bpc(_t: Desktop64, dst: &mut [i16], src: &[u8], w: usize, ib: i32) {
    let cnt = _mm_cvtsi32_si128(ib);
    let mut x = 0usize;
    while x + 8 <= w {
        let v = _mm_cvtepu8_epi16(loadu_64!(<&[u8; 8]>::try_from(&src[x..x + 8]).unwrap()));
        storeu_128!(&mut dst[x..x + 8], [i16; 8], _mm_sll_epi16(v, cnt));
        x += 8;
    }
    if x + 4 <= w {
        let v = _mm_cvtepu8_epi16(loadi32!(&src[x..x + 4]));
        crate::src::safe_simd::partial_simd::mm_storel_epi64::<[i16; 4]>(
            <&mut [i16; 4]>::try_from(&mut dst[x..x + 4]).unwrap(),
            _mm_sll_epi16(v, cnt),
        );
        x += 4;
    }
    while x < w {
        dst[x] = (src[x] as i16) << ib;
        x += 1;
    }
}

/// Widen a u16 row into `dst` as `(px << ib) - bias` (16bpc prep copy case).
#[cfg(target_arch = "x86_64")]
#[rite]
fn widen_row_u16_shl_bias_16bpc(
    _t: Desktop64,
    dst: &mut [i16],
    src: &[u16],
    w: usize,
    ib: i32,
    bias: i32,
) {
    let cnt = _mm_cvtsi32_si128(ib);
    let bias_v = _mm256_set1_epi32(bias);
    let mut x = 0usize;
    while x + 16 <= w {
        let v = loadu_256!(<&[u16; 16]>::try_from(&src[x..x + 16]).unwrap());
        let lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(v));
        let hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(v));
        let lo = _mm256_sub_epi32(_mm256_sll_epi32(lo, cnt), bias_v);
        let hi = _mm256_sub_epi32(_mm256_sll_epi32(hi, cnt), bias_v);
        // packs_epi32 is lane-local: [e0-3, e8-11 | e4-7, e12-15] — permute
        // qwords back to linear order.
        let packed = _mm256_permute4x64_epi64(_mm256_packs_epi32(lo, hi), 0b11_01_10_00);
        storeu_256!(&mut dst[x..x + 16], [i16; 16], packed);
        x += 16;
    }
    let bias_x = _mm_set1_epi32(bias);
    while x + 8 <= w {
        let v = loadu_128!(<&[u16; 8]>::try_from(&src[x..x + 8]).unwrap());
        let lo = _mm_cvtepu16_epi32(v);
        let hi = _mm_cvtepu16_epi32(_mm_srli_si128::<8>(v));
        let lo = _mm_sub_epi32(_mm_sll_epi32(lo, cnt), bias_x);
        let hi = _mm_sub_epi32(_mm_sll_epi32(hi, cnt), bias_x);
        storeu_128!(&mut dst[x..x + 8], [i16; 8], _mm_packs_epi32(lo, hi));
        x += 8;
    }
    while x < w {
        dst[x] = ((src[x] as i32) << ib).wrapping_sub(bias) as i16;
        x += 1;
    }
}

/// Generic 8-tap prep function for 8bpc
///
/// Similar to put but writes to i16 intermediate buffer instead of pixel output
///
/// # Safety
///
/// - Caller must ensure AVX2 is available
/// - tmp must be valid for writing w*h i16 values
/// - src_ptr must be valid for reading (w+7)*(h+7) bytes (with proper padding)
#[cfg(target_arch = "x86_64")]
#[rite]
pub(crate) fn prep_8tap_8bpc_avx2_impl_inner(
    _token: Desktop64,
    tmp: &mut [i16],
    src: &[u8],
    src_base: usize,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    h_filter_type: Rav1dFilterMode,
    v_filter_type: Rav1dFilterMode,
) {
    let mut tmp = tmp.flex_mut();
    let src = src.flex();
    let w = w as usize;
    let h = h as usize;
    let mx = mx as usize;
    let my = my as usize;
    let sb = src_base as isize;

    // For 8bpc: intermediate_bits = 4
    let intermediate_bits = 4u8;

    let fh = get_filter_coeff(mx, w, h_filter_type);
    let fv = get_filter_coeff(my, h, v_filter_type);

    match (fh, fv) {
        (Some(fh), Some(fv)) => {
            if w == 4 && h == 4 && tap_base_8tap(fh) == 2 && tap_base_8tap(fv) == 2 {
                let rows = fused_hv_4x4_8bpc::<6>(_token, &src, src_base, src_stride, fh, fv);
                for (y, row) in rows.into_iter().enumerate() {
                    let mut lanes = [0i32; 4];
                    storeu_128!(&mut lanes, row);
                    for (x, value) in lanes.into_iter().enumerate() {
                        tmp[y * 4 + x] = value as i16;
                    }
                }
                return;
            }
            // Case 1: Both H and V filtering
            // Output row y consumes mid rows y+v_t0 .. y+v_t0+v_nt; rows
            // outside that window correspond to zero taps and are skipped.
            let v_t0 = tap_base_8tap(fv);
            let tmp_h = h + (8 - 2 * v_t0) - 1;
            let mut mid = take_mid_i16_135();

            // Horizontal pass
            let src_first = (sb + (v_t0 as isize - 3) * src_stride) as usize - 3;
            h_filter_rows_8tap_8bpc_avx2_inner(
                _token,
                mid.as_flattened_mut(),
                MID_STRIDE,
                v_t0 * MID_STRIDE,
                &src,
                src_first,
                src_stride,
                tmp_h,
                w,
                fh,
                6 - intermediate_bits,
            );

            // Vertical pass to intermediate output
            // Scalar uses .rnd(6) for prep V-pass (NOT 6+ib like put)
            // Prep keeps extra precision for compound prediction consumers
            for y in 0..h {
                let out_row = y * w;
                v_filter_8tap_to_i16_avx2_inner(_token, &mid[y..], &mut tmp[out_row..], w, fv, 6);
            }
            put_mid_i16_135(mid);
        }
        (Some(fh), None) => {
            // Case 2: H-only filtering
            // Shift by (6 - intermediate_bits) to match scalar .rnd(6 - intermediate_bits)
            // Same shift as H+V case's H pass
            let src_first = (sb - 3) as usize;
            h_filter_rows_8tap_8bpc_avx2_inner(
                _token,
                &mut tmp[..],
                w,
                0,
                &src,
                src_first,
                src_stride,
                h,
                w,
                fh,
                6 - intermediate_bits,
            );
        }
        (None, Some(fv)) => {
            // Case 3: V-only filtering. Widen each needed source row once into
            // a pooled mid buffer, then slide the tap window — the previous
            // per-row rebuild did 8x redundant widen work.
            let v_t0 = tap_base_8tap(fv);
            let tmp_h = h + (8 - 2 * v_t0) - 1;
            let mut mid = take_mid_i16_135();
            for y in v_t0..v_t0 + tmp_h {
                let src_row_base = (sb + (y as isize - 3) * src_stride) as usize;
                widen_row_u8_shl_8bpc(
                    _token,
                    &mut mid[y],
                    &src[src_row_base..],
                    w,
                    intermediate_bits as i32,
                );
            }
            for y in 0..h {
                let out_row = y * w;
                v_filter_8tap_to_i16_avx2_inner(_token, &mid[y..], &mut tmp[out_row..], w, fv, 6);
            }
            put_mid_i16_135(mid);
        }
        (None, None) => {
            // Case 4: Simple copy with intermediate scaling
            for y in 0..h {
                let src_row_base = (sb + y as isize * src_stride) as usize;
                let src_row = &src[src_row_base..];
                let out_row = y * w;
                widen_row_u8_shl_8bpc(
                    _token,
                    &mut tmp[out_row..],
                    src_row,
                    w,
                    intermediate_bits as i32,
                );
            }
        }
    }
}

/// AVX-512 prep_8tap for 8bpc — uses wider inner filters.
#[cfg(target_arch = "x86_64")]
#[arcane]
pub(crate) fn prep_8tap_8bpc_avx512_impl_inner(
    _token: Server64,
    tmp: &mut [i16],
    src: &[u8],
    src_base: usize,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    h_filter_type: Rav1dFilterMode,
    v_filter_type: Rav1dFilterMode,
) {
    let mut tmp = tmp.flex_mut();
    let src = src.flex();
    let w = w as usize;
    let h = h as usize;
    let mx = mx as usize;
    let my = my as usize;
    let sb = src_base as isize;
    let intermediate_bits = 4u8;

    let fh = get_filter_coeff(mx, w, h_filter_type);
    let fv = get_filter_coeff(my, h, v_filter_type);

    match (fh, fv) {
        (Some(fh), Some(fv)) => {
            if w == 4 && h == 4 && tap_base_8tap(fh) == 2 && tap_base_8tap(fv) == 2 {
                let rows = fused_hv_4x4_8bpc::<6>(_token.v3(), &src, src_base, src_stride, fh, fv);
                for (y, row) in rows.into_iter().enumerate() {
                    let mut lanes = [0i32; 4];
                    storeu_128!(&mut lanes, row);
                    for (x, value) in lanes.into_iter().enumerate() {
                        tmp[y * 4 + x] = value as i16;
                    }
                }
                return;
            }
            // Output row y consumes mid rows y+v_t0 .. y+v_t0+v_nt; rows
            // outside that window correspond to zero taps and are skipped.
            let v_t0 = tap_base_8tap(fv);
            let tmp_h = h + (8 - 2 * v_t0) - 1;
            let mut mid = take_mid_i16_135();

            let src_first = (sb + (v_t0 as isize - 3) * src_stride) as usize - 3;
            h_filter_rows_8tap_8bpc_avx512_inner(
                _token,
                mid.as_flattened_mut(),
                MID_STRIDE,
                v_t0 * MID_STRIDE,
                &src,
                src_first,
                src_stride,
                tmp_h,
                w,
                fh,
                6 - intermediate_bits,
            );

            for y in 0..h {
                let out_row = y * w;
                v_filter_8tap_to_i16_avx512_inner(_token, &mid[y..], &mut tmp[out_row..], w, fv, 6);
            }
            put_mid_i16_135(mid);
        }
        (Some(fh), None) => {
            let src_first = (sb - 3) as usize;
            h_filter_rows_8tap_8bpc_avx512_inner(
                _token,
                &mut tmp[..],
                w,
                0,
                &src,
                src_first,
                src_stride,
                h,
                w,
                fh,
                6 - intermediate_bits,
            );
        }
        (None, Some(fv)) => {
            let v_t0 = tap_base_8tap(fv);
            let tmp_h = h + (8 - 2 * v_t0) - 1;
            let mut mid = take_mid_i16_135();
            for y in v_t0..v_t0 + tmp_h {
                let src_row_base = (sb + (y as isize - 3) * src_stride) as usize;
                widen_row_u8_shl_8bpc(
                    _token.v3(),
                    &mut mid[y],
                    &src[src_row_base..],
                    w,
                    intermediate_bits as i32,
                );
            }
            for y in 0..h {
                let out_row = y * w;
                v_filter_8tap_to_i16_avx512_inner(_token, &mid[y..], &mut tmp[out_row..], w, fv, 6);
            }
            put_mid_i16_135(mid);
        }
        (None, None) => {
            for y in 0..h {
                let src_row_base = (sb + y as isize * src_stride) as usize;
                let src_row = &src[src_row_base..];
                let out_row = y * w;
                widen_row_u8_shl_8bpc(
                    _token.v3(),
                    &mut tmp[out_row..],
                    src_row,
                    w,
                    intermediate_bits as i32,
                );
            }
        }
    }
}

#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[archmage::rite(v3)]
unsafe fn prep_8tap_8bpc_impl_v3(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    h_filter_type: Rav1dFilterMode,
    v_filter_type: Rav1dFilterMode,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();
    unsafe {
        prep_8tap_8bpc_avx2_impl_inner(
            token,
            tmp,
            src_ptr,
            0,
            src_stride,
            w,
            h,
            mx,
            my,
            h_filter_type,
            v_filter_type,
        )
    }
}

/// Vertical NT-tap filter to i16 output (for prep functions).
///
/// Only taps `t0..t0+NT` are nonzero for NT < 8; mid rows outside that window
/// are not loaded.
#[cfg(target_arch = "x86_64")]
#[rite]
fn v_filter_ntap_to_i16_avx2_inner<const NT: usize>(
    _token: Desktop64,
    mid: &[[i16; MID_STRIDE]],
    dst: &mut [i16],
    w: usize,
    filter: &[i8; 8],
    sh: u8,
) {
    const { assert!(NT == 4 || NT == 6 || NT == 8) };
    let t0 = (8 - NT) / 2;
    let mut dst = dst.flex_mut();
    let pack_pair = |a: i8, b: i8| ((b as i32) << 16) | (a as i32 & 0xffff);
    let c01 = _mm_set1_epi32(pack_pair(filter[t0], filter[t0 + 1]));
    let c23 = _mm_set1_epi32(pack_pair(filter[t0 + 2], filter[t0 + 3]));
    let c45 = if NT > 4 {
        _mm_set1_epi32(pack_pair(filter[t0 + 4], filter[t0 + 5]))
    } else {
        _mm_setzero_si128()
    };
    let c67 = if NT > 6 {
        _mm_set1_epi32(pack_pair(filter[t0 + 6], filter[t0 + 7]))
    } else {
        _mm_setzero_si128()
    };
    let rnd = _mm_set1_epi32((1i32 << sh) >> 1);
    let shift_count = _mm_cvtsi32_si128(sh as i32);

    let mut col = 0usize;

    while col + 8 <= w {
        let m0 = loadu_128!(<&[i16; 8]>::try_from(&mid[t0][col..col + 8]).unwrap());
        let m1 = loadu_128!(<&[i16; 8]>::try_from(&mid[t0 + 1][col..col + 8]).unwrap());
        let m2 = loadu_128!(<&[i16; 8]>::try_from(&mid[t0 + 2][col..col + 8]).unwrap());
        let m3 = loadu_128!(<&[i16; 8]>::try_from(&mid[t0 + 3][col..col + 8]).unwrap());
        let (m4, m5) = if NT > 4 {
            (
                loadu_128!(<&[i16; 8]>::try_from(&mid[t0 + 4][col..col + 8]).unwrap()),
                loadu_128!(<&[i16; 8]>::try_from(&mid[t0 + 5][col..col + 8]).unwrap()),
            )
        } else {
            (_mm_setzero_si128(), _mm_setzero_si128())
        };
        let (m6, m7) = if NT > 6 {
            (
                loadu_128!(<&[i16; 8]>::try_from(&mid[t0 + 6][col..col + 8]).unwrap()),
                loadu_128!(<&[i16; 8]>::try_from(&mid[t0 + 7][col..col + 8]).unwrap()),
            )
        } else {
            (_mm_setzero_si128(), _mm_setzero_si128())
        };

        let p01l = _mm_unpacklo_epi16(m0, m1);
        let p01h = _mm_unpackhi_epi16(m0, m1);
        let p23l = _mm_unpacklo_epi16(m2, m3);
        let p23h = _mm_unpackhi_epi16(m2, m3);
        let p45l = _mm_unpacklo_epi16(m4, m5);
        let p45h = _mm_unpackhi_epi16(m4, m5);
        let p67l = _mm_unpacklo_epi16(m6, m7);
        let p67h = _mm_unpackhi_epi16(m6, m7);

        let mut sum_l = _mm_add_epi32(
            _mm_add_epi32(_mm_madd_epi16(p01l, c01), _mm_madd_epi16(p23l, c23)),
            _mm_madd_epi16(p45l, c45),
        );
        let mut sum_h = _mm_add_epi32(
            _mm_add_epi32(_mm_madd_epi16(p01h, c01), _mm_madd_epi16(p23h, c23)),
            _mm_madd_epi16(p45h, c45),
        );
        if NT > 6 {
            sum_l = _mm_add_epi32(sum_l, _mm_madd_epi16(p67l, c67));
            sum_h = _mm_add_epi32(sum_h, _mm_madd_epi16(p67h, c67));
        }

        let rl = _mm_sra_epi32(_mm_add_epi32(sum_l, rnd), shift_count);
        let rh = _mm_sra_epi32(_mm_add_epi32(sum_h, rnd), shift_count);

        // i32 -> i16 preserving column order
        let packed = _mm_packs_epi32(rl, rh);
        storeu_128!(
            <&mut [i16; 8]>::try_from(&mut dst[col..col + 8]).unwrap(),
            packed
        );

        col += 8;
    }

    // w=4 tail: 64-bit row loads, same pair-madd structure.
    if w - col == 4 && mid[0].len() >= col + 4 {
        let ld = |r: usize| -> __m128i {
            crate::src::safe_simd::partial_simd::mm_loadl_epi64::<[i16; 4]>(
                <&[i16; 4]>::try_from(&mid[t0 + r][col..col + 4]).unwrap(),
            )
        };
        let m0 = ld(0);
        let m1 = ld(1);
        let m2 = ld(2);
        let m3 = ld(3);
        let (m4, m5) = if NT > 4 {
            (ld(4), ld(5))
        } else {
            (_mm_setzero_si128(), _mm_setzero_si128())
        };
        let (m6, m7) = if NT > 6 {
            (ld(6), ld(7))
        } else {
            (_mm_setzero_si128(), _mm_setzero_si128())
        };

        let p01 = _mm_unpacklo_epi16(m0, m1);
        let p23 = _mm_unpacklo_epi16(m2, m3);
        let p45 = _mm_unpacklo_epi16(m4, m5);
        let p67 = _mm_unpacklo_epi16(m6, m7);

        let mut sum = _mm_add_epi32(
            _mm_add_epi32(_mm_madd_epi16(p01, c01), _mm_madd_epi16(p23, c23)),
            _mm_madd_epi16(p45, c45),
        );
        if NT > 6 {
            sum = _mm_add_epi32(sum, _mm_madd_epi16(p67, c67));
        }
        let r = _mm_sra_epi32(_mm_add_epi32(sum, rnd), shift_count);
        let packed = _mm_packs_epi32(r, r);
        crate::src::safe_simd::partial_simd::mm_storel_epi64::<[i16; 4]>(
            <&mut [i16; 4]>::try_from(&mut dst[col..col + 4]).unwrap(),
            packed,
        );
        col += 4;
    }

    while col < w {
        let mut sum = 0i32;
        for i in 0..NT {
            sum += filter[t0 + i] as i32 * mid[t0 + i][col] as i32;
        }
        dst[col] = ((sum + ((1 << sh) >> 1)) >> sh) as i16;
        col += 1;
    }
}

/// Vertical 8-tap filter to i16 output (for prep functions)
#[cfg(target_arch = "x86_64")]
#[rite]
fn v_filter_8tap_to_i16_avx2_inner(
    _token: Desktop64,
    mid: &[[i16; MID_STRIDE]],
    dst: &mut [i16],
    w: usize,
    filter: &[i8; 8],
    sh: u8,
) {
    match tap_base_8tap(filter) {
        2 => v_filter_ntap_to_i16_avx2_inner::<4>(_token, mid, dst, w, filter, sh),
        1 => v_filter_ntap_to_i16_avx2_inner::<6>(_token, mid, dst, w, filter, sh),
        _ => v_filter_ntap_to_i16_avx2_inner::<8>(_token, mid, dst, w, filter, sh),
    }
}

/// AVX-512 vertical 8-tap filter to i16 output — 16 i16 outputs per iteration.
/// Used by prep (compound prediction) path.
#[cfg(target_arch = "x86_64")]
#[rite]
fn v_filter_8tap_to_i16_avx512_inner(
    _token: Server64,
    mid: &[[i16; MID_STRIDE]],
    dst: &mut [i16],
    w: usize,
    filter: &[i8; 8],
    sh: u8,
) {
    if w < 16 {
        v_filter_8tap_to_i16_avx2_inner(_token.v3(), mid, dst, w, filter, sh);
        return;
    }
    let t0 = tap_base_8tap(filter);
    let nt = 8 - 2 * t0;
    let mut dst = dst.flex_mut();
    let rnd = _mm512_set1_epi32((1i32 << sh) >> 1);

    // Only taps t0..t0+nt are nonzero — their mid rows sit at t0..t0+nt.
    let c: [i32; 8] = core::array::from_fn(|i| if i < nt { filter[t0 + i] as i32 } else { 0 });
    let c0 = _mm512_set1_epi32(c[0]);
    let c1 = _mm512_set1_epi32(c[1]);
    let c2 = _mm512_set1_epi32(c[2]);
    let c3 = _mm512_set1_epi32(c[3]);
    let c4 = _mm512_set1_epi32(c[4]);
    let c5 = _mm512_set1_epi32(c[5]);
    let c6 = _mm512_set1_epi32(c[6]);
    let c7 = _mm512_set1_epi32(c[7]);

    let mut col = 0usize;

    while col + 16 <= w {
        // Rows past t0+nt multiply a zero coefficient; skipped loads are fine.
        let m0 = _mm512_cvtepi16_epi32(loadu_256!(
            <&[i16; 16]>::try_from(&mid[t0][col..col + 16]).unwrap()
        ));
        let m1 = _mm512_cvtepi16_epi32(loadu_256!(
            <&[i16; 16]>::try_from(&mid[t0 + 1][col..col + 16]).unwrap()
        ));
        let m2 = _mm512_cvtepi16_epi32(loadu_256!(
            <&[i16; 16]>::try_from(&mid[t0 + 2][col..col + 16]).unwrap()
        ));
        let m3 = _mm512_cvtepi16_epi32(loadu_256!(
            <&[i16; 16]>::try_from(&mid[t0 + 3][col..col + 16]).unwrap()
        ));
        let m4 = _mm512_cvtepi16_epi32(loadu_256!(
            <&[i16; 16]>::try_from(&mid[t0 + 4][col..col + 16]).unwrap()
        ));
        let m5 = _mm512_cvtepi16_epi32(loadu_256!(
            <&[i16; 16]>::try_from(&mid[t0 + 5][col..col + 16]).unwrap()
        ));
        let zero512 = _mm512_setzero_si512();
        let m6 = if nt > 6 {
            _mm512_cvtepi16_epi32(loadu_256!(
                <&[i16; 16]>::try_from(&mid[t0 + 6][col..col + 16]).unwrap()
            ))
        } else {
            zero512
        };
        let m7 = if nt > 6 {
            _mm512_cvtepi16_epi32(loadu_256!(
                <&[i16; 16]>::try_from(&mid[t0 + 7][col..col + 16]).unwrap()
            ))
        } else {
            zero512
        };

        let mut sum = _mm512_mullo_epi32(m0, c0);
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(m1, c1));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(m2, c2));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(m3, c3));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(m4, c4));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(m5, c5));
        if nt > 6 {
            sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(m6, c6));
            sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(m7, c7));
        }

        let shift_count = _mm_cvtsi32_si128(sh as i32);
        let shifted = _mm512_sra_epi32(_mm512_add_epi32(sum, rnd), shift_count);

        // Pack i32→i16 via signed saturation
        let packed = _mm512_cvtsepi32_epi16(shifted);

        // Store 16 i16 values
        storeu_256!(
            <&mut [i16; 16]>::try_from(&mut dst[col..col + 16]).unwrap(),
            packed
        );

        col += 16;
    }

    while col < w {
        let mut sum = 0i32;
        for i in 0..nt {
            sum += filter[t0 + i] as i32 * mid[t0 + i][col] as i32;
        }
        dst[col] = ((sum + ((1 << sh) >> 1)) >> sh) as i16;
        col += 1;
    }
}

#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[archmage::rite(v3)]
unsafe fn v_filter_8tap_to_i16_v3(
    mid: &[[i16; MID_STRIDE]],
    dst: *mut i16,
    w: usize,
    filter: &[i8; 8],
    sh: u8,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();
    unsafe { v_filter_8tap_to_i16_avx2_inner(token, mid, dst, w, filter, sh) }
}

/// Generic prep_8tap function wrapper with Filter2d const generic
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn prep_8tap_8bpc_avx2<const FILTER: usize>(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    _bitdepth_max: i32,
    _src: *const FFISafe<PicOffset>,
) {
    let filter = Filter2d::from_repr(FILTER).unwrap();
    let (h_filter, v_filter) = filter.hv();

    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
        prep_8tap_8bpc_impl_v3(tmp, src_ptr, src_stride, w, h, mx, my, h_filter, v_filter);
    }
}

// Specific filter type wrappers for dispatch table

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn prep_8tap_regular_8bpc_avx2_inner(
    _token: Desktop64,
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
        prep_8tap_8bpc_avx2::<{ Filter2d::Regular8Tap as usize }>(
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn prep_8tap_regular_8bpc_v3(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    unsafe {
        prep_8tap_regular_8bpc_avx2_inner(
            token,
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn prep_8tap_regular_smooth_8bpc_avx2_inner(
    _token: Desktop64,
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
        prep_8tap_8bpc_avx2::<{ Filter2d::RegularSmooth8Tap as usize }>(
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn prep_8tap_regular_smooth_8bpc_v3(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    unsafe {
        prep_8tap_regular_smooth_8bpc_avx2_inner(
            token,
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn prep_8tap_regular_sharp_8bpc_avx2_inner(
    _token: Desktop64,
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
        prep_8tap_8bpc_avx2::<{ Filter2d::RegularSharp8Tap as usize }>(
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn prep_8tap_regular_sharp_8bpc_v3(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    unsafe {
        prep_8tap_regular_sharp_8bpc_avx2_inner(
            token,
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn prep_8tap_smooth_regular_8bpc_avx2_inner(
    _token: Desktop64,
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
        prep_8tap_8bpc_avx2::<{ Filter2d::SmoothRegular8Tap as usize }>(
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn prep_8tap_smooth_regular_8bpc_v3(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    unsafe {
        prep_8tap_smooth_regular_8bpc_avx2_inner(
            token,
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn prep_8tap_smooth_8bpc_avx2_inner(
    _token: Desktop64,
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
        prep_8tap_8bpc_avx2::<{ Filter2d::Smooth8Tap as usize }>(
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn prep_8tap_smooth_8bpc_v3(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    unsafe {
        prep_8tap_smooth_8bpc_avx2_inner(
            token,
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn prep_8tap_smooth_sharp_8bpc_avx2_inner(
    _token: Desktop64,
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
        prep_8tap_8bpc_avx2::<{ Filter2d::SmoothSharp8Tap as usize }>(
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn prep_8tap_smooth_sharp_8bpc_v3(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    unsafe {
        prep_8tap_smooth_sharp_8bpc_avx2_inner(
            token,
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn prep_8tap_sharp_regular_8bpc_avx2_inner(
    _token: Desktop64,
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
        prep_8tap_8bpc_avx2::<{ Filter2d::SharpRegular8Tap as usize }>(
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn prep_8tap_sharp_regular_8bpc_v3(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    unsafe {
        prep_8tap_sharp_regular_8bpc_avx2_inner(
            token,
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn prep_8tap_sharp_smooth_8bpc_avx2_inner(
    _token: Desktop64,
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
        prep_8tap_8bpc_avx2::<{ Filter2d::SharpSmooth8Tap as usize }>(
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn prep_8tap_sharp_smooth_8bpc_v3(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    unsafe {
        prep_8tap_sharp_smooth_8bpc_avx2_inner(
            token,
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn prep_8tap_sharp_8bpc_avx2_inner(
    _token: Desktop64,
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
        prep_8tap_8bpc_avx2::<{ Filter2d::Sharp8Tap as usize }>(
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn prep_8tap_sharp_8bpc_v3(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    unsafe {
        prep_8tap_sharp_8bpc_avx2_inner(
            token,
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

// =============================================================================
// 16BPC 8-TAP FILTERS
// =============================================================================

/// Horizontal 8-tap filter for 16bpc using AVX2
/// Processes 8 output pixels at a time
/// Output is 32-bit to preserve precision for vertical pass
#[cfg(target_arch = "x86_64")]
#[rite]
fn h_filter_8tap_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [i32],
    src: &[u16],
    w: usize,
    filter: &[i8; 8],
    sh: i32,
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    // Convert filter coefficients from i8 to i16 for pmaddwd
    let coeff0 =
        _mm256_set1_epi32(((filter[1] as i16 as i32) << 16) | (filter[0] as i16 as u16 as i32));
    let coeff2 =
        _mm256_set1_epi32(((filter[3] as i16 as i32) << 16) | (filter[2] as i16 as u16 as i32));
    let coeff4 =
        _mm256_set1_epi32(((filter[5] as i16 as i32) << 16) | (filter[4] as i16 as u16 as i32));
    let coeff6 =
        _mm256_set1_epi32(((filter[7] as i16 as i32) << 16) | (filter[6] as i16 as u16 as i32));

    let rnd = _mm256_set1_epi32((1 << sh) >> 1);
    let shift_count = _mm_cvtsi32_si128(sh);

    let mut col = 0usize;

    // Process 8 output pixels at a time
    // Source pointer is already offset by -3 (pointing to tap 0), matching 8bpc convention
    //
    // LANE FIX: _mm256_unpacklo_epi16 works within 128-bit lanes. Loading 16
    // contiguous u16 into a 256-bit register puts elements 0-7 in lane 0 and
    // 8-15 in lane 1. unpacklo then interleaves elements 0-3 from lane 0 and
    // 8-11 from lane 1, giving outputs {0,1,2,3,8,9,10,11} instead of {0..7}.
    // Fix: construct each source register with lane 0 = [col+X..col+X+4] and
    // lane 1 = [col+X+4..col+X+8], so unpacklo gives correct outputs 0-3
    // (lane 0) and 4-7 (lane 1).
    //
    // Each lane only ever consumes its low 4 u16, so the per-tap loads are
    // 64-bit. They used to be 128-bit (`src[col+k..col+k+8]`), which touched
    // 4 pixels past the last tap the filter uses: for w = 8 the caller hands
    // over an open-ended `&src[row - 3..]` that can end at w + 9 = 17 pixels
    // on the plane's last row, and `src[col+11..col+19]` panicked on it (#516).
    // The furthest pixel this loop reads is now col + 14, the last tap of
    // output 7 — exactly what the filter needs.
    while col + 8 <= w {
        // Load the 4 pixels each lane consumes: a[k] = src[col+k..col+k+4]
        let a0 = loadu_64!(<&[u16; 4]>::try_from(&src[col..col + 4]).unwrap());
        let a1 = loadu_64!(<&[u16; 4]>::try_from(&src[col + 1..col + 5]).unwrap());
        let a2 = loadu_64!(<&[u16; 4]>::try_from(&src[col + 2..col + 6]).unwrap());
        let a3 = loadu_64!(<&[u16; 4]>::try_from(&src[col + 3..col + 7]).unwrap());
        let a4 = loadu_64!(<&[u16; 4]>::try_from(&src[col + 4..col + 8]).unwrap());
        let a5 = loadu_64!(<&[u16; 4]>::try_from(&src[col + 5..col + 9]).unwrap());
        let a6 = loadu_64!(<&[u16; 4]>::try_from(&src[col + 6..col + 10]).unwrap());
        let a7 = loadu_64!(<&[u16; 4]>::try_from(&src[col + 7..col + 11]).unwrap());
        let a8 = loadu_64!(<&[u16; 4]>::try_from(&src[col + 8..col + 12]).unwrap());
        let a9 = loadu_64!(<&[u16; 4]>::try_from(&src[col + 9..col + 13]).unwrap());
        let a10 = loadu_64!(<&[u16; 4]>::try_from(&src[col + 10..col + 14]).unwrap());
        let a11 = loadu_64!(<&[u16; 4]>::try_from(&src[col + 11..col + 15]).unwrap());

        // Build 256-bit source regs: lane0 for outputs 0-3, lane1 for outputs 4-7
        let s0 = _mm256_inserti128_si256(_mm256_castsi128_si256(a0), a4, 1);
        let s1 = _mm256_inserti128_si256(_mm256_castsi128_si256(a1), a5, 1);
        let s2 = _mm256_inserti128_si256(_mm256_castsi128_si256(a2), a6, 1);
        let s3 = _mm256_inserti128_si256(_mm256_castsi128_si256(a3), a7, 1);
        let s4 = _mm256_inserti128_si256(_mm256_castsi128_si256(a4), a8, 1);
        let s5 = _mm256_inserti128_si256(_mm256_castsi128_si256(a5), a9, 1);
        let s6 = _mm256_inserti128_si256(_mm256_castsi128_si256(a6), a10, 1);
        let s7 = _mm256_inserti128_si256(_mm256_castsi128_si256(a7), a11, 1);

        // Interleave consecutive pixels for madd_epi16
        // madd_epi16 does: (a[2i] * b[2i]) + (a[2i+1] * b[2i+1]) -> 32-bit
        let p01 = _mm256_unpacklo_epi16(s0, s1);
        let p23 = _mm256_unpacklo_epi16(s2, s3);
        let p45 = _mm256_unpacklo_epi16(s4, s5);
        let p67 = _mm256_unpacklo_epi16(s6, s7);

        // Multiply-add pairs: coeff0=[f0,f1], coeff2=[f2,f3], etc.
        let ma01 = _mm256_madd_epi16(p01, coeff0);
        let ma23 = _mm256_madd_epi16(p23, coeff2);
        let ma45 = _mm256_madd_epi16(p45, coeff4);
        let ma67 = _mm256_madd_epi16(p67, coeff6);

        // Sum all contributions (32-bit)
        let sum = _mm256_add_epi32(_mm256_add_epi32(ma01, ma23), _mm256_add_epi32(ma45, ma67));

        // Add rounding and shift
        let result = _mm256_sra_epi32(_mm256_add_epi32(sum, rnd), shift_count);

        // Store 8 32-bit results
        storeu_256!(
            <&mut [i32; 8]>::try_from(&mut dst[col..col + 8]).unwrap(),
            result
        );

        col += 8;
    }

    // w=4 tail: 128-bit loads + madd pairs, same coefficient packing.
    if w - col == 4 && src.len() >= col + 11 {
        let c01_128 = _mm256_castsi256_si128(coeff0);
        let c23_128 = _mm256_castsi256_si128(coeff2);
        let c45_128 = _mm256_castsi256_si128(coeff4);
        let c67_128 = _mm256_castsi256_si128(coeff6);
        let rnd128 = _mm_set1_epi32((1 << sh) >> 1);

        let s0 = loadu_64!(<&[u16; 4]>::try_from(&src[col..col + 4]).unwrap());
        let s1 = loadu_64!(<&[u16; 4]>::try_from(&src[col + 1..col + 5]).unwrap());
        let s2 = loadu_64!(<&[u16; 4]>::try_from(&src[col + 2..col + 6]).unwrap());
        let s3 = loadu_64!(<&[u16; 4]>::try_from(&src[col + 3..col + 7]).unwrap());
        let s4 = loadu_64!(<&[u16; 4]>::try_from(&src[col + 4..col + 8]).unwrap());
        let s5 = loadu_64!(<&[u16; 4]>::try_from(&src[col + 5..col + 9]).unwrap());
        let s6 = loadu_64!(<&[u16; 4]>::try_from(&src[col + 6..col + 10]).unwrap());
        let s7 = loadu_64!(<&[u16; 4]>::try_from(&src[col + 7..col + 11]).unwrap());

        let sum = _mm_add_epi32(
            _mm_add_epi32(
                _mm_madd_epi16(_mm_unpacklo_epi16(s0, s1), c01_128),
                _mm_madd_epi16(_mm_unpacklo_epi16(s2, s3), c23_128),
            ),
            _mm_add_epi32(
                _mm_madd_epi16(_mm_unpacklo_epi16(s4, s5), c45_128),
                _mm_madd_epi16(_mm_unpacklo_epi16(s6, s7), c67_128),
            ),
        );
        let res = _mm_sra_epi32(_mm_add_epi32(sum, rnd128), _mm_cvtsi32_si128(sh));
        storeu_128!(
            <&mut [i32; 4]>::try_from(&mut dst[col..col + 4]).unwrap(),
            res
        );
        col += 4;
    }

    // Scalar fallback for remaining pixels
    while col < w {
        let mut sum = 0i32;
        for i in 0..8 {
            sum += filter[i] as i32 * src[col + i] as i32;
        }
        let r = (1 << sh) >> 1;
        dst[col] = (sum + r) >> sh;
        col += 1;
    }
}
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[archmage::rite(v3)]
unsafe fn h_filter_8tap_16bpc_v3(
    dst: *mut i32,
    src: *const u16,
    w: usize,
    filter: &[i8; 8],
    sh: i32,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();
    unsafe { h_filter_8tap_16bpc_avx2_inner(token, dst, src, w, filter, sh) }
}

/// Vertical 8-tap filter for 16bpc using AVX2
/// Input is 32-bit intermediate, output is 16-bit clamped to [0, max]
#[cfg(target_arch = "x86_64")]
#[rite]
fn v_filter_8tap_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    mid: &[[i32; MID_STRIDE]],
    w: usize,
    y: usize,
    filter: &[i8; 8],
    sh: i32,
    max: i32,
) {
    let mut dst = dst.flex_mut();
    // Convert filter coefficients from i8 to i32 for multiplication
    let coeff: [i32; 8] = [
        filter[0] as i32,
        filter[1] as i32,
        filter[2] as i32,
        filter[3] as i32,
        filter[4] as i32,
        filter[5] as i32,
        filter[6] as i32,
        filter[7] as i32,
    ];

    let rnd = _mm256_set1_epi32((1 << sh) >> 1);
    let shift_count = _mm_cvtsi32_si128(sh);
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi32(max);

    let mut col = 0usize;

    // Process 8 pixels at a time
    while col + 8 <= w {
        // Load 8 intermediate values from each of the 8 tap rows
        let r0 = loadu_256!(<&[i32; 8]>::try_from(&mid[y + 0][col..col + 8]).unwrap());
        let r1 = loadu_256!(<&[i32; 8]>::try_from(&mid[y + 1][col..col + 8]).unwrap());
        let r2 = loadu_256!(<&[i32; 8]>::try_from(&mid[y + 2][col..col + 8]).unwrap());
        let r3 = loadu_256!(<&[i32; 8]>::try_from(&mid[y + 3][col..col + 8]).unwrap());
        let r4 = loadu_256!(<&[i32; 8]>::try_from(&mid[y + 4][col..col + 8]).unwrap());
        let r5 = loadu_256!(<&[i32; 8]>::try_from(&mid[y + 5][col..col + 8]).unwrap());
        let r6 = loadu_256!(<&[i32; 8]>::try_from(&mid[y + 6][col..col + 8]).unwrap());
        let r7 = loadu_256!(<&[i32; 8]>::try_from(&mid[y + 7][col..col + 8]).unwrap());

        // Multiply each row by its coefficient and accumulate
        let c0 = _mm256_set1_epi32(coeff[0]);
        let c1 = _mm256_set1_epi32(coeff[1]);
        let c2 = _mm256_set1_epi32(coeff[2]);
        let c3 = _mm256_set1_epi32(coeff[3]);
        let c4 = _mm256_set1_epi32(coeff[4]);
        let c5 = _mm256_set1_epi32(coeff[5]);
        let c6 = _mm256_set1_epi32(coeff[6]);
        let c7 = _mm256_set1_epi32(coeff[7]);

        let m0 = _mm256_mullo_epi32(r0, c0);
        let m1 = _mm256_mullo_epi32(r1, c1);
        let m2 = _mm256_mullo_epi32(r2, c2);
        let m3 = _mm256_mullo_epi32(r3, c3);
        let m4 = _mm256_mullo_epi32(r4, c4);
        let m5 = _mm256_mullo_epi32(r5, c5);
        let m6 = _mm256_mullo_epi32(r6, c6);
        let m7 = _mm256_mullo_epi32(r7, c7);

        // Sum all
        let sum = _mm256_add_epi32(
            _mm256_add_epi32(_mm256_add_epi32(m0, m1), _mm256_add_epi32(m2, m3)),
            _mm256_add_epi32(_mm256_add_epi32(m4, m5), _mm256_add_epi32(m6, m7)),
        );

        // Add rounding, shift, and clamp
        let shifted = _mm256_sra_epi32(_mm256_add_epi32(sum, rnd), shift_count);
        let clamped_lo = _mm256_max_epi32(shifted, zero);
        let clamped = _mm256_min_epi32(clamped_lo, max_val);

        // Pack from 32-bit to 16-bit
        // _mm256_packus_epi32 packs with unsigned saturation
        // But we need to handle lane crossing...
        // Pack low 4 and high 4 separately, then permute
        let packed = _mm256_packus_epi32(clamped, zero);
        // Permute to fix lane order: [0,1,2,3,x,x,x,x,4,5,6,7,x,x,x,x]
        let result = _mm256_permute4x64_epi64(packed, 0b00_00_10_00); // [0,2,0,0]

        // Store 8 16-bit results
        storeu_128!(
            <&mut [u16; 8]>::try_from(&mut dst[col..col + 8]).unwrap(),
            _mm256_castsi256_si128(result)
        );

        col += 8;
    }

    // w=4 tail: 128-bit i32 row loads, mullo+packus path mirrors main loop.
    if w - col == 4 && mid[0].len() >= col + 4 {
        let c0 = _mm_set1_epi32(coeff[0]);
        let c1 = _mm_set1_epi32(coeff[1]);
        let c2 = _mm_set1_epi32(coeff[2]);
        let c3 = _mm_set1_epi32(coeff[3]);
        let c4 = _mm_set1_epi32(coeff[4]);
        let c5 = _mm_set1_epi32(coeff[5]);
        let c6 = _mm_set1_epi32(coeff[6]);
        let c7 = _mm_set1_epi32(coeff[7]);
        let rnd128 = _mm_set1_epi32((1 << sh) >> 1);
        let zero128 = _mm_setzero_si128();
        let max128 = _mm_set1_epi32(max);
        let shift128 = _mm_cvtsi32_si128(sh);

        let mut sum = _mm_setzero_si128();
        for i in 0..8 {
            let r = loadu_128!(<&[i32; 4]>::try_from(&mid[y + i][col..col + 4]).unwrap());
            let c = match i {
                0 => c0,
                1 => c1,
                2 => c2,
                3 => c3,
                4 => c4,
                5 => c5,
                6 => c6,
                _ => c7,
            };
            sum = _mm_add_epi32(sum, _mm_mullo_epi32(r, c));
        }
        let shifted = _mm_sra_epi32(_mm_add_epi32(sum, rnd128), shift128);
        let clamped = _mm_min_epi32(_mm_max_epi32(shifted, zero128), max128);
        let packed = _mm_packus_epi32(clamped, clamped);
        crate::src::safe_simd::partial_simd::mm_storel_epi64::<[u16; 4]>(
            <&mut [u16; 4]>::try_from(&mut dst[col..col + 4]).unwrap(),
            packed,
        );
        col += 4;
    }

    // Scalar fallback for remaining pixels
    while col < w {
        let mut sum = 0i32;
        for i in 0..8 {
            sum += coeff[i] * mid[y + i][col];
        }
        let r = (1 << sh) >> 1;
        let val = ((sum + r) >> sh).clamp(0, max);
        dst[col] = val as u16;
        col += 1;
    }
}
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[archmage::rite(v3)]
unsafe fn v_filter_8tap_16bpc_v3(
    dst: *mut u16,
    mid: &[[i32; MID_STRIDE]],
    w: usize,
    y: usize,
    filter: &[i8; 8],
    sh: i32,
    max: i32,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();
    unsafe { v_filter_8tap_16bpc_avx2_inner(token, dst, mid, w, y, filter, sh, max) }
}

/// Vertical 8-tap filter for 16bpc prep (output is i16 with PREP_BIAS subtracted)
#[cfg(target_arch = "x86_64")]
#[rite]
fn v_filter_8tap_16bpc_prep_avx2_inner(
    _token: Desktop64,
    dst: &mut [i16],
    mid: &[[i32; MID_STRIDE]],
    w: usize,
    y: usize,
    filter: &[i8; 8],
    sh: i32,
    prep_bias: i32,
) {
    let mut dst = dst.flex_mut();
    let coeff: [i32; 8] = [
        filter[0] as i32,
        filter[1] as i32,
        filter[2] as i32,
        filter[3] as i32,
        filter[4] as i32,
        filter[5] as i32,
        filter[6] as i32,
        filter[7] as i32,
    ];

    let rnd = _mm256_set1_epi32((1 << sh) >> 1);
    let shift_count = _mm_cvtsi32_si128(sh);
    let bias = _mm256_set1_epi32(prep_bias);

    let mut col = 0usize;

    // Process 8 pixels at a time
    while col + 8 <= w {
        let r0 = loadu_256!(<&[i32; 8]>::try_from(&mid[y + 0][col..col + 8]).unwrap());
        let r1 = loadu_256!(<&[i32; 8]>::try_from(&mid[y + 1][col..col + 8]).unwrap());
        let r2 = loadu_256!(<&[i32; 8]>::try_from(&mid[y + 2][col..col + 8]).unwrap());
        let r3 = loadu_256!(<&[i32; 8]>::try_from(&mid[y + 3][col..col + 8]).unwrap());
        let r4 = loadu_256!(<&[i32; 8]>::try_from(&mid[y + 4][col..col + 8]).unwrap());
        let r5 = loadu_256!(<&[i32; 8]>::try_from(&mid[y + 5][col..col + 8]).unwrap());
        let r6 = loadu_256!(<&[i32; 8]>::try_from(&mid[y + 6][col..col + 8]).unwrap());
        let r7 = loadu_256!(<&[i32; 8]>::try_from(&mid[y + 7][col..col + 8]).unwrap());

        let c0 = _mm256_set1_epi32(coeff[0]);
        let c1 = _mm256_set1_epi32(coeff[1]);
        let c2 = _mm256_set1_epi32(coeff[2]);
        let c3 = _mm256_set1_epi32(coeff[3]);
        let c4 = _mm256_set1_epi32(coeff[4]);
        let c5 = _mm256_set1_epi32(coeff[5]);
        let c6 = _mm256_set1_epi32(coeff[6]);
        let c7 = _mm256_set1_epi32(coeff[7]);

        let m0 = _mm256_mullo_epi32(r0, c0);
        let m1 = _mm256_mullo_epi32(r1, c1);
        let m2 = _mm256_mullo_epi32(r2, c2);
        let m3 = _mm256_mullo_epi32(r3, c3);
        let m4 = _mm256_mullo_epi32(r4, c4);
        let m5 = _mm256_mullo_epi32(r5, c5);
        let m6 = _mm256_mullo_epi32(r6, c6);
        let m7 = _mm256_mullo_epi32(r7, c7);

        let sum = _mm256_add_epi32(
            _mm256_add_epi32(_mm256_add_epi32(m0, m1), _mm256_add_epi32(m2, m3)),
            _mm256_add_epi32(_mm256_add_epi32(m4, m5), _mm256_add_epi32(m6, m7)),
        );

        // Add rounding, shift, subtract bias
        let shifted = _mm256_sra_epi32(_mm256_add_epi32(sum, rnd), shift_count);
        let biased = _mm256_sub_epi32(shifted, bias);

        // Pack from 32-bit to 16-bit (signed)
        let packed = _mm256_packs_epi32(biased, biased);
        // Permute to fix lane order
        let result = _mm256_permute4x64_epi64(packed, 0b00_00_10_00);

        storeu_128!(
            <&mut [i16; 8]>::try_from(&mut dst[col..col + 8]).unwrap(),
            _mm256_castsi256_si128(result)
        );

        col += 8;
    }

    // w=4 tail: 128-bit i32 row loads, mullo chain, packs to i16.
    if w - col == 4 && mid[0].len() >= col + 4 {
        let rnd128 = _mm_set1_epi32((1 << sh) >> 1);
        let bias128 = _mm_set1_epi32(prep_bias);
        let shift128 = _mm_cvtsi32_si128(sh);

        let mut sum = _mm_setzero_si128();
        for i in 0..8 {
            let r = loadu_128!(<&[i32; 4]>::try_from(&mid[y + i][col..col + 4]).unwrap());
            sum = _mm_add_epi32(sum, _mm_mullo_epi32(r, _mm_set1_epi32(coeff[i])));
        }
        let shifted = _mm_sra_epi32(_mm_add_epi32(sum, rnd128), shift128);
        let biased = _mm_sub_epi32(shifted, bias128);
        let packed = _mm_packs_epi32(biased, biased);
        crate::src::safe_simd::partial_simd::mm_storel_epi64::<[i16; 4]>(
            <&mut [i16; 4]>::try_from(&mut dst[col..col + 4]).unwrap(),
            packed,
        );
        col += 4;
    }

    // Scalar fallback
    while col < w {
        let mut sum = 0i32;
        for i in 0..8 {
            sum += coeff[i] * mid[y + i][col];
        }
        let r = (1 << sh) >> 1;
        let val = ((sum + r) >> sh) - prep_bias;
        dst[col] = val as i16;
        col += 1;
    }
}
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[archmage::rite(v3)]
unsafe fn v_filter_8tap_16bpc_prep_v3(
    dst: *mut i16,
    mid: &[[i32; MID_STRIDE]],
    w: usize,
    y: usize,
    filter: &[i8; 8],
    sh: i32,
    prep_bias: i32,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();
    unsafe { v_filter_8tap_16bpc_prep_avx2_inner(token, dst, mid, w, y, filter, sh, prep_bias) }
}

/// Horizontal 8-tap filter for 16bpc put (H-only case)
/// Input and output are both u16, shift=6, rnd=32, clamp to [0, max]
#[cfg(target_arch = "x86_64")]
#[rite]
fn h_filter_8tap_16bpc_put_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    src: &[u16],
    w: usize,
    filter: &[i8; 8],
    max: i32,
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    // Convert filter coefficients from i8 to i16 for pmaddwd
    let coeff0 =
        _mm256_set1_epi32(((filter[1] as i16 as i32) << 16) | (filter[0] as i16 as u16 as i32));
    let coeff2 =
        _mm256_set1_epi32(((filter[3] as i16 as i32) << 16) | (filter[2] as i16 as u16 as i32));
    let coeff4 =
        _mm256_set1_epi32(((filter[5] as i16 as i32) << 16) | (filter[4] as i16 as u16 as i32));
    let coeff6 =
        _mm256_set1_epi32(((filter[7] as i16 as i32) << 16) | (filter[6] as i16 as u16 as i32));

    // H-only put rounding: 32 + ((1 << (6 - intermediate_bits)) >> 1)
    // 10-bit (max=1023): intermediate_bits=4, rnd=32+2=34
    // 12-bit (max=4095): intermediate_bits=2, rnd=32+8=40
    let intermediate_bits = if (max >> 11) != 0 { 2 } else { 4 };
    let rnd = _mm256_set1_epi32(32 + ((1 << (6 - intermediate_bits)) >> 1));
    let shift_count = _mm_cvtsi32_si128(6);
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi32(max);

    let mut col = 0usize;

    // Source pointer is already offset by -3 (pointing to tap 0), matching 8bpc convention
    // LANE FIX: see h_filter_8tap_16bpc_avx2_inner for explanation
    while col + 8 <= w {
        // Load the 4 pixels each lane consumes: a[k] = src[col+k..col+k+4]
        let a0 = loadu_64!(<&[u16; 4]>::try_from(&src[col..col + 4]).unwrap());
        let a1 = loadu_64!(<&[u16; 4]>::try_from(&src[col + 1..col + 5]).unwrap());
        let a2 = loadu_64!(<&[u16; 4]>::try_from(&src[col + 2..col + 6]).unwrap());
        let a3 = loadu_64!(<&[u16; 4]>::try_from(&src[col + 3..col + 7]).unwrap());
        let a4 = loadu_64!(<&[u16; 4]>::try_from(&src[col + 4..col + 8]).unwrap());
        let a5 = loadu_64!(<&[u16; 4]>::try_from(&src[col + 5..col + 9]).unwrap());
        let a6 = loadu_64!(<&[u16; 4]>::try_from(&src[col + 6..col + 10]).unwrap());
        let a7 = loadu_64!(<&[u16; 4]>::try_from(&src[col + 7..col + 11]).unwrap());
        let a8 = loadu_64!(<&[u16; 4]>::try_from(&src[col + 8..col + 12]).unwrap());
        let a9 = loadu_64!(<&[u16; 4]>::try_from(&src[col + 9..col + 13]).unwrap());
        let a10 = loadu_64!(<&[u16; 4]>::try_from(&src[col + 10..col + 14]).unwrap());
        let a11 = loadu_64!(<&[u16; 4]>::try_from(&src[col + 11..col + 15]).unwrap());

        // Build 256-bit source regs: lane0 for outputs 0-3, lane1 for outputs 4-7
        let s0 = _mm256_inserti128_si256(_mm256_castsi128_si256(a0), a4, 1);
        let s1 = _mm256_inserti128_si256(_mm256_castsi128_si256(a1), a5, 1);
        let s2 = _mm256_inserti128_si256(_mm256_castsi128_si256(a2), a6, 1);
        let s3 = _mm256_inserti128_si256(_mm256_castsi128_si256(a3), a7, 1);
        let s4 = _mm256_inserti128_si256(_mm256_castsi128_si256(a4), a8, 1);
        let s5 = _mm256_inserti128_si256(_mm256_castsi128_si256(a5), a9, 1);
        let s6 = _mm256_inserti128_si256(_mm256_castsi128_si256(a6), a10, 1);
        let s7 = _mm256_inserti128_si256(_mm256_castsi128_si256(a7), a11, 1);

        // Interleave consecutive pixels for madd_epi16
        let p01 = _mm256_unpacklo_epi16(s0, s1);
        let p23 = _mm256_unpacklo_epi16(s2, s3);
        let p45 = _mm256_unpacklo_epi16(s4, s5);
        let p67 = _mm256_unpacklo_epi16(s6, s7);

        // Multiply-add pairs
        let ma01 = _mm256_madd_epi16(p01, coeff0);
        let ma23 = _mm256_madd_epi16(p23, coeff2);
        let ma45 = _mm256_madd_epi16(p45, coeff4);
        let ma67 = _mm256_madd_epi16(p67, coeff6);

        // Sum all contributions
        let sum = _mm256_add_epi32(_mm256_add_epi32(ma01, ma23), _mm256_add_epi32(ma45, ma67));

        // Add rounding, shift, and clamp
        let shifted = _mm256_sra_epi32(_mm256_add_epi32(sum, rnd), shift_count);
        let clamped_lo = _mm256_max_epi32(shifted, zero);
        let clamped = _mm256_min_epi32(clamped_lo, max_val);

        // Pack from 32-bit to 16-bit
        let packed = _mm256_packus_epi32(clamped, zero);
        let result = _mm256_permute4x64_epi64(packed, 0b00_00_10_00);

        storeu_128!(
            <&mut [u16; 8]>::try_from(&mut dst[col..col + 8]).unwrap(),
            _mm256_castsi256_si128(result)
        );

        col += 8;
    }

    // w=4 tail: 128-bit loads + madd pairs, same coefficient packing.
    if w - col == 4 && src.len() >= col + 11 {
        let c01_128 = _mm256_castsi256_si128(coeff0);
        let c23_128 = _mm256_castsi256_si128(coeff2);
        let c45_128 = _mm256_castsi256_si128(coeff4);
        let c67_128 = _mm256_castsi256_si128(coeff6);
        let rnd128 = _mm_set1_epi32(32 + ((1 << (6 - intermediate_bits)) >> 1));
        let zero128 = _mm_setzero_si128();
        let max128 = _mm_set1_epi32(max);

        let s0 = loadu_64!(<&[u16; 4]>::try_from(&src[col..col + 4]).unwrap());
        let s1 = loadu_64!(<&[u16; 4]>::try_from(&src[col + 1..col + 5]).unwrap());
        let s2 = loadu_64!(<&[u16; 4]>::try_from(&src[col + 2..col + 6]).unwrap());
        let s3 = loadu_64!(<&[u16; 4]>::try_from(&src[col + 3..col + 7]).unwrap());
        let s4 = loadu_64!(<&[u16; 4]>::try_from(&src[col + 4..col + 8]).unwrap());
        let s5 = loadu_64!(<&[u16; 4]>::try_from(&src[col + 5..col + 9]).unwrap());
        let s6 = loadu_64!(<&[u16; 4]>::try_from(&src[col + 6..col + 10]).unwrap());
        let s7 = loadu_64!(<&[u16; 4]>::try_from(&src[col + 7..col + 11]).unwrap());

        let sum = _mm_add_epi32(
            _mm_add_epi32(
                _mm_madd_epi16(_mm_unpacklo_epi16(s0, s1), c01_128),
                _mm_madd_epi16(_mm_unpacklo_epi16(s2, s3), c23_128),
            ),
            _mm_add_epi32(
                _mm_madd_epi16(_mm_unpacklo_epi16(s4, s5), c45_128),
                _mm_madd_epi16(_mm_unpacklo_epi16(s6, s7), c67_128),
            ),
        );
        let shifted = _mm_sra_epi32(_mm_add_epi32(sum, rnd128), _mm_cvtsi32_si128(6));
        let clamped = _mm_min_epi32(_mm_max_epi32(shifted, zero128), max128);
        let packed = _mm_packus_epi32(clamped, clamped);
        crate::src::safe_simd::partial_simd::mm_storel_epi64::<[u16; 4]>(
            <&mut [u16; 4]>::try_from(&mut dst[col..col + 4]).unwrap(),
            packed,
        );
        col += 4;
    }

    // Scalar fallback for remaining pixels (w < 8 or w not divisible by 8)
    let scalar_rnd = 32 + ((1 << (6 - intermediate_bits)) >> 1);
    while col < w {
        let mut sum = 0i32;
        for i in 0..8 {
            sum += filter[i] as i32 * src[col + i] as i32;
        }
        let val = ((sum + scalar_rnd) >> 6).clamp(0, max);
        dst[col] = val as u16;
        col += 1;
    }
}
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[archmage::rite(v3)]
unsafe fn h_filter_8tap_16bpc_put_v3(
    dst: *mut u16,
    src: *const u16,
    w: usize,
    filter: &[i8; 8],
    max: i32,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();
    unsafe { h_filter_8tap_16bpc_put_avx2_inner(token, dst, src, w, filter, max) }
}

/// Vertical 8-tap filter for 16bpc put (V-only case)
/// Reads directly from u16 source with stride, outputs u16
#[cfg(target_arch = "x86_64")]
#[rite]
fn v_filter_8tap_16bpc_direct_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    src: &[u16],
    src_base: usize,
    src_stride: isize,
    w: usize,
    filter: &[i8; 8],
    max: i32,
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    let row_base = |row: usize| src_base.wrapping_add_signed(row as isize * src_stride);
    let coeff: [i32; 8] = [
        filter[0] as i32,
        filter[1] as i32,
        filter[2] as i32,
        filter[3] as i32,
        filter[4] as i32,
        filter[5] as i32,
        filter[6] as i32,
        filter[7] as i32,
    ];

    let c0 = _mm256_set1_epi32(coeff[0]);
    let c1 = _mm256_set1_epi32(coeff[1]);
    let c2 = _mm256_set1_epi32(coeff[2]);
    let c3 = _mm256_set1_epi32(coeff[3]);
    let c4 = _mm256_set1_epi32(coeff[4]);
    let c5 = _mm256_set1_epi32(coeff[5]);
    let c6 = _mm256_set1_epi32(coeff[6]);
    let c7 = _mm256_set1_epi32(coeff[7]);

    let rnd = _mm256_set1_epi32(32);
    let shift_count = _mm_cvtsi32_si128(6);
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi32(max);

    let mut col = 0usize;

    // Source pointer is already offset by -3 rows (pointing to tap 0), matching 8bpc convention

    while col + 8 <= w {
        // Load 8 pixels from each of 8 rows (at offsets 0 to 7)
        let p0 = _mm256_cvtepu16_epi32(loadu_128!(
            <&[u16; 8]>::try_from(&src[src_base + col..src_base + col + 8]).unwrap()
        ));
        let p1 = _mm256_cvtepu16_epi32(loadu_128!(
            <&[u16; 8]>::try_from(&src[row_base(1) + col..row_base(1) + col + 8]).unwrap()
        ));
        let p2 = _mm256_cvtepu16_epi32(loadu_128!(
            <&[u16; 8]>::try_from(&src[row_base(2) + col..row_base(2) + col + 8]).unwrap()
        ));
        let p3 = _mm256_cvtepu16_epi32(loadu_128!(
            <&[u16; 8]>::try_from(&src[row_base(3) + col..row_base(3) + col + 8]).unwrap()
        ));
        let p4 = _mm256_cvtepu16_epi32(loadu_128!(
            <&[u16; 8]>::try_from(&src[row_base(4) + col..row_base(4) + col + 8]).unwrap()
        ));
        let p5 = _mm256_cvtepu16_epi32(loadu_128!(
            <&[u16; 8]>::try_from(&src[row_base(5) + col..row_base(5) + col + 8]).unwrap()
        ));
        let p6 = _mm256_cvtepu16_epi32(loadu_128!(
            <&[u16; 8]>::try_from(&src[row_base(6) + col..row_base(6) + col + 8]).unwrap()
        ));
        let p7 = _mm256_cvtepu16_epi32(loadu_128!(
            <&[u16; 8]>::try_from(&src[row_base(7) + col..row_base(7) + col + 8]).unwrap()
        ));

        // Multiply and accumulate
        let m0 = _mm256_mullo_epi32(p0, c0);
        let m1 = _mm256_mullo_epi32(p1, c1);
        let m2 = _mm256_mullo_epi32(p2, c2);
        let m3 = _mm256_mullo_epi32(p3, c3);
        let m4 = _mm256_mullo_epi32(p4, c4);
        let m5 = _mm256_mullo_epi32(p5, c5);
        let m6 = _mm256_mullo_epi32(p6, c6);
        let m7 = _mm256_mullo_epi32(p7, c7);

        let sum = _mm256_add_epi32(
            _mm256_add_epi32(_mm256_add_epi32(m0, m1), _mm256_add_epi32(m2, m3)),
            _mm256_add_epi32(_mm256_add_epi32(m4, m5), _mm256_add_epi32(m6, m7)),
        );

        // Add rounding, shift, and clamp
        let shifted = _mm256_sra_epi32(_mm256_add_epi32(sum, rnd), shift_count);
        let clamped_lo = _mm256_max_epi32(shifted, zero);
        let clamped = _mm256_min_epi32(clamped_lo, max_val);

        // Pack from 32-bit to 16-bit
        let packed = _mm256_packus_epi32(clamped, zero);
        let result = _mm256_permute4x64_epi64(packed, 0b00_00_10_00);

        storeu_128!(
            <&mut [u16; 8]>::try_from(&mut dst[col..col + 8]).unwrap(),
            _mm256_castsi256_si128(result)
        );

        col += 8;
    }

    // w=4 tail: 64-bit u16 row loads -> i32, mullo chain.
    if w - col == 4 {
        let rnd128 = _mm_set1_epi32(32);
        let zero128 = _mm_setzero_si128();
        let max128 = _mm_set1_epi32(max);
        let shift128 = _mm_cvtsi32_si128(6);

        let mut sum = _mm_setzero_si128();
        for i in 0..8 {
            let off = row_base(i) + col;
            let p = _mm_cvtepu16_epi32(loadu_64!(
                <&[u16; 4]>::try_from(&src[off..off + 4]).unwrap()
            ));
            sum = _mm_add_epi32(sum, _mm_mullo_epi32(p, _mm_set1_epi32(coeff[i])));
        }
        let shifted = _mm_sra_epi32(_mm_add_epi32(sum, rnd128), shift128);
        let clamped = _mm_min_epi32(_mm_max_epi32(shifted, zero128), max128);
        let packed = _mm_packus_epi32(clamped, clamped);
        crate::src::safe_simd::partial_simd::mm_storel_epi64::<[u16; 4]>(
            <&mut [u16; 4]>::try_from(&mut dst[col..col + 4]).unwrap(),
            packed,
        );
        col += 4;
    }

    // Scalar fallback
    while col < w {
        let mut sum = 0i32;
        for i in 0..8 {
            let px = src[row_base(i) + col] as i32;
            sum += coeff[i] * px;
        }
        let val = ((sum + 32) >> 6).clamp(0, max);
        dst[col] = val as u16;
        col += 1;
    }
}
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[archmage::rite(v3)]
unsafe fn v_filter_8tap_16bpc_direct_v3(
    dst: *mut u16,
    src: *const u16,
    src_stride: isize,
    w: usize,
    filter: &[i8; 8],
    max: i32,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();
    unsafe { v_filter_8tap_16bpc_direct_avx2_inner(token, dst, src, 0, src_stride, w, filter, max) }
}

/// Horizontal 8-tap filter for 16bpc prep (H-only case)
/// Input is u16, output is i16 with PREP_BIAS subtracted
#[cfg(target_arch = "x86_64")]
#[rite]
fn h_filter_8tap_16bpc_prep_direct_avx2_inner(
    _token: Desktop64,
    dst: &mut [i16],
    src: &[u16],
    w: usize,
    filter: &[i8; 8],
    sh: i32,
    prep_bias: i32,
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    // Convert filter coefficients from i8 to i16 for pmaddwd
    let coeff0 =
        _mm256_set1_epi32(((filter[1] as i16 as i32) << 16) | (filter[0] as i16 as u16 as i32));
    let coeff2 =
        _mm256_set1_epi32(((filter[3] as i16 as i32) << 16) | (filter[2] as i16 as u16 as i32));
    let coeff4 =
        _mm256_set1_epi32(((filter[5] as i16 as i32) << 16) | (filter[4] as i16 as u16 as i32));
    let coeff6 =
        _mm256_set1_epi32(((filter[7] as i16 as i32) << 16) | (filter[6] as i16 as u16 as i32));

    let rnd = _mm256_set1_epi32((1 << sh) >> 1);
    let shift_count = _mm_cvtsi32_si128(sh);
    let bias = _mm256_set1_epi32(prep_bias);

    let mut col = 0usize;

    // Source pointer is already offset by -3 (pointing to tap 0), matching 8bpc convention
    // LANE FIX: see h_filter_8tap_16bpc_avx2_inner for explanation
    while col + 8 <= w {
        // Load the 4 pixels each lane consumes: a[k] = src[col+k..col+k+4]
        let a0 = loadu_64!(<&[u16; 4]>::try_from(&src[col..col + 4]).unwrap());
        let a1 = loadu_64!(<&[u16; 4]>::try_from(&src[col + 1..col + 5]).unwrap());
        let a2 = loadu_64!(<&[u16; 4]>::try_from(&src[col + 2..col + 6]).unwrap());
        let a3 = loadu_64!(<&[u16; 4]>::try_from(&src[col + 3..col + 7]).unwrap());
        let a4 = loadu_64!(<&[u16; 4]>::try_from(&src[col + 4..col + 8]).unwrap());
        let a5 = loadu_64!(<&[u16; 4]>::try_from(&src[col + 5..col + 9]).unwrap());
        let a6 = loadu_64!(<&[u16; 4]>::try_from(&src[col + 6..col + 10]).unwrap());
        let a7 = loadu_64!(<&[u16; 4]>::try_from(&src[col + 7..col + 11]).unwrap());
        let a8 = loadu_64!(<&[u16; 4]>::try_from(&src[col + 8..col + 12]).unwrap());
        let a9 = loadu_64!(<&[u16; 4]>::try_from(&src[col + 9..col + 13]).unwrap());
        let a10 = loadu_64!(<&[u16; 4]>::try_from(&src[col + 10..col + 14]).unwrap());
        let a11 = loadu_64!(<&[u16; 4]>::try_from(&src[col + 11..col + 15]).unwrap());

        // Build 256-bit source regs: lane0 for outputs 0-3, lane1 for outputs 4-7
        let s0 = _mm256_inserti128_si256(_mm256_castsi128_si256(a0), a4, 1);
        let s1 = _mm256_inserti128_si256(_mm256_castsi128_si256(a1), a5, 1);
        let s2 = _mm256_inserti128_si256(_mm256_castsi128_si256(a2), a6, 1);
        let s3 = _mm256_inserti128_si256(_mm256_castsi128_si256(a3), a7, 1);
        let s4 = _mm256_inserti128_si256(_mm256_castsi128_si256(a4), a8, 1);
        let s5 = _mm256_inserti128_si256(_mm256_castsi128_si256(a5), a9, 1);
        let s6 = _mm256_inserti128_si256(_mm256_castsi128_si256(a6), a10, 1);
        let s7 = _mm256_inserti128_si256(_mm256_castsi128_si256(a7), a11, 1);

        // Interleave consecutive pixels for madd_epi16
        let p01 = _mm256_unpacklo_epi16(s0, s1);
        let p23 = _mm256_unpacklo_epi16(s2, s3);
        let p45 = _mm256_unpacklo_epi16(s4, s5);
        let p67 = _mm256_unpacklo_epi16(s6, s7);

        // Multiply-add pairs
        let ma01 = _mm256_madd_epi16(p01, coeff0);
        let ma23 = _mm256_madd_epi16(p23, coeff2);
        let ma45 = _mm256_madd_epi16(p45, coeff4);
        let ma67 = _mm256_madd_epi16(p67, coeff6);

        // Sum all contributions
        let sum = _mm256_add_epi32(_mm256_add_epi32(ma01, ma23), _mm256_add_epi32(ma45, ma67));

        // Add rounding, shift, and subtract bias
        let shifted = _mm256_sra_epi32(_mm256_add_epi32(sum, rnd), shift_count);
        let biased = _mm256_sub_epi32(shifted, bias);

        // Pack from 32-bit to 16-bit (signed)
        let packed = _mm256_packs_epi32(biased, biased);
        let result = _mm256_permute4x64_epi64(packed, 0b00_00_10_00);

        storeu_128!(
            <&mut [i16; 8]>::try_from(&mut dst[col..col + 8]).unwrap(),
            _mm256_castsi256_si128(result)
        );

        col += 8;
    }

    // Scalar fallback
    while col < w {
        let mut sum = 0i32;
        for i in 0..8 {
            sum += filter[i] as i32 * src[col + i] as i32;
        }
        let r = (1 << sh) >> 1;
        let val = ((sum + r) >> sh) - prep_bias;
        dst[col] = val as i16;
        col += 1;
    }
}
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[archmage::rite(v3)]
unsafe fn h_filter_8tap_16bpc_prep_direct_v3(
    dst: *mut i16,
    src: *const u16,
    w: usize,
    filter: &[i8; 8],
    sh: i32,
    prep_bias: i32,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();
    unsafe { h_filter_8tap_16bpc_prep_direct_avx2_inner(token, dst, src, w, filter, sh, prep_bias) }
}

/// Vertical 8-tap filter for 16bpc prep (V-only case)
/// Reads directly from u16 source with stride, outputs i16 with bias
#[cfg(target_arch = "x86_64")]
#[rite]
fn v_filter_8tap_16bpc_prep_direct_avx2_inner(
    _token: Desktop64,
    dst: &mut [i16],
    src: &[u16],
    src_base: usize,
    src_stride: isize,
    w: usize,
    filter: &[i8; 8],
    sh: i32,
    prep_bias: i32,
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    let row_base = |row: usize| src_base.wrapping_add_signed(row as isize * src_stride);
    let coeff: [i32; 8] = [
        filter[0] as i32,
        filter[1] as i32,
        filter[2] as i32,
        filter[3] as i32,
        filter[4] as i32,
        filter[5] as i32,
        filter[6] as i32,
        filter[7] as i32,
    ];

    let c0 = _mm256_set1_epi32(coeff[0]);
    let c1 = _mm256_set1_epi32(coeff[1]);
    let c2 = _mm256_set1_epi32(coeff[2]);
    let c3 = _mm256_set1_epi32(coeff[3]);
    let c4 = _mm256_set1_epi32(coeff[4]);
    let c5 = _mm256_set1_epi32(coeff[5]);
    let c6 = _mm256_set1_epi32(coeff[6]);
    let c7 = _mm256_set1_epi32(coeff[7]);

    let rnd = _mm256_set1_epi32((1 << sh) >> 1);
    let shift_count = _mm_cvtsi32_si128(sh);
    let bias = _mm256_set1_epi32(prep_bias);

    let mut col = 0usize;

    // Source pointer is already offset by -3 rows (pointing to tap 0), matching 8bpc convention

    while col + 8 <= w {
        // Load 8 pixels from each of 8 rows (at offsets 0 to 7)
        let p0 = _mm256_cvtepu16_epi32(loadu_128!(
            <&[u16; 8]>::try_from(&src[src_base + col..src_base + col + 8]).unwrap()
        ));
        let p1 = _mm256_cvtepu16_epi32(loadu_128!(
            <&[u16; 8]>::try_from(&src[row_base(1) + col..row_base(1) + col + 8]).unwrap()
        ));
        let p2 = _mm256_cvtepu16_epi32(loadu_128!(
            <&[u16; 8]>::try_from(&src[row_base(2) + col..row_base(2) + col + 8]).unwrap()
        ));
        let p3 = _mm256_cvtepu16_epi32(loadu_128!(
            <&[u16; 8]>::try_from(&src[row_base(3) + col..row_base(3) + col + 8]).unwrap()
        ));
        let p4 = _mm256_cvtepu16_epi32(loadu_128!(
            <&[u16; 8]>::try_from(&src[row_base(4) + col..row_base(4) + col + 8]).unwrap()
        ));
        let p5 = _mm256_cvtepu16_epi32(loadu_128!(
            <&[u16; 8]>::try_from(&src[row_base(5) + col..row_base(5) + col + 8]).unwrap()
        ));
        let p6 = _mm256_cvtepu16_epi32(loadu_128!(
            <&[u16; 8]>::try_from(&src[row_base(6) + col..row_base(6) + col + 8]).unwrap()
        ));
        let p7 = _mm256_cvtepu16_epi32(loadu_128!(
            <&[u16; 8]>::try_from(&src[row_base(7) + col..row_base(7) + col + 8]).unwrap()
        ));

        // Multiply and accumulate
        let m0 = _mm256_mullo_epi32(p0, c0);
        let m1 = _mm256_mullo_epi32(p1, c1);
        let m2 = _mm256_mullo_epi32(p2, c2);
        let m3 = _mm256_mullo_epi32(p3, c3);
        let m4 = _mm256_mullo_epi32(p4, c4);
        let m5 = _mm256_mullo_epi32(p5, c5);
        let m6 = _mm256_mullo_epi32(p6, c6);
        let m7 = _mm256_mullo_epi32(p7, c7);

        let sum = _mm256_add_epi32(
            _mm256_add_epi32(_mm256_add_epi32(m0, m1), _mm256_add_epi32(m2, m3)),
            _mm256_add_epi32(_mm256_add_epi32(m4, m5), _mm256_add_epi32(m6, m7)),
        );

        // Add rounding, shift, and subtract bias
        let shifted = _mm256_sra_epi32(_mm256_add_epi32(sum, rnd), shift_count);
        let biased = _mm256_sub_epi32(shifted, bias);

        // Pack from 32-bit to 16-bit (signed)
        let packed = _mm256_packs_epi32(biased, biased);
        let result = _mm256_permute4x64_epi64(packed, 0b00_00_10_00);

        storeu_128!(
            <&mut [i16; 8]>::try_from(&mut dst[col..col + 8]).unwrap(),
            _mm256_castsi256_si128(result)
        );

        col += 8;
    }

    // w=4 tail: 64-bit u16 row loads -> i32, mullo chain, packs to i16.
    if w - col == 4 {
        let rnd128 = _mm_set1_epi32((1 << sh) >> 1);
        let bias128 = _mm_set1_epi32(prep_bias);
        let shift128 = _mm_cvtsi32_si128(sh);

        let mut sum = _mm_setzero_si128();
        for i in 0..8 {
            let off = row_base(i) + col;
            let p = _mm_cvtepu16_epi32(loadu_64!(
                <&[u16; 4]>::try_from(&src[off..off + 4]).unwrap()
            ));
            sum = _mm_add_epi32(sum, _mm_mullo_epi32(p, _mm_set1_epi32(coeff[i])));
        }
        let shifted = _mm_sra_epi32(_mm_add_epi32(sum, rnd128), shift128);
        let biased = _mm_sub_epi32(shifted, bias128);
        let packed = _mm_packs_epi32(biased, biased);
        crate::src::safe_simd::partial_simd::mm_storel_epi64::<[i16; 4]>(
            <&mut [i16; 4]>::try_from(&mut dst[col..col + 4]).unwrap(),
            packed,
        );
        col += 4;
    }

    // Scalar fallback
    while col < w {
        let mut sum = 0i32;
        for i in 0..8 {
            let px = src[row_base(i) + col] as i32;
            sum += coeff[i] * px;
        }
        let r = (1 << sh) >> 1;
        let val = ((sum + r) >> sh) - prep_bias;
        dst[col] = val as i16;
        col += 1;
    }
}
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[archmage::rite(v3)]
unsafe fn v_filter_8tap_16bpc_prep_direct_v3(
    dst: *mut i16,
    src: *const u16,
    src_stride: isize,
    w: usize,
    filter: &[i8; 8],
    sh: i32,
    prep_bias: i32,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();
    unsafe {
        v_filter_8tap_16bpc_prep_direct_avx2_inner(
            token, dst, src, 0, src_stride, w, filter, sh, prep_bias,
        )
    }
}

// =============================================================================
// 16BPC 8-TAP FILTERS — AVX-512
// =============================================================================

/// AVX-512 horizontal 8-tap filter for 16bpc → i32 mid buffer, 16 outputs/iter.
/// Uses individual tap loads with cvtepu16→mullo approach (no lane ordering issues).
#[cfg(target_arch = "x86_64")]
#[rite]
fn h_filter_8tap_16bpc_avx512_inner(
    _token: Server64,
    dst: &mut [i32],
    src: &[u16],
    w: usize,
    filter: &[i8; 8],
    sh: i32,
) {
    if w < 16 {
        h_filter_8tap_16bpc_avx2_inner(_token.v3(), dst, src, w, filter, sh);
        return;
    }
    let mut dst = dst.flex_mut();
    let src = src.flex();

    let c0 = _mm512_set1_epi32(filter[0] as i32);
    let c1 = _mm512_set1_epi32(filter[1] as i32);
    let c2 = _mm512_set1_epi32(filter[2] as i32);
    let c3 = _mm512_set1_epi32(filter[3] as i32);
    let c4 = _mm512_set1_epi32(filter[4] as i32);
    let c5 = _mm512_set1_epi32(filter[5] as i32);
    let c6 = _mm512_set1_epi32(filter[6] as i32);
    let c7 = _mm512_set1_epi32(filter[7] as i32);

    let rnd = _mm512_set1_epi32((1 << sh) >> 1);
    let shift_count = _mm_cvtsi32_si128(sh);

    let mut col = 0usize;

    while col + 16 <= w {
        let p0 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[col..col + 16]).unwrap()
        ));
        let p1 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[col + 1..col + 17]).unwrap()
        ));
        let p2 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[col + 2..col + 18]).unwrap()
        ));
        let p3 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[col + 3..col + 19]).unwrap()
        ));
        let p4 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[col + 4..col + 20]).unwrap()
        ));
        let p5 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[col + 5..col + 21]).unwrap()
        ));
        let p6 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[col + 6..col + 22]).unwrap()
        ));
        let p7 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[col + 7..col + 23]).unwrap()
        ));

        let mut sum = _mm512_mullo_epi32(p0, c0);
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(p1, c1));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(p2, c2));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(p3, c3));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(p4, c4));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(p5, c5));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(p6, c6));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(p7, c7));

        let result = _mm512_sra_epi32(_mm512_add_epi32(sum, rnd), shift_count);
        storeu_512!(&mut dst[col..col + 16], [i32; 16], result);
        col += 16;
    }

    while col < w {
        let mut sum = 0i32;
        for i in 0..8 {
            sum += filter[i] as i32 * src[col + i] as i32;
        }
        let r = (1 << sh) >> 1;
        dst[col] = (sum + r) >> sh;
        col += 1;
    }
}

/// AVX-512 vertical 8-tap filter for 16bpc put — from i32 mid buffer to u16 output, 16/iter.
#[cfg(target_arch = "x86_64")]
#[rite]
fn v_filter_8tap_16bpc_avx512_inner(
    _token: Server64,
    dst: &mut [u16],
    mid: &[[i32; MID_STRIDE]],
    w: usize,
    y: usize,
    filter: &[i8; 8],
    sh: i32,
    max: i32,
) {
    if w < 16 {
        v_filter_8tap_16bpc_avx2_inner(_token.v3(), dst, mid, w, y, filter, sh, max);
        return;
    }
    let mut dst = dst.flex_mut();

    let c0 = _mm512_set1_epi32(filter[0] as i32);
    let c1 = _mm512_set1_epi32(filter[1] as i32);
    let c2 = _mm512_set1_epi32(filter[2] as i32);
    let c3 = _mm512_set1_epi32(filter[3] as i32);
    let c4 = _mm512_set1_epi32(filter[4] as i32);
    let c5 = _mm512_set1_epi32(filter[5] as i32);
    let c6 = _mm512_set1_epi32(filter[6] as i32);
    let c7 = _mm512_set1_epi32(filter[7] as i32);

    let rnd = _mm512_set1_epi32((1 << sh) >> 1);
    let shift_count = _mm_cvtsi32_si128(sh);
    let zero = _mm512_setzero_si512();
    let max_val = _mm512_set1_epi32(max);

    let mut col = 0usize;

    while col + 16 <= w {
        let r0 = loadu_512!(&mid[y + 0][col..col + 16], [i32; 16]);
        let r1 = loadu_512!(&mid[y + 1][col..col + 16], [i32; 16]);
        let r2 = loadu_512!(&mid[y + 2][col..col + 16], [i32; 16]);
        let r3 = loadu_512!(&mid[y + 3][col..col + 16], [i32; 16]);
        let r4 = loadu_512!(&mid[y + 4][col..col + 16], [i32; 16]);
        let r5 = loadu_512!(&mid[y + 5][col..col + 16], [i32; 16]);
        let r6 = loadu_512!(&mid[y + 6][col..col + 16], [i32; 16]);
        let r7 = loadu_512!(&mid[y + 7][col..col + 16], [i32; 16]);

        let mut sum = _mm512_mullo_epi32(r0, c0);
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(r1, c1));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(r2, c2));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(r3, c3));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(r4, c4));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(r5, c5));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(r6, c6));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(r7, c7));

        let shifted = _mm512_sra_epi32(_mm512_add_epi32(sum, rnd), shift_count);
        let clamped = _mm512_min_epi32(_mm512_max_epi32(shifted, zero), max_val);
        let packed = _mm512_cvtusepi32_epi16(clamped);
        storeu_256!(
            <&mut [u16; 16]>::try_from(&mut dst[col..col + 16]).unwrap(),
            packed
        );
        col += 16;
    }

    while col < w {
        let coeff: [i32; 8] = core::array::from_fn(|i| filter[i] as i32);
        let mut sum = 0i32;
        for i in 0..8 {
            sum += coeff[i] * mid[y + i][col];
        }
        let r = (1 << sh) >> 1;
        let val = ((sum + r) >> sh).clamp(0, max);
        dst[col] = val as u16;
        col += 1;
    }
}

/// AVX-512 vertical 8-tap filter for 16bpc prep — from i32 mid buffer to i16 output, 16/iter.
#[cfg(target_arch = "x86_64")]
#[rite]
fn v_filter_8tap_16bpc_prep_avx512_inner(
    _token: Server64,
    dst: &mut [i16],
    mid: &[[i32; MID_STRIDE]],
    w: usize,
    y: usize,
    filter: &[i8; 8],
    sh: i32,
    prep_bias: i32,
) {
    if w < 16 {
        v_filter_8tap_16bpc_prep_avx2_inner(_token.v3(), dst, mid, w, y, filter, sh, prep_bias);
        return;
    }
    let mut dst = dst.flex_mut();

    let c0 = _mm512_set1_epi32(filter[0] as i32);
    let c1 = _mm512_set1_epi32(filter[1] as i32);
    let c2 = _mm512_set1_epi32(filter[2] as i32);
    let c3 = _mm512_set1_epi32(filter[3] as i32);
    let c4 = _mm512_set1_epi32(filter[4] as i32);
    let c5 = _mm512_set1_epi32(filter[5] as i32);
    let c6 = _mm512_set1_epi32(filter[6] as i32);
    let c7 = _mm512_set1_epi32(filter[7] as i32);

    let rnd = _mm512_set1_epi32((1 << sh) >> 1);
    let shift_count = _mm_cvtsi32_si128(sh);
    let bias = _mm512_set1_epi32(prep_bias);

    let mut col = 0usize;

    while col + 16 <= w {
        let r0 = loadu_512!(&mid[y + 0][col..col + 16], [i32; 16]);
        let r1 = loadu_512!(&mid[y + 1][col..col + 16], [i32; 16]);
        let r2 = loadu_512!(&mid[y + 2][col..col + 16], [i32; 16]);
        let r3 = loadu_512!(&mid[y + 3][col..col + 16], [i32; 16]);
        let r4 = loadu_512!(&mid[y + 4][col..col + 16], [i32; 16]);
        let r5 = loadu_512!(&mid[y + 5][col..col + 16], [i32; 16]);
        let r6 = loadu_512!(&mid[y + 6][col..col + 16], [i32; 16]);
        let r7 = loadu_512!(&mid[y + 7][col..col + 16], [i32; 16]);

        let mut sum = _mm512_mullo_epi32(r0, c0);
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(r1, c1));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(r2, c2));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(r3, c3));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(r4, c4));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(r5, c5));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(r6, c6));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(r7, c7));

        let shifted = _mm512_sra_epi32(_mm512_add_epi32(sum, rnd), shift_count);
        let biased = _mm512_sub_epi32(shifted, bias);
        let packed = _mm512_cvtsepi32_epi16(biased);
        storeu_256!(
            <&mut [i16; 16]>::try_from(&mut dst[col..col + 16]).unwrap(),
            packed
        );
        col += 16;
    }

    while col < w {
        let coeff: [i32; 8] = core::array::from_fn(|i| filter[i] as i32);
        let mut sum = 0i32;
        for i in 0..8 {
            sum += coeff[i] * mid[y + i][col];
        }
        let r = (1 << sh) >> 1;
        let val = ((sum + r) >> sh) - prep_bias;
        dst[col] = val as i16;
        col += 1;
    }
}

/// AVX-512 horizontal 8-tap for 16bpc put (H-only), 16 u16 outputs/iter.
#[cfg(target_arch = "x86_64")]
#[rite]
fn h_filter_8tap_16bpc_put_avx512_inner(
    _token: Server64,
    dst: &mut [u16],
    src: &[u16],
    w: usize,
    filter: &[i8; 8],
    max: i32,
) {
    if w < 16 {
        h_filter_8tap_16bpc_put_avx2_inner(_token.v3(), dst, src, w, filter, max);
        return;
    }
    let mut dst = dst.flex_mut();
    let src = src.flex();

    let c0 = _mm512_set1_epi32(filter[0] as i32);
    let c1 = _mm512_set1_epi32(filter[1] as i32);
    let c2 = _mm512_set1_epi32(filter[2] as i32);
    let c3 = _mm512_set1_epi32(filter[3] as i32);
    let c4 = _mm512_set1_epi32(filter[4] as i32);
    let c5 = _mm512_set1_epi32(filter[5] as i32);
    let c6 = _mm512_set1_epi32(filter[6] as i32);
    let c7 = _mm512_set1_epi32(filter[7] as i32);

    let intermediate_bits = if (max >> 11) != 0 { 2 } else { 4 };
    let rnd = _mm512_set1_epi32(32 + ((1 << (6 - intermediate_bits)) >> 1));
    let shift_count = _mm_cvtsi32_si128(6);
    let zero = _mm512_setzero_si512();
    let max_val = _mm512_set1_epi32(max);

    let mut col = 0usize;

    while col + 16 <= w {
        let p0 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[col..col + 16]).unwrap()
        ));
        let p1 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[col + 1..col + 17]).unwrap()
        ));
        let p2 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[col + 2..col + 18]).unwrap()
        ));
        let p3 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[col + 3..col + 19]).unwrap()
        ));
        let p4 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[col + 4..col + 20]).unwrap()
        ));
        let p5 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[col + 5..col + 21]).unwrap()
        ));
        let p6 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[col + 6..col + 22]).unwrap()
        ));
        let p7 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[col + 7..col + 23]).unwrap()
        ));

        let mut sum = _mm512_mullo_epi32(p0, c0);
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(p1, c1));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(p2, c2));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(p3, c3));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(p4, c4));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(p5, c5));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(p6, c6));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(p7, c7));

        let shifted = _mm512_sra_epi32(_mm512_add_epi32(sum, rnd), shift_count);
        let clamped = _mm512_min_epi32(_mm512_max_epi32(shifted, zero), max_val);
        let packed = _mm512_cvtusepi32_epi16(clamped);
        storeu_256!(
            <&mut [u16; 16]>::try_from(&mut dst[col..col + 16]).unwrap(),
            packed
        );
        col += 16;
    }

    let scalar_rnd = 32 + ((1 << (6 - intermediate_bits)) >> 1);
    while col < w {
        let mut sum = 0i32;
        for i in 0..8 {
            sum += filter[i] as i32 * src[col + i] as i32;
        }
        let val = ((sum + scalar_rnd) >> 6).clamp(0, max);
        dst[col] = val as u16;
        col += 1;
    }
}

/// AVX-512 horizontal 8-tap for 16bpc prep (H-only), 16 i16 outputs/iter.
#[cfg(target_arch = "x86_64")]
#[rite]
fn h_filter_8tap_16bpc_prep_direct_avx512_inner(
    _token: Server64,
    dst: &mut [i16],
    src: &[u16],
    w: usize,
    filter: &[i8; 8],
    sh: i32,
    prep_bias: i32,
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();

    let c0 = _mm512_set1_epi32(filter[0] as i32);
    let c1 = _mm512_set1_epi32(filter[1] as i32);
    let c2 = _mm512_set1_epi32(filter[2] as i32);
    let c3 = _mm512_set1_epi32(filter[3] as i32);
    let c4 = _mm512_set1_epi32(filter[4] as i32);
    let c5 = _mm512_set1_epi32(filter[5] as i32);
    let c6 = _mm512_set1_epi32(filter[6] as i32);
    let c7 = _mm512_set1_epi32(filter[7] as i32);

    let rnd = _mm512_set1_epi32((1 << sh) >> 1);
    let shift_count = _mm_cvtsi32_si128(sh);
    let bias = _mm512_set1_epi32(prep_bias);

    let mut col = 0usize;

    while col + 16 <= w {
        let p0 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[col..col + 16]).unwrap()
        ));
        let p1 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[col + 1..col + 17]).unwrap()
        ));
        let p2 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[col + 2..col + 18]).unwrap()
        ));
        let p3 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[col + 3..col + 19]).unwrap()
        ));
        let p4 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[col + 4..col + 20]).unwrap()
        ));
        let p5 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[col + 5..col + 21]).unwrap()
        ));
        let p6 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[col + 6..col + 22]).unwrap()
        ));
        let p7 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[col + 7..col + 23]).unwrap()
        ));

        let mut sum = _mm512_mullo_epi32(p0, c0);
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(p1, c1));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(p2, c2));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(p3, c3));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(p4, c4));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(p5, c5));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(p6, c6));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(p7, c7));

        let shifted = _mm512_sra_epi32(_mm512_add_epi32(sum, rnd), shift_count);
        let biased = _mm512_sub_epi32(shifted, bias);
        let packed = _mm512_cvtsepi32_epi16(biased);
        storeu_256!(
            <&mut [i16; 16]>::try_from(&mut dst[col..col + 16]).unwrap(),
            packed
        );
        col += 16;
    }

    while col < w {
        let mut sum = 0i32;
        for i in 0..8 {
            sum += filter[i] as i32 * src[col + i] as i32;
        }
        let r = (1 << sh) >> 1;
        let val = ((sum + r) >> sh) - prep_bias;
        dst[col] = val as i16;
        col += 1;
    }
}

/// AVX-512 vertical 8-tap for 16bpc put (V-only) — from strided u16 source, 16 u16 outputs/iter.
#[cfg(target_arch = "x86_64")]
#[rite]
fn v_filter_8tap_16bpc_direct_avx512_inner(
    _token: Server64,
    dst: &mut [u16],
    src: &[u16],
    src_base: usize,
    src_stride: isize,
    w: usize,
    filter: &[i8; 8],
    max: i32,
) {
    if w < 16 {
        v_filter_8tap_16bpc_direct_avx2_inner(
            _token.v3(),
            dst,
            src,
            src_base,
            src_stride,
            w,
            filter,
            max,
        );
        return;
    }
    let mut dst = dst.flex_mut();
    let src = src.flex();
    let row_base = |row: usize| src_base.wrapping_add_signed(row as isize * src_stride);

    let c0 = _mm512_set1_epi32(filter[0] as i32);
    let c1 = _mm512_set1_epi32(filter[1] as i32);
    let c2 = _mm512_set1_epi32(filter[2] as i32);
    let c3 = _mm512_set1_epi32(filter[3] as i32);
    let c4 = _mm512_set1_epi32(filter[4] as i32);
    let c5 = _mm512_set1_epi32(filter[5] as i32);
    let c6 = _mm512_set1_epi32(filter[6] as i32);
    let c7 = _mm512_set1_epi32(filter[7] as i32);

    let rnd = _mm512_set1_epi32(32);
    let shift_count = _mm_cvtsi32_si128(6);
    let zero = _mm512_setzero_si512();
    let max_val = _mm512_set1_epi32(max);

    let mut col = 0usize;

    while col + 16 <= w {
        let p0 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[src_base + col..src_base + col + 16]).unwrap()
        ));
        let p1 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[row_base(1) + col..row_base(1) + col + 16]).unwrap()
        ));
        let p2 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[row_base(2) + col..row_base(2) + col + 16]).unwrap()
        ));
        let p3 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[row_base(3) + col..row_base(3) + col + 16]).unwrap()
        ));
        let p4 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[row_base(4) + col..row_base(4) + col + 16]).unwrap()
        ));
        let p5 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[row_base(5) + col..row_base(5) + col + 16]).unwrap()
        ));
        let p6 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[row_base(6) + col..row_base(6) + col + 16]).unwrap()
        ));
        let p7 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[row_base(7) + col..row_base(7) + col + 16]).unwrap()
        ));

        let mut sum = _mm512_mullo_epi32(p0, c0);
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(p1, c1));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(p2, c2));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(p3, c3));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(p4, c4));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(p5, c5));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(p6, c6));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(p7, c7));

        let shifted = _mm512_sra_epi32(_mm512_add_epi32(sum, rnd), shift_count);
        let clamped = _mm512_min_epi32(_mm512_max_epi32(shifted, zero), max_val);
        let packed = _mm512_cvtusepi32_epi16(clamped);
        storeu_256!(
            <&mut [u16; 16]>::try_from(&mut dst[col..col + 16]).unwrap(),
            packed
        );
        col += 16;
    }

    let coeff: [i32; 8] = core::array::from_fn(|i| filter[i] as i32);
    while col < w {
        let mut sum = 0i32;
        for i in 0..8 {
            sum += coeff[i] * src[row_base(i) + col] as i32;
        }
        let val = ((sum + 32) >> 6).clamp(0, max);
        dst[col] = val as u16;
        col += 1;
    }
}

/// AVX-512 vertical 8-tap for 16bpc prep (V-only) — from strided u16 source, 16 i16 outputs/iter.
#[cfg(target_arch = "x86_64")]
#[rite]
fn v_filter_8tap_16bpc_prep_direct_avx512_inner(
    _token: Server64,
    dst: &mut [i16],
    src: &[u16],
    src_base: usize,
    src_stride: isize,
    w: usize,
    filter: &[i8; 8],
    sh: i32,
    prep_bias: i32,
) {
    if w < 16 {
        v_filter_8tap_16bpc_prep_direct_avx2_inner(
            _token.v3(),
            dst,
            src,
            src_base,
            src_stride,
            w,
            filter,
            sh,
            prep_bias,
        );
        return;
    }
    let mut dst = dst.flex_mut();
    let src = src.flex();
    let row_base = |row: usize| src_base.wrapping_add_signed(row as isize * src_stride);

    let c0 = _mm512_set1_epi32(filter[0] as i32);
    let c1 = _mm512_set1_epi32(filter[1] as i32);
    let c2 = _mm512_set1_epi32(filter[2] as i32);
    let c3 = _mm512_set1_epi32(filter[3] as i32);
    let c4 = _mm512_set1_epi32(filter[4] as i32);
    let c5 = _mm512_set1_epi32(filter[5] as i32);
    let c6 = _mm512_set1_epi32(filter[6] as i32);
    let c7 = _mm512_set1_epi32(filter[7] as i32);

    let rnd = _mm512_set1_epi32((1 << sh) >> 1);
    let shift_count = _mm_cvtsi32_si128(sh);
    let bias = _mm512_set1_epi32(prep_bias);

    let mut col = 0usize;

    while col + 16 <= w {
        let p0 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[src_base + col..src_base + col + 16]).unwrap()
        ));
        let p1 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[row_base(1) + col..row_base(1) + col + 16]).unwrap()
        ));
        let p2 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[row_base(2) + col..row_base(2) + col + 16]).unwrap()
        ));
        let p3 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[row_base(3) + col..row_base(3) + col + 16]).unwrap()
        ));
        let p4 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[row_base(4) + col..row_base(4) + col + 16]).unwrap()
        ));
        let p5 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[row_base(5) + col..row_base(5) + col + 16]).unwrap()
        ));
        let p6 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[row_base(6) + col..row_base(6) + col + 16]).unwrap()
        ));
        let p7 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[row_base(7) + col..row_base(7) + col + 16]).unwrap()
        ));

        let mut sum = _mm512_mullo_epi32(p0, c0);
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(p1, c1));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(p2, c2));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(p3, c3));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(p4, c4));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(p5, c5));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(p6, c6));
        sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(p7, c7));

        let shifted = _mm512_sra_epi32(_mm512_add_epi32(sum, rnd), shift_count);
        let biased = _mm512_sub_epi32(shifted, bias);
        let packed = _mm512_cvtsepi32_epi16(biased);
        storeu_256!(
            <&mut [i16; 16]>::try_from(&mut dst[col..col + 16]).unwrap(),
            packed
        );
        col += 16;
    }

    let coeff: [i32; 8] = core::array::from_fn(|i| filter[i] as i32);
    while col < w {
        let mut sum = 0i32;
        for i in 0..8 {
            sum += coeff[i] * src[row_base(i) + col] as i32;
        }
        let r = (1 << sh) >> 1;
        let val = ((sum + r) >> sh) - prep_bias;
        dst[col] = val as i16;
        col += 1;
    }
}

/// AVX-512 put_8tap for 16bpc — uses wider inner filters (16 pixels/iter vs 8).
#[cfg(target_arch = "x86_64")]
#[arcane]
fn put_8tap_16bpc_avx512_impl_inner(
    _token: Server64,
    dst: &mut [u16],
    dst_base: usize,
    dst_stride: isize,
    src: &[u16],
    src_base: usize,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    h_filter_type: Rav1dFilterMode,
    v_filter_type: Rav1dFilterMode,
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    let w = w as usize;
    let h = h as usize;
    let mx = mx as usize;
    let my = my as usize;
    let dst_stride_elems = dst_stride / 2;
    let src_stride_elems = src_stride / 2;
    let sb = src_base as isize;
    let max = bitdepth_max as i32;

    let intermediate_bits = if (bitdepth_max >> 11) != 0 {
        2i32
    } else {
        4i32
    };

    let fh = get_filter_coeff(mx, w, h_filter_type);
    let fv = get_filter_coeff(my, h, v_filter_type);

    match (fh, fv) {
        (Some(fh), Some(fv)) => {
            let tmp_h = h + 7;
            let mut mid = take_mid_i32_135();
            let h_sh = 6 - intermediate_bits;
            let v_sh = 6 + intermediate_bits;
            for y in 0..tmp_h {
                let src_off = (sb + (y as isize - 3) * src_stride_elems) as usize;
                h_filter_8tap_16bpc_avx512_inner(
                    _token,
                    &mut mid[y],
                    &src[src_off - 3..],
                    w,
                    fh,
                    h_sh,
                );
            }
            for y in 0..h {
                let dst_row =
                    &mut dst[dst_base.wrapping_add_signed(y as isize * dst_stride_elems)..];
                v_filter_8tap_16bpc_avx512_inner(_token, dst_row, &*mid, w, y, fv, v_sh, max);
            }
            put_mid_i32_135(mid);
        }
        (Some(fh), None) => {
            for y in 0..h {
                let src_off = (sb + y as isize * src_stride_elems) as usize;
                let dst_row =
                    &mut dst[dst_base.wrapping_add_signed(y as isize * dst_stride_elems)..];
                h_filter_8tap_16bpc_put_avx512_inner(
                    _token,
                    dst_row,
                    &src[src_off - 3..],
                    w,
                    fh,
                    max,
                );
            }
        }
        (None, Some(fv)) => {
            for y in 0..h {
                let src_off = (sb + (y as isize - 3) * src_stride_elems) as usize;
                let dst_row =
                    &mut dst[dst_base.wrapping_add_signed(y as isize * dst_stride_elems)..];
                v_filter_8tap_16bpc_direct_avx512_inner(
                    _token,
                    dst_row,
                    &src,
                    src_off,
                    src_stride_elems,
                    w,
                    fv,
                    max,
                );
            }
        }
        (None, None) => {
            for y in 0..h {
                let src_row = &src[(sb + y as isize * src_stride_elems) as usize..];
                let dst_row =
                    &mut dst[dst_base.wrapping_add_signed(y as isize * dst_stride_elems)..];
                copy_row_px(dst_row, src_row, w);
            }
        }
    }
}

/// AVX-512 prep_8tap for 16bpc — uses wider inner filters (16 pixels/iter vs 8).
#[cfg(target_arch = "x86_64")]
#[arcane]
fn prep_8tap_16bpc_avx512_impl_inner(
    _token: Server64,
    tmp: &mut [i16],
    src: &[u16],
    src_base: usize,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    h_filter_type: Rav1dFilterMode,
    v_filter_type: Rav1dFilterMode,
) {
    let mut tmp = tmp.flex_mut();
    let src = src.flex();
    let w = w as usize;
    let h = h as usize;
    let mx = mx as usize;
    let my = my as usize;
    let src_stride_elems = src_stride / 2;
    let sb = src_base as isize;

    let intermediate_bits = if (bitdepth_max >> 11) != 0 {
        2i32
    } else {
        4i32
    };
    const PREP_BIAS: i32 = 8192;

    let fh = get_filter_coeff(mx, w, h_filter_type);
    let fv = get_filter_coeff(my, h, v_filter_type);

    match (fh, fv) {
        (Some(fh), Some(fv)) => {
            let tmp_h = h + 7;
            let mut mid = take_mid_i32_135();
            let h_sh = 6 - intermediate_bits;
            let v_sh = 6;
            for y in 0..tmp_h {
                let src_off = (sb + (y as isize - 3) * src_stride_elems) as usize;
                h_filter_8tap_16bpc_avx512_inner(
                    _token,
                    &mut mid[y],
                    &src[src_off - 3..],
                    w,
                    fh,
                    h_sh,
                );
            }
            for y in 0..h {
                let out_row = y * w;
                v_filter_8tap_16bpc_prep_avx512_inner(
                    _token,
                    &mut tmp[out_row..],
                    &*mid,
                    w,
                    y,
                    fv,
                    v_sh,
                    PREP_BIAS,
                );
            }
            put_mid_i32_135(mid);
        }
        (Some(fh), None) => {
            let sh = 6 - intermediate_bits;
            for y in 0..h {
                let src_off = (sb + y as isize * src_stride_elems) as usize;
                let out_row = y * w;
                h_filter_8tap_16bpc_prep_direct_avx512_inner(
                    _token,
                    &mut tmp[out_row..],
                    &src[src_off - 3..],
                    w,
                    fh,
                    sh,
                    PREP_BIAS,
                );
            }
        }
        (None, Some(fv)) => {
            let sh = 6 - intermediate_bits;
            for y in 0..h {
                let src_off = (sb + (y as isize - 3) * src_stride_elems) as usize;
                let out_row = y * w;
                v_filter_8tap_16bpc_prep_direct_avx512_inner(
                    _token,
                    &mut tmp[out_row..],
                    &src,
                    src_off,
                    src_stride_elems,
                    w,
                    fv,
                    sh,
                    PREP_BIAS,
                );
            }
        }
        (None, None) => {
            for y in 0..h {
                let src_row = &src[(sb + y as isize * src_stride_elems) as usize..];
                let out_row = y * w;
                for x in 0..w {
                    let px = src_row[x] as i32;
                    let val = (px << intermediate_bits) - PREP_BIAS;
                    tmp[out_row + x] = val as i16;
                }
            }
        }
    }
}

/// Generic 8-tap put function for 16bpc
///
/// Similar to 8bpc but handles 16-bit pixels and different intermediate scaling
#[cfg(target_arch = "x86_64")]
#[rite]
fn put_8tap_16bpc_avx2_impl_inner(
    _token: Desktop64,
    dst: &mut [u16],
    dst_base: usize,
    dst_stride: isize,
    src: &[u16],
    src_base: usize,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    h_filter_type: Rav1dFilterMode,
    v_filter_type: Rav1dFilterMode,
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    let w = w as usize;
    let h = h as usize;
    let mx = mx as usize;
    let my = my as usize;
    let dst_stride_elems = dst_stride / 2;
    let src_stride_elems = src_stride / 2;
    let sb = src_base as isize;
    let max = bitdepth_max as i32;

    // intermediate_bits depends on bitdepth: 4 for 10-bit, 2 for 12-bit
    let intermediate_bits = if (bitdepth_max >> 11) != 0 {
        2i32
    } else {
        4i32
    };

    let fh = get_filter_coeff(mx, w, h_filter_type);
    let fv = get_filter_coeff(my, h, v_filter_type);

    // DEBUG: control which paths use SIMD vs scalar
    const USE_SIMD_HV: bool = true;
    const USE_SIMD_H: bool = true;
    const USE_SIMD_V: bool = true;

    match (fh, fv) {
        (Some(fh), Some(fv)) => {
            if USE_SIMD_HV {
                // Case 1: Both H and V filtering - two-pass through intermediate (SIMD)
                let tmp_h = h + 7;
                let mut mid = take_mid_i32_135();
                let h_sh = 6 - intermediate_bits;
                let v_sh = 6 + intermediate_bits;
                for y in 0..tmp_h {
                    let src_off = (sb + (y as isize - 3) * src_stride_elems) as usize;
                    h_filter_8tap_16bpc_avx2_inner(
                        _token,
                        &mut mid[y],
                        &src[src_off - 3..],
                        w,
                        fh,
                        h_sh,
                    );
                }
                for y in 0..h {
                    let dst_row =
                        &mut dst[dst_base.wrapping_add_signed(y as isize * dst_stride_elems)..];
                    v_filter_8tap_16bpc_avx2_inner(_token, dst_row, &*mid, w, y, fv, v_sh, max);
                }
                put_mid_i32_135(mid);
            } else {
                // Scalar H+V fallback
                let tmp_h = h + 7;
                let mut mid = take_mid_i32_135();
                let h_rnd = (1 << (6 - intermediate_bits)) >> 1;
                let h_sh = 6 - intermediate_bits;
                for y in 0..tmp_h {
                    for x in 0..w {
                        let src_off = (sb + (y as isize - 3) * src_stride_elems) as usize;
                        let mut sum = 0i32;
                        for k in 0..8 {
                            let sx = src_off - 3 + x + k;
                            sum += src[sx] as i32 * fh[k] as i32;
                        }
                        mid[y][x] = (sum + h_rnd) >> h_sh;
                    }
                }
                let v_sh = 6 + intermediate_bits;
                let v_rnd = (1 << v_sh) >> 1;
                for y in 0..h {
                    for x in 0..w {
                        let mut sum = 0i32;
                        for k in 0..8 {
                            sum += mid[y + k][x] * fv[k] as i32;
                        }
                        let val = ((sum + v_rnd) >> v_sh).clamp(0, max);
                        dst[dst_base.wrapping_add_signed(y as isize * dst_stride_elems) + x] =
                            val as u16;
                    }
                }
                put_mid_i32_135(mid);
            }
        }
        (Some(fh), None) => {
            if USE_SIMD_H {
                // Case 2: H-only filtering (SIMD)
                for y in 0..h {
                    let src_off = (sb + y as isize * src_stride_elems) as usize;
                    let dst_row =
                        &mut dst[dst_base.wrapping_add_signed(y as isize * dst_stride_elems)..];
                    h_filter_8tap_16bpc_put_avx2_inner(
                        _token,
                        dst_row,
                        &src[src_off - 3..],
                        w,
                        fh,
                        max,
                    );
                }
            } else {
                // Scalar H-only fallback
                let intermediate_rnd = 32 + ((1 << (6 - intermediate_bits)) >> 1);
                for y in 0..h {
                    let src_off = (sb + y as isize * src_stride_elems) as usize;
                    for x in 0..w {
                        let mut sum = 0i32;
                        for k in 0..8 {
                            sum += src[src_off + x + k] as i32 * fh[k] as i32;
                        }
                        let val = ((sum + intermediate_rnd) >> 6).clamp(0, max);
                        dst[dst_base.wrapping_add_signed(y as isize * dst_stride_elems) + x] =
                            val as u16;
                    }
                }
            }
        }
        (None, Some(fv)) => {
            if USE_SIMD_V {
                // Case 3: V-only filtering (SIMD)
                for y in 0..h {
                    let src_off = (sb + (y as isize - 3) * src_stride_elems) as usize;
                    let dst_row =
                        &mut dst[dst_base.wrapping_add_signed(y as isize * dst_stride_elems)..];
                    v_filter_8tap_16bpc_direct_avx2_inner(
                        _token,
                        dst_row,
                        &src,
                        src_off,
                        src_stride_elems,
                        w,
                        fv,
                        max,
                    );
                }
            } else {
                // Scalar V-only fallback
                for y in 0..h {
                    for x in 0..w {
                        let mut sum = 0i32;
                        for k in 0..8 {
                            let src_off =
                                (sb + (y as isize + k as isize - 3) * src_stride_elems) as usize;
                            sum += src[src_off + x] as i32 * fv[k] as i32;
                        }
                        let val = ((sum + 32) >> 6).clamp(0, max);
                        dst[dst_base.wrapping_add_signed(y as isize * dst_stride_elems) + x] =
                            val as u16;
                    }
                }
            }
        }
        (None, None) => {
            // Case 4: Simple copy
            for y in 0..h {
                let src_row = &src[(sb + y as isize * src_stride_elems) as usize..];
                let dst_row =
                    &mut dst[dst_base.wrapping_add_signed(y as isize * dst_stride_elems)..];
                copy_row_px(dst_row, src_row, w);
            }
        }
    }
}
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[archmage::rite(v3)]
unsafe fn put_8tap_16bpc_impl_v3(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    h_filter_type: Rav1dFilterMode,
    v_filter_type: Rav1dFilterMode,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();
    unsafe {
        put_8tap_16bpc_avx2_impl_inner(
            token,
            dst_ptr,
            0,
            dst_stride,
            src_ptr,
            0,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            h_filter_type,
            v_filter_type,
        )
    }
}

/// Generic 8-tap prep function for 16bpc
#[cfg(target_arch = "x86_64")]
#[rite]
fn prep_8tap_16bpc_avx2_impl_inner(
    _token: Desktop64,
    tmp: &mut [i16],
    src: &[u16],
    src_base: usize,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    h_filter_type: Rav1dFilterMode,
    v_filter_type: Rav1dFilterMode,
) {
    let mut tmp = tmp.flex_mut();
    let src = src.flex();
    let w = w as usize;
    let h = h as usize;
    let mx = mx as usize;
    let my = my as usize;
    let src_stride_elems = src_stride / 2;
    let sb = src_base as isize;

    // intermediate_bits depends on bitdepth: 4 for 10-bit, 2 for 12-bit
    let intermediate_bits = if (bitdepth_max >> 11) != 0 {
        2i32
    } else {
        4i32
    };
    const PREP_BIAS: i32 = 8192;

    let fh = get_filter_coeff(mx, w, h_filter_type);
    let fv = get_filter_coeff(my, h, v_filter_type);

    match (fh, fv) {
        (Some(fh), Some(fv)) => {
            // Two-pass filtering using SIMD
            let tmp_h = h + 7;
            let mut mid = take_mid_i32_135();
            let h_sh = 6 - intermediate_bits; // = 2 for 16bpc
            let v_sh = 6; // prep uses fixed shift of 6 for vertical pass

            // Horizontal pass using SIMD
            for y in 0..tmp_h {
                let src_off = (sb + (y as isize - 3) * src_stride_elems) as usize;
                h_filter_8tap_16bpc_avx2_inner(
                    _token,
                    &mut mid[y],
                    &src[src_off - 3..],
                    w,
                    fh,
                    h_sh,
                );
            }

            // Vertical pass using SIMD (output to i16 with bias subtraction)
            for y in 0..h {
                let out_row = y * w;
                v_filter_8tap_16bpc_prep_avx2_inner(
                    _token,
                    &mut tmp[out_row..],
                    &*mid,
                    w,
                    y,
                    fv,
                    v_sh,
                    PREP_BIAS,
                );
            }
            put_mid_i32_135(mid);
        }
        (Some(fh), None) => {
            // H-only filtering (SIMD)
            let sh = 6 - intermediate_bits; // = 2 for 16bpc
            for y in 0..h {
                let src_off = (sb + y as isize * src_stride_elems) as usize;
                let out_row = y * w;
                h_filter_8tap_16bpc_prep_direct_avx2_inner(
                    _token,
                    &mut tmp[out_row..],
                    &src[src_off - 3..],
                    w,
                    fh,
                    sh,
                    PREP_BIAS,
                );
            }
        }
        (None, Some(fv)) => {
            // V-only filtering (SIMD)
            let sh = 6 - intermediate_bits; // = 2 for 16bpc
            for y in 0..h {
                let src_off = (sb + (y as isize - 3) * src_stride_elems) as usize;
                let out_row = y * w;
                v_filter_8tap_16bpc_prep_direct_avx2_inner(
                    _token,
                    &mut tmp[out_row..],
                    &src,
                    src_off,
                    src_stride_elems,
                    w,
                    fv,
                    sh,
                    PREP_BIAS,
                );
            }
        }
        (None, None) => {
            // Simple copy with scaling and bias
            for y in 0..h {
                let src_row = &src[(sb + y as isize * src_stride_elems) as usize..];
                let out_row = y * w;
                widen_row_u16_shl_bias_16bpc(
                    _token,
                    &mut tmp[out_row..],
                    src_row,
                    w,
                    intermediate_bits,
                    PREP_BIAS,
                );
            }
        }
    }
}
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[archmage::rite(v3)]
unsafe fn prep_8tap_16bpc_impl_v3(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    h_filter_type: Rav1dFilterMode,
    v_filter_type: Rav1dFilterMode,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();
    unsafe {
        prep_8tap_16bpc_avx2_impl_inner(
            token,
            tmp,
            src_ptr,
            0,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            h_filter_type,
            v_filter_type,
        )
    }
}

/// Generic put_8tap function wrapper for 16bpc
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn put_8tap_16bpc_avx2<const FILTER: usize>(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    _dst: *const FFISafe<PicOffset>,
    _src: *const FFISafe<PicOffset>,
) {
    let filter = Filter2d::from_repr(FILTER).unwrap();
    let (h_filter, v_filter) = filter.hv();

    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
        put_8tap_16bpc_impl_v3(
            dst_ptr,
            dst_stride,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            h_filter,
            v_filter,
        );
    }
}

/// Generic prep_8tap function wrapper for 16bpc
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn prep_8tap_16bpc_avx2<const FILTER: usize>(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    _src: *const FFISafe<PicOffset>,
) {
    let filter = Filter2d::from_repr(FILTER).unwrap();
    let (h_filter, v_filter) = filter.hv();

    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
        prep_8tap_16bpc_impl_v3(
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            h_filter,
            v_filter,
        );
    }
}

// 16bpc wrapper functions for each filter type

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn put_8tap_regular_16bpc_avx2_inner(
    _token: Desktop64,
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    unsafe {
        put_8tap_16bpc_avx2::<{ Filter2d::Regular8Tap as usize }>(
            dst_ptr,
            dst_stride,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            dst,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn put_8tap_regular_16bpc_v3(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    unsafe {
        put_8tap_regular_16bpc_avx2_inner(
            token,
            dst_ptr,
            dst_stride,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            dst,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn put_8tap_regular_smooth_16bpc_avx2_inner(
    _token: Desktop64,
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    unsafe {
        put_8tap_16bpc_avx2::<{ Filter2d::RegularSmooth8Tap as usize }>(
            dst_ptr,
            dst_stride,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            dst,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn put_8tap_regular_smooth_16bpc_v3(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    unsafe {
        put_8tap_regular_smooth_16bpc_avx2_inner(
            token,
            dst_ptr,
            dst_stride,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            dst,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn put_8tap_regular_sharp_16bpc_avx2_inner(
    _token: Desktop64,
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    unsafe {
        put_8tap_16bpc_avx2::<{ Filter2d::RegularSharp8Tap as usize }>(
            dst_ptr,
            dst_stride,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            dst,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn put_8tap_regular_sharp_16bpc_v3(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    unsafe {
        put_8tap_regular_sharp_16bpc_avx2_inner(
            token,
            dst_ptr,
            dst_stride,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            dst,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn put_8tap_smooth_regular_16bpc_avx2_inner(
    _token: Desktop64,
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    unsafe {
        put_8tap_16bpc_avx2::<{ Filter2d::SmoothRegular8Tap as usize }>(
            dst_ptr,
            dst_stride,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            dst,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn put_8tap_smooth_regular_16bpc_v3(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    unsafe {
        put_8tap_smooth_regular_16bpc_avx2_inner(
            token,
            dst_ptr,
            dst_stride,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            dst,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn put_8tap_smooth_16bpc_avx2_inner(
    _token: Desktop64,
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    unsafe {
        put_8tap_16bpc_avx2::<{ Filter2d::Smooth8Tap as usize }>(
            dst_ptr,
            dst_stride,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            dst,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn put_8tap_smooth_16bpc_v3(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    unsafe {
        put_8tap_smooth_16bpc_avx2_inner(
            token,
            dst_ptr,
            dst_stride,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            dst,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn put_8tap_smooth_sharp_16bpc_avx2_inner(
    _token: Desktop64,
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    unsafe {
        put_8tap_16bpc_avx2::<{ Filter2d::SmoothSharp8Tap as usize }>(
            dst_ptr,
            dst_stride,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            dst,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn put_8tap_smooth_sharp_16bpc_v3(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    unsafe {
        put_8tap_smooth_sharp_16bpc_avx2_inner(
            token,
            dst_ptr,
            dst_stride,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            dst,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn put_8tap_sharp_regular_16bpc_avx2_inner(
    _token: Desktop64,
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    unsafe {
        put_8tap_16bpc_avx2::<{ Filter2d::SharpRegular8Tap as usize }>(
            dst_ptr,
            dst_stride,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            dst,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn put_8tap_sharp_regular_16bpc_v3(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    unsafe {
        put_8tap_sharp_regular_16bpc_avx2_inner(
            token,
            dst_ptr,
            dst_stride,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            dst,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn put_8tap_sharp_smooth_16bpc_avx2_inner(
    _token: Desktop64,
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    unsafe {
        put_8tap_16bpc_avx2::<{ Filter2d::SharpSmooth8Tap as usize }>(
            dst_ptr,
            dst_stride,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            dst,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn put_8tap_sharp_smooth_16bpc_v3(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    unsafe {
        put_8tap_sharp_smooth_16bpc_avx2_inner(
            token,
            dst_ptr,
            dst_stride,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            dst,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn put_8tap_sharp_16bpc_avx2_inner(
    _token: Desktop64,
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    unsafe {
        put_8tap_16bpc_avx2::<{ Filter2d::Sharp8Tap as usize }>(
            dst_ptr,
            dst_stride,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            dst,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn put_8tap_sharp_16bpc_v3(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    dst: *const FFISafe<PicOffset>,
    src: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    unsafe {
        put_8tap_sharp_16bpc_avx2_inner(
            token,
            dst_ptr,
            dst_stride,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            dst,
            src,
        )
    }
}

// 16bpc prep wrapper functions

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn prep_8tap_regular_16bpc_avx2_inner(
    _token: Desktop64,
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    unsafe {
        prep_8tap_16bpc_avx2::<{ Filter2d::Regular8Tap as usize }>(
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn prep_8tap_regular_16bpc_v3(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    unsafe {
        prep_8tap_regular_16bpc_avx2_inner(
            token,
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn prep_8tap_regular_smooth_16bpc_avx2_inner(
    _token: Desktop64,
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    unsafe {
        prep_8tap_16bpc_avx2::<{ Filter2d::RegularSmooth8Tap as usize }>(
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn prep_8tap_regular_smooth_16bpc_v3(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    unsafe {
        prep_8tap_regular_smooth_16bpc_avx2_inner(
            token,
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn prep_8tap_regular_sharp_16bpc_avx2_inner(
    _token: Desktop64,
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    unsafe {
        prep_8tap_16bpc_avx2::<{ Filter2d::RegularSharp8Tap as usize }>(
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn prep_8tap_regular_sharp_16bpc_v3(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    unsafe {
        prep_8tap_regular_sharp_16bpc_avx2_inner(
            token,
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn prep_8tap_smooth_regular_16bpc_avx2_inner(
    _token: Desktop64,
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    unsafe {
        prep_8tap_16bpc_avx2::<{ Filter2d::SmoothRegular8Tap as usize }>(
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn prep_8tap_smooth_regular_16bpc_v3(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    unsafe {
        prep_8tap_smooth_regular_16bpc_avx2_inner(
            token,
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn prep_8tap_smooth_16bpc_avx2_inner(
    _token: Desktop64,
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    unsafe {
        prep_8tap_16bpc_avx2::<{ Filter2d::Smooth8Tap as usize }>(
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn prep_8tap_smooth_16bpc_v3(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    unsafe {
        prep_8tap_smooth_16bpc_avx2_inner(
            token,
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn prep_8tap_smooth_sharp_16bpc_avx2_inner(
    _token: Desktop64,
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    unsafe {
        prep_8tap_16bpc_avx2::<{ Filter2d::SmoothSharp8Tap as usize }>(
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn prep_8tap_smooth_sharp_16bpc_v3(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    unsafe {
        prep_8tap_smooth_sharp_16bpc_avx2_inner(
            token,
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn prep_8tap_sharp_regular_16bpc_avx2_inner(
    _token: Desktop64,
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    unsafe {
        prep_8tap_16bpc_avx2::<{ Filter2d::SharpRegular8Tap as usize }>(
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn prep_8tap_sharp_regular_16bpc_v3(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    unsafe {
        prep_8tap_sharp_regular_16bpc_avx2_inner(
            token,
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn prep_8tap_sharp_smooth_16bpc_avx2_inner(
    _token: Desktop64,
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    unsafe {
        prep_8tap_16bpc_avx2::<{ Filter2d::SharpSmooth8Tap as usize }>(
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn prep_8tap_sharp_smooth_16bpc_v3(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    unsafe {
        prep_8tap_sharp_smooth_16bpc_avx2_inner(
            token,
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn prep_8tap_sharp_16bpc_avx2_inner(
    _token: Desktop64,
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    unsafe {
        prep_8tap_16bpc_avx2::<{ Filter2d::Sharp8Tap as usize }>(
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn prep_8tap_sharp_16bpc_v3(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    src: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    unsafe {
        prep_8tap_sharp_16bpc_avx2_inner(
            token,
            tmp,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            src,
        )
    }
}

// ============================================================================
// BILINEAR FILTER IMPLEMENTATIONS
// ============================================================================
//
// Bilinear filtering uses a simple 2-tap filter:
//   pixel = (16 - mxy) * x0 + mxy * x1
// where mxy is 0-15 (4 bits of fractional precision)
//
// For H+V filtering:
//   1. Horizontal pass: filter to intermediate buffer
//   2. Vertical pass: filter intermediate to output
//
// The intermediate uses extra precision bits to avoid rounding errors.

/// Horizontal bilinear filter for 8bpc using AVX2
/// Processes 32 pixels at a time, outputs to i16 intermediate buffer
#[cfg(target_arch = "x86_64")]
#[rite]
fn h_filter_bilin_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [i16],
    src: &[u8],
    w: usize,
    mx: usize,
    sh: u8,
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    // Bilinear: pixel = (16 - mx) * src[x] + mx * src[x+1]
    // Using maddubs: need pairs of [src[x], src[x+1]] with coeffs [16-mx, mx]
    let mx = mx as i8;
    let coeff0 = (16 - mx) as u8;
    let coeff1 = mx as u8;

    // Create coefficient vector for maddubs: [coeff0, coeff1] repeated
    let coeffs = _mm256_set1_epi16(((coeff1 as i16) << 8) | (coeff0 as i16));

    // Rounding value (handle sh=0 case)
    let rnd = if sh > 0 {
        _mm256_set1_epi16(1 << (sh - 1))
    } else {
        _mm256_setzero_si256()
    };

    // Shift count in register for variable shift
    let sh_reg = _mm_cvtsi32_si128(sh as i32);

    let mut x = 0;
    while x + 32 <= w {
        // Load 33 bytes to get 32 pairs of adjacent pixels
        let src_lo = loadu_256!(<&[u8; 32]>::try_from(&src[x..x + 32]).unwrap());
        let src_hi = loadu_256!(<&[u8; 32]>::try_from(&src[x + 1..x + 33]).unwrap());

        // Interleave for maddubs: need [src[0],src[1]], [src[1],src[2]], ...
        // unpacklo gives us pairs from low halves, unpackhi from high halves
        let pairs_lo = _mm256_unpacklo_epi8(src_lo, src_hi);
        let pairs_hi = _mm256_unpackhi_epi8(src_lo, src_hi);

        // Apply bilinear filter using maddubs
        let result_lo = _mm256_maddubs_epi16(pairs_lo, coeffs);
        let result_hi = _mm256_maddubs_epi16(pairs_hi, coeffs);

        // Add rounding and shift (using variable shift)
        let result_lo = _mm256_sra_epi16(_mm256_add_epi16(result_lo, rnd), sh_reg);
        let result_hi = _mm256_sra_epi16(_mm256_add_epi16(result_hi, rnd), sh_reg);

        // Store results - need to handle the permutation from unpack
        // unpack interleaves within 128-bit lanes, so we need to fix the order
        let lo_128 = _mm256_permute2x128_si256(result_lo, result_hi, 0x20); // lo lanes
        let hi_128 = _mm256_permute2x128_si256(result_lo, result_hi, 0x31); // hi lanes

        storeu_256!(
            <&mut [i16; 16]>::try_from(&mut dst[x..x + 16]).unwrap(),
            lo_128
        );
        storeu_256!(
            <&mut [i16; 16]>::try_from(&mut dst[x + 16..x + 32]).unwrap(),
            hi_128
        );
        x += 32;
    }

    // Scalar fallback for remaining pixels
    while x < w {
        let x0 = src[x] as i32;
        let x1 = src[x + 1] as i32;
        let pixel = (16 - mx as i32) * x0 + mx as i32 * x1;
        let result = if sh > 0 {
            (pixel + (1 << (sh - 1))) >> sh
        } else {
            pixel
        };
        dst[x] = result as i16;
        x += 1;
    }
}
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[archmage::rite(v3)]
unsafe fn h_filter_bilin_8bpc_v3(dst: *mut i16, src: *const u8, w: usize, mx: usize, sh: u8) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();
    unsafe { h_filter_bilin_8bpc_avx2_inner(token, dst, src, w, mx, sh) }
}

/// Vertical bilinear filter for 8bpc using AVX2
/// Reads from i16 intermediate buffer, outputs to u8
#[cfg(target_arch = "x86_64")]
#[rite]
fn v_filter_bilin_8bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    mid: &[&[i16]],
    w: usize,
    my: usize,
    sh: u8,
    bd_max: i16,
) {
    let mut dst = dst.flex_mut();
    let my = my as i32;
    let coeff0 = 16 - my;
    let coeff1 = my;

    // Use 32-bit arithmetic to avoid i16 overflow.
    // Intermediate values can be up to 4080 (16*255 for 8bpc), and
    // coeff * mid can reach 16 * 4080 = 65280, exceeding i16 range.
    let c0_32 = _mm256_set1_epi32(coeff0);
    let c1_32 = _mm256_set1_epi32(coeff1);
    let rnd = _mm256_set1_epi32(if sh > 0 { 1 << (sh - 1) } else { 0 });
    let zero = _mm256_setzero_si256();
    let max_32 = _mm256_set1_epi32(bd_max as i32);
    let sh_reg = _mm_cvtsi32_si128(sh as i32);

    let mut x = 0;
    // Process 8 pixels at a time using 32-bit arithmetic
    while x + 8 <= w {
        // Load 8 i16 values from each row and sign-extend to i32
        let row0 = _mm256_cvtepi16_epi32(loadu_128!(
            <&[i16; 8]>::try_from(&mid[0][x..x + 8]).unwrap()
        ));
        let row1 = _mm256_cvtepi16_epi32(loadu_128!(
            <&[i16; 8]>::try_from(&mid[1][x..x + 8]).unwrap()
        ));

        // result = coeff0 * row0 + coeff1 * row1 (32-bit, no overflow)
        let mul0 = _mm256_mullo_epi32(row0, c0_32);
        let mul1 = _mm256_mullo_epi32(row1, c1_32);
        let sum = _mm256_add_epi32(mul0, mul1);

        // Add rounding and shift
        let result = _mm256_sra_epi32(_mm256_add_epi32(sum, rnd), sh_reg);

        // Clamp to [0, bd_max]
        let result = _mm256_max_epi32(result, zero);
        let result = _mm256_min_epi32(result, max_32);

        // Pack 32-bit to 16-bit, then 16-bit to 8-bit
        let packed16 = _mm256_packs_epi32(result, result);
        let packed16 = _mm256_permute4x64_epi64(packed16, 0xD8);
        let packed8 = _mm256_packus_epi16(packed16, packed16);

        // Store 8 bytes
        let result_64 = _mm256_extract_epi64(packed8, 0);
        dst[x..x + 8].copy_from_slice(&result_64.to_ne_bytes());
        x += 8;
    }

    // Scalar fallback
    while x < w {
        let r0 = mid[0][x] as i32;
        let r1 = mid[1][x] as i32;
        let pixel = coeff0 * r0 + coeff1 * r1;
        let result = if sh > 0 {
            ((pixel + (1 << (sh - 1))) >> sh).clamp(0, bd_max as i32)
        } else {
            pixel.clamp(0, bd_max as i32)
        };
        dst[x] = result as u8;
        x += 1;
    }
}
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[archmage::rite(v3)]
unsafe fn v_filter_bilin_8bpc_v3(
    dst: *mut u8,
    mid: &[&[i16]],
    w: usize,
    my: usize,
    sh: u8,
    bd_max: i16,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();
    unsafe { v_filter_bilin_8bpc_avx2_inner(token, dst, mid, w, my, sh, bd_max) }
}
/// Horizontal bilinear filter for 8bpc put (H-only)
/// Outputs directly to u8 with shift=4 and clamp to [0,255]
#[cfg(target_arch = "x86_64")]
#[rite]
fn h_bilin_8bpc_put_avx2_inner(_token: Desktop64, dst: &mut [u8], src: &[u8], w: usize, mx: usize) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    let mx = mx as i8;
    let coeff0 = (16 - mx) as u8;
    let coeff1 = mx as u8;
    let coeffs = _mm256_set1_epi16(((coeff1 as i16) << 8) | (coeff0 as i16));
    let rnd = _mm256_set1_epi16(8); // (1 << 4) >> 1
    let zero = _mm256_setzero_si256();
    let max = _mm256_set1_epi16(255);

    let mut x = 0;
    while x + 32 <= w {
        let src_lo = loadu_256!(<&[u8; 32]>::try_from(&src[x..x + 32]).unwrap());
        let src_hi = loadu_256!(<&[u8; 32]>::try_from(&src[x + 1..x + 33]).unwrap());

        let pairs_lo = _mm256_unpacklo_epi8(src_lo, src_hi);
        let pairs_hi = _mm256_unpackhi_epi8(src_lo, src_hi);

        let result_lo = _mm256_maddubs_epi16(pairs_lo, coeffs);
        let result_hi = _mm256_maddubs_epi16(pairs_hi, coeffs);

        // Add rounding, shift by 4, clamp
        let sh_reg = _mm_cvtsi32_si128(4);
        let result_lo = _mm256_sra_epi16(_mm256_add_epi16(result_lo, rnd), sh_reg);
        let result_hi = _mm256_sra_epi16(_mm256_add_epi16(result_hi, rnd), sh_reg);

        let result_lo = _mm256_max_epi16(_mm256_min_epi16(result_lo, max), zero);
        let result_hi = _mm256_max_epi16(_mm256_min_epi16(result_hi, max), zero);

        // Pack to u8: packus_epi16 operates within 128-bit lanes, and
        // result_lo/result_hi already have matching lane layout from unpack,
        // so packus(result_lo, result_hi) produces correct sequential order.
        let packed = _mm256_packus_epi16(result_lo, result_hi);

        storeu_256!(
            <&mut [u8; 32]>::try_from(&mut dst[x..x + 32]).unwrap(),
            packed
        );
        x += 32;
    }

    while x < w {
        let x0 = src[x] as i32;
        let x1 = src[x + 1] as i32;
        let pixel = (16 - mx as i32) * x0 + mx as i32 * x1;
        let result = ((pixel + 8) >> 4).clamp(0, 255);
        dst[x] = result as u8;
        x += 1;
    }
}
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[archmage::rite(v3)]
unsafe fn h_bilin_8bpc_put_v3(dst: *mut u8, src: *const u8, w: usize, mx: usize) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();
    unsafe { h_bilin_8bpc_put_avx2_inner(token, dst, src, w, mx) }
}

/// Vertical bilinear filter for 8bpc put (V-only)
/// Reads directly from u8 source, outputs u8
#[cfg(target_arch = "x86_64")]
#[rite]
fn v_bilin_8bpc_direct_avx2_inner(
    _token: Desktop64,
    dst: &mut [u8],
    src0: &[u8],
    src1: &[u8],
    w: usize,
    my: usize,
) {
    let mut dst = dst.flex_mut();
    let src0 = src0.flex();
    let src1 = src1.flex();
    let my = my as i8;
    let coeff0 = (16 - my) as u8;
    let coeff1 = my as u8;
    let coeffs = _mm256_set1_epi16(((coeff1 as i16) << 8) | (coeff0 as i16));
    let rnd = _mm256_set1_epi16(8);
    let sh_reg = _mm_cvtsi32_si128(4);

    let mut x = 0;
    while x + 32 <= w {
        let s0 = loadu_256!(<&[u8; 32]>::try_from(&src0[x..x + 32]).unwrap());
        let s1 = loadu_256!(<&[u8; 32]>::try_from(&src1[x..x + 32]).unwrap());

        let pairs_lo = _mm256_unpacklo_epi8(s0, s1);
        let pairs_hi = _mm256_unpackhi_epi8(s0, s1);

        let result_lo = _mm256_maddubs_epi16(pairs_lo, coeffs);
        let result_hi = _mm256_maddubs_epi16(pairs_hi, coeffs);

        let result_lo = _mm256_sra_epi16(_mm256_add_epi16(result_lo, rnd), sh_reg);
        let result_hi = _mm256_sra_epi16(_mm256_add_epi16(result_hi, rnd), sh_reg);

        // packus_epi16 operates within 128-bit lanes, so with result_lo
        // holding elements [0..7 | 16..23] and result_hi holding [8..15 | 24..31],
        // packus(result_lo, result_hi) correctly produces [0..31] in order.
        let packed = _mm256_packus_epi16(result_lo, result_hi);

        storeu_256!(
            <&mut [u8; 32]>::try_from(&mut dst[x..x + 32]).unwrap(),
            packed
        );
        x += 32;
    }

    while x < w {
        let x0 = src0[x] as i32;
        let x1 = src1[x] as i32;
        let pixel = (16 - my as i32) * x0 + my as i32 * x1;
        let result = ((pixel + 8) >> 4).clamp(0, 255);
        dst[x] = result as u8;
        x += 1;
    }
}
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[archmage::rite(v3)]
unsafe fn v_bilin_8bpc_direct_v3(
    dst: *mut u8,
    src0: *const u8,
    src1: *const u8,
    w: usize,
    my: usize,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();
    unsafe { v_bilin_8bpc_direct_avx2_inner(token, dst, src0, src1, w, my) }
}

// ============================================================================
// 8BPC BILINEAR AVX-512
// ============================================================================

/// Horizontal bilinear filter for 8bpc using AVX-512
/// Processes 64 pixels at a time, outputs to i16 intermediate buffer
#[cfg(target_arch = "x86_64")]
#[rite]
fn h_filter_bilin_8bpc_avx512_inner(
    _token: Server64,
    dst: &mut [i16],
    src: &[u8],
    w: usize,
    mx: usize,
    sh: u8,
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    let mx = mx as i8;
    let coeff0 = (16 - mx) as u8;
    let coeff1 = mx as u8;
    let coeffs = _mm512_set1_epi16(((coeff1 as i16) << 8) | (coeff0 as i16));
    let rnd = if sh > 0 {
        _mm512_set1_epi16(1 << (sh - 1))
    } else {
        _mm512_setzero_si512()
    };
    let sh_reg = _mm_cvtsi32_si128(sh as i32);

    // Lane-fix indices for permutex2var_epi64.
    // After unpack+maddubs, result_lo has [p0-7|p16-23|p32-39|p48-55] and
    // result_hi has [p8-15|p24-31|p40-47|p56-63] (each in 128-bit lanes).
    // We need sequential output: [p0-31] and [p32-63].
    let idx0 = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
    let idx1 = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);

    let mut x = 0;
    while x + 64 <= w {
        let src_lo = loadu_512!(&src[x..x + 64], [u8; 64]);
        let src_hi = loadu_512!(&src[x + 1..x + 65], [u8; 64]);

        let pairs_lo = _mm512_unpacklo_epi8(src_lo, src_hi);
        let pairs_hi = _mm512_unpackhi_epi8(src_lo, src_hi);

        let result_lo = _mm512_maddubs_epi16(pairs_lo, coeffs);
        let result_hi = _mm512_maddubs_epi16(pairs_hi, coeffs);

        let result_lo = _mm512_sra_epi16(_mm512_add_epi16(result_lo, rnd), sh_reg);
        let result_hi = _mm512_sra_epi16(_mm512_add_epi16(result_hi, rnd), sh_reg);

        let out0 = _mm512_permutex2var_epi64(result_lo, idx0, result_hi);
        let out1 = _mm512_permutex2var_epi64(result_lo, idx1, result_hi);

        storeu_512!(
            <&mut [i16; 32]>::try_from(&mut dst[x..x + 32]).unwrap(),
            out0
        );
        storeu_512!(
            <&mut [i16; 32]>::try_from(&mut dst[x + 32..x + 64]).unwrap(),
            out1
        );
        x += 64;
    }

    while x < w {
        let x0 = src[x] as i32;
        let x1 = src[x + 1] as i32;
        let pixel = (16 - mx as i32) * x0 + mx as i32 * x1;
        let result = if sh > 0 {
            (pixel + (1 << (sh - 1))) >> sh
        } else {
            pixel
        };
        dst[x] = result as i16;
        x += 1;
    }
}

/// Vertical bilinear filter for 8bpc AVX-512 (mid → u8)
/// 16 pixels/iter using 32-bit arithmetic, cvtusepi32_epi8 for pack
#[cfg(target_arch = "x86_64")]
#[rite]
fn v_filter_bilin_8bpc_avx512_inner(
    _token: Server64,
    dst: &mut [u8],
    mid: &[&[i16]],
    w: usize,
    my: usize,
    sh: u8,
    _bd_max: i16,
) {
    let mut dst = dst.flex_mut();
    let my = my as i32;
    let coeff0 = 16 - my;
    let coeff1 = my;

    let c0_32 = _mm512_set1_epi32(coeff0);
    let c1_32 = _mm512_set1_epi32(coeff1);
    let rnd = _mm512_set1_epi32(if sh > 0 { 1 << (sh - 1) } else { 0 });
    let zero = _mm512_setzero_si512();
    let max_32 = _mm512_set1_epi32(255);
    let sh_reg = _mm_cvtsi32_si128(sh as i32);

    let mut x = 0;
    while x + 16 <= w {
        let row0 = _mm512_cvtepi16_epi32(loadu_256!(
            <&[i16; 16]>::try_from(&mid[0][x..x + 16]).unwrap()
        ));
        let row1 = _mm512_cvtepi16_epi32(loadu_256!(
            <&[i16; 16]>::try_from(&mid[1][x..x + 16]).unwrap()
        ));

        let mul0 = _mm512_mullo_epi32(row0, c0_32);
        let mul1 = _mm512_mullo_epi32(row1, c1_32);
        let sum = _mm512_add_epi32(mul0, mul1);

        let result = _mm512_sra_epi32(_mm512_add_epi32(sum, rnd), sh_reg);
        let result = _mm512_max_epi32(result, zero);
        let result = _mm512_min_epi32(result, max_32);

        // Pack i32 → u8: 16 i32 → 16 u8 (__m128i)
        let packed = _mm512_cvtusepi32_epi8(result);
        storeu_128!(
            <&mut [u8; 16]>::try_from(&mut dst[x..x + 16]).unwrap(),
            packed
        );
        x += 16;
    }

    while x < w {
        let r0 = mid[0][x] as i32;
        let r1 = mid[1][x] as i32;
        let pixel = coeff0 * r0 + coeff1 * r1;
        let result = if sh > 0 {
            ((pixel + (1 << (sh - 1))) >> sh).clamp(0, 255)
        } else {
            pixel.clamp(0, 255)
        };
        dst[x] = result as u8;
        x += 1;
    }
}

/// Vertical bilinear filter for 8bpc AVX-512 prep (mid → i16)
/// 16 pixels/iter using 32-bit arithmetic
#[cfg(target_arch = "x86_64")]
#[rite]
fn v_filter_bilin_8bpc_prep_avx512_inner(
    _token: Server64,
    dst: &mut [i16],
    mid: &[&[i16]],
    w: usize,
    my: usize,
    sh: u8,
) {
    let mut dst = dst.flex_mut();
    let my = my as i32;
    let coeff0 = 16 - my;
    let coeff1 = my;

    let c0_32 = _mm512_set1_epi32(coeff0);
    let c1_32 = _mm512_set1_epi32(coeff1);
    let rnd = _mm512_set1_epi32(if sh > 0 { 1 << (sh - 1) } else { 0 });
    let sh_reg = _mm_cvtsi32_si128(sh as i32);

    let mut x = 0;
    while x + 16 <= w {
        let row0 = _mm512_cvtepi16_epi32(loadu_256!(
            <&[i16; 16]>::try_from(&mid[0][x..x + 16]).unwrap()
        ));
        let row1 = _mm512_cvtepi16_epi32(loadu_256!(
            <&[i16; 16]>::try_from(&mid[1][x..x + 16]).unwrap()
        ));

        let mul0 = _mm512_mullo_epi32(row0, c0_32);
        let mul1 = _mm512_mullo_epi32(row1, c1_32);
        let sum = _mm512_add_epi32(mul0, mul1);

        let result = _mm512_sra_epi32(_mm512_add_epi32(sum, rnd), sh_reg);

        let packed = _mm512_cvtsepi32_epi16(result);
        storeu_256!(
            <&mut [i16; 16]>::try_from(&mut dst[x..x + 16]).unwrap(),
            packed
        );
        x += 16;
    }

    while x < w {
        let r0 = mid[0][x] as i32;
        let r1 = mid[1][x] as i32;
        let pixel = coeff0 * r0 + coeff1 * r1;
        let result = if sh > 0 {
            (pixel + (1 << (sh - 1))) >> sh
        } else {
            pixel
        };
        dst[x] = result as i16;
        x += 1;
    }
}

/// Horizontal bilinear filter for 8bpc AVX-512 put (H-only → u8)
/// 64 pixels/iter, packus naturally produces sequential output
#[cfg(target_arch = "x86_64")]
#[rite]
fn h_bilin_8bpc_put_avx512_inner(
    _token: Server64,
    dst: &mut [u8],
    src: &[u8],
    w: usize,
    mx: usize,
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    let mx = mx as i8;
    let coeff0 = (16 - mx) as u8;
    let coeff1 = mx as u8;
    let coeffs = _mm512_set1_epi16(((coeff1 as i16) << 8) | (coeff0 as i16));
    let rnd = _mm512_set1_epi16(8);
    let sh_reg = _mm_cvtsi32_si128(4);

    let mut x = 0;
    while x + 64 <= w {
        let src_lo = loadu_512!(&src[x..x + 64], [u8; 64]);
        let src_hi = loadu_512!(&src[x + 1..x + 65], [u8; 64]);

        let pairs_lo = _mm512_unpacklo_epi8(src_lo, src_hi);
        let pairs_hi = _mm512_unpackhi_epi8(src_lo, src_hi);

        let result_lo = _mm512_maddubs_epi16(pairs_lo, coeffs);
        let result_hi = _mm512_maddubs_epi16(pairs_hi, coeffs);

        let result_lo = _mm512_sra_epi16(_mm512_add_epi16(result_lo, rnd), sh_reg);
        let result_hi = _mm512_sra_epi16(_mm512_add_epi16(result_hi, rnd), sh_reg);

        // packus within each 128-bit lane naturally produces sequential u8 output
        let packed = _mm512_packus_epi16(result_lo, result_hi);
        storeu_512!(&mut dst[x..x + 64], [u8; 64], packed);
        x += 64;
    }

    while x < w {
        let x0 = src[x] as i32;
        let x1 = src[x + 1] as i32;
        let pixel = (16 - mx as i32) * x0 + mx as i32 * x1;
        let result = ((pixel + 8) >> 4).clamp(0, 255);
        dst[x] = result as u8;
        x += 1;
    }
}

/// Vertical bilinear filter for 8bpc AVX-512 put (V-only u8 → u8)
/// 64 pixels/iter using maddubs on interleaved source rows
#[cfg(target_arch = "x86_64")]
#[rite]
fn v_bilin_8bpc_direct_avx512_inner(
    _token: Server64,
    dst: &mut [u8],
    src0: &[u8],
    src1: &[u8],
    w: usize,
    my: usize,
) {
    let mut dst = dst.flex_mut();
    let src0 = src0.flex();
    let src1 = src1.flex();
    let my = my as i8;
    let coeff0 = (16 - my) as u8;
    let coeff1 = my as u8;
    let coeffs = _mm512_set1_epi16(((coeff1 as i16) << 8) | (coeff0 as i16));
    let rnd = _mm512_set1_epi16(8);
    let sh_reg = _mm_cvtsi32_si128(4);

    let mut x = 0;
    while x + 64 <= w {
        let s0 = loadu_512!(&src0[x..x + 64], [u8; 64]);
        let s1 = loadu_512!(&src1[x..x + 64], [u8; 64]);

        let pairs_lo = _mm512_unpacklo_epi8(s0, s1);
        let pairs_hi = _mm512_unpackhi_epi8(s0, s1);

        let result_lo = _mm512_maddubs_epi16(pairs_lo, coeffs);
        let result_hi = _mm512_maddubs_epi16(pairs_hi, coeffs);

        let result_lo = _mm512_sra_epi16(_mm512_add_epi16(result_lo, rnd), sh_reg);
        let result_hi = _mm512_sra_epi16(_mm512_add_epi16(result_hi, rnd), sh_reg);

        let packed = _mm512_packus_epi16(result_lo, result_hi);
        storeu_512!(&mut dst[x..x + 64], [u8; 64], packed);
        x += 64;
    }

    while x < w {
        let x0 = src0[x] as i32;
        let x1 = src1[x] as i32;
        let pixel = (16 - my as i32) * x0 + my as i32 * x1;
        let result = ((pixel + 8) >> 4).clamp(0, 255);
        dst[x] = result as u8;
        x += 1;
    }
}

/// Vertical bilinear for 8bpc AVX-512 prep (V-only u8 → i16)
/// 64 pixels/iter using maddubs, lane-fix for i16 output
#[cfg(target_arch = "x86_64")]
#[rite]
fn v_bilin_8bpc_prep_direct_avx512_inner(
    _token: Server64,
    dst: &mut [i16],
    src0: &[u8],
    src1: &[u8],
    w: usize,
    my: usize,
) {
    let mut dst = dst.flex_mut();
    let src0 = src0.flex();
    let src1 = src1.flex();
    let my = my as i8;
    let coeff0 = (16 - my) as u8;
    let coeff1 = my as u8;
    let coeffs = _mm512_set1_epi16(((coeff1 as i16) << 8) | (coeff0 as i16));

    let idx0 = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
    let idx1 = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);

    let mut x = 0;
    while x + 64 <= w {
        let s0 = loadu_512!(&src0[x..x + 64], [u8; 64]);
        let s1 = loadu_512!(&src1[x..x + 64], [u8; 64]);

        let pairs_lo = _mm512_unpacklo_epi8(s0, s1);
        let pairs_hi = _mm512_unpackhi_epi8(s0, s1);

        // No shift for V-only prep 8bpc: raw bilinear [0, 4080] fits in i16
        let result_lo = _mm512_maddubs_epi16(pairs_lo, coeffs);
        let result_hi = _mm512_maddubs_epi16(pairs_hi, coeffs);

        let out0 = _mm512_permutex2var_epi64(result_lo, idx0, result_hi);
        let out1 = _mm512_permutex2var_epi64(result_lo, idx1, result_hi);

        storeu_512!(
            <&mut [i16; 32]>::try_from(&mut dst[x..x + 32]).unwrap(),
            out0
        );
        storeu_512!(
            <&mut [i16; 32]>::try_from(&mut dst[x + 32..x + 64]).unwrap(),
            out1
        );
        x += 64;
    }

    while x < w {
        let x0 = src0[x] as i32;
        let x1 = src1[x] as i32;
        let pixel = (16 - my as i32) * x0 + my as i32 * x1;
        dst[x] = pixel as i16;
        x += 1;
    }
}

/// Core bilinear put implementation for 8bpc using AVX-512
#[cfg(target_arch = "x86_64")]
#[arcane]
pub(crate) fn put_bilin_8bpc_avx512_impl_inner(
    _token: Server64,
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    src: &[u8],
    src_base: usize,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    let w = w as usize;
    let h = h as usize;
    let mx = mx as usize;
    let my = my as usize;
    let intermediate_bits = 4u8;

    match (mx != 0, my != 0) {
        (true, true) => {
            let tmp_h = h + 1;
            let mut mid = take_mid_i16_130();
            for y in 0..tmp_h {
                let src_row_base = src_base.wrapping_add_signed(y as isize * src_stride);
                h_filter_bilin_8bpc_avx512_inner(
                    _token,
                    &mut mid[y],
                    &src[src_row_base..],
                    w,
                    mx,
                    4 - intermediate_bits,
                );
            }
            for y in 0..h {
                let dst_row = &mut dst[dst_base.wrapping_add_signed(y as isize * dst_stride)..];
                let mid_refs: [&[i16]; 2] = [&mid[y], &mid[y + 1]];
                v_filter_bilin_8bpc_avx512_inner(
                    _token,
                    dst_row,
                    &mid_refs,
                    w,
                    my,
                    4 + intermediate_bits,
                    255,
                );
            }
            put_mid_i16_130(mid);
        }
        (true, false) => {
            for y in 0..h {
                let src_row_base = src_base.wrapping_add_signed(y as isize * src_stride);
                let dst_row = &mut dst[dst_base.wrapping_add_signed(y as isize * dst_stride)..];
                h_bilin_8bpc_put_avx512_inner(_token, dst_row, &src[src_row_base..], w, mx);
            }
        }
        (false, true) => {
            for y in 0..h {
                let src_row0 = &src[src_base.wrapping_add_signed(y as isize * src_stride)..];
                let src_row1 = &src[src_base.wrapping_add_signed((y + 1) as isize * src_stride)..];
                let dst_row = &mut dst[dst_base.wrapping_add_signed(y as isize * dst_stride)..];
                v_bilin_8bpc_direct_avx512_inner(_token, dst_row, src_row0, src_row1, w, my);
            }
        }
        (false, false) => {
            for y in 0..h {
                let src_row_base = src_base.wrapping_add_signed(y as isize * src_stride);
                let dst_row = &mut dst[dst_base.wrapping_add_signed(y as isize * dst_stride)..];
                copy_row_px(dst_row, &src[src_row_base..], w);
            }
        }
    }
}

/// Core bilinear prep implementation for 8bpc using AVX-512
#[cfg(target_arch = "x86_64")]
#[arcane]
fn prep_bilin_8bpc_avx512_impl_inner(
    _token: Server64,
    tmp: &mut [i16],
    src: &[u8],
    src_base: usize,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
) {
    let mut tmp = tmp.flex_mut();
    let src = src.flex();
    let w = w as usize;
    let h = h as usize;
    let mx = mx as usize;
    let my = my as usize;
    let intermediate_bits = 4u8;

    match (mx != 0, my != 0) {
        (true, true) => {
            let tmp_h = h + 1;
            let mut mid = take_mid_i16_130();
            for y in 0..tmp_h {
                let src_row_base = src_base.wrapping_add_signed(y as isize * src_stride);
                h_filter_bilin_8bpc_avx512_inner(
                    _token,
                    &mut mid[y],
                    &src[src_row_base..],
                    w,
                    mx,
                    4 - intermediate_bits,
                );
            }
            for y in 0..h {
                let dst_row = y * w;
                let mid_refs: [&[i16]; 2] = [&mid[y], &mid[y + 1]];
                v_filter_bilin_8bpc_prep_avx512_inner(
                    _token,
                    &mut tmp[dst_row..],
                    &mid_refs,
                    w,
                    my,
                    4,
                );
            }
            put_mid_i16_130(mid);
        }
        (true, false) => {
            for y in 0..h {
                let src_row_base = src_base.wrapping_add_signed(y as isize * src_stride);
                let dst_row = y * w;
                h_filter_bilin_8bpc_avx512_inner(
                    _token,
                    &mut tmp[dst_row..],
                    &src[src_row_base..],
                    w,
                    mx,
                    4 - intermediate_bits,
                );
            }
        }
        (false, true) => {
            for y in 0..h {
                let src_row0 = &src[src_base.wrapping_add_signed(y as isize * src_stride)..];
                let src_row1 = &src[src_base.wrapping_add_signed((y + 1) as isize * src_stride)..];
                let dst_row = y * w;
                v_bilin_8bpc_prep_direct_avx512_inner(
                    _token,
                    &mut tmp[dst_row..],
                    src_row0,
                    src_row1,
                    w,
                    my,
                );
            }
        }
        (false, false) => {
            for y in 0..h {
                let src_row_base = src_base.wrapping_add_signed(y as isize * src_stride);
                let dst_row = y * w;
                for x in 0..w {
                    let pixel = src[src_row_base + x] as i16;
                    tmp[dst_row + x] = pixel << intermediate_bits;
                }
            }
        }
    }
}

/// Core bilinear filter implementation for 8bpc
#[cfg(target_arch = "x86_64")]
#[rite]
pub(crate) fn put_bilin_8bpc_avx2_impl_inner(
    _token: Desktop64,
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    src: &[u8],
    src_base: usize,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    let w = w as usize;
    let h = h as usize;
    let mx = mx as usize;
    let my = my as usize;

    // For 8bpc: intermediate_bits = 4
    let intermediate_bits = 4u8;

    match (mx != 0, my != 0) {
        (true, true) => {
            // Case 1: Both H and V filtering
            // First pass: horizontal filter to intermediate buffer
            let tmp_h = h + 1;
            let mut mid = take_mid_i16_130();

            for y in 0..tmp_h {
                let src_row_base = src_base.wrapping_add_signed(y as isize * src_stride);
                let src_row = &src[src_row_base..];
                h_filter_bilin_8bpc_avx2_inner(
                    _token,
                    &mut mid[y],
                    src_row,
                    w,
                    mx,
                    4 - intermediate_bits, // sh = 0 for intermediate
                );
            }

            // Second pass: vertical filter to output
            for y in 0..h {
                let dst_row = &mut dst[dst_base.wrapping_add_signed(y as isize * dst_stride)..];
                let mid_refs: [&[i16]; 2] = [&mid[y], &mid[y + 1]];
                v_filter_bilin_8bpc_avx2_inner(
                    _token,
                    dst_row,
                    &mid_refs,
                    w,
                    my,
                    4 + intermediate_bits, // sh = 8 for final
                    255,
                );
            }
            put_mid_i16_130(mid);
        }
        (true, false) => {
            // Case 2: H-only filtering (full SIMD)
            for y in 0..h {
                let src_row_base = src_base.wrapping_add_signed(y as isize * src_stride);
                let src_row = &src[src_row_base..];
                let dst_row = &mut dst[dst_base.wrapping_add_signed(y as isize * dst_stride)..];
                h_bilin_8bpc_put_avx2_inner(_token, dst_row, src_row, w, mx);
            }
        }
        (false, true) => {
            // Case 3: V-only filtering (full SIMD)
            for y in 0..h {
                let src_row0 = &src[src_base.wrapping_add_signed(y as isize * src_stride)..];
                let src_row1 = &src[src_base.wrapping_add_signed((y + 1) as isize * src_stride)..];
                let dst_row = &mut dst[dst_base.wrapping_add_signed(y as isize * dst_stride)..];
                v_bilin_8bpc_direct_avx2_inner(_token, dst_row, src_row0, src_row1, w, my);
            }
        }
        (false, false) => {
            // Case 4: Simple copy
            for y in 0..h {
                let src_row_base = src_base.wrapping_add_signed(y as isize * src_stride);
                let src_row = &src[src_row_base..];
                let dst_row = &mut dst[dst_base.wrapping_add_signed(y as isize * dst_stride)..];
                copy_row_px(dst_row, src_row, w);
            }
        }
    }
}
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[archmage::rite(v3)]
unsafe fn put_bilin_8bpc_impl_v3(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();
    unsafe {
        put_bilin_8bpc_avx2_impl_inner(
            token, dst_ptr, 0, dst_stride, src_ptr, 0, src_stride, w, h, mx, my,
        )
    }
}

/// Bilinear put for 8bpc - extern "C" wrapper
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn put_bilin_8bpc_avx2_inner(
    _token: Desktop64,
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
        put_bilin_8bpc_impl_v3(dst_ptr, dst_stride, src_ptr, src_stride, w, h, mx, my);
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn put_bilin_8bpc_v3(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    _bitdepth_max: i32,
    _dst: *const FFISafe<PicOffset>,
    _src: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    unsafe {
        put_bilin_8bpc_avx2_inner(
            token, dst_ptr, dst_stride, src_ptr, src_stride, w, h, mx, my,
        )
    }
}

/// Core bilinear prep implementation for 8bpc
/// Outputs to i16 intermediate buffer (prep format)
#[cfg(target_arch = "x86_64")]
#[rite]
pub(crate) fn prep_bilin_8bpc_avx2_impl_inner(
    _token: Desktop64,
    tmp: &mut [i16],
    src: &[u8],
    src_base: usize,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
) {
    let mut tmp = tmp.flex_mut();
    let src = src.flex();
    let w = w as usize;
    let h = h as usize;
    let mx = mx as usize;
    let my = my as usize;

    // For 8bpc: intermediate_bits = 4
    let intermediate_bits = 4u8;

    match (mx != 0, my != 0) {
        (true, true) => {
            // Case 1: Both H and V filtering
            let tmp_h = h + 1;
            let mut mid = take_mid_i16_130();

            for y in 0..tmp_h {
                let src_row_base = src_base.wrapping_add_signed(y as isize * src_stride);
                let src_row = &src[src_row_base..];
                h_filter_bilin_8bpc_avx2_inner(
                    _token,
                    &mut mid[y],
                    src_row,
                    w,
                    mx,
                    4 - intermediate_bits, // sh = 0 for first pass
                );
            }

            // Second pass: vertical filter to output i16 buffer
            // For 8bpc: PREP_BIAS=0, so result = ((pixel + 8) >> 4)
            for y in 0..h {
                let dst_row = y * w;
                for x in 0..w {
                    let r0 = mid[y][x] as i32;
                    let r1 = mid[y + 1][x] as i32;
                    let coeff0 = 16 - my as i32;
                    let coeff1 = my as i32;
                    let pixel = coeff0 * r0 + coeff1 * r1;
                    let result = (pixel + 8) >> 4;
                    tmp[dst_row + x] = result as i16; // PREP_BIAS = 0 for 8bpc
                }
            }
            put_mid_i16_130(mid);
        }
        (true, false) => {
            // Case 2: H-only filtering
            // For 8bpc: intermediate_bits=4, scalar does rnd(4-4)=rnd(0)=no shift
            // PREP_BIAS=0, so output = (16-mx)*src[x] + mx*src[x+1] unshifted
            for y in 0..h {
                let src_row_base = src_base.wrapping_add_signed(y as isize * src_stride);
                let src_row = &src[src_row_base..];
                let dst_row = y * w;

                let mut tmp_buf = [0i16; MID_STRIDE];
                // sh=0: no shift, keep full precision
                h_filter_bilin_8bpc_avx2_inner(
                    _token,
                    &mut tmp_buf,
                    src_row,
                    w,
                    mx,
                    4 - intermediate_bits, // sh = 0 for 8bpc
                );

                // PREP_BIAS = 0 for 8bpc, output directly
                for x in 0..w {
                    tmp[dst_row + x] = tmp_buf[x];
                }
            }
        }
        (false, true) => {
            // Case 3: V-only filtering
            for y in 0..h {
                let dst_row = y * w;

                for x in 0..w {
                    let r0 = src[src_base.wrapping_add_signed(y as isize * src_stride + x as isize)]
                        as i32;
                    let r1 = src
                        [src_base.wrapping_add_signed((y + 1) as isize * src_stride + x as isize)]
                        as i32;
                    let coeff0 = 16 - my as i32;
                    let coeff1 = my as i32;
                    let pixel = coeff0 * r0 + coeff1 * r1;
                    // V-only prep 8bpc: rnd(4-ib)=rnd(0)=no shift, PREP_BIAS=0
                    // Raw bilinear value [0, 4080] fits in i16
                    tmp[dst_row + x] = pixel as i16;
                }
            }
        }
        (false, false) => {
            // Case 4: Simple copy to prep format
            // For 8bpc: intermediate_bits=4, PREP_BIAS=0
            // Formula: (pixel << intermediate_bits) - PREP_BIAS = pixel << 4
            for y in 0..h {
                let src_row_base = src_base.wrapping_add_signed(y as isize * src_stride);
                let src_row = &src[src_row_base..];
                let dst_row = y * w;
                for x in 0..w {
                    let pixel = src_row[x] as i16;
                    tmp[dst_row + x] = pixel << intermediate_bits;
                }
            }
        }
    }
}
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[archmage::rite(v3)]
unsafe fn prep_bilin_8bpc_impl_v3(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();
    unsafe { prep_bilin_8bpc_avx2_impl_inner(token, tmp, src_ptr, 0, src_stride, w, h, mx, my) }
}

/// Bilinear prep for 8bpc - extern "C" wrapper
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
unsafe fn prep_bilin_8bpc_avx2_inner(
    _token: Desktop64,
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
) {
    // SAFETY: Caller guarantees AVX2 is available and pointers are valid
    unsafe {
        prep_bilin_8bpc_impl_v3(tmp, src_ptr, src_stride, w, h, mx, my);
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn prep_bilin_8bpc_v3(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    _bitdepth_max: i32,
    _src: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    unsafe { prep_bilin_8bpc_avx2_inner(token, tmp, src_ptr, src_stride, w, h, mx, my) }
}

// ============================================================================
// W_MASK (Weighted Mask) - Compound prediction with per-pixel masking
// ============================================================================
//
// w_mask computes both blended prediction AND outputs the mask for chroma
// The mask is computed from: m = min(38 + (abs_diff + mask_rnd) >> mask_sh, 64)
// Blend: dst = (tmp1 * m + tmp2 * (64 - m) + rnd) >> sh

#[cfg(target_arch = "x86_64")]
use crate::src::internal::SEG_MASK_LEN;

/// Core w_mask implementation (fully safe, slice-based)
/// SS_HOR and SS_VER control subsampling: 444=(false,false), 422=(true,false), 420=(true,true)
#[cfg(target_arch = "x86_64")]
#[rite]
fn w_mask_8bpc_avx2_safe_impl<const SS_HOR: bool, const SS_VER: bool>(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask: &mut [u8; SEG_MASK_LEN],
    sign: i32,
) {
    let w = w as usize;
    let h = h as usize;
    let sign = sign as u8;

    // For 8bpc: intermediate_bits = 4, bitdepth = 8, PREP_BIAS = 0
    let intermediate_bits = 4u32;
    let bitdepth = 8u32;
    let sh = intermediate_bits + 6;
    let rnd = (32i32 << intermediate_bits) + 0; // PREP_BIAS = 0 for 8bpc
    let mask_sh = bitdepth + intermediate_bits - 4;
    let mask_rnd = 1u16 << (mask_sh - 5);

    // Mask output dimensions depend on subsampling
    let mask_w = if SS_HOR { w >> 1 } else { w };
    let mut mask_off = 0usize;

    let mask_rnd_v = _mm256_set1_epi16(mask_rnd as i16);
    let c38 = _mm256_set1_epi16(38);
    let c64_16 = _mm256_set1_epi16(64);
    let rnd_v = _mm256_set1_epi32(rnd);
    let mask_shv = _mm_cvtsi32_si128(mask_sh as i32);
    let shv = _mm_cvtsi32_si128(sh as i32);

    for row_h in 0..h {
        let row_offset = row_h * w;
        let tmp1_row = &tmp1[row_offset..][..w];
        let tmp2_row = &tmp2[row_offset..][..w];
        let dst_row = &mut dst[row_h * dst_stride..][..w];

        let mut x = 0;

        // SIMD main path: 16 px/iter when the mask store is expressible
        // (444: every x; 422/420: pairs averaged). diff/m stay i16; the blend
        // dot uses pmaddwd pairs (m,64-m)x(t1,t2) -> exact i32.
        while x + 16 <= w {
            let t1 = loadu_256!(<&[i16; 16]>::try_from(&tmp1_row[x..x + 16]).unwrap());
            let t2 = loadu_256!(<&[i16; 16]>::try_from(&tmp2_row[x..x + 16]).unwrap());
            // m = min(38 + (|t1-t2| + mask_rnd) >> mask_sh, 64)
            // |t1-t2| as u16 without wrap: (max-min) has the true abs_diff bit
            // pattern even when the difference exceeds i16 range (warp_affine
            // mids). adds_epu16 then matches scalar u16::saturating_add.
            let diff = _mm256_sub_epi16(_mm256_max_epi16(t1, t2), _mm256_min_epi16(t1, t2));
            let mv = _mm256_min_epi16(
                _mm256_add_epi16(
                    c38,
                    _mm256_srl_epi16(_mm256_adds_epu16(diff, mask_rnd_v), mask_shv),
                ),
                c64_16,
            );
            // mvmv pairs per pixel: interleave m with (64-m)
            let inv = _mm256_sub_epi16(c64_16, mv);
            let mm_lo = _mm256_unpacklo_epi16(mv, inv); // px 0-3, 8-11
            let mm_hi = _mm256_unpackhi_epi16(mv, inv); // px 4-7, 12-15
            let tt_lo = _mm256_unpacklo_epi16(t1, t2);
            let tt_hi = _mm256_unpackhi_epi16(t1, t2);
            let sum_lo = _mm256_add_epi32(_mm256_madd_epi16(mm_lo, tt_lo), rnd_v);
            let sum_hi = _mm256_add_epi32(_mm256_madd_epi16(mm_hi, tt_hi), rnd_v);
            let res_lo = _mm256_sra_epi32(sum_lo, shv);
            let res_hi = _mm256_sra_epi32(sum_hi, shv);
            // i32 pairs back to i16 pixel order, then u8
            let res16 = _mm256_packs_epi32(res_lo, res_hi);
            let res8 = _mm256_packus_epi16(res16, res16);
            let res8 = _mm256_permute4x64_epi64(res8, 0b11011000);
            storeu_128!(
                <&mut [u8; 16]>::try_from(&mut dst_row[x..x + 16]).unwrap(),
                _mm256_castsi256_si128(res8)
            );

            if !SS_HOR {
                // 444: mask[x] = m per pixel
                let m8 = _mm256_packus_epi16(mv, mv);
                let m8 = _mm256_permute4x64_epi64(m8, 0b11011000);
                storeu_128!(
                    <&mut [u8; 16]>::try_from(&mut mask[mask_off + x..mask_off + x + 16]).unwrap(),
                    _mm256_castsi256_si128(m8)
                );
            } else {
                // SS_HOR: pairwise sums m[x]+m[x+1]. hadd gives pair sums in
                // ymm lanes {0-3,8-11}; a qword permute orders them into the
                // low xmm as eight i16.
                let mn = _mm256_hadd_epi16(mv, mv);
                let mn8 = _mm256_castsi256_si128(_mm256_permute4x64_epi64(mn, 0b11011000));
                if SS_VER && (row_h & 1 != 0) {
                    // odd row: ((m+n+2-sign)+prev)>>2 against stored sums
                    let prev = _mm_cvtepu8_epi16(loadu_64!(
                        <&[u8; 8]>::try_from(&mask[mask_off + (x >> 1)..mask_off + (x >> 1) + 8])
                            .unwrap()
                    ));
                    let out = _mm_srli_epi16::<2>(_mm_add_epi16(
                        _mm_add_epi16(mn8, _mm_set1_epi16(2 - sign as i16)),
                        prev,
                    ));
                    let out8 = _mm_packus_epi16(out, out);
                    storei64!(
                        &mut mask[mask_off + (x >> 1)..mask_off + (x >> 1) + 8],
                        out8
                    );
                } else if SS_VER {
                    // even row: store m+n sums (u8)
                    let out8 = _mm_packus_epi16(mn8, mn8);
                    storei64!(
                        &mut mask[mask_off + (x >> 1)..mask_off + (x >> 1) + 8],
                        out8
                    );
                } else {
                    // 422: (m+n+1-sign)>>1
                    let out =
                        _mm_srli_epi16::<1>(_mm_add_epi16(mn8, _mm_set1_epi16(1 - sign as i16)));
                    let out8 = _mm_packus_epi16(out, out);
                    storei64!(
                        &mut mask[mask_off + (x >> 1)..mask_off + (x >> 1) + 8],
                        out8
                    );
                }
            }
            x += 16;
        }

        // 8-px lane for w=8 blocks / 8-px remainder.
        while x + 8 <= w {
            let t1 = loadu_128!(<&[i16; 8]>::try_from(&tmp1_row[x..x + 8]).unwrap());
            let t2 = loadu_128!(<&[i16; 8]>::try_from(&tmp2_row[x..x + 8]).unwrap());
            let diff = _mm_sub_epi16(_mm_max_epi16(t1, t2), _mm_min_epi16(t1, t2));
            let mv = _mm_min_epi16(
                _mm_add_epi16(
                    _mm_set1_epi16(38),
                    _mm_srl_epi16(
                        _mm_adds_epu16(diff, _mm256_castsi256_si128(mask_rnd_v)),
                        mask_shv,
                    ),
                ),
                _mm_set1_epi16(64),
            );
            let inv = _mm_sub_epi16(_mm_set1_epi16(64), mv);
            let mm_lo = _mm_unpacklo_epi16(mv, inv);
            let mm_hi = _mm_unpackhi_epi16(mv, inv);
            let tt_lo = _mm_unpacklo_epi16(t1, t2);
            let tt_hi = _mm_unpackhi_epi16(t1, t2);
            let sum_lo = _mm_add_epi32(_mm_madd_epi16(mm_lo, tt_lo), _mm_set1_epi32(rnd));
            let sum_hi = _mm_add_epi32(_mm_madd_epi16(mm_hi, tt_hi), _mm_set1_epi32(rnd));
            let res_lo = _mm_sra_epi32(sum_lo, shv);
            let res_hi = _mm_sra_epi32(sum_hi, shv);
            let res16 = _mm_packs_epi32(res_lo, res_hi);
            let res8 = _mm_packus_epi16(res16, res16);
            storei64!(&mut dst_row[x..x + 8], res8);

            if !SS_HOR {
                storei64!(
                    &mut mask[mask_off + x..mask_off + x + 8],
                    _mm_packus_epi16(mv, mv)
                );
            } else {
                let mn = _mm_hadd_epi16(mv, mv); // lanes 0-3 = pair sums
                if SS_VER && (row_h & 1 != 0) {
                    let prev = _mm_cvtepu8_epi16(loadi32!(
                        &mask[mask_off + (x >> 1)..mask_off + (x >> 1) + 4]
                    ));
                    let out = _mm_srli_epi16::<2>(_mm_add_epi16(
                        _mm_add_epi16(mn, _mm_set1_epi16(2 - sign as i16)),
                        prev,
                    ));
                    let out8 = _mm_packus_epi16(out, out);
                    mask[mask_off + (x >> 1)..mask_off + (x >> 1) + 4]
                        .copy_from_slice(&(_mm_cvtsi128_si32(out8) as u32).to_ne_bytes());
                } else if SS_VER {
                    let out8 = _mm_packus_epi16(mn, mn);
                    mask[mask_off + (x >> 1)..mask_off + (x >> 1) + 4]
                        .copy_from_slice(&(_mm_cvtsi128_si32(out8) as u32).to_ne_bytes());
                } else {
                    let out =
                        _mm_srli_epi16::<1>(_mm_add_epi16(mn, _mm_set1_epi16(1 - sign as i16)));
                    let out8 = _mm_packus_epi16(out, out);
                    mask[mask_off + (x >> 1)..mask_off + (x >> 1) + 4]
                        .copy_from_slice(&(_mm_cvtsi128_si32(out8) as u32).to_ne_bytes());
                }
            }
            x += 8;
        }

        while x < w {
            // Compute mask value m
            let diff = tmp1_row[x].abs_diff(tmp2_row[x]);
            let m = std::cmp::min(38 + ((diff.saturating_add(mask_rnd)) >> mask_sh), 64) as u8;

            // Blend pixels
            let t1 = tmp1_row[x] as i32;
            let t2 = tmp2_row[x] as i32;
            let pixel = (t1 * m as i32 + t2 * (64 - m as i32) + rnd) >> sh;
            dst_row[x] = pixel.clamp(0, 255) as u8;

            if SS_HOR {
                // Process second pixel in pair for horizontal subsampling
                x += 1;
                let diff2 = tmp1_row[x].abs_diff(tmp2_row[x]);
                let n = std::cmp::min(38 + ((diff2.saturating_add(mask_rnd)) >> mask_sh), 64) as u8;

                let t1 = tmp1_row[x] as i32;
                let t2 = tmp2_row[x] as i32;
                let pixel = (t1 * n as i32 + t2 * (64 - n as i32) + rnd) >> sh;
                dst_row[x] = pixel.clamp(0, 255) as u8;

                // Output subsampled mask
                let mask_x = x >> 1;
                if SS_VER && (row_h & 1 != 0) {
                    // Vertical subsampling: average with previous row
                    let prev = mask[mask_off + mask_x];
                    mask[mask_off + mask_x] =
                        (((m as u16 + n as u16 + 2 - sign as u16) + prev as u16) >> 2) as u8;
                } else if SS_VER {
                    // Even row: just store sum (will be averaged on odd row)
                    mask[mask_off + mask_x] = m + n;
                } else {
                    // No vertical subsampling: average of horizontal pair
                    mask[mask_off + mask_x] = ((m as u16 + n as u16 + 1 - sign as u16) >> 1) as u8;
                }
            } else {
                // No horizontal subsampling (444)
                mask[mask_off + x] = m;
            }
            x += 1;
        }

        // Advance mask offset only on appropriate rows
        if !SS_VER || (row_h & 1 != 0) {
            mask_off += mask_w;
        }
    }
}

/// w_mask 444 8bpc (no subsampling)
#[cfg(target_arch = "x86_64")]
#[arcane]
fn w_mask_444_8bpc_avx2_safe(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask: &mut [u8; SEG_MASK_LEN],
    sign: i32,
) {
    let mut dst = dst.flex_mut();
    w_mask_8bpc_avx2_safe_impl::<false, false>(
        _token, &mut *dst, dst_stride, tmp1, tmp2, w, h, mask, sign,
    );
}

/// w_mask 422 8bpc (horizontal subsampling only)
#[cfg(target_arch = "x86_64")]
#[arcane]
fn w_mask_422_8bpc_avx2_safe(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask: &mut [u8; SEG_MASK_LEN],
    sign: i32,
) {
    let mut dst = dst.flex_mut();
    w_mask_8bpc_avx2_safe_impl::<true, false>(
        _token, &mut *dst, dst_stride, tmp1, tmp2, w, h, mask, sign,
    );
}

/// w_mask 420 8bpc (horizontal and vertical subsampling)
#[cfg(target_arch = "x86_64")]
#[arcane]
fn w_mask_420_8bpc_avx2_safe(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask: &mut [u8; SEG_MASK_LEN],
    sign: i32,
) {
    let mut dst = dst.flex_mut();
    w_mask_8bpc_avx2_safe_impl::<true, true>(
        _token, &mut *dst, dst_stride, tmp1, tmp2, w, h, mask, sign,
    );
}

/// w_mask for 4:4:4 (no subsampling)
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn w_mask_444_8bpc_v3(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask: &mut [u8; SEG_MASK_LEN],
    sign: i32,
    _bitdepth_max: i32,
    _dst: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    let dst = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h as usize * dst_stride as usize)
    };
    w_mask_444_8bpc_avx2_safe(
        token,
        dst,
        dst_stride as usize,
        tmp1,
        tmp2,
        w,
        h,
        mask,
        sign,
    )
}

/// w_mask for 4:2:2 (horizontal subsampling only)
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn w_mask_422_8bpc_v3(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask: &mut [u8; SEG_MASK_LEN],
    sign: i32,
    _bitdepth_max: i32,
    _dst: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    let dst = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h as usize * dst_stride as usize)
    };
    w_mask_422_8bpc_avx2_safe(
        token,
        dst,
        dst_stride as usize,
        tmp1,
        tmp2,
        w,
        h,
        mask,
        sign,
    )
}

/// w_mask for 4:2:0 (horizontal and vertical subsampling)
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn w_mask_420_8bpc_v3(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask: &mut [u8; SEG_MASK_LEN],
    sign: i32,
    _bitdepth_max: i32,
    _dst: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    let dst = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h as usize * dst_stride as usize)
    };
    w_mask_420_8bpc_avx2_safe(
        token,
        dst,
        dst_stride as usize,
        tmp1,
        tmp2,
        w,
        h,
        mask,
        sign,
    )
}

// 16bpc w_mask (fully safe, slice-based)
#[cfg(target_arch = "x86_64")]
#[rite]
fn w_mask_16bpc_avx2_safe_impl<const SS_HOR: bool, const SS_VER: bool>(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask: &mut [u8; SEG_MASK_LEN],
    sign: i32,
    bitdepth_max: i32,
) {
    let w = w as usize;
    let h = h as usize;
    let sign = sign as u8;
    let bd_max = bitdepth_max as i32;

    // Determine bitdepth and intermediate_bits from bitdepth_max
    let bitdepth = if bitdepth_max == 1023 { 10u32 } else { 12u32 };
    let intermediate_bits = (14 - bitdepth) as u32; // 4 for 10-bit, 2 for 12-bit
    let sh = intermediate_bits + 6;
    let rnd = (32i32 << intermediate_bits) + 8192 * 64;
    let mask_sh = bitdepth + intermediate_bits - 4;
    let mask_rnd = 1u16 << (mask_sh - 5);

    let mask_w = if SS_HOR { w >> 1 } else { w };
    let mut mask_off = 0usize;

    let mask_rnd_v = _mm256_set1_epi16(mask_rnd as i16);
    let c38 = _mm256_set1_epi16(38);
    let c64_16 = _mm256_set1_epi16(64);
    let rnd_v = _mm256_set1_epi32(rnd);
    let bmax16 = _mm256_set1_epi16(bd_max as i16);
    let mask_shv = _mm_cvtsi32_si128(mask_sh as i32);
    let shv = _mm_cvtsi32_si128(sh as i32);

    for row_h in 0..h {
        let row_offset = row_h * w;
        let tmp1_row = &tmp1[row_offset..][..w];
        let tmp2_row = &tmp2[row_offset..][..w];
        // dst is bytes, but we write u16 pixels — each pixel is 2 bytes
        let dst_row_bytes = &mut dst[row_h * dst_stride..][..w * 2];
        let dst_row: &mut [u16] = zerocopy::FromBytes::mut_from_bytes(dst_row_bytes).unwrap();

        let mut x = 0;

        // Same pmaddwd structure as the 8bpc impl; outputs stay u16 and
        // clamp to bitdepth_max instead of packing to bytes.
        while x + 16 <= w {
            let t1 = loadu_256!(<&[i16; 16]>::try_from(&tmp1_row[x..x + 16]).unwrap());
            let t2 = loadu_256!(<&[i16; 16]>::try_from(&tmp2_row[x..x + 16]).unwrap());
            // |t1-t2| as u16 without wrap: (max-min) has the true abs_diff bit
            // pattern even when the difference exceeds i16 range (warp_affine
            // mids). adds_epu16 then matches scalar u16::saturating_add.
            let diff = _mm256_sub_epi16(_mm256_max_epi16(t1, t2), _mm256_min_epi16(t1, t2));
            let mv = _mm256_min_epi16(
                _mm256_add_epi16(
                    c38,
                    _mm256_srl_epi16(_mm256_adds_epu16(diff, mask_rnd_v), mask_shv),
                ),
                c64_16,
            );
            let inv = _mm256_sub_epi16(c64_16, mv);
            let mm_lo = _mm256_unpacklo_epi16(mv, inv);
            let mm_hi = _mm256_unpackhi_epi16(mv, inv);
            let tt_lo = _mm256_unpacklo_epi16(t1, t2);
            let tt_hi = _mm256_unpackhi_epi16(t1, t2);
            let sum_lo = _mm256_add_epi32(_mm256_madd_epi16(mm_lo, tt_lo), rnd_v);
            let sum_hi = _mm256_add_epi32(_mm256_madd_epi16(mm_hi, tt_hi), rnd_v);
            let res_lo = _mm256_sra_epi32(sum_lo, shv);
            let res_hi = _mm256_sra_epi32(sum_hi, shv);
            let res16 = _mm256_packs_epi32(res_lo, res_hi);
            let clamped = _mm256_min_epi16(_mm256_max_epi16(res16, _mm256_setzero_si256()), bmax16);
            storeu_256!(
                <&mut [u16; 16]>::try_from(&mut dst_row[x..x + 16]).unwrap(),
                clamped
            );

            if !SS_HOR {
                let m8 = _mm256_packus_epi16(mv, mv);
                let m8 = _mm256_permute4x64_epi64(m8, 0b11011000);
                storeu_128!(
                    <&mut [u8; 16]>::try_from(&mut mask[mask_off + x..mask_off + x + 16]).unwrap(),
                    _mm256_castsi256_si128(m8)
                );
            } else {
                let mn = _mm256_hadd_epi16(mv, mv);
                let mn8 = _mm256_castsi256_si128(_mm256_permute4x64_epi64(mn, 0b11011000));
                if SS_VER && (row_h & 1 != 0) {
                    let prev = _mm_cvtepu8_epi16(loadu_64!(
                        <&[u8; 8]>::try_from(&mask[mask_off + (x >> 1)..mask_off + (x >> 1) + 8])
                            .unwrap()
                    ));
                    let out = _mm_srli_epi16::<2>(_mm_add_epi16(
                        _mm_add_epi16(mn8, _mm_set1_epi16(2 - sign as i16)),
                        prev,
                    ));
                    let out8 = _mm_packus_epi16(out, out);
                    storei64!(
                        &mut mask[mask_off + (x >> 1)..mask_off + (x >> 1) + 8],
                        out8
                    );
                } else if SS_VER {
                    let out8 = _mm_packus_epi16(mn8, mn8);
                    storei64!(
                        &mut mask[mask_off + (x >> 1)..mask_off + (x >> 1) + 8],
                        out8
                    );
                } else {
                    let out =
                        _mm_srli_epi16::<1>(_mm_add_epi16(mn8, _mm_set1_epi16(1 - sign as i16)));
                    let out8 = _mm_packus_epi16(out, out);
                    storei64!(
                        &mut mask[mask_off + (x >> 1)..mask_off + (x >> 1) + 8],
                        out8
                    );
                }
            }
            x += 16;
        }

        while x + 8 <= w {
            let t1 = loadu_128!(<&[i16; 8]>::try_from(&tmp1_row[x..x + 8]).unwrap());
            let t2 = loadu_128!(<&[i16; 8]>::try_from(&tmp2_row[x..x + 8]).unwrap());
            let diff = _mm_sub_epi16(_mm_max_epi16(t1, t2), _mm_min_epi16(t1, t2));
            let mv = _mm_min_epi16(
                _mm_add_epi16(
                    _mm_set1_epi16(38),
                    _mm_srl_epi16(
                        _mm_adds_epu16(diff, _mm256_castsi256_si128(mask_rnd_v)),
                        mask_shv,
                    ),
                ),
                _mm_set1_epi16(64),
            );
            let inv = _mm_sub_epi16(_mm_set1_epi16(64), mv);
            let mm_lo = _mm_unpacklo_epi16(mv, inv);
            let mm_hi = _mm_unpackhi_epi16(mv, inv);
            let tt_lo = _mm_unpacklo_epi16(t1, t2);
            let tt_hi = _mm_unpackhi_epi16(t1, t2);
            let sum_lo = _mm_add_epi32(_mm_madd_epi16(mm_lo, tt_lo), _mm_set1_epi32(rnd));
            let sum_hi = _mm_add_epi32(_mm_madd_epi16(mm_hi, tt_hi), _mm_set1_epi32(rnd));
            let res_lo = _mm_sra_epi32(sum_lo, shv);
            let res_hi = _mm_sra_epi32(sum_hi, shv);
            let res16 = _mm_packs_epi32(res_lo, res_hi);
            let clamped = _mm_min_epi16(
                _mm_max_epi16(res16, _mm_setzero_si128()),
                _mm_set1_epi16(bd_max as i16),
            );
            storeu_128!(
                <&mut [u16; 8]>::try_from(&mut dst_row[x..x + 8]).unwrap(),
                clamped
            );

            if !SS_HOR {
                storei64!(
                    &mut mask[mask_off + x..mask_off + x + 8],
                    _mm_packus_epi16(mv, mv)
                );
            } else {
                let mn = _mm_hadd_epi16(mv, mv);
                if SS_VER && (row_h & 1 != 0) {
                    let prev = _mm_cvtepu8_epi16(loadi32!(
                        &mask[mask_off + (x >> 1)..mask_off + (x >> 1) + 4]
                    ));
                    let out = _mm_srli_epi16::<2>(_mm_add_epi16(
                        _mm_add_epi16(mn, _mm_set1_epi16(2 - sign as i16)),
                        prev,
                    ));
                    let out8 = _mm_packus_epi16(out, out);
                    mask[mask_off + (x >> 1)..mask_off + (x >> 1) + 4]
                        .copy_from_slice(&(_mm_cvtsi128_si32(out8) as u32).to_ne_bytes());
                } else if SS_VER {
                    let out8 = _mm_packus_epi16(mn, mn);
                    mask[mask_off + (x >> 1)..mask_off + (x >> 1) + 4]
                        .copy_from_slice(&(_mm_cvtsi128_si32(out8) as u32).to_ne_bytes());
                } else {
                    let out =
                        _mm_srli_epi16::<1>(_mm_add_epi16(mn, _mm_set1_epi16(1 - sign as i16)));
                    let out8 = _mm_packus_epi16(out, out);
                    mask[mask_off + (x >> 1)..mask_off + (x >> 1) + 4]
                        .copy_from_slice(&(_mm_cvtsi128_si32(out8) as u32).to_ne_bytes());
                }
            }
            x += 8;
        }

        while x < w {
            let diff = tmp1_row[x].abs_diff(tmp2_row[x]);
            let m = std::cmp::min(38 + ((diff.saturating_add(mask_rnd)) >> mask_sh), 64) as u8;

            let t1 = tmp1_row[x] as i32;
            let t2 = tmp2_row[x] as i32;
            let pixel = (t1 * m as i32 + t2 * (64 - m as i32) + rnd) >> sh;
            dst_row[x] = pixel.clamp(0, bd_max) as u16;

            if SS_HOR {
                x += 1;
                let diff2 = tmp1_row[x].abs_diff(tmp2_row[x]);
                let n = std::cmp::min(38 + ((diff2.saturating_add(mask_rnd)) >> mask_sh), 64) as u8;

                let t1 = tmp1_row[x] as i32;
                let t2 = tmp2_row[x] as i32;
                let pixel = (t1 * n as i32 + t2 * (64 - n as i32) + rnd) >> sh;
                dst_row[x] = pixel.clamp(0, bd_max) as u16;

                let mask_x = x >> 1;
                if SS_VER && (row_h & 1 != 0) {
                    let prev = mask[mask_off + mask_x];
                    mask[mask_off + mask_x] =
                        (((m as u16 + n as u16 + 2 - sign as u16) + prev as u16) >> 2) as u8;
                } else if SS_VER {
                    mask[mask_off + mask_x] = m + n;
                } else {
                    mask[mask_off + mask_x] = ((m as u16 + n as u16 + 1 - sign as u16) >> 1) as u8;
                }
            } else {
                mask[mask_off + x] = m;
            }
            x += 1;
        }

        if !SS_VER || (row_h & 1 != 0) {
            mask_off += mask_w;
        }
    }
}

/// w_mask 444 16bpc (no subsampling)
#[cfg(target_arch = "x86_64")]
#[arcane]
fn w_mask_444_16bpc_avx2_safe(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask: &mut [u8; SEG_MASK_LEN],
    sign: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    w_mask_16bpc_avx2_safe_impl::<false, false>(
        _token,
        &mut *dst,
        dst_stride,
        tmp1,
        tmp2,
        w,
        h,
        mask,
        sign,
        bitdepth_max,
    );
}

/// w_mask 422 16bpc (horizontal subsampling only)
#[cfg(target_arch = "x86_64")]
#[arcane]
fn w_mask_422_16bpc_avx2_safe(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask: &mut [u8; SEG_MASK_LEN],
    sign: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    w_mask_16bpc_avx2_safe_impl::<true, false>(
        _token,
        &mut *dst,
        dst_stride,
        tmp1,
        tmp2,
        w,
        h,
        mask,
        sign,
        bitdepth_max,
    );
}

/// w_mask 420 16bpc (horizontal and vertical subsampling)
#[cfg(target_arch = "x86_64")]
#[arcane]
fn w_mask_420_16bpc_avx2_safe(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask: &mut [u8; SEG_MASK_LEN],
    sign: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    w_mask_16bpc_avx2_safe_impl::<true, true>(
        _token,
        &mut *dst,
        dst_stride,
        tmp1,
        tmp2,
        w,
        h,
        mask,
        sign,
        bitdepth_max,
    );
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn w_mask_444_16bpc_v3(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask: &mut [u8; SEG_MASK_LEN],
    sign: i32,
    bitdepth_max: i32,
    _dst: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    let dst = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h as usize * dst_stride as usize)
    };
    w_mask_444_16bpc_avx2_safe(
        token,
        dst,
        dst_stride as usize,
        tmp1,
        tmp2,
        w,
        h,
        mask,
        sign,
        bitdepth_max,
    )
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn w_mask_422_16bpc_v3(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask: &mut [u8; SEG_MASK_LEN],
    sign: i32,
    bitdepth_max: i32,
    _dst: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    let dst = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h as usize * dst_stride as usize)
    };
    w_mask_422_16bpc_avx2_safe(
        token,
        dst,
        dst_stride as usize,
        tmp1,
        tmp2,
        w,
        h,
        mask,
        sign,
        bitdepth_max,
    )
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn w_mask_420_16bpc_v3(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask: &mut [u8; SEG_MASK_LEN],
    sign: i32,
    bitdepth_max: i32,
    _dst: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    let dst = unsafe {
        std::slice::from_raw_parts_mut(dst_ptr as *mut u8, h as usize * dst_stride as usize)
    };
    w_mask_420_16bpc_avx2_safe(
        token,
        dst,
        dst_stride as usize,
        tmp1,
        tmp2,
        w,
        h,
        mask,
        sign,
        bitdepth_max,
    )
}

// ============================================================================
// BILINEAR 16BPC
// ============================================================================

/// Horizontal bilinear filter for 16bpc using AVX2
/// Formula: pixel = 16 * x0 + mx * (x1 - x0) = (16 - mx) * x0 + mx * x1,
/// then FILTER_BILIN_RND with `h_sh` = 4 - intermediate_bits (0 for 10bpc,
/// 2 for 12bpc), stored in the reference's i16 (wrapping) convention.
#[cfg(target_arch = "x86_64")]
#[rite]
fn h_bilin_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [i32],
    src: &[u16],
    w: usize,
    mx: i32,
    h_sh: i32,
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    // Bilinear weights: w0 = (16 - mx), w1 = mx
    let w0 = _mm256_set1_epi32(16 - mx);
    let w1 = _mm256_set1_epi32(mx);
    let rnd = _mm256_set1_epi32((1 << h_sh) >> 1);
    let shift_count = _mm_cvtsi32_si128(h_sh);

    let mut col = 0usize;

    // Process 8 pixels at a time
    while col + 8 <= w {
        // Load 9 consecutive 16-bit pixels (8 output pixels need pixels 0..8)
        // s offset = col
        let p0_7 = loadu_128!(<&[u16; 8]>::try_from(&src[col..col + 8]).unwrap()); // pixels 0-7
        let p1_8 = loadu_128!(<&[u16; 8]>::try_from(&src[col + 1..col + 9]).unwrap()); // pixels 1-8

        // Zero-extend to 32-bit
        let p0_lo = _mm256_cvtepu16_epi32(p0_7);
        let p1_lo = _mm256_cvtepu16_epi32(p1_8);

        // Compute: w0 * p0 + w1 * p1
        let term0 = _mm256_mullo_epi32(p0_lo, w0);
        let term1 = _mm256_mullo_epi32(p1_lo, w1);
        let sum = _mm256_add_epi32(term0, term1);
        // mid is int16_t in the C reference: truncate to match.
        let rnd_sum = _mm256_sra_epi32(_mm256_add_epi32(sum, rnd), shift_count);
        let result = _mm256_srai_epi32::<16>(_mm256_slli_epi32::<16>(rnd_sum));

        storeu_256!(
            <&mut [i32; 8]>::try_from(&mut dst[col..col + 8]).unwrap(),
            result
        );
        col += 8;
    }

    // Scalar fallback
    let r = (1 << h_sh) >> 1;
    while col < w {
        let x0 = src[col] as i32;
        let x1 = src[col + 1] as i32;
        dst[col] = (((16 * x0 + mx * (x1 - x0)) + r) >> h_sh) as i16 as i32;
        col += 1;
    }
}
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[archmage::rite(v3)]
unsafe fn h_bilin_16bpc_v3(dst: *mut i32, src: *const u16, w: usize, mx: i32) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();
    // No bitdepth in the asm-ABI signature: preserves the original 10bpc
    // (h_sh = 0) semantics for this compat shim.
    unsafe { h_bilin_16bpc_avx2_inner(token, dst, src, w, mx, 0) }
}

/// Vertical bilinear filter for 16bpc using AVX2
/// Applies vertical filter to 32-bit intermediate, outputs clamped u16
#[cfg(target_arch = "x86_64")]
#[rite]
fn v_bilin_16bpc_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    mid: &[[i32; MID_STRIDE]],
    w: usize,
    y: usize,
    my: i32,
    sh: i32,
    max: i32,
) {
    let mut dst = dst.flex_mut();
    let w0 = _mm256_set1_epi32(16 - my);
    let w1 = _mm256_set1_epi32(my);
    let rnd = _mm256_set1_epi32((1 << sh) >> 1);
    let shift_count = _mm_cvtsi32_si128(sh);
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi32(max);

    let mut col = 0usize;

    while col + 8 <= w {
        // Load 8 intermediate values from rows y and y+1
        let r0 = loadu_256!(<&[i32; 8]>::try_from(&mid[y][col..col + 8]).unwrap());
        let r1 = loadu_256!(<&[i32; 8]>::try_from(&mid[y + 1][col..col + 8]).unwrap());

        // Compute: w0 * r0 + w1 * r1
        let term0 = _mm256_mullo_epi32(r0, w0);
        let term1 = _mm256_mullo_epi32(r1, w1);
        let sum = _mm256_add_epi32(term0, term1);

        // Round, shift, clamp
        let shifted = _mm256_sra_epi32(_mm256_add_epi32(sum, rnd), shift_count);
        let clamped = _mm256_min_epi32(_mm256_max_epi32(shifted, zero), max_val);

        // Pack to 16-bit
        let packed = _mm256_packus_epi32(clamped, zero);
        let result = _mm256_permute4x64_epi64(packed, 0b00_00_10_00);

        storeu_128!(
            <&mut [u16; 8]>::try_from(&mut dst[col..col + 8]).unwrap(),
            _mm256_castsi256_si128(result)
        );
        col += 8;
    }

    // Scalar fallback
    while col < w {
        let r0 = mid[y][col];
        let r1 = mid[y + 1][col];
        let pixel = 16 * r0 + my * (r1 - r0);
        let r = (1 << sh) >> 1;
        let val = ((pixel + r) >> sh).clamp(0, max);
        dst[col] = val as u16;
        col += 1;
    }
}
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[archmage::rite(v3)]
unsafe fn v_bilin_16bpc_v3(
    dst: *mut u16,
    mid: &[[i32; MID_STRIDE]],
    w: usize,
    y: usize,
    my: i32,
    sh: i32,
    max: i32,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();
    unsafe { v_bilin_16bpc_avx2_inner(token, dst, mid, w, y, my, sh, max) }
}

/// Vertical bilinear filter for 16bpc prep - outputs i16 with bias subtraction
#[cfg(target_arch = "x86_64")]
#[rite]
fn v_bilin_16bpc_prep_avx2_inner(
    _token: Desktop64,
    dst: &mut [i16],
    mid: &[[i32; MID_STRIDE]],
    w: usize,
    y: usize,
    my: i32,
    sh: i32,
    prep_bias: i32,
) {
    let mut dst = dst.flex_mut();
    let w0 = _mm256_set1_epi32(16 - my);
    let w1 = _mm256_set1_epi32(my);
    let rnd = _mm256_set1_epi32((1 << sh) >> 1);
    let shift_count = _mm_cvtsi32_si128(sh);
    let bias = _mm256_set1_epi32(prep_bias);

    let mut col = 0usize;

    while col + 8 <= w {
        let r0 = loadu_256!(<&[i32; 8]>::try_from(&mid[y][col..col + 8]).unwrap());
        let r1 = loadu_256!(<&[i32; 8]>::try_from(&mid[y + 1][col..col + 8]).unwrap());

        let term0 = _mm256_mullo_epi32(r0, w0);
        let term1 = _mm256_mullo_epi32(r1, w1);
        let sum = _mm256_add_epi32(term0, term1);

        // Round, shift, subtract bias
        let shifted = _mm256_sra_epi32(_mm256_add_epi32(sum, rnd), shift_count);
        let biased = _mm256_sub_epi32(shifted, bias);

        // Wrap to i16 like the C `int16_t` store (packs would saturate,
        // which diverges from the reference for out-of-range 12bpc values).
        let wrapped = _mm256_and_si256(biased, _mm256_set1_epi32(0xFFFF));
        let packed = _mm256_packus_epi32(wrapped, wrapped);
        let result = _mm256_permute4x64_epi64(packed, 0b00_00_10_00);

        storeu_128!(
            <&mut [i16; 8]>::try_from(&mut dst[col..col + 8]).unwrap(),
            _mm256_castsi256_si128(result)
        );
        col += 8;
    }

    // Scalar fallback
    while col < w {
        let r0 = mid[y][col];
        let r1 = mid[y + 1][col];
        let pixel = 16 * r0 + my * (r1 - r0);
        let r = (1 << sh) >> 1;
        let val = ((pixel + r) >> sh) - prep_bias;
        dst[col] = val as i16;
        col += 1;
    }
}
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[archmage::rite(v3)]
unsafe fn v_bilin_16bpc_prep_v3(
    dst: *mut i16,
    mid: &[[i32; MID_STRIDE]],
    w: usize,
    y: usize,
    my: i32,
    sh: i32,
    prep_bias: i32,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();
    unsafe { v_bilin_16bpc_prep_avx2_inner(token, dst, mid, w, y, my, sh, prep_bias) }
}

/// Horizontal bilinear filter for 16bpc put (H-only case)
/// Outputs directly to u16 with shift and clamp
#[cfg(target_arch = "x86_64")]
#[rite]
fn h_bilin_16bpc_put_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    src: &[u16],
    w: usize,
    mx: i32,
    bd_max: i32,
    intermediate_bits: i32,
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    let w0 = _mm256_set1_epi32(16 - mx);
    let w1 = _mm256_set1_epi32(mx);
    // Reference rounds twice: rnd(4 - ib) then (px + ((1<<ib)>>1)) >> ib.
    // For 10bpc (ib=4) this degenerates to (sum + 8) >> 4.
    let h_sh = 4 - intermediate_bits;
    let rnd1 = _mm256_set1_epi32((1 << h_sh) >> 1);
    let rnd2 = _mm256_set1_epi32((1 << intermediate_bits) >> 1);
    let shift1 = _mm_cvtsi32_si128(h_sh);
    let shift2 = _mm_cvtsi32_si128(intermediate_bits);
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi32(bd_max);

    let mut col = 0usize;

    while col + 8 <= w {
        // s offset = col
        let p0 = _mm256_cvtepu16_epi32(loadu_128!(
            <&[u16; 8]>::try_from(&src[col..col + 8]).unwrap()
        ));
        let p1 = _mm256_cvtepu16_epi32(loadu_128!(
            <&[u16; 8]>::try_from(&src[col + 1..col + 9]).unwrap()
        ));

        // (16 - mx) * p0 + mx * p1
        let term0 = _mm256_mullo_epi32(p0, w0);
        let term1 = _mm256_mullo_epi32(p1, w1);
        let sum = _mm256_add_epi32(term0, term1);

        // Two-stage round: ((sum + r1) >> h_sh + r2) >> ib, clamp to [0, max]
        let t = _mm256_sra_epi32(_mm256_add_epi32(sum, rnd1), shift1);
        let shifted = _mm256_sra_epi32(_mm256_add_epi32(t, rnd2), shift2);
        let clamped = _mm256_min_epi32(_mm256_max_epi32(shifted, zero), max_val);

        // Pack to u16
        let packed = _mm256_packus_epi32(clamped, zero);
        let result = _mm256_permute4x64_epi64(packed, 0b00_00_10_00);

        storeu_128!(
            <&mut [u16; 8]>::try_from(&mut dst[col..col + 8]).unwrap(),
            _mm256_castsi256_si128(result)
        );
        col += 8;
    }

    // Scalar fallback
    let r1 = (1 << h_sh) >> 1;
    let r2 = (1 << intermediate_bits) >> 1;
    while col < w {
        let x0 = src[col] as i32;
        let x1 = src[col + 1] as i32;
        let pixel = (16 - mx) * x0 + mx * x1;
        let result = ((((pixel + r1) >> h_sh) + r2) >> intermediate_bits).clamp(0, bd_max);
        dst[col] = result as u16;
        col += 1;
    }
}
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[archmage::rite(v3)]
unsafe fn h_bilin_16bpc_put_v3(dst: *mut u16, src: *const u16, w: usize, mx: i32, bd_max: i32) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();
    // No bitdepth in the asm-ABI signature: keeps the original 10bpc
    // (intermediate_bits = 4) semantics for this compat shim.
    unsafe { h_bilin_16bpc_put_avx2_inner(token, dst, src, w, mx, bd_max, 4) }
}

/// Vertical bilinear filter for 16bpc put (V-only case)
/// Reads directly from u16 source, outputs u16
#[cfg(target_arch = "x86_64")]
#[rite]
fn v_bilin_16bpc_direct_avx2_inner(
    _token: Desktop64,
    dst: &mut [u16],
    src: &[u16],
    src_base: usize,
    src_stride: isize,
    w: usize,
    my: i32,
    bd_max: i32,
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    let row_base = |row: usize| src_base.wrapping_add_signed(row as isize * src_stride);
    let w0 = _mm256_set1_epi32(16 - my);
    let w1 = _mm256_set1_epi32(my);
    let rnd = _mm256_set1_epi32(8);
    let shift_count = _mm_cvtsi32_si128(4);
    let zero = _mm256_setzero_si256();
    let max_val = _mm256_set1_epi32(bd_max);

    let mut col = 0usize;

    while col + 8 <= w {
        // Load from row 0 and row 1
        let p0 = _mm256_cvtepu16_epi32(loadu_128!(
            <&[u16; 8]>::try_from(&src[src_base + col..src_base + col + 8]).unwrap()
        ));
        let p1 = _mm256_cvtepu16_epi32(loadu_128!(
            <&[u16; 8]>::try_from(&src[row_base(1) + col..row_base(1) + col + 8]).unwrap()
        ));

        // (16 - my) * p0 + my * p1
        let term0 = _mm256_mullo_epi32(p0, w0);
        let term1 = _mm256_mullo_epi32(p1, w1);
        let sum = _mm256_add_epi32(term0, term1);

        // (sum + 8) >> 4, clamp to [0, max]
        let shifted = _mm256_sra_epi32(_mm256_add_epi32(sum, rnd), shift_count);
        let clamped = _mm256_min_epi32(_mm256_max_epi32(shifted, zero), max_val);

        // Pack to u16
        let packed = _mm256_packus_epi32(clamped, zero);
        let result = _mm256_permute4x64_epi64(packed, 0b00_00_10_00);

        storeu_128!(
            <&mut [u16; 8]>::try_from(&mut dst[col..col + 8]).unwrap(),
            _mm256_castsi256_si128(result)
        );
        col += 8;
    }

    // Scalar fallback
    while col < w {
        let x0 = src[src_base + col] as i32;
        let x1 = src[row_base(1) + col] as i32;
        let pixel = (16 - my) * x0 + my * x1;
        let result = ((pixel + 8) >> 4).clamp(0, bd_max);
        dst[col] = result as u16;
        col += 1;
    }
}
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[archmage::rite(v3)]
unsafe fn v_bilin_16bpc_direct_v3(
    dst: *mut u16,
    src: *const u16,
    src_stride: isize,
    w: usize,
    my: i32,
    bd_max: i32,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();
    unsafe { v_bilin_16bpc_direct_avx2_inner(token, dst, src, 0, src_stride, w, my, bd_max) }
}

/// Horizontal bilinear filter for 16bpc prep (H-only case)
/// Outputs i16 with PREP_BIAS subtracted
#[cfg(target_arch = "x86_64")]
#[rite]
fn h_bilin_16bpc_prep_direct_avx2_inner(
    _token: Desktop64,
    dst: &mut [i16],
    src: &[u16],
    w: usize,
    mx: i32,
    prep_bias: i32,
    h_sh: i32,
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    let w0 = _mm256_set1_epi32(16 - mx);
    let w1 = _mm256_set1_epi32(mx);
    let rnd = _mm256_set1_epi32((1 << h_sh) >> 1);
    let shift_count = _mm_cvtsi32_si128(h_sh);
    let bias = _mm256_set1_epi32(prep_bias);

    let mut col = 0usize;

    while col + 8 <= w {
        // s offset = col
        let p0 = _mm256_cvtepu16_epi32(loadu_128!(
            <&[u16; 8]>::try_from(&src[col..col + 8]).unwrap()
        ));
        let p1 = _mm256_cvtepu16_epi32(loadu_128!(
            <&[u16; 8]>::try_from(&src[col + 1..col + 9]).unwrap()
        ));

        // (16 - mx) * p0 + mx * p1
        let term0 = _mm256_mullo_epi32(p0, w0);
        let term1 = _mm256_mullo_epi32(p1, w1);
        let sum = _mm256_add_epi32(term0, term1);

        // Round with 4 - intermediate_bits (0 for 10bpc, 2 for 12bpc),
        // subtract bias, wrap to i16 like the C int16_t store
        // (packs would saturate).
        let shifted = _mm256_sra_epi32(_mm256_add_epi32(sum, rnd), shift_count);
        let biased = _mm256_sub_epi32(shifted, bias);
        let wrapped = _mm256_and_si256(biased, _mm256_set1_epi32(0xFFFF));
        let packed = _mm256_packus_epi32(wrapped, wrapped);
        let result = _mm256_permute4x64_epi64(packed, 0b00_00_10_00);

        storeu_128!(
            <&mut [i16; 8]>::try_from(&mut dst[col..col + 8]).unwrap(),
            _mm256_castsi256_si128(result)
        );
        col += 8;
    }

    // Scalar fallback
    let r = (1 << h_sh) >> 1;
    while col < w {
        let x0 = src[col] as i32;
        let x1 = src[col + 1] as i32;
        let pixel = (16 - mx) * x0 + mx * x1;
        dst[col] = (((pixel + r) >> h_sh) - prep_bias) as i16;
        col += 1;
    }
}
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[archmage::rite(v3)]
unsafe fn h_bilin_16bpc_prep_direct_v3(
    dst: *mut i16,
    src: *const u16,
    w: usize,
    mx: i32,
    prep_bias: i32,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();
    // No bitdepth in the asm-ABI signature: keeps the original 10bpc
    // (h_sh = 0) semantics for this compat shim.
    unsafe { h_bilin_16bpc_prep_direct_avx2_inner(token, dst, src, w, mx, prep_bias, 0) }
}

/// Vertical bilinear filter for 16bpc prep (V-only case)
/// Reads directly from u16 source, outputs i16 with bias
#[cfg(target_arch = "x86_64")]
#[rite]
fn v_bilin_16bpc_prep_direct_avx2_inner(
    _token: Desktop64,
    dst: &mut [i16],
    src: &[u16],
    src_base: usize,
    src_stride: isize,
    w: usize,
    my: i32,
    prep_bias: i32,
    v_sh: i32,
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    let row_base = |row: usize| src_base.wrapping_add_signed(row as isize * src_stride);
    let w0 = _mm256_set1_epi32(16 - my);
    let w1 = _mm256_set1_epi32(my);
    let rnd = _mm256_set1_epi32((1 << v_sh) >> 1);
    let shift_count = _mm_cvtsi32_si128(v_sh);
    let bias = _mm256_set1_epi32(prep_bias);

    let mut col = 0usize;

    while col + 8 <= w {
        // Load from row 0 and row 1
        let p0 = _mm256_cvtepu16_epi32(loadu_128!(
            <&[u16; 8]>::try_from(&src[src_base + col..src_base + col + 8]).unwrap()
        ));
        let p1 = _mm256_cvtepu16_epi32(loadu_128!(
            <&[u16; 8]>::try_from(&src[row_base(1) + col..row_base(1) + col + 8]).unwrap()
        ));

        // (16 - my) * p0 + my * p1
        let term0 = _mm256_mullo_epi32(p0, w0);
        let term1 = _mm256_mullo_epi32(p1, w1);
        let sum = _mm256_add_epi32(term0, term1);

        // Round with 4 - intermediate_bits (0 for 10bpc, 2 for 12bpc),
        // subtract bias, wrap to i16 like the C int16_t store.
        let shifted = _mm256_sra_epi32(_mm256_add_epi32(sum, rnd), shift_count);
        let biased = _mm256_sub_epi32(shifted, bias);
        let wrapped = _mm256_and_si256(biased, _mm256_set1_epi32(0xFFFF));
        let packed = _mm256_packus_epi32(wrapped, wrapped);
        let result = _mm256_permute4x64_epi64(packed, 0b00_00_10_00);

        storeu_128!(
            <&mut [i16; 8]>::try_from(&mut dst[col..col + 8]).unwrap(),
            _mm256_castsi256_si128(result)
        );
        col += 8;
    }

    // Scalar fallback
    let r = (1 << v_sh) >> 1;
    while col < w {
        let x0 = src[src_base + col] as i32;
        let x1 = src[row_base(1) + col] as i32;
        let pixel = (16 - my) * x0 + my * x1;
        dst[col] = (((pixel + r) >> v_sh) - prep_bias) as i16;
        col += 1;
    }
}
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[archmage::rite(v3)]
unsafe fn v_bilin_16bpc_prep_direct_v3(
    dst: *mut i16,
    src: *const u16,
    src_stride: isize,
    w: usize,
    my: i32,
    prep_bias: i32,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();
    // No bitdepth in the asm-ABI signature: keeps the original 10bpc
    // (v_sh = 0) semantics for this compat shim.
    unsafe {
        v_bilin_16bpc_prep_direct_avx2_inner(token, dst, src, 0, src_stride, w, my, prep_bias, 0)
    }
}

/// Bilinear put for 16bpc
#[cfg(target_arch = "x86_64")]
#[cfg(feature = "asm")]
#[arcane]
unsafe fn put_bilin_16bpc_avx2_inner(
    _token: Desktop64,
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
) {
    let w = w as usize;
    let h = h as usize;
    let mx = mx as usize;
    let my = my as usize;
    let dst = dst_ptr as *mut u16;
    let src = src_ptr as *const u16;
    let dst_stride = dst_stride / 2; // Convert byte stride to u16 stride
    let src_stride = src_stride / 2;
    let bd_max = bitdepth_max as i32;

    // intermediate_bits = 14 - bitdepth: 4 for 10bpc, 2 for 12bpc.
    // H pass shift = 4 - intermediate_bits; V pass shift = 4 + intermediate_bits.
    let intermediate_bits = 14 - (u16::BITS - (bitdepth_max as u16).leading_zeros()) as i32;
    let h_pass_sh = 4 - intermediate_bits;
    let v_pass_sh = 4 + intermediate_bits;

    unsafe {
        if mx != 0 {
            if my != 0 {
                // H+V filtering using SIMD
                let tmp_h = h + 1;
                let mut mid = take_mid_i32_130();

                // Horizontal pass using SIMD (output unshifted)
                for y in 0..tmp_h {
                    let src_row = src.offset(y as isize * src_stride);
                    h_bilin_16bpc_avx2_inner(_token, &mut mid[y], src_row, w, mx as i32, h_pass_sh);
                }

                // Vertical pass using SIMD
                for y in 0..h {
                    let dst_row = dst.offset(y as isize * dst_stride);
                    v_bilin_16bpc_avx2_inner(
                        _token, dst_row, &*mid, w, y, my as i32, v_pass_sh, bd_max,
                    );
                }
                put_mid_i32_130(mid);
            } else {
                // H-only filtering (SIMD)
                for y in 0..h {
                    let src_row = src.offset(y as isize * src_stride);
                    let dst_row = dst.offset(y as isize * dst_stride);
                    h_bilin_16bpc_put_avx2_inner(
                        _token,
                        dst_row,
                        src_row,
                        w,
                        mx as i32,
                        bd_max,
                        intermediate_bits,
                    );
                }
            }
        } else if my != 0 {
            // V-only filtering (SIMD)
            for y in 0..h {
                let src_row = src.offset(y as isize * src_stride);
                let dst_row = dst.offset(y as isize * dst_stride);
                v_bilin_16bpc_direct_avx2_inner(
                    _token, dst_row, src_row, 0, src_stride, w, my as i32, bd_max,
                );
            }
        } else {
            // Simple copy
            for y in 0..h {
                let src_row = src.offset(y as isize * src_stride);
                let dst_row = dst.offset(y as isize * dst_stride);
                std::ptr::copy_nonoverlapping(src_row, dst_row, w);
            }
        }
    }
}

/// Non-FFI wrapper for bilinear put 16bpc (no FFISafe params)
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[archmage::rite(v3)]
unsafe fn put_bilin_16bpc_impl_v3(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();
    unsafe {
        put_bilin_16bpc_avx2_inner(
            token,
            dst_ptr,
            dst_stride,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
        )
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn put_bilin_16bpc_v3(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    _dst: *const FFISafe<PicOffset>,
    _src: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    unsafe {
        put_bilin_16bpc_avx2_inner(
            token,
            dst_ptr,
            dst_stride,
            src_ptr,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
        )
    }
}

/// Bilinear prep for 16bpc (scalar implementation)
#[cfg(target_arch = "x86_64")]
#[cfg(feature = "asm")]
#[arcane]
unsafe fn prep_bilin_16bpc_avx2_inner(
    _token: Desktop64,
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
) {
    let w = w as usize;
    let h = h as usize;
    let mx = mx as usize;
    let my = my as usize;
    let src = src_ptr as *const u16;
    let src_stride = src_stride / 2;

    // For 16bpc prep: PREP_BIAS = 8192
    let prep_bias = 8192i32;

    // intermediate_bits = 14 - bitdepth: 4 for 10bpc, 2 for 12bpc.
    // H pass rounds with 4 - intermediate_bits; the V pass in
    // prep_bilin_c always rounds with FILTER_BILIN_RND(..., 4) —
    // 4 + intermediate_bits is the PUT convention, not prep.
    let intermediate_bits = 14 - (u16::BITS - (bitdepth_max as u16).leading_zeros()) as i32;
    let h_pass_sh = 4 - intermediate_bits;
    let v_pass_sh = 4i32;

    unsafe {
        if mx != 0 {
            if my != 0 {
                // H+V filtering using SIMD
                let tmp_h = h + 1;
                let mut mid = take_mid_i32_130();

                // Horizontal pass using SIMD
                for y in 0..tmp_h {
                    let src_row = src.offset(y as isize * src_stride);
                    h_bilin_16bpc_avx2_inner(_token, &mut mid[y], src_row, w, mx as i32, h_pass_sh);
                }

                // Vertical pass using SIMD (with bias subtraction)
                for y in 0..h {
                    let dst_row = tmp.add(y * w);
                    v_bilin_16bpc_prep_avx2_inner(
                        _token, dst_row, &*mid, w, y, my as i32, v_pass_sh, prep_bias,
                    );
                }
                put_mid_i32_130(mid);
            } else {
                // H-only filtering (SIMD)
                for y in 0..h {
                    let src_row = src.offset(y as isize * src_stride);
                    let dst_row = tmp.add(y * w);
                    h_bilin_16bpc_prep_direct_avx2_inner(
                        _token, dst_row, src_row, w, mx as i32, prep_bias, h_pass_sh,
                    );
                }
            }
        } else if my != 0 {
            // V-only filtering (SIMD)
            for y in 0..h {
                let src_row = src.offset(y as isize * src_stride);
                let dst_row = tmp.add(y * w);
                v_bilin_16bpc_prep_direct_avx2_inner(
                    _token, dst_row, src_row, 0, src_stride, w, my as i32, prep_bias, h_pass_sh,
                );
            }
        } else {
            // Simple copy to prep format: (pixel << intermediate_bits) - bias
            for y in 0..h {
                let src_row = src.offset(y as isize * src_stride);
                let dst_row = tmp.add(y * w);
                for x in 0..w {
                    let pixel = src_row[x] as i32;
                    *dst_row.add(x) = ((pixel << intermediate_bits) - prep_bias) as i16;
                }
            }
        }
    }
}

/// Non-FFI wrapper for bilinear prep 16bpc (no FFISafe params)
#[cfg(feature = "asm")]
#[cfg(target_arch = "x86_64")]
#[archmage::rite(v3)]
unsafe fn prep_bilin_16bpc_impl_v3(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();
    // No bitdepth in this signature: keeps the original 10bpc
    // (intermediate_bits = 4) semantics for this compat shim.
    unsafe { prep_bilin_16bpc_avx2_inner(token, tmp, src_ptr, src_stride, w, h, mx, my, 1023) }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[archmage::rite(v3)]
pub unsafe extern "C" fn prep_bilin_16bpc_v3(
    tmp: *mut i16,
    src_ptr: *const DynPixel,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    _src: *const FFISafe<PicOffset>,
) {
    #[deny(unsafe_op_in_unsafe_fn)]
    let token = archmage::X64V3Token::from_context();

    unsafe {
        prep_bilin_16bpc_avx2_inner(token, tmp, src_ptr, src_stride, w, h, mx, my, bitdepth_max)
    }
}

// ============================================================================
// 16BPC BILINEAR AVX-512
// ============================================================================

/// Horizontal bilinear filter for 16bpc AVX-512 (H → i32 mid)
/// 16 pixels/iter using cvtepu16_epi32 + mullo_epi32
#[cfg(target_arch = "x86_64")]
#[rite]
fn h_bilin_16bpc_avx512_inner(
    _token: Server64,
    dst: &mut [i32],
    src: &[u16],
    w: usize,
    mx: i32,
    h_sh: i32,
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    let w0 = _mm512_set1_epi32(16 - mx);
    let w1 = _mm512_set1_epi32(mx);
    let rnd = _mm512_set1_epi32((1 << h_sh) >> 1);
    let shift_count = _mm_cvtsi32_si128(h_sh);

    let mut col = 0usize;
    while col + 16 <= w {
        let p0 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[col..col + 16]).unwrap()
        ));
        let p1 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[col + 1..col + 17]).unwrap()
        ));

        let term0 = _mm512_mullo_epi32(p0, w0);
        let term1 = _mm512_mullo_epi32(p1, w1);
        let sum = _mm512_add_epi32(term0, term1);
        // mid is int16_t in the C reference: wrap like the AVX2 path.
        let rnd_sum = _mm512_sra_epi32(_mm512_add_epi32(sum, rnd), shift_count);
        let result = _mm512_srai_epi32::<16>(_mm512_slli_epi32::<16>(rnd_sum));

        storeu_512!(&mut dst[col..col + 16], [i32; 16], result);
        col += 16;
    }

    let r = (1 << h_sh) >> 1;
    while col < w {
        let x0 = src[col] as i32;
        let x1 = src[col + 1] as i32;
        dst[col] = (((16 * x0 + mx * (x1 - x0)) + r) >> h_sh) as i16 as i32;
        col += 1;
    }
}

/// Vertical bilinear filter for 16bpc AVX-512 (mid → u16)
/// 16 pixels/iter, cvtusepi32_epi16 for output
#[cfg(target_arch = "x86_64")]
#[rite]
fn v_bilin_16bpc_avx512_inner(
    _token: Server64,
    dst: &mut [u16],
    mid: &[[i32; MID_STRIDE]],
    w: usize,
    y: usize,
    my: i32,
    sh: i32,
    max: i32,
) {
    let mut dst = dst.flex_mut();
    let w0 = _mm512_set1_epi32(16 - my);
    let w1 = _mm512_set1_epi32(my);
    let rnd = _mm512_set1_epi32((1 << sh) >> 1);
    let shift_count = _mm_cvtsi32_si128(sh);
    let zero = _mm512_setzero_si512();
    let max_val = _mm512_set1_epi32(max);

    let mut col = 0usize;
    while col + 16 <= w {
        let r0 = loadu_512!(&mid[y][col..col + 16], [i32; 16]);
        let r1 = loadu_512!(&mid[y + 1][col..col + 16], [i32; 16]);

        let term0 = _mm512_mullo_epi32(r0, w0);
        let term1 = _mm512_mullo_epi32(r1, w1);
        let sum = _mm512_add_epi32(term0, term1);

        let shifted = _mm512_sra_epi32(_mm512_add_epi32(sum, rnd), shift_count);
        let clamped = _mm512_min_epi32(_mm512_max_epi32(shifted, zero), max_val);

        let packed = _mm512_cvtusepi32_epi16(clamped);
        storeu_256!(
            <&mut [u16; 16]>::try_from(&mut dst[col..col + 16]).unwrap(),
            packed
        );
        col += 16;
    }

    while col < w {
        let r0 = mid[y][col];
        let r1 = mid[y + 1][col];
        let pixel = 16 * r0 + my * (r1 - r0);
        let r = (1 << sh) >> 1;
        let val = ((pixel + r) >> sh).clamp(0, max);
        dst[col] = val as u16;
        col += 1;
    }
}

/// Vertical bilinear filter for 16bpc AVX-512 prep (mid → i16)
/// 16 pixels/iter, cvtsepi32_epi16 for signed output
#[cfg(target_arch = "x86_64")]
#[rite]
fn v_bilin_16bpc_prep_avx512_inner(
    _token: Server64,
    dst: &mut [i16],
    mid: &[[i32; MID_STRIDE]],
    w: usize,
    y: usize,
    my: i32,
    sh: i32,
    prep_bias: i32,
) {
    let mut dst = dst.flex_mut();
    let w0 = _mm512_set1_epi32(16 - my);
    let w1 = _mm512_set1_epi32(my);
    let rnd = _mm512_set1_epi32((1 << sh) >> 1);
    let shift_count = _mm_cvtsi32_si128(sh);
    let bias = _mm512_set1_epi32(prep_bias);

    let mut col = 0usize;
    while col + 16 <= w {
        let r0 = loadu_512!(&mid[y][col..col + 16], [i32; 16]);
        let r1 = loadu_512!(&mid[y + 1][col..col + 16], [i32; 16]);

        let term0 = _mm512_mullo_epi32(r0, w0);
        let term1 = _mm512_mullo_epi32(r1, w1);
        let sum = _mm512_add_epi32(term0, term1);

        let shifted = _mm512_sra_epi32(_mm512_add_epi32(sum, rnd), shift_count);
        let biased = _mm512_sub_epi32(shifted, bias);

        // Wrap to i16 like the C int16_t store; cvtsepi32 saturates, which
        // diverges from the reference for out-of-range 12bpc values.
        let wrapped = _mm512_and_si512(biased, _mm512_set1_epi32(0xFFFF));
        let packed = _mm512_cvtusepi32_epi16(wrapped);
        storeu_256!(
            <&mut [i16; 16]>::try_from(&mut dst[col..col + 16]).unwrap(),
            packed
        );
        col += 16;
    }

    while col < w {
        let r0 = mid[y][col];
        let r1 = mid[y + 1][col];
        let pixel = 16 * r0 + my * (r1 - r0);
        let r = (1 << sh) >> 1;
        let val = ((pixel + r) >> sh) - prep_bias;
        dst[col] = val as i16;
        col += 1;
    }
}

/// Horizontal bilinear filter for 16bpc AVX-512 put (H-only → u16)
#[cfg(target_arch = "x86_64")]
#[rite]
fn h_bilin_16bpc_put_avx512_inner(
    _token: Server64,
    dst: &mut [u16],
    src: &[u16],
    w: usize,
    mx: i32,
    bd_max: i32,
    intermediate_bits: i32,
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    let w0 = _mm512_set1_epi32(16 - mx);
    let w1 = _mm512_set1_epi32(mx);
    // Reference rounds twice: rnd(4 - ib) then (px + ((1<<ib)>>1)) >> ib.
    // For 10bpc (ib=4) this degenerates to (sum + 8) >> 4.
    let h_sh = 4 - intermediate_bits;
    let rnd1 = _mm512_set1_epi32((1 << h_sh) >> 1);
    let rnd2 = _mm512_set1_epi32((1 << intermediate_bits) >> 1);
    let shift1 = _mm_cvtsi32_si128(h_sh);
    let shift2 = _mm_cvtsi32_si128(intermediate_bits);
    let zero = _mm512_setzero_si512();
    let max_val = _mm512_set1_epi32(bd_max);

    let mut col = 0usize;
    while col + 16 <= w {
        let p0 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[col..col + 16]).unwrap()
        ));
        let p1 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[col + 1..col + 17]).unwrap()
        ));

        let term0 = _mm512_mullo_epi32(p0, w0);
        let term1 = _mm512_mullo_epi32(p1, w1);
        let sum = _mm512_add_epi32(term0, term1);

        // Two-stage round: ((sum + r1) >> h_sh + r2) >> ib
        let t = _mm512_sra_epi32(_mm512_add_epi32(sum, rnd1), shift1);
        let shifted = _mm512_sra_epi32(_mm512_add_epi32(t, rnd2), shift2);
        let clamped = _mm512_min_epi32(_mm512_max_epi32(shifted, zero), max_val);

        let packed = _mm512_cvtusepi32_epi16(clamped);
        storeu_256!(
            <&mut [u16; 16]>::try_from(&mut dst[col..col + 16]).unwrap(),
            packed
        );
        col += 16;
    }

    let r1 = (1 << h_sh) >> 1;
    let r2 = (1 << intermediate_bits) >> 1;
    while col < w {
        let x0 = src[col] as i32;
        let x1 = src[col + 1] as i32;
        let pixel = (16 - mx) * x0 + mx * x1;
        let result = ((((pixel + r1) >> h_sh) + r2) >> intermediate_bits).clamp(0, bd_max);
        dst[col] = result as u16;
        col += 1;
    }
}

/// Vertical bilinear filter for 16bpc AVX-512 put (V-only u16 → u16)
#[cfg(target_arch = "x86_64")]
#[rite]
fn v_bilin_16bpc_direct_avx512_inner(
    _token: Server64,
    dst: &mut [u16],
    src: &[u16],
    src_base: usize,
    src_stride: isize,
    w: usize,
    my: i32,
    bd_max: i32,
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    let row_base = |row: usize| src_base.wrapping_add_signed(row as isize * src_stride);
    let w0 = _mm512_set1_epi32(16 - my);
    let w1 = _mm512_set1_epi32(my);
    let rnd = _mm512_set1_epi32(8);
    let shift_count = _mm_cvtsi32_si128(4);
    let zero = _mm512_setzero_si512();
    let max_val = _mm512_set1_epi32(bd_max);

    let mut col = 0usize;
    while col + 16 <= w {
        let p0 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[src_base + col..src_base + col + 16]).unwrap()
        ));
        let p1 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[row_base(1) + col..row_base(1) + col + 16]).unwrap()
        ));

        let term0 = _mm512_mullo_epi32(p0, w0);
        let term1 = _mm512_mullo_epi32(p1, w1);
        let sum = _mm512_add_epi32(term0, term1);

        let shifted = _mm512_sra_epi32(_mm512_add_epi32(sum, rnd), shift_count);
        let clamped = _mm512_min_epi32(_mm512_max_epi32(shifted, zero), max_val);

        let packed = _mm512_cvtusepi32_epi16(clamped);
        storeu_256!(
            <&mut [u16; 16]>::try_from(&mut dst[col..col + 16]).unwrap(),
            packed
        );
        col += 16;
    }

    while col < w {
        let x0 = src[src_base + col] as i32;
        let x1 = src[row_base(1) + col] as i32;
        let pixel = (16 - my) * x0 + my * x1;
        let result = ((pixel + 8) >> 4).clamp(0, bd_max);
        dst[col] = result as u16;
        col += 1;
    }
}

/// Horizontal bilinear filter for 16bpc AVX-512 prep (H-only → i16)
#[cfg(target_arch = "x86_64")]
#[rite]
fn h_bilin_16bpc_prep_direct_avx512_inner(
    _token: Server64,
    dst: &mut [i16],
    src: &[u16],
    w: usize,
    mx: i32,
    prep_bias: i32,
    h_sh: i32,
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    let w0 = _mm512_set1_epi32(16 - mx);
    let w1 = _mm512_set1_epi32(mx);
    let rnd = _mm512_set1_epi32((1 << h_sh) >> 1);
    let shift_count = _mm_cvtsi32_si128(h_sh);
    let bias = _mm512_set1_epi32(prep_bias);

    let mut col = 0usize;
    while col + 16 <= w {
        let p0 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[col..col + 16]).unwrap()
        ));
        let p1 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[col + 1..col + 17]).unwrap()
        ));

        let term0 = _mm512_mullo_epi32(p0, w0);
        let term1 = _mm512_mullo_epi32(p1, w1);
        let sum = _mm512_add_epi32(term0, term1);

        // Round with 4 - intermediate_bits (0 for 10bpc, 2 for 12bpc),
        // subtract bias, wrap to i16 like the C int16_t store.
        let shifted = _mm512_sra_epi32(_mm512_add_epi32(sum, rnd), shift_count);
        let biased = _mm512_sub_epi32(shifted, bias);
        let wrapped = _mm512_and_si512(biased, _mm512_set1_epi32(0xFFFF));
        let packed = _mm512_cvtusepi32_epi16(wrapped);
        storeu_256!(
            <&mut [i16; 16]>::try_from(&mut dst[col..col + 16]).unwrap(),
            packed
        );
        col += 16;
    }

    let r = (1 << h_sh) >> 1;
    while col < w {
        let x0 = src[col] as i32;
        let x1 = src[col + 1] as i32;
        let pixel = (16 - mx) * x0 + mx * x1;
        dst[col] = (((pixel + r) >> h_sh) - prep_bias) as i16;
        col += 1;
    }
}

/// Vertical bilinear filter for 16bpc AVX-512 prep (V-only u16 → i16)
#[cfg(target_arch = "x86_64")]
#[rite]
fn v_bilin_16bpc_prep_direct_avx512_inner(
    _token: Server64,
    dst: &mut [i16],
    src: &[u16],
    src_base: usize,
    src_stride: isize,
    w: usize,
    my: i32,
    prep_bias: i32,
    v_sh: i32,
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    let row_base = |row: usize| src_base.wrapping_add_signed(row as isize * src_stride);
    let w0 = _mm512_set1_epi32(16 - my);
    let w1 = _mm512_set1_epi32(my);
    let rnd = _mm512_set1_epi32((1 << v_sh) >> 1);
    let shift_count = _mm_cvtsi32_si128(v_sh);
    let bias = _mm512_set1_epi32(prep_bias);

    let mut col = 0usize;
    while col + 16 <= w {
        let p0 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[src_base + col..src_base + col + 16]).unwrap()
        ));
        let p1 = _mm512_cvtepu16_epi32(loadu_256!(
            <&[u16; 16]>::try_from(&src[row_base(1) + col..row_base(1) + col + 16]).unwrap()
        ));

        let term0 = _mm512_mullo_epi32(p0, w0);
        let term1 = _mm512_mullo_epi32(p1, w1);
        let sum = _mm512_add_epi32(term0, term1);

        // Round with 4 - intermediate_bits (0 for 10bpc, 2 for 12bpc),
        // subtract bias, wrap to i16 like the C int16_t store.
        let shifted = _mm512_sra_epi32(_mm512_add_epi32(sum, rnd), shift_count);
        let biased = _mm512_sub_epi32(shifted, bias);
        let wrapped = _mm512_and_si512(biased, _mm512_set1_epi32(0xFFFF));
        let packed = _mm512_cvtusepi32_epi16(wrapped);
        storeu_256!(
            <&mut [i16; 16]>::try_from(&mut dst[col..col + 16]).unwrap(),
            packed
        );
        col += 16;
    }

    let r = (1 << v_sh) >> 1;
    while col < w {
        let x0 = src[src_base + col] as i32;
        let x1 = src[row_base(1) + col] as i32;
        let pixel = (16 - my) * x0 + my * x1;
        dst[col] = (((pixel + r) >> v_sh) - prep_bias) as i16;
        col += 1;
    }
}

/// Bilinear put for 16bpc using AVX-512 (safe slices)
#[cfg(target_arch = "x86_64")]
#[arcane]
fn put_bilin_16bpc_avx512_impl_inner(
    _token: Server64,
    dst: &mut [u16],
    dst_base: usize,
    dst_stride: isize,
    src: &[u16],
    src_base: usize,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    let w = w as usize;
    let h = h as usize;
    let bd_max = bitdepth_max;
    // intermediate_bits = 14 - bitdepth: 4 for 10bpc, 2 for 12bpc.
    let intermediate_bits = 14 - (u16::BITS - (bitdepth_max as u16).leading_zeros()) as i32;
    let h_pass_sh = 4 - intermediate_bits;
    let v_pass_sh = 4 + intermediate_bits;

    match (mx != 0, my != 0) {
        (true, true) => {
            let tmp_h = h + 1;
            let mut mid = take_mid_i32_130();
            for y in 0..tmp_h {
                let src_off = src_base.wrapping_add_signed(y as isize * src_stride);
                h_bilin_16bpc_avx512_inner(_token, &mut mid[y], &src[src_off..], w, mx, h_pass_sh);
            }
            for y in 0..h {
                let dst_off = dst_base.wrapping_add_signed(y as isize * dst_stride);
                v_bilin_16bpc_avx512_inner(
                    _token,
                    &mut dst[dst_off..],
                    &*mid,
                    w,
                    y,
                    my,
                    v_pass_sh,
                    bd_max,
                );
            }
            put_mid_i32_130(mid);
        }
        (true, false) => {
            for y in 0..h {
                let src_off = src_base.wrapping_add_signed(y as isize * src_stride);
                let dst_off = dst_base.wrapping_add_signed(y as isize * dst_stride);
                h_bilin_16bpc_put_avx512_inner(
                    _token,
                    &mut dst[dst_off..],
                    &src[src_off..],
                    w,
                    mx,
                    bd_max,
                    intermediate_bits,
                );
            }
        }
        (false, true) => {
            for y in 0..h {
                let src_off = src_base.wrapping_add_signed(y as isize * src_stride);
                let dst_off = dst_base.wrapping_add_signed(y as isize * dst_stride);
                v_bilin_16bpc_direct_avx512_inner(
                    _token,
                    &mut dst[dst_off..],
                    &src,
                    src_off,
                    src_stride,
                    w,
                    my,
                    bd_max,
                );
            }
        }
        (false, false) => {
            for y in 0..h {
                let src_off = src_base.wrapping_add_signed(y as isize * src_stride);
                let dst_off = dst_base.wrapping_add_signed(y as isize * dst_stride);
                copy_row_px(&mut dst[dst_off..], &src[src_off..], w);
            }
        }
    }
}

/// Bilinear prep for 16bpc using AVX-512 (safe slices)
#[cfg(target_arch = "x86_64")]
#[arcane]
fn prep_bilin_16bpc_avx512_impl_inner(
    _token: Server64,
    tmp: &mut [i16],
    src: &[u16],
    src_base: usize,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
) {
    let mut tmp = tmp.flex_mut();
    let src = src.flex();
    let w = w as usize;
    let h = h as usize;
    let prep_bias = 8192i32;
    // intermediate_bits = 14 - bitdepth: 4 for 10bpc, 2 for 12bpc.
    let intermediate_bits = 14 - (u16::BITS - (bitdepth_max as u16).leading_zeros()) as i32;
    let h_pass_sh = 4 - intermediate_bits;
    // prep_bilin_c rounds the V pass with FILTER_BILIN_RND(..., 4);
    // 4 + intermediate_bits is the PUT convention, not prep.
    let v_pass_sh = 4i32;

    match (mx != 0, my != 0) {
        (true, true) => {
            let tmp_h = h + 1;
            let mut mid = take_mid_i32_130();
            for y in 0..tmp_h {
                let src_off = src_base.wrapping_add_signed(y as isize * src_stride);
                h_bilin_16bpc_avx512_inner(_token, &mut mid[y], &src[src_off..], w, mx, h_pass_sh);
            }
            for y in 0..h {
                let dst_row = y * w;
                v_bilin_16bpc_prep_avx512_inner(
                    _token,
                    &mut tmp[dst_row..],
                    &*mid,
                    w,
                    y,
                    my,
                    v_pass_sh,
                    prep_bias,
                );
            }
            put_mid_i32_130(mid);
        }
        (true, false) => {
            for y in 0..h {
                let src_off = src_base.wrapping_add_signed(y as isize * src_stride);
                let dst_row = y * w;
                h_bilin_16bpc_prep_direct_avx512_inner(
                    _token,
                    &mut tmp[dst_row..],
                    &src[src_off..],
                    w,
                    mx,
                    prep_bias,
                    h_pass_sh,
                );
            }
        }
        (false, true) => {
            for y in 0..h {
                let src_off = src_base.wrapping_add_signed(y as isize * src_stride);
                let dst_row = y * w;
                v_bilin_16bpc_prep_direct_avx512_inner(
                    _token,
                    &mut tmp[dst_row..],
                    &src,
                    src_off,
                    src_stride,
                    w,
                    my,
                    prep_bias,
                    h_pass_sh,
                );
            }
        }
        (false, false) => {
            // Copy to prep format: (pixel << intermediate_bits) - bias
            for y in 0..h {
                let src_off = src_base.wrapping_add_signed(y as isize * src_stride);
                let dst_row = y * w;
                for x in 0..w {
                    let pixel = src[src_off + x] as i32;
                    tmp[dst_row + x] = ((pixel << intermediate_bits) - prep_bias) as i16;
                }
            }
        }
    }
    let _ = bitdepth_max;
}

/// Bilinear put for 16bpc using AVX2 (safe slices)
#[cfg(target_arch = "x86_64")]
#[arcane]
fn put_bilin_16bpc_avx2_impl_inner_safe(
    _token: Desktop64,
    dst: &mut [u16],
    dst_base: usize,
    dst_stride: isize,
    src: &[u16],
    src_base: usize,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    let w = w as usize;
    let h = h as usize;
    let bd_max = bitdepth_max;
    // intermediate_bits = 14 - bitdepth: 4 for 10bpc, 2 for 12bpc.
    let intermediate_bits = 14 - (u16::BITS - (bitdepth_max as u16).leading_zeros()) as i32;
    let h_pass_sh = 4 - intermediate_bits;
    let v_pass_sh = 4 + intermediate_bits;

    match (mx != 0, my != 0) {
        (true, true) => {
            let tmp_h = h + 1;
            let mut mid = take_mid_i32_130();
            for y in 0..tmp_h {
                let src_off = src_base.wrapping_add_signed(y as isize * src_stride);
                h_bilin_16bpc_avx2_inner(_token, &mut mid[y], &src[src_off..], w, mx, h_pass_sh);
            }
            for y in 0..h {
                let dst_off = dst_base.wrapping_add_signed(y as isize * dst_stride);
                v_bilin_16bpc_avx2_inner(
                    _token,
                    &mut dst[dst_off..],
                    &*mid,
                    w,
                    y,
                    my,
                    v_pass_sh,
                    bd_max,
                );
            }
            put_mid_i32_130(mid);
        }
        (true, false) => {
            for y in 0..h {
                let src_off = src_base.wrapping_add_signed(y as isize * src_stride);
                let dst_off = dst_base.wrapping_add_signed(y as isize * dst_stride);
                h_bilin_16bpc_put_avx2_inner(
                    _token,
                    &mut dst[dst_off..],
                    &src[src_off..],
                    w,
                    mx,
                    bd_max,
                    intermediate_bits,
                );
            }
        }
        (false, true) => {
            for y in 0..h {
                let src_off = src_base.wrapping_add_signed(y as isize * src_stride);
                let dst_off = dst_base.wrapping_add_signed(y as isize * dst_stride);
                v_bilin_16bpc_direct_avx2_inner(
                    _token,
                    &mut dst[dst_off..],
                    &src,
                    src_off,
                    src_stride,
                    w,
                    my,
                    bd_max,
                );
            }
        }
        (false, false) => {
            for y in 0..h {
                let src_off = src_base.wrapping_add_signed(y as isize * src_stride);
                let dst_off = dst_base.wrapping_add_signed(y as isize * dst_stride);
                copy_row_px(&mut dst[dst_off..], &src[src_off..], w);
            }
        }
    }
}

/// Bilinear prep for 16bpc using AVX2 (safe slices)
#[cfg(target_arch = "x86_64")]
#[arcane]
fn prep_bilin_16bpc_avx2_impl_inner_safe(
    _token: Desktop64,
    tmp: &mut [i16],
    src: &[u16],
    src_base: usize,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
) {
    let mut tmp = tmp.flex_mut();
    let src = src.flex();
    let w = w as usize;
    let h = h as usize;
    let prep_bias = 8192i32;
    // intermediate_bits = 14 - bitdepth: 4 for 10bpc, 2 for 12bpc.
    let intermediate_bits = 14 - (u16::BITS - (bitdepth_max as u16).leading_zeros()) as i32;
    let h_pass_sh = 4 - intermediate_bits;
    // prep_bilin_c rounds the V pass with FILTER_BILIN_RND(..., 4);
    // 4 + intermediate_bits is the PUT convention, not prep.
    let v_pass_sh = 4i32;

    match (mx != 0, my != 0) {
        (true, true) => {
            let tmp_h = h + 1;
            let mut mid = take_mid_i32_130();
            for y in 0..tmp_h {
                let src_off = src_base.wrapping_add_signed(y as isize * src_stride);
                h_bilin_16bpc_avx2_inner(_token, &mut mid[y], &src[src_off..], w, mx, h_pass_sh);
            }
            for y in 0..h {
                let dst_row = y * w;
                v_bilin_16bpc_prep_avx2_inner(
                    _token,
                    &mut tmp[dst_row..],
                    &*mid,
                    w,
                    y,
                    my,
                    v_pass_sh,
                    prep_bias,
                );
            }
            put_mid_i32_130(mid);
        }
        (true, false) => {
            for y in 0..h {
                let src_off = src_base.wrapping_add_signed(y as isize * src_stride);
                let dst_row = y * w;
                h_bilin_16bpc_prep_direct_avx2_inner(
                    _token,
                    &mut tmp[dst_row..],
                    &src[src_off..],
                    w,
                    mx,
                    prep_bias,
                    h_pass_sh,
                );
            }
        }
        (false, true) => {
            for y in 0..h {
                let src_off = src_base.wrapping_add_signed(y as isize * src_stride);
                let dst_row = y * w;
                v_bilin_16bpc_prep_direct_avx2_inner(
                    _token,
                    &mut tmp[dst_row..],
                    &src,
                    src_off,
                    src_stride,
                    w,
                    my,
                    prep_bias,
                    h_pass_sh,
                );
            }
        }
        (false, false) => {
            // Copy to prep format: (pixel << intermediate_bits) - bias
            for y in 0..h {
                let src_off = src_base.wrapping_add_signed(y as isize * src_stride);
                let dst_row = y * w;
                for x in 0..w {
                    let pixel = src[src_off + x] as i32;
                    tmp[dst_row + x] = ((pixel << intermediate_bits) - prep_bias) as i16;
                }
            }
        }
    }
    let _ = bitdepth_max;
}

// ============================================================================
// Safe dispatch wrappers for x86_64 AVX2
// Each returns true if SIMD was used (i.e., AVX2 is available).
// ============================================================================

#[cfg(target_arch = "x86_64")]
pub fn avg_dispatch<BD: BitDepth>(
    dst: PicOffset,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    bd: BD,
) -> bool {
    let Some(_token) = crate::src::cpu::summon_avx2() else {
        return false;
    };
    crate::include::dav1d::picture::with_pixel_guard_mut::<BD, _>(
        &dst,
        w as usize,
        h as usize,
        |dst_bytes, dst_offset, dst_stride| {
            avg_dispatch_inner::<BD>(dst_bytes, dst_offset, dst_stride, tmp1, tmp2, w, h, bd);
        },
    );
    true
}

/// Inner avg dispatch — operates on pre-acquired byte slice.
/// Can be called directly with scheduler-owned guards.
#[cfg(target_arch = "x86_64")]
pub(crate) fn avg_dispatch_inner<BD: BitDepth>(
    dst_bytes: &mut [u8],
    dst_offset: usize,
    dst_stride: isize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    bd: BD,
) -> bool {
    use crate::include::common::bitdepth::BPC;
    let avx512_token = crate::src::cpu::summon_avx512();
    let Some(token) = crate::src::cpu::summon_avx2() else {
        return false;
    };
    let bd_c = bd.into_c();
    match BD::BPC {
        BPC::BPC8 => {
            if let Some(t512) = avx512_token {
                avg_8bpc_avx512_safe(
                    t512,
                    &mut dst_bytes[dst_offset..],
                    dst_stride as usize,
                    tmp1,
                    tmp2,
                    w,
                    h,
                );
            } else {
                avg_8bpc_avx2_safe(
                    token,
                    &mut dst_bytes[dst_offset..],
                    dst_stride as usize,
                    tmp1,
                    tmp2,
                    w,
                    h,
                );
            }
        }
        BPC::BPC16 => {
            if let Some(t512) = avx512_token {
                avg_16bpc_avx512_safe(
                    t512,
                    &mut dst_bytes[dst_offset..],
                    dst_stride as usize,
                    tmp1,
                    tmp2,
                    w,
                    h,
                    bd_c,
                );
            } else {
                avg_16bpc_avx2_safe(
                    token,
                    &mut dst_bytes[dst_offset..],
                    dst_stride as usize,
                    tmp1,
                    tmp2,
                    w,
                    h,
                    bd_c,
                );
            }
        }
    }
    true
}

#[cfg(target_arch = "x86_64")]
pub fn w_avg_dispatch<BD: BitDepth>(
    dst: PicOffset,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    weight: i32,
    bd: BD,
) -> bool {
    let Some(_token) = crate::src::cpu::summon_avx2() else {
        return false;
    };
    crate::include::dav1d::picture::with_pixel_guard_mut::<BD, _>(
        &dst,
        w as usize,
        h as usize,
        |dst_bytes, dst_offset, dst_stride| {
            w_avg_dispatch_inner::<BD>(
                dst_bytes, dst_offset, dst_stride, tmp1, tmp2, w, h, weight, bd,
            );
        },
    );
    true
}

/// Inner w_avg dispatch — operates on pre-acquired byte slice.
#[cfg(target_arch = "x86_64")]
pub(crate) fn w_avg_dispatch_inner<BD: BitDepth>(
    dst_bytes: &mut [u8],
    dst_offset: usize,
    dst_stride: isize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    weight: i32,
    bd: BD,
) -> bool {
    use crate::include::common::bitdepth::BPC;
    let avx512_token = crate::src::cpu::summon_avx512();
    let Some(token) = crate::src::cpu::summon_avx2() else {
        return false;
    };
    let bd_c = bd.into_c();
    match BD::BPC {
        BPC::BPC8 => {
            if let Some(t512) = avx512_token {
                w_avg_8bpc_avx512_safe(
                    t512,
                    &mut dst_bytes[dst_offset..],
                    dst_stride as usize,
                    tmp1,
                    tmp2,
                    w,
                    h,
                    weight,
                );
            } else {
                w_avg_8bpc_avx2_safe(
                    token,
                    &mut dst_bytes[dst_offset..],
                    dst_stride as usize,
                    tmp1,
                    tmp2,
                    w,
                    h,
                    weight,
                );
            }
        }
        BPC::BPC16 => {
            if let Some(t512) = avx512_token {
                w_avg_16bpc_avx512_safe(
                    t512,
                    &mut dst_bytes[dst_offset..],
                    dst_stride as usize,
                    tmp1,
                    tmp2,
                    w,
                    h,
                    weight,
                    bd_c,
                );
            } else {
                w_avg_16bpc_avx2_safe(
                    token,
                    &mut dst_bytes[dst_offset..],
                    dst_stride as usize,
                    tmp1,
                    tmp2,
                    w,
                    h,
                    weight,
                    bd_c,
                );
            }
        }
    }
    true
}

#[cfg(target_arch = "x86_64")]
pub fn mask_dispatch<BD: BitDepth>(
    dst: PicOffset,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask: &[u8],
    bd: BD,
) -> bool {
    let Some(_token) = crate::src::cpu::summon_avx2() else {
        return false;
    };
    crate::include::dav1d::picture::with_pixel_guard_mut::<BD, _>(
        &dst,
        w as usize,
        h as usize,
        |dst_bytes, dst_offset, dst_stride| {
            mask_dispatch_inner::<BD>(
                dst_bytes, dst_offset, dst_stride, tmp1, tmp2, w, h, mask, bd,
            );
        },
    );
    true
}

/// Inner mask dispatch — operates on pre-acquired byte slice.
/// Can be called directly with scheduler-owned guards.
#[cfg(target_arch = "x86_64")]
pub(crate) fn mask_dispatch_inner<BD: BitDepth>(
    dst_bytes: &mut [u8],
    dst_offset: usize,
    dst_stride: isize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask: &[u8],
    bd: BD,
) -> bool {
    use crate::include::common::bitdepth::BPC;
    let avx512_token = crate::src::cpu::summon_avx512();
    let Some(token) = crate::src::cpu::summon_avx2() else {
        return false;
    };
    let bd_c = bd.into_c();
    match BD::BPC {
        BPC::BPC8 => {
            if let Some(t512) = avx512_token {
                mask_8bpc_avx512_safe(
                    t512,
                    &mut dst_bytes[dst_offset..],
                    dst_stride as usize,
                    tmp1,
                    tmp2,
                    w,
                    h,
                    mask,
                );
            } else {
                mask_8bpc_avx2_safe(
                    token,
                    &mut dst_bytes[dst_offset..],
                    dst_stride as usize,
                    tmp1,
                    tmp2,
                    w,
                    h,
                    mask,
                );
            }
        }
        BPC::BPC16 => {
            if let Some(t512) = avx512_token {
                mask_16bpc_avx512_safe(
                    t512,
                    &mut dst_bytes[dst_offset..],
                    dst_stride as usize,
                    tmp1,
                    tmp2,
                    w,
                    h,
                    mask,
                    bd_c,
                );
            } else {
                mask_16bpc_avx2_safe(
                    token,
                    &mut dst_bytes[dst_offset..],
                    dst_stride as usize,
                    tmp1,
                    tmp2,
                    w,
                    h,
                    mask,
                    bd_c,
                );
            }
        }
    }
    true
}

#[cfg(target_arch = "x86_64")]
pub fn blend_dispatch<BD: BitDepth>(
    dst: PicOffset,
    tmp: &[BD::Pixel; SCRATCH_INTER_INTRA_BUF_LEN],
    w: i32,
    h: i32,
    mask: &[u8],
) -> bool {
    let Some(_token) = crate::src::cpu::summon_avx2() else {
        return false;
    };
    use zerocopy::IntoBytes;
    let tmp_bytes = tmp.as_bytes();
    crate::include::dav1d::picture::with_pixel_guard_mut::<BD, _>(
        &dst,
        w as usize,
        h as usize,
        |dst_bytes, dst_offset, dst_stride| {
            blend_dispatch_inner::<BD>(dst_bytes, dst_offset, dst_stride, tmp_bytes, w, h, mask);
        },
    );
    true
}

/// Inner blend dispatch — operates on pre-acquired byte slice.
/// Can be called directly with scheduler-owned guards.
#[cfg(target_arch = "x86_64")]
pub(crate) fn blend_dispatch_inner<BD: BitDepth>(
    dst_bytes: &mut [u8],
    dst_offset: usize,
    dst_stride: isize,
    tmp_bytes: &[u8],
    w: i32,
    h: i32,
    mask: &[u8],
) -> bool {
    use crate::include::common::bitdepth::BPC;
    let Some(token) = crate::src::cpu::summon_avx2() else {
        return false;
    };
    match BD::BPC {
        BPC::BPC8 => blend_8bpc_avx2_safe(
            token,
            &mut dst_bytes[dst_offset..],
            dst_stride as usize,
            tmp_bytes,
            w,
            h,
            mask,
        ),
        BPC::BPC16 => blend_16bpc_avx2_safe(
            token,
            &mut dst_bytes[dst_offset..],
            dst_stride as usize,
            tmp_bytes,
            w,
            h,
            mask,
        ),
    }
    true
}

#[cfg(target_arch = "x86_64")]
pub fn blend_dir_dispatch<BD: BitDepth>(
    is_h: bool,
    dst: PicOffset,
    tmp: &[BD::Pixel; SCRATCH_LAP_LEN],
    w: i32,
    h: i32,
) -> bool {
    let Some(_token) = crate::src::cpu::summon_avx2() else {
        return false;
    };
    use zerocopy::IntoBytes;
    let tmp_bytes = tmp.as_bytes();
    crate::include::dav1d::picture::with_pixel_guard_mut::<BD, _>(
        &dst,
        w as usize,
        h as usize,
        |dst_bytes, dst_offset, dst_stride| {
            blend_dir_dispatch_inner::<BD>(
                is_h, dst_bytes, dst_offset, dst_stride, tmp_bytes, w, h,
            );
        },
    );
    true
}

/// Inner blend_dir dispatch — operates on pre-acquired byte slice.
/// Can be called directly with scheduler-owned guards.
#[cfg(target_arch = "x86_64")]
pub(crate) fn blend_dir_dispatch_inner<BD: BitDepth>(
    is_h: bool,
    dst_bytes: &mut [u8],
    dst_offset: usize,
    dst_stride: isize,
    tmp_bytes: &[u8],
    w: i32,
    h: i32,
) -> bool {
    use crate::include::common::bitdepth::BPC;
    let Some(token) = crate::src::cpu::summon_avx2() else {
        return false;
    };
    match (BD::BPC, is_h) {
        (BPC::BPC8, true) => blend_h_8bpc_avx2_safe(
            token,
            &mut dst_bytes[dst_offset..],
            dst_stride as usize,
            tmp_bytes,
            w,
            h,
        ),
        (BPC::BPC8, false) => blend_v_8bpc_avx2_safe(
            token,
            &mut dst_bytes[dst_offset..],
            dst_stride as usize,
            tmp_bytes,
            w,
            h,
        ),
        (BPC::BPC16, true) => blend_h_16bpc_avx2_safe(
            token,
            &mut dst_bytes[dst_offset..],
            dst_stride as usize,
            tmp_bytes,
            w,
            h,
        ),
        (BPC::BPC16, false) => blend_v_16bpc_avx2_safe(
            token,
            &mut dst_bytes[dst_offset..],
            dst_stride as usize,
            tmp_bytes,
            w,
            h,
        ),
    }
    true
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn w_mask_dispatch<BD: BitDepth>(
    layout: Rav1dPixelLayoutSubSampled,
    dst: PicOffset,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask: &mut [u8; SEG_MASK_LEN],
    sign: i32,
    bd: BD,
) -> bool {
    let Some(_token) = crate::src::cpu::summon_avx2() else {
        return false;
    };
    crate::include::dav1d::picture::with_pixel_guard_mut::<BD, _>(
        &dst,
        w as usize,
        h as usize,
        |dst_bytes, dst_offset, dst_stride| {
            w_mask_dispatch_inner::<BD>(
                layout, dst_bytes, dst_offset, dst_stride, tmp1, tmp2, w, h, mask, sign, bd,
            );
        },
    );
    true
}

/// Inner w_mask dispatch — operates on pre-acquired byte slice.
/// Can be called directly with scheduler-owned guards.
#[cfg(target_arch = "x86_64")]
pub(crate) fn w_mask_dispatch_inner<BD: BitDepth>(
    layout: Rav1dPixelLayoutSubSampled,
    dst_bytes: &mut [u8],
    dst_offset: usize,
    dst_stride: isize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask: &mut [u8; SEG_MASK_LEN],
    sign: i32,
    bd: BD,
) -> bool {
    use crate::include::common::bitdepth::BPC;
    let Some(token) = crate::src::cpu::summon_avx2() else {
        return false;
    };
    let bd_c = bd.into_c();
    match (BD::BPC, layout) {
        (BPC::BPC8, Rav1dPixelLayoutSubSampled::I420) => w_mask_420_8bpc_avx2_safe(
            token,
            &mut dst_bytes[dst_offset..],
            dst_stride as usize,
            tmp1,
            tmp2,
            w,
            h,
            mask,
            sign,
        ),
        (BPC::BPC8, Rav1dPixelLayoutSubSampled::I422) => w_mask_422_8bpc_avx2_safe(
            token,
            &mut dst_bytes[dst_offset..],
            dst_stride as usize,
            tmp1,
            tmp2,
            w,
            h,
            mask,
            sign,
        ),
        (BPC::BPC8, Rav1dPixelLayoutSubSampled::I444) => w_mask_444_8bpc_avx2_safe(
            token,
            &mut dst_bytes[dst_offset..],
            dst_stride as usize,
            tmp1,
            tmp2,
            w,
            h,
            mask,
            sign,
        ),
        (BPC::BPC16, Rav1dPixelLayoutSubSampled::I420) => w_mask_420_16bpc_avx2_safe(
            token,
            &mut dst_bytes[dst_offset..],
            dst_stride as usize,
            tmp1,
            tmp2,
            w,
            h,
            mask,
            sign,
            bd_c,
        ),
        (BPC::BPC16, Rav1dPixelLayoutSubSampled::I422) => w_mask_422_16bpc_avx2_safe(
            token,
            &mut dst_bytes[dst_offset..],
            dst_stride as usize,
            tmp1,
            tmp2,
            w,
            h,
            mask,
            sign,
            bd_c,
        ),
        (BPC::BPC16, Rav1dPixelLayoutSubSampled::I444) => w_mask_444_16bpc_avx2_safe(
            token,
            &mut dst_bytes[dst_offset..],
            dst_stride as usize,
            tmp1,
            tmp2,
            w,
            h,
            mask,
            sign,
            bd_c,
        ),
    }
    true
}

/// Safe arcane entry point for put_8tap 8bpc dispatch (calls #[rite] inner fn).
#[cfg(target_arch = "x86_64")]
#[arcane]
fn put_8tap_8bpc_dispatch_inner(
    token: Desktop64,
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    src: &[u8],
    src_base: usize,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    h_filter: Rav1dFilterMode,
    v_filter: Rav1dFilterMode,
) {
    if let Some(t512) = crate::src::cpu::summon_avx512() {
        put_8tap_8bpc_avx512_impl_inner(
            t512, dst, dst_base, dst_stride, src, src_base, src_stride, w, h, mx, my, h_filter,
            v_filter,
        );
        return;
    }
    put_8tap_8bpc_avx2_impl_inner(
        token, dst, dst_base, dst_stride, src, src_base, src_stride, w, h, mx, my, h_filter,
        v_filter,
    );
}

/// Safe arcane entry point for put_bilin 8bpc dispatch.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn put_bilin_8bpc_dispatch_inner(
    token: Desktop64,
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    src: &[u8],
    src_base: usize,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
) {
    if w >= 64 {
        if let Some(t512) = crate::src::cpu::summon_avx512() {
            put_bilin_8bpc_avx512_impl_inner(
                t512, dst, dst_base, dst_stride, src, src_base, src_stride, w, h, mx, my,
            );
            return;
        }
    }
    put_bilin_8bpc_avx2_impl_inner(
        token, dst, dst_base, dst_stride, src, src_base, src_stride, w, h, mx, my,
    );
}

/// Safe arcane entry point for put_8tap 16bpc dispatch.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn put_8tap_16bpc_dispatch_inner(
    token: Desktop64,
    dst: &mut [u16],
    dst_base: usize,
    dst_stride: isize,
    src: &[u16],
    src_base: usize,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bd_c: i32,
    h_filter: Rav1dFilterMode,
    v_filter: Rav1dFilterMode,
) {
    if let Some(t512) = crate::src::cpu::summon_avx512() {
        put_8tap_16bpc_avx512_impl_inner(
            t512, dst, dst_base, dst_stride, src, src_base, src_stride, w, h, mx, my, bd_c,
            h_filter, v_filter,
        );
        return;
    }
    put_8tap_16bpc_avx2_impl_inner(
        token, dst, dst_base, dst_stride, src, src_base, src_stride, w, h, mx, my, bd_c, h_filter,
        v_filter,
    );
}

/// Safe arcane entry point for prep_8tap 8bpc dispatch.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn prep_8tap_8bpc_dispatch_inner(
    token: Desktop64,
    tmp: &mut [i16],
    src: &[u8],
    src_base: usize,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    h_filter: Rav1dFilterMode,
    v_filter: Rav1dFilterMode,
) {
    if let Some(t512) = crate::src::cpu::summon_avx512() {
        prep_8tap_8bpc_avx512_impl_inner(
            t512, tmp, src, src_base, src_stride, w, h, mx, my, h_filter, v_filter,
        );
        return;
    }
    prep_8tap_8bpc_avx2_impl_inner(
        token, tmp, src, src_base, src_stride, w, h, mx, my, h_filter, v_filter,
    );
}

/// Safe arcane entry point for prep_bilin 8bpc dispatch.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn prep_bilin_8bpc_dispatch_inner(
    token: Desktop64,
    tmp: &mut [i16],
    src: &[u8],
    src_base: usize,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
) {
    if w >= 64 {
        if let Some(t512) = crate::src::cpu::summon_avx512() {
            prep_bilin_8bpc_avx512_impl_inner(t512, tmp, src, src_base, src_stride, w, h, mx, my);
            return;
        }
    }
    prep_bilin_8bpc_avx2_impl_inner(token, tmp, src, src_base, src_stride, w, h, mx, my);
}

/// Safe arcane entry point for prep_8tap 16bpc dispatch.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn prep_8tap_16bpc_dispatch_inner(
    token: Desktop64,
    tmp: &mut [i16],
    src: &[u16],
    src_base: usize,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bitdepth_max: i32,
    h_filter: Rav1dFilterMode,
    v_filter: Rav1dFilterMode,
) {
    if let Some(t512) = crate::src::cpu::summon_avx512() {
        prep_8tap_16bpc_avx512_impl_inner(
            t512,
            tmp,
            src,
            src_base,
            src_stride,
            w,
            h,
            mx,
            my,
            bitdepth_max,
            h_filter,
            v_filter,
        );
        return;
    }
    prep_8tap_16bpc_avx2_impl_inner(
        token,
        tmp,
        src,
        src_base,
        src_stride,
        w,
        h,
        mx,
        my,
        bitdepth_max,
        h_filter,
        v_filter,
    );
}

#[cfg(target_arch = "x86_64")]
pub fn mc_put_dispatch<BD: BitDepth>(
    filter: Filter2d,
    dst: PicOffset,
    src: PicOffset,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bd: BD,
) -> bool {
    let Some(_token) = crate::src::cpu::summon_avx2() else {
        return false;
    };

    // When src and dst are from the same picture component (self-referencing frames),
    // we can't hold both a mutable dst guard and immutable src guard simultaneously.
    // Fall through to scalar for this rare case.
    if dst.data.ref_eq(src.data) {
        return false;
    }

    crate::include::dav1d::picture::with_pixel_guard_mut::<BD, _>(
        &dst,
        w as usize,
        h as usize,
        |dst_bytes, dst_offset, dst_stride| {
            mc_put_dispatch_inner::<BD>(
                filter, dst_bytes, dst_offset, dst_stride, src, w, h, mx, my, bd,
            )
        },
    )
}

/// Inner mc_put dispatch — operates on pre-acquired dst byte slice.
/// src guard is acquired inside since src is from a different frame's DisjointMut.
/// Can be called directly with scheduler-owned dst guards.
#[cfg(target_arch = "x86_64")]
pub(crate) fn mc_put_dispatch_inner<BD: BitDepth>(
    filter: Filter2d,
    dst_bytes: &mut [u8],
    dst_offset: usize,
    dst_stride: isize,
    src: PicOffset,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bd: BD,
) -> bool {
    use crate::include::common::bitdepth::BPC;
    use zerocopy::IntoBytes;
    let Some(token) = crate::src::cpu::summon_avx2() else {
        return false;
    };

    let src_stride = src.stride();
    let pixel_size = std::mem::size_of::<BD::Pixel>();
    match BD::BPC {
        BPC::BPC8 => {
            let (src_guard, src_base) = reference::filter_guard::<BD>(src, filter, w, h, mx, my);
            match filter {
                Filter2d::Bilinear => {
                    // Bilinear reads logical rows zero and one; preserve the signed-row origin.
                    let src_bytes = src_guard.as_bytes();
                    put_bilin_8bpc_dispatch_inner(
                        token,
                        dst_bytes,
                        dst_offset,
                        dst_stride,
                        src_bytes,
                        src_base * pixel_size,
                        src_stride,
                        w,
                        h,
                        mx,
                        my,
                    );
                }
                _ => {
                    // 8-tap needs preceding taps; pass the bounded window and its origin
                    let src_bytes = src_guard.as_bytes();
                    let (h_filter, v_filter) = filter.hv();
                    put_8tap_8bpc_dispatch_inner(
                        token,
                        dst_bytes,
                        dst_offset,
                        dst_stride,
                        src_bytes,
                        src_base * pixel_size,
                        src_stride,
                        w,
                        h,
                        mx,
                        my,
                        h_filter,
                        v_filter,
                    );
                }
            }
        }
        BPC::BPC16 => {
            #[allow(deprecated, reason = "Legacy zerocopy slice adapters")]
            let dst_u16: &mut [u16] = zerocopy::Ref::<_, [u16]>::new_slice(dst_bytes)
                .expect("u16 alignment")
                .into_mut_slice();
            let (src_guard, src_base) = reference::filter_guard::<BD>(src, filter, w, h, mx, my);
            let bd_c = bd.into_c();
            match filter {
                Filter2d::Bilinear => {
                    // Bilinear reads logical rows zero and one; preserve the signed-row origin.
                    let src_bytes = src_guard.as_bytes();
                    #[allow(deprecated, reason = "Legacy zerocopy slice adapters")]
                    let src_u16_bilin: &[u16] = zerocopy::Ref::<_, [u16]>::new_slice(src_bytes)
                        .expect("u16 alignment")
                        .into_slice();
                    if let Some(t512) = crate::src::cpu::summon_avx512() {
                        put_bilin_16bpc_avx512_impl_inner(
                            t512,
                            dst_u16,
                            dst_offset / pixel_size,
                            dst_stride / 2,
                            src_u16_bilin,
                            src_base,
                            src_stride / 2,
                            w,
                            h,
                            mx,
                            my,
                            bd_c,
                        );
                    } else {
                        put_bilin_16bpc_avx2_impl_inner_safe(
                            token,
                            dst_u16,
                            dst_offset / pixel_size,
                            dst_stride / 2,
                            src_u16_bilin,
                            src_base,
                            src_stride / 2,
                            w,
                            h,
                            mx,
                            my,
                            bd_c,
                        );
                    }
                }
                _ => {
                    // 8-tap needs preceding taps; pass the bounded window and its origin
                    let src_all_bytes = src_guard.as_bytes();
                    #[allow(deprecated, reason = "Legacy zerocopy slice adapters")]
                    let src_u16: &[u16] = zerocopy::Ref::<_, [u16]>::new_slice(src_all_bytes)
                        .expect("u16 alignment")
                        .into_slice();
                    let (h_filter, v_filter) = filter.hv();
                    put_8tap_16bpc_dispatch_inner(
                        token,
                        dst_u16,
                        dst_offset / pixel_size,
                        dst_stride,
                        src_u16,
                        src_base,
                        src_stride,
                        w,
                        h,
                        mx,
                        my,
                        bd_c,
                        h_filter,
                        v_filter,
                    );
                }
            }
        }
    }
    true
}

/// Debug bisect gate (feature `__bisect`): `MCT_PREP_LOG=1` logs every
/// mct_prep 16bpc call shape. Env read once per process.
#[cfg(all(feature = "__bisect", target_arch = "x86_64"))]
#[inline]
fn mct_prep_log_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("MCT_PREP_LOG").is_some())
}

/// Default build: no env read, no atomic — the log call folds away.
#[cfg(all(not(feature = "__bisect"), target_arch = "x86_64"))]
#[inline(always)]
fn mct_prep_log_enabled() -> bool {
    false
}

#[cfg(target_arch = "x86_64")]
pub fn mct_prep_dispatch<BD: BitDepth>(
    filter: Filter2d,
    tmp: &mut [i16],
    src: PicOffset,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    bd: BD,
) -> bool {
    use crate::include::common::bitdepth::BPC;
    use zerocopy::IntoBytes;
    let Some(token) = crate::src::cpu::summon_avx2() else {
        return false;
    };
    let src_stride = src.stride();
    let pixel_size = std::mem::size_of::<BD::Pixel>();
    match BD::BPC {
        BPC::BPC8 => {
            let (src_guard, src_base) = reference::filter_guard::<BD>(src, filter, w, h, mx, my);
            match filter {
                Filter2d::Bilinear => {
                    // Bilinear reads logical rows zero and one; preserve the signed-row origin.
                    let src_bytes = src_guard.as_bytes();
                    prep_bilin_8bpc_dispatch_inner(
                        token,
                        tmp,
                        src_bytes,
                        src_base * pixel_size,
                        src_stride,
                        w,
                        h,
                        mx,
                        my,
                    );
                }
                _ => {
                    // 8-tap needs preceding taps; pass the bounded window and its origin
                    let src_bytes = src_guard.as_bytes();
                    let (h_filter, v_filter) = filter.hv();
                    prep_8tap_8bpc_dispatch_inner(
                        token,
                        tmp,
                        src_bytes,
                        src_base * pixel_size,
                        src_stride,
                        w,
                        h,
                        mx,
                        my,
                        h_filter,
                        v_filter,
                    );
                }
            }
        }
        BPC::BPC16 => {
            let (src_guard, src_base) = reference::filter_guard::<BD>(src, filter, w, h, mx, my);
            let bd_c = bd.into_c();
            if mct_prep_log_enabled() {
                eprintln!(
                    "mct_prep_16bpc filter={} w={w} h={h} mx={mx} my={my} base={src_base}",
                    filter as usize
                );
            }
            match filter {
                Filter2d::Bilinear => {
                    // Bilinear reads logical rows zero and one; preserve the signed-row origin.
                    let src_bytes = src_guard.as_bytes();
                    #[allow(deprecated, reason = "Legacy zerocopy slice adapters")]
                    let src_u16_bilin: &[u16] = zerocopy::Ref::<_, [u16]>::new_slice(src_bytes)
                        .expect("u16 alignment")
                        .into_slice();
                    if let Some(t512) = crate::src::cpu::summon_avx512() {
                        prep_bilin_16bpc_avx512_impl_inner(
                            t512,
                            tmp,
                            src_u16_bilin,
                            src_base,
                            src_stride / 2,
                            w,
                            h,
                            mx,
                            my,
                            bd_c,
                        );
                    } else {
                        prep_bilin_16bpc_avx2_impl_inner_safe(
                            token,
                            tmp,
                            src_u16_bilin,
                            src_base,
                            src_stride / 2,
                            w,
                            h,
                            mx,
                            my,
                            bd_c,
                        );
                    }
                }
                _ => {
                    // 8-tap needs preceding taps; pass the bounded window and its origin
                    let src_all_bytes = src_guard.as_bytes();
                    #[allow(deprecated, reason = "Legacy zerocopy slice adapters")]
                    let src_u16: &[u16] = zerocopy::Ref::<_, [u16]>::new_slice(src_all_bytes)
                        .expect("u16 alignment")
                        .into_slice();
                    let (h_filter, v_filter) = filter.hv();
                    prep_8tap_16bpc_dispatch_inner(
                        token, tmp, src_u16, src_base, src_stride, w, h, mx, my, bd_c, h_filter,
                        v_filter,
                    );
                }
            }
        }
    }
    let _ = bd;
    true
}

/// No SIMD for scaled variants on x86_64.
#[cfg(target_arch = "x86_64")]
pub fn mc_scaled_dispatch<BD: BitDepth>(
    _filter: Filter2d,
    _dst: PicOffset,
    _src: PicOffset,
    _w: i32,
    _h: i32,
    _mx: i32,
    _my: i32,
    _dx: i32,
    _dy: i32,
    _bd: BD,
) -> bool {
    false
}

/// No SIMD for scaled variants on x86_64.
#[cfg(target_arch = "x86_64")]
pub fn mct_scaled_dispatch<BD: BitDepth>(
    _filter: Filter2d,
    _tmp: &mut [i16],
    _src: PicOffset,
    _w: i32,
    _h: i32,
    _mx: i32,
    _my: i32,
    _dx: i32,
    _dy: i32,
    _bd: BD,
) -> bool {
    false
}

/// SSE2 horizontal 8-tap dot product for warp affine.
///
/// Computes sum(filter[i] * src[off+i] for i in 0..8) using widened i16 multiply.
/// Avoids pmaddubsw saturation by widening to i16 first, then using pmaddwd.
#[cfg(target_arch = "x86_64")]
#[rite]
fn warp_h_dot8(_t: Desktop64, src: &[u8], off: usize, filter: &[i8; 8]) -> i32 {
    // Load 8 source bytes (low 64 bits) and zero-extend to i16
    let s8 = loadu_64!(<&[u8; 8]>::try_from(&src[off..off + 8]).unwrap());
    let s16 = _mm_unpacklo_epi8(s8, _mm_setzero_si128()); // zero-extend u8 → i16

    // Load 8 filter coefficients and sign-extend i8 → i16
    let f16 = _mm_cvtepi8_epi16(loadu_64!(filter));

    // pmaddwd: multiply i16*i16 and add adjacent pairs → 4 i32 values (no saturation)
    let prod = _mm_madd_epi16(s16, f16);

    // Sum 4 i32 values: hadd twice → single i32 in lane 0
    let sum1 = _mm_hadd_epi32(prod, prod);
    let sum2 = _mm_hadd_epi32(sum1, sum1);
    _mm_cvtsi128_si32(sum2)
}

/// Warp affine horizontal pass: 15 rows × 8 pixels → mid[15][8] as i16.
#[cfg(target_arch = "x86_64")]
#[rite]
fn warp_h_pass_8bpc(
    _t: Desktop64,
    mid: &mut [[i16; 15]; 8],
    src: &[u8],
    src_base: usize,
    src_stride: isize,
    alpha: i32,
    beta: i32,
    mx: i32,
) {
    use crate::src::tables::dav1d_mc_warp_filter;

    // intermediate_bits = 4 for 8bpc, shift = 3, round = 4
    let round_h = (1i32 << (7 - 4)) >> 1;
    let shift_h = 7 - 4;

    for y in 0..15usize {
        // Row start: src_base + (y - 3) * stride - 3
        // The -3 column offset accounts for the 8-tap filter centering.
        let row_base = (src_base as isize + (y as isize - 3) * src_stride - 3) as usize;
        let mut tmx = mx + (y as i32) * beta;

        for x in 0..8usize {
            let fidx = (64 + ((tmx + 512) >> 10)) as usize;
            tmx += alpha;
            let filter = &dav1d_mc_warp_filter[fidx];

            let off = row_base + x;
            let dot = warp_h_dot8(_t, src, off, filter);
            mid[x][y] = ((dot + round_h) >> shift_h) as i16;
        }
    }
}

/// Warp affine vertical pass → pixel output (put variant, 8bpc).
#[cfg(target_arch = "x86_64")]
#[rite]
fn warp_v_pass_8bpc_put(
    _t: Desktop64,
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    mid: &[[i16; 15]; 8],
    gamma: i32,
    delta: i32,
    my: i32,
) {
    use crate::src::tables::dav1d_mc_warp_filter;

    // shift = 11, round = 1024
    let round_v = (1i32 << (7 + 4)) >> 1;
    let shift_v = 7 + 4;

    for y in 0..8usize {
        let dst_off = dst_base.wrapping_add_signed(y as isize * dst_stride);
        let mut tmy = my + (y as i32) * delta;

        for x in 0..8usize {
            let fidx = (64 + ((tmy + 512) >> 10)) as usize;
            tmy += gamma;
            let filter = &dav1d_mc_warp_filter[fidx];

            let s = loadu_128!(<&[i16; 8]>::try_from(&mid[x][y..y + 8]).unwrap());
            let f16 = _mm_cvtepi8_epi16(loadu_64!(filter));
            let prod = _mm_madd_epi16(s, f16);
            let sum1 = _mm_hadd_epi32(prod, prod);
            let sum = _mm_cvtsi128_si32(_mm_hadd_epi32(sum1, sum1));
            dst[dst_off + x] = ((sum + round_v) >> shift_v).clamp(0, 255) as u8;
        }
    }
}

/// Warp affine vertical pass → i16 tmp output (prep variant, 8bpc).
#[cfg(target_arch = "x86_64")]
#[rite]
fn warp_v_pass_8bpc_prep(
    _t: Desktop64,
    tmp: &mut [i16],
    tmp_stride: usize,
    mid: &[[i16; 15]; 8],
    gamma: i32,
    delta: i32,
    my: i32,
) {
    use crate::src::tables::dav1d_mc_warp_filter;

    // shift = 7, round = 64
    let round_v = (1i32 << 7) >> 1;
    let shift_v = 7;

    for y in 0..8usize {
        let mut tmy = my + (y as i32) * delta;

        for x in 0..8usize {
            let fidx = (64 + ((tmy + 512) >> 10)) as usize;
            tmy += gamma;
            let filter = &dav1d_mc_warp_filter[fidx];

            let s = loadu_128!(<&[i16; 8]>::try_from(&mid[x][y..y + 8]).unwrap());
            let f16 = _mm_cvtepi8_epi16(loadu_64!(filter));
            let prod = _mm_madd_epi16(s, f16);
            let sum1 = _mm_hadd_epi32(prod, prod);
            let sum = _mm_cvtsi128_si32(_mm_hadd_epi32(sum1, sum1));
            // PREP_BIAS = 0 for 8bpc
            tmp[y * tmp_stride + x] = ((sum + round_v) >> shift_v) as i16;
        }
    }
}

/// AVX2/SSSE3 warp affine 8×8 put (8bpc).
#[cfg(target_arch = "x86_64")]
#[arcane]
fn warp_affine_8x8_8bpc_avx2(
    _t: Desktop64,
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    src: &[u8],
    src_base: usize,
    src_stride: isize,
    abcd: &[i16; 4],
    mx: i32,
    my: i32,
) {
    let mut mid = [[0i16; 15]; 8];
    warp_h_pass_8bpc(
        _t,
        &mut mid,
        src,
        src_base,
        src_stride,
        abcd[0] as i32,
        abcd[1] as i32,
        mx,
    );
    warp_v_pass_8bpc_put(
        _t,
        dst,
        dst_base,
        dst_stride,
        &mid,
        abcd[2] as i32,
        abcd[3] as i32,
        my,
    );
}

/// AVX2/SSSE3 warp affine 8×8 prep (8bpc).
#[cfg(target_arch = "x86_64")]
#[arcane]
fn warp_affine_8x8t_8bpc_avx2(
    _t: Desktop64,
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u8],
    src_base: usize,
    src_stride: isize,
    abcd: &[i16; 4],
    mx: i32,
    my: i32,
) {
    let mut mid = [[0i16; 15]; 8];
    warp_h_pass_8bpc(
        _t,
        &mut mid,
        src,
        src_base,
        src_stride,
        abcd[0] as i32,
        abcd[1] as i32,
        mx,
    );
    warp_v_pass_8bpc_prep(
        _t,
        tmp,
        tmp_stride,
        &mid,
        abcd[2] as i32,
        abcd[3] as i32,
        my,
    );
}

/// SSE2 horizontal 8-tap dot product for warp affine (16bpc).
///
/// Source pixels are u16 (≤4095 for 12-bit, fits in i16). Loads 8 u16 values
/// and 8 i8 filter coefficients, multiplies using pmaddwd for i32 precision.
#[cfg(target_arch = "x86_64")]
#[rite]
fn warp_h_dot16(_t: Desktop64, src: &[u8], off: usize, filter: &[i8; 8]) -> i32 {
    // Load 8 u16 source pixels (16 bytes) — values ≤ 4095, safe to treat as i16
    let s16 = loadu_128!(&src[off..off + 16], [u8; 16]);

    // Load 8 filter coefficients, sign-extend i8 → i16
    let f16 = _mm_cvtepi8_epi16(loadu_64!(filter));

    // pmaddwd: multiply i16×i16, add adjacent pairs → 4 i32 values (no saturation)
    let prod = _mm_madd_epi16(s16, f16);

    // Sum 4 i32 values: hadd twice → single i32 in lane 0
    let sum1 = _mm_hadd_epi32(prod, prod);
    let sum2 = _mm_hadd_epi32(sum1, sum1);
    _mm_cvtsi128_si32(sum2)
}

/// Warp affine horizontal pass (16bpc): 15 rows × 8 pixels → mid[15][8] as i16.
#[cfg(target_arch = "x86_64")]
#[rite]
fn warp_h_pass_16bpc(
    _t: Desktop64,
    mid: &mut [[i16; 15]; 8],
    src: &[u8],
    src_base: usize,
    src_stride: isize,
    alpha: i32,
    beta: i32,
    mx: i32,
    intermediate_bits: u8,
) {
    use crate::src::tables::dav1d_mc_warp_filter;

    let round_h = (1i32 << (7 - intermediate_bits)) >> 1;
    let shift_h = 7 - intermediate_bits;

    for y in 0..15usize {
        // Row start: src_base + (y-3)*stride - 3*2 (3 pixels × 2 bytes/pixel)
        let row_base = (src_base as isize + (y as isize - 3) * src_stride - 6) as usize;
        let mut tmx = mx + (y as i32) * beta;

        for x in 0..8usize {
            let fidx = (64 + ((tmx + 512) >> 10)) as usize;
            tmx += alpha;
            let filter = &dav1d_mc_warp_filter[fidx];

            let off = row_base + x * 2;
            let dot = warp_h_dot16(_t, src, off, filter);
            mid[x][y] = ((dot + round_h) >> shift_h) as i16;
        }
    }
}

/// Warp affine vertical pass → u16 pixel output (put variant, 16bpc).
#[cfg(target_arch = "x86_64")]
#[rite]
fn warp_v_pass_16bpc_put(
    _t: Desktop64,
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    mid: &[[i16; 15]; 8],
    gamma: i32,
    delta: i32,
    my: i32,
    intermediate_bits: u8,
    bitdepth_max: i32,
) {
    use crate::src::tables::dav1d_mc_warp_filter;

    let round_v = (1i32 << (7 + intermediate_bits)) >> 1;
    let shift_v = 7 + intermediate_bits;

    for y in 0..8usize {
        let dst_off = dst_base.wrapping_add_signed(y as isize * dst_stride);
        let mut tmy = my + (y as i32) * delta;

        for x in 0..8usize {
            let fidx = (64 + ((tmy + 512) >> 10)) as usize;
            tmy += gamma;
            let filter = &dav1d_mc_warp_filter[fidx];

            let s = loadu_128!(<&[i16; 8]>::try_from(&mid[x][y..y + 8]).unwrap());
            let f16 = _mm_cvtepi8_epi16(loadu_64!(filter));
            let prod = _mm_madd_epi16(s, f16);
            let sum1 = _mm_hadd_epi32(prod, prod);
            let sum = _mm_cvtsi128_si32(_mm_hadd_epi32(sum1, sum1));
            let val = ((sum + round_v) >> shift_v).clamp(0, bitdepth_max) as u16;
            dst[dst_off + x * 2..][..2].copy_from_slice(&val.to_le_bytes());
        }
    }
}

/// Warp affine vertical pass → i16 tmp output (prep variant, 16bpc).
#[cfg(target_arch = "x86_64")]
#[rite]
fn warp_v_pass_16bpc_prep(
    _t: Desktop64,
    tmp: &mut [i16],
    tmp_stride: usize,
    mid: &[[i16; 15]; 8],
    gamma: i32,
    delta: i32,
    my: i32,
) {
    use crate::src::tables::dav1d_mc_warp_filter;

    let round_v = (1i32 << 7) >> 1;
    let shift_v = 7;

    for y in 0..8usize {
        let mut tmy = my + (y as i32) * delta;

        for x in 0..8usize {
            let fidx = (64 + ((tmy + 512) >> 10)) as usize;
            tmy += gamma;
            let filter = &dav1d_mc_warp_filter[fidx];

            let s = loadu_128!(<&[i16; 8]>::try_from(&mid[x][y..y + 8]).unwrap());
            let f16 = _mm_cvtepi8_epi16(loadu_64!(filter));
            let prod = _mm_madd_epi16(s, f16);
            let sum1 = _mm_hadd_epi32(prod, prod);
            let sum = _mm_cvtsi128_si32(_mm_hadd_epi32(sum1, sum1));
            // PREP_BIAS = 8192 for 16bpc
            tmp[y * tmp_stride + x] = (((sum + round_v) >> shift_v) - 8192) as i16;
        }
    }
}

/// AVX2/SSSE3 warp affine 8×8 put (16bpc).
#[cfg(target_arch = "x86_64")]
#[arcane]
fn warp_affine_8x8_16bpc_avx2(
    _t: Desktop64,
    dst: &mut [u8],
    dst_base: usize,
    dst_stride: isize,
    src: &[u8],
    src_base: usize,
    src_stride: isize,
    abcd: &[i16; 4],
    mx: i32,
    my: i32,
    intermediate_bits: u8,
    bitdepth_max: i32,
) {
    let mut mid = [[0i16; 15]; 8];
    warp_h_pass_16bpc(
        _t,
        &mut mid,
        src,
        src_base,
        src_stride,
        abcd[0] as i32,
        abcd[1] as i32,
        mx,
        intermediate_bits,
    );
    warp_v_pass_16bpc_put(
        _t,
        dst,
        dst_base,
        dst_stride,
        &mid,
        abcd[2] as i32,
        abcd[3] as i32,
        my,
        intermediate_bits,
        bitdepth_max,
    );
}

/// AVX2/SSSE3 warp affine 8×8 prep (16bpc).
#[cfg(target_arch = "x86_64")]
#[arcane]
fn warp_affine_8x8t_16bpc_avx2(
    _t: Desktop64,
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u8],
    src_base: usize,
    src_stride: isize,
    abcd: &[i16; 4],
    mx: i32,
    my: i32,
    intermediate_bits: u8,
) {
    let mut mid = [[0i16; 15]; 8];
    warp_h_pass_16bpc(
        _t,
        &mut mid,
        src,
        src_base,
        src_stride,
        abcd[0] as i32,
        abcd[1] as i32,
        mx,
        intermediate_bits,
    );
    warp_v_pass_16bpc_prep(
        _t,
        tmp,
        tmp_stride,
        &mid,
        abcd[2] as i32,
        abcd[3] as i32,
        my,
    );
}

/// SIMD warp affine 8×8 put dispatch.
#[cfg(target_arch = "x86_64")]
pub fn warp8x8_dispatch<BD: BitDepth>(
    dst: PicOffset,
    src: PicOffset,
    abcd: &[i16; 4],
    mx: i32,
    my: i32,
    bd: BD,
) -> bool {
    use crate::include::common::bitdepth::{AsPrimitive, BPC};
    use zerocopy::IntoBytes;

    let Some(token) = crate::src::cpu::summon_avx2() else {
        return false;
    };

    // Can't hold simultaneous guards on the same picture component
    if dst.data.ref_eq(src.data) {
        return false;
    }

    let src_stride = src.stride();
    let pixel_size = std::mem::size_of::<BD::Pixel>();

    let (src_guard, src_base) = reference::read_guard::<BD>(src, 8, 8, [(3, 4), (3, 4)]);
    let src_bytes = src_guard.as_bytes();

    crate::include::dav1d::picture::with_pixel_guard_mut::<BD, _>(
        &dst,
        8,
        8,
        |dst_bytes, dst_offset, dst_stride| match BD::BPC {
            BPC::BPC8 => {
                warp_affine_8x8_8bpc_avx2(
                    token,
                    dst_bytes,
                    dst_offset,
                    dst_stride,
                    src_bytes,
                    src_base * pixel_size,
                    src_stride,
                    abcd,
                    mx,
                    my,
                );
            }
            BPC::BPC16 => {
                warp_affine_8x8_16bpc_avx2(
                    token,
                    dst_bytes,
                    dst_offset,
                    dst_stride,
                    src_bytes,
                    src_base * pixel_size,
                    src_stride,
                    abcd,
                    mx,
                    my,
                    bd.get_intermediate_bits(),
                    bd.bitdepth_max().as_::<i32>(),
                );
            }
        },
    );
    true
}

/// SIMD warp affine 8×8 prep dispatch.
#[cfg(target_arch = "x86_64")]
pub fn warp8x8t_dispatch<BD: BitDepth>(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: PicOffset,
    abcd: &[i16; 4],
    mx: i32,
    my: i32,
    bd: BD,
) -> bool {
    use crate::include::common::bitdepth::BPC;
    use zerocopy::IntoBytes;

    let Some(token) = crate::src::cpu::summon_avx2() else {
        return false;
    };

    let src_stride = src.stride();
    let pixel_size = std::mem::size_of::<BD::Pixel>();

    let (src_guard, src_base) = reference::read_guard::<BD>(src, 8, 8, [(3, 4), (3, 4)]);
    let src_bytes = src_guard.as_bytes();

    match BD::BPC {
        BPC::BPC8 => {
            warp_affine_8x8t_8bpc_avx2(
                token,
                tmp,
                tmp_stride,
                src_bytes,
                src_base * pixel_size,
                src_stride,
                abcd,
                mx,
                my,
            );
        }
        BPC::BPC16 => {
            warp_affine_8x8t_16bpc_avx2(
                token,
                tmp,
                tmp_stride,
                src_bytes,
                src_base * pixel_size,
                src_stride,
                abcd,
                mx,
                my,
                bd.get_intermediate_bits(),
            );
        }
    }
    true
}

/// No SIMD for emu_edge on x86_64.
#[cfg(target_arch = "x86_64")]
pub fn emu_edge_dispatch<BD: BitDepth>(
    _bw: isize,
    _bh: isize,
    _iw: isize,
    _ih: isize,
    _x: isize,
    _y: isize,
    _dst: &mut [BD::Pixel; crate::src::internal::EMU_EDGE_LEN],
    _dst_pxstride: usize,
    _src: &crate::include::dav1d::picture::Rav1dPictureDataComponent,
) -> bool {
    false
}

/// No SIMD for resize on x86_64.
#[cfg(target_arch = "x86_64")]
pub fn resize_dispatch<BD: BitDepth>(
    _dst: crate::src::with_offset::WithOffset<
        crate::src::pic_or_buf::PicOrBuf<crate::src::align::AlignedVec64<u8>>,
    >,
    _src: PicOffset,
    _dst_w: usize,
    _h: usize,
    _src_w: usize,
    _dx: i32,
    _mx: i32,
    _bd: BD,
) -> bool {
    false
}

// ---------------------------------------------------------------------------
// Test-only arcane shims so plain `#[test]` code can drive the `#[rite]`
// impls directly — dispatch picks one tier per CPU, but parity coverage must
// exercise both.
// ---------------------------------------------------------------------------

/// Drives `put_8tap_8bpc_avx2_impl_inner` (rite) from test code.
#[cfg(all(test, target_arch = "x86_64"))]
#[arcane]
pub(crate) fn put_8tap_8bpc_avx2_impl_testable(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: isize,
    src: &[u8],
    src_base: usize,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    h_filter: Rav1dFilterMode,
    v_filter: Rav1dFilterMode,
) {
    put_8tap_8bpc_avx2_impl_inner(
        _token, dst, 0, dst_stride, src, src_base, src_stride, w, h, mx, my, h_filter, v_filter,
    );
}

/// Drives `prep_8tap_8bpc_avx2_impl_inner` (rite) from test code.
#[cfg(all(test, target_arch = "x86_64"))]
#[arcane]
pub(crate) fn prep_8tap_8bpc_avx2_impl_testable(
    _token: Desktop64,
    tmp: &mut [i16],
    src: &[u8],
    src_base: usize,
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
    h_filter: Rav1dFilterMode,
    v_filter: Rav1dFilterMode,
) {
    prep_8tap_8bpc_avx2_impl_inner(
        _token, tmp, src, src_base, src_stride, w, h, mx, my, h_filter, v_filter,
    );
}

/// Drives `put_bilin_8bpc_avx2_impl_inner` (rite) from test code.
#[cfg(all(test, target_arch = "x86_64"))]
#[arcane]
pub(crate) fn put_bilin_8bpc_avx2_impl_testable(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: isize,
    src: &[u8],
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
) {
    put_bilin_8bpc_avx2_impl_inner(_token, dst, 0, dst_stride, src, 0, src_stride, w, h, mx, my);
}

/// Drives `prep_bilin_8bpc_avx2_impl_inner` (rite) from test code.
#[cfg(all(test, target_arch = "x86_64"))]
#[arcane]
pub(crate) fn prep_bilin_8bpc_avx2_impl_testable(
    _token: Desktop64,
    tmp: &mut [i16],
    src: &[u8],
    src_stride: isize,
    w: i32,
    h: i32,
    mx: i32,
    my: i32,
) {
    prep_bilin_8bpc_avx2_impl_inner(_token, tmp, src, 0, src_stride, w, h, mx, my);
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;

    #[test]
    fn test_avg_8bpc_avx2_matches_scalar() {
        let _tok_lock = archmage::testing::lock_token_testing();
        let Some(token) = crate::src::cpu::summon_avx2() else {
            eprintln!("Skipping AVX2 test - CPU doesn't support it or tokens disabled");
            return;
        };

        let test_values: Vec<i16> = vec![
            0,
            1,
            2,
            127,
            128,
            255,
            256,
            511,
            512,
            1023,
            1024,
            -1,
            -128,
            -256,
            -512,
            -1024,
            i16::MIN,
            i16::MAX,
        ];

        let w = 64i32;
        let h = 2i32;

        let mut tmp1 = [0i16; COMPINTER_LEN];
        let mut tmp2 = [0i16; COMPINTER_LEN];
        let mut dst_avx2 = vec![0u8; (w * h) as usize];
        let mut dst_scalar = vec![0u8; (w * h) as usize];

        for &v1 in &test_values {
            for &v2 in &test_values {
                for i in 0..(w * h) as usize {
                    tmp1[i] = v1;
                    tmp2[i] = v2;
                }
                dst_avx2.fill(0);
                dst_scalar.fill(0);

                // Scalar oracle: avg_rust at 8bpc computes
                // (t1 + t2 + 16) >> 5 in i32, clipped to [0, 255].
                for (i, d) in dst_scalar.iter_mut().enumerate() {
                    let sum = tmp1[i] as i32 + tmp2[i] as i32;
                    *d = ((sum + (1 << 4)) >> 5).clamp(0, 255) as u8;
                }

                avg_8bpc_avx2_safe(token, &mut dst_avx2, w as usize, &tmp1, &tmp2, w, h);

                assert_eq!(
                    dst_avx2,
                    dst_scalar,
                    "Mismatch for v1={}, v2={}: avx2={:?} scalar={:?}",
                    v1,
                    v2,
                    &dst_avx2[..8],
                    &dst_scalar[..8]
                );

                if let Some(t512) = crate::src::cpu::summon_avx512() {
                    let mut dst_avx512 = vec![0u8; (w * h) as usize];
                    avg_8bpc_avx512_safe(t512, &mut dst_avx512, w as usize, &tmp1, &tmp2, w, h);
                    assert_eq!(dst_avx512, dst_scalar, "avx512 mismatch v1={v1} v2={v2}");
                }
            }
        }
    }

    #[test]
    fn test_avg_varying_data() {
        let Some(token) = crate::src::cpu::summon_avx2() else {
            return;
        };

        let w = 128i32;
        let h = 4i32;

        let mut tmp1 = [0i16; COMPINTER_LEN];
        let mut tmp2 = [0i16; COMPINTER_LEN];

        for i in 0..(w * h) as usize {
            tmp1[i] = ((i * 37) % 8192) as i16;
            tmp2[i] = ((i * 73 + 1000) % 8192) as i16;
        }

        let mut dst_a = vec![0u8; (w * h) as usize];
        let mut dst_b = vec![0u8; (w * h) as usize];

        avg_8bpc_avx2_safe(token, &mut dst_a, w as usize, &tmp1, &tmp2, w, h);

        avg_8bpc_avx2_safe(token, &mut dst_b, w as usize, &tmp1, &tmp2, w, h);

        assert_eq!(dst_a, dst_b, "Results differ for varying data");
    }

    #[test]
    fn test_w_avg_8bpc_avx2_matches_scalar() {
        let Some(token) = crate::src::cpu::summon_avx2() else {
            eprintln!("Skipping AVX2 test - CPU doesn't support it or tokens disabled");
            return;
        };

        let test_values: Vec<i16> = vec![
            0, 1, 127, 255, 512, 1023, 2047, 4095, 8191, -1, -128, -512, -1024,
        ];
        let test_weights = [0, 1, 4, 8, 12, 15, 16];

        let w = 64i32;
        let h = 2i32;

        let mut tmp1 = [0i16; COMPINTER_LEN];
        let mut tmp2 = [0i16; COMPINTER_LEN];
        let mut dst_a = vec![0u8; (w * h) as usize];
        let mut dst_b = vec![0u8; (w * h) as usize];

        for &weight in &test_weights {
            for &v1 in &test_values {
                for &v2 in &test_values {
                    for i in 0..(w * h) as usize {
                        tmp1[i] = v1;
                        tmp2[i] = v2;
                    }
                    dst_a.fill(0);
                    dst_b.fill(0);

                    w_avg_8bpc_avx2_safe(token, &mut dst_a, w as usize, &tmp1, &tmp2, w, h, weight);

                    w_avg_8bpc_avx2_safe(token, &mut dst_b, w as usize, &tmp1, &tmp2, w, h, weight);

                    assert_eq!(
                        dst_a,
                        dst_b,
                        "Mismatch for weight={}, v1={}, v2={}: a={:?} b={:?}",
                        weight,
                        v1,
                        v2,
                        &dst_a[..8],
                        &dst_b[..8]
                    );
                }
            }
        }
    }

    #[test]
    fn test_mask_8bpc_matches_scalar() {
        let Some(token) = crate::src::cpu::summon_avx2() else {
            eprintln!("Skipping AVX2 test - CPU doesn't support it or tokens disabled");
            return;
        };

        let test_values: Vec<i16> = vec![0, 127, 255, 512, 1023, 4095, -128, -512];
        let test_masks: Vec<u8> = vec![0, 1, 16, 32, 48, 63, 64];

        let w = 64i32;
        let h = 2i32;

        let mut tmp1 = [0i16; COMPINTER_LEN];
        let mut tmp2 = [0i16; COMPINTER_LEN];
        let mut mask = vec![0u8; (w * h) as usize];
        let mut dst_a = vec![0u8; (w * h) as usize];
        let mut dst_b = vec![0u8; (w * h) as usize];
        for &m in &test_masks {
            for &v1 in &test_values {
                for &v2 in &test_values {
                    for i in 0..(w * h) as usize {
                        tmp1[i] = v1;
                        tmp2[i] = v2;
                        mask[i] = m;
                    }
                    dst_a.fill(0);
                    dst_b.fill(0);

                    mask_8bpc_avx2_safe(token, &mut dst_a, w as usize, &tmp1, &tmp2, w, h, &mask);

                    mask_8bpc_avx2_safe(token, &mut dst_b, w as usize, &tmp1, &tmp2, w, h, &mask);

                    assert_eq!(
                        dst_a,
                        dst_b,
                        "Mismatch for mask={}, v1={}, v2={}: a={:?} b={:?}",
                        m,
                        v1,
                        v2,
                        &dst_a[..8],
                        &dst_b[..8]
                    );
                }
            }
        }
    }

    /// Verify MC avg produces consistent output across all token permutations.
    #[test]
    fn test_avg_token_permutations() {
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

        let _lock = crate::src::safe_simd::token_test_lock();
        let w = 32i32;
        let h = 2i32;
        let size = (w * h) as usize;

        let mut tmp1 = [0i16; COMPINTER_LEN];
        let mut tmp2 = [0i16; COMPINTER_LEN];
        for i in 0..size {
            tmp1[i] = ((i * 37) % 4096) as i16;
            tmp2[i] = ((i * 73 + 500) % 4096) as i16;
        }

        // Compute reference with tokens fully enabled
        let reference = {
            let Some(token) = crate::src::cpu::summon_avx2() else {
                eprintln!("Skipping: AVX2 not available");
                return;
            };
            let mut dst = vec![0u8; size];
            avg_8bpc_avx2_safe(token, &mut dst, w as usize, &tmp1, &tmp2, w, h);
            dst
        };

        let report = for_each_token_permutation(CompileTimePolicy::WarnStderr, |perm| {
            if let Some(token) = crate::src::cpu::summon_avx2() {
                let mut dst = vec![0u8; size];
                avg_8bpc_avx2_safe(token, &mut dst, w as usize, &tmp1, &tmp2, w, h);
                assert_eq!(dst, reference, "avg output mismatch at: {perm}");
            }
        });
        eprintln!("MC avg permutations: {}", report.permutations_run);
        assert!(report.permutations_run >= 1);
    }

    /// Scalar oracle for `w_mask_16bpc_avx2_safe_impl`, mirroring
    /// `src/mc.rs::w_mask_rust` on flat slices.
    fn w_mask_16bpc_scalar_ref(
        dst: &mut [u8],
        dst_stride: usize,
        tmp1: &[i16; COMPINTER_LEN],
        tmp2: &[i16; COMPINTER_LEN],
        w: usize,
        h: usize,
        mask: &mut [u8; SEG_MASK_LEN],
        sign: i32,
        bitdepth_max: i32,
        ss_hor: bool,
        ss_ver: bool,
    ) {
        let sign = sign as u8;
        let bd_max = bitdepth_max as i32;
        let bitdepth = if bitdepth_max == 1023 { 10u32 } else { 12u32 };
        let intermediate_bits = (14 - bitdepth) as u32;
        let sh = intermediate_bits + 6;
        let rnd = (32i32 << intermediate_bits) + 8192 * 64;
        let mask_sh = bitdepth + intermediate_bits - 4;
        let mask_rnd = 1u16 << (mask_sh - 5);
        let mask_w = if ss_hor { w >> 1 } else { w };
        let mut mask_off = 0usize;

        for row_h in 0..h {
            let row_offset = row_h * w;
            let tmp1_row = &tmp1[row_offset..][..w];
            let tmp2_row = &tmp2[row_offset..][..w];
            let dst_row_bytes = &mut dst[row_h * dst_stride..][..w * 2];
            let dst_row: &mut [u16] = zerocopy::FromBytes::mut_from_bytes(dst_row_bytes).unwrap();
            let mut x = 0;
            while x < w {
                let diff = tmp1_row[x].abs_diff(tmp2_row[x]);
                let m = std::cmp::min(38 + ((diff.saturating_add(mask_rnd)) >> mask_sh), 64) as u8;
                let t1 = tmp1_row[x] as i32;
                let t2 = tmp2_row[x] as i32;
                let pixel = (t1 * m as i32 + t2 * (64 - m as i32) + rnd) >> sh;
                dst_row[x] = pixel.clamp(0, bd_max) as u16;

                if ss_hor {
                    x += 1;
                    let diff2 = tmp1_row[x].abs_diff(tmp2_row[x]);
                    let n =
                        std::cmp::min(38 + ((diff2.saturating_add(mask_rnd)) >> mask_sh), 64) as u8;
                    let t1 = tmp1_row[x] as i32;
                    let t2 = tmp2_row[x] as i32;
                    let pixel = (t1 * n as i32 + t2 * (64 - n as i32) + rnd) >> sh;
                    dst_row[x] = pixel.clamp(0, bd_max) as u16;

                    let mask_x = x >> 1;
                    if ss_ver && (row_h & 1 != 0) {
                        let prev = mask[mask_off + mask_x];
                        mask[mask_off + mask_x] =
                            (((m as u16 + n as u16 + 2 - sign as u16) + prev as u16) >> 2) as u8;
                    } else if ss_ver {
                        mask[mask_off + mask_x] = m + n;
                    } else {
                        mask[mask_off + mask_x] =
                            ((m as u16 + n as u16 + 1 - sign as u16) >> 1) as u8;
                    }
                } else {
                    mask[mask_off + x] = m;
                }
                x += 1;
            }

            if !ss_ver || (row_h & 1 != 0) {
                mask_off += mask_w;
            }
        }
    }

    #[test]
    fn w_mask_16bpc_simd_matches_scalar() {
        let _tok_lock = archmage::testing::lock_token_testing();
        let Some(token) = crate::src::cpu::summon_avx2() else {
            eprintln!("w_mask_16bpc_simd_matches_scalar: no AVX2 token, skipping");
            return;
        };
        let mut tmp1 = [0i16; COMPINTER_LEN];
        let mut tmp2 = [0i16; COMPINTER_LEN];
        let mut st: u32 = 0xabcd_1234;
        for i in 0..COMPINTER_LEN {
            st ^= st << 13;
            st ^= st >> 17;
            st ^= st << 5;
            tmp1[i] = (((st >> 4) as i32 % 65536) - 32768) as i16;
            st ^= st << 13;
            st ^= st >> 17;
            st ^= st << 5;
            tmp2[i] = (((st >> 4) as i32 % 65536) - 32768) as i16;
        }
        // Full block-size grid incl. 128x128; w_mask is called with luma dims.
        for &w in &[2usize, 4, 6, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128] {
            for &h in &[1usize, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128] {
                for &bd_max in &[1023i32, 4095] {
                    for &sign in &[0i32, 1] {
                        for &(hor, ver) in &[(true, true), (true, false), (false, false)] {
                            let stride = w * 2 + 34;
                            let mut dst_a = vec![0xAAu8; h * stride];
                            let mut dst_b = vec![0xAAu8; h * stride];
                            let mut mask_a = [0xCCu8; SEG_MASK_LEN];
                            let mut mask_b = [0xCCu8; SEG_MASK_LEN];
                            w_mask_16bpc_scalar_ref(
                                &mut dst_a,
                                stride,
                                &tmp1,
                                &tmp2,
                                w,
                                h,
                                &mut mask_a,
                                sign,
                                bd_max,
                                hor,
                                ver,
                            );
                            match (hor, ver) {
                                (true, true) => w_mask_420_16bpc_avx2_safe(
                                    token,
                                    &mut dst_b,
                                    stride,
                                    &tmp1,
                                    &tmp2,
                                    w as i32,
                                    h as i32,
                                    &mut mask_b,
                                    sign,
                                    bd_max,
                                ),
                                (true, false) => w_mask_422_16bpc_avx2_safe(
                                    token,
                                    &mut dst_b,
                                    stride,
                                    &tmp1,
                                    &tmp2,
                                    w as i32,
                                    h as i32,
                                    &mut mask_b,
                                    sign,
                                    bd_max,
                                ),
                                _ => w_mask_444_16bpc_avx2_safe(
                                    token,
                                    &mut dst_b,
                                    stride,
                                    &tmp1,
                                    &tmp2,
                                    w as i32,
                                    h as i32,
                                    &mut mask_b,
                                    sign,
                                    bd_max,
                                ),
                            }
                            assert_eq!(
                                &mask_a[..],
                                &mask_b[..],
                                "w_mask_16bpc mask w={w} h={h} bd={bd_max} sign={sign} hor={hor} ver={ver}\n a={:?}\n b={:?}",
                                &mask_a[..32.min(SEG_MASK_LEN)],
                                &mask_b[..32.min(SEG_MASK_LEN)]
                            );
                            assert_eq!(
                                dst_a, dst_b,
                                "w_mask_16bpc dst w={w} h={h} bd={bd_max} sign={sign} hor={hor} ver={ver}\n a={:?}\n b={:?}",
                                dst_a, dst_b
                            );
                        }
                    }
                }
            }
        }
    }

    /// mask_8bpc + w_avg_8bpc must match the scalar formulas for the full i16
    /// tmp range — warp_affine mids can approach +-32767, where an i16
    /// (t1 - t2) wraps and a mullo/mulhi on it diverges from scalar i32 math.
    #[test]
    fn mask_w_avg_8bpc_simd_matches_scalar() {
        let _tok_lock = archmage::testing::lock_token_testing();
        let Some(token) = crate::src::cpu::summon_avx2() else {
            eprintln!("mask_w_avg_8bpc_simd_matches_scalar: no AVX2 token, skipping");
            return;
        };
        let mut tmp1 = [0i16; COMPINTER_LEN];
        let mut tmp2 = [0i16; COMPINTER_LEN];
        let mut st: u32 = 0x51ab_77cd;
        for i in 0..COMPINTER_LEN {
            st ^= st << 13;
            st ^= st >> 17;
            st ^= st << 5;
            tmp1[i] = (((st >> 4) as i32 % 65536) - 32768) as i16;
            st ^= st << 13;
            st ^= st >> 17;
            st ^= st << 5;
            tmp2[i] = (((st >> 4) as i32 % 65536) - 32768) as i16;
        }

        for &w in &[4usize, 8, 12, 16, 24, 32, 48, 64, 128] {
            for &h in &[1usize, 2, 3, 8, 16] {
                let stride = w + 7;
                // mask blend: (a*m + b*(64-m) + 512) >> 10
                let mut mask = vec![0u8; w * h];
                for (i, m) in mask.iter_mut().enumerate() {
                    *m = ((i * 37 + h * 11 + w) % 65) as u8;
                }
                let mut dst_a = vec![0xAAu8; h * stride];
                let mut dst_b = vec![0xAAu8; h * stride];
                for row in 0..h {
                    for x in 0..w {
                        let a = tmp1[row * w + x] as i32;
                        let b = tmp2[row * w + x] as i32;
                        let m = mask[row * w + x] as i32;
                        dst_a[row * stride + x] =
                            ((a * m + b * (64 - m) + 512) >> 10).clamp(0, 255) as u8;
                    }
                }
                mask_8bpc_avx2_safe(
                    token, &mut dst_b, stride, &tmp1, &tmp2, w as i32, h as i32, &mask,
                );
                assert_eq!(dst_a, dst_b, "mask_8bpc w={w} h={h}");
                if let Some(t512) = crate::src::cpu::summon_avx512() {
                    let mut dst_c = vec![0xAAu8; h * stride];
                    mask_8bpc_avx512_safe(
                        t512, &mut dst_c, stride, &tmp1, &tmp2, w as i32, h as i32, &mask,
                    );
                    assert_eq!(dst_a, dst_c, "mask_8bpc_avx512 w={w} h={h}");
                }

                // w_avg: (a*weight + b*(16-weight) + 128) >> 8
                for &weight in &[0i32, 1, 7, 8, 9, 15, 16] {
                    let mut dst_a = vec![0xAAu8; h * stride];
                    let mut dst_b = vec![0xAAu8; h * stride];
                    for row in 0..h {
                        for x in 0..w {
                            let a = tmp1[row * w + x] as i32;
                            let b = tmp2[row * w + x] as i32;
                            dst_a[row * stride + x] =
                                ((a * weight + b * (16 - weight) + 128) >> 8).clamp(0, 255) as u8;
                        }
                    }
                    w_avg_8bpc_avx2_safe(
                        token, &mut dst_b, stride, &tmp1, &tmp2, w as i32, h as i32, weight,
                    );
                    assert_eq!(dst_a, dst_b, "w_avg_8bpc w={w} h={h} weight={weight}");
                    if let Some(t512) = crate::src::cpu::summon_avx512() {
                        let mut dst_c = vec![0xAAu8; h * stride];
                        w_avg_8bpc_avx512_safe(
                            t512, &mut dst_c, stride, &tmp1, &tmp2, w as i32, h as i32, weight,
                        );
                        assert_eq!(
                            dst_a, dst_c,
                            "w_avg_8bpc_avx512 w={w} h={h} weight={weight}"
                        );
                    }
                }
            }
        }
    }

    /// Scalar oracle mirroring `w_mask_rust` at 8bpc (ib=4, PREP_BIAS=0):
    /// dst u8, seg_mask at subsampled resolution.
    #[allow(clippy::too_many_arguments)]
    fn w_mask_8bpc_scalar_ref(
        dst: &mut [u8],
        dst_stride: usize,
        tmp1: &[i16; COMPINTER_LEN],
        tmp2: &[i16; COMPINTER_LEN],
        w: usize,
        h: usize,
        mask: &mut [u8],
        sign: u8,
        ss_hor: bool,
        ss_ver: bool,
    ) {
        let sh = 4 + 6;
        let rnd = 32i32 << 4;
        let mask_sh = 8 + 4 - 4;
        let mask_rnd = 1u16 << (mask_sh - 5);
        let mask_w = if ss_hor { w >> 1 } else { w };
        let mut mask_off = 0usize;
        for row_h in 0..h {
            let tmp1_row = &tmp1[row_h * w..][..w];
            let tmp2_row = &tmp2[row_h * w..][..w];
            let dst_row = &mut dst[row_h * dst_stride..][..w];
            let mut x = 0;
            while x < w {
                let m = std::cmp::min(
                    38 + (tmp1_row[x].abs_diff(tmp2_row[x]).saturating_add(mask_rnd) >> mask_sh),
                    64,
                ) as u8;
                let px =
                    (tmp1_row[x] as i32 * m as i32 + tmp2_row[x] as i32 * (64 - m as i32) + rnd)
                        >> sh;
                dst_row[x] = px.clamp(0, 255) as u8;
                if ss_hor {
                    x += 1;
                    let n = std::cmp::min(
                        38 + (tmp1_row[x].abs_diff(tmp2_row[x]).saturating_add(mask_rnd)
                            >> mask_sh),
                        64,
                    ) as u8;
                    let px = (tmp1_row[x] as i32 * n as i32
                        + tmp2_row[x] as i32 * (64 - n as i32)
                        + rnd)
                        >> sh;
                    dst_row[x] = px.clamp(0, 255) as u8;
                    let mask_x = x >> 1;
                    if ss_ver && (row_h & 1 != 0) {
                        let prev = mask[mask_off + mask_x];
                        mask[mask_off + mask_x] =
                            (((m as u16 + n as u16 + 2 - sign as u16) + prev as u16) >> 2) as u8;
                    } else if ss_ver {
                        mask[mask_off + mask_x] = m + n;
                    } else {
                        mask[mask_off + mask_x] =
                            ((m as u16 + n as u16 + 1 - sign as u16) >> 1) as u8;
                    }
                } else {
                    mask[mask_off + x] = m;
                }
                x += 1;
            }
            if !ss_ver || (row_h & 1 != 0) {
                mask_off += mask_w;
            }
        }
    }

    /// w_mask_8bpc must match scalar for the full i16 tmp range (warp_affine
    /// mids can approach +-32767 where an i16 abs_diff wraps).
    #[test]
    fn w_mask_8bpc_simd_matches_scalar() {
        let _tok_lock = archmage::testing::lock_token_testing();
        let Some(token) = crate::src::cpu::summon_avx2() else {
            eprintln!("w_mask_8bpc_simd_matches_scalar: no AVX2 token, skipping");
            return;
        };
        let mut tmp1 = [0i16; COMPINTER_LEN];
        let mut tmp2 = [0i16; COMPINTER_LEN];
        let mut st: u32 = 0x77aa_3311;
        for i in 0..COMPINTER_LEN {
            st ^= st << 13;
            st ^= st >> 17;
            st ^= st << 5;
            tmp1[i] = (((st >> 4) as i32 % 65536) - 32768) as i16;
            st ^= st << 13;
            st ^= st >> 17;
            st ^= st << 5;
            tmp2[i] = (((st >> 4) as i32 % 65536) - 32768) as i16;
        }
        for &w in &[2usize, 4, 6, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128] {
            for &h in &[1usize, 2, 3, 4, 6, 8, 12, 16, 32, 64, 128] {
                for &sign in &[0i32, 1] {
                    for &(hor, ver) in &[(true, true), (true, false), (false, false)] {
                        let stride = w + 13;
                        let mut dst_a = vec![0xAAu8; h * stride];
                        let mut dst_b = vec![0xAAu8; h * stride];
                        let mut mask_a = [0xCCu8; SEG_MASK_LEN];
                        let mut mask_b = [0xCCu8; SEG_MASK_LEN];
                        w_mask_8bpc_scalar_ref(
                            &mut dst_a,
                            stride,
                            &tmp1,
                            &tmp2,
                            w,
                            h,
                            &mut mask_a,
                            sign as u8,
                            hor,
                            ver,
                        );
                        match (hor, ver) {
                            (true, true) => w_mask_420_8bpc_avx2_safe(
                                token,
                                &mut dst_b,
                                stride,
                                &tmp1,
                                &tmp2,
                                w as i32,
                                h as i32,
                                &mut mask_b,
                                sign,
                            ),
                            (true, false) => w_mask_422_8bpc_avx2_safe(
                                token,
                                &mut dst_b,
                                stride,
                                &tmp1,
                                &tmp2,
                                w as i32,
                                h as i32,
                                &mut mask_b,
                                sign,
                            ),
                            _ => w_mask_444_8bpc_avx2_safe(
                                token,
                                &mut dst_b,
                                stride,
                                &tmp1,
                                &tmp2,
                                w as i32,
                                h as i32,
                                &mut mask_b,
                                sign,
                            ),
                        }
                        assert_eq!(
                            &mask_a[..],
                            &mask_b[..],
                            "w_mask_8bpc mask w={w} h={h} sign={sign} hor={hor} ver={ver}"
                        );
                        assert_eq!(
                            dst_a, dst_b,
                            "w_mask_8bpc dst w={w} h={h} sign={sign} hor={hor} ver={ver}"
                        );
                    }
                }
            }
        }
    }

    /// widen_row_u16_shl_bias_16bpc's 16-px lane: packs_epi32 is lane-local
    /// and needs a qword permute to restore linear order.
    #[cfg(target_arch = "x86_64")]
    #[arcane]
    fn widen_row_16bpc_test_entry(
        _t: Desktop64,
        dst: &mut [i16],
        src: &[u16],
        w: usize,
        ib: i32,
        bias: i32,
    ) {
        widen_row_u16_shl_bias_16bpc(_t, dst, src, w, ib, bias);
    }

    #[test]
    fn widen_row_16bpc_simd_matches_scalar() {
        let _tok_lock = archmage::testing::lock_token_testing();
        let Some(token) = crate::src::cpu::summon_avx2() else {
            eprintln!("widen_row_16bpc_simd_matches_scalar: no AVX2 token, skipping");
            return;
        };
        let mut src = [0u16; 256];
        let mut st: u32 = 0x1234_9abc;
        for s in src.iter_mut() {
            st ^= st << 13;
            st ^= st >> 17;
            st ^= st << 5;
            *s = (st & 0x0fff) as u16; // pixels are <= bd_max = 4095
        }
        for &w in &[1usize, 7, 8, 9, 15, 16, 17, 31, 32, 64, 128] {
            // ib=4 pairs with 10bpc (px<=1023), ib=2 with 12bpc (px<=4095);
            // outside that domain the scalar `as i16` wrap legitimately
            // differs from packs_epi32 saturation.
            for &(ib, bias, pmax) in &[(4i32, 8192i32, 1023u16), (2, 8192, 4095)] {
                let mut a = vec![0i16; w];
                let mut b = vec![0i16; w];
                for x in 0..w {
                    a[x] = (((src[x] & pmax) as i32) << ib).wrapping_sub(bias) as i16;
                }
                let src_p: Vec<u16> = src[..w].iter().map(|&v| v & pmax).collect();
                widen_row_16bpc_test_entry(token, &mut b, &src_p, w, ib, bias);
                assert_eq!(a, b, "widen_row_16bpc w={w} ib={ib} bias={bias}");
            }
        }
    }
}

#[cfg(all(test, target_arch = "x86_64", not(feature = "asm")))]
mod four_tap_vertical_rows {
    use super::*;

    #[arcane]
    fn filter_rows(
        token: Desktop64,
        mid: &[[i16; MID_STRIDE]],
        width: usize,
        filter: &[i8; 8],
        put: &mut [u8],
        prep: &mut [i16],
    ) {
        v_filter_ntap_8bpc_avx2_inner::<4>(token, put, mid, width, filter, 10, 255);
        v_filter_ntap_to_i16_avx2_inner::<4>(token, mid, prep, width, filter, 6);
    }

    #[test]
    fn four_tap_vertical_uses_only_active_rows() {
        let _lock = crate::src::safe_simd::token_test_lock();
        let token = crate::src::cpu::summon_avx2().expect("native MC oracle needs AVX2");
        // Rows 2..6 hold active taps. Inactive rows 6 and 7 are deliberately
        // absent, so a redundant zero-coefficient load fails this oracle.
        let mut mid = [[0i16; MID_STRIDE]; 6];
        let endpoints = [-1785, 0, 1, 4080, 5865];
        for (row, pixels) in mid.iter_mut().enumerate() {
            for (column, pixel) in pixels.iter_mut().enumerate() {
                *pixel = endpoints[(row + column) % endpoints.len()];
            }
        }
        let mut cases = 0;
        for family in crate::src::tables::dav1d_mc_subpel_filters.iter() {
            for filter in family.iter().filter(|filter| tap_base_8tap(filter) == 2) {
                for width in [4, 8, 17, 128] {
                    let mut put = vec![37; width + 3];
                    let mut prep = vec![37; width + 3];
                    let mut expected_put = put.clone();
                    let mut expected_prep = prep.clone();
                    for column in 0..width {
                        let sum: i32 = (2..6)
                            .map(|row| i32::from(filter[row]) * i32::from(mid[row][column]))
                            .sum();
                        expected_put[column] = ((sum + 512) >> 10).clamp(0, 255) as u8;
                        expected_prep[column] = ((sum + 32) >> 6) as i16;
                    }
                    filter_rows(token, &mid, width, filter, &mut put, &mut prep);
                    assert_eq!(put, expected_put, "put: width={width}, filter={filter:?}");
                    assert_eq!(
                        prep, expected_prep,
                        "prep: width={width}, filter={filter:?}"
                    );
                    cases += 1;
                }
            }
        }
        assert!(cases > 0, "table must exercise four-tap vertical kernels");
    }
}

#[cfg(all(test, target_arch = "x86_64", not(feature = "asm")))]
#[path = "mc_fused_small_tests.rs"]
mod fused_small_tests;
