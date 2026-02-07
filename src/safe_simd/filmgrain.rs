//! Safe SIMD implementations of film grain synthesis functions
//!
//! Film grain synthesis adds artificial grain to decoded video to match
//! the artistic intent of the original content.
//!
//! Functions implemented:
//! - generate_grain_y: Generates luma grain LUT (scalar - LFSR is serial)
//! - generate_grain_uv: Generates chroma grain LUT (scalar - LFSR is serial)
//! - fgy_32x32xn: Apply luma grain to 32x32 blocks (AVX2 SIMD)
//! - fguv_32x32xn: Apply chroma grain to 32x32 blocks (AVX2 SIMD)

#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use std::cmp;
use std::ffi::c_int;
use std::ffi::c_uint;

use libc::{intptr_t, ptrdiff_t};

#[cfg(target_arch = "x86_64")]
use archmage::{arcane, Desktop64, SimdToken};

use crate::include::common::bitdepth::{DynEntry, DynPixel, DynScaling};
use crate::include::dav1d::headers::{Dav1dFilmGrainData, Rav1dFilmGrainData};
use crate::include::dav1d::picture::PicOffset;
use crate::src::ffi_safe::FFISafe;
use crate::src::filmgrain::{FG_BLOCK_SIZE, GRAIN_HEIGHT, GRAIN_WIDTH};
use crate::src::internal::GrainLut;
use crate::src::tables::dav1d_gaussian_sequence;

// ============================================================================
// Helper functions (match the filmgrain.rs versions)
// ============================================================================

#[inline(always)]
fn get_random_number(bits: u8, state: &mut c_uint) -> c_int {
    let r = *state;
    let bit = (r ^ (r >> 1) ^ (r >> 3) ^ (r >> 12)) & 1;
    *state = (r >> 1) | bit << 15;
    (*state >> (16 - bits) & ((1 << bits) - 1)) as c_int
}

#[inline(always)]
fn round2(x: i32, shift: u8) -> i32 {
    (x + (1i32 << shift >> 1)) >> shift
}

fn row_seed(rows: usize, row_num: usize, data: &Rav1dFilmGrainData) -> [c_uint; 2] {
    let mut seed = [0u32; 2];
    for (i, s) in seed.iter_mut().enumerate().take(rows) {
        *s = data.seed;
        *s ^= ((((row_num - i) * 37 + 178) & 0xFF) << 8) as c_uint;
        *s ^= (((row_num - i) * 173 + 105) & 0xFF) as c_uint;
    }
    seed
}

const AR_PAD: usize = 3;

// ============================================================================
// generate_grain_y - 8bpc (scalar, LFSR is inherently serial)
// ============================================================================

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
pub unsafe extern "C" fn generate_grain_y_8bpc_avx2(
    buf: *mut GrainLut<DynEntry>,
    data: &Dav1dFilmGrainData,
    _bitdepth_max: c_int,
) {
    let buf = unsafe { &mut *buf.cast::<GrainLut<i8>>() };
    let data: Rav1dFilmGrainData = unsafe { data.clone().into() };
    generate_grain_y_inner_8bpc(buf, &data);
}

fn generate_grain_y_inner_8bpc(buf: &mut GrainLut<i8>, data: &Rav1dFilmGrainData) {
    let mut seed = data.seed;
    let shift = 4 + data.grain_scale_shift;

    for row in &mut buf[..GRAIN_HEIGHT] {
        for entry in &mut row[..GRAIN_WIDTH] {
            let value = get_random_number(11, &mut seed);
            *entry = round2(dav1d_gaussian_sequence[value as usize] as i32, shift) as i8;
        }
    }

    let ar_lag = data.ar_coeff_lag as usize & 3;
    if ar_lag == 0 {
        return;
    }

    for y in 0..GRAIN_HEIGHT - AR_PAD {
        for x in 0..GRAIN_WIDTH - 2 * AR_PAD {
            let mut coeff_idx = 0usize;
            let mut sum: i32 = 0;
            for dy in (AR_PAD - ar_lag)..=AR_PAD {
                for dx in (AR_PAD - ar_lag)..=(AR_PAD + ar_lag) {
                    if dx == ar_lag && dy - (AR_PAD - ar_lag) == ar_lag {
                        break;
                    }
                    sum += data.ar_coeffs_y[coeff_idx] as i32 * buf[y + dy][x + dx] as i32;
                    coeff_idx += 1;
                }
            }
            let grain = buf[y + AR_PAD][x + AR_PAD] as i32 + round2(sum, data.ar_coeff_shift);
            buf[y + AR_PAD][x + AR_PAD] = grain.clamp(-128, 127) as i8;
        }
    }
}

// ============================================================================
// generate_grain_y - 16bpc (scalar)
// ============================================================================

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
pub unsafe extern "C" fn generate_grain_y_16bpc_avx2(
    buf: *mut GrainLut<DynEntry>,
    data: &Dav1dFilmGrainData,
    bitdepth_max: c_int,
) {
    let buf = unsafe { &mut *buf.cast::<GrainLut<i16>>() };
    let data: Rav1dFilmGrainData = unsafe { data.clone().into() };
    let bitdepth = if bitdepth_max >= 4095 { 12 } else { 10 };
    generate_grain_y_inner_16bpc(buf, &data, bitdepth);
}

fn generate_grain_y_inner_16bpc(buf: &mut GrainLut<i16>, data: &Rav1dFilmGrainData, bitdepth: u8) {
    let bitdepth_min_8 = (bitdepth - 8) as u8;
    let mut seed = data.seed;
    let shift = 4 - bitdepth_min_8 + data.grain_scale_shift;
    let grain_ctr = 128i32 << bitdepth_min_8;
    let grain_min = -grain_ctr;
    let grain_max = grain_ctr - 1;

    for row in &mut buf[..GRAIN_HEIGHT] {
        for entry in &mut row[..GRAIN_WIDTH] {
            let value = get_random_number(11, &mut seed);
            *entry = round2(dav1d_gaussian_sequence[value as usize] as i32, shift) as i16;
        }
    }

    let ar_lag = data.ar_coeff_lag as usize & 3;
    if ar_lag == 0 {
        return;
    }

    for y in 0..GRAIN_HEIGHT - AR_PAD {
        for x in 0..GRAIN_WIDTH - 2 * AR_PAD {
            let mut coeff_idx = 0usize;
            let mut sum: i32 = 0;
            for dy in (AR_PAD - ar_lag)..=AR_PAD {
                for dx in (AR_PAD - ar_lag)..=(AR_PAD + ar_lag) {
                    if dx == ar_lag && dy - (AR_PAD - ar_lag) == ar_lag {
                        break;
                    }
                    sum += data.ar_coeffs_y[coeff_idx] as i32 * buf[y + dy][x + dx] as i32;
                    coeff_idx += 1;
                }
            }
            let grain = buf[y + AR_PAD][x + AR_PAD] as i32 + round2(sum, data.ar_coeff_shift);
            buf[y + AR_PAD][x + AR_PAD] = grain.clamp(grain_min, grain_max) as i16;
        }
    }
}

// ============================================================================
// generate_grain_uv - 8bpc (scalar, LFSR is serial)
// ============================================================================

fn generate_grain_uv_inner_8bpc(
    buf: &mut GrainLut<i8>,
    buf_y: &GrainLut<i8>,
    data: &Rav1dFilmGrainData,
    is_uv: bool,
    is_subx: bool,
    is_suby: bool,
) {
    let uv = is_uv as usize;
    let (chromah, chromaw) = if is_suby {
        (38usize, if is_subx { 44usize } else { GRAIN_WIDTH })
    } else {
        (GRAIN_HEIGHT, if is_subx { 44 } else { GRAIN_WIDTH })
    };

    let mut seed = data.seed ^ if is_uv { 0x49d8 } else { 0xb524 };
    let shift = 4 + data.grain_scale_shift;

    for row in &mut buf[..chromah] {
        for entry in &mut row[..chromaw] {
            let value = get_random_number(11, &mut seed);
            *entry = round2(dav1d_gaussian_sequence[value as usize] as i32, shift) as i8;
        }
    }

    let ar_lag = data.ar_coeff_lag as usize & 3;

    let len_y = chromah - AR_PAD;
    let len_x = chromaw - 2 * AR_PAD;

    for y in 0..len_y {
        for x in 0..len_x {
            let mut coeff_idx = 0usize;
            let mut sum: i32 = 0;
            for dy in (AR_PAD - ar_lag)..=AR_PAD {
                for dx in (AR_PAD - ar_lag)..=(AR_PAD + ar_lag) {
                    if dx == ar_lag && dy - (AR_PAD - ar_lag) == ar_lag {
                        // Luma contribution
                        let luma_y = (y << is_suby as usize) + AR_PAD;
                        let luma_x = (x << is_subx as usize) + AR_PAD;
                        let mut luma: i32 = 0;
                        for i in 0..1 + is_suby as usize {
                            for j in 0..1 + is_subx as usize {
                                luma += buf_y[luma_y + i][luma_x + j] as i32;
                            }
                        }
                        luma = round2(luma, is_suby as u8 + is_subx as u8);
                        sum += luma * data.ar_coeffs_uv[uv][coeff_idx] as i32;
                        break;
                    }
                    sum += data.ar_coeffs_uv[uv][coeff_idx] as i32 * buf[y + dy][x + dx] as i32;
                    coeff_idx += 1;
                }
            }
            let grain = buf[y + AR_PAD][x + AR_PAD] as i32 + round2(sum, data.ar_coeff_shift);
            buf[y + AR_PAD][x + AR_PAD] = grain.clamp(-128, 127) as i8;
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
pub unsafe extern "C" fn generate_grain_uv_420_8bpc_avx2(
    buf: *mut GrainLut<DynEntry>,
    buf_y: *const GrainLut<DynEntry>,
    data: &Dav1dFilmGrainData,
    uv: intptr_t,
    _bitdepth_max: c_int,
) {
    let buf = unsafe { &mut *buf.cast::<GrainLut<i8>>() };
    let buf_y = unsafe { &*buf_y.cast::<GrainLut<i8>>() };
    let data: Rav1dFilmGrainData = unsafe { data.clone().into() };
    generate_grain_uv_inner_8bpc(buf, buf_y, &data, uv != 0, true, true);
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
pub unsafe extern "C" fn generate_grain_uv_422_8bpc_avx2(
    buf: *mut GrainLut<DynEntry>,
    buf_y: *const GrainLut<DynEntry>,
    data: &Dav1dFilmGrainData,
    uv: intptr_t,
    _bitdepth_max: c_int,
) {
    let buf = unsafe { &mut *buf.cast::<GrainLut<i8>>() };
    let buf_y = unsafe { &*buf_y.cast::<GrainLut<i8>>() };
    let data: Rav1dFilmGrainData = unsafe { data.clone().into() };
    generate_grain_uv_inner_8bpc(buf, buf_y, &data, uv != 0, true, false);
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
pub unsafe extern "C" fn generate_grain_uv_444_8bpc_avx2(
    buf: *mut GrainLut<DynEntry>,
    buf_y: *const GrainLut<DynEntry>,
    data: &Dav1dFilmGrainData,
    uv: intptr_t,
    _bitdepth_max: c_int,
) {
    let buf = unsafe { &mut *buf.cast::<GrainLut<i8>>() };
    let buf_y = unsafe { &*buf_y.cast::<GrainLut<i8>>() };
    let data: Rav1dFilmGrainData = unsafe { data.clone().into() };
    generate_grain_uv_inner_8bpc(buf, buf_y, &data, uv != 0, false, false);
}

// ============================================================================
// generate_grain_uv - 16bpc (scalar)
// ============================================================================

fn generate_grain_uv_inner_16bpc(
    buf: &mut GrainLut<i16>,
    buf_y: &GrainLut<i16>,
    data: &Rav1dFilmGrainData,
    is_uv: bool,
    is_subx: bool,
    is_suby: bool,
    bitdepth: u8,
) {
    let uv = is_uv as usize;
    let bitdepth_min_8 = (bitdepth - 8) as u8;
    let grain_ctr = 128i32 << bitdepth_min_8;
    let grain_min = -grain_ctr;
    let grain_max = grain_ctr - 1;

    let (chromah, chromaw) = if is_suby {
        (38usize, if is_subx { 44usize } else { GRAIN_WIDTH })
    } else {
        (GRAIN_HEIGHT, if is_subx { 44 } else { GRAIN_WIDTH })
    };

    let mut seed = data.seed ^ if is_uv { 0x49d8 } else { 0xb524 };
    let shift = 4 - bitdepth_min_8 + data.grain_scale_shift;

    for row in &mut buf[..chromah] {
        for entry in &mut row[..chromaw] {
            let value = get_random_number(11, &mut seed);
            *entry = round2(dav1d_gaussian_sequence[value as usize] as i32, shift) as i16;
        }
    }

    let ar_lag = data.ar_coeff_lag as usize & 3;

    let len_y = chromah - AR_PAD;
    let len_x = chromaw - 2 * AR_PAD;

    for y in 0..len_y {
        for x in 0..len_x {
            let mut coeff_idx = 0usize;
            let mut sum: i32 = 0;
            for dy in (AR_PAD - ar_lag)..=AR_PAD {
                for dx in (AR_PAD - ar_lag)..=(AR_PAD + ar_lag) {
                    if dx == ar_lag && dy - (AR_PAD - ar_lag) == ar_lag {
                        let luma_y = (y << is_suby as usize) + AR_PAD;
                        let luma_x = (x << is_subx as usize) + AR_PAD;
                        let mut luma: i32 = 0;
                        for i in 0..1 + is_suby as usize {
                            for j in 0..1 + is_subx as usize {
                                luma += buf_y[luma_y + i][luma_x + j] as i32;
                            }
                        }
                        luma = round2(luma, is_suby as u8 + is_subx as u8);
                        sum += luma * data.ar_coeffs_uv[uv][coeff_idx] as i32;
                        break;
                    }
                    sum += data.ar_coeffs_uv[uv][coeff_idx] as i32 * buf[y + dy][x + dx] as i32;
                    coeff_idx += 1;
                }
            }
            let grain = buf[y + AR_PAD][x + AR_PAD] as i32 + round2(sum, data.ar_coeff_shift);
            buf[y + AR_PAD][x + AR_PAD] = grain.clamp(grain_min, grain_max) as i16;
        }
    }
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
pub unsafe extern "C" fn generate_grain_uv_420_16bpc_avx2(
    buf: *mut GrainLut<DynEntry>,
    buf_y: *const GrainLut<DynEntry>,
    data: &Dav1dFilmGrainData,
    uv: intptr_t,
    bitdepth_max: c_int,
) {
    let buf = unsafe { &mut *buf.cast::<GrainLut<i16>>() };
    let buf_y = unsafe { &*buf_y.cast::<GrainLut<i16>>() };
    let data: Rav1dFilmGrainData = unsafe { data.clone().into() };
    let bitdepth = if bitdepth_max >= 4095 { 12 } else { 10 };
    generate_grain_uv_inner_16bpc(buf, buf_y, &data, uv != 0, true, true, bitdepth);
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
pub unsafe extern "C" fn generate_grain_uv_422_16bpc_avx2(
    buf: *mut GrainLut<DynEntry>,
    buf_y: *const GrainLut<DynEntry>,
    data: &Dav1dFilmGrainData,
    uv: intptr_t,
    bitdepth_max: c_int,
) {
    let buf = unsafe { &mut *buf.cast::<GrainLut<i16>>() };
    let buf_y = unsafe { &*buf_y.cast::<GrainLut<i16>>() };
    let data: Rav1dFilmGrainData = unsafe { data.clone().into() };
    let bitdepth = if bitdepth_max >= 4095 { 12 } else { 10 };
    generate_grain_uv_inner_16bpc(buf, buf_y, &data, uv != 0, true, false, bitdepth);
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
pub unsafe extern "C" fn generate_grain_uv_444_16bpc_avx2(
    buf: *mut GrainLut<DynEntry>,
    buf_y: *const GrainLut<DynEntry>,
    data: &Dav1dFilmGrainData,
    uv: intptr_t,
    bitdepth_max: c_int,
) {
    let buf = unsafe { &mut *buf.cast::<GrainLut<i16>>() };
    let buf_y = unsafe { &*buf_y.cast::<GrainLut<i16>>() };
    let data: Rav1dFilmGrainData = unsafe { data.clone().into() };
    let bitdepth = if bitdepth_max >= 4095 { 12 } else { 10 };
    generate_grain_uv_inner_16bpc(buf, buf_y, &data, uv != 0, false, false, bitdepth);
}

// ============================================================================
// fgy_32x32xn - 8bpc AVX2 (apply luma grain)
// ============================================================================

/// Compute grain offsets from random seed value
#[inline(always)]
fn grain_offsets(randval: c_int, is_subx: bool, is_suby: bool) -> (usize, usize) {
    let subx = is_subx as usize;
    let suby = is_suby as usize;
    let offx = 3 + (2 >> subx) * (3 + ((randval as usize) >> 4));
    let offy = 3 + (2 >> suby) * (3 + ((randval as usize) & 0xF));
    (offx, offy)
}

/// Inner SIMD loop for fgy: process pixels using AVX2 with safe slice access.
/// Uses scalar scaling lookups + SIMD multiply/round/add/clamp.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn fgy_row_simd_8bpc_safe(
    _token: Desktop64,
    dst: &mut [u8],
    src: &[u8],
    scaling: &[u8],
    grain_row: &[i8],
    bw: usize,
    xstart: usize,
    mul: __m256i,
    min_vec: __m256i,
    max_vec: __m256i,
    scaling_shift: u8,
    min_value: i32,
    max_value: i32,
) {
    use safe_unaligned_simd::x86_64 as safe_simd;
    let zero = _mm256_setzero_si256();

    // Process aligned 32-pixel chunks
    let mut x = xstart;
    while x + 32 <= bw {
        // Load 32 source pixels
        let src_arr: &[u8; 32] = src[x..x + 32].try_into().unwrap();
        let src_vec = safe_simd::_mm256_loadu_si256(src_arr);
        let src_lo = _mm256_unpacklo_epi8(src_vec, zero);
        let src_hi = _mm256_unpackhi_epi8(src_vec, zero);

        // Scalar scaling lookup, pack into vectors
        let mut sc_lo_bytes = [0u8; 32];
        let mut sc_hi_bytes = [0u8; 32];

        // Low lane (bytes 0-7)
        for i in 0..8 {
            sc_lo_bytes[i * 2] = scaling[src[x + i] as usize];
        }
        // High lane of lo (bytes 16-23)
        for i in 0..8 {
            sc_lo_bytes[16 + i * 2] = scaling[src[x + 16 + i] as usize];
        }
        // Low lane of hi (bytes 8-15)
        for i in 0..8 {
            sc_hi_bytes[i * 2] = scaling[src[x + 8 + i] as usize];
        }
        // High lane of hi (bytes 24-31)
        for i in 0..8 {
            sc_hi_bytes[16 + i * 2] = scaling[src[x + 24 + i] as usize];
        }

        let sc_lo = safe_simd::_mm256_loadu_si256(&sc_lo_bytes);
        let sc_hi = safe_simd::_mm256_loadu_si256(&sc_hi_bytes);

        // Load 32 grain values as bytes for SIMD
        let mut grain_bytes = [0u8; 32];
        for i in 0..32 {
            grain_bytes[i] = grain_row[x + i] as u8;
        }
        let grain_vec = safe_simd::_mm256_loadu_si256(&grain_bytes);
        let grain_lo = _mm256_unpacklo_epi8(grain_vec, zero);
        let grain_hi = _mm256_unpackhi_epi8(grain_vec, zero);

        let noise_lo = _mm256_maddubs_epi16(sc_lo, grain_lo);
        let noise_hi = _mm256_maddubs_epi16(sc_hi, grain_hi);
        let noise_lo = _mm256_mulhrs_epi16(noise_lo, mul);
        let noise_hi = _mm256_mulhrs_epi16(noise_hi, mul);
        let result_lo = _mm256_add_epi16(src_lo, noise_lo);
        let result_hi = _mm256_add_epi16(src_hi, noise_hi);
        let result = _mm256_packus_epi16(result_lo, result_hi);
        let result = _mm256_max_epu8(result, min_vec);
        let result = _mm256_min_epu8(result, max_vec);

        let dst_arr: &mut [u8; 32] = (&mut dst[x..x + 32]).try_into().unwrap();
        safe_simd::_mm256_storeu_si256(dst_arr, result);
        x += 32;
    }

    // Process remaining 16-pixel chunk if present
    if x + 16 <= bw {
        let src_arr: &[u8; 16] = src[x..x + 16].try_into().unwrap();
        let src_vec = safe_simd::_mm_loadu_si128(src_arr);
        let src_lo = _mm256_cvtepu8_epi16(src_vec);

        let mut sc_bytes = [0u8; 32];
        for i in 0..8 {
            sc_bytes[i * 2] = scaling[src[x + i] as usize];
        }
        for i in 0..8 {
            sc_bytes[16 + i * 2] = scaling[src[x + 8 + i] as usize];
        }

        let sc_vec = safe_simd::_mm256_loadu_si256(&sc_bytes);

        let mut grain_16 = [0u8; 16];
        for i in 0..16 {
            grain_16[i] = grain_row[x + i] as u8;
        }
        let grain_128 = safe_simd::_mm_loadu_si128(&grain_16);
        let grain_interleaved = _mm256_unpacklo_epi8(_mm256_castsi128_si256(grain_128), zero);
        let grain_128_hi = _mm_srli_si128::<8>(grain_128);
        let grain_interleaved_hi = _mm256_unpacklo_epi8(_mm256_castsi128_si256(grain_128_hi), zero);
        let grain_combined = _mm256_inserti128_si256(
            grain_interleaved,
            _mm256_castsi256_si128(grain_interleaved_hi),
            1,
        );

        let noise = _mm256_maddubs_epi16(sc_vec, grain_combined);
        let noise = _mm256_mulhrs_epi16(noise, mul);
        let result = _mm256_add_epi16(src_lo, noise);
        let result = _mm256_packus_epi16(result, result);

        let lo128 = _mm256_castsi256_si128(result);
        let hi128 = _mm256_extracti128_si256::<1>(result);
        let combined = _mm_unpacklo_epi64(lo128, hi128);
        let combined = _mm_max_epu8(combined, _mm256_castsi256_si128(min_vec));
        let combined = _mm_min_epu8(combined, _mm256_castsi256_si128(max_vec));

        let dst_arr: &mut [u8; 16] = (&mut dst[x..x + 16]).try_into().unwrap();
        safe_simd::_mm_storeu_si128(dst_arr, combined);
        x += 16;
    }

    // Scalar for remaining pixels
    while x < bw {
        let sv = src[x] as usize;
        let grain = grain_row[x] as i32;
        let sc = scaling[sv] as i32;
        let noise = round2(sc * grain, scaling_shift);
        let result = (src[x] as i32 + noise).clamp(min_value, max_value);
        dst[x] = result as u8;
        x += 1;
    }
}

/// Inner SIMD loop for fgy: process 32 pixels at once using AVX2 (raw pointer version for FFI).
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
fn fgy_row_simd_8bpc(
    _token: Desktop64,
    dst: *mut u8,
    src: *const u8,
    scaling: *const u8,
    grain_row: *const i8,
    bw: usize,
    xstart: usize,
    mul: __m256i,
    min_vec: __m256i,
    max_vec: __m256i,
    _offsets: &[[c_int; 2]; 2],
    _grain_lut: *const [[i8; GRAIN_WIDTH]; GRAIN_HEIGHT + 1],
    _offy: usize,
    _y: usize,
    _grain_min: i32,
    _grain_max: i32,
    scaling_shift: u8,
) {
    let zero = _mm256_setzero_si256();

    // Process aligned 32-pixel chunks
    let mut x = xstart;
    while x + 32 <= bw {
        let src_vec = unsafe { _mm256_loadu_si256(src.add(x) as *const __m256i) };
        let src_lo = _mm256_unpacklo_epi8(src_vec, zero);
        let src_hi = _mm256_unpackhi_epi8(src_vec, zero);

        let mut sc_lo_bytes = [0u8; 32];
        let mut sc_hi_bytes = [0u8; 32];

        for i in 0..8 {
            unsafe {
                let sv = *src.add(x + i);
                sc_lo_bytes[i * 2] = *scaling.add(sv as usize);
            }
        }
        for i in 0..8 {
            unsafe {
                let sv = *src.add(x + 16 + i);
                sc_lo_bytes[16 + i * 2] = *scaling.add(sv as usize);
            }
        }
        for i in 0..8 {
            unsafe {
                let sv = *src.add(x + 8 + i);
                sc_hi_bytes[i * 2] = *scaling.add(sv as usize);
            }
        }
        for i in 0..8 {
            unsafe {
                let sv = *src.add(x + 24 + i);
                sc_hi_bytes[16 + i * 2] = *scaling.add(sv as usize);
            }
        }

        let sc_lo = unsafe { _mm256_loadu_si256(sc_lo_bytes.as_ptr() as *const __m256i) };
        let sc_hi = unsafe { _mm256_loadu_si256(sc_hi_bytes.as_ptr() as *const __m256i) };

        let grain_vec = unsafe { _mm256_loadu_si256(grain_row.add(x) as *const __m256i) };
        let grain_lo = _mm256_unpacklo_epi8(grain_vec, zero);
        let grain_hi = _mm256_unpackhi_epi8(grain_vec, zero);

        let noise_lo = _mm256_maddubs_epi16(sc_lo, grain_lo);
        let noise_hi = _mm256_maddubs_epi16(sc_hi, grain_hi);
        let noise_lo = _mm256_mulhrs_epi16(noise_lo, mul);
        let noise_hi = _mm256_mulhrs_epi16(noise_hi, mul);
        let result_lo = _mm256_add_epi16(src_lo, noise_lo);
        let result_hi = _mm256_add_epi16(src_hi, noise_hi);
        let result = _mm256_packus_epi16(result_lo, result_hi);
        let result = _mm256_max_epu8(result, min_vec);
        let result = _mm256_min_epu8(result, max_vec);

        unsafe { _mm256_storeu_si256(dst.add(x) as *mut __m256i, result) };
        x += 32;
    }

    if x + 16 <= bw {
        let src_vec = unsafe { _mm_loadu_si128(src.add(x) as *const __m128i) };
        let src_lo = _mm256_cvtepu8_epi16(src_vec);

        let mut sc_bytes = [0u8; 32];
        for i in 0..8 {
            unsafe {
                let sv = *src.add(x + i);
                sc_bytes[i * 2] = *scaling.add(sv as usize);
            }
        }
        for i in 0..8 {
            unsafe {
                let sv = *src.add(x + 8 + i);
                sc_bytes[16 + i * 2] = *scaling.add(sv as usize);
            }
        }

        let sc_vec = unsafe { _mm256_loadu_si256(sc_bytes.as_ptr() as *const __m256i) };

        let grain_bytes = unsafe { _mm_loadu_si128(grain_row.add(x) as *const __m128i) };
        let _grain_lo = _mm256_cvtepi8_epi16(grain_bytes);
        let grain_interleaved = _mm256_unpacklo_epi8(_mm256_castsi128_si256(grain_bytes), zero);
        let grain_128_hi = unsafe { _mm_srli_si128(grain_bytes, 8) };
        let grain_interleaved_hi = _mm256_unpacklo_epi8(_mm256_castsi128_si256(grain_128_hi), zero);
        let grain_combined = _mm256_inserti128_si256(
            grain_interleaved,
            _mm256_castsi256_si128(grain_interleaved_hi),
            1,
        );

        let noise = _mm256_maddubs_epi16(sc_vec, grain_combined);
        let noise = _mm256_mulhrs_epi16(noise, mul);
        let result = _mm256_add_epi16(src_lo, noise);
        let result = _mm256_packus_epi16(result, result);

        let lo128 = _mm256_castsi256_si128(result);
        let hi128 = _mm256_extracti128_si256::<1>(result);
        let combined = unsafe { _mm_unpacklo_epi64(lo128, hi128) };
        let combined = unsafe { _mm_max_epu8(combined, _mm256_castsi256_si128(min_vec)) };
        let combined = unsafe { _mm_min_epu8(combined, _mm256_castsi256_si128(max_vec)) };

        unsafe { _mm_storeu_si128(dst.add(x) as *mut __m128i, combined) };
        x += 16;
    }

    while x < bw {
        unsafe {
            let sv = *src.add(x) as usize;
            let grain = *grain_row.add(x) as i32;
            let sc = *scaling.add(sv) as i32;
            let noise = round2(sc * grain, scaling_shift);
            let result = (*src.add(x) as i32 + noise).clamp(
                _mm256_extract_epi8::<0>(min_vec) as i32,
                _mm256_extract_epi8::<0>(max_vec) as i32,
            );
            *dst.add(x) = result as u8;
        }
        x += 1;
    }
}

/// Apply luma grain - 8bpc AVX2 (FFI entry point for asm mode)
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn fgy_32x32xn_8bpc_avx2(
    dst_row_ptr: *mut DynPixel,
    src_row_ptr: *const DynPixel,
    stride: ptrdiff_t,
    data: &Dav1dFilmGrainData,
    pw: usize,
    scaling: *const DynScaling,
    grain_lut: *const GrainLut<DynEntry>,
    bh: c_int,
    row_num: c_int,
    _bitdepth_max: c_int,
    _dst_row: *const FFISafe<PicOffset>,
    _src_row: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };

    let dst = dst_row_ptr as *mut u8;
    let src = src_row_ptr as *const u8;
    let stride = stride;
    let scaling = scaling as *const u8;
    let grain_lut = grain_lut as *const [[i8; GRAIN_WIDTH]; GRAIN_HEIGHT + 1];
    let data: Rav1dFilmGrainData = data.clone().into();
    let bh = bh as usize;
    let row_num = row_num as usize;

    let rows = 1 + (data.overlap_flag && row_num > 0) as usize;
    let scaling_shift = data.scaling_shift;

    let (min_value, max_value): (i32, i32) = if data.clip_to_restricted_range {
        (16, 235)
    } else {
        (0, 255)
    };

    let mut seed = row_seed(rows, row_num, &data);

    // SIMD constants
    let mul = _mm256_set1_epi16(1i16 << (15 - scaling_shift));
    let min_vec = _mm256_set1_epi8(min_value as i8);
    let max_vec = _mm256_set1_epi8(max_value as u8 as i8);

    let mut offsets: [[c_int; 2]; 2] = [[0; 2]; 2];

    static W: [[i32; 2]; 2] = [[27, 17], [17, 27]];

    for bx in (0..pw).step_by(FG_BLOCK_SIZE) {
        let bw = cmp::min(FG_BLOCK_SIZE, pw - bx);

        if data.overlap_flag && bx != 0 {
            for i in 0..rows {
                offsets[1][i] = offsets[0][i];
            }
        }
        for i in 0..rows {
            offsets[0][i] = get_random_number(8, &mut seed[i]);
        }

        let ystart = if data.overlap_flag && row_num != 0 {
            cmp::min(2, bh)
        } else {
            0
        };
        let xstart = if data.overlap_flag && bx != 0 {
            cmp::min(2, bw)
        } else {
            0
        };

        let (offx, offy) = grain_offsets(offsets[0][0], false, false);
        let (prev_offx, _) = if data.overlap_flag && bx != 0 {
            grain_offsets(offsets[1][0], false, false)
        } else {
            (0, 0)
        };
        let (_, prev_offy) = if data.overlap_flag && row_num != 0 {
            grain_offsets(offsets[0][1], false, false)
        } else {
            (0, 0)
        };

        // Main body: no overlap
        for y in ystart..bh {
            let (src_ptr, dst_ptr, grain_row) = unsafe {
                let src_ptr = src.offset(y as isize * stride).add(bx);
                let dst_ptr = dst.offset(y as isize * stride).add(bx);
                let grain_row = (*grain_lut)[offy + y].as_ptr().add(offx);
                (src_ptr, dst_ptr, grain_row)
            };

            // Handle x-overlap region with scalar blending
            for x in 0..xstart {
                unsafe {
                    let sv = *src_ptr.add(x) as usize;
                    let grain = (*grain_lut)[offy + y][offx + x] as i32;
                    let old = (*grain_lut)[offy + y][prev_offx + x + FG_BLOCK_SIZE] as i32;
                    let blended = round2(old * W[x][0] + grain * W[x][1], 5);
                    let blended = blended.clamp(-128, 127);
                    let sc = *scaling.add(sv) as i32;
                    let noise = round2(sc * blended, scaling_shift);
                    *dst_ptr.add(x) =
                        ((*src_ptr.add(x) as i32 + noise).clamp(min_value, max_value)) as u8;
                }
            }

            // SIMD for the rest
            fgy_row_simd_8bpc(
                token,
                dst_ptr,
                src_ptr,
                scaling,
                grain_row,
                bw,
                xstart,
                mul,
                min_vec,
                max_vec,
                &offsets,
                grain_lut,
                offy,
                y,
                -128,
                127,
                scaling_shift,
            );
        }

        // y-overlap rows
        for y in 0..ystart {
            let (src_ptr, dst_ptr) = unsafe {
                let src_ptr = src.offset(y as isize * stride).add(bx);
                let dst_ptr = dst.offset(y as isize * stride).add(bx);
                (src_ptr, dst_ptr)
            };

            for x in xstart..bw {
                unsafe {
                    let sv = *src_ptr.add(x) as usize;
                    let grain = (*grain_lut)[offy + y][offx + x] as i32;
                    let old = (*grain_lut)[prev_offy + y + FG_BLOCK_SIZE][offx + x] as i32;
                    let blended = round2(old * W[y][0] + grain * W[y][1], 5);
                    let blended = blended.clamp(-128, 127);
                    let sc = *scaling.add(sv) as i32;
                    let noise = round2(sc * blended, scaling_shift);
                    *dst_ptr.add(x) =
                        ((*src_ptr.add(x) as i32 + noise).clamp(min_value, max_value)) as u8;
                }
            }

            // Corner overlap (both x and y)
            for x in 0..xstart {
                unsafe {
                    let sv = *src_ptr.add(x) as usize;

                    let top = (*grain_lut)[prev_offy + y + FG_BLOCK_SIZE][offx + x] as i32;
                    let old_top = (*grain_lut)[prev_offy + y + FG_BLOCK_SIZE]
                        [prev_offx + x + FG_BLOCK_SIZE] as i32;
                    let top = round2(old_top * W[x][0] + top * W[x][1], 5);
                    let top = top.clamp(-128, 127);

                    let grain = (*grain_lut)[offy + y][offx + x] as i32;
                    let old = (*grain_lut)[offy + y][prev_offx + x + FG_BLOCK_SIZE] as i32;
                    let grain = round2(old * W[x][0] + grain * W[x][1], 5);
                    let grain = grain.clamp(-128, 127);

                    let blended = round2(top * W[y][0] + grain * W[y][1], 5);
                    let blended = blended.clamp(-128, 127);
                    let sc = *scaling.add(sv) as i32;
                    let noise = round2(sc * blended, scaling_shift);
                    *dst_ptr.add(x) =
                        ((*src_ptr.add(x) as i32 + noise).clamp(min_value, max_value)) as u8;
                }
            }
        }
    }
}

// ============================================================================
// fgy_32x32xn - 16bpc AVX2
// ============================================================================

/// Apply luma grain - 16bpc AVX2 (FFI entry point for asm mode)
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn fgy_32x32xn_16bpc_avx2(
    dst_row_ptr: *mut DynPixel,
    src_row_ptr: *const DynPixel,
    stride: ptrdiff_t,
    data: &Dav1dFilmGrainData,
    pw: usize,
    scaling: *const DynScaling,
    grain_lut: *const GrainLut<DynEntry>,
    bh: c_int,
    row_num: c_int,
    bitdepth_max: c_int,
    _dst_row: *const FFISafe<PicOffset>,
    _src_row: *const FFISafe<PicOffset>,
) {
    let dst = dst_row_ptr as *mut u16;
    let src = src_row_ptr as *const u16;
    let stride_u16 = stride / 2;
    let scaling = scaling as *const u8;
    let grain_lut = grain_lut as *const [[i16; GRAIN_WIDTH]; GRAIN_HEIGHT + 1];
    let data: Rav1dFilmGrainData = data.clone().into();
    let bh = bh as usize;
    let row_num = row_num as usize;
    let bitdepth_max = bitdepth_max;

    let bitdepth_min_8 = if bitdepth_max >= 4095 { 4u8 } else { 2u8 };
    let grain_ctr = 128i32 << bitdepth_min_8;
    let grain_min = -grain_ctr;
    let grain_max = grain_ctr - 1;

    let rows = 1 + (data.overlap_flag && row_num > 0) as usize;
    let scaling_shift = data.scaling_shift;

    let (min_value, max_value): (i32, i32) = if data.clip_to_restricted_range {
        (16 << bitdepth_min_8 as i32, 235 << bitdepth_min_8 as i32)
    } else {
        (0, bitdepth_max as i32)
    };

    let mut seed = row_seed(rows, row_num, &data);
    let mut offsets: [[c_int; 2]; 2] = [[0; 2]; 2];

    static W: [[i32; 2]; 2] = [[27, 17], [17, 27]];

    let min_vec = _mm256_set1_epi16(min_value as i16);
    let max_vec = _mm256_set1_epi16(max_value as i16);

    for bx in (0..pw).step_by(FG_BLOCK_SIZE) {
        let bw = cmp::min(FG_BLOCK_SIZE, pw - bx);

        if data.overlap_flag && bx != 0 {
            for i in 0..rows {
                offsets[1][i] = offsets[0][i];
            }
        }
        for i in 0..rows {
            offsets[0][i] = get_random_number(8, &mut seed[i]);
        }

        let ystart = if data.overlap_flag && row_num != 0 {
            cmp::min(2, bh)
        } else {
            0
        };
        let xstart = if data.overlap_flag && bx != 0 {
            cmp::min(2, bw)
        } else {
            0
        };

        let (offx, offy) = grain_offsets(offsets[0][0], false, false);
        let (prev_offx, _) = if data.overlap_flag && bx != 0 {
            grain_offsets(offsets[1][0], false, false)
        } else {
            (0, 0)
        };
        let (_, prev_offy) = if data.overlap_flag && row_num != 0 {
            grain_offsets(offsets[0][1], false, false)
        } else {
            (0, 0)
        };

        for y in ystart..bh {
            let (src_ptr, dst_ptr) = unsafe {
                let src_ptr = src.offset(y as isize * stride_u16 as isize).add(bx);
                let dst_ptr = dst.offset(y as isize * stride_u16 as isize).add(bx);
                (src_ptr, dst_ptr)
            };

            // Handle x-overlap with scalar
            for x in 0..xstart {
                unsafe {
                    let sv = *src_ptr.add(x) as usize;
                    let grain = (*grain_lut)[offy + y][offx + x] as i32;
                    let old = (*grain_lut)[offy + y][prev_offx + x + FG_BLOCK_SIZE] as i32;
                    let blended = round2(old * W[x][0] + grain * W[x][1], 5);
                    let blended = blended.clamp(grain_min, grain_max);
                    let sc = *scaling.add(cmp::min(sv, bitdepth_max as usize)) as i32;
                    let noise = round2(sc * blended, scaling_shift);
                    *dst_ptr.add(x) =
                        ((*src_ptr.add(x) as i32 + noise).clamp(min_value, max_value)) as u16;
                }
            }

            // SIMD for main body - 16 pixels at a time
            let mut x = xstart;
            while x + 16 <= bw {
                let src_vec = unsafe { _mm256_loadu_si256(src_ptr.add(x) as *const __m256i) };

                // Scalar scaling lookup (16 values)
                let mut noise_vals = [0i16; 16];
                for i in 0..16 {
                    unsafe {
                        let sv = cmp::min(*src_ptr.add(x + i) as usize, bitdepth_max as usize);
                        let grain = (*grain_lut)[offy + y][offx + x + i] as i32;
                        let sc = *scaling.add(sv) as i32;
                        noise_vals[i] = round2(sc * grain, scaling_shift) as i16;
                    }
                }

                unsafe {
                    let noise = _mm256_loadu_si256(noise_vals.as_ptr() as *const __m256i);
                    let result = _mm256_add_epi16(src_vec, noise);
                    let result = _mm256_max_epi16(result, min_vec);
                    let result = _mm256_min_epi16(result, max_vec);
                    _mm256_storeu_si256(dst_ptr.add(x) as *mut __m256i, result);
                }

                x += 16;
            }

            // Scalar remainder
            while x < bw {
                unsafe {
                    let sv = cmp::min(*src_ptr.add(x) as usize, bitdepth_max as usize);
                    let grain = (*grain_lut)[offy + y][offx + x] as i32;
                    let sc = *scaling.add(sv) as i32;
                    let noise = round2(sc * grain, scaling_shift);
                    *dst_ptr.add(x) =
                        ((*src_ptr.add(x) as i32 + noise).clamp(min_value, max_value)) as u16;
                }
                x += 1;
            }
        }

        // y-overlap rows (scalar)
        for y in 0..ystart {
            let (src_ptr, dst_ptr) = unsafe {
                let src_ptr = src.offset(y as isize * stride_u16 as isize).add(bx);
                let dst_ptr = dst.offset(y as isize * stride_u16 as isize).add(bx);
                (src_ptr, dst_ptr)
            };

            for x in xstart..bw {
                unsafe {
                    let sv = cmp::min(*src_ptr.add(x) as usize, bitdepth_max as usize);
                    let grain = (*grain_lut)[offy + y][offx + x] as i32;
                    let old = (*grain_lut)[prev_offy + y + FG_BLOCK_SIZE][offx + x] as i32;
                    let blended = round2(old * W[y][0] + grain * W[y][1], 5);
                    let blended = blended.clamp(grain_min, grain_max);
                    let sc = *scaling.add(sv) as i32;
                    let noise = round2(sc * blended, scaling_shift);
                    *dst_ptr.add(x) =
                        ((*src_ptr.add(x) as i32 + noise).clamp(min_value, max_value)) as u16;
                }
            }

            for x in 0..xstart {
                unsafe {
                    let sv = cmp::min(*src_ptr.add(x) as usize, bitdepth_max as usize);
                    let top = (*grain_lut)[prev_offy + y + FG_BLOCK_SIZE][offx + x] as i32;
                    let old_top = (*grain_lut)[prev_offy + y + FG_BLOCK_SIZE]
                        [prev_offx + x + FG_BLOCK_SIZE] as i32;
                    let top =
                        round2(old_top * W[x][0] + top * W[x][1], 5).clamp(grain_min, grain_max);

                    let grain = (*grain_lut)[offy + y][offx + x] as i32;
                    let old = (*grain_lut)[offy + y][prev_offx + x + FG_BLOCK_SIZE] as i32;
                    let grain =
                        round2(old * W[x][0] + grain * W[x][1], 5).clamp(grain_min, grain_max);

                    let blended =
                        round2(top * W[y][0] + grain * W[y][1], 5).clamp(grain_min, grain_max);
                    let sc = *scaling.add(sv) as i32;
                    let noise = round2(sc * blended, scaling_shift);
                    *dst_ptr.add(x) =
                        ((*src_ptr.add(x) as i32 + noise).clamp(min_value, max_value)) as u16;
                }
            }
        }
    }
}

// ============================================================================
// fguv_32x32xn - 8bpc AVX2 (apply chroma grain)
// ============================================================================

/// Apply chroma grain - 8bpc AVX2 (raw pointer version for FFI)
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
fn fguv_inner_8bpc(
    _token: Desktop64,
    dst: *mut u8,
    src: *const u8,
    stride: isize,
    data: &Rav1dFilmGrainData,
    pw: usize,
    scaling: *const u8,
    grain_lut: *const [[i8; GRAIN_WIDTH]; GRAIN_HEIGHT + 1],
    bh: usize,
    row_num: usize,
    luma: *const u8,
    luma_stride: isize,
    is_uv: bool,
    is_id: bool,
    is_sx: bool,
    is_sy: bool,
) {
    let uv = is_uv as usize;
    let sx = is_sx as usize;
    let sy = is_sy as usize;

    let rows = 1 + (data.overlap_flag && row_num > 0) as usize;
    let scaling_shift = data.scaling_shift;

    let (min_value, max_value): (i32, i32) = if data.clip_to_restricted_range {
        (16, if is_id { 235 } else { 240 })
    } else {
        (0, 255)
    };

    let grain_min = -128i32;
    let grain_max = 127i32;

    let mut seed = row_seed(rows, row_num, data);
    let mut offsets: [[c_int; 2]; 2] = [[0; 2]; 2];

    static W: [[[i32; 2]; 2]; 2] = [[[27, 17], [17, 27]], [[23, 22], [0; 2]]];

    let mul = _mm256_set1_epi16(1i16 << (15 - scaling_shift));
    let min_vec = _mm256_set1_epi8(min_value as i8);
    let max_vec = _mm256_set1_epi8(max_value as u8 as i8);
    let zero = _mm256_setzero_si256();

    for bx in (0..pw).step_by(FG_BLOCK_SIZE >> sx) {
        let bw = cmp::min(FG_BLOCK_SIZE >> sx, pw - bx);

        if data.overlap_flag && bx != 0 {
            for i in 0..rows {
                offsets[1][i] = offsets[0][i];
            }
        }
        for i in 0..rows {
            offsets[0][i] = get_random_number(8, &mut seed[i]);
        }

        let ystart = if data.overlap_flag && row_num != 0 {
            cmp::min(2 >> sy, bh)
        } else {
            0
        };
        let xstart = if data.overlap_flag && bx != 0 {
            cmp::min(2 >> sx, bw)
        } else {
            0
        };

        let (offx, offy) = grain_offsets(offsets[0][0], is_sx, is_sy);
        let (prev_offx, _) = if data.overlap_flag && bx != 0 {
            grain_offsets(offsets[1][0], is_sx, is_sy)
        } else {
            (0, 0)
        };
        let (_, prev_offy) = if data.overlap_flag && row_num != 0 {
            grain_offsets(offsets[0][1], is_sx, is_sy)
        } else {
            (0, 0)
        };

        let noise_uv = |src_val: u8, grain: i32, luma_ptr: *const u8, luma_x: usize| -> u8 {
            unsafe {
                let mut avg = *luma_ptr.add(luma_x) as i32;
                if is_sx {
                    avg = (avg + *luma_ptr.add(luma_x + 1) as i32 + 1) >> 1;
                }
                let val = if data.chroma_scaling_from_luma {
                    avg
                } else {
                    let combined = avg * data.uv_luma_mult[uv] + src_val as i32 * data.uv_mult[uv];
                    ((combined >> 6) + data.uv_offset[uv]).clamp(0, 255)
                };
                let sc = *scaling.add(val as usize) as i32;
                let noise = round2(sc * grain, scaling_shift);
                ((src_val as i32 + noise).clamp(min_value, max_value)) as u8
            }
        };

        // Main rows (no y-overlap)
        for y in ystart..bh {
            let (src_ptr, dst_ptr, luma_ptr, grain_row) = unsafe {
                let src_ptr = src.offset(y as isize * stride).add(bx);
                let dst_ptr = dst.offset(y as isize * stride).add(bx);
                let luma_ptr = luma.offset((y << sy) as isize * luma_stride).add(bx << sx);
                let grain_row = (*grain_lut)[offy + y].as_ptr().add(offx);
                (src_ptr, dst_ptr, luma_ptr, grain_row)
            };

            // x-overlap (scalar)
            for x in 0..xstart {
                unsafe {
                    let grain = (*grain_lut)[offy + y][offx + x] as i32;
                    let old = (*grain_lut)[offy + y][prev_offx + x + (FG_BLOCK_SIZE >> sx)] as i32;
                    let blended = round2(old * W[sx][x][0] + grain * W[sx][x][1], 5);
                    let blended = blended.clamp(grain_min, grain_max);
                    *dst_ptr.add(x) = noise_uv(*src_ptr.add(x), blended, luma_ptr, x << sx);
                }
            }

            // SIMD for main body
            let mut x = xstart;
            while x + 32 <= bw {
                let src_vec = unsafe { _mm256_loadu_si256(src_ptr.add(x) as *const __m256i) };
                let src_lo = _mm256_unpacklo_epi8(src_vec, zero);
                let src_hi = _mm256_unpackhi_epi8(src_vec, zero);

                // Compute scaling values with luma dependency
                let mut sc_lo_bytes = [0u8; 32];
                let mut sc_hi_bytes = [0u8; 32];

                for i in 0..8 {
                    let val = unsafe {
                        compute_uv_scaling_val(
                            src_ptr.add(x + i),
                            luma_ptr.add((x + i) << sx),
                            is_sx,
                            data,
                            uv,
                            scaling,
                        )
                    };
                    sc_lo_bytes[i * 2] = val;
                }
                for i in 0..8 {
                    let val = unsafe {
                        compute_uv_scaling_val(
                            src_ptr.add(x + 16 + i),
                            luma_ptr.add((x + 16 + i) << sx),
                            is_sx,
                            data,
                            uv,
                            scaling,
                        )
                    };
                    sc_lo_bytes[16 + i * 2] = val;
                }
                for i in 0..8 {
                    let val = unsafe {
                        compute_uv_scaling_val(
                            src_ptr.add(x + 8 + i),
                            luma_ptr.add((x + 8 + i) << sx),
                            is_sx,
                            data,
                            uv,
                            scaling,
                        )
                    };
                    sc_hi_bytes[i * 2] = val;
                }
                for i in 0..8 {
                    let val = unsafe {
                        compute_uv_scaling_val(
                            src_ptr.add(x + 24 + i),
                            luma_ptr.add((x + 24 + i) << sx),
                            is_sx,
                            data,
                            uv,
                            scaling,
                        )
                    };
                    sc_hi_bytes[16 + i * 2] = val;
                }

                let sc_lo = unsafe { _mm256_loadu_si256(sc_lo_bytes.as_ptr() as *const __m256i) };
                let sc_hi = unsafe { _mm256_loadu_si256(sc_hi_bytes.as_ptr() as *const __m256i) };

                let grain_vec = unsafe { _mm256_loadu_si256(grain_row.add(x) as *const __m256i) };
                let grain_lo = _mm256_unpacklo_epi8(grain_vec, zero);
                let grain_hi = _mm256_unpackhi_epi8(grain_vec, zero);

                let noise_lo = _mm256_maddubs_epi16(sc_lo, grain_lo);
                let noise_hi = _mm256_maddubs_epi16(sc_hi, grain_hi);
                let noise_lo = _mm256_mulhrs_epi16(noise_lo, mul);
                let noise_hi = _mm256_mulhrs_epi16(noise_hi, mul);

                let result_lo = _mm256_add_epi16(src_lo, noise_lo);
                let result_hi = _mm256_add_epi16(src_hi, noise_hi);
                let result = _mm256_packus_epi16(result_lo, result_hi);
                let result = _mm256_max_epu8(result, min_vec);
                let result = _mm256_min_epu8(result, max_vec);
                unsafe { _mm256_storeu_si256(dst_ptr.add(x) as *mut __m256i, result) };

                x += 32;
            }

            // Scalar remainder
            while x < bw {
                unsafe {
                    let grain = *grain_row.add(x) as i32;
                    *dst_ptr.add(x) = noise_uv(*src_ptr.add(x), grain, luma_ptr, x << sx);
                }
                x += 1;
            }
        }

        // y-overlap rows (scalar)
        for y in 0..ystart {
            let (src_ptr, dst_ptr, luma_ptr) = unsafe {
                let src_ptr = src.offset(y as isize * stride).add(bx);
                let dst_ptr = dst.offset(y as isize * stride).add(bx);
                let luma_ptr = luma.offset((y << sy) as isize * luma_stride).add(bx << sx);
                (src_ptr, dst_ptr, luma_ptr)
            };

            for x in xstart..bw {
                unsafe {
                    let grain = (*grain_lut)[offy + y][offx + x] as i32;
                    let old = (*grain_lut)[prev_offy + y + (FG_BLOCK_SIZE >> sy)][offx + x] as i32;
                    let blended = round2(old * W[sy][y][0] + grain * W[sy][y][1], 5);
                    let blended = blended.clamp(grain_min, grain_max);
                    *dst_ptr.add(x) = noise_uv(*src_ptr.add(x), blended, luma_ptr, x << sx);
                }
            }

            for x in 0..xstart {
                unsafe {
                    let top = (*grain_lut)[prev_offy + y + (FG_BLOCK_SIZE >> sy)][offx + x] as i32;
                    let old_top = (*grain_lut)[prev_offy + y + (FG_BLOCK_SIZE >> sy)]
                        [prev_offx + x + (FG_BLOCK_SIZE >> sx)]
                        as i32;
                    let top = round2(old_top * W[sx][x][0] + top * W[sx][x][1], 5)
                        .clamp(grain_min, grain_max);

                    let grain = (*grain_lut)[offy + y][offx + x] as i32;
                    let old = (*grain_lut)[offy + y][prev_offx + x + (FG_BLOCK_SIZE >> sx)] as i32;
                    let grain = round2(old * W[sx][x][0] + grain * W[sx][x][1], 5)
                        .clamp(grain_min, grain_max);

                    let blended = round2(top * W[sy][y][0] + grain * W[sy][y][1], 5)
                        .clamp(grain_min, grain_max);
                    *dst_ptr.add(x) = noise_uv(*src_ptr.add(x), blended, luma_ptr, x << sx);
                }
            }
        }
    }
}

/// Helper: compute the scaling table index for chroma pixels (raw pointer version for FFI)
#[cfg(feature = "asm")]
#[inline(always)]
unsafe fn compute_uv_scaling_val(
    src_ptr: *const u8,
    luma_ptr: *const u8,
    is_sx: bool,
    data: &Rav1dFilmGrainData,
    uv: usize,
    scaling: *const u8,
) -> u8 {
    unsafe {
        let src_val = *src_ptr as i32;
        let mut avg = *luma_ptr as i32;
        if is_sx {
            avg = (avg + *luma_ptr.add(1) as i32 + 1) >> 1;
        }
        let val = if data.chroma_scaling_from_luma {
            avg
        } else {
            let combined = avg * data.uv_luma_mult[uv] + src_val * data.uv_mult[uv];
            ((combined >> 6) + data.uv_offset[uv]).clamp(0, 255)
        };
        *scaling.add(val as usize)
    }
}

// fguv FFI wrappers for each subsampling mode (8bpc)

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
pub unsafe extern "C" fn fguv_32x32xn_i420_8bpc_avx2(
    dst_row_ptr: *mut DynPixel,
    src_row_ptr: *const DynPixel,
    stride: ptrdiff_t,
    data: &Dav1dFilmGrainData,
    pw: usize,
    scaling: *const DynScaling,
    grain_lut: *const GrainLut<DynEntry>,
    bh: c_int,
    row_num: c_int,
    luma_row_ptr: *const DynPixel,
    luma_stride: ptrdiff_t,
    uv_pl: c_int,
    is_id: c_int,
    _bitdepth_max: c_int,
    _dst_row: *const FFISafe<PicOffset>,
    _src_row: *const FFISafe<PicOffset>,
    _luma_row: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    let data: Rav1dFilmGrainData = unsafe { data.clone().into() };
    fguv_inner_8bpc(
        token,
        dst_row_ptr as *mut u8,
        src_row_ptr as *const u8,
        stride as isize,
        &data,
        pw,
        scaling as *const u8,
        grain_lut as *const [[i8; GRAIN_WIDTH]; GRAIN_HEIGHT + 1],
        bh as usize,
        row_num as usize,
        luma_row_ptr as *const u8,
        luma_stride as isize,
        uv_pl != 0,
        is_id != 0,
        true, // is_sx
        true, // is_sy
    );
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
pub unsafe extern "C" fn fguv_32x32xn_i422_8bpc_avx2(
    dst_row_ptr: *mut DynPixel,
    src_row_ptr: *const DynPixel,
    stride: ptrdiff_t,
    data: &Dav1dFilmGrainData,
    pw: usize,
    scaling: *const DynScaling,
    grain_lut: *const GrainLut<DynEntry>,
    bh: c_int,
    row_num: c_int,
    luma_row_ptr: *const DynPixel,
    luma_stride: ptrdiff_t,
    uv_pl: c_int,
    is_id: c_int,
    _bitdepth_max: c_int,
    _dst_row: *const FFISafe<PicOffset>,
    _src_row: *const FFISafe<PicOffset>,
    _luma_row: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    let data: Rav1dFilmGrainData = unsafe { data.clone().into() };
    fguv_inner_8bpc(
        token,
        dst_row_ptr as *mut u8,
        src_row_ptr as *const u8,
        stride as isize,
        &data,
        pw,
        scaling as *const u8,
        grain_lut as *const [[i8; GRAIN_WIDTH]; GRAIN_HEIGHT + 1],
        bh as usize,
        row_num as usize,
        luma_row_ptr as *const u8,
        luma_stride as isize,
        uv_pl != 0,
        is_id != 0,
        true,  // is_sx
        false, // is_sy
    );
}

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
pub unsafe extern "C" fn fguv_32x32xn_i444_8bpc_avx2(
    dst_row_ptr: *mut DynPixel,
    src_row_ptr: *const DynPixel,
    stride: ptrdiff_t,
    data: &Dav1dFilmGrainData,
    pw: usize,
    scaling: *const DynScaling,
    grain_lut: *const GrainLut<DynEntry>,
    bh: c_int,
    row_num: c_int,
    luma_row_ptr: *const DynPixel,
    luma_stride: ptrdiff_t,
    uv_pl: c_int,
    is_id: c_int,
    _bitdepth_max: c_int,
    _dst_row: *const FFISafe<PicOffset>,
    _src_row: *const FFISafe<PicOffset>,
    _luma_row: *const FFISafe<PicOffset>,
) {
    let token = unsafe { Desktop64::forge_token_dangerously() };
    let data: Rav1dFilmGrainData = unsafe { data.clone().into() };
    fguv_inner_8bpc(
        token,
        dst_row_ptr as *mut u8,
        src_row_ptr as *const u8,
        stride as isize,
        &data,
        pw,
        scaling as *const u8,
        grain_lut as *const [[i8; GRAIN_WIDTH]; GRAIN_HEIGHT + 1],
        bh as usize,
        row_num as usize,
        luma_row_ptr as *const u8,
        luma_stride as isize,
        uv_pl != 0,
        is_id != 0,
        false, // is_sx
        false, // is_sy
    );
}

// ============================================================================
// fguv_32x32xn - 16bpc (scalar for now, complex luma dependency)
// ============================================================================

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[arcane]
fn fguv_inner_16bpc(
    _token: Desktop64,
    dst: *mut u16,
    src: *const u16,
    stride_u16: isize,
    data: &Rav1dFilmGrainData,
    pw: usize,
    scaling: *const u8,
    grain_lut: *const [[i16; GRAIN_WIDTH]; GRAIN_HEIGHT + 1],
    bh: usize,
    row_num: usize,
    luma: *const u16,
    luma_stride_u16: isize,
    is_uv: bool,
    is_id: bool,
    is_sx: bool,
    is_sy: bool,
    bitdepth_max: i32,
) {
    let uv = is_uv as usize;
    let sx = is_sx as usize;
    let sy = is_sy as usize;

    let bitdepth_min_8 = if bitdepth_max >= 4095 { 4u8 } else { 2u8 };
    let grain_ctr = 128i32 << bitdepth_min_8;
    let grain_min = -grain_ctr;
    let grain_max = grain_ctr - 1;

    let rows = 1 + (data.overlap_flag && row_num > 0) as usize;
    let scaling_shift = data.scaling_shift;

    let (min_value, max_value): (i32, i32) = if data.clip_to_restricted_range {
        (
            16 << bitdepth_min_8 as i32,
            (if is_id { 235 } else { 240 }) << bitdepth_min_8 as i32,
        )
    } else {
        (0, bitdepth_max as i32)
    };

    let mut seed = row_seed(rows, row_num, data);
    let mut offsets: [[c_int; 2]; 2] = [[0; 2]; 2];

    static W: [[[i32; 2]; 2]; 2] = [[[27, 17], [17, 27]], [[23, 22], [0; 2]]];

    let noise_uv = |src_val: u16, grain: i32, luma_ptr: *const u16, luma_x: usize| -> u16 {
        unsafe {
            let mut avg = *luma_ptr.add(luma_x) as i32;
            if is_sx {
                avg = (avg + *luma_ptr.add(luma_x + 1) as i32 + 1) >> 1;
            }
            let val = if data.chroma_scaling_from_luma {
                avg
            } else {
                let combined = avg * data.uv_luma_mult[uv] + src_val as i32 * data.uv_mult[uv];
                ((combined >> 6) + data.uv_offset[uv] * (1 << bitdepth_min_8))
                    .clamp(0, bitdepth_max as i32)
            };
            let sc = *scaling.add(cmp::min(val as usize, bitdepth_max as usize)) as i32;
            let noise = round2(sc * grain, scaling_shift);
            ((src_val as i32 + noise).clamp(min_value, max_value)) as u16
        }
    };

    for bx in (0..pw).step_by(FG_BLOCK_SIZE >> sx) {
        let bw = cmp::min(FG_BLOCK_SIZE >> sx, pw - bx);

        if data.overlap_flag && bx != 0 {
            for i in 0..rows {
                offsets[1][i] = offsets[0][i];
            }
        }
        for i in 0..rows {
            offsets[0][i] = get_random_number(8, &mut seed[i]);
        }

        let ystart = if data.overlap_flag && row_num != 0 {
            cmp::min(2 >> sy, bh)
        } else {
            0
        };
        let xstart = if data.overlap_flag && bx != 0 {
            cmp::min(2 >> sx, bw)
        } else {
            0
        };

        let (offx, offy) = grain_offsets(offsets[0][0], is_sx, is_sy);
        let (prev_offx, _) = if data.overlap_flag && bx != 0 {
            grain_offsets(offsets[1][0], is_sx, is_sy)
        } else {
            (0, 0)
        };
        let (_, prev_offy) = if data.overlap_flag && row_num != 0 {
            grain_offsets(offsets[0][1], is_sx, is_sy)
        } else {
            (0, 0)
        };

        for y in ystart..bh {
            let (src_ptr, dst_ptr, luma_ptr) = unsafe {
                let src_ptr = src.offset(y as isize * stride_u16).add(bx);
                let dst_ptr = dst.offset(y as isize * stride_u16).add(bx);
                let luma_ptr = luma
                    .offset((y << sy) as isize * luma_stride_u16)
                    .add(bx << sx);
                (src_ptr, dst_ptr, luma_ptr)
            };

            for x in 0..xstart {
                unsafe {
                    let grain = (*grain_lut)[offy + y][offx + x] as i32;
                    let old = (*grain_lut)[offy + y][prev_offx + x + (FG_BLOCK_SIZE >> sx)] as i32;
                    let blended = round2(old * W[sx][x][0] + grain * W[sx][x][1], 5)
                        .clamp(grain_min, grain_max);
                    *dst_ptr.add(x) = noise_uv(*src_ptr.add(x), blended, luma_ptr, x << sx);
                }
            }

            for x in xstart..bw {
                unsafe {
                    let grain = (*grain_lut)[offy + y][offx + x] as i32;
                    *dst_ptr.add(x) = noise_uv(*src_ptr.add(x), grain, luma_ptr, x << sx);
                }
            }
        }

        for y in 0..ystart {
            let (src_ptr, dst_ptr, luma_ptr) = unsafe {
                let src_ptr = src.offset(y as isize * stride_u16).add(bx);
                let dst_ptr = dst.offset(y as isize * stride_u16).add(bx);
                let luma_ptr = luma
                    .offset((y << sy) as isize * luma_stride_u16)
                    .add(bx << sx);
                (src_ptr, dst_ptr, luma_ptr)
            };

            for x in xstart..bw {
                unsafe {
                    let grain = (*grain_lut)[offy + y][offx + x] as i32;
                    let old = (*grain_lut)[prev_offy + y + (FG_BLOCK_SIZE >> sy)][offx + x] as i32;
                    let blended = round2(old * W[sy][y][0] + grain * W[sy][y][1], 5)
                        .clamp(grain_min, grain_max);
                    *dst_ptr.add(x) = noise_uv(*src_ptr.add(x), blended, luma_ptr, x << sx);
                }
            }

            for x in 0..xstart {
                unsafe {
                    let top = (*grain_lut)[prev_offy + y + (FG_BLOCK_SIZE >> sy)][offx + x] as i32;
                    let old_top = (*grain_lut)[prev_offy + y + (FG_BLOCK_SIZE >> sy)]
                        [prev_offx + x + (FG_BLOCK_SIZE >> sx)]
                        as i32;
                    let top = round2(old_top * W[sx][x][0] + top * W[sx][x][1], 5)
                        .clamp(grain_min, grain_max);

                    let grain = (*grain_lut)[offy + y][offx + x] as i32;
                    let old = (*grain_lut)[offy + y][prev_offx + x + (FG_BLOCK_SIZE >> sx)] as i32;
                    let grain = round2(old * W[sx][x][0] + grain * W[sx][x][1], 5)
                        .clamp(grain_min, grain_max);

                    let blended = round2(top * W[sy][y][0] + grain * W[sy][y][1], 5)
                        .clamp(grain_min, grain_max);
                    *dst_ptr.add(x) = noise_uv(*src_ptr.add(x), blended, luma_ptr, x << sx);
                }
            }
        }
    }
}

macro_rules! fguv_16bpc_wrapper {
    ($name:ident, $is_sx:expr, $is_sy:expr) => {
        #[cfg(all(feature = "asm", target_arch = "x86_64"))]
        pub unsafe extern "C" fn $name(
            dst_row_ptr: *mut DynPixel,
            src_row_ptr: *const DynPixel,
            stride: ptrdiff_t,
            data: &Dav1dFilmGrainData,
            pw: usize,
            scaling: *const DynScaling,
            grain_lut: *const GrainLut<DynEntry>,
            bh: c_int,
            row_num: c_int,
            luma_row_ptr: *const DynPixel,
            luma_stride: ptrdiff_t,
            uv_pl: c_int,
            is_id: c_int,
            bitdepth_max: c_int,
            _dst_row: *const FFISafe<PicOffset>,
            _src_row: *const FFISafe<PicOffset>,
            _luma_row: *const FFISafe<PicOffset>,
        ) {
            let token = unsafe { Desktop64::forge_token_dangerously() };
            let data: Rav1dFilmGrainData = unsafe { data.clone().into() };
            fguv_inner_16bpc(
                token,
                dst_row_ptr as *mut u16,
                src_row_ptr as *const u16,
                stride / 2,
                &data,
                pw,
                scaling as *const u8,
                grain_lut as *const [[i16; GRAIN_WIDTH]; GRAIN_HEIGHT + 1],
                bh as usize,
                row_num as usize,
                luma_row_ptr as *const u16,
                luma_stride / 2,
                uv_pl != 0,
                is_id != 0,
                $is_sx,
                $is_sy,
                bitdepth_max,
            );
        }
    };
}

fguv_16bpc_wrapper!(fguv_32x32xn_i420_16bpc_avx2, true, true);
fguv_16bpc_wrapper!(fguv_32x32xn_i422_16bpc_avx2, true, false);
fguv_16bpc_wrapper!(fguv_32x32xn_i444_16bpc_avx2, false, false);

// ============================================================================
// Safe inner implementations (slice-based, no raw pointer access)
// ============================================================================

/// Safe inner implementation of fgy_32x32xn for 8bpc.
/// Takes slices instead of raw pointers, uses safe_unaligned_simd for SIMD.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn fgy_inner_8bpc(
    token: Desktop64,
    dst: &mut [u8],
    src: &[u8],
    stride: isize,
    data: &Rav1dFilmGrainData,
    pw: usize,
    scaling: &[u8],
    grain_lut: &[[i8; GRAIN_WIDTH]; GRAIN_HEIGHT + 1],
    bh: usize,
    row_num: usize,
) {
    let rows = 1 + (data.overlap_flag && row_num > 0) as usize;
    let scaling_shift = data.scaling_shift;

    let (min_value, max_value): (i32, i32) = if data.clip_to_restricted_range {
        (16, 235)
    } else {
        (0, 255)
    };

    let mut seed = row_seed(rows, row_num, data);

    let mul = _mm256_set1_epi16(1i16 << (15 - scaling_shift));
    let min_vec = _mm256_set1_epi8(min_value as i8);
    let max_vec = _mm256_set1_epi8(max_value as u8 as i8);

    let mut offsets: [[c_int; 2]; 2] = [[0; 2]; 2];
    static W: [[i32; 2]; 2] = [[27, 17], [17, 27]];

    // Helper to compute row offset in the flat buffer
    let row_off = |y: usize| -> usize {
        // stride can be negative, so use wrapping arithmetic
        (y as isize * stride) as usize
    };

    for bx in (0..pw).step_by(FG_BLOCK_SIZE) {
        let bw = cmp::min(FG_BLOCK_SIZE, pw - bx);

        if data.overlap_flag && bx != 0 {
            for i in 0..rows {
                offsets[1][i] = offsets[0][i];
            }
        }
        for i in 0..rows {
            offsets[0][i] = get_random_number(8, &mut seed[i]);
        }

        let ystart = if data.overlap_flag && row_num != 0 {
            cmp::min(2, bh)
        } else {
            0
        };
        let xstart = if data.overlap_flag && bx != 0 {
            cmp::min(2, bw)
        } else {
            0
        };

        let (offx, offy) = grain_offsets(offsets[0][0], false, false);
        let (prev_offx, _) = if data.overlap_flag && bx != 0 {
            grain_offsets(offsets[1][0], false, false)
        } else {
            (0, 0)
        };
        let (_, prev_offy) = if data.overlap_flag && row_num != 0 {
            grain_offsets(offsets[0][1], false, false)
        } else {
            (0, 0)
        };

        // Main body: no overlap
        for y in ystart..bh {
            let base = row_off(y).wrapping_add(bx);

            // Handle x-overlap region with scalar blending
            for x in 0..xstart {
                let sv = src[base + x] as usize;
                let grain = grain_lut[offy + y][offx + x] as i32;
                let old = grain_lut[offy + y][prev_offx + x + FG_BLOCK_SIZE] as i32;
                let blended = round2(old * W[x][0] + grain * W[x][1], 5);
                let blended = blended.clamp(-128, 127);
                let sc = scaling[sv] as i32;
                let noise = round2(sc * blended, scaling_shift);
                dst[base + x] =
                    ((src[base + x] as i32 + noise).clamp(min_value, max_value)) as u8;
            }

            // SIMD for the rest
            fgy_row_simd_8bpc_safe(
                token,
                &mut dst[base..base + bw],
                &src[base..base + bw],
                scaling,
                &grain_lut[offy + y][offx..offx + bw],
                bw,
                xstart,
                mul,
                min_vec,
                max_vec,
                scaling_shift,
                min_value,
                max_value,
            );
        }

        // y-overlap rows
        for y in 0..ystart {
            let base = row_off(y).wrapping_add(bx);

            for x in xstart..bw {
                let sv = src[base + x] as usize;
                let grain = grain_lut[offy + y][offx + x] as i32;
                let old = grain_lut[prev_offy + y + FG_BLOCK_SIZE][offx + x] as i32;
                let blended = round2(old * W[y][0] + grain * W[y][1], 5);
                let blended = blended.clamp(-128, 127);
                let sc = scaling[sv] as i32;
                let noise = round2(sc * blended, scaling_shift);
                dst[base + x] =
                    ((src[base + x] as i32 + noise).clamp(min_value, max_value)) as u8;
            }

            // Corner overlap (both x and y)
            for x in 0..xstart {
                let sv = src[base + x] as usize;

                let top = grain_lut[prev_offy + y + FG_BLOCK_SIZE][offx + x] as i32;
                let old_top = grain_lut[prev_offy + y + FG_BLOCK_SIZE]
                    [prev_offx + x + FG_BLOCK_SIZE] as i32;
                let top = round2(old_top * W[x][0] + top * W[x][1], 5).clamp(-128, 127);

                let grain = grain_lut[offy + y][offx + x] as i32;
                let old = grain_lut[offy + y][prev_offx + x + FG_BLOCK_SIZE] as i32;
                let grain = round2(old * W[x][0] + grain * W[x][1], 5).clamp(-128, 127);

                let blended = round2(top * W[y][0] + grain * W[y][1], 5).clamp(-128, 127);
                let sc = scaling[sv] as i32;
                let noise = round2(sc * blended, scaling_shift);
                dst[base + x] =
                    ((src[base + x] as i32 + noise).clamp(min_value, max_value)) as u8;
            }
        }
    }
}

/// Safe inner implementation of fgy_32x32xn for 16bpc.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn fgy_inner_16bpc(
    _token: Desktop64,
    dst: &mut [u16],
    src: &[u16],
    stride_u16: isize,
    data: &Rav1dFilmGrainData,
    pw: usize,
    scaling: &[u8],
    grain_lut: &[[i16; GRAIN_WIDTH]; GRAIN_HEIGHT + 1],
    bh: usize,
    row_num: usize,
    bitdepth_max: i32,
) {
    use safe_unaligned_simd::x86_64 as safe_simd;

    let bitdepth_min_8 = if bitdepth_max >= 4095 { 4u8 } else { 2u8 };
    let grain_ctr = 128i32 << bitdepth_min_8;
    let grain_min = -grain_ctr;
    let grain_max = grain_ctr - 1;

    let rows = 1 + (data.overlap_flag && row_num > 0) as usize;
    let scaling_shift = data.scaling_shift;

    let (min_value, max_value): (i32, i32) = if data.clip_to_restricted_range {
        (16 << bitdepth_min_8 as i32, 235 << bitdepth_min_8 as i32)
    } else {
        (0, bitdepth_max as i32)
    };

    let mut seed = row_seed(rows, row_num, data);
    let mut offsets: [[c_int; 2]; 2] = [[0; 2]; 2];
    static W: [[i32; 2]; 2] = [[27, 17], [17, 27]];

    let min_vec = _mm256_set1_epi16(min_value as i16);
    let max_vec = _mm256_set1_epi16(max_value as i16);

    let row_off = |y: usize| -> usize {
        (y as isize * stride_u16) as usize
    };

    for bx in (0..pw).step_by(FG_BLOCK_SIZE) {
        let bw = cmp::min(FG_BLOCK_SIZE, pw - bx);

        if data.overlap_flag && bx != 0 {
            for i in 0..rows {
                offsets[1][i] = offsets[0][i];
            }
        }
        for i in 0..rows {
            offsets[0][i] = get_random_number(8, &mut seed[i]);
        }

        let ystart = if data.overlap_flag && row_num != 0 {
            cmp::min(2, bh)
        } else {
            0
        };
        let xstart = if data.overlap_flag && bx != 0 {
            cmp::min(2, bw)
        } else {
            0
        };

        let (offx, offy) = grain_offsets(offsets[0][0], false, false);
        let (prev_offx, _) = if data.overlap_flag && bx != 0 {
            grain_offsets(offsets[1][0], false, false)
        } else {
            (0, 0)
        };
        let (_, prev_offy) = if data.overlap_flag && row_num != 0 {
            grain_offsets(offsets[0][1], false, false)
        } else {
            (0, 0)
        };

        for y in ystart..bh {
            let base = row_off(y).wrapping_add(bx);

            // x-overlap scalar
            for x in 0..xstart {
                let sv = src[base + x] as usize;
                let grain = grain_lut[offy + y][offx + x] as i32;
                let old = grain_lut[offy + y][prev_offx + x + FG_BLOCK_SIZE] as i32;
                let blended = round2(old * W[x][0] + grain * W[x][1], 5).clamp(grain_min, grain_max);
                let sc = scaling[cmp::min(sv, bitdepth_max as usize)] as i32;
                let noise = round2(sc * blended, scaling_shift);
                dst[base + x] =
                    ((src[base + x] as i32 + noise).clamp(min_value, max_value)) as u16;
            }

            // SIMD 16 pixels at a time
            let mut x = xstart;
            while x + 16 <= bw {
                let src_chunk: &[u16; 16] = src[base + x..base + x + 16].try_into().unwrap();
                let src_vec = safe_simd::_mm256_loadu_si256(src_chunk);

                let mut noise_vals = [0i16; 16];
                for i in 0..16 {
                    let sv = cmp::min(src[base + x + i] as usize, bitdepth_max as usize);
                    let grain = grain_lut[offy + y][offx + x + i] as i32;
                    let sc = scaling[sv] as i32;
                    noise_vals[i] = round2(sc * grain, scaling_shift) as i16;
                }

                let noise = safe_simd::_mm256_loadu_si256(&noise_vals);
                let result = _mm256_add_epi16(src_vec, noise);
                let result = _mm256_max_epi16(result, min_vec);
                let result = _mm256_min_epi16(result, max_vec);
                let dst_chunk: &mut [u16; 16] = (&mut dst[base + x..base + x + 16]).try_into().unwrap();
                safe_simd::_mm256_storeu_si256(dst_chunk, result);
                x += 16;
            }

            // Scalar remainder
            while x < bw {
                let sv = cmp::min(src[base + x] as usize, bitdepth_max as usize);
                let grain = grain_lut[offy + y][offx + x] as i32;
                let sc = scaling[sv] as i32;
                let noise = round2(sc * grain, scaling_shift);
                dst[base + x] =
                    ((src[base + x] as i32 + noise).clamp(min_value, max_value)) as u16;
                x += 1;
            }
        }

        // y-overlap rows
        for y in 0..ystart {
            let base = row_off(y).wrapping_add(bx);

            for x in xstart..bw {
                let sv = cmp::min(src[base + x] as usize, bitdepth_max as usize);
                let grain = grain_lut[offy + y][offx + x] as i32;
                let old = grain_lut[prev_offy + y + FG_BLOCK_SIZE][offx + x] as i32;
                let blended = round2(old * W[y][0] + grain * W[y][1], 5).clamp(grain_min, grain_max);
                let sc = scaling[sv] as i32;
                let noise = round2(sc * blended, scaling_shift);
                dst[base + x] =
                    ((src[base + x] as i32 + noise).clamp(min_value, max_value)) as u16;
            }

            for x in 0..xstart {
                let sv = cmp::min(src[base + x] as usize, bitdepth_max as usize);
                let top = grain_lut[prev_offy + y + FG_BLOCK_SIZE][offx + x] as i32;
                let old_top = grain_lut[prev_offy + y + FG_BLOCK_SIZE]
                    [prev_offx + x + FG_BLOCK_SIZE] as i32;
                let top = round2(old_top * W[x][0] + top * W[x][1], 5).clamp(grain_min, grain_max);

                let grain = grain_lut[offy + y][offx + x] as i32;
                let old = grain_lut[offy + y][prev_offx + x + FG_BLOCK_SIZE] as i32;
                let grain = round2(old * W[x][0] + grain * W[x][1], 5).clamp(grain_min, grain_max);

                let blended = round2(top * W[y][0] + grain * W[y][1], 5).clamp(grain_min, grain_max);
                let sc = scaling[sv] as i32;
                let noise = round2(sc * blended, scaling_shift);
                dst[base + x] =
                    ((src[base + x] as i32 + noise).clamp(min_value, max_value)) as u16;
            }
        }
    }
}

/// Safe helper: compute scaling index for chroma pixels with luma dependency.
#[inline(always)]
fn compute_uv_scaling_val_safe(
    src_val: u8,
    luma: &[u8],
    luma_x: usize,
    is_sx: bool,
    data: &Rav1dFilmGrainData,
    uv: usize,
    scaling: &[u8],
) -> u8 {
    let mut avg = luma[luma_x] as i32;
    if is_sx {
        avg = (avg + luma[luma_x + 1] as i32 + 1) >> 1;
    }
    let val = if data.chroma_scaling_from_luma {
        avg
    } else {
        let combined = avg * data.uv_luma_mult[uv] + src_val as i32 * data.uv_mult[uv];
        ((combined >> 6) + data.uv_offset[uv]).clamp(0, 255)
    };
    scaling[val as usize]
}

/// Safe inner implementation of fguv_32x32xn for 8bpc.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn fguv_inner_safe_8bpc(
    _token: Desktop64,
    dst: &mut [u8],
    src: &[u8],
    stride: isize,
    data: &Rav1dFilmGrainData,
    pw: usize,
    scaling: &[u8],
    grain_lut: &[[i8; GRAIN_WIDTH]; GRAIN_HEIGHT + 1],
    bh: usize,
    row_num: usize,
    luma: &[u8],
    luma_stride: isize,
    is_uv: bool,
    is_id: bool,
    is_sx: bool,
    is_sy: bool,
) {
    use safe_unaligned_simd::x86_64 as safe_simd;

    let uv = is_uv as usize;
    let sx = is_sx as usize;
    let sy = is_sy as usize;

    let rows = 1 + (data.overlap_flag && row_num > 0) as usize;
    let scaling_shift = data.scaling_shift;

    let (min_value, max_value): (i32, i32) = if data.clip_to_restricted_range {
        (16, if is_id { 235 } else { 240 })
    } else {
        (0, 255)
    };

    let grain_min = -128i32;
    let grain_max = 127i32;

    let mut seed = row_seed(rows, row_num, data);
    let mut offsets: [[c_int; 2]; 2] = [[0; 2]; 2];
    static W: [[[i32; 2]; 2]; 2] = [[[27, 17], [17, 27]], [[23, 22], [0; 2]]];

    let mul = _mm256_set1_epi16(1i16 << (15 - scaling_shift));
    let min_vec = _mm256_set1_epi8(min_value as i8);
    let max_vec = _mm256_set1_epi8(max_value as u8 as i8);
    let zero = _mm256_setzero_si256();

    let row_off = |y: usize| -> usize { (y as isize * stride) as usize };
    let luma_row_off = |y: usize| -> usize { ((y << sy) as isize * luma_stride) as usize };

    let noise_uv = |src_val: u8, grain: i32, luma_row: &[u8], luma_x: usize| -> u8 {
        let mut avg = luma_row[luma_x] as i32;
        if is_sx {
            avg = (avg + luma_row[luma_x + 1] as i32 + 1) >> 1;
        }
        let val = if data.chroma_scaling_from_luma {
            avg
        } else {
            let combined = avg * data.uv_luma_mult[uv] + src_val as i32 * data.uv_mult[uv];
            ((combined >> 6) + data.uv_offset[uv]).clamp(0, 255)
        };
        let sc = scaling[val as usize] as i32;
        let noise = round2(sc * grain, scaling_shift);
        ((src_val as i32 + noise).clamp(min_value, max_value)) as u8
    };

    for bx in (0..pw).step_by(FG_BLOCK_SIZE >> sx) {
        let bw = cmp::min(FG_BLOCK_SIZE >> sx, pw - bx);

        if data.overlap_flag && bx != 0 {
            for i in 0..rows {
                offsets[1][i] = offsets[0][i];
            }
        }
        for i in 0..rows {
            offsets[0][i] = get_random_number(8, &mut seed[i]);
        }

        let ystart = if data.overlap_flag && row_num != 0 {
            cmp::min(2 >> sy, bh)
        } else {
            0
        };
        let xstart = if data.overlap_flag && bx != 0 {
            cmp::min(2 >> sx, bw)
        } else {
            0
        };

        let (offx, offy) = grain_offsets(offsets[0][0], is_sx, is_sy);
        let (prev_offx, _) = if data.overlap_flag && bx != 0 {
            grain_offsets(offsets[1][0], is_sx, is_sy)
        } else {
            (0, 0)
        };
        let (_, prev_offy) = if data.overlap_flag && row_num != 0 {
            grain_offsets(offsets[0][1], is_sx, is_sy)
        } else {
            (0, 0)
        };

        // Main rows (no y-overlap)
        for y in ystart..bh {
            let base = row_off(y).wrapping_add(bx);
            let luma_base = luma_row_off(y).wrapping_add(bx << sx);

            // x-overlap (scalar)
            for x in 0..xstart {
                let grain = grain_lut[offy + y][offx + x] as i32;
                let old = grain_lut[offy + y][prev_offx + x + (FG_BLOCK_SIZE >> sx)] as i32;
                let blended = round2(old * W[sx][x][0] + grain * W[sx][x][1], 5)
                    .clamp(grain_min, grain_max);
                dst[base + x] = noise_uv(src[base + x], blended, &luma[luma_base..], x << sx);
            }

            // SIMD for main body
            let mut x = xstart;
            while x + 32 <= bw {
                let src_arr: &[u8; 32] = src[base + x..base + x + 32].try_into().unwrap();
                let src_vec = safe_simd::_mm256_loadu_si256(src_arr);
                let src_lo = _mm256_unpacklo_epi8(src_vec, zero);
                let src_hi = _mm256_unpackhi_epi8(src_vec, zero);

                let mut sc_lo_bytes = [0u8; 32];
                let mut sc_hi_bytes = [0u8; 32];

                for i in 0..8 {
                    sc_lo_bytes[i * 2] = compute_uv_scaling_val_safe(
                        src[base + x + i], luma, luma_base + ((x + i) << sx),
                        is_sx, data, uv, scaling,
                    );
                }
                for i in 0..8 {
                    sc_lo_bytes[16 + i * 2] = compute_uv_scaling_val_safe(
                        src[base + x + 16 + i], luma, luma_base + ((x + 16 + i) << sx),
                        is_sx, data, uv, scaling,
                    );
                }
                for i in 0..8 {
                    sc_hi_bytes[i * 2] = compute_uv_scaling_val_safe(
                        src[base + x + 8 + i], luma, luma_base + ((x + 8 + i) << sx),
                        is_sx, data, uv, scaling,
                    );
                }
                for i in 0..8 {
                    sc_hi_bytes[16 + i * 2] = compute_uv_scaling_val_safe(
                        src[base + x + 24 + i], luma, luma_base + ((x + 24 + i) << sx),
                        is_sx, data, uv, scaling,
                    );
                }

                let sc_lo = safe_simd::_mm256_loadu_si256(&sc_lo_bytes);
                let sc_hi = safe_simd::_mm256_loadu_si256(&sc_hi_bytes);

                let mut grain_bytes = [0u8; 32];
                for i in 0..32 {
                    grain_bytes[i] = grain_lut[offy + y][offx + x + i] as u8;
                }
                let grain_vec = safe_simd::_mm256_loadu_si256(&grain_bytes);
                let grain_lo = _mm256_unpacklo_epi8(grain_vec, zero);
                let grain_hi = _mm256_unpackhi_epi8(grain_vec, zero);

                let noise_lo = _mm256_maddubs_epi16(sc_lo, grain_lo);
                let noise_hi = _mm256_maddubs_epi16(sc_hi, grain_hi);
                let noise_lo = _mm256_mulhrs_epi16(noise_lo, mul);
                let noise_hi = _mm256_mulhrs_epi16(noise_hi, mul);

                let result_lo = _mm256_add_epi16(src_lo, noise_lo);
                let result_hi = _mm256_add_epi16(src_hi, noise_hi);
                let result = _mm256_packus_epi16(result_lo, result_hi);
                let result = _mm256_max_epu8(result, min_vec);
                let result = _mm256_min_epu8(result, max_vec);

                let dst_arr: &mut [u8; 32] = (&mut dst[base + x..base + x + 32]).try_into().unwrap();
                safe_simd::_mm256_storeu_si256(dst_arr, result);
                x += 32;
            }

            // Scalar remainder
            while x < bw {
                let grain = grain_lut[offy + y][offx + x] as i32;
                dst[base + x] = noise_uv(src[base + x], grain, &luma[luma_base..], x << sx);
                x += 1;
            }
        }

        // y-overlap rows (scalar)
        for y in 0..ystart {
            let base = row_off(y).wrapping_add(bx);
            let luma_base = luma_row_off(y).wrapping_add(bx << sx);

            for x in xstart..bw {
                let grain = grain_lut[offy + y][offx + x] as i32;
                let old = grain_lut[prev_offy + y + (FG_BLOCK_SIZE >> sy)][offx + x] as i32;
                let blended = round2(old * W[sy][y][0] + grain * W[sy][y][1], 5)
                    .clamp(grain_min, grain_max);
                dst[base + x] = noise_uv(src[base + x], blended, &luma[luma_base..], x << sx);
            }

            for x in 0..xstart {
                let top = grain_lut[prev_offy + y + (FG_BLOCK_SIZE >> sy)][offx + x] as i32;
                let old_top = grain_lut[prev_offy + y + (FG_BLOCK_SIZE >> sy)]
                    [prev_offx + x + (FG_BLOCK_SIZE >> sx)] as i32;
                let top = round2(old_top * W[sx][x][0] + top * W[sx][x][1], 5)
                    .clamp(grain_min, grain_max);

                let grain = grain_lut[offy + y][offx + x] as i32;
                let old = grain_lut[offy + y][prev_offx + x + (FG_BLOCK_SIZE >> sx)] as i32;
                let grain = round2(old * W[sx][x][0] + grain * W[sx][x][1], 5)
                    .clamp(grain_min, grain_max);

                let blended = round2(top * W[sy][y][0] + grain * W[sy][y][1], 5)
                    .clamp(grain_min, grain_max);
                dst[base + x] = noise_uv(src[base + x], blended, &luma[luma_base..], x << sx);
            }
        }
    }
}

/// Safe inner implementation of fguv_32x32xn for 16bpc.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn fguv_inner_safe_16bpc(
    _token: Desktop64,
    dst: &mut [u16],
    src: &[u16],
    stride_u16: isize,
    data: &Rav1dFilmGrainData,
    pw: usize,
    scaling: &[u8],
    grain_lut: &[[i16; GRAIN_WIDTH]; GRAIN_HEIGHT + 1],
    bh: usize,
    row_num: usize,
    luma: &[u16],
    luma_stride_u16: isize,
    is_uv: bool,
    is_id: bool,
    is_sx: bool,
    is_sy: bool,
    bitdepth_max: i32,
) {
    let uv = is_uv as usize;
    let sx = is_sx as usize;
    let sy = is_sy as usize;

    let bitdepth_min_8 = if bitdepth_max >= 4095 { 4u8 } else { 2u8 };
    let grain_ctr = 128i32 << bitdepth_min_8;
    let grain_min = -grain_ctr;
    let grain_max = grain_ctr - 1;

    let rows = 1 + (data.overlap_flag && row_num > 0) as usize;
    let scaling_shift = data.scaling_shift;

    let (min_value, max_value): (i32, i32) = if data.clip_to_restricted_range {
        (
            16 << bitdepth_min_8 as i32,
            (if is_id { 235 } else { 240 }) << bitdepth_min_8 as i32,
        )
    } else {
        (0, bitdepth_max as i32)
    };

    let mut seed = row_seed(rows, row_num, data);
    let mut offsets: [[c_int; 2]; 2] = [[0; 2]; 2];
    static W: [[[i32; 2]; 2]; 2] = [[[27, 17], [17, 27]], [[23, 22], [0; 2]]];

    let row_off = |y: usize| -> usize { (y as isize * stride_u16) as usize };
    let luma_row_off = |y: usize| -> usize { ((y << sy) as isize * luma_stride_u16) as usize };

    let noise_uv = |src_val: u16, grain: i32, luma_row: &[u16], luma_x: usize| -> u16 {
        let mut avg = luma_row[luma_x] as i32;
        if is_sx {
            avg = (avg + luma_row[luma_x + 1] as i32 + 1) >> 1;
        }
        let val = if data.chroma_scaling_from_luma {
            avg
        } else {
            let combined = avg * data.uv_luma_mult[uv] + src_val as i32 * data.uv_mult[uv];
            ((combined >> 6) + data.uv_offset[uv] * (1 << bitdepth_min_8))
                .clamp(0, bitdepth_max as i32)
        };
        let sc = scaling[cmp::min(val as usize, bitdepth_max as usize)] as i32;
        let noise = round2(sc * grain, scaling_shift);
        ((src_val as i32 + noise).clamp(min_value, max_value)) as u16
    };

    for bx in (0..pw).step_by(FG_BLOCK_SIZE >> sx) {
        let bw = cmp::min(FG_BLOCK_SIZE >> sx, pw - bx);

        if data.overlap_flag && bx != 0 {
            for i in 0..rows {
                offsets[1][i] = offsets[0][i];
            }
        }
        for i in 0..rows {
            offsets[0][i] = get_random_number(8, &mut seed[i]);
        }

        let ystart = if data.overlap_flag && row_num != 0 {
            cmp::min(2 >> sy, bh)
        } else {
            0
        };
        let xstart = if data.overlap_flag && bx != 0 {
            cmp::min(2 >> sx, bw)
        } else {
            0
        };

        let (offx, offy) = grain_offsets(offsets[0][0], is_sx, is_sy);
        let (prev_offx, _) = if data.overlap_flag && bx != 0 {
            grain_offsets(offsets[1][0], is_sx, is_sy)
        } else {
            (0, 0)
        };
        let (_, prev_offy) = if data.overlap_flag && row_num != 0 {
            grain_offsets(offsets[0][1], is_sx, is_sy)
        } else {
            (0, 0)
        };

        for y in ystart..bh {
            let base = row_off(y).wrapping_add(bx);
            let luma_base = luma_row_off(y).wrapping_add(bx << sx);

            // All scalar for 16bpc (no SIMD loop for complex luma dependency)
            for x in 0..xstart {
                let grain = grain_lut[offy + y][offx + x] as i32;
                let old = grain_lut[offy + y][prev_offx + x + (FG_BLOCK_SIZE >> sx)] as i32;
                let blended = round2(old * W[sx][x][0] + grain * W[sx][x][1], 5)
                    .clamp(grain_min, grain_max);
                dst[base + x] = noise_uv(src[base + x], blended, &luma[luma_base..], x << sx);
            }

            for x in xstart..bw {
                let grain = grain_lut[offy + y][offx + x] as i32;
                dst[base + x] = noise_uv(src[base + x], grain, &luma[luma_base..], x << sx);
            }
        }

        for y in 0..ystart {
            let base = row_off(y).wrapping_add(bx);
            let luma_base = luma_row_off(y).wrapping_add(bx << sx);

            for x in xstart..bw {
                let grain = grain_lut[offy + y][offx + x] as i32;
                let old = grain_lut[prev_offy + y + (FG_BLOCK_SIZE >> sy)][offx + x] as i32;
                let blended = round2(old * W[sy][y][0] + grain * W[sy][y][1], 5)
                    .clamp(grain_min, grain_max);
                dst[base + x] = noise_uv(src[base + x], blended, &luma[luma_base..], x << sx);
            }

            for x in 0..xstart {
                let top = grain_lut[prev_offy + y + (FG_BLOCK_SIZE >> sy)][offx + x] as i32;
                let old_top = grain_lut[prev_offy + y + (FG_BLOCK_SIZE >> sy)]
                    [prev_offx + x + (FG_BLOCK_SIZE >> sx)] as i32;
                let top = round2(old_top * W[sx][x][0] + top * W[sx][x][1], 5)
                    .clamp(grain_min, grain_max);

                let grain = grain_lut[offy + y][offx + x] as i32;
                let old = grain_lut[offy + y][prev_offx + x + (FG_BLOCK_SIZE >> sx)] as i32;
                let grain = round2(old * W[sx][x][0] + grain * W[sx][x][1], 5)
                    .clamp(grain_min, grain_max);

                let blended = round2(top * W[sy][y][0] + grain * W[sy][y][1], 5)
                    .clamp(grain_min, grain_max);
                dst[base + x] = noise_uv(src[base + x], blended, &luma[luma_base..], x << sx);
            }
        }
    }
}

// ============================================================================
// Safe dispatch wrappers — encapsulate unsafe pointer creation and FFI calls
// ============================================================================

use crate::include::common::bitdepth::{BitDepth, BPC};
use crate::include::dav1d::headers::Rav1dPixelLayoutSubSampled;
use crate::include::dav1d::picture::Rav1dPictureDataComponent;
use crate::src::cpu::CpuFlags;
use crate::src::strided::Strided as _;

/// Safe dispatch for generate_grain_y (x86_64 AVX2).
/// Returns true if SIMD was used.
#[cfg(target_arch = "x86_64")]
pub fn generate_grain_y_dispatch<BD: BitDepth>(
    buf: &mut GrainLut<BD::Entry>,
    data: &Rav1dFilmGrainData,
    bd: BD,
) -> bool {
    if !crate::src::cpu::rav1d_get_cpu_flags().contains(CpuFlags::AVX2) {
        return false;
    }
    // Call inner functions directly, bypassing FFI wrappers.
    unsafe {
        match BD::BPC {
            BPC::BPC8 => {
                let buf = &mut *(buf as *mut GrainLut<BD::Entry> as *mut GrainLut<i8>);
                generate_grain_y_inner_8bpc(buf, data);
            }
            BPC::BPC16 => {
                let buf = &mut *(buf as *mut GrainLut<BD::Entry> as *mut GrainLut<i16>);
                let bitdepth = if bd.into_c() >= 4095 { 12 } else { 10 };
                generate_grain_y_inner_16bpc(buf, data, bitdepth);
            }
        }
    }
    true
}

/// Safe dispatch for generate_grain_uv (x86_64 AVX2).
/// Returns true if SIMD was used.
#[cfg(target_arch = "x86_64")]
pub fn generate_grain_uv_dispatch<BD: BitDepth>(
    layout: Rav1dPixelLayoutSubSampled,
    buf: &mut GrainLut<BD::Entry>,
    buf_y: &GrainLut<BD::Entry>,
    data: &Rav1dFilmGrainData,
    is_uv: bool,
    bd: BD,
) -> bool {
    if !crate::src::cpu::rav1d_get_cpu_flags().contains(CpuFlags::AVX2) {
        return false;
    }
    // Call inner functions directly, bypassing FFI wrappers.
    let (is_subx, is_suby) = match layout {
        Rav1dPixelLayoutSubSampled::I420 => (true, true),
        Rav1dPixelLayoutSubSampled::I422 => (true, false),
        Rav1dPixelLayoutSubSampled::I444 => (false, false),
    };
    unsafe {
        match BD::BPC {
            BPC::BPC8 => {
                let buf = &mut *(buf as *mut GrainLut<BD::Entry> as *mut GrainLut<i8>);
                let buf_y = &*(buf_y as *const GrainLut<BD::Entry> as *const GrainLut<i8>);
                generate_grain_uv_inner_8bpc(buf, buf_y, data, is_uv, is_subx, is_suby);
            }
            BPC::BPC16 => {
                let buf = &mut *(buf as *mut GrainLut<BD::Entry> as *mut GrainLut<i16>);
                let buf_y = &*(buf_y as *const GrainLut<BD::Entry> as *const GrainLut<i16>);
                let bitdepth = if bd.into_c() >= 4095 { 12 } else { 10 };
                generate_grain_uv_inner_16bpc(buf, buf_y, data, is_uv, is_subx, is_suby, bitdepth);
            }
        }
    }
    true
}

/// Safe dispatch for fgy_32x32xn (x86_64 AVX2).
/// Returns true if SIMD was used.
#[cfg(target_arch = "x86_64")]
pub fn fgy_32x32xn_dispatch<BD: BitDepth>(
    dst: &Rav1dPictureDataComponent,
    src: &Rav1dPictureDataComponent,
    data: &Rav1dFilmGrainData,
    pw: usize,
    scaling: &BD::Scaling,
    grain_lut: &GrainLut<BD::Entry>,
    bh: usize,
    row_num: usize,
    bd: BD,
) -> bool {
    let token = match Desktop64::summon() {
        Some(t) => t,
        None => return false,
    };
    if !crate::src::cpu::rav1d_get_cpu_flags().contains(CpuFlags::AVX2) {
        return false;
    }
    let row_strides = (row_num * FG_BLOCK_SIZE) as isize;
    let dst_row = dst.with_offset::<BD>() + row_strides * dst.pixel_stride::<BD>();
    let src_row = src.with_offset::<BD>() + row_strides * src.pixel_stride::<BD>();
    let pixel_stride = dst.pixel_stride::<BD>();
    let scaling_bytes: &[u8] = scaling.as_ref();

    // Total pixels needed: (bh-1) rows * stride + pw pixels in last row
    let total_pixels = if bh > 0 {
        (bh - 1) * pixel_stride.unsigned_abs() + pw
    } else {
        0
    };

    // Convert grain_lut bytes for safe reinterpretation.
    // SAFETY: BD::Entry is i8 (8bpc) or i16 (16bpc), both POD types with same layout as u8/u16.
    // This is the one remaining unsafe: creating a byte view of the generic grain_lut.
    let grain_lut_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            grain_lut as *const _ as *const u8,
            std::mem::size_of_val(grain_lut),
        )
    };

    match BD::BPC {
        BPC::BPC8 => {
            let src_guard = src_row.slice::<BD>(total_pixels);
            let src_slice: &[u8] =
                crate::src::safe_simd::pixel_access::reinterpret_slice(&*src_guard)
                    .expect("8bpc pixel reinterpret");
            let mut dst_guard = dst_row.slice_mut::<BD>(total_pixels);
            let dst_slice: &mut [u8] =
                crate::src::safe_simd::pixel_access::reinterpret_slice_mut(&mut *dst_guard)
                    .expect("8bpc pixel reinterpret");
            let grain_lut_8: &[[i8; GRAIN_WIDTH]; GRAIN_HEIGHT + 1] =
                zerocopy::Ref::<_, [[i8; GRAIN_WIDTH]; GRAIN_HEIGHT + 1]>::new(grain_lut_bytes)
                    .expect("grain_lut reinterpret to i8")
                    .into_ref();
            fgy_inner_8bpc(
                token,
                dst_slice,
                src_slice,
                pixel_stride as isize,
                data,
                pw,
                scaling_bytes,
                grain_lut_8,
                bh,
                row_num,
            );
        }
        BPC::BPC16 => {
            let src_guard = src_row.slice::<BD>(total_pixels);
            let src_slice: &[u16] =
                crate::src::safe_simd::pixel_access::reinterpret_slice(&*src_guard)
                    .expect("16bpc pixel reinterpret");
            let mut dst_guard = dst_row.slice_mut::<BD>(total_pixels);
            let dst_slice: &mut [u16] =
                crate::src::safe_simd::pixel_access::reinterpret_slice_mut(&mut *dst_guard)
                    .expect("16bpc pixel reinterpret");
            let grain_lut_16: &[[i16; GRAIN_WIDTH]; GRAIN_HEIGHT + 1] =
                zerocopy::Ref::<_, [[i16; GRAIN_WIDTH]; GRAIN_HEIGHT + 1]>::new(grain_lut_bytes)
                    .expect("grain_lut reinterpret to i16")
                    .into_ref();
            fgy_inner_16bpc(
                token,
                dst_slice,
                src_slice,
                pixel_stride as isize,
                data,
                pw,
                scaling_bytes,
                grain_lut_16,
                bh,
                row_num,
                bd.into_c(),
            );
        }
    }
    true
}

/// Safe dispatch for fguv_32x32xn (x86_64 AVX2).
/// Returns true if SIMD was used.
#[cfg(target_arch = "x86_64")]
pub fn fguv_32x32xn_dispatch<BD: BitDepth>(
    layout: Rav1dPixelLayoutSubSampled,
    dst: &Rav1dPictureDataComponent,
    src: &Rav1dPictureDataComponent,
    data: &Rav1dFilmGrainData,
    pw: usize,
    scaling: &BD::Scaling,
    grain_lut: &GrainLut<BD::Entry>,
    bh: usize,
    row_num: usize,
    luma: &Rav1dPictureDataComponent,
    is_uv: bool,
    is_id: bool,
    bd: BD,
) -> bool {
    let token = match Desktop64::summon() {
        Some(t) => t,
        None => return false,
    };
    if !crate::src::cpu::rav1d_get_cpu_flags().contains(CpuFlags::AVX2) {
        return false;
    }
    let ss_y = (layout == Rav1dPixelLayoutSubSampled::I420) as usize;
    let (is_sx, is_sy) = match layout {
        Rav1dPixelLayoutSubSampled::I420 => (true, true),
        Rav1dPixelLayoutSubSampled::I422 => (true, false),
        Rav1dPixelLayoutSubSampled::I444 => (false, false),
    };

    let row_strides = (row_num * FG_BLOCK_SIZE) as isize;
    let dst_row = dst.with_offset::<BD>() + (row_strides * dst.pixel_stride::<BD>() >> ss_y);
    let src_row = src.with_offset::<BD>() + (row_strides * src.pixel_stride::<BD>() >> ss_y);
    let luma_row = luma.with_offset::<BD>() + (row_strides * luma.pixel_stride::<BD>());

    let scaling_bytes: &[u8] = scaling.as_ref();
    let pixel_stride = dst.pixel_stride::<BD>();
    let luma_pixel_stride = luma.pixel_stride::<BD>();

    // Total pixels for chroma (may be subsampled)
    let total_pixels = if bh > 0 {
        (bh - 1) * pixel_stride.unsigned_abs() + pw
    } else {
        0
    };
    // Luma needs full resolution rows (bh << ss_y rows, width pw << ss_x)
    let luma_pw = pw << (is_sx as usize);
    let luma_bh = bh << ss_y;
    let total_luma_pixels = if luma_bh > 0 {
        (luma_bh - 1) * luma_pixel_stride.unsigned_abs() + luma_pw
    } else {
        0
    };

    // SAFETY: grain_lut byte view for reinterpretation
    let grain_lut_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            grain_lut as *const _ as *const u8,
            std::mem::size_of_val(grain_lut),
        )
    };

    match BD::BPC {
        BPC::BPC8 => {
            let src_guard = src_row.slice::<BD>(total_pixels);
            let src_slice: &[u8] =
                crate::src::safe_simd::pixel_access::reinterpret_slice(&*src_guard)
                    .expect("8bpc pixel reinterpret");
            let mut dst_guard = dst_row.slice_mut::<BD>(total_pixels);
            let dst_slice: &mut [u8] =
                crate::src::safe_simd::pixel_access::reinterpret_slice_mut(&mut *dst_guard)
                    .expect("8bpc pixel reinterpret");
            let luma_guard = luma_row.slice::<BD>(total_luma_pixels);
            let luma_slice: &[u8] =
                crate::src::safe_simd::pixel_access::reinterpret_slice(&*luma_guard)
                    .expect("8bpc luma reinterpret");
            let grain_lut_8: &[[i8; GRAIN_WIDTH]; GRAIN_HEIGHT + 1] =
                zerocopy::Ref::<_, [[i8; GRAIN_WIDTH]; GRAIN_HEIGHT + 1]>::new(grain_lut_bytes)
                    .expect("grain_lut reinterpret to i8")
                    .into_ref();
            fguv_inner_safe_8bpc(
                token,
                dst_slice,
                src_slice,
                pixel_stride as isize,
                data,
                pw,
                scaling_bytes,
                grain_lut_8,
                bh,
                row_num,
                luma_slice,
                luma_pixel_stride as isize,
                is_uv,
                is_id,
                is_sx,
                is_sy,
            );
        }
        BPC::BPC16 => {
            let src_guard = src_row.slice::<BD>(total_pixels);
            let src_slice: &[u16] =
                crate::src::safe_simd::pixel_access::reinterpret_slice(&*src_guard)
                    .expect("16bpc pixel reinterpret");
            let mut dst_guard = dst_row.slice_mut::<BD>(total_pixels);
            let dst_slice: &mut [u16] =
                crate::src::safe_simd::pixel_access::reinterpret_slice_mut(&mut *dst_guard)
                    .expect("16bpc pixel reinterpret");
            let luma_guard = luma_row.slice::<BD>(total_luma_pixels);
            let luma_slice: &[u16] =
                crate::src::safe_simd::pixel_access::reinterpret_slice(&*luma_guard)
                    .expect("16bpc luma reinterpret");
            let grain_lut_16: &[[i16; GRAIN_WIDTH]; GRAIN_HEIGHT + 1] =
                zerocopy::Ref::<_, [[i16; GRAIN_WIDTH]; GRAIN_HEIGHT + 1]>::new(grain_lut_bytes)
                    .expect("grain_lut reinterpret to i16")
                    .into_ref();
            fguv_inner_safe_16bpc(
                token,
                dst_slice,
                src_slice,
                pixel_stride as isize,
                data,
                pw,
                scaling_bytes,
                grain_lut_16,
                bh,
                row_num,
                luma_slice,
                luma_pixel_stride as isize,
                is_uv,
                is_id,
                is_sx,
                is_sy,
                bd.into_c(),
            );
        }
    }
    true
}
