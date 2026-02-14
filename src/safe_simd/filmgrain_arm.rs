//! Safe ARM NEON implementations of film grain synthesis functions.
//!
//! Replaces ARM NEON ASM when the `asm` feature is disabled on aarch64.
//! - generate_grain_y/uv: Scalar (LFSR is inherently serial)
//! - fgy_32x32xn: NEON SIMD inner loop for grain application
//! - fguv_32x32xn: NEON SIMD inner loop for chroma grain application

#![cfg_attr(not(feature = "unchecked"), forbid(unsafe_code))]
#![cfg_attr(feature = "unchecked", deny(unsafe_code))]

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

use crate::src::safe_simd::pixel_access::Flex;
#[cfg(target_arch = "aarch64")]
use archmage::{arcane, Arm64, SimdToken};
#[cfg(target_arch = "aarch64")]
use safe_unaligned_simd::aarch64 as safe_simd;

use std::cmp;
use std::ffi::c_int;
use std::ffi::c_uint;

use libc::{intptr_t, ptrdiff_t};

use crate::include::common::bitdepth::{DynEntry, DynPixel, DynScaling};
use crate::include::dav1d::headers::{Dav1dFilmGrainData, Rav1dFilmGrainData};
use crate::include::dav1d::picture::PicOffset;
use crate::src::ffi_safe::FFISafe;
use crate::src::filmgrain::{FG_BLOCK_SIZE, GRAIN_HEIGHT, GRAIN_WIDTH};
use crate::src::internal::GrainLut;
use crate::src::tables::dav1d_gaussian_sequence;

// ============================================================================
// Helper functions (shared with x86 filmgrain)
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

fn grain_offsets(randval: c_int, is_subx: bool, is_suby: bool) -> (usize, usize) {
    let subx = is_subx as usize;
    let suby = is_suby as usize;
    let offx = 3 + (2 >> subx) * (3 + ((randval as usize) >> 4));
    let offy = 3 + (2 >> suby) * (3 + ((randval as usize) & 0xF));
    (offx, offy)
}

// ============================================================================
// generate_grain_y - 8bpc (scalar)
// ============================================================================

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
                    if dx == AR_PAD && dy == AR_PAD {
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

#[cfg(feature = "asm")]
pub unsafe extern "C" fn generate_grain_y_8bpc_neon(
    buf: *mut GrainLut<DynEntry>,
    data: &Dav1dFilmGrainData,
    _bitdepth_max: c_int,
) {
    let buf = unsafe { &mut *buf.cast::<GrainLut<i8>>() };
    let data: Rav1dFilmGrainData = data.clone().into();
    generate_grain_y_inner_8bpc(buf, &data);
}

// ============================================================================
// generate_grain_y - 16bpc (scalar)
// ============================================================================

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
                    if dx == AR_PAD && dy == AR_PAD {
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

#[cfg(feature = "asm")]
pub unsafe extern "C" fn generate_grain_y_16bpc_neon(
    buf: *mut GrainLut<DynEntry>,
    data: &Dav1dFilmGrainData,
    bitdepth_max: c_int,
) {
    let buf = unsafe { &mut *buf.cast::<GrainLut<i16>>() };
    let data: Rav1dFilmGrainData = data.clone().into();
    let bitdepth = if bitdepth_max >= 4095 { 12 } else { 10 };
    generate_grain_y_inner_16bpc(buf, &data, bitdepth);
}

// ============================================================================
// generate_grain_uv - 8bpc (scalar)
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
    for y in 0..chromah - AR_PAD {
        for x in 0..chromaw - 2 * AR_PAD {
            let mut coeff_idx = 0usize;
            let mut sum: i32 = 0;
            for dy in (AR_PAD - ar_lag)..=AR_PAD {
                for dx in (AR_PAD - ar_lag)..=(AR_PAD + ar_lag) {
                    if dx == AR_PAD && dy == AR_PAD {
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

macro_rules! gen_grain_uv_8bpc {
    ($name:ident, $is_subx:expr, $is_suby:expr) => {
        #[cfg(feature = "asm")]
        pub unsafe extern "C" fn $name(
            buf: *mut GrainLut<DynEntry>,
            buf_y: *const GrainLut<DynEntry>,
            data: &Dav1dFilmGrainData,
            uv: intptr_t,
            _bitdepth_max: c_int,
        ) {
            let buf = unsafe { &mut *buf.cast::<GrainLut<i8>>() };
            let buf_y = unsafe { &*buf_y.cast::<GrainLut<i8>>() };
            let data: Rav1dFilmGrainData = data.clone().into();
            generate_grain_uv_inner_8bpc(buf, buf_y, &data, uv != 0, $is_subx, $is_suby);
        }
    };
}

gen_grain_uv_8bpc!(generate_grain_uv_420_8bpc_neon, true, true);
gen_grain_uv_8bpc!(generate_grain_uv_422_8bpc_neon, true, false);
gen_grain_uv_8bpc!(generate_grain_uv_444_8bpc_neon, false, false);

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
    for y in 0..chromah - AR_PAD {
        for x in 0..chromaw - 2 * AR_PAD {
            let mut coeff_idx = 0usize;
            let mut sum: i32 = 0;
            for dy in (AR_PAD - ar_lag)..=AR_PAD {
                for dx in (AR_PAD - ar_lag)..=(AR_PAD + ar_lag) {
                    if dx == AR_PAD && dy == AR_PAD {
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

macro_rules! gen_grain_uv_16bpc {
    ($name:ident, $is_subx:expr, $is_suby:expr) => {
        #[cfg(feature = "asm")]
        pub unsafe extern "C" fn $name(
            buf: *mut GrainLut<DynEntry>,
            buf_y: *const GrainLut<DynEntry>,
            data: &Dav1dFilmGrainData,
            uv: intptr_t,
            bitdepth_max: c_int,
        ) {
            let buf = unsafe { &mut *buf.cast::<GrainLut<i16>>() };
            let buf_y = unsafe { &*buf_y.cast::<GrainLut<i16>>() };
            let data: Rav1dFilmGrainData = data.clone().into();
            let bitdepth = if bitdepth_max >= 4095 { 12 } else { 10 };
            generate_grain_uv_inner_16bpc(buf, buf_y, &data, uv != 0, $is_subx, $is_suby, bitdepth);
        }
    };
}

gen_grain_uv_16bpc!(generate_grain_uv_420_16bpc_neon, true, true);
gen_grain_uv_16bpc!(generate_grain_uv_422_16bpc_neon, true, false);
gen_grain_uv_16bpc!(generate_grain_uv_444_16bpc_neon, false, false);

// ============================================================================
// fgy_32x32xn - 8bpc NEON
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[arcane]
fn fgy_row_neon_8bpc(
    _token: Arm64,
    dst: &mut [u8],
    src: &[u8],
    scaling: &[u8],
    grain_row: &[i8],
    bw: usize,
    xstart: usize,
    min_val: u8,
    max_val: u8,
    scaling_shift: u8,
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    let scaling = scaling.flex();
    let grain_row = grain_row.flex();
    let mul = vdupq_n_s16(1i16 << (15 - scaling_shift));
    let min_vec = vdupq_n_u8(min_val);
    let max_vec = vdupq_n_u8(max_val);
    let mut x = xstart;

    // Process 16 pixels at a time with NEON
    while x + 16 <= bw {
        let src_arr: &[u8; 16] = src[x..x + 16].try_into().unwrap();
        let src_vec = safe_simd::vld1q_u8(src_arr);
        let src_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(src_vec)));
        let src_hi = vreinterpretq_s16_u16(vmovl_high_u8(src_vec));

        // Scalar scaling lookups
        let mut sc_arr = [0i16; 16];
        for i in 0..16 {
            sc_arr[i] = scaling[src[x + i] as usize] as i16;
        }
        let sc_lo = safe_simd::vld1q_s16(<&[i16; 8]>::try_from(&sc_arr[..8]).unwrap());
        let sc_hi = safe_simd::vld1q_s16(<&[i16; 8]>::try_from(&sc_arr[8..16]).unwrap());

        // Load grain and widen to i16
        let grain_arr: &[i8; 16] = grain_row[x..x + 16].try_into().unwrap();
        let grain_vec = safe_simd::vld1q_s8(grain_arr);
        let grain_lo = vmovl_s8(vget_low_s8(grain_vec));
        let grain_hi = vmovl_high_s8(grain_vec);

        // noise = sc * grain (fits i16 for 8bpc: max 255*127=32385)
        let noise_lo = vmulq_s16(sc_lo, grain_lo);
        let noise_hi = vmulq_s16(sc_hi, grain_hi);

        // Round: (noise * mul + 16384) >> 15 (same as pmulhrsw)
        let noise_lo = vqrdmulhq_s16(noise_lo, mul);
        let noise_hi = vqrdmulhq_s16(noise_hi, mul);

        // Add noise to source
        let result_lo = vaddq_s16(src_lo, noise_lo);
        let result_hi = vaddq_s16(src_hi, noise_hi);

        // Saturating narrow to u8
        let result = vcombine_u8(vqmovun_s16(result_lo), vqmovun_s16(result_hi));
        let result = vmaxq_u8(result, min_vec);
        let result = vminq_u8(result, max_vec);
        let dst_arr: &mut [u8; 16] = (&mut dst[x..x + 16]).try_into().unwrap();
        safe_simd::vst1q_u8(dst_arr, result);
        x += 16;
    }

    // Scalar remainder
    while x < bw {
        let sv = src[x] as usize;
        let grain = grain_row[x] as i32;
        let sc = scaling[sv] as i32;
        let noise = round2(sc * grain, scaling_shift);
        dst[x] = ((src[x] as i32 + noise).clamp(min_val as i32, max_val as i32)) as u8;
        x += 1;
    }
}

fn fgy_inner_8bpc(
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

    #[cfg(target_arch = "aarch64")]
    let token = Arm64::summon().unwrap();

    let mut offsets: [[c_int; 2]; 2] = [[0; 2]; 2];
    static W: [[i32; 2]; 2] = [[27, 17], [17, 27]];

    let row_off = |y: usize| -> usize { (y as isize * stride) as usize };

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
        let (offx_10, offy_10) = if data.overlap_flag && bx != 0 {
            grain_offsets(offsets[1][0], false, false)
        } else {
            (0, 0)
        };
        let (offx_01, offy_01) = if data.overlap_flag && row_num != 0 {
            grain_offsets(offsets[0][1], false, false)
        } else {
            (0, 0)
        };
        let (offx_11, offy_11) = if data.overlap_flag && bx != 0 && row_num != 0 {
            grain_offsets(offsets[1][1], false, false)
        } else {
            (0, 0)
        };

        // Main rows
        for y in ystart..bh {
            let base = row_off(y).wrapping_add(bx);

            // x-overlap (scalar)
            for x in 0..xstart {
                let sv = src[base + x] as usize;
                let grain = grain_lut[offy + y][offx + x] as i32;
                let old = grain_lut[offy_10 + y][offx_10 + x + FG_BLOCK_SIZE] as i32;
                let blended = round2(old * W[x][0] + grain * W[x][1], 5).clamp(-128, 127);
                let sc = scaling[sv] as i32;
                let noise = round2(sc * blended, scaling_shift);
                dst[base + x] = ((src[base + x] as i32 + noise).clamp(min_value, max_value)) as u8;
            }

            #[cfg(target_arch = "aarch64")]
            fgy_row_neon_8bpc(
                token,
                &mut dst[base..],
                &src[base..],
                scaling,
                &grain_lut[offy + y][offx..],
                bw,
                xstart,
                min_value as u8,
                max_value as u8,
                scaling_shift,
            );

            // Scalar fallback for non-aarch64 (xstart..bw)
            #[cfg(not(target_arch = "aarch64"))]
            for x in xstart..bw {
                let sv = src[base + x] as usize;
                let grain = grain_lut[offy + y][offx + x] as i32;
                let sc = scaling[sv] as i32;
                let noise = round2(sc * grain, scaling_shift);
                dst[base + x] = ((src[base + x] as i32 + noise).clamp(min_value, max_value)) as u8;
            }
        }

        // y-overlap rows (scalar)
        for y in 0..ystart {
            let base = row_off(y).wrapping_add(bx);

            for x in xstart..bw {
                let sv = src[base + x] as usize;
                let grain = grain_lut[offy + y][offx + x] as i32;
                let old = grain_lut[offy_01 + y + FG_BLOCK_SIZE][offx_01 + x] as i32;
                let blended = round2(old * W[y][0] + grain * W[y][1], 5).clamp(-128, 127);
                let sc = scaling[sv] as i32;
                let noise = round2(sc * blended, scaling_shift);
                dst[base + x] = ((src[base + x] as i32 + noise).clamp(min_value, max_value)) as u8;
            }

            for x in 0..xstart {
                let sv = src[base + x] as usize;
                let top = grain_lut[offy_01 + y + FG_BLOCK_SIZE][offx_01 + x] as i32;
                let old_top =
                    grain_lut[offy_11 + y + FG_BLOCK_SIZE][offx_11 + x + FG_BLOCK_SIZE] as i32;
                let top = round2(old_top * W[x][0] + top * W[x][1], 5).clamp(-128, 127);
                let grain = grain_lut[offy + y][offx + x] as i32;
                let old = grain_lut[offy_10 + y][offx_10 + x + FG_BLOCK_SIZE] as i32;
                let grain = round2(old * W[x][0] + grain * W[x][1], 5).clamp(-128, 127);
                let blended = round2(top * W[y][0] + grain * W[y][1], 5).clamp(-128, 127);
                let sc = scaling[sv] as i32;
                let noise = round2(sc * blended, scaling_shift);
                dst[base + x] = ((src[base + x] as i32 + noise).clamp(min_value, max_value)) as u8;
            }
        }
    }
}

#[cfg(feature = "asm")]
pub unsafe extern "C" fn fgy_32x32xn_8bpc_neon(
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
    let data: Rav1dFilmGrainData = data.clone().into();
    let bh = bh as usize;
    let row_num = row_num as usize;
    let total_size = bh * stride.unsigned_abs() + pw;
    let dst = unsafe { std::slice::from_raw_parts_mut(dst_row_ptr as *mut u8, total_size) };
    let src = unsafe { std::slice::from_raw_parts(src_row_ptr as *const u8, total_size) };
    let scaling = unsafe { std::slice::from_raw_parts(scaling as *const u8, 256) };
    let grain_lut = unsafe { &*(grain_lut as *const [[i8; GRAIN_WIDTH]; GRAIN_HEIGHT + 1]) };
    fgy_inner_8bpc(dst, src, stride, &data, pw, scaling, grain_lut, bh, row_num);
}

// ============================================================================
// fgy_32x32xn - 16bpc NEON
// ============================================================================

/// NEON helper for 16bpc: add precomputed noise to src, clamp, store to dst.
/// Processes 8 u16 pixels at a time.
#[cfg(target_arch = "aarch64")]
#[arcane]
fn fgy_row_neon_16bpc(
    _token: Arm64,
    dst: &mut [u16],
    src: &[u16],
    scaling: &[u8],
    grain_row: &[i16],
    bw: usize,
    xstart: usize,
    min_value: i32,
    max_value: i32,
    scaling_shift: u8,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    let scaling = scaling.flex();
    let grain_row = grain_row.flex();
    let min_vec = vdupq_n_s16(min_value as i16);
    let max_vec = vdupq_n_s16(max_value as i16);
    let mut x = xstart;

    while x + 8 <= bw {
        let mut noise_vals = [0i16; 8];
        for i in 0..8 {
            let sv = cmp::min(src[x + i] as usize, bitdepth_max as usize);
            let grain = grain_row[x + i] as i32;
            let sc = scaling[sv] as i32;
            noise_vals[i] = round2(sc * grain, scaling_shift) as i16;
        }
        // Load src u16 as u16, reinterpret to s16 in NEON
        let src_u16_arr: &[u16; 8] = (&src[x..x + 8]).try_into().unwrap();
        let src_vec = vreinterpretq_s16_u16(safe_simd::vld1q_u16(src_u16_arr));
        let noise = safe_simd::vld1q_s16(<&[i16; 8]>::try_from(&noise_vals[..8]).unwrap());
        let result = vaddq_s16(src_vec, noise);
        let result = vmaxq_s16(result, min_vec);
        let result = vminq_s16(result, max_vec);
        // Store s16 result as u16
        let result_u16 = vreinterpretq_u16_s16(result);
        let dst_u16_arr: &mut [u16; 8] = (&mut dst[x..x + 8]).try_into().unwrap();
        safe_simd::vst1q_u16(dst_u16_arr, result_u16);
        x += 8;
    }

    // Scalar remainder
    while x < bw {
        let sv = cmp::min(src[x] as usize, bitdepth_max as usize);
        let grain = grain_row[x] as i32;
        let sc = scaling[sv] as i32;
        let noise = round2(sc * grain, scaling_shift);
        dst[x] = ((src[x] as i32 + noise).clamp(min_value, max_value)) as u16;
        x += 1;
    }
}

fn fgy_inner_16bpc(
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

    #[cfg(target_arch = "aarch64")]
    let token = Arm64::summon().unwrap();

    let row_off = |y: usize| -> usize { (y as isize * stride_u16) as usize };

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
        let (offx_10, offy_10) = if data.overlap_flag && bx != 0 {
            grain_offsets(offsets[1][0], false, false)
        } else {
            (0, 0)
        };
        let (offx_01, offy_01) = if data.overlap_flag && row_num != 0 {
            grain_offsets(offsets[0][1], false, false)
        } else {
            (0, 0)
        };
        let (offx_11, offy_11) = if data.overlap_flag && bx != 0 && row_num != 0 {
            grain_offsets(offsets[1][1], false, false)
        } else {
            (0, 0)
        };

        for y in ystart..bh {
            let base = row_off(y).wrapping_add(bx);

            for x in 0..xstart {
                let sv = cmp::min(src[base + x] as usize, bitdepth_max as usize);
                let grain = grain_lut[offy + y][offx + x] as i32;
                let old = grain_lut[offy_10 + y][offx_10 + x + FG_BLOCK_SIZE] as i32;
                let blended =
                    round2(old * W[x][0] + grain * W[x][1], 5).clamp(grain_min, grain_max);
                let sc = scaling[sv] as i32;
                let noise = round2(sc * blended, scaling_shift);
                dst[base + x] = ((src[base + x] as i32 + noise).clamp(min_value, max_value)) as u16;
            }

            #[cfg(target_arch = "aarch64")]
            fgy_row_neon_16bpc(
                token,
                &mut dst[base..],
                &src[base..],
                scaling,
                &grain_lut[offy + y][offx..],
                bw,
                xstart,
                min_value,
                max_value,
                scaling_shift,
                bitdepth_max,
            );

            // Scalar fallback for non-aarch64
            #[cfg(not(target_arch = "aarch64"))]
            {
                let mut x = xstart;
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
        }

        // y-overlap (scalar)
        for y in 0..ystart {
            let base = row_off(y).wrapping_add(bx);

            for x in xstart..bw {
                let sv = cmp::min(src[base + x] as usize, bitdepth_max as usize);
                let grain = grain_lut[offy + y][offx + x] as i32;
                let old = grain_lut[offy_01 + y + FG_BLOCK_SIZE][offx_01 + x] as i32;
                let blended =
                    round2(old * W[y][0] + grain * W[y][1], 5).clamp(grain_min, grain_max);
                let sc = scaling[sv] as i32;
                let noise = round2(sc * blended, scaling_shift);
                dst[base + x] = ((src[base + x] as i32 + noise).clamp(min_value, max_value)) as u16;
            }
            for x in 0..xstart {
                let sv = cmp::min(src[base + x] as usize, bitdepth_max as usize);
                let top = grain_lut[offy_01 + y + FG_BLOCK_SIZE][offx_01 + x] as i32;
                let old_top =
                    grain_lut[offy_11 + y + FG_BLOCK_SIZE][offx_11 + x + FG_BLOCK_SIZE] as i32;
                let top = round2(old_top * W[x][0] + top * W[x][1], 5).clamp(grain_min, grain_max);
                let grain = grain_lut[offy + y][offx + x] as i32;
                let old = grain_lut[offy_10 + y][offx_10 + x + FG_BLOCK_SIZE] as i32;
                let grain = round2(old * W[x][0] + grain * W[x][1], 5).clamp(grain_min, grain_max);
                let blended =
                    round2(top * W[y][0] + grain * W[y][1], 5).clamp(grain_min, grain_max);
                let sc = scaling[sv] as i32;
                let noise = round2(sc * blended, scaling_shift);
                dst[base + x] = ((src[base + x] as i32 + noise).clamp(min_value, max_value)) as u16;
            }
        }
    }
}

#[cfg(feature = "asm")]
pub unsafe extern "C" fn fgy_32x32xn_16bpc_neon(
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
    let data: Rav1dFilmGrainData = data.clone().into();
    let bh = bh as usize;
    let row_num = row_num as usize;
    let stride_u16 = stride / 2;
    let total_size = bh * (stride / 2).unsigned_abs() + pw;
    let dst = unsafe { std::slice::from_raw_parts_mut(dst_row_ptr as *mut u16, total_size) };
    let src = unsafe { std::slice::from_raw_parts(src_row_ptr as *const u16, total_size) };
    let scaling_len = if bitdepth_max >= 4095 { 4096 } else { 1024 };
    let scaling = unsafe { std::slice::from_raw_parts(scaling as *const u8, scaling_len) };
    let grain_lut = unsafe { &*(grain_lut as *const [[i16; GRAIN_WIDTH]; GRAIN_HEIGHT + 1]) };
    fgy_inner_16bpc(
        dst,
        src,
        stride_u16,
        &data,
        pw,
        scaling,
        grain_lut,
        bh,
        row_num,
        bitdepth_max as i32,
    );
}

// ============================================================================
// fguv_32x32xn - 8bpc NEON (chroma grain application)
// ============================================================================

/// NEON helper for 8bpc chroma: apply grain noise with UV scaling.
/// Processes 16 pixels at a time, with scalar remainder.
#[cfg(target_arch = "aarch64")]
#[arcane]
fn fguv_row_neon_8bpc(
    _token: Arm64,
    dst: &mut [u8],
    src: &[u8],
    scaling: &[u8],
    grain_row: &[i8],
    luma_row: &[u8],
    bw: usize,
    xstart: usize,
    min_val: u8,
    max_val: u8,
    scaling_shift: u8,
    is_sx: bool,
    sx: usize,
    data: &Rav1dFilmGrainData,
    uv: usize,
) {
    let mut dst = dst.flex_mut();
    let src = src.flex();
    let scaling = scaling.flex();
    let grain_row = grain_row.flex();
    let luma_row = luma_row.flex();
    let mul = vdupq_n_s16(1i16 << (15 - scaling_shift));
    let min_vec = vdupq_n_u8(min_val);
    let max_vec = vdupq_n_u8(max_val);
    let mut x = xstart;

    while x + 16 <= bw {
        let src_arr: &[u8; 16] = src[x..x + 16].try_into().unwrap();
        let src_vec = safe_simd::vld1q_u8(src_arr);
        let src_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(src_vec)));
        let src_hi = vreinterpretq_s16_u16(vmovl_high_u8(src_vec));

        let mut sc_arr = [0i16; 16];
        for i in 0..16 {
            sc_arr[i] = compute_uv_scaling_val(
                src[x + i],
                &*luma_row,
                (x + i) << sx,
                is_sx,
                data,
                uv,
                &*scaling,
            ) as i16;
        }
        let sc_lo = safe_simd::vld1q_s16(<&[i16; 8]>::try_from(&sc_arr[..8]).unwrap());
        let sc_hi = safe_simd::vld1q_s16(<&[i16; 8]>::try_from(&sc_arr[8..16]).unwrap());

        let grain_arr: &[i8; 16] = grain_row[x..x + 16].try_into().unwrap();
        let grain_vec = safe_simd::vld1q_s8(grain_arr);
        let grain_lo = vmovl_s8(vget_low_s8(grain_vec));
        let grain_hi = vmovl_high_s8(grain_vec);

        let noise_lo = vmulq_s16(sc_lo, grain_lo);
        let noise_hi = vmulq_s16(sc_hi, grain_hi);
        let noise_lo = vqrdmulhq_s16(noise_lo, mul);
        let noise_hi = vqrdmulhq_s16(noise_hi, mul);

        let result_lo = vaddq_s16(src_lo, noise_lo);
        let result_hi = vaddq_s16(src_hi, noise_hi);
        let result = vcombine_u8(vqmovun_s16(result_lo), vqmovun_s16(result_hi));
        let result = vmaxq_u8(result, min_vec);
        let result = vminq_u8(result, max_vec);
        let dst_arr: &mut [u8; 16] = (&mut dst[x..x + 16]).try_into().unwrap();
        safe_simd::vst1q_u8(dst_arr, result);
        x += 16;
    }

    // Scalar remainder
    while x < bw {
        let sc =
            compute_uv_scaling_val(src[x], &*luma_row, x << sx, is_sx, data, uv, &*scaling) as i32;
        let grain = grain_row[x] as i32;
        let noise = round2(sc * grain, scaling_shift);
        dst[x] = ((src[x] as i32 + noise).clamp(min_val as i32, max_val as i32)) as u8;
        x += 1;
    }
}

#[inline(always)]
fn compute_uv_scaling_val(
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

fn fguv_inner_8bpc(
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

    #[cfg(target_arch = "aarch64")]
    let token = Arm64::summon().unwrap();

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
        let (offx_10, offy_10) = if data.overlap_flag && bx != 0 {
            grain_offsets(offsets[1][0], is_sx, is_sy)
        } else {
            (0, 0)
        };
        let (offx_01, offy_01) = if data.overlap_flag && row_num != 0 {
            grain_offsets(offsets[0][1], is_sx, is_sy)
        } else {
            (0, 0)
        };
        let (offx_11, offy_11) = if data.overlap_flag && bx != 0 && row_num != 0 {
            grain_offsets(offsets[1][1], is_sx, is_sy)
        } else {
            (0, 0)
        };

        for y in ystart..bh {
            let base = row_off(y).wrapping_add(bx);
            let luma_base = luma_row_off(y).wrapping_add(bx << sx);

            for x in 0..xstart {
                let grain = grain_lut[offy + y][offx + x] as i32;
                let old = grain_lut[offy_10 + y][offx_10 + x + (FG_BLOCK_SIZE >> sx)] as i32;
                let blended =
                    round2(old * W[sx][x][0] + grain * W[sx][x][1], 5).clamp(grain_min, grain_max);
                dst[base + x] = noise_uv(src[base + x], blended, &luma[luma_base..], x << sx);
            }

            // NEON inner loop
            #[cfg(target_arch = "aarch64")]
            fguv_row_neon_8bpc(
                token,
                &mut dst[base..],
                &src[base..],
                scaling,
                &grain_lut[offy + y][offx..],
                &luma[luma_base..],
                bw,
                xstart,
                min_value as u8,
                max_value as u8,
                scaling_shift,
                is_sx,
                sx,
                data,
                uv,
            );

            // Scalar fallback for non-aarch64
            #[cfg(not(target_arch = "aarch64"))]
            {
                let mut x = xstart;
                while x < bw {
                    let grain = grain_lut[offy + y][offx + x] as i32;
                    dst[base + x] = noise_uv(src[base + x], grain, &luma[luma_base..], x << sx);
                    x += 1;
                }
            }
        }

        // y-overlap (scalar)
        for y in 0..ystart {
            let base = row_off(y).wrapping_add(bx);
            let luma_base = luma_row_off(y).wrapping_add(bx << sx);

            for x in xstart..bw {
                let grain = grain_lut[offy + y][offx + x] as i32;
                let old = grain_lut[offy_01 + y + (FG_BLOCK_SIZE >> sy)][offx_01 + x] as i32;
                let blended =
                    round2(old * W[sy][y][0] + grain * W[sy][y][1], 5).clamp(grain_min, grain_max);
                dst[base + x] = noise_uv(src[base + x], blended, &luma[luma_base..], x << sx);
            }
            for x in 0..xstart {
                let top = grain_lut[offy_01 + y + (FG_BLOCK_SIZE >> sy)][offx_01 + x] as i32;
                let old_top = grain_lut[offy_11 + y + (FG_BLOCK_SIZE >> sy)]
                    [offx_11 + x + (FG_BLOCK_SIZE >> sx)] as i32;
                let top = round2(old_top * W[sx][x][0] + top * W[sx][x][1], 5)
                    .clamp(grain_min, grain_max);
                let grain = grain_lut[offy + y][offx + x] as i32;
                let old = grain_lut[offy_10 + y][offx_10 + x + (FG_BLOCK_SIZE >> sx)] as i32;
                let grain =
                    round2(old * W[sx][x][0] + grain * W[sx][x][1], 5).clamp(grain_min, grain_max);
                let blended =
                    round2(top * W[sy][y][0] + grain * W[sy][y][1], 5).clamp(grain_min, grain_max);
                dst[base + x] = noise_uv(src[base + x], blended, &luma[luma_base..], x << sx);
            }
        }
    }
}

// 8bpc fguv FFI wrappers
macro_rules! fguv_8bpc_wrapper {
    ($name:ident, $is_sx:expr, $is_sy:expr) => {
        #[cfg(feature = "asm")]
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
            _bitdepth_max: c_int,
            _dst_row: *const FFISafe<PicOffset>,
            _src_row: *const FFISafe<PicOffset>,
            _luma_row: *const FFISafe<PicOffset>,
        ) {
            let data: Rav1dFilmGrainData = data.clone().into();
            let bh = bh as usize;
            let row_num = row_num as usize;
            let total_size = bh * stride.unsigned_abs() + pw;
            let sy = $is_sy as usize;
            let sx = $is_sx as usize;
            let luma_total = (bh << sy) * luma_stride.unsigned_abs() + (pw << sx) + sx;
            let dst = unsafe { std::slice::from_raw_parts_mut(dst_row_ptr as *mut u8, total_size) };
            let src = unsafe { std::slice::from_raw_parts(src_row_ptr as *const u8, total_size) };
            let scaling = unsafe { std::slice::from_raw_parts(scaling as *const u8, 256) };
            let grain_lut =
                unsafe { &*(grain_lut as *const [[i8; GRAIN_WIDTH]; GRAIN_HEIGHT + 1]) };
            let luma = unsafe { std::slice::from_raw_parts(luma_row_ptr as *const u8, luma_total) };
            fguv_inner_8bpc(
                dst,
                src,
                stride as isize,
                &data,
                pw,
                scaling,
                grain_lut,
                bh,
                row_num,
                luma,
                luma_stride as isize,
                uv_pl != 0,
                is_id != 0,
                $is_sx,
                $is_sy,
            );
        }
    };
}

fguv_8bpc_wrapper!(fguv_32x32xn_i420_8bpc_neon, true, true);
fguv_8bpc_wrapper!(fguv_32x32xn_i422_8bpc_neon, true, false);
fguv_8bpc_wrapper!(fguv_32x32xn_i444_8bpc_neon, false, false);

// ============================================================================
// fguv_32x32xn - 16bpc (scalar with NEON add/clamp)
// ============================================================================

fn fguv_inner_16bpc(
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
        let (offx_10, offy_10) = if data.overlap_flag && bx != 0 {
            grain_offsets(offsets[1][0], is_sx, is_sy)
        } else {
            (0, 0)
        };
        let (offx_01, offy_01) = if data.overlap_flag && row_num != 0 {
            grain_offsets(offsets[0][1], is_sx, is_sy)
        } else {
            (0, 0)
        };
        let (offx_11, offy_11) = if data.overlap_flag && bx != 0 && row_num != 0 {
            grain_offsets(offsets[1][1], is_sx, is_sy)
        } else {
            (0, 0)
        };

        for y in ystart..bh {
            let base = row_off(y).wrapping_add(bx);
            let luma_base = luma_row_off(y).wrapping_add(bx << sx);

            for x in 0..xstart {
                let grain = grain_lut[offy + y][offx + x] as i32;
                let old = grain_lut[offy_10 + y][offx_10 + x + (FG_BLOCK_SIZE >> sx)] as i32;
                let blended =
                    round2(old * W[sx][x][0] + grain * W[sx][x][1], 5).clamp(grain_min, grain_max);
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
                let old = grain_lut[offy_01 + y + (FG_BLOCK_SIZE >> sy)][offx_01 + x] as i32;
                let blended =
                    round2(old * W[sy][y][0] + grain * W[sy][y][1], 5).clamp(grain_min, grain_max);
                dst[base + x] = noise_uv(src[base + x], blended, &luma[luma_base..], x << sx);
            }
            for x in 0..xstart {
                let top = grain_lut[offy_01 + y + (FG_BLOCK_SIZE >> sy)][offx_01 + x] as i32;
                let old_top = grain_lut[offy_11 + y + (FG_BLOCK_SIZE >> sy)]
                    [offx_11 + x + (FG_BLOCK_SIZE >> sx)] as i32;
                let top = round2(old_top * W[sx][x][0] + top * W[sx][x][1], 5)
                    .clamp(grain_min, grain_max);
                let grain = grain_lut[offy + y][offx + x] as i32;
                let old = grain_lut[offy_10 + y][offx_10 + x + (FG_BLOCK_SIZE >> sx)] as i32;
                let grain =
                    round2(old * W[sx][x][0] + grain * W[sx][x][1], 5).clamp(grain_min, grain_max);
                let blended =
                    round2(top * W[sy][y][0] + grain * W[sy][y][1], 5).clamp(grain_min, grain_max);
                dst[base + x] = noise_uv(src[base + x], blended, &luma[luma_base..], x << sx);
            }
        }
    }
}

macro_rules! fguv_16bpc_wrapper {
    ($name:ident, $is_sx:expr, $is_sy:expr) => {
        #[cfg(feature = "asm")]
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
            let data: Rav1dFilmGrainData = data.clone().into();
            let bh = bh as usize;
            let row_num = row_num as usize;
            let stride_u16 = stride / 2;
            let sy = $is_sy as usize;
            let sx = $is_sx as usize;
            let total_size = bh * (stride / 2).unsigned_abs() + pw;
            let luma_total = (bh << sy) * (luma_stride / 2).unsigned_abs() + (pw << sx) + sx;
            let scaling_len = if bitdepth_max >= 4095 { 4096 } else { 1024 };
            let dst =
                unsafe { std::slice::from_raw_parts_mut(dst_row_ptr as *mut u16, total_size) };
            let src = unsafe { std::slice::from_raw_parts(src_row_ptr as *const u16, total_size) };
            let scaling = unsafe { std::slice::from_raw_parts(scaling as *const u8, scaling_len) };
            let grain_lut =
                unsafe { &*(grain_lut as *const [[i16; GRAIN_WIDTH]; GRAIN_HEIGHT + 1]) };
            let luma =
                unsafe { std::slice::from_raw_parts(luma_row_ptr as *const u16, luma_total) };
            fguv_inner_16bpc(
                dst,
                src,
                stride_u16,
                &data,
                pw,
                scaling,
                grain_lut,
                bh,
                row_num,
                luma,
                luma_stride / 2,
                uv_pl != 0,
                is_id != 0,
                $is_sx,
                $is_sy,
                bitdepth_max as i32,
            );
        }
    };
}

fguv_16bpc_wrapper!(fguv_32x32xn_i420_16bpc_neon, true, true);
fguv_16bpc_wrapper!(fguv_32x32xn_i422_16bpc_neon, true, false);
fguv_16bpc_wrapper!(fguv_32x32xn_i444_16bpc_neon, false, false);

// ============================================================================
// Safe dispatch wrappers — encapsulate unsafe pointer creation and FFI calls
// ============================================================================

use crate::include::common::bitdepth::{BitDepth, BPC};
use crate::include::dav1d::headers::Rav1dPixelLayoutSubSampled;
use crate::include::dav1d::picture::Rav1dPictureDataComponent;
use crate::src::strided::Strided as _;

/// Safe dispatch for generate_grain_y (aarch64 NEON).
/// NEON is always available on aarch64, so this always dispatches and returns true.
#[cfg(target_arch = "aarch64")]
pub fn generate_grain_y_dispatch<BD: BitDepth>(
    buf: &mut GrainLut<BD::Entry>,
    data: &Rav1dFilmGrainData,
    bd: BD,
) -> bool {
    use zerocopy::{FromBytes, IntoBytes};
    match BD::BPC {
        BPC::BPC8 => {
            let buf: &mut GrainLut<i8> = FromBytes::mut_from_bytes(buf.as_mut_bytes()).unwrap();
            generate_grain_y_inner_8bpc(buf, data);
        }
        BPC::BPC16 => {
            let buf: &mut GrainLut<i16> = FromBytes::mut_from_bytes(buf.as_mut_bytes()).unwrap();
            let bitdepth = if bd.into_c() >= 4095 { 12 } else { 10 };
            generate_grain_y_inner_16bpc(buf, data, bitdepth);
        }
    }
    true
}

/// Safe dispatch for generate_grain_uv (aarch64 NEON).
/// NEON is always available on aarch64, so this always dispatches and returns true.
#[cfg(target_arch = "aarch64")]
pub fn generate_grain_uv_dispatch<BD: BitDepth>(
    layout: Rav1dPixelLayoutSubSampled,
    buf: &mut GrainLut<BD::Entry>,
    buf_y: &GrainLut<BD::Entry>,
    data: &Rav1dFilmGrainData,
    is_uv: bool,
    bd: BD,
) -> bool {
    let (is_subx, is_suby) = match layout {
        Rav1dPixelLayoutSubSampled::I420 => (true, true),
        Rav1dPixelLayoutSubSampled::I422 => (true, false),
        Rav1dPixelLayoutSubSampled::I444 => (false, false),
    };
    use zerocopy::{FromBytes, IntoBytes};
    match BD::BPC {
        BPC::BPC8 => {
            let buf: &mut GrainLut<i8> = FromBytes::mut_from_bytes(buf.as_mut_bytes()).unwrap();
            let buf_y: &GrainLut<i8> = FromBytes::ref_from_bytes(buf_y.as_bytes()).unwrap();
            generate_grain_uv_inner_8bpc(buf, buf_y, data, is_uv, is_subx, is_suby);
        }
        BPC::BPC16 => {
            let buf: &mut GrainLut<i16> = FromBytes::mut_from_bytes(buf.as_mut_bytes()).unwrap();
            let buf_y: &GrainLut<i16> = FromBytes::ref_from_bytes(buf_y.as_bytes()).unwrap();
            let bitdepth = if bd.into_c() >= 4095 { 12 } else { 10 };
            generate_grain_uv_inner_16bpc(buf, buf_y, data, is_uv, is_subx, is_suby, bitdepth);
        }
    }
    true
}

/// Safe dispatch for fgy_32x32xn (aarch64 NEON).
/// NEON is always available on aarch64, so this always dispatches and returns true.
#[cfg(target_arch = "aarch64")]
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
    use zerocopy::{FromBytes, IntoBytes};
    let row_strides = (row_num * FG_BLOCK_SIZE) as isize;
    let dst_row = dst.with_offset::<BD>() + row_strides * dst.pixel_stride::<BD>();
    let src_row = src.with_offset::<BD>() + row_strides * src.pixel_stride::<BD>();

    let stride = dst.stride();

    match BD::BPC {
        BPC::BPC8 => {
            let (mut dst_guard, dst_base) = dst_row.full_guard_mut::<BD>();
            let dst_bytes = dst_guard.as_mut_bytes();
            let dst_slice = &mut dst_bytes[dst_base..];
            let (src_guard, src_base) = src_row.full_guard::<BD>();
            let src_bytes = src_guard.as_bytes();
            let src_slice = &src_bytes[src_base..];
            let scaling_bytes: &[u8] = scaling.as_ref();
            let grain_lut_8: &[[i8; GRAIN_WIDTH]; GRAIN_HEIGHT + 1] =
                FromBytes::ref_from_bytes(grain_lut.as_bytes()).unwrap();
            fgy_inner_8bpc(
                dst_slice,
                src_slice,
                stride,
                data,
                pw,
                scaling_bytes,
                grain_lut_8,
                bh,
                row_num,
            );
        }
        BPC::BPC16 => {
            let (mut dst_guard, dst_base) = dst_row.full_guard_mut::<BD>();
            let dst_bytes = dst_guard.as_mut_bytes();
            let base_byte = dst_base * std::mem::size_of::<BD::Pixel>();
            let dst_u16: &mut [u16] =
                FromBytes::mut_from_bytes(&mut dst_bytes[base_byte..]).unwrap();
            let (src_guard, src_base) = src_row.full_guard::<BD>();
            let src_bytes = src_guard.as_bytes();
            let src_base_byte = src_base * std::mem::size_of::<BD::Pixel>();
            let src_u16: &[u16] = FromBytes::ref_from_bytes(&src_bytes[src_base_byte..]).unwrap();
            let stride_u16 = stride / 2;
            let scaling_bytes: &[u8] = scaling.as_ref();
            let grain_lut_16: &[[i16; GRAIN_WIDTH]; GRAIN_HEIGHT + 1] =
                FromBytes::ref_from_bytes(grain_lut.as_bytes()).unwrap();
            fgy_inner_16bpc(
                dst_u16,
                src_u16,
                stride_u16,
                data,
                pw,
                scaling_bytes,
                grain_lut_16,
                bh,
                row_num,
                bd.into_c() as i32,
            );
        }
    }
    true
}

/// Safe dispatch for fguv_32x32xn (aarch64 NEON).
/// NEON is always available on aarch64, so this always dispatches and returns true.
#[cfg(target_arch = "aarch64")]
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
    use zerocopy::{FromBytes, IntoBytes};
    let ss_y = (layout == Rav1dPixelLayoutSubSampled::I420) as usize;
    let row_strides = (row_num * FG_BLOCK_SIZE) as isize;
    let dst_row = dst.with_offset::<BD>() + (row_strides * dst.pixel_stride::<BD>() >> ss_y);
    let src_row = src.with_offset::<BD>() + (row_strides * src.pixel_stride::<BD>() >> ss_y);

    let stride = dst.stride();
    let luma_row = luma.with_offset::<BD>() + (row_strides * luma.pixel_stride::<BD>());
    let luma_stride = luma.stride();

    let (is_sx, is_sy) = match layout {
        Rav1dPixelLayoutSubSampled::I420 => (true, true),
        Rav1dPixelLayoutSubSampled::I422 => (true, false),
        Rav1dPixelLayoutSubSampled::I444 => (false, false),
    };

    match BD::BPC {
        BPC::BPC8 => {
            let (mut dst_guard, dst_base) = dst_row.full_guard_mut::<BD>();
            let dst_bytes = dst_guard.as_mut_bytes();
            let dst_slice = &mut dst_bytes[dst_base..];
            let (src_guard, src_base) = src_row.full_guard::<BD>();
            let src_bytes = src_guard.as_bytes();
            let src_slice = &src_bytes[src_base..];
            let scaling_bytes: &[u8] = scaling.as_ref();
            let grain_lut_8: &[[i8; GRAIN_WIDTH]; GRAIN_HEIGHT + 1] =
                FromBytes::ref_from_bytes(grain_lut.as_bytes()).unwrap();
            let (luma_guard, luma_base) = luma_row.full_guard::<BD>();
            let luma_bytes = luma_guard.as_bytes();
            let luma_slice = &luma_bytes[luma_base..];
            fguv_inner_8bpc(
                dst_slice,
                src_slice,
                stride as isize,
                data,
                pw,
                scaling_bytes,
                grain_lut_8,
                bh,
                row_num,
                luma_slice,
                luma_stride as isize,
                is_uv,
                is_id,
                is_sx,
                is_sy,
            );
        }
        BPC::BPC16 => {
            let (mut dst_guard, dst_base) = dst_row.full_guard_mut::<BD>();
            let dst_bytes = dst_guard.as_mut_bytes();
            let base_byte = dst_base * std::mem::size_of::<BD::Pixel>();
            let dst_u16: &mut [u16] =
                FromBytes::mut_from_bytes(&mut dst_bytes[base_byte..]).unwrap();
            let (src_guard, src_base) = src_row.full_guard::<BD>();
            let src_bytes = src_guard.as_bytes();
            let src_base_byte = src_base * std::mem::size_of::<BD::Pixel>();
            let src_u16: &[u16] = FromBytes::ref_from_bytes(&src_bytes[src_base_byte..]).unwrap();
            let scaling_bytes: &[u8] = scaling.as_ref();
            let grain_lut_16: &[[i16; GRAIN_WIDTH]; GRAIN_HEIGHT + 1] =
                FromBytes::ref_from_bytes(grain_lut.as_bytes()).unwrap();
            let (luma_guard, luma_base) = luma_row.full_guard::<BD>();
            let luma_bytes = luma_guard.as_bytes();
            let luma_base_byte = luma_base * std::mem::size_of::<BD::Pixel>();
            let luma_u16: &[u16] =
                FromBytes::ref_from_bytes(&luma_bytes[luma_base_byte..]).unwrap();
            fguv_inner_16bpc(
                dst_u16,
                src_u16,
                stride / 2,
                data,
                pw,
                scaling_bytes,
                grain_lut_16,
                bh,
                row_num,
                luma_u16,
                luma_stride / 2,
                is_uv,
                is_id,
                is_sx,
                is_sy,
                bd.into_c() as i32,
            );
        }
    }
    true
}
