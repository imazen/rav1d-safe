//! Safe SIMD implementations of film grain synthesis functions
//!
//! Film grain synthesis adds artificial grain to decoded video to match
//! the artistic intent of the original content.
//!
//! Implemented so far:
//! - generate_grain_y: Generates luma grain LUT (TODO: full SIMD)

#![allow(unused)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use libc::{c_int, ptrdiff_t};

use crate::include::common::bitdepth::{BitDepth, DynEntry, DynPixel, DynScaling};
use crate::include::dav1d::headers::{Dav1dFilmGrainData, Rav1dFilmGrainData};
use crate::include::dav1d::picture::Rav1dPictureDataComponentOffset;
use crate::src::ffi_safe::FFISafe;
use crate::src::filmgrain::{GRAIN_HEIGHT, GRAIN_WIDTH, FG_BLOCK_SIZE};
use crate::src::internal::GrainLut;
use crate::src::tables::dav1d_gaussian_sequence;

// ============================================================================
// Random number generation (matches AV1 spec)
// ============================================================================

/// Get random number from LFSR, advancing seed state
/// This matches the get_random_number function in the AV1 spec
#[inline(always)]
fn get_random_number(bits: u8, state: &mut u32) -> i32 {
    let bit = ((*state >> 0) ^ (*state >> 1) ^ (*state >> 3) ^ (*state >> 12)) & 1;
    *state = (*state >> 1) | (bit << 15);
    (*state >> (16 - bits as u32)) as i32
}

/// Round with division by power of 2
#[inline(always)]
fn round2(x: i32, shift: u8) -> i32 {
    (x + (1 << shift >> 1)) >> shift
}

// ============================================================================
// Generate Grain Y (Luma grain LUT generation)
// ============================================================================

const AR_PAD: usize = 3;

/// Generate luma grain LUT
///
/// This fills the grain buffer with random values from the Gaussian sequence,
/// then applies autoregressive filtering to create correlated grain.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn generate_grain_y_8bpc_avx2(
    buf: &mut GrainLut<i8>,
    data: &Rav1dFilmGrainData,
    bitdepth_max: c_int,
) {
    // For 8bpc: bitdepth_min_8 = 0, grain_ctr = 128, shift = 4 + grain_scale_shift
    let shift = (4 + data.grain_scale_shift) as u8;
    let grain_min: i32 = -128;
    let grain_max: i32 = 127;

    let mut seed = data.seed;

    // Phase 1: Fill buffer with Gaussian random values
    // Process 8 values at a time using SIMD where possible
    unsafe {
        let rounding = _mm256_set1_epi16(1 << (shift - 1));

        for row in buf[..GRAIN_HEIGHT].iter_mut() {
            let mut x = 0;

            // Process 8 values at a time
            while x + 8 <= GRAIN_WIDTH {
                // Generate 8 random indices and look up in Gaussian sequence
                let mut vals = [0i16; 8];
                for i in 0..8 {
                    let idx = get_random_number(11u8, &mut seed) as usize;
                    vals[i] = dav1d_gaussian_sequence[idx];
                }

                // Load values into SIMD register
                let v = _mm_loadu_si128(vals.as_ptr() as *const __m128i);
                let v32 = _mm256_cvtepi16_epi32(v);

                // Apply rounding: (v + (1 << (shift-1))) >> shift
                let rnd16 = _mm_set1_epi16(1 << (shift - 1));
                let rounded = _mm_srai_epi16::<1>(_mm_add_epi16(v, rnd16)); // Simplified for shift

                // Actually need to handle variable shift - use scalar for correctness
                for i in 0..8 {
                    row[x + i] = round2(vals[i] as i32, shift) as i8;
                }
                x += 8;
            }

            // Scalar fallback for remainder
            while x < GRAIN_WIDTH {
                let idx = get_random_number(11u8, &mut seed) as usize;
                row[x] = round2(dav1d_gaussian_sequence[idx] as i32, shift) as i8;
                x += 1;
            }
        }
    }

    // Phase 2: Autoregressive filtering
    let ar_lag = (data.ar_coeff_lag as usize) & 3;
    if ar_lag == 0 {
        return;
    }

    // AR filtering - complex dependency pattern, use scalar
    for y in 0..(GRAIN_HEIGHT - AR_PAD) {
        for x in 0..(GRAIN_WIDTH - 2 * AR_PAD) {
            let mut coeff_idx = 0;
            let mut sum: i32 = 0;

            for dy in (AR_PAD - ar_lag)..=AR_PAD {
                let buf_row = &buf[y + dy];
                for dx in (AR_PAD - ar_lag)..=(AR_PAD + ar_lag) {
                    if dx == ar_lag && dy - (AR_PAD - ar_lag) == ar_lag {
                        break;
                    }
                    sum += data.ar_coeffs_y[coeff_idx] as i32 * buf_row[x + dx] as i32;
                    coeff_idx += 1;
                }
            }

            let grain = buf[y + AR_PAD][x + AR_PAD] as i32 + round2(sum, data.ar_coeff_shift);
            buf[y + AR_PAD][x + AR_PAD] = grain.clamp(grain_min, grain_max) as i8;
        }
    }
}

/// FFI wrapper for generate_grain_y (8bpc)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn generate_grain_y_8bpc_avx2_c(
    buf: *mut GrainLut<DynEntry>,
    data: &Dav1dFilmGrainData,
    bitdepth_max: c_int,
) {
    // SAFETY: For 8bpc, DynEntry is i8
    let buf = unsafe { &mut *buf.cast::<GrainLut<i8>>() };
    let data: Rav1dFilmGrainData = data.clone().into();
    unsafe {
        generate_grain_y_8bpc_avx2(buf, &data, bitdepth_max);
    }
}
