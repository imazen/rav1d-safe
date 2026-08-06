//! Safe wasm128 SIMD implementations of motion compensation functions
//!
//! Port of the AVX2 MC functions from mc.rs to wasm128, processing
//! half the width per iteration (128-bit vs 256-bit registers).
//!
//! Key intrinsic differences from AVX2:
//! - No pmulhrsw: synthesize from i32x4_extmul + add + shift + narrow
//! - No pmulhi: synthesize from i32x4_extmul + shift + narrow
//! - No cross-lane pack: i16x8_narrow_i32x4 / u8x16_narrow_i16x8 are in-order

#![forbid(unsafe_code)]

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

#[cfg(target_arch = "wasm32")]
use archmage::{Wasm128Token, arcane};

#[cfg(target_arch = "wasm32")]
use crate::include::common::bitdepth::BitDepth;
#[cfg(target_arch = "wasm32")]
use crate::include::dav1d::picture::PicOffset;
#[cfg(target_arch = "wasm32")]
use crate::src::internal::COMPINTER_LEN;

#[cfg(target_arch = "wasm32")]
use crate::src::safe_simd::pixel_access::wasm_load_128;

/// Rounding constant for pmulhrsw equivalent: 1024 = (1 << 10)
#[cfg(target_arch = "wasm32")]
const PW_1024: i16 = 1024;

/// Rounding constant for w_avg: 2048 = (1 << 11)
#[cfg(target_arch = "wasm32")]
const PW_2048: i16 = 2048;

// ============================================================
// AVG (average two predictions)
// ============================================================

/// AVG operation for 8-bit pixels using wasm128
///
/// Processes 8 pixels per iteration (vs 32 for AVX2).
/// Synthesizes pmulhrsw from widening multiply + add + shift + narrow.
#[cfg(target_arch = "wasm32")]
#[arcane]
fn avg_8bpc_wasm128(
    _token: Wasm128Token,
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
) {
    let w = w as usize;
    let h = h as usize;

    let round_const = i32x4_splat(16384);
    let pw_1024 = i16x8_splat(PW_1024);
    let zero = i16x8_splat(0);

    for row in 0..h {
        let tmp1_row = &tmp1[row * w..][..w];
        let tmp2_row = &tmp2[row * w..][..w];
        let dst_row = &mut dst[row * dst_stride..][..w];

        let mut col = 0;
        while col + 8 <= w {
            let t1 = wasm_load_128!(&tmp1_row[col..col + 8], [i16; 8]);
            let t2 = wasm_load_128!(&tmp2_row[col..col + 8], [i16; 8]);

            let sum = i16x8_add(t1, t2);

            // Synthesize pmulhrsw(sum, 1024): (sum * 1024 + 16384) >> 15
            let prod_lo = i32x4_extmul_low_i16x8(sum, pw_1024);
            let prod_hi = i32x4_extmul_high_i16x8(sum, pw_1024);
            let rounded_lo = i32x4_add(prod_lo, round_const);
            let rounded_hi = i32x4_add(prod_hi, round_const);
            let shifted_lo = i32x4_shr(rounded_lo, 15);
            let shifted_hi = i32x4_shr(rounded_hi, 15);

            // Narrow i32→i16→u8
            let narrowed = i16x8_narrow_i32x4(shifted_lo, shifted_hi);
            let packed = u8x16_narrow_i16x8(narrowed, zero);

            // Store low 8 bytes
            let val = i64x2_extract_lane::<0>(packed);
            dst_row[col..col + 8].copy_from_slice(&val.to_ne_bytes());

            col += 8;
        }

        // Scalar tail
        while col < w {
            let sum = tmp1_row[col].wrapping_add(tmp2_row[col]);
            let avg = ((sum as i32 * 1024 + 16384) >> 15).clamp(0, 255) as u8;
            dst_row[col] = avg;
            col += 1;
        }
    }
}

/// AVG operation for 16-bit pixels using wasm128
///
/// Processes 4 pixels per iteration using i32x4 arithmetic.
#[cfg(target_arch = "wasm32")]
#[arcane]
fn avg_16bpc_wasm128(
    _token: Wasm128Token,
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    bitdepth_max: i32,
) {
    let w = w as usize;
    let h = h as usize;

    let intermediate_bits = if (bitdepth_max >> 11) != 0 {
        2i32
    } else {
        4i32
    };
    let sh = intermediate_bits + 1;
    let rnd = (1 << intermediate_bits) + 8192 * 2;
    let max = bitdepth_max;

    let rnd_vec = i32x4_splat(rnd);
    let zero = i32x4_splat(0);
    let max_vec = i32x4_splat(max);

    for row in 0..h {
        let tmp1_row = &tmp1[row * w..][..w];
        let tmp2_row = &tmp2[row * w..][..w];
        let dst_row_bytes = &mut dst[row * dst_stride..][..w * 2];
        let dst_row: &mut [u16] = zerocopy::FromBytes::mut_from_bytes(dst_row_bytes).unwrap();

        let mut col = 0usize;

        // Process 4 pixels at a time
        while col + 4 <= w {
            // Sign-extend 4 x i16 → i32x4
            let t1_i16: [i16; 4] = tmp1_row[col..col + 4].try_into().unwrap();
            let t2_i16: [i16; 4] = tmp2_row[col..col + 4].try_into().unwrap();
            let t1 = i32x4(
                t1_i16[0] as i32,
                t1_i16[1] as i32,
                t1_i16[2] as i32,
                t1_i16[3] as i32,
            );
            let t2 = i32x4(
                t2_i16[0] as i32,
                t2_i16[1] as i32,
                t2_i16[2] as i32,
                t2_i16[3] as i32,
            );

            let sum = i32x4_add(i32x4_add(t1, t2), rnd_vec);

            let result = if sh == 3 {
                i32x4_shr(sum, 3)
            } else {
                i32x4_shr(sum, 5)
            };

            let clamped = i32x4_min(i32x4_max(result, zero), max_vec);

            // Extract and store as u16
            dst_row[col] = i32x4_extract_lane::<0>(clamped) as u16;
            dst_row[col + 1] = i32x4_extract_lane::<1>(clamped) as u16;
            dst_row[col + 2] = i32x4_extract_lane::<2>(clamped) as u16;
            dst_row[col + 3] = i32x4_extract_lane::<3>(clamped) as u16;

            col += 4;
        }

        while col < w {
            let sum = tmp1_row[col] as i32 + tmp2_row[col] as i32;
            let val = ((sum + rnd) >> sh).clamp(0, max) as u16;
            dst_row[col] = val;
            col += 1;
        }
    }
}

// ============================================================
// W_AVG (weighted average)
// ============================================================

/// Weighted average for 8-bit pixels using wasm128
///
/// Processes 8 pixels per iteration. Synthesizes pmulhi and pmulhrsw.
#[cfg(target_arch = "wasm32")]
#[arcane]
fn w_avg_8bpc_wasm128(
    _token: Wasm128Token,
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    weight: i32,
) {
    let w = w as usize;
    let h = h as usize;

    let (tmp1_ptr, tmp2_ptr, weight_scaled) = if weight > 7 {
        (tmp1, tmp2, ((weight - 16) << 12) as i16)
    } else {
        (tmp2, tmp1, ((-weight) << 12) as i16)
    };

    let weight_vec = i16x8_splat(weight_scaled);
    let zero = i16x8_splat(0);
    let round_const = i32x4_splat(16384);
    let pw_2048 = i16x8_splat(PW_2048);

    for row in 0..h {
        let tmp1_row = &tmp1_ptr[row * w..][..w];
        let tmp2_row = &tmp2_ptr[row * w..][..w];
        let dst_row = &mut dst[row * dst_stride..][..w];

        let mut col = 0;
        while col + 8 <= w {
            let t1 = wasm_load_128!(&tmp1_row[col..col + 8], [i16; 8]);
            let t2 = wasm_load_128!(&tmp2_row[col..col + 8], [i16; 8]);

            // diff = tmp1 - tmp2
            let diff = i16x8_sub(t1, t2);

            // Synthesize pmulhi(diff, weight): high 16 bits of diff * weight
            // = (diff * weight) >> 16
            let prod_lo = i32x4_extmul_low_i16x8(diff, weight_vec);
            let prod_hi = i32x4_extmul_high_i16x8(diff, weight_vec);
            let shifted_lo = i32x4_shr(prod_lo, 16);
            let shifted_hi = i32x4_shr(prod_hi, 16);
            let scaled = i16x8_narrow_i32x4(shifted_lo, shifted_hi);

            // sum = tmp1 + scaled
            let sum = i16x8_add(t1, scaled);

            // Synthesize pmulhrsw(sum, 2048): (sum * 2048 + 16384) >> 15
            let rnd_prod_lo = i32x4_extmul_low_i16x8(sum, pw_2048);
            let rnd_prod_hi = i32x4_extmul_high_i16x8(sum, pw_2048);
            let rnd_lo = i32x4_add(rnd_prod_lo, round_const);
            let rnd_hi = i32x4_add(rnd_prod_hi, round_const);
            let avg_lo = i32x4_shr(rnd_lo, 15);
            let avg_hi = i32x4_shr(rnd_hi, 15);
            let avg = i16x8_narrow_i32x4(avg_lo, avg_hi);

            // Pack to u8
            let packed = u8x16_narrow_i16x8(avg, zero);
            let val = i64x2_extract_lane::<0>(packed);
            dst_row[col..col + 8].copy_from_slice(&val.to_ne_bytes());

            col += 8;
        }

        // Scalar tail
        while col < w {
            let a = tmp1_row[col] as i32;
            let b = tmp2_row[col] as i32;
            let diff = a - b;
            let scaled = (diff * (weight_scaled as i32)) >> 16;
            let sum = a + scaled;
            let avg = ((sum + 8) >> 4).clamp(0, 255) as u8;
            dst_row[col] = avg;
            col += 1;
        }
    }
}

/// Weighted average for 16-bit pixels using wasm128
///
/// Processes 4 pixels per iteration with i32x4 arithmetic.
#[cfg(target_arch = "wasm32")]
#[arcane]
fn w_avg_16bpc_wasm128(
    _token: Wasm128Token,
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    weight: i32,
    bitdepth_max: i32,
) {
    let w = w as usize;
    let h = h as usize;

    let intermediate_bits = if (bitdepth_max >> 11) != 0 {
        2i32
    } else {
        4i32
    };
    let sh = intermediate_bits + 4;
    let rnd = (8 << intermediate_bits) + 8192 * 16;
    let max = bitdepth_max;
    let inv_weight = 16 - weight;

    let rnd_vec = i32x4_splat(rnd);
    let zero = i32x4_splat(0);
    let max_vec = i32x4_splat(max);
    let weight_vec = i32x4_splat(weight);
    let inv_weight_vec = i32x4_splat(inv_weight);

    for row in 0..h {
        let tmp1_row = &tmp1[row * w..][..w];
        let tmp2_row = &tmp2[row * w..][..w];
        let dst_row_bytes = &mut dst[row * dst_stride..][..w * 2];
        let dst_row: &mut [u16] = zerocopy::FromBytes::mut_from_bytes(dst_row_bytes).unwrap();

        let mut col = 0usize;

        while col + 4 <= w {
            let t1_i16: [i16; 4] = tmp1_row[col..col + 4].try_into().unwrap();
            let t2_i16: [i16; 4] = tmp2_row[col..col + 4].try_into().unwrap();
            let t1 = i32x4(
                t1_i16[0] as i32,
                t1_i16[1] as i32,
                t1_i16[2] as i32,
                t1_i16[3] as i32,
            );
            let t2 = i32x4(
                t2_i16[0] as i32,
                t2_i16[1] as i32,
                t2_i16[2] as i32,
                t2_i16[3] as i32,
            );

            // val = a * weight + b * inv_weight + rnd
            let term1 = i32x4_mul(t1, weight_vec);
            let term2 = i32x4_mul(t2, inv_weight_vec);
            let sum = i32x4_add(i32x4_add(term1, term2), rnd_vec);

            let result = if sh == 6 {
                i32x4_shr(sum, 6)
            } else {
                i32x4_shr(sum, 8)
            };

            let clamped = i32x4_min(i32x4_max(result, zero), max_vec);

            dst_row[col] = i32x4_extract_lane::<0>(clamped) as u16;
            dst_row[col + 1] = i32x4_extract_lane::<1>(clamped) as u16;
            dst_row[col + 2] = i32x4_extract_lane::<2>(clamped) as u16;
            dst_row[col + 3] = i32x4_extract_lane::<3>(clamped) as u16;

            col += 4;
        }

        while col < w {
            let a = tmp1_row[col] as i32;
            let b = tmp2_row[col] as i32;
            let val = (a * weight + b * inv_weight + rnd) >> sh;
            dst_row[col] = val.clamp(0, max) as u16;
            col += 1;
        }
    }
}

// ============================================================
// MASK (mask-weighted blend)
// ============================================================

/// Mask blend for 8-bit pixels using wasm128
///
/// Processes 4 pixels per iteration (needs i32 for overflow safety).
/// Formula: dst = (tmp1 * mask + tmp2 * (64 - mask) + 512) >> 10
#[cfg(target_arch = "wasm32")]
#[arcane]
fn mask_8bpc_wasm128(
    _token: Wasm128Token,
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask: &[u8],
) {
    let w = w as usize;
    let h = h as usize;

    let rnd = i32x4_splat(512);
    let sixty_four = i32x4_splat(64);
    let zero = i32x4_splat(0);
    let max_255 = i32x4_splat(255);

    for row in 0..h {
        let tmp1_row = &tmp1[row * w..][..w];
        let tmp2_row = &tmp2[row * w..][..w];
        let mask_row = &mask[row * w..][..w];
        let dst_row = &mut dst[row * dst_stride..][..w];

        let mut col = 0usize;

        // Process 4 pixels at a time (i32x4 for overflow safety)
        while col + 4 <= w {
            let t1 = i32x4(
                tmp1_row[col] as i32,
                tmp1_row[col + 1] as i32,
                tmp1_row[col + 2] as i32,
                tmp1_row[col + 3] as i32,
            );
            let t2 = i32x4(
                tmp2_row[col] as i32,
                tmp2_row[col + 1] as i32,
                tmp2_row[col + 2] as i32,
                tmp2_row[col + 3] as i32,
            );
            let m = i32x4(
                mask_row[col] as i32,
                mask_row[col + 1] as i32,
                mask_row[col + 2] as i32,
                mask_row[col + 3] as i32,
            );

            let inv_m = i32x4_sub(sixty_four, m);
            let term1 = i32x4_mul(t1, m);
            let term2 = i32x4_mul(t2, inv_m);
            let sum = i32x4_add(i32x4_add(term1, term2), rnd);
            let result = i32x4_shr(sum, 10);
            let clamped = i32x4_min(i32x4_max(result, zero), max_255);

            dst_row[col] = i32x4_extract_lane::<0>(clamped) as u8;
            dst_row[col + 1] = i32x4_extract_lane::<1>(clamped) as u8;
            dst_row[col + 2] = i32x4_extract_lane::<2>(clamped) as u8;
            dst_row[col + 3] = i32x4_extract_lane::<3>(clamped) as u8;

            col += 4;
        }

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

/// Mask blend for 16-bit pixels using wasm128
///
/// Processes 4 pixels per iteration with i32x4 arithmetic.
#[cfg(target_arch = "wasm32")]
#[arcane]
fn mask_16bpc_wasm128(
    _token: Wasm128Token,
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask: &[u8],
    bitdepth_max: i32,
) {
    let w = w as usize;
    let h = h as usize;
    let max = bitdepth_max;

    let intermediate_bits = if (bitdepth_max >> 11) != 0 {
        2i32
    } else {
        4i32
    };
    let sh = intermediate_bits + 6;
    let rnd_val = (32 << intermediate_bits) + 8192 * 64;

    let rnd_vec = i32x4_splat(rnd_val);
    let zero = i32x4_splat(0);
    let max_vec = i32x4_splat(max);
    let sixty_four = i32x4_splat(64);

    for row in 0..h {
        let tmp1_row = &tmp1[row * w..][..w];
        let tmp2_row = &tmp2[row * w..][..w];
        let mask_row = &mask[row * w..][..w];
        let dst_row_bytes = &mut dst[row * dst_stride..][..w * 2];
        let dst_row: &mut [u16] = zerocopy::FromBytes::mut_from_bytes(dst_row_bytes).unwrap();

        let mut col = 0usize;

        while col + 4 <= w {
            let t1 = i32x4(
                tmp1_row[col] as i32,
                tmp1_row[col + 1] as i32,
                tmp1_row[col + 2] as i32,
                tmp1_row[col + 3] as i32,
            );
            let t2 = i32x4(
                tmp2_row[col] as i32,
                tmp2_row[col + 1] as i32,
                tmp2_row[col + 2] as i32,
                tmp2_row[col + 3] as i32,
            );
            let m = i32x4(
                mask_row[col] as i32,
                mask_row[col + 1] as i32,
                mask_row[col + 2] as i32,
                mask_row[col + 3] as i32,
            );

            let inv_m = i32x4_sub(sixty_four, m);
            let term1 = i32x4_mul(t1, m);
            let term2 = i32x4_mul(t2, inv_m);
            let sum = i32x4_add(i32x4_add(term1, term2), rnd_vec);

            let result = if sh == 8 {
                i32x4_shr(sum, 8)
            } else {
                i32x4_shr(sum, 10)
            };

            let clamped = i32x4_min(i32x4_max(result, zero), max_vec);

            dst_row[col] = i32x4_extract_lane::<0>(clamped) as u16;
            dst_row[col + 1] = i32x4_extract_lane::<1>(clamped) as u16;
            dst_row[col + 2] = i32x4_extract_lane::<2>(clamped) as u16;
            dst_row[col + 3] = i32x4_extract_lane::<3>(clamped) as u16;

            col += 4;
        }

        while col < w {
            let a = tmp1_row[col] as i32;
            let b = tmp2_row[col] as i32;
            let m = mask_row[col] as i32;
            let val = (a * m + b * (64 - m) + rnd_val) >> sh;
            dst_row[col] = val.clamp(0, max) as u16;
            col += 1;
        }
    }
}

// ============================================================
// Dispatch functions
// ============================================================

#[cfg(target_arch = "wasm32")]
pub fn avg_dispatch<BD: BitDepth>(
    dst: PicOffset,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    bd: BD,
) -> bool {
    let pixel_size = std::mem::size_of::<BD::Pixel>();
    // Tile-threading-safe block view: contiguous when threading is off, a
    // compact per-row copy when it is on. See `WithOffset::block_mut`.
    let mut block = dst.block_mut::<BD>(w as usize, h as usize);
    let dst_stride = block.byte_stride();
    let dst_offset = block.base() * pixel_size;
    let dst_bytes = block.as_mut_bytes();
    avg_dispatch_inner::<BD>(dst_bytes, dst_offset, dst_stride, tmp1, tmp2, w, h, bd)
}

/// Inner avg dispatch — operates on pre-acquired byte slice.
/// Can be called directly with scheduler-owned guards.
#[cfg(target_arch = "wasm32")]
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
    let Some(token) = crate::src::cpu::summon_wasm128() else {
        return false;
    };
    let bd_c = bd.into_c();
    match BD::BPC {
        BPC::BPC8 => avg_8bpc_wasm128(
            token,
            &mut dst_bytes[dst_offset..],
            dst_stride as usize,
            tmp1,
            tmp2,
            w,
            h,
        ),
        BPC::BPC16 => avg_16bpc_wasm128(
            token,
            &mut dst_bytes[dst_offset..],
            dst_stride as usize,
            tmp1,
            tmp2,
            w,
            h,
            bd_c,
        ),
    }
    true
}

#[cfg(target_arch = "wasm32")]
pub fn w_avg_dispatch<BD: BitDepth>(
    dst: PicOffset,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    weight: i32,
    bd: BD,
) -> bool {
    let pixel_size = std::mem::size_of::<BD::Pixel>();
    // Tile-threading-safe block view: contiguous when threading is off, a
    // compact per-row copy when it is on. See `WithOffset::block_mut`.
    let mut block = dst.block_mut::<BD>(w as usize, h as usize);
    let dst_stride = block.byte_stride();
    let dst_offset = block.base() * pixel_size;
    let dst_bytes = block.as_mut_bytes();
    w_avg_dispatch_inner::<BD>(
        dst_bytes, dst_offset, dst_stride, tmp1, tmp2, w, h, weight, bd,
    )
}

/// Inner w_avg dispatch — operates on pre-acquired byte slice.
/// Can be called directly with scheduler-owned guards.
#[cfg(target_arch = "wasm32")]
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
    let Some(token) = crate::src::cpu::summon_wasm128() else {
        return false;
    };
    let bd_c = bd.into_c();
    match BD::BPC {
        BPC::BPC8 => w_avg_8bpc_wasm128(
            token,
            &mut dst_bytes[dst_offset..],
            dst_stride as usize,
            tmp1,
            tmp2,
            w,
            h,
            weight,
        ),
        BPC::BPC16 => w_avg_16bpc_wasm128(
            token,
            &mut dst_bytes[dst_offset..],
            dst_stride as usize,
            tmp1,
            tmp2,
            w,
            h,
            weight,
            bd_c,
        ),
    }
    true
}

#[cfg(target_arch = "wasm32")]
pub fn mask_dispatch<BD: BitDepth>(
    dst: PicOffset,
    tmp1: &[i16; COMPINTER_LEN],
    tmp2: &[i16; COMPINTER_LEN],
    w: i32,
    h: i32,
    mask: &[u8],
    bd: BD,
) -> bool {
    let pixel_size = std::mem::size_of::<BD::Pixel>();
    // Tile-threading-safe block view: contiguous when threading is off, a
    // compact per-row copy when it is on. See `WithOffset::block_mut`.
    let mut block = dst.block_mut::<BD>(w as usize, h as usize);
    let dst_stride = block.byte_stride();
    let dst_offset = block.base() * pixel_size;
    let dst_bytes = block.as_mut_bytes();
    mask_dispatch_inner::<BD>(
        dst_bytes, dst_offset, dst_stride, tmp1, tmp2, w, h, mask, bd,
    )
}

/// Inner mask dispatch — operates on pre-acquired byte slice.
/// Can be called directly with scheduler-owned guards.
#[cfg(target_arch = "wasm32")]
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
    let Some(token) = crate::src::cpu::summon_wasm128() else {
        return false;
    };
    let bd_c = bd.into_c();
    match BD::BPC {
        BPC::BPC8 => mask_8bpc_wasm128(
            token,
            &mut dst_bytes[dst_offset..],
            dst_stride as usize,
            tmp1,
            tmp2,
            w,
            h,
            mask,
        ),
        BPC::BPC16 => mask_16bpc_wasm128(
            token,
            &mut dst_bytes[dst_offset..],
            dst_stride as usize,
            tmp1,
            tmp2,
            w,
            h,
            mask,
            bd_c,
        ),
    }
    true
}
