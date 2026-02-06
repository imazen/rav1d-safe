//! Safe SIMD implementation of pal_idx_finish using AVX2.
//!
//! Packs pairs of palette indices (4-bit each) into single bytes:
//! dst[x] = src[2*x] | (src[2*x+1] << 4)

#[cfg(target_arch = "x86_64")]
use archmage::{arcane, Desktop64, SimdToken};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
#[cfg(target_arch = "x86_64")]
use safe_unaligned_simd::x86_64 as safe_simd;
#[cfg(target_arch = "x86_64")]
use super::partial_simd;
use std::ffi::c_int;

/// Inner implementation using archmage for safe SIMD.
///
/// Packs pairs of bytes into nibbles using pmaddubsw trick:
/// Since src[2k] | (src[2k+1] << 4) == src[2k] * 1 + src[2k+1] * 16
/// (when values are 0..15 and thus nibbles don't overlap),
/// we can use _mm256_maddubs_epi16 with coefficients [1, 16, 1, 16, ...]
/// to compute this in parallel.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn pal_idx_finish_inner(
    _token: Desktop64,
    dst: &mut [u8],
    src: &[u8],
    bw: usize,
    bh: usize,
    w: usize,
    h: usize,
) {
    let dst_bw = bw / 2;
    let dst_w = w / 2;

    // Coefficients for pmaddubsw: multiplies even bytes by 1, odd bytes by 16
    // This computes src[2k]*1 + src[2k+1]*16 = src[2k] | (src[2k+1] << 4)
    let coeff = _mm256_set1_epi16(0x1001_u16 as i16); // bytes: [1, 16, 1, 16, ...]
    let coeff128 = _mm_set1_epi16(0x1001_u16 as i16);

    // Process visible rows
    for y in 0..h {
        let src_row = &src[y * bw..];
        let dst_row = &mut dst[y * dst_bw..];
        let mut x = 0usize;

        // Process 64 source bytes → 32 dst bytes at a time (AVX2)
        while x + 32 <= dst_w {
            let src_a: &[u8; 32] = src_row[x * 2..][..32].try_into().unwrap();
            let src_b: &[u8; 32] = src_row[x * 2 + 32..][..32].try_into().unwrap();
            let a = safe_simd::_mm256_loadu_si256(src_a);
            let b = safe_simd::_mm256_loadu_si256(src_b);
            let a16 = _mm256_maddubs_epi16(a, coeff);
            let b16 = _mm256_maddubs_epi16(b, coeff);
            let packed = _mm256_packus_epi16(a16, b16);
            // packus interleaves across lanes: [a_lo, b_lo, a_hi, b_hi]
            // permute to get contiguous: [a_lo, a_hi, b_lo, b_hi]
            let packed = _mm256_permute4x64_epi64::<0xD8>(packed);
            let dst_arr: &mut [u8; 32] = (&mut dst_row[x..x + 32]).try_into().unwrap();
            safe_simd::_mm256_storeu_si256(dst_arr, packed);
            x += 32;
        }

        // Process 32 source bytes → 16 dst bytes (AVX2, pack with zeros)
        if x + 16 <= dst_w {
            let src_a: &[u8; 32] = src_row[x * 2..][..32].try_into().unwrap();
            let a = safe_simd::_mm256_loadu_si256(src_a);
            let a16 = _mm256_maddubs_epi16(a, coeff);
            let zero = _mm256_setzero_si256();
            let packed = _mm256_packus_epi16(a16, zero);
            // Result: [a_lo(8 bytes), zeros(8), a_hi(8 bytes), zeros(8)]
            let packed = _mm256_permute4x64_epi64::<0xD8>(packed);
            // Now: [a_lo(8), a_hi(8), zeros(16)]
            // Store lower 16 bytes
            let dst_arr: &mut [u8; 16] = (&mut dst_row[x..x + 16]).try_into().unwrap();
            safe_simd::_mm_storeu_si128(dst_arr, _mm256_castsi256_si128(packed));
            x += 16;
        }

        // Process 16 source bytes → 8 dst bytes (SSE)
        if x + 8 <= dst_w {
            let src_a: &[u8; 16] = src_row[x * 2..][..16].try_into().unwrap();
            let a = safe_simd::_mm_loadu_si128(src_a);
            let a16 = _mm_maddubs_epi16(a, coeff128);
            let zero = _mm_setzero_si128();
            let packed = _mm_packus_epi16(a16, zero);
            // Store lower 8 bytes using safe partial_simd wrapper
            let dst_arr: &mut [u8; 8] = (&mut dst_row[x..x + 8]).try_into().unwrap();
            partial_simd::mm_storel_epi64(dst_arr, packed);
            x += 8;
        }

        // Process 8 source bytes → 4 dst bytes (SSE, partial)
        if x + 4 <= dst_w {
            // Load 8 bytes using safe partial_simd wrapper
            let src_a: &[u8; 8] = src_row[x * 2..][..8].try_into().unwrap();
            let a = partial_simd::mm_loadl_epi64(src_a);
            let a16 = _mm_maddubs_epi16(a, coeff128);
            let zero = _mm_setzero_si128();
            let packed = _mm_packus_epi16(a16, zero);
            // Store lower 4 bytes
            let val = _mm_cvtsi128_si32(packed) as u32;
            dst_row[x..x + 4].copy_from_slice(&val.to_ne_bytes());
            x += 4;
        }

        // Remaining pairs (for w not divisible by 8)
        while x < dst_w {
            dst_row[x] = src_row[x * 2] | (src_row[x * 2 + 1] << 4);
            x += 1;
        }

        // Fill invisible columns with repeated last visible pixel
        if dst_w < dst_bw {
            let fill_val = {
                let last_src = src_row[w];
                0x11u8.wrapping_mul(last_src)
            };
            let fill_slice = &mut dst_row[dst_w..dst_bw];
            let fill_len = fill_slice.len();

            if fill_len >= 32 {
                let fill_vec = _mm256_set1_epi8(fill_val as i8);
                let mut i = 0;
                while i + 32 <= fill_len {
                    let dst_arr: &mut [u8; 32] = (&mut fill_slice[i..i + 32]).try_into().unwrap();
                    safe_simd::_mm256_storeu_si256(dst_arr, fill_vec);
                    i += 32;
                }
                while i < fill_len {
                    fill_slice[i] = fill_val;
                    i += 1;
                }
            } else {
                fill_slice.fill(fill_val);
            }
        }
    }

    // Fill invisible rows by copying the last visible row
    if h < bh {
        // Copy last visible row data first
        let last_row_data: Vec<u8> = dst[(h - 1) * dst_bw..h * dst_bw].to_vec();
        for y in h..bh {
            dst[y * dst_bw..(y + 1) * dst_bw].copy_from_slice(&last_row_data);
        }
    }
}

/// Safe dispatch for pal_idx_finish.
///
/// dst and src must NOT alias. Returns true if SIMD was used.
#[cfg(target_arch = "x86_64")]
pub fn pal_idx_finish_dispatch(
    dst: &mut [u8],
    src: &[u8],
    bw: usize,
    bh: usize,
    w: usize,
    h: usize,
) -> bool {
    if let Some(token) = Desktop64::summon() {
        pal_idx_finish_inner(token, dst, src, bw, bh, w, h);
        true
    } else {
        false
    }
}

/// AVX2 implementation of pal_idx_finish - FFI wrapper.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn pal_idx_finish_avx2(
    dst: *mut u8,
    src: *const u8,
    bw: c_int,
    bh: c_int,
    w: c_int,
    h: c_int,
) {
    let bw = bw as usize;
    let bh = bh as usize;
    let w = w as usize;
    let h = h as usize;
    let dst_bw = bw / 2;

    // SAFETY: Caller guarantees valid pointers with sufficient size
    let dst_slice = unsafe { std::slice::from_raw_parts_mut(dst, dst_bw * bh) };
    let src_slice = unsafe { std::slice::from_raw_parts(src, bw * bh) };

    // SAFETY: We're in an AVX2 function, so the token is valid
    let token = unsafe { Desktop64::forge_token_dangerously() };

    pal_idx_finish_inner(token, dst_slice, src_slice, bw, bh, w, h);
}
