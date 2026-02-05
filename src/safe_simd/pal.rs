#![allow(unsafe_op_in_unsafe_fn)]

//! Safe SIMD implementation of pal_idx_finish using AVX2.
//!
//! Packs pairs of palette indices (4-bit each) into single bytes:
//! dst[x] = src[2*x] | (src[2*x+1] << 4)

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::ffi::c_int;

/// AVX2 implementation of pal_idx_finish.
///
/// Packs pairs of bytes into nibbles using pmaddubsw trick:
/// Since src[2k] | (src[2k+1] << 4) == src[2k] * 1 + src[2k+1] * 16
/// (when values are 0..15 and thus nibbles don't overlap),
/// we can use _mm256_maddubs_epi16 with coefficients [1, 16, 1, 16, ...]
/// to compute this in parallel.
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
    let dst_w = w / 2;

    // Coefficients for pmaddubsw: multiplies even bytes by 1, odd bytes by 16
    // This computes src[2k]*1 + src[2k+1]*16 = src[2k] | (src[2k+1] << 4)
    let coeff = _mm256_set1_epi16(0x1001_u16 as i16); // bytes: [1, 16, 1, 16, ...]
    let coeff128 = _mm_set1_epi16(0x1001_u16 as i16);

    // Process visible rows
    for y in 0..h {
        let src_row = src.add(y * bw);
        let dst_row = dst.add(y * dst_bw);
        let mut x = 0usize;

        // Process 64 source bytes → 32 dst bytes at a time (AVX2)
        while x + 32 <= dst_w {
            let a = _mm256_loadu_si256(src_row.add(x * 2) as *const __m256i);
            let b = _mm256_loadu_si256(src_row.add(x * 2 + 32) as *const __m256i);
            let a16 = _mm256_maddubs_epi16(a, coeff);
            let b16 = _mm256_maddubs_epi16(b, coeff);
            let packed = _mm256_packus_epi16(a16, b16);
            // packus interleaves across lanes: [a_lo, b_lo, a_hi, b_hi]
            // permute to get contiguous: [a_lo, a_hi, b_lo, b_hi]
            let packed = _mm256_permute4x64_epi64::<0xD8>(packed);
            _mm256_storeu_si256(dst_row.add(x) as *mut __m256i, packed);
            x += 32;
        }

        // Process 32 source bytes → 16 dst bytes (AVX2, pack with zeros)
        if x + 16 <= dst_w {
            let a = _mm256_loadu_si256(src_row.add(x * 2) as *const __m256i);
            let a16 = _mm256_maddubs_epi16(a, coeff);
            let zero = _mm256_setzero_si256();
            let packed = _mm256_packus_epi16(a16, zero);
            // Result: [a_lo(8 bytes), zeros(8), a_hi(8 bytes), zeros(8)]
            let packed = _mm256_permute4x64_epi64::<0xD8>(packed);
            // Now: [a_lo(8), a_hi(8), zeros(16)]
            // Store lower 16 bytes
            _mm_storeu_si128(
                dst_row.add(x) as *mut __m128i,
                _mm256_castsi256_si128(packed),
            );
            x += 16;
        }

        // Process 16 source bytes → 8 dst bytes (SSE)
        if x + 8 <= dst_w {
            let a = _mm_loadu_si128(src_row.add(x * 2) as *const __m128i);
            let a16 = _mm_maddubs_epi16(a, coeff128);
            let zero = _mm_setzero_si128();
            let packed = _mm_packus_epi16(a16, zero);
            // Store lower 8 bytes
            _mm_storel_epi64(dst_row.add(x) as *mut __m128i, packed);
            x += 8;
        }

        // Process 8 source bytes → 4 dst bytes (SSE, partial)
        if x + 4 <= dst_w {
            let a = _mm_loadl_epi64(src_row.add(x * 2) as *const __m128i);
            let a16 = _mm_maddubs_epi16(a, coeff128);
            let zero = _mm_setzero_si128();
            let packed = _mm_packus_epi16(a16, zero);
            // Store lower 4 bytes
            let val = _mm_cvtsi128_si32(packed) as u32;
            *(dst_row.add(x) as *mut u32) = val;
            x += 4;
        }

        // Remaining pairs (for w not divisible by 8)
        while x < dst_w {
            *dst_row.add(x) = *src_row.add(x * 2) | (*src_row.add(x * 2 + 1) << 4);
            x += 1;
        }

        // Fill invisible columns with repeated last visible pixel
        if dst_w < dst_bw {
            let fill_val = {
                let last_src = *src_row.add(w);
                0x11u8.wrapping_mul(last_src)
            };
            let fill_start = dst_row.add(dst_w);
            let fill_len = dst_bw - dst_w;

            if fill_len >= 32 {
                let fill_vec = _mm256_set1_epi8(fill_val as i8);
                let mut i = 0;
                while i + 32 <= fill_len {
                    _mm256_storeu_si256(fill_start.add(i) as *mut __m256i, fill_vec);
                    i += 32;
                }
                while i < fill_len {
                    *fill_start.add(i) = fill_val;
                    i += 1;
                }
            } else {
                for i in 0..fill_len {
                    *fill_start.add(i) = fill_val;
                }
            }
        }
    }

    // Fill invisible rows by copying the last visible row
    if h < bh {
        let last_row = dst.add((h - 1) * dst_bw);
        for y in h..bh {
            let dst_row = dst.add(y * dst_bw);
            // Copy using SIMD for larger widths
            if dst_bw >= 32 {
                let mut i = 0;
                while i + 32 <= dst_bw {
                    let data = _mm256_loadu_si256(last_row.add(i) as *const __m256i);
                    _mm256_storeu_si256(dst_row.add(i) as *mut __m256i, data);
                    i += 32;
                }
                while i < dst_bw {
                    *dst_row.add(i) = *last_row.add(i);
                    i += 1;
                }
            } else {
                std::ptr::copy_nonoverlapping(last_row, dst_row, dst_bw);
            }
        }
    }
}
