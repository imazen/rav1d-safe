//! Safe SIMD implementation of refmvs functions using AVX2.
//!
//! splat_mv: Fills rows of RefMvsBlock arrays with a single value.
//! The value is 12 bytes (RefMvsBlock) stored in a 16-byte aligned wrapper.
//! Uses 16-byte stores at stride 12, with R_PAD ensuring safe overwrite.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::src::align::Align16;
use crate::src::refmvs::RefMvsBlock;

/// AVX2 implementation of splat_mv.
///
/// Fills bh4 rows of RefMvsBlock arrays with the same value.
/// Each row pointer rr[y] points to an array, and we fill rr[y][bx4..bx4+bw4].
///
/// RefMvsBlock is 12 bytes. We use 16-byte unaligned stores at stride 12,
/// which safely overwrites 4 bytes into the next element (or padding at end).
/// The R_PAD allocation padding ensures we don't write out of bounds.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn splat_mv_avx2(
    rr: *mut *mut RefMvsBlock,
    rmv: &Align16<RefMvsBlock>,
    bx4: i32,
    bw4: i32,
    bh4: i32,
) {
    let bx4 = bx4 as usize;
    let bw4 = bw4 as usize;
    let bh4 = bh4 as usize;

    // Load the 16-byte aligned value (12 bytes data + 4 bytes padding)
    let rmv_ptr = rmv as *const Align16<RefMvsBlock> as *const __m128i;
    let val128 = unsafe { _mm_loadu_si128(rmv_ptr) };

    for y in 0..bh4 {
        let row = unsafe { *rr.add(y) };
        if row.is_null() {
            continue;
        }
        let base = unsafe { (row as *mut u8).add(bx4 * 12) };

        // Each RefMvsBlock is 12 bytes. Store 16 bytes at stride 12.
        // The extra 4 bytes overlap into the next element (safe due to R_PAD).
        let mut i = 0;
        while i < bw4 {
            unsafe { _mm_storeu_si128(base.add(i * 12) as *mut __m128i, val128) };
            i += 1;
        }
    }
}
