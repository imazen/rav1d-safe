//! Safe ARM NEON implementation of refmvs functions.
//!
//! splat_mv: Fills rows of RefMvsBlock arrays with a single value.
//! Uses 16-byte NEON stores for the 12-byte RefMvsBlock struct (with R_PAD overflow).

#![cfg_attr(not(feature = "unchecked"), forbid(unsafe_code))]
#![cfg_attr(feature = "unchecked", deny(unsafe_code))]

#[cfg(all(feature = "asm", target_arch = "aarch64"))]
use core::arch::aarch64::*;

#[cfg(feature = "asm")]
use crate::src::align::Align16;
#[cfg(feature = "asm")]
use crate::src::refmvs::RefMvsBlock;

/// ARM NEON implementation of splat_mv.
///
/// Fills bh4 rows of RefMvsBlock arrays with the same value.
/// Each row pointer rr[y] points to an array, and we fill rr[y][bx4..bx4+bw4].
///
/// RefMvsBlock is 12 bytes. We use 16-byte stores at stride 12,
/// which safely overwrites 4 bytes into the next element (or padding at end).
#[cfg(all(feature = "asm", target_arch = "aarch64"))]
#[allow(unsafe_code)]
pub unsafe extern "C" fn splat_mv_neon(
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
    let rmv_ptr = rmv as *const Align16<RefMvsBlock> as *const u8;
    let val128 = unsafe { vld1q_u8(rmv_ptr) };

    for y in 0..bh4 {
        let row = unsafe { *rr.add(y) };
        if row.is_null() {
            continue;
        }
        let base = unsafe { (row as *mut u8).add(bx4 * 12) };

        // Each RefMvsBlock is 12 bytes. Store 16 bytes at stride 12.
        let mut i = 0;
        while i < bw4 {
            unsafe { vst1q_u8(base.add(i * 12), val128) };
            i += 1;
        }
    }
}
