//! Regression test for the `PicBuf::from_vec_aligned` arithmetic-overflow
//! memory-safety bug (commit 68ab197).
//!
//! Before the fix, `align_offset + usable_len` was an unchecked add. With a
//! non-zero `align_offset` and a `usable_len` near `usize::MAX`, that sum could
//! wrap to a small value, the bounds `assert!` would pass even though
//! `usable_len > vec.len()`, and the resulting `PicBuf` would expose a usable
//! region larger than its backing `Vec` — an out-of-bounds read/write on
//! access (reachable on 32-bit targets with crafted picture dimensions).
//!
//! The invariant under test: `from_vec_aligned` must *always reject* such
//! inputs by panicking, never silently construct an oversized region.
//!
//! Run with: `cargo test -p rav1d-disjoint-mut --features pic-buf`
#![cfg(feature = "pic-buf")]

use rav1d_disjoint_mut::PicBuf;

/// `usable_len == usize::MAX` over a small backing `Vec` must panic. Pre-fix,
/// a non-zero `align_offset` made `align_offset + usable_len` wrap and the
/// bounds assert pass, silently producing an out-of-bounds region.
#[test]
#[should_panic]
fn rejects_overflowing_usable_len() {
    // High alignment makes a non-zero align_offset likely (exercising the
    // overflow branch); either way the call must panic rather than succeed.
    let _ = PicBuf::from_vec_aligned(vec![0u8; 64], 4096, usize::MAX);
}

/// `usable_len` that merely exceeds the `Vec` (no overflow) must also panic.
#[test]
#[should_panic]
fn rejects_oversized_usable_len() {
    let _ = PicBuf::from_vec_aligned(vec![0u8; 64], 1, 65);
}

/// A correctly-sized call (backing `Vec` includes alignment slack) succeeds.
#[test]
fn accepts_valid_aligned_region() {
    // 64 usable bytes + up to 63 bytes of alignment slack fits in 128.
    let _ = PicBuf::from_vec_aligned(vec![0u8; 128], 64, 64);
}
