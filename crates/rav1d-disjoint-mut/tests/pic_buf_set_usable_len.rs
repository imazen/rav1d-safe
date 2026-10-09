//! `PicBuf::set_usable_len` must refuse a length whose end wraps past `usize::MAX`.
//!
//! The original check was `align_offset + usable_len <= buf.len()`, which wraps to a small
//! number in release builds, so a safe caller could authorize a usable region far larger than
//! the allocation and every later guard would trust it. Found by the independent audit of
//! 0.4.0 (finding A-01). The constructor already used `checked_add`; only the setter did not.
#![cfg(feature = "pic-buf")]
#![forbid(unsafe_code)]

use rav1d_disjoint_mut::{DisjointMut, PicBuf};
use std::panic::{AssertUnwindSafe, catch_unwind};

const STORAGE: usize = 65536;

/// A buffer whose aligned start is not the Vec's start (so `align_offset > 0`), and that offset.
fn nonzero_offset_buffer() -> (PicBuf, usize) {
    let storage = vec![0u8; STORAGE];
    let addr = storage.as_ptr() as usize;
    let alignment = 1usize << (addr.trailing_zeros() + 1);
    let offset = storage.as_ptr().align_offset(alignment);
    assert!(
        offset > 0 && offset < storage.len(),
        "need a Vec that is not already aligned"
    );
    (PicBuf::from_vec_aligned(storage, alignment, 1), offset)
}

#[test]
fn setter_rejects_wrapped_region_end() {
    let (mut buffer, offset) = nonzero_offset_buffer();
    // `offset + wrapping_len` wraps to exactly 0 if the setter does not check the addition.
    let wrapping_len = usize::MAX - offset + 1;
    let result = catch_unwind(AssertUnwindSafe(|| buffer.set_usable_len(wrapping_len)));
    assert!(
        result.is_err(),
        "safe setter accepted a region larger than its allocation"
    );
}

#[test]
fn rejected_setter_leaves_the_view_inside_the_allocation() {
    const USABLE: usize = 4096;
    let (mut buffer, offset) = nonzero_offset_buffer();
    buffer.set_usable_len(USABLE);
    let _ = catch_unwind(AssertUnwindSafe(|| {
        buffer.set_usable_len(usize::MAX - offset + 1)
    }));
    // The rejected call must not have changed the usable region.
    let dm = DisjointMut::new(buffer);
    assert_eq!(
        dm.index(USABLE - 1..USABLE)[0],
        0,
        "last usable byte is still reachable"
    );
    // Last: a refused view poisons the container, so nothing may be read after this.
    assert!(
        catch_unwind(AssertUnwindSafe(|| drop(dm.index(USABLE..USABLE + 1)))).is_err(),
        "a view one byte past the usable region must be refused"
    );
}

#[test]
fn setter_accepts_the_largest_valid_length_and_rejects_one_more() {
    let (mut buffer, offset) = nonzero_offset_buffer();
    let max = buffer.storage_len() - offset;
    buffer.set_usable_len(max);
    assert!(catch_unwind(AssertUnwindSafe(|| buffer.set_usable_len(max + 1))).is_err());
}
