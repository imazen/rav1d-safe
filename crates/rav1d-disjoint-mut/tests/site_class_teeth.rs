//! Teeth for the THROWAWAY `__probe_class` instrument.
//!
//! The instrument's whole claim is "this arm stops tracking exactly the class I
//! named". Nothing about a decode run can check that: the pixels are identical
//! either way (the tracker never touched them), and a wall-clock delta is what
//! is being measured, so it cannot also be the proof that the measurement is
//! live. An arm that silently nulls NOTHING would look like "this class costs
//! 0 ms", which is the exact wrong answer to the question the campaign asked.
//!
//! So the proof is a planted overlap: register two overlapping MUTABLE borrows
//! from this file. With no class nulled the tracker must panic; with this
//! file's class (`other` — a test file is not a decoder source file) nulled it
//! must not. Both directions are asserted, because only asserting the second
//! would pass against a tracker that was never checking in the first place.

#![cfg(feature = "__probe_class")]

use rav1d_disjoint_mut::DisjointMut;
use rav1d_disjoint_mut::site_class;
use std::panic::AssertUnwindSafe;

/// This test file classifies as `other`; assert that rather than assume it, so
/// a change to `class_for_file` cannot make the rest of the test vacuous.
#[test]
fn this_file_is_class_other() {
    let c = site_class::class_for_file_pub(file!());
    assert_eq!(
        c,
        site_class::OTHER,
        "expected this test file ({}) to classify as `other`, got `{}`",
        file!(),
        site_class::CLASS_NAMES[c as usize]
    );
}

/// `len` selects which side of the `SHARD_MIN_LEN` (64 KiB) size split the
/// instance lands on, so the `big` / `small` modifiers can be tested too.
fn overlapping_borrow_panics(len: usize) -> bool {
    let v: DisjointMut<Vec<u8>> = DisjointMut::new(vec![0u8; len]);
    std::panic::catch_unwind(AssertUnwindSafe(|| {
        let a = v.index_mut(0..64);
        let b = v.index_mut(32..96);
        // Keep both live across the second registration: an overlap that is
        // already dropped is legal, so holding them is what makes the hazard
        // real.
        core::hint::black_box((&a, &b));
    }))
    .is_err()
}

#[test]
fn nulling_this_class_removes_the_check_and_not_nulling_keeps_it() {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));

    site_class::set_null_mask(0);
    let tracked = overlapping_borrow_panics(4096);

    site_class::set_null_mask(1 << site_class::OTHER);
    let nulled = overlapping_borrow_panics(4096);

    // A class this file does NOT belong to must leave the check in place —
    // otherwise the mask is a global off-switch wearing a class label, and
    // every per-class number the campaign reports would be the same number.
    site_class::set_null_mask(1 << site_class::RECON);
    let other_class_nulled = overlapping_borrow_panics(4096);

    // Size modifiers: `other,big` must leave a 4 KiB instance checked and a
    // 256 KiB one unchecked, and `other,small` the reverse. Without this the
    // modifier could be a no-op and the "picture-buffer only" arm would
    // silently be the whole-class arm.
    site_class::set_null_mask((1 << site_class::OTHER) | site_class::ONLY_BIG);
    let big_arm_small_inst = overlapping_borrow_panics(4096);
    let big_arm_big_inst = overlapping_borrow_panics(256 * 1024);
    site_class::set_null_mask((1 << site_class::OTHER) | site_class::ONLY_SMALL);
    let small_arm_small_inst = overlapping_borrow_panics(4096);
    let small_arm_big_inst = overlapping_borrow_panics(256 * 1024);

    site_class::set_null_mask(0);
    std::panic::set_hook(prev);

    assert!(
        tracked,
        "tracker did not catch a planted overlap with nothing nulled — the \
         instrument's baseline arm is not checking, so no per-class delta means anything"
    );
    assert!(
        !nulled,
        "planted overlap still panicked with class `other` nulled — the arm is \
         not actually nulling, and would report this class as costing 0 ms"
    );
    assert!(
        other_class_nulled,
        "nulling `recon` also disabled a borrow from an `other` site — the mask \
         is not selective"
    );
    assert!(
        big_arm_small_inst,
        "`big` modifier nulled a sub-SHARD_MIN_LEN instance"
    );
    assert!(
        !big_arm_big_inst,
        "`big` modifier failed to null a large instance"
    );
    assert!(
        !small_arm_small_inst,
        "`small` modifier failed to null a small instance"
    );
    assert!(
        small_arm_big_inst,
        "`small` modifier nulled a large instance"
    );
}

#[test]
fn a_size_modifier_with_no_class_is_rejected() {
    // Otherwise `RAV1D_CLS_NULL=big` would be a baseline arm wearing a
    // different name and would report every class as free.
    assert!(site_class::mask_from_str("big").is_none());
    assert!(site_class::mask_from_str("recon,big").is_some());
}
