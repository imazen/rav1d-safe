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

use rav1d_disjoint_mut::site_class;
use rav1d_disjoint_mut::DisjointMut;
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

fn overlapping_borrow_panics() -> bool {
    let v: DisjointMut<Vec<u8>> = DisjointMut::new(vec![0u8; 4096]);
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
    let tracked = overlapping_borrow_panics();

    site_class::set_null_mask(1 << site_class::OTHER);
    let nulled = overlapping_borrow_panics();

    // A class this file does NOT belong to must leave the check in place —
    // otherwise the mask is a global off-switch wearing a class label, and
    // every per-class number the campaign reports would be the same number.
    site_class::set_null_mask(1 << site_class::RECON);
    let other_class_nulled = overlapping_borrow_panics();

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
}
