//! `probe::report` must not be able to form an invalid `&Location` from state that safe code
//! can write (audit finding A-02). Feature `__probe_count` only; internal measurement tooling.
//!
//! Before the fix `Slot::loc` was a public `AtomicUsize` that `report` turned back into a
//! reference with `unsafe`; storing `1` there made Miri diagnose a dangling `&Location`.
//! The location is now a `OnceLock<&'static Location>`: the type cannot hold an arbitrary
//! integer, so this file only needs to show the public surface still compiles and reports.
#![cfg(feature = "__probe_count")]
#![forbid(unsafe_code)]

use core::sync::atomic::Ordering::Relaxed;
use rav1d_disjoint_mut::probe;

#[test]
fn report_cannot_be_driven_into_an_invalid_reference_from_safe_code() {
    probe::NEXT_SLOT.store(1, Relaxed);
    probe::SLOTS[0].n_mut.store(1, Relaxed);
    // There is no integer to store into `loc` any more; the only writer is a real
    // `&'static Location` registered by the tracker, so an unset slot reports "?".
    assert!(probe::SLOTS[0].loc.get().is_none());
    let text = probe::report(1);
    assert!(text.contains("SLOT"), "report still renders rows: {text}");
}
