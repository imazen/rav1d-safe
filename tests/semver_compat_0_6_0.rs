//! Items that 0.6.0 published and later releases keep so that code written against 0.6.0
//! still compiles (`cargo semver-checks` against the published crate flags their removal).
//!
//! * the `unchecked` feature, now an alias of `untracked`
//! * `managed::is_unchecked()`, deprecated in favour of `is_untracked()`
//! * `include::dav1d::picture::{rect_hull_arm, cdef_double_reads}`, constant `false` since
//!   the measurement arms they reported were removed

use rav1d_safe::include::dav1d::picture::{cdef_double_reads, rect_hull_arm};
use rav1d_safe::src::managed::is_untracked;

#[test]
#[allow(deprecated)]
fn is_unchecked_is_the_untracked_flag() {
    assert_eq!(rav1d_safe::src::managed::is_unchecked(), is_untracked());
}

#[test]
fn the_unchecked_feature_implies_untracked() {
    if cfg!(feature = "unchecked") {
        assert!(cfg!(feature = "untracked") && is_untracked());
    }
}

#[test]
fn removed_measurement_arms_report_false() {
    assert!(!rect_hull_arm());
    assert!(!cdef_double_reads());
}
