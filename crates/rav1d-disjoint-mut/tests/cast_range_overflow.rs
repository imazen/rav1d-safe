//! Byte-range translation for the zerocopy cast API must refuse overflow.
//!
//! Found by an independent review: `TranslateRange::mul` used unchecked `*`, so in
//! RELEASE builds `slice_as::<u32>((usize::MAX/4+1)..(usize::MAX/4+2))` wrapped to
//! bytes `0..4` and returned a view of element 0 instead of panicking (no UB, but
//! a silently wrong view from a safe API). Debug builds already panicked.
use rav1d_disjoint_mut::DisjointMut;
use std::panic::AssertUnwindSafe;
use std::panic::catch_unwind;

fn panics(f: impl FnOnce()) -> bool {
    catch_unwind(AssertUnwindSafe(f)).is_err()
}

#[test]
fn huge_element_ranges_panic_instead_of_wrapping() {
    let dm = DisjointMut::new(vec![0u8; 64]);
    let big = usize::MAX / 4 + 1;
    assert!(panics(|| drop(dm.slice_as::<_, u32>(big..big + 1))));
    assert!(panics(|| drop(dm.mut_slice_as::<_, u32>(big..big + 1))));
    assert!(panics(|| drop(dm.slice_as::<_, u32>(big..))));
    assert!(panics(|| drop(dm.slice_as::<_, u32>(..big))));
    assert!(panics(|| drop(dm.slice_as::<_, u32>(..=usize::MAX))));
    assert!(panics(|| drop(dm.slice_as::<_, u32>(0..=usize::MAX))));
    assert!(panics(|| drop(dm.element_as::<u32>(usize::MAX))));
    assert!(panics(|| drop(dm.mut_element_as::<u32>(usize::MAX))));
    assert!(panics(|| drop(dm.element_as::<u32>(big))));
}

#[test]
fn ordinary_element_ranges_still_work() {
    let dm = DisjointMut::new(vec![0u8; 64]);
    assert_eq!(dm.slice_as::<_, u32>(0..16).len(), 16);
    assert_eq!(dm.slice_as::<_, u32>(3..=5).len(), 3);
    assert_eq!(dm.slice_as::<_, u32>(..4).len(), 4);
    assert_eq!(dm.slice_as::<_, u32>(..=3).len(), 4);
    assert_eq!(dm.slice_as::<_, u32>(10..).len(), 6);
    drop(dm.element_as::<u32>(15));
    assert!(panics(|| drop(dm.element_as::<u32>(16)))); // one past the end
}
