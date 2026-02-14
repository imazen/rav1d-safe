//! Targeted Miri test for Aligned<A, [V; N]> with concurrent mutable guards.
//!
//! The ExternalAsMutPtr blanket's default as_mut_slice calls (*ptr).len()
//! which creates &Aligned<A, [V; N]> — a SharedReadOnly covering inline data.
//! Under Stacked Borrows, this invalidates concurrent &mut guards.
//!
//! Run: cargo +nightly miri test --test aligned_miri --features aligned

#![cfg(feature = "aligned")]

use rav1d_disjoint_mut::DisjointMut;

/// Two disjoint mutable guards on an Aligned array, used concurrently.
/// This triggers the as_mut_slice → &Aligned bug under Stacked Borrows.
#[test]
fn test_aligned_two_disjoint_mut_guards() {
    let dm = DisjointMut::new(aligned::Aligned::<aligned::A16, _>([0u8; 64]));
    let mut g1 = dm.index_mut(0..32);
    let mut g2 = dm.index_mut(32..64);
    g1[0] = 1;
    g2[0] = 2;
    assert_eq!(g1[0], 1);
    assert_eq!(g2[0], 2);
}

/// Same test with immut + mut disjoint guards.
#[test]
fn test_aligned_immut_and_mut_disjoint() {
    let dm = DisjointMut::new(aligned::Aligned::<aligned::A16, _>([0u8; 64]));
    let g1 = dm.index(0..32);
    let mut g2 = dm.index_mut(32..64);
    g2[0] = 42;
    assert_eq!(g1[0], 0);
    assert_eq!(g2[0], 42);
}
