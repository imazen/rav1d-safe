//! Adversarial tests — `untracked` configuration.
//! Written by an independent reviewer; not part of the crate's own suite.
#![cfg(feature = "untracked")]

use rav1d_disjoint_mut::{DisjointMut, DisjointMutArcSlice, PicBuf};
use std::panic::{AssertUnwindSafe, catch_unwind};

fn panics(f: impl FnOnce()) -> bool {
    catch_unwind(AssertUnwindSafe(f)).is_err()
}

#[test]
fn tracking_is_compiled_out_everywhere() {
    let a = DisjointMut::new(vec![0u8; 8]);
    let b = DisjointMut::new_eager(vec![0u8; 8]);
    let c: DisjointMut<Vec<u8>> = Default::default();
    let d = DisjointMutArcSlice::try_new(8, 0u8).unwrap();
    assert!(!a.is_checked() && !b.is_checked() && !c.is_checked() && !d.is_checked());
    // overlapping mut/mut accepted (single thread, same bytes written: benign in practice)
    let mut g1 = a.index_mut(0..8);
    let mut g2 = a.index_mut(0..8);
    g1[0] = 1;
    g2[0] = 1;
}

#[test]
fn oob_every_index_shape_still_panics_and_nothing_poisons() {
    macro_rules! case {
        ($dm:ident, $e:expr) => {{
            let $dm = DisjointMut::new(vec![0u8; 64]);
            assert!(panics(|| drop($e)), "expected panic for {}", stringify!($e));
            // No poisoning machinery: subsequent in-bounds access must still work.
            drop($dm.index(0));
            drop($dm.index_mut(0..64));
        }};
    }
    case!(dm, dm.index_mut(64));
    case!(dm, dm.index(usize::MAX - 1));
    case!(dm, dm.index_mut(60..65));
    case!(dm, dm.index_mut(70..));
    case!(dm, dm.index_mut(70..70));
    case!(dm, dm.index_mut(65..60));
    case!(dm, dm.index_mut(..65));
    case!(dm, dm.index_mut(..=64));
    case!(dm, dm.index_mut(0..=64));
    case!(dm, dm.index_mut((60.., ..5)));
    case!(dm, dm.index_mut((64.., ..1)));
    case!(dm, dm.index_mut((usize::MAX.., ..0)));
    case!(dm, dm.index_mut((usize::MAX - 1.., ..4)));
    case!(dm, dm.index(0..=usize::MAX));
    case!(dm, dm.index(..=usize::MAX));
    case!(dm, dm.index_mut(usize::MAX));
}

#[test]
fn oob_against_live_length_after_resize() {
    let mut dm = DisjointMut::new(vec![0u8; 64]);
    dm.resize(16, 0);
    assert!(panics(|| drop(dm.index_mut(16))));
    assert!(panics(|| drop(dm.index_mut(0..17))));
    drop(dm.index_mut(0..16));
    dm.resize(0, 0);
    assert!(panics(|| drop(dm.index(0))));
    assert_eq!(dm.index(..).len(), 0);
    dm.resize(200, 7);
    assert_eq!(dm.index(199).clone(), 7);
    assert!(panics(|| drop(dm.index(200))));
}

#[test]
fn oob_other_containers() {
    // array
    static ARR: DisjointMut<[u16; 4]> = DisjointMut::new([0; 4]);
    assert!(panics(|| drop(ARR.index_mut(4))));
    assert!(panics(|| drop(ARR.index(2..5))));
    drop(ARR.index_mut(0..4));
    // &mut [V]
    let mut backing = [0u32; 8];
    let dm = DisjointMut::new(&mut backing[2..6]);
    assert!(panics(|| drop(dm.index_mut(4))));
    drop(dm.index_mut(3));
    // Box<[V]>
    let dm: DisjointMut<Box<[i16]>> = DisjointMut::new(vec![0i16; 3].into_boxed_slice());
    assert_eq!(dm.index(3..).len(), 0);
    assert!(panics(|| drop(dm.index(4..))));
    // PicBuf with alignment offset: usable_len is the bound, not the Vec length
    let pb = PicBuf::from_vec_aligned(vec![0u8; 256], 64, 100);
    let dm = DisjointMut::new(pb);
    drop(dm.index(99));
    assert!(panics(|| drop(dm.index(100))));
    assert!(panics(|| drop(dm.index_mut(0..101))));
    // aligned
    use rav1d_disjoint_mut::align::{Align64, AlignedVec64};
    let dm: DisjointMut<Align64<[u8; 32]>> = DisjointMut::new(aligned::Aligned([0u8; 32]));
    assert!(panics(|| drop(dm.index(32))));
    let mut av = AlignedVec64::<u8>::new();
    av.resize(10, 0);
    let dm = DisjointMut::new(av);
    assert!(panics(|| drop(dm.index_mut(10))));
    drop(dm.index_mut(9));
}

#[test]
fn rect_geometry_still_refuses_oob() {
    let mut dm = DisjointMut::new(vec![0u8; 64]);
    dm.declare_row_stride(8);
    assert!(dm.index_rect_mut(0, 8, 8, 8).is_some());
    assert!(dm.index_rect_mut(0, 8, 9, 8).is_none());
    assert!(dm.index_rect(56, 8, 9, -8).is_none());
    assert!(dm.index_rect(60, 8, 1, 8).is_none());
    assert!(dm.index_rect(0, 8, 8, 7).is_none());
    assert!(dm.index_rect(usize::MAX, 1, 1, 1).is_none());
    assert!(dm.index_rect(0, 8, 2, isize::MIN).is_none());
    let mut r = dm.index_rect_mut(0, 8, 8, 8).unwrap();
    assert!(panics(|| {
        r.row_mut(8);
    }));
    assert!(dm.index_rect_mut_as::<u16>(0, 4, 8, 4).is_some());
    assert!(dm.index_rect_mut_as::<u16>(0, 4, 9, 4).is_none());
}

#[test]
fn zerocopy_casts_still_bounds_checked() {
    let dm = DisjointMut::new(vec![0u8; 7]);
    assert!(panics(|| drop(dm.mut_slice_as::<_, u16>(0..4))));
    assert!(panics(|| drop(dm.slice_as::<_, u32>(1..2))));
    assert!(panics(|| drop(dm.element_as::<u64>(0))));
    drop(dm.mut_slice_as::<_, u8>(0..7));
    let mut backing = [0u8; 16];
    let slice = &mut backing[1..];
    let dm = DisjointMut::new(slice);
    let r = catch_unwind(AssertUnwindSafe(|| dm.mut_slice_as::<_, u16>(0..2).len()));
    eprintln!(
        "misaligned u16 cast -> {:?}",
        r.as_ref().map_err(|_| "panic")
    );
}

#[test]
fn forgotten_guard_then_shrink_no_dangling_reference_possible() {
    let mut dm = DisjointMut::new(vec![0u8; 64]);
    std::mem::forget(dm.index_mut(0..64));
    dm.resize(4, 0);
    drop(dm.index_mut(0..4));
    assert!(panics(|| drop(dm.index(4))));
}

// Deliberately overlapping cross-thread writes of the SAME value. Accepted by
// the crate under `untracked`; this is the documented-UB case (Miri will flag it).
#[test]
fn overlapping_cross_thread_same_value_writes() {
    let dm = DisjointMut::new(vec![0u8; 1024]);
    std::thread::scope(|s| {
        for _ in 0..4 {
            let dm = &dm;
            s.spawn(move || {
                for _ in 0..50 {
                    let mut g = dm.index_mut(0..1024);
                    for b in g.iter_mut() {
                        *b = 0x5A;
                    }
                }
            });
        }
    });
    assert!(dm.index(..).iter().all(|&b| b == 0x5A));
}
