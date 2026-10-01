//! Adversarial tests -- DEFAULT (tracked) configuration.
//! Written by an independent clean-context reviewer (2026-10-01) trying to break
//! the PlainData / `untracked` / lazy-bounds changes, then adopted into the suite.
#![cfg(not(feature = "untracked"))]

use rav1d_disjoint_mut::{DisjointMut, DisjointMutArcSlice, PlainData};
use std::mem::MaybeUninit;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Barrier, mpsc};
use std::thread;

fn panics(f: impl FnOnce()) -> bool {
    catch_unwind(AssertUnwindSafe(f)).is_err()
}

// ---------- OOB forms: every shape must panic, and poison afterwards ----------

#[test]
fn oob_every_index_shape_panics_then_poisons() {
    macro_rules! case {
        ($dm:ident, $e:expr) => {{
            let $dm = DisjointMut::new(vec![0u8; 64]);
            assert!(panics(|| drop($e)), "expected panic for {}", stringify!($e));
            // Poisoned: even a trivially in-bounds borrow must now fail.
            assert!(
                panics(|| drop($dm.index(0))),
                "expected poison after {}",
                stringify!($e)
            );
            // Found by this review (fixed): an EMPTY range used to be admitted on a
            // poisoned buffer because the tracker returned before the poison check.
            assert!(
                panics(|| drop($dm.index(64..64))),
                "empty range admitted after poison ({})",
                stringify!($e)
            );
        }};
    }
    case!(dm, dm.index_mut(64));
    case!(dm, dm.index(usize::MAX - 1));
    case!(dm, dm.index_mut(60..65));
    case!(dm, dm.index_mut(70..));
    case!(dm, dm.index_mut(70..70)); // empty but past end
    case!(dm, dm.index_mut(65..60)); // inverted
    case!(dm, dm.index_mut(..65));
    case!(dm, dm.index_mut(..=64));
    case!(dm, dm.index_mut(0..=64));
    case!(dm, dm.index_mut((60.., ..5)));
    case!(dm, dm.index_mut((64.., ..1)));
    case!(dm, dm.index_mut((usize::MAX.., ..0)));
}

#[test]
fn oob_arithmetic_overflow_forms_panic_without_ub() {
    // These overflow inside the Bounds conversion (debug) or wrap (release).
    // Either way the outcome must be a panic, never an accepted view.
    let dm = DisjointMut::new(vec![0u8; 64]);
    assert!(panics(|| drop(dm.index(0..=usize::MAX))));
    assert!(panics(|| drop(dm.index(..=usize::MAX))));
    assert!(panics(|| drop(dm.index_mut((usize::MAX - 1.., ..4)))));
    assert!(panics(|| drop(dm.index_mut((usize::MAX.., ..1)))));
    assert!(panics(|| drop(dm.index_mut(usize::MAX))));
    // Whatever poisoning state resulted, a later access is a panic or a
    // correct view — never anything else. Record which.
    let later = catch_unwind(AssertUnwindSafe(|| *dm.index(0)));
    eprintln!(
        "after overflow forms, index(0) => {:?}",
        later.as_ref().map_err(|_| "poisoned")
    );
}

#[test]
fn zero_length_buffer_edges() {
    let dm = DisjointMut::new(Vec::<u8>::new());
    assert_eq!(dm.index(..).len(), 0);
    assert_eq!(dm.index(0..).len(), 0);
    assert_eq!(dm.index(0..0).len(), 0);
    assert!(panics(|| drop(dm.index(0))));
    let dm = DisjointMut::new(Vec::<u8>::new());
    assert!(panics(|| drop(dm.index(1..))));
}

// ---------- overlap / leak / thread behaviour ----------

#[test]
fn forgotten_guard_locks_region_forever() {
    let dm = DisjointMut::new(vec![0u8; 64]);
    let g = dm.index_mut(0..8);
    std::mem::forget(g);
    assert!(panics(|| drop(dm.index(4))));
    assert!(panics(|| drop(dm.index_mut(7..9))));
    drop(dm.index(8..64)); // disjoint still fine
}

#[test]
fn forgotten_guard_then_resize_behaviour() {
    let mut dm = DisjointMut::new(vec![0u8; 64]);
    std::mem::forget(dm.index_mut(0..8));
    dm.resize(128, 0);
    // Sound either way (the forgotten guard can never be dereferenced).
    // Record whether the leaked record survives reprovisioning.
    let r = catch_unwind(AssertUnwindSafe(|| drop(dm.index_mut(0..8))));
    eprintln!("forgotten record survives resize: {}", r.is_err());
    dm.resize(4, 0);
    let r = catch_unwind(AssertUnwindSafe(|| drop(dm.index_mut(0..4))));
    eprintln!("forgotten record survives shrink: {}", r.is_err());
    assert!(panics(|| drop(dm.index(4))));
}

#[test]
fn cross_thread_overlap_is_rejected() {
    let dm = Arc::new(DisjointMut::new(vec![0u8; 64]));
    let b = Arc::new(Barrier::new(2));
    let (tx, rx) = mpsc::channel::<()>();
    let t = {
        let dm = dm.clone();
        let b = b.clone();
        thread::spawn(move || {
            let mut g = dm.index_mut(0..32);
            g[0] = 1;
            b.wait(); // guard live while main tries
            rx.recv().unwrap();
            g[0]
        })
    };
    b.wait();
    assert!(panics(|| drop(dm.index(16))));
    assert!(panics(|| drop(dm.index_mut(31..40))));
    drop(dm.index_mut(32..64));
    tx.send(()).unwrap();
    assert_eq!(t.join().unwrap(), 1);
    drop(dm.index_mut(0..64));
}

#[test]
fn panic_in_other_thread_with_live_mut_guard_poisons() {
    let dm = Arc::new(DisjointMut::new(vec![0u8; 64]));
    let t = {
        let dm = dm.clone();
        thread::spawn(move || {
            let _g = dm.index_mut(0..8);
            panic!("boom with live mut guard");
        })
    };
    assert!(t.join().is_err());
    assert!(panics(|| drop(dm.index(40))));
}

#[test]
fn panic_in_other_thread_with_live_immut_guard_does_not_poison_but_releases() {
    let dm = Arc::new(DisjointMut::new(vec![0u8; 64]));
    let t = {
        let dm = dm.clone();
        thread::spawn(move || {
            let _g = dm.index(0..8);
            panic!("boom with live immut guard");
        })
    };
    assert!(t.join().is_err());
    // Not poisoned; region released.
    drop(dm.index_mut(0..8));
}

#[test]
fn guard_sent_to_other_thread_and_dropped_there_releases() {
    let dm = Arc::new(DisjointMut::new(vec![0u8; 64]));
    thread::scope(|s| {
        let g = dm.index(0..8);
        s.spawn(move || {
            let _ = g[0];
            drop(g);
        })
        .join()
        .unwrap();
        drop(dm.index_mut(0..8));
    });
}

#[test]
fn arc_slice_clones_share_one_tracker() {
    let a = DisjointMutArcSlice::try_new(64, 0u8).unwrap();
    let b = a.clone();
    let _g = a.index_mut(0..8);
    assert!(panics(|| drop(b.index_mut(4..12))));
    drop(b.index_mut(8..16));
}

#[test]
fn rect_mut_vs_linear_overlap_rejected_and_gap_allowed() {
    let mut dm = DisjointMut::new(vec![0u8; 64]);
    dm.declare_row_stride(8);
    let r = dm.index_rect_mut(0, 4, 2, 8).expect("representable");
    assert!(panics(|| drop(dm.index(8..12))));
    assert!(panics(|| drop(dm.index(3))));
    drop(dm.index(4..8)); // the inter-row gap is NOT reserved
    drop(r);
    drop(dm.index_mut(0..16));
}

#[test]
fn rect_negative_stride_geometry_oob_refused() {
    let mut dm = DisjointMut::new(vec![0u8; 64]);
    dm.declare_row_stride(8);
    assert!(dm.index_rect(56, 8, 8, -8).is_some());
    assert!(dm.index_rect(56, 8, 9, -8).is_none());
    assert!(dm.index_rect(60, 8, 1, 8).is_none());
    assert!(dm.index_rect(0, 8, 8, 7).is_none()); // rows overlap
    assert!(dm.index_rect(0, 8, 8, isize::MIN).is_none());
    assert!(dm.index_rect(usize::MAX, 1, 1, 1).is_none());
}

// ---------- zerocopy cast surfaces ----------

#[test]
fn zerocopy_misaligned_cast_panics_not_ub() {
    let mut backing = [0u8; 16];
    let slice = &mut backing[1..];
    let dm = DisjointMut::new(slice);
    // bytes 0..4 of a slice starting at odd address -> &[u16] misaligned
    let r = catch_unwind(AssertUnwindSafe(|| {
        let g = dm.mut_slice_as::<_, u16>(0..2);
        g.len()
    }));
    eprintln!(
        "misaligned u16 cast -> {:?}",
        r.as_ref().map_err(|_| "panic")
    );
    assert!(r.is_err() || backing_is_odd_aligned(&backing));
}
fn backing_is_odd_aligned(b: &[u8; 16]) -> bool {
    (b.as_ptr() as usize) % 2 == 1
}

#[test]
fn zerocopy_oob_cast_panics() {
    let dm = DisjointMut::new(vec![0u8; 7]);
    assert!(panics(|| drop(dm.mut_slice_as::<_, u16>(0..4))));
    assert!(panics(|| drop(dm.slice_as::<_, u32>(1..2)))); // bytes 4..8 > 7
    assert!(panics(|| drop(dm.element_as::<u64>(0))));
}

// ---------- PlainData surface ----------

fn assert_plain<T: PlainData>() {}

#[test]
fn plaindata_admits_maybe_uninit_of_pointers() {
    // Documented claim: "no pointers, references or niches". zerocopy makes
    // MaybeUninit<T>: FromBytes for EVERY T, and MaybeUninit<T>: Copy for T: Copy.
    assert_plain::<MaybeUninit<&'static u8>>();
    assert_plain::<MaybeUninit<*mut u8>>();
    assert_plain::<MaybeUninit<fn()>>();
    let dm = DisjointMut::new(vec![MaybeUninit::<&'static u8>::uninit(); 4]);
    let _g = dm.index(0..4); // reading MaybeUninit is fine; assume_init is unsafe
}

// ---------- lazy tracker init race ----------

#[test]
fn lazy_tracker_concurrent_first_borrows_then_overlap_rejected() {
    static BUF: DisjointMut<[u8; 256]> = DisjointMut::new([0; 256]);
    let b = Arc::new(Barrier::new(8));
    thread::scope(|s| {
        for t in 0..8usize {
            let b = b.clone();
            s.spawn(move || {
                b.wait();
                let mut g = BUF.index_mut(t * 32..(t + 1) * 32);
                g.fill(t as u8);
            });
        }
    });
    let g = BUF.index_mut(0..256);
    for t in 0..8 {
        assert!(g[t * 32..(t + 1) * 32].iter().all(|&x| x == t as u8));
    }
    drop(g);
    let _h = BUF.index(0..1);
    let _k = BUF.index_mut(1..2);
    assert!(panics(|| drop(BUF.index_mut(0..2))));
}
