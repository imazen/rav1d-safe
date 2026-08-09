//! Aliasing gate for MOVING a guard: `drop(g)` must not be UB.
//!
//! # The defect this exists to catch
//!
//! A guard is a value, so safe code may move it into a call — `drop(g)` is the
//! everyday case, `ManuallyDrop::new(g)` and `f(g)` are others. Rust's aliasing
//! model gives every reference passed **by value into a call** a *protector*:
//! for the whole of that call the reference must stay valid and nothing may
//! invalidate it.
//!
//! When the guard carried its region as `&'a mut V` (as it did until
//! 2026-08-09), that protector covered the buffer bytes — and the guard's
//! `Drop`, which runs inside that very call, retires the tracker record.
//! Retiring is exactly what authorises **another thread** to take the region
//! and retag those bytes for itself. The protected reference is invalidated
//! mid-call. UB, reachable from safe code, on the release path the whole crate
//! is built around.
//!
//! Both guard kinds had it. Under Stacked Borrows it reads
//! `not granting access to tag <A> because that would remove [Unique for <B>]`
//! (`[SharedReadOnly for <B>]` for the shared guard) `which is strongly
//! protected`; under Tree Borrows,
//! `this foreign write access would cause the protected tag <B> ... to become
//! Disabled`. **The two models agree**, which is why the fix is a fix and not
//! an appeasement of one experimental checker.
//!
//! Fix: the guards hold `*mut V` / `*const V` and materialise the reference in
//! `Deref`/`DerefMut`, where borrowck bounds it to a region in which the guard
//! cannot be dropped. `core::cell::RefMut` holds a `NonNull<T>` for the same
//! reason.
//!
//! # Teeth
//!
//! This file is a **Miri** gate: on native hardware nothing here is observable,
//! because no real machine invalidates anything on retag. Verified 2026-08-09
//! by running it against the pre-fix guards — both tests fail under both memory
//! models, and the `*_scope_exit` control arms, which differ only in NOT moving
//! the guard, pass. That pairing is what pins the cause to the move rather than
//! to the tracker: the tracker's grant/refusal counts are the same in all four.
//!
//! Keep the explicit `drop(g)`. It is the whole point of the test, and an
//! "unnecessary" -looking `drop` is the first thing a tidying pass deletes.

use rav1d_disjoint_mut::DisjointMut;
use std::panic::{self, AssertUnwindSafe};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

const LEN: usize = 32 * 1024;
/// Contended by every thread, so the race is between a release and an
/// acquisition of the SAME bytes — which is the only pairing that can pop a
/// protected tag. Disjoint ranges have disjoint borrow stacks and cannot.
const SHARED: core::ops::Range<usize> = 4096..4104;
const THREADS: usize = 7;

/// Miri interprets, so rounds are ~4 orders of magnitude dearer there. The
/// aliasing violation is not rare — it needs one release to land inside one
/// concurrent acquisition — so a few hundred rounds is plenty: measured
/// 2026-08-09 on the pre-fix guards, both tests abort in the first few hundred
/// acquisitions under both models.
const ROUNDS: usize = if cfg!(miri) { 400 } else { 200_000 };

/// Liveness floors, as a fraction of the attempts each arm makes. A run that
/// refused everything, or one whose threads never met, would prove nothing.
const MIN_GRANTED: usize = THREADS * ROUNDS / 4;

fn contend(mutable_only: bool, move_the_guard: bool) -> (usize, usize) {
    // One side of every race is a refusal, and refusals are panics.
    let prev = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));

    let dm = Arc::new(DisjointMut::new(vec![0u8; LEN]));
    let granted = Arc::new(AtomicUsize::new(0));
    let refused = Arc::new(AtomicUsize::new(0));

    let mut hs = Vec::new();
    for t in 0..THREADS {
        let dm = Arc::clone(&dm);
        let granted = Arc::clone(&granted);
        let refused = Arc::clone(&refused);
        // In the mixed arm most threads read: shared borrows COEXIST, so a
        // reader's release overlaps other readers' live borrows, and a writer
        // arriving next is the foreign retag that pops the protected shared
        // reference.
        let reader = !mutable_only && t < 5;
        hs.push(std::thread::spawn(move || {
            for _ in 0..ROUNDS {
                let ok = panic::catch_unwind(AssertUnwindSafe(|| {
                    if reader {
                        let g = dm.index(SHARED);
                        core::hint::black_box(g[0]);
                        if move_the_guard {
                            drop(g);
                        }
                    } else {
                        let mut g = dm.index_mut(SHARED);
                        g[0] = g[0].wrapping_add(1);
                        if move_the_guard {
                            drop(g);
                        }
                    }
                }))
                .is_ok();
                if ok {
                    granted.fetch_add(1, Ordering::Relaxed);
                } else {
                    refused.fetch_add(1, Ordering::Relaxed);
                }
            }
        }));
    }
    for h in hs {
        h.join().unwrap();
    }
    panic::set_hook(prev);
    (
        granted.load(Ordering::Relaxed),
        refused.load(Ordering::Relaxed),
    )
}

fn check(label: &str, mutable_only: bool) {
    let (moved_ok, moved_refused) = contend(mutable_only, true);
    // The control arm: identical work, guard destroyed by scope exit instead of
    // by a move. If this one ever fails, the cause is NOT the move and the
    // diagnosis above is wrong.
    let (scoped_ok, scoped_refused) = contend(mutable_only, false);

    assert!(
        moved_ok > MIN_GRANTED && scoped_ok > MIN_GRANTED,
        "{label}: the contended range was granted too rarely to have raced \
         (moved {moved_ok}, scoped {scoped_ok}, floor {MIN_GRANTED})"
    );
    assert!(
        moved_refused > 0 && scoped_refused > 0,
        "{label}: not one acquisition was refused, so the {THREADS} threads \
         never actually contended (moved {moved_refused}, scoped \
         {scoped_refused})"
    );
}

#[test]
fn moving_a_mut_guard_into_drop_is_not_ub() {
    check("mut", true);
}

#[test]
fn moving_a_shared_guard_into_drop_is_not_ub() {
    check("mixed", false);
}
