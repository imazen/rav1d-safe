//! Pins for the A-01 and A-03 fixes that discriminate in every build profile.
//!
//! Added after the second review of the audit fixes:
//! * `pic_buf_set_usable_len.rs` only separates fixed from unfixed code in release (the
//!   unchecked add panics with its own overflow message in the dev profile), and the CI jobs
//!   for this crate run the dev profile. The test below pins the checked message instead.
//! * `lock_liveness_after_panic.rs` proves locks are released; it does not prove the capacity
//!   refusal happens before the record is pushed. A mutant that pushes first and releases the
//!   locks afterwards passes it. The second test below fails on that mutant: a ghost wide
//!   record would refuse an unrelated mutable borrow.
#![forbid(unsafe_code)]

use rav1d_disjoint_mut::DisjointMut;
use std::panic::{AssertUnwindSafe, catch_unwind};

fn message(payload: &Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

/// Dev profile: the old unchecked add panics with "attempt to add with overflow", so a
/// bare `is_err()` cannot tell the fixed code from the broken one. The fixed setter names the
/// overflow in its own message.
#[cfg(feature = "pic-buf")]
#[test]
fn set_usable_len_refusal_names_the_overflow_in_every_profile() {
    use rav1d_disjoint_mut::PicBuf;
    const STORAGE: usize = 65536;
    let storage = vec![0u8; STORAGE];
    let addr = storage.as_ptr() as usize;
    let alignment = 1usize << (addr.trailing_zeros() + 1);
    let offset = storage.as_ptr().align_offset(alignment);
    assert!(
        offset > 0 && offset < STORAGE,
        "need a Vec that is not already aligned"
    );
    let mut buffer = PicBuf::from_vec_aligned(storage, alignment, 1);
    let err = catch_unwind(AssertUnwindSafe(|| {
        buffer.set_usable_len(usize::MAX - offset + 1)
    }))
    .expect_err("a region end that wraps must be refused");
    let msg = message(&err);
    assert!(
        msg.contains("overflows usize"),
        "refusal must come from the checked addition, got: {msg}"
    );
}

/// 7 narrow + 65536 wide immutable guards on a one-shard buffer exhaust the u16 wide ids;
/// the next wide request is refused. Dropping every guard must leave NO record behind: a later
/// mutable borrow that has to consult the wide list (one live wide reader keeps `state != 0`)
/// must not see a ghost of the refused request.
#[cfg(not(any(feature = "untracked", feature = "__probe_count")))]
#[test]
fn capacity_refusal_leaves_no_stale_wide_record() {
    let dm = DisjointMut::new(vec![0u8; 8]);
    let mut held = Vec::new();
    let refusal = catch_unwind(AssertUnwindSafe(|| {
        for _ in 0..65544 {
            held.push(dm.index(0..1));
        }
    }))
    .expect_err("expected the finite wide-record limit");
    assert!(message(&refusal).contains("too many concurrent wide borrows"));
    assert_eq!(held.len(), 7 + 65536, "7 narrow slots + 65536 wide ids");
    drop(held);

    // Re-create exactly one live wide reader far from byte 0 (7 narrow readers fill the shard,
    // the 8th is wide), so every following registration consults the wide list.
    let pins: Vec<_> = (0..8).map(|_| dm.index(5..6)).collect();
    let probe = catch_unwind(AssertUnwindSafe(|| drop(dm.index_mut(0..1))));
    drop(pins);
    assert!(
        probe.is_ok(),
        "a ghost record from the refused wide request still refuses byte 0: {:?}",
        probe.err().map(|e| message(&e))
    );
}
