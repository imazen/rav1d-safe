//! Mechanism gate for the TILE-KEYED borrow arms.
//!
//! The property these arms add is invisible in decoded pixels — every arm of
//! this branch produces md5 `a00c11f454328023c58af14d55544cff` on `v4k_8tile`
//! at every thread count — so nothing else in the tree can catch a regression
//! in it. Each test therefore asserts the MECHANISM (which borrows the tracker
//! declares disjoint), not an outcome.
//!
//! It also records, as an executable assertion rather than a comment, the
//! precise sense in which these arms are UNSOUND: a keyed borrow and an
//! UNKEYED one that genuinely overlap are NOT detected. That is not a bug in
//! the implementation, it is the design's open edge — the tile key partitions
//! the shard space, so a borrow outside the partition never meets one inside
//! it. Closing it means keying the filter chain too. Asserting it here means
//! the next agent cannot mistake it for something that was overlooked.
//!
//! Build: `cargo test -p rav1d-disjoint-mut --test tile_key
//!         --features __probe_tilekey_shard,zerocopy`
//! (`zerocopy` gates the cast twins this file exercises.)
//! Without the feature every test below asserts the INERT behaviour instead,
//! so the file is meaningful in both configurations.

use rav1d_disjoint_mut::{DisjointMut, TILE_ANY};
use std::panic::{self, AssertUnwindSafe};

/// Past `SHARD_MIN_LEN` (64 KiB) so `mask_for` gives the instance a real shard
/// set. Below it every borrow is single-shard and the key is never consulted —
/// a test on a small buffer would pass for the wrong reason.
const LEN: usize = 8 * 1024 * 1024;

fn parallel_instance() -> DisjointMut<Vec<u8>> {
    // Parallelism must be declared BEFORE the instance is built: `mask` is read
    // once in `BorrowTracker::new` and immutable for that tracker's life.
    rav1d_disjoint_mut::set_parallelism(8);
    DisjointMut::new(vec![0u8; LEN])
}

/// Is a second mutable borrow of `b` rejected while a keyed borrow of `a` is
/// live? Returns `true` when the tracker panicked, i.e. detected the overlap.
fn conflicts(
    dm: &DisjointMut<Vec<u8>>,
    a: std::ops::Range<usize>,
    ka: u16,
    b: std::ops::Range<usize>,
    kb: u16,
) -> bool {
    let held = dm.index_mut_keyed(a, ka);
    let prev = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let r = panic::catch_unwind(AssertUnwindSafe(|| {
        let g = dm.index_mut_keyed(b, kb);
        drop(g);
    }));
    panic::set_hook(prev);
    drop(held);
    r.is_err()
}

/// Two borrows in the SAME tile still meet. This is the anti-vacuity control:
/// without it, a mutation that turns the key into a global "never conflict"
/// switch would leave every other test in this file green.
#[test]
fn same_tile_overlapping_borrows_still_conflict() {
    let dm = parallel_instance();
    assert!(
        conflicts(&dm, 0..4096, 3, 2048..8192, 3),
        "two overlapping mutable borrows in the SAME tile must be detected \
         whatever the sharding scheme"
    );
}

/// An UNKEYED pair still meets — today's behaviour, unchanged.
#[test]
fn unkeyed_overlapping_borrows_still_conflict() {
    let dm = parallel_instance();
    assert!(
        conflicts(&dm, 0..4096, TILE_ANY, 2048..8192, TILE_ANY),
        "TILE_ANY is the conservative key: two unkeyed overlapping borrows \
         must be detected exactly as they are on main"
    );
}

/// THE MECHANISM. Two borrows that overlap as INTERVALS but name different
/// tiles are declared disjoint — which is the whole point, because a strided
/// hull in tile column 0 and one in tile column 1 overlap as intervals and
/// never share a byte.
///
/// Without the feature the same pair must still conflict, so this test also
/// gates that the arm is genuinely off by default.
#[test]
fn different_tiles_are_disjoint_only_with_the_feature() {
    let dm = parallel_instance();
    let detected = conflicts(&dm, 0..4096, 0, 2048..8192, 1);
    if cfg!(feature = "__probe_tilekey_shard") {
        assert!(
            !detected,
            "with __probe_tilekey_shard, borrows naming DIFFERENT tiles must \
             not be compared — this is the mechanism the arm exists to provide"
        );
    } else {
        assert!(
            detected,
            "without the feature the key must be inert: an overlapping pair \
             must still be detected however it is keyed"
        );
    }
}

/// The tile key must reach the tracker through EVERY keyed entry point, not
/// just `index_mut_keyed`. A twin that dropped the key on the floor would make
/// its call sites silently unkeyed — slower, but green.
#[cfg(feature = "zerocopy")]
#[test]
fn every_keyed_entry_point_carries_the_key() {
    let dm = parallel_instance();

    // slice_as_keyed / mut_slice_as_keyed go through index{,_mut}_keyed.
    let held = dm.mut_slice_as_keyed::<_, u8>((0.., ..4096), 0);
    let prev = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let other = panic::catch_unwind(AssertUnwindSafe(|| {
        drop(dm.mut_slice_as_keyed::<_, u8>((2048.., ..4096), 1));
    }));
    let same = panic::catch_unwind(AssertUnwindSafe(|| {
        drop(dm.mut_slice_as_keyed::<_, u8>((2048.., ..4096), 0));
    }));
    panic::set_hook(prev);
    drop(held);

    assert!(
        same.is_err(),
        "mut_slice_as_keyed must still detect an overlap within one tile"
    );
    if cfg!(feature = "__probe_tilekey_shard") {
        assert!(
            other.is_ok(),
            "mut_slice_as_keyed dropped the key: a different tile was still \
             compared, so its call sites are silently unkeyed"
        );
    } else {
        assert!(other.is_err(), "the key must be inert without the feature");
    }
}

/// THE OPEN EDGE, asserted rather than described.
///
/// A keyed borrow lands in its tile's shard; an unkeyed one lands in an
/// address-hashed shard. They are in different partitions of the same shard
/// array, so a genuine overlap between them is MISSED. Every picture access
/// made by the filter chain is unkeyed today (6,389,542 registrations/frame at
/// t=8 on v4k_8tile), which is why these arms are measurement-only.
///
/// If a future change keys the filter chain, or makes unkeyed borrows meet
/// keyed ones some other way, THIS TEST MUST BE UPDATED TO EXPECT DETECTION —
/// and its failure is the signal that the design became sound.
#[test]
fn keyed_versus_unkeyed_overlap_is_missed_and_that_is_the_open_edge() {
    let dm = parallel_instance();
    let detected = conflicts(&dm, 0..4096, 7, 2048..8192, TILE_ANY);
    if cfg!(feature = "__probe_tilekey_shard") {
        assert!(
            !detected,
            "if this now DETECTS, the shard partition has been closed and the \
             design may have become sound — re-read docs/TILE_KEYED_BORROWS.md \
             and update this gate deliberately"
        );
    } else {
        assert!(detected, "the key must be inert without the feature");
    }
}

/// A borrow keyed to a tile whose index exceeds the shard array wraps, and two
/// tiles that wrap onto the same shard must still be compared as intervals.
/// Left unasserted this would be a silent missed overlap at high tile counts
/// (AV1 permits 64 tile columns; `N_SHARDS` is 128, so the wrap needs > 128
/// tiles — reachable only with tile rows, but reachable).
#[test]
fn tile_keys_that_alias_onto_one_shard_are_still_compared() {
    let dm = parallel_instance();
    // Any two keys congruent modulo the shard array land together. 128 shards
    // is the compile-time maximum, so k and k+128 alias for every active mask.
    let detected = conflicts(&dm, 0..4096, 5, 2048..8192, 5 + 128);
    if cfg!(feature = "__probe_tilekey_shard") {
        assert!(
            detected,
            "two tile keys that alias onto one shard share a record set, so \
             their intervals must still be compared — otherwise a >128-tile \
             frame has silently missed overlaps"
        );
    } else {
        assert!(detected, "the key must be inert without the feature");
    }
}
