//! Does a rectangle guard hand out MORE than the tracker reserved?
//!
//! `index_rect_mut` registers the rectangle (so the inter-row gaps stay
//! available to other tile columns — the whole point of `StridedRows`) but the
//! reference it returns is `rect.hull()`, i.e. the gaps included. Two blocks on
//! the same rows at different columns are therefore accepted by the tracker
//! while their `&mut [u8]` ranges overlap.
//!
//! Run under Miri to decide it:
//! `cargo +nightly miri test -p rav1d-disjoint-mut --test rect_hull_aliasing`

use rav1d_disjoint_mut::{DisjointMut, StridedRows, set_parallelism};

/// Small enough for Miri, big enough to be sharded (`SHARD_MIN_LEN` = 64 KiB)
/// and to get a row map (stride >= 16).
const STRIDE: usize = 256;
const ROWS: usize = 256;
const LEN: usize = STRIDE * ROWS;

fn rect(start: usize, w: usize, h: usize) -> StridedRows {
    StridedRows {
        start,
        w,
        h,
        stride: STRIDE,
    }
}

/// First, without Miri: state the geometry as a plain fact.
///
/// This asserts only arithmetic — it passes on any toolchain and documents
/// exactly what the Miri test below is exercising.
#[test]
fn the_hulls_of_two_accepted_rectangles_overlap() {
    let a = rect(0, 16, 16);
    let b = rect(64, 16, 16);
    let a_hull = a.start..a.start + (a.h - 1) * a.stride + a.w;
    let b_hull = b.start..b.start + (b.h - 1) * b.stride + b.w;
    assert!(
        a_hull.start < b_hull.end && b_hull.start < a_hull.end,
        "the two hulls must overlap for this to be a question: {a_hull:?} vs {b_hull:?}"
    );
    // ... while the BYTES each one actually covers are disjoint (columns
    // 0..=15 versus 64..=79 of the same sixteen rows).
    for row in 0..16 {
        let ar = a.start + row * STRIDE..a.start + row * STRIDE + a.w;
        let br = b.start + row * STRIDE..b.start + row * STRIDE + b.w;
        assert!(ar.end <= br.start, "row {row}: {ar:?} vs {br:?}");
    }
}

/// The tracker accepts both — that part is CORRECT and is what the row map is
/// for. Kept as its own test so a failure here is read as "the row map broke",
/// not "the aliasing question changed".
#[test]
fn the_tracker_accepts_both() {
    set_parallelism(8);
    let mut dm: DisjointMut<Vec<u8>> = DisjointMut::new(vec![0u8; LEN]);
    dm.set_row_stride(STRIDE);
    assert!(
        dm.rect_exact_for(STRIDE),
        "the row map must be live, or this test proves nothing"
    );
    let _a = dm.index_rect_mut(rect(0, 16, 16));
    let _b = dm.index_rect_mut(rect(64, 16, 16));
}

/// CONTROL 1: the shape `main` uses under tile threading — `h` per-row guards
/// per block, two tile columns' worth, all live at once.
///
/// If this also tripped Miri, the finding below would be "`DisjointMut` has
/// always been UB", not "the rectangle guard is". It must pass.
#[test]
fn control_per_row_guards_from_two_tile_columns_do_not_alias() {
    set_parallelism(8);
    let mut dm: DisjointMut<Vec<u8>> = DisjointMut::new(vec![0u8; LEN]);
    dm.set_row_stride(STRIDE);

    let mut held = Vec::new();
    for tile in 0..2 {
        for row in 0..16 {
            let off = tile * 64 + row * STRIDE;
            held.push(dm.index_mut(off..off + 16));
        }
    }
    for (i, g) in held.iter_mut().enumerate() {
        g[0] = i as u8;
    }
    for (i, g) in held.iter().enumerate() {
        assert_eq!(g[0], i as u8);
    }
}

/// CONTROL 2: the shape `main` uses when nothing is parallel — ONE hull guard,
/// which the tracker also RESERVES as the hull. Exclusive, so it is sound; it
/// is the correspondence between reserved and handed-out that the rectangle
/// breaks.
#[test]
fn control_one_hull_guard_is_exclusive() {
    set_parallelism(8);
    let mut dm: DisjointMut<Vec<u8>> = DisjointMut::new(vec![0u8; LEN]);
    dm.set_row_stride(STRIDE);
    let mut hull = dm.index_mut(0..15 * STRIDE + 16);
    hull[0] = 1;
    hull[15 * STRIDE] = 2;
    assert_eq!(hull[0], 1);
}

/// The question: two live `&mut [u8]` over overlapping memory.
///
/// Under Stacked/Tree Borrows, creating `b` retags the shared allocation and
/// invalidates `a`'s claim to the bytes they share; the write through `a`
/// afterwards is then UB. Both writes land in bytes each guard genuinely owns —
/// the aliasing is in the REFERENCES, not in the accesses, which is precisely
/// why no amount of tracker exactness can fix it.
#[test]
fn writing_through_both_guards_while_both_are_live() {
    set_parallelism(8);
    let mut dm: DisjointMut<Vec<u8>> = DisjointMut::new(vec![0u8; LEN]);
    dm.set_row_stride(STRIDE);

    let mut a = dm.index_rect_mut(rect(0, 16, 16));
    let mut b = dm.index_rect_mut(rect(64, 16, 16));

    // Byte 0 of `a`'s slice is column 0; byte 0 of `b`'s slice is column 64.
    // Disjoint bytes, overlapping references.
    b[0] = 2;
    a[0] = 1;
    assert_eq!(b[0], 2);
    assert_eq!(a[0], 1);
}

/// The same shape the shipped mechanism test holds: four tile columns plus the
/// 1-pixel intra left column each of them reads. Seven live guards, every pair
/// of hulls overlapping.
#[test]
fn four_tile_columns_the_way_the_decoder_holds_them() {
    set_parallelism(8);
    let mut dm: DisjointMut<Vec<u8>> = DisjointMut::new(vec![0u8; LEN]);
    dm.set_row_stride(STRIDE);

    let mut held = Vec::new();
    for tile in 0..4 {
        held.push(dm.index_rect_mut(rect(tile * 64, 16, 16)));
    }
    // Touch every guard while all four are live.
    for (i, g) in held.iter_mut().enumerate() {
        g[0] = i as u8;
    }
    for (i, g) in held.iter().enumerate() {
        assert_eq!(g[0], i as u8);
    }
}
