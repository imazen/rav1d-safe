//! Does a rectangle guard hand out MORE than the tracker reserved?
//!
//! `index_rect_mut` registers the rectangle, so the inter-row gaps stay
//! available to other tile columns — the whole point of `StridedRows`. The
//! question this file decides is whether the REFERENCE it hands out agrees:
//! when it was `rect.hull()` (gaps included), two blocks on the same rows at
//! different columns were accepted by the tracker while their `&mut [u8]`
//! ranges overlapped, and Miri rejected the second write. `DisjointMutRect`
//! answers it by handing out one ROW at a time, so every reference is a subset
//! of the record.
//!
//! Run under Miri to decide it, ONE TEST AT A TIME (Miri aborts on the first
//! UB, so a failure would mask the later cases):
//! `cargo +nightly miri test -p rav1d-disjoint-mut --test rect_hull_aliasing -- --exact <name>`
//!
//! # Which case catches which regression — MEASURED, not assumed
//!
//! There are two ways to put the hull reference back, and they are NOT caught
//! by the same tests. Planting each and re-running all nine under Miri:
//!
//! | regression | caught by |
//! |---|---|
//! | the guard STORES a `&mut`/`&` over the hull (what `index_rect_mut` did before) | `four_tile_columns_…`, `writing_through_both_guards_…`, `a_row_reference_survives_…`, `an_immutable_rectangle_…` |
//! | `row`/`row_mut` builds a TRANSIENT hull reference and indexes the row out of it | `a_row_reference_survives_…` and `an_immutable_rectangle_…` ONLY |
//!
//! The transient form was planted verbatim (rebuild `from_raw_parts_mut(base,
//! (h-1)*stride + w)` inside `row_mut`, then `&mut hull[row*stride..][..w]`):
//! `writing_through_both_guards_while_both_are_live` and
//! `four_tile_columns_the_way_the_decoder_holds_them` **both stayed GREEN**,
//! because each of their row references dies at the end of its own statement.
//! So the two cases that originally caught the defect are not sufficient on
//! their own once the API is row-shaped; the two that hold a row reference
//! ACROSS the next guard's retag are what keep this file honest. Do not delete
//! them as duplicates.

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

/// The question: two rectangle guards whose HULLS overlap, both live, both
/// written through.
///
/// Under Stacked/Tree Borrows, if creating `b`'s reference covered the hull it
/// would retag bytes `a` also claims and invalidate `a`; the write through `a`
/// afterwards was then UB. With the row view, each reference covers exactly one
/// row's `w` bytes, so the two are disjoint and both writes are fine. The
/// SCENARIO is unchanged — that is what makes this a regression gate rather
/// than a different test.
#[test]
fn writing_through_both_guards_while_both_are_live() {
    set_parallelism(8);
    let mut dm: DisjointMut<Vec<u8>> = DisjointMut::new(vec![0u8; LEN]);
    dm.set_row_stride(STRIDE);

    let mut a = dm.index_rect_mut(rect(0, 16, 16));
    let mut b = dm.index_rect_mut(rect(64, 16, 16));

    // Byte 0 of `a`'s row 0 is column 0; byte 0 of `b`'s row 0 is column 64.
    // Disjoint bytes, and now disjoint references.
    b.row_mut(0)[0] = 2;
    a.row_mut(0)[0] = 1;
    assert_eq!(b.row(0)[0], 2);
    assert_eq!(a.row(0)[0], 1);
}

/// The strongest form: hold a `&mut [u8]` FROM `a` across the creation of one
/// from `b`, then write through the older one.
///
/// This is the exact retag-then-use sequence a hull reference fails. It is a
/// stronger statement than the test above (where each row reference dies at the
/// end of its statement), and it is what the decoder does whenever two tile
/// workers are inside `for_rows_mut` at once.
#[test]
fn a_row_reference_survives_the_next_column_taking_one() {
    set_parallelism(8);
    let mut dm: DisjointMut<Vec<u8>> = DisjointMut::new(vec![0u8; LEN]);
    dm.set_row_stride(STRIDE);

    let mut a = dm.index_rect_mut(rect(0, 16, 16));
    let mut b = dm.index_rect_mut(rect(64, 16, 16));

    let ra = a.row_mut(3);
    let rb = b.row_mut(3);
    rb[0] = 2;
    ra[0] = 1; // use of `ra` AFTER `rb` exists and has been written through
    assert_eq!(ra[0], 1);
    assert_eq!(rb[0], 2);
}

/// The same shape the shipped mechanism test holds: four tile columns of the
/// same 16 picture rows. Four live guards, every pair of hulls overlapping.
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
        g.row_mut(0)[0] = i as u8;
    }
    for (i, g) in held.iter().enumerate() {
        assert_eq!(g.row(0)[0], i as u8);
    }
}

/// The immutable half: a `&` over the hull aliases another column's `&mut` in
/// the inter-row gaps, so `index_rect` needs the row view for the same reason.
#[test]
fn an_immutable_rectangle_does_not_alias_a_neighbours_writes() {
    set_parallelism(8);
    let mut dm: DisjointMut<Vec<u8>> = DisjointMut::new(vec![0u8; LEN]);
    dm.set_row_stride(STRIDE);

    let a = dm.index_rect(rect(0, 16, 16));
    let mut b = dm.index_rect_mut(rect(64, 16, 16));

    let ra = a.row(7);
    b.row_mut(7)[0] = 9;
    assert_eq!(ra[0], 0); // read through `a` AFTER `b` wrote
    assert_eq!(b.row(7)[0], 9);
}

/// Liveness: a rectangle guard must hand out exactly the rectangle — `h` rows
/// of `w`, at the declared stride — and NOTHING that spans a gap.
///
/// Without this, the two Miri cases above could pass by handing out an empty or
/// truncated row. It is a plain assertion, so it runs on any toolchain.
#[test]
fn the_rows_handed_out_are_exactly_the_rectangle() {
    set_parallelism(8);
    let mut dm: DisjointMut<Vec<u8>> = DisjointMut::new(vec![0u8; LEN]);
    dm.set_row_stride(STRIDE);
    let mut g = dm.index_rect_mut(rect(64, 16, 16));
    assert_eq!(g.rows(), 16);
    assert_eq!(g.row_len(), 16);
    for row in 0..16 {
        // Write a per-row marker through the row reference...
        g.row_mut(row)[0] = (row + 1) as u8;
        assert_eq!(g.row(row).len(), 16);
    }
    drop(g);
    // ...and confirm it landed at `start + row * stride`, i.e. the rows really
    // are strided and the gaps really were skipped.
    let all = dm.index(0..LEN);
    for row in 0..16usize {
        assert_eq!(all[64 + row * STRIDE], (row + 1) as u8, "row {row} marker");
        assert_eq!(
            all[64 + row * STRIDE + 16],
            0,
            "row {row} gap must be clean"
        );
    }
}
