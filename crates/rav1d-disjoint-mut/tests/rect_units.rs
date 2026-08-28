//! The rectangle borrows must be registered in the SAME coordinate system as
//! every other borrow of the buffer: `T::Target` elements.
//!
//! Until 2026-08-28 `index_rect{,_mut}` scaled the rectangle by
//! `size_of::<V>()` and registered BYTES, while `index{,_mut}` registers a
//! `Bounds` in `T::Target` elements. On a `u8` buffer (every caller in the
//! decoder) the two coincide; on a `Vec<u16>` a rectangle over elements
//! `{0,1,8,9}` was recorded as bytes `{0..4, 16..20}`, so an `index_mut(8..10)`
//! over the same elements was compared against the wrong bytes, found no
//! overlap, and handed out a second live `&mut` — undefined behaviour reached
//! from safe code. The first test here is that exact input; it MUST panic.
//!
//! Run under Miri as well (`cargo +nightly miri test --test rect_units`): the
//! `should_panic` tests panic before any aliasing write, and the disjoint
//! controls write through both guards, which Miri would reject if the two
//! rows or the row and the range actually aliased.

use rav1d_disjoint_mut::DisjointMut;

fn buf16() -> DisjointMut<Vec<u16>> {
    let mut dm = DisjointMut::new((0..64u16).collect::<Vec<_>>());
    // Stride in `T::Target` elements: rows of 8 `u16`s.
    dm.declare_row_stride(8);
    dm
}

/// The counterexample. Rectangle rows `{0,1}` and `{8,9}` (elements), then a
/// mutable range over `8..10` (elements) — the second row. Registered in
/// bytes, the rectangle's second row sat at `16..20` and the range at `8..10`
/// never met it.
#[test]
#[should_panic(expected = "overlapping DisjointMut")]
fn u16_rect_mut_then_overlapping_range_panics() {
    let dm = buf16();
    let rect = dm
        .index_rect_mut(0, 2, 2, 8)
        .expect("a 2x2 rectangle at the buffer start is representable");
    assert_eq!(rect.rows(), 2);
    let _range = dm.index_mut(8..10);
}

/// The mirror image: the range first, then the rectangle whose second row
/// covers it. The rectangle path's own scan (`find_from_rect`) must see the
/// range in the same unit.
#[test]
#[should_panic(expected = "overlapping DisjointMut")]
fn u16_range_then_overlapping_rect_mut_panics() {
    let dm = buf16();
    let _range = dm.index_mut(8..10);
    let _rect = dm
        .index_rect_mut(0, 2, 2, 8)
        .expect("the tracker must accept the geometry and then reject the overlap");
}

/// Immutable rectangle against a mutable range over one of its rows.
#[test]
#[should_panic(expected = "overlapping DisjointMut")]
fn u16_rect_then_mut_range_over_a_row_panics() {
    let dm = buf16();
    let _rect = dm.index_rect(0, 2, 2, 8).expect("representable");
    let _range = dm.index_mut(1..2);
}

/// Control: the same rectangle and a range in the inter-row GAP (elements
/// `2..8`) are disjoint in element units and must coexist — and both must be
/// writable, which Miri checks for real aliasing. Under the old byte-scaled
/// registration `2..8` collided with the rectangle's first row `0..4` (bytes)
/// and this was a false positive.
#[test]
fn u16_rect_mut_and_gap_range_coexist() {
    let dm = buf16();
    let mut rect = dm.index_rect_mut(0, 2, 2, 8).expect("representable");
    let mut gap = dm.index_mut(2..8);
    rect.row_mut(0)[0] = 100;
    rect.row_mut(1)[1] = 101;
    gap[0] = 200;
    gap[5] = 205;
    drop(gap);
    drop(rect);
    let all = dm.index(..);
    assert_eq!(all[0], 100);
    assert_eq!(all[9], 101);
    assert_eq!(all[2], 200);
    assert_eq!(all[7], 205);
}

/// The bound is in the buffer's own elements: a 64-element `u16` buffer holds
/// a rectangle in its upper half. Under the old `len / size_of::<V>()` bound
/// the buffer looked 32 elements long and this was refused.
#[test]
fn u16_rect_in_the_upper_half_is_representable() {
    let dm = buf16();
    let rect = dm
        .index_rect(48, 8, 2, 8)
        .expect("rows 48..56 and 56..64 are in bounds");
    assert_eq!(rect.row(0), &(48..56).collect::<Vec<u16>>()[..]);
    assert_eq!(rect.row(1), &(56..64).collect::<Vec<u16>>()[..]);
    assert!(
        dm.index_rect(48, 8, 3, 8).is_none(),
        "a third row would run past the end and must be refused"
    );
}

/// Negative stride: row 0 is the highest row. The registration covers the
/// same element set, so a range over the LOWEST row must still collide.
#[test]
#[should_panic(expected = "overlapping DisjointMut")]
fn u16_negative_stride_rect_covers_the_low_row() {
    let dm = buf16();
    // Rows at 16..20 (row 0) and 8..12 (row 1).
    let _rect = dm.index_rect_mut(16, 4, 2, -8).expect("representable");
    let _range = dm.index(8..9);
}

/// A 64-byte-aligned byte buffer, as the decoder's picture planes are. A plain
/// `Vec<u8>` is only 1-byte aligned in principle (and in practice under Miri),
/// and `index_rect_as::<u16>` correctly REFUSES a misaligned base — which
/// would turn these controls into a test of the allocator.
#[cfg(all(feature = "zerocopy", feature = "aligned"))]
fn bytes128() -> DisjointMut<rav1d_disjoint_mut::align::AlignedVec64<u8>> {
    let mut v = rav1d_disjoint_mut::align::AlignedVec64::<u8>::new();
    v.resize(128, 0u8);
    let mut dm = DisjointMut::new(v);
    dm.declare_row_stride(16); // 8 u16s per row, in bytes = elements
    dm
}

/// The byte-buffer path the decoder uses is unchanged: a `u16` rectangle over a
/// `u8` buffer registers `size_of::<u16>()` bytes per element, exactly as
/// `mut_slice_as::<u16>` does, so the two still meet.
#[cfg(all(feature = "zerocopy", feature = "aligned"))]
#[test]
#[should_panic(expected = "overlapping DisjointMut")]
fn u8_buffer_rect_as_u16_still_collides_with_slice_as_u16() {
    let dm = bytes128();
    let _rect = dm
        .index_rect_mut_as::<u16>(0, 2, 2, 8)
        .expect("representable");
    // Elements 8..10 of u16 = bytes 16..20 = the rectangle's second row.
    let _range = dm.mut_slice_as::<_, u16>(8..10);
}

/// And the byte-buffer control: the gap is free.
#[cfg(all(feature = "zerocopy", feature = "aligned"))]
#[test]
fn u8_buffer_rect_as_u16_gap_is_free() {
    let dm = bytes128();
    let mut rect = dm
        .index_rect_mut_as::<u16>(0, 2, 2, 8)
        .expect("representable");
    let mut gap = dm.mut_slice_as::<_, u16>(2..8);
    rect.row_mut(1)[0] = 7;
    gap[0] = 9;
    drop(gap);
    drop(rect);
    assert_eq!(dm.slice_as::<_, u16>(8..9)[0], 7);
    assert_eq!(dm.slice_as::<_, u16>(2..3)[0], 9);
}
