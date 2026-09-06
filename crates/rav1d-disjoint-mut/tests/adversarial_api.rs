//! Safe-client attacks. The geometry oracle enumerates elements; it shares no
//! hull arithmetic or overlap predicates with the implementation. Run natively
//! and under both Miri models (including with no default features).

use rav1d_disjoint_mut::DisjointMut;
use std::panic::{AssertUnwindSafe, catch_unwind};

fn rejected(f: impl FnOnce()) -> bool {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(()) => false,
        Err(e) => {
            let msg = e
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| e.downcast_ref::<&str>().copied())
                .unwrap_or("");
            assert!(
                msg.contains("overlapping DisjointMut"),
                "unexpected rejection: {msg}"
            );
            true
        }
    }
}

#[test]
fn rectangles_and_ranges_match_an_element_set_oracle_in_both_orders() {
    let geometries = [(0, 2, 3, 8), (3, 3, 2, 8), (24, 2, 3, -8), (16, 8, 2, -8)];
    let ranges = [
        0..0,
        0..1,
        1..3,
        2..8,
        8..10,
        10..16,
        16..25,
        24..32,
        31..32,
    ];
    let mut accepted = 0;
    let mut refused = 0;
    for (start, width, rows, stride) in geometries {
        let mut occupied = [false; 32];
        for row in 0..rows {
            let base = (start as isize + row as isize * stride) as usize;
            for col in 0..width {
                occupied[base + col] = true;
            }
        }
        for range in &ranges {
            let conflict = range.clone().any(|i| occupied[i]);
            for rect_first in [true, false] {
                for mutable_range in [true, false] {
                    let mut dm = DisjointMut::new(vec![0u16; 32]);
                    dm.declare_row_stride(8);
                    let was_rejected = if rect_first {
                        let mut rect = dm
                            .index_rect_mut(start, width, rows, stride)
                            .expect("geometry");
                        let result = rejected(|| {
                            if mutable_range {
                                dm.index_mut(range.clone()).fill(9);
                            } else {
                                assert!(dm.index(range.clone()).iter().all(|&x| x == 0));
                            }
                        });
                        // Reuse the old reference after the competing attempt:
                        // Miri catches an invalidating temporary hull reference.
                        for row in 0..rows {
                            rect.row_mut(row).fill(7);
                        }
                        result
                    } else if mutable_range {
                        let mut held = dm.index_mut(range.clone());
                        let result = rejected(|| {
                            let mut rect = dm
                                .index_rect_mut(start, width, rows, stride)
                                .expect("geometry");
                            for row in 0..rows {
                                rect.row_mut(row).fill(7);
                            }
                        });
                        held.fill(9);
                        result
                    } else {
                        let held = dm.index(range.clone());
                        let result = rejected(|| {
                            let mut rect = dm
                                .index_rect_mut(start, width, rows, stride)
                                .expect("geometry");
                            for row in 0..rows {
                                rect.row_mut(row).fill(7);
                            }
                        });
                        assert!(held.iter().all(|&x| x == 0));
                        result
                    };
                    assert_eq!(
                        was_rejected, conflict,
                        "start={start} width={width} rows={rows} stride={stride} range={range:?} rect_first={rect_first} mutable_range={mutable_range}"
                    );
                    if was_rejected {
                        refused += 1;
                    } else {
                        accepted += 1;
                    }
                }
            }
        }
    }
    assert!(
        accepted > 0 && refused > 0,
        "both oracle outcomes must be exercised"
    );
}

#[test]
fn forgotten_guard_survives_unrelated_slot_reuse() {
    let dm = DisjointMut::new(vec![0u8; 32]);
    std::mem::forget(dm.index_mut(8..16));
    for value in 0..32 {
        dm.index_mut(0..8).fill(value);
        assert!(rejected(|| {
            drop(dm.index(10..11));
        }));
    }
    assert_eq!(&*dm.index(0..8), &[31; 8]);
}

#[test]
fn moved_borrowed_storage_and_guards_keep_their_reference_authority() {
    let mut storage = [0u32; 32];
    let dm = DisjointMut::new(&mut storage[..]);
    let relocated = Box::new(dm);
    let mut left = relocated.index_mut(..16);
    let mut right = relocated.index_mut(16..);
    let raw_identity = relocated.as_mut_ptr(); // metadata only while guards live
    assert!(!raw_identity.is_null());
    left.fill(1);
    right.fill(2);
    let mut moved = Box::new(left);
    moved[0] = 3;
    assert_eq!(right[0], 2);
    drop(moved);
    drop(right);
    drop(relocated);
    assert_eq!(storage[0], 3);
    assert_eq!(storage[16], 2);
}

#[test]
fn extreme_rectangle_geometry_never_wraps_into_an_accepted_view() {
    let mut dm = DisjointMut::new(vec![0u16; 32]);
    dm.declare_row_stride(8);
    for (start, width, rows, stride) in [
        (usize::MAX, 1, 1, 8),
        (0, usize::MAX, 2, 8),
        (0, 1, usize::MAX, 8),
        (0, 1, 2, isize::MIN),
        (0, 1, 2, -8),
        (31, 2, 1, 8),
    ] {
        assert!(dm.index_rect_mut(start, width, rows, stride).is_none());
    }
    dm.index_mut(..).fill(5);
    assert!(dm.index(..).iter().all(|&x| x == 5));
}

#[cfg(all(feature = "zerocopy", feature = "aligned"))]
#[test]
fn typed_views_and_byte_views_agree_at_inclusive_endpoints() {
    use rav1d_disjoint_mut::align::AlignedVec64;
    let mut storage = AlignedVec64::<u8>::new();
    storage.resize(64, 0);
    let dm = DisjointMut::new(storage);
    let mut typed = dm.mut_slice_as::<_, u32>(1..=2); // bytes 4..12
    for i in 4..12 {
        assert!(rejected(|| {
            drop(dm.index(i));
        }));
    }
    dm.index_mut(3..4)[0] = 3;
    dm.index_mut(12..13)[0] = 12;
    typed[0] = 0x0102_0304;
    typed[1] = 0x0506_0708;
    drop(typed);
    assert_eq!(dm.slice_as::<_, u32>(1..=2)[1], 0x0506_0708);
}

#[cfg(feature = "std")]
#[test]
fn unwind_with_a_live_guard_poisoning_cannot_be_cleared_by_hints() {
    let dm = DisjointMut::new(vec![0u8; 32]);
    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut guard = dm.index_mut(0..1);
        guard[0] = 9;
        panic!("interrupt a mutation");
    }));
    assert!(result.is_err());
    rav1d_disjoint_mut::set_parallelism(8);
    rav1d_disjoint_mut::set_parallelism(1);
    assert!(catch_unwind(AssertUnwindSafe(|| dm.index(0..1))).is_err());
}
