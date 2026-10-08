//! Whole-buffer warp put/prep comparisons with the original scalar kernels.

use super::{warp_affine_8x8_rust, warp_affine_8x8t_rust};
use crate::include::common::bitdepth::{AsPrimitive, BitDepth, BitDepth8, BitDepth16};
use crate::include::dav1d::picture::{
    PictureThreading, Rav1dPictureDataComponent, Rav1dPictureDataComponentInner,
};
use crate::src::safe_simd::mc::{warp8x8_dispatch, warp8x8t_dispatch};
use crate::src::with_offset::WithOffset;
use zerocopy::IntoBytes;

const STRIDE: usize = 64;
const ROWS: usize = 32;

fn plane(bytes: &[u8], stride: isize) -> Rav1dPictureDataComponent {
    let mut pic = Rav1dPictureDataComponent::from_parts(
        Rav1dPictureDataComponentInner::from_slice_copy(bytes),
        stride,
    );
    pic.set_threading_policy(PictureThreading::new(2, 4));
    pic
}

#[test]
fn warp_put_prep_match_scalar_with_signed_source_rows_and_all_filter_indices() {
    let _lock = crate::src::safe_simd::token_test_lock();
    if crate::src::cpu::summon_avx2().is_none() {
        return;
    }
    let affine = [
        [0, 0, 0, 0],
        [128, 0, 0, 128],
        [-128, 0, 0, -128],
        [128, 64, -64, 128],
        [-128, -64, 64, -128],
        [4096, 0, 0, 4096],
        [-4096, 0, 0, -4096],
        [1024, 2048, -1024, -2048],
    ];
    let phases = [-16384, -1024, -512, 0, 511, 1023, 16384];
    let mut cases = Vec::new();
    for abcd in affine {
        for mx in phases {
            for my in phases {
                cases.push((abcd, mx, my));
            }
        }
    }
    // Zero affine deltas make every table row valid, including both endpoints.
    for index in 0..193 {
        let phase = (index - 64) * 1024;
        cases.extend([
            ([0; 4], phase, 0),
            ([0; 4], 0, phase),
            ([0; 4], phase, phase),
        ]);
    }
    assert_eq!(cases.len(), 8 * 7 * 7 + 193 * 3);
    let bd = BitDepth8::new(());
    let mut comparisons = 0;
    for negative in [false, true] {
        let stride = if negative {
            -(STRIDE as isize)
        } else {
            STRIDE as isize
        };
        let origin = (if negative { 24 } else { 8 }) * STRIDE + 8;
        for pattern in 0..4 {
            let pixels: Vec<u8> = (0..ROWS * STRIDE)
                .map(|i| match pattern {
                    0 => 0,
                    1 => 255,
                    2 => {
                        if (i / STRIDE + i % STRIDE) % 2 == 0 {
                            0
                        } else {
                            255
                        }
                    }
                    _ => ((i * 73 + i / STRIDE * 19) ^ (i >> 3)) as u8,
                })
                .collect();
            let source = plane(&pixels, stride);
            let src = WithOffset {
                data: &source,
                offset: origin,
            };
            // Outside the complete 15-row reference hull for either sign.
            let held = source.index_mut::<BitDepth8>(0);
            for &(abcd, mx, my) in &cases {
                for y in 0..15 {
                    for x in 0..8 {
                        let index = 64
                            + ((mx + y * i32::from(abcd[1]) + x * i32::from(abcd[0]) + 512) >> 10);
                        assert!((0..193).contains(&index));
                    }
                }
                for y in 0..8 {
                    for x in 0..8 {
                        let index = 64
                            + ((my + y * i32::from(abcd[3]) + x * i32::from(abcd[2]) + 512) >> 10);
                        assert!((0..193).contains(&index));
                    }
                }
                let initial = [0xa5u8; STRIDE * 16];
                let expected = plane(&initial, STRIDE as isize);
                let actual = plane(&initial, STRIDE as isize);
                let expected_at = WithOffset {
                    data: &expected,
                    offset: 3 * STRIDE + 7,
                };
                let actual_at = WithOffset {
                    data: &actual,
                    offset: 3 * STRIDE + 7,
                };
                warp_affine_8x8_rust(expected_at, src, &abcd, mx, my, bd);
                assert!(warp8x8_dispatch(actual_at, src, &abcd, mx, my, bd));
                let expected_pixels = expected.dm().slice_as::<_, u8>(..initial.len());
                let actual_pixels = actual.dm().slice_as::<_, u8>(..initial.len());
                assert_eq!(
                    &*actual_pixels, &*expected_pixels,
                    "put negative={negative} pattern={pattern} abcd={abcd:?} mx={mx} my={my}"
                );
                let mut expected_tmp = [i16::MIN; 13 * 8];
                let mut actual_tmp = expected_tmp;
                warp_affine_8x8t_rust(&mut expected_tmp, 13, src, &abcd, mx, my, bd);
                assert!(warp8x8t_dispatch(
                    &mut actual_tmp,
                    13,
                    src,
                    &abcd,
                    mx,
                    my,
                    bd
                ));
                assert_eq!(
                    actual_tmp, expected_tmp,
                    "prep negative={negative} pattern={pattern} abcd={abcd:?} mx={mx} my={my}"
                );
                comparisons += 1;
            }
            drop(held);
            let unchanged = source.dm().slice_as::<_, u8>(..pixels.len());
            assert_eq!(&*unchanged, &pixels);
        }
    }
    assert_eq!(comparisons, 2 * 4 * (8 * 7 * 7 + 193 * 3));
}

#[test]
fn warp_put_matches_scalar_with_negative_destination_rows() {
    let _lock = crate::src::safe_simd::token_test_lock();
    let token = crate::src::cpu::summon_avx2();
    assert!(token.is_some(), "native negative-row oracle needs AVX2");
    let bd = BitDepth8::new(());
    let pixels: Vec<u8> = (0..ROWS * STRIDE)
        .map(|i| ((i * 73 + i / STRIDE * 19) ^ (i >> 3)) as u8)
        .collect();
    let source = plane(&pixels, STRIDE as isize);
    let src = WithOffset {
        data: &source,
        offset: 8 * STRIDE + 8,
    };
    let held = source.index_mut::<BitDepth8>(0);
    for workers in [1, 4] {
        for (abcd, mx, my) in [([0; 4], 0, 0), ([128, 64, -64, 128], -512, 1023)] {
            let initial = [0xa5u8; STRIDE * 16];
            let mut expected = plane(&initial, -(STRIDE as isize));
            let mut actual = plane(&initial, -(STRIDE as isize));
            let policy = PictureThreading::new(workers, 4);
            expected.set_threading_policy(policy);
            actual.set_threading_policy(policy);
            let origin = 12 * STRIDE + 7;
            let expected_at = WithOffset {
                data: &expected,
                offset: origin,
            };
            let actual_at = WithOffset {
                data: &actual,
                offset: origin,
            };
            warp_affine_8x8_rust(expected_at, src, &abcd, mx, my, bd);
            assert!(warp8x8_dispatch(actual_at, src, &abcd, mx, my, bd));
            let expected_pixels = expected.dm().slice_as::<_, u8>(..initial.len());
            let actual_pixels = actual.dm().slice_as::<_, u8>(..initial.len());
            assert_eq!(
                &*actual_pixels, &*expected_pixels,
                "negative destination: workers={workers} abcd={abcd:?} mx={mx} my={my}"
            );
        }
    }
    drop(held);
}

#[test]
fn warp_put_high_depth_matches_scalar_with_signed_destination_rows() {
    let _lock = crate::src::safe_simd::token_test_lock();
    assert!(crate::src::cpu::summon_avx2().is_some());
    check_high_depth_destination(BitDepth16::new(1023));
    check_high_depth_destination(BitDepth16::new(4095));
}

fn check_high_depth_destination<BD: BitDepth>(bd: BD) {
    let max = bd.bitdepth_max().as_::<i32>();
    let pixel_size = core::mem::size_of::<BD::Pixel>();
    let make_plane = |pixels: &[BD::Pixel], stride: isize, workers| {
        let mut picture = Rav1dPictureDataComponent::from_parts(
            Rav1dPictureDataComponentInner::from_slice_copy(pixels.as_bytes()),
            stride * pixel_size as isize,
        );
        picture.set_threading_policy(PictureThreading::new(workers, 4));
        picture
    };
    let mut comparisons = 0;
    for source_sign in [1isize, -1] {
        for pattern in 0..4 {
            let pixels: Vec<BD::Pixel> = (0..ROWS * STRIDE)
                .map(|i| {
                    let value = match pattern {
                        0 => 0,
                        1 => max,
                        2 => {
                            if (i / STRIDE + i % STRIDE) % 2 == 0 {
                                0
                            } else {
                                max
                            }
                        }
                        _ => (((i * 73 + i / STRIDE * 19) ^ (i >> 3)) as i32) & max,
                    };
                    value.as_()
                })
                .collect();
            let source = make_plane(&pixels, source_sign * STRIDE as isize, 2);
            let src = WithOffset {
                data: &source,
                offset: (if source_sign < 0 { 24 } else { 8 }) * STRIDE + 8,
            };
            let held = source.index_mut::<BD>(0);
            for workers in [1, 4] {
                for (abcd, mx, my) in [([0; 4], 0, 0), ([128, 64, -64, 128], -512, 1023)] {
                    let initial = vec![(max / 3).as_(); STRIDE * 16];
                    let expected = make_plane(&initial, -(STRIDE as isize), workers);
                    let actual = make_plane(&initial, -(STRIDE as isize), workers);
                    let origin = 12 * STRIDE + 7;
                    warp_affine_8x8_rust(
                        WithOffset {
                            data: &expected,
                            offset: origin,
                        },
                        src,
                        &abcd,
                        mx,
                        my,
                        bd,
                    );
                    assert!(warp8x8_dispatch(
                        WithOffset {
                            data: &actual,
                            offset: origin
                        },
                        src,
                        &abcd,
                        mx,
                        my,
                        bd
                    ));
                    let expected_pixels = expected.dm().slice_as::<_, BD::Pixel>(..initial.len());
                    let actual_pixels = actual.dm().slice_as::<_, BD::Pixel>(..initial.len());
                    assert_eq!(
                        actual_pixels.as_bytes(),
                        expected_pixels.as_bytes(),
                        "max={max} source_sign={source_sign} pattern={pattern} workers={workers} abcd={abcd:?} mx={mx} my={my}"
                    );
                    comparisons += 1;
                }
            }
            drop(held);
            let unchanged = source.dm().slice_as::<_, BD::Pixel>(..pixels.len());
            assert_eq!(unchanged.as_bytes(), pixels.as_bytes());
        }
    }
    assert_eq!(comparisons, 2 * 4 * 2 * 2);
}
