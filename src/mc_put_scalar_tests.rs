//! Whole-picture MC put parity with signed destination and source rows.

use super::{prep_8tap_rust, prep_bilin_rust, put_8tap_rust, put_bilin_rust};
use crate::include::common::bitdepth::{AsPrimitive, BitDepth, BitDepth8, BitDepth16};
use crate::include::dav1d::picture::{
    PictureThreading, Rav1dPictureDataComponent, Rav1dPictureDataComponentInner,
};
use crate::src::levels::Filter2d;
use crate::src::with_offset::WithOffset;
use zerocopy::IntoBytes;

#[test]
fn mc_put_matches_original_scalar_with_signed_destination_rows() {
    let _lock = crate::src::safe_simd::token_test_lock();
    assert!(crate::src::cpu::summon_avx2().is_some());
    check(BitDepth8::new(()));
    check(BitDepth16::new(1023));
    check(BitDepth16::new(4095));
}

#[test]
fn mc_avx2_put_prep_match_scalar_with_signed_rows() {
    let _lock = crate::src::safe_simd::token_test_lock();
    let original = crate::src::cpu::rav1d_get_cpu_flags();
    struct RestoreMask(crate::src::cpu::CpuFlags);
    impl Drop for RestoreMask {
        fn drop(&mut self) {
            crate::src::cpu::rav1d_set_cpu_flags_mask(self.0.bits());
        }
    }
    // Detection is cached; restoring the available flags restores the same
    // effective mask for this process, including on a failing assertion.
    let _restore = RestoreMask(original);
    crate::src::cpu::rav1d_set_cpu_flags_mask(
        (original & !crate::src::cpu::CpuFlags::AVX512ICL).bits(),
    );
    assert!(crate::src::cpu::summon_avx2().is_some());
    assert!(crate::src::cpu::summon_avx512().is_none());
    check(BitDepth8::new(()));
    check(BitDepth16::new(1023));
    check(BitDepth16::new(4095));
}

fn check<BD: BitDepth>(bd: BD) {
    const STRIDE: usize = 192;
    const ROWS: usize = 160;
    let max = bd.bitdepth_max().as_::<i32>();
    let pixel_size = core::mem::size_of::<BD::Pixel>();
    let plane = |pixels: &[BD::Pixel], sign: isize, workers| {
        let mut picture = Rav1dPictureDataComponent::from_parts(
            Rav1dPictureDataComponentInner::from_slice_copy(pixels.as_bytes()),
            sign * STRIDE as isize * pixel_size as isize,
        );
        picture.set_threading_policy(PictureThreading::new(workers, 4));
        picture
    };
    let pixels: Vec<BD::Pixel> = (0..STRIDE * ROWS)
        .map(|i| ((((i * 73 + i / STRIDE * 19) ^ (i >> 3)) as i32) & max).as_())
        .collect();
    let initial = vec![0xa5.as_::<BD::Pixel>(); STRIDE * ROWS];
    let mut comparisons = 0;
    for source_sign in [1isize, -1] {
        let source = plane(&pixels, source_sign, 1);
        let src = WithOffset {
            data: &source,
            offset: (if source_sign < 0 { 144 } else { 8 }) * STRIDE + 8,
        };
        for destination_sign in [1isize, -1] {
            for workers in [1, 4] {
                for (w, h) in [(4, 4), (8, 8), (16, 8), (65, 9), (128, 128)] {
                    for filter_id in 0..10 {
                        let filter = Filter2d::from_repr(filter_id).unwrap();
                        for (mx, my) in [(0, 0), (1, 0), (0, 15), (1, 15)] {
                            let expected = plane(&initial, destination_sign, workers);
                            let actual = plane(&initial, destination_sign, workers);
                            let origin = (if destination_sign < 0 { 144 } else { 8 }) * STRIDE + 8;
                            let expected_at = WithOffset {
                                data: &expected,
                                offset: origin,
                            };
                            let actual_at = WithOffset {
                                data: &actual,
                                offset: origin,
                            };
                            match filter {
                                Filter2d::Bilinear => {
                                    put_bilin_rust(expected_at, src, w, h, mx, my, bd)
                                }
                                _ => put_8tap_rust(expected_at, src, w, h, mx, my, filter.hv(), bd),
                            }
                            assert!(crate::src::safe_simd::mc::mc_put_dispatch(
                                filter, actual_at, src, w as i32, h as i32, mx as i32, my as i32,
                                bd,
                            ));
                            let expected_bytes =
                                expected.dm().slice_as::<_, u8>(..initial.as_bytes().len());
                            let actual_bytes =
                                actual.dm().slice_as::<_, u8>(..initial.as_bytes().len());
                            assert_eq!(
                                &*actual_bytes,
                                &*expected_bytes,
                                "depth={} src={source_sign} dst={destination_sign} workers={workers} w={w} h={h} filter={filter_id} mx={mx} my={my}",
                                bd.bitdepth()
                            );
                            let mut expected_tmp = vec![i16::MIN; w * h + 7];
                            let mut actual_tmp = expected_tmp.clone();
                            match filter {
                                Filter2d::Bilinear => {
                                    prep_bilin_rust(&mut expected_tmp, src, w, h, mx, my, bd)
                                }
                                _ => prep_8tap_rust(
                                    &mut expected_tmp,
                                    src,
                                    w,
                                    h,
                                    mx,
                                    my,
                                    filter.hv(),
                                    bd,
                                ),
                            }
                            assert!(crate::src::safe_simd::mc::mct_prep_dispatch(
                                filter,
                                &mut actual_tmp,
                                src,
                                w as i32,
                                h as i32,
                                mx as i32,
                                my as i32,
                                bd,
                            ));
                            assert_eq!(
                                actual_tmp,
                                expected_tmp,
                                "prep depth={} src={source_sign} workers={workers} w={w} h={h} filter={filter_id} mx={mx} my={my}",
                                bd.bitdepth()
                            );
                            comparisons += 1;
                        }
                    }
                }
            }
        }
    }
    assert_eq!(comparisons, 2 * 2 * 2 * 5 * 10 * 4);
    println!(
        "{}-bit signed-row SIMD put/prep pairs: {comparisons}",
        bd.bitdepth()
    );
}
