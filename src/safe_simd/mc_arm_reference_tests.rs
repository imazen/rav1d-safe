//! ARM MC must leave unrelated reconstruction rows available while reading
//! exactly its interpolation window. Compare both outputs with scalar MC.

use crate::include::common::bitdepth::{AsPrimitive, BitDepth, BitDepth8, BitDepth16};
use crate::include::dav1d::picture::{Rav1dPictureDataComponent, Rav1dPictureDataComponentInner};
use crate::src::levels::Filter2d;
use crate::src::strided::Strided;
use archmage::SimdToken;
use zerocopy::IntoBytes;

#[test]
fn interpolation_reads_leave_unrelated_reconstruction_rows_available() {
    let _lock = crate::src::safe_simd::token_test_lock();
    // This is a native ARM regression; refusing the token would compare only
    // scalar fallbacks and would not exercise the reservation being repaired.
    assert!(archmage::Arm64::summon().is_some());
    for full_range in [false, true] {
        check_depth(BitDepth8::new(()), full_range);
        check_depth(BitDepth16::new(1023), full_range);
        check_depth(BitDepth16::new(4095), full_range);
    }
}

fn check_depth<BD: BitDepth>(bd: BD, full_range: bool) {
    const STRIDE: usize = 192;
    const ROWS: usize = 144;
    let max = bd.bitdepth_max().as_::<i32>();
    let mut state = 0x5eed_cdef_8bad_f00du64;
    let pixels: Vec<BD::Pixel> = (0..STRIDE * ROWS)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let value = if !full_range {
                ((i * 73 + i / 7) as i32) & max
            } else if i % 17 == 0 {
                0
            } else if i % 19 == 0 {
                max
            } else {
                (state as i32) & max
            };
            value.as_()
        })
        .collect();
    if full_range {
        assert_eq!(pixels.iter().map(|&p| p.as_::<i32>()).min(), Some(0));
        assert_eq!(pixels.iter().map(|&p| p.as_::<i32>()).max(), Some(max));
    }
    let pixel_size = core::mem::size_of::<BD::Pixel>();
    let picture = Rav1dPictureDataComponent::from_parts(
        Rav1dPictureDataComponentInner::from_slice_copy(pixels.as_bytes()),
        (STRIDE * pixel_size) as isize,
    );
    let source = picture.with_offset::<BD>() + (4 * STRIDE + 8);
    // Last row is outside every source window below. The old full_guard
    // refused this live reconstruction write before reaching the kernel.
    let _reconstruction = picture.index_mut::<BD>((ROWS - 1) * STRIDE);
    for (w, h) in [
        (2, 2),
        (4, 8),
        (8, 4),
        (7, 9),
        (17, 5),
        (16, 16),
        (127, 128),
        (128, 128),
    ] {
        for filter in [
            Filter2d::Regular8Tap,
            Filter2d::RegularSmooth8Tap,
            Filter2d::RegularSharp8Tap,
            Filter2d::SmoothRegular8Tap,
            Filter2d::Smooth8Tap,
            Filter2d::SmoothSharp8Tap,
            Filter2d::SharpRegular8Tap,
            Filter2d::SharpSmooth8Tap,
            Filter2d::Sharp8Tap,
            Filter2d::Bilinear,
        ] {
            let filter_id = filter as u8;
            for mx in 0..16 {
                for my in 0..16 {
                    let mut put: Vec<BD::Pixel> = vec![0i32.as_(); w * h];
                    let output_pixels = (w * h * pixel_size).div_ceil(64) * 64 / pixel_size;
                    let mut expected: Vec<BD::Pixel> = vec![0i32.as_(); output_pixels];
                    let output = Rav1dPictureDataComponent::wrap_buf::<BD>(&mut expected, w);
                    assert!(super::mc_put_dispatch_inner::<BD>(
                        filter,
                        put.as_mut_bytes(),
                        0,
                        (w * pixel_size) as isize,
                        source,
                        w as i32,
                        h as i32,
                        mx,
                        my,
                        bd,
                    ));
                    let dst = output.with_offset::<BD>();
                    let mut prep = vec![0i16; w * h];
                    let mut expected_prep = vec![0i16; w * h];
                    assert!(super::mct_prep_dispatch::<BD>(
                        filter, &mut prep, source, w as i32, h as i32, mx, my, bd,
                    ));
                    if filter == Filter2d::Bilinear {
                        crate::src::mc::put_bilin_rust::<BD>(
                            dst,
                            source,
                            w,
                            h,
                            mx as usize,
                            my as usize,
                            bd,
                        );
                        crate::src::mc::prep_bilin_rust::<BD>(
                            &mut expected_prep,
                            source,
                            w,
                            h,
                            mx as usize,
                            my as usize,
                            bd,
                        );
                    } else {
                        crate::src::mc::put_8tap_rust::<BD>(
                            dst,
                            source,
                            w,
                            h,
                            mx as usize,
                            my as usize,
                            filter.hv(),
                            bd,
                        );
                        crate::src::mc::prep_8tap_rust::<BD>(
                            &mut expected_prep,
                            source,
                            w,
                            h,
                            mx as usize,
                            my as usize,
                            filter.hv(),
                            bd,
                        );
                    }
                    let actual = output.dm().slice_as::<_, BD::Pixel>(..w * h);
                    assert_eq!(
                        put.as_bytes(),
                        actual.as_bytes(),
                        "put {w}x{h} filter={filter_id} {mx},{my} max={max} full_range={full_range}"
                    );
                    assert_eq!(
                        prep, expected_prep,
                        "prep {w}x{h} filter={filter_id} {mx},{my} max={max} full_range={full_range}"
                    );
                }
            }
        }
    }
}
