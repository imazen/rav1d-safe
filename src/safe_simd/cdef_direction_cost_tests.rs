//! Compare direction and variance with the unchanged scalar reference.

use crate::include::common::bitdepth::{AsPrimitive, BPC, BitDepth, BitDepth8, BitDepth16};
use crate::include::dav1d::picture::{
    PictureThreading, Rav1dPictureDataComponent, Rav1dPictureDataComponentInner,
};
use crate::src::with_offset::WithOffset;
use archmage::SimdToken;
use zerocopy::IntoBytes;

#[test]
fn direction_costs_match_scalar_at_all_depths_and_stride_signs() {
    let _lock = crate::src::safe_simd::token_test_lock();
    let Some(token) = archmage::Desktop64::summon() else {
        return;
    };
    check(BitDepth8::new(()), token);
    check(BitDepth16::new(1023), token);
    check(BitDepth16::new(4095), token);
}

fn check<BD: BitDepth>(bd: BD, token: archmage::Desktop64) {
    const STRIDE: usize = 32;
    const ROWS: usize = 16;
    let max = bd.bitdepth_max().as_::<u32>();
    let ps = core::mem::size_of::<BD::Pixel>();
    let mut state = 0x8bad_f00d_5eed_cdefu64;
    let mut comparisons = 0;
    for negative in [false, true] {
        let stride = if negative {
            -(STRIDE as isize)
        } else {
            STRIDE as isize
        };
        let origin = (if negative { 13 } else { 2 }) * STRIDE + 5;
        for pattern in 0..(390 + 4096) {
            let mut pixels = vec![0u32.as_::<BD::Pixel>(); STRIDE * ROWS];
            for y in 0..8 {
                for x in 0..8 {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    let value = match pattern {
                        0..=255 => pattern * max / 255,
                        256 => {
                            if (x + y) % 2 == 0 {
                                0
                            } else {
                                max
                            }
                        }
                        257 => {
                            if (x + y) % 2 == 0 {
                                max
                            } else {
                                0
                            }
                        }
                        258 => x as u32 * max / 7,
                        259 => y as u32 * max / 7,
                        260 => (x + y) as u32 * max / 14,
                        261..=324 => {
                            if y * 8 + x == (pattern - 261) as usize {
                                max
                            } else {
                                0
                            }
                        }
                        325..=388 => {
                            if y * 8 + x == (pattern - 325) as usize {
                                0
                            } else {
                                max
                            }
                        }
                        _ => state as u32 & max,
                    };
                    pixels[origin.wrapping_add_signed(y as isize * stride) + x] = value.as_();
                }
            }
            let mut picture = Rav1dPictureDataComponent::from_parts(
                Rav1dPictureDataComponentInner::from_slice_copy(pixels.as_bytes()),
                stride * ps as isize,
            );
            picture.set_threading_policy(PictureThreading::new(2, 4));
            let img = WithOffset {
                data: &picture,
                offset: origin,
            };
            let gap_row = origin.wrapping_add_signed(if negative { stride } else { 0 });
            let _gap = picture.index_mut::<BD>(gap_row + 12);
            let mut expected_variance = u32::MAX;
            let expected = super::cdef_find_dir_scalar(img, &mut expected_variance, bd);
            let mut actual_variance = u32::MAX;
            let actual = match BD::BPC {
                BPC::BPC8 => super::cdef_find_dir_simd_8bpc(token, img, &mut actual_variance),
                BPC::BPC16 => super::cdef_find_dir_simd_16bpc(
                    token,
                    img,
                    &mut actual_variance,
                    bd.bitdepth() as u8,
                ),
            };
            assert_eq!(
                (actual, actual_variance),
                (expected, expected_variance),
                "pattern={pattern} max={max} negative={negative}"
            );
            comparisons += 1;
        }
    }
    assert_eq!(comparisons, 2 * (390 + 4096));
}
