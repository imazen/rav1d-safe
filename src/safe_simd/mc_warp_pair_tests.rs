//! Warp pair arithmetic against independent scalar integer dot products.

use crate::src::tables::dav1d_mc_warp_filter;
use archmage::{Desktop64, SimdToken, arcane};

#[arcane]
fn paired(token: Desktop64, bytes: &[u8], filters: [&[i8; 8]; 2]) -> [i32; 2] {
    super::warp_h_pair8(token, bytes, [0, 8], filters)
}

fn scalar(bytes: &[u8], filter: &[i8; 8]) -> i32 {
    bytes
        .iter()
        .zip(filter)
        .map(|(&p, &f)| i32::from(p) * i32::from(f))
        .sum()
}

#[test]
fn every_warp_filter_pair_matches_scalar_at_linear_extrema_and_random_pixels() {
    let _lock = crate::src::safe_simd::token_test_lock();
    let Some(token) = Desktop64::summon() else {
        return;
    };
    let mut state = 0x6a09_e667_f3bc_c909u64;
    let mut comparisons = 0;
    for (i, first) in dav1d_mc_warp_filter.iter().enumerate() {
        for (j, second) in dav1d_mc_warp_filter.iter().enumerate() {
            for pattern in 0..3 {
                let mut bytes = [0u8; 16];
                for (lane, filter) in [first, second].into_iter().enumerate() {
                    for tap in 0..8 {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        bytes[lane * 8 + tap] = match pattern {
                            0 => {
                                if filter[tap] > 0 {
                                    255
                                } else {
                                    0
                                }
                            }
                            1 => {
                                if filter[tap] < 0 {
                                    255
                                } else {
                                    0
                                }
                            }
                            _ => state as u8,
                        };
                    }
                }
                assert_eq!(
                    paired(token, &bytes, [first, second]),
                    [scalar(&bytes[..8], first), scalar(&bytes[8..], second)],
                    "filters={i},{j} pattern={pattern}"
                );
                comparisons += 1;
            }
        }
    }
    assert_eq!(comparisons, 193 * 193 * 3);
}

#[test]
fn warp_pair_byte_values_and_impulses_match_scalar() {
    let _lock = crate::src::safe_simd::token_test_lock();
    let Some(token) = Desktop64::summon() else {
        return;
    };
    for (i, filter) in dav1d_mc_warp_filter.iter().enumerate() {
        // Each input-byte value, complementary lanes, and each coefficient's
        // maximum impulse expose byte widening, sign extension and lane mixing.
        for value in 0..=255u8 {
            let mut bytes = [value; 16];
            bytes[8..].fill(255 - value);
            assert_eq!(
                paired(token, &bytes, [filter, filter]),
                [scalar(&bytes[..8], filter), scalar(&bytes[8..], filter)],
                "filter={i} byte={value}"
            );
        }
        for tap in 0..16 {
            let mut bytes = [0u8; 16];
            bytes[tap] = 255;
            assert_eq!(
                paired(token, &bytes, [filter, filter]),
                [scalar(&bytes[..8], filter), scalar(&bytes[8..], filter)],
                "filter={i} impulse={tap}"
            );
        }
        // Even arbitrary signed-byte coefficients and every intermediate i32
        // partial sum are bounded by 8 * 255 * 128, below i32::MAX.
        let bound = filter
            .iter()
            .map(|&f| i64::from(f).abs() * 255)
            .sum::<i64>();
        assert!(bound <= 8 * 255 * 128);
        assert!(bound < i64::from(i32::MAX));
    }
}
