//! LAYOUT-NOISE CONTROLS: never-executed `__text`, emitted from a module that
//! is FAR from the loop filter, so an A/B can separate "the binary grew" from
//! "the hot loop-filter symbols moved".
//!
//! # Why this exists
//!
//! `docs/RECT_RECORDS.md` §5d measured **+1.0% to +1.3% wall at t=1 on
//! `v4k8tile`, 0 of 11 rounds below 1.000 in two sessions**, in an arm whose new
//! code CANNOT execute at t=1, and attributed it to code size by elimination.
//! Its layout control was the same source built in a second worktree, which
//! differs only in embedded path strings and moves nothing — so the band looked
//! like ±0.1% and the effect looked specific to the mechanism.
//!
//! The near control (`loopfilter::text_pad`) refutes that: 4.8 KiB of dead text
//! in the loop-filter module, with **every hot symbol's instruction stream
//! byte-identical** (`scripts/perf/text_symbol_diff.sh`) and only its ADDRESS
//! changed, reproduces the whole effect. This far control is the follow-up
//! question — does the cost need the hot symbols to MOVE, or does any growth of
//! the binary do it? Check with `scripts/perf/text_layout_diff.py` which symbols
//! actually moved in each arm before reading the clock.
//!
//! Measurement only: absent from `default` and from every published feature.

/// One ~600-byte unit of dead text. `K` only forces a distinct
/// monomorphisation per slot.
#[inline(never)]
pub(crate) extern "C" fn unit<const K: usize>(x: &mut [u64; 32]) -> u64 {
    let mut acc = K as u64;
    for i in 0..32 {
        acc = acc
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(x[i] ^ (i as u64));
        x[i] = acc;
        acc ^= acc >> 29;
    }
    acc
}

/// `#[used]`-anchored so nothing can call them and nothing can eliminate them.
#[used]
static PAD_FAR: [extern "C" fn(&mut [u64; 32]) -> u64; 8] = [
    unit::<500>,
    unit::<501>,
    unit::<502>,
    unit::<503>,
    unit::<504>,
    unit::<505>,
    unit::<506>,
    unit::<507>,
];
