//! Loop restoration on aarch64: **no safe-SIMD tier**, by measurement.
//!
//! # What used to be here
//!
//! 1,531 lines that opened `//! Safe ARM NEON implementations for Loop
//! Restoration`, imported `core::arch::aarch64::*`, and contained **zero
//! aarch64 intrinsic calls**. The bodies were a hand-written scalar
//! re-implementation of `src/looprestoration.rs` — a second copy of the Wiener
//! filter, `boxsum3`/`boxsum5`, and the self-guided filter, at both bit depths
//! — and `lr_filter_dispatch` returned `true` unconditionally, so every loop
//! restoration call on aarch64 ran the copy instead of the reference.
//!
//! # Why it is gone
//!
//! It was slower than the code it shadowed. Measured 2026-08-07 on an Apple M4
//! Pro (8P+4E, macOS 26.5.2), release, default features, no `nice` on a timed
//! run, `examples/profile_ivf` one process per cell, arm order rotated per
//! round, median of 9 (`benchmarks/lr_arm_vs_reference_2026-08-07.tsv`):
//!
//! | vector | duplicate | reference | ratio |
//! |---|---|---|---|
//! | `8-bit/data/00001147` (LR = 33.4 Mpx over 3 frames, its largest kernel) | 204.42 ms/frame | 192.83 | **1.060** |
//! | `10-bit/issues/318_tx_4x4` (LR = 76% of all kernel pixel work) | 9.7187 | 9.7243 | 0.999 |
//!
//! Whole-decode, not kernel-local: the 8bpc duplicate cost 6.0% of the entire
//! frame. The A/B is the `__ablate` switch, which makes this dispatcher return
//! `false` so `lr_filter_direct` falls through to the generic reference — so
//! the two arms are the same binary and differ only in which implementation
//! runs.
//!
//! Deleting it is free of bit-exactness risk in the direction that matters:
//! `src/looprestoration.rs` is the conformance oracle. The corpus confirms it —
//! 766/766 dav1d-test-data vectors pass with byte-identical per-vector MD5s
//! before and after, which is also the proof that the duplicate had been made
//! bit-exact (issue #14) and therefore bought nothing at all.
//!
//! # Which vectors this even affects
//!
//! `ROADMAP_SIMD_PORTING.md` recorded loop restoration at **0.0 ms/frame** and
//! warned not to port it on the strength of its line count. That number is a
//! property of the vectors, not of the kernel: LR is switched off in
//! `v4k_8tile{,_10b}`. A corpus-wide scan with the new
//! `md5_inventory --activity` counters (needs `--features __ablate`) finds LR
//! active in **696 of 768** vectors, and `sample` on `318_tx_4x4` puts
//! `selfguided_filter` at 9.5% and the Wiener filter at 2.0% of decode
//! self-time. So the kernel is worth a real NEON tier — it just cannot be
//! measured on the 4K stills the gap-to-dav1d table uses.
//!
//! # If you are here to write that tier
//!
//! Port `src/looprestoration.rs`, not a copy of it, and start from the two
//! things the profile blames: `selfguided_filter`'s per-pixel
//! `.get(..).unwrap_or(0)` neighbour gather, and the ~300 KB of stack scratch
//! (`sumsq` as `i64`, `sum`, `dst0`, `dst1`) that every restoration unit zeroes
//! on entry. `src/itx.rs`'s `__simd_test` differential has a loop-restoration
//! twin in `lr_filter_direct`, which is the per-call oracle to gate on.

#![cfg_attr(not(feature = "unchecked"), forbid(unsafe_code))]
#![cfg_attr(feature = "unchecked", deny(unsafe_code))]

use std::ffi::c_int;

use crate::include::common::bitdepth::BitDepth;
use crate::include::common::bitdepth::LeftPixelRow;
use crate::include::dav1d::picture::PicOffset;
use crate::src::align::AlignedVec64;
use crate::src::disjoint_mut::DisjointMut;
use crate::src::looprestoration::{LooprestorationParams, LrEdgeFlags};

/// Always `false`: there is no aarch64 loop-restoration tier, so the caller
/// runs the generic reference in `src/looprestoration.rs`.
///
/// Kept as a function rather than deleted at the call site because it is also
/// where the `__ablate` activity counter records how much loop restoration a
/// bitstream actually asks for — the measurement that showed the 4K gap
/// vectors do none at all.
#[cfg(target_arch = "aarch64")]
pub fn lr_filter_dispatch<BD: BitDepth>(
    _variant: usize,
    _dst: PicOffset,
    _left: &[LeftPixelRow<BD::Pixel>],
    _lpf: &DisjointMut<AlignedVec64<u8>>,
    _lpf_off: isize,
    w: c_int,
    h: c_int,
    _params: &LooprestorationParams,
    _edges: LrEdgeFlags,
    _bd: BD,
) -> bool {
    crate::src::ablate::note(
        crate::src::ablate::Family::LoopRestoration,
        (w as i64 * h as i64).unsigned_abs(),
    );
    false
}
