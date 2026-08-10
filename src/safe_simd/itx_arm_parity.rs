//! Differential parity for the aarch64 inverse transforms, against the
//! generic scalar reference in `src/itx.rs`.
//!
//! ## Why this exists
//!
//! `src/itx.rs` already dual-computes NEON against scalar under the
//! `__simd_test` feature, which is how the 293 `ITX_MISMATCH` vectors in
//! `benchmarks/aarch64_md5_attribution_2026-08-07.meta` were found. That
//! instrument needs a full corpus decode to say anything, and it only ever
//! sees the (size, type, eob) triples a particular bitstream happens to
//! contain. This module drives the same comparison directly, over a swept
//! parameter space, in ~1 second.
//!
//! The oracle is `crate::src::itx::itxfm_add_scalar_fallback` — the actual
//! reference, not a transcription of it. `--ablate itx` proves that reference
//! conformant: with every SIMD family switched off, all 766 dav1d-test-data
//! vectors match dav1d's published MD5s.
//!
//! ## What makes the sweep legal rather than merely random
//!
//! Several kernels skip whole row or column groups based on `eob`, which is
//! only sound because a real bitstream cannot place a nonzero coefficient
//! past scan position `eob`. A test that fills the coefficient array densely
//! and then passes a small `eob` would report divergence the decoder can
//! never hit. So coefficients here are placed at exactly the positions
//! scan-reachable for the given `eob`, using the same mapping
//! `decode_coefs_class` uses:
//!
//!   * `TxClass::TwoD` — position `dav1d_scans[tx][i]`,
//!   * `TxClass::H`    — position `i` (the class-H `rc` IS the scan index),
//!   * `TxClass::V`    — position `(i & (w-1)) * h + (i >> log2 w)`.
//!
//! With that, a kernel whose skip threshold is too aggressive drops
//! coefficients a legal stream can carry, and the test fails — which is
//! exactly the 16x16 `H_DCT` defect (dav1d's `def_fn_16x16 dct, identity`
//! uses `eob_half = 8`; the port hardcoded 36 for every type).

#![cfg(all(test, target_arch = "aarch64", not(feature = "asm")))]

use crate::include::common::bitdepth::{BitDepth, BitDepth8, BitDepth16};
use crate::include::dav1d::picture::Rav1dPictureDataComponent;
use crate::src::levels::{self, TxClass, TxfmSize, TxfmType};
use crate::src::scan::dav1d_scans;
use crate::src::tables::dav1d_tx_type_class;

/// xorshift64*, so any failure reproduces from its seed.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn in_range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next() % ((hi - lo + 1) as u64)) as i32
    }
}

fn log2(n: usize) -> u32 {
    n.trailing_zeros()
}

/// Coefficient positions a legal bitstream can make nonzero at this `eob`.
///
/// Mirrors `src/recon.rs::decode_coefs_class`'s `eob -> rc` derivation.
fn reachable_positions(tx: TxfmSize, tx_type: TxfmType, eob: usize) -> Vec<usize> {
    let (w, h) = tx.to_wh();
    let (sw, sh) = (w.min(32), h.min(32));
    match dav1d_tx_type_class[tx_type as usize] {
        TxClass::TwoD => {
            let scan = dav1d_scans[tx as usize];
            (0..=eob).map(|i| scan[i].get() as usize).collect()
        }
        // `rc = eob`: the class-H scan index and the coefficient index coincide.
        TxClass::H => (0..=eob).collect(),
        // `rc = (eob & (4*sw-1)) << (lh+2) | (eob >> (lw+2))`.
        TxClass::V => (0..=eob)
            .map(|i| (i & (sw - 1)) * sh + (i >> log2(sw)))
            .collect(),
    }
}

/// Run one (size, type, eob, coefficients, pixels) cell through both paths.
///
/// Returns `(neon_pixels, scalar_pixels, neon_took_the_simd_path)`. The third
/// element is the liveness signal: `itxfm_add_dispatch` returns `false` for
/// any shape it has no kernel for, and without checking it this test would
/// happily compare the scalar reference against itself.
fn run_cell(
    tx: TxfmSize,
    tx_type: TxfmType,
    eob: i32,
    coeff: &[i16],
    pixels: &[u8],
    stride: usize,
) -> (Vec<u8>, Vec<u8>, bool) {
    let bd = BitDepth8::new(());

    let run = |simd: bool| -> (Vec<u8>, bool) {
        let mut px = pixels.to_vec();
        let mut cf = coeff.to_vec();
        let mut out = vec![0u8; px.len()];
        let handled = {
            let comp = Rav1dPictureDataComponent::wrap_buf::<BitDepth8>(&mut px, stride);
            let mut dst = crate::src::owned_recon::ReconDst::Pic(comp.with_offset::<BitDepth8>());
            let handled = if simd {
                crate::src::safe_simd::itx_arm::itxfm_add_dispatch::<BitDepth8>(
                    tx as usize,
                    tx_type as usize,
                    &mut dst,
                    &mut cf,
                    eob,
                    bd,
                )
            } else {
                crate::src::itx::itxfm_add_scalar_fallback::<BitDepth8>(
                    tx as usize,
                    tx_type,
                    &mut dst,
                    &mut cf,
                    eob,
                    bd,
                );
                true
            };
            comp.copy_pixels_to::<BitDepth8>(&mut out);
            handled
        };
        (out, handled)
    };

    let (neon, handled) = run(true);
    let (scalar, _) = run(false);
    (neon, scalar, handled)
}

/// Result accumulator: which parameter cells ran, which diverged.
#[derive(Default)]
struct Report {
    cells: usize,
    live: usize,
    bad: Vec<String>,
    first: Option<String>,
}

impl Report {
    fn record(&mut self, label: &str, live: bool, ok: bool, detail: impl FnOnce() -> String) {
        self.cells += 1;
        if live {
            self.live += 1;
        }
        if !ok {
            if self.first.is_none() {
                self.first = Some(format!("{label}: {}", detail()));
            }
            if !self.bad.iter().any(|b| b == label) {
                self.bad.push(label.to_string());
            }
        }
    }

    /// Fail on any divergence, AND on a run that never reached a NEON kernel.
    ///
    /// Both halves of the liveness gate matter. `cells >= min_live` catches a
    /// sweep that stopped enumerating; `live == cells` catches the subtler
    /// one — `itxfm_add_dispatch` returning `false` for a shape, which would
    /// silently turn this into scalar-vs-scalar and go green forever.
    fn finish(self, what: &str, min_live: usize) {
        assert!(
            self.cells >= min_live,
            "{what}: only {} parameter cells ran (expected >= {min_live}) — \
             the sweep is not reaching the kernels",
            self.cells
        );
        assert_eq!(
            self.live,
            self.cells,
            "{what}: {} of {} cells did NOT take the NEON path. \
             `itxfm_add_dispatch` returned false, so those cells compared the \
             scalar reference against itself and proved nothing.",
            self.cells - self.live,
            self.cells
        );
        assert!(
            self.bad.is_empty(),
            "{what}: {} of {} parameter cells diverge from the scalar reference.\n  \
             first: {}\n  cells: {:?}",
            self.bad.len(),
            self.cells,
            self.first.unwrap_or_default(),
            self.bad
        );
    }
}

/// The (size, type) pairs `itxfm_add_dispatch` wires to a NEON kernel at 8bpc.
///
/// WHT_WHT is excluded: it is lossless-only and has no `eob` skip structure.
fn wired_cells() -> Vec<(TxfmSize, TxfmType)> {
    use TxfmSize::*;
    let all16: [TxfmType; 16] = [
        levels::DCT_DCT,
        levels::ADST_DCT,
        levels::DCT_ADST,
        levels::ADST_ADST,
        levels::FLIPADST_DCT,
        levels::DCT_FLIPADST,
        levels::FLIPADST_FLIPADST,
        levels::ADST_FLIPADST,
        levels::FLIPADST_ADST,
        levels::IDTX,
        levels::V_DCT,
        levels::H_DCT,
        levels::V_ADST,
        levels::H_ADST,
        levels::V_FLIPADST,
        levels::H_FLIPADST,
    ];
    let mut out = Vec::new();
    for &sz in &[S4x4, S8x8, S16x16] {
        for &t in &all16 {
            out.push((sz, t));
        }
    }
    for &sz in &[S32x32, R8x32, R32x8, R16x32, R32x16] {
        out.push((sz, levels::DCT_DCT));
        out.push((sz, levels::IDTX));
    }
    // Any size with a 64 dimension is DCT-only in AV1, and the scalar
    // reference enforces it: there is no `inv_identity64_1d`, so `IDTX` at
    // 16x64 / 64x16 / 32x64 / 64x32 / 64x64 is not a legal cell to compare.
    // (`itxfm_add_dispatch` wires NEON identity kernels for those shapes
    // anyway; they are unreachable from a conformant bitstream.)
    for &sz in &[S64x64, R64x32, R32x64, R16x64, R64x16] {
        out.push((sz, levels::DCT_DCT));
    }
    out
}

/// `eob` values worth trying for a given scan length.
///
/// Every group-skip threshold in the aarch64 kernels sits at one of dav1d's
/// `eob_*` constants, so the sweep includes each of them and each one minus
/// one — the pair that brackets a wrong threshold — plus the small values and
/// the dense end. `itx_8bpc_every_eob_small_coeffs` covers the gaps between
/// them exhaustively; this sampled list keeps the other magnitude arms cheap.
fn eob_sweep(n: usize) -> Vec<i32> {
    let mut v: Vec<i32> = vec![0, 1, 2, 3, 7, 8, 9, 15, 16, 17];
    for &t in &[
        29usize, 32, 35, 36, 37, 43, 64, 107, 136, 151, 171, 256, 279, 300, 512, 1024,
    ] {
        v.push(t as i32 - 1);
        v.push(t as i32);
    }
    v.push(n as i32 - 1);
    v.retain(|&e| e >= 0 && (e as usize) < n);
    v.sort_unstable();
    v.dedup();
    v
}

fn sweep(scale: i32, seed: u64, what: &str) {
    sweep_with(scale, seed, what, false)
}

fn sweep_with(scale: i32, seed: u64, what: &str, every_eob: bool) {
    let _lock = crate::src::safe_simd::token_test_lock();
    let mut rep = Report::default();

    for (tx, tx_type) in wired_cells() {
        let (w, h) = tx.to_wh();
        let (sw, sh) = (w.min(32), h.min(32));
        let n = sw * sh;
        let stride = (w + 16).next_multiple_of(16);
        let mut rng = Rng(seed ^ ((tx as u64) << 40) ^ ((tx_type as u64) << 32));

        // A fresh destination per size, reused across eobs so a divergence is
        // attributable to the coefficients rather than the backdrop.
        let pixels: Vec<u8> = (0..stride * h)
            .map(|_| rng.in_range(0, 255) as u8)
            .collect();

        let eobs: Vec<i32> = if every_eob {
            (0..n as i32).collect()
        } else {
            eob_sweep(n)
        };
        for eob in eobs {
            let mut coeff = vec![0i16; 32 * 32];
            for pos in reachable_positions(tx, tx_type, eob as usize) {
                // Nonzero, like a real decoded coefficient at a scan position
                // at or before the eob.
                let mut c = rng.in_range(-scale, scale);
                if c == 0 {
                    c = 1;
                }
                coeff[pos] = c as i16;
            }

            let (neon, scalar, live) = run_cell(tx, tx_type, eob, &coeff, &pixels, stride);
            let bad = (0..h)
                .flat_map(|y| (0..w).map(move |x| (x, y)))
                .find(|&(x, y)| neon[y * stride + x] != scalar[y * stride + x]);
            rep.record(
                &format!("{w}x{h} type={tx_type} eob={eob}"),
                live,
                bad.is_none(),
                || {
                    let (x, y) = bad.unwrap();
                    format!(
                        "at ({x},{y}) neon={} scalar={}",
                        neon[y * stride + x],
                        scalar[y * stride + x]
                    )
                },
            );
        }
    }
    rep.finish(what, 1400);
}

/// Coefficient magnitudes in the range ordinary inter/intra residuals occupy.
#[test]
fn itx_8bpc_matches_scalar_small_coeffs() {
    sweep(
        64,
        0x1234_5678_9ABC_DEF0,
        "aarch64 8bpc itx (|coeff| <= 64)",
    );
}

/// An order of magnitude up: enough for the row transform's intermediate
/// clipping to engage on the larger sizes.
#[test]
fn itx_8bpc_matches_scalar_medium_coeffs() {
    sweep(
        1024,
        0x0BAD_C0DE_1234_5678,
        "aarch64 8bpc itx (|coeff| <= 1024)",
    );
}

/// EVERY legal `eob`, for every wired (size, type).
///
/// The sampled list above brackets the thresholds this session knew about; a
/// wrong constant nobody has named yet would sit in one of its gaps. This arm
/// leaves no gap: ~13k cells, about a second in release.
#[test]
fn itx_8bpc_every_eob_small_coeffs() {
    sweep_with(
        48,
        0xFEED_FACE_0000_0001,
        "aarch64 8bpc itx, every eob (|coeff| <= 48)",
        true,
    );
}

// ============================================================================
// 16bpc (`itx_arm_hbd`) — the same instrument, 10/12-bit
// ============================================================================

/// One 16bpc cell through both paths. Same shape as [`run_cell`], but the
/// coefficients are `i32`, the pixels `u16`, and `bd` carries `bitdepth_max`.
fn run_cell_16(
    tx: TxfmSize,
    tx_type: TxfmType,
    eob: i32,
    coeff: &[i32],
    pixels: &[u16],
    stride: usize,
    bitdepth_max: u16,
) -> (Vec<u16>, Vec<u16>, bool) {
    let bd = BitDepth16::new(bitdepth_max);

    let run = |simd: bool| -> (Vec<u16>, bool) {
        let mut px = pixels.to_vec();
        let mut cf = coeff.to_vec();
        let mut out = vec![0u16; px.len()];
        let handled = {
            let comp = Rav1dPictureDataComponent::wrap_buf::<BitDepth16>(&mut px, stride);
            let mut dst = crate::src::owned_recon::ReconDst::Pic(comp.with_offset::<BitDepth16>());
            let handled = if simd {
                crate::src::safe_simd::itx_arm::itxfm_add_dispatch::<BitDepth16>(
                    tx as usize,
                    tx_type as usize,
                    &mut dst,
                    &mut cf,
                    eob,
                    bd,
                )
            } else {
                crate::src::itx::itxfm_add_scalar_fallback::<BitDepth16>(
                    tx as usize,
                    tx_type,
                    &mut dst,
                    &mut cf,
                    eob,
                    bd,
                );
                true
            };
            comp.copy_pixels_to::<BitDepth16>(&mut out);
            handled
        };
        (out, handled)
    };

    let (neon, handled) = run(true);
    let (scalar, _) = run(false);
    (neon, scalar, handled)
}

/// The (size, type) pairs `itxfm_add_dispatch` wires to `itx_arm_hbd`.
///
/// Every shape with `max(w, h) <= 16`, every non-WHT type — that is the whole
/// set `hbd_supported` admits, so this list and the dispatch cannot drift
/// apart without the liveness gate in [`Report::finish`] firing.
fn wired_cells_16() -> Vec<(TxfmSize, TxfmType)> {
    use TxfmSize::*;
    let all16: [TxfmType; 16] = [
        levels::DCT_DCT,
        levels::ADST_DCT,
        levels::DCT_ADST,
        levels::ADST_ADST,
        levels::FLIPADST_DCT,
        levels::DCT_FLIPADST,
        levels::FLIPADST_FLIPADST,
        levels::ADST_FLIPADST,
        levels::FLIPADST_ADST,
        levels::IDTX,
        levels::V_DCT,
        levels::H_DCT,
        levels::V_ADST,
        levels::H_ADST,
        levels::V_FLIPADST,
        levels::H_FLIPADST,
    ];
    let mut out = Vec::new();
    for &sz in &[S4x4, S8x8, S16x16, R4x8, R8x4, R4x16, R16x4, R8x16, R16x8] {
        for &t in &all16 {
            out.push((sz, t));
        }
    }
    out
}

fn sweep16(scale: i32, seed: u64, bitdepth_max: u16, what: &str, every_eob: bool) {
    let _lock = crate::src::safe_simd::token_test_lock();
    let mut rep = Report::default();

    for (tx, tx_type) in wired_cells_16() {
        let (w, h) = tx.to_wh();
        let n = w * h;
        let stride = (w + 16).next_multiple_of(16);
        let mut rng = Rng(seed ^ ((tx as u64) << 40) ^ ((tx_type as u64) << 32));

        let pixels: Vec<u16> = (0..stride * h)
            .map(|_| rng.in_range(0, bitdepth_max as i32) as u16)
            .collect();

        let eobs: Vec<i32> = if every_eob {
            (0..n as i32).collect()
        } else {
            eob_sweep(n)
        };
        for eob in eobs {
            let mut coeff = vec![0i32; 32 * 32];
            for pos in reachable_positions(tx, tx_type, eob as usize) {
                let mut c = rng.in_range(-scale, scale);
                if c == 0 {
                    c = 1;
                }
                coeff[pos] = c;
            }

            let (neon, scalar, live) =
                run_cell_16(tx, tx_type, eob, &coeff, &pixels, stride, bitdepth_max);
            let bad = (0..h)
                .flat_map(|y| (0..w).map(move |x| (x, y)))
                .find(|&(x, y)| neon[y * stride + x] != scalar[y * stride + x]);
            rep.record(
                &format!("{w}x{h} type={tx_type} eob={eob}"),
                live,
                bad.is_none(),
                || {
                    let (x, y) = bad.unwrap();
                    format!(
                        "at ({x},{y}) neon={} scalar={}",
                        neon[y * stride + x],
                        scalar[y * stride + x]
                    )
                },
            );
        }
    }
    rep.finish(what, 1400);
}

/// 10-bit, ordinary residual magnitudes.
#[test]
fn itx_10bpc_matches_scalar_small_coeffs() {
    sweep16(
        64,
        0x1234_5678_9ABC_DEF0,
        1023,
        "aarch64 10bpc itx (|coeff| <= 64)",
        false,
    );
}

/// 10-bit, magnitudes large enough to drive the row/column clips.
///
/// At 10bpc those clips are `+-(1 << 17)` (row) and `+-(1 << 15)` (column), so
/// a port that kept the 8bpc `i16` state — which is exactly what the
/// `itx_arm_neon_*` 16bpc entry points do — fails here and not in the small
/// arm above.
#[test]
fn itx_10bpc_matches_scalar_large_coeffs() {
    sweep16(
        1 << 14,
        0x0BAD_C0DE_1234_5678,
        1023,
        "aarch64 10bpc itx (|coeff| <= 16384)",
        false,
    );
}

/// 12-bit: the widest clips the decoder can ask for.
#[test]
fn itx_12bpc_matches_scalar_large_coeffs() {
    sweep16(
        1 << 16,
        0xC0FF_EE00_1234_5678,
        4095,
        "aarch64 12bpc itx (|coeff| <= 65536)",
        false,
    );
}

/// EVERY legal `eob` at 10bpc, for every wired (size, type).
#[test]
fn itx_10bpc_every_eob_small_coeffs() {
    sweep16(
        48,
        0xFEED_FACE_0000_0002,
        1023,
        "aarch64 10bpc itx, every eob (|coeff| <= 48)",
        true,
    );
}

/// The sanity check on the sweep itself: the positions it fills must be the
/// ones the coefficient decoder can reach, and the mapping must actually
/// differ between the three transform classes. If this fails, every other
/// result in this module is suspect.
#[test]
fn reachable_positions_track_the_transform_class() {
    // 16x16, class H (H_DCT): the scan index IS the coefficient index, so
    // eob = 8 already reaches coefficient row 8 (index 8 = row 8, col 0).
    // That is precisely why dav1d's 16x16 `dct, identity` kernel uses
    // eob_half = 8 and not 36.
    assert_eq!(dav1d_tx_type_class[levels::H_DCT as usize], TxClass::H);
    let p = reachable_positions(TxfmSize::S16x16, levels::H_DCT, 8);
    assert_eq!(p, (0..=8).collect::<Vec<_>>());
    assert!(
        p.iter().any(|&rc| rc % 16 >= 8),
        "eob=8 must reach row >= 8"
    );

    // 16x16, class V (V_DCT): the column index cycles fastest, so eob = 8 is
    // still entirely inside row 0 — which is why the port's wrong 36 never
    // showed up on V_DCT in the corpus.
    assert_eq!(dav1d_tx_type_class[levels::V_DCT as usize], TxClass::V);
    let p = reachable_positions(TxfmSize::S16x16, levels::V_DCT, 8);
    assert!(
        p.iter().all(|&rc| rc % 16 == 0),
        "class V at eob=8 stays in coefficient row 0"
    );

    // 16x16, class 2D (DCT_DCT): the zigzag, and eob < 36 stays in rows 0..8.
    assert_eq!(dav1d_tx_type_class[levels::DCT_DCT as usize], TxClass::TwoD);
    let p = reachable_positions(TxfmSize::S16x16, levels::DCT_DCT, 35);
    assert!(
        p.iter().all(|&rc| rc % 16 < 8),
        "the 2D scan's first 36 positions are what makes eob_half = 36 sound"
    );
}
