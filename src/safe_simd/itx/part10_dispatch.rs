// ============================================================================
// DISPATCH: Safe entry points for ITX SIMD dispatch
// ============================================================================
//
// These functions encapsulate ALL unsafe (pointer creation + extern "C" calls)
// so that src/itx.rs can remain fully safe in the default build.

use crate::include::common::bitdepth::BPC;
use crate::src::levels::ADST_ADST;
use crate::src::levels::ADST_DCT;
use crate::src::levels::ADST_FLIPADST;
use crate::src::levels::DCT_ADST;
use crate::src::levels::DCT_DCT;
use crate::src::levels::DCT_FLIPADST;
use crate::src::levels::FLIPADST_ADST;
use crate::src::levels::FLIPADST_DCT;
use crate::src::levels::FLIPADST_FLIPADST;
use crate::src::levels::H_ADST;
use crate::src::levels::H_DCT;
use crate::src::levels::H_FLIPADST;
use crate::src::levels::IDTX;
use crate::src::levels::TxfmSize;
use crate::src::levels::TxfmType;
use crate::src::levels::V_ADST;
use crate::src::levels::V_DCT;
use crate::src::levels::V_FLIPADST;
use crate::src::levels::WHT_WHT;
use crate::src::strided::Strided as _;

/// Macro to generate the per-arch/bpc direct dispatch functions for ITX.
/// Each generated function matches on (tx_size, tx_type) and calls the right SIMD function.
macro_rules! impl_itxfm_direct_dispatch {
    (
        fn $fn_name:ident, $mod_path:path,
        itx16: [$(($sz16:expr, $w16:literal, $h16:literal)),* $(,)?],
        itx12: [$(($sz12:expr, $w12:literal, $h12:literal)),* $(,)?],
        itx2: [$(($sz2:expr, $w2:literal, $h2:literal)),* $(,)?],
        itx1: [$(($sz1:expr, $w1:literal, $h1:literal)),* $(,)?],
        wht: ($szw:expr, $ww:literal, $hw:literal),
        $bpc:literal bpc, $ext:ident,
        h_dct_fn: $h_dct_fn:ident, v_dct_fn: $v_dct_fn:ident,
        h_adst_fn: $h_adst_fn:ident, v_adst_fn: $v_adst_fn:ident,
        h_flipadst_fn: $h_flipadst_fn:ident, v_flipadst_fn: $v_flipadst_fn:ident
    ) => {
        paste::paste! {
            #[allow(non_upper_case_globals)]
            #[cfg(feature = "asm")]
            fn $fn_name(
                tx_size: usize,
                tx_type: usize,
                dst_ptr: *mut DynPixel,
                dst_stride: isize,
                coeff: *mut DynCoef,
                eob: i32,
                bitdepth_max: i32,
                coeff_len: u16,
                dst: *const FFISafe<PicOffset>,
            ) -> bool {
                use $mod_path as si;

                macro_rules! c {
                    ($func:expr) => {{
                        // SAFETY: SIMD feature verified by caller. All pointers from valid Rust refs.
                        unsafe { $func(dst_ptr, dst_stride, coeff, eob, bitdepth_max, coeff_len, dst) };
                        return true;
                    }};
                }

                // TxfmSize constants for matching
                const s4x4: usize = TxfmSize::S4x4 as usize;
                const s8x8: usize = TxfmSize::S8x8 as usize;
                const s16x16: usize = TxfmSize::S16x16 as usize;
                const s32x32: usize = TxfmSize::S32x32 as usize;
                const s64x64: usize = TxfmSize::S64x64 as usize;
                const r4x8: usize = TxfmSize::R4x8 as usize;
                const r8x4: usize = TxfmSize::R8x4 as usize;
                const r8x16: usize = TxfmSize::R8x16 as usize;
                const r16x8: usize = TxfmSize::R16x8 as usize;
                const r16x32: usize = TxfmSize::R16x32 as usize;
                const r32x16: usize = TxfmSize::R32x16 as usize;
                const r32x64: usize = TxfmSize::R32x64 as usize;
                const r64x32: usize = TxfmSize::R64x32 as usize;
                const r4x16: usize = TxfmSize::R4x16 as usize;
                const r16x4: usize = TxfmSize::R16x4 as usize;
                const r8x32: usize = TxfmSize::R8x32 as usize;
                const r32x8: usize = TxfmSize::R32x8 as usize;
                const r16x64: usize = TxfmSize::R16x64 as usize;
                const r64x16: usize = TxfmSize::R64x16 as usize;

                match (tx_size, tx_type as TxfmType) {
                    // WHT_WHT (only 4x4)
                    ($szw, WHT_WHT) => c!(si::[<inv_txfm_add_wht_wht_ $ww x $hw _ $bpc bpc_ $ext>]),

                    // itx16 sizes: all 16 non-WHT transform types
                    $(
                        ($sz16, DCT_DCT) => c!(si::[<inv_txfm_add_dct_dct_ $w16 x $h16 _ $bpc bpc_ $ext>]),
                        ($sz16, IDTX) => c!(si::[<inv_txfm_add_identity_identity_ $w16 x $h16 _ $bpc bpc_ $ext>]),
                        ($sz16, ADST_DCT) => c!(si::[<inv_txfm_add_dct_adst_ $w16 x $h16 _ $bpc bpc_ $ext>]),
                        ($sz16, DCT_ADST) => c!(si::[<inv_txfm_add_adst_dct_ $w16 x $h16 _ $bpc bpc_ $ext>]),
                        ($sz16, ADST_ADST) => c!(si::[<inv_txfm_add_adst_adst_ $w16 x $h16 _ $bpc bpc_ $ext>]),
                        ($sz16, FLIPADST_DCT) => c!(si::[<inv_txfm_add_dct_flipadst_ $w16 x $h16 _ $bpc bpc_ $ext>]),
                        ($sz16, DCT_FLIPADST) => c!(si::[<inv_txfm_add_flipadst_dct_ $w16 x $h16 _ $bpc bpc_ $ext>]),
                        ($sz16, FLIPADST_FLIPADST) => c!(si::[<inv_txfm_add_flipadst_flipadst_ $w16 x $h16 _ $bpc bpc_ $ext>]),
                        ($sz16, ADST_FLIPADST) => c!(si::[<inv_txfm_add_flipadst_adst_ $w16 x $h16 _ $bpc bpc_ $ext>]),
                        ($sz16, FLIPADST_ADST) => c!(si::[<inv_txfm_add_adst_flipadst_ $w16 x $h16 _ $bpc bpc_ $ext>]),
                        ($sz16, H_DCT) => c!(si::[<inv_txfm_add_ $h_dct_fn _ $w16 x $h16 _ $bpc bpc_ $ext>]),
                        ($sz16, V_DCT) => c!(si::[<inv_txfm_add_ $v_dct_fn _ $w16 x $h16 _ $bpc bpc_ $ext>]),
                        ($sz16, H_ADST) => c!(si::[<inv_txfm_add_ $h_adst_fn _ $w16 x $h16 _ $bpc bpc_ $ext>]),
                        ($sz16, V_ADST) => c!(si::[<inv_txfm_add_ $v_adst_fn _ $w16 x $h16 _ $bpc bpc_ $ext>]),
                        ($sz16, H_FLIPADST) => c!(si::[<inv_txfm_add_ $h_flipadst_fn _ $w16 x $h16 _ $bpc bpc_ $ext>]),
                        ($sz16, V_FLIPADST) => c!(si::[<inv_txfm_add_ $v_flipadst_fn _ $w16 x $h16 _ $bpc bpc_ $ext>]),
                    )*

                    // itx12 sizes: 12 transform types (no H/V ADST/FLIPADST hybrids)
                    $(
                        ($sz12, DCT_DCT) => c!(si::[<inv_txfm_add_dct_dct_ $w12 x $h12 _ $bpc bpc_ $ext>]),
                        ($sz12, IDTX) => c!(si::[<inv_txfm_add_identity_identity_ $w12 x $h12 _ $bpc bpc_ $ext>]),
                        ($sz12, ADST_DCT) => c!(si::[<inv_txfm_add_dct_adst_ $w12 x $h12 _ $bpc bpc_ $ext>]),
                        ($sz12, DCT_ADST) => c!(si::[<inv_txfm_add_adst_dct_ $w12 x $h12 _ $bpc bpc_ $ext>]),
                        ($sz12, ADST_ADST) => c!(si::[<inv_txfm_add_adst_adst_ $w12 x $h12 _ $bpc bpc_ $ext>]),
                        ($sz12, FLIPADST_DCT) => c!(si::[<inv_txfm_add_dct_flipadst_ $w12 x $h12 _ $bpc bpc_ $ext>]),
                        ($sz12, DCT_FLIPADST) => c!(si::[<inv_txfm_add_flipadst_dct_ $w12 x $h12 _ $bpc bpc_ $ext>]),
                        ($sz12, FLIPADST_FLIPADST) => c!(si::[<inv_txfm_add_flipadst_flipadst_ $w12 x $h12 _ $bpc bpc_ $ext>]),
                        ($sz12, ADST_FLIPADST) => c!(si::[<inv_txfm_add_flipadst_adst_ $w12 x $h12 _ $bpc bpc_ $ext>]),
                        ($sz12, FLIPADST_ADST) => c!(si::[<inv_txfm_add_adst_flipadst_ $w12 x $h12 _ $bpc bpc_ $ext>]),
                        ($sz12, H_DCT) => c!(si::[<inv_txfm_add_ $h_dct_fn _ $w12 x $h12 _ $bpc bpc_ $ext>]),
                        ($sz12, V_DCT) => c!(si::[<inv_txfm_add_ $v_dct_fn _ $w12 x $h12 _ $bpc bpc_ $ext>]),
                    )*

                    // itx2 sizes: DCT_DCT + IDTX only
                    $(
                        ($sz2, DCT_DCT) => c!(si::[<inv_txfm_add_dct_dct_ $w2 x $h2 _ $bpc bpc_ $ext>]),
                        ($sz2, IDTX) => c!(si::[<inv_txfm_add_identity_identity_ $w2 x $h2 _ $bpc bpc_ $ext>]),
                    )*

                    // itx1 sizes: DCT_DCT only
                    $(
                        ($sz1, DCT_DCT) => c!(si::[<inv_txfm_add_dct_dct_ $w1 x $h1 _ $bpc bpc_ $ext>]),
                    )*

                    _ => return false,
                }
            }
        }
    };
}

// x86_64 8bpc direct dispatch
#[cfg(target_arch = "x86_64")]
impl_itxfm_direct_dispatch!(
    fn itxfm_add_direct_x86_8bpc, crate::src::safe_simd::itx,
    itx16: [
        (s4x4, 4, 4),
        (s8x8, 8, 8),
        (r4x8, 4, 8), (r8x4, 8, 4),
        (r4x16, 4, 16), (r16x4, 16, 4),
        (r8x16, 8, 16), (r16x8, 16, 8),
    ],
    itx12: [
        (s16x16, 16, 16),
    ],
    itx2: [
        (r8x32, 8, 32), (r32x8, 32, 8),
        (r16x32, 16, 32), (r32x16, 32, 16),
        (s32x32, 32, 32),
    ],
    itx1: [
        (r16x64, 16, 64), (r32x64, 32, 64),
        (r64x16, 64, 16), (r64x32, 64, 32),
        (s64x64, 64, 64),
    ],
    wht: (s4x4, 4, 4),
    8 bpc, avx2,
    h_dct_fn: dct_identity, v_dct_fn: identity_dct,
    h_adst_fn: adst_identity, v_adst_fn: identity_adst,
    h_flipadst_fn: flipadst_identity, v_flipadst_fn: identity_flipadst
);

// x86_64 16bpc direct dispatch
#[cfg(target_arch = "x86_64")]
impl_itxfm_direct_dispatch!(
    fn itxfm_add_direct_x86_16bpc, crate::src::safe_simd::itx,
    itx16: [
        (s4x4, 4, 4),
        (s8x8, 8, 8),
        (r4x8, 4, 8), (r8x4, 8, 4),
        (r4x16, 4, 16), (r16x4, 16, 4),
        (r8x16, 8, 16), (r16x8, 16, 8),
    ],
    itx12: [
        (s16x16, 16, 16),
    ],
    itx2: [
        (r8x32, 8, 32), (r32x8, 32, 8),
        (r16x32, 16, 32), (r32x16, 32, 16),
        (s32x32, 32, 32),
    ],
    itx1: [
        (r16x64, 16, 64), (r32x64, 32, 64),
        (r64x16, 64, 16), (r64x32, 64, 32),
        (s64x64, 64, 64),
    ],
    wht: (s4x4, 4, 4),
    16 bpc, avx2,
    h_dct_fn: dct_identity, v_dct_fn: identity_dct,
    h_adst_fn: adst_identity, v_adst_fn: identity_adst,
    h_flipadst_fn: flipadst_identity, v_flipadst_fn: identity_flipadst
);

/// Compute the scalar DC value used by the DCT_DCT DC-only fast path.
///
/// Mirrors the scalar reference in `src/itx.rs:89-105`:
///
/// ```text
/// dc = coeff[0] as i32
/// if rect2: dc = (dc * 181 + 128) >> 8
/// dc = (dc * 181 + 128) >> 8
/// dc = (dc + (1 << shift >> 1)) >> shift
/// dc = (dc * 181 + 128 + 2048) >> 12
/// ```
#[cfg(not(feature = "asm"))]
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn dc_only_compute(coeff0: i32, rect2: bool, shift: u32) -> i32 {
    let mut dc = coeff0;
    if rect2 {
        dc = (dc * 181 + 128) >> 8;
    }
    dc = (dc * 181 + 128) >> 8;
    let rnd: i32 = if shift == 0 { 0 } else { 1 << (shift - 1) };
    dc = (dc + rnd) >> shift;
    dc = (dc * 181 + 128 + 2048) >> 12;
    dc
}

/// Look up the per-size `shift` value for the DC-only formula. Matches the
/// `shift` table in `inv_txfm_add_rust` (src/itx.rs).
#[cfg(not(feature = "asm"))]
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn dc_only_shift(w: usize, h: usize) -> u32 {
    match (w, h) {
        (4, 4) | (4, 8) | (8, 4) => 0,
        (4, 16)
        | (8, 8)
        | (8, 16)
        | (16, 4)
        | (16, 8)
        | (32, 16)
        | (16, 32)
        | (32, 64)
        | (64, 32) => 1,
        (8, 32) | (16, 16) | (16, 64) | (32, 8) | (32, 32) | (64, 16) | (64, 64) => 2,
        _ => 0,
    }
}

/// DC-only fast path for 8bpc DCT_DCT (eob == 0). Broadcasts the precomputed
/// `dc` scalar across the block and adds + clamps with SIMD.
#[cfg(not(feature = "asm"))]
#[cfg(target_arch = "x86_64")]
#[arcane]
fn dc_only_add_8bpc(
    _token: Desktop64,
    dst: &mut [u8],
    dst_stride: usize,
    w: usize,
    h: usize,
    dc: i32,
) {
    let mut dst = dst.flex_mut();
    let dc_i16 = dc.clamp(i16::MIN as i32, i16::MAX as i32) as i16;

    if w >= 32 {
        // 32-byte rows: one AVX2 vector per chunk of 32 columns
        let dc_v = _mm256_set1_epi16(dc_i16);
        let zero = _mm256_setzero_si256();
        for y in 0..h {
            let row_off = y * dst_stride;
            let mut x = 0;
            while x + 32 <= w {
                let d =
                    loadu_256!(<&[u8; 32]>::try_from(&dst[row_off + x..row_off + x + 32]).unwrap());
                // Unpack u8 -> i16, add dc, pack back with unsigned saturation.
                // Note: unpack is per-lane, so we pack the lanes back the same way.
                let d_lo = _mm256_unpacklo_epi8(d, zero);
                let d_hi = _mm256_unpackhi_epi8(d, zero);
                let sum_lo = _mm256_add_epi16(d_lo, dc_v);
                let sum_hi = _mm256_add_epi16(d_hi, dc_v);
                // packus saturates signed i16 -> u8 [0, 255] = exactly what we want
                let packed = _mm256_packus_epi16(sum_lo, sum_hi);
                storeu_256!(
                    <&mut [u8; 32]>::try_from(&mut dst[row_off + x..row_off + x + 32]).unwrap(),
                    packed
                );
                x += 32;
            }
        }
    } else if w == 16 {
        let dc_v = _mm_set1_epi16(dc_i16);
        let zero = _mm_setzero_si128();
        for y in 0..h {
            let row_off = y * dst_stride;
            let d = loadu_128!(<&[u8; 16]>::try_from(&dst[row_off..row_off + 16]).unwrap());
            let d_lo = _mm_unpacklo_epi8(d, zero);
            let d_hi = _mm_unpackhi_epi8(d, zero);
            let sum_lo = _mm_add_epi16(d_lo, dc_v);
            let sum_hi = _mm_add_epi16(d_hi, dc_v);
            let packed = _mm_packus_epi16(sum_lo, sum_hi);
            storeu_128!(
                <&mut [u8; 16]>::try_from(&mut dst[row_off..row_off + 16]).unwrap(),
                packed
            );
        }
    } else if w == 8 {
        let dc_v = _mm_set1_epi16(dc_i16);
        let zero = _mm_setzero_si128();
        for y in 0..h {
            let row_off = y * dst_stride;
            let d = loadi64!(&dst[row_off..row_off + 8]);
            let d_lo = _mm_unpacklo_epi8(d, zero);
            let sum = _mm_add_epi16(d_lo, dc_v);
            let packed = _mm_packus_epi16(sum, sum);
            storei64!(&mut dst[row_off..row_off + 8], packed);
        }
    } else {
        // w == 4: scalar fallback (small h max 16 → ≤ 64 adds, not worth SIMD)
        let dc_i32 = dc;
        for y in 0..h {
            let row_off = y * dst_stride;
            for x in 0..w {
                dst[row_off + x] = (dst[row_off + x] as i32 + dc_i32).clamp(0, 255) as u8;
            }
        }
    }
}

/// DC-only fast path for 16bpc DCT_DCT (eob == 0). 16bpc uses u16 pixels and
/// per-pixel clamp to `bitdepth_max` (1023 for 10-bit, 4095 for 12-bit).
#[cfg(not(feature = "asm"))]
#[cfg(target_arch = "x86_64")]
#[arcane]
fn dc_only_add_16bpc(
    _token: Desktop64,
    dst: &mut [u16],
    px_stride: usize,
    w: usize,
    h: usize,
    dc: i32,
    bitdepth_max: i32,
) {
    let mut dst = dst.flex_mut();

    if w >= 16 {
        let dc_v = _mm256_set1_epi32(dc);
        let max_v = _mm256_set1_epi32(bitdepth_max);
        let zero = _mm256_setzero_si256();
        for y in 0..h {
            let row_off = y * px_stride;
            let mut x = 0;
            while x + 16 <= w {
                let d = loadu_256!(
                    <&[u16; 16]>::try_from(&dst[row_off + x..row_off + x + 16]).unwrap()
                );
                let d_lo = _mm256_unpacklo_epi16(d, zero);
                let d_hi = _mm256_unpackhi_epi16(d, zero);
                // Reshuffle to true low/high halves
                let d_0_4 = _mm256_permute2x128_si256(d_lo, d_hi, 0x20);
                let d_4_8 = _mm256_permute2x128_si256(d_lo, d_hi, 0x31);
                let sum_lo = _mm256_add_epi32(d_0_4, dc_v);
                let sum_hi = _mm256_add_epi32(d_4_8, dc_v);
                let clamped_lo = _mm256_max_epi32(_mm256_min_epi32(sum_lo, max_v), zero);
                let clamped_hi = _mm256_max_epi32(_mm256_min_epi32(sum_hi, max_v), zero);
                // packus_epi32 saturates signed-i32 to unsigned-u16
                let packed = _mm256_packus_epi32(clamped_lo, clamped_hi);
                // packus_epi32 interleaves lanes; permute4x64 fixes ordering
                let packed = _mm256_permute4x64_epi64::<0xd8>(packed);
                storeu_256!(
                    <&mut [u16; 16]>::try_from(&mut dst[row_off + x..row_off + x + 16]).unwrap(),
                    packed
                );
                x += 16;
            }
        }
    } else if w == 8 {
        let dc_v = _mm_set1_epi32(dc);
        let max_v = _mm_set1_epi32(bitdepth_max);
        let zero = _mm_setzero_si128();
        for y in 0..h {
            let row_off = y * px_stride;
            let d = loadu_128!(<&[u16; 8]>::try_from(&dst[row_off..row_off + 8]).unwrap());
            let d_lo = _mm_unpacklo_epi16(d, zero);
            let d_hi = _mm_unpackhi_epi16(d, zero);
            let sum_lo = _mm_add_epi32(d_lo, dc_v);
            let sum_hi = _mm_add_epi32(d_hi, dc_v);
            let clamped_lo = _mm_max_epi32(_mm_min_epi32(sum_lo, max_v), zero);
            let clamped_hi = _mm_max_epi32(_mm_min_epi32(sum_hi, max_v), zero);
            let packed = _mm_packus_epi32(clamped_lo, clamped_hi);
            storeu_128!(
                <&mut [u16; 8]>::try_from(&mut dst[row_off..row_off + 8]).unwrap(),
                packed
            );
        }
    } else {
        // w == 4: scalar fallback
        for y in 0..h {
            let row_off = y * px_stride;
            for x in 0..w {
                dst[row_off + x] = (dst[row_off + x] as i32 + dc).clamp(0, bitdepth_max) as u16;
            }
        }
    }
}

/// 8bpc dispatch: calls inner SIMD functions directly with slices.
/// Arcane functions take (token, dst, stride_usize, coeff, eob, bdmax).
/// Scalar functions take (dst, base, stride_isize, coeff, eob, bdmax).
#[cfg(not(feature = "asm"))]
#[cfg(target_arch = "x86_64")]
#[allow(non_upper_case_globals)]
fn itxfm_dispatch_8bpc(
    token: Desktop64,
    tx_size: usize,
    tx_type: TxfmType,
    dst: &mut [u8],
    base: usize,
    stride_u: usize,
    stride_i: isize,
    coeff: &mut [i16],
    eob: i32,
    bdmax: i32,
) -> bool {
    use crate::src::levels::TxfmSize;

    // DC-only fast path: DCT_DCT with eob == 0. Mirrors the scalar shortcut
    // in `src/itx.rs:89-105`.
    if eob == 0 && tx_type == DCT_DCT {
        let txfm = match TxfmSize::from_repr(tx_size) {
            Some(t) => t,
            None => return false,
        };
        let (w, h) = txfm.to_wh();
        let rect2 = w * 2 == h || h * 2 == w;
        let shift = dc_only_shift(w, h);
        let dc = dc_only_compute(coeff[0] as i32, rect2, shift);
        coeff[0] = 0;
        dc_only_add_8bpc(token, &mut dst[base..], stride_u, w, h, dc);
        return true;
    }

    // Arcane functions: dst starts at pixel (base=0 for positive stride)
    macro_rules! arcane {
        ($func:ident) => {{
            $func(token, &mut dst[base..], stride_u, coeff, eob, bdmax);
            return true;
        }};
    }
    // Scalar functions: pass dst with base offset and signed stride
    macro_rules! scalar {
        ($func:ident) => {{
            $func(dst, base, stride_i, coeff, eob, bdmax);
            return true;
        }};
    }

    const S4x4: usize = TxfmSize::S4x4 as usize;
    const S8x8: usize = TxfmSize::S8x8 as usize;
    const S16x16: usize = TxfmSize::S16x16 as usize;
    const S32x32: usize = TxfmSize::S32x32 as usize;
    const S64x64: usize = TxfmSize::S64x64 as usize;
    const R4x8: usize = TxfmSize::R4x8 as usize;
    const R8x4: usize = TxfmSize::R8x4 as usize;
    const R8x16: usize = TxfmSize::R8x16 as usize;
    const R16x8: usize = TxfmSize::R16x8 as usize;
    const R16x32: usize = TxfmSize::R16x32 as usize;
    const R32x16: usize = TxfmSize::R32x16 as usize;
    const R32x64: usize = TxfmSize::R32x64 as usize;
    const R64x32: usize = TxfmSize::R64x32 as usize;
    const R4x16: usize = TxfmSize::R4x16 as usize;
    const R16x4: usize = TxfmSize::R16x4 as usize;
    const R8x32: usize = TxfmSize::R8x32 as usize;
    const R32x8: usize = TxfmSize::R32x8 as usize;
    const R16x64: usize = TxfmSize::R16x64 as usize;
    const R64x16: usize = TxfmSize::R64x16 as usize;

    match (tx_size, tx_type) {
        // ===== WHT (SSE2 SIMD, 4x4 only) =====
        (S4x4, WHT_WHT) => arcane!(inv_txfm_add_wht_wht_4x4_8bpc_avx2_inner),

        // ===== DCT_DCT (arcane, all 19 sizes) =====
        (S4x4, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_4x4_8bpc_avx2_inner),
        (R4x8, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_4x8_8bpc_avx2_inner),
        (R8x4, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_8x4_8bpc_avx2_inner),
        (R4x16, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_4x16_8bpc_avx2_inner),
        (R16x4, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_16x4_8bpc_avx2_inner),
        (S8x8, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_8x8_8bpc_avx2_inner),
        (R8x16, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_8x16_8bpc_avx2_inner),
        (R16x8, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_16x8_8bpc_avx2_inner),
        (R8x32, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_8x32_8bpc_avx2_inner),
        (R32x8, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_32x8_8bpc_avx2_inner),
        (S16x16, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_16x16_8bpc_avx2_inner),
        (R16x32, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_16x32_8bpc_avx2_inner),
        (R32x16, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_32x16_8bpc_avx2_inner),
        (R16x64, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_16x64_8bpc_avx2_inner),
        (R64x16, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_64x16_8bpc_avx2_inner),
        (S32x32, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_32x32_8bpc_avx2_inner),
        (R32x64, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_32x64_8bpc_avx2_inner),
        (R64x32, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_64x32_8bpc_avx2_inner),
        (S64x64, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_64x64_8bpc_avx2_inner),

        // ===== IDTX (arcane, 8 sizes) =====
        (S4x4, IDTX) => arcane!(inv_identity_add_4x4_8bpc_avx2),
        (S8x8, IDTX) => arcane!(inv_identity_add_8x8_8bpc_avx2),
        (S16x16, IDTX) => arcane!(inv_identity_add_16x16_8bpc_avx2),
        (R8x32, IDTX) => arcane!(inv_txfm_add_identity_identity_8x32_8bpc_avx2_inner),
        (R32x8, IDTX) => arcane!(inv_txfm_add_identity_identity_32x8_8bpc_avx2_inner),
        (R16x32, IDTX) => arcane!(inv_txfm_add_identity_identity_16x32_8bpc_avx2_inner),
        (R32x16, IDTX) => arcane!(inv_txfm_add_identity_identity_32x16_8bpc_avx2_inner),
        (S32x32, IDTX) => arcane!(inv_txfm_add_identity_identity_32x32_8bpc_avx2_inner),

        // ===== 4x4 ADST/FLIPADST/hybrid (scalar, 14 types) =====
        (S4x4, ADST_DCT) => scalar!(inv_txfm_add_dct_adst_4x4_8bpc_avx2_inner),
        (S4x4, DCT_ADST) => scalar!(inv_txfm_add_adst_dct_4x4_8bpc_avx2_inner),
        (S4x4, ADST_ADST) => scalar!(inv_txfm_add_adst_adst_4x4_8bpc_avx2_inner),
        (S4x4, FLIPADST_DCT) => scalar!(inv_txfm_add_dct_flipadst_4x4_8bpc_avx2_inner),
        (S4x4, DCT_FLIPADST) => scalar!(inv_txfm_add_flipadst_dct_4x4_8bpc_avx2_inner),
        (S4x4, FLIPADST_FLIPADST) => scalar!(inv_txfm_add_flipadst_flipadst_4x4_8bpc_avx2_inner),
        (S4x4, ADST_FLIPADST) => scalar!(inv_txfm_add_flipadst_adst_4x4_8bpc_avx2_inner),
        (S4x4, FLIPADST_ADST) => scalar!(inv_txfm_add_adst_flipadst_4x4_8bpc_avx2_inner),
        (S4x4, H_DCT) => scalar!(inv_txfm_add_dct_identity_4x4_8bpc_avx2_inner),
        (S4x4, V_DCT) => scalar!(inv_txfm_add_identity_dct_4x4_8bpc_avx2_inner),
        (S4x4, H_ADST) => scalar!(inv_txfm_add_h_adst_4x4_8bpc_avx2_inner),
        (S4x4, V_ADST) => scalar!(inv_txfm_add_v_adst_4x4_8bpc_avx2_inner),
        (S4x4, H_FLIPADST) => scalar!(inv_txfm_add_h_flipadst_4x4_8bpc_avx2_inner),
        (S4x4, V_FLIPADST) => scalar!(inv_txfm_add_v_flipadst_4x4_8bpc_avx2_inner),

        _ => return false,
    }
}

/// 16bpc dispatch: calls inner SIMD functions directly with slices.
/// All arcane functions take (token, dst: &mut [u16], byte_stride: usize, coeff, eob, bdmax).
/// WHT takes (dst: &mut [u16], base, px_stride: isize, coeff, eob, bdmax).
#[cfg(not(feature = "asm"))]
#[cfg(target_arch = "x86_64")]
#[allow(non_upper_case_globals)]
fn itxfm_dispatch_16bpc(
    token: Desktop64,
    tx_size: usize,
    tx_type: TxfmType,
    dst: &mut [u16],
    base: usize,
    byte_stride: usize,
    coeff_i16: &mut [i16],
    eob: i32,
    bdmax: i32,
) -> bool {
    use crate::src::levels::TxfmSize;

    // For 16bpc, BD::Coef = i32. The dispatch receives coeff reinterpreted as [i16],
    // but the actual data is [i32] (each i32 = two consecutive i16s in memory).
    // Reinterpret back to [i32] so functions can read full 32-bit coefficients.
    let coeff: &mut [i32] =
        zerocopy::FromBytes::mut_from_bytes(zerocopy::IntoBytes::as_mut_bytes(coeff_i16))
            .expect("coeff alignment/size mismatch for i32 reinterpretation");

    // DC-only fast path: DCT_DCT with eob == 0. Mirrors the scalar shortcut
    // in `src/itx.rs:89-105`.
    if eob == 0 && tx_type == DCT_DCT {
        let txfm = match TxfmSize::from_repr(tx_size) {
            Some(t) => t,
            None => return false,
        };
        let (w, h) = txfm.to_wh();
        let rect2 = w * 2 == h || h * 2 == w;
        let shift = dc_only_shift(w, h);
        let dc = dc_only_compute(coeff[0], rect2, shift);
        coeff[0] = 0;
        // Convert byte_stride to u16 pixel stride
        let px_stride = byte_stride / 2;
        dc_only_add_16bpc(token, &mut dst[base..], px_stride, w, h, dc, bdmax);
        return true;
    }

    // Arcane 16bpc functions take byte_stride as usize
    macro_rules! arcane {
        ($func:ident) => {{
            $func(token, &mut dst[base..], byte_stride, coeff, eob, bdmax);
            return true;
        }};
    }
    const S4x4: usize = TxfmSize::S4x4 as usize;
    const S8x8: usize = TxfmSize::S8x8 as usize;
    const S16x16: usize = TxfmSize::S16x16 as usize;
    const S32x32: usize = TxfmSize::S32x32 as usize;
    const S64x64: usize = TxfmSize::S64x64 as usize;
    const R4x8: usize = TxfmSize::R4x8 as usize;
    const R8x4: usize = TxfmSize::R8x4 as usize;
    const R8x16: usize = TxfmSize::R8x16 as usize;
    const R16x8: usize = TxfmSize::R16x8 as usize;
    const R16x32: usize = TxfmSize::R16x32 as usize;
    const R32x16: usize = TxfmSize::R32x16 as usize;
    const R32x64: usize = TxfmSize::R32x64 as usize;
    const R64x32: usize = TxfmSize::R64x32 as usize;
    const R4x16: usize = TxfmSize::R4x16 as usize;
    const R16x4: usize = TxfmSize::R16x4 as usize;
    const R8x32: usize = TxfmSize::R8x32 as usize;
    const R32x8: usize = TxfmSize::R32x8 as usize;
    const R16x64: usize = TxfmSize::R16x64 as usize;
    const R64x16: usize = TxfmSize::R64x16 as usize;

    match (tx_size, tx_type) {
        // ===== WHT (SSE2 SIMD, 4x4 only) =====
        (S4x4, WHT_WHT) => arcane!(inv_txfm_add_wht_wht_4x4_16bpc_avx2_inner),

        // ===== DCT_DCT (arcane, all 19 sizes) =====
        (S4x4, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_4x4_16bpc_avx2_inner),
        (R4x8, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_4x8_16bpc_avx2_inner),
        (R8x4, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_8x4_16bpc_avx2_inner),
        (R4x16, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_4x16_16bpc_avx2_inner),
        (R16x4, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_16x4_16bpc_avx2_inner),
        (S8x8, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_8x8_16bpc_avx2_inner),
        (R8x16, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_8x16_16bpc_avx2_inner),
        (R16x8, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_16x8_16bpc_avx2_inner),
        (R8x32, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_8x32_16bpc_avx2_inner),
        (R32x8, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_32x8_16bpc_avx2_inner),
        (S16x16, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_16x16_16bpc_avx2_inner),
        (R16x32, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_16x32_16bpc_avx2_inner),
        (R32x16, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_32x16_16bpc_avx2_inner),
        (R16x64, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_16x64_16bpc_avx2_inner),
        (R64x16, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_64x16_16bpc_avx2_inner),
        (S32x32, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_32x32_16bpc_avx2_inner),
        (R32x64, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_32x64_16bpc_avx2_inner),
        (R64x32, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_64x32_16bpc_avx2_inner),
        (S64x64, DCT_DCT) => arcane!(inv_txfm_add_dct_dct_64x64_16bpc_avx2_inner),

        // ===== IDTX (arcane, 14 sizes) =====
        (S4x4, IDTX) => arcane!(inv_identity_add_4x4_16bpc_avx2),
        (R4x8, IDTX) => arcane!(inv_txfm_add_identity_identity_4x8_16bpc_avx2_inner),
        (R8x4, IDTX) => arcane!(inv_txfm_add_identity_identity_8x4_16bpc_avx2_inner),
        (R4x16, IDTX) => arcane!(inv_txfm_add_identity_identity_4x16_16bpc_avx2_inner),
        (R16x4, IDTX) => arcane!(inv_txfm_add_identity_identity_16x4_16bpc_avx2_inner),
        (S8x8, IDTX) => arcane!(inv_identity_add_8x8_16bpc_avx2),
        (R8x16, IDTX) => arcane!(inv_txfm_add_identity_identity_8x16_16bpc_avx2_inner),
        (R16x8, IDTX) => arcane!(inv_txfm_add_identity_identity_16x8_16bpc_avx2_inner),
        (R8x32, IDTX) => arcane!(inv_txfm_add_identity_identity_8x32_16bpc_avx2_inner),
        (R32x8, IDTX) => arcane!(inv_txfm_add_identity_identity_32x8_16bpc_avx2_inner),
        (S16x16, IDTX) => arcane!(inv_identity_add_16x16_16bpc_avx2),
        (R16x32, IDTX) => arcane!(inv_txfm_add_identity_identity_16x32_16bpc_avx2_inner),
        (R32x16, IDTX) => arcane!(inv_txfm_add_identity_identity_32x16_16bpc_avx2_inner),
        (S32x32, IDTX) => arcane!(inv_txfm_add_identity_identity_32x32_16bpc_avx2_inner),

        _ => return false,
    }
}

/// Safe dispatch entry point for ITX SIMD on x86_64 (non-asm path).
///
/// Calls inner SIMD functions directly with slices — no FFI wrappers, no raw pointers.
/// Returns `true` if a SIMD implementation handled the call.
#[cfg(not(feature = "asm"))]
pub fn itxfm_add_dispatch<BD: BitDepth>(
    tx_size: usize,
    tx_type: usize,
    dst: &mut crate::src::owned_recon::ReconDst<'_>,
    coeff: &mut [BD::Coef],
    eob: i32,
    bd: BD,
) -> bool {
    use zerocopy::IntoBytes;

    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (tx_size, tx_type, &dst, &coeff, eob, &bd);
        return false;
    }

    #[cfg(target_arch = "x86_64")]
    {
        let Some(token) = crate::src::cpu::summon_avx2() else {
            return false;
        };

        let txfm = match crate::src::levels::TxfmSize::from_repr(tx_size) {
            Some(t) => t,
            None => return false,
        };
        let (w, h) = txfm.to_wh();
        let bd_c = bd.into_c();

        // Reinterpret coeff as &mut [i16] (safe via zerocopy)
        let coeff_i16: &mut [i16] = zerocopy::FromBytes::mut_from_bytes(coeff.as_mut_bytes())
            .expect("coeff alignment/size mismatch for i16 reinterpretation");

        dst.with_block_mut::<BD, _>(
            w,
            h,
            |bytes, offset, stride| match BD::BPC {
                BPC::BPC8 => itxfm_dispatch_8bpc(
                    token,
                    tx_size,
                    tx_type as TxfmType,
                    bytes,
                    offset,
                    stride.unsigned_abs(),
                    stride,
                    coeff_i16,
                    eob,
                    bd_c,
                ),
                BPC::BPC16 => {
                    let dst_u16: &mut [u16] = zerocopy::FromBytes::mut_from_bytes(&mut bytes[..])
                        .expect("dst alignment/size mismatch for u16 reinterpretation");
                    itxfm_dispatch_16bpc(
                        token,
                        tx_size,
                        tx_type as TxfmType,
                        dst_u16,
                        offset / 2,
                        stride.unsigned_abs(),
                        coeff_i16,
                        eob,
                        bd_c,
                    )
                }
            },
        )
    }
}

/// Safe dispatch entry point for ITX SIMD on x86_64.
///
/// Takes safe Rust types, creates raw pointers internally, dispatches to the
/// right SIMD function. Returns `true` if a SIMD implementation handled the call.
#[cfg(feature = "asm")]
#[allow(unsafe_code)]
pub fn itxfm_add_dispatch<BD: BitDepth>(
    tx_size: usize,
    tx_type: usize,
    dst: &mut crate::src::owned_recon::ReconDst<'_>,
    coeff: &mut [BD::Coef],
    eob: i32,
    bd: BD,
) -> bool {
    use crate::src::levels::TxfmSize;
    use crate::src::safe_simd::pixel_access::Flex;
    use archmage::Desktop64;
    use zerocopy::IntoBytes;

    let Some(_token) = crate::src::cpu::summon_avx2() else {
        return false;
    };

    // Get transform dimensions for tracked guard
    let txfm = TxfmSize::from_repr(tx_size).unwrap_or_default();
    let (w, h) = txfm.to_wh();

    // Create tracked guard — ensures borrow tracker knows about this access
    let dst = dst
        .as_pic()
        .expect("owned recon band is never armed under `c-ffi`/`asm`");
    let (mut dst_guard, _dst_base) = dst.strided_slice_mut::<BD>(w, h);
    let dst_ptr: *mut DynPixel = dst_guard.as_mut_bytes().as_mut_ptr() as *mut DynPixel;
    let dst_stride = dst.stride();
    let coeff_len = coeff.len() as u16;
    let coeff_ptr = coeff.as_mut_ptr().cast();
    let bd_c = bd.into_c();
    let dst_ffi = FFISafe::new(&dst);

    match BD::BPC {
        BPC::BPC8 => itxfm_add_direct_x86_8bpc(
            tx_size, tx_type, dst_ptr, dst_stride, coeff_ptr, eob, bd_c, coeff_len, dst_ffi,
        ),
        BPC::BPC16 => itxfm_add_direct_x86_16bpc(
            tx_size, tx_type, dst_ptr, dst_stride, coeff_ptr, eob, bd_c, coeff_len, dst_ffi,
        ),
    }
}
