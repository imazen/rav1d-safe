use strum::EnumCount;

use crate::include::common::bitdepth::AsPrimitive;
use crate::include::common::bitdepth::BitDepth;
use crate::include::common::bitdepth::DynCoef;
use crate::include::common::bitdepth::DynPixel;
use crate::include::common::intops::iclip;
use crate::include::dav1d::picture::PicOffset;
use crate::src::cpu::CpuFlags;
use crate::src::enum_map::DefaultValue;
use crate::src::ffi_safe::FFISafe;
use crate::src::itx_1d::rav1d_inv_adst16_1d_c;
use crate::src::itx_1d::rav1d_inv_adst4_1d_c;
use crate::src::itx_1d::rav1d_inv_adst8_1d_c;
use crate::src::itx_1d::rav1d_inv_dct16_1d_c;
use crate::src::itx_1d::rav1d_inv_dct32_1d_c;
use crate::src::itx_1d::rav1d_inv_dct4_1d_c;
use crate::src::itx_1d::rav1d_inv_dct64_1d_c;
use crate::src::itx_1d::rav1d_inv_dct8_1d_c;
use crate::src::itx_1d::rav1d_inv_flipadst16_1d_c;
use crate::src::itx_1d::rav1d_inv_flipadst4_1d_c;
use crate::src::itx_1d::rav1d_inv_flipadst8_1d_c;
use crate::src::itx_1d::rav1d_inv_identity16_1d_c;
use crate::src::itx_1d::rav1d_inv_identity32_1d_c;
use crate::src::itx_1d::rav1d_inv_identity4_1d_c;
use crate::src::itx_1d::rav1d_inv_identity8_1d_c;
use crate::src::itx_1d::rav1d_inv_wht4_1d_c;
use crate::src::levels::TxfmSize;
use crate::src::levels::TxfmType;
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
use crate::src::levels::N_TX_TYPES_PLUS_LL;
use crate::src::levels::V_ADST;
use crate::src::levels::V_DCT;
use crate::src::levels::V_FLIPADST;
use crate::src::levels::WHT_WHT;
use crate::src::strided::Strided as _;
use crate::src::wrap_fn_ptr::wrap_fn_ptr;
use std::cmp;
use std::num::NonZeroUsize;
use std::slice;

#[cfg(all(
    feature = "asm",
    not(any(target_arch = "riscv64", target_arch = "riscv32"))
))]
use crate::include::common::bitdepth::bd_fn;

#[cfg(all(feature = "asm", any(target_arch = "x86", target_arch = "x86_64")))]
use crate::include::common::bitdepth::bpc_fn;

pub type Itx1dFn = fn(c: &mut [i32], stride: NonZeroUsize, min: i32, max: i32);

#[inline(never)]
fn inv_txfm_add<BD: BitDepth>(
    dst: PicOffset,
    coeff: &mut [BD::Coef],
    eob: i32,
    w: usize,
    h: usize,
    shift: u8,
    first_1d_fn: Itx1dFn,
    second_1d_fn: Itx1dFn,
    has_dc_only: bool,
    bd: BD,
) {
    let bitdepth_max = bd.bitdepth_max().as_::<i32>();

    assert!(w >= 4 && w <= 64);
    assert!(h >= 4 && h <= 64);
    assert!(eob >= 0);

    let is_rect2 = w * 2 == h || h * 2 == w;
    let rnd = 1 << shift >> 1;

    if eob < has_dc_only as i32 {
        let mut dc = coeff[0].as_::<i32>();
        coeff[0] = 0.as_();
        if is_rect2 {
            dc = dc * 181 + 128 >> 8;
        }
        dc = dc * 181 + 128 >> 8;
        dc = dc + rnd >> shift;
        dc = dc * 181 + 128 + 2048 >> 12;
        for y in 0..h {
            let dst = dst + (y as isize * dst.pixel_stride::<BD>());
            let dst = &mut *dst.slice_mut::<BD>(w);
            for x in 0..w {
                dst[x] = bd.iclip_pixel(dst[x].as_::<i32>() + dc);
            }
        }
        return;
    }

    let sh = cmp::min(h, 32);
    let sw = cmp::min(w, 32);

    let coeff = &mut coeff[..sh * sw];

    let row_clip_min;
    let col_clip_min;
    if BD::BITDEPTH == 8 {
        row_clip_min = i16::MIN as i32;
        col_clip_min = i16::MIN as i32;
    } else {
        row_clip_min = (!bitdepth_max) << 7;
        col_clip_min = (!bitdepth_max) << 5;
    }
    let row_clip_max = !row_clip_min;
    let col_clip_max = !col_clip_min;

    let mut tmp = [0; 64 * 64];
    let mut c = &mut tmp[..];
    for y in 0..sh {
        if is_rect2 {
            for x in 0..sw {
                c[x] = coeff[y + x * sh].as_::<i32>() * 181 + 128 >> 8;
            }
        } else {
            for x in 0..sw {
                c[x] = coeff[y + x * sh].as_();
            }
        }
        first_1d_fn(c, 1.try_into().unwrap(), row_clip_min, row_clip_max);
        c = &mut c[w..];
    }

    coeff.fill(0.into());
    for i in 0..w * sh {
        tmp[i] = iclip(tmp[i] + rnd >> shift, col_clip_min, col_clip_max);
    }

    for x in 0..w {
        second_1d_fn(
            &mut tmp[x..],
            w.try_into().unwrap(),
            col_clip_min,
            col_clip_max,
        );
    }

    for y in 0..h {
        let dst = dst + (y as isize * dst.pixel_stride::<BD>());
        let dst = &mut *dst.slice_mut::<BD>(w);
        for x in 0..w {
            dst[x] = bd.iclip_pixel(dst[x].as_::<i32>() + (tmp[y * w + x] + 8 >> 4));
        }
    }
}

fn inv_txfm_add_rust<const W: usize, const H: usize, const TYPE: TxfmType, BD: BitDepth>(
    dst: PicOffset,
    coeff: &mut [BD::Coef],
    eob: i32,
    bd: BD,
) {
    let shift = match (W, H) {
        (4, 4) => 0,
        (4, 8) => 0,
        (4, 16) => 1,
        (8, 4) => 0,
        (8, 8) => 1,
        (8, 16) => 1,
        (8, 32) => 2,
        (16, 4) => 1,
        (16, 8) => 1,
        (16, 16) => 2,
        (16, 32) => 1,
        (16, 64) => 2,
        (32, 8) => 2,
        (32, 16) => 1,
        (32, 32) => 2,
        (32, 64) => 1,
        (64, 16) => 2,
        (64, 32) => 1,
        (64, 64) => 2,
        _ => unreachable!(),
    };
    let has_dc_only = TYPE == DCT_DCT;

    enum Type {
        Identity,
        Dct,
        Adst,
        FlipAdst,
    }
    use Type::*;
    // For some reason, this is flipped.
    let (second, first) = match TYPE {
        IDTX => (Identity, Identity),
        DCT_DCT => (Dct, Dct),
        ADST_DCT => (Adst, Dct),
        FLIPADST_DCT => (FlipAdst, Dct),
        H_DCT => (Identity, Dct),
        DCT_ADST => (Dct, Adst),
        ADST_ADST => (Adst, Adst),
        FLIPADST_ADST => (FlipAdst, Adst),
        DCT_FLIPADST => (Dct, FlipAdst),
        ADST_FLIPADST => (Adst, FlipAdst),
        FLIPADST_FLIPADST => (FlipAdst, FlipAdst),
        V_DCT => (Dct, Identity),
        H_ADST => (Identity, Adst),
        H_FLIPADST => (Identity, FlipAdst),
        V_ADST => (Adst, Identity),
        V_FLIPADST => (FlipAdst, Identity),
        WHT_WHT if (W, H) == (4, 4) => return inv_txfm_add_wht_wht_4x4_rust(dst, coeff, bd),
        _ => unreachable!(),
    };

    fn resolve_1d_fn(r#type: Type, n: usize) -> Itx1dFn {
        match (r#type, n) {
            (Identity, 4) => rav1d_inv_identity4_1d_c,
            (Identity, 8) => rav1d_inv_identity8_1d_c,
            (Identity, 16) => rav1d_inv_identity16_1d_c,
            (Identity, 32) => rav1d_inv_identity32_1d_c,
            (Dct, 4) => rav1d_inv_dct4_1d_c,
            (Dct, 8) => rav1d_inv_dct8_1d_c,
            (Dct, 16) => rav1d_inv_dct16_1d_c,
            (Dct, 32) => rav1d_inv_dct32_1d_c,
            (Dct, 64) => rav1d_inv_dct64_1d_c,
            (Adst, 4) => rav1d_inv_adst4_1d_c,
            (Adst, 8) => rav1d_inv_adst8_1d_c,
            (Adst, 16) => rav1d_inv_adst16_1d_c,
            (FlipAdst, 4) => rav1d_inv_flipadst4_1d_c,
            (FlipAdst, 8) => rav1d_inv_flipadst8_1d_c,
            (FlipAdst, 16) => rav1d_inv_flipadst16_1d_c,
            _ => unreachable!(),
        }
    }

    let first_1d_fn = resolve_1d_fn(first, W);
    let second_1d_fn = resolve_1d_fn(second, H);

    inv_txfm_add(
        dst,
        coeff,
        eob,
        W,
        H,
        shift,
        first_1d_fn,
        second_1d_fn,
        has_dc_only,
        bd,
    )
}

/// # Safety
///
/// Must be called by [`itxfm::Fn::call`].
#[deny(unsafe_op_in_unsafe_fn)]
unsafe extern "C" fn inv_txfm_add_c_erased<
    const W: usize,
    const H: usize,
    const TYPE: TxfmType,
    BD: BitDepth,
>(
    _dst_ptr: *mut DynPixel,
    _stride: isize,
    coeff: *mut DynCoef,
    eob: i32,
    bitdepth_max: i32,
    coeff_len: u16,
    dst: *const FFISafe<PicOffset>,
) {
    // SAFETY: Was passed as `FFISafe::new(_)` in `itxfm::Fn::call`.
    let dst = *unsafe { FFISafe::get(dst) };
    // SAFETY: `fn itxfm::Fn::call` passes `coeff.len()` as `coeff_len`.
    let coeff = unsafe { slice::from_raw_parts_mut(coeff.cast(), coeff_len.into()) };
    let bd = BD::from_c(bitdepth_max);
    inv_txfm_add_rust::<W, H, TYPE, BD>(dst, coeff, eob, bd)
}

/// Scalar fallback for ITX when no function pointer table is available.
/// Dispatches to `inv_txfm_add_rust` based on runtime (tx_size, tx_type).
#[cfg(not(any(feature = "asm", feature = "c-ffi")))]
fn itxfm_add_scalar_fallback<BD: BitDepth>(
    tx_size: usize,
    tx_type: TxfmType,
    dst: PicOffset,
    coeff: &mut [BD::Coef],
    eob: i32,
    bd: BD,
) {
    macro_rules! call {
        ($w:literal, $h:literal, $ty:expr) => {
            inv_txfm_add_rust::<$w, $h, { $ty }, BD>(dst, coeff, eob, bd)
        };
    }
    macro_rules! dispatch_type_16 {
        ($w:literal, $h:literal) => {
            match tx_type {
                DCT_DCT => call!($w, $h, DCT_DCT),
                IDTX => call!($w, $h, IDTX),
                DCT_ADST => call!($w, $h, DCT_ADST),
                ADST_DCT => call!($w, $h, ADST_DCT),
                ADST_ADST => call!($w, $h, ADST_ADST),
                DCT_FLIPADST => call!($w, $h, DCT_FLIPADST),
                FLIPADST_DCT => call!($w, $h, FLIPADST_DCT),
                FLIPADST_FLIPADST => call!($w, $h, FLIPADST_FLIPADST),
                ADST_FLIPADST => call!($w, $h, ADST_FLIPADST),
                FLIPADST_ADST => call!($w, $h, FLIPADST_ADST),
                H_DCT => call!($w, $h, H_DCT),
                V_DCT => call!($w, $h, V_DCT),
                H_ADST => call!($w, $h, H_ADST),
                V_ADST => call!($w, $h, V_ADST),
                H_FLIPADST => call!($w, $h, H_FLIPADST),
                V_FLIPADST => call!($w, $h, V_FLIPADST),
                _ => unreachable!(),
            }
        };
    }
    macro_rules! dispatch_type_12 {
        ($w:literal, $h:literal) => {
            match tx_type {
                DCT_DCT => call!($w, $h, DCT_DCT),
                IDTX => call!($w, $h, IDTX),
                DCT_ADST => call!($w, $h, DCT_ADST),
                ADST_DCT => call!($w, $h, ADST_DCT),
                ADST_ADST => call!($w, $h, ADST_ADST),
                DCT_FLIPADST => call!($w, $h, DCT_FLIPADST),
                FLIPADST_DCT => call!($w, $h, FLIPADST_DCT),
                FLIPADST_FLIPADST => call!($w, $h, FLIPADST_FLIPADST),
                ADST_FLIPADST => call!($w, $h, ADST_FLIPADST),
                FLIPADST_ADST => call!($w, $h, FLIPADST_ADST),
                H_DCT => call!($w, $h, H_DCT),
                V_DCT => call!($w, $h, V_DCT),
                _ => unreachable!(),
            }
        };
    }
    macro_rules! dispatch_type_2 {
        ($w:literal, $h:literal) => {
            match tx_type {
                DCT_DCT => call!($w, $h, DCT_DCT),
                IDTX => call!($w, $h, IDTX),
                _ => unreachable!(),
            }
        };
    }
    macro_rules! dispatch_type_1 {
        ($w:literal, $h:literal) => {
            match tx_type {
                DCT_DCT => call!($w, $h, DCT_DCT),
                _ => unreachable!(),
            }
        };
    }

    use TxfmSize::*;
    match tx_size {
        x if x == S4x4 as usize => match tx_type {
            WHT_WHT => call!(4, 4, WHT_WHT),
            _ => dispatch_type_16!(4, 4),
        },
        // itx16 sizes (W*H <= 8*16 or 16x16)
        x if x == R4x8 as usize => dispatch_type_16!(4, 8),
        x if x == R8x4 as usize => dispatch_type_16!(8, 4),
        x if x == S8x8 as usize => dispatch_type_16!(8, 8),
        x if x == R4x16 as usize => dispatch_type_16!(4, 16),
        x if x == R16x4 as usize => dispatch_type_16!(16, 4),
        x if x == R8x16 as usize => dispatch_type_16!(8, 16),
        x if x == R16x8 as usize => dispatch_type_16!(16, 8),
        x if x == S16x16 as usize => dispatch_type_16!(16, 16),
        // itx12 sizes (max_wh == 32)
        x if x == R8x32 as usize => dispatch_type_12!(8, 32),
        x if x == R32x8 as usize => dispatch_type_12!(32, 8),
        x if x == R16x32 as usize => dispatch_type_12!(16, 32),
        x if x == R32x16 as usize => dispatch_type_12!(32, 16),
        x if x == S32x32 as usize => dispatch_type_12!(32, 32),
        // itx2 sizes (max_wh == 64)
        x if x == R16x64 as usize => dispatch_type_2!(16, 64),
        x if x == R64x16 as usize => dispatch_type_2!(64, 16),
        x if x == R32x64 as usize => dispatch_type_2!(32, 64),
        x if x == R64x32 as usize => dispatch_type_2!(64, 32),
        // itx1 sizes
        x if x == S64x64 as usize => dispatch_type_1!(64, 64),
        _ => unreachable!(),
    }
}

wrap_fn_ptr!(unsafe extern "C" fn itxfm(
    dst_ptr: *mut DynPixel,
    dst_stride: isize,
    coeff: *mut DynCoef,
    eob: i32,
    bitdepth_max: i32,
    _coeff_len: u16,
    _dst: *const FFISafe<PicOffset>,
) -> ());


/// Macro to generate the per-arch/bpc direct dispatch functions for ITX.
/// Each generated function matches on (tx_size, tx_type) and calls the right SIMD function.
#[cfg(not(feature = "asm"))]
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
            #[cfg(not(feature = "asm"))]
            #[allow(non_upper_case_globals)]
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
#[cfg(all(not(feature = "asm"), target_arch = "x86_64"))]
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
    h_dct_fn: identity_dct, v_dct_fn: dct_identity,
    h_adst_fn: identity_adst, v_adst_fn: adst_identity,
    h_flipadst_fn: identity_flipadst, v_flipadst_fn: flipadst_identity
);

// x86_64 16bpc direct dispatch
#[cfg(all(not(feature = "asm"), target_arch = "x86_64"))]
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
    // 16bpc x86: V_DCT->identity_dct, H_DCT->dct_identity (swapped from 8bpc)
    h_dct_fn: dct_identity, v_dct_fn: identity_dct,
    h_adst_fn: adst_identity, v_adst_fn: identity_adst,
    h_flipadst_fn: flipadst_identity, v_flipadst_fn: identity_flipadst
);

// ARM aarch64 8bpc direct dispatch
#[cfg(all(not(feature = "asm"), target_arch = "aarch64"))]
impl_itxfm_direct_dispatch!(
    fn itxfm_add_direct_arm_8bpc, crate::src::safe_simd::itx_arm,
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
    8 bpc, neon,
    h_dct_fn: dct_identity, v_dct_fn: identity_dct,
    h_adst_fn: adst_identity, v_adst_fn: identity_adst,
    h_flipadst_fn: flipadst_identity, v_flipadst_fn: identity_flipadst
);

// ARM aarch64 16bpc direct dispatch
#[cfg(all(not(feature = "asm"), target_arch = "aarch64"))]
impl_itxfm_direct_dispatch!(
    fn itxfm_add_direct_arm_16bpc, crate::src::safe_simd::itx_arm,
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
    16 bpc, neon,
    h_dct_fn: dct_identity, v_dct_fn: identity_dct,
    h_adst_fn: adst_identity, v_adst_fn: identity_adst,
    h_flipadst_fn: flipadst_identity, v_flipadst_fn: identity_flipadst
);

/// Direct dispatch for itxfm_add - bypasses function pointer table.
/// Returns true if a SIMD function handled the call, false to fall through to scalar.
#[cfg(not(feature = "asm"))]
fn itxfm_add_direct<BD: BitDepth>(
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
    use crate::include::common::bitdepth::BPC;

    #[cfg(target_arch = "x86_64")]
    {
        use crate::src::cpu::CpuFlags;
        if crate::src::cpu::rav1d_get_cpu_flags().contains(CpuFlags::AVX2) {
            return match BD::BPC {
                BPC::BPC8 => itxfm_add_direct_x86_8bpc(tx_size, tx_type, dst_ptr, dst_stride, coeff, eob, bitdepth_max, coeff_len, dst),
                BPC::BPC16 => itxfm_add_direct_x86_16bpc(tx_size, tx_type, dst_ptr, dst_stride, coeff, eob, bitdepth_max, coeff_len, dst),
            };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return match BD::BPC {
            BPC::BPC8 => itxfm_add_direct_arm_8bpc(tx_size, tx_type, dst_ptr, dst_stride, coeff, eob, bitdepth_max, coeff_len, dst),
            BPC::BPC16 => itxfm_add_direct_arm_16bpc(tx_size, tx_type, dst_ptr, dst_stride, coeff, eob, bitdepth_max, coeff_len, dst),
        };
    }

    #[allow(unreachable_code)]
    {
        let _ = (tx_size, tx_type, dst_ptr, dst_stride, coeff, eob, bitdepth_max, coeff_len, dst);
        false
    }
}
impl itxfm::Fn {
    pub fn call<BD: BitDepth>(
        &self,
        tx_size: usize,
        tx_type: usize,
        dst: PicOffset,
        coeff: &mut [BD::Coef],
        eob: i32,
        bd: BD,
    ) {
        let dst_ptr = dst.as_mut_ptr::<BD>().cast();
        let dst_stride = dst.stride();
        let coeff_len = coeff.len() as u16;
        let coeff_ptr = coeff.as_mut_ptr().cast();
        let bd_c = bd.into_c();
        let dst_ffi = FFISafe::new(&dst);

        cfg_if::cfg_if! {
            if #[cfg(feature = "asm")] {
                let _ = (tx_size, tx_type);
                // SAFETY: Fallback `fn inv_txfm_add_rust` is safe; asm is supposed to do the same.
                unsafe { self.get()(dst_ptr, dst_stride, coeff_ptr, eob, bd_c, coeff_len, dst_ffi) }
            } else if #[cfg(feature = "c-ffi")] {
                // Direct dispatch: bypass function pointer for SIMD implementations.
                if itxfm_add_direct::<BD>(tx_size, tx_type, dst_ptr, dst_stride, coeff_ptr, eob, bd_c, coeff_len, dst_ffi) {
                    return;
                }
                // Fall through to scalar via function pointer
                // SAFETY: Fallback `fn inv_txfm_add_rust` is safe.
                unsafe { self.get()(dst_ptr, dst_stride, coeff_ptr, eob, bd_c, coeff_len, dst_ffi) }
            } else {
                // No function pointers: direct dispatch for SIMD, direct call for scalar.
                let _ = (dst_ptr, dst_stride, coeff_ptr, bd_c, coeff_len, dst_ffi);
                if itxfm_add_direct::<BD>(tx_size, tx_type, dst_ptr, dst_stride, coeff_ptr, eob, bd_c, coeff_len, dst_ffi) {
                    return;
                }
                // Scalar fallback
                itxfm_add_scalar_fallback::<BD>(tx_size, tx_type as TxfmType, dst, coeff, eob, bd);
            }
        }
    }
}

pub struct Rav1dInvTxfmDSPContext {
    pub itxfm_add: [[itxfm::Fn; N_TX_TYPES_PLUS_LL]; TxfmSize::COUNT],
}

fn inv_txfm_add_wht_wht_4x4_rust<BD: BitDepth>(
    dst: PicOffset,
    coeff: &mut [BD::Coef],
    bd: BD,
) {
    const H: usize = 4;
    const W: usize = 4;

    let coeff = &mut coeff[..W * H];

    let mut tmp = [0; W * H];
    let mut c = &mut tmp[..];
    for y in 0..H {
        for x in 0..W {
            c[x] = coeff[y + x * H].as_::<i32>() >> 2;
        }
        rav1d_inv_wht4_1d_c(c, 1.try_into().unwrap());
        c = &mut c[W..];
    }
    coeff.fill(0.into());

    for x in 0..W {
        rav1d_inv_wht4_1d_c(&mut tmp[x..], H.try_into().unwrap());
    }

    for y in 0..H {
        let dst = dst + (y as isize * dst.pixel_stride::<BD>());
        let dst = &mut *dst.slice_mut::<BD>(W);
        for x in 0..W {
            dst[x] = bd.iclip_pixel(dst[x].as_::<i32>() + tmp[y * W + x]);
        }
    }
}

#[cfg(all(
    feature = "asm",
    not(any(target_arch = "riscv64", target_arch = "riscv32"))
))]
macro_rules! assign_itx_fn {
    ($c:ident, $BD:ty, $w:literal, $h:literal, $type:ident, $type_enum:ident, $ext:ident) => {{
        use paste::paste;

        let tx = TxfmSize::from_wh($w, $h) as usize;

        paste! {
            $c.itxfm_add[tx][$type_enum as usize]
                = bd_fn!(itxfm::decl_fn, BD, [< inv_txfm_add_ $type _ $w x $h >], $ext);
        }
    }};
}

#[cfg(all(feature = "asm", any(target_arch = "x86", target_arch = "x86_64")))]
macro_rules! assign_itx_bpc_fn {
    ($c:ident, $w:literal, $h:literal, $type:ident, $type_enum:ident, $bpc:literal bpc, $ext:ident) => {{
        use paste::paste;

        let tx = TxfmSize::from_wh($w, $h) as usize;

        paste! {
            $c.itxfm_add[tx][$type_enum as usize]
                = bpc_fn!(itxfm::decl_fn, $bpc bpc, [< inv_txfm_add_ $type _ $w x $h >], $ext);
        }
    }};
}

#[cfg(all(feature = "asm", any(target_arch = "x86", target_arch = "x86_64")))]
macro_rules! assign_itx1_bpc_fn {
    ($c:ident, $w:literal, $h:literal, $bpc:literal bpc, $ext:ident) => {{
        assign_itx_bpc_fn!($c, $w, $h, dct_dct, DCT_DCT, $bpc bpc, $ext)
    }};
}

#[cfg(all(feature = "asm", any(target_arch = "arm", target_arch = "aarch64")))]
macro_rules! assign_itx1_fn {
    ($c:ident, $BD:ty, $w:literal, $h:literal, $ext:ident) => {{
        assign_itx_fn!($c, BD, $w, $h, dct_dct, DCT_DCT, $ext)
    }};
}

#[cfg(all(feature = "asm", any(target_arch = "x86", target_arch = "x86_64")))]
macro_rules! assign_itx2_bpc_fn {
    ($c:ident, $w:literal, $h:literal, $bpc:literal bpc, $ext:ident) => {{
        assign_itx1_bpc_fn!($c, $w, $h, $bpc bpc, $ext);
        assign_itx_bpc_fn!($c, $w, $h, identity_identity, IDTX, $bpc bpc, $ext)
    }};
}

#[cfg(all(feature = "asm", any(target_arch = "arm", target_arch = "aarch64")))]
macro_rules! assign_itx2_fn {
    ($c:ident, $BD:ty, $w:literal, $h:literal, $ext:ident) => {{
        assign_itx1_fn!($c, BD, $w, $h, $ext);
        assign_itx_fn!($c, BD, $w, $h, identity_identity, IDTX, $ext)
    }};
}

#[cfg(all(feature = "asm", any(target_arch = "x86", target_arch = "x86_64")))]
macro_rules! assign_itx12_bpc_fn {
    ($c:ident, $w:literal, $h:literal, $bpc:literal bpc, $ext:ident) => {{
        assign_itx2_bpc_fn!($c, $w, $h, $bpc bpc, $ext);
        assign_itx_bpc_fn!($c, $w, $h, dct_adst, ADST_DCT, $bpc bpc, $ext);
        assign_itx_bpc_fn!($c, $w, $h, dct_flipadst, FLIPADST_DCT, $bpc bpc, $ext);
        assign_itx_bpc_fn!($c, $w, $h, dct_identity, H_DCT, $bpc bpc, $ext);
        assign_itx_bpc_fn!($c, $w, $h, adst_dct, DCT_ADST, $bpc bpc, $ext);
        assign_itx_bpc_fn!($c, $w, $h, adst_adst, ADST_ADST, $bpc bpc, $ext);
        assign_itx_bpc_fn!($c, $w, $h, adst_flipadst, FLIPADST_ADST, $bpc bpc, $ext);
        assign_itx_bpc_fn!($c, $w, $h, flipadst_dct, DCT_FLIPADST, $bpc bpc, $ext);
        assign_itx_bpc_fn!($c, $w, $h, flipadst_adst, ADST_FLIPADST, $bpc bpc, $ext);
        assign_itx_bpc_fn!($c, $w, $h, flipadst_flipadst, FLIPADST_FLIPADST, $bpc bpc, $ext);
        assign_itx_bpc_fn!($c, $w, $h, identity_dct, V_DCT, $bpc bpc, $ext);
    }};
}

#[cfg(all(feature = "asm", any(target_arch = "arm", target_arch = "aarch64")))]
macro_rules! assign_itx12_fn {
    ($c:ident, $BD:ty, $w:literal, $h:literal, $ext:ident) => {{
        assign_itx2_fn!($c, BD, $w, $h, $ext);
        assign_itx_fn!($c, BD, $w, $h, dct_flipadst, FLIPADST_DCT, $ext);
        assign_itx_fn!($c, BD, $w, $h, dct_adst, ADST_DCT, $ext);
        assign_itx_fn!($c, BD, $w, $h, dct_identity, H_DCT, $ext);
        assign_itx_fn!($c, BD, $w, $h, adst_dct, DCT_ADST, $ext);
        assign_itx_fn!($c, BD, $w, $h, adst_adst, ADST_ADST, $ext);
        assign_itx_fn!($c, BD, $w, $h, adst_flipadst, FLIPADST_ADST, $ext);
        assign_itx_fn!($c, BD, $w, $h, flipadst_dct, DCT_FLIPADST, $ext);
        assign_itx_fn!($c, BD, $w, $h, flipadst_adst, ADST_FLIPADST, $ext);
        assign_itx_fn!($c, BD, $w, $h, flipadst_flipadst, FLIPADST_FLIPADST, $ext);
        assign_itx_fn!($c, BD, $w, $h, identity_dct, V_DCT, $ext);
    }};
}

#[cfg(all(feature = "asm", any(target_arch = "x86", target_arch = "x86_64")))]
macro_rules! assign_itx16_bpc_fn {
    ($c:ident, $w:literal, $h:literal, $bpc:literal bpc, $ext:ident) => {{
        assign_itx12_bpc_fn!($c, $w, $h, $bpc bpc, $ext);
        assign_itx_bpc_fn!($c, $w, $h, adst_identity, H_ADST, $bpc bpc, $ext);
        assign_itx_bpc_fn!($c, $w, $h, flipadst_identity, H_FLIPADST, $bpc bpc, $ext);
        assign_itx_bpc_fn!($c, $w, $h, identity_adst, V_ADST, $bpc bpc, $ext);
        assign_itx_bpc_fn!($c, $w, $h, identity_flipadst, V_FLIPADST, $bpc bpc, $ext);
    }};
}

#[cfg(all(feature = "asm", any(target_arch = "arm", target_arch = "aarch64")))]
macro_rules! assign_itx16_fn {
    ($c:ident, $BD:ty, $w:literal, $h:literal, $ext:ident) => {{
        assign_itx12_fn!($c, BD, $w, $h, $ext);
        assign_itx_fn!($c, BD, $w, $h, adst_identity, H_ADST, $ext);
        assign_itx_fn!($c, BD, $w, $h, flipadst_identity, H_FLIPADST, $ext);
        assign_itx_fn!($c, BD, $w, $h, identity_adst, V_ADST, $ext);
        assign_itx_fn!($c, BD, $w, $h, identity_flipadst, V_FLIPADST, $ext);
    }};
}

impl Rav1dInvTxfmDSPContext {
    const fn assign<const W: usize, const H: usize, BD: BitDepth>(mut self) -> Self {
        let tx = TxfmSize::from_wh(W, H) as usize;

        macro_rules! assign {
            ($type:expr) => {{
                self.itxfm_add[tx][$type as usize] =
                    itxfm::Fn::new(inv_txfm_add_c_erased::<W, H, $type, BD>);
            }};
        }

        let max_wh = if W > H { W } else { H };

        let assign84 = W * H <= 8 * 16;
        let assign16 = assign84 || (W == 16 && H == 16);
        let assign32 = assign16 || max_wh == 32;
        let assign64 = assign32 || max_wh == 64;

        if assign84 {
            assign!(H_FLIPADST);
            assign!(V_FLIPADST);
            assign!(H_ADST);
            assign!(V_ADST);
        }
        if assign16 {
            assign!(DCT_ADST);
            assign!(ADST_DCT);
            assign!(ADST_ADST);
            assign!(ADST_FLIPADST);
            assign!(FLIPADST_ADST);
            assign!(DCT_FLIPADST);
            assign!(FLIPADST_DCT);
            assign!(FLIPADST_FLIPADST);
            assign!(H_DCT);
            assign!(V_DCT);
        }
        if assign32 {
            assign!(IDTX);
        }
        if assign64 {
            assign!(DCT_DCT);
        }

        if W == 4 && H == 4 {
            assign!(WHT_WHT);
        }

        self
    }

    pub const fn default<BD: BitDepth>() -> Self {
        let mut c = Self {
            itxfm_add: [[itxfm::Fn::DEFAULT; N_TX_TYPES_PLUS_LL]; TxfmSize::COUNT],
        };

        c = c.assign::<4, 4, BD>();
        c = c.assign::<4, 8, BD>();
        c = c.assign::<4, 16, BD>();
        c = c.assign::<8, 4, BD>();
        c = c.assign::<8, 8, BD>();
        c = c.assign::<8, 16, BD>();
        c = c.assign::<8, 32, BD>();
        c = c.assign::<16, 4, BD>();
        c = c.assign::<16, 8, BD>();
        c = c.assign::<16, 16, BD>();
        c = c.assign::<16, 32, BD>();
        c = c.assign::<16, 64, BD>();
        c = c.assign::<32, 8, BD>();
        c = c.assign::<32, 16, BD>();
        c = c.assign::<32, 32, BD>();
        c = c.assign::<32, 64, BD>();
        c = c.assign::<64, 16, BD>();
        c = c.assign::<64, 32, BD>();
        c = c.assign::<64, 64, BD>();

        c
    }

    #[cfg(all(feature = "asm", any(target_arch = "x86", target_arch = "x86_64")))]
    #[inline(always)]
    const fn init_x86<BD: BitDepth>(mut self, flags: CpuFlags, bpc: u8) -> Self {
        if !flags.contains(CpuFlags::SSE2) {
            return self;
        }

        assign_itx_fn!(self, BD, 4, 4, wht_wht, WHT_WHT, sse2);

        if !flags.contains(CpuFlags::SSSE3) {
            return self;
        }

        if BD::BITDEPTH == 8 {
            assign_itx16_bpc_fn!(self,  4,  4, 8 bpc, ssse3);
            assign_itx16_bpc_fn!(self,  4,  8, 8 bpc, ssse3);
            assign_itx16_bpc_fn!(self,  8,  4, 8 bpc, ssse3);
            assign_itx16_bpc_fn!(self,  8,  8, 8 bpc, ssse3);
            assign_itx16_bpc_fn!(self,  4, 16, 8 bpc, ssse3);
            assign_itx16_bpc_fn!(self, 16,  4, 8 bpc, ssse3);
            assign_itx16_bpc_fn!(self,  8, 16, 8 bpc, ssse3);
            assign_itx16_bpc_fn!(self, 16,  8, 8 bpc, ssse3);
            assign_itx12_bpc_fn!(self, 16, 16, 8 bpc, ssse3);
            assign_itx2_bpc_fn! (self,  8, 32, 8 bpc, ssse3);
            assign_itx2_bpc_fn! (self, 32,  8, 8 bpc, ssse3);
            assign_itx2_bpc_fn! (self, 16, 32, 8 bpc, ssse3);
            assign_itx2_bpc_fn! (self, 32, 16, 8 bpc, ssse3);
            assign_itx2_bpc_fn! (self, 32, 32, 8 bpc, ssse3);
            assign_itx1_bpc_fn! (self, 16, 64, 8 bpc, ssse3);
            assign_itx1_bpc_fn! (self, 32, 64, 8 bpc, ssse3);
            assign_itx1_bpc_fn! (self, 64, 16, 8 bpc, ssse3);
            assign_itx1_bpc_fn! (self, 64, 32, 8 bpc, ssse3);
            assign_itx1_bpc_fn! (self, 64, 64, 8 bpc, ssse3);
        }

        if !flags.contains(CpuFlags::SSE41) {
            return self;
        }

        if BD::BITDEPTH == 16 {
            if bpc == 10 {
                assign_itx16_bpc_fn!(self,  4,  4, 16 bpc, sse4);
                assign_itx16_bpc_fn!(self,  4,  8, 16 bpc, sse4);
                assign_itx16_bpc_fn!(self,  4, 16, 16 bpc, sse4);
                assign_itx16_bpc_fn!(self,  8,  4, 16 bpc, sse4);
                assign_itx16_bpc_fn!(self,  8,  8, 16 bpc, sse4);
                assign_itx16_bpc_fn!(self,  8, 16, 16 bpc, sse4);
                assign_itx16_bpc_fn!(self, 16,  4, 16 bpc, sse4);
                assign_itx16_bpc_fn!(self, 16,  8, 16 bpc, sse4);
                assign_itx12_bpc_fn!(self, 16, 16, 16 bpc, sse4);
                assign_itx2_bpc_fn! (self,  8, 32, 16 bpc, sse4);
                assign_itx2_bpc_fn! (self, 16, 32, 16 bpc, sse4);
                assign_itx2_bpc_fn! (self, 32,  8, 16 bpc, sse4);
                assign_itx2_bpc_fn! (self, 32, 16, 16 bpc, sse4);
                assign_itx2_bpc_fn! (self, 32, 32, 16 bpc, sse4);
                assign_itx1_bpc_fn! (self, 16, 64, 16 bpc, sse4);
                assign_itx1_bpc_fn! (self, 32, 64, 16 bpc, sse4);
                assign_itx1_bpc_fn! (self, 64, 16, 16 bpc, sse4);
                assign_itx1_bpc_fn! (self, 64, 32, 16 bpc, sse4);
                assign_itx1_bpc_fn! (self, 64, 64, 16 bpc, sse4);
            }
        }

        #[cfg(target_arch = "x86_64")]
        {
            if !flags.contains(CpuFlags::AVX2) {
                return self;
            }

            assign_itx_fn!(self, BD, 4, 4, wht_wht, WHT_WHT, avx2);

            if BD::BITDEPTH == 8 {
                assign_itx16_bpc_fn!(self,  4,  4, 8 bpc, avx2);
                assign_itx16_bpc_fn!(self,  4,  8, 8 bpc, avx2);
                assign_itx16_bpc_fn!(self,  4, 16, 8 bpc, avx2);
                assign_itx16_bpc_fn!(self,  8,  4, 8 bpc, avx2);
                assign_itx16_bpc_fn!(self,  8,  8, 8 bpc, avx2);
                assign_itx16_bpc_fn!(self,  8, 16, 8 bpc, avx2);
                assign_itx16_bpc_fn!(self, 16,  4, 8 bpc, avx2);
                assign_itx16_bpc_fn!(self, 16,  8, 8 bpc, avx2);
                assign_itx12_bpc_fn!(self, 16, 16, 8 bpc, avx2);
                assign_itx2_bpc_fn! (self,  8, 32, 8 bpc, avx2);
                assign_itx2_bpc_fn! (self, 16, 32, 8 bpc, avx2);
                assign_itx2_bpc_fn! (self, 32,  8, 8 bpc, avx2);
                assign_itx2_bpc_fn! (self, 32, 16, 8 bpc, avx2);
                assign_itx2_bpc_fn! (self, 32, 32, 8 bpc, avx2);
                assign_itx1_bpc_fn! (self, 16, 64, 8 bpc, avx2);
                assign_itx1_bpc_fn! (self, 32, 64, 8 bpc, avx2);
                assign_itx1_bpc_fn! (self, 64, 16, 8 bpc, avx2);
                assign_itx1_bpc_fn! (self, 64, 32, 8 bpc, avx2);
                assign_itx1_bpc_fn! (self, 64, 64, 8 bpc, avx2);
            } else {
                if bpc == 10 {
                    assign_itx16_bpc_fn!(self,  4,  4, 10 bpc, avx2);
                    assign_itx16_bpc_fn!(self,  4,  8, 10 bpc, avx2);
                    assign_itx16_bpc_fn!(self,  4, 16, 10 bpc, avx2);
                    assign_itx16_bpc_fn!(self,  8,  4, 10 bpc, avx2);
                    assign_itx16_bpc_fn!(self,  8,  8, 10 bpc, avx2);
                    assign_itx16_bpc_fn!(self,  8, 16, 10 bpc, avx2);
                    assign_itx16_bpc_fn!(self, 16,  4, 10 bpc, avx2);
                    assign_itx16_bpc_fn!(self, 16,  8, 10 bpc, avx2);
                    assign_itx12_bpc_fn!(self, 16, 16, 10 bpc, avx2);
                    assign_itx2_bpc_fn! (self,  8, 32, 10 bpc, avx2);
                    assign_itx2_bpc_fn! (self, 16, 32, 10 bpc, avx2);
                    assign_itx2_bpc_fn! (self, 32,  8, 10 bpc, avx2);
                    assign_itx2_bpc_fn! (self, 32, 16, 10 bpc, avx2);
                    assign_itx2_bpc_fn! (self, 32, 32, 10 bpc, avx2);
                    assign_itx1_bpc_fn! (self, 16, 64, 10 bpc, avx2);
                    assign_itx1_bpc_fn! (self, 32, 64, 10 bpc, avx2);
                    assign_itx1_bpc_fn! (self, 64, 16, 10 bpc, avx2);
                    assign_itx1_bpc_fn! (self, 64, 32, 10 bpc, avx2);
                    assign_itx1_bpc_fn! (self, 64, 64, 10 bpc, avx2);
                } else {
                    assign_itx16_bpc_fn!(self,  4,  4, 12 bpc, avx2);
                    assign_itx16_bpc_fn!(self,  4,  8, 12 bpc, avx2);
                    assign_itx16_bpc_fn!(self,  4, 16, 12 bpc, avx2);
                    assign_itx16_bpc_fn!(self,  8,  4, 12 bpc, avx2);
                    assign_itx16_bpc_fn!(self,  8,  8, 12 bpc, avx2);
                    assign_itx16_bpc_fn!(self,  8, 16, 12 bpc, avx2);
                    assign_itx16_bpc_fn!(self, 16,  4, 12 bpc, avx2);
                    assign_itx16_bpc_fn!(self, 16,  8, 12 bpc, avx2);
                    assign_itx12_bpc_fn!(self, 16, 16, 12 bpc, avx2);
                    assign_itx2_bpc_fn! (self,  8, 32, 12 bpc, avx2);
                    assign_itx2_bpc_fn! (self, 32,  8, 12 bpc, avx2);
                    assign_itx_bpc_fn!  (self, 16, 32, identity_identity, IDTX, 12 bpc, avx2);
                    assign_itx_bpc_fn!  (self, 32, 16, identity_identity, IDTX, 12 bpc, avx2);
                    assign_itx_bpc_fn!  (self, 32, 32, identity_identity, IDTX, 12 bpc, avx2);
                }
            }

            if !flags.contains(CpuFlags::AVX512ICL) {
                return self;
            }

            if BD::BITDEPTH == 8 {
                assign_itx16_bpc_fn!(self,  4,  4, 8 bpc, avx512icl); // no wht
                assign_itx16_bpc_fn!(self,  4,  8, 8 bpc, avx512icl);
                assign_itx16_bpc_fn!(self,  4, 16, 8 bpc, avx512icl);
                assign_itx16_bpc_fn!(self,  8,  4, 8 bpc, avx512icl);
                assign_itx16_bpc_fn!(self,  8,  8, 8 bpc, avx512icl);
                assign_itx16_bpc_fn!(self,  8, 16, 8 bpc, avx512icl);
                assign_itx16_bpc_fn!(self, 16,  4, 8 bpc, avx512icl);
                assign_itx16_bpc_fn!(self, 16,  8, 8 bpc, avx512icl);
                assign_itx12_bpc_fn!(self, 16, 16, 8 bpc, avx512icl);
                assign_itx2_bpc_fn! (self,  8, 32, 8 bpc, avx512icl);
                assign_itx2_bpc_fn! (self, 16, 32, 8 bpc, avx512icl);
                assign_itx2_bpc_fn! (self, 32,  8, 8 bpc, avx512icl);
                assign_itx2_bpc_fn! (self, 32, 16, 8 bpc, avx512icl);
                assign_itx2_bpc_fn! (self, 32, 32, 8 bpc, avx512icl);
                assign_itx1_bpc_fn! (self, 16, 64, 8 bpc, avx512icl);
                assign_itx1_bpc_fn! (self, 32, 64, 8 bpc, avx512icl);
                assign_itx1_bpc_fn! (self, 64, 16, 8 bpc, avx512icl);
                assign_itx1_bpc_fn! (self, 64, 32, 8 bpc, avx512icl);
                assign_itx1_bpc_fn! (self, 64, 64, 8 bpc, avx512icl);
            } else {
                if bpc == 10 {
                    assign_itx16_bpc_fn!(self,  8,  8, 10 bpc, avx512icl);
                    assign_itx16_bpc_fn!(self,  8, 16, 10 bpc, avx512icl);
                    assign_itx16_bpc_fn!(self, 16,  8, 10 bpc, avx512icl);
                    assign_itx12_bpc_fn!(self, 16, 16, 10 bpc, avx512icl);
                    assign_itx2_bpc_fn! (self,  8, 32, 10 bpc, avx512icl);
                    assign_itx2_bpc_fn! (self, 16, 32, 10 bpc, avx512icl);
                    assign_itx2_bpc_fn! (self, 32,  8, 10 bpc, avx512icl);
                    assign_itx2_bpc_fn! (self, 32, 16, 10 bpc, avx512icl);
                    assign_itx2_bpc_fn! (self, 32, 32, 10 bpc, avx512icl);
                    assign_itx1_bpc_fn! (self, 16, 64, 10 bpc, avx512icl);
                    assign_itx1_bpc_fn! (self, 32, 64, 10 bpc, avx512icl);
                    assign_itx1_bpc_fn! (self, 64, 16, 10 bpc, avx512icl);
                    assign_itx1_bpc_fn! (self, 64, 32, 10 bpc, avx512icl);
                    assign_itx1_bpc_fn! (self, 64, 64, 10 bpc, avx512icl);
                }
            }
        }

        self
    }

    #[cfg(all(feature = "asm", any(target_arch = "arm", target_arch = "aarch64")))]
    #[inline(always)]
    const fn init_arm<BD: BitDepth>(mut self, flags: CpuFlags, bpc: u8) -> Self {
        if !flags.contains(CpuFlags::NEON) {
            return self;
        }

        assign_itx_fn!(self, BD, 4, 4, wht_wht, WHT_WHT, neon);

        if BD::BITDEPTH == 16 && bpc != 10 {
            return self;
        }

        #[rustfmt::skip]
        const fn assign<BD: BitDepth>(mut c: Rav1dInvTxfmDSPContext) -> Rav1dInvTxfmDSPContext {
            assign_itx16_fn!(c, BD,  4,  4, neon);
            assign_itx16_fn!(c, BD,  4,  8, neon);
            assign_itx16_fn!(c, BD,  4, 16, neon);
            assign_itx16_fn!(c, BD,  8,  4, neon);
            assign_itx16_fn!(c, BD,  8,  8, neon);
            assign_itx16_fn!(c, BD,  8, 16, neon);
            assign_itx16_fn!(c, BD, 16,  4, neon);
            assign_itx16_fn!(c, BD, 16,  8, neon);
            assign_itx12_fn!(c, BD, 16, 16, neon);
            assign_itx2_fn! (c, BD,  8, 32, neon);
            assign_itx2_fn! (c, BD, 16, 32, neon);
            assign_itx2_fn! (c, BD, 32,  8, neon);
            assign_itx2_fn! (c, BD, 32, 16, neon);
            assign_itx2_fn! (c, BD, 32, 32, neon);
            assign_itx1_fn! (c, BD, 16, 64, neon);
            assign_itx1_fn! (c, BD, 32, 64, neon);
            assign_itx1_fn! (c, BD, 64, 16, neon);
            assign_itx1_fn! (c, BD, 64, 32, neon);
            assign_itx1_fn! (c, BD, 64, 64, neon);

            c
        }

        assign::<BD>(self)
    }

    /// Safe SIMD initialization for x86_64 without hand-written assembly.
    /// Uses Rust intrinsics instead.
    #[cfg(all(not(feature = "asm"), feature = "c-ffi", target_arch = "x86_64"))]
    #[inline(always)]
    const fn init_x86_safe_simd<BD: BitDepth>(mut self, flags: CpuFlags) -> Self {
        use crate::include::common::bitdepth::BPC;
        use crate::src::safe_simd::itx as safe_itx;

        if !flags.contains(CpuFlags::AVX2) {
            return self;
        }

        // Handle 8bpc and 16bpc separately
        match BD::BPC {
            BPC::BPC8 => return self.init_x86_safe_simd_8bpc(flags),
            BPC::BPC16 => return self.init_x86_safe_simd_16bpc(flags),
        }
    }

    /// Safe SIMD 8bpc initialization
    #[cfg(all(not(feature = "asm"), feature = "c-ffi", target_arch = "x86_64"))]
    #[inline(always)]
    const fn init_x86_safe_simd_8bpc(mut self, _flags: CpuFlags) -> Self {
        use crate::src::safe_simd::itx as safe_itx;

        let tx_4x4 = TxfmSize::from_wh(4, 4) as usize;

        // DCT_DCT 4x4
        self.itxfm_add[tx_4x4][DCT_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_dct_4x4_8bpc_avx2);

        // WHT_WHT 4x4
        self.itxfm_add[tx_4x4][WHT_WHT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_wht_wht_4x4_8bpc_avx2);

        // IDTX 4x4
        self.itxfm_add[tx_4x4][IDTX as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_identity_4x4_8bpc_avx2);
        // ADST_DCT 4x4
        self.itxfm_add[tx_4x4][ADST_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_dct_4x4_8bpc_avx2);

        // DCT_ADST 4x4
        self.itxfm_add[tx_4x4][DCT_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_adst_4x4_8bpc_avx2);

        // ADST_ADST 4x4
        self.itxfm_add[tx_4x4][ADST_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_adst_4x4_8bpc_avx2);
        // FLIPADST_DCT 4x4
        self.itxfm_add[tx_4x4][FLIPADST_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_dct_4x4_8bpc_avx2);

        // DCT_FLIPADST 4x4
        self.itxfm_add[tx_4x4][DCT_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_flipadst_4x4_8bpc_avx2);

        // FLIPADST_FLIPADST 4x4
        self.itxfm_add[tx_4x4][FLIPADST_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_flipadst_4x4_8bpc_avx2);

        // ADST_FLIPADST 4x4
        self.itxfm_add[tx_4x4][ADST_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_flipadst_4x4_8bpc_avx2);

        // FLIPADST_ADST 4x4
        self.itxfm_add[tx_4x4][FLIPADST_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_adst_4x4_8bpc_avx2);
        // V_ADST, H_ADST, V_FLIPADST, H_FLIPADST 4x4
        self.itxfm_add[tx_4x4][H_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_adst_4x4_8bpc_avx2);
        self.itxfm_add[tx_4x4][V_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_identity_4x4_8bpc_avx2);
        self.itxfm_add[tx_4x4][H_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_flipadst_4x4_8bpc_avx2);
        self.itxfm_add[tx_4x4][V_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_identity_4x4_8bpc_avx2);

        // V_DCT, H_DCT 4x4
        self.itxfm_add[tx_4x4][H_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_dct_4x4_8bpc_avx2);
        self.itxfm_add[tx_4x4][V_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_identity_4x4_8bpc_avx2);
        // 8x8 transforms
        let tx_8x8 = TxfmSize::from_wh(8, 8) as usize;

        // DCT_DCT 8x8
        self.itxfm_add[tx_8x8][DCT_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_dct_8x8_8bpc_avx2);

        // IDTX 8x8
        self.itxfm_add[tx_8x8][IDTX as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_identity_8x8_8bpc_avx2);
        // ADST/FlipADST 8x8
        self.itxfm_add[tx_8x8][ADST_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_dct_8x8_8bpc_avx2);
        self.itxfm_add[tx_8x8][DCT_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_adst_8x8_8bpc_avx2);
        self.itxfm_add[tx_8x8][ADST_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_adst_8x8_8bpc_avx2);
        self.itxfm_add[tx_8x8][FLIPADST_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_dct_8x8_8bpc_avx2);
        self.itxfm_add[tx_8x8][DCT_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_flipadst_8x8_8bpc_avx2);
        self.itxfm_add[tx_8x8][FLIPADST_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_flipadst_8x8_8bpc_avx2);
        self.itxfm_add[tx_8x8][ADST_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_flipadst_8x8_8bpc_avx2);
        self.itxfm_add[tx_8x8][FLIPADST_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_adst_8x8_8bpc_avx2);
        // V/H ADST/DCT 8x8
        self.itxfm_add[tx_8x8][H_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_adst_8x8_8bpc_avx2);
        self.itxfm_add[tx_8x8][V_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_identity_8x8_8bpc_avx2);
        self.itxfm_add[tx_8x8][H_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_flipadst_8x8_8bpc_avx2);
        self.itxfm_add[tx_8x8][V_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_identity_8x8_8bpc_avx2);
        self.itxfm_add[tx_8x8][H_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_dct_8x8_8bpc_avx2);
        self.itxfm_add[tx_8x8][V_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_identity_8x8_8bpc_avx2);

        // 16x16 transforms
        let tx_16x16 = TxfmSize::from_wh(16, 16) as usize;

        // DCT_DCT 16x16
        self.itxfm_add[tx_16x16][DCT_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_dct_16x16_8bpc_avx2);

        // IDTX 16x16
        self.itxfm_add[tx_16x16][IDTX as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_identity_16x16_8bpc_avx2);

        // ADST/FlipADST combinations 16x16
        self.itxfm_add[tx_16x16][ADST_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_dct_16x16_8bpc_avx2);
        self.itxfm_add[tx_16x16][DCT_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_adst_16x16_8bpc_avx2);
        self.itxfm_add[tx_16x16][ADST_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_adst_16x16_8bpc_avx2);
        self.itxfm_add[tx_16x16][FLIPADST_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_dct_16x16_8bpc_avx2);
        self.itxfm_add[tx_16x16][DCT_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_flipadst_16x16_8bpc_avx2);
        self.itxfm_add[tx_16x16][FLIPADST_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_flipadst_16x16_8bpc_avx2);
        self.itxfm_add[tx_16x16][ADST_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_flipadst_16x16_8bpc_avx2);
        self.itxfm_add[tx_16x16][FLIPADST_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_adst_16x16_8bpc_avx2);

        // V/H transforms 16x16 (Identity + DCT/ADST/FlipADST)
        self.itxfm_add[tx_16x16][H_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_dct_16x16_8bpc_avx2);
        self.itxfm_add[tx_16x16][V_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_identity_16x16_8bpc_avx2);
        self.itxfm_add[tx_16x16][H_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_adst_16x16_8bpc_avx2);
        self.itxfm_add[tx_16x16][V_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_identity_16x16_8bpc_avx2);
        self.itxfm_add[tx_16x16][H_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_flipadst_16x16_8bpc_avx2);
        self.itxfm_add[tx_16x16][V_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_identity_16x16_8bpc_avx2);

        // 32x32 transforms
        let tx_32x32 = TxfmSize::from_wh(32, 32) as usize;

        // DCT_DCT 32x32
        self.itxfm_add[tx_32x32][DCT_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_dct_32x32_8bpc_avx2);

        // IDTX 32x32
        self.itxfm_add[tx_32x32][IDTX as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_identity_32x32_8bpc_avx2);

        // 64x64 transforms
        let tx_64x64 = TxfmSize::from_wh(64, 64) as usize;

        // DCT_DCT 64x64
        self.itxfm_add[tx_64x64][DCT_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_dct_64x64_8bpc_avx2);

        // Rectangular transforms: 4x8, 8x4, 8x16, 16x8
        let tx_4x8 = TxfmSize::from_wh(4, 8) as usize;
        let tx_8x4 = TxfmSize::from_wh(8, 4) as usize;
        let tx_8x16 = TxfmSize::from_wh(8, 16) as usize;
        let tx_16x8 = TxfmSize::from_wh(16, 8) as usize;

        // DCT_DCT 4x8
        self.itxfm_add[tx_4x8][DCT_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_dct_4x8_8bpc_avx2);
        // DCT_DCT 8x4
        self.itxfm_add[tx_8x4][DCT_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_dct_8x4_8bpc_avx2);

        // ADST/FLIPADST variants for 4x8
        self.itxfm_add[tx_4x8][ADST_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_dct_4x8_8bpc_avx2);
        self.itxfm_add[tx_4x8][DCT_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_adst_4x8_8bpc_avx2);
        self.itxfm_add[tx_4x8][ADST_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_adst_4x8_8bpc_avx2);
        self.itxfm_add[tx_4x8][FLIPADST_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_dct_4x8_8bpc_avx2);
        self.itxfm_add[tx_4x8][DCT_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_flipadst_4x8_8bpc_avx2);
        self.itxfm_add[tx_4x8][FLIPADST_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_flipadst_4x8_8bpc_avx2);
        self.itxfm_add[tx_4x8][ADST_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_flipadst_4x8_8bpc_avx2);
        self.itxfm_add[tx_4x8][FLIPADST_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_adst_4x8_8bpc_avx2);

        // ADST/FLIPADST variants for 8x4
        self.itxfm_add[tx_8x4][ADST_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_dct_8x4_8bpc_avx2);
        self.itxfm_add[tx_8x4][DCT_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_adst_8x4_8bpc_avx2);
        self.itxfm_add[tx_8x4][ADST_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_adst_8x4_8bpc_avx2);
        self.itxfm_add[tx_8x4][FLIPADST_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_dct_8x4_8bpc_avx2);
        self.itxfm_add[tx_8x4][DCT_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_flipadst_8x4_8bpc_avx2);
        self.itxfm_add[tx_8x4][FLIPADST_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_flipadst_8x4_8bpc_avx2);
        self.itxfm_add[tx_8x4][ADST_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_flipadst_8x4_8bpc_avx2);
        self.itxfm_add[tx_8x4][FLIPADST_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_adst_8x4_8bpc_avx2);

        // Identity transforms for 4x8 and 8x4
        self.itxfm_add[tx_4x8][IDTX as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_identity_4x8_8bpc_avx2);
        self.itxfm_add[tx_8x4][IDTX as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_identity_8x4_8bpc_avx2);
        self.itxfm_add[tx_4x8][H_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_dct_4x8_8bpc_avx2);
        self.itxfm_add[tx_4x8][V_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_identity_4x8_8bpc_avx2);
        self.itxfm_add[tx_8x4][H_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_dct_8x4_8bpc_avx2);
        self.itxfm_add[tx_8x4][V_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_identity_8x4_8bpc_avx2);
        self.itxfm_add[tx_4x8][H_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_adst_4x8_8bpc_avx2);
        self.itxfm_add[tx_4x8][V_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_identity_4x8_8bpc_avx2);
        self.itxfm_add[tx_4x8][H_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_flipadst_4x8_8bpc_avx2);
        self.itxfm_add[tx_4x8][V_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_identity_4x8_8bpc_avx2);
        self.itxfm_add[tx_8x4][H_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_adst_8x4_8bpc_avx2);
        self.itxfm_add[tx_8x4][V_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_identity_8x4_8bpc_avx2);
        self.itxfm_add[tx_8x4][H_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_flipadst_8x4_8bpc_avx2);
        self.itxfm_add[tx_8x4][V_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_identity_8x4_8bpc_avx2);

        // DCT_DCT 8x16
        self.itxfm_add[tx_8x16][DCT_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_dct_8x16_8bpc_avx2);
        // DCT_DCT 16x8
        self.itxfm_add[tx_16x8][DCT_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_dct_16x8_8bpc_avx2);

        // ADST/FLIPADST variants for 8x16
        self.itxfm_add[tx_8x16][ADST_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_dct_8x16_8bpc_avx2);
        self.itxfm_add[tx_8x16][DCT_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_adst_8x16_8bpc_avx2);
        self.itxfm_add[tx_8x16][ADST_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_adst_8x16_8bpc_avx2);
        self.itxfm_add[tx_8x16][FLIPADST_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_dct_8x16_8bpc_avx2);
        self.itxfm_add[tx_8x16][DCT_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_flipadst_8x16_8bpc_avx2);
        self.itxfm_add[tx_8x16][FLIPADST_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_flipadst_8x16_8bpc_avx2);
        self.itxfm_add[tx_8x16][ADST_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_flipadst_8x16_8bpc_avx2);
        self.itxfm_add[tx_8x16][FLIPADST_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_adst_8x16_8bpc_avx2);

        // ADST/FLIPADST variants for 16x8
        self.itxfm_add[tx_16x8][ADST_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_dct_16x8_8bpc_avx2);
        self.itxfm_add[tx_16x8][DCT_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_adst_16x8_8bpc_avx2);
        self.itxfm_add[tx_16x8][ADST_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_adst_16x8_8bpc_avx2);
        self.itxfm_add[tx_16x8][FLIPADST_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_dct_16x8_8bpc_avx2);
        self.itxfm_add[tx_16x8][DCT_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_flipadst_16x8_8bpc_avx2);
        self.itxfm_add[tx_16x8][FLIPADST_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_flipadst_16x8_8bpc_avx2);
        self.itxfm_add[tx_16x8][ADST_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_flipadst_16x8_8bpc_avx2);
        self.itxfm_add[tx_16x8][FLIPADST_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_adst_16x8_8bpc_avx2);

        // Identity transforms for 8x16 and 16x8
        self.itxfm_add[tx_8x16][IDTX as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_identity_8x16_8bpc_avx2);
        self.itxfm_add[tx_16x8][IDTX as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_identity_16x8_8bpc_avx2);
        self.itxfm_add[tx_8x16][H_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_dct_8x16_8bpc_avx2);
        self.itxfm_add[tx_8x16][V_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_identity_8x16_8bpc_avx2);
        self.itxfm_add[tx_16x8][H_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_dct_16x8_8bpc_avx2);
        self.itxfm_add[tx_16x8][V_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_identity_16x8_8bpc_avx2);
        self.itxfm_add[tx_8x16][H_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_adst_8x16_8bpc_avx2);
        self.itxfm_add[tx_8x16][V_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_identity_8x16_8bpc_avx2);
        self.itxfm_add[tx_8x16][H_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_flipadst_8x16_8bpc_avx2);
        self.itxfm_add[tx_8x16][V_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_identity_8x16_8bpc_avx2);
        self.itxfm_add[tx_16x8][H_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_adst_16x8_8bpc_avx2);
        self.itxfm_add[tx_16x8][V_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_identity_16x8_8bpc_avx2);
        self.itxfm_add[tx_16x8][H_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_flipadst_16x8_8bpc_avx2);
        self.itxfm_add[tx_16x8][V_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_identity_16x8_8bpc_avx2);

        // Larger rectangular transforms: 16x32, 32x16, 32x64, 64x32
        let tx_16x32 = TxfmSize::from_wh(16, 32) as usize;
        let tx_32x16 = TxfmSize::from_wh(32, 16) as usize;
        let tx_32x64 = TxfmSize::from_wh(32, 64) as usize;
        let tx_64x32 = TxfmSize::from_wh(64, 32) as usize;

        // DCT_DCT 16x32
        self.itxfm_add[tx_16x32][DCT_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_dct_16x32_8bpc_avx2);
        // DCT_DCT 32x16
        self.itxfm_add[tx_32x16][DCT_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_dct_32x16_8bpc_avx2);
        // DCT_DCT 32x64
        self.itxfm_add[tx_32x64][DCT_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_dct_32x64_8bpc_avx2);
        // DCT_DCT 64x32
        self.itxfm_add[tx_64x32][DCT_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_dct_64x32_8bpc_avx2);

        // IDTX 16x32 and 32x16
        self.itxfm_add[tx_16x32][IDTX as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_identity_16x32_8bpc_avx2);
        self.itxfm_add[tx_32x16][IDTX as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_identity_32x16_8bpc_avx2);

        // 4:1 aspect ratio rectangles: 4x16, 16x4, 8x32, 32x8
        let tx_4x16 = TxfmSize::from_wh(4, 16) as usize;
        let tx_16x4 = TxfmSize::from_wh(16, 4) as usize;
        let tx_8x32 = TxfmSize::from_wh(8, 32) as usize;
        let tx_32x8 = TxfmSize::from_wh(32, 8) as usize;

        // DCT_DCT 4x16
        self.itxfm_add[tx_4x16][DCT_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_dct_4x16_8bpc_avx2);
        // DCT_DCT 16x4
        self.itxfm_add[tx_16x4][DCT_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_dct_16x4_8bpc_avx2);

        // ADST/FLIPADST variants for 4x16
        self.itxfm_add[tx_4x16][ADST_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_dct_4x16_8bpc_avx2);
        self.itxfm_add[tx_4x16][DCT_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_adst_4x16_8bpc_avx2);
        self.itxfm_add[tx_4x16][ADST_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_adst_4x16_8bpc_avx2);
        self.itxfm_add[tx_4x16][FLIPADST_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_dct_4x16_8bpc_avx2);
        self.itxfm_add[tx_4x16][DCT_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_flipadst_4x16_8bpc_avx2);
        self.itxfm_add[tx_4x16][FLIPADST_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_flipadst_4x16_8bpc_avx2);
        self.itxfm_add[tx_4x16][ADST_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_flipadst_4x16_8bpc_avx2);
        self.itxfm_add[tx_4x16][FLIPADST_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_adst_4x16_8bpc_avx2);

        // ADST/FLIPADST variants for 16x4
        self.itxfm_add[tx_16x4][ADST_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_dct_16x4_8bpc_avx2);
        self.itxfm_add[tx_16x4][DCT_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_adst_16x4_8bpc_avx2);
        self.itxfm_add[tx_16x4][ADST_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_adst_16x4_8bpc_avx2);
        self.itxfm_add[tx_16x4][FLIPADST_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_dct_16x4_8bpc_avx2);
        self.itxfm_add[tx_16x4][DCT_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_flipadst_16x4_8bpc_avx2);
        self.itxfm_add[tx_16x4][FLIPADST_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_flipadst_16x4_8bpc_avx2);
        self.itxfm_add[tx_16x4][ADST_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_flipadst_16x4_8bpc_avx2);
        self.itxfm_add[tx_16x4][FLIPADST_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_adst_16x4_8bpc_avx2);

        // Identity transforms for 4x16 and 16x4
        self.itxfm_add[tx_4x16][IDTX as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_identity_4x16_8bpc_avx2);
        self.itxfm_add[tx_16x4][IDTX as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_identity_16x4_8bpc_avx2);
        self.itxfm_add[tx_4x16][H_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_dct_4x16_8bpc_avx2);
        self.itxfm_add[tx_4x16][V_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_identity_4x16_8bpc_avx2);
        self.itxfm_add[tx_16x4][H_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_dct_16x4_8bpc_avx2);
        self.itxfm_add[tx_16x4][V_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_identity_16x4_8bpc_avx2);
        self.itxfm_add[tx_4x16][H_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_adst_4x16_8bpc_avx2);
        self.itxfm_add[tx_4x16][V_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_identity_4x16_8bpc_avx2);
        self.itxfm_add[tx_4x16][H_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_flipadst_4x16_8bpc_avx2);
        self.itxfm_add[tx_4x16][V_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_identity_4x16_8bpc_avx2);
        self.itxfm_add[tx_16x4][H_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_adst_16x4_8bpc_avx2);
        self.itxfm_add[tx_16x4][V_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_identity_16x4_8bpc_avx2);
        self.itxfm_add[tx_16x4][H_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_flipadst_16x4_8bpc_avx2);
        self.itxfm_add[tx_16x4][V_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_identity_16x4_8bpc_avx2);

        // DCT_DCT 8x32
        self.itxfm_add[tx_8x32][DCT_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_dct_8x32_8bpc_avx2);
        // DCT_DCT 32x8
        self.itxfm_add[tx_32x8][DCT_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_dct_32x8_8bpc_avx2);

        // IDTX 8x32 and 32x8
        self.itxfm_add[tx_8x32][IDTX as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_identity_8x32_8bpc_avx2);
        self.itxfm_add[tx_32x8][IDTX as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_identity_32x8_8bpc_avx2);

        // Largest rectangular transforms: 16x64, 64x16
        let tx_16x64 = TxfmSize::from_wh(16, 64) as usize;
        let tx_64x16 = TxfmSize::from_wh(64, 16) as usize;

        // DCT_DCT 16x64
        self.itxfm_add[tx_16x64][DCT_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_dct_16x64_8bpc_avx2);
        // DCT_DCT 64x16
        self.itxfm_add[tx_64x16][DCT_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_dct_64x16_8bpc_avx2);

        self
    }

    /// Safe SIMD 16bpc initialization
    #[cfg(all(not(feature = "asm"), feature = "c-ffi", target_arch = "x86_64"))]
    #[inline(always)]
    const fn init_x86_safe_simd_16bpc(mut self, _flags: CpuFlags) -> Self {
        use crate::src::safe_simd::itx as safe_itx;

        // Square transforms
        let tx_4x4 = TxfmSize::from_wh(4, 4) as usize;
        let tx_8x8 = TxfmSize::from_wh(8, 8) as usize;
        let tx_16x16 = TxfmSize::from_wh(16, 16) as usize;
        let tx_32x32 = TxfmSize::from_wh(32, 32) as usize;
        let tx_64x64 = TxfmSize::from_wh(64, 64) as usize;

        // Rectangular transforms
        let tx_4x8 = TxfmSize::from_wh(4, 8) as usize;
        let tx_8x4 = TxfmSize::from_wh(8, 4) as usize;
        let tx_8x16 = TxfmSize::from_wh(8, 16) as usize;
        let tx_16x8 = TxfmSize::from_wh(16, 8) as usize;
        let tx_4x16 = TxfmSize::from_wh(4, 16) as usize;
        let tx_16x4 = TxfmSize::from_wh(16, 4) as usize;
        let tx_16x32 = TxfmSize::from_wh(16, 32) as usize;
        let tx_32x16 = TxfmSize::from_wh(32, 16) as usize;
        let tx_8x32 = TxfmSize::from_wh(8, 32) as usize;
        let tx_32x8 = TxfmSize::from_wh(32, 8) as usize;
        let tx_32x64 = TxfmSize::from_wh(32, 64) as usize;
        let tx_64x32 = TxfmSize::from_wh(64, 32) as usize;
        let tx_16x64 = TxfmSize::from_wh(16, 64) as usize;
        let tx_64x16 = TxfmSize::from_wh(64, 16) as usize;

        // DCT_DCT square transforms 16bpc
        self.itxfm_add[tx_4x4][DCT_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_dct_4x4_16bpc_avx2);
        // WHT_WHT 4x4 16bpc
        self.itxfm_add[tx_4x4][WHT_WHT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_wht_wht_4x4_16bpc_avx2);
        self.itxfm_add[tx_8x8][DCT_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_dct_8x8_16bpc_avx2);
        self.itxfm_add[tx_16x16][DCT_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_dct_16x16_16bpc_avx2);
        self.itxfm_add[tx_32x32][DCT_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_dct_32x32_16bpc_avx2);
        self.itxfm_add[tx_64x64][DCT_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_dct_64x64_16bpc_avx2);

        // DCT_DCT rectangular transforms 16bpc
        self.itxfm_add[tx_4x8][DCT_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_dct_4x8_16bpc_avx2);
        self.itxfm_add[tx_8x4][DCT_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_dct_8x4_16bpc_avx2);
        self.itxfm_add[tx_8x16][DCT_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_dct_8x16_16bpc_avx2);
        self.itxfm_add[tx_16x8][DCT_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_dct_16x8_16bpc_avx2);
        self.itxfm_add[tx_4x16][DCT_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_dct_4x16_16bpc_avx2);
        self.itxfm_add[tx_16x4][DCT_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_dct_16x4_16bpc_avx2);
        self.itxfm_add[tx_16x32][DCT_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_dct_16x32_16bpc_avx2);
        self.itxfm_add[tx_32x16][DCT_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_dct_32x16_16bpc_avx2);
        self.itxfm_add[tx_8x32][DCT_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_dct_8x32_16bpc_avx2);
        self.itxfm_add[tx_32x8][DCT_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_dct_32x8_16bpc_avx2);
        self.itxfm_add[tx_32x64][DCT_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_dct_32x64_16bpc_avx2);
        self.itxfm_add[tx_64x32][DCT_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_dct_64x32_16bpc_avx2);
        self.itxfm_add[tx_16x64][DCT_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_dct_16x64_16bpc_avx2);
        self.itxfm_add[tx_64x16][DCT_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_dct_64x16_16bpc_avx2);

        // 8x8 ADST/FLIPADST transforms 16bpc
        self.itxfm_add[tx_8x8][ADST_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_dct_8x8_16bpc_avx2);
        self.itxfm_add[tx_8x8][DCT_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_adst_8x8_16bpc_avx2);
        self.itxfm_add[tx_8x8][ADST_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_adst_8x8_16bpc_avx2);
        self.itxfm_add[tx_8x8][FLIPADST_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_dct_8x8_16bpc_avx2);
        self.itxfm_add[tx_8x8][DCT_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_flipadst_8x8_16bpc_avx2);
        self.itxfm_add[tx_8x8][FLIPADST_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_flipadst_8x8_16bpc_avx2);
        self.itxfm_add[tx_8x8][ADST_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_flipadst_8x8_16bpc_avx2);
        self.itxfm_add[tx_8x8][FLIPADST_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_adst_8x8_16bpc_avx2);

        // 4x4 ADST/FLIPADST transforms 16bpc
        self.itxfm_add[tx_4x4][ADST_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_dct_4x4_16bpc_avx2);
        self.itxfm_add[tx_4x4][DCT_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_adst_4x4_16bpc_avx2);
        self.itxfm_add[tx_4x4][ADST_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_adst_4x4_16bpc_avx2);
        self.itxfm_add[tx_4x4][FLIPADST_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_dct_4x4_16bpc_avx2);
        self.itxfm_add[tx_4x4][DCT_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_flipadst_4x4_16bpc_avx2);
        self.itxfm_add[tx_4x4][FLIPADST_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_flipadst_4x4_16bpc_avx2);
        self.itxfm_add[tx_4x4][ADST_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_flipadst_4x4_16bpc_avx2);
        self.itxfm_add[tx_4x4][FLIPADST_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_adst_4x4_16bpc_avx2);

        // 16x16 ADST/FLIPADST transforms 16bpc
        self.itxfm_add[tx_16x16][ADST_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_dct_16x16_16bpc_avx2);
        self.itxfm_add[tx_16x16][DCT_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_adst_16x16_16bpc_avx2);
        self.itxfm_add[tx_16x16][ADST_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_adst_16x16_16bpc_avx2);
        self.itxfm_add[tx_16x16][FLIPADST_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_dct_16x16_16bpc_avx2);
        self.itxfm_add[tx_16x16][DCT_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_flipadst_16x16_16bpc_avx2);
        self.itxfm_add[tx_16x16][FLIPADST_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_flipadst_16x16_16bpc_avx2);
        self.itxfm_add[tx_16x16][ADST_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_flipadst_16x16_16bpc_avx2);
        self.itxfm_add[tx_16x16][FLIPADST_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_adst_16x16_16bpc_avx2);

        // IDTX (identity) transforms 16bpc - square
        self.itxfm_add[tx_4x4][IDTX as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_identity_4x4_16bpc_avx2);
        self.itxfm_add[tx_8x8][IDTX as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_identity_8x8_16bpc_avx2);
        self.itxfm_add[tx_16x16][IDTX as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_identity_16x16_16bpc_avx2);
        self.itxfm_add[tx_32x32][IDTX as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_identity_32x32_16bpc_avx2);

        // IDTX (identity) transforms 16bpc - rectangular
        self.itxfm_add[tx_4x8][IDTX as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_identity_4x8_16bpc_avx2);
        self.itxfm_add[tx_8x4][IDTX as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_identity_8x4_16bpc_avx2);
        self.itxfm_add[tx_8x16][IDTX as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_identity_8x16_16bpc_avx2);
        self.itxfm_add[tx_16x8][IDTX as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_identity_16x8_16bpc_avx2);
        self.itxfm_add[tx_4x16][IDTX as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_identity_4x16_16bpc_avx2);
        self.itxfm_add[tx_16x4][IDTX as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_identity_16x4_16bpc_avx2);
        self.itxfm_add[tx_16x32][IDTX as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_identity_16x32_16bpc_avx2);
        self.itxfm_add[tx_32x16][IDTX as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_identity_32x16_16bpc_avx2);
        self.itxfm_add[tx_8x32][IDTX as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_identity_8x32_16bpc_avx2);
        self.itxfm_add[tx_32x8][IDTX as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_identity_32x8_16bpc_avx2);

        // 4x8 ADST/FLIPADST transforms 16bpc
        self.itxfm_add[tx_4x8][ADST_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_dct_4x8_16bpc_avx2);
        self.itxfm_add[tx_4x8][DCT_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_adst_4x8_16bpc_avx2);
        self.itxfm_add[tx_4x8][ADST_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_adst_4x8_16bpc_avx2);
        self.itxfm_add[tx_4x8][FLIPADST_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_dct_4x8_16bpc_avx2);
        self.itxfm_add[tx_4x8][DCT_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_flipadst_4x8_16bpc_avx2);
        self.itxfm_add[tx_4x8][FLIPADST_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_flipadst_4x8_16bpc_avx2);
        self.itxfm_add[tx_4x8][ADST_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_flipadst_4x8_16bpc_avx2);
        self.itxfm_add[tx_4x8][FLIPADST_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_adst_4x8_16bpc_avx2);

        // 8x4 ADST/FLIPADST transforms 16bpc
        self.itxfm_add[tx_8x4][ADST_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_dct_8x4_16bpc_avx2);
        self.itxfm_add[tx_8x4][DCT_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_adst_8x4_16bpc_avx2);
        self.itxfm_add[tx_8x4][ADST_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_adst_8x4_16bpc_avx2);
        self.itxfm_add[tx_8x4][FLIPADST_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_dct_8x4_16bpc_avx2);
        self.itxfm_add[tx_8x4][DCT_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_flipadst_8x4_16bpc_avx2);
        self.itxfm_add[tx_8x4][FLIPADST_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_flipadst_8x4_16bpc_avx2);
        self.itxfm_add[tx_8x4][ADST_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_flipadst_8x4_16bpc_avx2);
        self.itxfm_add[tx_8x4][FLIPADST_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_adst_8x4_16bpc_avx2);

        // 8x16 ADST/FLIPADST transforms 16bpc
        self.itxfm_add[tx_8x16][ADST_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_dct_8x16_16bpc_avx2);
        self.itxfm_add[tx_8x16][DCT_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_adst_8x16_16bpc_avx2);
        self.itxfm_add[tx_8x16][ADST_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_adst_8x16_16bpc_avx2);
        self.itxfm_add[tx_8x16][FLIPADST_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_dct_8x16_16bpc_avx2);
        self.itxfm_add[tx_8x16][DCT_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_flipadst_8x16_16bpc_avx2);
        self.itxfm_add[tx_8x16][FLIPADST_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_flipadst_8x16_16bpc_avx2);
        self.itxfm_add[tx_8x16][ADST_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_flipadst_8x16_16bpc_avx2);
        self.itxfm_add[tx_8x16][FLIPADST_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_adst_8x16_16bpc_avx2);

        // 16x8 ADST/FLIPADST transforms 16bpc
        self.itxfm_add[tx_16x8][ADST_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_dct_16x8_16bpc_avx2);
        self.itxfm_add[tx_16x8][DCT_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_adst_16x8_16bpc_avx2);
        self.itxfm_add[tx_16x8][ADST_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_adst_16x8_16bpc_avx2);
        self.itxfm_add[tx_16x8][FLIPADST_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_dct_16x8_16bpc_avx2);
        self.itxfm_add[tx_16x8][DCT_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_flipadst_16x8_16bpc_avx2);
        self.itxfm_add[tx_16x8][FLIPADST_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_flipadst_16x8_16bpc_avx2);
        self.itxfm_add[tx_16x8][ADST_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_flipadst_16x8_16bpc_avx2);
        self.itxfm_add[tx_16x8][FLIPADST_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_adst_16x8_16bpc_avx2);

        // 4x16 ADST/FLIPADST transforms 16bpc
        self.itxfm_add[tx_4x16][ADST_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_dct_4x16_16bpc_avx2);
        self.itxfm_add[tx_4x16][DCT_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_adst_4x16_16bpc_avx2);
        self.itxfm_add[tx_4x16][ADST_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_adst_4x16_16bpc_avx2);
        self.itxfm_add[tx_4x16][FLIPADST_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_dct_4x16_16bpc_avx2);
        self.itxfm_add[tx_4x16][DCT_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_flipadst_4x16_16bpc_avx2);
        self.itxfm_add[tx_4x16][FLIPADST_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_flipadst_4x16_16bpc_avx2);
        self.itxfm_add[tx_4x16][ADST_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_flipadst_4x16_16bpc_avx2);
        self.itxfm_add[tx_4x16][FLIPADST_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_adst_4x16_16bpc_avx2);

        // 16x4 ADST/FLIPADST transforms 16bpc
        self.itxfm_add[tx_16x4][ADST_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_dct_16x4_16bpc_avx2);
        self.itxfm_add[tx_16x4][DCT_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_adst_16x4_16bpc_avx2);
        self.itxfm_add[tx_16x4][ADST_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_adst_16x4_16bpc_avx2);
        self.itxfm_add[tx_16x4][FLIPADST_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_dct_16x4_16bpc_avx2);
        self.itxfm_add[tx_16x4][DCT_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_flipadst_16x4_16bpc_avx2);
        self.itxfm_add[tx_16x4][FLIPADST_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_flipadst_16x4_16bpc_avx2);
        self.itxfm_add[tx_16x4][ADST_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_flipadst_16x4_16bpc_avx2);
        self.itxfm_add[tx_16x4][FLIPADST_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_adst_16x4_16bpc_avx2);

        // Hybrid identity transforms 16bpc - 4x8
        self.itxfm_add[tx_4x8][V_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_dct_4x8_16bpc_avx2);
        self.itxfm_add[tx_4x8][H_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_identity_4x8_16bpc_avx2);
        self.itxfm_add[tx_4x8][V_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_adst_4x8_16bpc_avx2);
        self.itxfm_add[tx_4x8][H_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_identity_4x8_16bpc_avx2);
        self.itxfm_add[tx_4x8][V_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_flipadst_4x8_16bpc_avx2);
        self.itxfm_add[tx_4x8][H_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_identity_4x8_16bpc_avx2);

        // Hybrid identity transforms 16bpc - 8x4
        self.itxfm_add[tx_8x4][V_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_dct_8x4_16bpc_avx2);
        self.itxfm_add[tx_8x4][H_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_identity_8x4_16bpc_avx2);
        self.itxfm_add[tx_8x4][V_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_adst_8x4_16bpc_avx2);
        self.itxfm_add[tx_8x4][H_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_identity_8x4_16bpc_avx2);
        self.itxfm_add[tx_8x4][V_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_flipadst_8x4_16bpc_avx2);
        self.itxfm_add[tx_8x4][H_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_identity_8x4_16bpc_avx2);

        // Hybrid identity transforms 16bpc - 8x16
        self.itxfm_add[tx_8x16][V_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_dct_8x16_16bpc_avx2);
        self.itxfm_add[tx_8x16][H_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_identity_8x16_16bpc_avx2);
        self.itxfm_add[tx_8x16][V_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_adst_8x16_16bpc_avx2);
        self.itxfm_add[tx_8x16][H_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_identity_8x16_16bpc_avx2);
        self.itxfm_add[tx_8x16][V_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_flipadst_8x16_16bpc_avx2);
        self.itxfm_add[tx_8x16][H_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_identity_8x16_16bpc_avx2);

        // Hybrid identity transforms 16bpc - 16x8
        self.itxfm_add[tx_16x8][V_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_dct_16x8_16bpc_avx2);
        self.itxfm_add[tx_16x8][H_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_identity_16x8_16bpc_avx2);
        self.itxfm_add[tx_16x8][V_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_adst_16x8_16bpc_avx2);
        self.itxfm_add[tx_16x8][H_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_identity_16x8_16bpc_avx2);
        self.itxfm_add[tx_16x8][V_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_flipadst_16x8_16bpc_avx2);
        self.itxfm_add[tx_16x8][H_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_identity_16x8_16bpc_avx2);

        // Hybrid identity transforms 16bpc - 4x16
        self.itxfm_add[tx_4x16][V_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_dct_4x16_16bpc_avx2);
        self.itxfm_add[tx_4x16][H_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_identity_4x16_16bpc_avx2);
        self.itxfm_add[tx_4x16][V_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_adst_4x16_16bpc_avx2);
        self.itxfm_add[tx_4x16][H_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_identity_4x16_16bpc_avx2);
        self.itxfm_add[tx_4x16][V_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_flipadst_4x16_16bpc_avx2);
        self.itxfm_add[tx_4x16][H_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_identity_4x16_16bpc_avx2);

        // Hybrid identity transforms 16bpc - 16x4
        self.itxfm_add[tx_16x4][V_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_dct_16x4_16bpc_avx2);
        self.itxfm_add[tx_16x4][H_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_identity_16x4_16bpc_avx2);
        self.itxfm_add[tx_16x4][V_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_adst_16x4_16bpc_avx2);
        self.itxfm_add[tx_16x4][H_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_identity_16x4_16bpc_avx2);
        self.itxfm_add[tx_16x4][V_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_flipadst_16x4_16bpc_avx2);
        self.itxfm_add[tx_16x4][H_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_identity_16x4_16bpc_avx2);

        // Hybrid identity transforms 16bpc - 8x8 square
        self.itxfm_add[tx_8x8][V_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_dct_8x8_16bpc_avx2);
        self.itxfm_add[tx_8x8][H_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_identity_8x8_16bpc_avx2);
        self.itxfm_add[tx_8x8][V_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_adst_8x8_16bpc_avx2);
        self.itxfm_add[tx_8x8][H_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_identity_8x8_16bpc_avx2);
        self.itxfm_add[tx_8x8][V_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_flipadst_8x8_16bpc_avx2);
        self.itxfm_add[tx_8x8][H_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_identity_8x8_16bpc_avx2);

        // Hybrid identity transforms 16bpc - 4x4 square
        self.itxfm_add[tx_4x4][V_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_dct_4x4_16bpc_avx2);
        self.itxfm_add[tx_4x4][H_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_identity_4x4_16bpc_avx2);
        self.itxfm_add[tx_4x4][V_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_adst_4x4_16bpc_avx2);
        self.itxfm_add[tx_4x4][H_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_identity_4x4_16bpc_avx2);
        self.itxfm_add[tx_4x4][V_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_flipadst_4x4_16bpc_avx2);
        self.itxfm_add[tx_4x4][H_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_identity_4x4_16bpc_avx2);

        // Hybrid identity transforms 16bpc - 16x16 square
        self.itxfm_add[tx_16x16][V_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_dct_16x16_16bpc_avx2);
        self.itxfm_add[tx_16x16][H_DCT as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_dct_identity_16x16_16bpc_avx2);
        self.itxfm_add[tx_16x16][V_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_adst_16x16_16bpc_avx2);
        self.itxfm_add[tx_16x16][H_ADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_adst_identity_16x16_16bpc_avx2);
        self.itxfm_add[tx_16x16][V_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_identity_flipadst_16x16_16bpc_avx2);
        self.itxfm_add[tx_16x16][H_FLIPADST as usize] =
            itxfm::Fn::new(safe_itx::inv_txfm_add_flipadst_identity_16x16_16bpc_avx2);

        self
    }

    #[cfg(all(not(feature = "asm"), feature = "c-ffi", target_arch = "aarch64"))]
    #[inline(always)]
    const fn init_arm_safe_simd<BD: BitDepth>(mut self, _flags: CpuFlags) -> Self {
        use crate::include::common::bitdepth::BPC;
        use crate::src::safe_simd::itx_arm as safe_itx;

        // Helper macro: assign a single ITX function to the dispatch table
        macro_rules! a {
            ($w:literal, $h:literal, $type_name:ident, $type_enum:ident, $bpc:literal) => {
                paste::paste! {
                    self.itxfm_add[TxfmSize::from_wh($w, $h) as usize][$type_enum as usize] =
                        itxfm::Fn::new(safe_itx::[<inv_txfm_add_ $type_name _ $w x $h _ $bpc bpc_neon>]);
                }
            };
        }

        // itx16: 16 transform types (DCT_DCT + IDTX + 10 hybrids + 4 identity hybrids)
        // Used for sizes 4x4, 8x8, 16x16 where all types are valid
        macro_rules! itx16 {
            ($w:literal, $h:literal, $bpc:literal) => {
                a!($w, $h, dct_dct, DCT_DCT, $bpc);
                a!($w, $h, identity_identity, IDTX, $bpc);
                a!($w, $h, adst_dct, DCT_ADST, $bpc);
                a!($w, $h, dct_adst, ADST_DCT, $bpc);
                a!($w, $h, adst_adst, ADST_ADST, $bpc);
                a!($w, $h, flipadst_dct, DCT_FLIPADST, $bpc);
                a!($w, $h, dct_flipadst, FLIPADST_DCT, $bpc);
                a!($w, $h, flipadst_flipadst, FLIPADST_FLIPADST, $bpc);
                a!($w, $h, adst_flipadst, FLIPADST_ADST, $bpc);
                a!($w, $h, flipadst_adst, ADST_FLIPADST, $bpc);
                a!($w, $h, dct_identity, H_DCT, $bpc);
                a!($w, $h, identity_dct, V_DCT, $bpc);
                a!($w, $h, adst_identity, H_ADST, $bpc);
                a!($w, $h, identity_adst, V_ADST, $bpc);
                a!($w, $h, flipadst_identity, H_FLIPADST, $bpc);
                a!($w, $h, identity_flipadst, V_FLIPADST, $bpc);
            };
        }

        // itx12: 12 transform types (DCT_DCT + IDTX + 10 hybrids, no identity hybrids)
        // Used for rectangular sizes (4x8, 8x4, 4x16, etc.)
        macro_rules! itx12 {
            ($w:literal, $h:literal, $bpc:literal) => {
                a!($w, $h, dct_dct, DCT_DCT, $bpc);
                a!($w, $h, identity_identity, IDTX, $bpc);
                a!($w, $h, adst_dct, DCT_ADST, $bpc);
                a!($w, $h, dct_adst, ADST_DCT, $bpc);
                a!($w, $h, adst_adst, ADST_ADST, $bpc);
                a!($w, $h, flipadst_dct, DCT_FLIPADST, $bpc);
                a!($w, $h, dct_flipadst, FLIPADST_DCT, $bpc);
                a!($w, $h, flipadst_flipadst, FLIPADST_FLIPADST, $bpc);
                a!($w, $h, adst_flipadst, FLIPADST_ADST, $bpc);
                a!($w, $h, flipadst_adst, ADST_FLIPADST, $bpc);
                a!($w, $h, dct_identity, H_DCT, $bpc);
                a!($w, $h, identity_dct, V_DCT, $bpc);
            };
        }

        // itx2: DCT_DCT + IDTX only (large rectangular sizes)
        macro_rules! itx2 {
            ($w:literal, $h:literal, $bpc:literal) => {
                a!($w, $h, dct_dct, DCT_DCT, $bpc);
                a!($w, $h, identity_identity, IDTX, $bpc);
            };
        }

        match BD::BPC {
            BPC::BPC8 => {
                a!(4, 4, wht_wht, WHT_WHT, 8);

                // Square sizes: full 16 transform types
                itx16!(4, 4, 8);
                itx16!(8, 8, 8);
                itx16!(16, 16, 8);

                // Rectangular sizes: all 16 transform types
                itx16!(4, 8, 8);
                itx16!(8, 4, 8);
                itx16!(4, 16, 8);
                itx16!(16, 4, 8);
                itx16!(8, 16, 8);
                itx16!(16, 8, 8);

                // Large rectangular: DCT_DCT + IDTX
                itx2!(8, 32, 8);
                itx2!(32, 8, 8);
                itx2!(16, 32, 8);
                itx2!(32, 16, 8);
                itx2!(32, 32, 8);

                // DCT-only large sizes
                a!(32, 64, dct_dct, DCT_DCT, 8);
                a!(64, 32, dct_dct, DCT_DCT, 8);
                a!(64, 64, dct_dct, DCT_DCT, 8);
                a!(16, 64, dct_dct, DCT_DCT, 8);
                a!(64, 16, dct_dct, DCT_DCT, 8);
            }
            BPC::BPC16 => {
                a!(4, 4, wht_wht, WHT_WHT, 16);

                // Square sizes: full 16 transform types
                itx16!(4, 4, 16);
                itx16!(8, 8, 16);
                itx16!(16, 16, 16);

                // Rectangular sizes: all 16 transform types
                itx16!(4, 8, 16);
                itx16!(8, 4, 16);
                itx16!(4, 16, 16);
                itx16!(16, 4, 16);
                itx16!(8, 16, 16);
                itx16!(16, 8, 16);

                // Large rectangular: DCT_DCT + IDTX
                itx2!(8, 32, 16);
                itx2!(32, 8, 16);
                itx2!(16, 32, 16);
                itx2!(32, 16, 16);
                itx2!(32, 32, 16);

                // DCT-only large sizes
                a!(32, 64, dct_dct, DCT_DCT, 16);
                a!(64, 32, dct_dct, DCT_DCT, 16);
                a!(64, 64, dct_dct, DCT_DCT, 16);
                a!(16, 64, dct_dct, DCT_DCT, 16);
                a!(64, 16, dct_dct, DCT_DCT, 16);
            }
        }

        self
    }

    #[inline(always)]
    const fn init<BD: BitDepth>(self, flags: CpuFlags, bpc: u8) -> Self {
        #[cfg(feature = "asm")]
        {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                return self.init_x86::<BD>(flags, bpc);
            }
            #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
            {
                return self.init_arm::<BD>(flags, bpc);
            }
        }

        #[cfg(all(not(feature = "asm"), feature = "c-ffi", target_arch = "x86_64"))]
        {
            let _ = bpc;
            return self.init_x86_safe_simd::<BD>(flags);
        }

        #[cfg(all(not(feature = "asm"), feature = "c-ffi", target_arch = "aarch64"))]
        {
            let _ = bpc;
            return self.init_arm_safe_simd::<BD>(flags);
        }

        #[allow(unreachable_code)] // Reachable on some #[cfg]s.
        {
            let _ = flags;
            let _ = bpc;
            self
        }
    }

    pub const fn new<BD: BitDepth>(flags: CpuFlags, bpc: u8) -> Self {
        Self::default::<BD>().init::<BD>(flags, bpc)
    }
}
