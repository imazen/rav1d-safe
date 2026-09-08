//! Verify the ARM assembly header against the actual target Rust ABI.
//!
//! Read the header itself so changing either side cannot leave stale duplicate
//! Rust constants. These assertions run during every ARM assembly build.
use crate::include::dav1d::headers::Dav1dFilmGrainData;

const HEADER: &[u8] = include_bytes!("arm/asm-offsets.h");

const fn starts_at(at: usize, text: &[u8]) -> bool {
    if at + text.len() > HEADER.len() {
        return false;
    }
    let mut i = 0;
    while i < text.len() {
        if HEADER[at + i] != text[i] {
            return false;
        }
        i += 1;
    }
    true
}

const fn offset(name: &str) -> usize {
    let mut at = 0;
    while at < HEADER.len() {
        if (at == 0 || HEADER[at - 1] == b'\n') && starts_at(at, b"#define ") {
            let start = at + 8;
            let end = start + name.len();
            if starts_at(start, name.as_bytes()) && end < HEADER.len() && HEADER[end] == b' ' {
                let mut i = end;
                while i < HEADER.len() && HEADER[i] == b' ' {
                    i += 1;
                }
                assert!(i < HEADER.len() && HEADER[i] >= b'0' && HEADER[i] <= b'9');
                let mut value = 0;
                while i < HEADER.len() && HEADER[i] >= b'0' && HEADER[i] <= b'9' {
                    value = value * 10 + (HEADER[i] - b'0') as usize;
                    i += 1;
                }
                assert!(i == HEADER.len() || HEADER[i] == b'\n' || HEADER[i] == b'\r');
                return value;
            }
        }
        at += 1;
    }
    panic!("missing numeric ARM assembly offset");
}

macro_rules! check_offsets {
    ($ty:ty; $($name:ident => $field:ident),+ $(,)?) => {
        $(const _: () = assert!(
            core::mem::offset_of!($ty, $field) == offset(stringify!($name)),
            concat!(stringify!($name), " does not match Rust layout"),
        );)+
    };
}

check_offsets! { Dav1dFilmGrainData;
    FGD_SEED => seed,
    FGD_AR_COEFF_LAG => ar_coeff_lag,
    FGD_AR_COEFFS_Y => ar_coeffs_y,
    FGD_AR_COEFFS_UV => ar_coeffs_uv,
    FGD_AR_COEFF_SHIFT => ar_coeff_shift,
    FGD_GRAIN_SCALE_SHIFT => grain_scale_shift,
    FGD_SCALING_SHIFT => scaling_shift,
    FGD_UV_MULT => uv_mult,
    FGD_UV_LUMA_MULT => uv_luma_mult,
    FGD_UV_OFFSET => uv_offset,
    FGD_CLIP_TO_RESTRICTED_RANGE => clip_to_restricted_range,
}

#[cfg(target_arch = "aarch64")]
const _: () = {
    use crate::src::refmvs::AsmRefMvsFrame;
    check_offsets! { AsmRefMvsFrame<'static>;
        RMVSF_IW8 => iw8,
        RMVSF_IH8 => ih8,
        RMVSF_MFMV_REF => mfmv_ref,
        RMVSF_MFMV_REF2CUR => mfmv_ref2cur,
        RMVSF_MFMV_REF2REF => mfmv_ref2ref,
        RMVSF_N_MFMVS => n_mfmvs,
        RMVSF_RP_REF => rp_ref,
        RMVSF_RP_PROJ => rp_proj,
        RMVSF_RP_STRIDE => rp_stride,
        RMVSF_N_TILE_THREADS => n_tile_threads,
    }
};
