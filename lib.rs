#![allow(non_upper_case_globals)]
#![cfg_attr(target_arch = "arm", feature(stdarch_arm_feature_detection))]
#![cfg_attr(
    any(target_arch = "riscv32", target_arch = "riscv64"),
    feature(stdarch_riscv_feature_detection)
)]
// When neither `asm` nor `c-ffi` is enabled, deny unsafe code crate-wide.
// Remaining #[allow(unsafe_code)] items (16 sound abstractions):
//   safe_simd module: partial_simd.rs safety boundary (SIMD load/store wrappers)
//   align.rs(4), assume.rs(1), c_arc.rs(3), c_box.rs(1), disjoint_mut.rs(1),
//   internal.rs(4), msac.rs(2): Send/Sync impls, Pin, AlignedVec, unreachable_unchecked
// 16 safe_simd sub-modules use forbid(unsafe_code) when asm off — compiler-enforced.
#![cfg_attr(not(any(feature = "asm", feature = "c-ffi")), deny(unsafe_code))]
#![cfg_attr(any(feature = "asm", feature = "c-ffi"), deny(unsafe_op_in_unsafe_fn))]
#![allow(clippy::all)]
#![cfg_attr(
    any(feature = "asm", feature = "c-ffi"),
    deny(clippy::undocumented_unsafe_blocks)
)]
#![cfg_attr(
    any(feature = "asm", feature = "c-ffi"),
    deny(clippy::missing_safety_doc)
)]

#[cfg(not(any(feature = "bitdepth_8", feature = "bitdepth_16")))]
compile_error!("No bitdepths enabled. Enable one or more of the following features: `bitdepth_8`, `bitdepth_16`");

pub mod include {
    pub mod common {
        pub(crate) mod attributes;
        pub(crate) mod bitdepth;
        pub(crate) mod dump;
        pub(crate) mod intops;
        pub(crate) mod validate;
    } // mod common
    #[cfg_attr(feature = "c-ffi", allow(unsafe_code))]
    pub mod dav1d {
        pub mod common;
        pub mod data;
        pub mod dav1d;
        pub mod headers;
        pub mod picture;
    } // mod dav1d
} // mod include
pub mod src {
    // === Module Safety Annotations ===
    // Module-level #[allow(unsafe_code)] is used only when the entire module
    // needs unsafe (SIMD intrinsics, pointer operations).
    // For modules with isolated unsafe, item-level #[allow(unsafe_code)] is used
    // on specific functions/impls instead, keeping the rest of the module deny'd.
    // Conditional annotations (#[cfg_attr(feature = "c-ffi", allow(unsafe_code))])
    // are used when unsafe is only needed for C FFI support.

    // Core primitives
    pub mod align;
    pub(crate) mod assume;
    #[cfg_attr(feature = "c-ffi", allow(unsafe_code))]
    pub(crate) mod c_arc;
    #[cfg_attr(feature = "c-ffi", allow(unsafe_code))]
    pub(crate) mod c_box;
    pub mod cpu;
    pub(crate) mod disjoint_mut;
    mod ffi_safe;
    mod in_range;
    pub(super) mod internal;
    mod intra_edge;
    #[cfg_attr(not(feature = "c-ffi"), deny(unsafe_code))]
    #[cfg_attr(feature = "c-ffi", allow(unsafe_code))]
    pub(crate) mod log;
    pub(crate) mod pixels;
    #[cfg(any(feature = "asm", feature = "c-ffi"))]
    #[allow(unsafe_code)]
    pub mod send_sync_non_null;
    mod tables;

    // Data/picture management
    mod data;
    #[cfg_attr(not(feature = "c-ffi"), deny(unsafe_code))]
    #[cfg_attr(feature = "c-ffi", allow(unsafe_code))]
    mod picture;

    // DSP dispatch modules (contain _erased functions and fn ptr dispatch)
    mod cdef;
    mod filmgrain;
    mod ipred;
    mod itx;
    mod lf_mask;
    mod loopfilter;
    mod looprestoration;
    mod mc;
    mod pal;
    mod recon;
    #[cfg_attr(feature = "asm", allow(unsafe_code))]
    mod refmvs;

    // Entropy coding (inline SIMD, safe on both x86_64 and aarch64 when asm off)
    #[cfg_attr(feature = "asm", allow(unsafe_code))]
    mod msac;

    // Safe SIMD implementations
    #[cfg(not(feature = "asm"))]
    #[allow(unsafe_code)]
    pub mod safe_simd;

    // C API entry point
    #[cfg_attr(not(feature = "c-ffi"), deny(unsafe_code))]
    #[cfg_attr(feature = "c-ffi", allow(unsafe_code))]
    pub mod lib;

    // === Modules WITHOUT unsafe_code (enforced by deny) ===
    mod cdef_apply;
    mod cdf;
    mod const_fn;
    mod ctx;
    mod cursor;
    mod decode;
    mod dequant_tables;
    pub(crate) mod enum_map;
    mod env;
    pub(crate) mod error;
    mod extensions;
    mod fg_apply;
    mod getbits;
    mod ipred_prepare;
    mod iter;
    mod itx_1d;
    pub(crate) mod levels;
    mod lf_apply;
    mod lr_apply;
    pub(crate) mod mem;
    mod obu;
    pub(crate) mod pic_or_buf;
    mod qm;
    pub(crate) mod relaxed_atomic;
    mod scan;
    pub(crate) mod strided;
    mod thread_task;
    mod warpmv;
    mod wedge;
    pub(crate) mod with_offset;
    pub(crate) mod wrap_fn_ptr;

    #[cfg(test)]
    mod decode_test;

    // === Managed Safe API ===
    /// 100% safe Rust API for AV1 decoding
    ///
    /// This module provides a fully safe, zero-copy API wrapping rav1d's internal decoder.
    pub mod managed;
} // mod src

pub use src::error::Dav1dResult;

// Re-export the managed API at the crate root for convenience.
// Users can write `rav1d_safe::Decoder` instead of `rav1d_safe::src::managed::Decoder`.
pub use src::managed::{
    enabled_features, ColorInfo, ColorPrimaries, ColorRange, ContentLightLevel, DecodeFrameType,
    Decoder, Error, Frame, InloopFilters, MasteringDisplay, MatrixCoefficients, PixelLayout,
    PlaneView16, PlaneView8, Planes, Planes16, Planes8, Settings, TransferCharacteristics,
};
