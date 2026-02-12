#![allow(non_upper_case_globals)]
#![cfg_attr(target_arch = "arm", feature(stdarch_arm_feature_detection))]
#![cfg_attr(
    any(target_arch = "riscv32", target_arch = "riscv64"),
    feature(stdarch_riscv_feature_detection)
)]
// Crate-wide forbid(unsafe_code) when neither `asm` nor `c-ffi` is enabled.
// All unsafe must live in separate crates (rav1d-disjoint-mut, rav1d-align, etc.)
// or be gated behind cfg(feature = "asm") / cfg(feature = "c-ffi").
// forbid cannot be overridden by #[allow] — any unsafe in the default build is a hard error.
#![cfg_attr(
    not(any(feature = "asm", feature = "c-ffi", feature = "unchecked")),
    forbid(unsafe_code)
)]
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
    // - Modules with zero unsafe use forbid(unsafe_code) internally
    // - Modules with isolated unsafe items use item-level #[allow(unsafe_code)]
    // - Modules that need unsafe only for c-ffi use cfg_attr(feature, allow)
    // - safe_simd sub-modules set their own forbid/deny (no parent blanket allow)

    // Core primitives
    pub(crate) mod align;
    #[cfg(feature = "c-ffi")]
    pub(crate) mod assume;
    #[cfg_attr(feature = "c-ffi", allow(unsafe_code))]
    pub(crate) mod c_arc;
    #[cfg_attr(feature = "c-ffi", allow(unsafe_code))]
    pub(crate) mod c_box;
    pub(crate) mod cpu;
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

    // Safe SIMD implementations (internal, not part of the public API)
    #[cfg(not(feature = "asm"))]
    pub(crate) mod safe_simd;

    // Rust core API (rav1d_open, rav1d_send_data, etc.)
    #[cfg_attr(not(feature = "c-ffi"), deny(unsafe_code))]
    #[cfg_attr(feature = "c-ffi", allow(unsafe_code))]
    pub(crate) mod lib;

    // C FFI wrappers (dav1d_* extern "C" functions)
    #[cfg(feature = "c-ffi")]
    #[allow(unsafe_code)]
    pub mod dav1d_api;

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

// Re-export the managed API at the crate root for convenience.
// Users can write `rav1d_safe::Decoder` instead of `rav1d_safe::src::managed::Decoder`.
pub use src::managed::{
    enabled_features, ColorInfo, ColorPrimaries, ColorRange, ContentLightLevel, CpuLevel,
    DecodeFrameType, Decoder, Error, Frame, InloopFilters, MasteringDisplay, MatrixCoefficients,
    PixelLayout, PlaneView16, PlaneView8, Planes, Planes16, Planes8, Result, Settings,
    TransferCharacteristics,
};
