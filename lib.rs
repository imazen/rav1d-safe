#![allow(non_upper_case_globals)]
#![cfg_attr(target_arch = "arm", feature(stdarch_arm_feature_detection))]
#![cfg_attr(
    any(target_arch = "riscv32", target_arch = "riscv64"),
    feature(stdarch_riscv_feature_detection)
)]
// When neither `asm` nor `c-ffi` is enabled, deny unsafe code crate-wide.
// Essential modules that encapsulate unsafe behind safe APIs get #[allow(unsafe_code)].
#![cfg_attr(not(any(feature = "asm", feature = "c-ffi")), deny(unsafe_code))]
#![cfg_attr(any(feature = "asm", feature = "c-ffi"), deny(unsafe_op_in_unsafe_fn))]
#![allow(clippy::all)]
#![cfg_attr(any(feature = "asm", feature = "c-ffi"), deny(clippy::undocumented_unsafe_blocks))]
#![cfg_attr(any(feature = "asm", feature = "c-ffi"), deny(clippy::missing_safety_doc))]

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
    pub mod dav1d {
        #[cfg_attr(not(any(feature = "asm", feature = "c-ffi")), allow(unsafe_code))]
        pub mod common;
        #[cfg_attr(not(any(feature = "asm", feature = "c-ffi")), allow(unsafe_code))]
        pub mod data;
        pub mod dav1d;
        #[cfg_attr(not(any(feature = "asm", feature = "c-ffi")), allow(unsafe_code))]
        pub mod headers;
        #[cfg_attr(not(any(feature = "asm", feature = "c-ffi")), allow(unsafe_code))]
        pub mod picture;
    } // mod dav1d
} // mod include
#[cfg_attr(not(any(feature = "asm", feature = "c-ffi")), allow(unsafe_code))]
pub mod src {
    pub mod align;
    pub(crate) mod assume;
    pub(crate) mod c_arc;
    pub(crate) mod c_box;
    mod cdef;
    mod cdef_apply;
    mod cdf;
    mod const_fn;
    pub mod cpu;
    mod ctx;
    mod cursor;
    mod data;
    mod decode;
    mod dequant_tables;
    pub(crate) mod disjoint_mut;
    pub(crate) mod enum_map;
    mod env;
    pub(crate) mod error;
    mod ffi_safe;
    mod fg_apply;
    mod filmgrain;
    mod getbits;
    pub(crate) mod pic_or_buf;
    pub(crate) mod pixels;
    pub(crate) mod relaxed_atomic;
    pub mod send_sync_non_null;
    pub(crate) mod strided;
    pub(crate) mod with_offset;
    pub(crate) mod wrap_fn_ptr;
    // TODO(kkysen) Temporarily `pub(crate)` due to a `pub use` until TAIT.
    mod extensions;
    mod in_range;
    pub(super) mod internal;
    mod intra_edge;
    mod ipred;
    mod ipred_prepare;
    mod iter;
    mod itx;
    mod itx_1d;
    pub(crate) mod levels;
    mod lf_apply;
    mod lf_mask;
    pub mod lib;
    pub(crate) mod log;
    mod loopfilter;
    mod looprestoration;
    mod lr_apply;
    mod mc;
    mod mem;

    // Safe SIMD implementations using archmage
    mod msac;
    mod obu;
    mod pal;
    mod picture;
    mod qm;
    mod recon;
    mod refmvs;
    #[cfg(not(feature = "asm"))]
    pub mod safe_simd;
    mod scan;
    mod tables;
    mod thread_task;
    mod warpmv;
    mod wedge;
} // mod src

pub use src::error::Dav1dResult;
