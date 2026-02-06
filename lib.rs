#![allow(non_upper_case_globals)]
#![cfg_attr(target_arch = "arm", feature(stdarch_arm_feature_detection))]
#![cfg_attr(
    any(target_arch = "riscv32", target_arch = "riscv64"),
    feature(stdarch_riscv_feature_detection)
)]
// When neither `asm` nor `c-ffi` is enabled, deny unsafe code crate-wide.
// Modules that encapsulate unsafe behind safe APIs get #[allow(unsafe_code)].
// As modules are made fully safe, their #[allow] annotations should be removed.
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
    #[allow(unsafe_code)]
    pub mod dav1d {
        pub mod common;
        pub mod data;
        pub mod dav1d;
        pub mod headers;
        pub mod picture;
    } // mod dav1d
} // mod include
pub mod src {
    // === Modules with #[allow(unsafe_code)] ===
    // These modules encapsulate unsafe behind safe APIs or contain
    // fundamental primitives that require unsafe. Each should be
    // audited and the allow removed as the code is made safe.

    // Core primitives
    #[allow(unsafe_code)]
    pub mod align;
    #[allow(unsafe_code)]
    pub(crate) mod assume;
    #[allow(unsafe_code)]
    pub(crate) mod c_arc;
    #[allow(unsafe_code)]
    pub(crate) mod c_box;
    pub mod cpu;
    #[allow(unsafe_code)]
    pub(crate) mod disjoint_mut;
    #[allow(unsafe_code)]
    mod ffi_safe;
    pub(crate) mod pixels;
    #[allow(unsafe_code)]
    pub mod send_sync_non_null;
    #[allow(unsafe_code)]
    pub(super) mod internal;
    mod in_range;
    mod intra_edge;
    #[allow(unsafe_code)]
    pub(crate) mod log;
    mod tables;

    // Data/picture management
    mod data;
    #[allow(unsafe_code)]
    mod picture;

    // DSP dispatch modules (contain _erased functions and fn ptr dispatch)
    #[allow(unsafe_code)]
    mod cdef;
    #[allow(unsafe_code)]
    mod filmgrain;
    #[allow(unsafe_code)]
    mod ipred;
    #[allow(unsafe_code)]
    mod itx;
    #[allow(unsafe_code)]
    mod loopfilter;
    #[allow(unsafe_code)]
    mod looprestoration;
    #[allow(unsafe_code)]
    mod mc;
    mod pal;
    #[allow(unsafe_code)]
    mod refmvs;
    #[allow(unsafe_code)]
    mod lf_mask;
    mod recon;

    // Entropy coding (inline SIMD)
    #[allow(unsafe_code)]
    mod msac;

    // Safe SIMD implementations
    #[cfg(not(feature = "asm"))]
    #[allow(unsafe_code)]
    pub mod safe_simd;

    // C API entry point
    #[allow(unsafe_code)]
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
    mod fg_apply;
    mod getbits;
    pub(crate) mod pic_or_buf;
    pub(crate) mod relaxed_atomic;
    pub(crate) mod strided;
    pub(crate) mod with_offset;
    pub(crate) mod wrap_fn_ptr;
    mod extensions;
    mod ipred_prepare;
    mod iter;
    mod itx_1d;
    pub(crate) mod levels;
    mod lf_apply;
    mod lr_apply;
    mod mem;
    mod obu;
    mod qm;
    mod scan;
    mod thread_task;
    mod warpmv;
    mod wedge;
} // mod src

pub use src::error::Dav1dResult;
