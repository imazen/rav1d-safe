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
// Clippy lint policy: suppress pervasive C-port patterns at crate level,
// enable everything else. Each lint has a reason and warning count.
//
// Pervasive C-port patterns (too many to fix individually):
#![allow(clippy::precedence)] // 652: C-style arithmetic
#![allow(clippy::too_many_arguments)] // 282: C function signatures
#![allow(clippy::unnecessary_cast)] // 189: generic/bitdepth code
#![allow(clippy::identity_op)] // 156: readability in transforms
#![allow(clippy::needless_range_loop)] // 143: C-port loop idiom
#![allow(clippy::explicit_auto_deref)] // 85: deref style
#![allow(clippy::erasing_op)] // 53: generic transform constants
#![allow(clippy::needless_return)] // 24: C-port return style
#![allow(clippy::nonminimal_bool)] // 23: C-port booleans
#![allow(clippy::needless_borrow)] // 19: borrow style
#![allow(clippy::doc_overindented_list_items)] // 16: doc formatting
#![allow(clippy::zero_prefixed_literal)] // 12: decimal C constants
#![allow(clippy::collapsible_if)] // 11: nested ifs from C
#![allow(clippy::needless_late_init)] // 10: C-style init
//
// Structural C-port patterns (changing would obscure correspondence):
#![allow(clippy::upper_case_acronyms)] // 9: AV1 spec names
#![allow(clippy::type_complexity)] // 4: internal C-port types
#![allow(clippy::misrefactored_assign_op)] // 4: boundary clamping
#![allow(clippy::neg_multiply)] // 4: C-style negation
//
// Intentional patterns (deny-by-default, false positives here):
#![allow(clippy::eq_op)] // 2: intentional 2-2
#![allow(clippy::overly_complex_bool_expr)] // 2: debug gates
#![allow(clippy::let_underscore_lock)] // 2: lock drop via take()
//
// Informational lints not worth acting on:
#![allow(clippy::module_inception)] // 1: dav1d::dav1d
#![allow(clippy::large_enum_variant)] // 1: internal enum
//
// Newer clippy lints (1.87+) firing on C-port patterns:
#![allow(clippy::duplicated_attributes)] // new in clippy 1.87+: repeated cfg_attr
#![allow(clippy::manual_is_multiple_of)] // new in clippy 1.87+: x % n == 0 patterns
#![allow(clippy::let_and_return)] // new in clippy 1.87+: C-port let-then-return
#![allow(clippy::unnecessary_map_on_constructor)] // new in clippy 1.87+: Option/Result::map on constructor
#![allow(clippy::clone_on_copy)] // new in clippy 1.87+: explicit .clone() on Copy types
#![allow(clippy::option_map_unit_fn)] // new in clippy 1.87+: .map(|x| side_effect)
#![allow(clippy::unnecessary_lazy_evaluations)] // new in clippy 1.87+: .unwrap_or_else(|| val)
#![cfg_attr(
    any(feature = "asm", feature = "c-ffi"),
    deny(clippy::undocumented_unsafe_blocks)
)]
#![cfg_attr(
    any(feature = "asm", feature = "c-ffi"),
    deny(clippy::missing_safety_doc)
)]

#[cfg(not(any(feature = "bitdepth_8", feature = "bitdepth_16")))]
compile_error!(
    "No bitdepths enabled. Enable one or more of the following features: `bitdepth_8`, `bitdepth_16`"
);

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

    // Per-kernel SIMD ablation switch (measurement infrastructure). Compiles to
    // a constant `false` guard unless the `__ablate` feature is on; `pub` so the
    // ablation harness in `examples/` can drive it.
    pub mod ablate;

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
    pub(crate) mod lf_band;
    mod lf_mask;
    // `pub` only to let `examples/probe_tracker` reach `loopfilter::lf_hist`.
    // The default build keeps it private.
    #[cfg(feature = "__probe_lf_hist")]
    pub mod loopfilter;
    #[cfg(not(feature = "__probe_lf_hist"))]
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
    pub(crate) mod owned_recon;
    pub(crate) mod pic_or_buf;
    /// THROWAWAY P1 measurement probe. Never merge.
    #[cfg(feature = "probe-tasktime")]
    pub mod probe_tasktime;
    mod qm;
    pub(crate) mod relaxed_atomic;
    mod scan;
    pub(crate) mod strided;
    // Public ONLY under the private test feature, for the induced-worker-panic
    // hook (tests/worker_panic_recovery.rs); crate-private otherwise.
    #[cfg(feature = "__test_induce_worker_panic")]
    pub mod thread_task;
    #[cfg(not(feature = "__test_induce_worker_panic"))]
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
    ColorInfo, ColorPrimaries, ColorRange, ContentLightLevel, CpuLevel, DecodeFrameType, Decoder,
    Error, Frame, InloopFilters, MasteringDisplay, MatrixCoefficients, PixelLayout, PlaneView8,
    PlaneView16, Planes, Planes8, Planes16, Result, Settings, TransferCharacteristics,
    enabled_features,
};
