//! Safe SIMD implementations using Rust intrinsics
//!
//! This module provides safe Rust implementations of the SIMD functions
//! that are normally implemented in hand-written x86/ARM assembly.
//!
//! Used automatically when built without `--features asm`.

#![deny(unsafe_code)]
// The load/store macros in pixel_access expand to `unsafe {}` blocks when
// `unchecked` feature is enabled (which is implied by c-ffi). These are
// bounds-checked by construction — the macro verifies slice length before
// performing the raw pointer operation.
#![allow(clippy::undocumented_unsafe_blocks)]

pub mod partial_simd;
pub mod pixel_access;

pub mod mc;

#[cfg(target_arch = "aarch64")]
pub mod mc_arm;

#[cfg(target_arch = "aarch64")]
pub mod ipred_arm;

#[cfg(target_arch = "aarch64")]
pub mod cdef_arm;

#[cfg(target_arch = "aarch64")]
pub mod loopfilter_arm;

#[cfg(target_arch = "aarch64")]
pub mod looprestoration_arm;

#[cfg(target_arch = "aarch64")]
pub mod itx_arm;

// The aarch64 NEON itx kernels (issue #400) are bit-exact and dispatched for
// 8bpc; the 16bpc variants exist but aren't NEON-dispatched yet, so allow dead
// code in these modules.
macro_rules! itx_neon_mod {
    ($name:ident) => {
        #[cfg(target_arch = "aarch64")]
        #[allow(dead_code)]
        pub mod $name;
    };
}
itx_neon_mod!(itx_arm_neon_wht);
itx_neon_mod!(itx_arm_neon_common);
itx_neon_mod!(itx_arm_neon_4x4);
itx_neon_mod!(itx_arm_neon_8x8);
itx_neon_mod!(itx_arm_neon_rect);
itx_neon_mod!(itx_arm_neon_16x16);
itx_neon_mod!(itx_arm_neon_32);
itx_neon_mod!(itx_arm_neon_rect_large);
itx_neon_mod!(itx_arm_neon_large_rect);
itx_neon_mod!(itx_arm_neon_64);

pub mod cdef;

#[cfg(target_arch = "wasm32")]
pub mod cdef_wasm;

#[cfg(target_arch = "wasm32")]
pub mod mc_wasm;

#[cfg(target_arch = "wasm32")]
pub mod itx_wasm;

#[cfg(target_arch = "x86_64")]
pub mod itx;

#[cfg(any(target_arch = "x86_64", target_arch = "wasm32"))]
pub mod loopfilter;

#[cfg(target_arch = "x86_64")]
pub mod looprestoration;

#[cfg(target_arch = "x86_64")]
pub mod ipred;

#[cfg(target_arch = "x86_64")]
pub mod filmgrain;

#[cfg(target_arch = "x86_64")]
pub mod pal;

#[cfg(target_arch = "x86_64")]
pub mod refmvs;

#[cfg(target_arch = "aarch64")]
pub mod filmgrain_arm;

#[cfg(target_arch = "aarch64")]
pub mod refmvs_arm;
