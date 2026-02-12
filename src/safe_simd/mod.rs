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

pub mod cdef;

#[cfg(target_arch = "x86_64")]
pub mod itx;

#[cfg(target_arch = "x86_64")]
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
