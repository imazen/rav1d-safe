//! Safe SIMD implementations using Rust intrinsics
//!
//! This module provides safe Rust implementations of the SIMD functions
//! that are normally implemented in hand-written x86/ARM assembly.
//!
//! Used automatically when built without `--features asm`.

pub mod mc;

#[cfg(target_arch = "aarch64")]
pub mod mc_arm;

pub mod cdef;

#[cfg(target_arch = "x86_64")]
pub mod itx;

#[cfg(target_arch = "x86_64")]
pub mod loopfilter;

#[cfg(target_arch = "x86_64")]
pub mod looprestoration;

// Re-export x86 implementations
pub use mc::*;

// Re-export ARM implementations
#[cfg(target_arch = "aarch64")]
pub use mc_arm::*;
