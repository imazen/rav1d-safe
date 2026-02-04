//! Safe SIMD implementations using archmage tokens
//!
//! This module provides safe Rust implementations of the SIMD functions
//! that are normally implemented in hand-written x86/ARM assembly.
//!
//! Enable with `--features safe-simd` to use these instead of asm.

pub mod mc;

#[cfg(target_arch = "aarch64")]
pub mod mc_arm;

pub mod cdef;

// Re-export x86 implementations
pub use mc::*;

// Re-export ARM implementations
#[cfg(target_arch = "aarch64")]
pub use mc_arm::*;
