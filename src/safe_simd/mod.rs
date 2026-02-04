//! Safe SIMD implementations using archmage
//!
//! This module provides safe Rust implementations of the SIMD functions
//! that are normally implemented in hand-written x86/ARM assembly.
//!
//! Enable with `--features safe-simd` to use these instead of asm.

pub mod mc;
pub mod cdef;

// Re-export for convenience
pub use mc::*;
