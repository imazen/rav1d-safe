//! Re-exports from `rav1d_disjoint_mut::align`.
//!
//! All alignment types (`Align*`, `AlignedVec`, `ArrayDefault`, etc.) live in the
//! `rav1d-disjoint-mut` crate so their unsafe impls don't require
//! `#[allow(unsafe_code)]` in the main crate.

pub use rav1d_disjoint_mut::align::*;
