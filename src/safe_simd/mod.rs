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

/// Serializes tests against `archmage::testing`'s PROCESS-GLOBAL token switch.
///
/// The `testable_dispatch` dev-feature lets a test disable compile-time SIMD
/// tokens so it can exercise the fallback paths — but the switch is global, so
/// while `for_each_token_permutation` is running, ANY concurrently-running test
/// that calls `Arm64::summon()` can get `None` back. `cargo test --lib` runs
/// tests in parallel, so this is not hypothetical: it made
/// `loopfilter::neon_parity` fail with "kernel refused" in roughly a quarter of
/// full-suite runs, at random cells, while passing 20/20 when filtered to
/// itself.
///
/// Any test that either PERMUTES tokens or DEPENDS on a token being available
/// must hold this lock. Poisoning is ignored: a panicking test has already
/// failed, and the lock exists to order the survivors, not to protect data.
#[cfg(test)]
pub(crate) fn token_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Backing store for a test plane that is about to be handed to
/// [`Rav1dPictureDataComponent::wrap_buf`].
///
/// `wrap_buf` requires its buffer to start on a
/// [`RAV1D_PICTURE_ALIGNMENT`]-byte boundary — that alignment is a type
/// invariant of the c-ffi `Rav1dPictureDataComponentInner`, and
/// `ExternalAsMutPtr::as_mut_ptr` `assume`s it. A plain `Vec<BD::Pixel>` has
/// alignment 1 (8bpc) or 2 (16bpc) and does not satisfy it. Every production
/// caller does, by construction: the scratch types in `src/internal.rs` are
/// `#[repr(C, align(64))]`.
///
/// This is invisible in the default build, where `wrap_buf` COPIES into a
/// 64-byte-aligned `PicBuf` and the caller's alignment is irrelevant — which
/// is exactly how two aarch64 parity harnesses came to violate it and only
/// fail under `--features c-ffi`.
///
/// [`Rav1dPictureDataComponent::wrap_buf`]: crate::include::dav1d::picture::Rav1dPictureDataComponent::wrap_buf
/// [`RAV1D_PICTURE_ALIGNMENT`]: crate::include::dav1d::picture::DAV1D_PICTURE_ALIGNMENT
#[cfg(test)]
pub(crate) fn aligned_plane<T: Copy + Default>(
    src: &[T],
) -> rav1d_disjoint_mut::align::AlignedVec64<T> {
    let mut v = rav1d_disjoint_mut::align::AlignedVec64::<T>::new();
    v.resize(src.len(), T::default());
    v.copy_from_slice(src);
    v
}

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

/// 10/12-bit inverse transforms in 32-bit NEON lanes (the 16-bit-lane
/// `itx_arm_neon_*` ports are 8bpc-only by construction).
#[cfg(target_arch = "aarch64")]
pub mod itx_arm_hbd;

/// Differential parity for the aarch64 itx kernels vs the scalar reference.
/// Test-only; the module itself is `#![cfg(all(test, ...))]`.
#[cfg(target_arch = "aarch64")]
mod itx_arm_parity;

/// Differential parity for the aarch64 `prep` kernels vs the scalar reference.
/// Test-only; the module itself is `#![cfg(all(test, ...))]`.
#[cfg(target_arch = "aarch64")]
mod mc_arm_prep_parity;

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
