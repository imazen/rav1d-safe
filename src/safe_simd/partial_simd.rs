//! Safe wrappers for partial (64-bit) SIMD load/store operations.
//!
//! These fill a gap in safe_unaligned_simd which lacks Is64BitsUnaligned trait.
//! When/if safe_unaligned_simd adds these, this module can re-export from there.
//!
//! The wrappers are safe functions with #[target_feature], callable from #[arcane]
//! contexts without needing unsafe blocks.
//!
//! # Safety boundary
//! This module provides the safe abstraction boundary — it contains targeted unsafe
//! blocks to wrap raw SIMD intrinsics, exposed as safe functions with #[target_feature].
//!
//! # Future compatibility
//! ```ignore
//! // When safe_unaligned_simd adds 64-bit support, change this module to:
//! #[cfg(target_arch = "x86_64")]
//! pub use safe_unaligned_simd::x86_64::{mm_loadl_epi64, mm_storel_epi64, ...};
//! ```
//!
//! # Example
//! ```ignore
//! use archmage::{arcane, Desktop64};
//! use crate::safe_simd::partial_simd;
//!
//! #[arcane]
//! fn process(_token: Desktop64, src: &[u8; 8], dst: &mut [u8; 8]) {
//!     let v = partial_simd::mm_loadl_epi64(src);   // SAFE - no unsafe block!
//!     let v = _mm_add_epi8(v, v);                   // Also safe in #[arcane]
//!     partial_simd::mm_storel_epi64(dst, v);       // SAFE - no unsafe block!
//! }
//! ```

// ============================================================================
// x86_64 SSE2 64-bit operations
// ============================================================================

#[cfg(target_arch = "x86_64")]
pub use x86_64::*;

#[cfg(target_arch = "x86_64")]
mod x86_64 {
    use core::arch::x86_64::*;

    /// Trait for types that can be used as 64-bit unaligned memory operands.
    pub trait Is64BitsUnaligned: private::Sealed {
        const SIZE: usize = 8;
    }

    mod private {
        pub trait Sealed {}
        impl Sealed for [u8; 8] {}
        impl Sealed for [i8; 8] {}
        impl Sealed for [u16; 4] {}
        impl Sealed for [i16; 4] {}
        impl Sealed for [u32; 2] {}
        impl Sealed for [i32; 2] {}
        impl Sealed for [f32; 2] {}
        impl Sealed for u64 {}
        impl Sealed for i64 {}
        impl Sealed for f64 {}
    }

    impl Is64BitsUnaligned for [u8; 8] {}
    impl Is64BitsUnaligned for [i8; 8] {}
    impl Is64BitsUnaligned for [u16; 4] {}
    impl Is64BitsUnaligned for [i16; 4] {}
    impl Is64BitsUnaligned for [u32; 2] {}
    impl Is64BitsUnaligned for [i32; 2] {}
    impl Is64BitsUnaligned for [f32; 2] {}
    impl Is64BitsUnaligned for u64 {}
    impl Is64BitsUnaligned for i64 {}
    impl Is64BitsUnaligned for f64 {}

    /// Load 64 bits into the low half of an __m128i, zeroing the high half.
    ///
    /// This is a **safe** function callable from any `#[target_feature(enable = "sse2")]`
    /// context (including `#[arcane]` functions) without needing an `unsafe` block.
    ///
    /// # Safety guarantees
    /// - Reference validity ensures src points to valid memory
    /// - `_mm_loadl_epi64` only reads 8 bytes (the trait ensures correct size)
    /// - Operation is documented as unaligned-safe
    #[inline]
    #[target_feature(enable = "sse2")]
    pub fn mm_loadl_epi64<T: Is64BitsUnaligned>(src: &T) -> __m128i {
        // SAFETY: src is valid for 8 bytes, _mm_loadl_epi64 reads exactly 8 bytes
        unsafe { _mm_loadl_epi64(core::ptr::from_ref(src).cast()) }
    }

    /// Store the low 64 bits of an __m128i to memory.
    ///
    /// This is a **safe** function callable from any `#[target_feature(enable = "sse2")]`
    /// context (including `#[arcane]` functions) without needing an `unsafe` block.
    ///
    /// # Safety guarantees
    /// - Reference validity ensures dst points to valid, writable memory
    /// - `_mm_storel_epi64` only writes 8 bytes (the trait ensures correct size)
    /// - Operation is documented as unaligned-safe
    #[inline]
    #[target_feature(enable = "sse2")]
    pub fn mm_storel_epi64<T: Is64BitsUnaligned>(dst: &mut T, val: __m128i) {
        // SAFETY: dst is valid for 8 bytes, _mm_storel_epi64 writes exactly 8 bytes
        unsafe { _mm_storel_epi64(core::ptr::from_mut(dst).cast(), val) }
    }

    /// Load 64 bits as a double-precision float.
    #[inline]
    #[target_feature(enable = "sse2")]
    pub fn mm_load_sd(src: &f64) -> __m128d {
        // SAFETY: src is valid for 8 bytes
        unsafe { _mm_load_sd(src) }
    }

    /// Store the low double-precision float to memory.
    #[inline]
    #[target_feature(enable = "sse2")]
    pub fn mm_store_sd(dst: &mut f64, val: __m128d) {
        // SAFETY: dst is valid for 8 bytes
        unsafe { _mm_store_sd(dst, val) }
    }
}

// ============================================================================
// aarch64 NEON 64-bit operations
// ============================================================================

#[cfg(target_arch = "aarch64")]
pub use aarch64::*;

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use core::arch::aarch64::*;

    /// Trait for types that can be used as 64-bit unaligned memory operands.
    pub trait Is64BitsUnaligned: private::Sealed {
        const SIZE: usize = 8;
    }

    mod private {
        pub trait Sealed {}
        impl Sealed for [u8; 8] {}
        impl Sealed for [i8; 8] {}
        impl Sealed for [u16; 4] {}
        impl Sealed for [i16; 4] {}
        impl Sealed for [u32; 2] {}
        impl Sealed for [i32; 2] {}
        impl Sealed for [f32; 2] {}
        impl Sealed for u64 {}
        impl Sealed for i64 {}
        impl Sealed for f64 {}
    }

    impl Is64BitsUnaligned for [u8; 8] {}
    impl Is64BitsUnaligned for [i8; 8] {}
    impl Is64BitsUnaligned for [u16; 4] {}
    impl Is64BitsUnaligned for [i16; 4] {}
    impl Is64BitsUnaligned for [u32; 2] {}
    impl Is64BitsUnaligned for [i32; 2] {}
    impl Is64BitsUnaligned for [f32; 2] {}
    impl Is64BitsUnaligned for u64 {}
    impl Is64BitsUnaligned for i64 {}
    impl Is64BitsUnaligned for f64 {}

    /// Load 8 bytes as uint8x8_t.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn vld1_u8(src: &[u8; 8]) -> uint8x8_t {
        // SAFETY: src is valid for 8 bytes
        unsafe { core::arch::aarch64::vld1_u8(src.as_ptr()) }
    }

    /// Store uint8x8_t (8 bytes) to memory.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn vst1_u8(dst: &mut [u8; 8], val: uint8x8_t) {
        // SAFETY: dst is valid for 8 bytes
        unsafe { core::arch::aarch64::vst1_u8(dst.as_mut_ptr(), val) }
    }

    /// Load 4 i16 values as int16x4_t.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn vld1_s16(src: &[i16; 4]) -> int16x4_t {
        // SAFETY: src is valid for 8 bytes (4 * 2)
        unsafe { core::arch::aarch64::vld1_s16(src.as_ptr()) }
    }

    /// Store int16x4_t (4 i16 values) to memory.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn vst1_s16(dst: &mut [i16; 4], val: int16x4_t) {
        // SAFETY: dst is valid for 8 bytes (4 * 2)
        unsafe { core::arch::aarch64::vst1_s16(dst.as_mut_ptr(), val) }
    }

    /// Load 2 i32 values as int32x2_t.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn vld1_s32(src: &[i32; 2]) -> int32x2_t {
        // SAFETY: src is valid for 8 bytes (2 * 4)
        unsafe { core::arch::aarch64::vld1_s32(src.as_ptr()) }
    }

    /// Store int32x2_t (2 i32 values) to memory.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn vst1_s32(dst: &mut [i32; 2], val: int32x2_t) {
        // SAFETY: dst is valid for 8 bytes (2 * 4)
        unsafe { core::arch::aarch64::vst1_s32(dst.as_mut_ptr(), val) }
    }

    // ========================================================================
    // 128-bit (quad-word) NEON operations
    // ========================================================================

    /// Load 16 bytes as uint8x16_t.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn vld1q_u8(src: &[u8; 16]) -> uint8x16_t {
        unsafe { core::arch::aarch64::vld1q_u8(src.as_ptr()) }
    }

    /// Store uint8x16_t (16 bytes) to memory.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn vst1q_u8(dst: &mut [u8; 16], val: uint8x16_t) {
        unsafe { core::arch::aarch64::vst1q_u8(dst.as_mut_ptr(), val) }
    }

    /// Load 8 i16 values as int16x8_t.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn vld1q_s16(src: &[i16; 8]) -> int16x8_t {
        unsafe { core::arch::aarch64::vld1q_s16(src.as_ptr()) }
    }

    /// Store int16x8_t (8 i16 values) to memory.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn vst1q_s16(dst: &mut [i16; 8], val: int16x8_t) {
        unsafe { core::arch::aarch64::vst1q_s16(dst.as_mut_ptr(), val) }
    }

    /// Load 8 u16 values as uint16x8_t.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn vld1q_u16(src: &[u16; 8]) -> uint16x8_t {
        unsafe { core::arch::aarch64::vld1q_u16(src.as_ptr()) }
    }

    /// Store uint16x8_t (8 u16 values) to memory.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn vst1q_u16(dst: &mut [u16; 8], val: uint16x8_t) {
        unsafe { core::arch::aarch64::vst1q_u16(dst.as_mut_ptr(), val) }
    }

    /// Load 4 i32 values as int32x4_t.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn vld1q_s32(src: &[i32; 4]) -> int32x4_t {
        unsafe { core::arch::aarch64::vld1q_s32(src.as_ptr()) }
    }

    /// Store int32x4_t (4 i32 values) to memory.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn vst1q_s32(dst: &mut [i32; 4], val: int32x4_t) {
        unsafe { core::arch::aarch64::vst1q_s32(dst.as_mut_ptr(), val) }
    }

    /// Load 4 u32 values as uint32x4_t.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn vld1q_u32(src: &[u32; 4]) -> uint32x4_t {
        unsafe { core::arch::aarch64::vld1q_u32(src.as_ptr()) }
    }

    /// Store uint32x4_t (4 u32 values) to memory.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn vst1q_u32(dst: &mut [u32; 4], val: uint32x4_t) {
        unsafe { core::arch::aarch64::vst1q_u32(dst.as_mut_ptr(), val) }
    }

    /// Load 2 u64 values as uint64x2_t.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn vld1q_u64(src: &[u64; 2]) -> uint64x2_t {
        unsafe { core::arch::aarch64::vld1q_u64(src.as_ptr()) }
    }

    /// Store uint64x2_t (2 u64 values) to memory.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn vst1q_u64(dst: &mut [u64; 2], val: uint64x2_t) {
        unsafe { core::arch::aarch64::vst1q_u64(dst.as_mut_ptr(), val) }
    }

    // ========================================================================
    // Additional 64-bit variants needed by mc_arm
    // ========================================================================

    /// Load 4 u16 values as uint16x4_t.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn vld1_u16(src: &[u16; 4]) -> uint16x4_t {
        unsafe { core::arch::aarch64::vld1_u16(src.as_ptr()) }
    }

    /// Store uint16x4_t (4 u16 values) to memory.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn vst1_u16(dst: &mut [u16; 4], val: uint16x4_t) {
        unsafe { core::arch::aarch64::vst1_u16(dst.as_mut_ptr(), val) }
    }

    /// Load 2 u32 values as uint32x2_t.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn vld1_u32(src: &[u32; 2]) -> uint32x2_t {
        unsafe { core::arch::aarch64::vld1_u32(src.as_ptr()) }
    }

    /// Store uint32x2_t (2 u32 values) to memory.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn vst1_u32(dst: &mut [u32; 2], val: uint32x2_t) {
        unsafe { core::arch::aarch64::vst1_u32(dst.as_mut_ptr(), val) }
    }

    /// Load 1 u64 value as uint64x1_t.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn vld1_u64(src: &[u64; 1]) -> uint64x1_t {
        unsafe { core::arch::aarch64::vld1_u64(src.as_ptr()) }
    }

    /// Store uint64x1_t (1 u64 value) to memory.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn vst1_u64(dst: &mut [u64; 1], val: uint64x1_t) {
        unsafe { core::arch::aarch64::vst1_u64(dst.as_mut_ptr(), val) }
    }
}
