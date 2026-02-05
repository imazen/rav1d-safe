//! Safe wrappers for partial (64-bit) SIMD load/store operations.
//!
//! These are NOT provided by safe_unaligned_simd because it lacks an Is64BitsUnaligned trait.
//! The wrappers are safe functions with #[target_feature], callable from #[arcane] contexts
//! without needing unsafe blocks.
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
}
