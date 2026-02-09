//! Safe wrappers for partial (64-bit) SIMD load/store operations.
//!
//! These delegate to `safe_unaligned_simd` for the actual implementations.
//! This module provides aliased function names matching our codebase conventions.

#![forbid(unsafe_code)]

// ============================================================================
// x86_64 SSE2 64-bit operations
// ============================================================================

#[cfg(target_arch = "x86_64")]
pub use x86_64_aliases::*;

#[cfg(target_arch = "x86_64")]
mod x86_64_aliases {
    use core::arch::x86_64::__m128i;
    use safe_unaligned_simd::x86_64::Is64BitsUnaligned;

    /// Load 64 bits into the low half of an __m128i, zeroing the high half.
    ///
    /// Delegates to `safe_unaligned_simd::x86_64::_mm_loadu_si64`.
    ///
    /// Requires SSE2 context (all callers are `#[arcane]` with AVX2).
    #[inline]
    #[target_feature(enable = "sse2")]
    pub fn mm_loadl_epi64<T: Is64BitsUnaligned>(src: &T) -> __m128i {
        safe_unaligned_simd::x86_64::_mm_loadu_si64(src)
    }

    /// Store the low 64 bits of an __m128i to memory.
    ///
    /// Delegates to `safe_unaligned_simd::x86_64::_mm_storeu_si64`.
    ///
    /// Requires SSE2 context (all callers are `#[arcane]` with AVX2).
    #[inline]
    #[target_feature(enable = "sse2")]
    pub fn mm_storel_epi64<T: Is64BitsUnaligned>(dst: &mut T, val: __m128i) {
        safe_unaligned_simd::x86_64::_mm_storeu_si64(dst, val)
    }
}
