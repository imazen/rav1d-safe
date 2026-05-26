//! Safe SIMD implementations for ITX (Inverse Transforms)
#![allow(deprecated)] // FFI wrappers need to forge tokens
#![cfg_attr(not(feature = "unchecked"), forbid(unsafe_code))]
#![cfg_attr(feature = "unchecked", deny(unsafe_code))]
//!
//! ITX is the largest DSP module (~42k asm lines). Strategy:
//! 1. Implement full 2D transforms (not just 1D) for common sizes
//! 2. Process multiple rows/columns in parallel
//! 3. Use in-register transposition
//!
//! Most common transforms: DCT_DCT 4x4, 8x8, 16x16
//!
//! This module is ~23K LOC. To keep it navigable it is split across part
//! files included verbatim via `include!`, in definition order. Everything
//! stays in ONE module so all `macro_rules!` definitions and private
//! functions remain mutually visible exactly as if the file were unsplit —
//! no visibility changes, no macro re-export. `include!` is textual
//! concatenation, so the part files MUST be included in the original source
//! order (macros are defined before use, functions reference each other).

#![allow(unused_imports)]
#![allow(dead_code)]

include!("itx/part01_macros_and_2d_transforms.rs");
include!("itx/part02_col_1d_avx512_pmaddwd.rs");
include!("itx/part03_adst16_and_small_rect.rs");
include!("itx/part04_rect_idtx_aspect.rs");
include!("itx/part05_tests.rs");
include!("itx/part06_adst_flipadst_vh.rs");
include!("itx/part07_32x32_64x64_dct.rs");
include!("itx/part08_rect_dct_adst_16bpc.rs");
include!("itx/part09_identity_hybrid_16bpc.rs");
include!("itx/part10_dispatch.rs");
