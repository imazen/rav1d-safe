# Changelog

All notable changes to the `rav1d-safe` crate are documented in this file. Format follows [Keep a Changelog](https://keepachangelog.com/). `rav1d-safe` is a fork of [rav1d](https://github.com/memorysafety/rav1d), which is itself a Rust port of [dav1d](https://code.videolan.org/videolan/dav1d); this fork adds archmage-based SIMD dispatch and removes the C FFI path. Entries below cover only changes made in this fork — upstream rav1d and dav1d release notes remain the canonical record for the shared decoder core. This file was backfilled from git history on 2026-04-15; the `[0.5.4]` date reflects the commit date of tag `v0.5.4` rather than the crates.io publish date.

## [Unreleased]

### QUEUED BREAKING CHANGES
<!-- Breaking changes that will ship together in the next major (or minor for 0.x) release.
     Add items here as you discover them. Do NOT ship these piecemeal — batch them. -->

### Performance
- SIMD row 1D transforms for 8bpc dct8/dct16/dct32 paths (8x8, 8x16, 8x32; 16x16, 16x8, 16x32, 16x64; 32x32, 32x16, 32x8, 32x64) via new `simd_row_dct{8,16,32}_8bpc_8rows` helpers — load 8 rows × N cols column-major, run existing `dct*_1d_cols8` 8-rows-in-parallel, 8x8 i32 transpose chunks, store row-major (464bcc3, edd008a, 6becd5b)
- SIMD row 1D transform for 16x16 dct-row mixed variants (dct_adst, dct_flipadst, dct_identity) via `impl_16x16_transform_simd_row_dct_col!` macro (0caef66)
- SIMD row 1D transform for 16x16 adst/flipadst-row mixed variants (8 of them) via `simd_row_adst16_8bpc_8rows` + `impl_16x16_transform_simd_row_adst_col!` macro; `flipped` flag reverses register order after ADST (a6a8457)
- 4K AVIF safe-checked: 1.78x → **1.66x of ASM** (~7% session gap closure, ~16% closed since 2026-02 baseline of 1.98x); safe-unchecked 1.70x → **1.57x** (eee9005..a6a8457)

### Changed
- Hoist target_feature region to outer loopfilter dispatch: `lpf_h_sb_y_8bpc_inner`, `lpf_v_sb_y_8bpc_inner`, `lpf_h_sb_uv_8bpc_inner`, `lpf_v_sb_uv_8bpc_inner` wrapped in `#[arcane]` with `X64V2Token`; inner `loop_filter_4_8bpc` switched to `#[rite]` so per-edge SIMD helpers inline directly into the per-superblock target_feature region. Adds `summon_v2_x64()` in `src/cpu.rs`. Symbol-table cleanup with no wall-clock cost (eee9005, 2d0d05c)

## [0.5.5] - 2026-04-17

### Changed
- Replace blanket `#![allow(clippy::all)]` with a targeted lint policy across 27 files: 22 specific lint allows (each documented with warning count and rationale) cover pervasive C-port patterns such as `precedence`, `too_many_arguments`, `unnecessary_cast`, `identity_op`, and `needless_range_loop`, while ~100 warnings for the remaining enabled lints were fixed in place (db99f94, #7)
- Add crate-level allows for seven additional clippy lints that fire on CI's clippy 1.87+ (`duplicated_attributes`, `manual_is_multiple_of`, `let_and_return`, `unnecessary_map_on_constructor`, `clone_on_copy`, `option_map_unit_fn`, `unnecessary_lazy_evaluations`) — all pervasive C-port patterns not worth fixing individually (8c6621c, #7)

### Fixed
- Restore `MsacAsmContext` visibility for asm builds: the lint-audit refactor had accidentally gated the type behind `#[cfg(not(asm_msac))]`, breaking the `asm`-feature CI job; the erroneous cfg gate is removed and the manual `Default` impl (needed because the conditionally-compiled `symbol_adapt16` fn-pointer field doesn't derive `Default`) is reinstated (96dde32, #7)

### Tests
- `CpuLevel` doctest in `src/managed.rs` builds `Settings` via `Settings::default()` plus field mutation instead of a bare struct expression, avoiding E0639 on `#[non_exhaustive]` structs across the crate boundary that doctests compile against (008a811)

### Internal
- Ignore the `.workongoing` coordination marker file (008a811)

## [0.5.4] - 2026-04-10

Patch release focused on concurrency safety, parser hardening, and fuzz coverage.

### Fixed
- CDEF tile threading race
- MV parsing overflow guard
- `wrapping_sub` in `read_golomb`

### Tests
- AV1 fuzz dictionary expansion
