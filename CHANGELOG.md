# Changelog

All notable changes to the `rav1d-safe` crate are documented in this file. Format follows [Keep a Changelog](https://keepachangelog.com/). `rav1d-safe` is a fork of [rav1d](https://github.com/memorysafety/rav1d), which is itself a Rust port of [dav1d](https://code.videolan.org/videolan/dav1d); this fork adds archmage-based SIMD dispatch and removes the C FFI path. Entries below cover only changes made in this fork — upstream rav1d and dav1d release notes remain the canonical record for the shared decoder core. This file was backfilled from git history on 2026-04-15; the `[0.5.4]` date reflects the commit date of tag `v0.5.4` rather than the crates.io publish date.

## [Unreleased]

### QUEUED BREAKING CHANGES
<!-- Breaking changes that will ship together in the next major (or minor for 0.x) release.
     Add items here as you discover them. Do NOT ship these piecemeal — batch them. -->

### Performance
- Loopfilter YMM x8 widen for wd=8 (v+h) and wd=16 (v only) filters: detects 2 adjacent edges sharing the same filter level `l` and dispatches to a single 8-lane YMM kernel processing both at once. Generalized outer dispatcher (`lpf_v_sb_y_8bpc_inner` + `derive_levels` helper in `lpf_h_sb_y_8bpc_inner`) routes wd=4/wd=8/wd=16 to their respective x8 paths. Combined with the wd=4 narrow x8 in `c320fa1` this covers the v-filter pipeline fully and h-filter through wd=8. wd=16 h-filter remains 4-lane XMM (most fragile kernel, deferred). Quiet-system bench: 4K AVIF safe-checked ratio 1.60 → ~1.51 of ASM (~6-9% absolute wall-clock) (70576f3, cb09031, a2c2b24)
- SIMD wd=6 h-filter for UV horizontal edges with width-6 filter (previously scalar). Mirrors `wd6_simd_v` math + uses `wd8_simd_h`'s 4×4 i32 transpose-load pattern. Stores 4 contiguous bytes per row at offset -2 (40b6b3e)
- SIMD row 1D transforms for 8bpc dct8/dct16/dct32 paths (8x8, 8x16, 8x32; 16x16, 16x8, 16x32, 16x64; 32x32, 32x16, 32x8, 32x64) via new `simd_row_dct{8,16,32}_8bpc_8rows` helpers — load 8 rows × N cols column-major, run existing `dct*_1d_cols8` 8-rows-in-parallel, 8x8 i32 transpose chunks, store row-major (464bcc3, edd008a, 6becd5b)
- SIMD row 1D transform for 16x16 dct-row mixed variants (dct_adst, dct_flipadst, dct_identity) via `impl_16x16_transform_simd_row_dct_col!` macro (0caef66)
- SIMD row 1D transform for 16x16 adst/flipadst-row mixed variants (8 of them) via `simd_row_adst16_8bpc_8rows` + `impl_16x16_transform_simd_row_adst_col!` macro; `flipped` flag reverses register order after ADST (a6a8457)
- 4K AVIF safe-checked: 1.78x → **1.66x of ASM** (~7% session gap closure, ~16% closed since 2026-02 baseline of 1.98x); safe-unchecked 1.70x → **1.57x** (eee9005..a6a8457)

### Changed
- Hoist target_feature region to outer loopfilter dispatch: `lpf_h_sb_y_8bpc_inner`, `lpf_v_sb_y_8bpc_inner`, `lpf_h_sb_uv_8bpc_inner`, `lpf_v_sb_uv_8bpc_inner` wrapped in `#[arcane]` with `X64V2Token`; inner `loop_filter_4_8bpc` switched to `#[rite]` so per-edge SIMD helpers inline directly into the per-superblock target_feature region. Adds `summon_v2_x64()` in `src/cpu.rs`. Symbol-table cleanup with no wall-clock cost (eee9005, 2d0d05c)
- Switch loopfilter SIMD helpers from `X64V2Token` (SSE4.2) to `Desktop64`/`X64V3Token` (AVX2+FMA+BMI2, Haswell 2013+/Zen 1+ — universal modern x86_64). Removes the now-unused `summon_v2_x64` helper. Unlocks YMM 8-lane width for narrow v-filter (aa23eb8, 9d1fe25)
- Convert 11 inner SIMD helpers (`dct{16x16,32x32}_cols_simd`, `adst/flipadst/identity_16x16_cols_simd`, `add_{16x16,32x32,64x64}_to_dst{,_16bpc}`, `dct4_2rows_avx2`) from `#[arcane]` to `#[rite]` — they're called only from `#[arcane]` outers, so the boundary is just a hard inline barrier per CLAUDE.md rule "`#[arcane]` ONLY at outermost entry point". AVX-512 helpers correctly retain `#[arcane]` for feature elevation (f018bb5)
- Force-inline 5 SIMD row helpers (`simd_row_dct{8,16,32}_8bpc_8rows`, `simd_row_adst{8,16}_8bpc_8rows`) with `#[inline(always)]` on top of `#[rite]` — LLVM was declining the inline due to function size (ffbdca4)

### Added
- `itx_mul2x_pack!` macro: bit-exact equivalent of dav1d's `ITX_MUL2X_PACK` (`pmaddwd` + `paddd` rnd + `psrad` shift) for future i16-packed butterfly implementations. Pure safe Rust; verified across 13,312 input pairings including sign extremes (8cadc48, a9fbce9)
- `transpose_8x8_i32!` macro: consolidates the 24-instruction 8x8 i32 in-register transpose used by all five `simd_row_*_8rows` helpers (5 callers, 1 source of truth) (8cadc48)
- DC-only fast path for DCT_DCT transforms in `itxfm_dispatch_{8,16}bpc` when `eob == 0`. Computes `dc` via the canonical formula (matches `src/itx.rs:89-105` scalar wrapper), broadcasts via new `#[arcane]` helpers `dc_only_add_8bpc` / `dc_only_add_16bpc` with width-tiered AVX2/SSE2/scalar paths. dav1d hits its `.dconly` shortcut on every `eob==0` block; we now do too (33f7402)
- SIMD `cfl_ac_dispatch` in `src/safe_simd/ipred.rs` covering 4:2:0, 4:2:2, 4:4:4 chroma sampling at 8bpc. `cfl_ac_420_8bpc_inner` uses `_mm256_maddubs_epi16` for horizontal pair-sum, `cfl_ac_422_8bpc_inner` single-row pair-sum, `cfl_ac_444_8bpc_inner` `_mm256_cvtepu8_epi16` widening. ST path uses zero-copy `narrow_guard`; MT path uses `compact_read_per_row` for tile-thread safety. 16bpc remains scalar (returns false → existing fallback). `cfl_ac_rust` profile share dropped 1.49% → 1.05% (6512b9b, d48656c)
- YMM x8 narrow v-filter (`loop_filter_4_8bpc_narrow_simd_v_x8`) for adjacent wd=4 edges with same `l`. New ~155 LOC kernel doubles per-call width when consecutive vmask bits share filter level. `cargo asm` confirms 98 YMM mnemonics in the dispatcher (was 0) (c320fa1)
- `with_pixel_guard_immut` sibling to `with_pixel_guard_mut`: one immut guard for ST path (zero-copy `narrow_guard`, saves N-1 BorrowTracker calls per column), per-row compact reads for MT path. Wired into `ipred_prepare` LEFT/BOTTOM_LEFT loops — 32 guards → 1 guard for h=32 block (475e61d)

### Performance
- Right-size scalar `inv_txfm_add` tmp buffer: moved 16KB `[0; 64*64]` stack array out of runtime-sized outer fn into const-generic `inv_txfm_add_rust` as `[[0i32; W]; H]`. 4x4 now allocates 64 B; 8x8 → 256 B; 32x32 → 4 KB. Drops the per-call memset that was visible at 3% of profile via `__memset_avx512_unaligned_erms` (96e93d7)
- `ctx_refill` bulk 8-byte BSWAP load: replaces byte-by-byte loop with single `u64::from_be_bytes(buf[..8])` + mask + shift. Matches dav1d's `src/x86/msac.asm:166-178` 5-instruction refill. Pure safe Rust on 64-bit targets; falls back to byte loop for `buf.len() < 8` (4e145dc)
- Replace 8x8 scatter-loads (8 individual `_mm_set_epi32(tmp[y*8+i], ...)` cascades = 27 `vmovd` per block) with one contiguous `loadu_256!(&tmp[y*8..y*8+8])` then `castsi256_si128`/`extracti128_si256` split. Both bpc (ac490f9)

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
