# SIMD Porting Roadmap — AVX-512 (V4/V4x) + modern ARM (dotprod/rdm/i8mm)

Goal: close the gap to the latest dav1d (1.5.1) by adding the ISA tiers our
safe-SIMD build lacks. We vendored asm from dav1d 1.0.0; dav1d already shipped
AVX-512 for 8bpc mc/cdef/itx at 1.0 and added AVX-512ICL ipred (1.4) plus ARM
dotprod/i8mm mc (1.4.2/1.5.0). Our safe build is largely AVX2-only on x86 and
baseline-NEON on ARM. Per-kernel AVX-512 ≈ 1.5–2× over AVX2.

## Token tiers (confirmed against archmage 0.9.23 — `cargo read archmage`)

x86:
- `X64V3Token` / `Desktop64` = AVX2 + FMA + BMI2 (current baseline for most kernels)
- `X64V4Token` / `Server64` = x86-64-v4: AVX-512 F/BW/CD/DQ/VL
- `X64V4xToken` = V4 + ICL extras: VBMI, VBMI2, VNNI, BITALG, VPOPCNTDQ, IFMA, GFNI, VAES, VPCLMULQDQ
  - **`vpermb`/`vpermi2b` (register-resident byte permute) require VBMI → use `X64V4xToken`.** dav1d's "AVX-512ICL" == this tier.

aarch64:
- `NeonToken` = baseline NEON (current)
- `Arm64V2Token` = NEON + CRC + **RDM** (sqrdmulh) + **DotProd** (sdot/udot) + FP16 + AES + SHA2  (ARMv8.2)
- `Arm64V3Token` = Arm64V2 + FHM + FCMA + SHA3 + **I8MM** (smmla/usmmla) + BF16  (ARMv8.6)

Dispatch pattern: `incant!` for multi-tier, or `if let Some(t) = X64V4xToken::summon() { ... } else if Desktop64 ... else scalar`. Summon ONCE at the kernel entry, pass the token down. `#[arcane]` at the entry boundary, `#[rite]` for inner helpers. NEVER add `#[allow(unsafe_code)]` to `#[arcane]`/`#[rite]`.

## MANDATORY for every agent before writing code

1. `cargo read archmage` and `cargo read magetypes` — read the token system, the `#[arcane]`/`#[rite]`/`incant!` macros, and the magetypes generic SIMD primitive types (`u8x16`, `i16x8`, `i32x16`, etc.). Prefer `#[magetypes(...)]` generic kernels where the algorithm is uniform across lane widths.
2. Gate on a runtime CPU-flags check via `crate::src::cpu::summon_*` (add a `summon_avx512x()` / `summon_arm64v2/v3()` helper if missing — see `src/cpu.rs`).
3. Each kernel keeps its existing AVX2/NEON path as the fallback when the higher tier isn't available.

## Conventions

- **One agent per file** (x86 and ARM files are disjoint; itx is split into `itx/part*.rs`).
- **Worktree-isolated.** Commit per kernel. DO NOT push to origin — the parent merges.
- **Gates (x86):** default `forbid(unsafe)` + unchecked builds clean; `cargo clippy --all-targets` 0 warnings; 14/14 `decode_md5_verify`; relevant `decode_cpu_levels` at v4/native (`--test-threads=1`).
- **Gates (ARM):** `cargo check --target aarch64-unknown-linux-gnu` clean; if QEMU available, `just test-aarch64` subset. Full ARM conformance runs in CI.
- **Update your row in the matrix below** on completion (status + commit).
- **Honest-stop > false completion.** AVX-512 mask-register and `vpermb` shuffles are subtle — bit-exact vs the existing path is mandatory; use the per-kernel scalar/AVX2 reference as the oracle.

## Work matrix

| ID | Kernel | Target ISA | File | Owner | Status | Commit |
|----|--------|-----------|------|-------|--------|--------|
| X1 | itx — extend AVX-512 to row pass + dct8/adst/IDTX cols16 | X64V4 (+V4x where wider) | `safe_simd/itx/part*.rs` | x1-recovery | DONE (DCT+IDTX); ADST cols16 remains | 75d2e68, 954a252, 4b81aa8, 0d74989 |
| X2 | loopfilter 8bpc AVX-512 | X64V4 | `safe_simd/loopfilter.rs` | merged | PARTIAL: wd=16 v-filter x16 (bit-exact); wd 4/6/8 v + all h remain. Flat perf (trigger fires 0× on conformance + Zen4 downclock) | 2e24a30, 572cb77 |
| X3 | cdef 8bpc AVX-512 | X64V4 | `safe_simd/cdef.rs` | merged DONE (bit-exact; CDEF below noise floor on photo) | 629e454 |
| X4 | ipred directional z1/z2/z3 AVX-512ICL (vpermb) | X64V4x | `safe_simd/ipred.rs` | merged DONE (z1/z2/z3 bit-exact; z3 was scalar before; ~0.6% of photo profile) | ef03927, 430e7a8, 4b76f48 |
| X5 | itx/cdef/loopfilter 16bpc AVX-512 | X64V4 | (after X1–X3) | — | TODO | |
| R1 | mc 8tap dotprod + i8mm | Arm64V2/V3 | `safe_simd/mc_arm.rs`, `cpu.rs`, `build.rs` | merged | SCAFFOLDED, cfg-gated OFF (nightly intrinsics — see note). Default build = NEON | 75f044c |
| R2 | itx NEON tier (rdm sqrdmulh) | Arm64V2 | `safe_simd/itx_arm*.rs` | — | TODO | |
| R3 | loopfilter/cdef ARM tier | NEON (baseline) | `safe_simd/{loopfilter,cdef}_arm.rs` | p2-kernels | CDEF 8bpc DONE (bit-exact, 71.7 -> ~13 ms/frame at 4K t=1). CDEF 16bpc + loopfilter still scalar — see note below | 70c1a70 |

## Cross-cutting findings (2026-05-26 AVX-512/ARM wave — all merged to main)
- **Zen4 double-pumps AVX-512** (256-bit execution units) → every AVX-512 kernel benches FLAT on this dev box. Bit-exactness (14/14 MD5 on this Zen4, which executes the V4 path) is the validation; wall-clock payoff is on native-512 hardware (Intel Ice Lake server / Sapphire Rapids, Zen5). Do NOT chase Zen4 speedups for AVX-512.
- **ARM dotprod/i8mm are nightly-only std intrinsics** — the modern-ARM track is blocked on stable until #117223/#117224 stabilize, OR until a safe `sdot`/`usmmla` primitive is shipped to magetypes via inline-asm-in-audited-unsafe (the `safe_unaligned_simd` pattern). `rdm`/sqrdmulh stability for R2 still to be verified.
- Merged result: 14/14 MD5, 30/30 lib tests, 4 build combos + aarch64 cross-check clean, 0 clippy. Pure safe Rust — zero new `#[allow(unsafe_code)]`.

## R3 status (measured 2026-08-07)

The three aarch64 files this row covers opened with `//! Safe ARM NEON
implementations for ...` and imported `core::arch::aarch64::*` but contained
**zero aarch64 intrinsic calls** — the bodies were scalar per-pixel loops, so on
aarch64 the "NEON" tier for CDEF, the loop filter and loop restoration WAS the
scalar reference. Measured cost on a 4K 8bpc still at t=1
(`benchmarks/p2_kernel_profile_2026-08-07.meta`): CDEF 71.7 ms/frame against
dav1d's 4.1 (17.3x), loop filter 34.2 against 3.0 (11.3x).

- **CDEF 8bpc: ported** (`cdef_filter_block_8bpc_neon`, commit 70c1a70). One
  vector per destination row; twelve tap loads per ROW instead of per PIXEL.
  Bit-exact against `cdef_filter_block_rust` (zero CDEF_MISMATCH under
  `__simd_test_log` on three vectors) and the full conformance corpus is
  unchanged. Worth 1.143x of whole-decode t=1 wall on v4k_8tile.
- **CDEF 16bpc: still scalar**, and separately NOT bit-exact with the scalar
  reference — a `__simd_test_log` decode of the 10bpc 4K vector logs ~147k
  CDEF_MISMATCH, all +-1 (e.g. simd=525 scalar=524). That is PR #448's subject
  (`fix/446-arm-cdef-highbd-pri-tap`), not this row's; port the 16bpc vector
  path only after that lands, or you will be chasing its rounding bug.
- **Loop filter: still scalar**, but the profile moved. After the P1 compaction
  work and `perf(loopfilter)` commits 51f76a7 + 2fc646e, most of the family's
  cost is guard/copy overhead around the filter (`LfBlock::open` + `close`),
  not the filter arithmetic — vectorising `loop_filter` alone would leave that
  untouched. Read `benchmarks/p2_kernels_2026-08-07.meta` before starting.
- **Loop restoration: still scalar**, 1,527 lines, and measured **0.0 ms/frame**
  on the 4K vectors used here because loop restoration is off in those
  bitstreams. Do not port it on the strength of its line count; find a vector
  that exercises it first.

## Notes
- **R1 BLOCKER (verified 2026-05-26):** the ARM DotProd/I8MM compute intrinsics
  needed for the 8-tap dot-product kernels — `vdotq_s32` (DotProd) and
  `vusdotq_s32`/`vusmmlaq_s32` (I8MM) — are still **unstable library features**
  on the stable channel as of Rust 1.95 (project toolchain) AND latest nightly
  1.97: `stdarch_neon_dotprod` (rust-lang/rust#117224), `stdarch_neon_i8mm`
  (rust-lang/rust#117223). They require `#![feature(...)]`, which only compiles
  on nightly — incompatible with this crate's pinned `stable` toolchain and its
  `#![forbid(unsafe_code)]` mandate (which bans blocking on nightly-only
  features). archmage 0.9.23 confirms this in its own test suite ("ALL dotprod
  intrinsics are nightly-only", "I8MM ... ALL UNSTABLE"). The `summon_arm64v2/v3`
  helpers, the DotProd/I8MM H-filter kernels, the tier-selecting dispatch, and a
  CI bit-exactness test are all **landed and cross-check clean**, but the kernels
  are gated behind the OFF-by-default rustc cfgs `rav1d_arm_dotprod` /
  `rav1d_arm_i8mm` so the stable build is byte-for-byte unchanged. Equivalence vs
  NEON is proven: the DotProd `-128` source-bias correction (`+128*Σfilter`) is
  exhaustively bit-exact (8M random + boundary inputs/filter, host test +
  `test_dotprod_bias_correction_bit_exact`), and the kernel arithmetic + intrinsic
  signatures compile clean on nightly. **To activate:** when the intrinsics
  stabilize, drop the cfg gates and (on nightly until then) add
  `#![feature(stdarch_neon_dotprod, stdarch_neon_i8mm)]`; the dispatch in
  `put_8tap_8bpc_inner` / `prep_8tap_8bpc_inner` already prefers the higher tier
  via `summon_arm64v3()`/`summon_arm64v2()`. Runtime bit-exactness on real ARM
  silicon is deferred to aarch64 CI (host is x86; QEMU validated the NEON-path
  tests).
- Targeting v0.5.7 (v0.5.6 is release-prepped, pending publish go-ahead — does not block this work).
- Independent algorithmic ports (ISA-independent — help every CPU including this Zen4):
  - **decode_coefs 1.5.0 index-offset (`5ef6b241`): DONE** — ported `e24b479`, **~2.6% faster** 4K AVIF checked (bit-exact, 14/14 MD5). 2D path indexes `levels` by `rc` directly, dropping a per-coefficient `imul`. See `benchmarks/decode_coefs_index_offset_2026-05-26.md`.
  - **msac `d22de29c` minor optimizations: ALREADY PRESENT** in `src/msac.rs` (invert-once refill, `dif<<d` norm, `dif=0` init) — rav1d ported newer-than-1.0.0 msac. No work.
  - **SGR/wiener 1.5.1 C rewrites: N/A** — they rewrite the scalar reference C + cut stack; our hot path runs AVX2 SIMD. The 1.5.1 SGR speed gain was SSSE3 asm, not portable to our AVX2 safe path.
- Stale-doc fix needed: CLAUDE.md claims ADST 8x8/16x16 column SIMD is wired, but the safe `itxfm_dispatch_8bpc` routes non-DCT_DCT/non-IDTX ≥8x8 to scalar. Verify/wire/correct.
- **X1 recovery status (2026-05-26):** AVX-512 column passes now cover dct4/dct8/dct16/dct32 + IDTX (identity8/16/32) at all wide (total_w≥16) sites, plus the 16-row DCT row passes. New helpers in `part02`: `dct4_cols_avx512`, `dct8_cols_avx512`, `identity_shift_cols_avx512<SHIFT>`, `identity16_cols_avx512`. All bit-exact vs scalar oracle (lib tests `test_{dct4,dct8,identity}_cols_avx512_matches_scalar`) and 14/14 MD5. **Remaining for full X1: ADST cols16** — no `adst{8,16}_1d_cols16` exists yet (AVX2 `adst*_1d_cols8` only); needs a 16-lane ADST helper + wiring at the wide adst column sites (`part02:2007/2058`, `part04:1951`, etc.). Perf flat on this Zen4 (double-pumped 512); payoff is native-512 hardware.
