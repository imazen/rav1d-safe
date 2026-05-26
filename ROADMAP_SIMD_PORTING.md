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
| X1 | itx — extend AVX-512 to row pass + dct8/adst/IDTX cols16 | X64V4 (+V4x where wider) | `safe_simd/itx/part*.rs` | — | TODO | |
| X2 | loopfilter 8bpc AVX-512 (all widths v+h) | X64V4 | `safe_simd/loopfilter.rs` | — | TODO | |
| X3 | cdef 8bpc AVX-512 | X64V4 | `safe_simd/cdef.rs` | — | TODO | |
| X4 | ipred directional z1/z2/z3 AVX-512ICL (vpermb) | X64V4x | `safe_simd/ipred.rs` | — | TODO | |
| X5 | itx/cdef/loopfilter 16bpc AVX-512 | X64V4 | (after X1–X3) | — | TODO | |
| R1 | mc 8tap/6tap dotprod + i8mm | Arm64V2/V3 | `safe_simd/mc_arm.rs` | — | TODO | |
| R2 | itx NEON tier (rdm sqrdmulh) | Arm64V2 | `safe_simd/itx_arm*.rs` | — | TODO | |
| R3 | loopfilter/cdef ARM tier | Arm64V2 | `safe_simd/{loopfilter,cdef}_arm.rs` | — | TODO | |

## Notes
- Targeting v0.5.7 (v0.5.6 is release-prepped, pending publish go-ahead — does not block this work).
- Independent algorithmic ports (decode_coefs 1.5.0 index-offset, msac 1.4.1, SGR 1.5.1) tracked separately — they help the AVX2 baseline on all CPUs and are ISA-independent.
- Stale-doc fix needed: CLAUDE.md claims ADST 8x8/16x16 column SIMD is wired, but the safe `itxfm_dispatch_8bpc` routes non-DCT_DCT/non-IDTX ≥8x8 to scalar. Verify/wire/correct.
