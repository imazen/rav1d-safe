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
| R2 | itx NEON tier (rdm sqrdmulh) | Arm64V2 | `safe_simd/itx_arm*.rs` | — | TODO — *the Arm64V2 half*. The **baseline-NEON 16bpc tier landed 2026-08-07** (`safe_simd/itx_arm_hbd.rs`, every shape with max(w,h) ≤ 16, all 16 non-WHT types); 32/64-point 16bpc still scalar. See R2-16bpc below | 560f4ec |
| R3 | loopfilter/cdef/looprestoration ARM tier | NEON (baseline) | `safe_simd/{loopfilter,cdef,looprestoration}_arm.rs` | p2-kernels, lf-neon-port, cdef-neon, lr-neon-tier | **ALL THREE DONE.** CDEF (filter 8bpc+16bpc, all 3 shapes, all 3 strength branches, direction search), LOOP FILTER (all 4 widths x both directions x 8/10/12 bpc), LOOP RESTORATION (wiener7/wiener5/sgr_5x5/sgr_3x3/sgr_mix at both bit depths, 2026-08-08). All bit-exact against the corpus. See the R3 note below | 70c1a70, 8c1fa2d, 998d743, 3b44f6d, d751493, ac962fa, 2bd52c7 |

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

- **CDEF: DONE** (2026-08-07, `perf/cdef-neon` 8c1fa2d + 998d743; record
  `benchmarks/cdef_neon_2026-08-07.meta`). Filter at 8bpc and 16bpc, all three
  block shapes, all three strength branches, plus the direction search. One
  vector per destination row; twelve tap loads per ROW instead of per PIXEL.
  Measured paired base->head: 8bpc 1.019x at t=1/2/4/8, 10bpc 1.154x / 1.140x /
  1.120x / 1.077x. Per-kernel: 10bpc CDEF 17.32% of decode (104 ms/frame) ->
  4.50% (23.6 ms/frame); 8bpc 4.86% (20.3 ms) -> 4.21% (17.3 ms).
- **Do not repeat the three-vector bit-identity check.** 70c1a70's 8bpc kernel
  passed `__simd_test_log` on three vectors and was still NOT bit-exact: it used
  8191 as the padding sentinel and folded it into `max`, disabling the upper half
  of the reference's `iclip` on edge blocks. The FULL dav1d-test-data corpus (783
  vectors, `scripts/perf/cdef_oracle_sweep.sh`) caught it — 5 mismatching blocks
  across 4 8-bit vectors. Use `0x8000` (== `i16::MIN`, dav1d's own convention)
  with unsigned `min` / signed `max` and the sentinel is inert everywhere.
- **CDEF 16bpc's `pri_tap` fix** (#446, PR #448 `fix/446-arm-cdef-highbd-pri-tap`)
  is carried on `perf/cdef-neon` because the 16bpc vector path cannot be bit-exact
  without it. The same corpus sweep measured its blast radius at the branch point:
  139 10-bit vectors / 401,567 blocks and 46 12-bit vectors / 43,456 blocks.
- **CDEF's remaining cost is borrow COUNT, not arithmetic.** At t=1 the filter row
  is 0.22% of decode and DisjointMut traffic around it is 1.60%. The per-block
  guard count is down 28 -> 20 (folding the block-row copy and the two
  right-context pixels into one same-extent guard); 20 is the floor without
  either a wide guard (banned under tile threading) or dav1d's `edges == 0xf`
  path that reads the 12x12 window straight out of `dst`.
  Bit-exact against `cdef_filter_block_rust` (zero CDEF_MISMATCH under
  `__simd_test_log` on three vectors) and the full conformance corpus is
  unchanged. Worth 1.143x of whole-decode t=1 wall on v4k_8tile.
- **CDEF 16bpc: still scalar**, and separately NOT bit-exact with the scalar
  reference — a `__simd_test_log` decode of the 10bpc 4K vector logs ~147k
  CDEF_MISMATCH, all +-1 (e.g. simd=525 scalar=524). That is PR #448's subject
  (`fix/446-arm-cdef-highbd-pri-tap`), not this row's; port the 16bpc vector
  path only after that lands, or you will be chasing its rounding bug.
- **Loop filter: PORTED** (3b44f6d + d751493, `benchmarks/lf_neon_2026-08-07.meta`).
  All four widths (`wd` 4 / 6 / 8 / 16 = the spec's filter4 / filter6 / filter8 /
  filter14), both edge directions, at 8, 10 and 12 bits, over fused runs of 1..4
  groups. One `u16`-lane kernel serves every bit depth. The seam is the compact
  scratch rectangle the scalar driver already opens, so no DisjointMut guard
  changed extent or count.

  **The warning this bullet used to carry was right, and it is the main finding.**
  Vectorising the filter arithmetic alone would have bought ~10 ms of the ~45:
  `loop_filter` itself was only 341 of 9,123 t=1 sample leaves (3.74%), while
  `LfBlock::close`'s write-back diff scan was 286 and `open`'s guards and row
  copies were another ~400. So the port had to take the surrounding machinery
  too — the diff scan is now one `vceqq` + nibble movemask (286 -> 67 leaves) and
  the row copies are monomorphized on `w` instead of a `memmove` call per row.
  Composed: t=1 v4k_8tile 417.6 -> 390.7 ms/frame.

  **What is left in this family is DisjointMut, not arithmetic.** `open` takes
  one immutable guard per picture row, and at up to 16 rows per fused run that
  is ~230 of the ~400 remaining leaves. Do NOT "fix" it by widening to a single
  rectangle guard — that is the shape this codebase documents as unsound under
  tile threading, and the x86 dispatch only takes it behind
  `tile_threading_active()`. The sound version is a tracker API that registers N
  DISJOINT ranges in one operation (`add_multi` already does 2), which is R2/R4
  work, not a kernel port.
- **Loop restoration** (HISTORICAL — see the 2026-08-08 entry below for the
  current state): **still scalar**, 1,527 lines, and measured **0.0 ms/frame**
  on the 4K vectors used here because loop restoration is off in those
  bitstreams. Do not port it on the strength of its line count; find a vector
  that exercises it first.

  2026-08-07 recount confirms this is now the LAST of R3's three: intrinsic call
  counts at `verify/compose` are cdef_arm 104, loopfilter_arm 223,
  **looprestoration_arm 0**. Two things worth knowing before porting it. It is
  not a correctness problem — per-kernel ablation over the full dav1d corpus
  says loop restoration breaks **0** of the 464 failing vectors
  (`benchmarks/aarch64_md5_attribution_2026-08-07.meta`). But it *is* a
  hand-written duplicate of `src/looprestoration.rs` that `lr_filter_dispatch`
  runs unconditionally (it ends in `true`) at both bit depths, so any drift
  between the two is a silent bug by construction — its 16bpc self-guided
  rounding already was one once. Deleting it in favour of the reference is a
  legitimate outcome of this row, not a failure to do the work.

  **2026-08-07 (later) — that is exactly what happened, and the "0.0 ms/frame"
  above is now retired as a vector artefact.** Full record:
  `benchmarks/lr_arm_vs_reference_2026-08-07.meta`.
  * The reason nobody could size this row is that a profiler reports "never
    called" and "free" as the same 0.0 ms. `src/ablate.rs` now carries
    per-family activity counters (`note` / `activity_snapshot` /
    `activity_reset`, `__ablate`-gated exactly like `is_off`, so nothing lands
    on a production hot path) and `md5_inventory --activity` emits them per
    vector. Loop restoration is active in **696 of 768** corpus vectors; it is
    off in `v4k_8tile{,_10b}`, which is the whole of the 0.0 ms.
  * On `10-bit/issues/318_tx_4x4` (LR = 76% of all kernel pixel work) `sample`
    puts `selfguided_filter` at 9.5% and the Wiener filter at 2.0% of decode
    self-time. The kernel is worth a real tier.
  * The duplicate was **slower than the reference it shadowed**: interleaved
    `__ablate` A/B, rotating order, median of 9 — 8bpc `00001147` 204.42 vs
    192.83 ms/frame (**1.060**), 10bpc `318_tx_4x4` 9.7187 vs 9.7243 (0.999).
    So it was deleted (1,436 lines); the dispatcher returns `false` and the
    caller runs the reference. 766/766 with byte-identical per-vector MD5s,
    which is also the proof the duplicate bought nothing.
  * **2026-08-08 — the tier is now WRITTEN, and R3 is DONE.** Full record:
    `benchmarks/lr_neon_tier_2026-08-08.meta`. All five variants at both bit
    depths (wiener7, wiener5, sgr_5x5, sgr_3x3, sgr_mix), ported from
    `src/looprestoration.rs` rather than from a copy of it. Both starting
    points the previous entry named were taken: the box sums are FUSED
    (one row of vertical sums into a row buffer, slid horizontally straight
    into the destination — row-major on the source instead of the reference's
    390-element column stride, and no in-place aliasing), and the scratch is a
    per-thread buffer that is never cleared instead of ~285-460 KB of stack
    memset per restoration unit.

    Measured, whole-frame decode, `__ablate` A/B on the same binary, rotated,
    idle-verified, median of 5:

    | vector | neon | scalar | ratio |
    |---|---|---|---|
    | `8-bit/data/00001147` (LR 42.6% of kernel px) | 172.22 | 196.70 | **0.876** |
    | `10-bit/issues/318_tx_4x4` (LR 76.1%) | 9.121 | 9.693 | **0.941** |
    | `8-bit/data/00000855` (LR 70.0%) | 26.334 | 26.627 | **0.989** |
    | `8-bit/data/00000645` (LR 40.4%) | 5.568 | 5.594 | 0.995 — ranges OVERLAP, not resolvable |

    Correctness: `--features __simd_test_log` over all 768 corpus vectors
    compares EVERY loop-restoration call against the scalar reference — 766
    PASS / 2 SKIP / **zero** `LR_MISMATCH` — and `md5_inventory` set-diffs BY
    NAME with the actual md5 in the key. Both gates, plus the `__lrpoison`
    read-before-write check on the reused scratch, were proved live by planting
    hazards and watching them fire.

    Still open on this row: no dotprod/i8mm and no SVE (baseline NEON only),
    and the four A/B vectors are the corpus extremes plus two mid-range cases —
    no claim is made about the median of the 696 LR-active vectors.

## R2-16bpc status (measured + landed 2026-08-07)

`itxfm_add_dispatch`'s arms were all `BPC8` on purpose: the `itx_arm_neon_*`
kernels keep transform state in `int16x8_t`, which *is* the spec's 8bpc row and
column clip and nothing wider, and their `*_16bpc_*` entry points clamp to `i16`
(5,038 `ITX_MISMATCH` on the 4K 10-bit vector for 16x16 alone, `nbad = 256` on
3,814 of them).

`safe_simd/itx_arm_hbd.rs` closes it from the other side: it vectorises the
*generic* reference (`src/itx.rs` + `src/itx_1d.rs`) in `int32x4_t` lanes, four
independent 1-D transforms per vector, each lane running the identical i32 op
sequence the scalar reference runs. Both passes load contiguously — `coeff` is
column-major so four rows at one column index are one `vld1q_s32`, and `tmp` is
row-major so four adjacent columns are one — with a 4x4 s32 transpose between
them folded together with the reference's `iclip(tmp + rnd >> shift)` step.

- Wired: every shape with `max(w, h) <= 16`, all 16 non-WHT types
  (4x4, 8x8, 16x16, 4x8, 8x4, 4x16, 16x4, 8x16, 16x8).
- **Not** wired, still scalar: anything with a 32 or 64 dimension, and WHT_WHT.
  `hbd_supported()` is the single place that says so.
- Effect at t=1 on `v4k_8tile_10b` (`sample` self/inclusive time): itx went from
  **21.93% inclusive to 12.79%**, and no `itx_1d::inv_*_1d_internal_c` sample
  remains at all. Of what is left, 5.88 points are `block_mut` guard traffic
  (`mut_slice_as` 3.29 + `remove_wide` 2.59) rather than transform arithmetic —
  the same `block_mut` cost the P2 profile flagged, now the dominant term.
- Gates: 766/766 dav1d-test-data with byte-identical per-vector MD5s; new
  `itx_arm_parity.rs` arms at 10bpc and 12bpc, small **and large** coefficients,
  plus every legal `eob` at 10bpc (12,544 cells). The large-coefficient arms are
  load-bearing: a mutation that swaps the 16bpc clips back to `i16::MIN..MAX`
  (the exact defect the old ports carry) passes both small-coefficient arms and
  fails only the large ones.

### Guard shape: per-row narrow, not `block_mut`

The 5.88 points above are why. `block_mut` / `with_pixel_guard_mut` take one
guard spanning the whole strided w x h region; both this path and the new CfL
kernel only ever touch `dst` a row at a time, so that extent bought nothing and
the wide-guard bookkeeping cost more than the kernels saved. Measured on the
CfL kernel, foreign = 0, medians of 9: the NEON kernel removes 2.46% of self
time, a wide guard adds back +0.79% `add_wide::<true>` and +0.73%
`remove_wide`, and the net is **0.14%** of wall — per-row `slice_mut(w)` guards
make the same kernel worth **0.99%**. The itx path moved 0.8858 -> 0.8535 of
base at 10bpc under the same change. Guard extents only ever NARROW here, to
exactly the pixels each kernel writes, which is the extent the scalar reference
already guards.

**Generalisation worth testing on the other aarch64 kernels:** every
`safe_simd/*_arm.rs` dispatcher that reaches for `block_mut` when its kernel is
row-sequential is paying the same toll. The 8bpc itx arms are the obvious next
candidates (`block_mut` shows at 3.54% + 2.50% `remove_wide` on `v4k_8tile`),
but their kernels index `dst` by an arbitrary row inside the transform, so the
change is not the same one-line move it was here.

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
