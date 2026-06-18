# Handoff: aarch64 NEON bit-exactness (issue #414)

Status as of 2026-06-18. **Tracking issue: #414.**

The `mc_arm` 8-tap OOB crash (Bug A) masked the entire aarch64 NEON DSP on inter
and 16bpc frames — itx/looprestoration/cdef NEON kernels on inter-frame
transforms never ran under the `simd_test` gate, so several were never
bit-exact-verified. Fixing the crash unblocked them. The handoff's earlier "itx
is bit-exact" was true only for intra/8bpc.

---

## 1. DONE this session (all verified on main@origin)

| Commit | What |
|---|---|
| 626a3d8 | mc_arm 8-tap **OOB crash fixed (all bitdepths)** + **8bpc MC bit-exact** (put+prep, all filters) + MC dual-compute harness |
| 9fe68bd5 | **env-gate** all 6 simd_test harnesses (mc/itx/lr/cdef/lf): panic by default, `SIMD_TEST_LOG=1` logs-and-continues |
| e8af9a8b | **16bpc MC bit-exact** (10+12 bit) — dynamic shifts + bitdepth-dependent rounding |
| 710537f8 | **looprestoration wiener5 bit-exact** — was mis-centered (13k mismatches) |

**MC is fully bit-exact across 8/10/12-bit (0 mismatches over the full dav1d
corpus). LR wiener5 + wiener7 bit-exact. Loopfilter bit-exact (0). MC_PREP 0.**

---

## 2. The SIMD_TEST_LOG workflow (the tool)

All 6 harnesses are env-gated: unset = panic on first NEON/scalar divergence (the
bit-exactness gate); `SIMD_TEST_LOG=1` = log `MODULE_MISMATCH … nbad= max_diff=`
and continue → ONE decode pass = full cross-module inventory.
- Native arm64: **Hetzner arm-big** (`ssh arm-big`, repo `~/work/zen/rav1d-safe-itx`,
  `source ~/.cargo/env`, `CARGO_BUILD_JOBS=7`, dav1d corpus at `test-vectors/`).
- **Run SINGLE-THREADED** (`--test-threads=1 --nocapture`) when reading the log live —
  `--test-threads=3` buffers per-test output and the mismatch lines don't stream.
- Build `--features bitdepth_8,bitdepth_16,simd_test` (~1m07s). Verify a fix:
  re-run WITHOUT SIMD_TEST_LOG → no panic == bit-exact.

---

## 3. REMAINING (measured 2026-06-18, full corpus)

### ITX (type9=IDTX, type11=H_DCT, type0=DCT_DCT) — inter-frame configs the prior itx session never reached
**DONE (`518bde8a`):** IDTX 8x32 + 32x8 (non-rect2 identity-rect).
**Remaining:** IDTX 16x32/32x16 (rect2 — need the `scale_input` ×1/√2 the NEON
omits), H_DCT @ 16x16 (type11), DCT_DCT @ 64x16/16x64 (type0).
Recipe (worked for 8x32/32x8): the scalar `inv_txfm_add` chains
row-1d → intermediate `round2(·,shift)` → col-1d → final `round2(·,4)`, which does
NOT collapse to one shift (the two rounding points matter). Identity scales: id8
×2, id16 ×2√2, id32 ×4; rect2 √2 iff log2(w)+log2(h) is odd. Also drop the `eob`
row-group early-break for identity (it can skip a non-zero row past a scan-order
threshold; identity of the zeroed tail adds nothing). Diagnose with
`SIMD_TEST_LOG=1` → `ITX_MISMATCH size= type= eob= max_diff=` (harness in
`src/itx.rs` vs `inv_txfm_add_rust`). Files: `src/safe_simd/itx_arm.rs` +
`itx_arm_neon_64.rs` + `itx_arm_neon_large_rect.rs`. Per-size shift math in memory
`arm64-neon-full-inventory`.

### CDEF: ~745 mismatches
Root finding: arm `padding_8bpc`/`16bpc` (cdef_arm.rs:53) use a **u16** tmp with
sentinel `8191`, but scalar `fill()` (cdef.rs:406) uses an **i16** tmp with
`i16::MIN`. In the pri+sec branch this corrupts `max` (8191 vs the real neighbour
max). The clean fix is to make the arm CDEF tmp `i16` + `i16::MIN` sentinel
throughout (padding + `cdef_filter_block_{8,16}bpc_inner`). NOTE: a reported
mismatch was *pri-only* (no min/max) / 3-pixel / diff=1 — both sentinels give
constrain()==0 there, so that case is a *separate* subtle bug; runtime-trace the
tmp buffer for one mismatching block (variant=0 pri=3 sec=0 dir=4 damping=5) to
pin it. The filter bodies themselves match scalar; the bug is in padding/edges.

### 12bpc latent: prep_8tap_16bpc_inner still hardcodes intermediate_bits=4
MC_PREP 16bpc is 0 (unexercised by the corpus), so it didn't surface, but
`prep_8tap_16bpc_inner` (mc_arm.rs) hardcodes `intermediate_bits = 4u8` and the
prep V-filter `v_filter_8tap_16bpc_to_i16_neon` hardcodes `::<8>`. If 16bpc
compound prediction ever appears, fix as in put: ib = `bitdepth_max.leading_zeros()-2`,
plumb `bitdepth_max` into the prep inner, make the V-filter shift dynamic.

---

## 4. Lessons (apply to itx/cdef)
- **MEASURE, don't eyeball.** Every MC fix this session came from the harness
  localizing the exact (size/filter/case); guessing the 16bpc shift conventions
  repeatedly failed until I read the actual `sh` args.
- **Compensating bugs hide rounding.** The 16bpc put had two wrong hardcoded
  shifts that cancelled in the 2-pass, leaving only max_diff=2 — fixed by honoring
  `sh` dynamically (vshl by negative count), not by patching one shift.
- **The scalar reference is ground truth.** wiener5 was fixed by making the arm
  path do *exactly* what `wiener_rust` does (always 7-tap, zero outer coeffs).
- Memory: [[arm64-neon-full-inventory]], [[feedback-fix-simd-not-scalar-fallback]].
