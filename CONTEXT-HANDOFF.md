# Handoff: aarch64 NEON bit-exactness (issue #414)

Status as of 2026-06-18. Continuation plan after the itx fix.
**Tracking issue: #414 — read it (and its root-cause comment) first.**

> (This file previously documented "DisjointMut Strided Guard Support" — that
> work is long done; tile threading works. Superseded by this handoff.)

---

## 1. Where things stand

| Area | State |
|---|---|
| #412 cooperative cancellation | ✅ done, CI green |
| #14 aarch64 16bpc looprestoration panic | ✅ verified fixed (da53bfa3) |
| #400 aarch64 NEON **itx** (8bpc) | ✅ **bit-exact**, re-enabled, CI green on native arm64 (commits 486d91b4, a8a1bb7b) |
| **mc_arm** (motion comp) | ❌ **broken** — OOB panic + 3-tap offset (this doc) |
| cdef / ipred / looprestoration / loopfilter / filmgrain on arm64 | ❓ **unknown / masked** — mc_arm panics first, the conformance never reaches them |
| itx **16bpc** large DCTs | ⚠️ not NEON-dispatched (scalar fallback — correct but unoptimized) |

The full native-arm64 dav1d md5 conformance **does not pass** yet. `decode_md5_committed`
(intra still-images) passes because those vectors have no motion compensation.

---

## 2. The mc_arm bug (precise)

Two coupled bugs, both from the arm 8-tap MC using a **block-positioned slice** instead of
x86's `(full_buffer, src_base)` contract.

### Bug A — OOB panic
`put_8tap_8bpc_inner` (+ prep/16bpc siblings) compute, for output rows `y < 3` in the H+V and
V-only paths:
```rust
let src_offset = if y >= 3 { (y - 3) * src_stride }
                 else { 0usize.wrapping_sub((3 - y) * src_stride) };  // fakes a negative index
let src_row = &src[src_offset..];                                     // panics: index ~usize::MAX
```
A slice cannot index before its start. It only "survived" because `mc_put_dispatch_inner` *also*
does `src_base.wrapping_sub(3*stride + 3)` unconditionally, while recon.rs gives the emu_edge
buffer only a **conditional** margin:
```
recon.rs:~1801   offset: stride * (my != 0) as usize * 3 + (mx != 0) as usize * 3
```
So `my==0` / `mx==0` blocks (or edge blocks missing that margin direction) underflow. Observed:
`range start index 18446744073709547773 out of range for slice of length 983040`.

### Bug B — systematic 3-col / 3-row tap offset
`h_filter_8tap_8bpc_put_neon` (mc_arm.rs:3017) reads **forward**: `src[col+0..7], src[col+1..],
… src[col+7..]`. So its `src_row` must start at the **filter-left** = block col −3. The dispatch
hands it block col 0 (`src_adj = &src_full[3*stride + 3..]`). So even where it doesn't panic, the
taps are shifted by 3 → arm MC was **never bit-exact**. A "no longer panics" check will NOT catch
this — you must verify zero diff vs the scalar reference.

### The x86 reference (correct contract, passes conformance) — `src/safe_simd/mc.rs`
- `mc_put_dispatch_inner` x86 (~12153): for 8-tap passes `(src_guard.as_bytes() /*full*/,
  src_base*pixel_size, src_stride)` — **no sub-slicing**.
- `put_8tap_8bpc_dispatch_inner` (11915) takes `src: &[u8], src_base: usize, src_stride: isize`
  and indexes `src[src_base + signed_offset]`. Full buffer + conditional emu_edge/border margin
  keeps reads in bounds; the offset includes the −3 filter-left.
- emu_edge: arm `emu_edge_dispatch` (mc_arm.rs:6237) returns `false`, so **both** arches use
  `emu_edge_rust` (mc.rs:1326) — identical temp buffer. The bug is purely in how arm *reads*.

---

## 3. Fix plan

Convert the arm 8-tap MC to x86's `(full_buffer, src_base)` contract. All in
`src/safe_simd/mc_arm.rs`.

### Inners — add `src_base: usize`; replace every `&src[off..]` with
### `&src[src_base.wrapping_add_signed(signed_off)..]` using the correct filter-top-left offset:
- `put_8tap_8bpc_inner`   (3437) — `0usize.wrapping_sub` sites: 3467, 3507
- `prep_8tap_8bpc_inner`  (3719) — 3748, 3838
- `put_8tap_16bpc_inner`  (4252) — 4282, 4320
- `prep_8tap_16bpc_inner` (4493) — 4521, 4616

### Dispatches — drop `src_start = src_base.wrapping_sub(...)` + `src_full`/`src_adj`; pass `(src_bytes /*full*/, src_base)`:
- `mc_put_dispatch_inner`  (5580) — `wrapping_sub` sites: 5903 (8bpc), 5956 (16bpc)
- `mc_prep_dispatch_inner` (~6000) — 6107 (8bpc), 6147 (16bpc)

### Correct per-case offsets (relative to `src_base` = block col 0 / row 0)
H-filter reads forward `src_row[c..c+7]`, so `src_row` must start at **block col −3**:
- **H+V (Case 1)**, mid row `y` in `0..h+7`: `off = (y as isize - 3) * stride - 3`.
- **H-only (Case 2)**, row `y` in `0..h`: `off = y * stride - 3`  (current `y*stride` w/o −3 is Bug B).
- **V-only (Case 3)**, row `y`: `off = (y as isize - 3) * stride` (col 0 — confirm v-filter col read).
- prep variants: same shape; verify the prep H/V helpers' read direction independently.

`wrapping_add_signed(off)` stays in bounds because the margin is present **in the filtered
direction** (recon.rs's conditional offset guarantees exactly that), and each inner only takes the
negative-offset branch for the filtered direction.

### Watch out for
- 16bpc mixes **u16-unit** and **byte** strides (`src_stride_u16` vs `src_stride_u`) — keep units
  consistent per call.
- `put_bilin` is fine (no negative offsets) — leave it.
- Case 1's v-filter reads `&mid[y..]` (the intermediate), not `src` — no change there.

---

## 4. How to verify — MEASURE, don't eyeball the offset math

### Native arm64 box: Hetzner **arm-big**
- `ssh arm-big` (167.233.18.12, Neoverse-N1, 8c/15G, user `ubuntu`). Dedicated remote — **exempt
  from the `nice` workstation rule**.
- **`source ~/.cargo/env` first** (non-interactive ssh lacks PATH). rustc 1.96; nextest NOT
  installed → use `cargo test`. `export CARGO_BUILD_JOBS=7`.
- Checkout: `~/work/zen/rav1d-safe-itx` (rsync copy, **not git**). Full dav1d corpus symlinked at
  `~/work/zen/rav1d-safe-itx/test-vectors`.
- Loop: edit locally → `rsync -az src/safe_simd/mc_arm.rs arm-big:~/work/zen/rav1d-safe-itx/src/safe_simd/mc_arm.rs`
  → build+test on arm-big (~1m17s cold). For a full re-sync use the same exclude list as before
  (`target/ .git/ .jj/ test-vectors/ fuzz/target/ *.log .workongoing`).

### Build a MC dual-compute harness (do this first — mirrors itx's `simd_test`)
itx's harness lives in `src/itx.rs` ~439-498 behind feature `simd_test`: at the dispatch site it
runs the SIMD, snapshots output, restores inputs, runs the **generic scalar** reference, and
asserts equal. Replicate this for the MC dispatch (`mc_put_dispatch_inner` / `mc_prep_dispatch_inner`
call sites — scalar refs are `put_8tap_rust`/`prep_8tap_rust` in mc.rs:139/266). Then change the
assert to `eprintln!("MC_MISMATCH w=.. h=.. mx=.. my=.. max_diff=..")` and decode any inter vector
to scope ALL divergences in one pass (magnitude → rounding vs structural vs edge-only). This lets
you verify each of the 4 inners in isolation without needing all of them fixed first.

### Ground truth = full conformance (zero tolerance)
```
ssh arm-big 'source ~/.cargo/env && cd ~/work/zen/rav1d-safe-itx && \
  cargo test --release --no-default-features --features bitdepth_8,bitdepth_16 \
  --test decode_md5_verify -- --nocapture'
```
It only goes green once **all four** 8-tap inners are fixed (it panics on whichever path is still
broken). Require **zero** diff vs scalar — Bug B means non-panicking blocks can still be wrong.

---

## 5. After mc_arm: finish the inventory

Once MC is bit-exact the conformance reaches **cdef_arm / ipred_arm / looprestoration_arm /
loopfilter_arm / filmgrain_arm**, currently masked. Expect more bugs (e.g. documented
`loopfilter_arm.rs:69` negative-stride OOB). For each: dual-compute harness → fix bit-exact on
arm-big → re-run conformance → repeat until green. Then:
- Flip the native-arm64 `decode_md5_verify` CI job from `continue-on-error` to a **hard gate**
  (`.github/workflows/ci.yml`, step added in a8a1bb7b).
- Dispatch **16bpc** itx large-DCT NEON (currently scalar) and verify with `simd_test`.

---

## 6. Lessons from the itx fix (apply here)

- **The dual-compute diagnostic is the tool.** Log every mismatch + parameters to scope all
  divergences in one decode; the magnitude tells you the bug class.
- **Cores are usually fine; the wiring is the bug.** itx's idct cores were correct; per-size
  shift/scale/layout wiring was wrong. MC's filter helpers are likely fine; the src-positioning
  contract is the bug.
- **Verify to ZERO tolerance.** itx's NEON-vs-scalar unit tests passed under `MAX_DIFF=15/40` and
  that masked real bugs. Accept nothing less than bit-exact.
- **Keep `main` correct while iterating.** Work in `rav1d-safe-itx` on arm-big until a path is
  verified bit-exact; only then push to `main`.

---

## 7. Key references

- Tracking: **#414** (+ root-cause comment).
- x86 correct reference: `src/safe_simd/mc.rs` — `put_8tap_8bpc_dispatch_inner` (11915), dispatch
  (~12153), `emu_edge_rust` (1326), `put_8tap_rust`/`prep_8tap_rust` (139/266).
- recon.rs conditional emu_edge offset: ~line 1801.
- itx dual-compute model: `src/itx.rs` ~439-498 (feature `simd_test`).
- Memory: `feedback-fix-simd-not-scalar-fallback` (arm-big details, the itx recipe, the "fix SIMD
  not scalar / no perf on the table" directive) and `compact-buffer-tile-threading`.
