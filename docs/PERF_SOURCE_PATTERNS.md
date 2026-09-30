# Source-level performance patterns (measured)

A running ledger of which *Rust source shapes* moved decode perf/codegen in
this codebase, with the commits that measured them. Ordered roughly by how
often the pattern pays. The audience is future kernel work: when you write or
touch a `safe_simd` kernel, check this list against the shape you produced.

Measure first: `iai-callgrind` (`Ir` instruction counts, per-line attribution)
and `cargo asm`/`cargo-show-asm` (look for `panic_bounds_check`,
`core::slice::index::slice_index_fail` call sites, `movzbl`/`vpinsrb`
reconstruction chains, libc `memcpy` calls in hot loops). Iterate on
`--profile release-thin` (~2x faster rebuilds, decode parity with fat LTO).

## 1. Bounds checks — the dominant safe-Rust tax

Per-element `buf[i]` / `buf[start..start+k]` in a hot loop each emit a
trappable check; LLVM cannot merge them, so per-pixel code lowers to
`movzbl`+`vpinsrb` chains instead of one wide load.

- **Single range-checked wide load/store.** Replace `for k { v[k] = buf[start+k] }`
  with one `try_into()` on a fixed-size array, or `copy_from_slice` for stores.
  `fc30cb79`: `lpf_v_sb_y_8bpc_inner` code size 14000→8441 B, panic sites
  87→2, vpinsrb 182→0, self-Ir −44%.
- **Fixed-size window refs.** `let w: &[u8; 14] = buf[r*stride..r*stride+14].try_into().unwrap()`
  per row — one check per row; every intra-row access is then provably
  in-bounds by construction. Used across `loopfilter*.rs` h-helpers.
- **Slice once, then index.** `let row = &masks[c]` outside the loop so the
  row-bound check hoists and per-element checks fold against the loop bound.
  `93207ba7`: slice `dav1d_scans` to `[..eob+1]` once per call → per-coeff
  checks dropped, −3.9M Ir on the 4K clip.
- **`chunks_exact(N)` for stride-patterned fills.** `level_cache` fills in
  `lf_mask.rs` went from per-element `4*(off+x)` indexing to
  `chunks_exact(4)` rows — `62ab5783`, −9.7M Ir total.
- **`try_into()` on a known-N window beats `buf[i]` even when the index
  math is "obviously" in-range** — the compiler keeps the check unless the
  slice's length is a literal or a dominating bound.

## 2. Give LLVM compile-time lengths

Runtime-length `copy_from_slice`/`memcpy` per row calls libc even for 4–16
byte rows.

- **Match on block width → `copy_n_px` / const-N helpers.** `607627b6`:
  no-filter MC put rows went from a libc `memcpy` call per row to inline
  vector copies; `fbff067d`: same idiom for intra edge copies, −1.5M Ir.
  Also `docs/SIZE_SWEEP.md` measured the libc-call cost behind it.
- **`&[T; N]` params instead of `[T; N]` by value.** `51153dce`: itx row-pass
  fns took `[i16; N]` by value — a 128 B–2 KB stack memcpy *inside every
  transform call*. By-reference costs nothing.

## 3. Layout the data for the instruction you want

- **Transpose intermediates so the second pass is contiguous.**
  `e57fc9cd` warp_affine: store the 15×8 mid as `mid[x][y]` so each output
  pixel's 8 taps are contiguous → `loadu_128 + madd_epi16 + 2×hadd_epi32`
  instead of 8 scalar multiplies (all 64 dots per 8×8 became single-instr).
  `892b489d` ipred z3: reversing the left edge into an ascending `lbuf` made
  each row's tap pair contiguous → 8×8 `maddubs`+transpose block kernel,
  z3 self-Ir −59%.
- **Pair-blend with `maddubs`/`pmaddwd`, not widen+`mullo`.**
  `06eadf20` z1/z2/z3: `unpacklo/hi_epi8` byte-interleave + `maddubs` for
  edge pairs; on AVX-512 one `permutex2var_epi8` pair-index + `maddubs`
  replaced gather+widen+2×mullo (~10→5 ops per 32 px). `a2d65d1e` MC 8-tap:
  `unpacklo/hi + pmaddwd` vertical pairs halve multiplies.
- **Widen once, reuse.** `607627b6`: `widen_row` was a per-element
  u8→i16<<ib loop; now `cvtepu8_epi16 + sll_epi16` lanes.
- **Pack small/adjacent work into wider lanes.** Chroma loopfilter groups
  packed into 16-bit lanes (`6a48ea32`); small-width blocks got 128-bit
  lane tails instead of scalar fallback (`a2d65d1e` w=4/8, `6b226a5b`
  z3 w4 4-column transpose twin, `67335953` upsample_edge 4w lanes,
  `fab265e4` smooth w4 lanes, `7bc1f75b` w_mask w4/8 tails).

## 4. Kill per-call/per-block bookkeeping

- **Pool scratch instead of alloc+init per block.** `3eadabce`: per-block
  `wrap_buf` scratch was ~65% of the 10-bit profile (tracker re-init + 60 KB
  memcpy per call); thread-local pooled slots removed ~5.7B Ir, −18% total.
- **Slide windows, don't rebuild.** `9af76c99`: V-only prep rebuilt an
  8-row mid buffer per output row (8h widen calls) — widen each source row
  once and slide the window. −6.1M Ir.
- **Accumulate in registers, store once.** `62ab5783` lf_mask: per-row
  atomic RMW per edge → OR bits in a `[lvl][sidx]` register grid, one atomic
  update per cell. Counter-example below.
- **Batch DisjointMut guards per row-group.** One wide `strided_slice_mut`
  covering a block instead of a guard per row (CLAUDE.md notes; the residual
  tracker cost is the known `unchecked`-only win).
- **Early-out on census-measured dead states.** `93207ba7`: loopfilter
  returns early when `fm_mask` is all-false — a census measured ~85%/64% of
  calls in that state, so the whole filter/store tail was dead work.
- **Table-lookup beats per-element scans.** `562cf8de`: ORDER_TAIL static
  table replaced 8 bit-tests per index in `order_palette`; read_pal_indices
  self 76.2M→43.8M.

## 5. Archmage dispatch hygiene

- **`#[rite]` for same-tier helpers, `#[arcane]` only at real boundaries.**
  Inside a `#[rite]`/`#[arcane]`/`#[autoversion]` fn the context already has
  the target features — `incant!` (no token) or direct calls reach inner fns
  without trampolines. Tokens are for vanilla→feature trampolines only.
  `d0d458f2`: 13 NEON helpers + `fgy_row` were `#[arcane]` called from
  same-tier fns — dead vanilla trampolines; converted to `#[rite]`.
  **Warning**: an *unimported* `#[rite]` silently emits a featureless fn —
  it must be in scope (this produced an E0133 swarm once).
- **Dispatchers gate on `summon()`, never unwrap.** NEON is architecturally
  mandatory but the token is not — permutation tests disable it
  process-wide. `let Some(t) = Arm64::summon() else { return false }`.
- **Tier-upgrades need the real entry.** Callers doing `summon_avx512()`
  mid-function need the callee to stay `#[arcane]` (safe outer) — rename to
  `_v4` so `incant!` resolves, but don't demote to `#[rite]` (`d0d458f2`).
- `tools/archmage-audit` automates the lint set (tier-boundary, trampoline
  detection, loop-boundary, allow-pragma hygiene).

## 6. Cheap layout hints approximating PGO

`0c486ec6`: `likely()`/`unlikely()` on entropy + MC cold paths (msac refill
check, `eob_tok==2`, emu_edge/warp/obmc branches) approximates PGO's
cold-side layout for the default build. Small, free — measure before
believing it moved anything.

## 7. Bit-exactness-preserving freebies

Some fixes are *also* free or faster:

- **Saturating ops where the result clamps anyway.** `avg`'s `t1+t2` could
  wrap in i16, but the result lands in the clamped region — `adds_epi16`/
  `vqaddq_s16` is bit-exact AND one op (`c006cd4e`/`532cad61` family).
- **`max−min`/uabs instead of wrapping diff+abs.** The compound-mask kernels
  used `t − dst<<4` which wraps at 16bpc — widening or u16 abs-diff is both
  correct and the same cost.
- **u8-domain masks, i16 math.** `4795fc56` loopfilter: build `fm`/flat
  masks in the u8 domain (`subs_epu8` abs + `cmpeq`), keep the filter math
  in i16 — halves the lane pressure vs i32 cores.
- **Branchless three-way mask-select** (`4795fc56`, wd14/wd8/wd4 pick by
  `fm + flat8in + flat8out`) replaces cascading branches.

## 8. What did NOT pay (don't retry blindly)

- **Register-grid OR-accumulation on *short* runs.** `62ab5783` tried the
  same intra-edge trick on `mask_edges_chroma`: the ≤4-cell flush loop +
  grid init costs more than it saves at `ch4`≈1–4 (420 chroma edges are
  short). Reverted; accumulation only wins when rows/cells are long.
- **`#[arcane]`→`#[rite]` on its own is wall-clock neutral** (dispatch
  refactor commit) — worth it for symbol-table/inlining hygiene, not a perf
  lever by itself. The win is that `incant!`/inlining *enables* the other
  fixes.
- **Per-element atomic RMW → grid accumulation** won only where the cell
  count justified the flush loop (see above).

## 9. Build-level levers

- **`--profile release-thin`** for iteration (thin LTO, CGU=16 — measured
  decode parity with fat LTO on test22; ~2x faster rebuilds).
- **fat LTO + CGU=1** remains the shipped `release` profile.
- **PGO**: `bench_pgo.sh` — `llvm-profdata` profile → `-Cprofile-use`.
- **`target-cpu=native`**: give the compiler hints, but *don't* measure
  dispatch coverage with it — it can mask which `#[rite]` tiers actually
  engaged (zenav1-svt CLAUDE.md makes the same point).
- **`__bisect` diagnostic feature** env knobs to disable SIMD classes for
  triage (`a3f6e9f7`); `__simd_test` per-transform NEON-vs-scalar gate.

## 10. Gate after every kernel

The CLAUDE.md checklist is the short form. For codegen review specifically:
`cargo asm` the new inner and grep for `slice_index_fail`, `movzbl`,
`vpinsrb`, `memcpy` in the loop body — if they're there, one of §1–3 applies.
