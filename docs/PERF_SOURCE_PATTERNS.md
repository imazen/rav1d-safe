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
  Measured (4K AVIF, znver4): **−7.0%** (45.5→42.3 ms/iter); 10-bit IVF
  **−10.4%**. Biggest single build lever.
- **BOLT post-link** (`llvm-bolt`, instrumentation mode — works without
  `perf_event_paranoid`): build with `-C link-arg=-Wl,--emit-relocs`,
  `llvm-bolt --instrument`, run workload, `-reorder-blocks=ext-tsp
  -reorder-functions=hfsort+ -split-functions -split-all-cold -split-eh`.
  Measured **−4.4%** alone (45.4→43.4 ms/iter), but **no stack gain over
  PGO** (PGO+BOLT ≈ PGO — rustc PGO already captures the layout win). Use
  BOLT when PGO isn't an option, or on release binaries post-hoc.
  `-indirect-call-promotion=all` promoted the 5 hot indirect callsites
  (99% of indirect calls) with no further wall gain.
- **MLGO**: not available — rustc's bundled LLVM 21 has no MLGO options
  compiled in (`--enable-ml-inliner` removed upstream). BOLT is its
  effective successor.
- **`target-cpu=native`**: give the compiler hints, but *don't* measure
  dispatch coverage with it — it can mask which `#[rite]` tiers actually
  engaged (zenav1-svt CLAUDE.md makes the same point).
- **`__bisect` diagnostic feature** env knobs to disable SIMD classes for
  triage (`a3f6e9f7`); `__simd_test` per-transform NEON-vs-scalar gate.

### Kernel-vs-kernel measurement tooling

- `scripts/perf/cg_resolve_asm.py <cg.out>` — resolves dav1d nasm
  `..@N`/bare-address callgrind records to `dav1d_*` symbols: the true
  per-kernel Ir table. On photo_4k (10 decodes): dav1d total asm
  ~1.57B Ir vs our ~3.3B in safe_simd inners+cores. Largest deltas:
  loopfilter family ~2.3B (h_sb_y 847M vs dav1d 110M), itx ~860M vs ~220M,
  ipred ~800M vs ~32M. `dav1d_msac_decode_symbol_adapt4_sse2` alone is
  712M — dav1d's biggest kernel too.
- `scripts/perf/mca_cmp.sh <our_bin> <our_sym> <asm_bin> <dav1d_sym>` —
  disassembles each function (spanning nasm `.sublabel`s via symbol
  ranges) and runs `llvm-mca -mcpu=znver4`. Per-pass cycles, znver4:

  | kernel | ours | dav1d |
  |---|---|---|
  | lpf_v_sb_y | 332 | 326 — **at parity** |
  | lpf_h_sb_y | 662 | 548 |
  | lpf_h_sb_uv | 313 | 155 |
  | itx 32x32 | 386 | 117 |
  | ipred_z2 | 1003 | 273 (IPC 1.26 vs 2.75) |
  | ipred_dc | 149 | 24 (IPC 0.82 vs 5.71) |

  Two structural lessons: (1) `lpf_v_sb_y` proves our codegen reaches
  parity *when the structure matches*; (2) the h-filter's 4× dynamic
  gap is path coverage, not throughput — dav1d's mask-driven dispatch
  skips more work per superblock than our per-row `test_all_zeros`
  early-outs.
- Whole-binary BOLT ICP stats: only **5 indirect callsites cover 99%**
  of all indirect calls — the dispatch surface is tiny.

## 10. Gate after every kernel

The CLAUDE.md checklist is the short form. For codegen review specifically:
`cargo asm` the new inner and grep for `slice_index_fail`, `movzbl`,
`vpinsrb`, `memcpy` in the loop body — if they're there, one of §1–3 applies.

## §11 — 2026-09-30: DC-prediction edge sums + a falsified gather-table

**Win — `edge_sum_u8_v3`/`edge_sum_u16_v3` (commit `b5ffc078`).** All twelve
`ipred_dc{,_top,_left}_{8,16}bpc` inners summed edge pixels one element at a
time (u8 indexing; u16 via per-element `from_ne_bytes`). Replaced with
`_mm256_sad_epu8`-vs-zero / `_mm256_madd_epi16`-by-ones reductions chunked
32/16/8/4 bytes. `ipred_dc_8bpc_inner` self Ir 392.0M → 117.7M (−70%) on 32
4K-intra decodes; total −0.62%.

**Dead end — `edge_pairs` gather table for `ipred_z2` (not committed).**
Idea: precompute `(edge[i], edge[i+1])` u16/u32 pairs once per call so the
per-lane left-edge gather becomes a single indexed load. Measured +86M Ir
net on the same stream — LLVM already compiles `u16::from_le_bytes(
fixed_array[i..i+2].try_into().unwrap())` to one unaligned load + bounds
check, so the table's 128-entry fill is pure overhead. Lesson: check what a
`try_into()` pair-load on a *fixed-size* array actually emits before
"optimizing" the gather; the cheap version is already there.

## §12 — 2026-10-01: itx row-pass fusion + 16bpc SIMD row passes

**Win — fused intermediate shift into the i16-packed row passes.** The square
DCT_DCT 8bpc paths staged row output through a scratch `[i32; N]`, then ran a
second elementwise pass applying `(v + rnd) >> shift` + col_clip into `tmp`.
`dct{8,16,32}_row_pass_i16_simd` now take `const POST_SHIFT: i32` + a scalar
`post_rnd` and apply the shift at their transpose-store stage, deleting the
scratch copy loop entirely (128 loads + 128 stores on 32x32, 32+32 on 16x16,
8+8 on 8x8). The post-shift `col_clip` is provably redundant for 8bpc — row
output is already clipped to i16 and >>2 keeps it inside [-8192, 8192] — so it
is dropped (the next stage's i16 pack saturates identically anyway).
4K intra IVF ×2 decodes: **37.620B → 37.515B Ir (−105M, −0.28%)**; stream
bit-exact vs dav1d.

**Win — 16bpc 16x16/32x32 DCT_DCT row passes.** Both ran a *scalar* `dct16_1d`
/`dct32_1d` call per row plus a per-element `Into<i32>` scratch gather. The
`impl_simd_row_rect_16bpc!` macro already generated `simd_row_dct16_16bpc_8rows`
(used by the mixed 16x16 16bpc transforms); added the missing
`simd_row_dct32_16bpc_8rows` instantiation and rewired both inners to process
8 rows per call through `dct{16,32}_1d_cols8` lanes. 10-bit IVF ×20:
**−17.7M Ir (−0.93%)** — small because that vector has few large blocks; the
structural fix matters more on dense high-res 10/12-bit content.

Pattern: when a pass ends in transpose+store, fold the consumer's first
elementwise stage into the store — don't materialize an intermediate buffer
just to rescale it.

## §13 — 2026-10-01: wall-clock sampling vs instruction counting (instrument cross-check)

`perf_event_paranoid=4` blocks the PMU outright on this box (samply/perf both
dead). Two samplers now live in `scripts/perf/` that work anyway:

- `RAV1D_PPROF=/tmp/x ./profile_ivf file N` — in-process SIGPROF sampling
  (pprof crate, libunwind backtraces). Writes `x.svg` + `x.collapsed`
  (inferno-compatible folded stacks). Full call-tree attribution.
- `ptrace_sample.py <dur> <tick_ms> <skip_ms> <out> <cmd...>` — fork+ATTACH
  leaf-PC sampler; the tracee is our child so yama scope-1 allows it. Works
  on ANY binary (used on the `--features asm` arm, whose vendored nasm
  objects carry full `dav1d_*` symbols — the stripped system libdav1d does
  not). Symbolization must translate `rip−map_start+map_offset` through the
  ELF LOAD phdrs (`p_vaddr ≠ p_offset` in rustc output ⇒ naive file-offset
  lookup attributes every PC to the fn ~4KB earlier — symptom: impossible
  `BitDepth16` hits on an 8bpc stream).

**Same stream (intra_4k.ivf), same decode loop, wall-clock leaf distribution:**

| family | safe wall% | asm-arm wall% | ms/f safe | ms/f asm | per-frame ratio |
|---|---|---|---|---|---|
| msac entropy | 37.4 | 48.9 | 16.5 | 11.7 | **1.4×** |
| lpf | 13.4 | 10.2 | 5.9 | 2.4 | 2.4× |
| ipred | 12.3 | 7.2 | 5.4 | 1.7 | 3.2× |
| itx | 8.1 | 1.0 | 3.6 | 0.2 | **14.7×** |
| tracker/guards | 11.6 | 15.3 | 5.1 | 3.7 | 1.4× |
| spine/other | 17.2 | 17.3 | 7.6 | 4.1 | 1.8× |

(asm-arm 23.8ms/f vs safe 44.1ms/f; leaf-only attribution, ~6k+3.5k samples)

**What sampling changed vs the Ir inventory:**

1. **msac is the #1 wall item on BOTH sides** — dav1d's own
   `msac_decode_symbol_adapt4_sse2` alone eats 21.8% of the asm arm's wall.
   Entropy decode is a shared serial-dependency bottleneck; our branchless
   scalar adapt4 (inlined into `decode_coefs_class_v3`, ~19% leaf) is NOT the
   special gap Ir suggested — dav1d spends the same share in asm.
2. **itx is the worst RELATIVE kernel gap** (14.7× per frame) even though its
   Ir gap (+2.1B) ranked below lpf — dav1d's itx is nearly free (0.2ms/f)
   while ours is 3.6ms/f. Leaf-level: our `itxfm`+transform inners vs
   dav1d's fused register-resident butterflies.
3. **Tracker/guard tax ≈ 11.6% wall** — atomics and bounds machinery that Ir
   under-counts (each atomic op is 1 instruction but ~20-cycle latency). It's
   ~15% of the asm arm too (shared spine) — pure structural overhead dav1d
   lacks entirely.
4. Ir family *ratios* were directionally right for lpf/ipred (13×/3× Ir →
   2.4×/3.2× wall) but Ir massively over-weighted SIMD kernels' share of the
   total: SIMD retires at high IPC, so instruction-count gaps compress ~3-5×
   in wall time, while scalar serial code (msac, tracker) compresses the
   other way.

**Dispatch note:** dav1d picks `avx512icl` kernels on this Zen4; our safe
side runs `avx2_inner` for most of lpf/ipred — `CpuFlags::AVX512ICL` *is*
granted (full ICL set present incl. gfni/vaes/vpclmulqdq); it's a coverage
gap (no v4 inners), not a detection bug.

## §14 — 2026-10-05: dispatch-level hull slicing for with_block_mut kernels (itx + ipred)

**Pattern:** kernels that receive `(dst, base, stride)` and address rows as
`dst[base + y*stride + x]` get ONE upfront slice to the block hull
`(h-1)*stride + w` at the dispatch boundary. Inside, every per-row/per-chunk
access becomes `y*stride + x <= (h-1)*stride + w` — provable via umax — so
LLVM elides the checked `try_into`/slice-create per row instead of once per
SIMD store on a runtime-offset index into the whole picture tail.

**Applied (safe build, `#![forbid(unsafe_code)]` intact):**

- `itxfm_dispatch_{8,16}bpc` (`safe_simd/itx/part10_dispatch.rs`): hoisted
  `TxfmSize::to_wh` to function scope; the `arcane!`/`dc_only` call sites now
  pass `&mut dst[base..base + hull]` (u16 elements for 16bpc,
  `hull = (h-1)*(byte_stride/2) + w`). Post-change profile: no `index_mut`/
  `try_into` leaves under any `__arcane_inv_txfm_*` inner.
- `intra_pred_dispatch` (`safe_simd/ipred.rs`): same rebase inside the
  `with_block_mut` closure — `(bytes, base, stride)` →
  `(&mut dst[base..base+hull], 0, stride)` for positive strides. For the
  negative-stride arm the hull already starts at the last row
  (`base = (h-1)*|stride|`), so the transform is a no-op there; kernels keep
  signed `dst_base + y*stride` addressing and stay correct.
- `ipred` v4x inners (smooth/smooth_v/smooth_h/z1/z2): per-row `dst` narrowed
  to `row[..width]`, `topleft` edge slices narrowed once, z1/z2's
  tmp-store+`copy_from_slice` tail replaced by direct `storeu_256!` for
  full-width chunks (tmp retained only for partial tails), `ebuf`/`tbuf`
  edge fills switched to `copy_from_slice`. `index_mut<u8>` leaf stacks
  under the smooth/z2 inners disappeared from the post-change profile.

**Bug found while hull-checking the itx dispatch:** the arcane itx path had
no negative-stride gate — on a negative-stride picture the unsigned
`stride_u` walk goes upward from a `base` that is `(h-1)*|stride|`, i.e.
past the hull end (guaranteed panic for h>1, slice-check or kernel-side).
Both `itxfm_dispatch_{8,16}bpc` now `return false` on `stride_i < 0` so the
call falls back to `itxfm_add_scalar_fallback`, which uses signed
`pxstride`/`wrapping_add_signed` correctly. 16bpc gained a `stride_i` param
to see the sign. (Decoder-owned pictures are always positive-stride; the
path is only reachable through c-ffi consumer buffers, but panic→correct
decode is a strict improvement and protects the new `base + hull` slice.)

**Measured (photo-4k-min.ivf, 3840x2160 intra, t=1):**
safe 94.2 → ~91.5 ms/frame (≈ −2.7ms, mostly the ipred in-kernel row
narrowing + direct stores; the itx/ipred dispatch hull is sub-noise on this
stream but confirmed-check-free and defensive). asm arm ~65.4 ms/frame ⇒
**~1.40×** (was ~1.44×).

**Verification:** mixed-parity dispatch tests 23/23 + itx suite 44/44
(incl. the "write outside block" sentinel over stride>w + nonzero offset),
ipred v4x parity 7/7 + z2 bounds crash tests, cross-tier MD5 identical at
scalar/v2/v3/v4/native, `decode_permutations` 19/19, `gen_cover`,
aarch64 + wasm32 `cargo check`.

### 2026-09 — MT tracker traffic: banded rectangle records

Context: the DisjointMut `add` path is already latency-tuned (documented
above); the remaining lever is FEWER REGISTRATIONS. `__probe_sites` on
`photo-4k-t8` at t=4 put `compact_read_per_row`'s per-row immut borrows at
**54.6% of all registrations** (1.68M/frame, mean extent 14 B — the
loopfilter's MT compact read does `h` row-guards per window, ~117k windows
per frame).

**Change (`include/dav1d/picture.rs`):** `compact_read_per_row`,
`compact_write_back_per_row`, `for_rows` and `for_rows_mut` now register
`DisjointMut::index_rect{,_mut}` records — ONE tracker record per row-band
instead of one per row. Bands are 8 rows so a tall window's *hull* stays
under `MAX_SHARDS_PER_BORROW` blocks whenever the adaptive block rule has
armed (a block then holds >= ROWS_PER_BLOCK_MIN = 4 picture rows); `h <= 8`
collapses to the single-rect case; `None` still falls back to per-row.
`compact_write_back_per_row_diff` is deliberately untouched — mutably
guarding unmodified tap rows is exactly what zenavif#30 removed.

Registrations/frame: 3.08M → 1.62M (−47%). photo-4k-t8 ms/frame:
t=1 92.9 → 90.5, t=4 38.5 → 38.2, **t=8 27.1 → 24.9 (−8.1%)**.
asm at t=8 is ~14.1 — the tracked/untracked tax at t=8 went 1.49× → 1.37×.

Measured dead ends alongside: per-call `summon()` is a relaxed atomic load
+ flag test (~a few cycles) — caching tokens is not worth plumbing. The
`*_diff` write-back must NOT take a rect record (see above).

**Verification:** `tile_threading_overlap`, `reproduce_overlap`,
`decode_concurrent_md5`, `mt_stress`, `flush_drains`, `cancellation`,
`worker_panic_recovery`, `filmgrain_threads` 24/24 (incl. the ignored
reproducers under `--run-ignored all`); row-guard policy tests incl.
`for_rows_mut_never_reserves_an_inter_row_gap_when_tile_threading_is_on`;
photo/map-4k-t8 bit-exact at t=1 and t=8 vs dav1d MD5; 140-frame clip
identical at t1 / t8d1 / t8d8; `gen_cover`, aarch64 + wasm32 checks.

Note: `probe_sites_ivf` (`--features __probe_sites`/`__probe_wide`) added —
the IVF sibling of `probe_tracker`, for the stills corpus.

## §15 — 2026-11-05: element-granularity `f.a` (one ctx guard per block, not per field)

`BlockContext` had 16 fields each wrapped in `DisjointMut`, so every
above-context read/write registered a whole shard trip — ~800k of the
1.62M records/frame left after §14, including pathological single-byte
reads (`filter[0].index(x).get()`).

**Change:** fields are now plain arrays; `f.a` itself became
`DisjointMut<Vec<BlockContext>>`. `decode_b` holds one
`index_mut(t.a..t.a+1)` guard for the block's whole body and threads
`ta: &mut BlockContext` into the recon/read helpers; `decode_sb`'s
partition-ctx read/writeback use short-scoped element borrows;
`lf_apply`'s tile-row-boundary fix reads an EXACTLY-sb128w element
window — the earlier `index(row-1..)` unbounded tail collided with a
live `decode_b` element guard (the field-level trackers never collided
because disjoint byte ranges inside an element coexisted; element
granularity makes over-wide ranges fatal, not just wasteful).
`case_set_al!`'s above arm is now plain `&mut` slices like the left arm.
`BlockContext` derives `Copy` + zerocopy (`FromBytes`/`IntoBytes`/
`KnownLayout`/`Immutable`) for `Vec<BlockContext>: AsMutPtr`; `Align8`
wrappers dropped.

Safety argument is identical to before, hoisted: `t.a` is the
worker-owned tile-column slot, elements are disjoint across concurrent
workers, and the tracker panics loudly on any mistake (observed during
development — the `lf_apply` tail over-borrow). Frame-MT pass-2 uses
`off_2pass` elements, also disjoint.

Registrations/frame: 1.62M → 650k (−60%); `f.a` is now 3 sites /
~131k acquires. photo-4k-t8 ms/frame (release): t=1 94.4 → 82.7 (−12%),
t=4 37.9 → 31.5 (−17%), t=8 neutral inside its ±15% run-to-run band.

Iteration loop added: `scripts/quick_gate.sh` (~16s warm) — release-thin
build, 1-frame t1+t8 md5s vs dav1d sidecars, tier identity, frame-MT
delay1-vs-8, row-guard lib tests, 5-iter probe.

Next lever visible in the post-change census: the remaining ~131k `f.a`
acquires are per-partition-NODE (`decode_sb` ctx read at :3562, `decode_b`
guard at :1225, node writeback at :3864). `t.a` is constant within a
`decode_sb` tree, so threading `ta` down the recursion could collapse
them to ~#tiles·sbrows — needs guard-lifetime care around the
`index_mut`-vs-`index` sequencing.
