# Proving memory safety of the `asm` feature's hand-written assembly

Status: DESIGN (2026-07-19). Nothing here is implemented yet; this is the
plan of record for making `asm` / `partial_asm` builds carry a machine-checked
safety story instead of "trust dav1d's asm".

## 1. Problem statement

rav1d-safe's default build is `forbid(unsafe_code)` end to end. The `asm`
feature (and `partial_asm` = msac + loopfilter only) re-enables dav1d's
hand-written assembly kernels — ~160k+ lines of NEON (`.S`, gas) and
AVX2/AVX-512 (`.asm`, nasm) — called through `extern "C"` fn pointers in
per-ISA dispatch tables. Each call site lowers safe types
(`Rav1dPictureDataComponentOffset`, `DisjointMut` guards) to
`(ptr, stride, w, h, edges, …)` raw arguments. The safety contract at every
call is:

> Given arguments satisfying the kernel's (implicit, undocumented)
> precondition, the asm reads and writes only within the picture/scratch
> allocations implied by those arguments, restores sp and callee-saved
> registers, and jumps only within its own code.

Today that contract is enforced by nothing but dav1d's track record and
checkasm. The goal: make it *checked* — per kernel, per ISA level, in CI —
with a maintenance burden that survives dav1d upstream asm updates.

## 2. Prior art: CLAMS/BUMS (scaspin/memory-safe-assembly)

The closest existing work is CLAMS ("Checking Linked Assembly for Memory
Safety"; engine crate `bums`), Caspin/Pimpalkhare/Levy, PLOS '25:
"From Rust Till Run: Extending Memory Safety From Rust to Cryptographic
Assembly" — https://amitlevy.com/papers/2025-plos-clams.pdf, repo
https://github.com/scaspin/memory-safe-assembly.

How it works:

- A proc macro `#[bums_macros::check_mem_safe("file.S", args…, [preconds])]`
  wraps a safe Rust signature around an asm symbol. Memory regions are
  derived from the Rust types (`&mut [u8]` → writable region of symbolic
  length `param_len`; `&[u32; 8]` → readable 32-byte region).
- At **build time** (inside macro expansion) a from-scratch symbolic
  execution engine (own aarch64 model, Z3 backend) explores every path of
  the asm and proves: every load hits accessible memory, every store hits
  mutable memory, sp/callee-saved registers are restored, jumps stay
  in-program. Loops with symbolic trip counts are handled by constant-step
  loop acceleration (inductive proof over the induction variable).
- Preconditions double as runtime `assert!`s at the call site, so the proof
  is "safe *given* preconditions" and the preconditions are enforced by
  panic, not UB.

Why it is not directly usable here — the honest read of its own
`rav1d-playground/`:

- **Coverage**: 4 of ~60 kernel families verified (`ipred_reverse` 8/16bpc,
  `z1/z2_upsample_edge`), all tiny edge-prep helpers; two required a
  **hand-edited copy of the asm** (`edited-ipred.S`). Attempts at `cfl_ac`,
  `pal_pred`, `w_avg`, `z1_filter_edge` are commented out.
- **ISA model**: ~50 mnemonics, thin NEON coverage (no `tbl/ext/zip/umull/
  sqrshrun/ld2-ld4`…), **no x86 decoder at all**. dav1d's `mc.S` alone is
  132 KB of exactly that vocabulary. A hand-written ISA model is also a
  soundness liability (cf. DJB's symexemu argument,
  https://cr.yp.to/papers/symexemu-20260201.pdf).
- **Region model**: linear buffers with symbolic length. Video kernels need
  2-D strided regions where stride is a signed, symbolic runtime value —
  the paper's own future-work item.
- **Macro asm**: dav1d asm is macro-templated; the playground pipeline
  compiles it and **disassembles the objects back to flat text** to feed the
  engine, and that pipeline fails on several files (0-byte outputs for
  `cdef_tmpl.S`, `mc_dotprod.S`).
- **Verification in macro expansion** (release builds hard-error, debug
  builds warn) does not scale to thousands of kernel×shape checks and makes
  rustc runtime the verifier budget.

What CLAMS gets *right*, and what we keep: contracts co-located with the
call site, derived from Rust types; preconditions that double as executable
documentation and debug asserts; per-kernel granularity.

## 3. The insight that makes video tractable: finite shape lattices

Crypto verification needs symbolic loop bounds ("any number of 64-byte
blocks"). Video kernels are the opposite: **every shape parameter ranges
over a tiny finite set** — `w, h ∈ {4, 8, 16, 32, 64, 128}` (per-kernel
subsets), `mx, my ∈ 0..16`, `edges` is a small flag set, bitdepth ∈
{8, 10, 12}. Only the buffer base pointers and **stride** are genuinely
unbounded. So instead of one hard inductive proof per kernel, verification
decomposes into thousands of small, embarrassingly-parallel checks with all
shapes concrete and only pointers/strides symbolic — under which most
data-dependent branches collapse and the paths are near-straight-line.

## 4. Plan of record: three layers

Layers 1–2 are engineering (buildable now, no research risk) and produce
the artifacts layer 3 consumes. Layer 3 is the eventual full proof.

### Layer 1 — `asm-contracts`: one declarative contract per kernel

A small crate (or module under `src/asm_contract.rs`) where every asm entry
point in the dispatch tables gets a machine-readable region contract,
co-located with its `wrap_fn_ptr!` site:

```rust
asm_contract! {
    fn put_8tap_regular_8bpc_neon;
    reads:  region(src, stride = src_stride, rows = h + 7, cols = w + 7,
                   elem = u8, align = 1, may_read_padding = true),
    writes: region(dst, stride = dst_stride, rows = h, cols = w,
                   elem = u8, align = 16),
    shapes: w in [2, 4, 8, 16, 32, 64, 128], h in 2..=128,
            mx in 0..16, my in 0..16,
    clobbers: callee_saved_restored, stack_bounded(512),
}
```

Key contract subtleties the DSL must express (all real dav1d behaviors):

- **Padding reads**: some kernels intentionally read within dav1d's picture
  padding (alignment + edge padding). The true region is "allocation
  bounds", not "tight w×h" — `may_read_padding` widens the read region by
  the documented padding geometry, and the *allocator side* of the contract
  (layer 3 of the Rust code: `Rav1dPictureDataComponent` construction)
  guarantees that padding exists.
- **Negative strides** (flipped pictures) where legal.
- **Scratch buffers** (`itx` intermediate arrays, cdef tmp) with exact
  sizes.
- The trailing `FFISafe<…>` shadow params are declared inert (asm never
  dereferences them).

Deliverable value even before any verifier exists: the contracts are the
first precise, greppable documentation of what each kernel is allowed to
touch, and they generate debug-build `debug_assert!`s at every dispatch
call site (CLAMS-style precondition enforcement).

### Layer 2 — enumerative bounds-shadowing validation (the CI gate)

A harness (extending the checkasm pattern rav1d already ships) that
*empirically falsifies* contracts, deterministically, for every enumerated
shape:

1. **Guard-page placement**: allocate each declared region so its contract
   boundary lands exactly on a `PROT_NONE` page edge — one pass with the
   end at the page edge, one with the start. Any read/write even 1 byte
   outside the declared region faults immediately. Sweep stride regimes:
   minimal legal, huge, negative-where-legal, and the
   "row-end-on-page-edge" placement that catches per-row overreads that a
   tail guard page misses.
2. **Canary tiling** for the interior structure (unwritten gaps between
   rows of the write region must remain poison).
3. **Differential output check** vs the safe-Rust implementation of the
   same kernel — rav1d-safe uniquely *has* a complete safe reference for
   every kernel (`src/safe_simd/`), so bit-exact output equality is
   checkable at the same time (correctness ⊃ "the writes that matter landed
   in the right place").
4. **Clobber checking**: keep checkasm's callee-saved/stack-canary checks.
5. **Full shape-lattice enumeration** from the layer-1 contract's `shapes`
   clause — not random sampling. Content bytes are randomized (content
   never affects addressing in these kernels except via documented paths
   like palette indices; where it does, enumerate that too).

This is not a proof, but it is *deterministic detection of any spatial
violation reachable under any enumerated shape* — for NEON, AVX2 and
AVX-512 alike, today, with zero per-kernel annotation beyond the layer-1
contract it shares. It survives dav1d asm updates untouched (re-run CI).
Ship `asm` builds gated on this.

### Layer 3 — machine-checked proofs, two halves

**(a) Caller side, provable now with Kani/Verus**: for each dispatch call
site, prove that the safe-Rust argument lowering is *total* into the
contract — "for all inputs accepted by the safe wrapper, the derived
(ptr, stride, w, h, …) tuple satisfies the layer-1 contract's precondition
and the referenced allocation really contains the declared region." This is
pure safe-Rust verification (Kani bounded model checking fits; the
picture-allocation padding invariant becomes a proven lemma, not a comment).
It also covers hazards asm verification never sees: the `DisjointMut`
aliasing discipline and the `FFISafe` shadow-arg conventions.

**(b) Asm side, the research track**: per-kernel symbolic execution with
shapes concrete (from the lattice) and pointers/strides symbolic, proving
no access outside the contract regions + callee-saved restoration for
*every* shape cell. Run in CI as a cached matrix (key: asm-object hash ×
contract hash), never inside rustc. Backend options, in preference order:

  1. an existing maintained lifter (angr/VEX or BINSEC) driven by a small
     contract-compiler that emits the region constraints — sidesteps
     writing an ISA model;
  2. a Sail-derived executor (Islaris-style authoritative ARM semantics)
     if lifter SIMD coverage falls short;
  3. collaborating with the CLAMS authors — extending BUMS with strided
     2-D regions + shape-enumeration mode + dense NEON coverage is
     literally their stated future work, and rav1d is their own playground
     target.

When (a) and (b) both hold for a kernel, the composed statement is a real
proof: *Rust never calls outside the contract; the asm never strays inside
it.* Kernels verified under (b) get a `verified` badge in the contract;
`asm` builds could eventually offer a `verified-asm-only` mode that routes
unverified kernels to the safe SIMD fallback.

## 5. What this proves — and does not

Proved (eventually, per kernel): spatial memory safety within declared
regions under all enumerated shapes, control-flow containment, ABI
(callee-saved/sp) discipline. Enforced at runtime in debug builds:
preconditions. NOT proved: functional correctness (layer 2's differential
check covers it empirically), termination, constant-time, and concurrent
aliasing (that's the `DisjointMut` discipline — layer 3a's caller-side
proofs are the only layer that touches it).

## 6. Sequencing

1. Contract DSL + contracts for the `partial_asm` surface first (msac +
   loopfilter — the small, highest-value set), generating debug asserts.
2. Layer-2 harness over those contracts; wire into CI for `asm` +
   `partial_asm` feature builds.
3. Contracts for the full `asm` surface, family by family (mc → itx →
   ipred → cdef → lr → filmgrain → pal/refmvs), layer-2 gated as they land.
4. Layer-3a Kani proofs for the dispatch call sites of covered families.
5. Layer-3b backend spike: one mc kernel through angr with concrete shapes;
   evaluate solver time; then decide build-out vs CLAMS collaboration.

## 7. References

- CLAMS paper: https://dl.acm.org/doi/10.1145/3764860.3768333 (PDF:
  https://amitlevy.com/papers/2025-plos-clams.pdf)
- CLAMS/BUMS repo + rav1d playground:
  https://github.com/scaspin/memory-safe-assembly
- DJB, symbolic execution of emulators instead of hand ISA models:
  https://cr.yp.to/papers/symexemu-20260201.pdf
- Islaris (authoritative ISA semantics proofs):
  https://dl.acm.org/doi/10.1145/3519939.3523434
- Jasmin (verified-asm rewrite alternative, rejected: loses dav1d upstream
  sync): https://github.com/jasmin-lang/jasmin
- Kani: https://model-checking.github.io/kani · Verus:
  https://github.com/verus-lang/verus
- dav1d checkasm (the existing per-kernel differential + clobber harness
  this plan extends).
