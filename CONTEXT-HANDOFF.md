# Handoff: Path to a Fully Safe rav1d-safe Crate

**Date:** 2026-02-05
**Branch:** `feat/fully-safe-intrinsics`
**Last commit:** `fe5b8a5` (docs: add FIXED.md)

## Current State

rav1d-safe is a fork of rav1d (Rust AV1 decoder) that replaces 160k lines of hand-written x86/ARM assembly with safe Rust SIMD intrinsics. All DSP modules are ported. Decode parity is verified — safe-simd produces pixel-identical output to the asm path.

**What's already safe:** 29 modules compile with `#![deny(unsafe_code)]` when `asm` and `c-ffi` are disabled. All DSP logic (cdef, filmgrain, ipred, itx, loopfilter, looprestoration, mc, pal, recon) is fully safe.

**What still uses unsafe:** 17 modules retain `#[allow(unsafe_code)]`. The remaining unsafe falls into five categories, totaling ~2,500 occurrences.

## Remaining Unsafe: Five Categories

### 1. SIMD Intrinsics (~2,311 occurrences) — `src/safe_simd/`

Every `_mm256_*` / `vld1q_*` intrinsic call is `unsafe` because Rust's `#[target_feature]` functions require unsafe callers. The operations are semantically safe (correct types, correct alignment, runtime feature detection) but syntactically unsafe.

**Path to safe:** Replace raw `core::arch` intrinsics with `archmage` token-gated calls. The crate already depends on `archmage 0.4`. The pattern:

```rust
// BEFORE: unsafe intrinsics
#[target_feature(enable = "avx2")]
unsafe fn add_pixels(a: __m256i, b: __m256i) -> __m256i {
    unsafe { _mm256_add_epi16(a, b) }
}

// AFTER: archmage token-gated (safe)
#[arcane]
fn add_pixels<T: X64V3Token>(t: T, a: m256i, b: m256i) -> m256i {
    t.add_epi16(a, b)
}
```

**Effort:** Large but mechanical. Each safe_simd file needs rewriting to use archmage tokens instead of raw intrinsics. The dispatch functions already do runtime CPU detection — they'd summon a token once and pass it down.

**Files (by size, descending):**
- `itx.rs` (786 unsafe) — largest, most repetitive (160 transform variants)
- `itx_arm.rs` (657)
- `mc.rs` (237)
- `ipred.rs` (124)
- `ipred_arm.rs` (80)
- `loopfilter.rs` (66)
- `looprestoration.rs` (65)
- `cdef.rs` (55)
- `mc_arm.rs` (45)
- `looprestoration_arm.rs` (42)
- `cdef_arm.rs` (39)
- `filmgrain.rs` (38)
- `partial_simd.rs` (33)
- Remaining files: <20 each

### 2. FFI Dispatch Wrappers (~100 occurrences) — `src/*.rs`

The `wrap_fn_ptr!` macro generates `unsafe extern "C" fn` wrappers for function pointer dispatch. When `asm` is disabled, these wrappers are **dead code** — the dispatch uses direct Rust calls via `*_direct` functions. But `wrap_fn_ptr!` still generates the types, and some modules (refmvs, msac) still use `unsafe extern "C" fn` signatures for the safe-simd path.

**Path to safe:** When `asm` is off, `wrap_fn_ptr!` already generates a zero-sized `Fn(())` that discards the pointer. The `call` methods already use `cfg_if!` to call `*_direct` functions. What remains:

- **msac:** Has inline AVX2/NEON SIMD in `src/msac.rs` (not in safe_simd/). The `unsafe extern "C" fn` wrappers around `symbol_adapt16` are unnecessary on the non-asm path. Refactor to call the SIMD functions directly without extern "C" ABI.
- **refmvs:** Still uses `unsafe extern "C" fn` for `splat_mv`, `save_tmvs`, `load_tmvs`. The `call` method on the non-asm path still goes through `unsafe { self.get()(...) }` for splat_mv. Needs the same `*_direct` pattern as other modules.

**Effort:** Small. Mostly cfg-gating existing unsafe behind `feature = "asm"`.

### 3. Core Unsafe Abstractions (~80 occurrences) — 10 modules

These are the trusted computing base — types that encapsulate unsafe behind safe public APIs:

| Module | What it provides | Can it be eliminated? |
|--------|-----------------|----------------------|
| `align` | `AlignedVec`, `Align*` types | Partially — use `aligned-vec` crate or `alloc::Layout` |
| `assume` | `debug_assert + assume` optimization hint | No — `unreachable_unchecked` is inherently unsafe |
| `c_arc` | `CArc<T>` — Arc with C-compatible free | Could wrap std `Arc` fully if C allocator support dropped |
| `c_box` | `CBox<T>` — Box with C-compatible free | Same as c_arc |
| `disjoint_mut` | Safe aliased mutable access to disjoint regions | No — this is a fundamental unsafe abstraction |
| `ffi_safe` | `FFISafe` pointer wrapper | Only needed for `asm`/`c-ffi` dispatch; dead without those features |
| `internal` | `Send/Sync` impls for `Rav1dContext` | Could be safe if context fields were made thread-safe |
| `log` | Logger callback type | Could use safe fn pointer or trait object |
| `send_sync_non_null` | `SendSyncNonNull` | Only needed for C FFI interop |
| `picture` | Picture allocator callbacks | Could use safe Rust allocator API |

**Path to safe:** Some of these can be eliminated when `c-ffi` is off. `ffi_safe`, `send_sync_non_null`, `c_arc`, `c_box` exist primarily for C interop. Without `c-ffi`, pictures could use plain `Arc<Vec<u8>>` instead of `CArc<CBox<[u8]>>`. This is a larger refactor but would remove the entire C memory management layer.

**Effort:** Medium. Requires redesigning the picture/data ownership model for the pure-Rust path.

### 4. MaybeUninit (~20 occurrences) — `lf_mask.rs`, `refmvs.rs`

`assume_init()` calls on arrays that are fully initialized but can't be proven so by the type system.

**Path to safe:** Replace with zero-initialization (`[0u8; N]`) or use `array::from_fn`. Performance impact is negligible — these are small arrays initialized once per frame/block.

**Effort:** Small. Direct replacement.

### 5. Picture/Data Memory Management (~15 occurrences) — `src/picture.rs`, `include/dav1d/picture.rs`

Picture allocation uses `unsafe extern "C" fn` callbacks (matching dav1d's allocator API), raw pointer arithmetic for stride handling, and `into_arc`/`from_raw` for reference counting.

**Path to safe:** On the pure-Rust path (no `c-ffi`), replace the C allocator callbacks with a safe Rust trait:

```rust
trait PictureAllocator {
    fn alloc(&self, width: usize, height: usize, bpc: u8) -> Box<[u8]>;
}
```

The stride calculations and `Rav1dPictureDataComponentInner` raw pointer logic could be replaced with safe slice indexing if the picture data used a flat `Vec<u8>` with computed offsets instead of raw pointers.

**Effort:** Medium. The picture data abstraction is deeply embedded.

## Recommended Order of Work

### Phase 1: Low-hanging fruit (days)

1. **MaybeUninit elimination** in `lf_mask.rs` and `refmvs.rs` — replace `assume_init()` with zero-init or `array::from_fn`.

2. **Cfg-gate remaining FFI wrappers** — in `msac.rs` and `refmvs.rs`, gate `unsafe extern "C" fn` behind `feature = "asm"` and use direct calls on the safe path.

3. **Remove dead `#[allow(unsafe_code)]`** from `src/lib.rs` on the non-asm/non-c-ffi path.

4. **Gate `ffi_safe`, `send_sync_non_null`** behind `c-ffi` — these modules are unused without C interop.

### Phase 2: SIMD migration to archmage (weeks)

5. **Port `safe_simd/` to archmage tokens** — start with smallest modules (pal, refmvs, filmgrain) to establish the pattern, then tackle the large ones (itx, mc). This is ~2,300 intrinsic calls to convert but is highly mechanical.

6. **Port inline msac SIMD to archmage** — the AVX2/NEON symbol_adapt16 implementations in `src/msac.rs`.

### Phase 3: Core abstractions (weeks)

7. **Safe picture allocation** — replace C allocator callbacks with safe Rust trait when `c-ffi` is off. Redesign `Rav1dPictureDataComponentInner` to use safe slice indexing.

8. **Simplify memory ownership** — when `c-ffi` is off, replace `CArc`/`CBox` with standard `Arc`/`Box`. The C-compatible free callbacks become unnecessary.

9. **Safe `disjoint_mut`** — this is the hardest piece. The `DisjointMut` abstraction allows mutable access to non-overlapping regions of a single allocation (critical for video decoding where multiple planes share a buffer). Options:
   - Keep as a well-audited unsafe core (acceptable — it's a small, testable abstraction)
   - Use `Cell`/`RefCell` with runtime checks (performance cost)
   - Restructure to use separate allocations per plane (memory overhead)

### Phase 4: Final cleanup

10. **Audit `assume.rs`** — the `assume()` hint uses `unreachable_unchecked`. Consider whether the optimization benefit justifies keeping this unsafe, or if `debug_assert` alone is sufficient.

11. **Thread safety** — make `Rav1dContext` fields properly `Send + Sync` (replace raw shared state with `Arc<Mutex<_>>` or atomics) to eliminate the unsafe `Send/Sync` impls in `internal.rs`.

12. **Remove `#![deny(unsafe_code)]` conditional** — once all unsafe is eliminated (or irreducibly small), make the deny unconditional.

## What "Fully Safe" Means

After all phases, the crate would have:

- **Zero** `unsafe` in DSP logic, dispatch, or decode pipeline
- **Zero** `unsafe` in SIMD (via archmage token-gated safe intrinsics)
- **Minimal irreducible unsafe** in:
  - `disjoint_mut` (aliasing abstraction — ~50 lines, well-audited)
  - `assume` (optimization hint — 1 function, optional)
  - `internal.rs` Send/Sync impls (2 impls, removable with threading refactor)
- **Feature-gated unsafe** in `c-ffi` and `asm` paths only

The irreducible core is roughly 60 lines of unsafe across 3 modules, all behind safe public APIs.

## Infrastructure In Place

- **CI:** `.github/workflows/ci.yml` — builds safe-simd, asm, c-ffi across Linux/macOS/Windows + aarch64 cross-check, clippy, fmt
- **Parity test:** `src/decode_test.rs` — decode IVF files, compare pixel hashes between asm and safe-simd. Env vars: `RAV1D_TEST_IVF`, `RAV1D_TEST_EXPECTED_HASH`
- **Justfile:** `just build-safe`, `just test`, `just clippy`, `just ci`, `just test-decode <file>`, `just test-decode-asm <file>`
- **Bug tracker:** `FIXED.md` — documents bugs found and fixed with reproduction steps

## Key Files

| File | Purpose |
|------|---------|
| `lib.rs` | Crate root, module declarations, safety attributes |
| `src/wrap_fn_ptr.rs` | Function pointer dispatch abstraction (zero-sized when asm off) |
| `src/safe_simd/*.rs` | All safe SIMD implementations (~42k lines) |
| `src/msac.rs` | Inline AVX2/NEON for entropy coding |
| `src/decode_test.rs` | Parity test harness |
| `.github/workflows/ci.yml` | CI configuration |
| `justfile` | Build/test commands |
| `FIXED.md` | Bug documentation |
