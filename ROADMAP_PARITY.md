# rav1d-safe → libdav1d performance parity roadmap

## Current state (2026-05-23)

Safe-checked (`#![forbid(unsafe_code)]`) vs libdav1d hand-written ASM:

- IVF allintra 8bpc: **1.62x slower** (was 1.68x in Feb)
- 4K AVIF: **1.77–1.81x slower** (was 1.98x in Feb)
- 8K AVIF: **1.77–1.78x slower** (was 1.95x in Feb)

~9–10% of the gap has been closed by widening SIMD column-transform
coverage and batching per-row BorrowTracker guards.

## Profile breakdown (4K AVIF safe-checked, ~228 ms)

| % | Function | Notes |
|---|---|---|
| 38.6 | `decode_coefs` | msac entropy decode + coefficient parse, sequential |
| 8.8  | `loop_filter_4_8bpc` | mostly scalar; v-narrow case has SIMD path |
| 7.4  | `rav1d_recon_b_intra` | orchestration + per-block setup |
| 5.3  | `BorrowTracker::add_immut` | runtime borrow tracking (checked-only) |
| 3.0  | `__memset_avx512_unaligned_erms` | libc |
| 2.8  | `BorrowTracker::add_mut` | runtime borrow tracking |
| 2.6  | `read_pal_indices` | sequential entropy decode |
| ~1.8 | `DisjointMutGuard::drop` | tracker cleanup |

Total ~9.6% in BorrowTracker, ~9% in loopfilter, ~39% in entropy decode.

## Remaining tractable wins

These are concrete SIMD opportunities that follow the existing patterns.

| Task | Scope | Est. % saved | Difficulty |
|---|---|---|---|
| SIMD v-filter wd=6/8/16 (8bpc) | ~400 LOC new SIMD per width | 1–2 | Medium |
| SIMD h-filter via transpose load | ~600 LOC | 2–3 | High |
| AVX-512 col-transform variants (`dct16/32_1d_cols16`) | ~600 LOC | 1–2 | Medium |
| SIMD col for 4-wide transforms (4x4, 4x8, 4x16, etc.) | ~200 LOC | 0.5–1 | Easy |
| SIMD col for dct64 (`dct64_1d_cols8`) | ~400 LOC | 1 | Medium |
| SIMD msac adapt8/16 with `Sse2Token` | ~200 LOC | 0–3 | Easy but unclear ROI |

## Big architectural plays (multi-week, multi-month)

### 1. Type-state region-tracked PicBuf (eliminates BorrowTracker overhead)

The ~10% BorrowTracker overhead is structurally unavoidable with the
current runtime-checked design. Without resorting to `unsafe`, we can
eliminate it by proving disjointness at compile time:

```rust
// Picture buffer becomes a token-like type that's split into row partitions.
let mut pic = PicPartitionMut::<'_, BD>::from(...);
let (top, bottom) = pic.split_row_mut(64);  // disjoint &mut views

// Dispatch sites take PicSubViewMut<'_, BD> instead of PicOffset.
// No runtime BorrowTracker — Rust's borrow checker enforces disjointness.
process_block(top.sub_block(0, 0, 64, 64));
process_block(bottom.sub_block(0, 0, 64, 64));
```

For tile threading, `std::thread::scope` + `chunks_mut` splits picture
data into disjoint pieces that each thread owns.

This is a **multi-month refactor** touching every dispatch site (~150
call sites). Payoff: ~9% perf-percentage savings, plus eliminates an
entire crate of runtime safety machinery.

### 2. Restructure decode_coefs to enable SIMD msac

The msac state has serial dependency (each symbol's CDF lookup depends
on prior state). But the WORK around msac — coefficient masking, EOB
tracking, position lookup — has parallelism. With a redesigned
decode_coefs that batches symbol-decoding work and uses SIMD for the
non-msac parts, we could save 5–10% of profile.

Multi-week refactor of one of the most complex functions in the
codebase. Requires careful correctness verification against the
extensive MD5 parity test suite.

### 3. Complete dav1d ASM port to safe SIMD

The dav1d ASM has ~160k LOC. Our safe SIMD has ~86k LOC. The gap is:
~26k LOC of AVX-512 paths (mc, itx, cdef, filmgrain, ipred,
loopfilter) and ~52k LOC of SSE-only paths (mostly subsumed by AVX2).

Port the AVX-512 paths first; ~6–9 months of focused SIMD work for a
single developer. Each kernel ported saves 1–3% of profile.

### 4. AVX-512 token-based dispatch refactor

archmage 0.9 supports `X64V4Token` (AVX-512). Currently only
looprestoration uses it. Adding AVX-512 dispatch to the major hot
kernels (cdef, mc, itx, ipred, filmgrain) would give consistent
1.5–2x speedup over AVX2 on those kernels.

## Theoretical floor without `unsafe`

The BorrowTracker overhead (~9%) is the only structural floor that
cannot be eliminated without `unsafe`. Architectural play (1) above
removes it via compile-time proofs.

**Estimated reachable ratio with all of (1)-(4) implemented: 1.05–1.15x
of ASM.** True 1.0x parity would require the type-state refactor PLUS
all major kernels ported AND msac restructured — all without `unsafe`.

This is achievable but is a **6–18 month effort** for a single
developer. The current trajectory of 5–10% gap closure per session
suggests roughly 5–10 more focused sessions to reach the next big
milestone (1.4–1.5x).
