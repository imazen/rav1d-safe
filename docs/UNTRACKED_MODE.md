# `untracked`: race-tolerant fast mode

An opt-in Cargo feature (`--features untracked`). It is **not** in `default`.

## What it changes

| | default | `untracked` |
|---|---|---|
| `DisjointMut` overlap tracking | on (overlap panics) | **off** |
| slice bounds checks | on | **on** (unchanged) |
| t>1 pixel access | compact copy-in + diff write-back | **zero-copy in place** |
| `forbid(unsafe_code)` in the main crate | yes | yes |

There is deliberately **no** bounds-unchecked variant of this mode. Measured on
`intra_4k`, dropping slice bounds checks on top of tracker removal changes
nothing (38.32 vs 38.32 ms/frame), so none is offered.

## What overlap can and cannot do

Overlapping `&mut` / racing access is undefined behaviour in the Rust memory
model; this mode accepts that. What bounds the *consequence*:

1. **Out-of-bounds is not reachable.** `get_mut` bounds-checks against the
   owner's live length, which is not part of the contested bytes. Pinned by
   `crates/rav1d-disjoint-mut/tests/untracked_mode.rs`.
2. **Every element is `PlainData`** (`Copy + zerocopy::FromBytes`): all bit
   patterns valid, no pointers, no niches. This is a compile-time bound on
   `AsMutPtr::Target`, in every build, so a racing or aliased read yields *some
   valid value* -- never an invalid enum, a forged index or a dangling pointer.
   (Upstream rav1d relies on the same argument but "checks it manually".)
   Enum-valued context arrays therefore store the `*Byte` newtypes in
   `src/plain.rs` (total decode: any byte maps to a valid variant). `Av1Block`,
   a tagged-union enum, is not `DisjointMut` data at all: it is a `Mutex` per
   slot, used only by 2-pass frame threading.
3. **All unsafe is in `rav1d-disjoint-mut`.** The main crate is
   `forbid(unsafe_code)`, so no data race can exist outside `DisjointMut`.

## Measured (intra_4k, 3840x2160 x16, interleaved A/B, identical output md5)

| threads | default | `untracked` | asm |
|---|---:|---:|---:|
| 1 | 42.7 ms/f | 37.6 (-12%) | 31.7 |
| 4 | 18.3 | 11.6 (-36%) | 10.7 |
| 8 | 17.3 | 12.1 (-30%) | 10.8 |

(Final build; the pre-change default was 43.6 / 18.8 / 18.1, so the `PlainData`
conversions cost nothing.) The t>1 gain is mostly the zero-copy path, not the
tracker: at t=8 the loop filter's copy-in / diff-write-back
(`compact_write_back_per_row_diff`) was ~21% of samples. At t=1 only tracker
removal applies.

## Validation

* **ThreadSanitizer (byte-exact):** all 803 conformance vectors at 8 threads,
  0 reports. Positive control: TSan flags a deliberate overlapping write in
  `untracked_mode.rs`. On this corpus zero-copy overlaps *reservations* but not
  unordered *accesses*; the task graph orders them.
* **Output stability:** 803/803 vectors x3 at t=8; 800 runs of the 40 largest
  vectors at t=8/16; 30/30 runs of the 4K stream at t=8. 0 mismatches.
* **Overlap census** (`__probe_bounds` + `untracked`, 649M acquisitions, t=8):
  0.012% had an overlapping reservation; ~40k write-involving events, nearly
  all strided-hull reservations (`picture.rs` `block_mut`) over gap bytes the
  guard never writes.
* **Not done:** Miri on a full decode (over 6 CPU-minutes on a 600-byte stream
  without finishing). This is evidence over the corpus, not a proof over every
  input.

## Caveats

* Cargo feature unification: any crate in the graph enabling `untracked`
  enables it for all. Do not enable it in a library.
* `docs/SOUNDNESS_AND_PERFORMANCE.md` says no distributable feature should
  silently turn safe calls into unchecked accesses. This one is opt-in, keeps
  bounds checks and has a compile-time element contract, but it is still UB by
  the language's rules.
* Tests that assert tracker refusals, poisoning or the compact-copy policy are
  gated off under this feature (`cfg(not(feature = "untracked"))`).
* `__probe_untracked` remains as an alias (tracker off, copy path kept) for the
  historical benchmark scripts.
