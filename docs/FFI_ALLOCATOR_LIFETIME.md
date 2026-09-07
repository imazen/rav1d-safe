# Retained-picture allocation after decoder destruction

PR #528's broader CI exposed a lifetime defect in `c-ffi` builds, including
`asm`. The default allocator's cookie pointed to the `Arc<MemPool<u8>>` slot
inside the originating `Rav1dContext`. Pictures retained a clone of the
allocator, but that clone retained only the raw pointer to the context slot.
After the decoder dropped, `rav1d_picture_alloc_copy` could call through that
allocator and dereference the freed slot. Keeping the pool's allocation alive
inside a picture buffer did not keep the context's Arc *slot* alive.

`picture_policy_is_local_and_survives_decoder_lifetimes` reproduced SIGSEGV
both in GitHub CI and locally. The entropy candidates are compiled out in
the failing assembly configuration. The default checked build uses a
different allocator that already owns its pool Arc.

The internal FFI allocator now owns its pool from construction. It passes
an Arc clone directly to a private allocation helper; no default callback
cookie points to an Arc slot. Each picture retains an allocator clone, and
each buffer separately owns its pool until the release callback returns it.
Dropping or moving any originating decoder, allocator, or picture cannot
invalidate the next allocator's pool handle. Custom C allocator cookies keep
their documented caller-owned lifetime contract.

Miri then exposed a second dependency: `is_default()` compared independently
materialized callback addresses, which can compare unequal for the same Rust
function. The initial cookie repair still aborted because the pool had not
been bound. The final implementation identifies internal defaults through
owned state. Imported default callbacks are recognized only to enable pool
reuse; an unrecognized default callback uses a fresh owned pool and never
interprets its cookie as Rust state. The old assertion that allocation and
release address comparisons must agree is removed. Same-signature equal
function addresses permit equivalent calls; a false negative is harmless.

The public `Dav1dPicAllocator` struct and callback signatures are unchanged.
The extra ownership field is internal Rust state; conversion to C settings
does not export it as an owning cookie. Default settings are bound to the
new decoder's pool when opened. No disjoint-mut API or borrow policy changes.

The first repair passed all 21 assembly library tests and all four `c-ffi`
lifecycle tests, including the former crashing test. An additional behavioral
test allocates
16 successive picture generations, dropping the decoder first and each
source picture before using its successor. It exercises ownership without
depending on the particular representation of the retained pool.

A minimal Miri test parses only the committed sequence header and allocates
16×16 pictures. It drops the decoder, then the allocator, then each source
picture across four generations. It also exports/imports default C callbacks
and exercises them directly without default-address recognition. The test
checks a visible row after the final allocation. This makes the callback
fallback and pool ownership paths live without expensive entropy decoding.

The final owned-helper implementation passes this test under both Stacked
Borrows and Tree Borrows with strict provenance (nightly 2026-07-25,
`rustc 1.99.0-nightly da86f4d07`). All 27 assembly library/committed-vector
checks also pass. CI runs the minimal Miri test under both models.

Reproduce through the workspace heavy-job wrapper:

```sh
cargo nextest run --release --lib --no-default-features \
  --features bitdepth_8,bitdepth_16,asm \
  -E 'test(picture_policy_is_local_and_survives_decoder_lifetimes) | test(copied_picture_can_allocate_after_decoder_and_source_drop)'
```

Run the minimal ownership test under both aliasing models:

```sh
MIRIFLAGS='-Zmiri-strict-provenance' cargo +nightly miri test --lib \
  --no-default-features --features bitdepth_8,bitdepth_16,c-ffi \
  retained_allocator_allocates_after_decoder_drop
MIRIFLAGS='-Zmiri-strict-provenance -Zmiri-tree-borrows' cargo +nightly miri test --lib \
  --no-default-features --features bitdepth_8,bitdepth_16,c-ffi \
  retained_allocator_allocates_after_decoder_drop
```

Before/after logs are recorded with the still campaign under
`benchmarks/still-entropy-2026-09-07/results/`. This fixes a lifetime failure;
it does not establish that all custom allocator uses or all FFI operations
are sound. Cross-platform CI remains a release gate.
