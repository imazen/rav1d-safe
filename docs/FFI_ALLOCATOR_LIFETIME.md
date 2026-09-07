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

The internal FFI allocator now retains its own pool Arc. It creates the raw
callback cookie from the Arc slot in the currently borrowed allocator only
for the callback invocation. `&self` keeps that slot alive and stationary for
the call, and every allocator clone independently owns the pool. Picture
destruction still returns the buffer to its pool through the existing release
callback. Custom C allocator cookies retain their documented caller-owned
lifetime contract.

The public `Dav1dPicAllocator` struct and callback signatures are unchanged.
The extra ownership field is internal Rust state; conversion to C settings
does not export it as an owning cookie. Default settings are bound to the
new decoder's pool when opened. No disjoint-mut API or borrow policy changes.

All 21 assembly library tests and all four `c-ffi` lifecycle tests now pass,
including the former crashing test. An additional behavioral test allocates
16 successive picture generations, dropping the decoder first and each
source picture before using its successor. It exercises ownership without
depending on the particular representation of the retained pool.

Reproduce through the workspace heavy-job wrapper:

```sh
cargo nextest run --release --lib --no-default-features \
  --features bitdepth_8,bitdepth_16,asm \
  -E 'test(picture_policy_is_local_and_survives_decoder_lifetimes) | test(copied_picture_can_allocate_after_decoder_and_source_drop)'
```

Before/after logs are recorded with the still campaign under
`benchmarks/still-entropy-2026-09-07/results/`. This fixes a lifetime failure;
it does not establish that all custom allocator uses or all FFI operations
are sound. Cross-platform CI remains a release gate.
