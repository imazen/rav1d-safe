# Unsafe Audit: Items Blocking `#![forbid(unsafe_code)]`

These are all `#[allow(unsafe_code)]` items that are NOT already gated behind
`cfg(feature = "asm")` or `cfg(feature = "c-ffi")`. Each needs a resolution
to enable crate-level `forbid(unsafe_code)`.

Items marked ❓ = I don't know how to make safe. Please advise.

---

## Category 1: Send/Sync/Pin impls (10 items)

These are `unsafe impl Send/Sync` on types containing raw pointers or non-auto-trait fields.

### 1a. `src/c_arc.rs:73,77` — StableRef Send/Sync
```rust
unsafe impl<T: Send + ?Sized> Send for StableRef<T> {}
unsafe impl<T: Send + ?Sized> Sync for StableRef<T> {}
```
StableRef wraps `NonNull<T>`. Semantically acts like `&T`.

### 1b. `src/internal.rs:308,312` — TaskThreadDataDelayedFg Send/Sync
```rust
// TODO(SJC): Remove when TaskThreadDataDelayedFg is thread-safe.
unsafe impl Send for TaskThreadDataDelayedFg {}
unsafe impl Sync for TaskThreadDataDelayedFg {}
```
Protected by external mutex synchronization. Has a TODO to remove.

### 1c. `src/internal.rs:444,448` — Rav1dContext Send/Sync
```rust
// TODO(SJC): Remove when Rav1dContext is thread-safe.
unsafe impl Send for Rav1dContext {}
unsafe impl Sync for Rav1dContext {}
```
Main decoder context. Thread safety managed by caller API contract.

### 1d. `src/msac.rs:190,197` — MsacAsmContextBuf Send/Sync
```rust
unsafe impl Send for MsacAsmContextBuf {}
unsafe impl Sync for MsacAsmContextBuf {}
```
Contains `*const u8` pos/end pointers into owned `CArc<[u8]>` data.

### 1e. `src/c_box.rs:155` — Pin::new_unchecked
```rust
pub fn into_pin(self) -> Pin<Self> {
    unsafe { Pin::new_unchecked(self) }
}
```
CBox data is heap-allocated (Box) or C-owned (never moved until Drop).

❓ Can't see how to avoid `unsafe` for any of these. Move to separate crate?

---

## Category 2: AlignedVec / Align* (4 items, one macro-expanded 7x)

### 2a. `src/align.rs:105` — ExternalAsMutPtr for Align* (7 expansions)
```rust
unsafe impl<V: Copy, const N: usize> ExternalAsMutPtr for $name<[V; N]> {
    unsafe fn as_mut_ptr(ptr: *mut Self) -> *mut V {
        unsafe { assume(ptr.is_aligned()) }
        ptr.cast()
    }
}
```
Expanded for Align1/2/4/8/16/32/64. Provides raw pointer cast from aligned wrapper.

### 2b. `src/align.rs:179` — AlignedVec::as_slice
```rust
pub fn as_slice(&self) -> &[T] {
    unsafe { slice::from_raw_parts(self.as_ptr(), self.len) }
}
```

### 2c. `src/align.rs:190` — AlignedVec::as_mut_slice
```rust
pub fn as_mut_slice(&mut self) -> &mut [T] {
    unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), self.len) }
}
```

### 2d. `src/align.rs:200` — AlignedVec::resize
```rust
pub fn resize(&mut self, new_len: usize, value: T) {
    // Writes T values into MaybeUninit<C> storage through raw pointer casts
}
```

❓ AlignedVec stores `MaybeUninit<C>` chunks but exposes `T` elements.
Fundamentally needs unsafe for the type-punning. Move to separate crate?

---

## Category 3: Low-level primitives (3 items)

### 3a. `src/assume.rs:11` — unreachable_unchecked hint
```rust
pub const unsafe fn assume(condition: bool) {
    if !condition {
        unsafe { unreachable_unchecked() };
    }
}
```
Compiler optimization hint. UB if condition is false.

### 3b. `src/ffi_safe.rs:26,35` — FFISafe pointer-to-ref casts
```rust
pub unsafe fn get(this: *const Self) -> &'a T { unsafe { &*this.cast() } }
pub unsafe fn _get_mut(this: *mut Self) -> &'a mut T { unsafe { &mut *this.cast() } }
```
Used at FFI boundaries to unwrap opaque pointers back to references.

❓ FFISafe is only used by asm FFI wrappers. Could gate behind `cfg(feature = "asm")`?

---

## Category 4: CArc interior pointer (1 item)

### 4a. `src/c_arc.rs:81` — CArc::as_ref dereferences NonNull
```rust
impl<T: ?Sized> AsRef<T> for CArc<T> {
    fn as_ref(&self) -> &T {
        unsafe { self.stable_ref.0.as_ref() }
    }
}
```
Dereferences `NonNull<T>` stored in `StableRef`. The pointer is always valid
because it's derived from the owned `CBox`.

❓ Fundamental to CArc. Move to separate crate?

---

## Category 5: DisjointMut bridge (1 item)

### 5a. `src/disjoint_mut.rs:42` — ExternalAsMutPtr for AlignedVec
```rust
unsafe impl<T: Copy, C: AlignedByteChunk> ExternalAsMutPtr for AlignedVec<T, C> {
    unsafe fn as_mut_ptr(ptr: *mut Self) -> *mut Self::Target {
        let ptr = unsafe { &mut *ptr }.as_mut_ptr();
        unsafe { assume(ptr.cast::<C>().is_aligned()) };
        ptr
    }
}
```
Part of DisjointMut safety abstraction. Dereferences raw pointer, asserts alignment.

❓ DisjointMut is already a separate crate at `crates/disjoint-mut/`.
Could this impl move there?

---

## Category 6: refmvs dispatch (2 items)

### 6a. `src/refmvs.rs:485` — splat_mv_direct
```rust
#[cfg(not(feature = "asm"))]
fn splat_mv_direct(
    rr: *mut *mut RefMvsBlock,  // raw pointer
    rmv: &Align16<RefMvsBlock>,
    bx4: i32, bw4: i32, bh4: i32,
) {
```
Takes `*mut *mut RefMvsBlock`, calls arch-specific SIMD functions.
On x86_64 calls `splat_mv_avx2` (unsafe extern C), on aarch64 calls `splat_mv_neon`.
Scalar fallback uses `slice::from_raw_parts_mut` on raw pointers.

### 6b. `src/refmvs.rs:531` — splat_mv::Fn::call
```rust
pub fn call(&self, rf: &RefMvsFrame, rt: &RefmvsTile, rmv: &Align16<RefMvsBlock>,
            b4: Bxy, bw4: usize, bh4: usize) {
    // Creates raw pointers from DisjointMutGuard
    unsafe { guard.as_mut_ptr().sub(bx4) }
}
```

❓ Could the SIMD inner functions take slices instead of `*mut *mut`?

---

## Category 7: safe_simd NEON (aarch64-only, 19 items)

All gated behind `#[cfg(target_arch = "aarch64")]`. NEON intrinsics are still
`unsafe` in Rust (unlike x86 intrinsics which became safe in 1.93 with target_feature).
Using `#[arcane]` with `Arm64`/`NeonToken` makes computation intrinsics safe,
but load/store still needs safe_unaligned_simd macros.

### 7a. `src/msac.rs:642` — NEON symbol_adapt16
Inner NEON function with vld1q/vst1q/vaddq/vmull etc.
Can convert to `#[arcane]` + safe macros.

### 7b. `src/safe_simd/refmvs.rs:25` — splat_mv_avx2 (x86_64, not aarch64)
```rust
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn splat_mv_avx2(rr: *mut *mut RefMvsBlock, ...)
```
NOT gated behind asm. Called from safe-simd dispatch. Takes raw pointers.

### 7c. `src/safe_simd/refmvs_arm.rs:22` — splat_mv_neon
Same but NEON. Takes raw pointers.

### 7d. `src/safe_simd/mc_arm.rs` — 10 items
- `:1372` w_mask_8bpc_inner (NEON target_feature + intrinsics)
- `:2435` w_mask_16bpc_inner (same)
- `:4255,4310,4370,4432,4513,4644,4729,4881` — 8 dispatch fns
  (extract raw pointers from safe types to call inner NEON fns)

### 7e. `src/safe_simd/loopfilter_arm.rs` — 3 items
- `:224` lpf_h_sb_inner (raw pointers for dst/lvl)
- `:298` lpf_v_sb_inner (same)
- `:591` loopfilter_sb_dispatch (bridge from safe types to raw pointers)

### 7f. `src/safe_simd/filmgrain_arm.rs` — 8 items
- `:330` fgy_row_neon_8bpc (raw pointers + NEON)
- `:399` fgy_inner_8bpc (raw pointers for pixel rows)
- `:572` fgy_inner_16bpc (same, 16bpc)
- `:768` compute_uv_scaling_val (raw pointer dereference)
- `:791` fguv_inner_8bpc (raw pointers for chroma grain)
- `:1038` fguv_inner_16bpc (same, 16bpc)
- `:1309` fgy_32x32xn_dispatch (bridge safe types → raw pointers)
- `:1381` fguv_32x32xn_dispatch (same for chroma)

❓ The x86_64 equivalents of all these are fully safe (slices + archmage).
ARM modules were left as raw-pointer scalar fallbacks. Need full conversion
to slices + `#[arcane]`/`Arm64` + safe NEON macros — same pattern as x86.

---

## Category 8: partial_simd.rs (1 module-level allow)

### 8a. `src/safe_simd/partial_simd.rs:36`
```rust
#![allow(unsafe_code)]
```
Module of ~50-100 functions wrapping individual 64-bit SSE2 intrinsics
(`_mm_loadl_epi64`, `_mm_storel_epi64`, etc.) in safe `#[target_feature]` fns.
Exists because `safe_unaligned_simd` doesn't cover 64-bit operations.

❓ Move to safe_unaligned_simd crate? Or to a separate internal crate?

---

## Category 9: Test-only (2 items)

### 9a. `src/safe_simd/cdef.rs:155,192` — test calls to target_feature fns
```rust
#[cfg(test)]
#[allow(unsafe_code)]
mod tests { ... }
```
Tests call `#[target_feature(enable = "avx2")]` functions, which requires unsafe.

### 9b. `src/decode_test.rs:13` — test using internal API
```rust
#[cfg(test)]
#[allow(unsafe_code)]
mod tests { ... }
```
Calls `rav1d_open`/`rav1d_send_data`/`rav1d_get_picture`/`rav1d_close` directly.

❓ Can these tests be rewritten to use the safe managed API instead?
Or gate the test modules behind a feature?

---

## Summary

| Category | Count | Resolution options |
|----------|-------|--------------------|
| Send/Sync/Pin impls | 10 | Move types to sub-crate? |
| AlignedVec/Align* | 4+7 | Move to sub-crate? |
| Low-level primitives | 3 | Gate behind feature / move to sub-crate? |
| CArc AsRef | 1 | Move to sub-crate? |
| DisjointMut bridge | 1 | Move to disjoint-mut crate? |
| refmvs dispatch | 2 | Convert to slices? |
| ARM NEON safe_simd | 19 | Convert to archmage+safe macros (like x86) |
| partial_simd | 1 | Move to sub-crate? |
| Test-only | 2 | Rewrite tests / gate behind feature |

**Total: ~50 items** (counting macro expansions)

The ARM NEON items (19) are straightforward — same conversion pattern as x86.
The rest are sound abstractions or foundational types that fundamentally need unsafe.

## KEY FINDING

**All 50 items are already conditionally compiled out** when building with
`--no-default-features --features "bitdepth_8,bitdepth_16"` on both x86_64
and aarch64. The build passes with `forbid(unsafe_code)` today on both targets.
This file documents what would need resolution if those gates were ever removed.
