# SPEC: Safe Pixel Access Architecture

## Status: DESIGN PHASE

## 1. Current Architecture

### Type Graph

```
Rav1dPictureData
  └── data: [Rav1dPictureDataComponent; 3]   // Y, U, V planes
        └── Rav1dPictureDataComponent(DisjointMut<Rav1dPictureDataComponentInner>)
              └── DisjointMut<T>
                    ├── tracker: Option<BorrowTracker>   // runtime overlap detection
                    └── inner: UnsafeCell<T>
                          └── Rav1dPictureDataComponentInner
                                ├── ptr: NonNull<u8>     // raw pixel data (aligned 64)
                                ├── len: usize           // byte length (multiple of 4096)
                                └── stride: isize        // byte stride (may be negative!)
```

### Access Paths (Current)

**Tracked (28% of accesses — 62 call sites):**
```
Rav1dPictureDataComponent
  → .slice::<BD>(range)     → DisjointImmutGuard<[BD::Pixel]>
  → .slice_mut::<BD>(range) → DisjointMutGuard<[BD::Pixel]>
  → .index::<BD>(i)         → DisjointImmutGuard<BD::Pixel>
  → .index_mut::<BD>(i)     → DisjointMutGuard<BD::Pixel>
```
Used in: recon.rs (19), decode.rs (18), lf_apply.rs (4), lf_mask.rs (4), refmvs.rs (5)

**Untracked (72% of accesses — 161 call sites):**
```
Rav1dPictureDataComponent
  → impl Pixels: .as_byte_mut_ptr() → *mut u8     // BYPASSES TRACKER
  → impl Pixels: .as_mut_ptr::<BD>() → *mut BD::Pixel

DisjointMut<T>
  → .as_mut_ptr() → *mut T::Target                 // BYPASSES TRACKER
  → .inner() → *mut T                               // BYPASSES TRACKER (stride access)
```
Used in: looprestoration.rs (37), safe_simd/mc.rs (20), msac.rs (15), refmvs.rs (11)

### Data Flow: Frame → Dispatch → SIMD

```
recon.rs: Create PicOffset = WithOffset<&Rav1dPictureDataComponent>
    │         { data: &component, offset: pixel_offset }
    ▼
cdef_apply.rs / lf_apply.rs / lr_apply.rs: Pass PicOffset to DSP
    │
    ▼
cdef.rs / mc.rs / etc. (dispatch layer): Match CPU features
    │         Passes PicOffset unchanged
    ▼
safe_simd/cdef.rs (SIMD dispatch): Match bitdepth + variant
    │         Passes PicOffset unchanged
    ▼
safe_simd/cdef.rs (SIMD inner fn): 
    Extract stride: dst.pixel_stride::<BD>()
    Get pixel slices: (dst + y*stride).slice::<BD>(w)  ← USES DisjointMut GUARDS!
    Get mutable slices: (dst + y*stride).slice_mut::<BD>(w)
    Write back filtered pixels
```

**Key Finding:** CDEF inner functions already use tracked `.slice()`/`.slice_mut()` from 
the PicOffset. Other modules (mc, ipred, itx) use raw pointers extracted from PicOffset 
via the Pixels trait instead.

### WithOffset<T> — The Offset Carrier

```rust
pub struct WithOffset<T> {
    pub data: T,        // &Rav1dPictureDataComponent, PicOrBuf, etc.
    pub offset: usize,  // pixel offset into data
}
// Supports: +=/-= usize/isize, + and - operators
// impl Pixels: delegates to data.as_ptr_at(offset) → *mut BD::Pixel
// impl Strided: delegates to data.stride()
```

WithOffset is used as an "iterable cursor" — recon.rs advances the offset through blocks:
```rust
bptrs[0] += 8usize;     // Advance 8 pixels for next block
```

### PicOrBuf — Dispatch Between Frame and Temp Buffer

```rust
pub enum PicOrBuf<'a, T: AsMutPtr<Target = u8>> {
    Pic(&'a Rav1dPictureDataComponent),   // frame pixel data
    Buf(WithStride<&'a DisjointMut<T>>),  // temporary buffer (e.g. line buffers)
}
// impl Pixels: dispatches to either Pic or Buf
// impl Strided: dispatches to either Pic or Buf
```

Used in: CDEF (top/bottom edge buffers), loop restoration (line buffer), loop filter.

### Strided Trait

```rust
pub trait Strided {
    fn stride(&self) -> isize;  // in BYTES
    fn pixel_stride<BD: BitDepth>(&self) -> isize;  // in pixels
}
```

Stride is always in **bytes** at the trait level. BD::pxstride() converts to pixel units.

**How stride is currently accessed for Rav1dPictureDataComponent:**
```rust
impl Strided for Rav1dPictureDataComponent {
    fn stride(&self) -> isize {
        unsafe { (*self.0.inner()).stride }  // ← BYPASSES TRACKER via inner()
    }
}
```
This is the ONLY call to `.inner()` in the codebase. It reads stride metadata, 
not pixel data, so it doesn't alias — but it's still technically unsafe.

## 2. Lifetime Soundness Audit

### ✅ DisjointMutGuard Lifetimes — SOUND

```rust
pub struct DisjointMutGuard<'a, T: ?Sized + AsMutPtr, V: ?Sized> {
    slice: &'a mut V,                       // borrows from UnsafeCell
    phantom: PhantomData<&'a DisjointMut<T>>,
    parent: Option<&'a DisjointMut<T>>,     // keeps parent alive
    borrow_id: checked::BorrowId,           // for deregistration
}
```

- `'a` from `index_mut(&'a self)` → guard cannot outlive the DisjointMut borrow
- Slice cannot outlive guard (guard holds the `&'a mut`)
- Drop deregisters via `parent.tracker.remove(borrow_id)`

### ✅ ManuallyDrop in cast_slice/cast — SOUND

```rust
fn cast_slice<V>(self) -> DisjointMutGuard<'a, T, [V]> {
    let mut old = ManuallyDrop::new(self);  // prevent double-deregister
    let bytes = mem::take(&mut old.slice);
    DisjointMutGuard {
        slice: V::mut_slice_from(bytes).unwrap(),
        parent: old.parent,       // same parent
        borrow_id: old.borrow_id, // same registration
        phantom: old.phantom,
    }
}
```

- Old guard suppressed (ManuallyDrop), new guard takes ownership of registration
- Lifetime `'a` preserved across transfer
- No double-deregister possible

### ✅ Rav1dPictureDataComponentOffset Lifetimes — SOUND

```rust
type PicOffset<'a> = WithOffset<&'a Rav1dPictureDataComponent>;

impl<'a> PicOffset<'a> {
    fn slice_mut<BD>(&self, len: usize) 
        -> DisjointMutGuard<'a, ..., [BD::Pixel]>
    {
        self.data.slice_mut::<BD, _>((self.offset.., ..len))
    }
}
```

- Guard lifetime `'a` matches the component reference lifetime
- Guard cannot outlive the PicOffset, which cannot outlive the component

### ✅ PlaneView in managed.rs — SOUND

```rust
pub struct PlaneView8<'a> {
    guard: DisjointImmutGuard<'a, Rav1dPictureDataComponentInner, [u8]>,
    stride: usize, width: usize, height: usize,
}
```

- Struct lifetime matches guard lifetime — drops together
- Read-only access via `&self.guard[..]`

### ⚠️ Known Soundness Hole: Pixels Trait

```rust
pub trait Pixels {
    fn as_byte_mut_ptr(&self) -> *mut u8;  // RETURNS UNTRACKED *mut FROM &self
}

impl Pixels for Rav1dPictureDataComponent {
    fn as_byte_mut_ptr(&self) -> *mut u8 {
        self.0.as_mut_ptr()  // bypasses tracker entirely
    }
}
```

The Pixels file has `#![forbid(unsafe_code)]` but launders untracked mutable pointers 
through a safe trait. Multiple callers can get overlapping `*mut` without detection.

This is the central problem this spec addresses.

## 3. Proposed Architecture

### New Type: Tracked Pixel Slices via PlaneRegion

Instead of passing `PicOffset` (which carries an untracked pointer), pass slices 
extracted from DisjointMut guards at the dispatch boundary.

**Dispatch boundary (cdef.rs, mc.rs, etc.):**
```rust
// BEFORE: PicOffset carries untracked pointer
fn cdef_dispatch<BD>(dst: PicOffset, ...) {
    cdef_inner(dst, ...);  // inner fn extracts ptr from PicOffset
}

// AFTER: Guard created at dispatch, slice passed to inner fn  
fn cdef_dispatch<BD>(dst: PicOffset, ...) {
    // Create tracked guard covering the block we're about to filter
    let mut guard = dst.data.slice_mut::<BD, _>((dst.offset.., ..block_size));
    let stride = dst.data.pixel_stride::<BD>();
    cdef_inner(&mut guard, stride, ...);  // inner fn receives &mut [BD::Pixel]
}
```

**Inner SIMD function:**
```rust
// BEFORE: uses raw pointers
#[arcane]
unsafe fn cdef_inner(dst: PicOffset, ...) {
    let stride = dst.pixel_stride::<BD>();
    let row = unsafe { from_raw_parts_mut(dst.as_mut_ptr().offset(y * stride), w) };
}

// AFTER: fully safe, slice-based
#[arcane]
fn cdef_inner(dst: &mut [u8], stride: usize, ...) {
    let row = bd_row_mut(dst, y, stride, w);  // safe or unchecked via feature
    let arr: &mut [u8; 32] = simd_store(row, col);
    safe_unaligned_simd::_mm256_storeu_si256(arr, result);
}
```

### pixel_access Helpers (dual-mode)

Already exists at `src/safe_simd/pixel_access.rs`. Will be expanded:

```rust
#![cfg_attr(not(feature = "unchecked"), deny(unsafe_code))]

/// Row from a strided buffer. Safe mode: bounds-checked. Unchecked: debug_assert.
#[inline(always)]
pub fn bd_row<T>(buf: &[T], row: usize, stride: usize, width: usize) -> &[T] {
    let offset = row * stride;
    #[cfg(feature = "unchecked")]
    {
        debug_assert!(offset + width <= buf.len());
        unsafe { buf.get_unchecked(offset..offset + width) }
    }
    #[cfg(not(feature = "unchecked"))]
    &buf[offset..offset + width]
}

/// Fixed-size array ref for SIMD load. Safe mode: first_chunk. Unchecked: pointer cast.
#[inline(always)]
pub fn simd_load<T, const N: usize>(slice: &[T], offset: usize) -> &[T; N] {
    #[cfg(feature = "unchecked")]
    {
        debug_assert!(offset + N <= slice.len());
        unsafe { &*(slice.as_ptr().add(offset) as *const [T; N]) }
    }
    #[cfg(not(feature = "unchecked"))]
    slice[offset..offset + N].first_chunk().unwrap()
}
```

**Codegen verified:**
- `unchecked` mode: identical to raw `ptr.add()` — single `lea` instruction
- Safe mode: 2 extra `cmp` + cold-path `jmp` per access (~5% overhead in tight loops)

### Stride Access Without inner()

Replace the unsafe `.inner()` call with a safe metadata accessor:

```rust
impl Rav1dPictureDataComponent {
    /// Returns the byte stride. Safe because stride is metadata, not pixel data.
    pub fn stride(&self) -> isize {
        // Option 1: Add a safe stride() method to DisjointMut for types that impl Strided
        // Option 2: Store stride separately alongside the DisjointMut
        // Option 3: Add metadata accessor to AsMutPtr trait
    }
}
```

**Preferred: Store stride alongside DisjointMut in the wrapper struct:**
```rust
pub struct Rav1dPictureDataComponent {
    data: DisjointMut<Rav1dPictureDataComponentInner>,
    stride: isize,  // cached at construction time, no need for inner()
}
```

This eliminates the last `inner()` call.

### What Happens to PicOffset

PicOffset itself doesn't need to change immediately. The migration is at the 
**dispatch → inner** boundary:

```
BEFORE:
  recon.rs → PicOffset → dispatch → PicOffset → inner SIMD → Pixels::as_mut_ptr()

AFTER:
  recon.rs → PicOffset → dispatch → guard = PicOffset.slice_mut() → inner SIMD(&mut [P])
```

PicOffset still works for offset arithmetic in recon.rs. The change is that dispatch 
functions create guards and pass slices instead of forwarding PicOffset into inner fns.

## 4. Module-by-Module Impact Analysis

### CDEF — Easiest Migration Target

**Current inner SIMD pattern:**
```
PicOffset → .pixel_stride() → stride
PicOffset → .slice() / .slice_mut() → already uses guards!
```
CDEF already uses tracked access in its inner functions. Migration is mostly:
- Remove Pixels trait usage from dispatch
- Change inner fn signature from PicOffset to slice+stride
- 36+32 unsafe to eliminate (x86+arm)

### Loop Filter — Clean Boundaries

Uses `WithOffset<WithStride<&DisjointMut<...>>>` for lpf buffers.
Inner functions use `.mut_slice_as()` guards on the buffer.
Migration similar to CDEF.

### Loop Restoration — Largest Untracked User

37 `.as_mut_ptr()` calls — most untracked of any module.
Uses both PicOffset (for frame data) and DisjointMut (for line buffers).
Migration requires careful guard placement.

### MC (Motion Compensation) — Most Complex

101 SIMD functions, many operating on temporary `[i16]` arrays that are already 
stack-allocated slices. The PicOffset → raw ptr path is for the destination only.
Migration: change dst parameter from PicOffset to &mut [P].

### ITX (Inverse Transform) — Largest Volume

80+94 SIMD functions. Operates on coefficient arrays and destination blocks.
Highly mechanical migration — same pattern repeated 160+ times.

### IPred (Intra Prediction) — Heavy Pointer Arithmetic

29+20 functions with complex pointer patterns (diagonal scans, etc.).
Most complex individual function signatures.

## 5. Negative Stride Handling

AV1 uses negative strides for bottom-up frame buffers. Current handling:

```rust
// Rav1dPictureDataComponentInner::new():
let ptr = if stride < 0 {
    ptr.offset(-stride).sub(len)  // points to START of buffer
} else {
    ptr
};
```

After construction, `ptr` always points to the buffer start. Stride is negative.
Row access: `row(y) = ptr + y * stride` (where stride < 0, so higher y → lower address).

**For the slice-based approach:**
The guard always covers the full buffer (positive range). Row computation adjusts:
```rust
fn bd_row(buf: &[u8], row: usize, stride: isize, width: usize) -> &[u8] {
    let offset = if stride >= 0 {
        row * stride as usize
    } else {
        // Row 0 is at the logical "top" = buffer end - abs_stride
        buf.len() - (row + 1) * (-stride as usize) 
    };
    &buf[offset..offset + width]
}
```

**Alternative:** The pixel_offset calculation in `Rav1dPictureDataComponent::pixel_offset()` 
already handles this, and PicOffset carries the correct base offset. So dispatch functions 
can extract a guard for the exact block range, and inner functions just use positive 
row indexing within that guard.

## 6. Feature Flags & Safety Levels

```
                  safe_simd/ safety level
                  ═══════════════════════
Default build     │ forbid(unsafe_code)  │  bounds-checked slices
                  │ + safe_unaligned_simd│  + first_chunk().unwrap()
                  │ + #[arcane] macro    │  for SIMD load/store
                  ├──────────────────────┤
--features        │ deny(unsafe_code)    │  get_unchecked() slices
  unchecked       │ + allow(unsafe_code) │  + debug_assert!()
                  │   in pixel_access.rs │  zero-overhead SIMD access
                  ├──────────────────────┤
--features        │ allow(unsafe_code)   │  FFI wrappers compiled
  asm             │ extern "C" fn + raw  │  pointer conversion at boundary
                  │ pointer dispatch     │  assembly hot paths
                  ╘══════════════════════╛
```

## 7. Open Design Questions

### Q1: Should inner SIMD functions take `&mut [BD::Pixel]` or `&mut [u8]`?

**Option A: `&mut [BD::Pixel]`** (type-safe)
- Pro: Stride is in pixels, SIMD loads/stores use correct element type
- Pro: No byte-to-pixel conversion needed in inner functions
- Con: Requires the dispatch layer to know BD at guard-creation time (it already does)

**Option B: `&mut [u8]`** (byte-level)
- Pro: Matches current DisjointMut<Rav1dPictureDataComponentInner> Target = u8
- Con: Inner functions must convert bytes to pixels manually
- Con: Stride conversion needed (bytes → pixels)

**Recommendation: Option A** — the dispatch layer already knows BD, so it can create 
`DisjointMutGuard<..., [BD::Pixel]>` via `.slice_mut::<BD, _>()`.

### Q2: Guard creation granularity

**Option A: Per-block guards** (one guard per 4x4/8x8 block)
- Pro: Precise tracking, catches fine-grained overlap bugs
- Con: Mutex lock per block = significant overhead (thousands of blocks per frame)

**Option B: Per-tile/row guards** (one guard per tile or row of blocks)
- Pro: Amortized overhead (tens of guards per frame, not thousands)
- Con: Less precise overlap detection
- This is what recon.rs already does

**Option C: Per-filter-call guards** (one guard for each DSP call)
- Pro: Minimal overhead (guard created once at dispatch entry)
- Con: Guard covers exactly the region the filter will touch
- This is what PicOffset.slice_mut() already provides

**Recommendation: Option C** — matches current PicOffset.slice_mut() usage and has 
minimal overhead. The dispatch function creates one guard per DSP call.

### Q3: Should PlaneRegion be a separate type or just use slices?

**Option A: PlaneRegion wrapper type** (carries guard + width + height + stride)
- Pro: Self-describing, can enforce invariants
- Con: Another type to pass around, generics explosion

**Option B: Just pass `(&mut [BD::Pixel], stride, width, height)` tuples**
- Pro: Simple, no new types, direct slice access
- Con: Easy to mix up parameters, no type safety

**Option C: Pass `&mut [BD::Pixel]` + stride, let inner fn compute rows**
- Pro: Minimal API surface, inner fn already knows w/h from its parameters
- Con: Stride must be passed separately

**Recommendation: Option C for now** — inner SIMD functions already receive w/h as 
parameters. Adding stride as a parameter is sufficient. PlaneRegion can be added later 
as a convenience type wrapping this pattern.

### Q4: Stride access for Rav1dPictureDataComponent

**Option A: Cache stride in wrapper struct** (add `stride: isize` field)
- Pro: Completely eliminates `inner()` call
- Pro: Zero-cost access
- Con: Must keep in sync (but stride never changes after construction)

**Option B: Add `fn stride(&self) -> isize` to AsMutPtr trait**
- Pro: Generic, works for any container with stride metadata
- Con: Not all AsMutPtr types have stride

**Option C: Keep `inner()` but mark it `unsafe`**
- Pro: Minimal change
- Con: Still an unsafe escape hatch

**Recommendation: Option A** — stride is immutable metadata set at construction time. 
Caching it removes the only `inner()` call site and makes Strided impl trivially safe.

## 8. Migration Sequence

Phase 0: Foundation
  ├── Cache stride in Rav1dPictureDataComponent (eliminate inner())
  ├── Mark DisjointMut::as_mut_ptr()/as_mut_slice()/inner() → unsafe fn
  ├── Propagate unsafe through call sites
  └── Commit

Phase 1: pixel_access expansion
  ├── Add generic bd_row/bd_row_mut (replace type-specific versions)
  ├── Add simd_load/simd_store const-generic helpers
  └── Unit tests

Phase 2: Module migration (one module at a time)
  ├── Change dispatch fn: create guard, extract slice, pass to inner
  ├── Change inner fn: take (&mut [P], stride) instead of PicOffset
  ├── Replace ptr arithmetic with bd_row/simd_load/simd_store
  ├── Replace std intrinsic loads/stores with safe_unaligned_simd
  ├── Add #[arcane] to inner fns not already using it
  ├── Test (cargo test --release)
  └── When all fns in module converted: add forbid(unsafe_code)

  Order: pal → refmvs → cdef → loopfilter → looprestoration
       → filmgrain → ipred → itx → mc

Phase 3: Cleanup
  ├── Remove Pixels trait
  ├── Remove WithOffset (or keep for offset arithmetic only, without Pixels)
  ├── Add forbid(unsafe_code) to safe_simd/mod.rs
  └── Final test suite pass
