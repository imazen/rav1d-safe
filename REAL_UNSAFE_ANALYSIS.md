# Real Unsafe Analysis — DisjointMut and the Managed API

This is a source-verified analysis of the actual unsafe code in rav1d-safe, focused on
DisjointMut and the managed API's safety story. Every claim traces to a specific file and line.

## DisjointMut: What It Actually Is

**File:** `src/disjoint_mut.rs` (1,216 lines)

DisjointMut wraps a collection in `UnsafeCell` to allow concurrent mutable access to
non-overlapping regions. This is the core primitive enabling tile-parallel video decoding —
multiple threads write to different parts of the same pixel buffer simultaneously.

```rust
// src/disjoint_mut.rs:42-49
#[cfg_attr(not(debug_assertions), repr(transparent))]
pub struct DisjointMut<T: ?Sized + AsMutPtr> {
    #[cfg(debug_assertions)]
    bounds: debug::DisjointMutAllBounds,  // tracking state, debug only

    inner: UnsafeCell<T>,
}
```

In release builds, `repr(transparent)` makes DisjointMut zero-cost — it's just `UnsafeCell<T>`.
In debug builds, the extra `bounds` field tracks every active borrow and panics on overlap.

## The Unsafe Surface Area

DisjointMut has 6 categories of unsafe code. All are in `src/disjoint_mut.rs`.

### 1. Manual Send/Sync impls (lines 54, 83)

```rust
unsafe impl<T: ?Sized + AsMutPtr + Send> Send for DisjointMut<T> {}
unsafe impl<T: ?Sized + AsMutPtr + Sync> Sync for DisjointMut<T> {}
```

These are the load-bearing unsafe. The `Sync` impl is what lets `&DisjointMut` hand out
`&mut` slices to different threads. The safety argument (documented in a thorough comment
at lines 56-82) rests on two claims:

1. **Disjointness** — callers only access non-overlapping regions. Checked at runtime in
   debug builds; unchecked in release.
2. **Provenanceless data** — the element types (`u8`, `u16`) contain no pointers. So even
   if disjointness is violated, the worst case is wrong pixel values, not dangling pointers
   or use-after-free. A data race on `u8` can't create memory unsafety.

### 2. The `AsMutPtr` unsafe trait (line 232)

```rust
pub unsafe trait AsMutPtr {
    type Target;
    unsafe fn as_mut_ptr(ptr: *mut Self) -> *mut Self::Target;
    fn len(&self) -> usize;
}
```

The key invariant: `as_mut_ptr` must return a pointer to the underlying data **without
materializing a `&mut` reference** to the whole slice. This matters because DisjointMut
may have concurrent immutable borrows into the same backing allocation.

Implemented for: `Vec<V>` (line 974), `[V; N]` (line 991), `[V]` (line 1004),
`Box<[V]>` (line 1018), and `Rav1dPictureDataComponentInner` (in `include/dav1d/picture.rs:202`).

Each implementation is short and documented. The `Vec` impl calls `Vec::as_mut_ptr()` which
is documented to not materialize `&mut [V]`.

### 3. Core index/index_mut methods (lines 318-366)

```rust
// index_mut (line 328):
let slice = unsafe { &mut *index.get_mut(self.as_mut_slice()) };

// index (line 364):
let slice = unsafe { &*index.get_mut(self.as_mut_slice()).cast_const() };
```

These create `&mut`/`&` references from raw pointers. Sound only if the disjointness
invariant holds — no overlapping mutable borrows exist.

### 4. DisjointMutIndex impls (lines 699-750)

Pointer arithmetic: `(slice as *mut T).add(index)` and `(slice as *mut T).add(start)`.
Bounds-checked — panics on out-of-bounds, just like normal slice indexing.

### 5. AsMutPtr impls (lines 974-1029)

Straightforward: dereference a `*mut Vec<V>` to call `as_mut_ptr()`, or cast `*mut [V; N]`
to `*mut V`. All short, all documented.

### 6. The Arc transmute in DisjointMutArcSlice (line 1206)

```rust
// Release-only path:
unsafe { Arc::from_raw(Arc::into_raw(arc_slice) as *const DisjointMut<[_]>) }
```

Transmutes `Arc<[T]>` to `Arc<DisjointMut<[T]>>`. Sound because `DisjointMut` is
`repr(transparent)` wrapping `UnsafeCell`, which is also `repr(transparent)`. Verified
by a const assertion at lines 1198-1201. Only used in release builds; debug builds use
a `Box` indirection instead.

## What Could Go Wrong

### Risk 1: Overlapping mutable borrows in release

**Severity:** High in theory, mitigated in practice.

If two threads call `index_mut` on overlapping ranges in release mode, that's UB per the
Rust memory model. However:

- Debug mode catches this with runtime tracking (Mutex<Vec<Bounds>> per borrow type)
- The provenanceless-data argument means a race on `u8`/`u16` produces wrong pixels, not
  memory corruption
- All usage is `pub(crate)` — there are exactly ~271 call sites to audit
- Tile boundary math is the critical correctness dependency

### Risk 2: Tile boundary math errors

Disjointness depends entirely on correct tile/row boundary calculations elsewhere in the
decoder. An off-by-one in tile assignment → overlapping writes → UB in release. This is
the most realistic attack vector for soundness bugs. Debug-mode testing is the primary
defense.

### Risk 3: Non-provenanceless types

The safety argument assumes `AsMutPtr::Target` has no pointers. Currently all targets are
`u8`, `u16`, or `[u8; N]` — all provenanceless. If someone adds a target type with pointers,
the data-race-is-just-wrong-values argument breaks. This is only enforced by code review.

## The Managed API

**File:** `src/managed.rs` (1,013 lines)

### It really is safe

Line 46: `#![deny(unsafe_code)]`

This is `deny`, not `forbid`. The difference: `deny` can be overridden with
`#[allow(unsafe_code)]`. But there are **zero** `unsafe` blocks, **zero** pointer
operations, and **zero** transmutes anywhere in managed.rs. The `deny` is never overridden.

Confirmed by grep — the only occurrences of "unsafe" in the file are the deny attribute
and a comment.

### How DisjointMut reaches the managed API

The chain:

```
Rav1dPictureDataComponentInner     (picture.rs:132)
  ├── NonNull<u8>                  ← raw pointer to pixel allocation
  ├── len: usize
  └── stride: isize

wrapped by DisjointMut<...>        (picture.rs:228)
  = Rav1dPictureDataComponent

wrapped by Rav1dPictureData        (picture.rs:392)
  = [Rav1dPictureDataComponent; 3] (Y, U, V planes)

wrapped by Arc<Rav1dPictureData>   (in Rav1dPicture)

wrapped by Frame                   (managed.rs, safe wrapper)
```

The managed API accesses pixel data through `Rav1dPictureDataComponent::slice()`:

```rust
// picture.rs:329-338
pub fn slice<'a, BD, I>(&'a self, index: I)
    -> DisjointImmutGuard<'a, Rav1dPictureDataComponentInner, [BD::Pixel]>
{
    self.0.slice_as(index)   // calls DisjointMut::slice_as
}
```

This returns a `DisjointImmutGuard`, which (in release mode) is just `&'a [u8]` or
`&'a [u16]` behind a `repr(transparent)` newtype.

### What PlaneView actually holds

```rust
// managed.rs:683-688
pub struct PlaneView8<'a> {
    guard: DisjointImmutGuard<'a, Rav1dPictureDataComponentInner, [u8]>,
    stride: usize,
    width: usize,
    height: usize,
}
```

`guard` is a **private field**. All public methods on PlaneView return standard Rust types:

- `row(y) -> &[u8]` — bounds-checked
- `pixel(x, y) -> u8` — bounds-checked
- `rows() -> impl Iterator<Item = &[u8]>`
- `as_slice() -> &[u8]`
- `width() / height() / stride() -> usize`

No DisjointMut types, no pointers, no raw anything in the public API surface.

### DisjointImmutGuard internals

```rust
// disjoint_mut.rs:167-177 (release mode):
#[repr(transparent)]
pub struct DisjointImmutGuard<'a, T: ?Sized + AsMutPtr, V: ?Sized> {
    slice: &'a V,              // ← just a safe Rust reference
    phantom: PhantomData<...>, // zero-sized
}
```

In release mode: literally `&'a [u8]` with a PhantomData. Zero overhead.
In debug mode: adds parent reference and bounds tracking for overlap detection.

The `Deref` impl (line 213) just returns `self.slice`. Nothing exciting.

## Summary: Where Unsafe Actually Lives

| Location | What | Risk |
|----------|------|------|
| `disjoint_mut.rs:54,83` | Send/Sync impls | Core safety bet — disjointness + provenanceless |
| `disjoint_mut.rs:328,364` | index/index_mut | Creates refs from raw ptrs (depends on disjointness) |
| `disjoint_mut.rs:232` | AsMutPtr trait | Must not materialize &mut to backing slice |
| `disjoint_mut.rs:699-750` | DisjointMutIndex | Pointer arithmetic (bounds-checked) |
| `disjoint_mut.rs:974-1029` | AsMutPtr impls | Short, documented, straightforward |
| `disjoint_mut.rs:1206` | Arc transmute | repr(transparent) chain, const-verified |
| `picture.rs:154-184` | ComponentInner::new | Raw pointer math for negative strides |
| `picture.rs:202-225` | AsMutPtr for ComponentInner | Returns NonNull ptr without materializing &mut |
| `picture.rs:270-282` | as_strided_byte_mut_ptr | Pointer offset for negative stride |
| `managed.rs` | **nothing** | Zero unsafe code |

## The Safety Story, Honestly

**DisjointMut is a sound abstraction** under its stated invariants. The provenanceless-data
argument is the key insight — even if you get a data race on `u8` pixels, you get wrong
pixels, not memory corruption. This is a weaker guarantee than "no data races ever" but a
stronger guarantee than "full UB on any race."

**The managed API is genuinely safe.** It contains zero unsafe code. It holds DisjointMut
guards as private fields and only exposes standard Rust types. A user of the managed API
cannot trigger UB through it — all the unsafe is encapsulated behind bounds-checked,
lifetime-tracked safe interfaces.

**The realistic risk** is bugs in tile boundary calculations causing overlapping writes
during parallel decode. Debug-mode overlap detection is the primary defense. Running the
full test suite with debug assertions enabled exercises this checking.

**What this is NOT:** a fully verified, formally-proven-correct abstraction. It's a pragmatic
unsafe primitive with good documentation, debug-mode checking, and a sound theoretical
basis. The `pub(crate)` visibility keeps the audit surface finite.
