# rav1d-disjoint-mut

Runtime-checked disjoint mutable access to contiguous storage.

`DisjointMut` wraps a collection (`Vec<T>`, `Box<[T]>`, `[T; N]`) and allows non-overlapping mutable borrows through a shared `&` reference. Like `RefCell`, it enforces borrowing rules at runtime — but instead of whole-container borrows, it tracks *ranges* and panics only on truly overlapping access.

## Use case

Multiple threads need to write to different regions of the same buffer simultaneously. Standard Rust won't let you split `&mut [T]` across threads without unsafe code. `DisjointMut` adds runtime tracking so the borrow checker doesn't have to:

```rust
use rav1d_disjoint_mut::DisjointMut;
use std::sync::Arc;
use std::thread;

let buf = Arc::new(DisjointMut::new(vec![0u8; 100]));

let b1 = buf.clone();
let b2 = buf.clone();

let t1 = thread::spawn(move || {
    let mut guard = b1.index_mut(0..50);
    guard.fill(1);
});

let t2 = thread::spawn(move || {
    let mut guard = b2.index_mut(50..100);
    guard.fill(2);
});

t1.join().unwrap();
t2.join().unwrap();

let all = buf.index(0..100);
assert!(all[..50].iter().all(|&x| x == 1));
assert!(all[50..].iter().all(|&x| x == 2));
```

## How it works

Every `.index()` and `.index_mut()` call validates that the requested range doesn't overlap with any outstanding borrow. Mutable borrows conflict with everything; immutable borrows only conflict with mutable borrows (multiple readers are fine).

Guards act as locks — the borrow is tracked for the guard's lifetime and released on drop.

### Element types must be `Copy`

All container element types must be `Copy`. Concurrent mutable access to different regions of the same buffer means a torn read on a region boundary is possible in theory. With `Copy` types, a torn read produces a wrong value, not a dangling pointer or double free. Non-`Copy` types could have drop glue or internal invariants that torn reads would violate.

### Borrow tracking

Borrows are tracked in a **64-slot inline array** with a `u64` bitmask for O(1) allocation and deallocation. Each `index()` or `index_mut()` call occupies one slot; the slot is freed when the guard drops.

If all 64 inline slots are occupied, additional borrows spill into a heap-allocated `Vec`. The overflow `Vec` is never allocated unless you actually exceed 64 concurrent borrows on a single instance. The theoretical maximum is 254 concurrent borrows per instance (u8 encoding limit).

Empty ranges (start >= end) are free — they skip slot allocation and overlap checks entirely.

### Poisoning

Like `std::sync::Mutex`, `DisjointMut` poisons the data structure when a thread panics while holding a mutable borrow guard. After poisoning, all future borrow attempts panic.

Immutable guards do **not** poison on panic. Poisoning also triggers on out-of-bounds panics during indexing.

### Unchecked mode

`unsafe fn dangerously_unchecked()` creates an instance without runtime tracking. The caller must guarantee that all borrows are non-overlapping.

`new()` always creates a tracked instance.

### Open-ended ranges are conservative

Open-ended ranges like `5..` are tracked as `5..usize::MAX`. The tracker may reject borrows beyond the collection's actual length that wouldn't truly overlap with the guarded data. In practice this rarely matters, since out-of-bounds access would panic anyway.

## Ways to get it wrong

The runtime tracker catches overlapping borrows, but the crate also has `unsafe` extension points that can be misused. Here are the subtle ones:

### `as_mut_slice` that creates `&Self` for inline-data types

`as_mut_slice` is a required method — you have to write it. The subtle mistake is creating `&Self` inside it for a type where element data is stored inline (not behind a pointer). The shared reference produces a SharedReadOnly tag covering the data, which invalidates concurrent `&mut` guards under Stacked Borrows.

```rust
// WRONG — data is inline, &MyArray covers the element bytes
unsafe impl ExternalAsMutPtr for MyArray {
    type Target = u8;
    unsafe fn as_mut_ptr(ptr: *mut Self) -> *mut u8 { ptr.cast() }
    unsafe fn as_mut_slice(ptr: *mut Self) -> *mut [u8] {
        let this = unsafe { &*ptr }; // SharedReadOnly over entire struct including data!
        core::ptr::slice_from_raw_parts_mut(ptr.cast(), this.len)
    }
    fn len(&self) -> usize { self.len }
}

// RIGHT — read length without creating a reference to the data
unsafe impl ExternalAsMutPtr for MyArray {
    type Target = u8;
    unsafe fn as_mut_ptr(ptr: *mut Self) -> *mut u8 { ptr.cast() }
    unsafe fn as_mut_slice(ptr: *mut Self) -> *mut [u8] {
        // Read the length field directly through the raw pointer
        let len = unsafe { core::ptr::addr_of!((*ptr).len).read() };
        core::ptr::slice_from_raw_parts_mut(ptr.cast(), len)
    }
    fn len(&self) -> usize { self.len }
}
```

For heap-backed containers (like `Vec`), creating `&Self` in `as_mut_slice` is fine — `&Self` only covers the container metadata (ptr, len, cap), and the heap data has separate provenance.

### `as_mut_ptr` that creates `&mut Self`

```rust
// WRONG — creates &mut Vec (Unique retag), invalidating concurrent readers
unsafe impl ExternalAsMutPtr for MyVec {
    type Target = u8;
    unsafe fn as_mut_ptr(ptr: *mut Self) -> *mut u8 {
        unsafe { (*ptr).inner.as_mut_ptr() } // &mut Vec → Unique retag on Vec struct
    }
    fn len(&self) -> usize { self.inner.len() }
}

// RIGHT — only create &Self (SharedReadOnly on Vec struct, not on heap data)
unsafe impl ExternalAsMutPtr for MyVec {
    type Target = u8;
    unsafe fn as_mut_ptr(ptr: *mut Self) -> *mut u8 {
        unsafe { (*ptr).inner.as_ptr().cast_mut() } // &Vec → SharedReadOnly, heap unaffected
    }
    fn len(&self) -> usize { self.inner.len() }
}
```

The difference is `as_mut_ptr(&mut self)` vs `as_ptr(&self).cast_mut()`. Both return the same pointer value, but the first creates `&mut Vec` which retags the struct with Unique provenance, invalidating any concurrent `&Vec` on other threads.

### Leaking guards to exhaust the slot pool

```rust
let dm = DisjointMut::new(vec![0u8; 1000]);
for i in 0..254 {
    std::mem::forget(dm.index_mut(i..i+1)); // slot never freed
}
// 255th borrow panics — all slots consumed by leaked guards
let _ = dm.index_mut(500..501); // panic!
```

Each `mem::forget`'d guard permanently consumes a slot. After 254 (64 inline + 190 overflow), the instance is bricked. This isn't UB, but it's a denial-of-service if guards are leaked in a loop.

### `dangerously_unchecked` with overlapping borrows

```rust
// This compiles and runs without panicking, but is UB
let dm = unsafe { DisjointMut::dangerously_unchecked(vec![0u8; 100]) };
let mut g1 = dm.index_mut(0..50);
let mut g2 = dm.index_mut(25..75); // overlaps! no runtime check to catch it
g1[30] = 1;
g2[5] = 2; // aliasing &mut — undefined behavior
```

With `dangerously_unchecked`, you get zero protection. The `unsafe` constructor is the safety boundary — if you use it, you're asserting that all borrows will be disjoint for the instance's entire lifetime.

## Raw pointer escape hatches

`as_mut_ptr()` and `as_mut_slice()` return raw pointers to element data, bypassing the tracker. These exist for FFI boundaries where assembly or C code needs a base pointer. The returned pointers require `unsafe` to dereference, so the caller is responsible for disjointness.

The primary API is `index()` / `index_mut()`, which return tracked guards. Prefer guards over raw pointers wherever possible.

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `std` | yes | Enables `std::thread::panicking()` for mutable guard poisoning on panic. |
| `aligned` | no | Aligned newtypes (`Align4`..`Align64`) and `AlignedVec32`/`AlignedVec64` for SIMD-friendly layout. |
| `pic-buf` | no | `PicBuf`: owned byte buffer with alignment offset for `DisjointMut`. |
| `zerocopy` | no | Zero-copy typed access via zerocopy's `AsBytes`/`FromBytes` traits. |

## `no_std` support

This crate is `no_std` compatible (requires `alloc`). Without `std`, the borrow tracker still works, but **poisoning is disabled** since `std::thread::panicking()` is unavailable. A panic while holding a mutable guard frees the borrow slot but doesn't prevent future access to potentially inconsistent data. Because elements are `Copy`, inconsistent data can't cause memory unsafety — only logic errors.

## Running tests under Miri

```bash
cargo +nightly miri test -p rav1d-disjoint-mut --all-features
MIRIFLAGS="-Zmiri-tree-borrows" cargo +nightly miri test -p rav1d-disjoint-mut --all-features
```

## License

BSD-2-Clause
