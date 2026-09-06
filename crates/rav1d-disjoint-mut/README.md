# rav1d-disjoint-mut

Runtime-checked disjoint mutable access to contiguous storage.

`DisjointMut` wraps a collection (`Vec<T>`, `Box<[T]>`, `[T; N]`) and allows non-overlapping mutable borrows through a shared `&` reference. Like `RefCell`, it enforces borrowing rules at runtime — but instead of whole-container borrows, it tracks *ranges* and panics only on truly overlapping access.

## Use case

Multiple threads need to write to dynamically chosen regions of a shared buffer. A fixed partition can use `split_at_mut` and scoped threads entirely in safe Rust. `DisjointMut` handles regions chosen during shared access by checking each borrow at runtime:

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

All container element types must be `Copy`. This excludes element destructors;
it does not permit torn reads or data races. A data race is undefined behavior
even for `u8`, and `Copy` types can still have validity requirements. Disjoint
regions must not overlap at their boundaries, and the tracker must synchronize
successive conflicting accesses. Safety comes from those guarantees, not from
the `Copy` bound.

### Borrow tracking

The default tracker assigns address blocks to independently locked shards.
Overlapping borrows always meet in a shared conflict domain; large ranges and
shard overflow use a wide path that coordinates with every active shard.
Strided-rectangle guards register exactly their rows and expose each row
separately, allowing other guards to access the gaps.

`DisjointMut::new` remains `const`, including for statics. It initializes one
boxed tracker on first use through `spin::Once`; simultaneous callers all use
the same tracker. `DisjointMut::new_eager` allocates immediately and avoids the
Once check on each borrow. `Default` and allocating slice constructors use the
eager path. Both constructors enforce the same borrowing rules.

An empty range consumes no record. Invalid or reversed ranges are refused
before creating a reference. Tracker capacity and placement are implementation
details, not a stable slot-count guarantee.

### Poisoning

Like `std::sync::Mutex`, `DisjointMut` poisons the data structure when a thread panics while holding a mutable borrow guard. After poisoning, all future borrow attempts panic.

Immutable guards do **not** poison on panic. Poisoning also triggers on out-of-bounds panics during indexing.

### Unchecked mode

`unsafe fn dangerously_unchecked()` creates an instance without runtime tracking. The caller must guarantee that all borrows are non-overlapping.

`new()` always creates a tracked instance.

### Open-ended ranges

Open-ended ranges like `5..` are clamped to the storage length for tracking.
Index validation still rejects out-of-bounds access before returning a guard.

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

### Leaking guards

Forgetting a guard retains its reservation. Later conflicting access is still
refused, even after moving the buffer. Repeated leaks can consume tracking
resources and cause allocation or admission failure; they must never permit a
conflicting reference. Memory safety does not require every guard to be dropped.

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
| `zerocopy` | no | Zero-copy typed access via zerocopy's `IntoBytes`/`FromBytes` traits. |

## `no_std` support

This crate supports `no_std` with `alloc`, including both constructors. Without
`std`, guard drop cannot detect thread unwinding and therefore does not poison
on that event. Indexing cleanup still poisons after an out-of-bounds panic.
Exclusion and reference validity remain enforced in both configurations.

## Running tests under Miri

```bash
cargo +nightly miri test -p rav1d-disjoint-mut --features aligned,pic-buf,zerocopy --no-fail-fast
MIRIFLAGS="-Zmiri-tree-borrows" cargo +nightly miri test -p rav1d-disjoint-mut --features aligned,pic-buf,zerocopy --no-fail-fast
```

## License

Triple-licensed: [BSD-2-Clause](../../COPYING) OR [MIT](LICENSE-MIT) OR [Apache-2.0](LICENSE-APACHE).

Started from `disjoint_mut.rs` in [rav1d](https://github.com/memorysafety/rav1d) (BSD-2-Clause, copyright VideoLAN, dav1d authors, and ISRG). The borrow tracker, poisoning, slot allocator, `no_std` support, and aligned storage are new; roughly 12% of the current code traces back to upstream type definitions and trait plumbing. BSD-2-Clause is included to honor that origin.
