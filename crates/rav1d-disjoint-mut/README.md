# rav1d-disjoint-mut

A provably safe abstraction for concurrent disjoint mutable access to contiguous storage.

`DisjointMut` wraps a collection (`Vec<T>`, `Box<[T]>`, `[T; N]`) and allows non-overlapping mutable borrows through a shared `&` reference. Like `RefCell`, it enforces borrowing rules at runtime — but instead of whole-container borrows, it tracks *ranges* and panics only on truly overlapping access.

## Use case

Multiple threads need to write to different regions of the same buffer simultaneously. Standard Rust won't let you split `&mut [T]` across threads without unsafe code. `DisjointMut` makes this safe:

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

## Safety model

Every `.index()` and `.index_mut()` call validates that the requested range doesn't overlap with any outstanding borrow. Mutable borrows conflict with everything; immutable borrows only conflict with mutable borrows (multiple readers are fine).

Guards act as locks — the borrow is tracked for the guard's lifetime and released on drop.

### Poisoning

Like `std::sync::Mutex`, `DisjointMut` poisons the data structure when a thread panics while holding a mutable borrow guard. After poisoning, all future borrow attempts panic. This prevents access to potentially corrupted data.

Immutable guards do **not** poison on panic (read-only access can't corrupt data).

### Unchecked mode

For audited hot paths, the `unchecked` feature enables `dangerously_unchecked()`, an unsafe constructor that skips all runtime tracking. The caller must guarantee disjointness.

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `std` | yes | Uses `parking_lot::Mutex` for tracking. Without this, uses `spin::Mutex`. |
| `unchecked` | no | Enables `dangerously_unchecked()` constructor (unsafe, no tracking). |

## `no_std` support

This crate is `no_std` compatible (requires `alloc`). Without the `std` feature, it uses a spin-lock for borrow tracking and panic-on-drop poisoning is disabled (requires `std::thread::panicking()`).

## External types

Implement the `ExternalAsMutPtr` unsafe trait to use your own container with `DisjointMut`. See the trait docs for safety requirements — the key constraint is that `as_mut_ptr` must never create `&mut` references to the container or its data.

## Zerocopy integration

For `DisjointMut<T>` where `T` stores `u8` data, the `mut_slice_as`, `mut_element_as`, `slice_as`, and `element_as` methods provide zero-copy typed access via zerocopy's `AsBytes`/`FromBytes` traits.

## Running tests under Miri

The soundness tests should pass under both Stacked Borrows and Tree Borrows:

```bash
cargo +nightly miri test -p rav1d-disjoint-mut
MIRIFLAGS="-Zmiri-tree-borrows" cargo +nightly miri test -p rav1d-disjoint-mut
```

## License

BSD-2-Clause
