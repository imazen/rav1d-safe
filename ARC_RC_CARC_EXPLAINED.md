# Arc/Rc/CArc Explained - All Contexts in rav1d-safe

## Overview: Three Types of Reference Counting

rav1d-safe uses three different reference-counted pointer types depending on context:

1. **`std::sync::Arc<T>`** - Thread-safe, atomic reference counting
2. **`std::rc::Rc<T>`** - Single-threaded, non-atomic reference counting  
3. **`CArc<T>`** - Custom Arc with C FFI compatibility

## 1. std::sync::Arc<T> - Multi-threaded Context

### What It Is
Standard library's **atomic reference counted** smart pointer.
- Thread-safe (implements Send + Sync)
- Uses atomic operations for ref counting (slower than Rc)
- Allows sharing data across threads

### Where It's Used in rav1d-safe

**Context sharing across worker threads:**
```rust
// src/lib.rs, src/internal.rs
pub struct Rav1dContext {
    task_thread: Arc<TaskThreadData>,  // Shared across workers
    fc: Box<[Rav1dFrameContext]>,      // Frame contexts
    tc: Box<[Rav1dContextTaskThread]>, // Task threads
    // ...
}

// Worker threads hold Arc<Rav1dContext>
pub(crate) fn rav1d_worker_task(task_thread: Arc<Rav1dTaskContextTaskThread>) {
    let c = &*task_thread.c.lock().take().unwrap(); // Arc<Rav1dContext>
    // Worker can access shared context
}
```

**Picture pool:**
```rust
// src/picture.rs
pub(crate) struct PicturePool {
    pub entries: Arc<Mutex<Vec<PicturePoolEntry>>>,
}
```

**Managed API:**
```rust
// src/managed.rs
pub struct Decoder {
    ctx: Arc<Rav1dContext>,  // Multiple Frames may reference same context
    worker_handles: Vec<JoinHandle<()>>,
}
```

### When quite-safe Feature is Enabled

`Arc` is allowed when `feature = "quite-safe"` is enabled (Level 1 safety).

```rust
#[cfg(feature = "quite-safe")]
use std::sync::Arc;
```

## 2. std::rc::Rc<T> - Single-threaded Context

### What It Is
Standard library's **non-atomic reference counted** smart pointer.
- NOT thread-safe (doesn't implement Send or Sync)
- Faster than Arc (no atomic operations)
- Only for single-threaded use

### Where It Will Be Used (Target Architecture)

When `quite-safe` is **disabled** (Level 0 forbid_unsafe), we'll switch to single-threaded alternatives:

```rust
// Target: src/sync_primitives.rs
#[cfg(not(feature = "quite-safe"))]
mod single_threaded {
    pub use std::rc::Rc as Arc;  // Type alias!
    pub use std::cell::RefCell as Mutex;
}

// Then throughout codebase:
use crate::sync_primitives::Arc;  // Will be Rc or Arc depending on feature

struct Context {
    shared_data: Arc<Data>,  // Rc in forbid_unsafe mode, Arc in quite-safe mode
}
```

### Not Yet Implemented

Currently, Arc is used unconditionally. Migration to conditional Rc/Arc is planned but not done.

## 3. CArc<T> - C FFI Compatible Arc

### What It Is
**Custom atomic reference counted pointer** designed for C FFI compatibility.

Key features:
- Wraps `Arc<Pin<CBox<T>>>`
- Stores stable pointer for performance (avoids double indirection)
- Supports custom deallocation functions (C `free` callback)
- Allows wrapping C-allocated memory

### Implementation Details

```rust
// src/c_arc.rs
pub struct CArc<T: ?Sized> {
    owner: Arc<Pin<CBox<T>>>,      // Actual ownership
    stable_ref: StableRef<T>,      // Cached pointer for perf
    
    #[cfg(debug_assertions)]
    base_stable_ref: StableRef<T>, // Validation in debug
}

struct StableRef<T: ?Sized>(NonNull<T>);  // Raw pointer with stable address
```

**Why the complexity?**
- `Arc<Pin<CBox<T>>>` would require two indirections: Arc → CBox → T
- Storing `stable_ref` inline allows direct access: CArc → T (one indirection)
- Soundness: `Pin<CBox<T>>` ensures stable address, so storing raw pointer is safe

### Where It's Used

**1. Data buffers (OBU bitstream):**
```rust
// include/dav1d/data.rs
pub(crate) struct Rav1dData {
    pub data: Option<CArc<[u8]>>,  // Bitstream data
    pub m: Rav1dDataProps,
}

// C FFI version:
pub struct Dav1dData {
    pub data: Option<NonNull<u8>>,       // Raw pointer
    pub sz: usize,
    pub r#ref: Option<RawCArc<[u8]>>,    // CArc as opaque handle
    pub m: Dav1dDataProps,
}
```

**2. Wrapping C-allocated memory:**
```rust
// src/data.rs
impl Rav1dData {
    /// Wrap C-allocated data with custom free function
    #[cfg(feature = "c-ffi")]
    pub unsafe fn wrap(
        data: NonNull<[u8]>,
        free_callback: Option<FnFree>,
        cookie: Option<SendSyncNonNull<c_void>>,
    ) -> Result<Self> {
        let free = Free { free: free_callback?, cookie };
        let data = unsafe { CBox::from_c(data, free) };  // Wrap in CBox
        let data = CArc::wrap(data)?;                    // Wrap in CArc
        Ok(data.into())
    }
}
```

**3. Picture data:**
```rust
// src/picture.rs
// Pictures may be allocated by custom allocators or C code
// CArc allows reference counting regardless of allocation source
```

### CArc vs Arc Comparison

| Feature | std::sync::Arc | CArc |
|---------|---------------|------|
| Thread-safe | ✅ Yes | ✅ Yes |
| Atomic refcount | ✅ Yes | ✅ Yes (via inner Arc) |
| C FFI compatible | ❌ No | ✅ Yes |
| Custom deallocator | ❌ No | ✅ Yes (via CBox) |
| Overhead | 1 allocation | 2 allocations (Arc + CBox) |
| Deref performance | Direct | Cached stable_ref (fast) |
| Use case | Rust-only | C interop, custom alloc |

## CBox<T> - C-Compatible Box

### What It Is
Custom `Box`-like type that supports **custom deallocation functions**.

```rust
// src/c_box.rs
pub struct CBox<T: ?Sized> {
    ptr: NonNull<T>,
    free: Free,  // Custom free function + cookie
}

pub struct Free {
    pub free: FnFree,                      // fn(*mut c_void, *mut c_void)
    pub cookie: Option<SendSyncNonNull<c_void>>,
}

pub type FnFree = unsafe extern "C" fn(*mut c_void, *mut c_void);
```

**Purpose:** Allows Rust to manage C-allocated memory with C's `free` semantics.

## RawCArc<T> - Opaque CArc Handle

### What It Is
Type-erased `CArc` for C FFI (opaque pointer pattern).

```rust
// src/c_arc.rs
pub type RawCArc<T> = NonNull<T>;

impl<T: ?Sized> CArc<T> {
    pub fn into_raw(self) -> RawCArc<T> {
        // Convert to raw pointer, forget self (don't drop)
    }
    
    pub unsafe fn from_raw(raw: RawCArc<T>) -> Self {
        // Reconstruct CArc from raw pointer
    }
}
```

**Usage in C FFI:**
```rust
// C sees: void* opaque_ref
// Rust sees: RawCArc<[u8]>
pub struct Dav1dData {
    pub r#ref: Option<RawCArc<[u8]>>,  // Opaque to C
}
```

## Context: When Each Type Is Used

### During Decoding (Multi-threaded)

```
User Input (OBU data)
    ↓
CArc<[u8]> ← Wraps input bitstream (may be C-allocated)
    ↓
Rav1dData { data: CArc<[u8]> }
    ↓
Arc<Rav1dContext> ← Shared across worker threads
    ↓
Worker threads decode frames
    ↓
Rav1dPicture with Arc<PictureData>
    ↓
Frame { data: DisjointMut<Arc<...>> } ← Managed API
```

### Key Transitions

**C → Rust (Input):**
```
C pointer + free callback
    ↓ CBox::from_c()
CBox<[u8]>
    ↓ CArc::wrap()
CArc<[u8]>
    ↓ Rav1dData::from()
Rav1dData { data: CArc<[u8]> }
```

**Rust → C (Output):**
```
CArc<[u8]>
    ↓ CArc::into_raw()
RawCArc<[u8]> (opaque NonNull<[u8]>)
    ↓
Dav1dData { r#ref: RawCArc<[u8]> }
    ↓
C receives void* opaque handle
```

## Safety Levels and Arc Types

### Level 0: forbid_unsafe (Target - Not Implemented)
- Use `Rc<T>` instead of `Arc<T>`
- Single-threaded only
- No `CArc` (requires unsafe for FFI)

### Level 1: quite-safe (Current Default)
- Use `Arc<T>` for multi-threading
- `CArc<T>` available
- Threading allowed

### Level 2+: unchecked, c-ffi, asm
- All arc types available
- `CArc` fully utilized for C FFI

## Common Patterns

### Pattern 1: Cloning Contexts
```rust
// Worker threads share context
let ctx_clone = Arc::clone(&self.ctx);
thread::spawn(move || {
    worker_task(ctx_clone);
});
```

### Pattern 2: Wrapping C Data
```rust
// C gives us data + free callback
let c_data: *mut u8;
let c_len: usize;
let c_free: extern "C" fn(*mut c_void, *mut c_void);

// Wrap in CArc
let data = NonNull::slice_from_raw_parts(NonNull::new(c_data).unwrap(), c_len);
let rav1d_data = unsafe { 
    Rav1dData::wrap(data, Some(c_free), None)
}?;
```

### Pattern 3: Conditional Compilation (Planned)
```rust
// sync_primitives.rs (to be created)
#[cfg(feature = "quite-safe")]
pub use std::sync::Arc;

#[cfg(not(feature = "quite-safe"))]
pub use std::rc::Rc as Arc;

// Usage everywhere:
use crate::sync_primitives::Arc;  // Switches based on feature
```

## Summary Table

| Type | Thread-Safe | C FFI | Custom Free | Use Case |
|------|-------------|-------|-------------|----------|
| `Arc<T>` | ✅ Atomic | ❌ No | ❌ No | Multi-threaded Rust |
| `Rc<T>` | ❌ No | ❌ No | ❌ No | Single-threaded Rust |
| `CArc<T>` | ✅ Atomic | ✅ Yes | ✅ Yes | C interop, custom alloc |

## Migration Plan: Arc → Rc Conditional

**Step 1:** Create `src/sync_primitives.rs`
```rust
#[cfg(feature = "quite-safe")]
pub use std::sync::{Arc, Mutex};

#[cfg(not(feature = "quite-safe"))]
pub use std::rc::Rc as Arc;
#[cfg(not(feature = "quite-safe"))]
pub use std::cell::RefCell as Mutex;
```

**Step 2:** Replace all `use std::sync::Arc` with `use crate::sync_primitives::Arc`

**Step 3:** Conditionally disable threading when quite-safe disabled
```rust
#[cfg(feature = "quite-safe")]
fn spawn_workers(...) { ... }

#[cfg(not(feature = "quite-safe"))]
fn spawn_workers(...) { /* no-op */ }
```

**Result:** Codebase works with both Rc (single-threaded) and Arc (multi-threaded) via feature flag!

## Key Takeaways

1. **Arc = multi-threaded** (atomic refcount, thread-safe)
2. **Rc = single-threaded** (faster, no atomics)
3. **CArc = C FFI** (custom free, wraps C memory)
4. **Current:** Uses Arc unconditionally (requires quite-safe)
5. **Target:** Conditional Rc/Arc via sync_primitives module
6. **CArc is separate** - Used for FFI regardless of Arc/Rc choice
