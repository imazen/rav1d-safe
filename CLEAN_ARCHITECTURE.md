# Clean Architecture - Avoiding cfg_attr Nightmares

## Problem

Scattering `#[cfg(feature = "quite-safe")]` everywhere creates unmaintainable code:
```rust
// BAD - cfg_attr nightmare:
#[cfg(feature = "quite-safe")]
use std::sync::Arc;
#[cfg(not(feature = "quite-safe"))]
use std::rc::Rc as Arc;

#[cfg(feature = "quite-safe")]
fn spawn_workers(...) { ... }
#[cfg(not(feature = "quite-safe"))]
fn spawn_workers(...) { /* no-op */ }
```

## Solution: Module Structure with Re-exports

Use separate modules for each safety level, re-export the active one.

### Pattern 1: Type Aliases Module

**src/sync_primitives.rs:**
```rust
//! Synchronized primitives - switches between single/multi-threaded

#[cfg(feature = "quite-safe")]
mod multi_threaded {
    pub use std::sync::{Arc, Mutex};
    pub use std::sync::atomic::{AtomicU32, AtomicBool, Ordering};
    pub use parking_lot::Mutex as ParkingMutex;
}

#[cfg(not(feature = "quite-safe"))]
mod single_threaded {
    pub use std::rc::Rc as Arc;
    pub use std::cell::RefCell as Mutex;
    
    // Single-threaded atomic stub
    #[derive(Debug)]
    pub struct AtomicU32(std::cell::Cell<u32>);
    
    impl AtomicU32 {
        pub const fn new(val: u32) -> Self { Self(std::cell::Cell::new(val)) }
        pub fn load(&self, _: Ordering) -> u32 { self.0.get() }
        pub fn store(&self, val: u32, _: Ordering) { self.0.set(val) }
        pub fn fetch_add(&self, val: u32, _: Ordering) -> u32 {
            let old = self.0.get();
            self.0.set(old.wrapping_add(val));
            old
        }
    }
    
    pub use AtomicU32 as AtomicBool; // Stub
    
    // Re-export Ordering even though we don't use it
    pub use std::sync::atomic::Ordering;
    
    // ParkingMutex stub
    pub use RefCell as ParkingMutex;
}

// Re-export the active implementation
#[cfg(feature = "quite-safe")]
pub use multi_threaded::*;

#[cfg(not(feature = "quite-safe"))]
pub use single_threaded::*;

// Now anywhere in the codebase just:
// use crate::sync_primitives::{Arc, Mutex, AtomicU32};
// No cfg_attr needed!
```

**Usage in lib.rs:**
```rust
use crate::sync_primitives::{Arc, Mutex, AtomicU32};

// Works in both single and multi-threaded modes!
struct Context {
    data: Arc<Mutex<Data>>,
    counter: AtomicU32,
}
```

### Pattern 2: Separate Implementation Modules

**src/decoder/mod.rs:**
```rust
#[cfg(feature = "quite-safe")]
mod multi_threaded;

#[cfg(not(feature = "quite-safe"))]
mod single_threaded;

// Re-export the active decoder
#[cfg(feature = "quite-safe")]
pub use multi_threaded::DecoderImpl;

#[cfg(not(feature = "quite-safe"))]
pub use single_threaded::DecoderImpl;

// Public API stays the same
pub struct Decoder {
    inner: DecoderImpl,
}

impl Decoder {
    pub fn new() -> Self {
        Self { inner: DecoderImpl::new() }
    }
    
    pub fn decode(&mut self, data: &[u8]) -> Result<Frame> {
        self.inner.decode(data)
    }
}
```

**src/decoder/multi_threaded.rs:**
```rust
use crate::sync_primitives::{Arc, Mutex};
use std::thread;

pub(super) struct DecoderImpl {
    workers: Vec<thread::JoinHandle<()>>,
    // ... multi-threaded implementation
}

impl DecoderImpl {
    pub(super) fn new() -> Self {
        // Spawn worker threads
        ...
    }
}
```

**src/decoder/single_threaded.rs:**
```rust
use crate::sync_primitives::{Arc, Mutex};

pub(super) struct DecoderImpl {
    // No workers, same API
}

impl DecoderImpl {
    pub(super) fn new() -> Self {
        // Single-threaded, no workers
        ...
    }
}
```

### Pattern 3: Conditional Slice Access

**src/safe_simd/pixel_access.rs** (already exists!):
```rust
#[inline(always)]
pub fn row_slice(buf: &[u8], offset: usize, len: usize) -> &[u8] {
    #[cfg(feature = "unchecked")]
    unsafe { buf.get_unchecked(offset..offset + len) }
    
    #[cfg(not(feature = "unchecked"))]
    &buf[offset..offset + len]
}
```

**Usage - no cfg_attr needed:**
```rust
use pixel_access::row_slice;

fn process(buf: &[u8], stride: usize) {
    for y in 0..height {
        let row = row_slice(buf, y * stride, width);
        // Works in both checked and unchecked modes!
    }
}
```

### Pattern 4: FFI Wrapper Module

**src/safe_simd/mc.rs:**
```rust
// Safe inner implementation - always compiled
fn mc_inner(dst: &mut [u8], src: &[u8], ...) {
    // Safe SIMD implementation
}

// FFI wrappers in separate module
#[cfg(feature = "c-ffi")]
mod ffi {
    use super::mc_inner;
    use crate::ffi_safe::FFISafe;
    
    pub unsafe extern "C" fn mc_ffi(
        dst: *mut u8,
        dst_len: usize,
        src: *const u8,
        src_len: usize,
        ...
    ) {
        let dst = unsafe { std::slice::from_raw_parts_mut(dst, dst_len) };
        let src = unsafe { std::slice::from_raw_parts(src, src_len) };
        super::mc_inner(dst, src, ...)
    }
}

#[cfg(feature = "c-ffi")]
pub use ffi::*;
```

## Complete Module Structure

```
src/
  lib.rs                    - Main entry, uses sync_primitives
  sync_primitives.rs        - Arc/Mutex/AtomicU32 re-exports
  
  decoder/
    mod.rs                  - Re-exports active impl
    single_threaded.rs      - Rc/RefCell implementation
    multi_threaded.rs       - Arc/Mutex implementation
  
  safe_simd/
    pixel_access.rs         - Slice helpers (checked/unchecked)
    
    mc.rs
      fn mc_inner(...)      - Safe implementation
      mod ffi { ... }       - #[cfg(c-ffi)] wrappers
    
    ipred.rs
      fn ipred_inner(...)
      mod ffi { ... }
```

## Benefits

1. ✅ **Zero cfg_attr in most code** - Just import from sync_primitives
2. ✅ **Clear separation** - Each safety level in its own module
3. ✅ **Easy auditing** - Look at single_threaded.rs vs multi_threaded.rs
4. ✅ **Testable** - Can test both implementations independently
5. ✅ **Maintainable** - No scattered conditional compilation

## Migration Steps

### Step 1: Create sync_primitives.rs
```rust
// Single file with all Arc/Mutex/AtomicU32 re-exports
```

### Step 2: Replace imports throughout codebase
```bash
# Find all Arc/Mutex imports
rg "use std::sync::" --type rust

# Replace with:
# use crate::sync_primitives::{Arc, Mutex, ...};
```

### Step 3: Extract multi-threaded code
```bash
# Move worker thread spawning to decoder/multi_threaded.rs
# Create decoder/single_threaded.rs stub
```

### Step 4: Gate FFI wrappers
```rust
// In each SIMD module:
#[cfg(feature = "c-ffi")]
mod ffi { ... }
```

## Example: Complete mc.rs Structure

```rust
//! Motion compensation - safe SIMD implementation
#![cfg_attr(not(feature = "quite-safe"), forbid(unsafe_code))]

use crate::safe_simd::pixel_access::{row_slice, row_slice_mut};

// ===== SAFE INNER FUNCTIONS (always compiled) =====

fn mc_put_8tap_inner(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
) {
    for y in 0..h {
        let dst_row = row_slice_mut(dst, y * dst_stride, w);
        let src_row = row_slice(src, y * src_stride, w);
        // SIMD implementation using safe intrinsics
    }
}

// ===== DISPATCH (calls inner functions) =====

#[cfg(feature = "quite-safe")]
pub fn mc_put_8tap_dispatch(...) -> bool {
    if let Ok(token) = Desktop64::summon() {
        mc_put_8tap_avx2(token, ...);
        true
    } else {
        false
    }
}

// ===== FFI WRAPPERS (only when c-ffi enabled) =====

#[cfg(feature = "c-ffi")]
mod ffi {
    use super::*;
    
    #[no_mangle]
    pub unsafe extern "C" fn dav1d_mc_put_8tap(
        dst: *mut u8,
        dst_stride: isize,
        src: *const u8,
        src_stride: isize,
        w: c_int,
        h: c_int,
    ) {
        let dst_len = (h as usize * dst_stride as usize) + w as usize;
        let src_len = (h as usize * src_stride as usize) + w as usize;
        
        let dst = unsafe { slice::from_raw_parts_mut(dst, dst_len) };
        let src = unsafe { slice::from_raw_parts(src, src_len) };
        
        super::mc_put_8tap_inner(
            dst, dst_stride as usize,
            src, src_stride as usize,
            w as usize, h as usize
        );
    }
}

#[cfg(feature = "c-ffi")]
pub use ffi::*;
```

## Summary

**Key principle:** Isolate conditional compilation to module boundaries, not scattered throughout code.

- ✅ `sync_primitives.rs` - Single file for Arc/Mutex switching
- ✅ `decoder/{single,multi}_threaded.rs` - Separate impls
- ✅ `pixel_access.rs` - Checked/unchecked slice access
- ✅ `mod ffi { ... }` - FFI wrappers gated at module level

This gives us clean code with minimal cfg_attr usage!
