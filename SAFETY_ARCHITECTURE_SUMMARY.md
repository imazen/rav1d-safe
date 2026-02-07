# Progressive Safety Architecture - Implementation Summary

## What Was Done ✅

### 1. Feature Dependency Chain Implemented

**Cargo.toml:**
```toml
[features]
default = ["bitdepth_8", "bitdepth_16"]  # Changed: removed asm from default

# Progressive safety levels:
quite-safe = []                          # Level 1: Threading + sound abstractions
unchecked = ["quite-safe"]               # Level 2: Unchecked slice access
c-ffi = ["unchecked"]                    # Level 3: C FFI API
asm = ["c-ffi"]                          # Level 4: Hand-written assembly
```

### 2. Crate-Level Safety Enforcement

**src/lib.rs:**
```rust
#![cfg_attr(not(feature = "quite-safe"), forbid(unsafe_code))]
```

When `quite-safe` is disabled, the compiler enforces zero unsafe code.

### 3. Documentation Updated

**CLAUDE.md now includes:**
- Safety level descriptions (0-4)
- Feature dependency chain
- Build commands for each level
- Code structure guidelines
- Audit guidance for reviewers
- Migration checklist for modules

## Safety Levels Explained

### Level 0: Default (forbid_unsafe)
```bash
cargo build --no-default-features --features "bitdepth_8,bitdepth_16"
```
- ✅ `#![forbid(unsafe_code)]` enforced
- ✅ Single-threaded (Rc/RefCell)
- ✅ Bounds-checked slices
- ⚠️ **NOT YET IMPLEMENTED** - requires Arc→Rc migration

### Level 1: quite-safe
```bash
cargo build --features "quite-safe,bitdepth_8,bitdepth_16"
```
- ✅ Arc/Mutex/AtomicU32 allowed
- ✅ Multi-threaded
- ✅ Bounds-checked slices
- ✅ **THIS IS THE CURRENT PRACTICAL DEFAULT**

### Level 2: unchecked
```bash
cargo build --features "unchecked,bitdepth_8,bitdepth_16"
```
- ✅ Everything from quite-safe
- ⚠️ Unchecked slice access (via pixel_access helpers)
- ⚠️ debug_assert! still validates in debug builds

### Level 3: c-ffi
```bash
cargo build --features "c-ffi,bitdepth_8,bitdepth_16"
```
- ✅ Everything from unchecked
- ⚠️ `unsafe extern "C"` FFI functions
- ⚠️ dav1d_* C API available

### Level 4: asm
```bash
cargo build --features "asm,bitdepth_8,bitdepth_16"
```
- ✅ Everything from c-ffi
- ⚠️ Hand-written x86_64/aarch64 assembly

## What's NOT Yet Implemented ⚠️

### Default forbid_unsafe Mode Doesn't Build

**Problem:** Code currently uses Arc/Mutex unconditionally

**Solution needed:**
```rust
// In src/internal.rs, src/lib.rs, etc.
#[cfg(feature = "quite-safe")]
use std::sync::{Arc, Mutex, AtomicU32};

#[cfg(not(feature = "quite-safe"))]
use std::{
    rc::Rc as Arc,
    cell::RefCell as Mutex,
    // No AtomicU32 equivalent for single-threaded
};
```

**Work estimate:** ~1 week to conditionally compile threading code

### Modules Not Yet Migrated

~20 safe_simd modules still need:
1. Inner functions changed from raw pointers → slices
2. `#![cfg_attr(not(feature = "quite-safe"), forbid(unsafe_code))]` added
3. FFI wrappers gated with `#[cfg(feature = "c-ffi")]`

## Next Steps

### Option A: Implement forbid_unsafe Default
1. Audit all Arc/Mutex/AtomicU32 usage
2. Create single-threaded alternatives with Rc/RefCell
3. Conditionally compile based on `quite-safe` feature
4. Update Decoder to not spawn threads when quite-safe disabled
5. Test default mode builds and passes tests

**Estimate:** 5-7 days

### Option B: Migrate SIMD Modules First
1. Pick a module (e.g., mc.rs)
2. Change inner functions to take slices
3. Use pixel_access helpers
4. Gate FFI wrappers with `#[cfg(feature = "c-ffi")]`
5. Add `#![cfg_attr(not(feature = "quite-safe"), forbid(unsafe_code))]`
6. Test in all safety levels
7. Repeat for remaining ~19 modules

**Estimate:** 2-5 days for all modules

### Option C: Do Both (Recommended Order)

**Week 1:** Migrate SIMD modules to slices
- Immediate benefit: better APIs, easier auditing
- Reduces unsafe code surface area
- Makes FFI boundary clear

**Week 2:** Implement forbid_unsafe default
- Conditional threading compilation
- Single-threaded safe alternatives
- Complete the safety level architecture

## How to Audit the Safety Levels

For code reviewers:

1. **Check feature gates are correct:**
   ```bash
   rg "#\[cfg\(feature = \"quite-safe\"\)\]" | less
   rg "#\[cfg\(feature = \"c-ffi\"\)\]" | less
   ```

2. **Verify default mode forbids unsafe:**
   ```bash
   cargo build --no-default-features --features "bitdepth_8"
   # Should fail until Arc→Rc migration done
   ```

3. **Verify quite-safe allows only sound abstractions:**
   ```bash
   cargo build --features "quite-safe,bitdepth_8"
   # Should only use Arc/Mutex/AtomicU32, no other unsafe
   ```

4. **Check FFI wrappers are gated:**
   ```bash
   rg "unsafe extern \"C\"" | grep -v "#\[cfg\(feature"
   # Should return no results (all FFI should be gated)
   ```

## Benefits of This Architecture

1. ✅ **Auditable** - Clear separation of safety levels
2. ✅ **Progressive** - Opt-in to less safety for more performance
3. ✅ **Testable** - Can test all levels independently
4. ✅ **Fuzzing-friendly** - Default mode ideal for fuzzing
5. ✅ **Trust minimization** - Use only the features you need

## Current Build Status

```bash
# Does NOT work yet (needs Arc→Rc migration):
cargo build --no-default-features --features "bitdepth_8,bitdepth_16"

# Works now (practical default):
cargo build --features "quite-safe,bitdepth_8,bitdepth_16"

# Works (adds unchecked slice access):
cargo build --features "unchecked,bitdepth_8,bitdepth_16"

# Works (adds C FFI):
cargo build --features "c-ffi,bitdepth_8,bitdepth_16"

# Works (adds asm):
cargo build --features "asm,bitdepth_8,bitdepth_16"
```
