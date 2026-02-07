# DisjointMut Analysis - Critical Safety Examination

## What DisjointMut Is

**Purpose:** Allows concurrent mutable access to **disjoint** regions of a collection from multiple threads.

**Key insight:** If you can prove accesses are disjoint (non-overlapping), concurrent mutation is safe - even from `&DisjointMut` (shared reference).

```rust
// Multiple threads can each write to different parts:
thread1: disjoint_vec.index_mut(0..100)   // mutable access to [0..100)
thread2: disjoint_vec.index_mut(100..200) // mutable access to [100..200)
// Safe! Non-overlapping regions
```

## Implementation Strategy

### Debug Mode (cfg(debug_assertions))
- **Tracks all active borrows** in a bounds check structure
- **Panics on overlap** between mutable borrows
- **Allows** multiple immutable borrows (even if overlapping)
- **Zero cost at runtime** after bounds check (guard just holds slice)

### Release Mode (production)
- **NO RUNTIME CHECKING** - relies entirely on manual correctness
- **Zero overhead** - `repr(transparent)` guard is just a slice
- **Unsafe if used incorrectly** - overlapping mutable borrows = UB

## Safety Guarantees & Assumptions

### Critical Safety Contract (from Sync impl docs)

```rust
unsafe impl<T: ?Sized + AsMutPtr + Sync> Sync for DisjointMut<T> {}
```

**The safety relies on THREE key assumptions:**

1. **Disjointness manually guaranteed** - `pub(crate)` means we control all usage
2. **Provenanceless data** - `AsMutPtr::Target` has no pointers/references
3. **Data races = wrong results, not UB** - because types are provenanceless

### What "Provenanceless" Means

From the docs:
> Furthermore, all `T`s used have [`AsMutPtr::Target`]s
> that are provenanceless, i.e. they have no internal references or pointers
> or integers that hold pointer provenance.
> Thus, a data race due the lack of runtime disjointness checking in release mode
> would only result in wrong results, and cannot result in memory safety.

**Currently used types:**
- `u8`, `u16` - Primitive integers (no provenance) ✅
- `[u8; N]`, `[u16; N]` - Arrays of primitives (no provenance) ✅
- `AlignedVec64<u8>` - Vec of primitives (Vec itself has provenance, but elements don't) ⚠️

## Where DisjointMut Is Used

### 1. Pixel Buffers (Primary Use Case)

```rust
// Loop filter destination buffer
dst: WithOffset<WithStride<&DisjointMut<AlignedVec64<u8>>>>

// CDEF top buffer
top: &DisjointMut<AlignedVec64<u8>>

// Loop restoration lpf buffer  
lpf: &DisjointMut<AlignedVec64<u8>>
```

**Usage pattern:**
- Different threads process different tile rows/cols
- Each thread indexes into disjoint regions of pixel buffer
- Must manually ensure no overlap (done via tile boundaries)

### 2. Loop Filter Levels

```rust
lvl: WithOffset<&DisjointMut<Vec<u8>>>
```

**Usage pattern:**
- Each thread writes loop filter levels for its tiles
- Tile boundaries ensure disjointness

### 3. Pixels Trait Implementation

```rust
impl<T: AsMutPtr<Target = u8>> Pixels for DisjointMut<T> {
    // Provides pixel access interface for DisjointMut-wrapped buffers
}
```

## Critical Questions & Concerns

### ⚠️ Concern 1: Release Mode Has No Checking

**Problem:** In release builds, overlapping mutable borrows are UB but unchecked.

**Mitigations:**
- `pub(crate)` - all uses are internal, we control them
- Debug builds catch bugs during development
- Extensive testing with debug assertions enabled

**Recommendation:**
- Keep thorough debug-mode tests
- Consider fuzzing in debug mode to catch disjointness violations
- Document every DisjointMut usage with disjointness proof

### ⚠️ Concern 2: Tile Boundary Calculations

**Problem:** Disjointness relies on correct tile boundary math.

**Current approach:**
- Tile boundaries calculated in decode logic
- Each thread gets non-overlapping tile regions
- Math errors → overlapping regions → UB in release mode

**Recommendation:**
- Audit all tile boundary calculations
- Add debug assertions at indexing sites
- Consider property-based testing for tile math

### ⚠️ Concern 3: Vec Provenance

**Question:** Does `DisjointMut<Vec<u8>>` violate provenanceless assumption?

**Analysis:**
```rust
unsafe impl<V> AsMutPtr for Vec<V> {
    unsafe fn as_mut_ptr(ptr: *mut Self) -> *mut V {
        unsafe { (*ptr).as_mut_ptr() }
    }
}
```

- Vec **itself** has provenance (owns heap allocation)
- But `AsMutPtr::Target = u8` - the **elements** are provenanceless
- DisjointMut provides access to **elements**, not Vec itself
- **Verdict:** Safe, because we never race on Vec's internal pointers

### ⚠️ Concern 4: AlignedVec Alignment

**Question:** Does alignment affect safety?

```rust
DisjointMut<AlignedVec64<u8>>
```

**Analysis:**
- AlignedVec is just Vec with alignment guarantee
- Elements are still `u8` (provenanceless)
- Alignment is a property of the allocation, not the data
- **Verdict:** Safe for same reason as Vec

### ✅ Concern 5: Debug Mode Overhead

**Not actually a concern:**
- Debug builds are for testing, not production
- Bounds checking is exactly what we want in debug

## Current Safety Status

### What's Safe ✅

1. **Element types are provenanceless** - u8, u16, fixed arrays
2. **pub(crate) visibility** - all uses are auditable
3. **Debug mode catches bugs** - overlaps panic during development
4. **Purpose-designed** - solves real problem (tile-parallel decoding)

### What's Risky ⚠️

1. **Release mode unchecked** - relies on manual correctness
2. **Tile math complexity** - sophisticated boundary calculations
3. **No static verification** - Rust can't prove disjointness
4. **Silent UB if wrong** - incorrect tile math = undefined behavior

### What Needs Audit 🔍

1. **All tile boundary calculations** - verify non-overlapping
2. **All .index_mut() call sites** - check disjointness reasoning
3. **Thread spawning logic** - verify tile assignment
4. **Edge cases** - frame boundaries, non-divisible sizes

## Recommendations

### 1. Document Disjointness Proofs

At every `index_mut` call site, add comment proving disjointness:

```rust
// DISJOINT: Thread i processes tiles [i*n..(i+1)*n),
// no overlap possible since i < num_threads
let region = disjoint_buf.index_mut(start..end);
```

### 2. Add Bounds Assertions

Even in release mode, add strategic assertions:

```rust
debug_assert!(start < end);
debug_assert!(end <= total_len);
debug_assert_eq!(end - start, expected_tile_size);
```

### 3. Fuzz Test in Debug Mode

Create fuzzer that:
- Randomly assigns tiles to threads
- Runs in debug mode (catches overlaps)
- Validates output correctness

### 4. Consider Runtime Option

Add feature flag for optional bounds checking in release:

```rust
#[cfg(any(debug_assertions, feature = "paranoid"))]
// Bounds checking enabled

#[cfg(not(any(debug_assertions, feature = "paranoid")))]
// No bounds checking (current release behavior)
```

### 5. Static Analysis

Use `cargo-careful` or Miri to detect potential issues:
```bash
cargo +nightly miri test --features "bitdepth_8"
```

## Alternative Approaches (for comparison)

### Alternative 1: RwLock-based

**Replace with:**
```rust
Arc<RwLock<Vec<u8>>>
```

**Pros:**
- Runtime checked in all builds
- No manual disjointness reasoning

**Cons:**
- Massive performance hit (lock contention)
- Defeats purpose of parallelism

### Alternative 2: Separate Buffers

**Replace with:**
```rust
// Each thread gets its own buffer
Vec<Vec<u8>>  // One per thread
```

**Pros:**
- No shared mutation, trivially safe

**Cons:**
- More memory usage
- Complex merge logic
- Doesn't match frame buffer semantics

### Alternative 3: Owned Regions (Crossbeam)

**Use crossbeam-utils `Scope`:**
```rust
crossbeam::scope(|s| {
    for chunk in buffer.chunks_mut(n) {
        s.spawn(move |_| process(chunk));
    }
});
```

**Pros:**
- Compiler-verified disjointness
- No custom unsafe code

**Cons:**
- Requires known-size chunks
- Harder to express complex tiling patterns

**Verdict:** DisjointMut is the right choice for this use case, but needs careful audit.

## Migration to Safer Alternative

If we wanted to reduce unsafe surface area:

### Option: Scoped Threads + chunks_mut

```rust
// Current (DisjointMut):
let dst = DisjointMut::new(vec![0u8; total_size]);
for tile in tiles {
    let region = dst.index_mut(tile.range);
    thread::spawn(move || process(region, tile));
}

// Alternative (safe):
crossbeam::scope(|s| {
    for (region, tile) in dst.chunks_mut(tile_size).zip(tiles) {
        s.spawn(move |_| process(region, tile));
    }
});
```

**Feasibility:** Requires refactoring tile assignment logic.

## Summary & Action Items

### Status Quo Assessment

DisjointMut is:
- ✅ **Conceptually sound** for its purpose
- ✅ **Well-implemented** with debug checking
- ⚠️ **Unsafe in release** if used incorrectly
- ⚠️ **Relies on manual reasoning** about disjointness

### Critical Action Items

**Priority 1: Audit**
- [ ] Review all tile boundary calculations
- [ ] Verify disjointness at each index_mut site
- [ ] Add disjointness proof comments
- [ ] Test with Miri/cargo-careful

**Priority 2: Test**
- [ ] Fuzz tile assignments in debug mode
- [ ] Property-based tests for tile math
- [ ] Stress test with varied frame sizes

**Priority 3: Document**
- [ ] Safety contract at module level
- [ ] Invariants at each usage site
- [ ] Known limitations and assumptions

**Priority 4: Consider (Optional)**
- [ ] Evaluate crossbeam-based alternative
- [ ] Add optional runtime checking feature
- [ ] Profile performance with RwLock baseline

## Relationship to Safety Levels

### Level 0 (forbid_unsafe) 
DisjointMut itself is unsafe (UnsafeCell interior), so:
- ❌ Cannot use DisjointMut in forbid_unsafe mode
- Must use alternative (single-threaded or crossbeam chunks_mut)

### Level 1 (quite-safe)
- ✅ DisjointMut allowed as "sound abstraction"
- ⚠️ Must document safety contract
- ⚠️ Must audit all usage sites

### Level 2+ (unchecked/c-ffi/asm)
- ✅ Fully available, same as quite-safe

**Implication:** DisjointMut is a quite-safe-level feature, not default-safe.

## Final Verdict

**DisjointMut is safe IF:**
1. All tile boundary math is correct (non-overlapping)
2. AsMutPtr::Target remains provenanceless (currently true)
3. Usage remains pub(crate) and audited
4. Debug mode tests catch any violations

**To increase confidence:**
1. Audit tile calculations (Priority 1)
2. Add disjointness assertions
3. Fuzz test in debug mode
4. Document safety proofs at call sites

The current implementation is sound, but relies on manual verification.
This is acceptable for a pub(crate) API with careful auditing.
