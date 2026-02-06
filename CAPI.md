# C API Backward Compatibility Plan

## Overview

rav1d-safe provides a C-compatible API (`dav1d_*` functions) behind the `c-ffi` feature flag. This API mirrors the upstream [dav1d C API](https://videolan.videolan.me/dav1d/dav1d_8h.html) (v7.0.0) and is the primary integration point for consumers like zenavif.

The goal: keep the C API working but make the entire codebase underneath safe Rust, with `unsafe` confined to a thin FFI shim layer.

## Current Architecture

Two parallel type systems exist for every API type:

| C API (FFI) | Internal (Rust) |
|---|---|
| `Dav1dContext` = `RawArc<Rav1dContext>` | `Arc<Rav1dContext>` |
| `Dav1dSettings` (`#[repr(C)]`) | `Rav1dSettings` (native Rust) |
| `Dav1dPicture` (`#[repr(C)]`) | `Rav1dPicture` (native Rust) |
| `Dav1dData` (`#[repr(C)]`) | `Rav1dData` (native Rust) |
| `Dav1dDataProps` (`#[repr(C)]`) | `Rav1dDataProps` (native Rust) |
| `Dav1dSequenceHeader` (`#[repr(C)]`) | `Rav1dSequenceHeader` (native Rust) |
| `Dav1dFrameHeader` (`#[repr(C)]`) | `Rav1dFrameHeader` (native Rust) |
| `Dav1dPicAllocator` (fn ptrs) | `Rav1dPicAllocator` (fn ptrs) |

Each `Dav1d*` type has `From`/`TryFrom` conversions to/from its `Rav1d*` counterpart.

## C API Functions (all in `src/lib.rs`, gated on `feature = "c-ffi"`)

### Must Remain Backward-Compatible

These are the public `dav1d_*` functions that external C/Rust consumers call:

| Function | Signature | Safety |
|---|---|---|
| `dav1d_version` | `() -> *const c_char` | safe extern "C" |
| `dav1d_version_api` | `() -> c_uint` | safe extern "C" |
| `dav1d_default_settings` | `(NonNull<Dav1dSettings>)` | unsafe (ptr write) |
| `dav1d_get_frame_delay` | `(Option<NonNull<Dav1dSettings>>) -> Dav1dResult` | unsafe (ptr read) |
| `dav1d_open` | `(Option<NonNull<Option<Dav1dContext>>>, Option<NonNull<Dav1dSettings>>) -> Dav1dResult` | unsafe (ptr read/write) |
| `dav1d_parse_sequence_header` | `(Option<NonNull<Dav1dSequenceHeader>>, Option<NonNull<u8>>, usize) -> Dav1dResult` | unsafe (ptr read/write) |
| `dav1d_send_data` | `(Option<Dav1dContext>, Option<NonNull<Dav1dData>>) -> Dav1dResult` | unsafe (RawArc deref, ptr read/write) |
| `dav1d_get_picture` | `(Option<Dav1dContext>, Option<NonNull<Dav1dPicture>>) -> Dav1dResult` | unsafe (RawArc deref, ptr write) |
| `dav1d_apply_grain` | `(Option<Dav1dContext>, Option<NonNull<Dav1dPicture>>, Option<NonNull<Dav1dPicture>>) -> Dav1dResult` | unsafe (RawArc deref, ptr read/write) |
| `dav1d_flush` | `(Dav1dContext)` | unsafe (RawArc deref) |
| `dav1d_close` | `(Option<NonNull<Option<Dav1dContext>>>)` | unsafe (RawArc into_arc, ptr read/write) |
| `dav1d_get_event_flags` | `(Option<Dav1dContext>, Option<NonNull<Dav1dEventFlags>>) -> Dav1dResult` | unsafe (RawArc deref, ptr write) |
| `dav1d_get_decode_error_data_props` | `(Option<Dav1dContext>, Option<NonNull<Dav1dDataProps>>) -> Dav1dResult` | unsafe (RawArc deref, ptr write) |
| `dav1d_picture_unref` | `(Option<NonNull<Dav1dPicture>>)` | unsafe (ptr read/write, drop) |
| `dav1d_data_create` | `(Option<NonNull<Dav1dData>>, usize) -> *mut u8` | unsafe (ptr write) |
| `dav1d_data_wrap` | `(Option<NonNull<Dav1dData>>, Option<NonNull<u8>>, usize, Option<FnFree>, ...) -> Dav1dResult` | unsafe (ptr write, fn ptr) |
| `dav1d_data_wrap_user_data` | `(Option<NonNull<Dav1dData>>, Option<NonNull<u8>>, Option<FnFree>, ...) -> Dav1dResult` | unsafe (ptr write, fn ptr) |
| `dav1d_data_unref` | `(Option<NonNull<Dav1dData>>)` | unsafe (ptr read/write, drop) |
| `dav1d_data_props_unref` | `(Option<NonNull<Dav1dDataProps>>)` | unsafe (ptr read/write, drop) |

**Total: 19 `extern "C"` functions**

### Constants (also backward-compatible)

- `DAV1D_API_VERSION_MAJOR/MINOR/PATCH`
- `DAV1D_PICTURE_ALIGNMENT`
- `DAV1D_INLOOPFILTER_*` constants
- `DAV1D_DECODEFRAMETYPE_*` constants
- `DAV1D_EVENT_FLAG_*` constants

### Types That Must Keep `#[repr(C)]` Layout

These are the types that cross the FFI boundary and must maintain their exact C layout:

- `Dav1dSettings` - decoder configuration
- `Dav1dPicture` - decoded picture output
- `Dav1dPictureParameters` - picture metadata
- `Dav1dData` - input bitstream data
- `Dav1dDataProps` - data metadata/timestamps
- `Dav1dUserData` - user-attached data
- `Dav1dSequenceHeader` - AV1 sequence header (large, ~500 bytes)
- `Dav1dFrameHeader` - AV1 frame header (large)
- `Dav1dPicAllocator` - picture allocation callbacks (fn ptrs + cookie)
- `Dav1dLogger` - logging callback
- All sub-structs of the above (Dav1dFilmGrainData, Dav1dSegmentationData, etc.)
- Various `type Dav1d* = c_uint` enum aliases

## Sources of `unsafe` in the Codebase

### Category 1: FFI Shim (`src/lib.rs`) — MUST be unsafe, isolatable

Every `extern "C"` function is inherently unsafe because:
- Raw pointer parameters (`*const`, `*mut`, `NonNull`)
- `RawArc` lifecycle management (manual reference counting)
- Type erasure through `c_void`
- Callback function pointers (`FnFree`, allocator callbacks)

**Strategy: Keep this. It's a ~960-line file. Confine ALL C API unsafe here.**

### Category 2: RawArc / CArc / CBox (`src/c_arc.rs`, `src/c_box.rs`) — Needed for C API only

`RawArc<T>` wraps `Arc<T>` as a raw pointer for C consumption. `Dav1dContext = RawArc<Rav1dContext>`. This is fundamentally unsafe — it's manual `Arc::into_raw`/`Arc::from_raw`.

`CArc<T>` wraps `Arc<Pin<CBox<T>>>` for C-allocated data that rav1d takes ownership of (e.g., `Dav1dData` wrapping a user's buffer with a free callback).

**Strategy: Keep these behind `c-ffi`. Internal code uses `Arc` directly.**

### Category 3: FFISafe (`src/ffi_safe.rs`) — DSP dispatch fn ptrs

`FFISafe<'a, T>` is a type-erased wrapper used in DSP function pointer dispatch. The DSP dispatch tables store `fn(*const FFISafe<Dst>, ...)` pointers that are called with type-erased arguments.

This exists because:
1. The original dav1d has function pointer tables for runtime dispatch (8bpc vs 16bpc, AVX2 vs SSE4, etc.)
2. When `asm` is enabled, these point to assembly functions with C calling convention
3. When safe-simd is used, these point to Rust functions wrapped in `unsafe extern "C"`

**Strategy: This is the hardest part. The function pointer dispatch pattern requires `unsafe` for type erasure. Options discussed below.**

### Category 4: DisjointMut (`src/disjoint_mut.rs`) — Core primitive

`DisjointMut<T>` provides interior mutability for pixel buffers where different threads write to non-overlapping regions. This is a fundamental safety primitive — it CHECKS disjointness at runtime in debug builds.

**Strategy: Keep as-is. This is a sound `unsafe` abstraction that makes the rest of the codebase safe.**

### Category 5: align, assume, send_sync_non_null — Utility unsafe

Small utility modules with inherent unsafe that provides safe APIs:
- `align`: Aligned allocation (`Align16`, etc.)
- `assume`: Compiler hint for optimizer
- `send_sync_non_null`: `NonNull` wrapper that is `Send + Sync`

**Strategy: Keep as-is. These are foundational primitives with small, auditable unsafe surfaces.**

### Category 6: DSP dispatch modules (itx, looprestoration, refmvs, lf_mask, msac)

These still have `#[allow(unsafe_code)]` because they contain FFI wrapper functions (`unsafe extern "C" fn`) that bridge between the function pointer dispatch tables and the safe implementations.

**Strategy: Move all FFI wrappers into `safe_simd/` modules (already done for filmgrain, ipred, mc). The dispatch modules themselves become fully safe.**

### Category 7: picture.rs — Picture allocation

Contains the default picture allocator and picture lifecycle management. Unsafe due to raw pointer manipulation for pixel buffer allocation.

**Strategy: Gate the C-compatible allocator behind `c-ffi`. For pure Rust use, provide a safe allocator using `Vec<u8>` or `DisjointMut`.**

## Plan: Layered Safety Architecture

### Layer 1: Pure Safe Rust Core (no `c-ffi`, no `asm`)

When built with just `--features "bitdepth_8,bitdepth_16"`:
- `#![deny(unsafe_code)]` applies crate-wide
- Only excepted modules: `align`, `assume`, `c_arc` (unused), `disjoint_mut`, `send_sync_non_null`, `safe_simd`
- These are sound abstractions with safe public APIs
- **Goal: 0 unsafe in application logic. All decode, transform, filter, prediction code is safe.**

Remaining work to reach this:
1. Move FFI wrappers out of `itx.rs`, `looprestoration.rs`, `refmvs.rs`, `lf_mask.rs` into `safe_simd/`
2. Make `msac.rs` safe (inline SIMD already uses `#[target_feature]`)
3. Make `picture.rs` safe (safe allocator for non-FFI builds)
4. Make `internal.rs` safe (may need refactoring of fn ptr tables)

### Layer 2: C FFI Shim (`feature = "c-ffi"`)

When `c-ffi` is enabled:
- `src/lib.rs` is allowed unsafe (the 19 `extern "C"` functions)
- `src/c_arc.rs`, `src/c_box.rs` are allowed unsafe (Arc↔raw ptr)
- `src/ffi_safe.rs` is allowed unsafe (type erasure for dispatch)
- `include/dav1d/*` types maintain `#[repr(C)]` layout
- All conversions between `Dav1d*` ↔ `Rav1d*` happen at the boundary

**This layer is ~1500 lines total. It's the entire unsafe surface for the C API.**

### Layer 3: ASM Backend (`feature = "asm"`)

When `asm` is enabled:
- Links to hand-written assembly via `extern "C"` FFI
- All DSP dispatch modules need unsafe for the FFI calls
- Not our focus — the safe-simd path replaces this entirely

## The Function Pointer Problem

The biggest challenge for full safety is the DSP dispatch tables. Currently:

```rust
// In wrap_fn_ptr! macro
pub struct FnPtr(Option<unsafe extern "C" fn(/* erased args */)>);
```

Every DSP function (MC, ITX, CDEF, loopfilter, etc.) is called through a function pointer table. This is necessary because:

1. **Bitdepth dispatch**: 8bpc and 16bpc have different pixel types but share the same code structure
2. **SIMD dispatch**: Different CPU feature levels get different implementations
3. **ASM compatibility**: ASM functions have C calling convention

### Options for the fn ptr dispatch

**Option A: Keep fn ptrs, confine unsafe to call sites (current approach)**
- Each call site uses `unsafe { (table.fn_ptr)(args) }`
- The dispatch modules create safe wrapper functions
- Works, but leaves `unsafe` scattered across decode.rs, recon.rs, etc.

**Option B: Trait-based dispatch (major refactor)**
- Replace fn ptr tables with trait objects or enums
- `trait McFns { fn put_8tap(&self, ...) }`
- Eliminates unsafe at call sites
- Problem: huge refactor, may hurt performance, doesn't work with ASM backend

**Option C: Safe fn ptr wrappers with runtime checks**
- Wrap each fn ptr in a type-safe struct that validates arguments
- Call sites use safe methods: `table.mc.put_8tap(dst, src, ...)`
- The wrapper does the unsafe call internally
- Moderate refactor, preserves performance

**Option D: Hybrid — safe wrappers for non-FFI, fn ptrs for FFI**
- When `c-ffi` is disabled: use direct function calls or trait dispatch
- When `c-ffi` is enabled: keep fn ptr tables for compatibility
- Best of both worlds but duplicates dispatch logic

### Recommended: Option C (safe fn ptr wrappers)

The `wrap_fn_ptr!` macro already exists. Extend it to provide safe call methods that encapsulate the `unsafe extern "C" fn` call. The dispatch table becomes a struct of safe-callable wrappers instead of raw fn ptrs.

## Rust-Only API (Alternative to C API)

For pure Rust consumers (like zenavif), we should provide a safe Rust API:

```rust
// Future safe Rust API (no unsafe needed by consumer)
pub struct Decoder { /* ... */ }

impl Decoder {
    pub fn new(settings: DecoderSettings) -> Result<Self, Error>;
    pub fn send_data(&mut self, data: &[u8]) -> Result<(), Error>;
    pub fn get_picture(&mut self) -> Result<Option<Picture>, Error>;
    pub fn flush(&mut self);
}

pub struct Picture { /* ... */ }
impl Picture {
    pub fn plane(&self, index: usize) -> &[u8];
    pub fn stride(&self, index: usize) -> isize;
    pub fn width(&self) -> u32;
    pub fn height(&self) -> u32;
    pub fn bit_depth(&self) -> u8;
    // ...
}
```

This would let zenavif drop all its `unsafe` blocks. The internal implementation calls the same `rav1d_open`, `rav1d_send_data`, etc. functions that already exist as safe Rust functions.

**The C API functions are thin unsafe wrappers around these safe internal functions.** The safe Rust API would be even thinner safe wrappers — or just re-exports.

## Priority Order

1. **Finish making DSP dispatch modules safe** (move remaining FFI wrappers to safe_simd/)
2. **Make msac.rs safe** (target_feature wrappers)
3. **Make picture.rs safe** (safe allocator for non-FFI builds)
4. **Make internal.rs safe** (safe fn ptr wrapper pattern)
5. **Add safe Rust API** (Decoder/Picture wrapper types)
6. **Migrate zenavif** to safe Rust API (drop all unsafe in consumer)

## What's NOT Changing

- The 19 `dav1d_*` extern "C" functions stay exactly as they are
- All `Dav1d*` struct layouts stay `#[repr(C)]`
- The `Dav1d* ↔ Rav1d*` conversion pattern stays
- ABI version stays at 7.0.0
- Feature flag behavior: `asm` for assembly, `c-ffi` for C API, neither for pure safe Rust
