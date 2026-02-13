# rav1d-safe

A safe Rust AV1 decoder. Forked from [rav1d](https://github.com/memorysafety/rav1d), with 160k lines of hand-written x86/ARM assembly replaced by safe Rust SIMD intrinsics.

## Quick Start

Add to your `Cargo.toml`:
```toml
[dependencies]
rav1d-safe = "0.2"
```

Decode an AV1 bitstream:
```rust
use rav1d_safe::{Decoder, Planes};

fn decode(obu_data: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    let mut decoder = Decoder::new()?;

    // Feed raw OBU data (not IVF/WebM containers)
    if let Some(frame) = decoder.decode(obu_data)? {
        println!("{}x{} @ {}bpc", frame.width(), frame.height(), frame.bit_depth());

        match frame.planes() {
            Planes::Depth8(planes) => {
                for row in planes.y().rows() {
                    // row is &[u8] — zero-copy, no allocation
                }
            }
            Planes::Depth16(planes) => {
                let px = planes.y().pixel(0, 0); // 10 or 12-bit value
            }
        }
    }

    // Drain any buffered frames
    for frame in decoder.flush()? {
        // ...
    }
    Ok(())
}
```

## API Overview

The public API lives in `src/managed.rs` and is re-exported at the crate root.

**Core types:**

| Type | Purpose |
|------|---------|
| `Decoder` | Decodes AV1 OBU data into frames |
| `Frame` | Decoded frame with metadata (cloneable, Arc-backed) |
| `Planes` | Enum dispatching to `Planes8` or `Planes16` by bit depth |
| `PlaneView8` / `PlaneView16` | Zero-copy 2D view with `row()`, `pixel()`, `rows()` |
| `Settings` | Thread count, film grain, filters, frame size limit, CPU level |
| `CpuLevel` | SIMD dispatch level (Scalar, X86V2, X86V3, X86V4, Neon, Native) |
| `Error` | Enum: `InvalidData`, `OutOfMemory`, `NeedMoreData`, etc. |

**Metadata types:** `ColorInfo`, `ColorPrimaries`, `TransferCharacteristics`, `MatrixCoefficients`, `ColorRange`, `ContentLightLevel`, `MasteringDisplay`, `PixelLayout`

### Input Format

The decoder expects raw AV1 Open Bitstream Unit (OBU) data. If you have IVF or WebM containers, strip the container framing first and pass the OBU payload. See `tests/ivf_parser.rs` for an IVF parser example. For AVIF images, use [zenavif-parse](https://crates.io/crates/zenavif-parse) to extract the OBU data from the ISOBMFF container.

### Threading

```rust
use rav1d_safe::{Decoder, Settings, CpuLevel};

// Single-threaded (default) — synchronous, deterministic
let decoder = Decoder::new()?;

// Multi-threaded — frame threading, better throughput
let decoder = Decoder::with_settings(Settings {
    threads: 0, // auto-detect core count
    ..Default::default()
})?;

// Constrained decoding — limit frame size and CPU features
let decoder = Decoder::with_settings(Settings {
    frame_size_limit: 3840 * 2160, // reject frames larger than 4K
    cpu_level: CpuLevel::Native,   // use best available SIMD
    ..Default::default()
})?;
```

With `threads >= 2` or `threads == 0`, the decoder uses frame threading. `decode()` may return `None` for complete frames because processing is asynchronous — call it repeatedly or use `flush()` to drain.

### HDR Metadata

```rust
if let Some(cll) = frame.content_light() {
    println!("MaxCLL: {} nits", cll.max_content_light_level);
}
if let Some(mdcv) = frame.mastering_display() {
    println!("Peak: {} nits", mdcv.max_luminance_nits());
}
let color = frame.color_info();
// color.primaries, color.transfer_characteristics, color.matrix_coefficients
```

### Error Handling

All fallible operations return `Result<T, rav1d_safe::Error>`. Error variants: `InvalidData`, `OutOfMemory`, `NeedMoreData`, `InitFailed`, `InvalidSettings`, `Other`.

## Safety Model

The default build (`forbid(unsafe_code)` crate-wide) contains zero `unsafe` in the main crate. The only unsafe code lives in the [rav1d-disjoint-mut](crates/rav1d-disjoint-mut/) workspace sub-crate, a provably sound `RefCell`-for-ranges abstraction with always-on bounds checking.

The SIMD path uses:
- [archmage](https://crates.io/crates/archmage) for token-based target-feature dispatch (no manual `#[target_feature]`)
- [safe_unaligned_simd](https://crates.io/crates/safe_unaligned_simd) for reference-based SIMD load/store (no raw pointers)
- Value-type SIMD intrinsics, which are safe functions since Rust 1.93
- Slice-based APIs throughout — no pointer arithmetic in SIMD code

Verify at runtime with `rav1d_safe::enabled_features()` — returns a comma-delimited list including the active safety level (e.g. `"bitdepth_8, bitdepth_16, safety:forbid-unsafe"`).

## What's Been Ported

The default build compiles under `forbid(unsafe_code)` in the main crate. All SIMD work lives in `src/safe_simd/` (59k lines of safe Rust replacing 233k lines of hand-written assembly across x86 and ARM).

### Ported: All DSP Kernels (AVX2 + NEON)

Every DSP kernel family has a safe Rust SIMD implementation that compiles under `forbid(unsafe_code)`:

| Module | x86 ASM replaced | ARM ASM replaced | Safe Rust |
|--------|-------------------|-------------------|-----------|
| mc (motion compensation) | 3 files (SSE/AVX2/AVX-512) x2 bitdepths | 4 files (32+64-bit) x2 bitdepths + SVE/dotprod | `mc.rs` + `mc_arm.rs` |
| itx (inverse transforms) | 3 files x2 bitdepths | 2 files x2 bitdepths | `itx.rs` + `itx_arm.rs` |
| ipred (intra prediction) | 3 files x2 bitdepths | 2 files x2 bitdepths | `ipred.rs` + `ipred_arm.rs` |
| cdef (directional enhancement) | 3 files x2 bitdepths | 2+tmpl files x2 bitdepths | `cdef.rs` + `cdef_arm.rs` |
| loopfilter | 3 files x2 bitdepths | 2 files x2 bitdepths | `loopfilter.rs` + `loopfilter_arm.rs` |
| looprestoration (Wiener + SGR) | 3 files x2 bitdepths | 2+common+tmpl files x2 bitdepths | `looprestoration.rs` + `looprestoration_arm.rs` |
| filmgrain | 3+common files x2 bitdepths | 2 files x2 bitdepths | `filmgrain.rs` + `filmgrain_arm.rs` |
| pal (palette) | 1 file | *(none — ARM uses scalar)* | `pal.rs` |
| refmvs (reference MVs) | 1 file | 2 files (32+64-bit) | `refmvs.rs` + `refmvs_arm.rs` |
| msac (entropy decoder) | 1 file (shared) | 1 file (shared) | inline in `msac.rs` |
| cpuid | 1 file (55 lines) | — | replaced by `std::arch` detection in `cpu.rs` |

msac uses branchless scalar for adapt4/adapt8 and a serial loop with early exit for adapt16. When the `unchecked` feature is enabled on x86_64, adapt4/adapt8/hi_tok switch to inlined SSE2 intrinsics via the `sse2!()` macro pattern (no function call overhead). The bool functions (bool_adapt, bool_equi) stay scalar — they have no data parallelism to exploit.

### Not Ported (With Rationale)

**Scaled MC** (put_8tap_scaled, prep_8tap_scaled, put_bilin_scaled, prep_bilin_scaled) — These functions use per-pixel variable step sizes with per-pixel filter selection, making them fundamentally different from fixed-block MC. The ASM versions are heavily register-scheduled for this pattern. Falls back to scalar Rust. ~2% of profile on inter-frame content.

**SSE-only paths** — 14 files, ~52k lines. The safe SIMD dispatch jumps straight to AVX2 when available. On pre-AVX2 hardware (pre-Haswell, 2013), the decoder falls back to scalar Rust rather than SSE intrinsics. SSE-only x86 hardware is rare enough that maintaining a second intrinsics tier isn't worth the code.

**ARM SVE2, dotprod, i8mm extensions** — `mc_dotprod.S` (1,880 lines) and `mc16_sve.S` (1,649 lines) are optional fast paths for newer ARM cores. The safe SIMD covers baseline NEON; these extension paths fall back to the NEON implementation.

**Remaining AVX-512 paths** — Some AVX-512 paths have been ported (itx, mc, ipred, looprestoration Wiener), but others remain unported. Falls back to AVX2 where not implemented.

**ASM infrastructure files** — `x86inc.asm` (1,983 lines), `asm.S`, `util.S`, `*_tmpl.S`, `*_common.S` are macro libraries and constants that only exist to support the raw assembly. No independent functionality to port.

## Performance

All benchmarks: x86_64 (Zen 4, AVX2), single-threaded, Rust 1.93, fat LTO. Run with `just profile` (500 iterations for IVF, 20 iterations for AVIF).

### Real photographs (AVIF decode, single image)

Single still images at web-typical quality (YUV420, q60). Source: Google-native 8K photo, downscaled with ImageMagick. These numbers reflect real-world AVIF decode performance where SIMD kernels dominate.

| Resolution | ASM | Partial ASM | Safe (checked) | Safe (unchecked) | Safe vs ASM |
|------------|-----|-------------|----------------|------------------|-------------|
| 4K (3840x2561) | 114 ms | 175 ms | 225 ms | 229 ms | 2.0x |
| 8K (8192x5464) | 512 ms | 725 ms | 999 ms | 976 ms | 1.9-2.0x |

### Small test vectors (IVF, multi-frame decode)

dav1d-test-data allintra 352x288 (39 frames). Entropy-heavy bitstream where the serial msac decoder dominates, which compresses the ratio compared to pixel-heavy workloads.

| Build | ms/iter | ms/frame | vs ASM |
|-------|---------|----------|--------|
| ASM | 92.6 | 2.37 | 1.0x |
| Partial ASM | 131.3 | 3.37 | 1.42x |
| Safe (checked) | 155.6 | 3.99 | 1.68x |
| Safe (unchecked) | 151.8 | 3.89 | 1.64x |

### Where the gap comes from

The safe build is ~2x slower on real images and ~1.7x on entropy-heavy vectors. The gap breaks down:

- **Entropy decoder (msac)**: ~35% of decode time. Serial dependency chain — the core symbol decode loop can't be parallelized. The `partial_asm` feature uses hand-tuned ASM for msac and loopfilter, bringing real photo decodes down to 1.4-1.5x vs full ASM.
- **Calling conventions**: The ASM kernels use custom register allocation across function boundaries. Rust's ABI reloads registers at each call site.
- **Scaled MC**: Falls back to scalar Rust (~2% of inter-frame content). The ASM version uses per-pixel variable-step register scheduling that doesn't map cleanly to safe intrinsics.
- **Bounds checking**: The `unchecked` feature removes runtime bounds checks and DisjointMut borrow tracking. This saves ~2-3% on real photos, confirming the tracking overhead is modest.

### Reproduce locally

```sh
just generate-bench-avif  # create 4K/8K AVIF test images (requires avifdec + avifenc)
just profile              # all four modes side-by-side (ASM, partial ASM, checked, unchecked)
just profile-quick        # same with fewer iterations
```

## Conformance

Tested against the [dav1d-test-data](https://code.videolan.org/videolan/dav1d-test-data) suite. MD5 hashes verified at all CPU dispatch levels (scalar, SSE4.2, AVX2, native).

**784 of 803 test vectors pass** across all levels.

19 vectors are not exercised by the test harness:

| Category | Count | Reason |
|----------|-------|--------|
| sframe | 1 | Requires S-frame support |
| svc (operating points) | 6 | Always decodes at default operating point |
| argon (vq_suite) | 12 | Various decode modes and operating point selection tests |

Run conformance tests with `cargo test --release --test decode_cpu_levels`.

## Building

Requires Rust 1.93+ (stable). Install via [rustup.rs](https://rustup.rs).

```sh
# Default safe-SIMD build (recommended)
cargo build --release

# With original hand-written assembly (for benchmarking)
cargo build --features asm --release

# Run tests
cargo test --release
```

### Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `bitdepth_8` | on | 8-bit pixel support |
| `bitdepth_16` | on | 10/12-bit pixel support |
| `unchecked` | off | Skip bounds checks in SIMD hot paths; enables SSE2 msac on x86_64 |
| `partial_asm` | off | ASM for entropy decoding (msac) and loopfilter only; safe SIMD everything else. Implies `unchecked` |
| `c-ffi` | off | C API entry points (`dav1d_*` symbols). Implies `unchecked` |
| `asm` | off | Full hand-written assembly. Implies `c-ffi` |

Safety chain: `default` (`forbid(unsafe_code)`) -> `unchecked` -> `c-ffi` -> `asm`. Each level relaxes the safety constraint.

### Cross-Compilation

```sh
# aarch64
RUSTFLAGS="-C linker=aarch64-linux-gnu-gcc" \
  cargo build --target aarch64-unknown-linux-gnu --release

# Verify aarch64 NEON compiles
cargo check --target aarch64-unknown-linux-gnu
```

Supported targets: `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`, `i686-unknown-linux-gnu`, `armv7-unknown-linux-gnueabihf`, `riscv64gc-unknown-linux-gnu`.

## License

New code in this fork (safe SIMD implementations, managed API, tooling) is dual-licensed:

- **[AGPL-3.0-or-later](LICENSE-AGPL)** for open-source use
- **Commercial license** available from [Imazen](https://imazen.io) for proprietary use

The upstream rav1d/dav1d code retains its original **[BSD-2-Clause](COPYING)** license.

### Upstream Contribution

This fork exists because maintaining a separate safe SIMD implementation is the fastest path to getting safe Rust AV1 decoding into production. If the rav1d maintainers are interested in upstreaming any of this work under the original BSD-2-Clause license, we'd be happy to contribute. Open an issue or reach out.

## Acknowledgments

Built on the work of the [dav1d](https://code.videolan.org/videolan/dav1d) team (VideoLAN) and the [rav1d](https://github.com/memorysafety/rav1d) team (ISRG/Prossimo). The original C and assembly implementations are exceptional — this fork demonstrates that safe Rust SIMD can get within 2x of hand-written assembly while eliminating entire classes of memory safety bugs.
