# rav1d-safe [![CI](https://img.shields.io/github/actions/workflow/status/imazen/rav1d-safe/ci.yml?style=flat-square)](https://github.com/imazen/rav1d-safe/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/rav1d-safe?style=flat-square)](https://crates.io/crates/rav1d-safe) [![lib.rs](https://img.shields.io/badge/lib.rs-rav1d--safe-orange?style=flat-square)](https://lib.rs/crates/rav1d-safe) [![docs.rs](https://img.shields.io/docsrs/rav1d-safe?style=flat-square)](https://docs.rs/rav1d-safe) [![license](https://img.shields.io/crates/l/rav1d-safe?style=flat-square)](https://github.com/imazen/rav1d-safe#license)

A safe Rust AV1 decoder. Forked from [rav1d](https://github.com/memorysafety/rav1d), with 160k lines of hand-written x86/ARM assembly replaced by safe Rust SIMD intrinsics.

578 commits since the fork (+90,813 / -6,958 lines across 1,194 files). 68k of those net new lines are safe SIMD in `src/safe_simd/`.

## Quick Start

Add to your `Cargo.toml`:
```toml
[dependencies]
rav1d-safe = "0.5"
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

With `threads >= 2` or `threads == 0`, the decoder uses tile threading to parallelize decode within each frame. `decode()` may return `None` for complete frames because processing is asynchronous — call it repeatedly or use `flush()` to drain.

**Tile threading works under `forbid(unsafe_code)` without the `unchecked` feature.** No special feature flags needed:

```toml
rav1d-safe = { version = "0.5", features = ["bitdepth_8", "bitdepth_16"] }
```

With 2 threads, expect ~2x speedup on photo decode. Frame threading (`max_frame_delay > 1`) still requires the `unchecked` feature.

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

All benchmarks: x86_64 (Zen 4, AVX2), single-threaded, Rust 1.93+, fat LTO. Run with `just profile` (500 iterations for IVF, 20 iterations for AVIF).

### Real photographs (AVIF decode, single image)

Single still images at web-typical quality (YUV420, q60). Source: Google-native 8K photo, downscaled with ImageMagick. These numbers reflect real-world AVIF decode performance where SIMD kernels dominate.

| Resolution | ASM | Safe (checked) | Safe (unchecked) | Safe vs ASM |
|------------|-----|----------------|------------------|-------------|
| 4K (3840x2561) | 122 ms | 190 ms | 188 ms | 1.55x |
| 8K (8192x5464) | 724 ms | 1319 ms | 1289 ms | 1.8x |

(4K figures are post-0.5.6 SIMD work — i16-packed `pmaddwd` transforms + YMM-widened
loopfilter brought the checked ratio from 2.0× to ~1.55×. The 8K row predates 0.5.6 and
will improve similarly on re-measure.)

### Small test vectors (IVF, multi-frame decode)

dav1d-test-data allintra 352x288 (39 frames). Entropy-heavy bitstream where the serial msac decoder dominates, which compresses the ratio compared to pixel-heavy workloads.

| Build | ms/iter | ms/frame | vs ASM |
|-------|---------|----------|--------|
| ASM | 107.3 | 2.75 | 1.0x |
| Safe (checked) | 173.1 | 4.44 | 1.61x |
| Safe (unchecked) | 164.1 | 4.21 | 1.53x |

### Where the gap comes from

The safe build is ~1.55x slower on 4K real images and ~1.6x on entropy-heavy vectors. The gap breaks down:

- **Entropy decoder (msac)**: ~35% of decode time. Serial dependency chain — the core symbol decode loop can't be parallelized. The `partial_asm` feature uses hand-tuned ASM for msac and loopfilter, bringing real photo decodes down to 1.4-1.5x vs full ASM.
- **Calling conventions**: The ASM kernels use custom register allocation across function boundaries. Rust's ABI reloads registers at each call site.
- **Scaled MC**: Falls back to scalar Rust (~2% of inter-frame content). The ASM version uses per-pixel variable-step register scheduling that doesn't map cleanly to safe intrinsics.
- **Bounds checking**: The `unchecked` feature skips DisjointMut borrow tracking. This saves ~5% on photos, confirming the tracking overhead is modest.

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
| `unchecked` | off | Skip DisjointMut borrow tracking; enables frame threading and SSE2 msac on x86_64 |
| `partial_asm` | off | ASM for entropy decoding (msac) and loopfilter only; safe SIMD everything else. Implies `unchecked` |
| `c-ffi` | off | C API entry points (`dav1d_*` symbols). Implies `unchecked` |
| `asm` | off | Full hand-written assembly. Implies `c-ffi` |

Safety chain: `default` (`forbid(unsafe_code)`, tile threading) -> `unchecked` (frame threading) -> `c-ffi` -> `asm`. Each level relaxes the safety constraint.

### Cross-Compilation

```sh
# aarch64
RUSTFLAGS="-C linker=aarch64-linux-gnu-gcc" \
  cargo build --target aarch64-unknown-linux-gnu --release

# Verify aarch64 NEON compiles
cargo check --target aarch64-unknown-linux-gnu
```

Supported targets: `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`, `i686-unknown-linux-gnu`, `armv7-unknown-linux-gnueabihf`, `riscv64gc-unknown-linux-gnu`.

## Image tech I maintain

| | |
|:--|:--|
| State of the art codecs* | [zenjpeg] · [zenpng] · [zenwebp] · [zengif] · [zenavif] (**rav1d-safe** · [zenrav1e] · [zenavif-parse] · [zenavif-serialize]) · [zenjxl] ([jxl-encoder] · [zenjxl-decoder]) · [zentiff] · [zenbitmaps] · [heic] · [zenraw] · [zenpdf] · [ultrahdr] · [mozjpeg-rs] · [webpx] |
| Compression | [zenflate] · [zenzop] |
| Processing | [zenresize] · [zenfilters] · [zenquant] · [zenblend] |
| Metrics | [zensim] · [fast-ssim2] · [butteraugli] · [resamplescope-rs] · [codec-eval] · [codec-corpus] |
| Pixel types & color | [zenpixels] · [zenpixels-convert] · [linear-srgb] · [garb] |
| Pipeline | [zenpipe] · [zencodec] · [zencodecs] · [zenlayout] · [zennode] |
| ImageResizer | [ImageResizer] (C#) — 24M+ NuGet downloads across all packages |
| [Imageflow][] | Image optimization engine (Rust) — [.NET][imageflow-dotnet] · [node][imageflow-node] · [go][imageflow-go] — 9M+ NuGet downloads across all packages |
| [Imageflow Server][] | [The fast, safe image server](https://www.imazen.io/) (Rust+C#) — 552K+ NuGet downloads, deployed by Fortune 500s and major brands |

<sub>* as of 2026</sub>

### General Rust awesomeness

[archmage] · [magetypes] · [enough] · [whereat] · [zenbench] · [cargo-copter]

[And other projects](https://www.imazen.io/open-source) · [GitHub @imazen](https://github.com/imazen) · [GitHub @lilith](https://github.com/lilith) · [lib.rs/~lilith](https://lib.rs/~lilith) · [NuGet](https://www.nuget.org/profiles/imazen) (over 30 million downloads / 87 packages)

## License

Dual-licensed: [AGPL-3.0](LICENSE-AGPL3) or [commercial](LICENSE-COMMERCIAL).

I've maintained and developed open-source image server software — and the 40+
library ecosystem it depends on — full-time since 2011. Fifteen years of
continual maintenance, backwards compatibility, support, and the (very rare)
security patch. That kind of stability requires sustainable funding, and
dual-licensing is how we make it work without venture capital or rug-pulls.
Support sustainable and secure software; swap patch tuesday for patch leap-year.

[Our open-source products](https://www.imazen.io/open-source)

**Your options:**

- **Startup license** — $1 if your company has under $1M revenue and fewer
  than 5 employees. [Get a key →](https://www.imazen.io/pricing)
- **Commercial subscription** — Governed by the Imazen Site-wide Subscription
  License v1.1 or later. Apache 2.0-like terms, no source-sharing requirement.
  Sliding scale by company size.
  [Pricing & 60-day free trial →](https://www.imazen.io/pricing)
- **AGPL v3** — Free and open. Share your source if you distribute.

See [LICENSE-COMMERCIAL](LICENSE-COMMERCIAL) for details.

Upstream code from [memorysafety/rav1d](https://github.com/memorysafety/rav1d) is licensed under BSD-2-Clause.
Our additions and improvements are dual-licensed (AGPL-3.0 or commercial) as above.

### Upstream Contribution

We are willing to release our improvements under the original BSD-2-Clause
license if upstream takes over maintenance of those improvements. We'd rather
contribute back than maintain a parallel codebase. Open an issue or reach out.
## Acknowledgments

Built on the work of the [dav1d](https://code.videolan.org/videolan/dav1d) team (VideoLAN) and the [rav1d](https://github.com/memorysafety/rav1d) team (ISRG/Prossimo). The original C and assembly implementations are exceptional — this fork demonstrates that safe Rust SIMD can get within 2x of hand-written assembly while eliminating entire classes of memory safety bugs.

[zenjpeg]: https://github.com/imazen/zenjpeg
[zenpng]: https://github.com/imazen/zenpng
[zenwebp]: https://github.com/imazen/zenwebp
[zengif]: https://github.com/imazen/zengif
[zenavif]: https://github.com/imazen/zenavif
[zenjxl]: https://github.com/imazen/zenjxl
[zentiff]: https://github.com/imazen/zentiff
[zenbitmaps]: https://github.com/imazen/zenbitmaps
[heic]: https://github.com/imazen/heic-decoder-rs
[zenraw]: https://github.com/imazen/zenraw
[zenpdf]: https://github.com/imazen/zenpdf
[ultrahdr]: https://github.com/imazen/ultrahdr
[jxl-encoder]: https://github.com/imazen/jxl-encoder
[zenjxl-decoder]: https://github.com/imazen/zenjxl-decoder
[zenrav1e]: https://github.com/imazen/zenrav1e
[mozjpeg-rs]: https://github.com/imazen/mozjpeg-rs
[zenavif-parse]: https://github.com/imazen/zenavif-parse
[zenavif-serialize]: https://github.com/imazen/zenavif-serialize
[webpx]: https://github.com/imazen/webpx
[zenflate]: https://github.com/imazen/zenflate
[zenzop]: https://github.com/imazen/zenzop
[zenresize]: https://github.com/imazen/zenresize
[zenfilters]: https://github.com/imazen/zenfilters
[zenquant]: https://github.com/imazen/zenquant
[zenblend]: https://github.com/imazen/zenblend
[zensim]: https://github.com/imazen/zensim
[fast-ssim2]: https://github.com/imazen/fast-ssim2
[butteraugli]: https://github.com/imazen/butteraugli
[zenpixels]: https://github.com/imazen/zenpixels
[zenpixels-convert]: https://github.com/imazen/zenpixels
[linear-srgb]: https://github.com/imazen/linear-srgb
[garb]: https://github.com/imazen/garb
[zenpipe]: https://github.com/imazen/zenpipe
[zencodec]: https://github.com/imazen/zencodec
[zencodecs]: https://github.com/imazen/zencodecs
[zenlayout]: https://github.com/imazen/zenlayout
[zennode]: https://github.com/imazen/zennode
[Imageflow]: https://github.com/imazen/imageflow
[Imageflow Server]: https://github.com/imazen/imageflow-server
[imageflow-dotnet]: https://github.com/imazen/imageflow-dotnet
[imageflow-node]: https://github.com/imazen/imageflow-node
[imageflow-go]: https://github.com/imazen/imageflow-go
[ImageResizer]: https://github.com/imazen/resizer
[archmage]: https://github.com/imazen/archmage
[magetypes]: https://github.com/imazen/archmage
[enough]: https://github.com/imazen/enough
[whereat]: https://github.com/lilith/whereat
[zenbench]: https://github.com/imazen/zenbench
[cargo-copter]: https://github.com/imazen/cargo-copter
[resamplescope-rs]: https://github.com/imazen/resamplescope-rs
[codec-eval]: https://github.com/imazen/codec-eval
[codec-corpus]: https://github.com/imazen/codec-corpus
