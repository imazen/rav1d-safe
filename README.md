# rav1d-safe

A safe Rust AV1 decoder. Forked from [rav1d](https://github.com/memorysafety/rav1d), with 160k lines of hand-written x86/ARM assembly replaced by safe Rust SIMD intrinsics.

## Quick Start

Add to your `Cargo.toml`:
```toml
[dependencies]
rav1d-safe = { git = "https://github.com/imazen/rav1d-safe" }
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
| `Settings` | Thread count, film grain, filters, frame type selection |

**Metadata types:** `ColorInfo`, `ColorPrimaries`, `TransferCharacteristics`, `MatrixCoefficients`, `ColorRange`, `ContentLightLevel`, `MasteringDisplay`, `PixelLayout`

### Input Format

The decoder expects raw AV1 Open Bitstream Unit (OBU) data. If you have IVF or WebM containers, strip the container framing first and pass the OBU payload. See `tests/ivf_parser.rs` for an IVF parser example. For AVIF images, use [zenavif-parse](https://crates.io/crates/zenavif-parse) to extract the OBU data from the ISOBMFF container.

### Threading

```rust
use rav1d_safe::{Decoder, Settings};

// Single-threaded (default) — synchronous, deterministic
let decoder = Decoder::new()?;

// Multi-threaded — frame threading, better throughput
let decoder = Decoder::with_settings(Settings {
    threads: 0, // auto-detect core count
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

`deny(unsafe_code)` is enforced crate-wide when built without `asm` or `c-ffi` features. The managed API module uses `forbid(unsafe_code)` unconditionally.

The SIMD path uses:
- [archmage](https://crates.io/crates/archmage) for token-based target-feature dispatch (no manual `#[target_feature]`)
- [safe_unaligned_simd](https://crates.io/crates/safe_unaligned_simd) for reference-based SIMD load/store (no raw pointers)
- Value-type SIMD intrinsics, which are safe functions since Rust 1.93
- Slice-based APIs throughout — no pointer arithmetic in SIMD code

16 sound abstractions (`Send`/`Sync` impls, `AlignedVec`, `Pin`) require item-level `#[allow(unsafe_code)]` and cannot be eliminated without sub-crate extraction.

## Performance

The safe SIMD path matches hand-written assembly performance. Tested on x86_64 with a 768x512 YUV420 8-bit AVIF (20 iterations via zenavif):

| Build | Time |
|-------|------|
| ASM (hand-written assembly) | ~1.17s |
| Safe-SIMD (this crate, default) | ~1.11s |

The safe path is competitive because Rust's SIMD intrinsics compile to the same instructions as hand-written assembly, and the compiler can inline across function boundaries in ways that assembly can't.

## Building

Requires Rust 1.89+ (stable). Install via [rustup.rs](https://rustup.rs).

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
| `asm` | off | Hand-written assembly (implies `c-ffi`) |
| `c-ffi` | off | C API entry points (implies `unchecked`) |
| `unchecked` | off | Skip bounds checks in SIMD hot paths |

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

Built on the work of the [dav1d](https://code.videolan.org/videolan/dav1d) team (VideoLAN) and the [rav1d](https://github.com/memorysafety/rav1d) team (ISRG/Prossimo). The original C and assembly implementations are exceptional — this fork just proves you can match that performance in safe Rust.
