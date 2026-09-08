# Rust AV1 and AVIF workflows

`rav1d-safe` provides a native Rust decoder API. Import `Decoder`, `Settings`,
`Frame`, and `Planes` directly from `rav1d_safe`; enabling `c-ffi` is unnecessary
for Rust applications. Its default checked build uses safe Rust SIMD and
runtime overlap checking.

| You have / need | Crate and interface |
|---|---|
| Planar YCbCr → raw AV1 | `zenrav1e::Context`, yielding packets of OBU data |
| Raw AV1 → planar YUV frames | `rav1d_safe::Decoder` |
| RGB pixels → complete AVIF file | `zenavif::encode`, with its `encode` feature |
| Complete AVIF file → pixel buffer | `zenavif::decode`, using rav1d-safe by default |

zenavif's encoding path uses zenravif and zenrav1e. Its decoding path handles
AVIF container items, alpha, and color conversion around rav1d-safe's AV1
Rust API. Raw zenrav1e packet bytes are not an AVIF file: renaming them `.avif`
does not add the required container. Conversely, pass complete AVIF bytes to
zenavif, not directly to `rav1d_safe::Decoder::decode`.

## Run the examples

The [standalone consumer](../examples/rust-codec-workflow/Cargo.toml) pins
published versions verified on 2026-09-08: rav1d-safe **0.5.7**, zenrav1e
**0.1.4**, and zenavif **0.1.6**. It disables default features explicitly and
selects both decoder bit depths plus zenavif encoding; no assembly, C FFI, or
unchecked decoder feature is enabled. This example package has its own
workspace, so the decoder's dev-dependency features do not leak into it.
The repository's staged 0.6.0 decoder is the subject of the separate benchmark.

```sh
cargo run --locked --release --manifest-path examples/rust-codec-workflow/Cargo.toml --bin raw_av1
cargo run --locked --release --manifest-path examples/rust-codec-workflow/Cargo.toml --bin avif
```

Both programs generate a small image in memory, encode it, decode it, and check
the output dimensions. The raw AV1 example also checks frame count, bit depth,
layout, and luma-row count. Neither example needs an input file, NASM, or a C
compiler. They use lossy encoding, so they do not assert input/output pixel
identity.

### zenrav1e → rav1d-safe

[Full runnable source](../examples/rust-codec-workflow/src/bin/raw_av1.rs).
Configure `EncoderConfig` for a still, allocate a frame, and fill its separate
Y, U, and V planes. With 8-bit 4:2:0, chroma is half width and half height.
`copy_from_raw_u8` takes the source stride in bytes and a sample width of one
byte; the example derives each plane width from its decimation.

After `send_frame`, flush the encoder and collect its packet data. Decode the
raw OBU bytes through the Rust API:

```rust
use rav1d_safe::{Decoder, Frame};

fn decode_still(obu: &[u8]) -> rav1d_safe::Result<Vec<Frame>> {
    let mut decoder = Decoder::new()?;
    let mut frames = Vec::new();
    if let Some(frame) = decoder.decode(obu)? {
        frames.push(frame);
    }
    frames.extend(decoder.flush()?);
    Ok(frames)
}
```

Keep the `Frame` alive while borrowing its plane views. `Planes::Depth8` exposes
`u8` samples; `Planes::Depth16` exposes the native 10/12-bit samples in `u16`.
Those planes are YUV, not interleaved RGB. For multiple input packets, drain
`get_frame()` between packets and flush after the final packet; `Ok(None)` is
not an end-of-stream marker.

### Full AVIF with zenavif

[Full runnable source](../examples/rust-codec-workflow/src/bin/avif.rs).
The example builds a typed RGB pixel buffer, then uses:

```rust
fn round_trip(input: &zenavif::PixelBuffer) -> Result<(), Box<dyn std::error::Error>> {
let encoded = zenavif::encode(input)?;
let decoded = zenavif::decode(&encoded.avif_file)?;
println!("{}x{}", decoded.width(), decoded.height());
Ok(())
}
```

`encoded.avif_file` contains the complete file bytes. `decoded` is a
`PixelBuffer`, carrying geometry, pixel representation, and color metadata;
it is not a flat RGBA byte vector. Consult the
[zenavif API](https://docs.rs/zenavif/0.1.6/zenavif/) for output conversion and
metadata handling. Avoid its `unsafe-asm` feature when selecting the default
rav1d-safe backend.

## Configure checked decoding

`Settings` is non-exhaustive. Start with defaults and assign fields:

```rust
use rav1d_safe::{Decoder, Settings};

fn configured() -> rav1d_safe::Result<Decoder> {
let mut settings = Settings::default();
settings.threads = 4;
settings.max_frame_delay = 1;
settings.frame_size_limit = 3840 * 2160; // total luma pixels
Decoder::with_settings(settings)
}
```

More workers can help when the encoded still has multiple tiles; they cannot
make a single tile independently decodable. Keep `unchecked`, `c-ffi`, and
`asm` disabled for rav1d-safe's default crate-wide `forbid(unsafe_code)` policy.
Cargo features are additive: another dependency enabling them on the same
package instance also changes your build. Inspect the resolved feature graph
with `cargo tree -e features`.

Upstream rav1d's assembly-disabled Rust implementation still contains unsafe
code. It has no equivalent `forbid(unsafe_code)` mode. The
[matched no-assembly benchmark](../benchmarks/noasm-2026-09-08/README.md)
compares that configuration against checked rav1d-safe without calling both
implementations memory-safe.
