# rav1d-safe

[![License](https://img.shields.io/badge/license-BSD--2--Clause-blue.svg)](LICENSE)

**rav1d-safe** is a safe Rust AV1 decoder forked from [rav1d](https://github.com/memorysafety/rav1d).
It replaces 160k lines of hand-written x86/ARM assembly with safe Rust SIMD intrinsics
using [archmage](https://crates.io/crates/archmage) for zero-overhead dispatch and
[safe_unaligned_simd](https://crates.io/crates/safe_unaligned_simd) for safe memory access.

## Safety

**`deny(unsafe_code)` enforced across all 20 safe_simd modules when `asm` is disabled.**

When built without the `asm` feature, the SIMD path contains **zero `unsafe` blocks**.
All SIMD implementations use:
- `#[arcane]` / `#[rite]` for safe target-feature dispatch (via archmage tokens)
- `safe_unaligned_simd` for reference-based SIMD load/store (no raw pointers)
- Slice-based APIs throughout (no pointer arithmetic)
- `FlexSlice` zero-cost wrapper for hot-loop indexing with optional bounds elision

The `asm` feature gates only FFI wrappers (`pub unsafe extern "C" fn`) for function-pointer dispatch compatibility.

## Performance

Full-stack benchmark (20 decodes of test.avif via zenavif):
- **ASM (hand-written assembly): ~1.17s**
- **Safe-SIMD (safe Rust intrinsics): ~1.11s**
- Safe-SIMD **matches or beats** ASM performance

## Safe Rust API

A **100% safe Rust API** for decoding AV1 video (`src/managed.rs`).
No `unsafe` code required:

```rust
use rav1d_safe::src::managed::{Decoder, Planes};

let mut decoder = Decoder::new()?;
if let Some(frame) = decoder.decode(obu_data)? {
    match frame.planes() {
        Planes::Depth8(planes) => {
            for row in planes.y().rows() {
                // Process 8-bit luma row
            }
        }
        Planes::Depth16(planes) => {
            let pixel = planes.y().pixel(0, 0); // Zero-copy 16-bit access
        }
    }
}
```

Features: zero-copy pixel access, HDR metadata, type-safe color spaces,
configurable threading, 8-bit and 10/12-bit support.

See `examples/managed_decode.rs` for a complete example.

# Building

rav1d is written in Rust and uses the standard Rust toolchain to build. The Rust
toolchain can be installed by going to https://rustup.rs. The rav1d library
builds on stable Rust for `x86`, `x86_64`, and `aarch64`, but currently
requires a nightly compiler for `arm` and `riscv64`. The project is configured
to use a nightly compiler by default via `rust-toolchain.toml`, but a stable
library build can be made with the `+stable` cargo flag.

For x86 targets, you'll also need to install [`nasm`](https://nasm.us/) in order
to build with assembly support.

A release build can then be made using cargo:

```sh
cargo build --release
```

For development purposes you may also want to use the `opt-dev` profile, which
runs faster than a regular debug build but has all debug checks still enabled:

```sh
cargo build --profile opt-dev
```

To build just `librav1d` using a stable compiler:

```sh
cargo +stable build --lib --release
```

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `bitdepth_8` | ✅ | 8-bit pixel support |
| `bitdepth_16` | ✅ | 10/12-bit pixel support |
| `asm` | ❌ | Hand-written assembly + FFI wrappers (implies `c-ffi`) |
| `c-ffi` | ❌ | C API (`dav1d_*` entry points, implies `unchecked`) |
| `unchecked` | ❌ | Skip bounds checks in SIMD hot paths (`debug_assert!` only) |

**Safe-SIMD build** (default, recommended):
```sh
cargo build --release
```

**With hand-written assembly** (for comparison/benchmarking):
```sh
cargo build --features asm --release
```

## Cross-Compiling

rav1d can be cross-compiled for a target other than the host platform using the
`cargo` `--target` flag. This will require passing additional arguments to
`rustc` to tell it what linker to use. This can be done by setting the
`RUSTFLAGS` enviroment variable and specifying the `linker` compiler flag. For
example, compiling for `aarch64-unknown-linux-gnu` from an Ubuntu Linux machine
would be done as follows:

```sh
RUSTFLAGS="-C linker=aarch64-linux-gnu-gcc" cargo build --target aarch64-unknown-linux-gnu
```

If you're cross-compiling in order to run tests under QEMU (`qemu-*-static`)
you'll also need to specify the `+crt-static` target feature.

```sh
RUSTFLAGS="-C target-feature=+crt-static -C linker=aarch64-linux-gnu-gcc" cargo build --target aarch64-unknown-linux-gnu
```

This will require installing the `rustup` component for the target platform and
the appropriate cross-platform compiler/linker toolchain for your target
platform. Examples of how we cross-compile rav1d in CI can be found in
[`.github/workflows/build-and-test-qemu.yml`](.github/workflows/build-and-test-qemu.yml).

The following targets are currently supported:

* `x86_64-unknown-linux-gnu`
* `i686-unknown-linux-gnu`
* `armv7-unknown-linux-gnueabihf`
* `aarch64-unknown-linux-gnu`
* `riscv64gc-unknown-linux-gnu`

## Running Tests

Currently we use the original [Meson](https://mesonbuild.com/) test suite for
testing the Rust port. This means you'll need to [have Meson
installed](https://mesonbuild.com/Getting-meson.html) to run tests.

To setup and run the tests, do the following:

First, build `rav1d` using `cargo`. You'll need to do this step manually before
running any tests because it is not built automatically when tests are run. It's
recommended to run tests with either the `release` or `opt-dev` profile as the
debug build runs slowly and often causes tests to timeout. The `opt-dev` profile
is generally ideal for development purposes as it enables some optimizations
while leaving debug checks enabled.

```sh
cargo build --release
```

Or:

```sh
cargo build --profile opt-dev
```

Then you can run the tests with the [`test.sh`](.github/workflows/test.sh)
helper script:

```sh
.github/workflows/test.sh -r target/release/dav1d
```

Or:

```sh
.github/workflows/test.sh -r target/opt-dev/dav1d
```

The test script accepts additional arguments to configure how tests are run:

* `-s PATH` - Specify a path to the `seek_stress` binary in order to run the
  `seek_stress` tests. This is generally in the same output directory as the
  main `dav1d` binary, e.g. `target/release/seek_stress`.
* `-t MULTIPLIER` - Specify a multiplier for the test timeout. Allows for tests
  to take longer to run, e.g. if running tests with a debug build.
* `-f DELAY` - Specify a frame delay for the tests. If specified the tests will
  also be run with multiple threads.
* `-n` - Test with negative strides.
* `-w WRAPPER` - Specify a wrapper binary to use to run the tests. This is
  necessary for testing under QEMU for platforms other than the host platform.

You can learn more about how to build and test by referencing the CI scripts in
the [`.github/workflows`](.github/workflows) folder.

# Using rav1d

`librav1d` is designed to be a drop-in replacement for `libdav1d`, so it
primarily exposes a C API with the same usage as `libdav1d`'s. This is found in
the `librav1d.a` library generated by `cargo build`. [`libdav1d`'s primary API
documentation can be found
here](https://videolan.videolan.me/dav1d/dav1d_8h.html) for reference, and the
equivalent Rust functions can be found in [`src/lib.rs`](src/lib.rs). You can
also reference the `dav1d` binary's code to see how it uses the API, which can
be found at [`tools/dav1d.rs`](tools/dav1d.rs).

A [Rust API is planned](https://github.com/memorysafety/rav1d/issues/1252) for
addition in the future.
