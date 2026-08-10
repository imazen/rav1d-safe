#!/usr/bin/env bash
# Build an x86_64 binary on an aarch64 macOS host and RUN it, for correctness work.
#
# Why this shape: Rosetta 2 is absent on this box (`arch -x86_64 /usr/bin/true` =>
# "Bad CPU type in executable"), so an `x86_64-apple-darwin` build cannot be
# executed. What CAN execute x86_64 here is the colima QEMU-TCG Linux VM
# (`colima start --profile x86 --arch x86_64`). Building INSIDE that VM means an
# emulated rustc (slow); building on the host and running the artifact in the VM
# takes ~34 s per binary instead.
#
# The link needs care twice over:
#   * `alloca` (a tango-bench dev-dep) needs a C compiler for the target, and
#     `zig cc` is the only one on this box. cc-rs detects it as clang and passes
#     `--target=x86_64-unknown-linux-musl`, which zig rejects (it wants
#     `x86_64-linux-musl`), so the wrapper strips that flag.
#   * Rust's musl target ships its own `rcrt1.o` and zig ships `crt1.o`; both
#     define `_start`. `-C link-self-contained=no` hands the CRT to zig.
#
# ONLY correctness numbers come out of this. TCG has no cache model, no store
# buffer and no PMU: never quote a wall-clock or ms/frame figure from it.
# See docs/X64_APPLICABILITY.md section F.
#
# Usage:
#   scripts/x64/build_and_run_x86.sh build --example md5_inventory
#   scripts/x64/build_and_run_x86.sh run  ./target/x86_64-unknown-linux-musl/release/examples/md5_inventory --threads 8
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="${TMPDIR:-$HOME/tmp}/x64-zig-wrappers"
TARGET=x86_64-unknown-linux-musl
IMAGE="${X64_IMAGE:-alpine:3.20}"

mkdir -p "$BIN"
cat > "$BIN/x86_64-linux-musl-gcc" <<'EOF'
#!/bin/sh
args=""
for a in "$@"; do
  case "$a" in --target=*) continue ;; esac
  args="$args '$(printf '%s' "$a" | sed "s/'/'\\\\''/g")'"
done
eval exec zig cc -target x86_64-linux-musl $args
EOF
cat > "$BIN/x86_64-linux-musl-ar" <<'EOF'
#!/bin/sh
exec zig ar "$@"
EOF
chmod +x "$BIN/x86_64-linux-musl-gcc" "$BIN/x86_64-linux-musl-ar"

cmd="${1:?build|run}"; shift

case "$cmd" in
  build)
    # `nice` is fine for a build (it keeps P-cores free for another agent's
    # measurements); it is NEVER fine for a timed run — and nothing timed comes
    # out of this script anyway.
    cd "$REPO"
    PATH="$BIN:$PATH" \
    CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER=x86_64-linux-musl-gcc \
    CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_RUSTFLAGS="-C link-self-contained=no" \
      nice -n 19 cargo build --release --target "$TARGET" "$@"
    ;;
  run)
    # The test-vector path is baked in at compile time (`env!("CARGO_MANIFEST_DIR")`),
    # so the repo MUST be mounted at its host path inside the container.
    docker run --rm --platform linux/amd64 -v "$REPO:$REPO" -w "$REPO" "$IMAGE" "$@"
    ;;
  *) echo "usage: $0 build|run ..." >&2; exit 2 ;;
esac
