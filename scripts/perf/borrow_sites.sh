#!/usr/bin/env bash
# Per-call-site borrow-registration counts for one vector/thread cell.
#
# Builds `examples/probe_tracker` with `--features probe-sites` and dumps the
# `SITE` rows. That feature also widens the `#[cfg_attr(debug_assertions,
# track_caller)]` wrappers in picture.rs / with_offset.rs / pixels.rs / recon.rs
# so `Location::caller()` inside the tracker's `add` resolves to the real borrow
# site instead of collapsing onto three lines in `picture.rs`.
#
# NOT a timing harness — the probe's per-registration atomics perturb wall clock
# by several percent. Counts only.
#
# Usage: borrow_sites.sh <out.tsv> [vec] [threads] [iters]
set -eu
OUT=${1:?out.tsv}
VEC=${2:-v4k_8tile}
THREADS=${3:-1}
ITERS=${4:-3}
AVIF=${AVIF:-$HOME/tmp/rav1d-perf/vec}
cargo build --release --features probe-sites --example probe_tracker >&2
./target/release/examples/probe_tracker "$AVIF/$VEC.avif" "$THREADS" "$ITERS" \
  | grep -E '^(RUN|SITES|SITE)' > "$OUT"
echo "wrote $OUT" >&2
