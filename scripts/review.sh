#!/usr/bin/env bash
# Reproducible local gates for the release/soundness review. Linux host runner.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export TMPDIR="${HOME}/tmp"
mkdir -p "$TMPDIR"
runner="${REVIEW_HEAVY_RUNNER:-$HOME/work/zen/scripts/run-heavy}"
logs="${REVIEW_LOG_DIR:-$HOME/tmp/rav1d-review-2026-09-05/logs}"
mkdir -p "$logs"
mode="${1:-help}"
if (( $# )); then shift; fi
features=aligned,pic-buf,zerocopy
case "$mode" in
  disjoint) cmd=(cargo test -p rav1d-disjoint-mut --features "$features" --no-fail-fast) ;;
  no-std) cmd=(cargo test -p rav1d-disjoint-mut --no-default-features --no-fail-fast) ;;
  loom)
    cmd=(cargo test -p rav1d-disjoint-mut --features __shards_4 --lib "${LOOM_TEST_FILTER:-loom_protocol}" -- --test-threads=1)
    ;;
  miri-stacked|miri-tree)
    export MIRIFLAGS=""
    [[ "$mode" != miri-tree ]] || export MIRIFLAGS=-Zmiri-tree-borrows
    cmd=(cargo +nightly miri test -p rav1d-disjoint-mut --features "$features" --no-fail-fast)
    ;;
  decoder-smoke|decoder-debug)
    cmd=(cargo nextest run -p rav1d-safe --test decode_md5_committed --test safe_simd_crashes --test fuzz_regression --test-threads 1)
    [[ "$mode" != decoder-smoke ]] || cmd+=(--release)
    ;;
  threading-protocol)
    cmd=(cargo test -p rav1d-safe --lib live_block_keeps_its_storage_when_another_decoder_changes_threading)
    ;;
  clippy) cmd=(cargo clippy -p rav1d-safe --lib -- -D warnings) ;;
  bench-list|bench-smoke)
    test -f test-vectors/dav1d-test-data/8-bit/data/meson.build || {
      echo 'Missing dav1d corpus; run scripts/download-test-vectors.sh first.' >&2; exit 1;
    }
    cmd=(cargo bench --bench decode --)
    if [[ "$mode" == bench-list ]]; then cmd+=(--list); else cmd+=(--test); fi
    ;;
  *)
    echo 'Usage: scripts/review.sh {disjoint|no-std|loom|miri-stacked|miri-tree|threading-protocol|decoder-smoke|decoder-debug|clippy|bench-list|bench-smoke} [additional arguments]'
    exit 0
    ;;
esac
# Never accidentally bake this host's ISA into a baseline. For layout experiments,
# use a separate documented invocation and apply identical flags to both arms.
unset RUSTFLAGS CARGO_ENCODED_RUSTFLAGS
if [[ "$mode" == loom ]]; then
  export RUSTFLAGS='--cfg disjoint_mut_loom'
  export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-target/review-loom}"
fi
export CARGO_BUILD_JOBS=8
log=$(mktemp "$logs/$mode-XXXXXXXX.log")
exec > >(tee "$log") 2>&1
echo "Log: $log"
date -u --iso-8601=seconds
git rev-parse HEAD
git status --short
rustc -Vv
[[ "$mode" != miri-* ]] || cargo +nightly miri --version
printf 'MIRIFLAGS=%s\n' "${MIRIFLAGS:-}"
printf 'RUSTFLAGS=%s\n' "${RUSTFLAGS:-}"
printf 'CARGO_TARGET_DIR=%s\n' "${CARGO_TARGET_DIR:-target}"
hostname
printf 'Command:'; printf ' %q' "${cmd[@]}" "$@"; printf '\n'
[[ ! -f Cargo.lock ]] || sha256sum Cargo.lock
# Cooperating review runs serialize; unrelated jobs still require an idle-host check.
exec 9>"$HOME/tmp/rav1d-review.lock"
flock 9
exec "$runner" --mem 16G --jobs 8 -- "${cmd[@]}" "$@"
