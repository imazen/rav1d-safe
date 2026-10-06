#!/usr/bin/env bash
# quick_gate.sh — fast correctness + perf smoke for ownership/tracker changes.
# ~30s: release-thin build, 1-frame tile-MT decodes vs sidecar MD5s, frame-MT
# delay self-consistency, row-guard unit tests, and a short timing probe.
# For the full gate (nextest suite, permutations, gen_cover, cross-arch) use
# `just test` / the ledger checklist — this is the iteration loop, not the bar.
#
# Env overrides:
#   STILLS=<dir>   stills corpus (default /home/lilith/tmp/rav1d-stills-2026-09-07)
#   BENCH=0        skip the timing probe
#   TESTS=0        skip the lib unit-test subset
set -euo pipefail
cd "$(dirname "$0")/.."

P=release-thin
BIN=./target/$P/examples
S=${STILLS:-/home/lilith/tmp/rav1d-stills-2026-09-07}
CLIP=test-vectors/dav1d-test-data/8-bit/data/00000003.ivf

step() { printf '\n=== %s ===\n' "$*"; }

step "build ($P)"
cargo build --profile $P --example decode_md5 --example profile_ivf

if [ -d "$S" ]; then
  step "tile-MT bit-exactness (t1 + t8 vs dav1d sidecar, 1 frame)"
  for s in photo-2k-t8 photo-4k-t8 map-4k-t8; do
    want=$(cut -d' ' -f1 < "$S/$s.dav1d.md5")
    $BIN/decode_md5 --threads 1 --limit 1 -q "$S/$s.ivf" "$want" >/dev/null
    $BIN/decode_md5 --threads 8 --limit 1 -q "$S/$s.ivf" "$want" >/dev/null
    echo "  $s t1+t8 OK"
  done

  step "CPU-tier identity (scalar/v3/native, 1 frame)"
  want=$(cut -d' ' -f1 < "$S/photo-2k-t8.dav1d.md5")
  for l in scalar v3 native; do
    $BIN/decode_md5 --level $l --threads 8 --limit 1 -q "$S/photo-2k-t8.ivf" "$want" >/dev/null
    echo "  $l OK"
  done
else
  echo "STILLS=$S missing — skipping stills legs"
fi

if [ -f "$CLIP" ]; then
  step "frame-MT self-consistency (t8 delay1 vs delay8, 16 frames)"
  r1=$($BIN/decode_md5 --threads 8 --delay 1 --limit 16 -q "$CLIP")
  r8=$($BIN/decode_md5 --threads 8 --delay 8 --limit 16 -q "$CLIP")
  if [ "$r1" = "$r8" ]; then echo "  $r8"; else
    echo "  MISMATCH: delay1=$r1 delay8=$r8"; exit 1; fi
fi

if [ "${TESTS:-1}" = 1 ]; then
  step "row-guard + ctx lib tests"
  cargo test --profile $P --lib picture:: --quiet 2>&1 | tail -3
fi

if [ "${BENCH:-1}" = 1 ] && [ -d "$S" ]; then
  step "timing probe (5 iters, t8/t4/t1)"
  for s in photo-2k-t8 photo-4k-t8; do
    for t in 8 4 1; do
      out=$(RAV1D_THREADS=$t $BIN/profile_ivf "$S/$s.ivf" 5 | tail -1)
      echo "  $s t$t: $out"
    done
  done
fi

step "quick gate passed"
