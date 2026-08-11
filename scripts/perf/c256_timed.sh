#!/usr/bin/env bash
# The TIMED half of the `c256x2048` t=8 contention round.
#
# Phase L1 — the shift ladder, INCLUDING shifts FINER than the block-count rule
#   can reach. The shipped rows rule ends in `base.max(rows_shift.min(cap))`, so
#   on a 256-wide plane (8 rows/block already) it cannot refine; `probe-shiftpin`
#   is the only instrument that can, and the only one that separates the planes.
#   `pinL11C9` pins the shifts the rule ALREADY computes, so it is an identity
#   control: its spread against `plain` is this grid's noise floor.
#
# Phase L2 — the shard lock's waiting policy, re-opened on this cell. Both of
#   AGENT_BRIEF §6's nulls were taken where contention is ~0.02% of
#   registrations; here `lock_slow` is 1.136 CPU ms/frame. The t=1 row of the
#   same cell is the control: with one thread there is no contention at all, so
#   every lock arm must read 1.000 there or the arm is measuring something else.
#
# NO `nice` on a timed run (Darwin maps positive nice to background QoS and
# lands the process on E-cores, ~40x). Everything here runs under `measlock`.
#
# ROUNDS default to 8 and the report DISCARDS ROUND 0 — the first touch of each
# (arm, cell) pair is cold.
#
# Usage: c256_timed.sh <outdir> [rounds]
set -u
OUT=${1:?outdir}; R=${2:-8}
BIN=${BIN:-$HOME/tmp/c256/bin}
VEC=${VEC:-$HOME/tmp/bpsrows/vec}
IVF=${IVF:-$HOME/tmp/bpsrows/ivf}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
export PATH="$HOME/bin:$PATH"

# Pinned-shift wrappers. c256x2048 8-bit 4:2:0: luma stride 256, chroma 128.
# (11, 9) is what the shipped rule computes — the identity control.
mk_pin() { # <name> <L> <C>
  printf '#!/bin/sh\nRAV1D_PIN_SHIFT="256:%s,128:%s" exec "%s/bench_pin" "$@"\n' \
    "$2" "$3" "$BIN" > "$BIN/bench_$1"
  chmod +x "$BIN/bench_$1"
}
mk_pin pinL11C9 11 9     # identity control
mk_pin pinL10C9 10 9
mk_pin pinL10C8 10 8
mk_pin pinL9C9   9 9
mk_pin pinL8C9   8 9
mk_pin pinL7C9   7 9
mk_pin pinL6C6   6 6
mk_pin pinL12C10 12 10   # the coarse direction, for the other side of the fit

L1_ARMS=${L1_ARMS:-"plain pinL11C9 pinL10C9 pinL10C8 pinL9C9 pinL8C9 pinL7C9 pinL6C6 pinL12C10 untracked dav1d_fd1"}
L2_ARMS=${L2_ARMS:-"plain lockbackoff lockyield lockpark untracked dav1d_fd1"}

echo "[$(date +%H:%M:%S)] phase L1 — shift ladder, c256x2048 t=8, $R rounds" >&2
BIN="$BIN" AVIF="$VEC" IVF="$IVF" ARMS="$L1_ARMS" \
  CELLS="C256x2048_420_8b__t8:8:22:225" \
  measlock c256-l1 -- "$here/tiled_wallcpu.sh" "$OUT/l1_t8.tsv" "$R" \
  2> >(tee "$OUT/l1.log" >&2)

echo "[$(date +%H:%M:%S)] phase L2 — lock waiting policy, $R rounds" >&2
# Three cells on purpose: the target, its own contention-free t=1 control, and
# the cell where granularity ALREADY won (low contention at t=8) so the answer
# can be reported as cell-specific or not.
BIN="$BIN" AVIF="$VEC" IVF="$IVF" ARMS="$L2_ARMS" \
  CELLS="C256x2048_420_8b__t8:8:22:225 C256x2048_420_8b__t8:1:22:225 C1024x576_420_8b__t8:8:20:200" \
  measlock c256-l2 -- "$here/tiled_wallcpu.sh" "$OUT/l2.tsv" "$R" \
  2> >(tee "$OUT/l2.log" >&2)

echo "[$(date +%H:%M:%S)] done -> $OUT" >&2
