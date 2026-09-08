#!/usr/bin/env bash
# Build the arms for the `c256x2048` t=8 contention round.
#
# Two levers, one grid:
#   L1  let the derived rows rule go FINER than the block-count answer. The
#       shipped rule ends in `base.max(rows_shift.min(cap_shift))`, so it can
#       only coarsen; on a 256-wide plane the block-count answer is already
#       8 rows/block and the rows target of 4 is unreachable. `__probe_shiftpin`
#       pins a shift directly and is the ONLY instrument that can go finer, and
#       the only one that separates luma from chroma.
#   L2  the shard lock's WAITING policy, re-opened on the one cell where
#       `lock_slow` is 1.136 CPU ms/frame (AGENT_BRIEF §6's two nulls were both
#       taken at ~0.02% contention).
#
# One `bench_pin` build covers the whole L1 ladder because the pin is an env
# var, which also makes `RAV1D_PIN_SHIFT` at the cell's OWN shifts an identity
# control for the pin path itself.
#
# Copied to names cargo never writes, so a later build cannot swap the inode
# under a running measurement (AGENT_BRIEF §2). Builds are NICED; the
# measurement scripts never are.
set -euo pipefail
OUT=${1:-$HOME/tmp/c256/bin}
mkdir -p "$OUT"
cd "$(dirname "$0")/../.."

TIMED=(
  "plain=-"                       # HEAD/base: the shipped decoder
  "pin=__probe_shiftpin"            # the L1 ladder (env-var driven)
  "untracked=__probe_untracked"     # tracker-removed ceiling (bit-identical)
  "lockbackoff=__probe_lock_backoff" # L2: spin 64 -> yield
  "lockyield=__probe_lock_yield"    # L2: yield every iteration
  "lockpark=__probe_lock_park"      # L2: parking_lot::RawMutex, a real park
)
COUNT=(
  "pin__probewide=__probe_wide,__probe_shiftpin"
  "pin__probebounds=__probe_bounds,__probe_shiftpin"
  "plain__probewide=__probe_wide"
  "park__probewide=__probe_wide,__probe_lock_park"
  "backoff__probewide=__probe_wide,__probe_lock_backoff"
)

build_one() { # <example> <outname> <features>
  local ex=$1 name=$2 feat=$3
  echo "[$(date +%H:%M:%S)] building $name ($ex, features: $feat)" >&2
  if [ "$feat" = "-" ]; then
    nice -n 19 cargo build --release --example "$ex" >/dev/null
  else
    nice -n 19 cargo build --release --example "$ex" --features "$feat" >/dev/null
  fi
  cp "target/release/examples/$ex" "$OUT/$name"
}

for spec in "${TIMED[@]}"; do
  build_one bench_ab_decode "bench_${spec%%=*}" "${spec#*=}"
done
for spec in "${COUNT[@]}"; do
  build_one probe_tracker "pt_${spec%%=*}" "${spec#*=}"
done

echo "[$(date +%H:%M:%S)] wrote $OUT" >&2
ls -la "$OUT" >&2
