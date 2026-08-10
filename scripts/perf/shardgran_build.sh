#!/usr/bin/env bash
# Build one `bench_ab_decode` per rung of the shard-GRANULARITY ladder, plus the
# probe-tasktime twin of each (that is where `TAIL_CONC` lives) and the
# tracker-removed ceiling.
#
# The ladder knob is `BPS` (blocks per shard the adaptive block shift aims for,
# `crates/rav1d-disjoint-mut/src/tracker_shard.rs`). A rung with a SMALLER ratio
# is one shift COARSER, i.e. more picture rows share a block:
#
#   bpsq   1/4  -> +2 shifts vs default    bps1  1/1 -> +1
#   bpshalf 1/2 -> +1 shift                bps4  4/1 -> -1
#   plain   2/1 -> the shipped default     bps8  8/1 -> -2
#
# Copied to a name cargo never writes, so a later build cannot swap the inode
# under a running measurement (AGENT_BRIEF §2).
#
# Builds are NICED (they must not steal P-cores from another agent's timed run);
# the measurement scripts never are.
set -euo pipefail
OUT=${1:-$HOME/tmp/shardgran/bin}
mkdir -p "$OUT"
cd "$(dirname "$0")/../.."

# arm=cargo-features ("-" = default features)
ARMS=(
  "plain=-"
  "bpsq=bps-quarter"
  "bpshalf=bps-half"
  "bps1=bps-1"
  "bps4=bps-4"
  "bps8=bps-8"
  "untracked=probe-untracked"
  "tt=probe-tasktime"
  "tt_bpsq=probe-tasktime,bps-quarter"
  "tt_bpshalf=probe-tasktime,bps-half"
  "tt_bps1=probe-tasktime,bps-1"
  "tt_bps4=probe-tasktime,bps-4"
  "tt_bps8=probe-tasktime,bps-8"
  "ttu=probe-tasktime-untracked"
)

for spec in "${ARMS[@]}"; do
  arm=${spec%%=*}; feat=${spec#*=}
  echo "[$(date +%H:%M:%S)] building $arm (features: $feat)" >&2
  if [ "$feat" = "-" ]; then
    nice -n 19 cargo build --release --example bench_ab_decode >/dev/null
  else
    nice -n 19 cargo build --release --example bench_ab_decode --features "$feat" >/dev/null
  fi
  cp target/release/examples/bench_ab_decode "$OUT/bench_$arm"
done
echo "[$(date +%H:%M:%S)] wrote $OUT" >&2
ls -la "$OUT" >&2
