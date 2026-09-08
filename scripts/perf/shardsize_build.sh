#!/usr/bin/env bash
# Build the binaries the PICTURE-SIZE sweep of the shard-granularity ladder needs.
#
# Two families, because the size question has a counting half and a timing half:
#
#   bench_ab_decode  — the timed arms (plain / the rungs / the tracker-free ceiling)
#   probe_tracker    — the timer-free counters (`__probe_wide` wide promotions,
#                      `__probe_bounds` row_shards / pct_row_wide), which is what
#                      actually tests the HEIGHT model in docs/SHARD_GRANULARITY.md §2
#                      and does not need a measurement lock.
#
# Copied to names cargo never writes, so a later build cannot swap the inode under
# a running measurement (AGENT_BRIEF §2).
#
# Builds are NICED. The measurement scripts never are.
set -euo pipefail
OUT=${1:-$HOME/tmp/shardsize/bin}
mkdir -p "$OUT"
cd "$(dirname "$0")/../.."

# arm=cargo-features ("-" = default features)
TIMED=(
  "plain=-"
  "bps1=__bps_1"
  "bpshalf=__bps_half"
  "bpsq=__bps_quarter"
  "bps4=__bps_4"
  "untracked=__probe_untracked"
)
COUNT=(
  "plain__probewide=__probe_wide"
  "bpshalf__probewide=__probe_wide,__bps_half"
  "bps1__probewide=__probe_wide,__bps_1"
  "plain__probebounds=__probe_bounds"
  "bpshalf__probebounds=__probe_bounds,__bps_half"
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
