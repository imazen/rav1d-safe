#!/usr/bin/env bash
# Build the arms for the "derived rows-per-block rule as the DEFAULT" round.
#
# The polarity is the inverse of the previous two rounds: the thing under test
# is the DEFAULT build, and the BASE it must be differenced against is
# `bps-blocks` — the block-count rule that shipped before 2026-08-11. A sweep
# that quotes a `--features` arm as if it were the shipped decoder is exactly
# the reporting error this round exists to fix.
#
# Copied to names cargo never writes, so a later build cannot swap the inode
# under a running measurement (AGENT_BRIEF §2).
#
# Builds are NICED. The measurement scripts never are.
set -euo pipefail
OUT=${1:-$HOME/tmp/bpsrows/bin}
mkdir -p "$OUT"
cd "$(dirname "$0")/../.."

# arm=cargo-features ("-" = default features = the SHIPPED decoder)
TIMED=(
  "plain=-"                      # HEAD: the derived rows-per-block rule
  "bpsblocks=bps-blocks"         # BASE: the pre-2026-08-11 block-count rule
  "bpshalf=bps-half"             # the best global constant the ladder offers
  "untracked=probe-untracked"    # tracker-removed ceiling (bit-identical)
)
COUNT=(
  "plain__probebounds=__probe_bounds"
  "bpsblocks__probebounds=__probe_bounds,bps-blocks"
  "plain__probewide=probe-wide"
  "bpsblocks__probewide=probe-wide,bps-blocks"
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
