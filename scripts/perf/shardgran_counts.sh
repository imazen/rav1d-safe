#!/usr/bin/env bash
# LEVER 2, priced WITHOUT a timer, at every rung of the granularity ladder.
#
# Two questions, two instruments, neither of which can be answered by a wall
# clock and both of which the AGENT_BRIEF says to answer FIRST:
#
#  1. `--features probe-wide` -- does the SHIPPED decoder promote to the wide
#     path at all, and does a coarser block change that? The wide path holds
#     EVERY active shard, so any rate there is disproportionate. Three doors
#     (w_shards / w_blocks / w_full) which move in OPPOSITE directions as the
#     block grows: a coarser block funnels more simultaneous borrows onto one
#     shard, so `w_full` (slot exhaustion) gets commoner exactly as the other two
#     get rarer.
#
#  2. `--features __probe_bounds` -- `pct_row_wide`, the quantity that refuted
#     the strided-2D record (`benchmarks/strided_2d_2026-08-10.meta` §4): the
#     fraction of would-be 2-D registrations whose row set spans more distinct
#     shards than `MAX_SHARDS_PER_BORROW`. This build also reports the same
#     fraction at caps 5 / 8 / 16, so "would raising the cap help?" is answered
#     from the same run instead of from another build.
#
# These builds are NOT timing-valid (`__probe_bounds` publishes, fences and scans
# on every registration; ~3.5x slower). They are counting runs, so they are NICED
# and take no measurement lock.
#
# Usage: shardgran_counts.sh <outdir>
# Env:   VEC (dir of .avif), CELLS ("<vec>:<threads>:<iters>"), GROUP (corpus group)
set -euo pipefail
OUT=${1:?outdir}
VEC=${VEC:-$HOME/tmp/t8gap/vec}
mkdir -p "$OUT"
cd "$(dirname "$0")/../.."

IFS=' ' read -r -a CELLS <<< "${CELLS:-L1024x576_420_8b__t8:8:20 L3840x2160_420_8b__t8:8:4 L1024x576_420_8b__t8:1:20}"

# rung=cargo-feature ("-" = shipped default)
IFS=' ' read -r -a RUNGS <<< "${RUNGS:-plain=- bpsq=bps-quarter bpshalf=bps-half bps1=bps-1 bps4=bps-4 bps8=bps-8}"
IFS=' ' read -r -a PROBES <<< "${PROBES:-probe-wide __probe_bounds}"

for spec in "${RUNGS[@]}"; do
  rung=${spec%%=*}; feat=${spec#*=}
  for probe in "${PROBES[@]}"; do
    tag="${rung}__${probe//_/}"
    if [ "$feat" = "-" ]; then f="$probe"; else f="$probe,$feat"; fi
    echo "[$(date +%H:%M:%S)] build $tag ($f)" >&2
    nice -n 19 cargo build --release --example probe_tracker --features "$f" >/dev/null
    cp target/release/examples/probe_tracker "$OUT/pt_$tag"
    for cell in "${CELLS[@]}"; do
      IFS=: read -r vec t iters <<< "$cell"
      echo "[$(date +%H:%M:%S)]   run $tag $vec t=$t x$iters" >&2
      nice -n 19 "$OUT/pt_$tag" "$VEC/$vec.avif" "$t" "$iters" \
        > "$OUT/${tag}__${vec}__t${t}.txt" 2>&1 || echo "FAILED $tag $vec t$t" >&2
      printf 'cell\t%s\t%s\t%s\t%s\t%s\n' "$rung" "$probe" "$vec" "$t" "$iters" \
        >> "$OUT/${tag}__${vec}__t${t}.txt"
    done
  done
done
echo "wrote $OUT" >&2
