#!/usr/bin/env bash
# The TIMER-FREE half of the `c256x2048` t=8 contention round.
#
# Everything the two levers predict is countable before any clock is involved,
# and one of the predictions is a KILL SWITCH for lever 1:
#
#   lockslow / spins       (probe-wide, added this round) how often a thread had
#                          to WAIT for a shard lock, and how deep the wait was.
#                          This is the quantity `c256x2048` is limited by and
#                          the one the multi/wide counters cannot see.
#   multi / w_shards /     the SHIPPED multi-shard and all-shards promotions. A
#   w_blocks / w_full      finer block makes a strided borrow touch MORE shards;
#                          past MAX_SHARDS_PER_BORROW it promotes to the wide
#                          path, which holds every shard. If refining pushes
#                          this cell onto the wide path, lever 1 is dead and
#                          this table says so BEFORE anything is timed.
#   pct_row_wide /         (__probe_bounds) the strided-2-D counterfactual at
#   row_shards_max         each per-row helper.
#   shifts                 the tracker's OWN per-instance shift — the pin's
#                          liveness proof, not a prediction.
#
# NICED, no measurement lock: `__probe_bounds` publishes/fences/scans on every
# registration and no wall number from it is valid.
#
# Usage: c256_counts.sh <outdir>
# Env:   VEC BIN CELLS PINS
set -euo pipefail
OUT=${1:?outdir}
VEC=${VEC:-$HOME/tmp/bpsrows/vec}
BIN=${BIN:-$HOME/tmp/c256/bin}
mkdir -p "$OUT"

# "<vector>:<threads>:<iters>"
IFS=' ' read -r -a CELLS <<< "${CELLS:-C256x2048_420_8b__t8:8:12}"
# "<tag>=<RAV1D_PIN_SHIFT value>"; "-" = no pin (the shipped rule).
IFS=' ' read -r -a PINS <<< "${PINS:-base=-}"

for tag_spec in "${PINS[@]}"; do
  tag=${tag_spec%%=*}; pin=${tag_spec#*=}
  for probe in probewide probebounds; do
    b="$BIN/pt_pin__$probe"
    [ -x "$b" ] || { echo "missing $b (run c256_build.sh)" >&2; exit 1; }
    for cell in "${CELLS[@]}"; do
      IFS=: read -r vec t iters <<< "$cell"
      # __probe_bounds is ~20x slower; it needs fewer iterations for the same
      # per-frame quantities (all of them are ratios or per-frame counts).
      [ "$probe" = probebounds ] && iters=$(( iters > 4 ? 4 : iters ))
      o="$OUT/${probe}__${tag}__${vec}__t${t}.txt"
      [ -s "$o" ] && continue
      echo "[$(date +%H:%M:%S)] $probe $tag $vec t=$t x$iters" >&2
      if [ "$pin" = "-" ]; then
        nice -n 19 "$b" "$VEC/$vec.avif" "$t" "$iters" > "$o" 2>&1 \
          || echo "FAILED $probe $tag $vec" >&2
      else
        RAV1D_PIN_SHIFT="$pin" nice -n 19 "$b" "$VEC/$vec.avif" "$t" "$iters" > "$o" 2>&1 \
          || echo "FAILED $probe $tag $vec" >&2
      fi
      printf 'cell\t%s\t%s\t%s\t%s\t%s\n' "$tag" "$pin" "$vec" "$t" "$iters" >> "$o"
    done
  done
done
echo "wrote $OUT" >&2
