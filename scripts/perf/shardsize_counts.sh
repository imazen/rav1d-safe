#!/usr/bin/env bash
# The TIMER-FREE half of the picture-size sweep.
#
# docs/SHARD_GRANULARITY.md §2 makes a falsifiable geometric claim: the tracker's
# adaptive rule targets a fixed BLOCK COUNT, so rows-per-block scales with picture
# HEIGHT (~h/256), and a filter's tap window — a fixed number of ROWS — therefore
# spreads over more shards as the picture gets shorter. Everything that claim
# predicts is countable without a clock:
#
#   row_shards_mean / row_shards_max   how many shard lines one strided access touches
#   pct_row_wide                       what fraction blow past MAX_SHARDS_PER_BORROW
#   shifts                             the block shift the instance actually used
#                                      (also this rung's liveness proof: a rung whose
#                                      shifts equal the default's did not arm)
#   multi / w_shards (probe-wide)      the SHIPPED multi-shard and all-shards counts
#
# So this runs first and the wall clock only has to confirm it. NICED, no
# measurement lock — `__probe_bounds` publishes/fences/scans on every registration
# (~20x slower here) and no wall number may be quoted from it.
#
# Usage: shardsize_counts.sh <outdir>
# Env:   VEC, BIN, CELLS ("<vec>:<threads>:<iters>" ...), TAGS
set -euo pipefail
OUT=${1:?outdir}
VEC=${VEC:-$HOME/tmp/shardsize/vec}
BIN=${BIN:-$HOME/tmp/shardsize/bin}
mkdir -p "$OUT"

# Iterations scale inversely with area so every cell aggregates a comparable
# number of registrations; the reported quantities are all per-frame or ratios.
DEFAULT_CELLS=""
for v in "$VEC"/*.avif; do
  n=$(basename "$v" .avif)
  wh=$(sed -E 's/^C([0-9]+)x([0-9]+)_.*/\1 \2/' <<< "$n")
  it=$(python3 -c "w,h=map(int,'$wh'.split()); print(max(4,min(40,round(20*589824/(w*h)))))")
  DEFAULT_CELLS="$DEFAULT_CELLS $n:8:$it"
done
IFS=' ' read -r -a CELLS <<< "${CELLS:-$DEFAULT_CELLS}"
IFS=' ' read -r -a TAGS <<< "${TAGS:-plain__probebounds bpshalf__probebounds plain__probewide bpshalf__probewide bps1__probewide}"

for tag in "${TAGS[@]}"; do
  b="$BIN/pt_$tag"
  [ -x "$b" ] || { echo "missing $b (run shardsize_build.sh)" >&2; exit 1; }
  for cell in "${CELLS[@]}"; do
    IFS=: read -r vec t iters <<< "$cell"
    o="$OUT/${tag}__${vec}__t${t}.txt"
    [ -s "$o" ] && continue
    echo "[$(date +%H:%M:%S)] $tag $vec t=$t x$iters" >&2
    nice -n 19 "$b" "$VEC/$vec.avif" "$t" "$iters" > "$o" 2>&1 \
      || echo "FAILED $tag $vec" >&2
    printf 'cell\t%s\t%s\t%s\t%s\n' "$tag" "$vec" "$t" "$iters" >> "$o"
  done
done
echo "wrote $OUT" >&2
