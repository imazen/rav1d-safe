#!/usr/bin/env bash
# Rotating-order interleaved A/B over (arm x vector x threads), in-process timer.
#
# Generalises scripts/perf/bin_ab.sh, which is pinned to t=1 and to one staged
# bin directory. This campaign needs the SAME cell measured at two bit depths
# and two thread counts, because the effect under test (the tracker's block
# shift) has opposite signs across those axes — see
# benchmarks/tracker_blockshift_bpc_2026-08-08.meta.
#
# The timer is `bench_ab_decode`'s own `ms_per_frame` (a warmed-up in-process
# `Instant`), not process wall clock: at t=1 on a 4K vector the decode is
# hundreds of ms and startup is ~30 ms, so the two-point wall fit buys nothing
# the warmup does not already give, and one process per cell keeps the
# `set_tile_threading` / `set_parallelism` latches honest.
#
# NO `nice` on a timed run — Darwin maps a niced process onto E-cores.
#
# Usage: tshift_ab.sh <out.tsv> <rounds> <iters>
# Env:   BIN (dir of bench_<arm>), VECDIR, ARMS, CELLS ("vec:threads" list)
# Out:   round  arm  vec  threads  ms_per_frame  md5  foreign
set -u
export LC_ALL=C
OUT=${1:?out.tsv}; ROUNDS=${2:-5}; ITERS=${3:-6}
BIN=${BIN:-$HOME/tmp/bpcshift/bin}
VECDIR=${VECDIR:-$HOME/tmp/rav1d-perf/vec}
IFS=' ' read -r -a ARMS <<< "${ARMS:-base adapt}"
IFS=' ' read -r -a CELLS <<< "${CELLS:-v4k_8tile:1 v4k_8tile_10b:1 v4k_8tile:8 v4k_8tile_10b:8}"

# Foreign = >25% CPU and not us. The arms themselves are excluded via $BIN so
# macOS's decaying %cpu for a just-exited arm cannot poison the next cell.
BIN_RE=$(printf '%s' "$BIN" | sed 's/[][\.*^$/(){}?+|]/\\&/g')
busy() {
  ps -A -o %cpu,comm -r | awk -v me="$BIN_RE" \
    'NR>1 && $1>25 && $2 !~ /claude|ClaudeCode|versions\/|python3/ && $2 !~ me {c++} END {print c+0}'
}

: > "$OUT"
n=${#ARMS[@]}
for r in $(seq 0 $((ROUNDS-1))); do
  for cell in "${CELLS[@]}"; do
    IFS=: read -r vec t <<< "$cell"
    for k in $(seq 0 $((n-1))); do
      arm=${ARMS[$(( (k + r) % n ))]}
      out=$("$BIN/bench_$arm" "$VECDIR/$vec.avif" "$t" "$ITERS" 1 "$arm" 2>/dev/null)
      ms=$(printf '%s' "$out" | awk -F'\t' '/^RESULT/{print $8}')
      md5=$(printf '%s' "$out" | awk -F'\t' '/^CHECKSUM/{print $5}')
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$r" "$arm" "$vec" "$t" "$ms" "$md5" "$(busy)" >> "$OUT"
    done
    echo "[$(date +%H:%M:%S)] r$r $vec t=$t" >&2
  done
done
echo "wrote $OUT" >&2
