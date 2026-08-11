#!/usr/bin/env bash
# The CONTENTION census for `c256x2048` t=8 — how often a shard lock is actually
# fought over, and how long the fight lasts.
#
# WHY THIS IS NOT `c256_counts.sh`: that script is `nice -n 19`, which on Darwin
# maps to background QoS and lands every worker on the E-cores. Registration
# COUNTS are indifferent to that; CONTENTION counts are not — eight workers on
# four E-cores collide roughly ten times as often as eight workers on eight
# P-cores. A first pass niced read `lockslow = 1072/frame` where the un-niced
# build reads ~104. So this census runs UN-NICED and under `measlock`, exactly
# like a timed arm, even though it only reports counters.
#
# Counters (all `--features probe-wide`, all per frame):
#   contended  the single-block fast path's `try_lock` LOST
#   lockslow   the retry inside `lock()` ALSO lost, so the thread spun
#   spins      total spin-loop iterations, i.e. the entire cost `lock_slow` can
#              possibly be carrying
#   multi      multi-shard registrations (2+ locks each)
#   w_*        wide-path promotions — 0 is the precondition for any refinement
#
# Usage: c256_contention.sh <out.tsv> [reps]
# Env:   VEC BIN CELLS ARMS ITERS
set -u
export LC_ALL=C
OUT=${1:?out.tsv}; REPS=${2:-5}
BIN=${BIN:-$HOME/tmp/c256/bin}
VEC=${VEC:-$HOME/tmp/bpsrows/vec}
ITERS=${ITERS:-12}
IFS=' ' read -r -a ARMS <<< "${ARMS:-plain__probewide backoff__probewide yield__probewide park__probewide}"
IFS=' ' read -r -a CELLS <<< "${CELLS:-C256x2048_420_8b__t8:8 C256x2048_420_8b__t8:1 C1024x576_420_8b__t8:8}"

printf 'rep\tarm\tpin\tvec\tthreads\titers\tslow\tmulti\tw_shards\tw_blocks\tw_full\tcontended\tlockslow\tspins\tms_per_frame\n' > "$OUT"
for rep in $(seq 0 $((REPS-1))); do
  for cell in "${CELLS[@]}"; do
    IFS=: read -r vec t <<< "$cell"
    for arm in "${ARMS[@]}"; do
      pin=${PIN:--}
      if [ "$pin" = "-" ]; then
        r=$("$BIN/pt_$arm" "$VEC/$vec.avif" "$t" "$ITERS" 2>/dev/null)
      else
        r=$(RAV1D_PIN_SHIFT="$pin" "$BIN/pt_$arm" "$VEC/$vec.avif" "$t" "$ITERS" 2>/dev/null)
      fi
      w=$(grep -m1 '^WIDE	' <<< "$r" | cut -f2-)
      ms=$(grep -m1 '^RUN' <<< "$r" | sed -E 's/.*ms_per_frame=([0-9.]+).*/\1/')
      # w = const_shift slow multi w_shards w_blocks w_full wide_total contended lockslow spins
      IFS=$'\t' read -r _cs slow multi ws wb wf _wt cd ls sp <<< "$w"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$rep" "$arm" "$pin" "$vec" "$t" "$ITERS" \
        "$slow" "$multi" "$ws" "$wb" "$wf" "$cd" "$ls" "$sp" "$ms" >> "$OUT"
    done
    echo "[$(date +%H:%M:%S)] rep$rep $vec t=$t done" >&2
  done
done
echo "wrote $OUT" >&2
