#!/usr/bin/env bash
# Self-time profiles of the TILED arm at several thread counts.
#
# Every profile taken in this campaign so far is single-tile or t=1. This takes
# the same vector at t=1 / t=2 / t=8 with the SHIPPING binary (no probe feature,
# so the codegen is what ships) and lets the report diff the self-time leaves.
# The question is only ever "what appears at t=8 that is absent at t=2" -- so
# the two arms must be the same binary on the same input, differing in one
# argument.
#
# `sample` is a 1 ms wall sampler over ALL threads, so its totals scale with
# thread count: at t=8 a 30 s window collects ~8x the samples of t=1. Compare
# PERCENTAGES between cells, and total sample counts only within a cell.
#
# NO `nice` (Darwin maps it to background QoS -> E-cores). Run under `measlock`.
#
# Usage: tiled_prof.sh <outdir> [seconds]
# Env:   BIN, VEC, CELLS (<vector>:<threads>:<iters>)
#        MODE=avif|ivf|profivf
#                        -- `ivf` drives bench_ivf_limit off VEC/<vec>.ivf so
#                           RAV1D_INLOOP is available (see INLOOP). It EXITS AT
#                           END OF STREAM, so it cannot outlive a sample window
#                           (200 frames = 0.66 s at t=8) -- use `profivf`
#                           (examples/profile_ivf), which loops `iters` passes
#                           and takes threads from RAV1D_THREADS. That is the
#                           tool docs/AGENT_BRIEF.md §7 already names for this.
#        INLOOP=all|nodeblock|nocdef|norestoration|none  (MODE=ivf only)
#                           CHANGES OUTPUT PIXELS -- attribution only. Used to
#                           test whether a stage's measured cost DISAPPEARS when
#                           its work does: with a spinning lock it need not.
#        TAG=<suffix>    -- appended to the output name so two INLOOP values of
#                           the same cell do not overwrite each other
set -u
export LC_ALL=C
OUT=${1:?outdir}; SECS=${2:-30}
MODE=${MODE:-avif}
INLOOP=${INLOOP:-all}
TAG=${TAG:-}
BIN=${BIN:-$HOME/tmp/tiledprof/bin/bench_plain}
VEC=${VEC:-$HOME/tmp/t8gap/vec}
IFS=' ' read -r -a CELLS <<< "${CELLS:-\
L1024x576_420_8b__t8:1:3000 L1024x576_420_8b__t8:2:6000 L1024x576_420_8b__t8:8:14000 \
L3840x2160_420_8b__t8:1:250  L3840x2160_420_8b__t8:2:400 L3840x2160_420_8b__t8:8:1400 \
L1024x576_420_8b:8:3000}"
mkdir -p "$OUT"
for cell in "${CELLS[@]}"; do
  IFS=: read -r vec t iters <<< "$cell"
  tag="${vec}__t${t}${TAG}"
  if [ "$MODE" = profivf ]; then
    RAV1D_INLOOP="$INLOOP" RAV1D_THREADS="$t" "$BIN" "$VEC/$vec.ivf" "$iters" \
      > "$OUT/$tag.run" 2>&1 &
  elif [ "$MODE" = ivf ]; then
    RAV1D_INLOOP="$INLOOP" "$BIN" "$VEC/$vec.ivf" "$t" "$iters" prof \
      > "$OUT/$tag.run" 2>&1 &
  else
    "$BIN" "$VEC/$vec.avif" "$t" "$iters" 1 prof > "$OUT/$tag.run" 2>&1 &
  fi
  PID=$!
  sleep 2
  /usr/bin/sample "$PID" "$SECS" 1 -file "$OUT/$tag.sample" >/dev/null 2>&1
  # The payload is sized to outlive the window; if it exited early the sample is
  # short and the report must say so, so record the fact instead of hiding it.
  if kill -0 "$PID" 2>/dev/null; then echo "outlived_window=1" >> "$OUT/$tag.run"; else echo "outlived_window=0" >> "$OUT/$tag.run"; fi
  wait $PID 2>/dev/null
  echo "[$(date +%H:%M:%S)] $tag done" >&2
done
echo "wrote $OUT" >&2
