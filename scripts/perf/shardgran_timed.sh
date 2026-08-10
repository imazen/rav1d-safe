#!/usr/bin/env bash
# The TIMED half of the granularity sweep, in two measlock holds.
#
# Phase T -- TAIL: `probe-tasktime` per rung, so each rung is scored on tail
#   concurrency (the objective the original fit did NOT use) as well as on wall.
#   Low arm is t=2, NOT t=1: at `--threads 1` every stage counter reads 0.000
#   because `rav1d_task_run` is never entered, so a t8/t1 per-stage ratio does not
#   exist (AGENT_BRIEF §2).
#
# Phase W -- WALL/CPU vs dav1d in the SAME interleaved sweep, two frame counts per
#   cell so `total = a + b*frames` drops process startup, and t=1 as well as t=8
#   because the three-term decomposition needs cpu(1).
#
# NO `nice` here -- Darwin maps a positive nice to background QoS and lands the
# process on E-cores (~40x distortion). Builds are niced; timed runs never are.
#
# Usage: shardgran_timed.sh <outdir> [tail_rounds] [wall_rounds]
set -u
OUT=${1:?outdir}; TR=${2:-5}; WR=${3:-5}
BIN=${BIN:-$HOME/tmp/shardgran/bin}
VEC=${VEC:-$HOME/tmp/t8gap/vec}
IVF=${IVF:-$HOME/tmp/t8gap/ivf}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
export PATH="$HOME/bin:$PATH"

TAIL_ARMS=${TAIL_ARMS:-"tt tt_bpsq tt_bpshalf tt_bps1 tt_bps4 tt_bps8 ttu"}
TAIL_CELLS=${TAIL_CELLS:-"L1024x576_420_8b__t8:8:100 L3840x2160_420_8b__t8:8:20 L1024x576_420_8b__t8:2:100 L3840x2160_420_8b__t8:2:20"}
WALL_ARMS=${WALL_ARMS:-"plain bpsq bpshalf bps1 bps4 bps8 untracked dav1d_fd1"}
WALL_CELLS=${WALL_CELLS:-"L1024x576_420_8b__t8:1:20:200 L1024x576_420_8b__t8:8:20:200 L3840x2160_420_8b__t8:1:2:16 L3840x2160_420_8b__t8:8:2:16"}

echo "[$(date +%H:%M:%S)] phase T (tail concurrency), $TR rounds" >&2
BIN="$BIN" VEC="$VEC" ARMS="$TAIL_ARMS" CELLS="$TAIL_CELLS" \
  measlock shardgran-tail -- "$here/tiled_taskprobe.sh" "$OUT/probe" "$TR"

echo "[$(date +%H:%M:%S)] phase W (wall+cpu vs dav1d), $WR rounds" >&2
BIN="$BIN" AVIF="$VEC" IVF="$IVF" ARMS="$WALL_ARMS" CELLS="$WALL_CELLS" \
  measlock shardgran-wallcpu -- "$here/tiled_wallcpu.sh" "$OUT/wallcpu.tsv" "$WR"

echo "[$(date +%H:%M:%S)] done -> $OUT" >&2
