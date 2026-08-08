#!/usr/bin/env bash
# Concurrent multi-decoder pressure: N processes x M in-process decoders, mixed
# thread counts, all on the same inputs, every output md5 compared to a serial
# reference md5 taken first.
#
# What this is for. The borrow tracker's shard locks, the tile-worker pool and
# (new on this compose) loop restoration's per-thread scratch are all things a
# single quiet decode cannot stress. The failure modes this reproduces are:
#   * a tile-worker `overlapping DisjointMut` panic under oversubscription,
#   * the futex wedge that followed one (rav1d-safe#422 / zenavif#30 shape):
#     0% CPU, no output, forever — which is why there is a hard timeout here
#     and why a timeout is reported as a FAILURE, not as "still running",
#   * cross-thread contamination of any process-global latch
#     (`set_parallelism`, `set_tile_concurrency`) — caught by the md5 compare,
#     since a wrong block shift is a perf choice but a wrong *anything else*
#     is a wrong pixel.
#
# NOT a timing harness: this deliberately oversubscribes the box.
#
# Usage: multi_decoder_pressure.sh <bench_binary> <procs> <iters> [timeout_s]
# Env:   AVIF (dir of .avif inputs), VECS (space-separated basenames)
set -u
BENCH=${1:?path to bench_ab_decode}
PROCS=${2:-12}
ITERS=${3:-3}
TMO=${4:-300}
AVIF=${AVIF:-$HOME/tmp/rav1d-perf/vec}
IFS=' ' read -r -a VECS <<< "${VECS:-v4k_8tile v4k_8tile_10b v4k_1tile v1024 v256}"
WORK=$(mktemp -d "${TMPDIR:-$HOME/tmp}/mdp.XXXXXX")
trap 'rm -rf "$WORK"' EXIT

echo "== reference md5s (serial, t=1) =="
for v in "${VECS[@]}"; do
  "$BENCH" "$AVIF/$v.avif" 1 1 1 ref 2>/dev/null | awk -F'\t' '/^CHECKSUM/{print $5}' > "$WORK/ref.$v"
  echo "  $v $(cat "$WORK/ref.$v")"
  [ -s "$WORK/ref.$v" ] || { echo "FAIL: no reference md5 for $v"; exit 1; }
done

echo "== $PROCS concurrent decoders x $ITERS iters, thread counts 1/2/4/8/16 =="
pids=(); i=0
for p in $(seq 0 $((PROCS-1))); do
  v=${VECS[$((p % ${#VECS[@]}))]}
  t=$(( 1 << (p % 5) ))   # 1 2 4 8 16
  ( "$BENCH" "$AVIF/$v.avif" "$t" "$ITERS" 1 "p$p" > "$WORK/out.$p" 2> "$WORK/err.$p"
    echo $? > "$WORK/rc.$p" ) &
  pids+=($!); i=$((i+1))
done

# Hard deadline: a wedged tile worker sits at 0% CPU forever, and "the harness
# is still going" must never be how that is discovered.
deadline=$((SECONDS + TMO))
while :; do
  live=0
  for pid in "${pids[@]}"; do kill -0 "$pid" 2>/dev/null && live=$((live+1)); done
  [ "$live" -eq 0 ] && break
  if [ $SECONDS -ge $deadline ]; then
    echo "FAIL: $live decoder(s) still alive after ${TMO}s — WEDGE"
    for pid in "${pids[@]}"; do kill -9 "$pid" 2>/dev/null; done
    exit 2
  fi
  sleep 2
done

fail=0
for p in $(seq 0 $((PROCS-1))); do
  v=${VECS[$((p % ${#VECS[@]}))]}
  rc=$(cat "$WORK/rc.$p" 2>/dev/null || echo 99)
  got=$(awk -F'\t' '/^CHECKSUM/{print $5}' "$WORK/out.$p" 2>/dev/null)
  want=$(cat "$WORK/ref.$v")
  if [ "$rc" != 0 ]; then
    echo "FAIL p$p ($v): exit $rc"; sed -n '1,6p' "$WORK/err.$p"; fail=1
  elif [ "$got" != "$want" ]; then
    echo "FAIL p$p ($v): md5 $got != $want"; fail=1
  fi
done
[ $fail -eq 0 ] && echo "PASS: $PROCS concurrent decoders, all md5s match the serial reference"
exit $fail
