#!/usr/bin/env bash
# Sample-profile one arm binary on one vector and write the raw `sample` output.
#   prof_sample.sh <bin> <vec.avif> <threads> <out.txt> [iters] [seconds]
# NO `nice` (Darwin background QoS distorts wall clock ~40x).
set -u
BIN=${1:?bin}; VEC=${2:?vec}; T=${3:-1}; OUT=${4:?out}; ITERS=${5:-140}; SECS=${6:-60}
"$BIN" "$VEC" "$T" "$ITERS" 1 prof > "${OUT%.txt}.run" 2>&1 &
PID=$!
sleep 2
/usr/bin/sample "$PID" "$SECS" 1 -file "$OUT" >/dev/null 2>&1
wait $PID 2>/dev/null
echo "wrote $OUT" >&2
