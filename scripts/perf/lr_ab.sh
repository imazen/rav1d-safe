#!/usr/bin/env bash
# Interleaved A/B of the aarch64 "NEON" loop-restoration duplicate against the
# generic scalar reference it shadows, via the __ablate switch.
#
#   arm "arm"    = safe_simd/looprestoration_arm.rs  (what ships)
#   arm "scalar" = src/looprestoration.rs            (dispatch returns false)
#
# Rotating arm order per round, one process per cell, median taken by the
# caller. NO nice on a timed run.
set -u
BIN=${BIN:-./target/release/examples/profile_ivf}
OUT=${1:?out.tsv}
VEC=${2:?vec.ivf}
ITERS=${3:-20}
ROUNDS=${4:-7}
busy() {
  ps -A -o %cpu,comm -r | awk 'NR>1 && $1>25 && $2 !~ /claude|ClaudeCode|versions\/|profile_ivf/ {c++} END {print c+0}'
}
: > "$OUT"
for r in $(seq 0 $((ROUNDS-1))); do
  for k in 0 1; do
    if [ $(( (k + r) % 2 )) -eq 0 ]; then A=arm; ABL=""; else A=scalar; ABL="looprestoration"; fi
    RAV1D_ABLATE="$ABL" RAV1D_LABEL="$A" "$BIN" "$VEC" "$ITERS" 2>/dev/null \
      | awk -v r="$r" -v f="$(busy)" -F'\t' '/^RESULT/{print r"\t"$2"\t"$6"\t"f}' >> "$OUT"
  done
  echo "[$(date +%H:%M:%S)] round $r done" >&2
done
