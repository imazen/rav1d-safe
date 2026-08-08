#!/usr/bin/env bash
# Interleaved A/B of the aarch64 NEON loop-restoration tier against the generic
# scalar reference it dispatches over, via the `__ablate` switch.
#
#   arm "neon"   = safe_simd/looprestoration_arm.rs   (what ships)
#   arm "scalar" = src/looprestoration.rs             (dispatch returns false)
#
# SAME BINARY on both arms — only which implementation runs differs — so this
# cannot be confounded by codegen, layout or build flags.
#
# Differences from the older `lr_ab.sh` this supersedes:
#   * it WAITS for the box to go idle before a cell and DISCARDS + reruns any
#     cell during which a foreign process went over 25% CPU, instead of merely
#     recording a `foreign` column (the 2026-08-07 run had to be caveated
#     because a sibling agent's benchmark was resident);
#   * arm order rotates per round AND the whole round is committed or thrown
#     away as a unit, so the two arms of a kept round saw the same box.
#
# NO `nice` ON A TIMED RUN — Darwin maps a niced process onto E-cores and the
# wall clock distorts by ~40x. NO -C target-cpu=native. Default features.
#
# It also does cross-BUILD A/B: each ARMS entry is `name:binary:ablate-list`,
# where an empty binary means $BIN and an empty ablate-list means none. So
#   ARMS="neon::  scalar::looprestoration"      is the same-binary ablation A/B
#   ARMS="v2:$D/pv2:  v1:$D/pv1:"               is two commits interleaved
# and both get the same wait-for-idle / discard-the-whole-round treatment.
#
# Usage: lr_neon_ab.sh <out.tsv> [rounds]
# Env:   BIN (a profile_ivf built --features __ablate), VECDIR, CELLS, ARMS
#        CELLS entries are "<relative ivf path without .ivf>:<iters>"
#
# Columns: round  arm  vec  iters  ms_per_frame  foreign_max
set -u
OUT=${1:?out.tsv}; ROUNDS=${2:-7}
BIN=${BIN:?set BIN to a profile_ivf built with --features __ablate}
VECDIR=${VECDIR:-test-vectors/dav1d-test-data}
IFS=' ' read -r -a CELLS <<< "${CELLS:-8-bit/data/00001147:6 10-bit/issues/318_tx_4x4:6 8-bit/data/00000645:3 8-bit/data/00000855:40}"
IFS=' ' read -r -a ARMS <<< "${ARMS:-neon:: scalar::looprestoration}"

BIN_RE=$(printf '%s' "$BIN" | sed 's/[][\.*^$/(){}?+|]/\\&/g')
busy_count() {
  ps -A -o %cpu,comm -r | awk -v me="$BIN_RE" \
    'NR>1 && $1>25 && $2 !~ /claude|ClaudeCode|versions\// && $2 !~ me {c++} END {print c+0}'
}
wait_quiet() {
  local w=0
  while [ "$(busy_count)" -gt 0 ]; do
    sleep 5; w=$((w+5))
    [ $w -ge 3600 ] && { echo "box never went idle" >&2; exit 4; }
  done
}

: > "$OUT"
for round in $(seq 0 $((ROUNDS-1))); do
  for cell in "${CELLS[@]}"; do
    vec=${cell%:*}; iters=${cell##*:}
    while :; do
      wait_quiet; rows=(); dirty=0; fmax=0
      n=${#ARMS[@]}
      for k in $(seq 0 $((n-1))); do
        spec=${ARMS[$(( (k + round) % n ))]}
        arm=${spec%%:*}; rest=${spec#*:}; bin=${rest%%:*}; abl=${rest#*:}
        [ -z "$bin" ] && bin=$BIN
        ms=$(RAV1D_ABLATE="$abl" RAV1D_LABEL="$arm" "$bin" "$VECDIR/$vec.ivf" "$iters" 2>/dev/null \
             | awk -F'\t' '/^RESULT/{print $6}')
        f=$(busy_count); [ "$f" -gt "$fmax" ] && fmax=$f
        [ "$f" -gt 0 ] && dirty=1
        rows+=("$round	$arm	$vec	$iters	$ms")
      done
      if [ $dirty -eq 0 ]; then
        for r in "${rows[@]}"; do printf '%s\t%s\n' "$r" "$fmax" >> "$OUT"; done
        echo "[$(date +%H:%M:%S)] r$round $vec committed (idle)" >&2
        break
      fi
      echo "[$(date +%H:%M:%S)] r$round $vec DISCARDED (foreign=$fmax)" >&2
    done
  done
done
echo "wrote $OUT" >&2
