#!/usr/bin/env bash
# Price the DEBLOCK chain on BOTH decoders with one instrument.
#
# §1 of docs/TILED_SCALING.md attributes the tiled t=8 CPU inflation to the
# borrow tracker inside `filter_sbrow_deblock_cols`. That is an ours-only
# measurement: the task probe cannot be pointed at dav1d. This arm can --
# `RAV1D_INLOOP` takes dav1d's own `--inloopfilters` spelling, so `all` vs
# `nodeblock` is literally the same string on both sides, and the difference is
# each decoder's own deblock cost, measured by the same clock.
#
# The point is the SCALING of that difference: if our deblock chain costs
# proportionally the same as dav1d's at t=1 and more at t=8, the inflation is
# ours and it is threading-dependent, which is exactly the claim.
#
# CHANGES OUTPUT PIXELS. Attribution only -- never compare an md5 across
# `RAV1D_INLOOP` / `--inloopfilters` values, and no `RAV1D_MD5=1` on a timed run
# (it is a per-pixel hash against a `--muxer null` arm that hashes nothing).
#
# NO `nice`. Run under `measlock`.
#
# Usage: tiled_inloop_ab.sh <out.tsv> [rounds]
# Env:   BIN IVF ARMS CELLS
# Cell:  <vector>:<threads>:<n_lo>:<n_hi>
# Cols:  round arm inloop vec threads nlo wall_lo user_lo sys_lo nhi wall_hi user_hi sys_hi foreign
set -u
export LC_ALL=C
OUT=${1:?out.tsv}; ROUNDS=${2:-5}
BIN=${BIN:-$HOME/tmp/tiledprof/bin}
IVF=${IVF:-$HOME/tmp/t8gap/ivf}
IFS=' ' read -r -a ARMS <<< "${ARMS:-rs untracked dav1d}"
IFS=' ' read -r -a INLOOPS <<< "${INLOOPS:-all nodeblock}"
DEFAULT_CELLS=""
for t in 1 8; do
  DEFAULT_CELLS="$DEFAULT_CELLS L1024x576_420_8b__t8:$t:20:200"
  DEFAULT_CELLS="$DEFAULT_CELLS L3840x2160_420_8b__t8:$t:2:16"
done
IFS=' ' read -r -a CELLS <<< "${CELLS:-$DEFAULT_CELLS}"

BIN_RE=$(printf '%s' "$BIN" | sed 's/[][\.*^$/(){}?+|]/\\&/g')
busy_count() {
  ps -A -o %cpu,comm -r | awk -v me="$BIN_RE" \
    'NR>1 && $1>25 && $2 !~ /claude|ClaudeCode|versions\/|dav1d|python3/ && $2 !~ me {c++} END {print c+0}'
}

TIMEFORMAT='%3R %3U %3S'
time_one() {
  local arm=$1 il=$2 vec=$3 t=$4 n=$5 r
  case "$arm" in
    dav1d) r=$( { time dav1d -i "$IVF/$vec.ivf" --muxer null --threads "$t" --framedelay 1 \
                       --inloopfilters "$il" -q --limit "$n" >/dev/null 2>/dev/null; } 2>&1 ) ;;
    rs)    r=$( { time env RAV1D_INLOOP="$il" "$BIN/ivf_plain" "$IVF/$vec.ivf" "$t" "$n" w \
                       >/dev/null 2>/dev/null; } 2>&1 ) ;;
    *)     r=$( { time env RAV1D_INLOOP="$il" "$BIN/ivf_$arm" "$IVF/$vec.ivf" "$t" "$n" w \
                       >/dev/null 2>/dev/null; } 2>&1 ) ;;
  esac
  awk '{printf "%d %d %d\n", $1*1000+0.5, $2*1000+0.5, $3*1000+0.5}' <<< "$r"
}

: > "$OUT"
n=${#ARMS[@]}
for round in $(seq 0 $((ROUNDS-1))); do
  for cell in "${CELLS[@]}"; do
    IFS=: read -r vec t nlo nhi <<< "$cell"
    for il in "${INLOOPS[@]}"; do
      for k in $(seq 0 $((n-1))); do
        arm=${ARMS[$(( (k + round) % n ))]}
        f0=$(busy_count)
        read -r wlo ulo slo <<< "$(time_one "$arm" "$il" "$vec" "$t" "$nlo")"
        read -r whi uhi shi <<< "$(time_one "$arm" "$il" "$vec" "$t" "$nhi")"
        f1=$(busy_count); f=$(( f0 > f1 ? f0 : f1 ))
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
          "$round" "$arm" "$il" "$vec" "$t" "$nlo" "$wlo" "$ulo" "$slo" \
          "$nhi" "$whi" "$uhi" "$shi" "$f" >> "$OUT"
      done
    done
    echo "[$(date +%H:%M:%S)] r$round $vec t=$t done" >&2
  done
done
echo "wrote $OUT" >&2
