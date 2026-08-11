#!/usr/bin/env bash
# Fresh cost census, timing half: ours(plain) / ceiling(untracked) / dav1d, per cell.
#
# Same instrument on both sides as scripts/perf/verify_gap.sh: whole-process wall
# at TWO frame counts, `total = alpha + beta*frames` fitted so process startup
# drops out. Arms rotate inside every round. NLO/NHI are PER CELL and never
# exceed the stream's frame count (AGENT_BRIEF: `--limit N` past end of stream
# halves the gap in dav1d's favour).
#
# Usage: gap_sweep.sh <out.tsv> [rounds]
# Columns: round arm vec threads nlo ms_lo nhi ms_hi foreign_max
set -u
export LC_ALL=C
OUT=${1:?out.tsv}; ROUNDS=${2:-7}
BIN=$HOME/tmp/census/bin
AVIF=$HOME/tmp/census/stage/avif
IVF=$HOME/tmp/census/stage/ivf
IFS=' ' read -r -a ARMS <<< "${ARMS:-plain untracked dav1d_fd1}"
# vec:threads:nlo:nhi
IFS=' ' read -r -a CELLS <<< "${CELLS:-\
ui_q20:8:20:200 ui_q20:1:20:200 text_q20:8:20:200 text_q20:1:20:200 \
c256x2048:8:20:200 c256x2048:1:20:200 c1024x384:8:20:200 c1024x192:8:20:200 \
c3840x256:8:12:120 c1024x576:8:20:200 \
v4k8tile:8:4:40 v4k8tile:4:4:40 v4k8tile:2:4:40 v4k8tile:1:4:40}"

BIN_RE=$(printf '%s' "$BIN" | sed 's/[][\.*^$/(){}?+|]/\\&/g')
busy_count() {
  ps -A -o %cpu,comm -r | awk -v me="$BIN_RE" \
    'NR>1 && $1>25 && $2 !~ /claude|ClaudeCode|versions\/|dav1d|python3/ && $2 !~ me {c++} END {print c+0}'
}
now_ms() { local t=$EPOCHREALTIME; echo $(( ${t%%.*} * 1000 + 10#${t#*.} / 1000 )); }

time_one() { # arm vec threads n
  local arm=$1 vec=$2 t=$3 n=$4 t0 t1
  t0=$(now_ms)
  case "$arm" in
    dav1d_fd1) dav1d -i "$IVF/$vec.ivf" --muxer null --threads "$t" --framedelay 1 -q --limit "$n" >/dev/null 2>&1 ;;
    dav1d_def) dav1d -i "$IVF/$vec.ivf" --muxer null --threads "$t"                 -q --limit "$n" >/dev/null 2>&1 ;;
    *)         "$BIN/bench_$arm" "$AVIF/$vec.avif" "$t" "$n" 1 w >/dev/null 2>&1 ;;
  esac
  t1=$(now_ms); echo $((t1 - t0))
}

printf 'round\tarm\tvec\tthreads\tnlo\tms_lo\tnhi\tms_hi\tforeign_max\n' > "$OUT"
for r in $(seq 1 "$ROUNDS"); do
  for cell in "${CELLS[@]}"; do
    IFS=: read -r vec t nlo nhi <<< "$cell"
    # rotate the arm order every round so no arm always runs first
    na=${#ARMS[@]}; off=$(( (r - 1) % na ))
    for i in $(seq 0 $((na - 1))); do
      arm=${ARMS[$(( (i + off) % na ))]}
      fmax=$(busy_count)
      lo=$(time_one "$arm" "$vec" "$t" "$nlo")
      f=$(busy_count); [ "$f" -gt "$fmax" ] && fmax=$f
      hi=$(time_one "$arm" "$vec" "$t" "$nhi")
      f=$(busy_count); [ "$f" -gt "$fmax" ] && fmax=$f
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$r" "$arm" "$vec" "$t" "$nlo" "$lo" "$nhi" "$hi" "$fmax" >> "$OUT"
    done
    echo "[$(date +%H:%M:%S)] r$r $vec t=$t done" >&2
  done
done
echo "SWEEP_DONE" >&2
