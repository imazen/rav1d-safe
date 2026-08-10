#!/usr/bin/env bash
# Wall AND CPU per frame on the FORCED-MULTI-TILE cells, ours vs dav1d.
#
# Two reasons this exists rather than quoting benchmarks/size_sweep_t8_*:
#
#  1. That record is LOAD-TAGGED (its own meta says a second agent's miri plus a
#     timed sweep were live throughout), so its absolutes are inflated. The
#     attribution needs an absolute ms/frame for the CPU that the task probe
#     canNOT see, so it needs an idle-box number.
#  2. The probe measures only IN-STAGE busy. Subtracting it from the process's
#     total CPU is how "where does the extra CPU go" gets an answer that
#     separates work inside a task from the pool around it -- and that
#     subtraction is only meaningful if both sides come from the same box state.
#
# `time` keyword with TIMEFORMAT='%3R %3U %3S' -> the child's getrusage. Two
# frame counts per cell, `total = a + b*frames` fitted separately for wall and
# for user+sys, so exec/mmap/decoder-construction/pool-spinup drop out of both.
#
# NO `nice` on a timed run. NO -C target-cpu=native. Run under `measlock`.
#
# Usage: tiled_wallcpu.sh <out.tsv> [rounds]
# Env:   BIN AVIF IVF ARMS CELLS
# Cell:  <vector>:<threads>:<n_lo>:<n_hi>
# Cols:  round arm vec threads nlo wall_lo user_lo sys_lo nhi wall_hi user_hi sys_hi foreign
set -u
export LC_ALL=C
OUT=${1:?out.tsv}; ROUNDS=${2:-7}
BIN=${BIN:-$HOME/tmp/tiledprof/bin}
AVIF=${AVIF:-$HOME/tmp/t8gap/vec}
IVF=${IVF:-$HOME/tmp/t8gap/ivf}
IFS=' ' read -r -a ARMS <<< "${ARMS:-rs dav1d_fd1 dav1d_def}"
# n_hi MUST NOT EXCEED THE IVF's FRAME COUNT. `bench_plain` re-decodes one OBU
# exactly `n` times whatever you ask for, but `dav1d --limit N` silently stops
# at end of stream -- so an n_hi past the end divides a SHORT total by a LONG
# frame delta and makes dav1d look faster by n_hi/n_frames. It read 94.1 ms
# instead of ~152 ms/frame on the 4K cell (16 frames, n_hi=24 => 1.615x) before
# the `verify_frames` gate below caught it. The counts here are the streams'
# real lengths (`~/tmp/szsweep/mkivf.sh`: 200 at 1024x576, 16 at 4K).
DEFAULT_CELLS=""
for t in 1 2 4 8; do
  DEFAULT_CELLS="$DEFAULT_CELLS L1024x576_420_8b__t8:$t:20:200"
  DEFAULT_CELLS="$DEFAULT_CELLS L3840x2160_420_8b__t8:$t:2:16"
done
IFS=' ' read -r -a CELLS <<< "${CELLS:-$DEFAULT_CELLS}"

# Fail loud rather than silently mis-fit: count the IVF's frames and refuse any
# cell whose n_hi is past the end.
ivf_frames() {
  python3 - "$1" <<'PY'
import struct, sys
d = open(sys.argv[1], 'rb').read()
off, n = 32, 0
while off + 12 <= len(d):
    off += 12 + struct.unpack_from('<I', d, off)[0]
    n += 1
print(n)
PY
}
for cell in "${CELLS[@]}"; do
  IFS=: read -r vec t nlo nhi <<< "$cell"
  have=$(ivf_frames "$IVF/$vec.ivf")
  if [ "$nhi" -gt "$have" ]; then
    echo "FATAL: $vec.ivf has $have frames but cell asks for n_hi=$nhi." >&2
    echo "       dav1d --limit would stop early and the two-point fit would" >&2
    echo "       divide a short total by a long frame delta." >&2
    exit 2
  fi
done

BIN_RE=$(printf '%s' "$BIN" | sed 's/[][\.*^$/(){}?+|]/\\&/g')
busy_count() {
  ps -A -o %cpu,comm -r | awk -v me="$BIN_RE" \
    'NR>1 && $1>25 && $2 !~ /claude|ClaudeCode|versions\/|dav1d|python3/ && $2 !~ me {c++} END {print c+0}'
}

TIMEFORMAT='%3R %3U %3S'
time_one() {
  local arm=$1 vec=$2 t=$3 n=$4 r
  case "$arm" in
    dav1d_fd1) r=$( { time dav1d -i "$IVF/$vec.ivf" --muxer null --threads "$t" --framedelay 1 -q --limit "$n" >/dev/null 2>/dev/null; } 2>&1 ) ;;
    dav1d_def) r=$( { time dav1d -i "$IVF/$vec.ivf" --muxer null --threads "$t"                 -q --limit "$n" >/dev/null 2>/dev/null; } 2>&1 ) ;;
    rs)        r=$( { time "$BIN/bench_plain" "$AVIF/$vec.avif" "$t" "$n" 1 w >/dev/null 2>/dev/null; } 2>&1 ) ;;
    *)         r=$( { time "$BIN/bench_$arm" "$AVIF/$vec.avif" "$t" "$n" 1 w >/dev/null 2>/dev/null; } 2>&1 ) ;;
  esac
  awk '{printf "%d %d %d\n", $1*1000+0.5, $2*1000+0.5, $3*1000+0.5}' <<< "$r"
}

: > "$OUT"
n=${#ARMS[@]}
for round in $(seq 0 $((ROUNDS-1))); do
  for cell in "${CELLS[@]}"; do
    IFS=: read -r vec t nlo nhi <<< "$cell"
    for k in $(seq 0 $((n-1))); do
      arm=${ARMS[$(( (k + round) % n ))]}
      f0=$(busy_count)
      read -r wlo ulo slo <<< "$(time_one "$arm" "$vec" "$t" "$nlo")"
      read -r whi uhi shi <<< "$(time_one "$arm" "$vec" "$t" "$nhi")"
      f1=$(busy_count)
      f=$(( f0 > f1 ? f0 : f1 ))
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$round" "$arm" "$vec" "$t" "$nlo" "$wlo" "$ulo" "$slo" \
        "$nhi" "$whi" "$uhi" "$shi" "$f" >> "$OUT"
    done
    echo "[$(date +%H:%M:%S)] r$round $vec t=$t done" >&2
  done
done
echo "wrote $OUT" >&2
