#!/usr/bin/env bash
# Size ladder x THREAD COUNT, measuring BOTH latency and CPU burned.
#
# scripts/perf/size_sweep.sh answered "how long does a frame take" at t=1 only.
# The product question needs two numbers per cell, not one: a 5x latency win
# that costs 4x the cores is a bad trade for a server. So every run here is
# timed for wall AND for user+sys CPU of the child, and both go through the same
# two-point fit `total = a + b*frames`, which removes process startup (exec,
# mmap, decoder construction, thread-pool spin-up) from BOTH.
#
# CPU comes from bash's `time` keyword (TIMEFORMAT %3R %3U %3S), i.e. the
# child's own getrusage as the kernel reports it -- not a sampler, not `ps`.
#
# NO `nice` ON A TIMED RUN (Darwin maps niced processes onto E-cores). Run the
# whole thing under `measlock`.
#
# Usage: size_sweep_t8.sh <out.tsv> [rounds]
# Env:   BIN AVIF IVF ARMS CELLS ALLOW_LOAD
# Cell:  <vector>:<threads>:<n_lo>:<n_hi>
# Cols:  round arm vec threads nlo wall_lo user_lo sys_lo nhi wall_hi user_hi sys_hi foreign_max
set -u
export LC_ALL=C
OUT=${1:?out.tsv}; ROUNDS=${2:-9}
BIN=${BIN:-$HOME/tmp/szsweep/bin}
AVIF=${AVIF:-$HOME/tmp/szsweep/vec}
IVF=${IVF:-$HOME/tmp/szsweep/ivf}
IFS=' ' read -r -a ARMS <<< "${ARMS:-rs dav1d_fd1 dav1d_def}"

# 420 only (the product case for AVIF stills), 6 sizes x 2 depths x t{1,2,4,8}.
DEFAULT_CELLS=""
for t in 1 2 4 8; do
  for d in 8b 10b; do
    DEFAULT_CELLS="$DEFAULT_CELLS L64x36_420_${d}:$t:5000:50000"
    DEFAULT_CELLS="$DEFAULT_CELLS L256x144_420_${d}:$t:500:5000"
    DEFAULT_CELLS="$DEFAULT_CELLS L512x288_420_${d}:$t:100:1000"
    DEFAULT_CELLS="$DEFAULT_CELLS L1024x576_420_${d}:$t:20:200"
    DEFAULT_CELLS="$DEFAULT_CELLS L2048x1152_420_${d}:$t:5:50"
    DEFAULT_CELLS="$DEFAULT_CELLS L3840x2160_420_${d}:$t:3:24"
  done
done
IFS=' ' read -r -a CELLS <<< "${CELLS:-$DEFAULT_CELLS}"

BIN_RE=$(printf '%s' "$BIN" | sed 's/[][\.*^$/(){}?+|]/\\&/g')
busy_count() {
  ps -A -o %cpu,comm -r | awk -v me="$BIN_RE" \
    'NR>1 && $1>25 && $2 !~ /claude|ClaudeCode|versions\/|dav1d|python3/ && $2 !~ me {c++} END {print c+0}'
}
ALLOW_LOAD=${ALLOW_LOAD:-0}
wait_quiet() {
  local w=0
  [ "$ALLOW_LOAD" = 1 ] && return 0
  while [ "$(busy_count)" -gt 0 ]; do
    sleep 5; w=$((w+5))
    [ $w -ge 1800 ] && { echo "box never went idle" >&2; return 0; }
  done
}

TIMEFORMAT='%3R %3U %3S'
# Echoes "wall_ms user_ms sys_ms".
time_one() {
  local arm=$1 vec=$2 t=$3 n=$4 r
  case "$arm" in
    dav1d_fd1) r=$( { time dav1d -i "$IVF/$vec.ivf" --muxer null --threads "$t" --framedelay 1 -q --limit "$n" >/dev/null 2>/dev/null; } 2>&1 ) ;;
    dav1d_def) r=$( { time dav1d -i "$IVF/$vec.ivf" --muxer null --threads "$t"                 -q --limit "$n" >/dev/null 2>/dev/null; } 2>&1 ) ;;
    *)         r=$( { time "$BIN/bench_$arm" "$AVIF/$vec.avif" "$t" "$n" 1 w >/dev/null 2>/dev/null; } 2>&1 ) ;;
  esac
  # "%3R %3U %3S" -> three floats in seconds; emit integer milliseconds.
  awk '{printf "%d %d %d\n", $1*1000+0.5, $2*1000+0.5, $3*1000+0.5}' <<< "$r"
}

: > "$OUT"
n=${#ARMS[@]}
for round in $(seq 0 $((ROUNDS-1))); do
  for cell in "${CELLS[@]}"; do
    IFS=: read -r vec t nlo nhi <<< "$cell"
    tries=0
    while :; do
      wait_quiet; rows=(); dirty=0; fmax=0
      for k in $(seq 0 $((n-1))); do
        arm=${ARMS[$(( (k + round) % n ))]}
        read -r wlo ulo slo <<< "$(time_one "$arm" "$vec" "$t" "$nlo")"
        read -r whi uhi shi <<< "$(time_one "$arm" "$vec" "$t" "$nhi")"
        f=$(busy_count); [ "$f" -gt "$fmax" ] && fmax=$f
        [ "$f" -gt 0 ] && dirty=1
        rows+=("$round	$arm	$vec	$t	$nlo	$wlo	$ulo	$slo	$nhi	$whi	$uhi	$shi")
      done
      tries=$((tries+1))
      if [ $dirty -eq 0 ] || [ "$ALLOW_LOAD" = 1 ] || [ $tries -ge 3 ]; then
        for r in "${rows[@]}"; do printf '%s\t%s\n' "$r" "$fmax" >> "$OUT"; done
        echo "[$(date +%H:%M:%S)] r$round $vec t=$t committed (foreign=$fmax tries=$tries)" >&2
        break
      fi
      echo "[$(date +%H:%M:%S)] r$round $vec t=$t DISCARDED (foreign=$fmax)" >&2
    done
  done
done
echo "wrote $OUT" >&2
