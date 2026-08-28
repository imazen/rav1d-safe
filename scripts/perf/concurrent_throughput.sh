#!/usr/bin/env bash
# N-concurrent single-threaded decoders: aggregate DECODES PER SECOND.
#
# Why this and not per-decode latency. For an image server the efficient model
# is many concurrent single-threaded decodes, not one multi-threaded decode:
# tile threading has a scaling deficit (4.93x at t=8 against dav1d's tile-only
# 6.84x) and costs tracker traffic that t=1 does not pay. Throughput under
# concurrency is the number that model is graded on, and nothing in ~15 rounds
# of this campaign has measured it.
#
# Same two-point trick as the size sweep: each N is run at n_lo and n_hi frames
# per process and the batch wall clock is differenced, so process startup and
# fork/exec storm drop out of the throughput figure.
#
# NO `nice`. Run under `measlock`.
#
# Usage: concurrent_throughput.sh <out.tsv> [rounds]
# Env:   BIN, AVIF, IVF, VEC, NPROCS, NLO, NHI, ARMS
# Output: round arm vec nproc nlo ms_lo nhi ms_hi foreign_max
set -u
export LC_ALL=C
OUT=${1:?out.tsv}; ROUNDS=${2:-7}
BIN=${BIN:-$HOME/tmp/szsweep/bin}
AVIF=${AVIF:-$HOME/tmp/szsweep/vec}
IVF=${IVF:-$HOME/tmp/szsweep/ivf}
VEC=${VEC:-L1024x576_420_8b}
NLO=${NLO:-10}; NHI=${NHI:-100}
IFS=' ' read -r -a NPROCS <<< "${NPROCS:-1 2 4 8 12 16}"
IFS=' ' read -r -a ARMS <<< "${ARMS:-rs dav1d_fd1}"

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
    sleep 5; w=$((w+5)); [ $w -ge 1800 ] && { echo "box never went idle" >&2; exit 4; }
  done
}
if [ -n "${EPOCHREALTIME:-}" ]; then
  now_ms() { local t=$EPOCHREALTIME; echo $(( ${t%%.*} * 1000 + 10#${t#*.} / 1000 )); }
else
  now_ms() { python3 -c 'import time;print(int(time.time()*1000))'; }
fi

batch() {  # arm nproc nframes -> elapsed ms for the WHOLE batch
  local arm=$1 np=$2 n=$3 t0 t1 pids=()
  t0=$(now_ms)
  for _ in $(seq 1 "$np"); do
    case "$arm" in
      dav1d_fd1) dav1d -i "$IVF/$VEC.ivf" --muxer null --threads 1 --framedelay 1 -q --limit "$n" >/dev/null 2>&1 & ;;
      *)         "$BIN/bench_$arm" "$AVIF/$VEC.avif" 1 "$n" 1 c >/dev/null 2>&1 & ;;
    esac
    pids+=($!)
  done
  wait "${pids[@]}" 2>/dev/null
  t1=$(now_ms); echo $((t1 - t0))
}

: > "$OUT"
na=${#ARMS[@]}
for round in $(seq 0 $((ROUNDS-1))); do
  for np in "${NPROCS[@]}"; do
    while :; do
      wait_quiet; rows=(); dirty=0; fmax=0
      for k in $(seq 0 $((na-1))); do
        arm=${ARMS[$(( (k + round) % na ))]}
        lo=$(batch "$arm" "$np" "$NLO")
        hi=$(batch "$arm" "$np" "$NHI")
        f=$(busy_count); [ "$f" -gt "$fmax" ] && fmax=$f
        [ "$f" -gt 0 ] && dirty=1
        rows+=("$round	$arm	$VEC	$np	$NLO	$lo	$NHI	$hi")
      done
      if [ $dirty -eq 0 ] || [ "$ALLOW_LOAD" = 1 ]; then
        for r in "${rows[@]}"; do printf '%s\t%s\n' "$r" "$fmax" >> "$OUT"; done
        echo "[$(date +%H:%M:%S)] r$round np=$np committed (foreign=$fmax)" >&2
        break
      fi
      echo "[$(date +%H:%M:%S)] r$round np=$np DISCARDED (foreign=$fmax)" >&2
    done
  done
done
echo "wrote $OUT" >&2
