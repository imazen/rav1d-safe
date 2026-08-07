#!/usr/bin/env bash
# Interleaved rav1d-safe vs dav1d (C reference) decode-throughput sweep.
#
# Three arms per (vector, threads) cell, rotated each round so no arm
# systematically inherits the others' thermal wake:
#   rs        rav1d-safe (this crate, main), managed API, in-process iters loop
#   dav1d1    dav1d 1.5.4 --framedelay 1  (tile threading only == rav1d-safe's model)
#   dav1dA    dav1d 1.5.4 default framedelay (frame threading allowed)
#
# NO `nice` on any timed run (Darwin maps positive nice to background QoS).
# dav1d per-frame times come from --frametimes, so process startup and file IO
# are excluded; the first 3 frames of each run are dropped as warmup.
#
# CONTENTION HANDLING. This box is shared with a sibling agent whose cargo
# builds spike to 3x100% CPU without warning. A pre-cell quiet check is NOT
# enough — a build that starts mid-cell silently invalidates it. So the whole
# cell (all three arms, back to back) is run into a staging file and is only
# committed if the box was quiet before the cell AND after every arm in it.
# Otherwise the cell is discarded and retried. Cells are never split across a
# pause: that would break the back-to-back interleaving the A/B rests on.
set -u

OUT=${1:?out.tsv}
ROUNDS=${2:-3}

BENCH=$HOME/tmp/recon-yard/target-main/release/examples/bench_ab_decode
AVIF=$HOME/tmp/rav1d-perf/vec
IVF=$HOME/tmp/recon-yard/vec
FT=$HOME/tmp/recon-yard/ft.tmp
STAGE=$HOME/tmp/recon-yard/cell.stage
MAX_ATTEMPTS=40

others=$(pgrep -f '[y]ardstick_sweep.sh' | grep -v "^$$\$" | wc -l | tr -d ' ')
if [ "$others" -gt 1 ]; then echo "another sweep running; refusing" >&2; exit 3; fi

# Count processes above 25% CPU, ignoring the agent runtime itself (it is
# unavoidable, single-digit-to-~30% and idles down while a cell runs).
busy_count() {
  ps -A -o %cpu,comm -r \
    | awk 'NR>1 && $1>25 && $2 !~ /claude|ClaudeCode|versions\// {c++} END {print c+0}'
}

wait_for_quiet() {
  local waited=0
  while [ "$(busy_count)" -gt 0 ]; do
    [ $waited -eq 0 ] && echo "[$(date +%H:%M:%S)] box busy, pausing..." >&2
    sleep 10; waited=$((waited + 10))
    if [ $waited -ge 5400 ]; then echo "box busy ${waited}s; refusing" >&2; exit 4; fi
  done
  [ $waited -gt 0 ] && echo "[$(date +%H:%M:%S)] quiet after ${waited}s, resuming" >&2
  return 0
}

# vector : rav1d-safe iters : dav1d frame limit
CELLS=(
  "v256:300:500"
  "v1024:20:60"
  "v1024_10b:20:60"
  "v4k_1tile:2:12"
  "v4k_1tile_10b:2:12"
  "v4k_8tile:2:12"
  "v4k_8tile_10b:2:12"
)
THREADS=(1 2 4 8)
REPS=3
ARMS=(rs dav1d1 dav1dA)

run_dav1d() {  # vec threads limit framedelay label round
  local vec=$1 t=$2 lim=$3 fd=$4 label=$5 round=$6
  rm -f "$FT"
  local extra=()
  [ "$fd" != "auto" ] && extra=(--framedelay "$fd")
  dav1d -i "$IVF/$vec.ivf" --muxer null --threads "$t" "${extra[@]}" \
        -q --limit "$lim" --frametimes "$FT" >/dev/null 2>&1
  local rc=$?
  if [ $rc -ne 0 ] || [ ! -s "$FT" ]; then
    printf '%s\tFAIL\t%s\t%s\t%s\trc=%s\n' "$round" "$label" "$vec" "$t" "$rc" >> "$STAGE"
    return
  fi
  awk -v r="$round" -v l="$label" -v v="$vec" -v t="$t" \
      'NR>3 {printf "%s\tRESULT\t%s\t%s\t%s\t%d\t1\t%.6f\t%.6f\n", r,l,v,t,NR-4,$1/1e6,$1/1e6}' \
      "$FT" >> "$STAGE"
}

: > "$OUT"
for round in $(seq 0 $((ROUNDS - 1))); do
  for cell in "${CELLS[@]}"; do
    IFS=: read -r vec iters lim <<< "$cell"
    for t in "${THREADS[@]}"; do
      attempt=0
      while : ; do
        attempt=$((attempt + 1))
        if [ $attempt -gt $MAX_ATTEMPTS ]; then
          echo "GIVING UP on $vec t=$t after $MAX_ATTEMPTS contaminated attempts" >&2
          printf '%s\tGIVEUP\t-\t%s\t%s\t%s\n' "$round" "$vec" "$t" "$MAX_ATTEMPTS" >> "$OUT"
          break
        fi
        wait_for_quiet
        : > "$STAGE"
        dirty=0
        n=${#ARMS[@]}
        for k in $(seq 0 $((n - 1))); do
          arm=${ARMS[$(( (k + round) % n ))]}
          case "$arm" in
            rs)
              out=$("$BENCH" "$AVIF/$vec.avif" "$t" "$iters" "$REPS" rs 2>&1); rc=$?
              if [ $rc -ne 0 ]; then
                printf '%s\tFAIL\trs\t%s\t%s\trc=%s\n' "$round" "$vec" "$t" "$rc" >> "$STAGE"
              fi
              printf '%s\n' "$out" | grep -E '^(RESULT|CHECKSUM|GEOM)' \
                | sed "s/^/$round\t/" >> "$STAGE"
              ;;
            dav1d1) run_dav1d "$vec" "$t" "$lim" 1 dav1d1 "$round" ;;
            dav1dA) run_dav1d "$vec" "$t" "$lim" auto dav1dA "$round" ;;
          esac
          [ "$(busy_count)" -gt 0 ] && dirty=1
        done
        load=$(sysctl -n vm.loadavg | awk '{print $2}')
        if [ $dirty -eq 0 ]; then
          cat "$STAGE" >> "$OUT"
          printf '%s\tLOAD\t-\t%s\t%s\t%s\n' "$round" "$vec" "$t" "$load" >> "$OUT"
          printf '[%s] round=%s %-16s t=%s  committed (attempt %s, load %s)\n' \
            "$(date +%H:%M:%S)" "$round" "$vec" "$t" "$attempt" "$load" >&2
          break
        fi
        printf '[%s] round=%s %-16s t=%s  DISCARDED (contended, attempt %s)\n' \
          "$(date +%H:%M:%S)" "$round" "$vec" "$t" "$attempt" >&2
      done
    done
  done
done
rm -f "$STAGE"
echo "wrote $OUT" >&2
