#!/usr/bin/env bash
# How much of the rav1d-safe -> dav1d gap, and how much of the tile-threading
# anti-scaling, is the DisjointMut safety machinery?
#
#   chk     rav1d-safe, default features (checked DisjointMut, forbid unsafe)
#   unchk   rav1d-safe, --features unchecked (tracker off for the hot-path
#           instances via dangerously_unchecked(), plus get_unchecked slice
#           access — an UPPER bound on the safety cost, not tracker-only)
#   dav1d1  dav1d 1.5.4 --framedelay 1, the C reference
#
# Both rav1d-safe arms pin max_frame_delay = 1 so `unchecked` does not silently
# gain frame threading (n_fc is forced to 1 only in the checked build).
set -u
OUT=${1:?out.tsv}
ROUNDS=${2:-3}
BIN=$HOME/tmp/recon-yard/bin
AVIF=$HOME/tmp/rav1d-perf/vec
IVF=$HOME/tmp/recon-yard/vec
FT=$HOME/tmp/recon-yard/ft_unchk.tmp

busy_count() {
  ps -A -o %cpu,comm -r \
    | awk 'NR>1 && $1>25 && $2 !~ /claude|ClaudeCode|versions\// {c++} END {print c+0}'
}
wait_for_quiet() {
  local w=0
  while [ "$(busy_count)" -gt 0 ]; do
    sleep 10; w=$((w+10)); [ $w -ge 5400 ] && { echo "box busy; refusing" >&2; exit 4; }
  done
}

CELLS=("v1024:20:60" "v4k_1tile:2:12" "v4k_8tile:2:12" "v4k_8tile_10b:2:12")
THREADS=(1 2 4 8)
REPS=3
ARMS=(chk unchk dav1d1)
STAGE=$HOME/tmp/recon-yard/unchk.stage

: > "$OUT"
for round in $(seq 0 $((ROUNDS-1))); do
  for cell in "${CELLS[@]}"; do
    IFS=: read -r vec iters lim <<< "$cell"
    for t in "${THREADS[@]}"; do
      while : ; do
        wait_for_quiet
        : > "$STAGE"; dirty=0
        n=${#ARMS[@]}
        for k in $(seq 0 $((n-1))); do
          arm=${ARMS[$(( (k + round) % n ))]}
          case "$arm" in
            chk|unchk)
              out=$("$BIN/$arm" "$AVIF/$vec.avif" "$t" "$iters" "$REPS" "$arm" 2>&1); rc=$?
              [ $rc -ne 0 ] && printf '%s\tFAIL\t%s\t%s\t%s\trc=%s\n' \
                 "$round" "$arm" "$vec" "$t" "$rc" >> "$STAGE"
              printf '%s\n' "$out" | grep -E '^(RESULT|CHECKSUM)' \
                | sed "s/^/$round\t/" >> "$STAGE"
              ;;
            dav1d1)
              rm -f "$FT"
              dav1d -i "$IVF/$vec.ivf" --muxer null --threads "$t" --framedelay 1 \
                    -q --limit "$lim" --frametimes "$FT" >/dev/null 2>&1
              awk -v r="$round" -v v="$vec" -v t="$t" \
                'NR>3 {printf "%s\tRESULT\tdav1d1\t%s\t%s\t%d\t1\t%.6f\t%.6f\n", r,v,t,NR-4,$1/1e6,$1/1e6}' \
                "$FT" >> "$STAGE"
              ;;
          esac
          [ "$(busy_count)" -gt 0 ] && dirty=1
        done
        if [ $dirty -eq 0 ]; then
          cat "$STAGE" >> "$OUT"
          echo "[$(date +%H:%M:%S)] round=$round $vec t=$t committed" >&2
          break
        fi
        echo "[$(date +%H:%M:%S)] round=$round $vec t=$t DISCARDED (contended)" >&2
      done
    done
  done
done
rm -f "$STAGE"
echo "wrote $OUT" >&2
