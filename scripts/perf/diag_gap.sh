#!/usr/bin/env bash
# Gap-to-dav1d with ONE instrument on both sides: wall clock of the whole
# process at two frame counts, `total = alpha + beta*frames` fitted so process
# startup drops out and beta is ms/frame. Same shape as p2_dav1d_wall.sh.
#
# dav1d runs at --framedelay 1: tile-threading only, which is the model we
# implement. Its default enables frame threading and is not comparable.
#
# NO `nice` on a timed run. Arms interleave back to back with a rotating order;
# a round whose cell saw a foreign process over 25% CPU is discarded and re-run.
#
# Usage: diag_gap.sh <out.tsv> [rounds]
set -u
export LC_ALL=C
OUT=${1:?out.tsv}; ROUNDS=${2:-5}
BIN=${BIN:-$HOME/tmp/rav1d-diag/bin}
AVIF=${AVIF:-$HOME/tmp/rav1d-perf/vec}
IVF=${IVF:-$HOME/tmp/recon-yard/vec}
NLO=${NLO:-2}; NHI=${NHI:-20}
IFS=' ' read -r -a ARMS <<< "${ARMS:-base ttu dav1d_fd1}"
IFS=' ' read -r -a CELLS <<< "${CELLS:-v4k_8tile:1 v4k_8tile:2 v4k_8tile:4 v4k_8tile:8 v4k_8tile_10b:1 v4k_8tile_10b:8}"

BIN_RE=$(printf '%s' "$BIN" | sed 's/[][\.*^$/(){}?+|]/\\&/g')
busy_count() {
  ps -A -o %cpu,comm -r | awk -v me="$BIN_RE" \
    'NR>1 && $1>25 && $2 !~ /claude|ClaudeCode|versions\/|dav1d|python3/ && $2 !~ me {c++} END {print c+0}'
}
wait_quiet() { local w=0; while [ "$(busy_count)" -gt 0 ]; do sleep 5; w=$((w+5)); [ $w -ge 900 ] && exit 4; done; }
now_ms() { python3 -c 'import time;print(int(time.time()*1000))'; }
time_one() {
  local arm=$1 vec=$2 t=$3 n=$4 t0 t1
  t0=$(now_ms)
  case "$arm" in
    dav1d_fd1) dav1d -i "$IVF/$vec.ivf" --muxer null --threads "$t" --framedelay 1 -q --limit "$n" >/dev/null 2>&1 ;;
    *)         "$BIN/bench_$arm" "$AVIF/$vec.avif" "$t" "$n" 1 w >/dev/null 2>&1 ;;
  esac
  t1=$(now_ms); echo $((t1 - t0))
}
printf 'round\tarm\tvec\tthreads\tnlo\tms_lo\tnhi\tms_hi\n' > "$OUT"
n=${#ARMS[@]}
for round in $(seq 0 $((ROUNDS-1))); do
  for cell in "${CELLS[@]}"; do
    IFS=: read -r vec t <<< "$cell"
    while :; do
      wait_quiet; stage=""; dirty=0
      for k in $(seq 0 $((n-1))); do
        arm=${ARMS[$(( (k + round) % n ))]}
        lo=$(time_one "$arm" "$vec" "$t" "$NLO")
        hi=$(time_one "$arm" "$vec" "$t" "$NHI")
        stage="${stage}${round}\t${arm}\t${vec}\t${t}\t${NLO}\t${lo}\t${NHI}\t${hi}\n"
        [ "$(busy_count)" -gt 0 ] && dirty=1
      done
      if [ $dirty -eq 0 ]; then
        printf "$stage" >> "$OUT"; echo "[$(date +%H:%M:%S)] $vec t=$t r$round ok" >&2; break
      fi
      echo "[$(date +%H:%M:%S)] $vec t=$t r$round DISCARDED" >&2
    done
  done
done
echo "wrote $OUT" >&2
