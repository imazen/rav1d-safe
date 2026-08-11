#!/usr/bin/env bash
# CPU-time twin of ctxread_wall.sh, adapted from scripts/perf/census_cpu.sh.
# Original header: CPU-time twin of gap_sweep.sh. Same two-point fit, but on user+sys instead of
# wall, because the ns-per-registration model is a CPU quantity: at t=8 a wall
# ms buys ~5-7 CPU ms and comparing a wall delta to the campaign's CPU-derived
# 4.5-6.4 ns band would under-read it by that factor.
set -u
export LC_ALL=C
OUT=${1:?out.tsv}; ROUNDS=${2:-5}
BIN=$HOME/tmp/lfg/bin
AVIF=$HOME/tmp/lfg/stage/avif
IVF=$HOME/tmp/lfg/stage/ivf
IFS=' ' read -r -a ARMS <<< "${ARMS:-base head dav1d_fd1}"
IFS=' ' read -r -a CELLS <<< "${CELLS:-ui_q20:8:20:200 ui_q20:1:20:200 text_q20:8:20:200 text_q20:1:20:200 c256x2048:8:20:200 c1024x384:8:20:200 c1024x192:8:20:200 c3840x256:8:12:120 c1024x576:8:20:200 v4k8tile:8:4:40 v4k8tile:1:4:40}"

BIN_RE=$(printf '%s' "$BIN" | sed 's/[][\.*^$/(){}?+|]/\\&/g')
busy_count() {
  ps -A -o %cpu,comm -r | awk -v me="$BIN_RE" \
    'NR>1 && $1>25 && $2 !~ /claude|ClaudeCode|versions\/|dav1d|python3/ && $2 !~ me {c++} END {print c+0}'
}
# bash's `time` KEYWORD (not /usr/bin/time): it writes to the SHELL's stderr, so
# the command's own stderr can be discarded independently. /usr/bin/time shares
# its fd 2 with the payload and macOS has no `-o`, which is why the first
# attempt read 0.
TIMEFORMAT='CPU %3U %3S'
cpu_ms() { # arm vec threads n  -> (user+sys) in ms
  local arm=$1 vec=$2 t=$3 n=$4 e
  case "$arm" in
    dav1d_fd1) e=$( { time dav1d -i "$IVF/$vec.ivf" --muxer null --threads "$t" --framedelay 1 -q --limit "$n" >/dev/null 2>&1; } 2>&1 ) ;;
    *)         e=$( { time "$BIN/bench_$arm" "$AVIF/$vec.avif" "$t" "$n" 1 c >/dev/null 2>&1; } 2>&1 ) ;;
  esac
  printf '%s\n' "$e" | awk '/^CPU/{printf "%d\n",($2+$3)*1000}'
}

printf 'round\tarm\tvec\tthreads\tnlo\tcpu_lo\tnhi\tcpu_hi\tforeign_max\n' > "$OUT"
for r in $(seq 1 "$ROUNDS"); do
  for cell in "${CELLS[@]}"; do
    IFS=: read -r vec t nlo nhi <<< "$cell"
    na=${#ARMS[@]}; off=$(( (r - 1) % na ))
    for i in $(seq 0 $((na - 1))); do
      arm=${ARMS[$(( (i + off) % na ))]}
      fmax=$(busy_count)
      lo=$(cpu_ms "$arm" "$vec" "$t" "$nlo")
      hi=$(cpu_ms "$arm" "$vec" "$t" "$nhi")
      f=$(busy_count); [ "$f" -gt "$fmax" ] && fmax=$f
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$r" "$arm" "$vec" "$t" "$nlo" "$lo" "$nhi" "$hi" "$fmax" >> "$OUT"
    done
    echo "[$(date +%H:%M:%S)] r$r $vec t=$t cpu done" >&2
  done
done
echo "CPU_DONE" >&2
