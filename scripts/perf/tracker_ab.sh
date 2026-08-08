#!/usr/bin/env bash
# Rotating-order interleaved A/B over staged bench arms, one cell per
# (vector, threads). In-process ms/frame (bench_ab_decode's own Instant), so
# process startup never enters the number.
#
# NO `nice` on a timed run (Darwin maps it to background QoS -> E-cores, ~40x).
#
#   tracker_ab.sh <out.tsv> <rounds> <iters> <cells...> -- <arms...>
# cells are  vec:threads  (vec resolves under $VEC_DIR as <vec>.avif)
#
# Columns: round arm vec threads ms_per_frame foreign_busy md5
set -u
export LC_ALL=C
OUT=${1:?out.tsv}; ROUNDS=${2:-7}; ITERS=${3:-8}; shift 3
CELLS=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do CELLS+=("$1"); shift; done
shift || true
ARMS=("$@")
BIN=${BIN:-$HOME/tmp/bc/bin}
VEC_DIR=${VEC_DIR:-$HOME/tmp/rav1d-perf/vec}
BIN_RE=$(printf '%s' "$BIN" | sed 's/[][\.*^$/(){}?+|]/\\&/g')
busy() {
  ps -A -o %cpu,comm -r | awk -v me="$BIN_RE" \
    'NR>1 && $1>25 && $2 !~ /claude|ClaudeCode|versions\/|python3/ && $2 !~ me {c++} END {print c+0}'
}
: > "$OUT"
n=${#ARMS[@]}
for r in $(seq 0 $((ROUNDS-1))); do
  for cell in "${CELLS[@]}"; do
    IFS=: read -r vec t <<< "$cell"
    for k in $(seq 0 $((n-1))); do
      a=${ARMS[$(( (k + r) % n ))]}
      out=$("$BIN/bench_$a" "$VEC_DIR/$vec.avif" "$t" "$ITERS" 1 "$a" 2>/dev/null)
      ms=$(printf '%s\n' "$out" | awk -F'\t' '/^RESULT/{print $8}')
      md5=$(printf '%s\n' "$out" | awk -F'\t' '/^CHECKSUM/{print $5}')
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$r" "$a" "$vec" "$t" "$ms" "$(busy)" "$md5" >> "$OUT"
    done
    echo "[$(date +%H:%M:%S)] r$r $vec t=$t" >&2
  done
done
echo "wrote $OUT" >&2
