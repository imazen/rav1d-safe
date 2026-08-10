#!/usr/bin/env bash
# Paired base-vs-head A/B across the CONTENT-CLASS corpus.
#
# One instrument: `bench_ab_decode`'s in-process `Instant` over `iters` decodes
# of one still, reported as ms/frame. Process startup is outside the timer, so
# no two-point fit is needed here (that shape exists for the dav1d arm, which
# can only be timed as a whole process — see verify_gap.sh).
#
# Arms interleave with a ROTATING order inside every round, so a drift in box
# state hits both arms equally. Every row carries the count of foreign
# >25%-CPU processes; a round with any foreign load is committed but TAGGED, and
# the report must then quote paired ratios, never absolute ms.
#
# NO `nice` here. On Darwin `nice` maps to background QoS and lands the process
# on E-cores: ~40x wall distortion. Wrap the whole invocation in `measlock`
# instead, which is what serialises against other agents.
#
# Usage: content_ab.sh <out.tsv> [rounds] [iters]
# Env:   BIN, VEC, ARMS, CELLS  (CELLS entries are vec:threads[:iters])
# Columns: round arm vec threads iters ms_per_frame md5 foreign
set -u
export LC_ALL=C
OUT=${1:?out.tsv}; ROUNDS=${2:-9}; ITERS=${3:-100}
BIN=${BIN:-$HOME/tmp/ctxtl/bin}
VEC=${VEC:-$HOME/tmp/ctxtl/vec}
IFS=' ' read -r -a ARMS <<< "${ARMS:-base head}"
IFS=' ' read -r -a CELLS <<< "${CELLS:-\
Cui_1024x576_q20:1 Cui_1024x576_q20:8 Cui_1024x576_q70:1 Cui_1024x576_q70:8 \
Ctext_1024x576_q20:1 Ctext_1024x576_q20:8 Ctext_1024x576_q70:1 Ctext_1024x576_q70:8 \
Cphoto_1024x576_q20:1 Cphoto_1024x576_q20:8 Cphoto_1024x576_q70:1 Cphoto_1024x576_q70:8}"

BIN_RE=$(printf '%s' "$BIN" | sed 's/[][\.*^$/(){}?+|]/\\&/g')
busy() {
  ps -A -o %cpu,comm -r | awk -v me="$BIN_RE" \
    'NR>1 && $1>25 && $2 !~ /claude|ClaudeCode|versions\/|python3/ && $2 !~ me {c++} END {print c+0}'
}

printf 'round\tarm\tvec\tthreads\titers\tms_per_frame\tmd5\tforeign\n' > "$OUT"
n=${#ARMS[@]}
for r in $(seq 0 $((ROUNDS-1))); do
  for cell in "${CELLS[@]}"; do
    IFS=: read -r vec t it <<< "$cell"
    it=${it:-$ITERS}
    for k in $(seq 0 $((n-1))); do
      arm=${ARMS[$(( (k + r) % n ))]}
      raw=$("$BIN/bench_$arm" "$VEC/$vec.avif" "$t" "$it" 1 "$arm" 2>/dev/null)
      ms=$(printf '%s\n' "$raw" | awk -F'\t' '/^RESULT/{print $8}')
      md5=$(printf '%s\n' "$raw" | awk -F'\t' '/^CHECKSUM/{print $5}')
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$r" "$arm" "$vec" "$t" "$it" "$ms" "$md5" "$(busy)" >> "$OUT"
    done
  done
  echo "[$(date +%H:%M:%S)] round $r done" >&2
done
echo "wrote $OUT" >&2
