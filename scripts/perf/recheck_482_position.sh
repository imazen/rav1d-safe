#!/usr/bin/env bash
# Is EXECUTION POSITION inside a burst worth anything, with arm identity held
# constant?
#
# The paired sweep showed the first arm of a (round, cell) group running slow
# at 4K 4:2:0 8bpc t=1. That is confounded: position 0 is also whichever arm
# the rotation happened to put there. This removes the confound by running ONE
# binary N times back-to-back and reporting ms/frame by repetition index. Any
# slope is position, not code.
#
# It also settles the mechanism partly by itself: the in-process timer starts
# AFTER the file read, the container parse, the decoder construction and a
# warmup decode, so a first-run penalty visible in the `int` column cannot be
# file I/O or process setup.
#
# Usage: recheck_482_position.sh <out.tsv> [reps] [arm] [vector] [threads]
set -u
export LC_ALL=C
OUT=${1:?out.tsv}; REPS=${2:-12}; ARM=${3:-parent}
VEC=${4:-L3840x2160_420_8b}; THR=${5:-1}; N=${6:-16}
BIN=${BIN:-$HOME/tmp/recheck482/bin}
AVIF=${AVIF:-$HOME/tmp/recheck482/vec}

if [ -n "${EPOCHREALTIME:-}" ]; then
  now_ms() { local t=$EPOCHREALTIME; echo $(( ${t%%.*} * 1000 + 10#${t#*.} / 1000 )); }
else
  now_ms() { python3 -c 'import time;print(int(time.time()*1000))'; }
fi

printf 'rep\text_ms\tint_ms_per_frame\n' > "$OUT"
for i in $(seq 0 $((REPS-1))); do
  t0=$(now_ms)
  out=$(RAV1D_OWNED_RECON=1 "$BIN/bench_$ARM" "$AVIF/$VEC.avif" "$THR" "$N" 1 w 2>/dev/null)
  t1=$(now_ms)
  ipf=$(printf '%s\n' "$out" | awk -F'\t' '/^RESULT/{print $8}')
  printf '%s\t%s\t%s\n' "$i" "$((t1 - t0))" "${ipf:-NA}" >> "$OUT"
done
column -t -s $'\t' "$OUT" >&2
awk -F'\t' 'NR>2{s+=$3; n++} END{if(n) printf "\nrep0 vs mean(rep1..): %.3f vs %.3f  ->  %+.2f%%\n", f, s/n, (f/(s/n)-1)*100}' \
  f="$(awk -F'\t' 'NR==2{print $3}' "$OUT")" "$OUT" >&2
