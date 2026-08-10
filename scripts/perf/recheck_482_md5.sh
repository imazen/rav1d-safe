#!/usr/bin/env bash
# Correctness gate for the #482 re-measurement. An arm that decodes differently
# is not an arm, so this runs BEFORE anything is timed.
#
# Set-diff BY NAME with the hash in the key: every (vector, threads) key must
# carry the same md5 on all three arms. A count of matches would hide a change
# that repairs one vector and breaks another.
#
# The md5 is over visible pixels row-by-row (stride padding excluded) — the
# CHECKSUM line bench_ab_decode already emits.
set -u
export LC_ALL=C
OUT=${1:?out.tsv}
BIN=${BIN:-$HOME/tmp/recheck482/bin}
AVIF=${AVIF:-$HOME/tmp/recheck482/vec}
IFS=' ' read -r -a ARMS <<< "${ARMS:-parent head main}"
IFS=' ' read -r -a VECS <<< "${VECS:-L3840x2160_420_8b L3840x2160_420_10b L3840x2160_444_8b L3840x2160_444_10b v4k_8tile v4k_8tile_10b}"
IFS=' ' read -r -a THREADS <<< "${THREADS:-1 8}"

{
  printf 'vector\tthreads\tgeom'
  for a in "${ARMS[@]}"; do printf '\t%s' "$a"; done
  printf '\tverdict\n'
} > "$OUT"

fail=0
for v in "${VECS[@]}"; do
  for t in "${THREADS[@]}"; do
    hashes=(); first=""; geom=""; ok=MATCH
    for a in "${ARMS[@]}"; do
      out=$(nice -n 19 "$BIN/bench_$a" "$AVIF/$v.avif" "$t" 1 1 "$a" 2>/dev/null)
      h=$(printf '%s\n' "$out" | awk -F'\t' '/^CHECKSUM/{print $5}')
      g=$(printf '%s\n' "$out" | awk -F'\t' '/^GEOM/{print $5"/"$6}')
      [ -z "$h" ] && { h=NO_OUTPUT; ok=BROKEN; }
      [ -z "$geom" ] && geom=$g
      [ -z "$first" ] && first=$h
      [ "$h" != "$first" ] && ok=DIFFER
      hashes+=("$h")
    done
    { printf '%s\t%s\t%s' "$v" "$t" "$geom"
      for h in "${hashes[@]}"; do printf '\t%s' "$h"; done
      printf '\t%s\n' "$ok"; } >> "$OUT"
    [ "$ok" = MATCH ] || fail=1
  done
done
column -t -s $'\t' "$OUT" >&2
if [ "$fail" -ne 0 ]; then echo "MD5 GATE FAILED" >&2; exit 1; fi
echo "MD5 GATE PASSED: all arms byte-identical on every (vector, threads) key" >&2
