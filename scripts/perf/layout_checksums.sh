#!/usr/bin/env bash
# Every arm must decode every timed cell to the SAME md5, at both thread counts,
# BEFORE anything is timed. Niced (not a timed run).
# Usage: layout_checksums.sh <out.tsv> <arm...>
set -eu
OUT=${1:?out.tsv}; shift
BIN=${BIN:-$HOME/tmp/layout/bin}
AVIF=${AVIF:-$HOME/tmp/lfg/stage/avif}
CELLS=${CELLS:-"v4k8tile:1 v4k8tile:8 c1024x576:1 c1024x576:8 c256x2048:1 c256x2048:8"}
: > "$OUT"
for cell in $CELLS; do
  IFS=: read -r vec t <<< "$cell"
  for arm in "$@"; do
    md5=$(nice -n 19 "$BIN/bench_$arm" "$AVIF/$vec.avif" "$t" 2 1 c 2>/dev/null \
          | awk '$1=="CHECKSUM"{print $NF}')
    printf '%s\t%s\t%s\t%s\n' "$vec" "$t" "$arm" "$md5" >> "$OUT"
  done
done
awk -F'\t' '{k=$1":"$2; if (!(k in m)) m[k]=$4; if ($4!=m[k] || $4=="") bad[k]=bad[k]" "$3"="$4}
END{n=0; for (k in m) {printf "%-20s %s", k, m[k]; if (k in bad) {printf "   MISMATCH:%s", bad[k]; n++}; printf "\n"}
     printf "cells_with_mismatch=%d\n", n; exit (n>0)}' "$OUT"
