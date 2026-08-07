#!/usr/bin/env bash
# Per-kernel CDEF oracle sweep: decode every vector with a `__simd_test_log`
# build (which runs the NEON kernel AND the scalar reference on every block and
# diffs them) and count CDEF_MISMATCH / CDEF_DIR_MISMATCH.
#
#   cdef_oracle_sweep.sh <decode_md5_binary> <corpus_root> <out.log>
#
# This is a CORRECTNESS sweep, not a timed one, so `nice` is fine and wanted.
set -u
BIN=${1:?decode_md5 binary}
ROOT=${2:?corpus root}
OUT=${3:?out.log}
: > "$OUT"
n=0
find "$ROOT" \( -name '*.ivf' -o -name '*.obu' \) | sort | while read -r f; do
  n=$((n + 1))
  echo "### $f" >> "$OUT"
  nice -n 19 "$BIN" -q "$f" 2>&1 | grep -E 'MISMATCH' >> "$OUT"
  if [ $((n % 50)) -eq 0 ]; then echo "[$(date +%H:%M:%S)] $n vectors" >&2; fi
done
echo "done" >&2
