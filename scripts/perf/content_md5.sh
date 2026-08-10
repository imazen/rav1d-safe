#!/usr/bin/env bash
# Prove every content-class vector decodes bit-identically to dav1d BEFORE any
# timing is taken on it. A fast wrong decode is not a result.
#
# Emits a BY-NAME table (vector \t dav1d_md5 \t ours_md5 \t verdict) so a later
# run is a set-diff, never a count comparison — a change that repairs one vector
# and breaks another is invisible in "12/12 OK" and obvious in a joined table.
#
# Usage: content_md5.sh [ivfdir] [out.tsv]
set -euo pipefail
IVF="${1:-$HOME/tmp/ctxtl/ivf}"
OUT="${2:-$HOME/tmp/ctxtl/content_md5.tsv}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="${DECODE_MD5:-$REPO/target/release/examples/decode_md5}"
[ -x "$BIN" ] || { echo "build it: cargo build --release --example decode_md5" >&2; exit 3; }

printf 'vector\tdav1d_md5\tours_md5\tverdict\n' > "$OUT"
fail=0
for f in "$IVF"/*.ivf; do
  v=$(basename "$f" .ivf)
  # Both sides hash EVERY frame in the file (decode_md5 has no frame limit), so
  # the comparison covers the whole IVF, not just frame 1. dav1d's md5 muxer
  # requires an explicit `-o`.
  o=$(nice -n 19 "$BIN" -q "$f" 2>/dev/null | tr -d ' \n')
  d_all=$(dav1d -i "$f" --muxer md5 -o - -q 2>/dev/null | tr -d ' \n')
  if [ "$o" = "$d_all" ]; then verdict=OK; else verdict=MISMATCH; fail=$((fail+1)); fi
  printf '%s\t%s\t%s\t%s\n' "$v" "$d_all" "$o" "$verdict" >> "$OUT"
  printf '%-28s %s %s\n' "$v" "$verdict" "$d_all"
done
echo "wrote $OUT"
[ "$fail" -eq 0 ] || { echo "$fail MISMATCH — do not time this corpus" >&2; exit 1; }
echo "all vectors bit-identical to dav1d $(dav1d --version 2>&1 | head -1)"
