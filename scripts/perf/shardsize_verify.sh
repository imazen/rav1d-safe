#!/usr/bin/env bash
# Bit-identity gate for the size-sweep vectors, BEFORE any timing.
#
# A fast wrong decode is not a result. Every cell is decoded by rav1d-safe at
# t=1 AND at t=8 and by dav1d, and all three md5s must agree — t=8 included,
# because the whole point of this grid is multi-tile behaviour at eight threads
# and a tile-threading defect is exactly the kind a t=1-only gate cannot see.
#
# Verification runs on a ONE-FRAME IVF per cell (built here), not on the long
# timing stream: the timing streams repeat one OBU hundreds of times, so their
# md5 adds nothing but minutes.
#
# Counting run, not a timed one: NICED, no measurement lock.
#
# Usage: shardsize_verify.sh <out.tsv>
# Env:   VEC (dir of .avif), MD5BIN, TOIVF, THREADS ("1 8")
set -u
export LC_ALL=C
OUT=${1:?out.tsv}
VEC=${VEC:-$HOME/tmp/shardsize/vec}
ONE=${ONE:-$HOME/tmp/shardsize/one}
MD5BIN=${MD5BIN:-target/release/examples/decode_md5}
TOIVF=${TOIVF:-target/release/examples/avif_to_ivf}
IFS=' ' read -r -a THREADS <<< "${THREADS:-1 8}"
for b in "$MD5BIN" "$TOIVF"; do
  [ -x "$b" ] || { echo "missing $b — cargo build --release --example ..." >&2; exit 1; }
done
mkdir -p "$ONE"

{
  printf 'vector\tdav1d_md5\t'
  for t in "${THREADS[@]}"; do printf 'ours_t%s\t' "$t"; done
  printf 'verdict\n'
} > "$OUT"

bad=0
for f in "$VEC"/*.avif; do
  v=$(basename "$f" .avif)
  one="$ONE/$v.ivf"
  [ -s "$one" ] || nice -n 19 "$TOIVF" "$f" 1 "$one" >/dev/null
  # `-o -` is REQUIRED: without it dav1d exits with "Output file is required"
  # and an md5-scraping pipeline reads an empty string, which compares unequal
  # and looks like a decode mismatch instead of a missing reference.
  d=$(nice -n 19 dav1d -i "$one" --muxer md5 -o - -q 2>/dev/null | grep -oE '[0-9a-f]{32}' | head -1)
  row=""; verdict=OK
  for t in "${THREADS[@]}"; do
    o=$(nice -n 19 "$MD5BIN" -q --threads "$t" "$one" 2>/dev/null \
        | grep -oE '[0-9a-f]{32}' | head -1)
    row="$row$o\t"
    { [ -n "$o" ] && [ "$o" = "$d" ]; } || verdict=MISMATCH
  done
  [ -n "$d" ] || verdict=NO_DAV1D
  [ "$verdict" = OK ] || bad=$((bad+1))
  printf "%s\t%s\t${row}%s\n" "$v" "$d" "$verdict" >> "$OUT"
done
echo "mismatches: $bad" >&2
column -t "$OUT" >&2
[ "$bad" -eq 0 ] || exit 1
