#!/usr/bin/env bash
# Disassemble ONE symbol out of two Mach-O binaries and diff the INSTRUCTIONS,
# with addresses and address-relative operands normalised away.
#
# This is the question a wall-clock A/B cannot answer: when an arm is slower on
# a path where its new code provably cannot execute, is the executed code's
# CODEGEN different (a real change) or only its PLACEMENT (a layout effect)?
# A byte-identical instruction stream says placement.
#
# Usage: text_symbol_diff.sh <base-bin> <head-bin> <symbol-regex> [outdir]
set -euo pipefail
export LC_ALL=C
BASE=${1:?base binary}; HEAD=${2:?head binary}; RE=${3:?symbol regex}
OUT=${4:-$HOME/tmp/rectship/asm}
mkdir -p "$OUT"

# The crate disambiguator (`Cs<base62>_`) differs between two builds with
# different feature sets, so match on the readable part of the mangled name.
dump() {
  local bin=$1 tag=$2
  objdump -d --no-show-raw-insn "$bin" \
    | awk -v re="$RE" '
        /^[0-9a-f]+ <.*>:$/ { name=$0; on = (name ~ re); if (on) print "\n" name; next }
        on { print }
      ' \
    | sed -E 's/^[[:space:]]*[0-9a-f]+:[[:space:]]*//' \
    | sed -E 's/0x[0-9a-f]+ <[^>]*>/<ADDR>/g; s/#[[:space:]]*0x[0-9a-f]+$/# <IMM>/' \
    | sed -E 's/Cs[0-9A-Za-z]{10,}_/Cs_/g' \
    > "$OUT/$tag.asm"
  wc -l < "$OUT/$tag.asm"
}

b=$(dump "$BASE" base)
h=$(dump "$HEAD" head)
echo "base_insns=$b head_insns=$h"
if diff -q "$OUT/base.asm" "$OUT/head.asm" >/dev/null; then
  echo "IDENTICAL instruction stream"
else
  echo "DIFFERS:"
  diff "$OUT/base.asm" "$OUT/head.asm" | head -80
fi
