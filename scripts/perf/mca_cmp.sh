#!/usr/bin/env bash
# llvm-mca throughput comparison of one function each from two binaries.
# dav1d nasm fns span .sublabel symbols, so we disassemble the address range
# [sym_start, next non-dotted global symbol).
# Usage: mca_cmp.sh <our_bin> <our_sym> <asm_bin> <dav1d_sym> [iterations]
set -u
OUR=$1; OSYM=$2; ASM=$3; DSYM=$4; ITER=${5:-300}
range_of() { # binary, symbol -> "start stop"
  nm -n "$1" | awk -v s="$2" '
    $3==s && $2 ~ /[tT]/ { st=strtonum("0x"$1); grab=1; next }
    grab && $2 ~ /[tT]/ && $3 !~ /\./ && $3 != s { en=strtonum("0x"$1); exit }
    END { if (st && en) printf "0x%x 0x%x\n", st, en }'
}
dump() {
  read -r st en <<< "$(range_of "$1" "$2")"
  llvm-objdump -d --no-show-raw-insn --start-address=$st --stop-address=$en "$1" 2>/dev/null \
    | awk '/^\s*[0-9a-f]+:/ { sub(/^\s*[0-9a-f]+:\s*/,""); print }' \
    | sed 's/<[^>]*>//g; s/#.*//; s/\s*$//; /^\s*$/d'
}
for spec in "$OUR|$OSYM|OURS" "$ASM|$DSYM|DAV1D"; do
  IFS='|' read -r bin sym tag <<< "$spec"
  f=$(mktemp --suffix=.s)
  dump "$bin" "$sym" > "$f"
  n=$(grep -c . "$f")
  echo "=== $tag $sym ($n insns) ==="
  llvm-mca -mcpu=znver4 --iterations=$ITER --instruction-info=0 --skip-unsupported-instructions=parse-failure "$f" 2>/dev/null \
    | grep -E "Iterations:|Total Cycles|Total uOps|uOps Per Cycle|IPC:|Block RThroughput"
  rm -f "$f"
done
