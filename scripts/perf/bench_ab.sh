#!/usr/bin/env bash
# Interleaved A/B benchmark: alternates binaries to cancel thermal drift.
# Usage: bench_ab.sh <avif> <iters> <rounds> [binaries...]
set -u
AVIF="${1:-test-vectors/bench/photo_4k.avif}"
ITERS="${2:-15}"
ROUNDS="${3:-3}"
shift 3 || true
BINS=("$@")
if [ ${#BINS[@]} -eq 0 ]; then
    BINS=(target/asm/release/examples/profile_avif target/safe/release/examples/profile_avif)
fi

declare -A TIMES
for r in $(seq 1 "$ROUNDS"); do
    for b in "${BINS[@]}"; do
        out=$("$b" "$AVIF" "$ITERS" 2>&1 | grep "ms/iter" | grep -oE "[0-9]+\.[0-9]+ms" | tr -d 'ms')
        TIMES[$b]+="$out "
        echo "round$r $(basename $(dirname $(dirname $(dirname $b))))/$(basename $b): ${out}ms"
    done
done
echo "=== summary (median ms/iter) ==="
for b in "${BINS[@]}"; do
    med=$(echo ${TIMES[$b]} | tr ' ' '\n' | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
    echo "$b : $med ms"
done
