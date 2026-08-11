#!/usr/bin/env bash
# Miri for the strided-RECTANGLE round: both aliasing models x the default
# feature set and `__rect_1shard`, ONE TARGET AT A TIME.
#
# Miri is the only gate here that checks the ALIASING MODEL rather than the
# tracker's own bookkeeping, and it is the gate this mechanism most needs: the
# March-2026 strided tracker had an exact record and a reference over the whole
# hull, and that combination is UB under both models — 766 corpus vectors did not
# see it and Miri did. `DisjointImmutRectGuard` has no `Deref` and derives each
# row from the buffer pointer for exactly that reason; these legs are what says
# so rather than asserting it.
#
# One target at a time because Miri aborts the process on first UB and cargo
# stops at the first failing TARGET — a batch run lets later targets never
# execute, and their silence reads as health.
#
# `__rect_1shard` is the second feature set because it changes which of
# `add_rect`'s two registration shapes runs (single-shard fast path vs the
# sort/lock/scan loop), and those are different pointer/reference sequences.
#
# `shard_liveness` is expected to TIME OUT on aarch64 (docs/BPS_ROWS_DEFAULT.md
# §8c); rc=124 is reported AS a timeout, never as green.
#
# Usage: c256_miri.sh <out.tsv> [timeout_secs]
set -u
OUT=${1:?out.tsv}; T=${2:-900}
cd "$(dirname "$0")/../.."
LOGDIR=$(dirname "$OUT")/miri_logs; mkdir -p "$LOGDIR"
printf 'model\tfeatures\ttarget\trc\tresult\n' > "$OUT"
for model in sb tb; do
  case $model in
    sb) FLAGS="-Zmiri-disable-isolation" ;;
    tb) FLAGS="-Zmiri-disable-isolation -Zmiri-tree-borrows" ;;
  esac
  for feat in default __rect_1shard; do
    for tgt in --lib narrow_release soundness wide_exclusion guard_move_release \
               pic_buf_overflow aligned_miri shard_liveness; do
      if [ "$tgt" = "--lib" ]; then SEL="--lib"; else SEL="--test $tgt"; fi
      if [ "$feat" = default ]; then FS=""; else FS="--features $feat"; fi
      log="$LOGDIR/${model}_${feat}_${tgt#--}.log"
      echo "[$(date +%H:%M:%S)] miri $model $feat $tgt" >&2
      MIRIFLAGS="$FLAGS" nice -n 19 timeout "$T" \
        cargo +nightly miri test -p rav1d-disjoint-mut --no-fail-fast $FS $SEL \
        > "$log" 2>&1
      rc=$?
      res=$(grep -m1 -oE 'test result: [a-zA-Z]+\. [0-9]+ passed' "$log" | tail -1)
      [ "$rc" = 124 ] && res="TIMEOUT(${T}s)"
      [ -z "$res" ] && res="(no test result line)"
      printf '%s\t%s\t%s\t%s\t%s\n' "$model" "$feat" "$tgt" "$rc" "$res" >> "$OUT"
    done
  done
done
echo "wrote $OUT" >&2
