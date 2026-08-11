#!/usr/bin/env bash
# Miri for the c256 contention round: both aliasing models x the default lock
# and the PARK arm, ONE TARGET AT A TIME.
#
# One target at a time because Miri aborts the process on first UB and cargo
# stops at the first failing TARGET — a batch run lets later targets never
# execute, and their silence reads as health.
#
# `__probe_lock_park` is the feature set worth a second pass: it replaces the
# shard lock's whole implementation, and the tracker's soundness argument rests
# on that lock's mutual exclusion. `__probe_lock_relax`/`_yield`/`_backoff` only
# change what a waiter does BETWEEN attempts, so they cannot move the argument
# and are covered by the unit-test legs instead.
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
  for feat in default __probe_lock_park; do
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
