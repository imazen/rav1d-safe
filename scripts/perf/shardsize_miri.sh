#!/usr/bin/env bash
# Miri over the tracker crate, BOTH aliasing models, ONE TARGET AT A TIME.
#
# One target at a time is not fussiness: Miri aborts the process on first UB and
# cargo stops at the first failing TARGET, so running them together once let five
# targets never run at all and their silence read as health.
#
# A `rc=1` whose log has no `test result:` line is an INVOCATION error, not a
# failure. A timeout is `rc=124` and is reported AS a timeout, never as green.
# "0 tests ran" is reported as 0 — two targets are feature-gated and select
# nothing under default features.
#
# The `__bps_rows` leg is the point of this run: the derived rule moves block
# boundaries, and the tracker's whole soundness argument is that both registrants
# of a shared byte agree on them.
#
# Usage: shardsize_miri.sh <outdir> [timeout_seconds]
set -u
OUT=${1:?outdir}; TMO=${2:-900}
mkdir -p "$OUT"
cd "$(dirname "$0")/../.."

# `shard_liveness` LAST, deliberately: it is the slow one (the AGENT_BRIEF warns
# it may time out on aarch64), and with it first a single timeout stalls the
# whole matrix behind one target. Ordered cheap-to-dear so a partial run still
# covers the models and feature sets.
TARGETS="${TARGETS:---lib narrow_release soundness wide_exclusion guard_move_release pic_buf_overflow aligned_miri shard_liveness}"

for model in sb tb; do
  case $model in
    sb) FLAGS="" ;;
    tb) FLAGS="-Zmiri-tree-borrows" ;;
  esac
  for feat in "" "__bps_rows"; do
    tag="${model}_${feat:-default}"
    for t in $TARGETS; do
      if [ "$t" = "--lib" ]; then sel="--lib"; else sel="--test $t"; fi
      log="$OUT/miri_${tag}_${t#--}.log"
      # shellcheck disable=SC2086
      if [ -z "$feat" ]; then
        MIRIFLAGS="$FLAGS" nice -n 19 timeout "$TMO" \
          cargo +nightly miri test -p rav1d-disjoint-mut --no-fail-fast $sel \
          > "$log" 2>&1
      else
        MIRIFLAGS="$FLAGS" nice -n 19 timeout "$TMO" \
          cargo +nightly miri test -p rav1d-disjoint-mut --no-fail-fast --features "$feat" $sel \
          > "$log" 2>&1
      fi
      rc=$?
      res=$(grep -c "^test result:" "$log"); res=${res:-0}
      n=$(grep -oE "^test result: ok\. [0-9]+ passed" "$log" | grep -oE "[0-9]+" | head -1)
      if [ "$rc" = "124" ]; then
        verdict="TIMEOUT(${TMO}s)"
      elif [ "$res" -eq 0 ]; then
        verdict="NO_TEST_RESULT_LINE(rc=$rc) — invocation error, not a pass"
      elif [ "${n:-0}" = "0" ]; then
        verdict="0 tests ran (rc=$rc) — proves nothing here"
      else
        verdict="rc=$rc passed=${n:-?}"
      fi
      printf '%s\t%s\t%s\n' "$tag" "${t#--}" "$verdict" | tee -a "$OUT/miri_summary.tsv"
    done
  done
done
echo "wrote $OUT/miri_summary.tsv" >&2
