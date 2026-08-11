#!/usr/bin/env bash
# Correctness gates for the strided-rectangle record round.
#
# Two things this does that a plain `cargo test` does not:
#
#  1. The corpus runs in BOTH arms (`--features __lf_rect` and the DEFAULT
#     codegen) at t=1 AND t=8, with NO `--skip-group`, and the result is
#     set-diffed BY NAME against the committed baseline. The default arm is not
#     optional: the tracker's hot `find` was edited, so "the rectangle path is
#     off" is a claim about a file that changed and 766/766 by name is the
#     evidence rather than the assertion. t=1 is not optional either: the
#     rectangle path is gated on `tile_threading_active()`, so t=1 exercises the
#     MACHINERY with the path never taken — which is exactly the configuration
#     the default would ship to a single-threaded caller.
#  2. The tracker crate's unit tests run once per feature configuration that can
#     change the rectangle code's meaning, ONE AT A TIME (cargo stops at the
#     first failing target, so a batch run lets later configurations never
#     execute and their silence reads as health).
#
# NICED throughout and it takes NO measurement lock: nothing here is timed.
#
# Usage: rect_gates.sh [outdir]
set -u
OUT=${1:-$HOME/tmp/rectrec/gates}
mkdir -p "$OUT"
cd "$(dirname "$0")/../.."
BASELINE=benchmarks/aarch64_md5_fixes_2026-08-07_final.tsv.zst
rc_all=0
note() { printf '%s\t%s\n' "$1" "$2" | tee -a "$OUT/summary.tsv"; }

: > "$OUT/summary.tsv"

echo "== tracker unit tests, one feature configuration at a time ==" >&2
for feat in "-" "--no-default-features" "--features __rect_1shard" \
            "--features __probe_wide" "--features __probe_wide,__rect_1shard" \
            "--features __probe_sites" "--features zerocopy" \
            "--features __probe_lock_park" "--features __bps_blocks" \
            "--features __msb_5" "--features __shards_1" "--all-features"; do
  tag=$(echo "$feat" | tr -c 'a-zA-Z0-9' '_')
  # shellcheck disable=SC2086
  if [ "$feat" = "-" ]; then set -- ; else set -- $feat; fi
  if nice -n 19 cargo test -p rav1d-disjoint-mut --lib "$@" \
       > "$OUT/unit_$tag.log" 2>&1; then
    note "unit$tag" "PASS $(grep -c '^test .* ok$' "$OUT/unit_$tag.log") tests"
  else
    note "unit$tag" "FAIL"; rc_all=1
  fi
done

echo "== release + debug lib tests (debug arms every debug_assert!) ==" >&2
for prof in "--release" ""; do
  tag=${prof:-debug}; tag=${tag#--}
  # shellcheck disable=SC2086
  if nice -n 19 cargo test --lib $prof > "$OUT/lib_$tag.log" 2>&1; then
    note "lib_$tag" "PASS"
  else
    note "lib_$tag" "FAIL"; rc_all=1
  fi
done

echo "== corpus, BOTH arms x t=1 and t=8, no --skip-group ==" >&2
for arm in default rect; do
  feat=""; [ "$arm" = rect ] && feat="--features __lf_rect"
  # shellcheck disable=SC2086
  nice -n 19 cargo build --release --example md5_inventory $feat \
    > "$OUT/build_$arm.log" 2>&1 || { note "build_$arm" FAIL; rc_all=1; continue; }
  cp target/release/examples/md5_inventory "$OUT/mi_$arm"
  for t in 1 8; do
    tsv="$OUT/md5_${arm}_t$t.tsv"
    nice -n 19 "$OUT/mi_$arm" --threads "$t" > "$tsv" 2> "$OUT/md5_${arm}_t$t.err"
    tot=$(grep '^TOTAL' "$OUT/md5_${arm}_t$t.err" | tail -1)
    note "corpus_${arm}_t$t" "$tot"
    case "$tot" in *"mismatch=0 error=0"*) ;; *) rc_all=1 ;; esac
    if python3 scripts/perf/md5_setdiff.py "$BASELINE" "$tsv" \
         > "$OUT/setdiff_${arm}_t$t.log" 2>&1; then
      note "setdiff_${arm}_t$t" "CLEAN"
    else
      note "setdiff_${arm}_t$t" "DIFFERS"; rc_all=1
    fi
  done
  # t=1 vs t=8 in the same arm: a threading-only divergence shows here and
  # nowhere else.
  if python3 scripts/perf/md5_setdiff.py "$OUT/md5_${arm}_t1.tsv" \
       "$OUT/md5_${arm}_t8.tsv" > "$OUT/setdiff_${arm}_t1t8.log" 2>&1; then
    note "setdiff_${arm}_t1_vs_t8" "CLEAN"
  else
    note "setdiff_${arm}_t1_vs_t8" "DIFFERS"; rc_all=1
  fi
done

echo "== concurrency + threading tests, rectangle arm ==" >&2
for t in "decode_md5_verify" "thread_cleanup_test" "tile_threading_overlap" \
         "reproduce_overlap" "mt_stress"; do
  if nice -n 19 cargo test --release --features __lf_rect --test "$t" \
       > "$OUT/test_$t.log" 2>&1; then
    note "test_$t" "PASS"
  else
    # a target that selects no tests is reported as such, never as green
    if grep -q '0 passed; 0 failed' "$OUT/test_$t.log"; then
      note "test_$t" "NO TESTS SELECTED"
    else
      note "test_$t" "FAIL"; rc_all=1
    fi
  fi
  if nice -n 19 cargo test --release --features __lf_rect --test "$t" -- --ignored \
       > "$OUT/test_${t}_ignored.log" 2>&1; then
    note "test_${t}_ignored" "PASS"
  else
    note "test_${t}_ignored" "FAIL-or-none (see log)"
  fi
done

echo "== clippy legs + fmt ==" >&2
for leg in "-p rav1d-disjoint-mut --all-targets" \
           "-p rav1d-disjoint-mut --no-default-features --all-targets" \
           "-p rav1d-disjoint-mut --all-features --all-targets" \
           "--lib" "--lib --features __lf_rect" "--lib --features __lf_rect1"; do
  tag=$(echo "$leg" | tr -c 'a-zA-Z0-9' '_')
  # shellcheck disable=SC2086
  if nice -n 19 cargo clippy $leg -- -D warnings > "$OUT/clippy_$tag.log" 2>&1; then
    note "clippy_$tag" "rc=0"
  else
    note "clippy_$tag" "rc!=0"; rc_all=1
  fi
done
if nice -n 19 cargo fmt --all -- --check > "$OUT/fmt.log" 2>&1; then
  note "fmt_check" "rc=0"
else
  note "fmt_check" "rc!=0"; rc_all=1
fi

echo "== summary ==" >&2
cat "$OUT/summary.tsv" >&2
echo "rc=$rc_all" >&2
exit "$rc_all"
