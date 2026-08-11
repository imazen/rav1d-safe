#!/usr/bin/env bash
# Correctness gates for the layout-attribution round (`docs/RECT_SHIP.md`).
#
# This branch's headline claim is a NEGATIVE — the t=1 cost #505 attributed to
# the rectangle is code placement — so the gates have a different shape from
# #505's:
#
#  1. **The DEFAULT build's codegen must be unchanged.** The round adds two
#     measurement features and splits `LfBlock::fill`, and the whole finding is
#     that a default binary which is not byte-equivalent to `main`'s pays ~1.1%
#     at t=1 on `v4k8tile`. So `text_layout_diff.py` against a binary built from
#     the base commit is a GATE here, not a diagnostic: 0 symbols resized, 0
#     symbols added, identical `__text`.
#  2. The corpus still runs in BOTH arms at t=1 AND t=8 with no `--skip-group`,
#     set-diffed BY NAME — `src/loopfilter.rs`, `include/dav1d/picture.rs`,
#     `src/cdef_apply.rs` and `src/safe_simd/cdef_arm.rs` all changed.
#  3. The two new measurement arms are built and run, because a probe that does
#     not compile is not a probe.
#
# NICED throughout and it takes NO measurement lock: nothing here is timed.
#
# Usage: rectship_gates.sh [outdir] [base-binary]
set -u
OUT=${1:-$HOME/tmp/rectship/gates}
BASEBIN=${2:-$HOME/tmp/rectship/bin/bench_plain}
mkdir -p "$OUT"
cd "$(dirname "$0")/../.."
BASELINE=benchmarks/aarch64_md5_fixes_2026-08-07_final.tsv.zst
rc_all=0
note() { printf '%s\t%s\n' "$1" "$2" | tee -a "$OUT/summary.tsv"; }
: > "$OUT/summary.tsv"

echo "== 1. DEFAULT codegen unchanged vs the base commit's binary ==" >&2
nice -n 19 cargo build --release --example bench_ab_decode \
  > "$OUT/build_default.log" 2>&1 || { note build_default FAIL; rc_all=1; }
cp target/release/examples/bench_ab_decode "$OUT/bench_default"
python3 scripts/perf/text_layout_diff.py "$BASEBIN" "$OUT/bench_default" \
  > "$OUT/layout_default.txt" 2>&1
res=$(awk -F'\t' '$1=="resized_in_both"{print $2}' "$OUT/layout_default.txt")
add=$(awk -F'\t' '$1=="only_in_head"{print $2}' "$OUT/layout_default.txt")
tb=$(awk -F'\t' '$1=="text_base"{print $2}' "$OUT/layout_default.txt")
th=$(awk -F'\t' '$1=="text_head"{print $2}' "$OUT/layout_default.txt")
note "default_codegen" "resized=$res added=$add text $tb -> $th"
[ "$res" = 0 ] && [ "$add" = 0 ] && [ "$tb" = "$th" ] || rc_all=1

echo "== 2. release + debug lib tests ==" >&2
for prof in "--release" ""; do
  tag=${prof:-debug}; tag=${tag#--}
  if [ -n "$prof" ]; then
    nice -n 19 cargo test --lib --release > "$OUT/lib_$tag.log" 2>&1
  else
    nice -n 19 cargo test --lib > "$OUT/lib_$tag.log" 2>&1
  fi
  [ $? -eq 0 ] && note "lib_$tag" PASS || { note "lib_$tag" FAIL; rc_all=1; }
done

echo "== 3. corpus, BOTH arms x t=1 and t=8, no --skip-group ==" >&2
for arm in default rect; do
  if [ "$arm" = rect ]; then
    nice -n 19 cargo build --release --example md5_inventory --features __lf_rect \
      > "$OUT/build_mi_$arm.log" 2>&1
  else
    nice -n 19 cargo build --release --example md5_inventory \
      > "$OUT/build_mi_$arm.log" 2>&1
  fi
  [ $? -eq 0 ] || { note "build_mi_$arm" FAIL; rc_all=1; continue; }
  cp target/release/examples/md5_inventory "$OUT/mi_$arm"
  for t in 1 8; do
    tsv="$OUT/md5_${arm}_t$t.tsv"
    nice -n 19 "$OUT/mi_$arm" --threads "$t" > "$tsv" 2> "$OUT/md5_${arm}_t$t.err"
    tot=$(grep '^TOTAL' "$OUT/md5_${arm}_t$t.err" | tail -1)
    note "corpus_${arm}_t$t" "$tot"
    case "$tot" in *"mismatch=0 error=0"*) ;; *) rc_all=1 ;; esac
    if python3 scripts/perf/md5_setdiff.py "$BASELINE" "$tsv" \
         > "$OUT/setdiff_${arm}_t$t.log" 2>&1; then
      note "setdiff_${arm}_t$t" CLEAN
    else
      note "setdiff_${arm}_t$t" DIFFERS; rc_all=1
    fi
  done
  if python3 scripts/perf/md5_setdiff.py "$OUT/md5_${arm}_t1.tsv" \
       "$OUT/md5_${arm}_t8.tsv" > "$OUT/setdiff_${arm}_t1t8.log" 2>&1; then
    note "setdiff_${arm}_t1_vs_t8" CLEAN
  else
    note "setdiff_${arm}_t1_vs_t8" DIFFERS; rc_all=1
  fi
done

echo "== 4. the measurement arms build AND run ==" >&2
for feat in __probe_cdef_double __pad_text __pad_small __pad2 __pad3 __pad4 \
            __pad_far __lf_rect __lf_rect1 __probe_lf_hull __probe_bounds; do
  if nice -n 19 cargo build --release --example bench_ab_decode --features "$feat" \
       --target-dir "$OUT/tgt" > "$OUT/build_$feat.log" 2>&1; then
    note "build_$feat" rc=0
  else
    note "build_$feat" FAIL; rc_all=1
  fi
done

echo "== 5. concurrency + threading tests, rectangle arm ==" >&2
for t in decode_md5_verify thread_cleanup_test tile_threading_overlap \
         reproduce_overlap mt_stress; do
  nice -n 19 cargo test --release --features __lf_rect --test "$t" \
    > "$OUT/test_$t.log" 2>&1 \
    && note "test_$t" PASS || { note "test_$t" FAIL; rc_all=1; }
  nice -n 19 cargo test --release --features __lf_rect --test "$t" -- --ignored \
    > "$OUT/test_${t}_ignored.log" 2>&1 \
    && note "test_${t}_ignored" PASS || note "test_${t}_ignored" "see log"
done

echo "== 6. clippy legs + fmt ==" >&2
run_clippy() { tag=$1; shift; if nice -n 19 cargo clippy "$@" -- -D warnings \
    > "$OUT/clippy_$tag.log" 2>&1; then note "clippy_$tag" rc=0; \
  else note "clippy_$tag" 'rc!=0'; rc_all=1; fi; }
run_clippy tracker            -p rav1d-disjoint-mut --all-targets
run_clippy tracker_nodefault  -p rav1d-disjoint-mut --no-default-features --all-targets
run_clippy lib                --lib
run_clippy lib_rect           --lib --features __lf_rect
run_clippy lib_rect1          --lib --features __lf_rect1
run_clippy lib_cdefdouble     --lib --features __probe_cdef_double
run_clippy lib_pad            --lib --features __pad4
run_clippy lib_padfar         --lib --features __pad_far
if nice -n 19 cargo fmt --all -- --check > "$OUT/fmt.log" 2>&1; then
  note fmt_check rc=0
else
  note fmt_check 'rc!=0'; rc_all=1
fi

echo "== summary ==" >&2
cat "$OUT/summary.tsv" >&2
echo "rc=$rc_all" >&2
exit "$rc_all"
