#!/usr/bin/env bash
# Correctness gates for the layout / CDEF-rectangle round (`docs/LAYOUT_CDEF.md`).
#
# Two things are on trial and they need different gates:
#
#  1. **Function alignment** (`-C llvm-args=-align-all-functions=N`) changes no
#     source at all, so its correctness gate is that the corpus still passes and
#     that every timed arm decodes to the SAME md5 — which is checked before any
#     clock by `layout_checksums.sh`, and again here over the whole corpus.
#  2. **`__rows_rect`** collapses `for_rows` / `for_rows_mut`'s per-row
#     registrations into ONE exact strided-rectangle record, immutable and
#     MUTABLE respectively. That is a live change to the borrow tracker's view
#     of the decoder at t=8, so it needs the corpus at BOTH thread counts,
#     set-diffed BY NAME, plus the concurrency tests.
#
# And the round inherits #506's gate 1: the DEFAULT build's codegen must be
# unchanged, because a default binary that is not byte-equivalent to `main`'s
# pays ~1.1% at t=1 on `v4k8tile` from placement alone.
#
# NICED throughout, and it takes NO measurement lock: nothing here is timed.
#
# Usage: layout_gates.sh [outdir] [base-binary]
set -u
OUT=${1:-$HOME/tmp/layout/gates}
BASEBIN=${2:-$HOME/tmp/layout/bin/bench_a0plain}
mkdir -p "$OUT"
cd "$(dirname "$0")/../.."
BASELINE=benchmarks/aarch64_md5_fixes_2026-08-07_final.tsv.zst
rc_all=0
note() { printf '%s\t%s\n' "$1" "$2" | tee -a "$OUT/summary.tsv"; }
: > "$OUT/summary.tsv"

echo "== 1. DEFAULT codegen unchanged vs the pre-change binary ==" >&2
nice -n 19 cargo build --release --example bench_ab_decode -j 6 \
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
nice -n 19 cargo test --lib --release -j 6 > "$OUT/lib_release.log" 2>&1 \
  && note lib_release PASS || { note lib_release FAIL; rc_all=1; }
nice -n 19 cargo test --lib -j 6 > "$OUT/lib_debug.log" 2>&1 \
  && note lib_debug PASS || { note lib_debug FAIL; rc_all=1; }
nice -n 19 cargo test -p rav1d-disjoint-mut --all-features -j 6 \
  > "$OUT/tracker_tests.log" 2>&1 \
  && note tracker_tests PASS || { note tracker_tests FAIL; rc_all=1; }

echo "== 3. corpus: default, __rows_rect, and the 32-byte-aligned default ==" >&2
run_corpus() { # <tag> <rustflags> <features...>
  local tag=$1 rf=$2; shift 2
  local args=(build --release --example md5_inventory -j 6 --target-dir "$OUT/tgt_$tag")
  [ $# -gt 0 ] && [ -n "$1" ] && args+=(--features "$1")
  RUSTFLAGS="$rf" nice -n 19 cargo "${args[@]}" > "$OUT/build_mi_$tag.log" 2>&1 \
    || { note "build_mi_$tag" FAIL; rc_all=1; return; }
  cp "$OUT/tgt_$tag/release/examples/md5_inventory" "$OUT/mi_$tag"
  for t in 1 8; do
    local tsv="$OUT/md5_${tag}_t$t.tsv"
    nice -n 19 "$OUT/mi_$tag" --threads "$t" > "$tsv" 2> "$OUT/md5_${tag}_t$t.err"
    local tot; tot=$(grep '^TOTAL' "$OUT/md5_${tag}_t$t.err" | tail -1)
    note "corpus_${tag}_t$t" "$tot"
    case "$tot" in *"mismatch=0 error=0"*) ;; *) rc_all=1 ;; esac
    if python3 scripts/perf/md5_setdiff.py "$BASELINE" "$tsv" \
         > "$OUT/setdiff_${tag}_t$t.log" 2>&1; then
      note "setdiff_${tag}_t$t" CLEAN
    else
      note "setdiff_${tag}_t$t" DIFFERS; rc_all=1
    fi
  done
  if python3 scripts/perf/md5_setdiff.py "$OUT/md5_${tag}_t1.tsv" \
       "$OUT/md5_${tag}_t8.tsv" > "$OUT/setdiff_${tag}_t1t8.log" 2>&1; then
    note "setdiff_${tag}_t1_vs_t8" CLEAN
  else
    note "setdiff_${tag}_t1_vs_t8" DIFFERS; rc_all=1
  fi
}
run_corpus default   ""                                        ""
run_corpus rowsrect  ""                                        "__rows_rect"
run_corpus a5        "-C llvm-args=-align-all-functions=5"      ""
run_corpus a5rowsrect "-C llvm-args=-align-all-functions=5"     "__rows_rect"

echo "== 4. every measurement arm still builds ==" >&2
for feat in __rows_rect __probe_cdef_double __pad_text __pad_small __pad2 __pad3 \
            __pad4 __pad_far __lf_rect __lf_rect1 __probe_lf_hull __probe_bounds; do
  if nice -n 19 cargo build --release --example bench_ab_decode -j 6 --features "$feat" \
       --target-dir "$OUT/tgt" > "$OUT/build_$feat.log" 2>&1; then
    note "build_$feat" rc=0
  else
    note "build_$feat" FAIL; rc_all=1
  fi
done

echo "== 5. concurrency + threading tests, __rows_rect arm ==" >&2
for t in decode_md5_verify thread_cleanup_test tile_threading_overlap \
         reproduce_overlap mt_stress; do
  nice -n 19 cargo test --release --features __rows_rect --test "$t" -j 6 \
    > "$OUT/test_$t.log" 2>&1 \
    && note "test_$t" PASS || { note "test_$t" FAIL; rc_all=1; }
  nice -n 19 cargo test --release --features __rows_rect --test "$t" -j 6 -- --ignored \
    > "$OUT/test_${t}_ignored.log" 2>&1 \
    && note "test_${t}_ignored" PASS || note "test_${t}_ignored" "see log"
done

echo "== 6. clippy legs + fmt ==" >&2
run_clippy() { tag=$1; shift; if nice -n 19 cargo clippy "$@" -j 6 -- -D warnings \
    > "$OUT/clippy_$tag.log" 2>&1; then note "clippy_$tag" rc=0; \
  else note "clippy_$tag" 'rc!=0'; rc_all=1; fi; }
run_clippy tracker             -p rav1d-disjoint-mut --all-targets
run_clippy tracker_nodefault   -p rav1d-disjoint-mut --no-default-features --all-targets
run_clippy tracker_rectmut     -p rav1d-disjoint-mut --features __rect_mut --all-targets
run_clippy lib                 --lib
run_clippy lib_rowsrect        --lib --features __rows_rect
run_clippy lib_rect            --lib --features __lf_rect
run_clippy lib_cdefdouble      --lib --features __probe_cdef_double
run_clippy alltargets_rowsrect --all-targets --features __rows_rect
if nice -n 19 cargo fmt --all -- --check > "$OUT/fmt.log" 2>&1; then
  note fmt_check rc=0
else
  note fmt_check 'rc!=0'; rc_all=1
fi

echo "== summary ==" >&2
cat "$OUT/summary.tsv" >&2
echo "rc=$rc_all" >&2
exit "$rc_all"
