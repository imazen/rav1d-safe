#!/usr/bin/env bash
# Correctness gates for the `c256x2048` contention round.
# NOT a measurement — everything here is NICED so it can share the box.
#
# This round adds no shipping behaviour: every change is behind a `__`-gated
# feature absent from `default` and from every published feature. The corpus
# legs run ANYWAY, on the DEFAULT build at t=1 AND t=8 with no `--skip-group`,
# because "the default codegen is unchanged" is a claim about a file that was
# edited, and 766/766 by NAME is the evidence rather than the assertion. The
# four lock arms and the pin probe get their own compile+unit legs, since an arm
# that does not build is an arm whose measurement means nothing.
#
# Usage: c256_gates.sh <outdir>
set -u
OUT=${1:?outdir}
mkdir -p "$OUT"
cd "$(dirname "$0")/../.."
BASE=benchmarks/aarch64_md5_fixes_2026-08-07_final.tsv.zst

run() { echo "[$(date +%H:%M:%S)] $*" >&2; }

# --- 1. unit tests, release AND debug ---------------------------------------
run "unit tests (release)"
nice -n 19 cargo test --release --lib > "$OUT/units_release.log" 2>&1; echo "units_release rc=$?" >&2
run "unit tests (debug)"
nice -n 19 cargo test --lib > "$OUT/units_debug.log" 2>&1; echo "units_debug rc=$?" >&2

run "tracker crate: default + every lock arm + the shift ladders + no-default-features"
for f in "" __probe_lock_backoff __probe_lock_yield __probe_lock_relax __probe_lock_park \
         __probe_wide __probe_wide,__probe_lock_park __probe_wide,__probe_lock_relax \
         __probe_shiftpin __bps_blocks __rpb_2 __rpb_8 __rpb_16 __msb_5; do
  if [ -z "$f" ]; then
    (cd crates/rav1d-disjoint-mut && nice -n 19 cargo test --release)
  else
    (cd crates/rav1d-disjoint-mut && nice -n 19 cargo test --release --features "$f")
  fi > "$OUT/tracker_${f:-default}.log" 2>&1
  echo "tracker ${f:-default} rc=$?" >&2
done
(cd crates/rav1d-disjoint-mut && nice -n 19 cargo test --release --no-default-features) \
  > "$OUT/tracker_nodefault.log" 2>&1
echo "tracker --no-default-features rc=$?" >&2

# --- 2. corpus, BY NAME, no --skip-group ------------------------------------
corpus() { # <arm> <features> <threads>
  local arm=$1 feat=$2 t=$3
  run "corpus $arm t=$t"
  if [ "$feat" = "-" ]; then
    nice -n 19 cargo build --release --example md5_inventory > /dev/null 2>&1
  else
    nice -n 19 cargo build --release --example md5_inventory --features "$feat" > /dev/null 2>&1
  fi
  cp target/release/examples/md5_inventory "$OUT/mi_$arm"
  nice -n 19 "$OUT/mi_$arm" --threads "$t" > "$OUT/corpus_${arm}_t${t}.tsv" 2> "$OUT/corpus_${arm}_t${t}.err"
  echo "corpus $arm t=$t rc=$? lines=$(wc -l < "$OUT/corpus_${arm}_t${t}.tsv")" >&2
  python3 scripts/perf/md5_setdiff.py "$BASE" "$OUT/corpus_${arm}_t${t}.tsv" \
    > "$OUT/setdiff_${arm}_t${t}.txt" 2>&1
  echo "setdiff $arm t=$t rc=$?" >&2
}
corpus default - 1
corpus default - 8
# The two arms that change the tracker's own machinery rather than a constant.
corpus lockpark probe-lock-park 8
corpus lockrelax probe-lock-relax 8

python3 scripts/perf/md5_setdiff.py "$OUT/corpus_default_t1.tsv" "$OUT/corpus_default_t8.tsv" \
  > "$OUT/setdiff_default_t1_vs_t8.txt" 2>&1
echo "setdiff t1-vs-t8 rc=$?" >&2

# --- 3. the loop-filter window invariant, armed -----------------------------
run "debug-assertions corpus leg (8-bit/data, t=8), DEFAULT build"
RUSTFLAGS="-C debug-assertions=on" nice -n 19 cargo build --release --example md5_inventory \
  > "$OUT/build_dbgassert.log" 2>&1
cp target/release/examples/md5_inventory "$OUT/mi_dbgassert"
nice -n 19 "$OUT/mi_dbgassert" --threads 8 --group 8-bit/data \
  > "$OUT/dbgassert_8bitdata_t8.tsv" 2> "$OUT/dbgassert_8bitdata_t8.err"
echo "dbgassert rc=$? lines=$(wc -l < "$OUT/dbgassert_8bitdata_t8.tsv")" >&2

# --- 4. stress ---------------------------------------------------------------
run "mt_stress + overlap/cleanup + multi_decoder_pressure"
nice -n 19 cargo test --release --test mt_stress > "$OUT/mt_stress.log" 2>&1
echo "mt_stress rc=$?" >&2
# 9 of these 11 are `#[ignore]`d, so the plain invocation runs TWO of them.
nice -n 19 cargo test --release --test tile_threading_overlap --test reproduce_overlap \
  --test thread_cleanup_test > "$OUT/overlap_tests.log" 2>&1
echo "overlap/cleanup tests (default: runs 2 of 11) rc=$?" >&2
nice -n 19 cargo test --release --test tile_threading_overlap --test reproduce_overlap \
  --test thread_cleanup_test -- --ignored > "$OUT/overlap_tests_ignored.log" 2>&1
echo "overlap/cleanup tests (--ignored: the other 9) rc=$?" >&2
AVIF=${MDP_AVIF:-$HOME/tmp/bpsrows/vec} \
VECS=${MDP_VECS:-"C1024x576_420_8b__t8 C3840x256_420_8b__t8 C256x2048_420_8b__t8 C512x576_420_8b__t8 v4k_8tile"} \
  nice -n 19 bash scripts/perf/multi_decoder_pressure.sh \
  "${MDP_BIN:-$HOME/tmp/c256/bin/bench_plain}" 12 3 600 \
  > "$OUT/multi_decoder_pressure.log" 2>&1
echo "multi_decoder_pressure rc=$?" >&2

# --- 5. clippy, both targets, and the CI legs exactly -----------------------
run "x86_64 clippy"
nice -n 19 cargo clippy --release --target x86_64-apple-darwin --all-targets \
  -- -D warnings > "$OUT/clippy_x86.log" 2>&1
echo "clippy x86 rc=$?" >&2
run "aarch64 clippy"
nice -n 19 cargo clippy --release --all-targets -- -D warnings > "$OUT/clippy_arm.log" 2>&1
echo "clippy arm rc=$?" >&2
run "aarch64 clippy, the new arms"
nice -n 19 cargo clippy --release --features probe-lock-park,probe-wide --all-targets \
  -- -D warnings > "$OUT/clippy_arm_park.log" 2>&1
echo "clippy arm probe-lock-park,probe-wide rc=$?" >&2

run "the EXACT CI legs"
nice -n 19 cargo clippy --no-default-features --features "bitdepth_8,bitdepth_16" \
  -- -D warnings > "$OUT/ci_clippy_lib.log" 2>&1; echo "ci clippy lib rc=$?" >&2
nice -n 19 cargo clippy -p rav1d-disjoint-mut -- -D warnings \
  > "$OUT/ci_clippy_dm.log" 2>&1; echo "ci clippy dm rc=$?" >&2
nice -n 19 cargo clippy -p rav1d-disjoint-mut --no-default-features -- -D warnings \
  > "$OUT/ci_clippy_dm_nodefault.log" 2>&1; echo "ci clippy dm --no-default-features rc=$?" >&2
RUSTDOCFLAGS="-D warnings" nice -n 19 cargo doc -p rav1d-disjoint-mut --no-deps \
  > "$OUT/ci_doc_dm.log" 2>&1; echo "ci doc dm rc=$?" >&2
nice -n 19 cargo fmt --all -- --check > "$OUT/ci_fmt.log" 2>&1; echo "ci fmt rc=$?" >&2

echo "[$(date +%H:%M:%S)] gates written to $OUT" >&2
