#!/usr/bin/env bash
# Correctness gates for the size-sweep branch. NOT a measurement — everything
# here is NICED so it can share the box with another agent's timed run.
#
# The DEFAULT build must be byte-unchanged by this branch: `bps-rows` is a
# compile-time A/B arm, `BorrowTracker::set_row_stride` is a no-op without it,
# and `DisjointMut::declare_row_stride` is called unconditionally by the picture
# allocator precisely so the seam is compiled on the shipped path. The corpus
# legs are the check on that claim, plus a leg on the arm itself — a rule that
# changes which BLOCK a byte lands in must still never lose an overlap.
#
# Usage: shardsize_gates.sh <outdir>
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
run "tracker crate, default + every ladder rung + the derived rule"
for f in "" __bps_quarter __bps_half __bps_1 __bps_4 __bps_8 __bps_rows __msb_5 __msb_5,__bps_rows; do
  if [ -z "$f" ]; then
    (cd crates/rav1d-disjoint-mut && nice -n 19 cargo test --release)
  else
    (cd crates/rav1d-disjoint-mut && nice -n 19 cargo test --release --features "$f")
  fi > "$OUT/tracker_${f:-default}.log" 2>&1
  echo "tracker ${f:-default} rc=$?" >&2
done

# --- 2. corpus, BY NAME, no --skip-group ------------------------------------
# Set-diff key is (group, name), value is (status, the ACTUAL md5) — a change
# that repairs 5 and breaks 5 is invisible in a count and obvious here.
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
corpus bpsrows bps-rows 8
corpus bpsrows bps-rows 1

python3 scripts/perf/md5_setdiff.py "$OUT/corpus_default_t1.tsv" "$OUT/corpus_default_t8.tsv" \
  > "$OUT/setdiff_default_t1_vs_t8.txt" 2>&1
echo "setdiff t1-vs-t8 rc=$?" >&2

# --- 3. the loop-filter window invariant, armed -----------------------------
run "debug-assertions corpus leg (8-bit/data, t=8)"
RUSTFLAGS="-C debug-assertions=on" nice -n 19 cargo build --release --example md5_inventory \
  > "$OUT/build_dbgassert.log" 2>&1
cp target/release/examples/md5_inventory "$OUT/mi_dbgassert"
nice -n 19 "$OUT/mi_dbgassert" --threads 8 --group 8-bit/data \
  > "$OUT/dbgassert_8bitdata_t8.tsv" 2> "$OUT/dbgassert_8bitdata_t8.err"
echo "dbgassert rc=$? lines=$(wc -l < "$OUT/dbgassert_8bitdata_t8.tsv")" >&2
# Same leg on the ARM, because the arm is what moves block boundaries.
RUSTFLAGS="-C debug-assertions=on" nice -n 19 cargo build --release --example md5_inventory \
  --features bps-rows > "$OUT/build_dbgassert_rows.log" 2>&1
cp target/release/examples/md5_inventory "$OUT/mi_dbgassert_rows"
nice -n 19 "$OUT/mi_dbgassert_rows" --threads 8 --group 8-bit/data \
  > "$OUT/dbgassert_rows_8bitdata_t8.tsv" 2> "$OUT/dbgassert_rows_8bitdata_t8.err"
echo "dbgassert rows rc=$? lines=$(wc -l < "$OUT/dbgassert_rows_8bitdata_t8.tsv")" >&2

# --- 4. stress ---------------------------------------------------------------
run "mt_stress + overlap/cleanup + multi_decoder_pressure"
nice -n 19 cargo test --release --test mt_stress > "$OUT/mt_stress.log" 2>&1
echo "mt_stress rc=$?" >&2
nice -n 19 cargo test --release --features bps-rows --test mt_stress > "$OUT/mt_stress_rows.log" 2>&1
echo "mt_stress rows rc=$?" >&2
nice -n 19 cargo test --release --test tile_threading_overlap --test reproduce_overlap \
  --test thread_cleanup_test > "$OUT/overlap_tests.log" 2>&1
echo "overlap/cleanup tests rc=$?" >&2
nice -n 19 bash scripts/perf/multi_decoder_pressure.sh > "$OUT/multi_decoder_pressure.log" 2>&1
echo "multi_decoder_pressure rc=$?" >&2

# --- 5. clippy, both targets, reproduced locally ----------------------------
run "x86_64 clippy"
nice -n 19 cargo clippy --release --target x86_64-apple-darwin --all-targets \
  -- -D warnings > "$OUT/clippy_x86.log" 2>&1
echo "clippy x86 rc=$?" >&2
run "aarch64 clippy"
nice -n 19 cargo clippy --release --all-targets -- -D warnings > "$OUT/clippy_arm.log" 2>&1
echo "clippy arm rc=$?" >&2
run "aarch64 clippy, the arm"
nice -n 19 cargo clippy --release --features bps-rows --all-targets \
  -- -D warnings > "$OUT/clippy_arm_rows.log" 2>&1
echo "clippy arm bps-rows rc=$?" >&2

echo "[$(date +%H:%M:%S)] gates written to $OUT" >&2
