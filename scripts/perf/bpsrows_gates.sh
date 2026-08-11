#!/usr/bin/env bash
# Correctness gates for "the derived rows-per-block rule is the DEFAULT".
# NOT a measurement — everything here is NICED so it can share the box.
#
# The polarity matters: for the two previous rounds the default build was
# byte-unchanged and the corpus legs were a formality on the arm. Here the
# DEFAULT build is the change — every picture plane's block boundaries move — so
# the corpus is the gate, not a formality, and it runs on the shipped features
# with no `--skip-group`, at t=1 AND t=8, set-diffed BY NAME with the ACTUAL md5
# as the value (a change that repairs 5 and breaks 5 is invisible in a count).
#
# `bps-blocks` gets its own corpus leg because it is the new base arm: if the
# arm that is supposed to reproduce the old rule cannot decode the corpus, no
# A/B measured against it means anything.
#
# Usage: bpsrows_gates.sh <outdir>
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
run "tracker crate: default + BOTH ladders + the base arm + msb-5 + shiftpin"
for f in "" __bps_blocks __bps_quarter __bps_half __bps_1 __bps_4 __bps_8 \
         __rpb_2 __rpb_8 __rpb_16 __msb_5 __msb_5,__bps_blocks __probe_shiftpin; do
  if [ -z "$f" ]; then
    (cd crates/rav1d-disjoint-mut && nice -n 19 cargo test --release)
  else
    (cd crates/rav1d-disjoint-mut && nice -n 19 cargo test --release --features "$f")
  fi > "$OUT/tracker_${f:-default}.log" 2>&1
  echo "tracker ${f:-default} rc=$?" >&2
done

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
corpus bpsblocks bps-blocks 8

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
# 9 of these 11 are `#[ignore]`d in the repo, so the plain invocation reports
# `ok` having run TWO of them. Both invocations, or the line above the count is
# a green tick that cannot fail.
nice -n 19 cargo test --release --test tile_threading_overlap --test reproduce_overlap \
  --test thread_cleanup_test > "$OUT/overlap_tests.log" 2>&1
echo "overlap/cleanup tests (default: runs 2 of 11) rc=$?" >&2
nice -n 19 cargo test --release --test tile_threading_overlap --test reproduce_overlap \
  --test thread_cleanup_test -- --ignored > "$OUT/overlap_tests_ignored.log" 2>&1
echo "overlap/cleanup tests (--ignored: the other 9) rc=$?" >&2
# Takes the bench binary and a vector dir as ARGUMENTS; with none it exits 1 on
# its own usage check, which reads as a failure and is not one.
AVIF=${MDP_AVIF:-$HOME/tmp/bpsrows/vec} \
VECS=${MDP_VECS:-"C1024x576_420_8b__t8 C3840x256_420_8b__t8 C256x2048_420_8b__t8 C512x576_420_8b__t8 v4k_8tile"} \
  nice -n 19 bash scripts/perf/multi_decoder_pressure.sh \
  "${MDP_BIN:-$HOME/tmp/bpsrows/bin/bench_plain}" 12 3 600 \
  > "$OUT/multi_decoder_pressure.log" 2>&1
echo "multi_decoder_pressure rc=$?" >&2

# --- 5. clippy, both targets ------------------------------------------------
run "x86_64 clippy"
nice -n 19 cargo clippy --release --target x86_64-apple-darwin --all-targets \
  -- -D warnings > "$OUT/clippy_x86.log" 2>&1
echo "clippy x86 rc=$?" >&2
run "aarch64 clippy"
nice -n 19 cargo clippy --release --all-targets -- -D warnings > "$OUT/clippy_arm.log" 2>&1
echo "clippy arm rc=$?" >&2
run "aarch64 clippy, base arm + shiftpin probe"
nice -n 19 cargo clippy --release --features bps-blocks,probe-shiftpin --all-targets \
  -- -D warnings > "$OUT/clippy_arm_arms.log" 2>&1
echo "clippy arm bps-blocks,probe-shiftpin rc=$?" >&2

echo "[$(date +%H:%M:%S)] gates written to $OUT" >&2
