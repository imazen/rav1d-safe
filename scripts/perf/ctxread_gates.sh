#!/usr/bin/env bash
# Correctness gates for the LEFT-context read split. NOT a measurement —
# everything here is NICED so it can share the box with another agent's timed
# run, and nothing here takes `measlock`.
#
# The change removes a tracker RECORD; it changes no extent and no pixel. The
# corpus legs are the check on the second half of that claim, by name, with the
# actual md5 as the value.
#
# Usage: ctxread_gates.sh <outdir>
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
run "tracker crate"
(cd crates/rav1d-disjoint-mut && nice -n 19 cargo test --release) \
  > "$OUT/tracker_default.log" 2>&1; echo "tracker rc=$?" >&2

# --- 2. corpus, BY NAME, no --skip-group ------------------------------------
corpus() { # <threads>
  local t=$1
  run "corpus t=$t"
  nice -n 19 "$OUT/mi_head" --threads "$t" > "$OUT/corpus_t${t}.tsv" 2> "$OUT/corpus_t${t}.err"
  echo "corpus t=$t rc=$? lines=$(wc -l < "$OUT/corpus_t${t}.tsv")" >&2
  python3 scripts/perf/md5_setdiff.py "$BASE" "$OUT/corpus_t${t}.tsv" \
    > "$OUT/setdiff_t${t}.txt" 2>&1
  echo "setdiff t=$t rc=$?" >&2
}
run "build md5_inventory"
nice -n 19 cargo build --release --example md5_inventory > "$OUT/build_mi.log" 2>&1
cp target/release/examples/md5_inventory "$OUT/mi_head"
corpus 1
corpus 8
python3 scripts/perf/md5_setdiff.py "$OUT/corpus_t1.tsv" "$OUT/corpus_t8.tsv" \
  > "$OUT/setdiff_t1_vs_t8.txt" 2>&1
echo "setdiff t1-vs-t8 rc=$?" >&2

# --- 3. the loop-filter window invariant, armed -----------------------------
run "debug-assertions corpus leg (8-bit/data, t=8)"
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
nice -n 19 cargo test --release --test tile_threading_overlap --test reproduce_overlap \
  --test thread_cleanup_test > "$OUT/overlap_tests.log" 2>&1
echo "overlap/cleanup tests rc=$?" >&2
AVIF=${MDP_AVIF:-$HOME/tmp/shardsize/vec} \
VECS=${MDP_VECS:-"C1024x576_420_8b__t8 C3840x256_420_8b__t8 C256x2048_420_8b__t8 C512x288_420_8b__t8 C3840x2160_420_8b__t8"} \
  nice -n 19 bash scripts/perf/multi_decoder_pressure.sh \
  "${MDP_BIN:-$HOME/tmp/lfg/bin/bench_head}" 12 3 600 \
  > "$OUT/multi_decoder_pressure.log" 2>&1
echo "multi_decoder_pressure rc=$?" >&2

# --- 5. clippy, both targets, reproduced locally ----------------------------
run "x86_64 clippy"
nice -n 19 cargo clippy --release --target x86_64-apple-darwin --all-targets \
  -- -D warnings > "$OUT/clippy_x86.log" 2>&1
echo "clippy x86 rc=$?" >&2
run "aarch64 clippy"
nice -n 19 cargo clippy --release --all-targets -- -D warnings > "$OUT/clippy_arm.log" 2>&1
echo "clippy arm rc=$?" >&2

echo "[$(date +%H:%M:%S)] gates written to $OUT" >&2
echo "GATES_DONE" >&2
