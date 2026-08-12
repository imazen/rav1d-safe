#!/usr/bin/env bash
# Planted mutations for the layout / CDEF-rectangle round. A green test that
# cannot fail proves nothing, so every path this branch adds gets a mutation
# that MUST be caught, and the round reports the ones that are not.
#
# Six subjects:
#
#   1. the `__rows_rect` READ path (`for_rows` -> `rect.row`)
#   2. the `__rows_rect` WRITE path (`for_rows_mut` -> `rect.row_mut`)
#   3. the new record's MUTABILITY. A mutable rectangle that silently registered
#      as immutable would never conflict with a concurrent reader and would
#      still decode correctly on an uncontended run — the exact silent-corruption
#      shape this campaign keeps hitting. Making `add_rect_mut` call
#      `add_rect::<false>` must make the tracker's rect-vs-rect overlap test
#      FAIL (it asserts a panic that only a mutable record can raise).
#   4. the rectangle must actually FIRE at both seams under tile threading:
#      `probe-wide`'s `n_rect` must be nonzero, and `n_rect` must GROW when the
#      write side is armed. A timed arm whose rectangle never fires measures
#      nothing.
#   5. the wide path must stay unreached (`w_shards`/`w_blocks`/`w_full` = 0):
#      a rectangle that promoted to the wide list would degrade to its hull.
#   6. `forbid(unsafe_code)` proven ACTIVE, not read.
#
# Every mutation is restored from a backup COPY (never `git checkout --`) and
# verified byte-exact by sha256 AND `git diff --exit-code`.
#
# NICED throughout; nothing here is timed.
#
# Usage: layout_teeth.sh [outdir]
set -u
OUT=${1:-$HOME/tmp/layout/teeth}
mkdir -p "$OUT/bak"
cd "$(dirname "$0")/../.."
VEC=${VEC:-$HOME/tmp/lfg/stage/avif/c1024x576.avif}
note() { printf '%s\t%s\n' "$1" "$2" | tee -a "$OUT/summary.tsv"; }
: > "$OUT/summary.tsv"

backup() { cp "$1" "$OUT/bak/$(echo "$1" | tr / _)"; shasum -a 256 "$1"; }
restore() {
  cp "$OUT/bak/$(echo "$1" | tr / _)" "$1"
  shasum -a 256 "$1"
  git diff --exit-code -- "$1" > /dev/null && echo "restored clean: $1"
}

md5_of() {  # md5_of <features> <threads>
  local feat=$1 t=$2
  local args=(build --release --example bench_ab_decode -j 6 --target-dir "$OUT/tgt")
  [ -n "$feat" ] && args+=(--features "$feat")
  nice -n 19 cargo "${args[@]}" > "$OUT/build.log" 2>&1 || { echo BUILDFAIL; return; }
  nice -n 19 "$OUT/tgt/release/examples/bench_ab_decode" "$VEC" "$t" 2 1 teeth 2>&1 \
    | awk -F'\t' '$1=="CHECKSUM"{print $5}' | head -1
}

echo "== control ==" >&2
REF=$(md5_of "" 8);            note control_default_t8 "$REF"
REFR=$(md5_of "__rows_rect" 8); note control_rowsrect_t8 "$REFR"
[ "$REF" = "$REFR" ] && note control_arms_agree OK || { note control_arms_agree MISMATCH; }

echo "== 1. __rows_rect READ path: rows reversed ==" >&2
backup include/dav1d/picture.rs > "$OUT/sha_pic_before.txt"
python3 - <<'PY'
p='include/dav1d/picture.rs'; s=open(p).read()
old = "                for row in 0..h {\n                    f(row, rect.row(row));\n                }"
assert s.count(old) == 1, s.count(old)
# `row + 1` would trip the guard's own bounds assert and report as a panic
# rather than as wrong pixels; reversing stays in range, so the md5 is what
# has to catch it.
new = "                for row in 0..h {\n                    f(row, rect.row(h - 1 - row));\n                }"
open(p,'w').write(s.replace(old, new, 1))
PY
M=$(md5_of "__rows_rect" 8)
[ "$M" != "$REFR" ] && note mut_rows_rect_read_reversed "CAUGHT ($M)" \
                    || note mut_rows_rect_read_reversed "NOT CAUGHT"
restore include/dav1d/picture.rs > "$OUT/sha_pic_after1.txt"

echo "== 2. __rows_rect WRITE path: rows reversed ==" >&2
backup include/dav1d/picture.rs > /dev/null
python3 - <<'PY'
p='include/dav1d/picture.rs'; s=open(p).read()
old = "                for row in 0..h {\n                    f(row, rect.row_mut(row));\n                }"
assert s.count(old) == 1, s.count(old)
new = "                for row in 0..h {\n                    f(row, rect.row_mut(h - 1 - row));\n                }"
open(p,'w').write(s.replace(old, new, 1))
PY
M=$(md5_of "__rows_rect" 8)
[ "$M" != "$REFR" ] && note mut_rows_rect_write_reversed "CAUGHT ($M)" \
                    || note mut_rows_rect_write_reversed "NOT CAUGHT"
restore include/dav1d/picture.rs > "$OUT/sha_pic_after2.txt"

echo "== 3. the mutable record's MUTABILITY has teeth ==" >&2
backup crates/rav1d-disjoint-mut/src/tracker_shard.rs > /dev/null
python3 - <<'PY'
p='crates/rav1d-disjoint-mut/src/tracker_shard.rs'; s=open(p).read()
old = "    ) -> Option<BorrowId> {\n        self.add_rect::<true>(lo, seg, rows, stride)\n    }"
assert s.count(old) == 1, s.count(old)
new = "    ) -> Option<BorrowId> {\n        self.add_rect::<false>(lo, seg, rows, stride)\n    }"
open(p,'w').write(s.replace(old, new, 1))
PY
if nice -n 19 cargo test -p rav1d-disjoint-mut --all-features -j 6 \
     rect_vs_rect > "$OUT/rectvsrect_mutated.log" 2>&1; then
  note mut_add_rect_mut_becomes_immut "NOT CAUGHT — the rect-vs-rect test still passes"
else
  note mut_add_rect_mut_becomes_immut "CAUGHT ($(grep -c '^test .* FAILED\|panicked' "$OUT/rectvsrect_mutated.log") failure lines)"
fi
restore crates/rav1d-disjoint-mut/src/tracker_shard.rs > "$OUT/sha_tracker_after.txt"
nice -n 19 cargo test -p rav1d-disjoint-mut --all-features -j 6 rect_vs_rect \
  > "$OUT/rectvsrect_restored.log" 2>&1 \
  && note rect_vs_rect_restored PASS || note rect_vs_rect_restored FAIL

echo "== 4+5. liveness: the rectangle fires, and the wide path does not ==" >&2
wide() { # wide <features>
  nice -n 19 cargo build --release --example probe_tracker -j 6 \
    --features "$1" --target-dir "$OUT/tgtw" > "$OUT/build_w.log" 2>&1 \
    || { echo BUILDFAIL; return; }
  nice -n 19 "$OUT/tgtw/release/examples/probe_tracker" "$VEC" 8 10 2>&1 \
    | grep -E '^WIDE|^RECT|n_rect' | head -20
}
wide "probe-wide" > "$OUT/wide_base.txt";              note wide_base "$(tr '\n' ' ' < "$OUT/wide_base.txt")"
wide "probe-wide,__rows_rect" > "$OUT/wide_rows.txt";  note wide_rows "$(tr '\n' ' ' < "$OUT/wide_rows.txt")"
wide "probe-wide,__lf_rect,__rows_rect" > "$OUT/wide_both.txt"
note wide_both "$(tr '\n' ' ' < "$OUT/wide_both.txt")"

echo "== 6. forbid(unsafe_code) proven ACTIVE ==" >&2
backup src/picture.rs > /dev/null
python3 - <<'PY'
p='src/picture.rs'; s=open(p).read()
open(p,'w').write(s + "\nfn _teeth_unsafe_probe(x: u64) -> f64 { unsafe { core::mem::transmute(x) } }\n")
PY
if nice -n 19 cargo check --release -j 6 --target-dir "$OUT/tgt3" > "$OUT/unsafe.log" 2>&1; then
  note forbid_unsafe_active "NOT ENFORCED — build succeeded"
elif grep -q "lib.rs:13:12" "$OUT/unsafe.log"; then
  note forbid_unsafe_active "ENFORCED at lib.rs:13:12"
else
  note forbid_unsafe_active "build failed but not at the forbid anchor (see log)"
fi
restore src/picture.rs > "$OUT/sha_pic_after6.txt"

echo "== worktree clean? ==" >&2
git diff --stat -- src include lib.rs crates | tail -3
cat "$OUT/summary.tsv" >&2
