#!/usr/bin/env bash
# Planted mutations for the layout-attribution round. A green test that cannot
# fail proves nothing, so every path this branch touches gets a mutation that
# MUST be caught, and the round reports the ones that are not.
#
# Four subjects, and the third is the one that makes the headline sound:
#
#   1. the restructured DEFAULT per-row path (`fill_threaded`)
#   2. the rectangle path (`fill_rect`), which #505 gated the same way
#   3. **the layout pad must NEVER EXECUTE.** The whole finding is that dead
#      text costs 1.1% at t=1; if the pad ran, it would be measuring work. A
#      `panic!` planted in `text_pad::unit` must leave the corpus green.
#   4. the CDEF doubling arm must file EXACTLY the registrations it claims —
#      removing one of the five `dup_rows` call sites must drop the count by
#      that site's population and no other number.
#
# Every mutation is restored from a backup COPY (never `git checkout --`) and
# verified byte-exact by sha256 AND `git diff --exit-code`.
#
# Usage: rectship_teeth.sh [outdir]
set -u
OUT=${1:-$HOME/tmp/rectship/teeth}
mkdir -p "$OUT/bak"
cd "$(dirname "$0")/../.."
VEC=$HOME/tmp/lfg/stage/avif/c256x2048.avif
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
  if [ -n "$feat" ]; then
    nice -n 19 cargo build --release --example bench_ab_decode --features "$feat" \
      --target-dir "$OUT/tgt" > "$OUT/build.log" 2>&1 || { echo BUILDFAIL; return; }
  else
    nice -n 19 cargo build --release --example bench_ab_decode \
      --target-dir "$OUT/tgt" > "$OUT/build.log" 2>&1 || { echo BUILDFAIL; return; }
  fi
  nice -n 19 "$OUT/tgt/release/examples/bench_ab_decode" "$VEC" "$t" 2 1 teeth 2>&1 \
    | awk -F'\t' '$1=="CHECKSUM"{print $5}' | head -1
}

echo "== control ==" >&2
REF=$(md5_of "" 8); note control_default_t8 "$REF"
REFR=$(md5_of "__lf_rect" 8); note control_rect_t8 "$REFR"
[ "$REF" = "$REFR" ] || note control_arms_agree MISMATCH

echo "== 1. default per-row path: off-by-one row ==" >&2
backup src/loopfilter.rs > "$OUT/sha_lf_before.txt"
python3 - <<'PY'
p='src/loopfilter.rs'; s=open(p).read()
# The same line also appears in `fill`'s non-const-W dispatch arm, so anchor on
# the comment that follows it inside `fill_threaded`.
old = ("            let off = origin.offset.wrapping_add_signed(row as isize * stride);\n"
       "            // The MARGINAL price of one filter-chain registration")
assert s.count(old) == 1, s.count(old)
new = ("            let off = origin.offset.wrapping_add_signed((row + 1) as isize * stride);\n"
       "            // The MARGINAL price of one filter-chain registration")
open(p,'w').write(s.replace(old, new, 1))
PY
M=$(md5_of "" 8)
[ "$M" != "$REF" ] && note mut_default_row_plus_1 "CAUGHT ($M)" || note mut_default_row_plus_1 "NOT CAUGHT"
restore src/loopfilter.rs > "$OUT/sha_lf_after1.txt"

echo "== 2. rectangle path: rows reversed ==" >&2
backup src/loopfilter.rs > /dev/null
python3 - <<'PY'
p='src/loopfilter.rs'; s=open(p).read()
old = "            let src: &[BD::Pixel; W] = rect.row(row).try_into().expect(\"row is W long\");"
assert s.count(old) == 1, s.count(old)
# `row + 1` would trip `DisjointImmutRectGuard::row`'s own bounds assert and
# report as a panic rather than as wrong pixels; reversing the rows stays in
# range and makes the corpus/md5 the thing that catches it.
open(p,'w').write(s.replace(old, "            let src: &[BD::Pixel; W] = rect.row(h - 1 - row).try_into().expect(\"row is W long\");", 1))
PY
M=$(md5_of "__lf_rect" 8)
[ "$M" != "$REFR" ] && note mut_rect_rows_reversed "CAUGHT ($M)" || note mut_rect_rows_reversed "NOT CAUGHT"
restore src/loopfilter.rs > "$OUT/sha_lf_after2.txt"

echo "== 3. the layout pad must NEVER EXECUTE ==" >&2
backup src/loopfilter.rs > /dev/null
python3 - <<'PY'
p='src/loopfilter.rs'; s=open(p).read()
old = "    pub(crate) extern \"C\" fn unit<const K: usize>(x: &mut [u64; 32]) -> u64 {\n        let mut acc = K as u64;"
assert s.count(old) == 1, s.count(old)
new = "    pub(crate) extern \"C\" fn unit<const K: usize>(x: &mut [u64; 32]) -> u64 {\n        panic!(\"layout pad executed\");\n        #[allow(unreachable_code)]\n        let mut acc = K as u64;"
open(p,'w').write(s.replace(old, new, 1))
PY
M=$(md5_of "__pad4" 8)
[ "$M" = "$REF" ] && note pad_never_executes "CONFIRMED (md5 unchanged with panic! planted)" \
                  || note pad_never_executes "PAD RAN OR BUILD BROKE ($M)"
restore src/loopfilter.rs > "$OUT/sha_lf_after3.txt"

echo "== 4. CDEF doubling arm files exactly what it claims ==" >&2
nice -n 19 cargo build --release --example probe_tracker \
  --features "probe-sites,__probe_cdef_double" --target-dir "$OUT/tgt2" \
  > "$OUT/build_ps.log" 2>&1
count() { RAV1D_CDEF_DOUBLE=$1 nice -n 19 "$OUT/tgt2/release/examples/probe_tracker" \
  "$VEC" 8 3 2>&1 | awk -F'total_per_frame=' '/^SITES/{split($2,a," ");print a[1];exit}'; }
C0=$(count 0); C1=$(count 1); note cdef_counts "off=$C0 on=$C1 delta=$((C1-C0))"
backup src/safe_simd/cdef_arm.rs > /dev/null
python3 - <<'PY'
p='src/safe_simd/cdef_arm.rs'; s=open(p).read()
old = "    img.dup_rows::<BitDepth8>(8, 8);\n"
assert s.count(old) == 1
open(p,'w').write(s.replace(old, "", 1))
PY
nice -n 19 cargo build --release --example probe_tracker \
  --features "probe-sites,__probe_cdef_double" --target-dir "$OUT/tgt2" \
  > "$OUT/build_ps2.log" 2>&1
C1M=$(count 1)
note cdef_mut_drop_one_site "on=$C1M delta_vs_full=$((C1M-C1)) (expect the cdef_find_dir site's population)"
restore src/safe_simd/cdef_arm.rs > "$OUT/sha_cdef_after.txt"

echo "== 5. forbid(unsafe_code) proven ACTIVE ==" >&2
backup src/picture.rs > /dev/null
python3 - <<'PY'
p='src/picture.rs'; s=open(p).read()
open(p,'w').write(s + "\nfn _teeth_unsafe_probe(x: u64) -> f64 { unsafe { core::mem::transmute(x) } }\n")
PY
if nice -n 19 cargo check --release --target-dir "$OUT/tgt3" > "$OUT/unsafe.log" 2>&1; then
  note forbid_unsafe_active "NOT ENFORCED — build succeeded"
else
  if rg -q "lib.rs:13:12" "$OUT/unsafe.log"; then
    note forbid_unsafe_active "ENFORCED at lib.rs:13:12"
  else
    note forbid_unsafe_active "build failed but not at the forbid anchor (see log)"
  fi
fi
restore src/picture.rs > "$OUT/sha_pic_after.txt"

echo "== worktree clean? ==" >&2
git diff --stat -- src include lib.rs | tail -3
cat "$OUT/summary.tsv" >&2
