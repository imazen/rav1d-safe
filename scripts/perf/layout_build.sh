#!/usr/bin/env bash
# Build the code-placement grid: {no alignment, 16B, 32B, 64B function
# alignment} x {no pad, +4.8 KB, +9.7 KB, +19.4 KB of provably-dead text}.
#
# The question this grid answers is NOT "is an aligned binary faster" but
# "does whole-function alignment SHRINK THE SPREAD across the pad rungs" --
# i.e. can the +-1.5% code-placement lottery documented in docs/RECT_SHIP.md
# be removed. So every family gets the same four rungs and the report reduces
# each family to a spread.
#
# One target dir per alignment family: RUSTFLAGS is part of the fingerprint, so
# sharing one dir would rebuild the whole dep graph on every family switch.
# Feature changes inside a family only rebuild the root crate.
#
# NO -C target-cpu=native. Builds are niced (they are not timed).
set -eu
R=${R:-$HOME/work/zen/rav1d-safe--layout}
OUT=${OUT:-$HOME/tmp/layout}
FAMS=${FAMS:-"a0 a4 a5 a6"}
RUNGS=${RUNGS:-"plain pad1 pad2 pad4"}
mkdir -p "$OUT/bin" "$OUT/logs" "$OUT/tgt"

flags_for() {
  case "$1" in
    a0) echo "" ;;
    a4) echo "-C llvm-args=-align-all-functions=4" ;;
    a5) echo "-C llvm-args=-align-all-functions=5" ;;
    a6) echo "-C llvm-args=-align-all-functions=6" ;;
    b5) echo "-C llvm-args=-align-all-nofallthru-blocks=5" ;;
    b6) echo "-C llvm-args=-align-all-nofallthru-blocks=6" ;;
    a5b5) echo "-C llvm-args=-align-all-functions=5 -C llvm-args=-align-all-nofallthru-blocks=5" ;;
    *) echo "UNKNOWN-FAMILY" ; exit 2 ;;
  esac
}
feat_for() {
  case "$1" in
    plain) echo "" ;;
    pad1)  echo "__pad_text" ;;
    pad2)  echo "__pad2" ;;
    pad3)  echo "__pad3" ;;
    pad4)  echo "__pad4" ;;
    *) echo "UNKNOWN-RUNG" ; exit 2 ;;
  esac
}

cd "$R"
for fam in $FAMS; do
  rf=$(flags_for "$fam")
  for rung in $RUNGS; do
    feat=$(feat_for "$rung")
    name="$fam$rung"
    args=(build --release --example bench_ab_decode -j 6 --target-dir "$OUT/tgt/$fam")
    [ -n "$feat" ] && args+=(--features "$feat")
    echo "[$(date +%H:%M:%S)] build $name  RUSTFLAGS='$rf' feat='$feat'" >&2
    RUSTFLAGS="$rf" nice -n 19 cargo "${args[@]}" \
      > "$OUT/logs/build_$name.log" 2>&1
    cp "$OUT/tgt/$fam/release/examples/bench_ab_decode" "$OUT/bin/bench_$name"
    printf '%s\t%s\t%s\n' "$name" \
      "$(shasum -a 256 "$OUT/bin/bench_$name" | cut -c1-16)" \
      "$(stat -f %z "$OUT/bin/bench_$name")"
  done
done
