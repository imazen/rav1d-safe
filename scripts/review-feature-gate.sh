#!/usr/bin/env bash
# Cargo feature unification must never SILENTLY make a safe constructor
# untracked. `untracked` is the one deliberate, opt-in exception (the decoder's
# internal `__probe_untracked` measurement arm also enables it); this gate pins
# exactly where it can come from.
#
#   1. a default build of either crate must NOT enable it;
#   2. the documented implicit sources must (c-ffi, asm, partial_asm) -- if one
#      stops enabling it, frame threading / the C API change behaviour silently;
#   3. the tracked soundness suite must actually be present in a default build,
#      so a leg that unifies `untracked` in cannot report green with the
#      tracker-dependent tests silently cfg'd out.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

enables_untracked() { # <cargo args...>: does rav1d-disjoint-mut get `untracked`?
  cargo tree -e features -i rav1d-disjoint-mut "$@" 2>/dev/null | grep -Fq 'feature "untracked"'
}

if enables_untracked -p rav1d-safe; then
  echo "FAIL: a default rav1d-safe build enables rav1d-disjoint-mut/untracked" >&2; exit 1
fi
if enables_untracked -p rav1d-disjoint-mut; then
  echo "FAIL: a default rav1d-disjoint-mut build enables untracked" >&2; exit 1
fi
echo "PASS: default builds do not enable untracked"

for f in c-ffi asm partial_asm untracked; do
  if ! enables_untracked -p rav1d-safe --no-default-features --features "$f"; then
    echo "FAIL: rav1d-safe/$f no longer enables untracked (documented in docs/UNTRACKED_MODE.md)" >&2; exit 1
  fi
  echo "PASS: rav1d-safe/$f enables untracked (documented)"
done

if ! cargo test -p rav1d-disjoint-mut --test soundness -- --list 2>/dev/null | grep -Fq 'test_overlapping_mut_panics'; then
  echo "FAIL: the tracked soundness suite is not present in a default build" >&2; exit 1
fi
echo "PASS: tracked soundness suite present"
