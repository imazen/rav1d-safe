#!/usr/bin/env bash
# Cargo feature unification must never make a safe constructor unchecked.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
log_root="${TMPDIR:-${RUNNER_TEMP:-$HOME/tmp}}"
mkdir -p "$log_root"
for feature in __probe_untracked __probe_noscan __probe_lockonly __probe_tinynop __probe_addnop; do
  log=$(mktemp "$log_root/disjoint-feature-gate-XXXXXXXX.log")
  if cargo check -p rav1d-disjoint-mut --features "$feature" >"$log" 2>&1; then
    echo "FAIL: $feature compiled and can weaken safe access; log: $log" >&2
    exit 1
  fi
  # A network failure or unrelated compiler error is not a successful gate.
  if ! rg -q 'unsound measurement probes are disabled' "$log"; then
    cat "$log" >&2
    exit 1
  fi
  echo "PASS: $feature rejected by the safety boundary"
done
