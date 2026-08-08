#!/usr/bin/env bash
# Rotating-order interleaved A/B of staged bench binaries, one vector per run.
#
# Generalises `bin_ab.sh`: the staging dir, thread count and vector are all
# parameters, and the box-quiet guard EXCLUDES the arms under test. macOS `ps`
# reports a decaying %cpu for a process that has only just exited, so a guard
# that does not exclude `bench_*` sees the arm it just finished and marks every
# cell dirty forever (the P2 campaign livelocked on exactly this).
#
# NO `nice` ON A TIMED RUN — Darwin maps a positive nice value to background QoS
# and distorts wall clock by ~40x. Builds may be niced; this may not.
#
# Usage: st1_ab.sh <out.tsv> <vec.avif> [iters] [rounds] [threads] -- <arm> ...
# Env:   BIN (default ~/tmp/rav1d-st1/bin)
# Output: round  arm  ms_per_frame  md5  foreign
set -u
OUT=${1:?out.tsv}; VEC=${2:?vec}; ITERS=${3:-8}; ROUNDS=${4:-5}; T=${5:-1}
shift 5
[ "${1:-}" = "--" ] && shift
ARMS=("$@")
BIN=${BIN:-$HOME/tmp/rav1d-st1/bin}

BIN_RE=$(printf '%s' "$BIN" | sed 's/[][\.*^$/(){}?+|]/\\&/g')
busy() {
  ps -A -o %cpu,comm -r | awk -v me="$BIN_RE" \
    'NR>1 && $1>25 && $2 !~ /claude|ClaudeCode|versions\/|dav1d/ && $2 !~ me {c++} END {print c+0}'
}

: > "$OUT"
n=${#ARMS[@]}
for r in $(seq 0 $((ROUNDS-1))); do
  for k in $(seq 0 $((n-1))); do
    A=${ARMS[$(( (k + r) % n ))]}
    line=$("$BIN/bench_$A" "$VEC" "$T" "$ITERS" 1 "$A" 2>/dev/null)
    ms=$(printf '%s\n' "$line" | awk -F'\t' '/^RESULT/{print $8}')
    md5=$(printf '%s\n' "$line" | awk -F'\t' '/^CHECKSUM/{print $5}')
    printf '%s\t%s\t%s\t%s\t%s\n' "$r" "$A" "$ms" "$md5" "$(busy)" >> "$OUT"
  done
  echo "[$(date +%H:%M:%S)] round $r done" >&2
done
echo "wrote $OUT" >&2
