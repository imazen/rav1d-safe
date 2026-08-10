#!/usr/bin/env bash
# Counterbalanced (mirrored) confirmation of the #482 4K verdict.
#
# WHY THIS EXISTS. The campaign's standard rotation runs the arms once per
# round and advances the starting arm by one each round. With N arms that puts
# any given PAIR adjacent in N-1 of every N rounds and maximally separated in
# the Nth — so if execution position inside a group carries a cost (it does at
# 4K 4:2:0 8bpc t=1; see the position table in recheck_482_report.py), the pair
# inherits a bias that rotation does NOT average away, because the two
# orderings are not visited equally often.
#
# The fix is a mirrored order: A,B,...,N,N,...,B,A. Every arm occupies a
# position and its mirror, so the position indices each arm sees sum to the
# same constant. Any drift that is monotone across the group — cache warmth,
# frequency ramp, a neighbour's job starting — cancels in every pair at once,
# by construction rather than by averaging. The two runs of an arm are reduced
# by taking the MINIMUM (the least-disturbed observation of a fixed quantity),
# which is the standard estimator when noise is one-sided-positive.
#
# NO `nice` on a timed run. No -C target-cpu=native. Run under `measlock`.
#
# Usage: recheck_482_mirror.sh <out.tsv> [rounds]
# Output columns: round arm vec threads nlo ext_lo nhi ext_hi int_lo int_hi f_arm f_grp pass
set -u
export LC_ALL=C
OUT=${1:?out.tsv}; ROUNDS=${2:-15}
BIN=${BIN:-$HOME/tmp/recheck482/bin}
AVIF=${AVIF:-$HOME/tmp/recheck482/vec}
IVF=${IVF:-$HOME/tmp/recheck482/ivf}
IFS=' ' read -r -a ARMS <<< "${ARMS:-parent head szrs szrs2}"
IFS=' ' read -r -a CELLS <<< "${CELLS:-L3840x2160_420_8b:1:2:16 L3840x2160_420_8b:8:2:16}"

BIN_RE=$(printf '%s' "$BIN" | sed 's/[][\.*^$/(){}?+|]/\\&/g')
busy_count() {
  ps -A -o %cpu,comm -r | awk -v me="$BIN_RE" \
    'NR>1 && $1>25 && $2 !~ /claude|ClaudeCode|versions\/|dav1d|python3/ && $2 !~ me {c++} END {print c+0}'
}
if [ -n "${EPOCHREALTIME:-}" ]; then
  now_ms() { local t=$EPOCHREALTIME; echo $(( ${t%%.*} * 1000 + 10#${t#*.} / 1000 )); }
else
  now_ms() { python3 -c 'import time;print(int(time.time()*1000))'; }
fi

time_one() {
  local arm=$1 vec=$2 t=$3 n=$4 t0 t1 ipf out bin own
  case "$arm" in
    dav1d_fd1)
      t0=$(now_ms)
      dav1d -i "$IVF/$vec.ivf" --muxer null --threads "$t" --framedelay 1 -q --limit "$n" >/dev/null 2>&1
      t1=$(now_ms); ipf=NA ;;
    *)
      bin=$arm; own=1
      case "$arm" in *off) bin=${arm%off}; own=0 ;; esac
      t0=$(now_ms)
      out=$(RAV1D_OWNED_RECON=$own "$BIN/bench_$bin" "$AVIF/$vec.avif" "$t" "$n" 1 w 2>/dev/null)
      t1=$(now_ms)
      ipf=$(printf '%s\n' "$out" | awk -F'\t' '/^RESULT/{print $8}') ;;
  esac
  printf '%s\t%s' "$((t1 - t0))" "${ipf:-NA}"
}

: > "$OUT"
n=${#ARMS[@]}
for round in $(seq 0 $((ROUNDS-1))); do
  for cell in "${CELLS[@]}"; do
    IFS=: read -r vec t nlo nhi <<< "$cell"
    # Forward pass then reverse pass. The starting arm still rotates per round
    # so the mirror axis moves too, but the cancellation does not depend on it.
    order=()
    for k in $(seq 0 $((n-1))); do order+=("${ARMS[$(( (k + round) % n ))]}"); done
    for k in $(seq $((n-1)) -1 0);  do order+=("${ARMS[$(( (k + round) % n ))]}"); done
    rows=(); passes=(); fmax=0; pass=0
    for arm in "${order[@]}"; do
      IFS=$'\t' read -r elo ilo < <(time_one "$arm" "$vec" "$t" "$nlo")
      IFS=$'\t' read -r ehi ihi < <(time_one "$arm" "$vec" "$t" "$nhi")
      f=$(busy_count); [ "$f" -gt "$fmax" ] && fmax=$f
      # Row stops at f_arm (col 11); f_grp (12) and pass (13) are appended
      # below, once fmax is known for the whole group.
      rows+=("$round	$arm	$vec	$t	$nlo	$elo	$nhi	$ehi	$ilo	$ihi	$f")
      passes+=("$pass")
      pass=$((pass+1))
    done
    for i in "${!rows[@]}"; do
      printf '%s\t%s\t%s\n' "${rows[$i]}" "$fmax" "${passes[$i]}" >> "$OUT"
    done
    echo "[$(date +%H:%M:%S)] r$round $vec t=$t committed (f_grp=$fmax, ${#order[@]} runs)" >&2
  done
done
echo "wrote $OUT" >&2
