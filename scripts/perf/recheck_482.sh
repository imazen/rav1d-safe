#!/usr/bin/env bash
# Settle one question: did #482 (tile-owned intra recon, merged 2fae4fe)
# REGRESS 4K 8bpc?
#
# Two prior measurements disagree at exactly that cell:
#   * #482's own round:      4K 4:4:4 8-tile, t=1, head/base 0.9823, band
#                            [0.9699..1.0218] — the one cell whose arms overlapped.
#   * the size sweep round:  4K 4:2:0 1-tile, t=1, head/base 1.0854 (a 9%
#                            REGRESSION), 4 of 5 rounds above 1.04, band
#                            [0.8821..1.1232] — and every row load-tagged.
# Neither is conclusive. Three arms:
#   parent  = b0a00c3  (#482's first parent on main)
#   head    = 2fae4fe  (#482 merged)
#   main    = 0f6bf10  (current main; #483 has since cut the ReconDst seam tax
#                       from 1.2-3.0% to 0.3-1.3%, so the seam's price today is
#                       not the price #482 shipped at)
#   headoff = the SAME BINARY as head with RAV1D_OWNED_RECON=0
#   mainoff = the SAME BINARY as main with RAV1D_OWNED_RECON=0
#
# The `*off` arms are what make this a root cause and not just a verdict. #482
# is two things at once: a NEW owned-band recon path, and a `ReconDst` enum seam
# that every SHARED-path write now branches through. Disarming the band leaves
# the seam, so within one interleaved sweep:
#   headoff / parent  = the seam tax alone   (#482 measured 1.0115 at 8bpc t=1)
#   head    / headoff = the band alone
#   head    / parent  = what shipped
# Same binary for the armed and disarmed arms, so no inter-arm delta can be a
# codegen artefact (#455's `probe-*` convention).
#
# TWO INSTRUMENTS per cell, deliberately:
#   * external wall of the whole process at two frame counts, fitted
#     `total = a + b*frames`, so exec/mmap/container-parse/decoder-construction
#     drop out of `b`. This is the instrument BOTH disputed measurements used.
#   * the harness's OWN in-process timer (`RESULT` column 8), which brackets
#     exactly the N timed decodes and excludes the warmup decode entirely.
# They are independent. If they disagree, that is the finding, not a nuisance.
#
# LOAD POLICY. The box carries other agents' multi-hour jobs, so this does NOT
# discard-and-retry — that burns wall clock and yields nothing while a `miri`
# run holds a core. Every row is COMMITTED and TAGGED with two counts:
#   f_arm   foreign processes >25% CPU seen right after THIS arm's two runs
#   f_grp   max of f_arm across the whole (round, cell) group
# A paired ratio is only trustworthy if the arms it pairs saw the same box, so
# the analysis keeps groups with f_grp == 0 as the headline and uses the rest
# only as a cross-check. Filtering at analysis time strictly dominates
# discarding at collection time: same clean subset, plus a loaded subset.
#
# NO `nice` on a timed run (Darwin maps niced processes to E-cores, ~40x wall
# distortion). No -C target-cpu=native. Run the whole thing under `measlock`.
#
# Usage: recheck_482.sh <out.tsv> [rounds]
# Env:   BIN, AVIF, IVF, ARMS, CELLS
# Cell syntax:    <vector>:<threads>:<n_lo>:<n_hi>
# Output columns: round arm vec threads nlo ext_lo nhi ext_hi int_lo int_hi f_arm f_grp
set -u
export LC_ALL=C
OUT=${1:?out.tsv}; ROUNDS=${2:-9}
BIN=${BIN:-$HOME/tmp/recheck482/bin}
AVIF=${AVIF:-$HOME/tmp/recheck482/vec}
IVF=${IVF:-$HOME/tmp/recheck482/ivf}
IFS=' ' read -r -a ARMS <<< "${ARMS:-parent head headoff main mainoff}"

# 4K only. The disputed geometry is 4:2:0 single-tile; 4:4:4 single-tile and
# 4:4:4 8-tile are here to separate "subsampling" from "tiling" from "the band"
# if the regression reproduces. t=1 and t=8 on the disputed geometry because
# #482's headline claim was much larger at t=8 (1.873 -> 1.474) than at t=1.
DEFAULT_CELLS="L3840x2160_420_8b:1:2:16 L3840x2160_420_8b:8:2:16"
DEFAULT_CELLS="$DEFAULT_CELLS L3840x2160_420_10b:1:2:16 L3840x2160_420_10b:8:2:16"
DEFAULT_CELLS="$DEFAULT_CELLS L3840x2160_444_8b:1:2:16"
DEFAULT_CELLS="$DEFAULT_CELLS v4k_8tile:1:2:16 v4k_8tile:8:2:16"
IFS=' ' read -r -a CELLS <<< "${CELLS:-$DEFAULT_CELLS}"

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

# Echoes "<external_ms>\t<internal_ms_per_frame>". dav1d has no in-process
# instrument, so its internal column is NA and only the fitted wall is usable.
time_one() {
  local arm=$1 vec=$2 t=$3 n=$4 t0 t1 ipf out bin own
  case "$arm" in
    dav1d_fd1)
      t0=$(now_ms)
      dav1d -i "$IVF/$vec.ivf" --muxer null --threads "$t" --framedelay 1 -q --limit "$n" >/dev/null 2>&1
      t1=$(now_ms); ipf=NA ;;
    *)
      # `<arm>off` is the same binary as `<arm>` with the owned band disarmed.
      # The variable is set EXPLICITLY on both arms (1 or 0) rather than being
      # absent on the armed one, so neither arm depends on a default.
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
    rows=(); fmax=0
    for k in $(seq 0 $((n-1))); do
      # Rotate the arm order every round so no arm always runs first.
      arm=${ARMS[$(( (k + round) % n ))]}
      IFS=$'\t' read -r elo ilo < <(time_one "$arm" "$vec" "$t" "$nlo")
      IFS=$'\t' read -r ehi ihi < <(time_one "$arm" "$vec" "$t" "$nhi")
      f=$(busy_count); [ "$f" -gt "$fmax" ] && fmax=$f
      rows+=("$round	$arm	$vec	$t	$nlo	$elo	$nhi	$ehi	$ilo	$ihi	$f")
    done
    for r in "${rows[@]}"; do printf '%s\t%s\n' "$r" "$fmax" >> "$OUT"; done
    echo "[$(date +%H:%M:%S)] r$round $vec t=$t committed (f_grp=$fmax)" >&2
  done
done
echo "wrote $OUT" >&2
