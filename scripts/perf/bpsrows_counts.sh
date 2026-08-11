#!/usr/bin/env bash
# Timer-free counters for the default-flip, per cell, both arms.
#
# Three questions a clock cannot answer:
#
#  1. **What shift does each plane actually get?** (`__probe_bounds`, the
#     `shifts` column) — the liveness proof that the rule is doing anything, per
#     cell, read off the tracker rather than predicted.
#  2. **Did the coarser block trade wide-by-shard-count for wide-by-SLOT-
#     EXHAUSTION?** (`probe-wide`, `w_full`) — the standing trap for anything
#     that funnels more simultaneous borrows onto one shard. It is the reason
#     `SLOTS` is not touched, and it has to be re-checked whenever blocks get
#     coarser, on the cells that coarsen MOST.
#  3. **Is the registration COUNT unchanged?** — the knob is supposed to change
#     the COST of a registration, not how many there are. If the count moved,
#     every ns/registration number in the record is comparing two populations.
#
# NICED, no measurement lock: no timer is read.
#
# Usage: bpsrows_counts.sh <outdir> [iters]
set -u
OUT=${1:?outdir}; IT=${2:-20}
BIN=${BIN:-$HOME/tmp/bpsrows/bin}
VEC=${VEC:-$HOME/tmp/bpsrows/vec}
CELLS=${CELLS:-"C1024x192_420_8b__t8 C1024x384_420_8b__t8 C3840x256_420_8b__t8 C1024x576_420_8b__t8 C256x2048_420_8b__t8 C512x576_420_8b__t8 L1024x576_420_10b__t8 v4k_8tile"}
mkdir -p "$OUT"

wide="$OUT/wide.tsv"
{ printf 'cell\tarm\tthreads\tslow\tmulti\tw_shards\tw_blocks\tw_full\n'; } > "$wide"
shifts="$OUT/shifts.tsv"
{ printf 'cell\tarm\tsite\tshifts\trow_shards_mean\trow_shards_max\tpct_row_wide\n'; } > "$shifts"
regs="$OUT/regs.tsv"
{ printf 'cell\tarm\tthreads\tregs_per_frame\tdistinct_sites\n'; } > "$regs"

for c in $CELLS; do
  for arm in plain bpsblocks; do
    for t in 8 1; do
      r=$(nice -n 19 "$BIN/pt_${arm}__probewide" "$VEC/$c.avif" "$t" "$IT" 2>/dev/null \
          | awk -F'\t' '/^WIDE\t/{print $3"\t"$4"\t"$5"\t"$6"\t"$7}')
      printf '%s\t%s\t%s\t%s\n' "$c" "$arm" "$t" "$r" >> "$wide"
    done
    out=$(nice -n 19 "$BIN/pt_${arm}__probebounds" "$VEC/$c.avif" 8 4 2>/dev/null)
    n=$(printf '%s' "$out" | awk -F'\t' '/^BOUNDS\t/{split($2,a,"="); print a[2]; exit}')
    d=$(printf '%s' "$out" | awk -F'\t' '/^BOUNDS\t/{split($3,a,"="); print a[2]; exit}')
    printf '%s\t%s\t8\t%s\t%s\n' "$c" "$arm" "${n:-NA}" "${d:-NA}" >> "$regs"
    printf '%s' "$out" | awk -F'\t' -v c="$c" -v a="$arm" \
      '/^RECT\t/{print c"\t"a"\t"$23"\t"$22"\t"$14"\t"$15"\t"$17}' >> "$shifts"
  done
  echo "[$(date +%H:%M:%S)] $c done" >&2
done

echo "--- w_full must be 0 everywhere (slot-exhaustion trap) ---" >&2
awk -F'\t' 'NR>1 && $8+0 != 0 {print "  !! w_full nonzero: "$0; bad=1} END{if(!bad) print "  w_full = 0 on every row"}' "$wide" >&2
echo "--- registration count must match between arms ---" >&2
awk -F'\t' 'NR>1{k=$1; if(k in v){ if(v[k] != $4) print "  !! regs differ on "k": "v[k]" vs "$4; else print "  "k": "$4" (equal)"} else v[k]=$4}' "$regs" >&2
echo "wrote $OUT/{wide,shifts,regs}.tsv" >&2
