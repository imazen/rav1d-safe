#!/usr/bin/env bash
# Registration census over the content-class corpus, both arms, both thread
# counts. Counts only — this build's tracker is instrumented, so NO wall-clock
# number from it is valid (that is what scripts/perf/bin_ab.sh is for).
#
# Emits two TSVs:
#   <out>.tot.tsv   arm  vec  threads  total_per_frame  distinct_sites
#   <out>.site.tsv  arm  vec  threads  per_frame  mut  immut  mean_bytes  site
#
# Per-site rows are only meaningful on an arm built with the `probe-sites`
# `track_caller` on `CaseSetter::set_disjoint`; without it every `CaseSet` call
# site collapses onto `ctx.rs:99:27`.
#
# Usage: ctx_census.sh <outprefix> [iters]
# Env: BIN (staged probe_<arm> binaries), VEC (avif dir), ARMS, CELLS
set -u
OUT=${1:?outprefix}; ITERS=${2:-3}
BIN=${BIN:-$HOME/tmp/ctxtl/bin}
VEC=${VEC:-$HOME/tmp/ctxtl/vec}
IFS=' ' read -r -a ARMS <<< "${ARMS:-base head}"
IFS=' ' read -r -a CELLS <<< "${CELLS:-Cui_1024x576_q20:1 Cui_1024x576_q20:8 Cui_1024x576_q70:1 Ctext_1024x576_q20:1 Ctext_1024x576_q20:8 Ctext_1024x576_q70:1 Cphoto_1024x576_q20:1 Cphoto_1024x576_q20:8 Cphoto_1024x576_q70:1}"

printf 'arm\tvec\tthreads\ttotal_per_frame\tdistinct\n' > "$OUT.tot.tsv"
printf 'arm\tvec\tthreads\tper_frame\tmut\timmut\tmean_bytes\tsite\n' > "$OUT.site.tsv"
for cell in "${CELLS[@]}"; do
  IFS=: read -r vec t <<< "$cell"
  for arm in "${ARMS[@]}"; do
    raw=$(nice -n 19 "$BIN/probe_$arm" "$VEC/$vec.avif" "$t" "$ITERS" 2>/dev/null)
    printf '%s\t%s\t%s\t%s\n' "$arm" "$vec" "$t" \
      "$(printf '%s\n' "$raw" | awk -F'[\t=]' '/^SITES/{print $3"\t"$5}')" >> "$OUT.tot.tsv"
    printf '%s\n' "$raw" | awk -F'\t' -v a="$arm" -v v="$vec" -v t="$t" \
      '/^SITE\t/{print a"\t"v"\t"t"\t"$2"\t"$3"\t"$4"\t"$5"\t"$7}' >> "$OUT.site.tsv"
  done
  echo "[$(date +%H:%M:%S)] $vec t=$t done" >&2
done
echo "wrote $OUT.tot.tsv $OUT.site.tsv" >&2
