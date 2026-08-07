#!/usr/bin/env bash
# SIMD-tier spread, both decoders, same vectors, threads=1.
#
#   rs_native   rav1d-safe, CpuLevel::Native
#   rs_scalar   rav1d-safe, CpuLevel::Scalar  (flag-gated DSP tables only — the
#               unconditional Arm64::summon() sites in mc_arm/filmgrain_arm stay
#               NEON regardless, so this is a PARTIAL scalar, an upper bound on
#               the scalar time and therefore a LOWER bound on the SIMD win)
#   dav1d_neon  dav1d 1.5.4, default cpumask
#   dav1d_scal  dav1d 1.5.4, --cpumask 0   (true scalar C)
#
# Same interleaving/rotation and busy-box discipline as yardstick_sweep.sh.
set -u
OUT=${1:?out.tsv}
ROUNDS=${2:-3}
BENCH=$HOME/tmp/recon-yard/target-main/release/examples/bench_ab_decode
AVIF=$HOME/tmp/rav1d-perf/vec
IVF=$HOME/tmp/recon-yard/vec
FT=$HOME/tmp/recon-yard/ft_tier.tmp

wait_for_quiet() {
  local waited=0
  while [ "$(ps -A -o %cpu,comm -r | awk 'NR>1 && $1>25 && $2 !~ /claude|ClaudeCode|versions\// {c++} END {print c+0}')" -gt 0 ]; do
    [ $waited -eq 0 ] && echo "[$(date +%H:%M:%S)] box busy, pausing..." >&2
    sleep 15; waited=$((waited + 15))
    [ $waited -ge 5400 ] && { echo "box still busy; refusing" >&2; exit 4; }
  done
  return 0
}
wait_for_quiet

# vector : rs iters (native) : rs iters (scalar, slower) : dav1d limit : dav1d scalar limit
CELLS=(
  "v1024:20:6:60:20"
  "v1024_10b:20:6:60:20"
  "v4k_1tile:2:1:12:4"
)
ARMS=(rs_native rs_scalar dav1d_neon dav1d_scal)
T=1
REPS=3

run_dav1d() { # vec limit cpumask label round
  local vec=$1 lim=$2 mask=$3 label=$4 round=$5
  rm -f "$FT"
  local extra=()
  [ "$mask" != "-" ] && extra=(--cpumask "$mask")
  dav1d -i "$IVF/$vec.ivf" --muxer null --threads "$T" --framedelay 1 \
        "${extra[@]}" -q --limit "$lim" --frametimes "$FT" >/dev/null 2>&1
  local rc=$?
  if [ $rc -ne 0 ] || [ ! -s "$FT" ]; then
    printf '%s\tFAIL\t%s\t%s\t%s\trc=%s\n' "$round" "$label" "$vec" "$T" "$rc" >> "$OUT"; return
  fi
  awk -v r="$round" -v l="$label" -v v="$vec" -v t="$T" \
    'NR>3 {printf "%s\tRESULT\t%s\t%s\t%s\t%d\t1\t%.6f\t%.6f\n", r,l,v,t,NR-4,$1/1e6,$1/1e6}' \
    "$FT" >> "$OUT"
}

: > "$OUT"
for round in $(seq 0 $((ROUNDS - 1))); do
  for cell in "${CELLS[@]}"; do
    IFS=: read -r vec it_n it_s lim_n lim_s <<< "$cell"
    wait_for_quiet
    n=${#ARMS[@]}
    for k in $(seq 0 $((n - 1))); do
      arm=${ARMS[$(( (k + round) % n ))]}
      case "$arm" in
        rs_native)
          out=$("$BENCH" "$AVIF/$vec.avif" "$T" "$it_n" "$REPS" rs_native native 2>&1) ;;
        rs_scalar)
          out=$("$BENCH" "$AVIF/$vec.avif" "$T" "$it_s" "$REPS" rs_scalar scalar 2>&1) ;;
        dav1d_neon) run_dav1d "$vec" "$lim_n" - dav1d_neon "$round"; out="" ;;
        dav1d_scal) run_dav1d "$vec" "$lim_s" 0 dav1d_scal "$round"; out="" ;;
      esac
      if [ -n "$out" ]; then
        printf '%s\n' "$out" | grep -E '^(RESULT|CHECKSUM|GEOM)' | sed "s/^/$round\t/" >> "$OUT"
      fi
      printf 'round=%s %-14s %-11s done\n' "$round" "$vec" "$arm" >&2
    done
  done
done
echo "wrote $OUT" >&2
