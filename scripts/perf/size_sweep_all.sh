#!/usr/bin/env bash
# Regenerate every table in docs/SIZE_SWEEP.md from the raw TSVs, so the record
# is reproducible from the committed data rather than from a transcript.
# Usage: size_sweep_all.sh <raw-dir> <out-dir>
set -u
RAW=${1:-$HOME/tmp/szsweep}; OUT=${2:?out-dir}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
D=2026-08-10

python3 "$HERE/size_sweep_report.py" "$RAW/size_gap.tsv" \
  --tsv "$OUT/size_sweep_cells_$D.tsv" > "$OUT/size_sweep_report_$D.txt"
cp "$RAW/size_gap.tsv" "$OUT/size_sweep_gap_$D.tsv"
cp "$RAW/vector_md5_crosscheck.tsv" "$OUT/size_sweep_vector_md5_$D.tsv" 2>/dev/null

# The depth table: our 8->10 bpc penalty against dav1d's, per size.
python3 - "$RAW/size_gap.tsv" > "$OUT/size_sweep_depth_$D.tsv" <<'PY'
import sys
from collections import defaultdict
rows = []
for line in open(sys.argv[1]):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 9:
        rows.append((int(p[0]), p[1], p[2], int(p[4]), float(p[5]), int(p[6]), float(p[7])))
per = defaultdict(list)
for rd, arm, vec, nlo, lo, nhi, hi in rows:
    per[(arm, vec, rd)].append((hi - lo) / (nhi - nlo))
def med(xs):
    s = sorted(xs); n = len(s)
    return s[n//2] if n % 2 else 0.5*(s[n//2-1]+s[n//2])
cell = {k: med(v) for k, v in per.items()}
rounds = sorted({r for (_, _, r) in cell})
print("fmt\tsize\tours_8b_ms\tours_10b_ms\tdav1d_8b_ms\tdav1d_10b_ms\t"
      "ours_depth_pct\tdav1d_depth_pct\texcess_ms\tratio_8b\tratio_10b")
for fmt in ("420", "444"):
    for wh in ("64x36", "256x144", "512x288", "1024x576", "2048x1152", "3840x2160"):
        def m(a, d):
            xs = [cell[(a, f"L{wh}_{fmt}_{d}", r)] for r in rounds
                  if (a, f"L{wh}_{fmt}_{d}", r) in cell]
            return med(xs) if xs else float("nan")
        o8, o10 = m("rs", "8b"), m("rs", "10b")
        d8, d10 = m("dav1d_fd1", "8b"), m("dav1d_fd1", "10b")
        print(f"{fmt}\t{wh}\t{o8:.4f}\t{o10:.4f}\t{d8:.4f}\t{d10:.4f}\t"
              f"{100*(o10/o8-1):.1f}\t{100*(d10/d8-1):.1f}\t{(o10-o8)-(d10-d8):.3f}\t"
              f"{o8/d8:.3f}\t{o10/d10:.3f}")
PY

for f in "$RAW/p23/conc_1024_8b.tsv" "$RAW/p23/conc_1024_10b.tsv"; do
  [ -s "$f" ] || continue
  b=$(basename "$f" .tsv)
  python3 "$HERE/concurrent_throughput_report.py" "$f" > "$OUT/size_sweep_${b}_$D.txt"
  cp "$f" "$OUT/size_sweep_${b}_$D.tsv"
done
for f in "$RAW/p23/anchor_gap.tsv" "$RAW/p23/t8_gap.tsv"; do
  [ -s "$f" ] || continue
  b=$(basename "$f" .tsv)
  python3 "$HERE/size_sweep_report.py" "$f" > "$OUT/size_sweep_${b}_$D.txt" 2>/dev/null
  cp "$f" "$OUT/size_sweep_${b}_$D.tsv"
done
echo "wrote tables to $OUT" >&2
