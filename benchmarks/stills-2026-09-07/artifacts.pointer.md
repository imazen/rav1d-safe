# Still-performance artifacts

Working artifact root on r7900x:
`/home/lilith/tmp/rav1d-stills-2026-09-07`.

`/mnt/v` and the Tower mount are unavailable on this machine. Sources, Y4M,
encoded IVF, and raw `profiles/*.data` are retained in this scratch directory;
they have **not** been mirrored to R2 or Tower. Do not delete this directory
before transferring any evidence that must be retained. This is a storage
limitation, not a claim that a backup exists.

- `sources.json` in this benchmark directory pins the public source URLs and
  hashes. Recreate input Y4M/IVF with `prepare.py`, system Pillow 12.1.1, and
  libaom aomenc 3.13.1. Encodes are a development corpus, not a holdout.
- `results/large-artifacts.json.gz` records every IVF/Y4M/perf.data size and
  SHA-256, relative to the scratch root.
- `results/*provenance.json.gz` pins the benchmark binaries. The baseline
  checked/unchecked/ASM binaries remain in
  `/home/lilith/tmp/rav1d-perf-solution-2026-09-07/bin/solution-*`.
- Upstream binary:
  `/home/lilith/tmp/rav1d-release-perf-2026-09-07/bin/upstream`, source commit
  `d3d1cd67059f47803919be8276650e5870c9fd02`.
- `probe-driver/` and `bin/still-probe` are the separate diagnostic consumer;
  `results/probes-build.json.gz` records its command and binary hash.
- Compressed raw text, summaries, encoder logs, profile reports, census logs,
  and validation records are committed in `results/`, independently of the
  scratch copy. `results/SHA256.json.gz` checksums those tracked artifacts.

Reproduction commands and measurement boundaries are in [README.md](README.md).
