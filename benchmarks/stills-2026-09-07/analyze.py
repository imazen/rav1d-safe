"""Rebuild descriptive profile buckets and borrowing counts from raw evidence."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

p = argparse.ArgumentParser()
p.add_argument('work_dir', type=Path)
a = p.parse_args()
root = a.work_dir
profiles = []
for path in sorted((root / 'profiles').glob('*.txt')):
    buckets = dict(coefficient_entropy=0, transforms=0, tracker=0, spin=0,
                   loopfilter=0, memory_copy=0, anonymous=0)
    for line in path.read_text().splitlines():
        match = re.match(r'\s*([0-9.]+)%\s*;(.+?);', line)
        if not match:
            continue
        percent, symbol = float(match[1]), match[2]
        if 'decode_coefs' in symbol or 'msac' in symbol:
            buckets['coefficient_entropy'] += percent
        if any(s in symbol for s in ['::itx::', '::itx_1d::', 'inv_txfm', 'inv_dct', 'inv_adst']):
            buckets['transforms'] += percent
        if 'rav1d_disjoint_mut' in symbol:
            buckets['tracker'] += percent
        if 'spin_loop' in symbol or 'lock_slow' in symbol:
            buckets['spin'] += percent
        if 'loopfilter' in symbol or 'lpf_' in symbol:
            buckets['loopfilter'] += percent
        if 'memcpy' in symbol or 'memmove' in symbol:
            buckets['memory_copy'] += percent
        if '..@' in symbol or '[.] 0x' in symbol:
            buckets['anonymous'] += percent
    # These are name-based lower bounds. ASM local labels are not attributed;
    # spin may overlap tracker. Do not sum the buckets or compare a tiny named
    # ASM transform bucket with complete Rust transform attribution.
    profiles.append(dict(profile=path.stem, **{k: round(v, 2) for k, v in buckets.items()}))
(root / 'profiles/buckets.json').write_text(json.dumps(profiles, indent=2) + '\n')

probes = []
for path in sorted((root / 'probes').glob('*-t*.log')):
    counts, bysite = defaultdict(int), defaultdict(int)
    occupancy, stages = {}, {}
    lifetime = None
    for line in path.read_text().splitlines():
        fields = line.split('\t')
        if fields[0] == 'BORROW':
            n = int(fields[7])
            counts['borrows'] += n
            counts['mutable' if fields[2] == 'true' else 'shared'] += n
            counts['registered_bytes'] += int(fields[8])
            bysite[fields[1]] += n
        elif fields[0] == 'OCCUPANCY':
            occupancy[int(fields[1])] = int(fields[2])
        elif fields[0] == 'VALIDATED':
            lifetime = int(fields[2].split('=')[1])
        elif line.startswith('PROBE stage_ms_per_frame '):
            words = line.split()
            stages[words[2]] = float(words[3])
    assert lifetime
    probes.append(dict(case=path.stem, lifetime_frames=lifetime,
        mean_per_lifetime_frame={k: round(v/lifetime, 1) for k, v in counts.items()},
        occupancy_attempts=occupancy,
        top_borrow_sites=[(k, round(v/lifetime, 1)) for k, v in
                          sorted(bysite.items(), key=lambda i: -i[1])[:8]],
        instrumented_stage_ms=stages))
(root / 'probes/summary.json').write_text(json.dumps(probes, indent=2) + '\n')
