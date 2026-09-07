"""Group timer-only perf self percentages; retain every reported symbol."""
import argparse
import hashlib
import json
from pathlib import Path
import re

p = argparse.ArgumentParser()
p.add_argument('--work-dir', type=Path, required=True)
p.add_argument('--name', default='wire16')
p.add_argument('--profile-dir', default='profiles')
a = p.parse_args()
root = a.work_dir.resolve()
records = []
for path in sorted((root / a.profile_dir).glob('*.txt')):
    groups = dict(entropy=0.0, transforms=0.0, loopfilter=0.0, tracker=0.0,
                  anonymous_asm=0.0, other=0.0)
    symbols = []
    for line in path.read_text().splitlines():
        parts = line.split(';')
        if len(parts) < 2 or not re.fullmatch(r'\s*[0-9.]+%\s*', parts[0]):
            continue
        percent = float(parts[0].strip().removesuffix('%'))
        name = parts[1].strip().removeprefix('[.] ').strip()
        if 'decode_coefs' in name or 'msac' in name:
            group = 'entropy'
        elif ('::itx' in name or 'inv_txfm' in name
              or re.search(r'inv_(dct|adst|identity|wht)|dav1d_i(dct|adst|flipadst|identity|wht)', name)):
            group = 'transforms'
        elif 'loopfilter' in name or 'loop_filter' in name or 'lpf_' in name:
            group = 'loopfilter'
        elif 'rav1d_disjoint_mut::' in name or 'BorrowTracker' in name:
            group = 'tracker'
        elif name.startswith('..@'):
            group = 'anonymous_asm'
        else:
            group = 'other'
        groups[group] += percent
        symbols.append(dict(symbol=name, self_percent=percent, group=group))
    assert symbols, path
    records.append(dict(profile=path.name, report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        reported_self_percent=sum(groups.values()),
                        groups={k: round(v, 2) for k, v in groups.items()}, symbols=symbols))
result = dict(method='Summed perf report --no-children self cycle percentages by symbol. '
                    'Compiler inlining affects attribution. These are diagnostic proportions, '
                    'not speed estimates or absolute per-stage latency; reported symbols below '
                    'the 0.05% report threshold are absent. Anonymous NASM labels remain '
                    'unattributed; upstream group totals are incomplete and cannot establish '
                    'stage-level slowdown ratios. Group rules are in profile_summary.py.',
              profile_directory=a.profile_dir, profiles=records)
(root / (a.name + '-profile-summary.json')).write_text(json.dumps(result, indent=2) + '\n')
for r in records:
    print(r['profile'], r['groups'])
