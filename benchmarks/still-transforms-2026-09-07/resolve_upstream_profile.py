"""Resolve anonymous NASM labels through the pinned ELF symbol table.

Zero-sized NASM FUNC symbols use the next FUNC in the same section as an
inferred end. Ambiguous names whose instances imply different stage groups
remain unattributed. This refines existing profiles; it runs no decoder.
"""
import argparse
from bisect import bisect_right
import hashlib
import json
from pathlib import Path
import re
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--work-dir', type=Path, required=True)
p.add_argument('--upstream', type=Path, required=True)
p.add_argument('--name', default='wire16', help='Prefix of confirm provenance and profile summary')
a = p.parse_args()
root = a.work_dir.resolve()
binary = a.upstream.resolve()
provenance = json.loads((root / (a.name + '-confirm-provenance.json')).read_text())
source = json.loads((root / (a.name + '-profile-summary.json')).read_text())
profile_dir = root / source.get('profile_directory', 'profiles')
binary_sha = hashlib.sha256(binary.read_bytes()).hexdigest()
assert binary_sha == provenance['upstream']['sha256']
symbols = []
command = ['readelf', '--wide', '--syms', str(binary)]
for line in subprocess.check_output(command, text=True).splitlines():
    x = line.split(maxsplit=7)
    if len(x) != 8 or not x[0].removesuffix(':').isdigit() or not x[6].isdigit():
        continue
    symbols.append(dict(address=int(x[1], 16), size=int(x[2]), kind=x[3],
                        section=x[6], name=x[7]))
funcs = {}
for s in symbols:
    if s['kind'] == 'FUNC':
        funcs.setdefault(s['section'], {}).setdefault(s['address'], []).append(s)
starts = {section: sorted(entries) for section, entries in funcs.items()}
owners = {}
for s in symbols:
    if not s['name'].startswith('..@'):
        continue
    locations = starts.get(s['section'], [])
    i = bisect_right(locations, s['address']) - 1
    if i < 0:
        continue
    for f in funcs[s['section']][locations[i]]:
        end = (f['address'] + f['size'] if f['size'] else
               locations[i + 1] if i + 1 < len(locations) else None)
        if end is None or s['address'] >= end:
            continue
        item = dict(address=s['address'], section=s['section'], owner=f['name'],
                    owner_start=f['address'], owner_end=end,
                    end_inferred_from_next_function=f['size'] == 0)
        entries = owners.setdefault(s['name'], [])
        if item not in entries:
            entries.append(item)


def group(name):
    if 'msac' in name or 'decode_coefs' in name:
        return 'entropy'
    if ('::itx' in name or 'inv_txfm' in name or
            re.search(r'inv_(dct|adst|identity|wht)|dav1d_i(dct|adst|flipadst|identity|wht)', name)):
        return 'transforms'
    if 'lpf_' in name or 'loopfilter' in name or 'loop_filter' in name:
        return 'loopfilter'
    return 'other'


frames = {}
for r in json.loads((profile_dir / 'commands.json').read_text()):
    name, threads, arm = r['case'].split(':')
    results = [line.split('\t') for line in r['stdout'].splitlines()
               if line.startswith('RESULT\t')]
    assert len(results) == 1 and r['exit'] == 0
    frames[f'{name}-t{threads}-{arm}.txt'] = int(results[0][2])
records = []
for r in source['profiles']:
    path = profile_dir / r['profile']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == r['report_sha256']
    groups = {k: 0.0 for k in r['groups']}
    resolved = []
    for s in r['symbols']:
        key = s['group']
        if key == 'anonymous_asm' and 'upstream' in r['profile']:
            candidates = owners.get(s['symbol'], [])
            possible_groups = {group(f['owner']) for f in candidates}
            if len(possible_groups) == 1:
                key = next(iter(possible_groups))
                resolved.append(dict(symbol=s['symbol'], self_percent=s['self_percent'],
                                     group=key, owners=candidates))
        groups[key] += s['self_percent']
    cycles = int(re.search(r'Event count \(approx\.\):\s*(\d+)', path.read_text())[1])
    frame_count = frames[r['profile']]
    assert frame_count > 0
    records.append(dict(profile=r['profile'], groups={k: round(v, 2) for k, v in groups.items()},
                        resolved=resolved, sampled_cycle_estimate=cycles, timed_frames=frame_count,
                        million_cycles_per_frame=cycles / frame_count / 1e6,
                        group_million_cycles_per_frame={k: cycles * v / 100 / frame_count / 1e6
                                                        for k, v in groups.items()}))
result = dict(upstream=str(binary), upstream_sha256=binary_sha, symbol_command=command,
              profile_directory=str(profile_dir),
              method='ELF FUNC intervals, with next-function ends inferred for zero-sized NASM '
                     'symbols. All same-name local symbol instances must imply one stage group. '
                     'Approximate sampled cycles are divided by the harness timed-frame count. '
                     'Single three-second profiles guide priorities, not acceptance; inlining '
                     'and sampling affect attribution and there are no confidence bounds.',
              profiles=records)
(root / (a.name + '-profile-resolved.json')).write_text(json.dumps(result, indent=2) + '\n')
for r in records:
    print(r['profile'], {k: round(v, 2) for k, v in r['group_million_cycles_per_frame'].items()})
