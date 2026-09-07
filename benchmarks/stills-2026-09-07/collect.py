"""Commit complete compressed text evidence; preserve large artifacts externally."""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('work_dir', type=Path)
a = p.parse_args()
root = a.work_dir.resolve()
out = Path(__file__).resolve().parent / 'results'
out.mkdir(exist_ok=True)


def save(name, data):
    assert len(data) <= 30_000, (name, len(data))
    (out / name).write_bytes(data)


def compressed(name, data):
    save(name + '.gz', gzip.compress(data, mtime=0))


totals = defaultdict(int)
for path in sorted(root.glob('*.jsonl')):
    groups = defaultdict(list)
    for line in path.read_text().splitlines():
        row = json.loads(line)
        assert row['exit'] == 0, row
        groups[row['input']].append(line)
        result = [l.split('\t') for l in row['stdout'].splitlines() if l.startswith('RESULT\t')]
        checked = [l.split('\t') for l in row['stdout'].splitlines() if l.startswith('VALIDATED\t')]
        assert len(result) == len(checked) == 1
        totals['pilot_runs' if row['rep'] < 0 else 'measured_runs'] += 1
        totals['pilot_frames' if row['rep'] < 0 else 'timed_frames'] += int(result[0][2])
        totals['ordered_frame_hash_checks'] += int(checked[0][1])
    for name, lines in groups.items():
        compressed(path.stem + '-' + name + '.jsonl', ('\n'.join(lines) + '\n').encode())
table = io.StringIO()
writer = csv.writer(table, delimiter='\t', lineterminator='\n')
writer.writerow(['run', 'input', 'threads', 'instances', 'prime', 'arm', 'median_ms', 'upstream_ms', 'ratio'])
for path in sorted(root.glob('*-summary.json')):
    for r in json.loads(path.read_text()):
        u = r['medians'].get('upstream')
        for arm, ms in r['medians'].items():
            writer.writerow([path.stem.removesuffix('-summary'), r['input'], r['threads'],
                             r['instances'], r['prime'], arm, ms, u, ms / u if u else ''])
save('summary.tsv', table.getvalue().encode())
for path in sorted(root.glob('*.json')):
    compressed(path.name, path.read_bytes())
for path in sorted(root.glob('*.log')):
    compressed(path.name, path.read_bytes())
for folder in ['profiles', 'probes', 'validation']:
    for path in sorted((root / folder).glob('*')):
        if path.suffix in ['.txt', '.json', '.log']:
            compressed(folder + '-' + path.name, path.read_bytes())
save('verification.json', (json.dumps(dict(totals), indent=2) + '\n').encode())
# The manifest is enough to identify all large immutable evidence even without
# /mnt/v on this host. Reproduction/download commands accompany the pointer.
large = {}
for path in root.rglob('*'):
    if path.is_file() and path.suffix in ['.data', '.ivf', '.y4m']:
        large[str(path.relative_to(root))] = dict(bytes=path.stat().st_size,
                                                 sha256=hashlib.sha256(path.read_bytes()).hexdigest())
compressed('large-artifacts.json', (json.dumps(large, indent=2) + '\n').encode())
hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in out.iterdir()
          if path.name != 'SHA256.json.gz'}
compressed('SHA256.json', (json.dumps(hashes, indent=2) + '\n').encode())
print(dict(totals), flush=True)
