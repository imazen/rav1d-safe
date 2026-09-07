"""Preserve complete small text evidence; large binaries/perf.data stay outside git."""
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

timings = frames = validations = 0
table = []
for path in sorted(root.glob('*.jsonl')):
    groups = defaultdict(list)
    for line in path.read_text().splitlines():
        row = json.loads(line)
        assert row['exit'] == 0, (path, row)
        groups[row['input']].append(line)
        result = [l.split('\t') for l in row['stdout'].splitlines() if l.startswith('RESULT\t')]
        checked = [l.split('\t') for l in row['stdout'].splitlines() if l.startswith('VALIDATED\t')]
        assert len(result) == len(checked) == 1, path
        timings += 1
        frames += int(result[0][2])
        validations += int(checked[0][1])
    for name, lines in groups.items():
        compressed(path.stem + '-' + name + '.jsonl', ('\n'.join(lines) + '\n').encode())
for path in sorted(root.glob('*-summary.json')):
    for row in json.loads(path.read_text()):
        medians = row['medians']
        candidate = next(k for k in ['solution', 'final', 'shard1k'] if k in medians)
        table.append([path.stem.removesuffix('-summary'), row['input'], row['threads'],
                      row['instances'], row['prime'], candidate, medians['baseline'],
                      medians[candidate], medians.get('upstream', ''),
                      (medians[candidate] / medians['baseline'] - 1) * 100])
    save(path.name, path.read_bytes())
buffer = io.StringIO()
writer = csv.writer(buffer, delimiter='\t', lineterminator='\n')
writer.writerow(['run', 'input', 'threads', 'instances', 'prime', 'candidate',
                 'baseline_ms', 'candidate_ms', 'upstream_ms', 'change_percent'])
writer.writerows(table)
save('summary.tsv', buffer.getvalue().encode())
for path in sorted(root.glob('*.log')):
    if path.name.startswith('collect'):
        continue
    compressed(path.name, path.read_bytes())
for path in sorted(root.glob('*.json')):
    if path.name.endswith('-summary.json'):
        continue
    save(path.name, path.read_bytes())
for path in sorted((root / 'profiles').iterdir()):
    if path.suffix in ['.txt', '.json']:
        compressed('profile-' + path.name, path.read_bytes())
latest = {}
for path in sorted((root / 'validation').glob('results-*.json'), key=lambda p: p.stat().st_mtime):
    for row in json.loads(path.read_text()):
        latest[row['name']] = row
save('validation.json', (json.dumps(list(latest.values()), indent=2) + '\n').encode())
for path in sorted((root / 'validation').glob('*.log')):
    compressed('validation-' + path.name, path.read_bytes())
save('verification.json', (json.dumps(dict(timing_runs=timings, timed_frames=frames,
                                         ordered_hash_passes=validations), indent=2) + '\n').encode())
checksums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()
             if p.name != 'SHA256.json.gz'}
compressed('SHA256.json', (json.dumps(checksums, indent=2) + '\n').encode())
print(f'Preserved {timings} runs, {frames} timed frames, {validations} ordered hash passes', flush=True)
