"""Archive small, lossless experiment evidence; large binaries stay in scratch."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--work-dir', type=Path, required=True)
p.add_argument('--output-dir', type=Path, help='Archive a separate experiment campaign')
a = p.parse_args()
root = a.work_dir.resolve()
out = a.output_dir or (Path(__file__).resolve().parent / 'results')
out.mkdir(parents=True, exist_ok=True)
records = []


def save(name, data, source):
    compressed = gzip.compress(data, mtime=0)
    assert len(compressed) < 30_000, name
    (out / (name + '.gz')).write_bytes(compressed)
    records.append(dict(file=name + '.gz', source=str(source), bytes=len(data),
                        sha256=hashlib.sha256(data).hexdigest()))


for path in sorted(root.glob('*')):
    if path.suffix in ['.json', '.log']:
        save(path.name, path.read_bytes(), path)
    elif path.suffix == '.jsonl':
        rows = path.read_bytes().splitlines(keepends=True)
        for first in range(0, len(rows), 40):
            save(f'{path.stem}-{first // 40:03}.jsonl',
                 b''.join(rows[first:first + 40]), path)
for path in sorted((root / 'profiles').glob('*')):
    if path.suffix in ['.txt', '.json']:
        save('profile-' + path.name, path.read_bytes(), path)
pages = []
for first in range(0, len(records), 40):
    name = f'index-{first // 40:03}.json'
    data = (json.dumps(records[first:first + 40], indent=2) + '\n').encode()
    assert len(data) < 30_000, name
    (out / name).write_bytes(data)
    pages.append(dict(file=name, records=len(records[first:first + 40]),
                      sha256=hashlib.sha256(data).hexdigest()))
(out / 'index.json').write_text(json.dumps(
    dict(records=len(records), pages=pages), indent=2) + '\n')
print(f'Archived {len(records)} evidence files from {root}')
