"""Build matched alternate-placement controls; restore source in finally.

Run through run-heavy, with no concurrent build or source writer. The baseline
restores both modified existing files from the pinned revision. The two new
files remain on disk but the baseline does not include or compile them.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--repo', type=Path, required=True)
p.add_argument('--work-dir', type=Path, required=True)
p.add_argument('--baseline-revision', required=True)
p.add_argument('--shared-target', type=Path, required=True)
p.add_argument('--alignment', type=int, choices=range(3, 7), default=5)
a = p.parse_args()
repo, root = a.repo.resolve(), a.work_dir.resolve()
existing = ['src/loopfilter.rs', 'src/safe_simd/loopfilter.rs']
new = ['src/safe_simd/loopfilter_packed6.rs', 'src/safe_simd/loopfilter_parity.rs']
before = {name: (repo / name).read_bytes() for name in existing + new}
audit = json.loads((root / 'packed6-source-audit.json').read_text())
assert {name: hashlib.sha256(data).hexdigest() for name, data in before.items()} == audit['files']
baseline = {name: subprocess.check_output(
    ['git', 'show', a.baseline_revision + ':' + name], cwd=repo) for name in existing}
assert 'codex-still-parity' in (repo / '.workongoing').read_text()
labels = [(f'packed6-align{a.alignment}', before),
          (f'packed6-baseline-align{a.alignment}', baseline)]
for label, _ in labels:
    assert not (root / 'bin' / (label + '-checked')).exists(), label
record_path = root / f'packed6-placement-align{a.alignment}-sources.json'
assert not record_path.exists(), record_path
record = dict(baseline_revision=a.baseline_revision, variants=[], restored=False)


def save():
    record_path.write_text(json.dumps(record, indent=2) + '\n')


try:
    for label, source in labels:
        for name, data in source.items():
            (repo / name).write_bytes(data)
        driver = root / (label + '-driver')
        driver.mkdir(exist_ok=True)
        (driver / 'target').symlink_to(a.shared_target.resolve(), target_is_directory=True)
        (repo / '.workongoing').write_text(datetime.now(timezone.utc).isoformat()
            + ' codex-still-parity building loopfilter placement control ' + label + '\n')
        command = [sys.executable, str(repo / 'benchmarks/tracker-sharding-2026-09-07/build.py'),
                   '--repo', str(repo), '--work-dir', str(root), '--label', label,
                   '--modes', 'checked', '--function-alignment', str(a.alignment)]
        entry = dict(label=label, command=command,
                     files={name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
                            for name in before})
        record['variants'].append(entry)
        save()
        result = subprocess.run(command, cwd=repo)
        entry['exit'] = result.returncode
        save()
        result.check_returncode()
finally:
    for name, data in before.items():
        (repo / name).write_bytes(data)
    record['restored'] = all((repo / name).read_bytes() == data for name, data in before.items())
    save()
    (repo / '.workongoing').write_text(datetime.now(timezone.utc).isoformat()
        + ' codex-still-parity placement builds ended; packed6 source restored\n')
