"""Build matched code-placement controls, restoring candidate source in finally.

Run through run-heavy. No builds, benchmarks or other source writers may
overlap this script. Large targets remain in the existing scratch directory.
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
a = p.parse_args()
repo, root = a.repo.resolve(), a.work_dir.resolve()
path = repo / 'src/itx.rs'
candidate = path.read_bytes()
audit = json.loads((root / 'inline8-source-audit.json').read_text())
assert hashlib.sha256(candidate).hexdigest() == audit['source_sha256']
baseline = subprocess.check_output(['git', 'show', a.baseline_revision + ':src/itx.rs'], cwd=repo)
assert baseline == (root / 'before-itx.rs').read_bytes()
marker = repo / '.workongoing'
assert 'codex-still-parity' in marker.read_text()
records = []
try:
    for label, source in [('baseline-align5', baseline), ('inline8-align5', candidate)]:
        marker.write_text(datetime.now(timezone.utc).isoformat()
                          + ' codex-still-parity building matched alignment control ' + label + '\n')
        driver = root / (label + '-driver')
        driver.mkdir(exist_ok=True)
        (driver / 'target').symlink_to((root / 'census-driver/target').resolve(), target_is_directory=True)
        path.write_bytes(source)
        command = [sys.executable, str(repo / 'benchmarks/tracker-sharding-2026-09-07/build.py'),
                   '--repo', str(repo), '--work-dir', str(root), '--label', label,
                   '--modes', 'checked', '--function-alignment', '5']
        subprocess.run(command, cwd=repo, check=True)
        records.append(dict(label=label, source_sha256=hashlib.sha256(source).hexdigest(), command=command))
        (root / 'alignment-controls.json').write_text(json.dumps(records, indent=2) + '\n')
finally:
    path.write_bytes(candidate)
    marker.write_text(datetime.now(timezone.utc).isoformat()
                      + ' codex-still-parity alignment build ended; tested candidate source restored\n')
