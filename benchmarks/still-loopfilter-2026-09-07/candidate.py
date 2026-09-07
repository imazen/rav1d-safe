"""Build early-return experiments; restore the checked baseline in finally.

Run through run-heavy with no other builds/source writers. These are
uninstrumented timing binaries. Their masks and reference footprints do not
change: a zero filter mask selects the original pixels in every lane.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--repo', type=Path, required=True)
p.add_argument('--work-dir', type=Path, required=True)
p.add_argument('--baseline-revision', required=True)
p.add_argument('--shared-target', type=Path, required=True)
a = p.parse_args()
repo, root = a.repo.resolve(), a.work_dir.resolve()
relative = 'src/safe_simd/loopfilter.rs'
path = repo / relative
before = path.read_bytes()
assert before == subprocess.check_output(['git', 'show', a.baseline_revision + ':' + relative], cwd=repo)
assert 'codex-still-parity' in (repo / '.workongoing').read_text()
source = before.decode()
matches = list(re.finditer(r'^fn (loop_filter_4_8bpc_[^(]+)\(', source, re.M))
assert len(matches) == 13
records = []
try:
    for label, widths in [('early8', [8]), ('earlywide', [8, 16])]:
        edits, names = [], []
        for index, match in enumerate(matches):
            name = match[1]
            if not any(f'_wd{w}_' in name for w in widths):
                continue
            end = matches[index + 1].start() if index + 1 < len(matches) else source.index('fn read_lvl')
            body = source[match.end():end]
            start = body.index('let fm_mask =')
            at = match.end() + body.index(';', start) + 1
            test = ('fm_mask == 0' if name.endswith('_x16') else
                    '_mm256_testz_si256(fm_mask, fm_mask) != 0' if name.endswith('_x8') else
                    '_mm_testz_si128(fm_mask, fm_mask) != 0')
            edits.append((at, '\n    // No lane can change: avoid computing unused filter alternatives.\n'
                          f'    if {test} {{\n        return;\n    }}\n'))
            names.append(name)
        assert len(edits) == 4 * len(widths)
        changed = source
        for at, text in reversed(edits):
            changed = changed[:at] + text + changed[at:]
        path.write_text(changed)
        patch = subprocess.check_output(['git', 'diff', a.baseline_revision, '--', relative], cwd=repo)
        (repo / f'benchmarks/still-loopfilter-2026-09-07/experiments/{label}.patch.gz').write_bytes(
            gzip.compress(patch, mtime=0))
        (root / (label + '-source.rs')).write_text(changed)
        records.append(dict(label=label, baseline_revision=a.baseline_revision, functions=names,
                            baseline_sha256=hashlib.sha256(before).hexdigest(),
                            candidate_sha256=hashlib.sha256(changed.encode()).hexdigest()))
        (root / 'early-return-source-audit.json').write_text(json.dumps(records, indent=2) + '\n')
        driver = root / (label + '-driver')
        driver.mkdir(exist_ok=True)
        (driver / 'target').symlink_to(a.shared_target.resolve(), target_is_directory=True)
        (repo / '.workongoing').write_text(datetime.now(timezone.utc).isoformat()
            + ' codex-still-parity building loopfilter experiment ' + label + '\n')
        subprocess.run([sys.executable, str(repo / 'benchmarks/tracker-sharding-2026-09-07/build.py'),
                        '--repo', str(repo), '--work-dir', str(root), '--label', label,
                        '--modes', 'checked'], cwd=repo, check=True)
finally:
    path.write_bytes(before)
    (repo / '.workongoing').write_text(datetime.now(timezone.utc).isoformat()
        + ' codex-still-parity early-return experiment builds ended; baseline restored\n')
