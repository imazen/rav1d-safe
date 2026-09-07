"""Prove the mixed-transform sweep detects a swapped row/column dispatch.

Run through run-heavy with exclusive source ownership. This never produces a
timing consumer, and restores the original source before checking the result.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--repo', type=Path, required=True)
p.add_argument('--work-dir', type=Path, required=True)
a = p.parse_args()
assert platform.machine() == 'x86_64'
repo, root = a.repo.resolve(), a.work_dir.resolve()
source = repo / 'src/safe_simd/itx/part10_dispatch.rs'
original = source.read_bytes()
needle = b'(S16x16, ADST_DCT) => arcane!(inv_txfm_add_dct_adst_16x16_8bpc_avx2_inner)'
replacement = b'(S16x16, ADST_DCT) => arcane!(inv_txfm_add_adst_dct_16x16_8bpc_avx2_inner)'
assert original.count(needle) == 1
mutant = original.replace(needle, replacement)
marker = repo / '.workongoing'
assert 'codex-still-parity' in marker.read_text()
(root / 'wire16-before-mutation.rs').write_bytes(original)
command = ['cargo', 'nextest', 'run', '--locked', '--release', '--lib',
           '--no-default-features', '--features', 'bitdepth_8,bitdepth_16',
           '-E', 'test(test_mixed16_dispatch_matches_scalar)']
log = root / 'wire16-row-column-mutation.log'
try:
    marker.write_text(datetime.now(timezone.utc).isoformat()
                      + ' codex-still-parity checking deliberate row/column mutation\n')
    source.write_bytes(mutant)
    with log.open('x') as output:
        output.write('Deliberately swap the ADST_DCT row/column kernel.\n')
        output.write('Command: ' + repr(command) + '\n')
        output.flush()
        result = subprocess.run(command, cwd=repo, stdout=output, stderr=subprocess.STDOUT)
finally:
    source.write_bytes(original)
    marker.write_text(datetime.now(timezone.utc).isoformat()
                      + ' codex-still-parity mutation run ended; candidate dispatch restored\n')
assert source.read_bytes() == original
output = re.sub(r'\x1b\[[0-9;]*m', '', log.read_text())
detected = result.returncode != 0 and bool(re.search(
    r'FAIL[^\n]*test_mixed16_dispatch_matches_scalar', output))
(root / 'wire16-mutation-result.json').write_text(json.dumps(dict(
    command=command, exit=result.returncode, detected=detected,
    original_sha256=hashlib.sha256(original).hexdigest(),
    mutant_sha256=hashlib.sha256(mutant).hexdigest(), restored=True), indent=2) + '\n')
assert detected, output[-4000:]
print('Row/column mutation detected; original dispatch restored.')
