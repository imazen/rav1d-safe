"""Check that the exhaustive CDF test detects a wrong SIMD count lane.

Run on x86-64 through run-heavy with exclusive ownership of the checkout.
The original source is backed up and restored in finally before evaluating
the expected test failure. This never builds a timed consumer.
"""
import argparse
import hashlib
from pathlib import Path
import platform
import re
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--repo', type=Path, required=True)
p.add_argument('--work-dir', type=Path, required=True)
a = p.parse_args()
assert platform.machine() == 'x86_64', 'this mutation targets the SSE2 kernel'
source = a.repo / 'src/msac.rs'
original = source.read_bytes()
needle = b'_mm_insert_epi16::<3>(updated,'
assert original.count(needle) == 1
a.work_dir.mkdir(parents=True, exist_ok=True)
(a.work_dir / 'msac-before-mutation.rs').write_bytes(original)
command = ['cargo', 'nextest', 'run', '--release', '--lib', '--no-default-features',
           '--features', 'bitdepth_8,bitdepth_16', '-E',
           'test(cdf_update_matches_directional_arithmetic_for_every_u16)']
log = a.work_dir / 'cdf-count-lane-mutation.log'
with log.open('x') as output:
    output.write('Deliberate mutation: replace count lane 3 with lane 2.\n')
    output.write('Original source SHA-256: ' + hashlib.sha256(original).hexdigest() + '\n')
    output.write('Command: ' + repr(command) + '\n')
    output.flush()
    try:
        source.write_bytes(original.replace(needle, b'_mm_insert_epi16::<2>(updated,'))
        result = subprocess.run(command, cwd=a.repo, stdout=output, stderr=subprocess.STDOUT)
    finally:
        source.write_bytes(original)
assert source.read_bytes() == original
text = re.sub(r'\x1b\[[0-9;]*m', '', log.read_text())
assert result.returncode != 0
assert re.search(r'FAIL[^\n]*cdf_update_matches_directional_arithmetic_for_every_u16', text), text[-4000:]
print('Mutation detected by the exhaustive CDF test; original source restored.')
