"""Build and run the borrowing/task census, separately from timing arms.

Run through run-heavy. Counter totals include reference/warmup/validation;
use VALIDATED lifetime_frames as their denominator. Stage hooks only execute
on the multi-worker task path. Probe times are not benchmark results.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--work-dir', type=Path, required=True)
p.add_argument('--build', action='store_true')
p.add_argument('--cases', nargs='+', default=[
    'photo-2k-min:1', 'photo-2k-t8:8', 'photo-4k-min:1', 'photo-4k-min:8',
    'photo-4k-t8:8', 'photo-8k-min:8', 'photo-8k-t8:8', 'map-4k-t8:8', 'map-8k-t8:8'])
a = p.parse_args()
repo = Path(__file__).resolve().parents[2]
root = a.work_dir.resolve()
out = root / 'probes'
out.mkdir(exist_ok=True)
binary = root / 'bin/still-probe'
if a.build:
    source = repo / 'benchmarks/tracker-sharding-2026-09-07/drivers/current'
    driver = root / 'probe-driver'
    (driver / 'src').mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / 'main.rs', driver / 'src/main.rs')
    shutil.copy2(source / 'Cargo.lock.snapshot', driver / 'Cargo.lock')
    manifest = (source / 'Cargo.toml').read_text().replace(
        '/home/lilith/work/zen/rav1d-safe', str(repo)).replace(
        '__probe_tasktime = []', '__probe_tasktime = ["rav1d-safe/__probe_tasktime"]')
    (driver / 'Cargo.toml').write_text(manifest)
    env = os.environ.copy()
    env.pop('CARGO_ENCODED_RUSTFLAGS', None)
    env['RUSTFLAGS'] = '-C llvm-args=-align-all-functions=4'
    command = ['cargo', 'build', '--locked', '--release', '--manifest-path',
               str(driver / 'Cargo.toml'), '--features', '__probe_usage,__probe_tasktime']
    with (out / 'build.log').open('w') as log:
        subprocess.run(command, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
    binary.parent.mkdir(exist_ok=True)
    assert not binary.exists()
    shutil.copy2(driver / 'target/release/rav1d-release-profile-current', binary)
    (out / 'build.json').write_text(json.dumps(dict(command=command, rustflags=env['RUSTFLAGS'],
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest()), indent=2) + '\n')
corpus = {c['name']: c for c in json.loads((root / 'corpus.json').read_text())}
records = []
for case in a.cases:
    name, threads = case.split(':')
    source = corpus[name]
    command = [str(binary), source['path'], threads, '1', '2', '1']
    env = {k: v for k, v in os.environ.items() if not k.startswith('RAV1D_')}
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
    (out / (name + '-t' + threads + '.log')).write_text(result.stdout + result.stderr)
    assert result.returncode == 0, case
    lines = result.stdout.splitlines()
    md5 = (root / (name + '.dav1d.md5')).read_text().split()[0]
    assert f"FRAME\t0\t{source['width']}x{source['height']}\t{source['bit_depth']}\t{md5}" in lines
    assert 'VALIDATED\t3\tlifetime_frames=5\ttimed_frames=2' in lines
    expected = f"GEOMETRY\t0\t{source['width']}\t{source['height']}\t({source['tile_cols']}, {source['tile_rows']}, 0, 0)"
    assert expected in lines, (case, [l for l in lines if l.startswith('GEOMETRY')])
    records.append(dict(case=case, command=command, exit=result.returncode, geometry=expected))
    (out / 'commands.json').write_text(json.dumps(records, indent=2) + '\n')
    print(case, 'borrow/stage census validated', flush=True)
