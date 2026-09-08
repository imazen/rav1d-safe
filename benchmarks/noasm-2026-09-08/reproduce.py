#!/usr/bin/env python3
"""Rebuild the no-assembly comparison and replay the archived still inputs."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--upstream', type=Path, required=True)
p.add_argument('--work-dir', type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[2]
evidence = Path(__file__).resolve().parent / 'evidence'
w = a.work_dir.resolve()
w.mkdir(parents=True, exist_ok=False)
upstream = a.upstream.resolve()
revision = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
assert revision == 'd3d1cd67059f47803919be8276650e5870c9fd02', revision
assert not subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain'])

def restore(name):
    return gzip.decompress((evidence / (name + '.gz')).read_bytes())

corpus = json.loads(restore('corpus.json'))
for item in corpus:
    data = restore('inputs/' + item['name'] + '.ivf')
    assert len(data) == item['bytes'] and hashlib.sha256(data).hexdigest() == item['sha256']
    target = w / (item['name'] + '.ivf')
    target.write_bytes(data)
    item['path'] = str(target)
(w / 'corpus.json').write_text(json.dumps(corpus, indent=2) + '\n')
(w / 'bin').mkdir()
driver = w / 'upstream-driver'
(driver / 'src').mkdir(parents=True)
source = root / 'benchmarks/upstream-2026-09-07/driver'
shutil.copyfile(source / 'src/main.rs', driver / 'src/main.rs')
(driver / 'Cargo.lock').write_bytes(restore('upstream-driver/Cargo.lock'))
manifest = (source / 'Cargo.toml').read_text().split('[[bin]]')[0]
manifest = manifest.replace('rav1d = { path = "../upstream" }',
    'rav1d = { path = ' + json.dumps(str(upstream)) + ', default-features = false, features = ["bitdepth_8", "bitdepth_16"] }')
(driver / 'Cargo.toml').write_text(manifest)
env = os.environ.copy()
env.pop('CARGO_ENCODED_RUSTFLAGS', None)
env.pop('CARGO_TARGET_DIR', None)
env['RUSTFLAGS'] = '-C llvm-args=-align-all-functions=4'
subprocess.run(['cargo', 'build', '--release', '--locked', '--manifest-path', str(driver / 'Cargo.toml')], env=env, check=True)
shutil.copy2(driver / 'target/release/rav1d-upstream-profile', w / 'bin/upstream-noasm')
lock = w / 'safe-Cargo.lock'
lock.write_bytes(restore('safe-driver/Cargo.lock'))
subprocess.run(['python3', str(root / 'benchmarks/tracker-sharding-2026-09-07/build.py'),
    '--repo', str(root), '--work-dir', str(w), '--label', 'safe', '--modes', 'checked',
    '--lockfile', str(lock)], env=env, check=True)
subprocess.run(['python3', str(root / 'benchmarks/stills-2026-09-07/compare.py'),
    '--work-dir', str(w), '--name', 'noasm-vs-checked', '--upstream', str(w / 'bin/upstream-noasm'),
    '--arms', 'rav1d-noasm=' + str(w / 'bin/upstream-noasm'),
    'rav1d-safe-checked=' + str(w / 'bin/safe-checked'),
    '--cells', '1:1:0', '4:1:0', '8:1:0', '--reps', '5', '--target-ms', '150'], check=True)
