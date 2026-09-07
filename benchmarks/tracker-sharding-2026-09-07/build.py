"""Build isolated consumers without the decoder's dev-dependency features.

Invoke through run-heavy; builds and benchmark runs must never overlap.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--repo', type=Path, required=True)
p.add_argument('--work-dir', type=Path, required=True)
p.add_argument('--label', required=True)
p.add_argument('--driver', choices=['current', 'auto'], default='current')
p.add_argument('--modes', nargs='+', default=['checked', 'unchecked', 'asm'])
p.add_argument('--lockfile', type=Path, help='Pinned consumer lockfile override')
p.add_argument('--archmage-repo', type=Path, help='Local archmage/magetypes patch checkout')
p.add_argument('--refresh-lock', action='store_true', help='Allow only an explicit new dependency control to update its lockfile')
a = p.parse_args()
repo = a.repo.resolve()
root = a.work_dir.resolve()
driver = root / (a.label + '-driver')
(driver / 'src').mkdir(parents=True, exist_ok=True)
(root / 'bin').mkdir(exist_ok=True)
source = Path(__file__).resolve().parent / 'drivers' / a.driver
shutil.copyfile(source / 'main.rs', driver / 'src/main.rs')
shutil.copyfile(a.lockfile or source / 'Cargo.lock.snapshot', driver / 'Cargo.lock')
manifest = (source / 'Cargo.toml').read_text()
old = '/home/lilith/work/zen/rav1d-safe'
for suffix in ['', '/crates/rav1d-disjoint-mut']:
    manifest = manifest.replace(json.dumps(old + suffix), json.dumps(str(repo) + suffix))
if a.archmage_repo:
    patch = a.archmage_repo.resolve()
    manifest += '\n[patch.crates-io]\n'
    for name, suffix in [('archmage', ''), ('archmage-macros', 'archmage-macros'), ('magetypes', 'magetypes')]:
        manifest += f'{name} = {{ path = {json.dumps(str(patch / suffix))} }}\n'
(driver / 'Cargo.toml').write_text(manifest)
env = os.environ.copy()
env.pop('CARGO_ENCODED_RUSTFLAGS', None)
env['RUSTFLAGS'] = '-C llvm-args=-align-all-functions=4'
env['CARGO_TERM_COLOR'] = 'never'
records = []
for mode in a.modes:
    command = ['cargo', 'build', '--release', '--manifest-path',
               str(driver / 'Cargo.toml')]
    if not a.refresh_lock:
        command += ['--locked']
    if mode != 'checked':
        command += ['--features', mode]
    print(f'Building {a.label} {mode}', flush=True)
    with (root / f'{a.label}-{mode}-build.log').open('w') as log:
        result = subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
    assert result.returncode == 0, f'{mode} build failed; see its log'
    target = root / 'bin' / f'{a.label}-{mode}'
    assert not target.exists(), f'refusing to overwrite immutable binary {target}'
    shutil.copy2(driver / 'target/release' / f'rav1d-release-profile-{a.driver}', target)
    records.append(dict(mode=mode, command=command,
                        sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                        manifest_sha256=hashlib.sha256((driver / 'Cargo.toml').read_bytes()).hexdigest(),
                        lockfile_sha256=hashlib.sha256((driver / 'Cargo.lock').read_bytes()).hexdigest()))
    (root / f'{a.label}-builds.json').write_text(json.dumps(records, indent=2) + '\n')
    print(f'Built {target}', flush=True)
