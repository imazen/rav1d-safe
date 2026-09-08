#!/usr/bin/env python3
"""Build Cargo's selected package sources before dependencies reach crates.io.

This checks file inclusion, not registry publication. The isolated manifest
retains the pinned archmage Git dependency and uses the local disjoint-mut
source. Run cargo publish --dry-run separately after dependencies are published.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--work-dir', type=Path, required=True)
p.add_argument('--features', nargs='+', default=['default', 'unchecked', 'c-ffi', 'asm', 'partial_asm'])
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
w = a.work_dir.resolve()
w.mkdir(parents=True, exist_ok=False)
stage = w / 'source'
stage.mkdir()
listing = subprocess.run(['cargo', 'package', '-p', 'rav1d-safe', '--list', '--allow-dirty'],
                         cwd=root, capture_output=True, text=True, check=True)
(w / 'package-list.txt').write_text(listing.stdout)
records = []
for name in listing.stdout.splitlines():
    source = root / name
    if not source.is_file():
        continue  # Cargo-generated manifest/lock/VCS metadata.
    target = stage / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    records.append({'path': name, 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
required = ['src/ext/x86/x86inc.asm', 'src/x86/msac.asm', 'src/arm/asm.S',
            'src/arm/32/msac.S', 'src/arm/64/msac.S', 'docs/RUST_CODEC_WORKFLOW.md']
for name in required:
    assert (stage / name).is_file(), 'Missing package input: ' + name
manifest = (stage / 'Cargo.toml').read_text()
manifest = manifest.replace('members = [".", "crates/rav1d-disjoint-mut"]', 'members = ["."]')
manifest = manifest.replace('path = "crates/rav1d-disjoint-mut"',
                            'path = ' + json.dumps(str(root / 'crates/rav1d-disjoint-mut')))
# Cargo's normalized package manifest omits excluded benchmark targets too.
manifest = re.sub(r'\[\[bench\]\][\s\S]*?(?=\n\[|\Z)', '', manifest)
(stage / 'Cargo.toml').write_text(manifest)
env = os.environ.copy()
env.pop('CARGO_ENCODED_RUSTFLAGS', None)
env.pop('RUSTFLAGS', None)
env['CARGO_TARGET_DIR'] = str(w / 'target')
env['CARGO_TERM_COLOR'] = 'never'
results = []
for features in a.features:
    command = ['cargo', 'build', '--manifest-path', str(stage / 'Cargo.toml'), '--lib']
    if features != 'default':
        command += ['--features', features]
    start = time.monotonic()
    log = w / (features + '.log')
    with log.open('w') as output:
        result = subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT)
    results.append({'features': features, 'command': command, 'exit': result.returncode,
                    'seconds': round(time.monotonic() - start, 2),
                    'log_sha256': hashlib.sha256(log.read_bytes()).hexdigest()})
    (w / 'results.json').write_text(json.dumps({'source_files': records, 'builds': results}, indent=2) + '\n')
    print(features, result.returncode, flush=True)
    if result.returncode:
        print(log.read_text()[-5000:])
        raise SystemExit(result.returncode)
