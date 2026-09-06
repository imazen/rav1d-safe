#!/usr/bin/env python3
from pathlib import Path
import subprocess, os, hashlib, difflib, json, time
import argparse
parser = argparse.ArgumentParser(description='Diff verified release tarballs against the current API')
parser.add_argument('--releases', required=True, type=Path, help='directory with extracted crate-version directories')
parser.add_argument('--output', required=True, type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
releases = args.releases.resolve()
out = args.output.resolve()
out.mkdir(exist_ok=True)
env = os.environ.copy()
env.pop('RUSTFLAGS', None)
env.pop('CARGO_ENCODED_RUSTFLAGS', None)
env['CARGO_TARGET_DIR'] = str(root / 'target/review-api')
records = []
for crate, versions, features in [
    ('rav1d-disjoint-mut', ['0.2.1', '0.3.0', '0.3.1', 'current'], 'aligned,pic-buf,zerocopy'),
    ('rav1d-safe', ['0.5.5', '0.5.6', '0.5.7', 'current'], None),
]:
    for version in versions:
        manifest = root / 'Cargo.toml' if version == 'current' else releases / f'{crate}-{version}' / 'Cargo.toml'
        cmd = ['cargo', '+nightly', 'public-api', '--manifest-path', str(manifest), '-p', crate, '--omit', 'blanket-impls', '--color', 'never']
        if features: cmd += ['--features', features]
        name = f'{crate}-{version}'
        start = time.monotonic()
        with (out / f'{name}.txt').open('w') as stdout, (out / f'{name}.log').open('w') as stderr:
            result = subprocess.run(cmd, cwd=root, env=env, stdout=stdout, stderr=stderr)
        record = dict(crate=crate, version=version, command=cmd, exit_code=result.returncode, seconds=round(time.monotonic()-start,2))
        if result.returncode == 0:
            api = (out / f'{name}.txt').read_bytes()
            record.update(sha256=hashlib.sha256(api).hexdigest(), lines=len(api.splitlines()))
        records.append(record)
        (out/'manifest.json').write_text(json.dumps(records, indent=2)+'\n')
        print(json.dumps(record), flush=True)
        if result.returncode:
            print((out/f'{name}.log').read_text()[-5000:],flush=True)
            raise SystemExit(result.returncode)
    for a,b in zip(versions, versions[1:]):
        first = (out / f'{crate}-{a}.txt').read_text().splitlines(keepends=True)
        second = (out / f'{crate}-{b}.txt').read_text().splitlines(keepends=True)
        diff = ''.join(difflib.unified_diff(first,second,fromfile=f'{crate}-{a}',tofile=f'{crate}-{b}'))
        (out/f'{crate}-{a}-to-{b}.diff').write_text(diff)
        print(f'DIFF {crate} {a} -> {b}: {len(diff.splitlines())} lines',flush=True)
