#!/usr/bin/env python3
"""Check diagnostic naming, feature reachability and source environment gates."""
from pathlib import Path
import subprocess
import tomllib

root = Path(__file__).resolve().parents[1]
public = {
    'rav1d-safe': {'default', 'bitdepth_8', 'bitdepth_16', 'unchecked', 'c-ffi',
                   'dav1d-compat', 'partial_asm', 'asm', 'asm_arm64_dotprod',
                   'asm_arm64_i8mm', 'asm_arm64_sve2'},
    'rav1d-disjoint-mut': {'default', 'std', 'aligned', 'pic-buf', 'zerocopy', 'instrument'},
}
features = {}
for relative in ['Cargo.toml', 'crates/rav1d-disjoint-mut/Cargo.toml']:
    manifest = tomllib.loads((root / relative).read_text())
    name = manifest['package']['name']
    fs = features[name] = manifest['features']
    unexpected = set(fs) - public[name] - {f for f in fs if f.startswith('__')}
    assert not unexpected, f'{name}: unprefixed experimental features {unexpected}'
for package, names in public.items():
    for name in names:
        pending = [(package, name)]
        seen = set()
        while pending:
            pkg, feature = pending.pop()
            if (pkg, feature) in seen:
                continue
            seen.add((pkg, feature))
            assert not feature.startswith('__'), f'{package}/{name} activates internal {pkg}/{feature}'
            for dep in features[pkg].get(feature, []):
                if dep.startswith('dep:'):
                    continue
                if '/' in dep:
                    other, target = dep.split('/', 1)
                    if other in features:
                        pending.append((other, target))
                elif dep in features[pkg]:
                    pending.append((pkg, dep))
print('Feature names and public/default dependency closures pass', flush=True)
subprocess.run(['cargo', 'run', '--locked', '--manifest-path', str(root / 'tools/feature-policy/Cargo.toml'),
                '--', str(root / 'lib.rs'), str(root / 'src'), str(root / 'include'),
                str(root / 'crates/rav1d-disjoint-mut/src')], cwd=root, check=True)
