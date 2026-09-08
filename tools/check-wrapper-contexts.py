#!/usr/bin/env python3
"""Compile the feature-acquisition contracts of dormant Rust FFI wrappers.

The decoder excludes safe_simd in ASM builds, while these legacy wrappers are
ASM-only. Normal cargo checks cannot type-check their contracts. This projects
ONLY each source annotation and first token-acquisition statement into an
isolated crate. It does not claim to compile or validate the raw-pointer bodies.
Run under the workspace heavy-job limiter, with an explicit scratch directory.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import tomllib

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--work-dir', type=Path, required=True)
p.add_argument('--target', default='x86_64-unknown-linux-gnu')
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
work = a.work_dir.resolve()
(work / 'src').mkdir(parents=True, exist_ok=True)
contracts = []
fn = re.compile(r'(?m)^\s*(?:pub(?:\([^)]*\))?\s+)?unsafe (?:extern "C" )?fn (\$?\w+)\(')
ctor = re.compile(r'archmage::(X64V3Token|NeonToken)::from_context\(\)')
for path in sorted((root / 'src/safe_simd').rglob('*.rs')):
    source = path.read_text()
    # Anchored calls, not documentation mentions of the retired API.
    assert not re.search(r'\b\w+::forge_token_dangerously\s*\(', source), path
    functions = list(fn.finditer(source))
    for call in ctor.finditer(source):
        function = [f for f in functions if f.start() < call.start()][-1]
        body = source.index('{', function.end())
        prefix = source[body + 1:call.end() + 1].strip()
        assert re.fullmatch(r'#\[deny\(unsafe_op_in_unsafe_fn\)\]\s*let \w+ = '
                            + re.escape(call[0]) + ';', prefix), (path, function[1], prefix)
        attrs = source[source.rfind('\n\n', 0, function.start()):function.start()]
        tier = 'v3' if call[1] == 'X64V3Token' else 'neon'
        assert f'#[archmage::rite({tier})]' in attrs, (path, function[1], attrs)
        assert function[1].startswith('$') or function[1].endswith('_' + tier), (path, function[1])
        if function[1].startswith('$'):
            macro = list(re.finditer(r'macro_rules!\s+(\w+)', source[:function.start()]))[-1][1]
            names = re.findall(r'\b' + macro + r'!\s*\(\s*(\w+)', source)
            assert names and all(n.endswith('_' + tier) for n in names), (path, macro, names)
        contracts.append(dict(file=str(path.relative_to(root)), function=function[1],
                              line=source.count('\n', 0, function.start()) + 1,
                              tier=tier, statement=prefix))
assert len(contracts) == 237, f'Update the reviewed inventory intentionally: {len(contracts)}'
(work / 'contracts.json').write_text(json.dumps(contracts, indent=2) + '\n')
dep = tomllib.loads((root / 'Cargo.toml').read_text())['dependencies']['archmage']
dependency = ', '.join(f'{key} = {json.dumps(dep[key])}'
                       for key in ('version', 'git', 'rev') if key in dep)
(work / 'Cargo.toml').write_text('[package]\nname="wrapper-context-contracts"\nversion="0.0.0"\nedition="2024"\n'
                                '[workspace]\n[dependencies]\narchmage = { '
                                + dependency + ', features = ["macros", "avx512"] }\n')
# Seed transitive versions from the decoder lockfile; Cargo removes unrelated packages.
shutil.copyfile(root / 'Cargo.lock', work / 'Cargo.lock')
selected = [c for c in contracts if c['tier'] == ('neon' if a.target.startswith('aarch64') else 'v3')]
assert selected
for mutation in [False, True] if a.target.startswith('x86_64') else [False]:
    text = '#![allow(dead_code, unused_variables)]\n'
    expected_lines = set()
    for i, c in enumerate(selected):
        text += '#[archmage::rite(' + ('v2' if mutation else c['tier']) + ')]\n'
        text += f'unsafe extern "C" fn contract_{i}() {{\n' + c['statement'] + '\n}\n'
        expected_lines.add(text.count('\n') - 1)
    (work / 'src/lib.rs').write_text(text)
    command = ['cargo', 'check', '--manifest-path', str(work / 'Cargo.toml'),
               '--target', a.target, '--message-format=json']
    result = subprocess.run(command, capture_output=True, text=True)
    label = ('weakened' if mutation else 'valid') + '-' + a.target
    (work / (label + '.stdout')).write_text(result.stdout)
    (work / (label + '.stderr')).write_text(result.stderr)
    if not mutation:
        assert result.returncode == 0, result.stderr + result.stdout
    else:
        errors = [r['message'] for line in result.stdout.splitlines()
                  if (r := json.loads(line)).get('reason') == 'compiler-message'
                  and (r['message'].get('code') or {}).get('code') == 'E0133']
        rejected = {span['line_start'] for error in errors for span in error['spans'] if span['is_primary']}
        assert result.returncode != 0 and rejected == expected_lines, (len(errors), rejected ^ expected_lines)
    print(label, len(selected), 'PASS', flush=True)
