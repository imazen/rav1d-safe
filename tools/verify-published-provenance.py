#!/usr/bin/env python3
"""Map every crates.io release back to the git commit it was published from.

Complements tools/check-package-source.py, which inspects what *would* be
packaged before publication. This checks what actually reached the registry:
for each published version it reads the `.cargo_vcs_info.json` that Cargo embeds
in the archive, then compares every packaged file byte-for-byte against
`git show <sha>:<path>`. A release is only reported as reproduced when the
recorded commit regenerates its packaged source exactly.

Cargo rewrites `Cargo.toml` and regenerates `Cargo.lock` at package time, so
neither can be compared against the tree; `Cargo.toml.orig` preserves the
committed manifest and is compared in its place.

    python3 tools/verify-published-provenance.py
    python3 tools/verify-published-provenance.py --crate rav1d-safe --check-tags
"""
import argparse
import json
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

UA = 'imazen rav1d-safe release provenance check'
# Regenerated or rewritten by Cargo at package time; not comparable to the tree.
GENERATED = {'Cargo.toml', 'Cargo.lock', '.cargo_vcs_info.json', 'Cargo.toml.orig'}


def fetch(url, binary=False):
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read() if binary else json.load(r)


def versions(crate):
    data = fetch(f'https://crates.io/api/v1/crates/{crate}')
    return sorted(data['versions'], key=lambda v: v['created_at'])


def git(root, *args):
    return subprocess.run(['git', '-C', str(root)] + list(args), capture_output=True)


def tag_name(crate, version):
    """rav1d-safe owns the bare vX.Y.Z namespace; members are prefixed."""
    return f'v{version}' if crate == 'rav1d-safe' else f'{crate}-v{version}'


def verify(root, crate, version, cache):
    archive = cache / f'{crate}-{version}.crate'
    if not archive.exists():
        archive.write_bytes(fetch(
            f'https://static.crates.io/crates/{crate}/{crate}-{version}.crate', binary=True))
    base = f'{crate}-{version}'
    with tarfile.open(archive, 'r:gz') as tf:
        info = json.load(tf.extractfile(f'{base}/.cargo_vcs_info.json'))
        sha = info['git']['sha1']
        prefix = info.get('path_in_vcs') or ''
        result = {'crate': crate, 'version': version, 'sha': sha,
                  'dirty': info['git'].get('dirty', False),
                  'same': 0, 'differs': [], 'absent': [], 'manifest': 'absent'}
        for name in tf.getnames():
            rel = name[len(base) + 1:]
            member = tf.getmember(name)
            if not rel or not member.isfile() or rel in GENERATED:
                continue
            want = tf.extractfile(name).read()
            path = f'{prefix}/{rel}' if prefix else rel
            got = git(root, 'show', f'{sha}:{path}')
            if got.returncode != 0:
                result['absent'].append(rel)
            elif got.stdout == want:
                result['same'] += 1
            else:
                result['differs'].append(rel)
        orig = f'{base}/Cargo.toml.orig'
        if orig in tf.getnames():
            want = tf.extractfile(orig).read()
            path = f'{prefix}/Cargo.toml' if prefix else 'Cargo.toml'
            got = git(root, 'show', f'{sha}:{path}')
            result['manifest'] = 'match' if (got.returncode == 0 and got.stdout == want) else 'differs'
    # Files present in the archive but not the commit are untracked leftovers from a
    # dirty publish worktree; they mean the commit does not fully reproduce the release.
    result['reproduced'] = (not result['differs'] and not result['absent']
                            and result['manifest'] == 'match')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--crate', action='append', dest='crates',
                   help='Crate to check (repeatable). Default: both published crates.')
    p.add_argument('--cache-dir', type=Path, default=Path.home() / 'tmp' / 'rav1d-release-audit')
    p.add_argument('--check-tags', action='store_true',
                   help='Also report whether a release tag exists at the publishing commit.')
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    crates = a.crates or ['rav1d-safe', 'rav1d-disjoint-mut']
    a.cache_dir.mkdir(parents=True, exist_ok=True)

    tags = {}
    if a.check_tags:
        out = git(root, 'for-each-ref', '--format=%(refname:short) %(objectname)', 'refs/tags')
        for line in out.stdout.decode().splitlines():
            name, sha = line.split()
            tags[name] = sha

    header = f"{'crate':<20}{'version':<9}{'commit':<12}{'files':>6}{'dirty':>7}  {'status'}"
    if a.check_tags:
        header += '   tag'
    print(header)
    print('-' * (len(header) + 20))
    failures = 0
    for crate in crates:
        for v in versions(crate):
            if v['yanked']:
                pass  # Yanked releases still shipped; their provenance still matters.
            r = verify(root, crate, v['num'], a.cache_dir)
            status = 'reproduced' if r['reproduced'] else 'DIFFERS'
            line = (f"{r['crate']:<20}{r['version']:<9}{r['sha'][:10]:<12}"
                    f"{r['same']:>6}{str(r['dirty']):>7}  {status}")
            if a.check_tags:
                name = tag_name(crate, v['num'])
                have = tags.get(name)
                if have is None:
                    line += f'   MISSING {name}'
                    failures += 1
                elif have != r['sha']:
                    line += f'   {name} -> {have[:10]} (differs)'
                else:
                    line += f'   {name}'
            print(line)
            for path in r['differs']:
                print(f'      differs: {path}')
            for path in r['absent']:
                print(f'      not in commit: {path}')
            if not r['reproduced']:
                failures += 1
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
