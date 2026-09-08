#!/usr/bin/env python3
"""Restore the pinned external benchmark evidence, verifying every byte.

No credentials or third-party packages required. Existing mismatched files are
never overwritten. --check-only verifies the bundle without restoring files.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
import urllib.request


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verified_files(data, manifest):
    archive = manifest['archive']
    require(len(data) == archive['bytes'], 'Archive length mismatch')
    require(digest(data) == archive['sha256'], 'Archive SHA256 mismatch')
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        require(len(names) == len(set(names)), 'Duplicate archive members')
        require(all(m.isfile() for m in members), 'Non-file archive member')
        index_record = manifest['index']
        index_data = tar.extractfile(index_record['member']).read()
        require(len(index_data) == index_record['bytes'], 'Index length mismatch')
        require(digest(index_data) == index_record['sha256'], 'Index SHA256 mismatch')
        index = json.loads(index_data)
        require(index['source_commit'] == manifest['source_commit'], 'Source commit mismatch')
        records = index['files']
        paths = [r['path'] for r in records]
        require(len(paths) == len(set(paths)) == manifest['artifact_count'], 'Artifact count mismatch')
        require(set(names) == set(paths) | {index_record['member']}, 'Unexpected archive members')
        require(sum(r['bytes'] for r in records) == manifest['artifact_bytes'], 'Artifact size mismatch')
        files = {}
        for record in records:
            path = PurePosixPath(record['path'])
            require(not path.is_absolute() and '..' not in path.parts
                    and path.parts[0] == 'benchmarks' and path.suffix == '.gz',
                    'Invalid artifact path: ' + str(path))
            content = tar.extractfile(record['path']).read()
            require(len(content) == record['bytes'] and digest(content) == record['sha256'],
                    'Artifact hash/length mismatch: ' + str(path))
            files[str(path)] = content
        return files


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=root / 'benchmarks/EXTERNAL_ARTIFACTS.json')
    parser.add_argument('--destination', type=Path, default=root)
    parser.add_argument('--archive', type=Path, help='Use a downloaded bundle instead of fetching R2')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if args.archive:
        data = args.archive.read_bytes()
    else:
        request = urllib.request.Request(manifest['archive']['url'],
                                         headers={'User-Agent': 'rav1d-benchmark-artifacts/1.0'})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read(manifest['archive']['bytes'] + 1)
    files = verified_files(data, manifest)
    if args.check_only:
        print(f'Verified {len(files)} artifacts; archive and individual SHA256s match')
        return
    destination = args.destination.resolve()
    # Preflight every destination before writing any file.
    for name, content in files.items():
        target = destination / name
        require(target.resolve().is_relative_to(destination), 'Destination escapes root: ' + name)
        require(not target.is_symlink(), 'Refusing symlink destination: ' + name)
        if target.exists():
            require(target.is_file() and target.read_bytes() == content,
                    'Refusing to overwrite different contents: ' + str(target))
    written = 0
    for name, content in files.items():
        target = destination / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            written += 1
    print(f'Verified {len(files)} artifacts; restored {written} files under {destination}')


if __name__ == '__main__':
    main()
