#!/usr/bin/env python3
"""Preserve benchmark bytes in bounded parts with a verified hash manifest."""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def fingerprint(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True,
                        help="reviewed JSON provenance to extend with artifact hashes")
    parser.add_argument("--input", action="append", required=True,
                        help="artifact basename=source path; repeat for each raw artifact")
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text())
    if not isinstance(metadata, dict):
        parser.error("metadata must be a JSON object")
    if "files" in metadata:
        parser.error("metadata already contains an artifact manifest")
    planned = {}
    manifest = {}
    for item in args.input:
        name, separator, source = item.partition("=")
        if not separator or Path(name).name != name or name in ("", ".", ".."):
            parser.error("each input must use a plain artifact basename=source path")
        if name in manifest:
            parser.error(f"duplicate artifact: {name}")
        data = Path(source).read_bytes()
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            parser.error(f"input is not UTF-8 text: {source}")
        parts = []
        start = 0
        while start < len(data):
            end = min(start + 28000, len(data))
            while end < len(data) and data[end] & 0xc0 == 0x80:
                end -= 1
            parts.append(data[start:end])
            start = end
        parts = parts or [b""]
        names = [name] if len(parts) == 1 else [
            f"{Path(name).stem}.part{i + 1:02d}{Path(name).suffix}" for i in range(len(parts))
        ]
        manifest[name] = {**fingerprint(data), "parts": []}
        for part_name, part in zip(names, parts):
            if part_name in planned or part_name == "artifacts.meta.json":
                parser.error(f"artifact path collision: {part_name}")
            planned[part_name] = part
            manifest[name]["parts"].append({"path": part_name, **fingerprint(part)})
        assert b"".join(parts) == data
    metadata["files"] = manifest
    metadata["recording_command"] = ["python3", "tools/record_benchmark_artifacts.py", *sys.argv[1:]]
    encoded = (json.dumps(metadata, indent=2) + "\n").encode()
    if len(encoded) > 30000:
        parser.error("metadata exceeds 30000 bytes; use a smaller reviewed provenance file")
    planned["artifacts.meta.json"] = encoded
    # Validate every destination before writing; never replace protected data.
    for name in planned:
        path = args.output / name
        if path.exists() or path.is_symlink():
            parser.error(f"destination already exists: {path}")
    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in planned.items():
        with (args.output / name).open("xb") as target:
            target.write(data)
    for original, entry in manifest.items():
        data = b"".join((args.output / p["path"]).read_bytes() for p in entry["parts"])
        assert fingerprint(data) == {k: entry[k] for k in ("bytes", "sha256")}, original
    print(json.dumps({"output": str(args.output), "artifacts": len(manifest),
                      "files": len(planned), "verified": True}))


if __name__ == "__main__":
    main()
