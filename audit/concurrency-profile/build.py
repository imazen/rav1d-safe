#!/usr/bin/env python3
"""Run under run-heavy; builds sequentially with the repository release profile."""
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys

repo = pathlib.Path(__file__).resolve().parents[2]
out = pathlib.Path(sys.argv[1]).resolve()
(out / "bin").mkdir(parents=True, exist_ok=True)
records = []
for arm, features in [("base", ""), ("usage", "probe-usage,probe-tasktime,probe-wide"), ("tasks", "probe-tasktime,probe-wide")]:
    cmd = ["cargo", "build", "--release", "--locked", "--example", "profile_concurrency"]
    if features:
        cmd += ["--features", features]
    with (out / f"build-{arm}.log").open("w") as log:
        subprocess.run(cmd, cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True)
    binary = out / "bin" / arm
    shutil.copy2(repo / "target/release/examples/profile_concurrency", binary)
    records.append(dict(arm=arm, command=cmd, sha256=hashlib.sha256(binary.read_bytes()).hexdigest()))
    print(f"built {arm}", flush=True)
(out / "builds.json").write_text(json.dumps(records, indent=2) + "\n")
