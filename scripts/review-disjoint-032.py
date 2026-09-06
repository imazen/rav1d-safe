#!/usr/bin/env python3
"""Serial, recorded release gates for the 0.3.2 maintenance branch.

Run through the workspace run-heavy wrapper on shared machines. Baseline must
be the checksum-verified extracted crates.io 0.3.1, not a guessed git snapshot.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--group", choices=["native", "loom", "miri", "semver", "package"], required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--baseline", type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
args.output.mkdir(parents=True, exist_ok=True)
base_env = os.environ.copy()
base_env.pop("RUSTFLAGS", None)
base_env.pop("CARGO_ENCODED_RUSTFLAGS", None)
base_env.pop("MIRIFLAGS", None)
base_env["CARGO_TERM_COLOR"] = "never"
base_env["CARGO_BUILD_JOBS"] = "8"
package = ["-p", "rav1d-disjoint-mut"]
gates = []
if args.group == "native":
    gates = [
        ("fmt", ["cargo", "fmt", *package, "--", "--check"], {}),
        ("all-targets", ["cargo", "test", *package, "--all-features", "--all-targets", "--no-fail-fast"], {}),
        ("no-std", ["cargo", "test", *package, "--no-default-features", "--no-fail-fast"], {}),
        ("no-std-storage", ["cargo", "test", *package, "--no-default-features", "--features", "aligned,pic-buf,zerocopy", "--no-fail-fast"], {}),
        ("doctests", ["cargo", "test", *package, "--all-features", "--doc"], {}),
        ("clippy", ["cargo", "clippy", *package, "--all-features", "--all-targets", "--", "-D", "warnings"], {}),
        ("clippy-no-std", ["cargo", "clippy", *package, "--no-default-features", "--", "-D", "warnings"], {}),
        ("docs", ["cargo", "doc", *package, "--all-features", "--no-deps"], {"RUSTDOCFLAGS": "-D warnings"}),
    ]
elif args.group == "loom":
    gates = [("loom", ["cargo", "test", *package, "--lib", "loom_032", "--", "--test-threads=1"],
              {"RUSTFLAGS": "--cfg disjoint_mut_loom", "CARGO_TARGET_DIR": str(root / "target/loom")})]
elif args.group == "miri":
    for model, flags in [("stacked", ""), ("tree", "-Zmiri-tree-borrows")]:
        gates.append((model, ["cargo", "+nightly", "miri", "test", *package, "--all-features", "--no-fail-fast"], {"MIRIFLAGS": flags}))
        gates.append((model + "-no-std", ["cargo", "+nightly", "miri", "test", *package, "--no-default-features", "--test", "patch_032"], {"MIRIFLAGS": flags}))
elif args.group == "semver":
    if not args.baseline:
        parser.error("--baseline is required for semver")
    for name, features in [("default", ["--default-features"]), ("no-std", ["--only-explicit-features"]),
                           ("all", ["--only-explicit-features", "--features", "std,instrument,aligned,pic-buf,zerocopy"])]:
        gates.append((name, ["cargo", "semver-checks", *package, "--baseline-root", str(args.baseline.resolve()), "--release-type", "patch", *features], {}))
elif args.group == "package":
    manifest = root / "target/package/rav1d-disjoint-mut-0.3.2/Cargo.toml"
    gates = [
        ("package", ["cargo", "package", *package, "--all-features", "--allow-dirty"], {}),
        ("msrv", ["cargo", "+1.85.0", "check", "--manifest-path", str(manifest), "--all-features"], {}),
        ("msrv-no-std", ["cargo", "+1.85.0", "check", "--manifest-path", str(manifest), "--no-default-features"], {}),
        ("i686", ["cargo", "check", "--manifest-path", str(manifest), "--all-features", "--target", "i686-unknown-linux-gnu"], {}),
        ("wasm-no-std", ["cargo", "check", "--manifest-path", str(manifest), "--no-default-features", "--target", "wasm32-unknown-unknown"], {}),
    ]

records = []
for name, command, extra_env in gates:
    log = args.output / (args.group + "-" + name + ".log")
    print("RUN", name, command, flush=True)
    start = time.monotonic()
    with log.open("w") as output:
        result = subprocess.run(command, cwd=root, env=base_env | extra_env, stdout=output, stderr=subprocess.STDOUT)
    record = dict(gate=name, command=command, env=extra_env, exit_code=result.returncode,
                  seconds=round(time.monotonic()-start, 2), log=log.name,
                  log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
                  finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    records.append(record)
    (args.output / (args.group + ".json")).write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    if result.returncode:
        print(log.read_text()[-8000:], flush=True)
        raise SystemExit(result.returncode)
