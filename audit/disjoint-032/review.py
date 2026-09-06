#!/usr/bin/env python3
"""Extra independent evidence for the 0.3.2 maintenance candidate.

Run serially through run-heavy. This never publishes or alters either crate.
"""
import argparse
import difflib
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import time
import tomllib

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("group", choices=["api", "controls", "matrix", "downstream"])
parser.add_argument("--candidate", required=True, type=Path, help="maintenance repository root")
parser.add_argument("--baseline", required=True, type=Path, help="verified 0.3.1 extracted crate")
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--patched-crate", type=Path, help="packaged crate source for downstream checks")
args = parser.parse_args()
here = Path(__file__).resolve().parent
args.output.mkdir(parents=True, exist_ok=True)
crate = args.candidate / "crates/rav1d-disjoint-mut"
env = os.environ.copy()
env.pop("RUSTFLAGS", None)
env.pop("CARGO_ENCODED_RUSTFLAGS", None)
env.pop("MIRIFLAGS", None)
env["CARGO_TERM_COLOR"] = "never"
records = []

def run(name, command, cwd, flags=None, expected_ub=False, split=False):
    log = args.output / (name + ".log")
    start = time.monotonic()
    print("RUN", name, command, flush=True)
    with log.open("w") as out, (args.output / (name + ".stderr")).open("w") as err:
        result = subprocess.run(command, cwd=cwd, env=env | (flags or {}), stdout=out,
                                stderr=err if split else subprocess.STDOUT)
    ok = result.returncode == 0
    if expected_ub:
        ok = result.returncode != 0 and "error: Undefined Behavior" in log.read_text()
    records.append(dict(name=name, command=command, cwd=str(cwd), env=flags or {},
                        exit_code=result.returncode, expected_ub=expected_ub, passed=ok,
                        seconds=round(time.monotonic()-start,2), log=log.name,
                        sha256=hashlib.sha256(log.read_bytes()).hexdigest()))
    (args.output / (args.group + ".json")).write_text(json.dumps(records, indent=2)+"\n")
    print(json.dumps(records[-1]), flush=True)
    if not ok:
        print(log.read_text()[-5000:], flush=True)
        raise SystemExit(1)
    return log.read_text()

if args.group == "api":
    baseline = tomllib.loads((args.baseline / "Cargo.toml.orig").read_text())
    current = tomllib.loads((crate / "Cargo.toml").read_text())
    for key in ["features", "dependencies"]:
        assert baseline[key] == current[key], f"changed normal {key} contract"
    for key in ["edition", "rust-version"]:
        assert baseline["package"][key] == current["package"][key]
    snapshots = []
    for version, manifest in [("0.3.1", args.baseline / "Cargo.toml"), ("0.3.2", crate / "Cargo.toml")]:
        snapshots.append(run("api-"+version, ["cargo", "+nightly", "public-api", "--manifest-path", str(manifest),
            "-p", "rav1d-disjoint-mut", "--omit", "blanket-impls", "--color", "never", "--all-features"], args.candidate,
            {"CARGO_TARGET_DIR": str(args.candidate / "target/api")}, split=True))
    diff = "".join(difflib.unified_diff(snapshots[0].splitlines(keepends=True), snapshots[1].splitlines(keepends=True),
                                      fromfile="rav1d-disjoint-mut-0.3.1", tofile="rav1d-disjoint-mut-0.3.2"))
    (args.output / "api.diff").write_text(diff)
    (args.output / "feature-contract.json").write_text(json.dumps({"features": current["features"],
        "normal_dependencies": current["dependencies"], "rust-version": current["package"]["rust-version"],
        "identical_to_031": True}, indent=2)+"\n")
elif args.group == "controls":
    for model, flags in [("stacked", ""), ("tree", "-Zmiri-tree-borrows")]:
        for kind in ["mut", "shared"]:
            run("published-"+model+"-"+kind, ["cargo", "+nightly", "miri", "test", "--manifest-path",
                str(here / "controls/Cargo.toml"), "--test", "guard_move_release", "--",
                "--exact", "moving_a_"+kind+"_guard_into_drop_is_not_ub"], here,
                {"MIRIFLAGS": flags}, expected_ub=True)
elif args.group == "downstream":
    patched = args.patched_crate or crate
    config = "patch.crates-io.rav1d-disjoint-mut.path=" + json.dumps(str(patched.resolve()))
    run("downstream", ["cargo", "test", "--manifest-path", str(here / "downstream/Cargo.toml"),
        "--config", config, "--test", "decode"], here,
        {"CARGO_TARGET_DIR": str(args.candidate / "target/downstream")})
    metadata = subprocess.check_output(["cargo", "metadata", "--format-version", "1", "--manifest-path",
        str(here / "downstream/Cargo.toml"), "--config", config], cwd=here, env=env, text=True)
    selected = [p for p in json.loads(metadata)["packages"] if p["name"] in ["rav1d-safe", "rav1d-disjoint-mut"]]
    assert {(p["name"], p["version"]) for p in selected} == {("rav1d-safe", "0.5.7"), ("rav1d-disjoint-mut", "0.3.2")}
    (args.output / "downstream-resolution.json").write_text(json.dumps(selected, indent=2)+"\n")
elif args.group == "matrix":
    names = ["std", "instrument", "aligned", "pic-buf", "zerocopy"]
    combinations = set()
    for bits in itertools.product([False, True], repeat=len(names)):
        features = {name for name, enabled in zip(names, bits) if enabled}
        if "instrument" in features:
            features.add("std")
        combinations.add(tuple(sorted(features)))
    assert len(combinations) == 24
    for i, features in enumerate(sorted(combinations)):
        command = ["cargo", "check", "-p", "rav1d-disjoint-mut", "--no-default-features"]
        if features:
            command += ["--features", ",".join(features)]
        run("features-"+str(i), command, args.candidate)
