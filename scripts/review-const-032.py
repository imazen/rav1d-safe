#!/usr/bin/env python3
"""Serialize release gates for const-compatible 0.3.2 on the current tracker.

Run under scripts/run-heavy from zen-workspace. Each command, environment,
result and log hash is saved; failures stop the group. No all-features build:
historical safety-disabling probes deliberately fail compilation.
"""
import argparse
import datetime
import difflib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import tomllib

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("group", choices=["native", "miri", "miri-focused", "loom", "semver", "api", "decoder", "package", "downstream"])
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--baseline", type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parent.parent
os.chdir(root)
args.output = args.output.resolve()
args.output.mkdir(parents=True, exist_ok=True)
features = "aligned,pic-buf,zerocopy"
package = ["-p", "rav1d-disjoint-mut"]
gates = []


def add(name, command, env=None):
    gates.append((name, command, env or {}))


if args.group == "native":
    add("native", ["cargo", "test", *package, "--features", features, "--all-targets"])
    add("no-std-storage", ["cargo", "test", *package, "--no-default-features", "--features", features])
    add("docs", ["cargo", "test", *package, "--features", features, "--doc"])
    add("clippy", ["cargo", "clippy", *package, "--all-targets", "--features", features, "--", "-D", "warnings"])
    add("clippy-no-std", ["cargo", "clippy", *package, "--no-default-features", "--features", features, "--lib", "--", "-D", "warnings"])
    add("i686", ["cargo", "check", *package, "--features", features, "--target", "i686-unknown-linux-gnu"])
    add("wasm-no-std", ["cargo", "check", *package, "--no-default-features", "--target", "wasm32-unknown-unknown"])
elif args.group == "miri":
    for model, flags in [("stacked", ""), ("tree", "-Zmiri-tree-borrows")]:
        add(f"miri-{model}", ["cargo", "+nightly", "miri", "test", *package, "--features", features, "--no-fail-fast"], {"MIRIFLAGS": flags})
        add(f"miri-{model}-no-std", ["cargo", "+nightly", "miri", "test", *package, "--no-default-features", "--test", "const_constructor"], {"MIRIFLAGS": flags})
elif args.group == "miri-focused":
    targets = ["adversarial_api", "aligned_miri", "const_constructor", "guard_move_release",
               "pic_buf_overflow", "rect_units", "soundness", "wide_exclusion"]
    for model, flags in [("stacked", ""), ("tree", "-Zmiri-tree-borrows")]:
        add(f"miri-{model}-init", ["cargo", "+nightly", "miri", "test", *package, "--lib", "tracker_storage"], {"MIRIFLAGS": flags})
        add(f"miri-{model}-focused", ["cargo", "+nightly", "miri", "test", *package, "--features", features, "--no-fail-fast",
            *[arg for target in targets for arg in ["--test", target]]], {"MIRIFLAGS": flags})
        add(f"miri-{model}-no-std-focused", ["cargo", "+nightly", "miri", "test", *package, "--no-default-features", "--test", "const_constructor"], {"MIRIFLAGS": flags})
elif args.group == "loom":
    add("loom", ["cargo", "test", *package, "--features", "__shards_4", "--lib", "loom_protocol", "--", "--test-threads=1"],
        {"RUSTFLAGS": "--cfg disjoint_mut_loom", "CARGO_TARGET_DIR": str(root / "target/review-loom")})
    add("safe-feature-boundary", ["bash", "scripts/review-feature-gate.sh"])
elif args.group == "semver":
    assert args.baseline, "--baseline must point at verified published 0.3.1"
    for name, flags in [("default", ["--default-features"]), ("no-std", ["--only-explicit-features"]),
                        ("storage", ["--only-explicit-features", "--features", "std,instrument," + features])]:
        add(f"semver-{name}", ["cargo", "semver-checks", *package, "--baseline-root", str(args.baseline), "--release-type", "patch", *flags])
elif args.group == "api":
    assert args.baseline, "--baseline must point at verified published 0.3.1"
    for version, manifest in [("0.3.1", args.baseline / "Cargo.toml"), ("0.3.2", root / "Cargo.toml")]:
        add(f"api-{version}", ["cargo", "+nightly", "public-api", "--manifest-path", str(manifest), *package,
            "--omit", "blanket-impls", "--color", "never", "--features", features],
            {"CARGO_TARGET_DIR": str(root / "target/review-api")})
elif args.group == "decoder":
    add("decoder-regressions", ["cargo", "nextest", "run", "-p", "rav1d-safe", "--test", "decode_md5_committed", "--test", "safe_simd_crashes", "--test", "fuzz_regression", "--test-threads", "1"])
    add("decoder-task-size", ["cargo", "test", "-p", "rav1d-safe", "--lib", "task_context_stays_stack_light", "--", "--nocapture"])
    add("decoder-global-threading", ["cargo", "test", "-p", "rav1d-safe", "--lib", "live_block_keeps_its_storage_when_another_decoder_changes_threading"])
elif args.group == "package":
    add("msrv-const-client", ["cargo", "+1.85.0", "test", *package, "--no-default-features", "--test", "const_constructor"])
    add("package", ["cargo", "package", *package, "--features", features])
    manifest = root / "target/package/rav1d-disjoint-mut-0.3.2/Cargo.toml"
    for name, flags in [("storage", ["--features", "std,instrument," + features]), ("no-std", ["--no-default-features", "--features", features])]:
        add(f"msrv-{name}", ["cargo", "+1.85.0", "check", "--manifest-path", str(manifest), *flags])
elif args.group == "downstream":
    manifest = root / "audit/disjoint-032-current/downstream/Cargo.toml"
    patched = root / "target/package/rav1d-disjoint-mut-0.3.2"
    config = "patch.crates-io.rav1d-disjoint-mut.path=" + json.dumps(str(patched))
    add("downstream", ["cargo", "test", "--manifest-path", str(manifest), "--config", config, "--test", "decode"],
        {"CARGO_TARGET_DIR": str(root / "target/review-downstream")})

results = []
for name, command, overrides in gates:
    env = os.environ.copy()
    for key in ["RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "MIRIFLAGS"]:
        env.pop(key, None)
    env.update(overrides)
    env.update({"CARGO_BUILD_JOBS": "8", "CARGO_TERM_COLOR": "never"})
    print(f"START {name}: {command}", flush=True)
    start = time.monotonic()
    log = args.output / f"{name}.log"
    with log.open("w") as output, (args.output / f"{name}.stderr").open("w") as errors:
        proc = subprocess.run(command, env=env, stdout=output,
                              stderr=errors if args.group == "api" else subprocess.STDOUT)
    record = {"gate": name, "command": command, "env": overrides, "exit_code": proc.returncode,
              "seconds": round(time.monotonic() - start, 2), "log": log.name,
              "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
              "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    results.append(record)
    (args.output / f"{args.group}.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"END {name}: exit={proc.returncode}, seconds={record['seconds']}, log={log}", flush=True)
    if proc.returncode:
        print(log.read_text()[-8000:], flush=True)
        raise SystemExit(proc.returncode)

if args.group == "api":
    snapshots = [(args.output / f"api-{v}.log").read_text().splitlines(keepends=True) for v in ["0.3.1", "0.3.2"]]
    (args.output / "api.diff").write_text("".join(difflib.unified_diff(*snapshots, fromfile="0.3.1", tofile="0.3.2")))
    old = tomllib.loads((args.baseline / "Cargo.toml.orig").read_text())
    new = tomllib.loads((root / "crates/rav1d-disjoint-mut/Cargo.toml").read_text())
    assert all(new["features"][k] == v for k, v in old["features"].items()), "published feature contract changed"
    assert new["package"]["rust-version"] == old["package"]["rust-version"]
    (args.output / "feature-contract.json").write_text(json.dumps({
        "published_features_preserved": old["features"],
        "added_features": {k: v for k, v in new["features"].items() if k not in old["features"]},
        "dependencies_before": old["dependencies"], "dependencies_after": new["dependencies"],
        "rust-version": new["package"]["rust-version"],
    }, indent=2) + "\n")
elif args.group == "downstream":
    metadata = json.loads(subprocess.check_output(["cargo", "metadata", "--format-version", "1",
        "--manifest-path", str(manifest), "--config", config], text=True))
    selected = [{k: p[k] for k in ["name", "version", "source", "manifest_path"]}
                for p in metadata["packages"] if p["name"] in ["rav1d-safe", "rav1d-disjoint-mut"]]
    assert {(p["name"], p["version"]) for p in selected} == {("rav1d-safe", "0.5.7"), ("rav1d-disjoint-mut", "0.3.2")}
    assert next(p["manifest_path"] for p in selected if p["name"] == "rav1d-disjoint-mut") == str(patched / "Cargo.toml")
    (args.output / "downstream-resolution.json").write_text(json.dumps(selected, indent=2) + "\n")
