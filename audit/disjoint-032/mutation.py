#!/usr/bin/env python3
"""Challenge the 0.3 tracker model using a separate, deliberately broken copy."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--candidate", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
source = args.candidate / "crates/rav1d-disjoint-mut"
scratch = args.output / "mutation-no-mutable-scan"
# Never replace a previous experiment or mutate the maintenance checkout.
shutil.copytree(source, scratch)
path = scratch / "src/lib.rs"
original = path.read_text()
before = "match slots.find_overlap_any(start, end) {"
assert original.count(before) == 1
mutated = original.replace(before, "match None::<(usize, usize, bool)> {")
path.write_text(mutated)
env = os.environ.copy()
env.pop("CARGO_ENCODED_RUSTFLAGS", None)
env["RUSTFLAGS"] = "--cfg disjoint_mut_loom"
env["CARGO_TERM_COLOR"] = "never"
env["CARGO_TARGET_DIR"] = str(args.candidate / "target/loom-mutation")
command = ["cargo", "test", "--manifest-path", str(scratch / "Cargo.toml"), "--lib",
           "loom_032::inline_exclusion_reuse_and_payload_handoff", "--", "--test-threads=1"]
log = args.output / "mutation.log"
with log.open("w") as output:
    result = subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT)
text = log.read_text()
passed = result.returncode != 0 and any(message in text for message in
    ["Causality violation", "currently writing to cell", "Concurrent write accesses"])
record = dict(command=command, env={k: env[k] for k in ["RUSTFLAGS", "CARGO_TARGET_DIR"]},
              exit_code=result.returncode, detected_payload_violation=passed,
              baseline_lib_sha256=hashlib.sha256(original.encode()).hexdigest(),
              mutated_lib_sha256=hashlib.sha256(mutated.encode()).hexdigest(),
              original_checkout_unchanged=(source / "src/lib.rs").read_text() == original,
              log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
(args.output / "mutation.json").write_text(json.dumps(record, indent=2)+"\n")
print(json.dumps(record), flush=True)
if not passed or not record["original_checkout_unchanged"]:
    print(text[-5000:])
    raise SystemExit(1)
