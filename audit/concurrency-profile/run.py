#!/usr/bin/env python3
"""Serialized matrix. Run under run-heavy. Raw logs and perf.data stay in scratch."""
import datetime
import gzip
import json
import math
import os
import pathlib
import statistics
import subprocess
import sys
import time

repo = pathlib.Path(__file__).resolve().parents[2]
out = pathlib.Path(sys.argv[1]).resolve()
phase = sys.argv[2]
inputs = {
    "multi": repo / "test-vectors/dav1d-test-data/8-bit/features/non_uniform_tiling.ivf",
    "tiled_first": out / "inputs/non_uniform_first.obu",
    "tiled_stress": repo / "tests/crash_vectors/tile_threading_cdef_lpf_race.obu",
    "single_10b": repo / "tests/crash_vectors/lr_sgr_10bpc_noisy_nocdef.obu",
}
cells = [(1,1),(2,1),(4,1),(8,1),(16,1),(24,1),(1,2),(1,4),(1,8),(2,4),(4,4),(8,4)]
(out / phase).mkdir(exist_ok=True)
records = []

def mark():
    (repo / ".workongoing").write_text(datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") + f" codex-borrow-profile running {phase}\n")

def run(arm, name, threads, instances, passes, reps=1, suffix="", prime=None, perf=None):
    mark()
    key = f"{name}-t{threads}-i{instances}{suffix}"
    cmd = [str(out / "bin" / arm), str(inputs[name]), str(threads), str(instances), str(passes), str(reps)]
    env = dict(os.environ)
    if prime is not None:
        env["RAV1D_PRIME_THREADS"] = str(prime)
    if perf:
        ctl, ack = out / phase / f"{key}.ctl", out / phase / f"{key}.ack"
        for p in (ctl, ack):
            if p.exists(): p.unlink()
            os.mkfifo(p)
        env["RAV1D_PERF_CONTROL"] = f"{ctl},{ack}"
        control = ["--delay=-1", f"--control=fifo:{ctl},{ack}"]
        if perf == "record":
            cmd = ["perf", "record", "-e", "cycles:u", "-F", "499", "--call-graph", "dwarf,8192", "-o", str(out / phase / f"{key}.data"), *control, "--", *cmd]
        else:
            cmd = ["perf", "stat", "-x", ";", "-o", str(out / phase / f"{key}.stat"), "-e", "task-clock,cycles:u,instructions:u,cache-references:u,cache-misses:u,context-switches,cpu-migrations", *control, "--", *cmd]
    t0 = time.monotonic()
    p = subprocess.run(cmd, cwd=repo, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=100)
    raw = p.stdout + p.stderr
    with gzip.open(out / phase / f"{key}.log.gz", "wt") as f: f.write(raw)
    if p.returncode:
        raise RuntimeError(f"{key}: exit {p.returncode}\n{raw[-5000:]}")
    assert "VALIDATED\t" in p.stdout, key
    timing = [float(s.split("\t")[4]) for s in p.stdout.splitlines() if s.startswith("RESULT\t")]
    rec = dict(phase=phase, arm=arm, input=name, threads=threads, instances=instances, passes=passes, reps=reps, prime=prime, ms_per_frame=timing, median_ms=statistics.median(timing), command=cmd, seconds=time.monotonic()-t0)
    records.append(rec)
    with (out / f"{phase}.jsonl").open("a") as f: f.write(json.dumps(rec) + "\n")
    print(f"{phase} {key}: {rec['median_ms']:.4f} ms/frame ({rec['seconds']:.1f}s)", flush=True)
    if perf:
        ctl.unlink(); ack.unlink()
    return rec

if phase == "smoke":
    for name in inputs:
        run("usage", name, 4, 1, 1)
elif phase == "baseline":
    for name in inputs:
        for threads, instances in cells:
            calibration = run("base", name, threads, instances, 1, suffix="-cal")
            # At least ~300 ms wall per replicate; fixed pass count after calibration.
            raw = gzip.open(out / phase / f"{name}-t{threads}-i{instances}-cal.log.gz", "rt").read()
            frames = int(next(s for s in raw.splitlines() if s.startswith("RESULT\t")).split("\t")[2])
            passes = max(1, min(300, math.ceil(300 / (calibration["median_ms"] * frames))))
            run("base", name, threads, instances, passes, 5)
elif phase in ("tasks", "usage"):
    for name in inputs:
        for threads, instances in [(1,1),(4,1),(8,1),(24,1),(1,4),(8,4)]:
            run(phase, name, threads, instances, 1 if phase == "usage" else (3 if name == "multi" else 20))
elif phase == "history":
    for name in inputs:
        for threads, instances in [(1,1),(4,1)]:
            for round in range(3):
                for prime in ([None,24] if round % 2 == 0 else [24,None]):
                    run("base", name, threads, instances, 8 if name == "multi" else 40, 1, suffix=f"-r{round}-prime{prime}", prime=prime)
elif phase in ("profile", "stat"):
    base = [json.loads(s) for s in (out / "baseline.jsonl").read_text().splitlines()]
    for name in inputs:
        for threads, instances in [(1,1),(8,1),(8,4)]:
            b = next(r for r in base if r["input"]==name and r["threads"]==threads and r["instances"]==instances and r["reps"]==5)
            run("base", name, threads, instances, b["passes"] * (6 if phase=="profile" else 3), perf="record" if phase=="profile" else "stat")
else:
    raise ValueError(phase)
