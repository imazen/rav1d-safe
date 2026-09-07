#!/usr/bin/env python3
"""Build compact summaries, retaining every timing replicate and raw census log."""
import collections
import gzip
import json
import pathlib
import statistics
import subprocess
import sys

out = pathlib.Path(sys.argv[1]).resolve()
summary = {}
for phase in ["baseline", "history", "tasks", "usage"]:
    records = [json.loads(s) for s in (out / f"{phase}.jsonl").read_text().splitlines()]
    if phase == "baseline":
        records = [r for r in records if r["reps"] == 5]
    summary[phase] = []
    for r in records:
        key = f"{r['input']}-t{r['threads']}-i{r['instances']}"
        s = {k:r[k] for k in ["input", "threads", "instances", "passes", "reps", "prime", "ms_per_frame", "median_ms"]}
        if phase in ["usage", "tasks"]:
            lines = gzip.open(out / phase / f"{key}.log.gz", "rt").read().splitlines()
            for row in lines:
                a = row.split("\t")
                if a[0] == "VALIDATED":
                    for part in a[2:]:
                        k,v = part.split("="); s[k] = int(v)
                if a[0] == "WIDEHDR": keys = a[1:]
                if a[0] == "WIDE": s["wide"] = dict(zip(keys, map(int,a[1:])))
                if a[0] == "WAIT": s["wait_ns"], s["wait_max_ns"] = map(int,a[1:])
            s["tasks"] = [l for l in lines if l.startswith("PROBE ")]
            s["geometry"] = [l for l in lines if l.startswith("GEOMETRY\t")]
            if phase == "usage":
                total = collections.Counter(); sites = collections.defaultdict(lambda: [0,0,0,0]); occ = {}; policies=[]; sizes=collections.Counter()
                for row in lines:
                    a = row.split("\t")
                    if a[0] == "BORROW":
                        loc, mutable, bkt, rows, shards, shift, n, byte_count, hull = a[1:]
                        n,byte_count,hull = map(int,(n,byte_count,hull))
                        total["registrations"] += n
                        total["mutable"] += n * (mutable == "true")
                        total["one_active_shard"] += n * (shards == "1")
                        # bucket 4 is [8,15]; exact 16 shares bucket 5 with [17,31].
                        total["bytes_under_16"] += n * (int(bkt) <= 4)
                        total["bytes_sum"] += byte_count
                        total["hull_bytes_sum"] += hull
                        total["multirow_rectangle"] += n * (int(rows) > 1)
                        v=sites[loc];v[0]+=n;v[1]+=n*(mutable=="true");v[2]+=byte_count;v[3]+=hull
                        sizes[(bkt, rows, shards, shift)] += n
                    elif a[0] == "POLICY":
                        policies.append(a[1:])
                        if a[1] == "new":
                            total["tracker_constructions"] += int(a[6])
                            total["tracker_fixed_bytes_constructed"] += int(a[7])
                    elif a[0] == "OCCUPANCY": occ[a[1]] = int(a[2])
                s["totals"]=dict(total);s["occupancy"]=occ
                s["top_sites_by_count"]=sorted(sites.items(),key=lambda kv:kv[1][0],reverse=True)[:20]
                s["top_sites_by_bytes"]=sorted(sites.items(),key=lambda kv:kv[1][2],reverse=True)[:12]
                s["top_shapes"]=sizes.most_common(16)
                s["policies"]=policies
        summary[phase].append(s)

for phase, rows in summary.items():
    with (out / f"summary-{phase}.jsonl").open("w") as f:
        for s in rows: f.write(json.dumps(s)+"\n")
    print(phase,len(rows))

for data in sorted((out / "profile").glob("*.data")):
    for style, args in [("self", ["--no-children", "--sort", "dso,symbol", "--percent-limit", "0.3"]), ("calls", ["--children", "--sort", "symbol", "--percent-limit", "1", "--call-graph", "graph,1,caller"])]:
        cmd = ["perf", "report", "--stdio", "-i", str(data), *args]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        text = p.stdout + "\n" + p.stderr
        with gzip.open(data.with_suffix(f".{style}.txt.gz"), "wt") as f: f.write(text)
        if style == "self": print(data.stem, "\n" + "\n".join(l for l in p.stdout.splitlines() if l.strip() and not l.startswith('#'))[:2200])
