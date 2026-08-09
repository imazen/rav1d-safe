#!/usr/bin/env python3
"""Turn probe-tasktime's exact interval log into an occupancy/tail report.

Input: the TSV written by `RAV1D_EVLOG=<path>` — one row per stage body, park
and driver-level decode call, as `t0_ns  dur_ns  worker  stage`.

Everything here is derived from exact intervals, not from a sampler, so
"how many workers were inside a body at time X" is answered without a sampling
period and frame boundaries are real rather than inferred.

Reported per frame (medianed across frames):
  head   ns before the first task stage of the frame starts  (serial setup)
  tail   ns after the last task stage of the frame ends      (serial teardown)
  body   the span between them, and the mean worker occupancy inside it
  tile-exhausted tail: the span after the LAST tile_recon ends, which is the
         part of the frame where only the post-tile filter chain can run —
         the Amdahl term the filter-chain hypothesis predicts.
"""

import sys
import statistics as st
from collections import defaultdict

TILE = {"tile_recon", "tile_entropy"}
FILTER = {"deblock_cols", "deblock_rows", "cdef", "superres", "loop_restore"}


def load(path):
    ev = []
    with open(path) as f:
        next(f)
        for line in f:
            t0, dur, w, stage = line.rstrip("\n").split("\t")
            ev.append((int(t0), int(dur), int(w), stage))
    return ev


def occupancy(intervals, lo, hi):
    """Time-weighted mean count of overlapping intervals over [lo, hi), plus the
    total time at each distinct count."""
    pts = []
    for a, b in intervals:
        a, b = max(a, lo), min(b, hi)
        if b > a:
            pts.append((a, 1))
            pts.append((b, -1))
    if not pts:
        return 0.0, {0: hi - lo}
    pts.sort()
    hist = defaultdict(int)
    cur, prev, area = 0, lo, 0
    for t, d in pts:
        if t > prev:
            hist[cur] += t - prev
            area += cur * (t - prev)
            prev = t
        cur += d
    if hi > prev:
        hist[cur] += hi - prev
    span = hi - lo
    return (area / span if span else 0.0), dict(hist)


def main(path, label):
    ev = load(path)
    frames = sorted((t0, t0 + d) for t0, d, _, s in ev if s == "frame")
    stages = [e for e in ev if e[3] not in ("frame", "park")]
    if not frames:
        print(f"{label}: no frame marks")
        return

    rows = []
    for fi, (f0, f1) in enumerate(frames):
        mine = [e for e in stages if e[0] >= f0 and e[0] < f1]
        if not mine:
            continue
        first = min(e[0] for e in mine)
        last = max(e[0] + e[1] for e in mine)
        tile_last = max(
            (e[0] + e[1] for e in mine if e[3] in TILE), default=first
        )
        tile_iv = [(e[0], e[0] + e[1]) for e in mine if e[3] in TILE]
        filt_iv = [(e[0], e[0] + e[1]) for e in mine if e[3] in FILTER]
        all_iv = tile_iv + filt_iv
        occ_body, hist = occupancy(all_iv, first, last)
        occ_frame, _ = occupancy(all_iv, f0, f1)
        # The post-tile tail: only filter work can run there.
        occ_tail, _ = occupancy(filt_iv, tile_last, last) if last > tile_last else (0.0, {})
        per_stage = defaultdict(int)
        for e in mine:
            per_stage[e[3]] += e[1]
        rows.append(
            dict(
                frame=fi,
                wall=f1 - f0,
                head=first - f0,
                tail_after_last=f1 - last,
                body=last - first,
                occ_body=occ_body,
                occ_frame=occ_frame,
                post_tile=last - tile_last,
                occ_post_tile=occ_tail,
                per_stage=dict(per_stage),
                hist=hist,
            )
        )

    def med(k):
        return st.median(r[k] for r in rows)

    print(f"\n===== {label}  ({len(rows)} frames) =====")
    w = med("wall") / 1e6
    print(f"  frame wall            {w:8.3f} ms   (driver-timed decode call)")
    print(
        f"  serial HEAD           {med('head')/1e6:8.3f} ms  "
        f"{100*med('head')/med('wall'):5.1f}%   before any task stage starts"
    )
    print(
        f"  serial TAIL           {med('tail_after_last')/1e6:8.3f} ms  "
        f"{100*med('tail_after_last')/med('wall'):5.1f}%   after the last task stage ends"
    )
    print(
        f"  parallel BODY         {med('body')/1e6:8.3f} ms  "
        f"{100*med('body')/med('wall'):5.1f}%   mean occupancy {med('occ_body'):.3f}"
    )
    print(
        f"  post-tile TAIL        {med('post_tile')/1e6:8.3f} ms  "
        f"{100*med('post_tile')/med('wall'):5.1f}%   mean filter occupancy {med('occ_post_tile'):.3f}"
    )
    print(f"  occupancy over the whole frame window: {med('occ_frame'):.3f}")

    tot = defaultdict(list)
    for r in rows:
        for k, v in r["per_stage"].items():
            tot[k].append(v)
    print("  CPU ms/frame by stage:")
    s_all = 0.0
    for k in sorted(tot, key=lambda k: -st.median(tot[k])):
        m = st.median(tot[k]) / 1e6
        s_all += m
        print(f"      {k:<14} {m:8.3f}")
    print(f"      {'TOTAL':<14} {s_all:8.3f}   -> CPU/wall = {s_all/w:.3f}")

    hist = defaultdict(int)
    for r in rows:
        for k, v in r["hist"].items():
            hist[k] += v
    tot_ns = sum(hist.values())
    print("  time-weighted occupancy histogram inside BODY (exact, not sampled):")
    for k in sorted(hist):
        print(f"      {k:>2} workers  {100*hist[k]/tot_ns:6.2f}%   {hist[k]/1e6/len(rows):7.3f} ms/frame")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p, p.rsplit("/", 1)[-1])
