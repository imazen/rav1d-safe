#!/usr/bin/env python3
"""Bucket the tiled self-time profiles and diff t=8 against t=1 / t=2.

Two things this does that `bucket_selftime.py` does not, both needed because
these profiles are MULTITHREADED:

  * `sample` samples every thread, including PARKED ones, so `__psynch_cvwait`
    is 37% of the leaves at t=8 and 0% at t=1. Left in the denominator, every
    busy symbol's share is deflated at high thread counts by exactly the amount
    the pool sleeps -- which is the opposite of the quantity being attributed.
    Idle leaves go to their own bucket and every percentage below is normalised
    on BUSY samples only.
  * The `sync` bucket is split out of `tracker`: `TinyLock::lock_slow` is a core
    WAITING for another core's shard, not the tracker's own arithmetic, so it
    answers a different question (contention vs volume).

`sample`'s total sample count scales with thread count (1 ms x N threads), so
compare the PERCENT columns across cells and the absolute columns only within
one.
"""

import glob
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

IDLE = re.compile(r"psynch|semwait|workq_kernreturn|mach_msg|kevent|"
                  r"thread_switch|swtch_pri|__wait4|usleep|nanosleep", re.I)
RULES = [
    ("sync", re.compile(r"TinyLock|lock_slow|park|cond_wait|futex", re.I)),
    ("tracker", re.compile(
        r"DisjointMut|BorrowTracker|ShardGuard|disjoint_mut|"
        r"cast_slice|slice_as|borrow_id|ShardRecs|wide_probe", re.I)),
    ("entropy", re.compile(
        r"msac|decode_coefs|cdf|read_coef|decode_symbol|bool_adapt|"
        r"golomb|decode_b\b|read_mv", re.I)),
    ("lf", re.compile(r"loopfilter|lpf|loop_filter|LfBlock|lf_mask|lf_apply|"
                      r"lf_compact|lf_run", re.I)),
    ("cdef", re.compile(r"cdef", re.I)),
    ("lr", re.compile(r"looprestor|lr_|selfguided|wiener|sgr", re.I)),
    ("kernels", re.compile(
        r"itx|inv_txfm|\bmc_|put_|prep_|ipred|intra_pred|filmgrain|"
        r"film_grain|safe_simd|recon_b|blend|warp|emu_edge", re.I)),
    ("runtime", re.compile(
        r"libsystem|malloc|free|memcpy|memset|bzero|_platform|nanov2|"
        r"pthread|kernel|mach_|szone|calloc|realloc", re.I)),
]
ORDER = ["entropy", "kernels", "lf", "cdef", "lr", "tracker", "sync",
         "runtime", "other"]


def bucket(sym):
    for name, rx in RULES:
        if rx.search(sym):
            return name
    return "other"


def selftime(path):
    out = subprocess.run(
        [sys.executable, os.path.join(HERE, "sample_selftime.py"), path,
         "--demangle"],
        capture_output=True, text=True).stdout
    rows = []
    for line in out.split("\n"):
        f = line.rstrip("\n").split("\t")
        if len(f) < 2:
            continue
        try:
            n = float(f[0].rstrip("%"))
        except ValueError:
            continue
        rows.append((n, f[-1].strip()))
    return rows


def main():
    outdir = sys.argv[1]
    cells = {}
    for p in sorted(glob.glob(os.path.join(outdir, "*.sample"))):
        tag = os.path.basename(p)[:-len(".sample")]
        rows = selftime(p)
        idle = sum(n for n, s in rows if IDLE.search(s))
        busy = sum(n for n, s in rows if not IDLE.search(s))
        per = {}
        syms = {}
        for n, s in rows:
            if IDLE.search(s):
                continue
            b = bucket(s)
            per[b] = per.get(b, 0.0) + n
            syms[s] = syms.get(s, 0.0) + n
        cells[tag] = {"idle": idle, "busy": busy, "per": per, "syms": syms}

    print("=" * 118)
    print("BUCKETED SELF TIME, normalised on BUSY samples (parked threads "
          "excluded and reported separately as idle%)")
    print("=" * 118)
    hdr = f"{'cell':30} {'busy':>8} {'idle%':>6} " + \
          " ".join(f"{b:>8}" for b in ORDER)
    print(hdr)
    for tag in sorted(cells):
        c = cells[tag]
        tot = c["busy"] or 1
        cols = " ".join(f"{100 * c['per'].get(b, 0) / tot:8.2f}" for b in ORDER)
        print(f"{tag:30} {c['busy']:8.0f} "
              f"{100 * c['idle'] / (c['idle'] + c['busy']):6.2f} {cols}")
    print()

    # t8 vs t1 and t8 vs t2 per vector, in busy-normalised percentage points.
    vecs = sorted({t.rsplit("__t", 1)[0] for t in cells})
    for vec in vecs:
        for lo in ("1", "2"):
            a, b = f"{vec}__t{lo}", f"{vec}__t8"
            if a not in cells or b not in cells:
                continue
            print("=" * 118)
            print(f"DELTA in busy-normalised share: {vec}   t={lo} -> t=8   "
                  f"(pp = percentage points of busy self time)")
            print("=" * 118)
            print(f"{'bucket':10} {'t' + lo + '%':>8} {'t8%':>8} {'pp':>8} "
                  f"{'x':>7}")
            for k in ORDER:
                pa = 100 * cells[a]["per"].get(k, 0) / (cells[a]["busy"] or 1)
                pb = 100 * cells[b]["per"].get(k, 0) / (cells[b]["busy"] or 1)
                if pa == 0 and pb == 0:
                    continue
                print(f"{k:10} {pa:8.2f} {pb:8.2f} {pb - pa:+8.2f} "
                      f"{(pb / pa if pa else float('nan')):7.2f}")
            # the individual leaves that moved most
            allsyms = set(cells[a]["syms"]) | set(cells[b]["syms"])
            movers = []
            for s in allsyms:
                pa = 100 * cells[a]["syms"].get(s, 0) / (cells[a]["busy"] or 1)
                pb = 100 * cells[b]["syms"].get(s, 0) / (cells[b]["busy"] or 1)
                movers.append((pb - pa, pa, pb, s))
            movers.sort(reverse=True)
            print("  top risers (pp):")
            for d, pa, pb, s in movers[:8]:
                if d < 0.10:
                    break
                print(f"    {d:+6.2f}  {pa:5.2f} -> {pb:5.2f}  {s[:88]}")
            print("  top fallers (pp):")
            for d, pa, pb, s in movers[-6:][::-1]:
                if d > -0.10:
                    continue
                print(f"    {d:+6.2f}  {pa:5.2f} -> {pb:5.2f}  {s[:88]}")
            print()


if __name__ == "__main__":
    main()
