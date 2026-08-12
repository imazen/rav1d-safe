#!/usr/bin/env python3
"""Per-site VERDICT table from the bounds map's raw aggregate rows.

Reads the `B*` rows emitted by `--features __probe_bounds`
(benchmarks/bounds_map_raw_2026-08-10.tar.zst) and applies the verdict rules
mechanically.  No hand-picked numbers: everything printed here is a join over
BSITE / BCONC / BGAPMUT / BPAIR.

VERDICT RULES (auditable, applied in this order):

  UNMEASURED            fp_kind == whole AND res_mean > WHOLE_TRUST_B.  The
                        instrument declined to answer the over-reservation
                        question at this site, so NO narrowing verdict may be
                        issued.  Fix = one probe_declare_rows call.
  BLOCKED               n_fp_ovl > 0 with a mutable counterparty (a real
                        concurrent footprint conflict), OR min_gap_mut == 0
                        (a concurrent foreign WRITE was butt-adjacent, so
                        widening by ONE byte collides).
  NARROWABLE            over_ratio > 1.02 with DECLARED geometry
                        (fp_kind == rows): the reservation exceeds the
                        footprint and the row set is the tight extent.
  COARSENABLE-UP-TO-N   N = min_gap_mut, the closest OBSERVED approach to a
                        concurrent foreign WRITE.  N = INF when the site never
                        met a concurrent write at any distance.

Below WHOLE_TRUST_B an undeclared (`whole`) footprint still bounds the
over-reservation by the reservation itself, which is what the conflict
question needs; above it the column is an artefact.
"""

import os
import sys

WHOLE_TRUST_B = 64

CELLS = [
    ("v4k_8tile t=1", "bounds_v4k_8tile_t1.txt", 1, 1),
    ("v4k_8tile t=8", "bounds_v4k_8tile_t8.txt", 8, 1),
    ("v4k_8tile_10b t=1", "bounds_v4k_8tile_10b_t1.txt", 1, 1),
    ("v4k_8tile_10b t=8", "bounds_v4k_8tile_10b_t8.txt", 8, 1),
    ("corpus 8-bit/data t=1", "corpus_8bitdata_t1.txt", 1, 1406),
    ("corpus 8-bit/data t=8", "corpus_8bitdata_t8.txt", 8, 1406),
    ("corpus 10-bit/data t=8", "corpus_10bitdata_t8.txt", 8, 284),
    ("corpus 8-bit/features t=8", "corpus_8bitfeat_t8.txt", 8, 26),
]
EDGES = [0, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 1 << 20]


def load(path):
    site, conc, gapmut, pair, hdr = {}, {}, {}, [], {}
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        k = f[0]
        if k in ("RUN", "CORPUS"):
            hdr["run"] = f[1:]
        elif k == "BSITE":
            site[f[18]] = dict(
                n=float(f[1]), mut=float(f[2]), res=float(f[3]), fp=float(f[4]),
                over=float(f[5]), kind=f[6], rows=float(f[10]), w=float(f[11]),
                gap=float(f[12]), lead=float(f[13]), tail=float(f[14]),
                n_read=float(f[15]), n_write=float(f[16]))
        elif k == "BCONC":
            conc[f[25]] = dict(
                n=int(f[1]), n_conc=int(f[2]), n_conc_mut=int(f[4]),
                n_res_ovl=int(f[5]), n_fp_ovl=int(f[6]), n_row_ovl=int(f[7]),
                n_row_ovl_mut=int(f[8]), n_row_fp_ovl=int(f[9]),
                n_row_fp_ovl_mut=int(f[10]),
                min_gap=int(f[11]) if f[11] != "-" else -1,
                min_gap_mut=int(f[12]) if f[12] != "-" else -1)
        elif k == "BGAPMUT":
            gapmut[f[14]] = [int(x) for x in f[2:14]]
        elif k == "BPAIR":
            pair.append(dict(n=int(f[1]), fp_ovl=int(f[3]), row_ovl=int(f[4]),
                             fmut=int(f[5]), min_gap=int(f[6]), acq=f[7], con=f[8]))
    return hdr, site, conc, gapmut, pair


def cum(h):
    out, t = [], 0
    for v in h[:-1]:
        t += v
        out.append(t)
    return out


def short(s):
    return s.replace("src/", "").replace("include/dav1d/", "")


def verdict_for(st, cc, gg, pairs, s):
    c = cum(gg)
    nm = cc["n_conc_mut"]
    mgm = cc["min_gap_mut"]
    cp = sorted([p for p in pairs if p["acq"] == s and p["fmut"] > 0],
                key=lambda p: p["min_gap"])
    who = f"{short(cp[0]['con'])}@{cp[0]['min_gap']}B" if cp else "-"
    if st["kind"] == "whole" and st["res"] > WHOLE_TRUST_B:
        return "UNMEASURED", who, c, nm, mgm
    if cc["n_fp_ovl"] > 0 and nm > 0:
        return "BLOCKED-fp", who, c, nm, mgm
    if nm == 0:
        return "COARSENABLE-INF", who, c, nm, mgm
    if mgm == 0:
        return "BLOCKED-0", who, c, nm, mgm
    if st["kind"] == "rows" and st["over"] > 1.02:
        return f"NARROWABLE-{st['over']:.0f}x", who, c, nm, mgm
    return f"COARSEN-{mgm}", who, c, nm, mgm


def main():
    raw = sys.argv[1] if len(sys.argv) > 1 else "raw"
    cells = {n: (load(os.path.join(raw, f)), t, fr)
             for n, f, t, fr in CELLS if os.path.exists(os.path.join(raw, f))}
    V4, CO = "v4k_8tile t=8", "corpus 8-bit/data t=8"
    (_, s4, c4, g4, p4), _, _ = cells[V4]
    (_, s8, c8, g8, p8), _, fr8 = cells[CO]
    (_, s4b, c4b, g4b, p4b), _, _ = cells["v4k_8tile_10b t=8"]

    def row(s, pf4, pfc):
        # Conflict facts come from the corpus cell where the site ran there
        # (LR live, inter live, ~1000x the collision rate); otherwise 4K.
        src = CO if s in c8 else V4
        st = (s8 if src == CO else s4)[s]
        cc = (c8 if src == CO else c4)[s]
        gg = (g8 if src == CO else g4)[s]
        pr = p8 if src == CO else p4
        v, who, c, nm, mgm = verdict_for(st, cc, gg, pr, s)
        if st["kind"] == "rows" and st["over"] > 1.02:
            waste = f"gap {st['gap']:.0f} / lead {st['lead']:.0f} / tail {st['tail']:.0f}"
        elif st["kind"] == "whole" and st["res"] > WHOLE_TRUST_B:
            waste = "NOT MEASURED"
        else:
            waste = f"<= {st['res']:.0f} B"
        # divergence: same site's headroom on the 4K cell
        h4 = "-"
        if s in c4:
            h4 = "INF" if c4[s]["n_conc_mut"] == 0 else str(c4[s]["min_gap_mut"])
        mw = "R" if st["n_write"] == 0 else ("W" if st["n_read"] == 0 else "RW")
        return dict(site=s, pf4=pf4, pfc=pfc, res=st["res"], over=st["over"],
                    kind=st["kind"], waste=waste, rw=mw, nm=nm,
                    mgm="INF" if nm == 0 else mgm, h4=h4, c=c, v=v, who=who,
                    src="corpus" if src == CO else "v4k")

    hdr = ("| site | reg/frame | R/W | res B | over | waste | conc-write encounters "
           "| headroom N | 4K-cell N | k=64 | k=256 | k=1K | verdict | nearest concurrent writer |")
    sep = "|" + "---|" * 14

    print("## Table 1 — sites live on `v4k_8tile` t=8, by registrations/frame\n")
    print(hdr)
    print(sep)
    rows4 = sorted(s4.items(), key=lambda kv: -kv[1]["n"])
    tsv = [("cell", "site", "regs_per_frame", "rw", "res_mean_B", "over_ratio", "fp_kind",
            "waste", "n_conc_mut", "headroom_min_gap_mut_B", "headroom_v4k_cell",
            "coll_k64", "coll_k256", "coll_k1k", "verdict", "nearest_concurrent_writer",
            "conflict_facts_from")]
    for s, d in rows4:
        pfc = s8[s]["n"] / fr8 if s in s8 else None
        r = row(s, d["n"], pfc)
        print(f"| `{short(s)}` | {d['n']:,.0f} | {r['rw']} | {r['res']:.2f} | "
              f"{r['over']:.3f} | {r['waste']} | {r['nm']:,} | {r['mgm']} | {r['h4']} | "
              f"{r['c'][3]:,} | {r['c'][4]:,} | {r['c'][5]:,} | **{r['v']}** | {r['who']} |")
        tsv.append(("v4k_8tile_t8", s, f"{d['n']:.0f}", r["rw"], f"{r['res']:.2f}",
                    f"{r['over']:.3f}", r["kind"], r["waste"], str(r["nm"]), str(r["mgm"]),
                    r["h4"], str(r["c"][3]), str(r["c"][4]), str(r["c"][5]), r["v"],
                    r["who"], r["src"]))

    print("\n## Table 2 — sites the 4K gap vectors NEVER EXECUTE, by corpus registrations/frame\n")
    print(hdr.replace("| 4K-cell N ", "| 4K-cell N "))
    print(sep)
    only = [(s, d) for s, d in s8.items() if s not in s4]
    only.sort(key=lambda kv: -kv[1]["n"])
    shown = 0
    for s, d in only:
        pfc = d["n"] / fr8
        if pfc < 300 and shown > 24:
            continue
        r = row(s, None, pfc)
        shown += 1
        print(f"| `{short(s)}` | {pfc:,.0f} | {r['rw']} | {r['res']:.2f} | "
              f"{r['over']:.3f} | {r['waste']} | {r['nm']:,} | {r['mgm']} | n/a | "
              f"{r['c'][3]:,} | {r['c'][4]:,} | {r['c'][5]:,} | **{r['v']}** | {r['who']} |")
        tsv.append(("corpus_8bitdata_t8", s, f"{pfc:.1f}", r["rw"], f"{r['res']:.2f}",
                    f"{r['over']:.3f}", r["kind"], r["waste"], str(r["nm"]), str(r["mgm"]),
                    "n/a", str(r["c"][3]), str(r["c"][4]), str(r["c"][5]), r["v"],
                    r["who"], r["src"]))

    with open("verdict_table.tsv", "w") as f:
        for t in tsv:
            f.write("\t".join(t) + "\n")

    # ---------------- coverage + class shares ----------------
    print("\n## Coverage\n")
    print(f"distinct sites: v4k_8tile t=8 = {len(s4)}, corpus 8-bit/data t=8 = {len(s8)}, "
          f"corpus-only = {len(only)}")
    n4 = sum(d["n"] for d in s4.values())
    nc = sum(d["n"] for d in s8.values())
    ncov = sum(d["n"] for s, d in s8.items() if s in s4)
    print(f"corpus registrations at a site the 4K vector also runs: "
          f"{ncov:,.0f} / {nc:,.0f} = {100*ncov/nc:.1f}%")
    for fam in ("src/mc", "src/safe_simd/mc_arm", "looprestoration", "lr_apply",
                "refmvs", "picture.rs:2027", "owned_recon"):
        tot = sum(d["n"] for s, d in s8.items() if fam in s)
        in4 = sum(d["n"] for s, d in s4.items() if fam in s)
        print(f"  {fam:28s} corpus {tot:>13,.0f}   4K {in4:>10,.0f}")

    print("\n## Verdict-class shares (corpus 8-bit/data t=8, by registrations)\n")
    agg = {}
    for s, d in s8.items():
        r = row(s, None, d["n"] / fr8)
        cls = r["v"].split("-")[0] if not r["v"].startswith("COARSEN") else "COARSENABLE"
        if r["v"] == "COARSENABLE-INF":
            cls = "COARSENABLE-INF"
        agg[cls] = agg.get(cls, 0) + d["n"]
    for k in sorted(agg, key=lambda k: -agg[k]):
        print(f"  {k:20s} {agg[k]:>14,.0f}  {100*agg[k]/nc:5.1f}%")

    # ---------------- the three refutations ----------------
    print("\n## Refutation gate\n")
    (_, s1, c1, g1, p1), _, _ = cells["v4k_8tile t=1"]
    print("### #475 hull — sites with DECLARED geometry at t=1 (the hull path)\n")
    print("| site | n/frame | reserved B | footprint B | over | gap waste | lead | tail | verdict |")
    print("|" + "---|" * 9)
    for s, d in sorted(s1.items(), key=lambda kv: -kv[1]["over"]):
        if d["kind"] != "rows":
            continue
        print(f"| `{short(s)}` | {d['n']:,.0f} | {d['res']:,.1f} | {d['fp']:.2f} | "
              f"**{d['over']:.1f}x** | {d['gap']:,.1f} | {d['lead']:.2f} | {d['tail']:.2f} "
              f"| NARROWABLE (row set) |")

    print("\n### #485 band / any widening of `loopfilter.rs:710:14`\n")
    print("| cell | frames | n | headroom N | k<=64 | k<=256 | k<=1K | k<=4K |")
    print("|" + "---|" * 8)
    for cell in [V4, "v4k_8tile_10b t=8", CO, "corpus 10-bit/data t=8"]:
        (_, ss, ccm, ggm, _), _, fr = cells[cell]
        s = "src/loopfilter.rs:710:14"
        if s not in ggm:
            continue
        cu, cc = cum(ggm[s]), ccm[s]
        print(f"| {cell} | {fr} | {cc['n']:,} | {cc['min_gap_mut']} | {cu[3]:,} | "
              f"{cu[4]:,} | {cu[5]:,} | {cu[6]:,} |")

    print("\n### #469 rectangle — the ROWBAND counterfactual (widen to the whole picture rows)\n")
    print("| cell | acquisitions whose row-band hits a foreign reservation | ... a MUTABLE one "
          "| ... a foreign FOOTPRINT | ... a MUTABLE footprint |")
    print("|" + "---|" * 5)
    for cell in [V4, "v4k_8tile_10b t=8"]:
        (_, ss, ccm, ggm, _), _, fr = cells[cell]
        t = [sum(v[k] for v in ccm.values()) for k in
             ("n_row_ovl", "n_row_ovl_mut", "n_row_fp_ovl", "n_row_fp_ovl_mut")]
        print(f"| {cell} | {t[0]:,} | **{t[1]:,}** | {t[2]:,} | **{t[3]:,}** |")
    print("\nper-site, v4k_8tile t=8, sites whose row-band would hit a concurrent WRITE:\n")
    for s, v in sorted(c4.items(), key=lambda kv: -kv[1]["n_row_ovl_mut"])[:10]:
        if v["n_row_ovl_mut"]:
            print(f"  {short(s):44s} row_ovl_mut={v['n_row_ovl_mut']:,} of n={v['n']:,}")

    # V-batch prediction
    print("\n## The in-flight V-batch (LF_BATCH_V 4 -> 32) priced against the budget\n")
    for cell, bpp in ((CO, 1), ("corpus 10-bit/data t=8", 2),
                      (V4, 1), ("v4k_8tile_10b t=8", 2)):
        (_, ss, ccm, ggm, _), _, fr = cells[cell]
        s = "src/loopfilter.rs:710:14"
        if s not in ggm:
            continue
        cu = cum(ggm[s])
        delta = (128 - 16) * bpp
        lo = hi = 0
        for i, e in enumerate(EDGES):
            if e < delta:
                lo = cu[i]
            if e >= delta:
                hi = cu[i]
                break
        vshare = 0.307
        print(f"| {cell} | +{delta} B | predicted collisions {lo:,}..{hi:,} over {fr} frames "
              f"| x V-share {vshare} = {lo*vshare:.1f}..{hi*vshare:.1f} |")


main()
