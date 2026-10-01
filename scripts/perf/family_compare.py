#!/usr/bin/env python3
"""Side-by-side per-family wall time (ms/frame) for dav1d vs rav1d-safe.

Input: two `ptrace_sample.py` outputs (`count fn(module)` per line, runs merged
by summing) plus each program's ms/frame. Both programs' symbol names are mapped
onto the same families, so the table answers "which family is where we lose to
dav1d" and the DELTA column ranks where a fix pays off, in ms/frame.

  family_compare.py <dav1d_samples.txt> <dav1d_ms_per_frame> \
                    <ours_samples.txt>  <ours_ms_per_frame> [--top 8]

Caveats: leaf-PC sampling attributes inlined code to the enclosing symbol, and
the families are name-regex heuristics (check the per-family top symbols the
script prints before trusting a small bucket).
"""
import collections, re, sys

# Most specific first. dav1d (C + asm) and rav1d-safe (Rust + safe SIMD) names.
FAMILIES = [
    # dav1d runs msac as separate asm functions; rav1d-safe inlines it into
    # decode_coefs. Only the MERGED entropy-decoding family compares fairly.
    ("entropy",     r"msac|decode_symbol|decode_bool|decode_hi_tok|adapt4|adapt8|adapt16|update_cdf|decode_coefs|read_coef_tree|get_lo_ctx|get_dc_sign|golomb|coef"),
    ("itx",         r"itx|inv_txfm|inv_dct|inv_adst|inv_identity|inv_wht|iadst|idct|iflipadst|iidentity|wht4"),
    ("cdef",        r"cdef"),
    ("loopfilter",  r"lpf|loopfilter|loop_filter|apply_h|apply_v|filter_edge_run|lf_apply|deblock"),
    ("lf_mask",     r"lf_mask|mask_edges|decomp_tx|create_lf"),
    ("looprestor",  r"wiener|sgr|selfguided|looprestor|lr_apply|lr_filter|boxsum|box_"),
    ("ipred",       r"ipred|intra_pred|cfl|pal_|filter_edge|prepare_intra|smooth|paeth|z1_|z2_|z3_|dc_pred|splat_dc"),
    ("mc",          r"put_|prep_|8tap|6tap|bilin|avg|w_mask|blend|obmc|warp|emu_edge|resize|\bmc\b|mc::|mc_|mask_8bpc"),
    ("refmvs",      r"refmvs|splat_mv|scan_row|scan_col|add_spatial|add_temporal|find_matching|save_tmvs|load_tmvs|mvstack"),
    ("tracker",     r"DisjointMut|BorrowTracker|try_lock|TinyLock|ShardGuard|borrow_id|index_mut|tracker"),
    ("filmgrain",   r"filmgrain|film_grain|fgy|fguv|grain"),
    ("decode_ctx",  r"decode_b|decode_sb|read_|case_set|small_memset|ctx|decode_tile|decode_frame|parse_obus"),
    ("recon_glue",  r"recon_b|filter_sbrow|backup_ipred|recon"),
    ("libc/mem",    r"memcpy|memset|memmove|malloc|free|calloc|realloc|copy_from_slice|__memcpy|__memset"),
]
RX = [(n, re.compile(p, re.I)) for n, p in FAMILIES]

def fam(sym):
    for n, rx in RX:
        if rx.search(sym):
            return n
    return "other"

def demangle(syms):
    """Batch-demangle Rust v0/legacy names via rustfilt if installed."""
    import shutil, subprocess
    if not shutil.which('rustfilt'):
        return syms
    out = subprocess.run(['rustfilt'], input='\n'.join(syms), capture_output=True, text=True).stdout.split('\n')
    return out[:len(syms)] if len(out) >= len(syms) else syms

def load(path):
    fams, top = collections.Counter(), collections.defaultdict(collections.Counter)
    rows = []
    for l in open(path, errors='replace'):
        m = re.match(r'\s*(\d+)\s+(.*)$', l)
        if m:
            rows.append((int(m.group(1)), m.group(2)))
    names = demangle([r[1] for r in rows])
    for (n, _), sym in zip(rows, names):
        f = fam(sym)
        fams[f] += n
        top[f][re.sub(r'<[^>]*>', '<>', sym)[:70]] += n
    return fams, top

def main():
    a = sys.argv[1:]
    dpath, dms, opath, oms = a[0], float(a[1]), a[2], float(a[3])
    ntop = 8 if '--top' not in a else int(a[a.index('--top') + 1])
    d, dt = load(dpath); o, ot = load(opath)
    dtot, otot = sum(d.values()), sum(o.values())
    rows = []
    for f in set(d) | set(o):
        dm = dms * d[f] / dtot if dtot else 0
        om = oms * o[f] / otot if otot else 0
        rows.append((om - dm, f, dm, om))
    rows.sort(reverse=True)
    print(f"{'family':12s} {'dav1d':>8s} {'ours':>8s} {'delta':>8s}   (ms/frame; total dav1d {dms:.3f}, ours {oms:.3f}, +{100*(oms/dms-1):.0f}%)")
    for delta, f, dm, om in rows:
        print(f"{f:12s} {dm:8.3f} {om:8.3f} {delta:+8.3f}")
    print("\nlargest delta families: top symbols ours | dav1d")
    for delta, f, dm, om in rows[:4]:
        print(f"\n== {f} (delta {delta:+.3f} ms/f)")
        for s, n in ot[f].most_common(ntop):
            print(f"   ours  {oms*n/otot:6.3f}  {s}")
        for s, n in dt[f].most_common(max(3, ntop // 2)):
            print(f"   dav1d {dms*n/dtot:6.3f}  {s}")

if __name__ == '__main__':
    main()
