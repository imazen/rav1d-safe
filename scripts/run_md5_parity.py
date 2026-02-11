#!/usr/bin/env python3
"""Run MD5 parity tests against dav1d reference hashes.

Parses meson.build files from dav1d-test-data to extract expected MD5 hashes,
then decodes each test vector with rav1d-safe and compares the result.

Usage:
    python3 scripts/run_md5_parity.py [--bitdepth 8|10|12|multi|all] [--stop-on-fail]
"""

import re
import os
import subprocess
import sys
import argparse
from collections import Counter

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'test-vectors', 'dav1d-test-data')
DECODE_MD5 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'target', 'release', 'examples', 'decode_md5')


def parse_test_cases(base_dir):
    """Parse meson.build files to extract (bitdepth, ivf_path, expected_md5)."""
    test_cases = []
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f != 'meson.build':
                continue
            path = os.path.join(root, f)
            with open(path) as fh:
                content = fh.read()
            for m in re.finditer(
                r"\['\w+',\s*files\('([^']+)'\),\s*'([a-f0-9]{32})'\]", content
            ):
                ivf_rel = m.group(1)
                md5 = m.group(2)
                ivf_path = os.path.join(root, ivf_rel)
                if not os.path.exists(ivf_path):
                    continue
                rel = os.path.relpath(root, base_dir)
                if rel.startswith('8-bit'):
                    bd = '8'
                elif rel.startswith('10-bit'):
                    bd = '10'
                elif rel.startswith('12-bit'):
                    bd = '12'
                elif rel.startswith('multi-bit'):
                    bd = 'multi'
                else:
                    bd = '?'
                test_cases.append((bd, ivf_path, md5))
    return test_cases


def run_test(ivf_path, expected_md5):
    """Decode an IVF file and compare MD5. Returns (actual_md5, passed, error_msg)."""
    try:
        result = subprocess.run(
            [DECODE_MD5, ivf_path, expected_md5],
            capture_output=True, text=True, timeout=30
        )
        # stdout has the actual MD5
        actual_md5 = result.stdout.strip()
        if result.returncode == 0:
            return actual_md5, True, None
        else:
            return actual_md5, False, result.stderr.strip()
    except subprocess.TimeoutExpired:
        return "", False, "TIMEOUT"
    except Exception as e:
        return "", False, str(e)


def main():
    parser = argparse.ArgumentParser(description='MD5 parity test against dav1d reference')
    parser.add_argument('--bitdepth', choices=['8', '10', '12', 'multi', 'all'], default='all')
    parser.add_argument('--stop-on-fail', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--category', help='Filter by subdirectory (e.g. "data", "film_grain")')
    args = parser.parse_args()

    if not os.path.exists(DECODE_MD5):
        print(f"ERROR: {DECODE_MD5} not found. Build first with:")
        print("  cargo build --release --no-default-features --features 'bitdepth_8,bitdepth_16' --example decode_md5")
        sys.exit(2)

    test_cases = parse_test_cases(BASE)

    if args.bitdepth != 'all':
        test_cases = [(bd, p, m) for bd, p, m in test_cases if bd == args.bitdepth]

    if args.category:
        test_cases = [(bd, p, m) for bd, p, m in test_cases if f'/{args.category}/' in p]

    test_cases.sort(key=lambda x: x[1])

    passed = 0
    failed = 0
    errors = 0
    fail_details = []

    total = len(test_cases)
    print(f"Running {total} MD5 parity tests...")

    for i, (bd, ivf_path, expected_md5) in enumerate(test_cases, 1):
        short_path = os.path.relpath(ivf_path, BASE)
        actual_md5, ok, err_msg = run_test(ivf_path, expected_md5)

        if ok:
            passed += 1
            if args.verbose:
                print(f"  [{i}/{total}] PASS {short_path}")
        elif err_msg and ('TIMEOUT' in err_msg or 'Decode error' in err_msg or 'parse error' in err_msg):
            errors += 1
            print(f"  [{i}/{total}] ERROR {short_path}: {err_msg[:80]}")
            fail_details.append((bd, short_path, 'ERROR', err_msg[:80]))
        else:
            failed += 1
            print(f"  [{i}/{total}] FAIL {short_path}: got {actual_md5} expected {expected_md5}")
            fail_details.append((bd, short_path, actual_md5, expected_md5))
            if args.stop_on_fail:
                break

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} passed, {failed} failed, {errors} errors")

    if fail_details:
        # Summarize by bitdepth
        bd_counts = Counter()
        bd_totals = Counter(bd for bd, _, _ in test_cases)
        for bd, _, _, _ in fail_details:
            bd_counts[bd] += 1
        print(f"\nFailures by bit depth:")
        for bd in sorted(bd_totals):
            f = bd_counts.get(bd, 0)
            t = bd_totals[bd]
            p = t - f
            print(f"  {bd}-bit: {p}/{t} passed ({f} failures)")

    if not fail_details:
        print("\nAll tests passed!")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
