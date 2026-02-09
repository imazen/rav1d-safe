#!/usr/bin/env python3
"""Run dav1d test suite against decode_md5 binary and report results."""

import re
import subprocess
import sys
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parent.parent
TEST_DATA = ROOT / "test-vectors" / "dav1d-test-data"
BINARY = ROOT / "target" / "release" / "examples" / "decode_md5"


def parse_meson_tests(meson_path, base_dir):
    """Parse meson.build for test entries: ['name', files('file'), 'md5']"""
    with open(meson_path) as f:
        content = f.read()

    entries = []
    # Match both .ivf and .obu files
    for m in re.finditer(
        r"\['([^']+)',\s*files\('([^']+)'\),\s*'([a-f0-9]{32})'\]", content
    ):
        name, fname, md5 = m.groups()
        fpath = base_dir / fname
        if fpath.exists():
            entries.append((name, str(fpath), md5))
        else:
            print(f"WARNING: {fpath} not found", file=sys.stderr)
    return entries


def run_test(name, filepath, expected_md5):
    """Run decode_md5 and return (name, passed, actual_md5, error)."""
    try:
        result = subprocess.run(
            [str(BINARY), filepath, expected_md5],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # stdout has the actual md5
        actual_md5 = result.stdout.strip()
        if result.returncode == 0:
            return (name, True, actual_md5, None)
        else:
            # Check if it's a decode error or mismatch
            stderr = result.stderr
            if "MISMATCH" in stderr:
                return (name, False, actual_md5, "mismatch")
            else:
                return (name, False, actual_md5, stderr.strip())
    except subprocess.TimeoutExpired:
        return (name, False, "", "timeout")
    except Exception as e:
        return (name, False, "", str(e))


def collect_tests():
    """Collect all test entries from all bit-depth directories."""
    all_tests = {}

    for bitdir in ["8-bit", "10-bit", "12-bit", "multi-bit"]:
        bitpath = TEST_DATA / bitdir
        if not bitpath.exists():
            continue

        tests = []
        for meson_file in bitpath.rglob("meson.build"):
            parent = meson_file.parent
            tests.extend(parse_meson_tests(meson_file, parent))

        if tests:
            all_tests[bitdir] = tests

    return all_tests


def main():
    if not BINARY.exists():
        print(f"ERROR: {BINARY} not found. Build with:")
        print(
            "  cargo build --release --no-default-features --features 'bitdepth_8,bitdepth_16' --example decode_md5"
        )
        sys.exit(1)

    all_tests = collect_tests()
    total = sum(len(t) for t in all_tests.values())
    print(f"Found {total} tests across {len(all_tests)} categories\n")

    overall_pass = 0
    overall_fail = 0
    failures = []

    for category, tests in sorted(all_tests.items()):
        print(f"=== {category} ({len(tests)} tests) ===")
        cat_pass = 0
        cat_fail = 0

        # Run tests in parallel (4 threads)
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(run_test, name, fpath, md5): (name, fpath, md5)
                for name, fpath, md5 in tests
            }

            for future in as_completed(futures):
                name, passed, actual_md5, error = future.result()
                if passed:
                    cat_pass += 1
                else:
                    cat_fail += 1
                    failures.append((category, name, actual_md5, error))

        print(f"  PASS: {cat_pass}  FAIL: {cat_fail}")
        overall_pass += cat_pass
        overall_fail += cat_fail

    print(f"\n{'='*60}")
    print(f"TOTAL: {overall_pass} passed, {overall_fail} failed out of {total}")

    if failures:
        print(f"\n--- FAILURES ({len(failures)}) ---")
        for cat, name, actual, error in sorted(failures):
            if error == "mismatch":
                print(f"  {cat}/{name}: got {actual}")
            else:
                print(f"  {cat}/{name}: {error}")

    sys.exit(0 if overall_fail == 0 else 1)


if __name__ == "__main__":
    main()
