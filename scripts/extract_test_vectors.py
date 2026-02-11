#!/usr/bin/env python3
"""
Extract test vectors from dav1d-test-data meson.build files.

Outputs TSV to stdout with columns:
    bitdepth, category, test_name, file_path (absolute), expected_md5, filmgrain (0 or 1)

Handles these meson.build patterns:
1. tests += [['name', files('path'), 'md5'], ...]
2. tests_obu += [['name', files('path'), 'md5'], ...]
3. fg_tests = [['name', files('path'), 'md5'], ...]  (filmgrain=1)
4. Standalone test() calls with --filmgrain and --verify
5. oss-fuzz: sanitizer test lists (no md5, no filmgrain)
6. Standalone test() calls for oss-fuzz with named files
"""

import os
import re
import sys

BASE = "/home/lilith/work/rav1d-safe/test-vectors/dav1d-test-data"

# Pattern for individual array entries: ['name', files('path'), 'md5']
ENTRY_RE = re.compile(
    r"\[\s*'([^']+)'\s*,\s*files\(\s*'([^']+)'\s*\)\s*,\s*'([0-9a-f]{32})'\s*\]"
)


def find_variable_context(text):
    """Find which variable each entry belongs to.

    Returns list of (var_name, name, filepath_rel, md5) tuples.
    We track variable assignments like:
        tests += [...]
        tests_obu += [...]
        fg_tests = [...]
    """
    results = []

    # Strategy: find assignment starts (e.g., "tests += [", "fg_tests = [")
    # then find all entries between that start and its matching close bracket.
    # We need to handle nested brackets.

    assign_re = re.compile(r"(tests|tests_obu|fg_tests)\s*[+=]+\s*\[")

    pos = 0
    while pos < len(text):
        m = assign_re.search(text, pos)
        if not m:
            break

        var_name = m.group(1)
        bracket_start = m.end() - 1  # position of the opening [

        # Find the matching closing bracket
        depth = 1
        i = bracket_start + 1
        while i < len(text) and depth > 0:
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
            i += 1

        block = text[bracket_start:i]

        for entry in ENTRY_RE.finditer(block):
            results.append((var_name, entry.group(1), entry.group(2), entry.group(3)))

        pos = i

    return results


def find_standalone_tests(text):
    """Find standalone test() calls with dav1d (not dav1d_fuzzer).

    Returns list of (name, filepath_rel, md5, filmgrain) tuples.
    """
    results = []

    # Find all test() calls to dav1d
    # We need to handle multi-line test() calls
    # Strategy: find "test(" then read until matching ")"
    test_start_re = re.compile(r"test\(\s*'([^']+)'\s*,\s*dav1d\s*,")

    pos = 0
    while pos < len(text):
        m = test_start_re.search(text, pos)
        if not m:
            break

        name = m.group(1)
        # Find matching closing paren
        paren_start = text.index("(", m.start())
        depth = 1
        i = paren_start + 1
        while i < len(text) and depth > 0:
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
            i += 1

        block = text[paren_start:i]

        # Extract file path
        file_match = re.search(r"files\(\s*'([^']+)'\s*\)", block)
        if not file_match:
            pos = i
            continue

        filepath_rel = file_match.group(1)

        # Extract md5 from --verify
        verify_match = re.search(r"'--verify'\s*,\s*'([0-9a-f]{32})'", block)
        if not verify_match:
            pos = i
            continue

        md5 = verify_match.group(1)

        # Check for filmgrain
        filmgrain = 1 if "'--filmgrain'" in block else 0

        results.append((name, filepath_rel, md5, filmgrain))
        pos = i

    return results


def parse_oss_fuzz(meson_path):
    """Parse the oss-fuzz meson.build which has a different format."""
    results = []
    meson_dir = os.path.dirname(meson_path)

    with open(meson_path) as f:
        text = f.read()

    NAME = "clusterfuzz-testcase-minimized-dav1d_fuzzer"

    # Parse the named test arrays
    array_configs = [
        ("asan_tests", "asan", NAME + "-"),
        ("msan_tests", "msan", NAME + "-"),
        ("ubsan_tests", "ubsan", NAME + "-"),
        ("asan_mt_tests", "asan", NAME + "_mt-"),
        ("msan_mt_tests", "msan", NAME + "_mt-"),
        ("ubsan_mt_tests", "ubsan", NAME + "_mt-"),
    ]

    for var_name, sanitizer, prefix in array_configs:
        # Find the array block with balanced brackets
        pattern = re.compile(rf"{var_name}\s*=\s*\[")
        m = pattern.search(text)
        if not m:
            continue

        bracket_start = m.end() - 1
        depth = 1
        i = bracket_start + 1
        while i < len(text) and depth > 0:
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
            i += 1

        block = text[bracket_start:i]
        ids = re.findall(r"'(\d+)'", block)

        for tc_id in ids:
            filename = prefix + tc_id
            filepath = os.path.join(meson_dir, sanitizer, filename)
            # Determine suite from sanitizer + whether mt
            if "mt" in var_name:
                suite = f"oss-fuzz-{sanitizer}"
            else:
                suite = f"oss-fuzz-{sanitizer}"
            results.append((
                "oss-fuzz",
                suite,
                tc_id,
                os.path.abspath(filepath),
                "",
                0,
            ))

    # Parse standalone test() calls for oss-fuzz (both dav1d_fuzzer and dav1d_fuzzer_mt)
    test_re = re.compile(r"test\(\s*'([^']+)'\s*,\s*dav1d_fuzzer(?:_mt)?\s*,")
    pos = 0
    while pos < len(text):
        m = test_re.search(text, pos)
        if not m:
            break

        test_name = m.group(1)
        paren_start = text.index("(", m.start())
        depth = 1
        i = paren_start + 1
        while i < len(text) and depth > 0:
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
            i += 1

        block = text[paren_start:i]

        # Extract file path
        # Handle both files('path') and files('path') inside args list
        file_match = re.search(r"files\(\s*'([^']+)'\s*\)", block)
        if not file_match:
            pos = i
            continue

        filepath_rel = file_match.group(1)
        filepath = os.path.join(meson_dir, filepath_rel)

        # Extract suite
        suite_match = re.search(r"suite:\s*'([^']+)'", block)
        suite = suite_match.group(1) if suite_match else "oss-fuzz"

        results.append((
            "oss-fuzz",
            suite,
            test_name,
            os.path.abspath(filepath),
            "",
            0,
        ))
        pos = i

    return results


def determine_bitdepth_and_category(meson_path):
    """Determine bitdepth and category from the meson.build path."""
    rel = os.path.relpath(meson_path, BASE)
    parts = rel.split(os.sep)
    bitdepth = parts[0]
    if len(parts) >= 3:
        category = parts[1]
    else:
        category = ""
    return bitdepth, category


def process_subdir_meson(meson_path):
    """Process a subdirectory meson.build file."""
    results = []
    meson_dir = os.path.dirname(meson_path)
    bitdepth, category = determine_bitdepth_and_category(meson_path)

    with open(meson_path) as f:
        text = f.read()

    # Parse variable-assigned entries
    var_entries = find_variable_context(text)
    for var_name, name, filepath_rel, md5 in var_entries:
        filepath = os.path.abspath(os.path.join(meson_dir, filepath_rel))
        filmgrain = 1 if var_name == "fg_tests" else 0
        results.append((bitdepth, category, name, filepath, md5, filmgrain))

    # Parse standalone test() calls
    standalone = find_standalone_tests(text)
    for name, filepath_rel, md5, filmgrain in standalone:
        filepath = os.path.abspath(os.path.join(meson_dir, filepath_rel))
        results.append((bitdepth, category, name, filepath, md5, filmgrain))

    return results


def process_bitdepth_meson(meson_path):
    """Process a bitdepth-level meson.build for standalone test() calls."""
    results = []
    meson_dir = os.path.dirname(meson_path)
    bitdepth, _ = determine_bitdepth_and_category(meson_path)

    with open(meson_path) as f:
        text = f.read()

    # Parse standalone test() calls
    standalone = find_standalone_tests(text)
    for name, filepath_rel, md5, filmgrain in standalone:
        filepath = os.path.abspath(os.path.join(meson_dir, filepath_rel))
        # Derive category from relative file path
        rel_parts = filepath_rel.split("/")
        category = rel_parts[0] if len(rel_parts) > 1 else "standalone"
        results.append((bitdepth, category, name, filepath, md5, filmgrain))

    return results


def main():
    all_results = []

    for root, dirs, files_list in os.walk(BASE):
        if "meson.build" not in files_list:
            continue

        meson_path = os.path.join(root, "meson.build")
        rel = os.path.relpath(meson_path, BASE)
        parts = rel.split(os.sep)

        # Skip top-level meson.build
        if rel == "meson.build":
            continue

        # Handle oss-fuzz
        if parts[0] == "oss-fuzz":
            all_results.extend(parse_oss_fuzz(meson_path))
            continue

        # Bitdepth-level meson.build
        if len(parts) == 2:
            all_results.extend(process_bitdepth_meson(meson_path))
            continue

        # Subdirectory meson.build
        all_results.extend(process_subdir_meson(meson_path))

    # Print header
    print("bitdepth\tcategory\ttest_name\tfile_path\texpected_md5\tfilmgrain")

    # Sort and print
    for row in sorted(all_results, key=lambda r: (r[0], r[1], r[2])):
        print(f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}\t{row[4]}\t{row[5]}")

    print(f"# Total: {len(all_results)} test vectors", file=sys.stderr)


if __name__ == "__main__":
    main()
