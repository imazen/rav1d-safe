#!/usr/bin/env python3
"""Add flex() / flex_mut() shadows to #[arcane] and #[rite] SIMD functions.

For each function decorated with #[arcane] or #[rite] that takes slice parameters,
this script adds shadow bindings at the top of the function body so that all
subsequent indexing goes through FlexSlice/FlexSliceMut (which supports unchecked
mode when the feature flag is enabled).

Usage: python3 scripts/add_flex_shadows.py src/safe_simd/itx.rs [--dry-run]
"""

import re
import sys

# Slice parameter patterns we want to shadow
# Match: `name: &[T]` or `name: &mut [T]` where T is a simple type
SLICE_PARAM_RE = re.compile(
    r'(\w+)\s*:\s*&(mut\s+)?\[(\w+)\]'
)

# Types that are fixed-size arrays, not dynamic slices (skip these)
SKIP_PARAM_NAMES = {'_token', 'token', '_eob', '_offsets'}

# Parameters whose type is a reference to a fixed-size array like &[i8; 8]
# These don't need flex() since they're not dynamically indexed
FIXED_ARRAY_RE = re.compile(r'(\w+)\s*:\s*&(mut\s+)?\[\w+;\s*\w+\]')


def find_arcane_rite_functions(lines):
    """Find all #[arcane] or #[rite] function definitions and their slice params."""
    functions = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Look for #[arcane] or #[rite]
        if line in ('#[arcane]', '#[rite]'):
            # Find the fn line (may be next line or a few lines down)
            fn_start = None
            for j in range(i + 1, min(i + 5, len(lines))):
                if 'fn ' in lines[j]:
                    fn_start = j
                    break
            if fn_start is None:
                i += 1
                continue

            # Collect full signature (may span multiple lines)
            sig_lines = []
            j = fn_start
            brace_found = False
            while j < len(lines):
                sig_lines.append(lines[j])
                if '{' in lines[j]:
                    brace_found = True
                    break
                j += 1

            if not brace_found:
                i += 1
                continue

            full_sig = ' '.join(sig_lines)

            # Check if this is gated behind asm
            # Look back a few lines for #[cfg(feature = "asm")]
            is_asm_gated = False
            for k in range(max(0, i - 5), i):
                if 'feature = "asm"' in lines[k]:
                    is_asm_gated = True
                    break
            if is_asm_gated:
                i = j + 1
                continue

            # Find the opening brace line
            brace_line = j

            # Extract slice parameters (skip fixed-size arrays)
            fixed_arrays = set()
            for m in FIXED_ARRAY_RE.finditer(full_sig):
                fixed_arrays.add(m.group(1))

            params = []
            for m in SLICE_PARAM_RE.finditer(full_sig):
                name = m.group(1)
                is_mut = bool(m.group(2))
                elem_type = m.group(3)
                if name in SKIP_PARAM_NAMES:
                    continue
                if name in fixed_arrays:
                    continue
                # Skip if param name starts with _
                if name.startswith('_'):
                    continue
                params.append((name, is_mut, elem_type))

            if params:
                functions.append({
                    'brace_line': brace_line,
                    'params': params,
                    'fn_sig': full_sig.strip(),
                })

            i = j + 1
        else:
            i += 1

    return functions


def add_flex_import(lines):
    """Add Flex import to the use section if not already present."""
    # Check if already imported
    for line in lines:
        if 'pixel_access::Flex' in line or 'pixel_access::{' in line and 'Flex' in line:
            return lines

    # Find last use statement in the top-level imports
    last_use = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('use ') and not stripped.startswith('use super'):
            last_use = i

    # Insert after the last use
    indent = ''
    import_line = f'{indent}use crate::src::safe_simd::pixel_access::Flex;\n'
    lines.insert(last_use + 1, import_line)
    return lines


def insert_flex_shadows(lines, functions):
    """Insert flex() shadow bindings at the top of each function body."""
    # Process in reverse order so line numbers stay valid
    for func in reversed(functions):
        brace_line = func['brace_line']
        params = func['params']

        # Determine indentation (typically 4 spaces inside function body)
        existing_indent = ''
        if brace_line + 1 < len(lines):
            next_line = lines[brace_line + 1]
            # Count leading whitespace of next non-empty line
            for j in range(brace_line + 1, min(brace_line + 10, len(lines))):
                if lines[j].strip():
                    existing_indent = re.match(r'^(\s*)', lines[j]).group(1)
                    break
        if not existing_indent:
            existing_indent = '    '

        # Build shadow lines
        shadows = []
        for name, is_mut, _elem_type in params:
            if is_mut:
                shadows.append(f'{existing_indent}let mut {name} = {name}.flex_mut();\n')
            else:
                shadows.append(f'{existing_indent}let {name} = {name}.flex();\n')

        # Insert after the opening brace line
        # But check if there's already a `use` statement right after - insert after that
        insert_pos = brace_line + 1
        while insert_pos < len(lines) and lines[insert_pos].strip().startswith('use '):
            insert_pos += 1

        for j, shadow in enumerate(shadows):
            lines.insert(insert_pos + j, shadow)

    return lines


def process_file(filepath, dry_run=False):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    functions = find_arcane_rite_functions(lines)
    if not functions:
        print(f"No eligible functions found in {filepath}")
        return

    print(f"Found {len(functions)} functions to modify in {filepath}")
    for func in functions:
        params_str = ', '.join(
            f"{'mut ' if m else ''}{n}: &{'mut ' if m else ''}[{t}]"
            for n, m, t in func['params']
        )
        print(f"  Line {func['brace_line'] + 1}: {params_str}")

    if dry_run:
        return

    # Add import
    lines = add_flex_import(lines)
    # Re-find functions after import insertion (line numbers shifted by 1)
    functions = find_arcane_rite_functions(lines)

    # Insert shadows
    lines = insert_flex_shadows(lines, functions)

    with open(filepath, 'w') as f:
        f.writelines(lines)

    print(f"Modified {filepath}: {len(functions)} functions updated")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 add_flex_shadows.py <file.rs> [--dry-run]")
        sys.exit(1)

    filepath = sys.argv[1]
    dry_run = '--dry-run' in sys.argv
    process_file(filepath, dry_run)
