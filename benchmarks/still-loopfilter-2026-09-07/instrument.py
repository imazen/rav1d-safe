"""Build a diagnostic mask census, restoring both production files in finally.

Run through run-heavy, without concurrent builds or source writers. The
existing census driver reports after all clients finish; no counters reset.
Its one serial reference decode is accounted for by census.py.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--repo', type=Path, required=True)
p.add_argument('--work-dir', type=Path, required=True)
p.add_argument('--baseline-revision', required=True)
p.add_argument('--shared-target', type=Path, required=True)
a = p.parse_args()
repo, root = a.repo.resolve(), a.work_dir.resolve()
paths = ['src/safe_simd/loopfilter.rs', 'src/ablate.rs']
before = {f: (repo / f).read_bytes() for f in paths}
for f, data in before.items():
    assert data == subprocess.check_output(['git', 'show', a.baseline_revision + ':' + f], cwd=repo)
    (root / ('before-' + Path(f).name)).write_bytes(data)
assert 'codex-still-parity' in (repo / '.workongoing').read_text()
s = before[paths[0]].decode()
matches = list(re.finditer(r'^fn (loop_filter_4_8bpc_[^(]+)\(', s, re.M))
assert len(matches) == 13
sites, insertions = [], []
for index, match in enumerate(matches):
    name = match[1]
    end = matches[index + 1].start() if index + 1 < len(matches) else s.index('fn read_lvl')
    body = s[match.end():end]
    width = int(re.search(r'_wd(\d+)_', name)[1]) if '_wd' in name else 4
    lanes = 16 if name.endswith('_x16') else 8 if name.endswith('_x8') else 4
    last = 'flat8in_mask' if width == 16 else 'flat_mask' if width != 4 else 'fm_mask'
    start = body.index('let ' + last + ' =')
    at = match.end() + body.index(';', start) + 1

    def bits(mask):
        if lanes == 16:
            return mask + ' as u32'
        prefix = '_mm256' if lanes == 8 else '_mm'
        cast = '_mm256_castsi256_ps' if lanes == 8 else '_mm_castsi128_ps'
        return f'{prefix}_movemask_ps({cast}({mask})) as u32'

    fm = bits('fm_mask')
    inner = bits('flat8in_mask' if width == 16 else 'flat_mask') if width != 4 else '0'
    outer = bits('flat8out_mask') if width == 16 else '0'
    call = (f'\n    #[cfg(feature = "__ablate")]\n'
            f'    crate::src::ablate::lf_mask_census::note({index}, {lanes}, {fm}, {inner}, {outer});\n')
    insertions.append((at, call))
    sites.append(dict(index=index, kernel=name, width=width, lanes=lanes,
                      direction='h' if '_simd_h' in name else 'v'))
for at, text in reversed(insertions):
    s = s[:at] + text + s[at:]
names = ',\n        '.join(json.dumps(site['kernel']) for site in sites)
module = '''
// Diagnostic only: monotonic process counters, no reset or dispatch changes.
#[cfg(feature = "__ablate")]
pub(crate) mod lf_mask_census {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    static COUNTS: [[AtomicU64; 10]; 13] =
        [const { [const { AtomicU64::new(0) }; 10] }; 13];
    pub(crate) fn note(id: usize, lanes: u32, fm: u32, inner: u32, outer: u32) {
        let wide = fm & inner & outer;
        let mid = fm & inner & !outer;
        let narrow = fm & !inner;
        let values = [1, lanes as u64, fm.count_ones() as u64,
            mid.count_ones() as u64, wide.count_ones() as u64, narrow.count_ones() as u64,
            (fm == 0) as u64, (mid == 0) as u64, (wide == 0) as u64, (narrow == 0) as u64];
        for (counter, value) in COUNTS[id].iter().zip(values) {
            counter.fetch_add(value, Relaxed);
        }
    }
    pub(super) fn report() -> String {
        use core::fmt::Write as _;
        const NAMES: [&str; 13] = [
        NAMES_PLACEHOLDER
        ];
        let mut out = String::from("LF_MASK_CENSUS_BEGIN\\n");
        out.push_str("kernel\\tcalls\\tlanes\\tpassing\\tmid\\twide\\tnarrow\\tnone_calls\\tmid_unused_calls\\twide_unused_calls\\tnarrow_unused_calls\\n");
        for (name, row) in NAMES.iter().zip(&COUNTS) {
            if row[0].load(Relaxed) == 0 { continue; }
            out.push_str(name);
            for counter in row { let _ = write!(out, "\\t{}", counter.load(Relaxed)); }
            out.push('\\n');
        }
        out.push_str("LF_MASK_CENSUS_END\\n");
        out
    }
}
'''.replace('NAMES_PLACEHOLDER', names)
ablate = before[paths[1]].decode()
start = ablate.index('pub fn itx_shape_report() -> String {')
at = ablate.index('    out\n}', start)
ablate = ablate[:at] + '    out.push_str(&lf_mask_census::report());\n' + ablate[at:] + module
changed = {paths[0]: s.encode(), paths[1]: ablate.encode()}
try:
    for f, data in changed.items():
        (repo / f).write_bytes(data)
    patch = subprocess.check_output(['git', 'diff', a.baseline_revision, '--', *paths], cwd=repo)
    (repo / 'benchmarks/still-loopfilter-2026-09-07/experiments/mask-census.patch.gz').write_bytes(
        gzip.compress(patch, mtime=0))
    (root / 'mask-census-source-audit.json').write_text(json.dumps(dict(
        baseline_revision=a.baseline_revision, sites=sites,
        sources={f: dict(baseline_sha256=hashlib.sha256(before[f]).hexdigest(),
                         instrumented_sha256=hashlib.sha256(data).hexdigest())
                 for f, data in changed.items()},
    ), indent=2) + '\n')
    driver = root / 'mask-census-driver'
    driver.mkdir(exist_ok=True)
    (driver / 'target').symlink_to(a.shared_target.resolve(), target_is_directory=True)
    (repo / '.workongoing').write_text(datetime.now(timezone.utc).isoformat()
        + ' codex-still-parity building loopfilter mask census\n')
    subprocess.run([sys.executable, str(repo / 'benchmarks/tracker-sharding-2026-09-07/build.py'),
                    '--repo', str(repo), '--work-dir', str(root), '--label', 'mask-census',
                    '--driver', 'itx-census', '--modes', 'checked'], cwd=repo, check=True)
finally:
    for f, data in before.items():
        (repo / f).write_bytes(data)
    (repo / '.workongoing').write_text(datetime.now(timezone.utc).isoformat()
        + ' codex-still-parity mask census build ended; production source restored\n')
