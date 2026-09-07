# Mixed 8×8 SIMD investigation — 2026-09-07

Follow-up to the retained [8×8 specialization and mixed 16×16 dispatch](README.md)
in draft [PR #528](https://github.com/imazen/rav1d-safe/pull/528).
The starting implementation is `b6fe78eb`. Its timer-only profiles still
show scalar ADST-8/DCT-8 and mixed 8×8 fallbacks. The shape census attributes
about 22% of 4K photo coefficient area to outer 8×8 fallback, which is a
usage count rather than an elapsed-time share.

## Correctness before timing

Simply connecting the existing fourteen mixed 8×8 kernels fails both new
scalar differential and CPU-token permutation tests. The first differential
failure is type 1 (`ADST_DCT`), eob 0, maximum positive coefficients; the
permutation test also fails at eob 0 with ordinary random coefficients.
`wire8-original-kernel-check.patch.gz` and `wire8-original-kernels.log.gz`
preserve the failed probe. No timing binary is built from those kernels.
The baseline checked dispatch declines these kernels, so this
finding does not establish wrong output on the existing checked decode path.

The row pass reads row-major data, while decoder coefficients are stored
column-major. It also omits the 8×8 intermediate `(x + 1) >> 1` and clipping.
Correcting both steps makes the direct sweep pass all 4,928 cells, along with
the existing 5,824-cell 16×16 sweep. Each shape also passes 504 SIMD and 336
declined token cells on this host. The tests call production dispatch and
the real scalar fallback, rather than a separately transcribed oracle.

Coefficient values are placed only in the scan-reachable prefix, with a
nonzero final coefficient. Cases include full i16 endpoints, random values,
scan boundaries, destination offsets/strides, whole-buffer sentinels, and
coefficient tails. Bounds-checked prefixes validate the complete 8×8 footprint
before Flex access: `7 * stride + 8` bytes with checked arithmetic and 64
coefficients. This also protects the newly used kernels when `unchecked` is
enabled; invalid views must panic before changing either buffer.

The initial invalid-view fixture accidentally used the exactly valid span
127 for stride 17. Its expected-panic assertion failed; the fixture now uses
126 to omit the last pixel. The failure and correction are preserved in
`wire8-prefix-tests.log.gz` and `wire8-test-boundary-correction.json.gz`.
The corrected implementation passes all 81 checked release library/fixture
tests, including the new negative calls and exactly sized positive control.

## Candidates

`wire8-col` fixes the existing scalar row pass and uses the existing SIMD
column pass. Its immutable binary and source audit are recorded before the
next change. `wire8-row` additionally uses the existing SIMD row DCT/ADST
helpers. Identity8's doubling followed by `(2*x + 1) >> 1` recovers every i16
input exactly, so that row pass only converts and transposes coefficients.
The column pass and validated input prefixes remain the same.

All small raw evidence is indexed by `results/index.json` in this directory;
large binaries, source snapshots, and inputs remain at the recorded scratch
paths under `/home/lilith/tmp`, without a verified off-host backup. These
experiments use the frozen two-source development seed and do not satisfy
the full corpus or parity acceptance protocol.

## Short paired screen

Five rotated paired rounds, 150 ms calibrated upstream work, ten cells and
four arms: all 200 measured runs match upstream rav1d and independent dav1d.
The baseline is the retained mixed 16×16 implementation. Changes below are
paired median elapsed-time changes, not ratios of independently taken medians.

| Input | Workers | Column SIMD vs baseline | Row+column SIMD vs baseline | Row+column vs column only |
|---|---:|---:|---:|---:|
| photo-2k-min | 1 | -5.98% | -6.51% | -0.30% |
| photo-2k-min | 8 | -6.71% | -6.05% | +0.96% |
| photo-4k-t8 | 1 | -5.27% | -6.25% | -1.11% |
| photo-4k-t8 | 8 | -4.99% | -3.75% | -1.36% |
| photo-8k-t8 | 1 | -2.89% | -3.12% | -0.18% |
| photo-8k-t8 | 8 | -2.79% | -3.28% | -0.92% |
| map-4k-t8 | 1 | -1.79% | -2.36% | -0.38% |
| map-4k-t8 | 8 | -2.08% | -1.92% | -0.51% |
| map-8k-t8 | 1 | -2.36% | -2.99% | -0.39% |
| map-8k-t8 | 8 | -1.00% | -1.47% | -3.06% |

Both implementations are promising. The row+column variant also has 8,384
fewer ELF text bytes than the column-only variant (its increase over baseline
is 34,328 bytes, versus 42,712). Small row-versus-column timing differences
remain unresolved by this short screen; longer confirmation includes both.
The row+column candidate passes all five direct mixed-transform tests in
both checked and unchecked builds, including rejected-input calls that must
panic before writes. The final row+column candidate also passes the full 81-test checked release
gate and 766 conformance vectors at both one and eight workers (two existing
infrastructure exclusions per run). Final debug, alignment, and profile
checks are recorded below.

## Upstream profile attribution correction

The earlier profile reports leave 10–14% of upstream cycles under anonymous
NASM labels. `resolve_upstream_profile.py` resolves each label through ELF
FUNC intervals in the pinned upstream binary. Zero-sized NASM functions use
the next FUNC in the same section as an inferred end. All instances of a
same-name local symbol must imply the same group; otherwise the symbol stays
unattributed. All sampled anonymous labels in these three upstream reports
resolve to entropy routines. For example, `..@1565.loop` and `..@1565.end`
are within `dav1d_msac_decode_hi_tok_sse2`; `..@1034.branch_instr` is within
`dav1d_msac_decode_symbol_adapt4_sse2`.

Approximate sampled cycles normalized by the harness timed-frame count,
using the existing three-second profiles (before the 8×8 change):

| Workload | Checked entropy Mcycles/frame | Upstream entropy Mcycles/frame | Checked transforms Mcycles/frame | Upstream transforms Mcycles/frame | Checked loopfilter Mcycles/frame | Upstream loopfilter Mcycles/frame |
|---|---:|---:|---:|---:|---:|---:|
| photo-4k-t8-t1 | 268.30 | 229.03 | 68.49 | 3.36 | 61.68 | 7.25 |
| photo-8k-t8-t1 | 580.78 | 574.48 | 167.88 | 14.20 | 172.48 | 23.54 |
| photo-8k-t8-t8 | 684.64 | 625.24 | 170.25 | 9.41 | 331.82 | 39.26 |

These are single-profile estimates with no confidence bounds. Inlining can
move work between named groups; differences between baseline/candidate
profiles also show sampling and code-placement variability. They guide
priorities, not acceptance claims. Entropy remains a large share of both
decoders but is much closer in sampled cost per frame than total decode time.
This redirects the next investigation toward remaining transform and
loopfilter gaps, alongside tracker overhead, rather than assuming that the
largest checked self percentage is automatically the largest upstream gap.
The original reports and earlier unresolved classification remain archived.

The input-view argument is local and independent of worker count. For
`0 <= x,y < 8`, `y*stride + x <= 7*stride + 7`, within the checked destination
prefix. Checked span arithmetic excludes usize overflow. Column-major
coefficient indices and each eight-lane row load stay within elements 0–63.
The row/column temporary has exactly 64 i32 elements. Both source arguments
are exclusive mutable slices supplied by the existing dispatcher callback;
no new unregistered reference, shared mode switch, or extended lifetime is
introduced. Invalid-view tests exercise the prefix rejection in both checked
and unchecked builds. This argument covers these kernels, not every other
unchecked API in the crate.

## Longer four-arm confirmation

Nine rotated paired rounds, 500 ms calibrated upstream work: all 432 measured
runs match both references. No samples discarded. Peak host load 3.23,
minimum available memory 27,256 MiB. Baseline includes the previously
retained CDF, 8×8 fallback specialization, and 16×16 SIMD dispatch changes.

| Input | Workers | Baseline ms | Row+column ms | Upstream ms | Paired change | Upper ratio |
|---|---:|---:|---:|---:|---:|---:|
| photo-2k-min | 1 | 30.485 | 28.463 | 17.340 | -6.70% | 0.9364 |
| photo-2k-min | 8 | 32.550 | 30.575 | 20.064 | -6.27% | 0.9423 |
| photo-4k-min | 1 | 104.938 | 98.883 | 59.839 | -5.79% | 0.9444 |
| photo-4k-min | 8 | 106.568 | 100.570 | 65.931 | -5.85% | 0.9469 |
| photo-4k-t8 | 1 | 104.487 | 98.210 | 59.634 | -6.04% | 0.9413 |
| photo-4k-t8 | 8 | 24.472 | 23.821 | 13.170 | -3.35% | 0.9760 |
| photo-8k-t8 | 1 | 244.513 | 237.889 | 146.317 | -2.70% | 0.9745 |
| photo-8k-t8 | 8 | 56.975 | 54.363 | 31.061 | -2.47% | 0.9935 |
| map-4k-t8 | 1 | 107.033 | 104.683 | 57.262 | -2.20% | 0.9802 |
| map-4k-t8 | 8 | 31.975 | 31.529 | 14.225 | -1.02% | 0.9962 |
| map-8k-t8 | 1 | 314.861 | 306.970 | 163.702 | -2.55% | 0.9758 |
| map-8k-t8 | 8 | 87.617 | 85.384 | 38.665 | -2.47% | 0.9784 |

The row+column variant improves every selected cell under the one-sided
95% paired-median confidence rule. Against the column-only candidate, serial
changes are smaller (about 0.2–0.5%, all upper bounds below 1.0). Threaded
row-versus-column changes remain unresolved, including a +0.64% median for
minimum-tile 4K photo; no claim of universally faster SIMD rows is made.
The row+column implementation is selected for its serial gains and 8,384-byte
smaller text section. Both candidates' samples and comparisons remain archived.

Candidate/upstream paired medians still span about 1.52–2.22×. This remains
outside the parity goal, and the expanded corpus, holdouts, lifecycle,
memory, video, and remaining concurrency gates remain outstanding.

## Alignment controls and final validation

Both baseline and selected candidate are rebuilt with LLVM
`-align-all-functions=5` instead of 4. The symbol audit verifies every named
decoder text function starts on a 32-byte boundary in both binaries. Candidate
sources are restored and SHA-256 checked after the two builds. Nine rotated
paired rounds at 500 ms upstream calibration validate all 108 measured runs:

| Input | Workers | Paired change at alternate alignment | Upper ratio |
|---|---:|---:|---:|
| photo-2k-min | 1 | -6.64% | 0.9340 |
| photo-2k-min | 8 | -6.51% | 0.9444 |
| photo-4k-t8 | 1 | -6.25% | 0.9393 |
| photo-4k-t8 | 8 | -1.21% | 0.9951 |
| photo-8k-t8 | 1 | -2.95% | 0.9738 |
| photo-8k-t8 | 8 | +1.35% | 1.0182 |

The serial improvements and 2K/4K threaded improvements survive this control.
The 8K threaded result reverses, so a follow-up interleaves both baseline /
candidate alignments in one 15-round, four-arm cell: all 60 measured runs
match both references. Normal alignment again improves by 2.31% (upper
0.9922); alternate alignment has a +0.62% median (upper 1.0090). The original
alternate-alignment run was +1.35%. All original and follow-up samples are
retained. This supports keeping the substantial serial and smaller-input
gains, but **does not establish a placement-independent threaded 8K gain**.
The candidate's extra code and its interaction with the rest of the decoder
remain a concrete limitation of that cell.

The final implementation passes all 81 release library/fixture tests, all ten
selected debug tests, strict release/debug Clippy, root formatting, and
explicit formatting checks on the included dispatch and parity-test files.
The five mixed-transform tests also pass with `unchecked` enabled, including
invalid views that must panic before writes. Checked conformance passes
766 vectors at one worker and again at eight workers; each run has the same
two infrastructure exclusions. CPU-permutation smoke/CDF tests pass.

`wire8-row-api-source-audit.json.gz` compares explicit public declarations
in the four changed Rust files against `b6fe78eb`; none change. The private
macro and row helpers change, and existing kernel bodies are corrected.
This source declaration audit is narrower than rustdoc API comparison or
cargo-semver-checks. No dependency, borrow policy, or constructor change is
introduced. All 26 GitHub checks passed on the previous `b6fe78eb` head;
CI must rerun for this implementation. Local success is not a release gate
for every platform or the full performance corpus.

## Profiles of the selected implementation

Nine fresh timer-only profiles compare `wire16` baseline, selected row+column
candidate, and upstream at 4K/one worker and 8K/one and eight workers. Each
samples approximately three seconds at 499 Hz. All outputs validate. The
new `profiles-wire8` directory preserves the previous profile evidence;
profile helpers now refuse to overwrite earlier command/provenance records.
The following are approximate self Mcycles/frame with anonymous upstream
labels resolved by the same ELF method above:

| Input | Workers | Baseline transforms | Candidate transforms | Upstream transforms | Candidate loopfilter | Upstream loopfilter | Candidate tracker |
|---|---:|---:|---:|---:|---:|---:|---:|
| photo-4k-t8 | 1 | 76.32 | 45.86 | 4.73 | 57.55 | 6.84 | 45.47 |
| photo-8k-t8 | 1 | 129.10 | 115.15 | 16.24 | 175.14 | 28.26 | 89.21 |
| photo-8k-t8 | 8 | 145.88 | 106.11 | 7.72 | 312.59 | 42.81 | 260.34 |

Single-profile estimates have no confidence bounds and are affected by
inlining and sampling. They are priority evidence, not stage timing or
acceptance measurements. Remaining transform work, loopfilter computation,
and tracker/copy overhead deserve further measurement. The SIMD loopfilter
currently computes its wide and narrow alternatives even when their masks
select no output lanes. Measuring mask eligibility and actual transform EOB
sparsity can test whether avoiding unused arithmetic is worthwhile without
changing reservation footprints. Neither lead has been implemented or timed
in this change.

The commands and exit codes for final profiles/debug/lint checks are in
`wire8-row-final-checks.json.gz`. Build and benchmark provenance records
pin all binaries and inputs, while `wire8-row-source-audit.json.gz` pins the
selected production sources. Reproduce timings with the parent README's
standalone-consumer setup and `compare.py`, using `wire16-checked` as baseline
and `wire8-row-checked` as candidate. The four-arm comparison also retains
`wire8-col-checked` and upstream. The alternate placement builds use
`alignment.py --variant wire8-row --baseline-revision b6fe78eb`.
