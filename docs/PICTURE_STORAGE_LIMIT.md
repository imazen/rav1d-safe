# Native picture-plane admission

`Settings::max_picture_bytes: Option<usize>` admits the sum of requested native
plane allocations before a picture allocates its first plane. The total includes
native row padding and alignment, reference frames, reconstruction pictures,
film-grain copies and pictures retained by callers. `None` retains the existing
pooled allocation path; `Some(0)` refuses every picture.

This is a scoped storage contract. It does not bound decoder workspace, input,
metadata, thread stacks, allocator overhead or excess allocator capacity. It is
not a bound on process RSS. The existing pixel-count limit remains independent.
The managed C-FFI build refuses this new requirement at construction because
its custom allocation callbacks have not been qualified for it.

The reservation belongs to the native picture data, shared by its references
and frame clones. It remains live after reset or decoder destruction while an
owner retains the picture. An allocator copied from a retained picture shares
the same budget. Refusal reports the traced native
`Error::Other("native picture storage limit exceeded")`; the existing exhaustive
public error enum is unchanged.

Bounded mode frees plane vectors instead of returning them to the reusable
pool. Otherwise the charge could disappear while cached storage remained
allocated. This trades cache reuse for a smaller ownership contract. Its cost
must be measured separately; these tests establish no throughput claim. A
future charged cache must retain admission with the cached allocation and
admit any replacement before allocating it.

`just qualify-picture-budget` verifies exact admission boundaries across
native plane geometries, atomic contention and overflow, unwind cleanup,
cloned/retained owners, allocation through copied pictures and real 10-bit
video on one and four workers. Tracked and untracked builds pass; the C-FFI
rejection gate, existing packet/provenance checks, strict Clippy, WASM check,
doctests and API capture also pass. The doctest run retains the pre-existing
ignored low-level examples.

Validation on 2026-10-25: run-heavy rc=0, 81 s, peak RSS 1.15 GiB,
minimum available 14231 MiB, peak load 2.43. This is the test/build process
receipt, not a measurement of the decoder's picture budget.

The existing generated-vector gate now checks both allocation modes against
the same committed ASM-derived MD5s: 28 original streams, five CPU settings,
grain enabled/disabled and pooled/bounded storage. It passed on 2026-10-25
without changing any oracle. `just picture-budget-golden` and strict benchmark
Clippy: run-heavy rc=0, 19 s, peak RSS 1.09 GiB, minimum available 15231 MiB,
peak load 1.14.

`just picture-budget-bench` compares pooled, a second pooled control and bounded
allocation in interleaved zenbench groups. It uses committed small 8/10-bit
inter and grain streams plus the 1024×1024 tiled still, on one and four workers.
Each sample primes its decoder before timing, then reuses and resets it between
passes. That makes idle buffer reuse part of even a single-call sample. Both
allocation modes must match every visible sample's
digest before timing. These controls do not establish production-video FPS;
the benchmark results must be recorded separately before making a cost claim.

The native grain lifetime test also exercises a refusal after reconstruction:
one picture fits but the separate grain output does not. At twice that
allowance, output succeeds; a returned frame retained across reset then blocks
the next reconstruction-plus-grain pair until released. After release the same
decoder succeeds again. Both 8/10-bit fixtures and one/four workers pass in
tracked and untracked builds. The test drains with `end_input`/`receive` so
reference release is tested separately from reset (`flush` drains and resets).

The expanded `qualify-picture-budget` gate passed: run-heavy rc=0, 102 s,
peak RSS 1.15 GiB, minimum available 14434 MiB, peak load 3.04. This is
build/test-process usage, not native plane consumption.

The first complete exploratory timing pass used a fresh decoder per sample
without priming. The 1024×1024 groups received only one timed call, so they
could not measure warm pool reuse. Those results are retained as a cold-sample
pilot, not evidence about the cache tradeoff. The benchmark now primes all
three arms before timing; fresh qualified measurements are required for that
question.

## Warm allocation controls, 2026-10-25

The primed benchmark at `027cb22428b467bc066b74aa30585a9dc36a5f93` completed
in both tracked and untracked builds. This measures the entire reset/decode/drain
pass, including allocation, after priming each decoder. It is one exploratory
run per build on a Ryzen 9 7900X with unrestricted CPU scheduling; frequency was
not pinned. It does not establish production-video throughput or noninferiority.

Bounded allocation has a measurable cost in some threaded fixtures. In the
untracked build, the four-worker 10-bit inter fixture reports +2.35%
(95% interval +1.48% to +3.21%) and the four-worker 10-bit grain fixture reports
+1.96% (+1.02% to +2.90%). Both intervals exceed zenbench's configured 1%
practical threshold. Their second pooled controls are +0.17% (-0.76% to +1.14%)
and +0.72% (-0.27% to +1.68%), respectively. Other workloads include small
positive intervals and intervals crossing zero; they do not prove zero cost.

The table preserves zenbench's emitted relative-time statistic and 95% interval
against the first pooled arm. It is not a ratio recomputed from rounded means.
The duplicate pooled arm measures experimental variation. Results have no
multiple-comparison correction. The console's "CI crosses zero" footnote is
not reliable here: zenbench 0.1.10 also emits it when the numeric interval
excludes zero but does not clear its practical threshold. The numeric intervals
and full machine-readable output are retained.

| Build | Fixture/workers | Pooled control vs pooled | Bounded vs pooled |
|---|---|---|---|
| tracked | inter8_128x128/1t | -0.96% [-1.64% -0.25%] | -0.85% [-1.40% -0.31%] |
| tracked | inter8_128x128/4t | +0.28% [-0.43% +0.99%] | +0.92% [+0.10% +1.74%] |
| tracked | inter10_96x96/1t | -0.23% [-1.42% +0.94%] | -0.37% [-1.40% +0.69%] |
| tracked | inter10_96x96/4t | -0.28% [-1.09% +0.52%] | +1.36% [+0.54% +2.17%] |
| tracked | grain8_128x128/1t | +0.05% [-0.87% +0.96%] | +0.03% [-0.81% +0.88%] |
| tracked | grain8_128x128/4t | -0.05% [-0.81% +0.68%] | +0.36% [-0.45% +1.14%] |
| tracked | grain10_128x128/1t | -0.45% [-1.33% +0.42%] | -0.15% [-1.05% +0.77%] |
| tracked | grain10_128x128/4t | +0.39% [-0.41% +1.17%] | +0.57% [-0.17% +1.34%] |
| tracked | tiled_1024x1024/1t | -0.27% [-0.55% +0.01%] | -0.31% [-0.57% -0.05%] |
| tracked | tiled_1024x1024/4t | +0.04% [-1.06% +1.13%] | +1.23% [+0.24% +2.22%] |
| untracked | inter8_128x128/1t | -0.38% [-1.30% +0.51%] | -0.25% [-1.06% +0.58%] |
| untracked | inter8_128x128/4t | -0.46% [-1.36% +0.43%] | +0.15% [-0.70% +0.99%] |
| untracked | inter10_96x96/1t | -0.21% [-1.56% +1.12%] | -0.78% [-1.96% +0.38%] |
| untracked | inter10_96x96/4t | +0.17% [-0.76% +1.14%] | +2.35% [+1.48% +3.21%] |
| untracked | grain8_128x128/1t | -0.29% [-1.15% +0.59%] | -0.18% [-0.95% +0.59%] |
| untracked | grain8_128x128/4t | -0.19% [-1.18% +0.79%] | +1.06% [+0.09% +2.02%] |
| untracked | grain10_128x128/1t | -0.32% [-1.21% +0.59%] | -0.03% [-0.91% +0.90%] |
| untracked | grain10_128x128/4t | +0.72% [-0.27% +1.68%] | +1.96% [+1.02% +2.90%] |
| untracked | tiled_1024x1024/1t | -0.21% [-0.71% +0.29%] | -0.25% [-0.64% +0.15%] |
| untracked | tiled_1024x1024/4t | -0.59% [-1.53% +0.35%] | +0.95% [-0.26% +2.19%] |

Full data and commands:
[tracked](../benchmarks/picture_budget_warm_tracked_2026-10-25.txt),
[untracked](../benchmarks/picture_budget_warm_untracked_2026-10-25.txt).
Every visible sample digest matched between allocation modes before timing.
The earlier unprimed pilot remains separate.

Process receipts, including any build work:

- Tracked: run-heavy rc=0, 309 s, peak RSS 0.38 GiB, minimum available
  15959 MiB, peak load 1.46; `/usr/bin/time -v` maximum RSS 398904 KiB.
- Untracked: run-heavy rc=0, 318 s, peak RSS 0.37 GiB, minimum available
  15888 MiB, peak load 1.34; `/usr/bin/time -v` maximum RSS 386744 KiB.

These are process peaks, not measurements of admitted plane bytes. The tested
fixtures are small inter/grain sequences and one 1024×1024 still; long,
large-resolution, reordered streams with retained consumer frames have not
been timed. A charged reusable pool remains a possible optimization, requiring
allocation profiles and equivalent lifetime/admission tests before adoption.
