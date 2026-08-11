# The derived rows-per-block shard rule, shipped as the default

**Status: FLIPPED. A picture plane's tracker block shift is now derived from its
row stride; the block-COUNT rule it replaced is the `bps-blocks` A/B arm. The
corpus is bit-identical at t=1 and t=8 on the DEFAULT build, and the four cells
the campaign has been mis-quoting move 2.10-2.27x -> 1.60-1.71x of dav1d.**

Prior art, not re-derived: `docs/SHARD_GRANULARITY.md` (PR #500, the mechanism)
and `docs/SHARD_SIZE_SWEEP.md` (PR #501, the 17-cell height sweep that built the
rule and measured it default-off). This round is the flip, the numbers for the
SHIPPED build, and the one cell #501 could not explain.

Record: `benchmarks/bps_rows_default_2026-08-11.{meta,txt}` + the TSVs beside it.

---

## 1. Why this had to happen, and what it fixes

#501 left the rule default-off for one stated reason — two picture sizes were
not enough and the error is a function of picture HEIGHT — and then measured the
17-cell height sweep that answers exactly that. The rule matched the best global
constant's geomean (0.9119 vs 0.9113 wall) while being **the only arm that
regressed no cell** (worst 0.998 against 1.004 / 1.013 / 1.031 for the three
rungs), and reached a three-shift step at 3840x256 that no rung on the ladder
offers.

Leaving it off had a cost beyond the lost time. `docs/SHARD_SIZE_SWEEP.md` §5d
tabulates **1.61x / 1.64x / 1.70x** against dav1d for the small-height tiled
cells; those are the GATED ARM's figures. Shipped `main` measured
**2.06x / 2.24x / 2.29x** on the same cells six hours later
(`benchmarks/cost_census_2026-08-10.meta` §1). Both records are internally
honest and the confusion is a foot-gun of the arrangement: the best number in
the file belonged to a build nobody ships. After this change they are the same
build.

## 2. What is NOT covered, first

* **x86_64 is not measured.** Every lock acquisition is dearer there, so the
  sign should carry and the magnitude should not. Reproduced only as a clippy
  target locally.
* **4:4:4, 12-bit, film grain, and inter/video content are not measured.** All
  cells here are 8-bit and 10-bit 4:2:0, all-intra, one key-frame OBU
  re-decoded. 4:4:4 chroma planes are luma-shaped, so the rule treats them like
  a wider picture — argued, not measured.
* **`ROWS_PER_BLOCK_MIN = 4` is still fitted on #501's grid with no held-out
  size.** That weakness is unchanged by shipping it; the ladder is kept
  buildable precisely so the next person can re-fit it.
* **Three of the eight cells are NULL** (§4): `c256x2048`, `v4k8tile` and — with
  a caveat — `c512x576`. Two of those three are *provably* null (§4a) and are
  used as the noise control.
* **The residual wide path is NOT removed the way a global rung removes it**
  (§6). At 1024x576 `w_shards` reads 153/frame in BOTH arms, because the derived
  rule only moves buffers that declare a stride and a global rung moves every
  buffer. Priced at about one point of wall by #500 §5e, and not attacked here.
* **`msb-5` is not bundled.** It is ~1% and not separated from zero at n=7.
* **`SLOTS` is not touched**, and the trap that goes with it (a 64-byte shard
  raising the shard-full rate) is re-checked rather than assumed: §6.
* **No reservation was widened anywhere**, per the four standing refutations.
* One box (Apple M4 Pro, 8P+4E, macOS, aarch64), one content class (photo), one
  quality point. Loop restoration executes zero blocks on every cell here, the
  same structural blindness the 4K gap vectors have.

## 3. What changed, exactly

```
    stride = (w + 127 & !127) << hbd,   + 64 when that is a multiple of 1024
    len    = stride * (h + 127 & !127)
    base   = ilog2(len / TARGET_BLOCKS)                 # the old rule, TARGET_BLOCKS = 256
    shift  = max(base, min(ceil(log2(ROWS_PER_BLOCK_MIN * stride)),
                           ilog2(len / MIN_BLOCKS)))    # the new one
```

`ROWS_PER_BLOCK_MIN = 4`, `MIN_BLOCKS = 32`, both unchanged from #501. Three
properties are worth restating because they bound the blast radius:

1. **Never finer than the old rule.** The `max` is what makes this a coarsening
   and nothing else, so no buffer gets MORE blocks than it used to.
2. **Only buffers that declare a stride move.** `declare_row_stride` is called
   from `Rav1dPictureDataComponent::from_parts` and nowhere else, so every
   non-picture instance keeps the block-count rule exactly. (That is also why
   the wide path does not go to zero — §6.)
3. **The serial and single-tile gates are the SAME gates**, restated in the rows
   rule rather than inferred from `base`. At `--threads 1`, or on a single-tile
   frame at any thread count, the derived rule provably returns the block-count
   answer, so nothing about a serial decode moves. That is why the whole t=1
   table in §5 is a noise control rather than a claim.

Polarity of the arms is inverted from the two previous rounds:

| build | rule |
|---|---|
| **default** | the derived rows-per-block rule |
| `--features bps-blocks` | the block-count rule that shipped before this commit — **the base arm** |
| `--features bps-1` / `bps-half` / `bps-quarter` / `bps-4` / `bps-8` | the ladder, each pinning its constant and turning the derived rule OFF |
| `--features probe-shiftpin` | `RAV1D_PIN_SHIFT="<stride>:<shift>,…"`, the only instrument that separates a LUMA shift from a CHROMA one |

`bps-rows` is gone. It was the default the moment this landed, and a feature flag
for the default is a flag that can never fail.

## 4. Liveness, read off the tracker

`--features __probe_bounds`, t=8, the `shifts` column is the tracker's own
per-instance shift, not a prediction:

| cell | base luma,chroma | head luma,chroma | `pct_row_wide` (4 CDEF sites) | `row_shards_max` lf |
|---|---|---|---|---|
| c1024x576 | 11, 9 | **13, 11** | 72.0-72.4% -> **0.00%** | 16 -> 4 |
| c512x576 | 10, 8 | **11, 10** | (see §7) | |
| c256x2048 | 11, 9 | 11, 9 | unchanged | unchanged |
| v4k8tile | 14, 13 | 14, 13 | unchanged | unchanged |

The last two rows are the point of §4a.

### 4a. Two cells in this grid are IDENTITY controls, and that is deliberate

At `c256x2048` (8.00 rows per block already) and `v4k8tile` (4.27) the derived
rule computes the SAME shift as the block-count rule, on every plane — verified
off the tracker, not predicted. So on those two cells `plain` and `bps-blocks`
are the same decoder doing the same thing, the true ratio is exactly 1.000, and
whatever they measure is this grid's noise floor.

**Every cell at t=1 is the same kind of control**, for a different reason: the
rule is gated on `shards >= SHARDS_CONCURRENT && tiles >= 2`, so at one thread it
returns the block-count answer for every size. The whole t=1 table is therefore
a floor measurement plus an unchanged gap-to-dav1d, and none of it is a claim
about the change.

A sweep that contains its own controls is the difference between "0.95 is a win"
and "0.95 is a win against a measured floor of X".
