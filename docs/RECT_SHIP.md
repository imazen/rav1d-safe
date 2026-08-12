# The `+1% at t=1` that blocked the rectangle record is CODE PLACEMENT, not the
# mechanism — measured, and NOT fixable by shrinking, moving or out-of-lining

**Read `docs/RECT_RECORDS.md` first.** PR #505 built the exact strided
rectangle, proved it sound, measured **−1.0% to −1.8% wall at t=8 on 5 of 6
multi-tile cells**, and left it default-OFF for one reason: **+1.0% to +1.3%
wall at t=1 on `v4k8tile`, 0 of 11 rounds below 1.000, in two sessions, in a
configuration where the rectangle path provably never executes.** It attributed
that to code size by elimination, took no `cargo asm` and no `cargo llvm-lines`,
and recorded that `#[inline(never)]` on `fill_rect` moved it 1.0103 → 1.0088 and
left the sign at 0/11.

This round took the instruments, and the answer is:

> **4,828 bytes of provably-dead text — `#[used]`-anchored functions nothing can
> call, in a build where `text_layout_diff.py` reports ZERO symbols resized and
> `text_symbol_diff.sh` shows every hot loop-filter symbol keeping a
> byte-identical instruction stream — costs +1.10% wall at t=1 on `v4k8tile`,
> 0 of 11 rounds below 1.000.**

Nine binaries that differ from `main`'s default by as little as **+1,132 bytes**
— including one that only *shrinks* the hot function — all land in **+1.1% to
+1.6%** on that cell with perfect or near-perfect signs, and are mutually within
±0.4%. A byte-identical copy of `main`'s binary reads **1.0006 (4/11)**.

**So the t=1 tax is the price of not being `main`'s exact binary, and the
rectangle does not pay it: measured against a same-source control, the rectangle
costs 0.9967 (7/9) at t=1 on `v4k8tile`.**

**Nothing ships.** The default stays off, `__lf_rect` stays, and this branch's
default build is verified byte-equivalent to the base commit's (`__text`
1,839,536 → 1,839,536, 0 symbols resized, 0 added). The reason is in §5: the
question "is the rectangle worth it?" is now separable from "is a new layout
worth it?", and the second one is a decision about a ~1% tax that this round can
measure but not remove.

Record: `benchmarks/rect_ship_{P,P2,R,cdefdouble}_2026-08-11.tsv`,
`benchmarks/rect_ship_{layout,llvmlines}_2026-08-11.txt`,
`benchmarks/rect_ship_2026-08-11.meta`.

---

## 1. What is NOT covered, first

* **The t=1 cost is EXPLAINED but NOT REMOVED**, and this round argues it cannot
  be removed by any code change: a pure source refactor that adds 1,132 net
  bytes and *shrinks* `LfBlock::open` by 1,976 pays the same tax as 19 KB of
  dead code. The task's acceptance bar for this round — the t=1 cell inside
  ±0.5% of `main` with a coin-flip sign — is not met by any arm, including arms
  that change no executed instruction anywhere.
* **The default was NOT flipped and `__lf_rect` was NOT deleted.**
* **One box** (Apple M4 Pro, 8P+4E, macOS 26.5.2, aarch64), **one toolchain**,
  **8-bit 4:2:0 only** in every timed grid. The layout finding is a statement
  about THIS binary on THIS core; nothing here says an x86_64 or a different
  microarchitecture behaves the same way, and the campaign should not assume it.
* **The mechanism is localised to "placement", not below that.** Whether it is
  I-cache set conflicts, 64-byte fetch alignment of a hot loop, or page
  behaviour is NOT established. `-C llvm-args=-align-all-functions` was NOT
  tried; neither was a linker order file; neither was `samply` on the two arms
  (the static instruments answered the question the profile was going to be
  asked, and the profile would not distinguish the sub-mechanisms either).
* **The `padfar` arm does not isolate LOCATION.** It emits its dead text from a
  module far from the loop filter, but the linker still places it at
  `0x10000f6e0` — before the loop-filter block at `0x100058000` — so it shifts
  the same symbols by the same kind of amount as the near pad. It is a second
  independent SIZE/module control, not a location control.
* **The CDEF pricing arm (§7) prices the population; it does NOT implement a
  rectangle there** and says nothing about what a rectangle would actually
  deliver, only what the whole population costs.
* **Miri was run on the DEFAULT feature set only**, both models: this round adds
  no `unsafe` and does not touch `crates/rav1d-disjoint-mut` at all (§8c).
* `text_q20`'s CDEF arm changes **zero** registrations, which makes it a
  null-control, not a measurement of screen content.

## 2. What the static instruments say — no clock involved

Three tools, all new here, all committed:

* `scripts/perf/text_layout_diff.py` — per-symbol `__text` size and placement
  diff of two Mach-O binaries, with the `Cs<base62>_` crate disambiguator
  normalised away (without that, every symbol reads as "only in head" and the
  diff is useless — two builds with different feature sets never share it).
* `scripts/perf/text_symbol_diff.sh` — disassembles ONE symbol out of each
  binary and diffs the instruction stream with addresses and address-relative
  operands normalised away.
* `cargo llvm-lines` on both arms, differenced per function.

### 2a. The executed code at t=1 is BYTE-IDENTICAL, except in one function

`bench_ab_decode`, release, `plain` (#505's default codegen) vs
`--features __lf_rect` at #505's final shape (`fill_rect` already
`#[inline(never)]`):

| quantity | plain | rect | delta |
|---|---|---|---|
| `__text` | 1,837,492 | 1,851,192 | **+13,700 (+0.75%)** |
| symbols in `__text` | 1,613 | 1,626 | +13 |
| **symbols present in both that changed SIZE** | — | — | **2** |
| LLVM IR lines (whole binary) | 497,012 | 499,558 | +2,546 (+0.51%) |
| IR functions that changed at all | — | — | **18** |

The two resized symbols are `LfBlock::<BitDepth16>::open` (**+212 bytes**) and
`LfBlock::<BitDepth8>::open` (**+8 bytes**) — `open` is what `fill` is inlined
into. Everything else in the filter chain is the same size, and it is the same
INSTRUCTIONS:

| 8bpc symbol (every timed grid is 8-bit) | instructions plain/rect | verdict |
|---|---|---|
| `loopfilter::loop_filter::<BitDepth8, DirectTaps>` | 2,647 / 2,647 | **identical** |
| `loopfilter::loopfilter_sb_direct::<BitDepth8>` | 1,638 / 1,638 | **identical** |
| `lf_apply::backup_lpf::<BitDepth8>` | 911 / 911 | **identical** |
| `safe_simd::loopfilter_arm::lf_compact_run_neon` | 4,193 / 4,193 | **identical** |
| `safe_simd::loopfilter_arm::lf_core::<16>` | 243 / 243 | **identical** |
| **`LfBlock::<BitDepth8>::open`** | **2,893 / 2,895** | **+2 insns, 7 new `bl`, register allocation churned** |

### 2b. The added text is 88% `fill_rect`'s twelve monomorphisations

`cargo llvm-lines`, differenced per function (18 rows change; 14 are new
functions, 2 are `open`, 2 are a `core::fmt` renumbering):

| function | IR lines | copies |
|---|---|---|
| `BorrowTracker::add_rect::<false>` | 0 → 432 | +1 |
| `LfBlock::fill_rect::<{4,6,8,12,14,16}>` × 2 bit depths | 0 → 2,061 total | **+12** |
| `BorrowId::from_pairs` | 0 → 29 | +1 |
| `LfBlock::<BitDepth8>::open` | 1,870 → 1,882 | 0 |
| `LfBlock::<BitDepth16>::open` | 1,587 → 1,599 | 0 |

In machine code those twelve monomorphisations are 984–1,080 bytes each,
≈ 12,000 of the 13,700 added bytes. **`fill` being `#[inline(always)]` and
monomorphised twelve ways is confirmed as the source of the added text** — #505
§5e guessed right about the source — but every byte of it is in functions t=1
never calls.

That leaves exactly two candidate mechanisms, and they are separable:

1. `open`'s +2 instructions and register-allocation churn — a real change to
   executed code.
2. Placement: 13.7 KB inserted into `__text` moves every hot symbol.

## 3. The layout control the campaign did not have

`loopfilter::text_pad` (near) and `src/text_pad.rs` (far) emit N KiB of
`#[used]`-anchored, never-called functions. With a pad on,
`text_layout_diff.py` reports **`resized_in_both = 0`** — not one symbol in the
binary changes size — and every hot loop-filter symbol's disassembly is
byte-identical. The only change is that everything after the pad shifts by its
size (measured: a uniform +4,828 bytes for `__pad_text`, moving every function's
64-byte line offset by 28).

#505's layout control was `plainC`: the same source built in a second worktree.
That differs only in embedded path strings and moves nothing, which is why it
read ±0.1% and made the band look tight enough for a 1% effect to be "real and
specific".

### 3a. Grid P — `v4k8tile` t=1, n=11, idle, ratios vs `plain`

`scripts/perf/tiled_wallcpu.sh` under `measlock`, never niced, two-point fit at
4 and 40 frames, round 0 dropped as cold, any round in which an arm saw a
foreign process above 25% CPU dropped whole (`rect_report.py`).

| arm | added `__text` | codegen changed? | wall/plain | [min..max] | sign |
|---|---|---|---|---|---|
| `plainB` — byte-identical copy | 0 | — | **1.0006** | [0.9950..1.0042] | **4/11** |
| **`pad1`** dead text | +4,828 | **no symbol resized** | **1.0110** | [1.0087..1.0139] | **0/11** |
| **`pad2`** dead text | +9,692 | **no symbol resized** | **1.0117** | [1.0091..1.0149] | **0/11** |
| **`pad3`** dead text | +14,556 | **no symbol resized** | **1.0109** | [1.0073..1.0161] | **0/11** |
| **`pad4`** dead text | +19,420 | **no symbol resized** | **1.0156** | [1.0060..1.0215] | **0/11** |
| `rect` (#505's arm) | +13,700 | `open` +8 B | 1.0147 | [1.0117..1.0172] | 0/11 |

**Dead text reproduces the whole effect**, at the same magnitude and the same
perfect sign, from 4.8 KB up. There is no size trend inside the band: 4.8 KB and
14.6 KB are indistinguishable.

### 3b. Grid P2 — the restructure isolated, two more pads, three cells

`plain2` is a source restructure (§4) that adds 12 small out-of-line functions
(+5,176 B) and **shrinks `LfBlock::<BitDepth8>::open` by 1,976 bytes**, for a net
`__text` of **+1,132**. `ship` = `plain2` + `--features __lf_rect`. `plain2B` is
a byte-identical copy of `plain2`. n=9..12 after dropping loaded rounds.

`v4k8tile` t=1:

| arm | vs `plain2` | sign | vs `plain` |
|---|---|---|---|
| `plain` (main's default) | **0.9861** | **9/9** | 1.0000 |
| `plain2B` byte-identical to `plain2` | 1.0006 | 3/9 | 1.0132 |
| `padsmall` (+2,348 B dead text) | 0.9975 | 9/9 | 1.0112 |
| `padfar` (+5,996 B dead text, other module) | 1.0001 | 4/9 | 1.0142 |
| `plain2` (restructure only, +1,132 B) | 1.0000 | — | **1.0141** |
| **`ship`** (restructure + rectangle) | **0.9967** | **7/9** | **1.0110** |

Two readings, and the second is the one that matters:

1. **Every binary that is not `plain` is +1.1% to +1.4% against `plain`, and
   they are all within ±0.4% of each other.** That includes a pure refactor that
   makes the hot function 17% smaller. `plain`'s own byte-identical copy is
   1.0006. `plain` is a layout local optimum on this cell and essentially any
   perturbation loses it.
2. **Against a same-source control, the rectangle is free at t=1**: `ship` /
   `plain2` = 0.9967, 7/9 — slightly favourable, well inside the `plain2B`
   band.

The other two t=1 cells scale with working set, which is what a placement /
I-cache story predicts and a "the rectangle costs something" story does not:

| cell t=1 | `plain` vs `plain2` | `ship` vs `plain2` | `plain2B` vs `plain2` |
|---|---|---|---|
| `v4k8tile` (4K, 8 tiles) | **0.9861 (9/9)** | 0.9967 (7/9) | 1.0006 (3/9) |
| `c1024x576` | 0.9929 (9/9) | 1.0016 (3/9) | 1.0008 (3/9) |
| `c256x2048` | 1.0004 (4/12) | 0.9993 (8/12) | 1.0004 (3/12) |

**The tax is +1.4% at 4K, +0.7% at 1024×576, and ZERO at 256×2048.**

## 4. The restructure, kept but inert by default

`LfBlock::fill`'s tile-threading tail — the rectangle attempt and the per-row
fallback — moved into one `fill_threaded::<W>` (`src/loopfilter.rs`). In the
rectangle arm it is `#[inline(never)]`, so `open` carries only the hull path plus
a call; in the DEFAULT arm it is `#[inline(always)]`, so the default build's
codegen is exactly what it was. That is gated, not asserted:

| gate | result |
|---|---|
| `text_layout_diff.py` default build vs base-commit binary | **`__text` 1,839,536 → 1,839,536, resized=0, added=0** |

It does what it was built to do — `LfBlock::<BitDepth8>::open` goes
11,560 → 9,584 bytes and, between `plain2` and `ship`, **the only 12 symbols
that change size are the 12 cold `fill_threaded` monomorphisations**; `open` and
every hot symbol are byte-identical in both bit depths. So the rectangle arm no
longer changes one executed instruction at t=1.

It is not enough, because the cost was never `open`.

## 5. Verdict, as fractions

| claim | status |
|---|---|
| the t=1 cost is CODE SIZE in `fill`, as #505 supposed | **NO** — `#[inline(never)]` already refuted the inlined-size form, and this round refutes the size form entirely: 2.3 KB and 19.4 KB of dead text cost the same |
| the t=1 cost is `open`'s changed codegen | **NO** — with `open` byte-identical (`ship` vs `plain2`) the cost is unchanged; and dead text with NO codegen change anywhere reproduces it |
| **the t=1 cost is code PLACEMENT** | **YES** — 9 independent binaries, +1,132 B to +19,420 B, near and far modules, refactor and dead code, all +1.1..+1.6% with 9/9-0/11 signs; byte-identical control 1.0006 |
| the cost is attributable to the rectangle mechanism | **NO** — against a same-source control the rectangle reads 0.9967 (7/9) at t=1 |
| the cost can be removed by shrinking / out-of-lining / relocating the code | **NO on all three, measured** |
| **it clears the round's acceptance bar (±0.5% vs `main`, coin-flip sign, n≥11)** | **NO — and no arm can**, including arms that change no executed instruction |
| the t=8 win replicates AT THE SHIPPED CONFIGURATION | **YES on the 1024-wide family, weaker breadth than #505** — see §6 |
| **should the default flip** | **NO, this round.** The mechanism is free; adopting it into this binary is not, and the ~1% is a layout draw this round can measure but not steer |

## 6. Grid R — the t=8 win, re-measured at the shipped configuration

Never inherited from the gated arm's numbers. `ship` = restructure + rectangle;
base `plain2` isolates the rectangle from the layout draw, and the `vs plain`
column is what a user would actually get. n=11..13 after dropping loaded rounds,
`foreign_max` 0–1.

| cell t=8 | `ship`/`plain2` wall | sign | `ship`/`plain2` CPU | sign | `ship`/`plain` wall |
|---|---|---|---|---|---|
| `c1024x192` | **0.9851** | **12/13** | 0.9922 | 9/13 | 0.9900 |
| `c1024x384` | **0.9762** | **12/12** | **0.9845** | **12/12** | 0.9924 |
| `c1024x576` | **0.9826** | **11/11** | **0.9844** | **11/11** | 0.9925 |
| `text_q20` | 0.9962 | 8/13 | **0.9739** | **13/13** | 0.9923 |
| `ui_q20` | 0.9930 | 7/12 | 0.9929 | 9/12 | 0.9988 |
| `c3840x256` | 0.9981 | 7/13 | 0.9964 | 9/13 | 1.0063 |
| `c256x2048` (the cell #505 was built for) | 1.0015 | 5/11 | 0.9962 | 8/11 | 1.0075 |
| `v4k8tile` | 0.9973 | 8/12 | 0.9951 | 11/12 | 1.0058 |

Control `plain2B` (byte-identical to `plain2`): 0.9962–1.0060 wall, signs
3/13–8/13, on every cell.

**Three of eight cells are −1.5% to −2.4% wall with 11/11 and 12/12 signs**, and
`text_q20` is −2.6% CPU at 13/13. That is a stronger sign count than #505's B3
grid on the cells that overlap, and a **narrower breadth**: `c3840x256`, which
B3 read at 0.9867 (6/7), is null here (0.9981, 7/13). The `ship`/`plain`
column shows the layout tax eating roughly half the win and flipping three cells
adverse — which is the whole shipping problem in one column.

## 7. Task 2 — the CDEF sites priced, without implementing anything

`docs/RECT_RECORDS.md` §7b asked for a `RAV1D_CDEF_DOUBLE` before anyone builds
a CDEF rectangle. Built (`--features __probe_cdef_double`,
`picture::{dup_rows, dup_rows_mut}` at the five sites §7b names, plus the
`cdef_apply` UV site): each CDEF per-row registration is taken TWICE in ONE
binary, immutably at the read sites and mutably at
`cdef_filter_block_*_neon`'s write site, dropped before the real one — sound by
construction, and it changes the count and nothing else.

**Instrument verified before any clock** (`probe-sites`, `lost = 0`):

| cell | regs/frame off | on | delta | delta as % of population |
|---|---|---|---|---|
| `c256x2048` t=8 | **569,690** (identical to #505's count) | 729,114 | **+159,424** | 28.0% |
| `c1024x576` t=8 | 529,092 | 650,948 | **+121,856** | 23.0% |
| `text_q20` t=8 | 199,287 | 199,287 | **+0** | — |

`text_q20` is the null control the arm needed and got for free: CDEF files **zero**
registrations there, so the arm must not move, **and it does not** (§7a).

### 7a. Timed, `cdefon` vs `cdefoff` in ONE binary, n=12..13

`cdefoff2` is a third alias of the same binary with the same environment — an
A/A identity control inside the grid.

| cell t=8 | wall | sign | CPU | sign | `cdefoff2` wall (sign) |
|---|---|---|---|---|---|
| `c1024x576` | **1.0409** | **0/12** | **1.0386** | **0/12** | 1.0010 (4/12) |
| `c256x2048` | **1.0134** | 1/13 | **1.0223** | **0/13** | 1.0000 (5/13) |
| `text_q20` (0 extra regs) | **1.0000** | 5/12 | 1.0009 | 5/12 | 0.9962 (6/12) |

### 7b. The marginal price, and what it implies

| cell | +regs/frame | +CPU ms/frame | **ns per registration** | share of the cell's wall |
|---|---|---|---|---|
| `c256x2048` t=8 | 159,424 | +0.522 | **3.27** | +1.34% wall for doubling |
| `c1024x576` t=8 | 121,856 | +0.642 | **5.27** | +4.09% wall for doubling |

Against #505's `LfBlock::fill` marginal price of **2.42–2.71 ns** and
`c256x2048`'s **19.71 ns/registration AVERAGE** (cited from
`docs/C256_CONTENTION.md` §7, not re-measured here).

* On `c256x2048` the five CDEF sites are **28.0% of the population and 0.52 of
  the tracker's 11.23 CPU ms/frame = 4.6%** (tracker CPU cited from
  `docs/C256_CONTENTION.md` §7). §7b's expectation was "≈2.7% of the tracker,
  probably nothing to win" — measured 4.6%, so the expectation was low by ~1.7x
  but the conclusion for THIS cell stands: it is small, and `fill`'s own 28.1%
  count cut delivered a coin flip here.
* **On `c1024x576` it is much bigger than §7b expected: +4.09% wall, 0/12.**
  A rectangle collapses 8 rows to 1, so ~87.5% of that population — an
  arithmetic ceiling of ≈ **3.6% wall** on the cell family where the `fill`
  rectangle already delivers −1.7% to −2.4%.
* **A registration at these sites is DEARER than at `fill`** (3.27 / 5.27 ns vs
  2.42–2.71) even though each files 7.3–8.0 records onto **1.000** shard lines
  against `fill`'s 8.98 onto 2.090. So "records per distinct shard line" does
  not by itself order the marginal prices, and the corrected cost model in
  `docs/AGENT_BRIEF.md` §6 should be read as "the cost tracks distinct shard
  lines" for a COUNT CUT's *value*, not as a formula for a site's marginal
  price.

**Verdict for the next round: the CDEF sites are NOT refuted, and they are the
better target than `c256x2048` ever was** — but the standing caveat applies with
full force: `fill`'s doubling priced its population at +3.37% wall on
`c256x2048` and the actual collapse delivered ~0 there. A doubling is an upper
bound on what a collapse buys, not a forecast.

## 8. Gates

Driver `scripts/perf/rectship_gates.sh`, mutations `scripts/perf/rectship_teeth.sh`,
logs `~/tmp/rectship/{gates,teeth}`. Nothing here is timed, so everything is
`nice`d and nothing takes the measurement lock.

### 8a. Correctness

| gate | result |
|---|---|
| **DEFAULT build's codegen vs the base commit's binary** (`text_layout_diff.py`) | **`__text` 1,839,536 → 1,839,536, `resized_in_both` = 0, `only_in_head` = 0** |
| **corpus, DEFAULT arm, t=1**, no `--skip-group` | **766 PASS + 2 SKIP**, mismatch=0 error=0 |
| **corpus, DEFAULT arm, t=8** | **766 PASS + 2 SKIP**, mismatch=0 error=0 |
| **corpus, `__lf_rect` arm, t=1** | **766 PASS + 2 SKIP**, mismatch=0 error=0 |
| **corpus, `__lf_rect` arm, t=8** | **766 PASS + 2 SKIP**, mismatch=0 error=0 |
| set-diff BY NAME (key `(group, name)`, value `(status, ACTUAL md5)`) vs `benchmarks/aarch64_md5_fixes_2026-08-07_final.tsv.zst` | **CLEAN on all four** |
| set-diff t=1 vs t=8 within each arm | CLEAN, both arms |
| `cargo test --lib`, release AND debug | pass, both |
| every measurement feature builds: `__probe_cdef_double`, `__pad_text`, `__pad_small`, `__pad2/3/4`, `__pad_far`, `__lf_rect`, `__lf_rect1`, `__probe_lf_hull`, `__probe_bounds` | rc=0, all 11 |
| `decode_md5_verify`, `thread_cleanup_test`, `tile_threading_overlap`, `reproduce_overlap`, `mt_stress`, plain AND `-- --ignored`, `__lf_rect` arm | pass, all 10 invocations |
| every timed arm's `CHECKSUM` before any timing | **9 arms × 2 thread counts → ONE md5 per cell**, on 4 cells; then 5 arms × 2 → one md5 per cell on all 8 cells |
| `cargo fmt --all --check` | rc=0 |
| clippy `-D warnings`: tracker `--all-targets`, tracker `--no-default-features --all-targets`, root `--lib`, `--lib --features {__lf_rect, __lf_rect1, __probe_cdef_double, __pad4, __pad_far}` | rc=0, all 8 |

### 8b. Test teeth, proven by planting

Every mutation restored from a `~/tmp` backup COPY, never `git checkout --`, and
verified byte-exact by sha256 AND `git diff --exit-code`.

| planted mutation | result |
|---|---|
| (control) default and `__lf_rect` arms | one md5, `248f0077…`, both |
| `fill_threaded`'s per-row loop reads `row + 1` | **CAUGHT** — the last row's guard is out of bounds and the decode PANICS (loud failure, not wrong pixels) |
| `fill_rect` reads `rect.row(h - 1 - row)` (rows reversed, always in range) | **CAUGHT** — md5 `dc9dace7…` ≠ `248f0077…` |
| **`panic!` planted in `text_pad::unit`, `--features __pad4`** | **md5 UNCHANGED — the layout pad provably never executes.** This is the teeth test the headline rests on: if the pad ran, §3 would be measuring work rather than placement |
| one of the five `dup_rows` call sites (`cdef_find_dir_8bpc_neon`) deleted, `RAV1D_CDEF_DOUBLE=1` | **CAUGHT with the exact expected value** — registrations 729,114 → 692,250, i.e. **−36,864**, precisely that site's measured population |
| `unsafe { core::mem::transmute(x) }` planted in `src/picture.rs` (no module-level forbid of its own) | **build FAILS at `lib.rs:13:12`** — `forbid(unsafe_code)` proven ACTIVE, not read. Restored, `git diff` clean |

**No `unsafe` was added to `rav1d-safe`, and `crates/rav1d-disjoint-mut` is not
touched by this branch at all** — the diff is `Cargo.toml`, `lib.rs`,
`include/dav1d/picture.rs`, `src/loopfilter.rs`, `src/text_pad.rs`,
`src/cdef_apply.rs`, `src/safe_simd/cdef_arm.rs`, `scripts/`, `docs/`,
`benchmarks/`.

### 8c. Miri

`cargo +nightly miri test -p rav1d-disjoint-mut --no-fail-fast --test <target>`,
ONE TARGET AT A TIME (Miri aborts the process on first UB and cargo stops at the
first failing TARGET, so a batch run lets later targets never execute and their
silence reads as health), Stacked Borrows and Tree Borrows, default features and
`__rect_1shard`. Driver `scripts/perf/rect_miri.sh`, record
`benchmarks/rect_ship_miri_2026-08-11.tsv`.

**Both models, both feature sets, CLEAN on every non-timeout target:**

| target | SB default | SB `__rect_1shard` | TB default | TB `__rect_1shard` |
|---|---|---|---|---|
| `--lib` | **42 passed** | **40 passed** | **42 passed** | **40 passed** |
| `narrow_release` | 1 | 1 | 1 | 1 |
| `soundness` | 25 | 25 | 25 | 25 |
| `wide_exclusion` | 1 | 1 | 1 | 1 |
| `guard_move_release` | 2 | 2 | 2 | 2 |
| `pic_buf_overflow` | **0 tests ran** | **0 tests ran** | **0 tests ran** | **0 tests ran** |
| `aligned_miri` | **0 tests ran** | **0 tests ran** | **0 tests ran** | **0 tests ran** |
| `shard_liveness` | **TIMEOUT(900s)** | **TIMEOUT(900s)** | **TIMEOUT(900s)** | **TIMEOUT(900s)** |

`shard_liveness` times out (rc=124) in all four configurations exactly as
`docs/AGENT_BRIEF.md` warns and as #504/#505 recorded; it is reported AS a
timeout, never as green. **CI's Linux Miri legs — whole package,
`--all-features`, which DO cover it — are green on this branch under both
models** (`Miri (Stacked Borrows)` 53m37s and `Miri (Tree Borrows)` 1h21m42s on
PR #506). `pic_buf_overflow` and `aligned_miri` select **0 tests** under these
feature sets and are reported as 0, never as green.

This is a formality rather than a discovery for this branch — it adds no
`unsafe` and does not touch `crates/rav1d-disjoint-mut` — and it was run at the
full #505 matrix anyway so the two rounds' tables are comparable.

## 9. The predicted mechanism for the t=8 win is REFUTED by its own instrument

The brief for this round proposed a prediction to test: the rectangle's win on
the OTHER cells cannot be a shard-line-count effect (a rectangle's shard set is
the hull's blocks, a *superset* of the per-row union), so it should be fewer
`add`/`remove` pairs and fewer lock acquisitions **where lines are not being
reused** — and if so, `rows_mean / row_shards_mean` at the `fill` site should
predict which cells benefit.

`--features __probe_bounds`, t=8, the `fill` site's `RECT` row, against grid R's
measured result:

| cell | `rows_mean` | `row_shards_mean` | **rows/shards** | `pct_row_wide` | `ship`/`plain2` wall (sign) |
|---|---|---|---|---|---|
| `c1024x192` | 9.01 | 2.322 | 3.88 | 0.00% | **0.9851 (12/13)** |
| `c1024x384` | 9.01 | 2.332 | 3.86 | 0.00% | **0.9762 (12/12)** |
| `c1024x576` | 8.92 | 2.318 | 3.85 | 0.00% | **0.9826 (11/11)** |
| `text_q20` | 9.68 | 2.457 | 3.94 | 0.00% | 0.9962 (8/13), **CPU 0.9739 (13/13)** |
| **`c256x2048`** | 8.98 | 2.090 | **4.30 — the highest** | 0.00% | **1.0015 (5/11) — the only null** |
| `c3840x256` | 9.04 | 3.666 | 2.47 | **23.07%** | 0.9981 (7/13) |
| `v4k8tile` | 10.31 | 3.174 | 3.25 | **20.21%** | 0.9973 (8/12) |

**The prediction does not hold.** The cell with the MOST distinct shard-line
touches removed per `fill` — `c256x2048`, at 4.30 against the winners' 3.85–3.88
— is the one cell that measures null on wall, which is the same inversion #505
found from the count side and `docs/C256_CONTENTION.md` §7 found from the
thread-scaling side. What the table does separate cleanly is the **refusals**:
both weak cells are the two with ~20–23% of rows too wide to be one record, and
all four cells with 0.00% refusals are the ones where the mechanism does
something. So `pct_row_wide == 0` looks necessary and is demonstrably not
sufficient, and `c256x2048` remains the exception every lever has hit.

`ui_q20` records no `RECT` row at all at this site (the counterfactual never
fires there) while still reading 0.9930 (7/12) — unexplained, and small enough
that it is reported rather than chased.

**So the ordering of the t=8 win across cells is still unexplained**, and the
honest statement is that the mechanism works where the geometry is
representable, does not work on `c256x2048` for reasons that are that cell's own
(four previous levers agree), and its size is not predicted by any per-site
quantity the bounds probe measures.
