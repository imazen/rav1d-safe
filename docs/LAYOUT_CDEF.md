# The code-placement lottery: can it be removed? And the CDEF rectangle, priced
# against a MEASURED transfer coefficient rather than its ceiling

**Read `docs/RECT_SHIP.md` first.** PR #506 established that the `+0.9..1.3%`
t=1 cost that kept the rectangle record default-off is **code placement, not
work**: 4,828 bytes of provably-dead `#[used]` text — zero symbols resized,
every hot loop-filter symbol byte-identical — costs the same `+1.1%` as the real
feature, and nine binaries differing from `main`'s by 1–19 KB all land in
`+1.1%..+1.6%` while a byte-identical copy reads `1.0006`.

That leaves a **±1.5% layout lottery underneath every t=1 measurement in this
repo**, with `main`'s current binary sitting on a lucky draw. This round asks
the two questions that follow.

*(§ numbers are filled in as each grid lands; the decision rule in §2 was
written and committed BEFORE the grid that decides it finished — see the commit
date on this file's first revision.)*

---

## 1. What is NOT covered, first

* **One box** (Apple M4 Pro, 8P+4E, macOS 26.5.2, aarch64), **one toolchain**
  (rustc 1.97.1), **8-bit 4:2:0 only** in every timed grid. Nothing here says an
  x86_64 or a different microarchitecture has the same lottery or responds the
  same way to alignment.
* **The sub-mechanism of the lottery is still not identified.** This round tests
  whether *forcing function alignment removes it*; it does not distinguish
  I-cache set conflicts from fetch-window effects from branch-predictor
  aliasing. A negative here closes "can alignment fix it", not "what is it".
* **A linker order file was NOT tried.** It is the remaining lever and it is
  platform-specific and maintenance-heavy; §5 prices what it would have to beat.
* **`-C llvm-args=-align-all-nofallthru-blocks`** (basic-block alignment) is
  reported only if whole-function alignment failed.
* No `unsafe` is added to `rav1d-safe`; `crates/rav1d-disjoint-mut` DOES change
  (the mutable rectangle guard), so it is Miri'd under both models and its CI
  legs actually fire on this branch — see §8.

## 2. The decision rule, pre-registered

The success criterion for Task 1 is **spread reduction, not a good draw**. For
each alignment family (a0 = none, a4/a5/a6 = 16/32/64-byte) the grid measures
four rungs — `plain`, and `+4,828 / +9,692 / +19,420` bytes of dead text — and
reduces the family to

  `SPREAD = max(rung medians) − min(rung medians)`, paired per round against
  that family's OWN unpadded build.

Alignment is worth shipping **for measurement quality alone** iff, on
`v4k8tile` t=1 (the cell with the largest tax):

1. `SPREAD` falls by at least **2x** against `a0`, **and**
2. the absolute cost `aNplain / a0plain` is **≤ 1.003**.

If (1) fails, that is a clean negative and it goes in `docs/AGENT_BRIEF.md` §6.
If (1) holds and (2) fails, the trade is reported with both numbers and no
default changes.

## 3. Grid L — alignment × pad rungs

*(filled in below)*

## 4. Grid D — the CDEF question, decided on a measured transfer coefficient

## 5. The rectangle default

## 6. Gates
