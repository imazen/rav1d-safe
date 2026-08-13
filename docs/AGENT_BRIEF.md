# Agent brief — zen workspace

Read this FIRST. A workflow prompt that says "follow ~/work/zen/AGENT_BRIEF.md" means all of it
applies. Only the task-specific part will be in your prompt.

Written 2026-08-08 from what actually went wrong across ~20 agents on the rav1d-safe / zenav1-svt
campaigns. Every rule below cost real time to learn.

---

## 1. Environment (do this, don't rediscover it)

- **Worktrees:** work in a SIBLING dir — `/Users/lilith/work/zen/<repo>--<slug>`. Never
  `.claude/worktrees`. Never touch another agent's worktree or branch.
- **Scratch:** `~/tmp`. NEVER `/tmp` (it gets wiped mid-session; this destroyed a whole parity
  harness once).
- **`.workongoing`:** write one in YOUR worktree. Never delete anyone else's.
- **sccache is wired up globally** (`~/.cargo/config.toml`, 60 GiB). A fresh-worktree
  `cargo build --release --all-targets` is ~92 s at an 85% hit rate. **Do not set
  `RUSTC_WRAPPER`, do not set `CARGO_TARGET_DIR` to a shared dir** (lock contention), and do not
  "optimize" the build by disabling it.
- **Clean up:** when your branch is merged, `git worktree remove <path> --force`. Thirteen stale
  worktrees accumulated in one session and cost 14 GB.
- Tools already installed: `sccache cargo-nextest hyperfine samply zstd jq fd rg tokei cargo-asm
  cargo-flamegraph cargo-llvm-lines cargo-hack cargo-semver-checks cross`. Install more if useful.

## 2. Measurement — most of these were learned the hard way

- **NEVER `nice` a timed run.** On macOS `nice` maps to background QoS and lands the process on
  E-cores: ~40x wall-clock distortion. Builds and tests may be niced (it keeps P-cores free for
  another agent's measurements). Timed runs never.
- **No `-C target-cpu=native`.** Runtime dispatch is what ships.
- **For any t=1 A/B, build BOTH arms with `-C llvm-args=-align-all-functions=4`.** Without it a
  t=1 delta on a large-working-set cell is a draw from a **±1.4% code-placement lottery** that dead
  code reproduces exactly (#506), and `main`'s binary happens to hold a winning ticket, so every
  patch reads ~+1% against it. 16-byte function alignment collapses the spread across four dead-text
  rungs from **1.37% to 0.18%** on `v4k8tile` t=1 (32-byte 0.38%, 64-byte 0.75% — more is worse), for
  +0.55% of `__text`. It is a MEASUREMENT flag: it is not shown to stabilise t=8, and a published
  crate's consumers do not inherit it. `docs/LAYOUT_CDEF.md` §3.
- **A "same source, second worktree" build is NOT a layout control** — it moves nothing and reads
  ±0.1%. Use a `__pad_text` rung (dead `#[used]` text, 0 symbols resized) or a same-source arm.
- **Wrap every timed run in `measlock`** — `measlock <label> -- <your command>`. It takes an
  exclusive lock across ALL agents on this box, then waits for the box to actually go quiet before
  running. Without it, two agents measuring at once either corrupt each other's cells or both poll
  for "idle" forever: three rounds of the rav1d-safe campaign were degraded exactly that way (one
  capped at n=4 instead of n>=7; another had 354/360 rows load-tagged). Builds and tests do NOT
  need the lock — nice them instead, so they run on the E-cores while someone else measures.
  (Its release is ownership-checked: a holder whose lock was reclaimed as stale will NOT delete the
  next holder's lock. That bug existed for one day and let two agents measure at once — if you see
  `[measlock] not releasing: lock now held by ...`, that is the guard working, not an error.)
- **Interleave arms back-to-back with rotating order, median >= 5.** Discard and re-run any cell
  where a foreign process exceeded 25% CPU. State whether the box was idle.
- **A busy box invalidates absolutes but not paired ratios.** One campaign had 354/360 rows under
  load: its ratios held to three digits, its absolute ms were inflated ~2.5%. If you must measure
  on a busy box, report ratios and TAG the rows.
- **Two-point wall fits** (run at 2 and 20 frames, fit `total = a + b*frames`) remove process
  startup and agreed with an in-process instrument to <=0.5%.
- **Check the noise band before believing a sub-3% claim.** An agent reported `88.0 -> 85.6`
  (2.7%) whose own raw rows were base [85.50..91.11] vs head [84.89..91.50] at n=5 — overlapping.
  Re-measured: null. Print min/max per arm, not just the median.
- **Report the intercept, not just the slope.** Fit `total = alpha + beta*pixels` for BOTH time and
  encoded bytes. A ms/MP or bpp number without the intercept is meaningless — a 1 KB header is
  +0.4 bpp at 64x64 and ~0 at 4K. This is why the size sweep must include a TINY cell.
- **Sweep grid** (source-informing work): sizes tiny/256/1024/4K; quality q5-q100 with **low-q
  density equal to high-q** (web-focused work lives at aggressive settings); >= 3 content classes
  (photo, screen/synthetic, line-art). Not Kodak, not smooth gradients.
- **Verify your instrument sees the code.** Loop restoration is switched OFF in both 4K gap
  vectors, so an entire campaign's numbers contained no LR at all while it was active in 696/768
  corpus vectors. A null from a vector that never runs the code is not a result — prove execution.
  **And then check the opposite error.** When LR was finally measured (2026-08-10) it turned out to
  be the *least* of our problems — 1.9-2.3x of dav1d and <=9.3% of our wall — while the same sweep
  found the real-content gap is **3.7x-46x** and that **every LR-active cell is SLOWER at t=8 than
  at t=1**, 93% of it the borrow tracker. "This subsystem was never measured" predicts nothing
  about whether it is expensive; measure the SHARE before budgeting a port, and always take an
  ablation arm for the thing you suspect AND for the whole rest of the decoder.
- **Profile before optimizing, and profile self-time leaves**, not inclusive stacks. `samply` is
  installed and gives real call-tree attribution; macOS `sample` works too. Two sessions were
  saved by a profile contradicting the "obvious" target, and one was wasted by not taking one.
- **A self-time leaf can be the WRONG NAME for the cost — count the calls before you port
  anything.** Inlining collapses a dispatcher, a driver and a fallback onto one symbol, and a code
  path that is never taken costs nothing and appears nowhere. `<itx::itxfm::Fn>::call` carried the
  largest itx share at 4K 10bpc and read as "the scalar fallback for shapes above 16x16" — the
  named open item. A census at that seam (`examples/itx_shape_census.rs`, `--features __ablate`)
  put the fallback at **20 calls out of 272,949**, 0.15% of coefficient area, and **0** on the
  campaign's own gap vector; the symbol was holding the inlined SIMD dispatch instead. Days of
  kernel porting avoided by one counter. Same lesson as the LR blindness above, one level down:
  prove the code runs, and prove it runs *often*, before optimising the symbol it hides behind.
- **Never let a build rewrite a binary that a bind-mounted container has already exec'd.** A
  `docker run -v $REPO:$REPO .../target/.../my_binary` holds that path open through the VM's shared
  filesystem, and a `cargo build` landing a new file there mid-run makes the arm untrustworthy —
  page faults may resolve against the new inode. Copy the artifact to a name cargo never writes
  (`mi_head_pinned`) and run THAT. One #494 arm had to be discarded for exactly this.
- **Rust's stdout is BLOCK-buffered when redirected to a file, so a row count lags reality.** A
  harness writing TSV rows to stdout and progress lines to stderr will look stalled if you poll the
  TSV; poll the stderr side. Two #494 progress readings were wrong by 5x before this was noticed.
- **Never edit a shell script while it is running.** Bash reads a script incrementally and keeps a
  file offset; an in-place edit that changes byte lengths makes it resume parsing at the wrong
  place. If you must change a tool that may be running (`measlock`), write a temp file and
  **`mv`** it into place — an atomic rename leaves the running process on the old inode.
- **`--limit N` past the end of the stream halves the gap, silently and in dav1d's favour.** Our
  drivers re-decode one OBU exactly `n` times whatever you ask; `dav1d --limit N` stops at end of
  stream. A two-point fit then divides a SHORT total by a LONG frame delta: `n_hi = 24` on a
  16-frame IVF made dav1d read **94.1 ms/frame instead of 152.1** (×24/13) and would have put the
  4K t=1 gap at 1.96x instead of 1.21x. Count the stream's frames and refuse `n_hi > n_frames`
  (`scripts/perf/tiled_wallcpu.sh` now fails loud). Caught only by disagreeing with a prior record —
  so when a new harness makes the other decoder look suddenly *better*, suspect the harness.
- **A stage's measured cost under a SPINNING lock is not that stage's opportunity.** rav1d-safe's
  `TinyLock::lock_slow` spins on a relaxed load and never yields (its `yield_now` is behind
  `__probe_lock_backoff`), so a waiting core burns CPU and a stage's cost partly reappears
  elsewhere when its work is removed. Measured: at t=8 on a tiled 1024x576 vector, ablating deblock
  saves 0.250 ms/frame and ablating CDEF 1.811, but ablating **both** saves 4.978 — super-additive.
  With the tracker compiled out the same arms are additive to 0.6%. **Test additivity** (ablate each
  and then all) before pricing any single-stage change on a contended path.
- **`(plain − untracked) / registrations` is a LOWER BOUND on what removing a registration buys, not
  a point estimate — and it is per cell, never a constant.** The census refuted the 4.5-6.4 ns band
  as a cross-cell constant (2.78 at screen text t=1 to 18.78 at `c256x2048` t=8); #502 then measured
  what an actual removal is worth and it came in ABOVE the cell's own rate on every cell that moved:
  10.02 vs 2.78 ns at `text_q20` t=1, 6.21 vs 2.78-3.89 at `ui_q20` t=1, 5.05 vs 4.72 at `ui_q20`
  t=8. The multiplier runs **1.07x to 3.60x** and is NOT uniform — largest where the census rate is
  smallest (3.60x at screen text t=1) and barely 1.07x at `ui_q20`/`c1024x576` t=8, so do not carry
  one factor across cells. Same mechanism as the row above: the removal also makes the
  entropy/kernel buckets cheaper. **So price a count cut with the cell's own rate and expect to
  beat it; if you do NOT beat it, the cell is contention-bound and a count cut is the wrong lever**
  (`c3840x256`: 0.77 measured against 8.12 nominal, and it measured null). Two independent removals
  on `ui_q20` t=1 agree to 5% (6.21 ns here, 5.9 ns for #492's read-side part one), which is the
  campaign's best calibration for screen content.
- **An instrument that reads zero at `--threads 1` may not be measuring zero.** rav1d-safe's whole
  task/stage census lives inside `rav1d_task_run`, which `n_tc == 1` never enters, so every stage
  counter is 0.000 at t=1 — not "free", *not executed*. A t8/t1 ratio from such a probe does not
  exist; use t=2 or t=4 as the low arm, which is the better comparison anyway (same code path, so
  the ratio isolates adding workers from switching paths).
- **macOS `sample` samples PARKED threads.** `__psynch_cvwait` was 37.3% of leaves at t=8 and 0.0%
  at t=1, so leaving idle in the denominator deflates every busy symbol by exactly the amount the
  pool sleeps — the opposite of what you are attributing. Bucket idle separately and normalise on
  busy samples. (`scripts/perf/tiled_prof_report.py` does; `bucket_selftime.py` does not.)
- **A NICED counted run measures a DIFFERENT MACHINE, and only some counters notice.** The
  convention "counts don't need `measlock`, so nice them" is right for registration counts and
  WRONG for anything contention-shaped: `nice -n 19` maps to background QoS and lands all eight
  workers on the four E-cores, where they collide ~10x more often. The same build read
  `lockslow = 1,072/frame` niced and **96** un-niced. If a counter's value depends on two threads
  being on-CPU at once, run it un-niced under `measlock` like a timed arm
  (`scripts/perf/c256_contention.sh`).
- **When a counter and a profile disagree by 100x, reconcile them IN ONE BINARY before believing
  either.** `TinyLock::lock_slow` read 0.86-1.20 CPU ms/frame in the profile against a counted
  1,007-1,373 spin iterations/frame and a directly-measured **7.6 ns** per `core::hint::spin_loop()`
  on an idle core — a 150x contradiction. Running `probe-wide` and `sample` on the SAME process
  settled it: ~**627 ns** per iteration under real contention, 80x the idle-core price. Check the
  symbol's extent with `nm` before blaming misattribution (`lock_slow` is 32 bytes of text and
  `add_contended` is a separate symbol), and price the primitive with a 10-line microbenchmark
  rather than assuming — the assumption "a spin iteration is a few ns" was wrong by two orders of
  magnitude in exactly the place it mattered.
- **`measlock --load-ok`** (or `MEASLOCK_LOAD_OK=1`) keeps the mutual exclusion and skips the
  wait-for-quiet. Use it when another agent holds the box with a long-running NON-timed job (a
  multi-hour `miri`): the quiet gate can never be satisfied and the default behaviour is the worst
  of both — wait 20 minutes, then run anyway. With `--load-ok` you MUST record `foreign_max` per
  row and report paired ratios, never absolutes.
- **`grep -c` exits 1 when the count is zero, so `n=$(grep -c X f || echo 0)` yields `"0\n0"` and
  every `[ "$n" != "0" ]` test fires.** A watchdog written that way reports a panic on a clean log —
  the mirror of "silence is not success", and it cost one false alarm mid-#494. Use
  `n=$(grep -c X f); n=${n:-0}; [ "$n" -gt 0 ]`, and sanity-check a new watchdog against a log you
  KNOW is clean before trusting it against one you don't.
- **A "disjoint bands" tick has to compare the arms the CLAIM compares.** Printing
  ours-vs-dav1d disjointness for a claim about base-vs-head is trivially true for two different
  decoders: a green tick that can never fail. Same family as a vacuous `wide_exclusion`.
- **`kill`ing a `measlock` used to release the lock and keep running the payload UNLOCKED.**
  A bash trap handler *returns* unless it exits, and `trap cleanup EXIT INT TERM` never exited,
  so a TERM'd holder deleted its own lock, sat out the 20-minute politeness wait with no lock at
  all, and then started a timed sweep on top of another agent's — truncating the first run's TSV
  (2026-08-10, this happened to one agent's own two runs). Fixed in `~/bin/measlock`: `INT`/`TERM`
  go to a handler that kills the payload and exits, and the payload runs in the background so
  `wait` is interruptible. Two consequences that remain yours to handle: **the trap can take up to
  10 s to fire** (bash defers it until the in-flight `sleep 10` returns), and **`kill -9` on a
  holder leaves the lock dir behind** — the stale-reclaim path now `rm -rf`s it after re-reading
  the owner, but only on the *next* acquire, so verify `~/tmp/.measlock.d` is gone before you
  believe a relaunch is exclusive. Always confirm with `ps` that a killed run is actually dead.
- **A same-code control arm is the only way to know a sub-2% A/B's floor.** Two binaries that
  differ only inside `if BD::BPC == BPC16` still have different code LAYOUT, and that alone
  measured **+1.4% (0 of 6 rounds faster, sign-test p=0.031) on an 8bpc cell whose code path
  cannot have changed** — larger than every 10bpc effect in the same sweep. Put at least two
  provably-unaffected cells in every A/B and report them next to the claim; a "significant"
  sign test on the unaffected arm is the number that tells you to stop claiming.
  **Measured 2026-08-10; CONFIRMED and generalised 2026-08-11 by #506** (§6's code-placement row):
  the effect is not specific to a `BPC16` branch — provably-dead `#[used]` text with **0 symbols
  resized** costs the same **+1.1% to +1.6%** at t=1, so the floor is a ±1.5% *layout lottery* on
  every t=1 arm, and `--features __pad_text` / `__pad_far` are the purpose-built rungs for
  measuring it. The unaffected-cell control here and the byte-identical-binary control there are
  complementary, not alternatives: one bounds the CELL, the other bounds the BINARY.
- **Diff against your recorded base SHA, never against the `main` ref.** All worktrees share one
  `.git`, so another agent's merge silently turns up in `git diff main..HEAD` as reverse-deletions
  in your branch. `main` moved twice under one 2026-08-10 sweep.
- **Frame counts must be per cell.** A 64x36 frame decodes in 45 us, so the standard 2-vs-20 fit
  measures timer noise there. Scale them: 5,000/50,000 at the tiny end down to 2/16 at 4K.
- **Permission is not execution.** `enable_cdef = 1` in the sequence header at every size, yet CDEF
  executes **zero blocks** at 512x288. Read the profile, not the flag.

## 3. Never fabricate

- Every number you report is one you **ran**, with rep count and host.
- **Never extrapolate** across sizes, thread counts, or bit depths. Measure the cell you quote.
- Do not cite a number from an older record as if you measured it. If you must reuse a committed
  baseline, SAY SO explicitly and name the file — one agent's headline was ~8% optimistic because
  it cited a stale dav1d column after its own dav1d arm failed to produce data.
- **A measured negative, reported with its number, is a real deliverable.** It stops the next agent
  repeating it. Say "X measured 0.997x, n=9, reverted" — not silence.

## 4. Correctness gates

- **Set-diff BY NAME, never by count.** Put the actual hash in the key. A change that repairs 5 and
  breaks 5 shows as "no change" in a count.
- **Prove your test has teeth: plant a mutation, confirm the test FAILS, restore, confirm green.**
  A green test that cannot fail proves nothing. This caught four real defects and disproved one
  committed claim that "no regression gate exists" for a hazard a test catches in 0.03 s.
- **Assert liveness.** A parity test must fail if the interesting branches were never taken.
- **Prefer the actual reference implementation as the oracle**, not a transcription of it — a
  transcription drifts from the thing it is checking.
- Never relax a test, threshold, or expectation. Never add `#[ignore]`. If something fails, say so.
- `#![forbid(unsafe_code)]` stays. Prove it ACTIVELY when it matters (plant an `unsafe` block, see
  the build fail) — reading the attribute is not proof.

## 5. Reporting

- **Put what is INCOMPLETE or BROKEN before what works.** Name the variants you did NOT do.
- Fractions, not adjectives: "5 of 16 transform types", not "transforms done".
- Never call a stub complete. Never use test count as evidence of completeness.
- If you are the verifier: your job is to **contradict**, not to aggregate. Re-run the headline
  claims yourself. State contradictions bluntly; do not soften them into caveats.

## 6. Things already measured NULL — do not redo without a contradicting profile

| Idea | Result |
|---|---|
| **16bpc itx above 16x16** (32/64-point, `WHT_WHT`) — rav1d-safe #455 open item 5 | **not a target**: 20 of 272,949 16bpc transform calls on `L3840x2160_420_10b`, 0 on `v4k_8tile_10b`. Census, not a guess — `examples/itx_shape_census.rs` |
| **Porting more of the loop-restoration NEON tier** (rav1d-safe #455) | **not a target, measured 2026-08-10**: differenced on BOTH decoders via `RAV1D_INLOOP` / `dav1d --inloopfilters norestoration` (and our md5 is byte-identical to dav1d's at `all`/`norestoration`/`none`), LR is **1.88x** of dav1d's LR at 4K 8bpc and **2.25x** at 1080p 10bpc, while the rest of the decoder is 3.9-6.8x. LR is **9.3% of our wall at the worst cell** and 1.3-2.4% at the small ones; zeroing it entirely moves 4K t=1 from 3.96x to 3.59x. It is our BEST subsystem, not our worst. `benchmarks/lr_gap_2026-08-10.meta` |
| **"`looprestoration_arm.rs` has zero aarch64 intrinsics"** | **false since edb55a1**: 1,812 lines, 61 distinct NEON intrinsics over 249 call sites — more sites than `ipred_arm` (88), `cdef_arm` (44) or `filmgrain_arm` (30). Count before repeating the claim. |
| `TinyLock` backoff/yield (rav1d-safe) | null, measured twice — **but both were taken on the 4K multi-tile grid, where `lock_slow` is not the top symbol.** On real multi-frame content it is **39.5% of self time** at t=8 and the tracker as a whole is 22-93% of wall. That is a contradicting profile: the *target* is live again even if the *cheaper-lock* shape stayed refuted. `benchmarks/lr_gap_2026-08-10.meta` §3 |
| **The shard lock's WAITING POLICY, measured a third time ON the cell where `lock_slow` IS a top symbol** (rav1d-safe #455) | **CLOSED AT t <= 8 — null, 2026-08-11; OPEN UNDER OVERSUBSCRIPTION, see the row below.** Four arms (`probe-lock-{backoff,yield,relax,park}`: spin-64-then-yield, yield-every-iteration, exponential pause between loads, a real park via `parking_lot::RawMutex`) on `c256x2048` t=8, n=15, against an **in-grid identity control** (a byte-identical copy of the plain binary): every arm inside its ±1.8% band, best point estimate −1.4%, and that one is flat on five other cells. **Priced BEFORE believing any of it**: the whole waiting population is 1.20 CPU ms/frame = **10.7%** of the tracker's 11.23, so a *perfect* lock was worth 2.39x → 2.27x against a 1.4x bar. The park arm shows exactly where it goes — `sync` 4.75% → 0.69% of busy, idle 25.3% → 29.0%, wall 0.9922, CPU 1.0002: **the waiting is real, removable as CPU, and not on the critical path.** An n=7 first pass read `lockbackoff` 0.9739 (6/7) and did NOT reproduce at n=15 (0.9896). **Scope, added 2026-08-13:** every arm here is t=8, and this record's own not-measured list (`benchmarks/c256_contention_2026-08-11.meta:128`) ends with `t=16`. `docs/C256_CONTENTION.md` §5 |
| **The same waiting policy UNDER OVERSUBSCRIPTION (threads > cores)** (rav1d-safe #455/#471) | **OPEN — not refuted, and the one arm that has been measured there WON.** PR #471 (records landed 2026-08-13, code NOT landed) ran spin-64-then-yield against the unbounded spin on `v4k_8tile`{,`_10b`} at n=16 on an **idle** box, every committed row `foreign_max = 0`: null at t=1/2/4/8 on every statistic (agreeing with the row above), and **0.955 (8bpc) / 0.960 (10bpc) at t=16**, with p90 and max moving by the same factor — the whole distribution shifted, not a tail clipped. t=16 is the only cell on this host where threads exceed cores (M4 Pro, 8 P + 4 E = 12 logical). Liveness: the 64-spin bound is reached 2,657-3,638 times per 6-frame run, and the worst single base acquisition burned **~110 us at t=16** against ~11.5 us at t=8. Aggregate spin work falls ~23% = 0.64 ms/frame of CPU against a 3.3 ms/frame WALL win, so most of the gain is **the holder getting a core back**, not the waiter's own burn. What would settle it: the t=12/24/32 ladder (queued, never got an idle window) and x86_64, where `spin_loop()` is `pause` and the scheduler is not heterogeneous, so none of this transfers. Do NOT re-litigate it at t <= 8. `benchmarks/park_not_spin_2026-08-09.meta` |
| `block_mut` held row guards (rav1d-safe) | null — halving guard COUNT bought nothing; shard GRANULARITY was the whole win |
| `CompInterType` guard drop glue | not a real target (ICF-folded shared glue) |
| Allocator traffic past the first fix (zenav1-svt) | null despite malloc+memset at 15.5% self time — `sample` attributes page-fault/zone work there |
| Lazy deblock/CDEF application (zenav1-svt) | REFUTED — changes bytes; loop restoration reads post-CDEF recon |
| 2D-dot hoist in `compute_stats` (zenav1-svt) | 0.944-0.977x regression |
| Loop-filter reads as one strided HULL instead of `h` per-row guards (rav1d-safe) | **2.65x SLOWER at t=8** despite removing 3.46 M registrations/frame — the hull is 50-60 KB and lands on the tracker's wide path. `--features __probe_lf_hull` reproduces it |
| Restoring `rav1d_recon_b_intra`'s incremental destination addressing (rav1d-safe) | 1.021 vs 1.0060 — the hoist keeps a live 40-byte `ReconDst` across `decode_coefs` |
| Coarsening a guard extent at t=8 in rav1d-safe, ANY site | **measure it first, do not build it**: `--features __probe_bounds` (`docs/BOUNDS_MAP.md`) prints each site's distance to the nearest concurrently-live foreign WRITE. At t=8 every hot site already reserves exactly what it touches (`over_ratio = 1.000`, 1-16 bytes); `ctx.rs:99:27` has a concurrent write at gap **0**; `loopfilter.rs:710:14` has **60** bytes of room (to `loopfilter.rs:887:14`, its own write-back — *corrected 2026-08-10*; the 232 B first reported was `cdef_arm.rs:622:9`, the THIRD-nearest, a per-pair figure quoted as the site's). The 4K gap vectors under-report collision risk ~1000x vs the corpus, and **205 of 253 sites never execute there at all** (zero `mc`, zero LR). Per-site VERDICTs + the ranked candidate list: `docs/BOUNDS_MAP.md` Part 2 — but **re-price its ranking against `row_shards_mean`**, not registrations/frame (see the `LfBlock::fill` rectangle row below). A widening past a site's ceiling now FAILS `tests/guard_extent_budget.rs` / CI `extent-gate` |
| Raising the loop filter's **H** batch cap (rav1d-safe) | structurally null: `LFCAP` measures ratio **1.000** at caps 4/8/16/32/64, because H's rectangle grows in the ROW direction, so a run of `n` groups costs `4n` registrations however it is split |
| The loop filter's **V** batch cap **with a fixed-wide scratch stride, a `params`-read threshold table, and an always-on write-back chunk loop** (rav1d-safe) | **+3.0% t=1 / +7.9% t=8** — and the cost was the MACHINERY, not the batch: an isolation arm holding the machinery at cap 4 was **+18.7% at t=1**. The cap itself is a win once the machinery is made free on the runs base could already open. `benchmarks/lf_vbatch_iso_2026-08-10_v1.tsv` |
| **dav1d's tiled task scheduler as the explanation for the tiled t=8 deficit** (rav1d-safe #455) | **not the target, measured 2026-08-10**: `src/thread_task.rs` IS the port of dav1d's model, and profiling the tiled arm for the first time put **100% of the added-CPU half of the gap on the borrow tracker** — CPU growth t=1→t=8 is +27.3%/+10.5% for us, +4.4%/+3.6% with the tracker compiled out, +3.2%/+3.1% for dav1d. `deblock_cols` alone is +1.349 of +2.239 ms/frame (2.570x one stage) and `TinyLock::lock_slow` is 0.00% of busy self time at t=1 AND t=2 and 1.19% at t=8. `docs/TILED_SCALING.md` |
| **Single-tile or t=1 profiles as a proxy for tiled t=8 behaviour** (rav1d-safe) | **REFUTED — this is why the deficit went 3 measurements / 0 profiles.** Every stage is flat from t=2 to t=8 on a single-tile vector in BOTH arms, so the whole effect is invisible there: a one-tile frame never has two tile workers or two filter tasks live at once. If the claim is about tiling, the vector must be tiled (`~/tmp/t8gap/vec/*__t8.avif`) |
| **The post-tile filter tail** (rav1d-safe, the residual after the tracker) | **live, unpriced**: 34.1% of wall at a mean 3.22 of 8 workers at 1024x576/t=8 (17.5% at 2.59 with the tracker out), ~9-12% of wall structural at the ceiling. Instrument exists (`TAIL_CONC` in `probe-tasktime`). The axis is `sbh` vs `n_workers` — 9 rows on 8 workers is the hard case, 34 rows is visibly easier |
| **A registration COUNT cut on `c256x2048` t=8** (rav1d-safe #455/#502) | **null on both instruments, measured 2026-08-10**: 35,283 of 604,973 registrations/frame removed (−5.8%) reads wall **1.0030** (2/9 rounds faster) and CPU **1.0016** (4/7). That cell's tracker cost is CONTENTION, not count — **18.78 ns/registration** against 2.78-8.11 on every other cell, mostly `TinyLock::lock_slow` spin at 8 rows/block with 4 tile columns per block. It is also the ONE cell whose tracker-removed ceiling is band-safely under 1.4x (1.315), so it is worth attacking — just not by removing records. `docs/CTX_TL_SPLIT.md` §6c **— and "mostly `lock_slow` spin" is CORRECTED by the row below: it is 10.7%, and it is not recoverable.** |
| **A FINER shard block on `c256x2048` t=8** (letting the derived rule refine past the block-count answer instead of only coarsening) | **adverse and monotone, measured 2026-08-11**: against a pinned identity control, **0.9947 / 1.0221 / 1.1179 / 1.2026 / 1.2106** at 1/2/3/4/5 shifts finer (n=7, idle, 7/7 sign from −2 down), while one shift COARSER reads 0.9882. The wide path is **never reached at any rung** (`w_shards = w_blocks = w_full = 0` down to shift 6 on both planes), so the counter-force did not kill it — **shard-line FOOTPRINT** did: at shift 8 the multi-shard count is still exactly 1,684/frame and the four tile columns still share a block (256 B = one picture row), yet the arm is already **+11.8%**, purely because each worker now touches 8x as many distinct shard lines. `docs/C256_CONTENTION.md` §3-4 |
| **`c256x2048` t=8 as a CONTENTION problem at all** | **refuted, 2026-08-11 — it is a COHERENCE problem.** The registration count is *identical* at t=2/4/8 (569,690/frame, the tile-threading latch is already thrown at t=2) and the tracker's cost per registration **doubles with every doubling of workers**: 2.19 (t=1, 272,505 regs) → **4.52 → 9.18 → 19.71 ns**. At most 10.7% of that is waiting. So ≥89% is the UNCONTENDED `add`/`remove` pair getting slower as more cores touch the same shard — about one cross-core transfer of the shard's own cache line per registration. Four levers have now declined this cell (count, coarser, finer, waiting policy); **the next one must remove the SHARING**, i.e. register nothing (the `get_mut` direction) or key a record on its worker. And the prize is bounded: the tracker-removed ceiling here is 1.32x, so the cell asks for the tracker to be nearly FREE at t=8, not cheaper. `docs/C256_CONTENTION.md` §7 |
| **Registration DENSITY at `LfBlock::fill`, the decoder's largest site, as an exact strided-RECTANGLE record** (rav1d-safe #505) | **built, sound, live — and it CORRECTS THE COST MODEL IN THIS TABLE.** One exact record replacing `h` per-row ones cuts registrations 569,690 → 409,349/frame (−28.1%) on `c256x2048` t=8 and measures **null there** (0.9955 wall 4/7, 0.9908 8/15) while being **−1.0..−1.8% wall / up to −3.3% CPU on 5 of 6 OTHER multi-tile t=8 cells** (replicated, 10/10-11/11 signs) and **+0.9..+1.3% at t=1 on `v4k8tile`, 0/11 twice, where the path cannot execute**. Default-off for that last reason. **The row below's "~19.71 ns per registration" is an AVERAGE, not a marginal price**: doubling `fill`'s population in ONE binary (`RAV1D_LF_DOUBLE`, 0 of 25 rounds on the other side) prices a `fill` registration at **2.42-2.71 ns**, so the site is 31.7% of the population and 3.9-4.4% of the tracker's CPU. **The cost tracks DISTINCT SHARD LINES a worker visits, not records filed** — `fill` files 8.98 records per 2.09 shard lines — so a count cut at the DENSEST site is the least valuable one available. Price any future count cut with `__probe_bounds`' `row_shards_mean`, not with the registration count. `docs/RECT_RECORDS.md` |
| **The +1% t=1 cost that kept the rectangle record default-off** (rav1d-safe #505) | **NOT the mechanism — it is CODE PLACEMENT, measured 2026-08-11.** 4,828 bytes of provably-dead `#[used]` text, in a build where `text_layout_diff.py` reports **0 symbols resized** and `text_symbol_diff.sh` shows every hot loop-filter symbol keeping a **byte-identical instruction stream**, costs **+1.10% wall at t=1 on `v4k8tile`, 0 of 11**. Nine binaries differing from `main`'s by **+1,132 B to +19,420 B** — dead code, a pure refactor that SHRINKS the hot function 17%, near and far modules — all land in **+1.1%..+1.6%** with 9/9-0/11 signs and are mutually within ±0.4%, while a byte-identical copy reads **1.0006 (4/11)**. Against a SAME-SOURCE control the rectangle costs **0.9967 (7/9)** at t=1. The tax scales with working set: +1.4% at 4K, +0.7% at 1024x576, **0** at 256x2048. **Consequence for every future round: "same source, second worktree" is NOT a layout control** (it moves nothing and reads ±0.1%); use `--features __pad_text` / `__pad_far`, and judge a t=1 claim against a same-source arm, not against `main`'s binary. `docs/RECT_SHIP.md` |
| **The five CDEF registration sites** (rav1d-safe #505 §7b's ten-minute arm) | **priced, NOT refuted, 2026-08-11.** `RAV1D_CDEF_DOUBLE` (`--features __probe_cdef_double`) doubles them in ONE binary: **3.27 ns/registration on `c256x2048` t=8** (+159,424 regs/frame = 28.0% of the population, +1.34% wall, CPU 0/13) and **5.27 ns on `c1024x576` t=8** (+121,856, **+4.09% wall, 0/12**). Null control: `text_q20` files **zero** CDEF registrations and the arm reads 1.0000 (5/12). So on `c256x2048` the sites are 4.6% of the tracker (§7b guessed 2.7%) — small — but on the **1024-wide family they are +4.09% of WALL**, i.e. an arithmetic ceiling of ~3.6% for an 8-rows-to-1 collapse, on exactly the cells where the `fill` rectangle already pays −1.7%..−2.4%. Caveat that still binds: `fill`'s doubling priced +3.37% wall on `c256x2048` and its actual collapse delivered ~0 there. `docs/RECT_SHIP.md` §7 |
| **Whether the ±1.5% CODE-PLACEMENT LOTTERY can be REMOVED** (the question #506 left open) | **YES, most of it, at 16-BYTE function alignment — measured 2026-08-11.** Sweeping four dead-text rungs under `-C llvm-args=-align-all-functions={none,4,5,6}` (18 arms in one rotation, `v4k8tile` t=1, n=12), the rung SPREAD goes **1.37% → 0.18% (7.6x) at 16 B**, 0.38% at 32 B, 0.75% at 64 B; the byte-identical controls read 0.9996 (7/12) and 1.0006 (5/12), so the floor is ~0.05% and the unaligned spread is 20x it. **More alignment is WORSE** — 64-byte buys 1.8x for +2.66% of `__text` against 16-byte's +0.55% — so sweep N, never assume it. Cost: `a4plain/a0plain` = **1.0153** at 4K t=1, but `main`'s binary is the lottery WINNER; against the a0 family's own pad rungs the aligned build is **+0.34%**. **NOT shown to stabilise t=8** (one rung per family: ±0.95% aligned vs ±1.13% not). Use the flag for any t=1 A/B grid. `docs/LAYOUT_CDEF.md` §3 |
| **The five CDEF registration sites, COLLAPSED to one exact rectangle each** (rav1d-safe #506 §7's open item) | **BUILT AND IT PAYS: −2.0% to −2.5% wall at t=8 on the 1024-wide family, 9/10–12/12 signs, replicated in BOTH alignment families**, and composed with `fill`'s rectangle **−3.9% to −4.5%** — the largest t=8 win in the record. Null (1.0038, 4/11) on `text_q20`, which files zero CDEF registrations: the other-side control. It is ONE seam, not five sites — `for_rows` / `for_rows_mut` in `include/dav1d/picture.rs` — and the write side needed a MUTABLE rectangle guard the tracker did not have. `c256x2048` moved in the unaligned family (0.9794, 12/12) and NOT in the aligned one (1.0007, 6/12), so that cell STAYS on the declined list. `docs/LAYOUT_CDEF.md` §4 |
| **Reading a DOUBLING arm as a forecast for the matching collapse** | **it over-promises by 3.3–4.7x, and the discount is now measurable.** Pricing `fill`'s own population by doubling on the three cells where its collapse was already known gives `tau = delivered / ((1−1/rows) × doubling)` = **0.21–0.30**; applying that `tau` to the CDEF doubling predicted −0.9% to −1.6% and the collapse delivered −2.0% to −2.5% (under-predicted by 1.3–1.8x). So **price an unbuilt collapse as `(1 − 1/rows) × doubling × ~0.26` and expect a factor-2 band** — and do NOT price it with records-per-distinct-shard-line, which is 3.85–4.30 at `fill` and 4.13–4.18 at CDEF on the same cells and orders the two backwards. `docs/LAYOUT_CDEF.md` §4 |
| **rav1d-safe's filter chain vs dav1d's, single-threaded** | **the biggest number in the record and NOT a threading problem**: with the tracker compiled out our filter chain is 2.8x (1024x576) / 4.0x (4K) dav1d's at t=1, deblock alone **5.6x** (11.5 vs 2.07 ms/frame at 4K), and the ratio is flat across thread count. That is the compact copy-in/write-back the safe-guard model needs, not the tracker. Belongs to the single-thread campaign. `docs/TILED_SCALING.md` §6 |

**The meta-lesson from the top two rows: a large self-time share is not automatically a large
opportunity, and reducing the COUNT of an operation is not the same as reducing its COST.** The
loop-filter hull row sharpens it in the other direction: a count reduction bought with a WIDER
extent can be actively harmful. Price the extent, not just the count — and price the count on its
own by ADDING a duplicate rather than removing the original, which is sound for immutable
reservations and is the only arm that changes nothing else (rav1d-safe `RAV1D_LF_DOUBLE`).
**The last row adds a third clause: price the MACHINERY the count reduction needs.** A correct
count reduction whose new machinery costs more than the registrations it removes is a regression,
and you only find that out by building an isolation arm that KEEPS the machinery and REMOVES the
reduction.

## 7. Repo-specific

- **rav1d-safe:** corpus gate is `cargo test --release --test decode_md5_verify` (14 tests, ~40 s)
  or `examples/md5_inventory` for per-vector TSV. Baseline to diff against:
  `benchmarks/aarch64_md5_fixes_2026-08-07_final.tsv.zst`. Must stay 766/766.
  **`--skip-group film_grain` is no longer needed** — #479 is fixed, so
  `--threads 8` reaches 766/766 with no filter. If you find yourself skipping a group to make a
  run pass, that is a bug report, not a flag.
  `CpuLevel::Scalar` does NOT disable safe SIMD — use the `__ablate` feature.
  `md5_inventory --activity` only counts **itx, cdef, looprestoration** (the three `ablate::note()`
  sites); the other six columns are UNINSTRUMENTED and read 0 whatever ran.
  **`shard_liveness` under Miri is a BUDGET problem, not a hang**: every prior
  round recorded it as a timeout in all four configurations at 900 s, and at
  **1500 s it completes under Stacked Borrows (5 passed)**; Tree Borrows still
  runs out. Give it 1500 s before reporting a timeout (`docs/LAYOUT_CDEF.md` §6c).
  **To name the counterparty of an `overlapping DisjointMut` panic, build
  `--release -C debug-assertions=on` — NOT `--features probe-sites`.** Per-record `Loc`s
  (`ShardRecs::locs`) and the `track_caller` propagation through `picture.rs`'s `slice`/`slice_mut`
  helpers are gated on `any(debug_assertions, feature = "probe-sites")`, so the debug-assertions
  build names both sides for one non-atomic store per registration and no hash table. `probe-sites`
  adds a hash plus three atomic RMWs per registration and PERTURBED THE #494 RACE AWAY (358/358
  clean); the debug-assertions build reproduced it at a HIGHER rate (its two aborts were the 20th and
  27th vectors attempted) and named the writer on the first try. It also arms every `debug_assert!` in the decoder, which is how #494's invariant became
  a t=1 abort instead of a t=8 race.
  Perf harness: `scripts/perf/verify_gap.sh` for the 4K AVIF grid,
  `scripts/perf/verify_gap_ivf.sh` for real multi-frame streams — the 4K grid is **structurally
  blind** to loop restoration (both its vectors switch LR off), to thread-scaling inversion, and to
  every non-420 layout. Its `ARMS` take an `@<inloopfilters>` suffix so a filter's cost is
  differenced on both decoders at once. `probe-untracked` is the tracker-removed ceiling arm and is
  bit-identical. `examples/profile_ivf` takes `RAV1D_THREADS` (default 1) — needed because
  `bench_ivf_limit` exits before `sample` can attach.
  See `docs/X64_APPLICABILITY.md`, `benchmarks/lr_gap_2026-08-10.meta` and issue #455.
- **zenav1-svt:** C reference in-tree at `reference/svt-av1`, prebuilt at `cbuild-static/`.
  Bit-identity grid: `rust/tools/byteid_fingerprint.sh` (120 cells). Decoder gate:
  `rust/tools/decode_gate_grid.sh` (aomdec + dav1d).
- **zenavif:** `just gates` before AND after any refactor. `cargo fmt --all` is BANNED (it reaches
  sibling path-deps) — use `just fmt`.
