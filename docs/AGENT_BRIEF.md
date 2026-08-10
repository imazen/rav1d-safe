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
- **Never edit a shell script while it is running.** Bash reads a script incrementally and keeps a
  file offset; an in-place edit that changes byte lengths makes it resume parsing at the wrong
  place. If you must change a tool that may be running (`measlock`), write a temp file and
  **`mv`** it into place — an atomic rename leaves the running process on the old inode.
- **`measlock --load-ok`** (or `MEASLOCK_LOAD_OK=1`) keeps the mutual exclusion and skips the
  wait-for-quiet. Use it when another agent holds the box with a long-running NON-timed job (a
  multi-hour `miri`): the quiet gate can never be satisfied and the default behaviour is the worst
  of both — wait 20 minutes, then run anyway. With `--load-ok` you MUST record `foreign_max` per
  row and report paired ratios, never absolutes.
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
| `TinyLock` backoff/yield (rav1d-safe) | null, measured twice |
| `block_mut` held row guards (rav1d-safe) | null — halving guard COUNT bought nothing; shard GRANULARITY was the whole win |
| `CompInterType` guard drop glue | not a real target (ICF-folded shared glue) |
| Allocator traffic past the first fix (zenav1-svt) | null despite malloc+memset at 15.5% self time — `sample` attributes page-fault/zone work there |
| Lazy deblock/CDEF application (zenav1-svt) | REFUTED — changes bytes; loop restoration reads post-CDEF recon |
| 2D-dot hoist in `compute_stats` (zenav1-svt) | 0.944-0.977x regression |
| Loop-filter reads as one strided HULL instead of `h` per-row guards (rav1d-safe) | **2.65x SLOWER at t=8** despite removing 3.46 M registrations/frame — the hull is 50-60 KB and lands on the tracker's wide path. `--features __probe_lf_hull` reproduces it |
| Restoring `rav1d_recon_b_intra`'s incremental destination addressing (rav1d-safe) | 1.021 vs 1.0060 — the hoist keeps a live 40-byte `ReconDst` across `decode_coefs` |
| Raising the loop filter's **H** batch cap (rav1d-safe) | structurally null: `LFCAP` measures ratio **1.000** at caps 4/8/16/32/64, because H's rectangle grows in the ROW direction, so a run of `n` groups costs `4n` registrations however it is split |
| The loop filter's **V** batch cap **with a fixed-wide scratch stride, a `params`-read threshold table, and an always-on write-back chunk loop** (rav1d-safe) | **+3.0% t=1 / +7.9% t=8** — and the cost was the MACHINERY, not the batch: an isolation arm holding the machinery at cap 4 was **+18.7% at t=1**. The cap itself is a win once the machinery is made free on the runs base could already open. `benchmarks/lf_vbatch_iso_2026-08-10_v1.tsv` |

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
  `CpuLevel::Scalar` does NOT disable safe SIMD — use the `__ablate` feature.
  Perf harness: `scripts/perf/verify_gap.sh`. See `docs/X64_APPLICABILITY.md` and issue #455.
- **zenav1-svt:** C reference in-tree at `reference/svt-av1`, prebuilt at `cbuild-static/`.
  Bit-identity grid: `rust/tools/byteid_fingerprint.sh` (120 cells). Decoder gate:
  `rust/tools/decode_gate_grid.sh` (aomdec + dav1d).
- **zenavif:** `just gates` before AND after any refactor. `cargo fmt --all` is BANNED (it reaches
  sibling path-deps) — use `just fmt`.
