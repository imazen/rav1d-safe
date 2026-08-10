# The `&mut [u8]` recon-kernel refactor, written

Round of 2026-08-09. Base `main` @ `cebb97f`. Branch `perf/mut-recon-kernels`.

PR #481 priced this change and did not write it. This writes it, on a band that
is column-compact instead of full-plane, so the memory blocker #481 named is
paid down in the same pass.

---

## 0. What is NOT done — before anything that works

* **Inter frames are not converted, at all.** The scope is frames whose
  reconstruction is entirely intra (`KEY` / `INTRA_ONLY`). `mc`, OBMC,
  inter-intra blending and warp still write the shared picture through the
  tracker, and a frame containing them declines wholesale. Every writer of a
  recon plane has to move together — intra prediction, palette, CfL and the
  inverse transform all write the same pixels — so a subset conversion leaves
  the tracker on the path and measures null. Inter is the next coherent slice,
  and it is bigger than this one: **228 `PicOffset` parameters remain against
  the 39 converted**, and `src/mc.rs` alone holds 59 (§8 names them all).
* **The filter chain is not converted.** `loopfilter`, `cdef`, `looprestoration`
  and `filmgrain` genuinely cross tile boundaries and run after the stitch on
  the unified picture, exactly as before and with exactly today's tracking.
  They are 47% of the post-#474 registration population (#481 §3) and no
  picture-buffer design reaches them.
* **`src/ctx.rs:99` is untouched and is larger than everything here.**
  2,534,988 registrations/frame at `v4k_8tile` t=8 — `CaseSetter::set_disjoint`
  on 32-byte instances, mean extent 2.1 bytes. It is not a picture buffer.
* **Also declining:** `allow_intrabc`, frame threading (`n_fc > 1`), the
  `c-ffi`/`asm` configuration, `__simd_test`, negative strides, and a frame
  with no picture data. Each leaves `f.owned_recon = false`, i.e. today's path.
* **Not measured:** x86_64 and wasm32 (compile-checked only), `--features asm`
  and `--features c-ffi` (compile-checked only, and only cross — see §7b),
  `--features unchecked`,
  t=16, any vector below 4K, and **any vector with loop restoration live** —
  #455 item 4's structural blindness is unchanged.

## 1. The design, and why it needs no run-time proof

#474 gave each TILE a private full-plane `Rav1dPictureDataComponent` hanging off
`Rav1dFrameData`. Tile tasks reach that through `fc.data.try_read()` — a SHARED
reference — so the only mutation it can express is interior, which is the
tracker. That is why #474 kept the tracker and #481 could only remove it by
setting `tracker: None`, which is unsound.

The band here is a plain `Vec<Chunk>` field of **`Rav1dTaskContext`**, the
worker's own struct, which is already `&mut` in every reconstruction signature
(`rav1d_decode_tile_sbrow`, `rav1d_recon_b_intra`, `rav1d_backup_ipred_edge`).
So `ReconPlanes::dst()` hands out a genuine `&mut [u8]`, exclusion is a
**borrowck fact**, and no record is taken at run time. `#![forbid(unsafe_code)]`
is untouched and was proved actively (§6).

The kernels take `&mut ReconDst<'_>`, a two-arm enum: `Pic` for the shared
picture (unchanged, tracked) and `Own` for the band. Rust reborrows `&mut T`
implicitly in argument position, so passing it down a dispatch chain costs no
syntax, and the two backings share ONE code path — the shared-picture arm is
byte-identical to today at every one of the 24 md5 cells in §5.

### Geometry: column-compact, one superblock row, its own stride

#474 bought a three-line seam by giving each tile buffer the picture's byte
length AND stride, so every frame-coordinate offset indexed the same pixel. The
cost was `tile_columns × plane` residency: **+96.3 MB at 8bpc, +191.0 MB at
10bpc**, measured, and #481 called that unshippable.

A kernel signature taking `(&mut [u8], base, stride)` already carries the origin
and stride a compact buffer needs, so the translation is free once the signature
is changing anyway. `recon.rs` therefore asks for a destination by **plane pixel
`(row, col)`** instead of a flat frame offset — #473's 22-site coordinate cost,
paid once, in the same change rather than twice.

The band is `max_tile_width × (sb_step * 4)` per plane, per WORKER, with its own
64-byte-aligned stride. At 4K 4:4:4 with 8 workers that is
`8 × 960 × 128 × 3 ≈ 2.9 MB` against #474's 100 MB.

**Why one superblock row is enough, and how it is enforced.** Reconstruction
never reads a picture row above the current superblock row: `src/recon.rs`
gates the top edge on `t.b.y & f.sb_step - 1 == 0` and, at a superblock-row
boundary, sources it from `f.ipred_edge` rather than the plane
(`rav1d_prepare_intra_edges`'s `prefilter_toplevel_sb_edge`). Left, top-right
and bottom-left reads are clamped to the TILE
(`ts.tiling.col_start/col_end/row_end`) — AV1 tile independence. This is
load-bearing, so `ReconBand::at` subtracts the band origin and asserts the row
is in range: an access above the band is a **panic**, never a silent wrong
pixel. Across 766 corpus vectors × {t=1, t=8} the assert never fired.

## 2. Where the registrations went — census, by name, `lost=0`

`--features probe-sites`, `v4k_8tile` 8bpc, 3 iterations, registrations per
frame. Both arms are the SAME BINARY; `RAV1D_OWNED_RECON=0` disarms the band, so
an inter-arm delta cannot be a codegen artefact.

| arm | t=1 | t=8 | distinct sites |
|---|---|---|---|
| band off (= `main`) | 7,924,706 | 22,700,725 | 54 |
| **band on** | **6,005,602** | **11,401,399** | 49 / 48 |

The off column reproduces #455's and #481's committed census for `base` **bit
for bit** (7,924,706 / 22,700,725), which is the instrument's control.

**The on column is #481's unsound ceiling arm, exactly: 6,005,602 and
11,401,399.** #481 obtained those by composing #474 with `tracker: None` on the
private planes. This reaches the same two numbers with borrowck, from `main`,
without #474 — because the same change that removes the tracker also removes the
per-row split that took `base` from 7.9 M to 22.7 M at t=8.

Removed: **1,919,104/frame at t=1 and 11,299,326/frame at t=8 (49.8%)**.

## 3. Memory — the blocker, closed

Peak RSS, `/usr/bin/time -l`, 20 frames, one decoder.

| arm | 8bpc t=1 | 8bpc t=8 | 10bpc t=1 | 10bpc t=8 |
|---|---|---|---|---|
| `main` | 99.4 MB | 106.2 MB | 149.1 MB | 157.2 MB |
| **band on** | **99.6** | **107.8** | **149.5** | **160.4** |
| #474 / #481 (measured there) | 195.7 | 202.5 | 340.1 | 348.3 |

**+1.6 MB at 8bpc t=8 and +3.2 MB at 10bpc t=8**, against #474's +96.3 and
+191.0 — 59× and 60× smaller, and the t=1 column is +0.25 / +0.44 MB.
The arithmetic: 8 workers × 960 columns × 128 rows × 3 planes ≈ 2.9 MB at
8bpc, and the band is allocated once and reused across superblock rows and
across frames.

## 4. Wall clock — the sound arm lands ON #481's unsound ceiling

n = 9 complete rounds, 288 rows, **`foreign_max = 0` on every one**, idle Apple
M4 Pro (12 cores, 8P+4E), macOS 26.5.2. `scripts/perf/verify_gap.sh`: two-point
wall fit at 2 and 20 frames, rotating arm order, dav1d 1.5.4 `--framedelay 1` in
the SAME interleaved sweep, strict idle gate, no `nice` on any timed run, no
`-C target-cpu=native`. One cell-round discarded and re-run. `head` and
`headoff` are the SAME BINARY (`RAV1D_OWNED_RECON=0`).

ms/frame, median [min..max]:

| | t=1 | t=2 | t=4 | t=8 |
|---|---|---|---|---|
| `base` 8bpc | 319.4 [317.9..322.8] | 197.1 [196.3..199.4] | 106.8 [106.5..108.3] | 67.8 [64.4..71.0] |
| **`head`** 8bpc | **313.7** [312.7..325.3] | **166.7** [165.7..167.8] | **86.8** [86.1..87.6] | **53.4** [49.3..54.9] |
| `headoff` 8bpc | 323.3 | 201.3 | 108.8 | 70.1 |
| dav1d 8bpc | 246.4 | 125.2 | 65.7 | 36.2 |

Ratio to dav1d `--framedelay 1`:

| | t=1 | t=2 | t=4 | t=8 |
|---|---|---|---|---|
| `base` 8bpc | 1.296 | 1.574 | 1.627 | 1.873 |
| **`head`** 8bpc | **1.273** | **1.332** | **1.321** | **1.474** |
| `headoff` 8bpc | 1.312 | 1.609 | 1.657 | 1.936 |
| **#481's unsound ceiling** | *1.276* | *1.333* | *1.313* | *1.475* |
| whole-tracker ceiling (#467) | *1.160* | *1.264* | *1.262* | *1.345* |
| `base` 10bpc | 1.442 | 1.621 | 1.658 | 1.915 |
| **`head`** 10bpc | **1.395** | **1.456** | **1.471** | **1.591** |
| #481's ceiling, 10bpc | *1.393* | *1.452* | *1.467* | *1.610* |

**The sound arm reproduces #481's `tracker: None` ceiling at all eight cells**,
to within 0.008 at 8bpc and 0.019 at 10bpc, and is marginally *better* at 10bpc
t=8. 8bpc t=1 at **1.273** is under the ~1.30× bar — the only arm in this
campaign that is.

Paired within-round ratios (the arms in one round saw the same machine state;
exact two-sided sign test):

| `head` / `base` | median | band | wins | p |
|---|---|---|---|---|
| 8bpc t=1 | 0.9823 | [0.9699..1.0218] | 8/9 | 0.039 |
| 8bpc t=2 | **0.8424** | [0.8376..0.8492] | 9/9 | 0.004 |
| 8bpc t=4 | **0.8134** | [0.8000..0.8180] | 9/9 | 0.004 |
| 8bpc t=8 | **0.7845** | [0.7551..0.7932] | 9/9 | 0.004 |
| 10bpc t=1/2/4/8 | 0.9710 / 0.9004 / 0.8822 / 0.8279 | | 9/9 each | 0.004 |

Arm-band disjointness (per `gap_bands.py`): **disjoint at 7 of 8 cells; 8bpc
t=1 OVERLAPS** — its 1.8% is the one number here that is not separated, and it
is reported as such rather than rounded into the headline.

Scaling t=1 → t=8: base 4.71× → **head 5.88×** at 8bpc (dav1d 6.80×);
4.97× → **5.79×** at 10bpc (dav1d 6.60×).

### The honest negative: the seam costs the SHARED path 1.2–3.0%

`headoff` is the same binary with the band disarmed, i.e. every frame the
change declines — which today is **every inter frame**. It is consistently
SLOWER than `base`:

| `headoff` / `base` | t=1 | t=2 | t=4 | t=8 |
|---|---|---|---|---|
| 8bpc | 1.0115 | 1.0180 | 1.0193 | 1.0216 |
| 10bpc | 1.0198 | 1.0266 | 1.0301 | 1.0245 |

9/9 rounds slower at 7 of 8 cells (p = 0.004); 8bpc t=8 is 7/9 (p = 0.18). So
the `ReconDst` enum's branch on the tracked path is **not free** — it is worth
1–3%, and until the inter path is converted a mixed-GOP stream pays it on every
inter frame while collecting the win only on key frames. That is the strongest
argument for treating this as the first half of a conversion rather than a
shippable end state.

### The first sweep was invalid, and how it was caught

A full n=9 / 288-row sweep was run and **thrown away**. It reported the head
arm at 9.13× dav1d at 8bpc t=8 against base's 1.89× — a 5× regression with no
thread scaling at all — and, tellingly, the band-DISARMED arm was *worse* than
the band-armed one, which no memory- or seam-cost story explains.

Cause: the staged `bench_head` binary had been overwritten by a
`cargo build --release --features probe-sites --example bench_ab_decode`
earlier in the session — cargo writes both feature sets to the same
`target/release/examples/bench_ab_decode` path. The `base` arm was a clean
build. So the A/B compared an instrumented binary against an uninstrumented
one.

A `sample` profile settled it in one command: `site_probe::record` was **51.4%
of leaf samples** in the head arm. `/usr/bin/time -l` had already said the same
thing more cheaply — head burned 45.5 s of user CPU against base's 10.0 s for
the same work, i.e. the workers were running and doing 4.5× the work, not
serializing.

Two rules out of it: **`strings <binary> | grep site_probe` before staging any
arm**, and **when an A/B shows a regression larger than the mechanism can
explain, profile before believing it** — the arm's own composition is the first
suspect, not the code under test.

## 5. Correctness gates

* **Corpus, set-diffed BY NAME with the md5 as the value**, against
  `benchmarks/aarch64_md5_fixes_2026-08-07_final.tsv.zst`:
  * `--threads 1`: **766 PASS / 0 mismatch / 0 error**, 768 keys,
    0 only-in-baseline / 0 only-in-head / **0 differing**.
  * `--threads 8`: **753 PASS**, 755 keys after dropping the two film-grain
    groups (#479: `rav1d_apply_grain_row` bands rows per worker while
    `filmgrain_arm.rs` takes a whole-plane mutable guard, which aborts 13 of 768
    vectors at `threads > 1` on unmodified `main`), **0 differing**.
    `md5_inventory --skip-group` was added for this and prints
    `skipped_groups=` on the TOTAL line so an empty filter cannot pass as a full
    run. Groups skipped: `8-bit/film_grain`, `10-bit/film_grain`.
  * Band-on vs band-off at both thread counts: 0 differing, both directions.
* **Frame md5 across `main` / band-on / band-off**, t=1/2/4/8, 8bpc and 10bpc:
  **24 of 24 identical** (`a00c11f454328023c58af14d55544cff` /
  `4f218411bc6ee4cc9c630fe827337fa2`).
* `mt_stress` 1/2/4/8/16 × 5: pass.
  `multi_decoder_pressure` 12 procs × 3 iters × {1,2,4,8,16}: **PASS**, every
  md5 matching the serial reference.
* **Liveness is asserted, not assumed.** An `assert!(false)` planted in
  `arm_sbrow` after `ReconBand::arm` fires on the first superblock row of
  `v4k_8tile` at t=1 — the band is on the path, before any timing is quoted.
  (Restored; the registration census in §2 is the standing proof.)

## 6. `#![forbid(unsafe_code)]`, proved actively and NON-VACUOUSLY

#481 recorded that its first proof was vacuous: an `unsafe` planted in
`src/tile_recon.rs` built cleanly because that whole module is
`#[cfg(feature = "tile-owned-recon")]`.

`src/owned_recon.rs` carries **no `cfg` on its `mod` declaration** — it is
compiled on the default feature set, which is what makes a plant there
meaningful. Two plants, on the plain default build:

| file | result |
|---|---|
| `src/owned_recon.rs:75` (the new module) | `error: usage of an 'unsafe' block`, `lib.rs:13` |
| `include/dav1d/picture.rs:384` | `error: usage of an 'unsafe' block`, `lib.rs:13` |

Both restored **byte-exact**, verified by sha256 AND `git diff --exit-code`:
`owned_recon.rs` `2372afdf…81732`, `picture.rs` `aa92199b…62875d`.

There is no compile-time feature to re-prove under: arming is a runtime switch
(`RAV1D_OWNED_RECON`) precisely so both A/B arms are one binary. The band's code
is in the default build either way.

### And the exclusion property itself, the same way

`forbid(unsafe_code)` says no `unsafe` was written. It does not say the design
*needs* none. That is a separate claim — that two live regions over one band
cannot exist — and it is also a compile-time fact, so it was planted too:

```rust
let mut a = dst_at(&mut b, 16, 32);
let mut c = dst_at(&mut b, 17, 32);
a.set::<BitDepth8>(1);
c.set::<BitDepth8>(2);
```

```
error[E0499]: cannot borrow `b` as mutable more than once at a time
   --> src/owned_recon.rs:663:28
662 |         let mut a = dst_at(&mut b, 16, 32);
    |                            ------ first mutable borrow occurs here
663 |         let mut c = dst_at(&mut b, 17, 32);
    |                            ^^^^^^ second mutable borrow occurs here
```

Restored byte-exact (sha256 verified). There is deliberately **no run-time test
asserting this** — a test that passes because a hardcoded string matches would
prove nothing.

## 7. Standing hazards, replanted

`--features __probe_wide` on `rav1d-disjoint-mut` (untouched by this branch —
`git diff cebb97f -- crates/` is empty).

| plant | `wide_exclusion` |
|---|---|
| baseline | ok (0.06 s) |
| the `4af62ae` in-lock `state` re-read deleted from **`add_contended` alone** (the CONTENDED path; the `try_lock` half is not touched) | **FAILED** |
| `active()` cut to one shard | **FAILED** |
| after both restores | ok |

Both restored byte-exact (`sha256 d4e03d4a…423176d`, `git diff --exit-code`
clean).

**Gotcha worth recording:** restoring a file with `shutil.move` preserves its
mtime, and cargo then reuses the *mutated* build artefact — the first re-run
after restore reported FAILED on a byte-identical tree. `touch` the file after
any restore that does not bump mtime, and re-run before believing a green (or a
red).

## 7b. Miri, and the compile matrix

`crates/rav1d-disjoint-mut` is **byte-identical to `cebb97f`** on this branch
(`git diff cebb97f -- crates/` is empty), so neither memory model can regress
from a change that is not there. Both were re-run anyway, each target in
isolation via `--no-fail-fast`, `cargo +nightly miri test -p
rav1d-disjoint-mut`:

| model | result |
|---|---|
| Stacked Borrows (`MIRIFLAGS=""`) | **61 passed, 0 failed**, no UB (9 targets; `soundness` 992.7 s) |
| Tree Borrows (`-Zmiri-tree-borrows`) | **61 passed, 0 failed**, no UB (9 targets; `soundness` 993.1 s) |

**The `asm` arms did not compile, and nothing on this box could have told me.**
`--features asm` cannot be built on macOS at all: `main` itself fails c-ffi's
`EAGAIN` const-assert there (macOS 35 vs Linux 11), and the aarch64 asm leg
wants a cross-assembler that is not installed. Cross-checking against
`--target x86_64-unknown-linux-gnu` found three real breakages — every
`*_c_erased` FFI shim handing a recovered `PicOffset` to a fallback that now
takes `&mut ReconDst`, `pal_pred::Fn::call`'s asm arm still calling
`as_mut_ptr` on the enum, and two `Strided` imports the asm arms need. Fixed
in `9492cfc`. Matrix now green:

| configuration | |
|---|---|
| default, aarch64-apple-darwin | OK (and every runtime gate above) |
| default, x86_64-apple-darwin | OK (compile only) |
| default, wasm32-unknown-unknown | OK (compile only) |
| default, aarch64-unknown-linux-gnu | OK (compile only) |
| `--features c-ffi`, aarch64-unknown-linux-gnu | OK (compile only) |
| `--features asm`, x86_64-unknown-linux-gnu | OK (compile only) |

The full workspace suite (`cargo test --release --workspace`) is green: 30
targets, 0 failures, including `decode_md5_verify` (14) and
`decode_permutations` (19).

## 8. Mechanical scope of the conversion

Function signatures moved from `WithOffset<&Rav1dPictureDataComponent>` to
`&mut ReconDst<'_>` (writes) or `&ReconSrc<'_>` (reads):

| file | signatures |
|---|---|
| `src/ipred.rs` | 24 |
| `src/itx.rs` | 5 |
| `src/safe_simd/ipred.rs` (x86-64) | 3 |
| `src/safe_simd/ipred_arm.rs` | 2 |
| `src/safe_simd/itx/part10_dispatch.rs` (x86-64) | 2 |
| `include/common/dump.rs` | 2 |
| `src/safe_simd/itx_arm.rs` | 1 |
| `src/safe_simd/itx_wasm.rs` | 1 |
| `src/ipred_prepare.rs` | 1 |
| **converted total** | **39** |

Plus the seam and 15 destination constructions in `src/recon.rs`.

**228 `PicOffset` parameters remain**, and naming them is the honest measure of
what is left: `src/mc.rs` 59, `src/looprestoration.rs` 25,
`src/safe_simd/cdef.rs` 18, `src/safe_simd/mc.rs` 17,
`src/safe_simd/mc_arm.rs` 17, `src/safe_simd/cdef_arm.rs` 16,
`src/safe_simd/looprestoration.rs` 15, `looprestoration_arm.rs` 10,
`cdef_wasm.rs` 9, `src/loopfilter.rs` 9, `src/lf_apply.rs` 7, `src/cdef.rs` 7,
`src/filmgrain.rs` 5, `src/recon.rs` 4 (the inter path), `mc_wasm.rs` 3,
`src/lr_apply.rs` 3, and 3 more. Inter is ~99 of those; the filter chain is
~120.

Two kernel bodies needed more than a type change:

* **`ipred_filter_rust`** held a live READ guard on `dst` (`top`) across the
  write loop, refreshed once per row pair. An owned band cannot hand out `&` and
  `&mut` to itself simultaneously — that is the whole point — so the row is
  copied into a 64-pixel local first. Same values, same output; the copy is of a
  row this kernel wrote one iteration earlier.
* **`ipred_z3_rust`** took `dst.pixel_stride()` inside a `dst + …` expression;
  the stride is hoisted.

The `asm` dispatch arms recover the tracked `PicOffset` via `ReconDst::as_pic()`
and `expect`. That is not a latent panic: `asm` implies `c-ffi`, and
`frame_setup` refuses to arm under `c-ffi`.

---

# Round 2 of 2026-08-10 — the seam's tax, and what the filter chain is actually made of

Branch `perf/filter-band`, based on `perf/mut-recon-kernels` @ `3b7242c`.
Same box: idle-or-tagged Apple M4 Pro (12 cores, 8P+4E), macOS 26.5.2.

Two problems were left open above. §9 closes the first. §11 answers the second
with a measurement that says **do not build the thing that was proposed**.

## 9. The seam's tax on the TRACKED path — closed

§4's honest negative: `headoff`, the same binary with the band disarmed — every
frame the conversion declines, which today is every inter frame — ran at
1.0115 / 1.0180 / 1.0193 / 1.0216 of `base` at 8bpc t=1/2/4/8, 9/9 rounds
slower at 7 of 8 cells. A net loss for video in exchange for a key-frame-only
win.

**It was not the enum's branch. It was that the seam's accessors were being
CALLED.** A `sample` self-time profile at t=1 with the band disarmed named it
in one line: `owned_recon::ReconSrc::slice::<BitDepth8>` at **0.460% of leaf
samples as an out-of-line symbol**, against 0.000% in `base`, where the same
read is `PicOffset::slice` inlined into `rav1d_prepare_intra_edges`. `#[inline]`
is a hint, and these wrappers lost the coin toss: each returns an enum WRAPPING
a borrow guard (`Px` / `PxMut`), so the callee looks big to the inliner even
though the `Pic` arm is a passthrough and the `Own` arm is dead in that build.

`#[inline(always)]` on the 37 seam accessors (`2a7ff51`). No signature, arm or
arithmetic changed. Paired user CPU over 20 frames of `v4k_8tile` 8bpc at t=1,
`/usr/bin/time -l`, arms interleaved base,head within each round:

| | base | headoff | headoff/base | slower |
|---|---|---|---|---|
| before | 6.62 s | 6.76 s | **1.0212** | 5/5 |
| after | 6.71 s | 6.75 s | **1.0060** | 6/7 |

and re-profiling finds **no `owned_recon` symbol in the head arm's self-time
table at all** — the seam is gone from the emitted code, not merely cheap. Code
size went DOWN (2,887,376 bytes against the pre-inline head's 2,888,816; base
is 2,868,624): forcing thin wrappers to inline does not bloat, leaving them out
of line duplicates a call frame. The armed path is unaffected — t=8 user CPU
7.23 s against base's 9.90 s.

The wall-clock gate for this is §12.

### The other half, measured and REVERTED

The second suspected mechanism was addressing. §8's conversion made
`rav1d_recon_b_intra` re-derive its destination per transform block —
`planes.dst(0, 4*t.b.y, 4*t.b.x)`, a multiply by the destination's pixel
stride — where `main` advanced it with `y_dst += 4 * t_dim.w`, an add. Restoring
the incremental form is backing-agnostic (`ReconDst::at` is a pixel delta on
whichever stride the destination carries), so it was written for both the luma
and the chroma loop and measured:

| | base | headoff | headoff/base |
|---|---|---|---|
| `#[inline(always)]` only | 6.71 s | 6.75 s | **1.0060** |
| + row-origin hoist | 6.71 s | 6.85 s | **1.021** (n=7) |

**A regression, back to where the tax started.** Reverted. The hoist keeps a
live 40-byte `ReconDst` across the whole block body — including the call to
`decode_coefs`, which is 62% of leaf samples — and that costs more than the
multiply it saves. Re-confirmed at n=7 after reverting: 1.0060.

## 10. What is STILL not done, after this round

* **The filter chain is still not converted**, and §11 argues against converting
  it the way this round was asked to. Nothing in `loopfilter`, `cdef`,
  `looprestoration`, `lf_apply`, `lr_apply` or `filmgrain` changed except one
  default-off, feature-gated measurement branch.
* **Inter frames are still not converted.** Unchanged from §0: 228 `PicOffset`
  parameters remain, `src/mc.rs` holds 59, and a frame containing inter
  reconstruction still declines wholesale.
* **`src/ctx.rs:99` is still untouched**, 2,534,988 registrations/frame at
  `v4k_8tile` t=8 — now 22.2% of the population and, again, not a picture
  buffer.
* **`ipred_filter_rust`'s per-row-pair copy** (§8) is still there. It is extra
  work on the TRACKED path too, and it was NOT removed, because
  `ipred_filter_rust` does not appear in a `sample` self-time profile of this
  vector at all — there is no instrument here that could show a change. It is
  removable (the row it copies back is the row the kernel wrote one iteration
  earlier, so it can be stashed as it is written), and that is a strictly
  smaller amount of work than either arm does today; it is left undone rather
  than done blind.
* **#479 did not fall out of this work.** `looprestoration.rs:212,258` and
  `loopfilter.rs:140,182` — the whole-plane-guard shape named in §0 — were not
  audited or changed. The two film-grain groups are still skipped at
  `--threads 8`.
* **The seam tax is reduced, not gone.** §12b: 1.0081 at 8bpc t=1
  (p=0.007), 1.0126 / 1.0111 at 10bpc t=1 / t=2 (p=0.007 / 0.000). The brief
  for this round set parity as the bar for shippability; it is met at 8bpc
  t=2/4/8 and NOT met at the other three cells retested.
* **No idle box, at any point.** All 288 rows of §12's sweep, all 120 of
  §12b's, and every user-CPU table here were taken with at least one foreign
  95-110% CPU process. A strict-idle-gate attempt returned zero rows in seven
  minutes. Ratios only; the absolute ms/frame in §12 are ~10% inflated.
* **`RAV1D_LF_DOUBLE`'s number is a marginal cost, not the prize** (§11f), and
  the difference between the two was not measured — only bounded.
* **The band's copy cost is not measured.** §11f sizes the budget; nothing here
  spends against it.
* Not measured, unchanged: x86_64 and wasm32 (compile-checked only), `asm` /
  `c-ffi` (compile-checked only), `unchecked`, t=16, any vector below 4K, and
  **any vector with loop restoration live** — the structural blindness of
  #455 item 4 is still there, and §11's census inherits it (see the caveat at
  the end of §11).

## 11. The filter chain: the census, the halo question, and a large negative

### 11a. The census, measured here

`--features probe-sites`, `v4k_8tile` 8bpc, 3 iterations, registrations per
frame, `lost=0`. The band-on t=8 total reproduces §2's 11,401,399 exactly,
which is the instrument's control.

| | t=1 | t=8 |
|---|---|---|
| whole decoder, band ON | 6,005,602 | 11,401,399 |
| **filter chain** | **995,665** (16.6%) | **6,391,462** (56.1%) |
| `src/loopfilter.rs:566` (`LfBlock::fill`) | on the hull path | **3,835,042** (33.6% of everything) |
| `src/safe_simd/cdef_arm.rs` (6 sites) | | 1,863,648 |
| `src/cdef_apply.rs:104,121` | | 669,376 |
| 8 remaining filter sites (`lf_apply`, `cdef_apply` line buffers) | | 5,544 |
| `src/loopfilter.rs:739` (`LfBlock::close`, the WRITE side) | | 17,852 |
| `src/ctx.rs:99` (not the filter chain, not a picture) | | 2,534,988 |

Two things in that table were not in the brief and change the problem.

**First: the filter chain is 6.4x bigger at t=8 than at t=1.** 5,395,797 of its
6,391,462 registrations do not exist at t=1. They are not what the filter chain
reads; they are one policy branch. `LfBlock::fill` (`src/loopfilter.rs:551`)
takes ONE guard over the strided hull when `tile_threading_active()` is false
and `h` per-row guards when it is true, because a hull also reserves the
inter-row gaps and those gaps are other columns of the same picture rows, which
a concurrent tile worker may legitimately be writing. That is the same per-row
split #482 removed from reconstruction — recon's 7.9M -> 22.7M at t=8 was the
same mechanism.

**Second: the write side is already tiny.** `LfBlock::close` writes back only
the rows that changed and only the changed span within them
(`src/loopfilter.rs:722-741`), so the entire mutable population of the biggest
filter kernel is 17,852 per frame. Anything aimed at the loop filter is aimed
at an IMMUTABLE population.

### 11b. The row-band-with-halo: expressible, but not exclusive

The brief's proposal was a row band with a halo — a filter worker owns sbrow
N's rows plus the tap rows above and below, with the halo read-only for one of
the two neighbours. Three facts, with file:line:

1. **Filter stages for different superblock rows DO overlap in time.**
   `src/thread_task.rs:1030-1043`: when a filter task for `sby` is *selected*,
   the replacement task for `sby+1` is inserted immediately, before the
   selected one runs. So a second worker can be in `DeblockCols(N+1)` while the
   first is still in `Cdef(N)` / `LoopRestoration(N)`. Only `DeblockRows` is
   serialised against itself, by `ensure_progress` on
   `fc.frame_thread_progress.deblock` (`src/thread_task.rs:538-559`,
   `:1351-1362`).

2. **The pipeline is nevertheless built so their WRITE sets are row-disjoint,
   and the margin is one row.** `rav1d_filter_sbrow_cdef`
   (`src/recon.rs:3780-3789`) filters `[sby*sbsz - 2, sby*sbsz)` and then stops
   `2 * ((sby+1) < sbh)` block rows short of the end of its own superblock row.
   `rav1d_loopfilter_sbrow_rows(N)` writes sbrow N plus the tail of sbrow N-1,
   reaching at most `lf_reach` = 7 rows above the boundary
   (`src/loopfilter.rs:455-467`, and the argument recorded at
   `src/thread_task.rs:596-604`). So CDEF(N-1)'s last written luma row is
   `N*sbsz*4 - 9` and deblock(N)'s topmost row is `N*sbsz*4 - 7`. Disjoint —
   by exactly one row.

3. **A shared halo is not expressible in safe Rust, and a copied one is.**
   `&mut` exclusion is a static fact, not a temporal one. Two workers whose
   bands overlap in a halo need the halo handed from one to the other at run
   time, and a run-time ownership handoff is a borrow tracker wearing a
   different hat — which is the thing being removed. What IS expressible is the
   shape `stitch_sbrow` already uses (`src/owned_recon.rs:995-1013`): copy the
   halo rows IN under row guards, filter entirely inside an owned
   `Vec<Chunk>`, copy the owned rows back OUT under row guards. That is safe,
   needs no `unsafe`, and turns millions of kernel-level registrations into a
   few hundred row-level ones per superblock row.

So the answer to "is it expressible" is: **yes as copy-in/copy-out, no as a
shared halo.** Which would be the design — except for §10c.

One more thing before anyone builds it: **the copy is not free, and it is not
measured here.** A 4K 4:4:4 superblock-row band is roughly
`135 rows x 3840 B x 3 planes` in and the same out — about 3.1 MB per
superblock row, ~53 MB per frame at 17 superblock rows. Nothing in this round
measured that, and no conversion should be started without measuring it.

### 11c. The measurement that says stop: removing 3.46 M registrations made it 2.65x SLOWER

`--features __probe_lf_hull` + `RAV1D_LF_HULL=1` makes `fill` take the hull
under tile threading as well. This arm is **sound in the detection sense** and
that is why it can be run at all: widening an IMMUTABLE reservation is a
superset of the narrow rows, so it cannot MISS an overlap they would have
caught. Its only new failure mode is a false positive, which is a loud
`overlapping DisjointMut` panic, never a wrong pixel. (Contrast #481's
`tracker: None`, which could only be run because nothing checked it.)

It removes **3,463,025 registrations per frame** at t=8 — 11,401,399 ->
7,938,374, 30.4% of the whole post-#482 population, and 54% of the filter
chain's.

User CPU, `/usr/bin/time -l`, `v4k_8tile` 8bpc, 20 frames, median of 5, arms
interleaved within every round, band armed in all three columns:

| | base | band on | band on + hull | hull / band-on |
|---|---|---|---|---|
| t=2 | 8.31 s | 6.91 s | 7.30 s | **1.053** |
| t=4 | 8.80 s | 7.02 s | 7.44 s | **1.052** |
| t=8 | 9.93 s | 7.40 s | 19.63 s | **2.65** |

and with the recon band *also* disarmed — i.e. what an inter frame would see —
t=8 goes to 30.94 s, **3.12x base**.

The mechanism is extent, not count. The hull of an `LfBlock` read is 14-16
picture rows: at a 3840-byte stride that is 50-60 KB, far past the sharded
tracker's block size, so every one of them takes the wide path and is checked
against every shard. Fourteen 8-byte registrations are cheaper than one 50 KB
one — and at t=8, where the wide path also contends with eight tile workers,
they are cheaper by 2.65x.

**This is the campaign's "count is not cost" lesson for the third time, and the
first time in the direction where reducing the count is actively harmful.**
#455 recorded `block_mut` halving the guard count for nothing; #481 recorded
recon's removals costing 9.4/5.4/2.8 ns each against a 1.00 ns average. This
one says a filter-chain registration is worth *less* than nothing if you pay
extent for it.

The switch is kept, behind `#[cfg(feature = "__probe_lf_hull")]`, as the
reproduction. The default build compiles `lf_hull_reads()` to `false` and the
branch folds out of `fill`. Evidence that it folds rather than merely being
cheap: the default binary's SIZE is unchanged at 2,887,376 bytes across the
build before either probe feature existed, the build after the hull switch, and
the build after the doubling switch — three edits to the decoder's hottest
guard site, zero bytes of code. (Size, not a byte-for-byte compare: the
binaries carry debug info that differs per build.)

### 11d. The clean ablation: the count IS worth something — you just cannot buy it with extent

`RAV1D_LF_PERROW=1` forces the per-row path at t=1, where no second thread
exists, so it is unconditionally sound and completely uncontended. It ADDS the
same population the hull REMOVED, measured from the other side:

| t=1, band on | registrations/frame |
|---|---|
| default (hull at t=1) | 6,005,602 |
| `RAV1D_LF_PERROW=1` | 9,468,627 |
| difference | **+3,463,025** |

That is **the same 3,463,025** the hull removed at t=8 (11,401,399 ->
7,938,374), to the registration — the two arms are pricing one population from
opposite directions, which is the instrument's control.

User CPU, `v4k_8tile` 8bpc t=1, 20 frames, paired within round, n=7:

| | median | band | slower | p |
|---|---|---|---|---|
| hull (the t=1 default) | 6.790 s | [6.75..7.14] | | |
| per-row (forced) | 6.990 s | [6.88..7.21] | | |
| **per-row / hull** | **1.0264** | [1.0098..1.0619] | 7/7 | 0.016 |

**So the count is not free: 3.46 M uncontended narrow registrations cost 2.64%
of a t=1 frame — 0.20 s over 20 frames, ~10 ms/frame, ~2.9 ns each.** That
lands inside the 2.8-9.4 ns/registration range #481 measured for recon's
population, and it contradicts a lazy reading of §11c.

Put the two together and the finding is sharper than "count is not cost":

* Buying the count with **EXTENT** (one wide guard instead of `h` narrow ones)
  is REFUTED: +5.2% at t=2 and t=4, +165% at t=8.
* The count itself is worth about **2.6% at t=1 for this one site**, and at t=8
  those same registrations are additionally contended — which this round did
  NOT measure, because no arm removes them without adding extent. **That is the
  gap the next attempt has to close, and a band is the only instrument that
  can.**

### 11f. The number that decides it: a filter-chain registration at t=8 is worth 3.61 ms/frame of WALL

Neither §11c nor §11d prices the population **at t=8, contended**, which is the
only cell where it matters. `RAV1D_LF_HULL` cannot, because it substitutes a
worse extent. `RAV1D_LF_PERROW` cannot, because at t=1 there is no contention.

`RAV1D_LF_DOUBLE=1` (same feature) takes **each** per-row read guard TWICE —
same bytes, same extent, same output, nothing changed but the count. Sound by
construction: the extra reservation is IMMUTABLE and covers exactly the bytes
the next one covers, and two immutable reservations never conflict, so it
cannot invent an overlap the single guard would not already have found.

Census, `v4k_8tile` 8bpc t=8, band on: 11,401,399 -> 15,236,441, i.e.
**+3,835,042 — exactly `LfBlock::fill`'s population**, added back on top of
itself.

**Control first.** At t=1 the hull path is taken and the doubling code is
unreachable. It measures null: 1.0058 [0.8888..1.0594], 4/7 slower, p=1.000.
The probe fires only where it is supposed to.

| t=8, `v4k_8tile` 8bpc | single | double | double/single | n | p |
|---|---|---|---|---|---|
| user CPU, 20 frames | 9.740 s | 10.050 s | **1.0340** [0.9764..1.0539], 14/15 | 15 | **0.0010** |
| **wall, two-point fit at 2 and 20 frames** | **66.28 ms/frame** | **69.89** | **1.0536** [1.0316..1.0976], 11/11 | 11 | **0.0010** |

**3,835,042 registrations = 3.61 ms/frame of wall and 15.5 ms/frame of CPU at
t=8, i.e. 4.04 ns of CPU each** — inside the 2.8-9.4 ns/registration range #481
measured for recon's population, and 1.4x the 2.9 ns §11d measured for the same
registrations uncontended at t=1. Contention is worth about 40% on top.

**Read it as an UPPER bound on the prize, not the prize.** This is the MARGINAL
cost of adding 3.84 M on top of 11.4 M. Tracker cost is demonstrably sublinear
in population going the other way — #467/#482 show 22.7 M -> 11.4 M buying
0.399 of dav1d ratio at t=8 while the remaining 11.4 M -> 0 is worth only 0.129
— so removing the original 3.84 M will save LESS than adding a second copy
costs.

With that caveat, the sizing for a next attempt: **the distance from #482's
1.474 to the whole-tracker ceiling's 1.345 is 8.8% of `head`'s wall, and this
one site's READ population is worth up to ~5.4% of it.** An owned filter band
that removes those registrations therefore has room for a copy costing under
roughly 3.6 ms/frame at 4K t=8, and the copy is ~3.1 MB in and out per
superblock row. That is the arithmetic to check FIRST — before writing any of
the conversion.

### 11e. What a next attempt has to answer first

Whatever that says, two things are now fixed points for anyone converting the
filter chain:

* The prize is smaller than the share suggests. 56.1% of registrations is
  **not** 56.1% of anything else, and 5.4 M of the 6.4 M exist only because of
  a policy branch that is there to avoid false positives, not because the
  filter chain reads a lot.
* The loop filter's read population is IMMUTABLE and its write population is
  already 0.3% of it. A conversion that gives the loop filter an owned band is
  buying out immutable registrations, which are the cheap kind.


## 12. Wall clock — and the sweep was NOT idle

**Every one of the 288 rows in this sweep is load-tagged.** Another agent held a
95-110% CPU benchmark on this box for the whole run; `foreign_max` is 1 on 128
rows, 2 on 152, 3 on 4 and 6 on 4. `ALLOW_LOAD=1` was used deliberately rather
than letting the strict gate discard forever (it discarded 3 cells in a row
before the switch). Per the campaign's own rule, **the paired ratios below are
believable and the absolute ms/frame are inflated** — `base` at 8bpc t=4 reads
118.7 ms here against §4's idle 106.8, so treat the absolutes as ~10% high and
do NOT compare them to §4's table.

n = 9 complete rounds, 288 rows, `scripts/perf/verify_gap_arms.sh` (verify_gap
with per-arm env), two-point wall fit at 2 and 20 frames, rotating arm order,
dav1d 1.5.4 `--framedelay 1` in the SAME interleaved sweep, no `nice` on any
timed run, no `-C target-cpu=native`. `head` and `headoff` are one binary
(`RAV1D_OWNED_RECON`).

ms/frame, median [min..max] — **loaded, see above**:

| | t=1 | t=2 | t=4 | t=8 |
|---|---|---|---|---|
| `base` 8bpc | 321.1 [317.5..333.2] | 202.9 [200.8..214.2] | 118.7 [114.1..122.9] | 74.9 [71.7..80.2] |
| **`head`** 8bpc | **315.3** [313.6..323.9] | **167.1** [166.2..178.4] | **90.0** [86.0..93.4] | **57.9** [54.6..61.9] |
| `headoff` 8bpc | 324.8 [321.5..333.4] | 203.8 | 118.3 | 74.7 |
| dav1d 8bpc | 249.3 [248.9..264.3] | 127.2 | 66.4 | 39.2 |

Ratio to dav1d `--framedelay 1` (loaded):

| | t=1 | t=2 | t=4 | t=8 |
|---|---|---|---|---|
| `base` 8bpc | 1.29 | 1.60 | 1.79 | 1.91 |
| **`head`** 8bpc | **1.26** | **1.31** | **1.35** | **1.48** |
| `head` 10bpc | 1.40 | 1.45 | 1.45 | 1.58 |

which reproduces §4's shape (1.273 / 1.332 / 1.321 / 1.474 idle) closely enough
that the conversion's win is clearly intact, but **these are loaded numbers and
§4's idle table remains the citable one for the dav1d ratio.**

Paired within-round `head`/`base`, exact two-sided sign test — the win, and it
is unambiguous:

| | median | band | wins | p |
|---|---|---|---|---|
| 8bpc t=1 | 0.9790 | [0.9702..0.9920] | 9/9 | 0.004 |
| 8bpc t=2 | 0.8256 | [0.8186..0.8454] | 9/9 | 0.004 |
| 8bpc t=4 | 0.7571 | [0.7428..0.7679] | 9/9 | 0.004 |
| 8bpc t=8 | 0.7662 | [0.7573..0.7724] | 9/9 | 0.004 |
| 10bpc t=1/2/4/8 | 0.9707 / 0.8938 / 0.8050 / 0.8007 | | 9/9 each | 0.004 |

### The tax, before and after — REDUCED, NOT ELIMINATED

Paired within-round `headoff`/`base`, the number §9 exists to move. #482's
column is its own idle measurement; this column is loaded, which costs the sign
test power, so a non-significant cell here is weaker evidence than a
non-significant cell there.

| | #482 (idle) | this branch (loaded) | wins for headoff | p |
|---|---|---|---|---|
| 8bpc t=1 | 1.0115 | **1.0116** | 2/9 | 0.180 |
| 8bpc t=2 | 1.0180 | **1.0041** | 1/9 | 0.039 |
| 8bpc t=4 | 1.0193 | **1.0056** | 2/9 | 0.180 |
| 8bpc t=8 | 1.0216 | **0.9993** | 5/9 | 1.000 |
| 10bpc t=1 | 1.0198 | **1.0133** | 0/9 | 0.004 |
| 10bpc t=2 | 1.0266 | **1.0110** | 0/9 | 0.004 |
| 10bpc t=4 | 1.0301 | **1.0085** | 3/9 | 0.508 |
| 10bpc t=8 | 1.0245 | **1.0048** | 4/9 | 1.000 |

**Honest reading.** #482 was 9/9-slower at 7 of 8 cells. This is 9/9-slower at
2 of 8, both of them 10bpc t=1 and t=2, at 1.3% and 1.1%. The 8bpc t=2/4/8
cells are at parity (t=8's median is below 1.0). **8bpc t=1 did not move at
all** — 1.0116 against #482's 1.0115 — although it is no longer separated from
noise on this data.

That is a real improvement and it is NOT the parity the brief asked for. The
seam still costs a 10bpc inter frame ~1%, and the 8bpc t=1 cell is unexplained:
the low-noise user-CPU instrument put that same cell at 1.0060 after the fix
against 1.0212 before, on a quiet box, so wall and work disagree there and only
one of them can be right. **Do not merge #482 on the strength of this table
alone; re-run `headoff` against `base` on an idle box.**

## 13. Correctness gates, this round

* **Corpus, set-diffed BY NAME with the md5 as the value**, against
  `benchmarks/aarch64_md5_fixes_2026-08-07_final.tsv.zst`:
  * `--threads 1`: **766 PASS**, 768 keys, 0 only-in-baseline, 0 only-in-head,
    **0 differing**. `SETDIFF: CLEAN`.
  * `--threads 8`, `--skip-group 8-bit/film_grain --skip-group
    10-bit/film_grain` (#479), the same two groups dropped from BOTH sides:
    **753 PASS**, 755 keys, **0 differing**. `SETDIFF: CLEAN`. The 2 non-PASS
    keys are `8-bit/features annexb` and `section5`, `SKIP` in the baseline
    too. Inventories committed: `benchmarks/filter_band_md5_t{1,8}_2026-08-10.tsv.zst`.
* **Frame md5 across all four arms** — band {on, off} x hull {on, off} — at
  t=1 and t=8: 8 of 8 identical, `a00c11f454328023c58af14d55544cff`, matching
  `base` at both thread counts.
* `mt_stress` 1/2/4/8/16 x 5: **pass**.
  `multi_decoder_pressure.sh` 12 procs x 3 iters x {1,2,4,8,16}: **PASS**,
  every md5 matching the serial reference on all five vectors.
* `cargo test --lib`: 75 passed in the DEBUG profile and 75 in release, and the
  4 `owned_recon` tests pass in both — the profile-dependent-panic trap #482
  hit on its Coverage leg does not recur.
* **Clippy.** Both CI legs green on BOTH architectures, 0 warnings:
  `cargo clippy --no-default-features --features bitdepth_8,bitdepth_16 --
  -D warnings` on `aarch64-apple-darwin` and on `x86_64-apple-darwin`.
  The stricter `--all-targets` x86 run still fails, but every remaining error
  is one `cebb97f` also has, in a file this branch does not touch
  (`src/safe_simd/mod.rs:33`, `tests/thread_cleanup_test.rs:11`,
  `examples/bench_ivf_limit.rs:126`, `examples/profile_ivf.rs:69`,
  `benches/tier_isolation.rs:221`, `examples/md5_ablate.rs`); the one that was
  NEW to #482's branch, `src/owned_recon.rs:694 items after a test module`, is
  fixed here (`a59e652`).
* **`#![forbid(unsafe_code)]`, proved actively, twice, on the default build**,
  in files that are unconditionally compiled:

  | plant | diagnostic | lint anchored at |
  |---|---|---|
  | `src/owned_recon.rs:89` (`pub(crate) mod owned_recon;`, `lib.rs:184`, no cfg) | `error: usage of an 'unsafe' block` | **`lib.rs:13`** |
  | `src/loopfilter.rs:481` (`mod loopfilter;`, `lib.rs:136`, no cfg) | `error: usage of an 'unsafe' block` | `src/loopfilter.rs:1` (module-local `cfg_attr(not(asm_loopfilter), forbid(...))`) |

  Both restored byte-exact, verified by sha256 AND `git diff --exit-code`:
  `owned_recon.rs` `29572fc4…af1ad6`, `loopfilter.rs` `87d3828a…70353c`.
  The `loopfilter.rs` plant is the weaker of the two — its `forbid` is
  module-local and cfg-conditional — which is exactly why the `owned_recon.rs`
  plant, anchored at `lib.rs:13`, is the one that carries the claim.
* **Standing hazards replanted** under `--features __probe_wide`.
  `crates/rav1d-disjoint-mut` is byte-identical to `cebb97f` on this branch
  (`git diff cebb97f -- crates/` is empty, and `tracker_shard.rs` hashes to
  `d4e03d4a…423176d`, the value #482 recorded):

  | plant | `wide_exclusion` |
  |---|---|
  | baseline | ok (0.06 s) |
  | the in-lock `state` re-read deleted from `add_contended` ALONE | **FAILED** |
  | `active()` cut to one shard | **FAILED** |
  | after both restores | ok (0.05 s) |

  Both restored byte-exact (sha256 + `git diff --exit-code`), and the test was
  re-run green after restore with the file `touch`ed first — the mtime trap
  #482 recorded.
* **Miri, both models**, `cargo +nightly miri test -p rav1d-disjoint-mut
  --no-fail-fast`: see §14.
* **The corpus was re-run AFTER the last probe commit** (`6dc7cd5`, which adds a
  branch to `LfBlock::fill` — the decoder's hottest guard site — even though it
  folds to `false` in the default build): `--threads 1` 766 PASS / 768 keys /
  **0 differing**; `--threads 8` 753 PASS / 755 keys / **0 differing**; frame
  md5 `a00c11f454328023c58af14d55544cff` at t=1 and t=8 with the band both
  armed and disarmed. The default binary's SIZE is unchanged at 2,887,376 bytes
  across all three probe-adding commits, which is the evidence that the
  branches fold rather than merely being cheap.

### Memory, re-measured on this branch

Peak RSS, `/usr/bin/time -l` maximum resident set size, 20 frames, one decoder.
Unchanged by §9's edit, as it must be — nothing here allocates:

| arm | 8bpc t=1 | 8bpc t=8 | 10bpc t=1 | 10bpc t=8 |
|---|---|---|---|---|
| `base` | 99.4 MB | 106.3 MB | 149.2 MB | 157.3 MB |
| `head` (band armed) | 99.7 | 107.9 | 149.5 | 160.4 |
| `headoff` | 99.5 | 106.3 | 149.1 | 157.5 |

which reproduces §3 (99.4 / 107.8 / 149.1 / 160.4) to 0.1 MB.

### How close to the all-tracking-off ceiling

`head` against #467's whole-tracker-off ceiling, 8bpc — **the head column is
loaded and the ceiling column is #467's idle measurement, so this comparison is
indicative, not a like-for-like**:

| | t=1 | t=2 | t=4 | t=8 |
|---|---|---|---|---|
| `base` (loaded) | 1.29 | 1.60 | 1.79 | 1.91 |
| **`head`** (loaded) | **1.26** | **1.31** | **1.35** | **1.48** |
| whole-tracker ceiling (#467, idle) | 1.160 | 1.264 | 1.262 | 1.345 |

The remaining distance to the ceiling is ~0.10 / 0.05 / 0.09 / 0.14. §11a says
where 6.39 M of the 11.4 M registrations that produce it live, and §11c/§11d
say the obvious way to remove them does not work.

Arm-band disjointness (`gap_bands.py`), `head` vs `base`: **disjoint at 7 of 8
cells; 8bpc t=1 OVERLAPS** — the same cell #482 reported as its one
unseparated number, for the same reason (the win there is only 1.8%).

## 14. Miri, both models — re-run, not inherited

`crates/rav1d-disjoint-mut` is byte-identical to `cebb97f` on this branch
(`git diff cebb97f -- crates/` is empty; `src/tracker_shard.rs` hashes to
`d4e03d4a70183660cde4ef18cde777d5ef29530501c5a0e029a524e9c423176d`, the value
#482 recorded), so neither model can regress from a change that is not there.
Both were re-run anyway, each target in isolation via `--no-fail-fast`:
`cargo +nightly miri test -p rav1d-disjoint-mut`.

| model | targets | result | UB |
|---|---|---|---|
| Stacked Borrows (`MIRIFLAGS=""`) | 9 | **61 passed, 0 failed**, `rc=0` | none reported |
| Tree Borrows (`-Zmiri-tree-borrows`) | 9 | **61 passed, 0 failed**, `rc=0` | none reported |

Same 61/61 as §7b. `soundness` is the long pole at 992.7 s under Stacked
Borrows.

Nothing in this round touches the sub-crate, and nothing in `rav1d-safe` here
is reachable by Miri's test set — which is precisely why the corpus, `mt_stress`
and `multi_decoder_pressure` gates in §13 carry the runtime claim and Miri only
carries the tracker's.

### 12b. The tax, re-measured at n=15 — REAL at 0.8–1.3%, and no idle window ever came

§12's 9-round table left 8bpc t=1 at 1.0116 with p=0.180, i.e. not separated
from noise. A strict-idle-gate re-run was attempted and **returned zero rows in
seven minutes** (`rc=124`): at no point in this session was this box idle —
three agents were measuring on it. So the retest is n=15 on four cells, still
load-tagged (116 of 120 rows), trading breadth for sign-test power:

| | median | band | wins for headoff | p |
|---|---|---|---|---|
| 8bpc t=1 | **1.0081** | [0.9944..1.0610] | 2/15 | **0.007** |
| 8bpc t=8 | 1.0031 | [0.9634..1.0507] | 7/15 | 1.000 |
| 10bpc t=1 | **1.0126** | [0.9115..1.1964] | 2/15 | **0.007** |
| 10bpc t=2 | **1.0111** | [1.0000..1.0571] | 0/15 | **0.000** |

**So the honest verdict is: the tax is REAL and it is NOT at parity.** It is
~0.8% at 8bpc t=1, ~1.1–1.3% at 10bpc t=1 and t=2, and gone at 8bpc t=8. §9
took it from #482's 1.15–3.01% band down to 0.3–1.3%, which is a real
improvement and is not zero.

An inter frame on a mixed-GOP 10bpc stream therefore still pays about 1% for a
seam it gets nothing from. **That remains an argument for treating #482 as the
first half of a conversion.** What it is no longer is an argument that the seam
is *expensive*: at 8bpc — the common case — t=2/4/8 are at parity and only t=1
is measurably positive, at 0.8%.

Every band above overlaps, and every arm-pair ratio here is from a loaded box.

---

## 19. The V pass's batch cap — built, measured, and NOT SHIPPED

§15 called this "a separate, smaller, and *much* cheaper change" than the band,
"blocked on two mechanical things". Both asserts came out and the count came out
exactly as predicted. **It is still not shipped, because the wall clock says no**
— and the reason generalises further than the change does.

### 19a. The verdict first

| | |
|---|---|
| count | **1.971x on the V pass, exactly as §16 priced it**: 1,178,490 -> 597,876 regs/frame, whole decoder 11,401,399 -> 10,820,785 |
| the prize, in CPU | 580,614 regs x #485's own **4.04 ns** each = **2.35 ms/frame**, i.e. **<= 0.51% of a t=8 frame's CPU** and <= 0.74% of a t=1 frame's |
| wall clock, n=7-8, load-tagged | 8bpc t=1 **1.0157** (1/8 rounds faster), t=2 **1.0162** (0/8, p=0.008), t=4 1.0150 (3/8), t=8 **1.0005** (4/8, p=1.000); 10bpc all within [0.993, 1.003] and never significant. **No cell disjoint.** |
| shipped | **no.** `main`'s `LF_BATCH_MAX` stays 4. The implementation is preserved on this branch at `61f88dc` + `362e5d9`. |

**The framing "a cheap, sound, un-taken win" was wrong in one specific way, and
this is the transferable part: 1.971x is a RATIO on 30.7% of ONE site, and
nobody had converted it into milliseconds.** Two lines of arithmetic against
#485's own per-registration price put the whole lever under 0.51% of frame CPU
— at or below what this box resolves at n=9 under load — before a line was
written. Do that conversion first, every time.

### 19b. What is NOT done

* **The H pass is untouched and cannot be helped by this route.** 69.3% of
  `LfBlock::fill`, and its cap ratio is exactly **1.000** at 4, 8, 16, 32 and 64
  (`benchmarks/lf_cap_census_2026-08-10.txt`). Structural: its rectangle grows
  in the ROW direction, so a run of `n` groups costs `4n` however it is split.
* **`cdef_arm`'s 1,863,648 and `ctx.rs:99`'s 2,534,988 are not reduced.** §20
  attributes the second for the first time; it does not shrink it.
* **A cap-8 variant with a stack scratch was built and NOT measured to n=9.**
  Its prize is 365,712 regs = 1.48 ms = <= 0.32% of t=8 CPU, i.e. further below
  the resolution than the arm that already measured null. Its correctness was
  verified (parity 4/4, corpus 14/14, frame md5 identical, census 812,778 —
  exactly `LFCAP`'s cap-8 row) and it is left unmeasured deliberately.
* **The wall-clock numbers are load-tagged**, `foreign` 14-24 per row, taken
  under `measlock --load-ok` because three other agents held the box. Paired,
  interleaved, rotating-order ratios only; absolutes are not comparable to an
  idle campaign's. The 9th round was lost when two sweeps restarted (see 19f).
  **How bad the load was, stated rather than hidden:** the raw beta bands are
  8bpc t=8 base [52.33..121.89] ms/frame against head [52.28..127.67] — a 2.3x
  spread within one arm. That is why the `disjoint` column is `no` everywhere
  and why it CANNOT be used here: only the per-round pairing (both arms inside
  one cell, rotating order, same load) carries any signal at all. A reader
  should treat the +1.6% at t=1/t=2 as "sign-consistent across 8 rounds"
  (0/8 and 1/8) rather than as a calibrated magnitude, and the t=8 1.0005 as
  "no effect this instrument can see".

### 19c. The lever, priced at every cap — this part is solid and reusable

`--features __probe_lf_hist` (ported from #485, extended with `LFCAP`, which
adds for each NATURAL run what it would cost at each candidate cap). It is
**kept on `main`** because it is what any future attempt needs. The base column
reproduces #485's attribution to the registration:

| t=8, `v4k_8tile` 8bpc | regs/frame | share | regs/open | mean natural run |
|---|---|---|---|---|
| H — `filter_plane_cols_*` | 2,656,552 | 69.3% | 14.30 | 6.91 |
| V — `filter_plane_rows_*` | 1,178,490 | 30.7% | 6.33 | 7.00 |

| cap | V regs/frame | vs 4 | H regs/frame | vs 4 |
|---|---|---|---|---|
| 4 | 1,178,490 | 1.000 | 2,656,552 | 1.000 |
| 8 | 812,778 | 1.450 | 2,656,552 | 1.000 |
| 16 | 657,708 | 1.792 | 2,656,552 | 1.000 |
| 32 | 597,876 | **1.971** | 2,656,552 | 1.000 |
| 64 | 597,876 | 1.971 | 2,656,552 | 1.000 |

32 is the true maximum: `vm` is a `u32`, so an edge has at most 32 groups.

### 19d. It WAS sound, and the soundness argument is worth keeping

The band is REFUTED (§18) because the filter's read set is 2-D SPARSE and any
contiguous band reserves bytes nothing reads. A fused run has **no slack**:
every one of its `4 * groups` columns belongs to a group that filters, at the
same `wd` and therefore over the same `2 * reach` rows, so the union of the
members' rectangles IS the fused rectangle. Raising the cap changes the COUNT
and nothing about the relationship between extent and read set. The write side
was kept identical by scanning and writing back in 16-column chunks.

**The one-line test that separates the three schemes**, worth more than any of
them: *does the reservation contain a byte no member of the batch reads?*
Strided hull — yes, the gaps between rows (2.65x slower). Read band — yes, the
gaps between edges (decode failure). Fused run — **no** (sound; just not worth
it).

An honest gap in the evidence: planting the un-chunked write-back (a genuine
over-reservation above 16 columns) and running `md5_inventory --threads 8
--group 8-bit/data` gave **358/358 pass, 0 errors**. The corpus did not catch
it in one run — #485's lesson again — and the liveness of that arm on the
corpus vectors was not proved, so the null bounds nothing.

### 19e. Where the time went: the machinery, three times

The first cut measured **+3.0% t=1 / +7.9% t=8**
(`benchmarks/lf_vbatch_2026-08-10_v1.tsv`). An isolation arm — whole machinery
present, `LF_BATCH_V` pinned back to 4 — separated machinery from batching
(`benchmarks/lf_vbatch_iso_2026-08-10_v1.tsv`, `v4k_8tile` 8bpc):

| arm | t=1 | t=8 |
|---|---|---|
| base | 1.000 | 1.000 |
| v1 (batch 32) | 1.030 | 1.086 |
| **machinery only (batch 4)** | **1.187** | 1.049 |

A `sample` profile of the machinery-only arm named two leaves base does not
have — `LfBlock::close` 2.36% and `___arcane_lf_dispatch_u8` 1.30%. Three
causes, each removed, each a general trap:

1. **Reading per-group thresholds from `params` instead of a materialised
   table.** Done to avoid zero-filling 32 entries per run; it put a slice bounds
   check and an `Option` branch in the 8-lane chunk loop of BOTH kernels, so the
   H pass paid for a feature it cannot use.
2. **A fixed 128-pixel V scratch stride.** A 1-group V rectangle spanned
   14 x 128 = 1,792 bytes where base spanned 224.
3. **A write-back chunk loop that ran when `w <= 16`,** where it can only ever
   iterate once.

After all three (`benchmarks/lf_vbatch_iso_2026-08-10_v3.tsv`, n=3): head
1.014 / 0.981, machinery-only 1.019 / 0.998. The residual ~2% at t=1 is the
thread-local scratch and the `&mut`-through-`&mut` scratch handle that the
2,048-pixel array forced — and ~2% is four times the whole prize.

**The general rule, and it is §11's lesson from a third side.** §11 said: price
the EXTENT, not just the count. This adds: **price the MACHINERY, and price the
count in MILLISECONDS before building anything.** A correct count reduction
whose machinery costs more than the registrations it removes is a regression,
and the only way to see that is an isolation arm that KEEPS the machinery and
REMOVES the reduction.

### 19f. A measurement-integrity note, because it cost the 9th round

Two of this round's sweeps restarted from round 0 simultaneously and then ran
CONCURRENTLY, after `measlock`'s owner-file format changed under a running
holder (another agent was fixing a real release bug in it at the time). The
committed table is the pre-restart snapshot at n=7-8, which is why the row
counts are 7 and 8 rather than 9. Two lessons already in the brief were the
ones that bit: do not change a tool a run depends on while the run is live, and
`--load-ok` mutual exclusion is only as good as the lock file's format
agreement.

## 20. `ctx.rs:99` — attributed for the first time, NOT reduced

2,534,988 registrations per frame at t=8, 22.2% of the decoder's population, and
until now nobody could say what it was. `CaseSetter::set_disjoint` takes its
borrow through one `index_mut` line, so `--features probe-sites` reported all of
it as a single site. A **cfg-gated `#[track_caller]`** on `set_disjoint` (probe
builds only — a `#[track_caller]` shim changes codegen at every call site, and a
probe must not) pushes the `Location` up to the real caller.

The 12 rows it splits into sum to **2,534,988 exactly**, so the attribution is
complete and nothing is lost. `benchmarks/ctx99_sites_2026-08-10.tsv`:

| regs/frame | site | what it writes |
|---|---|---|
| 376,260 | `src/recon.rs:2767` | `ccoef` context, both chroma planes |
| 277,428 | `src/recon.rs:2380` | `lcoef` context |
| 8 x 188,130 | `src/decode.rs:1997..2005` | `tx_intra`, `tx`, `mode`, `pal_sz`, `seg_pred`, `skip_mode`, `intra`, `skip` |
| 188,130 | `src/decode.rs:2029` | `uvmode` |
| 188,130 | `src/decode.rs:3811` | inter context |

**All 12 are MUTABLE, and the mean extent is 2.3 bytes.** The count is that
high for a structural reason and not a sloppy one: `BlockContext` is ~20
SEPARATE `DisjointMut<Align8<[T; 32]>>` fields (`src/env.rs:32`), and a
`CaseSet::many` call updates 8-13 of them for the same block at the same
`offset..offset + len`. One registration per field per direction, on a 32-byte
array.

### 20a. The reduction that exists, and why it is not in this PR

Every one of the 12 is a `CaseSet::<_, _>::many([(&t.l, ..), (ta, ..)], ..)`
over exactly two directions, so the population splits **exactly 50/50**:

* **`f.a[t.a]`** — `Vec<BlockContext>` on `Rav1dFrameData`, reached through the
  shared `fc.data.try_read()` guard. This is §1's blocker verbatim: no `&mut`
  exists at any point, and its disjointness across concurrent tiles is a
  TILE-KEYED argument, which §3 measured and rejected as unsound when partial.
  **1,267,494/frame, not reducible by ownership.**
* **`t.l`** — a `BlockContext` field of `Rav1dTaskContext`, which is per worker
  thread and is already reached through `&mut` everywhere `recon_band` is
  (§4/#482's model, `src/internal.rs:1246`). `DisjointMut::get_mut(&mut self)`
  already exists and takes no registration. **1,267,494/frame, 11.1% of the
  decoder's whole population, removable with borrowck as the proof and no
  `unsafe`.**

What blocks it is shape, not soundness: `CaseSet::many` takes `[T; N]` — a
HOMOGENEOUS array — so both directions must be the same type, and `&mut
BlockContext` and `&BlockContext` are not. Converting means either splitting all
22 `CaseSet` sites that mention `t.l` into two calls with duplicated closure
bodies, or a macro that expands the body twice, plus a `set_exclusive` on
`CaseSetter`. That is a real refactor across `decode.rs` and `recon.rs` with a
full corpus + Miri gate behind it, and it is **not attempted here** — landing it
half-done would be worse than the study.

**Do not attack this by coalescing fields.** Laying the 8 same-offset fields out
adjacently so one registration covers them makes the reservation span the gaps
BETWEEN fields, which other blocks legitimately write — §19c's test says no.

## 21. Correctness gates for this round

All on the final tree (`e19c4f4` onward), whose source diff against `f87b12c`
is **pure addition** — 112 lines in `src/loopfilter.rs`, 9 in `src/ctx.rs`, 3 in
`lib.rs`, 7 in `examples/probe_tracker.rs`, 6 in `Cargo.toml`, and **zero
removed lines in any of them**; `src/safe_simd/loopfilter_arm.rs` is
byte-identical to base.

| gate | result |
|---|---|
| corpus, by NAME with the md5 as the value, `--threads 1` | 766 PASS / 768 keys / **0 differing**, `SETDIFF: CLEAN` |
| corpus, `--threads 8`, `8-bit/film_grain` + `10-bit/film_grain` dropped from BOTH sides (#479) | 753 PASS / 755 keys / **0 differing**, `SETDIFF: CLEAN` |
| census, `probe-sites`, `lost=0` | **11,401,399**/frame at t=8 — base's number exactly |
| frame md5 vs base | identical at 8bpc and 10bpc, t=1 and t=8 |
| `cargo test --lib`, release AND **debug** | 75 passed each (#482 hit a profile-dependent panic; this checks both) |
| `mt_stress` 1/2/4/8/16 | 25/25 cells |
| `multi_decoder_pressure.sh` 12 procs x 3 iters | PASS, every md5 matches the serial reference |
| x86 clippy, `--target x86_64-apple-darwin --all-targets --keep-going -D warnings` | 11 failing sites, **set-diff against base empty in both directions**. Note this is STRICTER than CI, whose clippy job is `cargo clippy --no-default-features --features ... -- -D warnings` with **no `--all-targets`**, so it never lints `examples/`; all 11 sites are in `_dev` examples plus one test-only `fn`, and CI's Clippy job is green on both base and head. |
| Miri, Stacked Borrows, each target in isolation | **61 passed / 0 failed, rc=0, no UB** |
| Miri, Tree Borrows, each target in isolation | **61 passed / 0 failed, rc=0, no UB** |

**`#![forbid(unsafe_code)]`, proved ACTIVELY and non-vacuously.** Planted in
`src/ablate.rs` — an always-compiled file with NO module-local `cfg_attr`, so
the failure can only come from the crate attribute — with an `#[allow(unsafe_code)]`
on it. Both errors anchor on **`lib.rs:13`**: `allow(unsafe_code) incompatible
with previous forbid ... overruled by previous forbid` and `usage of an unsafe
block`. Restored byte-exact (sha256 match + `git diff --exit-code`).

**Teeth, planted and confirmed FAILING, each restored byte-exact.**

| mutation | gate that caught it |
|---|---|
| `lane_thr` indexes `c` instead of `2c` | NEON parity 3/4 FAILED at `groups=3` |
| V kernel dispatched at the wrong const stride (`sw=64` -> `32`) | NEON parity FAILED at `groups=9` — i.e. the WIDE path specifically |
| `copy_row_v`'s 4-element tail deleted | corpus 13 of 14 tests FAILED |
| `note_natural` prices every cap as cap 4 | `LFCAP` cap-32 collapses 1.971 -> 1.000 |
| un-chunked write-back (a real over-reservation) | **NOT caught** — 358/358 at t=8. Recorded in §19d as a gap, not glossed. |

**Standing hazards, `--features __probe_wide`, run even though `crates/` is
untouched** — "the diff is empty" is reasoning, not measurement (`git diff
f87b12c -- crates/` IS empty and `tracker_shard.rs` hashes to
`d4e03d4a70183660cde4ef18cde777d5ef29530501c5a0e029a524e9c423176d`, and they
were run anyway):

| arm | `wide_exclusion` |
|---|---|
| control | **passes** |
| in-lock `state` re-read deleted from `add_contended` | **FAILS** |
| `active()` cut to one shard | **FAILS** |

Both restored byte-exact (`shasum -c` OK, `git diff --exit-code` clean).
