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

# The loop-filter COLUMN band (branch `perf/lf-band`, base `main` @ `0f6bf10`)

Same box: Apple M4 Pro (12 cores, 8P+4E), macOS 26.5.2. Every timed run below
is wrapped in `measlock`, which takes a cross-agent exclusive lock and then
waits for the box to go quiet.

§11 priced `LfBlock::fill` and told the next attempt what it had to answer
first. This section answers it and builds the thing.

## 15. What is NOT done — read this before the parts that work

* **The V pass is untouched.** The band covers `filter_plane_cols_*` (the H
  pass) only. `fill`'s residual 1,181,034 registrations/frame at t=8 are
  essentially all V (`filter_plane_rows_*`), and §16 measures that lifting
  `LF_BATCH_MAX` from 4 to 32 would divide them by **1.971**. That is a
  separate, smaller, and *much* cheaper change than this one, and it is NOT
  made here — it needs a rectangular scratch (`LF_BW` is 16 and `LF_BLOCK_LEN`
  is asserted to be `LF_BW * LF_BW`) and the NEON kernel's
  `LF_GROUPS == LF_BATCH_MAX` assert decoupled into a guard batch and a kernel
  batch. Nothing here does any of that.
* **`cdef_arm`'s 1,863,648 and `ctx.rs:99`'s 2,534,988 are untouched**, as in
  every prior round. After this branch they are the two largest sites left.
* **The mirror in `close_band` is NOT covered by the corpus.** §18 plants its
  removal and **766-vector-subset gate still passes 14/14**. It is kept anyway,
  because it is what makes "the band never differs from the picture" an
  invariant rather than an assumption — but the honest statement is that no
  test here can tell whether it is load-bearing, and the reason it appears
  inert (a wider filter implies a more distant next edge) is an *argument*, not
  a measurement.
* **Not measured, unchanged from §10:** x86_64 and wasm32 (compile-checked
  only), `asm` / `c-ffi` (the band is dropped there — `call` takes the picture
  path and the arm is simply unaccelerated), `unchecked`, t=16, any vector
  below 4K, and any vector with loop restoration live.
* **The band's copy is not separately priced.** Its cost is inside the A/B
  below, not isolated from the registrations it removes.

## 16. Attribution: which HALF of `fill` is it, measured

`--features __probe_lf_hist` (new, counts only) splits the site by pass. It
counts the whole process including `probe_tracker`'s warmup decode, so its raw
totals are `(iters+1)/iters` of the per-frame figure; at `iters=3` the scale is
exactly 3/4, and the two halves then sum to 3,835,042 **to the registration**,
which is the instrument's control.

| t=8, `v4k_8tile` 8bpc | regs/frame | share | regs/open | shape |
|---|---|---|---|---|
| H — `filter_plane_cols_*`, vertical edges | **2,656,552** | **69.3%** | 14.30 | `4 * groups` rows x `2 * reach` px |
| V — `filter_plane_rows_*`, horizontal edges | **1,178,490** | **30.7%** | 6.33 | `2 * reach` rows x `4 * groups` px |

and the run-length histogram says **79% of opens hit the `LF_BATCH_MAX = 4`
cap exactly** (mean natural run 6.91 H / 7.00 V, with a spike at 32 on V). So
the obvious lever — fuse harder — was priced from the same run, by counting
what each natural run WOULD have cost uncapped:

| cap 4 -> 32 | now | uncapped | ratio |
|---|---|---|---|
| V | 1,178,490 | 597,876 | **1.971** |
| H | 2,656,552 | 2,656,552 | **1.000** |

**Exactly 1.000 for H, and that is structural, not a rounding artefact.** The
H rectangle grows in the ROW direction (`h = 4 * groups`), so a run of `n`
groups costs `4n` registrations however it is split; the V rectangle grows
along a picture row (`w = 4 * groups`) at a fixed `h = 2 * reach`, so fusing
divides its count. **69.3% of the site cannot be bought by fusing along the
run at all** — only by fusing ACROSS the caller's `x` loop.

## 17. The band, and why it needs no halo

§11b's proposal was a row band with a halo, and its answer was: expressible as
copy-in/copy-out, NOT as a shared halo, because `&mut` exclusion is static and
a run-time ownership handoff is a borrow tracker wearing a different hat. It
also sized the copy at ~3.1 MB per superblock row and ~53 MB per frame, and
said to check that arithmetic before writing any conversion.

**The halo question does not arise here, because this band never owns
anything.** `src/lf_band.rs`:

* `LfBand::fill_from` copies `4 * len` picture rows x `4 * w + 14` pixels into
  a plain `Vec<BD::Pixel>` — **one immutable guard per picture row, over a
  CONTIGUOUS span**, once per superblock, for all of that superblock's `x`
  positions.
* `LfBlock::open_band` / `fill_band` read the rectangle out of that `Vec`. No
  guard, no tracker, no policy branch.
* `LfBlock::close_band` writes each changed span **to the picture, with the
  same mutable guard `close` always took**, and mirrors it into the band.

So the band is a read CACHE that is equal to the picture at every point, which
is what makes `open_band` returning `None` (rectangle off the plane, band too
small) *ordinary* rather than a special case: the caller falls through to the
picture path and reads identical bytes. Nothing is handed between workers, so
there is no halo protocol to express.

**The copy is 170x smaller than §11b's sizing** because the unit is a
superblock, not a superblock row: 128 rows x 142 px is ~18 KB at 8bpc, against
3.1 MB. And it is not even net-new traffic — the per-edge windows it replaces
are 14 px wide every 4 px, i.e. ~3.5x redundant.

### Registrations, measured — `--features probe-sites`, `lost=0`

One binary, `RAV1D_LF_BAND=0/1`. `v4k_8tile` 8bpc t=8:

| site | band OFF | band ON |
|---|---|---|
| whole decoder | **11,401,399** | **8,941,599** |
| `LfBlock::fill` | 3,835,042 | 1,181,034 |
| `LfBand::fill_from` (copy-in) | 0 | 194,208 @ **142 B** |
| `LfBlock::close` (write) | 17,852 | 12,616 |
| `LfBlock::close_band` (write) | 0 | 5,236 |

Three things to read off it:

1. The band-off column **reproduces §11a's 11,401,399 exactly**, so the
   `Option` threaded through the dispatch changes no registration anywhere.
2. `fill` falls by **2,654,008**, against the 2,656,552 §16 attributed to the H
   pass — i.e. 99.9% of the H pass is banded and the residual 2,544 are opens
   that declined. The remainder of `fill` is the V pass, untouched.
3. **The write population is identical: 17,852 = 12,616 + 5,236.** The only
   MUTABLE reservation in the loop filter is bit-for-bit what it was, merely
   split across two functions. Nothing here widens a mutable extent — which is
   the direction #479 and #469 were burned by.

Net **-2,459,800 registrations/frame, -21.6% of the whole decoder's
population**, bought at an extent of 142 contiguous bytes per guard. That is
the opposite trade from §11c's hull, which removed 3.46 M by paying 50-60 KB
of strided extent and measured **2.65x SLOWER**.

## 18. The band is REFUTED under concurrent filtering — and the first run passed

§17's registration table is real and the band is nevertheless **not
shippable**. It ships **default OFF** (`RAV1D_LF_BAND=1` to arm), in the shape
`__probe_lf_hull` already uses for the strided-hull negative.

### What went wrong

**The loop filter's read set is 2-D SPARSE, and every per-row band is
contiguous.** A band reserves, for each of its rows, one span covering the
whole superblock; the pass actually reads only the `+-reach` tap windows
around the edges that filter, and a row in which nothing filters is not read
at all. The difference is columns and rows that are reserved but never read —
and under concurrent filtering another worker is legitimately writing them:

```text
current:  &     _[163840..163968]   <- the band's 128-px row copy-in
existing: &mut  _[163944..163952]   <- a concurrent 8-px write inside it
```

That is the SAME defect class as §11c's hull, transposed. The hull reserved
the gaps BETWEEN rows; this reserves the gaps BETWEEN edges. §11c's version
was merely slow because its extent hit the wide path; this one is a false
positive, i.e. a decode failure.

### The numbers, and the sample that lied

`examples/md5_inventory --threads 8`, group `8-bit/data` (358 vectors), same
box, same load, repeated:

| arm | pass | error |
|---|---|---|
| base commit `0f6bf10`, its own binary | 358 / 358 / 358 | **0 / 0 / 0** |
| band off (this binary) | 358 / 358 | 0 / 0 |
| **column band only** | 357 / 356 / 358 | **1 / 2 / 0** |
| **column + row band** | 294 | **64** |
| default build (band off), after the fix | 358 x4 | **0 x4** |

**The column-only band's first full `--threads 8` corpus run passed 753/755
`SETDIFF: CLEAN`.** One sample. It was committed on that basis, and the
failure only appeared when the row band widened the same defect enough to fire
reliably. "766 vectors passed" is evidence, not a proof — and a rare
false positive is a decode failure, not a wrong pixel, so it cannot be
detected by an md5 diff at all, only by an error count.

### `tile_threading_active()` is the WRONG gate, measured

The obvious rescue is the latch that already lets `fill_hull`, `block_mut` and
`compact_read` widen a reservation. It does not work, because it gates
concurrent **tile** workers and the loop filter's concurrency is between
**sbrow filter tasks** — inserted for `sby+1` before the selected task for
`sby` even runs (`src/thread_task.rs:1030-1043`) — which exist whenever
`n_tc > 1` however many tiles a frame has. Gated that way:

* the `v4k_8tile` t=8 census is byte-identical to `main` (11,401,399), so the
  band provably never armed on that vector, and
* `8-bit/data` at t=8 **still produced 8 errors in one run of two.**

A gate that is right for this site would have to mean "no other thread can be
filtering this picture", which the decoder does not currently expose.

### What the default build is

Byte-for-byte `main`'s behaviour, verified rather than assumed:

* census, `--features probe-sites`, `lost=0`: **6,005,602** at t=1 and
  **11,401,399** at t=8 — both exactly §11a's numbers;
* corpus set-diffed BY NAME with the md5 as the value, against
  `benchmarks/aarch64_md5_fixes_2026-08-07_final.tsv.zst`: `--threads 1`
  **766 PASS / 768 keys / 0 differing, SETDIFF: CLEAN**; `--threads 8` with
  both film-grain groups dropped from BOTH sides (#479) **753 PASS / 755 keys
  / 0 differing, SETDIFF: CLEAN**;
* the write population is untouched in every arm: 17,852 per frame, split
  17,327 + 525 between `close` and `close_band` when armed.

### The one number the armed arm is still good for

Paired user CPU, `v4k_8tile` 8bpc t=8, 20 frames, one binary, arms interleaved
with rotating order within each round, under `measlock`, n=9 (foreign_max=1,
so load-tagged):

| | median | band | faster |
|---|---|---|---|
| band / band-off | **0.9628** | [0.9523..0.9781] | **9/9** |

**That is a real 3.7% and it is not available**, because the arm that produced
it is the arm that fails 1-64 vectors per run. It is recorded because it
prices what a SOUND removal of this population would be worth: it is the
first direct measurement that removing ~2.4 M of `fill`'s registrations
without paying extent is worth several percent of CPU, which §11f could only
bound from above by doubling.

### What the next attempt should and should not do

* **Do not build another contiguous band.** Column-major, row-major, and both
  together are now all measured. The obstruction is that the read set is
  sparse in 2-D and the tracker's unit is an interval.
* **The V pass has a cheap, sound, un-taken win**: §16 measured that lifting
  `LF_BATCH_MAX` from 4 to 32 divides the V pass's registrations by **1.971**
  (1,178,490 -> 597,876/frame) with **no extent change beyond a contiguous
  in-row span that the fused groups genuinely read**. That is the one lever
  here that buys count without buying extent. It is blocked on two mechanical
  things, both untouched: `LF_BW` is 16 with
  `assert!(LF_BW * LF_BW == LF_BLOCK_LEN)`, so the scratch has to become
  rectangular rather than square; and `src/safe_simd/loopfilter_arm.rs:156`
  asserts `LF_GROUPS == LF_BATCH_MAX`, so the GUARD batch has to be decoupled
  from the KERNEL batch. Watch `LfScratch::new`'s zero-init while doing it —
  it runs once per DSP call, ~100 K times a frame.
* **H's 69.3% needs a different mechanism entirely**, not a bigger batch:
  §16's cap-lift ratio for it is exactly **1.000**.
