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
  and it is bigger than this one (`src/mc.rs` alone has 54 `PicOffset`
  parameters against `src/ipred.rs`'s 20).
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
* **Not measured:** x86_64 (compile-checked only), `--features unchecked`,
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
