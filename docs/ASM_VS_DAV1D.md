# rav1d `asm` build vs dav1d: where the gap is (2026-10-01)

Question: our `asm` build links dav1d's assembly kernels, so why is it still
6-22% slower than dav1d 1.5.3? Short answer, measured below: **not the kernels,
not memory traffic. It is Rust glue instructions**, mostly inlined bounds-check /
`Option` / `min`-`max` code, spread over the per-block control path.

## Results

Wall clock (dav1d wall / frames vs `profile_ivf` inner ms/frame, 1 thread):

| stream | dav1d | rav1d-asm | gap |
|---|---:|---:|---:|
| 4K intra, 16 frames | 24.4 ms/f | 26.1 | +7% |
| 480p inter (sframe, 1800 frames) | 0.82 | 1.00 | +22% |
| 8-bit intra, 39 frames | 1.95 | 2.07 | +6% |

Retired instructions (callgrind, both on the AVX2 asm path, same streams):

| stream | dav1d | rav1d-asm | ratio |
|---|---:|---:|---:|
| 480p inter, 150 frames | 22.7 MIr/f | 28.6 | 1.233 |
| 4K intra, 4 frames | 363 MIr/f | 440 | 1.211 |

The instruction ratio matches the wall-clock gap on small inter frames, so it is
an instruction excess, not stalls or IPC.

What it is NOT:
- **Kernel calls are identical.** Counting every MC kernel entry with gdb
  breakpoints: 25,936 calls/frame in both programs, equal per kernel
  (`put_6tap`, `put_8tap_*`, `blend_*`, `warp_affine_8x8`, ...). Loop filter,
  msac, ipred, itx and refmvs asm instruction totals also match.
- **Not memcpy/memset.** An `LD_PRELOAD` shim (`scripts/perf/libc_call_shim.c`)
  found real waste (a 37-entry guard array zeroed per `splat_mv` call, a libc
  `memcpy` per palette pixel, small `px_copy` fall-through; fixed in 2dae7c50)
  but removing it bought only -0.8..-1.5%. Page faults and kernel time match.

Where the excess is (inclusive instructions per frame, 480p inter, `decode_b`
total +5.5 M of 20.6 M): `decode_coefs` +2.0 M, `recon_b_inter` +2.1 M
(`mc` +0.55, `obmc` +0.53), `refmvs_find` +0.7 M, `recon_b_intra` +0.5 M.
Our own source files total LESS than dav1d's C (12.7 vs 14.6 MIr/f); the excess
is **inlined std/dep code attributed to `core::cmp`, `num::uint_macros`,
`option`, `slice::index`, `iter::range`: 3.5 MIr/f on 480p inter, 58 MIr/f
(13% of all instructions) on 4K intra.** Functions incurring the most of it
(480p inter, kIr/f inlined-std of function total): `lf_mask::create_lf_mask_inter`
644/1402, `decode_b` 413/2760, `decode_coefs_class` 337/1638, `refmvs_find`
250/1261, `recon_b_inter` 216/1475, `mc` 165/1884, `splat_mv::call` 155/355.
On 4K intra `ipred_prepare` is +5 MIr/f over dav1d's (17.7 vs 12.8).

Next steps this points at (not done): cut the inlined-std tax in those
functions (hoist bounds checks, `get`-free fixed-size array access, replace
`Option`/`checked_*` chains on hot paths, drop `min`/`max` on proven ranges),
starting with `splat_mv::call`'s per-row guard, `ipred_prepare`,
`create_lf_mask_inter`, then `decode_coefs`.

## Method (reproducible)

1. **Source-built dav1d with symbols.** The distro dav1d is stripped.
   `git clone --depth 1 --branch 1.5.3 https://code.videolan.org/videolan/dav1d.git`,
   `meson setup build -Dbuildtype=release -Ddebug=true -Ddefault_library=static
   -Denable_tests=false -Denable_examples=false && ninja -C build tools/dav1d`.
   Its callgrind total is within 2% of the distro binary.
2. **Cut the stream** to a short prefix (IVF is trivial to truncate; fix the frame
   count at byte 24). Callgrind runs ~50x slower.
3. **Instruction counts:** `valgrind --tool=callgrind` on
   `dav1d -i f.ivf --muxer null --threads 1` and on
   `profile_ivf f.ivf N` for N=1 and N=2 with `RAV1D_THREADS=1`. `profile_ivf`
   does a warm-up pass, so **N=2 minus N=1 is exactly one pass**. Under valgrind
   both programs run the AVX2 kernels (valgrind has no AVX-512), so the totals are
   comparable. Analyze with `scripts/perf/cg_lines.py files|std`.
4. **Call counts:** gdb `break <kernel>` + `ignore N 1000000000` for every
   `dav1d_put_*`/`prep_*`/`blend_*`/... symbol, `info breakpoints` reports hits.
   Run this natively (real execution dispatches the AVX-512 kernels).
5. **libc calls:** `scripts/perf/libc_call_shim.c` counts `memcpy`/`memmove`/
   `memset` calls and bytes by size bucket and records callers of every call;
   resolve callers with `addr2line -f -C -e <binary> <offset>`.

## Pitfalls hit (so you don't)

- **Symbol-attribution artifacts for asm.** Self cost by symbol made dav1d look
  like it ran no `put_*` kernels and us like we ran 1.75 MIr/f of them. It was
  attribution (asm local labels without sizes), not behaviour; the gdb call
  counts settled it. Compare INCLUSIVE cost of C/Rust entry points, or call
  counts, not self cost by asm symbol.
- **Callgrind attributes by the line-table file.** Inlined `core::cmp` code is
  charged to `cmp.rs`, not to the function that inlined it; use
  `cg_lines.py std` to find who incurs it.
- **`profile_ivf` wall time is not comparable to dav1d's.** It makes a warm-up
  pass plus a measured one. Use dav1d wall / frame count against the inner
  `RESULT` ms/frame; very short streams (2 frames) are dominated by startup.
- **`profile_ivf` sets `max_frame_delay = 1` (tile threads only).** Pass
  `--framedelay 1` to dav1d for a like-for-like comparison at t>1.
- Picture buffers are zero-filled once at allocation (about 54% of the 4K
  stream's memset volume): a per-decoder cost, ~1% for a single-frame AVIF.
