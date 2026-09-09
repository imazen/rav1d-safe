# Published 0.5.x decoders against published disjoint-mut 0.3.2

Whether an *older* `rav1d-safe` still works once `rav1d-disjoint-mut` 0.3.2 reaches
it. This matters because the upgrade is automatic: 0.5.5/0.5.6 require `^0.3.0` and
0.5.7 requires `^0.3.1`, and both ranges resolve to 0.3.2. Existing lockfiles hold
until `cargo update`; nothing pins them away from it.

Measured 2026-09-08 against the archives crates.io serves, not local builds.

## API surface actually used

Published 0.5.7 references eleven disjoint-mut items: `DisjointMut`,
`DisjointMutGuard`, `DisjointImmutGuard`, `DisjointMutSlice`, `DisjointMutArcSlice`,
`AsMutPtr`, `ExternalAsMutPtr`, `SliceBounds`, `TryResizable`, `TryResizableWith`,
and the `align` module. All are present in 0.3.2 with unchanged generic signatures,
and no public item was removed anywhere (0.3.1 → 0.3.2 is additive, 62 → 164 names).

## Auto-traits: the part that could have broken silently

0.3.2 changed `DisjointMutGuard`'s payload from `slice: &'a mut V` to a `NonNull`.
`&'a mut V` inherits `Send`/`Sync` from `V`; `NonNull<V>` has neither. Guards would
have silently stopped crossing thread boundaries, which no consumer's compile error
would explain. 0.3.2 restores them explicitly:

    unsafe impl<T: ?Sized + AsMutPtr + Sync, V: ?Sized + Send> Send for DisjointMutGuard<'_, T, V> {}
    unsafe impl<T: ?Sized + AsMutPtr + Sync, V: ?Sized + Sync> Sync for DisjointMutGuard<'_, T, V> {}

Those bounds match what the reference field derived. Verified by compiling
assertions against both published versions rather than by reading the bounds —
5 types x {Send, Sync, UnwindSafe, RefUnwindSafe}, **20/20 identical**.

## Build and decode identity

Six build combinations, all succeed:

| rav1d-safe | requires | its original | with 0.3.2 |
|---|---|---|---|
| 0.5.5 | `^0.3.0` | builds | builds |
| 0.5.6 | `^0.3.0` | builds | builds |
| 0.5.7 | `^0.3.1` | builds | builds |

Decoding the 25 committed `tests/crash_vectors/*.obu` with `examples/decode_md5`,
comparing each version's original disjoint-mut against 0.3.2: **byte-identical output
at 1, 4 and 8 threads**, every version. 17 of the 25 produce real pixel data; the
other 8 are correctly rejected before any frame (MD5 of empty) under both versions.

Earlier releases cannot reach 0.3.2 at all — rav1d-safe 0.1.0/0.3.x pin `^0.1.1` and
`^0.2.1`, and cargo will not cross a 0.x minor.

## Caveat: these decoders predate the tile-threading fix

At 4 threads under **concurrent system load**, 0.5.5 on
`tile_threading_cdef_lpf_race.obu` returns more than one distinct MD5 across runs and
panics in `include/dav1d/picture.rs` with `overlapping DisjointMut`; one run of 0.5.6
wedged for over ten minutes. On an idle machine 60 consecutive runs of each version
are clean, so it is contention-dependent and easy to miss.

This is not a 0.3.2 regression. It reproduces under 0.3.0 and 0.3.1 as well, and it is
the pre-fix behaviour of the race that `49df1fc` (2026-07-03) fixed — the same commit
that added this trigger vector and the wedge fix. 0.5.5, 0.5.6 and 0.5.7 all shipped
before it. Divergence counts differed slightly between disjoint-mut versions, but
across 150 runs the sample is far too small to claim 0.3.2 changes the rate either way.

What 0.3.2 does contribute is that the overlap is *detected*: the guard checks turn a
silent wrong-pixel race into a loud panic at the offending call site.

Anyone still on 0.5.x who decodes with `threads > 1` should move to 0.6.0 for the
threading fix, independently of which disjoint-mut they resolve.
