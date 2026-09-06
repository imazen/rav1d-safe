# Published DisjointMut regression check

This independent workspace resolves **crates.io rav1d-disjoint-mut =0.3.1**,
not the decoder's local path dependency. Its lockfile records the package
checksum. It runs the existing guard-move regression test unchanged.

From the repository root, with nightly Miri installed:

```sh
export TMPDIR="$HOME/tmp"
cargo +nightly miri test --locked --manifest-path audit/published-disjoint-mut/Cargo.toml --test guard_move_release
MIRIFLAGS=-Zmiri-tree-borrows cargo +nightly miri test --locked --manifest-path audit/published-disjoint-mut/Cargo.toml --test guard_move_release
```

Use the workspace `run-heavy --mem 16G --jobs 8 --` wrapper on shared hosts.
Capture stdout and stderr; an error exit is expected for this affected release.

Verified 2026-09-05 with Miri `da86f4d072` (2026-07-24): **both invocations
report undefined behavior** in `moving_a_mut_guard_into_drop_is_not_ub`.
Stacked Borrows identifies a protected Unique tag carried by `drop(g)`;
Tree Borrows identifies a foreign write disabling a protected tag. The test
uses only safe caller code. Each process aborts at the first test; the shared
guard test is not covered by these two invocations.

The same test target passed on main under both models during setup. This
supports the existing guard representation fix, not a verdict on all of main.
Published rav1d-safe 0.5.7 depends on DisjointMut `0.3.1`; this reproducer tests
the dependency abstraction, not reachability from a particular AV1 bitstream.

Raw logs are preserved as [stacked.log](stacked.log) and [tree.log](tree.log),
with the original copies at
`~/tmp/rav1d-review-2026-09-05/logs/published-031-guard-move-{stacked,tree}.log`.
