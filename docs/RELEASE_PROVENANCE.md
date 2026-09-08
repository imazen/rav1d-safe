# Release provenance: crates.io versions to source commits

Every published version of `rav1d-safe` and `rav1d-disjoint-mut`, mapped to the
commit it was published from and verified against that commit's tree. Recorded
2026-09-08. Regenerate with:

    python3 tools/verify-published-provenance.py --check-tags

## Method

Cargo embeds `.cargo_vcs_info.json` in every archive it uploads, recording the
commit the publish ran from. That file is a claim, so the tool checks it: each
packaged file is compared byte-for-byte against `git show <sha>:<path>`. A
release counts as *reproduced* only when the recorded commit regenerates the
packaged source exactly.

Cargo rewrites `Cargo.toml` and regenerates `Cargo.lock` when packaging, so
neither is comparable to the tree — the shipped `Cargo.lock` for 0.5.7 matches
no commit in the repository, because Cargo prunes it to the published crate's
own dependency closure. `Cargo.toml.orig` preserves the committed manifest and
is compared in its place.

## rav1d-safe

| Version | Published from | Tag | Reproduced |
|---|---|---|---|
| 0.1.0 | `441dad0c` | `v0.1.0` | yes, 111 files |
| 0.3.0 | `2ed82900` | `v0.3.0` | yes, 115 files |
| 0.3.1 | `290cb679` | `v0.3.1` | tracked source yes, 118 files; see note |
| 0.4.0 | `c95d77a7` | `v0.4.0` | yes, 134 files |
| 0.5.0 (yanked) | `d62def54` | `v0.5.0` | yes, 142 files |
| 0.5.1 (yanked) | `dc4be822` | `v0.5.1` | yes, 142 files |
| 0.5.2 | `9168f969` | `v0.5.2` | yes, 142 files |
| 0.5.3 | `951f44ed` | `v0.5.3` | yes, 142 files |
| 0.5.4 | `914b8a9c` | `v0.5.4` | yes, 136 files |
| 0.5.5 | `2ee582b3` | `v0.5.5` | yes, 128 files |
| 0.5.6 (yanked) | `6a677a78` | `v0.5.6` | yes, 138 files |
| 0.5.7 | `6d9720fc` | `v0.5.7` → `956cdacc` | yes, 138 files; see note |
| 0.6.0 | `7a1fffd4` | `v0.6.0` | yes, 264 files |

## rav1d-disjoint-mut

| Version | Published from | Tag | Reproduced |
|---|---|---|---|
| 0.1.0 | `707e4cb8` | `rav1d-disjoint-mut-v0.1.0` | yes, 4 files |
| 0.1.1 | `441dad0c` | `rav1d-disjoint-mut-v0.1.1` | yes, 5 files |
| 0.2.1 | `5a8428b3` | `rav1d-disjoint-mut-v0.2.1` | yes, 8 files |
| 0.3.0 | `2ed82900` | `rav1d-disjoint-mut-v0.3.0` | yes, 8 files |
| 0.3.1 | `dd60e0a6` | `rav1d-disjoint-mut-v0.3.1` | yes, 12 files |
| 0.3.2 | `7a1fffd4` | `rav1d-disjoint-mut-v0.3.2` | yes, 18 files |

`rav1d-safe` owns the bare `vX.Y.Z` tag namespace; workspace members are
prefixed with the crate name. Two commits each carry two releases: `441dad0c`
published rav1d-safe 0.1.0 alongside disjoint-mut 0.1.1, and `2ed82900`
published rav1d-safe 0.3.0 alongside disjoint-mut 0.3.0.

## Tags added retroactively

Six releases predated the tagging habit and had no tag until 2026-09-08:
rav1d-safe `v0.1.0` and `v0.3.0`, and disjoint-mut `v0.1.0`, `v0.1.1`, `v0.2.1`
and `v0.3.0`. Each was placed at the commit recorded in its published archive,
after that commit was confirmed to reproduce the archive's source exactly. All
six commits are ancestors of `main`.

## Notes on the two imperfect records

**0.3.1 was published from a dirty worktree.** Its `.cargo_vcs_info.json` sets
`dirty: true` and the archive carries `wasmtime-guest-profile.json`, an
untracked profiling artifact that was never committed. All 118 tracked source
files match `290cb679`; the extra file is not source and does not affect what a
consumer compiles. 0.5.5 is also flagged dirty, but the dirty file was excluded
from the package, so it reproduces exactly.

**The `v0.5.7` tag points one commit before the publish.** The tag is on
`956cdacc` ("release: rav1d-safe 0.5.7"); the publish ran 34 seconds later from
its direct child `6d9720fc` ("chore: update Cargo.lock"). The two differ only in
`Cargo.lock`, and since Cargo regenerates that file when packaging, both commits
reproduce all 138 packaged source files byte-for-byte. The tag is left where it
is: retagging a published release would rewrite a reference consumers may
already hold, in exchange for no difference in shipped source. Prefer recording
the discrepancy over moving the tag.

The general lesson is that a release commit and the commit a publish actually
ran from can drift apart when a lockfile update lands between them. Tag after
the last commit that touches the release, not at the version bump.
