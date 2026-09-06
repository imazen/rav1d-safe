# Upstream rav1d bugfix triage

Reviewed against `memorysafety/rav1d` main
`d3d1cd67059f47803919be8276650e5870c9fd02`, and rav1d-safe `26acb2ba` plus local
audit changes. These histories have no common ancestor after the fork's history
rewrite. A raw `origin/main..upstream/main` list includes thousands of old commits
and is not a list of missing fixes. Compare behavior and relevant hunks.

| Upstream change | Local inspection | Disposition |
| --- | --- | --- |
| [T.35 payload underflow, 9d78516f](https://github.com/memorysafety/rav1d/commit/9d78516f3232fe8b9c2bfe9d9a7506163033aae6) | `src/obu.rs` uses checked subtraction for the trailing byte, country code, and extension, plus a payload size limit | Equivalent protection already present; no duplicate port |
| [Remove non-thread-safe `Rc` from `CRef`, b3884f74](https://github.com/memorysafety/rav1d/commit/b3884f746997d24caa5ab0c2376adb7290cc5ca6) and [trait assertions, 580e55cc](https://github.com/memorysafety/rav1d/commit/580e55ccf42c1f14ef07478c3fed33341b900fff) | This fork's `CBox` has owned `Box` and optional C storage; default `CArc` uses `Arc<Box<T>>`. There is no `CRef::Rc` variant to remove | Exact upstream defect is absent; thread-trait assertions remain useful for future storage changes |
| [Fallible chroma location, 921942d6](https://github.com/memorysafety/rav1d/commit/921942d60c70a58a602e2746d7a2bc8823913c42) | The changed upstream `rust_api::Picture::chroma_location` accessor is not the fork's managed API; no matching panicking accessor exists there | No direct port; preserve fallibility if adding this accessor |
| [Optional refidx construction, 9f00b03b](https://github.com/memorysafety/rav1d/commit/9f00b03bc58d94f14e157b0d365a26cf8d3b1b77) | `Some(...).filter(...)` becomes `bool::then_some(...)` | Equivalent refactor, not a decoder bugfix |

The C-FFI-only private `StableRef<T>` had an inconsistent `Sync` bound/comment
(`T: Send` in code, `T: Sync` in the explanation). Its owning `CArc` also contains
`Arc<Pin<CBox<T>>>`, which independently requires `T: Send + Sync` for sharing,
so this inspection does **not** establish a safe-client exploit through `CArc`.
The local hardening now gives both marker traits the `T: Sync` bound appropriate
for shared reference access; the owning Arc still supplies the destruction
requirements. The default safe-storage path does not compile this marker.
A complete C callback/assembly audit is
separate from the checked-disjoint protocol established here.

These inspected candidates yielded no missing default-path bugfix to port.
This is bounded triage of the recent candidates, not a proof that every fix in
the diverged histories is present. Preserve a regression vector and compare
the intended before/after behavior for any future port.
