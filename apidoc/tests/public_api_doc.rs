//! Public-API surface snapshots for the PARENT workspace (docs/public-api/).
//! Shared implementation + format docs: the `zenutils-apidoc` crate.
//!
//! The explicit list mirrors the pre-runner snapshot test: the two published
//! crates. Note: rav1d-safe's features section enables `asm`, so regenerating
//! requires NASM on PATH (same as any `--features asm` build).
#[test]
fn public_api_surface_docs_are_current() {
    zenutils_apidoc::ApiDoc::new()
        .workspace_dir("..")
        .crates(["rav1d-safe", "rav1d-disjoint-mut"])
        .run();
}
