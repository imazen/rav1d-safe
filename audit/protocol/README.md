# Soundness gate evidence

Source: rav1d-safe `26acb2ba` plus local audit commits/changes, prepared as
rav1d-disjoint-mut 0.4.0 and rav1d-safe 0.6.0. The
[release protocol](../../docs/RELEASE_SOUNDNESS_PROTOCOL.md) states the argument,
model assumptions, supported scope, and remaining review obligations.

All three deliberate mutations were rejected, and the original source was
restored after each run. These are expected-failure controls, not unfixed
production failures. [results.json](results.json) records original source
hashes, commands, and checked diagnostics.

| Mutation | Failing gate | Observed reason |
| --- | --- | --- |
| Retire a slot with Relaxed instead of Release | [Loom log](retire-without-release.log) | Payload causality violation: sequential reuse lacked a happens-before edge |
| Delete both in-lock wide-state rechecks | [Loom log](wide-state-outside-lock.log) | Concurrent writers to the modeled payload cell |
| Allow a single-threaded opener to store false globally | [Decoder transition log](global-threading-reset.log) | The live multithreaded decoder's latch was reset by another opener |

The mutation harness first checked the exact source pattern/count, saved the
original file, changed only the selected operation, ran the gate, and restored
the file in `finally`. To repeat manually, use an exclusive disposable copy of
the current source, make one mutation at a time, and require the stated
diagnostic; an unrelated compilation failure is not evidence. Run the restored
version afterward and require a healthy control.

[gates.json](gates.json) records the final native/no-std/Loom/decoder/corpus/
package commands, feature settings, and exit codes. Logs are retained locally
under `$HOME/tmp/rav1d-review-2026-09-05/final-gates`. The full-corpus runs each
reported **766 passes, zero failures, two skips**, at 1 and 8 decoder threads.
The two skips are `annexb.obu` and `section5.obu`: this harness reads IVF files.
The corpus revision is `61afa1ceb6029be1a8ea3f6b9c9a9672700f13bb`; this is the
reference-MD5 manifest set, not a claim that every file in the repository ran.

[Final verification](verification.json) includes the MSRV, target compilation,
semver, extra Miri, documentation and package checks, with each command and
outcome. The intentional decoder semver failure inventories permitted 0.6.0
breaks. The first storage-feature documentation build found a private rustdoc
link; that link was corrected and the strict documentation check passed.
The C-FFI marker hardening passed a `c-ffi` library build; callback/assembly
runtime soundness was not established by that compilation.

Native disjoint coverage: 108 tests with production storage features, 85 without
default features; each also passed one runnable example and four compile-fail
examples. Two pre-existing illustrative storage snippets remain ignored.
Six Loom models passed at the default bound. The new same-process transition
test passed; the committed decoder/fuzz-regression selection passed all 23
tests in both debug and release profiles.

Miri checked the six adversarial tests, eight rectangle-coordinate tests and
27 existing soundness tests under both Stacked and Tree Borrows. Additional
guard-move, aligned-storage and PicBuf checks are recorded in the final review
results. During development, an initial raw-spin Loom model exceeded its branch
bound, and one successful Tree test run had a wrapper error after an executing
shell script was edited. Neither was counted as a successful command; the
corrected model and runner were rerun.

No publication or external notification was performed. CI changes make Loom
and unsafe-feature rejection ongoing gates and include production storage
features in both Miri jobs. The complete proof still depends on the explicitly
stated lock, storage, lifetime, geometry and publication obligations; these
finite test results do not establish arbitrary-client soundness by themselves.
