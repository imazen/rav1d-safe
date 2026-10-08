# Archmage version-only performance comparison

Missing: A/A controls and A/B timings. No performance effect is established.

On 2026-10-08, the live GitHub latest release and main revision and the
package index resolve to archmage 0.9.30 / e2dbab66. The comparison uses the
previous 0.9.29 pin b8d6c077 and the current pin on identical decoder source
at 8ea8bf18. All 258 tracked Rust files match. Only archmage and
archmage-macros entries differ in the lockfiles; other package versions and
sources match. Both modes use Rust 1.99, release fat LTO, empty RUSTFLAGS
and runtime dispatch. Both tracked binaries have 11,368 bytes of TLS.

[Build logs, binary hashes and source inventory](../benchmarks/archmage_matched_builds_2026-10-08.meta.json)
record successful fresh builds. Each build peaked at 1.29 GiB RSS; the old
scope took 64 seconds and the new scope took 63 seconds. These are build
resource observations, not decoder performance results.

Before timing, all 32 whole-clip comparisons match dav1d 1.5.3: four clips,
two pins, tracked/untracked modes, and one/four workers at frame delay one.
The clips include 8-bit 1080p and 4K footage and 10-bit 1080p footage. The
output check validates nonzero frame counts and rejects decode/flush errors.
It completed in 121 seconds with peak RSS 0.20 GiB, minimum available
25,795 MiB and peak load 1.19.

`just bench-md5` accepts explicit binaries and streams. `just
bench-paired-modes` runs sequential A/A controls and interleaved A/B tests
for both modes, with exclusive result-file creation. Each phase retains
all timing observations and executable/stream hashes. Compare minimums
and medians against the A/A control before attributing a change to archmage.
Results measured here apply to Zen 4; Zen 5 is not measured.

Resolve release state with the GitHub latest-release and main-commit API
endpoints and the sparse package index rather than a cached release page:
[GitHub releases](https://github.com/imazen/archmage/releases),
[main](https://github.com/imazen/archmage/tree/main),
[package index](https://index.crates.io/ar/ch/archmage).
