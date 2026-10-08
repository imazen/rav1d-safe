# Current dependency MC comparison

Missing: full candidate decoder gates and current-revision A/A and A/B in
tracked and untracked modes. The candidate is local and is not a production
landing. Zen 5 was not measured.

Both before and after binaries use Rust 1.99.0, archmage 0.9.30 at `e2dbab6`,
release fat LTO, and unset target-CPU flags. The baseline is `2b96f04d`;
the independently reviewed MC candidate snapshot is
`f910b2f254630aed2893329daf1a90cb6678b940`. Cargo.lock SHA256 is identical,
and the compiled dependency/version sets in both logs match.

The focused scalar-oracle, endpoint/tail and linear-arithmetic gates pass
3/3 in 26.835 seconds. They test dispatch and direct AVX2/AVX-512 outputs
against scalar, whole buffers including padding, every filter and phase,
all AV1 inter sizes, and odd/vector-tail fixtures through 128x128.
The arithmetic bounds cover every byte input independently of the sampled
pixel patterns: all adjacent coefficient pairs and every subset fit i16,
including the largest rounding addition.

[Baseline build provenance](../benchmarks/mc_current_baseline_build_2026-10-08.meta.json)
and [candidate build provenance](../benchmarks/mc_current_candidate_build_2026-10-08.meta.json)
record source and executable hashes, commands, flags, TLS sections and raw logs.
The baseline wrapped build peaks at 1.29 GiB RSS; the candidate build and
focused oracle peak at 1.63 GiB RSS. These are resource observations, not
throughput measurements. Historical Rust 1.98.1 / archmage 0.9.29 samples stay
separate from the current comparison.
