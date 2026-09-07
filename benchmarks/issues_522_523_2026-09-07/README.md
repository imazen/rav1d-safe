# Differential fuzz repros #522 and #523

Both [#522](https://github.com/imazen/rav1d-safe/issues/522) and
[#523](https://github.com/imazen/rav1d-safe/issues/523) name the same artifact.
Replayed on Apple M4 Pro, macOS, current decoder source at `44b29236`,
with nightly AddressSanitizer and dav1d 1.5.4. The target still panics with
`dav1d decoded a frame but rav1d-safe errored: InvalidData`.
This is a policy mismatch on malformed input, not an ARM SIMD pixel mismatch.
The differential assertion remains unchanged and therefore still reproduces.

## Artifact

- Name: `crash-16c021d73995521b6baafff5d542fc040325b6d4`
- Size: 36 bytes; SHA-256 `54618c389e1f7bda7de0eacd5b3938fe589bb6511a76d35fd9e71a29264bd40f`.
- Verified R2 prefix: `s3://zenfuzz/crashes/rav1d-safe/differential_dav1d/arm64/389bcebb78213983/`.
  The issue bodies' `b50550d53dccc872/` prefix has no objects.
- Farm mirror: `/mnt/v/fuzzes/_farm/crashes/rav1d-safe/differential_dav1d/arm64/389bcebb78213983/`.
- Local archive: `/Users/lilith/work/codec-artifacts/rav1d-issues-522-523/`,
  including the original `meta.json` and `repro.txt`. No NAS mirror was made.
- The exact bytes are retained as a Rust array in
  `tests/strictness.rs::fuzz_522_523_out_of_range_segment_id_is_rejected`.

## Diagnosis

Temporary diagnostics reported `segment_id=6 LastActiveSegId=4` at the
pre-skip segment check in `src/decode.rs`.
[AV1 read-segment-ID semantics](https://github.com/AOMediaCodec/av1-spec/blob/master/07.bitstream.semantics.md#read-segment-id-semantics)
requires the postprocessed ID to be between zero and LastActiveSegId.
`Strictness::Strict` and default settings reject it. `Lenient` decodes a
243x173, 10-bit monochrome frame. dav1d 1.5.4 with strict compliance enabled
also decodes that frame; its strict policy does not include this segment bound.
The same raw payload wrapped in a single-frame IVF is rejected by libaom
3.14.1 with `Bitstream not supported by this decoder`; that message alone
does not identify libaom's rejection site.

The new strictness regression asserts default/strict rejection and a lenient
positive control. Removing both segment bound checks makes the new test fail.
All temporary production diagnostics and mutations were removed. No decoder
policy, pixel expectation or differential fuzz assertion was relaxed.

The remaining harness question is how to adjudicate differing conformance
policies. Simply treating every dav1d-accepted frame as conforming is not
valid for this artifact. A third reference or a precisely classified rejection
would be needed to change that contract without hiding other divergences.

## Commands and logs

```sh
cargo +nightly fuzz run differential_dav1d --features differential <artifact> -- -runs=1
cargo run --example strictness_sweep -- --jobs 1 <artifact>
cargo test --test strictness
```

Commands ran with `nice -n19`, four Cargo jobs and scratch under `~/tmp`.
[ASan reproduction](repro-current.log), [temporary segment diagnostic](trace-segment.log),
[dav1d IVF decode](dav1d-ivf.log), [libaom IVF rejection](aom-ivf.log),
[regression mutation](mutation.log), [six strictness tests](strictness-six.log).
