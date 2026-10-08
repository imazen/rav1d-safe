# Recording benchmark artifacts

Run the benchmark through the resource wrapper and preserve full stdout and
stderr. Require every timing phase to complete before reporting a comparison;
`just bench-paired-report <directory>` validates the observations and controls.

`just record-benchmark-artifacts --output <results-directory> --metadata
<reviewed-provenance.json> --input timing.log=<raw-log>` preserves the exact
UTF-8 input bytes. Repeat `--input` for phase JSONL and computed analysis.
Supply compiler, source and executable hashes, commands, frame counts and the
wrapper's measured resource line in the reviewed provenance JSON.

The recorder limits each raw part to 28,000 bytes without splitting UTF-8
characters. `artifacts.meta.json` contains whole-input and per-part hashes,
ordered part names and the recording command. Concatenating the named parts
must reproduce the original input; the command verifies this after writing.
Metadata must fit within 30,000 bytes. Binary input, duplicate output names
and existing destinations are rejected before any artifact is written.

The [native validation](../benchmarks/artifact_recorder_2026-10-08.json)
checks exact reconstruction across a UTF-8 boundary, bounded part sizes,
collision refusal without modifying existing files and binary-input refusal.
This is artifact-integrity validation, not decoder or throughput evidence.
