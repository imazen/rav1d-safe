# External benchmark evidence

The 948 compressed evidence files added during PR #528 are stored in a public,
content-addressed R2 bundle. Reports, experiment code, manifests, hashes, and
reproduction scripts remain in Git. No decoder code changes as part of this
storage migration. The unchecked regression documented in the
[wrapper report](wrapper-contexts-2026-09-07/README.md) remains open.

From the repository root, restore all evidence to its original relative paths:

```sh
python3 tools/fetch-benchmark-artifacts.py
```

The download needs no credentials. The tool verifies the bundle SHA256, the
index SHA256, and the byte count and SHA256 of every artifact before restoring
files. It refuses to overwrite different existing contents. Restored `.gz`
files in the five migrated experiment directories are ignored by Git.

For verification without writes, add `--check-only`. For an external checkout
location, use `--destination /path/to/evidence`. To use a previously downloaded
bundle, use `--archive /path/to/evidence.tar.gz`. The original experiment
manifests still describe the same bytes, including their decompressed hashes.

[Download the bundle](https://codec-corpus.r2.imazen.org/benchmarks/rav1d-safe/pr-528/sha256/e786e462070305a051131101dec14fd84efeeca0f286dfd891f8cb3bb4d18479.tar.gz)
(1,190,015 bytes). The exact URL, bundle/index hashes, source commit, file count,
and aggregate artifact size are pinned in
[EXTERNAL_ARTIFACTS.json](EXTERNAL_ARTIFACTS.json).

The tar contains `ARTIFACT_INDEX.json` with all 948 paths, byte counts and
SHA256s, followed by the files at their original `benchmarks/...` paths.
The R2 bundle was publicly downloaded and all 948 member hashes verified
before the tracked copies were removed. An independent extraction reproduced
every original byte. Corrupt-download and conflicting-destination checks
verify that the restoration tool fails without writing artifacts.

This trims the PR's final tree. A squash merge is needed to keep the earlier
artifact-adding commits out of the destination branch's history; simply deleting
files does not erase blobs from those earlier commits.
