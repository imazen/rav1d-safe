# Expanded still corpus preparation — 2026-09-07

The initial admission audit below is now followed by a [24-source checkpoint](SOURCE_FREEZE.md):
**12 development and 12 holdout source memberships/splits are pinned**. The
encoder assignments, bitstreams, and full workload matrix are not frozen yet.
No newly selected source has been decoded for AV1 performance. The historical
registry/mirror findings below explain why additional sources were acquired.

## Canonical registry and exposure correction

The imazen-26 registry is pinned to
[`187fbf338ce08e8e6654db7f04ddae58d5263da2`](https://github.com/imazen/imazen-26/tree/187fbf338ce08e8e6654db7f04ddae58d5263da2).
The root membership manifest, source documentation, canonical train/test
manifests, and family map were fetched and hashed. `registry_audit.py` verifies
those metadata hashes and emits [registry-audit.json](registry-audit.json).

The earlier still baseline incorrectly called sources **1407** (coastline)
and **5017** (trail map) training images. Both belong to the canonical **test**
split. Their source paths and SHA-256 values match the pinned test manifest;
the benchmark did not silently substitute different pixels. They are already
exposed investigation sources in this campaign and cannot count as untouched
holdouts. The audit records their IDs and all linked family members for
exclusion from future holdout selection. The original byte-count metadata
for 1407 predates the metadata rewrite; its actual canonical byte count is
10,246,694 and its existing source SHA remains correct.

The metadata lists only **17 files** with width≥7680 and height≥4320, or the
portrait equivalent. These include six nature images, five textures, four
renders, and two related NPS maps. This is a geometry screen, not confirmation
of original native detail, unique source families, suitable content, or
successful downloads. It cannot supply the required 24 distinct sources.
Additional native large photographs and text/map sources are necessary; the
resolution requirement will not be met by enlarging smaller raster images.

Registry snapshots remain in
`/home/lilith/tmp/rav1d-still-expanded-2026-09-07/registry`. The committed audit
contains the revision, exact acquisition commands, source-file hashes,
geometry candidates, and exposure exclusions. There is no verified off-host
backup of local working assets.

## A dataset name is not resolution evidence

DIV8K is a possible source family, but its name alone does not establish that
an individual file can provide a native UHD crop. The
[ETH publication record](https://www.research-collection.ethz.ch/items/9535ac6e-9215-4bdc-9dc7-0c717247aaa0)
describes resolutions *up to* 8K. An old ETH training-archive URL returned
HTTP 404 in this investigation.

The public [Iceclear mirror](https://huggingface.co/datasets/Iceclear/DIV8K_TrainingSet/tree/2e4255369a8bb0c1e2f0c281d4b3bd051743ff28)
is pinned to `2e4255369a8bb0c1e2f0c281d4b3bd051743ff28`. Its archive is
46,340,249,930 bytes; the declared LFS SHA-256 is
`9a4cec7078eb3ea195e49d9d69e6752c8bacb950dd5caa82105aeafa175985c7`.
The entire archive was **not** downloaded or hash-verified.

`zip_inventory.py` uses checked HTTP byte ranges to read the ZIP directory
and PNG headers, with a two-megabyte limit per request. The directory contains
1,500 PNG members. A predeclared SHA-256 ordering of member names selected
64 headers; their PNG IHDR CRCs all pass. **None of those 64** meets the native
UHD landscape/portrait geometry rule. Their observed dimensions reach 6,720
on the longer axis. They are not admitted as 8K sources, and no inference is
made that every other member has the same limitation.

The probe transferred 421,525 bytes plus a previously fetched 65,536-byte
tail. No image pixels were decoded, no benchmark result was inspected, and
no final source selection was frozen. Range blocks, directory records,
headers, and request/hash provenance remain in the local `div8k-inventory`
directory; small summary/header evidence accompanies this record. A mirror
label or advertised archive checksum is not treated as proof of unseen bytes.

## Next admission steps

Find additional sources with verifiable native geometry and content coverage;
check source/family overlap with all exposed inputs. Pin complete selected
file hashes, document crops or vector rasterization, assign the development
and holdout roles before decoding for performance, and freeze encoder,
quality, bit-depth/layout, tile, thread, lifecycle, and edge-case assignments.
Keep holdout timing and profiling out of optimization. The full requirements
remain in [PERFORMANCE_PARITY_GOAL.md](../../docs/PERFORMANCE_PARITY_GOAL.md).
