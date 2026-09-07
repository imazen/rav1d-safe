# Source checkpoint: 24 sources, 12 development + 12 holdout

[PR #528](https://github.com/imazen/rav1d-safe/pull/528) now pins the membership,
source bytes, family grouping, centered 16:9 crops, and source splits in
[sources-frozen.json](sources-frozen.json). **Encoder assignments, generated
raster/bitstream hashes, and the full workload matrix are not frozen yet.**
No selected source has undergone AV1 decoding, timing, or profiling. Source
JPEG decoding and PDF rendering were used only for admission and visual review.

Each split contains four photographs, three textures, two maps, and three
text documents. The 24 source SHA-256 values and manually reviewed scene/document
families are distinct. The original exposed imazen-26 sources 1407/5017 and
related Great Smoky Mountains maps are absent. These are campaign holdouts;
this does not relabel imazen-26's canonical test/train sources.

`freeze_sources.py` assigns each predeclared content stratum by a fixed SHA-256
ordering. The first half is development, the second half holdout. Each side
receives a ground photograph, aerial photographs, detailed foliage, varied
texture, maps, and distinct document layouts. Source category corrections
follow visual review: the Apache winter image is a road/sky photograph, and
the El Salvador image supplies aerial natural texture. All these decisions
precede AV1 measurement.

| Source ID | Split | Content | Original geometry/page |
|---|---|---|---|
| map-choh | development | map | PDF p1 |
| map-mora | development | map | PDF p1 |
| photo-brazil-rivers | development | photograph | 8256×5504 |
| photo-kinshasa | development | photograph | 8256×5504 |
| photo-orenburg-snow | development | photograph | 8256×5504 |
| texture-apache-winter | development | photograph | 8256×5504 |
| text-census-income | development | text | PDF p9 |
| text-nist-sha | development | text | PDF p15 |
| text-noaa-ian | development | text | PDF p3 |
| photo-el-salvador | development | texture | 8256×5504 |
| texture-coleus | development | texture | 8256×5504 |
| texture-wood-wall | development | texture | 8256×5208 |
| map-muwo | holdout | map | PDF p1 |
| map-shen | holdout | map | PDF p1 |
| photo-emi-koussi | holdout | photograph | 8256×5504 |
| photo-london-tram | holdout | photograph | 7952×5304 |
| photo-naples | holdout | photograph | 8256×5504 |
| photo-new-orleans | holdout | photograph | 8256×5504 |
| text-cdc-mmwr | holdout | text | PDF p2 |
| text-irs-form | holdout | text | PDF p1 |
| text-nasa-api | holdout | text | PDF p3 |
| texture-armeria | holdout | texture | 7952×5304 |
| texture-bergen-forest | holdout | texture | 8256×5504 |
| texture-galium | holdout | texture | 7952×5304 |

## Native detail and validation

All **fourteen JPEG sources** pass complete pixel decoding with system Pillow,
expected dimensions, SHA-256 verification, and the native UHD geometry rule.
Wikimedia downloads also match the API's full-file SHA-1 and byte count. Camera
make/model and orientation are recorded. Images are cropped/downsampled;
smaller raster sources are never enlarged to stand in for 8K detail.

The other **ten sources are PDF vector renderings**, not camera photographs.
Seven selected pages have no embedded raster objects. The Shenandoah, Muir
Woods, and NOAA pages contain small raster logos outside their centered 16:9
crop. [pdf-crop-audit.json](pdf-crop-audit.json) records the page/crop coordinates,
Cairo SVG hashes, every image-use bounding box (including masks), and separation
checks. The script rejects unsupported ancestor transforms; this is a narrow,
source-specific audit, not a general PDF verifier. Actual rasterization and
output hashes remain a next step. Portrait/odd-dimension edge crops still need
their own declared source/geometry checks.

Source material is obtained from the [NASA astronaut-photo catalog](https://eol.jsc.nasa.gov/),
[Wikimedia Commons](https://commons.wikimedia.org/), and the exact government
PDF URLs in the manifest. NASA's library [API documentation](https://images.nasa.gov/docs/images.nasa.gov_api_docs.pdf)
also supplies a text source. Commons author/license metadata and description
links are preserved; large original image bytes are not redistributed in Git.
These 8-bit source rasters make no HDR claim. Promoting samples to 10-bit later
will exercise that decoder path without creating genuine HDR source coverage.

## Rejected sources and acquisition correction

Candidate plans v1–v5 preserve the admission sequence. The old Gates of the
Arctic PDF URL now serves an HTML redirect page with HTTP 200. PDF signature
validation rejects it. Grand Canyon `GRCAmap2.pdf` embeds the whole map as a
single 1826×1429 JPEG and is excluded from native 8K admission. Other map
candidates contain raster relief/backgrounds, flattened image strips, or
unhelpful page layouts; they remain acquired experiments, not workload inputs.
Redwood page 1 was visitor text, so page 2 was inspected and then excluded with
the other mixed raster/vector maps. No rejection depends on decoder speed.

The first acquisition script expected `ffprobe`, which is absent on this host.
It retained complete, hashed JPEG downloads but reported their geometry step
as failed. Independent `validate_sources.py`, using `/usr/bin/python3` and
system Pillow, then decoded and verified those pixels. The original failure
logs and script hash remain intact. The current downloader uses system Pillow;
the final two JPEG acquisitions and all subsequent validation pass.

## Reproduction and storage

Use `fetch_candidates.py` with the recorded candidate plan and a fresh output
directory, then `validate_sources.py` with that acquisition directory. Both
run with `/usr/bin/python3` through `run-heavy --mem 16G --jobs 8`, sequentially.
The first download batch took 293 seconds with peak RSS 0.03 GiB; its independent
23/24 source validation took eight seconds with peak RSS 0.56 GiB. These are
preparation-resource measurements, not decoder latency or memory results.

For the three logo audits, regenerate the SVG using the pinned source PDF:

```sh
pdftocairo -f PAGE -l PAGE -svg source.pdf validation-dir/SOURCE-ID.svg
```

Use page 1 for `map-shen`/`map-muwo`, page 3 for `text-noaa-ian`.
Then run `freeze_sources.py` with the campaign work directory and fresh output
manifest/audit paths. Source hashes, split assignments, and crop checks must
match; the timestamp naturally differs. No encoder or decoder is invoked.

Large originals, previews, SVGs, full rejected-map raster-object listings, and
acquisition snapshots remain under
`/home/lilith/tmp/rav1d-still-expanded-2026-09-07`. Small provenance, acquisition
logs, validation summaries, and rejection evidence are compressed/indexed in
`evidence`. The pointer index includes hashes for larger local records.
There is **no verified off-host backup**. Preserve these assets when resuming.

The next steps are to freeze the encoder/quality/layout/edge assignments,
render and hash the selected crops, encode the development and reserved
holdout inputs, and establish the expanded development baseline. Holdout
performance remains sealed until candidate acceptance. Full concurrency,
lifecycle, memory, video, and parity gates remain outstanding under
[PERFORMANCE_PARITY_GOAL.md](../../docs/PERFORMANCE_PARITY_GOAL.md).
