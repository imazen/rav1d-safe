# Film-grain frame and tile concurrency

Verified on Apple ARM on 2026-09-06, test commit `6115e06b`, based on
`e73811f5`. This extends the regression coverage for
[issue #526](https://github.com/imazen/rav1d-safe/issues/526); its ARM row-borrow
and worker-panic cleanup fixes were already on `main`.

Run `just test-filmgrain-concurrency`. It runs the selected tests twice in the
dev profile, first checked and then with `unchecked`, with one nextest test at
a time. The tests themselves create the worker/decoder concurrency.

| Configuration | Input and checks | Result |
|---|---|---|
| Checked tile-only | 13 film-grain vectors, 1/2/4/8 workers, explicit frame delay 1; reference MD5s | Pass |
| Unchecked tile-only | Same corpus and thread grid | Pass |
| Unchecked frame pipeline | 13 vectors, including 2 multi-frame sequences; (workers, frame contexts) = (4,2), (8,2), (8,4), three repetitions per cell | 117 runs pass; exact reference MD5 and serial frame count |
| Independent decoders | Three simultaneous four-worker decoders, all 13 vectors; checked uses delay 1, unchecked mixes delays 1 and 2 | Pass |
| Multiple tiles and frame contexts | Committed 1024x1024 stream with 4x8 tiles, repeated 12 times; eight workers with 1/2/4 frame contexts | All output frame hashes match serial; actual context and worker counts asserted |

The final frame-pipeline run reported 165 packet submissions whose output was
deferred. The test also requires multi-frame input, so a corpus made entirely
of synchronous stills cannot satisfy its liveness assertions. The 32-tile
fixture independently asserts that its decoded frame header has multiple tiles.
These establish the exercised configurations and asynchronous output behavior;
no scheduler trace or measurement of peak simultaneously executing tasks was
collected.

The complete selected matrix passed five tests in the checked build and six in
the unchecked build. Clippy for the library and film-grain integration target
passed in both configurations. Filtering excludes unrelated library tests;
none of the selected tests is ignored.

## Input backpressure

`rav1d_send_data` returns `EAGAIN` before consuming new input when a prior packet
still has pending bytes. `Decoder::decode` exposes that as `NeedMoreData`.
The test caller must retain the rejected packet, call `get_frame`, account for
any returned frame, and retry the same packet. `get_frame` can return `None`
after moving pending input into a frame context; that does not mean the new
packet was consumed. The test uses a deadline to catch a stalled retry loop.
It drains with `flush` only at end of input and verifies frame counts and hashes.

## Limits

The checked build still forces one frame context per decoder in
`src/lib.rs::get_num_threads`. Multiple frame contexts require `unchecked`;
this PR does not enable checked frame threading or change public APIs. The
film-grain corpus covers 8/10-bit streams and all three chroma layouts. The
separate 32-tile fixture covers repeated keyframes; the grain corpus supplies
the multi-frame sequences. Throughput was not measured, and the optional
assembly feature was not part of this matrix.
