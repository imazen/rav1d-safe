# Incremental managed packet decoding

`Packet` owns a nonempty compressed buffer. Submit complete OBUs, normally one
temporal unit per packet; arbitrary network read fragments need assembling by
the demuxer first. The decoder does not require the whole file.

```rust
use rav1d_safe::{Decoder, Packet, ReceiveStatus, SendStatus};

fn decode_packets(chunks: Vec<Vec<u8>>) -> rav1d_safe::Result<usize> {
    let mut decoder = Decoder::new()?;
    let mut pictures = 0;
    for (index, bytes) in chunks.into_iter().enumerate() {
        let mut packet = Packet::new(bytes)?
            .with_timestamp(index as i64)
            .with_duration(1);
        loop {
            match decoder.send_packet(&mut packet)? {
                SendStatus::Accepted => break,
                SendStatus::ReceivePending => {
                    while let ReceiveStatus::Frame(frame) = decoder.receive()? {
                        pictures += 1;
                        // `frame` owns its pixels; it may be retained or mapped.
                        let _pts = frame.timestamp();
                    }
                }
                _ => unreachable!("unrecognized submit state"),
            }
        }
        while let ReceiveStatus::Frame(_) = decoder.receive()? {
            pictures += 1;
        }
    }
    decoder.end_input();
    loop {
        match decoder.receive()? {
            ReceiveStatus::Frame(_) => pictures += 1,
            ReceiveStatus::EndOfStream => break,
            _ => unreachable!("input has ended"),
        }
    }
    Ok(pictures)
}
```

Successful submission empties the caller's packet. Backpressure preserves it
for retry without copying. A submission error also retains it, but partial
stream processing may have happened: reset before trying to start again.
Do not interpret an error as backpressure or as successful consumption.

Timestamp, duration and offset are uninterpreted caller metadata, carried to
output pictures. Timestamp defaults to `i64::MIN` (unknown), duration to zero,
and offset to -1. Negative timestamps are valid except for the unknown sentinel.
Time bases and checked rescaling belong to the container/media adapter.
Every presentation produced by one packet inherits its metadata, so combine
temporal units only if this association is intentional. A FIFO guess based on
one input packet equaling one output frame is not the protocol.

`Packet::with_context(Arc<dyn Any + Send + Sync + RefUnwindSafe>)` carries additional owned,
opaque presentation context. `Frame::context()` returns a borrowed value; use
`downcast_ref` to recover the adapter's type. An adapter can snapshot its clock,
configuration epoch and container color at acceptance, then use that snapshot
when delayed output arrives. It must not consult the latest track configuration
for an older accepted packet. Caller ordinals need not be unique map keys.

Native input, reference pictures and output frames share the context. Several
presentations from one packet retain it independently; show-existing output
receives the later presentation packet's context. Frame clones and film-grain
picture copies retain it too. A retry keeps the caller's packet and context.
The final owner releases the value. This does not acknowledge completion after
one frame or guarantee release at end-of-input: the decoder may still retain
reference pictures. Avoid a reference cycle back to the decoder. Context is
not parsed metadata and is not included in decoder diagnostics.
The `RefUnwindSafe` bound preserves the pre-existing unwind-safety traits of
`Packet`; immutable configuration snapshots satisfy it.

`end_input` declares EOF without allocating a result vector. `receive` returns
one picture at a time and waits for delayed frame-threaded output before EOS.
It distinguishes `NeedInput` from terminal `EndOfStream`; repeated EOS is stable.
Input after `end_input` is rejected until `reset`. `reset` discards pending input,
output and reference state, retaining settings and the cancellation token.
Existing frame owners and mappings remain valid. `flush` remains the allocating
convenience operation: drain all output, then reset (also resetting on error).

`Frame::planes()` provides native U8 or right-aligned 10/12-bit U16 storage.
`PlaneView16::stride()` is measured in U16 elements; multiply by two when
adapting to a byte-stride interface. A mapped plane owns a borrow guard:
its slice borrows the mapping, not merely the frame owner. The zenmedia adapter
keeps guards in a mapping object and creates checked views from that object.

`raw_color_info` retains primaries, transfer, matrix, range and raw chroma
position. AV1 position 0 is unknown, 1 vertical, 2 colocated, and 3 reserved.
Do not replace unknown with centered, or infer RGB transfer from its YCbCr
matrix. `render_size` reports intended dimensions without resizing the pixels.

AV1 show-existing-frame output retains the original coded picture header, while
its packet timestamp and offset describe the later presentation packet. A coded
picture's frame type therefore cannot by itself identify the random-access
properties of that later packet. Seek indexing needs packet-level evidence.

`tests/managed_packets.rs` exercises exact synthetic pixels, backpressure and
retry, timing, reset, end-of-input without an earlier receive, decoder drop,
and keeping a mapped frame alive while decoding later inter frames. Run it in
both checked mode and with `untracked` to exercise actual frame threading:

```sh
cargo nextest run --release --no-default-features --features bitdepth_8,bitdepth_16 --test managed_packets
cargo nextest run --release --no-default-features --features bitdepth_8,bitdepth_16,untracked --test managed_packets
```

## Presentation provenance

`Frame::is_show_existing` describes this presentation, independently of the
older coded header retained with a reference picture. `is_keyframe` is true
only for a newly coded, visible key frame. A hidden key frame or a replay of an
older key frame is not a new visible keyframe. These flags survive output
queueing, frame threading, cloning, and the picture copy used for grain.

This is not a complete random-access promise. A demuxer still needs packet
boundaries, the applicable sequence header and container initialization data.
`input_offset` identifies the presentation packet, not the original coded
reference. The C picture ABI is unchanged and does not expose these Rust flags.

The native reordered fixture in `tests/media_vectors/` records its clean encoder
revision, source formula and hashes in an adjacent JSON manifest.

## Packet-context qualification (2026-10-25)

`just test-packet-context` adds native checks for all presentations from one
packet, rejected/backpressured ownership, queued-output reset, retained frame
clones, film-grain picture copies and reordered/show-existing provenance.
The existing authored sample oracle remains active. The checked-mode run
passed with one and four configured workers; the two native test binaries
passed every selected case. Resource-wrapper receipt: rc=0, 32 s, peak RSS
1.10 GiB, minimum available 15284 MiB, peak load 1.49. This measures the build
and test process, not decoder workspace or context overhead.

Final validation adds native parse-error and cancellation cleanup and a
compile-time check that `Packet` keeps its original Send/Sync/unwind traits.
`just qualify-packet-context` passes the native cases in checked and untracked
modes, strict library/integration Clippy, C-FFI and WASM builds, ten active
doctests (the thirteen existing ignored examples are unchanged), and generated
public API snapshots. The borrowed context accessor performs no Arc clone.
The final command ran with a ten-minute deadline and completed normally:
rc=0, 59 s, peak RSS 1.10 GiB, minimum available 15337 MiB, peak load 1.65.
Decoder workspace and performance overhead remain unmeasured here.
