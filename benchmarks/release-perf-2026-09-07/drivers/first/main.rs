//! Strict persistent-decoder concurrency profiler. See audit/concurrency-profile/README.md.
//! profile_concurrency INPUT THREADS INSTANCES PASSES [REPS]
//! Baseline timings exclude hashes and pool creation. Validation before AND after
//! the timed section hashes every visible output plane against a serial reference.
use rav1d_safe::src::managed::{Decoder, Frame, Planes, Settings};
use std::hint::black_box;
use std::sync::{Arc, Barrier};
use std::time::Instant;

fn frame_md5(frame: &Frame) -> String {
    let mut ctx = md5::Context::new();
    match frame.planes() {
        Planes::Depth8(planes) => {
            for row in planes.y().rows() {
                ctx.consume(row);
            }
            if let Some(u) = planes.u() {
                for row in u.rows() {
                    ctx.consume(row);
                }
            }
            if let Some(v) = planes.v() {
                for row in v.rows() {
                    ctx.consume(row);
                }
            }
        }
        Planes::Depth16(planes) => {
            for row in planes.y().rows() {
                for &px in row {
                    ctx.consume(px.to_le_bytes());
                }
            }
            if let Some(u) = planes.u() {
                for row in u.rows() {
                    for &px in row {
                        ctx.consume(px.to_le_bytes());
                    }
                }
            }
            if let Some(v) = planes.v() {
                for row in v.rows() {
                    for &px in row {
                        ctx.consume(px.to_le_bytes());
                    }
                }
            }
        }
    }
    format!("{:x}", ctx.finalize())
}

// Strict packet parsing: reject partial/truncated frame headers, never silently
// shorten a workload. The IVF header's frame count is advisory; count packets.
fn packets(path: &str) -> Vec<Vec<u8>> {
    let b = std::fs::read(path).expect("read input");
    if !b.starts_with(b"DKIF") {
        assert!(!b.is_empty());
        return vec![b];
    }
    assert!(b.len() >= 32 && &b[8..12] == b"AV01", "AV1 IVF header");
    assert_eq!(u16::from_le_bytes(b[4..6].try_into().unwrap()), 0);
    let mut pos = u16::from_le_bytes(b[6..8].try_into().unwrap()) as usize;
    assert!(pos >= 32 && pos <= b.len());
    let mut out = Vec::new();
    while pos < b.len() {
        assert!(b.len() - pos >= 12, "partial IVF packet header");
        let n = u32::from_le_bytes(b[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 12;
        assert!(n > 0 && n <= b.len() - pos, "truncated or empty IVF packet");
        out.push(b[pos..pos + n].to_vec());
        pos += n;
    }
    assert!(!out.is_empty());
    out
}

fn decoder(threads: u32) -> Decoder {
    let mut s = Settings::default();
    s.threads = threads;
    s.strict_std_compliance = true;
    s.frame_size_limit = 120_000_000;
    Decoder::with_settings(s).expect("create decoder")
}

fn run(dec: &mut Decoder, packets: &[Vec<u8>], mut frame: impl FnMut(Frame)) -> usize {
    let mut n = 0;
    for packet in packets {
        if let Some(f) = dec.decode(black_box(packet)).expect("decode packet") {
            frame(f);
            n += 1;
        }
        while let Some(f) = dec.get_frame().expect("drain packet") {
            frame(f);
            n += 1;
        }
    }
    for f in dec.flush().expect("drain and reset stream") {
        frame(f);
        n += 1;
    }
    assert!(n > 0, "no displayed frames");
    n
}

fn validate(dec: &mut Decoder, packets: &[Vec<u8>], reference: &[String]) {
    let mut got = Vec::new();
    run(dec, packets, |f| got.push(frame_md5(&f)));
    assert_eq!(
        got, reference,
        "output sequence differs from serial reference"
    );
}

// perf record/stat --delay=-1 --control=fifo:CTL,ACK; enable ONLY the timed
// region. Acknowledgement is outside the measured interval.
struct PerfControl(std::fs::File, std::io::BufReader<std::fs::File>);
impl PerfControl {
    fn open() -> Option<Self> {
        let paths = std::env::var("RAV1D_PERF_CONTROL").ok()?;
        let (ctl, ack) = paths.split_once(',').expect("CTL,ACK");
        Some(Self(
            std::fs::OpenOptions::new().write(true).open(ctl).unwrap(),
            std::io::BufReader::new(std::fs::File::open(ack).unwrap()),
        ))
    }
    fn command(&mut self, command: &str) {
        use std::io::{BufRead, Write};
        writeln!(self.0, "{command}").unwrap();
        self.0.flush().unwrap();
        let mut ack = String::new();
        assert!(
            self.1.read_line(&mut ack).unwrap() > 0,
            "perf control closed"
        );
        assert_eq!(
            ack.trim().trim_start_matches('\0'),
            "ack",
            "perf control acknowledgement"
        );
    }
}

fn main() {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        hook(info);
        std::process::exit(1);
    }));
    let a: Vec<_> = std::env::args().collect();
    assert!(a.len() >= 5, "INPUT THREADS INSTANCES PASSES [REPS]");
    let threads: u32 = a[2].parse().unwrap();
    let instances: usize = a[3].parse().unwrap();
    let passes: usize = a[4].parse().unwrap();
    let reps: usize = a.get(5).map(|v| v.parse().unwrap()).unwrap_or(1);
    assert!(threads > 0 && instances > 0 && passes > 0 && reps > 0);
    assert!(
        threads as usize * instances <= 64,
        "bound instrumentation worker slots"
    );
    let input = packets(&a[1]);
    if let Ok(t) = std::env::var("RAV1D_PRIME_THREADS") {
        drop(decoder(t.parse().expect("prime threads")));
    }
    // Run the reference in a separate joined thread, so usage TLS is merged.
    let reference = std::thread::scope(|scope| {
        scope
            .spawn(|| {
                let mut reference = Vec::new();
                let mut dec = decoder(1);
                run(&mut dec, &input, |f| {
                    let hash = frame_md5(&f);
                    #[cfg(feature = "probe-tasktime")]
                    println!(
                        "GEOMETRY\t{}\t{}\t{}\t{:?}",
                        reference.len(),
                        f.width(),
                        f.height(),
                        f.probe_geometry()
                    );
                    println!(
                        "FRAME\t{}\t{}x{}\t{}\t{}",
                        reference.len(),
                        f.width(),
                        f.height(),
                        f.bit_depth(),
                        hash
                    );
                    reference.push(hash);
                });
                reference
            })
            .join()
            .unwrap()
    });
    let frames = reference.len();
    println!(
        "CONFIG\tpackets={}\tframes={}\tthreads={}\tinstances={}\tpasses={}\treps={}\tframe_delay=auto\tusage={}\twide={}\ttasks={}",
        input.len(),
        frames,
        threads,
        instances,
        passes,
        reps,
        cfg!(feature = "probe-usage"),
        cfg!(feature = "probe-wide"),
        cfg!(feature = "probe-tasktime")
    );
    let gate = Arc::new(Barrier::new(instances + 1));
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for i in 0..instances {
            let gate = gate.clone();
            let input = &input;
            let reference = &reference;
            handles.push(scope.spawn(move || {
                let mut dec = decoder(threads);
                validate(&mut dec, input, reference);
                gate.wait(); // warmup complete, controller resets diagnostic counters
                for rep in 0..reps {
                    gate.wait();
                    let start = Instant::now();
                    for _ in 0..passes {
                        let n = run(&mut dec, input, |f| {
                            black_box(&f);
                        });
                        assert_eq!(n, reference.len(), "timed frame count");
                    }
                    let ms = start.elapsed().as_secs_f64() * 1000.0;
                    gate.wait();
                    println!("INSTANCE\t{rep}\t{i}\t{ms:.6}");
                }
                gate.wait(); // let controller report before validation/teardown
                validate(&mut dec, input, reference);
            }));
        }
        gate.wait();
        #[cfg(feature = "probe-wide")]
        rav1d_disjoint_mut::wide_probe::reset();
        #[cfg(feature = "probe-tasktime")]
        {
            rav1d_safe::src::probe_tasktime::reset();
            rav1d_safe::src::probe_tasktime::start_monitor();
        }
        let mut perf = PerfControl::open();
        if let Some(p) = &mut perf {
            p.command("enable");
        }
        for rep in 0..reps {
            let start = Instant::now();
            gate.wait();
            gate.wait();
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            let n = frames * instances * passes;
            println!("RESULT\t{rep}\t{n}\t{ms:.6}\t{:.6}", ms / n as f64);
        }
        if let Some(p) = &mut perf {
            p.command("disable");
        }
        #[cfg(feature = "probe-wide")]
        print!("{}", rav1d_disjoint_mut::wide_probe::report());
        #[cfg(feature = "probe-tasktime")]
        rav1d_safe::src::probe_tasktime::report((frames * instances * passes * reps) as u64);
        gate.wait();
        for h in handles {
            h.join().unwrap();
        }
    });
    println!(
        "VALIDATED\t{}\tlifetime_frames={}\ttimed_frames={}",
        instances * 2 + 1,
        frames * (1 + instances * (2 + passes * reps)),
        frames * instances * passes * reps
    );
    #[cfg(feature = "probe-usage")]
    print!("{}", rav1d_disjoint_mut::usage_probe::report());
}
