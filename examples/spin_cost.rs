//! Price one `core::hint::spin_loop()` on this box.
//!
//! It exists because assuming that number cost a day. `TinyLock::lock_slow`'s
//! entire cost is (iterations x this), the tracker's contention census counts
//! the iterations exactly, and a profile put the symbol at 0.86-1.20 CPU
//! ms/frame on `c256x2048` t=8 against 1,007-1,373 counted iterations/frame.
//! Those two are only consistent if an iteration costs ~1 us, and the obvious
//! assumption ("a spin hint is a few nanoseconds") is what made the
//! contradiction look like an instrument bug instead of a real 80x contention
//! penalty. See `docs/C256_CONTENTION.md` §6.
//!
//! Reports the loop WITH the hint, an empty control loop of the same shape, and
//! the difference — the control is what separates the hint from the loop
//! overhead, and on aarch64 the hint is an `isb` that dominates it.
//!
//! Run it UN-NICED and under `measlock` like any other timed thing:
//!   measlock spin-cost -- cargo run --release --example spin_cost
//!
//! Usage: spin_cost [iterations] [rounds]

use std::hint::black_box;
use std::hint::spin_loop;
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let n: u64 = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200_000_000);
    let rounds: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(3);

    println!("round\tns_per_spin_loop\tns_per_control_iter\tns_hint_only");
    for _ in 0..rounds {
        let t = Instant::now();
        let mut k = 0u64;
        while k < n {
            spin_loop();
            k += 1;
        }
        let hinted = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let mut k = 0u64;
        while k < n {
            black_box(&k);
            k += 1;
        }
        let control = t.elapsed().as_secs_f64();

        let per = |s: f64| s * 1e9 / n as f64;
        println!(
            "{n}\t{:.3}\t{:.3}\t{:.3}",
            per(hinted),
            per(control),
            per(hinted - control)
        );
    }
}
