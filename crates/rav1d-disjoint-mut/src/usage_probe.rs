//! Diagnostic census of the production tracker. NO timing claims from this build.
//!
//! Hot counters are thread local; each thread merges once when it exits. Join
//! every worker before `report`. There is deliberately no reset: setup, validation,
//! and teardown are included and must be counted in the driver's denominator.
//! Occupancy counts allocation attempts per shard, not logical borrows. Range
//! counts are attempts (including empty ranges); rectangles count acceptances.

use core::panic::Location;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::format;
use std::string::String;
use std::sync::{Mutex, OnceLock};

type BorrowKey = (&'static Location<'static>, bool, u32, u32, usize, u32);
type BorrowTotals = (u64, u64, u64);
type PolicyKey = (&'static str, usize, usize, u32, usize);

#[derive(Default)]
struct Counts {
    // Location, mutable, bytes log bucket, rows log bucket, active shards, shift.
    borrows: BTreeMap<BorrowKey, BorrowTotals>,
    // event, length in container elements, shards, block shift, row stride bytes.
    policies: BTreeMap<PolicyKey, (u64, u64)>,
    occupancy: [u64; 8],
    threads: u64,
}

static TOTAL: OnceLock<Mutex<Counts>> = OnceLock::new();
fn total() -> &'static Mutex<Counts> {
    TOTAL.get_or_init(|| Mutex::new(Counts::default()))
}

#[derive(Default)]
struct Local(Counts);
impl Drop for Local {
    fn drop(&mut self) {
        let mut t = total().lock().unwrap();
        for (k, (n, bytes, hull)) in core::mem::take(&mut self.0.borrows) {
            let v = t.borrows.entry(k).or_default();
            v.0 += n;
            v.1 += bytes;
            v.2 += hull;
        }
        for (k, (n, bytes)) in core::mem::take(&mut self.0.policies) {
            let v = t.policies.entry(k).or_default();
            v.0 += n;
            v.1 += bytes;
        }
        for (dst, src) in t.occupancy.iter_mut().zip(self.0.occupancy) {
            *dst += src;
        }
        t.threads += 1;
    }
}

std::thread_local! { static LOCAL: RefCell<Local> = RefCell::new(Local::default()); }

// Bucket 0 means zero, bucket k>0 means [2^(k-1), 2^k-1].
fn bucket(n: usize) -> u32 {
    usize::BITS - n.leading_zeros()
}

pub fn borrow(
    loc: &'static Location<'static>,
    mutable: bool,
    bytes: usize,
    rows: usize,
    hull: usize,
    mask: usize,
    shift: u32,
) {
    LOCAL.with(|c| {
        let mut c = c.borrow_mut();
        let v =
            c.0.borrows
                .entry((loc, mutable, bucket(bytes), bucket(rows), mask + 1, shift))
                .or_default();
        v.0 += 1;
        v.1 += bytes as u64;
        v.2 += hull as u64;
    });
}

pub fn occupancy(n: usize) {
    LOCAL.with(|c| c.borrow_mut().0.occupancy[n] += 1);
}

pub fn policy(
    event: &'static str,
    len: usize,
    mask: usize,
    shift: u32,
    stride: usize,
    allocated: usize,
) {
    LOCAL.with(|c| {
        let mut c = c.borrow_mut();
        let v =
            c.0.policies
                .entry((event, len, mask + 1, shift, stride))
                .or_default();
        v.0 += 1;
        v.1 += allocated as u64;
    });
}

/// Flush the reporting thread too. Call exactly once, after joining all workers.
pub fn report() -> String {
    LOCAL.with(|c| drop(core::mem::take(&mut *c.borrow_mut())));
    let t = total().lock().unwrap();
    let mut out = format!("USAGE_THREADS\t{}\n", t.threads);
    use core::fmt::Write;
    for ((loc, mutable, bytes_bucket, rows_bucket, shards, shift), (n, bytes, hull)) in &t.borrows {
        let _ = writeln!(
            out,
            "BORROW\t{loc}\t{mutable}\t{bytes_bucket}\t{rows_bucket}\t{shards}\t{shift}\t{n}\t{bytes}\t{hull}"
        );
    }
    for ((event, len, shards, shift, stride), (n, bytes)) in &t.policies {
        let _ = writeln!(
            out,
            "POLICY\t{event}\t{len}\t{shards}\t{shift}\t{stride}\t{n}\t{bytes}"
        );
    }
    for (n, count) in t.occupancy.iter().enumerate() {
        let _ = writeln!(out, "OCCUPANCY\t{n}\t{count}");
    }
    out
}
