use rav1d_disjoint_mut::{DisjointMut, set_parallelism};
use std::{hint::black_box, sync::{Arc, Barrier}, time::Instant};
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let case = &args[1];
    let n: usize = args[2].parse().unwrap();
    if case == "construct" {
        let t = Instant::now();
        let buffers: Vec<_> = (0..n).map(|_| DisjointMut::new_eager([0u8;32])).collect();
        black_box(&buffers);
        let elapsed=t.elapsed().as_nanos();
        println!("{case}\t1\t{n}\t{elapsed}");
        return;
    }
    let threads: usize=args[3].parse().unwrap();
    set_parallelism(threads);
    let len=if case=="small" {4096} else {1024*1024};
    let buf=Arc::new(DisjointMut::new_eager(vec![0u64;len]));
    let start=Arc::new(Barrier::new(threads+1));
    let done=Arc::new(Barrier::new(threads+1));
    let mut workers=Vec::new();
    for worker in 0..threads {
        let buf=buf.clone();let start=start.clone();let done=done.clone();
        workers.push(std::thread::spawn(move|| {
            let stride=len/threads; let mut state=(worker as u64+1)*7919;
            start.wait();
            for _ in 0..n {
                state^=state<<13; state^=state>>7; state^=state<<17;
                let i=worker*stride+(state as usize%stride);
                let mut g=buf.index_mut(black_box(i));
                *g+=1;
                black_box(&g);
            }
            done.wait();
        }));
    }
    let t=Instant::now(); start.wait(); done.wait(); let elapsed=t.elapsed().as_nanos();
    for w in workers {w.join().unwrap();}
    let buf=Arc::try_unwrap(buf).unwrap().into_inner();
    assert_eq!(buf.iter().sum::<u64>(),(n*threads) as u64);
    println!("{case}\t{threads}\t{n}\t{elapsed}");
}
