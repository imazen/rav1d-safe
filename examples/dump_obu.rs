//! THROWAWAY probe: extract the primary item's AV1 OBU stream from an AVIF and
//! write it out, optionally wrapped as an IVF with N repeated frames so that an
//! external decoder CLI (dav1d) can be timed on the same bitstream this crate's
//! benches decode in-process.
//!
//! Usage:
//!   dump_obu <input.avif> <out.obu> [out.ivf n_frames width height]

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <in.avif> <out.obu> [out.ivf n w h]", args[0]);
        std::process::exit(2);
    }
    let avif = std::fs::read(&args[1]).expect("read avif");
    let parser = zenavif_parse::AvifParser::from_bytes(&avif).expect("avif parse");
    let obu = parser
        .primary_data()
        .expect("avif primary item")
        .into_owned();
    std::fs::write(&args[2], &obu).expect("write obu");
    eprintln!("obu bytes: {}", obu.len());

    if args.len() >= 7 {
        let n: u32 = args[4].parse().expect("n");
        let w: u16 = args[5].parse().expect("w");
        let h: u16 = args[6].parse().expect("h");
        let mut ivf = Vec::with_capacity(32 + n as usize * (12 + obu.len()));
        ivf.extend_from_slice(b"DKIF");
        ivf.extend_from_slice(&0u16.to_le_bytes()); // version
        ivf.extend_from_slice(&32u16.to_le_bytes()); // header len
        ivf.extend_from_slice(b"AV01");
        ivf.extend_from_slice(&w.to_le_bytes());
        ivf.extend_from_slice(&h.to_le_bytes());
        ivf.extend_from_slice(&30u32.to_le_bytes()); // timebase den
        ivf.extend_from_slice(&1u32.to_le_bytes()); // timebase num
        ivf.extend_from_slice(&n.to_le_bytes()); // frame count
        ivf.extend_from_slice(&0u32.to_le_bytes()); // unused
        for i in 0..n {
            ivf.extend_from_slice(&(obu.len() as u32).to_le_bytes());
            ivf.extend_from_slice(&(i as u64).to_le_bytes());
            ivf.extend_from_slice(&obu);
        }
        std::fs::write(&args[3], &ivf).expect("write ivf");
        eprintln!("ivf bytes: {} frames: {}", ivf.len(), n);
    }
}
