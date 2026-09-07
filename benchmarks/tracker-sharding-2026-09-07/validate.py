import os,subprocess,json,time,sys
from pathlib import Path
root=Path('/home/lilith/tmp/rav1d-perf-solution-2026-09-07'); logs=root/'validation';logs.mkdir(exist_ok=True)
gates=[
('disjoint',['cargo','nextest','run','-p','rav1d-disjoint-mut','--features','aligned,pic-buf,zerocopy','--test-threads','1'],{}),
('docs',['cargo','test','-p','rav1d-disjoint-mut','--features','aligned,pic-buf,zerocopy','--doc'],{}),
('decoder',['cargo','nextest','run','--release','-p','rav1d-safe','--lib','--test','decode_md5_verify','--test','filmgrain_threads','--test','strictness','--test','decode_md5_committed','--test','safe_simd_crashes','--test','fuzz_regression','--test-threads','1'],{'RAV1D_MD5_THREADS':'8'}),
('decoder-debug',['cargo','nextest','run','-p','rav1d-safe','--test','decode_md5_committed','--test','safe_simd_crashes','--test','fuzz_regression','--test-threads','1'],{}),
('loom',['cargo','test','-p','rav1d-disjoint-mut','--features','__shards_4','--lib','loom_protocol','--','--test-threads=1'],{'RUSTFLAGS':'--cfg disjoint_mut_loom','CARGO_TARGET_DIR':'target/review-loom'}),
('miri-stacked',['cargo','+nightly','miri','test','-p','rav1d-disjoint-mut','--test','medium_buffers'],{'MIRIFLAGS':''}),
('miri-tree',['cargo','+nightly','miri','test','-p','rav1d-disjoint-mut','--test','medium_buffers'],{'MIRIFLAGS':'-Zmiri-tree-borrows'}),
('no-std',['cargo','nextest','run','-p','rav1d-disjoint-mut','--no-default-features','--test','medium_buffers','--test-threads','1'],{}),
('clippy',['cargo','clippy','-p','rav1d-safe','-p','rav1d-disjoint-mut','--lib','--','-D','warnings'],{}),
('semver',['cargo','semver-checks','--manifest-path','crates/rav1d-disjoint-mut/Cargo.toml','--baseline-root','/home/lilith/tmp/rav1d-review-2026-09-05/releases/rav1d-disjoint-mut-0.3.1','--features','aligned,pic-buf,zerocopy','--release-type','patch'],{}),
]
results=[]
for name,cmd,extra in gates:
 if len(sys.argv)>1 and name not in sys.argv[1:]:continue
 env={k:v for k,v in os.environ.items() if not k.startswith('RAV1D_') and k not in ['RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS','MIRIFLAGS']};env['CARGO_TERM_COLOR']='never';env.update(extra)
 start=time.monotonic(); print('START',name,flush=True)
 with (logs/(name+'.log')).open('w') as f:result=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
 results.append(dict(name=name,command=cmd,env=extra,exit=result.returncode,seconds=round(time.monotonic()-start,2)))
 (logs/('results-'+str(os.getpid())+'.json')).write_text(json.dumps(results,indent=2)+'\n');print(results[-1],flush=True)
 if result.returncode:
  print((logs/(name+'.log')).read_text()[-7000:],flush=True);sys.exit(result.returncode)
