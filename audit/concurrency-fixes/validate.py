import os, pathlib, subprocess, json, time, sys
root=pathlib.Path('/home/lilith/tmp/rav1d-policy-mc-2026-09-07')
logs=root/'validation'; logs.mkdir(exist_ok=True)
gates=[
('decoder-corpus', ['cargo','nextest','run','--release','-p','rav1d-safe','--test','decode_md5_verify','--test','filmgrain_threads','--test','strictness','--test','decode_md5_committed','--test','safe_simd_crashes','--test','fuzz_regression','--test-threads','1'], {}),
('decoder-debug', ['cargo','nextest','run','-p','rav1d-safe','--test','decode_md5_committed','--test','safe_simd_crashes','--test','fuzz_regression','--test-threads','1'], {}),
('disjoint', ['cargo','test','-p','rav1d-disjoint-mut','--features','aligned,pic-buf,zerocopy','--','--test-threads=1'], {}),
('clippy', ['cargo','clippy','-p','rav1d-safe','-p','rav1d-disjoint-mut','--lib','--','-D','warnings'], {}),
('clippy-c-ffi', ['cargo','clippy','-p','rav1d-safe','--lib','--features','c-ffi','--','-D','warnings'], {}),
('clippy-probes', ['cargo','clippy','-p','rav1d-safe','--lib','--features','probe-sites,probe-usage','--','-D','warnings'], {}),
('no-std', ['cargo','test','-p','rav1d-disjoint-mut','--no-default-features','--test','adversarial_api','explicit_policy','--','--test-threads=1'], {}),
('legacy', ['cargo','check','-p','rav1d-disjoint-mut','--features','__tracker_legacy'], {}),
('loom', ['cargo','test','-p','rav1d-disjoint-mut','--features','__shards_4','--lib','loom_protocol','--','--test-threads=1'], {'RUSTFLAGS':'--cfg disjoint_mut_loom','CARGO_TARGET_DIR':'target/review-loom'}),
('miri-stacked', ['cargo','+nightly','miri','test','-p','rav1d-disjoint-mut','--features','aligned,pic-buf,zerocopy','--test','adversarial_api','explicit_policy'], {'MIRIFLAGS':''}),
('miri-tree', ['cargo','+nightly','miri','test','-p','rav1d-disjoint-mut','--features','aligned,pic-buf,zerocopy','--test','adversarial_api','explicit_policy'], {'MIRIFLAGS':'-Zmiri-tree-borrows'}),
('extent', ['cargo','nextest','run','--release','-p','rav1d-safe','--features','probe-sites','--test','guard_extent_budget','--test-threads','1'], {}),
('api', ['cargo','+nightly','public-api','-p','rav1d-disjoint-mut','--features','aligned,pic-buf,zerocopy','--omit','blanket-impls','--color','never'], {'CARGO_TARGET_DIR':'target/review-api'}),
('semver', ['cargo','semver-checks','--manifest-path','crates/rav1d-disjoint-mut/Cargo.toml','--baseline-root','/home/lilith/tmp/rav1d-review-2026-09-05/releases/rav1d-disjoint-mut-0.3.1','--features','aligned,pic-buf,zerocopy','--release-type','patch'], {}),
]
results=[]
for name,cmd,extra in gates:
 if len(sys.argv)>1 and name not in sys.argv[1:]: continue
 env=os.environ.copy();env.pop('RUSTFLAGS',None);env.pop('CARGO_ENCODED_RUSTFLAGS',None);env['CARGO_TERM_COLOR']='never';env.update(extra)
 start=time.monotonic(); print('START',name,flush=True)
 with (logs/(name+'.log')).open('w') as f:
  p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
 row=dict(name=name,command=cmd,env=extra,exit=p.returncode,seconds=round(time.monotonic()-start,2));results.append(row)
 (logs/'results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(row),flush=True)
 if p.returncode:
  print((logs/(name+'.log')).read_text()[-8000:],flush=True)
  raise SystemExit(p.returncode)
