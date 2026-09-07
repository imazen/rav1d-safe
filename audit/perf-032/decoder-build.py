from pathlib import Path
import subprocess,os,json,time,shutil
root=Path(__file__).resolve().parent; rows=[]
for arm in ['base','compact','layout']:
    env=os.environ.copy();env.pop('RUSTFLAGS',None);env.pop('CARGO_ENCODED_RUSTFLAGS',None)
    env['CARGO_TARGET_DIR']=str(root/'decoder-target');env['CARGO_TERM_COLOR']='never'
    cmd=['cargo','build','--offline','--release','--manifest-path',str(root/arm/'decoder/Cargo.toml'),'--example','bench_ab_decode']
    start=time.monotonic()
    with (root/(arm+'-decoder-build.log')).open('w') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
    rows.append(dict(arm=arm,command=cmd,seconds=time.monotonic()-start,exit_code=p.returncode));(root/'decoder-build.json').write_text(json.dumps(rows,indent=2))
    print(arm,'decoder build',p.returncode,flush=True)
    if p.returncode:print((root/(arm+'-decoder-build.log')).read_text()[-3000:]);raise SystemExit(p.returncode)
    shutil.copy2(root/'decoder-target/release/examples/bench_ab_decode',root/('decode-'+arm))
