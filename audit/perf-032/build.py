from pathlib import Path
import subprocess,os,json,time
root=Path(__file__).resolve().parent
results=[]
for arm in ['base','compact','layout']:
    env=os.environ.copy();env.pop('RUSTFLAGS',None);env.pop('CARGO_ENCODED_RUSTFLAGS',None)
    env['CARGO_TARGET_DIR']=str(root/'target');env['CARGO_TERM_COLOR']='never'
    cmd=['cargo','build','--offline','--release','--manifest-path',str(root/arm/'harness/Cargo.toml')]
    start=time.monotonic()
    with (root/(arm+'-build.log')).open('w') as f: p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
    results.append(dict(arm=arm,command=cmd,seconds=time.monotonic()-start,exit_code=p.returncode))
    (root/'build.json').write_text(json.dumps(results,indent=2))
    print(arm,p.returncode,flush=True)
    if p.returncode: print((root/(arm+'-build.log')).read_text()[-5000:]);raise SystemExit(p.returncode)
    import shutil
    shutil.copy2(root/'target/release/tracker-perf-screen',root/('bench-'+arm))
