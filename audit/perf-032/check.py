from pathlib import Path
import subprocess,os,json,time
root=Path(__file__).resolve().parent; rows=[]
for arm in ['compact','layout']:
    env=os.environ.copy();env.pop('RUSTFLAGS',None);env.pop('CARGO_ENCODED_RUSTFLAGS',None)
    env['CARGO_TARGET_DIR']=str(root/'test-target');env['CARGO_TERM_COLOR']='never'
    cmd=['cargo','test','--offline','--manifest-path',str(root/arm/'disjoint/Cargo.toml'),'--features','aligned,pic-buf,zerocopy','--lib','--tests']
    start=time.monotonic()
    with (root/(arm+'-tests.log')).open('w') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
    rows.append(dict(arm=arm,command=cmd,seconds=time.monotonic()-start,exit_code=p.returncode));(root/'tests.json').write_text(json.dumps(rows,indent=2))
    print(arm,'tests',p.returncode,flush=True)
    if p.returncode:print((root/(arm+'-tests.log')).read_text()[-6000:]);raise SystemExit(p.returncode)
