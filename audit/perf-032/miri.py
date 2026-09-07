from pathlib import Path
import subprocess,os,json,time
root=Path(__file__).resolve().parent;rows=[]
for arm in ['compact','layout']:
    for model,flags in [('stacked',''),('tree','-Zmiri-tree-borrows')]:
        env=os.environ.copy();env.pop('RUSTFLAGS',None);env.pop('CARGO_ENCODED_RUSTFLAGS',None)
        env['CARGO_TARGET_DIR']=str(root/'miri-target');env['CARGO_TERM_COLOR']='never';env['MIRIFLAGS']=flags
        cmd=['cargo','+nightly','miri','test','--offline','--manifest-path',str(root/arm/'disjoint/Cargo.toml'),'--test','compact_transition','--test','guard_move_release']
        start=time.monotonic()
        with (root/(arm+'-'+model+'.log')).open('w') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
        rows.append(dict(arm=arm,model=model,command=cmd,flags=flags,seconds=time.monotonic()-start,exit_code=p.returncode));(root/'miri.json').write_text(json.dumps(rows,indent=2))
        print(arm,model,p.returncode,flush=True)
        if p.returncode:print((root/(arm+'-'+model+'.log')).read_text()[-5000:]);raise SystemExit(p.returncode)
