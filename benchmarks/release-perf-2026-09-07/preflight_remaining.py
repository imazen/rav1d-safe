import json,subprocess,os
from pathlib import Path
root=Path(__file__).parent
repo=Path('/home/lilith/work/zen/rav1d-safe')
inputs={'multi':repo/'test-vectors/dav1d-test-data/8-bit/features/non_uniform_tiling.ivf','first':Path('/home/lilith/tmp/rav1d-upstream-perf-2026-09-07/non_uniform_first.obu'),'stress':repo/'tests/crash_vectors/tile_threading_cdef_lpf_race.obu','single10':repo/'tests/crash_vectors/lr_sgr_10bpc_noisy_nocdef.obu'}
oracle=json.loads((repo/'benchmarks/upstream-2026-09-07/frame-reference.json').read_text())
records=json.loads((root/'preflight.json').read_text())
seen={(r['arm'],r['input'],r['threads']) for r in records}
for arm in ['current-checked','current-unchecked','current-asm','first-checked','first-unchecked','restored-asm','latest-checked','latest-unchecked','example-control','upstream']:
 for name,path in inputs.items():
  for t in [1,8]:
   if (arm,name,t) in seen:continue
   cmd=[str(root/'bin'/arm),str(path),str(t),'1','2','1']
   try:
    r=subprocess.run(cmd,capture_output=True,text=True,timeout=15)
    row=dict(arm=arm,input=name,threads=t,command=cmd,exit=r.returncode,stdout=r.stdout,stderr=r.stderr)
    frames=[x for x in r.stdout.splitlines() if x.startswith('FRAME\t')]
    row['oracle_match']=frames==oracle[name]
    row['valid']=r.returncode==0 and row['oracle_match'] and 'VALIDATED\t3\t' in r.stdout
    vals=[x for x in r.stdout.splitlines() if x.startswith('RESULT\t')]
    row['ms']=float(vals[0].split('\t')[-1]) if len(vals)==1 else None
   except subprocess.TimeoutExpired as e:
    row=dict(arm=arm,input=name,threads=t,command=cmd,exit='timeout',valid=False,stdout=str(e.stdout),stderr=str(e.stderr))
   records.append(row);(root/'preflight.json').write_text(json.dumps(records,indent=2)+'\n')
   print(arm,name,t,'PASS' if row['valid'] else 'FAIL',row.get('ms'),row['exit'],flush=True)
