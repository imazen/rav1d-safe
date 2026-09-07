import os, subprocess, pathlib, json, sys, statistics
root=pathlib.Path('/home/lilith/tmp/rav1d-policy-mc-2026-09-07')
repo=pathlib.Path('/home/lilith/work/zen/rav1d-safe')
inputs={'multi':repo/'test-vectors/dav1d-test-data/8-bit/features/non_uniform_tiling.ivf','first':pathlib.Path('/home/lilith/tmp/rav1d-concurrency-2026-09-06/inputs/non_uniform_first.obu'),'stress':repo/'tests/crash_vectors/tile_threading_cdef_lpf_race.obu','single10':repo/'tests/crash_vectors/lr_sgr_10bpc_noisy_nocdef.obu'}
mode=sys.argv[1]; arms=sys.argv[2].split(','); reps=int(sys.argv[3])
cells=[(1,1,0),(1,1,24),(8,1,0)] if mode=='policy' else [(t,1,0) for t in [1,2,4,8,16,24]]+[(1,4,0),(8,4,0),(1,1,24)]
if mode=='reconcile':
 inputs={k:v for k,v in inputs.items() if k in ['multi','stress']}
 cells=[(8,1,0),(16,1,0),(24,1,0)]
rows=[]
out=root/(mode+'.jsonl')
with out.open('w') as f:
 for rep in range(reps):
  for name,path in inputs.items():
   for threads,instances,prime in cells:
    rotated=arms[rep%len(arms):]+arms[:rep%len(arms)]
    order=rotated if (rep+threads+prime)%2==0 else rotated[::-1]
    for arm in order:
     env=os.environ.copy(); env.pop('RAV1D_PRIME_THREADS',None)
     if prime: env['RAV1D_PRIME_THREADS']=str(prime)
     passes=8 if name=='multi' else (24 if name=='stress' else 64)
     cmd=[str(root/'bin'/arm),str(path),str(threads),str(instances),str(passes),'1']
     p=subprocess.run(cmd,cwd=repo,env=env,text=True,capture_output=True,timeout=60)
     row=dict(rep=rep,input=name,threads=threads,instances=instances,prime=prime,arm=arm,command=cmd,exit=p.returncode,stdout=p.stdout,stderr=p.stderr)
     if p.returncode: print(row,flush=True); raise SystemExit(p.returncode)
     vals=[x.split('\t') for x in p.stdout.splitlines() if x.startswith('RESULT\t')]
     assert len(vals)==1, row
     row['ms']=float(vals[0][-1]); rows.append(row);f.write(json.dumps(row)+'\n'); f.flush()
  print('completed round',rep+1,flush=True)
for name in inputs:
 for threads,instances,prime in cells:
  m=[statistics.median(r['ms'] for r in rows if (r['input'],r['threads'],r['instances'],r['prime'],r['arm'])==(name,threads,instances,prime,a)) for a in arms]
  print(name,threads,instances,prime,*[round(x,4) for x in m],*[round(x/m[0],4) for x in m[1:]],flush=True)
