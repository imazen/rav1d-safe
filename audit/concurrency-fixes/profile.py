import pathlib, subprocess, os, json, time
root=pathlib.Path('/home/lilith/tmp/rav1d-policy-mc-2026-09-07'); out=root/'profiles'; out.mkdir(exist_ok=True)
repo=pathlib.Path('/home/lilith/work/zen/rav1d-safe'); stream=repo/'test-vectors/dav1d-test-data/8-bit/features/non_uniform_tiling.ivf'
records=[]
for threads,instances in [(8,1),(8,4)]:
 for arm in ['base','combined']:
  name=f'{arm}-t{threads}-i{instances}'; ctl=out/(name+'.ctl');ack=out/(name+'.ack')
  for p in [ctl,ack]:
   if p.exists():p.unlink()
   os.mkfifo(p)
  env=os.environ.copy();env['RAV1D_PERF_CONTROL']=str(ctl)+','+str(ack)
  cmd=['perf','record','-e','cycles:u','-F','499','--call-graph','dwarf,8192','-o',str(out/(name+'.data')),'--delay=-1','--control=fifo:'+str(ctl)+','+str(ack),'--',str(root/'bin'/arm),str(stream),str(threads),str(instances),'72','1']
  start=time.monotonic();p=subprocess.run(cmd,cwd=repo,env=env,capture_output=True,text=True,timeout=60)
  row=dict(arm=arm,threads=threads,instances=instances,command=cmd,exit=p.returncode,seconds=time.monotonic()-start,stdout=p.stdout,stderr=p.stderr);records.append(row)
  (out/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
  assert p.returncode==0,row
  report=['perf','report','--stdio','--no-children','--percent-limit','0.1','-i',str(out/(name+'.data')),'--sort','symbol','--field-separator',';']
  r=subprocess.run(report,capture_output=True,text=True);assert r.returncode==0,r.stderr
  (out/(name+'.txt')).write_text(r.stdout)
  print(name,flush=True)
