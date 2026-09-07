from pathlib import Path
import subprocess,json,statistics,hashlib
root=Path(__file__).resolve().parent;repo=Path('/home/lilith/work/zen/rav1d-safe')
# Keep input files fixed and compare in-process decoder timings, never process wall.
cases=[('lr_sgr_10bpc_noisy_nocdef',1),('lr_sgr_10bpc_noisy_nocdef',4),('tile_threading_cdef_lpf_race',1),('tile_threading_cdef_lpf_race',4)]
rows=[]; hashes={}
for rnd in range(6):
    arms=['base','compact','layout'];arms=arms[rnd%3:]+arms[:rnd%3]
    for vec,threads in cases:
        path=repo/'tests/crash_vectors'/(vec+'.obu')
        for arm in arms:
            cmd=[str(root/('decode-'+arm)),str(path),str(threads),'12','1',arm]
            p=subprocess.run(cmd,capture_output=True,text=True,timeout=15)
            (root/'decoder-raw.log').open('a').write('COMMAND '+repr(cmd)+'\n'+p.stdout+p.stderr)
            if p.returncode:print(p.stdout,p.stderr);raise SystemExit(p.returncode)
            checksum=None;ms=None;geom=None
            for line in p.stdout.splitlines():
                cols=line.split('\t')
                if cols[0]=='CHECKSUM':checksum=cols[-1]
                if cols[0]=='RESULT':ms=float(cols[-1])
                if cols[0]=='GEOM':geom=cols[-2:]
            assert checksum is not None and ms is not None
            expected=hashes.setdefault((vec,threads),checksum);assert checksum==expected,(vec,threads,arm,checksum,expected)
            rows.append(dict(round=rnd,arm=arm,vector=vec,threads=threads,ms=ms,checksum=checksum,geometry=geom))
    (root/'decoder-timings.json').write_text(json.dumps(rows,indent=2))
    print('decoder round',rnd,'done',flush=True)
summary=[]
for vec,t in cases:
    sel=[x for x in rows if x['vector']==vec and x['threads']==t and x['round']>0]
    base={x['round']:x['ms'] for x in sel if x['arm']=='base'}
    for arm in ['base','compact','layout']:
        vals=[x for x in sel if x['arm']==arm];ratios=[x['ms']/base[x['round']] for x in vals]
        summary.append(dict(vector=vec,threads=t,arm=arm,median_ms=statistics.median(x['ms'] for x in vals),paired_median_ratio=statistics.median(ratios),min_ratio=min(ratios),max_ratio=max(ratios),faster=sum(x<1 for x in ratios),rounds=len(vals)))
(root/'decoder-summary.json').write_text(json.dumps(summary,indent=2))
(root/'decoder-hashes.json').write_text(json.dumps({a:hashlib.sha256((root/('decode-'+a)).read_bytes()).hexdigest() for a in ['base','compact','layout']},indent=2))
for x in summary:print(json.dumps(x),flush=True)
