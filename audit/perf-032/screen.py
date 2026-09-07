from pathlib import Path
import subprocess,os,json,time,statistics,hashlib
root=Path(__file__).resolve().parent
arms=['base','compact','layout']
cases=[('construct',4000,1),('small',100000,1),('small',100000,4),('large',100000,1),('large',100000,4),('large',100000,8)]
rows=[]
for rnd in range(8):
    for case,n,t in cases:
        for arm in arms[rnd%3:]+arms[:rnd%3]:
            cmd=[str(root/('bench-'+arm)),case,str(n),str(t)]
            p=subprocess.run(cmd,check=True,capture_output=True,text=True,timeout=20)
            val=int(p.stdout.strip().split('\t')[-1])
            rows.append(dict(round=rnd,arm=arm,case=case,iterations=n,threads=t,ns=val,load=Path('/proc/loadavg').read_text().strip()))
    (root/'timings.json').write_text(json.dumps(rows,indent=2))
    print('round',rnd,'done',flush=True)
summary=[]
for case,n,t in cases:
    selected=[x for x in rows if x['case']==case and x['threads']==t and x['round']>0]
    base={x['round']:x['ns'] for x in selected if x['arm']=='base'}
    for arm in arms:
        vals=[x for x in selected if x['arm']==arm]
        ratios=[x['ns']/base[x['round']] for x in vals]
        summary.append(dict(case=case,threads=t,arm=arm,median_ns=statistics.median(x['ns'] for x in vals),paired_median_ratio=statistics.median(ratios),min_ratio=min(ratios),max_ratio=max(ratios),faster=sum(x<1 for x in ratios),rounds=len(vals)))
(root/'timing-summary.json').write_text(json.dumps(summary,indent=2))
(root/'binary-hashes.json').write_text(json.dumps({a:hashlib.sha256((root/('bench-'+a)).read_bytes()).hexdigest() for a in arms},indent=2))
for x in summary:print(json.dumps(x),flush=True)
