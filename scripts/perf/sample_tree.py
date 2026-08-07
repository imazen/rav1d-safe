import re,collections,subprocess,sys
f=sys.argv[1]
lines=open(f,errors='replace').read().split('\n')
start=next(i for i,l in enumerate(lines) if l.startswith('Call graph:'))
try: end=next(i for i,l in enumerate(lines) if l.startswith('Total number in stack'))
except StopIteration: end=next(i for i,l in enumerate(lines) if l.startswith('Binary Images'))
row=re.compile(r'^([ +!:|]*?)(\d+) (.*)$')
ent=[];base=None
for ln in lines[start+1:end]:
    m=row.match(ln)
    if not m: continue
    plen,c,rest=len(m.group(1)),int(m.group(2)),m.group(3)
    if 'Thread_' in rest: base=plen; ent.append((0,c,'<thread>')); continue
    if base is None: continue
    nm=re.match(r'^(.*?)\s+\(in ',rest)
    ent.append(((plen-base)//2,c,(nm.group(1) if nm else rest).strip()))
names=sorted({n for _,_,n in ent})
dem=dict(zip(names,subprocess.run(['rustfilt'],input='\n'.join(names),capture_output=True,text=True).stdout.split('\n')))
D=[(d,c,dem.get(n,n)) for d,c,n in ent]
total=sum(c for d,c,n in D if d==0)
def short(n):
    n=n.replace('rav1d_safe::src::','').replace('rav1d_disjoint_mut::','')
    n=re.sub(r'::<[^>]*BitDepth8[^>]*>','',n)
    return n[:96]
# children of the first big occurrence of a symbol pattern, aggregated over all occurrences
def children_of(pat):
    agg=collections.Counter(); selftot=0; incl=0; i=0
    while i<len(D):
        d,c,n=D[i]
        if pat in n:
            incl+=c; ch=0; j=i+1
            while j<len(D) and D[j][0]>d:
                if D[j][0]==d+1: agg[short(D[j][2])]+=D[j][1]; ch+=D[j][1]
                j+=1
            selftot+=c-ch; i=j
        else: i+=1
    return incl, selftot, agg
for pat in sys.argv[2:]:
    incl,slf,agg=children_of(pat)
    print(f"\n### {pat}   inclusive {incl} ({100.0*incl/total:.2f}%)  self {slf} ({100.0*slf/total:.2f}%)")
    for n,v in agg.most_common(10):
        print(f"    {v:7d} {100.0*v/total:6.2f}%  {n}")
