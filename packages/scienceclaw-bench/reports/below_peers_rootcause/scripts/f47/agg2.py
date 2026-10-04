import json,glob,collections,math
import numpy as np
from scipy.optimize import curve_fit
R=collections.defaultdict(list)
for f in sorted(glob.glob('out/curve_*_*.json')):
    d=json.load(open(f))
    if d['epochs']==8: R[d['n_train']].append(d)
ns=np.array(sorted(R));
# saturating power law fit on seed means: L(n)=Linf - a*n^-b
for p in ['dev','test','pud']:
  for key in ['las','uas']:
    y=np.array([np.mean([d['pools'][p][key] for d in R[n]]) for n in ns])
    f=lambda n,Li,a,b: Li-a*(n/1000.0)**(-b)
    best=None
    for Li0 in [0.85,0.9,0.95]:
        try:
            popt,_=curve_fit(f,ns,y,p0=[Li0,0.15,0.4],bounds=([y.max(),0,0.05],[1.0,2,2]),maxfev=20000)
            r=np.sum((f(ns,*popt)-y)**2)
            if best is None or r<best[1]: best=(popt,r)
        except Exception as e: pass
    popt=best[0]
    print(p,key,'Linf=%.3f a=%.3f b=%.3f'%tuple(popt),' pred 4000: %.3f 8000: %.3f 14554: %.3f'%tuple(f(np.array([4000,8000,14554]),*popt)))
# also constrained ceilings: fixed Linf=0.95
for p in ['dev','test','pud']:
    y=np.array([np.mean([d['pools'][p]['las'] for d in R[n]]) for n in ns])
    for Li in [0.90,0.95]:
        g=lambda n,a,b: Li-a*(n/1000.0)**(-b)
        popt,_=curve_fit(g,ns,y,p0=[0.15,0.4],maxfev=20000)
        print(p,'fixed Linf',Li,'a=%.3f b=%.3f'%tuple(popt),'pred 4000 %.3f 14554 %.3f'%(g(4000,*popt),g(14554,*popt)), 'resid max %.4f'%np.max(np.abs(g(ns,*popt)-y)))
# per-relation table on test, n=1000 and 2000 (mean over seeds), sorted by lost words at 2000
def rel_table(p,n):
    agg=collections.defaultdict(lambda:[0,0,0,0]);
    for d in R[n]:
        for r,v in d['pools'][p]['rel'].items():
            for i in range(4): agg[r][i]+=v[i]
    k=len(R[n]); return {r:[x/k for x in v] for r,v in agg.items()}
for p in ['test','pud']:
    t1=rel_table(p,1000); t2=rel_table(p,2000); t250=rel_table(p,250)
    tot=sum(v[0] for v in t2.values())
    print('\n==',p,'relations: gold count share, LAS-recall@250 / @1000 / @2000, UAS-recall@2000, lost words@2000 (share of all errors)')
    lost=lambda t:sum(v[0]-v[1] for v in t.values())
    L2=lost(t2)
    rows=sorted(t2.items(), key=lambda kv:-(kv[1][0]-kv[1][1]))
    for r,v in rows[:16]:
        rec=lambda t:(t[r][1]/t[r][0] if r in t and t[r][0] else float('nan'))
        prec=v[1]/v[3] if v[3] else float('nan')
        print(f"{r:10s} share {v[0]/tot:5.1%} rec250 {rec(t250):.2f} rec1000 {rec(t1):.2f} rec2000 {rec(t2):.2f} uas-rec2000 {v[2]/v[0]:.2f} prec2000 {prec:.2f} lost {(v[0]-v[1])/L2:5.1%} of errors")
def pos_table(p,n):
    agg=collections.defaultdict(lambda:[0,0,0])
    for d in R[n]:
        for r,v in d['pools'][p]['pos'].items():
            for i in range(3): agg[r][i]+=v[i]
    k=len(R[n]); return {r:[x/k for x in v] for r,v in agg.items()}
for p in ['test']:
    t1=pos_table(p,1000); t2=pos_table(p,2000); tot=sum(v[0] for v in t2.values()); L2=sum(v[0]-v[1] for v in t2.values())
    print('\n== POS (gold UPOS) on',p)
    for r,v in sorted(t2.items(), key=lambda kv:-(kv[1][0]-kv[1][1]))[:10]:
        print(f"{r:6s} share {v[0]/tot:5.1%} LASrec1000 {t1[r][1]/t1[r][0]:.2f} LASrec2000 {v[1]/v[0]:.2f} UASrec2000 {v[2]/v[0]:.2f} lost {(v[0]-v[1])/L2:5.1%}")
