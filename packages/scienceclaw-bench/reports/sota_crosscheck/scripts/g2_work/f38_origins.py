import sys, pickle, numpy as np, pandas as pd, time
R='/Users/admin/Library/Application Support/Claude/scratch-workspaces/b286c85c-fb50-4909-a7a4-692fbcc9e2e6/67544b4d-5c42-4571-a504-64db11cd447d/scratch-2026-09-28-749df2/scienceclaw'
sys.path.insert(0,R)
from scienceclaw.bench.tasks import for38_worldbank as m
from scilib.macro import fit_predict
ad=m.Adapter(data_root='/Users/admin/Datasets/ScienceClaw-rebuild-20260928/datasets')
ep=ad.build_episodes('id',1,0xf94d6fe0)[0]
T={t.name:t for t in ep.tools}
tr=T['load_train'].fn({},{}); P=tr['panel']; kinds=tr['indicator_kinds']
cols=[c for c in P.columns if c.startswith('t')]; print(len(cols),cols[0],cols[-1]); print(kinds)
tk=[k for k,v in kinds.items() if 'change' not in v]; print('target kinds',tk)
Pt=P[P.indicator.isin(tk)].reset_index(drop=True)
X=Pt[cols].to_numpy(float)   # (n,32)
def sm(y,p):
    d=np.abs(y)+np.abs(p); return 200*np.mean(np.where(d>0,np.abs(y-p)/np.where(d>0,d,1),0),axis=-1)
out=[]
for o in [15,17,19,21,23,25,27]:
    L=o+1
    if o+4<=31:
        tg=X[:,o+1:o+5]
        ok=np.isfinite(X[:,o])&(np.isfinite(X[:,:L]).sum(1)>=min(20,L-2))&np.all(np.isfinite(tg),1)&np.all(np.where(np.isfinite(X),X,1)>0,1)
    else:
        continue
    idx=np.where(ok)[0]
    sub=Pt.iloc[idx]; hist=X[idx,:L]
    cov=np.stack([np.stack([P[(P.economy_id==e)&(P.indicator==k)][cols[:L]].to_numpy(float)[0] if len(P[(P.economy_id==e)&(P.indicator==k)]) else np.full(L,np.nan) for k in sorted(kinds)]) for e in sub.economy_id])
    t0=time.time()
    y=np.asarray(fit_predict(P,history=hist,indicator=list(sub.indicator),indicator_kinds=kinds,covariates=cov,covariate_indicators=sorted(kinds),economy_id=list(sub.economy_id),region=list(sub.region),income_level=list(sub.income_level),horizon=4))
    nv=np.repeat(hist[:,[-1]],4,1)
    isr=np.array([('percentage' in kinds[k]) for k in sub.indicator])
    a,b=sm(tg[idx*0+np.arange(len(idx))] if False else X[idx,o+1:o+5],y),sm(X[idx,o+1:o+5],nv)
    print(f"origin col {o} (=year {1990+o}) targets {1990+o+1}-{1990+o+4}: n={len(idx)} lib {a.mean():.2f} naive {b.mean():.2f} ratio {a.mean()/b.mean():.3f} | level: lib {a[~isr].mean():.2f} naive {b[~isr].mean():.2f} ({a[~isr].mean()/b[~isr].mean():.3f}) | percent: lib {a[isr].mean():.2f} naive {b[isr].mean():.2f} ({a[isr].mean()/b[isr].mean():.3f}) [{time.time()-t0:.0f}s]",flush=True)
