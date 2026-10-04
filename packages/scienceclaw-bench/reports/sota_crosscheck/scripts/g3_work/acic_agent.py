import sys, glob, pickle, json, numpy as np, pandas as pd
R=sys.argv[1]; sys.path.insert(0,R)
from pathlib import Path
from scienceclaw.bench.tasks import for44_acic as A
root=Path('/Users/admin/Datasets/ScienceClaw-rebuild-20260928/datasets')
ad=A.ACIC2016Adapter(data_root=str(root))
want={}
for d in glob.glob(R+'/runs/final_a0_*/FoR44-*'):
    e=Path(d).name; want[e]=want.get(e,[])+[d]
print({k:len(v) for k,v in want.items()})
found={}
for split,hx in (('id','da552e7c'),('val','25e8987b')):
    seed=int(hx,16)
    for ep in ad.build_episodes(split,4,seed):
        if ep.id in want: found[ep.id]=(ep,seed)
print({k:v[1] for k,v in found.items()})
rows=[]
for eid,(ep,seed) in found.items():
    items=ep.lineage['item_ids']
    sims=[ad._sim(i) for i in items]
    satt=np.array([s.satt for s in sims]); sd=np.array([s.sd_y for s in sims])
    for d in want[eid]:
        y=pickle.load(open(d+'/y.pkl','rb')); run=Path(d).parent.name
        err=(np.asarray(y)-satt)/sd
        for i,(it,e_,sa) in enumerate(zip(items,err,satt)):
            rows.append(dict(ep=eid,run=run,item=it,p=A.scenario_of(it),err=e_,abs_err_raw=(y[i]-sa),satt=sa,sd=sd[i]))
df=pd.DataFrame(rows); df.to_csv('acic_agent_errs.csv',index=False)
for (run,ep),g in df.groupby(['run','ep']):
    print(run,ep,'nrmse %.4f mean err %.4f median|err| %.4f max|err| %.4f (p=%d)'%(np.sqrt((g.err**2).mean()),g.err.mean(),g.err.abs().median(),g.err.abs().max(),g.loc[g.err.abs().idxmax(),'p']))
