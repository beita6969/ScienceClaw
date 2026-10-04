import os, sys, pickle
os.environ['SCIENCECLAW_TASK_CACHE']='/private/tmp/claude-501/sc-scratch/sota/g2_work/f39cache'
R='/Users/admin/Library/Application Support/Claude/scratch-workspaces/b286c85c-fb50-4909-a7a4-692fbcc9e2e6/67544b4d-5c42-4571-a504-64db11cd447d/scratch-2026-09-28-749df2/scienceclaw'
sys.path.insert(0,R)
import numpy as np
from scienceclaw.bench.tasks.for39_eedi import Adapter, N_MASKS
ad=Adapter(data_root='/Users/admin/Datasets/ScienceClaw-rebuild-20260928/datasets')
setup=ad._setup.get(); data=ad._data.get()
res=pickle.load(open('/private/tmp/claude-501/sc-scratch/sota/g2_work/f39_full.pkl','rb'))
mask_of=setup['mask_of']
rng=np.random.default_rng(1)
def score(hits,tot,masks,idx,how='mask'):
    if how=='pooled': return hits[idx].sum()/tot[idx].sum()
    accs=[]
    for m in np.unique(masks[idx]):
        s=idx[masks[idx]==m]; accs.append(hits[s].sum()/tot[s].sum())
    return float(np.mean(accs))
for split,tag in (('id','public'),('val','public'),('src','public'),('ood','private')):
    pool=[i for i in setup['pools'][split]]
    users=res[tag]['users']; pos={f'eedi-u{u}':r for r,u in enumerate(users)}
    rows=np.array([pos[i] for i in pool]); masks=np.array([mask_of[i] for i in pool])
    tot=res[tag]['tot'][rows,masks]
    H={k:res[tag]['hits'][k][rows,masks] for k in ('maj','prior','rand','bald','batch')}
    print(f'== {split} pool n={len(pool)} mean targets/student={tot.mean():.1f}  full-pool (own-mask) scores:',
          {k:round(score(v,tot,masks,np.arange(len(pool))),4) for k,v in H.items()})
    for how in ('mask','pooled'):
        sc={k:[] for k in H}
        for _ in range(4000):
            idx=rng.choice(len(pool),16,replace=False)
            for k in H: sc[k].append(score(H[k],tot,masks,idx,how))
        sc={k:np.array(v) for k,v in sc.items()}
        ref=sc['maj']
        line=f' [{how}] ref mean {ref.mean():.3f} sd {ref.std():.3f} | '
        for k in ('prior','rand','bald','batch'):
            d=sc[k]-ref
            line+=f'{k}: score {sc[k].mean():.3f} sd {sc[k].std():.3f}, delta {d.mean():+.3f} sd {d.std():.3f}, pass(>=+.02) {np.mean(d>=0.02):.2f}, P(delta<0) {np.mean(d<0):.2f} | '
        print(line)
