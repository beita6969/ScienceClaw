import os, sys
os.environ['SCIENCECLAW_TASK_CACHE']='/private/tmp/claude-501/sc-scratch/sota/g2_work/f39cache'
R='/Users/admin/Library/Application Support/Claude/scratch-workspaces/b286c85c-fb50-4909-a7a4-692fbcc9e2e6/67544b4d-5c42-4571-a504-64db11cd447d/scratch-2026-09-28-749df2/scienceclaw'
sys.path.insert(0,R)
import numpy as np
from scienceclaw.bench.tasks.for39_eedi import Adapter, N_MASKS, N_Q
import scilib.adaptive as A
ad=Adapter(data_root='/Users/admin/Datasets/ScienceClaw-rebuild-20260928/datasets')
data=ad._data.get(); setup=ad._setup.get()
tr=data.train.correct[setup['train_rows']]; model=A.fit_item_curves(tr)
rng=np.random.default_rng(3)
for tag,mat in (('public',data.public),('private',data.private)):
    C=mat.correct; n=C.shape[0]
    print(tag,'answers/student mean',round((C>=0).sum(1).mean(),1),'median',np.median((C>=0).sum(1)))
    for k in (0,5,10,20,40,10**6):
        accs=[]
        for m in range(N_MASKS):
            T=mat.targets[:,m,:]; cq=(C>=0)&~T
            rev=np.full((n,N_Q),-1)
            for i in range(n):
                idx=rng.permutation(np.flatnonzero(cq[i]))[:k]; rev[i,idx]=C[i,idx]
            P=A.predict(model,rev,T).clip(0)
            accs.append(((P==C)&T).sum()/T.sum())
        print('  random k=',k if k<1e5 else 'all',round(float(np.mean(accs)),4))
