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
rng=np.random.default_rng(11)
for tag,mat in (('public',data.public),('private',data.private)):
    C=mat.correct; n=C.shape[0]
    R_={k:[] for k in ('strict_batch','starter_batch_modelpred','starter_batch_copyrevealed','starter_rand_copyrevealed','starter_rand_modelpred')}
    frac=[]
    for m in range(N_MASKS):
        T=mat.targets[:,m,:]; answered=(C>=0)
        cq_strict=answered&~T
        def rev_of(sel):
            r=np.full((n,N_Q),-1)
            for i in range(n):
                q=sel[i][sel[i]>=0]; r[i,q]=C[i,q]
            return r
        # strict (harness rule)
        s=A.select_queries(model,cq_strict,[],10,10,'batch'); r=rev_of(s)
        P=A.predict(model,[r],T).clip(0); R_['strict_batch'].append(((P==C)&T).sum()/T.sum())
        # starter rule: policy sees only 'answered' (targets not known, queryable)
        s=A.select_queries(model,answered,[],10,10,'batch'); r=rev_of(s)
        P=A.predict(model,[r],T).clip(0)
        R_['starter_batch_modelpred'].append(((P==C)&T).sum()/T.sum())
        P2=np.where(r>=0,r,P); R_['starter_batch_copyrevealed'].append(((P2==C)&T).sum()/T.sum())
        frac.append(((r>=0)&T).sum()/T.sum())
        sel=np.full((n,10),-1)
        for i in range(n):
            idx=np.flatnonzero(answered[i]); sel[i,:min(10,idx.size)]=rng.permutation(idx)[:10]
        r=rev_of(sel); P=A.predict(model,[r],T).clip(0)
        R_['starter_rand_modelpred'].append(((P==C)&T).sum()/T.sum()); P2=np.where(r>=0,r,P); R_['starter_rand_copyrevealed'].append(((P2==C)&T).sum()/T.sum())
    print(tag,{k:round(float(np.mean(v)),4) for k,v in R_.items()},'share of target cells revealed (batch, starter rule)',round(float(np.mean(frac)),4),flush=True)
