import os, sys, time, pickle
os.environ['SCIENCECLAW_TASK_CACHE']='/private/tmp/claude-501/sc-scratch/sota/g2_work/f39cache'
R='/Users/admin/Library/Application Support/Claude/scratch-workspaces/b286c85c-fb50-4909-a7a4-692fbcc9e2e6/67544b4d-5c42-4571-a504-64db11cd447d/scratch-2026-09-28-749df2/scienceclaw'
sys.path.insert(0,R)
import numpy as np
from scienceclaw.bench.tasks.for39_eedi import Adapter, N_MASKS, N_Q
import scilib.adaptive as A
ad=Adapter(data_root='/Users/admin/Datasets/ScienceClaw-rebuild-20260928/datasets')
data=ad._data.get(); setup=ad._setup.get()
tr=data.train.correct[setup['train_rows']]
print('train',tr.shape,'public',data.public.correct.shape,'private',data.private.correct.shape)
# leakage checks: student overlap
print('student overlap train/public',len(set(data.train.users)&set(data.public.users)),'train/private',len(set(data.train.users)&set(data.private.users)),'public/private',len(set(data.public.users)&set(data.private.users)))
maj=setup['majority']
t0=time.time(); model=A.fit_item_curves(tr); print('fit',time.time()-t0)
res={}
rng=np.random.default_rng(0)
for tag,mat in (('public',data.public),('private',data.private)):
    n=mat.users.size
    C=mat.correct
    out={k:np.zeros((n,N_MASKS)) for k in ('maj','prior','rand','bald','batch','maj_leak')}
    tot=np.zeros((n,N_MASKS))
    for m in range(N_MASKS):
        T=mat.targets[:,m,:]; cq=(C>=0)&~T
        tot[:,m]=T.sum(1)
        hit=lambda P: ((P==C)&T).sum(1)
        out['maj'][:,m]=hit(np.repeat(maj[None,:],n,0))
        empty=np.full((n,N_Q),-1)
        out['prior'][:,m]=hit(A.predict(model,empty,T).clip(0))
        # random 10 queries among can_query
        sel=np.full((n,10),-1)
        for i in range(n):
            idx=np.flatnonzero(cq[i]); sel[i,:min(10,idx.size)]=rng.permutation(idx)[:10]
        def reveal(sel):
            r=np.full((n,N_Q),-1)
            for i in range(n):
                q=sel[i][sel[i]>=0]; r[i,q]=C[i,q]
            return r
        out['rand'][:,m]=hit(A.predict(model,reveal(sel),T).clip(0))
        for meth in ('bald','batch'):
            s=A.select_queries(model,cq,[],10,10,meth)
            out[meth][:,m]=hit(A.predict(model,reveal(s),T).clip(0))
        print(tag,m,'done',time.time()-t0,flush=True)
    res[tag]={'hits':out,'tot':tot,'users':mat.users}
pickle.dump(res,open('/private/tmp/claude-501/sc-scratch/sota/g2_work/f39_full.pkl','wb'))
for tag in res:
    tot=res[tag]['tot']
    print(tag,'official-protocol mean over 10 masks (pooled per mask):')
    for k,h in res[tag]['hits'].items():
        if k=='maj_leak': continue
        acc=h.sum(0)/tot.sum(0)
        print('  ',k,round(acc.mean(),4),'sd over masks',round(acc.std(),4))
