import os, sys
os.environ['SCIENCECLAW_TASK_CACHE']='/private/tmp/claude-501/sc-scratch/sota/g2_work/f39cache'
R='/Users/admin/Library/Application Support/Claude/scratch-workspaces/b286c85c-fb50-4909-a7a4-692fbcc9e2e6/67544b4d-5c42-4571-a504-64db11cd447d/scratch-2026-09-28-749df2/scienceclaw'
sys.path.insert(0,R)
import numpy as np
from sklearn.linear_model import LogisticRegression
from scienceclaw.bench.tasks.for39_eedi import Adapter, N_MASKS, N_Q
import scilib.adaptive as A
ad=Adapter(data_root='/Users/admin/Datasets/ScienceClaw-rebuild-20260928/datasets')
data=ad._data.get(); setup=ad._setup.get()
trrows=setup['train_rows']; tr=data.train.correct[trrows]
model=A.fit_item_curves(tr)
rng=np.random.default_rng(5)
TrAns=(tr>=0).astype(np.float32)
TrCor=(tr==1).astype(np.float32)
nrm=np.linalg.norm(TrAns,axis=1)+1e-9
def knn_prob(ans_pat, self_idx=None, K=40):
    a=ans_pat.astype(np.float32); an=np.linalg.norm(a,axis=1)+1e-9
    S=(a@TrAns.T)/(an[:,None]*nrm[None,:])
    if self_idx is not None: S[np.arange(len(a)),self_idx]=-1
    top=np.argpartition(-S,K,axis=1)[:,:K]
    P=np.zeros((len(a),N_Q),dtype=np.float32); Nn=np.zeros_like(P)
    for i in range(len(a)):
        t=top[i]; w=S[i,t]**2
        P[i]=(w[:,None]*TrCor[t]).sum(0); Nn[i]=(w[:,None]*TrAns[t]).sum(0)
    base=(TrCor.sum(0)+1)/(TrAns.sum(0)+2)
    return (P+2*base)/(Nn+2)
def logit(p): p=np.clip(p,1e-4,1-1e-4); return np.log(p/(1-p))
def build(C, T, cq, k_q, self_idx=None):
    n=C.shape[0]
    rev=np.full((n,N_Q),-1)
    for i in range(n):
        idx=rng.permutation(np.flatnonzero(cq[i]))[:k_q]; rev[i,idx]=C[i,idx]
    p_irt=A.predict_proba(model,[rev])
    pat=(C>=0)                       # answered pattern (can_query U targets): visible in the protocol
    p_knn=knn_prob(pat,self_idx)
    nans=np.log((C>=0).sum(1,keepdims=True)+1)*np.ones((1,N_Q))
    return rev,p_irt,p_knn,nans
# fit set: 900 training students (self excluded from neighbours) with 20% random targets
fit_idx=rng.choice(tr.shape[0],900,replace=False)
Cf=tr[fit_idx]; Tf=np.zeros_like(Cf,dtype=bool)
for i in range(len(fit_idx)):
    obs=np.flatnonzero(Cf[i]>=0); Tf[i,rng.permutation(obs)[:max(1,int(round(.2*obs.size)))]]=True
cqf=(Cf>=0)&~Tf
_,pi,pk,nf=build(Cf,Tf,cqf,10,self_idx=fit_idx)
X=lambda pi,pk,nf,T: np.c_[logit(pi[T]),logit(pk[T]),nf[T]]
lr_all=LogisticRegression(C=10,max_iter=500).fit(X(pi,pk,nf,Tf),Cf[Tf])
lr_irt=LogisticRegression(C=10,max_iter=500).fit(logit(pi[Tf])[:,None],Cf[Tf])
print('coef',lr_all.coef_,lr_all.intercept_)
for tag,mat in (('public',data.public),('private',data.private)):
    C=mat.correct; a_irt=[];a_all=[];a_knn=[];a_base=[]
    for m in range(N_MASKS):
        T=mat.targets[:,m,:]; cq=(C>=0)&~T
        rev,pi_,pk_,nn_=build(C,T,cq,10)
        y=C[T]
        a_base.append(((pi_[T]>.5)==y).mean())
        a_irt.append((lr_irt.predict(logit(pi_[T])[:,None])==y).mean())
        a_knn.append(((pk_[T]>.5)==y).mean())
        a_all.append((lr_all.predict(X(pi_,pk_,nn_,T))==y).mean())
    print(tag,'IRT10',round(np.mean(a_base),4),'IRT10+LR',round(np.mean(a_irt),4),'kNN-on-answered-pattern only (0 queries)',round(np.mean(a_knn),4),'IRT10+kNN pattern+n_answered LR',round(np.mean(a_all),4),flush=True)
