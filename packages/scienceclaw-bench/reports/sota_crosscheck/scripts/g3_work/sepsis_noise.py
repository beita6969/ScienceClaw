import sys, numpy as np, time
R=sys.argv[1]; sys.path.insert(0,R)
from scienceclaw.bench.tasks import for42_sepsis as F
import tempfile, pathlib
ad=F.Adapter(cache_dir=tempfile.mkdtemp())
print(ad.available())
P=ad._pools(); st=ad._stays()
print({k:len(v) for k,v in P.items()})
import collections
print('septic in train', sum(st[p]['septic'] for p in P['train']), 'of', len(P['train']))
def triplets(pids,preds):
    return np.array([F.utility_triplet(ad._labels(p),pr) for p,pr in zip(pids,preds)])
res={}
for name in ('id','ood','val'):
    pids=P[name]
    rp=ad.reference_predictions(pids)
    T=triplets(pids,rp)
    sept=np.array([st[p]['septic'] for p in pids])
    nu=F.normalized_utility(*T.sum(0))
    # last-12-hours exploit, all-ones, all-zero
    L12=[np.r_[np.zeros(max(0,st[p]['n_hours']-12),int),np.ones(min(12,st[p]['n_hours']),int)] for p in pids]
    T12=triplets(pids,L12)
    ones=triplets(pids,[np.ones(st[p]['n_hours'],int) for p in pids])
    print(name,'ref NU',round(nu,3),'| last-12h NU',round(F.normalized_utility(*T12.sum(0)),3),'| all-ones NU',round(F.normalized_utility(*ones.sum(0)),3),'| n septic',sept.sum(),'/',len(pids),'mean hours septic %.1f nonseptic %.1f'%(np.mean([st[p]['n_hours'] for p in pids if st[p]['septic']]),np.mean([st[p]['n_hours'] for p in pids if not st[p]['septic']])))
    res[name]=(pids,T,sept,rp)
# episode noise & prevalence effect using id+val+src+ood pool stays (ref model, hospital-A trained)
rng=np.random.default_rng(0)
def sim(T,sept,k_sept,n=16,reps=4000):
    ip=np.flatnonzero(sept); ineg=np.flatnonzero(~sept); out=[]
    for _ in range(reps):
        idx=np.r_[rng.choice(ip,k_sept,replace=False),rng.choice(ineg,n-k_sept,replace=False)]
        v=F.normalized_utility(*T[idx].sum(0));
        if v is not None: out.append(v)
    return np.array(out)
# larger evaluation pool: all A non-train stays (id+val+src+dev) with ref model
pids=P['id']+P['val']+P['src']+P['dev']
rp=ad.reference_predictions(pids); T=triplets(pids,rp); sept=np.array([st[p]['septic'] for p in pids])
print('pool of',len(pids),'A stays,',sept.sum(),'septic; pooled ref NU (12.5%%-free natural %.3f) = %.3f'%(sept.mean(),F.normalized_utility(*T.sum(0))))
for k in (1,2,3):
    v=sim(T,sept,k); print('16-stay episodes with %d septic: mean NU %.3f sd %.3f p5 %.3f p95 %.3f'%(k,v.mean(),v.std(),*np.percentile(v,[5,95])))
# natural prevalence: binomial number of septic stays
ps=0.073; ks=rng.binomial(16,ps,size=4000); vv=[]
for k in ks:
    if k==0: continue
    vv.append(sim(T,sept,int(k),reps=1)[0] if True else 0)
print('natural prevalence 7.3%%: P(no septic in 16)=%.2f; NU among defined: mean %.3f sd %.3f'%((1-ps)**16,np.mean(vv),np.std(vv)))
# pooled-over-many-stays NU reweighted to natural prevalence: stay-level sums
sp=np.flatnonzero(sept); sn=np.flatnonzero(~sept)
def pooled(prev):
    # expected sums per stay class scaled to prevalence prev
    ts=T[sp].mean(0); tn=T[sn].mean(0)
    tot=prev*ts+(1-prev)*tn
    return F.normalized_utility(*tot)
for prev in (0.125,0.088,0.073,0.05):
    print('prevalence %.3f -> expected pooled NU %.3f'%(prev,pooled(prev)))
