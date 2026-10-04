import sys, numpy as np, time
R=sys.argv[1]; sys.path.insert(0,R)
from scipy import integrate, stats
from scienceclaw.bench.tasks._forecast_common import crps_normal
# (1) crps_normal parity: numerical integral of (F(x)-1[x>=y])^2 and sample-based estimator
rng=np.random.default_rng(1)
mx=0
for _ in range(40):
    mu,sg,y=rng.normal(),abs(rng.normal())+0.2,rng.normal()*2
    f=lambda x:(stats.norm.cdf(x,mu,sg)-(x>=y))**2
    num=integrate.quad(f,-60,y,limit=400)[0]+integrate.quad(f,y,60,limit=400)[0]
    mx=max(mx,abs(num-crps_normal(np.array([mu]),np.array([sg]),np.array([y]))[0]))
x=rng.normal(0.3,1.5,size=200000); xp=rng.permutation(x)
mc=np.mean(np.abs(x-1.1))-0.5*np.mean(np.abs(x-xp))
print('crps_normal vs numerical integral, max abs diff',mx,'; MC check',mc,crps_normal(np.array([0.3]),np.array([1.5]),np.array([1.1]))[0])
# (2) data
from scienceclaw.bench.tasks import for41_neon as F
root=__import__('pathlib').Path('/Users/admin/Datasets/ScienceClaw-rebuild-20260928/datasets')
t=time.time()
d=F.load_neon(root,20260928)
print('loaded',time.time()-t,{k:len(v) for k,v in d.pools.items()})
H=F.H
def build(split, refonly=True):
    rows=[]
    for iid in d.pools[split]:
        s,ti=d.items[iid]
        vis=d.visible[s].copy(); vis[ti+1:ti+1+H]=np.nan   # own window withheld
        hist=vis[max(0,ti-F.L+1):ti+1]; hd=d.days[max(0,ti-F.L+1):ti+1]
        tdays=d.days[ti]+np.arange(1,H+1).astype('timedelta64[D]')
        obs=d.raw[s][ti+1:ti+1+H,:2]
        rows.append((s,ti,hist,hd,tdays,obs))
    return rows
def score_rows(rows, fn):
    out=[]
    for (s,ti,hist,hd,tdays,obs) in rows:
        rec=[]
        for vi in (0,1):
            mu,sd=fn(hist[:,vi],hd,tdays,vi)
            o=obs[:,vi]; m=np.isfinite(o)
            c=np.full(H,np.nan); c[m]=crps_normal(mu[m],sd[m],o[m])
            rec.append(c)
        out.append(rec)
    return np.array(out)   # (n,2,H)
def clim(h,hd,td,vi): return F.climatology(h,hd,td)
def persist(h,hd,td,vi):
    # last observed value within 7 days, sd = std of 30d-ahead changes estimated from history at lag k (crude: growing sd)
    fin=np.where(np.isfinite(h))[0]; last=h[fin[-1]]
    mu=np.full(H,last)
    hh=h[np.isfinite(h)]
    sd0=np.nanstd(np.diff(h[-365:])[np.isfinite(np.diff(h[-365:]))]) if len(hh)>10 else 1.0
    sd=np.maximum(sd0*np.sqrt(np.arange(1,H+1)),0.1)
    sd=np.minimum(sd, max(np.nanstd(hh[-365:]),0.1)*1.5)
    return mu,sd
def anom(h,hd,td,vi):
    mu,sd=F.climatology(h,hd,td)
    fin=np.where(np.isfinite(h))[0];
    # anomaly of last obs vs clim for that day
    mu_h,sd_h=F.climatology(h,hd,np.array([hd[fin[-1]]]))
    a=h[fin[-1]]-mu_h[0]
    rho=0.9 if vi==1 else 0.85
    mu2=mu+a*rho**(np.arange(1,H+1)+ (len(h)-1-fin[-1]))
    sd2=sd*np.sqrt(1-0.5*(rho**(2*np.arange(1,H+1))))
    return mu2,np.maximum(sd2,0.1)
res={}
for split in ('id','ood','val'):
    rows=build(split)
    for nm,fn in (('clim',clim),('persist',persist),('clim+anom',anom)):
        C=score_rows(rows,fn)
        per=[np.nanmean(C[:,vi,:]) for vi in (0,1)]
        res[(split,nm)]=C
        print(split,nm,'full-pool CRPS oxy %.3f temp %.3f mean %.3f'%(per[0],per[1],np.mean(per)))
# 16-item slice noise for id pool clim and clim+anom, distinct sites where possible
rng=np.random.default_rng(0)
for split in ('id','ood'):
    C0=res[(split,'clim')]; C1=res[(split,'clim+anom')]
    sites=np.array([d.items[i][0] for i in d.pools[split]])
    v0=[];v1=[];ratio=[]
    for _ in range(3000):
        idx=rng.choice(len(sites),16,replace=False)
        a=np.mean([np.nanmean(C0[idx,vi,:]) for vi in (0,1)]); b=np.mean([np.nanmean(C1[idx,vi,:]) for vi in (0,1)])
        v0.append(a);v1.append(b);ratio.append(b/a)
    v0,v1,ratio=map(np.array,(v0,v1,ratio))
    print(split,'16-item slices: clim mean %.3f sd %.3f [p5 %.3f p95 %.3f]; clim+anom mean %.3f sd %.3f; ratio(anom/clim) mean %.3f p5 %.3f p95 %.3f'%(v0.mean(),v0.std(),*np.percentile(v0,[5,95]),v1.mean(),v1.std(),ratio.mean(),*np.percentile(ratio,[5,95])))
