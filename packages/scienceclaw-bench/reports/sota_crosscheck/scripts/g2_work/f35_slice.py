import sys, numpy as np, json
sys.path.insert(0,'/Users/admin/Library/Application Support/Claude/scratch-workspaces/b286c85c-fb50-4909-a7a4-692fbcc9e2e6/67544b4d-5c42-4571-a504-64db11cd447d/scratch-2026-09-28-749df2/scienceclaw')
from pathlib import Path
from scienceclaw.bench.tasks.for35_tourism import read_tsf, snaive
from scienceclaw.bench.tasks._forecast_common import mase
S=read_tsf(Path('/Users/admin/Datasets/ScienceClaw-rebuild-20260928/datasets/for35-monash-tourism-monthly/data/tourism_monthly_dataset.tsf'),'tm')
F=json.load(open('f35_forecasts.json'))
def M(pred): return np.array([mase(S[i].target,np.asarray(pred[i]),S[i].history,12) for i in range(len(S))])
sn=M([snaive(s.history,24,12) for s in S]); et=M(F['ets']); th=M(F['theta'])
comb=M([ (np.asarray(F['ets'][i])+np.asarray(F['theta'][i])+snaive(S[i].history,24,12))/3 for i in range(len(S))])
iid=np.array([s.start[:4]=='1980' for s in S])
for nm,a in [('snaive',sn),('ets',et),('theta',th),('comb3',comb)]:
    print(f"{nm:7s} all366 {a.mean():.3f} | iid264 {a[iid].mean():.3f} | non1980(102) {a[~iid].mean():.3f}")
rng=np.random.default_rng(0)
for name,mask in [('iid',iid),('non1980',~iid)]:
    idx=np.where(mask)[0]
    for nm,a in [('ets',et),('theta',th),('comb3',comb)]:
        rs=[];ms=[];ss=[]
        for _ in range(4000):
            j=rng.choice(idx,16,replace=False); rs.append(a[j].mean()/sn[j].mean()); ms.append(a[j].mean()); ss.append(sn[j].mean())
        print(f"{name} {nm}: 16-slice MASE mean {np.mean(ms):.3f} sd {np.std(ms):.3f} [5,95%] {np.percentile(ms,5):.2f}-{np.percentile(ms,95):.2f}; ratio-to-snaive mean {np.mean(rs):.3f} sd {np.std(rs):.3f} [5,95] {np.percentile(rs,5):.2f}-{np.percentile(rs,95):.2f}; snaive slice sd {np.std(ss):.3f}")
