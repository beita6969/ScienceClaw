import sys, numpy as np, collections
sys.path.insert(0,'/Users/admin/Library/Application Support/Claude/scratch-workspaces/b286c85c-fb50-4909-a7a4-692fbcc9e2e6/67544b4d-5c42-4571-a504-64db11cd447d/scratch-2026-09-28-749df2/scienceclaw')
from pathlib import Path
from scienceclaw.bench.tasks.for35_tourism import read_tsf, snaive
from scienceclaw.bench.tasks._forecast_common import mase, hash_order
p=Path('/Users/admin/Datasets/ScienceClaw-rebuild-20260928/datasets/for35-monash-tourism-monthly/data/tourism_monthly_dataset.tsf')
S=read_tsf(p,'tm')
print(len(S), collections.Counter(s.start[:4] for s in S))
L=[len(s.values) for s in S]; print('len min/max',min(L),max(L))
m=12;H=24
ms=[mase(s.target, snaive(s.history,H,m), s.history, m) for s in S]
print('SNaive mean MASE all 366:', np.mean(ms), 'median', np.median(ms), 'sd', np.std(ms))
iid=[s for s in S if s.start[:4]=='1980']
mi=[mase(s.target, snaive(s.history,H,m), s.history, m) for s in iid]
print('IID(1980) n',len(iid),'SNaive mean',np.mean(mi),'sd',np.std(mi,ddof=1),'SE16 =',np.std(mi,ddof=1)/4*np.sqrt(1-15/len(iid)))
oth=[s for s in S if s.start[:4]!='1980']
mo=[mase(s.target, snaive(s.history,H,m), s.history, m) for s in oth]
print('non-1980 n',len(oth),'mean',np.mean(mo),'sd',np.std(mo,ddof=1), 'SE16',np.std(mo,ddof=1)/4)
# calendar-overlap leak audit: end date of each series (months since 1980-01)
import datetime
def mo_idx(start):
    y,mn=int(start[:4]),int(start[5:7]); return y*12+mn-1
for s in iid[:3]: print(s.id,s.start,len(s.values))
ends={s.id: mo_idx(s.start)+len(s.values)-1 for s in S}
hist_end={s.id: mo_idx(s.start)+len(s.history)-1 for s in S}
tgt_start={s.id: hist_end[s.id]+1 for s in S}
import statistics
# for each iid series: how many other iid series' histories cover (some part of) its target window
cov=[]; covfull=[]
for s in iid:
    a,b=tgt_start[s.id],ends[s.id]
    n_any=sum(1 for t in iid if t.id!=s.id and hist_end[t.id]>=a)
    n_full=sum(1 for t in iid if t.id!=s.id and hist_end[t.id]>=b)
    cov.append(n_any); covfull.append(n_full)
print('IID series whose target window is (partly) inside the visible history of other IID series: mean #others',np.mean(cov),'frac with >=1:',np.mean(np.array(cov)>0),' fully covered by >=1:',np.mean(np.array(covfull)>0), 'median #full',np.median(covfull))
ends_hist=np.array([hist_end[s.id] for s in iid]); print('hist_end (iid) min/max/median', ends_hist.min(), ends_hist.max(), np.median(ends_hist))
ends_all=np.array([ends[s.id] for s in iid]); print('series_end (iid) min/max/median', ends_all.min(), ends_all.max())
print(collections.Counter(ends_all.tolist()).most_common(6))
