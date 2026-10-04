import sys, numpy as np, collections, pickle
sys.path.insert(0, ".")
from scienceclaw.bench.tasks import for52_psych201 as m
ad=m.Psych201Adapter()
D=ad._data.get()
print({k:len(v) for k,v in D.pools.items()}, "iid studies",len(D.iid_studies),"ood studies",len(D.ood_studies))
S=D.sessions
def last(h,opts):
    p=[r for r in m.previous_responses(h) if r in opts]; return p[-1] if p else opts[0]
rows=[]
for pool in ("train","dev","src","val","id","ood"):
    for u in D.pools[pool]:
        s=S[u]; h=s.history; o=s.options
        ref=m.reference_prediction(h,o); ls=last(h,o)
        rows.append(dict(pool=pool,study=s.study,uid=u,flag=s.flagged,nopt=len(o),ref=ref==s.target,last=ls==s.target,first=o[0]==s.target,chance=1/len(o),
                         nprior=len(m.previous_responses(h)),target=s.target,opts=tuple(o),hist_len=len(h)))
import pandas as pd
df=pd.DataFrame(rows); df.to_pickle("/private/tmp/claude-501/sc-scratch/sota/g5_work/f52_rows.pkl")
g=df.groupby("pool").agg(n=("ref","size"),ref=("ref","mean"),last=("last","mean"),first=("first","mean"),chance=("chance","mean"),flag=("flag","mean"),nstud=("study","nunique"),nprior=("nprior","median"))
print(g.round(3))
print("options distribution:",df.nopt.value_counts().to_dict())
