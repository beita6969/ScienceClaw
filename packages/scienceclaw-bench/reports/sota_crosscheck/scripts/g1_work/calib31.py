import pandas as pd, numpy as np, glob, os, re
from scipy.stats import spearmanr
D="/Users/admin/Datasets/ScienceClaw-rebuild-20260928/datasets/for31-proteingym-substitutions/data/DMS_ProteinGym_substitutions"
S="/Users/admin/Datasets/ScienceClaw-rebuild-20260928/reference_code/FoR31/ProteinGym/benchmarks/DMS_supervised/substitutions/Spearman/DMS_substitutions_Spearman_DMS_level_fold_random_5.csv"
off=pd.read_csv(S).set_index("DMS_id")
rows=[]
rng=np.random.default_rng(0)
for f in sorted(glob.glob(D+"/*.csv")):
    a=os.path.basename(f)[:-4]
    df=pd.read_csv(f,usecols=["mutant","DMS_score"])
    df=df[~df.mutant.str.contains(":")].copy()
    m=df.mutant.str.extract(r"^([A-Z])(\d+)([A-Z])$")
    df=df[m[0].notna()].copy(); m=m[m[0].notna()]
    df["pos"]=m[1].astype(int); n=len(df)
    if n<160: continue
    nq,nd=min(128,int(.4*n)),min(32,int(.1*n)); nt=min(1024,n-nq-nd)
    perm=rng.permutation(n); q,t=perm[:nq],perm[nq+nd:nq+nd+nt]
    tr=df.iloc[t]; qq=df.iloc[q]
    mu=tr.groupby("pos").DMS_score.mean(); g=tr.DMS_score.mean()
    p=qq.pos.map(mu).fillna(g).to_numpy(float)
    r=spearmanr(p,qq.DMS_score).statistic
    r=0.0 if not np.isfinite(r) else r
    rows.append(dict(a=a,n=n,ntrain=nt,site=r,ohe=off["One-Hot Encodings"].get(a,np.nan),kermut=off["Kermut"].get(a,np.nan),pnpt=off["ProteinNPT"].get(a,np.nan),tsub="Tsuboyama" in a))
r=pd.DataFrame(rows)
print(len(r),"assays")
for name,m in [("functional",~r.tsub),("tsuboyama",r.tsub)]:
    x=r[m]
    print(name,len(x),"site-mean(ours proto) %.3f | OHE official-random %.3f | Kermut %.3f | PNPT %.3f | corr(site,ohe) %.2f | mean n_train %.0f (frac of n %.2f)"%(x.site.mean(),x.ohe.mean(),x.kermut.mean(),x.pnpt.mean(),np.corrcoef(x.site,x.ohe)[0,1],x.ntrain.mean(),(x.ntrain/x.n).mean()))
r.to_csv("calib31.csv",index=False)
