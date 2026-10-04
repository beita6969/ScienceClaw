import sys, numpy as np, pandas as pd, collections
R=sys.argv[1]; sys.path.insert(0,R)
from pathlib import Path
from scienceclaw.bench.tasks import for44_acic as A
root=Path('/Users/admin/Datasets/ScienceClaw-rebuild-20260928/datasets')
ad=A.ACIC2016Adapter(data_root=str(root))
print(ad.available())
parts=ad._parts.get()
print({k:len(v) for k,v in parts.items()})
rows=[]
X=ad._X.get()
for role in ('dev','src','val','id'):
    for it in parts[role]:
        s=ad._sim(it)
        ols=A.ols_effect(X,s.z,s.y)
        rows.append(dict(role=role,item=it,p=A.scenario_of(it),satt=s.satt,sdy=s.sd_y,ols=ols,
            ate=float(np.mean(s.mu1-s.mu0)),ytr=s.y[s.z==1].mean()-s.y[s.z==0].mean(),ntr=int(s.z.sum()),
            sd_mu1_mu0=float(np.std(s.mu1-s.mu0))))
df=pd.DataFrame(rows)
df['ols_n']=(df.ols-df.satt)/df.sdy
df['zero_n']=(0-df.satt)/df.sdy
df['diff_n']=(df.ytr-df.satt)/df.sdy
df['mean_satt_n']=(df.satt.mean()-df.satt)/df.sdy
print(df.groupby('role')[['satt','sdy','ntr','sd_mu1_mu0']].describe().T.to_string())
rm=lambda v: float(np.sqrt(np.mean(v**2)))
print('normalised RMSE by role (SATT error / sd(y)):')
for role,g in df.groupby('role'):
    print(role,len(g),'OLS %.4f'%rm(g.ols_n),'zero-effect %.4f'%rm(g.zero_n),'naive diff-in-means %.4f'%rm(g.diff_n),'|SATT|/sd mean %.3f'%np.mean(np.abs(g.satt)/g.sdy))
df['absolute_ols']=df.ols-df.satt
for role,g in df.groupby('role'):
    print(role,'abs (unnormalised) RMSE OLS %.4f  zero %.4f  diff %.4f'%(rm(g.ols-g.satt),rm(g.satt),rm(g.ytr-g.satt)))
df.to_csv('acic_local.csv',index=False)
