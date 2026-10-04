import pickle, numpy as np, pandas as pd, warnings, json
warnings.filterwarnings('ignore')
from statsforecast import StatsForecast
from statsforecast.models import Naive, RandomWalkWithDrift, AutoETS, AutoTheta, AutoARIMA, SeasonalNaive
D=pickle.load(open('f38_items.pkl','rb'))
def sm(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float);d=np.abs(y)+np.abs(p)
    return 200*np.mean(np.where(d>0,np.abs(y-p)/np.where(d>0,d,1),0),axis=-1)
def clean(h):
    s=pd.Series(h).interpolate(limit_area='inside'); s=s[s.first_valid_index():]; return s.values
recs=[(k,r) for k,v in D['pools'].items() for r in v]
rows=[]
for i,(k,r) in enumerate(recs):
    v=clean(r['hist'])
    for t,val in enumerate(v): rows.append((str(i),pd.Timestamp(2000+ (t//1),1,1) if False else pd.Timestamp('1900-01-01')+pd.DateOffset(years=t),val))
df=pd.DataFrame(rows,columns=['unique_id','ds','y'])
models=[Naive(),RandomWalkWithDrift(),AutoETS(season_length=1),AutoTheta(season_length=1)]
sf=StatsForecast(models=models,freq='YS',n_jobs=1)
fc=sf.forecast(df=df,h=4).reset_index()
names=['Naive','RWD','AutoETS','AutoTheta']
P={n:np.stack([fc[fc.unique_id==str(i)][n].values for i in range(len(recs))]) for n in names}
# also 5y-drift on log-level and ETS on log for level kind
def drift5(h,k=5,log=False):
    v=clean(h);
    if log: v=np.log(v)
    g=(v[-1]-v[-1-k])/k; out=v[-1]+g*np.arange(1,5); return np.exp(out) if log else out
P['Drift5']=np.stack([np.where(r['ind']=='SL.UEM.TOTL.ZS',drift5(r['hist'],5,False),drift5(r['hist'],5,True)) for _,r in recs])
P['Drift5log_all']=np.stack([drift5(r['hist'],5,True) for _,r in recs])
tg=np.stack([r['tgt'] for _,r in recs]); pool=np.array([k for k,_ in recs]); ind=np.array([r['ind'] for _,r in recs])
for n in P: P[n]=np.clip(P[n],1e-9,None)
S={n:sm(tg,P[n]) for n in P}
def pr(label,mask):
    print(f"{label:22s} n={mask.sum():3d} "+' '.join(f"{n}={S[n][mask].mean():.2f}" for n in P))
IID=np.isin(pool,['val','id','src']); OOD=pool=='ood'
gdp=ind=='NY.GDP.PCAP.KD'; un=~gdp
for lab,m in [('IID all',IID),('IID gdp',IID&gdp),('IID unemp',IID&un),('OOD all',OOD),('OOD gdp',OOD&gdp),('OOD unemp',OOD&un),('val',pool=='val'),('id',pool=='id'),('src',pool=='src')]: pr(lab,m)
# slice noise: random 16-item draws from each pool
rng=np.random.default_rng(1)
print('--- 16-item slice noise (5000 random draws)')
for lab,m in [('id',pool=='id'),('val',pool=='val'),('ood',OOD),('src',pool=='src')]:
    idx=np.where(m)[0]; R={n:[] for n in ['Naive','AutoTheta','AutoETS','Drift5']};
    for _ in range(5000):
        j=rng.choice(idx,16,replace=False)
        for n in R: R[n].append(S[n][j].mean())
    nv=np.array(R['Naive'])
    s=f"{lab:4s} naive mean {nv.mean():.2f} sd {nv.std():.2f} [5,95] {np.percentile(nv,5):.1f}-{np.percentile(nv,95):.1f}"
    for n in ['AutoTheta','AutoETS','Drift5']:
        a=np.array(R[n]); q=a/nv; s+=f" | {n}/naive mean {q.mean():.3f} sd {q.std():.3f} P(<=0.97) {np.mean(q<=0.97):.2f}"
    print(s)
pickle.dump(dict(P=P,S=S,pool=pool,ind=ind,tg=tg,recs=[r['id'] for _,r in recs]),open('f38_base.pkl','wb'))
