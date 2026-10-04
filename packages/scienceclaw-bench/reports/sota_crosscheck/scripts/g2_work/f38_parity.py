import json, pickle, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from utilsforecast.losses import smape as u_smape
API='/Users/admin/Datasets/ScienceClaw-rebuild-20260928/datasets/for38-worldbank-wdi/source-api/'
D=pickle.load(open('f38_items.pkl','rb'))
# --- (1) independent data / eligibility rebuild from the raw API JSON
countries=json.load(open(API+'countries.json'))[1]
meta={c['id']:(c['region']['value'].strip(),c['incomeLevel']['value']) for c in countries if c['region']['value'].strip()!='Aggregates'}
def load(ind):
    out={}
    for r in json.load(open(API+ind+'.json'))[1]:
        c=r['countryiso3code']
        if c in meta and 1990<=int(r['date'])<=2025 and r['value'] is not None: out.setdefault(c,{})[int(r['date'])]=float(r['value'])
    return out
elig={}
for ind in ['NY.GDP.PCAP.KD','SL.UEM.TOTL.ZS']:
    for c,s in load(ind).items():
        h=[s.get(y,np.nan) for y in range(1990,2022)]; t=[s.get(y,np.nan) for y in range(2022,2026)]
        allv=[v for v in s.values()]
        if np.isfinite(h[-1]) and np.isfinite(h).sum()>=20 and np.all(np.isfinite(t)) and min(allv)>0: elig[(c,ind)]=(np.array(h),np.array(t))
print('independent eligible items',len(elig), 'adapter items', sum(len(v) for v in D['pools'].values()))
ad_keys={(r['econ'],r['ind']) for v in D['pools'].values() for r in v}
print('same key set:', ad_keys==set(elig))
mm=0
for v in D['pools'].values():
    for r in v:
        h,t=elig[(r['econ'],r['ind'])]
        mm+=int(not (np.allclose(h,r['hist'],equal_nan=True) and np.allclose(t,r['tgt'])))
print('hist/tgt mismatches',mm)
# --- (2) sMAPE parity: repo smape vs utilsforecast (fraction 0-1 -> x200) on real naive forecasts + agent forecasts + random
import sys
def repo_smape(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float);den=np.abs(y)+np.abs(p);num=np.abs(y-p)
    return float(200*np.mean(np.where(den>0,num/np.where(den>0,den,1),0)))
def u200(y,p):
    df=pd.DataFrame({'unique_id':0,'ds':np.arange(len(y)),'y':y,'m':p})
    return float(200*u_smape(df,models=['m'])['m'].iloc[0]) if True else None
# utilsforecast smape definition check
df=pd.DataFrame({'unique_id':0,'ds':range(4),'y':[1.,2,3,4],'m':[2.,2,1,8]})
print('utilsforecast raw smape',u_smape(df,models=['m'])['m'].iloc[0],' by hand 0-1 form', np.mean(np.abs(df.y-df.m)/(df.y.abs()+df.m.abs())))
rng=np.random.default_rng(0); mx=0
for _ in range(2000):
    y=rng.uniform(0.5,50,4); p=y*np.exp(rng.normal(0,0.5,4)); mx=max(mx,abs(repo_smape(y,p)-u200(y,p)))
print('random max |repo - utilsforecast*200|',mx)
for rd,r in D['runs'].items():
    per=[];
    for j,rec in enumerate(r['recs']):
        per.append(u200(rec['tgt'],r['y'][j]))
    nav=[u200(rec['tgt'],np.full(4,rec['hist'][-1])) for rec in r['recs']]
    print(rd.split('/')[1], 'agent(indep)',round(np.mean(per),4),'harness',round(r['metrics']['mean_smape_pct'],4),'| naive(indep)',round(np.mean(nav),4),'harness ref',round(r['ref'],4))
