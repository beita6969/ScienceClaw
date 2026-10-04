# hindsight two-number prior: unemployment x median path, GDP-pc x median path (fitted on the targets = oracle upper bound of a world-knowledge prior)
import pickle, numpy as np
B=pickle.load(open('f38_base.pkl','rb')); D=pickle.load(open('f38_items.pkl','rb'))
recs=[r for v in D['pools'].values() for r in v]
last=np.array([r['hist'][-1] for r in recs]); tg=B['tg']; ind=B['ind']; pool=B['pool']
def sm(y,p):
    d=np.abs(y)+np.abs(p); return 200*np.mean(np.where(d>0,np.abs(y-p)/np.where(d>0,d,1),0),axis=-1)
un=ind=='SL.UEM.TOTL.ZS'; iid=np.isin(pool,['val','id','src']); n=sm(tg,np.repeat(last[:,None],4,1))
for lab,fit in [('fit on all',pool!='x'),('fit on IID only',iid)]:
    pu=np.median(tg[fit&un]/last[fit&un][:,None],axis=0); pg=np.median(tg[fit&~un]/last[fit&~un][:,None],axis=0)
    s=sm(tg,np.where(un[:,None],last[:,None]*pu,last[:,None]*pg))
    print(lab,'path un',np.round(pu,3),'gdp',np.round(pg,3),'| IID ratio %.3f OOD ratio %.3f'%(s[iid].mean()/n[iid].mean(),s[pool=='ood'].mean()/n[pool=='ood'].mean()))
