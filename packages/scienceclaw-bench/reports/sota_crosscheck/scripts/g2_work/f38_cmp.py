import pickle, numpy as np
D=pickle.load(open('f38_items.pkl','rb')); B=pickle.load(open('f38_base.pkl','rb'))
idx={i:j for j,i in enumerate(B['recs'])}
def sm(y,p):
    d=np.abs(y)+np.abs(p); return 200*np.mean(np.where(d>0,np.abs(y-p)/np.where(d>0,d,1),0),axis=-1)
print('run | n_gdp n_un | agent all/gdp/un | naive | ETS | Theta | RWD  | agent/naive  ETS/naive Theta/naive')
tot={k:[] for k in ['agent','Naive','AutoETS','AutoTheta','RWD']}
for rd,r in D['runs'].items():
    js=[idx[i] for i in r['ids']]; tg=B['tg'][js]; ind=B['ind'][js]; g=ind=='NY.GDP.PCAP.KD'
    a=sm(tg,r['y']); res={'agent':a}
    for n in ['Naive','AutoETS','AutoTheta','RWD']: res[n]=B['S'][n][js]
    f=lambda x:f"{x.mean():.2f}/{x[g].mean():.2f}/{x[~g].mean():.2f}"
    print(rd.split('/')[1],g.sum(),(~g).sum(),'|',' | '.join(f(res[n]) for n in res),'| ratios',f"{res['agent'].mean()/res['Naive'].mean():.3f} {res['AutoETS'].mean()/res['Naive'].mean():.3f} {res['AutoTheta'].mean()/res['Naive'].mean():.3f}")
    for k in tot: tot[k].append(res[k])
print('mean over 5 eps: '+' '.join(f"{k}={np.mean([x.mean() for x in v]):.2f}" for k,v in tot.items()))
# agent forecast sanity: direction vs naive on unemployment
for rd,r in D['runs'].items():
    js=[idx[i] for i in r['ids']]; ind=B['ind'][js]; un=ind!='NY.GDP.PCAP.KD'
    last=np.array([rc['hist'][-1] for rc in r['recs']]); tg=B['tg'][js]
    chg_pred=(r['y'][:,3]/last-1)[un]; chg_true=(tg[:,3]/last-1)[un]
    print(rd.split('/')[1],'unemp h4 change: pred mean %.3f  true mean %.3f  corr %.2f'%(chg_pred.mean(),chg_true.mean(),np.corrcoef(chg_pred,chg_true)[0,1]))
