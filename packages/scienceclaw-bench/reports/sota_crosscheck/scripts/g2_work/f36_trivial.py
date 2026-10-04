import pickle, numpy as np, re
src=open('/Users/admin/Library/Application Support/Claude/scratch-workspaces/b286c85c-fb50-4909-a7a4-692fbcc9e2e6/67544b4d-5c42-4571-a504-64db11cd447d/scratch-2026-09-28-749df2/scienceclaw/scienceclaw/bench/tasks/for36_musdb.py').read()
a=src.index('def framewise_sdr'); b=src.index('# ====', a)
ns={'np':np,'SR':22050,'WIN_S':1.0}; exec(src[a:b],ns)
item_scores,aggregate=ns['item_scores'],ns['aggregate']
D=pickle.load(open('f36_items.pkl','rb'))
names=['vocals','drums','bass','other']
for key,d in D.items():
    mix,stems,y=d['mix'].astype(np.float64),d['stems'].astype(np.float64),d['y']
    n=len(mix)
    def score(estf):
        per=np.array([item_scores(stems[i],estf(i)) for i in range(n)])
        return aggregate(per), [float(np.nanmedian(per[:,j])) for j in range(4)]
    ref=score(lambda i: np.repeat(mix[i][None],4,axis=0))[0]
    ag=score(lambda i:y[i])[0]
    tiny=score(lambda i: 1e-3*np.repeat(mix[i][None],4,axis=0))[0]
    # oracle per-item per-stem scalar gain on mixture
    def orac(i):
        out=[]
        for j in range(4):
            g=np.sum(stems[i,j]*mix[i])/(np.sum(mix[i]**2)+1e-12); out.append(g*mix[i])
        return np.stack(out)
    og=score(orac)[0]
    # leave-one-out fixed gain per stem (energy fraction from other items in this episode)
    G=np.array([[np.sum(stems[i,j]*mix[i])/(np.sum(mix[i]**2)+1e-12) for j in range(4)] for i in range(n)])
    loo=score(lambda i: np.stack([np.mean(np.delete(G[:,j],i))*mix[i] for j in range(4)]))[0]
    print(f"{key}: MIX ref {ref:.2f} | 1e-3*mix {tiny:.2f} | LOO fixed-gain*mix {loo:.2f} | oracle gain*mix {og:.2f} | agent {ag:.2f} | accept threshold(ref+3) {ref+3:.2f}")
