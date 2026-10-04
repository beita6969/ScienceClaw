import sys, pickle, numpy as np, time, collections
R='/Users/admin/Library/Application Support/Claude/scratch-workspaces/b286c85c-fb50-4909-a7a4-692fbcc9e2e6/67544b4d-5c42-4571-a504-64db11cd447d/scratch-2026-09-28-749df2/scienceclaw'
sys.path.insert(0,R)
from scienceclaw.bench.tasks import for38_worldbank as m
from scilib.macro import fit_predict
ad=m.Adapter(data_root='/Users/admin/Datasets/ScienceClaw-rebuild-20260928/datasets')
cap={'val':2,'id':4,'src':8,'ood':8}
res=collections.defaultdict(list); per_item=collections.defaultdict(lambda:[[],[]])
for split,c in cap.items():
    for seed in range(0,6):
        eps=ad.build_episodes(split,c,seed)
        for ep in eps:
            T={t.name:t for t in ep.tools}
            tr=T['load_train'].fn({},{}); ev=T['load_eval_inputs'].fn({},{})
            y=np.asarray(fit_predict(tr['panel'],indicator_kinds=tr['indicator_kinds'],**ev))
            r=ep.evaluate(y,None)
            res[split].append((r.metrics['mean_smape_pct'],r.metrics['reference_mean_smape_pct'],r.metrics.get('smape_gdp_per_capita_pct'),r.metrics.get('smape_unemployment_pct')))
for s,v in res.items():
    a=np.array(v,float); q=a[:,0]/a[:,1]
    print(f"{s:4s} n_ep={len(a)} lib primary mean {a[:,0].mean():.2f} sd {a[:,0].std():.2f} (range {a[:,0].min():.1f}-{a[:,0].max():.1f}) | naive mean {a[:,1].mean():.2f} sd {a[:,1].std():.2f} | ratio mean {q.mean():.3f} sd {q.std():.3f} P(ratio<=0.97) {np.mean(q<=0.97):.2f} | gdp {np.nanmean(a[:,2]):.2f} unemp {np.nanmean(a[:,3]):.2f}")
pickle.dump(dict(res),open('f38_lib2.pkl','wb'))
