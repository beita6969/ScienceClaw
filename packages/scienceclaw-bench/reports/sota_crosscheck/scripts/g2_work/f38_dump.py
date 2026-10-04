import sys, os, pickle, numpy as np
R='/Users/admin/Library/Application Support/Claude/scratch-workspaces/b286c85c-fb50-4909-a7a4-692fbcc9e2e6/67544b4d-5c42-4571-a504-64db11cd447d/scratch-2026-09-28-749df2/scienceclaw'
sys.path.insert(0,R); os.chdir(R)
from scienceclaw.bench.tasks import for38_worldbank as m
ad=m.Adapter(data_root='/Users/admin/Datasets/ScienceClaw-rebuild-20260928/datasets')
print('avail', ad.available())
d=ad.data()
print({k:len(v) for k,v in d.pools.items()}, 'items',len(d.items))
out={'pools':{},'runs':{}}
def rec(iid):
    c,ind=d.items[iid]; v=d.values[ind][c]
    return dict(id=iid,econ=c,ind=ind,hist=v[:32].copy(),tgt=v[32:36].copy(),region=d.meta[c]['region'],income=d.meta[c]['income'])
for k,ids in d.pools.items(): out['pools'][k]=[rec(i) for i in ids]
runs=[('final_a0_main_id/FoR38-id-sf94d6fe0-e00','id',0xf94d6fe0,0),('final_a0_main_id/FoR38-id-sf94d6fe0-e01','id',0xf94d6fe0,1),
      ('final_a0_main_ood/FoR38-ood-s29efdb4f-e00','ood',0x29efdb4f,0),('final_a0_main_ood/FoR38-ood-s29efdb4f-e01','ood',0x29efdb4f,1),
      ('final_a0_v_val/FoR38-val-s05ad886f-e00','val',0x05ad886f,0)]
for rd,split,seed,k in runs:
    if not os.path.exists(f'runs/{rd}/y.pkl'):
        print('missing',rd); continue
    ep=ad.build_episodes(split,k+1,seed)[k]
    ids=ep.lineage['item_ids']
    y=np.asarray(pickle.load(open(f'runs/{rd}/y.pkl','rb')),float)
    res=ep.evaluate(y,None)
    print(rd, ep.id, len(ids), res.metrics if res else None, 'primary',res.primary if hasattr(res,'primary') else None)
    out['runs'][rd]=dict(ids=ids,y=y,recs=[rec(i) for i in ids],ref=res.metrics.get('reference_mean_smape_pct'),metrics=res.metrics)
pickle.dump(out,open('f38_items.pkl','wb'))
