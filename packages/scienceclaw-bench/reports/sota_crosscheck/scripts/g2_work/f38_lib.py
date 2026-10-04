import sys, pickle, numpy as np, time
R='/Users/admin/Library/Application Support/Claude/scratch-workspaces/b286c85c-fb50-4909-a7a4-692fbcc9e2e6/67544b4d-5c42-4571-a504-64db11cd447d/scratch-2026-09-28-749df2/scienceclaw'
sys.path.insert(0,R)
from scienceclaw.bench.tasks import for38_worldbank as m
from scilib.macro import fit_predict
ad=m.Adapter(data_root='/Users/admin/Datasets/ScienceClaw-rebuild-20260928/datasets')
runs=[('final_a0_main_id/FoR38-id-sf94d6fe0-e00','id',0xf94d6fe0,0),('final_a0_main_id/FoR38-id-sf94d6fe0-e01','id',0xf94d6fe0,1),('final_a0_main_ood/FoR38-ood-s29efdb4f-e00','ood',0x29efdb4f,0),('final_a0_main_ood/FoR38-ood-s29efdb4f-e01','ood',0x29efdb4f,1),('final_a0_v_val/FoR38-val-s05ad886f-e00','val',0x05ad886f,0)]
for rd,split,seed,k in runs:
    ep=ad.build_episodes(split,k+1,seed)[k]
    T={t.name:t for t in ep.tools}
    tr=T['load_train'].fn({},{}); ev=T['load_eval_inputs'].fn({},{})
    t0=time.time(); yd=fit_predict(tr['panel'],indicator_kinds=tr['indicator_kinds'],**ev)
    y=np.asarray(pickle.load(open(f'{R}/runs/{rd}/y.pkl','rb')),float)
    r=ep.evaluate(np.asarray(yd),None)
    print(rd.split('/')[1],'default fit_predict primary %.4f | agent %.4f | max rel diff y %.4f | %.1fs'%(r.metrics['mean_smape_pct'],ep.evaluate(y,None).metrics['mean_smape_pct'],np.max(np.abs(yd/y-1)),time.time()-t0))
