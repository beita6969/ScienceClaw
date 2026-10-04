import os, sys, time, pickle, json
os.environ['SCIENCECLAW_TASK_CACHE']='/private/tmp/claude-501/sc-scratch/sota/g2_work/f39cache'
R='/Users/admin/Library/Application Support/Claude/scratch-workspaces/b286c85c-fb50-4909-a7a4-692fbcc9e2e6/67544b4d-5c42-4571-a504-64db11cd447d/scratch-2026-09-28-749df2/scienceclaw'
sys.path.insert(0,R)
import numpy as np
from scienceclaw.bench.tasks.for39_eedi import Adapter, N_MASKS, N_Q
import scilib.adaptive as A
ad=Adapter(data_root='/Users/admin/Datasets/ScienceClaw-rebuild-20260928/datasets')
runs={'id':('final_a0_main_id','b1a7d345'),'ood':('final_a0_main_ood','e22dca2b'),'val':('final_a0_v_val','a3787b77')}
out={}
for split,(rd,hx) in runs.items():
    eps=ad.build_episodes(split,2,int(hx,16))
    for ep in eps:
        yp=f'{R}/runs/{rd}/{ep.id}/y.pkl'
        if not os.path.exists(yp): print('no y',ep.id); continue
        y=pickle.load(open(yp,'rb'))
        y=np.asarray(y)
        r=ep.evaluate(y,None)
        # toolkit default policies through the tools
        T={t.name:t for t in ep.tools}
        tr=T['load_train'].fn({},{})['answers']; ev=T['load_eval_inputs'].fn({},{})
        cq,tg=ev['can_query'],ev['targets']
        res={'agent':r.primary,'ref':r.metrics['reference_accuracy'],'agent_pass':r.accepted}
        model=A.fit_item_curves(tr)
        for meth in ('batch','bald'):
            sel=A.select_queries(model,cq,[],10,10,meth)
            rev=T['query_answers'].fn({'selections':sel},{})['revealed']
            yy=A.predict(model,[rev],tg)
            rr=ep.evaluate(yy,None); res[meth]=rr.primary
            if meth=='batch':
                res['agree_with_toolkit_batch']=float(((yy==y)&tg).sum()/tg.sum())
                res['n_targets']=int(tg.sum()); res['n_items']=int(tg.shape[0])
                res['y_dtype']=str(y.dtype); res['y_shape']=list(y.shape)
        # majority + prior-only
        res['prior_only']=ep.evaluate(A.predict(model,[],tg),None).primary
        out[ep.id]=res
        print(ep.id,json.dumps({k:(round(v,4) if isinstance(v,float) else v) for k,v in res.items()}),flush=True)
json.dump(out,open('/private/tmp/claude-501/sc-scratch/sota/g2_work/f39_eps.json','w'),indent=1)
