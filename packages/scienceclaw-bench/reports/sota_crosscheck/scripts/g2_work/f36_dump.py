import sys, pickle, numpy as np, os
sys.path.insert(0, os.getcwd())
os.environ.setdefault('SCIENCECLAW_TASK_CACHE','/private/tmp/claude-501/sc-scratch/sota/g2_work/taskcache')
import shutil
# reuse repo cache read-only by copying the FoR36 excerpt cache dir if not present
src='cache/tasks/FoR36'; dst=os.environ['SCIENCECLAW_TASK_CACHE']+'/FoR36'
if not os.path.exists(dst):
    os.makedirs(os.path.dirname(dst),exist_ok=True); shutil.copytree(src,dst)
from scienceclaw.bench.tasks.for36_musdb import MusdbAdapter, item_scores, aggregate, framewise_sdr, SR
ad=MusdbAdapter(data_root='/Users/admin/Datasets/ScienceClaw-rebuild-20260928/datasets')
print(ad.available())
runs=[('final_a0_main_id/FoR36-id-c8b64f24-00',3367391012,'id',0),('final_a0_main_id/FoR36-id-c8b64f24-01',3367391012,'id',1),('final_a0_v_val/FoR36-val-56a7997e-00',1453824382,'val',0)]
out={}
for rd,seed,split,k in runs:
    eps=ad.build_episodes(split,k+1,seed)
    ep=eps[k]; print(ep.id, ep.lineage['tracks'][:3], len(ep.lineage['item_ids']))
    data=ad._data.get()
    ex=[data.excerpts[i] for i in ep.lineage['item_ids']]
    clips=[ad._clip(e) for e in ex]
    mix=np.stack([c[0] for c in clips]); stems=np.stack([c[1] for c in clips])
    y=pickle.load(open(f'runs/{rd}/y.pkl','rb'))
    y=np.asarray(y,dtype=np.float64)
    print(rd,'y',y.shape,'mix',mix.shape,'stems',stems.shape)
    res=ep.evaluate(y,None)
    print('eval:', res.metrics if res else None)
    out[rd]=dict(mix=mix,stems=stems,y=y,ids=ep.lineage['item_ids'])
pickle.dump(out,open('/private/tmp/claude-501/sc-scratch/sota/g2_work/f36_items.pkl','wb'))
