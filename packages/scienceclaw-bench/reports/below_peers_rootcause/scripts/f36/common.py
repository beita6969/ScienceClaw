import os, sys, numpy as np
REPO="/Users/admin/Library/Application Support/Claude/scratch-workspaces/b286c85c-fb50-4909-a7a4-692fbcc9e2e6/67544b4d-5c42-4571-a504-64db11cd447d/scratch-2026-09-28-749df2/scienceclaw"
os.environ["PYTHONDONTWRITEBYTECODE"]="1"; sys.dont_write_bytecode=True
os.environ["SCIENCECLAW_TASK_CACHE"]="/private/tmp/claude-501/sc-scratch/fix/f36_work/taskcache"
sys.path.insert(0, REPO)
from scienceclaw.bench.tasks.for36_musdb import MusdbAdapter, item_scores, aggregate, framewise_sdr, SR
from scienceclaw.bench.tasks import for36_musdb as M
import scilib.audiosep as A
ROOT="/Users/admin/Datasets/ScienceClaw-rebuild-20260928/datasets"
_ad=None
def adapter():
    global _ad
    if _ad is None: _ad=MusdbAdapter(data_root=ROOT)
    return _ad
def pool_excerpts(pool):
    """list of (item_id, track_id, mix (n,2), stems (4,n,2)) for a pool (src,val,id,train,dev)"""
    ad=adapter(); data=ad._data.get(); out=[]
    for iid in data.pools[pool]:
        ex=data.excerpts[iid]; m,s=ad._clip(ex)
        out.append((iid, ex.track.track_id, m, s))
    return out
