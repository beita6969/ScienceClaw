"""Reduced learning curve (1 thread): SoftMaskSeparator trained on n excerpts, evaluated on 24 fixed id excerpts."""
import time, numpy as np
from common import A, item_scores, aggregate
ze=np.load("sub_eval.npz",allow_pickle=True); emix,est_=ze["mix"],ze["stems"]
zt=np.load("sub_train16.npz",allow_pickle=True); zs=np.load("sub_src32.npz",allow_pickle=True)
def run(name, mix, stems, **kw):
    t=time.time(); e=A.separate(mix, stems, emix, n_jobs=1, **kw)
    v=np.array([item_scores(est_[i].astype(np.float64), e[i].astype(np.float64)) for i in range(len(emix))])
    print(f"{name:30s} n_train={len(mix):3d} agg={aggregate(v):5.2f} per-target={[round(float(np.nanmedian(v[:,j])),2) for j in range(4)]} t={time.time()-t:.0f}s", flush=True)
tm,ts=zt["mix"],zt["stems"]
for sd in (0,1):
    idx=np.random.default_rng(10+sd).choice(len(tm),8,replace=False)
    run(f"train-pool 8 (draw {sd})", tm[idx], ts[idx], seed=sd)
run("train-pool 16", tm, ts)
run("src 32 (32 distinct tracks)", zs["mix"], zs["stems"])
