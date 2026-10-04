"""Score one-shot pool predictions with the task's own scorer (Adapter._score: mean over cases of the mean Dice of labels 1 and 2, plus NSD)
and the location-atlas reference on the same cases. Mac side; usage: python scripts/f32/pool_score.py out.npz"""
import os, sys, json
import numpy as np
sys.path.insert(0, ".")
from scienceclaw.bench.tasks.for32_msd_hippocampus import Adapter, atlas_predict

a = Adapter(cache_dir=os.environ.get("SCIENCECLAW_TASK_CACHE", "/tmp/scienceclaw-task-cache"))
z = np.load(sys.argv[1]); res = {}
for pool in ("val", "id", "ood"):
    sel = [i for i, q in enumerate(z["pools"]) if q == pool]
    cases = [str(z["ids"][i]) for i in sel]
    preds = [z[f"pred_{i:03d}"] for i in sel]
    s = a._score(cases, preds)
    ref = a._score(cases, [atlas_predict(a._atlas(), p.shape) for p in preds])
    sc = np.array(s["dsc_cases"]).mean(axis=1)
    rng = np.random.default_rng(0); bs = [rng.choice(sc, len(sc)).mean() for _ in range(2000)]
    res[pool] = {"n": len(sc), "mean_dsc": round(s["mean_dsc"], 4), "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
                 "ant": round(s["dsc_anterior"], 4), "post": round(s["dsc_posterior"], 4), "nsd_1mm": round(s["mean_nsd_1mm"], 4),
                 "min_case": round(float(sc.min()), 4), "median_case": round(float(np.median(sc)), 4), "reference_atlas": round(ref["mean_dsc"], 4)}
print(json.dumps(res, indent=1))
