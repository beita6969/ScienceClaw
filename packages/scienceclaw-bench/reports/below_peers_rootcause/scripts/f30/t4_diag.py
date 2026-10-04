"""Diagnostics (1 process): (1) confusion of learned protocol-24 model, argmax vs default post-processing; (2) oracle-semantics ablation of the
post-processing parameters (tuned on 8 src images, applied to 8 id / 8 ood); (3) consistency check of my cached-feature replicate against the
real ps.fit_predict (stride 2, its own pixel sample) on the same 24 training images."""
import sys, time; sys.path.insert(0, '/private/tmp/claude-501/sc-scratch/fix/f30_work')
from lc_lib import *
T0 = time.time(); out = WORK + "/out/t4.jsonl"
def log(**r):
    r["t"] = round(time.time() - T0); open(out, "a").write(json.dumps(r) + "\n"); print(json.dumps(r)[:600], flush=True)
a = adapter(2); p = a._pools(); test = json.load(open(WORK + "/out/test_ids.json")); g = groups(a)["src"]
tr = g["P0030692"] + g["P0030947"]
def conf(ids, sem):
    cm = np.zeros((3, 3), np.int64)
    for j, i in enumerate(ids):
        cm += F30.semantic_confusion(sem[j], a._load(i)["semantics"])
    return cm.tolist()          # rows = GT (soil, crop, weed), cols = prediction (verify orientation below)
# (1) confusion
m = fit(tr, seed=0)
for k, ids in test.items():
    P = probs(m, ids); prm = dict(ps.DEFAULT_PARAMS, plant_size=m.plant_size)
    am = P.astype(np.float32).argmax(-1).astype(np.int16)
    pan = ps.panoptic_from_probs(P, prm)
    log(stage="confusion", pool=k, argmax=conf(ids, am), postproc=conf(ids, pan["semantics"]),
        gt_frac=[float(np.mean(np.concatenate([ps._merge_partial(a._load(i)["semantics"]).ravel() for i in ids]) == c)) for c in range(3)])
# (2) oracle-semantics ablation
rng = np.random.default_rng(1); src8 = sorted(rng.choice(p["src"], 8, replace=False).tolist())
PS = 8.97
def orc(ids):
    sem = np.stack([ps._merge_partial(a._load(i)["semantics"]) for i in ids]); return np.eye(3, dtype=np.float16)[sem]
O = {"src": (src8, orc(src8)), "id": (test["id"][:8], orc(test["id"][:8])), "ood": (test["ood"][:8], orc(test["ood"][:8]))}
grid = [dict(), dict(smooth=0.3, veg_thr=0.5, crop_weight=1.0), dict(smooth=0.3, veg_thr=0.5, crop_weight=1.0, leaf_sigma=0.5, leaf_dist=3, leaf_area=6),
        dict(smooth=0.3, veg_thr=0.5, crop_weight=1.0, leaf_sigma=1.5, leaf_dist=5, leaf_area=16),
        dict(smooth=0.3, veg_thr=0.5, crop_weight=1.0, plant_sigma=4.0, plant_dist=20, plant_area=150),
        dict(smooth=0.3, veg_thr=0.5, crop_weight=1.0, leaf_sigma=0.5, leaf_dist=3, leaf_area=6, plant_sigma=4.0, plant_dist=20, plant_area=150),
        dict(smooth=0.3, veg_thr=0.5, crop_weight=1.0, overlap_scale=99.0), dict(smooth=0.3, veg_thr=0.5, crop_weight=1.0, overlap_scale=0.0)]
for prm in grid:
    row = dict(stage="oracle_ablate", params=prm)
    for k, (ids, P) in O.items(): row[k] = evaluate(a, ids, P, PS, prm)
    log(**row)
# (3) real toolkit fit_predict (default stride 2) on the protocol-24 images vs my replicate
L = {k: np.stack([a._load(i)[k] for i in tr]) for k in ("images", "semantics")}
for k, ids in test.items():
    imgs = np.stack([a._load(i)["images"] for i in ids]); row = dict(stage="real_fit_predict", pool=k)
    if k == "id": mod = ps.fit_pixel_classifier(L["images"], L["semantics"], seed=0)
    row.update(score(a, ids, ps.panoptic_from_probs(ps.predict_probs(mod, imgs, stride=2), dict(ps.DEFAULT_PARAMS, plant_size=mod.plant_size))))
    log(**row)
log(stage="done")
