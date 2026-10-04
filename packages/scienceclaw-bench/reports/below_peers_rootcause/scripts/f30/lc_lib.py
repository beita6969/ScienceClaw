"""Helpers on the compacted cached features (t1_features.py + t2b_compact.py). Same LightGBM params / post-processing as scilib.phenoseg
(read-only import); only the per-image training sample is fixed (4000 px/class, drawn once) instead of re-drawn per fit."""
import os, json, time
from common import *
import lightgbm as lgb
from scipy import ndimage as ndi
FD = WORK + "/feats_c"
def fkey(i): return FD + "/" + i.replace("/", "__") + ".npz"
def fit(ids, seed=0, n_estimators=200):
    Xs, ys, sz = [], [], []
    for i in ids:
        z = np.load(fkey(i)); Xs.append(z["X"]); ys.append(z["y"]); cs = float(z["crop_scale"])
        if cs > 0: sz.append(cs)
    X, y = np.concatenate(Xs), np.concatenate(ys).astype(np.int64)
    t = time.time()
    b = lgb.train(dict(ps._LGB, seed=int(seed)), lgb.Dataset(X, y), num_boost_round=n_estimators)
    return ps.PixelModel(b, float(np.median(sz)) if sz else 10.0, b.current_iteration(), time.time() - t)
def probs(model, ids):
    out = []
    for i in ids:
        lat = np.load(fkey(i))["lat4"]; h, w, d = lat.shape
        p = model.predict_proba(lat.reshape(-1, d)).reshape(h, w, 3)
        out.append(np.clip(ndi.zoom(p, (512 / h, 512 / w, 1), order=1), 0, 1).astype(np.float16))
    return np.stack(out)
def evaluate(a, ids, P, plant_size, params=None):
    pred = ps.panoptic_from_probs(P, dict(ps.DEFAULT_PARAMS, plant_size=plant_size, **(params or {})))
    return score(a, ids, pred)
