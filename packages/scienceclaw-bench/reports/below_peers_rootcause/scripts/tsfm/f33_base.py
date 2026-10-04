"""FoR33: scilib.loadforecast candidates (fit on the visible histories of the pool's buildings) for every pool, eval and dev views."""
import pickle, sys, time
import numpy as np
from scilib import loadforecast as L

d = pickle.load(open(sys.argv[1], "rb")); out = sys.argv[2]
res = {}
for c in sys.argv[3].split(","):
    bs = d["split_buildings"][c]
    B = d["buildings"]
    load = np.stack([B[b]["history"] for b in bs]); hs = [B[b]["history_start"] for b in bs]; cat = [B[b]["category"] for b in bs]
    t0 = time.time()
    model = L.fit(load, hs, bs, cat)
    rows = d["pools"][c]
    ev = L.forecast_candidates(load, hs, bs, cat, np.stack([r["context"] for r in rows]), [r["target_start"] for r in rows],
                               [r["building"] for r in rows], [r["category"] for r in rows], model=model)
    dv_c = np.concatenate([B[b]["dev_context"] for b in bs]); dv_t = [s for b in bs for s in B[b]["dev_target_start"]]
    dv_b = [b for b in bs for _ in range(len(B[b]["dev_target"]))]
    dv = L.forecast_candidates(load, hs, bs, cat, dv_c, dv_t, dv_b, [B[b]["category"] for b in dv_b], model=model)
    res[c] = {"eval": {k: np.asarray(v) for k, v in ev.items()}, "dev": {k: np.asarray(v) for k, v in dv.items()}}
    print(c, len(rows), len(dv_b), round(time.time() - t0, 1), "s", list(ev), flush=True)
    pickle.dump(res, open(out, "wb"))
