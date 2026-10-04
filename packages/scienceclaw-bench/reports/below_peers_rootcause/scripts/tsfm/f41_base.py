"""FoR41: scilib.aquatics.fit_predict (default recipe, groups of 16 items as in an episode) on every pool, eval and dev views."""
import pickle, sys, time
import numpy as np
from scilib import aquatics as aq

pools = pickle.load(open(sys.argv[1], "rb")); out = sys.argv[2]
res = {}
for c in sys.argv[3].split(","):
    rows = pools[c]; t0 = time.time()
    res[c] = {}
    for view in ("eval", "dev"):
        mu = {"oxygen": [], "temperature": []}; sg = {"oxygen": [], "temperature": []}
        for g in range(0, len(rows), 16):
            part = rows[g:g + 16]
            hist = np.stack([np.concatenate([r["hist" if view == "eval" else "dev_hist"], np.full((r["hist"].shape[0], 1), np.nan, np.float32)], 1) for r in part]).astype(float)
            ref = [str(np.datetime64(r["t0"], "D") - (np.timedelta64(30, "D") if view == "dev" else np.timedelta64(0, "D"))) for r in part]
            p = aq.fit_predict(hist, ref)
            for v in mu:
                mu[v].append(p[f"{v}_mu"]); sg[v].append(p[f"{v}_sigma"])
        res[c][view] = {f"{v}_{k}": np.concatenate(a) for v in mu for k, a in (("mu", mu[v]), ("sigma", sg[v]))}
    print(c, len(rows), round(time.time() - t0, 1), "s", flush=True)
    pickle.dump(res, open(out, "wb"))
