"""argv: pickle pool_name out.json method [method ...]; methods = scilib.causal names or bart_methods names.
Stores per-dataset ATT estimates (clipped as estimate_effects does: +-4 sd(y)); scoring is done by score.py."""
import json, pickle, sys, time, warnings
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from scilib import causal
from threadpoolctl import threadpool_limits
import bart_methods

d = pickle.load(open(sys.argv[1], "rb")); out = Path(sys.argv[2]); meths = sys.argv[3:]
res = json.load(open(out)) if out.exists() else {}
X = causal.design_matrix(d["cov"])
for m in meths:
    if m in res:
        continue
    t0 = time.time(); est = []
    for z, y in zip(d["Z"], d["Y"]):
        z = z.astype(int)
        with warnings.catch_warnings(), threadpool_limits(limits=1):
            warnings.simplefilter("ignore")
            f = bart_methods.METHODS.get(m) or causal.get_method(m, "att")
            try:
                r = f(X, z, y); r = float(r.tau) if hasattr(r, "tau") else float(r)
            except Exception as ex:
                print("fail", m, ex, flush=True); r = float(causal.regression_adjustment(X, z, y).tau)
        est.append(float(np.clip(r, -4 * y.std(ddof=1), 4 * y.std(ddof=1))))
    res[m] = est
    e = (np.array(est) - d["satt"]) / d["sd_y"]
    print(f"{m:18s} rmse/sd {np.sqrt(np.mean(e**2)):.4f}  {time.time()-t0:.0f}s", flush=True)
    json.dump(res, open(out, "w"))
