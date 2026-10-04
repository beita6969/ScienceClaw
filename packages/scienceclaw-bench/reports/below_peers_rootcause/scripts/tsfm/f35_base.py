"""FoR35: forecasts of the existing library (scilib.forecast.fit_predict, default recipe) and seasonal naive for every pool, eval and dev views."""
import pickle, sys, time
import numpy as np
from scilib import forecast as F

src, out = sys.argv[1], sys.argv[2]
cohorts = sys.argv[3].split(",") if len(sys.argv) > 3 else ["src", "val", "id", "ood"]
d = pickle.load(open(src, "rb"))
H, m = 24, 12
train = d["train"]
res = {}
for c in cohorts:
    rows = d["pools"][c]
    ev = [r["hist"] for r in rows]
    dv = [r["hist"][:-H] for r in rows]
    t0 = time.time()
    fe = F.fit_predict(ev, H, m, train=train)
    fd = F.fit_predict(dv, H, m, train=train)
    res[c] = {"eval": np.asarray(fe), "dev": np.asarray(fd)}
    print(c, len(rows), round(time.time() - t0, 1), "s", flush=True)
    pickle.dump(res, open(out, "wb"))
