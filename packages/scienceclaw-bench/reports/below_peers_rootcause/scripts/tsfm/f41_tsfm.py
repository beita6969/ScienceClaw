"""FoR41 (server side): Chronos quantiles (0.1, 0.25, 0.5, 0.75, 0.9) of oxygen and temperature (30 days), eval and dev views, context 1461 or 365 days."""
import pickle, sys, itertools, time
import numpy as np
from scilib import tsfm

pools = pickle.load(open(sys.argv[1], "rb")); out = sys.argv[2]
models = sys.argv[3].split(","); cohorts = sys.argv[4].split(",")
Q = (0.1, 0.25, 0.5, 0.75, 0.9)
res = {}
for mdl, ctx in itertools.product(models, [None, 365]):
    for c in cohorts:
        t0 = time.time()
        for view, key in (("eval", "hist"), ("dev", "dev_hist")):
            for vi, v in enumerate(("oxygen", "temperature")):
                res[(mdl, ctx, c, view, v)] = tsfm.forecast([r[key][:, vi] for r in pools[c]], 30, quantiles=Q, model=mdl, context_length=ctx)
        print(mdl, ctx, c, round(time.time() - t0, 1), "s", flush=True)
    pickle.dump(res, open(out, "wb"))
