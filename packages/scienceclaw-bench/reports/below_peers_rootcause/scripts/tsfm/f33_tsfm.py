"""FoR33 (server side): Chronos-2 / Bolt median forecasts of the 168 h contexts (eval and dev windows) of every pool; context length 168 or 72 h."""
import pickle, sys, itertools
import numpy as np
from scilib import tsfm

d = pickle.load(open(sys.argv[1], "rb")); out = sys.argv[2]
models = sys.argv[3].split(",")
res = {}
for mdl, ctx in itertools.product(models, [None, 72]):
    for c, rows in d["pools"].items():
        bs = d["split_buildings"][c]
        ev = np.stack([r["context"] for r in rows])
        dv = np.concatenate([d["buildings"][b]["dev_context"] for b in bs])
        for view, x in (("eval", ev), ("dev", dv)):
            res[(mdl, ctx, c, view)] = tsfm.forecast(list(x), 24, model=mdl, context_length=ctx)
    print(mdl, ctx, flush=True)
    pickle.dump(res, open(out, "wb"))
