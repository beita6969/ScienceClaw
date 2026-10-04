"""FoR35 (server side): pre-declared grid on src and val only: model x context length x (raw | log1p) -> median forecasts, eval and dev views."""
import pickle, sys, itertools
import numpy as np
from scilib import tsfm

src, out = sys.argv[1], sys.argv[2]
d = pickle.load(open(src, "rb"))
H = 24
res = {}
for mdl, ctx, tr in itertools.product(sys.argv[3].split(","), [None, 120, 72], ["raw", "log1p"]):
    for c in ("src", "val"):
        rows = d["pools"][c]
        for view in ("eval", "dev"):
            hs = [r["hist"] if view == "eval" else r["hist"][:-H] for r in rows]
            if tr == "log1p":
                f = np.expm1(tsfm.forecast([np.log1p(np.maximum(h, 0)) for h in hs], H, model=mdl, context_length=ctx))
            else:
                f = tsfm.forecast(hs, H, model=mdl, context_length=ctx)
            res[(mdl, ctx, tr, c, view)] = f[:, :, 1]
    print(mdl, ctx, tr, flush=True)
pickle.dump(res, open(out, "wb"))
