"""FoR35 (server side): Chronos-2 / Chronos-Bolt quantile forecasts (0.1, 0.5, 0.9) of every pool, eval view and dev view; timings per call."""
import pickle, sys, time
import numpy as np
from scilib import tsfm

src, out = sys.argv[1], sys.argv[2]
models = sys.argv[3].split(",") if len(sys.argv) > 3 else ["chronos_2", "chronos_bolt"]
d = pickle.load(open(src, "rb"))
H = 24
res = {}
for mdl in models:
    for c, rows in d["pools"].items():
        ev = [r["hist"] for r in rows]
        dv = [r["hist"][:-H] for r in rows]
        t0 = time.time()
        fe = tsfm.forecast(ev, H, model=mdl)
        fd = tsfm.forecast(dv, H, model=mdl)
        res[(mdl, c)] = {"eval": fe, "dev": fd}
        print(mdl, c, len(rows), fe.shape, round(time.time() - t0, 2), "s", flush=True)
        pickle.dump(res, open(out, "wb"))
