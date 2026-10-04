"""FoR35 (server side): the pre-declared Chronos-2 variant (context 120, log1p, median) on the eval view of the named cohorts."""
import pickle, sys
import numpy as np
from scilib import tsfm

d = pickle.load(open(sys.argv[1], "rb")); out = sys.argv[2]; res = {}
for c in sys.argv[3].split(","):
    hs = [r["hist"] for r in d["pools"][c]]
    res[c] = np.expm1(tsfm.forecast([np.log1p(np.maximum(h, 0)) for h in hs], 24, model="chronos_2", context_length=120))[:, :, 1]
    print(c, res[c].shape, flush=True)
pickle.dump(res, open(out, "wb"))
