"""Export design matrix, treatment/outcome and truth of one pool (argv[2]: dev|src|val|id) of FoR44 items to a pickle."""
import pickle, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from scienceclaw.bench.tasks.for44_acic import ACIC2016Adapter

ad = ACIC2016Adapter()
parts = ad._parts.get()
print({k: len(v) for k, v in parts.items()})
pool = sys.argv[2]
ids = parts[pool]
sims = [ad._sim(i) for i in ids]
out = {"ids": ids, "cov": ad._require_x(), "Z": np.stack([s.z for s in sims]), "Y": np.stack([s.y for s in sims]),
       "satt": np.array([s.satt for s in sims]), "sd_y": np.array([s.sd_y for s in sims])}
print(pool, len(ids), out["Z"].shape)
pickle.dump(out, open(sys.argv[1], "wb"))
