"""FoR35: dump the series of every pool (src/val/id/ood) with the eval view and the dev view, plus the visible training histories."""
import os, pickle, sys
import numpy as np
from scienceclaw.bench.tasks.for35_tourism import Adapter

OUT = sys.argv[1]
a = Adapter()
d = a.data()
H, m = 24, 12
pools = {}
for split, ids in d.pools.items():
    rows = []
    for i in ids:
        s = d.series[i]
        rows.append(dict(id=i, hist=s.history.astype(np.float64), target=s.target.astype(np.float64), period=s.period, horizon=s.horizon))
    pools[split] = rows
train = [d.series[i].history.astype(np.float64) for i in d.iid_ids]
pickle.dump(dict(pools=pools, train=train, train_ids=list(d.iid_ids), ood_kind=d.ood_kind), open(OUT, "wb"))
print({k: len(v) for k, v in pools.items()}, len(train), d.ood_kind)
