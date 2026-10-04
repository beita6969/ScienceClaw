import numpy as np
from common import aggregate
z=np.load("oracle22_items.npz")
rng=np.random.default_rng(0)
for pool in ["id","val","src"]:
    for k in ["0.5","1.0","mix"]:
        v=z[f"{pool}_{k}"]
        print(pool,k,"n",len(v),"pooled", round(aggregate(v),2), "per-target", [round(float(np.nanmedian(v[:,j])),2) for j in range(4)], "nanitems", int(np.isnan(v).any(1).sum()))
# slice noise: 16-item draws from id pool, SD of pooled score
for k in ["0.5","1.0","mix"]:
    v=z[f"id_{k}"]; s=[aggregate(v[rng.choice(len(v),16,replace=False)]) for _ in range(2000)]
    print("id",k,"16-draw mean",round(np.mean(s),2),"sd",round(np.std(s),2))
