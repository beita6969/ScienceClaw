"""FoR33: dump windows (context/target/building/category/starts) of every pool and the visible history of every building."""
import pickle, sys
import numpy as np
from scienceclaw.bench.tasks.for33_buildingsbench import Adapter

d = Adapter().data()
pools = {}
for split, ids in d.pools.items():
    pools[split] = [dict(id=i, building=d.windows[i].building, category=d.windows[i].category, context=d.windows[i].context.astype(np.float64),
                         target=d.windows[i].target.astype(np.float64), context_start=d.windows[i].context_start, target_start=d.windows[i].target_start)
                    for i in ids]
blds = {b: dict(category=x.category, role=x.role, history=x.history.astype(np.float64), history_start=x.history_start,
                dev_context=x.dev_context.astype(np.float64), dev_target=x.dev_target.astype(np.float64),
                dev_context_start=list(x.dev_context_start), dev_target_start=list(x.dev_target_start)) for b, x in d.buildings.items()}
pickle.dump(dict(pools=pools, buildings=blds, split_buildings=d.split_buildings), open(sys.argv[1], "wb"))
print({k: len(v) for k, v in pools.items()}, {k: len(v) for k, v in d.split_buildings.items()}, next(iter(blds.values()))["history"].shape)
