"""FoR41: dump every pool item (src: every third item; oxygen and temperature only, float32) with its eval history (item's own target window withheld), the dev history (ends 30 days earlier), targets and dates."""
import pickle, sys
import numpy as np
from scienceclaw.bench.tasks import for41_neon as N

a = N.Adapter(); d = a.data(); H = N.H
pools = {}
for split, ids in d.pools.items():
    rows = []
    for i in (ids[::3] if split == "src" else ids):
        s, ti = d.items[i]
        vis = d.visible[s].copy(); vis[ti + 1:ti + 1 + H] = np.nan
        h, hd = a._window(vis, ti)
        dh, dhd = a._window(vis, ti - H)
        rows.append(dict(id=i, site=s, site_type=N.SITE_TYPE[s], t0=str(d.days[ti]), hist=h[:, :2].astype(np.float32),
                         obs=d.raw[s][ti + 1:ti + 1 + H, :2].copy(), dev_hist=dh[:, :2].astype(np.float32),
                         dev_obs=vis[ti - H + 1:ti + 1, :2].copy()))
    pools[split] = rows
pickle.dump(pools, open(sys.argv[1], "wb"))
print({k: len(v) for k, v in pools.items()}, pools["src"][0]["hist"].shape)
