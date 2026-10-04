import time, numpy as np, pickle
from common import *
t=time.time()
ad=adapter(); print(ad.available())
data=ad._data.get()
print({k:len(v) for k,v in data.pools.items()}, data.track_counts)
out={}
for pool in ["id","val","src","train","dev"]:
    t0=time.time(); ex=pool_excerpts(pool)
    ids=[e[0] for e in ex]; tr=[e[1] for e in ex]
    mix=np.stack([e[2] for e in ex]).astype(np.float32); st=np.stack([e[3] for e in ex]).astype(np.float32)
    np.savez(f"pool_{pool}.npz", mix=mix, stems=st, ids=np.array(ids), tracks=np.array(tr))
    print(pool, mix.shape, st.shape, round(time.time()-t0,1),"s", flush=True)
print("total", time.time()-t)
