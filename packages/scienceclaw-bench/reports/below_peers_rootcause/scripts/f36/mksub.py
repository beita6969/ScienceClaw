import numpy as np, gc
def take(pool, idx):
    z=np.load(f"pool_{pool}.npz", allow_pickle=True)
    ids=z["ids"]; tr=z["tracks"]
    mix=z["mix"][idx].copy(); st=z["stems"][idx].copy()
    return mix, st, ids[idx], tr[idx]
n_id=len(np.load("pool_id.npz", allow_pickle=True)["ids"])
rng=np.random.default_rng(1)
ev=np.sort(rng.choice(n_id,48,replace=False))[:24]
m,s,i,t=take("id",ev); np.savez("sub_eval.npz",mix=m,stems=s,ids=i,tracks=t); del m,s; gc.collect()
# train pool: 16 excerpts from distinct tracks where possible
z=np.load("pool_train.npz", allow_pickle=True); tr=z["tracks"]
r=np.random.default_rng(3); order=r.permutation(len(tr)); seen=set(); idx=[]
for k in order:
    if tr[k] not in seen: seen.add(tr[k]); idx.append(k)
idx=np.array(sorted(idx[:16]))
m,s,i,t=take("train",idx); np.savez("sub_train16.npz",mix=m,stems=s,ids=i,tracks=t); del m,s; gc.collect()
z=np.load("pool_src.npz", allow_pickle=True); tr=z["tracks"]
order=np.random.default_rng(5).permutation(len(tr)); seen=set(); idx=[]
for k in order:
    if tr[k] not in seen: seen.add(tr[k]); idx.append(k)
idx=np.array(sorted(idx[:32]))
m,s,i,t=take("src",idx); np.savez("sub_src32.npz",mix=m,stems=s,ids=i,tracks=t)
print("ok", len(ev), len(idx))
