import sys, time, numpy as np, pickle
sys.path.insert(0, ".")
from scienceclaw.bench.tasks import for52_psych201 as m
from scilib import psych
ad=m.Psych201Adapter(); D=ad._data.get(); S=D.sessions
rng=np.random.default_rng(0)
def tr_payload(k):
    us=D.pools["train"]; sel=[us[j] for j in rng.permutation(len(us))[:k]]
    return [{"study": S[u].study, "text": S[u].text, "responses":[{"pos":int(r["pos"]),"response":r["response"],"options":list(r["options"])} for r in S[u].responses]} for u in sel]
train=tr_payload(32)
dev=[S[u] for u in D.pools["dev"]]; dev_items=[s.item() for s in dev]
out={}
t0=time.time()
for pool in ("id","ood"):
    uids=D.pools[pool]; res={}
    for c in range(0,len(uids),88):
        ch=[S[u] for u in uids[c:c+88]]
        items=[s.item() for s in ch]
        r=psych.fit_predict(items,train,"all",extra_items=dev_items)
        for name,keys in r.items():
            res.setdefault(name,[]).extend(keys)
        print(pool,c,round(time.time()-t0),"s",flush=True)
    y=[S[u].target for u in uids]
    out[pool]={"y":y,"uids":uids,"pred":res}
    print(pool,{k:round(float(np.mean([a==b for a,b in zip(v,y)])),3) for k,v in res.items()},flush=True)
pickle.dump(out,open("/private/tmp/claude-501/sc-scratch/sota/g5_work/f52_pool_preds.pkl","wb"))
