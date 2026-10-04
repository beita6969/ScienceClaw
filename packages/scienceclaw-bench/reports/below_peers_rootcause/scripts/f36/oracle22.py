"""IRM oracle under OUR protocol (22.05 kHz, 6 s excerpts, 2048/1024 Hann STFT, mixture phase) on id/val/src pools."""
import sys, time, numpy as np
from multiprocessing import Pool
from common import A, item_scores, aggregate
TARGETS=A.TARGETS
def oracle_one(args):
    mix, stems, powers = args
    out={}
    g=A._rms(mix)
    P=A._ratio_masks(stems/g)          # (4,F,T) power ratio (Wiener-like, "IRM2")
    Z=A.stft(mix/g)
    n=mix.shape[0]
    for pw in powers:
        m=P**pw; m=m/(m.sum(0,keepdims=True)+1e-12)
        est=np.stack([A.istft(Z*m[j][None], n)*g for j in range(4)])
        est+=1e-7*A._square(n)
        out[pw]=item_scores(stems.astype(np.float64), est.astype(np.float64))
    # mixture-as-estimate
    out["mix"]=item_scores(stems.astype(np.float64), np.repeat(mix[None],4,0).astype(np.float64))
    return out
if __name__=="__main__":
    powers=[0.5,1.0]   # 0.5 = magnitude-ratio mask (IRM1), 1.0 = power-ratio (Wiener/IRM2)
    res={}
    for pool in ["id","val","src"]:
        z=np.load(f"pool_{pool}.npz"); mix,st=z["mix"],z["stems"]
        t=time.time()
        with Pool(8) as p: outs=p.map(oracle_one,[(mix[i],st[i],powers) for i in range(len(mix))])
        res[pool]={k:np.array([o[k] for o in outs]) for k in outs[0]}
        print(pool, len(mix), round(time.time()-t,1),"s")
        for k,v in res[pool].items():
            print(f"   {k}: pooled SDR {aggregate(v):.2f}  per-target median {[round(float(np.nanmedian(v[:,j])),2) for j in range(4)]}  nan_items {int(np.isnan(v).any(1).sum())}")
    np.savez("oracle22_items.npz", **{f"{p}_{k}":v for p,d in res.items() for k,v in d.items()})
