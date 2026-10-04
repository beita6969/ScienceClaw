"""Single-process, low-memory: IRM oracle at native 44.1 kHz vs after decimation to 22.05 kHz, on the middle 30 s of 4 official-test tracks.
Variants: A44 (process+evaluate at 44.1k), A44lp (same estimate, ref+est decimated to 22.05k before SDR), A22 (decimate first = our adapter),
powers 0.5 (IRM1) and 1.0 (IRM2)."""
import gc, numpy as np, pickle
from scipy.signal import resample_poly
from common import adapter, framewise_sdr, M
import oracle44b as O
CH=30
ad=adapter(); data=ad._data.get()
tracks=sorted([t for t in data.tracks.values() if t.pool=="id"], key=lambda t:t.track_id)[::12][:4]
out={}
for t in tracks:
    T={nm:M.decode_stream(ad._ffmpeg, t.path, idx) for nm,(idx,_s) in t.streams.items()}
    n=min(v.shape[0] for v in T.values()); a=(n//44100//2)*44100 - 15*44100; b=a+CH*44100
    mix=T["mixture"][a:b].astype(np.float64); st=np.stack([T[k][a:b] for k in M.TARGETS]).astype(np.float64)
    del T; gc.collect()
    r={}
    e=O.oracle_est(mix, st, 44100, 2048, 1024, (0.5,1.0))
    mix2,st2=O.dec(mix),O.dec(st)
    for pw,est in e.items():
        est=est+1e-7*np.sign(np.sin(np.arange(est.shape[1])))[None,:,None]
        r[("A44",pw)]=framewise_sdr(st, est, 44100)
        r[("A44lp",pw)]=framewise_sdr(O.dec(st), O.dec(est), 22050)
    e2=O.oracle_est(mix2, st2, 22050, 2048, 1024, (0.5,1.0))
    for pw,est in e2.items():
        est=est+1e-7*np.sign(np.sin(np.arange(est.shape[1])))[None,:,None]
        r[("A22",pw)]=framewise_sdr(st2, est, 22050)
    r[("MIX44",1.0)]=framewise_sdr(st, np.repeat(mix[None],4,0), 44100)
    r[("MIX22",1.0)]=framewise_sdr(st2, np.repeat(mix2[None],4,0), 22050)
    out[t.track_id]=r
    print(t.track_id, {f"{k[0]}{k[1]}":[round(float(np.nanmedian(v[j])),2) for j in range(4)] for k,v in r.items()}, flush=True)
pickle.dump(out, open("oracle44c.pkl","wb"))
# aggregate: per-target median over all windows of all tracks, then mean over targets
print("== pooled over", len(out), "tracks x 30 s (median over all 1-s windows per target, then mean over targets)")
for k in [("A44",0.5),("A44lp",0.5),("A22",0.5),("A44",1.0),("A44lp",1.0),("A22",1.0),("MIX44",1.0),("MIX22",1.0)]:
    v=np.concatenate([out[tid][k] for tid in out],axis=1)
    print(k, round(float(np.mean([np.nanmedian(v[j]) for j in range(4)])),2), [round(float(np.nanmedian(v[j])),2) for j in range(4)])
