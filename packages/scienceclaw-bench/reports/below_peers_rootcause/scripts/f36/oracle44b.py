"""IRM oracle on the 50 official test tracks (our id pool) at native 44.1 kHz / full tracks vs decimated 22.05 kHz.
Variants per track (all mixture phase, mask = stem power ratio ** p, p in {0.5 (IRM1), 1.0 (IRM2/Wiener)}):
  A44_2048 / A44_4096 : process at 44.1k (Hann nperseg 2048 or 4096, hop 1024), evaluate full band
  A44lp_2048          : same estimate as A44_2048, but ref+est decimated to 22.05k before SDR (metric band-limit only)
  A22                 : decimate mixture+stems to 22.05k first (our adapter), IRM with 2048/1024, evaluate at 22.05k
Per-window (1 s) SDR stored for every window; museval NaN rule (any ref/est silent) via adapter.framewise_sdr."""
import sys, time, pickle, numpy as np
from multiprocessing import Pool
from scipy.signal import stft as sstft, istft as sistft, resample_poly
from common import adapter, framewise_sdr, M
CH=30  # seconds per chunk
def ost(x, fs, nper, hop):  # (n,2)->(2,F,T)
    return sstft(x.astype(np.float64).T, fs=fs, nperseg=nper, noverlap=nper-hop, boundary="zeros", padded=True)[2]
def ist(Z, n, fs, nper, hop):
    x=sistft(Z, fs=fs, nperseg=nper, noverlap=nper-hop)[1][:, :n]
    return np.pad(x, ((0,0),(0,max(0,n-x.shape[1])))).T
def oracle_est(mix, stems, fs, nper, hop, powers):
    n=mix.shape[0]
    g=float(np.sqrt(np.mean(mix.astype(np.float64)**2))); g=g if g>1e-9 else 1.0
    Z=ost(mix/g, fs, nper, hop)
    P=np.stack([np.mean(np.abs(ost(s/g,fs,nper,hop))**2,axis=0) for s in stems]); P=P/(P.sum(0,keepdims=True)+1e-12)
    out={}
    for pw in powers:
        m=P**pw; m=m/(m.sum(0,keepdims=True)+1e-12)
        out[pw]=np.stack([ist(Z*m[j][None], n, fs, nper, hop) for j in range(4)])*g
    return out
def dec(x):  # (...,n,2) -> decimate axis -2 by 2
    return resample_poly(x, 1, 2, axis=-2)
def work(args):
    tid, path, streams, slots, ffm = args
    T={}
    for nm,(idx,_sha) in streams.items():
        T[nm]=M.decode_stream(ffm, path, idx)        # (n,2) float32 at 44.1k, no hash check here
    n=min(v.shape[0] for v in T.values())
    nsec=n//44100
    res={k:{p:[] for p in (0.5,1.0)} for k in ["A44_2048","A44lp_2048","A22","MIX44","MIX22"]}
    chunk_hi=[]
    mid=(nsec//CH)//2*CH
    hi_chunks={mid, mid+CH} if mid+CH+CH<=nsec else {mid}
    for c0 in range(0, nsec, CH):
        c1=min(c0+CH, nsec); a,b=c0*44100,c1*44100
        hi = c0 in hi_chunks; chunk_hi.append((c0,c1,hi))
        mix=T["mixture"][a:b].astype(np.float64); st=np.stack([T[t][a:b] for t in M.TARGETS]).astype(np.float64)
        w=44100
        # 44.1k, two window sizes
        for key,nper in ((("A44_2048",2048),) if hi else ()):
            e=oracle_est(mix, st, 44100, nper, 1024, (0.5,1.0))
            for pw,est in e.items():
                est=est+1e-7*np.sign(np.sin(np.arange(est.shape[1])))[None,:,None]
                res[key][pw].append(framewise_sdr(st, est, w))
                if key=="A44_2048":
                    res["A44lp_2048"][pw].append(framewise_sdr(dec(st), dec(est), w//2))
        # decimate first (our path)
        mix2, st2 = dec(mix), dec(st)
        e=oracle_est(mix2, st2, 22050, 2048, 1024, (0.5,1.0))
        for pw,est in e.items():
            est=est+1e-7*np.sign(np.sin(np.arange(est.shape[1])))[None,:,None]
            res["A22"][pw].append(framewise_sdr(st2, est, w//2))
        # mixture-as-estimate at both rates
        if hi: res["MIX44"][1.0].append(framewise_sdr(st, np.repeat(mix[None],4,0), w))
        res["MIX22"][1.0].append(framewise_sdr(st2, np.repeat(mix2[None],4,0), w//2))
    out={k:{p:(np.concatenate(v,axis=1) if v else None) for p,v in d.items()} for k,d in res.items()}
    out["_chunks"]=chunk_hi
    return tid, slots, out
if __name__=="__main__":
    ad=adapter(); data=ad._data.get()
    tracks=sorted([t for t in data.tracks.values() if t.pool=="id"], key=lambda t:t.track_id)[::4][:12]
    jobs=[(t.track_id, t.path, t.streams, t.slots, ad._ffmpeg) for t in tracks]
    t0=time.time(); allres={}
    with Pool(5) as p:
        for i,(tid,slots,out) in enumerate(p.imap_unordered(work, jobs)):
            allres[tid]=(slots,out)
            if i%5==0: print(i, tid, round(time.time()-t0,1),"s", flush=True)
    pickle.dump(allres, open("oracle44b_frames.pkl","wb"))
    print("done", time.time()-t0)
