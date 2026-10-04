import numpy as np, json, sys
R="/Users/admin/Library/Application Support/Claude/scratch-workspaces/b286c85c-fb50-4909-a7a4-692fbcc9e2e6/67544b4d-5c42-4571-a504-64db11cd447d/scratch-2026-09-28-749df2/scienceclaw"
X=np.load(R+"/cache/tasks/FoR37/era5_t2m_2018_2020_6h_64x32.npy").astype(np.float64)   # (T,64,32)
T=X.shape[0]; print("shape",X.shape)
lat=np.linspace(-87.1875,87.1875,32)
# ---- WB2 weights (copied from weatherbench2/metrics.py) vs repo
def wb2_w(latd):
    x=np.deg2rad(latd); b=np.concatenate([[-np.pi/2],(x[:-1]+x[1:])/2,[np.pi/2]]); w=np.sin(b[1:])-np.sin(b[:-1]); return w/w.mean()
def repo_w(lat_deg):
    lat=np.asarray(lat_deg,float); d=float(np.abs(np.diff(lat)).mean())
    up=np.deg2rad(np.clip(lat+d/2,-90,90)); lo=np.deg2rad(np.clip(lat-d/2,-90,90)); w=np.sin(up)-np.sin(lo); return w/w.mean()
print("weights max abs diff", np.abs(wb2_w(lat)-repo_w(lat)).max())
w=wb2_w(lat)
def per_rmse(p,o): return np.sqrt(np.mean((p-o)**2*w[None,None,:],axis=(1,2)))
# index: t=0 -> 2018-01-01 00; 2019-01-01 idx=1460; 2020-01-01 idx=2920 (365d*4); 2020 leap 1464
i19=365*4; i20=730*4; i21=i20+366*4; print("T",T,"expect",i21)
def rng(a,b,lead=4): return np.arange(a,b-lead)  # t0 idx s.t. t0+lead < b
for name,(a,b) in {"2019":(i19,i20),"2020":(i20,i21)}.items():
    t0=rng(a,b); p=X[t0]; o=X[t0+4]
    r=per_rmse(p,o); print(f"persist24 {name}: mean-of-RMSE {r.mean():.3f}  sqrt-of-mean-MSE {np.sqrt((r**2).mean()):.3f}  n={len(t0)}  00/12 only: {per_rmse(X[t0[(t0%4)%2==0]],X[t0[(t0%4)%2==0]+4]).mean():.3f}")
# ---- harmonic climatology fitted on 2018 Jan-Nov (visible train)
tr_end=int((334)*4)  # up to 2018-11-30 18 => days 0..333 -> 334*4 samples
tt=np.arange(T); doy=(tt//4)/365.25*2*np.pi; hod=(tt%4)/4*2*np.pi
def feats(t):
    d=(t//4)/365.25*2*np.pi; h=(t%4)/4*2*np.pi
    cols=[np.ones_like(d)]
    for k in (1,2): cols+= [np.cos(k*d),np.sin(k*d)]
    for k in (1,2):
        for a in (np.ones_like(d),np.cos(d),np.sin(d)): cols+=[a*np.cos(k*h),a*np.sin(k*h)]
    return np.stack(cols,1)
F=feats(tt); Y=X.reshape(T,-1)
beta=np.linalg.lstsq(F[:tr_end],Y[:tr_end],rcond=None)[0]
clim=(F@beta).reshape(X.shape)
for name,(a,b) in {"2019":(i19,i20),"2020":(i20,i21)}.items():
    t0=rng(a,b); r=per_rmse(clim[t0+4],X[t0+4]); print(f"harmonic-clim(2018 fit) {name}: {r.mean():.3f}")
    # damped anomaly persistence, oracle scalar on 2018 dev
A=X-clim
tr=np.arange(8,tr_end-4)
# per-gridpoint OLS of anomaly(t+24) on 4 lag anomalies + 3x3 neighbourhood mean lags (ridge)
def nb_mean(A):
    P=np.pad(A,((0,0),(1,1),(0,0)),mode="wrap")
    P=np.pad(P,((0,0),(0,0),(1,1)),mode="edge")
    return sum(P[:,i:i+A.shape[1],j:j+A.shape[2]] for i in range(3) for j in range(3))/9
Nb=nb_mean(A)
def design(t0):
    cols=[A[t0-k] for k in (0,1,2,3)]+[Nb[t0-k] for k in (0,2)]
    return np.stack(cols,-1)  # (n,64,32,6)
Dtr=design(tr); ytr=A[tr+4]
lam=1.0
coef=np.zeros((64,32,6))
for i in range(64):
    for j in range(32):
        Xd=Dtr[:,i,j,:]; yy=ytr[:,i,j]
        coef[i,j]=np.linalg.solve(Xd.T@Xd+lam*np.eye(6),Xd.T@yy)
def pred(t0): return clim[t0+4]+np.einsum("nijk,ijk->nij",design(t0),coef)
for name,(a,b) in {"2019":(i19,i20),"2020":(i20,i21)}.items():
    t0=rng(a,b); r=per_rmse(pred(t0),X[t0+4]); rp=per_rmse(X[t0],X[t0+4]); print(f"ridge-local(2018 fit) {name}: {r.mean():.3f}  persist {rp.mean():.3f}  ratio {r.mean()/rp.mean():.3f}")
np.save("/private/tmp/claude-501/sc-scratch/sota/g2_work/f37_ridge_coef.npy",coef)

# ---- slice noise: lanes of 16 items 120 h (20 idx) apart on the 60h+{0,12h} pattern
print("--- lane noise")
for name,(a,b) in {"2019":(i19,i20),"2020":(i20,i21)}.items():
    pat=[a+10*j+o for j in range(0,(b-a)//10) for o in (0,2)]
    pat=[t for t in pat if t+4<b]
    ps=set(pat); res=[]
    for s in pat:
        lane=[s+20*k for k in range(16)]
        if all(t in ps for t in lane):
            t0=np.array(lane); rp=per_rmse(X[t0],X[t0+4]).mean(); rr=per_rmse(pred(t0),X[t0+4]).mean(); res.append((rp,rr))
    res=np.array(res); print(name,"n_lanes",len(res),"persist mean/sd/min/max %.3f %.3f %.3f %.3f"%(res[:,0].mean(),res[:,0].std(),res[:,0].min(),res[:,0].max()),
      "| ridge %.3f %.3f %.3f %.3f"%(res[:,1].mean(),res[:,1].std(),res[:,1].min(),res[:,1].max()),
      "| ratio sd %.3f mean %.3f"%((res[:,1]/res[:,0]).std(),(res[:,1]/res[:,0]).mean()))
