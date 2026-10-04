import sys, numpy as np
R="/Users/admin/Library/Application Support/Claude/scratch-workspaces/b286c85c-fb50-4909-a7a4-692fbcc9e2e6/67544b4d-5c42-4571-a504-64db11cd447d/scratch-2026-09-28-749df2/scienceclaw"
sys.path.insert(0,R); sys.dont_write_bytecode=True
from scilib import weather
X=np.load(R+"/cache/tasks/FoR37/era5_t2m_2018_2020_6h_64x32.npy").astype(np.float64)
T0=np.datetime64("2018-01-01T00","h")
times=[str(T0+np.timedelta64(6*i,"h"))+":00" for i in range(X.shape[0])]
tr_end=334*4
lat=np.linspace(-87.1875,87.1875,32)
x=np.deg2rad(lat); b=np.concatenate([[-np.pi/2],(x[:-1]+x[1:])/2,[np.pi/2]]); w=np.sin(b[1:])-np.sin(b[:-1]); w/=w.mean()
per=lambda p,o: np.sqrt(np.mean((p-o)**2*w[None,None,:],axis=(1,2)))
m=weather.fit_patch_ridge(X[:tr_end],times[:tr_end],radius=2,lam=30.0,n_harmonics=3,hour_bins=4)
i19=1460; i20=2920; i21=4384
for name,(a,bb) in {"2019":(i19,i20),"2020":(i20,i21)}.items():
    t0=np.arange(a,bb-4)
    ctx=np.stack([X[t0-3],X[t0-2],X[t0-1],X[t0]],1)
    P=weather.predict_patch_ridge(m,ctx,[times[t] for t in t0])
    r=per(P,X[t0+4]); rp=per(X[t0],X[t0+4]); print(name,"scilib patch_ridge %.3f persist %.3f ratio %.3f  (n=%d)"%(r.mean(),rp.mean(),r.mean()/rp.mean(),len(t0)))
