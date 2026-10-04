import numpy as np, pickle, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import GradientBoostingRegressor
D=pickle.load(open("f51_data.pkl","rb"))
T=D["targets"]; F=D["features"]; P=D["parts"]
X=lambda ids_: np.array([F[i] for i in ids_]); y=lambda ids_: np.array([T[i] for i in ids_])
tr=P["train"]
g=GradientBoostingRegressor(n_estimators=400,max_depth=3,learning_rate=0.05,subsample=0.8,random_state=0).fit(X(tr),y(tr))
rng=np.random.default_rng(1)
for sp in ("dev","src","val","id","ood"):
    yt=y(P[sp]); ae=np.abs(g.predict(X(P[sp]))-yt)
    sl=[ae[rng.choice(len(ae),16,replace=False)].mean() for _ in range(20000)]
    print(sp,"n",len(ae),"GBM(agent) pool MAE %.2f"%ae.mean(),"| 16-slice p5/p50/p95: %.1f %.1f %.1f"%tuple(np.percentile(sl,[5,50,95])), "| MAE/MAD(pool) %.3f"%(ae.mean()/np.abs(yt-yt.mean()).mean()))
# SOTA-relative: matbench dummy 323.99? report ratio using full data
ids=D["ids"]; yy=y(ids); print("full MAD",np.abs(yy-yy.mean()).mean())
