import numpy as np, pickle, collections, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
D=pickle.load(open("f51_data.pkl","rb"))
ids=D["ids"]; T=D["targets"]; F=D["features"]; P=D["parts"]
names=None
NC=28  # composition features: 7 props x 4 stats
def X(ids_, comp_only):
    A=np.array([F[i] for i in ids_]); return A[:,:NC] if comp_only else A
def y(ids_): return np.array([T[i] for i in ids_])
tr=P["train"]
res={}
def evalset(model_fn, comp_only, tag, train_ids=tr):
    m=model_fn(); m.fit(X(train_ids,comp_only), y(train_ids))
    out={}
    for sp in ("dev","src","val","id","ood"):
        p=m.predict(X(P[sp],comp_only)); out[sp]=(np.abs(p-y(P[sp])).mean(), p)
    print(f"{tag:34s}", " ".join(f"{sp}={out[sp][0]:7.1f}" for sp in out))
    return m,out
# baselines
for sp in ("dev","src","val","id","ood"):
    yy=y(P[sp]); print(sp,"predict-train-mean MAE %.1f"%np.abs(yy-y(tr).mean()).mean(), " predict-train-median %.1f"%np.abs(yy-np.median(y(tr))).mean())
for comp in (True, False):
    lab="comp" if comp else "comp+struct"
    evalset(lambda: RandomForestRegressor(300,random_state=0,n_jobs=-1), comp, f"RF {lab}")
    evalset(lambda: ExtraTreesRegressor(300,random_state=0,n_jobs=-1), comp, f"ExtraTrees {lab}")
    evalset(lambda: GradientBoostingRegressor(n_estimators=400,max_depth=4,learning_rate=0.05,subsample=0.8,loss="absolute_error",random_state=0), comp, f"GBM-L1 {lab}")
# 5-NN standardized as adapter reference
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
evalset(lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(5)), False, "5NN std (adapter ref)")
evalset(lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(5)), True, "5NN std comp-only")
