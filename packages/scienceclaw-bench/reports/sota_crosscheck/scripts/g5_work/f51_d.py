import numpy as np, pickle, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
D=pickle.load(open("f51_data.pkl","rb"))
T=D["targets"]; F=D["features"]; P=D["parts"]
X=lambda ids_: np.array([F[i] for i in ids_]); y=lambda ids_: np.array([T[i] for i in ids_])
tr=P["train"]
et=ExtraTreesRegressor(500,random_state=0,n_jobs=-1).fit(X(tr),y(tr))
knn=make_pipeline(StandardScaler(),KNeighborsRegressor(5)).fit(X(tr),y(tr))
rng=np.random.default_rng(0)
for sp in ("id","ood","val"):
    ids_=P[sp]; yt=y(ids_); pe=et.predict(X(ids_)); pk=knn.predict(X(ids_))
    ae=np.abs(pe-yt); ak=np.abs(pk-yt)
    me=[];mk=[];mm=[]
    for _ in range(20000):
        s=rng.choice(len(ids_),16,replace=False)
        me.append(ae[s].mean()); mk.append(ak[s].mean()); mm.append(np.abs(yt[s]-y(tr).mean()).mean())
    q=lambda a:(np.percentile(a,5),np.percentile(a,25),np.median(a),np.percentile(a,75),np.percentile(a,95))
    print(sp,"pool n",len(ids_),"ET pool MAE %.1f (median AE %.1f); 5NN pool MAE %.1f"%(ae.mean(),np.median(ae),ak.mean()))
    print("   16-slice ET MAE  p5/p25/p50/p75/p95:", " ".join("%.1f"%v for v in q(me)))
    print("   16-slice 5NN MAE p5/p25/p50/p75/p95:", " ".join("%.1f"%v for v in q(mk)))
    print("   16-slice mean-predictor MAE p5..p95:", " ".join("%.1f"%v for v in q(mm)))
    print("   P(ET slice MAE<=28.3):%.3f  P(<=52.9):%.3f  P(<=27):%.3f"%(np.mean(np.array(me)<=28.3),np.mean(np.array(me)<=52.9),np.mean(np.array(me)<=27)))
    # largest errors
    o=np.argsort(-ae)[:5]; print("   top ET abs errors:",[ (round(yt[j]),round(pe[j])) for j in o])
# full official-like check: 5-fold CV over all 1265 non-nested with ET (comp+struct) to compare with published
from sklearn.model_selection import KFold
ids=D["ids"]; A=X(ids); Y=y(ids); errs=[]
for trn,tst in KFold(5,shuffle=True,random_state=18012019).split(A):
    m=ExtraTreesRegressor(300,random_state=0,n_jobs=-1).fit(A[trn],Y[trn]); errs.append(np.abs(m.predict(A[tst])-Y[tst]).mean())
print("ET comp+struct, matbench-style 5-fold CV (random KFold seed 18012019) MAE per fold:",[round(e,1) for e in errs],"mean %.1f"%np.mean(errs))
A2=A[:,:28]; errs=[]
for trn,tst in KFold(5,shuffle=True,random_state=18012019).split(A2):
    m=ExtraTreesRegressor(300,random_state=0,n_jobs=-1).fit(A2[trn],Y[trn]); errs.append(np.abs(m.predict(A2[tst])-Y[tst]).mean())
print("ET comp-only same CV:",[round(e,1) for e in errs],"mean %.1f"%np.mean(errs))
