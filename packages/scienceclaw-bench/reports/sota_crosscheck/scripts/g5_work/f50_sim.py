import csv, numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
R="/Users/admin/Datasets/ScienceClaw-rebuild-20260928/datasets/for50-valueeval-2023/raw/"
def rd(f): return list(csv.DictReader(open(R+f,encoding='utf-8-sig'),delimiter='\t'))
def load(split):
    a=rd(f"arguments-{split}.tsv"); l={r['Argument ID']:r for r in rd(f"labels-{split}.tsv")}
    cols=[c for c in list(l.values())[0] if c!='Argument ID']
    a=[r for r in a if r['Argument ID'] in l]
    Y=np.array([[int(l[r['Argument ID']][c]) for c in cols] for r in a])
    return a,Y
def official(t,p):
    P=[];Rr=[]
    for c in range(t.shape[1]):
        rel=t[:,c].sum()
        if rel==0: continue
        pos=p[:,c].sum(); tp=((p[:,c]==1)&(t[:,c]==1)).sum()
        P.append(tp/pos if pos else 0); Rr.append(tp/rel)
    Pm,Rm=np.mean(P),np.mean(Rr); return 2*Pm*Rm/(Pm+Rm) if Pm+Rm else 0
tr,Ytr=load("training"); va,Yva=load("validation"); te,Yte=load("test"); nj,Ynj=load("test-nahjalbalagha")
txt=lambda A:[r['Conclusion']+" "+r['Stance']+" "+r['Premise'] for r in A]
# leakage: overlap of conclusions and premises between train and eval pools
trc={r['Conclusion'] for r in tr}; trp={r['Premise'] for r in tr}
for name,A in [("val",va),("test",te),("nahj",nj)]:
    print(name,len(A),"conclusion in train: %.3f"%np.mean([r['Conclusion'] in trc for r in A]),"premise in train: %.4f"%np.mean([r['Premise'] in trp for r in A]),
          "(conc,prem) in train: %.4f"%np.mean([(r['Conclusion'],r['Premise']) in {(x['Conclusion'],x['Premise']) for x in tr} for r in A]))
vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True); Xtr=vec.fit_transform(txt(tr));
Xva,Xte,Xnj=[vec.transform(txt(A)) for A in (va,te,nj)]
def fit(Xa,Ya,Xb):
    out=np.zeros((Xb.shape[0],Ya.shape[1]))
    for c in range(Ya.shape[1]):
        m=LogisticRegression(C=3,class_weight='balanced',max_iter=300).fit(Xa,Ya[:,c]); out[:,c]=m.predict_proba(Xb)[:,1]
    return out
Pva=fit(Xtr,Ytr,Xva); Pte=fit(Xtr,Ytr,Xte); Pnj=fit(Xtr,Ytr,Xnj)
th=np.array([max(np.linspace(.2,.8,13),key=lambda t: official(Yva[:,[c]],(Pva[:,[c]]>t).astype(int))) for c in range(20)])
rng=np.random.default_rng(0)
for name,P,Y in [("test",Pte,Yte),("nahj",Pnj,Ynj)]:
    pred=(P>th).astype(int); full=official(Y,pred); one=official(Y,np.ones_like(Y))
    s=[];s1=[];s2=[]
    for _ in range(3000):
        i=rng.choice(len(Y),16,replace=False); s.append(official(Y[i],pred[i])); s1.append(official(Y[i],np.ones_like(Y[i])))
    s=np.array(s);s1=np.array(s1)
    print(name,"TFIDF-LR full F1=%.3f | 16-slice mean=%.3f sd=%.3f p5=%.3f p95=%.3f | 1-baseline full=%.3f slice mean=%.3f sd=%.3f"%(full,s.mean(),s.std(),*np.percentile(s,[5,95]),one,s1.mean(),s1.std()))
    s2=[]
    for _ in range(2000):
        i=rng.choice(len(Y),32,replace=False); s2.append(official(Y[i],pred[i]))
    print("   32-item pooled slice mean=%.3f sd=%.3f"%(np.mean(s2),np.std(s2)))
