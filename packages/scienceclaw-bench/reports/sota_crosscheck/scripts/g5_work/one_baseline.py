import csv, numpy as np, sys
R="/Users/admin/Datasets/ScienceClaw-rebuild-20260928/datasets/for50-valueeval-2023/raw/"
def load(f):
    rows=list(csv.DictReader(open(R+f,encoding='utf-8-sig'),delimiter='\t'))
    cols=[c for c in rows[0] if c!='Argument ID']
    return cols, np.array([[int(r[c]) for c in cols] for r in rows])
def official(t,p):
    P=[];Rr=[];F=[]
    for c in range(t.shape[1]):
        rel=t[:,c].sum()
        if rel==0: continue
        pos=p[:,c].sum(); tp=((p[:,c]==1)&(t[:,c]==1)).sum()
        pr=tp/pos if pos else 0; rc=tp/rel
        P.append(pr);Rr.append(rc);F.append(2*pr*rc/(pr+rc) if pr+rc else 0)
    Pm,Rm=np.mean(P),np.mean(Rr)
    return 2*Pm*Rm/(Pm+Rm), np.mean(F), Pm, Rm, len(P)
for f in ["labels-training.tsv","labels-validation.tsv","labels-test.tsv","labels-test-nahjalbalagha.tsv","labels-test-nyt.tsv","labels-validation-zhihu.tsv"]:
    cols,t=load(f)
    o=official(t,np.ones_like(t))
    print(f, t.shape, "1-baseline: official(harmonic of macro P,R)=%.4f  mean-per-value-F1=%.4f  macroP=%.4f macroR=%.4f nvals=%d"%o)
