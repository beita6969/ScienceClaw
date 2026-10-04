"""Data-only check (no training): do PUD gold UPOS/DEPREL follow GSD-train conventions? per-form majority label from GSD train, scored on GSD test and PUD."""
import sys, collections
ROOT="/Users/admin/Datasets/ScienceClaw-rebuild-20260928/datasets/for47-ud-conll2018/subset/ud-treebanks-v2.2"
def rd(p):
    out=[]
    for blk in open(p,encoding="utf-8").read().split("\n\n"):
        toks=[l.split("\t") for l in blk.split("\n") if l and not l.startswith("#")]
        toks=[t for t in toks if t[0].isdigit()]
        if toks: out.append(toks)
    return out
tr=rd(f"{ROOT}/UD_French-GSD/fr_gsd-ud-train.conllu")
te=rd(f"{ROOT}/UD_French-GSD/fr_gsd-ud-test.conllu")
pu=rd(f"{ROOT}/UD_French-PUD/fr_pud-ud-test.conllu")
up=collections.defaultdict(collections.Counter); dr=collections.defaultdict(collections.Counter)
for s in tr:
    for t in s:
        f=t[1].lower(); up[f][t[3]]+=1; dr[f][t[7].split(":")[0]]+=1
def score(name,S):
    n=k=ku=kd=0; conf=collections.Counter(); confd=collections.Counter()
    for s in S:
        for t in s:
            n+=1; f=t[1].lower()
            if f in up:
                k+=1; pu_=up[f].most_common(1)[0][0]; pd=dr[f].most_common(1)[0][0]
                ku+=pu_==t[3]; kd+=pd==t[7].split(":")[0]
                if pu_!=t[3]: conf[(t[3],pu_)]+=1
                if pd!=t[7].split(":")[0]: confd[(t[7].split(":")[0],pd)]+=1
    print(f"{name}: tokens {n}, known-form share {k/n:.3f}, majority-UPOS acc on known {ku/k:.4f}, majority-DEPREL acc on known {kd/k:.4f}")
    print("  top UPOS conf (gold,majority):",[(a,b,c) for (a,b),c in conf.most_common(6)])
    print("  top DEPREL conf (gold,majority):",[(a,b,c) for (a,b),c in confd.most_common(6)])
score("GSD test",te); score("PUD",pu)
# upos distribution
for name,S in (("GSD train",tr),("PUD",pu)):
    c=collections.Counter(t[3] for s in S for t in s); n=sum(c.values())
    print(name,{k:round(v/n,3) for k,v in c.most_common(9)})
