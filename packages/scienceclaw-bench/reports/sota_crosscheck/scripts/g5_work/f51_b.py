import sys, numpy as np, pickle, collections
sys.path.insert(0, ".")
D=pickle.load(open("/private/tmp/claude-501/sc-scratch/sota/g5_work/f51_data.pkl","rb"))
ids=D["ids"]; T=D["targets"]; S=D["structures"]; P=D["parts"]
print(list(S[ids[0]].keys()))
def formula(i): return S[i]["formula"]
def reduced(i):
    from collections import Counter
    c=Counter(S[i]["species"]);
    import math
    g=0
    for v in c.values(): g=math.gcd(g,v)
    return "".join(f"{k}{v//g}" for k,v in sorted(c.items()))
def chemsys(i): return "-".join(sorted(set(S[i]["species"])))
def anon(i):
    from collections import Counter
    c=sorted(Counter(S[i]["species"]).values());
    import math; g=0
    for v in c: g=math.gcd(g,v)
    return tuple(v//g for v in c)
for nm,f in (("formula",formula),("reduced",reduced),("chemsys",chemsys)):
    cnt=collections.Counter(f(i) for i in ids)
    dup=sum(1 for v in cnt.values() if v>1)
    print(nm,"unique",len(cnt),"of",len(ids),"formulas with >1 entries:",dup, "max mult",max(cnt.values()))
train=set(P["train"])
def overlap(split, f):
    tf=set(f(i) for i in train)
    n=sum(1 for i in P[split] if f(i) in tf)
    return n, len(P[split])
for sp in ("dev","src","val","id","ood"):
    print(sp, "reduced-formula in train:", overlap(sp,reduced), " chemsys in train:", overlap(sp,chemsys), " formula:", overlap(sp,formula))
# pairs with same reduced formula: target differences
by=collections.defaultdict(list)
for i in ids: by[reduced(i)].append(i)
diffs=[]
for k,v in by.items():
    if len(v)>1:
        for a in range(len(v)):
            for b in range(a+1,len(v)):
                diffs.append(abs(T[v[a]]-T[v[b]]))
print("same-reduced-formula pairs",len(diffs),"median |dy|",np.median(diffs) if diffs else None, "mean",np.mean(diffs) if diffs else None)
# exact duplicate structures? compare lattice+coords hash
import hashlib
def sh(i):
    s=S[i]; L=np.round(np.array(s["lattice"]),3); return hashlib.md5((str(L.tolist())+str(s["species"])+str(np.round(np.array(s["frac_coords"]),3).tolist())).encode()).hexdigest()
h=collections.Counter(sh(i) for i in ids); print("exact dup structures:",sum(1 for v in h.values() if v>1))
# how many of the 179 OOD elements count: Se vs Te
print("ood with Se",sum(1 for i in D["ood"] if "Se" in S[i]["species"]),"Te",sum(1 for i in D["ood"] if "Te" in S[i]["species"]))
# elements frequency
print(collections.Counter(len(set(S[i]["species"])) for i in ids))
