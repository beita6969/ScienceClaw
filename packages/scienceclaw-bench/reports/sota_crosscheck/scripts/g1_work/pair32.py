import sys, re, json
sys.path.insert(0, ".")
from collections import Counter
from scienceclaw.bench.tasks.for32_msd_hippocampus import Adapter
a = Adapter(cache_dir="/private/tmp/claude-501/sc-scratch/sota/g1_work/cache32")
idx = a._index()
num = {int(re.search(r"_(\d+)$", c).group(1)): c for c in idx}
print("group-size histogram over all 260:", Counter(Counter(v["subject"] for v in idx.values()).values()))
def sim(i, j):
    if i not in num or j not in num: return None
    A, B = idx[num[i]], idx[num[j]]
    return (A["encoding"] == B["encoding"], A["dtype"] == B["dtype"], abs(A["p99"]-B["p99"])/max(A["p99"],B["p99"]) < 0.02 , A["shape"] == B["shape"])
odd = [sim(i, i+1) for i in range(1, 394, 2) if sim(i, i+1)]
even = [sim(i, i+1) for i in range(2, 394, 2) if sim(i, i+1)]
for nm, L in (("odd-start pairs (2k-1,2k)", odd), ("even-start pairs (2k,2k+1)", even)):
    n = len(L); print(nm, "n=", n, "same enc %.2f" % (sum(x[0] for x in L)/n), "p99 within2%% %.2f" % (sum(x[2] for x in L)/n), "same shape %.2f" % (sum(x[3] for x in L)/n))
