import sys, json
sys.path.insert(0, ".")
from scienceclaw.bench.tasks.for32_msd_hippocampus import Adapter
a = Adapter(cache_dir="/private/tmp/claude-501/sc-scratch/sota/g1_work/cache32")
ok, why = a.available(); print(ok, why)
P = a._pools(); idx = a._index()
subj = {k: {idx[c]["subject"] for c in v} for k, v in P.items()}
ks = list(P)
for i in range(len(ks)):
    for j in range(i+1, len(ks)):
        print(ks[i], ks[j], "shared subjects:", len(subj[ks[i]] & subj[ks[j]]), "shared volumes:", len(set(P[ks[i]]) & set(P[ks[j]])))
print(a.describe_pools())
# pairing: for ids in each subject in a pool, how many volumes
from collections import Counter
print({k: Counter(Counter(idx[c]["subject"] for c in v).values()) for k, v in P.items()})
# raw ids
import re
ids = sorted(int(re.search(r"_(\d+)$", c).group(1)) for c in idx)
print(ids[:20], ids[-5:], len(ids))
