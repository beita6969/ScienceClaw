import sys, numpy as np, collections, re
R=sys.argv[1]; sys.path.insert(0,R)
from pathlib import Path
from scienceclaw.bench.tasks import for43_hipe as F
root=Path('/Users/admin/Datasets/ScienceClaw-rebuild-20260928/datasets')
units=F._load_units(root, 12000)
pools=F._build_pools(units, F.PARTITION_SEED)
print('excluded train/dev units due to group overlap:', pools.excluded_train_groups)
ev_ids=[uid for sp in pools.eval_split.values() for l in sp.values() for uid in l]
print('eval units total', len(ev_ids), {sp:sum(len(l) for l in d.values()) for sp,d in pools.eval_split.items()})
tr_ids=[uid for p in pools.train.values() for l in p.values() for uid in l]
print('visible train units', len(tr_ids), {k:sum(len(l) for l in p.values()) for k,p in pools.train.items()})
# group overlap
eg={units[u].group for u in ev_ids}; tg={units[u].group for u in tr_ids}
print('group overlap eval/train:', len(eg&tg))
# page overlap (dta pages)
ep={units[u].page for u in ev_ids if units[u].dataset.startswith('dta')}; tp={units[u].page for u in tr_ids if units[u].dataset.startswith('dta')}
print('dta page overlap:', len(ep&tp))
# exact gt / ocr duplicates and 6-word shingle overlap
def sh(t,n=6):
    w=F.hipe_norm(t).split(); return {' '.join(w[i:i+n]) for i in range(max(0,len(w)-n+1))}
trsh=collections.defaultdict(set)
tr_sh={}
idx=collections.defaultdict(set)
for i,u in enumerate(tr_ids):
    s=sh(units[u].gt); tr_sh[u]=s
    for x in s: idx[x].add(u)
ndup=0; frac=[]
for u in ev_ids:
    s=sh(units[u].gt)
    if not s: continue
    hit=set()
    for x in s:
        if x in idx: hit.add(x)
    f=len(hit)/len(s); frac.append(f)
frac=np.array(frac)
print('eval GT 6-word shingles present in visible train GT: mean %.3f, items with >=50%%: %d, >=20%%: %d of %d'%(frac.mean(),(frac>=.5).sum(),(frac>=.2).sum(),len(frac)))
# same for ocr text
# also the dev-pool (episode dev items) - are they disjoint from eval items?
# check split uid disjointness
print('eval id/val/ood pairwise overlap:', {(a,b):len({u for l in pools.eval_split[a].values() for u in l}&{u for l in pools.eval_split[b].values() for u in l}) for a in pools.eval_split for b in pools.eval_split if a<b})
# train units in official-test pool?
print('train units file_split counts', collections.Counter(units[u].file_split for u in tr_ids))
