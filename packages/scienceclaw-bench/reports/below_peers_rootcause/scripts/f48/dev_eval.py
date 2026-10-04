"""Contract-grouped train-holdout comparison of ContractModel configurations (train partition only; no id/ood contract is read).

usage: python dev_eval.py <npz> <pool: iid|ood> <trials> <n_fit> <n_hold> <out.json> <cfg> [<cfg> ...]
cfg = name:groups (groups comma-separated from rerank,nli,cos,emb_lr,emb_knn; 'none' = original model)
"""
import json
import sys
import time

import numpy as np

import common as K
import lookup

npz, pool, trials, n_fit, n_hold, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
cfgs = {}
for c in sys.argv[7:]:
    name, g = c.split(":")
    kw = {}
    if "=" in g:                         # name:groups;key=val
        pass
    cfgs[name] = () if g == "none" else tuple(g.split(","))
lookup.install(npz)
data = K.load()
ids = data.train_docs[pool]
hyps = K.hyp_table(data)
res = {n: {"ap": [], "trial": [], "doc": [], "key": []} for n in cfgs}
for t in range(trials):
    rng = np.random.default_rng(1000 + t)
    perm = [ids[j] for j in rng.permutation(len(ids))]
    fit_ids, hold_ids = perm[:n_fit], perm[n_fit:n_fit + n_hold]
    train_docs = [K.public(data.docs[d]) for d in fit_ids]
    hold_docs = [K.public(data.docs[d]) for d in hold_ids]
    ann = K.annotations(data, fit_ids)
    items = [{"doc_id": d, "hypothesis_key": k, "hypothesis": hyps[k]["hypothesis"]} for d, k in K.pairs_of(data, hold_ids)]
    for name, groups in cfgs.items():
        t0 = time.time()
        m = K.C.ContractModel(seed=0, n_jobs=2, plm=groups).fit(train_docs, ann, hyps)
        sc = m.predict_scores(items, hold_docs)
        r = res[name]
        for it, s in zip(items, sc):
            r["ap"].append(K.C.average_precision(K.gold(data, it["doc_id"], it["hypothesis_key"]), s))
            r["trial"].append(t); r["doc"].append(it["doc_id"]); r["key"].append(it["hypothesis_key"])
        print(f"trial {t} {name:14s} mAP {np.mean(r['ap'][-len(items):]):.4f}  ({len(items)} pairs, {time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open(out, "w"))
print("--- pooled over", trials, "trials")
base = next(iter(cfgs))
grp = [f"{a}|{b}" for a, b in zip(res[base]["trial"], res[base]["doc"])]
for n in cfgs:
    d = K.paired_boot(res[base]["ap"], res[n]["ap"], grp) if n != base else (0, 0, 0)
    print(f"{n:14s} mAP {np.mean(res[n]['ap']):.4f}   vs {base}: {d[0]:+.4f} ({d[1]:+.4f}, {d[2]:+.4f})", flush=True)
