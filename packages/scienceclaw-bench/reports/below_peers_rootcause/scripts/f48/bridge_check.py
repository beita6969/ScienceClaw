"""One 40-contract fit + 16-pair prediction through the real remote bridge versus the precomputed-table lookup (same model, same seeds)."""
import sys
import time

import numpy as np

import common as K
import lookup

npz, pool = sys.argv[1], sys.argv[2]
from scilib import textenc  # noqa: E402
assert textenc.available(), "bridge not enabled"
data = K.load()
ids = data.train_docs[pool]
rng = np.random.default_rng(1000)
perm = [ids[j] for j in rng.permutation(len(ids))]
fit_ids, hold_ids = perm[:40], perm[40:70]
hyps = K.hyp_table(data)
train_docs = [K.public(data.docs[d]) for d in fit_ids]
hold_docs = [K.public(data.docs[d]) for d in hold_ids]
ann = K.annotations(data, fit_ids)
pairs = K.pairs_of(data, hold_ids)[:16] if isinstance(K.pairs_of(data, hold_ids), list) else list(K.pairs_of(data, hold_ids))[:16]
items = [{"doc_id": d, "hypothesis_key": k, "hypothesis": hyps[k]["hypothesis"]} for d, k in pairs]
t0 = time.time()
m = K.C.ContractModel(seed=0, n_jobs=2, plm=True).fit(train_docs, ann, hyps)
t1 = time.time()
real = m.predict_scores(items, hold_docs)
t2 = time.time()
print(f"bridge: fit {t1 - t0:.0f}s, predict {t2 - t1:.0f}s", flush=True)
lookup.install(npz)
m2 = K.C.ContractModel(seed=0, n_jobs=2, plm=True).fit(train_docs, ann, hyps)
tab = m2.predict_scores(items, hold_docs)
d = max(float(np.max(np.abs(a - b))) for a, b in zip(real, tab))
ap_real = np.mean([K.C.average_precision(K.gold(data, it["doc_id"], it["hypothesis_key"]), s) for it, s in zip(items, real)])
ap_tab = np.mean([K.C.average_precision(K.gold(data, it["doc_id"], it["hypothesis_key"]), s) for it, s in zip(items, tab)])
print(f"max |score diff| {d:.4g}; mAP bridge {ap_real:.4f} table {ap_tab:.4f}")
