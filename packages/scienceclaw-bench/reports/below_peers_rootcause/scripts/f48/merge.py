"""usage: python merge.py <contract-nli dir> <out prefix> <n_shards> <out.npz>"""
import json
import sys

import numpy as np

ddir, prefix, nsh, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
texts, hyps = {}, {}
for part in ("train", "dev", "test"):
    d = json.load(open(f"{ddir}/{part}.json"))
    hyps.update(d["labels"])
    for doc in d["documents"]:
        for a, b in doc["spans"]:
            texts.setdefault(doc["text"][a:b], len(texts))
texts = list(texts)
keys = sorted(hyps)
cat = lambda w, n: np.concatenate([np.load(f"{prefix}_{w}_{s}.npy") for s in range(n)])
emb, rr, nli = cat("embed", 1), cat("relevance", nsh), cat("nli", nsh)
assert len(emb) == len(rr) == len(nli) == len(texts), (len(emb), len(rr), len(nli), len(texts))
np.savez(out, texts=np.array(texts, dtype=object), hyp_keys=np.array(keys),
         hyp_text=np.array([hyps[k]["hypothesis"] for k in keys], dtype=object), rr=rr, nli=nli, emb=emb,
         hyp_emb_q=np.load(f"{prefix}_hyp_q.npy"), hyp_emb_plain=np.load(f"{prefix}_hyp_plain.npy"))
print("saved", out, rr.shape, nli.shape, emb.shape)
