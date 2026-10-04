"""Precompute pretrained-model scores for every ContractNLI span x hypothesis (run on the GPU host with scilib.textenc).

usage: python precompute.py <contract-nli dir> <out prefix> <what: embed|relevance|nli> <shard> <n_shards>
Writes <out prefix>_<what>_<shard>.npy (rows = this shard's slice of the unique span texts); merge.py assembles the table.
"""
import json
import sys
import time

import numpy as np

from scilib import textenc

ddir, prefix, what, shard, nsh = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
texts, hyps = {}, {}
for part in ("train", "dev", "test"):
    d = json.load(open(f"{ddir}/{part}.json"))
    hyps.update(d["labels"])
    for doc in d["documents"]:
        for a, b in doc["spans"]:
            texts.setdefault(doc["text"][a:b], len(texts))
texts = list(texts)
keys = sorted(hyps)
htext = [hyps[k]["hypothesis"] for k in keys]
lo, hi = shard * len(texts) // nsh, (shard + 1) * len(texts) // nsh
mine = texts[lo:hi]
print(what, shard, nsh, len(mine), "spans", flush=True)
t0 = time.time()
if what == "embed":
    out = np.zeros((len(mine), 1024), dtype=np.float16)
    for i in range(0, len(mine), 4000):
        out[i:i + 4000] = textenc.embed(mine[i:i + 4000]).astype(np.float16)
    if shard == 0:
        np.save(f"{prefix}_hyp_q.npy", textenc.embed(htext, query=True))
        np.save(f"{prefix}_hyp_plain.npy", textenc.embed(htext))
else:
    out = np.zeros((len(mine), len(keys)) + ((3,) if what == "nli" else ()), dtype=np.float32)
    for i in range(0, len(mine), 1000):
        out[i:i + 1000] = getattr(textenc, what)(mine[i:i + 1000], htext)
        if (i // 1000) % 5 == 0:
            print(what, shard, i, round(time.time() - t0), "s", flush=True)
np.save(f"{prefix}_{what}_{shard}.npy", out)
print("done", what, shard, round(time.time() - t0), "s", flush=True)
