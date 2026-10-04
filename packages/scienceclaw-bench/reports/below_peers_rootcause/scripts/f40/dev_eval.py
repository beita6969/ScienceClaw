"""Development comparison on the src and val cohorts (30 probe clips + 24 normal support clips per machine; the id cohorts are not read here).
usage: python dev_eval.py <base_scores.pkl>"""
import pickle
import sys

import numpy as np
from scipy.stats import rankdata
from sklearn.covariance import LedoitWolf

import common as C

base = pickle.load(open(sys.argv[1], "rb"))
co = C.cohorts(("src", "val"))
pid = C.public_ids()
E, idx = C.embeddings()


def feats(clips, kind):
    return np.stack([E[kind][idx[pid[c.official_id]]] for c in clips])


def prep(F, S, how):
    if how == "l2":
        n = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
        return n(F), n(S)
    mu, sd = S.mean(0), S.std(0) + 1e-6
    if how == "zs":
        return (F - mu) / sd, (S - mu) / sd
    if how == "zl2":
        Fz, Sz = (F - mu) / sd, (S - mu) / sd
        n = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
        return n(Fz), n(Sz)
    if how == "cl2":                                   # centre on the support mean, then L2
        n = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
        return n(F - mu), n(S - mu)
    raise ValueError(how)


def dist(A, B):
    return np.sqrt(np.maximum(((A[:, None] - B[None]) ** 2).sum(-1), 0))


def scores(F, S, mode):
    """anomaly scores of probe rows F against support rows S"""
    if mode == "nn":
        return dist(F, S).min(1)
    if mode == "knn3":
        return np.sort(dist(F, S), 1)[:, :3].mean(1)
    if mode == "nn2_pool":                              # 2nd nearest in {support} + {other probes}
        d = np.concatenate([dist(F, S), dist(F, F) + np.diag(np.full(len(F), np.inf))], 1)
        return np.sort(d, 1)[:, 1]
    if mode == "nn_pool":
        d = np.concatenate([dist(F, S), dist(F, F) + np.diag(np.full(len(F), np.inf))], 1)
        return d.min(1)
    if mode == "maha":
        lw = LedoitWolf().fit(S)
        P = lw.precision_
        d = F - S.mean(0)
        return np.einsum("ij,jk,ik->i", d, P, d)
    raise ValueError(mode)


CAND = {}
for kind in ("layer4", "layer8", "layer12", "pooled", "logits", "clap_proj", "clap_pooled"):
    for how in ("l2", "zs", "cl2"):
        for mode in ("nn", "nn2_pool", "maha") if how == "zs" else ("nn", "nn2_pool"):
            CAND[f"{kind}/{how}/{mode}"] = (kind, how, mode)


def run(split, name):
    kind, how, mode = CAND[name]
    out = {}
    for m, (probe, sup) in co[split].items():
        F, S = prep(feats(probe, kind), feats(sup, kind), how)
        out[m] = scores(F, S, mode)
    return out


def table(per_split_scores):
    res = {}
    for split in ("src", "val"):
        pm = {}
        for m, (probe, sup) in co[split].items():
            y = [c.label for c in probe]
            d = [c.domain for c in probe]
            pm[m] = (y, d, per_split_scores[split][m])
        res[split] = C.pooled(pm)
    return res


def ranked(sc):
    return {m: rankdata(v) / len(v) for m, v in sc.items()}


def base_scores(split, key):
    out = {}
    for m, (probe, sup) in co[split].items():
        b = base[(split, m)]
        assert b["ids"] == [c.official_id for c in probe]
        out[m] = b["ens"] if key == "ens" else b["comp"][key]
    return out


rows = []
b = table({s: base_scores(s, "ens") for s in ("src", "val")})
rows.append(("anomsound ensemble (current)", b["src"], b["val"]))
for k in ("nn_train", "nn2_pool", "nn_pool", "lof", "band_max", "maha"):
    t = table({s: base_scores(s, k) for s in ("src", "val")})
    rows.append((f"mel/{k}", t["src"], t["val"]))
cache = {}
for name in CAND:
    cache[name] = {s: run(s, name) for s in ("src", "val")}
    t = table(cache[name])
    rows.append((name, t["src"], t["val"]))
for name, s, v in sorted(rows, key=lambda r: -(r[1] + r[2])):
    print(f"{name:34s} src {s:.4f} val {v:.4f} mean {(s + v) / 2:.4f}")
pickle.dump(cache, open("/private/tmp/claude-501/sc-scratch/f40/dev_cache.pkl", "wb"))
