"""Rank-average combinations of AST-embedding scores and the log-mel ensemble on the src / val cohorts (reads dev_cache.pkl from dev_eval.py).
usage: python combo_eval.py <base_scores.pkl> <dev_cache.pkl>"""
import itertools
import pickle
import sys

import numpy as np
from scipy.stats import rankdata

import common as C

base = pickle.load(open(sys.argv[1], "rb"))
cache = pickle.load(open(sys.argv[2], "rb"))
co = C.cohorts(("src", "val"))
rng = np.random.default_rng(0)


def pool(sc, split):
    return {m: (np.array([c.label for c in co[split][m][0]]), np.array([c.domain for c in co[split][m][0]]), sc[m]) for m in sc}


def r(v):
    return rankdata(v) / len(v)


def combo(names, wmel=0.0):
    out = {}
    for s in ("src", "val"):
        out[s] = {}
        for m in co[s]:
            parts = [r(cache[n][s][m]) for n in names]
            if wmel:
                parts.append(wmel * r(base[(s, m)]["ens"]))
            out[s][m] = np.sum(parts, 0)
    return out


def score(sc):
    return {s: C.pooled(pool(sc[s], s)) for s in sc}


def boot(sc_a, sc_b, n=500):
    """paired stratified bootstrap (clips resampled within each machine's (domain, label) stratum) of the mean over src and val of the pooled score a - b"""
    P = {s: (pool(sc_a[s], s), pool(sc_b[s], s)) for s in ("src", "val")}
    d = []
    for _ in range(n):
        v = []
        for s in ("src", "val"):
            pa, pb = {}, {}
            for m in co[s]:
                y, dom, a = P[s][0][m]
                _, _, b = P[s][1][m]
                i = np.concatenate([rng.choice(np.where((dom == dd) & (y == yy))[0], size=((dom == dd) & (y == yy)).sum())
                                    for dd in ("source", "target") for yy in (0, 1)])
                pa[m], pb[m] = (y[i], dom[i], a[i]), (y[i], dom[i], b[i])
            v.append(C.pooled(pa) - C.pooled(pb))
        d.append(np.mean(v))
    return np.percentile(d, [2.5, 97.5])


ens = {s: {m: base[(s, m)]["ens"] for m in co[s]} for s in ("src", "val")}
e = score(ens)
print(f"current ensemble                          src {e['src']:.4f} val {e['val']:.4f} mean {np.mean(list(e.values())):.4f}")
A = ["logits/zs/nn2_pool", "pooled/l2/nn2_pool", "layer12/zs/nn2_pool", "clap_pooled/l2/nn2_pool", "clap_proj/l2/nn2_pool", "clap_pooled/zs/nn2_pool"]
res = []
for k in (1, 2, 3):
    for names in itertools.combinations(A, k):
        for w in (0.0, 1.0, 2.0):
            sc = combo(names, w)
            q = score(sc)
            res.append((np.mean(list(q.values())), q["src"], q["val"], names, w))
res.sort(key=lambda t: -t[0])
for mean, s, v, names, w in res[:25]:
    print(f"mean {mean:.4f} src {s:.4f} val {v:.4f}  w_mel={w}  {'+'.join(names)}")
print("--- fixed a-priori combinations (one pooled AST kNN, one CLAP kNN)")
for names, w in ((("pooled/l2/nn2_pool", "clap_pooled/l2/nn2_pool"), 0.0), (("pooled/l2/nn2_pool", "clap_pooled/l2/nn2_pool"), 1.0),
                 (("logits/zs/nn2_pool", "pooled/l2/nn2_pool", "clap_pooled/l2/nn2_pool"), 0.0),
                 (("clap_pooled/l2/nn2_pool",), 0.0), (("pooled/l2/nn2_pool",), 0.0)):
    sc = combo(names, w)
    q = score(sc)
    ci = boot(sc, ens)
    print(f"mean {np.mean(list(q.values())):.4f} src {q['src']:.4f} val {q['val']:.4f} w_mel={w} {names}  vs current ensemble 95% CI of mean diff ({ci[0]:+.4f}, {ci[1]:+.4f})")
