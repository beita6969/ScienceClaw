"""FoR34 offline experiment on the GPU host (CPU part). Compares the default scilib.molecules recipe with CheMeleon-fingerprint models.
usage: exp.py sv                      candidates scored on the src+val pools only (selection)
       exp.py final <m1,m2,...>       the named, pre-declared methods on the id and ood pools (run once)
Four training draws (the adapter's own 4000-molecule label-stratified sampler); AUC per draw on the pool, then the mean over draws."""
import sys, time, json
import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, "/home/bedicloud/sharestore2/zxc/scienceclaw/code")
from scilib import molecules as M

S = "/home/bedicloud/sharestore2/zxc/scienceclaw/data/f34/"
smi = [l.rstrip("\n") for l in open(S + "smiles.txt")]
Z = np.load(S + "inputs.npz"); y_all = Z["labels"].astype(int)
E = np.load(S + "emb_chemeleon.npy")
stage = sys.argv[1]
if stage == "sv":
    pools = {"sv": np.concatenate([Z["pool_src"], Z["pool_val"]])}
else:
    pools = {"id": Z["pool_id"], "ood": Z["pool_ood"]}
need = sorted(set(np.concatenate([Z[f"train_{k}"] for k in range(4)] + list(pools.values())).tolist()))
t0 = time.time(); X = {}
Xc, _, _ = M.featurize([smi[i] for i in need], "concat")
for r, i in enumerate(need):
    X[i] = Xc[r]
print("features", len(need), f"{time.time() - t0:.0f}s", flush=True)
mat = lambda idx: np.stack([X[i] for i in idx])
rk = lambda s: rankdata(s) / len(s)


def run_methods(tr, ytr, q):
    Xtr, Xq = mat(tr), mat(q)
    Etr, Eq = E[tr], E[q]
    sc = StandardScaler().fit(Etr)
    out = {"m0_default": M.fit_predict(Xtr, ytr, Xq)}
    for C in (0.01, 0.1, 1.0):
        lr = LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(sc.transform(Etr), ytr)
        out[f"lr_C{C}"] = lr.predict_proba(sc.transform(Eq))[:, 1]
    et = ExtraTreesClassifier(500, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=8, random_state=0).fit(Etr, ytr)
    out["et_emb"] = et.predict_proba(Eq)[:, 1]
    out["concat_emb"] = M.fit_predict(np.hstack([Xtr, Etr]), ytr, np.hstack([Xq, Eq]))
    for b in ("lr_C0.01", "lr_C0.1", "lr_C1.0", "et_emb", "concat_emb"):
        out[f"avg_m0+{b}"] = 0.5 * rk(out["m0_default"]) + 0.5 * rk(out[b])
    return out


res = {p: {} for p in pools}
for k in range(4):
    tr = Z[f"train_{k}"]; ytr = y_all[tr]
    for p, q in pools.items():
        t1 = time.time()
        o = run_methods(tr, ytr, q)
        for name, s in o.items():
            res[p].setdefault(name, []).append(float(roc_auc_score(y_all[q], s)))
            res[p].setdefault("_scores_" + name, []).append(np.asarray(s))
    print("draw", k, f"{time.time() - t0:.0f}s", flush=True)
names = [n for n in res[next(iter(pools))] if not n.startswith("_")]
if stage == "final":
    keep = set(sys.argv[2].split(",")) | {"m0_default"}
    names = [n for n in names if n in keep]
for p, q in pools.items():
    print(f"== pool {p}: n={len(q)} actives={int(y_all[q].sum())}")
    for n in names:
        a = res[p][n]
        print(f"  {n:22s} mean AUC {np.mean(a):.4f}  per-draw {[round(x, 4) for x in a]}")
if stage == "final":
    rng = np.random.default_rng(0)
    for p, q in pools.items():
        yq = y_all[q]; pos, neg = np.where(yq == 1)[0], np.where(yq == 0)[0]
        for n in names:
            if n == "m0_default":
                continue
            d = []
            for _ in range(1000):
                ix = np.concatenate([rng.choice(pos, pos.size), rng.choice(neg, neg.size)])
                d.append(np.mean([roc_auc_score(yq[ix], res[p]["_scores_" + n][k][ix]) - roc_auc_score(yq[ix], res[p]["_scores_m0_default"][k][ix]) for k in range(4)]))
            d.sort()
            print(f"  {p} {n} - m0_default: mean-AUC diff {np.mean(res[p][n]) - np.mean(res[p]['m0_default']):+.4f} (95% CI {d[25]:+.4f}, {d[974]:+.4f})")
