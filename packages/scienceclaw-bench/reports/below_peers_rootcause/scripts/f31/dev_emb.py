"""Src-pool dev: ESM log-prob columns with / without low-dimensional site-embedding columns (id/ood pools never read)."""
import pickle, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from scilib import proteinfit as pf
items = pickle.load(open(sys.argv[1], "rb")); feats = pickle.load(open(sys.argv[2], "rb")); emb = pickle.load(open(sys.argv[3], "rb"))
out = Path(sys.argv[4])
sel = [a for a in items if a in feats and a in emb]
print(len(sel), "assays", flush=True)
ESM6 = ["esm_llr_masked", "esm_llr_wt", "esm_logp_mut_masked", "esm_logp_wt_masked", "esm_entropy_masked", "esm_site_mean_llr"]
res = {}
for a in sel:
    it = items[a]; tr, q, wt = it["train"], it["query"], it["wild_type"]
    F = feats[a]; n_tr, n_dev = len(tr), len(it["dev"])
    Ftr, Fq = F.iloc[:n_tr][ESM6].reset_index(drop=True), F.iloc[n_tr + n_dev:][ESM6].reset_index(drop=True)
    E = emb[a]; Ec = E - E.mean(0)
    U, S, Vt = np.linalg.svd(Ec, full_matrices=False)
    r = {}
    base = pf.fit_predict(tr, wt, q, extra_train=Ftr, extra_query=Fq)
    r["esm6"] = pf.spearman(base, q["DMS_score"].to_numpy())
    for k in (4, 8, 16):
        Z = Ec @ Vt[:k].T
        cols = [f"pc{i}" for i in range(k)]
        Ztr = pd.DataFrame(Z[tr["position"].to_numpy() - 1], columns=cols)
        Zq = pd.DataFrame(Z[q["position"].to_numpy() - 1], columns=cols)
        p = pf.fit_predict(tr, wt, q, extra_train=pd.concat([Ftr, Ztr], axis=1), extra_query=pd.concat([Fq, Zq], axis=1))
        r[f"esm6+pc{k}"] = pf.spearman(p, q["DMS_score"].to_numpy())
    res[a] = r
    print(a[:38], " ".join(f"{k}={v:.3f}" for k, v in r.items()), flush=True)
df = pd.DataFrame(res).T.astype(float)
print(df.mean().round(4).to_string())
rng = np.random.default_rng(0)
for c in df.columns[1:]:
    d = (df[c] - df["esm6"]).to_numpy(); bs = [rng.choice(d, len(d)).mean() for _ in range(4000)]
    print(f"{c} - esm6: {d.mean():+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}] better on {(d > 0).mean():.0%}")
df.to_json(out)
