"""Development comparison on the src-pool assays (train -> the 128 query variants of the src assays; id/ood pools are never read)."""
import json, pickle, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from scilib import proteinfit as pf

items = pickle.load(open(sys.argv[1], "rb"))
feats = pickle.load(open(sys.argv[2], "rb"))
out = Path(sys.argv[3])
sel = {a: v for a, v in items.items() if a in feats}
print(len(sel), "assays", flush=True)
CFG = {
    "baseline": None,
    "esm6": ["esm_llr_masked", "esm_llr_wt", "esm_logp_mut_masked", "esm_logp_wt_masked", "esm_entropy_masked", "esm_site_mean_llr"],
    "esm2": ["esm_llr_masked", "esm_llr_wt"],
    "esm1": ["esm_llr_masked"],
}
res = {}
for a, it in sel.items():
    tr, q, wt = it["train"], it["query"], it["wild_type"]
    F = feats[a]
    n_tr, n_dev, n_q = len(tr), len(it["dev"]), len(q)
    Ftr, Fq = F.iloc[:n_tr], F.iloc[n_tr + n_dev:]
    r = {"n_train": n_tr, "len": len(wt)}
    r["zs_masked"] = pf.spearman(Fq["esm_llr_masked"].to_numpy(), q["DMS_score"].to_numpy())
    r["zs_wt"] = pf.spearman(Fq["esm_llr_wt"].to_numpy(), q["DMS_score"].to_numpy())
    for name, cols in CFG.items():
        kw = {} if cols is None else {"extra_train": Ftr[cols], "extra_query": Fq[cols]}
        p = pf.fit_predict(tr, wt, q, **kw)
        r[name] = pf.spearman(p, q["DMS_score"].to_numpy())
    res[a] = r
    print(a[:38], " ".join(f"{k}={v:.3f}" for k, v in r.items() if k not in ("n_train", "len")), flush=True)
df = pd.DataFrame(res).T
print(df.drop(columns=["n_train", "len"]).astype(float).mean().round(4).to_string())
rng = np.random.default_rng(0)
for c in ["esm6", "esm2", "esm1"]:
    d = (df[c] - df["baseline"]).astype(float).to_numpy()
    bs = [rng.choice(d, len(d)).mean() for _ in range(4000)]
    print(f"{c} - baseline: {d.mean():+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]  better on {(d > 0).mean():.0%}")
df.to_json(out)
