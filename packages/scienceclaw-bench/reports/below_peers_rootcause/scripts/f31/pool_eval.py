"""One-shot pool evaluation (id or ood): proteinfit defaults with vs without the six ESM-2 columns, plus the official random-fold columns of the same assays."""
import json, pickle, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from scilib import proteinfit as pf, proteinplm
items = pickle.load(open(sys.argv[1], "rb")); feats = pickle.load(open(sys.argv[2], "rb")); out = Path(sys.argv[3])
OFF = pd.read_csv("/Users/admin/Datasets/ScienceClaw-rebuild-20260928/reference_code/FoR31/ProteinGym/benchmarks/DMS_supervised/substitutions/Spearman/DMS_substitutions_Spearman_DMS_level_fold_random_5.csv").set_index("DMS_id")
cols = list(proteinplm.FEATURES)
res = {}
for a, it in items.items():
    tr, q, wt = it["train"], it["query"], it["wild_type"]
    F = feats[a]; n_tr, n_dev = len(tr), len(it["dev"])
    Ftr, Fq = F.iloc[:n_tr][cols].reset_index(drop=True), F.iloc[n_tr + n_dev:][cols].reset_index(drop=True)
    y = q["DMS_score"].to_numpy()
    r = {"baseline": pf.spearman(pf.fit_predict(tr, wt, q), y),
         "esm6": pf.spearman(pf.fit_predict(tr, wt, q, extra_train=Ftr, extra_query=Fq), y),
         "zs_masked": pf.spearman(Fq["esm_llr_masked"].to_numpy(), y)}
    if a in OFF.index:
        for k, c in (("kermut", "Kermut"), ("proteinnpt", "ProteinNPT"), ("ohe", "One-Hot Encodings")):
            r[k] = float(OFF.loc[a, c])
    res[a] = r
    print(a[:40], " ".join(f"{k}={v:.3f}" for k, v in r.items()), flush=True)
df = pd.DataFrame(res).T.astype(float)
print(df.mean().round(4).to_string())
rng = np.random.default_rng(0)
def boot(d):
    d = np.asarray(d, float); bs = [rng.choice(d, len(d)).mean() for _ in range(4000)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), (d > 0).mean()
for a_, b_ in (("esm6", "baseline"), ("esm6", "kermut"), ("esm6", "proteinnpt"), ("baseline", "kermut"), ("baseline", "proteinnpt")):
    m, lo, hi, fr = boot(df[a_] - df[b_])
    print(f"{a_} - {b_}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]  better on {fr:.0%}")
df.to_json(out)
