import sys, pickle, numpy as np
sys.path.insert(0, ".")
from sklearn.metrics import roc_auc_score
from scienceclaw.bench.tasks.for34_molhiv import MolhivAdapter
ad = MolhivAdapter()
runs = {"id": ("final_a0_main_id", 0x4b7b695a, 2), "ood": ("final_a0_main_ood", 0x69b9a85d, 2), "val": ("final_a0_v_val", 0x9e25e78a, 1)}
Y = {}; S = {}
for sp, (rd, seed, ne) in runs.items():
    for ep in ad.build_episodes(sp, ne, seed):
        y = np.asarray(pickle.load(open(f"runs/{rd}/{ep.id}/y.pkl", "rb")), float).reshape(-1)
        r = ep.evaluate(y, None)
        pl = r.details.get("pooled_payload") or {}
        yt = np.array(pl["y_true"]); ys = np.array(pl.get("y_score", y))
        print(ep.id, "auc", round(r.primary, 4), "ref", round(r.details["reference"], 4), "pos", int(yt.sum()), "n", len(yt), "ties in y_score:", len(ys) - len(set(np.round(ys, 12))))
        Y.setdefault(sp, []).append(yt); S.setdefault(sp, []).append(ys)
ya = np.concatenate([np.concatenate(v) for v in Y.values()]); sa = np.concatenate([np.concatenate(v) for v in S.values()])
for sp in Y: print(sp, "pooled AUC", round(roc_auc_score(np.concatenate(Y[sp]), np.concatenate(S[sp])), 4))
print("all pooled AUC", round(roc_auc_score(ya, sa), 4), "n", len(ya), "pos", int(ya.sum()))
# note: pooling across episodes mixes per-episode score offsets (episode-specific RF); compare with mean of per-episode AUC
rng = np.random.default_rng(0); bs = []
pos = np.where(ya == 1)[0]; neg = np.where(ya == 0)[0]
for _ in range(4000):
    ii = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]); bs.append(roc_auc_score(ya[ii], sa[ii]))
print("bootstrap 95% CI:", np.percentile(bs, [2.5, 97.5]).round(3))
