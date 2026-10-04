"""One-shot id/ood evaluation of the pre-declared pipelines with paired bootstrap CIs (relative change of the pipeline vs the library alone; negative = better).
FoR35: 0.5 fit_predict + 0.5 Chronos-2(context 120, log1p, median); FoR33: 0.5 ens + 0.5 Chronos-2(168 h, median); FoR41: mu and sigma = 0.5 fit_predict + 0.5 Chronos-2(1461 d; sigma = q75-q25 / 1.349)."""
import pickle, sys
import numpy as np
from scilib import forecast as F, aquatics as aq
from scienceclaw.bench.tasks.for33_buildingsbench import balanced_nrmse

T = "/private/tmp/claude-501/sc-scratch/tsfm/"
rng = np.random.default_rng(0)
B = 4000
ld = lambda n: pickle.load(open(T + n, "rb"))


def ci(x):
    return np.percentile(x, [2.5, 97.5])


# ---------------------------------------------------------------- FoR35
p35, b35, f35 = ld("f35_pools.pkl"), ld("f35_base.pkl"), ld("f35_final_c2.pkl")
print("== FoR35 Tourism Monthly (MASE, eval view)")
for c in ("id", "ood"):
    rows = p35["pools"][c]
    sn = np.array([F.mase(r["target"], np.array([r["hist"][len(r["hist"]) - 12 + (h % 12)] for h in range(24)]), r["hist"], 12) for r in rows])
    fp = np.array([F.mase(r["target"], p, r["hist"], 12) for r, p in zip(rows, b35[c]["eval"])])
    c2 = np.array([F.mase(r["target"], p, r["hist"], 12) for r, p in zip(rows, f35[c])])
    pl = np.array([F.mase(r["target"], 0.5 * a + 0.5 * b, r["hist"], 12) for r, a, b in zip(rows, b35[c]["eval"], f35[c])])
    idx = rng.integers(0, len(rows), (B, len(rows)))
    rel = np.array([pl[i].mean() / fp[i].mean() - 1 for i in idx])
    print(f"{c:4s} n={len(rows):3d} snaive {sn.mean():.4f}  fit_predict {fp.mean():.4f}  chronos2 {c2.mean():.4f}  pipeline {pl.mean():.4f}  "
          f"rel {pl.mean() / fp.mean() - 1:+.3%} CI [{ci(rel)[0]:+.3%}, {ci(rel)[1]:+.3%}]  pipeline/snaive {pl.mean() / sn.mean():.3f}")

# ---------------------------------------------------------------- FoR33
p33, b33, c33 = ld("f33_pools.pkl"), ld("f33_base.pkl"), ld("f33_c2.pkl")
cat_of = {b: x["category"] for b, x in p33["buildings"].items()}
print("== FoR33 BuildingsBench (balanced CVRMSE %, eval view; cluster bootstrap over buildings)")
for c in ("id", "ood"):
    rows = p33["pools"][c]
    y = np.stack([r["target"] for r in rows]); bl = np.array([r["building"] for r in rows])
    ens = np.asarray(b33[c]["eval"]["ens"]); c2 = np.maximum(c33[("chronos_2", None, c, "eval")][:, :, 1], 0); pl = 0.5 * ens + 0.5 * c2
    f = lambda pred, ix: balanced_nrmse(y[ix], pred[ix], list(bl[ix]), cat_of)[0]
    allx = np.arange(len(rows)); bs = sorted(set(bl))
    rel = []
    for _ in range(B):
        pick = rng.choice(bs, len(bs))
        ix = np.concatenate([np.where(bl == b)[0] for b in pick])
        names = np.concatenate([[f"{b}#{k}"] * (bl == b).sum() for k, b in enumerate(pick)])
        cat2 = {n: cat_of[n.split("#")[0]] for n in set(names)}
        a = balanced_nrmse(y[ix], ens[ix], list(names), cat2)[0]; b = balanced_nrmse(y[ix], pl[ix], list(names), cat2)[0]
        rel.append(b / a - 1)
    rel = np.array(rel)
    print(f"{c:4s} n={len(rows):3d} ({len(bs)} buildings) ens {f(ens, allx):.2f}  chronos2 {f(c2, allx):.2f}  pipeline {f(pl, allx):.2f}  rel {f(pl, allx) / f(ens, allx) - 1:+.3%} CI [{ci(rel)[0]:+.3%}, {ci(rel)[1]:+.3%}]")

# ---------------------------------------------------------------- FoR41
p41, b41 = ld("f41_pools.pkl"), ld("f41_base.pkl")
t41 = ld("f41_c2_idood.pkl")
V = ("oxygen", "temperature")
print("== FoR41 NEON aquatics (mean of the two per-variable CRPS, eval view; item bootstrap)")
for c in ("id", "ood"):
    rows = p41[c]; o = np.stack([r["obs"] for r in rows])
    base = b41[c]["eval"]
    ch = {}
    for v in V:
        q = t41[("chronos_2", None, c, "eval", v)].astype(float)
        ch[f"{v}_mu"] = np.clip(q[:, :, 2], 0, 45); ch[f"{v}_sigma"] = np.clip((q[:, :, 3] - q[:, :, 1]) / 1.349, 0.05, 50)
    pl = {k: 0.5 * base[k] + 0.5 * ch[k] for k in base}

    def sums(pred):
        s, n = [], []
        for vi, v in enumerate(V):
            y = o[:, :, vi]; m = np.isfinite(y); cr = np.zeros_like(y)
            cr[m] = aq.crps_normal(pred[f"{v}_mu"][m], pred[f"{v}_sigma"][m], y[m]); s.append(cr.sum(1)); n.append(m.sum(1))
        return np.array(s), np.array(n)

    sc = lambda S, ix: float(np.mean(S[0][:, ix].sum(1) / np.maximum(S[1][:, ix].sum(1), 1)))
    Sb, Sc, Sp = sums(base), sums(ch), sums(pl)
    allx = np.arange(len(rows)); idx = rng.integers(0, len(rows), (B, len(rows)))
    rel = np.array([sc(Sp, i) / sc(Sb, i) - 1 for i in idx])
    print(f"{c:4s} n={len(rows):3d} fit_predict {sc(Sb, allx):.4f}  chronos2 {sc(Sc, allx):.4f}  pipeline {sc(Sp, allx):.4f}  rel {sc(Sp, allx) / sc(Sb, allx) - 1:+.3%} CI [{ci(rel)[0]:+.3%}, {ci(rel)[1]:+.3%}]")
