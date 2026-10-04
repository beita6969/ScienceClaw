"""FoR35 offline comparison: MASE per cohort and view for seasonal naive, scilib.forecast.fit_predict, Chronos-2/Bolt median and averages.
usage: f35_eval.py <pools.pkl> <base.pkl> <tsfm.pkl[,tsfm2.pkl]> <cohorts>   (cohorts are read only when named: tune on src,val; id/ood once)"""
import pickle, sys
import numpy as np
from scilib import forecast as F

pools = pickle.load(open(sys.argv[1], "rb"))
base = pickle.load(open(sys.argv[2], "rb"))
ts = {}
for p in sys.argv[3].split(","):
    ts.update(pickle.load(open(p, "rb")))
cohorts = sys.argv[4].split(",")
H, m = 24, 12


def snaive(x):
    x = np.asarray(x)
    return np.array([x[len(x) - m + (h % m)] for h in range(H)])


def per_series(rows, pred, view):
    out = []
    for r, p in zip(rows, pred):
        h = r["hist"]
        ins, y = (h, r["target"]) if view == "eval" else (h[:-H], h[-H:])
        out.append(F.mase(y, p, ins, m))
    return np.array(out)


def methods(c, view):
    rows = pools["pools"][c]
    ins = [r["hist"] if view == "eval" else r["hist"][:-H] for r in rows]
    M = {"snaive": np.stack([snaive(x) for x in ins]), "fit_predict": np.asarray(base[c][view])}
    for mdl in ("chronos_2", "chronos_bolt"):
        if (mdl, c) in ts:
            M[mdl] = ts[(mdl, c)][view][:, :, 1].astype(float)
    if "chronos_2" in M and "chronos_bolt" in M:
        M["c2+bolt"] = (M["chronos_2"] + M["chronos_bolt"]) / 2
    if "chronos_2" in M:
        M["fp+c2"] = (M["fit_predict"] + M["chronos_2"]) / 2
    if "chronos_2" in M and "chronos_bolt" in M:
        M["fp+c2+bolt"] = (M["fit_predict"] + M["chronos_2"] + M["chronos_bolt"]) / 3
    return rows, M


res = {}
for c in cohorts:
    for view in ("eval", "dev"):
        rows, M = methods(c, view)
        res[(c, view)] = {k: per_series(rows, v, view) for k, v in M.items()}
names = list(next(iter(res.values())).keys())
print("%-16s" % "cohort/view", *["%12s" % k for k in names])
for (c, view), d in res.items():
    print("%-16s" % f"{c}/{view}", *["%12.4f" % d[k].mean() for k in names])
pickle.dump(res, open("/private/tmp/claude-501/sc-scratch/tsfm/f35_eval_%s.pkl" % "_".join(cohorts), "wb"))
