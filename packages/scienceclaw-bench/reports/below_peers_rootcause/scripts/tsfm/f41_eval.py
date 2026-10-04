"""FoR41 offline comparison (mean of the two per-variable CRPS): fit_predict, Chronos (median, sigma from the 0.1-0.9 / 0.25-0.75 quantile spread), and their averages.
usage: f41_eval.py pools base tsfm[,tsfm2] cohorts"""
import pickle, sys
import numpy as np
from scilib import aquatics as aq

pools = pickle.load(open(sys.argv[1], "rb")); base = pickle.load(open(sys.argv[2], "rb"))
ts = {}
for p in sys.argv[3].split(","):
    ts.update(pickle.load(open(p, "rb")))
cohorts = sys.argv[4].split(",")
V = ("oxygen", "temperature")
LAB = {"chronos_2": "c2", "chronos_bolt": "bolt"}
Z90, Z75 = 2.5631, 1.3490    # q90-q10 = 2.563 sigma; q75-q25 = 1.349 sigma


def obs(c, view):
    k = "obs" if view == "eval" else "dev_obs"
    return np.stack([r[k] for r in pools[c]])


def score(pred, o):
    per = []
    for vi, v in enumerate(V):
        y = o[:, :, vi]; m = np.isfinite(y)
        per.append(float(aq.crps_normal(pred[f"{v}_mu"][m], pred[f"{v}_sigma"][m], y[m]).mean()))
    return float(np.mean(per))


def chron(mdl, ctx, c, view, how):
    out = {}
    for v in V:
        q = ts[(mdl, ctx, c, view, v)].astype(float)           # (n, 30, 5): .1 .25 .5 .75 .9
        out[f"{v}_mu"] = np.clip(q[:, :, 2], 0, 45)
        sd = (q[:, :, 4] - q[:, :, 0]) / Z90 if how == "w90" else (q[:, :, 3] - q[:, :, 1]) / Z75
        out[f"{v}_sigma"] = np.clip(sd, 0.05, 50)
    return out


def mix(a, b, wmu=0.5, wsd=0.5):
    return {k: (wmu if k.endswith("mu") else wsd) * a[k] + (1 - (wmu if k.endswith("mu") else wsd)) * b[k] for k in a}


rows = []
for c in cohorts:
    for view in ("eval", "dev"):
        o = obs(c, view); b = base[c][view]
        M = {"fit_predict": b}
        for (mdl, ctx, cc, vv, v) in sorted({k for k in ts if k[2] == c and k[3] == view and k[4] == "oxygen"}, key=str):
            for how in ("w90", "w50"):
                M[f"{LAB[mdl]}{ctx or ''}-{how}"] = chron(mdl, ctx, c, view, how)
                M[f"fp+{LAB[mdl]}{ctx or ''}-{how}"] = mix(b, M[f"{LAB[mdl]}{ctx or ''}-{how}"])
        rows.append((f"{c}/{view}", {k: score(p, o) for k, p in M.items()}))
names = list(rows[0][1])
print("%-26s" % "method", *["%10s" % lab for lab, _ in rows])
for n in names:
    print("%-26s" % n, *["%10.4f" % d[n] for _, d in rows])
