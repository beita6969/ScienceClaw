"""FoR33 offline comparison: balanced CVRMSE (task metric) of library candidates, Chronos median forecasts and averages. usage: f33_eval.py pools base tsfm[,tsfm2] cohorts"""
import pickle, sys
import numpy as np
from scienceclaw.bench.tasks.for33_buildingsbench import balanced_nrmse

pools = pickle.load(open(sys.argv[1], "rb")); base = pickle.load(open(sys.argv[2], "rb"))
ts = {}
for p in sys.argv[3].split(","):
    ts.update(pickle.load(open(p, "rb")))
cohorts = sys.argv[4].split(",")
LAB = {"chronos_2": "c2", "chronos_bolt": "bolt"}
cat_of = {b: x["category"] for b, x in pools["buildings"].items()}


def targets(c, view):
    rows = pools["pools"][c]
    if view == "eval":
        return np.stack([r["target"] for r in rows]), [r["building"] for r in rows]
    bs = pools["split_buildings"][c]
    return (np.concatenate([pools["buildings"][b]["dev_target"] for b in bs]),
            [b for b in bs for _ in range(len(pools["buildings"][b]["dev_target"]))])


def methods(c, view):
    M = {k: v for k, v in base[c][view].items() if k in ("yesterday", "median7", "core", "ml", "ens")}
    for (mdl, ctx, cc, vv), f in ts.items():
        if cc == c and vv == view:
            M[f"{LAB[mdl]}{ctx or ''}"] = np.maximum(f[:, :, 1], 0)
    names = [k for k in M if k.startswith(("c2", "bolt"))]
    for k in names:
        M["ens+" + k] = (M["ens"] + M[k]) / 2
    return M


tab = {}
for c in cohorts:
    for view in ("eval", "dev"):
        y, b = targets(c, view)
        tab[(c, view)] = {k: balanced_nrmse(y, v, b, cat_of)[0] for k, v in methods(c, view).items()}
names = list(next(iter(tab.values())).keys())
print("%-12s" % "", *["%11s" % k[:11] for k in names])
for k, d in tab.items():
    print("%-12s" % "/".join(k), *["%11.2f" % d[n] for n in names])
