"""Summarise f35_grid.py output: mean MASE per (model, ctx, transform) on src/val eval+dev, alone and averaged with fit_predict."""
import pickle, sys
import numpy as np
from scilib import forecast as F

pools = pickle.load(open(sys.argv[1], "rb")); base = pickle.load(open(sys.argv[2], "rb"))
grid = {}
for p in sys.argv[3].split(","):
    grid.update(pickle.load(open(p, "rb")))
H, m = 24, 12


def mase_mean(c, view, pred):
    rows = pools["pools"][c]
    out = []
    for r, p in zip(rows, pred):
        h = r["hist"]
        ins, y = (h, r["target"]) if view == "eval" else (h[:-H], h[-H:])
        out.append(F.mase(y, p, ins, m))
    return float(np.mean(out))


keys = sorted({k[:3] for k in grid}, key=str)
print("%-28s" % "model/ctx/tr", *["%10s" % f"{c}/{v}" for c in ("src", "val") for v in ("eval", "dev")], "%8s" % "pooled", "| +fit_predict:", "pooled")
fp = {(c, v): mase_mean(c, v, base[c][v]) for c in ("src", "val") for v in ("eval", "dev")}
print("%-28s" % "fit_predict", *["%10.4f" % fp[(c, v)] for c in ("src", "val") for v in ("eval", "dev")], "%8.4f" % np.mean(list(fp.values())))
for k in keys:
    a = [mase_mean(c, v, grid[k + (c, v)]) for c in ("src", "val") for v in ("eval", "dev")]
    b = [mase_mean(c, v, (grid[k + (c, v)] + np.asarray(base[c][v])) / 2) for c in ("src", "val") for v in ("eval", "dev")]
    print("%-28s" % (k,), *["%10.4f" % x for x in a], "%8.4f" % np.mean(a), "|", "%8.4f" % np.mean(b))
