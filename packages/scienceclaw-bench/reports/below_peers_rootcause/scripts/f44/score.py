"""argv: pool.pkl est.json -> RMSE/sd(y) of every stored method and of method combinations (mean), with paired bootstrap vs the current default."""
import json, pickle, sys, itertools
import numpy as np
d = pickle.load(open(sys.argv[1], "rb")); E = {k: np.array(v) for k, v in json.load(open(sys.argv[2])).items()}
base_name = sys.argv[3] if len(sys.argv) > 3 else "default"
E["default"] = (E["impute_lgbm"] + E["xlearner_lgbm"]) / 2
names = [k for k in E if k not in ("regression_adjustment",)]
cands = {k: E[k] for k in names}
for a, b in itertools.combinations([k for k in names if k != "default"], 2):
    cands[f"{a}+{b}"] = (E[a] + E[b]) / 2
bart_like = [k for k in names if k in ("bart", "joint_bart", "bcf")]
if len(bart_like) >= 2 and "xlearner_lgbm" in E:
    cands["+".join(bart_like + ["xlearner_lgbm"])] = np.mean([E[k] for k in bart_like + ["xlearner_lgbm"]], axis=0)
se = lambda est: ((est - d["satt"]) / d["sd_y"]) ** 2
rng = np.random.default_rng(0)
B = rng.integers(0, len(d["satt"]), size=(4000, len(d["satt"])))
ref = se(cands["default"])
print(f"n={len(d['satt'])}")
for k, v in sorted(cands.items(), key=lambda kv: se(kv[1]).mean()):
    s = se(v); r = np.sqrt(s.mean())
    diff = np.sqrt(s[B].mean(1)) - np.sqrt(ref[B].mean(1))
    lo, hi = np.percentile(diff, [2.5, 97.5])
    print(f"{k:40s} {r:.4f}  vs default {r - np.sqrt(ref.mean()):+.4f} [{lo:+.4f},{hi:+.4f}]")
