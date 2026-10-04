import sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
REPO = "/Users/admin/Library/Application Support/Claude/scratch-workspaces/b286c85c-fb50-4909-a7a4-692fbcc9e2e6/67544b4d-5c42-4571-a504-64db11cd447d/scratch-2026-09-28-749df2/scienceclaw"
sys.path.insert(0, REPO)
from scienceclaw.bench.tasks.for51_matbench import Adapter
W = "/private/tmp/claude-501/sc-scratch/f51/"
X = np.load(W + "f51_X.npy"); y = np.load(W + "f51_y.npy")
a = Adapter(); d = a._data.get(); parts = a._parts.get()
idx = {i: k for k, i in enumerate(d.ids)}
ix = lambda name: np.array([idx[i] for i in parts[name]])
tr, dv, idp, ood = ix("train"), ix("dev"), ix("id"), ix("ood")
C = np.array([d.features[i] for i in d.ids], float)
test = np.concatenate([dv, idp, ood])
ns = {}
exec(open(W + "model_eval_mlip.py").read(), ns)
out = ns["run"]({"X_train_base": C[tr], "X_eval_base": C[test], "X_train_mlip": X[tr], "X_eval_mlip": X[test], "y_train": y[tr]}, {})
p = out["y"]
n1, n2 = len(dv), len(dv) + len(idp)
mae = lambda q, t: float(np.mean(np.abs(q - t)))
res = {"dev": mae(p[:n1], y[dv]), "id": mae(p[n1:n2], y[idp]), "ood": mae(p[n2:], y[ood])}
res["n"] = {"dev": len(dv), "id": len(idp), "ood": len(ood)}
json.dump(res, open(W + "pool_eval2_result.json", "w"), indent=1)
np.save(W + "pool_eval2_pred.npy", p)
print(json.dumps(res, indent=1))
