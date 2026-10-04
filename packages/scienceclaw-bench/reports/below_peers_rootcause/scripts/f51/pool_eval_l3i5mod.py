import sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
REPO = "/Users/admin/Library/Application Support/Claude/scratch-workspaces/b286c85c-fb50-4909-a7a4-692fbcc9e2e6/67544b4d-5c42-4571-a504-64db11cd447d/scratch-2026-09-28-749df2/scienceclaw"
sys.path.insert(0, REPO)
from scienceclaw.bench.tasks.for51_matbench import Adapter
W = "/private/tmp/claude-501/sc-scratch/f51/"
X = np.load(W + "X_l3i5_module.npy"); y = np.load(W + "f51_y.npy")
a = Adapter(); d = a._data.get(); parts = a._parts.get()
idx = {i: k for k, i in enumerate(d.ids)}
ix = lambda name: np.array([idx[i] for i in parts[name]])
tr, dv, idp, ood = ix("train"), ix("dev"), ix("id"), ix("ood")
print("sizes train/dev/id/ood", len(tr), len(dv), len(idp), len(ood), flush=True)
C = np.array([d.features[i] for i in d.ids], float)
test = np.concatenate([dv, idp, ood])
ns = {}
exec(open(W + "agent_ph_dev_pred.py").read().replace("n_jobs=-1", "n_jobs=2"), ns)
out = ns["run"]({"X_train": C[tr], "X_dev": C[test], "X_train_ph": X[tr], "X_dev_ph": X[test], "y_train": y[tr]}, {})
p = out["pred"]
n1, n2 = len(dv), len(dv) + len(idp)
mae = lambda q, t: float(np.mean(np.abs(q - t)))
res = {"dev": mae(p[:n1], y[dv]), "id": mae(p[n1:n2], y[idp]), "ood": mae(p[n2:], y[ood])}
ref = lambda ids: mae(a._ref_predict(d, [d.ids[i] for i in ids]), y[ids])
res["ref"] = {"dev": ref(dv), "id": ref(idp), "ood": ref(ood)}
res["n"] = {"id": len(idp), "ood": len(ood)}
res["nan_ph_rows_in_test"] = int(np.isnan(X[test]).any(axis=1).sum())
json.dump(res, open(W + "pool_eval_l3i5mod_result.json", "w"), indent=1)
np.save(W + "pool_eval_l3i5mod_pred.npy", p)
print(json.dumps(res, indent=1))
