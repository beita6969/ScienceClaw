"""dev-only comparison of phonon-feature sources with the agent's dev-selected ensemble (model B): train 676 -> dev 110. usage: dev_eval.py X.npy"""
import sys, json, numpy as np, warnings, os
warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "2")
REPO = "/Users/admin/Library/Application Support/Claude/scratch-workspaces/b286c85c-fb50-4909-a7a4-692fbcc9e2e6/67544b4d-5c42-4571-a504-64db11cd447d/scratch-2026-09-28-749df2/scienceclaw"
sys.path.insert(0, REPO)
from scienceclaw.bench.tasks.for51_matbench import Adapter
W = "/private/tmp/claude-501/sc-scratch/f51/"
X = np.load(sys.argv[1]); y = np.load(W + "f51_y.npy")
a = Adapter(); d = a._data.get(); parts = a._parts.get()
idx = {i: k for k, i in enumerate(d.ids)}
ix = lambda name: np.array([idx[i] for i in parts[name]])
tr, dv = ix("train"), ix("dev")
C = np.array([d.features[i] for i in d.ids], float)
src = open(W + "agent_ph_dev_pred.py").read().replace("n_jobs=-1", "n_jobs=2")
ns = {}
exec(src, ns)
out = ns["run"]({"X_train": C[tr], "X_dev": C[dv], "X_train_ph": X[tr], "X_dev_ph": X[dv], "y_train": y[tr]}, {})
p = out["pred"]
res = {"dev_mae": float(np.mean(np.abs(p - y[dv]))), "nan_rows_train": int(np.isnan(X[tr]).any(axis=1).sum()), "nan_rows_dev": int(np.isnan(X[dv]).any(axis=1).sum())}
json.dump(res, open(sys.argv[1].replace(".npy", "_dev.json"), "w"))
np.save(sys.argv[1].replace(".npy", "_devpred.npy"), p)
print(json.dumps(res))
