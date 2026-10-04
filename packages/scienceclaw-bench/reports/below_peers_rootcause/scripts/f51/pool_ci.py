import sys, numpy as np
REPO = "/Users/admin/Library/Application Support/Claude/scratch-workspaces/b286c85c-fb50-4909-a7a4-692fbcc9e2e6/67544b4d-5c42-4571-a504-64db11cd447d/scratch-2026-09-28-749df2/scienceclaw"
sys.path.insert(0, REPO)
from scienceclaw.bench.tasks.for51_matbench import Adapter
W = "/private/tmp/claude-501/sc-scratch/f51/"
y = np.load(W + "f51_y.npy"); a = Adapter(); d = a._data.get(); parts = a._parts.get()
idx = {i: k for k, i in enumerate(d.ids)}; ix = lambda n: np.array([idx[i] for i in parts[n]])
dv, idp, ood = ix("dev"), ix("id"), ix("ood")
pn, pc = np.load(W + "pool_eval_l3i5_pred.npy"), np.load(W + "pool_eval_pred.npy")
n1, n2 = len(dv), len(dv) + len(idp)
rng = np.random.default_rng(0)
for name, sl, ids in (("dev", slice(0, n1), dv), ("id", slice(n1, n2), idp), ("ood", slice(n2, None), ood)):
    en, ec = np.abs(pn[sl] - y[ids]), np.abs(pc[sl] - y[ids])
    bs = lambda e: np.percentile([e[rng.integers(0, len(e), len(e))].mean() for _ in range(4000)], [2.5, 97.5])
    diff = ec - en
    bd = np.percentile([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(4000)], [2.5, 97.5])
    print(f"{name}: SevenNet {en.mean():.2f} {bs(en).round(1)}  CHGNet {ec.mean():.2f} {bs(ec).round(1)}  paired diff {diff.mean():.2f} {bd.round(1)}  better on {(en<ec).mean():.0%}  median {np.median(en):.1f}/{np.median(ec):.1f} p90 {np.percentile(en,90):.1f}/{np.percentile(ec,90):.1f} max {en.max():.0f}/{ec.max():.0f}")
mad = lambda ids: np.mean(np.abs(y[ids] - np.median(y[ids])))
print("MAE/MAD id", np.abs(pn[n1:n2]-y[idp]).mean()/mad(idp), "ood", np.abs(pn[n2:]-y[ood]).mean()/mad(ood))
