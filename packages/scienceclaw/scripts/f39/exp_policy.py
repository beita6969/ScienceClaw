"""Server-side: multi-round selection with the IRT toolkit on the same pseudo-validation as exp_nn.py (visible training students only)."""
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ.get("SCIENCECLAW_CODE_DIR", str(Path(__file__).resolve().parents[2])))
from scilib import adaptive  # noqa: E402

Z = np.load(sys.argv[1])
A = Z["answers"].astype(np.int64)
n, Q = A.shape
rng = np.random.default_rng(7)
perm = rng.permutation(n)
V, F = np.sort(perm[:800]), np.sort(perm[800:])


def protocol(rows, seed):
    r = np.random.default_rng(seed)
    a = A[rows]
    tg = np.zeros_like(a, dtype=bool)
    for i in range(len(rows)):
        obs = np.flatnonzero(a[i] >= 0)
        tg[i, r.permutation(obs)[:max(1, int(round(0.2 * obs.size)))]] = True
    return a, tg, (a >= 0) & ~tg


av, tgv, canv = protocol(V, 123)
model = adaptive.fit_item_curves(A[F])


def reveal(sel):
    rev = np.full(av.shape, -1, dtype=np.int64)
    for i in range(av.shape[0]):
        for q in sel[i]:
            if q >= 0:
                rev[i, q] = av[i, q]
    return rev


def run(method, sizes):
    rev = np.full(av.shape, -1, dtype=np.int64)
    for k in sizes:
        sel = adaptive.select_queries(model, canv, rev, k, 10, method)
        new = reveal(sel)
        rev = np.where(new >= 0, new, rev)
    p = adaptive.predict_proba(model, rev)
    return float(((p > 0.5).astype(np.int64) == av)[tgv].mean()), int((rev >= 0).sum(axis=1).mean())


for method, sizes in (("batch", [10]), ("batch", [5, 5]), ("batch", [2] * 5), ("batch", [1] * 10), ("bald", [1] * 10), ("bald", [10])):
    t0 = time.time()
    a, used = run(method, sizes)
    print(method, sizes, f"acc {a:.4f} used {used} ({time.time()-t0:.0f}s)", flush=True)
