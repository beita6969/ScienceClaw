"""Measured cost of scilib.tsfm.forecast (weights on the GPU host; run on a shared GPU)."""
import time
import numpy as np
from scilib import tsfm

rng = np.random.default_rng(0)
def series(n, L):
    t = np.arange(L)
    return [10 + 3 * np.sin(2 * np.pi * t / 12 + rng.uniform(0, 6)) + 0.01 * t + rng.normal(0, 0.5, L) for _ in range(n)]

for model in ("chronos_2", "chronos_bolt"):
    t0 = time.time(); tsfm.forecast(series(2, 100), 6, model=model); load = time.time() - t0
    for n, L, H in ((256, 120, 24), (256, 300, 24), (64, 1461, 7), (64, 1461, 28)):
        X = series(n, L)
        tsfm.forecast(X[:4], H, model=model)
        best = 1e9
        for _ in range(3):
            t0 = time.time(); out = tsfm.forecast(X, H, model=model); best = min(best, time.time() - t0)
        print(f"{model:13s} first call {load:5.1f}s | n={n:3d} len={L:4d} H={H:2d}: {best:6.3f}s -> {n / best:7.1f} series/s  out {out.shape}", flush=True)
