import sys, importlib.util, numpy as np, torch
sys.path.insert(0, ".")
from scienceclaw.bench.tasks.for33_buildingsbench import per_building_nrmse, category_medians, balanced_score, persistence
spec = importlib.util.spec_from_file_location("bbm", "/Users/admin/Datasets/ScienceClaw-rebuild-20260928/reference_code/FoR33/BuildingsBench/buildings_bench/evaluation/metrics.py")
bbm = importlib.util.module_from_spec(spec); spec.loader.exec_module(bbm)
rng = np.random.default_rng(1)
nb, w = 12, 2
cats = {f"b{i}": ("residential" if i < 7 else "commercial") for i in range(nb)}
y = []; p = []; bl = []
for i in range(nb):
    base = rng.uniform(0.2, 5) * (1 + 0.5 * np.sin(np.arange(24) / 24 * 2 * np.pi))
    for _ in range(w):
        yy = np.maximum(base * rng.lognormal(0, 0.4, 24), 0.01); pp = np.maximum(base * rng.lognormal(0, 0.3, 24), 0)
        y.append(yy); p.append(pp); bl.append(f"b{i}")
y = np.array(y); p = np.array(p)
ours = per_building_nrmse(y, p, bl)
off = {}
for b in sorted(set(bl)):
    m = bbm.Metric('cvrmse', bbm.MetricType.SCALAR, bbm.squared_error, normalize=True, sqrt=True)
    for k in np.where(np.array(bl) == b)[0]:
        m(torch.tensor(y[k][None], dtype=torch.float64), torch.tensor(p[k][None], dtype=torch.float64))
    m.mean(); off[b] = 100 * float(m.value)
print("max |ours-official| per-building CVRMSE %:", max(abs(ours[b] - off[b]) for b in ours))
med_off = {c: float(np.median([off[b] for b in off if cats[b] == c])) for c in ("residential", "commercial")}
print("ours cat medians:", category_medians(ours, cats), "official-style:", med_off)
print("balanced ours", balanced_score(ours, cats)[0], "vs", 0.5 * sum(med_off.values()))
# AveragePersistence vs. previous day on a noisy series
ctx = rng.gamma(2, 1, (5, 168))
print("persistence check:", np.allclose(persistence(ctx), ctx[:, 144:168]))
