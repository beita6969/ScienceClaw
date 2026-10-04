"""Single sequential run (1 process): A oracle ceiling, B learning curve (1 perm seed), C protocol-24 + in-sample. Threads limited."""
import sys, time; sys.path.insert(0, '/private/tmp/claude-501/sc-scratch/fix/f30_work')
from lc_lib import *
T0 = time.time(); out = WORK + "/out/t3.jsonl"
def log(**r):
    r["t"] = round(time.time() - T0); open(out, "a").write(json.dumps(r) + "\n"); print(json.dumps(r), flush=True)
a = adapter(2); p = a._pools(); test = json.load(open(WORK + "/out/test_ids.json")); pool = p["src"] + p["val"]
csz = [float(np.load(fkey(i))["crop_scale"]) for i in pool]; PS = float(np.median([c for c in csz if c > 0]))
log(stage="setup", plant_size_pool=PS)
# ---- A: oracle semantics -> panoptic stage (no learning); 8 src + 8 id + 8 ood images
rng = np.random.default_rng(1); orc = {"src": sorted(rng.choice(p["src"], 8, replace=False).tolist()), "id": test["id"][:8], "ood": test["ood"][:8]}
for k, ids in orc.items():
    sem = np.stack([ps._merge_partial(a._load(i)["semantics"]) for i in ids])
    P = np.eye(3, dtype=np.float16)[sem]
    log(stage="oracle_sem", pool=k, n=len(ids), **evaluate(a, ids, P, PS))
# ---- B: learning curve, train n random official-train images (src+val), test 12 id + 12 ood (held out)
perm = [pool[i] for i in np.random.default_rng(100).permutation(len(pool))]
for n in (2, 4, 8, 16, 24, 32, 64):
    m = fit(perm[:n], seed=0); row = dict(stage="curve", n=n, fit_s=round(m.fit_s, 1), plant_size=m.plant_size)
    for k, ids in test.items(): row[k] = evaluate(a, ids, probs(m, ids), m.plant_size)
    log(**row)
# ---- C: protocol-24 (captures P0030692 + P0030947 = what id/ood episodes see), held-out and in-sample
g = groups(a)["src"]; tr = g["P0030692"] + g["P0030947"]
m = fit(tr, seed=0); row = dict(stage="protocol24", n=len(tr), fit_s=round(m.fit_s, 1), plant_size=m.plant_size)
for k, ids in test.items(): row[k] = evaluate(a, ids, probs(m, ids), m.plant_size)
dev = g["P0030855"]; row["dev_capture_P0030855"] = evaluate(a, dev, probs(m, dev), m.plant_size)
row["in_sample"] = evaluate(a, tr, probs(m, tr), m.plant_size)
log(**row)
# ---- D: single-capture training (what src-episodes see: one free capture) -> test on id+ood
for cap in ("P0030855", "P0030692", "P0030947"):
    m = fit(g[cap], seed=0); row = dict(stage="single_capture", cap=cap, n=len(g[cap]), fit_s=round(m.fit_s, 1))
    for k, ids in test.items(): row[k] = evaluate(a, ids, probs(m, ids), m.plant_size)
    log(**row)
log(stage="done")
