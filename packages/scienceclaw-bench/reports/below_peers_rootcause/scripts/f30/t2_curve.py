"""Learning curve: train on n random images of the official-train pools (src+val, 64 imgs, nested prefixes of a seeded permutation),
test on 20 id + 20 ood held-out images (never in training), stride-4 lattice. usage: t2_curve.py <perm_seed> <n1,n2,..> """
import sys; sys.path.insert(0, '/private/tmp/claude-501/sc-scratch/fix/f30_work')
from lc_lib import *
seed = int(sys.argv[1]); ns = [int(x) for x in sys.argv[2].split(',')]
a = adapter(2); p = a._pools(); pool = p["src"] + p["val"]; T = test_sets(a)
perm = [pool[i] for i in np.random.default_rng(100 + seed).permutation(len(pool))]
out = f"{WORK}/out/t2_seed{seed}.jsonl"
for n in ns:
    t = time.time(); m = fit(perm[:n], seed=seed); row = dict(n=n, seed=seed, fit_s=round(m.fit_s, 1), plant_size=m.plant_size)
    for k, ids in T.items():
        P = probs(m, ids, stride=4); row[k] = evaluate(a, ids, P, m.plant_size)
    row["total_s"] = round(time.time() - t, 1)
    open(out, "a").write(json.dumps(row) + "\n"); print(json.dumps(row), flush=True)
