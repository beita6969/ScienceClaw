"""Same one-shot pool prediction as pool_fit_predict.py but with the previous toolkit scilib.hippo (patch label fusion + LightGBM), for a paired
comparison on exactly the same volumes. Output format is identical so pool_score.py applies. Mac side, 2 threads."""
import sys, time
import numpy as np
sys.path.insert(0, ".")
from scilib import hippo as hp

d = np.load(sys.argv[1]); p = np.load(sys.argv[2])
tr = [d[f"train_img_{i:02d}"] for i in range(28)]; tl = [d[f"train_lab_{i:02d}"] for i in range(28)]
ids = [str(x) for x in d["train_ids"]]
n = len(p["ids"])
ev = [p[f"img_{i:03d}"] for i in range(n)]
t0 = time.time()
preds = hp.fit_predict(tr, tl, ev, train_ids=ids, seed=0)
np.savez_compressed(sys.argv[3], pools=p["pools"], ids=p["ids"], **{f"pred_{i:03d}": np.asarray(a, np.uint8) for i, a in enumerate(preds)})
print("fit+predict s", round(time.time() - t0, 1), "n", n, flush=True)
