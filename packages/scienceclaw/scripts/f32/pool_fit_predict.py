"""One-shot pool prediction (run on the GPU host): default scilib.hippo_unet.fit_predict on the 28 visible training volumes,
predictions for every volume in pool_images.npz (id / ood / val). Scoring happens on the Mac (scripts/f32/pool_score.py); nothing here is tuned.
python scripts/f32/pool_fit_predict.py dev_data.npz pool_images.npz out.npz"""
import sys, time
import numpy as np
sys.path.insert(0, ".")
from scilib import hippo_unet as hu

d = np.load(sys.argv[1]); p = np.load(sys.argv[2])
tr = [d[f"train_img_{i:02d}"] for i in range(28)]; tl = [d[f"train_lab_{i:02d}"] for i in range(28)]
n = len(p["ids"])
ev = [p[f"img_{i:03d}"] for i in range(n)]
t0 = time.time()
preds = hu.fit_predict(tr, tl, ev, seed=0)
out = {f"pred_{i:03d}": np.asarray(a, np.uint8) for i, a in enumerate(preds)}
np.savez_compressed(sys.argv[3], pools=p["pools"], ids=p["ids"], **out)
print("fit+predict s", round(time.time() - t0, 1), "n", n, flush=True)
