"""Development-time check of scilib.hippo_unet on the visible data only: fit on the 28 visible training volumes, score the 8 dev volumes.
python scripts/f32/dev_unet.py dev_data.npz '{"iters": 300, "n_models": 1}'   (run on the GPU host; no id/ood volume is read)"""
import json, sys, time
import numpy as np
sys.path.insert(0, ".")
from scilib import hippo as hp, hippo_unet as hu

d = np.load(sys.argv[1])
tr = [d[f"train_img_{i:02d}"] for i in range(28)]; tl = [d[f"train_lab_{i:02d}"] for i in range(28)]
dv = [d[f"dev_img_{i:02d}"] for i in range(8)]; dl = [d[f"dev_lab_{i:02d}"] for i in range(8)]
kw = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
tta = kw.pop("tta", True)
t0 = time.time()
m = hu.fit_unet(tr, tl, **kw)
t1 = time.time()
pr = hu.predict_unet(m, dv, tta=tta)
r = hp.mean_dsc(pr, dl)
pc = [hp.case_dsc(p, g) for p, g in zip(pr, dl)]
print(json.dumps({"cfg": kw, "tta": tta, "dev_mean_dsc": round(float(r), 4), "ant": round(float(np.mean([c[0] for c in pc])), 4),
                  "post": round(float(np.mean([c[1] for c in pc])), 4), "fit_s": round(t1 - t0, 1), "pred_s": round(time.time() - t1, 1)}), flush=True)
