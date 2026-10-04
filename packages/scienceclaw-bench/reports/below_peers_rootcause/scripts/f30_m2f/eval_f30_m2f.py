"""FoR30: PRBonn Mask2Former pair (phenoseg_m2f, defaults 0.8 / 0.8, nothing fitted) scored with the adapter's own evaluate() per episode and pooled.
id (= official val) is the clean pool; src / val (official train) and ood (06-05, train + val) are listed only to show the train-partition contamination."""
import sys, time, json
import numpy as np
from scienceclaw.config import load_config
from scienceclaw.bench.splits import SplitPlan, load_adapters
from scilib import phenoseg_m2f as m2f

cfg = load_config("/leonardo_scratch/fast/AIFAC_F02_774/rqian000/scienceclaw/configs/toolon_leo.yaml")
cfg.bench.disciplines = ["FoR30"]; cfg.bench.n_id = 4; cfg.bench.n_ood = 4; cfg.bench.rounds = 4
ad = load_adapters(cfg.bench); plan = SplitPlan.build(cfg.bench, ad)
splits = sys.argv[1].split(",") if len(sys.argv) > 1 else ["id"]
adapter = ad["FoR30"]
summary = {}
for sp in splits:
    eps = plan.episodes[sp]["FoR30"]; payloads = []; rows = []
    for ep in eps:
        t = time.time()
        imgs = ep.tool("load_eval_inputs").fn({}, {})["images"]
        y = m2f.predict_panoptic(imgs)
        res = ep.evaluate(y, None)
        payloads.append(res.details["pooled_payload"])
        rows.append((ep.id, round(float(res.primary), 2), round(float(res.details.get("reference", float("nan"))), 2), {k: round(v, 1) for k, v in res.metrics.items() if k in ("iou_weed", "iou_crop", "pq_crop", "pq_leaf", "pq_weed_plants")}, f"{time.time() - t:.0f}s"))
        print(sp, rows[-1], flush=True)
    pooled = adapter.pooled_metric(payloads)
    summary[sp] = {"episodes": [r[1] for r in rows], "pooled_pq_plus": pooled}
    print(f"== {sp}: n_images={sum(len(p['items']) for p in payloads)} episode PQ+ {[r[1] for r in rows]} pooled PQ+ {pooled}", flush=True)
json.dump(summary, open("/leonardo_scratch/large/userexternal/rqian000/sc-runs/f30_m2f_eval.json", "w"))
