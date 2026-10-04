import sys, pickle, json, numpy as np
sys.path.insert(0, ".")
from scienceclaw.bench.tasks.for33_buildingsbench import Adapter
ad = Adapter()
runs = {"id": ("final_a0_main_id", 0xeb786f33, "FoR33-id-seb786f33-e0{}", 2), "ood": ("final_a0_main_ood", 0x8f3fc651, "FoR33-ood-s8f3fc651-e0{}", 2),
        "val": ("final_a0_v_val", 0xa6d10e36, "FoR33-val-sa6d10e36-e0{}", 1)}
def avgpers(ctx):  # official AveragePersistence: mean of same hour over the 7 context days
    return ctx.reshape(len(ctx), 7, 24).mean(1)
out = {}
for split, (rd, seed, pat, ne) in runs.items():
    eps = ad.build_episodes(split, ne, seed)
    for ep in eps:
        tl = {t.name: t for t in ep.tools}
        ev = tl["load_eval_inputs"].fn({}, {})
        ctx = np.asarray(ev["context"], float)
        y = pickle.load(open(f"runs/{rd}/{ep.id}/y.pkl", "rb"))
        y = np.asarray(y, float)
        res = {}
        for nm, pred in (("agent", y), ("prevday", ctx[:, 144:168]), ("avg7", avgpers(ctx)), ("prevweek", ctx[:, :24])):
            r = ep.evaluate(pred, None)
            d = r.details
            res[nm] = (round(r.primary, 2), {k: round(v, 1) for k, v in r.details.get("per_building_nrmse_pct", {}).items()} if nm == "agent" else None,
                       {k: round(v, 1) for k, v in r.metrics.items() if "median" in k})
        print(ep.id, "n_items", ep.n_items, "n_buildings", len(set(ev["building_id"])))
        for nm, v in res.items(): print("  ", nm, v[0], v[2])
        print("   agent per-building:", res["agent"][1], {b: c for b, c in zip(ev["building_id"], ev["category"])} if False else "")
        out[ep.id] = {nm: v[0] for nm, v in res.items()}
print(json.dumps(out))
