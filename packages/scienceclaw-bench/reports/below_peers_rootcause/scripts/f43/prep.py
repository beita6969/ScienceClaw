"""FoR43: src / val / id / ood OCR-only unit files for the GPU host (no gold), and eval_units.json (with gold, local only).

Reads pool.pkl written by dump_pools.py. Writes units_ocr.json (src+val), units_ocr_id.json, units_ocr_ood.json (uid -> {lang, ocr})
and eval_units.json (src+val with gold, used by eval_sv.py)."""
import json, pickle
from pathlib import Path

OUT = Path("/private/tmp/claude-501/sc-scratch/f43")
P = pickle.load(open(OUT / "pool.pkl", "rb"))
U = P["units"]


def units(splits):
    return {uid: {"split": sp, "ts": ts, "lang": U[uid]["language"], "ocr": U[uid]["ocr"], "gt": U[uid]["gt"]}
            for sp in splits for ts, uids in P["pools_eval"][sp].items() for uid in uids}


ev = units(("src", "val"))
json.dump(ev, open(OUT / "eval_units.json", "w"), ensure_ascii=False)
json.dump({u: {"lang": v["lang"], "ocr": v["ocr"]} for u, v in ev.items()}, open(OUT / "units_ocr.json", "w"), ensure_ascii=False)
for sp in ("id", "ood"):
    d = units((sp,))
    json.dump({u: {"lang": v["lang"], "ocr": v["ocr"]} for u, v in d.items()}, open(OUT / f"units_ocr_{sp}.json", "w"), ensure_ascii=False)
    print(sp, len(d))
print("src+val", len(ev))
