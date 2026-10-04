"""Score ByT5 outputs on src/val units with the repo's metric. usage: eval_sv.py preds.jsonl [preds2.jsonl ...]"""
import json, sys
from collections import defaultdict
from scilib import ocrfix as O

EV = json.load(open("/private/tmp/claude-501/sc-scratch/f43/eval_units.json"))


def assemble(path, hunk=None, maxrate=None, join=False):
    by = defaultdict(list)
    for l in open(path):
        r = json.loads(l)
        by[r["uid"]].append(r)
    out = {}
    for uid, rs in by.items():
        rs.sort(key=lambda r: r["k"]); parts = []
        for r in rs:
            tail = r["ocr"][len(r["ocr"].rstrip()):]
            pred = (r["pred"] or "").strip()
            if not pred:
                parts.append(r["ocr"]); continue
            body = r["ocr"].rstrip()
            if maxrate is not None and O.edit_rate(body, pred) > maxrate:
                parts.append(r["ocr"]); continue
            if hunk is not None:
                pred = O.filter_hunks(body, pred, hunk)[0]
            parts.append(pred + tail)
        t = "".join(parts)
        out[uid] = O.join_line_hyphens(t) if join else t
    return out


def pooled(texts, split):
    ids = [u for u, v in EV.items() if v["split"] == split]
    r = O.score([EV[u]["ts"] for u in ids], [EV[u]["gt"] for u in ids], [texts.get(u, EV[u]["ocr"]) for u in ids])
    return r["weighted_cmer_micro"], r["per_test_set"]


if __name__ == "__main__":
    ref = {u: v["ocr"] for u, v in EV.items()}
    for sp in ("src", "val"):
        s, pt = pooled(ref, sp); print(f"copy-OCR {sp}: {s:.4f}", {k: round(v, 4) for k, v in pt.items()})
    for p in sys.argv[1:]:
        for cfg in [dict(), dict(hunk=4), dict(hunk=3), dict(hunk=2), dict(hunk=6), dict(maxrate=0.15), dict(hunk=4, maxrate=0.15), dict(hunk=4, join=True)]:
            t = assemble(p, **cfg)
            res = [pooled(t, sp) for sp in ("src", "val")]
            print(p.split("/")[-1], cfg, "src %.4f val %.4f" % (res[0][0], res[1][0]), {k: round(v, 4) for k, v in res[0][1].items()})
