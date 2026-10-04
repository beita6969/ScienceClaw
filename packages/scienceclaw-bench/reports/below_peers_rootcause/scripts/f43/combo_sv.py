"""Exploratory (src/val only): stored LLM-pipeline predictions (y.pkl of earlier agent episodes) vs OCRonos hunks on the same units.
Matches each stored text to its pool unit by OCR text similarity, checks the episode score against eval.json, then scores
LLM only / OCRonos hunk<=k only / agreement / union on the matched src+val units."""
import difflib, glob, json, pickle, re, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from scilib import ocrfix as O
from eval_sv import assemble, EV
from eval_vote import hunks

P = pickle.load(open("/private/tmp/claude-501/sc-scratch/f43/pool.pkl", "rb"))
U = P["units"]
OC = assemble("/private/tmp/claude-501/sc-scratch/f43/ocronos_c700.jsonl")
pool_uids = [u for u, v in EV.items()]


def match(y):
    best = max(pool_uids, key=lambda u: difflib.SequenceMatcher(None, y[:150], EV[u]["ocr"][:150], autojunk=False).quick_ratio())
    return best, difflib.SequenceMatcher(None, y[:300], EV[best]["ocr"][:300], autojunk=False).ratio()


def combine(ocr, llm, oc, k, mode):
    a, hl = hunks(ocr, llm)
    _, ho = hunks(ocr, oc)
    chosen = {}
    for span, new in hl.items():
        if mode in ("llm", "union"):
            chosen[span] = new
        elif mode == "agree":
            old = "".join(a[span[0]:span[1]])
            if span in ho and O.norm(ho[span]) == O.norm(new) or (O.norm(old) == O.norm(new)):
                chosen[span] = new
    if mode == "union":
        for span, new in ho.items():
            old = "".join(a[span[0]:span[1]])
            if span in chosen or any(not (span[1] <= s[0] or span[0] >= s[1]) for s in chosen):
                continue
            if O._norm_edits(old, new) <= k or O.norm(old) == O.norm(new):
                chosen[span] = new
    out, i = [], 0
    for (i1, i2) in sorted(chosen):
        out.append("".join(a[i:i1])); out.append(chosen[(i1, i2)]); i = i2
    out.append("".join(a[i:]))
    return "".join(out)



def main():
    rows = []
    seen = set()
    for d in sorted(glob.glob("runs/*/FoR43-src-*")) + sorted(glob.glob("runs/*/FoR43-val-*")):
        try:
            y = pickle.load(open(d + "/y.pkl", "rb")); ev = json.load(open(d + "/eval.json"))
        except Exception:
            continue
        if not isinstance(y, list) or len(y) != 16 or ev.get("primary") is None:
            continue
        m = [match(s) for s in y]
        if min(r for _, r in m) < 0.8:
            print("skip (unmatched)", d); continue
        uids = [u for u, _ in m]
        if tuple(sorted(uids)) in seen:
            continue
        seen.add(tuple(sorted(uids)))
        sc = O.score([EV[u]["ts"] for u in uids], [EV[u]["gt"] for u in uids], y)["weighted_cmer_micro"]
        ref = O.score([EV[u]["ts"] for u in uids], [EV[u]["gt"] for u in uids], [EV[u]["ocr"] for u in uids])["weighted_cmer_micro"]
        print(f"{d}: stored primary {ev['primary']:.5f} recomputed {sc:.5f} copy {ref:.5f}")
        rows.append((d, uids, y))
    allu = [(u, y) for _, uids, ys in rows for u, y in zip(uids, ys)]
    print("units", len(allu))
    ts = [EV[u]["ts"] for u, _ in allu]; gt = [EV[u]["gt"] for u, _ in allu]
    sc = lambda hyp: O.score(ts, gt, hyp)["weighted_cmer_micro"]
    print("copy-OCR", round(sc([EV[u]["ocr"] for u, _ in allu]), 5))
    print("LLM pipeline (stored)", round(sc([y for _, y in allu]), 5))
    for k in (2, 3, 4):
        oc = [O.filter_hunks(EV[u]["ocr"], OC[u], k)[0] for u, _ in allu]
        print(f"OCRonos hunk<={k}", round(sc(oc), 5))
    for k in (2, 3, 4):
        print(f"union LLM + OCRonos<={k}", round(sc([combine(EV[u]["ocr"], y, OC[u], k, "union") for u, y in allu]), 5))
    print("agreement LLM&OCRonos(raw)", round(sc([combine(EV[u]["ocr"], y, OC[u], 0, "agree") for u, y in allu]), 5))



if __name__ == "__main__":
    main()
