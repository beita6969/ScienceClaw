"""FoR43 id / ood, each pre-declared pipeline evaluated once (no tuning on these pools).
P1 = OCRonos (PleIAs/OCRonos, greedy, ~700-char pieces) + ocrfix.filter_hunks(max_hunk_edits=3) per piece, on the whole id / ood pool.
P2 = hunks of the stored LLM-pipeline output (earlier agent episodes, runs/final_a0_main_{id,ood}) united with OCRonos hunks of at most 2
     normalised edits where they do not overlap, on the 4 stored episodes (64 units)."""
import difflib, glob, json, pickle, random, sys
from collections import defaultdict
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from scilib import ocrfix as O
from eval_sv import assemble
from combo_sv import combine

P = pickle.load(open("/private/tmp/claude-501/sc-scratch/f43/pool.pkl", "rb"))
U = P["units"]
D = "/private/tmp/claude-501/sc-scratch/f43/"


def wcm(ts, cnt):
    per = defaultdict(lambda: [0, 0, 0, 0])
    for t, c in zip(ts, cnt):
        for i in range(4):
            per[t][i] += c[i]
    sc = {t: (c[1] + c[2] + c[3]) / max(1, c[0] + c[1] + c[2] + c[3]) for t, c in per.items()}
    w = {t: O.official_weight(t) for t in sc}
    return sum(w[t] * sc[t] for t in sc) / sum(w.values()), sc


def boot(ts, ca, cb, n=2000, seed=0):
    rng = random.Random(seed)
    idx = defaultdict(list)
    for i, t in enumerate(ts):
        idx[t].append(i)
    out = []
    for _ in range(n):
        pick = [i for t, l in idx.items() for i in (rng.choice(l) for _ in l)]
        a = wcm([ts[i] for i in pick], [ca[i] for i in pick])[0]
        b = wcm([ts[i] for i in pick], [cb[i] for i in pick])[0]
        out.append(b / a - 1)
    out.sort()
    return out[int(0.025 * n)], out[int(0.975 * n)]


for sp, run in (("id", "final_a0_main_id"), ("ood", "final_a0_main_ood")):
    uids = [u for ts in P["pools_eval"][sp].values() for u in ts]
    ts = [U[u]["test_set"] for u in uids]
    oc = assemble(D + f"ocronos_c700_{sp}.jsonl", hunk=3)
    cc = [O.char_counts(U[u]["gt"], U[u]["ocr"]) for u in uids]
    cp = [O.char_counts(U[u]["gt"], oc[u]) for u in uids]
    s0, p0 = wcm(ts, cc); s1, p1 = wcm(ts, cp)
    lo, hi = boot(ts, cc, cp)
    print(f"== {sp} pool n={len(uids)}: copy-OCR {s0:.5f}  P1 {s1:.5f}  rel {s1 / s0 - 1:+.1%} (95% CI {lo:+.1%}, {hi:+.1%})")
    print("   per test set copy ->P1:", {t: (round(p0[t], 4), round(p1[t], 4)) for t in p0})
    raw = assemble(D + f"ocronos_c700_{sp}.jsonl")
    cr = [O.char_counts(U[u]["gt"], raw[u]) for u in uids]
    print(f"   (unfiltered OCRonos {wcm(ts, cr)[0]:.5f})")
    # episodes with stored LLM-pipeline predictions
    eu, ey = [], []
    for e in ("00", "01"):
        d = glob.glob(f"runs/{run}/FoR43-*-{e}")[0]
        y = pickle.load(open(d + "/y.pkl", "rb"))
        stored = json.load(open(d + "/eval.json"))["primary"]
        m = []
        for s in y:
            b = max(uids, key=lambda u: difflib.SequenceMatcher(None, s[:150], U[u]["ocr"][:150], autojunk=False).quick_ratio())
            assert difflib.SequenceMatcher(None, s[:300], U[b]["ocr"][:300], autojunk=False).ratio() > 0.8
            m.append(b)
        chk = wcm([U[u]["test_set"] for u in m], [O.char_counts(U[u]["gt"], t) for u, t in zip(m, y)])[0]
        print(f"   episode {e}: stored primary {stored:.5f} recomputed {chk:.5f}")
        eu += m; ey += y
    ets = [U[u]["test_set"] for u in eu]
    c_copy = [O.char_counts(U[u]["gt"], U[u]["ocr"]) for u in eu]
    c_llm = [O.char_counts(U[u]["gt"], y) for u, y in zip(eu, ey)]
    c_p1 = [O.char_counts(U[u]["gt"], oc[u]) for u in eu]
    c_p2 = [O.char_counts(U[u]["gt"], combine(U[u]["ocr"], y, raw[u], 2, "union")) for u, y in zip(eu, ey)]
    base = wcm(ets, c_copy)[0]
    for name, c in (("LLM pipeline (stored)", c_llm), ("P1 OCRonos+hunk3", c_p1), ("P2 LLM + OCRonos<=2 union", c_p2)):
        v = wcm(ets, c)[0]
        print(f"   64-unit episodes: copy {base:.5f}  {name} {v:.5f} ({v / base - 1:+.1%})")
    lo, hi = boot(ets, c_llm, c_p2)
    print(f"   P2 vs LLM pipeline: rel {wcm(ets, c_p2)[0] / wcm(ets, c_llm)[0] - 1:+.1%} (95% CI {lo:+.1%}, {hi:+.1%})")
