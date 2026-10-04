"""Hunk-level agreement of several correction candidates. usage: eval_vote.py <jsonl_a> <jsonl_b> [<jsonl_c> ...]
A hunk (word-level change of a candidate relative to the OCR text) is taken when at least `need` candidates propose the same
normalised replacement for the same OCR span and it needs at most k normalised character edits."""
import difflib, re, sys
from collections import Counter
from scilib import ocrfix as O
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from eval_sv import assemble, pooled, EV


def hunks(ocr, cand):
    a = re.findall(r"\s+|\S+", ocr); b = re.findall(r"\s+|\S+", cand)
    out = {}
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag != "equal":
            out[(i1, i2)] = "".join(b[j1:j2])
    return a, out


def vote(ocr, cands, k, need):
    a, hs = hunks(ocr, cands[0])
    others = [hunks(ocr, c)[1] for c in cands[1:]]
    taken = {}
    for span, new in hs.items():
        old = "".join(a[span[0]:span[1]])
        e = O._norm_edits(old, new)
        if e == 0 and O.norm(old) == O.norm(new):
            taken[span] = new; continue
        if e > k:
            continue
        agree = 1 + sum(1 for h in others if span in h and O.norm(h[span]) == O.norm(new))
        if agree >= need:
            taken[span] = new
    out, i = [], 0
    for (i1, i2) in sorted(taken):
        out.append("".join(a[i:i1])); out.append(taken[(i1, i2)]); i = i2
    out.append("".join(a[i:]))
    return "".join(out)


if __name__ == "__main__":
    raws = [assemble(p) for p in sys.argv[1:]]
    for k in (2, 3, 4, 6):
        for need in range(1, len(raws) + 1):
            t = {u: vote(EV[u]["ocr"], [r[u] for r in raws], k, need) for u in raws[0]}
            res = [pooled(t, sp) for sp in ("src", "val")]
            print(f"k={k} need={need}/{len(raws)} src {res[0][0]:.4f} val {res[1][0]:.4f}", {x: round(v, 4) for x, v in res[0][1].items()}, flush=True)
