"""The pre-declared embedding pipeline, computed with the shipped code (scilib.audioenc embeddings, scilib.anomsound.embedding_scores).

Pipeline (fixed before any id clip was scored): for each of the two models (AST pooled, CLAP pooled) the 'nn2_pool' score of the probe
clips against the machine's 24 normal support clips and the other probe clips of the same machine, then the mean over the two models of
rank / n. No log-mel member.
usage: python shipped_eval.py dev|id <shipped_emb.npz> <base_scores.pkl>
  dev: src and val cohorts (30 probe clips per machine).
  id : the id cohorts (20 probe clips per machine), evaluated once; secondary view = the harness episodes (16 of the 20 clips, one per machine).
"""
import pickle
import sys

import numpy as np
from scipy.stats import hmean

import common as C
from scilib import anomsound as A
from scienceclaw.bench.tasks import for40_dcase as D

mode, emb_path, base_path = sys.argv[1:4]
Z = np.load(emb_path)
idx = {p: i for i, p in enumerate(Z["ids"])}
pid = C.public_ids()
MODELS = ("ast_audioset__pooled", "clap_htsat__pooled")
base = pickle.load(open(base_path, "rb"))
rng = np.random.default_rng(0)


def feats(clips, k):
    return np.stack([Z[k][idx[pid[c.official_id]]] for c in clips])


def new_scores(probe, sup):
    return np.mean([A.rank_average(A.embedding_scores(feats(sup, k), feats(probe, k)), ("nn2_pool",)) for k in MODELS], axis=0)


def per_metric(per_machine):
    out = {}
    for m, (y, d, s) in per_machine.items():
        o = D.official_scores(np.asarray(y), np.asarray(d), np.asarray(s))
        out[m] = (o["auc_source"], o["auc_target"], o["pauc"])
    return out


def pooled_of(pm):
    return float(hmean(np.maximum([v for t in per_metric(pm).values() for v in t], D.EPS)))


def pack(probe_by_m, sc):
    return {m: (np.array([c.label for c in probe_by_m[m]]), np.array([c.domain for c in probe_by_m[m]]), sc[m]) for m in sc}


def boot(pa, pb, n=1000):
    """paired stratified bootstrap (clips resampled within each machine's domain x label stratum) of pooled(a) - pooled(b)"""
    d = []
    for _ in range(n):
        A_, B_ = {}, {}
        for m in pa:
            y, dom, a = pa[m]
            _, _, b = pb[m]
            i = np.concatenate([rng.choice(np.where((dom == dd) & (y == yy))[0], size=int(((dom == dd) & (y == yy)).sum()))
                                for dd in ("source", "target") for yy in (0, 1)])
            A_[m], B_[m] = (y[i], dom[i], a[i]), (y[i], dom[i], b[i])
        d.append(pooled_of(A_) - pooled_of(B_))
    return np.percentile(d, [2.5, 97.5])


def report(tag, probe_by_m, new, cur, bootstrap=True):
    pn, pc = pack(probe_by_m, new), pack(probe_by_m, cur)
    vn, vc = pooled_of(pn), pooled_of(pc)
    line = f"{tag}: new {vn:.4f}  current ensemble {vc:.4f}  diff {vn - vc:+.4f}"
    if bootstrap:
        lo, hi = boot(pn, pc)
        line += f"  95% CI ({lo:+.4f}, {hi:+.4f})"
    print(line, flush=True)
    for m in pn:
        a, b = per_metric(pn)[m], per_metric(pc)[m]
        print(f"   {m:9s} new AUCs/pAUC {a[0]:.3f} {a[1]:.3f} {a[2]:.3f} | current {b[0]:.3f} {b[1]:.3f} {b[2]:.3f}")
    return vn, vc


if mode == "all":                                   # descriptive: mean over the src, val and id cohort scores (no new scoring)
    co = C.cohorts(("src", "val", "id"))
    P = {}
    for s_ in co:
        probe = {m: co[s_][m][0] for m in co[s_]}
        new = {m: new_scores(co[s_][m][0], co[s_][m][1]) for m in co[s_]}
        cur = {m: base[(s_, m)]["ens"] for m in co[s_]}
        P[s_] = (pack(probe, new), pack(probe, cur))
    vn = [pooled_of(P[s_][0]) for s_ in P]
    vc = [pooled_of(P[s_][1]) for s_ in P]
    d = []
    for _ in range(1000):
        x = []
        for s_ in P:
            A_, B_ = {}, {}
            for m in P[s_][0]:
                y, dom, a = P[s_][0][m]
                _, _, b = P[s_][1][m]
                i = np.concatenate([rng.choice(np.where((dom == dd) & (y == yy))[0], size=int(((dom == dd) & (y == yy)).sum()))
                                    for dd in ("source", "target") for yy in (0, 1)])
                A_[m], B_[m] = (y[i], dom[i], a[i]), (y[i], dom[i], b[i])
            x.append(pooled_of(A_) - pooled_of(B_))
        d.append(np.mean(x))
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"cohorts {list(P)}: new {np.round(vn, 4)} mean {np.mean(vn):.4f} | current {np.round(vc, 4)} mean {np.mean(vc):.4f} | "
          f"diff {np.mean(vn) - np.mean(vc):+.4f} 95% CI ({lo:+.4f}, {hi:+.4f}); cohorts better {int(np.sum(np.array(vn) > np.array(vc)))}/3")
elif mode == "dev":
    co = C.cohorts(("src", "val"))
    res = []
    for s in ("src", "val"):
        probe = {m: co[s][m][0] for m in co[s]}
        new = {m: new_scores(co[s][m][0], co[s][m][1]) for m in co[s]}
        cur = {m: base[(s, m)]["ens"] for m in co[s]}
        res.append(report(s, probe, new, cur))
    print(f"dev mean: new {np.mean([r[0] for r in res]):.4f}  current {np.mean([r[1] for r in res]):.4f}")
else:
    co = C.cohorts(("id",))["id"]
    probe = {m: co[m][0] for m in co}
    new = {m: new_scores(co[m][0], co[m][1]) for m in co}
    cur = {m: base[("id", m)]["ens"] for m in co}
    print("=== id cohorts, 20 probe clips per machine (primary)")
    report("id", probe, new, cur)
    # secondary: the harness episodes (16 of the 20 clips, one per machine); the log-mel ensemble is recomputed on the same 16 clips
    ad = D.DCASE2024Task2Adapter(str(C.ROOT.parent))
    design = D.load_design(C.ROOT)
    plan = ad._plan("id", 16, 20260928)
    print(f"=== harness episodes: {len(plan)} (one per machine: {sorted(m for m, _ in plan)}), max_train_clips {ad.max_train_clips}")

    def mels(cs):
        W = [D.read_clip(c).astype(np.float32) for c in cs]
        L = np.array([len(w) for w in W])
        M = np.zeros((len(W), L.max()), np.float32)
        for i, w in enumerate(W):
            M[i, :len(w)] = w
        return A.log_mel_list(M, L, 16000.0)

    probe16, new16, cur16 = {}, {}, {}
    for m, items in plan:
        cl = [design.cohorts["id"][m].clips[i] for i in items]
        sup = design.support[m][:ad.max_train_clips]
        probe16[m] = cl
        new16[m] = new_scores(cl, sup)
        cur16[m] = A.rank_average(A.component_scores(mels(sup), mels(cl)), None)
    report("id episodes", probe16, new16, cur16)
