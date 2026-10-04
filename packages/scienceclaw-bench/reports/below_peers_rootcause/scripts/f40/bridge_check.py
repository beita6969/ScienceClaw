"""Two harness id episodes (one 10 s machine, one 12 s machine) through the real remote bridge: AST + CLAP embeddings of the 24 support and
16 probe clips with scilib.audioenc, the pipeline scores, and a comparison with the rows of the precomputed table (same models, same clips).
usage: python bridge_check.py <shipped_emb.npz> <machine> [<machine> ...]   (needs SCIENCECLAW_REMOTE_SPOOL and a running broker)"""
import sys
import time

import numpy as np

import common as C
from scilib import anomsound as A
from scilib import audioenc
from scienceclaw.bench.tasks import for40_dcase as D

assert audioenc.available() and not audioenc._local_ok("ast_audioset"), "bridge not enabled (or weights are local)"
Z = np.load(sys.argv[1])
idx = {p: i for i, p in enumerate(Z["ids"])}
pid = C.public_ids()
ad = D.DCASE2024Task2Adapter(str(C.ROOT.parent))
design = D.load_design(C.ROOT)
plan = dict(ad._plan("id", 16, 20260928))


def stack(cs):
    ws = [D.read_clip(c) for c in cs]
    ln = np.array([w.size for w in ws])
    W = np.zeros((len(ws), int(ln.max())), np.float32)
    for r, w in enumerate(ws):
        W[r, :w.size] = w
    return W, ln


for machine in sys.argv[2:]:
    cl = [design.cohorts["id"][machine].clips[i] for i in plan[machine]]
    sup = design.support[machine][:ad.max_train_clips]
    (Ws, ls), (We, le) = stack(sup), stack(cl)
    parts, parts_tab, maxdiff = [], [], 0.0
    for model, kind, key in (("ast_audioset", "pooled", "ast_audioset__pooled"), ("clap_htsat", "pooled", "clap_htsat__pooled")):
        t0 = time.time()
        Es = audioenc.embed(Ws, ls, 16000.0, model, kind)
        Ee = audioenc.embed(We, le, 16000.0, model, kind)
        dt = time.time() - t0
        Ts = np.stack([Z[key][idx[pid[c.official_id]]] for c in sup])
        Te = np.stack([Z[key][idx[pid[c.official_id]]] for c in cl])
        maxdiff = max(maxdiff, float(np.abs(Es - Ts).max()), float(np.abs(Ee - Te).max()))
        parts.append(A.rank_average(A.embedding_scores(Es, Ee), ("nn2_pool",)))
        parts_tab.append(A.rank_average(A.embedding_scores(Ts, Te), ("nn2_pool",)))
        print(f"{machine} {model}: {len(sup)}+{len(cl)} clips in {dt:.0f} s", flush=True)
    s_b, s_t = np.mean(parts, 0), np.mean(parts_tab, 0)
    y, d = np.array([c.label for c in cl]), np.array([c.domain for c in cl])
    ob, ot = D.official_scores(y, d, s_b), D.official_scores(y, d, s_t)
    print(f"{machine}: max |embedding diff| {maxdiff:.3g}; max |score diff| {np.abs(s_b - s_t).max():.3g}; "
          f"official bridge {ob['auc_source']:.3f}/{ob['auc_target']:.3f}/{ob['pauc']:.3f} table {ot['auc_source']:.3f}/{ot['auc_target']:.3f}/{ot['pauc']:.3f}", flush=True)
