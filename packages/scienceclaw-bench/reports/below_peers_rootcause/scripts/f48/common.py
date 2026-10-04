import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from scienceclaw.bench.tasks import for48_contractnli as A      # noqa: E402
from scilib import contracts as C                               # noqa: E402

ROOT = Path(os.environ.get("SCIENCECLAW_DATA_ROOT", "/Users/admin/Datasets/ScienceClaw-rebuild-20260928/datasets"))
SCRATCH = Path("/private/tmp/claude-501/sc-scratch/f48")


def load():
    return A._load(ROOT, A.PARTITION_SEED)


def public(doc):
    return {"doc_id": doc.doc_id, "span_texts": doc.span_texts()}


def annotations(data, ids):
    return [{"doc_id": d, "hypothesis_key": k, "label": a["choice"], "evidence_spans": list(a["spans"])}
            for d in ids for k, a in sorted(data.docs[d].annotations.items())]


def hyp_table(data):
    return {k: {"hypothesis": v["hypothesis"], "short_description": v["short_description"]} for k, v in sorted(data.hypotheses.items())}


def pairs_of(data, ids):
    return [(d, k) for d in ids for k, a in sorted(data.docs[d].annotations.items()) if a["choice"] in A.LABELS and a["spans"]]


def gold(data, d, k):
    doc = data.docs[d]
    return np.isin(np.arange(len(doc.spans)), doc.annotations[k]["spans"]).astype(int)


def paired_boot(a, b, groups, n=4000, seed=0):
    """mean(b)-mean(a) with a cluster bootstrap over ``groups``."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    g = np.asarray(groups)
    u = np.unique(g)
    idx = {x: np.where(g == x)[0] for x in u}
    rng = np.random.default_rng(seed)
    diffs = np.empty(n)
    for i in range(n):
        pick = rng.choice(u, len(u))
        sel = np.concatenate([idx[x] for x in pick])
        diffs[i] = b[sel].mean() - a[sel].mean()
    return float(b.mean() - a.mean()), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))
